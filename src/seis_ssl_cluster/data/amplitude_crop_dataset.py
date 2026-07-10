"""Generic token-aligned amplitude crops with pluggable target providers."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from seis_ssl_cluster.data.crop_sampler import (
	rng_for_sample,
	sample_random_token_aligned_local_crop,
	select_round_robin_index,
	validate_crop_fits,
)
from seis_ssl_cluster.data.normalization import (
	AmplitudeAgcConfig,
	SurveyNormalizationStats,
	load_normalization_stats,
)
from seis_ssl_cluster.data.target_providers import (
	NoTargetProvider,
	TargetProviderContext,
)
from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
from seis_ssl_cluster.data.window_preprocessing import (
	AmplitudePreprocessSettings,
	PreparedAmplitudeCrop,
	read_amplitude_crop,
	resolve_manifest_path,
)
from seis_ssl_cluster.data.zero_mask import (
	DEFAULT_ZERO_MASK_CONFIG,
	ZeroMaskConfig,
)
from seis_ssl_cluster.masking import compute_token_grid_shape

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence
	from pathlib import Path

	from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest
	from seis_ssl_cluster.data.target_providers import TargetProvider

XYZ = tuple[int, int, int]


class NopimsAmplitudeCropDataset:
	"""Return deterministic token-aligned amplitude crops with optional targets."""

	def __init__(  # noqa: D107, PLR0913
		self,
		manifests: Sequence[SurveyManifest],
		*,
		local_crop_size_xyz: Sequence[int] = (128, 128, 128),
		patch_size_xyz: Sequence[int] = (8, 8, 8),
		seed: int = 42,
		samples_per_epoch: int | None = None,
		zero_mask: ZeroMaskConfig = DEFAULT_ZERO_MASK_CONFIG,
		min_valid_fraction: float = 0.0,
		max_resample_attempts: int = 16,
		normalized_clip_abs: float | None = None,
		amplitude_agc: AmplitudeAgcConfig | Mapping[str, object] | None = None,
		target_provider: TargetProvider | None = None,
	) -> None:
		self.manifests = tuple(manifests)
		if not self.manifests:
			msg = 'manifests must contain at least one survey'
			raise ValueError(msg)
		self.local_crop_size_xyz = _validate_xyz(
			local_crop_size_xyz,
			'local_crop_size_xyz',
		)
		self.patch_size_xyz = _validate_xyz(patch_size_xyz, 'patch_size_xyz')
		self.token_grid_shape_xyz = compute_token_grid_shape(
			self.local_crop_size_xyz,
			self.patch_size_xyz,
		)
		self.seed = _validate_nonnegative_int(seed, 'seed')
		self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
		if samples_per_epoch is None:
			self.samples_per_epoch = len(self.manifests)
		else:
			self.samples_per_epoch = _validate_positive_int(
				samples_per_epoch,
				'samples_per_epoch',
			)
		if not isinstance(zero_mask, ZeroMaskConfig):
			msg = f'zero_mask must be a ZeroMaskConfig; got {zero_mask!r}'
			raise TypeError(msg)
		zero_mask.validate()
		self.zero_mask = zero_mask
		self.min_valid_fraction = _validate_fraction(
			min_valid_fraction,
			'min_valid_fraction',
		)
		self.max_resample_attempts = _validate_positive_int(
			max_resample_attempts,
			'max_resample_attempts',
		)
		self.normalized_clip_abs = _validate_optional_positive_float(
			normalized_clip_abs,
			'normalized_clip_abs',
		)
		self.amplitude_agc = _amplitude_agc_from_config(amplitude_agc)
		self.target_provider = (
			NoTargetProvider() if target_provider is None else target_provider
		)

		self._store = NpyMemmapVolumeStore()
		self._normalization_stats: dict[Path, SurveyNormalizationStats] = {}
		self._validate_manifests()

	def __len__(self) -> int:
		"""Return configured epoch length."""
		return self.samples_per_epoch

	@property
	def epoch(self) -> int:
		"""Return the current shared sampling epoch."""
		return int(self._epoch.item())

	def set_epoch(self, epoch: int) -> None:
		"""Set the sampling epoch used to seed deterministic sample draws."""
		self._epoch.fill_(_validate_nonnegative_int(epoch, 'epoch'))

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return one token-aligned amplitude crop with provider targets."""
		index = self._normalize_index(index)
		manifest = self.manifests[
			select_round_robin_index(len(self.manifests), index)
		]
		rng = rng_for_sample(self.seed, self.epoch, index)
		last_valid_fraction = 0.0
		for _ in range(self.max_resample_attempts):
			request = sample_random_token_aligned_local_crop(
				manifest.amplitude.shape_xyz,
				self.local_crop_size_xyz,
				self.patch_size_xyz,
				rng,
				survey_id=manifest.survey_id,
			)
			prepared = self._read_amplitude_crop(manifest, request)
			last_valid_fraction = float(np.mean(prepared.local_valid_mask))
			if last_valid_fraction < self.min_valid_fraction:
				continue
			sample: dict[str, object] = {
				'x': prepared.x,
				'local_valid_mask': prepared.local_valid_mask,
				'coords': {
					'survey_id': manifest.survey_id,
					'local_start_xyz': request.start_xyz,
					'local_size_xyz': request.size_xyz,
				},
			}
			token_start_xyz = tuple(
				start_axis // patch_axis
				for start_axis, patch_axis in zip(
					request.start_xyz,
					self.patch_size_xyz,
					strict=True,
				)
			)
			context = TargetProviderContext(
				manifest=manifest,
				crop_request=request,
				patch_size_xyz=self.patch_size_xyz,
				token_start_xyz=cast('XYZ', token_start_xyz),
				token_size_xyz=self.token_grid_shape_xyz,
				token_valid_mask=prepared.token_valid_mask,
			)
			self.target_provider.add_targets(sample, context)
			if self.target_provider.sample_is_acceptable(sample):
				return sample

		provider_message = self.target_provider.rejection_message(
			survey_id=manifest.survey_id,
			max_resample_attempts=self.max_resample_attempts,
			last_valid_fraction=last_valid_fraction,
		)
		msg = (
			f'{provider_message} Amplitude validity requirement was '
			f'min_valid_fraction={self.min_valid_fraction:.6f}; last local valid '
			f'fraction was {last_valid_fraction:.6f}.'
		)
		raise ValueError(msg)

	def _normalize_index(self, index: int) -> int:
		if not isinstance(index, Integral):
			msg = f'index must be an integer; got {index!r}'
			raise TypeError(msg)
		normalized = int(index)
		if normalized < 0:
			normalized += len(self)
		if normalized < 0 or normalized >= len(self):
			msg = f'index out of range: {index!r}'
			raise IndexError(msg)
		return normalized

	def _validate_manifests(self) -> None:
		for manifest in self.manifests:
			manifest.validate()
			validate_crop_fits(
				manifest.amplitude.shape_xyz,
				self.local_crop_size_xyz,
			)
			amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
			if not amplitude_path.is_file():
				msg = (
					f'survey {manifest.survey_id!r} amplitude file does not '
					f'exist: {amplitude_path}'
				)
				raise FileNotFoundError(msg)
			stats_path = resolve_manifest_path(
				manifest,
				manifest.amplitude.normalization_stats_path,
			)
			if not stats_path.is_file():
				msg = (
					f'survey {manifest.survey_id!r} normalization stats file '
					f'does not exist: {stats_path}'
				)
				raise FileNotFoundError(msg)
		self.target_provider.validate_manifests(
			self.manifests,
			local_crop_size_xyz=self.local_crop_size_xyz,
			patch_size_xyz=self.patch_size_xyz,
			token_grid_shape_xyz=self.token_grid_shape_xyz,
		)

	def _read_amplitude_crop(
		self,
		manifest: SurveyManifest,
		request: CropRequest,
	) -> PreparedAmplitudeCrop:
		return read_amplitude_crop(
			request=request,
			amplitude_path=resolve_manifest_path(manifest, manifest.amplitude.path),
			stats=self._stats_for_manifest(manifest),
			store=self._store,
			patch_size_xyz=self.patch_size_xyz,
			settings=AmplitudePreprocessSettings(
				zero_mask=self.zero_mask,
				normalized_clip_abs=self.normalized_clip_abs,
				amplitude_agc=self.amplitude_agc,
				min_token_valid_fraction=1.0,
			),
		)

	def _stats_for_manifest(self, manifest: SurveyManifest) -> SurveyNormalizationStats:
		path = resolve_manifest_path(
			manifest,
			manifest.amplitude.normalization_stats_path,
		)
		if path not in self._normalization_stats:
			self._normalization_stats[path] = load_normalization_stats(path)
		return self._normalization_stats[path]


def _amplitude_agc_from_config(
	value: AmplitudeAgcConfig | Mapping[str, object] | None,
) -> AmplitudeAgcConfig:
	if isinstance(value, AmplitudeAgcConfig):
		value.validate()
		return value
	return AmplitudeAgcConfig.from_mapping(value)


def _validate_xyz(value: Sequence[int], name: str) -> XYZ:
	if (
		isinstance(value, str)
		or len(value) != 3
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(int(axis) for axis in value)
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return cast('XYZ', xyz)


def _validate_positive_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count <= 0:
		msg = f'{name} must be positive; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_nonnegative_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count < 0:
		msg = f'{name} must be nonnegative; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_fraction(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{name} must be in [0, 1]; got {fraction!r}'
		raise ValueError(msg)
	return fraction


def _validate_optional_positive_float(value: object, name: str) -> float | None:
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number or None; got {value!r}'
		raise TypeError(msg)
	result = float(value)
	if not np.isfinite(result) or result <= 0.0:
		msg = f'{name} must be a finite positive number; got {value!r}'
		raise ValueError(msg)
	return result


__all__ = ['NopimsAmplitudeCropDataset']
