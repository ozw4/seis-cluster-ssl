"""Amplitude-only crop dataset for MAE pretraining bootstrap work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from seis_ssl_cluster.data.crop_sampler import (
	expand_request_with_margin,
	required_zero_mask_margin_xyz,
	rng_for_sample,
	sample_random_local_crop,
	select_round_robin_index,
	validate_crop_fits,
)
from seis_ssl_cluster.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
	normalize_amplitude,
)
from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
from seis_ssl_cluster.data.zero_mask import (
	DEFAULT_ZERO_MASK_CONFIG,
	ZeroMaskConfig,
	compute_zero_amplitude_invalid_mask,
)
from seis_ssl_cluster.masking import (
	build_spatial_masking_plan,
	compute_token_grid_shape,
)

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest

XYZ = tuple[int, int, int]


class NopimsAmplitudePretrainDataset:
	"""Return deterministic random amplitude-only crops from NOPIMS manifests."""

	def __init__(  # noqa: D107, PLR0913
		self,
		manifests: Sequence[SurveyManifest],
		local_crop_size_xyz: Sequence[int] = (128, 128, 128),
		patch_size_xyz: Sequence[int] = (8, 8, 8),
		spatial_mask_ratio: float = 0.75,
		spatial_mask_mode: str = 'block',
		block_size_tokens_xyz: Sequence[int] = (2, 2, 2),
		seed: int = 42,
		samples_per_epoch: int | None = None,
		zero_mask: ZeroMaskConfig = DEFAULT_ZERO_MASK_CONFIG,
		min_valid_fraction: float = 0.0,
		max_resample_attempts: int = 16,
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
		if int(np.prod(self.token_grid_shape_xyz)) < 2:
			msg = (
				'token grid must contain at least two tokens to keep one '
				'masked and one visible token'
			)
			raise ValueError(msg)
		self.spatial_mask_ratio = _validate_open_fraction(
			spatial_mask_ratio,
			'spatial_mask_ratio',
		)
		if spatial_mask_mode != 'block':
			msg = f"spatial_mask_mode must be 'block'; got {spatial_mask_mode!r}"
			raise ValueError(msg)
		self.spatial_mask_mode = spatial_mask_mode
		self.block_size_tokens_xyz = _validate_xyz(
			block_size_tokens_xyz,
			'block_size_tokens_xyz',
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

		self._store = NpyMemmapVolumeStore()
		self._normalization_stats: dict[Path, SurveyNormalizationStats] = {}
		self._validate_manifests()

	@classmethod
	def from_config(
		cls,
		manifests: Sequence[SurveyManifest],
		config: Mapping[str, object],
		*,
		samples_per_epoch: int | None = None,
	) -> NopimsAmplitudePretrainDataset:
		"""Build an amplitude-only dataset from validated config sections."""
		data = _require_config_mapping(config, 'data')
		model = _require_config_mapping(config, 'model')
		masking = _require_config_mapping(config, 'masking')
		train = _require_config_mapping(config, 'train')
		resolved_samples = samples_per_epoch
		if resolved_samples is None and 'samples_per_epoch' in train:
			resolved_samples = _validate_positive_int(
				train['samples_per_epoch'],
				'train.samples_per_epoch',
			)
		return cls(
			manifests,
			local_crop_size_xyz=data.get('local_crop_size', (128, 128, 128)),
			patch_size_xyz=model.get('patch_size', (8, 8, 8)),
			spatial_mask_ratio=masking.get('spatial_mask_ratio', 0.75),
			spatial_mask_mode=masking.get('spatial_mask_mode', 'block'),
			block_size_tokens_xyz=masking.get(
				'block_size_tokens',
				(2, 2, 2),
			),
			seed=train.get('seed', 42),
			samples_per_epoch=resolved_samples,
			zero_mask=_zero_mask_from_config(config),
			min_valid_fraction=data.get('min_valid_fraction', 0.0),
			max_resample_attempts=data.get('max_resample_attempts', 16),
		)

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
		"""Return one amplitude-only crop sample."""
		index = self._normalize_index(index)
		manifest = self.manifests[
			select_round_robin_index(len(self.manifests), index)
		]
		rng = rng_for_sample(self.seed, self.epoch, index)
		last_sample: dict[str, object] | None = None
		last_valid_fraction = 0.0
		for _ in range(self.max_resample_attempts):
			local_request = sample_random_local_crop(
				manifest.amplitude.shape_xyz,
				self.local_crop_size_xyz,
				rng,
				survey_id=manifest.survey_id,
			)
			sample = self._read_sample(manifest, local_request)
			valid_fraction = float(np.mean(sample['local_valid_mask']))
			if valid_fraction >= self.min_valid_fraction:
				self._add_spatial_masks(sample, rng)
				return sample
			last_sample = sample
			last_valid_fraction = valid_fraction

		msg = (
			f'survey {manifest.survey_id!r} did not produce a crop with '
			f'min_valid_fraction={self.min_valid_fraction!r} after '
			f'{self.max_resample_attempts} attempts; last fraction was '
			f'{last_valid_fraction:.6f}'
		)
		if last_sample is None:
			raise RuntimeError(msg)
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
			amplitude_path = _resolve_manifest_path(manifest, manifest.amplitude.path)
			if not amplitude_path.is_file():
				msg = (
					f'survey {manifest.survey_id!r} amplitude file does not '
					f'exist: {amplitude_path}'
				)
				raise FileNotFoundError(msg)
			stats_path = _resolve_manifest_path(
				manifest,
				manifest.amplitude.normalization_stats_path,
			)
			if not stats_path.is_file():
				msg = (
					f'survey {manifest.survey_id!r} normalization stats file '
					f'does not exist: {stats_path}'
				)
				raise FileNotFoundError(msg)

	def _read_sample(
		self,
		manifest: SurveyManifest,
		local_request: CropRequest,
	) -> dict[str, object]:
		amplitude_path = _resolve_manifest_path(manifest, manifest.amplitude.path)
		margin_xyz = self._zero_mask_margin_xyz()
		compute_request, payload_slices = expand_request_with_margin(
			local_request,
			margin_xyz,
		)
		raw_compute, compute_valid_mask = self._store.read_crop_with_padding(
			amplitude_path,
			compute_request.start_xyz,
			compute_request.size_xyz,
		)
		raw_crop = raw_compute[payload_slices].astype(np.float32, copy=False)
		source_valid_mask = compute_valid_mask[payload_slices]

		zero_invalid = compute_zero_amplitude_invalid_mask(
			raw_compute,
			valid_mask=compute_valid_mask,
			config=self.zero_mask,
		)[payload_slices]
		local_valid_mask = np.logical_and(source_valid_mask, ~zero_invalid)

		amplitude_norm = normalize_amplitude(
			raw_crop,
			self._stats_for_manifest(manifest),
		)
		amplitude_norm = amplitude_norm.astype(np.float32, copy=True)
		amplitude_norm[~local_valid_mask] = 0.0
		x = amplitude_norm[np.newaxis, ...]
		return {
			'x': x,
			'target': x.copy(),
			'local_valid_mask': local_valid_mask.astype(bool, copy=False),
			'coords': {
				'survey_id': manifest.survey_id,
				'local_start_xyz': local_request.start_xyz,
				'local_size_xyz': local_request.size_xyz,
			},
		}

	def _add_spatial_masks(
		self,
		sample: dict[str, object],
		rng: np.random.Generator,
	) -> None:
		plan = build_spatial_masking_plan(
			local_crop_size_xyz=self.local_crop_size_xyz,
			patch_size_xyz=self.patch_size_xyz,
			spatial_mask_ratio=self.spatial_mask_ratio,
			spatial_mask_mode=self.spatial_mask_mode,
			block_size_tokens_xyz=self.block_size_tokens_xyz,
			rng=rng,
		)
		sample['spatial_mask'] = plan.spatial_mask
		sample['visible_spatial_mask'] = plan.visible_spatial_mask

	def _zero_mask_margin_xyz(self) -> XYZ:
		if not self.zero_mask.enabled:
			return (0, 0, 0)
		return required_zero_mask_margin_xyz(
			z_sample_influence_radius=self.zero_mask.z_sample_influence_radius,
			xy_trace_influence_radius=self.zero_mask.xy_trace_influence_radius,
		)

	def _stats_for_manifest(self, manifest: SurveyManifest) -> SurveyNormalizationStats:
		path = _resolve_manifest_path(
			manifest,
			manifest.amplitude.normalization_stats_path,
		)
		if path not in self._normalization_stats:
			self._normalization_stats[path] = load_normalization_stats(path)
		return self._normalization_stats[path]


AmplitudePretrainDataset = NopimsAmplitudePretrainDataset


def _zero_mask_from_config(config: Mapping[str, object]) -> ZeroMaskConfig:
	value = config.get('zero_mask')
	if value is None:
		data = config.get('data')
		if isinstance(data, Mapping):
			value = data.get('zero_mask')
	if value is None:
		return DEFAULT_ZERO_MASK_CONFIG
	if not isinstance(value, Mapping):
		msg = f'zero_mask config must be a mapping; got {value!r}'
		raise TypeError(msg)
	return ZeroMaskConfig(**dict(value))


def _resolve_manifest_path(manifest: SurveyManifest, path: Path) -> Path:
	if path.is_absolute():
		return path
	return manifest.root / path


def _require_config_mapping(
	config: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = config[key]
	if not isinstance(value, Mapping):
		msg = f'config.{key} must be a mapping'
		raise TypeError(msg)
	return value


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


def _validate_open_fraction(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 < fraction < 1.0:
		msg = f'{name} must be in (0, 1); got {fraction!r}'
		raise ValueError(msg)
	return fraction


__all__ = [
	'AmplitudePretrainDataset',
	'NopimsAmplitudePretrainDataset',
]
