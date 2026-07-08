"""Token-aligned amplitude crops with stratigraphic pseudo-target slices."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from seis_ssl_cluster.data.crop_sampler import (
	expand_request_with_margin,
	required_zero_mask_margin_xyz,
	rng_for_sample,
	sample_random_token_aligned_local_crop,
	select_round_robin_index,
	validate_crop_fits,
)
from seis_ssl_cluster.data.normalization import (
	AmplitudeAgcConfig,
	SurveyNormalizationStats,
	apply_configured_agc,
	load_normalization_stats,
	normalize_amplitude,
)
from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
from seis_ssl_cluster.data.zero_mask import (
	DEFAULT_ZERO_MASK_CONFIG,
	ZeroMaskConfig,
	compute_zero_amplitude_invalid_mask,
)
from seis_ssl_cluster.masking import compute_token_grid_shape
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetArrays,
	StratPseudoTargetInput,
	load_pseudo_target_arrays,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence
	from pathlib import Path

	from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest

XYZ = tuple[int, int, int]


class NopimsStratPseudoTargetDataset:
	"""Return token-aligned amplitude crops and HMM pseudo-target labels."""

	def __init__(  # noqa: D107, PLR0913
		self,
		manifests: Sequence[SurveyManifest],
		pseudo_target_inputs: Sequence[StratPseudoTargetInput],
		local_crop_size_xyz: Sequence[int] = (128, 128, 128),
		patch_size_xyz: Sequence[int] = (8, 8, 8),
		seed: int = 42,
		samples_per_epoch: int | None = None,
		zero_mask: ZeroMaskConfig = DEFAULT_ZERO_MASK_CONFIG,
		min_valid_fraction: float = 0.0,
		max_resample_attempts: int = 16,
		normalized_clip_abs: float | None = None,
		amplitude_agc: AmplitudeAgcConfig | Mapping[str, object] | None = None,
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

		self._store = NpyMemmapVolumeStore()
		self._normalization_stats: dict[Path, SurveyNormalizationStats] = {}
		self._pseudo_target_inputs = _pseudo_targets_by_survey(pseudo_target_inputs)
		self._pseudo_target_arrays: dict[str, StratPseudoTargetArrays] = {}
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
		"""Return one token-aligned pseudo-target supervision sample."""
		index = self._normalize_index(index)
		manifest = self.manifests[
			select_round_robin_index(len(self.manifests), index)
		]
		rng = rng_for_sample(self.seed, self.epoch, index)
		last_valid_fraction = 0.0
		for _ in range(self.max_resample_attempts):
			local_request = sample_random_token_aligned_local_crop(
				manifest.amplitude.shape_xyz,
				self.local_crop_size_xyz,
				self.patch_size_xyz,
				rng,
				survey_id=manifest.survey_id,
			)
			sample = self._read_amplitude_sample(manifest, local_request)
			last_valid_fraction = float(np.mean(sample['local_valid_mask']))
			if last_valid_fraction < self.min_valid_fraction:
				continue
			self._add_pseudo_targets(manifest.survey_id, local_request, sample)
			if bool(np.any(sample['strat_valid_mask'])):
				return sample

		msg = (
			f'survey {manifest.survey_id!r} did not produce a pseudo-target '
			f'crop with at least one valid supervised token after '
			f'{self.max_resample_attempts} attempts; last local valid '
			f'fraction was {last_valid_fraction:.6f}'
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
		manifest_ids = {manifest.survey_id for manifest in self.manifests}
		pseudo_ids = set(self._pseudo_target_inputs)
		missing_ids = sorted(manifest_ids - pseudo_ids)
		if missing_ids:
			msg = f'missing pseudo-target inputs for surveys: {missing_ids!r}'
			raise ValueError(msg)

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
			self._validate_pseudo_target_grid(manifest)

	def _validate_pseudo_target_grid(self, manifest: SurveyManifest) -> None:
		arrays = self._pseudo_targets_for_survey(manifest.survey_id)
		required_shape = tuple(
			((shape_axis - crop_axis) // patch_axis) + token_axis
			for shape_axis, crop_axis, patch_axis, token_axis in zip(
				manifest.amplitude.shape_xyz,
				self.local_crop_size_xyz,
				self.patch_size_xyz,
				self.token_grid_shape_xyz,
				strict=True,
			)
		)
		if any(
			target_axis < required_axis
			for target_axis, required_axis in zip(
				arrays.labels.shape,
				required_shape,
				strict=True,
			)
		):
			msg = (
				f'survey {manifest.survey_id!r} pseudo-target grid is too small '
				f'for token-aligned crops; got {arrays.labels.shape!r}, need at '
				f'least {required_shape!r}'
			)
			raise ValueError(msg)

	def _read_amplitude_sample(
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
			normalized_clip_abs=self.normalized_clip_abs,
		)
		amplitude_model = apply_configured_agc(
			amplitude_norm,
			local_valid_mask,
			self.amplitude_agc,
		)
		amplitude_model[~local_valid_mask] = 0.0
		return {
			'x': amplitude_model[np.newaxis, ...].astype(np.float32, copy=False),
			'local_valid_mask': local_valid_mask.astype(bool, copy=False),
			'coords': {
				'survey_id': manifest.survey_id,
				'local_start_xyz': local_request.start_xyz,
				'local_size_xyz': local_request.size_xyz,
			},
		}

	def _add_pseudo_targets(
		self,
		survey_id: str,
		local_request: CropRequest,
		sample: dict[str, object],
	) -> None:
		arrays = self._pseudo_targets_for_survey(survey_id)
		token_start = tuple(
			local_axis // patch_axis
			for local_axis, patch_axis in zip(
				local_request.start_xyz,
				self.patch_size_xyz,
				strict=True,
			)
		)
		token_stop = tuple(
			start_axis + size_axis
			for start_axis, size_axis in zip(
				token_start,
				self.token_grid_shape_xyz,
				strict=True,
			)
		)
		token_slices = tuple(
			slice(start_axis, stop_axis)
			for start_axis, stop_axis in zip(token_start, token_stop, strict=True)
		)
		labels = np.asarray(arrays.labels[token_slices], dtype=np.int64).copy()
		confidence = np.asarray(
			arrays.confidence[token_slices],
			dtype=np.float32,
		).copy()
		pseudo_valid = np.asarray(arrays.valid_tokens[token_slices], dtype=bool)
		local_valid_mask = _require_bool_array(sample, 'local_valid_mask')
		token_valid = _token_valid_from_local_valid_mask(
			local_valid_mask,
			self.patch_size_xyz,
		)
		strat_valid_mask = np.logical_and(pseudo_valid, token_valid)
		labels[~strat_valid_mask] = -1
		confidence[~strat_valid_mask] = 0.0
		sample['strat_labels'] = labels
		sample['strat_confidence'] = confidence
		sample['strat_valid_mask'] = strat_valid_mask.astype(bool, copy=False)
		coords = sample['coords']
		if not isinstance(coords, dict):
			msg = 'sample coords must be a mapping'
			raise TypeError(msg)
		coords['token_start_xyz'] = cast('XYZ', token_start)
		coords['token_size_xyz'] = self.token_grid_shape_xyz

	def _pseudo_targets_for_survey(self, survey_id: str) -> StratPseudoTargetArrays:
		if survey_id not in self._pseudo_target_arrays:
			self._pseudo_target_arrays[survey_id] = load_pseudo_target_arrays(
				self._pseudo_target_inputs[survey_id],
				mmap_mode='r',
			)
		return self._pseudo_target_arrays[survey_id]

	def _stats_for_manifest(self, manifest: SurveyManifest) -> SurveyNormalizationStats:
		path = _resolve_manifest_path(
			manifest,
			manifest.amplitude.normalization_stats_path,
		)
		if path not in self._normalization_stats:
			self._normalization_stats[path] = load_normalization_stats(path)
		return self._normalization_stats[path]

	def _zero_mask_margin_xyz(self) -> XYZ:
		if not self.zero_mask.enabled:
			return (0, 0, 0)
		return required_zero_mask_margin_xyz(
			z_sample_influence_radius=self.zero_mask.z_sample_influence_radius,
			xy_trace_influence_radius=self.zero_mask.xy_trace_influence_radius,
		)


def _token_valid_from_local_valid_mask(
	local_valid_mask: np.ndarray,
	patch_size_xyz: XYZ,
) -> np.ndarray:
	shape = _validate_xyz(local_valid_mask.shape, 'local_valid_mask.shape')
	if any(
		shape_axis % patch_axis != 0
		for shape_axis, patch_axis in zip(shape, patch_size_xyz, strict=True)
	):
		msg = (
			'local_valid_mask shape must be divisible by patch_size_xyz; '
			f'got {shape!r} and {patch_size_xyz!r}'
		)
		raise ValueError(msg)
	token_shape = tuple(
		shape_axis // patch_axis
		for shape_axis, patch_axis in zip(shape, patch_size_xyz, strict=True)
	)
	patch_view = local_valid_mask.reshape(
		token_shape[0],
		patch_size_xyz[0],
		token_shape[1],
		patch_size_xyz[1],
		token_shape[2],
		patch_size_xyz[2],
	)
	return np.all(patch_view, axis=(1, 3, 5))


def _pseudo_targets_by_survey(
	pseudo_target_inputs: Sequence[StratPseudoTargetInput],
) -> dict[str, StratPseudoTargetInput]:
	if not pseudo_target_inputs:
		msg = 'pseudo_target_inputs must contain at least one survey'
		raise ValueError(msg)
	by_survey: dict[str, StratPseudoTargetInput] = {}
	for item in pseudo_target_inputs:
		if not isinstance(item, StratPseudoTargetInput):
			msg = (
				'pseudo_target_inputs must contain StratPseudoTargetInput '
				f'items; got {type(item).__name__}'
			)
			raise TypeError(msg)
		if item.survey_id in by_survey:
			msg = f'duplicate pseudo-target input for survey {item.survey_id!r}'
			raise ValueError(msg)
		by_survey[item.survey_id] = item
	return by_survey


def _require_bool_array(sample: Mapping[str, object], key: str) -> np.ndarray:
	value = sample[key]
	if not isinstance(value, np.ndarray):
		msg = f'{key} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	if value.dtype != np.bool_:
		msg = f'{key} dtype must be bool; got {value.dtype}'
		raise TypeError(msg)
	return value


def _amplitude_agc_from_config(
	value: AmplitudeAgcConfig | Mapping[str, object] | None,
) -> AmplitudeAgcConfig:
	if isinstance(value, AmplitudeAgcConfig):
		value.validate()
		return value
	return AmplitudeAgcConfig.from_mapping(value)


def _resolve_manifest_path(manifest: SurveyManifest, path: Path) -> Path:
	if path.is_absolute():
		return path
	return manifest.root / path


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


def _validate_optional_positive_float(
	value: object,
	name: str,
) -> float | None:
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


__all__ = ['NopimsStratPseudoTargetDataset']
