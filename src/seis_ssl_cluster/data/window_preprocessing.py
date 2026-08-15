"""Shared amplitude crop preprocessing for window-like callers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from seis_ssl_cluster.data.crop_sampler import (
	expand_request_with_margin,
	required_zero_mask_margin_xyz,
)
from seis_ssl_cluster.data.normalization import (
	AmplitudeAgcConfig,
	SurveyNormalizationStats,
	_normalize_amplitude_inplace,
	apply_configured_agc,
)
from seis_ssl_cluster.data.zero_mask import (
	ZeroMaskConfig,
	compute_zero_amplitude_invalid_mask,
)

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest
	from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore

XYZ = tuple[int, int, int]
FiniteCheckMode = Literal['strict', 'output_only', 'off']


@dataclass(frozen=True)
class AmplitudePreprocessSettings:
	"""Configuration for shared amplitude crop preprocessing."""

	zero_mask: ZeroMaskConfig
	normalized_clip_abs: float | None
	amplitude_agc: AmplitudeAgcConfig
	min_token_valid_fraction: float
	finite_check_mode: FiniteCheckMode = 'strict'


@dataclass(frozen=True)
class PreparedAmplitudeCrop:
	"""A preprocessed amplitude crop and its voxel/token validity masks."""

	request: CropRequest
	x: np.ndarray
	local_valid_mask: np.ndarray
	token_valid_mask: np.ndarray


@dataclass(frozen=True)
class AmplitudeCropCandidate:
	"""A cheaply prepared crop awaiting amplitude preprocessing."""

	request: CropRequest
	raw_crop: np.ndarray
	local_valid_mask: np.ndarray
	valid_fraction: float


def zero_mask_margin_xyz(config: ZeroMaskConfig) -> XYZ:
	"""Return the halo margin needed for zero-mask preprocessing."""
	if not isinstance(config, ZeroMaskConfig):
		msg = f'zero_mask config must be a ZeroMaskConfig; got {config!r}'
		raise TypeError(msg)
	config.validate()
	if not config.enabled:
		return (0, 0, 0)
	return required_zero_mask_margin_xyz(
		z_sample_influence_radius=config.z_sample_influence_radius,
		xy_trace_influence_radius=config.xy_trace_influence_radius,
	)


def resolve_manifest_path(manifest: SurveyManifest, path: Path) -> Path:
	"""Resolve a manifest-relative path against ``manifest.root``."""
	if path.is_absolute():
		return path
	return manifest.root / path


def reduce_valid_mask_to_tokens(
	valid_mask_xyz: np.ndarray,
	*,
	patch_size_xyz: Sequence[int],
	min_valid_fraction: float,
) -> np.ndarray:
	"""Reduce a voxel-valid mask to token validity by patch valid fraction."""
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	threshold = _validate_fraction(min_valid_fraction, 'min_valid_fraction')
	mask = np.asarray(valid_mask_xyz, dtype=bool)
	if mask.ndim != 3:
		msg = f'valid_mask_xyz must be 3D; got shape={mask.shape!r}'
		raise ValueError(msg)
	if any(
		shape_axis % patch_axis != 0
		for shape_axis, patch_axis in zip(mask.shape, patch, strict=True)
	):
		msg = (
			'valid_mask_xyz shape must be divisible by patch_size_xyz; '
			f'got {mask.shape!r} and {patch!r}'
		)
		raise ValueError(msg)
	token_shape = tuple(
		shape_axis // patch_axis
		for shape_axis, patch_axis in zip(mask.shape, patch, strict=True)
	)
	patch_view = mask.reshape(
		token_shape[0],
		patch[0],
		token_shape[1],
		patch[1],
		token_shape[2],
		patch[2],
	)
	fractions = patch_view.mean(axis=(1, 3, 5))
	return (fractions >= threshold).astype(bool, copy=False)


def read_amplitude_crop(  # noqa: PLR0913
	*,
	request: CropRequest,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	store: NpyMemmapVolumeStore,
	patch_size_xyz: Sequence[int],
	settings: AmplitudePreprocessSettings,
	valid_mask_path: Path | None = None,
) -> PreparedAmplitudeCrop:
	"""Read and preprocess one amplitude crop in `[1, X, Y, Z]` format."""
	candidate = read_amplitude_crop_candidate(
		request=request,
		amplitude_path=amplitude_path,
		store=store,
		settings=settings,
		valid_mask_path=valid_mask_path,
	)
	return finalize_amplitude_crop(
		candidate=candidate,
		stats=stats,
		patch_size_xyz=patch_size_xyz,
		settings=settings,
	)


def read_prepared_survey_amplitude_crop(
	*,
	request: CropRequest,
	normalized_amplitude: np.ndarray,
	zero_like_mask: np.ndarray,
	patch_size_xyz: Sequence[int],
	settings: AmplitudePreprocessSettings,
) -> PreparedAmplitudeCrop:
	"""Slice cached survey data and finish the window-local preprocessing."""
	_validate_settings(settings)
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	normalized = np.asarray(normalized_amplitude)
	zero_like = np.asarray(zero_like_mask, dtype=bool)
	if normalized.ndim != 3 or zero_like.shape != normalized.shape:
		msg = (
			'prepared survey amplitude and zero-like mask must be matching 3D '
			f'arrays; got {normalized.shape!r} and {zero_like.shape!r}'
		)
		raise ValueError(msg)
	margin_xyz = zero_mask_margin_xyz(settings.zero_mask)
	compute_request, payload_slices = expand_request_with_margin(request, margin_xyz)
	normalized_compute, compute_valid_mask = _slice_array_with_padding(
		normalized,
		compute_request.start_xyz,
		compute_request.size_xyz,
		pad_value=0.0,
	)
	if settings.finite_check_mode == 'strict':
		_validate_finite_valid_voxels(
			normalized_compute,
			compute_valid_mask,
			'prepared survey amplitude',
		)
	source_valid_mask = compute_valid_mask[payload_slices]
	if settings.zero_mask.enabled:
		zero_compute, _ = _slice_array_with_padding(
			zero_like,
			compute_request.start_xyz,
			compute_request.size_xyz,
			pad_value=True,
		)
		zero_proxy = np.logical_not(zero_compute).astype(np.float32)
		zero_invalid = compute_zero_amplitude_invalid_mask(
			zero_proxy,
			valid_mask=compute_valid_mask,
			config=replace(settings.zero_mask, zero_atol=0.0),
		)[payload_slices]
		local_valid_mask = np.logical_and(source_valid_mask, ~zero_invalid)
	else:
		local_valid_mask = source_valid_mask
	local_valid_mask = local_valid_mask.astype(bool, copy=False)
	amplitude_model = np.array(
		normalized_compute[payload_slices],
		dtype=np.float32,
		copy=True,
	)
	if settings.amplitude_agc.enabled:
		amplitude_model = apply_configured_agc(
			amplitude_model,
			local_valid_mask,
			settings.amplitude_agc,
		)
	if settings.finite_check_mode == 'strict' and settings.amplitude_agc.enabled:
		_validate_finite_array(amplitude_model, 'AGC amplitude')
	amplitude_model[~local_valid_mask] = 0.0
	if settings.finite_check_mode == 'output_only':
		_validate_finite_array(amplitude_model, 'preprocessed amplitude')
	token_valid_mask = reduce_valid_mask_to_tokens(
		local_valid_mask,
		patch_size_xyz=patch,
		min_valid_fraction=settings.min_token_valid_fraction,
	)
	return PreparedAmplitudeCrop(
		request=request,
		x=amplitude_model[np.newaxis, ...],
		local_valid_mask=local_valid_mask,
		token_valid_mask=token_valid_mask,
	)


def read_amplitude_crop_candidate(
	*,
	request: CropRequest,
	amplitude_path: Path,
	store: NpyMemmapVolumeStore,
	settings: AmplitudePreprocessSettings,
	valid_mask_path: Path | None = None,
) -> AmplitudeCropCandidate:
	"""Read a crop and compute validity without amplitude transformations."""
	_validate_settings(settings)
	margin_xyz = zero_mask_margin_xyz(settings.zero_mask)
	compute_request, payload_slices = expand_request_with_margin(request, margin_xyz)
	raw_compute, compute_valid_mask = store.read_crop_with_padding(
		amplitude_path,
		compute_request.start_xyz,
		compute_request.size_xyz,
	)
	if valid_mask_path is not None:
		explicit_valid_mask = store.read_source_valid_mask_crop_with_padding(
			valid_mask_path,
			compute_request.start_xyz,
			compute_request.size_xyz,
			cast('XYZ', tuple(int(axis) for axis in store.open(amplitude_path).shape)),
		)
		compute_valid_mask = np.logical_and(
			compute_valid_mask,
			explicit_valid_mask,
		)
	if settings.zero_mask.enabled:
		zero_invalid_compute = compute_zero_amplitude_invalid_mask(
			raw_compute,
			valid_mask=compute_valid_mask,
			config=settings.zero_mask,
		)
		local_valid_compute = np.logical_and(
			compute_valid_mask,
			~zero_invalid_compute,
		)
	else:
		local_valid_compute = compute_valid_mask
	if settings.finite_check_mode == 'strict':
		_validate_finite_valid_voxels(
			raw_compute,
			local_valid_compute,
			amplitude_path,
		)
	raw_crop = raw_compute[payload_slices]
	local_valid_mask = local_valid_compute[payload_slices]
	local_valid_mask = local_valid_mask.astype(bool, copy=False)
	return AmplitudeCropCandidate(
		request=request,
		raw_crop=raw_crop,
		local_valid_mask=local_valid_mask,
		valid_fraction=float(np.mean(local_valid_mask)),
	)


def finalize_amplitude_crop(
	*,
	candidate: AmplitudeCropCandidate,
	stats: SurveyNormalizationStats,
	patch_size_xyz: Sequence[int],
	settings: AmplitudePreprocessSettings,
) -> PreparedAmplitudeCrop:
	"""Apply expensive amplitude transforms to a previously read candidate."""
	if not isinstance(candidate, AmplitudeCropCandidate):
		msg = f'candidate must be an AmplitudeCropCandidate; got {candidate!r}'
		raise TypeError(msg)
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	_validate_settings(settings)
	work_buffer = np.array(candidate.raw_crop, dtype=np.float32, copy=True)
	work_buffer[~candidate.local_valid_mask] = np.float32(stats.median)
	amplitude_norm = _normalize_amplitude_inplace(
		work_buffer,
		stats,
		normalized_clip_abs=settings.normalized_clip_abs,
	)
	if settings.finite_check_mode == 'strict':
		_validate_finite_array(amplitude_norm, 'normalized amplitude')
	if settings.amplitude_agc.enabled:
		amplitude_model = apply_configured_agc(
			amplitude_norm,
			candidate.local_valid_mask,
			settings.amplitude_agc,
		)
	else:
		amplitude_model = amplitude_norm
	if settings.finite_check_mode == 'strict' and settings.amplitude_agc.enabled:
		_validate_finite_array(amplitude_model, 'AGC amplitude')
	amplitude_model[~candidate.local_valid_mask] = 0.0
	if settings.finite_check_mode == 'output_only':
		_validate_finite_array(amplitude_model, 'preprocessed amplitude')
	token_valid_mask = reduce_valid_mask_to_tokens(
		candidate.local_valid_mask,
		patch_size_xyz=patch,
		min_valid_fraction=settings.min_token_valid_fraction,
	)
	return PreparedAmplitudeCrop(
		request=candidate.request,
		x=amplitude_model[np.newaxis, ...].astype(np.float32, copy=False),
		local_valid_mask=candidate.local_valid_mask,
		token_valid_mask=token_valid_mask,
	)


def _validate_settings(settings: AmplitudePreprocessSettings) -> None:
	if not isinstance(settings, AmplitudePreprocessSettings):
		msg = (
			'settings must be an AmplitudePreprocessSettings; '
			f'got {settings!r}'
		)
		raise TypeError(msg)
	if not isinstance(settings.zero_mask, ZeroMaskConfig):
		msg = f'settings.zero_mask must be a ZeroMaskConfig; got {settings.zero_mask!r}'
		raise TypeError(msg)
	settings.zero_mask.validate()
	if not isinstance(settings.amplitude_agc, AmplitudeAgcConfig):
		msg = (
			'settings.amplitude_agc must be an AmplitudeAgcConfig; '
			f'got {settings.amplitude_agc!r}'
		)
		raise TypeError(msg)
	settings.amplitude_agc.validate()
	if settings.finite_check_mode not in {'strict', 'output_only', 'off'}:
		msg = (
			'settings.finite_check_mode must be "strict", "output_only", or '
			f'"off"; got {settings.finite_check_mode!r}'
		)
		raise ValueError(msg)
	if settings.normalized_clip_abs is not None:
		_validate_positive_finite_float(
			settings.normalized_clip_abs,
			'settings.normalized_clip_abs',
		)
	_validate_fraction(
		settings.min_token_valid_fraction,
		'settings.min_token_valid_fraction',
	)


def _validate_finite_valid_voxels(
	values: np.ndarray,
	valid_mask: np.ndarray,
	path: object,
) -> None:
	valid_values = np.asarray(values)[np.asarray(valid_mask, dtype=bool)]
	if not np.isfinite(valid_values).all():
		msg = f'amplitude crop contains non-finite source voxels: {path}'
		raise ValueError(msg)


def _validate_finite_array(values: np.ndarray, label: str) -> None:
	if not np.isfinite(values).all():
		msg = f'{label} contains non-finite values'
		raise ValueError(msg)


def _slice_array_with_padding(
	array: np.ndarray,
	start_xyz: XYZ,
	size_xyz: XYZ,
	*,
	pad_value: float | bool,
) -> tuple[np.ndarray, np.ndarray]:
	stop_xyz = tuple(
		start + size for start, size in zip(start_xyz, size_xyz, strict=True)
	)
	source_start = tuple(max(start, 0) for start in start_xyz)
	source_stop = tuple(
		min(stop, shape)
		for stop, shape in zip(stop_xyz, array.shape, strict=True)
	)
	result = np.full(size_xyz, pad_value, dtype=array.dtype)
	valid = np.zeros(size_xyz, dtype=bool)
	if not all(
		stop > start for start, stop in zip(source_start, source_stop, strict=True)
	):
		return result, valid
	dest_start = tuple(
		source - requested
		for source, requested in zip(source_start, start_xyz, strict=True)
	)
	dest_stop = tuple(
		dest + stop - start
		for dest, start, stop in zip(
			dest_start,
			source_start,
			source_stop,
			strict=True,
		)
	)
	source_slices = tuple(
		slice(start, stop)
		for start, stop in zip(source_start, source_stop, strict=True)
	)
	dest_slices = tuple(
		slice(start, stop)
		for start, stop in zip(dest_start, dest_stop, strict=True)
	)
	result[dest_slices] = array[source_slices]
	valid[dest_slices] = True
	return result, valid


def _validate_positive_xyz(value: Sequence[int], name: str) -> XYZ:
	if (
		isinstance(value, str)
		or not isinstance(value, Sequence)
		or len(value) != 3
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = cast('XYZ', tuple(int(axis) for axis in value))
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_fraction(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
		msg = f'{name} must be in [0, 1]; got {fraction!r}'
		raise ValueError(msg)
	return fraction


def _validate_positive_finite_float(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not np.isfinite(number) or number <= 0.0:
		msg = f'{name} must be a finite positive number; got {value!r}'
		raise ValueError(msg)
	return number


__all__ = [
	'AmplitudeCropCandidate',
	'AmplitudePreprocessSettings',
	'FiniteCheckMode',
	'PreparedAmplitudeCrop',
	'finalize_amplitude_crop',
	'read_amplitude_crop',
	'read_amplitude_crop_candidate',
	'read_prepared_survey_amplitude_crop',
	'reduce_valid_mask_to_tokens',
	'resolve_manifest_path',
	'zero_mask_margin_xyz',
]
