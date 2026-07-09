"""Shared amplitude crop preprocessing for window-like callers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.data.crop_sampler import (
	expand_request_with_margin,
	required_zero_mask_margin_xyz,
)
from seis_ssl_cluster.data.normalization import (
	AmplitudeAgcConfig,
	SurveyNormalizationStats,
	apply_configured_agc,
	normalize_amplitude,
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


@dataclass(frozen=True)
class AmplitudePreprocessSettings:
	"""Configuration for shared amplitude crop preprocessing."""

	zero_mask: ZeroMaskConfig
	normalized_clip_abs: float | None
	amplitude_agc: AmplitudeAgcConfig
	min_token_valid_fraction: float


@dataclass(frozen=True)
class PreparedAmplitudeCrop:
	"""A preprocessed amplitude crop and its voxel/token validity masks."""

	request: CropRequest
	x: np.ndarray
	local_valid_mask: np.ndarray
	token_valid_mask: np.ndarray


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
) -> PreparedAmplitudeCrop:
	"""Read and preprocess one amplitude crop in `[1, X, Y, Z]` format."""
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	_validate_settings(settings)
	margin_xyz = zero_mask_margin_xyz(settings.zero_mask)
	compute_request, payload_slices = expand_request_with_margin(request, margin_xyz)
	raw_compute, compute_valid_mask = store.read_crop_with_padding(
		amplitude_path,
		compute_request.start_xyz,
		compute_request.size_xyz,
	)
	_validate_finite_valid_voxels(raw_compute, compute_valid_mask, amplitude_path)
	raw_crop = raw_compute[payload_slices].astype(np.float32, copy=False)
	source_valid_mask = compute_valid_mask[payload_slices]
	zero_invalid = compute_zero_amplitude_invalid_mask(
		raw_compute,
		valid_mask=compute_valid_mask,
		config=settings.zero_mask,
	)[payload_slices]
	local_valid_mask = np.logical_and(source_valid_mask, ~zero_invalid)
	amplitude_norm = normalize_amplitude(
		raw_crop,
		stats,
		normalized_clip_abs=settings.normalized_clip_abs,
	)
	_validate_finite_array(amplitude_norm, 'normalized amplitude')
	amplitude_model = apply_configured_agc(
		amplitude_norm,
		local_valid_mask,
		settings.amplitude_agc,
	)
	_validate_finite_array(amplitude_model, 'AGC amplitude')
	amplitude_model[~local_valid_mask] = 0.0
	token_valid_mask = reduce_valid_mask_to_tokens(
		local_valid_mask,
		patch_size_xyz=patch,
		min_valid_fraction=settings.min_token_valid_fraction,
	)
	return PreparedAmplitudeCrop(
		request=request,
		x=amplitude_model[np.newaxis, ...].astype(np.float32, copy=False),
		local_valid_mask=local_valid_mask.astype(bool, copy=False),
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
	path: Path,
) -> None:
	valid_values = np.asarray(values)[np.asarray(valid_mask, dtype=bool)]
	if not np.isfinite(valid_values).all():
		msg = f'amplitude crop contains non-finite source voxels: {path}'
		raise ValueError(msg)


def _validate_finite_array(values: np.ndarray, label: str) -> None:
	if not np.isfinite(values).all():
		msg = f'{label} contains non-finite values'
		raise ValueError(msg)


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
	'AmplitudePreprocessSettings',
	'PreparedAmplitudeCrop',
	'read_amplitude_crop',
	'reduce_valid_mask_to_tokens',
	'resolve_manifest_path',
	'zero_mask_margin_xyz',
]
