"""Raw zero-amplitude invalid-mask utilities for amplitude pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class ZeroMaskConfig:
	"""Configuration for deterministic raw zero-amplitude invalid masking."""

	enabled: bool = True
	zero_atol: float = 0.0
	z_sample_influence_radius: int = 16
	xy_trace_influence_radius: int = 1

	def validate(self) -> None:
		"""Validate zero-mask settings."""
		if not isinstance(self.enabled, bool):
			msg = f'enabled must be a bool; got {self.enabled!r}'
			raise TypeError(msg)
		_validate_zero_atol(self.zero_atol)
		_validate_nonnegative_int(
			self.z_sample_influence_radius,
			'z_sample_influence_radius',
		)
		_validate_nonnegative_int(
			self.xy_trace_influence_radius,
			'xy_trace_influence_radius',
		)


DEFAULT_ZERO_MASK_CONFIG = ZeroMaskConfig()


def compute_zero_amplitude_invalid_mask(
	amplitude_xyz: np.ndarray,
	*,
	valid_mask: np.ndarray | None = None,
	config: ZeroMaskConfig = DEFAULT_ZERO_MASK_CONFIG,
) -> np.ndarray:
	"""Return a boolean [x, y, z] invalid mask for raw zero regions."""
	config.validate()
	amplitude, valid = _as_amplitude_and_valid_mask(amplitude_xyz, valid_mask)
	if not config.enabled:
		return np.zeros(amplitude.shape, dtype=bool)

	amplitude = _replace_nonfinite_with_zero(amplitude)
	np.abs(amplitude, out=amplitude)
	zero_like = np.less_equal(amplitude, np.float32(config.zero_atol))
	zero_z_samples = _detect_all_zero_z_samples(zero_like, valid)
	zero_traces = _detect_all_zero_traces(zero_like, valid)
	dilated_z = _dilate_boolean_axis(
		zero_z_samples,
		config.z_sample_influence_radius,
		axis=0,
	)
	dilated_traces = _dilate_boolean_axis(
		_dilate_boolean_axis(
			zero_traces,
			config.xy_trace_influence_radius,
			axis=0,
		),
		config.xy_trace_influence_radius,
		axis=1,
	)
	invalid = np.empty(amplitude.shape, dtype=bool)
	np.logical_or(
		dilated_z[np.newaxis, np.newaxis, :],
		dilated_traces[:, :, np.newaxis],
		out=invalid,
	)
	return invalid


def detect_all_zero_z_samples(
	amplitude_xyz: np.ndarray,
	*,
	valid_mask: np.ndarray | None = None,
	zero_atol: float = 0.0,
) -> np.ndarray:
	"""Return [z] flags where every valid voxel at that z sample is zero-like."""
	amplitude, valid = _prepare_amplitude_and_valid_mask(amplitude_xyz, valid_mask)
	_validate_zero_atol(zero_atol)
	zero_like = np.abs(amplitude) <= np.float32(zero_atol)
	return _detect_all_zero_z_samples(zero_like, valid)


def detect_all_zero_traces(
	amplitude_xyz: np.ndarray,
	*,
	valid_mask: np.ndarray | None = None,
	zero_atol: float = 0.0,
) -> np.ndarray:
	"""Return [x, y] flags where every valid z sample in a trace is zero-like."""
	amplitude, valid = _prepare_amplitude_and_valid_mask(amplitude_xyz, valid_mask)
	_validate_zero_atol(zero_atol)
	zero_like = np.abs(amplitude) <= np.float32(zero_atol)
	return _detect_all_zero_traces(zero_like, valid)


def dilate_zero_sample_mask(
	zero_z_samples: np.ndarray,
	shape_xyz: tuple[int, int, int],
	*,
	radius_z: int,
) -> np.ndarray:
	"""Expand [z] zero-sample flags along z into a [x, y, z] invalid mask."""
	shape = _validate_shape_xyz(shape_xyz)
	radius = _validate_nonnegative_int(radius_z, 'radius_z')
	zero_z = np.asarray(zero_z_samples, dtype=bool)
	if zero_z.shape != (shape[2],):
		msg = f'zero_z_samples shape must be {(shape[2],)!r}; got {zero_z.shape!r}'
		raise ValueError(msg)

	dilated = _dilate_boolean_axis(zero_z, radius, axis=0)
	return np.broadcast_to(dilated, shape).copy()


def dilate_zero_trace_mask(
	zero_traces_xy: np.ndarray,
	shape_xyz: tuple[int, int, int],
	*,
	radius_xy: int,
) -> np.ndarray:
	"""Expand [x, y] zero-trace flags in x/y over the full z range."""
	shape = _validate_shape_xyz(shape_xyz)
	radius = _validate_nonnegative_int(radius_xy, 'radius_xy')
	zero_traces = np.asarray(zero_traces_xy, dtype=bool)
	if zero_traces.shape != shape[:2]:
		msg = f'zero_traces_xy shape must be {shape[:2]!r}; got {zero_traces.shape!r}'
		raise ValueError(msg)

	dilated = _dilate_boolean_axis(
		_dilate_boolean_axis(zero_traces, radius, axis=0),
		radius,
		axis=1,
	)
	return np.broadcast_to(dilated[:, :, np.newaxis], shape).copy()


def _prepare_amplitude_and_valid_mask(
	amplitude_xyz: np.ndarray,
	valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
	amplitude, valid = _as_amplitude_and_valid_mask(amplitude_xyz, valid_mask)
	return _replace_nonfinite_with_zero(amplitude), valid


def _as_amplitude_and_valid_mask(
	amplitude_xyz: np.ndarray,
	valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
	amplitude = np.asarray(amplitude_xyz, dtype=np.float32)
	if amplitude.ndim != 3:
		msg = f'amplitude_xyz must be a 3D [x, y, z] array; got ndim={amplitude.ndim}'
		raise ValueError(msg)
	if any(axis <= 0 for axis in amplitude.shape):
		msg = (
			'amplitude_xyz shape must be non-empty in all dimensions; '
			f'got {amplitude.shape!r}'
		)
		raise ValueError(msg)
	if valid_mask is None:
		valid = None
	else:
		valid = np.asarray(valid_mask, dtype=bool)
		if valid.shape != amplitude.shape:
			msg = (
				'valid_mask shape must match amplitude_xyz shape; got '
				f'{valid.shape!r} and {amplitude.shape!r}'
			)
			raise ValueError(msg)
	return amplitude, valid


def _replace_nonfinite_with_zero(amplitude: np.ndarray) -> np.ndarray:
	return np.nan_to_num(
		amplitude,
		nan=0.0,
		posinf=0.0,
		neginf=0.0,
	).astype(np.float32, copy=False)


def _detect_all_zero_z_samples(
	zero_like: np.ndarray,
	valid: np.ndarray | None,
) -> np.ndarray:
	if valid is None:
		return np.all(zero_like, axis=(0, 1))
	any_valid = np.any(valid, axis=(0, 1))
	all_valid_zero = np.all(zero_like | ~valid, axis=(0, 1))
	return (any_valid & all_valid_zero).astype(bool, copy=False)


def _detect_all_zero_traces(
	zero_like: np.ndarray,
	valid: np.ndarray | None,
) -> np.ndarray:
	if valid is None:
		return np.all(zero_like, axis=2)
	any_valid = np.any(valid, axis=2)
	all_valid_zero = np.all(zero_like | ~valid, axis=2)
	return (any_valid & all_valid_zero).astype(bool, copy=False)


def _dilate_boolean_axis(
	mask: np.ndarray,
	radius: int,
	*,
	axis: int,
) -> np.ndarray:
	if radius == 0:
		return np.asarray(mask, dtype=bool)
	pad_width = [(0, 0)] * mask.ndim
	pad_width[axis] = (radius, radius)
	padded = np.pad(mask, pad_width, mode='constant', constant_values=False)
	cumulative = np.cumsum(padded, axis=axis, dtype=np.int32)
	zero_shape = list(cumulative.shape)
	zero_shape[axis] = 1
	cumulative = np.concatenate(
		(np.zeros(zero_shape, dtype=np.int32), cumulative),
		axis=axis,
	)
	window = 2 * radius + 1
	upper = [slice(None)] * mask.ndim
	lower = [slice(None)] * mask.ndim
	upper[axis] = slice(window, None)
	lower[axis] = slice(None, -window)
	return (cumulative[tuple(upper)] - cumulative[tuple(lower)]) > 0


def _validate_shape_xyz(shape_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
	if len(shape_xyz) != 3:
		msg = f'shape_xyz must contain three dimensions; got {shape_xyz!r}'
		raise ValueError(msg)
	return tuple(_validate_positive_int(axis, 'shape_xyz axis') for axis in shape_xyz)


def _validate_positive_int(value: int, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be a positive integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{name} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return integer


def _validate_nonnegative_int(value: int, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be a non-negative integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer < 0:
		msg = f'{name} must be a non-negative integer; got {value!r}'
		raise ValueError(msg)
	return integer


def _validate_zero_atol(zero_atol: float) -> None:
	if isinstance(zero_atol, bool) or not isinstance(zero_atol, Real):
		msg = f'zero_atol must be a non-negative real number; got {zero_atol!r}'
		raise TypeError(msg)
	if zero_atol < 0.0:
		msg = f'zero_atol must be a non-negative real number; got {zero_atol!r}'
		raise ValueError(msg)


__all__ = [
	'DEFAULT_ZERO_MASK_CONFIG',
	'ZeroMaskConfig',
	'compute_zero_amplitude_invalid_mask',
	'detect_all_zero_traces',
	'detect_all_zero_z_samples',
	'dilate_zero_sample_mask',
	'dilate_zero_trace_mask',
]
