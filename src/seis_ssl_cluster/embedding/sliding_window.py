"""Patch-aligned sliding windows for full-volume embedding extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
	from collections.abc import Iterator, Sequence

XYZ = tuple[int, int, int]


@dataclass(frozen=True)
class SlidingWindow:
	"""One deterministic full-volume extraction window in voxel coordinates."""

	start_xyz: XYZ
	size_xyz: XYZ

	@property
	def stop_xyz(self) -> XYZ:
		"""Return exclusive voxel stop coordinates."""
		return cast(
			'XYZ',
			tuple(
				start + size
				for start, size in zip(self.start_xyz, self.size_xyz, strict=True)
			),
		)


def iter_sliding_windows(
	volume_shape_xyz: Sequence[int],
	*,
	window_size_xyz: Sequence[int],
	overlap_xyz: Sequence[int],
	patch_size_xyz: Sequence[int],
) -> Iterator[SlidingWindow]:
	"""Yield patch-aligned windows covering a possibly non-divisible volume."""
	shape = _validate_positive_xyz(volume_shape_xyz, 'volume_shape_xyz')
	window = _validate_positive_xyz(window_size_xyz, 'window_size_xyz')
	overlap = _validate_nonnegative_xyz(overlap_xyz, 'overlap_xyz')
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	_validate_window_geometry(window, overlap, patch)

	stride = compute_stride_xyz(window, overlap)
	padded_shape = padded_volume_shape_xyz(shape, patch)
	starts_by_axis = tuple(
		_axis_starts(padded_axis, window_axis, stride_axis)
		for padded_axis, window_axis, stride_axis in zip(
			padded_shape,
			window,
			stride,
			strict=True,
		)
	)
	for x_start in starts_by_axis[0]:
		for y_start in starts_by_axis[1]:
			for z_start in starts_by_axis[2]:
				yield SlidingWindow(
					start_xyz=(x_start, y_start, z_start),
					size_xyz=window,
				)


def compute_stride_xyz(
	window_size_xyz: Sequence[int],
	overlap_xyz: Sequence[int],
) -> XYZ:
	"""Return positive per-axis extraction stride."""
	window = _validate_positive_xyz(window_size_xyz, 'window_size_xyz')
	overlap = _validate_nonnegative_xyz(overlap_xyz, 'overlap_xyz')
	stride = tuple(
		window_axis - overlap_axis
		for window_axis, overlap_axis in zip(window, overlap, strict=True)
	)
	if any(axis <= 0 for axis in stride):
		msg = (
			'overlap_xyz must be smaller than window_size_xyz on every axis; '
			f'got window_size_xyz={window!r}, overlap_xyz={overlap!r}'
		)
		raise ValueError(msg)
	return cast('XYZ', stride)


def padded_volume_shape_xyz(
	volume_shape_xyz: Sequence[int],
	patch_size_xyz: Sequence[int],
) -> XYZ:
	"""Return the full-volume shape padded up to patch multiples."""
	shape = _validate_positive_xyz(volume_shape_xyz, 'volume_shape_xyz')
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	return cast(
		'XYZ',
		tuple(
			math.ceil(shape_axis / patch_axis) * patch_axis
			for shape_axis, patch_axis in zip(shape, patch, strict=True)
		),
	)


def token_grid_shape_xyz(
	volume_shape_xyz: Sequence[int],
	patch_size_xyz: Sequence[int],
) -> XYZ:
	"""Return the full-volume token grid, including boundary padding tokens."""
	padded = padded_volume_shape_xyz(volume_shape_xyz, patch_size_xyz)
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	return cast(
		'XYZ',
		tuple(
			padded_axis // patch_axis
			for padded_axis, patch_axis in zip(padded, patch, strict=True)
		),
	)


def _validate_window_geometry(
	window: XYZ,
	overlap: XYZ,
	patch: XYZ,
) -> None:
	stride = compute_stride_xyz(window, overlap)
	for name, values in (
		('window_size_xyz', window),
		('overlap_xyz', overlap),
		('stride_xyz', stride),
	):
		if any(
			value % patch_axis != 0
			for value, patch_axis in zip(values, patch, strict=True)
		):
			msg = (
				f'{name} must align to patch_size_xyz; '
				f'got {name}={values!r}, patch_size_xyz={patch!r}'
			)
			raise ValueError(msg)


def _axis_starts(
	padded_size: int,
	window_size: int,
	stride: int,
) -> tuple[int, ...]:
	if padded_size <= window_size:
		return (0,)
	last_start = padded_size - window_size
	starts = list(range(0, last_start + 1, stride))
	if starts[-1] != last_start:
		starts.append(last_start)
	return tuple(starts)


def _validate_positive_xyz(value: Sequence[int], name: str) -> XYZ:
	xyz = _validate_xyz(value, name)
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_nonnegative_xyz(value: Sequence[int], name: str) -> XYZ:
	xyz = _validate_xyz(value, name)
	if any(axis < 0 for axis in xyz):
		msg = f'{name} values must be nonnegative; got {xyz!r}'
		raise ValueError(msg)
	return xyz


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
	return cast('XYZ', tuple(int(axis) for axis in value))


__all__ = [
	'SlidingWindow',
	'compute_stride_xyz',
	'iter_sliding_windows',
	'padded_volume_shape_xyz',
	'token_grid_shape_xyz',
]
