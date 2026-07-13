"""Pure token-to-voxel geometry helpers for the F3 lithology benchmark."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Sequence


def project_token_grid_nearest(
	values: np.ndarray,
	*,
	patch_size_xyz: Sequence[int],
	volume_shape_xyz: Sequence[int],
) -> np.ndarray:
	"""Repeat the first three token axes and crop to the original XYZ volume."""
	array = np.asarray(values)
	if array.ndim < 3:
		msg = f'values must have at least three token axes; got shape {array.shape!r}'
		raise ValueError(msg)
	patch_size = _positive_int_triplet(patch_size_xyz, name='patch_size_xyz')
	volume_shape = _positive_int_triplet(volume_shape_xyz, name='volume_shape_xyz')
	covered_shape = tuple(
		token_count * patch_size[axis]
		for axis, token_count in enumerate(array.shape[:3])
	)
	if any(
		covered < required
		for covered, required in zip(covered_shape, volume_shape, strict=True)
	):
		msg = (
			'token grid does not cover volume_shape_xyz after repetition; '
			f'covered shape is {covered_shape!r}, volume shape is {volume_shape!r}'
		)
		raise ValueError(msg)

	projected = array
	for axis, repeats in enumerate(patch_size):
		projected = np.repeat(projected, repeats, axis=axis)
	return projected[
		: volume_shape[0],
		: volume_shape[1],
		: volume_shape[2],
		...,
	]


def valid_tokens_to_voxel_mask(
	valid_tokens: np.ndarray,
	*,
	patch_size_xyz: Sequence[int],
	volume_shape_xyz: Sequence[int],
) -> np.ndarray:
	"""Return a bool XYZ mask using the nearest token-to-voxel mapping."""
	array = np.asarray(valid_tokens)
	if array.ndim != 3:
		msg = f'valid_tokens must be 3D [TX, TY, TZ]; got shape {array.shape!r}'
		raise ValueError(msg)
	if array.dtype != np.bool_:
		msg = f'valid_tokens dtype must be bool; got {array.dtype}'
		raise TypeError(msg)
	return project_token_grid_nearest(
		array,
		patch_size_xyz=patch_size_xyz,
		volume_shape_xyz=volume_shape_xyz,
	)


def _positive_int_triplet(
	value: Sequence[int],
	*,
	name: str,
) -> tuple[int, int, int]:
	if isinstance(value, (str, bytes)) or len(value) != 3:
		msg = f'{name} must be a positive integer triple'
		raise ValueError(msg)
	if any(not isinstance(item, Integral) or isinstance(item, bool) for item in value):
		msg = f'{name} must be a positive integer triple'
		raise TypeError(msg)
	result = tuple(int(item) for item in value)
	if any(item <= 0 for item in result):
		msg = f'{name} must be a positive integer triple; got {result!r}'
		raise ValueError(msg)
	return result  # type: ignore[return-value]
