"""3D patch conversion utilities for amplitude MAE tensors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import torch


def compute_num_patches(
	volume_size_xyz: tuple[int, int, int],
	patch_size_xyz: tuple[int, int, int],
) -> tuple[int, int, int, int]:
	"""Return token grid dimensions and total patch count for an XYZ volume."""
	x_size, y_size, z_size = _validate_positive_int_triple(
		volume_size_xyz,
		'volume_size_xyz',
	)
	px_size, py_size, pz_size = _validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	if (
		x_size % px_size != 0
		or y_size % py_size != 0
		or z_size % pz_size != 0
	):
		msg = (
			'volume_size_xyz must be exactly divisible by patch_size_xyz; '
			f'got volume_size_xyz={volume_size_xyz!r}, '
			f'patch_size_xyz={patch_size_xyz!r}'
		)
		raise ValueError(msg)

	tx_size = x_size // px_size
	ty_size = y_size // py_size
	tz_size = z_size // pz_size
	return tx_size, ty_size, tz_size, math.prod((tx_size, ty_size, tz_size))


def patchify_3d(
	x: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	"""Convert ``[B, C, X, Y, Z]`` volumes to ``[B, N, C, PX * PY * PZ]``."""
	if x.ndim != 5:
		msg = (
			'x must be a 5D tensor with shape [B, C, X, Y, Z]; '
			f'got shape={tuple(x.shape)!r}'
		)
		raise ValueError(msg)

	px_size, py_size, pz_size = _validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	batch_size, channels, x_size, y_size, z_size = x.shape
	tx_size, ty_size, tz_size, _num_patches = compute_num_patches(
		(int(x_size), int(y_size), int(z_size)),
		patch_size_xyz,
	)

	return (
		x.reshape(
			batch_size,
			channels,
			tx_size,
			px_size,
			ty_size,
			py_size,
			tz_size,
			pz_size,
		)
		.permute(0, 2, 4, 6, 1, 3, 5, 7)
		.contiguous()
		.reshape(
			batch_size,
			tx_size * ty_size * tz_size,
			channels,
			px_size * py_size * pz_size,
		)
	)


def unpatchify_3d(
	patches: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	grid_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	"""Convert ``[B, N, C, PX * PY * PZ]`` patches to ``[B, C, X, Y, Z]``."""
	if patches.ndim != 4:
		msg = (
			'patches must be a 4D tensor with shape [B, N, C, PX * PY * PZ]; '
			f'got shape={tuple(patches.shape)!r}'
		)
		raise ValueError(msg)

	px_size, py_size, pz_size = _validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	tx_size, ty_size, tz_size = _validate_positive_int_triple(
		grid_size_xyz,
		'grid_size_xyz',
	)
	expected_num_patches = tx_size * ty_size * tz_size
	expected_patch_volume = px_size * py_size * pz_size
	batch_size, num_patches, channels, patch_volume = patches.shape

	if num_patches != expected_num_patches or patch_volume != expected_patch_volume:
		msg = (
			'patches shape must match grid_size_xyz and patch_size_xyz; '
			f'got shape={tuple(patches.shape)!r}, '
			f'grid_size_xyz={grid_size_xyz!r}, '
			f'patch_size_xyz={patch_size_xyz!r}, '
			f'expected_num_patches={expected_num_patches}, '
			f'expected_patch_volume={expected_patch_volume}'
		)
		raise ValueError(msg)

	return (
		patches.reshape(
			batch_size,
			tx_size,
			ty_size,
			tz_size,
			channels,
			px_size,
			py_size,
			pz_size,
		)
		.permute(0, 4, 1, 5, 2, 6, 3, 7)
		.contiguous()
		.reshape(
			batch_size,
			channels,
			tx_size * px_size,
			ty_size * py_size,
			tz_size * pz_size,
		)
	)


def _validate_positive_int_triple(
	value: tuple[int, int, int],
	name: str,
) -> tuple[int, int, int]:
	if (
		not isinstance(value, tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
		or any(item <= 0 for item in value)
	):
		msg = f'{name} must be a positive integer triple; got {value!r}'
		raise ValueError(msg)
	return value


__all__ = ['compute_num_patches', 'patchify_3d', 'unpatchify_3d']
