"""Shared array helpers for training-time visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

ArrayLike: TypeAlias = torch.Tensor | np.ndarray


@dataclass(frozen=True)
class ImagePanel:
	"""One rendered image panel and its display metadata."""

	title: str
	image: np.ndarray
	valid_mask: np.ndarray | None = None
	range_name: str = 'amplitude'


def as_numpy(value: ArrayLike, name: str) -> np.ndarray:
	"""Detach tensors and return NumPy arrays for rendering."""
	if isinstance(value, np.ndarray):
		return value
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().numpy()
	msg = f'{name} must be a torch.Tensor or np.ndarray; got {type(value).__name__}'
	raise TypeError(msg)


def apply_visual_invalid_mask(
	image: np.ndarray,
	valid_mask: np.ndarray | None,
) -> np.ndarray | np.ma.MaskedArray:
	"""Mask invalid voxels for display without changing numeric values."""
	if valid_mask is None:
		return image
	return np.ma.masked_where(~np.asarray(valid_mask, dtype=bool), image, copy=True)


def upsample_token_mask_to_voxels(
	spatial_mask: ArrayLike,
	*,
	patch_size_xyz: tuple[int, int, int],
) -> np.ndarray:
	"""Convert a token mask from ``[B, TX, TY, TZ]`` to voxel space."""
	mask = as_numpy(spatial_mask, 'spatial_mask')
	if mask.ndim != 4:
		msg = (
			'spatial_mask must have shape [B, TX, TY, TZ]; '
			f'got shape={mask.shape!r}'
		)
		raise ValueError(msg)
	px_size, py_size, pz_size = validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	return (
		mask.astype(bool, copy=False)
		.repeat(px_size, axis=1)
		.repeat(py_size, axis=2)
		.repeat(pz_size, axis=3)
	)


def unpatchify_mae_predictions(
	pred_patches: ArrayLike,
	*,
	token_grid_shape: tuple[int, int, int],
	patch_size_xyz: tuple[int, int, int],
) -> np.ndarray:
	"""Convert MAE patch predictions from ``[B, N, 1, PV]`` to voxels."""
	patches = as_numpy(pred_patches, 'pred_patches')
	if patches.ndim != 4:
		msg = (
			'pred_patches must have shape [B, N, C, patch_volume]; '
			f'got shape={patches.shape!r}'
		)
		raise ValueError(msg)
	tx_size, ty_size, tz_size = validate_positive_int_triple(
		token_grid_shape,
		'token_grid_shape',
	)
	px_size, py_size, pz_size = validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	batch_size, num_tokens, channels, patch_volume = patches.shape
	expected_num_tokens = tx_size * ty_size * tz_size
	expected_patch_volume = px_size * py_size * pz_size
	if num_tokens != expected_num_tokens or patch_volume != expected_patch_volume:
		msg = (
			'pred_patches shape must match token_grid_shape and patch_size_xyz; '
			f'got shape={patches.shape!r}, '
			f'token_grid_shape={token_grid_shape!r}, '
			f'patch_size_xyz={patch_size_xyz!r}, '
			f'expected_num_tokens={expected_num_tokens}, '
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
		.transpose(0, 4, 1, 5, 2, 6, 3, 7)
		.reshape(
			batch_size,
			channels,
			tx_size * px_size,
			ty_size * py_size,
			tz_size * pz_size,
		)
	)


def slice_image(volume: np.ndarray, *, view: str, slice_index: int) -> np.ndarray:
	"""Return a display-oriented XY or XZ slice from an XYZ volume."""
	if view == 'xy':
		return np.asarray(volume[:, :, slice_index]).T
	if view == 'xz':
		return np.asarray(volume[:, slice_index, :]).T
	msg = f'unknown view: {view!r}'
	raise ValueError(msg)


def slice_mask(
	mask: np.ndarray | None,
	*,
	view: str,
	slice_index: int,
) -> np.ndarray | None:
	"""Return a display-oriented boolean mask slice."""
	if mask is None:
		return None
	return slice_image(mask, view=view, slice_index=slice_index).astype(
		bool,
		copy=False,
	)


def display_limits(
	image: np.ndarray | np.ma.MaskedArray,
	clip_percentiles: tuple[float, float],
	*,
	error: bool = False,
) -> tuple[float | None, float | None]:
	"""Return robust display limits while ignoring invalid and masked values."""
	values = np.ma.masked_invalid(image).compressed()
	if values.size == 0:
		return None, None
	if error:
		return 0.0, float(np.percentile(values, clip_percentiles[1]))
	vmin, vmax = np.percentile(values, clip_percentiles)
	if np.isclose(vmin, vmax):
		center = float(np.mean(values))
		half = float(np.std(values)) or 1.0
		return center - half, center + half
	return float(vmin), float(vmax)


def validate_positive_int_triple(
	value: tuple[int, int, int],
	name: str,
) -> tuple[int, int, int]:
	"""Validate an XYZ integer triple."""
	if (
		not isinstance(value, tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
		or any(item <= 0 for item in value)
	):
		msg = f'{name} must be a positive integer triple; got {value!r}'
		raise ValueError(msg)
	return value


__all__ = [
	'ArrayLike',
	'ImagePanel',
	'apply_visual_invalid_mask',
	'as_numpy',
	'display_limits',
	'slice_image',
	'slice_mask',
	'unpatchify_mae_predictions',
	'upsample_token_mask_to_voxels',
	'validate_positive_int_triple',
]
