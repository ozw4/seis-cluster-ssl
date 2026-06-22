"""Count-averaging merge utilities for overlapping token embeddings."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Sequence
	from pathlib import Path

	from seis_ssl_cluster.embedding.sliding_window import SlidingWindow

XYZ = tuple[int, int, int]


class EmbeddingMerger:
	"""Accumulate overlapping token embeddings and write count averages."""

	def __init__(
		self,
		*,
		token_grid_shape_xyz: Sequence[int],
		embedding_dim: int,
		sum_array: np.ndarray | None = None,
		count_array: np.ndarray | None = None,
	) -> None:
		"""Initialize accumulation arrays for one full-volume token grid."""
		self.token_grid_shape_xyz = _validate_positive_xyz(
			token_grid_shape_xyz,
			'token_grid_shape_xyz',
		)
		self.embedding_dim = _validate_positive_int(embedding_dim, 'embedding_dim')
		sum_shape = (*self.token_grid_shape_xyz, self.embedding_dim)
		count_shape = self.token_grid_shape_xyz
		self.sums = (
			np.zeros(sum_shape, dtype=np.float32)
			if sum_array is None
			else _validate_array(sum_array, sum_shape, np.float32, 'sum_array')
		)
		self.counts = (
			np.zeros(count_shape, dtype=np.uint32)
			if count_array is None
			else _validate_array(count_array, count_shape, np.uint32, 'count_array')
		)

	def add_window(
		self,
		window: SlidingWindow,
		*,
		patch_size_xyz: Sequence[int],
		token_embeddings: np.ndarray,
		token_valid_mask: np.ndarray,
	) -> None:
		"""Add one window's token embeddings using valid-token count weights."""
		patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
		token_start = cast(
			'XYZ',
			tuple(
				start // patch_axis
				for start, patch_axis in zip(window.start_xyz, patch, strict=True)
			),
		)
		window_token_shape = cast(
			'XYZ',
			tuple(
				size // patch_axis
				for size, patch_axis in zip(window.size_xyz, patch, strict=True)
			),
		)
		embeddings = np.asarray(token_embeddings, dtype=np.float32)
		expected_embedding_shape = (*window_token_shape, self.embedding_dim)
		if embeddings.shape != expected_embedding_shape:
			msg = (
				'token_embeddings shape must match window token grid and '
				'embedding_dim; got '
				f'{embeddings.shape!r}, expected={expected_embedding_shape!r}'
			)
			raise ValueError(msg)
		valid = np.asarray(token_valid_mask, dtype=bool)
		if valid.shape != window_token_shape:
			msg = (
				'token_valid_mask shape must match window token grid; '
				f'got {valid.shape!r}, expected={window_token_shape!r}'
			)
			raise ValueError(msg)
		if not valid.any():
			return

		target_slices, source_slices = _merge_slices(
			token_start,
			window_token_shape,
			self.token_grid_shape_xyz,
		)
		target = cast('tuple[slice, slice, slice]', target_slices)
		source = cast('tuple[slice, slice, slice]', source_slices)
		valid_source = valid[source]
		self.sums[target] += embeddings[source] * valid_source[..., np.newaxis]
		self.counts[target] += valid_source.astype(np.uint32, copy=False)

	def finalize(
		self,
		*,
		output_dtype: np.dtype | str = np.float16,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Return averaged embeddings and the derived valid-token mask."""
		dtype = np.dtype(output_dtype)
		valid = self.counts > 0
		embeddings = np.zeros(
			(*self.token_grid_shape_xyz, self.embedding_dim),
			dtype=dtype,
		)
		with np.errstate(divide='ignore', invalid='ignore'):
			averaged = self.sums / np.maximum(self.counts[..., np.newaxis], 1)
		embeddings[valid] = averaged[valid].astype(dtype, copy=False)
		return embeddings, valid

	def write_average(
		self,
		*,
		embedding_path: Path,
		valid_tokens_path: Path,
		output_dtype: np.dtype | str,
	) -> None:
		"""Write averaged embeddings and valid mask as `.npy` memmaps."""
		dtype = np.dtype(output_dtype)
		embedding_path.parent.mkdir(parents=True, exist_ok=True)
		valid_tokens_path.parent.mkdir(parents=True, exist_ok=True)
		embeddings = np.lib.format.open_memmap(
			embedding_path,
			mode='w+',
			dtype=dtype,
			shape=(*self.token_grid_shape_xyz, self.embedding_dim),
		)
		valid_tokens = np.lib.format.open_memmap(
			valid_tokens_path,
			mode='w+',
			dtype=np.bool_,
			shape=self.token_grid_shape_xyz,
		)
		for x_index in range(self.token_grid_shape_xyz[0]):
			counts = self.counts[x_index]
			valid = counts > 0
			out = np.zeros(
				(*self.token_grid_shape_xyz[1:], self.embedding_dim),
				dtype=np.float32,
			)
			out[valid] = self.sums[x_index][valid] / counts[valid, np.newaxis]
			embeddings[x_index] = out.astype(dtype, copy=False)
			valid_tokens[x_index] = valid
		embeddings.flush()
		valid_tokens.flush()


def _merge_slices(
	token_start: XYZ,
	window_token_shape: XYZ,
	token_grid_shape: XYZ,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
	target: list[slice] = []
	source: list[slice] = []
	for start, size, limit in zip(
		token_start,
		window_token_shape,
		token_grid_shape,
		strict=True,
	):
		stop = start + size
		clipped_start = max(start, 0)
		clipped_stop = min(stop, limit)
		if clipped_stop <= clipped_start:
			msg = (
				'window token range does not intersect output token grid; '
				f'token_start={token_start!r}, '
				f'window_token_shape={window_token_shape!r}, '
				f'token_grid_shape={token_grid_shape!r}'
			)
			raise ValueError(msg)
		target.append(slice(clipped_start, clipped_stop))
		source.append(slice(clipped_start - start, clipped_stop - start))
	return (
		cast('tuple[slice, slice, slice]', tuple(target)),
		cast('tuple[slice, slice, slice]', tuple(source)),
	)


def _validate_array(
	array: np.ndarray,
	shape: tuple[int, ...],
	dtype: np.dtype,
	name: str,
) -> np.ndarray:
	if array.shape != shape:
		msg = f'{name} shape must be {shape!r}; got {array.shape!r}'
		raise ValueError(msg)
	if array.dtype != dtype:
		msg = f'{name} dtype must be {dtype}; got {array.dtype}'
		raise TypeError(msg)
	array[...] = 0
	return array


def _validate_positive_xyz(value: Sequence[int], name: str) -> XYZ:
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
	xyz = cast('XYZ', tuple(int(axis) for axis in value))
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_positive_int(value: int, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{name} must be positive; got {integer!r}'
		raise ValueError(msg)
	return integer


__all__ = ['EmbeddingMerger']
