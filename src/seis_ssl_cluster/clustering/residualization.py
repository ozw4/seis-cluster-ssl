"""Local token position residualization for clustering features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from seis_ssl_cluster.clustering.features import EmbeddingInput, load_embedding_metadata

SUPPORTED_RESIDUALIZATION_MODES = frozenset({'local_token_position'})
SUPPORTED_RESIDUALIZATION_GROUP_BY = frozenset(
	{'token_phase', 'local_token_position'},
)


@dataclass(frozen=True)
class LocalTokenPositionResidualizer:
	"""Remove group-wise mean feature bias before downstream clustering."""

	mode: str
	group_by: str
	add_global_mean_back: bool
	min_group_count: int
	global_mean: np.ndarray
	group_means: dict[tuple[int, int, int], np.ndarray]
	group_counts: dict[tuple[int, int, int], int]

	def transform(
		self,
		embeddings: np.ndarray,
		group_keys: np.ndarray,
	) -> np.ndarray:
		"""Apply fitted group mean residualization to a feature matrix."""
		matrix = np.asarray(embeddings, dtype=np.float32)
		if matrix.ndim != 2:
			msg = f'embeddings must be a 2D matrix; got {matrix.shape!r}'
			raise ValueError(msg)
		keys = _validate_group_keys(group_keys, expected_count=matrix.shape[0])
		transformed = matrix.copy()
		for key in np.unique(keys, axis=0):
			group_key = _group_key_tuple(key)
			mask = np.all(keys == key, axis=1)
			group_mean = self.group_means.get(group_key, self.global_mean)
			transformed[mask] -= group_mean
			if self.add_global_mean_back:
				transformed[mask] += self.global_mean
		return transformed.astype(np.float32, copy=False)

	def summary(self) -> dict[str, object]:
		"""Return compact JSON-safe metadata for fitted residualization."""
		counts = list(self.group_counts.values())
		return {
			'enabled': True,
			'mode': self.mode,
			'group_by': self.group_by,
			'add_global_mean_back': self.add_global_mean_back,
			'min_group_count': self.min_group_count,
			'groups': len(self.group_counts),
			'min_observed_group_count': int(min(counts)) if counts else 0,
			'max_observed_group_count': int(max(counts)) if counts else 0,
			'global_mean_l2_norm': float(np.linalg.norm(self.global_mean)),
		}


def fit_local_token_position_residualizer(
	embeddings: np.ndarray,
	group_keys: np.ndarray,
	*,
	group_by: str,
	add_global_mean_back: bool,
	min_group_count: int,
) -> LocalTokenPositionResidualizer:
	"""Fit group-wise mean residualization statistics."""
	if group_by not in SUPPORTED_RESIDUALIZATION_GROUP_BY:
		msg = f'unsupported residualization group_by: {group_by!r}'
		raise ValueError(msg)
	if min_group_count <= 0:
		msg = f'min_group_count must be positive; got {min_group_count!r}'
		raise ValueError(msg)
	matrix = np.asarray(embeddings, dtype=np.float32)
	if matrix.ndim != 2 or matrix.shape[0] == 0:
		msg = f'embeddings must be a non-empty 2D matrix; got {matrix.shape!r}'
		raise ValueError(msg)
	keys = _validate_group_keys(group_keys, expected_count=matrix.shape[0])
	global_mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
	group_means: dict[tuple[int, int, int], np.ndarray] = {}
	group_counts: dict[tuple[int, int, int], int] = {}
	for key in np.unique(keys, axis=0):
		group_key = _group_key_tuple(key)
		mask = np.all(keys == key, axis=1)
		count = int(np.count_nonzero(mask))
		group_counts[group_key] = count
		if count < min_group_count:
			group_means[group_key] = global_mean.copy()
		else:
			group_means[group_key] = (
				matrix[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
			)
	return LocalTokenPositionResidualizer(
		mode='local_token_position',
		group_by=group_by,
		add_global_mean_back=add_global_mean_back,
		min_group_count=int(min_group_count),
		global_mean=global_mean,
		group_means=group_means,
		group_counts=group_counts,
	)


def token_phase_keys_for_grid(
	token_grid_shape_xyz: tuple[int, int, int],
	*,
	patch_size_xyz: tuple[int, int, int],
	window_size_xyz: tuple[int, int, int],
	overlap_xyz: tuple[int, int, int],
	valid_mask: np.ndarray | None = None,
) -> np.ndarray:
	"""Return token phase keys for a token grid, optionally only valid tokens."""
	shape = _positive_int_triplet(token_grid_shape_xyz, 'token_grid_shape_xyz')
	stride_tokens = _stride_tokens(
		patch_size_xyz=patch_size_xyz,
		window_size_xyz=window_size_xyz,
		overlap_xyz=overlap_xyz,
	)
	coords = np.indices(shape, dtype=np.int64).reshape(3, -1).T
	keys = coords % np.asarray(stride_tokens, dtype=np.int64)
	if valid_mask is None:
		return keys.astype(np.int64, copy=False)
	valid = np.asarray(valid_mask)
	if valid.shape != shape:
		msg = f'valid_mask shape must be {shape!r}; got {valid.shape!r}'
		raise ValueError(msg)
	if valid.dtype != np.bool_:
		msg = f'valid_mask dtype must be bool; got {valid.dtype}'
		raise TypeError(msg)
	return keys[valid.reshape(-1)].astype(np.int64, copy=False)


def residualization_keys_for_flat_indices(
	embedding_input: EmbeddingInput,
	token_indices: np.ndarray,
	*,
	group_by: str,
) -> np.ndarray:
	"""Return residualization group keys for flattened token indices."""
	indices = np.asarray(token_indices, dtype=np.int64)
	if indices.ndim != 1:
		msg = f'token_indices must be 1D; got {indices.shape!r}'
		raise ValueError(msg)
	metadata = load_embedding_metadata(embedding_input)
	if group_by == 'token_phase':
		return _token_phase_keys_for_flat_indices(metadata, indices)
	if group_by == 'local_token_position':
		msg = (
			'clustering.residualization.group_by=local_token_position requires '
			'exact per-token local position metadata, which is not present in '
			f'{embedding_input.metadata_path}'
		)
		raise ValueError(msg)
	msg = f'unsupported residualization group_by: {group_by!r}'
	raise ValueError(msg)


def sample_residualization_keys(
	embedding_inputs: tuple[EmbeddingInput, ...],
	per_survey_token_indices: Mapping[str, np.ndarray],
	*,
	group_by: str,
) -> np.ndarray:
	"""Return group keys ordered like sampled feature blocks."""
	blocks: list[np.ndarray] = []
	for item in embedding_inputs:
		indices = per_survey_token_indices[item.survey_id]
		if indices.size:
			blocks.append(
				residualization_keys_for_flat_indices(
					item,
					indices,
					group_by=group_by,
				),
			)
	if not blocks:
		return np.empty((0, 3), dtype=np.int64)
	return np.concatenate(blocks, axis=0).astype(np.int64, copy=False)


def write_residualizer_npz(
	path: str | Path,
	residualizer: LocalTokenPositionResidualizer,
) -> None:
	"""Persist residualizer statistics in a reusable compact NPZ file."""
	npz_path = Path(path)
	npz_path.parent.mkdir(parents=True, exist_ok=True)
	group_keys = np.asarray(list(residualizer.group_means), dtype=np.int64)
	group_means = np.asarray(
		[residualizer.group_means[_group_key_tuple(key)] for key in group_keys],
		dtype=np.float32,
	)
	group_counts = np.asarray(
		[residualizer.group_counts[_group_key_tuple(key)] for key in group_keys],
		dtype=np.int64,
	)
	np.savez(
		npz_path,
		global_mean=np.asarray(residualizer.global_mean, dtype=np.float32),
		group_keys=group_keys,
		group_means=group_means,
		group_counts=group_counts,
		mode=np.asarray(residualizer.mode),
		group_by=np.asarray(residualizer.group_by),
		add_global_mean_back=np.asarray(residualizer.add_global_mean_back),
		min_group_count=np.asarray(residualizer.min_group_count, dtype=np.int64),
	)


def read_residualizer_npz(path: str | Path) -> LocalTokenPositionResidualizer:
	"""Load residualizer statistics saved by write_residualizer_npz."""
	with np.load(path, allow_pickle=False) as payload:
		group_keys = np.asarray(payload['group_keys'], dtype=np.int64)
		group_means_array = np.asarray(payload['group_means'], dtype=np.float32)
		group_counts_array = np.asarray(payload['group_counts'], dtype=np.int64)
		return LocalTokenPositionResidualizer(
			mode=str(payload['mode'].item()),
			group_by=str(payload['group_by'].item()),
			add_global_mean_back=bool(payload['add_global_mean_back'].item()),
			min_group_count=int(payload['min_group_count'].item()),
			global_mean=np.asarray(payload['global_mean'], dtype=np.float32),
			group_means={
				_group_key_tuple(key): np.asarray(mean, dtype=np.float32)
				for key, mean in zip(group_keys, group_means_array, strict=True)
			},
			group_counts={
				_group_key_tuple(key): int(count)
				for key, count in zip(group_keys, group_counts_array, strict=True)
			},
		)


def residualization_metadata_disabled() -> dict[str, object]:
	"""Return explicit disabled residualization metadata."""
	return {'enabled': False}


def _token_phase_keys_for_flat_indices(
	metadata: Mapping[str, object],
	indices: np.ndarray,
) -> np.ndarray:
	shape = _metadata_triplet(metadata, 'token_grid_shape')
	patch_size = _metadata_triplet(metadata, 'patch_size')
	window_size = _metadata_triplet(metadata, 'window_size')
	overlap = _metadata_triplet(metadata, 'overlap')
	stride_tokens = _stride_tokens(
		patch_size_xyz=patch_size,
		window_size_xyz=window_size,
		overlap_xyz=overlap,
	)
	coords = np.column_stack(np.unravel_index(indices, shape)).astype(np.int64)
	return (coords % np.asarray(stride_tokens, dtype=np.int64)).astype(
		np.int64,
		copy=False,
	)


def _stride_tokens(
	*,
	patch_size_xyz: tuple[int, int, int],
	window_size_xyz: tuple[int, int, int],
	overlap_xyz: tuple[int, int, int],
) -> tuple[int, int, int]:
	patch_size = _positive_int_triplet(patch_size_xyz, 'patch_size_xyz')
	window_size = _positive_int_triplet(window_size_xyz, 'window_size_xyz')
	overlap = _nonnegative_int_triplet(overlap_xyz, 'overlap_xyz')
	stride_voxels = tuple(
		window - over
		for window, over in zip(window_size, overlap, strict=True)
	)
	if any(stride <= 0 for stride in stride_voxels):
		msg = (
			'window_size_xyz - overlap_xyz must be positive; '
			f'got window_size={window_size!r}, overlap={overlap!r}'
		)
		raise ValueError(msg)
	if any(
		stride % patch
		for stride, patch in zip(stride_voxels, patch_size, strict=True)
	):
		msg = (
			'window_size_xyz - overlap_xyz must be divisible by patch_size_xyz; '
			f'got stride_voxels={stride_voxels!r}, patch_size={patch_size!r}'
		)
		raise ValueError(msg)
	return tuple(
		stride // patch
		for stride, patch in zip(stride_voxels, patch_size, strict=True)
	)


def _metadata_triplet(
	metadata: Mapping[str, object],
	key: str,
) -> tuple[int, int, int]:
	if key not in metadata:
		msg = f'embedding metadata missing required field for token_phase: {key}'
		raise ValueError(msg)
	value = metadata[key]
	if not isinstance(value, list | tuple):
		msg = f'embedding metadata field {key} must be a length-3 sequence'
		raise TypeError(msg)
	return _positive_int_triplet(cast('tuple[int, int, int]', tuple(value)), key)


def _positive_int_triplet(
	value: tuple[int, int, int],
	name: str,
) -> tuple[int, int, int]:
	if len(value) != 3 or any(not isinstance(item, int) or item <= 0 for item in value):
		msg = f'{name} must be a length-3 positive integer sequence; got {value!r}'
		raise ValueError(msg)
	return tuple(int(item) for item in value)


def _nonnegative_int_triplet(
	value: tuple[int, int, int],
	name: str,
) -> tuple[int, int, int]:
	if len(value) != 3 or any(not isinstance(item, int) or item < 0 for item in value):
		msg = f'{name} must be a length-3 nonnegative integer sequence; got {value!r}'
		raise ValueError(msg)
	return tuple(int(item) for item in value)


def _validate_group_keys(group_keys: np.ndarray, *, expected_count: int) -> np.ndarray:
	keys = np.asarray(group_keys, dtype=np.int64)
	if keys.shape != (expected_count, 3):
		msg = f'group_keys must have shape {(expected_count, 3)!r}; got {keys.shape!r}'
		raise ValueError(msg)
	return keys


def _group_key_tuple(key: np.ndarray) -> tuple[int, int, int]:
	return tuple(int(item) for item in key)


__all__ = [
	'LocalTokenPositionResidualizer',
	'SUPPORTED_RESIDUALIZATION_GROUP_BY',
	'SUPPORTED_RESIDUALIZATION_MODES',
	'fit_local_token_position_residualizer',
	'read_residualizer_npz',
	'residualization_keys_for_flat_indices',
	'residualization_metadata_disabled',
	'sample_residualization_keys',
	'token_phase_keys_for_grid',
	'write_residualizer_npz',
]
