"""Local token position residualization for clustering features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.clustering.features import EmbeddingInput, load_embedding_metadata

if TYPE_CHECKING:
	from collections.abc import Mapping

SUPPORTED_RESIDUALIZATION_MODES = frozenset({'local_token_position'})
SUPPORTED_RESIDUALIZATION_GROUP_BY = frozenset(
	{'token_phase', 'local_token_position'},
)
RESIDUALIZER_SCHEMA_VERSION = 2
_DEFAULT_FIT_CHUNK_SIZE = 65_536


@dataclass(frozen=True)
class LocalTokenPositionResidualizer:
	"""Remove group-wise mean feature bias before downstream clustering."""

	mode: str
	group_by: str
	add_global_mean_back: bool
	min_group_count: int
	means: np.ndarray
	counts: np.ndarray
	group_shape: tuple[int, int, int] | None
	fallback_mean: np.ndarray
	legacy_group_keys: np.ndarray | None = None

	@property
	def global_mean(self) -> np.ndarray:
		"""Return the fallback mean under the legacy attribute name."""
		return self.fallback_mean

	@property
	def group_means(self) -> dict[tuple[int, int, int], np.ndarray]:
		"""Return observed dense means keyed by XYZ coordinate."""
		if self.legacy_group_keys is not None:
			return {
				_group_key_tuple(key): mean
				for key, mean in zip(self.legacy_group_keys, self.means, strict=True)
			}
		group_shape = _require_dense_group_shape(self.group_shape)
		return {
			tuple(
				int(value)
				for value in np.unravel_index(group_id, group_shape)
			): mean
			for group_id, mean in enumerate(self.means)
			if self.counts[group_id] > 0
		}

	@property
	def group_counts(self) -> dict[tuple[int, int, int], int]:
		"""Return observed dense counts keyed by XYZ coordinate."""
		if self.legacy_group_keys is not None:
			return {
				_group_key_tuple(key): int(count)
				for key, count in zip(
					self.legacy_group_keys,
					self.counts,
					strict=True,
				)
			}
		group_shape = _require_dense_group_shape(self.group_shape)
		return {
			tuple(
				int(value)
				for value in np.unravel_index(group_id, group_shape)
			): int(count)
			for group_id, count in enumerate(self.counts)
			if count > 0
		}

	def transform(
		self,
		embeddings: np.ndarray,
		groups: np.ndarray,
	) -> np.ndarray:
		"""Apply fitted group mean residualization to a feature matrix."""
		matrix = np.asarray(embeddings, dtype=np.float32)
		if matrix.ndim != 2:
			msg = f'embeddings must be a 2D matrix; got {matrix.shape!r}'
			raise ValueError(msg)
		if matrix.shape[1] != self.means.shape[1]:
			msg = (
				'embeddings feature dimension must match residualizer means; '
				f'got {matrix.shape[1]} and {self.means.shape[1]}'
			)
			raise ValueError(msg)
		if self.legacy_group_keys is not None:
			selected_means = _legacy_means_for_groups(
				groups,
				expected_count=matrix.shape[0],
				group_keys=self.legacy_group_keys,
				group_means=self.means,
				fallback_mean=self.fallback_mean,
			)
		else:
			group_shape = _require_dense_group_shape(self.group_shape)
			group_ids = _group_ids(
				groups,
				group_shape=group_shape,
				expected_count=matrix.shape[0],
			)
			selected_means = self.means[group_ids]
		transformed = matrix - selected_means
		if self.add_global_mean_back:
			transformed += self.fallback_mean
		return transformed.astype(np.float32, copy=False)

	def summary(self) -> dict[str, object]:
		"""Return compact JSON-safe metadata for fitted residualization."""
		observed_counts = self.counts[self.counts > 0]
		return {
			'enabled': True,
			'mode': self.mode,
			'group_by': self.group_by,
			'add_global_mean_back': self.add_global_mean_back,
			'min_group_count': self.min_group_count,
			'groups': int(observed_counts.size),
			'observed_groups': int(observed_counts.size),
			'group_shape': (
				list(self.group_shape) if self.group_shape is not None else None
			),
			'min_observed_group_count': (
				int(observed_counts.min()) if observed_counts.size else 0
			),
			'max_observed_group_count': (
				int(observed_counts.max()) if observed_counts.size else 0
			),
			'global_mean_l2_norm': float(np.linalg.norm(self.fallback_mean)),
		}


def fit_local_token_position_residualizer(  # noqa: PLR0913
	embeddings: np.ndarray,
	groups: np.ndarray,
	*,
	group_by: str,
	add_global_mean_back: bool,
	min_group_count: int,
	group_shape: tuple[int, int, int] | None = None,
	chunk_size: int = _DEFAULT_FIT_CHUNK_SIZE,
) -> LocalTokenPositionResidualizer:
	"""Fit group-wise mean residualization statistics."""
	if group_by not in SUPPORTED_RESIDUALIZATION_GROUP_BY:
		msg = f'unsupported residualization group_by: {group_by!r}'
		raise ValueError(msg)
	if min_group_count <= 0:
		msg = f'min_group_count must be positive; got {min_group_count!r}'
		raise ValueError(msg)
	if chunk_size <= 0:
		msg = f'chunk_size must be positive; got {chunk_size!r}'
		raise ValueError(msg)
	matrix = np.asarray(embeddings, dtype=np.float32)
	if matrix.ndim != 2:
		msg = f'embeddings must be a 2D matrix; got {matrix.shape!r}'
		raise ValueError(msg)
	resolved_shape = _resolve_group_shape(groups, group_shape, matrix.shape[0])
	group_ids = _group_ids(
		groups,
		group_shape=resolved_shape,
		expected_count=matrix.shape[0],
	)
	group_count = int(np.prod(resolved_shape))
	sums = np.zeros((group_count, matrix.shape[1]), dtype=np.float64)
	counts = np.zeros(group_count, dtype=np.int64)
	fallback_sum = np.zeros(matrix.shape[1], dtype=np.float64)
	for start in range(0, matrix.shape[0], chunk_size):
		stop = min(start + chunk_size, matrix.shape[0])
		chunk = matrix[start:stop]
		chunk_ids = group_ids[start:stop]
		np.add.at(sums, chunk_ids, chunk)
		counts += np.bincount(chunk_ids, minlength=group_count)
		fallback_sum += chunk.sum(axis=0, dtype=np.float64)
	if matrix.shape[0]:
		fallback_mean = (fallback_sum / matrix.shape[0]).astype(np.float32)
	else:
		fallback_mean = np.zeros(matrix.shape[1], dtype=np.float32)
	means = np.broadcast_to(fallback_mean, sums.shape).copy()
	trusted = counts >= min_group_count
	means[trusted] = sums[trusted] / counts[trusted, np.newaxis]
	return LocalTokenPositionResidualizer(
		mode='local_token_position',
		group_by=group_by,
		add_global_mean_back=add_global_mean_back,
		min_group_count=int(min_group_count),
		means=means.astype(np.float32, copy=False),
		counts=counts,
		group_shape=resolved_shape,
		fallback_mean=fallback_mean,
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


def token_phase_group_ids_for_grid(
	token_grid_shape_xyz: tuple[int, int, int],
	*,
	patch_size_xyz: tuple[int, int, int],
	window_size_xyz: tuple[int, int, int],
	overlap_xyz: tuple[int, int, int],
	valid_mask: np.ndarray | None = None,
) -> np.ndarray:
	"""Return dense token-phase IDs for a token grid."""
	group_shape = _stride_tokens(
		patch_size_xyz=patch_size_xyz,
		window_size_xyz=window_size_xyz,
		overlap_xyz=overlap_xyz,
	)
	keys = token_phase_keys_for_grid(
		token_grid_shape_xyz,
		patch_size_xyz=patch_size_xyz,
		window_size_xyz=window_size_xyz,
		overlap_xyz=overlap_xyz,
		valid_mask=valid_mask,
	)
	return encode_dense_group_ids(keys, group_shape)


def encode_dense_group_ids(
	group_keys: np.ndarray,
	group_shape: tuple[int, int, int],
) -> np.ndarray:
	"""Encode finite-grid XYZ group keys as dense row-major IDs."""
	shape = _positive_int_triplet(group_shape, 'group_shape')
	keys = np.asarray(group_keys)
	if keys.ndim != 2 or keys.shape[1] != 3:
		msg = f'group_keys must have shape [N, 3]; got {keys.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(keys.dtype, np.integer):
		msg = f'group_keys must have integer dtype; got {keys.dtype}'
		raise TypeError(msg)
	keys = keys.astype(np.int64, copy=False)
	if keys.size:
		shape_array = np.asarray(shape, dtype=np.int64)
		invalid = np.any((keys < 0) | (keys >= shape_array), axis=1)
		if np.any(invalid):
			first = tuple(int(value) for value in keys[np.flatnonzero(invalid)[0]])
			msg = f'group key {first!r} is outside group_shape {shape!r}'
			raise ValueError(msg)
	return np.ravel_multi_index(keys.T, shape).astype(np.int64, copy=False)


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


def residualization_group_ids_for_flat_indices(
	embedding_input: EmbeddingInput,
	token_indices: np.ndarray,
	*,
	group_by: str,
) -> np.ndarray:
	"""Return dense residualization IDs for flattened token indices."""
	indices = np.asarray(token_indices, dtype=np.int64)
	if indices.ndim != 1:
		msg = f'token_indices must be 1D; got {indices.shape!r}'
		raise ValueError(msg)
	metadata = load_embedding_metadata(embedding_input)
	if group_by == 'token_phase':
		return _token_phase_group_ids_for_flat_indices(metadata, indices)
	if group_by == 'local_token_position':
		msg = (
			'clustering.residualization.group_by=local_token_position requires '
			'exact per-token local position metadata, which is not present in '
			f'{embedding_input.metadata_path}'
		)
		raise ValueError(msg)
	msg = f'unsupported residualization group_by: {group_by!r}'
	raise ValueError(msg)


def residualization_groups_for_flat_indices(
	embedding_input: EmbeddingInput,
	token_indices: np.ndarray,
	*,
	residualizer: LocalTokenPositionResidualizer,
) -> np.ndarray:
	"""Return the group representation required by a residualizer artifact."""
	if residualizer.group_shape is None:
		return residualization_keys_for_flat_indices(
			embedding_input,
			token_indices,
			group_by=residualizer.group_by,
		)
	return residualization_group_ids_for_flat_indices(
		embedding_input,
		token_indices,
		group_by=residualizer.group_by,
	)


def residualization_group_shape(
	embedding_input: EmbeddingInput,
	*,
	group_by: str,
) -> tuple[int, int, int]:
	"""Return the finite dense group grid declared by embedding metadata."""
	if group_by == 'token_phase':
		metadata = load_embedding_metadata(embedding_input)
		return _stride_tokens(
			patch_size_xyz=_metadata_triplet(metadata, 'patch_size'),
			window_size_xyz=_metadata_triplet(metadata, 'window_size'),
			overlap_xyz=_metadata_triplet(metadata, 'overlap'),
		)
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


def sample_residualization_group_ids(
	embedding_inputs: tuple[EmbeddingInput, ...],
	per_survey_token_indices: Mapping[str, np.ndarray],
	*,
	group_by: str,
) -> tuple[np.ndarray, tuple[int, int, int]]:
	"""Return sampled dense group IDs and their shared finite grid shape."""
	if not embedding_inputs:
		msg = 'at least one embedding input is required'
		raise ValueError(msg)
	shapes = tuple(
		residualization_group_shape(item, group_by=group_by)
		for item in embedding_inputs
	)
	if any(shape != shapes[0] for shape in shapes[1:]):
		msg = f'residualization group shapes must match; got {shapes!r}'
		raise ValueError(msg)
	blocks = [
		residualization_group_ids_for_flat_indices(
			item,
			per_survey_token_indices[item.survey_id],
			group_by=group_by,
		)
		for item in embedding_inputs
		if per_survey_token_indices[item.survey_id].size
	]
	if not blocks:
		return np.empty(0, dtype=np.int64), shapes[0]
	return np.concatenate(blocks).astype(np.int64, copy=False), shapes[0]


def write_residualizer_npz(
	path: str | Path,
	residualizer: LocalTokenPositionResidualizer,
) -> None:
	"""Persist residualizer statistics in a reusable compact NPZ file."""
	if residualizer.group_shape is None:
		msg = 'legacy residualizers without a finite group shape cannot be rewritten'
		raise ValueError(msg)
	npz_path = Path(path)
	npz_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez(
		npz_path,
		schema_version=np.asarray(RESIDUALIZER_SCHEMA_VERSION, dtype=np.int64),
		means=np.asarray(residualizer.means, dtype=np.float32),
		counts=np.asarray(residualizer.counts, dtype=np.int64),
		group_shape=np.asarray(residualizer.group_shape, dtype=np.int64),
		fallback_mean=np.asarray(residualizer.fallback_mean, dtype=np.float32),
		mode=np.asarray(residualizer.mode),
		group_by=np.asarray(residualizer.group_by),
		add_global_mean_back=np.asarray(residualizer.add_global_mean_back),
		min_group_count=np.asarray(residualizer.min_group_count, dtype=np.int64),
	)


def read_residualizer_npz(path: str | Path) -> LocalTokenPositionResidualizer:
	"""Load residualizer statistics saved by write_residualizer_npz."""
	with np.load(path, allow_pickle=False) as payload:
		if 'schema_version' in payload.files:
			return _read_dense_residualizer(payload)
		return _read_legacy_residualizer(payload)


def residualization_metadata_disabled() -> dict[str, object]:
	"""Return explicit disabled residualization metadata."""
	return {'enabled': False}


def _read_dense_residualizer(
	payload: np.lib.npyio.NpzFile,
) -> LocalTokenPositionResidualizer:
	version = int(payload['schema_version'].item())
	if version != RESIDUALIZER_SCHEMA_VERSION:
		msg = f'unsupported residualizer schema_version: {version}'
		raise ValueError(msg)
	group_shape = _positive_int_triplet(
		tuple(int(value) for value in np.asarray(payload['group_shape']).tolist()),
		'group_shape',
	)
	means = np.asarray(payload['means'], dtype=np.float32)
	counts = np.asarray(payload['counts'], dtype=np.int64)
	fallback_mean = np.asarray(payload['fallback_mean'], dtype=np.float32)
	_validate_dense_statistics(means, counts, group_shape, fallback_mean)
	return LocalTokenPositionResidualizer(
		mode=str(payload['mode'].item()),
		group_by=str(payload['group_by'].item()),
		add_global_mean_back=bool(payload['add_global_mean_back'].item()),
		min_group_count=int(payload['min_group_count'].item()),
		means=means,
		counts=counts,
		group_shape=group_shape,
		fallback_mean=fallback_mean,
	)


def _read_legacy_residualizer(
	payload: np.lib.npyio.NpzFile,
) -> LocalTokenPositionResidualizer:
	group_keys = np.asarray(payload['group_keys'], dtype=np.int64)
	if group_keys.size == 0:
		group_keys = np.empty((0, 3), dtype=np.int64)
	group_means = np.asarray(payload['group_means'], dtype=np.float32)
	group_counts = np.asarray(payload['group_counts'], dtype=np.int64)
	fallback_mean = np.asarray(payload['global_mean'], dtype=np.float32)
	_validate_legacy_statistics(
		group_keys,
		group_means,
		group_counts,
		fallback_mean,
	)
	return LocalTokenPositionResidualizer(
		mode=str(payload['mode'].item()),
		group_by=str(payload['group_by'].item()),
		add_global_mean_back=bool(payload['add_global_mean_back'].item()),
		min_group_count=int(payload['min_group_count'].item()),
		means=group_means,
		counts=group_counts,
		group_shape=None,
		fallback_mean=fallback_mean,
		legacy_group_keys=group_keys,
	)


def _validate_legacy_statistics(
	group_keys: np.ndarray,
	group_means: np.ndarray,
	group_counts: np.ndarray,
	fallback_mean: np.ndarray,
) -> None:
	if group_keys.ndim != 2 or group_keys.shape[1] != 3:
		msg = f'legacy group_keys must have shape [N, 3]; got {group_keys.shape!r}'
		raise ValueError(msg)
	if np.any(group_keys < 0):
		msg = 'legacy group_keys must be nonnegative'
		raise ValueError(msg)
	if group_means.ndim != 2 or group_means.shape[0] != group_keys.shape[0]:
		msg = (
			'legacy group_means must have shape [N, D] matching group_keys; '
			f'got {group_means.shape!r} and {group_keys.shape!r}'
		)
		raise ValueError(msg)
	if group_counts.shape != (group_keys.shape[0],):
		msg = (
			'legacy group_counts must have shape '
			f'{(group_keys.shape[0],)!r}; got {group_counts.shape!r}'
		)
		raise ValueError(msg)
	if fallback_mean.shape != (group_means.shape[1],):
		msg = (
			'legacy global_mean must match the feature dimension; '
			f'got {fallback_mean.shape!r} and {group_means.shape!r}'
		)
		raise ValueError(msg)
	if np.any(group_counts < 0):
		msg = 'legacy group_counts must be nonnegative'
		raise ValueError(msg)


def _legacy_means_for_groups(
	groups: np.ndarray,
	*,
	expected_count: int,
	group_keys: np.ndarray,
	group_means: np.ndarray,
	fallback_mean: np.ndarray,
) -> np.ndarray:
	keys = np.asarray(groups)
	if keys.shape != (expected_count, 3):
		msg = (
			'legacy residualizers require coordinate group keys with shape '
			f'{(expected_count, 3)!r}; got {keys.shape!r}'
		)
		raise ValueError(msg)
	if not np.issubdtype(keys.dtype, np.integer):
		msg = f'group keys must have integer dtype; got {keys.dtype}'
		raise TypeError(msg)
	keys = keys.astype(np.int64, copy=False)
	if np.any(keys < 0):
		msg = 'group keys must be nonnegative'
		raise ValueError(msg)
	lookup = {
		_group_key_tuple(key): index for index, key in enumerate(group_keys)
	}
	selected = np.broadcast_to(
		fallback_mean,
		(expected_count, fallback_mean.size),
	).copy()
	for row, key in enumerate(keys):
		group_index = lookup.get(_group_key_tuple(key))
		if group_index is not None:
			selected[row] = group_means[group_index]
	return selected


def _group_key_tuple(key: np.ndarray) -> tuple[int, int, int]:
	return cast('tuple[int, int, int]', tuple(int(value) for value in key))


def _require_dense_group_shape(
	group_shape: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
	if group_shape is None:
		msg = 'dense residualizer is missing group_shape'
		raise ValueError(msg)
	return group_shape


def _validate_dense_statistics(
	means: np.ndarray,
	counts: np.ndarray,
	group_shape: tuple[int, int, int],
	fallback_mean: np.ndarray,
) -> None:
	group_count = int(np.prod(group_shape))
	if means.ndim != 2 or means.shape[0] != group_count:
		msg = f'means must have shape [{group_count}, D]; got {means.shape!r}'
		raise ValueError(msg)
	if counts.shape != (group_count,):
		msg = f'counts must have shape {(group_count,)!r}; got {counts.shape!r}'
		raise ValueError(msg)
	if fallback_mean.shape != (means.shape[1],):
		msg = (
			f'fallback_mean must have shape {(means.shape[1],)!r}; '
			f'got {fallback_mean.shape!r}'
		)
		raise ValueError(msg)
	if np.any(counts < 0):
		msg = 'counts must be nonnegative'
		raise ValueError(msg)


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


def _token_phase_group_ids_for_flat_indices(
	metadata: Mapping[str, object],
	indices: np.ndarray,
) -> np.ndarray:
	shape = _metadata_triplet(metadata, 'token_grid_shape')
	group_shape = _stride_tokens(
		patch_size_xyz=_metadata_triplet(metadata, 'patch_size'),
		window_size_xyz=_metadata_triplet(metadata, 'window_size'),
		overlap_xyz=_metadata_triplet(metadata, 'overlap'),
	)
	x_coord, y_coord, z_coord = np.unravel_index(indices, shape)
	return (
		((x_coord % group_shape[0]) * group_shape[1] + y_coord % group_shape[1])
		* group_shape[2]
		+ z_coord % group_shape[2]
	).astype(np.int64, copy=False)


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


def _resolve_group_shape(
	groups: np.ndarray,
	group_shape: tuple[int, int, int] | None,
	expected_count: int,
) -> tuple[int, int, int]:
	if group_shape is not None:
		return _positive_int_triplet(group_shape, 'group_shape')
	values = np.asarray(groups)
	if values.shape != (expected_count, 3):
		msg = 'group_shape is required when groups are dense IDs'
		raise ValueError(msg)
	if expected_count == 0:
		msg = 'group_shape is required when fitting empty groups'
		raise ValueError(msg)
	if not np.issubdtype(values.dtype, np.integer):
		msg = f'group keys must have integer dtype; got {values.dtype}'
		raise TypeError(msg)
	if np.any(values < 0):
		msg = 'group keys must be nonnegative'
		raise ValueError(msg)
	return tuple(int(value) + 1 for value in values.max(axis=0))


def _group_ids(
	groups: np.ndarray,
	*,
	group_shape: tuple[int, int, int],
	expected_count: int,
) -> np.ndarray:
	values = np.asarray(groups)
	if values.ndim == 2:
		if values.shape != (expected_count, 3):
			msg = (
				f'group keys must have shape {(expected_count, 3)!r}; '
				f'got {values.shape!r}'
			)
			raise ValueError(msg)
		return encode_dense_group_ids(values, group_shape)
	if values.shape != (expected_count,):
		msg = f'group IDs must have shape {(expected_count,)!r}; got {values.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(values.dtype, np.integer):
		msg = f'group IDs must have integer dtype; got {values.dtype}'
		raise TypeError(msg)
	values = values.astype(np.int64, copy=False)
	group_count = int(np.prod(group_shape))
	if values.size and (np.any(values < 0) or np.any(values >= group_count)):
		msg = f'group IDs must be in [0, {group_count}); got out-of-range values'
		raise ValueError(msg)
	return values


__all__ = [
	'RESIDUALIZER_SCHEMA_VERSION',
	'SUPPORTED_RESIDUALIZATION_GROUP_BY',
	'SUPPORTED_RESIDUALIZATION_MODES',
	'LocalTokenPositionResidualizer',
	'encode_dense_group_ids',
	'fit_local_token_position_residualizer',
	'read_residualizer_npz',
	'residualization_group_ids_for_flat_indices',
	'residualization_group_shape',
	'residualization_groups_for_flat_indices',
	'residualization_keys_for_flat_indices',
	'residualization_metadata_disabled',
	'sample_residualization_group_ids',
	'sample_residualization_keys',
	'token_phase_group_ids_for_grid',
	'token_phase_keys_for_grid',
	'write_residualizer_npz',
]
