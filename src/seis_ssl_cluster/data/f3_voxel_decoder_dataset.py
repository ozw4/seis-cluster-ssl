"""Memory-mapped deterministic tiles for the F3 voxel decoder."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	ALL_TILES_ONCE,
	UNIFORM_TILES_WITH_REPLACEMENT,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_tiles import (
	VoxelTileManifest,
	read_voxel_tile_manifest,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.f3.lithology.voxel_tiles import VoxelTileRecord


class F3VoxelDecoderDataset(Dataset[dict[str, Any]]):
	"""Expose fixed-size, halo-padded crops without random sampling."""

	def __init__(  # noqa: PLR0913
		self,
		embedding_path: str | Path,
		valid_tokens_path: str | Path,
		embedding_metadata_path: str | Path,
		label_volume_path: str | Path,
		supervision_split_grid_path: str | Path,
		tile_manifest: VoxelTileManifest | str | Path,
		*,
		supervision_metadata_path: str | Path,
		canonical_valid_tokens_path: str | Path | None = None,
	) -> None:
		"""Open and validate all artifacts, retaining only memory maps."""
		super().__init__()
		self.embedding_path = Path(embedding_path)
		self.valid_tokens_path = Path(valid_tokens_path)
		self.embedding_metadata_path = Path(embedding_metadata_path)
		self.label_volume_path = Path(label_volume_path)
		self.supervision_split_grid_path = Path(supervision_split_grid_path)
		self.supervision_metadata_path = Path(supervision_metadata_path)
		self.canonical_valid_tokens_path = (
			None
			if canonical_valid_tokens_path is None
			else Path(canonical_valid_tokens_path)
		)
		self.manifest = (
			tile_manifest
			if isinstance(tile_manifest, VoxelTileManifest)
			else read_voxel_tile_manifest(tile_manifest)
		)
		for path in self._input_paths():
			if not path.is_file():
				raise FileNotFoundError(f'missing voxel decoder dataset input: {path}')
		self._embeddings: np.ndarray | None = None
		self._valid_tokens: np.ndarray | None = None
		self._labels: np.ndarray | None = None
		self._split_grid: np.ndarray | None = None
		self._open_arrays()
		with self.embedding_metadata_path.open(encoding='utf-8') as file_obj:
			metadata = json.load(file_obj)
		if not isinstance(metadata, Mapping):
			raise TypeError('embedding metadata must contain an object')
		self.embedding_metadata = dict(metadata)
		self._validate_contract()

	def __len__(self) -> int:
		"""Return the number of non-empty deterministic core tiles."""
		return len(self.manifest.tiles)

	def __getitem__(self, index: int) -> dict[str, Any]:
		"""Load and pad one tile crop."""
		self._open_arrays()
		tile = self.manifest.tiles[index]
		embeddings = self._require_array(self._embeddings)
		valid_tokens = self._require_array(self._valid_tokens)
		labels = self._require_array(self._labels)
		split_grid = self._require_array(self._split_grid)
		patch = self.manifest.patch_size_xyz
		input_tokens = tuple(
			self.manifest.core_size_tokens[axis]
			+ 2 * self.manifest.context_halo_tokens[axis]
			for axis in range(3)
		)
		input_voxels = tuple(input_tokens[axis] * patch[axis] for axis in range(3))

		token_source = _slices(tile.input_start_token_xyz, tile.input_stop_token_xyz)
		token_destination_start = tile.input_padding_before_xyz
		token_destination_stop = tuple(
			token_destination_start[axis]
			+ tile.input_stop_token_xyz[axis]
			- tile.input_start_token_xyz[axis]
			for axis in range(3)
		)
		token_destination = _slices(token_destination_start, token_destination_stop)
		embedding_crop = np.zeros(
			(*input_tokens, embeddings.shape[3]), dtype=np.float32
		)
		embedding_crop[token_destination] = np.asarray(
			embeddings[token_source], dtype=np.float32
		)
		token_mask = np.zeros(input_tokens, dtype=np.bool_)
		token_mask[token_destination] = valid_tokens[token_source]

		voxel_source_start = tuple(
			tile.input_start_token_xyz[axis] * patch[axis] for axis in range(3)
		)
		voxel_source_stop = tuple(
			min(
				tile.input_stop_token_xyz[axis] * patch[axis],
				self.manifest.volume_shape_xyz[axis],
			)
			for axis in range(3)
		)
		voxel_source = _slices(voxel_source_start, voxel_source_stop)
		voxel_destination_start = tuple(
			tile.input_padding_before_xyz[axis] * patch[axis] for axis in range(3)
		)
		voxel_destination_stop = tuple(
			voxel_destination_start[axis]
			+ voxel_source_stop[axis]
			- voxel_source_start[axis]
			for axis in range(3)
		)
		voxel_destination = _slices(voxel_destination_start, voxel_destination_stop)
		label_crop = np.full(input_voxels, -1, dtype=np.int64)
		source_labels = np.asarray(labels[voxel_source], dtype=np.int64)
		known = np.isin(source_labels, self.manifest.class_ids)
		label_crop[voxel_destination] = np.where(known, source_labels, -1)

		core_mask = np.zeros(input_voxels, dtype=np.bool_)
		core_destination_start = tuple(
			self.manifest.context_halo_tokens[axis] * patch[axis] for axis in range(3)
		)
		core_destination_stop = tuple(
			core_destination_start[axis]
			+ tile.core_voxel_stop_xyz[axis]
			- tile.core_voxel_start_xyz[axis]
			for axis in range(3)
		)
		core_mask[_slices(core_destination_start, core_destination_stop)] = True

		valid_voxels = token_mask
		for axis, repeats in enumerate(patch):
			valid_voxels = np.repeat(valid_voxels, repeats, axis=axis)
		split_mask = np.zeros(input_voxels, dtype=np.bool_)
		split_mask[voxel_destination] = split_grid[voxel_source] == _split_code(
			self.manifest.split
		)
		supervision_mask = core_mask & valid_voxels & split_mask & (label_crop >= 0)
		if int(np.count_nonzero(supervision_mask)) != tile.supervised_voxel_count:
			raise RuntimeError(
				f'tile {tile.tile_id} supervision count no longer matches manifest'
			)

		return {
			'embeddings': torch.from_numpy(
				np.ascontiguousarray(np.moveaxis(embedding_crop, -1, 0))
			),
			'token_valid_mask': torch.from_numpy(np.ascontiguousarray(token_mask)),
			'labels': torch.from_numpy(label_crop),
			'supervision_mask': torch.from_numpy(supervision_mask),
			'core_mask': torch.from_numpy(core_mask),
			'tile_id': tile.tile_id,
			'geometry': _tile_geometry(tile),
		}

	def __getstate__(self) -> dict[str, object]:
		"""Drop process-owned memory maps before DataLoader worker pickling."""
		state = self.__dict__.copy()
		for key in ('_embeddings', '_valid_tokens', '_labels', '_split_grid'):
			state[key] = None
		return state

	def _input_paths(self) -> tuple[Path, ...]:
		paths = (
			self.embedding_path,
			self.valid_tokens_path,
			self.embedding_metadata_path,
			self.label_volume_path,
			self.supervision_split_grid_path,
			self.supervision_metadata_path,
		)
		extra = tuple(
			path for path in (self.canonical_valid_tokens_path,) if path is not None
		)
		return (*paths, *extra)

	def _open_arrays(self) -> None:
		if self._embeddings is None:
			self._embeddings = np.load(
				self.embedding_path, mmap_mode='r', allow_pickle=False
			)
			self._valid_tokens = np.load(
				self.valid_tokens_path, mmap_mode='r', allow_pickle=False
			)
			self._labels = np.load(
				self.label_volume_path, mmap_mode='r', allow_pickle=False
			)
			self._split_grid = np.load(
				self.supervision_split_grid_path, mmap_mode='r', allow_pickle=False
			)

	def _validate_contract(self) -> None:
		embeddings = self._require_array(self._embeddings)
		valid_tokens = self._require_array(self._valid_tokens)
		labels = self._require_array(self._labels)
		split_grid = self._require_array(self._split_grid)
		if embeddings.ndim != 4 or not np.issubdtype(embeddings.dtype, np.floating):
			raise ValueError('embeddings must be floating [TX,TY,TZ,D]')
		if valid_tokens.dtype != np.bool_ or valid_tokens.ndim != 3:
			raise TypeError('valid_tokens must be a 3D bool array')
		if labels.ndim != 3 or split_grid.shape != labels.shape:
			raise ValueError('label and supervision arrays must be matching 3D arrays')
		if tuple(embeddings.shape[:3]) != self.manifest.token_grid_shape_xyz:
			raise ValueError('embedding token grid does not match tile manifest')
		if valid_tokens.shape != embeddings.shape[:3]:
			raise ValueError('valid_tokens shape does not match embedding token grid')
		if tuple(labels.shape) != self.manifest.volume_shape_xyz:
			raise ValueError('label volume shape does not match tile manifest')
		_validate_candidate_metadata(
			self.embedding_metadata,
			embedding_shape=embeddings.shape,
			manifest=self.manifest,
		)
		with self.supervision_metadata_path.open(encoding='utf-8') as file_obj:
			reference = json.load(file_obj)
		if not isinstance(reference, Mapping):
			raise TypeError('supervision metadata must contain an object')
		validate_encoder_pairing(
			candidate_metadata=self.embedding_metadata,
			reference_metadata=reference,
			candidate_valid_tokens_path=self.valid_tokens_path,
			candidate_embedding_shape=embeddings.shape,
		)
		if self.canonical_valid_tokens_path is not None:
			canonical = np.load(
				self.canonical_valid_tokens_path, mmap_mode='r', allow_pickle=False
			)
			if not np.array_equal(valid_tokens, canonical):
				raise ValueError('candidate valid_tokens do not match canonical mask')

	@staticmethod
	def _require_array(value: np.ndarray | None) -> np.ndarray:
		if value is None:
			raise RuntimeError('dataset arrays are not open')
		return value


def validate_encoder_pairing(  # noqa: C901, PLR0912, PLR0913
	*,
	candidate_metadata: Mapping[str, object],
	reference_metadata: Mapping[str, object],
	candidate_valid_tokens_path: str | Path,
	candidate_embedding_shape: Sequence[int],
	reference_valid_tokens_path: str | Path | None = None,
	verify_valid_token_hash: bool = True,
) -> None:
	"""Reject any candidate that could silently change decoder evaluation geometry."""
	reference_embedding = reference_metadata.get('reference_embedding')
	if isinstance(reference_embedding, Mapping):
		reference_details = reference_embedding.get('metadata')
		if not isinstance(reference_details, Mapping):
			reference_details = reference_embedding
	else:
		reference_details = reference_metadata
	for key in ('volume_shape_xyz', 'patch_size', 'token_grid_shape'):
		if candidate_metadata.get(key) != reference_details.get(key):
			raise ValueError(f'encoder pairing mismatch for {key}')
	for key in ('preprocessing', 'zero_mask'):
		if key not in candidate_metadata or key not in reference_details:
			raise ValueError(f'encoder pairing mismatch for {key}')
		if key == 'preprocessing':
			if not _preprocessing_pairing_matches(
				candidate_metadata[key], reference_details[key]
			):
				raise ValueError(f'encoder pairing mismatch for {key}')
		elif candidate_metadata[key] != reference_details[key]:
			raise ValueError(f'encoder pairing mismatch for {key}')
	candidate_dim = int(candidate_embedding_shape[3])
	reference_dim = _metadata_embedding_dim(reference_details)
	if reference_dim is None or candidate_dim != reference_dim:
		raise ValueError('encoder pairing mismatch for embedding dim')
	identity = reference_metadata.get('reference_valid_tokens')
	expected_sha = identity.get('sha256') if isinstance(identity, Mapping) else None
	if not isinstance(expected_sha, str) or not expected_sha:
		raise ValueError(
			'supervision metadata must contain reference_valid_tokens.sha256'
		)
	if (
		verify_valid_token_hash
		and file_sha256(candidate_valid_tokens_path) != expected_sha
	):
		raise ValueError('candidate valid-token hash does not match canonical hash')
	if reference_valid_tokens_path is not None:
		candidate = np.load(
			candidate_valid_tokens_path, mmap_mode='r', allow_pickle=False
		)
		reference = np.load(
			reference_valid_tokens_path, mmap_mode='r', allow_pickle=False
		)
		if not np.array_equal(candidate, reference):
			raise ValueError('candidate valid_tokens are not bitwise identical')


def _preprocessing_pairing_matches(
	candidate: object, reference: object
) -> bool:
	"""Compare input-transform identity while allowing finite-check migration.

	``finite_check_mode`` determines validation timing only; it does not alter
	the amplitude transform, valid-token geometry, or decoder inputs.  Legacy
	embedding metadata pre-dates this field and therefore represents ``off``.
	A current ``strict`` extraction may pair with that legacy representation,
	but every actual preprocessing field remains exact and no other finite mode
	may silently cross a representation boundary.
	"""
	if not isinstance(candidate, Mapping) or not isinstance(reference, Mapping):
		return False
	candidate_values = dict(candidate)
	reference_values = dict(reference)
	candidate_finite = candidate_values.pop('finite_check_mode', 'off')
	reference_finite = reference_values.pop('finite_check_mode', 'off')
	if candidate_values != reference_values:
		return False
	if (
		not isinstance(candidate_finite, str)
		or not isinstance(reference_finite, str)
		or candidate_finite not in {'strict', 'output_only', 'off'}
		or reference_finite not in {'strict', 'output_only', 'off'}
	):
		return False
	return candidate_finite == reference_finite or {
		candidate_finite,
		reference_finite,
	} <= {'strict', 'off'}


def build_f3_voxel_decoder_dataloader(  # noqa: PLR0913
	dataset: F3VoxelDecoderDataset,
	*,
	batch_size: int,
	shuffle: bool,
	seed: int,
	num_workers: int = 0,
	sampling_mode: str = ALL_TILES_ONCE,
	steps_per_epoch: int | None = None,
	**kwargs: object,
) -> DataLoader[dict[str, Any]]:
	"""Build a deterministic all-tiles or replacement-sampled DataLoader."""
	for value, name, allow_zero in (
		(batch_size, 'batch_size', False),
		(seed, 'seed', True),
		(num_workers, 'num_workers', True),
	):
		if (
			not isinstance(value, Integral)
			or isinstance(value, bool)
			or value < int(not allow_zero)
		):
			qualifier = 'non-negative' if allow_zero else 'positive'
			raise ValueError(f'{name} must be a {qualifier} integer')
	if sampling_mode == ALL_TILES_ONCE:
		if steps_per_epoch is not None:
			raise ValueError('steps_per_epoch must be null for all_tiles_once sampling')
		# Keep the legacy generator coupling intact: DataLoader consumes this
		# generator for its worker base seed before RandomSampler draws the order.
		generator = torch.Generator()
		generator.manual_seed(int(seed))
		return DataLoader(
			dataset,
			batch_size=int(batch_size),
			shuffle=shuffle,
			num_workers=int(num_workers),
			generator=generator,
			**kwargs,
		)
	if sampling_mode != UNIFORM_TILES_WITH_REPLACEMENT:
		raise ValueError(
			'sampling_mode must be all_tiles_once or uniform_tiles_with_replacement'
		)
	if (
		not isinstance(steps_per_epoch, Integral)
		or isinstance(steps_per_epoch, bool)
		or steps_per_epoch <= 0
	):
		raise ValueError(
			'steps_per_epoch must be a positive integer for replacement sampling'
		)
	if not shuffle:
		raise ValueError('replacement sampling requires shuffle=True')

	# The sampler and DataLoader worker generators are deliberately separate.
	# This makes the first sampled tile a direct function of ``seed`` instead of
	# also depending on DataLoader's internal worker-base-seed draw.
	sampler_generator = torch.Generator()
	sampler_generator.manual_seed(int(seed))
	worker_generator = torch.Generator()
	worker_generator.manual_seed(int(seed))
	sampler = RandomSampler(
		dataset,
		replacement=True,
		num_samples=int(steps_per_epoch) * int(batch_size),
		generator=sampler_generator,
	)
	return DataLoader(
		dataset,
		batch_size=int(batch_size),
		sampler=sampler,
		num_workers=int(num_workers),
		generator=worker_generator,
		**kwargs,
	)


def _validate_candidate_metadata(
	metadata: Mapping[str, object],
	*,
	embedding_shape: Sequence[int],
	manifest: VoxelTileManifest,
) -> None:
	expected = {
		'volume_shape_xyz': list(manifest.volume_shape_xyz),
		'patch_size': list(manifest.patch_size_xyz),
		'token_grid_shape': list(manifest.token_grid_shape_xyz),
	}
	for key, value in expected.items():
		if metadata.get(key) != value:
			raise ValueError(f'embedding metadata {key} does not match tile manifest')
	dimension = _metadata_embedding_dim(metadata)
	if dimension is not None and dimension != int(embedding_shape[3]):
		raise ValueError('embedding metadata dimension does not match embedding array')


def _metadata_embedding_dim(metadata: Mapping[str, object]) -> int | None:
	value = metadata.get('embedding_dim')
	geometry = metadata.get('model_geometry')
	if value is None and isinstance(geometry, Mapping):
		value = geometry.get('encoder_dim')
	if value is None:
		return None
	if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
		raise ValueError('embedding metadata dimension must be a positive integer')
	return int(value)


def _tile_geometry(tile: VoxelTileRecord) -> dict[str, object]:
	return {
		'core_start_token_xyz': tile.core_start_token_xyz,
		'core_stop_token_xyz': tile.core_stop_token_xyz,
		'input_start_token_xyz': tile.input_start_token_xyz,
		'input_stop_token_xyz': tile.input_stop_token_xyz,
		'input_padding_before_xyz': tile.input_padding_before_xyz,
		'input_padding_after_xyz': tile.input_padding_after_xyz,
		'core_voxel_start_xyz': tile.core_voxel_start_xyz,
		'core_voxel_stop_xyz': tile.core_voxel_stop_xyz,
	}


def _split_code(split: str) -> int:
	if split == 'train':
		return 1
	if split == 'validation':
		return 2
	return -1


def _slices(start: Sequence[int], stop: Sequence[int]) -> tuple[slice, slice, slice]:
	return tuple(slice(a, b) for a, b in zip(start, stop, strict=True))  # type: ignore[return-value]


__all__ = [
	'F3VoxelDecoderDataset',
	'build_f3_voxel_decoder_dataloader',
	'validate_encoder_pairing',
]
