"""Deterministic core-tile manifests for voxel-decoder supervision."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from seis_ssl_cluster.f3.lithology.voxel_split import (
	TRAIN_VOXEL_SPLIT,
	VALIDATION_VOXEL_SPLIT,
)

if TYPE_CHECKING:
	from collections.abc import Mapping


_SPLIT_CODES = {
	'train': int(TRAIN_VOXEL_SPLIT),
	'validation': int(VALIDATION_VOXEL_SPLIT),
}


@dataclass(frozen=True)
class VoxelTileRecord:
	"""Geometry and supervision counts for one deterministic core tile."""

	tile_id: str
	split: str
	core_start_token_xyz: tuple[int, int, int]
	core_stop_token_xyz: tuple[int, int, int]
	input_start_token_xyz: tuple[int, int, int]
	input_stop_token_xyz: tuple[int, int, int]
	input_padding_before_xyz: tuple[int, int, int]
	input_padding_after_xyz: tuple[int, int, int]
	core_voxel_start_xyz: tuple[int, int, int]
	core_voxel_stop_xyz: tuple[int, int, int]
	supervised_voxel_count: int
	per_class_supervised_counts: dict[str, int]

	def to_dict(self) -> dict[str, object]:
		"""Return the JSON representation of this record."""
		return asdict(self)


@dataclass(frozen=True)
class VoxelTileManifest:
	"""Encoder-independent tile geometry for one supervision split."""

	split: str
	volume_shape_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	patch_size_xyz: tuple[int, int, int]
	core_size_tokens: tuple[int, int, int]
	context_halo_tokens: tuple[int, int, int]
	class_ids: tuple[int, ...]
	tiles: tuple[VoxelTileRecord, ...]
	schema_version: int = 1

	@property
	def identity_sha256(self) -> str:
		"""Return the SHA-256 identity of the canonical JSON payload."""
		return voxel_tile_manifest_sha256(self)

	def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
		"""Return a stable, JSON-compatible manifest mapping."""
		payload: dict[str, object] = {
			'artifact_type': 'f3_voxel_tile_manifest',
			'schema_version': self.schema_version,
			'split': self.split,
			'volume_shape_xyz': list(self.volume_shape_xyz),
			'token_grid_shape_xyz': list(self.token_grid_shape_xyz),
			'patch_size_xyz': list(self.patch_size_xyz),
			'core_size_tokens': list(self.core_size_tokens),
			'context_halo_tokens': list(self.context_halo_tokens),
			'class_ids': list(self.class_ids),
			'tile_count': len(self.tiles),
			'supervised_voxel_count': sum(
				tile.supervised_voxel_count for tile in self.tiles
			),
			'tiles': [tile.to_dict() for tile in self.tiles],
		}
		if include_identity:
			payload['identity_sha256'] = voxel_tile_manifest_sha256(self)
		return payload


def build_voxel_tile_manifest(  # noqa: C901, PLR0913
	supervision_split_grid: np.ndarray,
	label_volume: np.ndarray,
	*,
	split: str,
	patch_size_xyz: Sequence[int],
	token_grid_shape_xyz: Sequence[int] | None = None,
	core_size_tokens: Sequence[int] = (8, 8, 8),
	context_halo_tokens: Sequence[int] = (1, 1, 1),
	class_ids: Sequence[int],
) -> VoxelTileManifest:
	"""Partition a token grid and retain cores containing requested supervision."""
	if split not in _SPLIT_CODES:
		raise ValueError(f'split must be train or validation; got {split!r}')
	grid = np.asarray(supervision_split_grid)
	labels = np.asarray(label_volume)
	if grid.ndim != 3 or labels.ndim != 3 or grid.shape != labels.shape:
		raise ValueError(
			'supervision_split_grid and label_volume must be matching 3D arrays'
		)
	if not np.issubdtype(grid.dtype, np.integer) or grid.dtype == np.bool_:
		raise TypeError('supervision_split_grid dtype must be integer')
	if not np.issubdtype(labels.dtype, np.integer) or labels.dtype == np.bool_:
		raise TypeError('label_volume dtype must be integer')
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	core_size = _positive_triplet(core_size_tokens, 'core_size_tokens')
	halo = _nonnegative_triplet(context_halo_tokens, 'context_halo_tokens')
	expected_tokens = tuple(
		(size + step - 1) // step for size, step in zip(grid.shape, patch, strict=True)
	)
	token_shape = (
		expected_tokens
		if token_grid_shape_xyz is None
		else _positive_triplet(token_grid_shape_xyz, 'token_grid_shape_xyz')
	)
	if token_shape != expected_tokens:
		raise ValueError(
			'token_grid_shape_xyz is inconsistent with volume and patch size; '
			f'expected {expected_tokens!r}, got {token_shape!r}'
		)
	known_ids = _class_ids(class_ids)
	requested = (grid == _SPLIT_CODES[split]) & np.isin(labels, known_ids)

	tiles: list[VoxelTileRecord] = []
	covered = np.zeros(grid.shape, dtype=np.uint8)
	for x in range(0, token_shape[0], core_size[0]):
		for y in range(0, token_shape[1], core_size[1]):
			for z in range(0, token_shape[2], core_size[2]):
				start = (x, y, z)
				stop = tuple(
					min(start[axis] + core_size[axis], token_shape[axis])
					for axis in range(3)
				)
				voxel_start = tuple(start[axis] * patch[axis] for axis in range(3))
				voxel_stop = tuple(
					min(stop[axis] * patch[axis], grid.shape[axis]) for axis in range(3)
				)
				voxel_slices = _slices(voxel_start, voxel_stop)
				core_supervision = requested[voxel_slices]
				count = int(np.count_nonzero(core_supervision))
				if count == 0:
					continue
				input_start = tuple(max(0, start[a] - halo[a]) for a in range(3))
				desired_stop = tuple(
					start[a] + core_size[a] + halo[a] for a in range(3)
				)
				input_stop = tuple(
					min(token_shape[a], desired_stop[a]) for a in range(3)
				)
				pad_before = tuple(
					input_start[a] - (start[a] - halo[a]) for a in range(3)
				)
				pad_after = tuple(desired_stop[a] - input_stop[a] for a in range(3))
				counts = {
					str(class_id): int(
						np.count_nonzero(
							core_supervision & (labels[voxel_slices] == class_id)
						)
					)
					for class_id in known_ids
				}
				tile = VoxelTileRecord(
					tile_id=f'{split}_{len(tiles):06d}',
					split=split,
					core_start_token_xyz=start,
					core_stop_token_xyz=stop,
					input_start_token_xyz=input_start,
					input_stop_token_xyz=input_stop,
					input_padding_before_xyz=pad_before,
					input_padding_after_xyz=pad_after,
					core_voxel_start_xyz=voxel_start,
					core_voxel_stop_xyz=voxel_stop,
					supervised_voxel_count=count,
					per_class_supervised_counts=counts,
				)
				tiles.append(tile)
				covered[voxel_slices] += core_supervision.astype(np.uint8)

	if not np.array_equal(covered, requested.astype(np.uint8)):
		raise AssertionError('tile cores must cover every requested voxel exactly once')
	return VoxelTileManifest(
		split=split,
		volume_shape_xyz=tuple(int(item) for item in grid.shape),
		token_grid_shape_xyz=token_shape,
		patch_size_xyz=patch,
		core_size_tokens=core_size,
		context_halo_tokens=halo,
		class_ids=known_ids,
		tiles=tuple(tiles),
	)


def build_voxel_tile_manifests(  # noqa: PLR0913
	supervision_split_grid: np.ndarray,
	label_volume: np.ndarray,
	*,
	patch_size_xyz: Sequence[int],
	token_grid_shape_xyz: Sequence[int] | None = None,
	core_size_tokens: Sequence[int] = (8, 8, 8),
	context_halo_tokens: Sequence[int] = (1, 1, 1),
	class_ids: Sequence[int],
) -> dict[str, VoxelTileManifest]:
	"""Build independent train and validation manifests from common geometry."""
	return {
		split: build_voxel_tile_manifest(
			supervision_split_grid,
			label_volume,
			split=split,
			patch_size_xyz=patch_size_xyz,
			token_grid_shape_xyz=token_grid_shape_xyz,
			core_size_tokens=core_size_tokens,
			context_halo_tokens=context_halo_tokens,
			class_ids=class_ids,
		)
		for split in ('train', 'validation')
	}


def voxel_tile_manifest_sha256(manifest: VoxelTileManifest) -> str:
	"""Hash canonical UTF-8 JSON without embedding the hash into itself."""
	payload = json.dumps(
		manifest.to_dict(include_identity=False),
		sort_keys=True,
		separators=(',', ':'),
		ensure_ascii=False,
	).encode()
	return hashlib.sha256(payload).hexdigest()


def write_voxel_tile_manifest(path: str | Path, manifest: VoxelTileManifest) -> None:
	"""Write a manifest and its canonical identity as formatted JSON."""
	Path(path).write_text(
		json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def read_voxel_tile_manifest(path: str | Path) -> VoxelTileManifest:
	"""Read a manifest, validating its schema and canonical identity."""
	with Path(path).open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if not isinstance(payload, dict):
		raise TypeError('voxel tile manifest must contain an object')
	tiles_value = payload.get('tiles')
	if not isinstance(tiles_value, list):
		raise TypeError('voxel tile manifest tiles must be a list')
	tiles = tuple(_record_from_mapping(item) for item in tiles_value)
	manifest = VoxelTileManifest(
		split=_required_str(payload, 'split'),
		volume_shape_xyz=_payload_triplet(payload, 'volume_shape_xyz'),
		token_grid_shape_xyz=_payload_triplet(payload, 'token_grid_shape_xyz'),
		patch_size_xyz=_payload_triplet(payload, 'patch_size_xyz'),
		core_size_tokens=_payload_triplet(payload, 'core_size_tokens'),
		context_halo_tokens=_payload_triplet(
			payload, 'context_halo_tokens', positive=False
		),
		class_ids=_class_ids(payload.get('class_ids', ())),
		tiles=tiles,
		schema_version=int(payload.get('schema_version', 0)),
	)
	identity = payload.get('identity_sha256')
	if identity is not None and identity != manifest.identity_sha256:
		raise ValueError('voxel tile manifest identity_sha256 mismatch')
	return manifest


def _record_from_mapping(value: object) -> VoxelTileRecord:
	if not isinstance(value, dict):
		raise TypeError('voxel tile record must contain an object')
	counts = value.get('per_class_supervised_counts')
	if not isinstance(counts, dict):
		raise TypeError('per_class_supervised_counts must contain an object')
	return VoxelTileRecord(
		tile_id=_required_str(value, 'tile_id'),
		split=_required_str(value, 'split'),
		core_start_token_xyz=_payload_triplet(
			value, 'core_start_token_xyz', positive=False
		),
		core_stop_token_xyz=_payload_triplet(value, 'core_stop_token_xyz'),
		input_start_token_xyz=_payload_triplet(
			value, 'input_start_token_xyz', positive=False
		),
		input_stop_token_xyz=_payload_triplet(value, 'input_stop_token_xyz'),
		input_padding_before_xyz=_payload_triplet(
			value, 'input_padding_before_xyz', positive=False
		),
		input_padding_after_xyz=_payload_triplet(
			value, 'input_padding_after_xyz', positive=False
		),
		core_voxel_start_xyz=_payload_triplet(
			value, 'core_voxel_start_xyz', positive=False
		),
		core_voxel_stop_xyz=_payload_triplet(value, 'core_voxel_stop_xyz'),
		supervised_voxel_count=int(value['supervised_voxel_count']),
		per_class_supervised_counts={
			str(key): int(item) for key, item in counts.items()
		},
	)


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
	value = mapping.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{key} must be a non-empty string')
	return value


def _payload_triplet(
	mapping: Mapping[str, Any], key: str, *, positive: bool = True
) -> tuple[int, int, int]:
	value = mapping.get(key)
	return (
		_positive_triplet(value, key) if positive else _nonnegative_triplet(value, key)
	)


def _class_ids(value: object) -> tuple[int, ...]:
	if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
		raise TypeError('class_ids must be a non-empty sequence of integers')
	if not value:
		raise ValueError('class_ids must be non-empty')
	if any(not isinstance(item, Integral) or isinstance(item, bool) for item in value):
		raise TypeError('class_ids must contain integers')
	result = tuple(int(item) for item in value)
	if len(set(result)) != len(result):
		raise ValueError('class_ids must be unique')
	return result


def _positive_triplet(value: object, name: str) -> tuple[int, int, int]:
	result = _nonnegative_triplet(value, name)
	if any(item == 0 for item in result):
		raise ValueError(f'{name} must be a positive integer triple')
	return result


def _nonnegative_triplet(value: object, name: str) -> tuple[int, int, int]:
	if (
		isinstance(value, (str, bytes))
		or not isinstance(value, Sequence)
		or len(value) != 3
	):
		raise ValueError(f'{name} must be an integer triple')
	if any(not isinstance(item, Integral) or isinstance(item, bool) for item in value):
		raise TypeError(f'{name} must be an integer triple')
	result = tuple(int(item) for item in value)
	if any(item < 0 for item in result):
		raise ValueError(f'{name} must be a non-negative integer triple')
	return result  # type: ignore[return-value]


def _slices(
	start: tuple[int, int, int], stop: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
	return tuple(slice(a, b) for a, b in zip(start, stop, strict=True))  # type: ignore[return-value]


__all__ = [
	'VoxelTileManifest',
	'VoxelTileRecord',
	'build_voxel_tile_manifest',
	'build_voxel_tile_manifests',
	'read_voxel_tile_manifest',
	'voxel_tile_manifest_sha256',
	'write_voxel_tile_manifest',
]
