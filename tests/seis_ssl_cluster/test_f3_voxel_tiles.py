from __future__ import annotations

import json

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.voxel_tiles import (
	build_voxel_tile_manifest,
	read_voxel_tile_manifest,
	write_voxel_tile_manifest,
)


def test_non_divisible_core_partition_and_edge_halo(tmp_path) -> None:
	labels = np.zeros((5, 3, 2), dtype=np.int16)
	grid = np.ones(labels.shape, dtype=np.uint8)

	manifest = build_voxel_tile_manifest(
		grid,
		labels,
		split='train',
		patch_size_xyz=(1, 1, 1),
		core_size_tokens=(4, 2, 2),
		context_halo_tokens=(1, 1, 1),
		class_ids=(0,),
		canonical_valid_tokens=np.ones((5, 3, 2), dtype=np.bool_),
	)

	assert len(manifest.tiles) == 4
	assert sum(tile.supervised_voxel_count for tile in manifest.tiles) == grid.size
	edge = manifest.tiles[-1]
	assert edge.core_start_token_xyz == (4, 2, 0)
	assert edge.core_stop_token_xyz == (5, 3, 2)
	assert edge.input_padding_after_xyz == (4, 2, 1)
	path = tmp_path / 'tiles.json'
	write_voxel_tile_manifest(path, manifest)
	loaded = read_voxel_tile_manifest(path)
	assert loaded == manifest
	assert loaded.identity_sha256 == manifest.identity_sha256


def test_split_manifests_cover_known_voxels_once_without_leakage() -> None:
	labels = np.zeros((4, 2, 2), dtype=np.int16)
	labels[3, 1, 1] = -1
	grid = np.zeros(labels.shape, dtype=np.uint8)
	grid[:2] = 1
	grid[2:] = 2

	train = build_voxel_tile_manifest(
		grid,
		labels,
		split='train',
		patch_size_xyz=(1, 1, 1),
		core_size_tokens=(2, 2, 2),
		context_halo_tokens=(1, 1, 1),
		class_ids=(0,),
		canonical_valid_tokens=np.ones((4, 2, 2), dtype=np.bool_),
	)
	validation = build_voxel_tile_manifest(
		grid,
		labels,
		split='validation',
		patch_size_xyz=(1, 1, 1),
		core_size_tokens=(2, 2, 2),
		context_halo_tokens=(1, 1, 1),
		class_ids=(0,),
		canonical_valid_tokens=np.ones((4, 2, 2), dtype=np.bool_),
	)

	assert sum(item.supervised_voxel_count for item in train.tiles) == 8
	assert sum(item.supervised_voxel_count for item in validation.tiles) == 7
	assert {item.split for item in train.tiles} == {'train'}
	assert {item.split for item in validation.tiles} == {'validation'}


def test_manifest_excludes_voxels_under_invalid_canonical_tokens() -> None:
	labels = np.zeros((4, 1, 1), dtype=np.int16)
	grid = np.ones(labels.shape, dtype=np.uint8)
	valid_tokens = np.asarray([False, True], dtype=np.bool_).reshape(2, 1, 1)

	manifest = build_voxel_tile_manifest(
		grid,
		labels,
		split='train',
		patch_size_xyz=(2, 1, 1),
		core_size_tokens=(1, 1, 1),
		context_halo_tokens=(1, 1, 1),
		class_ids=(0,),
		canonical_valid_tokens=valid_tokens,
	)

	assert len(manifest.tiles) == 1
	assert manifest.tiles[0].core_start_token_xyz == (1, 0, 0)
	assert manifest.tiles[0].supervised_voxel_count == 2


@pytest.mark.parametrize(
	('key', 'value', 'message'),
	[
		('artifact_type', 'other', 'artifact_type'),
		('schema_version', 2, 'schema_version'),
		('tile_count', 99, 'tile_count'),
		('supervised_voxel_count', 99, 'supervised_voxel_count'),
	],
)
def test_manifest_reader_rejects_invalid_schema_and_totals(
	tmp_path, key: str, value: object, message: str
) -> None:
	manifest = build_voxel_tile_manifest(
		np.ones((1, 1, 1), dtype=np.uint8),
		np.zeros((1, 1, 1), dtype=np.int16),
		split='train',
		patch_size_xyz=(1, 1, 1),
		class_ids=(0,),
		canonical_valid_tokens=np.ones((1, 1, 1), dtype=np.bool_),
	)
	payload = manifest.to_dict()
	payload[key] = value
	path = tmp_path / 'invalid.json'
	path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match=message):
		read_voxel_tile_manifest(path)


def test_manifest_reader_requires_identity(tmp_path) -> None:
	manifest = build_voxel_tile_manifest(
		np.ones((1, 1, 1), dtype=np.uint8),
		np.zeros((1, 1, 1), dtype=np.int16),
		split='train',
		patch_size_xyz=(1, 1, 1),
		class_ids=(0,),
		canonical_valid_tokens=np.ones((1, 1, 1), dtype=np.bool_),
	)
	payload = manifest.to_dict()
	del payload['identity_sha256']
	path = tmp_path / 'missing-identity.json'
	path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='identity_sha256'):
		read_voxel_tile_manifest(path)
