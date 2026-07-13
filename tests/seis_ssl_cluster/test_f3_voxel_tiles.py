from __future__ import annotations

import numpy as np

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
	)
	validation = build_voxel_tile_manifest(
		grid,
		labels,
		split='validation',
		patch_size_xyz=(1, 1, 1),
		core_size_tokens=(2, 2, 2),
		context_halo_tokens=(1, 1, 1),
		class_ids=(0,),
	)

	assert sum(item.supervised_voxel_count for item in train.tiles) == 8
	assert sum(item.supervised_voxel_count for item in validation.tiles) == 7
	assert {item.split for item in train.tiles} == {'train'}
	assert {item.split for item in validation.tiles} == {'validation'}
