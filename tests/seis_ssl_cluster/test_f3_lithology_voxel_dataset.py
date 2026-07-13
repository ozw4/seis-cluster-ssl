from __future__ import annotations

import csv
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_dataset import (
	F3LithologyVoxelDatasetConfig,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_dataset import (
	build_f3_lithology_voxel_dataset,
	inspect_f3_lithology_voxel_dataset,
)

if TYPE_CHECKING:
	from pathlib import Path


def _fixture(tmp_path: Path) -> F3LithologyVoxelDatasetConfig:
	artifact_root = tmp_path / 'artifacts'
	f3_root = tmp_path / 'f3'
	artifact_root.mkdir()
	f3_root.mkdir()
	labels = np.ones((4, 4, 4), dtype=np.int16)
	labels[0, 0, 1] = 0
	np.save(artifact_root / 'labels.npy', labels)
	np.save(artifact_root / 'valid.npy', np.ones((2, 2, 2), dtype=np.bool_))
	_write_json(
		artifact_root / 'embedding.json',
		{
			'patch_size': [2, 2, 2],
			'token_grid_shape': [2, 2, 2],
			'volume_shape_xyz': [4, 4, 4],
		},
	)
	_write_json(
		artifact_root / 'class_info.json',
		{
			'0': {'name': 'zero', 'color': [0, 0, 0]},
			'1': {'name': 'one', 'color': [1, 1, 1]},
			'2': {'name': 'zero count', 'color': [2, 2, 2]},
		},
	)
	_write_json(
		artifact_root / 'geometry.json',
		{
			'segy_files': {
				'label': {
					'cube_shape': [4, 4, 4],
					'iline_min': 100,
					'iline_max': 103,
					'xline_min': 200,
					'xline_max': 203,
				}
			}
		},
	)
	with (artifact_root / 'inventory.csv').open(
		'w', encoding='utf-8', newline=''
	) as file_obj:
		writer = csv.DictWriter(
			file_obj,
			fieldnames=('relative_path', 'split', 'slice_type', 'slice_index'),
		)
		writer.writeheader()
		writer.writerows(
			[
				{
					'relative_path': 'train.png',
					'split': 'train',
					'slice_type': 'inline',
					'slice_index': 100,
				},
				{
					'relative_path': 'validation.png',
					'split': 'validation',
					'slice_type': 'crossline',
					'slice_index': 203,
				},
			]
		)
	(f3_root / 'labels.sgy').write_bytes(b'label segy identity')
	return F3LithologyVoxelDatasetConfig(
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset={'name': 'f3_facies_benchmark', 'version': 'v1'},
		source_label_volume=artifact_root / 'labels.npy',
		source_label_segy=f3_root / 'labels.sgy',
		png_label_inventory=artifact_root / 'inventory.csv',
		class_info=artifact_root / 'class_info.json',
		segy_geometry_json=artifact_root / 'geometry.json',
		reference_metadata_json=artifact_root / 'embedding.json',
		reference_valid_tokens=artifact_root / 'valid.npy',
		output_dir=artifact_root / 'voxel',
		ignore_z_border_samples=1,
		overwrite=False,
	)


def test_builds_complete_voxel_supervision_artifact(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	result = build_f3_lithology_voxel_dataset(config)
	assert sorted(path.name for path in result.output_dir.iterdir()) == [
		'class_counts.csv',
		'split_manifest.json',
		'supervision_split_grid.npy',
		'voxel_dataset_metadata.json',
		'voxel_dataset_summary.md',
	]
	grid = np.load(result.split_grid, mmap_mode='r')
	assert grid.shape == (4, 4, 4)
	assert grid.dtype == np.uint8
	assert set(np.unique(grid)) == {0, 1, 2}
	metadata = json.loads(result.metadata_json.read_text())
	assert metadata['label_volume']['sha256'] == file_sha256(config.source_label_volume)
	assert metadata['inventory']['sha256'] == file_sha256(config.png_label_inventory)
	assert metadata['validation_precedence'] is True
	rows = list(csv.DictReader(result.class_counts_csv.open()))
	assert len(rows) == 9
	assert [row for row in rows if row['class_id'] == '2']
	assert all(row['count'] == '0' for row in rows if row['class_id'] == '2')
	assert metadata['summary']['final_train_voxels'] == result.train_voxel_count


def test_collision_overwrite_and_dry_inspection(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	inspection = inspect_f3_lithology_voxel_dataset(config)
	assert inspection.volume_shape_xyz == (4, 4, 4)
	assert not config.output_dir.exists()
	(config.output_dir.parent / '.voxel.staging-interrupted').mkdir()
	build_f3_lithology_voxel_dataset(config)
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		build_f3_lithology_voxel_dataset(config)
	build_f3_lithology_voxel_dataset(replace(config, overwrite=True))


def test_rejects_embedding_geometry_mismatch(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	_write_json(
		config.reference_metadata_json,
		{
			'patch_size': [2, 2, 2],
			'token_grid_shape': [2, 2, 2],
			'volume_shape_xyz': [5, 4, 4],
		},
	)
	with pytest.raises(ValueError, match='volume_shape_xyz'):
		inspect_f3_lithology_voxel_dataset(config)


def test_dry_inspection_constructs_and_fully_validates_split(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	inspection = inspect_f3_lithology_voxel_dataset(config)
	assert inspection.split.summary.final_train_voxels > 0
	assert inspection.split.summary.final_validation_voxels > 0

	labels = np.load(config.source_label_volume).astype(np.float32)
	np.save(config.source_label_volume, labels)
	with pytest.raises(TypeError, match='label_volume dtype must be integer'):
		inspect_f3_lithology_voxel_dataset(config)


def test_dry_inspection_rejects_invalid_physical_slice_index(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	rows = list(csv.DictReader(config.png_label_inventory.open()))
	rows[0]['slice_index'] = '99'
	with config.png_label_inventory.open(
		'w', encoding='utf-8', newline=''
	) as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=rows[0])
		writer.writeheader()
		writer.writerows(rows)

	with pytest.raises(ValueError, match='resolves outside F3 cube axis'):
		inspect_f3_lithology_voxel_dataset(config)


def test_dry_inspection_rejects_empty_supervision_split(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	valid_tokens = np.load(config.reference_valid_tokens)
	valid_tokens[0, :, :] = False
	np.save(config.reference_valid_tokens, valid_tokens)

	with pytest.raises(ValueError, match='at least one train voxel'):
		inspect_f3_lithology_voxel_dataset(config)


def _write_json(path: Path, payload: object) -> None:
	path.write_text(json.dumps(payload), encoding='utf-8')
