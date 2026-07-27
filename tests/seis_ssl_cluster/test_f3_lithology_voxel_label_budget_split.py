from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_dataset import GRID_NAME, METADATA_NAME
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split import (
	_array_sha,
	_complete,
	_require_selected_tokens_cover_train_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split_runner import (
	_run_manifest_root,
	_write_run_manifest,
)
from seis_ssl_cluster.f3.lithology.voxel_split import TRAIN_VOXEL_SPLIT


def test_selected_unique_tokens_require_canonical_train_voxels() -> None:
	full_grid = np.zeros((16, 16, 16), dtype=np.uint8)
	with pytest.raises(
		ValueError, match='selected unique token has no canonical train voxel'
	):
		_require_selected_tokens_cover_train_voxels(
			np.asarray([[0, 0, 0]], dtype=np.int64), full_grid
		)

	full_grid[0, 0, 0] = TRAIN_VOXEL_SPLIT
	_require_selected_tokens_cover_train_voxels(
		np.asarray([[0, 0, 0]], dtype=np.int64), full_grid
	)


def test_selected_token_coverage_uses_only_each_clipped_token_block(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	full_grid = np.zeros((9, 9, 9), dtype=np.uint8)
	full_grid[8, 8, 8] = TRAIN_VOXEL_SPLIT

	def fail_full_volume_mask(*args: object, **kwargs: object) -> object:
		raise AssertionError(
			f'coverage must not allocate a full-volume mask: {args!r}/{kwargs!r}'
		)

	monkeypatch.setattr(
		'seis_ssl_cluster.f3.lithology.voxel_label_budget_split.np.zeros',
		fail_full_volume_mask,
	)
	_require_selected_tokens_cover_train_voxels(
		np.asarray([[1, 1, 1]], dtype=np.int64), full_grid,
		split_id='split_005', budget_id='cap50',
	)
	with pytest.raises(ValueError, match=r'split_005/cap50/\[0, 0, 0\]'):
		_require_selected_tokens_cover_train_voxels(
			np.asarray([[1, 1, 1], [0, 0, 0]], dtype=np.int64),
			full_grid,
			split_id='split_005', budget_id='cap50',
		)
	with pytest.raises(ValueError, match='outside the volume'):
		_require_selected_tokens_cover_train_voxels(
			np.asarray([[2, 0, 0]], dtype=np.int64), full_grid
		)


def test_only_missing_reuses_only_a_complete_identity_matching_dataset(
	tmp_path,
) -> None:
	root = tmp_path / 'voxel_supervision'
	root.mkdir()
	source_manifest = tmp_path / 'source_split_manifest.json'
	source_manifest.write_text('{"source": true}\n', encoding='utf-8')
	grid = np.asarray([[[TRAIN_VOXEL_SPLIT]]], dtype=np.uint8)
	selected = np.asarray([[0, 0, 0]], dtype=np.int64)
	row = {
		'split_id': 'split_000',
		'budget_id': 'cap25',
		'per_class_cap': 25,
		'label_subset_seed': 0,
		'voxel_dataset_root': str(root),
		'selected_token_row_count': 1,
		'unique_selected_token_xyz_count': 1,
		'duplicate_selected_row_count': 0,
		'selected_token_identity_sha256': 'selected-row-identity',
		'unique_token_xyz_sha256': _array_sha(selected),
		'train_voxel_count': 1,
		'actual_train_voxel_count': 1,
		'validation_voxel_count': 0,
		'per_class_train_voxel_counts': {'0': 1},
		'per_class_validation_voxel_counts': {'0': 0},
		'train_mask_sha256': 'train-mask',
		'validation_mask_sha256': 'validation-mask',
		'grid_array_sha256': _array_sha(grid),
		'supervision_split_grid': {
			'path': str(root / GRID_NAME),
			'sha256': '',
		},
		'canonical_valid_tokens_sha256': 'valid-tokens',
		'class_order': [0],
		'patch_size_xyz': [8, 8, 8],
		'source_full_voxel_dataset': {
			'slice_split_manifest': {
				'path': str(source_manifest),
				'sha256': file_sha256(source_manifest),
			},
		},
	}
	np.save(root / GRID_NAME, grid, allow_pickle=False)
	row['supervision_split_grid']['sha256'] = file_sha256(root / GRID_NAME)
	np.save(root / 'selected_token_xyz.npy', selected, allow_pickle=False)
	identity = {
		key: value for key, value in row.items() if key != 'source_full_voxel_dataset'
	}
	(root / METADATA_NAME).write_text(
		json.dumps(
			{
				'outputs': {
					'supervision_split_grid': str(root / GRID_NAME),
					'metadata_json': str(root / METADATA_NAME),
				},
				'voxel_label_budget_split': {
					'split_id': 'split_000',
					'budget_id': 'cap25',
					'dense_voxel_labels_preserved': True,
					'validation_reuse': 'canonical_full_validation_bitwise',
					'identity': identity,
				},
			}
		),
		encoding='utf-8',
	)
	(root / 'low_label_split_metadata.json').write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_voxel_label_budget_split_dataset',
				'identity': identity,
				'sources': {'full_voxel_dataset': row['source_full_voxel_dataset']},
			}
		),
		encoding='utf-8',
	)
	(root / 'split_manifest.json').write_bytes(source_manifest.read_bytes())
	with (root / 'class_counts.csv').open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(handle, fieldnames=('split', 'class_id', 'count'))
		writer.writeheader()
		writer.writerows(
			(
				{'split': 'train', 'class_id': '0', 'count': '1'},
				{'split': 'validation', 'class_id': '0', 'count': '0'},
			)
		)

	assert _complete(root, row)

	for filename in (
		GRID_NAME,
		'selected_token_xyz.npy',
		METADATA_NAME,
		'low_label_split_metadata.json',
		'class_counts.csv',
		'split_manifest.json',
	):
		content = (root / filename).read_bytes()
		(root / filename).unlink()
		assert not _complete(root, row)
		(root / filename).write_bytes(content)

	(root / 'low_label_split_metadata.json').write_text('{}', encoding='utf-8')
	assert not _complete(root, row)


def test_smoke_run_manifest_is_separate_from_scientific_manifest(tmp_path) -> None:
	config = SimpleNamespace(output_root=tmp_path)
	scientific_root = _run_manifest_root(config, smoke_only=False)
	smoke_root = _run_manifest_root(config, smoke_only=True)
	_write_run_manifest(
		scientific_root,
		[{
			'split_id': 'split_000',
			'budget_id': 'cap25',
			'model_role': 'mae',
			'status': 'running',
		}],
	)
	_write_run_manifest(
		smoke_root,
		[{
			'split_id': 'split_000',
			'budget_id': 'cap25',
			'model_role': 'mae',
			'status': 'smoke_running',
		}],
	)

	scientific = json.loads(
		(tmp_path / 'low_label_split_run_manifest.json').read_text(encoding='utf-8')
	)
	smoke = json.loads(
		(tmp_path / 'smoke' / 'low_label_split_run_manifest.json').read_text(
			encoding='utf-8'
		)
	)
	assert scientific['rows'][0]['status'] == 'running'
	assert smoke['rows'][0]['status'] == 'smoke_running'
