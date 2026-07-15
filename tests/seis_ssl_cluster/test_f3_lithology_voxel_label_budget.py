from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_label_budget import (
	F3VoxelLabelBudgetDatasetConfig,
	f3_lithology_voxel_label_budget_dataset_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.robustness import (
	load_token_dataset_npz,
	paired_token_identity_hash,
	save_token_dataset_npz,
)
from seis_ssl_cluster.f3.lithology.token_dataset import F3LithologyTokenDataset
from seis_ssl_cluster.f3.lithology.voxel_label_budget import (
	LABEL_BUDGET_METADATA_NAME,
	REQUIRED_CONDITION_FILES,
	build_f3_lithology_voxel_label_budget_datasets,
	build_low_label_supervision_grid,
	expand_selected_token_blocks,
	inspect_f3_lithology_voxel_label_budget_datasets,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = (
	REPO_ROOT
	/ 'proc/seis_ssl_cluster/build_f3_lithology_voxel_label_budget_datasets.py'
)


def test_expands_eight_voxel_blocks_and_clips_volume_edges() -> None:
	mask = expand_selected_token_blocks(
		np.asarray([[0, 0, 0], [1, 1, 0]], dtype=np.int64),
		patch_size_xyz=(8, 8, 8),
		volume_shape_xyz=(10, 9, 7),
	)

	assert np.all(mask[:8, :8, :])
	assert np.all(mask[8:, 8:, :])
	assert int(np.count_nonzero(mask)) == 8 * 8 * 7 + 2 * 1 * 7


def test_low_label_grid_intersects_train_and_preserves_validation() -> None:
	full = np.zeros((4, 4, 4), dtype=np.uint8)
	full[0, 0, 0] = 2
	full[0, 0, 1] = 1
	full[1, 1, 1] = 1
	full[3, 3, 3] = 1
	full[3, 0, 0] = 2

	low = build_low_label_supervision_grid(
		full,
		np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int64),
		patch_size_xyz=(2, 2, 2),
	)

	assert low.dtype == full.dtype
	assert low.shape == full.shape
	assert np.array_equal(low == 2, full == 2)
	assert low[0, 0, 0] == 2
	assert low[0, 0, 1] == 1
	assert low[1, 1, 1] == 1
	assert low[3, 3, 3] == 0


def test_low_label_grid_rejects_invalid_or_uncovered_tokens() -> None:
	full = np.zeros((4, 4, 4), dtype=np.uint8)
	full[0, 0, 0] = 1
	full[3, 3, 3] = 2

	with pytest.raises(ValueError, match='non-negative'):
		build_low_label_supervision_grid(
			full,
			np.asarray([[-1, 0, 0]], dtype=np.int64),
			patch_size_xyz=(2, 2, 2),
		)
	with pytest.raises(ValueError, match='no canonical train voxel'):
		build_low_label_supervision_grid(
			full,
			np.asarray([[1, 1, 1]], dtype=np.int64),
			patch_size_xyz=(2, 2, 2),
		)


def test_inspection_records_duplicate_rows_and_dense_voxel_classes(
	tmp_path: Path,
) -> None:
	config, _raw, _paths = _write_fixture(tmp_path, duplicate_train_xyz=True)

	inspection = inspect_f3_lithology_voxel_label_budget_datasets(config)
	condition = inspection.conditions[0]

	assert condition.selected_token_rows == 2
	assert condition.unique_token_xyz.shape == (1, 3)
	assert condition.duplicate_selected_rows == 1
	assert condition.train_voxel_count == 8
	assert condition.per_class_train_voxel_counts == {'0': 7, '1': 1}


def test_build_roundtrip_overwrite_reuse_and_invalid_quarantine(
	tmp_path: Path,
) -> None:
	config, _raw, _paths = _write_fixture(tmp_path)

	first = build_f3_lithology_voxel_label_budget_datasets(config)
	condition_root = first.condition_roots[0]
	assert first.rows[0]['action'] == 'NEW'
	assert all((condition_root / name).is_file() for name in REQUIRED_CONDITION_FILES)
	metadata = _read_json(condition_root / LABEL_BUDGET_METADATA_NAME)
	identity = metadata['identity']
	assert identity['selected_token_row_count'] == 2
	assert identity['actual_train_voxel_count'] == 16
	assert identity['per_class_train_voxel_counts'] == {'0': 15, '1': 1}

	with pytest.raises(FileExistsError, match='refusing existing'):
		build_f3_lithology_voxel_label_budget_datasets(config)

	reused = build_f3_lithology_voxel_label_budget_datasets(
		config, only_missing=True
	)
	assert reused.rows[0]['action'] == 'REUSED'
	assert reused.quarantines == ()

	grid_path = condition_root / 'supervision_split_grid.npy'
	corrupt = np.load(grid_path, allow_pickle=False)
	corrupt[0, 0, 0] = 0
	np.save(grid_path, corrupt, allow_pickle=False)
	rebuilt = build_f3_lithology_voxel_label_budget_datasets(
		config, only_missing=True
	)
	assert rebuilt.rows[0]['action'] == 'REBUILT_AFTER_QUARANTINE'
	assert len(rebuilt.quarantines) == 1
	quarantine = rebuilt.quarantines[0]
	assert quarantine.is_dir()
	assert np.load(quarantine / grid_path.name)[0, 0, 0] == 0
	assert np.load(grid_path)[0, 0, 0] == 1


@pytest.mark.parametrize(
	('suite_key', 'model_tag', 'error'),
	[
		('mae_m1_manifest', 'm1_tag', 'token identity rows differ'),
		('m1_m2a_manifest', 'm1_tag', 'token identity rows differ'),
	],
)
def test_rejects_within_or_cross_suite_token_identity_mismatch(
	tmp_path: Path,
	suite_key: str,
	model_tag: str,
	error: str,
) -> None:
	config, _raw, paths = _write_fixture(tmp_path)
	_mutate_selected_token_identity(paths[suite_key], model_tag=model_tag)

	with pytest.raises(ValueError, match=error):
		inspect_f3_lithology_voxel_label_budget_datasets(config)


def test_rejects_declared_paired_hash_mismatch(tmp_path: Path) -> None:
	config, _raw, paths = _write_fixture(tmp_path)
	manifest_path = paths['mae_m1_manifest']
	manifest = _read_json(manifest_path)
	row = next(row for row in manifest['rows'] if row['model_tag'] == 'mae_tag')
	row['paired_identity_hash'] = '0' * 64
	metadata_path = Path(row['metadata_json'])
	metadata = _read_json(metadata_path)
	metadata['paired_identity_hash'] = row['paired_identity_hash']
	_write_json(metadata_path, metadata)
	_write_json(manifest_path, manifest)

	with pytest.raises(ValueError, match='paired identity hash mismatch'):
		inspect_f3_lithology_voxel_label_budget_datasets(config)


def test_rejects_missing_dense_voxel_class(tmp_path: Path) -> None:
	config, _raw, paths = _write_fixture(tmp_path)
	labels_path = paths['labels']
	labels = np.load(labels_path, allow_pickle=False)
	labels[1, 1, 1] = 0
	np.save(labels_path, labels, allow_pickle=False)
	common_metadata_path = config.common_voxel_dataset / 'voxel_dataset_metadata.json'
	metadata = _read_json(common_metadata_path)
	metadata['label_volume']['sha256'] = file_sha256(labels_path)
	_write_json(common_metadata_path, metadata)

	with pytest.raises(ValueError, match='missing a required class'):
		inspect_f3_lithology_voxel_label_budget_datasets(config)


def test_cli_dry_run_validates_without_writing(tmp_path: Path) -> None:
	config, raw, _paths = _write_fixture(tmp_path)
	config_path = tmp_path / 'builder.json'
	_write_json(config_path, raw)
	environment = os.environ.copy()
	environment['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), environment.get('PYTHONPATH', ''))
	)

	completed = subprocess.run(  # noqa: S603
		[
			sys.executable,
			str(CLI),
			'--config',
			str(config_path),
			'--dry-run',
			'--only-missing',
		],
		cwd=REPO_ROOT,
		env=environment,
		text=True,
		capture_output=True,
		check=True,
		timeout=30,
	)

	assert 'condition_count: 1' in completed.stdout
	assert 'common_validation_voxel_count: 25' in completed.stdout
	assert 'execution: dry-run; no artifacts written' in completed.stdout
	assert not config.output_root.exists()


def _write_fixture(
	tmp_path: Path,
	*,
	duplicate_train_xyz: bool = False,
) -> tuple[
	F3VoxelLabelBudgetDatasetConfig,
	dict[str, object],
	dict[str, Path],
]:
	artifact_root = tmp_path / 'artifacts'
	common = artifact_root / 'common'
	common.mkdir(parents=True)
	shape = (5, 5, 5)
	patch = (2, 2, 2)
	token_shape = (3, 3, 3)
	labels = np.zeros(shape, dtype=np.int16)
	labels[1, 1, 1] = 1
	for y in range(shape[1]):
		for z in range(shape[2]):
			labels[4, y, z] = (y + z) % 2
	labels_path = artifact_root / 'labels.npy'
	np.save(labels_path, labels, allow_pickle=False)
	full_grid = np.zeros(shape, dtype=np.uint8)
	full_grid[0:2, 0:2, 0:2] = 1
	full_grid[2:4, 2:4, 2:4] = 1
	full_grid[4, :, :] = 2
	np.save(common / 'supervision_split_grid.npy', full_grid, allow_pickle=False)
	embedding_metadata = artifact_root / 'embedding_metadata.json'
	_write_json(embedding_metadata, {'fixture': True})
	valid_tokens = artifact_root / 'valid_tokens.npy'
	np.save(valid_tokens, np.ones(token_shape, dtype=np.bool_), allow_pickle=False)
	inventory = artifact_root / 'inventory.csv'
	inventory.write_text('fixture\n', encoding='utf-8')
	(common / 'class_counts.csv').write_text('fixture\n', encoding='utf-8')
	_write_json(common / 'split_manifest.json', {'fixture': 'canonical split'})
	common_metadata = {
		'artifact_type': 'f3_lithology_voxel_supervision',
		'schema_version': 1,
		'dataset': {'name': 'fixture', 'version': 'v1'},
		'classes': [
			{'class_id': 0, 'class_name': 'zero'},
			{'class_id': 1, 'class_name': 'one'},
		],
		'split_codes': {'unsupervised': 0, 'train': 1, 'validation': 2},
		'reference_embedding': {
			'path': str(embedding_metadata),
			'sha256': file_sha256(embedding_metadata),
			'patch_size': list(patch),
			'token_grid_shape': list(token_shape),
			'volume_shape_xyz': list(shape),
		},
		'reference_valid_tokens': _identity(valid_tokens),
		'label_volume': _identity(labels_path),
		'inventory': _identity(inventory),
		'summary': {
			'final_train_voxels': 16,
			'final_validation_voxels': 25,
		},
		'outputs': {
			'supervision_split_grid': str(
				common / 'supervision_split_grid.npy'
			),
			'metadata_json': str(common / 'voxel_dataset_metadata.json'),
			'class_counts_csv': str(common / 'class_counts.csv'),
			'split_manifest_json': str(common / 'split_manifest.json'),
		},
	}
	_write_json(common / 'voxel_dataset_metadata.json', common_metadata)
	train_xyz = np.asarray(
		[[0, 0, 0], [0, 0, 0] if duplicate_train_xyz else [1, 1, 1]],
		dtype=np.int64,
	)
	train = _token_dataset('train', token_xyz=train_xyz, feature_value=0.0)
	validation = _token_dataset(
		'validation',
		token_xyz=np.asarray([[2, 0, 0], [2, 1, 1]], dtype=np.int64),
		feature_value=0.0,
	)
	mae_m1_manifest = _write_token_suite(
		artifact_root / 'label_budget_m1',
		models=(('mae_tag', 'baseline'), ('m1_tag', 'candidate')),
		train=train,
		validation=validation,
	)
	m1_m2a_manifest = _write_token_suite(
		artifact_root / 'label_budget_m2a',
		models=(('m1_tag', 'baseline'), ('m2a_tag', 'candidate')),
		train=train,
		validation=validation,
	)
	raw: dict[str, object] = {
		'paths': {'artifact_root': str(artifact_root)},
		'suite': {
			'name': 'fixture_voxel_label_budget',
			'output_root': str(artifact_root / 'low_label'),
		},
		'inputs': {
			'common_voxel_dataset': str(common),
			'mae_m1_label_budget_manifest': str(mae_m1_manifest),
			'm1_m2a_label_budget_manifest': str(m1_m2a_manifest),
		},
		'models': {'mae': 'mae_tag', 'm1': 'm1_tag', 'm2a': 'm2a_tag'},
		'label_budget': {
			'budgets': ['cap1'],
			'subsample_seeds': [0],
			'patch_size_xyz': list(patch),
			'require_all_classes': True,
		},
		'outputs': {'overwrite': False},
	}
	config = f3_lithology_voxel_label_budget_dataset_config_from_mapping(raw)
	return config, raw, {
		'labels': labels_path,
		'mae_m1_manifest': mae_m1_manifest,
		'm1_m2a_manifest': m1_m2a_manifest,
	}


def _write_token_suite(
	root: Path,
	*,
	models: tuple[tuple[str, str], ...],
	train: F3LithologyTokenDataset,
	validation: F3LithologyTokenDataset,
) -> Path:
	rows: list[dict[str, object]] = []
	for index, (model_tag, model_role) in enumerate(models):
		dataset_root = root / model_tag
		model_train = replace(
			train,
			features=np.full_like(train.features, float(index + 1)),
		)
		model_validation = replace(
			validation,
			features=np.full_like(validation.features, float(index + 1)),
		)
		train_path = dataset_root / 'train_tokens.npz'
		validation_path = dataset_root / 'validation_tokens.npz'
		metadata_path = dataset_root / 'token_dataset_metadata.json'
		save_token_dataset_npz(model_train, train_path)
		save_token_dataset_npz(model_validation, validation_path)
		paired_hash = paired_token_identity_hash(model_train, model_validation)
		selected_counts = {'0': 1, '1': 1}
		validation_counts = {'0': 1, '1': 1}
		metadata = {
			'artifact_type': 'f3_lithology_label_budget_token_dataset',
			'model': {'model_tag': model_tag, 'role': model_role},
			'label_budget': {
				'budget_id': 'cap1',
				'per_class_cap': 1,
				'subsample_seed': 0,
			},
			'validation': {'reuse_full_validation': True},
			'selected_train_token_count': 2,
			'validation_token_count': 2,
			'selected_class_counts': selected_counts,
			'validation_class_counts': validation_counts,
			'paired_identity_hash': paired_hash,
		}
		_write_json(metadata_path, metadata)
		rows.append(
			{
				'model_role': model_role,
				'model_tag': model_tag,
				'budget_id': 'cap1',
				'per_class_cap': 1,
				'subsample_seed': 0,
				'token_dataset_root': str(dataset_root),
				'train_tokens': str(train_path),
				'validation_tokens': str(validation_path),
				'metadata_json': str(metadata_path),
				'selected_train_token_count': 2,
				'validation_token_count': 2,
				'selected_class_counts': selected_counts,
				'validation_class_counts': validation_counts,
				'paired_identity_hash': paired_hash,
			}
		)
	manifest = root / 'suite_manifest.json'
	_write_json(
		manifest,
		{
			'artifact_type': 'f3_lithology_label_budget_suite_manifest',
			'contract_version': 'fixture_v1',
			'rows': rows,
		},
	)
	return manifest


def _mutate_selected_token_identity(
	manifest_path: Path, *, model_tag: str
) -> None:
	manifest = _read_json(manifest_path)
	row = next(row for row in manifest['rows'] if row['model_tag'] == model_tag)
	train_path = Path(row['train_tokens'])
	validation_path = Path(row['validation_tokens'])
	train = load_token_dataset_npz(train_path)
	validation = load_token_dataset_npz(validation_path)
	coordinates = np.asarray(train.token_xyz).copy()
	coordinates[0] = np.asarray([0, 1, 0])
	changed = replace(train, token_xyz=coordinates)
	save_token_dataset_npz(changed, train_path)
	paired_hash = paired_token_identity_hash(changed, validation)
	row['paired_identity_hash'] = paired_hash
	metadata_path = Path(row['metadata_json'])
	metadata = _read_json(metadata_path)
	metadata['paired_identity_hash'] = paired_hash
	_write_json(metadata_path, metadata)
	_write_json(manifest_path, manifest)


def _token_dataset(
	split: str,
	*,
	token_xyz: np.ndarray,
	feature_value: float,
) -> F3LithologyTokenDataset:
	count = int(token_xyz.shape[0])
	return F3LithologyTokenDataset(
		features=np.full((count, 4), feature_value, dtype=np.float32),
		labels=np.asarray([0, 1], dtype=np.int64),
		survey_id=np.asarray(['fixture'] * count),
		split=np.asarray([split] * count),
		slice_type=np.asarray(['inline', 'crossline']),
		slice_index=np.asarray([100, 200], dtype=np.int64),
		token_xyz=np.asarray(token_xyz, dtype=np.int64),
		voxel_center_xyz=np.asarray(token_xyz, dtype=np.float32) * 2 + 0.5,
		majority_fraction=np.ones(count, dtype=np.float32),
		labeled_fraction=np.ones(count, dtype=np.float32),
		metadata={},
	)


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _read_json(path: Path) -> dict[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	assert isinstance(payload, dict)
	return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
