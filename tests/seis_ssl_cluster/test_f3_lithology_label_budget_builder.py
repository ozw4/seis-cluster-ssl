from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_label_budget_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	build_f3_lithology_label_budget_datasets,
	class_count_dict,
	load_token_dataset_npz,
	save_token_dataset_npz,
)
from seis_ssl_cluster.f3.lithology.token_dataset import F3LithologyTokenDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / 'proc' / 'seis_ssl_cluster' / (
	'build_f3_lithology_label_budget_datasets.py'
)


def test_dry_run_summary_does_not_write_files(tmp_path: Path) -> None:
	baseline, candidate = _write_paired_sources(tmp_path)
	output_root = tmp_path / 'out'
	config_path = tmp_path / 'label_budget.yaml'
	config_path.write_text(
		_config_yaml(baseline, candidate, output_root, caps='[2, null]', seeds='[0]'),
		encoding='utf-8',
	)
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), env.get('PYTHONPATH', '')),
	)

	completed = subprocess.run(  # noqa: S603
		[sys.executable, str(CLI), '--config', str(config_path), '--dry-run'],
		cwd=REPO_ROOT,
		env=env,
		text=True,
		capture_output=True,
		check=True,
		timeout=30,
	)

	assert 'suite name: label_budget_fixture' in completed.stdout
	assert 'expected dataset count: 4' in completed.stdout
	assert 'execution: dry-run; label-budget datasets skipped' in completed.stdout
	assert not output_root.exists()


def test_builder_writes_expected_directory_structure(tmp_path: Path) -> None:
	baseline, candidate = _write_paired_sources(tmp_path)
	config = _config(baseline, candidate, tmp_path / 'out', caps=(2,), seeds=(0,))

	result = build_f3_lithology_label_budget_datasets(config)

	expected_root = (
		config.output_root
		/ 'datasets'
		/ 'model=baseline_model'
		/ 'budget=cap2'
		/ 'subsample_seed=0'
		/ 'token_dataset'
	)
	assert result.suite_manifest_json == config.output_root / 'suite_manifest.json'
	assert expected_root in result.dataset_roots
	for filename in (
		'train_tokens.npz',
		'validation_tokens.npz',
		'all_labeled_tokens.npz',
		'token_dataset_metadata.json',
		'class_counts.csv',
		'token_dataset_summary.md',
	):
		assert (expected_root / filename).is_file()


def test_paired_identity_hashes_match_across_models(tmp_path: Path) -> None:
	baseline, candidate = _write_paired_sources(tmp_path)
	config = _config(baseline, candidate, tmp_path / 'out', caps=(2,), seeds=(0,))

	build_f3_lithology_label_budget_datasets(config)

	manifest = json.loads((config.output_root / 'suite_manifest.json').read_text())
	hashes = {
		row['model_role']: row['paired_identity_hash']
		for row in manifest['rows']
		if row['budget_id'] == 'cap2' and row['subsample_seed'] == 0
	}
	assert hashes['baseline'] == hashes['candidate']


def test_per_class_caps_are_respected(tmp_path: Path) -> None:
	baseline, candidate = _write_paired_sources(tmp_path)
	config = _config(baseline, candidate, tmp_path / 'out', caps=(2,), seeds=(0,))

	build_f3_lithology_label_budget_datasets(config)

	train = load_token_dataset_npz(
		config.output_root
		/ 'datasets/model=baseline_model/budget=cap2/subsample_seed=0'
		/ 'token_dataset/train_tokens.npz',
	)
	counts = class_count_dict(train.labels)
	assert counts == {'0': 2, '1': 2, '2': 2}


def test_full_budget_keeps_all_train_tokens(tmp_path: Path) -> None:
	baseline, candidate = _write_paired_sources(tmp_path)
	config = _config(baseline, candidate, tmp_path / 'out', caps=(None,), seeds=(3,))

	build_f3_lithology_label_budget_datasets(config)

	train = load_token_dataset_npz(
		config.output_root
		/ 'datasets/model=baseline_model/budget=full/subsample_seed=3'
		/ 'token_dataset/train_tokens.npz',
	)
	assert train.count == load_token_dataset_npz(baseline / 'train_tokens.npz').count


def test_overwrite_false_rejects_existing_outputs(tmp_path: Path) -> None:
	baseline, candidate = _write_paired_sources(tmp_path)
	config = _config(baseline, candidate, tmp_path / 'out', caps=(2,), seeds=(0,))
	build_f3_lithology_label_budget_datasets(config)

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		build_f3_lithology_label_budget_datasets(config)


def test_source_identity_mismatch_fails(tmp_path: Path) -> None:
	baseline, candidate = _write_paired_sources(tmp_path, mismatch=True)
	config = _config(baseline, candidate, tmp_path / 'out', caps=(2,), seeds=(0,))

	with pytest.raises(ValueError, match='reference_token_dataset'):
		build_f3_lithology_label_budget_datasets(config)


def _config(
	baseline: Path,
	candidate: Path,
	output_root: Path,
	*,
	caps: tuple[int | None, ...],
	seeds: tuple[int, ...],
):
	return f3_lithology_label_budget_config_from_mapping(
		{
			'paths': {'artifact_root': str(output_root.parent)},
			'suite': {
				'name': 'label_budget_fixture',
				'output_root': str(output_root),
			},
			'models': {
				'baseline': {
					'model_tag': 'baseline_model',
					'token_dataset_root': str(baseline),
				},
				'candidate': {
					'model_tag': 'candidate_model',
					'token_dataset_root': str(candidate),
				},
			},
			'label_budget': {
				'mode': 'per_class_cap',
				'per_class_caps': list(caps),
				'subsample_seeds': list(seeds),
				'require_all_classes': True,
			},
			'validation': {'reuse_full_validation': True},
			'outputs': {'overwrite': False},
		},
	)


def _config_yaml(
	baseline: Path,
	candidate: Path,
	output_root: Path,
	*,
	caps: str,
	seeds: str,
) -> str:
	return f"""
paths:
  artifact_root: {output_root.parent}
suite:
  name: label_budget_fixture
  output_root: {output_root}
models:
  baseline:
    model_tag: baseline_model
    token_dataset_root: {baseline}
  candidate:
    model_tag: candidate_model
    token_dataset_root: {candidate}
label_budget:
  mode: per_class_cap
  per_class_caps: {caps}
  subsample_seeds: {seeds}
  require_all_classes: true
validation:
  reuse_full_validation: true
outputs:
  overwrite: false
"""


def _write_paired_sources(
	tmp_path: Path,
	*,
	mismatch: bool = False,
) -> tuple[Path, Path]:
	baseline = tmp_path / 'baseline_tokens'
	candidate = tmp_path / 'candidate_tokens'
	train = _dataset(split='train')
	validation = _dataset(split='validation', count=6)
	candidate_train = _dataset(
		split='train',
		features=np.full((12, 4), 10.0, dtype=np.float32),
		token_xyz=(
			_train_token_xyz(mismatch_index=4)
			if mismatch
			else _train_token_xyz()
		),
	)
	candidate_validation = _dataset(
		split='validation',
		count=6,
		features=np.full((6, 4), 11.0, dtype=np.float32),
	)
	for root, train_dataset, validation_dataset in (
		(baseline, train, validation),
		(candidate, candidate_train, candidate_validation),
	):
		root.mkdir(parents=True)
		save_token_dataset_npz(train_dataset, root / 'train_tokens.npz')
		save_token_dataset_npz(validation_dataset, root / 'validation_tokens.npz')
		(root / 'token_dataset_metadata.json').write_text(
			json.dumps({'source': root.name}) + '\n',
			encoding='utf-8',
		)
	return baseline, candidate


def _dataset(
	*,
	split: str,
	count: int = 12,
	features: np.ndarray | None = None,
	token_xyz: np.ndarray | None = None,
) -> F3LithologyTokenDataset:
	labels = np.asarray(([0, 1, 2] * ((count + 2) // 3))[:count], dtype=np.int64)
	return F3LithologyTokenDataset(
		features=(
			np.arange(count * 4, dtype=np.float32).reshape(count, 4)
			if features is None
			else features
		),
		labels=labels,
		survey_id=np.asarray(['f3'] * count),
		split=np.asarray([split] * count),
		slice_type=np.asarray(['inline'] * count),
		slice_index=np.arange(count, dtype=np.int64),
		token_xyz=(
			_train_token_xyz(count=count)
			if token_xyz is None
			else np.asarray(token_xyz, dtype=np.int64)
		),
		voxel_center_xyz=np.asarray(_train_token_xyz(count=count), dtype=np.float32)
		+ 0.5,
		majority_fraction=np.full(count, 0.9, dtype=np.float32),
		labeled_fraction=np.full(count, 1.0, dtype=np.float32),
		metadata={'fixture': True},
	)


def _train_token_xyz(
	*,
	count: int = 12,
	mismatch_index: int | None = None,
) -> np.ndarray:
	token_xyz = np.stack(
		[
			np.arange(count, dtype=np.int64),
			np.zeros(count, dtype=np.int64),
			np.zeros(count, dtype=np.int64),
		],
		axis=1,
	)
	if mismatch_index is not None:
		token_xyz[mismatch_index] = np.asarray([999, 999, 999], dtype=np.int64)
	return token_xyz
