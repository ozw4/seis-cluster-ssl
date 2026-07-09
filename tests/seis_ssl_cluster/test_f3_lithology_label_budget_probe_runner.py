from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from proc.seis_ssl_cluster.run_f3_lithology_label_budget_probes import (
	run_f3_lithology_label_budget_probes,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_label_budget_config_from_mapping,
	f3_lithology_label_budget_probe_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	build_f3_lithology_label_budget_datasets,
	save_token_dataset_npz,
)
from seis_ssl_cluster.f3.lithology.token_dataset import F3LithologyTokenDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / 'proc' / 'seis_ssl_cluster' / (
	'run_f3_lithology_label_budget_probes.py'
)


def test_dry_run_does_not_write_probe_outputs(tmp_path: Path) -> None:
	config_path, output_root = _write_suite_and_runner_config(tmp_path)
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

	assert 'condition count: 4' in completed.stdout
	assert 'probe type: logistic_regression' in completed.stdout
	assert 'probe random_state: 42' in completed.stdout
	assert 'execution: dry-run; probe training skipped' in completed.stdout
	assert not (output_root / 'probes').exists()
	assert not (output_root / 'probe_run_manifest.json').exists()


def test_runner_trains_all_manifest_rows_and_writes_fixed_random_state(
	tmp_path: Path,
) -> None:
	config_path, output_root = _write_suite_and_runner_config(tmp_path)
	config = f3_lithology_label_budget_probe_config_from_mapping(
		_load_yaml(config_path),
	)

	manifest_path = run_f3_lithology_label_budget_probes(config)

	run_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
	assert len(run_manifest['rows']) == 4
	for row in run_manifest['rows']:
		metrics_json = Path(row['metrics_json'])
		assert metrics_json.is_file()
		resolved = json.loads(
			(metrics_json.parent / 'probe_config_resolved.json').read_text(
				encoding='utf-8',
			),
		)
		assert resolved['probe']['random_state'] == 42
	assert (
		output_root
		/ 'probes/model=baseline_model/budget=cap4/subsample_seed=0'
		/ 'linear_balanced_v1/metrics.json'
	).is_file()


def test_only_missing_skips_existing_metrics(tmp_path: Path) -> None:
	config_path, _output_root = _write_suite_and_runner_config(tmp_path)
	config = f3_lithology_label_budget_probe_config_from_mapping(
		_load_yaml(config_path),
	)
	run_f3_lithology_label_budget_probes(config)
	metrics_json = config.probe_configs[0].outputs.metrics_json
	metrics_json.write_text(
		json.dumps({'sentinel': True}) + '\n',
		encoding='utf-8',
	)

	run_f3_lithology_label_budget_probes(config, only_missing=True)

	assert json.loads(metrics_json.read_text(encoding='utf-8')) == {'sentinel': True}


def test_mismatched_paired_hashes_fail_before_training(tmp_path: Path) -> None:
	config_path, output_root = _write_suite_and_runner_config(tmp_path)
	manifest_path = output_root / 'suite_manifest.json'
	manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
	for row in manifest['rows']:
		if row['model_role'] == 'candidate':
			row['paired_identity_hash'] = 'mismatched'
			break
	manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
	config = f3_lithology_label_budget_probe_config_from_mapping(
		_load_yaml(config_path),
	)

	with pytest.raises(ValueError, match='paired_identity_hash mismatch'):
		run_f3_lithology_label_budget_probes(config)

	assert not (output_root / 'probes').exists()


def _write_suite_and_runner_config(tmp_path: Path) -> tuple[Path, Path]:
	baseline, candidate = _write_paired_sources(tmp_path)
	output_root = tmp_path / 'out'
	label_budget_config = f3_lithology_label_budget_config_from_mapping(
		{
			'paths': {'artifact_root': str(tmp_path)},
			'suite': {
				'name': 'label_budget_probe_fixture',
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
				'per_class_caps': [4],
				'subsample_seeds': [0, 1],
				'require_all_classes': True,
			},
			'validation': {'reuse_full_validation': True},
			'outputs': {'overwrite': False},
		},
	)
	build_f3_lithology_label_budget_datasets(label_budget_config)
	class_info = _write_class_info(tmp_path)
	config_path = tmp_path / 'probe_runner.yaml'
	config_path.write_text(
		f"""
suite:
  manifest: {output_root / 'suite_manifest.json'}
  output_root: {output_root}
probe:
  spec: linear_balanced_v1
  type: logistic_regression
  feature_scaling: standard
  class_weight: balanced
  max_iter: 200
labels:
  class_info: {class_info}
evaluation:
  metrics: [accuracy, balanced_accuracy, macro_f1, weighted_f1, mean_iou]
  figure:
    dpi: 100
outputs:
  overwrite: false
""",
		encoding='utf-8',
	)
	return config_path, output_root


def _write_paired_sources(tmp_path: Path) -> tuple[Path, Path]:
	baseline = tmp_path / 'baseline_tokens'
	candidate = tmp_path / 'candidate_tokens'
	train = _dataset(split='train', count=18, token_xyz_start=0)
	validation = _dataset(split='validation', count=9, token_xyz_start=10_000)
	candidate_train = _dataset(
		split='train',
		count=18,
		token_xyz_start=0,
		feature_offset=0.2,
	)
	candidate_validation = _dataset(
		split='validation',
		count=9,
		token_xyz_start=10_000,
		feature_offset=0.2,
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
	count: int,
	token_xyz_start: int,
	feature_offset: float = 0.0,
) -> F3LithologyTokenDataset:
	labels = np.asarray(([0, 1, 2] * ((count + 2) // 3))[:count], dtype=np.int64)
	features = np.column_stack(
		(
			labels.astype(np.float32) * 4.0 + feature_offset,
			labels.astype(np.float32) * -3.0 + feature_offset,
		),
	).astype(np.float32)
	token_indices = np.arange(token_xyz_start, token_xyz_start + count, dtype=np.int64)
	return F3LithologyTokenDataset(
		features=features,
		labels=labels,
		survey_id=np.asarray(['f3'] * count),
		split=np.asarray([split] * count),
		slice_type=np.asarray(['inline'] * count),
		slice_index=np.arange(count, dtype=np.int64),
		token_xyz=np.column_stack(
			(
				token_indices,
				np.zeros(count, dtype=np.int64),
				np.zeros(count, dtype=np.int64),
			),
		),
		voxel_center_xyz=np.zeros((count, 3), dtype=np.float32),
		majority_fraction=np.full(count, 0.9, dtype=np.float32),
		labeled_fraction=np.ones(count, dtype=np.float32),
		metadata={'fixture': True},
	)


def _write_class_info(tmp_path: Path) -> Path:
	path = tmp_path / 'class_info.json'
	path.write_text(
		json.dumps(
			{
				'classes': [
					{'class_id': 0, 'class_name': 'class 0', 'rgb': [230, 159, 0]},
					{'class_id': 1, 'class_name': 'class 1', 'rgb': [86, 180, 233]},
					{'class_id': 2, 'class_name': 'class 2', 'rgb': [0, 158, 115]},
				],
			},
		)
		+ '\n',
		encoding='utf-8',
	)
	return path


def _load_yaml(path: Path) -> dict[str, object]:
	return load_config(path)
