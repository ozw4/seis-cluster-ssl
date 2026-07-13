from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from proc.seis_ssl_cluster.run_f3_lithology_split_sweep_probes import (
	f3_lithology_split_sweep_probe_config_from_mapping,
	run_f3_lithology_split_sweep_probes,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.robustness import (
	paired_token_identity_hash,
	save_token_dataset_npz,
)
from seis_ssl_cluster.f3.lithology.token_dataset import F3LithologyTokenDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / 'proc' / 'seis_ssl_cluster' / (
	'run_f3_lithology_split_sweep_probes.py'
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

	assert 'row count: 4' in completed.stdout
	assert 'expected run count: 4' in completed.stdout
	assert 'probe type: logistic_regression' in completed.stdout
	assert 'probe random_state: 42' in completed.stdout
	assert 'execution: dry-run; probe training skipped' in completed.stdout
	assert not (output_root / 'probes').exists()
	assert not (output_root / 'split_probe_run_manifest.json').exists()


def test_runner_trains_all_split_model_rows_and_writes_fixed_random_state(
	tmp_path: Path,
) -> None:
	config_path, output_root = _write_suite_and_runner_config(tmp_path)
	config = f3_lithology_split_sweep_probe_config_from_mapping(
		_load_yaml(config_path),
	)

	manifest_path = run_f3_lithology_split_sweep_probes(config)

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
		assert row['paired_identity_hash']
		assert row['probe_spec'] == 'linear_balanced_v1'
		for key, name in (
			('probe_joblib', 'probe.joblib'),
			('scaler_joblib', 'scaler.joblib'),
		):
			assert row[key]['path'] == str(metrics_json.parent / name)
			assert row[key]['sha256']
	assert (
		output_root
		/ 'probes/split=split_000/model=baseline_model'
		/ 'linear_balanced_v1/metrics.json'
	).is_file()


def test_missing_split_pair_fails_before_training(tmp_path: Path) -> None:
	config_path, output_root = _write_suite_and_runner_config(tmp_path)
	manifest_path = output_root / 'split_dataset_manifest.json'
	manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
	manifest['rows'] = [
		row for row in manifest['rows'] if row['model_role'] != 'candidate'
	]
	manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
	config = f3_lithology_split_sweep_probe_config_from_mapping(
		_load_yaml(config_path),
	)

	with pytest.raises(ValueError, match='requires baseline and candidate rows'):
		run_f3_lithology_split_sweep_probes(config)

	assert not (output_root / 'probes').exists()


def test_mismatched_paired_hashes_fail_before_training(tmp_path: Path) -> None:
	config_path, output_root = _write_suite_and_runner_config(tmp_path)
	manifest_path = output_root / 'split_dataset_manifest.json'
	manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
	for row in manifest['rows']:
		if row['split_id'] == 'split_001' and row['model_role'] == 'candidate':
			row['paired_identity_hash'] = 'mismatched'
			break
	manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
	config = f3_lithology_split_sweep_probe_config_from_mapping(
		_load_yaml(config_path),
	)

	with pytest.raises(ValueError, match='paired_identity_hash mismatch'):
		run_f3_lithology_split_sweep_probes(config)

	assert not (output_root / 'probes').exists()


def test_only_missing_skips_existing_metrics(tmp_path: Path) -> None:
	config_path, _output_root = _write_suite_and_runner_config(tmp_path)
	config = f3_lithology_split_sweep_probe_config_from_mapping(
		_load_yaml(config_path),
	)
	run_f3_lithology_split_sweep_probes(config)
	metrics_json = config.probe_configs[0].outputs.metrics_json
	metrics_json.write_text(
		json.dumps({'sentinel': True}) + '\n',
		encoding='utf-8',
	)

	run_f3_lithology_split_sweep_probes(config, only_missing=True)

	assert json.loads(metrics_json.read_text(encoding='utf-8')) == {'sentinel': True}


def test_only_missing_runs_missing_rows_when_run_manifest_exists(
	tmp_path: Path,
) -> None:
	config_path, output_root = _write_suite_and_runner_config(tmp_path)
	config = f3_lithology_split_sweep_probe_config_from_mapping(
		_load_yaml(config_path),
	)
	run_f3_lithology_split_sweep_probes(config)
	existing_metrics_json = config.probe_configs[0].outputs.metrics_json
	existing_metrics_json.write_text(
		json.dumps({'sentinel': True}) + '\n',
		encoding='utf-8',
	)
	missing_config = config.probe_configs[1]
	shutil.rmtree(missing_config.outputs.output_dir)

	run_f3_lithology_split_sweep_probes(config, only_missing=True)

	assert json.loads(existing_metrics_json.read_text(encoding='utf-8')) == {
		'sentinel': True,
	}
	assert missing_config.outputs.metrics_json.is_file()
	manifest = json.loads(
		(output_root / 'split_probe_run_manifest.json').read_text(encoding='utf-8'),
	)
	assert len(manifest['rows']) == 4


def test_existing_run_manifest_refuses_overwrite_before_training(
	tmp_path: Path,
) -> None:
	config_path, output_root = _write_suite_and_runner_config(tmp_path)
	manifest_path = output_root / 'split_probe_run_manifest.json'
	manifest_path.write_text('{"sentinel": true}\n', encoding='utf-8')
	config = f3_lithology_split_sweep_probe_config_from_mapping(
		_load_yaml(config_path),
	)

	with pytest.raises(FileExistsError, match='split probe run manifest'):
		run_f3_lithology_split_sweep_probes(config)

	assert json.loads(manifest_path.read_text(encoding='utf-8')) == {'sentinel': True}
	assert not (output_root / 'probes').exists()


def _write_suite_and_runner_config(tmp_path: Path) -> tuple[Path, Path]:
	output_root = tmp_path / 'out'
	class_info = _write_class_info(tmp_path)
	_write_split_dataset_manifest(output_root)
	config_path = tmp_path / 'probe_runner.yaml'
	config_path.write_text(
		f"""
suite:
  dataset_manifest: {output_root / 'split_dataset_manifest.json'}
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


def _write_split_dataset_manifest(output_root: Path) -> None:
	rows: list[dict[str, object]] = []
	for split_index, split_id in enumerate(('split_000', 'split_001')):
		baseline = _write_token_dataset_root(
			output_root,
			split_id=split_id,
			model_tag='baseline_model',
			feature_offset=0.0,
			token_xyz_start=split_index * 100_000,
		)
		candidate = _write_token_dataset_root(
			output_root,
			split_id=split_id,
			model_tag='candidate_model',
			feature_offset=0.2,
			token_xyz_start=split_index * 100_000,
		)
		paired_hash = paired_token_identity_hash(
			_dataset(split='train', count=18, token_xyz_start=split_index * 100_000),
			_dataset(
				split='validation',
				count=9,
				token_xyz_start=split_index * 100_000 + 10_000,
			),
		)
		for role, model_tag, root in (
			('baseline', 'baseline_model', baseline),
			('candidate', 'candidate_model', candidate),
		):
			rows.append(
				{
					'split_id': split_id,
					'model_role': role,
					'model_tag': model_tag,
					'token_dataset_root': str(root),
					'train_tokens': str(root / 'train_tokens.npz'),
					'validation_tokens': str(root / 'validation_tokens.npz'),
					'metadata_json': str(root / 'token_dataset_metadata.json'),
					'class_counts_csv': str(root / 'class_counts.csv'),
					'train_token_count': 18,
					'validation_token_count': 9,
					'paired_identity_hash': paired_hash,
				},
			)
	manifest_path = output_root / 'split_dataset_manifest.json'
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(
		json.dumps({'artifact_type': 'fixture', 'rows': rows}, indent=2) + '\n',
		encoding='utf-8',
	)


def _write_token_dataset_root(
	output_root: Path,
	*,
	split_id: str,
	model_tag: str,
	feature_offset: float,
	token_xyz_start: int,
) -> Path:
	root = (
		output_root
		/ 'datasets'
		/ f'split={split_id}'
		/ f'model={model_tag}'
		/ 'token_dataset'
	)
	root.mkdir(parents=True, exist_ok=True)
	save_token_dataset_npz(
		_dataset(
			split='train',
			count=18,
			token_xyz_start=token_xyz_start,
			feature_offset=feature_offset,
		),
		root / 'train_tokens.npz',
	)
	save_token_dataset_npz(
		_dataset(
			split='validation',
			count=9,
			token_xyz_start=token_xyz_start + 10_000,
			feature_offset=feature_offset,
		),
		root / 'validation_tokens.npz',
	)
	(root / 'token_dataset_metadata.json').write_text(
		json.dumps({'source': model_tag}) + '\n',
		encoding='utf-8',
	)
	(root / 'class_counts.csv').write_text('split,class_id,count\n', encoding='utf-8')
	return root


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
