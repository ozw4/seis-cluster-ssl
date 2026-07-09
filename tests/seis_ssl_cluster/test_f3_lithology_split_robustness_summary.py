from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from proc.seis_ssl_cluster.summarize_f3_lithology_split_robustness import (
	summarize_split_robustness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = (
	REPO_ROOT
	/ 'proc'
	/ 'seis_ssl_cluster'
	/ 'summarize_f3_lithology_split_robustness.py'
)


def test_summary_writes_paired_metrics_and_deltas(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)

	result = summarize_split_robustness(suite_root)

	assert result.paired_metric_count == 4
	assert result.pair_count == 2
	for path in (
		result.paired_metrics_csv,
		result.paired_deltas_csv,
		result.summary_csv,
		result.summary_markdown,
	):
		assert path.is_file()

	paired_rows = _read_csv(result.paired_metrics_csv)
	assert paired_rows[0].keys() == {
		'split_id',
		'model_role',
		'model_tag',
		'train_token_count',
		'validation_token_count',
		'validation_class_counts',
		'accuracy',
		'balanced_accuracy',
		'macro_f1',
		'weighted_f1',
		'mean_iou',
		'class_0_f1',
		'class_3_f1',
		'class_5_f1',
		'metrics_json',
	}
	candidate_split_000 = next(
		row
		for row in paired_rows
		if row['model_role'] == 'candidate' and row['split_id'] == 'split_000'
	)
	assert candidate_split_000['model_tag'] == 'strat_hmm_m1'
	assert candidate_split_000['macro_f1'] == '0.72'
	assert candidate_split_000['class_5_f1'] == '0.54'

	delta_rows = _read_csv(result.paired_deltas_csv)
	assert delta_rows[0].keys() == {
		'split_id',
		'delta_accuracy',
		'delta_balanced_accuracy',
		'delta_macro_f1',
		'delta_weighted_f1',
		'delta_mean_iou',
		'delta_class_0_f1',
		'delta_class_3_f1',
		'delta_class_5_f1',
		'baseline_metrics_json',
		'candidate_metrics_json',
		'validation_class_counts',
		'validation_slices',
	}
	split_000 = next(row for row in delta_rows if row['split_id'] == 'split_000')
	assert float(split_000['delta_macro_f1']) == pytest.approx(0.12)
	assert float(split_000['delta_mean_iou']) == pytest.approx(0.11)
	assert float(split_000['delta_class_3_f1']) == pytest.approx(-0.03)


def test_split_summary_win_rates(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)

	result = summarize_split_robustness(suite_root)

	rows = _read_csv(result.summary_csv)
	assert len(rows) == 1
	summary = rows[0]
	assert summary['n_splits'] == '2'
	assert float(summary['mean_delta_macro_f1']) == pytest.approx(0.03)
	assert float(summary['median_delta_macro_f1']) == pytest.approx(0.03)
	assert float(summary['win_rate_macro_f1']) == pytest.approx(0.5)
	assert float(summary['mean_delta_mean_iou']) == pytest.approx(0.03)
	assert float(summary['win_rate_mean_iou']) == pytest.approx(0.5)


def test_missing_split_pair_fails(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)
	probe_manifest_path = suite_root / 'split_probe_run_manifest.json'
	probe_manifest = json.loads(probe_manifest_path.read_text(encoding='utf-8'))
	probe_manifest['rows'] = [
		row
		for row in probe_manifest['rows']
		if not (
			row['split_id'] == 'split_001'
			and row['model_role'] == 'candidate'
		)
	]
	probe_manifest_path.write_text(
		json.dumps(probe_manifest, indent=2) + '\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='manifest row mismatch'):
		summarize_split_robustness(suite_root)


def test_inventory_split_without_dataset_or_probe_rows_fails(
	tmp_path: Path,
) -> None:
	suite_root = _write_suite(tmp_path)
	_append_inventory_split(suite_root, 'split_002')

	with pytest.raises(ValueError, match='missing_manifest_splits') as exc_info:
		summarize_split_robustness(suite_root)
	assert 'split_002' in str(exc_info.value)


def test_validation_class_counts_are_propagated(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)

	result = summarize_split_robustness(suite_root)

	delta_rows = _read_csv(result.paired_deltas_csv)
	split_001 = next(row for row in delta_rows if row['split_id'] == 'split_001')
	assert json.loads(split_001['validation_class_counts']) == {
		'0': 10,
		'3': 2,
		'5': 1,
	}


def test_markdown_contains_split_ids_and_go_hold_stop_guidance(
	tmp_path: Path,
) -> None:
	suite_root = _write_suite(tmp_path)

	result = summarize_split_robustness(suite_root)

	markdown = result.summary_markdown.read_text(encoding='utf-8')
	assert 'baseline model: mae_baseline' in markdown
	assert 'candidate model: strat_hmm_m1' in markdown
	assert '| split_000 |' in markdown
	assert '| split_001 |' in markdown
	assert 'split_000: [{"slice_index":100,"slice_type":"inline"}]' in markdown
	assert 'split_001: {"0":10,"3":2,"5":1}' in markdown
	assert 'Go if strat-HMM wins macro_f1 and mean_iou' in markdown
	assert 'Hold if wins are split-dependent or balanced_accuracy degrades.' in markdown
	assert 'Stop if improvements vanish outside the original split.' in markdown
	assert 'class 3 validation support is small' in markdown
	assert 'class 5 F1 deltas are consistently negative' in markdown


def test_cli_produces_expected_files_from_synthetic_input(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), env.get('PYTHONPATH', '')),
	)

	completed = subprocess.run(  # noqa: S603
		[sys.executable, str(CLI), '--suite-root', str(suite_root)],
		cwd=REPO_ROOT,
		env=env,
		text=True,
		capture_output=True,
		check=True,
		timeout=30,
	)

	assert 'f3_lithology_split_summary.pairs: 2' in completed.stdout
	for filename in (
		'split_paired_metrics.csv',
		'split_paired_deltas.csv',
		'split_summary.csv',
		'summary.md',
	):
		assert (suite_root / 'reports' / filename).is_file()


def _write_suite(tmp_path: Path) -> Path:
	suite_root = tmp_path / 'split_m1_v1'
	split_ids = ('split_000', 'split_001')
	inventory_rows = []
	dataset_rows = []
	probe_rows = []
	for split_id in split_ids:
		_write_split_metadata(suite_root, split_id)
		inventory_rows.append(
			{
				'split_id': split_id,
				'random_seed': 0 if split_id == 'split_000' else 1,
				'png_label_inventory': str(
					suite_root
					/ 'split_inventories'
					/ split_id
					/ 'png_label_inventory.csv',
				),
				'split_metadata': str(
					suite_root
					/ 'split_inventories'
					/ split_id
					/ 'split_metadata.json',
				),
				'validation_slice_count': 1,
			},
		)
		for role in ('baseline', 'candidate'):
			model_tag = 'mae_baseline' if role == 'baseline' else 'strat_hmm_m1'
			metrics_path = (
				suite_root / 'metrics' / f'split={split_id}' / role / 'metrics.json'
			)
			_write_json(metrics_path, _metrics(role, split_id))
			class_counts_csv = (
				suite_root
				/ 'datasets'
				/ f'split={split_id}'
				/ f'model={model_tag}'
				/ 'token_dataset'
				/ 'class_counts.csv'
			)
			_write_class_counts(class_counts_csv)
			paired_hash = f'hash-{split_id}'
			common = {
				'split_id': split_id,
				'model_role': role,
				'model_tag': model_tag,
				'token_dataset_root': str(class_counts_csv.parent),
				'train_token_count': 50,
				'validation_token_count': 13,
				'paired_identity_hash': paired_hash,
			}
			dataset_rows.append(
				{
					**common,
					'train_tokens': str(class_counts_csv.parent / 'train_tokens.npz'),
					'validation_tokens': str(
						class_counts_csv.parent / 'validation_tokens.npz',
					),
					'metadata_json': str(
						class_counts_csv.parent / 'token_dataset_metadata.json',
					),
					'class_counts_csv': str(class_counts_csv),
				},
			)
			probe_rows.append(
				{
					**common,
					'probe_output_dir': str(metrics_path.parent),
					'metrics_json': str(metrics_path),
				},
			)
	_write_json(
		suite_root / 'split_inventory_manifest.json',
		{
			'artifact_type': 'f3_lithology_split_inventory_manifest',
			'contract_version': 'f3_lithology_robustness_m1_v1',
			'rows': inventory_rows,
		},
	)
	_write_json(
		suite_root / 'split_dataset_manifest.json',
		{
			'artifact_type': 'f3_lithology_split_sweep_token_dataset_manifest',
			'contract_version': 'f3_lithology_robustness_m1_v1',
			'suite': {
				'output_root': str(suite_root),
				'split_inventory_manifest': str(
					suite_root / 'split_inventory_manifest.json',
				),
			},
			'rows': dataset_rows,
		},
	)
	_write_json(
		suite_root / 'split_probe_run_manifest.json',
		{
			'artifact_type': 'f3_lithology_split_probe_run_manifest',
			'rows': probe_rows,
		},
	)
	return suite_root


def _append_inventory_split(suite_root: Path, split_id: str) -> None:
	_write_split_metadata(suite_root, split_id)
	manifest_path = suite_root / 'split_inventory_manifest.json'
	manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
	manifest['rows'].append(
		{
			'split_id': split_id,
			'random_seed': 2,
			'png_label_inventory': str(
				suite_root
				/ 'split_inventories'
				/ split_id
				/ 'png_label_inventory.csv',
			),
			'split_metadata': str(
				suite_root
				/ 'split_inventories'
				/ split_id
				/ 'split_metadata.json',
			),
			'validation_slice_count': 1,
		},
	)
	manifest_path.write_text(
		json.dumps(manifest, indent=2) + '\n',
		encoding='utf-8',
	)


def _write_split_metadata(suite_root: Path, split_id: str) -> None:
	slice_index = 100 if split_id == 'split_000' else 201
	_write_json(
		suite_root / 'split_inventories' / split_id / 'split_metadata.json',
		{
			'split_id': split_id,
			'random_seed': 0 if split_id == 'split_000' else 1,
			'validation_slices': [
				{'slice_type': 'inline', 'slice_index': slice_index},
			],
			'validation_class_counts_estimated': {'0': 10, '3': 2, '5': 1},
		},
	)
	inventory = suite_root / 'split_inventories' / split_id / 'png_label_inventory.csv'
	inventory.parent.mkdir(parents=True, exist_ok=True)
	inventory.write_text(
		'relative_path,absolute_path,split,slice_type,slice_index\n',
		encoding='utf-8',
	)


def _write_class_counts(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=('split', 'class_id', 'class_name', 'count', 'fraction'),
		)
		writer.writeheader()
		for class_id, count in (('0', 10), ('3', 2), ('5', 1)):
			writer.writerow(
				{
					'split': 'validation',
					'class_id': class_id,
					'class_name': f'class_{class_id}',
					'count': count,
					'fraction': 0.0,
				},
			)


def _metrics(role: str, split_id: str) -> dict[str, object]:
	baseline = {
		'split_000': (0.62, 0.60, 0.60, 0.58, 0.46, 0.70, 0.33, 0.61),
		'split_001': (0.62, 0.60, 0.60, 0.58, 0.46, 0.70, 0.33, 0.61),
	}
	candidate = {
		'split_000': (0.74, 0.71, 0.72, 0.70, 0.57, 0.82, 0.30, 0.54),
		'split_001': (0.58, 0.57, 0.54, 0.55, 0.41, 0.66, 0.25, 0.50),
	}
	values = (baseline if role == 'baseline' else candidate)[split_id]
	return {
		'accuracy': values[0],
		'balanced_accuracy': values[1],
		'macro_f1': values[2],
		'weighted_f1': values[3],
		'mean_iou': values[4],
		'per_class_f1': {
			'0': values[5],
			'3': values[6],
			'5': values[7],
		},
	}


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline='', encoding='utf-8') as handle:
		return list(csv.DictReader(handle))
