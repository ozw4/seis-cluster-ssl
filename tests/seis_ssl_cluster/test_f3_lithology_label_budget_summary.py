from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from proc.seis_ssl_cluster.summarize_f3_lithology_label_budget_robustness import (
	summarize_label_budget_robustness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = (
	REPO_ROOT
	/ 'proc'
	/ 'seis_ssl_cluster'
	/ ('summarize_f3_lithology_label_budget_robustness.py')
)


def test_summary_writes_paired_metrics_and_deltas(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)

	result = summarize_label_budget_robustness(suite_root)

	assert result.paired_metric_count == 8
	assert result.pair_count == 4
	assert result.budget_count == 2
	for path in (
		result.paired_metrics_csv,
		result.paired_deltas_csv,
		result.summary_by_budget_csv,
		result.summary_markdown,
	):
		assert path.is_file()

	paired_rows = _read_csv(result.paired_metrics_csv)
	assert paired_rows[0].keys() == {
		'model_role',
		'model_tag',
		'budget_id',
		'per_class_cap',
		'subsample_seed',
		'train_token_count',
		'validation_token_count',
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
	candidate_cap2_seed0 = next(
		row
		for row in paired_rows
		if row['model_role'] == 'candidate'
		and row['budget_id'] == 'cap2'
		and row['subsample_seed'] == '0'
	)
	assert candidate_cap2_seed0['model_tag'] == 'strat_hmm_m1'
	assert candidate_cap2_seed0['macro_f1'] == '0.72'
	assert candidate_cap2_seed0['class_5_f1'] == '0.54'

	delta_rows = _read_csv(result.paired_deltas_csv)
	assert delta_rows[0].keys() == {
		'budget_id',
		'per_class_cap',
		'subsample_seed',
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
	}
	cap2_seed0 = next(
		row
		for row in delta_rows
		if row['budget_id'] == 'cap2' and row['subsample_seed'] == '0'
	)
	assert float(cap2_seed0['delta_macro_f1']) == pytest.approx(0.12)
	assert float(cap2_seed0['delta_mean_iou']) == pytest.approx(0.11)
	assert float(cap2_seed0['delta_class_3_f1']) == pytest.approx(-0.03)


def test_summary_by_budget_win_rates(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)

	result = summarize_label_budget_robustness(suite_root)

	rows = _read_csv(result.summary_by_budget_csv)
	cap2 = next(row for row in rows if row['budget_id'] == 'cap2')
	full = next(row for row in rows if row['budget_id'] == 'full')
	assert cap2['n_pairs'] == '2'
	assert float(cap2['mean_delta_macro_f1']) == pytest.approx(0.03)
	assert float(cap2['median_delta_macro_f1']) == pytest.approx(0.03)
	assert float(cap2['win_rate_macro_f1']) == pytest.approx(0.5)
	assert float(cap2['mean_delta_mean_iou']) == pytest.approx(0.03)
	assert float(cap2['win_rate_mean_iou']) == pytest.approx(0.5)
	assert full['per_class_cap'] == ''
	assert float(full['win_rate_macro_f1']) == pytest.approx(1.0)


def test_missing_pair_fails(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)
	probe_manifest_path = suite_root / 'probe_run_manifest.json'
	probe_manifest = json.loads(probe_manifest_path.read_text(encoding='utf-8'))
	probe_manifest['rows'] = [
		row
		for row in probe_manifest['rows']
		if not (
			row['budget_id'] == 'cap2'
			and row['subsample_seed'] == 1
			and row['model_role'] == 'candidate'
		)
	]
	probe_manifest_path.write_text(
		json.dumps(probe_manifest, indent=2) + '\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='suite/probe manifest row mismatch'):
		summarize_label_budget_robustness(suite_root)


def test_non_finite_metric_fails(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)
	metrics_path = (
		suite_root / 'metrics' / 'candidate' / 'cap2' / 'seed0' / 'metrics.json'
	)
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics['macro_f1'] = float('nan')
	metrics_path.write_text(json.dumps(metrics) + '\n', encoding='utf-8')

	with pytest.raises(ValueError, match='metric must be finite'):
		summarize_label_budget_robustness(suite_root)


def test_markdown_contains_tags_budgets_and_guidance(tmp_path: Path) -> None:
	suite_root = _write_suite(tmp_path)

	result = summarize_label_budget_robustness(suite_root)

	markdown = result.summary_markdown.read_text(encoding='utf-8')
	assert 'suite: label_budget_m1_v1' in markdown
	assert 'baseline model: mae_baseline' in markdown
	assert 'candidate model: strat_hmm_m1' in markdown
	assert '| cap2 |' in markdown
	assert '| full |' in markdown
	assert 'Go if low-budget and full-budget conditions' in markdown
	assert 'Hold if only full budget wins or metrics conflict.' in markdown
	assert 'Stop if low-budget deltas are negative' in markdown
	assert 'paired label-budget robustness, not probe seed sweep' in markdown
	assert 'especially class 3 has negative mean F1 delta' in markdown
	assert 'especially class 5 has negative mean F1 delta' in markdown


def test_cli_produces_expected_files_from_synthetic_metrics(tmp_path: Path) -> None:
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

	assert 'f3_lithology_label_budget_summary.pairs: 4' in completed.stdout
	for filename in (
		'paired_metrics.csv',
		'paired_deltas.csv',
		'summary_by_budget.csv',
		'summary.md',
	):
		assert (suite_root / 'reports' / filename).is_file()


def _write_suite(tmp_path: Path) -> Path:
	suite_root = tmp_path / 'label_budget_m1_v1'
	conditions = (
		('cap2', 2, 0),
		('cap2', 2, 1),
		('full', None, 0),
		('full', None, 1),
	)
	suite_rows = []
	probe_rows = []
	for budget_id, per_class_cap, seed in conditions:
		for role in ('baseline', 'candidate'):
			model_tag = 'mae_baseline' if role == 'baseline' else 'strat_hmm_m1'
			metrics_path = (
				suite_root
				/ 'metrics'
				/ role
				/ budget_id
				/ f'seed{seed}'
				/ 'metrics.json'
			)
			_write_json(metrics_path, _metrics(role, budget_id, seed))
			paired_hash = f'hash-{budget_id}-{seed}'
			suite_rows.append(
				{
					'model_role': role,
					'model_tag': model_tag,
					'budget_id': budget_id,
					'per_class_cap': per_class_cap,
					'subsample_seed': seed,
					'token_dataset_root': str(
						suite_root / 'datasets' / role / budget_id / f'seed{seed}',
					),
					'train_tokens': str(suite_root / 'train.npz'),
					'validation_tokens': str(suite_root / 'validation.npz'),
					'metadata_json': str(suite_root / 'metadata.json'),
					'selected_train_token_count': 12 if per_class_cap else 60,
					'validation_token_count': 30,
					'paired_identity_hash': paired_hash,
				},
			)
			probe_rows.append(
				{
					'model_role': role,
					'model_tag': model_tag,
					'budget_id': budget_id,
					'per_class_cap': per_class_cap,
					'subsample_seed': seed,
					'token_dataset_root': str(
						suite_root / 'datasets' / role / budget_id / f'seed{seed}',
					),
					'probe_output_dir': str(metrics_path.parent),
					'metrics_json': str(metrics_path),
					'train_token_count': 12 if per_class_cap else 60,
					'validation_token_count': 30,
				},
			)
	_write_json(
		suite_root / 'suite_manifest.json',
		{
			'artifact_type': 'f3_lithology_label_budget_suite_manifest',
			'contract_version': 1,
			'suite': {'name': 'label_budget_m1_v1', 'output_root': str(suite_root)},
			'rows': suite_rows,
		},
	)
	_write_json(
		suite_root / 'probe_run_manifest.json',
		{
			'artifact_type': 'f3_lithology_label_budget_probe_run_manifest',
			'suite_manifest': str(suite_root / 'suite_manifest.json'),
			'probe': {'spec': 'linear_balanced_v1'},
			'rows': probe_rows,
		},
	)
	return suite_root


def _metrics(role: str, budget_id: str, seed: int) -> dict[str, object]:
	baseline = {
		'cap2': {
			0: (0.62, 0.60, 0.60, 0.58, 0.46, 0.70, 0.33, 0.61),
			1: (0.62, 0.60, 0.60, 0.58, 0.46, 0.70, 0.33, 0.61),
		},
		'full': {
			0: (0.72, 0.70, 0.70, 0.68, 0.56, 0.78, 0.43, 0.66),
			1: (0.72, 0.70, 0.70, 0.68, 0.56, 0.78, 0.43, 0.66),
		},
	}
	candidate = {
		'cap2': {
			0: (0.74, 0.71, 0.72, 0.70, 0.57, 0.82, 0.30, 0.54),
			1: (0.58, 0.57, 0.54, 0.55, 0.41, 0.66, 0.25, 0.50),
		},
		'full': {
			0: (0.80, 0.79, 0.78, 0.77, 0.63, 0.84, 0.42, 0.65),
			1: (0.79, 0.78, 0.77, 0.76, 0.62, 0.83, 0.41, 0.64),
		},
	}
	values = (baseline if role == 'baseline' else candidate)[budget_id][seed]
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
