from __future__ import annotations

import csv
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.f3.lithology import m2_results
from seis_ssl_cluster.f3.lithology.m2_results import (
	F3StratHMMM2PublishConfig,
	F3StratHMMM2ResultsConfig,
	consolidate_f3_strat_hmm_m2_results,
	publish_f3_strat_hmm_m2_results,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_m2a_go_writes_complete_lightweight_report(tmp_path: Path) -> None:
	pytest.importorskip('matplotlib').use('Agg', force=True)
	config = _fixture(tmp_path)
	result = consolidate_f3_strat_hmm_m2_results(config)
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))

	assert result.decision == 'go'
	assert payload['decision']['reason_codes'] == ['all_go_conditions_met']
	assert payload['decision']['evidence']['pareto_improved_class_ids'] == [3]
	assert payload['split_index']['joint_win_rate_macro_f1_mean_iou'] == 2 / 3
	assert {path.name for path in result.figure_paths} == {
		'single_run_metric_comparison.png',
		'label_budget_delta_curves.png',
		'split_index_deltas.png',
		'monitored_class_deltas.png',
	}
	assert all(
		path.stat().st_size > 0 for path in (*result.figure_paths, *result.table_paths)
	)


@pytest.mark.parametrize(
	('mutation', 'expected'),
	[
		('negative_full_balanced_accuracy', 'hold'),
		('split_majority_missing', 'hold'),
		('fully_negative', 'stop'),
	],
)
def test_m2a_decision_cases(tmp_path: Path, mutation: str, expected: str) -> None:
	pytest.importorskip('matplotlib').use('Agg', force=True)
	config = _fixture(tmp_path)
	if mutation == 'negative_full_balanced_accuracy':
		_rewrite_comparison(config.baseline_comparison_csv, candidate_balanced=0.59)
	elif mutation == 'split_majority_missing':
		_write_split(
			config.split_index_suite_root, (0.02, 0.02, 0.02, -0.01, -0.01, 0.0)
		)
	else:
		_write_budgets(config.label_budget_suite_root, delta=-0.02)
		_write_split(
			config.split_index_suite_root,
			(-0.02, -0.01, -0.02, -0.02, -0.01, -0.02),
		)
	result = consolidate_f3_strat_hmm_m2_results(config)
	assert result.decision == expected


def test_m2a_rejects_model_identity_mismatch(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	payload = json.loads(config.m2a_metrics_json.read_text(encoding='utf-8'))
	payload['model']['tag'] = 'wrong-model'
	_write_json(config.m2a_metrics_json, payload)
	with pytest.raises(ValueError, match='identity mismatch'):
		consolidate_f3_strat_hmm_m2_results(config)


@pytest.mark.parametrize('class_ids', [(3,), (5,), (3, 5, 7)])
def test_m2a_requires_exact_monitored_classes(
	tmp_path: Path, class_ids: tuple[int, ...]
) -> None:
	config = replace(_fixture(tmp_path), monitored_class_ids=class_ids)
	with pytest.raises(ValueError, match='required classes 3 and 5'):
		consolidate_f3_strat_hmm_m2_results(config)


def test_m2a_rejects_missing_required_budget(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	path = config.label_budget_suite_root / 'reports' / 'paired_deltas.csv'
	rows = [row for row in _read_csv(path) if row['budget_id'] != 'cap50']
	_write_csv(path, tuple(rows[0]), rows)
	with pytest.raises(ValueError, match='missing required budgets'):
		consolidate_f3_strat_hmm_m2_results(config)


def test_m2a_rejects_split_evidence_missing_from_csv(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	path = config.split_index_suite_root / 'reports' / 'split_paired_deltas.csv'
	rows = _read_csv(path)[:-1]
	_write_csv(path, tuple(rows[0]), rows)
	with pytest.raises(ValueError, match='split/index evidence does not match'):
		consolidate_f3_strat_hmm_m2_results(config)


@pytest.mark.parametrize('condition_kind', ['full_budget', 'budget_seed', 'split'])
def test_m2a_rejects_incomplete_preregistered_inventory(
	tmp_path: Path, condition_kind: str
) -> None:
	config = _fixture(tmp_path)
	if condition_kind in {'full_budget', 'budget_seed'}:
		csv_path = config.label_budget_suite_root / 'reports' / 'paired_deltas.csv'
		manifest_path = config.label_budget_suite_root / 'suite_manifest.json'
		if condition_kind == 'full_budget':
			csv_rows = [
				row for row in _read_csv(csv_path) if row['budget_id'] != 'full'
			]
		else:
			csv_rows = [
				row
				for row in _read_csv(csv_path)
				if not (
					row['budget_id'] == 'cap25' and row['subsample_seed'] == '4'
				)
			]
	else:
		csv_path = config.split_index_suite_root / 'reports' / 'split_paired_deltas.csv'
		manifest_path = config.split_index_suite_root / 'split_dataset_manifest.json'
		csv_rows = [
			row for row in _read_csv(csv_path) if row['split_id'] != 'split_005'
		]
	_write_csv(csv_path, tuple(csv_rows[0]), csv_rows)
	manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
	if condition_kind == 'full_budget':
		manifest['rows'] = [
			row for row in manifest['rows'] if row['budget_id'] != 'full'
		]
	elif condition_kind == 'budget_seed':
		manifest['rows'] = [
			row
			for row in manifest['rows']
			if not (row['budget_id'] == 'cap25' and row['subsample_seed'] == 4)
		]
	else:
		manifest['rows'] = [
			row for row in manifest['rows'] if row['split_id'] != 'split_005'
		]
	_write_json(manifest_path, manifest)

	with pytest.raises(ValueError, match='preregistered condition inventory'):
		consolidate_f3_strat_hmm_m2_results(config)


def test_m2a_rejects_missing_suite_manifest(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	(config.label_budget_suite_root / 'suite_manifest.json').unlink()
	with pytest.raises(FileNotFoundError, match='suite_manifest_json'):
		consolidate_f3_strat_hmm_m2_results(config)


def test_m2a_rejects_split_manifest_identity_mismatch(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	path = config.split_index_suite_root / 'split_dataset_manifest.json'
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['rows'][0]['model_tag'] = 'wrong-model'
	_write_json(path, payload)
	with pytest.raises(ValueError, match='identity mismatch'):
		consolidate_f3_strat_hmm_m2_results(config)


@pytest.mark.parametrize(
	('suite_attribute', 'manifest_name'),
	[
		('label_budget_suite_root', 'suite_manifest.json'),
		('split_index_suite_root', 'split_dataset_manifest.json'),
	],
)
def test_m2a_rejects_paired_identity_hash_mismatch(
	tmp_path: Path, suite_attribute: str, manifest_name: str
) -> None:
	config = _fixture(tmp_path)
	path = getattr(config, suite_attribute) / manifest_name
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['rows'][1]['paired_identity_hash'] = 'mismatched'
	_write_json(path, payload)
	with pytest.raises(ValueError, match='paired_identity_hash mismatch'):
		consolidate_f3_strat_hmm_m2_results(config)


def test_m2a_publish_wrapper_enforces_exact_allowlist_and_size_guard(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	pytest.importorskip('matplotlib').use('Agg', force=True)
	result = consolidate_f3_strat_hmm_m2_results(_fixture(tmp_path))
	publish_root = tmp_path / 'results'
	monkeypatch.setattr(m2_results, 'DEFAULT_RESULTS_ROOT', publish_root)
	publish_config = F3StratHMMM2PublishConfig(
		enabled=True,
		output_dir=publish_root / 'm2a',
		max_file_size_bytes=10 * 1024 * 1024,
	)

	manifest = publish_f3_strat_hmm_m2_results(result, publish_config)
	assert manifest is not None
	assert {
		item.target.relative_to(publish_config.output_dir).as_posix()
		for item in manifest.items
	} == {
		'm2a_results_summary.md',
		'm2a_results_summary.json',
		*(f'tables/{name}' for name in m2_results.M2_RESULTS_TABLE_NAMES),
		*(f'figures/{name}' for name in m2_results.M2_RESULTS_FIGURE_NAMES),
	}

	wrong_tables = replace(
		result,
		table_paths=(
			*result.table_paths[:-1],
			result.table_paths[-1].with_name('extra.csv'),
		),
	)
	with pytest.raises(ValueError, match='table allowlist must be exactly'):
		publish_f3_strat_hmm_m2_results(wrong_tables, publish_config)
	with pytest.raises(ValueError, match='exceeds max_file_size_bytes'):
		publish_f3_strat_hmm_m2_results(
			result, replace(publish_config, max_file_size_bytes=1)
		)


def _fixture(tmp_path: Path) -> F3StratHMMM2ResultsConfig:
	comparison = tmp_path / 'comparison.csv'
	label_root = tmp_path / 'label'
	split_root = tmp_path / 'split'
	m1_metrics = tmp_path / 'm1_metrics.json'
	m2_metrics = tmp_path / 'm2_metrics.json'
	_rewrite_comparison(comparison, candidate_balanced=0.62)
	_write_budgets(label_root, delta=0.02)
	_write_split(split_root, (0.03, 0.02, 0.01, 0.01, -0.01, -0.01))
	_write_json(m1_metrics, _metrics('M1', f1=(0.40, 0.50), iou=(0.30, 0.40)))
	_write_json(m2_metrics, _metrics('M2-A', f1=(0.42, 0.48), iou=(0.31, 0.39)))
	return F3StratHMMM2ResultsConfig(
		baseline_comparison_csv=comparison,
		m1_metrics_json=m1_metrics,
		m2a_metrics_json=m2_metrics,
		label_budget_suite_root=label_root,
		split_index_suite_root=split_root,
		output_dir=tmp_path / 'output',
	)


def _rewrite_comparison(path: Path, *, candidate_balanced: float) -> None:
	rows = []
	for model, offset, balanced in (
		('M1', 0.0, 0.60),
		('M2-A', 0.04, candidate_balanced),
	):
		rows.append(
			{
				'MODEL_TAG': model,
				'accuracy': 0.65 + offset,
				'balanced_accuracy': balanced,
				'macro_f1': 0.55 + offset,
				'weighted_f1': 0.64 + offset,
				'mean_iou': 0.45 + offset,
			}
		)
	_write_csv(path, tuple(rows[0]), rows)


def _write_budgets(root: Path, *, delta: float) -> None:
	rows = [
		{
			'budget_id': budget,
			'per_class_cap': '' if budget == 'full' else budget.removeprefix('cap'),
			'subsample_seed': str(seed),
			'delta_macro_f1': delta,
			'delta_mean_iou': delta,
			'delta_balanced_accuracy': delta,
		}
		for budget in ('cap25', 'cap50', 'cap100', 'full')
		for seed in range(5)
	]
	_write_csv(root / 'reports' / 'paired_deltas.csv', tuple(rows[0]), rows)
	_write_suite_manifest(
		root / 'suite_manifest.json',
		rows,
		condition_keys=('budget_id', 'subsample_seed'),
	)


def _write_split(root: Path, deltas: tuple[float, ...]) -> None:
	rows = [
		{
			'split_id': f'split_{index:03d}',
			'delta_macro_f1': delta,
			'delta_mean_iou': delta,
			'delta_balanced_accuracy': delta,
		}
		for index, delta in enumerate(deltas)
	]
	_write_csv(root / 'reports' / 'split_paired_deltas.csv', tuple(rows[0]), rows)
	_write_suite_manifest(
		root / 'split_dataset_manifest.json', rows, condition_keys=('split_id',)
	)


def _write_suite_manifest(
	path: Path,
	rows: list[dict[str, object]],
	*,
	condition_keys: tuple[str, ...],
) -> None:
	manifest_rows = []
	for row in rows:
		for role, model in (('baseline', 'M1'), ('candidate', 'M2-A')):
			condition = {
				key: int(row[key]) if key == 'subsample_seed' else row[key]
				for key in condition_keys
			}
			manifest_rows.append(
				{
					**condition,
					'model_role': role,
					'model_tag': model,
					'paired_identity_hash': '-'.join(
						str(row[key]) for key in condition_keys
					),
				}
			)
	_write_json(path, {'rows': manifest_rows})


def _metrics(
	model: str, *, f1: tuple[float, float], iou: tuple[float, float]
) -> dict[str, object]:
	return {
		'model': {'tag': model},
		'per_class_f1': {'3': f1[0], '5': f1[1]},
		'per_class_iou': {'3': iou[0], '5': iou[1]},
		'per_class_support': {'3': 20, '5': 10},
	}


def _write_csv(
	path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline='', encoding='utf-8') as handle:
		return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')
