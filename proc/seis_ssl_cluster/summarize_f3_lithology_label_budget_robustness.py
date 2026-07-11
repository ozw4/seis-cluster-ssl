"""Summarize paired F3 lithology label-budget robustness probe metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

from seis_ssl_cluster.cli import load_config_for_cli, resolve_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_label_budget_summary_config_from_mapping,
)

STAGE = 'summarize_f3_lithology_label_budget_robustness'
OVERALL_METRICS = (
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'mean_iou',
)
SUMMARY_FIELDNAMES = (
	'budget_id',
	'per_class_cap',
	'n_pairs',
	'mean_delta_macro_f1',
	'median_delta_macro_f1',
	'std_delta_macro_f1',
	'min_delta_macro_f1',
	'max_delta_macro_f1',
	'win_rate_macro_f1',
	'mean_delta_mean_iou',
	'win_rate_mean_iou',
	'mean_delta_balanced_accuracy',
	'win_rate_balanced_accuracy',
)


@dataclass(frozen=True)
class LabelBudgetSummaryResult:
	"""Paths written by the label-budget robustness summarizer."""

	paired_metrics_csv: Path
	paired_deltas_csv: Path
	summary_by_budget_csv: Path
	summary_markdown: Path
	paired_metric_count: int
	pair_count: int
	budget_count: int


@dataclass(frozen=True)
class _JoinedRun:
	model_role: str
	model_tag: str
	budget_id: str
	per_class_cap: int | None
	subsample_seed: int
	train_token_count: int
	validation_token_count: int
	paired_identity_hash: str
	metrics_json: Path
	metrics: Mapping[str, object]


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for paired label-budget robustness summaries."""
	parser = argparse.ArgumentParser(
		description='Summarize paired F3 lithology label-budget robustness metrics.',
	)
	source = parser.add_mutually_exclusive_group(required=True)
	source.add_argument(
		'--config',
		type=Path,
		help='Path to a YAML summary configuration file.',
	)
	source.add_argument(
		'--suite-root',
		type=Path,
		help='Directory containing suite_manifest.json and probe_run_manifest.json.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate inputs and print planned report paths without writing files.',
	)
	return parser


def main() -> None:
	"""Summarize paired label-budget probe metrics."""
	parser = build_parser()
	args = parser.parse_args()
	suite_root = args.suite_root
	if args.config is not None:
		raw_config = load_config_for_cli(args.config, loader=load_config)
		config = resolve_config_for_cli(
			raw_config,
			resolver=f3_lithology_label_budget_summary_config_from_mapping,
			config_path=args.config,
		)
		suite_root = config.suite_root
	if args.dry_run:
		_print_dry_run(suite_root)
		print('execution: dry-run; label-budget robustness summary skipped')
		return
	result = summarize_label_budget_robustness(suite_root)
	print(
		f'f3_lithology_label_budget_summary.paired_rows: {result.paired_metric_count}'
	)
	print(f'f3_lithology_label_budget_summary.pairs: {result.pair_count}')
	print(f'f3_lithology_label_budget_summary.budgets: {result.budget_count}')
	print(f'label_budget_summary.paired_metrics_csv: {result.paired_metrics_csv}')
	print(f'label_budget_summary.paired_deltas_csv: {result.paired_deltas_csv}')
	print(f'label_budget_summary.summary_by_budget_csv: {result.summary_by_budget_csv}')
	print(
		f'f3_lithology_label_budget_summary.summary_markdown: {result.summary_markdown}'
	)


def summarize_label_budget_robustness(suite_root: Path) -> LabelBudgetSummaryResult:
	"""Read manifests and metrics, then write paired delta summary reports."""
	suite_root = suite_root.resolve()
	payload = _load_summary_payload(suite_root)
	reports_dir = suite_root / 'reports'
	reports_dir.mkdir(parents=True, exist_ok=True)

	_write_csv(
		reports_dir / 'paired_metrics.csv',
		payload.paired_metric_fieldnames,
		payload.paired_metric_rows,
	)
	_write_csv(
		reports_dir / 'paired_deltas.csv',
		payload.paired_delta_fieldnames,
		payload.paired_delta_rows,
	)
	_write_csv(
		reports_dir / 'summary_by_budget.csv',
		SUMMARY_FIELDNAMES,
		payload.summary_rows,
	)
	(reports_dir / 'summary.md').write_text(
		_render_summary_markdown(payload),
		encoding='utf-8',
	)
	return LabelBudgetSummaryResult(
		paired_metrics_csv=reports_dir / 'paired_metrics.csv',
		paired_deltas_csv=reports_dir / 'paired_deltas.csv',
		summary_by_budget_csv=reports_dir / 'summary_by_budget.csv',
		summary_markdown=reports_dir / 'summary.md',
		paired_metric_count=len(payload.paired_metric_rows),
		pair_count=len(payload.paired_delta_rows),
		budget_count=len(payload.summary_rows),
	)


def _print_dry_run(suite_root: Path) -> None:
	payload = _load_summary_payload(suite_root.resolve())
	reports_dir = suite_root.resolve() / 'reports'
	print(f'stage: {STAGE}')
	print(f'suite_root: {suite_root.resolve()}')
	print(f'suite_manifest: {suite_root.resolve() / "suite_manifest.json"}')
	print(f'probe_run_manifest: {suite_root.resolve() / "probe_run_manifest.json"}')
	print(f'paired_metric_rows: {len(payload.paired_metric_rows)}')
	print(f'paired_delta_rows: {len(payload.paired_delta_rows)}')
	print(f'budget_rows: {len(payload.summary_rows)}')
	print(f'paired_metrics_csv: {reports_dir / "paired_metrics.csv"}')
	print(f'paired_deltas_csv: {reports_dir / "paired_deltas.csv"}')
	print(f'summary_by_budget_csv: {reports_dir / "summary_by_budget.csv"}')
	print(f'summary_markdown: {reports_dir / "summary.md"}')


@dataclass(frozen=True)
class _SummaryPayload:
	suite_name: str
	model_tags: Mapping[str, str]
	paired_metric_fieldnames: tuple[str, ...]
	paired_delta_fieldnames: tuple[str, ...]
	paired_metric_rows: tuple[Mapping[str, object], ...]
	paired_delta_rows: tuple[Mapping[str, object], ...]
	summary_rows: tuple[Mapping[str, object], ...]


def _load_summary_payload(suite_root: Path) -> _SummaryPayload:
	suite_manifest = _read_json(suite_root / 'suite_manifest.json')
	probe_manifest = _read_json(suite_root / 'probe_run_manifest.json')
	_validate_artifact_types(suite_manifest, probe_manifest)
	suite_rows = _rows(suite_manifest, label='suite_manifest')
	probe_rows = _rows(probe_manifest, label='probe_run_manifest')
	suite_name = _suite_name(suite_manifest)

	joined = _joined_runs(suite_rows, probe_rows)
	_validate_complete_pairs(joined)
	class_columns = _class_columns(run.metrics for run in joined)
	pairs = _paired_runs(joined)

	paired_metric_rows = tuple(
		_paired_metric_row(run, class_columns=class_columns) for run in joined
	)
	paired_delta_rows = tuple(
		_delta_row(
			baseline,
			candidate,
			class_columns=class_columns,
		)
		for baseline, candidate in pairs
	)
	summary_rows = _summary_rows(paired_delta_rows)
	return _SummaryPayload(
		suite_name=suite_name,
		model_tags=_model_tags(joined),
		paired_metric_fieldnames=(
			'model_role',
			'model_tag',
			'budget_id',
			'per_class_cap',
			'subsample_seed',
			'train_token_count',
			'validation_token_count',
			*OVERALL_METRICS,
			*class_columns,
			'metrics_json',
		),
		paired_delta_fieldnames=(
			'budget_id',
			'per_class_cap',
			'subsample_seed',
			*(f'delta_{metric}' for metric in OVERALL_METRICS),
			*(f'delta_{column}' for column in class_columns),
			'baseline_metrics_json',
			'candidate_metrics_json',
		),
		paired_metric_rows=paired_metric_rows,
		paired_delta_rows=paired_delta_rows,
		summary_rows=summary_rows,
	)


def _validate_artifact_types(
	suite_manifest: Mapping[str, object],
	probe_manifest: Mapping[str, object],
) -> None:
	if (
		suite_manifest.get('artifact_type')
		!= 'f3_lithology_label_budget_suite_manifest'
	):
		msg = 'suite_manifest.json is not an F3 label-budget suite manifest'
		raise ValueError(msg)
	if (
		probe_manifest.get('artifact_type')
		!= 'f3_lithology_label_budget_probe_run_manifest'
	):
		msg = 'probe_run_manifest.json is not an F3 label-budget probe manifest'
		raise ValueError(msg)


def _joined_runs(
	suite_rows: Sequence[Mapping[str, object]],
	probe_rows: Sequence[Mapping[str, object]],
) -> tuple[_JoinedRun, ...]:
	suite_by_key = _index_rows(suite_rows, label='suite_manifest')
	probe_by_key = _index_rows(probe_rows, label='probe_run_manifest')
	if set(suite_by_key) != set(probe_by_key):
		missing_probe = sorted(set(suite_by_key) - set(probe_by_key))
		missing_suite = sorted(set(probe_by_key) - set(suite_by_key))
		msg = (
			'suite/probe manifest row mismatch; '
			f'missing_probe={missing_probe!r}, missing_suite={missing_suite!r}'
		)
		raise ValueError(msg)
	joined = []
	for key in sorted(suite_by_key, key=_run_key_sort_key):
		suite_row = suite_by_key[key]
		probe_row = probe_by_key[key]
		_validate_manifest_row_agreement(suite_row, probe_row)
		metrics_path = Path(_required_str(probe_row, 'metrics_json'))
		metrics = _read_json(metrics_path)
		_validate_metrics(metrics, metrics_path=metrics_path)
		joined.append(
			_JoinedRun(
				model_role=str(key[0]),
				model_tag=_required_str(probe_row, 'model_tag'),
				budget_id=str(key[1]),
				per_class_cap=_optional_int(probe_row.get('per_class_cap')),
				subsample_seed=int(key[2]),
				train_token_count=_required_int(probe_row, 'train_token_count'),
				validation_token_count=_required_int(
					probe_row,
					'validation_token_count',
				),
				paired_identity_hash=_required_str(
					suite_row,
					'paired_identity_hash',
				),
				metrics_json=metrics_path,
				metrics=metrics,
			),
		)
	return tuple(joined)


def _validate_complete_pairs(runs: Sequence[_JoinedRun]) -> None:
	by_condition: dict[tuple[str, int], list[_JoinedRun]] = defaultdict(list)
	for run in runs:
		by_condition[(run.budget_id, run.subsample_seed)].append(run)
	for (budget_id, subsample_seed), condition_runs in by_condition.items():
		roles = sorted(run.model_role for run in condition_runs)
		if roles != ['baseline', 'candidate']:
			msg = (
				'label-budget summary requires exactly one baseline and one '
				'candidate for every pair; '
				f'budget_id={budget_id!r}, subsample_seed={subsample_seed}, '
				f'roles={roles!r}'
			)
			raise ValueError(msg)
		hashes = {run.model_role: run.paired_identity_hash for run in condition_runs}
		if hashes['baseline'] != hashes['candidate']:
			msg = (
				'paired_identity_hash mismatch; '
				f'budget_id={budget_id!r}, subsample_seed={subsample_seed}, '
				f'baseline={hashes["baseline"]}, candidate={hashes["candidate"]}'
			)
			raise ValueError(msg)


def _paired_runs(
	runs: Sequence[_JoinedRun],
) -> tuple[tuple[_JoinedRun, _JoinedRun], ...]:
	by_condition: dict[tuple[str, int], dict[str, _JoinedRun]] = defaultdict(dict)
	for run in runs:
		by_condition[(run.budget_id, run.subsample_seed)][run.model_role] = run
	pairs = []
	for key in sorted(by_condition, key=_condition_sort_key):
		condition = by_condition[key]
		baseline = condition['baseline']
		candidate = condition['candidate']
		if baseline.per_class_cap != candidate.per_class_cap:
			msg = (
				'per_class_cap mismatch for paired condition; '
				f'budget_id={key[0]!r}, subsample_seed={key[1]}'
			)
			raise ValueError(msg)
		_validate_per_class_pair(baseline, candidate)
		pairs.append((baseline, candidate))
	return tuple(pairs)


def _paired_metric_row(
	run: _JoinedRun,
	*,
	class_columns: Sequence[str],
) -> dict[str, object]:
	row: dict[str, object] = {
		'model_role': run.model_role,
		'model_tag': run.model_tag,
		'budget_id': run.budget_id,
		'per_class_cap': run.per_class_cap,
		'subsample_seed': run.subsample_seed,
		'train_token_count': run.train_token_count,
		'validation_token_count': run.validation_token_count,
		'metrics_json': str(run.metrics_json),
	}
	for metric in OVERALL_METRICS:
		row[metric] = _finite_float(run.metrics[metric], f'{run.metrics_json}:{metric}')
	per_class_f1 = _per_class_f1(run.metrics)
	for column in class_columns:
		class_id = column.removeprefix('class_').removesuffix('_f1')
		row[column] = _finite_float(
			per_class_f1[class_id],
			f'{run.metrics_json}:per_class_f1.{class_id}',
		)
	return row


def _delta_row(
	baseline: _JoinedRun,
	candidate: _JoinedRun,
	*,
	class_columns: Sequence[str],
) -> dict[str, object]:
	row: dict[str, object] = {
		'budget_id': baseline.budget_id,
		'per_class_cap': baseline.per_class_cap,
		'subsample_seed': baseline.subsample_seed,
		'baseline_metrics_json': str(baseline.metrics_json),
		'candidate_metrics_json': str(candidate.metrics_json),
	}
	for metric in OVERALL_METRICS:
		row[f'delta_{metric}'] = _metric_delta(baseline, candidate, metric)
	baseline_f1 = _per_class_f1(baseline.metrics)
	candidate_f1 = _per_class_f1(candidate.metrics)
	for column in class_columns:
		class_id = column.removeprefix('class_').removesuffix('_f1')
		row[f'delta_{column}'] = _finite_float(
			candidate_f1[class_id],
			f'{candidate.metrics_json}:per_class_f1.{class_id}',
		) - _finite_float(
			baseline_f1[class_id],
			f'{baseline.metrics_json}:per_class_f1.{class_id}',
		)
	return row


def _summary_rows(
	paired_delta_rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
	by_budget: dict[str, list[Mapping[str, object]]] = defaultdict(list)
	for row in paired_delta_rows:
		by_budget[str(row['budget_id'])].append(row)
	rows = []
	for budget_id in sorted(by_budget, key=_budget_sort_key):
		budget_rows = by_budget[budget_id]
		macro = _metric_values(budget_rows, 'delta_macro_f1')
		mean_iou = _metric_values(budget_rows, 'delta_mean_iou')
		balanced_accuracy = _metric_values(
			budget_rows,
			'delta_balanced_accuracy',
		)
		rows.append(
			{
				'budget_id': budget_id,
				'per_class_cap': budget_rows[0]['per_class_cap'],
				'n_pairs': len(budget_rows),
				'mean_delta_macro_f1': mean(macro),
				'median_delta_macro_f1': median(macro),
				'std_delta_macro_f1': stdev(macro) if len(macro) > 1 else 0.0,
				'min_delta_macro_f1': min(macro),
				'max_delta_macro_f1': max(macro),
				'win_rate_macro_f1': _win_rate(macro),
				'mean_delta_mean_iou': mean(mean_iou),
				'win_rate_mean_iou': _win_rate(mean_iou),
				'mean_delta_balanced_accuracy': mean(balanced_accuracy),
				'win_rate_balanced_accuracy': _win_rate(balanced_accuracy),
			},
		)
	return tuple(rows)


def _render_summary_markdown(payload: _SummaryPayload) -> str:
	lines = [
		'# F3 Lithology Label-Budget Robustness Summary',
		'',
		f'- suite: {payload.suite_name}',
		f'- baseline model: {payload.model_tags.get("baseline", "")}',
		f'- candidate model: {payload.model_tags.get("candidate", "")}',
		'- analysis: paired label-budget robustness, not probe seed sweep',
		'',
		'## Budget Summary',
		'',
		(
			'| budget_id | per_class_cap | n_pairs | mean_delta_macro_f1 | '
			'win_rate_macro_f1 | mean_delta_mean_iou | win_rate_mean_iou | '
			'mean_delta_balanced_accuracy | win_rate_balanced_accuracy |'
		),
		'| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
	]
	lines.extend(
		(
			'| '
			f'{row["budget_id"]} | {_display(row["per_class_cap"])} | '
			f'{row["n_pairs"]} | {_format_float(row["mean_delta_macro_f1"])} | '
			f'{_format_float(row["win_rate_macro_f1"])} | '
			f'{_format_float(row["mean_delta_mean_iou"])} | '
			f'{_format_float(row["win_rate_mean_iou"])} | '
			f'{_format_float(row["mean_delta_balanced_accuracy"])} | '
			f'{_format_float(row["win_rate_balanced_accuracy"])} |'
		)
		for row in payload.summary_rows
	)
	lines.extend(
		(
			'',
			'## Go/Hold/Stop Guidance',
			'',
			(
				'Go if low-budget and full-budget conditions have positive '
				'mean delta_macro_f1 and delta_mean_iou, with win_rate >= 0.7.'
			),
			'Hold if only full budget wins or metrics conflict.',
			(
				'Stop if low-budget deltas are negative or balanced_accuracy '
				'consistently degrades.'
			),
			'',
			f'Current guidance: {_guidance(payload.summary_rows)}.',
			'',
			'## Per-Class Warnings',
			'',
		),
	)
	warnings = _per_class_warnings(payload.paired_delta_rows)
	lines.extend(warnings or ['- none'])
	lines.append('')
	return '\n'.join(lines)


def _guidance(summary_rows: Sequence[Mapping[str, object]]) -> str:
	if not summary_rows:
		return 'Hold'
	low = _low_budget_row(summary_rows)
	full = _full_budget_row(summary_rows)
	if low is not None and (
		float(low['mean_delta_macro_f1']) < 0 or float(low['mean_delta_mean_iou']) < 0
	):
		return 'Stop'
	if summary_rows and all(
		float(row['mean_delta_balanced_accuracy']) < 0 for row in summary_rows
	):
		return 'Stop'
	if (
		low is not None
		and full is not None
		and all(
			float(row['mean_delta_macro_f1']) > 0
			and float(row['mean_delta_mean_iou']) > 0
			and float(row['win_rate_macro_f1']) >= 0.7
			and float(row['win_rate_mean_iou']) >= 0.7
			for row in (low, full)
		)
	):
		return 'Go'
	for row in summary_rows:
		if row['budget_id'] == 'full' and (
			float(row['mean_delta_macro_f1']) > 0
			or float(row['mean_delta_mean_iou']) > 0
		):
			return 'Hold'
	return 'Hold'


def _per_class_warnings(
	paired_delta_rows: Sequence[Mapping[str, object]],
) -> list[str]:
	columns = sorted(
		{
			key
			for row in paired_delta_rows
			for key in row
			if key.startswith('delta_class_') and key.endswith('_f1')
		},
		key=lambda value: _class_metric_sort_key(value.removeprefix('delta_')),
	)
	warnings = []
	for column in columns:
		values = _metric_values(paired_delta_rows, column)
		if mean(values) < 0:
			class_id = column.removeprefix('delta_class_').removesuffix('_f1')
			prefix = 'especially ' if class_id in {'3', '5'} else ''
			warnings.append(
				(
					f'- {prefix}class {class_id} has negative mean F1 delta: '
					f'{_format_float(mean(values))}'
				),
			)
	return warnings


def _validate_manifest_row_agreement(
	suite_row: Mapping[str, object],
	probe_row: Mapping[str, object],
) -> None:
	for key in (
		'model_role',
		'model_tag',
		'budget_id',
		'per_class_cap',
		'subsample_seed',
		'paired_identity_hash',
	):
		if suite_row.get(key) != probe_row.get(key):
			msg = f'suite/probe manifest value mismatch for {key}: {suite_row!r}'
			raise ValueError(msg)


def _index_rows(
	rows: Sequence[Mapping[str, object]],
	*,
	label: str,
) -> dict[tuple[str, str, int], Mapping[str, object]]:
	by_key: dict[tuple[str, str, int], Mapping[str, object]] = {}
	for row in rows:
		key = _run_key(row)
		if key in by_key:
			msg = f'duplicate {label} row for key {key!r}'
			raise ValueError(msg)
		by_key[key] = row
	return by_key


def _run_key(row: Mapping[str, object]) -> tuple[str, str, int]:
	return (
		_required_str(row, 'model_role'),
		_required_str(row, 'budget_id'),
		_required_int(row, 'subsample_seed'),
	)


def _run_key_sort_key(
	key: tuple[str, str, int],
) -> tuple[tuple[int, int | str], int, str]:
	role, budget_id, subsample_seed = key
	return (_budget_sort_key(budget_id), subsample_seed, role)


def _condition_sort_key(key: tuple[str, int]) -> tuple[tuple[int, int | str], int]:
	return (_budget_sort_key(key[0]), key[1])


def _budget_sort_key(value: object) -> tuple[int, int | str]:
	text = str(value)
	if text == 'full':
		return (1, 0)
	if text.startswith('cap') and text[3:].isdigit():
		return (0, int(text[3:]))
	return (0, text)


def _metric_delta(
	baseline: _JoinedRun,
	candidate: _JoinedRun,
	metric: str,
) -> float:
	return _finite_float(
		candidate.metrics[metric],
		f'{candidate.metrics_json}:{metric}',
	) - _finite_float(baseline.metrics[metric], f'{baseline.metrics_json}:{metric}')


def _validate_metrics(metrics: Mapping[str, object], *, metrics_path: Path) -> None:
	_validate_finite_json_numbers(metrics, label=str(metrics_path))
	for key in OVERALL_METRICS:
		if key not in metrics:
			msg = f'metrics missing required key {key!r}: {metrics_path}'
			raise ValueError(msg)
		_finite_float(metrics[key], f'{metrics_path}:{key}')
	for class_id, value in _per_class_f1(metrics).items():
		_finite_float(value, f'{metrics_path}:per_class_f1.{class_id}')


def _validate_finite_json_numbers(value: object, *, label: str) -> None:
	if isinstance(value, bool):
		return
	if isinstance(value, int | float):
		_finite_float(value, label)
		return
	if isinstance(value, Mapping):
		for key, item in value.items():
			_validate_finite_json_numbers(item, label=f'{label}.{key}')
		return
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		for index, item in enumerate(value):
			_validate_finite_json_numbers(item, label=f'{label}[{index}]')


def _validate_per_class_pair(baseline: _JoinedRun, candidate: _JoinedRun) -> None:
	baseline_classes = set(_per_class_f1(baseline.metrics))
	candidate_classes = set(_per_class_f1(candidate.metrics))
	if baseline_classes != candidate_classes:
		msg = (
			'per_class_f1 class mismatch for paired condition; '
			f'budget_id={baseline.budget_id!r}, '
			f'subsample_seed={baseline.subsample_seed}, '
			f'baseline={sorted(baseline_classes)!r}, '
			f'candidate={sorted(candidate_classes)!r}'
		)
		raise ValueError(msg)


def _class_columns(metrics_items: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
	columns = {
		f'class_{class_id}_f1'
		for metrics in metrics_items
		for class_id in _per_class_f1(metrics)
	}
	return tuple(sorted(columns, key=_class_metric_sort_key))


def _class_metric_sort_key(value: str) -> tuple[int, int | str]:
	class_id = value.removeprefix('class_').removesuffix('_f1')
	return (0, int(class_id)) if class_id.isdigit() else (1, class_id)


def _per_class_f1(metrics: Mapping[str, object]) -> Mapping[str, object]:
	value = metrics.get('per_class_f1')
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = 'metrics per_class_f1 must be a mapping'
		raise TypeError(msg)
	return {str(key): item for key, item in value.items()}


def _metric_values(
	rows: Sequence[Mapping[str, object]],
	key: str,
) -> tuple[float, ...]:
	return tuple(_finite_float(row[key], key) for row in rows)


def _win_rate(values: Sequence[float]) -> float:
	return sum(value > 0 for value in values) / len(values)


def _low_budget_row(
	rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
	capped = [
		row for row in rows if _optional_int(row.get('per_class_cap')) is not None
	]
	if not capped:
		return None
	return min(capped, key=lambda row: int(row['per_class_cap']))


def _full_budget_row(
	rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
	for row in rows:
		if row['budget_id'] == 'full' or row['per_class_cap'] is None:
			return row
	return None


def _model_tags(runs: Sequence[_JoinedRun]) -> Mapping[str, str]:
	tags = {}
	for run in runs:
		tags.setdefault(run.model_role, run.model_tag)
	return tags


def _suite_name(payload: Mapping[str, object]) -> str:
	suite = payload.get('suite')
	if isinstance(suite, Mapping):
		name = suite.get('name')
		if isinstance(name, str) and name:
			return name
	return ''


def _rows(
	payload: Mapping[str, object],
	*,
	label: str,
) -> tuple[Mapping[str, object], ...]:
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		msg = f'{label} rows must be a list'
		raise TypeError(msg)
	result = []
	for index, row in enumerate(rows):
		if not isinstance(row, Mapping):
			msg = f'{label} row {index} must be a mapping'
			raise TypeError(msg)
		result.append(row)
	if not result:
		msg = f'{label} contains no rows'
		raise ValueError(msg)
	return tuple(result)


def _read_json(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		msg = f'required file does not exist: {path}'
		raise FileNotFoundError(msg)
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		msg = f'JSON file must contain an object: {path}'
		raise TypeError(msg)
	return payload


def _write_csv(
	path: Path,
	fieldnames: Sequence[str],
	rows: Sequence[Mapping[str, object]],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _required_str(row: Mapping[str, object], key: str) -> str:
	value = row.get(key)
	if not isinstance(value, str) or not value:
		msg = f'required string missing or empty: {key}'
		raise ValueError(msg)
	return value


def _required_int(row: Mapping[str, object], key: str) -> int:
	value = row.get(key)
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'required integer missing: {key}'
		raise TypeError(msg)
	return value


def _optional_int(value: object) -> int | None:
	if value is None:
		return None
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'optional integer value is invalid: {value!r}'
		raise TypeError(msg)
	return value


def _finite_float(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'metric must be numeric: {label}'
		raise TypeError(msg)
	result = float(value)
	if not math.isfinite(result):
		msg = f'metric must be finite: {label}'
		raise ValueError(msg)
	return result


def _display(value: object) -> str:
	return 'full' if value is None else str(value)


def _format_float(value: object) -> str:
	return f'{float(value):.6f}'


if __name__ == '__main__':
	main()
