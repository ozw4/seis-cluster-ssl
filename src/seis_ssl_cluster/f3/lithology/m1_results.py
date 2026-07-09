"""Milestone-1 F3 strat-HMM result consolidation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

CORE_METRICS = (
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'mean_iou',
)
DELTA_METRICS = (
	'macro_f1',
	'mean_iou',
	'balanced_accuracy',
)


@dataclass(frozen=True)
class F3StratHMMM1ResultsConfig:
	"""Resolved inputs and outputs for M1 result consolidation."""

	baseline_comparison_csv: Path
	label_budget_suite_root: Path
	split_index_suite_root: Path
	output_dir: Path
	baseline_model: str
	candidate_model: str


@dataclass(frozen=True)
class F3StratHMMM1ResultsResult:
	"""Paths written by the M1 result consolidation."""

	summary_json: Path
	summary_markdown: Path
	warnings: tuple[str, ...]


def f3_strat_hmm_m1_results_config_from_mapping(
	config: Mapping[str, object],
) -> F3StratHMMM1ResultsConfig:
	"""Validate and normalize an M1 result consolidation config mapping."""
	_validate_allowed_keys(config, {'inputs', 'models', 'outputs'}, prefix='config')
	inputs = _required_mapping(config, 'inputs')
	models = _required_mapping(config, 'models')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		inputs,
		{
			'baseline_comparison_csv',
			'label_budget_suite_root',
			'split_index_suite_root',
		},
		prefix='inputs',
	)
	_validate_allowed_keys(models, {'baseline', 'candidate'}, prefix='models')
	_validate_allowed_keys(outputs, {'output_dir'}, prefix='outputs')
	resolved = F3StratHMMM1ResultsConfig(
		baseline_comparison_csv=_required_path(
			inputs,
			'baseline_comparison_csv',
			prefix='inputs',
		),
		label_budget_suite_root=_required_path(
			inputs,
			'label_budget_suite_root',
			prefix='inputs',
		),
		split_index_suite_root=_required_path(
			inputs,
			'split_index_suite_root',
			prefix='inputs',
		),
		output_dir=_required_path(outputs, 'output_dir', prefix='outputs'),
		baseline_model=_required_str(models, 'baseline', prefix='models'),
		candidate_model=_required_str(models, 'candidate', prefix='models'),
	)
	validate_f3_strat_hmm_m1_results_config(resolved)
	return resolved


def validate_f3_strat_hmm_m1_results_config(
	config: F3StratHMMM1ResultsConfig,
) -> None:
	"""Validate required input paths for an M1 consolidation config."""
	_require_file(config.baseline_comparison_csv, label='baseline_comparison_csv')
	_require_dir(config.label_budget_suite_root, label='label_budget_suite_root')
	_require_dir(config.split_index_suite_root, label='split_index_suite_root')
	_require_file(
		_label_budget_paired_deltas_csv(config),
		label='label_budget paired_deltas_csv',
	)
	_require_file(
		_split_paired_deltas_csv(config),
		label='split_index paired_deltas_csv',
	)


def consolidate_f3_strat_hmm_m1_results(
	config: F3StratHMMM1ResultsConfig,
) -> F3StratHMMM1ResultsResult:
	"""Read M1 artifacts and write deterministic JSON and Markdown summaries."""
	validate_f3_strat_hmm_m1_results_config(config)
	warnings: list[str] = []
	single_split = _single_split_summary(
		config.baseline_comparison_csv,
		baseline_model=config.baseline_model,
		candidate_model=config.candidate_model,
	)
	label_budget = _label_budget_summary(
		_label_budget_paired_deltas_csv(config),
		config.label_budget_suite_root / 'suite_manifest.json',
		warnings=warnings,
	)
	split_index = _split_index_summary(_split_paired_deltas_csv(config))
	if _full_budget_balanced_accuracy_caveat(label_budget):
		warnings.append(
			'full budget balanced_accuracy delta is negative; monitor this caveat',
		)
	decision = _decision(single_split, label_budget, split_index)
	payload = {
		'schema_version': 1,
		'baseline_model': config.baseline_model,
		'candidate_model': config.candidate_model,
		'single_split': single_split,
		'label_budget': label_budget,
		'split_index': split_index,
		'decision': decision,
		'warnings': warnings,
	}
	config.output_dir.mkdir(parents=True, exist_ok=True)
	json_path = config.output_dir / 'm1_results_summary.json'
	markdown_path = config.output_dir / 'm1_results_summary.md'
	json_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
	markdown_path.write_text(_render_markdown(payload), encoding='utf-8')
	return F3StratHMMM1ResultsResult(
		summary_json=json_path,
		summary_markdown=markdown_path,
		warnings=tuple(warnings),
	)


def _single_split_summary(
	path: Path,
	*,
	baseline_model: str,
	candidate_model: str,
) -> dict[str, object]:
	rows = _read_csv(path)
	baseline = _model_row(rows, baseline_model, path=path)
	candidate = _model_row(rows, candidate_model, path=path)
	baseline_metrics = _metrics_from_row(baseline, path=path)
	candidate_metrics = _metrics_from_row(candidate, path=path)
	return {
		'baseline': baseline_metrics,
		'candidate': candidate_metrics,
		'delta': {
			metric: candidate_metrics[metric] - baseline_metrics[metric]
			for metric in CORE_METRICS
		},
	}


def _label_budget_summary(
	paired_deltas_csv: Path,
	suite_manifest_json: Path,
	*,
	warnings: list[str],
) -> dict[str, object]:
	rows = _read_csv(paired_deltas_csv)
	identity_by_condition = _label_budget_identity_by_condition(suite_manifest_json)
	if identity_by_condition is None and _has_repeated_full_budget(rows):
		warnings.append(
			'full budget has repeated seed rows; duplicate independence cannot be '
			'inferred from available label-budget artifacts',
		)
	by_budget: dict[str, list[Mapping[str, str]]] = defaultdict(list)
	for row in rows:
		by_budget[_required_cell(row, 'budget_id', path=paired_deltas_csv)].append(row)
	budgets = []
	for budget_id in sorted(by_budget, key=_budget_sort_key):
		budget_rows = _deduplicated_budget_rows(
			by_budget[budget_id],
			identity_by_condition=identity_by_condition,
			budget_id=budget_id,
			warnings=warnings,
		)
		budgets.append(_budget_summary_row(budget_id, budget_rows, paired_deltas_csv))
	return {'budgets': budgets}


def _split_index_summary(paired_deltas_csv: Path) -> dict[str, object]:
	rows = _read_csv(paired_deltas_csv)
	splits = [
		_split_summary_row(row, paired_deltas_csv)
		for row in sorted(
			rows,
			key=lambda item: _required_cell(
				item,
				'split_id',
				path=paired_deltas_csv,
			),
		)
	]
	return {
		'splits': splits,
		'win_rates': {
			metric: _win_rate([split[f'delta_{metric}'] for split in splits])
			for metric in DELTA_METRICS
		},
	}


def _split_summary_row(row: Mapping[str, str], path: Path) -> dict[str, object]:
	return {
		'split_id': _required_cell(row, 'split_id', path=path),
		'delta_macro_f1': _required_float(row, 'delta_macro_f1', path=path),
		'delta_mean_iou': _required_float(row, 'delta_mean_iou', path=path),
		'delta_balanced_accuracy': _required_float(
			row,
			'delta_balanced_accuracy',
			path=path,
		),
	}


def _budget_summary_row(
	budget_id: str,
	rows: Sequence[Mapping[str, str]],
	path: Path,
) -> dict[str, object]:
	if not rows:
		msg = f'label-budget summary has no rows for budget_id={budget_id!r}'
		raise ValueError(msg)
	per_class_cap = _optional_int_cell(rows[0], 'per_class_cap', path=path)
	macro = [_required_float(row, 'delta_macro_f1', path=path) for row in rows]
	mean_iou = [_required_float(row, 'delta_mean_iou', path=path) for row in rows]
	balanced_accuracy = [
		_required_float(row, 'delta_balanced_accuracy', path=path) for row in rows
	]
	return {
		'budget_id': budget_id,
		'per_class_cap': per_class_cap,
		'n_pairs': len(rows),
		'mean_delta_macro_f1': mean(macro),
		'win_rate_macro_f1': _win_rate(macro),
		'mean_delta_mean_iou': mean(mean_iou),
		'win_rate_mean_iou': _win_rate(mean_iou),
		'mean_delta_balanced_accuracy': mean(balanced_accuracy),
		'win_rate_balanced_accuracy': _win_rate(balanced_accuracy),
	}


def _deduplicated_budget_rows(
	rows: Sequence[Mapping[str, str]],
	*,
	identity_by_condition: Mapping[tuple[str, int], str] | None,
	budget_id: str,
	warnings: list[str],
) -> tuple[Mapping[str, str], ...]:
	if identity_by_condition is None:
		return tuple(rows)
	seen: set[str] = set()
	deduplicated = []
	for row in rows:
		seed = _required_int_cell(row, 'subsample_seed')
		identity = identity_by_condition.get((budget_id, seed))
		if identity is None:
			return tuple(rows)
		if identity in seen:
			continue
		seen.add(identity)
		deduplicated.append(row)
	if len(deduplicated) != len(rows):
		warnings.append(
			f'{budget_id} duplicate label-budget rows collapsed by paired identity',
		)
	return tuple(deduplicated)


def _label_budget_identity_by_condition(
	suite_manifest_json: Path,
) -> Mapping[tuple[str, int], str] | None:
	if not suite_manifest_json.is_file():
		return None
	with suite_manifest_json.open(encoding='utf-8') as handle:
		payload = json.load(handle)
	if not isinstance(payload, Mapping):
		msg = f'suite_manifest.json must contain a mapping: {suite_manifest_json}'
		raise TypeError(msg)
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		msg = f'suite_manifest.json rows must be a list: {suite_manifest_json}'
		raise TypeError(msg)
	by_condition: dict[tuple[str, int], set[str]] = defaultdict(set)
	for raw_row in rows:
		if not isinstance(raw_row, Mapping):
			msg = f'suite_manifest.json row must be a mapping: {suite_manifest_json}'
			raise TypeError(msg)
		budget_id = _required_object_str(raw_row, 'budget_id')
		seed = _required_object_int(raw_row, 'subsample_seed')
		identity = _required_object_str(raw_row, 'paired_identity_hash')
		by_condition[(budget_id, seed)].add(identity)
	result = {}
	for condition, identities in by_condition.items():
		if len(identities) != 1:
			return None
		result[condition] = next(iter(identities))
	return result


def _model_row(
	rows: Sequence[Mapping[str, str]],
	model: str,
	*,
	path: Path,
) -> Mapping[str, str]:
	matches = [row for row in rows if row.get('MODEL_TAG') == model]
	if not matches:
		matches = [row for row in rows if row.get('BASELINE_TAG') == model]
	if len(matches) != 1:
		msg = (
			f'expected exactly one comparison row for model {model!r}; '
			f'found {len(matches)} in {path}'
		)
		raise ValueError(msg)
	return matches[0]


def _metrics_from_row(row: Mapping[str, str], *, path: Path) -> dict[str, float]:
	return {metric: _required_float(row, metric, path=path) for metric in CORE_METRICS}


def _decision(
	single_split: Mapping[str, object],
	label_budget: Mapping[str, object],
	split_index: Mapping[str, object],
) -> dict[str, str]:
	single_delta = _mapping(single_split['delta'])
	budgets = _sequence(label_budget['budgets'])
	splits = _sequence(split_index['splits'])
	label_positive = bool(budgets) and all(
		float(_mapping(row)['mean_delta_macro_f1']) > 0
		and float(_mapping(row)['mean_delta_mean_iou']) > 0
		for row in budgets
	)
	split_positive = bool(splits) and all(
		float(_mapping(row)['delta_macro_f1']) > 0
		and float(_mapping(row)['delta_mean_iou']) > 0
		for row in splits
	)
	single_positive = (
		float(single_delta['macro_f1']) > 0 and float(single_delta['mean_iou']) > 0
	)
	guidance = 'go' if single_positive and label_positive and split_positive else 'hold'
	summary = (
		'Strat-HMM M1 is positive on single-run macro F1/mean IoU, '
		'label-budget robustness, and split/index macro F1/mean IoU.'
		if guidance == 'go'
		else (
			'Strat-HMM M1 has mixed consolidation evidence; inspect deltas '
			'before publication.'
		)
	)
	return {'guidance': guidance, 'summary': summary}


def _render_markdown(payload: Mapping[str, object]) -> str:
	single = _mapping(payload['single_split'])
	delta = _mapping(single['delta'])
	label_budget = _mapping(payload['label_budget'])
	split_index = _mapping(payload['split_index'])
	decision = _mapping(payload['decision'])
	warnings = _sequence(payload['warnings'])
	lines = [
		'# F3 Strat-HMM Milestone-1 Results Summary',
		'',
		f'- baseline model: {payload["baseline_model"]}',
		f'- candidate model: {payload["candidate_model"]}',
		'- HMM labels are a structured pretext signal, not final lithology outputs.',
		(
			'- Single-run result is strong positive: '
			f'delta_macro_f1={_format_float(delta["macro_f1"])}, '
			f'delta_mean_iou={_format_float(delta["mean_iou"])}.'
		),
		(
			'- Label-budget robustness is strongest in low-label regimes; '
			'monitor the full-budget balanced accuracy caveat.'
		),
		(
			'- Split/index robustness shows positive macro F1 and mean IoU '
			'deltas on all tested splits.'
		),
		'',
		'## Label Budget',
		'',
		(
			'| budget_id | per_class_cap | n_pairs | mean_delta_macro_f1 | '
			'mean_delta_mean_iou | mean_delta_balanced_accuracy |'
		),
		'| --- | ---: | ---: | ---: | ---: | ---: |',
	]
	for row in _sequence(label_budget['budgets']):
		budget = _mapping(row)
		lines.append(
			(
				f'| {budget["budget_id"]} | {_display(budget["per_class_cap"])} | '
				f'{budget["n_pairs"]} | '
				f'{_format_float(budget["mean_delta_macro_f1"])} | '
				f'{_format_float(budget["mean_delta_mean_iou"])} | '
				f'{_format_float(budget["mean_delta_balanced_accuracy"])} |'
			),
		)
	lines.extend(
		(
			'',
			'## Split Index',
			'',
			'| split_id | delta_macro_f1 | delta_mean_iou | delta_balanced_accuracy |',
			'| --- | ---: | ---: | ---: |',
		),
	)
	for row in _sequence(split_index['splits']):
		split = _mapping(row)
		lines.append(
			(
				f'| {split["split_id"]} | '
				f'{_format_float(split["delta_macro_f1"])} | '
				f'{_format_float(split["delta_mean_iou"])} | '
				f'{_format_float(split["delta_balanced_accuracy"])} |'
			),
		)
	lines.extend(
		(
			'',
			'## Decision',
			'',
			f'- guidance: {decision["guidance"]}',
			f'- summary: {decision["summary"]}',
			'',
			'## Warnings',
			'',
		),
	)
	lines.extend(f'- {warning}' for warning in warnings)
	if not warnings:
		lines.append('- none')
	lines.append('')
	return '\n'.join(lines)


def _full_budget_balanced_accuracy_caveat(
	label_budget: Mapping[str, object],
) -> bool:
	for row in _sequence(label_budget['budgets']):
		budget = _mapping(row)
		if (
			budget.get('budget_id') == 'full'
			and float(budget['mean_delta_balanced_accuracy']) < 0
		):
			return True
	return False


def _has_repeated_full_budget(rows: Sequence[Mapping[str, str]]) -> bool:
	return sum(1 for row in rows if row.get('budget_id') == 'full') > 1


def _win_rate(values: Sequence[float]) -> float:
	if not values:
		msg = 'win_rate requires at least one value'
		raise ValueError(msg)
	return sum(value > 0 for value in values) / len(values)


def _read_csv(path: Path) -> tuple[Mapping[str, str], ...]:
	_require_file(path, label='csv')
	with path.open(newline='', encoding='utf-8') as handle:
		rows = tuple(csv.DictReader(handle))
	if not rows:
		msg = f'csv file contains no rows: {path}'
		raise ValueError(msg)
	return rows


def _label_budget_paired_deltas_csv(config: F3StratHMMM1ResultsConfig) -> Path:
	return config.label_budget_suite_root / 'reports' / 'paired_deltas.csv'


def _split_paired_deltas_csv(config: F3StratHMMM1ResultsConfig) -> Path:
	return config.split_index_suite_root / 'reports' / 'split_paired_deltas.csv'


def _required_float(row: Mapping[str, str], key: str, *, path: Path) -> float:
	text = _required_cell(row, key, path=path)
	try:
		value = float(text)
	except ValueError as exc:
		msg = f'{path}:{key} must be a finite number; got {text!r}'
		raise ValueError(msg) from exc
	if not math.isfinite(value):
		msg = f'{path}:{key} must be finite; got {value!r}'
		raise ValueError(msg)
	return value


def _required_cell(row: Mapping[str, str], key: str, *, path: Path) -> str:
	if key not in row:
		msg = f'csv missing required column {key!r}: {path}'
		raise ValueError(msg)
	value = row[key]
	if value is None or value == '':
		msg = f'csv missing required value for {key!r}: {path}'
		raise ValueError(msg)
	return value


def _optional_int_cell(row: Mapping[str, str], key: str, *, path: Path) -> int | None:
	if key not in row:
		msg = f'csv missing required column {key!r}: {path}'
		raise ValueError(msg)
	value = row[key]
	if value in (None, ''):
		return None
	try:
		return int(value)
	except ValueError as exc:
		msg = f'{path}:{key} must be an integer or empty; got {value!r}'
		raise ValueError(msg) from exc


def _required_int_cell(row: Mapping[str, str], key: str) -> int:
	value = row.get(key)
	if value in (None, ''):
		msg = f'csv missing required value for {key!r}'
		raise ValueError(msg)
	try:
		return int(value)
	except ValueError as exc:
		msg = f'csv value for {key!r} must be an integer; got {value!r}'
		raise ValueError(msg) from exc


def _required_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = mapping.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		msg = f'expected mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _sequence(value: object) -> Sequence[object]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'expected sequence; got {value!r}'
		raise TypeError(msg)
	return value


def _required_path(mapping: Mapping[str, object], key: str, *, prefix: str) -> Path:
	value = mapping.get(key)
	if not isinstance(value, str | Path) or not str(value):
		msg = f'{prefix}.{key} must be a path string'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _required_str(mapping: Mapping[str, object], key: str, *, prefix: str) -> str:
	value = mapping.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string'
		raise TypeError(msg)
	return value


def _required_object_str(mapping: Mapping[object, object], key: str) -> str:
	value = mapping.get(key)
	if not isinstance(value, str) or not value:
		msg = f'suite_manifest row {key} must be a non-empty string'
		raise TypeError(msg)
	return value


def _required_object_int(mapping: Mapping[object, object], key: str) -> int:
	value = mapping.get(key)
	if not isinstance(value, int):
		msg = f'suite_manifest row {key} must be an integer'
		raise TypeError(msg)
	return value


def _validate_allowed_keys(
	mapping: Mapping[str, object],
	allowed: set[str],
	*,
	prefix: str,
) -> None:
	extra = sorted(set(mapping) - allowed)
	if extra:
		msg = f'{prefix} contains unsupported keys: {extra!r}'
		raise ValueError(msg)


def _require_file(path: Path, *, label: str) -> None:
	if not path.is_file():
		msg = f'required input file does not exist for {label}: {path}'
		raise FileNotFoundError(msg)


def _require_dir(path: Path, *, label: str) -> None:
	if not path.is_dir():
		msg = f'required input directory does not exist for {label}: {path}'
		raise FileNotFoundError(msg)


def _budget_sort_key(value: object) -> tuple[int, int | str]:
	text = str(value)
	if text == 'full':
		return (1, 0)
	if text.startswith('cap') and text[3:].isdigit():
		return (0, int(text[3:]))
	return (0, text)


def _display(value: object) -> str:
	return '' if value is None else str(value)


def _format_float(value: object) -> str:
	return f'{float(value):.6f}'
