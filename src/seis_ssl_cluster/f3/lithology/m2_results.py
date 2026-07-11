"""M2-A versus M1 result consolidation and mechanical decision contract."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from seis_ssl_cluster.f3.lithology import m1_results as m1
from seis_ssl_cluster.paths import DEFAULT_RESULTS_ROOT, ensure_under_root
from seis_ssl_cluster.results import (
	DEFAULT_MAX_FILE_SIZE_BYTES,
	PublishItem,
	PublishManifest,
	publish_selected_results,
)

REQUIRED_LOW_BUDGETS = ('cap25', 'cap50', 'cap100')
REQUIRED_BUDGETS = (*REQUIRED_LOW_BUDGETS, 'full')
REQUIRED_SUBSAMPLE_SEEDS = (0, 1, 2, 3, 4)
REQUIRED_SPLIT_IDS = tuple(f'split_{index:03d}' for index in range(6))
REQUIRED_MONITORED_CLASS_IDS = frozenset((3, 5))
MONITORED_CLASS_FIGURE = 'monitored_class_deltas.png'
MONITORED_CLASS_TABLE = 'monitored_class_deltas.csv'
M2_RESULTS_PUBLISH_SUFFIXES = frozenset({'.md', '.json', '.csv', '.png'})
M2_RESULTS_TABLE_NAMES = frozenset(
	{
		m1.SINGLE_SPLIT_TABLE,
		m1.LABEL_BUDGET_TABLE,
		m1.SPLIT_INDEX_TABLE,
		MONITORED_CLASS_TABLE,
	}
)
M2_RESULTS_FIGURE_NAMES = frozenset(
	{
		m1.SINGLE_RUN_FIGURE,
		m1.LABEL_BUDGET_FIGURE,
		m1.SPLIT_INDEX_FIGURE,
		MONITORED_CLASS_FIGURE,
	}
)


@dataclass(frozen=True)
class F3StratHMMM2PublishConfig:
	"""Settings for publishing lightweight M2-A result artifacts."""

	enabled: bool = False
	output_dir: Path | None = None
	include_figures: bool = True
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES


@dataclass(frozen=True)
class F3StratHMMM2ResultsConfig:
	"""Resolved inputs and outputs for M2-A result consolidation."""

	baseline_comparison_csv: Path
	m1_metrics_json: Path
	m2a_metrics_json: Path
	label_budget_suite_root: Path
	split_index_suite_root: Path
	output_dir: Path
	baseline_model: str = 'M1'
	candidate_model: str = 'M2-A'
	monitored_class_ids: tuple[int, ...] = (3, 5)
	class_info_json: Path | None = None
	publish: F3StratHMMM2PublishConfig = field(
		default_factory=F3StratHMMM2PublishConfig,
	)


@dataclass(frozen=True)
class F3StratHMMM2ResultsResult:
	"""Paths written by M2-A result consolidation."""

	summary_json: Path
	summary_markdown: Path
	table_paths: tuple[Path, ...]
	figure_paths: tuple[Path, ...]
	decision: str
	publish_manifest: PublishManifest | None = None


@dataclass(frozen=True)
class _RunManifestContract:
	label: str
	dataset_manifest: Path
	probe_manifest: Path
	paired_metrics_csv: Path
	condition_keys: tuple[str, ...]
	dataset_artifact_type: str
	probe_artifact_type: str
	probe_dataset_field: str
	expected_tags: Mapping[str, str]


def f3_strat_hmm_m2_results_config_from_mapping(
	config: Mapping[str, object],
) -> F3StratHMMM2ResultsConfig:
	"""Validate and normalize an M2-A consolidation config mapping."""
	m1._validate_allowed_keys(
		config, {'inputs', 'models', 'outputs', 'publish'}, prefix='config'
	)
	inputs = m1._required_mapping(config, 'inputs')
	models = m1._required_mapping(config, 'models')
	outputs = m1._required_mapping(config, 'outputs')
	publish = m1._optional_mapping(config, 'publish')
	m1._validate_allowed_keys(
		inputs,
		{
			'baseline_comparison_csv',
			'm1_metrics_json',
			'm2a_metrics_json',
			'label_budget_suite_root',
			'split_index_suite_root',
			'class_info_json',
			'monitored_class_ids',
		},
		prefix='inputs',
	)
	m1._validate_allowed_keys(models, {'baseline', 'candidate'}, prefix='models')
	m1._validate_allowed_keys(outputs, {'output_dir'}, prefix='outputs')
	m1._validate_allowed_keys(
		publish,
		{'enabled', 'output_dir', 'include_figures', 'max_file_size_mb'},
		prefix='publish',
	)
	class_info = inputs.get('class_info_json')
	if class_info is not None:
		class_info = m1._required_path(inputs, 'class_info_json', prefix='inputs')
	monitored = inputs.get('monitored_class_ids', [3, 5])
	if not isinstance(monitored, Sequence) or isinstance(monitored, str | bytes):
		raise TypeError('inputs.monitored_class_ids must be a sequence of integers')
	if not monitored or any(
		isinstance(value, bool) or not isinstance(value, int) for value in monitored
	):
		raise ValueError('inputs.monitored_class_ids must contain integers')
	if len(set(monitored)) != len(monitored):
		raise ValueError('inputs.monitored_class_ids must not contain duplicates')
	resolved = F3StratHMMM2ResultsConfig(
		baseline_comparison_csv=m1._required_path(
			inputs, 'baseline_comparison_csv', prefix='inputs'
		),
		m1_metrics_json=m1._required_path(inputs, 'm1_metrics_json', prefix='inputs'),
		m2a_metrics_json=m1._required_path(inputs, 'm2a_metrics_json', prefix='inputs'),
		label_budget_suite_root=m1._required_path(
			inputs, 'label_budget_suite_root', prefix='inputs'
		),
		split_index_suite_root=m1._required_path(
			inputs, 'split_index_suite_root', prefix='inputs'
		),
		class_info_json=class_info,
		monitored_class_ids=tuple(monitored),
		baseline_model=m1._required_str(models, 'baseline', prefix='models'),
		candidate_model=m1._required_str(models, 'candidate', prefix='models'),
		output_dir=m1._required_path(outputs, 'output_dir', prefix='outputs'),
		publish=_publish_config(publish),
	)
	validate_f3_strat_hmm_m2_results_config(resolved)
	return resolved


def validate_f3_strat_hmm_m2_results_config(config: F3StratHMMM2ResultsConfig) -> None:
	"""Fail before decision generation when an input is absent or incomplete."""
	if (
		len(config.monitored_class_ids) != len(REQUIRED_MONITORED_CLASS_IDS)
		or set(config.monitored_class_ids) != REQUIRED_MONITORED_CLASS_IDS
	):
		raise ValueError(
			'monitored_class_ids must contain exactly the required classes 3 and 5'
		)
	for label, path in (
		('baseline_comparison_csv', config.baseline_comparison_csv),
		('m1_metrics_json', config.m1_metrics_json),
		('m2a_metrics_json', config.m2a_metrics_json),
	):
		m1._require_file(path, label=label)
	if config.class_info_json is not None:
		m1._require_file(config.class_info_json, label='class_info_json')
	for label, root in (
		('label_budget_suite_root', config.label_budget_suite_root),
		('split_index_suite_root', config.split_index_suite_root),
	):
		m1._require_dir(root, label=label)
	for label, path in (
		(
			'label-budget paired_metrics_csv',
			config.label_budget_suite_root / 'reports' / 'paired_metrics.csv',
		),
		(
			'label-budget paired_deltas_csv',
			config.label_budget_suite_root / 'reports' / 'paired_deltas.csv',
		),
		(
			'split/index paired_metrics_csv',
			config.split_index_suite_root / 'reports' / 'split_paired_metrics.csv',
		),
		(
			'split/index paired_deltas_csv',
			config.split_index_suite_root / 'reports' / 'split_paired_deltas.csv',
		),
		(
			'label-budget suite_manifest_json',
			config.label_budget_suite_root / 'suite_manifest.json',
		),
		(
			'label-budget probe_run_manifest_json',
			config.label_budget_suite_root / 'probe_run_manifest.json',
		),
		(
			'split/index split_dataset_manifest_json',
			config.split_index_suite_root / 'split_dataset_manifest.json',
		),
		(
			'split/index split_probe_run_manifest_json',
			config.split_index_suite_root / 'split_probe_run_manifest.json',
		),
	):
		m1._require_file(path, label=label)
	if config.baseline_model == config.candidate_model:
		raise ValueError('baseline and candidate model tags must differ')


def consolidate_f3_strat_hmm_m2_results(
	config: F3StratHMMM2ResultsConfig,
) -> F3StratHMMM2ResultsResult:
	"""Aggregate complete M2-A evidence and write the decision artifacts."""
	validate_f3_strat_hmm_m2_results_config(config)
	baseline_metrics = _read_json_object(config.m1_metrics_json)
	candidate_metrics = _read_json_object(config.m2a_metrics_json)
	_validate_metrics_identity(
		baseline_metrics, config.baseline_model, config.m1_metrics_json
	)
	_validate_metrics_identity(
		candidate_metrics, config.candidate_model, config.m2a_metrics_json
	)
	single = m1._single_split_summary(
		config.baseline_comparison_csv,
		baseline_model=config.baseline_model,
		candidate_model=config.candidate_model,
	)
	_validate_single_split_metrics(
		single,
		baseline_metrics=baseline_metrics,
		candidate_metrics=candidate_metrics,
		config=config,
	)
	warnings: list[str] = []
	label_budget, anchors = _label_budget_summary(
		config.label_budget_suite_root / 'reports' / 'paired_deltas.csv',
		config.label_budget_suite_root / 'suite_manifest.json',
		warnings=warnings,
	)
	split_index = m1._split_index_summary(
		config.split_index_suite_root / 'reports' / 'split_paired_deltas.csv',
	)
	if not split_index['splits']:
		raise ValueError('split/index evidence must contain at least one split')
	_validate_suite_evidence(config)
	monitored = _monitored_classes(
		baseline_metrics,
		candidate_metrics,
		config.monitored_class_ids,
		class_info_json=config.class_info_json,
	)
	decision = _decision(single, label_budget, split_index, monitored)
	payload: dict[str, object] = {
		'schema_version': 1,
		'baseline_model': config.baseline_model,
		'candidate_model': config.candidate_model,
		'monitored_class_ids': list(config.monitored_class_ids),
		'single_split': single,
		'per_class': {'classes': monitored},
		'label_budget': label_budget,
		'split_index': _with_split_aggregate(split_index),
		'decision': decision,
		'warnings': warnings,
	}
	config.output_dir.mkdir(parents=True, exist_ok=True)
	figures = (
		*_write_core_figures(payload, config.output_dir, anchors=anchors),
		_write_class_figure(payload, config.output_dir),
	)
	tables = (
		*m1._write_tables(payload, config.output_dir),
		_write_class_table(payload, config.output_dir),
	)
	json_path = config.output_dir / 'm2a_results_summary.json'
	markdown_path = config.output_dir / 'm2a_results_summary.md'
	json_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
	markdown_path.write_text(_render_markdown(payload), encoding='utf-8')
	result = F3StratHMMM2ResultsResult(
		json_path, markdown_path, tables, figures, str(decision['guidance'])
	)
	manifest = publish_f3_strat_hmm_m2_results(result, config.publish)
	return F3StratHMMM2ResultsResult(
		json_path, markdown_path, tables, figures, str(decision['guidance']), manifest
	)


def _decision(
	single: Mapping[str, object],
	label_budget: Mapping[str, object],
	split_index: Mapping[str, object],
	monitored: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	budgets = {str(row['budget_id']): row for row in label_budget['budgets']}
	low_positive = all(
		float(budgets[key]['mean_delta_macro_f1']) > 0
		and float(budgets[key]['mean_delta_mean_iou']) > 0
		for key in REQUIRED_LOW_BUDGETS
	)
	splits = list(split_index['splits'])
	joint_split_wins = sum(
		float(row['delta_macro_f1']) > 0 and float(row['delta_mean_iou']) > 0
		for row in splits
	)
	split_majority = joint_split_wins / len(splits) > 0.5
	balanced_nonnegative = float(single['delta']['balanced_accuracy']) >= 0
	pareto = [
		int(row['class_id'])
		for row in monitored
		if float(row['delta_f1']) >= 0
		and float(row['delta_iou']) >= 0
		and (float(row['delta_f1']) > 0 or float(row['delta_iou']) > 0)
	]
	go_checks = {
		'low_budget_positive': low_positive,
		'split_joint_win_rate_strict_majority': split_majority,
		'full_split_balanced_accuracy_nonnegative': balanced_nonnegative,
		'monitored_class_pareto_improvement': bool(pareto),
	}
	all_low_nonpositive = all(
		float(budgets[key]['mean_delta_macro_f1']) <= 0
		and float(budgets[key]['mean_delta_mean_iou']) <= 0
		for key in REQUIRED_LOW_BUDGETS
	)
	joint_split_losses = sum(
		float(row['delta_macro_f1']) < 0 and float(row['delta_mean_iou']) < 0
		for row in splits
	)
	split_majority_negative = joint_split_losses / len(splits) > 0.5
	all_classes_worse = all(
		float(row['delta_f1']) < 0 and float(row['delta_iou']) < 0 for row in monitored
	)
	primary_improvement = any(
		float(budgets[key]['mean_delta_macro_f1']) > 0
		and float(budgets[key]['mean_delta_mean_iou']) > 0
		for key in REQUIRED_LOW_BUDGETS
	) or bool(joint_split_wins)
	stop_checks = {
		'all_low_budgets_nonpositive': all_low_nonpositive,
		'split_joint_loss_rate_strict_majority': split_majority_negative,
		'all_monitored_classes_worse_without_primary_improvement': all_classes_worse
		and not primary_improvement,
	}
	if all(go_checks.values()):
		guidance = 'go'
		reason_codes = ['all_go_conditions_met']
	elif any(stop_checks.values()):
		guidance = 'stop'
		reason_codes = [key for key, passed in stop_checks.items() if passed]
	else:
		guidance = 'hold'
		reason_codes = [key for key, passed in go_checks.items() if not passed]
	return {
		'guidance': guidance,
		'reason_codes': reason_codes,
		'go_checks': go_checks,
		'stop_checks': stop_checks,
		'evidence': {
			'split_joint_win_count': joint_split_wins,
			'split_joint_win_rate': joint_split_wins / len(splits),
			'split_joint_loss_count': joint_split_losses,
			'split_joint_loss_rate': joint_split_losses / len(splits),
			'pareto_improved_class_ids': pareto,
		},
		'summary': f'M2-A versus M1 mechanical decision: {guidance.upper()}.',
	}


def _validate_single_split_metrics(
	single: Mapping[str, object],
	*,
	baseline_metrics: Mapping[str, object],
	candidate_metrics: Mapping[str, object],
	config: F3StratHMMM2ResultsConfig,
) -> None:
	for role, metrics, metrics_path in (
		('baseline', baseline_metrics, config.m1_metrics_json),
		('candidate', candidate_metrics, config.m2a_metrics_json),
	):
		comparison_metrics = single[role]
		if not isinstance(comparison_metrics, Mapping):
			raise TypeError(f'single_split.{role} must be a mapping')
		for metric in m1.CORE_METRICS:
			json_value = _finite(metrics.get(metric), f'{metrics_path}:{metric}')
			csv_value = float(comparison_metrics[metric])
			if csv_value != json_value:
				raise ValueError(
					'single-split metric mismatch for '
					f'{role}.{metric}: {config.baseline_comparison_csv} has '
					f'{csv_value!r}, '
					f'{metrics_path} has {json_value!r}'
				)


def _monitored_classes(
	baseline: Mapping[str, object],
	candidate: Mapping[str, object],
	class_ids: Sequence[int],
	*,
	class_info_json: Path | None,
) -> list[dict[str, object]]:
	names = _class_names(class_info_json)
	base_f1 = _metric_map(baseline, 'per_class_f1')
	cand_f1 = _metric_map(candidate, 'per_class_f1')
	base_iou = _metric_map(baseline, 'per_class_iou')
	cand_iou = _metric_map(candidate, 'per_class_iou')
	base_support = _metric_map(baseline, 'per_class_support')
	cand_support = _metric_map(candidate, 'per_class_support')
	rows = []
	for class_id in class_ids:
		key = str(class_id)
		for mapping, metric in (
			(base_f1, 'baseline per_class_f1'),
			(cand_f1, 'candidate per_class_f1'),
			(base_iou, 'baseline per_class_iou'),
			(cand_iou, 'candidate per_class_iou'),
			(base_support, 'baseline per_class_support'),
			(cand_support, 'candidate per_class_support'),
		):
			if key not in mapping:
				raise ValueError(f'{metric} is missing monitored class {class_id}')
		bf, cf = (
			_finite(base_f1[key], f'per_class_f1[{key}]'),
			_finite(cand_f1[key], f'per_class_f1[{key}]'),
		)
		bi, ci = (
			_finite(base_iou[key], f'per_class_iou[{key}]'),
			_finite(cand_iou[key], f'per_class_iou[{key}]'),
		)
		bs, cs = _support(base_support[key], key), _support(cand_support[key], key)
		if bs != cs:
			raise ValueError(
				'per-class support identity mismatch for class '
				f'{class_id}: M1={bs}, M2-A={cs}'
			)
		rows.append(
			{
				'class_id': class_id,
				'class_name': names.get(class_id, f'class_{class_id}'),
				'baseline_f1': bf,
				'candidate_f1': cf,
				'delta_f1': cf - bf,
				'baseline_iou': bi,
				'candidate_iou': ci,
				'delta_iou': ci - bi,
				'support': cs,
			}
		)
	return rows


def _with_split_aggregate(split_index: Mapping[str, object]) -> dict[str, object]:
	result = dict(split_index)
	splits = list(split_index['splits'])
	result['mean_deltas'] = {
		metric: sum(float(row[f'delta_{metric}']) for row in splits) / len(splits)
		for metric in m1.DELTA_METRICS
	}
	result['joint_win_rate_macro_f1_mean_iou'] = sum(
		float(row['delta_macro_f1']) > 0 and float(row['delta_mean_iou']) > 0
		for row in splits
	) / len(splits)
	return result


def publish_f3_strat_hmm_m2_results(
	result: F3StratHMMM2ResultsResult, publish_config: F3StratHMMM2PublishConfig | None
) -> PublishManifest | None:
	"""Publish only the declared lightweight M2-A summary allowlist."""
	if publish_config is None or not publish_config.enabled:
		return None
	if publish_config.output_dir is None:
		raise ValueError('publish output_dir is required when publishing is enabled')
	ensure_under_root(
		publish_config.output_dir, root=DEFAULT_RESULTS_ROOT, label='publish.output_dir'
	)
	_validate_m2_publish_contract(result)
	markdown_without_figures = None
	if not publish_config.include_figures:
		markdown_without_figures = (
			'\n'.join(
				line
				for line in result.summary_markdown.read_text(
					encoding='utf-8'
				).splitlines()
				if not line.startswith('![')
			)
			+ '\n'
		)
	items = [
		PublishItem(
			result.summary_markdown,
			Path('m2a_results_summary.md'),
			content_text=markdown_without_figures,
		),
		PublishItem(result.summary_json, Path('m2a_results_summary.json')),
	]
	items.extend(
		PublishItem(path, Path('tables') / path.name) for path in result.table_paths
	)
	if publish_config.include_figures:
		items.extend(
			PublishItem(path, Path('figures') / path.name)
			for path in result.figure_paths
		)
	return publish_selected_results(
		items=tuple(items),
		output_dir=publish_config.output_dir,
		allowed_suffixes=M2_RESULTS_PUBLISH_SUFFIXES,
		max_file_size_bytes=publish_config.max_file_size_bytes,
	)


def _validate_m2_publish_contract(result: F3StratHMMM2ResultsResult) -> None:
	root = result.summary_json.parent.resolve(strict=False)
	expected_summaries = (
		(result.summary_json.resolve(strict=False), root / 'm2a_results_summary.json'),
		(
			result.summary_markdown.resolve(strict=False),
			root / 'm2a_results_summary.md',
		),
	)
	for actual, expected in expected_summaries:
		if actual != expected:
			raise ValueError(
				'M2-A publish summary path does not match the required contract: '
				f'expected {expected}, got {actual}'
			)
	_validate_named_publish_paths(
		result.table_paths,
		expected_names=M2_RESULTS_TABLE_NAMES,
		expected_parent=root / 'tables',
		label='table',
	)
	_validate_named_publish_paths(
		result.figure_paths,
		expected_names=M2_RESULTS_FIGURE_NAMES,
		expected_parent=root / 'figures',
		label='figure',
	)


def _validate_named_publish_paths(
	paths: Sequence[Path],
	*,
	expected_names: frozenset[str],
	expected_parent: Path,
	label: str,
) -> None:
	by_name = {path.name: path.resolve(strict=False) for path in paths}
	if len(by_name) != len(paths) or set(by_name) != expected_names:
		raise ValueError(
			f'M2-A publish {label} allowlist must be exactly '
			f'{sorted(expected_names)!r}; got {sorted(path.name for path in paths)!r}'
		)
	for name, path in by_name.items():
		expected = expected_parent / name
		if path != expected:
			raise ValueError(
				f'M2-A publish {label} path does not match the required contract: '
				f'expected {expected}, got {path}'
			)


def _publish_config(raw: Mapping[str, object]) -> F3StratHMMM2PublishConfig:
	enabled = m1._optional_bool(raw, 'enabled', default=False, prefix='publish')
	include = m1._optional_bool(raw, 'include_figures', default=True, prefix='publish')
	output = m1._optional_publish_path(raw, 'output_dir')
	if enabled and output is None:
		raise ValueError('publish.output_dir must be set when publish.enabled is true')
	if output is not None:
		ensure_under_root(output, root=DEFAULT_RESULTS_ROOT, label='publish.output_dir')
	return F3StratHMMM2PublishConfig(
		enabled, output, include, m1._max_file_size_bytes(raw)
	)


def _required_low_budgets(label_budget: Mapping[str, object]) -> None:
	found = {row['budget_id'] for row in label_budget['budgets']}
	missing = [budget for budget in REQUIRED_LOW_BUDGETS if budget not in found]
	if missing:
		raise ValueError(f'label-budget evidence missing required budgets: {missing}')


def _label_budget_summary(
	paired_deltas_csv: Path,
	suite_manifest_json: Path,
	*,
	warnings: list[str],
) -> tuple[dict[str, object], tuple[str, ...]]:
	"""Summarize M2-A budgets without imposing M1's larger budget grid."""
	rows = m1._read_csv(paired_deltas_csv)
	by_budget: dict[str, list[Mapping[str, str]]] = defaultdict(list)
	for row in rows:
		budget_id = m1._required_cell(row, 'budget_id', path=paired_deltas_csv)
		by_budget[budget_id].append(row)
	_required_low_budgets({'budgets': [{'budget_id': key} for key in by_budget]})
	identity = m1._label_budget_identity_by_condition(suite_manifest_json)
	budgets = []
	anchors = []
	preferred_order = (*REQUIRED_LOW_BUDGETS, 'full')
	for budget_id in (
		*preferred_order,
		*sorted(set(by_budget) - set(preferred_order)),
	):
		if budget_id not in by_budget:
			continue
		raw_rows = by_budget[budget_id]
		if budget_id == 'full' and len(raw_rows) > 1:
			anchors.append(budget_id)
		budget_rows = m1._deduplicated_budget_rows(
			raw_rows,
			identity_by_condition=identity,
			budget_id=budget_id,
			warnings=warnings,
		)
		budgets.append(
			m1._budget_summary_row(budget_id, budget_rows, paired_deltas_csv)
		)
	return {'budgets': budgets}, tuple(anchors)


def _validate_suite_evidence(config: F3StratHMMM2ResultsConfig) -> None:
	_label_budget_manifest_conditions(config)
	_split_manifest_ids(config)
	_validate_run_manifest_binding(
		_RunManifestContract(
			label='label-budget',
			dataset_manifest=config.label_budget_suite_root / 'suite_manifest.json',
			probe_manifest=config.label_budget_suite_root / 'probe_run_manifest.json',
			paired_metrics_csv=(
				config.label_budget_suite_root / 'reports' / 'paired_metrics.csv'
			),
			condition_keys=('budget_id', 'subsample_seed'),
			dataset_artifact_type='f3_lithology_label_budget_suite_manifest',
			probe_artifact_type='f3_lithology_label_budget_probe_run_manifest',
			probe_dataset_field='suite_manifest',
			expected_tags={
				'baseline': config.baseline_model,
				'candidate': config.candidate_model,
			},
		),
	)
	_validate_run_manifest_binding(
		_RunManifestContract(
			label='split/index',
			dataset_manifest=(
				config.split_index_suite_root / 'split_dataset_manifest.json'
			),
			probe_manifest=(
				config.split_index_suite_root / 'split_probe_run_manifest.json'
			),
			paired_metrics_csv=(
				config.split_index_suite_root / 'reports' / 'split_paired_metrics.csv'
			),
			condition_keys=('split_id',),
			dataset_artifact_type='f3_lithology_split_sweep_token_dataset_manifest',
			probe_artifact_type='f3_lithology_split_probe_run_manifest',
			probe_dataset_field='dataset_manifest',
			expected_tags={
				'baseline': config.baseline_model,
				'candidate': config.candidate_model,
			},
		),
	)
	_validate_paired_report_provenance(
		label='label-budget',
		paired_metrics_csv=(
			config.label_budget_suite_root / 'reports' / 'paired_metrics.csv'
		),
		paired_deltas_csv=(
			config.label_budget_suite_root / 'reports' / 'paired_deltas.csv'
		),
		condition_keys=('budget_id', 'subsample_seed'),
		expected_tags={
			'baseline': config.baseline_model,
			'candidate': config.candidate_model,
		},
	)
	_validate_paired_report_provenance(
		label='split/index',
		paired_metrics_csv=(
			config.split_index_suite_root / 'reports' / 'split_paired_metrics.csv'
		),
		paired_deltas_csv=(
			config.split_index_suite_root / 'reports' / 'split_paired_deltas.csv'
		),
		condition_keys=('split_id',),
		expected_tags={
			'baseline': config.baseline_model,
			'candidate': config.candidate_model,
		},
	)


def _validate_run_manifest_binding(contract: _RunManifestContract) -> None:
	dataset_payload = _read_json_object(contract.dataset_manifest)
	probe_payload = _read_json_object(contract.probe_manifest)
	for path, payload, expected_type in (
		(
			contract.dataset_manifest,
			dataset_payload,
			contract.dataset_artifact_type,
		),
		(contract.probe_manifest, probe_payload, contract.probe_artifact_type),
	):
		if payload.get('artifact_type') != expected_type:
			raise ValueError(
				f'{path} artifact_type mismatch: expected {expected_type!r}'
			)
	linked_manifest = _required_manifest_string(
		probe_payload, contract.probe_dataset_field, contract.probe_manifest
	)
	if Path(linked_manifest).resolve(strict=False) != contract.dataset_manifest.resolve(
		strict=False
	):
		raise ValueError(
			f'{contract.label} probe manifest does not reference '
			f'{contract.dataset_manifest}: '
			f'{linked_manifest!r}'
		)
	dataset_rows = _evidence_manifest_rows(
		dataset_payload,
		path=contract.dataset_manifest,
		condition_keys=contract.condition_keys,
		expected_tags=contract.expected_tags,
	)
	probe_rows = _evidence_manifest_rows(
		probe_payload,
		path=contract.probe_manifest,
		condition_keys=contract.condition_keys,
		expected_tags=contract.expected_tags,
	)
	if set(dataset_rows) != set(probe_rows):
		raise ValueError(
			f'{contract.label} dataset/probe manifest row mismatch; '
			f'missing_probe={sorted(set(dataset_rows) - set(probe_rows))!r}, '
			f'missing_dataset={sorted(set(probe_rows) - set(dataset_rows))!r}'
		)
	_validate_dataset_probe_rows(contract, dataset_rows, probe_rows)
	_validate_probe_report_rows(contract, probe_rows)


def _validate_dataset_probe_rows(
	contract: _RunManifestContract,
	dataset_rows: Mapping[tuple[str, ...], Mapping[str, object]],
	probe_rows: Mapping[tuple[str, ...], Mapping[str, object]],
) -> None:
	for key, dataset_row in dataset_rows.items():
		probe_row = probe_rows[key]
		for field_name in ('model_tag', 'paired_identity_hash', 'token_dataset_root'):
			dataset_value = _required_manifest_string(
				dataset_row, field_name, contract.dataset_manifest
			)
			probe_value = _required_manifest_string(
				probe_row, field_name, contract.probe_manifest
			)
			if dataset_value != probe_value:
				raise ValueError(
					f'{contract.label} dataset/probe manifest {field_name} '
					'mismatch for '
					f'{key!r}: dataset={dataset_value!r}, probe={probe_value!r}'
				)


def _validate_probe_report_rows(
	contract: _RunManifestContract,
	probe_rows: Mapping[tuple[str, ...], Mapping[str, object]],
) -> None:
	report_rows = _evidence_report_rows(
		contract.paired_metrics_csv,
		condition_keys=contract.condition_keys,
		expected_tags=contract.expected_tags,
	)
	if set(report_rows) != set(probe_rows):
		raise ValueError(
			f'{contract.label} probe manifest/paired report row mismatch; '
			f'missing_report={sorted(set(probe_rows) - set(report_rows))!r}, '
			f'missing_probe={sorted(set(report_rows) - set(probe_rows))!r}'
		)
	for key, report_row in report_rows.items():
		expected_metrics = _required_manifest_string(
			probe_rows[key], 'metrics_json', contract.probe_manifest
		)
		actual_metrics = m1._required_cell(
			report_row, 'metrics_json', path=contract.paired_metrics_csv
		)
		if actual_metrics != expected_metrics:
			raise ValueError(
				f'{contract.label} probe/paired report metrics provenance '
				'mismatch for '
				f'{key!r}: expected {expected_metrics!r}, got {actual_metrics!r}'
			)


def _evidence_manifest_rows(
	payload: Mapping[str, object],
	*,
	path: Path,
	condition_keys: tuple[str, ...],
	expected_tags: Mapping[str, str],
) -> dict[tuple[str, ...], Mapping[str, object]]:
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError(f'{path} rows must be a list')
	result: dict[tuple[str, ...], Mapping[str, object]] = {}
	for row in rows:
		if not isinstance(row, Mapping):
			raise TypeError(f'{path} row must be a mapping')
		condition = tuple(
			str(_required_manifest_value(row, key, path)) for key in condition_keys
		)
		role = _required_manifest_string(row, 'model_role', path)
		if role not in expected_tags:
			raise ValueError(f'{path} has unexpected model_role={role!r}')
		tag = _required_manifest_string(row, 'model_tag', path)
		if tag != expected_tags[role]:
			raise ValueError(
				f'{path} {role} model identity mismatch: expected '
				f'{expected_tags[role]!r}, got {tag!r}'
			)
		key = (*condition, role)
		if key in result:
			raise ValueError(f'{path} has duplicate row for {key!r}')
		result[key] = row
	return result


def _evidence_report_rows(
	path: Path,
	*,
	condition_keys: tuple[str, ...],
	expected_tags: Mapping[str, str],
) -> dict[tuple[str, ...], Mapping[str, str]]:
	result: dict[tuple[str, ...], Mapping[str, str]] = {}
	for row in m1._read_csv(path):
		condition = tuple(
			m1._required_cell(row, key, path=path) for key in condition_keys
		)
		role = m1._required_cell(row, 'model_role', path=path)
		if role not in expected_tags:
			raise ValueError(f'{path} has unexpected model_role={role!r}')
		key = (*condition, role)
		if key in result:
			raise ValueError(f'{path} has duplicate row for {key!r}')
		result[key] = row
	return result


def _validate_paired_report_provenance(
	*,
	label: str,
	paired_metrics_csv: Path,
	paired_deltas_csv: Path,
	condition_keys: tuple[str, ...],
	expected_tags: Mapping[str, str],
) -> None:
	metrics_by_condition = _paired_metrics_provenance(
		paired_metrics_csv, condition_keys, expected_tags
	)
	delta_conditions: set[tuple[str, ...]] = set()
	for row in m1._read_csv(paired_deltas_csv):
		condition = tuple(
			m1._required_cell(row, key, path=paired_deltas_csv)
			for key in condition_keys
		)
		if condition in delta_conditions:
			raise ValueError(f'{paired_deltas_csv} has duplicate row for {condition!r}')
		delta_conditions.add(condition)
		if condition not in metrics_by_condition:
			raise ValueError(
				f'{label} paired report condition {condition!r} is missing from '
				f'{paired_metrics_csv}'
			)
		for role in expected_tags:
			column = f'{role}_metrics_json'
			actual = m1._required_cell(row, column, path=paired_deltas_csv)
			expected = metrics_by_condition[condition][role]['metrics_json']
			if actual != expected:
				raise ValueError(
					f'{label} metrics provenance mismatch for {condition!r} '
					f'{role}: expected {expected!r}, got {actual!r}'
				)
		_validate_paired_delta_values(
			condition=condition,
			delta_row=row,
			metric_rows=metrics_by_condition[condition],
			paired_metrics_csv=paired_metrics_csv,
			paired_deltas_csv=paired_deltas_csv,
		)
	if delta_conditions != set(metrics_by_condition):
		missing = sorted(set(metrics_by_condition) - delta_conditions)
		raise ValueError(
			f'{label} paired metrics conditions are missing from '
			f'{paired_deltas_csv}: {missing!r}'
		)


def _paired_metrics_provenance(
	paired_metrics_csv: Path,
	condition_keys: tuple[str, ...],
	expected_tags: Mapping[str, str],
) -> dict[tuple[str, ...], dict[str, dict[str, str]]]:
	metrics_by_condition: dict[tuple[str, ...], dict[str, dict[str, str]]] = (
		defaultdict(dict)
	)
	for row in m1._read_csv(paired_metrics_csv):
		condition = tuple(
			m1._required_cell(row, key, path=paired_metrics_csv)
			for key in condition_keys
		)
		role = m1._required_cell(row, 'model_role', path=paired_metrics_csv)
		if role not in expected_tags:
			raise ValueError(f'{paired_metrics_csv} has unexpected model_role={role!r}')
		tag = m1._required_cell(row, 'model_tag', path=paired_metrics_csv)
		if tag != expected_tags[role]:
			raise ValueError(
				f'{paired_metrics_csv} {role} model identity mismatch: expected '
				f'{expected_tags[role]!r}, got {tag!r}'
			)
		if role in metrics_by_condition[condition]:
			raise ValueError(
				f'{paired_metrics_csv} has duplicate {role} row for {condition!r}'
			)
		m1._required_cell(row, 'metrics_json', path=paired_metrics_csv)
		metrics_by_condition[condition][role] = dict(row)
	for condition, roles in metrics_by_condition.items():
		if set(roles) != set(expected_tags):
			raise ValueError(
				f'{paired_metrics_csv} condition {condition!r} requires baseline '
				'and candidate rows'
			)
	return dict(metrics_by_condition)


def _validate_paired_delta_values(
	*,
	condition: tuple[str, ...],
	delta_row: Mapping[str, str],
	metric_rows: Mapping[str, Mapping[str, str]],
	paired_metrics_csv: Path,
	paired_deltas_csv: Path,
) -> None:
	baseline = metric_rows['baseline']
	candidate = metric_rows['candidate']
	for delta_column in (key for key in delta_row if key.startswith('delta_')):
		metric = delta_column.removeprefix('delta_')
		if metric not in baseline or metric not in candidate:
			raise ValueError(
				f'paired metric column {metric!r} for {condition!r} is '
				f'missing from {paired_metrics_csv}'
			)
		values = (baseline[metric], candidate[metric], delta_row[delta_column])
		if all(value in (None, '') for value in values):
			continue
		if any(value in (None, '') for value in values):
			raise ValueError(
				f'paired delta mismatch for {condition!r} {metric}: '
				'baseline, candidate, and delta values must all be present'
			)
		baseline_value = m1._required_float(baseline, metric, path=paired_metrics_csv)
		candidate_value = m1._required_float(candidate, metric, path=paired_metrics_csv)
		actual_delta = m1._required_float(
			delta_row, delta_column, path=paired_deltas_csv
		)
		expected_delta = candidate_value - baseline_value
		if not math.isclose(actual_delta, expected_delta, rel_tol=1e-12, abs_tol=1e-12):
			raise ValueError(
				f'paired delta mismatch for {condition!r} {metric}: '
				f'expected {expected_delta!r} from {paired_metrics_csv}, got '
				f'{actual_delta!r} in {paired_deltas_csv}'
			)


def _label_budget_manifest_conditions(
	config: F3StratHMMM2ResultsConfig,
) -> None:
	csv_path = config.label_budget_suite_root / 'reports' / 'paired_deltas.csv'
	manifest_path = config.label_budget_suite_root / 'suite_manifest.json'
	csv_conditions = [
		(
			m1._required_cell(row, 'budget_id', path=csv_path),
			m1._required_int_cell(row, 'subsample_seed'),
		)
		for row in m1._read_csv(csv_path)
	]
	if len(csv_conditions) != len(set(csv_conditions)):
		raise ValueError(f'duplicate label-budget conditions in {csv_path}')
	manifest_conditions = _manifest_conditions_by_role(
		manifest_path,
		condition_keys=('budget_id', 'subsample_seed'),
		baseline_model=config.baseline_model,
		candidate_model=config.candidate_model,
	)
	expected_conditions = {
		(budget_id, seed)
		for budget_id in REQUIRED_BUDGETS
		for seed in REQUIRED_SUBSAMPLE_SEEDS
	}
	_validate_registered_inventory(
		'label-budget', manifest_conditions, expected_conditions, manifest_path
	)
	_validate_condition_coverage(
		'label-budget', set(csv_conditions), manifest_conditions, manifest_path
	)


def _split_manifest_ids(config: F3StratHMMM2ResultsConfig) -> None:
	csv_path = config.split_index_suite_root / 'reports' / 'split_paired_deltas.csv'
	manifest_path = config.split_index_suite_root / 'split_dataset_manifest.json'
	csv_ids = [
		(m1._required_cell(row, 'split_id', path=csv_path),)
		for row in m1._read_csv(csv_path)
	]
	if len(csv_ids) != len(set(csv_ids)):
		raise ValueError(f'duplicate split/index conditions in {csv_path}')
	manifest_ids = _manifest_conditions_by_role(
		manifest_path,
		condition_keys=('split_id',),
		baseline_model=config.baseline_model,
		candidate_model=config.candidate_model,
	)
	expected_ids = {(split_id,) for split_id in REQUIRED_SPLIT_IDS}
	_validate_registered_inventory(
		'split/index', manifest_ids, expected_ids, manifest_path
	)
	_validate_condition_coverage(
		'split/index', set(csv_ids), manifest_ids, manifest_path
	)


def _validate_registered_inventory(
	label: str,
	actual: set[tuple[object, ...]],
	expected: set[tuple[object, ...]],
	path: Path,
) -> None:
	if actual == expected:
		return
	missing = sorted(expected - actual)
	unexpected = sorted(actual - expected)
	raise ValueError(
		f'{label} manifest does not match the preregistered condition inventory in '
		f'{path}; missing={missing!r}, unexpected={unexpected!r}'
	)


def _manifest_conditions_by_role(
	path: Path,
	*,
	condition_keys: tuple[str, ...],
	baseline_model: str,
	candidate_model: str,
) -> set[tuple[object, ...]]:
	payload = _read_json_object(path)
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError(f'{path} rows must be a list')
	expected_tags = {'baseline': baseline_model, 'candidate': candidate_model}
	by_condition: dict[tuple[object, ...], dict[str, tuple[str, str]]] = defaultdict(
		dict
	)
	for row in rows:
		if not isinstance(row, Mapping):
			raise TypeError(f'{path} row must be a mapping')
		condition = tuple(
			_required_manifest_value(row, key, path) for key in condition_keys
		)
		role = _required_manifest_string(row, 'model_role', path)
		tag = _required_manifest_string(row, 'model_tag', path)
		paired_identity = _required_manifest_string(row, 'paired_identity_hash', path)
		if role not in expected_tags:
			raise ValueError(f'{path} has unexpected model_role={role!r}')
		if tag != expected_tags[role]:
			raise ValueError(
				f'{path} {role} model identity mismatch: expected '
				f'{expected_tags[role]!r}, got {tag!r}'
			)
		if role in by_condition[condition]:
			raise ValueError(f'{path} has duplicate {role} row for {condition!r}')
		by_condition[condition][role] = (tag, paired_identity)
	for condition, roles in by_condition.items():
		if set(roles) != set(expected_tags):
			raise ValueError(
				f'{path} condition {condition!r} requires baseline and candidate rows'
			)
		identities = {paired_identity for _, paired_identity in roles.values()}
		if len(identities) != 1:
			raise ValueError(
				f'{path} condition {condition!r} paired_identity_hash mismatch'
			)
	return set(by_condition)


def _required_manifest_value(row: Mapping[str, object], key: str, path: Path) -> object:
	value = row.get(key)
	if key == 'subsample_seed':
		if isinstance(value, bool) or not isinstance(value, int):
			raise TypeError(f'{path} row {key} must be an integer')
		return value
	if not isinstance(value, str) or not value:
		raise ValueError(f'{path} row {key} must be a non-empty string')
	return value


def _required_manifest_string(row: Mapping[str, object], key: str, path: Path) -> str:
	value = _required_manifest_value(row, key, path)
	if not isinstance(value, str):
		raise TypeError(f'{path} row {key} must be a string')
	return value


def _validate_condition_coverage(
	label: str,
	csv_conditions: set[tuple[object, ...]],
	manifest_conditions: set[tuple[object, ...]],
	manifest_path: Path,
) -> None:
	if csv_conditions == manifest_conditions:
		return
	missing = sorted(manifest_conditions - csv_conditions)
	unexpected = sorted(csv_conditions - manifest_conditions)
	raise ValueError(
		f'{label} evidence does not match {manifest_path}; '
		f'missing={missing!r}, unexpected={unexpected!r}'
	)


def _validate_metrics_identity(
	metrics: Mapping[str, object], expected: str, path: Path
) -> None:
	candidates: list[object] = [metrics.get('model_tag')]
	for key, nested_key in (
		('model', 'tag'),
		('feature_source', 'reference_model_tag'),
	):
		nested = metrics.get(key)
		if isinstance(nested, Mapping):
			candidates.append(nested.get(nested_key))
	values = {value for value in candidates if isinstance(value, str) and value}
	if not values:
		raise ValueError(f'metrics model identity is missing: {path}')
	if values != {expected}:
		raise ValueError(
			f'metrics model identity mismatch in {path}: expected '
			f'{expected!r}, got {sorted(values)!r}'
		)


def _metric_map(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = payload.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return value


def _read_json_object(path: Path) -> Mapping[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'invalid JSON: {path}: {exc.msg}') from exc
	if not isinstance(value, Mapping):
		raise TypeError(f'JSON must contain a mapping: {path}')
	return value


def _finite(value: object, label: str) -> float:
	if (
		isinstance(value, bool)
		or not isinstance(value, int | float)
		or not math.isfinite(float(value))
	):
		raise ValueError(f'{label} must be a finite number')
	return float(value)


def _support(value: object, class_id: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise ValueError(
			f'per_class_support[{class_id}] must be a non-negative integer'
		)
	return value


def _class_names(path: Path | None) -> dict[int, str]:
	if path is None:
		return {}
	payload = _read_json_object(path)
	rows: object = payload.get('classes', payload)
	result: dict[int, str] = {}
	if isinstance(rows, Mapping):
		iterator = (
			{'class_id': key, **value}
			for key, value in rows.items()
			if isinstance(value, Mapping)
		)
	elif isinstance(rows, Sequence) and not isinstance(rows, str | bytes):
		iterator = rows
	else:
		raise TypeError(f'class_info JSON has unsupported structure: {path}')
	for row in iterator:
		if isinstance(row, Mapping):
			class_id, name = row.get('class_id'), row.get('class_name', row.get('name'))
			try:
				class_id = int(class_id)
			except (TypeError, ValueError):
				continue
			if isinstance(name, str) and name:
				result[class_id] = name
	return result


def _write_class_table(payload: Mapping[str, object], output_dir: Path) -> Path:
	path = output_dir / 'tables' / MONITORED_CLASS_TABLE
	rows = payload['per_class']['classes']
	fields = (
		'class_id',
		'class_name',
		'baseline_f1',
		'candidate_f1',
		'delta_f1',
		'baseline_iou',
		'candidate_iou',
		'delta_iou',
		'support',
	)
	m1._write_csv_rows(
		path, fields, [{key: m1._csv_cell(row[key]) for key in fields} for row in rows]
	)
	return path


def _write_class_figure(payload: Mapping[str, object], output_dir: Path) -> Path:
	path = output_dir / 'figures' / MONITORED_CLASS_FIGURE
	rows = payload['per_class']['classes']
	plt = m1._matplotlib_pyplot()
	fig, axis = plt.subplots(figsize=(6.0, 3.6), facecolor='white')
	positions = list(range(len(rows)))
	width = 0.36
	axis.axhline(0.0, linewidth=0.8, linestyle='--')
	axis.bar(
		[x - width / 2 for x in positions],
		[row['delta_f1'] for row in rows],
		width=width,
		label='delta_f1',
	)
	axis.bar(
		[x + width / 2 for x in positions],
		[row['delta_iou'] for row in rows],
		width=width,
		label='delta_iou',
	)
	axis.set_xticks(
		positions, labels=[f'{row["class_id"]}: {row["class_name"]}' for row in rows]
	)
	axis.set_title('Monitored Class Deltas')
	axis.set_ylabel('M2-A - M1')
	axis.grid(axis='y', linewidth=0.6)
	axis.legend(frameon=False)
	fig.tight_layout()
	fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
	plt.close(fig)
	return path


def _write_core_figures(
	payload: Mapping[str, object], output_dir: Path, *, anchors: Sequence[str]
) -> tuple[Path, ...]:
	figures_dir = output_dir / 'figures'
	figures_dir.mkdir(parents=True, exist_ok=True)
	label_path = figures_dir / m1.LABEL_BUDGET_FIGURE
	split_path = figures_dir / m1.SPLIT_INDEX_FIGURE
	single_path = figures_dir / m1.SINGLE_RUN_FIGURE
	plt = m1._matplotlib_pyplot()
	rows = list(payload['label_budget']['budgets'])
	positions = list(range(len(rows)))
	fig, axis = plt.subplots(figsize=(6.0, 3.6), facecolor='white')
	axis.axhline(0.0, linewidth=0.8, linestyle='--')
	for metric in (
		'mean_delta_macro_f1',
		'mean_delta_mean_iou',
		'mean_delta_balanced_accuracy',
	):
		axis.plot(
			positions,
			[row[metric] for row in rows],
			marker='o',
			label=metric,
		)
	axis.set_xticks(
		positions,
		labels=[
			f'{row["budget_id"]}\nanchor'
			if row['budget_id'] in anchors
			else row['budget_id']
			for row in rows
		],
	)
	axis.set_title('Label-Budget Delta Curves')
	axis.set_ylabel('M2-A - M1')
	axis.grid(axis='y', linewidth=0.6)
	axis.legend(frameon=False, fontsize=8)
	fig.tight_layout()
	fig.savefig(label_path, dpi=300, facecolor='white', bbox_inches='tight')
	plt.close(fig)
	m1._save_split_index_deltas(payload['split_index'], split_path, plt=plt)
	m1._save_single_run_metric_comparison(payload['single_split'], single_path, plt=plt)
	return label_path, split_path, single_path


def _render_markdown(payload: Mapping[str, object]) -> str:
	decision = payload['decision']
	lines = [
		'# F3 Strat-HMM M2-A Results Summary',
		'',
		f'- baseline model: {payload["baseline_model"]}',
		f'- candidate model: {payload["candidate_model"]}',
		f'- decision: **{str(decision["guidance"]).upper()}**',
		f'- reason codes: {", ".join(decision["reason_codes"])}',
		'',
		'## Decision Checks',
		'',
	]
	for group in ('go_checks', 'stop_checks'):
		for key, value in decision[group].items():
			lines.append(f'- `{group}.{key}`: `{str(value).lower()}`')
	lines.extend(
		[
			'',
			'## Single Split',
			'',
			f'![Single-run metric comparison](figures/{m1.SINGLE_RUN_FIGURE})',
			'',
			'## Label Budget',
			'',
			f'![Label-budget delta curves](figures/{m1.LABEL_BUDGET_FIGURE})',
			'',
			'## Split/Index',
			'',
			f'![Split/index deltas](figures/{m1.SPLIT_INDEX_FIGURE})',
			'',
			'## Monitored Classes',
			'',
			f'![Monitored class deltas](figures/{MONITORED_CLASS_FIGURE})',
			'',
			'| class | F1 M1 | F1 M2-A | delta F1 | IoU M1 | IoU M2-A | '
			'delta IoU | support |',
			'| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
		]
	)
	lines.extend(
		f'| {row["class_id"]}: {row["class_name"]} | '
		f'{row["baseline_f1"]:.6f} | {row["candidate_f1"]:.6f} | '
		f'{row["delta_f1"]:.6f} | {row["baseline_iou"]:.6f} | '
		f'{row["candidate_iou"]:.6f} | {row["delta_iou"]:.6f} | '
		f'{row["support"]} |'
		for row in payload['per_class']['classes']
	)
	lines.append('')
	return '\n'.join(lines)


__all__ = [
	'F3StratHMMM2PublishConfig',
	'F3StratHMMM2ResultsConfig',
	'F3StratHMMM2ResultsResult',
	'consolidate_f3_strat_hmm_m2_results',
	'f3_strat_hmm_m2_results_config_from_mapping',
	'publish_f3_strat_hmm_m2_results',
	'validate_f3_strat_hmm_m2_results_config',
]
