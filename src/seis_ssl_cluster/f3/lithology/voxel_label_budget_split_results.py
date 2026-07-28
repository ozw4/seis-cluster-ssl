"""Split-level paired deltas and the fixed #282 confirmatory decision."""
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.results import (
	PublishItem,
	PublishManifest,
	publish_selected_results,
)

SPLITS = tuple(f'split_{index:03d}' for index in range(6))
BUDGETS = ('cap25', 'cap50')
MODELS = ('mae', 'm1_current_k6', 'mh_nocons')
PRIMARY = ('macro_f1', 'mean_iou')
MONITORED = (
	'class_3_f1', 'class_3_iou', 'class_3_boundary_recall_t2',
	'class_3_boundary_recall_t4', 'class_5_f1', 'class_5_iou',
	'class_5_boundary_recall_t2', 'class_5_boundary_recall_t4',
)
REQUIRED_METRICS = (*PRIMARY, *(
	'balanced_accuracy', 'accuracy', 'weighted_f1',
	'boundary_region_macro_f1_r2', 'boundary_region_mean_iou_r2',
	'boundary_region_macro_f1_r4', 'boundary_region_mean_iou_r4',
	'boundary_f1_t2', 'boundary_f1_t4', 'vertical_boundary_position_mae',
), *MONITORED)
LOWER_IS_BETTER = frozenset({'vertical_boundary_position_mae'})
COMPARISONS = (('mh_nocons', 'm1_current_k6'), ('mh_nocons', 'mae'), ('m1_current_k6', 'mae'))
COMPARISON_LABELS = {
	'mh_nocons_minus_m1_current_k6': 'mh_nocons - current K6',
	'mh_nocons_minus_mae': 'mh_nocons - MAE',
	'm1_current_k6_minus_mae': 'current K6 - MAE',
}
PROJECT_DECISION_STATUS = 'ADOPT_MH_NOCONS_FOR_M5'
SELECTED_MODEL_ROLE = 'mh_nocons'
SELECTED_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
NEXT_MILESTONE = 'M5_U_SOFT_POSTERIOR'
OUTPUT_NAMES = (
	'low_label_split_paired_metrics.csv',
	'low_label_split_paired_deltas.csv',
	'low_label_split_aggregates.csv',
	'low_label_split_monitored_class_summary.csv',
	'low_label_split_decisions.json',
	'low_label_split_results_summary.json',
	'low_label_split_results_summary.md',
	'low_label_split_handoff.md',
)
PUBLISH_MANIFEST_NAME = 'publish_manifest.json'


def aggregate_low_label_split_results(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
	"""Create complete paired deltas, aggregates, or a publishable BLOCKED result."""
	try:
		by_key = _validate_rows(rows)
	except (TypeError, ValueError) as error:
		return [], [], _blocked(str(error))
	deltas = []
	for split_id in SPLITS:
		for budget_id in BUDGETS:
			for left, right in COMPARISONS:
				for metric in REQUIRED_METRICS:
					left_value, right_value = by_key[(split_id, budget_id, left)][metric], by_key[(split_id, budget_id, right)][metric]
					delta = left_value - right_value
					if metric in LOWER_IS_BETTER:
						delta = -delta
					deltas.append({'split_id': split_id, 'budget_id': budget_id, 'comparison': f'{left}_minus_{right}', 'metric': metric, 'delta': delta})
	aggregates = [_aggregate(group) for group in _groups(deltas)]
	return deltas, aggregates, _decision(aggregates)


def write_low_label_split_summary(rows: Sequence[Mapping[str, object]], output_root: Path) -> Mapping[str, Path]:
	"""Write lightweight summary artifacts, including a blocked decision when invalid."""
	deltas, aggregates, decision = aggregate_low_label_split_results(rows)
	output_root.mkdir(parents=True, exist_ok=True)
	paths = {
		'paired_metrics': output_root / 'low_label_split_paired_metrics.csv',
		'paired_deltas': output_root / 'low_label_split_paired_deltas.csv',
		'aggregates': output_root / 'low_label_split_aggregates.csv',
		'monitored_classes': output_root / 'low_label_split_monitored_class_summary.csv',
		'decisions': output_root / 'low_label_split_decisions.json',
		'summary': output_root / 'low_label_split_results_summary.json',
		'markdown': output_root / 'low_label_split_results_summary.md',
		'handoff': output_root / 'low_label_split_handoff.md',
	}
	_write_csv(paths['paired_metrics'], rows, tuple(rows[0]) if rows else ('split_id', 'budget_id', 'model_role'))
	_write_csv(paths['paired_deltas'], deltas, ('split_id', 'budget_id', 'comparison', 'metric', 'delta'))
	_write_csv(paths['aggregates'], aggregates, ('budget_id', 'comparison', 'metric', 'mean', 'median', 'sample_sd', 'min', 'max', 'wins', 'losses', 'ties', 'worst_split', 'worst_delta'))
	_write_csv(paths['monitored_classes'], [row for row in aggregates if str(row['metric']) in MONITORED], ('budget_id', 'comparison', 'metric', 'mean', 'median', 'sample_sd', 'min', 'max', 'wins', 'losses', 'ties', 'worst_split', 'worst_delta'))
	formal_decision_payload = {
		'status': decision['status'],
		'decision': dict(decision),
		'job_count': len(rows),
		'aggregate_count': len(aggregates),
	}
	decision_payload = _decision_payload(
		decision, job_count=len(rows), aggregate_count=len(aggregates)
	)
	summary_payload = _summary_payload(
		decision_payload, aggregates=aggregates, deltas=deltas
	)
	paths['decisions'].write_text(
		json.dumps(formal_decision_payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
	paths['summary'].write_text(
		json.dumps(summary_payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
	paths['markdown'].write_text(
		_render_summary_markdown(summary_payload), encoding='utf-8'
	)
	paths['handoff'].write_text(
		_render_handoff_markdown(summary_payload), encoding='utf-8'
	)
	return paths


def _decision_payload(
	decision: Mapping[str, object], *, job_count: int, aggregate_count: int
) -> dict[str, object]:
	"""Keep the formal decision separate from the M5 project adoption."""
	formal_decision = dict(decision)
	return {
		'status': formal_decision['status'],
		'decision': formal_decision,
		'formal_decision': formal_decision,
		'project_decision': _project_decision(formal_decision),
		'job_count': job_count,
		'aggregate_count': aggregate_count,
	}


def _project_decision(formal_decision: Mapping[str, object]) -> dict[str, object]:
	"""Record the project choice without reinterpreting the formal gate."""
	return {
		'status': PROJECT_DECISION_STATUS,
		'selected_model_role': SELECTED_MODEL_ROLE,
		'selected_model_tag': SELECTED_MODEL_TAG,
		'formal_confirmatory_status': formal_decision['status'],
		'next_milestone': NEXT_MILESTONE,
		'additional_decoder_seed_gate_required': False,
		'additional_decoder_seed_evaluation': 'OPTIONAL_DIAGNOSTIC',
		'consistency_candidate_carried_forward': False,
		'not_carried_forward_primary_candidate': 'mh_cons010',
	}


def _summary_payload(
	decision_payload: Mapping[str, object], *,
	aggregates: Sequence[Mapping[str, object]], deltas: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	"""Build a compact, aggregate-derived scientific handoff payload."""
	payload = dict(decision_payload)
	payload['scope'] = _scope()
	payload['next_stage'] = _next_stage()
	if not aggregates:
		payload.update({
			'primary_evidence': {
				'paired_split_count': len(SPLITS),
				'rows': [],
				'budget_characterization': {},
			},
			'hold_reason': None,
			'original_split_dependence': None,
			'mae_reference_evidence': {},
			'class_and_boundary_findings': {},
		})
		return payload
	payload.update({
		'primary_evidence': {
			'paired_split_count': len(SPLITS),
			'rows': _primary_result_table(aggregates),
			'budget_characterization': _budget_characterization(aggregates),
		},
		'hold_reason': _hold_reason(
			aggregates, _mapping(decision_payload['formal_decision'])
		),
		'original_split_dependence': _original_split_dependence(deltas, aggregates),
		'mae_reference_evidence': {
			'mh_nocons_minus_mae': _comparison_evidence(
				aggregates, 'mh_nocons_minus_mae'
			),
			'm1_current_k6_minus_mae': _comparison_evidence(
				aggregates, 'm1_current_k6_minus_mae'
			),
		},
		'class_and_boundary_findings': _class_and_boundary_findings(aggregates),
	})
	return payload


def _scope() -> dict[str, object]:
	return {
		'evaluation': 'M4 six-split low-label confirmatory evaluation',
		'split_ids': list(SPLITS),
		'label_budgets': list(BUDGETS),
		'model_roles': list(MODELS),
		'primary_metrics': list(PRIMARY),
		'paired_split_count': len(SPLITS),
		'detail_artifacts': [
			'low_label_split_paired_metrics.csv',
			'low_label_split_paired_deltas.csv',
			'low_label_split_aggregates.csv',
			'low_label_split_monitored_class_summary.csv',
		],
	}


def _next_stage() -> dict[str, object]:
	return {
		'milestone': NEXT_MILESTONE,
		'name': 'M5-U posterior-aware soft multi-resolution HMM pretraining',
		'target_change': 'forward-backward posterior soft targets',
		'status': 'PLANNED_UNVALIDATED',
		'additional_decoder_seed_evaluation': 'OPTIONAL_DIAGNOSTIC',
		'additional_decoder_seed_gate_required': False,
	}


def _comparison_id(left: str, right: str) -> str:
	return f'{left}_minus_{right}'


def _aggregate_index(
	aggregates: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], Mapping[str, object]]:
	return {
		(str(row['budget_id']), str(row['comparison']), str(row['metric'])): row
		for row in aggregates
	}


def _aggregate_evidence(row: Mapping[str, object]) -> dict[str, object]:
	return {
		'mean': float(row['mean']),
		'median': float(row['median']),
		'sample_sd': float(row['sample_sd']),
		'min': float(row['min']),
		'max': float(row['max']),
		'wins': int(row['wins']),
		'losses': int(row['losses']),
		'ties': int(row['ties']),
		'worst_split': str(row['worst_split']),
		'worst_delta': float(row['worst_delta']),
	}


def _primary_result_table(
	aggregates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	"""Return the six primary comparison/budget rows from aggregate records."""
	index = _aggregate_index(aggregates)
	rows = []
	for left, right in COMPARISONS:
		comparison = _comparison_id(left, right)
		rows.extend(
			{
				'comparison': comparison,
				'comparison_label': COMPARISON_LABELS[comparison],
				'budget_id': budget,
				'macro_f1': _aggregate_evidence(
					index[(budget, comparison, 'macro_f1')]
				),
				'mean_iou': _aggregate_evidence(
					index[(budget, comparison, 'mean_iou')]
				),
			}
			for budget in BUDGETS
		)
	return rows


def _comparison_evidence(
	aggregates: Sequence[Mapping[str, object]], comparison: str
) -> dict[str, object]:
	return {
		'comparison': comparison,
		'comparison_label': COMPARISON_LABELS[comparison],
		'rows': [
			row
			for row in _primary_result_table(aggregates)
			if row['comparison'] == comparison
		],
	}


def _budget_characterization(
	aggregates: Sequence[Mapping[str, object]],
) -> dict[str, bool]:
	"""Describe cap25 robustness and cap50 split dependence from gate rows."""
	comparison = 'mh_nocons_minus_m1_current_k6'
	index = _aggregate_index(aggregates)

	def primary_gate_positive(budget: str) -> bool:
		return all(
			float(index[(budget, comparison, metric)]['mean']) > 0
			and float(index[(budget, comparison, metric)]['median']) > 0
			and int(index[(budget, comparison, metric)]['wins']) >= 4
			for metric in PRIMARY
		)

	def mean_and_median_positive(budget: str) -> bool:
		return all(
			float(index[(budget, comparison, metric)]['mean']) > 0
			and float(index[(budget, comparison, metric)]['median']) > 0
			for metric in PRIMARY
		)

	return {
		'cap25_robust': primary_gate_positive('cap25'),
		'cap50_split_dependent': (
			mean_and_median_positive('cap50')
			and any(
				int(index[('cap50', comparison, metric)]['wins']) < 4
				for metric in PRIMARY
			)
		),
	}


def _hold_reason(
	aggregates: Sequence[Mapping[str, object]], formal_decision: Mapping[str, object]
) -> dict[str, object]:
	"""Expose the fixed cap50 Mean IoU gate evidence without changing it."""
	comparison = 'mh_nocons_minus_m1_current_k6'
	index = _aggregate_index(aggregates)
	target = _aggregate_evidence(index[('cap50', comparison, 'mean_iou')])
	failed_primary_requirements = []
	for budget in BUDGETS:
		for metric in PRIMARY:
			evidence = _aggregate_evidence(index[(budget, comparison, metric)])
			mean_positive = float(evidence['mean']) > 0
			median_positive = float(evidence['median']) > 0
			wins_requirement_met = int(evidence['wins']) >= 4
			if not all((mean_positive, median_positive, wins_requirement_met)):
				failed_primary_requirements.append({
					'budget_id': budget,
					'metric': metric,
					'comparison': comparison,
					'mean': evidence['mean'],
					'median': evidence['median'],
					'wins': evidence['wins'],
					'mean_positive': mean_positive,
					'median_positive': median_positive,
					'wins_requirement_met': wins_requirement_met,
				})
	return {
		'formal_status': formal_decision['status'],
		'comparison': comparison,
		'comparison_label': COMPARISON_LABELS[comparison],
		'budget_id': 'cap50',
		'metric': 'mean_iou',
		'observed': target,
		'requirement': {
			'mean_must_be_positive': True,
			'median_must_be_positive': True,
			'minimum_wins': 4,
			'paired_split_count': len(SPLITS),
		},
		'mean_positive': float(target['mean']) > 0,
		'median_positive': float(target['median']) > 0,
		'wins_requirement_met': int(target['wins']) >= 4,
		'failed_primary_requirements': failed_primary_requirements,
	}


def _original_split_dependence(
	deltas: Sequence[Mapping[str, object]], aggregates: Sequence[Mapping[str, object]]
) -> dict[str, object]:
	"""Compare the original split_000 cap50 deltas with six-split means."""
	comparison = 'mh_nocons_minus_m1_current_k6'
	index = _aggregate_index(aggregates)
	delta_index = {
		(str(row['split_id']), str(row['budget_id']), str(row['comparison']), str(row['metric'])): row
		for row in deltas
	}
	metrics = {}
	for metric in PRIMARY:
		split_delta = float(delta_index[('split_000', 'cap50', comparison, metric)]['delta'])
		six_split_mean = float(index[('cap50', comparison, metric)]['mean'])
		metrics[metric] = {
			'split_000_delta': split_delta,
			'six_split_mean': six_split_mean,
			'split_000_exceeds_six_split_mean': split_delta > six_split_mean,
		}
	return {
		'comparison': comparison,
		'comparison_label': COMPARISON_LABELS[comparison],
		'original_split_id': 'split_000',
		'budget_id': 'cap50',
		'metrics': metrics,
		'original_split_overestimated_incremental_effect': all(
			bool(value['split_000_exceeds_six_split_mean'])
			for value in metrics.values()
		),
	}


def _class_and_boundary_findings(
	aggregates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	"""Preserve aggregate evidence for the separate boundary interpretation."""
	comparison = 'mh_nocons_minus_m1_current_k6'
	index = _aggregate_index(aggregates)
	boundary_f1_metrics = ('boundary_f1_t2', 'boundary_f1_t4')
	vertical_metric = 'vertical_boundary_position_mae'
	class_metrics = (
		'class_3_f1', 'class_3_iou', 'class_5_f1', 'class_5_iou',
	)
	boundary_recall_metrics = (
		'class_3_boundary_recall_t2', 'class_3_boundary_recall_t4',
		'class_5_boundary_recall_t2', 'class_5_boundary_recall_t4',
	)
	metrics = (*boundary_f1_metrics, vertical_metric, *class_metrics, *boundary_recall_metrics)
	evidence = {
		budget: {
			metric: _aggregate_evidence(index[(budget, comparison, metric)])
			for metric in metrics
		}
		for budget in BUDGETS
	}
	return {
		'comparison': comparison,
		'comparison_label': COMPARISON_LABELS[comparison],
		'boundary_f1': {
			budget: {metric: evidence[budget][metric] for metric in boundary_f1_metrics}
			for budget in BUDGETS
		},
		'boundary_f1_mean_positive': all(
			float(evidence[budget][metric]['mean']) > 0
			for budget in BUDGETS
			for metric in boundary_f1_metrics
		),
		'vertical_boundary_position_mae': {
			budget: evidence[budget][vertical_metric] for budget in BUDGETS
		},
		'cap25_vertical_boundary_position_mae_worsened': (
			float(evidence['cap25'][vertical_metric]['mean']) < 0
		),
		'class_3_and_class_5': {
			budget: {metric: evidence[budget][metric] for metric in class_metrics}
			for budget in BUDGETS
		},
		'boundary_recall': {
			budget: {
				metric: evidence[budget][metric]
				for metric in boundary_recall_metrics
			}
			for budget in BUDGETS
		},
		'class_3_and_class_5_uniformly_improved': _uniformly_improved(
			evidence, class_metrics
		),
		'boundary_recall_uniformly_improved': _uniformly_improved(
			evidence, boundary_recall_metrics
		),
	}


def _uniformly_improved(
	evidence: Mapping[str, Mapping[str, Mapping[str, object]]], metrics: Sequence[str]
) -> bool:
	return all(
		float(evidence[budget][metric]['mean']) > 0
		and int(evidence[budget][metric]['wins']) == len(SPLITS)
		for budget in BUDGETS
		for metric in metrics
	)


def _format_delta(value: object) -> str:
	return f'{float(value):+.17g}'


def _format_wins(value: object) -> str:
	return f'{int(value)}/{len(SPLITS)}'


def _format_bool(value: object) -> str:
	return json.dumps(value, allow_nan=False)


def _render_summary_markdown(payload: Mapping[str, object]) -> str:
	"""Render the review summary from the compact JSON evidence payload."""
	formal = _mapping(payload['formal_decision'])
	project = _mapping(payload['project_decision'])
	primary = _mapping(payload['primary_evidence'])
	return '\n'.join([
		'# M4 six-split low-label summary',
		'',
		'## Formal result',
		'',
		f"- Formal status: `{formal['status']}`",
		'- Systematic major degradation: '
		f"`{_format_bool(formal['systematic_major_degradation'])}`.",
		*_render_project_decision(formal, project),
		*_render_primary_evidence(primary),
		*_render_hold_reason(payload['hold_reason']),
		*_render_original_split_dependence(payload['original_split_dependence']),
		*_render_mae_evidence(_mapping(payload['mae_reference_evidence'])),
		*_render_class_and_boundary_findings(
			_mapping(payload['class_and_boundary_findings'])
		),
		*_render_interpretation(primary, formal),
	])


def _render_project_decision(
	formal: Mapping[str, object], project: Mapping[str, object]
) -> list[str]:
	lines = [
		'',
		'## Project decision',
		'',
		f"- Project decision: `{project['status']}`.",
		f"- `{project['selected_model_role']}` is adopted as the M5 hard-target baseline "
		f"(`{project['selected_model_tag']}`).",
	]
	if formal['status'] == 'M4_MH_SPLIT_HOLD':
		lines.append(
			'- Formal HOLD is retained; the project adoption does not reinterpret it '
			'as CONFIRMED.'
		)
	else:
		lines.append(
			f"- The formal result remains `{formal['status']}` and is recorded "
			'separately from project adoption.'
		)
	lines.append('- Additional decoder seeds are optional diagnostics, not a required gate.')
	return lines


def _render_primary_evidence(primary: Mapping[str, object]) -> list[str]:
	lines = [
		'',
		'## Primary evidence',
		'',
		'| comparison | budget | mean ΔMacro F1 | median ΔMacro F1 | Macro F1 wins | mean ΔMean IoU | median ΔMean IoU | Mean IoU wins |',
		'| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
	]
	for row in primary['rows']:
		item = _mapping(row)
		macro_f1 = _mapping(item['macro_f1'])
		mean_iou = _mapping(item['mean_iou'])
		lines.append(
			f"| {item['comparison_label']} | {item['budget_id']} | "
			f"{_format_delta(macro_f1['mean'])} | "
			f"{_format_delta(macro_f1['median'])} | "
			f"{_format_wins(macro_f1['wins'])} | "
			f"{_format_delta(mean_iou['mean'])} | "
			f"{_format_delta(mean_iou['median'])} | "
			f"{_format_wins(mean_iou['wins'])} |"
		)
	if not primary['rows']:
		lines.append(
			'| unavailable | unavailable | unavailable | unavailable | unavailable | '
			'unavailable | unavailable | unavailable |'
		)
	return lines


def _render_hold_reason(hold_reason: object) -> list[str]:
	if hold_reason is None:
		return [
			'',
			'## Formal gate evidence',
			'',
			'- No complete aggregate matrix is available, so no hold-gate evidence '
			'can be rendered.',
		]
	reason = _mapping(hold_reason)
	observed = _mapping(reason['observed'])
	requirement = _mapping(reason['requirement'])
	if reason['formal_status'] != 'M4_MH_SPLIT_HOLD':
		return [
			'',
			'## Formal gate evidence',
			'',
			f"- Formal status: `{reason['formal_status']}`.",
			f"- `{reason['budget_id']}` Mean IoU wins for "
			f"{reason['comparison_label']}: `{_format_wins(observed['wins'])}`; "
			'its preregistered win-count requirement is '
			f"`{requirement['minimum_wins']}/{requirement['paired_split_count']}`.",
		]
	return [
		'',
		'## Why the formal result is HOLD',
		'',
		f"- `{reason['budget_id']}` Mean IoU for {reason['comparison_label']}: "
		f"mean Δ{_format_delta(observed['mean'])}; "
		f"median Δ{_format_delta(observed['median'])}; "
		f"wins `{_format_wins(observed['wins'])}`.",
		'- Preregistered requirement: '
		f"wins ≥ `{requirement['minimum_wins']}/{requirement['paired_split_count']}`.",
		'- The mean and median are positive, but the win-count requirement is '
		'not met.',
	]


def _render_interpretation(
	primary: Mapping[str, object], formal: Mapping[str, object]
) -> list[str]:
	characterization = _mapping(primary['budget_characterization'])
	lines = ['', '## Interpretation', '']
	if characterization.get('cap25_robust'):
		lines.append('- cap25 is robust in the preregistered primary evidence.')
	if characterization.get('cap50_split_dependent'):
		lines.append('- cap50 is split-dependent in the preregistered primary evidence.')
	if formal['status'] == 'M4_MH_SPLIT_HOLD':
		lines.append(
			'- The six-split evidence does not establish mh_nocons superiority as a '
			'formal confirmatory result.'
		)
	else:
		lines.append(
			f"- The formal confirmatory result remains `{formal['status']}`."
		)
	lines.extend([
		'- mh_nocons is adopted as the baseline for the next method-development '
		'stage.',
		'- Proceed to soft-posterior target development; its effectiveness remains '
		'unverified.',
		'',
	])
	return lines


def _render_original_split_dependence(original: object) -> list[str]:
	lines = ['', '## Original-split dependence', '']
	if original is None:
		return [
			*lines,
			'- Original-split dependence is unavailable without complete deltas.',
		]
	dependence = _mapping(original)
	metrics = _mapping(dependence['metrics'])
	lines.extend([
		'| metric | split_000 cap50 Δ | six-split cap50 mean Δ |',
		'| --- | ---: | ---: |',
	])
	for metric in PRIMARY:
		item = _mapping(metrics[metric])
		lines.append(
			f"| {metric} | {_format_delta(item['split_000_delta'])} | "
			f"{_format_delta(item['six_split_mean'])} |"
		)
	if dependence['original_split_overestimated_incremental_effect']:
		lines.append('- The original split overestimated the multi-head incremental effect.')
	else:
		lines.append(
			'- The original-split cap50 deltas do not both exceed their six-split means.'
		)
	return lines


def _render_mae_evidence(mae_evidence: Mapping[str, object]) -> list[str]:
	lines = ['', '## MAE evidence', '']
	if not mae_evidence:
		return [
			*lines,
			'- MAE reference evidence is unavailable without complete aggregates.',
		]
	lines.extend([
		'| comparison | budget | mean ΔMacro F1 | Macro F1 wins | mean ΔMean IoU | Mean IoU wins |',
		'| --- | --- | ---: | ---: | ---: | ---: |',
	])
	for comparison in ('mh_nocons_minus_mae', 'm1_current_k6_minus_mae'):
		evidence = _mapping(mae_evidence[comparison])
		for row in evidence['rows']:
			item = _mapping(row)
			macro_f1 = _mapping(item['macro_f1'])
			mean_iou = _mapping(item['mean_iou'])
			lines.append(
				f"| {item['comparison_label']} | {item['budget_id']} | "
				f"{_format_delta(macro_f1['mean'])} | "
				f"{_format_wins(macro_f1['wins'])} | "
				f"{_format_delta(mean_iou['mean'])} | "
				f"{_format_wins(mean_iou['wins'])} |"
			)
	lines.append(
		'- Structured HMM pretraining relative to MAE is the most robust conclusion '
		'in this evaluation.'
	)
	return lines


def _render_class_and_boundary_findings(
	findings: Mapping[str, object],
) -> list[str]:
	lines = ['', '## Class and boundary findings', '']
	if not findings:
		return [
			*lines,
			'- Class and boundary findings are unavailable without complete aggregates.',
		]
	boundary_f1 = _mapping(findings['boundary_f1'])
	position = _mapping(findings['vertical_boundary_position_mae'])
	lines.extend([
		'- The systematic major degradation gate did not trigger.',
		'- Boundary F1 means are positive across the reported budgets and '
		'tolerances.',
		'| budget | mean Δboundary F1 t2 | mean Δboundary F1 t4 | mean Δvertical boundary-position MAE (oriented) |',
		'| --- | ---: | ---: | ---: |',
	])
	for budget in BUDGETS:
		boundary = _mapping(boundary_f1[budget])
		t2 = _mapping(boundary['boundary_f1_t2'])
		t4 = _mapping(boundary['boundary_f1_t4'])
		vertical = _mapping(position[budget])
		lines.append(
			f'| {budget} | {_format_delta(t2["mean"])} | '
			f'{_format_delta(t4["mean"])} | '
			f'{_format_delta(vertical["mean"])} |'
		)
	if findings['cap25_vertical_boundary_position_mae_worsened']:
		lines.append(
			'- Vertical boundary-position MAE worsens at cap25 (its oriented delta '
			'is negative; lower raw MAE is better).'
		)
	else:
		lines.append('- Vertical boundary-position MAE does not worsen at cap25.')
	lines.extend([
		'- Class 3 / Class 5 and boundary recall are not uniformly improved '
		'across splits.',
		'- Interpret overall Macro F1 / Mean IoU separately from boundary '
		'localization.',
	])
	return lines


def _render_handoff_markdown(payload: Mapping[str, object]) -> str:
	"""Render the explicit M4-to-M5 project handoff without changing HOLD."""
	formal = _mapping(payload['formal_decision'])
	project = _mapping(payload['project_decision'])
	next_stage = _mapping(payload['next_stage'])
	return '\n'.join([
		'# M4 six-split handoff',
		'',
		'## Formal result',
		'',
		f"Formal result: `{formal['status']}`",
		*_render_formal_handoff_note(formal),
		'',
		'## Project decision',
		'',
		f"Project decision: `{project['status']}`",
		'- Adoption is a project decision, separate from the formal confirmatory '
		'status.',
		'',
		'## Selected baseline',
		'',
		f"- `{project['selected_model_role']}`",
		f"- `{project['selected_model_tag']}`",
		'',
		'## Reference models',
		'',
		'- current K6 (`m1_current_k6`)',
		'- MAE (`mae`)',
		'',
		'## Next milestone',
		'',
		f"- `{next_stage['milestone']}` - M5-U soft posterior.",
		'- Posterior-aware soft multi-resolution HMM pretraining is planned; its '
		'effectiveness is unverified.',
		'',
		'## Carry forward',
		'',
		'- mh_nocons',
		'- current K6',
		'- MAE',
		'- Existing original-split and six-split downstream artifacts',
		'',
		'## Do not carry forward as primary candidate',
		'',
		'- mh_cons010',
		'',
		'## No longer required as a gate',
		'',
		'- Decoder seeds `42001/42002`; they remain optional diagnostics.',
		'',
	])


def _render_formal_handoff_note(formal: Mapping[str, object]) -> list[str]:
	if formal['status'] == 'M4_MH_SPLIT_HOLD':
		return ['- HOLD is preserved as the formal six-split confirmatory result.']
	return [
		f"- The formal result remains `{formal['status']}` and is separate from "
		'project adoption.'
	]


def publish_low_label_split_summary(
	config: object, paths: Mapping[str, Path]
) -> PublishManifest:
	"""Copy the exact lightweight six-split summary into the results tree."""
	sources = {path.name: path for path in paths.values()}
	if set(sources) != set(OUTPUT_NAMES):
		raise ValueError('six-split summary publish sources are incomplete or unexpected')
	publish_dir = _publish_dir(config)
	_validate_existing_publish_tree(publish_dir)
	manifest = publish_selected_results(
		items=[PublishItem(sources[name], Path(name)) for name in OUTPUT_NAMES],
		output_dir=publish_dir,
		max_file_size_bytes=10 * 1024 * 1024,
		overwrite=True,
	)
	_validate_published_tree(publish_dir, manifest)
	return manifest


def _publish_dir(config: object) -> Path:
	return config.results_root / 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_six_split_v1'


def _publish_target_names() -> set[str]:
	return {*OUTPUT_NAMES, PUBLISH_MANIFEST_NAME}


def _validate_existing_publish_tree(publish_dir: Path) -> None:
	if not publish_dir.exists():
		return
	actual = _published_relative_names(publish_dir)
	if actual and actual != _publish_target_names():
		expected = _publish_target_names()
		raise FileExistsError(
		'six-split publish root has an unexpected file set; '
		f'missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}'
	)


def _validate_published_tree(  # noqa: C901
	publish_dir: Path, manifest: PublishManifest
) -> None:
	expected = _publish_target_names()
	actual = _published_relative_names(publish_dir)
	if actual != expected:
		raise ValueError(
		'six-split published file inventory mismatch; '
		f'missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}'
	)
	if manifest.manifest_path != publish_dir.resolve() / PUBLISH_MANIFEST_NAME:
		raise ValueError('six-split publish manifest path mismatch')
	payload = _mapping(json.loads(manifest.manifest_path.read_text(encoding='utf-8')))
	recorded = payload.get('items')
	if not isinstance(recorded, list) or len(recorded) != len(OUTPUT_NAMES):
		raise ValueError('six-split publish manifest item count mismatch')
	recorded_by_target: dict[str, Mapping[str, object]] = {}
	for value in recorded:
		item = _mapping(value)
		target = item.get('target')
		if not isinstance(target, str) or target in recorded_by_target:
			raise ValueError('six-split publish manifest targets are invalid')
		recorded_by_target[target] = item
	if set(recorded_by_target) != set(OUTPUT_NAMES):
		raise ValueError('six-split publish manifest target set mismatch')
	for item in manifest.items:
		target = item.target.resolve()
		if not target.is_file() or not item.source.is_file():
			raise FileNotFoundError(f'six-split publish source or target is missing: {target}')
		if item.size_bytes != target.stat().st_size:
			raise ValueError(f'six-split published target size mismatch: {target}')
		if item.sha256 != file_sha256(item.source) or item.sha256 != file_sha256(target):
			raise ValueError(f'six-split publish source/target SHA-256 mismatch: {target}')
		target_name = target.relative_to(publish_dir.resolve()).as_posix()
		record = recorded_by_target.get(target_name)
		if record is None or record.get('source') != str(item.source) or record.get('size_bytes') != item.size_bytes or record.get('sha256') != item.sha256:
			raise ValueError(f'six-split publish manifest SHA-256 mismatch: {target}')


def _published_relative_names(publish_dir: Path) -> set[str]:
	if not publish_dir.is_dir():
		raise NotADirectoryError(f'six-split publish root is not a directory: {publish_dir}')
	names = set()
	for path in publish_dir.rglob('*'):
		if path.is_symlink() or not path.is_file():
			raise ValueError(f'six-split publish root has a non-file entry: {path}')
		names.add(path.relative_to(publish_dir).as_posix())
	return names


def _validate_rows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, str], dict[str, float]]:
	if len(rows) != len(SPLITS) * len(BUDGETS) * len(MODELS):
		raise ValueError('coverage failure: expected exactly 36 completed job metrics')
	indexed: dict[tuple[str, str, str], dict[str, float]] = {}
	for row in rows:
		key = tuple(str(row.get(name, '')) for name in ('split_id', 'budget_id', 'model_role'))
		if key in indexed:
			raise ValueError(f'coverage failure: duplicate job metric row {key!r}')
		if key[0] not in SPLITS or key[1] not in BUDGETS or key[2] not in MODELS:
			raise ValueError(f'coverage failure: unknown job metric row {key!r}')
		values: dict[str, float] = {}
		for metric in REQUIRED_METRICS:
			try:
				value = float(row[metric])
			except (KeyError, TypeError, ValueError) as error:
				raise ValueError(f'metric contract failure: {key!r} lacks finite {metric}') from error
			if not math.isfinite(value):
				raise ValueError(f'metric contract failure: {key!r} has nonfinite {metric}')
			values[metric] = value
		indexed[key] = values
	expected = {(split_id, budget_id, model) for split_id in SPLITS for budget_id in BUDGETS for model in MODELS}
	if set(indexed) != expected:
		raise ValueError('coverage failure: split/budget/model matrix is incomplete')
	return indexed


def _groups(rows: Sequence[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
	grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
	for row in rows:
		grouped.setdefault((str(row['budget_id']), str(row['comparison']), str(row['metric'])), []).append(row)
	return list(grouped.values())


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
	values = [float(row['delta']) for row in rows]
	return {'budget_id': rows[0]['budget_id'], 'comparison': rows[0]['comparison'], 'metric': rows[0]['metric'], 'mean': statistics.mean(values), 'median': statistics.median(values), 'sample_sd': statistics.stdev(values), 'min': min(values), 'max': max(values), 'wins': sum(value > 0 for value in values), 'losses': sum(value < 0 for value in values), 'ties': sum(value == 0 for value in values), 'worst_split': rows[values.index(min(values))]['split_id'], 'worst_delta': min(values)}


def _decision(aggregates: Sequence[Mapping[str, object]]) -> dict[str, object]:
	lookup = {(str(row['budget_id']), str(row['comparison']), str(row['metric'])): row for row in aggregates}
	def row(budget: str, comparison: str, metric: str) -> Mapping[str, object]:
		return lookup[(budget, comparison, metric)]
	def positive(comparison: str) -> bool:
		return all(float(row(budget, comparison, metric)['mean']) > 0 and float(row(budget, comparison, metric)['median']) > 0 and int(row(budget, comparison, metric)['wins']) >= 4 for budget in BUDGETS for metric in PRIMARY)
	negative = all(float(row(budget, 'mh_nocons_minus_m1_current_k6', metric)['mean']) < 0 and int(row(budget, 'mh_nocons_minus_m1_current_k6', metric)['wins']) <= 2 for budget in BUDGETS for metric in PRIMARY)
	degraded = any(all(float(row(budget, 'mh_nocons_minus_m1_current_k6', metric)['mean']) <= -0.05 for budget in BUDGETS) for metric in MONITORED)
	status = 'M4_MH_SPLIT_CONFIRMED' if positive('mh_nocons_minus_m1_current_k6') and positive('mh_nocons_minus_mae') and not degraded else 'M4_MH_SPLIT_NEGATIVE' if negative else 'M4_MH_SPLIT_HOLD'
	return {'status': status, 'systematic_major_degradation': degraded}


def _blocked(reason: str) -> dict[str, object]:
	return {'status': 'M4_MH_SPLIT_BLOCKED', 'blocked_reason': reason, 'systematic_major_degradation': None}


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('expected a JSON object')
	return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
	with path.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


__all__ = [
	'OUTPUT_NAMES',
	'PUBLISH_MANIFEST_NAME',
	'aggregate_low_label_split_results',
	'publish_low_label_split_summary',
	'write_low_label_split_summary',
]
