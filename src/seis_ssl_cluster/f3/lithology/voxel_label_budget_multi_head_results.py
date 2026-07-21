"""Aggregate the fixed K=6/8/10 multi-head voxel-label-budget matrix."""
# ruff: noqa: C901, SLF001

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology import voxel_label_budget_control as control
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as multi_head
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	METRIC_SPECS,
	inspect_f3_lithology_voxel_label_budget_mae_reference_run,
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)
from seis_ssl_cluster.results import (
	PublishItem,
	PublishManifest,
	publish_selected_results,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)

REQUIRED_COMPARISONS = (
	('mh_nocons', 'm1_current_k6'),
	('mh_cons010', 'mh_nocons'),
	('mh_cons010', 'm1_current_k6'),
	('mh_nocons', 'mae'),
	('mh_cons010', 'mae'),
	('m1_current_k6', 'mae'),
)
HISTORICAL_COMPARISONS = (
	('mh_nocons', 'm1'),
	('mh_cons010', 'm1'),
	('m1', 'mae'),
)
# Retain this name for callers that need the required comparison contract.
COMPARISONS = REQUIRED_COMPARISONS
OUTPUT_NAMES = (
	'multi_head_job_metrics.csv',
	'multi_head_paired_metrics.csv',
	'multi_head_paired_deltas.csv',
	'multi_head_summary_by_budget.csv',
	'multi_head_monitored_class_summary.csv',
	'multi_head_pretraining_summary.csv',
	'multi_head_target_diagnostics.csv',
	'multi_head_decisions.json',
	'multi_head_results_summary.json',
	'multi_head_results_summary.md',
	'multi_head_experiment_handoff.md',
)
PUBLISH_MANIFEST_NAME = 'publish_manifest.json'
TARGET_DIAGNOSTIC_FIELDS = (
	'record_type',
	'model_role',
	'model_tag',
	'target_manifest_sha256',
	'head_k',
	'survey_id',
	'head_pair',
	'status',
	'exact',
	'mae',
	'correlation',
	'rank_order_disagreement',
	'diagnostics_json',
)


@dataclass(frozen=True)
class F3VoxelLabelBudgetMultiHeadResultsInspection:
	"""Recomputed metrics, source bindings, and scientific decisions."""

	job_metrics: tuple[Mapping[str, object], ...]
	paired_metrics: tuple[Mapping[str, object], ...]
	paired_deltas: tuple[Mapping[str, object], ...]
	summary_by_budget: tuple[Mapping[str, object], ...]
	monitored_class_summary: tuple[Mapping[str, object], ...]
	pretraining_summary: tuple[Mapping[str, object], ...]
	target_diagnostics: tuple[Mapping[str, object], ...]
	decisions: Mapping[str, object]
	source_identities: Mapping[str, object]


@dataclass(frozen=True)
class F3VoxelLabelBudgetMultiHeadResultsResult:
	"""Files produced by the multi-head aggregation stage."""

	summary_json: Path
	summary_markdown: Path
	decision_json: Path
	table_paths: tuple[Path, ...]
	handoff_markdown: Path
	publish_manifest: PublishManifest | None
	decisions: Mapping[str, object]


def inspect_f3_lithology_voxel_label_budget_multi_head_results(
	config: object,
) -> F3VoxelLabelBudgetMultiHeadResultsInspection:
	"""Validate every source and independently reaggregate all paired deltas."""
	target_manifest = load_multi_head_target_manifest(config.multi_head_target_manifest)
	rows = multi_head.load_f3_lithology_voxel_label_budget_multi_head_rows(config)
	dataset_rows = multi_head._dataset_rows(config)
	current_rows = tuple(
		multi_head._current_k6_rows(config, dataset_rows).values()
	)
	reference = inspect_f3_lithology_voxel_label_budget_mae_reference_run(
		config.dataset_manifest,
		config.original_run_manifest,
		include_historical_m1=(
			config.references.historical_m1_model_id is not None
		),
	)
	members = _members(config, rows, current_rows, reference)
	_validate_pairing(config, members)
	comparisons = _comparisons(members)
	job_metrics = tuple(control._member_metric_row(value) for value in members.values())
	paired_metrics = tuple(_paired_metrics(config, members))
	deltas = tuple(_paired_deltas(config, members, comparisons=comparisons))
	summary = tuple(_summary(config, deltas, comparisons=comparisons))
	monitored = tuple(_monitored(config, summary, comparisons=comparisons))
	pretraining, candidate_diagnostics = _pretraining_evidence(
		config, target_manifest=target_manifest
	)
	diagnostics = _target_diagnostic_rows(
		target_manifest,
		candidate_bindings=candidate_diagnostics,
		target_manifest_sha256=file_sha256(config.multi_head_target_manifest),
	)
	current_k6_mae_parity = _validate_current_k6_mae_parity(
		config, paired_deltas=deltas, summary_by_budget=summary
	)
	decisions = decide_multi_head_comparisons(summary, budgets=config.budgets)
	return F3VoxelLabelBudgetMultiHeadResultsInspection(
		job_metrics=job_metrics,
		paired_metrics=paired_metrics,
		paired_deltas=deltas,
		summary_by_budget=summary,
		monitored_class_summary=monitored,
		pretraining_summary=tuple(pretraining),
		target_diagnostics=tuple(diagnostics),
		decisions=decisions,
		source_identities={
			'dataset_manifest': _json_source_identity(config.dataset_manifest),
			'expected_multi_head_target_manifest': _json_source_identity(
				config.multi_head_target_manifest
			),
			'original_mae_historical_m1_run_manifest': _json_source_identity(
				config.original_run_manifest
			),
			'current_k6_job_manifest': _json_source_identity(
				config.current_k6_run_manifest
			),
			'multi_head_job_manifest': _json_source_identity(
				multi_head.multi_head_run_manifest_path(config)
			),
			'current_k6_control_summary': _json_source_identity(
				_current_k6_control_summary_path(config)
			),
			'pretraining_handoffs': {
				item.model_id: _json_source_identity(
					item.pretraining_handoff, model_tag=item.model_tag
				)
				for item in config.candidates
			},
			'job_sources': _job_source_identities(members),
			'historical_m1_reference': {
				'status': (
					'INCLUDED' if _has_historical_members(members) else 'OMITTED'
				),
			},
			'current_k6_vs_mae_published_parity': current_k6_mae_parity,
		},
	)


def decide_multi_head_comparisons(
	summary_by_budget: Sequence[Mapping[str, object]], *, budgets: Sequence[str]
) -> dict[str, object]:
	"""Apply the preregistered descriptive gates to reaggregated rows."""
	index = {
		(str(row['budget_id']), str(row['comparison_id']), str(row['metric'])): row
		for row in summary_by_budget
	}

	def status(candidate: str, baseline: str) -> dict[str, object]:
		comparison_id = control._comparison_id(candidate, baseline)
		positive_budgets = []
		negative_budgets = []
		for budget in budgets:
			primary = [
				index[(budget, comparison_id, metric)]
				for metric in ('macro_f1', 'mean_iou')
			]
			if all(
				float(row['mean_delta']) > 0.0 and int(row['wins']) >= 4
				for row in primary
			):
				positive_budgets.append(budget)
			if all(
				float(row['mean_delta']) < 0.0 and int(row['wins']) <= 1
				for row in primary
			):
				negative_budgets.append(budget)
		degradations = []
		for class_id in (3, 5):
			for metric in ('f1', 'iou', 'boundary_recall_t2', 'boundary_recall_t4'):
				bad = [
					budget
					for budget in budgets
					if float(
						index[(budget, comparison_id, f'class_{class_id}_{metric}')][
							'mean_delta'
						]
					)
					<= -0.05
				]
				if len(bad) >= 2:
					degradations.append(
						{'class_id': class_id, 'metric': metric, 'budgets': bad}
					)
		state = (
			'POSITIVE'
			if len(positive_budgets) >= 2 and not degradations
			else ('NEGATIVE' if len(negative_budgets) >= 2 else 'HOLD')
		)
		return {
			'comparison_id': comparison_id,
			'status': state,
			'positive_budgets': positive_budgets,
			'negative_budgets': negative_budgets,
			'systematic_major_degradation': degradations,
		}

	effects = {
		'multi_task_value': status('mh_nocons', 'm1_current_k6'),
		'consistency_increment': status('mh_cons010', 'mh_nocons'),
		'main_total_value': status('mh_cons010', 'm1_current_k6'),
		'nocons_vs_mae': status('mh_nocons', 'mae'),
		'main_vs_mae': status('mh_cons010', 'mae'),
		'current_k6_vs_mae': status('m1_current_k6', 'mae'),
	}
	if all(
		effects[key]['status'] == 'POSITIVE'
		for key in ('main_total_value', 'consistency_increment', 'main_vs_mae')
	):
		overall, selected = 'M4_MH_GO_MAIN', 'mh_cons010'
	elif (
		effects['multi_task_value']['status'] == 'POSITIVE'
		and effects['nocons_vs_mae']['status'] == 'POSITIVE'
		and effects['consistency_increment']['status'] != 'POSITIVE'
	):
		overall, selected = 'M4_MH_GO_NOCONS', 'mh_nocons'
	elif (
		effects['multi_task_value']['status'] == 'POSITIVE'
		or effects['main_total_value']['status'] == 'POSITIVE'
	):
		overall, selected = 'M4_MH_HOLD_ATTRIBUTION', None
	else:
		overall, selected = 'M4_MH_HOLD', None
	return {
		'artifact_type': 'f3_multi_head_decisions',
		'schema_version': 1,
		'effects': effects,
		'overall_status': overall,
		'selected_candidate': selected,
		'gate': {
			'minimum_positive_budgets': 2,
			'minimum_primary_wins': 4,
			'major_degradation_delta': -0.05,
			'major_degradation_budget_count': 2,
		},
	}


def summarize_f3_lithology_voxel_label_budget_multi_head(
	config: object, *, publish: bool = True
) -> F3VoxelLabelBudgetMultiHeadResultsResult:
	"""Write report-owned outputs and optionally copy their exact lightweight set."""
	try:
		inspection = inspect_f3_lithology_voxel_label_budget_multi_head_results(config)
	except Exception as error:
		_write_blocked(config, error)
		raise
	reports = config.reports_dir
	reports.mkdir(parents=True, exist_ok=True)
	tables = (
		('multi_head_job_metrics.csv', inspection.job_metrics),
		('multi_head_paired_metrics.csv', inspection.paired_metrics),
		('multi_head_paired_deltas.csv', inspection.paired_deltas),
		('multi_head_summary_by_budget.csv', inspection.summary_by_budget),
		('multi_head_monitored_class_summary.csv', inspection.monitored_class_summary),
		('multi_head_pretraining_summary.csv', inspection.pretraining_summary),
		('multi_head_target_diagnostics.csv', inspection.target_diagnostics),
	)
	for name, rows in tables:
		_write_csv(reports / name, rows)
	payload = _payload(config, inspection)
	decision = reports / 'multi_head_decisions.json'
	summary_json = reports / 'multi_head_results_summary.json'
	summary_md = reports / 'multi_head_results_summary.md'
	_write_json(decision, inspection.decisions)
	_write_json(summary_json, payload)
	summary_md.write_text(_markdown(payload), encoding='utf-8')
	handoff = reports / 'multi_head_experiment_handoff.md'
	handoff.write_text(_handoff(inspection.decisions), encoding='utf-8')
	manifest = None
	if publish:
		items = [PublishItem(reports / name, Path(name)) for name in OUTPUT_NAMES]
		manifest = _publish_multi_head_results(config, items=items)
	return F3VoxelLabelBudgetMultiHeadResultsResult(
		summary_json,
		summary_md,
		decision,
		tuple(reports / name for name, _ in tables),
		handoff,
		manifest,
		inspection.decisions,
	)


def _members(
	config: object,
	candidate_rows: Sequence[Mapping[str, object]],
	current_rows: Sequence[Mapping[str, object]],
	reference: object,
) -> dict[tuple[str, int, str], Mapping[str, object]]:
	members = {}

	def register(
		key: tuple[str, int, str], value: Mapping[str, object]
	) -> None:
		if key in members:
			raise ValueError(f'duplicate multi-head comparison member: {key!r}')
		members[key] = value

	for row in (*candidate_rows, *current_rows):
		role = str(row['model_role'])
		metrics = load_f3_lithology_voxel_label_budget_evaluation_metrics(
			metrics_path=control._identity_path(
				row['evaluation_metrics'], 'evaluation metrics'
			),
			boundary_metrics_path=control._identity_path(
				row['evaluation_boundary_metrics'], 'boundary metrics'
			),
			boundary_region_metrics_path=control._identity_path(
				row['evaluation_boundary_region_metrics'], 'boundary region metrics'
			),
			label=role,
		)
		register((str(row['budget_id']), int(row['subsample_seed']), role), {
			'role': role,
			'model_tag': row['model_tag'],
			'row': row,
			'source_row': row,
			'metrics': metrics,
		})
	historical: dict[tuple[str, int, str], Mapping[str, object]] = {}
	for job in reference.jobs:
		reference_row = control._reference_member_row(job)
		source_row = getattr(job, 'row', reference_row)
		if job.model_role == 'mae':
			register(
				(job.dataset.budget_id, job.dataset.subsample_seed, job.model_role),
				{
					'role': job.model_role,
					'model_tag': job.model_tag,
					'row': reference_row,
					'source_row': source_row,
					'metrics': job.evaluation.metrics,
				},
			)
		elif job.model_role == 'm1':
			key = (job.dataset.budget_id, job.dataset.subsample_seed, job.model_role)
			if key in historical:
				raise ValueError(f'duplicate multi-head comparison member: {key!r}')
			historical[key] = {
				'role': job.model_role,
				'model_tag': job.model_tag,
				'row': reference_row,
				'source_row': source_row,
				'metrics': job.evaluation.metrics,
			}
	expected = {
		(budget, seed, role)
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in ('mae', 'm1_current_k6', 'mh_nocons', 'mh_cons010')
	}
	if set(members) != expected:
		raise ValueError(
			'multi-head comparison member matrix is incomplete or duplicated'
		)
	if _historical_members_are_paired(config, members, historical):
		members.update(historical)
	return members


def _validate_pairing(
	config: object, members: Mapping[tuple[str, int, str], Mapping[str, object]]
) -> None:
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			rows = [
				cast('Mapping[str, object]', members[(budget, seed, role)]['row'])
				for role in ('mae', 'm1_current_k6', 'mh_nocons', 'mh_cons010')
			]
			for key in control.PAIR_IDENTITY_KEYS:
				if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
					raise ValueError(
						f'paired identity mismatch: {budget}/seed{seed}/{key}'
					)


def _historical_members_are_paired(
	config: object,
	members: Mapping[tuple[str, int, str], Mapping[str, object]],
	historical: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> bool:
	"""Admit historical M1 only as a complete, exactly paired reference."""
	expected = {
		(budget, seed, 'm1')
		for budget in config.budgets
		for seed in config.subsample_seeds
	}
	if set(historical) != expected:
		return False
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			m1 = cast('Mapping[str, object]', historical[(budget, seed, 'm1')]['row'])
			for role in ('mae', 'm1_current_k6', 'mh_nocons', 'mh_cons010'):
				other = cast(
					'Mapping[str, object]', members[(budget, seed, role)]['row']
				)
				if any(
					m1.get(key) != other.get(key)
					for key in control.PAIR_IDENTITY_KEYS
				):
					return False
	return True


def _has_historical_members(
	members: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> bool:
	return any(role == 'm1' for _budget, _seed, role in members)


def _comparisons(
	members: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> tuple[tuple[str, str], ...]:
	return REQUIRED_COMPARISONS + (
		HISTORICAL_COMPARISONS if _has_historical_members(members) else ()
	)


def _paired_metrics(
	config: object, members: Mapping[tuple[str, int, str], Mapping[str, object]]
) -> list[dict[str, object]]:
	rows = []
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			condition = [
				members[(budget, seed, role)]
				for role in (
					('mae', 'm1_current_k6', 'mh_nocons', 'mh_cons010')
					+ (('m1',) if _has_historical_members(members) else ())
				)
			]
			first = cast('Mapping[str, object]', condition[0]['row'])
			row = {
				'budget_id': budget,
				'per_class_cap': first['per_class_cap'],
				'subsample_seed': seed,
				'decoder_seed': first['decoder_seed'],
				'validation_voxel_count': first['validation_voxel_count'],
			}
			for member in condition:
				for metric, value in cast(
					'Mapping[str, float]', member['metrics']
				).items():
					row[f'{member["role"]}_{metric}'] = value
			rows.append(row)
	return rows


def _paired_deltas(
	config: object,
	members: Mapping[tuple[str, int, str], Mapping[str, object]],
	*,
	comparisons: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
	rows = []
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			for candidate, baseline in comparisons:
				left, right = (
					members[(budget, seed, candidate)],
					members[(budget, seed, baseline)],
				)
				left_row = cast('Mapping[str, object]', left['row'])
				row = {
					'budget_id': budget,
					'per_class_cap': left_row['per_class_cap'],
					'subsample_seed': seed,
					'decoder_seed': left_row['decoder_seed'],
					'comparison_id': control._comparison_id(candidate, baseline),
					'comparison': f'{candidate} - {baseline}',
					'candidate_model_role': candidate,
					'baseline_model_role': baseline,
					'candidate_model_tag': left['model_tag'],
					'baseline_model_tag': right['model_tag'],
				}
				for metric in METRIC_SPECS:
					row[metric.name] = (
						cast('Mapping[str, float]', left['metrics'])[metric.name]
						- cast('Mapping[str, float]', right['metrics'])[metric.name]
					)
				rows.append(row)
	return rows


def _summary(
	config: object,
	deltas: Sequence[Mapping[str, object]],
	*,
	comparisons: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
	rows = []
	for budget in config.budgets:
		for candidate, baseline in comparisons:
			selected = [
				row
				for row in deltas
				if row['budget_id'] == budget
				and row['comparison_id'] == control._comparison_id(candidate, baseline)
			]
			if len(selected) != 5:
				raise ValueError(
					'every budget/comparison/metric requires five paired seeds'
				)
			for metric in METRIC_SPECS:
				values = [float(row[metric.name]) for row in selected]
				wins = [
					value > 0 if metric.higher_is_better else value < 0
					for value in values
				]
				losses = [
					value < 0 if metric.higher_is_better else value > 0
					for value in values
				]
				worst = (
					min(range(5), key=values.__getitem__)
					if metric.higher_is_better
					else max(range(5), key=values.__getitem__)
				)
				rows.append(
					{
						'budget_id': budget,
						'per_class_cap': int(str(budget).removeprefix('cap')),
						'comparison_id': control._comparison_id(candidate, baseline),
						'comparison': f'{candidate} - {baseline}',
						'candidate_model_role': candidate,
						'baseline_model_role': baseline,
						'metric': metric.name,
						'higher_is_better': metric.higher_is_better,
						'paired_seed_count': 5,
						'mean_delta': statistics.fmean(values),
						'median_delta': statistics.median(values),
						'sample_standard_deviation': statistics.stdev(values),
						'min_delta': min(values),
						'max_delta': max(values),
						'wins': sum(wins),
						'losses': sum(losses),
						'ties': sum(value == 0 for value in values),
						'worst_seed': selected[worst]['subsample_seed'],
						'worst_seed_delta': values[worst],
					}
				)
	return rows


def _current_k6_control_summary_path(config: object) -> Path:
	"""Return the published current-K6 control summary beside its manifest."""
	return config.current_k6_run_manifest.parent / control.CONTROL_SUMMARY_JSON


def _validate_current_k6_mae_parity(
	config: object,
	*,
	paired_deltas: Sequence[Mapping[str, object]],
	summary_by_budget: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
	"""Require recomputed current-K6--MAE values to match the control report."""
	path = _current_k6_control_summary_path(config)
	payload = _read_json(path)
	if (
		payload.get('artifact_type') != 'f3_current_k6_control_summary'
		or payload.get('schema_version') != 1
	):
		raise ValueError('current-K6 control summary type/schema mismatch')
	comparison_id = control._comparison_id('m1_current_k6', 'mae')
	published_deltas = _comparison_rows(
		payload.get('paired_deltas'), comparison_id, label='control paired deltas'
	)
	published_summary = _comparison_rows(
		payload.get('summary_by_budget'), comparison_id, label='control summary'
	)
	recomputed_deltas = [
		row for row in paired_deltas if row['comparison_id'] == comparison_id
	]
	recomputed_summary = [
		row for row in summary_by_budget if row['comparison_id'] == comparison_id
	]
	published_delta_index = {
		(str(row['budget_id']), int(row['subsample_seed'])): row
		for row in published_deltas
	}
	published_summary_index = {
		(str(row['budget_id']), str(row['metric'])): row for row in published_summary
	}
	if len(published_delta_index) != len(recomputed_deltas):
		raise ValueError(
			'published current-K6--MAE paired-delta parity row count mismatch'
		)
	if len(published_summary_index) != len(recomputed_summary):
		raise ValueError('published current-K6--MAE summary parity row count mismatch')
	for row in recomputed_deltas:
		published = published_delta_index.get(
			(str(row['budget_id']), int(row['subsample_seed']))
		)
		if published is None:
			raise ValueError('published current-K6--MAE paired delta is missing')
		for metric in METRIC_SPECS:
			control._assert_float_parity(
				float(row[metric.name]),
				published.get(metric.name),
				label=(
					'published current-K6--MAE paired delta '
					f'{row["budget_id"]}/seed{row["subsample_seed"]}/{metric.name}'
				),
			)
	for row in recomputed_summary:
		published = published_summary_index.get(
			(str(row['budget_id']), str(row['metric']))
		)
		if published is None:
			raise ValueError('published current-K6--MAE summary row is missing')
		for field in (
			'mean_delta',
			'median_delta',
			'sample_standard_deviation',
			'min_delta',
			'max_delta',
			'worst_seed_delta',
		):
			control._assert_float_parity(
				float(row[field]),
				published.get(field),
				label=(
					'published current-K6--MAE summary '
					f'{row["budget_id"]}/{row["metric"]}/{field}'
				),
			)
		for field in ('paired_seed_count', 'wins', 'losses', 'ties', 'worst_seed'):
			if str(row[field]) != str(published.get(field)):
				raise ValueError(
					'published current-K6--MAE summary parity mismatch: '
					f'{row["budget_id"]}/{row["metric"]}/{field}'
				)
	return {
		'status': 'PASS',
		'comparison_id': comparison_id,
		'summary': _json_source_identity(path),
		'paired_delta_row_count': len(recomputed_deltas),
		'summary_row_count': len(recomputed_summary),
	}


def _comparison_rows(
	value: object, comparison_id: str, *, label: str
) -> list[Mapping[str, object]]:
	if not isinstance(value, list) or not all(
		isinstance(row, Mapping) for row in value
	):
		raise TypeError(f'{label} must be a list of mappings')
	return [row for row in value if row.get('comparison_id') == comparison_id]


def _monitored(
	config: object,
	summary: Sequence[Mapping[str, object]],
	*,
	comparisons: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
	index = {
		(row['budget_id'], row['comparison_id'], row['metric']): row for row in summary
	}
	rows = []
	for budget in config.budgets:
		for candidate, baseline in comparisons:
			comparison = control._comparison_id(candidate, baseline)
			for class_id in (3, 5):
				row = {
					'budget_id': budget,
					'comparison_id': comparison,
					'class_id': class_id,
				}
				for metric in ('f1', 'iou', 'boundary_recall_t2', 'boundary_recall_t4'):
					source = index[(budget, comparison, f'class_{class_id}_{metric}')]
					for field in (
						'mean_delta',
						'median_delta',
						'sample_standard_deviation',
						'wins',
						'losses',
						'ties',
					):
						row[f'{metric}_{field}'] = source[field]
				rows.append(row)
	return rows


def _pretraining_evidence(
	config: object,
	*,
	require_embeddings: bool = True,
	target_manifest: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
	"""Validate paired pretraining, optionally requiring extracted embeddings."""
	rows, diagnostics, payloads, checkpoints = [], [], {}, {}
	# A matching digest in both handoffs is not evidence that the configured
	# target is a valid K=6/8/10 multi-head artifact.  The loader verifies its
	# schema, immutable references, and persisted diagnostics before the
	# handoff identities below are trusted.
	if target_manifest is None:
		target_manifest = load_multi_head_target_manifest(
			config.multi_head_target_manifest
		)
	if target_manifest['head_ks'] != [6, 8, 10]:
		raise ValueError('configured target manifest K identity mismatch')
	expected_target_manifest_sha256 = file_sha256(config.multi_head_target_manifest)
	for candidate in config.candidates:
		payload = _read_json(candidate.pretraining_handoff)
		if (
			payload.get('artifact_type') != 'f3_multi_head_pretraining_handoff'
			or payload.get('status') != 'PASS'
		):
			raise ValueError(
				f'pretraining handoff is not a PASS artifact: {candidate.model_id}'
			)
		strat = _mapping(
			payload.get('stratigraphy_pretext'), 'pretraining stratigraphy'
		)
		if strat.get('target_manifest_sha256') != expected_target_manifest_sha256:
			raise ValueError(
				'pretraining handoff target manifest does not match configured target'
			)
		if (
			strat.get('head_ks') != [6, 8, 10]
			or strat.get('head_spec') != 'multi_resolution_ordered_prototypes_v1'
		):
			raise ValueError('pretraining head specification/K mismatch')
		checkpoint = _pretraining_checkpoint(config, candidate, payload)
		if require_embeddings:
			checkpoint = {
				**checkpoint,
				'embedding': _validate_embedding_best_binding(
					config, candidate, checkpoint['path'], payload
				),
			}
			rows.append(
				_pretraining_summary_row(
					candidate=candidate,
					handoff=payload,
					stratigraphy=strat,
					checkpoint=checkpoint,
				)
			)
		diagnostics.append(
			_target_diagnostic_row(
				record_type='candidate_binding',
				model_role=candidate.model_id,
				model_tag=candidate.model_tag,
				target_manifest_sha256=str(
					strat.get('target_manifest_sha256')
				),
				status='PASS',
				diagnostics={
					'head_ks': strat.get('head_ks'),
					'head_spec': strat.get('head_spec'),
					'consistency_policy': strat.get('consistency_policy'),
					'consistency_weight': strat.get('consistency_weight'),
					'scientific_identity_sha256': strat.get(
						'scientific_identity_sha256'
					),
				},
			)
		)
		payloads[candidate.model_id] = payload
		checkpoints[candidate.model_id] = checkpoint
	left, right = (
		_mapping(payloads[key].get('stratigraphy_pretext'), key)
		for key in ('mh_nocons', 'mh_cons010')
	)
	for key in ('head_spec', 'head_ks', 'target_manifest_sha256', 'consistency_policy'):
		if left.get(key) != right.get(key):
			raise ValueError(f'pretraining scientific drift: {key}')
	if (left.get('consistency_weight'), right.get('consistency_weight')) != (0.0, 0.1):
		raise ValueError('pretraining consistency-weight contract mismatch')
	_validate_pretraining_pair(
		checkpoints['mh_nocons'], checkpoints['mh_cons010']
	)
	for row in rows:
		row['initial_state_parity'] = True
	return rows, diagnostics


def _target_diagnostic_rows(
	target_manifest: Mapping[str, object],
	*,
	candidate_bindings: Sequence[Mapping[str, object]],
	target_manifest_sha256: str,
) -> list[dict[str, object]]:
	"""Flatten the validated target-manifest audit evidence into CSV rows."""
	rows = [dict(row) for row in candidate_bindings]
	heads = _mapping(target_manifest.get('heads'), 'target manifest heads')
	for value in target_manifest['head_ks']:
		if isinstance(value, bool) or not isinstance(value, int):
			raise TypeError('target manifest head_ks must contain integers')
		head_k = value
		head = _mapping(heads.get(str(head_k)), f'target manifest head k={head_k}')
		diagnostics = _mapping(
			head.get('diagnostics'), f'target manifest head k={head_k} diagnostics'
		)
		per_survey = _mapping(
			diagnostics.get('per_survey'),
			f'target manifest head k={head_k} per-survey diagnostics',
		)
		rows.extend(
			_target_diagnostic_row(
				record_type='per_head_diagnostic',
				target_manifest_sha256=target_manifest_sha256,
				head_k=head_k,
				survey_id=survey_id,
				status='PASS',
				diagnostics=per_survey[survey_id],
			)
			for survey_id in sorted(per_survey)
		)
	cross_head = _mapping(
		target_manifest.get('cross_head_diagnostics'),
		'target manifest cross-head diagnostics',
	)
	for head_pair in sorted(cross_head):
		metrics = _mapping(
			cross_head[head_pair], f'target manifest cross-head {head_pair}'
		)
		rows.append(
			_target_diagnostic_row(
				record_type='cross_head_diagnostic',
				target_manifest_sha256=target_manifest_sha256,
				head_pair=head_pair,
				status='PASS',
				mae=metrics.get('mae'),
				correlation=metrics.get('correlation'),
				rank_order_disagreement=metrics.get('rank_order_disagreement'),
				diagnostics=metrics,
			)
		)
	if 'k6_replay_parity' in target_manifest:
		parity = _mapping(
			target_manifest['k6_replay_parity'], 'target manifest K=6 replay parity'
		)
		exact = parity.get('exact')
		if not isinstance(exact, bool):
			raise TypeError('target manifest K=6 replay parity exact must be boolean')
		rows.append(
			_target_diagnostic_row(
				record_type='k6_replay_parity',
				target_manifest_sha256=target_manifest_sha256,
				status='PASS' if exact else 'FAIL',
				exact=exact,
				diagnostics=parity,
			)
		)
	return rows


def _target_diagnostic_row(
	*,
	record_type: str,
	target_manifest_sha256: str,
	status: str,
	diagnostics: object,
	**values: object,
) -> dict[str, object]:
	"""Build a fixed-schema CSV record for one target-manifest evidence item."""
	row = {
		'record_type': record_type,
		'model_role': '',
		'model_tag': '',
		'target_manifest_sha256': target_manifest_sha256,
		'head_k': '',
		'survey_id': '',
		'head_pair': '',
		'status': status,
		'exact': '',
		'mae': '',
		'correlation': '',
		'rank_order_disagreement': '',
		'diagnostics_json': _json_cell(diagnostics),
	}
	unknown = set(values) - set(row)
	if unknown:
		raise ValueError(f'unknown target diagnostic fields: {sorted(unknown)!r}')
	row.update(values)
	return row


def _pretraining_checkpoint(
	config: object, candidate: object, handoff: Mapping[str, object]
) -> Mapping[str, object]:
	"""Validate a candidate handoff against its selected pretraining checkpoint."""
	handoff_checkpoint = _mapping(handoff.get('checkpoint'), 'checkpoint')
	checkpoint = Path(str(handoff_checkpoint.get('path', '')))
	if checkpoint.name != 'best.pt' or not checkpoint.is_file():
		raise ValueError('pretraining handoff checkpoint is not an existing best.pt')
	if handoff_checkpoint.get('sha256') != file_sha256(checkpoint):
		raise ValueError('pretraining handoff checkpoint SHA-256 mismatch')
	best = torch.load(checkpoint, map_location='cpu', weights_only=False)
	if not isinstance(best, Mapping):
		raise TypeError('pretraining best.pt payload must be a mapping')
	validate_stratigraphy_checkpoint_payload(best)
	identity = _mapping(
		best.get('stratigraphy_checkpoint'), 'best.pt stratigraphy identity'
	)
	training_config = _mapping(
		best.get('stratigraphy_config'), 'best.pt stratigraphy config'
	)
	if identity.get('model_tag') != candidate.model_tag:
		raise ValueError('pretraining best.pt model tag mismatch')
	if identity.get('head_spec') != 'multi_resolution_ordered_prototypes_v1':
		raise ValueError('pretraining best.pt head specification mismatch')
	if identity.get('head_ks') != [6, 8, 10]:
		raise ValueError('pretraining best.pt K identity mismatch')
	if identity.get('consistency_weight') != (
		0.0 if candidate.model_id == 'mh_nocons' else 0.1
	):
		raise ValueError('pretraining best.pt consistency-weight mismatch')
	latest_path = checkpoint.parent / 'latest.pt'
	if not latest_path.is_file():
		raise FileNotFoundError(f'pretraining latest.pt is missing: {latest_path}')
	latest = torch.load(latest_path, map_location='cpu', weights_only=False)
	if not isinstance(latest, Mapping):
		raise TypeError('pretraining latest.pt payload must be a mapping')
	validate_stratigraphy_checkpoint_payload(latest)
	latest_identity = _mapping(
		latest.get('stratigraphy_checkpoint'), 'latest.pt stratigraphy identity'
	)
	for key in (
		'model_tag',
		'head_spec',
		'head_ks',
		'target_manifest',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
	):
		if latest_identity.get(key) != identity.get(key):
			raise ValueError(
				f'pretraining best/latest scientific identity mismatch: {key}'
			)
	_configured_target_manifest_identity(config, identity)
	return {
		'identity': identity,
		'config': training_config,
		'best': best,
		'latest': latest,
		'path': checkpoint,
	}


def _configured_target_manifest_identity(
	config: object, identity: Mapping[str, object]
) -> None:
	"""Require the selected best checkpoint to use this experiment's target."""
	target = _mapping(
		identity.get('target_manifest'), 'pretraining best.pt target manifest'
	)
	target_path = target.get('path')
	target_sha256 = target.get('sha256')
	if not isinstance(target_path, str) or not isinstance(target_sha256, str):
		raise TypeError(
			'pretraining best.pt target manifest path and SHA-256 must be strings'
		)
	configured_path = Path(config.multi_head_target_manifest).resolve()
	if (
		Path(target_path).resolve() != configured_path
		or target_sha256 != file_sha256(configured_path)
	):
		raise ValueError(
			'pretraining best.pt target manifest does not match configured target'
		)


def _validate_embedding_best_binding(
	config: object,
	candidate: object,
	checkpoint: Path,
	handoff: Mapping[str, object],
) -> Mapping[str, object]:
	"""Require extracted embeddings to bind exactly to the selected best.pt."""
	dataset = _mapping(config.dataset, 'dataset')
	dataset_name = dataset.get('name')
	if not isinstance(dataset_name, str):
		raise TypeError('dataset.name must be a string')
	metadata_path = output_paths(candidate.embeddings_dir, dataset_name).metadata
	metadata = _read_json(metadata_path)
	if (
		Path(str(metadata.get('checkpoint_path', ''))).resolve() != checkpoint.resolve()
		or metadata.get('checkpoint_sha256') != file_sha256(checkpoint)
	):
		raise ValueError('embedding metadata does not bind the selected best.pt')
	if handoff.get('embedding_metadata_sha256') != file_sha256(metadata_path):
		raise ValueError('pretraining handoff metadata SHA-256 mismatch')
	files = output_paths(candidate.embeddings_dir, dataset_name)
	if not files.embeddings.is_file() or not files.valid_tokens.is_file():
		raise FileNotFoundError('pretraining embedding arrays are missing')
	embeddings = np.load(files.embeddings, mmap_mode='r')
	valid_tokens = np.load(files.valid_tokens, mmap_mode='r')
	if embeddings.shape[:3] != valid_tokens.shape or valid_tokens.dtype != np.bool_:
		raise ValueError('pretraining embedding/valid-token array contract mismatch')
	valid_count = int(valid_tokens.sum())
	if valid_count <= 0:
		raise ValueError('pretraining embedding valid-token count must be positive')
	nonfinite_count = _nonfinite_valid_embedding_count(embeddings, valid_tokens)
	if nonfinite_count != 0:
		raise ValueError(
			f'pretraining embeddings contain {nonfinite_count} non-finite valid values'
		)
	return {
		'embeddings': _identity(files.embeddings),
		'valid_tokens': _identity(files.valid_tokens),
		'metadata': _identity(metadata_path),
		'shape': list(embeddings.shape),
		'dtype': str(embeddings.dtype),
		'valid_token_count': valid_count,
		'nonfinite_valid_embedding_count': nonfinite_count,
	}


def _nonfinite_valid_embedding_count(
	embeddings: np.ndarray, valid_tokens: np.ndarray
) -> int:
	"""Count invalid floating-point values without materializing the full volume."""
	count = 0
	for start in range(0, embeddings.shape[0], 4):
		stop = min(embeddings.shape[0], start + 4)
		values = embeddings[start:stop][valid_tokens[start:stop]]
		count += int((~np.isfinite(values)).sum())
	return count


def _pretraining_summary_row(
	*,
	candidate: object,
	handoff: Mapping[str, object],
	stratigraphy: Mapping[str, object],
	checkpoint: Mapping[str, object],
) -> dict[str, object]:
	"""Flatten the mandatory #275 diagnostics for the aggregate CSV."""
	best = _mapping(checkpoint.get('best'), 'best.pt payload')
	latest = _mapping(checkpoint.get('latest'), 'latest.pt payload')
	identity = _mapping(checkpoint.get('identity'), 'best.pt stratigraphy identity')
	config = _mapping(checkpoint.get('config'), 'best.pt stratigraphy config')
	embedding = _mapping(checkpoint.get('embedding'), 'embedding diagnostics')
	best_metrics = _mapping(best.get('metrics'), 'best.pt metrics')
	latest_metrics = _mapping(latest.get('metrics'), 'latest.pt metrics')
	trainability = _mapping(
		latest.get('trainability_summary'), 'latest.pt trainability summary'
	)
	optimizer = _optimizer_contract(
		identity=identity,
		checkpoint=latest,
		training_config=config,
	)
	freeze = _freeze_contract(config=config, trainability=trainability)
	row: dict[str, object] = {
		'model_role': candidate.model_id,
		'model_tag': candidate.model_tag,
		'handoff_sha256': file_sha256(candidate.pretraining_handoff),
		'target_manifest_sha256': stratigraphy.get('target_manifest_sha256'),
		'consistency_weight': stratigraphy.get('consistency_weight'),
		'scientific_identity_sha256': stratigraphy.get('scientific_identity_sha256'),
		'checkpoint_sha256': _mapping(handoff.get('checkpoint'), 'checkpoint').get(
			'sha256'
		),
		'best_epoch': _required_checkpoint_int(best, 'epoch', label='best.pt'),
		'best_global_step': _required_checkpoint_int(
			best, 'global_step', label='best.pt'
		),
		'best_loss': _required_finite_metric(best_metrics, 'loss', label='best.pt'),
		'latest_epoch': _required_checkpoint_int(latest, 'epoch', label='latest.pt'),
		'latest_global_step': _required_checkpoint_int(
			latest, 'global_step', label='latest.pt'
		),
		'latest_loss': _required_finite_metric(
			latest_metrics, 'loss', label='latest.pt'
		),
		'initial_student_state_sha256': identity.get('initial_student_state_sha256'),
		'initial_head_state_sha256': identity.get('initial_head_state_sha256'),
		'freeze_contract_pass': freeze['pass'],
		'freeze_contract': _json_cell(freeze),
		'optimizer_contract_pass': optimizer['pass'],
		'optimizer_contract': _json_cell(optimizer),
		'embedding_identity': _json_cell(embedding),
		'embedding_shape': _json_cell(embedding.get('shape')),
		'embedding_dtype': embedding.get('dtype'),
		'embedding_valid_token_count': embedding.get('valid_token_count'),
		'embedding_nonfinite_valid_embedding_count': embedding.get(
			'nonfinite_valid_embedding_count'
		),
		'embedding_metadata_sha256': _mapping(
			embedding.get('metadata'), 'embedding metadata identity'
		).get('sha256'),
		'embedding_valid_tokens_sha256': _mapping(
			embedding.get('valid_tokens'), 'embedding valid-token identity'
		).get('sha256'),
	}
	row['best_step'] = row['best_global_step']
	row['latest_step'] = row['latest_global_step']
	for k in (6, 8, 10):
		for metric in (
			'loss_prototype',
			'loss_usage',
			'target_usage_entropy',
			'prototype_usage_entropy',
		):
			key = f'{metric}_k{k}'
			row[f'latest_{key}'] = _required_finite_metric(
				latest_metrics, key, label='latest.pt'
			)
	for first_k, second_k in ((6, 8), (6, 10), (8, 10)):
		key = f'loss_consistency_k{first_k}_k{second_k}'
		row[f'latest_{key}'] = _required_finite_metric(
			latest_metrics, key, label='latest.pt'
		)
	return row


def _required_checkpoint_int(
	payload: Mapping[str, object], key: str, *, label: str
) -> int:
	value = payload.get(key)
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise TypeError(f'{label} {key} must be a non-negative integer')
	return value


def _required_finite_metric(
	metrics: Mapping[str, object], key: str, *, label: str
) -> float:
	value = metrics.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{label} metric {key} must be numeric')
	value = float(value)
	if not math.isfinite(value):
		raise ValueError(f'{label} metric {key} must be finite')
	return value


def _freeze_contract(
	*, config: Mapping[str, object], trainability: Mapping[str, object]
) -> dict[str, object]:
	student = _mapping(config.get('student'), 'best.pt student config')
	trainable_names = trainability.get('trainable_names')
	if not isinstance(trainable_names, list) or not all(
		isinstance(name, str) for name in trainable_names
	):
		raise TypeError('latest.pt trainable_names must be a list of strings')
	trainable_count = trainability.get('trainable_parameter_count')
	frozen_count = trainability.get('frozen_parameter_count')
	if (
		isinstance(trainable_count, bool)
		or not isinstance(trainable_count, int)
		or isinstance(frozen_count, bool)
		or not isinstance(frozen_count, int)
	):
		raise TypeError('latest.pt trainability counts must be integers')
	passed = (
		student.get('unfreeze_top_blocks') == 1
		and trainable_count > 0
		and frozen_count > 0
		and bool(trainable_names)
	)
	return {
		'pass': passed,
		'unfreeze_top_blocks': student.get('unfreeze_top_blocks'),
		'trainable_parameter_count': trainable_count,
		'frozen_parameter_count': frozen_count,
		'trainable_names': trainable_names,
	}


def _optimizer_contract(
	*,
	identity: Mapping[str, object],
	checkpoint: Mapping[str, object],
	training_config: Mapping[str, object],
) -> dict[str, object]:
	recorded = identity.get('optimizer_group_identity')
	optimizer_state = _mapping(
		checkpoint.get('optimizer_state_dict'), 'latest.pt optimizer state'
	)
	groups = optimizer_state.get('param_groups')
	train = _mapping(training_config.get('train'), 'best.pt train config')
	if not isinstance(recorded, list) or not isinstance(groups, list):
		raise TypeError('latest.pt optimizer groups are missing')
	if not all(isinstance(group, Mapping) for group in (*recorded, *groups)):
		raise TypeError('latest.pt optimizer groups must be mappings')
	recorded_groups = [cast('Mapping[str, object]', group) for group in recorded]
	state_groups = [cast('Mapping[str, object]', group) for group in groups]
	expected = (('head', train.get('lr')), ('encoder', train.get('encoder_lr')))
	passed = len(recorded_groups) == len(state_groups) == len(expected) and all(
		recorded_group.get('name') == name
		and state_group.get('name') == name
		and recorded_group.get('lr') == lr
		and state_group.get('lr') == lr
		and isinstance(recorded_group.get('parameter_names'), list)
		and isinstance(state_group.get('params'), list)
		and len(recorded_group['parameter_names']) == len(state_group['params']) > 0
		for recorded_group, state_group, (name, lr) in zip(
			recorded_groups, state_groups, expected, strict=True
		)
	)
	return {
		'pass': passed,
		'groups': [
			{
				'name': group.get('name'),
				'lr': group.get('lr'),
				'parameter_count': len(group.get('params', [])),
			}
			for group in state_groups
		],
	}


def _json_cell(value: object) -> str:
	return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _validate_pretraining_pair(
	left: Mapping[str, object], right: Mapping[str, object]
) -> None:
	"""Enforce #275 initialization parity and its four-field diff contract."""
	left_identity = _mapping(left.get('identity'), 'nocons best.pt identity')
	right_identity = _mapping(right.get('identity'), 'cons010 best.pt identity')
	for key in (
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		if left_identity.get(key) != right_identity.get(key):
			raise ValueError(f'pretraining initialization drift: {key}')
	if (
		left_identity.get('teacher_checkpoint_sha256')
		!= left_identity.get('student_init_checkpoint_sha256')
	):
		raise ValueError('pretraining teacher/student initialization mismatch')
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if left_identity.get(key) != right_identity.get(key):
			raise ValueError(f'pretraining initial-state parity mismatch: {key}')
	left_config = _without_allowed_pretraining_differences(
		_mapping(left.get('config'), 'nocons best.pt config'),
		variant='nocons',
		consistency_weight=0.0,
	)
	right_config = _without_allowed_pretraining_differences(
		_mapping(right.get('config'), 'cons010 best.pt config'),
		variant='cons010',
		consistency_weight=0.1,
	)
	if left_config != right_config:
		raise ValueError('pretraining scientific diff exceeds allowed contract')


def _without_allowed_pretraining_differences(
	config: Mapping[str, object], *, variant: str, consistency_weight: float
) -> Mapping[str, object]:
	"""Remove the declared pairwise differences before exact config comparison."""
	copy = cast('dict[str, object]', json.loads(json.dumps(config)))
	loss = _mapping(copy.get('loss'), 'best.pt loss config')
	identity = _mapping(copy.get('identity'), 'best.pt identity config')
	scientific = _mapping(
		identity.get('scientific_identity'), 'best.pt scientific identity config'
	)
	paths = _mapping(copy.get('paths'), 'best.pt paths config')
	if (
		loss.get('consistency_weight') != consistency_weight
		or identity.get('model_tag')
		!= {
			'nocons': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
			'cons010': 'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
		}[variant]
		or scientific.get('variant') != variant
		or scientific.get('consistency_weight') != consistency_weight
		or not isinstance(paths.get('output_root'), str)
	):
		raise ValueError('pretraining allowed-difference values are invalid')
	cast('dict[str, object]', loss).pop('consistency_weight')
	cast('dict[str, object]', identity).pop('model_tag')
	cast('dict[str, object]', scientific).pop('variant')
	# This identity field is deterministically derived from the permitted loss
	# difference by the pretraining resolver.
	cast('dict[str, object]', scientific).pop('consistency_weight')
	cast('dict[str, object]', paths).pop('output_root')
	return copy


def _payload(
	config: object, inspection: F3VoxelLabelBudgetMultiHeadResultsInspection
) -> dict[str, object]:
	return {
		'artifact_type': 'f3_multi_head_voxel_label_budget_results_summary',
		'schema_version': 1,
		'model_tags': {item.model_id: item.model_tag for item in config.candidates},
		'scientific_scope': {
			'dataset_split': 'original F3 split',
			'label_budgets': list(config.budgets),
			'paired_seeds': list(config.subsample_seeds),
			'decoder': 'fixed frozen voxel decoder',
			'limitations': [
				'No voxel-independent p-values, confidence intervals, or '
				'significance claims.',
				'Five seeds are paired label-subset + decoder-init/data-order '
				'conditions, not independent surveys.',
				'K=6/8/10 multi-head results do not select a best K.',
				'consistency_weight=0.1 is one preregistered condition, '
				'not a weight sweep.',
			],
		},
		'decisions': dict(inspection.decisions),
		'source_identities': dict(inspection.source_identities),
		'job_metrics': list(inspection.job_metrics),
		'paired_metrics': list(inspection.paired_metrics),
		'paired_deltas': list(inspection.paired_deltas),
		'summary_by_budget': list(inspection.summary_by_budget),
		'monitored_class_summary': list(inspection.monitored_class_summary),
		'pretraining_summary': list(inspection.pretraining_summary),
		'target_diagnostics': list(inspection.target_diagnostics),
	}


def _markdown(payload: Mapping[str, object]) -> str:
	decisions = cast('Mapping[str, object]', payload['decisions'])
	effects = cast('Mapping[str, Mapping[str, object]]', decisions['effects'])
	return '\n'.join(
		[
			'# K=6/8/10 multi-head low-label voxel results',
			'',
			f'Overall status: `{decisions["overall_status"]}`',
			f'Selected candidate: `{decisions["selected_candidate"]}`',
			'',
			'## Effect decisions',
			'',
			*[f'- {name}: `{value["status"]}`' for name, value in effects.items()],
			'',
			'## Scope and interpretation',
			'',
			'Original F3 split; fixed label budgets; fixed decoder; five paired seeds. '
			'These are descriptive paired-condition results, not voxel-independent '
			'statistical tests or independent-survey inference. A multi-head effect '
			'does not establish a correct individual K, and consistency weight 0.1 was '
			'one fixed condition rather than a sweep. Class and boundary metrics are '
			'reported separately and are not assumed to improve uniformly.',
			'',
		]
	)


def _handoff(decisions: Mapping[str, object]) -> str:
	selected = decisions['selected_candidate']
	if selected is None:
		return '\n'.join(
			[
				'# Six-split confirmatory handoff',
				'',
				f'Overall status: `{decisions["overall_status"]}`',
				'Selected candidate: `None`',
				'',
				'No confirmatory run is authorized. Preserve this report and resolve '
				'the recorded HOLD or BLOCKED evidence before creating a new handoff.',
				'',
			]
		)
	return '\n'.join(
		[
			'# Six-split confirmatory handoff',
			'',
			f'Selected candidate: `{selected}`',
			'',
			'Run MAE, current K6, and the selected multi-head candidate only; '
			'cap25/cap50; split_000 through split_005; one paired decoder seed per '
			'split. Do not carry the unselected no-consistency ablation forward.',
			'',
		]
	)


def _publish_dir(config: object) -> Path:
	return config.results_root / 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_v1'


def _publish_multi_head_results(
	config: object, *, items: Sequence[PublishItem]
) -> PublishManifest:
	"""Publish only the declared review files after an exact-tree preflight."""
	publish_dir = _publish_dir(config)
	_validate_existing_multi_head_publish_tree(publish_dir)
	manifest = publish_selected_results(
		items=items,
		output_dir=publish_dir,
		max_file_size_bytes=10 * 1024 * 1024,
		overwrite=True,
	)
	_validate_published_multi_head_tree(publish_dir, manifest)
	return manifest


def _publish_target_names() -> set[str]:
	return {*OUTPUT_NAMES, PUBLISH_MANIFEST_NAME}


def _validate_existing_multi_head_publish_tree(publish_dir: Path) -> None:
	"""Reject stale outputs instead of publishing a manifest for a mixed tree."""
	if not publish_dir.exists():
		return
	actual = _published_relative_names(publish_dir)
	if not actual:
		return
	expected = _publish_target_names()
	if actual != expected:
		raise FileExistsError(
			'multi-head publish root has an unexpected file set; '
			f'missing={sorted(expected - actual)!r}, '
			f'extra={sorted(actual - expected)!r}'
		)


def _validate_published_multi_head_tree(  # noqa: PLR0912
	publish_dir: Path, manifest: PublishManifest
) -> None:
	"""Verify exact inventory and source/target digests after publication."""
	expected = _publish_target_names()
	actual = _published_relative_names(publish_dir)
	if actual != expected:
		raise ValueError(
			'multi-head published file inventory mismatch; '
			f'missing={sorted(expected - actual)!r}, '
			f'extra={sorted(actual - expected)!r}'
		)
	if manifest.manifest_path != publish_dir.resolve() / PUBLISH_MANIFEST_NAME:
		raise ValueError('multi-head publish manifest path mismatch')
	payload = _read_json(manifest.manifest_path)
	items = payload.get('items')
	if not isinstance(items, list) or len(items) != len(OUTPUT_NAMES):
		raise ValueError('multi-head publish manifest item count mismatch')
	manifest_items: dict[str, Mapping[str, object]] = {}
	for value in items:
		item = _mapping(value, 'multi-head publish manifest item')
		target = item.get('target')
		if not isinstance(target, str) or target in manifest_items:
			raise ValueError('multi-head publish manifest targets are invalid')
		manifest_items[target] = item
	if set(manifest_items) != set(OUTPUT_NAMES):
		raise ValueError('multi-head publish manifest target set mismatch')
	for item in manifest.items:
		target = item.target.resolve()
		if not target.is_file():
			raise FileNotFoundError(
				f'multi-head published target is missing: {target}'
			)
		if not item.source.is_file():
			raise FileNotFoundError(
				f'multi-head publish source is missing: {item.source}'
			)
		if item.size_bytes != target.stat().st_size:
			raise ValueError(f'multi-head published target size mismatch: {target}')
		if item.sha256 != file_sha256(item.source):
			raise ValueError(
				f'multi-head publish source SHA-256 mismatch: {item.source}'
			)
		if item.sha256 != file_sha256(target):
			raise ValueError(f'multi-head published target SHA-256 mismatch: {target}')
		relative_target = target.relative_to(publish_dir.resolve()).as_posix()
		recorded = manifest_items.get(relative_target)
		if recorded is None:
			raise ValueError(f'multi-head publish manifest target is missing: {target}')
		if (
			recorded.get('source') != str(item.source)
			or recorded.get('size_bytes') != item.size_bytes
			or recorded.get('sha256') != item.sha256
		):
			raise ValueError(f'multi-head publish manifest SHA-256 mismatch: {target}')


def _published_relative_names(publish_dir: Path) -> set[str]:
	"""Return an exact flat file inventory and reject opaque tree entries."""
	if not publish_dir.is_dir():
		raise NotADirectoryError(
			f'multi-head publish root is not a directory: {publish_dir}'
		)
	names = set()
	for path in publish_dir.rglob('*'):
		if path.is_symlink() or not path.is_file():
			raise ValueError(f'multi-head publish root has a non-file entry: {path}')
		names.add(path.relative_to(publish_dir).as_posix())
	return names


def _write_blocked(config: object, error: Exception) -> None:
	config.reports_dir.mkdir(parents=True, exist_ok=True)
	_write_json(
		config.reports_dir / 'multi_head_decisions.json',
		{
			'artifact_type': 'f3_multi_head_decisions',
			'schema_version': 1,
			'overall_status': 'M4_MH_BLOCKED',
			'selected_candidate': None,
			'error_type': type(error).__name__,
			'error': str(error),
		},
	)


def _identity(path: Path) -> dict[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _json_source_identity(
	path: Path, *, model_tag: str | None = None
) -> dict[str, object]:
	"""Record a JSON source digest together with its schema and identity fields."""
	payload = _read_json(path)
	rows = payload.get('rows')
	stratigraphy_pretext = payload.get('stratigraphy_pretext')
	if isinstance(stratigraphy_pretext, Mapping):
		scientific_identity = {
			key: stratigraphy_pretext.get(key)
			for key in (
				'head_spec',
				'head_ks',
				'target_manifest_sha256',
				'consistency_policy',
				'consistency_weight',
				'scientific_identity_sha256',
			)
		}
	else:
		scientific_identity = payload.get(
			'scientific_identity', payload.get('preregistered_contract')
		)
	model_tags = (
		{
			str(row['model_role']): row.get('model_tag')
			for row in rows
			if isinstance(row, Mapping)
			and isinstance(row.get('model_role'), str)
		}
		if isinstance(rows, list)
		else None
	)
	return {
		**_identity(path),
		'schema': {
			'artifact_type': payload.get('artifact_type'),
			'schema_version': payload.get('schema_version'),
		},
		'model_tag': model_tag if model_tag is not None else payload.get('model_tag'),
		'model_tags': model_tags,
		'scientific_identity': scientific_identity,
	}


def _job_source_identities(
	members: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
	"""Preserve per-job metric, coverage, schema, tag, and pairing provenance."""
	by_role: dict[str, list[Mapping[str, object]]] = {}
	for (budget, seed, role), member in sorted(members.items()):
		row = _mapping(member.get('row'), 'comparison member row')
		source_row = _mapping(member.get('source_row', row), 'source member row')
		artifacts = {}
		for key in (
			'evaluation_metrics',
			'evaluation_boundary_metrics',
			'evaluation_boundary_region_metrics',
			'prediction_metadata',
		):
			if key in source_row:
				artifacts[key] = _identity(
					control._identity_path(source_row[key], f'{role} {key}')
				)
		scientific_identity = {
			key: row.get(key) for key in control.PAIR_IDENTITY_KEYS
		}
		by_role.setdefault(role, []).append(
			{
				'budget_id': budget,
				'subsample_seed': seed,
				'model_tag': member['model_tag'],
				'schema': {'metric_schema_sha256': row.get('metric_schema_sha256')},
				'scientific_identity': scientific_identity,
				'artifacts': artifacts,
			}
		)
	return {role: tuple(rows) for role, rows in by_role.items()}


def _read_json(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as handle:
		payload = json.load(handle)
	if not isinstance(payload, Mapping):
		raise TypeError(f'expected mapping JSON: {path}')
	return payload


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	if not rows:
		raise ValueError(f'cannot write empty table: {path.name}')
	fields = list(rows[0])
	if any(list(row) != fields for row in rows):
		raise ValueError(f'inconsistent CSV columns: {path.name}')
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)
