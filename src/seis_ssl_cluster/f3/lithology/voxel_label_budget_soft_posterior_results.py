"""Aggregate the M5-U original-split screen and apply its preregistered gate."""
# ruff: noqa: SLF001

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology import voxel_label_budget_control as control
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as runner
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head_results as shared,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)
from seis_ssl_cluster.results import (
	PublishItem,
	publish_manifest_to_dict,
	publish_selected_results,
)

COMPARISONS = (
	('mh_soft_nocons', 'mh_nocons'),
	('mh_soft_nocons', 'm1_current_k6'),
	('mh_soft_nocons', 'mae'),
)
OUTPUT_NAMES = (
	'soft_posterior_job_metrics.csv',
	'soft_posterior_paired_deltas.csv',
	'soft_posterior_results_summary.json',
	'soft_posterior_results_summary.md',
	'soft_posterior_handoff.md',
)
_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'


def inspect_f3_lithology_voxel_label_budget_soft_posterior_results(
	config: object,
) -> Mapping[str, object]:
	"""Live-revalidate all sources before any M5-U result is written."""
	soft_rows = runner.load_f3_lithology_voxel_label_budget_multi_head_rows(config)
	dataset_rows = runner._dataset_rows(config)
	current_rows = tuple(runner._current_k6_rows(config, dataset_rows).values())
	reference = runner._mae_reference(config, dataset_rows)
	hard_config = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(config.hard_multi_head_config)
	)
	hard_rows = [
		row
		for row in runner.load_f3_lithology_voxel_label_budget_multi_head_rows(
			hard_config
		)
		if row['model_role'] == 'mh_nocons'
	]
	members = _members((*soft_rows, *current_rows, *hard_rows), reference)
	_expected_matrix(config, members)
	_validate_pairing(config, members)
	deltas = shared._paired_deltas(config, members, comparisons=COMPARISONS)
	summary = shared._summary(config, deltas, comparisons=COMPARISONS)
	decisions = decide_soft_posterior_original_gate(summary, budgets=config.budgets)
	return {
		'job_metrics': tuple(
			control._member_metric_row(value) for value in members.values()
		),
		'paired_deltas': tuple(deltas),
		'summary_by_budget': tuple(summary),
		'decisions': decisions,
		'source_identities': {
			'soft_job_manifest': runner._identity(
				runner.multi_head_run_manifest_path(config)
			),
			'hard_nocons_config': runner._identity(config.hard_multi_head_config),
		},
	}


def decide_soft_posterior_original_gate(
	summary: Sequence[Mapping[str, object]], *, budgets: Sequence[str]
) -> dict[str, object]:
	"""Use M4's existing class-degradation threshold without new thresholds."""
	index = {
		(str(row['budget_id']), str(row['metric'])): row
		for row in summary
		if row['comparison_id'] == 'mh_soft_nocons_vs_mh_nocons'
	}
	positive, negative = [], []
	for budget in budgets:
		primary = [index[(budget, metric)] for metric in ('macro_f1', 'mean_iou')]
		if all(
			float(row['mean_delta']) > 0 and int(row['wins']) >= 4 for row in primary
		):
			positive.append(budget)
		if all(
			float(row['mean_delta']) < 0 and int(row['wins']) <= 1 for row in primary
		):
			negative.append(budget)
	degradations = []
	for class_id in (3, 5):
		for metric in ('f1', 'iou', 'boundary_recall_t2', 'boundary_recall_t4'):
			bad = [
				budget
				for budget in budgets
				if float(index[(budget, f'class_{class_id}_{metric}')]['mean_delta'])
				<= -0.05
			]
			if len(bad) >= 2:
				degradations.append(
					{'class_id': class_id, 'metric': metric, 'budgets': bad}
				)
	if len(positive) >= 2 and not degradations:
		status = 'M5_U_ORIGINAL_GO'
	elif len(negative) >= 2 or degradations:
		status = 'M5_U_ORIGINAL_STOP'
	else:
		status = 'M5_U_ORIGINAL_HOLD'
	return {
		'artifact_type': 'f3_m5_soft_posterior_original_gate',
		'schema_version': 1,
		'overall_status': status,
		'hard_vs_soft': {
			'positive_budgets': positive,
			'negative_budgets': negative,
			'systematic_major_degradation': degradations,
		},
		'gate': {
			'minimum_positive_budgets': 2,
			'minimum_primary_wins': 4,
			'major_degradation_delta': -0.05,
			'major_degradation_budget_count': 2,
		},
		'six_split_follow_up': {
			'ready': status == 'M5_U_ORIGINAL_GO',
			'scientific_jobs_executed': 0,
		},
	}


def summarize_f3_lithology_voxel_label_budget_soft_posterior(
	config: object, *, publish: bool = True
) -> Mapping[str, object]:
	"""Write compact M5-U reports and publish only the allowlisted files."""
	inspection = inspect_f3_lithology_voxel_label_budget_soft_posterior_results(config)
	published_inspection = _portable_payload(inspection, config=config)
	reports = config.reports_dir
	reports.mkdir(parents=True, exist_ok=True)
	job_metrics, deltas = reports / OUTPUT_NAMES[0], reports / OUTPUT_NAMES[1]
	_write_csv(job_metrics, published_inspection['job_metrics'])
	_write_csv(deltas, published_inspection['paired_deltas'])
	payload = dict(published_inspection)
	summary_json, summary_md, handoff = (reports / name for name in OUTPUT_NAMES[2:])
	summary_json.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	status = inspection['decisions']['overall_status']
	ready = inspection['decisions']['six_split_follow_up']['ready']
	summary_md.write_text(
		f'# M5-U original-split screening\n\nStatus: `{status}`\n\n'
		f'Six-split follow-up ready: `{ready}`\n',
		encoding='utf-8',
	)
	handoff.write_text(
		f'M5-U original-split status: `{status}`. '
		'Six-split scientific jobs executed: `0`.\n',
		encoding='utf-8',
	)
	manifest = None
	if publish:
		output = (
			config.base.publish.results_root
			/ 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_soft_posterior_v1'
		)
		manifest = publish_selected_results(
			items=[PublishItem(reports / name, Path(name)) for name in OUTPUT_NAMES],
			output_dir=output,
			max_file_size_bytes=10 * 1024 * 1024,
		)
		_write_portable_publish_manifest(manifest, config=config)
	return {
		'summary_json': summary_json,
		'decisions': inspection['decisions'],
		'publish_manifest': manifest,
	}


def _members(
	rows: Sequence[Mapping[str, object]], reference: object
) -> dict[tuple[str, int, str], Mapping[str, object]]:
	members = {}
	for row in rows:
		role = str(row['model_role'])
		key = (str(row['budget_id']), int(row['subsample_seed']), role)
		if key in members:
			raise ValueError(f'duplicate M5-U member: {key!r}')
		members[key] = {
			'role': role,
			'model_tag': row['model_tag'],
			'row': row,
			'metrics': load_f3_lithology_voxel_label_budget_evaluation_metrics(
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
			),
		}
	for job in reference.jobs:
		key = (job.dataset.budget_id, job.dataset.subsample_seed, job.model_role)
		if key in members:
			raise ValueError(f'duplicate M5-U member: {key!r}')
		members[key] = {
			'role': job.model_role,
			'model_tag': job.model_tag,
			'row': control._reference_member_row(job),
			'metrics': job.evaluation.metrics,
		}
	return members


def _expected_matrix(
	config: object, members: Mapping[tuple[str, int, str], object]
) -> None:
	expected = {
		(budget, seed, role)
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in ('mae', 'm1_current_k6', 'mh_nocons', 'mh_soft_nocons')
	}
	if set(members) != expected:
		raise ValueError('M5-U comparison member matrix is incomplete or duplicated')


def _validate_pairing(
	config: object, members: Mapping[tuple[str, int, str], Mapping[str, object]]
) -> None:
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			rows = [
				members[(budget, seed, role)]['row']
				for role in ('mae', 'm1_current_k6', 'mh_nocons', 'mh_soft_nocons')
			]
			for key in control.PAIR_IDENTITY_KEYS:
				if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
					raise ValueError(
						f'M5-U paired identity mismatch: {budget}/seed{seed}/{key}'
					)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	if not rows:
		raise ValueError(f'no rows to write: {path.name}')
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def _portable_payload(
	payload: object, *, config: object
) -> Mapping[str, object]:
	"""Replace local paths before persisting an M5-U published artifact."""
	published = _portable_value(
		payload,
		artifact_root=config.base.artifact_root,
		workspace_root=config.base.results_root.parent,
	)
	if not isinstance(published, Mapping):
		raise TypeError('published M5-U payload must be a mapping')
	return published


def _portable_value(
	value: object, *, artifact_root: Path, workspace_root: Path
) -> object:
	if isinstance(value, Mapping):
		return {
			key: _portable_value(
				item,
				artifact_root=artifact_root,
				workspace_root=workspace_root,
			)
			for key, item in value.items()
		}
	if isinstance(value, (tuple, list)):
		values = [
			_portable_value(
				item, artifact_root=artifact_root, workspace_root=workspace_root
			)
			for item in value
		]
		return tuple(values) if isinstance(value, tuple) else values
	if not isinstance(value, str):
		return value
	return _portable_path(
		value, artifact_root=artifact_root, workspace_root=workspace_root
	)


def _portable_path(
	value: str, *, artifact_root: Path, workspace_root: Path
) -> str:
	path = Path(value)
	if not path.is_absolute():
		return value
	for root, replacement in (
		(artifact_root, _ARTIFACT_ROOT_PLACEHOLDER),
		(workspace_root, ''),
	):
		try:
			relative = path.relative_to(root)
		except ValueError:
			continue
		if replacement:
			return f'{replacement}/{relative.as_posix()}'
		return relative.as_posix()
	return value


def _write_portable_publish_manifest(manifest: object, *, config: object) -> None:
	"""Keep provenance while omitting machine-specific manifest locations."""
	payload = _portable_payload(publish_manifest_to_dict(manifest), config=config)
	manifest.manifest_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
