"""Aggregate the periodic-refresh original-split screening result."""
# ruff: noqa: C901, E501, PLR0911, SLF001, S603

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh as periodic_config,
)
from seis_ssl_cluster.config import (
	load_config,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked_periodic_refresh as periodic_runner,
)
from seis_ssl_cluster.f3.lithology import voxel_label_budget_control as control
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as runner
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head_results as shared_results,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	METRIC_SPECS,
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)

COMPARISONS = (
	(periodic_config.PERIODIC_REFRESH_MODEL_ID, 'mh_ctmask010_nocons'),
	(periodic_config.PERIODIC_REFRESH_MODEL_ID, 'mh_nocons'),
	(periodic_config.PERIODIC_REFRESH_MODEL_ID, 'm1_current_k6'),
	(periodic_config.PERIODIC_REFRESH_MODEL_ID, 'mae'),
)
REPORT_OUTPUT_NAMES = (
	'periodic_refresh_original_job_metrics.csv',
	'periodic_refresh_original_paired_deltas.csv',
	'periodic_refresh_original_results_summary.json',
	'periodic_refresh_original_results_summary.md',
	'periodic_refresh_original_handoff.json',
)
AUDIT_OUTPUT_NAMES = ('periodic_refresh_screening_audit.json',)
OUTPUT_NAMES = REPORT_OUTPUT_NAMES
PUBLISHED_OUTPUT_NAMES = AUDIT_OUTPUT_NAMES + REPORT_OUTPUT_NAMES
_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
_CANDIDATE_ROLE = periodic_config.PERIODIC_REFRESH_MODEL_ID
_CANDIDATE_TAG = periodic_config.PERIODIC_REFRESH_MODEL_TAG
_FIXED_CENTER_ROLE = 'mh_ctmask010_nocons'
_FIXED_CENTER_TAG = 'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
_SCIENTIFIC_JOB_COUNT = 15
_GATE_METRICS = (
	'macro_f1',
	'mean_iou',
	'class_3_f1',
	'class_3_iou',
	'class_3_boundary_recall_t2',
	'class_3_boundary_recall_t4',
	'class_5_f1',
	'class_5_iou',
	'class_5_boundary_recall_t2',
	'class_5_boundary_recall_t4',
)
_ALL_SUMMARY_METRICS = tuple(spec.name for spec in METRIC_SPECS)
_PUBLISHED_ROOT = Path(
	'f3/facies_benchmark_v1/'
	'strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_original_split_v1'
)


def inspect_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_results(
	config: object,
) -> Mapping[str, object]:
	"""Recompute the exact five-role, 75-row paired original-split matrix."""
	audit = _load_screening_audit(config)
	candidate_inspection = periodic_runner.inspect_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh(
		config
	)
	candidate_rows = periodic_runner.load_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_rows(
		config, inspection=candidate_inspection
	)
	current_rows = tuple(candidate_inspection.current_rows.values())
	hard_rows = candidate_inspection.hard_reference_rows
	fixed_rows = _load_fixed_center_rows(config)
	members = _members(
		(*candidate_rows, *fixed_rows, *current_rows, *hard_rows),
		reference=candidate_inspection.historical_reference,
	)
	_expected_matrix(config, members)
	_validate_pairing(config, members)
	deltas = shared_results._paired_deltas(config, members, comparisons=COMPARISONS)
	summary = shared_results._summary(config, deltas, comparisons=COMPARISONS)
	decisions = decide_center_trace_masked_periodic_refresh_original_gate(
		summary, budgets=config.budgets
	)
	candidate_identity = candidate_inspection.candidate_identities.get(_CANDIDATE_ROLE)
	if not isinstance(candidate_identity, Mapping):
		raise TypeError('periodic-refresh candidate lineage is missing')
	return {
		'job_metrics': tuple(
			control._member_metric_row(value) for value in members.values()
		),
		'paired_deltas': tuple(deltas),
		'summary_by_budget': tuple(summary),
		'decisions': decisions,
		'screening_audit': audit,
		'source_identities': {
			'screening_audit': _identity(Path(config.screening_audit)),
			'candidate_run_manifest': _identity(
				periodic_runner.center_trace_masked_periodic_refresh_run_manifest_path(
					config
				)
			),
			'candidate_provenance': _candidate_provenance(
				config, candidate_identity=candidate_identity
			),
			'periodic_refresh_handoff': _identity(
				Path(config.periodic_refresh_handoff)
			),
			'fixed_center_trace_run_manifest': _identity(
				Path(config.center_trace_masked_run_manifest)
			),
			'hard_decoder_config': _identity(Path(config.hard_multi_head_config)),
			'reference_run_manifests': {
				'hard_multi_head': _identity(
					runner.multi_head_run_manifest_path(_periodic_hard_config(config))
				),
				'current_k6': _identity(Path(config.current_k6_run_manifest)),
				'mae': _identity(Path(config.original_run_manifest)),
			},
			'paired_matrix_identity': {
				'roles': (
					'mae',
					'm1_current_k6',
					'mh_nocons',
					_FIXED_CENTER_ROLE,
					_CANDIDATE_ROLE,
				),
				'budgets': tuple(config.budgets),
				'subsample_seeds': tuple(config.subsample_seeds),
				'row_count': 5 * len(config.budgets) * len(config.subsample_seeds),
				'new_candidate_rows': _SCIENTIFIC_JOB_COUNT,
				'read_only_reference_rows': 60,
				'pair_identity_keys': tuple(control.PAIR_IDENTITY_KEYS),
			},
			'candidate_job_live_validation': {
				'status': 'PASS',
				'expected_count': config.job_count,
				'validated_count': len(candidate_rows),
			},
		},
	}


def decide_center_trace_masked_periodic_refresh_original_gate(
	summary: Sequence[Mapping[str, object]], *, budgets: Sequence[str]
) -> dict[str, object]:
	"""Apply the fixed periodic-refresh original-split GO/HOLD/STOP gate."""
	comparison_id = control._comparison_id(_CANDIDATE_ROLE, _FIXED_CENTER_ROLE)
	index = _validate_gate_summary(
		summary, budgets=budgets, comparison_id=comparison_id
	)
	positive, negative = [], []
	for budget in budgets:
		primary = [
			_summary_row(index, budget, metric) for metric in ('macro_f1', 'mean_iou')
		]
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
				if float(
					_summary_row(index, budget, f'class_{class_id}_{metric}')[
						'mean_delta'
					]
				)
				<= -0.05
			]
			if len(bad) >= 2:
				degradations.append(
					{'class_id': class_id, 'metric': metric, 'budgets': bad}
				)
	if len(positive) >= 2 and not degradations:
		status = 'CTMASK_REFRESH_ORIGINAL_GO'
	elif len(negative) >= 2 or degradations:
		status = 'CTMASK_REFRESH_ORIGINAL_STOP'
	else:
		status = 'CTMASK_REFRESH_ORIGINAL_HOLD'
	return {
		'artifact_type': 'f3_center_trace_masked_periodic_refresh_original_gate',
		'schema_version': 1,
		'overall_status': status,
		'periodic_refresh_vs_fixed_center_trace': {
			'comparison_id': comparison_id,
			'positive_budgets': positive,
			'negative_budgets': negative,
			'systematic_major_degradation': degradations,
		},
		'gate': {
			'primary_comparison_id': comparison_id,
			'primary_metrics': ('macro_f1', 'mean_iou'),
			'positive_mean_delta': '> 0',
			'negative_mean_delta': '< 0',
			'minimum_primary_wins': 4,
			'maximum_negative_primary_wins': 1,
			'minimum_positive_budgets': 2,
			'minimum_negative_budgets': 2,
			'major_degradation_classes': (3, 5),
			'major_degradation_metrics': (
				'f1',
				'iou',
				'boundary_recall_t2',
				'boundary_recall_t4',
			),
			'major_degradation_delta': -0.05,
			'major_degradation_budget_count': 2,
		},
		'six_split_follow_up': {
			'ready': status == 'CTMASK_REFRESH_ORIGINAL_GO',
			'scientific_jobs_executed': 0,
			'six_split_scientific_jobs_executed': 0,
			'six_split_jobs_executed': 0,
		},
		'scientific_jobs_executed': _SCIENTIFIC_JOB_COUNT,
		'six_split_scientific_jobs_executed': 0,
		'six_split_jobs_executed': 0,
	}


def _validate_gate_summary(
	summary: Sequence[Mapping[str, object]],
	*,
	budgets: Sequence[str],
	comparison_id: str,
) -> dict[tuple[str, str], Mapping[str, object]]:
	"""Validate the complete candidate/fixed-center gate matrix."""
	seen: set[tuple[str, str, str]] = set()
	index: dict[tuple[str, str], Mapping[str, object]] = {}
	allowed_comparisons = {
		control._comparison_id(candidate, baseline)
		for candidate, baseline in COMPARISONS
	}
	allowed_budgets = {str(budget) for budget in budgets}
	for row in summary:
		if not isinstance(row, Mapping):
			raise TypeError('periodic-refresh gate summary rows must be mappings')
		try:
			key = (
				str(row['budget_id']),
				str(row['comparison_id']),
				str(row['metric']),
			)
		except KeyError as error:
			raise ValueError(
				'periodic-refresh gate summary row lacks identity'
			) from error
		if key in seen:
			raise ValueError(
				f'periodic-refresh gate summary contains duplicate row: {key!r}'
			)
		seen.add(key)
		if key[0] not in allowed_budgets or key[1] not in allowed_comparisons:
			raise ValueError(f'periodic-refresh gate row has wrong identity: {key!r}')
		try:
			mean_delta = float(row['mean_delta'])
			wins = float(row['wins'])
		except (KeyError, TypeError, ValueError) as error:
			raise ValueError(
				f'periodic-refresh gate metric row is invalid: {key!r}'
			) from error
		if not math.isfinite(mean_delta) or not math.isfinite(wins):
			raise ValueError(f'periodic-refresh gate row is non-finite: {key!r}')
		if wins != int(wins) or not 0 <= wins <= 5:
			raise ValueError(f'periodic-refresh gate win count is invalid: {key!r}')
		if key[1] == comparison_id and key[2] in _GATE_METRICS:
			index[(key[0], key[2])] = row
	expected = {
		(str(budget), control._comparison_id(candidate, baseline), metric)
		for budget in budgets
		for candidate, baseline in COMPARISONS
		for metric in _ALL_SUMMARY_METRICS
	}
	if seen != expected:
		raise ValueError(
			'periodic-refresh gate summary comparison matrix is incomplete or '
			'duplicated: '
			f'missing={sorted(expected - seen)!r}, '
			f'extra={sorted(seen - expected)!r}'
		)
	primary_expected = {
		(str(budget), metric) for budget in budgets for metric in _GATE_METRICS
	}
	if set(index) != primary_expected:
		raise ValueError('periodic-refresh primary gate summary matrix is incomplete')
	return index


def summarize_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh(
	config: object, *, publish: bool = True
) -> Mapping[str, object]:
	"""Write lightweight paired reports and optionally publish the closed set."""
	inspection = inspect_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_results(
		config
	)
	execution = _execution_git_state(config)
	reports = Path(config.reports_dir)
	reports.mkdir(parents=True, exist_ok=True)
	published = _portable_payload(inspection, config=config)
	_write_json(reports / AUDIT_OUTPUT_NAMES[0], published['screening_audit'])
	_write_csv(reports / REPORT_OUTPUT_NAMES[0], published['job_metrics'])
	_write_csv(reports / REPORT_OUTPUT_NAMES[1], published['paired_deltas'])
	_write_json(reports / REPORT_OUTPUT_NAMES[2], published)
	decisions = inspection['decisions']
	(reports / REPORT_OUTPUT_NAMES[3]).write_text(
		'# Periodic-refresh original-split screening\n\n'
		f'Status: `{decisions["overall_status"]}`\n\n'
		f'Six-split follow-up ready: `{decisions["six_split_follow_up"]["ready"]}`\n\n'
		'This is paired original-split facies-label evidence only.\n',
		encoding='utf-8',
	)
	handoff = _handoff_payload(
		inspection,
		execution=execution,
		reports_dir=reports,
	)
	_write_json(
		reports / REPORT_OUTPUT_NAMES[4], _portable_payload(handoff, config=config)
	)
	published_files: tuple[Path, ...] = ()
	if publish:
		output = config.base.publish.reports_root / _PUBLISHED_ROOT
		output.mkdir(parents=True, exist_ok=True)
		published_files = tuple(output / name for name in PUBLISHED_OUTPUT_NAMES)
		for source, destination in zip(
			(reports / name for name in PUBLISHED_OUTPUT_NAMES),
			published_files,
			strict=True,
		):
			shutil.copyfile(source, destination)
	return {
		'summary_json': reports / REPORT_OUTPUT_NAMES[2],
		'decisions': decisions,
		'published_files': published_files,
	}


def _members(
	rows: Sequence[Mapping[str, object]], *, reference: object
) -> dict[tuple[str, int, str], Mapping[str, object]]:
	members = {}
	for row in rows:
		role = str(row['model_role'])
		key = (str(row['budget_id']), int(row['subsample_seed']), role)
		if key in members:
			raise ValueError(f'periodic-refresh duplicate member: {key!r}')
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
					row['evaluation_boundary_region_metrics'],
					'boundary region metrics',
				),
				label=role,
			),
		}
	for job in reference.jobs:
		key = (job.dataset.budget_id, job.dataset.subsample_seed, job.model_role)
		if key in members:
			raise ValueError(f'periodic-refresh duplicate historical member: {key!r}')
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
		for role in (
			'mae',
			'm1_current_k6',
			'mh_nocons',
			_FIXED_CENTER_ROLE,
			_CANDIDATE_ROLE,
		)
	}
	if set(members) != expected:
		raise ValueError(
			'periodic-refresh comparison member matrix is incomplete or duplicated'
		)


def _validate_pairing(
	config: object, members: Mapping[tuple[str, int, str], Mapping[str, object]]
) -> None:
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			rows = [
				members[(budget, seed, role)]['row']
				for role in (
					'mae',
					'm1_current_k6',
					'mh_nocons',
					_FIXED_CENTER_ROLE,
					_CANDIDATE_ROLE,
				)
			]
			for key in control.PAIR_IDENTITY_KEYS:
				if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
					raise ValueError(
						f'periodic-refresh paired identity mismatch: '
						f'{budget}/seed{seed}/{key}'
					)


def _load_fixed_center_rows(config: object) -> tuple[Mapping[str, object], ...]:
	path = Path(config.center_trace_masked_run_manifest)
	payload = runner._read_json(path)
	if payload.get('row_count') != 15 or payload.get('complete_count') != 15:
		raise ValueError(
			'fixed center-trace manifest is not a complete 15-row reference'
		)
	rows = payload.get('rows')
	if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
		raise TypeError('fixed center-trace reference rows are invalid')
	if len(rows) != 15 or any(
		row.get('status') != 'complete'
		or row.get('model_role') != _FIXED_CENTER_ROLE
		or row.get('model_tag') != _FIXED_CENTER_TAG
		for row in rows
	):
		raise ValueError('fixed center-trace reference identity/matrix mismatch')
	keys = {(str(row['budget_id']), int(row['subsample_seed'])) for row in rows}
	if keys != {
		(budget, seed) for budget in config.budgets for seed in config.subsample_seeds
	}:
		raise ValueError('fixed center-trace reference condition matrix mismatch')
	return tuple(rows)


def _periodic_hard_config(config: object) -> object:
	return f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(config.hard_multi_head_config)
	)


def _load_screening_audit(config: object) -> Mapping[str, object]:
	payload = periodic_config.validate_f3_center_trace_masked_periodic_refresh_screening_audit(
		config
	)
	if (
		payload.get('artifact_type')
		!= 'f3_center_trace_masked_periodic_refresh_original_screening_preflight'
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
	):
		raise ValueError('periodic-refresh screening audit type/schema/status mismatch')
	candidate = payload.get('candidate')
	if (
		not isinstance(candidate, Mapping)
		or candidate.get('model_id') != _CANDIDATE_ROLE
	):
		raise ValueError('periodic-refresh screening audit candidate mismatch')
	return payload


def _candidate_provenance(
	config: object, *, candidate_identity: Mapping[str, object]
) -> Mapping[str, object]:
	candidate = config.candidates[0]
	if candidate.model_id != _CANDIDATE_ROLE or candidate.model_tag != _CANDIDATE_TAG:
		raise ValueError('periodic-refresh candidate identity mismatch')
	return {
		'model_id': candidate.model_id,
		'model_tag': candidate.model_tag,
		'pretraining_handoff': candidate_identity['pretraining_handoff'],
		'selected_checkpoint': candidate_identity['checkpoint'],
		'embeddings': candidate_identity['embeddings'],
		'valid_tokens': candidate_identity['valid_tokens'],
		'embedding_metadata': candidate_identity['metadata'],
	}


def _handoff_payload(
	inspection: Mapping[str, object],
	*,
	execution: Mapping[str, object],
	reports_dir: Path,
) -> dict[str, object]:
	identities = inspection['source_identities']
	decisions = inspection['decisions']
	if not isinstance(identities, Mapping) or not isinstance(decisions, Mapping):
		raise TypeError('periodic-refresh summary evidence is incomplete')
	job_metrics = inspection.get('job_metrics')
	if not isinstance(job_metrics, Sequence) or isinstance(job_metrics, str):
		raise TypeError('periodic-refresh job metrics evidence is invalid')
	if len(job_metrics) != 75:
		raise ValueError('periodic-refresh paired matrix must contain exactly 75 rows')
	if not all(isinstance(row, Mapping) for row in job_metrics):
		raise TypeError('periodic-refresh paired matrix rows must be mappings')
	job_metrics_path = reports_dir / REPORT_OUTPUT_NAMES[0]
	paired_deltas_path = reports_dir / REPORT_OUTPUT_NAMES[1]
	summary_json_path = reports_dir / REPORT_OUTPUT_NAMES[2]
	summary_markdown_path = reports_dir / REPORT_OUTPUT_NAMES[3]
	for path in (
		job_metrics_path,
		paired_deltas_path,
		summary_json_path,
		summary_markdown_path,
	):
		if not path.is_file():
			raise FileNotFoundError(path)
	job_metric_columns = list(job_metrics[0]) if job_metrics else []
	return {
		'artifact_type': 'f3_center_trace_masked_periodic_refresh_original_screening_handoff',
		'schema_version': 1,
		'status': 'PASS',
		'formal_status': decisions['overall_status'],
		'pretraining_lineage': {
			'periodic_refresh_handoff': identities['periodic_refresh_handoff'],
			'candidate_provenance': identities['candidate_provenance'],
		},
		'screening_audit': identities['screening_audit'],
		'candidate_run_manifest': identities['candidate_run_manifest'],
		'candidate_job_live_validation': identities['candidate_job_live_validation'],
		'fixed_center_trace_reference': identities['fixed_center_trace_run_manifest'],
		'hard_decoder_config': identities['hard_decoder_config'],
		'reference_run_manifests': identities['reference_run_manifests'],
		'paired_matrix_identity': identities['paired_matrix_identity'],
		'reports': {
			'paired_matrix': {
				**_identity(job_metrics_path),
				'row_count': len(job_metrics),
				'columns': job_metric_columns,
			},
			'gate_reports': {
				'paired_deltas': _identity(paired_deltas_path),
				'summary_json': _identity(summary_json_path),
				'summary_markdown': _identity(summary_markdown_path),
			},
		},
		'gate_definition': decisions['gate'],
		'gate_result': decisions,
		'execution': dict(execution),
		'six_split_follow_up': decisions['six_split_follow_up'],
		'six_split_jobs_executed': 0,
		'six_split_scientific_jobs_executed': 0,
		'scientific_jobs_executed': decisions['scientific_jobs_executed'],
	}


def _execution_git_state(config: object) -> Mapping[str, object]:
	workspace = Path(config.base.reports_root).parent
	git = shutil.which('git')
	if git is None:
		raise RuntimeError('git executable is unavailable')
	try:
		sha = subprocess.run(
			(git, 'rev-parse', 'HEAD'),
			cwd=workspace,
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
		status = subprocess.run(
			(git, 'status', '--porcelain'),
			cwd=workspace,
			check=True,
			capture_output=True,
			text=True,
		).stdout
	except (OSError, subprocess.CalledProcessError) as error:
		raise RuntimeError(
			'unable to record periodic-refresh execution git state'
		) from error
	if len(sha) != 40:
		raise ValueError('execution git SHA is invalid')
	return {'git_sha': sha, 'dirty': bool(status.strip())}


def _summary_row(
	index: Mapping[tuple[str, str], Mapping[str, object]], budget: str, metric: str
) -> Mapping[str, object]:
	try:
		return index[(budget, metric)]
	except KeyError as error:
		raise ValueError(
			f'periodic-refresh primary gate is missing {budget}/{metric}'
		) from error


def _identity(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	if not rows:
		raise ValueError(f'cannot write empty table: {path.name}')
	fields = list(rows[0])
	if any(list(row) != fields for row in rows):
		raise ValueError(f'inconsistent CSV columns: {path.name}')
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
		writer.writeheader()
		writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _portable_payload(payload: object, *, config: object) -> object:
	return _portable_value(
		payload,
		artifact_root=Path(config.base.artifact_root),
		workspace_root=Path(config.base.reports_root).parent,
	)


def _portable_value(
	value: object, *, artifact_root: Path, workspace_root: Path
) -> object:
	if isinstance(value, Mapping):
		return {
			key: _portable_value(
				item, artifact_root=artifact_root, workspace_root=workspace_root
			)
			for key, item in value.items()
		}
	if isinstance(value, (tuple, list)):
		return [
			_portable_value(
				item, artifact_root=artifact_root, workspace_root=workspace_root
			)
			for item in value
		]
	if not isinstance(value, str):
		return value
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
			return (
				replacement
				if relative.as_posix() == '.'
				else f'{replacement}/{relative.as_posix()}'
			)
		return '.' if relative.as_posix() == '.' else relative.as_posix()
	return value


inspect_results = (
	inspect_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_results
)
summarize_results = (
	summarize_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh
)


__all__ = [
	'AUDIT_OUTPUT_NAMES',
	'COMPARISONS',
	'OUTPUT_NAMES',
	'PUBLISHED_OUTPUT_NAMES',
	'REPORT_OUTPUT_NAMES',
	'decide_center_trace_masked_periodic_refresh_original_gate',
	'inspect_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_results',
	'summarize_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh',
]
