"""Candidate-only runner and validation gates for the current-code K=6 control.

The original M3-V-LB suite is deliberately immutable: it owns the MAE, M1,
and M2-A 45-job matrix.  This module validates that source suite read-only and
adds only the 15 paired current-code K=6 decoder jobs.  It delegates decoder
training, best-checkpoint inference, evaluation, report generation, resume,
and quarantine mechanics to :mod:`voxel_label_budget_runner`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	METRIC_SPECS,
	F3VoxelLabelBudgetReferenceInspection,
	inspect_f3_lithology_voxel_label_budget_mae_reference_run,
	inspect_f3_lithology_voxel_label_budget_reference_run,
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	VoxelLabelBudgetJobPlan,
	classify_voxel_label_budget_job,
	completed_voxel_label_budget_job_row,
	quarantine_voxel_label_budget_output,
	run_voxel_label_budget_job,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)

FORBIDDEN_SUFFIXES = frozenset(
	{'.pt', '.pth', '.npy', '.npz', '.joblib', '.pkl', '.sgy', '.segy'}
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_control import (
		F3VoxelLabelBudgetControlConfig,
	)
	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_suite import (
		F3VoxelLabelBudgetSuiteConfig,
	)

CONTROL_RUN_MANIFEST_NAME = 'control_job_manifest.json'
CONTROL_RUN_MANIFEST_TYPE = 'f3_lithology_voxel_label_budget_current_k6_control'
CONTROL_RUN_SCHEMA_VERSION = 1
CONTROL_STATUS_CSV = 'control_job_status.csv'
CONTROL_JOB_METRICS_CSV = 'control_job_metrics.csv'
CONTROL_PAIRED_METRICS_CSV = 'control_paired_metrics.csv'
CONTROL_PAIRED_DELTAS_CSV = 'control_paired_deltas.csv'
CONTROL_SUMMARY_BY_BUDGET_CSV = 'control_summary_by_budget.csv'
CONTROL_MONITORED_CLASS_SUMMARY_CSV = 'control_monitored_class_summary.csv'
CONTROL_SUMMARY_JSON = 'current_k6_control_summary.json'
CONTROL_SUMMARY_MARKDOWN = 'current_k6_control_summary.md'
CONTROL_HANDOFF_MARKDOWN = 'current_k6_control_handoff.md'
CONTROL_PUBLISH_MANIFEST = 'publish_manifest.json'
BLOCKED_CONTROL_CONTRACT = 'BLOCKED_CONTROL_CONTRACT'
CURRENT_MODEL_ROLE = 'm1_current_k6'
REFERENCE_MODEL_ROLES = ('mae', 'm1')
REQUIRED_VALIDATION_VOXELS = 470136
PAIR_IDENTITY_KEYS = (
	'voxel_supervision_grid_sha256',
	'selected_token_identity_sha256',
	'unique_token_xyz_sha256',
	'train_voxel_count',
	'validation_voxel_count',
	'class_order',
	'validation_mask_sha256',
	'canonical_valid_token_sha256',
	'initial_model_state_sha256',
	'class_weights',
	'decoder_architecture',
	'decoder_seed',
	'sampling_mode',
	'steps_per_epoch',
	'sampling_sequence_sha256',
	'train_tile_manifest_sha256',
	'validation_tile_manifest_sha256',
	'train_tile_identity_sha256',
	'validation_tile_identity_sha256',
	'metric_schema_sha256',
	'uncovered_validation_voxel_count',
	'prediction_duplicate_write_count',
	'prediction_missing_write_count',
	'prediction_exact_once',
)


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlInspection:
	"""Read-only preflight for the candidate matrix and historical references."""

	jobs: tuple[VoxelLabelBudgetJob, ...]
	plans: tuple[VoxelLabelBudgetJobPlan, ...]
	reference: F3VoxelLabelBudgetReferenceInspection
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]]
	candidate_embedding_identity: Mapping[str, Mapping[str, object]]
	estimated_new_bytes: int
	disk_free_bytes: int


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlRunResult:
	"""Persisted control manifest and all retained candidate rows."""

	manifest_json: Path
	rows: tuple[Mapping[str, object], ...]
	quarantines: tuple[Path, ...]


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlResultsInspection:
	"""Validated current/reference metrics and paired descriptive summaries."""

	job_metrics: tuple[Mapping[str, object], ...]
	paired_metrics: tuple[Mapping[str, object], ...]
	paired_deltas: tuple[Mapping[str, object], ...]
	summary_by_budget: tuple[Mapping[str, object], ...]
	monitored_class_summary: tuple[Mapping[str, object], ...]
	readiness: Mapping[str, object]
	historical_m1_mae_parity: Mapping[str, object]
	source_identities: Mapping[str, object]


@dataclass(frozen=True)
class F3VoxelLabelBudgetControlResultsResult:
	"""Summary files and optional lightweight publication manifest."""

	summary_json: Path
	summary_markdown: Path
	handoff_markdown: Path
	table_paths: tuple[Path, ...]
	readiness: Mapping[str, object]
	published_files: tuple[Path, ...] = ()


def inspect_f3_lithology_voxel_label_budget_control(
	config: F3VoxelLabelBudgetControlConfig,
	*,
	budget: str | None = None,
	subsample_seed: int | None = None,
) -> F3VoxelLabelBudgetControlInspection:
	"""Validate immutable references, current embedding provenance, and jobs."""
	if config.validate_pairing_reference:
		reference = inspect_f3_lithology_voxel_label_budget_reference_run(
			config.dataset_manifest, config.historical_run_manifest
		)
	else:
		reference = inspect_f3_lithology_voxel_label_budget_mae_reference_run(
			config.dataset_manifest,
			config.historical_run_manifest,
			include_historical_m1=False,
		)
	dataset_rows = _dataset_rows(config, reference)
	_validate_reference_contract(config, reference, dataset_rows)
	candidate_embedding = _candidate_embedding_identity(config, reference)
	jobs = _jobs(config, dataset_rows)
	jobs = tuple(
		job
		for job in jobs
		if (budget is None or job.budget_id == budget)
		and (subsample_seed is None or job.subsample_seed == subsample_seed)
	)
	if not jobs:
		raise ValueError('job filters selected no current-K6 control jobs')
	stage_config = _stage_config(config)
	plans = tuple(
		classify_voxel_label_budget_job(stage_config, job) for job in jobs
	)
	reference_bytes = _estimated_candidate_job_bytes(reference)
	plans = tuple(
		VoxelLabelBudgetJobPlan(
			job=plan.job,
			state=plan.state,
			reason=plan.reason,
			estimated_bytes=(
				0 if plan.state == 'REUSE_COMPLETED' else reference_bytes
			),
		)
		for plan in plans
	)
	disk = shutil.disk_usage(config.output_root.parent)
	return F3VoxelLabelBudgetControlInspection(
		jobs=jobs,
		plans=plans,
		reference=reference,
		dataset_rows=dataset_rows,
		candidate_embedding_identity=candidate_embedding,
		estimated_new_bytes=sum(plan.estimated_bytes for plan in plans),
		disk_free_bytes=disk.free,
	)


def run_f3_lithology_voxel_label_budget_control(  # noqa: C901, PLR0912, PLR0913, PLR0915
	config: F3VoxelLabelBudgetControlConfig,
	*,
	only_missing: bool = False,
	resume: bool = False,
	device: str = 'auto',
	budget: str | None = None,
	subsample_seed: int | None = None,
) -> F3VoxelLabelBudgetControlRunResult:
	"""Run exactly the candidate jobs while preserving immutable references.

	``--only-missing`` revalidates complete jobs, resumes valid ``latest.pt``
	checkpoints, and quarantines invalid partial output before a fresh run.
	``resume`` is deliberately narrower: every selected job must have a valid,
	incomplete ``latest.pt`` and only that checkpoint may be resumed.
	"""
	if only_missing and resume:
		raise ValueError('--only-missing and --resume are mutually exclusive')
	try:
		inspection = inspect_f3_lithology_voxel_label_budget_control(
			config, budget=budget, subsample_seed=subsample_seed
		)
	except Exception as error:
		_write_blocked_control_contract(config, stage='runner_preflight', error=error)
		raise
	if inspection.estimated_new_bytes > inspection.disk_free_bytes:
		error = RuntimeError(
			'insufficient disk for planned jobs; '
			f'required={inspection.estimated_new_bytes}, '
			f'free={inspection.disk_free_bytes}'
		)
		_write_blocked_control_contract(config, stage='runner_disk_gate', error=error)
		raise error
	if resume:
		invalid = [plan for plan in inspection.plans if plan.state != 'RESUME_LATEST']
		if invalid:
			first = invalid[0]
			error = ValueError(
				'--resume requires an incomplete valid latest.pt for every selected '
				f'job; {first.job.output_root}: {first.state}'
			)
			_write_blocked_control_contract(
				config, stage='runner_resume_preflight', error=error
			)
			raise error
	elif not only_missing and any(plan.state != 'NEW' for plan in inspection.plans):
		first = next(plan for plan in inspection.plans if plan.state != 'NEW')
		raise FileExistsError(
			'non-new job requires --only-missing or --resume; '
			f'{first.job.output_root}: {first.state}: {first.reason}'
		)

	try:
		manifest_path = control_run_manifest_path(config)
		prior_rows, quarantines = _prior_control_state(
			manifest_path,
			config=config,
			candidate_embedding_identity=inspection.candidate_embedding_identity,
		)
		rows_by_key = {
			_control_row_key(row): row
			for row in prior_rows
			if _control_row_key(row) not in {_job_key(job) for job in inspection.jobs}
		}
		stage_config = _stage_config(config)
	except Exception as error:
		_write_blocked_control_contract(config, stage='runner_setup', error=error)
		raise
	for plan in inspection.plans:
		job = plan.job
		state = plan.state
		quarantine_path: Path | None = None
		quarantine_reason: str | None = None
		if state == 'INVALID_OR_PARTIAL':
			if not only_missing:
				raise FileExistsError(
					f'invalid/partial job requires --only-missing: {job.output_root}'
				)
			quarantine_reason = plan.reason or 'invalid_or_partial'
			quarantine_path = quarantine_voxel_label_budget_output(
				job.output_root, reason=quarantine_reason
			)
			quarantines.append(quarantine_path)
			state = 'NEW'
		try:
			if state == 'REUSE_COMPLETED':
				row = _completed_control_row(
					config,
					stage_config,
					job,
					candidate_embedding_identity=(
						inspection.candidate_embedding_identity
					),
					action='REUSED',
					quarantine_path=None,
					error=None,
				)
			else:
				checkpoint = (
					job.decoder_dir / 'latest.pt'
					if state == 'RESUME_LATEST'
					else None
				)
				action = 'RESUMED' if checkpoint is not None else 'NEW'
				print(
					f'control_job.start budget={job.budget_id} '
					f'seed={job.subsample_seed} action={action}',
					flush=True,
				)
				run_voxel_label_budget_job(
					stage_config, job, device=device, resume=checkpoint
				)
				row = _completed_control_row(
					config,
					stage_config,
					job,
					candidate_embedding_identity=(
						inspection.candidate_embedding_identity
					),
					action=action,
					quarantine_path=quarantine_path,
					error=quarantine_reason,
				)
				print(
					f'control_job.complete budget={job.budget_id} '
					f'seed={job.subsample_seed}',
					flush=True,
				)
			_validate_paired_identity(
				row,
				reference=inspection.reference,
				dataset_row=inspection.dataset_rows[
					(job.budget_id, job.subsample_seed)
				],
			)
		except Exception as error:
			rows_by_key[_job_key(job)] = _failed_row(
				job,
				action=(
					'REUSED'
					if state == 'REUSE_COMPLETED'
					else 'RESUMED' if state == 'RESUME_LATEST' else 'NEW'
				),
				error=error,
				quarantine_path=quarantine_path,
			)
			_write_control_manifest(
				manifest_path,
				config=config,
				rows=tuple(rows_by_key.values()),
				quarantines=quarantines,
				candidate_embedding_identity=inspection.candidate_embedding_identity,
			)
			_write_blocked_control_contract(config, stage='runner_job', error=error)
			raise
		rows_by_key[_job_key(job)] = row
		_write_control_manifest(
			manifest_path,
			config=config,
			rows=tuple(rows_by_key.values()),
			quarantines=quarantines,
			candidate_embedding_identity=inspection.candidate_embedding_identity,
		)
	ordered = tuple(sorted(rows_by_key.values(), key=_row_sort_key))
	_write_control_manifest(
		manifest_path,
		config=config,
		rows=ordered,
		quarantines=quarantines,
		candidate_embedding_identity=inspection.candidate_embedding_identity,
	)
	_write_status_csv(config.reports_dir / CONTROL_STATUS_CSV, ordered)
	return F3VoxelLabelBudgetControlRunResult(
		manifest_json=manifest_path,
		rows=ordered,
		quarantines=tuple(quarantines),
	)


def control_run_manifest_path(config: F3VoxelLabelBudgetControlConfig) -> Path:
	"""Return the report-owned, lightweight candidate run manifest path."""
	return config.reports_dir / CONTROL_RUN_MANIFEST_NAME


def load_f3_lithology_voxel_label_budget_control_rows(  # noqa: C901
	config: F3VoxelLabelBudgetControlConfig,
	*,
	require_complete: bool = True,
	run_manifest_path: Path | None = None,
) -> tuple[Mapping[str, object], ...]:
	"""Revalidate all candidate jobs represented by the control manifest."""
	inspection = inspect_f3_lithology_voxel_label_budget_control(config)
	path = (
		control_run_manifest_path(config)
		if run_manifest_path is None
		else run_manifest_path
	)
	payload = _read_json(path)
	_validate_control_manifest_header(
		payload,
		config=config,
		candidate_embedding_identity=inspection.candidate_embedding_identity,
	)
	rows = _mapping_rows(payload.get('rows'), 'control run manifest rows')
	expected_keys = {_job_key(job) for job in inspection.jobs}
	actual_keys = {_control_row_key(row) for row in rows}
	if actual_keys != expected_keys:
		raise ValueError(
			'control run manifest job matrix mismatch; '
			f'missing={sorted(expected_keys - actual_keys)!r}, '
			f'extra={sorted(actual_keys - expected_keys)!r}'
		)
	if len(actual_keys) != len(rows):
		raise ValueError('control run manifest contains duplicate job rows')
	jobs_by_key = {_job_key(job): job for job in inspection.jobs}
	stage_config = _stage_config(config)
	validated: list[Mapping[str, object]] = []
	for row in rows:
		key = _control_row_key(row)
		job = jobs_by_key[key]
		status = row.get('status')
		if status != 'complete':
			if require_complete:
				raise ValueError(f'control job is not complete: {key!r}')
			validated.append(row)
			continue
		action = row.get('action')
		if action not in {'NEW', 'RESUMED', 'REUSED'}:
			raise ValueError(f'control completed action is invalid: {key!r}')
		quarantine_value = row.get('quarantine_path')
		if quarantine_value is not None and not isinstance(quarantine_value, str):
			raise TypeError(
				'control completed quarantine_path must be a string or null'
			)
		error_value = row.get('error')
		if error_value is not None and not isinstance(error_value, str):
			raise TypeError('control completed error must be a string or null')
		actual = _completed_control_row(
			config,
			stage_config,
			job,
			candidate_embedding_identity=inspection.candidate_embedding_identity,
			action=cast('str', action),
			quarantine_path=(
				None if quarantine_value is None else Path(quarantine_value)
			),
			error=cast('str | None', error_value),
		)
		if dict(row) != actual:
			different = sorted(
				key for key in set(row) | set(actual) if row.get(key) != actual.get(key)
			)
			raise ValueError(
				f'control completed manifest row differs from live artifacts: '
				f'{key!r}: {different!r}'
			)
		_validate_paired_identity(
			actual,
			reference=inspection.reference,
			dataset_row=inspection.dataset_rows[(job.budget_id, job.subsample_seed)],
			reference_roles=(
				REFERENCE_MODEL_ROLES
				if config.validate_pairing_reference
				else ('mae',)
			),
		)
		validated.append(actual)
	if require_complete and len(validated) != len(expected_keys):
		raise ValueError('control run manifest completion count mismatch')
	return tuple(sorted(validated, key=_row_sort_key))


def inspect_f3_lithology_voxel_label_budget_control_results(
	config: F3VoxelLabelBudgetControlConfig,
) -> F3VoxelLabelBudgetControlResultsInspection:
	"""Recompute all current/M1/MAE paired descriptive comparisons."""
	candidate_rows = load_f3_lithology_voxel_label_budget_control_rows(config)
	reference = inspect_f3_lithology_voxel_label_budget_reference_run(
		config.dataset_manifest, config.historical_run_manifest
	)
	members = _comparison_members(config, candidate_rows, reference)
	job_metrics = tuple(_member_metric_row(member) for member in members.values())
	paired_metrics = tuple(_paired_metric_rows(config, members))
	paired_deltas = tuple(_paired_delta_rows(config, members))
	summary_by_budget = tuple(_summary_rows(config, paired_deltas))
	monitored = tuple(_monitored_class_summary(config, summary_by_budget))
	parity = _validate_historical_m1_mae_parity(
		config, paired_deltas=paired_deltas, summary_by_budget=summary_by_budget
	)
	readiness = _readiness_decision(config, summary_by_budget)
	return F3VoxelLabelBudgetControlResultsInspection(
		job_metrics=job_metrics,
		paired_metrics=paired_metrics,
		paired_deltas=paired_deltas,
		summary_by_budget=summary_by_budget,
		monitored_class_summary=monitored,
		readiness=readiness,
		historical_m1_mae_parity=parity,
		source_identities={
			'dataset_manifest': _identity(config.dataset_manifest),
			'historical_run_manifest': _identity(config.historical_run_manifest),
			'control_job_manifest': _identity(control_run_manifest_path(config)),
			'candidate_embeddings': _candidate_embedding_identity(
				config, reference
			),
		},
	)


def summarize_f3_lithology_voxel_label_budget_control(
	config: F3VoxelLabelBudgetControlConfig,
	*,
	require_evidence: bool = True,
) -> F3VoxelLabelBudgetControlResultsResult:
	"""Write the current-K6 control summary and optionally publish it."""
	try:
		inspection = inspect_f3_lithology_voxel_label_budget_control_results(config)
	except Exception as error:
		_write_blocked_control_contract(config, stage='summary_inspect', error=error)
		raise
	try:
		if require_evidence:
			_materialize_required_evidence(config)
		_validate_summary_output_availability(config)
	except Exception as error:
		_write_blocked_control_contract(config, stage='summary_evidence', error=error)
		raise
	try:
		candidate_rows = load_f3_lithology_voxel_label_budget_control_rows(config)
		_write_status_csv(config.reports_dir / CONTROL_STATUS_CSV, candidate_rows)
		table_paths = _write_summary_tables(config, inspection)
		payload = _control_summary_payload(config, inspection)
		summary_json = config.reports_dir / CONTROL_SUMMARY_JSON
		summary_markdown = config.reports_dir / CONTROL_SUMMARY_MARKDOWN
		handoff_markdown = config.reports_dir / CONTROL_HANDOFF_MARKDOWN
		_write_json(summary_json, payload)
		summary_markdown.write_text(
			_render_control_summary_markdown(payload), encoding='utf-8'
		)
		handoff_markdown.write_text(
			_render_control_handoff_markdown(payload), encoding='utf-8'
		)
		result = F3VoxelLabelBudgetControlResultsResult(
			summary_json=summary_json,
			summary_markdown=summary_markdown,
			handoff_markdown=handoff_markdown,
			table_paths=table_paths,
			readiness=inspection.readiness,
		)
	except Exception as error:
		_write_blocked_control_contract(config, stage='summary_write', error=error)
		raise
	try:
		published_files = _publish_control_results(config)
	except Exception as error:
		_write_blocked_control_contract(config, stage='summary_publish', error=error)
		raise
	return F3VoxelLabelBudgetControlResultsResult(
		summary_json=result.summary_json,
		summary_markdown=result.summary_markdown,
		handoff_markdown=result.handoff_markdown,
		table_paths=result.table_paths,
		readiness=result.readiness,
		published_files=published_files,
	)


def validate_f3_lithology_voxel_label_budget_control_summary_preflight(
	config: F3VoxelLabelBudgetControlConfig,
) -> None:
	"""Read-only validate evidence and any pre-existing publish tree.

	This is deliberately separate from summary materialization so the CLI
	``--dry-run`` path can validate every upstream evidence input and an existing
	publish inventory without copying the preflight manifest, creating report
	directories, or replacing any outputs.
	"""
	_validate_required_evidence(config)
	_validate_summary_output_availability(config, create_reports_dir=False)


def _comparison_members(
	config: F3VoxelLabelBudgetControlConfig,
	candidate_rows: Sequence[Mapping[str, object]],
	reference: F3VoxelLabelBudgetReferenceInspection,
) -> Mapping[tuple[str, int, str], Mapping[str, object]]:
	members: dict[tuple[str, int, str], Mapping[str, object]] = {}
	for row in candidate_rows:
		key = _control_row_key(row)
		metrics = load_f3_lithology_voxel_label_budget_evaluation_metrics(
			metrics_path=_identity_path(
				row.get('evaluation_metrics'), 'candidate evaluation metrics'
			),
			boundary_metrics_path=_identity_path(
				row.get('evaluation_boundary_metrics'),
				'candidate boundary metrics',
			),
			boundary_region_metrics_path=_identity_path(
				row.get('evaluation_boundary_region_metrics'),
				'candidate boundary region metrics',
			),
			label=f'current K6 {key[0]}/seed{key[1]}',
		)
		members[key] = {
			'role': config.candidate.model_id,
			'model_tag': config.candidate.model_tag,
			'row': row,
			'metrics': metrics,
		}
	for job in reference.jobs:
		if job.model_role not in REFERENCE_MODEL_ROLES:
			continue
		key = (job.dataset.budget_id, job.dataset.subsample_seed, job.model_role)
		members[key] = {
			'role': job.model_role,
			'model_tag': job.model_tag,
			'row': _reference_member_row(job),
			'metrics': dict(job.evaluation.metrics),
		}
	expected = {
		(budget, seed, role)
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in (
			config.candidate.model_id,
			config.references.historical_m1_model_id,
			config.references.mae_model_id,
		)
	}
	if set(members) != expected:
		raise ValueError('current/control/reference comparison matrix mismatch')
	return members


def _reference_member_row(job: object) -> Mapping[str, object]:
	loaded = cast('object', job)
	dataset = loaded.dataset
	evaluation = loaded.evaluation
	coverage = _prediction_coverage(
		Path(str(loaded.row['prediction_metadata']['path']))
	)
	return {
		'budget_id': dataset.budget_id,
		'per_class_cap': dataset.per_class_cap,
		'subsample_seed': dataset.subsample_seed,
		'decoder_seed': loaded.decoder_seed,
		'model_role': loaded.model_role,
		'model_tag': loaded.model_tag,
		'voxel_dataset_root': str(dataset.root),
		'voxel_supervision_grid_sha256': dataset.grid.sha256,
		'selected_token_identity_sha256': dataset.selected_token_identity_sha256,
		'unique_token_xyz_sha256': dataset.unique_token_xyz_sha256,
		'train_voxel_count': dataset.train_voxel_count,
		'validation_voxel_count': dataset.validation_voxel_count,
		'class_order': list(dataset.class_order),
		'validation_mask_sha256': dataset.validation_mask_sha256,
		'canonical_valid_token_sha256': loaded.canonical_valid_tokens_sha256,
		'initial_model_state_sha256': loaded.initial_model_state_sha256,
		'class_weights': list(loaded.class_weights),
		'decoder_architecture': dict(evaluation.decoder_architecture),
		'sampling_mode': loaded.sampling_mode,
		'steps_per_epoch': loaded.steps_per_epoch,
		'sampling_sequence_sha256': loaded.sampling_sequence_sha256,
		'train_tile_manifest_sha256': loaded.train_tile_manifest_sha256,
		'validation_tile_manifest_sha256': loaded.validation_tile_manifest_sha256,
		'train_tile_identity_sha256': loaded.train_tile_identity_sha256,
		'validation_tile_identity_sha256': loaded.validation_tile_identity_sha256,
		'metric_schema_sha256': evaluation.metric_schema_sha256,
		'uncovered_validation_voxel_count': 0,
		'prediction_duplicate_write_count': coverage['duplicate_write_count'],
		'prediction_missing_write_count': coverage['missing_write_count'],
		'prediction_exact_once': coverage['exact_once'],
	}


def _member_metric_row(member: Mapping[str, object]) -> dict[str, object]:
	row = _mapping(member.get('row'), 'comparison member row')
	metrics = cast('Mapping[str, float]', member.get('metrics'))
	return {
		'budget_id': row['budget_id'],
		'per_class_cap': row['per_class_cap'],
		'subsample_seed': row['subsample_seed'],
		'decoder_seed': row['decoder_seed'],
		'model_role': member['role'],
		'model': _model_label(str(member['role'])),
		'model_tag': member['model_tag'],
		'voxel_dataset_root': row['voxel_dataset_root'],
		'voxel_supervision_grid_sha256': row['voxel_supervision_grid_sha256'],
		'selected_token_identity_sha256': row['selected_token_identity_sha256'],
		'unique_token_xyz_sha256': row['unique_token_xyz_sha256'],
		'train_voxel_count': row['train_voxel_count'],
		'validation_voxel_count': row['validation_voxel_count'],
		'validation_mask_sha256': row['validation_mask_sha256'],
		'canonical_valid_token_sha256': row['canonical_valid_token_sha256'],
		'initial_model_state_sha256': row['initial_model_state_sha256'],
		'sampling_mode': row['sampling_mode'],
		'steps_per_epoch': row['steps_per_epoch'],
		'sampling_sequence_sha256': row['sampling_sequence_sha256'],
		'train_tile_manifest_sha256': row['train_tile_manifest_sha256'],
		'validation_tile_manifest_sha256': row['validation_tile_manifest_sha256'],
		'train_tile_identity_sha256': row['train_tile_identity_sha256'],
		'validation_tile_identity_sha256': row['validation_tile_identity_sha256'],
		'class_weights': json.dumps(row['class_weights'], separators=(',', ':')),
		'metric_schema_sha256': row['metric_schema_sha256'],
		'prediction_duplicate_write_count': row['prediction_duplicate_write_count'],
		'prediction_missing_write_count': row['prediction_missing_write_count'],
		'prediction_exact_once': row['prediction_exact_once'],
		**metrics,
	}


def _paired_metric_rows(
	config: F3VoxelLabelBudgetControlConfig,
	members: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	roles = (
		config.candidate.model_id,
		config.references.historical_m1_model_id,
		config.references.mae_model_id,
	)
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			condition = [members[(budget, seed, role)] for role in roles]
			first = _mapping(condition[0].get('row'), 'paired member row')
			row: dict[str, object] = {
				'budget_id': budget,
				'per_class_cap': first['per_class_cap'],
				'subsample_seed': seed,
				'decoder_seed': first['decoder_seed'],
				'voxel_supervision_grid_sha256': first[
					'voxel_supervision_grid_sha256'
				],
				'validation_voxel_count': first['validation_voxel_count'],
			}
			for member in condition:
				role = str(member['role'])
				for metric, value in cast(
					'Mapping[str, float]', member['metrics']
				).items():
					row[f'{role}_{metric}'] = value
			rows.append(row)
	return rows


def _paired_delta_rows(
	config: F3VoxelLabelBudgetControlConfig,
	members: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			for candidate_role, baseline_role in config.comparisons:
				candidate = members[(budget, seed, candidate_role)]
				baseline = members[(budget, seed, baseline_role)]
				candidate_row = _mapping(candidate.get('row'), 'candidate member row')
				row: dict[str, object] = {
					'budget_id': budget,
					'per_class_cap': candidate_row['per_class_cap'],
					'subsample_seed': seed,
					'decoder_seed': candidate_row['decoder_seed'],
					'comparison_id': _comparison_id(candidate_role, baseline_role),
					'comparison': (
						f'{_model_label(candidate_role)} - '
						f'{_model_label(baseline_role)}'
					),
					'baseline_model_role': baseline_role,
					'baseline_model_tag': baseline['model_tag'],
					'candidate_model_role': candidate_role,
					'candidate_model_tag': candidate['model_tag'],
				}
				candidate_metrics = cast('Mapping[str, float]', candidate['metrics'])
				baseline_metrics = cast('Mapping[str, float]', baseline['metrics'])
				for metric in METRIC_SPECS:
					row[metric.name] = (
						candidate_metrics[metric.name] - baseline_metrics[metric.name]
					)
				rows.append(row)
	return rows


def _summary_rows(
	config: F3VoxelLabelBudgetControlConfig,
	paired_deltas: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for budget in config.budgets:
		for candidate_role, baseline_role in config.comparisons:
			comparison_id = _comparison_id(candidate_role, baseline_role)
			selected = [
				row
				for row in paired_deltas
				if row['budget_id'] == budget
				and row['comparison_id'] == comparison_id
			]
			if len(selected) != len(config.subsample_seeds):
				raise ValueError('paired summary lost a required seed')
			for metric in METRIC_SPECS:
				values = [float(row[metric.name]) for row in selected]
				wins = [
					value > 0.0 if metric.higher_is_better else value < 0.0
					for value in values
				]
				losses = [
					value < 0.0 if metric.higher_is_better else value > 0.0
					for value in values
				]
				worst_index = (
					min(range(len(values)), key=values.__getitem__)
					if metric.higher_is_better
					else max(range(len(values)), key=values.__getitem__)
				)
				rows.append(
					{
						'budget_id': budget,
						'per_class_cap': int(budget.removeprefix('cap')),
						'comparison_id': comparison_id,
						'comparison': (
							f'{_model_label(candidate_role)} - '
							f'{_model_label(baseline_role)}'
						),
						'baseline_model_role': baseline_role,
						'candidate_model_role': candidate_role,
						'metric': metric.name,
						'higher_is_better': metric.higher_is_better,
						'paired_seed_count': len(values),
						'mean_delta': statistics.fmean(values),
						'median_delta': statistics.median(values),
						'sample_standard_deviation': statistics.stdev(values),
						'min_delta': min(values),
						'max_delta': max(values),
						'worst_seed': int(selected[worst_index]['subsample_seed']),
						'worst_seed_delta': values[worst_index],
						'wins': sum(wins),
						'losses': sum(losses),
						'ties': sum(value == 0.0 for value in values),
						'positive_win_count': sum(wins),
						'negative_count': sum(losses),
						'zero_count': sum(value == 0.0 for value in values),
					}
				)
	return rows


def _monitored_class_summary(
	config: F3VoxelLabelBudgetControlConfig,
	summary_by_budget: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	index = {
		(str(row['budget_id']), str(row['comparison_id']), str(row['metric'])): row
		for row in summary_by_budget
	}
	rows: list[dict[str, object]] = []
	for budget in config.budgets:
		for candidate_role, baseline_role in config.comparisons:
			comparison_id = _comparison_id(candidate_role, baseline_role)
			for class_id in config.decision.monitored_class_ids:
				row: dict[str, object] = {
					'budget_id': budget,
					'per_class_cap': int(budget.removeprefix('cap')),
					'comparison_id': comparison_id,
					'comparison': (
						f'{_model_label(candidate_role)} - '
						f'{_model_label(baseline_role)}'
					),
					'class_id': class_id,
				}
				for metric in (
					'f1',
					'iou',
					'boundary_recall_t2',
					'boundary_recall_t4',
				):
					source = index[
						(budget, comparison_id, f'class_{class_id}_{metric}')
					]
					for statistic in (
						'mean_delta',
						'median_delta',
						'sample_standard_deviation',
						'wins',
						'losses',
						'ties',
					):
						row[f'{metric}_{statistic}'] = source[statistic]
				rows.append(row)
	return rows


def _validate_historical_m1_mae_parity(
	config: F3VoxelLabelBudgetControlConfig,
	*,
	paired_deltas: Sequence[Mapping[str, object]],
	summary_by_budget: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
	"""Require reaggregation of M1-MAE to match the published M3-V-LB tables."""
	tables = config.historical_run_manifest.parent / 'reports' / 'tables'
	published_deltas_path = tables / 'paired_deltas.csv'
	published_summary_path = tables / 'summary_by_budget.csv'
	published_deltas = _read_csv(published_deltas_path)
	published_summary = _read_csv(published_summary_path)
	comparison_id = _comparison_id(
		config.references.historical_m1_model_id, config.references.mae_model_id
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
		if row.get('comparison_id') == comparison_id
	}
	published_summary_index = {
		(str(row['budget_id']), str(row['metric'])): row
		for row in published_summary
		if row.get('comparison_id') == comparison_id
	}
	if len(published_delta_index) != len(recomputed_deltas):
		raise ValueError('published M1-MAE paired-delta parity row count mismatch')
	if len(published_summary_index) != len(recomputed_summary):
		raise ValueError('published M1-MAE summary parity row count mismatch')
	for row in recomputed_deltas:
		published = published_delta_index[(
			str(row['budget_id']),
			int(row['subsample_seed']),
		)]
		for metric in METRIC_SPECS:
			_assert_float_parity(
				float(row[metric.name]),
				published.get(metric.name),
				label=(
					f'published M1-MAE paired delta '
					f'{row["budget_id"]}/seed{row["subsample_seed"]}/{metric.name}'
				),
			)
	for row in recomputed_summary:
		published = published_summary_index[(str(row['budget_id']), str(row['metric']))]
		for field in (
			'mean_delta',
			'median_delta',
			'min_delta',
			'max_delta',
			'worst_seed_delta',
		):
			_assert_float_parity(
				float(row[field]),
				published.get(field),
				label=(
					f'published M1-MAE summary '
					f'{row["budget_id"]}/{row["metric"]}/{field}'
				),
			)
		for local, published_key in (
			('sample_standard_deviation', 'standard_deviation'),
			('wins', 'positive_win_count'),
			('losses', 'negative_count'),
			('ties', 'zero_count'),
			('worst_seed', 'worst_seed'),
		):
			if str(row[local]) != str(published.get(published_key)):
				raise ValueError(
					f'published M1-MAE summary parity mismatch: '
					f'{row["budget_id"]}/{row["metric"]}/{local}'
				)
	return {
		'status': 'PASS',
		'comparison_id': comparison_id,
		'paired_deltas': _identity(published_deltas_path),
		'summary_by_budget': _identity(published_summary_path),
		'paired_delta_row_count': len(recomputed_deltas),
		'summary_row_count': len(recomputed_summary),
	}


def _assert_float_parity(actual: float, expected: object, *, label: str) -> None:
	try:
		expected_float = float(cast('str', expected))
	except (TypeError, ValueError) as error:
		raise ValueError(f'{label} published value is not numeric') from error
	if not math.isclose(actual, expected_float, rel_tol=0.0, abs_tol=1e-15):
		raise ValueError(
			f'{label} mismatch: current={actual!r}, published={expected_float!r}'
		)


def _readiness_decision(
	config: F3VoxelLabelBudgetControlConfig,
	summary_by_budget: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
	"""Apply descriptive readiness gates without voxel-level inference tests."""
	by_key = {
		(str(row['budget_id']), str(row['comparison_id']), str(row['metric'])): row
		for row in summary_by_budget
	}
	current_vs_mae = _comparison_id(
		config.candidate.model_id, config.references.mae_model_id
	)
	current_vs_m1 = _comparison_id(
		config.candidate.model_id, config.references.historical_m1_model_id
	)
	positive_budgets: list[str] = []
	for budget in config.budgets:
		primary = [
			by_key[(budget, current_vs_mae, metric)]
			for metric in ('macro_f1', 'mean_iou')
		]
		if all(
			float(row['mean_delta']) > 0.0
			and int(row['wins']) >= config.decision.minimum_primary_wins
			for row in primary
		):
			positive_budgets.append(budget)
	major_degradations: list[dict[str, object]] = []
	for class_id in config.decision.monitored_class_ids:
		for metric in (
			'f1',
			'iou',
			'boundary_recall_t2',
			'boundary_recall_t4',
		):
			degraded = [
				budget
				for budget in config.budgets
				if float(
					by_key[(budget, current_vs_mae, f'class_{class_id}_{metric}')][
						'mean_delta'
					]
				)
				<= config.decision.major_degradation_delta
			]
			if len(degraded) >= config.decision.systematic_degradation_budget_count:
				major_degradations.append(
					{
						'class_id': class_id,
						'metric': metric,
						'budgets': degraded,
					}
				)
	drift_triggers: list[dict[str, object]] = []
	for metric in ('macro_f1', 'mean_iou'):
		budgets = [
			budget
			for budget in config.budgets
			if abs(
				float(by_key[(budget, current_vs_m1, metric)]['mean_delta'])
			)
			>= config.decision.drift_absolute_mean_delta
		]
		if len(budgets) >= config.decision.drift_budget_count:
			drift_triggers.append({'metric': metric, 'budgets': budgets})
	positive = (
		len(positive_budgets) >= config.decision.minimum_positive_budgets
		and not major_degradations
	)
	if drift_triggers:
		status = 'CONTROL_READY_WITH_DRIFT'
	elif positive:
		status = 'CONTROL_READY_POSITIVE'
	else:
		status = 'CONTROL_READY_MIXED'
	return {
		'status': status,
		'artifacts_complete': True,
		'paired_identity_mismatch_count': 0,
		'coverage_complete': True,
		'current_k6_vs_mae': {
			'positive_budgets': positive_budgets,
			'minimum_positive_budgets': config.decision.minimum_positive_budgets,
			'minimum_primary_wins': config.decision.minimum_primary_wins,
		},
		'current_k6_vs_historical_m1_drift': {
			'absolute_mean_delta_threshold': config.decision.drift_absolute_mean_delta,
			'drift_budget_count': config.decision.drift_budget_count,
			'triggers': drift_triggers,
		},
		'monitored_class_major_degradations': major_degradations,
	}


def _write_summary_tables(
	config: F3VoxelLabelBudgetControlConfig,
	inspection: F3VoxelLabelBudgetControlResultsInspection,
) -> tuple[Path, ...]:
	paths = (
		(config.reports_dir / CONTROL_JOB_METRICS_CSV, inspection.job_metrics),
		(config.reports_dir / CONTROL_PAIRED_METRICS_CSV, inspection.paired_metrics),
		(config.reports_dir / CONTROL_PAIRED_DELTAS_CSV, inspection.paired_deltas),
		(
			config.reports_dir / CONTROL_SUMMARY_BY_BUDGET_CSV,
			inspection.summary_by_budget,
		),
		(
			config.reports_dir / CONTROL_MONITORED_CLASS_SUMMARY_CSV,
			inspection.monitored_class_summary,
		),
	)
	for path, rows in paths:
		_write_csv(path, rows)
	return tuple(path for path, _ in paths)


def _control_summary_payload(
	config: F3VoxelLabelBudgetControlConfig,
	inspection: F3VoxelLabelBudgetControlResultsInspection,
) -> dict[str, object]:
	return {
		'artifact_type': 'f3_current_k6_control_summary',
		'schema_version': 1,
		'model_tag': config.candidate.model_tag,
		'scientific_scope': {
			'dataset_split': 'F3 original split',
			'label_budgets': list(config.budgets),
			'paired_seeds': list(config.subsample_seeds),
			'decoder': 'frozen_embedding_decoder_nearest_voxel_ln_v1',
			'inference_unit': 'unique validation voxel',
			'inference_note': (
				'Descriptive paired-seed comparison only; no voxel-independent '
				'p-values or confidence intervals are reported.'
			),
		},
		'status': inspection.readiness['status'],
		'readiness': dict(inspection.readiness),
		'completion': {
			'current_candidate_job_count': 15,
			'historical_reference_job_count': 30,
			'comparison_member_count': len(inspection.job_metrics),
			'paired_condition_count': 15,
			'paired_identity_mismatch_count': 0,
			'uncovered_validation_voxel_count': 0,
			'prediction_duplicate_write_count': 0,
			'prediction_missing_write_count': 0,
			'prediction_exact_once': True,
			'validation_voxel_count': REQUIRED_VALIDATION_VOXELS,
			'best_checkpoint_inference_count': 15,
			'complete_evaluation_count': 15,
		},
		'source_identities': dict(inspection.source_identities),
		'historical_m1_mae_parity': dict(inspection.historical_m1_mae_parity),
		'final_git_provenance': _git_provenance(),
		'job_metrics': list(inspection.job_metrics),
		'paired_metrics': list(inspection.paired_metrics),
		'paired_deltas': list(inspection.paired_deltas),
		'summary_by_budget': list(inspection.summary_by_budget),
		'monitored_class_summary': list(inspection.monitored_class_summary),
	}


def _render_control_summary_markdown(payload: Mapping[str, object]) -> str:
	readiness = _mapping(payload.get('readiness'), 'control summary readiness')
	git = _mapping(payload.get('final_git_provenance'), 'final git provenance')
	positive = _mapping(readiness.get('current_k6_vs_mae'), 'positive budgets')
	drift = _mapping(
		readiness.get('current_k6_vs_historical_m1_drift'), 'historical M1 drift'
	)
	lines = [
		'# Current-code single-head K=6 control',
		'',
		f"Status: `{payload['status']}`",
		'',
		'## Scope',
		'',
		'F3 original split; cap25/cap50/cap100; five paired subsample seeds; '
		'fixed frozen voxel decoder. Results are descriptive paired-seed summaries.',
		'',
		'## Readiness',
		'',
		f"- Current K6 vs MAE positive budgets: {positive['positive_budgets']}",
		f"- Historical-M1 drift triggers: {drift['triggers']}",
		'- Monitored-class major degradations: '
		f"{readiness['monitored_class_major_degradations']}",
		'',
		'## Files',
		'',
		'- `control_paired_deltas.csv` contains all per-seed paired deltas.',
		'- `control_summary_by_budget.csv` contains means, medians, sample SD, '
		'ranges, wins/losses/ties, and worst seeds.',
		'- `control_monitored_class_summary.csv` contains class 3/5 F1, IoU, '
		'and boundary-recall summaries.',
		'',
		'## Final repository provenance',
		'',
		f"- HEAD: `{git['head']}`",
		'- `git diff --binary HEAD` SHA-256: '
		f"`{git['git_diff_binary_head_sha256']}`",
		f"- Changed files: `{git['changed_file_count']}`",
		*[
			f"  - `{path}`"
			for path in cast('Sequence[str]', git['changed_files'])
		],
		'',
	]
	return '\n'.join(lines)


def _render_control_handoff_markdown(payload: Mapping[str, object]) -> str:
	git = _mapping(payload.get('final_git_provenance'), 'final git provenance')
	return '\n'.join(
		[
			'# Multi-head handoff: current-code K=6 control',
			'',
			f"Control status: `{payload['status']}`",
			'',
			'Primary baseline for the next comparison matrix:',
			'',
			'`strat_hmm_pretext_m1_current_k6_topblock1_distill_v1`',
			'',
			'Fixed handoff contract:',
			'',
			'- Teacher: same MAE latest checkpoint.',
			'- Student initialization: same MAE latest checkpoint.',
			'- K=6 target: same exact K=6 pseudo-target identity.',
			'- Pretraining: same scientific settings.',
			'- Downstream: same cap25/cap50/cap100 datasets, five paired seeds, '
			'and voxel decoder.',
			'',
			'Next comparisons: current single-head K=6; multi-head K=6/8/10 '
			'with consistency=0; multi-head K=6/8/10 with consistency>0; and '
			'MAE reference. Historical M1 remains a report reference, not the '
			'primary multi-head baseline.',
			'',
			'Final repository provenance:',
			'',
			f"- HEAD: `{git['head']}`",
			'- `git diff --binary HEAD` SHA-256: '
			f"`{git['git_diff_binary_head_sha256']}`",
			f"- Changed files: `{git['changed_file_count']}`",
			*[
				f"  - `{path}`"
				for path in cast('Sequence[str]', git['changed_files'])
			],
			'',
		]
	)


def _validate_required_evidence(config: F3VoxelLabelBudgetControlConfig) -> Path:
	"""Validate all upstream control evidence without materializing it."""
	preflight_source = (
		config.artifact_root
		/ 'pretraining'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ config.candidate.model_tag
		/ 'preflight'
		/ 'control_input_manifest.json'
	)
	if not preflight_source.is_file():
		raise FileNotFoundError(
			f'missing current K6 preflight manifest: {preflight_source}'
		)
	for name in ('checkpoint_validation.json', 'embedding_validation.json'):
		path = config.reports_dir / name
		if not path.is_file():
			raise FileNotFoundError(f'missing current K6 validation evidence: {path}')
		payload = _read_json(path)
		if payload.get('status') != 'PASS':
			raise ValueError(f'current K6 validation evidence is not PASS: {path}')
	token = config.reports_dir / 'token_probe_comparison.csv'
	if not token.is_file():
		raise FileNotFoundError(f'missing current K6 token comparison: {token}')
	return preflight_source


def _materialize_required_evidence(config: F3VoxelLabelBudgetControlConfig) -> None:
	"""Bind upstream preflight evidence into the lightweight control report root."""
	preflight_source = _validate_required_evidence(config)
	preflight_target = config.reports_dir / 'control_input_manifest.json'
	_copy_evidence(preflight_source, preflight_target)


def _copy_evidence(source: Path, target: Path) -> None:
	target.parent.mkdir(parents=True, exist_ok=True)
	if target.is_file() and file_sha256(target) == file_sha256(source):
		return
	shutil.copyfile(source, target)


def _validate_summary_output_availability(
	config: F3VoxelLabelBudgetControlConfig,
	*,
	create_reports_dir: bool = True,
) -> None:
	if create_reports_dir and not config.reports_dir.exists():
		config.reports_dir.mkdir(parents=True, exist_ok=True)
	if not config.publish.enabled:
		return
	publish_root = config.publish.output_dir
	if publish_root.exists():
		_validate_existing_publish_tree(config, expected=_publish_target_names())


def _validate_existing_publish_tree(
	config: F3VoxelLabelBudgetControlConfig, *, expected: set[str]
) -> None:
	"""Read-only validate a nonempty prior publication before replacement."""
	publish_root = config.publish.output_dir
	actual = {
		path.relative_to(publish_root).as_posix()
		for path in publish_root.rglob('*')
		if path.is_file()
	}
	if not actual:
		return
	for path in publish_root.rglob('*'):
		if not path.is_file():
			continue
		if path.suffix.lower() in FORBIDDEN_SUFFIXES:
			raise ValueError(f'raw artifact was published: {path}')
		if path.stat().st_size > config.publish.max_file_size_bytes:
			raise ValueError(f'published file exceeds size limit: {path}')
	actual_without_historical_manifest = actual - {CONTROL_PUBLISH_MANIFEST}
	if actual_without_historical_manifest != expected:
		raise FileExistsError(
			'current K6 publish root has an unexpected file set; '
			f'missing={sorted(expected - actual_without_historical_manifest)!r}, '
			f'extra={sorted(actual_without_historical_manifest - expected)!r}'
		)


def _publish_control_results(
	config: F3VoxelLabelBudgetControlConfig,
) -> tuple[Path, ...]:
	if not config.publish.enabled:
		return ()
	files = []
	for name in sorted(_publish_target_names()):
		source = config.reports_dir / name
		if not source.is_file():
			raise FileNotFoundError(
				f'required control publish source is missing: {source}'
			)
		if source.stat().st_size > config.publish.max_file_size_bytes:
			raise ValueError(f'control publish source exceeds size limit: {source}')
		target = config.publish.output_dir / name
		if target.exists() and not config.publish.overwrite:
			raise FileExistsError(f'publish target already exists: {target}')
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(source, target)
		files.append(target)
	return tuple(files)


def _publish_target_names() -> set[str]:
	return {
		'control_input_manifest.json',
		'checkpoint_validation.json',
		'embedding_validation.json',
		'token_probe_comparison.csv',
		CONTROL_RUN_MANIFEST_NAME,
		CONTROL_STATUS_CSV,
		CONTROL_JOB_METRICS_CSV,
		CONTROL_PAIRED_METRICS_CSV,
		CONTROL_PAIRED_DELTAS_CSV,
		CONTROL_SUMMARY_BY_BUDGET_CSV,
		CONTROL_MONITORED_CLASS_SUMMARY_CSV,
		CONTROL_SUMMARY_JSON,
		CONTROL_SUMMARY_MARKDOWN,
		CONTROL_HANDOFF_MARKDOWN,
	}


def _identity_path(value: object, label: str) -> Path:
	identity = _mapping(value, label)
	path = Path(str(identity.get('path')))
	if not path.is_file():
		raise FileNotFoundError(path)
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} SHA-256 mismatch')
	return path


def _model_label(role: str) -> str:
	return {
		CURRENT_MODEL_ROLE: 'Current K6',
		'm1': 'Historical M1',
		'mae': 'MAE',
	}.get(role, role)


def _comparison_id(candidate_role: str, baseline_role: str) -> str:
	return f'{candidate_role}_vs_{baseline_role}'


def _git_provenance() -> dict[str, object]:
	"""Capture the final repository state used to write a control summary.

	The working tree is intentionally allowed to be dirty for this workflow.
	Record both the complete short-status listing (including untracked files) and
	the exact binary diff from ``HEAD`` so the handoff can identify the code
	state without mutating it.
	"""
	root = _git_text(('rev-parse', '--show-toplevel')).strip()
	if not root:
		raise RuntimeError('git did not return a repository root')
	repository = Path(root)
	head = _git_text(('rev-parse', 'HEAD'), cwd=repository).strip()
	if not head:
		raise RuntimeError('git did not return HEAD')
	status = _git_text(
		('status', '--short', '--untracked-files=all'), cwd=repository
	)
	diff = _git_bytes(('diff', '--binary', 'HEAD'), cwd=repository)
	changed_files = tuple(line for line in status.splitlines() if line)
	diff_sha256 = hashlib.sha256(diff).hexdigest()
	return {
		'repository_root': str(repository),
		'head': head,
		'git_status_short': status,
		'changed_files': changed_files,
		'changed_file_count': len(changed_files),
		'git_diff_binary_head_sha256': diff_sha256,
		'git_diff_sha256': diff_sha256,
	}


def _git_text(arguments: Sequence[str], *, cwd: Path | None = None) -> str:
	git = _git_executable()
	completed = subprocess.run(  # noqa: S603
		[git, *arguments],
		cwd=cwd,
		check=True,
		capture_output=True,
		text=True,
	)
	return completed.stdout


def _git_bytes(arguments: Sequence[str], *, cwd: Path | None = None) -> bytes:
	git = _git_executable()
	completed = subprocess.run(  # noqa: S603
		[git, *arguments],
		cwd=cwd,
		check=True,
		capture_output=True,
	)
	return completed.stdout


def _git_executable() -> str:
	git = shutil.which('git')
	if git is None:
		raise RuntimeError('git executable is unavailable')
	return git


def _stage_config(
	config: F3VoxelLabelBudgetControlConfig,
) -> F3VoxelLabelBudgetSuiteConfig:
	"""Use the model-agnostic shared stage helpers through a narrow adapter."""
	return cast('F3VoxelLabelBudgetSuiteConfig', config)


def _dataset_rows(
	config: F3VoxelLabelBudgetControlConfig,
	reference: F3VoxelLabelBudgetReferenceInspection,
) -> dict[tuple[str, int], Mapping[str, object]]:
	payload = _read_json(config.dataset_manifest)
	rows = _mapping_rows(payload.get('rows'), 'control dataset manifest rows')
	result: dict[tuple[str, int], Mapping[str, object]] = {}
	for row in rows:
		budget = row.get('budget_id')
		seed = row.get('subsample_seed')
		if not isinstance(budget, str) or not isinstance(seed, int):
			raise TypeError('control dataset row budget_id/subsample_seed is invalid')
		key = (budget, seed)
		if key in result:
			raise ValueError(f'duplicate control dataset condition: {key!r}')
		result[key] = row
	expected = {
		(budget, seed)
		for budget in config.budgets
		for seed in config.subsample_seeds
	}
	if set(result) != expected or set(reference.datasets) != expected:
		raise ValueError('control dataset manifest condition matrix mismatch')
	return result


def _validate_reference_contract(
	config: F3VoxelLabelBudgetControlConfig,
	reference: F3VoxelLabelBudgetReferenceInspection,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> None:
	"""Verify required reference pairing before candidate work may begin."""
	_validate_mae_reference_contract(config, reference, dataset_rows)
	if config.validate_pairing_reference:
		_validate_mae_m1_reference_contract(config, reference, dataset_rows)


def _validate_mae_reference_contract(
	config: F3VoxelLabelBudgetControlConfig,
	reference: F3VoxelLabelBudgetReferenceInspection,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> None:
	"""Verify MAE is valid for every configured paired condition."""
	if config.references.mae_model_id != 'mae':
		raise ValueError('references.mae_model_id must be mae')
	by_key = _reference_jobs_by_key(reference)
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			key = (budget, seed)
			mae = by_key[(budget, seed, 'mae')]
			values = _reference_pair_values(mae, dataset_rows[key])
			if values['validation_voxel_count'] != REQUIRED_VALIDATION_VOXELS:
				raise ValueError('historical validation voxel contract mismatch')


def _validate_mae_m1_reference_contract(
	config: F3VoxelLabelBudgetControlConfig,
	reference: F3VoxelLabelBudgetReferenceInspection,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> None:
	"""Verify the full historical MAE/M1 pairing for the control runner."""
	if config.references.historical_m1_model_id != 'm1':
		raise ValueError('references.historical_m1_model_id must be m1')
	by_key = _reference_jobs_by_key(reference)
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			key = (budget, seed)
			mae = by_key[(budget, seed, 'mae')]
			values = _reference_pair_values(mae, dataset_rows[key])
			m1 = by_key[(budget, seed, 'm1')]
			other = _reference_pair_values(m1, dataset_rows[key])
			for name in PAIR_IDENTITY_KEYS:
				if values[name] != other[name]:
					raise ValueError(
						f'historical MAE/M1 paired identity mismatch for '
						f'{budget}/seed{seed}: {name}'
					)
			for name in ('train_mask_sha256', 'resolved_amp_dtype', 'amp_scaler'):
				if values[name] != other[name]:
					raise ValueError(
						f'historical MAE/M1 runtime identity mismatch for '
						f'{budget}/seed{seed}: {name}'
					)


def _candidate_embedding_identity(
	config: F3VoxelLabelBudgetControlConfig,
	reference: F3VoxelLabelBudgetReferenceInspection,
) -> dict[str, Mapping[str, object]]:
	files = output_paths(config.candidate.embeddings_dir, config.dataset['name'])
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		if not path.is_file():
			raise FileNotFoundError(path)
	metadata = _read_json(files.metadata)
	checkpoint_value = metadata.get('checkpoint_path')
	if not isinstance(checkpoint_value, str) or not checkpoint_value:
		raise ValueError('candidate embedding metadata checkpoint_path is missing')
	checkpoint = Path(checkpoint_value)
	if (
		checkpoint.name != 'best.pt'
		or config.candidate.model_tag not in checkpoint.parts
	):
		raise ValueError('candidate embedding checkpoint/model identity mismatch')
	if not checkpoint.is_file():
		raise FileNotFoundError(checkpoint)
	if metadata.get('checkpoint_sha256') != file_sha256(checkpoint):
		raise ValueError('candidate embedding checkpoint SHA-256 mismatch')
	stratigraphy = _mapping(
		metadata.get('stratigraphy_pretext'), 'candidate embedding stratigraphy_pretext'
	)
	if stratigraphy.get('model_tag') != config.candidate.model_tag:
		raise ValueError('candidate embedding stratigraphy model tag mismatch')
	if metadata.get('output_dtype') != 'float16':
		raise ValueError('candidate embedding output dtype contract mismatch')
	valid_sha = file_sha256(files.valid_tokens)
	reference_valid = {
		job.canonical_valid_tokens_sha256 for job in reference.jobs
	}
	if len(reference_valid) != 1 or valid_sha not in reference_valid:
		raise ValueError('candidate valid-token identity differs from MAE/M1')
	return {
		'embeddings': _identity(files.embeddings),
		'valid_tokens': _identity(files.valid_tokens),
		'metadata': _identity(files.metadata),
		'checkpoint': _identity(checkpoint),
	}


def _jobs(
	config: F3VoxelLabelBudgetControlConfig,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[VoxelLabelBudgetJob, ...]:
	jobs: list[VoxelLabelBudgetJob] = []
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			row = dataset_rows[(budget, seed)]
			jobs.append(
				VoxelLabelBudgetJob(
					budget_id=budget,
					per_class_cap=int(row['per_class_cap']),
					subsample_seed=seed,
					decoder_seed=config.decoder_seed(seed),
					model_role=config.candidate.model_id,
					model_tag=config.candidate.model_tag,
					voxel_dataset_root=Path(str(row['voxel_dataset_root'])),
					output_root=(
						config.output_root
						/ 'jobs'
						/ f'budget={budget}'
						/ f'subsample_seed={seed}'
						/ f'model={config.candidate.model_tag}'
					),
					dataset_row=row,
				)
			)
	return tuple(jobs)


def _completed_control_row(  # noqa: PLR0913
	config: F3VoxelLabelBudgetControlConfig,
	stage_config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	candidate_embedding_identity: Mapping[str, Mapping[str, object]],
	action: str,
	quarantine_path: Path | None,
	error: str | None,
) -> dict[str, object]:
	row = completed_voxel_label_budget_job_row(
		stage_config,
		job,
		action=action,
		quarantine_path=quarantine_path,
		error=error,
	)
	if row.get('model_role') != config.candidate.model_id:
		raise ValueError('completed control candidate model role mismatch')
	if row.get('model_tag') != config.candidate.model_tag:
		raise ValueError('completed control candidate model tag mismatch')
	row['train_mask_sha256'] = _sha256_value(
		job.dataset_row.get('train_mask_sha256'), 'candidate train mask SHA-256'
	)
	row['candidate_embedding_identity'] = dict(candidate_embedding_identity)
	coverage = _prediction_coverage(job.prediction_dir / 'prediction_metadata.json')
	row['prediction_coverage'] = coverage
	row['prediction_duplicate_write_count'] = coverage['duplicate_write_count']
	row['prediction_missing_write_count'] = coverage['missing_write_count']
	row['prediction_exact_once'] = coverage['exact_once']
	row['runtime_contract'] = _runtime_contract(job.decoder_dir / 'latest.pt')
	return row


def _validate_paired_identity(
	row: Mapping[str, object],
	*,
	reference: F3VoxelLabelBudgetReferenceInspection,
	dataset_row: Mapping[str, object],
	reference_roles: Sequence[str] = REFERENCE_MODEL_ROLES,
) -> None:
	"""Reject a condition unless candidate and required references pair exactly."""
	if not reference_roles or any(
		role not in REFERENCE_MODEL_ROLES for role in reference_roles
	):
		raise ValueError('paired reference roles are invalid')
	by_key = _reference_jobs_by_key(reference)
	budget = str(row.get('budget_id'))
	seed = int(row.get('subsample_seed', -1))
	candidate = _candidate_pair_values(row)
	for role in reference_roles:
		other = _reference_pair_values(by_key[(budget, seed, role)], dataset_row)
		for name in PAIR_IDENTITY_KEYS:
			if candidate[name] != other[name]:
				raise ValueError(
					f'paired identity mismatch for {budget}/seed{seed} '
					f'current-K6 vs {role}: {name}'
				)
		for name in ('train_mask_sha256', 'resolved_amp_dtype', 'amp_scaler'):
			if candidate[name] != other[name]:
				raise ValueError(
					f'paired runtime identity mismatch for {budget}/seed{seed} '
					f'current-K6 vs {role}: {name}'
				)
	if candidate['validation_voxel_count'] != REQUIRED_VALIDATION_VOXELS:
		raise ValueError('candidate validation voxel contract mismatch')
	if candidate['uncovered_validation_voxel_count'] != 0:
		raise ValueError('candidate has uncovered validation voxels')


def _reference_jobs_by_key(
	reference: F3VoxelLabelBudgetReferenceInspection,
) -> Mapping[tuple[str, int, str], object]:
	return {
		(job.dataset.budget_id, job.dataset.subsample_seed, job.model_role): job
		for job in reference.jobs
	}


def _reference_pair_values(
	job: object, dataset_row: Mapping[str, object]
) -> dict[str, object]:
	"""Normalize one validated historical job to candidate manifest field names."""
	loaded = cast('object', job)
	# The public reference inspection intentionally exposes validated job records.
	# Attribute access is kept here, rather than duplicating its artifact parser.
	dataset = loaded.dataset
	evaluation = loaded.evaluation
	result = {
		'voxel_supervision_grid_sha256': dataset.grid.sha256,
		'selected_token_identity_sha256': dataset.selected_token_identity_sha256,
		'unique_token_xyz_sha256': dataset.unique_token_xyz_sha256,
		'train_voxel_count': dataset.train_voxel_count,
		'validation_voxel_count': dataset.validation_voxel_count,
		'class_order': list(evaluation.class_order),
		'validation_mask_sha256': dataset.validation_mask_sha256,
		'canonical_valid_token_sha256': loaded.canonical_valid_tokens_sha256,
		'initial_model_state_sha256': loaded.initial_model_state_sha256,
		'class_weights': list(loaded.class_weights),
		'decoder_architecture': dict(evaluation.decoder_architecture),
		'decoder_seed': loaded.decoder_seed,
		'sampling_mode': loaded.sampling_mode,
		'steps_per_epoch': loaded.steps_per_epoch,
		'sampling_sequence_sha256': loaded.sampling_sequence_sha256,
		'train_tile_manifest_sha256': loaded.train_tile_manifest_sha256,
		'validation_tile_manifest_sha256': loaded.validation_tile_manifest_sha256,
		'train_tile_identity_sha256': loaded.train_tile_identity_sha256,
		'validation_tile_identity_sha256': loaded.validation_tile_identity_sha256,
		'metric_schema_sha256': evaluation.metric_schema_sha256,
		'uncovered_validation_voxel_count': 0,
		'train_mask_sha256': _sha256_value(
			dataset_row.get('train_mask_sha256'), 'reference train mask SHA-256'
		),
	}
	runtime = _runtime_contract(
		Path(str(loaded.row['latest_checkpoint']['path']))
	)
	coverage = _prediction_coverage(
		Path(str(loaded.row['prediction_metadata']['path']))
	)
	result['prediction_duplicate_write_count'] = coverage['duplicate_write_count']
	result['prediction_missing_write_count'] = coverage['missing_write_count']
	result['prediction_exact_once'] = coverage['exact_once']
	result['resolved_amp_dtype'] = runtime['resolved_amp_dtype']
	result['amp_scaler'] = runtime['amp_scaler']
	return result


def _candidate_pair_values(row: Mapping[str, object]) -> dict[str, object]:
	result = {name: row.get(name) for name in PAIR_IDENTITY_KEYS}
	runtime = _mapping(row.get('runtime_contract'), 'candidate runtime contract')
	result['train_mask_sha256'] = _sha256_value(
		row.get('train_mask_sha256'), 'candidate train mask SHA-256'
	)
	result['resolved_amp_dtype'] = runtime.get('resolved_amp_dtype')
	result['amp_scaler'] = runtime.get('amp_scaler')
	return result


def _runtime_contract(checkpoint: Path) -> dict[str, object]:
	payload = load_voxel_decoder_checkpoint(checkpoint)
	runtime = _mapping(payload.get('runtime_identity'), 'decoder runtime_identity')
	device = runtime.get('device')
	scaler = runtime.get('amp_scaler')
	if not isinstance(device, str) or not device:
		raise TypeError('decoder runtime_identity.device must be a non-empty string')
	if not isinstance(scaler, bool):
		raise TypeError('decoder runtime_identity.amp_scaler must be boolean')
	device_type = device.split(':', maxsplit=1)[0]
	if scaler and device_type != 'cuda':
		raise ValueError('AMP scaler requires a CUDA decoder runtime')
	return {
		'device_type': device_type,
		'amp_scaler': scaler,
		'resolved_amp_dtype': 'float16' if scaler else 'disabled',
	}


def _prediction_coverage(metadata_path: Path) -> dict[str, object]:
	metadata = _read_json(metadata_path)
	coverage = _mapping(metadata.get('coverage'), 'prediction coverage')
	duplicate = coverage.get('duplicate_write_count')
	missing = coverage.get('missing_write_count')
	exact_once = coverage.get('exact_once')
	if duplicate != 0:
		raise ValueError('prediction coverage duplicate_write_count must be zero')
	if missing != 0:
		raise ValueError('prediction coverage missing_write_count must be zero')
	if exact_once is not True:
		raise ValueError('prediction coverage exact_once must be true')
	return {
		'duplicate_write_count': 0,
		'missing_write_count': 0,
		'exact_once': True,
	}


def _prior_control_state(
	path: Path,
	*,
	config: F3VoxelLabelBudgetControlConfig,
	candidate_embedding_identity: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], ...], list[Path]]:
	if not path.is_file():
		return (), []
	payload = _read_json(path)
	_validate_control_manifest_header(
		payload,
		config=config,
		candidate_embedding_identity=candidate_embedding_identity,
	)
	rows = _mapping_rows(payload.get('rows'), 'existing control run rows')
	keys = [_control_row_key(row) for row in rows]
	if len(keys) != len(set(keys)):
		raise ValueError('existing control run manifest contains duplicate rows')
	allowed = {
		(budget, seed, config.candidate.model_id)
		for budget in config.budgets
		for seed in config.subsample_seeds
	}
	if any(key not in allowed for key in keys):
		raise ValueError('existing control run manifest contains unknown job rows')
	quarantine_values = payload.get('quarantines', [])
	if not isinstance(quarantine_values, Sequence) or isinstance(
		quarantine_values, str | bytes
	):
		raise TypeError('existing control run manifest quarantines must be a list')
	return tuple(rows), [Path(str(value)) for value in quarantine_values]


def _write_control_manifest(
	path: Path,
	*,
	config: F3VoxelLabelBudgetControlConfig,
	rows: Sequence[Mapping[str, object]],
	quarantines: Sequence[Path],
	candidate_embedding_identity: Mapping[str, Mapping[str, object]],
) -> None:
	ordered = tuple(sorted(rows, key=_row_sort_key))
	_write_json(
		path,
		{
			'artifact_type': CONTROL_RUN_MANIFEST_TYPE,
			'schema_version': CONTROL_RUN_SCHEMA_VERSION,
			'control_contract': _control_contract(config),
			'dataset_manifest': _identity(config.dataset_manifest),
			'historical_run_manifest': _identity(config.historical_run_manifest),
			'candidate_embedding_identity': dict(candidate_embedding_identity),
			'row_count': len(ordered),
			'complete_count': sum(row.get('status') == 'complete' for row in ordered),
			'rows': list(ordered),
			'quarantines': [str(item) for item in quarantines],
			'updated_at_utc': datetime.now(timezone.utc).isoformat(),
		},
	)


def _validate_control_manifest_header(
	payload: Mapping[str, object],
	*,
	config: F3VoxelLabelBudgetControlConfig,
	candidate_embedding_identity: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
	if payload.get('artifact_type') != CONTROL_RUN_MANIFEST_TYPE:
		raise ValueError('control run manifest artifact_type mismatch')
	if payload.get('schema_version') != CONTROL_RUN_SCHEMA_VERSION:
		raise ValueError('control run manifest schema_version mismatch')
	expected_contract = _control_contract(config)
	if not config.validate_pairing_reference:
		# The immutable K=6 manifest was produced under the original MAE/M1
		# control contract. Multi-head validation reuses its candidate artifacts
		# without making historical M1 a live prerequisite.
		expected_contract['references'] = {
			'mae_model_id': 'mae',
			'historical_m1_model_id': 'm1',
		}
	if payload.get('control_contract') != expected_contract:
		raise ValueError('control run manifest contract mismatch')
	_validate_identity_at(
		payload.get('dataset_manifest'),
		config.dataset_manifest,
		label='control dataset manifest',
	)
	if config.validate_pairing_reference:
		_validate_identity_at(
			payload.get('historical_run_manifest'),
			config.historical_run_manifest,
			label='control historical run manifest',
		)
	if (
		candidate_embedding_identity is not None
		and payload.get('candidate_embedding_identity')
		!= dict(candidate_embedding_identity)
	):
		raise ValueError('control run manifest candidate embedding identity mismatch')
	if not isinstance(payload.get('row_count'), int):
		raise TypeError('control run manifest row_count must be an integer')
	if not isinstance(payload.get('complete_count'), int):
		raise TypeError('control run manifest complete_count must be an integer')


def _control_contract(config: F3VoxelLabelBudgetControlConfig) -> dict[str, object]:
	return {
		'candidate': {
			'model_id': config.candidate.model_id,
			'model_tag': config.candidate.model_tag,
			'embeddings_dir': str(config.candidate.embeddings_dir),
		},
		'references': {
			'mae_model_id': config.references.mae_model_id,
			'historical_m1_model_id': config.references.historical_m1_model_id,
		},
		'budgets': list(config.budgets),
		'subsample_seeds': list(config.subsample_seeds),
		'seed_policy': {
			'base_seed': config.base_seed,
			'add_subsample_seed': True,
		},
		'decoder': {
			'spec': config.decoder.spec,
			'embedding_dim': config.decoder.embedding_dim,
			'class_count': config.decoder.class_count,
			'hidden_channels': list(config.decoder.hidden_channels),
			'upsample_factors': [
				list(factor) for factor in config.decoder.upsample_factors
			],
			'upsample_mode': config.decoder.upsample_mode,
			'normalization': config.decoder.normalization,
		},
		'tiles': {
			'core_size_tokens': list(config.tiles.core_size_tokens),
			'context_halo_tokens': list(config.tiles.context_halo_tokens),
		},
		'train': {
			'epochs': config.train.epochs,
			'batch_size': config.train.batch_size,
			'learning_rate': config.train.learning_rate,
			'weight_decay': config.train.weight_decay,
			'class_weight': config.train.class_weight,
			'seed': config.train.seed,
			'num_workers': config.train.num_workers,
			'amp': config.train.amp,
			'gradient_clip_norm': config.train.gradient_clip_norm,
			'sampling_mode': config.train.sampling_mode,
			'steps_per_epoch': config.train.steps_per_epoch,
		},
		'evaluation': dict(config.evaluation),
	}


def _estimated_candidate_job_bytes(
	reference: F3VoxelLabelBudgetReferenceInspection,
) -> int:
	roots = [
		Path(str(job.row['latest_checkpoint']['path'])).parent.parent
		for job in reference.jobs
		if job.model_role == 'm1'
	]
	if not roots:
		roots = [
			Path(str(job.row['latest_checkpoint']['path'])).parent.parent
			for job in reference.jobs
			if job.model_role == 'mae'
		]
	if not roots:
		raise ValueError('historical reference is missing MAE jobs')
	sizes = [_tree_size(path) for path in roots]
	return max(1, math.ceil((sum(sizes) / len(sizes)) * 1.2))


def _failed_row(
	job: VoxelLabelBudgetJob,
	*,
	action: str,
	error: BaseException,
	quarantine_path: Path | None,
) -> dict[str, object]:
	return {
		'budget_id': job.budget_id,
		'per_class_cap': job.per_class_cap,
		'subsample_seed': job.subsample_seed,
		'decoder_seed': job.decoder_seed,
		'model_role': job.model_role,
		'model_tag': job.model_tag,
		'status': 'failed',
		'action': action,
		'error': f'{type(error).__name__}: {error}',
		'quarantine_path': None if quarantine_path is None else str(quarantine_path),
	}


def _write_status_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	fields = (
		'budget_id',
		'per_class_cap',
		'subsample_seed',
		'decoder_seed',
		'model_role',
		'model_tag',
		'status',
		'action',
		'global_step',
		'uncovered_validation_voxel_count',
		'error',
		'quarantine_path',
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction='ignore')
		writer.writeheader()
		writer.writerows(rows)


def _write_blocked_control_contract(
	config: F3VoxelLabelBudgetControlConfig,
	*,
	stage: str,
	error: Exception,
) -> None:
	"""Persist a lightweight blocked status without modifying scientific inputs."""
	config.reports_dir.mkdir(parents=True, exist_ok=True)
	payload = {
		'artifact_type': 'f3_current_k6_control_summary',
		'schema_version': 1,
		'model_tag': config.candidate.model_tag,
		'status': BLOCKED_CONTROL_CONTRACT,
		'blocked_stage': stage,
		'error': f'{type(error).__name__}: {error}',
		'created_at_utc': datetime.now(timezone.utc).isoformat(),
	}
	_write_json(config.reports_dir / CONTROL_SUMMARY_JSON, payload)
	_write_status_csv(
		config.reports_dir / CONTROL_STATUS_CSV,
		(
			{
				'budget_id': '',
				'per_class_cap': '',
				'subsample_seed': '',
				'decoder_seed': '',
				'model_role': config.candidate.model_id,
				'model_tag': config.candidate.model_tag,
				'status': BLOCKED_CONTROL_CONTRACT,
				'action': stage,
				'global_step': '',
				'uncovered_validation_voxel_count': '',
				'error': payload['error'],
				'quarantine_path': '',
			},
		),
	)


def _identity(path: Path) -> dict[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _validate_identity_at(value: object, path: Path, *, label: str) -> None:
	identity = _mapping(value, label)
	actual = Path(str(identity.get('path')))
	if actual.resolve(strict=False) != path.resolve(strict=False):
		raise ValueError(f'{label} path mismatch')
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} SHA-256 mismatch')
	if identity.get('byte_size') != path.stat().st_size:
		raise ValueError(f'{label} byte-size mismatch')


def _sha256_value(value: object, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value.lower())
	):
		raise ValueError(f'{label} must be a hexadecimal SHA-256')
	return value.lower()


def _tree_size(path: Path) -> int:
	if not path.exists():
		return 0
	return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	return _mapping(payload, f'JSON object: {path}')


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.tmp')
	temporary.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
	if not path.is_file():
		raise FileNotFoundError(path)
	with path.open(encoding='utf-8', newline='') as file_obj:
		return list(csv.DictReader(file_obj))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	if not rows:
		raise ValueError(f'cannot write an empty control table: {path}')
	fieldnames = tuple(
		dict.fromkeys(key for row in rows for key in row).keys()
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction='raise')
		writer.writeheader()
		writer.writerows(rows)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _mapping_rows(value: object, label: str) -> tuple[Mapping[str, object], ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a list')
	return tuple(_mapping(item, label) for item in value)


def _job_key(job: VoxelLabelBudgetJob) -> tuple[str, int, str]:
	return (job.budget_id, job.subsample_seed, job.model_role)


def _control_row_key(row: Mapping[str, object]) -> tuple[str, int, str]:
	budget = row.get('budget_id')
	seed = row.get('subsample_seed')
	role = row.get('model_role')
	if (
		not isinstance(budget, str)
		or not isinstance(seed, int)
		or not isinstance(role, str)
	):
		raise TypeError('control run row key is invalid')
	return (budget, seed, role)


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, int, str]:
	budget, seed, role = _control_row_key(row)
	try:
		cap = int(budget.removeprefix('cap'))
	except ValueError:
		cap = 10**9
	return (cap, seed, role)


__all__ = [
	'CONTROL_RUN_MANIFEST_NAME',
	'CONTROL_RUN_MANIFEST_TYPE',
	'CONTROL_STATUS_CSV',
	'CURRENT_MODEL_ROLE',
	'PAIR_IDENTITY_KEYS',
	'REQUIRED_VALIDATION_VOXELS',
	'F3VoxelLabelBudgetControlInspection',
	'F3VoxelLabelBudgetControlRunResult',
	'control_run_manifest_path',
	'inspect_f3_lithology_voxel_label_budget_control',
	'load_f3_lithology_voxel_label_budget_control_rows',
	'run_f3_lithology_voxel_label_budget_control',
]
