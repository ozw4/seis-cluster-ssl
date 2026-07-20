"""Resumable 45-job runner for the F3 voxel label-budget benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch.utils.data import RandomSampler

from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	F3LithologyVoxelDecoderConfig,
)
from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	F3LithologyVoxelEvaluationConfig,
)
from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	F3LithologyVoxelInferenceConfig,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.voxel_decoder_inference import (
	inspect_f3_lithology_voxel_inference,
	predict_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	BOUNDARY_METRICS_JSON,
	BOUNDARY_REGION_METRICS_CSV,
	EVALUATION_METADATA_JSON,
	METRICS_JSON,
	evaluate_f3_lithology_voxels,
	inspect_f3_lithology_voxel_evaluation,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget import (
	DATASET_MANIFEST_NAME,
	LABEL_BUDGET_METADATA_NAME,
	LABEL_BUDGET_SUMMARY_NAME,
	MANIFEST_ARTIFACT_TYPE,
	REQUIRED_CONDITION_FILES,
	validate_voxel_label_budget_condition_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget import (
	SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	f3_voxel_prediction_artifact_paths,
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_report import (
	F3LithologyVoxelPublishConfig,
	F3LithologyVoxelReportConfig,
	build_f3_lithology_voxel_report,
	build_f3_lithology_voxel_report_payload,
	inspect_f3_lithology_voxel_report,
	render_f3_lithology_voxel_report_markdown,
)
from seis_ssl_cluster.f3.lithology.voxel_tiles import (
	VoxelTileManifest,
	read_voxel_tile_manifest,
)
from seis_ssl_cluster.f3.lithology.voxel_visualization import (
	F3LithologyVoxelFigureConfig,
)
from seis_ssl_cluster.models.voxel_decoder.spec import (
	voxel_decoder_architecture_mapping,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)
from seis_ssl_cluster.training.voxel_decoder.losses import (
	balanced_class_weights_from_counts,
)
from seis_ssl_cluster.training.voxel_decoder.runner import (
	run_f3_lithology_voxel_decoder,
	validate_f3_lithology_voxel_decoder_resume,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_suite import (
		F3VoxelLabelBudgetSuiteConfig,
	)
	from seis_ssl_cluster.f3.lithology.voxel_report import (
		F3LithologyVoxelReportInspection,
	)

RUN_MANIFEST_NAME = 'voxel_label_budget_run_manifest.json'
RUN_MANIFEST_TYPE = 'f3_lithology_voxel_label_budget_run_manifest'
RUN_SCHEMA_VERSION = 1
JOB_STATES = ('NEW', 'RESUME_LATEST', 'REUSE_COMPLETED', 'INVALID_OR_PARTIAL')
MODEL_ORDER = ('mae', 'm1', 'm2a')
CHECKPOINT_REQUIRED = ('latest.pt', 'best.pt', 'history.csv', 'resolved_config.json')
DATASET_SUITE_NAME = 'f3_voxel_label_budget_original_v1'
DATASET_SOURCE_KEYS = frozenset(
	{
		'common_grid',
		'common_metadata',
		'common_class_counts',
		'common_split_manifest',
		'mae_m1_token_manifest',
		'm1_m2a_token_manifest',
	}
)
DATASET_ROW_FILE_NAMES = {
	'supervision_split_grid': 'supervision_split_grid.npy',
	'voxel_dataset_metadata': 'voxel_dataset_metadata.json',
	'class_counts': 'class_counts.csv',
	'split_manifest': 'split_manifest.json',
	'voxel_label_budget_metadata': LABEL_BUDGET_METADATA_NAME,
	'voxel_label_budget_summary': LABEL_BUDGET_SUMMARY_NAME,
}


@dataclass(frozen=True)
class VoxelLabelBudgetJob:
	"""One model member of a budget/seed triplet."""

	budget_id: str
	per_class_cap: int
	subsample_seed: int
	decoder_seed: int
	model_role: str
	model_tag: str
	voxel_dataset_root: Path
	output_root: Path
	dataset_row: Mapping[str, object]

	@property
	def decoder_dir(self) -> Path:
		"""Return the decoder artifact directory."""
		return self.output_root / 'decoder'

	@property
	def prediction_dir(self) -> Path:
		"""Return the prediction artifact directory."""
		return self.output_root / 'prediction'

	@property
	def evaluation_dir(self) -> Path:
		"""Return the evaluation artifact directory."""
		return self.output_root / 'evaluation'

	@property
	def report_dir(self) -> Path:
		"""Return the individual report artifact directory."""
		return self.output_root / 'report'

	@property
	def generated_configs_dir(self) -> Path:
		"""Return the generated stage-config snapshot directory."""
		return self.output_root / 'generated_configs'


@dataclass(frozen=True)
class VoxelLabelBudgetJobPlan:
	"""State classification and intended action for one job."""

	job: VoxelLabelBudgetJob
	state: str
	reason: str | None = None
	estimated_bytes: int = 0


@dataclass(frozen=True)
class VoxelLabelBudgetSuiteInspection:
	"""Preflighted jobs, full-label identity, and storage estimate."""

	jobs: tuple[VoxelLabelBudgetJob, ...]
	plans: tuple[VoxelLabelBudgetJobPlan, ...]
	canonical_steps_per_epoch: int
	canonical_train_tile_file_sha256: str
	canonical_validation_tile_file_sha256: str
	estimated_new_bytes: int
	disk_free_bytes: int


@dataclass(frozen=True)
class VoxelLabelBudgetSuiteResult:
	"""Persisted run manifest and completed rows."""

	manifest_json: Path
	rows: tuple[Mapping[str, object], ...]
	quarantines: tuple[Path, ...]


def inspect_f3_lithology_voxel_label_budget_suite(
	config: F3VoxelLabelBudgetSuiteConfig,
	*,
	budget: str | None = None,
	subsample_seed: int | None = None,
	model: str | None = None,
) -> VoxelLabelBudgetSuiteInspection:
	"""Validate preregistered sources and classify every selected job."""
	steps, train_sha, validation_sha, reference_bytes = _full_reference_contract(config)
	if config.train.steps_per_epoch != steps:
		raise ValueError(
			'preregistered train.steps_per_epoch does not match full-label tile count; '
			f'configured={config.train.steps_per_epoch}, canonical={steps}'
		)
	dataset_rows = _dataset_rows(config)
	_validate_model_embeddings(config)
	jobs = _jobs(config, dataset_rows=dataset_rows)
	jobs = tuple(
		job
		for job in jobs
		if (budget is None or job.budget_id == budget)
		and (subsample_seed is None or job.subsample_seed == subsample_seed)
		and (model is None or job.model_role == model)
	)
	if not jobs:
		raise ValueError('job filters selected no voxel label-budget jobs')
	plans = tuple(
		_classify_job(config, job, estimated_bytes=reference_bytes) for job in jobs
	)
	disk = shutil.disk_usage(config.output_root.parent)
	estimated = sum(
		plan.estimated_bytes for plan in plans if plan.state != 'REUSE_COMPLETED'
	)
	return VoxelLabelBudgetSuiteInspection(
		jobs=jobs,
		plans=plans,
		canonical_steps_per_epoch=steps,
		canonical_train_tile_file_sha256=train_sha,
		canonical_validation_tile_file_sha256=validation_sha,
		estimated_new_bytes=estimated,
		disk_free_bytes=disk.free,
	)


def run_f3_lithology_voxel_label_budget_smoke(
	config: F3VoxelLabelBudgetSuiteConfig,
	*,
	budget: str = 'cap25',
	subsample_seed: int = 0,
	device: str = 'cpu',
) -> tuple[Mapping[str, object], ...]:
	"""Run the non-scientific two-step three-model gate."""
	inspection = inspect_f3_lithology_voxel_label_budget_suite(
		config, budget=budget, subsample_seed=subsample_seed
	)
	jobs = tuple(job for job in inspection.jobs if job.model_role in MODEL_ORDER)
	if len(jobs) != 3:
		raise ValueError('smoke requires exactly one MAE/M1/M2-A triplet')
	rows: list[Mapping[str, object]] = []
	quarantines: list[Path] = []
	manifest_path = _smoke_manifest_path(
		config, budget=budget, subsample_seed=subsample_seed
	)
	# A stale success marker must never survive a failed rerun. Invalidate it
	# before moving or executing any model output, while preserving it for audit.
	if manifest_path.exists():
		quarantines.append(
			_quarantine(manifest_path, reason='smoke_rerun_manifest')
		)
	for job in jobs:
		smoke_root = _smoke_model_root(
			config,
			budget=budget,
			subsample_seed=subsample_seed,
			model_tag=job.model_tag,
		)
		if smoke_root.exists():
			quarantines.append(_quarantine(smoke_root, reason='smoke_rerun'))
		smoke_job = replace(job, output_root=smoke_root)
		train_config = _decoder_config(config, smoke_job)
		print(
			f'smoke.start budget={budget} seed={subsample_seed} model={job.model_role}',
			flush=True,
		)
		result = run_f3_lithology_voxel_decoder(
			train_config, device=device, max_steps=2
		)
		if result.completed or result.global_step != 2:
			raise RuntimeError('smoke must stop at exactly two optimizer steps')
		row = _validated_smoke_row(
			config, smoke_job, checkpoint=result.latest_checkpoint
		)
		rows.append(row)
		print(f'smoke.complete model={job.model_role} global_step=2', flush=True)
	_validate_triplet(rows, context='smoke')
	_write_json(
		manifest_path,
		{
			'artifact_type': 'f3_lithology_voxel_label_budget_smoke_manifest',
			'schema_version': 1,
			'scientific_result': False,
			'dataset_manifest': _identity(config.dataset_manifest),
			'contract': {
				'budget_id': budget,
				'subsample_seed': subsample_seed,
				'global_step': 2,
				'sampling_mode': config.train.sampling_mode,
				'steps_per_epoch': config.train.steps_per_epoch,
			},
			'quarantines': [str(path) for path in quarantines],
			'rows': rows,
		},
	)
	return tuple(rows)


def run_f3_lithology_voxel_label_budget_suite(  # noqa: C901, PLR0912, PLR0913, PLR0915
	config: F3VoxelLabelBudgetSuiteConfig,
	*,
	only_missing: bool = False,
	device: str = 'auto',
	budget: str | None = None,
	subsample_seed: int | None = None,
	model: str | None = None,
) -> VoxelLabelBudgetSuiteResult:
	"""Run selected jobs in budget/seed/triplet order with strict reuse."""
	_validate_smoke_gate(config)
	inspection = inspect_f3_lithology_voxel_label_budget_suite(
		config, budget=budget, subsample_seed=subsample_seed, model=model
	)
	if inspection.estimated_new_bytes > inspection.disk_free_bytes:
		raise RuntimeError(
			'insufficient disk for planned jobs; '
			f'required={inspection.estimated_new_bytes}, '
			f'free={inspection.disk_free_bytes}'
		)
	if not only_missing and any(plan.state != 'NEW' for plan in inspection.plans):
		first = next(plan for plan in inspection.plans if plan.state != 'NEW')
		raise FileExistsError(
			'non-new job requires --only-missing; '
			f'{first.job.output_root}: {first.state}: {first.reason}'
		)
	manifest_path = config.output_root / RUN_MANIFEST_NAME
	prior_rows, prior_quarantines, prior_disk_audits = _prior_run_state(
		manifest_path, config=config
	)
	rows_by_key = {
		_row_key(row): row
		for row in prior_rows
		if _row_key(row) not in {_job_key(plan.job) for plan in inspection.plans}
	}
	quarantines: list[Path] = list(prior_quarantines)
	disk_audits: list[Mapping[str, object]] = [
		*prior_disk_audits,
		_disk_audit(config.output_root, 'before'),
	]
	current_budget: str | None = None
	triplet_rows: dict[tuple[str, int], list[Mapping[str, object]]] = {}
	for plan in inspection.plans:
		job = plan.job
		if current_budget is not None and job.budget_id != current_budget:
			disk_audits.append(
				_disk_audit(config.output_root, f'after_{current_budget}')
			)
		current_budget = job.budget_id
		state = plan.state
		quarantine_path: Path | None = None
		quarantine_reason: str | None = None
		if state == 'INVALID_OR_PARTIAL':
			if not only_missing:
				raise FileExistsError(
					f'invalid/partial job requires --only-missing: {job.output_root}'
				)
			quarantine_reason = plan.reason or 'invalid_or_partial'
			quarantine_path = _quarantine(job.output_root, reason=quarantine_reason)
			quarantines.append(quarantine_path)
			state = 'NEW'
		try:
			if state == 'REUSE_COMPLETED':
				row = _completed_job_row(
					config, job, action='REUSED', quarantine_path=None, error=None
				)
			else:
				resume = (
					job.decoder_dir / 'latest.pt' if state == 'RESUME_LATEST' else None
				)
				action = 'RESUMED' if resume is not None else 'NEW'
				print(
					f'job.start budget={job.budget_id} seed={job.subsample_seed} '
					f'model={job.model_role} action={action}',
					flush=True,
				)
				_run_job(config, job, device=device, resume=resume)
				row = _completed_job_row(
					config,
					job,
					action=action,
					quarantine_path=quarantine_path,
					error=quarantine_reason,
				)
				print(
					f'job.complete budget={job.budget_id} seed={job.subsample_seed} '
					f'model={job.model_role}',
					flush=True,
				)
		except BaseException as error:
			failed = _failed_row(
				job,
				action=('RESUMED' if state == 'RESUME_LATEST' else 'NEW'),
				error=error,
				quarantine_path=quarantine_path,
			)
			rows_by_key[_job_key(job)] = failed
			_write_run_manifest(
				manifest_path,
				config=config,
				rows=tuple(rows_by_key.values()),
				quarantines=quarantines,
				disk_audits=disk_audits,
			)
			raise
		rows_by_key[_job_key(job)] = row
		condition_key = (job.budget_id, job.subsample_seed)
		condition_rows = triplet_rows.setdefault(condition_key, [])
		condition_rows.append(row)
		if len(condition_rows) == 3:
			try:
				_validate_triplet(
					condition_rows,
					context=f'{job.budget_id}/seed{job.subsample_seed}',
				)
			except BaseException as error:
				rows_by_key[_job_key(job)] = {
					**row,
					'status': 'failed',
					'error': f'{type(error).__name__}: {error}',
				}
				_write_run_manifest(
					manifest_path,
					config=config,
					rows=tuple(rows_by_key.values()),
					quarantines=quarantines,
					disk_audits=disk_audits,
				)
				raise
			del triplet_rows[condition_key]
		_write_run_manifest(
			manifest_path,
			config=config,
			rows=tuple(rows_by_key.values()),
			quarantines=quarantines,
			disk_audits=disk_audits,
		)
	if current_budget is not None:
		disk_audits.append(_disk_audit(config.output_root, f'after_{current_budget}'))
	if model is None and triplet_rows:
		raise RuntimeError(
			f'incomplete model triplets after execution: {triplet_rows!r}'
		)
	ordered = tuple(sorted(rows_by_key.values(), key=_row_sort_key))
	_write_run_manifest(
		manifest_path,
		config=config,
		rows=ordered,
		quarantines=quarantines,
		disk_audits=disk_audits,
	)
	return VoxelLabelBudgetSuiteResult(manifest_path, ordered, tuple(quarantines))


def classify_voxel_label_budget_job(
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	estimated_bytes: int = 0,
) -> VoxelLabelBudgetJobPlan:
	"""Classify one decoder job using the shared resumability contract.

	Specialized suites may supply a structurally compatible configuration with a
	different model matrix.  The stage-level contract is intentionally model
	agnostic: output identity, resume safety, and completed-artifact validation
	are all derived from ``job`` and its resolved configuration.
	"""
	return _classify_job(config, job, estimated_bytes=estimated_bytes)


def run_voxel_label_budget_job(
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	device: str,
	resume: Path | None,
) -> None:
	"""Run shared decoder, best-checkpoint inference, evaluation, and report."""
	_run_job(config, job, device=device, resume=resume)


def completed_voxel_label_budget_job_row(
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	action: str,
	quarantine_path: Path | None,
	error: str | None,
) -> dict[str, object]:
	"""Revalidate one completed decoder job and return its manifest row."""
	return _completed_job_row(
		config,
		job,
		action=action,
		quarantine_path=quarantine_path,
		error=error,
	)


def quarantine_voxel_label_budget_output(path: Path, *, reason: str) -> Path:
	"""Move an invalid output to a timestamped sibling without deleting it."""
	return _quarantine(path, reason=reason)


def _run_job(
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	device: str,
	resume: Path | None,
) -> None:
	job.generated_configs_dir.mkdir(parents=True, exist_ok=True)
	train_config = _decoder_config(config, job)
	_write_json(
		job.generated_configs_dir / 'decoder_config.json', train_config.to_dict()
	)
	result = run_f3_lithology_voxel_decoder(train_config, device=device, resume=resume)
	if not result.completed:
		raise RuntimeError(f'decoder did not complete: {result.latest_checkpoint}')
	latest = load_voxel_decoder_checkpoint(result.latest_checkpoint)
	if latest.get('checkpoint_kind') != 'completed':
		raise ValueError('latest.pt is not a completed checkpoint')
	best = result.best_checkpoint
	if latest.get('best_checkpoint_sha256') != file_sha256(best):
		raise ValueError('latest.pt does not bind the selected best.pt')
	inference = _inference_config(config, job, checkpoint=best)
	_write_json(
		job.generated_configs_dir / 'inference_config.json',
		_generated_inference_mapping(config, job, checkpoint=best),
	)
	predict_f3_lithology_voxels(inference, device=device)
	plan = inspect_f3_lithology_voxel_inference(inference, verify_array_hashes=True)
	if plan.checkpoint != best or plan.checkpoint.name != 'best.pt':
		raise ValueError('inference did not bind best.pt')
	evaluation = _evaluation_config(config, job)
	_write_json(
		job.generated_configs_dir / 'evaluation_config.json',
		_generated_evaluation_mapping(config, job),
	)
	evaluate_f3_lithology_voxels(evaluation)
	inspection = inspect_f3_lithology_voxel_evaluation(evaluation)
	if inspection.validation_voxel_count != int(
		job.dataset_row['validation_voxel_count']
	):
		raise ValueError('evaluation validation voxel count changed')
	report = _report_config(config, job)
	_write_json(
		job.generated_configs_dir / 'report_config.json',
		_generated_report_mapping(config, job),
	)
	build_f3_lithology_voxel_report(report)
	inspect_f3_lithology_voxel_report(report)


def _decoder_config(
	config: F3VoxelLabelBudgetSuiteConfig, job: VoxelLabelBudgetJob
) -> F3LithologyVoxelDecoderConfig:
	model = config.model_by_role[job.model_role]
	return F3LithologyVoxelDecoderConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		model={'tag': model.model_tag, 'freeze_encoder': True},
		embeddings_input_dir=model.embeddings_dir,
		voxel_dataset_input_dir=job.voxel_dataset_root,
		decoder=config.decoder,
		tiles=config.tiles,
		train=replace(config.train, seed=job.decoder_seed),
		output_dir=job.decoder_dir,
		embeddings={'spec': 'overlap_x16'},
	)


def _inference_config(
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	checkpoint: Path,
) -> F3LithologyVoxelInferenceConfig:
	model = config.model_by_role[job.model_role]
	return F3LithologyVoxelInferenceConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		model={'tag': model.model_tag, 'freeze_encoder': True},
		class_info=config.labels['class_info'],
		embeddings_input_dir=model.embeddings_dir,
		checkpoint=checkpoint,
		tiles=config.tiles,
		output_paths=f3_voxel_prediction_artifact_paths(job.prediction_dir),
		write_probabilities=False,
		overwrite=config.overwrite,
	)


def _evaluation_config(
	config: F3VoxelLabelBudgetSuiteConfig, job: VoxelLabelBudgetJob
) -> F3LithologyVoxelEvaluationConfig:
	policy = config.evaluation
	return F3LithologyVoxelEvaluationConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		prediction_input_dir=job.prediction_dir,
		voxel_dataset_input_dir=job.voxel_dataset_root,
		source_label_volume=config.labels['source_label_volume'],
		source_label_segy=config.labels['source_label_segy'],
		png_label_inventory=config.labels['png_label_inventory'],
		segy_geometry_json=config.labels['segy_geometry_json'],
		class_info=config.labels['class_info'],
		output_dir=job.evaluation_dir,
		monitored_class_ids=tuple(
			int(value)
			for value in cast('Sequence[object]', policy['monitored_class_ids'])
		),
		boundary_tolerances=tuple(
			int(value)
			for value in cast('Sequence[object]', policy['boundary_tolerances'])
		),
		boundary_region_radii=tuple(
			int(value)
			for value in cast('Sequence[object]', policy['boundary_region_radii'])
		),
		chunk_size_x=int(policy['chunk_size_x']),
		overwrite=config.overwrite,
	)


def _report_config(
	config: F3VoxelLabelBudgetSuiteConfig, job: VoxelLabelBudgetJob
) -> F3LithologyVoxelReportConfig:
	policy = config.report
	selected = cast('Mapping[str, Sequence[int]]', policy['selected_slices'])
	percentiles = cast('Sequence[float]', policy['amplitude_clip_percentiles'])
	return F3LithologyVoxelReportConfig(
		prediction_input_dir=job.prediction_dir,
		voxel_dataset_input_dir=job.voxel_dataset_root,
		evaluation_input_dir=job.evaluation_dir,
		seismic_volume=config.labels['seismic_volume'],
		label_volume=config.labels['source_label_volume'],
		class_info=config.labels['class_info'],
		png_label_inventory=config.labels['png_label_inventory'],
		segy_geometry_json=config.labels['segy_geometry_json'],
		output_dir=job.report_dir,
		dataset=config.dataset,
		selected_slices={key: tuple(values) for key, values in selected.items()},
		figure=F3LithologyVoxelFigureConfig(
			dpi=int(policy['dpi']),
			include_confidence=bool(policy['include_confidence']),
			amplitude_clip_percentiles=(float(percentiles[0]), float(percentiles[1])),
		),
		publish=F3LithologyVoxelPublishConfig(enabled=False),
		overwrite=config.overwrite,
	)


def _classify_job(  # noqa: C901, PLR0911
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	estimated_bytes: int,
) -> VoxelLabelBudgetJobPlan:
	if not job.output_root.exists():
		return VoxelLabelBudgetJobPlan(job, 'NEW', estimated_bytes=estimated_bytes)
	if not job.output_root.is_dir():
		return VoxelLabelBudgetJobPlan(
			job, 'INVALID_OR_PARTIAL', 'job output is not a directory', estimated_bytes
		)
	latest_path = job.decoder_dir / 'latest.pt'
	if not latest_path.is_file():
		return VoxelLabelBudgetJobPlan(
			job, 'INVALID_OR_PARTIAL', 'missing decoder/latest.pt', estimated_bytes
		)
	try:
		latest = load_voxel_decoder_checkpoint(latest_path)
	except Exception as error:  # noqa: BLE001 - invalid artifacts are classified.
		return VoxelLabelBudgetJobPlan(
			job,
			'INVALID_OR_PARTIAL',
			f'invalid latest.pt: {type(error).__name__}: {error}',
			estimated_bytes,
		)
	expected = _decoder_config(config, job).to_dict()
	if latest.get('resolved_config') != expected:
		return VoxelLabelBudgetJobPlan(
			job,
			'INVALID_OR_PARTIAL',
			'decoder resolved config mismatch',
			estimated_bytes,
		)
	kind = latest.get('checkpoint_kind')
	if kind != 'completed':
		extra = (job.prediction_dir, job.evaluation_dir, job.report_dir)
		if any(path.exists() for path in extra):
			return VoxelLabelBudgetJobPlan(
				job,
				'INVALID_OR_PARTIAL',
				'incomplete decoder has downstream outputs',
				estimated_bytes,
			)
		try:
			validated = validate_f3_lithology_voxel_decoder_resume(
				_decoder_config(config, job), latest_path
			)
		except Exception as error:  # noqa: BLE001 - corruption changes job state.
			return VoxelLabelBudgetJobPlan(
				job,
				'INVALID_OR_PARTIAL',
				f'invalid resume identity: {type(error).__name__}: {error}',
				estimated_bytes,
			)
		if validated.get('checkpoint_kind') != kind:
			raise AssertionError('resume validation changed checkpoint identity')
		return VoxelLabelBudgetJobPlan(
			job, 'RESUME_LATEST', estimated_bytes=estimated_bytes
		)
	try:
		_completed_job_row(
			config, job, action='REUSED', quarantine_path=None, error=None
		)
	except Exception as error:  # noqa: BLE001 - strict validation is state input.
		return VoxelLabelBudgetJobPlan(
			job,
			'INVALID_OR_PARTIAL',
			f'completed artifact validation failed: {type(error).__name__}: {error}',
			estimated_bytes,
		)
	return VoxelLabelBudgetJobPlan(job, 'REUSE_COMPLETED', estimated_bytes=0)


def _completed_job_row(  # noqa: C901, PLR0912, PLR0915
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	action: str,
	quarantine_path: Path | None,
	error: str | None,
) -> dict[str, object]:
	for name in CHECKPOINT_REQUIRED:
		if not (job.decoder_dir / name).is_file():
			raise FileNotFoundError(job.decoder_dir / name)
	resolved_config = _read_json(job.decoder_dir / 'resolved_config.json')
	if resolved_config != _decoder_config(config, job).to_dict():
		raise ValueError('completed decoder resolved config mismatch')
	latest = load_voxel_decoder_checkpoint(job.decoder_dir / 'latest.pt')
	if latest.get('checkpoint_kind') != 'completed':
		raise ValueError('completed job latest.pt kind mismatch')
	if latest.get('epoch') != config.train.epochs - 1:
		raise ValueError('completed job latest.pt epoch mismatch')
	expected_steps = config.train.epochs * cast('int', config.train.steps_per_epoch)
	if latest.get('global_step') != expected_steps:
		raise ValueError('completed job global step mismatch')
	history_rows = _read_csv(job.decoder_dir / 'history.csv')
	_validate_completed_history(
		latest,
		expected_rows=config.train.epochs,
		steps_per_epoch=cast('int', config.train.steps_per_epoch),
		csv_rows=history_rows,
		label='latest.pt',
	)
	best_path = job.decoder_dir / 'best.pt'
	if latest.get('best_checkpoint_sha256') != file_sha256(best_path):
		raise ValueError('latest.pt best checkpoint identity mismatch')
	best = load_voxel_decoder_checkpoint(best_path)
	best_state = _validate_completed_best_checkpoint(
		latest,
		best,
		resolved_config=resolved_config,
		epochs=config.train.epochs,
		steps_per_epoch=cast('int', config.train.steps_per_epoch),
	)
	_validate_generated_configs(config, job, best_path=best_path)
	run_metadata = _read_json(job.decoder_dir / 'run_metadata.json')
	for key, expected in (
		('sampling_mode', config.train.sampling_mode),
		('steps_per_epoch', config.train.steps_per_epoch),
		('train_seed', job.decoder_seed),
	):
		if run_metadata.get(key) != expected:
			raise ValueError(f'run metadata mismatch: {key}')
	train_manifest_path = job.decoder_dir / 'train_tile_manifest.json'
	validation_manifest_path = job.decoder_dir / 'validation_tile_manifest.json'
	train_manifest = read_voxel_tile_manifest(train_manifest_path)
	validation_manifest = read_voxel_tile_manifest(validation_manifest_path)
	expected_class_weights = _balanced_class_weights_from_train_manifest(
		train_manifest,
		expected_supervised_voxel_count=int(job.dataset_row['train_voxel_count']),
	)
	latest_class_weights = [
		float(value) for value in cast('Sequence[object]', latest['class_weights'])
	]
	best_class_weights = [
		float(value) for value in cast('Sequence[object]', best['class_weights'])
	]
	if (
		latest_class_weights != expected_class_weights
		or best_class_weights != expected_class_weights
	):
		raise ValueError(
			'completed checkpoint class weights do not match the shared train mask'
		)
	manifest_identities = {
		'train': train_manifest.identity_sha256,
		'validation': validation_manifest.identity_sha256,
	}
	if latest.get('tile_manifest_hashes') != manifest_identities:
		raise ValueError('completed checkpoint tile manifest identity mismatch')
	for key, expected in (
		('train_tile_manifest_sha256', train_manifest.identity_sha256),
		(
			'validation_tile_manifest_sha256',
			validation_manifest.identity_sha256,
		),
	):
		if run_metadata.get(key) != expected:
			raise ValueError(f'run metadata mismatch: {key}')
	artifacts = _mapping(latest.get('artifact_identities'), 'artifact identities')
	prediction = validate_f3_voxel_prediction_artifact(
		job.prediction_dir, mmap_mode='r'
	)
	if prediction.metadata.get('write_probabilities') is not False:
		raise ValueError('completed prediction unexpectedly wrote probabilities')
	source = _mapping(prediction.metadata.get('source_identity'), 'prediction source')
	checkpoint_identity = _mapping(
		source.get('decoder_checkpoint'), 'prediction decoder checkpoint'
	)
	if Path(str(checkpoint_identity.get('path'))).resolve() != best_path.resolve():
		raise ValueError('completed inference did not use best.pt')
	if checkpoint_identity.get('sha256') != file_sha256(best_path):
		raise ValueError('completed inference best.pt hash mismatch')
	training_sampling = {
		'sampling_mode': config.train.sampling_mode,
		'steps_per_epoch': config.train.steps_per_epoch,
		'train_seed': job.decoder_seed,
		'train_tile_manifest_sha256': train_manifest.identity_sha256,
		'validation_tile_manifest_sha256': validation_manifest.identity_sha256,
	}
	if prediction.metadata.get('training_sampling') != training_sampling:
		raise ValueError('completed prediction training-sampling contract mismatch')
	prediction_artifacts = _mapping(
		source.get('artifact_identities'), 'prediction artifact identities'
	)
	if prediction_artifacts != artifacts:
		raise ValueError('prediction/checkpoint source artifact identity mismatch')
	evaluation_config = _evaluation_config(config, job)
	evaluation = inspect_f3_lithology_voxel_evaluation(evaluation_config)
	if evaluation.validation_voxel_count != int(
		job.dataset_row['validation_voxel_count']
	):
		raise ValueError('completed evaluation validation count mismatch')
	report_inspection = inspect_f3_lithology_voxel_report(
		_report_config(config, job)
	)
	report_artifacts = _validate_completed_report(job, report_inspection)
	grid = np.load(
		job.voxel_dataset_root / 'supervision_split_grid.npy',
		mmap_mode='r',
		allow_pickle=False,
	)
	uncovered = int(
		np.count_nonzero(
			(grid == 2) & ~np.asarray(prediction.arrays.valid_mask, dtype=np.bool_)
		)
	)
	if uncovered != 0:
		raise ValueError('completed evaluation has uncovered validation voxels')
	metrics_path = job.evaluation_dir / METRICS_JSON
	boundary_path = job.evaluation_dir / BOUNDARY_METRICS_JSON
	regions_path = job.evaluation_dir / BOUNDARY_REGION_METRICS_CSV
	metrics = _read_json(metrics_path)
	metric_schema = hashlib.sha256(
		json.dumps(
			{
				'metrics': sorted(metrics),
				'boundary': sorted(_read_json(boundary_path)),
				'boundary_region_columns': list(_csv_fieldnames(regions_path)),
			},
			sort_keys=True,
			separators=(',', ':'),
		).encode('utf-8')
	).hexdigest()
	valid_identity = _mapping(artifacts.get('valid_tokens'), 'valid tokens identity')
	sampling_sequence = sampling_sequence_sha256(
		tile_count=len(train_manifest.tiles),
		batch_size=config.train.batch_size,
		steps_per_epoch=cast('int', config.train.steps_per_epoch),
		train_seed=job.decoder_seed,
		epochs=config.train.epochs,
	)
	return {
		'budget_id': job.budget_id,
		'per_class_cap': job.per_class_cap,
		'subsample_seed': job.subsample_seed,
		'decoder_seed': job.decoder_seed,
		'model_role': job.model_role,
		'model_tag': job.model_tag,
		'status': 'complete',
		'action': action,
		'voxel_dataset': _identity(
			job.voxel_dataset_root / 'supervision_split_grid.npy'
		),
		'voxel_dataset_root': str(job.voxel_dataset_root),
		'voxel_supervision_grid_sha256': job.dataset_row['supervision_split_grid'][
			'sha256'
		],  # type: ignore[index]
		'selected_token_identity_sha256': job.dataset_row[
			'selected_token_identity_sha256'
		],
		'unique_token_xyz_sha256': job.dataset_row['unique_token_xyz_sha256'],
		'train_voxel_count': int(job.dataset_row['train_voxel_count']),
		'validation_voxel_count': int(job.dataset_row['validation_voxel_count']),
		'class_order': list(job.dataset_row['class_order']),  # type: ignore[arg-type]
		'validation_mask_sha256': job.dataset_row['validation_mask_sha256'],
		'canonical_valid_token_sha256': valid_identity['sha256'],
		'source_artifact_identities': dict(artifacts),
		'class_weights': expected_class_weights,
		'initial_model_state_sha256': run_metadata['initial_model_state_sha256'],
		'decoder_architecture': dict(
			cast('Mapping[str, object]', latest['decoder_architecture'])
		),
		'sampling_mode': config.train.sampling_mode,
		'steps_per_epoch': config.train.steps_per_epoch,
		'sampling_sequence_sha256': sampling_sequence,
		'train_tile_manifest_sha256': file_sha256(train_manifest_path),
		'train_tile_identity_sha256': train_manifest.identity_sha256,
		'validation_tile_manifest_sha256': file_sha256(validation_manifest_path),
		'validation_tile_identity_sha256': validation_manifest.identity_sha256,
		'global_step': int(latest['global_step']),
		'latest_checkpoint': _identity(job.decoder_dir / 'latest.pt'),
		'best_checkpoint': _identity(best_path),
		'best_selection_epoch': int(best_state['epoch']),
		'best_selection_metrics': dict(
			cast('Mapping[str, object]', best_state['validation_metrics'])
		),
		'prediction_metadata': _identity(
			job.prediction_dir / 'prediction_metadata.json'
		),
		'prediction_checkpoint_kind': 'best',
		'evaluation_metadata': _identity(job.evaluation_dir / EVALUATION_METADATA_JSON),
		'evaluation_metrics': _identity(metrics_path),
		'evaluation_boundary_metrics': _identity(boundary_path),
		'evaluation_boundary_region_metrics': _identity(regions_path),
		'uncovered_validation_voxel_count': uncovered,
		'metric_schema_sha256': metric_schema,
		'report': _identity(job.report_dir / 'report.md'),
		'report_json': _identity(job.report_dir / 'report.json'),
		'report_artifacts': report_artifacts,
		'resolved_config': _identity(job.decoder_dir / 'resolved_config.json'),
		'generated_configs': _generated_config_identities(job),
		'error': error,
		'quarantine_path': None if quarantine_path is None else str(quarantine_path),
	}


def _validate_completed_history(  # noqa: C901
	payload: Mapping[str, object],
	*,
	expected_rows: int,
	steps_per_epoch: int,
	csv_rows: Sequence[Mapping[str, str]] | None,
	label: str,
) -> None:
	"""Validate checkpoint-internal history and its persisted CSV sequence."""
	history = payload.get('training_history')
	if not isinstance(history, Sequence) or isinstance(history, str | bytes):
		raise TypeError(f'{label} training_history must be a list')
	if len(history) != expected_rows:
		raise ValueError(f'{label} training_history row count mismatch')
	alias = payload.get('history')
	if alias is not None and alias != history:
		raise ValueError(f'{label} history alias mismatch')
	if csv_rows is not None and len(csv_rows) != expected_rows:
		raise ValueError('completed job history.csv row count mismatch')
	for index, value in enumerate(history):
		row = _mapping(value, f'{label} training_history row')
		epoch = row.get('epoch')
		global_step = row.get('global_step')
		if (
			not isinstance(epoch, int)
			or isinstance(epoch, bool)
			or epoch != index
		):
			raise ValueError(f'{label} training_history epoch sequence mismatch')
		if (
			not isinstance(global_step, int)
			or isinstance(global_step, bool)
			or global_step != (index + 1) * steps_per_epoch
		):
			raise ValueError(
				f'{label} training_history global-step sequence mismatch'
			)
		if csv_rows is None:
			continue
		csv_row = csv_rows[index]
		if set(csv_row) != set(row):
			raise ValueError('completed job history.csv column mismatch')
		for key, expected in row.items():
			if csv_row.get(key) != str(expected):
				raise ValueError(
					'completed job history.csv content mismatch: '
					f'row={index}, key={key}'
				)


def _validate_completed_best_checkpoint(
	latest: Mapping[str, object],
	best: Mapping[str, object],
	*,
	resolved_config: Mapping[str, object],
	epochs: int,
	steps_per_epoch: int,
) -> Mapping[str, object]:
	"""Validate best.pt as the exact selection snapshot named by latest.pt."""
	latest_state = _mapping(
		latest.get('best_selection_state'), 'latest best selection state'
	)
	best_state = _mapping(best.get('best_selection_state'), 'best selection state')
	if best_state != latest_state:
		raise ValueError('latest.pt and best.pt selection state mismatch')
	best_epoch = best.get('epoch')
	if best_epoch != best_state.get('epoch'):
		raise ValueError('best.pt epoch does not match best selection state')
	if (
		not isinstance(best_epoch, int)
		or isinstance(best_epoch, bool)
		or not 0 <= best_epoch < epochs
	):
		raise ValueError('best.pt epoch is outside the completed history')
	if best.get('global_step') != (best_epoch + 1) * steps_per_epoch:
		raise ValueError('best.pt global step mismatch')
	expected_kind = 'completed' if best_epoch + 1 == epochs else 'epoch'
	if best.get('checkpoint_kind') != expected_kind:
		raise ValueError('best.pt checkpoint kind mismatch')
	if best.get('resolved_config') != resolved_config:
		raise ValueError('best.pt resolved config mismatch')
	_validate_completed_history(
		best,
		expected_rows=best_epoch + 1,
		steps_per_epoch=steps_per_epoch,
		csv_rows=None,
		label='best.pt',
	)
	best_current = _mapping(best.get('current_metrics'), 'best current metrics')
	if best_current.get('validation') != best_state.get('validation_metrics'):
		raise ValueError('best.pt validation metrics do not match selection state')
	return best_state


def _validate_generated_configs(
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	best_path: Path,
) -> None:
	"""Bind generated stage snapshots to their expected resolved mappings."""
	expected = {
		'decoder_config.json': _decoder_config(config, job).to_dict(),
		'inference_config.json': _generated_inference_mapping(
			config, job, checkpoint=best_path
		),
		'evaluation_config.json': _generated_evaluation_mapping(config, job),
		'report_config.json': _generated_report_mapping(config, job),
	}
	for name, mapping in expected.items():
		if _read_json(job.generated_configs_dir / name) != mapping:
			raise ValueError(f'generated config content mismatch: {name}')


def _validate_completed_report(
	job: VoxelLabelBudgetJob,
	inspection: F3LithologyVoxelReportInspection,
) -> dict[str, Mapping[str, object]]:
	"""Reconstruct a completed report and validate its exact file inventory."""
	report_json = job.report_dir / 'report.json'
	report_markdown = job.report_dir / 'report.md'
	payload = _read_json(report_json)
	figure_paths = _validated_report_figure_paths(job.report_dir, payload)
	expected_payload = build_f3_lithology_voxel_report_payload(
		metrics=inspection.metrics,
		boundary_metrics=inspection.boundary_metrics,
		boundary_region_rows=inspection.boundary_region_rows,
		per_slice_rows=inspection.validation_slice_rows,
		prediction_metadata=inspection.prediction_artifact.metadata,
		evaluation_metadata=inspection.evaluation_metadata,
		supervision_metadata=inspection.supervision_metadata,
		figure_paths=figure_paths,
		output_dir=job.report_dir,
	)
	if payload != expected_payload:
		raise ValueError('completed report.json does not match its validated inputs')
	expected_markdown = render_f3_lithology_voxel_report_markdown(expected_payload)
	if report_markdown.read_text(encoding='utf-8') != expected_markdown:
		raise ValueError('completed report.md does not match report.json')
	paths = (report_markdown, report_json, *figure_paths)
	return {
		path.relative_to(job.report_dir).as_posix(): _identity(path) for path in paths
	}


def _validated_report_figure_paths(  # noqa: C901
	report_dir: Path, payload: Mapping[str, object]
) -> tuple[Path, ...]:
	figures = payload.get('figures')
	if not isinstance(figures, Sequence) or isinstance(figures, str | bytes):
		raise TypeError('completed report figures must be a list')
	relative_paths: list[Path] = []
	for value in figures:
		if not isinstance(value, str) or not value:
			raise TypeError('completed report figure paths must be non-empty strings')
		relative = Path(value)
		if (
			relative.is_absolute()
			or '..' in relative.parts
			or not relative.parts
			or relative.parts[0] != 'figures'
			or relative.suffix.lower() != '.png'
		):
			raise ValueError(f'invalid completed report figure path: {value!r}')
		relative_paths.append(relative)
	if len(set(relative_paths)) != len(relative_paths):
		raise ValueError('completed report contains duplicate figure paths')
	required = {
		Path('figures/confusion_matrix.png'),
		Path('figures/per_class_f1_iou.png'),
		Path('figures/boundary_f1_by_tolerance.png'),
		Path('figures/boundary_region_metrics.png'),
	}
	if not required.issubset(relative_paths):
		raise ValueError('completed report is missing aggregate figures')
	figure_paths = tuple(report_dir / relative for relative in relative_paths)
	for path in figure_paths:
		if not path.is_file() or path.stat().st_size <= 8:
			raise FileNotFoundError(f'incomplete completed report figure: {path}')
		with path.open('rb') as handle:
			if handle.read(8) != b'\x89PNG\r\n\x1a\n':
				raise ValueError(f'completed report figure is not a PNG: {path}')
	expected = {
		'report.md',
		'report.json',
		*(relative.as_posix() for relative in relative_paths),
	}
	actual = {
		path.relative_to(report_dir).as_posix()
		for path in report_dir.rglob('*')
		if path.is_file()
	}
	if actual != expected:
		raise ValueError(
			'completed report file inventory mismatch: '
			f'missing={sorted(expected - actual)!r}, '
			f'extra={sorted(actual - expected)!r}'
		)
	return figure_paths


def _balanced_class_weights_from_train_manifest(
	manifest: VoxelTileManifest,
	*,
	expected_supervised_voxel_count: int,
) -> list[float]:
	"""Recompute the exact training weights from a shared train-mask manifest."""
	if manifest.split != 'train':
		raise ValueError('class weights require a train tile manifest')
	class_counts = tuple(
		sum(
			tile.per_class_supervised_counts[str(class_id)]
			for tile in manifest.tiles
		)
		for class_id in manifest.class_ids
	)
	if sum(class_counts) != expected_supervised_voxel_count:
		raise ValueError(
			'train tile manifest count does not match the shared voxel mask'
		)
	return [
		float(value)
		for value in balanced_class_weights_from_counts(class_counts).tolist()
	]


def sampling_sequence_sha256(
	*,
	tile_count: int,
	batch_size: int,
	steps_per_epoch: int,
	train_seed: int,
	epochs: int,
) -> str:
	"""Hash the exact replacement sampler index sequence for all epochs."""
	hasher = hashlib.sha256()
	hasher.update(b'f3_voxel_replacement_sampler_v1\0')
	for epoch in range(epochs):
		generator = torch.Generator().manual_seed(train_seed + epoch)
		sampler = RandomSampler(
			range(tile_count),
			replacement=True,
			num_samples=steps_per_epoch * batch_size,
			generator=generator,
		)
		indices = np.asarray(list(sampler), dtype='<i8')
		hasher.update(epoch.to_bytes(8, 'little', signed=False))
		hasher.update(indices.tobytes())
	return hasher.hexdigest()


def _validate_triplet(rows: Sequence[Mapping[str, object]], *, context: str) -> None:
	if {str(row['model_role']) for row in rows} != set(MODEL_ORDER):
		raise ValueError(f'{context}: triplet must contain MAE, M1, and M2-A')
	keys = (
		'initial_model_state_sha256',
		'train_tile_manifest_sha256',
		'validation_tile_manifest_sha256',
		'train_tile_identity_sha256',
		'validation_tile_identity_sha256',
		'class_weights',
		'class_order',
		'decoder_architecture',
		'decoder_seed',
		'sampling_mode',
		'steps_per_epoch',
		'sampling_sequence_sha256',
	)
	if 'voxel_supervision_grid_sha256' in rows[0]:
		keys = (
			*keys,
			'voxel_supervision_grid_sha256',
			'selected_token_identity_sha256',
			'unique_token_xyz_sha256',
			'train_voxel_count',
			'validation_voxel_count',
			'validation_mask_sha256',
			'canonical_valid_token_sha256',
			'uncovered_validation_voxel_count',
			'metric_schema_sha256',
		)
	for key in keys:
		values = [row.get(key) for row in rows]
		if any(value != values[0] for value in values[1:]):
			raise ValueError(f'{context}: paired identity mismatch for {key}')


def _smoke_manifest_path(
	config: F3VoxelLabelBudgetSuiteConfig,
	*,
	budget: str,
	subsample_seed: int,
) -> Path:
	return (
		config.output_root
		/ 'smoke'
		/ f'budget={budget}'
		/ f'subsample_seed={subsample_seed}'
		/ 'smoke_manifest.json'
	)


def _smoke_model_root(
	config: F3VoxelLabelBudgetSuiteConfig,
	*,
	budget: str,
	subsample_seed: int,
	model_tag: str,
) -> Path:
	return (
		_smoke_manifest_path(
			config, budget=budget, subsample_seed=subsample_seed
		).parent
		/ f'model={model_tag}'
	)


def _smoke_jobs_from_dataset_manifest(
	config: F3VoxelLabelBudgetSuiteConfig,
	*,
	budget: str,
	subsample_seed: int,
) -> tuple[VoxelLabelBudgetJob, ...]:
	"""Resolve the smoke triplet without repeating the full dataset preflight."""
	payload = _read_json(config.dataset_manifest)
	values = payload.get('rows')
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		raise TypeError('voxel label-budget dataset manifest rows must be a list')
	matches = [
		_mapping(value, 'smoke dataset row')
		for value in values
		if isinstance(value, Mapping)
		and value.get('budget_id') == budget
		and value.get('subsample_seed') == subsample_seed
	]
	if len(matches) != 1:
		raise ValueError('smoke dataset manifest condition is missing or duplicated')
	row = matches[0]
	root = Path(str(row.get('voxel_dataset_root')))
	jobs: list[VoxelLabelBudgetJob] = []
	for role in MODEL_ORDER:
		model = config.model_by_role[role]
		jobs.append(
			VoxelLabelBudgetJob(
				budget_id=budget,
				per_class_cap=int(row['per_class_cap']),
				subsample_seed=subsample_seed,
				decoder_seed=(
					config.base_seed + subsample_seed
					if config.add_subsample_seed
					else config.base_seed
				),
				model_role=role,
				model_tag=model.model_tag,
				voxel_dataset_root=root,
				output_root=_smoke_model_root(
					config,
					budget=budget,
					subsample_seed=subsample_seed,
					model_tag=model.model_tag,
				),
				dataset_row=row,
			)
		)
	return tuple(jobs)


def _validated_smoke_row(
	config: F3VoxelLabelBudgetSuiteConfig,
	job: VoxelLabelBudgetJob,
	*,
	checkpoint: Path,
) -> dict[str, object]:
	"""Recompute one smoke row from its live checkpoint and snapshots."""
	expected_checkpoint = job.decoder_dir / 'latest.pt'
	if checkpoint.resolve(strict=False) != expected_checkpoint.resolve(strict=False):
		raise ValueError('smoke checkpoint path is not decoder/latest.pt')
	if not expected_checkpoint.is_file():
		raise FileNotFoundError(expected_checkpoint)
	train_config = _decoder_config(config, job)
	payload = validate_f3_lithology_voxel_decoder_resume(
		train_config, expected_checkpoint
	)
	if (
		payload.get('checkpoint_kind') != 'step'
		or payload.get('global_step') != 2
		or payload.get('epoch') != 0
		or payload.get('batch_index') != 1
	):
		raise ValueError('smoke latest.pt is not the expected step checkpoint')
	accumulator = _mapping(
		_mapping(payload.get('current_metrics'), 'smoke current_metrics').get(
			'train_accumulator'
		),
		'smoke train_accumulator',
	)
	if int(accumulator.get('supervised_voxel_count', 0)) <= 0:
		raise ValueError('smoke accumulated zero supervised voxels')
	for value in (
		accumulator.get('weighted_ce_sum'),
		accumulator.get('unweighted_ce_sum'),
		accumulator.get('class_weight_sum'),
	):
		if not isinstance(value, int | float) or isinstance(value, bool):
			raise TypeError('smoke accumulator values must be numeric')
		if not math.isfinite(float(value)):
			raise ValueError('smoke accumulator is non-finite')
	run_metadata_path = job.decoder_dir / 'run_metadata.json'
	train_manifest_path = job.decoder_dir / 'train_tile_manifest.json'
	validation_manifest_path = job.decoder_dir / 'validation_tile_manifest.json'
	run_metadata = _read_json(run_metadata_path)
	train_manifest = read_voxel_tile_manifest(train_manifest_path)
	validation_manifest = read_voxel_tile_manifest(validation_manifest_path)
	for key, expected in (
		('sampling_mode', config.train.sampling_mode),
		('steps_per_epoch', config.train.steps_per_epoch),
		('train_seed', job.decoder_seed),
		('train_tile_manifest_sha256', train_manifest.identity_sha256),
		(
			'validation_tile_manifest_sha256',
			validation_manifest.identity_sha256,
		),
	):
		if run_metadata.get(key) != expected:
			raise ValueError(f'smoke run metadata mismatch: {key}')
	return {
		'budget_id': job.budget_id,
		'subsample_seed': job.subsample_seed,
		'model_role': job.model_role,
		'model_tag': job.model_tag,
		'decoder_seed': job.decoder_seed,
		'global_step': 2,
		'initial_model_state_sha256': run_metadata['initial_model_state_sha256'],
		'train_tile_manifest_sha256': file_sha256(train_manifest_path),
		'validation_tile_manifest_sha256': file_sha256(validation_manifest_path),
		'train_tile_identity_sha256': train_manifest.identity_sha256,
		'validation_tile_identity_sha256': validation_manifest.identity_sha256,
		'class_weights': list(cast('Sequence[object]', payload['class_weights'])),
		'class_order': list(
			cast('Sequence[object]', job.dataset_row['class_order'])
		),
		'sampling_mode': config.train.sampling_mode,
		'steps_per_epoch': config.train.steps_per_epoch,
		'sampling_sequence_sha256': sampling_sequence_sha256(
			tile_count=len(train_manifest.tiles),
			batch_size=config.train.batch_size,
			steps_per_epoch=cast('int', config.train.steps_per_epoch),
			train_seed=job.decoder_seed,
			epochs=config.train.epochs,
		),
		'decoder_architecture': dict(
			cast('Mapping[str, object]', payload['decoder_architecture'])
		),
	}


def _validate_smoke_gate(
	config: F3VoxelLabelBudgetSuiteConfig,
) -> None:
	"""Require the canonical two-step triplet smoke before scientific jobs."""
	path = _smoke_manifest_path(
		config, budget='cap25', subsample_seed=0
	)
	payload = _read_json(path)
	if (
		payload.get('artifact_type') != 'f3_lithology_voxel_label_budget_smoke_manifest'
		or payload.get('schema_version') != 1
		or payload.get('scientific_result') is not False
	):
		raise ValueError('invalid voxel label-budget smoke manifest schema')
	smoke_dataset = _validate_identity(
		payload.get('dataset_manifest'), label='smoke dataset manifest'
	)
	if smoke_dataset.resolve() != config.dataset_manifest.resolve():
		raise ValueError('smoke manifest uses a different dataset manifest')
	expected_contract = {
		'budget_id': 'cap25',
		'subsample_seed': 0,
		'global_step': 2,
		'sampling_mode': config.train.sampling_mode,
		'steps_per_epoch': config.train.steps_per_epoch,
	}
	if payload.get('contract') != expected_contract:
		raise ValueError('smoke manifest contract mismatch')
	values = payload.get('rows')
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		raise TypeError('smoke manifest rows must be a list')
	rows = tuple(_mapping(value, 'smoke row') for value in values)
	if len(rows) != 3:
		raise ValueError('smoke manifest must contain exactly three rows')
	jobs = {
		job.model_role: job
		for job in _smoke_jobs_from_dataset_manifest(
			config, budget='cap25', subsample_seed=0
		)
	}
	seen: set[str] = set()
	for row in rows:
		role = str(row.get('model_role'))
		if role not in jobs or role in seen:
			raise ValueError(
				'smoke manifest contains an unknown or duplicate model role'
			)
		seen.add(role)
		job = jobs[role]
		actual = _validated_smoke_row(
			config, job, checkpoint=job.decoder_dir / 'latest.pt'
		)
		if dict(row) != actual:
			different = sorted(
				key
				for key in set(row) | set(actual)
				if row.get(key) != actual.get(key)
			)
			raise ValueError(
				f'smoke manifest row differs from live artifacts: {different!r}'
			)
	_validate_triplet(rows, context='smoke gate')


def _full_reference_contract(  # noqa: C901, PLR0912, PLR0915
	config: F3VoxelLabelBudgetSuiteConfig,
) -> tuple[int, str, str, int]:
	dataset_manifest = _read_json(config.dataset_manifest)
	sources = _mapping(dataset_manifest.get('sources'), 'dataset manifest sources')
	common_metadata = _mapping(
		sources.get('common_metadata'), 'dataset manifest common metadata'
	)
	common_voxel_root = Path(str(common_metadata.get('path'))).parent
	expected_architecture = voxel_decoder_architecture_mapping(
		spec=config.decoder.spec,
		embedding_dim=config.decoder.embedding_dim,
		class_count=config.decoder.class_count,
		hidden_channels=config.decoder.hidden_channels,
		upsample_factors=config.decoder.upsample_factors,
		upsample_mode=config.decoder.upsample_mode,
		normalization=config.decoder.normalization,
	)
	expected_tiles = {
		'core_size_tokens': list(config.tiles.core_size_tokens),
		'context_halo_tokens': list(config.tiles.context_halo_tokens),
	}
	expected_train = {
		'epochs': 50,
		'batch_size': 1,
		'learning_rate': 0.001,
		'weight_decay': 0.0001,
		'class_weight': 'balanced',
		'seed': 42,
		'num_workers': 0,
		'amp': True,
		'gradient_clip_norm': 1.0,
	}
	train_hashes: set[str] = set()
	validation_hashes: set[str] = set()
	train_identities: set[str] = set()
	validation_identities: set[str] = set()
	counts: set[int] = set()
	sizes: list[int] = []
	for role in MODEL_ORDER:
		root = config.full_label_decoder_runs[role]
		for name in (
			'latest.pt',
			'best.pt',
			'history.csv',
			'resolved_config.json',
			'train_tile_manifest.json',
			'validation_tile_manifest.json',
		):
			if not (root / name).is_file():
				raise FileNotFoundError(
					f'missing full-label reference file: {root / name}'
				)
		latest = load_voxel_decoder_checkpoint(root / 'latest.pt')
		if latest.get('checkpoint_kind') != 'completed' or latest.get('epoch') != 49:
			raise ValueError(f'full-label {role} reference is not completed epoch 49')
		resolved = _read_json(root / 'resolved_config.json')
		if latest.get('resolved_config') != resolved:
			raise ValueError(f'full-label {role} checkpoint/config mismatch')
		model = _mapping(resolved.get('model'), f'full-label {role} model')
		if model != {
			'tag': config.model_by_role[role].model_tag,
			'freeze_encoder': True,
		}:
			raise ValueError(f'full-label {role} model identity mismatch')
		embeddings = _mapping(
			resolved.get('embeddings'), f'full-label {role} embeddings'
		)
		if embeddings != {
			'spec': 'overlap_x16',
			'input_dir': str(config.model_by_role[role].embeddings_dir),
		}:
			raise ValueError(f'full-label {role} embedding identity mismatch')
		if resolved.get('dataset') != dict(config.dataset):
			raise ValueError(f'full-label {role} dataset identity mismatch')
		if resolved.get('decoder') != expected_architecture:
			raise ValueError(f'full-label {role} decoder architecture mismatch')
		if resolved.get('tiles') != expected_tiles:
			raise ValueError(f'full-label {role} tile geometry mismatch')
		if resolved.get('train') != expected_train:
			raise ValueError(f'full-label {role} training contract mismatch')
		voxel_dataset = _mapping(
			resolved.get('voxel_dataset'), f'full-label {role} voxel dataset'
		)
		if Path(str(voxel_dataset.get('input_dir'))).resolve() != (
			common_voxel_root.resolve()
		):
			raise ValueError(f'full-label {role} uses the wrong voxel dataset')
		if len(_read_csv(root / 'history.csv')) != 50:
			raise ValueError(f'full-label {role} history does not contain 50 epochs')
		train_path = root / 'train_tile_manifest.json'
		validation_path = root / 'validation_tile_manifest.json'
		train = read_voxel_tile_manifest(train_path)
		validation = read_voxel_tile_manifest(validation_path)
		if latest.get('global_step') != 50 * len(train.tiles):
			raise ValueError(f'full-label {role} global step/tile count mismatch')
		best_path = root / 'best.pt'
		if latest.get('best_checkpoint_sha256') != file_sha256(best_path):
			raise ValueError(f'full-label {role} best checkpoint binding mismatch')
		best = load_voxel_decoder_checkpoint(best_path)
		if best.get('resolved_config') != resolved:
			raise ValueError(f'full-label {role} best checkpoint config mismatch')
		best_state = _mapping(
			best.get('best_selection_state'), f'full-label {role} best state'
		)
		if best.get('epoch') != best_state.get('epoch'):
			raise ValueError(f'full-label {role} best-selection epoch mismatch')
		counts.add(len(train.tiles))
		train_hashes.add(file_sha256(train_path))
		validation_hashes.add(file_sha256(validation_path))
		train_identities.add(train.identity_sha256)
		validation_identities.add(validation.identity_sha256)
		sizes.append(_tree_size(root.parent.parent / 'voxel_predictions'))
		sizes.append(_tree_size(root))
	if config.require_shared_train_tile_identity and any(
		len(values) != 1
		for values in (
			counts,
			train_hashes,
			validation_hashes,
			train_identities,
			validation_identities,
		)
	):
		raise ValueError('full-label reference tile identity differs across models')
	if len(counts) != 1:
		raise ValueError('full-label train tile count differs across models')
	# Existing decoder and downstream footprint, with a conservative 20% margin.
	reference_bytes = max(1, int((sum(sizes) / max(1, len(sizes))) * 1.2))
	return (
		next(iter(counts)),
		next(iter(train_hashes)),
		next(iter(validation_hashes)),
		reference_bytes,
	)


def _dataset_rows(  # noqa: C901, PLR0912, PLR0915
	config: F3VoxelLabelBudgetSuiteConfig,
) -> dict[tuple[str, int], Mapping[str, object]]:
	path = config.dataset_manifest
	if path.name != DATASET_MANIFEST_NAME or not path.is_file():
		raise FileNotFoundError(path)
	if path.parent.resolve() != config.output_root.resolve():
		raise ValueError('voxel label-budget dataset manifest/output root mismatch')
	payload = _read_json(path)
	expected_top_level = {
		'artifact_type',
		'schema_version',
		'suite',
		'contract',
		'models',
		'sources',
		'common_validation_mask_sha256',
		'condition_count',
		'rows',
	}
	if set(payload) != expected_top_level:
		raise ValueError('voxel label-budget dataset manifest field inventory mismatch')
	if (
		payload.get('artifact_type') != MANIFEST_ARTIFACT_TYPE
		or payload.get('schema_version') != DATASET_SCHEMA_VERSION
	):
		raise ValueError('invalid voxel label-budget dataset manifest schema')
	expected_suite = {
		'name': DATASET_SUITE_NAME,
		'output_root': str(config.output_root),
		'budget_semantics': 'per_class_selected_token_row_cap',
	}
	if payload.get('suite') != expected_suite:
		raise ValueError('voxel label-budget dataset suite contract mismatch')
	expected_contract = {
		'budgets': list(config.budgets),
		'subsample_seeds': list(config.subsample_seeds),
		'patch_size_xyz': list(_decoder_patch_size(config)),
		'require_all_classes': True,
		'validation': 'canonical_full_validation_bitwise',
	}
	if payload.get('contract') != expected_contract:
		raise ValueError('voxel label-budget dataset scientific contract mismatch')
	expected_models = {
		role: config.model_by_role[role].model_tag for role in MODEL_ORDER
	}
	if payload.get('models') != expected_models:
		raise ValueError('voxel label-budget dataset model contract mismatch')
	sources = _mapping(payload.get('sources'), 'dataset manifest sources')
	if set(sources) != DATASET_SOURCE_KEYS:
		raise ValueError('voxel label-budget dataset source inventory mismatch')
	for label, identity in sources.items():
		_validate_identity(identity, label=f'dataset source {label}')
	common_validation_hash = payload.get('common_validation_mask_sha256')
	if not _is_sha256(common_validation_hash):
		raise ValueError('dataset common validation mask SHA-256 is invalid')
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError('voxel label-budget dataset manifest rows must be a list')
	if payload.get('condition_count') != len(rows):
		raise ValueError('voxel label-budget dataset condition_count mismatch')
	result: dict[tuple[str, int], Mapping[str, object]] = {}
	for value in rows:
		if not isinstance(value, Mapping):
			raise TypeError('voxel label-budget dataset row must be a mapping')
		key = (str(value.get('budget_id')), int(value.get('subsample_seed', -1)))
		if key in result:
			raise ValueError(f'duplicate voxel label-budget dataset row: {key!r}')
		root = Path(str(value.get('voxel_dataset_root')))
		expected_root = (
			config.output_root
			/ 'datasets'
			/ f'budget={key[0]}'
			/ f'subsample_seed={key[1]}'
			/ 'voxel_supervision'
		)
		if root.resolve() != expected_root.resolve():
			raise ValueError('voxel dataset row root/path mismatch')
		if set(DATASET_ROW_FILE_NAMES.values()) != set(REQUIRED_CONDITION_FILES):
			raise AssertionError('dataset row file mapping is incomplete')
		for identity_key, name in DATASET_ROW_FILE_NAMES.items():
			_validate_identity_at(
				value.get(identity_key),
				root / name,
				label=f'dataset {identity_key}',
			)
		metadata = validate_voxel_label_budget_condition_artifact(root)
		_validate_dataset_condition_metadata(
			config,
			row=value,
			metadata=metadata,
			sources=sources,
			common_validation_hash=cast('str', common_validation_hash),
		)
		result[key] = value
	expected = {
		(budget, seed) for budget in config.budgets for seed in config.subsample_seeds
	}
	if set(result) != expected:
		missing = sorted(expected - set(result))
		extra = sorted(set(result) - expected)
		raise ValueError(
			'voxel label-budget dataset matrix mismatch; '
			f'missing={missing!r}, extra={extra!r}'
		)
	return result


def _decoder_patch_size(
	config: F3VoxelLabelBudgetSuiteConfig,
) -> tuple[int, int, int]:
	factors = config.decoder.upsample_factors
	return tuple(
		math.prod(int(stage[axis]) for stage in factors) for axis in range(3)
	)  # type: ignore[return-value]


def _validate_dataset_condition_metadata(  # noqa: C901
	config: F3VoxelLabelBudgetSuiteConfig,
	*,
	row: Mapping[str, object],
	metadata: Mapping[str, object],
	sources: Mapping[str, object],
	common_validation_hash: str,
) -> None:
	expected_suite = {
		'name': DATASET_SUITE_NAME,
		'output_root': str(config.output_root),
	}
	if metadata.get('suite') != expected_suite:
		raise ValueError('voxel label-budget condition suite mismatch')
	identity = _mapping(metadata.get('identity'), 'dataset condition identity')
	metadata_sources = _mapping(
		metadata.get('sources'), 'dataset condition sources'
	)
	if set(metadata_sources) != DATASET_SOURCE_KEYS | {'selected_token_artifacts'}:
		raise ValueError('voxel label-budget condition source inventory mismatch')
	if {key: metadata_sources[key] for key in DATASET_SOURCE_KEYS} != dict(sources):
		raise ValueError('voxel label-budget condition common sources mismatch')
	selected_sources = _mapping(
		metadata_sources.get('selected_token_artifacts'),
		'dataset selected token artifacts',
	)
	if set(selected_sources) != {
		'mae_m1_mae',
		'mae_m1_m1',
		'm1_m2a_m1',
		'm1_m2a_m2a',
	}:
		raise ValueError('voxel label-budget selected source inventory mismatch')
	checks = (
		('budget_id', 'budget_id'),
		('per_class_cap', 'per_class_cap'),
		('subsample_seed', 'subsample_seed'),
		('train_voxel_count', 'actual_train_voxel_count'),
		('validation_voxel_count', 'validation_voxel_count'),
		('class_order', 'class_order'),
		('per_class_train_voxel_counts', 'per_class_train_voxel_counts'),
		('per_class_validation_voxel_counts', 'per_class_validation_voxel_counts'),
		('selected_token_row_count', 'selected_token_row_count'),
		('unique_selected_token_xyz_count', 'unique_selected_token_xyz_count'),
		('duplicate_selected_row_count', 'duplicate_selected_row_count'),
		('selected_token_identity_sha256', 'selected_token_identity_sha256'),
		('unique_token_xyz_sha256', 'unique_token_xyz_sha256'),
		('train_mask_sha256', 'train_mask_sha256'),
		('validation_mask_sha256', 'validation_mask_sha256'),
	)
	for row_key, identity_key in checks:
		if row.get(row_key) != identity.get(identity_key):
			raise ValueError(f'voxel label-budget dataset row mismatch: {row_key}')
	if identity.get('patch_size_xyz') != list(_decoder_patch_size(config)):
		raise ValueError('voxel label-budget condition patch size mismatch')
	if row.get('per_class_cap') != int(str(row.get('budget_id'))[3:]):
		raise ValueError('voxel label-budget condition cap mismatch')
	if row.get('class_order') != list(range(config.decoder.class_count)):
		raise ValueError('voxel label-budget condition class order mismatch')
	if (
		row.get('validation_mask_sha256') != common_validation_hash
		or identity.get('validation_mask_sha256') != common_validation_hash
	):
		raise ValueError('voxel label-budget common validation identity mismatch')


def _is_sha256(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _validate_model_embeddings(config: F3VoxelLabelBudgetSuiteConfig) -> None:
	valid_hashes: set[str] = set()
	for model in config.models:
		files = output_paths(model.embeddings_dir, config.dataset['name'])
		for path in (files.embeddings, files.valid_tokens, files.metadata):
			if not path.is_file():
				raise FileNotFoundError(path)
		metadata = _read_json(files.metadata)
		checkpoint = metadata.get('checkpoint_path')
		if (
			not isinstance(checkpoint, str)
			or model.model_tag not in Path(checkpoint).parts
		):
			raise ValueError(
				f'{model.role} embedding checkpoint/model identity mismatch'
			)
		checkpoint_path = Path(checkpoint)
		if not checkpoint_path.is_file() or metadata.get(
			'checkpoint_sha256'
		) != file_sha256(checkpoint_path):
			raise ValueError(f'{model.role} embedding checkpoint hash mismatch')
		if model.role == 'mae' and checkpoint_path.name != 'mae_latest.pt':
			raise ValueError('MAE embedding provenance must use mae_latest.pt')
		valid_hashes.add(file_sha256(files.valid_tokens))
	if len(valid_hashes) != 1:
		raise ValueError('three model embeddings do not share canonical valid tokens')


def _jobs(
	config: F3VoxelLabelBudgetSuiteConfig,
	*,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[VoxelLabelBudgetJob, ...]:
	jobs = []
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			row = dataset_rows[(budget, seed)]
			for role in MODEL_ORDER:
				model = config.model_by_role[role]
				jobs.append(
					VoxelLabelBudgetJob(
						budget_id=budget,
						per_class_cap=int(row['per_class_cap']),
						subsample_seed=seed,
						decoder_seed=(
							config.base_seed + seed
							if config.add_subsample_seed
							else config.base_seed
						),
						model_role=role,
						model_tag=model.model_tag,
						voxel_dataset_root=Path(str(row['voxel_dataset_root'])),
						output_root=(
							config.output_root
							/ 'jobs'
							/ f'budget={budget}'
							/ f'subsample_seed={seed}'
							/ f'model={model.model_tag}'
						),
						dataset_row=row,
					)
				)
	return tuple(jobs)


def _generated_inference_mapping(
	config: F3VoxelLabelBudgetSuiteConfig, job: VoxelLabelBudgetJob, *, checkpoint: Path
) -> dict[str, object]:
	return {
		'model_tag': job.model_tag,
		'embeddings_dir': str(config.model_by_role[job.model_role].embeddings_dir),
		'checkpoint': str(checkpoint),
		'checkpoint_sha256': file_sha256(checkpoint),
		'write_probabilities': False,
		'output_dir': str(job.prediction_dir),
	}


def _generated_evaluation_mapping(
	config: F3VoxelLabelBudgetSuiteConfig, job: VoxelLabelBudgetJob
) -> dict[str, object]:
	return {
		'prediction_dir': str(job.prediction_dir),
		'voxel_dataset_root': str(job.voxel_dataset_root),
		'policy': dict(config.evaluation),
		'output_dir': str(job.evaluation_dir),
	}


def _generated_report_mapping(
	config: F3VoxelLabelBudgetSuiteConfig, job: VoxelLabelBudgetJob
) -> dict[str, object]:
	return {
		'prediction_dir': str(job.prediction_dir),
		'evaluation_dir': str(job.evaluation_dir),
		'voxel_dataset_root': str(job.voxel_dataset_root),
		'report': dict(config.report),
		'publish_enabled': False,
		'output_dir': str(job.report_dir),
	}


def _generated_config_identities(
	job: VoxelLabelBudgetJob,
) -> Mapping[str, Mapping[str, object]]:
	result: dict[str, Mapping[str, object]] = {}
	for name in (
		'decoder_config.json',
		'inference_config.json',
		'evaluation_config.json',
		'report_config.json',
	):
		result[name] = _identity(job.generated_configs_dir / name)
	return result


def _write_run_manifest(
	path: Path,
	*,
	config: F3VoxelLabelBudgetSuiteConfig,
	rows: Sequence[Mapping[str, object]],
	quarantines: Sequence[Path],
	disk_audits: Sequence[Mapping[str, object]],
) -> None:
	_write_json(
		path,
		{
			'artifact_type': RUN_MANIFEST_TYPE,
			'schema_version': RUN_SCHEMA_VERSION,
			'preregistered_contract': _run_manifest_contract(config),
			'dataset_manifest': _identity(config.dataset_manifest),
			'row_count': len(rows),
			'complete_count': sum(row.get('status') == 'complete' for row in rows),
			'rows': list(rows),
			'quarantines': [str(path) for path in quarantines],
			'disk_audits': list(disk_audits),
		},
	)


def _run_manifest_contract(
	config: F3VoxelLabelBudgetSuiteConfig,
) -> Mapping[str, object]:
	return {
		'budgets': list(config.budgets),
		'subsample_seeds': list(config.subsample_seeds),
		'model_order': list(MODEL_ORDER),
		'epochs': config.train.epochs,
		'sampling_mode': config.train.sampling_mode,
		'steps_per_epoch': config.train.steps_per_epoch,
		'seed_policy': {
			'base_seed': config.base_seed,
			'add_subsample_seed': config.add_subsample_seed,
		},
	}


def _prior_run_state(  # noqa: C901, PLR0912
	path: Path,
	*,
	config: F3VoxelLabelBudgetSuiteConfig,
) -> tuple[
	tuple[Mapping[str, object], ...],
	tuple[Path, ...],
	tuple[Mapping[str, object], ...],
]:
	if not path.is_file():
		return (), (), ()
	payload = _read_json(path)
	if (
		payload.get('artifact_type') != RUN_MANIFEST_TYPE
		or payload.get('schema_version') != RUN_SCHEMA_VERSION
	):
		raise ValueError('existing run manifest schema mismatch')
	if payload.get('preregistered_contract') != _run_manifest_contract(config):
		raise ValueError('existing run manifest preregistered contract mismatch')
	dataset_path = _validate_identity(
		payload.get('dataset_manifest'), label='existing run dataset manifest'
	)
	if dataset_path.resolve() != config.dataset_manifest.resolve():
		raise ValueError('existing run manifest uses a different dataset manifest')
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError('existing run manifest rows must be a list')
	resolved_rows = tuple(_mapping(row, 'existing run row') for row in rows)
	if payload.get('row_count') != len(resolved_rows):
		raise ValueError('existing run manifest row_count mismatch')
	if payload.get('complete_count') != sum(
		row.get('status') == 'complete' for row in resolved_rows
	):
		raise ValueError('existing run manifest complete_count mismatch')
	keys = [_row_key(row) for row in resolved_rows]
	if len(set(keys)) != len(keys):
		raise ValueError('existing run manifest contains duplicate job rows')
	allowed = {
		(budget, seed, role)
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in MODEL_ORDER
	}
	if any(key not in allowed for key in keys):
		raise ValueError('existing run manifest contains an unknown job row')
	for row in resolved_rows:
		role = str(row.get('model_role'))
		if row.get('model_tag') != config.model_by_role[role].model_tag:
			raise ValueError('existing run manifest model identity mismatch')
	quarantines = payload.get('quarantines', [])
	if not isinstance(quarantines, Sequence) or isinstance(quarantines, str | bytes):
		raise TypeError('existing run manifest quarantines must be a list')
	disk_audits = payload.get('disk_audits', [])
	if not isinstance(disk_audits, Sequence) or isinstance(disk_audits, str | bytes):
		raise TypeError('existing run manifest disk_audits must be a list')
	return (
		resolved_rows,
		tuple(Path(str(item)) for item in quarantines),
		tuple(_mapping(item, 'existing disk audit') for item in disk_audits),
	)


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


def _disk_audit(root: Path, label: str) -> Mapping[str, object]:
	usage = shutil.disk_usage(root.parent)
	return {
		'label': label,
		'timestamp_utc': datetime.now(timezone.utc).isoformat(),
		'total_bytes': usage.total,
		'used_bytes': usage.used,
		'free_bytes': usage.free,
	}


def _tree_size(path: Path) -> int:
	if not path.exists():
		return 0
	return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def _quarantine(path: Path, *, reason: str) -> Path:
	stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
	safe = ''.join(character if character.isalnum() else '_' for character in reason)[
		:80
	]
	candidate = path.with_name(f'{path.name}.quarantine_{stamp}_{safe}')
	counter = 1
	while candidate.exists():
		candidate = path.with_name(f'{path.name}.quarantine_{stamp}_{safe}_{counter}')
		counter += 1
	path.replace(candidate)
	return candidate


def _identity(path: Path) -> dict[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _validate_identity(value: object, *, label: str) -> Path:
	identity = _mapping(value, label)
	path = Path(str(identity.get('path')))
	if not path.is_file():
		raise FileNotFoundError(path)
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} hash mismatch')
	if 'byte_size' in identity and identity.get('byte_size') != path.stat().st_size:
		raise ValueError(f'{label} byte-size mismatch')
	return path


def _validate_identity_at(value: object, path: Path, *, label: str) -> None:
	actual = _validate_identity(value, label=label)
	if actual.resolve() != path.resolve():
		raise ValueError(f'{label} path mismatch')


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON document must contain an object: {path}')
	return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.tmp')
	temporary.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(encoding='utf-8', newline='') as file_obj:
		return list(csv.DictReader(file_obj))


def _csv_fieldnames(path: Path) -> tuple[str, ...]:
	with path.open(encoding='utf-8', newline='') as file_obj:
		return tuple(csv.DictReader(file_obj).fieldnames or ())


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _job_key(job: VoxelLabelBudgetJob) -> tuple[str, int, str]:
	return (job.budget_id, job.subsample_seed, job.model_role)


def _row_key(row: Mapping[str, object]) -> tuple[str, int, str]:
	return (
		str(row.get('budget_id')),
		int(row.get('subsample_seed', -1)),
		str(row.get('model_role')),
	)


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, int, int]:
	budget = str(row.get('budget_id'))
	return (
		int(budget[3:]) if budget.startswith('cap') else 10**9,
		int(row.get('subsample_seed', -1)),
		MODEL_ORDER.index(str(row.get('model_role')))
		if str(row.get('model_role')) in MODEL_ORDER
		else 99,
	)


__all__ = [
	'JOB_STATES',
	'RUN_MANIFEST_NAME',
	'RUN_MANIFEST_TYPE',
	'VoxelLabelBudgetJob',
	'VoxelLabelBudgetJobPlan',
	'VoxelLabelBudgetSuiteInspection',
	'VoxelLabelBudgetSuiteResult',
	'classify_voxel_label_budget_job',
	'completed_voxel_label_budget_job_row',
	'inspect_f3_lithology_voxel_label_budget_suite',
	'quarantine_voxel_label_budget_output',
	'run_f3_lithology_voxel_label_budget_smoke',
	'run_f3_lithology_voxel_label_budget_suite',
	'run_voxel_label_budget_job',
	'sampling_sequence_sha256',
]
