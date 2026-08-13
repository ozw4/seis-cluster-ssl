"""Isolated runner for the center-trace masked original-split screen."""
# ruff: noqa: C901, PLR0912, PLR0913, PLR0915, SLF001

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_center_trace_masked as center_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import (
	center_trace_masked_pretraining_validation as center_validation,
)
from seis_ssl_cluster.f3.lithology import voxel_label_budget_control as control
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as shared
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	VoxelLabelBudgetJobPlan,
	classify_voxel_label_budget_job,
	quarantine_voxel_label_budget_output,
	run_voxel_label_budget_job,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
		F3VoxelLabelBudgetReferenceInspection,
	)


@dataclass(frozen=True)
class F3VoxelLabelBudgetCenterTraceMaskedInspection:
	"""Read-only 15-job plan and live candidate/reference identities."""

	jobs: tuple[VoxelLabelBudgetJob, ...]
	plans: tuple[VoxelLabelBudgetJobPlan, ...]
	historical_reference: F3VoxelLabelBudgetReferenceInspection
	hard_reference_rows: tuple[Mapping[str, object], ...]
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]]
	current_rows: Mapping[tuple[str, int], Mapping[str, object]]
	candidate_identities: Mapping[str, Mapping[str, object]]
	estimated_new_bytes: int
	disk_free_bytes: int


@dataclass(frozen=True)
class F3VoxelLabelBudgetCenterTraceMaskedRunResult:
	"""Persisted candidate manifest state."""

	manifest_json: Path
	rows: tuple[Mapping[str, object], ...]
	quarantines: tuple[Path, ...]


def center_trace_masked_run_manifest_path(config: object) -> Path:
	"""Return the center-trace candidate-owned manifest path."""
	return config.reports_dir / config.run_manifest_name


def inspect_f3_lithology_voxel_label_budget_center_trace_masked(
	config: object,
	*,
	candidate: str | None = None,
	budget: str | None = None,
	subsample_seed: int | None = None,
) -> F3VoxelLabelBudgetCenterTraceMaskedInspection:
	"""Validate references and classify exactly the selected candidate jobs."""
	_validate_screening_audit(config)
	dataset_rows = shared._dataset_rows(config)
	historical_reference = shared._mae_reference(config, dataset_rows)
	current_rows = shared._current_k6_rows(config, dataset_rows)
	for key, row in current_rows.items():
		control._validate_paired_identity(
			row,
			reference=historical_reference,
			dataset_row=dataset_rows[key],
			reference_roles=(config.references.mae_model_id,),
		)
	hard_rows = _hard_reference_rows(
		config,
		dataset_rows=dataset_rows,
		current_rows=current_rows,
		historical_reference=historical_reference,
	)
	canonical_valid_tokens_sha256 = shared._canonical_valid_tokens_sha256(current_rows)
	identities = {
		item.model_id: _candidate_identity(
			config,
			item,
			canonical_valid_tokens_sha256=canonical_valid_tokens_sha256,
		)
		for item in config.candidates
	}
	jobs = shared._jobs(config, dataset_rows)
	jobs = tuple(
		job
		for job in jobs
		if (candidate is None or job.model_role == candidate)
		and (budget is None or job.budget_id == budget)
		and (subsample_seed is None or job.subsample_seed == subsample_seed)
	)
	if not jobs:
		raise ValueError('job filters selected no center-trace masked jobs')
	bytes_per_job = shared._estimated_candidate_job_bytes(current_rows)
	plans = tuple(
		_classify_center_job(config, job, estimated_bytes=bytes_per_job) for job in jobs
	)
	disk = shutil.disk_usage(shared._existing_parent(config.output_root.parent))
	return F3VoxelLabelBudgetCenterTraceMaskedInspection(
		jobs=jobs,
		plans=plans,
		historical_reference=historical_reference,
		hard_reference_rows=hard_rows,
		dataset_rows=dataset_rows,
		current_rows=current_rows,
		candidate_identities=identities,
		estimated_new_bytes=sum(item.estimated_bytes for item in plans),
		disk_free_bytes=disk.free,
	)


def run_f3_lithology_voxel_label_budget_center_trace_masked(
	config: object,
	*,
	only_missing: bool = False,
	resume: bool = False,
	device: str = 'auto',
	candidate: str | None = None,
	budget: str | None = None,
	subsample_seed: int | None = None,
) -> F3VoxelLabelBudgetCenterTraceMaskedRunResult:
	"""Run, reuse, resume, or quarantine only the 15 owned jobs."""
	if only_missing and resume:
		raise ValueError('--only-missing and --resume are mutually exclusive')
	inspection = inspect_f3_lithology_voxel_label_budget_center_trace_masked(
		config,
		candidate=candidate,
		budget=budget,
		subsample_seed=subsample_seed,
	)
	if inspection.estimated_new_bytes > inspection.disk_free_bytes:
		raise RuntimeError('insufficient disk for planned center-trace masked jobs')
	if resume and any(
		plan.state != 'RESUME_SAME_IDENTITY' for plan in inspection.plans
	):
		raise ValueError(
			'--resume requires an incomplete valid same-identity latest.pt '
			'for every selected job'
		)
	if (
		not only_missing
		and not resume
		and any(plan.state != 'NEW' for plan in inspection.plans)
	):
		raise FileExistsError('non-new job requires --only-missing or --resume')
	path = center_trace_masked_run_manifest_path(config)
	prior_rows_by_key, quarantines = shared._prior_rows(
		path, config, inspection.candidate_identities
	)
	rows_by_key = dict(prior_rows_by_key)
	dataset_rows = inspection.dataset_rows
	current_rows = inspection.current_rows
	for plan in inspection.plans:
		job = plan.job
		key = shared._job_key(job)
		prior_row = prior_rows_by_key.get(key)
		quarantine: Path | None = None
		state = plan.state
		if state == 'INVALID_OR_PARTIAL':
			if not only_missing:
				raise FileExistsError(
					f'invalid/partial job requires --only-missing: {job.output_root}'
				)
			quarantine = quarantine_voxel_label_budget_output(
				job.output_root, reason=plan.reason or 'invalid_or_partial'
			)
			quarantines.append(quarantine)
			state = 'NEW'
		stage = shared._stage_config(config, job.model_role)
		identity = inspection.candidate_identities[job.model_role]
		try:
			if state == 'REUSE_COMPLETED':
				actual_row = control._completed_control_row(
					stage,
					stage,
					job,
					candidate_embedding_identity=identity,
					action='REUSED',
					quarantine_path=None,
					error=None,
				)
				row = (
					prior_row
					if shared._same_completed_row_except_action(prior_row, actual_row)
					else actual_row
				)
			else:
				checkpoint = (
					job.decoder_dir / 'latest.pt'
					if state == 'RESUME_SAME_IDENTITY'
					else None
				)
				run_voxel_label_budget_job(
					stage,
					job,
					device=device,
					resume=checkpoint,
				)
				row = control._completed_control_row(
					stage,
					stage,
					job,
					candidate_embedding_identity=identity,
					action='RESUMED' if checkpoint else 'NEW',
					quarantine_path=quarantine,
					error=None,
				)
			_validate_candidate_pairing(
				row,
				current_reference=current_rows[(job.budget_id, job.subsample_seed)],
				historical_reference=inspection.historical_reference,
				dataset_row=dataset_rows[(job.budget_id, job.subsample_seed)],
			)
		except BaseException as error:
			rows_by_key[key] = _failed_row(job, error)
			shared._write_manifest(
				path,
				config,
				tuple(rows_by_key.values()),
				quarantines,
				inspection.candidate_identities,
			)
			raise
		rows_by_key[key] = row
		if prior_row != row:
			shared._write_manifest(
				path,
				config,
				tuple(rows_by_key.values()),
				quarantines,
				inspection.candidate_identities,
			)
	ordered = tuple(sorted(rows_by_key.values(), key=shared._row_sort_key))
	if not path.is_file():
		shared._write_manifest(
			path,
			config,
			ordered,
			quarantines,
			inspection.candidate_identities,
		)
	return F3VoxelLabelBudgetCenterTraceMaskedRunResult(
		path, ordered, tuple(quarantines)
	)


def load_f3_lithology_voxel_label_budget_center_trace_masked_rows(
	config: object,
	*,
	inspection: F3VoxelLabelBudgetCenterTraceMaskedInspection | None = None,
) -> tuple[Mapping[str, object], ...]:
	"""Revalidate all 15 completed candidate rows without running jobs."""
	if inspection is None:
		inspection = inspect_f3_lithology_voxel_label_budget_center_trace_masked(config)
	path = center_trace_masked_run_manifest_path(config)
	payload = shared._read_json(path)
	shared._validate_manifest(payload, config, inspection.candidate_identities)
	rows = tuple(payload.get('rows', ()))
	if len(rows) != config.job_count:
		raise ValueError(
			f'center-trace masked manifest must own exactly {config.job_count} rows'
		)
	by_job = {shared._job_key(job): job for job in inspection.jobs}
	dataset_rows = inspection.dataset_rows
	current_rows = inspection.current_rows
	if {shared._row_key(row) for row in rows} != set(by_job):
		raise ValueError('center-trace masked manifest job matrix mismatch')
	for row in rows:
		if row.get('status') != 'complete':
			raise ValueError(
				f'center-trace masked job is not complete: {shared._row_key(row)!r}'
			)
		job = by_job[shared._row_key(row)]
		stage = shared._stage_config(config, job.model_role)
		quarantine_value = row.get('quarantine_path')
		actual = control._completed_control_row(
			stage,
			stage,
			job,
			candidate_embedding_identity=inspection.candidate_identities[
				job.model_role
			],
			action=str(row.get('action')),
			quarantine_path=(
				None if quarantine_value is None else Path(str(quarantine_value))
			),
			error=None if row.get('error') is None else str(row.get('error')),
		)
		if dict(row) != actual:
			raise ValueError(
				'center-trace masked completed row differs from live artifacts: '
				f'{shared._row_key(row)!r}'
			)
		_validate_candidate_pairing(
			row,
			current_reference=current_rows[(job.budget_id, job.subsample_seed)],
			historical_reference=inspection.historical_reference,
			dataset_row=dataset_rows[(job.budget_id, job.subsample_seed)],
		)
	return tuple(sorted(rows, key=shared._row_sort_key))


def _classify_center_job(
	config: object,
	job: VoxelLabelBudgetJob,
	*,
	estimated_bytes: int,
) -> VoxelLabelBudgetJobPlan:
	plan = classify_voxel_label_budget_job(
		shared._stage_config(config, job.model_role),
		job,
		estimated_bytes=estimated_bytes,
	)
	if plan.state == 'RESUME_LATEST':
		return VoxelLabelBudgetJobPlan(
			job,
			'RESUME_SAME_IDENTITY',
			plan.reason,
			plan.estimated_bytes,
		)
	return plan


def _validate_screening_audit(config: object) -> None:
	center_config.validate_f3_center_trace_masked_screening_audit(config)


def _hard_reference_rows(
	config: object,
	*,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
	current_rows: Mapping[tuple[str, int], Mapping[str, object]],
	historical_reference: F3VoxelLabelBudgetReferenceInspection,
) -> tuple[Mapping[str, object], ...]:
	hard_config = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(config.hard_multi_head_config)
	)
	manifest_path = shared.multi_head_run_manifest_path(hard_config)
	payload = shared._read_json(manifest_path)
	for name, path in (
		('dataset_manifest', hard_config.dataset_manifest),
		('original_run_manifest', hard_config.original_run_manifest),
		('current_k6_run_manifest', hard_config.current_k6_run_manifest),
	):
		control._validate_identity_at(payload.get(name), path, label=name)
	rows = payload.get('rows')
	if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
		raise TypeError('hard multi-head decoder manifest rows are invalid')
	if payload.get('row_count') != len(rows) or payload.get('complete_count') != len(
		rows
	):
		raise ValueError('hard multi-head decoder manifest is incomplete')
	primary = tuple(row for row in rows if row.get('model_role') == 'mh_nocons')
	if len(primary) != config.job_count:
		raise ValueError('hard mh_nocons reference matrix is incomplete')
	if any(
		row.get('model_tag') != 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
		for row in primary
	):
		raise ValueError('hard mh_nocons reference tag mismatch')
	for row in primary:
		key = (str(row['budget_id']), int(row['subsample_seed']))
		if key not in dataset_rows or key not in current_rows:
			raise ValueError('hard mh_nocons reference condition is not canonical')
		shared._validate_candidate_pairing(
			row,
			current_reference=current_rows[key],
			historical_reference=historical_reference,
			dataset_row=dataset_rows[key],
		)
	return primary


def _candidate_identity(
	config: object,
	candidate: object,
	*,
	canonical_valid_tokens_sha256: str,
) -> Mapping[str, object]:
	files = output_paths(candidate.embeddings_dir, config.dataset['name'])
	for path in (
		files.embeddings,
		files.valid_tokens,
		files.metadata,
		candidate.pretraining_handoff,
	):
		if not path.is_file():
			raise FileNotFoundError(path)
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	if embeddings.shape != (76, 113, 32, 384) or embeddings.dtype != np.float16:
		raise ValueError('center-trace embeddings must be float16 [76,113,32,384]')
	if valid.shape != (76, 113, 32) or valid.dtype != np.bool_ or int(valid.sum()) <= 0:
		raise ValueError('center-trace valid-token shape/dtype/count mismatch')
	if not _finite_valid_embeddings(embeddings, valid):
		raise ValueError('center-trace embeddings contain nonfinite valid values')
	valid_sha = file_sha256(files.valid_tokens)
	if valid_sha != canonical_valid_tokens_sha256:
		raise ValueError('center-trace valid-token identity differs from references')
	metadata = _read_json(files.metadata)
	checkpoint_value = metadata.get('checkpoint_path')
	checkpoint = Path(str(checkpoint_value)).resolve()
	if (
		checkpoint.name != 'best.pt'
		or candidate.model_tag not in checkpoint.parts
		or not checkpoint.is_file()
	):
		raise ValueError('center-trace embedding checkpoint/model identity mismatch')
	checkpoint_sha256 = file_sha256(checkpoint)
	if metadata.get('checkpoint_sha256') != checkpoint_sha256:
		raise ValueError('center-trace embedding checkpoint SHA-256 mismatch')
	stratigraphy = _mapping(
		metadata.get('stratigraphy_pretext'), 'center-trace embedding identity'
	)
	if (
		stratigraphy.get('model_tag') != candidate.model_tag
		or stratigraphy.get('head_spec') != 'multi_resolution_ordered_prototypes_v1'
		or stratigraphy.get('head_ks') != [6, 8, 10]
	):
		raise ValueError('center-trace embedding scientific identity mismatch')
	if metadata.get('output_dtype') != 'float16':
		raise ValueError('center-trace embedding output dtype mismatch')
	target = load_multi_head_target_manifest(config.multi_head_target_manifest)
	target_hashes = center_validation._multi_head_target_hashes(target)
	if stratigraphy.get('target_manifest_sha256') != file_sha256(
		config.multi_head_target_manifest
	):
		raise ValueError('center-trace embedding target manifest mismatch')
	if stratigraphy.get('per_head_target_sha256') != target_hashes:
		raise ValueError('center-trace embedding per-head target mismatch')
	handoff = center_validation.load_f3_center_trace_masked_pretraining_handoff(
		candidate.pretraining_handoff
	)
	if (
		handoff.get('model_tag') != candidate.model_tag
		or handoff.get('variant') != 'ctmask010_nocons'
	):
		raise ValueError('center-trace pretraining handoff identity mismatch')
	handoff_checkpoint = _mapping(handoff['checkpoint'], 'center-trace checkpoint')
	if (
		Path(str(handoff_checkpoint['path'])).resolve() != checkpoint
		or handoff_checkpoint.get('sha256') != checkpoint_sha256
	):
		raise ValueError('center-trace handoff checkpoint mismatch')
	handoff_embedding = _mapping(handoff['embedding'], 'center-trace embedding')
	for key, value in (
		('root', candidate.embeddings_dir),
		('embeddings_path', files.embeddings),
		('valid_tokens_path', files.valid_tokens),
		('metadata_path', files.metadata),
	):
		if Path(str(handoff_embedding[key])).resolve() != value.resolve():
			raise ValueError(f'center-trace handoff embedding {key} mismatch')
	for key, path in (
		('embeddings_sha256', files.embeddings),
		('valid_tokens_sha256', files.valid_tokens),
		('metadata_sha256', files.metadata),
	):
		if handoff_embedding.get(key) != file_sha256(path):
			raise ValueError(f'center-trace handoff embedding {key} hash mismatch')
	checkpoint_payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	if not isinstance(checkpoint_payload, Mapping):
		raise TypeError('center-trace best checkpoint must contain a mapping')
	validate_stratigraphy_checkpoint_payload(
		checkpoint_payload,
	)
	identity = _mapping(
		checkpoint_payload.get('stratigraphy_checkpoint'),
		'center-trace checkpoint identity',
	)
	if identity.get('schema_version') != 7:
		raise ValueError('center-trace checkpoint schema must be 7')
	if identity.get('model_tag') != candidate.model_tag:
		raise ValueError('center-trace checkpoint model tag mismatch')
	if identity.get('target_manifest_sha256') != file_sha256(
		config.multi_head_target_manifest
	):
		raise ValueError('center-trace checkpoint target mismatch')
	if stratigraphy.get('checkpoint_student_state_sha256') != identity.get(
		'student_state_sha256'
	):
		raise ValueError('center-trace extraction is not bound to student state')
	if 'checkpoint_spatial_context_state_sha256' not in stratigraphy:
		raise ValueError('center-trace extraction lineage lacks spatial-context state')
	return {
		'embeddings': _identity(files.embeddings),
		'valid_tokens': _identity(files.valid_tokens),
		'metadata': _identity(files.metadata),
		'checkpoint': _identity(checkpoint),
		'pretraining_handoff': _identity(candidate.pretraining_handoff),
	}


def _validate_candidate_pairing(
	row: Mapping[str, object],
	*,
	current_reference: Mapping[str, object],
	historical_reference: F3VoxelLabelBudgetReferenceInspection,
	dataset_row: Mapping[str, object],
) -> None:
	shared._validate_candidate_pairing(
		row,
		current_reference=current_reference,
		historical_reference=historical_reference,
		dataset_row=dataset_row,
	)


def _finite_valid_embeddings(embeddings: np.ndarray, valid: np.ndarray) -> bool:
	for index in range(embeddings.shape[0]):
		if (
			valid[index].any()
			and not np.isfinite(embeddings[index][valid[index]]).all()
		):
			return False
	return True


def _failed_row(job: VoxelLabelBudgetJob, error: BaseException) -> dict[str, object]:
	return {
		'budget_id': job.budget_id,
		'per_class_cap': job.per_class_cap,
		'subsample_seed': job.subsample_seed,
		'decoder_seed': job.decoder_seed,
		'model_role': job.model_role,
		'model_tag': job.model_tag,
		'status': 'failed',
		'action': 'FAILED',
		'error': f'{type(error).__name__}: {error}',
	}


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _identity(path: Path) -> Mapping[str, object]:
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	return _mapping(payload, f'JSON object {path}')


__all__ = [
	'F3VoxelLabelBudgetCenterTraceMaskedInspection',
	'F3VoxelLabelBudgetCenterTraceMaskedRunResult',
	'center_trace_masked_run_manifest_path',
	'inspect_f3_lithology_voxel_label_budget_center_trace_masked',
	'load_f3_lithology_voxel_label_budget_center_trace_masked_rows',
	'run_f3_lithology_voxel_label_budget_center_trace_masked',
]
