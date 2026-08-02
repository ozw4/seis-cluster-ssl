"""Two-candidate runner for the paired multi-head low-label voxel matrix.

This module deliberately delegates decoder execution, checkpoint selection,
coverage validation, resume classification, and quarantine to the established
voxel label-budget runner.  It owns only the candidate matrix and its compact
30-row manifest.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import importlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology import voxel_label_budget_control as control
from seis_ssl_cluster.f3.lithology import voxel_label_budget_results as results
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	F3VoxelLabelBudgetReferenceInspection,
	inspect_f3_lithology_voxel_label_budget_mae_reference_run,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	VoxelLabelBudgetJobPlan,
	classify_voxel_label_budget_job,
	quarantine_voxel_label_budget_output,
	run_voxel_label_budget_job,
)
from seis_ssl_cluster.f3.multi_head_pretraining_validation import (
	load_f3_multi_head_pretraining_handoff,
)
from seis_ssl_cluster.f3.soft_posterior_pretraining_validation import (
	load_f3_m5_soft_posterior_pretraining_handoff,
)
from seis_ssl_cluster.f3.xy_neighbor_consensus_pretraining_validation import (
	load_f3_xy_neighbor_consensus_pretraining_handoff,
)
from seis_ssl_cluster.f3.xy_neighbor_unanimous_pretraining_validation import (
	load_f3_xy_neighbor_unanimous_pretraining_handoff,
)
from seis_ssl_cluster.f3.xy_neighbor_unanimous_target_audit import (
	replay_f3_xy_neighbor_unanimous_target_audit,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	load_multi_head_xy_neighbor_unanimous_target_manifest,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
		F3VoxelLabelBudgetMultiHeadCandidate,
		F3VoxelLabelBudgetMultiHeadConfig,
	)


RUN_MANIFEST_NAME = 'multi_head_job_manifest.json'
RUN_MANIFEST_TYPE = 'f3_lithology_voxel_label_budget_multi_head'
RUN_SCHEMA_VERSION = 1
XY_MODEL_ID = 'mh_xycons1_nocons'
XY_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
XY_UNANIM_MODEL_ID = 'mh_xyunanim1_nocons'
XY_UNANIM_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1'
)


@dataclass(frozen=True)
class F3VoxelLabelBudgetMultiHeadInspection:
	"""Read-only action plan for the selected candidate jobs."""

	jobs: tuple[VoxelLabelBudgetJob, ...]
	plans: tuple[VoxelLabelBudgetJobPlan, ...]
	historical_reference: F3VoxelLabelBudgetReferenceInspection
	candidate_identities: Mapping[str, Mapping[str, object]]
	estimated_new_bytes: int
	disk_free_bytes: int


@dataclass(frozen=True)
class F3VoxelLabelBudgetMultiHeadRunResult:
	"""Persisted multi-head matrix state."""

	manifest_json: Path
	rows: tuple[Mapping[str, object], ...]
	quarantines: tuple[Path, ...]


class _CandidateStageConfig:
	"""Small adapter exposing one candidate through the shared stage API."""

	def __init__(
		self,
		config: F3VoxelLabelBudgetMultiHeadConfig,
		candidate: F3VoxelLabelBudgetMultiHeadCandidate,
	) -> None:
		self._config = config
		self.candidate = candidate

	def __getattr__(self, name: str) -> object:
		return getattr(self._config, name)

	@property
	def model_by_role(
		self,
	) -> Mapping[str, F3VoxelLabelBudgetMultiHeadCandidate]:
		"""Expose only this candidate to the shared single-job helper."""
		return {self.candidate.model_id: self.candidate}


def multi_head_run_manifest_path(config: F3VoxelLabelBudgetMultiHeadConfig) -> Path:
	"""Return the candidate-owned, compact job manifest path."""
	return config.reports_dir / str(
		getattr(config, 'run_manifest_name', RUN_MANIFEST_NAME)
	)


def inspect_f3_lithology_voxel_label_budget_multi_head(
	config: F3VoxelLabelBudgetMultiHeadConfig,
	*,
	candidate: str | None = None,
	budget: str | None = None,
	subsample_seed: int | None = None,
) -> F3VoxelLabelBudgetMultiHeadInspection:
	"""Validate references and produce the canonical candidate/budget/seed plan."""
	_validate_xy_neighbor_consensus_screening_audit(config)
	dataset_rows = _dataset_rows(config)
	historical_reference = _mae_reference(config, dataset_rows)
	current_rows = _current_k6_rows(config, dataset_rows)
	for key, row in current_rows.items():
		control._validate_paired_identity(
			row,
			reference=historical_reference,
			dataset_row=dataset_rows[key],
			reference_roles=(config.references.mae_model_id,),
		)
	canonical_valid_tokens_sha256 = _canonical_valid_tokens_sha256(current_rows)
	identities = {
		item.model_id: _candidate_identity(
			config,
			item,
			canonical_valid_tokens_sha256=canonical_valid_tokens_sha256,
		)
		for item in config.candidates
	}
	jobs = _jobs(config, dataset_rows)
	jobs = tuple(
		job
		for job in jobs
		if (candidate is None or job.model_role == candidate)
		and (budget is None or job.budget_id == budget)
		and (subsample_seed is None or job.subsample_seed == subsample_seed)
	)
	if not jobs:
		raise ValueError('job filters selected no multi-head voxel jobs')
	plans = tuple(
		classify_voxel_label_budget_job(_stage_config(config, job.model_role), job)
		for job in jobs
	)
	bytes_per_job = _estimated_candidate_job_bytes(current_rows)
	plans = tuple(
		VoxelLabelBudgetJobPlan(
			job=plan.job,
			state=plan.state,
			reason=plan.reason,
			estimated_bytes=0 if plan.state == 'REUSE_COMPLETED' else bytes_per_job,
		)
		for plan in plans
	)
	disk = shutil.disk_usage(_existing_parent(config.output_root.parent))
	return F3VoxelLabelBudgetMultiHeadInspection(
		jobs=jobs,
		plans=plans,
		historical_reference=historical_reference,
		candidate_identities=identities,
		estimated_new_bytes=sum(item.estimated_bytes for item in plans),
		disk_free_bytes=disk.free,
	)


def run_f3_lithology_voxel_label_budget_multi_head(  # noqa: C901, PLR0913
	config: F3VoxelLabelBudgetMultiHeadConfig,
	*,
	only_missing: bool = False,
	resume: bool = False,
	device: str = 'auto',
	candidate: str | None = None,
	budget: str | None = None,
	subsample_seed: int | None = None,
) -> F3VoxelLabelBudgetMultiHeadRunResult:
	"""Run/reuse exactly the selected rows, atomically updating owned rows."""
	if only_missing and resume:
		raise ValueError('--only-missing and --resume are mutually exclusive')
	inspection = inspect_f3_lithology_voxel_label_budget_multi_head(
		config, candidate=candidate, budget=budget, subsample_seed=subsample_seed
	)
	if inspection.estimated_new_bytes > inspection.disk_free_bytes:
		raise RuntimeError('insufficient disk for planned multi-head jobs')
	if resume and any(plan.state != 'RESUME_LATEST' for plan in inspection.plans):
		raise ValueError(
			'--resume requires an incomplete valid latest.pt for every selected job'
		)
	if (
		not only_missing
		and not resume
		and any(plan.state != 'NEW' for plan in inspection.plans)
	):
		raise FileExistsError('non-new job requires --only-missing or --resume')
	path = multi_head_run_manifest_path(config)
	prior_rows_by_key, quarantines = _prior_rows(
		path, config, inspection.candidate_identities
	)
	rows_by_key = dict(prior_rows_by_key)
	dataset_rows = _dataset_rows(config)
	current_rows = _current_k6_rows(config, dataset_rows)
	for plan in inspection.plans:
		job, state = plan.job, plan.state
		key = _job_key(job)
		prior_row = prior_rows_by_key.get(key)
		quarantine: Path | None = None
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
		stage = _stage_config(config, job.model_role)
		identity = inspection.candidate_identities[job.model_role]
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
				if _same_completed_row_except_action(prior_row, actual_row)
				else actual_row
			)
		else:
			checkpoint = (
				job.decoder_dir / 'latest.pt' if state == 'RESUME_LATEST' else None
			)
			run_voxel_label_budget_job(stage, job, device=device, resume=checkpoint)
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
		rows_by_key[key] = row
		if prior_row != row:
			_write_manifest(
				path,
				config,
				tuple(rows_by_key.values()),
				quarantines,
				inspection.candidate_identities,
			)
	ordered = tuple(sorted(rows_by_key.values(), key=_row_sort_key))
	if not path.is_file():
		_write_manifest(
			path,
			config,
			ordered,
			quarantines,
			inspection.candidate_identities,
		)
	return F3VoxelLabelBudgetMultiHeadRunResult(path, ordered, tuple(quarantines))


def load_f3_lithology_voxel_label_budget_multi_head_rows(
	config: F3VoxelLabelBudgetMultiHeadConfig,
) -> tuple[Mapping[str, object], ...]:
	"""Revalidate every completed owned row without rerunning decoder stages."""
	inspection = inspect_f3_lithology_voxel_label_budget_multi_head(config)
	payload = _read_json(multi_head_run_manifest_path(config))
	_validate_manifest(payload, config, inspection.candidate_identities)
	rows = tuple(payload.get('rows', ()))
	if len(rows) != config.job_count:
		raise ValueError(
			f'multi-head manifest must own exactly {config.job_count} rows'
		)
	by_job = {_job_key(job): job for job in inspection.jobs}
	dataset_rows = _dataset_rows(config)
	current_rows = _current_k6_rows(config, dataset_rows)
	if {_row_key(row) for row in rows} != set(by_job):
		raise ValueError('multi-head manifest job matrix mismatch')
	for row in rows:
		job = by_job[_row_key(row)]
		if row.get('status') != 'complete':
			raise ValueError(f'multi-head job is not complete: {_row_key(row)!r}')
		stage = _stage_config(config, job.model_role)
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
				'multi-head completed row differs from live artifacts: '
				f'{_row_key(row)!r}'
			)
		_validate_candidate_pairing(
			row,
			current_reference=current_rows[(job.budget_id, job.subsample_seed)],
			historical_reference=inspection.historical_reference,
			dataset_row=dataset_rows[(job.budget_id, job.subsample_seed)],
		)
	return tuple(sorted(rows, key=_row_sort_key))


def _candidate_identity(  # noqa: C901, PLR0912
	config: F3VoxelLabelBudgetMultiHeadConfig,
	candidate: F3VoxelLabelBudgetMultiHeadCandidate,
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
	embeddings = np.load(files.embeddings, mmap_mode='r')
	valid = np.load(files.valid_tokens, mmap_mode='r')
	if embeddings.shape != (76, 113, 32, 384) or embeddings.dtype != np.float16:
		raise ValueError('candidate embeddings must be float16 [76,113,32,384]')
	if valid.shape != (76, 113, 32) or valid.dtype != np.bool_ or int(valid.sum()) <= 0:
		raise ValueError('candidate valid-token shape/dtype/count mismatch')
	if not _finite_valid_embeddings(embeddings, valid):
		raise ValueError('candidate embeddings contain nonfinite valid values')
	valid_sha = file_sha256(files.valid_tokens)
	if valid_sha != canonical_valid_tokens_sha256:
		raise ValueError('candidate valid-token identity differs from MAE/current K6')
	metadata = _read_json(files.metadata)
	checkpoint = Path(str(metadata.get('checkpoint_path', '')))
	if (
		checkpoint.name != 'best.pt'
		or candidate.model_tag not in checkpoint.parts
		or not checkpoint.is_file()
	):
		raise ValueError('candidate embedding checkpoint/model identity mismatch')
	if metadata.get('checkpoint_sha256') != file_sha256(checkpoint):
		raise ValueError('candidate embedding checkpoint SHA-256 mismatch')
	stratigraphy = metadata.get('stratigraphy_pretext')
	if (
		not isinstance(stratigraphy, Mapping)
		or stratigraphy.get('model_tag') != candidate.model_tag
	):
		raise ValueError('candidate embedding stratigraphy model tag mismatch')
	if stratigraphy.get('head_spec') != 'multi_resolution_ordered_prototypes_v1':
		raise ValueError('candidate multi-head specification mismatch')
	if stratigraphy.get('head_ks') != [6, 8, 10]:
		raise ValueError('candidate multi-head K identity mismatch')
	if candidate.model_id == XY_MODEL_ID:
		_handoff_identity = _validate_xy_neighbor_consensus_handoff_provenance(
			candidate.pretraining_handoff,
			config=config,
			candidate=candidate,
			checkpoint=checkpoint,
			checkpoint_sha256=str(metadata['checkpoint_sha256']),
			embeddings_sha256=file_sha256(files.embeddings),
			valid_tokens_sha256=valid_sha,
			embedding_metadata_sha256=file_sha256(files.metadata),
			valid_token_count=int(valid.sum()),
			metadata=metadata,
			stratigraphy=stratigraphy,
		)
		return {
			'embeddings': _identity(files.embeddings),
			'valid_tokens': _identity(files.valid_tokens),
			'metadata': _identity(files.metadata),
			'checkpoint': _identity(checkpoint),
			'pretraining_handoff': _handoff_identity,
		}
	if candidate.model_id == XY_UNANIM_MODEL_ID:
		_handoff_identity = _validate_xy_neighbor_unanimous_handoff_provenance(
			candidate.pretraining_handoff,
			config=config,
			candidate=candidate,
			checkpoint=checkpoint,
			checkpoint_sha256=str(metadata['checkpoint_sha256']),
			embeddings_sha256=file_sha256(files.embeddings),
			valid_tokens_sha256=valid_sha,
			embedding_metadata_sha256=file_sha256(files.metadata),
			valid_token_count=int(valid.sum()),
			metadata=metadata,
			stratigraphy=stratigraphy,
		)
		return {
			'embeddings': _identity(files.embeddings),
			'valid_tokens': _identity(files.valid_tokens),
			'metadata': _identity(files.metadata),
			'checkpoint': _identity(checkpoint),
			'pretraining_handoff': _handoff_identity,
		}
	if candidate.model_id == 'mh_soft_nocons':
		_handoff_identity = _validate_soft_handoff_provenance(
			candidate.pretraining_handoff,
			candidate=candidate,
			checkpoint=checkpoint,
			checkpoint_sha256=str(metadata['checkpoint_sha256']),
			embeddings_sha256=file_sha256(files.embeddings),
			valid_tokens_sha256=valid_sha,
			embedding_metadata_sha256=file_sha256(files.metadata),
			stratigraphy=stratigraphy,
		)
		return {
			'embeddings': _identity(files.embeddings),
			'valid_tokens': _identity(files.valid_tokens),
			'metadata': _identity(files.metadata),
			'checkpoint': _identity(checkpoint),
			'pretraining_handoff': _handoff_identity,
		}
	expected_target_manifest_sha256 = file_sha256(config.multi_head_target_manifest)
	if stratigraphy.get('target_manifest_sha256') != expected_target_manifest_sha256:
		raise ValueError('candidate target manifest SHA-256 mismatch')
	consistency_weight = stratigraphy.get('consistency_weight')
	if candidate.model_id == 'mh_nocons' and (
		not isinstance(consistency_weight, float) or consistency_weight != 0.0
	):
		raise ValueError('nocons embedding consistency identity mismatch')
	if candidate.model_id == 'mh_cons010' and (
		not isinstance(consistency_weight, float) or consistency_weight != 0.1
	):
		raise ValueError('cons010 embedding consistency identity mismatch')
	_handoff_identity = _validate_handoff_provenance(
		candidate.pretraining_handoff,
		candidate=candidate,
		checkpoint=checkpoint,
		checkpoint_sha256=str(metadata['checkpoint_sha256']),
		embedding_metadata_sha256=file_sha256(files.metadata),
		stratigraphy=stratigraphy,
	)
	return {
		'embeddings': _identity(files.embeddings),
		'valid_tokens': _identity(files.valid_tokens),
		'metadata': _identity(files.metadata),
		'checkpoint': _identity(checkpoint),
		'pretraining_handoff': _handoff_identity,
	}


def _validate_handoff_provenance(  # noqa: PLR0913
	handoff_path: Path,
	*,
	candidate: F3VoxelLabelBudgetMultiHeadCandidate,
	checkpoint: Path,
	checkpoint_sha256: str,
	embedding_metadata_sha256: str,
	stratigraphy: Mapping[str, object],
) -> Mapping[str, object]:
	"""Bind #275's validated handoff to this embedding extraction.

	The handoff carries the immutable embedding-metadata digest, selected best
	checkpoint, and multi-head scientific identity, so a merely existing (or
	another candidate's) handoff cannot be used as provenance.
	"""
	handoff = load_f3_multi_head_pretraining_handoff(handoff_path)
	if (
		handoff.get('artifact_type') != 'f3_multi_head_pretraining_handoff'
		or handoff.get('schema_version') != 1
		or handoff.get('status') != 'PASS'
	):
		raise ValueError('candidate pretraining handoff type/status mismatch')
	if handoff.get('model_tag') != candidate.model_tag:
		raise ValueError('candidate pretraining handoff model tag mismatch')
	if handoff.get('embedding_metadata_sha256') != embedding_metadata_sha256:
		raise ValueError('candidate pretraining handoff metadata SHA-256 mismatch')
	handoff_checkpoint = handoff.get('checkpoint')
	if not isinstance(handoff_checkpoint, Mapping):
		raise TypeError('candidate pretraining handoff checkpoint is missing')
	if (
		Path(str(handoff_checkpoint.get('path', ''))).resolve()
		!= checkpoint.resolve()
		or handoff_checkpoint.get('sha256') != checkpoint_sha256
	):
		raise ValueError('candidate pretraining handoff checkpoint mismatch')
	handoff_stratigraphy = handoff.get('stratigraphy_pretext')
	if not isinstance(handoff_stratigraphy, Mapping):
		raise TypeError(
			'candidate pretraining handoff stratigraphy identity is missing'
		)
	for key in (
		'head_spec',
		'head_ks',
		'target_manifest_sha256',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'scientific_identity_sha256',
	):
		if handoff_stratigraphy.get(key) != stratigraphy.get(key):
			raise ValueError(
				'candidate pretraining handoff stratigraphy mismatch: '
				f'{key}'
			)
	return _identity(handoff_path)


def _validate_soft_handoff_provenance(  # noqa: PLR0913
	handoff_path: Path,
	*,
	candidate: F3VoxelLabelBudgetMultiHeadCandidate,
	checkpoint: Path,
	checkpoint_sha256: str,
	embeddings_sha256: str,
	valid_tokens_sha256: str,
	embedding_metadata_sha256: str,
	stratigraphy: Mapping[str, object],
) -> Mapping[str, object]:
	"""Bind M5-U's PASS handoff to the exact soft extraction artifacts."""
	handoff = load_f3_m5_soft_posterior_pretraining_handoff(handoff_path)
	if handoff.get('model_tag') != candidate.model_tag:
		raise ValueError('M5-U handoff model tag mismatch')
	embedding = handoff['embedding']
	checkpoint_evidence = handoff['checkpoint']
	targets = handoff['targets']
	if (
		embedding.get('metadata_sha256') != embedding_metadata_sha256
		or embedding.get('embeddings_sha256') != embeddings_sha256
		or embedding.get('valid_tokens_sha256') != valid_tokens_sha256
		or checkpoint_evidence.get('sha256') != checkpoint_sha256
		or Path(str(checkpoint_evidence.get('path', ''))).resolve()
		!= checkpoint.resolve()
	):
		raise ValueError('M5-U handoff extraction identity mismatch')
	if (
		stratigraphy.get('target_representation') != 'ordered_path_state_posterior_v1'
		or targets.get('target_representation') != 'ordered_path_state_posterior_v1'
		or stratigraphy.get('posterior_manifest_sha256')
		!= targets.get('posterior_manifest_sha256')
	):
		raise ValueError('M5-U soft-posterior representation identity mismatch')
	return _identity(handoff_path)


def _validate_xy_neighbor_consensus_handoff_provenance(  # noqa: C901, PLR0913
	handoff_path: Path,
	*,
	config: F3VoxelLabelBudgetMultiHeadConfig,
	candidate: F3VoxelLabelBudgetMultiHeadCandidate,
	checkpoint: Path,
	checkpoint_sha256: str,
	embeddings_sha256: str,
	valid_tokens_sha256: str,
	embedding_metadata_sha256: str,
	valid_token_count: int,
	metadata: Mapping[str, object],
	stratigraphy: Mapping[str, object],
) -> Mapping[str, object]:
	"""Bind only the immutable schema-v5 XY-consensus lineage.

	The strict handoff loader rejects M4, M5-U, and M5-LS handoffs before this
	function accepts any candidate artifact.  The checks below then bind its
	recorded references to the live checkpoint, embedding files, and immutable
	consensus target publication.
	"""
	if candidate.model_id != XY_MODEL_ID or candidate.model_tag != XY_MODEL_TAG:
		raise ValueError('XY-neighbour-consensus candidate identity mismatch')
	handoff = load_f3_xy_neighbor_consensus_pretraining_handoff(handoff_path)
	targets = _mapping_value(handoff.get('targets'), 'XY handoff targets')
	handoff_checkpoint = _mapping_value(
		handoff.get('checkpoint'), 'XY handoff checkpoint'
	)
	embedding = _mapping_value(handoff.get('embedding'), 'XY handoff embedding')
	if (
		Path(str(handoff_checkpoint.get('path', ''))).resolve()
		!= checkpoint.resolve()
		or handoff_checkpoint.get('sha256') != checkpoint_sha256
	):
		raise ValueError('XY-neighbour-consensus handoff checkpoint mismatch')
	if (
		Path(str(embedding.get('root', ''))).resolve()
		!= candidate.embeddings_dir.resolve()
		or Path(str(embedding.get('metadata_path', ''))).resolve()
		!= _embedding_metadata_path(candidate.embeddings_dir, config.dataset['name'])
		or embedding.get('metadata_sha256') != embedding_metadata_sha256
		or embedding.get('embeddings_sha256') != embeddings_sha256
		or embedding.get('valid_tokens_sha256') != valid_tokens_sha256
		or embedding.get('valid_token_count') != valid_token_count
	):
		raise ValueError('XY-neighbour-consensus handoff extraction identity mismatch')
	if (
		targets.get('target_representation')
		!= 'xy_neighbor_consensus_hard_labels_v1'
		or targets.get('target_semantics')
		!= 'xy_neighbor_consensus_hard_label_smoothing_v1'
		or targets.get('consistency_policy')
		!= 'disabled_for_xy_neighbor_consensus_v1'
	):
		raise ValueError('XY-neighbour-consensus handoff target identity mismatch')
	target_reference = _reference_value(
		targets.get('target_manifest'), 'XY handoff target manifest'
	)
	target_path = Path(str(target_reference['path'])).resolve()
	if (
		not target_path.is_file()
		or target_reference['sha256'] != file_sha256(target_path)
	):
		raise ValueError('XY-neighbour-consensus target manifest identity mismatch')
	target = load_multi_head_xy_neighbor_consensus_target_manifest(
		target_path,
		validate_array_semantics=False,
	)
	if target_reference['sha256'] != file_sha256(target_path):  # pragma: no cover
		raise ValueError('XY-neighbour-consensus target manifest changed during read')
	target_hashes = _xy_neighbor_consensus_target_head_hashes(target)
	if targets.get('xy_neighbor_consensus_target_head_hashes') != target_hashes:
		raise ValueError('XY-neighbour-consensus handoff target head hashes mismatch')
	if (
		stratigraphy.get('per_head_xy_neighbor_consensus_target_sha256')
		!= target_hashes
	):
		raise ValueError('XY-neighbour-consensus embedding target head hashes mismatch')
	target_smoothing = target.get('smoothing')
	if (
		not isinstance(target_smoothing, Mapping)
		or targets.get('xy_neighbor_consensus_smoothing') != target_smoothing
		or stratigraphy.get('xy_neighbor_consensus_smoothing') != target_smoothing
	):
		raise ValueError('XY-neighbour-consensus smoothing policy mismatch')
	target_source = _reference_value(
		target.get('source_hard_manifest'), 'XY target source hard manifest'
	)
	handoff_source = _reference_value(
		targets.get('source_hard_manifest'), 'XY handoff source hard manifest'
	)
	if target_source != handoff_source:
		raise ValueError('XY-neighbour-consensus handoff source hard manifest mismatch')
	source_path = Path(str(target_source['path'])).resolve()
	if (
		source_path != config.multi_head_target_manifest.resolve()
		or target_source['sha256'] != file_sha256(config.multi_head_target_manifest)
		or stratigraphy.get('source_hard_manifest_sha256') != target_source['sha256']
	):
		raise ValueError('XY-neighbour-consensus source hard manifest mismatch')
	_xy_neighbor_consensus_checkpoint_identity(
		checkpoint,
		candidate=candidate,
		target_reference=target_reference,
		target_hashes=target_hashes,
		source_reference=target_source,
		target_smoothing=target_smoothing,
		targets=targets,
		stratigraphy=stratigraphy,
		metadata=metadata,
	)
	return _identity(handoff_path)


def _validate_xy_neighbor_unanimous_handoff_provenance(  # noqa: C901, PLR0912, PLR0913
	handoff_path: Path,
	*,
	config: F3VoxelLabelBudgetMultiHeadConfig,
	candidate: F3VoxelLabelBudgetMultiHeadCandidate,
	checkpoint: Path,
	checkpoint_sha256: str,
	embeddings_sha256: str,
	valid_tokens_sha256: str,
	embedding_metadata_sha256: str,
	valid_token_count: int,
	metadata: Mapping[str, object],
	stratigraphy: Mapping[str, object],
) -> Mapping[str, object]:
	"""Bind only the immutable schema-v6 unanimous hard-target lineage."""
	if (
		candidate.model_id != XY_UNANIM_MODEL_ID
		or candidate.model_tag != XY_UNANIM_MODEL_TAG
	):
		raise ValueError('XY-neighbour-unanimous candidate identity mismatch')
	handoff = load_f3_xy_neighbor_unanimous_pretraining_handoff(handoff_path)
	targets = _mapping_value(handoff.get('targets'), 'unanimous handoff targets')
	handoff_checkpoint = _mapping_value(
		handoff.get('checkpoint'), 'unanimous handoff checkpoint'
	)
	embedding = _mapping_value(handoff.get('embedding'), 'unanimous handoff embedding')
	if (
		Path(str(handoff_checkpoint.get('path', ''))).resolve()
		!= checkpoint.resolve()
		or handoff_checkpoint.get('sha256') != checkpoint_sha256
	):
		raise ValueError('XY-neighbour-unanimous handoff checkpoint mismatch')
	if (
		Path(str(embedding.get('root', ''))).resolve()
		!= candidate.embeddings_dir.resolve()
		or Path(str(embedding.get('metadata_path', ''))).resolve()
		!= _embedding_metadata_path(candidate.embeddings_dir, config.dataset['name'])
		or embedding.get('metadata_sha256') != embedding_metadata_sha256
		or embedding.get('embeddings_sha256') != embeddings_sha256
		or embedding.get('valid_tokens_sha256') != valid_tokens_sha256
		or embedding.get('valid_token_count') != valid_token_count
	):
		raise ValueError('XY-neighbour-unanimous handoff extraction identity mismatch')
	if (
		targets.get('target_representation')
		!= 'xy_neighbor_unanimous_hard_labels_v1'
		or targets.get('target_semantics')
		!= 'xy_neighbor_unanimous_outlier_correction_v1'
		or targets.get('consistency_policy')
		!= 'disabled_for_xy_neighbor_unanimous_v1'
	):
		raise ValueError('XY-neighbour-unanimous handoff target identity mismatch')
	target_reference = _reference_value(
		targets.get('target_manifest'), 'unanimous handoff target manifest'
	)
	target_path = Path(str(target_reference['path'])).resolve()
	if (
		not target_path.is_file()
		or target_reference['sha256'] != file_sha256(target_path)
	):
		raise ValueError('XY-neighbour-unanimous target manifest identity mismatch')
	target_audit_reference = _reference_value(
		targets.get('target_audit'), 'unanimous handoff target audit'
	)
	target_audit_path = Path(str(target_audit_reference['path'])).resolve()
	if (
		not target_audit_path.is_file()
		or target_audit_reference['sha256'] != file_sha256(target_audit_path)
	):
		raise ValueError('XY-neighbour-unanimous target audit identity mismatch')
	target_audit = replay_f3_xy_neighbor_unanimous_target_audit(
		target_audit_path, artifact_root=config.artifact_root
	)
	if (
		target_audit.get('status') != 'XYUNANIM_TARGET_GO'
		or target_audit.get('xy_neighbor_unanimous_target_manifest')
		!= target_reference
	):
		raise ValueError('XY-neighbour-unanimous target audit binding mismatch')
	target = load_multi_head_xy_neighbor_unanimous_target_manifest(
		target_path,
		validate_array_semantics=False,
	)
	if target_reference['sha256'] != file_sha256(target_path):  # pragma: no cover
		raise ValueError('XY-neighbour-unanimous target manifest changed during read')
	target_hashes = _xy_neighbor_unanimous_target_head_hashes(target)
	if targets.get('xy_neighbor_unanimous_target_head_hashes') != target_hashes:
		raise ValueError('XY-neighbour-unanimous handoff target head hashes mismatch')
	if (
		stratigraphy.get('per_head_xy_neighbor_unanimous_target_sha256')
		!= target_hashes
	):
		raise ValueError('XY-neighbour-unanimous embedding target head hashes mismatch')
	target_smoothing = target.get('smoothing')
	if (
		not isinstance(target_smoothing, Mapping)
		or targets.get('xy_neighbor_unanimous_smoothing') != target_smoothing
		or stratigraphy.get('xy_neighbor_unanimous_smoothing') != target_smoothing
	):
		raise ValueError('XY-neighbour-unanimous smoothing policy mismatch')
	target_source = _reference_value(
		target.get('source_hard_manifest'), 'unanimous target source hard manifest'
	)
	handoff_source = _reference_value(
		targets.get('source_hard_manifest'), 'unanimous handoff source hard manifest'
	)
	if target_source != handoff_source:
		raise ValueError('XY-neighbour-unanimous handoff source hard manifest mismatch')
	if target_audit.get('source_hard_manifest') != target_source:
		raise ValueError('XY-neighbour-unanimous target audit source mismatch')
	source_path = Path(str(target_source['path'])).resolve()
	if (
		source_path != config.multi_head_target_manifest.resolve()
		or target_source['sha256'] != file_sha256(config.multi_head_target_manifest)
		or stratigraphy.get('source_hard_manifest_sha256') != target_source['sha256']
	):
		raise ValueError('XY-neighbour-unanimous source hard manifest mismatch')
	_xy_neighbor_unanimous_checkpoint_identity(
		checkpoint,
		candidate=candidate,
		target_reference=target_reference,
		target_hashes=target_hashes,
		source_reference=target_source,
		target_smoothing=target_smoothing,
		targets=targets,
		stratigraphy=stratigraphy,
		metadata=metadata,
	)
	return _identity(handoff_path)


def _embedding_metadata_path(embeddings_dir: Path, dataset_name: object) -> Path:
	if not isinstance(dataset_name, str) or not dataset_name:
		raise TypeError('candidate dataset name is invalid')
	return output_paths(embeddings_dir, dataset_name).metadata.resolve()


def _xy_neighbor_consensus_checkpoint_identity(  # noqa: C901, PLR0913
	checkpoint: Path,
	*,
	candidate: F3VoxelLabelBudgetMultiHeadCandidate,
	target_reference: Mapping[str, object],
	target_hashes: Mapping[str, object],
	source_reference: Mapping[str, object],
	target_smoothing: Mapping[str, object],
	targets: Mapping[str, object],
	stratigraphy: Mapping[str, object],
	metadata: Mapping[str, object],
) -> Mapping[str, object]:
	"""Check schema-v5 checkpoint identity without accepting legacy schemas."""
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError('XY-neighbour-consensus checkpoint must be a mapping')
	identity = _mapping_value(
		payload.get('stratigraphy_checkpoint'),
		'XY-neighbour-consensus checkpoint identity',
	)
	if identity.get('schema_version') != 5:
		raise ValueError('XY-neighbour-consensus checkpoint schema must be 5')
	for key, expected in (
		('model_tag', candidate.model_tag),
		('head_spec', 'multi_resolution_ordered_prototypes_v1'),
		('head_ks', [6, 8, 10]),
		('target_representation', 'xy_neighbor_consensus_hard_labels_v1'),
		('target_semantics', 'xy_neighbor_consensus_hard_label_smoothing_v1'),
		('xy_neighbor_consensus_target_manifest_sha256', target_reference['sha256']),
		('per_head_xy_neighbor_consensus_targets', target_hashes),
		('source_hard_manifest_sha256', source_reference['sha256']),
		('xy_neighbor_consensus_smoothing', target_smoothing),
		('consistency_policy', 'disabled_for_xy_neighbor_consensus_v1'),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
	):
		if identity.get(key) != expected:
			raise ValueError(
				'XY-neighbour-consensus checkpoint identity mismatch: '
				f'{key}'
			)
	checkpoint_target = _reference_value(
		identity.get('xy_neighbor_consensus_target_manifest'),
		'XY-neighbour-consensus checkpoint target manifest',
	)
	if checkpoint_target != target_reference:
		raise ValueError('XY-neighbour-consensus checkpoint target manifest mismatch')
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if identity.get(key) != targets.get(key):
			raise ValueError(
				f'XY-neighbour-consensus checkpoint initial state mismatch: {key}'
			)
	for key, expected in (
		('model_tag', candidate.model_tag),
		('head_spec', 'multi_resolution_ordered_prototypes_v1'),
		('head_ks', [6, 8, 10]),
		('target_representation', 'xy_neighbor_consensus_hard_labels_v1'),
		('target_semantics', 'xy_neighbor_consensus_hard_label_smoothing_v1'),
		('xy_neighbor_consensus_target_manifest_sha256', target_reference['sha256']),
		('per_head_xy_neighbor_consensus_target_sha256', target_hashes),
		('source_hard_manifest_sha256', source_reference['sha256']),
		('xy_neighbor_consensus_smoothing', target_smoothing),
		('consistency_policy', 'disabled_for_xy_neighbor_consensus_v1'),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('scientific_identity_sha256', identity.get('scientific_identity_sha256')),
		(
			'checkpoint_stratigraphy_state_sha256',
			identity.get('stratigraphy_state_sha256'),
		),
	):
		if stratigraphy.get(key) != expected:
			raise ValueError(
				'XY-neighbour-consensus embedding identity mismatch: '
				f'{key}'
			)
	if (
		stratigraphy.get('xy_neighbor_consensus_target_manifest_path')
		!= target_reference['path']
	):
		raise ValueError(
			'XY-neighbour-consensus embedding target manifest path mismatch'
		)
	if any(
		'posterior' in str(key) or 'lateral' in str(key)
		for key in (*stratigraphy, *identity, *metadata)
	):
		raise ValueError(
			'XY-neighbour-consensus provenance carries posterior/lateral fields'
		)
	return identity


def _xy_neighbor_unanimous_checkpoint_identity(  # noqa: C901, PLR0913
	checkpoint: Path,
	*,
	candidate: F3VoxelLabelBudgetMultiHeadCandidate,
	target_reference: Mapping[str, object],
	target_hashes: Mapping[str, object],
	source_reference: Mapping[str, object],
	target_smoothing: Mapping[str, object],
	targets: Mapping[str, object],
	stratigraphy: Mapping[str, object],
	metadata: Mapping[str, object],
) -> Mapping[str, object]:
	"""Check schema-v6 identity without accepting legacy target policies."""
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError('XY-neighbour-unanimous checkpoint must be a mapping')
	identity = _mapping_value(
		payload.get('stratigraphy_checkpoint'),
		'XY-neighbour-unanimous checkpoint identity',
	)
	if identity.get('schema_version') != 6:
		raise ValueError('XY-neighbour-unanimous checkpoint schema must be 6')
	for key, expected in (
		('model_tag', candidate.model_tag),
		('head_spec', 'multi_resolution_ordered_prototypes_v1'),
		('head_ks', [6, 8, 10]),
		('target_representation', 'xy_neighbor_unanimous_hard_labels_v1'),
		('target_semantics', 'xy_neighbor_unanimous_outlier_correction_v1'),
		('xy_neighbor_unanimous_target_manifest_sha256', target_reference['sha256']),
		('per_head_xy_neighbor_unanimous_targets', target_hashes),
		('source_hard_manifest_sha256', source_reference['sha256']),
		('xy_neighbor_unanimous_smoothing', target_smoothing),
		('consistency_policy', 'disabled_for_xy_neighbor_unanimous_v1'),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
	):
		if identity.get(key) != expected:
			raise ValueError(
				'XY-neighbour-unanimous checkpoint identity mismatch: '
				f'{key}'
			)
	checkpoint_target = _reference_value(
		identity.get('xy_neighbor_unanimous_target_manifest'),
		'XY-neighbour-unanimous checkpoint target manifest',
	)
	if checkpoint_target != target_reference:
		raise ValueError('XY-neighbour-unanimous checkpoint target manifest mismatch')
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if identity.get(key) != targets.get(key):
			raise ValueError(
				f'XY-neighbour-unanimous checkpoint initial state mismatch: {key}'
			)
	for key, expected in (
		('model_tag', candidate.model_tag),
		('head_spec', 'multi_resolution_ordered_prototypes_v1'),
		('head_ks', [6, 8, 10]),
		('target_representation', 'xy_neighbor_unanimous_hard_labels_v1'),
		('target_semantics', 'xy_neighbor_unanimous_outlier_correction_v1'),
		(
			'xy_neighbor_unanimous_target_manifest_sha256',
			target_reference['sha256'],
		),
		('per_head_xy_neighbor_unanimous_target_sha256', target_hashes),
		('source_hard_manifest_sha256', source_reference['sha256']),
		('xy_neighbor_unanimous_smoothing', target_smoothing),
		('consistency_policy', 'disabled_for_xy_neighbor_unanimous_v1'),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('scientific_identity_sha256', identity.get('scientific_identity_sha256')),
		(
			'checkpoint_stratigraphy_state_sha256',
			identity.get('stratigraphy_state_sha256'),
		),
	):
		if stratigraphy.get(key) != expected:
			raise ValueError(
				'XY-neighbour-unanimous embedding identity mismatch: '
				f'{key}'
			)
	if (
		stratigraphy.get('xy_neighbor_unanimous_target_manifest_path')
		!= target_reference['path']
	):
		raise ValueError(
			'XY-neighbour-unanimous embedding target manifest path mismatch'
		)
	if any(
		'posterior' in str(key)
		or 'lateral' in str(key)
		or 'xy_neighbor_consensus' in str(key)
		for key in (*stratigraphy, *identity, *metadata)
	):
		raise ValueError(
			'XY-neighbour-unanimous provenance carries posterior/lateral fields'
		)
	return identity


def _xy_neighbor_consensus_target_head_hashes(
	target: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	"""Extract the exact K=6/8/10 consensus target artifact hash matrix."""
	heads = _mapping_value(target.get('heads'), 'XY target heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in ('6', '8', '10'):
		head = _mapping_value(heads.get(k), f'XY target head k={k}')
		surveys = _mapping_value(head.get('surveys'), f'XY target surveys k={k}')
		result[k] = {}
		for survey_id, value in surveys.items():
			entry = _mapping_value(value, f'XY target survey k={k}/{survey_id}')
			result[k][str(survey_id)] = {}
			for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
				reference = _reference_value(entry.get(name), f'XY target {name}')
				result[k][str(survey_id)][name] = str(reference['sha256'])
	return result


def _xy_neighbor_unanimous_target_head_hashes(
	target: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	"""Extract the exact K=6/8/10 unanimous target artifact hash matrix."""
	heads = _mapping_value(target.get('heads'), 'unanimous target heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in ('6', '8', '10'):
		head = _mapping_value(heads.get(k), f'unanimous target head k={k}')
		surveys = _mapping_value(head.get('surveys'), f'unanimous target surveys k={k}')
		result[k] = {}
		for survey_id, value in surveys.items():
			entry = _mapping_value(value, f'unanimous target survey k={k}/{survey_id}')
			result[k][str(survey_id)] = {}
			for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
				reference = _reference_value(
					entry.get(name), f'unanimous target {name}'
				)
				result[k][str(survey_id)][name] = str(reference['sha256'])
	return result


def _mapping_value(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _reference_value(value: object, label: str) -> Mapping[str, object]:
	reference = _mapping_value(value, label)
	if not {'path', 'sha256'} <= set(reference):
		raise ValueError(f'{label} must contain path and SHA-256')
	path, sha256 = reference.get('path'), reference.get('sha256')
	if not isinstance(path, str) or not path:
		raise TypeError(f'{label}.path is missing')
	if not isinstance(sha256, str) or len(sha256) != 64:
		raise TypeError(f'{label}.sha256 is invalid')
	return reference


def _validate_xy_neighbor_consensus_screening_audit(
	config: F3VoxelLabelBudgetMultiHeadConfig,
) -> None:
	"""Revalidate policy-specific audits before planning successor jobs."""
	if any(item.model_id == XY_UNANIM_MODEL_ID for item in config.candidates):
		audit_path = getattr(config, 'screening_audit', None)
		if not isinstance(audit_path, Path):
			raise TypeError(
				'XY-neighbour-unanimous candidate requires its closed config'
			)
		audit = importlib.import_module(
			'seis_ssl_cluster.f3.lithology.xy_neighbor_unanimous_screening_audit'
		)
		payload = audit.load_f3_xy_neighbor_unanimous_screening_audit(audit_path)
		candidate = next(
			item for item in config.candidates if item.model_id == XY_UNANIM_MODEL_ID
		)
		audit.validate_f3_xy_neighbor_unanimous_screening_audit_binding(
			payload,
			model_id=candidate.model_id,
			model_tag=candidate.model_tag,
			pretraining_handoff=candidate.pretraining_handoff,
			embeddings_dir=candidate.embeddings_dir,
		)
	if not any(item.model_id == XY_MODEL_ID for item in config.candidates):
		return
	audit_path = getattr(config, 'screening_audit', None)
	if not isinstance(audit_path, Path):
		raise TypeError('XY-neighbour-consensus candidate requires its closed config')
	audit = importlib.import_module(
		'seis_ssl_cluster.f3.lithology.xy_neighbor_consensus_screening_audit'
	)
	payload = audit.load_f3_xy_neighbor_consensus_screening_audit(
		audit_path
	)
	candidate = next(
		item for item in config.candidates if item.model_id == XY_MODEL_ID
	)
	audit.validate_f3_xy_neighbor_consensus_screening_audit_binding(
		payload,
		model_id=candidate.model_id,
		model_tag=candidate.model_tag,
		pretraining_handoff=candidate.pretraining_handoff,
		embeddings_dir=candidate.embeddings_dir,
	)


def _finite_valid_embeddings(
	embeddings: np.ndarray, valid: np.ndarray
) -> bool:
	"""Check valid embedding values without materializing the full volume."""
	for inline_index in range(embeddings.shape[0]):
		inline_valid = valid[inline_index]
		if inline_valid.any() and not np.isfinite(
			embeddings[inline_index][inline_valid]
		).all():
			return False
	return True


def _jobs(
	config: F3VoxelLabelBudgetMultiHeadConfig,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[VoxelLabelBudgetJob, ...]:
	jobs = []
	for candidate in config.candidates:
		for budget in config.budgets:
			for seed in config.subsample_seeds:
				row = dataset_rows[(budget, seed)]
				jobs.append(
					VoxelLabelBudgetJob(
						budget_id=budget,
						per_class_cap=int(row['per_class_cap']),
						subsample_seed=seed,
						decoder_seed=config.decoder_seed(seed),
						model_role=candidate.model_id,
						model_tag=candidate.model_tag,
						voxel_dataset_root=Path(str(row['voxel_dataset_root'])),
						output_root=config.output_root
						/ 'jobs'
						/ f'candidate={candidate.model_id}'
						/ f'budget={budget}'
						/ f'subsample_seed={seed}',
						dataset_row=row,
					)
				)
	return tuple(jobs)


def _stage_config(
	config: F3VoxelLabelBudgetMultiHeadConfig, model_id: str
) -> _CandidateStageConfig:
	return _CandidateStageConfig(
		config, next(item for item in config.candidates if item.model_id == model_id)
	)


def _dataset_rows(
	config: F3VoxelLabelBudgetMultiHeadConfig,
) -> Mapping[tuple[str, int], Mapping[str, object]]:
	"""Load the immutable label-budget matrix."""
	datasets, _identity = results._load_dataset_manifest(config.dataset_manifest)
	payload = _read_json(config.dataset_manifest)
	rows = payload.get('rows')
	if not isinstance(rows, list):
		raise TypeError('multi-head dataset manifest rows must be a list')
	result: dict[tuple[str, int], Mapping[str, object]] = {}
	for row in rows:
		if not isinstance(row, Mapping):
			raise TypeError('multi-head dataset manifest row must be a mapping')
		key = (str(row.get('budget_id')), int(row.get('subsample_seed', -1)))
		if key in result:
			raise ValueError(f'duplicate multi-head dataset condition: {key!r}')
		result[key] = row
	if set(result) != set(datasets):
		raise ValueError('multi-head dataset manifest condition matrix mismatch')
	return result


def _current_k6_rows(
	config: F3VoxelLabelBudgetMultiHeadConfig,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> Mapping[tuple[str, int], Mapping[str, object]]:
	"""Admit only current-K6 rows revalidated against their live artifacts."""
	manifest = config.current_k6_run_manifest
	output_root = (
		manifest.parent.parent if manifest.parent.name == 'reports' else manifest.parent
	)
	control_base = config.base
	control_config = replace(
		control_base,
		references=replace(
			control_base.references,
			dataset_manifest=config.dataset_manifest,
			historical_run_manifest=config.original_run_manifest,
			mae_model_id=config.references.mae_model_id,
			historical_m1_model_id=config.references.historical_m1_model_id,
		),
		output_root=output_root,
		validate_pairing_reference=False,
	)
	rows = control.load_f3_lithology_voxel_label_budget_control_rows(
		control_config, run_manifest_path=manifest
	)
	result: dict[tuple[str, int], Mapping[str, object]] = {}
	for row in rows:
		if not isinstance(row, Mapping):
			raise TypeError('current K6 reference row must be a mapping')
		if row.get('model_role') != config.references.current_k6_model_id:
			raise ValueError('current K6 reference model role mismatch')
		if row.get('model_tag') != control_config.candidate.model_tag:
			raise ValueError('current K6 reference model tag mismatch')
		if row.get('status') != 'complete':
			raise ValueError('current K6 reference row is not complete')
		key = (str(row.get('budget_id')), int(row.get('subsample_seed', -1)))
		if key in result:
			raise ValueError('current K6 reference contains duplicate job row')
		result[key] = row
	if set(result) != set(dataset_rows):
		raise ValueError('current K6 reference matrix mismatch')
	return result


def _mae_reference(
	config: F3VoxelLabelBudgetMultiHeadConfig,
	dataset_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> F3VoxelLabelBudgetReferenceInspection:
	"""Load MAE and admit historical M1 only as an optional report source."""
	reference = inspect_f3_lithology_voxel_label_budget_mae_reference_run(
		config.dataset_manifest,
		config.original_run_manifest,
		include_historical_m1=(
			config.references.historical_m1_model_id is not None
		),
	)
	mae_rows = {
		(job.dataset.budget_id, job.dataset.subsample_seed)
		for job in reference.jobs
		if job.model_role == config.references.mae_model_id
	}
	if mae_rows != set(dataset_rows):
		raise ValueError('MAE reference matrix mismatch')
	return reference


def _canonical_valid_tokens_sha256(
	current_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> str:
	"""Return the unique valid-token identity from the primary K=6 baseline."""
	identities = {
		row.get('canonical_valid_token_sha256') for row in current_rows.values()
	}
	if len(identities) != 1:
		raise ValueError('current K6 valid-token identity differs across jobs')
	identity = next(iter(identities))
	if not isinstance(identity, str) or len(identity) != 64:
		raise ValueError('current K6 valid-token identity is invalid')
	return identity


def _estimated_candidate_job_bytes(
	current_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> int:
	"""Estimate candidate output from the primary paired baseline footprint."""
	roots: list[Path] = []
	for row in current_rows.values():
		checkpoint = row.get('latest_checkpoint')
		if not isinstance(checkpoint, Mapping):
			raise TypeError('current K6 latest checkpoint identity is missing')
		path = Path(str(checkpoint.get('path', '')))
		if not path.is_file():
			raise FileNotFoundError(path)
		roots.append(path.parent.parent)
	if not roots:
		raise ValueError('current K6 reference has no jobs')
	sizes = [control._tree_size(path) for path in roots]
	return max(1, math.ceil((sum(sizes) / len(sizes)) * 1.2))


def _existing_parent(path: Path) -> Path:
	"""Return the nearest existing directory for an initial output allocation."""
	candidate = path
	while not candidate.exists():
		parent = candidate.parent
		if parent == candidate:
			raise FileNotFoundError(path)
		candidate = parent
	return candidate


def _validate_current_pair(
	row: Mapping[str, object], reference: Mapping[str, object]
) -> None:
	for key in control.PAIR_IDENTITY_KEYS:
		if row.get(key) != reference.get(key):
			raise ValueError(f'candidate/current K6 paired identity mismatch: {key}')
	for key in ('train_mask_sha256', 'runtime_contract'):
		if row.get(key) != reference.get(key):
			raise ValueError(f'candidate/current K6 runtime identity mismatch: {key}')
	coverage = row.get('prediction_coverage')
	if (
		not isinstance(coverage, Mapping)
		or coverage.get('duplicate_write_count') != 0
		or coverage.get('missing_write_count') != 0
		or coverage.get('exact_once') is not True
		):
		raise ValueError('candidate prediction coverage mismatch')


def _validate_candidate_pairing(
	row: Mapping[str, object],
	*,
	current_reference: Mapping[str, object],
	historical_reference: F3VoxelLabelBudgetReferenceInspection,
	dataset_row: Mapping[str, object],
) -> None:
	"""Require every candidate to share identities with K=6 and MAE."""
	_validate_current_pair(row, current_reference)
	control._validate_paired_identity(
		row,
		reference=historical_reference,
		dataset_row=dataset_row,
		reference_roles=('mae',),
	)


def _prior_rows(
	path: Path,
	config: F3VoxelLabelBudgetMultiHeadConfig,
	identities: Mapping[str, Mapping[str, object]],
) -> tuple[dict[tuple[str, int, str], Mapping[str, object]], list[Path]]:
	if not path.is_file():
		return {}, []
	payload = _read_json(path)
	_validate_manifest(payload, config, identities)
	rows = payload['rows']
	if not isinstance(rows, list):  # Defensive narrowing after validation.
		raise TypeError('multi-head manifest rows must be a list')
	return {_row_key(row): row for row in rows if isinstance(row, Mapping)}, [
		Path(str(item)) for item in payload.get('quarantines', [])
	]


def _write_manifest(
	path: Path,
	config: F3VoxelLabelBudgetMultiHeadConfig,
	rows: Sequence[Mapping[str, object]],
	quarantines: Sequence[Path],
	identities: Mapping[str, Mapping[str, object]],
) -> None:
	ordered = tuple(sorted(rows, key=_row_sort_key))
	payload = {
		'artifact_type': getattr(config, 'run_manifest_type', RUN_MANIFEST_TYPE),
		'schema_version': RUN_SCHEMA_VERSION,
		'dataset_manifest': _identity(config.dataset_manifest),
		'original_run_manifest': _identity(config.original_run_manifest),
		'current_k6_run_manifest': _identity(config.current_k6_run_manifest),
		'candidate_identities': dict(identities),
		'row_count': len(ordered),
		'complete_count': sum(row.get('status') == 'complete' for row in ordered),
		'rows': list(ordered),
		'quarantines': [str(item) for item in quarantines],
		'updated_at_utc': datetime.now(timezone.utc).isoformat(),
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	temp = path.with_name(f'.{path.name}.tmp')
	temp.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temp.replace(path)


def _validate_manifest(
	payload: Mapping[str, object],
	config: F3VoxelLabelBudgetMultiHeadConfig,
	identities: Mapping[str, Mapping[str, object]],
) -> None:
	if (
		payload.get('artifact_type')
		!= getattr(config, 'run_manifest_type', RUN_MANIFEST_TYPE)
		or payload.get('schema_version') != RUN_SCHEMA_VERSION
	):
		raise ValueError('multi-head manifest type/schema mismatch')
	for name, path in (
		('dataset_manifest', config.dataset_manifest),
		('current_k6_run_manifest', config.current_k6_run_manifest),
	):
		control._validate_identity_at(payload.get(name), path, label=name)
	control._validate_identity_at(
		payload.get('original_run_manifest'),
		config.original_run_manifest,
		label='original_run_manifest',
	)
	if payload.get('candidate_identities') != dict(identities):
		raise ValueError('multi-head candidate identity mismatch')
	_validate_owned_rows(payload, config)


def _validate_owned_rows(
	payload: Mapping[str, object], config: F3VoxelLabelBudgetMultiHeadConfig
) -> None:
	"""Reject malformed rows before a prior manifest is indexed by job key.

	Partial manifests are valid because the runner writes atomically after each
	job.  Every row that is present must nevertheless be a unique member of the
	configured candidate/budget/seed matrix, and the aggregate counts must
	faithfully describe that partial set.
	"""
	rows = payload.get('rows')
	if not isinstance(rows, list):
		raise TypeError('multi-head manifest rows must be a list')
	if payload.get('row_count') != len(rows):
		raise ValueError('multi-head manifest row count mismatch')
	complete_count = sum(
		isinstance(row, Mapping) and row.get('status') == 'complete'
		for row in rows
	)
	if payload.get('complete_count') != complete_count:
		raise ValueError('multi-head manifest complete count mismatch')
	expected = {
		(budget_id, seed, item.model_id)
		for item in config.candidates
		for budget_id in config.budgets
		for seed in config.subsample_seeds
	}
	seen: set[tuple[str, int, str]] = set()
	for index, row in enumerate(rows):
		if not isinstance(row, Mapping):
			raise TypeError(f'multi-head manifest row {index} must be an object')
		budget_id = row.get('budget_id')
		subsample_seed = row.get('subsample_seed')
		model_role = row.get('model_role')
		if (
			not isinstance(budget_id, str)
			or type(subsample_seed) is not int
			or not isinstance(model_role, str)
		):
			raise ValueError(
				f'multi-head manifest row {index} has incomplete job identity'
			)
		key = (budget_id, subsample_seed, model_role)
		if key in seen:
			raise ValueError(f'multi-head manifest has duplicate row: {key!r}')
		if key not in expected:
			raise ValueError(f'multi-head manifest has non-matrix row: {key!r}')
		seen.add(key)


def _same_completed_row_except_action(
	prior: Mapping[str, object] | None,
	actual: Mapping[str, object],
) -> bool:
	"""Keep a valid completed row byte-stable during an all-reuse pass."""
	if prior is None or not isinstance(prior.get('action'), str):
		return False
	return {
		key: value for key, value in prior.items() if key != 'action'
	} == {key: value for key, value in actual.items() if key != 'action'}


def _identity(path: Path) -> Mapping[str, object]:
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON object required: {path}')
	return payload


def _job_key(job: VoxelLabelBudgetJob) -> tuple[str, int, str]:
	return (job.budget_id, job.subsample_seed, job.model_role)


def _row_key(row: Mapping[str, object]) -> tuple[str, int, str]:
	return (
		str(row.get('budget_id')),
		int(row.get('subsample_seed', -1)),
		str(row.get('model_role')),
	)


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, int, int]:
	role_order = {'mh_soft_nocons': 0, 'mh_nocons': 0, 'mh_cons010': 1}
	budget, seed, role = _row_key(row)
	return (role_order.get(role, 99), int(budget.removeprefix('cap')), seed)


__all__ = [
	'RUN_MANIFEST_NAME',
	'RUN_MANIFEST_TYPE',
	'F3VoxelLabelBudgetMultiHeadInspection',
	'F3VoxelLabelBudgetMultiHeadRunResult',
	'inspect_f3_lithology_voxel_label_budget_multi_head',
	'load_f3_lithology_voxel_label_budget_multi_head_rows',
	'multi_head_run_manifest_path',
	'run_f3_lithology_voxel_label_budget_multi_head',
]
