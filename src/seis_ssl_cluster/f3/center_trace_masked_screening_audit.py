"""Immutable preflight audit for the center-trace masked original-split screen."""
# ruff: noqa: C901, PLR0912, SLF001, S603

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.embedding.extractor import UNMASKED_ENCODER_INPUT_MODE
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import (
	center_trace_masked_pretraining_validation as center_validation,
)
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.f3.center_trace_masked_pretraining_validation import (
	load_f3_center_trace_masked_pretraining_handoff,
)
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head as decoder_runner,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	inspect_f3_lithology_voxel_label_budget_mae_reference_run,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)

ARTIFACT_TYPE = 'f3_center_trace_masked_original_screening_preflight'
SCHEMA_VERSION = 1
MODEL_ID = 'mh_ctmask010_nocons'
MODEL_TAG = 'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
HARD_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
VARIANT = 'ctmask010_nocons'
_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'workspace_root',
		'source_hard_manifest',
		'hard_full_config',
		'hard_pretraining_handoff',
		'candidate_full_config',
		'candidate_pretraining_handoff',
		'candidate_embeddings_dir',
		'output_path',
	}
)
_EXPECTED_TARGET_FIELDS = {
	'experiment_role': 'multi_head_center_trace_masked_hard_pretext',
	'variant': VARIANT,
	'head_spec': 'multi_resolution_ordered_prototypes_v1',
	'head_ks': [6, 8, 10],
	'target_representation': 'hard_viterbi_labels_v1',
	'objective_semantics': 'center_trace_masked_hmm_path_reconstruction_v1',
	'mask_semantics': 'xy_token_column_full_z_v1',
	'column_fraction': 0.10,
	'selection_policy': 'supervised_valid_xy_columns_round_half_up_leave_one_v1',
	'replacement': 'learned_encoder_mask_token_v1',
	'replacement_initialization': 'normal_std_0p02_train_seed_salted_v1',
	'rng_policy': 'stateless_step_seed_v1',
	'masked_prototype_weight': 0.50,
	'visible_prototype_weight': 0.50,
	'distillation_scope': 'visible_only_v1',
	'supervised_loss': 'structured_hmm_center_trace_masked_hard_v1',
	'consistency_policy': 'disabled_for_center_trace_masked_v1',
}
_EXPECTED_REFERENCE_TAGS = {
	'mae': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
	'current_k6': 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1',
	'mh_nocons': HARD_MODEL_TAG,
}


@dataclass(frozen=True)
class F3CenterTraceMaskedScreeningAuditConfig:
	"""Closed paths needed to construct the immutable screening audit."""

	artifact_root: Path
	workspace_root: Path
	source_hard_manifest: Path
	hard_full_config: Path
	hard_pretraining_handoff: Path
	candidate_full_config: Path
	candidate_pretraining_handoff: Path
	candidate_embeddings_dir: Path
	output_path: Path


@dataclass(frozen=True)
class F3CenterTraceMaskedScreeningAuditResult:
	"""Audit payload and the write/reuse action taken for it."""

	payload: Mapping[str, object]
	output_path: Path
	action: str
	quarantine_path: Path | None


def f3_center_trace_masked_screening_audit_config_from_mapping(
	config: Mapping[str, object],
) -> F3CenterTraceMaskedScreeningAuditConfig:
	"""Resolve the closed center-trace screening-audit YAML schema."""
	if not isinstance(config, Mapping):
		raise TypeError('center-trace screening audit config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown screening audit config keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing screening audit config keys: {sorted(missing)!r}')

	def path(key: str, *, must_exist: bool) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		resolved = Path(value).resolve()
		if must_exist and not resolved.exists():
			raise FileNotFoundError(f'{key} is missing: {resolved}')
		return resolved

	result = F3CenterTraceMaskedScreeningAuditConfig(
		artifact_root=path('artifact_root', must_exist=True),
		workspace_root=path('workspace_root', must_exist=True),
		source_hard_manifest=path('source_hard_manifest', must_exist=True),
		hard_full_config=path('hard_full_config', must_exist=True),
		hard_pretraining_handoff=path('hard_pretraining_handoff', must_exist=True),
		candidate_full_config=path('candidate_full_config', must_exist=True),
		candidate_pretraining_handoff=path(
			'candidate_pretraining_handoff', must_exist=True
		),
		candidate_embeddings_dir=path('candidate_embeddings_dir', must_exist=True),
		output_path=path('output_path', must_exist=False),
	)
	if not result.artifact_root.is_dir() or not result.workspace_root.is_dir():
		raise FileNotFoundError('artifact_root and workspace_root must be directories')
	return result


def load_f3_center_trace_masked_screening_audit_config(
	path: str | Path,
) -> F3CenterTraceMaskedScreeningAuditConfig:
	"""Load a center-trace screening-audit YAML file."""
	return f3_center_trace_masked_screening_audit_config_from_mapping(load_config(path))


def audit_f3_center_trace_masked_screening(
	config: F3CenterTraceMaskedScreeningAuditConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3CenterTraceMaskedScreeningAuditResult:
	"""Build, validate, or immutably reuse the read-only screening audit."""
	payload = _audit_payload(config, git=_clean_git_identity(config.workspace_root))
	if dry_run:
		return F3CenterTraceMaskedScreeningAuditResult(
			payload, config.output_path, 'DRY_RUN', None
		)
	if config.output_path.exists():
		if not config.output_path.is_file():
			raise FileExistsError(
				f'audit output path is not a file: {config.output_path}'
			)
		try:
			existing = load_f3_center_trace_masked_screening_audit(
				config.output_path, revalidate=False
			)
		except (TypeError, ValueError):
			existing = None
		if existing == payload:
			if only_missing:
				return F3CenterTraceMaskedScreeningAuditResult(
					payload, config.output_path, 'REUSE_COMPLETED', None
				)
			raise FileExistsError(
				f'audit already exists; use --only-missing: {config.output_path}'
			)
		if existing is not None and _source_identity_drift(existing, payload):
			raise ValueError('center-trace source/handoff identity drift is an error')
		if not quarantine_invalid:
			raise ValueError(
				'incompatible existing center-trace audit; '
				'use --quarantine-invalid to replace it'
			)
		quarantine = _quarantine_invalid(config.output_path)
	else:
		quarantine = None
	_write_json_atomically(config.output_path, payload)
	return F3CenterTraceMaskedScreeningAuditResult(
		payload, config.output_path, 'WRITTEN', quarantine
	)


def load_f3_center_trace_masked_screening_audit(
	path: str | Path,
	*,
	revalidate: bool = True,
) -> Mapping[str, object]:
	"""Load and, by default, fully revalidate a PASS screening audit."""
	payload = _read_json(Path(path))
	if (
		payload.get('artifact_type') != ARTIFACT_TYPE
		or payload.get('schema_version') != SCHEMA_VERSION
		or payload.get('status') != 'PASS'
	):
		raise ValueError('center-trace screening audit type/schema/status mismatch')
	for key in (
		'candidate',
		'handoff_contract',
		'checkpoint',
		'embedding',
		'valid_mask_parity',
		'hard_baseline_parity',
		'reference_run_manifests',
		'dataset_job_pairing',
		'git',
	):
		if not isinstance(payload.get(key), Mapping):
			raise TypeError(f'center-trace screening audit {key} is missing')
	if revalidate:
		_revalidate_persisted_audit(payload)
	return payload


def validate_f3_center_trace_masked_screening_audit_binding(
	payload: Mapping[str, object],
	*,
	model_id: str,
	model_tag: str,
	pretraining_handoff: Path,
	embeddings_dir: Path,
) -> None:
	"""Bind a PASS audit to the exact candidate admitted to decoder planning."""
	if (
		payload.get('artifact_type') != ARTIFACT_TYPE
		or payload.get('schema_version') != SCHEMA_VERSION
		or payload.get('status') != 'PASS'
	):
		raise ValueError('center-trace screening audit identity mismatch')
	candidate = _mapping(payload.get('candidate'), 'screening audit candidate')
	if candidate.get('model_id') != model_id or candidate.get('model_tag') != model_tag:
		raise ValueError('screening audit candidate identity mismatch')
	_identity_matches(
		candidate.get('pretraining_handoff'),
		pretraining_handoff,
		label='screening audit pretraining handoff',
	)
	embeddings = _mapping(candidate.get('embeddings'), 'screening audit embeddings')
	if Path(str(embeddings.get('root', ''))).resolve() != embeddings_dir.resolve():
		raise ValueError('screening audit candidate embeddings root mismatch')
	for key in ('embeddings', 'valid_tokens', 'metadata'):
		value = embeddings.get(key)
		if isinstance(value, Mapping):
			_identity_matches_live(value, label=f'screening audit {key}')
			continue
		path_key = f'{key}_path'
		sha_key = f'{key}_sha256'
		path = _required_live_path(embeddings.get(path_key), key)
		if embeddings.get(sha_key) != file_sha256(path):
			raise ValueError(f'screening audit {key} identity mismatch')
	if (
		_mapping(payload.get('hard_baseline_parity'), 'hard baseline parity').get(
			'status'
		)
		!= 'PASS'
	):
		raise ValueError('screening audit hard-baseline parity is not PASS')


def _audit_payload(
	config: F3CenterTraceMaskedScreeningAuditConfig,
	*,
	git: Mapping[str, object],
) -> dict[str, object]:
	"""Construct all live assertions before writing one audit byte."""
	target = load_multi_head_target_manifest(config.source_hard_manifest)
	target_hashes = center_validation._multi_head_target_hashes(target)
	hard = resolve_strat_hmm_pretext_config(load_config(config.hard_full_config))
	candidate = resolve_strat_hmm_pretext_config(
		load_config(config.candidate_full_config)
	)
	_validate_training_identity(config, hard=hard, candidate=candidate)
	_validate_target_lineage(target=target, target_hashes=target_hashes)
	hard_handoff = hard_validation.load_f3_multi_head_pretraining_handoff(
		config.hard_pretraining_handoff
	)
	center_handoff = load_f3_center_trace_masked_pretraining_handoff(
		config.candidate_pretraining_handoff
	)
	_validate_hard_handoff(
		config,
		handoff=hard_handoff,
		target_hashes=target_hashes,
	)
	parity = center_validation._hard_config_parity(hard, candidate)
	_validate_handoff_parity(center_handoff, parity)
	checkpoint = _checkpoint_evidence(
		config,
		candidate=candidate,
		handoff=center_handoff,
		target_hashes=target_hashes,
	)
	embedding = _embedding_evidence(
		config,
		checkpoint=checkpoint,
		target_hashes=target_hashes,
	)
	references, pairing = _reference_and_pairing_evidence(config)
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': 'PASS',
		'artifact_root': str(config.artifact_root),
		'workspace_root': str(config.workspace_root),
		'git': dict(git),
		'source_hard_manifest': _identity(config.source_hard_manifest),
		'candidate': {
			'model_id': MODEL_ID,
			'model_tag': MODEL_TAG,
			'full_config': _identity(config.candidate_full_config),
			'pretraining_handoff': _identity(config.candidate_pretraining_handoff),
			'embeddings': embedding,
		},
		'handoff_contract': {
			'artifact_type': center_handoff['artifact_type'],
			'schema_version': center_handoff['schema_version'],
			'status': center_handoff['status'],
			'model_tag': center_handoff['model_tag'],
			'variant': center_handoff['variant'],
			'objective_semantics': center_handoff['targets']['objective_semantics'],
			'mask_semantics': center_handoff['targets']['mask_semantics'],
			'supervised_loss': center_handoff['targets']['supervised_loss'],
			'replacement': center_handoff['targets']['replacement'],
			'replacement_initialization': center_handoff['targets'][
				'replacement_initialization'
			],
			'target_manifest': _identity(config.source_hard_manifest),
			'per_head_target_hashes': center_handoff['targets'][
				'per_head_target_hashes'
			],
			'initial_student_state_sha256': center_handoff['targets'][
				'initial_student_state_sha256'
			],
			'initial_head_state_sha256': center_handoff['targets'][
				'initial_head_state_sha256'
			],
			'initial_spatial_context_state_sha256': center_handoff['targets'][
				'initial_spatial_context_state_sha256'
			],
		},
		'checkpoint': checkpoint,
		'embedding': embedding,
		'valid_mask_parity': embedding['canonical_valid_token_identities'],
		'hard_baseline_parity': {
			'status': 'PASS',
			'config': _identity(config.hard_full_config),
			'handoff': _identity(config.hard_pretraining_handoff),
			'config_parity': parity,
			'target_manifest': _identity(config.source_hard_manifest),
			'per_head_target_hashes': target_hashes,
		},
		'reference_run_manifests': references,
		'dataset_job_pairing': pairing,
	}


def _validate_training_identity(
	config: F3CenterTraceMaskedScreeningAuditConfig,
	*,
	hard: Mapping[str, object],
	candidate: Mapping[str, object],
) -> None:
	if _model_tag(hard) != HARD_MODEL_TAG or _model_tag(candidate) != MODEL_TAG:
		raise ValueError('hard/candidate center-trace model tag mismatch')
	hard_target = _mapping(hard.get('pseudo_targets'), 'hard pseudo targets')
	candidate_target = _mapping(
		candidate.get('pseudo_targets'), 'candidate pseudo targets'
	)
	if (
		Path(str(hard_target.get('manifest', ''))).resolve()
		!= config.source_hard_manifest
	):
		raise ValueError('hard config target manifest mismatch')
	if (
		Path(str(candidate_target.get('manifest', ''))).resolve()
		!= config.source_hard_manifest
	):
		raise ValueError('center-trace config target manifest mismatch')
	center_identity = _mapping(candidate.get('identity'), 'candidate identity')
	scientific = _mapping(
		center_identity.get('scientific_identity'), 'candidate scientific identity'
	)
	for key, expected in _EXPECTED_TARGET_FIELDS.items():
		if scientific.get(key) != expected:
			raise ValueError(f'center-trace scientific identity mismatch: {key}')
	if candidate_target.get('target_representation') != 'hard_viterbi_labels_v1':
		raise ValueError('center-trace target representation mismatch')


def _validate_target_lineage(
	*,
	target: Mapping[str, object],
	target_hashes: Mapping[str, object],
) -> None:
	if target.get('head_ks') != [6, 8, 10]:
		raise ValueError('center-trace target must be the immutable K6/K8/K10 manifest')
	if target.get('artifact_type') != 'strat_hmm_multi_head_target_manifest':
		raise ValueError('center-trace target artifact type mismatch')
	if not target_hashes:
		raise ValueError('center-trace target head hashes are missing')


def _validate_hard_handoff(
	config: F3CenterTraceMaskedScreeningAuditConfig,
	*,
	handoff: Mapping[str, object],
	target_hashes: Mapping[str, object],
) -> None:
	identity = _mapping(handoff.get('stratigraphy_pretext'), 'hard handoff identity')
	if handoff.get('model_tag') != HARD_MODEL_TAG:
		raise ValueError('hard baseline handoff model tag mismatch')
	if (
		Path(str(identity.get('target_manifest_path', ''))).resolve()
		!= config.source_hard_manifest
	):
		raise ValueError('hard baseline handoff target path mismatch')
	if identity.get('target_manifest_sha256') != file_sha256(
		config.source_hard_manifest
	):
		raise ValueError('hard baseline handoff target SHA-256 mismatch')
	if identity.get('per_head_target_sha256') != target_hashes:
		raise ValueError('hard baseline handoff target head hashes mismatch')


def _validate_handoff_parity(
	handoff: Mapping[str, object], parity: Mapping[str, object]
) -> None:
	targets = _mapping(handoff.get('targets'), 'center-trace handoff targets')
	recorded = _mapping(
		targets.get('hard_baseline_config_parity'),
		'center-trace handoff baseline parity',
	)
	if recorded.get('status') != 'PASS':
		raise ValueError('center-trace handoff baseline parity is not PASS')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	):
		if _mapping(recorded['hard_runtime'], 'handoff hard runtime').get(
			key
		) != _mapping(parity['hard_runtime'], 'live hard runtime').get(key):
			raise ValueError(f'center-trace handoff baseline parity is stale: {key}')
	if _mapping(recorded['candidate_runtime'], 'handoff candidate runtime') != _mapping(
		parity['candidate_runtime'], 'live candidate runtime'
	):
		raise ValueError('center-trace handoff candidate runtime parity is stale')


def _checkpoint_evidence(
	config: F3CenterTraceMaskedScreeningAuditConfig,
	*,
	candidate: Mapping[str, object],
	handoff: Mapping[str, object],
	target_hashes: Mapping[str, object],
) -> dict[str, object]:
	record = _mapping(handoff.get('checkpoint'), 'center-trace handoff checkpoint')
	selected_path = _required_live_path(record.get('path'), 'selected checkpoint')
	latest_path = _required_live_path(record.get('latest_path'), 'latest checkpoint')
	if record.get('sha256') != file_sha256(selected_path):
		raise ValueError('selected checkpoint SHA-256 mismatch')
	if record.get('latest_sha256') != file_sha256(latest_path):
		raise ValueError('latest checkpoint SHA-256 mismatch')
	selected = _torch_mapping(selected_path, 'selected checkpoint')
	latest = _torch_mapping(latest_path, 'latest checkpoint')
	validate_stratigraphy_checkpoint_payload(selected, expected_config=candidate)
	validate_stratigraphy_checkpoint_payload(latest, expected_config=candidate)
	selected_identity = _mapping(
		selected.get('stratigraphy_checkpoint'), 'selected checkpoint identity'
	)
	latest_identity = _mapping(
		latest.get('stratigraphy_checkpoint'), 'latest checkpoint identity'
	)
	if selected_identity.get('schema_version') != 7:
		raise ValueError('selected checkpoint schema must be 7')
	if selected_identity.get('target_manifest_sha256') != file_sha256(
		config.source_hard_manifest
	):
		raise ValueError('selected checkpoint target SHA-256 mismatch')
	if selected_identity.get('per_head_targets') != target_hashes:
		raise ValueError('selected checkpoint per-head target mismatch')
	if selected_identity.get('model_tag') != MODEL_TAG:
		raise ValueError('selected checkpoint model tag mismatch')
	selection_evidence = hard_validation._validate_best_selection(
		selected, latest, variant='center_trace_masked'
	)
	event = selection_evidence['selected']
	for key, handoff_key in (
		('checkpoint_kind', 'selected_checkpoint_kind'),
		('epoch', 'selected_epoch'),
		('global_step', 'selected_global_step'),
		('loss', 'selected_loss'),
	):
		if event.get(key) != record.get(handoff_key):
			raise ValueError(f'selected checkpoint selection event mismatch: {key}')
	history_path = selected_path.parent / 'checkpoint_selection_history.csv'
	if not history_path.is_file():
		raise FileNotFoundError(
			f'selected checkpoint selection history is missing: {history_path}'
		)
	with history_path.open(newline='', encoding='utf-8') as handle:
		history_count = sum(1 for _ in csv.DictReader(handle))
	if record.get('selection_history_sha256') != selection_evidence['sha256']:
		raise ValueError('selected checkpoint selection history SHA-256 mismatch')
	if (
		record.get('selection_history_event_count') != history_count
		or history_count != selection_evidence['event_count']
	):
		raise ValueError('selected checkpoint selection event count mismatch')
	if latest_identity.get('schema_version') != 7:
		raise ValueError('latest checkpoint schema must be 7')
	spatial = selected.get('spatial_context_state_dict')
	if not isinstance(spatial, Mapping) or set(spatial) != {'replacement_token'}:
		raise ValueError('selected checkpoint replacement-token state is missing')
	if selected.get('spatial_context_state_sha256') != selected_identity.get(
		'spatial_context_state_sha256'
	):
		raise ValueError('selected checkpoint replacement-token hash mismatch')
	return {
		'path': str(selected_path),
		'sha256': record['sha256'],
		'latest_path': str(latest_path),
		'latest_sha256': record['latest_sha256'],
		'schema_version': selected_identity['schema_version'],
		'selected_checkpoint_kind': record['selected_checkpoint_kind'],
		'selected_epoch': record['selected_epoch'],
		'selected_global_step': record['selected_global_step'],
		'selected_loss': record['selected_loss'],
		'selection_event': dict(event),
		'selection_history_sha256': record['selection_history_sha256'],
		'selection_history_event_count': record['selection_history_event_count'],
		'scientific_identity_sha256': selected_identity['scientific_identity_sha256'],
		'student_state_sha256': selected_identity['student_state_sha256'],
		'head_state_sha256': selected_identity['stratigraphy_state_sha256'],
		'replacement_token_state_sha256': selected_identity[
			'spatial_context_state_sha256'
		],
		'initial_replacement_token_state_sha256': selected_identity[
			'initial_spatial_context_state_sha256'
		],
	}


def _embedding_evidence(
	config: F3CenterTraceMaskedScreeningAuditConfig,
	*,
	checkpoint: Mapping[str, object],
	target_hashes: Mapping[str, object],
) -> dict[str, object]:
	files = output_paths(config.candidate_embeddings_dir, 'f3_facies_benchmark')
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		if not path.is_file():
			raise FileNotFoundError(path)
	metadata = _read_json(files.metadata)
	execution_path = (
		config.candidate_embeddings_dir / 'embedding_extraction_execution.json'
	)
	execution = _read_json(execution_path)
	if (
		execution.get('artifact_type') != 'embedding_extraction_execution'
		or execution.get('schema_version') != 1
		or execution.get('encoder_input_mode') != UNMASKED_ENCODER_INPUT_MODE
		or execution.get('survey_count') != 1
		or execution.get('fresh', 0) + execution.get('reuse', 0)
			!= execution.get('survey_count')
	):
		raise ValueError('embedding extraction is not explicitly unmasked')
	selected = Path(str(checkpoint['path'])).resolve()
	if Path(str(metadata.get('checkpoint_path', ''))).resolve() != selected:
		raise ValueError(
			'embedding extraction checkpoint does not equal selected checkpoint'
		)
	if metadata.get('checkpoint_sha256') != checkpoint['sha256']:
		raise ValueError('embedding extraction checkpoint SHA-256 mismatch')
	stratigraphy = _mapping(metadata.get('stratigraphy_pretext'), 'embedding identity')
	if stratigraphy.get('model_tag') != MODEL_TAG:
		raise ValueError('embedding model tag mismatch')
	for key, expected in _EXPECTED_TARGET_FIELDS.items():
		if key in stratigraphy and stratigraphy[key] != expected:
			raise ValueError(f'center-trace embedding identity mismatch: {key}')
	if stratigraphy.get('target_manifest_sha256') != file_sha256(
		config.source_hard_manifest
	):
		raise ValueError('embedding target manifest SHA-256 mismatch')
	if stratigraphy.get('per_head_target_sha256') != target_hashes:
		raise ValueError('embedding per-head target hash mismatch')
	if (
		stratigraphy.get('checkpoint_student_state_sha256')
		!= checkpoint['student_state_sha256']
	):
		raise ValueError('embedding checkpoint student-state identity mismatch')
	# The extractor loads only model_state_dict and never spatial_context_state_dict.
	# Binding the extracted student state makes masked pretraining diagnostics and
	# masked forward outputs unusable as an accidental embedding source.
	if 'checkpoint_spatial_context_state_sha256' not in stratigraphy:
		raise ValueError('embedding extraction lacks center-trace checkpoint lineage')
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	if (
		embeddings.shape != (76, 113, 32, 384)
		or embeddings.dtype != np.float16
		or valid.shape != (76, 113, 32)
		or valid.dtype != np.bool_
		or int(valid.sum()) <= 0
	):
		raise ValueError('center-trace embedding shape/dtype contract mismatch')
	if not _finite_valid_embeddings(embeddings, valid):
		raise ValueError('center-trace embedding valid values are nonfinite')
	valid_identity = _canonical_valid_mask_identities(config.artifact_root)
	valid_sha = file_sha256(files.valid_tokens)
	if any(value['sha256'] != valid_sha for value in valid_identity.values()):
		raise ValueError(
			'center-trace valid-token mask differs from a canonical baseline'
		)
	return {
		'root': str(config.candidate_embeddings_dir),
		'metadata_path': str(files.metadata),
		'metadata_sha256': file_sha256(files.metadata),
		'embeddings_path': str(files.embeddings),
		'embeddings_sha256': file_sha256(files.embeddings),
		'valid_tokens_path': str(files.valid_tokens),
		'valid_tokens_sha256': valid_sha,
		'embeddings_shape': list(embeddings.shape),
		'embeddings_dtype': str(embeddings.dtype),
		'valid_tokens_shape': list(valid.shape),
		'valid_tokens_dtype': str(valid.dtype),
		'finite_valid_count': int(valid.sum()),
		'encoder_input_mode': execution['encoder_input_mode'],
		'execution_path': str(execution_path),
		'execution_sha256': file_sha256(execution_path),
		'canonical_valid_token_identities': valid_identity,
	}


def _reference_and_pairing_evidence(
	config: F3CenterTraceMaskedScreeningAuditConfig,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
	hard_decoder_path = (
		config.workspace_root
		/ (
			'experiments/f3/facies_benchmark_v1/'
			'95_strat_hmm_multi_head_k6810_low_label_v1'
		)
		/ '01_run_multi_head_voxel_label_budget.yaml'
	)
	if not hard_decoder_path.is_file():
		raise FileNotFoundError(hard_decoder_path)
	decoder_config = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(hard_decoder_path)
	)
	if decoder_config.dataset_manifest != _canonical_dataset_manifest(
		config.artifact_root
	):
		raise ValueError('decoder dataset manifest is not the canonical original split')
	dataset_rows = decoder_runner._dataset_rows(decoder_config)
	reference = inspect_f3_lithology_voxel_label_budget_mae_reference_run(
		decoder_config.dataset_manifest,
		decoder_config.original_run_manifest,
		include_historical_m1=False,
	)
	current = decoder_runner._current_k6_rows(decoder_config, dataset_rows)
	hard_manifest_path = decoder_runner.multi_head_run_manifest_path(decoder_config)
	hard_manifest = _read_json(hard_manifest_path)
	if (
		hard_manifest.get('artifact_type') != decoder_runner.RUN_MANIFEST_TYPE
		or hard_manifest.get('schema_version') != decoder_runner.RUN_SCHEMA_VERSION
	):
		raise ValueError('hard multi-head decoder manifest type/schema mismatch')
	hard_rows = hard_manifest.get('rows')
	if not isinstance(hard_rows, list) or not all(
		isinstance(row, Mapping) for row in hard_rows
	):
		raise TypeError('hard multi-head decoder manifest rows are invalid')
	hard_primary = {
		(str(row['budget_id']), int(row['subsample_seed'])): row
		for row in hard_rows
		if row.get('model_role') == 'mh_nocons'
	}
	if len(hard_primary) != 15:
		raise ValueError('hard mh_nocons decoder matrix must contain 15 rows')
	if len(reference.jobs) != 15 or len(current) != 15:
		raise ValueError('reference original-split matrices must contain 15 rows')
	for key, row in current.items():
		decoder_runner.control._validate_paired_identity(
			row,
			reference=reference,
			dataset_row=dataset_rows[key],
			reference_roles=('mae',),
		)
		decoder_runner._validate_candidate_pairing(
			hard_primary[key],
			current_reference=row,
			historical_reference=reference,
			dataset_row=dataset_rows[key],
		)
	return (
		{
			'dataset_manifest': _identity(decoder_config.dataset_manifest),
			'original_run_manifest': _identity(decoder_config.original_run_manifest),
			'current_k6_run_manifest': _identity(
				decoder_config.current_k6_run_manifest
			),
			'hard_multi_head_run_manifest': _identity(
				decoder_runner.multi_head_run_manifest_path(decoder_config)
			),
			'hard_decoder_config': _identity(hard_decoder_path),
		},
		{
			'roles': ['mae', 'm1_current_k6', 'mh_nocons', MODEL_ID],
			'budgets': list(decoder_config.budgets),
			'subsample_seeds': list(decoder_config.subsample_seeds),
			'decoder_seeds': {
				str(seed): decoder_config.decoder_seed(seed)
				for seed in decoder_config.subsample_seeds
			},
			'candidate_job_count': 15,
			'candidate_decoder_seed_policy': '42000 + subsample_seed',
			'dataset_row_count': len(dataset_rows),
			'pair_identity_keys': list(decoder_runner.control.PAIR_IDENTITY_KEYS),
		},
	)


def _canonical_dataset_manifest(artifact_root: Path) -> Path:
	return (
		artifact_root
		/ 'lithology/f3/facies_benchmark_v1/voxel_label_budget_v1'
		/ 'original_split/voxel_label_budget_dataset_manifest.json'
	)


def _canonical_valid_mask_identities(artifact_root: Path) -> dict[str, dict[str, str]]:
	result: dict[str, dict[str, str]] = {}
	for role, tag in _EXPECTED_REFERENCE_TAGS.items():
		path = output_paths(
			artifact_root / 'embeddings/f3/facies_benchmark_v1' / tag / 'overlap_x16',
			'f3_facies_benchmark',
		).valid_tokens
		if not path.is_file():
			raise FileNotFoundError(
				f'{role} canonical valid-token artifact is missing: {path}'
			)
		result[role] = {'path': str(path), 'sha256': file_sha256(path)}
	if len({item['sha256'] for item in result.values()}) != 1:
		raise ValueError('canonical valid-token masks are not bitwise identical')
	return result


def _source_identity_drift(
	existing: Mapping[str, object], current: Mapping[str, object]
) -> bool:
	for key in (
		'source_hard_manifest',
		'reference_run_manifests',
	):
		if existing.get(key) != current.get(key):
			return True
	for section in ('candidate', 'hard_baseline_parity'):
		left = existing.get(section)
		right = current.get(section)
		if isinstance(left, Mapping) and isinstance(right, Mapping):
			for key in (
				'full_config',
				'pretraining_handoff',
				'handoff',
				'config',
				'target_manifest',
			):
				if key in left and left.get(key) != right.get(key):
					return True
	return False


def _revalidate_persisted_audit(payload: Mapping[str, object]) -> None:
	"""Recompute every live assertion represented by a persisted audit."""
	config = _config_from_persisted_payload(payload)
	expected = _audit_payload(
		config,
		git=_mapping(payload.get('git'), 'persisted audit git identity'),
	)
	if dict(payload) != expected:
		raise ValueError('persisted center-trace screening audit is stale')


def _config_from_persisted_payload(
	payload: Mapping[str, object],
) -> F3CenterTraceMaskedScreeningAuditConfig:
	"""Recover the closed audit inputs recorded in the persisted payload."""
	artifact_root = _required_directory(payload.get('artifact_root'), 'artifact_root')
	workspace_root = _required_directory(
		payload.get('workspace_root'), 'workspace_root'
	)
	candidate = _mapping(payload.get('candidate'), 'persisted audit candidate')
	parity = _mapping(
		payload.get('hard_baseline_parity'), 'persisted audit hard baseline parity'
	)
	config_identity = _mapping(parity.get('config'), 'persisted hard config')
	handoff_identity = _mapping(parity.get('handoff'), 'persisted hard handoff')
	candidate_config_identity = _mapping(
		candidate.get('full_config'), 'persisted candidate config'
	)
	candidate_handoff_identity = _mapping(
		candidate.get('pretraining_handoff'), 'persisted candidate handoff'
	)
	source_identity = _mapping(
		payload.get('source_hard_manifest'), 'persisted source target manifest'
	)
	embeddings = _mapping(
		candidate.get('embeddings'), 'persisted candidate embeddings'
	)
	source = _identity_path(source_identity, 'source_hard_manifest')
	hard_config = _identity_path(config_identity, 'hard_full_config')
	hard_handoff = _identity_path(handoff_identity, 'hard_pretraining_handoff')
	candidate_config = _identity_path(
		candidate_config_identity, 'candidate_full_config'
	)
	candidate_handoff = _identity_path(
		candidate_handoff_identity, 'candidate_pretraining_handoff'
	)
	embedding_root = _required_directory(
		embeddings.get('root'), 'candidate_embeddings_dir'
	)
	return F3CenterTraceMaskedScreeningAuditConfig(
		artifact_root=artifact_root,
		workspace_root=workspace_root,
		source_hard_manifest=source,
		hard_full_config=hard_config,
		hard_pretraining_handoff=hard_handoff,
		candidate_full_config=candidate_config,
		candidate_pretraining_handoff=candidate_handoff,
		candidate_embeddings_dir=embedding_root,
		output_path=artifact_root / '.persisted_center_trace_masked_audit.json',
	)


def _clean_git_identity(workspace_root: Path) -> Mapping[str, object]:
	git = shutil.which('git')
	if git is None:
		raise RuntimeError('git executable is unavailable')
	try:
		sha = subprocess.run(
			(git, 'rev-parse', 'HEAD'),
			cwd=workspace_root,
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
		status = subprocess.run(
			(git, 'status', '--porcelain'),
			cwd=workspace_root,
			check=True,
			capture_output=True,
			text=True,
		).stdout
	except (OSError, subprocess.CalledProcessError) as error:
		raise RuntimeError('unable to record center-trace audit git state') from error
	if len(sha) != 40:
		raise ValueError('audit git SHA is invalid')
	return {'git_sha': sha, 'dirty': bool(status.strip())}


def _required_live_path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label}.path is missing')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	return path


def _required_directory(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty directory path')
	path = Path(value).resolve()
	if not path.is_dir():
		raise FileNotFoundError(f'{label} is missing: {path}')
	return path


def _identity_path(value: Mapping[str, object], label: str) -> Path:
	path = _required_live_path(value.get('path'), label)
	if value.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} identity is stale')
	return path


def _torch_mapping(path: Path, label: str) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a mapping')
	return payload


def _finite_valid_embeddings(embeddings: np.ndarray, valid: np.ndarray) -> bool:
	for index in range(embeddings.shape[0]):
		if (
			valid[index].any()
			and not np.isfinite(embeddings[index][valid[index]]).all()
		):
			return False
	return True


def _model_tag(training: Mapping[str, object]) -> str:
	identity = _mapping(training.get('identity'), 'training identity')
	value = identity.get('model_tag')
	if not isinstance(value, str) or not value:
		raise TypeError('training model tag is missing')
	return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _identity(path: Path) -> dict[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _identity_matches(value: object, path: Path, *, label: str) -> None:
	identity = _mapping(value, label)
	if identity.get('path') != str(path) or identity.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} identity mismatch')


def _identity_matches_live(value: Mapping[str, object], *, label: str) -> None:
	path = _required_live_path(value.get('path'), label)
	if value.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} identity mismatch')


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON object required: {path}')
	return payload


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.tmp')
	temporary.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temporary.replace(path)


def _quarantine_invalid(path: Path) -> Path:
	timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
	quarantine = path.with_name(f'{path.name}.quarantine.{timestamp}.invalid')
	index = 0
	while quarantine.exists():
		index += 1
		quarantine = path.with_name(
			f'{path.name}.quarantine.{timestamp}.{index}.invalid'
		)
	path.replace(quarantine)
	return quarantine


__all__ = [
	'ARTIFACT_TYPE',
	'MODEL_ID',
	'MODEL_TAG',
	'F3CenterTraceMaskedScreeningAuditConfig',
	'F3CenterTraceMaskedScreeningAuditResult',
	'audit_f3_center_trace_masked_screening',
	'f3_center_trace_masked_screening_audit_config_from_mapping',
	'load_f3_center_trace_masked_screening_audit',
	'load_f3_center_trace_masked_screening_audit_config',
	'validate_f3_center_trace_masked_screening_audit_binding',
]
