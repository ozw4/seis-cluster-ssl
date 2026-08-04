"""Read-only start audit for the center-trace masked six-split study."""
# ruff: noqa: C901, CPY001, E501, PLR0912, PLR0913, S603

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_center_trace_masked_split import (
	BASELINE_MODEL_ID,
	BASELINE_MODEL_TAG,
	BUDGETS,
	MONITORED_CLASS_IDS,
	MONITORED_CLASS_METRICS,
	PRIMARY_METRICS,
	SPLIT_IDS,
	F3CenterTraceMaskedSixSplitConfig,
	f3_lithology_voxel_label_budget_center_trace_masked_split_config_from_mapping,
)
from seis_ssl_cluster.config.io import load_config
from seis_ssl_cluster.embedding.extractor import UNMASKED_ENCODER_INPUT_MODE
from seis_ssl_cluster.embedding.writer import (
	EmbeddingOutputPaths,
	file_sha256,
	output_paths,
)

ARTIFACT_TYPE = 'f3_center_trace_masked_six_split_preflight'
SCHEMA_VERSION = 1
ORIGINAL_HANDOFF_TYPE = 'f3_center_trace_masked_original_screening_handoff'
CANDIDATE_HANDOFF_TYPE = 'f3_center_trace_masked_pretraining_handoff'
BASELINE_HANDOFF_TYPE = 'f3_multi_head_pretraining_handoff'
EXPECTED_OBJECTIVE = 'center_trace_masked_hmm_path_reconstruction_v1'
EXPECTED_TARGET_REPRESENTATION = 'hard_viterbi_labels_v1'
EXPECTED_HEAD_KS = [6, 8, 10]
EXPECTED_EMBEDDINGS_SHAPE = (76, 113, 32, 384)
EXPECTED_VALID_TOKENS_SHAPE = (76, 113, 32)


@dataclass(frozen=True)
class F3CenterTraceMaskedSixSplitAuditResult:
	"""Audit payload and the explicit output action."""

	payload: Mapping[str, object]
	output_path: Path
	action: str
	quarantine_path: Path | None


def f3_center_trace_masked_six_split_audit_config_from_mapping(
	config: Mapping[str, object],
) -> F3CenterTraceMaskedSixSplitConfig:
	"""Resolve the six-split audit YAML through its strict config contract."""
	return f3_lithology_voxel_label_budget_center_trace_masked_split_config_from_mapping(
		config
	)


def load_f3_center_trace_masked_six_split_audit_config(
	path: str | Path,
) -> F3CenterTraceMaskedSixSplitConfig:
	"""Load and resolve a six-split audit YAML file."""
	return f3_center_trace_masked_six_split_audit_config_from_mapping(load_config(path))


def audit_f3_center_trace_masked_six_split(
	config: F3CenterTraceMaskedSixSplitConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3CenterTraceMaskedSixSplitAuditResult:
	"""Validate all live inputs, then write only an immutable preflight artifact."""
	payload = _audit_payload(config, git=_git_provenance())
	if dry_run:
		return F3CenterTraceMaskedSixSplitAuditResult(
			payload, config.audit_output_path, 'DRY_RUN', None
		)

	output = config.audit_output_path
	if output.exists():
		if not output.is_file():
			raise FileExistsError(f'audit output is not a file: {output}')
		try:
			existing = load_f3_center_trace_masked_six_split_audit(
				output, revalidate=False
			)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == payload:
			if only_missing:
				return F3CenterTraceMaskedSixSplitAuditResult(
					payload, output, 'REUSE_COMPLETED', None
				)
			raise FileExistsError(
				'audit output already exists; use --only-missing: '
				f'{output}'
			)
		if not quarantine_invalid:
			raise ValueError(
				'existing six-split audit is stale or invalid; '
				'use --quarantine-invalid to replace it'
			)
		quarantine = _quarantine_invalid(output)
	else:
		quarantine = None
	_atomic_json(output, payload)
	return F3CenterTraceMaskedSixSplitAuditResult(
		payload, output, 'WRITTEN', quarantine
	)


def load_f3_center_trace_masked_six_split_audit(
	path: str | Path,
	*,
	revalidate: bool = True,
) -> Mapping[str, object]:
	"""Load a complete PASS audit and optionally replay its live inputs."""
	payload = _mapping(_read_json(Path(path)), 'six-split audit')
	_required_audit_structure(payload)
	if revalidate:
		_revalidate_persisted_audit(payload)
	return payload


def _audit_payload(
	config: F3CenterTraceMaskedSixSplitConfig,
	*,
	git: Mapping[str, object],
) -> dict[str, object]:
	"""Construct all assertions before the one-byte output boundary."""
	hashes: dict[Path, str] = {}
	sources = _validate_source_files(config, hashes)
	original = _validate_original_split_handoff(config, hashes)
	lineage = _validate_candidate_lineage(config, hashes)
	experiment96 = _validate_experiment96_evidence(
		config,
		canonical_valid_token_sha256=lineage['candidate_valid_tokens_sha256'],
		hashes=hashes,
	)
	start = _validate_six_split_start_state(config)
	payload = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': 'PASS',
		'config': _config_payload(config),
		'split_ids': list(config.split_ids),
		'budgets': list(config.budgets),
		'primary_model_roles': list(config.primary_model_roles),
		'primary_model_tags': dict(config.primary_model_tags),
		'primary_metrics': list(PRIMARY_METRICS),
		'monitored_class_ids': list(MONITORED_CLASS_IDS),
		'monitored_class_metrics': list(MONITORED_CLASS_METRICS),
		'primary_matrix_row_count': config.primary_matrix_row_count,
		'future_candidate_jobs': config.future_candidate_jobs,
		'future_new_baseline_jobs': config.future_new_baseline_jobs,
		'historical_baseline_rows': config.historical_baseline_rows,
		'future_new_scientific_jobs': config.future_new_scientific_jobs,
		'scientific_jobs_executed': 0,
		'smoke_jobs_executed': 0,
		'source_files': sources,
		'evidence': {
			'original_split_handoff': original,
			'candidate_lineage': lineage,
			'experiment96': experiment96,
			'six_split_start': start,
		},
		'git': dict(git),
	}
	_validate_json_tree(payload, context='six-split audit')
	return payload


def _validate_source_files(
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> dict[str, object]:
	result: dict[str, object] = {}
	for name, path in config.source_paths.items():
		expected = config.source_identities[name]
		if Path(expected['path']).resolve(strict=False) != path.resolve(strict=False):
			raise ValueError(f'source identity path mismatch: {name}')
		actual = _live_hash(path, hashes, label=name)
		if actual != expected['sha256']:
			raise ValueError(f'source SHA-256 drift: {name}')
		result[name] = {
			'path': str(path),
			'byte_size': path.stat().st_size,
			'sha256': actual,
		}
	return result


def _validate_original_split_handoff(
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> dict[str, object]:
	payload = _read_json(config.original_split_handoff)
	if payload.get('artifact_type') != ORIGINAL_HANDOFF_TYPE:
		raise ValueError('original-split handoff artifact type mismatch')
	if payload.get('status') != 'PASS':
		raise ValueError('original-split handoff status must be PASS')
	if payload.get('formal_status') != 'CTMASK_ORIGINAL_GO':
		raise ValueError('original-split handoff formal status must be GO')
	follow_up = _mapping(
		payload.get('six_split_follow_up'), 'original-split six_split_follow_up'
	)
	if follow_up.get('ready') is not True:
		raise ValueError('original-split six_split_follow_up.ready must be true')
	if follow_up.get('six_split_jobs_executed') != 0:
		raise ValueError('original-split six-split job counter must be zero')
	if follow_up.get('scientific_jobs_executed') != 0:
		raise ValueError('original-split six-split scientific counter must be zero')
	if payload.get('six_split_jobs_executed') != 0:
		raise ValueError('original-split top-level six-split counter must be zero')
	validation = _mapping(
		payload.get('candidate_job_live_validation'),
		'original-split candidate live validation',
	)
	if (
		validation.get('status') != 'PASS'
		or validation.get('expected_count') != 15
		or validation.get('validated_count') != 15
	):
		raise ValueError('original-split candidate live validation is not exactly 15/15 PASS')
	provenance = _mapping(
		payload.get('candidate_provenance'), 'original-split candidate provenance'
	)
	if provenance.get('model_id') != config.candidate_model_id:
		raise ValueError('original-split candidate model ID mismatch')
	if provenance.get('model_tag') != config.candidate_model_tag:
		raise ValueError('original-split candidate model tag mismatch')
	if payload.get('candidate_model_id') not in {None, config.candidate_model_id}:
		raise ValueError('original-split candidate model ID mismatch')
	if payload.get('candidate_model_tag') not in {None, config.candidate_model_tag}:
		raise ValueError('original-split candidate model tag mismatch')

	pretraining = _mapping(provenance.get('pretraining_handoff'), 'candidate handoff')
	pretraining_path = _resolve_recorded_path(
		pretraining.get('path'), config.artifact_root
	)
	if pretraining_path != config.candidate_pretraining_handoff:
		raise ValueError('original-split candidate handoff path mismatch')
	pretraining_identity = _validate_recorded_identity(
		pretraining,
		config.candidate_pretraining_handoff,
		hashes,
		label='candidate handoff',
		path_root=config.artifact_root,
	)
	candidate_files = output_paths(
		config.candidate_embeddings_dir, 'f3_facies_benchmark'
	)
	checkpoint_record = _mapping(
		provenance.get('best_checkpoint'), 'candidate best checkpoint'
	)
	metadata_record = _mapping(
		provenance.get('embedding_metadata'), 'candidate embedding metadata'
	)
	embeddings_record = _mapping(provenance.get('embeddings'), 'candidate embeddings')
	valid_record = _mapping(provenance.get('valid_tokens'), 'candidate valid tokens')
	checkpoint_path = _recorded_path_from_handoff(
		config.candidate_pretraining_handoff, 'checkpoint'
	)
	for record, expected_path, label in (
		(checkpoint_record, checkpoint_path, 'candidate best checkpoint'),
		(metadata_record, candidate_files.metadata, 'candidate embedding metadata'),
		(embeddings_record, candidate_files.embeddings, 'candidate embeddings'),
		(valid_record, candidate_files.valid_tokens, 'candidate valid tokens'),
	):
		_validate_recorded_identity(
			record,
			expected_path,
			hashes,
			label=label,
			path_root=config.artifact_root,
		)
	run_record = _mapping(
		payload.get('candidate_run_manifest'), 'candidate original run manifest'
	)
	run_path = _resolve_recorded_path(run_record.get('path'), config.artifact_root)
	run_identity = _validate_recorded_identity(
		run_record,
		run_path,
		hashes,
		label='candidate original run manifest',
		path_root=config.artifact_root,
	)
	run_payload = _read_json(run_path)
	_validate_candidate_original_run_manifest(run_payload, config, hashes)
	return {
		'handoff': _identity(config.original_split_handoff, hashes),
		'formal_status': payload['formal_status'],
		'six_split_follow_up': dict(follow_up),
		'candidate_job_live_validation': dict(validation),
		'candidate': {
			'model_id': config.candidate_model_id,
			'model_tag': config.candidate_model_tag,
			'pretraining_handoff': pretraining_identity,
			'best_checkpoint': _identity(checkpoint_path, hashes),
			'embedding_metadata': _identity(candidate_files.metadata, hashes),
			'embeddings': _identity(candidate_files.embeddings, hashes),
			'valid_tokens': _identity(candidate_files.valid_tokens, hashes),
		},
		'candidate_run_manifest': run_identity,
	}


def _validate_candidate_original_run_manifest(
	payload: Mapping[str, object],
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> None:
	if payload.get('artifact_type') != 'f3_lithology_voxel_label_budget_center_trace_masked':
		raise ValueError('candidate original run manifest artifact type mismatch')
	rows = payload.get('rows')
	if not isinstance(rows, list) or len(rows) != 15:
		raise ValueError('candidate original run manifest must contain 15 rows')
	keys: set[tuple[str, int]] = set()
	files = output_paths(config.candidate_embeddings_dir, 'f3_facies_benchmark')
	checkpoint_path = _recorded_path_from_handoff(
		config.candidate_pretraining_handoff, 'checkpoint'
	)
	for value in rows:
		row = _mapping(value, 'candidate original run row')
		key = (str(row.get('budget_id')), int(row.get('subsample_seed', -1)))
		if key in keys or key[0] not in BUDGETS or key[1] not in range(5):
			raise ValueError('candidate original run manifest matrix is not 3x5')
		keys.add(key)
		if row.get('status') != 'complete':
			raise ValueError('candidate original run row is not complete')
		if row.get('model_role') != config.candidate_model_id:
			raise ValueError('candidate original run role mismatch')
		if row.get('model_tag') != config.candidate_model_tag:
			raise ValueError('candidate original run tag mismatch')
		if row.get('decoder_seed') != 42000 + key[1]:
			raise ValueError('candidate original run decoder seed mismatch')
		identity = _mapping(
			row.get('candidate_embedding_identity'),
			'candidate original run embedding identity',
		)
		for key_name, path in (
			('checkpoint', checkpoint_path),
			('metadata', files.metadata),
			('embeddings', files.embeddings),
			('valid_tokens', files.valid_tokens),
			('pretraining_handoff', config.candidate_pretraining_handoff),
		):
			_validate_recorded_identity(
				_mapping(identity.get(key_name), f'candidate run {key_name}'),
				path,
				hashes,
				label=f'candidate original run {key_name}',
				path_root=config.artifact_root,
			)
	if keys != {(budget, seed) for budget in BUDGETS for seed in range(5)}:
		raise ValueError('candidate original run manifest does not cover the 3x5 matrix')


def _validate_candidate_lineage(
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> dict[str, object]:
	candidate = _read_json(config.candidate_pretraining_handoff)
	if candidate.get('artifact_type') != CANDIDATE_HANDOFF_TYPE:
		raise ValueError('candidate pretraining handoff artifact type mismatch')
	if candidate.get('status') != 'PASS':
		raise ValueError('candidate pretraining handoff status must be PASS')
	if candidate.get('model_tag') != config.candidate_model_tag:
		raise ValueError('candidate pretraining handoff model tag mismatch')
	targets = _mapping(candidate.get('targets'), 'candidate pretraining targets')
	if targets.get('objective_semantics') != EXPECTED_OBJECTIVE:
		raise ValueError('candidate objective mismatch')
	if targets.get('target_representation') != EXPECTED_TARGET_REPRESENTATION:
		raise ValueError('candidate target representation mismatch')
	per_head = _mapping(
		targets.get('per_head_target_hashes'), 'candidate per-head targets'
	)
	try:
		head_ks = sorted(int(key) for key in per_head)
	except (TypeError, ValueError) as error:
		raise ValueError('candidate K must be [6, 8, 10]') from error
	if head_ks != EXPECTED_HEAD_KS:
		raise ValueError('candidate K must be [6, 8, 10]')

	checkpoint = _mapping(candidate.get('checkpoint'), 'candidate checkpoint')
	checkpoint_path = _required_recorded_path(
		checkpoint.get('path'), 'candidate checkpoint'
	)
	checkpoint_identity = _validate_recorded_identity(
		checkpoint,
		checkpoint_path,
		hashes,
		label='candidate checkpoint',
	)
	if checkpoint.get('schema_version') != 7:
		raise ValueError('candidate checkpoint schema must be 7')
	latest_path = _required_recorded_path(
		checkpoint.get('latest_path'), 'candidate latest checkpoint'
	)
	latest_identity = _validate_recorded_identity(
		{
			'path': str(latest_path),
			'sha256': checkpoint.get('latest_sha256'),
		},
		latest_path,
		hashes,
		label='candidate latest checkpoint',
	)
	checkpoint_payload = _torch_mapping(checkpoint_path, 'candidate checkpoint')
	checkpoint_info = _mapping(
		checkpoint_payload.get('stratigraphy_checkpoint'),
		'candidate checkpoint identity',
	)
	if checkpoint_info.get('schema_version') != 7:
		raise ValueError('live candidate checkpoint schema must be 7')
	if checkpoint_info.get('model_tag') != config.candidate_model_tag:
		raise ValueError('live candidate checkpoint model tag mismatch')
	if checkpoint_info.get('objective_semantics') != EXPECTED_OBJECTIVE:
		raise ValueError('live candidate checkpoint objective mismatch')
	if checkpoint_info.get('target_representation') != EXPECTED_TARGET_REPRESENTATION:
		raise ValueError('live candidate checkpoint target representation mismatch')
	if checkpoint_info.get('head_ks') != EXPECTED_HEAD_KS:
		raise ValueError('live candidate checkpoint K mismatch')

	files = output_paths(config.candidate_embeddings_dir, 'f3_facies_benchmark')
	candidate_embedding = _validate_embedding_artifacts(
		config.candidate_embeddings_dir,
		files,
		model_tag=config.candidate_model_tag,
		expected_shape=EXPECTED_EMBEDDINGS_SHAPE,
		expected_valid_shape=EXPECTED_VALID_TOKENS_SHAPE,
		expected_checkpoint=checkpoint_identity,
		hashes=hashes,
		label='candidate',
	)
	_validate_handoff_embedding_identity(
		_mapping(candidate.get('embedding'), 'candidate handoff embedding'),
		files,
		hashes,
		label='candidate handoff embedding',
	)
	execution_path = config.candidate_embeddings_dir / 'embedding_extraction_execution.json'
	execution = _read_json(execution_path)
	if (
		execution.get('artifact_type') != 'embedding_extraction_execution'
		or execution.get('schema_version') != 1
		or execution.get('encoder_input_mode') != UNMASKED_ENCODER_INPUT_MODE
	):
		raise ValueError('candidate embedding input mode must be unmasked')
	if execution.get('fresh', 0) + execution.get('reuse', 0) != execution.get('survey_count'):
		raise ValueError('candidate embedding execution count is inconsistent')
	baseline = _validate_baseline_lineage(config, hashes)
	if candidate_embedding['valid_tokens']['sha256'] != baseline['valid_tokens_sha256']:
		raise ValueError('candidate and baseline valid-token SHA-256 differ')
	return {
		'handoff': _identity(config.candidate_pretraining_handoff, hashes),
		'checkpoint': checkpoint_identity,
		'latest_checkpoint': latest_identity,
		'checkpoint_schema_version': 7,
		'objective': EXPECTED_OBJECTIVE,
		'target_representation': EXPECTED_TARGET_REPRESENTATION,
		'K': EXPECTED_HEAD_KS,
		'embedding_input_mode': execution['encoder_input_mode'],
		'embedding_execution': _identity(execution_path, hashes),
		'embedding': candidate_embedding,
		'baseline': baseline,
		'candidate_valid_tokens_sha256': candidate_embedding['valid_tokens']['sha256'],
		'baseline_valid_tokens_sha256': baseline['valid_tokens_sha256'],
	}


def _validate_baseline_lineage(
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> dict[str, object]:
	baseline = _read_json(config.hard_baseline_pretraining_handoff)
	if baseline.get('artifact_type') != BASELINE_HANDOFF_TYPE:
		raise ValueError('hard baseline handoff artifact type mismatch')
	if baseline.get('status') != 'PASS':
		raise ValueError('hard baseline handoff status must be PASS')
	if baseline.get('model_tag') != config.baseline_model_tag:
		raise ValueError('hard baseline handoff model tag mismatch')
	pretext = _mapping(baseline.get('stratigraphy_pretext'), 'baseline pretext')
	if pretext.get('head_ks') != EXPECTED_HEAD_KS:
		raise ValueError('hard baseline K must be [6, 8, 10]')
	files = output_paths(config.hard_baseline_embeddings_dir, 'f3_facies_benchmark')
	checkpoint = _mapping(baseline.get('checkpoint'), 'baseline checkpoint')
	checkpoint_path = _required_recorded_path(
		checkpoint.get('path'), 'baseline checkpoint'
	)
	checkpoint_identity = _validate_recorded_identity(
		checkpoint,
		checkpoint_path,
		hashes,
		label='baseline checkpoint',
	)
	embedding = _validate_embedding_artifacts(
		config.hard_baseline_embeddings_dir,
		files,
		model_tag=config.baseline_model_tag,
		expected_shape=EXPECTED_EMBEDDINGS_SHAPE,
		expected_valid_shape=EXPECTED_VALID_TOKENS_SHAPE,
		expected_checkpoint=checkpoint_identity,
		hashes=hashes,
		label='hard baseline',
	)
	_validate_handoff_embedding_identity(
		_mapping(baseline.get('embedding'), 'baseline handoff embedding'),
		files,
		hashes,
		label='baseline handoff embedding',
	)
	return {
		'handoff': _identity(config.hard_baseline_pretraining_handoff, hashes),
		'checkpoint': checkpoint_identity,
		'embedding': embedding,
		'valid_tokens_sha256': embedding['valid_tokens']['sha256'],
	}


def _validate_embedding_artifacts(
	embedding_dir: Path,
	files: EmbeddingOutputPaths,
	*,
	model_tag: str,
	expected_shape: tuple[int, ...],
	expected_valid_shape: tuple[int, ...],
	expected_checkpoint: Mapping[str, object],
	hashes: dict[Path, str],
	label: str,
) -> dict[str, object]:
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		if not path.is_file():
			raise FileNotFoundError(f'{label} embedding artifact is missing: {path}')
	metadata = _read_json(files.metadata)
	if Path(str(metadata.get('checkpoint_path', ''))).resolve(strict=False) != Path(
		str(expected_checkpoint['path'])
	).resolve(strict=False):
		raise ValueError(f'{label} embedding checkpoint path mismatch')
	if metadata.get('checkpoint_sha256') != expected_checkpoint['sha256']:
		raise ValueError(f'{label} embedding checkpoint hash mismatch')
	stratigraphy = _mapping(metadata.get('stratigraphy_pretext'), f'{label} metadata')
	if stratigraphy.get('model_tag') != model_tag:
		raise ValueError(f'{label} embedding model tag mismatch')
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	if embeddings.shape != expected_shape or embeddings.dtype != np.float16:
		raise ValueError(f'{label} embedding shape/dtype mismatch')
	if valid.shape != expected_valid_shape or valid.dtype != np.bool_:
		raise ValueError(f'{label} valid-token shape/dtype mismatch')
	if not _finite_valid_embeddings(embeddings, valid):
		raise ValueError(f'{label} embedding valid values are nonfinite')
	embeddings_hash = _live_hash(files.embeddings, hashes, label=f'{label} embeddings')
	valid_hash = _live_hash(files.valid_tokens, hashes, label=f'{label} valid tokens')
	metadata_hash = _live_hash(files.metadata, hashes, label=f'{label} metadata')
	return {
		'root': str(embedding_dir),
		'embeddings': {
			'path': str(files.embeddings),
			'byte_size': files.embeddings.stat().st_size,
			'sha256': embeddings_hash,
		},
		'metadata': {
			'path': str(files.metadata),
			'byte_size': files.metadata.stat().st_size,
			'sha256': metadata_hash,
		},
		'valid_tokens': {
			'path': str(files.valid_tokens),
			'byte_size': files.valid_tokens.stat().st_size,
			'sha256': valid_hash,
		},
		'embeddings_shape': list(embeddings.shape),
		'embeddings_dtype': str(embeddings.dtype),
		'valid_tokens_shape': list(valid.shape),
		'valid_tokens_dtype': str(valid.dtype),
		'finite_valid_count': int(valid.sum()),
	}


def _validate_handoff_embedding_identity(
	record: Mapping[str, object],
	files: EmbeddingOutputPaths,
	hashes: dict[Path, str],
	*,
	label: str,
) -> None:
	for name, path in (
		('embeddings', files.embeddings),
		('metadata', files.metadata),
		('valid_tokens', files.valid_tokens),
	):
		path_value = record.get(f'{name}_path', str(path))
		sha_value = record.get(f'{name}_sha256')
		if not isinstance(sha_value, str):
			raise TypeError(f'{label} {name} SHA-256 is missing')
		_validate_recorded_identity(
			{'path': path_value, 'sha256': sha_value},
			path,
			hashes,
			label=f'{label} {name}',
		)


def _validate_experiment96_evidence(
	config: F3CenterTraceMaskedSixSplitConfig,
	*,
	canonical_valid_token_sha256: str,
	hashes: dict[Path, str],
) -> dict[str, object]:
	dataset = _read_json(config.experiment96_dataset_manifest)
	if dataset.get('artifact_type') != 'f3_lithology_voxel_label_budget_split_dataset_manifest':
		raise ValueError('experiment 96 dataset manifest artifact type mismatch')
	if dataset.get('schema_version') != 1:
		raise ValueError('experiment 96 dataset manifest schema mismatch')
	contract = _mapping(dataset.get('contract'), 'experiment 96 dataset contract')
	if (
		contract.get('split_ids') != list(SPLIT_IDS)
		or contract.get('budgets') != ['cap25', 'cap50']
		or contract.get('label_subset_seed') != 0
	):
		raise ValueError('experiment 96 dataset contract is not canonical')
	dataset_rows = _validate_experiment96_dataset_rows(
		dataset.get('rows'), config, canonical_valid_token_sha256, hashes
	)

	run = _read_json(config.experiment96_scientific_run_manifest)
	if run.get('artifact_type') != 'f3_lithology_voxel_label_budget_split_run_manifest':
		raise ValueError('experiment 96 scientific run manifest artifact type mismatch')
	if run.get('schema_version') != 1:
		raise ValueError('experiment 96 scientific run manifest schema mismatch')
	run_rows = _validate_experiment96_run_rows(
		run.get('rows'), dataset_rows, config, canonical_valid_token_sha256, hashes
	)

	split_inventory = _validate_split_inventory(config, hashes)
	split_tokens = _validate_split_token_manifest(config, hashes)
	full_voxel = _validate_full_voxel_manifest(config, hashes)
	original = _validate_original_dataset_manifest(config, hashes)
	return {
		'dataset_manifest': _identity(config.experiment96_dataset_manifest, hashes),
		'dataset_row_count': len(dataset_rows),
		'scientific_run_manifest': _identity(
			config.experiment96_scientific_run_manifest, hashes
		),
		'scientific_run_row_count': len(run_rows),
		'mh_nocons_cap25_cap50_completed_rows': sum(
			1
			for row in run_rows
			if row.get('model_role') == BASELINE_MODEL_ID
			and row.get('budget_id') in {'cap25', 'cap50'}
		),
		'decoder_seed': config.decoder_seed,
		'label_subset_seed': config.label_subset_seed,
		'split_inventory_manifest': split_inventory,
		'split_token_dataset_manifest': split_tokens,
		'full_voxel_split_dataset_manifest': full_voxel,
		'original_split_dataset_manifest': original,
	}


def _validate_experiment96_dataset_rows(
	rows_value: object,
	config: F3CenterTraceMaskedSixSplitConfig,
	canonical_valid_token_sha256: str,
	hashes: dict[Path, str],
) -> dict[tuple[str, str], Mapping[str, object]]:
	if not isinstance(rows_value, list) or len(rows_value) != 12:
		raise ValueError('experiment 96 dataset manifest must contain 12 rows')
	indexed: dict[tuple[str, str], Mapping[str, object]] = {}
	for value in rows_value:
		row = _mapping(value, 'experiment 96 dataset row')
		key = (str(row.get('split_id')), str(row.get('budget_id')))
		if key in indexed or key[0] not in SPLIT_IDS or key[1] not in {'cap25', 'cap50'}:
			raise ValueError('experiment 96 dataset rows are not canonical six-by-two')
		indexed[key] = row
		if row.get('label_subset_seed') != config.label_subset_seed:
			raise ValueError('experiment 96 label subset seed mismatch')
		if row.get('canonical_valid_tokens_sha256') != canonical_valid_token_sha256:
			raise ValueError('experiment 96 canonical valid-token identity mismatch')
		if row.get('source_identities') != dict(config.source_identities):
			raise ValueError('experiment 96 source identity mismatch')
		if row.get('source_identities_sha256') != _json_sha256(config.source_identities):
			raise ValueError('experiment 96 source identity digest mismatch')
		root = Path(str(row.get('voxel_dataset_root', '')))
		if not root.is_dir():
			raise FileNotFoundError(f'experiment 96 dataset root is missing: {root}')
		grid = _mapping(row.get('supervision_split_grid'), 'experiment 96 supervision grid')
		grid_path = _required_recorded_path(
			grid.get('path'), 'experiment 96 supervision grid'
		)
		_validate_recorded_identity(
			grid, grid_path, hashes, label='experiment 96 supervision grid'
		)
	if set(indexed) != {
		(split, budget) for split in SPLIT_IDS for budget in ('cap25', 'cap50')
	}:
		raise ValueError('experiment 96 dataset manifest coverage is incomplete')
	return indexed


def _validate_experiment96_run_rows(
	rows_value: object,
	dataset_rows: Mapping[tuple[str, str], Mapping[str, object]],
	config: F3CenterTraceMaskedSixSplitConfig,
	canonical_valid_token_sha256: str,
	hashes: dict[Path, str],
) -> list[Mapping[str, object]]:
	if not isinstance(rows_value, list) or len(rows_value) != 36:
		raise ValueError('experiment 96 scientific run manifest must contain 36 rows')
	roles = ('mae', 'm1_current_k6', BASELINE_MODEL_ID)
	tags = {
		'mae': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		'm1_current_k6': 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1',
		BASELINE_MODEL_ID: BASELINE_MODEL_TAG,
	}
	indexed: dict[tuple[str, str, str], Mapping[str, object]] = {}
	for value in rows_value:
		row = _mapping(value, 'experiment 96 scientific run row')
		key = (
			str(row.get('split_id')),
			str(row.get('budget_id')),
			str(row.get('model_role')),
		)
		if key in indexed or key[:2] not in dataset_rows or key[2] not in roles:
			raise ValueError('experiment 96 scientific run matrix has duplicate or unknown rows')
		indexed[key] = row
		if row.get('status') != 'complete':
			raise ValueError('experiment 96 scientific run row is not complete')
		if row.get('model_tag') != tags[key[2]]:
			raise ValueError(f'experiment 96 model tag mismatch: {key!r}')
		if row.get('decoder_seed') != config.decoder_seed:
			raise ValueError(f'experiment 96 decoder seed mismatch: {key!r}')
		if row.get('subsample_seed') != config.label_subset_seed:
			raise ValueError(f'experiment 96 label subset seed mismatch: {key!r}')
		if row.get('canonical_valid_token_sha256') != canonical_valid_token_sha256:
			raise ValueError(f'experiment 96 valid-token identity mismatch: {key!r}')
		dataset_row = dataset_rows[key[:2]]
		if row.get('voxel_supervision_grid_sha256') != _mapping(
			dataset_row.get('supervision_split_grid'), 'dataset grid'
		).get('sha256'):
			raise ValueError(f'experiment 96 dataset pairing mismatch: {key!r}')
		grid = row.get('voxel_dataset')
		if isinstance(grid, Mapping):
			grid_path = _required_recorded_path(
				grid.get('path'), 'experiment 96 run grid'
			)
			_validate_recorded_identity(
				grid, grid_path, hashes, label='experiment 96 run grid'
			)
	if set(indexed) != {
		(split, budget, role)
		for split in SPLIT_IDS
		for budget in ('cap25', 'cap50')
		for role in roles
	}:
		raise ValueError('experiment 96 scientific run coverage is incomplete')
	if sum(
		1
		for row in indexed.values()
		if row.get('model_role') == BASELINE_MODEL_ID
		and row.get('budget_id') in {'cap25', 'cap50'}
	) != 12:
		raise ValueError('experiment 96 mh_nocons cap25/cap50 rows are incomplete')
	return list(indexed.values())


def _validate_split_inventory(
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> dict[str, object]:
	payload = _read_json(config.split_inventory_manifest)
	if payload.get('artifact_type') != 'f3_lithology_split_inventory_manifest':
		raise ValueError('split inventory manifest artifact type mismatch')
	rows = payload.get('rows')
	if not isinstance(rows, list) or len(rows) != 6:
		raise ValueError('split inventory manifest must contain six rows')
	ids: list[str] = []
	for value in rows:
		row = _mapping(value, 'split inventory row')
		ids.append(str(row.get('split_id')))
		metadata = row.get('split_metadata')
		if isinstance(metadata, str):
			metadata_path = Path(metadata).resolve(strict=False)
			if not metadata_path.is_file():
				raise FileNotFoundError(f'split inventory metadata is missing: {metadata_path}')
	if ids != list(SPLIT_IDS):
		raise ValueError('split inventory IDs are not canonical')
	return {
		'identity': _identity(config.split_inventory_manifest, hashes),
		'row_count': len(rows),
		'split_ids': ids,
	}


def _validate_split_token_manifest(
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> dict[str, object]:
	payload = _read_json(config.split_token_dataset_manifest)
	if payload.get('artifact_type') != 'f3_lithology_split_sweep_token_dataset_manifest':
		raise ValueError('split token dataset manifest artifact type mismatch')
	rows = payload.get('rows')
	if not isinstance(rows, list) or len(rows) != 12:
		raise ValueError('split token dataset manifest must contain 12 rows')
	keys: set[tuple[str, str]] = set()
	for value in rows:
		row = _mapping(value, 'split token dataset row')
		key = (str(row.get('split_id')), str(row.get('model_role')))
		if key in keys or key[0] not in SPLIT_IDS or key[1] not in {'baseline', 'candidate'}:
			raise ValueError('split token dataset manifest coverage is invalid')
		keys.add(key)
		for field in ('metadata_json', 'train_tokens', 'validation_tokens', 'class_counts_csv'):
			value_path = row.get(field)
			if isinstance(value_path, Mapping):
				identity = _mapping(value_path, f'split token {field}')
				path = _required_recorded_path(identity.get('path'), f'split token {field}')
				_validate_recorded_identity(
					identity, path, hashes, label=f'split token {field}'
				)
			elif isinstance(value_path, str):
				path = Path(value_path).resolve(strict=False)
				if not path.is_file():
					raise FileNotFoundError(f'split token {field} is missing: {path}')
	if keys != {
		(split, role) for split in SPLIT_IDS for role in ('baseline', 'candidate')
	}:
		raise ValueError('split token dataset manifest is incomplete')
	return {
		'identity': _identity(config.split_token_dataset_manifest, hashes),
		'row_count': len(rows),
	}


def _validate_full_voxel_manifest(
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> dict[str, object]:
	payload = _read_json(config.full_voxel_split_dataset_manifest)
	if payload.get('artifact_type') != 'f3_lithology_voxel_split_dataset_manifest':
		raise ValueError('full voxel split manifest artifact type mismatch')
	if payload.get('schema_version') != 1:
		raise ValueError('full voxel split manifest schema mismatch')
	source_inventory = _mapping(
		payload.get('source_split_inventory_manifest'), 'full voxel source inventory'
	)
	_validate_recorded_identity(
		source_inventory,
		config.split_inventory_manifest,
		hashes,
		label='full voxel source inventory',
	)
	rows = payload.get('rows')
	if not isinstance(rows, list) or len(rows) != 6:
		raise ValueError('full voxel split manifest must contain six rows')
	ids: list[str] = []
	for value in rows:
		row = _mapping(value, 'full voxel row')
		ids.append(str(row.get('split_id')))
		for field in ('reference_valid_tokens', 'slice_split_manifest', 'split_grid'):
			identity = _mapping(row.get(field), f'full voxel {field}')
			path = _required_recorded_path(identity.get('path'), f'full voxel {field}')
			_validate_recorded_identity(
				identity, path, hashes, label=f'full voxel {field}'
			)
	if ids != list(SPLIT_IDS):
		raise ValueError('full voxel split IDs are not canonical')
	return {
		'identity': _identity(config.full_voxel_split_dataset_manifest, hashes),
		'row_count': len(rows),
	}


def _validate_original_dataset_manifest(
	config: F3CenterTraceMaskedSixSplitConfig,
	hashes: dict[Path, str],
) -> dict[str, object]:
	payload = _read_json(config.original_split_dataset_manifest)
	if payload.get('artifact_type') != 'f3_lithology_voxel_label_budget_dataset_manifest':
		raise ValueError('original-split dataset manifest artifact type mismatch')
	if payload.get('schema_version') != 1:
		raise ValueError('original-split dataset manifest schema mismatch')
	contract = _mapping(payload.get('contract'), 'original-split dataset contract')
	if (
		contract.get('budgets') != list(BUDGETS)
		or contract.get('subsample_seeds') != list(range(5))
	):
		raise ValueError('original-split dataset contract is not canonical')
	rows = payload.get('rows')
	if not isinstance(rows, list) or len(rows) != 15:
		raise ValueError('original-split dataset manifest must contain 15 rows')
	keys: set[tuple[str, int]] = set()
	for value in rows:
		row = _mapping(value, 'original-split dataset row')
		key = (str(row.get('budget_id')), int(row.get('subsample_seed', -1)))
		if key in keys or key[0] not in BUDGETS or key[1] not in range(5):
			raise ValueError('original-split dataset matrix is not 3x5')
		keys.add(key)
	if keys != {(budget, seed) for budget in BUDGETS for seed in range(5)}:
		raise ValueError('original-split dataset coverage is incomplete')
	return {
		'identity': _identity(config.original_split_dataset_manifest, hashes),
		'row_count': len(rows),
	}


def _validate_six_split_start_state(
	config: F3CenterTraceMaskedSixSplitConfig,
) -> dict[str, object]:
	root = config.output_root
	if not root.exists():
		return {
			'scientific_jobs_executed': 0,
			'smoke_jobs_executed': 0,
			'evidence_files': [],
		}
	if not root.is_dir():
		raise NotADirectoryError(f'candidate-owned output root is not a directory: {root}')
	scientific = 0
	smoke = 0
	evidence_files: list[str] = []
	for path in sorted(root.rglob('*.json')):
		if path == config.audit_output_path:
			continue
		payload = _read_json(path)
		name = path.name.lower()
		is_job_evidence = any(
			marker in name
			for marker in ('run_manifest', 'job_manifest', 'execution', 'smoke', 'job')
		)
		counter_present = False
		for key in (
			'six_split_jobs_executed',
			'scientific_jobs_executed',
			'six_split_scientific_execution_count',
		):
			if key not in payload:
				continue
			value = payload[key]
			if not isinstance(value, int) or isinstance(value, bool):
				raise TypeError(f'{path} has a non-integer {key} counter')
			counter_present = True
			scientific += value
		if not is_job_evidence and not counter_present:
			continue
		evidence_files.append(str(path))
		for key in ('smoke_jobs_executed', 'smoke_execution_count'):
			if key not in payload:
				continue
			value = payload[key]
			if not isinstance(value, int) or isinstance(value, bool):
				raise TypeError(f'{path} has a non-integer {key} counter')
			smoke += value
		rows = payload.get('rows')
		if isinstance(rows, list):
			count = sum(1 for row in rows if isinstance(row, Mapping))
			if 'smoke' in path.parts or 'smoke' in name:
				smoke += count
			elif is_job_evidence:
				scientific += count
	if scientific != 0 or smoke != 0:
		raise ValueError(
			'six-split start state has executed jobs: '
			f'scientific={scientific}, smoke={smoke}'
		)
	return {
		'scientific_jobs_executed': 0,
		'smoke_jobs_executed': 0,
		'evidence_files': evidence_files,
	}


def _config_payload(config: F3CenterTraceMaskedSixSplitConfig) -> dict[str, object]:
	return {
		'paths': {
			'artifact_root': str(config.artifact_root),
			'output_root': str(config.output_root),
		},
		'matrix': {
			'candidate': {
				'model_id': config.candidate_model_id,
				'model_tag': config.candidate_model_tag,
			},
			'baseline': {
				'model_id': config.baseline_model_id,
				'model_tag': config.baseline_model_tag,
			},
			'split_ids': list(config.split_ids),
			'budgets': list(config.budgets),
			'label_subset_seed': config.label_subset_seed,
			'decoder_seed': config.decoder_seed,
		},
		'inputs': {
			**{
				name: str(getattr(config, name))
				for name in (
					'original_split_handoff',
					'candidate_pretraining_handoff',
					'candidate_embeddings_dir',
					'hard_baseline_pretraining_handoff',
					'hard_baseline_embeddings_dir',
					'experiment96_dataset_manifest',
					'experiment96_scientific_run_manifest',
					'split_inventory_manifest',
					'split_token_dataset_manifest',
					'full_voxel_split_dataset_manifest',
					'original_split_dataset_manifest',
					'seismic_volume',
					'source_label_segy',
					'class_info',
					'segy_geometry_json',
				)
			},
			'source_identities': {
				name: dict(identity)
				for name, identity in config.source_identities.items()
			},
		},
	}


def _required_audit_structure(payload: Mapping[str, object]) -> None:
	if payload.get('artifact_type') != ARTIFACT_TYPE:
		raise ValueError('six-split audit artifact type mismatch')
	if payload.get('schema_version') != SCHEMA_VERSION or payload.get('status') != 'PASS':
		raise ValueError('six-split audit schema/status mismatch')
	for key in (
		'config',
		'split_ids',
		'budgets',
		'primary_model_roles',
		'primary_model_tags',
		'primary_matrix_row_count',
		'future_candidate_jobs',
		'future_new_baseline_jobs',
		'historical_baseline_rows',
		'future_new_scientific_jobs',
		'scientific_jobs_executed',
		'smoke_jobs_executed',
		'source_files',
		'evidence',
		'git',
	):
		if key not in payload:
			raise ValueError(f'six-split audit field is missing: {key}')
	if payload.get('split_ids') != list(SPLIT_IDS) or payload.get('budgets') != list(BUDGETS):
		raise ValueError('six-split audit matrix identity mismatch')
	if payload.get('primary_matrix_row_count') != 36:
		raise ValueError('six-split audit primary row count mismatch')
	if payload.get('future_candidate_jobs') != 18 or payload.get('future_new_baseline_jobs') != 6:
		raise ValueError('six-split audit future job counts mismatch')
	if payload.get('historical_baseline_rows') != 12 or payload.get('future_new_scientific_jobs') != 24:
		raise ValueError('six-split audit historical/future count mismatch')
	if payload.get('scientific_jobs_executed') != 0 or payload.get('smoke_jobs_executed') != 0:
		raise ValueError('six-split audit execution counters must be zero')
	_validate_json_tree(payload, context='persisted six-split audit')


def _revalidate_persisted_audit(payload: Mapping[str, object]) -> None:
	config_raw = _mapping(payload.get('config'), 'persisted six-split config')
	config = f3_lithology_voxel_label_budget_center_trace_masked_split_config_from_mapping(
		config_raw
	)
	expected = _audit_payload(config, git=_mapping(payload.get('git'), 'persisted Git'))
	if dict(payload) != expected:
		raise ValueError('persisted six-split audit is stale')


def _validate_recorded_identity(
	record: Mapping[str, object],
	expected_path: Path,
	hashes: dict[Path, str],
	*,
	label: str,
	path_root: Path | None = None,
) -> dict[str, object]:
	recorded_path = _resolve_recorded_path(
		record.get('path'), path_root or expected_path.parent
	)
	if recorded_path != expected_path.resolve(strict=False):
		raise ValueError(f'{label} path identity mismatch')
	actual = _live_hash(expected_path, hashes, label=label)
	if record.get('sha256') != actual:
		raise ValueError(f'{label} SHA-256 identity mismatch')
	if 'byte_size' in record and record.get('byte_size') != expected_path.stat().st_size:
		raise ValueError(f'{label} byte-size identity mismatch')
	return {
		'path': str(expected_path),
		'byte_size': expected_path.stat().st_size,
		'sha256': actual,
	}


def _recorded_path_from_handoff(path: Path, key: str) -> Path:
	payload = _read_json(path)
	record = _mapping(payload.get(key), f'{path.name}.{key}')
	return _required_recorded_path(record.get('path'), f'{path.name}.{key}')


def _required_recorded_path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label}.path is missing')
	path = Path(value).resolve(strict=False)
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	return path


def _resolve_recorded_path(value: object, root: Path) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError('recorded identity path must be a non-empty string')
	path = value.replace('${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}', str(root))
	return Path(path).resolve(strict=False)


def _identity(path: Path, hashes: dict[Path, str] | None = None) -> dict[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	identity_hashes = hashes if hashes is not None else {}
	return {
		'path': str(path),
		'byte_size': path.stat().st_size,
		'sha256': _live_hash(path, identity_hashes, label=str(path)),
	}


def _live_hash(path: Path, hashes: dict[Path, str], *, label: str) -> str:
	resolved = path.resolve(strict=False)
	if not resolved.is_file():
		raise FileNotFoundError(f'{label} is missing: {resolved}')
	if resolved not in hashes:
		hashes[resolved] = file_sha256(resolved)
	return hashes[resolved]


def _torch_mapping(path: Path, label: str) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a mapping')
	return payload


def _finite_valid_embeddings(embeddings: np.ndarray, valid: np.ndarray) -> bool:
	for index in range(embeddings.shape[0]):
		if valid[index].any() and not np.isfinite(embeddings[index][valid[index]]).all():
			return False
	return True


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	return _mapping(payload, str(path))


def _json_sha256(value: object) -> str:
	return hashlib.sha256(
		json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
	).hexdigest()


def _validate_json_tree(value: object, *, context: str) -> None:
	if isinstance(value, float) and not math.isfinite(value):
		raise ValueError(f'{context} contains a non-finite number')
	if isinstance(value, Path):
		raise TypeError(f'{context} contains a non-portable Path object')
	if isinstance(value, Mapping):
		for key, child in value.items():
			_validate_json_tree(key, context=context)
			_validate_json_tree(child, context=context)
	elif isinstance(value, list | tuple):
		for child in value:
			_validate_json_tree(child, context=context)


def _git_provenance() -> Mapping[str, object]:
	"""Record HEAD, complete short status, and the binary tracked diff hash."""
	repository = Path(__file__).resolve().parents[3]
	git = shutil.which('git')
	if git is None:
		raise RuntimeError('git executable is unavailable')
	try:
		head = subprocess.run(
			(git, 'rev-parse', 'HEAD'),
			cwd=repository,
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
		status = subprocess.run(
			(git, 'status', '--short', '--untracked-files=all'),
			cwd=repository,
			check=True,
			capture_output=True,
			text=True,
		).stdout
		diff = subprocess.run(
			(git, 'diff', '--binary', 'HEAD'),
			cwd=repository,
			check=True,
			capture_output=True,
		).stdout
	except (OSError, subprocess.CalledProcessError) as error:
		raise RuntimeError('unable to collect Git provenance') from error
	if len(head) != 40:
		raise ValueError('Git HEAD is not a SHA-1 commit')
	return {
		'git_commit': head,
		'git_status_short': status.splitlines(),
		'git_diff_sha256': hashlib.sha256(diff).hexdigest(),
	}


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		'w',
		encoding='utf-8',
		dir=path.parent,
		prefix=f'.{path.name}.',
		delete=False,
	) as handle:
		handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
		temporary = Path(handle.name)
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
	'BASELINE_HANDOFF_TYPE',
	'CANDIDATE_HANDOFF_TYPE',
	'EXPECTED_EMBEDDINGS_SHAPE',
	'EXPECTED_HEAD_KS',
	'EXPECTED_OBJECTIVE',
	'EXPECTED_TARGET_REPRESENTATION',
	'F3CenterTraceMaskedSixSplitAuditResult',
	'audit_f3_center_trace_masked_six_split',
	'f3_center_trace_masked_six_split_audit_config_from_mapping',
	'load_f3_center_trace_masked_six_split_audit',
	'load_f3_center_trace_masked_six_split_audit_config',
]
