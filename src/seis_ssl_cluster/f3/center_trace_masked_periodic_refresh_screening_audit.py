"""Immutable preflight audit for the periodic-refresh original-split screen."""
# ruff: noqa: C901, CPY001, E501, PLR0912, SLF001, S603

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.embedding.extractor import UNMASKED_ENCODER_INPUT_MODE
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import (
	center_trace_masked_periodic_refresh_validation as periodic_validation,
)
from seis_ssl_cluster.f3 import (
	center_trace_masked_pretraining_validation as center_validation,
)
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked as fixed_runner,
)
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head as decoder_runner,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	inspect_f3_lithology_voxel_label_budget_mae_reference_run,
)
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest

ARTIFACT_TYPE = 'f3_center_trace_masked_periodic_refresh_original_screening_preflight'
SCHEMA_VERSION = 1
MODEL_ID = 'mh_ctmask010_refresh3ep_hmm2_nocons'
MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_topblock1_distill_v1'
)
VARIANT = 'ctmask010_refresh3ep_hmm2_nocons'
GENERATION_IDS = (
	'refresh_0000_initial',
	'refresh_0001_epoch002',
	'refresh_0002_epoch005',
	'refresh_0003_epoch008',
	'refresh_0004_epoch011',
	'refresh_0005_epoch014',
	'refresh_0006_epoch017',
	'refresh_0007_epoch020',
)
REFRESH_EPOCHS = (2, 5, 8, 11, 14, 17, 20)
_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'workspace_root',
		'source_hard_manifest',
		'hard_full_config',
		'hard_pretraining_handoff',
		'center_trace_masked_config',
		'periodic_refresh_validation_config',
		'periodic_refresh_full_config',
		'periodic_refresh_handoff',
		'periodic_refresh_embeddings_dir',
		'output_path',
	}
)
_EXPECTED_PERIODIC_FIELDS = {
	'experiment_role': 'multi_head_center_trace_masked_periodic_hmm_refresh_hard_pretext',
	'variant': VARIANT,
	'model_role': MODEL_ID,
	'target_refresh_semantics': 'periodic_student_hmm_center_refresh_v1',
	'refresh_schedule_semantics': 'after_epochs_2_5_8_11_14_17_20_v1',
	'refresh_after_epochs': list(REFRESH_EPOCHS),
	'hmm_iterations_per_refresh': 2,
	'embedding_source': 'current_student',
	'embedding_mode': 'unmasked_eval_full_survey',
	'refresh_embedding_semantics': 'current_student_unmasked_eval_full_survey_v1',
	'center_initialization': 'previous_generation',
	'center_update': 'full_mean',
	'center_update_semantics': 'warm_start_full_mean_two_iterations_final_decode_v1',
	'preprocessing_policy': 'freeze_initial_residualizer_pca_v1',
	'target_activation_policy': 'atomic_next_epoch_activation_v1',
	'empty_state_policy': 'error',
	'checkpoint_selection_policy': 'final_completed_epoch_v1',
}


@dataclass(frozen=True)
class F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig:
	"""Closed paths required for the periodic-refresh screening audit."""

	artifact_root: Path
	workspace_root: Path
	source_hard_manifest: Path
	hard_full_config: Path
	hard_pretraining_handoff: Path
	center_trace_masked_config: Path
	periodic_refresh_validation_config: Path
	periodic_refresh_full_config: Path
	periodic_refresh_handoff: Path
	periodic_refresh_embeddings_dir: Path
	output_path: Path


@dataclass(frozen=True)
class F3CenterTraceMaskedPeriodicRefreshScreeningAuditResult:
	"""Audit payload and the owned-output action."""

	payload: Mapping[str, object]
	output_path: Path
	action: str
	quarantine_path: Path | None


def f3_center_trace_masked_periodic_refresh_screening_audit_config_from_mapping(
	config: Mapping[str, object],
) -> F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig:
	"""Resolve the closed periodic-refresh screening-audit YAML schema."""
	if not isinstance(config, Mapping):
		raise TypeError('periodic-refresh screening audit config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(
			f'unknown periodic-refresh screening audit keys: {sorted(unknown)!r}'
		)
	if missing:
		raise ValueError(
			f'missing periodic-refresh screening audit keys: {sorted(missing)!r}'
		)

	def path(key: str, *, must_exist: bool, directory: bool = False) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		resolved = Path(value).resolve()
		if must_exist and not resolved.exists():
			raise FileNotFoundError(f'{key} is missing: {resolved}')
		if directory and must_exist and not resolved.is_dir():
			raise FileNotFoundError(f'{key} must be a directory: {resolved}')
		return resolved

	result = F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig(
		artifact_root=path('artifact_root', must_exist=True, directory=True),
		workspace_root=path('workspace_root', must_exist=True, directory=True),
		source_hard_manifest=path('source_hard_manifest', must_exist=True),
		hard_full_config=path('hard_full_config', must_exist=True),
		hard_pretraining_handoff=path('hard_pretraining_handoff', must_exist=True),
		center_trace_masked_config=path('center_trace_masked_config', must_exist=True),
		periodic_refresh_validation_config=path(
			'periodic_refresh_validation_config', must_exist=True
		),
		periodic_refresh_full_config=path(
			'periodic_refresh_full_config', must_exist=True
		),
		periodic_refresh_handoff=path('periodic_refresh_handoff', must_exist=True),
		periodic_refresh_embeddings_dir=path(
			'periodic_refresh_embeddings_dir', must_exist=True, directory=True
		),
		output_path=path('output_path', must_exist=False),
	)
	for label, value in (
		('source_hard_manifest', result.source_hard_manifest),
		('hard_pretraining_handoff', result.hard_pretraining_handoff),
		('periodic_refresh_handoff', result.periodic_refresh_handoff),
		('periodic_refresh_embeddings_dir', result.periodic_refresh_embeddings_dir),
		('output_path', result.output_path),
	):
		ensure_under_root(value, root=result.artifact_root, label=label)
	for label, value in (
		('hard_full_config', result.hard_full_config),
		('center_trace_masked_config', result.center_trace_masked_config),
		(
			'periodic_refresh_validation_config',
			result.periodic_refresh_validation_config,
		),
		('periodic_refresh_full_config', result.periodic_refresh_full_config),
	):
		ensure_under_root(value, root=result.workspace_root, label=label)
	if result.periodic_refresh_handoff.name != 'periodic_refresh_handoff.json':
		raise ValueError('periodic refresh handoff name is not canonical')
	if result.output_path.name != 'periodic_refresh_screening_audit.json':
		raise ValueError('periodic refresh audit output name is not canonical')
	expected_handoff = (
		result.artifact_root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ MODEL_TAG
		/ 'preflight/periodic_refresh_handoff.json'
	)
	if result.periodic_refresh_handoff != expected_handoff:
		raise ValueError('periodic refresh handoff path is not canonical')
	expected_embeddings = (
		result.artifact_root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ MODEL_TAG
		/ 'overlap_x16'
	)
	if result.periodic_refresh_embeddings_dir != expected_embeddings:
		raise ValueError('periodic refresh embeddings path is not canonical')
	return result


def load_f3_center_trace_masked_periodic_refresh_screening_audit_config(
	path: str | Path,
) -> F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig:
	"""Load a periodic-refresh screening-audit YAML file."""
	return f3_center_trace_masked_periodic_refresh_screening_audit_config_from_mapping(
		load_config(path)
	)


def audit_f3_center_trace_masked_periodic_refresh_screening(
	config: F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3CenterTraceMaskedPeriodicRefreshScreeningAuditResult:
	"""Build, reuse, or explicitly quarantine only the owned audit file."""
	payload = _audit_payload(config, git=_clean_git_identity(config.workspace_root))
	if dry_run:
		return F3CenterTraceMaskedPeriodicRefreshScreeningAuditResult(
			payload, config.output_path, 'DRY_RUN', None
		)
	if config.output_path.exists():
		if not config.output_path.is_file():
			raise FileExistsError(
				f'audit output path is not a file: {config.output_path}'
			)
		try:
			existing = load_f3_center_trace_masked_periodic_refresh_screening_audit(
				config.output_path, revalidate=False
			)
		except (TypeError, ValueError):
			existing = None
		if existing == payload:
			if only_missing:
				return F3CenterTraceMaskedPeriodicRefreshScreeningAuditResult(
					payload, config.output_path, 'REUSE_COMPLETED', None
				)
			raise FileExistsError(
				f'audit already exists; use --only-missing: {config.output_path}'
			)
		if existing is not None and _source_identity_drift(existing, payload):
			raise ValueError(
				'periodic-refresh source/handoff identity drift is an error'
			)
		if not quarantine_invalid:
			raise ValueError(
				'incompatible existing periodic-refresh audit; '
				'use --quarantine-invalid to replace it'
			)
		quarantine = _quarantine_invalid(config.output_path)
	else:
		quarantine = None
	_write_json_atomically(config.output_path, payload)
	return F3CenterTraceMaskedPeriodicRefreshScreeningAuditResult(
		payload, config.output_path, 'WRITTEN', quarantine
	)


def load_f3_center_trace_masked_periodic_refresh_screening_audit(
	path: str | Path,
	*,
	revalidate: bool = True,
) -> Mapping[str, object]:
	"""Load and, by default, revalidate a PASS periodic-refresh audit."""
	payload = _read_json(Path(path))
	if (
		payload.get('artifact_type') != ARTIFACT_TYPE
		or payload.get('schema_version') != SCHEMA_VERSION
		or payload.get('status') != 'PASS'
	):
		raise ValueError('periodic-refresh screening audit type/schema/status mismatch')
	for key in (
		'candidate',
		'periodic_refresh_handoff',
		'checkpoint',
		'generation_chain',
		'embedding',
		'valid_mask_parity',
		'decoder_contract',
		'periodic_refresh_validation',
		'hard_baseline',
		'reference_run_manifests',
		'fixed_center_trace_reference',
		'dataset_job_pairing',
		'gate_inputs',
		'git',
	):
		if not isinstance(payload.get(key), Mapping):
			raise TypeError(f'periodic-refresh screening audit {key} is missing')
	gate_inputs = _mapping(payload['gate_inputs'], 'periodic-refresh gate inputs')
	for forbidden in (
		'pretraining_loss',
		'center_shift',
		'label_change_rate',
		'masked_accuracy',
	):
		if forbidden in gate_inputs:
			raise ValueError(f'forbidden periodic-refresh gate input: {forbidden}')
	if set(gate_inputs) != {
		'primary_metrics',
		'guardrail_classes',
		'guardrail_metrics',
	}:
		raise ValueError('periodic-refresh gate inputs are not the closed metric set')
	if revalidate:
		_revalidate_persisted_audit(payload)
	return payload


def validate_f3_center_trace_masked_periodic_refresh_screening_audit_binding(
	payload: Mapping[str, object],
	*,
	model_id: str,
	model_tag: str,
	pretraining_handoff: Path,
	embeddings_dir: Path,
) -> None:
	"""Bind a PASS audit to the exact periodic candidate admitted to planning."""
	if (
		payload.get('artifact_type') != ARTIFACT_TYPE
		or payload.get('schema_version') != SCHEMA_VERSION
		or payload.get('status') != 'PASS'
	):
		raise ValueError('periodic-refresh screening audit identity mismatch')
	candidate = _mapping(payload.get('candidate'), 'periodic audit candidate')
	if candidate.get('model_id') != model_id or candidate.get('model_tag') != model_tag:
		raise ValueError('periodic-refresh audit candidate identity mismatch')
	_identity_matches(
		candidate.get('pretraining_handoff'),
		pretraining_handoff,
		label='periodic-refresh audit pretraining handoff',
	)
	embedding = _mapping(candidate.get('embeddings'), 'periodic audit embeddings')
	if Path(str(embedding.get('root', ''))).resolve() != embeddings_dir.resolve():
		raise ValueError('periodic-refresh audit candidate embeddings root mismatch')
	for key in ('embeddings', 'valid_tokens', 'metadata'):
		value = embedding.get(key)
		if isinstance(value, Mapping):
			_identity_matches_live(value, label=f'periodic audit {key}')
			continue
		path = _required_live_path(embedding.get(f'{key}_path'), key)
		if embedding.get(f'{key}_sha256') != file_sha256(path):
			raise ValueError(f'periodic-refresh audit {key} identity mismatch')
	if (
		_mapping(payload.get('periodic_refresh_handoff'), 'periodic handoff').get(
			'status'
		)
		!= 'PASS'
	):
		raise ValueError('periodic-refresh handoff is not PASS')


def _audit_payload(
	config: F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig,
	*,
	git: Mapping[str, object],
) -> dict[str, object]:
	"""Construct every live assertion before publishing one audit byte."""
	target = load_multi_head_target_manifest(config.source_hard_manifest)
	target_hashes = center_validation._multi_head_target_hashes(target)
	hard = resolve_strat_hmm_pretext_config(load_config(config.hard_full_config))
	periodic = resolve_strat_hmm_pretext_config(
		load_config(config.periodic_refresh_full_config)
	)
	_validate_periodic_training_identity(
		config, periodic, target=target, target_hashes=target_hashes
	)
	hard_handoff = hard_validation.load_f3_multi_head_pretraining_handoff(
		config.hard_pretraining_handoff
	)
	_validate_hard_handoff(
		config, hard_handoff=hard_handoff, target_hashes=target_hashes
	)
	periodic_handoff = (
		periodic_validation.load_f3_center_trace_masked_periodic_refresh_handoff(
			config.periodic_refresh_handoff
		)
	)
	periodic_validation_evidence = _periodic_validation_evidence(config)
	checkpoint = _mapping(
		periodic_validation_evidence.get('checkpoint'),
		'periodic validation checkpoint evidence',
	)
	embedding = _embedding_evidence(config, checkpoint=checkpoint)
	refresh = _mapping(
		periodic_validation_evidence.get('refresh'),
		'periodic validation refresh evidence',
	)
	chain = {
		'path': refresh['chain_path'],
		'sha256': refresh['chain_sha256'],
		'generation_count': len(refresh['generations']),
		'generation_ids': [
			_mapping(item, 'periodic generation evidence')['generation_id']
			for item in refresh['generations']
		],
		'refresh_after_epochs': list(REFRESH_EPOCHS),
		'initial_target_manifest': _mapping(
			_mapping(periodic_handoff['targets'], 'periodic handoff targets')[
				'initial_hard_target_manifest'
			],
			'periodic initial target',
		),
		'fixed_target_head_ks': target.get('head_ks'),
		'final_generation_id': refresh['final_generation_id'],
	}
	references, pairing, fixed_reference = _reference_evidence(config)
	decoder_contract = _decoder_contract_evidence(
		config, target=target, hard=hard, periodic=periodic
	)
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
			'variant': VARIANT,
			'periodic_refresh_full_config': _identity(
				config.periodic_refresh_full_config
			),
			'pretraining_handoff': _identity(config.periodic_refresh_handoff),
			'embeddings': embedding,
		},
		'periodic_refresh_handoff': {
			'artifact_type': periodic_handoff['artifact_type'],
			'schema_version': periodic_handoff['schema_version'],
			'status': periodic_handoff['status'],
			'model_tag': periodic_handoff['model_tag'],
			'variant': periodic_handoff['variant'],
			'primary_checkpoint_role': periodic_handoff['primary_checkpoint_role'],
			'initial_target': _identity(config.source_hard_manifest),
			'final_target': _mapping(
				_mapping(periodic_handoff['targets'], 'periodic handoff targets')[
					'final_target_manifest'
				],
				'periodic final target',
			),
		},
		'periodic_refresh_validation': periodic_validation_evidence,
		'hard_baseline': {
			'config': _identity(config.hard_full_config),
			'handoff': _identity(config.hard_pretraining_handoff),
			'model_tag': hard_handoff['model_tag'],
			'target_manifest': _identity(config.source_hard_manifest),
			'per_head_target_hashes': target_hashes,
		},
		'checkpoint': checkpoint,
		'generation_chain': chain,
		'embedding': embedding,
		'valid_mask_parity': embedding['canonical_valid_token_identities'],
		'decoder_contract': decoder_contract,
		'reference_run_manifests': references,
		'fixed_center_trace_reference': fixed_reference,
		'dataset_job_pairing': pairing,
		'gate_inputs': {
			'primary_metrics': ['macro_f1', 'mean_iou'],
			'guardrail_classes': [3, 5],
			'guardrail_metrics': [
				'f1',
				'iou',
				'boundary_recall_t2',
				'boundary_recall_t4',
			],
		},
	}


def _validate_periodic_training_identity(
	config: F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig,
	periodic: Mapping[str, object],
	*,
	target: Mapping[str, object],
	target_hashes: Mapping[str, object],
) -> None:
	identity = _mapping(
		_mapping(periodic.get('identity'), 'periodic identity').get(
			'scientific_identity'
		),
		'periodic scientific identity',
	)
	for key, expected in _EXPECTED_PERIODIC_FIELDS.items():
		if identity.get(key) != expected:
			raise ValueError(f'periodic scientific identity mismatch: {key}')
	if identity.get('head_ks') != [6, 8, 10]:
		raise ValueError('periodic head identity mismatch')
	periodic_targets = _mapping(periodic.get('pseudo_targets'), 'periodic targets')
	periodic_target_path = Path(str(periodic_targets['manifest'])).resolve()
	if periodic_target_path != config.source_hard_manifest:
		raise ValueError('periodic target manifest path mismatch')
	if identity.get('target_manifest_sha256') != file_sha256(periodic_target_path):
		raise ValueError('periodic target manifest hash mismatch')
	if identity.get('target_head_hashes') != target_hashes:
		raise ValueError('periodic target head hashes mismatch')
	if target.get('head_ks') != [6, 8, 10] or not target_hashes:
		raise ValueError('periodic target lineage is not the fixed K6/K8/K10 target')
	train = _mapping(periodic.get('train'), 'periodic train')
	for key, expected in (
		('batch_size', 4),
		('samples_per_epoch', 4096),
		('epochs', 25),
		('lr', 0.0003),
		('encoder_lr', 0.00001),
		('seed', 42),
	):
		if train.get(key) != expected:
			raise ValueError(f'periodic training identity mismatch: {key}')
	if (
		_mapping(periodic.get('student'), 'periodic student').get('unfreeze_top_blocks')
		!= 1
	):
		raise ValueError('periodic student must unfreeze top block 1')
	refresh = _mapping(periodic.get('pseudo_target_refresh'), 'periodic refresh')
	if (
		refresh.get('refresh_after_epochs') != list(REFRESH_EPOCHS)
		or refresh.get('hmm_iterations_per_refresh') != 2
	):
		raise ValueError('periodic refresh schedule drift')


def _validate_hard_handoff(
	config: F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig,
	*,
	hard_handoff: Mapping[str, object],
	target_hashes: Mapping[str, object],
) -> None:
	"""Bind the hard K=6/8/10 handoff to the immutable initial target."""
	if hard_handoff.get('model_tag') != (
		'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
	):
		raise ValueError('hard baseline handoff model tag mismatch')
	identity = _mapping(
		hard_handoff.get('stratigraphy_pretext'), 'hard handoff identity'
	)
	if (
		Path(str(identity.get('target_manifest_path', ''))).resolve()
		!= config.source_hard_manifest
	):
		raise ValueError('hard baseline handoff target path mismatch')
	if identity.get('target_manifest_sha256') != file_sha256(
		config.source_hard_manifest
	):
		raise ValueError('hard baseline handoff target hash mismatch')
	if identity.get('per_head_target_sha256') != target_hashes:
		raise ValueError('hard baseline handoff target head hashes mismatch')


def _periodic_validation_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig,
) -> Mapping[str, object]:
	"""Reuse the complete periodic validator without writing its handoff."""
	validation_config = periodic_validation.load_f3_center_trace_masked_periodic_refresh_validation_config(
		config.periodic_refresh_validation_config
	)
	if validation_config.target_manifest != config.source_hard_manifest:
		raise ValueError('periodic validation target manifest mismatch')
	if validation_config.hard_handoff != config.hard_pretraining_handoff:
		raise ValueError('periodic validation hard handoff mismatch')
	if validation_config.periodic_refresh_full_config != (
		config.periodic_refresh_full_config
	):
		raise ValueError('periodic validation full config mismatch')
	extraction = resolve_embedding_extraction_config(
		load_config(validation_config.periodic_refresh_embedding_config)
	)
	extraction_root = Path(
		str(_mapping(extraction['embeddings'], 'periodic extraction')['output_dir'])
	).resolve()
	if extraction_root != config.periodic_refresh_embeddings_dir:
		raise ValueError('periodic validation embedding root mismatch')
	validation_result = (
		periodic_validation.validate_f3_center_trace_masked_periodic_refresh(
			validation_config,
			phase='complete',
			dry_run=True,
		)
	)
	evidence = _mapping(validation_result.evidence, 'periodic validation evidence')
	if evidence.get('status') != 'PASS':
		raise ValueError(
			'periodic complete validation failed: '
			f'{evidence.get("error", "unknown failure")}'
		)
	return evidence


def _embedding_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig,
	*,
	checkpoint: Mapping[str, object],
) -> dict[str, object]:
	files = output_paths(config.periodic_refresh_embeddings_dir, 'f3_facies_benchmark')
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		if not path.is_file():
			raise FileNotFoundError(path)
	metadata = _read_json(files.metadata)
	if (
		Path(str(metadata.get('checkpoint_path', ''))).resolve()
		!= Path(str(checkpoint['path'])).resolve()
	):
		raise ValueError('periodic extraction checkpoint does not equal selected.pt')
	if metadata.get('checkpoint_sha256') != checkpoint['sha256']:
		raise ValueError('periodic extraction checkpoint SHA-256 mismatch')
	execution_path = config.periodic_refresh_embeddings_dir / (
		'embedding_extraction_execution.json'
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
		raise ValueError('periodic extraction is not explicitly unmasked')
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	if (
		embeddings.shape != (76, 113, 32, 384)
		or embeddings.dtype != np.float16
		or valid.shape != (76, 113, 32)
		or valid.dtype != np.bool_
		or int(valid.sum()) <= 0
	):
		raise ValueError('periodic embedding shape/dtype contract mismatch')
	if not _finite_valid_embeddings(embeddings, valid):
		raise ValueError('periodic embedding valid values are nonfinite')
	canonical = _canonical_valid_mask_identities(config.artifact_root)
	valid_sha = file_sha256(files.valid_tokens)
	if any(value['sha256'] != valid_sha for value in canonical.values()):
		raise ValueError('periodic valid-token mask differs from canonical references')
	return {
		'root': str(config.periodic_refresh_embeddings_dir),
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
		'canonical_valid_token_identities': canonical,
	}


def _decoder_contract_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig,
	*,
	target: Mapping[str, object],
	hard: Mapping[str, object],
	periodic: Mapping[str, object],
) -> Mapping[str, object]:
	decoder_config = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(
			config.workspace_root
			/ 'experiments/f3/facies_benchmark_v1/'
			/ '95_strat_hmm_multi_head_k6810_low_label_v1'
			/ '01_run_multi_head_voxel_label_budget.yaml'
		)
	)
	periodic_target = _mapping(periodic.get('pseudo_targets'), 'periodic targets')
	if (
		Path(str(periodic_target.get('manifest'))).resolve()
		!= config.source_hard_manifest
	):
		raise ValueError('periodic decoder target manifest mismatch')
	if decoder_config.multi_head_target_manifest != config.source_hard_manifest:
		raise ValueError('decoder reference target manifest mismatch')
	if target.get('head_ks') != [6, 8, 10]:
		raise ValueError('decoder target head identity mismatch')
	for key, expected in (
		('embedding_dim', 384),
		('hidden_channels', [128, 64, 32]),
		('upsample_mode', 'nearest'),
		('normalization', 'voxelwise_layer_norm'),
	):
		if getattr(decoder_config.decoder, key) != expected:
			raise ValueError(f'decoder contract mismatch: {key}')
	return {
		'spec': decoder_config.decoder.spec,
		'embedding_dim': decoder_config.decoder.embedding_dim,
		'hidden_channels': list(decoder_config.decoder.hidden_channels),
		'upsample_mode': decoder_config.decoder.upsample_mode,
		'normalization': decoder_config.decoder.normalization,
		'epochs': decoder_config.train.epochs,
		'steps_per_epoch': decoder_config.train.steps_per_epoch,
		'class_weight': decoder_config.train.class_weight,
		'sampling_mode': decoder_config.train.sampling_mode,
		'paired_to_hard_config': _identity(config.hard_full_config),
		'periodic_scientific_identity': _mapping(
			_mapping(periodic.get('identity'), 'periodic identity').get(
				'scientific_identity'
			),
			'periodic scientific identity',
		),
		'hard_model_tag': _model_tag(hard),
	}


def _reference_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
	hard_decoder_path = (
		config.workspace_root
		/ 'experiments/f3/facies_benchmark_v1/'
		/ '95_strat_hmm_multi_head_k6810_low_label_v1'
		/ '01_run_multi_head_voxel_label_budget.yaml'
	)
	if not hard_decoder_path.is_file():
		raise FileNotFoundError(hard_decoder_path)
	hard_config = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(hard_decoder_path)
	)
	dataset_rows = decoder_runner._dataset_rows(hard_config)
	reference = inspect_f3_lithology_voxel_label_budget_mae_reference_run(
		hard_config.dataset_manifest,
		hard_config.original_run_manifest,
		include_historical_m1=False,
	)
	current = decoder_runner._current_k6_rows(hard_config, dataset_rows)
	hard_payload = _read_json(decoder_runner.multi_head_run_manifest_path(hard_config))
	hard_rows = [
		row
		for row in _mapping_rows(hard_payload.get('rows'), 'hard rows')
		if row.get('model_role') == 'mh_nocons'
	]
	if len(hard_rows) != 15 or len(current) != 15 or len(reference.jobs) != 15:
		raise ValueError('read-only hard/current-K6/MAE matrices must contain 15 rows')
	if any(
		row.get('model_tag') != 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
		for row in hard_rows
	):
		raise ValueError('hard reference identity mismatch')
	for key, row in current.items():
		decoder_runner.control._validate_paired_identity(
			row,
			reference=reference,
			dataset_row=dataset_rows[key],
			reference_roles=('mae',),
		)
	hard_by_key = {
		(str(row['budget_id']), int(row['subsample_seed'])): row for row in hard_rows
	}
	if set(hard_by_key) != set(current):
		raise ValueError('hard reference condition matrix mismatch')
	for key, row in current.items():
		decoder_runner._validate_candidate_pairing(
			row=hard_by_key[key],
			current_reference=row,
			historical_reference=reference,
			dataset_row=dataset_rows[key],
		)
	center_raw = load_config(config.center_trace_masked_config)
	center_config = importlib.import_module(
		'seis_ssl_cluster.config.f3_lithology_voxel_label_budget_center_trace_masked'
	).config_from_mapping(center_raw)
	center_rows = (
		fixed_runner.load_f3_lithology_voxel_label_budget_center_trace_masked_rows(
			center_config
		)
	)
	if len(center_rows) != 15:
		raise ValueError('fixed center-trace reference matrix must contain 15 rows')
	if any(
		row.get('model_role') != 'mh_ctmask010_nocons'
		or row.get('model_tag')
		!= 'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
		for row in center_rows
	):
		raise ValueError('fixed center-trace reference identity mismatch')
	center_by_key = {
		(str(row['budget_id']), int(row['subsample_seed'])): row for row in center_rows
	}
	if set(center_by_key) != set(current):
		raise ValueError('fixed center-trace condition matrix mismatch')
	for key, row in current.items():
		for other in (hard_by_key[key], center_by_key[key]):
			for identity_key in decoder_runner.control.PAIR_IDENTITY_KEYS:
				if other.get(identity_key) != row.get(identity_key):
					raise ValueError(
						f'read-only paired identity mismatch: {key!r}/{identity_key}'
					)
	return (
		{
			'dataset_manifest': _identity(hard_config.dataset_manifest),
			'original_run_manifest': _identity(hard_config.original_run_manifest),
			'current_k6_run_manifest': _identity(hard_config.current_k6_run_manifest),
			'hard_multi_head_run_manifest': _identity(
				decoder_runner.multi_head_run_manifest_path(hard_config)
			),
			'mae_rows': 15,
			'current_k6_rows': 15,
			'hard_mh_nocons_rows': 15,
		},
		{
			'roles': [
				'mae',
				'm1_current_k6',
				'mh_nocons',
				'mh_ctmask010_nocons',
				MODEL_ID,
			],
			'budgets': list(hard_config.budgets),
			'subsample_seeds': list(hard_config.subsample_seeds),
			'decoder_seeds': {
				str(seed): hard_config.decoder_seed(seed)
				for seed in hard_config.subsample_seeds
			},
			'candidate_job_count': 15,
			'candidate_decoder_seed_policy': '42000 + subsample_seed',
			'pair_identity_keys': list(decoder_runner.control.PAIR_IDENTITY_KEYS),
		},
		{
			'run_manifest': _identity(
				fixed_runner.center_trace_masked_run_manifest_path(center_config)
			),
			'model_role': 'mh_ctmask010_nocons',
			'model_tag': 'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1',
			'row_count': 15,
		},
	)


def _canonical_valid_mask_identities(
	artifact_root: Path,
) -> dict[str, dict[str, str]]:
	tags = {
		'mae': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		'current_k6': 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1',
		'mh_nocons': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		'center_trace_masked': 'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1',
	}
	result = {}
	for role, tag in tags.items():
		path = output_paths(
			artifact_root / 'embeddings/f3/facies_benchmark_v1' / tag / 'overlap_x16',
			'f3_facies_benchmark',
		).valid_tokens
		if not path.is_file():
			raise FileNotFoundError(path)
		result[role] = {'path': str(path), 'sha256': file_sha256(path)}
	if len({item['sha256'] for item in result.values()}) != 1:
		raise ValueError('canonical valid-token masks are not bitwise identical')
	return result


def _source_identity_drift(
	existing: Mapping[str, object], current: Mapping[str, object]
) -> bool:
	for key in ('source_hard_manifest', 'reference_run_manifests'):
		if existing.get(key) != current.get(key):
			return True
	for section in ('candidate', 'periodic_refresh_handoff', 'decoder_contract'):
		left, right = existing.get(section), current.get(section)
		if isinstance(left, Mapping) and isinstance(right, Mapping):
			for key in (
				'periodic_refresh_full_config',
				'pretraining_handoff',
				'paired_to_hard_config',
			):
				if key in left and left.get(key) != right.get(key):
					return True
	return False


def _revalidate_persisted_audit(payload: Mapping[str, object]) -> None:
	"""Recheck all recorded live identities without changing owned output."""
	for section, keys in (
		('candidate', ('pretraining_handoff',)),
		('checkpoint', ('path', 'latest_path')),
		('embedding', ('metadata_path', 'embeddings_path', 'valid_tokens_path')),
		('generation_chain', ('path',)),
		('source_hard_manifest', ('path',)),
	):
		value = payload.get(section)
		if section == 'source_hard_manifest':
			_identity_matches_live(value, label=section)
			continue
		mapping = _mapping(value, f'persisted {section}')
		for key in keys:
			if key == 'pretraining_handoff':
				_identity_matches_live(mapping.get(key), label=f'{section}.{key}')
			elif key.endswith('_path') or key in {'path', 'latest_path'}:
				path = _required_live_path(mapping.get(key), f'{section}.{key}')
				expected_key = (
					'sha256'
					if key == 'path'
					else 'latest_sha256'
					if key == 'latest_path'
					else key.removesuffix('_path') + '_sha256'
				)
				if mapping.get(expected_key) != file_sha256(path):
					raise ValueError(f'persisted {section}.{key} identity is stale')
	references = _mapping(
		payload.get('reference_run_manifests'), 'persisted reference manifests'
	)
	for key, value in references.items():
		if isinstance(value, Mapping) and 'path' in value:
			_identity_matches_live(value, label=f'reference_run_manifests.{key}')
	fixed = _mapping(
		payload.get('fixed_center_trace_reference'),
		'persisted fixed center-trace reference',
	)
	_identity_matches_live(
		fixed.get('run_manifest'), label='fixed center-trace run manifest'
	)
	hard = _mapping(payload.get('hard_baseline'), 'persisted hard baseline')
	for key in ('config', 'handoff', 'target_manifest'):
		_identity_matches_live(hard.get(key), label=f'hard baseline {key}')


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
		raise RuntimeError(
			'unable to record periodic-refresh audit git state'
		) from error
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


def _identity_matches_live(value: object, *, label: str) -> None:
	identity = _mapping(value, label)
	path = _required_live_path(identity.get('path'), label)
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} identity mismatch')


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _mapping_rows(value: object, label: str) -> tuple[Mapping[str, object], ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a list')
	return tuple(_mapping(item, label) for item in value)


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	return _mapping(payload, f'JSON object {path}')


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
	'GENERATION_IDS',
	'MODEL_ID',
	'MODEL_TAG',
	'REFRESH_EPOCHS',
	'SCHEMA_VERSION',
	'F3CenterTraceMaskedPeriodicRefreshScreeningAuditConfig',
	'F3CenterTraceMaskedPeriodicRefreshScreeningAuditResult',
	'audit_f3_center_trace_masked_periodic_refresh_screening',
	'f3_center_trace_masked_periodic_refresh_screening_audit_config_from_mapping',
	'load_f3_center_trace_masked_periodic_refresh_screening_audit',
	'load_f3_center_trace_masked_periodic_refresh_screening_audit_config',
	'validate_f3_center_trace_masked_periodic_refresh_screening_audit_binding',
]
