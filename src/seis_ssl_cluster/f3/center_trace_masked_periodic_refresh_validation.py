"""Strict validation and handoff publication for experiment 107."""
# ruff: noqa: E501, C901, PLR0912, PLR0913, PLR0915, S603, S607
# ruff: noqa: CPY001

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.clustering.features import (
	discover_embedding_inputs,
	file_sha256,
)
from seis_ssl_cluster.clustering.stratigraphic_hmm import edge_margin_mask_for_shape
from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.config.pretraining import _multi_head_target_hashes
from seis_ssl_cluster.embedding.writer import output_paths
from seis_ssl_cluster.f3 import (
	center_trace_masked_pretraining_validation as center_validation,
)
from seis_ssl_cluster.f3 import (
	multi_head_pretraining_validation as hard_validation,
)
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.periodic_refresh import (
	INITIAL_GENERATION_ID,
	load_periodic_refresh_generation,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	_periodic_fixed_preprocessing_identity_sha256,
	validate_stratigraphy_checkpoint_payload,
)

_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'experiment_root',
		'target_manifest',
		'hard_full_config',
		'hard_handoff',
		'center_trace_masked_full_config',
		'center_trace_masked_handoff',
		'periodic_refresh_smoke_config',
		'periodic_refresh_full_config',
		'periodic_refresh_embedding_config',
	}
)
_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_'
	'topblock1_distill_v1'
)
_VARIANT = 'ctmask010_refresh3ep_hmm2_nocons'
_ROLE = 'multi_head_center_trace_masked_periodic_hmm_refresh_hard_pretext'
_TARGET_REPRESENTATION = 'hard_viterbi_labels_v1'
_HEAD_KS = [6, 8, 10]
_SCHEDULE = [2, 5, 8, 11, 14, 17, 20]
_GENERATION_IDS = [
	'refresh_0000_initial',
	'refresh_0001_epoch002',
	'refresh_0002_epoch005',
	'refresh_0003_epoch008',
	'refresh_0004_epoch011',
	'refresh_0005_epoch014',
	'refresh_0006_epoch017',
	'refresh_0007_epoch020',
]
_CENTER_HANDOFF_ALLOWED_DIFFERENCES = [
	'paths.output_root',
	'identity.model_tag',
	'identity.scientific_identity center-trace fields',
	'spatial_context',
]
_PERIODIC_SCIENTIFIC_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'model_role',
		'target_refresh_semantics',
		'refresh_schedule_semantics',
		'refresh_after_epochs',
		'hmm_iterations_per_refresh',
		'embedding_source',
		'embedding_mode',
		'refresh_embedding_semantics',
		'center_initialization',
		'center_update',
		'center_update_semantics',
		'preprocessing_policy',
		'target_activation_policy',
		'empty_state_policy',
		'checkpoint_selection_policy',
		'initial_hard_target_manifest_sha256',
		'initial_hmm_artifacts',
		'fixed_preprocessor_sha256',
		'fixed_residualizer_sha256',
		'fixed_clustering_config_sha256',
		'source_embedding_metadata_sha256',
		'source_valid_token_hashes',
		'feature_dimension',
		'generation_root',
	}
)
_HANDOFF_TYPE = 'f3_center_trace_masked_periodic_refresh_pretraining_handoff'
_EXECUTION_ARTIFACT_TYPE = 'f3_center_trace_masked_periodic_refresh_execution'
_EXECUTION_EVIDENCE_FILENAME = (
	'.f3_center_trace_masked_periodic_refresh_execution.json'
)
_PHASE_EVIDENCE_PREFIX = '.f3_center_trace_masked_periodic_refresh'
_VALIDATION_ERRORS = (
	OSError,
	TypeError,
	ValueError,
	RuntimeError,
	EOFError,
	pickle.UnpicklingError,
)


@dataclass(frozen=True)
class F3CenterTraceMaskedPeriodicRefreshValidationConfig:
	"""Closed paths needed to validate experiment 107."""

	artifact_root: Path
	experiment_root: Path
	target_manifest: Path
	hard_full_config: Path
	hard_handoff: Path
	center_trace_masked_full_config: Path
	center_trace_masked_handoff: Path
	periodic_refresh_smoke_config: Path
	periodic_refresh_full_config: Path
	periodic_refresh_embedding_config: Path


@dataclass(frozen=True)
class F3CenterTraceMaskedPeriodicRefreshValidationResult:
	"""Evidence for one phase and the optional final handoff."""

	phase: str
	evidence: Mapping[str, object]
	published_handoff: Path | None


def f3_center_trace_masked_periodic_refresh_validation_config_from_mapping(
	config: Mapping[str, object],
) -> F3CenterTraceMaskedPeriodicRefreshValidationConfig:
	"""Resolve the deliberately closed experiment-107 validation schema."""
	if not isinstance(config, Mapping):
		raise TypeError('periodic refresh validation config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(
			f'unknown periodic refresh validation keys: {sorted(unknown)!r}'
		)
	if missing:
		raise ValueError(
			f'missing periodic refresh validation keys: {sorted(missing)!r}'
		)

	def path(key: str, *, must_exist: bool) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		result = Path(value).resolve()
		if must_exist and not result.is_file():
			raise FileNotFoundError(f'{key} is missing: {result}')
		return result

	result = F3CenterTraceMaskedPeriodicRefreshValidationConfig(
		artifact_root=path('artifact_root', must_exist=False),
		experiment_root=path('experiment_root', must_exist=False),
		target_manifest=path('target_manifest', must_exist=True),
		hard_full_config=path('hard_full_config', must_exist=True),
		hard_handoff=path('hard_handoff', must_exist=True),
		center_trace_masked_full_config=path(
			'center_trace_masked_full_config', must_exist=True
		),
		center_trace_masked_handoff=path(
			'center_trace_masked_handoff', must_exist=True
		),
		periodic_refresh_smoke_config=path(
			'periodic_refresh_smoke_config', must_exist=True
		),
		periodic_refresh_full_config=path(
			'periodic_refresh_full_config', must_exist=True
		),
		periodic_refresh_embedding_config=path(
			'periodic_refresh_embedding_config', must_exist=True
		),
	)
	if not result.artifact_root.is_dir() or not result.experiment_root.is_dir():
		raise FileNotFoundError(
			'artifact_root and experiment_root must be existing directories'
		)
	ensure_under_root(
		result.experiment_root,
		root=result.artifact_root,
		label='experiment_root',
	)
	for label, value in (
		('target_manifest', result.target_manifest),
		('hard_handoff', result.hard_handoff),
		('center_trace_masked_handoff', result.center_trace_masked_handoff),
	):
		ensure_under_root(value, root=result.artifact_root, label=label)
	return result


def load_f3_center_trace_masked_periodic_refresh_validation_config(
	path: str | Path,
) -> F3CenterTraceMaskedPeriodicRefreshValidationConfig:
	"""Load the experiment-107 validation YAML."""
	return (
		f3_center_trace_masked_periodic_refresh_validation_config_from_mapping(
			load_config(path)
		)
	)


def load_f3_center_trace_masked_periodic_refresh_handoff(
	path: str | Path,
) -> Mapping[str, object]:
	"""Load the complete, versioned experiment-107 PASS handoff."""
	payload = _mapping(_json(Path(path)), 'periodic refresh handoff')
	if set(payload) != {
		'artifact_type',
		'schema_version',
		'status',
		'model_tag',
		'variant',
		'primary_checkpoint_role',
		'targets',
		'checkpoint',
		'embedding',
		'fixed_preprocessing',
		'execution',
	}:
		raise ValueError('periodic refresh handoff top-level keys mismatch')
	if (
		payload.get('artifact_type') != _HANDOFF_TYPE
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
		or payload.get('model_tag') != _MODEL_TAG
		or payload.get('variant') != _VARIANT
		or payload.get('primary_checkpoint_role') != 'completed_final_selected'
	):
		raise ValueError('periodic refresh handoff identity mismatch')
	targets = _mapping(payload['targets'], 'periodic refresh handoff targets')
	if set(targets) != {
		'initial_hard_target_manifest',
		'initial_per_head_target_hashes',
		'final_target_manifest',
		'final_generation',
		'periodic_refresh_chain',
		'valid_token_hashes',
	}:
		raise ValueError('periodic refresh handoff target keys mismatch')
	for key in (
		'initial_hard_target_manifest',
		'final_target_manifest',
		'periodic_refresh_chain',
	):
		_validate_reference(targets[key], f'handoff targets.{key}')
	_validate_target_hashes(targets['initial_per_head_target_hashes'])
	_validate_valid_hashes(targets['valid_token_hashes'])
	final_generation = _mapping(
		targets['final_generation'], 'handoff final generation'
	)
	if set(final_generation) != {
		'generation_index',
		'generation_id',
		'manifest',
		'content_sha256',
	}:
		raise ValueError('periodic refresh handoff final generation keys mismatch')
	_validate_reference(final_generation['manifest'], 'handoff final generation manifest')
	_require_sha256(
		final_generation['content_sha256'],
		'handoff final generation content hash',
	)
	if final_generation['generation_index'] != 7 or final_generation['generation_id'] != (
		_GENERATION_IDS[-1]
	):
		raise ValueError('periodic refresh handoff final generation identity mismatch')
	checkpoint = _mapping(payload['checkpoint'], 'periodic refresh handoff checkpoint')
	if set(checkpoint) != {
		'path',
		'sha256',
		'latest_path',
		'latest_sha256',
		'epoch',
		'global_step',
		'schema_version',
		'scientific_identity_sha256',
		'target_refresh_state_sha256',
		'optimizer_group_identity',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
	}:
		raise ValueError('periodic refresh handoff checkpoint keys mismatch')
	for key in (
		'path',
		'latest_path',
	):
		if not isinstance(checkpoint.get(key), str) or not checkpoint[key]:
			raise TypeError(f'handoff checkpoint.{key} is missing')
	for key in (
		'sha256',
		'latest_sha256',
		'scientific_identity_sha256',
		'target_refresh_state_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
	):
		_require_sha256(checkpoint.get(key), f'handoff checkpoint.{key}')
	_validate_reference(
		{'path': checkpoint['path'], 'sha256': checkpoint['sha256']},
		'handoff checkpoint.selected',
	)
	_validate_reference(
		{'path': checkpoint['latest_path'], 'sha256': checkpoint['latest_sha256']},
		'handoff checkpoint.latest',
	)
	if checkpoint['schema_version'] != 8 or checkpoint['epoch'] != 25:
		raise ValueError('periodic refresh handoff checkpoint identity mismatch')
	embedding = _mapping(payload['embedding'], 'periodic refresh handoff embedding')
	if set(embedding) != {
		'root',
		'metadata_path',
		'metadata_sha256',
		'embeddings_path',
		'embeddings_sha256',
		'valid_tokens_path',
		'valid_tokens_sha256',
		'embeddings_shape',
		'embeddings_dtype',
		'valid_tokens_shape',
		'valid_tokens_dtype',
		'finite_valid_count',
	}:
		raise ValueError('periodic refresh handoff embedding keys mismatch')
	for key in ('metadata_path', 'embeddings_path', 'valid_tokens_path'):
		if not isinstance(embedding.get(key), str) or not embedding[key]:
			raise TypeError(f'handoff embedding.{key} is missing')
	for key in (
		'metadata_sha256',
		'embeddings_sha256',
		'valid_tokens_sha256',
	):
		_require_sha256(embedding.get(key), f'handoff embedding.{key}')
	_validate_reference(
		{'path': embedding['metadata_path'], 'sha256': embedding['metadata_sha256']},
		'handoff embedding.metadata',
	)
	_validate_reference(
		{
			'path': embedding['embeddings_path'],
			'sha256': embedding['embeddings_sha256'],
		},
		'handoff embedding.embeddings',
	)
	_validate_reference(
		{
			'path': embedding['valid_tokens_path'],
			'sha256': embedding['valid_tokens_sha256'],
		},
		'handoff embedding.valid_tokens',
	)
	if (
		embedding['embeddings_shape'] != [76, 113, 32, 384]
		or embedding['embeddings_dtype'] != 'float16'
		or embedding['valid_tokens_shape'] != [76, 113, 32]
		or embedding['valid_tokens_dtype'] != 'bool'
		or not _positive_int(embedding['finite_valid_count'])
	):
		raise ValueError('periodic refresh handoff embedding identity mismatch')
	fixed = _mapping(
		payload['fixed_preprocessing'], 'periodic refresh handoff preprocessing'
	)
	if set(fixed) != {
		'initial_hmm_artifacts',
		'fixed_preprocessor_sha256',
		'fixed_residualizer_sha256',
		'fixed_clustering_config_sha256',
		'source_embedding_metadata_sha256',
		'source_valid_token_hashes',
		'feature_dimension',
	}:
		raise ValueError('periodic refresh handoff preprocessing keys mismatch')
	_validate_valid_hashes(fixed['source_valid_token_hashes'])
	for key in (
		'fixed_preprocessor_sha256',
		'fixed_residualizer_sha256',
		'fixed_clustering_config_sha256',
		'source_embedding_metadata_sha256',
	):
		_require_sha256_or_none(fixed.get(key), f'handoff preprocessing.{key}')
	_validate_fixed_preprocessing_references(
		fixed['initial_hmm_artifacts'], 'handoff preprocessing'
	)
	execution = _mapping(payload['execution'], 'periodic refresh handoff execution')
	if set(execution) != {'before', 'after'}:
		raise ValueError('periodic refresh handoff execution keys mismatch')
	_validate_execution_state(execution['before'], 'handoff execution.before')
	_validate_execution_state(execution['after'], 'handoff execution.after')
	return payload


def validate_f3_center_trace_masked_periodic_refresh(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	phase: str,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3CenterTraceMaskedPeriodicRefreshValidationResult:
	"""Validate one immutable phase of experiment 107."""
	if phase not in {'inputs', 'smoke', 'checkpoints', 'complete'}:
		raise ValueError('phase must be inputs, smoke, checkpoints, or complete')
	try:
		inputs = _inputs_evidence(config)
		if phase == 'inputs':
			execution = _start_execution_evidence(
				config,
				dry_run=dry_run,
				quarantine_invalid=quarantine_invalid,
			)
			evidence = {'status': 'PASS', **inputs, 'execution': execution}
			if not dry_run:
				evidence['phase_evidence_path'] = str(
					_write_phase_evidence(
						config,
						phase=phase,
						evidence=evidence,
						only_missing=only_missing,
						quarantine_invalid=quarantine_invalid,
					)
				)
			return F3CenterTraceMaskedPeriodicRefreshValidationResult(
				phase, evidence, None
			)

		smoke = _training_config(config.periodic_refresh_smoke_config)
		if phase == 'smoke':
			smoke_evidence = _smoke_evidence(
				config,
				inputs=inputs,
				smoke=smoke,
				quarantine_invalid=quarantine_invalid,
				dry_run=dry_run,
			)
			execution = _update_execution_evidence(
				config, phase=phase, dry_run=dry_run
			)
			evidence = {
				'status': 'PASS',
				**inputs,
				'smoke': smoke_evidence,
				'execution': execution,
			}
			if not dry_run:
				evidence['phase_evidence_path'] = str(
					_write_phase_evidence(
						config,
						phase=phase,
						evidence=evidence,
						only_missing=only_missing,
						quarantine_invalid=quarantine_invalid,
					)
				)
			return F3CenterTraceMaskedPeriodicRefreshValidationResult(
				phase, evidence, None
			)

		checkpoints = _checkpoint_evidence(
			config,
			inputs=inputs,
			quarantine_invalid=quarantine_invalid,
			dry_run=dry_run,
		)
		evidence = {'status': 'PASS', **inputs, **checkpoints}
		if phase == 'checkpoints':
			if not dry_run:
				_write_checkpoint_report(
					Path(str(checkpoints['checkpoint']['root'])),
					checkpoint_sha256=str(
						checkpoints['checkpoint']['selected_sha256']
					),
					only_missing=only_missing,
					quarantine_invalid=quarantine_invalid,
				)
			return F3CenterTraceMaskedPeriodicRefreshValidationResult(
				phase, evidence, None
			)

		evidence['embedding'] = _embedding_evidence(
			config,
			inputs=inputs,
			checkpoint=checkpoints['checkpoint'],
		)
		# Complete validation must always bind the final handoff to the live
		# smoke evidence, including when the caller is only inspecting a plan.
		_validate_smoke_phase_evidence(config, inputs=inputs)
		evidence['execution'] = _update_execution_evidence(
			config, phase=phase, dry_run=dry_run
		)
		handoff = _handoff(evidence)
		handoff_path = (
			Path(str(checkpoints['checkpoint']['root']))
			/ 'preflight'
			/ 'periodic_refresh_handoff.json'
		)
		if dry_run:
			return F3CenterTraceMaskedPeriodicRefreshValidationResult(
				phase, evidence, None
			)
		_publish_handoff(
			handoff_path,
			handoff,
			only_missing=only_missing,
			quarantine_invalid=quarantine_invalid,
		)
		return F3CenterTraceMaskedPeriodicRefreshValidationResult(
			phase, evidence, handoff_path
		)
	except _VALIDATION_ERRORS as error:
		if not dry_run:
			raise
		return F3CenterTraceMaskedPeriodicRefreshValidationResult(
			phase,
			{'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'},
			None,
		)


def _inputs_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
) -> dict[str, object]:
	target = load_multi_head_target_manifest(config.target_manifest)
	hard = _training_config(config.hard_full_config)
	center = _training_config(config.center_trace_masked_full_config)
	smoke = _training_config(config.periodic_refresh_smoke_config)
	full = _training_config(config.periodic_refresh_full_config)
	extraction = resolve_embedding_extraction_config(
		load_config(config.periodic_refresh_embedding_config)
	)
	target_evidence = _target_evidence(config, target=target, full=full)
	baseline = _baseline_evidence(
		config,
		hard=hard,
		center=center,
		center_handoff_path=config.center_trace_masked_handoff,
		hard_handoff_path=config.hard_handoff,
	)
	initialization_checkpoints = _validate_initialization_checkpoint_hashes(
		baseline_handoff=_mapping(baseline['hard_handoff'], 'hard baseline handoff'),
		trainings={
			'hard': hard,
			'center': center,
			'full': full,
			'smoke': smoke,
		},
	)
	_config_contract(
		config,
		target=target,
		center=center,
		full=full,
		smoke=smoke,
		extraction=extraction,
		center_handoff=baseline['center_handoff'],
	)
	return {
		'target_manifest': target_evidence,
		'baseline': baseline,
		'full_config': _reference(config.periodic_refresh_full_config),
		'smoke_config': _reference(config.periodic_refresh_smoke_config),
		'embedding_config': _reference(config.periodic_refresh_embedding_config),
		'fixed_preprocessing': _fixed_preprocessing_evidence(full),
		'initialization_checkpoints': initialization_checkpoints,
	}


def _target_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	target: Mapping[str, object],
	full: Mapping[str, object],
) -> dict[str, object]:
	if (
		target.get('artifact_type') != 'strat_hmm_multi_head_target_manifest'
		or target.get('schema_version') not in {1, 2}
		or target.get('head_ks') != _HEAD_KS
	):
		raise ValueError('periodic refresh target must be the K6/K8/K10 hard manifest')
	if _manifest_path(full) != config.target_manifest:
		raise ValueError('periodic full config does not use the supplied target manifest')
	hashes = _multi_head_target_hashes(target)
	_validate_target_hashes(hashes)
	_validate_target_file_references(target)
	manifest_reference = _reference(config.target_manifest)
	if _scientific(full)['target_manifest_sha256'] != manifest_reference['sha256']:
		raise ValueError('periodic target manifest SHA-256 identity mismatch')
	return {
		'path': manifest_reference['path'],
		'sha256': manifest_reference['sha256'],
		'head_ks': list(_HEAD_KS),
		'per_head_target_hashes': hashes,
		'common_valid_token_hashes': dict(
			_mapping(target['common'], 'target common')['valid_tokens_sha256']
		),
		'source_embedding': _source_embedding_evidence(target),
	}


def _baseline_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	hard: Mapping[str, object],
	center: Mapping[str, object],
	center_handoff_path: Path,
	hard_handoff_path: Path,
) -> dict[str, object]:
	if _model_tag(hard) != 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1':
		raise ValueError('hard baseline model tag mismatch')
	if _model_tag(center) != (
		'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
	):
		raise ValueError('fixed center-trace baseline model tag mismatch')
	if _manifest_path(hard) != config.target_manifest or _manifest_path(center) != (
		config.target_manifest
	):
		raise ValueError('fixed baseline target manifest binding mismatch')
	hard_handoff = hard_validation.load_f3_multi_head_pretraining_handoff(
		hard_handoff_path
	)
	center_handoff = center_validation.load_f3_center_trace_masked_pretraining_handoff(
		center_handoff_path
	)
	if center_handoff['targets']['allowed_differences'] != (
		_CENTER_HANDOFF_ALLOWED_DIFFERENCES
	):
		raise ValueError('fixed center-trace baseline allowed-difference contract drift')
	center_targets = _mapping(center_handoff['targets'], 'center-trace baseline targets')
	if center_targets['target_manifest'] != _reference(config.target_manifest):
		raise ValueError('fixed center-trace handoff target manifest is stale')
	if _mapping(center_handoff['checkpoint'], 'center handoff checkpoint').get(
		'schema_version'
	) != 7:
		raise ValueError('fixed center-trace baseline must remain schema 7')
	for label, payload in (('hard handoff', hard_handoff), ('center handoff', center_handoff)):
		_validate_handoff_references(payload, label)
	return {
		'hard_config': _reference(config.hard_full_config),
		'hard_handoff': hard_handoff,
		'center_config': _reference(config.center_trace_masked_full_config),
		'center_handoff': center_handoff,
		'allowed_differences': list(_CENTER_HANDOFF_ALLOWED_DIFFERENCES),
		'initial_student_state_sha256': center_targets[
			'initial_student_state_sha256'
		],
		'initial_head_state_sha256': center_targets['initial_head_state_sha256'],
		'initial_spatial_context_state_sha256': center_targets[
			'initial_spatial_context_state_sha256'
		],
	}


def _validate_initialization_checkpoint_hashes(
	*,
	baseline_handoff: Mapping[str, object],
	trainings: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
	"""Bind every live initialization file to the fixed hard-baseline hash."""
	baseline_checkpoint = _mapping(
		baseline_handoff['checkpoint'], 'hard baseline checkpoint'
	)
	baseline_path = Path(str(baseline_checkpoint['path'])).resolve()
	if not baseline_path.is_file():
		raise FileNotFoundError(
			f'hard baseline checkpoint is missing: {baseline_path}'
		)
	baseline_sha256 = baseline_checkpoint.get('sha256')
	_require_sha256(baseline_sha256, 'hard baseline checkpoint.sha256')
	if file_sha256(baseline_path) != baseline_sha256:
		raise ValueError('hard baseline checkpoint hash drift')
	baseline_payload = load_checkpoint(baseline_path, map_location='cpu')
	baseline_identity = _mapping(
		baseline_payload['stratigraphy_checkpoint'],
		'hard baseline checkpoint identity',
	)
	expected: dict[str, str] = {}
	for identity_key in (
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		digest = baseline_identity.get(identity_key)
		_require_sha256(
			digest,
		f'hard baseline checkpoint.{identity_key}',
	)
		expected[identity_key] = str(digest)

	references: dict[str, dict[str, dict[str, str]]] = {}
	for label, training in trainings.items():
		label_references: dict[str, dict[str, str]] = {}
		for section, key, identity_key in (
			('teacher', 'checkpoint', 'teacher_checkpoint_sha256'),
			('student', 'init_checkpoint', 'student_init_checkpoint_sha256'),
		):
			section_value = _mapping(training[section], f'{label} {section}')
			path_value = section_value.get(key)
			if not isinstance(path_value, str) or not path_value:
				raise TypeError(f'{label} {section}.{key} must be a path')
			path = Path(path_value).resolve()
			if not path.is_file():
				raise FileNotFoundError(
					f'{label} {section} checkpoint is missing: {path}'
				)
			digest = file_sha256(path)
			if digest != expected[identity_key]:
				raise ValueError(
					f'periodic {label} {section} checkpoint SHA-256 drift'
					)
			label_references[identity_key] = {'path': str(path), 'sha256': digest}
		references[label] = label_references
	return {
		'expected_from_hard_baseline': {
			'path': str(baseline_path),
			'sha256': str(baseline_sha256),
		},
		'expected_initialization_hashes': expected,
		'configs': references,
	}


def _config_contract(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	target: Mapping[str, object],
	center: Mapping[str, object],
	full: Mapping[str, object],
	smoke: Mapping[str, object],
	extraction: Mapping[str, object],
	center_handoff: Mapping[str, object],
) -> None:
	identity = _scientific(full)
	for key, expected in {
		'experiment_role': _ROLE,
		'variant': _VARIANT,
		'model_role': 'mh_ctmask010_refresh3ep_hmm2_nocons',
		'target_representation': _TARGET_REPRESENTATION,
		'target_refresh_semantics': 'periodic_student_hmm_center_refresh_v1',
		'refresh_schedule_semantics': 'after_epochs_2_5_8_11_14_17_20_v1',
		'refresh_after_epochs': _SCHEDULE,
		'hmm_iterations_per_refresh': 2,
		'embedding_source': 'current_student',
		'embedding_mode': 'unmasked_eval_full_survey',
		'refresh_embedding_semantics': 'current_student_unmasked_eval_full_survey_v1',
		'center_initialization': 'previous_generation',
		'center_update': 'full_mean',
		'center_update_semantics': (
			'warm_start_full_mean_two_iterations_final_decode_v1'
		),
		'preprocessing_policy': 'freeze_initial_residualizer_pca_v1',
		'target_activation_policy': 'atomic_next_epoch_activation_v1',
		'empty_state_policy': 'error',
		'checkpoint_selection_policy': 'final_completed_epoch_v1',
	}.items():
		if identity.get(key) != expected:
			raise ValueError(f'periodic scientific identity mismatch: {key}')
	if _model_tag(full) != _MODEL_TAG or _model_tag(smoke) != _MODEL_TAG:
		raise ValueError('periodic full/smoke model tag mismatch')
	if identity['head_ks'] != _HEAD_KS or identity['head_spec'] != (
		'multi_resolution_ordered_prototypes_v1'
	):
		raise ValueError('periodic head identity mismatch')
	if identity['target_head_hashes'] != _multi_head_target_hashes(target):
		raise ValueError('periodic per-head target identity mismatch')
	train = _mapping(full['train'], 'periodic full train')
	for key, expected in {
		'batch_size': 4,
		'samples_per_epoch': 4096,
		'epochs': 25,
		'lr': 0.0003,
		'encoder_lr': 0.00001,
		'seed': 42,
		'device': 'auto',
		'max_steps': None,
	}.items():
		if train.get(key) != expected:
			raise ValueError(f'periodic full training identity mismatch: {key}')
	student = _mapping(full['student'], 'periodic student')
	if student.get('unfreeze_top_blocks') != 1:
		raise ValueError('periodic student must unfreeze top block 1')
	loss = _mapping(full['loss'], 'periodic loss')
	for key, expected in {
		'prototype_weight': 1.0,
		'usage_weight': 0.005,
		'consistency_weight': 0.0,
		'distillation_weight': 0.2,
	}.items():
		if loss.get(key) != expected:
			raise ValueError(f'periodic loss identity mismatch: {key}')
	spatial = _mapping(full['spatial_context'], 'periodic spatial_context')
	if spatial.get('column_fraction') != 0.10 or spatial.get(
		'masked_prototype_weight'
	) != 0.50 or spatial.get('visible_prototype_weight') != 0.50:
		raise ValueError('periodic center-trace mask identity mismatch')
	refresh = _mapping(full['pseudo_target_refresh'], 'periodic refresh')
	if refresh.get('refresh_after_epochs') != _SCHEDULE or refresh.get(
		'hmm_iterations_per_refresh'
	) != 2:
		raise ValueError('periodic refresh schedule drift')
	if refresh.get('generation_root') != identity['generation_root']:
		raise ValueError('periodic generation root identity mismatch')
	_compare_to_center_baseline(center, full)
	_compare_full_and_smoke(full, smoke)
	full_root = _training_output_root(full, config, 'periodic full output root')
	smoke_root = _training_output_root(smoke, config, 'periodic smoke output root')
	if full_root == smoke_root:
		raise ValueError('periodic smoke/full output roots must be distinct')
	if full_root == _training_output_root(center, config, 'center baseline output root'):
		raise ValueError('periodic full root collides with fixed center-trace root')
	generation_root = Path(str(refresh['generation_root'])).resolve()
	ensure_under_root(generation_root, root=full_root, label='periodic generation_root')
	smoke_generation_root = Path(
		str(_mapping(smoke['pseudo_target_refresh'], 'smoke refresh')['generation_root'])
	).resolve()
	ensure_under_root(
		smoke_generation_root,
		root=smoke_root,
		label='periodic smoke generation_root',
	)
	if generation_root == smoke_generation_root:
		raise ValueError('periodic generation roots must be distinct')
	extraction_paths = _mapping(extraction['embeddings'], 'periodic extraction')
	selected = full_root / 'selected.pt'
	if Path(str(extraction_paths['checkpoint'])).resolve() != selected:
		raise ValueError('periodic extraction must use full selected.pt')
	extraction_root = Path(str(extraction_paths['output_dir'])).resolve()
	ensure_under_root(extraction_root, root=config.artifact_root, label='final embedding root')
	if extraction_root in (full_root, smoke_root):
		raise ValueError('periodic final embedding root collides with training root')
	if Path(str(_mapping(extraction['manifests'], 'extraction manifests')['input'])).resolve() != (
		Path(str(_mapping(full['manifests'], 'full manifests')['train'])).resolve()
	):
		raise ValueError('periodic extraction manifest does not match full training')
	if _scientific(full)['checkpoint_selection_policy'] != (
		'final_completed_epoch_v1'
	):
		raise ValueError('periodic final checkpoint selection policy drift')
	if center_handoff['targets']['allowed_differences'] != (
		_CENTER_HANDOFF_ALLOWED_DIFFERENCES
	):
		raise ValueError('baseline allowed-difference set changed')
	_validate_initial_artifacts(
		full, target, config.target_manifest, config.artifact_root
	)
	_validate_initial_artifacts(
		smoke, target, config.target_manifest, config.artifact_root
	)
	for label, training in (('full', full), ('smoke', smoke)):
		for section, key in (('teacher', 'checkpoint'), ('student', 'init_checkpoint')):
			path = Path(str(_mapping(training[section], section)[key])).resolve()
			if not path.is_file():
				raise FileNotFoundError(f'periodic {label} {section} checkpoint is missing')
		if Path(str(_mapping(training['teacher'], 'teacher')['checkpoint'])).resolve() != Path(
			str(_mapping(full['teacher'], 'teacher')['checkpoint'])
		).resolve():
			raise ValueError('periodic teacher initialization drift')
		if Path(str(_mapping(training['student'], 'student')['init_checkpoint'])).resolve() != Path(
			str(_mapping(full['student'], 'student')['init_checkpoint'])
		).resolve():
			raise ValueError('periodic student initialization drift')


def _compare_to_center_baseline(
	center: Mapping[str, object], full: Mapping[str, object]
) -> None:
	center_sci = _scientific(center)
	full_sci = _scientific(full)
	if set(full_sci) - set(center_sci) - _PERIODIC_SCIENTIFIC_FIELDS:
		raise ValueError('periodic identity adds an undeclared scientific field')
	for key in set(center_sci) & set(full_sci):
		if key in _PERIODIC_SCIENTIFIC_FIELDS:
			continue
		if center_sci[key] != full_sci[key]:
			raise ValueError(f'periodic config drifted from experiment 104: {key}')
	left = deepcopy(center)
	right = deepcopy(full)
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		_mapping(value['identity'], 'identity').pop('model_tag', None)
		_mapping(value['identity'], 'identity').pop('scientific_identity', None)
	_mapping(right, 'periodic config').pop('pseudo_target_refresh', None)
	if left != right:
		raise ValueError(
			'periodic config differs from experiment 104 outside the closed '
			'allowed-difference set'
		)


def _compare_full_and_smoke(
	full: Mapping[str, object], smoke: Mapping[str, object]
) -> None:
	left = deepcopy(full)
	right = deepcopy(smoke)
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		identity = _mapping(value['identity'], 'identity')
		_mapping(identity['scientific_identity'], 'scientific_identity').pop(
			'generation_root', None
		)
		runtime_identity = identity.get('runtime_identity')
		if runtime_identity is not None:
			_mapping(runtime_identity, 'runtime identity').pop('device', None)
		_mapping(value['train'], 'train').pop('device', None)
		_mapping(value['train'], 'train').pop('max_steps', None)
		_mapping(value['pseudo_target_refresh'], 'refresh').pop(
			'generation_root', None
		)
	if left != right:
		raise ValueError('periodic smoke config drifted outside runtime/root overrides')
	if _mapping(smoke['train'], 'smoke train').get('device') != 'cpu':
		raise ValueError('periodic smoke YAML must declare device=cpu')


def _validate_initial_artifacts(
	training: Mapping[str, object],
	target: Mapping[str, object],
	target_manifest_path: Path,
	artifact_root: Path,
) -> None:
	scientific = _scientific(training)
	artifacts = _mapping(
		scientific['initial_hmm_artifacts'], 'initial HMM artifacts'
	)
	common = _mapping(artifacts['common'], 'initial HMM common artifacts')
	for key in ('clustering_config', 'preprocessor', 'residualizer', 'source_embedding_metadata'):
		value = common.get(key)
		if value is None:
			if key == 'residualizer' and scientific['fixed_residualizer_sha256'] is None:
				continue
			raise ValueError(f'initial HMM common artifact is missing: {key}')
		_validate_reference(value, f'initial HMM common {key}')
	heads = _mapping(artifacts['heads'], 'initial HMM heads')
	if set(heads) != {'6', '8', '10'}:
		raise ValueError('initial HMM heads must be exactly K=6/8/10')
	for key in ('6', '8', '10'):
		entry = _mapping(heads[key], f'initial HMM head {key}')
		for field in ('model_metadata', 'hmm_model', 'centers'):
			_validate_reference(entry[field], f'initial HMM {key}.{field}')
	_validate_canonical_initial_artifacts(
		artifacts,
		target=target,
		target_manifest_path=target_manifest_path,
		artifact_root=artifact_root,
	)
	if scientific['fixed_preprocessor_sha256'] != _reference_hash(common['preprocessor']):
		raise ValueError('fixed preprocessor hash drift')
	if scientific['fixed_clustering_config_sha256'] != _reference_hash(
		common['clustering_config']
	):
		raise ValueError('fixed clustering config hash drift')
	if scientific['source_embedding_metadata_sha256'] != _reference_hash(
		common['source_embedding_metadata']
	):
		raise ValueError('source embedding metadata hash drift')
	target_source = _source_embedding_evidence(target)
	metadata_ref = _mapping(
		target_source['f3_facies_benchmark']['metadata'],
		'target source metadata',
	)
	if Path(str(metadata_ref['path'])).resolve() != Path(
		str(_mapping(common['source_embedding_metadata'], 'source metadata')['path'])
	).resolve():
		raise ValueError('source embedding metadata path drift')
	head_metadata = _json(
		Path(str(_mapping(heads['6'], 'head 6')['model_metadata']['path']))
	)
	embedding_inputs = _mapping(head_metadata, 'model metadata').get('embedding_inputs')
	if not isinstance(embedding_inputs, list) or len(embedding_inputs) != 1:
		raise ValueError('initial HMM metadata must bind one source embedding')
	input_entry = _mapping(embedding_inputs[0], 'initial embedding input')
	for field, target_key in (
		('embeddings_path', 'embeddings'),
		('valid_tokens_path', 'valid_tokens'),
		('metadata_path', 'metadata'),
	):
		input_path = Path(str(input_entry[field])).resolve()
		target_path = Path(str(target_source['f3_facies_benchmark'][target_key]['path'])).resolve()
		if input_path != target_path:
			raise ValueError(f'initial HMM source {field} path drift')
		if field == 'metadata_path' and input_entry['metadata_sha256'] != metadata_ref['sha256']:
			raise ValueError('initial HMM source metadata hash drift')
	source_valid = np.load(
		Path(str(target_source['f3_facies_benchmark']['valid_tokens']['path'])),
		mmap_mode='r',
		allow_pickle=False,
	)
	common_target = _target_common_valid_tokens(target)
	edge_margin = _mapping(
		_mapping(head_metadata, 'model metadata')['stratigraphic_hmm'],
		'hmm metadata',
	)['edge_margin_tokens']
	if not np.array_equal(
		np.logical_and(source_valid, edge_margin_mask_for_shape(source_valid.shape, edge_margin)),
		common_target,
	):
		raise ValueError('source/HMM valid-mask parity drift')
	if scientific['source_valid_token_hashes'] != _target_common_hashes(target):
		raise ValueError('fixed HMM common valid-mask identity drift')


def _validate_canonical_initial_artifacts(
	artifacts: Mapping[str, object],
	*,
	target: Mapping[str, object],
	target_manifest_path: Path,
	artifact_root: Path,
) -> None:
	"""Bind periodic initial artifacts to the completed historical export."""
	export_path = target_manifest_path.parent / (
		'multi_head_pseudo_target_export_handoff.json'
	)
	export = _mapping(_json(export_path), 'initial pseudo-target export handoff')
	if (
		export.get('artifact_type')
		!= 'strat_hmm_multi_head_pseudo_target_export_handoff'
		or export.get('schema_version') != 2
		or export.get('completion_status') != 'COMPLETE'
		or Path(str(export.get('pseudo_target_root'))).resolve()
			!= target_manifest_path.parent
	):
		raise ValueError('initial pseudo-target export handoff identity drift')
	clustering = _mapping(export.get('clustering'), 'initial clustering export')
	config_path_value = clustering.get('config_path')
	config_digest = clustering.get('config_sha256')
	clustering_root_value = clustering.get('path')
	if (
		not isinstance(config_path_value, str)
		or not isinstance(config_digest, str)
		or not isinstance(clustering_root_value, str)
	):
		raise TypeError('initial clustering export identity is incomplete')
	config_path = (Path(__file__).resolve().parents[3] / config_path_value).resolve()
	if not config_path.is_file() or file_sha256(config_path) != config_digest:
		raise ValueError('initial clustering export config is stale')
	clustering_root = Path(clustering_root_value).resolve()
	if not clustering_root.is_dir():
		raise FileNotFoundError(
		f'initial clustering export root is missing: {clustering_root}'
	)
	ensure_under_root(
		clustering_root,
		root=artifact_root,
		label='initial clustering export root',
	)
	common = _mapping(artifacts['common'], 'initial HMM common artifacts')
	expected_common = {
		'clustering_config': _reference(config_path),
		'preprocessor': _reference(
			clustering_root / 'models' / 'k6' / 'preprocessor.joblib'
		),
		'residualizer': _reference(
			clustering_root / 'models' / 'residualizer.npz'
		),
		'source_embedding_metadata': _source_embedding_evidence(target)[
			'f3_facies_benchmark'
		]['metadata'],
	}
	for key, expected in expected_common.items():
		if common.get(key) != expected:
			raise ValueError(f'initial HMM common artifact is not historical: {key}')
	heads = _mapping(artifacts['heads'], 'initial HMM heads')
	export_metadata = _mapping(
		clustering.get('metadata_sha256'), 'initial clustering metadata hashes'
	)
	for k in _HEAD_KS:
		model_root = clustering_root / 'models' / f'k{k}'
		expected_head = {
			'model_metadata': _reference(model_root / 'clustering_metadata.json'),
			'hmm_model': _reference(model_root / 'hmm_model.joblib'),
			'centers': _reference(model_root / 'cluster_centers.npy'),
		}
		if _mapping(heads[str(k)], f'initial HMM head {k}') != expected_head:
			raise ValueError(f'initial HMM head {k} is not historical')
		if export_metadata.get(str(k)) != expected_head['model_metadata']['sha256']:
			raise ValueError(f'initial clustering metadata hash drift for K={k}')


def _smoke_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	inputs: Mapping[str, object],
	smoke: Mapping[str, object],
	quarantine_invalid: bool,
	dry_run: bool,
) -> dict[str, object]:
	full = _training_config(config.periodic_refresh_full_config)
	smoke_root = _training_output_root(smoke, config, 'periodic smoke output root')
	full_root = _training_output_root(full, config, 'periodic full output root')
	try:
		if full_root.exists():
			raise ValueError('periodic full output root must remain untouched during smoke')
		if not smoke_root.is_dir():
			raise FileNotFoundError(f'periodic smoke output root is missing: {smoke_root}')
		for name in ('selected.pt', 'best.pt'):
			if (smoke_root / name).exists():
				raise ValueError(f'periodic smoke output must not contain {name}')
		latest_path = smoke_root / 'latest.pt'
		if not latest_path.is_file():
			raise FileNotFoundError(f'periodic smoke latest checkpoint is missing: {latest_path}')
		runtime = deepcopy(smoke)
		_mapping(runtime['train'], 'runtime smoke train')['max_steps'] = 2
		_mapping(runtime['train'], 'runtime smoke train')['device'] = 'cpu'
		payload = load_checkpoint(latest_path, map_location='cpu')
		validate_stratigraphy_checkpoint_payload(payload, expected_config=runtime)
		if payload.get('stratigraphy_config') != runtime:
			raise ValueError('periodic smoke checkpoint config does not bind CLI overrides')
		_validate_finite_payload(payload, 'periodic smoke checkpoint')
		identity = _mapping(payload['stratigraphy_checkpoint'], 'smoke checkpoint identity')
		if identity.get('schema_version') != 8 or identity.get('model_tag') != _MODEL_TAG:
			raise ValueError('periodic smoke checkpoint must be schema 8')
		if payload.get('epoch') != 1 or payload.get('global_step') != 2:
			raise ValueError('periodic smoke checkpoint must end at epoch 1/global step 2')
		training_state = _mapping(payload['training_state'], 'smoke training state')
		if training_state.get('checkpoint_kind') != 'step' or training_state.get(
			'batch_index'
		) != 1:
			raise ValueError('periodic smoke checkpoint must end after two batches')
		state = _mapping(payload['target_refresh_state'], 'smoke refresh state')
		_validate_generation_zero_state(config, runtime, state)
		_baseline_initial_state_identity(identity, inputs['baseline'])
		return {
			'root': str(smoke_root),
			'latest_path': str(latest_path),
			'latest_sha256': file_sha256(latest_path),
			'epoch': payload['epoch'],
			'global_step': payload['global_step'],
			'batch_index': training_state['batch_index'],
			'schema_version': identity['schema_version'],
			'active_generation_id': state['active_generation_id'],
			'refresh_phase': state['refresh_phase'],
			'initial_student_state_sha256': identity['initial_student_state_sha256'],
			'initial_head_state_sha256': identity['initial_head_state_sha256'],
			'initial_spatial_context_state_sha256': identity[
				'initial_spatial_context_state_sha256'
			],
		}
	except _VALIDATION_ERRORS:
		if (
			quarantine_invalid
			and not dry_run
			and smoke_root.exists()
			and smoke_root != full_root
		):
			_quarantine_tree(smoke_root)
		raise


def _validate_generation_zero_state(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	runtime: Mapping[str, object],
	state: Mapping[str, object],
) -> None:
	if (
		state.get('schema_version') != 1
		or state.get('active_generation_index') != 0
		or state.get('active_generation_id') != INITIAL_GENERATION_ID
		or state.get('last_completed_refresh_epoch') != 0
		or state.get('next_scheduled_refresh_epoch') != 2
		or state.get('refresh_phase') != 'training'
		or state.get('source_student_state_sha256') is not None
	):
		raise ValueError('periodic smoke must bind training-phase generation zero')
	if state.get('fixed_preprocessing_hmm_identity_sha256') != (
		_periodic_fixed_preprocessing_identity_sha256(_scientific(runtime))
	):
		raise ValueError('periodic smoke fixed preprocessing identity drift')
	generations = state.get('generations')
	if not isinstance(generations, list) or len(generations) != 1:
		raise ValueError('periodic smoke must contain exactly generation zero')
	record = _mapping(generations[0], 'smoke generation record')
	if record.get('generation_id') != INITIAL_GENERATION_ID:
		raise ValueError('periodic smoke generation zero ID mismatch')
	manifest = Path(str(record['manifest_path'])).resolve()
	refresh_root = Path(
		str(_mapping(runtime['pseudo_target_refresh'], 'runtime refresh')['generation_root'])
	).resolve()
	if manifest != refresh_root / 'generations' / INITIAL_GENERATION_ID / 'refresh_generation.json':
		raise ValueError('periodic smoke generation zero path drift')
	load_periodic_refresh_generation(manifest)
	generations_root = manifest.parent.parent
	unexpected = [
		path.name
		for path in generations_root.iterdir()
		if path.name != INITIAL_GENERATION_ID and not path.name.startswith('.')
	]
	if unexpected:
		raise ValueError(f'periodic smoke contains refreshed generations: {unexpected!r}')
	if Path(str(state['active_target_manifest_path'])).resolve() != config.target_manifest:
		raise ValueError('periodic smoke active target is not initial hard target')
	if state.get('active_target_manifest_sha256') != file_sha256(config.target_manifest):
		raise ValueError('periodic smoke active target hash drift')


def _baseline_initial_state_identity(
	identity: Mapping[str, object], baseline: Mapping[str, object]
) -> None:
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
	):
		if identity.get(key) != baseline[key]:
			raise ValueError(f'periodic checkpoint initial-state identity drift: {key}')


def _checkpoint_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	inputs: Mapping[str, object],
	quarantine_invalid: bool,
	dry_run: bool,
) -> dict[str, object]:
	full = _training_config(config.periodic_refresh_full_config)
	root = _training_output_root(full, config, 'periodic full output root')
	try:
		latest_path = root / 'latest.pt'
		selected_path = root / 'selected.pt'
		if not latest_path.is_file() or not selected_path.is_file():
			raise FileNotFoundError('periodic full latest.pt and selected.pt are required')
		if (root / 'best.pt').exists():
			raise ValueError('periodic full checkpoint root must not contain best.pt')
		latest = _validated_checkpoint(latest_path, full)
		selected = _validated_checkpoint(selected_path, full)
		_validate_finite_payload(latest, 'periodic latest checkpoint')
		_validate_finite_payload(selected, 'periodic selected checkpoint')
		if file_sha256(latest_path) != file_sha256(selected_path):
			raise ValueError('periodic latest and selected checkpoints differ')
		identity = _mapping(selected['stratigraphy_checkpoint'], 'selected identity')
		state = _mapping(selected['target_refresh_state'], 'selected refresh state')
		if (
			selected.get('epoch') != 25
			or selected.get('global_step') != 25600
			or _mapping(selected['training_state'], 'selected training state').get(
				'checkpoint_kind'
			)
			!= 'epoch'
		):
			raise ValueError('periodic selected checkpoint is not final epoch 25')
		if identity.get('schema_version') != 8 or identity.get('model_tag') != _MODEL_TAG:
			raise ValueError('periodic full checkpoint schema or model identity mismatch')
		if state.get('active_generation_index') != 7 or state.get(
			'active_generation_id'
		) != _GENERATION_IDS[-1]:
			raise ValueError('periodic full checkpoint does not bind generation seven')
		if state.get('last_completed_refresh_epoch') != 20 or state.get(
			'next_scheduled_refresh_epoch'
		) is not None or state.get('refresh_phase') != 'training':
			raise ValueError('periodic full final refresh state is not complete')
		if selected.get('checkpoint_selection', {}).get('policy') != (
			'final_completed_epoch_v1'
		):
			raise ValueError('periodic full selection policy mismatch')
		selection = _mapping(selected['checkpoint_selection'], 'periodic selection')
		selected_event = selection.get('selected')
		if not isinstance(selected_event, Mapping) or selected_event.get('epoch') != 25:
			raise ValueError('periodic full selected event is not epoch 25')
		_baseline_initial_state_identity(identity, inputs['baseline'])
		refresh = _refresh_chain_evidence(inputs, full, state)
		_validate_optimizer_continuity(
			selected,
			global_step=int(selected['global_step']),
			optimizer_group_identity=identity['optimizer_group_identity'],
		)
		_events_evidence(
			root,
			state,
			refresh,
			final_student_state_sha256=_state_dict_sha256(
				selected['model_state_dict']
			),
			final_optimizer_state_sha256=_optimizer_state_sha256(
				selected['optimizer_state_dict']
			),
		)
		return {
			'checkpoint': {
				'root': str(root),
				'path': str(selected_path),
				'sha256': file_sha256(selected_path),
				'latest_path': str(latest_path),
				'latest_sha256': file_sha256(latest_path),
				'selected_sha256': file_sha256(selected_path),
				'epoch': selected['epoch'],
				'global_step': selected['global_step'],
				'schema_version': identity['schema_version'],
				'scientific_identity_sha256': identity[
					'scientific_identity_sha256'
				],
				'target_refresh_state_sha256': identity[
					'target_refresh_state_sha256'
				],
				'optimizer_group_identity': identity['optimizer_group_identity'],
				'optimizer_step': selected['global_step'],
				'initial_student_state_sha256': identity[
					'initial_student_state_sha256'
				],
				'initial_head_state_sha256': identity['initial_head_state_sha256'],
				'initial_spatial_context_state_sha256': identity[
					'initial_spatial_context_state_sha256'
				],
			},
			'refresh': refresh,
		}
	except _VALIDATION_ERRORS:
		if quarantine_invalid and not dry_run and root.exists():
			_quarantine_tree(root)
		raise


def _validated_checkpoint(path: Path, config: Mapping[str, object]) -> Mapping[str, object]:
	payload = load_checkpoint(path, map_location='cpu')
	validate_stratigraphy_checkpoint_payload(payload, expected_config=config)
	if payload.get('stratigraphy_config') != config:
		raise ValueError(f'checkpoint config differs from resolved full config: {path}')
	return payload


def _validate_optimizer_continuity(
	payload: Mapping[str, object],
	*,
	global_step: int,
	optimizer_group_identity: object,
) -> None:
	if not _positive_int(global_step):
		raise ValueError('periodic optimizer continuity requires a positive global step')
	identity = optimizer_group_identity
	if not isinstance(identity, list) or not identity:
		raise ValueError('periodic optimizer group identity is empty')
	optimizer = _mapping(
		payload.get('optimizer_state_dict'), 'periodic optimizer state'
	)
	groups = optimizer.get('param_groups')
	state = _mapping(optimizer.get('state'), 'periodic optimizer state entries')
	if not isinstance(groups, list) or len(groups) != len(identity):
		raise ValueError('periodic optimizer group count drift')
	expected_parameter_ids: list[int] = []
	for index, (group_value, identity_value) in enumerate(
		zip(groups, identity, strict=True)
	):
		group = _mapping(group_value, f'periodic optimizer group {index}')
		group_identity = _mapping(
			identity_value, f'periodic optimizer group identity {index}'
		)
		parameter_names = group_identity.get('parameter_names')
		parameter_ids = group.get('params')
		if (
			not isinstance(parameter_names, list)
			or not isinstance(parameter_ids, list)
			or len(parameter_names) != len(parameter_ids)
			or group.get('name') != group_identity.get('name')
			or group.get('lr') != group_identity.get('lr')
		):
			raise ValueError('periodic optimizer group identity drift')
		if any(
			not isinstance(parameter_id, int) or isinstance(parameter_id, bool)
			for parameter_id in parameter_ids
		):
			raise ValueError('periodic optimizer parameter IDs are invalid')
		expected_parameter_ids.extend(parameter_ids)
	if set(state) != set(expected_parameter_ids):
		raise ValueError('periodic optimizer state is not continuous across parameters')
	for parameter_id, value in state.items():
		entry = _mapping(value, f'periodic optimizer state {parameter_id}')
		_validate_finite_payload(entry, f'periodic optimizer state {parameter_id}')
		step = entry.get('step')
		if isinstance(step, torch.Tensor):
			if step.numel() != 1 or (not torch.is_floating_point(step) and step.dtype not in {
				torch.int8,
				torch.int16,
				torch.int32,
				torch.int64,
			}):
				raise ValueError('periodic optimizer step tensor is invalid')
			step = step.item()
		if isinstance(step, bool) or not isinstance(step, int | float) or step != global_step:
			raise ValueError('periodic optimizer step reset or drift detected')


def _optimizer_state_sha256(value: object) -> str:
	digest = hashlib.sha256()
	_update_optimizer_hash(digest, value)
	return digest.hexdigest()


def _update_optimizer_hash(digest: hashlib._Hash, value: object) -> None:
	if isinstance(value, torch.Tensor):
		tensor = value.detach().cpu().contiguous()
		digest.update(b'tensor')
		digest.update(str(tensor.dtype).encode('utf-8'))
		digest.update(str(tuple(tensor.shape)).encode('utf-8'))
		digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
		return
	if isinstance(value, Mapping):
		digest.update(b'mapping')
		for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
			_update_optimizer_hash(digest, key)
			_update_optimizer_hash(digest, value[key])
		return
	if isinstance(value, list | tuple):
		digest.update(b'list' if isinstance(value, list) else b'tuple')
		for child in value:
			_update_optimizer_hash(digest, child)
		return
	digest.update(type(value).__name__.encode('utf-8'))
	digest.update(repr(value).encode('utf-8'))


def _state_dict_sha256(value: object) -> str:
	state_dict = _mapping(value, 'student state dictionary')
	digest = hashlib.sha256()
	for name in sorted(state_dict):
		tensor = state_dict[name]
		if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
			raise TypeError('student state dictionary contains an invalid entry')
		tensor = tensor.detach().cpu().contiguous()
		digest.update(name.encode('utf-8'))
		digest.update(str(tensor.dtype).encode('utf-8'))
		digest.update(str(tuple(tensor.shape)).encode('utf-8'))
		digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _validate_refresh_diagnostics(
	manifest_path: Path,
	payload: Mapping[str, object],
) -> None:
	if payload.get('generation_index') == 0:
		if payload.get('refresh_diagnostics') is not None:
			raise ValueError('periodic initial generation must not have refresh diagnostics')
		return
	ref = _mapping(payload.get('refresh_diagnostics'), 'refresh diagnostics reference')
	_validate_reference(ref, 'refresh diagnostics reference')
	diagnostics_path = Path(str(ref['path'])).resolve()
	if not diagnostics_path.is_relative_to(manifest_path.parent):
		raise ValueError('refresh diagnostics escapes its generation directory')
	diagnostics = _mapping(_json(diagnostics_path), 'refresh diagnostics')
	if set(diagnostics) != {'artifact_type', 'schema_version', 'per_k'}:
		raise ValueError('refresh diagnostics fields are not closed')
	if (
		diagnostics.get('artifact_type') != 'strat_hmm_periodic_refresh_diagnostics'
		or diagnostics.get('schema_version') != 1
	):
		raise ValueError('refresh diagnostics identity is invalid')
	per_k = _mapping(diagnostics['per_k'], 'refresh diagnostics per_k')
	if set(per_k) != {str(k) for k in _HEAD_KS}:
		raise ValueError('refresh diagnostics must contain exactly K=6/8/10')
	_validate_finite_payload(diagnostics, 'refresh diagnostics')
	for k in _HEAD_KS:
		entry = _mapping(per_k[str(k)], f'refresh diagnostics K={k}')
		expected_entry_keys = {
			'iterations',
			'final_label_change_count',
			'final_label_change_rate',
			'final_state_counts',
			'boundary_counts',
			'transition_counts',
			'confidence_summary',
			'state_mean_z',
			'valid_token_count',
			'ordered_diagnostics',
			'boundary_summary',
		}
		if set(entry) != expected_entry_keys:
			raise ValueError(f'refresh diagnostics K={k} fields are not closed')
		valid_count = entry['valid_token_count']
		if not _positive_int(valid_count):
			raise ValueError(f'refresh diagnostics K={k} valid token count is invalid')
		iterations = entry['iterations']
		if not isinstance(iterations, list) or len(iterations) != 2:
			raise ValueError(f'refresh diagnostics K={k} iteration count is invalid')
		for iteration_index, iteration_value in enumerate(iterations):
			iteration = _mapping(
				iteration_value,
				f'refresh diagnostics K={k} iteration {iteration_index}',
			)
			if set(iteration) != {
				'iteration',
				'state_counts',
				'empty_states',
				'center_shift_l2_by_state',
				'total_center_shift_l2',
			} or iteration['iteration'] != iteration_index + 1:
				raise ValueError(f'refresh diagnostics K={k} iteration identity drift')
			if iteration['empty_states'] != []:
				raise ValueError(f'refresh diagnostics K={k} contains an empty state')
			_state_counts(
				iteration['state_counts'],
				k=k,
				valid_count=valid_count,
				label=f'refresh diagnostics K={k} iteration state counts',
			)
			shifts = iteration['center_shift_l2_by_state']
			if (
				not isinstance(shifts, list)
				or len(shifts) != k
				or any(
					not isinstance(value, int | float)
					or isinstance(value, bool)
					or not math.isfinite(float(value))
					or value < 0
					for value in shifts
				)
			):
				raise ValueError(f'refresh diagnostics K={k} center shifts are invalid')
			total_shift = iteration['total_center_shift_l2']
			if (
				not isinstance(total_shift, int | float)
				or isinstance(total_shift, bool)
				or not math.isfinite(float(total_shift))
				or total_shift < 0
			):
				raise ValueError(f'refresh diagnostics K={k} total center shift is invalid')
		_state_counts(
			entry['final_state_counts'],
			k=k,
			valid_count=valid_count,
			label=f'refresh diagnostics K={k} final state counts',
		)
		if (
			not isinstance(entry['final_label_change_count'], int)
			or isinstance(entry['final_label_change_count'], bool)
			or entry['final_label_change_count'] < 0
		):
			raise ValueError(f'refresh diagnostics K={k} label-change count is invalid')
		label_rate = entry['final_label_change_rate']
		if (
			not isinstance(label_rate, int | float)
			or isinstance(label_rate, bool)
			or not math.isfinite(float(label_rate))
			or not 0 <= label_rate <= 1
		):
			raise ValueError(f'refresh diagnostics K={k} label-change rate is invalid')
		_validate_finite_payload(
			entry['confidence_summary'], f'refresh diagnostics K={k} confidence'
		)
		confidence_summary = _mapping(
			entry['confidence_summary'], f'refresh diagnostics K={k} confidence'
		)
		if set(confidence_summary) != {'min', 'p05', 'median', 'p95', 'max', 'mean'}:
			raise ValueError(f'refresh diagnostics K={k} confidence fields are incomplete')
		if any(
			not isinstance(value, int | float)
			or isinstance(value, bool)
			or not math.isfinite(float(value))
			for value in confidence_summary.values()
		):
			raise ValueError(f'refresh diagnostics K={k} confidence is non-finite')
		_validate_count_mapping(
			entry['boundary_counts'],
			label=f'refresh diagnostics K={k} boundary counts',
		)
		_validate_count_mapping(
			entry['transition_counts'],
			label=f'refresh diagnostics K={k} transition counts',
		)
		state_mean_z = _mapping(
			entry['state_mean_z'], f'refresh diagnostics K={k} state means'
		)
		if set(state_mean_z) != {str(state) for state in range(k)} or any(
			not isinstance(value, int | float)
			or isinstance(value, bool)
			or not math.isfinite(float(value))
			for value in state_mean_z.values()
		):
			raise ValueError(f'refresh diagnostics K={k} state means are invalid')
		_validate_finite_payload(
			entry['state_mean_z'], f'refresh diagnostics K={k} state means'
		)
		_validate_finite_payload(
			entry['ordered_diagnostics'], f'refresh diagnostics K={k} ordered diagnostics'
		)
		_validate_finite_payload(
			entry['boundary_summary'], f'refresh diagnostics K={k} boundary summary'
		)


def _state_counts(
	value: object,
	*,
	k: int,
	valid_count: int,
	label: str,
) -> None:
	counts = _mapping(value, label)
	if set(counts) != {str(state) for state in range(k)}:
		raise ValueError(f'{label} must contain every state')
	if any(
		not isinstance(count, int) or isinstance(count, bool) or count <= 0
		for count in counts.values()
	):
		raise ValueError(f'{label} contains an empty or invalid state')
	if sum(counts.values()) != valid_count:
		raise ValueError(f'{label} does not cover the valid tokens')


def _validate_count_mapping(value: object, *, label: str) -> None:
	counts = _mapping(value, label)
	if any(
		not isinstance(count, int) or isinstance(count, bool) or count < 0
		for count in counts.values()
	):
		raise ValueError(f'{label} contains an invalid count')


def _refresh_chain_evidence(
	inputs: Mapping[str, object],
	full: Mapping[str, object],
	state: Mapping[str, object],
) -> dict[str, object]:
	refresh_root = Path(
		str(_mapping(full['pseudo_target_refresh'], 'full refresh')['generation_root'])
	).resolve()
	generations = state.get('generations')
	if not isinstance(generations, list) or len(generations) != 8:
		raise ValueError('periodic full must contain exactly eight generations')
	if [
		_mapping(item, 'generation record')['generation_id'] for item in generations
	] != _GENERATION_IDS:
		raise ValueError('periodic generation ID chain mismatch')
	expected_epochs = [0, *_SCHEDULE]
	loaded: list[Mapping[str, object]] = []
	for index, item in enumerate(generations):
		record = _mapping(item, f'generation record {index}')
		if record.get('generation_index') != index:
			raise ValueError('periodic generation indices are not contiguous')
		manifest_path = Path(str(record['manifest_path'])).resolve()
		if not manifest_path.is_relative_to(refresh_root):
			raise ValueError('periodic generation escapes generation root')
		if file_sha256(manifest_path) != record.get('manifest_sha256'):
			raise ValueError('periodic generation manifest hash drift')
		payload = load_periodic_refresh_generation(manifest_path)
		if (
			payload.get('status') != 'COMPLETE'
			or payload.get('generation_id') != _GENERATION_IDS[index]
			or payload.get('generation_index') != index
			or payload.get('refresh_after_epoch') != expected_epochs[index]
		):
			raise ValueError('periodic generation schedule or status mismatch')
		if payload.get('generation_content_sha256') != record.get(
			'generation_content_sha256'
		):
			raise ValueError('periodic generation content hash drift')
		common = _mapping(payload['valid_token_hashes'], 'generation valid hashes')
		if common != inputs['target_manifest']['common_valid_token_hashes']:
			raise ValueError('periodic generation valid-mask parity drift')
		_validate_refresh_diagnostics(manifest_path, payload)
		loaded.append(payload)
	generations_root = refresh_root / 'generations'
	if {path.name for path in generations_root.iterdir() if path.is_dir()} != set(
		_GENERATION_IDS
	):
		raise ValueError('periodic generation root contains an unexpected generation')
	chain_path = Path(str(state['periodic_refresh_chain_path'])).resolve()
	chain = _mapping(_json(chain_path), 'periodic refresh chain')
	if set(chain) != {
		'schema_version',
		'semantics',
		'refresh_after_epochs',
		'fixed_preprocessing_hmm_identity_sha256',
		'generations',
	}:
		raise ValueError('periodic refresh chain fields are not closed')
	expected_chain_generations = []
	for index, payload in enumerate(loaded):
		previous = payload.get('previous_generation_manifest')
		expected_chain_generations.append(
			{
				'generation_index': payload['generation_index'],
				'generation_id': payload['generation_id'],
				'refresh_after_epoch': payload['refresh_after_epoch'],
				'previous_generation_manifest_sha256': (
					None
					if previous is None
					else _reference_hash(previous)
				),
				'source_student_state_sha256': payload[
					'source_student_state_sha256'
				],
				'manifest_path': str(
					Path(str(generations[index]['manifest_path'])).resolve()
				),
				'manifest_sha256': generations[index]['manifest_sha256'],
				'generation_content_sha256': payload[
					'generation_content_sha256'
				],
			}
		)
	if (
		chain['schema_version'] != 1
		or chain['semantics'] != 'periodic_student_hmm_refresh_chain_v1'
		or chain['refresh_after_epochs'] != _SCHEDULE
		or chain['generations'] != expected_chain_generations
	):
		raise ValueError('periodic refresh chain identity drift')
	if chain['fixed_preprocessing_hmm_identity_sha256'] != (
		_periodic_fixed_preprocessing_identity_sha256(_scientific(full))
	):
		raise ValueError('periodic refresh chain preprocessing identity drift')
	pointer_path = refresh_root / 'active_target_generation.json'
	pointer = _mapping(_json(pointer_path), 'active target generation pointer')
	if set(pointer) != {'manifest_path', 'manifest_sha256'} or pointer != {
		'manifest_path': state['active_generation_manifest_path'],
		'manifest_sha256': state['active_generation_manifest_sha256'],
	}:
		raise ValueError('periodic active target pointer drift')
	return {
		'root': str(refresh_root),
		'chain_path': str(chain_path),
		'chain_sha256': file_sha256(chain_path),
		'generations': [
			{
				'generation_index': payload['generation_index'],
				'generation_id': payload['generation_id'],
				'manifest_path': str(
					Path(str(generations[index]['manifest_path'])).resolve()
				),
				'manifest_sha256': generations[index]['manifest_sha256'],
				'generation_content_sha256': payload['generation_content_sha256'],
				'source_student_state_sha256': payload[
					'source_student_state_sha256'
				],
				'active_target_manifest_path': str(
					Path(
						str(
							_mapping(
								payload['canonical_multi_head_target_manifest'],
								'generation target reference',
							)['path']
						)
					).resolve()
				),
				'active_target_manifest_sha256': _mapping(
					payload['canonical_multi_head_target_manifest'],
					'generation target reference',
				)['sha256'],
			}
			for index, payload in enumerate(loaded)
		],
		'final_generation_id': loaded[-1]['generation_id'],
		'final_target_manifest': loaded[-1]['canonical_multi_head_target_manifest'],
	}


def _events_evidence(
	root: Path,
	state: Mapping[str, object],
	refresh: Mapping[str, object],
	*,
	final_student_state_sha256: str,
	final_optimizer_state_sha256: str,
) -> None:
	path = root / 'target_refresh_events.jsonl'
	if not path.is_file():
		raise FileNotFoundError(f'periodic refresh events are missing: {path}')
	events = [
		_mapping(json.loads(line), 'periodic refresh event')
		for line in path.read_text(encoding='utf-8').splitlines()
		if line.strip()
	]
	events = _deduplicate_refresh_lifecycle_events(events)
	if not events:
		raise ValueError('periodic refresh events are empty')
	generations = {
		_mapping(value, 'refresh generation evidence')['generation_id']: _mapping(
			value, 'refresh generation evidence'
		)
		for value in _mapping(refresh, 'refresh evidence')['generations']
	}
	complete_refreshes: dict[int, Mapping[str, object]] = {}
	started_refreshes: dict[int, Mapping[str, object]] = {}
	checkpoint_steps: list[int] = []
	last_checkpoint_student_hash: str | None = None
	last_checkpoint_optimizer_hash: str | None = None
	final_epoch_checkpoint = False
	for event in events:
		if event.get('status') == 'start' and event.get('event_type') == 'refresh':
			epoch = event.get('refresh_epoch')
			generation_id = event.get('generation_id')
			generation = generations.get(generation_id)
			if (
				not isinstance(epoch, int)
				or epoch not in _SCHEDULE
				or epoch in started_refreshes
				or generation is None
				or generation['generation_index'] != _SCHEDULE.index(epoch) + 1
				or event.get('generation_index') != generation['generation_index']
				or event.get('source_student_state_sha256')
					!= last_checkpoint_student_hash
				or event.get('student_state_sha256') != last_checkpoint_student_hash
				or event.get('optimizer_state_sha256')
					!= last_checkpoint_optimizer_hash
			):
				raise ValueError('periodic refresh start is not checkpoint-continuous')
			_require_sha256(
				event.get('source_student_state_sha256'),
				'periodic refresh source student hash',
			)
			_require_sha256(
				event.get('student_state_sha256'),
				'periodic refresh student hash',
			)
			_require_sha256(
				event.get('optimizer_state_sha256'),
				'periodic refresh optimizer hash',
			)
			started_refreshes[epoch] = event
			continue
		if event.get('status') != 'complete':
			continue
		event_type = event.get('event_type')
		before = event.get('global_step_before')
		after = event.get('global_step_after')
		if (
			not isinstance(before, int)
			or isinstance(before, bool)
			or not isinstance(after, int)
			or isinstance(after, bool)
			or before < 0
			or after < before
		):
			raise ValueError('periodic event global-step evidence is invalid')
		if event_type in {'refresh', 'generation'} and before != after:
			raise ValueError('periodic refresh global step changed during refresh')
		if event_type == 'generation':
			if (
				event.get('generation_id') != _GENERATION_IDS[0]
				or event.get('generation_index') != 0
				or event.get('phase') != 'initial_bind'
				or before != 0
				or event.get('active_target_manifest_sha256')
					!= _mapping(generations[_GENERATION_IDS[0]], 'generation zero')[
						'active_target_manifest_sha256'
					]
			):
				raise ValueError('periodic initial generation event identity drift')
			continue
		if event_type == 'refresh' and event.get('phase') is None:
			epoch = event.get('refresh_epoch')
			generation_id = event.get('generation_id')
			generation = generations.get(generation_id)
			if (
				not isinstance(epoch, int)
				or epoch not in _SCHEDULE
				or epoch in complete_refreshes
				or generation is None
				or event.get('generation_index') != generation['generation_index']
				or event.get('source_student_state_sha256')
					!= generation['source_student_state_sha256']
				or event.get('output_generation_manifest_path')
					!= generation['manifest_path']
				or event.get('output_generation_manifest_sha256')
					!= generation['manifest_sha256']
				or event.get('active_target_manifest_path')
					!= generation['active_target_manifest_path']
				or event.get('active_target_manifest_sha256')
					!= generation['active_target_manifest_sha256']
				or event.get('student_state_sha256')
					!= started_refreshes.get(epoch, {}).get('student_state_sha256')
				or event.get('optimizer_state_sha256')
					!= started_refreshes.get(epoch, {}).get('optimizer_state_sha256')
			):
				raise ValueError('periodic refresh event is not bound to its generation')
			if epoch not in started_refreshes:
				raise ValueError('periodic refresh completion has no start event')
			complete_refreshes[epoch] = event
			continue
		if event_type == 'checkpoint':
			kind = event.get('checkpoint_kind')
			generation = generations.get(event.get('active_generation_id'))
			if kind not in {'step', 'epoch', 'refresh'} or generation is None:
				raise ValueError('periodic checkpoint event identity is invalid')
			if after - before != (1 if kind == 'step' else 0):
				raise ValueError('periodic checkpoint global-step transition is invalid')
			if (
				event.get('active_generation_manifest_sha256')
					!= generation['manifest_sha256']
				or event.get('active_generation_content_sha256')
					!= generation['generation_content_sha256']
				or event.get('active_target_manifest_sha256')
					!= generation['active_target_manifest_sha256']
				or event.get('source_student_state_sha256')
					!= generation['source_student_state_sha256']
			):
				raise ValueError('periodic checkpoint source-generation continuity drift')
			_require_sha256(
				event.get('student_state_sha256'),
				'periodic checkpoint student hash',
			)
			_require_sha256(
				event.get('optimizer_state_sha256'),
				'periodic checkpoint optimizer hash',
			)
			last_checkpoint_student_hash = event['student_state_sha256']
			last_checkpoint_optimizer_hash = event['optimizer_state_sha256']
			checkpoint_steps.append(after)
			if event.get('epoch') == 25 and kind == 'epoch' and after == 25600:
				if (
					event['student_state_sha256'] != final_student_state_sha256
					or event['optimizer_state_sha256']
						!= final_optimizer_state_sha256
				):
					raise ValueError('periodic final checkpoint event hash drift')
				final_epoch_checkpoint = True
	if checkpoint_steps != sorted(checkpoint_steps):
		raise ValueError('periodic checkpoint global steps are not monotonic')
	for epoch in _SCHEDULE:
		if epoch not in started_refreshes or epoch not in complete_refreshes:
			raise ValueError(f'periodic refresh event is missing for epoch {epoch}')
	if not final_epoch_checkpoint:
		raise ValueError('periodic refresh events are missing the final epoch checkpoint')
	if state.get('active_generation_id') != _GENERATION_IDS[-1]:
		raise ValueError('periodic event state is not final')


def _deduplicate_refresh_lifecycle_events(
	events: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
	"""Ignore exact lifecycle replays while rejecting conflicting retries."""
	seen: dict[tuple[object, ...], Mapping[str, object]] = {}
	result: list[Mapping[str, object]] = []
	for event in events:
		if event.get('event_type') != 'refresh' or event.get('status') not in {
			'start',
			'complete',
		}:
			result.append(event)
			continue
		key = (
			event.get('event_type'),
			event.get('status'),
			event.get('refresh_epoch'),
			event.get('generation_index'),
			event.get('generation_id'),
		)
		previous = seen.get(key)
		if previous is None:
			seen[key] = event
			result.append(event)
			continue
		if dict(previous) != dict(event):
			raise ValueError('conflicting duplicate periodic refresh lifecycle event')
	return result


def _embedding_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	inputs: Mapping[str, object],
	checkpoint: Mapping[str, object],
) -> dict[str, object]:
	extraction = resolve_embedding_extraction_config(
		load_config(config.periodic_refresh_embedding_config)
	)
	paths = _mapping(extraction['embeddings'], 'periodic extraction')
	root = Path(str(paths['output_dir'])).resolve()
	selected_path = Path(str(checkpoint['path'])).resolve()
	if Path(str(paths['checkpoint'])).resolve() != selected_path:
		raise ValueError('periodic extraction checkpoint does not match selected.pt')
	inputs_found = tuple(discover_embedding_inputs(root))
	if [item.survey_id for item in inputs_found] != ['f3_facies_benchmark']:
		raise ValueError('periodic final embedding survey set mismatch')
	item = inputs_found[0]
	files = output_paths(root, item.survey_id)
	metadata = _mapping(_json(files.metadata), 'periodic final embedding metadata')
	if metadata.get('embedding_semantics') != (
		'current_student_unmasked_eval_full_survey_v1'
	):
		raise ValueError('periodic final embedding must be unmasked student output')
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	if (
		tuple(embeddings.shape) != (76, 113, 32, 384)
		or embeddings.dtype != np.dtype('float16')
		or tuple(valid.shape) != (76, 113, 32)
		or valid.dtype != np.dtype('bool')
		or not int(valid.sum())
		or not np.isfinite(embeddings[valid]).all()
	):
		raise ValueError('periodic final embedding shape/dtype/finite contract mismatch')
	selected_payload = load_checkpoint(selected_path, map_location='cpu')
	state = _mapping(selected_payload['target_refresh_state'], 'selected refresh state')
	active_target = load_multi_head_target_manifest(
		Path(str(state['active_target_manifest_path']))
	)
	common_valid = _target_common_valid_tokens(active_target)
	target_source = _source_embedding_evidence(active_target)
	source_valid_path = Path(str(target_source['f3_facies_benchmark']['valid_tokens']['path']))
	if file_sha256(files.valid_tokens) != file_sha256(source_valid_path):
		raise ValueError('periodic final embedding raw valid-mask identity drift')
	metadata_hmm = _json(
		Path(str(_mapping(inputs['fixed_preprocessing'], 'fixed preprocessing')['initial_hmm_artifacts']['heads']['6']['model_metadata']['path']))
	)
	edge_margin = _mapping(metadata_hmm['stratigraphic_hmm'], 'HMM metadata')[
		'edge_margin_tokens'
	]
	if not np.array_equal(
		np.logical_and(valid, edge_margin_mask_for_shape(valid.shape, edge_margin)),
		common_valid,
	):
		raise ValueError('periodic final embedding common valid-mask parity drift')
	if metadata.get('checkpoint_path') != str(selected_path):
		raise ValueError('periodic final embedding checkpoint path drift')
	if metadata.get('checkpoint_sha256') != file_sha256(selected_path):
		raise ValueError('periodic final embedding checkpoint hash drift')
	pretext = _mapping(metadata.get('stratigraphy_pretext'), 'stratigraphy metadata')
	identity = _mapping(selected_payload['stratigraphy_checkpoint'], 'selected identity')
	for key, expected in {
		'model_tag': _MODEL_TAG,
		'scientific_identity_sha256': identity['scientific_identity_sha256'],
		'target_manifest_sha256': state['active_target_manifest_sha256'],
		'active_generation_id': state['active_generation_id'],
		'active_generation_manifest_sha256': state[
			'active_generation_manifest_sha256'
		],
		'active_generation_content_sha256': state['active_generation_content_sha256'],
		'active_target_manifest_path': state['active_target_manifest_path'],
		'active_target_manifest_sha256': state['active_target_manifest_sha256'],
		'periodic_refresh_chain_path': state['periodic_refresh_chain_path'],
		'periodic_refresh_chain_sha256': state['periodic_refresh_chain_sha256'],
		'fixed_preprocessing_hmm_identity_sha256': state[
			'fixed_preprocessing_hmm_identity_sha256'
		],
		'target_refresh_state_sha256': identity['target_refresh_state_sha256'],
	}.items():
		if pretext.get(key) != expected:
			raise ValueError(f'periodic final embedding metadata identity drift: {key}')
	if pretext.get('refresh_after_epochs') != _SCHEDULE or pretext.get(
		'target_refresh_semantics'
	) != 'periodic_student_hmm_center_refresh_v1':
		raise ValueError('periodic final embedding refresh identity drift')
	return {
		'root': str(root),
		'metadata_path': str(files.metadata),
		'metadata_sha256': file_sha256(files.metadata),
		'embeddings_path': str(files.embeddings),
		'embeddings_sha256': file_sha256(files.embeddings),
		'valid_tokens_path': str(files.valid_tokens),
		'valid_tokens_sha256': file_sha256(files.valid_tokens),
		'embeddings_shape': list(embeddings.shape),
		'embeddings_dtype': str(embeddings.dtype),
		'valid_tokens_shape': list(valid.shape),
		'valid_tokens_dtype': str(valid.dtype),
		'finite_valid_count': int(valid.sum()),
	}


def _validate_smoke_phase_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	inputs: Mapping[str, object],
) -> None:
	payload = _mapping(
		_json(_phase_evidence_path(config, 'smoke')),
		'periodic smoke phase evidence',
	)
	if payload.get('status') != 'PASS' or payload.get('phase') != 'smoke':
		raise ValueError('periodic smoke phase evidence is not PASS')
	evidence = _mapping(payload.get('evidence'), 'periodic smoke phase evidence body')
	smoke = _mapping(evidence.get('smoke'), 'periodic smoke evidence')
	path = Path(str(smoke['latest_path'])).resolve()
	if not path.is_file() or smoke.get('latest_sha256') != file_sha256(path):
		raise ValueError('periodic smoke evidence is stale')
	if evidence.get('target_manifest') != inputs['target_manifest']:
		raise ValueError('periodic smoke evidence target binding is stale')


def _fixed_preprocessing_evidence(training: Mapping[str, object]) -> dict[str, object]:
	scientific = _scientific(training)
	return {
		'initial_hmm_artifacts': scientific['initial_hmm_artifacts'],
		'fixed_preprocessor_sha256': scientific['fixed_preprocessor_sha256'],
		'fixed_residualizer_sha256': scientific['fixed_residualizer_sha256'],
		'fixed_clustering_config_sha256': scientific['fixed_clustering_config_sha256'],
		'source_embedding_metadata_sha256': scientific[
			'source_embedding_metadata_sha256'
		],
		'source_valid_token_hashes': scientific['source_valid_token_hashes'],
		'feature_dimension': scientific['feature_dimension'],
	}


def _handoff(evidence: Mapping[str, object]) -> dict[str, object]:
	checkpoint = _mapping(evidence['checkpoint'], 'checkpoint evidence')
	target_manifest = _mapping(
		evidence['target_manifest'], 'target manifest evidence'
	)
	refresh = _mapping(evidence['refresh'], 'refresh evidence')
	final_generation = _mapping(refresh['generations'][-1], 'final generation evidence')
	return {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': _MODEL_TAG,
		'variant': _VARIANT,
		'primary_checkpoint_role': 'completed_final_selected',
		'targets': {
			'initial_hard_target_manifest': {
				'path': target_manifest['path'],
				'sha256': target_manifest['sha256'],
			},
			'initial_per_head_target_hashes': target_manifest[
				'per_head_target_hashes'
			],
			'final_target_manifest': refresh['final_target_manifest'],
			'final_generation': {
				'generation_index': final_generation['generation_index'],
				'generation_id': final_generation['generation_id'],
				'manifest': {
					'path': final_generation['manifest_path'],
					'sha256': final_generation['manifest_sha256'],
				},
				'content_sha256': final_generation['generation_content_sha256'],
			},
			'periodic_refresh_chain': {
				'path': refresh['chain_path'],
				'sha256': refresh['chain_sha256'],
			},
			'valid_token_hashes': evidence['target_manifest'][
				'common_valid_token_hashes'
			],
		},
		'checkpoint': {
			'path': checkpoint['path'],
			'sha256': checkpoint['sha256'],
			'latest_path': checkpoint['latest_path'],
			'latest_sha256': checkpoint['latest_sha256'],
			'epoch': checkpoint['epoch'],
			'global_step': checkpoint['global_step'],
			'schema_version': checkpoint['schema_version'],
			'scientific_identity_sha256': checkpoint[
				'scientific_identity_sha256'
			],
			'target_refresh_state_sha256': checkpoint[
				'target_refresh_state_sha256'
			],
			'optimizer_group_identity': checkpoint['optimizer_group_identity'],
			'initial_student_state_sha256': checkpoint[
				'initial_student_state_sha256'
			],
			'initial_head_state_sha256': checkpoint['initial_head_state_sha256'],
			'initial_spatial_context_state_sha256': checkpoint[
				'initial_spatial_context_state_sha256'
			],
		},
		'embedding': evidence['embedding'],
		'fixed_preprocessing': evidence['fixed_preprocessing'],
		'execution': evidence['execution'],
	}


def _publish_handoff(
	path: Path,
	handoff: Mapping[str, object],
	*,
	only_missing: bool,
	quarantine_invalid: bool,
) -> None:
	if path.is_file():
		try:
			existing = load_f3_center_trace_masked_periodic_refresh_handoff(path)
		except _VALIDATION_ERRORS:
			existing = None
		if existing == handoff:
			# A valid PASS handoff is immutable.  ``only_missing`` is retained as
			# part of the CLI contract; exact outputs are reusable in either mode.
			if only_missing:
				return
			return
		if not quarantine_invalid:
			raise ValueError(
				'existing periodic refresh handoff is stale or invalid; pass '
				'--quarantine-invalid to replace it'
			)
		_quarantine_file(path)
	_atomic_json(path, handoff)


def _start_execution_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	dry_run: bool,
	quarantine_invalid: bool,
) -> Mapping[str, object]:
	path = _execution_evidence_path(config)
	if path.exists() or path.is_symlink():
		existing_execution: Mapping[str, object] | None = None
		completed_phase: str | None = None
		try:
			existing = _mapping(_json(path), 'periodic execution evidence')
			valid_identity = (
				existing.get('artifact_type') == _EXECUTION_ARTIFACT_TYPE
				and existing.get('schema_version') == 1
			)
			if valid_identity:
				phase = existing.get('phase')
				candidate_execution = _mapping(
					existing.get('execution'), 'periodic execution state'
				)
				_validate_execution_record(
					candidate_execution,
					phase=phase,
					label='periodic execution evidence',
				)
				if phase in {'smoke', 'complete'}:
					completed_phase = phase
				elif (
					phase == 'inputs'
					and existing.get('binding') == _execution_binding(config)
				):
					existing_execution = candidate_execution
		except _VALIDATION_ERRORS:
			existing_execution = None
			completed_phase = None
		if completed_phase is not None:
			raise ValueError(
				f'existing periodic {completed_phase} execution evidence is complete; '
				'refusing to restart it'
			)
		if existing_execution is not None:
			return existing_execution
		if not quarantine_invalid:
			raise ValueError(
				'existing periodic execution evidence is stale or invalid; pass '
				'--quarantine-invalid to start a new execution'
			)
		if not dry_run:
			_quarantine_file(path)
	before = _execution_identity()
	record = {
		'artifact_type': _EXECUTION_ARTIFACT_TYPE,
		'schema_version': 1,
		'phase': 'inputs',
		'binding': _execution_binding(config),
		'execution': {'before': before, 'after': None},
	}
	if not dry_run:
		_atomic_json(path, record)
	return _mapping(record['execution'], 'periodic execution state')


def _validate_execution_record(
	execution: Mapping[str, object], *, phase: object, label: str
) -> None:
	"""Validate an execution marker before deciding whether it is reusable."""
	if phase not in {'inputs', 'smoke', 'complete'}:
		raise ValueError(f'{label} phase is invalid')
	if set(execution) != {'before', 'after'}:
		raise ValueError(f'{label} state fields are invalid')
	_validate_execution_state(execution['before'], f'{label}.before')
	after = execution['after']
	if phase == 'inputs':
		if after is not None:
			raise ValueError(f'{label} inputs state is already complete')
		return
	if after is None:
		raise ValueError(f'{label} {phase} state is incomplete')
	_validate_execution_state(after, f'{label}.after')


def _update_execution_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	phase: str,
	dry_run: bool,
) -> Mapping[str, object]:
	record = _mapping(
		_json(_execution_evidence_path(config)),
		'periodic execution evidence',
	)
	if record.get('artifact_type') != _EXECUTION_ARTIFACT_TYPE or record.get(
		'schema_version'
	) != 1 or record.get('binding') != _execution_binding(config):
		raise ValueError('periodic execution evidence binding mismatch')
	if phase == 'smoke' and record.get('phase') not in {'inputs', 'smoke'}:
		raise ValueError('periodic smoke execution evidence must follow inputs')
	if phase == 'complete' and record.get('phase') not in {'smoke', 'complete'}:
		raise ValueError('periodic complete execution evidence requires smoke evidence')
	updated = deepcopy(record)
	updated['phase'] = phase
	execution = _mapping(updated['execution'], 'periodic execution')
	execution['after'] = _execution_identity()
	if not dry_run and updated == record:
		return execution
	if not dry_run:
		_atomic_json(_execution_evidence_path(config), updated)
	return _mapping(updated['execution'], 'periodic execution')


def _execution_binding(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
) -> dict[str, object]:
	return {
		key: _reference(getattr(config, field))
		for key, field in (
			('target_manifest', 'target_manifest'),
			('hard_full_config', 'hard_full_config'),
			('hard_handoff', 'hard_handoff'),
			('center_trace_masked_full_config', 'center_trace_masked_full_config'),
			('center_trace_masked_handoff', 'center_trace_masked_handoff'),
			('periodic_refresh_smoke_config', 'periodic_refresh_smoke_config'),
			('periodic_refresh_full_config', 'periodic_refresh_full_config'),
			('periodic_refresh_embedding_config', 'periodic_refresh_embedding_config'),
		)
	}


def _execution_evidence_path(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
) -> Path:
	return config.experiment_root / _EXECUTION_EVIDENCE_FILENAME


def _phase_evidence_path(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig, phase: str
) -> Path:
	if phase not in {'inputs', 'smoke'}:
		raise ValueError('phase evidence is only available for inputs and smoke')
	return config.experiment_root / f'{_PHASE_EVIDENCE_PREFIX}_{phase}.json'


def _write_phase_evidence(
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	*,
	phase: str,
	evidence: Mapping[str, object],
	only_missing: bool,
	quarantine_invalid: bool,
) -> Path:
	payload = {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'phase': phase,
		'status': 'PASS',
		'evidence': dict(evidence),
	}
	path = _phase_evidence_path(config, phase)
	if path.is_file():
		try:
			existing = _mapping(_json(path), f'periodic {phase} phase evidence')
		except _VALIDATION_ERRORS:
			existing = None
		if existing == payload:
			# Valid phase reports, like the final handoff, are immutable and can
			# be reused without changing their content or mtime.
			if only_missing:
				return path
			return path
		if not quarantine_invalid:
			raise ValueError(
				f'existing periodic {phase} phase evidence is stale or invalid; '
				'pass --quarantine-invalid to replace it'
			)
		_quarantine_file(path)
	_atomic_json(path, payload)
	return path


def _write_checkpoint_report(
	root: Path,
	*,
	checkpoint_sha256: str,
	only_missing: bool,
	quarantine_invalid: bool,
) -> Path:
	"""Persist the checkpoints PASS report without overwriting stale evidence."""
	payload = {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'phase': 'checkpoints',
		'status': 'PASS',
		'checkpoint_sha256': checkpoint_sha256,
	}
	path = root / 'preflight' / 'periodic_refresh_checkpoint_validation.json'
	if path.exists():
		try:
			existing = _mapping(_json(path), 'periodic checkpoints report')
		except _VALIDATION_ERRORS:
			existing = None
		if existing == payload:
			# Exact PASS reports are immutable and reusable in either mode.
			if only_missing:
				return path
			return path
		if not quarantine_invalid:
			raise ValueError(
				'existing periodic checkpoints report is stale or invalid; pass '
				'--quarantine-invalid to replace it'
			)
		_quarantine_file(path)
	_atomic_json(path, payload)
	return path


def _execution_identity() -> dict[str, object]:
	root = Path(__file__).resolve().parents[3]
	commit = _git_output(root, 'rev-parse', 'HEAD')
	status = _git_output(root, 'status', '--short')
	diff = _git_bytes(root, 'diff', '--binary', 'HEAD')
	if commit is None or status is None or diff is None:
		raise RuntimeError('unable to collect repository execution identity')
	return {
		'git_commit': commit,
		'git_status_short': status.splitlines(),
		'git_diff_sha256': hashlib.sha256(diff).hexdigest(),
	}


def _git_output(root: Path, *args: str) -> str | None:
	try:
		return subprocess.check_output(
			['git', '-C', str(root), *args],
			stderr=subprocess.DEVNULL,
			text=True,
		).strip()
	except (OSError, subprocess.CalledProcessError):
		return None


def _git_bytes(root: Path, *args: str) -> bytes | None:
	try:
		return subprocess.check_output(
			['git', '-C', str(root), *args], stderr=subprocess.DEVNULL
		)
	except (OSError, subprocess.CalledProcessError):
		return None


def _validate_execution_state(value: object, label: str) -> None:
	state = _mapping(value, label)
	if set(state) != {'git_commit', 'git_status_short', 'git_diff_sha256'}:
		raise ValueError(f'{label} keys mismatch')
	if not _valid_git_commit(state['git_commit']):
		raise ValueError(f'{label}.git_commit is invalid')
	if not isinstance(state['git_status_short'], list) or any(
		not isinstance(item, str) for item in state['git_status_short']
	):
		raise TypeError(f'{label}.git_status_short must be a string list')
	_require_sha256(state['git_diff_sha256'], f'{label}.git_diff_sha256')


def _validate_handoff_references(payload: Mapping[str, object], label: str) -> None:
	if label == 'center handoff':
		for_target = _mapping(payload['targets'], f'{label} targets')
		for key in ('target_manifest', 'hard_baseline_config', 'hard_baseline_handoff'):
			_validate_reference(for_target[key], f'{label}.{key}')
	checkpoint = _mapping(payload['checkpoint'], f'{label} checkpoint')
	for key in ('path', 'latest_path'):
		if key in checkpoint:
			path = Path(str(checkpoint[key]))
			if not path.is_file():
				raise FileNotFoundError(f'{label} checkpoint is missing: {path}')


def _validate_fixed_preprocessing_references(
	value: object,
	label: str,
) -> None:
	artifacts = _mapping(value, f'{label} HMM artifacts')
	common = _mapping(artifacts['common'], f'{label} common artifacts')
	for key in ('clustering_config', 'preprocessor', 'source_embedding_metadata'):
		_validate_reference(common[key], f'{label} common {key}')
	if common.get('residualizer') is not None:
		_validate_reference(common['residualizer'], f'{label} common residualizer')
	heads = _mapping(artifacts['heads'], f'{label} head artifacts')
	if set(heads) != {'6', '8', '10'}:
		raise ValueError(f'{label} HMM heads must be exactly K=6/8/10')
	for key in ('6', '8', '10'):
		head = _mapping(heads[key], f'{label} K={key} artifacts')
		for field in ('centers', 'hmm_model', 'model_metadata'):
			_validate_reference(head[field], f'{label} K={key} {field}')


def _validate_target_file_references(target: Mapping[str, object]) -> None:
	for head_k in _HEAD_KS:
		head = _mapping(_mapping(target['heads'], 'target heads')[str(head_k)], f'target K={head_k}')
		for survey_id, value in _mapping(head['surveys'], 'target surveys').items():
			entry = _mapping(value, f'target K={head_k} survey {survey_id}')
			for field in ('labels', 'confidence', 'valid_tokens', 'metadata'):
				_validate_reference(entry[field], f'target K={head_k} {survey_id} {field}')


def _source_embedding_evidence(target: Mapping[str, object]) -> dict[str, object]:
	source = _mapping(target['source_embedding'], 'target source embedding')
	surveys = _mapping(source['surveys'], 'target source surveys')
	result: dict[str, object] = {}
	for survey_id, value in surveys.items():
		entry = _mapping(value, f'target source survey {survey_id}')
		result[survey_id] = {
			'embeddings': _source_reference(entry, 'embedding_path', 'embedding_sha256'),
			'valid_tokens': _source_reference(
				entry, 'valid_tokens_path', 'valid_tokens_sha256'
			),
			'metadata': _source_reference(entry, 'metadata_path', 'metadata_sha256'),
		}
	return result


def _source_reference(
	entry: Mapping[str, object], path_key: str, hash_key: str
) -> dict[str, str]:
	path = entry.get(path_key)
	digest = entry.get(hash_key)
	if not isinstance(path, str) or not isinstance(digest, str):
		raise TypeError(f'source embedding reference {path_key} is invalid')
	ref = {'path': str(Path(path).resolve()), 'sha256': digest}
	_validate_reference(ref, f'source embedding {path_key}')
	return ref


def _target_common_valid_tokens(target: Mapping[str, object]) -> np.ndarray:
	head = _mapping(_mapping(target['heads'], 'target heads')['6'], 'target K=6')
	surveys = _mapping(head['surveys'], 'target K=6 surveys')
	if set(surveys) != {'f3_facies_benchmark'}:
		raise ValueError('periodic target must contain the F3 survey')
	entry = _mapping(surveys['f3_facies_benchmark'], 'target F3 survey')
	ref = _mapping(entry['valid_tokens'], 'target common valid tokens')
	return np.asarray(np.load(Path(str(ref['path'])), mmap_mode='r', allow_pickle=False))


def _target_common_hashes(target: Mapping[str, object]) -> Mapping[str, str]:
	return _mapping(_mapping(target['common'], 'target common')['valid_tokens_sha256'], 'target common valid hashes')


def _training_config(path: Path) -> Mapping[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(path))


def _training_output_root(
	training: Mapping[str, object],
	config: F3CenterTraceMaskedPeriodicRefreshValidationConfig,
	label: str,
) -> Path:
	root = Path(str(_mapping(training['paths'], label)['output_root'])).resolve()
	ensure_under_root(root, root=config.artifact_root, label=label)
	if root == config.artifact_root:
		raise ValueError(f'{label} must not be the artifact root')
	return root


def _manifest_path(training: Mapping[str, object]) -> Path:
	return Path(str(_mapping(training['pseudo_targets'], 'pseudo_targets')['manifest'])).resolve()


def _model_tag(training: Mapping[str, object]) -> str:
	value = _mapping(training['identity'], 'identity').get('model_tag')
	if not isinstance(value, str) or not value:
		raise TypeError('training model tag is invalid')
	return value


def _scientific(training: Mapping[str, object]) -> Mapping[str, object]:
	return _mapping(
		_mapping(training['identity'], 'identity')['scientific_identity'],
		'scientific identity',
	)


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path.resolve()), 'sha256': file_sha256(path)}


def _reference_hash(value: object) -> str:
	return str(_mapping(value, 'artifact reference')['sha256'])


def _validate_reference(value: object, label: str) -> None:
	ref = _mapping(value, label)
	if set(ref) != {'path', 'sha256'}:
		raise ValueError(f'{label} keys mismatch')
	path = ref.get('path')
	if not isinstance(path, str) or not path:
		raise TypeError(f'{label}.path is missing')
	_require_sha256(ref.get('sha256'), f'{label}.sha256')
	resolved = Path(path).resolve()
	if not resolved.is_file() or file_sha256(resolved) != ref['sha256']:
		raise ValueError(f'{label} is missing or hash-drifted: {resolved}')


def _validate_target_hashes(value: object) -> None:
	hashes = _mapping(value, 'target head hashes')
	if set(hashes) != {'6', '8', '10'}:
		raise ValueError('target head hashes must contain K=6/8/10')
	for head, surveys_value in hashes.items():
		for survey, artifacts_value in _mapping(
			surveys_value, f'target hashes K={head}'
		).items():
			artifacts = _mapping(artifacts_value, f'target hashes {head}/{survey}')
			if set(artifacts) != {'labels', 'confidence', 'valid_tokens', 'metadata'}:
				raise ValueError('target artifact hash keys mismatch')
			for name, digest in artifacts.items():
				_require_sha256(digest, f'target hash {head}/{survey}/{name}')


def _validate_valid_hashes(value: object) -> None:
	hashes = _mapping(value, 'valid-token hashes')
	if not hashes:
		raise ValueError('valid-token hashes must not be empty')
	for survey, digest in hashes.items():
		if not isinstance(survey, str) or not survey:
			raise ValueError('valid-token survey ID is invalid')
		_require_sha256(digest, f'valid-token hash {survey}')


def _json(path: Path) -> object:
	try:
		return json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as exc:
		raise ValueError(f'expected valid JSON: {path}') from exc


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _require_sha256(value: object, label: str) -> None:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')


def _require_sha256_or_none(value: object, label: str) -> None:
	if value is not None:
		_require_sha256(value, label)


def _positive_int(value: object) -> bool:
	return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_git_commit(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) in {40, 64}
		and bool(value)
		and all(character in '0123456789abcdef' for character in value.lower())
	)


def _validate_finite_payload(value: object, label: str) -> None:
	if isinstance(value, torch.Tensor):
		if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
			raise ValueError(f'{label} contains non-finite tensor values')
		return
	if isinstance(value, np.ndarray):
		if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
			raise ValueError(f'{label} contains non-finite array values')
		return
	if isinstance(value, Mapping):
		for key, child in value.items():
			_validate_finite_payload(child, f'{label}.{key}')
		return
	if isinstance(value, list | tuple):
		for index, child in enumerate(value):
			_validate_finite_payload(child, f'{label}[{index}]')
		return
	if isinstance(value, float) and not math.isfinite(value):
		raise ValueError(f'{label} contains a non-finite number')


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary_name = tempfile.mkstemp(
		prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
	)
	temporary = Path(temporary_name)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as handle:
			json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
			handle.write('\n')
			handle.flush()
			os.fsync(handle.fileno())
		temporary.replace(path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _quarantine_file(path: Path) -> Path:
	target = path.with_name(
		f'{path.name}.quarantine.{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
	)
	if target.exists():
		raise FileExistsError(f'quarantine path already exists: {target}')
	path.replace(target)
	return target


def _quarantine_tree(path: Path) -> Path:
	return _quarantine_file(path)
