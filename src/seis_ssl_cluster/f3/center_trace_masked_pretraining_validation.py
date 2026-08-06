"""Strict schema-7 validation and handoff publication for experiment 104."""
# ruff: noqa: E501, C901, PLR0912, PLR0913, PLR0915, S603, S607, SLF001

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.config.pretraining import _multi_head_target_hashes
from seis_ssl_cluster.data import read_manifest_json
from seis_ssl_cluster.data.normalization import load_normalization_stats
from seis_ssl_cluster.data.volume_store import inspect_npy_volume
from seis_ssl_cluster.data.window_preprocessing import resolve_manifest_path
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.training.strat_hmm.components import (
	build_strat_hmm_components,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	scientific_identity_sha256,
	validate_stratigraphy_checkpoint_payload,
)

_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'experiment_root',
		'target_manifest',
		'hard_full_config',
		'hard_handoff',
		'center_trace_masked_smoke_config',
		'center_trace_masked_full_config',
		'center_trace_masked_embedding_config',
	}
)
_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
_HARD_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
_VARIANT = 'ctmask010_nocons'
_ROLE = 'multi_head_center_trace_masked_hard_pretext'
_TARGET_REPRESENTATION = 'hard_viterbi_labels_v1'
_HEAD_SPEC = 'multi_resolution_ordered_prototypes_v1'
_HEAD_KS = [6, 8, 10]
_OBJECTIVE = 'center_trace_masked_hmm_path_reconstruction_v1'
_MASK_SEMANTICS = 'xy_token_column_full_z_v1'
_SELECTION_POLICY = 'supervised_valid_xy_columns_round_half_up_leave_one_v1'
_REPLACEMENT = 'learned_encoder_mask_token_v1'
_REPLACEMENT_INITIALIZATION = 'normal_std_0p02_train_seed_salted_v1'
_RNG_POLICY = 'stateless_step_seed_v1'
_DISTILLATION_SCOPE = 'visible_only_v1'
_SUPERVISED_LOSS = 'structured_hmm_center_trace_masked_hard_v1'
_CONSISTENCY_POLICY = 'disabled_for_center_trace_masked_v1'
_HANDOFF_TYPE = 'f3_center_trace_masked_pretraining_handoff'
_EXECUTION_ARTIFACT_TYPE = 'f3_center_trace_masked_pretraining_execution'
_EXECUTION_EVIDENCE_FILENAME = (
	'.f3_center_trace_masked_pretraining_execution.json'
)
_PHASE_EVIDENCE_PREFIX = '.f3_center_trace_masked_pretraining'
_MAE_MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
_CURRENT_K6_MODEL_TAG = 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
_ALLOWED_CONFIG_DIFFERENCES = (
	'paths.output_root',
	'identity.model_tag',
	'identity.scientific_identity center-trace fields',
	'spatial_context',
)
_CENTER_SCIENTIFIC_FIELDS = (
	'experiment_role',
	'variant',
	'target_representation',
	'objective_semantics',
	'mask_semantics',
	'column_fraction',
	'selection_policy',
	'replacement',
	'replacement_initialization',
	'rng_policy',
	'masked_prototype_weight',
	'visible_prototype_weight',
	'distillation_scope',
	'supervised_loss',
	'consistency_policy',
)
_SMOKE_METRIC_KEYS = frozenset(
	{
		'loss',
		'loss_prototype',
		'loss_prototype_masked',
		'loss_prototype_visible',
		'loss_usage',
		'loss_distillation',
		'masked_supervised_token_fraction',
		'visible_supervised_token_fraction',
		'valid_distillation_token_fraction',
		'eligible_xy_column_count',
		'selected_xy_column_count',
		'loss_consistency_contribution',
	}
)
_VALIDATION_ERRORS = (
	OSError,
	TypeError,
	ValueError,
	RuntimeError,
	EOFError,
	pickle.UnpicklingError,
)


@dataclass(frozen=True)
class F3CenterTraceMaskedPretrainingValidationConfig:
	"""Closed paths needed to validate experiment 104."""

	artifact_root: Path
	experiment_root: Path
	target_manifest: Path
	hard_full_config: Path
	hard_handoff: Path
	center_trace_masked_smoke_config: Path
	center_trace_masked_full_config: Path
	center_trace_masked_embedding_config: Path


@dataclass(frozen=True)
class F3CenterTraceMaskedPretrainingValidationResult:
	"""Evidence for one phase and the optional final handoff path."""

	phase: str
	evidence: Mapping[str, object]
	published_handoff: Path | None


def f3_center_trace_masked_pretraining_validation_config_from_mapping(
	config: Mapping[str, object],
) -> F3CenterTraceMaskedPretrainingValidationConfig:
	"""Resolve the deliberately closed experiment-104 validation schema."""
	if not isinstance(config, Mapping):
		raise TypeError('center-trace validation config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown center-trace validation keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing center-trace validation keys: {sorted(missing)!r}')

	def path(key: str, *, must_exist: bool) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		result = Path(value).resolve()
		if must_exist and not result.is_file():
			raise FileNotFoundError(f'{key} is missing: {result}')
		return result

	result = F3CenterTraceMaskedPretrainingValidationConfig(
		artifact_root=path('artifact_root', must_exist=False),
		experiment_root=path('experiment_root', must_exist=False),
		target_manifest=path('target_manifest', must_exist=True),
		hard_full_config=path('hard_full_config', must_exist=True),
		hard_handoff=path('hard_handoff', must_exist=True),
		center_trace_masked_smoke_config=path(
			'center_trace_masked_smoke_config', must_exist=True
		),
		center_trace_masked_full_config=path(
			'center_trace_masked_full_config', must_exist=True
		),
		center_trace_masked_embedding_config=path(
			'center_trace_masked_embedding_config', must_exist=True
		),
	)
	if not result.artifact_root.is_dir() or not result.experiment_root.is_dir():
		raise FileNotFoundError(
			'artifact_root and experiment_root must be existing directories'
		)
	return result


def load_f3_center_trace_masked_pretraining_validation_config(
	path: str | Path,
) -> F3CenterTraceMaskedPretrainingValidationConfig:
	"""Load the experiment-104 validation YAML."""
	return f3_center_trace_masked_pretraining_validation_config_from_mapping(
		load_config(path)
	)


def load_f3_center_trace_masked_pretraining_handoff(
	path: str | Path,
) -> Mapping[str, object]:
	"""Load only the complete, versioned schema-1 PASS handoff."""
	payload = _mapping(_json(Path(path)), 'center-trace handoff')
	if set(payload) != {
		'artifact_type',
		'schema_version',
		'status',
		'model_tag',
		'variant',
		'targets',
		'checkpoint',
		'training_diagnostics',
		'embedding',
		'execution',
	}:
		raise ValueError('center-trace handoff top-level keys mismatch')
	if (
		payload.get('artifact_type') != _HANDOFF_TYPE
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
		or payload.get('model_tag') != _MODEL_TAG
		or payload.get('variant') != _VARIANT
	):
		raise ValueError('center-trace handoff identity mismatch')
	targets = _mapping(payload['targets'], 'center-trace handoff targets')
	if set(targets) != {
		'model_tag',
		'experiment_role',
		'variant',
		'target_representation',
		'target_manifest',
		'per_head_target_hashes',
		'objective_semantics',
		'mask_semantics',
		'column_fraction',
		'selection_policy',
		'replacement',
		'replacement_initialization',
		'rng_policy',
		'masked_prototype_weight',
		'visible_prototype_weight',
		'distillation_scope',
		'supervised_loss',
		'consistency_policy',
		'scientific_identity_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
		'hard_baseline_config',
		'hard_baseline_handoff',
		'hard_baseline_config_parity',
		'real_data_inputs',
		'allowed_differences',
	}:
		raise ValueError('center-trace handoff target keys mismatch')
	if (
		targets['model_tag'] != _MODEL_TAG
		or targets['experiment_role'] != _ROLE
		or targets['variant'] != _VARIANT
		or targets['target_representation'] != _TARGET_REPRESENTATION
		or targets['objective_semantics'] != _OBJECTIVE
		or targets['mask_semantics'] != _MASK_SEMANTICS
		or targets['selection_policy'] != _SELECTION_POLICY
		or targets['replacement'] != _REPLACEMENT
		or targets['replacement_initialization'] != _REPLACEMENT_INITIALIZATION
		or targets['rng_policy'] != _RNG_POLICY
		or targets['distillation_scope'] != _DISTILLATION_SCOPE
		or targets['supervised_loss'] != _SUPERVISED_LOSS
		or targets['consistency_policy'] != _CONSISTENCY_POLICY
		or targets['masked_prototype_weight'] != 0.50
		or targets['visible_prototype_weight'] != 0.50
	):
		raise ValueError('center-trace handoff scientific identity mismatch')
	for key in ('column_fraction',):
		if targets[key] != 0.10:
			raise ValueError(f'center-trace handoff {key} mismatch')
	for key in (
		'scientific_identity_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
	):
		_require_sha256(targets.get(key), f'handoff targets.{key}')
	if targets['allowed_differences'] != list(_ALLOWED_CONFIG_DIFFERENCES):
		raise ValueError('center-trace handoff allowed-difference contract mismatch')
	_validate_config_parity_evidence(targets['hard_baseline_config_parity'])
	_validate_real_data_inputs_evidence(targets['real_data_inputs'])
	_validate_reference(targets['target_manifest'], 'handoff target manifest')
	_validate_reference(targets['hard_baseline_config'], 'handoff hard baseline config')
	_validate_reference(
		targets['hard_baseline_handoff'], 'handoff hard baseline handoff'
	)
	_validate_target_hashes(targets['per_head_target_hashes'])
	checkpoint = _mapping(payload['checkpoint'], 'center-trace handoff checkpoint')
	if set(checkpoint) != {
		'path',
		'sha256',
		'latest_path',
		'latest_sha256',
		'selected_checkpoint_kind',
		'selected_epoch',
		'selected_global_step',
		'selected_loss',
		'selection_history_sha256',
		'selection_history_event_count',
		'selection_history_schema_version',
		'optimizer_group_identity',
		'trainability_summary',
		'trainability_summary_sha256',
		'schema_version',
		'scientific_identity_sha256',
	}:
		raise ValueError('center-trace handoff checkpoint keys mismatch')
	for key in (
		'sha256',
		'latest_sha256',
		'selection_history_sha256',
		'trainability_summary_sha256',
		'scientific_identity_sha256',
	):
		_require_sha256(checkpoint.get(key), f'handoff checkpoint.{key}')
	for key in ('path', 'latest_path'):
		if not isinstance(checkpoint.get(key), str) or not checkpoint[key]:
			raise TypeError(f'handoff checkpoint.{key} is missing')
	if (
		checkpoint['selected_checkpoint_kind'] != 'step'
		and checkpoint['selected_checkpoint_kind'] != 'epoch'
	):
		raise ValueError('center-trace handoff checkpoint kind mismatch')
	for key in (
		'selected_epoch',
		'selected_global_step',
		'selection_history_event_count',
		'selection_history_schema_version',
		'schema_version',
	):
		if not _nonnegative_int(checkpoint.get(key)):
			raise TypeError(f'handoff checkpoint.{key} must be nonnegative integer')
	if not _finite_number(checkpoint['selected_loss']):
		raise TypeError('handoff checkpoint.selected_loss must be finite')
	if (
		checkpoint['schema_version'] != 7
		or checkpoint['scientific_identity_sha256']
		!= targets['scientific_identity_sha256']
	):
		raise ValueError('center-trace handoff schema or scientific hash mismatch')
	if (
		not isinstance(checkpoint['optimizer_group_identity'], list)
		or len(checkpoint['optimizer_group_identity']) != 3
	):
		raise ValueError('center-trace handoff optimizer groups are incomplete')
	trainability = _mapping(
		checkpoint['trainability_summary'], 'handoff trainability summary'
	)
	if checkpoint['trainability_summary_sha256'] != scientific_identity_sha256(
		trainability
	):
		raise ValueError('center-trace handoff trainability hash mismatch')
	diagnostics = _mapping(
		payload['training_diagnostics'], 'center-trace handoff training diagnostics'
	)
	if set(diagnostics) != {'path', 'sha256'}:
		raise ValueError('center-trace handoff training diagnostics keys mismatch')
	_require_sha256(
		diagnostics.get('sha256'), 'handoff training diagnostics.sha256'
	)
	if not isinstance(diagnostics.get('path'), str) or not diagnostics['path']:
		raise TypeError('handoff training diagnostics.path is missing')
	embedding = _mapping(payload['embedding'], 'center-trace handoff embedding')
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
		'canonical_valid_token_identities',
	}:
		raise ValueError('center-trace handoff embedding keys mismatch')
	for key in (
		'metadata_sha256',
		'embeddings_sha256',
		'valid_tokens_sha256',
	):
		_require_sha256(embedding.get(key), f'handoff embedding.{key}')
	for key in (
		'root',
		'metadata_path',
		'embeddings_path',
		'valid_tokens_path',
	):
		if not isinstance(embedding.get(key), str) or not embedding[key]:
			raise TypeError(f'handoff embedding.{key} is missing')
	if (
		embedding['embeddings_shape'] != [76, 113, 32, 384]
		or embedding['embeddings_dtype'] != 'float16'
		or embedding['valid_tokens_shape'] != [76, 113, 32]
		or embedding['valid_tokens_dtype'] != 'bool'
		or not _positive_int(embedding['finite_valid_count'])
	):
		raise ValueError('center-trace handoff embedding array identity mismatch')
	_validate_canonical_valid_token_identities(
		embedding['canonical_valid_token_identities'],
		embedding['valid_tokens_sha256'],
	)
	execution = _mapping(payload['execution'], 'center-trace handoff execution')
	if set(execution) != {'before', 'after'}:
		raise ValueError('center-trace handoff execution keys mismatch')
	_validate_execution_state(execution['before'], 'handoff execution.before')
	_validate_execution_state(execution['after'], 'handoff execution.after')
	return payload


def validate_f3_center_trace_masked_pretraining(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	phase: str,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3CenterTraceMaskedPretrainingValidationResult:
	"""Validate inputs, smoke, checkpoints, and final extraction in order."""
	if phase not in {'inputs', 'smoke', 'checkpoints', 'complete'}:
		raise ValueError('phase must be inputs, smoke, checkpoints, or complete')
	try:
		target = load_multi_head_target_manifest(config.target_manifest)
		hard = _training_config(config.hard_full_config)
		full = _training_config(config.center_trace_masked_full_config)
		target_evidence = _target_evidence(config, target=target, hard=hard, full=full)
		if phase == 'inputs':
			execution = _start_execution_evidence(config, dry_run=dry_run)
			evidence = {
				'status': 'PASS',
				**target_evidence,
				'execution': execution,
			}
			if not dry_run:
				evidence['phase_evidence_path'] = str(
					_write_phase_evidence(config, phase=phase, evidence=evidence)
				)
			return F3CenterTraceMaskedPretrainingValidationResult(
				phase,
				evidence,
				None,
			)
		if phase == 'smoke':
			smoke = _training_config(config.center_trace_masked_smoke_config)
			smoke_evidence = _smoke_evidence(
				config,
				full=full,
				smoke=smoke,
				quarantine_invalid=quarantine_invalid,
				dry_run=dry_run,
			)
			execution = _update_execution_evidence(
				config, phase=phase, dry_run=dry_run
			)
			evidence = {
				'status': 'PASS',
				**target_evidence,
				'smoke': smoke_evidence,
				'execution': execution,
			}
			if not dry_run:
				evidence['phase_evidence_path'] = str(
					_write_phase_evidence(config, phase=phase, evidence=evidence)
				)
			return F3CenterTraceMaskedPretrainingValidationResult(
				phase,
				evidence,
				None,
			)
		checkpoint = _checkpoint_evidence(
			full,
			runtime=_mapping(target_evidence['hard_baseline_config_parity'], 'parity')[
				'candidate_runtime'
			],
			expected_global_step=25600,
			require_full_epoch_history=True,
		)
		evidence: dict[str, object] = {
			'status': 'PASS',
			**target_evidence,
			**checkpoint,
		}
		if phase == 'checkpoints':
			if not dry_run:
				_atomic_json(
					Path(str(checkpoint['root']))
					/ 'preflight'
					/ 'center_trace_masked_checkpoint_validation.json',
					{
						'artifact_type': 'f3_center_trace_masked_pretraining_validation',
						'schema_version': 1,
						'phase': phase,
						'status': 'PASS',
						'target_manifest_sha256': target_evidence['target_manifest'][
							'sha256'
						],
						'checkpoint_sha256': checkpoint['selected_sha256'],
					},
				)
			return F3CenterTraceMaskedPretrainingValidationResult(phase, evidence, None)
		evidence['embedding'] = _embedding_evidence(config, checkpoint, full)
		if not dry_run:
			smoke = _training_config(config.center_trace_masked_smoke_config)
			smoke_phase_evidence = _load_phase_evidence(config, phase='smoke')
			_validate_smoke_phase_evidence(
				config,
				phase_evidence=smoke_phase_evidence,
				target_evidence=target_evidence,
				full=full,
				smoke=smoke,
			)
		evidence['execution'] = _update_execution_evidence(
			config, phase=phase, dry_run=dry_run
		)
		handoff = _handoff(evidence)
		handoff_path = (
			Path(str(checkpoint['root']))
			/ 'preflight'
			/ 'center_trace_masked_handoff.json'
		)
		if dry_run:
			return F3CenterTraceMaskedPretrainingValidationResult(phase, evidence, None)
		published = _publish_handoff(
			handoff_path,
			handoff,
			only_missing=only_missing,
			quarantine_invalid=quarantine_invalid,
		)
		return F3CenterTraceMaskedPretrainingValidationResult(
			phase, evidence, handoff_path if published else None
		)
	except _VALIDATION_ERRORS as error:
		if not dry_run:
			raise
		return F3CenterTraceMaskedPretrainingValidationResult(
			phase,
			{'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'},
			None,
		)


def _training_config(path: Path) -> Mapping[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(path))


def _target_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	target: Mapping[str, object],
	hard: Mapping[str, object],
	full: Mapping[str, object],
) -> dict[str, object]:
	if target.get('schema_version') not in {1, 2} or target.get('head_ks') != _HEAD_KS:
		raise ValueError('center-trace target must be the immutable K6/K8/K10 manifest')
	if target.get('artifact_type') != 'strat_hmm_multi_head_target_manifest':
		raise ValueError('center-trace target artifact type mismatch')
	target_hashes = _multi_head_target_hashes(target)
	_validate_target_hashes(target_hashes)
	_validate_target_file_references(target)
	_validate_hard_target_representation(full, 'center-trace')
	if (
		_manifest_path(hard) != config.target_manifest
		or _manifest_path(full) != config.target_manifest
	):
		raise ValueError(
			'hard and center-trace configs must use the supplied target manifest'
		)
	if (
		_model_tag(hard, 'hard') != _HARD_MODEL_TAG
		or _model_tag(full, 'center-trace') != _MODEL_TAG
	):
		raise ValueError('hard or center-trace model tag mismatch')
	for label, training, model_tag in (
		('hard', hard, _HARD_MODEL_TAG),
		('center-trace', full, _MODEL_TAG),
	):
		root = Path(
			str(_mapping(training['paths'], f'{label} paths')['output_root'])
		).resolve()
		if root != (config.experiment_root / model_tag).resolve():
			raise ValueError(f'{label} output root mismatch')
	_validate_center_config(full, config.target_manifest, target_hashes)
	hard_identity = _scientific(hard, 'hard')
	if hard_identity.get('target_manifest_sha256') != file_sha256(
		config.target_manifest
	):
		raise ValueError('hard baseline target manifest SHA-256 mismatch')
	hard_handoff = hard_validation.load_f3_multi_head_pretraining_handoff(
		config.hard_handoff
	)
	hard_checkpoint = _hard_baseline_checkpoint_evidence(
		config, hard, hard_handoff, target_hashes
	)
	parity = _hard_config_parity(hard, full)
	if (
		_mapping(parity['hard_runtime'], 'hard runtime')['initial_student_state_sha256']
		!= hard_checkpoint['initial_student_state_sha256']
	):
		raise ValueError(
			'hard baseline runtime/student initialization differs from handoff'
		)
	if (
		_mapping(parity['hard_runtime'], 'hard runtime')['initial_head_state_sha256']
		!= hard_checkpoint['initial_head_state_sha256']
	):
			raise ValueError(
			'hard baseline runtime/head initialization differs from handoff'
		)
	real_data_inputs = _real_data_input_evidence(full)
	return {
		'target_representation': _scientific(full, 'center-trace')[
			'target_representation'
		],
		'target_manifest': {
			'path': str(config.target_manifest),
			'sha256': file_sha256(config.target_manifest),
		},
		'per_head_target_hashes': target_hashes,
		'boundary_weight_semantics': 'valid_token_indicator_v1',
		'hard_baseline_config': _reference(config.hard_full_config),
		'hard_baseline_handoff': _reference(config.hard_handoff),
		'hard_baseline_checkpoint': hard_checkpoint,
		'hard_baseline_config_parity': parity,
		'real_data_inputs': real_data_inputs,
	}


def _real_data_input_evidence(training: Mapping[str, object]) -> dict[str, object]:
	"""Validate and fingerprint the real amplitude inputs used by training."""
	manifests = _mapping(training['manifests'], 'training manifests')
	manifest_path = _required_file_path(manifests, 'train', 'manifests.train')
	path_list_path = _required_file_path(
		manifests, 'train_path_list', 'manifests.train_path_list'
	)
	train_manifests = read_manifest_json(manifest_path)
	if not train_manifests:
		raise ValueError('real-data amplitude manifest must contain at least one survey')
	listed_paths = [
		str(Path(line).expanduser().resolve())
		for line in path_list_path.read_text(encoding='utf-8').splitlines()
		if line.strip()
	]
	surveys: list[dict[str, object]] = []
	expected_paths: list[str] = []
	for manifest in train_manifests:
		amplitude_path = resolve_manifest_path(
			manifest, manifest.amplitude.path
		).resolve()
		stats_path = resolve_manifest_path(
			manifest, manifest.amplitude.normalization_stats_path
		).resolve()
		if not amplitude_path.is_file():
			raise FileNotFoundError(
				f'real-data amplitude input is missing: {amplitude_path}'
			)
		if not stats_path.is_file():
			raise FileNotFoundError(
				f'real-data normalization input is missing: {stats_path}'
			)
		volume = inspect_npy_volume(amplitude_path)
		if (
			volume.shape_xyz != manifest.amplitude.shape_xyz
			or volume.dtype != manifest.amplitude.dtype
			or volume.ndim != 3
		):
			raise ValueError(
			f'real-data amplitude metadata mismatch for {manifest.survey_id!r}'
			)
		stats = load_normalization_stats(stats_path)
		if (
			stats.survey_id != manifest.survey_id
			or stats.grid_order != manifest.amplitude.grid_order
		):
			raise ValueError(
			f'real-data normalization metadata mismatch for {manifest.survey_id!r}'
			)
		expected_paths.append(str(amplitude_path))
		surveys.append(
			{
				'survey_id': manifest.survey_id,
				'amplitude': _reference(amplitude_path),
				'normalization_stats': _reference(stats_path),
				'shape_xyz': list(volume.shape_xyz),
				'dtype': volume.dtype,
				'grid_order': list(manifest.amplitude.grid_order),
			}
		)
	if listed_paths != expected_paths:
		raise ValueError(
			'manifests.train_path_list must exactly enumerate the amplitude manifest'
		)
	teacher = _mapping(training['teacher'], 'training teacher')
	student = _mapping(training['student'], 'training student')
	return {
		'train_manifest': _reference(manifest_path),
		'train_path_list': _reference(path_list_path),
		'teacher_checkpoint': _reference(
			_required_file_path(teacher, 'checkpoint', 'teacher.checkpoint')
		),
		'student_init_checkpoint': _reference(
			_required_file_path(
				student, 'init_checkpoint', 'student.init_checkpoint'
			)
		),
		'survey_count': len(surveys),
		'surveys': surveys,
	}


def _validate_real_data_inputs_evidence(value: object) -> None:
	inputs = _mapping(value, 'real-data input evidence')
	if set(inputs) != {
		'train_manifest',
		'train_path_list',
		'teacher_checkpoint',
		'student_init_checkpoint',
		'survey_count',
		'surveys',
	}:
		raise ValueError('real-data input evidence keys mismatch')
	for key in (
		'train_manifest',
		'train_path_list',
		'teacher_checkpoint',
		'student_init_checkpoint',
	):
		_validate_reference(inputs[key], f'real-data inputs.{key}')
	surveys = inputs['surveys']
	if (
		isinstance(inputs['survey_count'], bool)
		or not isinstance(inputs['survey_count'], int)
		or inputs['survey_count'] <= 0
		or not isinstance(surveys, list)
		or len(surveys) != inputs['survey_count']
	):
		raise ValueError('real-data input survey count is invalid')
	for index, survey_value in enumerate(surveys):
		survey = _mapping(survey_value, f'real-data input survey {index}')
		if set(survey) != {
			'survey_id',
			'amplitude',
			'normalization_stats',
			'shape_xyz',
			'dtype',
			'grid_order',
		}:
			raise ValueError('real-data input survey keys mismatch')
		if not isinstance(survey['survey_id'], str) or not survey['survey_id']:
			raise TypeError('real-data input survey_id is missing')
		_validate_reference(
			survey['amplitude'], f'real-data input survey {index}.amplitude'
		)
		_validate_reference(
			survey['normalization_stats'],
			f'real-data input survey {index}.normalization_stats',
		)
		shape = survey['shape_xyz']
		if (
			not isinstance(shape, list)
			or len(shape) != 3
			or any(
				isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0
				for axis in shape
			)
		):
			raise ValueError('real-data input survey shape is invalid')
		if not isinstance(survey['dtype'], str) or not survey['dtype']:
			raise TypeError('real-data input survey dtype is missing')
		if survey['grid_order'] != ['x', 'y', 'z']:
			raise ValueError('real-data input survey grid order is invalid')


def _validate_config_parity_evidence(value: object) -> None:
	parity = _mapping(value, 'hard baseline config parity')
	if set(parity) != {
		'status',
		'allowed_differences',
		'hard_runtime',
		'candidate_runtime',
	}:
		raise ValueError('hard baseline config parity keys mismatch')
	if (
		parity['status'] != 'PASS'
		or parity['allowed_differences'] != list(_ALLOWED_CONFIG_DIFFERENCES)
	):
		raise ValueError('hard baseline config parity identity mismatch')
	hard_runtime = _mapping(parity['hard_runtime'], 'hard baseline runtime parity')
	candidate_runtime = _mapping(
		parity['candidate_runtime'], 'candidate runtime parity'
	)
	_validate_runtime_contract_evidence(
		hard_runtime, 'hard baseline runtime parity', require_spatial_context=False
	)
	_validate_runtime_contract_evidence(
		candidate_runtime, 'candidate runtime parity', require_spatial_context=True
	)
	if hard_runtime['initial_spatial_context_state_sha256'] is not None:
		raise ValueError('hard baseline runtime unexpectedly has spatial context')
	hard_groups = hard_runtime['optimizer_group_identity']
	candidate_groups = candidate_runtime['optimizer_group_identity']
	if candidate_groups[:2] != hard_groups or len(candidate_groups) != 3:
		raise ValueError('config parity optimizer groups do not extend baseline')
	if candidate_groups[2] != {
		'name': 'spatial_context',
		'parameter_names': ['spatial_context.replacement_token'],
		'lr': candidate_groups[2]['lr'],
	}:
		raise ValueError('config parity replacement optimizer group is invalid')


def _validate_runtime_contract_evidence(
	value: Mapping[str, object], label: str, *, require_spatial_context: bool
) -> None:
	if set(value) != {
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
		'trainability_summary',
		'optimizer_group_identity',
	}:
		raise ValueError(f'{label} keys mismatch')
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		_require_sha256(value[key], f'{label}.{key}')
	spatial_hash = value['initial_spatial_context_state_sha256']
	if require_spatial_context:
		_require_sha256(spatial_hash, f'{label}.initial_spatial_context_state_sha256')
	elif spatial_hash is not None:
		raise ValueError(f'{label} unexpectedly contains spatial context')
	_mapping(value['trainability_summary'], f'{label}.trainability_summary')
	groups = value['optimizer_group_identity']
	if not isinstance(groups, list) or len(groups) not in {2, 3}:
		raise ValueError(f'{label}.optimizer_group_identity is invalid')
	for index, group_value in enumerate(groups):
		group = _mapping(group_value, f'{label}.optimizer_group_identity[{index}]')
		if set(group) != {'name', 'parameter_names', 'lr'}:
			raise ValueError(f'{label}.optimizer_group_identity keys mismatch')
		if not isinstance(group['name'], str) or not group['name']:
			raise TypeError(f'{label}.optimizer group name is missing')
		if not isinstance(group['parameter_names'], list) or any(
			not isinstance(name, str) for name in group['parameter_names']
		):
			raise TypeError(f'{label}.optimizer group parameter names are invalid')
		if not _finite_number(group['lr']):
			raise TypeError(f'{label}.optimizer group learning rate is invalid')
	if require_spatial_context and len(groups) != 3:
		raise ValueError(f'{label} must contain a spatial_context optimizer group')


def _required_file_path(
	section: Mapping[str, object], key: str, label: str
) -> Path:
	value = section.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	return path


def _validate_center_config(
	training: Mapping[str, object], target_manifest: Path, target_hashes: object
) -> None:
	identity = _mapping(training['identity'], 'center-trace identity')
	scientific = _mapping(identity['scientific_identity'], 'center scientific identity')
	if identity.get('model_tag') != _MODEL_TAG:
		raise ValueError('center-trace scientific model tag mismatch')
	if _manifest_path(training) != target_manifest:
		raise ValueError('center-trace pseudo target path mismatch')
	if scientific.get('target_manifest_sha256') != file_sha256(target_manifest):
		raise ValueError('center-trace scientific target SHA-256 mismatch')
	if scientific.get('target_head_hashes') != target_hashes:
		raise ValueError('center-trace scientific target head hashes mismatch')
	expected = {
		'experiment_role': _ROLE,
		'variant': _VARIANT,
		'head_spec': _HEAD_SPEC,
		'head_ks': _HEAD_KS,
		'target_representation': _TARGET_REPRESENTATION,
		'objective_semantics': _OBJECTIVE,
		'mask_semantics': _MASK_SEMANTICS,
		'column_fraction': 0.10,
		'selection_policy': _SELECTION_POLICY,
		'replacement': _REPLACEMENT,
		'replacement_initialization': _REPLACEMENT_INITIALIZATION,
		'rng_policy': _RNG_POLICY,
		'masked_prototype_weight': 0.50,
		'visible_prototype_weight': 0.50,
		'distillation_scope': _DISTILLATION_SCOPE,
		'supervised_loss': _SUPERVISED_LOSS,
		'consistency_policy': _CONSISTENCY_POLICY,
	}
	for key, value in expected.items():
		if scientific.get(key) != value:
			raise ValueError(f'center-trace scientific identity mismatch: {key}')
	for forbidden in (
		'posterior_manifest_sha256',
		'posterior_semantics',
		'lateral_target_manifest_sha256',
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_unanimous_target_manifest_sha256',
	):
		if forbidden in scientific:
			raise ValueError(f'center-trace identity must not carry {forbidden}')
	loss = _mapping(training['loss'], 'center loss')
	for key, value in (
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('entropy_floor', None),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('distillation_weight', 0.2),
	):
		if loss.get(key) != value:
			raise ValueError(f'center-trace loss contract drifted: {key}')
	if (
		_mapping(training['head'], 'center head').get('spec') != _HEAD_SPEC
		or _mapping(training['head'], 'center head').get('ks') != _HEAD_KS
	):
		raise ValueError('center-trace head contract drifted')
	spatial = _mapping(training.get('spatial_context'), 'spatial_context')
	spatial_expected = {
		'objective': _OBJECTIVE,
		'mask_semantics': _MASK_SEMANTICS,
		'column_fraction': 0.10,
		'selection_policy': _SELECTION_POLICY,
		'replacement': _REPLACEMENT,
		'replacement_initialization': _REPLACEMENT_INITIALIZATION,
		'rng_policy': _RNG_POLICY,
		'masked_prototype_weight': 0.50,
		'visible_prototype_weight': 0.50,
		'distillation_scope': _DISTILLATION_SCOPE,
	}
	if dict(spatial) != spatial_expected:
		raise ValueError('center-trace spatial_context contract drifted')


def _hard_config_parity(
	hard: Mapping[str, object], center: Mapping[str, object]
) -> dict[str, object]:
	"""Prove parity outside the four issue-305 difference categories."""
	_validate_hard_target_representation(hard, 'hard')
	_validate_hard_target_representation(center, 'center-trace')
	left, right = json.loads(json.dumps(hard)), json.loads(json.dumps(center))
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		_mapping(value['identity'], 'identity').pop('model_tag', None)
		value.pop('spatial_context', None)
		scientific = _mapping(
			_mapping(value['identity'], 'identity')['scientific_identity'],
			'scientific identity',
		)
		for key in _CENTER_SCIENTIFIC_FIELDS:
			scientific.pop(key, None)
		_mapping(value['pseudo_targets'], 'pseudo targets').pop(
			'target_representation', None
		)
	if left != right:
		raise ValueError(
			'center-trace config differs from hard baseline outside allowed fields'
		)
	hard_runtime = _runtime_contract(hard, center=False)
	center_runtime = _runtime_contract(center, center=True)
	if (
		hard_runtime['initial_student_state_sha256']
		!= center_runtime['initial_student_state_sha256']
	):
		raise ValueError(
			'center-trace initial student state differs from hard baseline'
		)
	if (
		hard_runtime['initial_head_state_sha256']
		!= center_runtime['initial_head_state_sha256']
	):
		raise ValueError('center-trace initial head state differs from hard baseline')
	if hard_runtime['trainability_summary'] != center_runtime['trainability_summary']:
		raise ValueError('center-trace trainability differs from hard baseline')
	hard_groups = hard_runtime['optimizer_group_identity']
	center_groups = center_runtime['optimizer_group_identity']
	if center_groups[:2] != hard_groups or len(center_groups) != 3:
		raise ValueError(
			'center-trace optimizer groups do not extend the hard baseline'
		)
	spatial_group = center_groups[2]
	if spatial_group.get('name') != 'spatial_context' or spatial_group.get(
		'parameter_names'
	) != ['spatial_context.replacement_token']:
		raise ValueError('center-trace replacement optimizer group mismatch')
	return {
		'status': 'PASS',
		'allowed_differences': list(_ALLOWED_CONFIG_DIFFERENCES),
		'hard_runtime': hard_runtime,
		'candidate_runtime': center_runtime,
	}


def _runtime_contract(
	training: Mapping[str, object], *, center: bool
) -> dict[str, object]:
	train = _mapping(training['train'], 'training train')
	seed = train.get('seed')
	if isinstance(seed, bool) or not isinstance(seed, int):
		raise TypeError('training seed must be an integer')
	state_before = torch.get_rng_state().clone()
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(seed)
		components = build_strat_hmm_components(training, device='cpu')
	state_after = torch.get_rng_state()
	if not torch.equal(state_before, state_after):
		raise ValueError('runtime contract construction consumed global RNG state')
	heads = getattr(components, 'heads', None)
	if not isinstance(heads, torch.nn.Module):
		raise TypeError('center-trace runtime contract requires multi-head components')
	spatial_context_module = getattr(components, 'replacement_token', None)
	if center:
		if not isinstance(spatial_context_module, torch.nn.Module):
			raise TypeError(
				'center-trace runtime contract requires spatial_context module'
			)
		replacement_token = getattr(spatial_context_module, 'replacement_token', None)
		if not isinstance(replacement_token, torch.nn.Parameter):
			raise TypeError(
				'center-trace runtime contract requires '
				'spatial_context.replacement_token parameter'
			)
		if list(spatial_context_module.named_parameters()) != [
			('replacement_token', replacement_token)
		]:
			raise ValueError(
				'center-trace spatial_context must own only '
				'replacement_token'
			)
	else:
		replacement_token = None
	if not center and spatial_context_module is not None:
		raise ValueError('hard baseline unexpectedly has spatial context')
	modules: list[tuple[str, torch.nn.Module]] = [
		('student', components.student),
		('head', heads),
	]
	if center:
		modules.append(('spatial_context', spatial_context_module))
	parameter_names = {
		id(parameter): f'{prefix}.{name}'
		for prefix, module in modules
		for name, parameter in module.named_parameters()
	}
	groups = []
	for group in components.optimizer.param_groups:
		parameters = group.get('params')
		if not isinstance(parameters, list):
			raise TypeError('optimizer parameter group must be a list')
		try:
			names = [parameter_names[id(parameter)] for parameter in parameters]
		except KeyError as error:
			raise ValueError('optimizer contains an unknown parameter') from error
		groups.append(
			{
				'name': group.get('name'),
				'parameter_names': names,
				'lr': float(group.get('lr', 0.0)),
			}
		)
	summary = components.trainability_summary
	result = {
		'initial_student_state_sha256': hard_validation._state_sha256(
			components.student.state_dict()
		),
		'initial_head_state_sha256': hard_validation._state_sha256(heads.state_dict()),
		'initial_spatial_context_state_sha256': None,
		'trainability_summary': {
			'trainable_parameter_count': int(summary.trainable_parameter_count),
			'frozen_parameter_count': int(summary.frozen_parameter_count),
			'trainable_names': list(summary.trainable_names),
		},
		'optimizer_group_identity': groups,
	}
	if center:
		result['initial_spatial_context_state_sha256'] = hard_validation._state_sha256(
			spatial_context_module.state_dict()
		)
		if result['initial_spatial_context_state_sha256'] in {
			result['initial_student_state_sha256'],
			result['initial_head_state_sha256'],
		}:
			raise ValueError('replacement token initial hash is not independent')
	return result


def _hard_baseline_checkpoint_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	hard: Mapping[str, object],
	handoff: Mapping[str, object],
	target_hashes: object,
) -> dict[str, object]:
	record = _mapping(handoff.get('checkpoint'), 'hard baseline handoff checkpoint')
	handoff_targets = _mapping(
		handoff.get('stratigraphy_pretext'), 'hard baseline handoff targets'
	)
	if (
		_model_tag(hard, 'hard') != _HARD_MODEL_TAG
		or handoff.get('model_tag') != _HARD_MODEL_TAG
	):
		raise ValueError('hard baseline handoff model identity mismatch')
	if (
		Path(str(handoff_targets.get('target_manifest_path'))).resolve()
		!= config.target_manifest
	):
		raise ValueError('hard baseline handoff target path mismatch')
	if handoff_targets.get('target_manifest_sha256') != file_sha256(
		config.target_manifest
	):
		raise ValueError('hard baseline handoff target SHA-256 mismatch')
	if handoff_targets.get('per_head_target_sha256') != target_hashes:
		raise ValueError('hard baseline handoff target hashes mismatch')
	path = Path(str(record.get('path'))).resolve()
	if not path.is_file() or record.get('sha256') != file_sha256(path):
		raise ValueError('hard baseline handoff checkpoint identity mismatch')
	payload = _torch_mapping(path)
	validate_stratigraphy_checkpoint_payload(payload, expected_config=hard)
	if _mapping(payload.get('stratigraphy_config'), 'hard checkpoint config') != hard:
		raise ValueError(
			'hard baseline checkpoint config differs from hard full config'
		)
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'), 'hard checkpoint identity'
	)
	for key in (
		'head_spec',
		'head_ks',
		'consistency_policy',
		'consistency_weight',
		'scientific_identity_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	):
		if identity.get(key) != handoff_targets.get(key):
			raise ValueError(f'hard baseline checkpoint/handoff mismatch: {key}')
	if _mapping(identity.get('target_manifest'), 'hard checkpoint target manifest').get(
		'sha256'
	) != file_sha256(config.target_manifest):
		raise ValueError('hard baseline checkpoint target hash mismatch')
	if identity.get('per_head_targets') != target_hashes:
		raise ValueError('hard baseline checkpoint per-head target mismatch')
	trainability = _mapping(payload.get('trainability_summary'), 'hard trainability')
	groups = identity.get('optimizer_group_identity')
	if not isinstance(groups, list) or not groups:
		raise ValueError('hard baseline optimizer group identity is missing')
	return {
		'path': str(path),
		'sha256': record['sha256'],
		'initial_student_state_sha256': identity['initial_student_state_sha256'],
		'initial_head_state_sha256': identity['initial_head_state_sha256'],
		'trainability_summary': dict(trainability),
		'optimizer_group_identity': groups,
		'selected_epoch': record.get('selected_epoch'),
		'selected_global_step': record.get('selected_global_step'),
	}


def _smoke_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	full: Mapping[str, object],
	smoke: Mapping[str, object],
	quarantine_invalid: bool = False,
	dry_run: bool = False,
) -> dict[str, object]:
	smoke_root = _training_output_root(
		smoke, label='smoke output root'
	)
	full_root = _training_output_root(full, label='full output root')
	try:
		_smoke_config_contract(config, full=full, smoke=smoke)
		full_runtime = _runtime_contract(full, center=True)
		smoke_runtime = _runtime_contract(smoke, center=True)
		if full_runtime != smoke_runtime:
			raise ValueError(
				'center-trace smoke initialization or optimizer drifted from full config'
			)
		evidence = _checkpoint_evidence(
			_runtime_max_steps_config(smoke),
			runtime=smoke_runtime,
			expected_global_step=2,
			require_full_epoch_history=False,
		)
		latest = _mapping(evidence['latest'], 'center-trace smoke latest checkpoint')
		state = _mapping(
			latest.get('training_state'), 'center-trace smoke training state'
		)
		if latest.get('epoch') != 1 or state.get('checkpoint_kind') != 'step':
			raise ValueError(
				'center-trace smoke must end at epoch 1 with a step checkpoint'
			)
		if state.get('batch_index') != 1:
			raise ValueError('center-trace smoke must end after two batches')
		return {
			'root': evidence['root'],
			'device': _mapping(smoke['train'], 'smoke train')['device'],
			'max_steps_override': 2,
			'latest_path': evidence['latest_path'],
			'latest_sha256': evidence['latest_sha256'],
			'global_step': latest['global_step'],
			'epoch': latest['epoch'],
			'batch_index': state['batch_index'],
			'metrics': latest['metrics'],
			'schema_version': evidence['identity']['schema_version'],
			'initial_student_state_sha256': evidence['identity'][
				'initial_student_state_sha256'
			],
			'initial_head_state_sha256': evidence['identity'][
				'initial_head_state_sha256'
			],
			'initial_spatial_context_state_sha256': evidence['identity'][
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
			_quarantine_smoke_output(smoke_root)
		raise


def _runtime_max_steps_config(
	training: Mapping[str, object],
) -> Mapping[str, object]:
	"""Add the documented two-step CLI limit without changing identity fields."""
	resolved = json.loads(json.dumps(training))
	_mapping(resolved['train'], 'runtime train')['max_steps'] = 2
	return resolved


def _smoke_config_contract(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	full: Mapping[str, object],
	smoke: Mapping[str, object],
) -> None:
	if (
		_manifest_path(smoke) != config.target_manifest
		or _model_tag(smoke, 'smoke') != _MODEL_TAG
	):
		raise ValueError('center-trace smoke target or model identity mismatch')
	full_root = _training_output_root(full, label='full output root')
	smoke_root = _training_output_root(smoke, label='smoke output root')
	if full_root == smoke_root:
		raise ValueError('center-trace smoke output root must differ from full root')
	if full_root.exists():
		raise ValueError(
			'center-trace full output root must remain untouched during smoke'
		)
	smoke_train = _mapping(smoke['train'], 'smoke train')
	if smoke_train.get('device') != 'cpu' or smoke_train.get('max_steps') is not None:
		raise ValueError(
			'center-trace smoke must use device=cpu and a CLI max_steps override'
		)
	left, right = json.loads(json.dumps(full)), json.loads(json.dumps(smoke))
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		_mapping(value['identity'], 'identity').pop('model_tag', None)
		_mapping(value['identity'], 'identity').get('runtime_identity', {}).pop(
			'device', None
		)
		_mapping(value['train'], 'train').pop('device', None)
	if left != right:
		raise ValueError(
			'center-trace smoke/full drifted outside CPU two-step settings'
		)


def _training_output_root(
	training: Mapping[str, object],
	*,
	label: str,
) -> Path:
	return Path(
		str(_mapping(training['paths'], f'{label} paths')['output_root'])
	).resolve()


def _checkpoint_evidence(
	training: Mapping[str, object],
	*,
	runtime: Mapping[str, object],
	expected_global_step: int,
	require_full_epoch_history: bool,
) -> dict[str, object]:
	root = Path(str(_mapping(training['paths'], 'paths')['output_root']))
	latest_path, best_path = root / 'latest.pt', root / 'best.pt'
	if not latest_path.is_file() or (
		require_full_epoch_history and not best_path.is_file()
	):
		raise FileNotFoundError('required center-trace checkpoint is missing')
	latest = _checkpoint(latest_path, training, runtime)
	best = _checkpoint(best_path, training, runtime) if best_path.is_file() else None
	for payload in (latest, best):
		if payload is not None:
			hard_validation._metrics_finite(payload)
			_validate_center_smoke_metrics(payload)
	if latest.get('global_step') != expected_global_step:
		raise ValueError(
			f'center-trace checkpoint must end at global step {expected_global_step}'
		)
	if require_full_epoch_history:
		diagnostics_path = root / 'multi_head_epoch_metrics.csv'
		if latest.get('epoch') != 25 or _checkpoint_kind(latest) != 'epoch':
			raise ValueError('center-trace full run must end at epoch 25')
		if best is None:
			raise AssertionError('center-trace full validation requires best.pt')
		rows = hard_validation._epoch_rows(diagnostics_path)
		if [row['epoch'] for row in rows] != list(range(1, 26)) or rows[-1][
			'global_step'
		] != expected_global_step:
			raise ValueError('center-trace epoch metrics coverage is incomplete')
		selection = hard_validation._validate_best_selection(
			best, latest, variant=_VARIANT
		)
		selected_path, selected = best_path, best
	else:
		if latest.get('epoch') != 1 or _checkpoint_kind(latest) != 'step':
			raise ValueError('center-trace smoke must end at a step checkpoint')
		rows, selection = [], None
		selected_path, selected = latest_path, latest
	identity = _mapping(selected['stratigraphy_checkpoint'], 'selected identity')
	return {
		'root': str(root),
		'latest_path': str(latest_path),
		'latest_sha256': file_sha256(latest_path),
		'best_path': str(best_path) if best is not None else None,
		'best_sha256': file_sha256(best_path) if best is not None else None,
		'selected_path': str(selected_path),
		'selected_sha256': file_sha256(selected_path),
		'selected_checkpoint_kind': _checkpoint_kind(selected),
		'selected_epoch': selected['epoch'],
		'selected_global_step': selected['global_step'],
		'selected_loss': _mapping(selected['metrics'], 'selected metrics')['loss'],
		'identity': identity,
		'latest': latest,
		'best': best,
		'epoch_rows': rows,
		'training_diagnostics_path': (
			str(diagnostics_path) if require_full_epoch_history else None
		),
		'training_diagnostics_sha256': (
			file_sha256(diagnostics_path) if require_full_epoch_history else None
		),
		'selection': selection,
	}


def _checkpoint(
	path: Path, training: Mapping[str, object], runtime: Mapping[str, object]
) -> Mapping[str, object]:
	payload = _torch_mapping(path)
	validate_stratigraphy_checkpoint_payload(payload, expected_config=training)
	if _mapping(payload.get('stratigraphy_config'), 'checkpoint config') != training:
		raise ValueError('center-trace checkpoint config differs from resolved config')
	identity = _mapping(payload.get('stratigraphy_checkpoint'), 'checkpoint identity')
	if identity.get('schema_version') != 7 or identity.get('model_tag') != _MODEL_TAG:
		raise ValueError('center-trace checkpoint requires schema-7 candidate identity')
	scientific = _scientific(training, 'checkpoint')
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('center-trace checkpoint scientific identity hash mismatch')
	for key in (
		'objective_semantics',
		'mask_semantics',
		'column_fraction',
		'selection_policy',
		'replacement',
		'replacement_initialization',
		'rng_policy',
		'masked_prototype_weight',
		'visible_prototype_weight',
		'distillation_scope',
		'supervised_loss',
		'consistency_policy',
		'target_representation',
	):
		if identity.get(key) != scientific.get(key):
			raise ValueError(f'center-trace checkpoint identity mismatch: {key}')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
	):
		if identity.get(key) != runtime.get(key):
			raise ValueError(f'center-trace checkpoint initialization mismatch: {key}')
	if (
		_mapping(payload.get('trainability_summary'), 'checkpoint trainability summary')
		!= runtime['trainability_summary']
	):
		raise ValueError('center-trace checkpoint trainability mismatch')
	if identity.get('optimizer_group_identity') != runtime['optimizer_group_identity']:
		raise ValueError('center-trace checkpoint optimizer identity mismatch')
	return payload


def _validate_center_smoke_metrics(payload: Mapping[str, object]) -> None:
	metrics = _mapping(payload.get('metrics'), 'center-trace metrics')
	missing = sorted(_SMOKE_METRIC_KEYS - set(metrics))
	for head_k in _HEAD_KS:
		for prefix in (
			'loss_prototype_masked',
			'loss_prototype_visible',
			'loss_usage',
			'target_usage_entropy',
			'prototype_usage_entropy',
			'masked_top1_accuracy',
		):
			key = f'{prefix}_k{head_k}'
			if key not in metrics:
				missing.append(key)
	if missing:
		raise ValueError(f'center-trace smoke metrics are missing: {missing!r}')
	consistency = metrics['loss_consistency_contribution']
	if not _finite_number(consistency) or float(consistency) != 0.0:
		raise ValueError('center-trace consistency contribution must be exactly zero')
	for head_k in _HEAD_KS:
		accuracy = metrics[f'masked_top1_accuracy_k{head_k}']
		if not _finite_number(accuracy) or not 0.0 <= float(accuracy) <= 1.0:
			raise ValueError(
				f'center-trace masked top-1 accuracy for K={head_k} must be in [0, 1]'
			)
	masked_fraction = float(metrics['masked_supervised_token_fraction'])
	visible_fraction = float(metrics['visible_supervised_token_fraction'])
	eligible = float(metrics['eligible_xy_column_count'])
	selected = float(metrics['selected_xy_column_count'])
	if masked_fraction <= 0.0 or visible_fraction <= 0.0:
		raise ValueError(
			'center-trace smoke must exercise masked and visible supervised tokens'
		)
	if eligible <= 0.0 or selected <= 0.0 or selected > eligible:
		raise ValueError('center-trace smoke XY selection counts are invalid')


def _embedding_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	checkpoint: Mapping[str, object],
	training: Mapping[str, object],
) -> dict[str, object]:
	extraction = resolve_embedding_extraction_config(
		load_config(config.center_trace_masked_embedding_config)
	)
	paths = _mapping(extraction['paths'], 'center-trace embedding paths')
	if Path(str(paths['artifact_root'])).resolve() != config.artifact_root:
		raise ValueError('center-trace embedding artifact root differs from validation')
	embeddings_config = _mapping(
		extraction['embeddings'], 'center-trace embedding extraction'
	)
	root = Path(str(embeddings_config['output_dir'])).resolve()
	files = output_paths(root, 'f3_facies_benchmark')
	if not all(
		path.is_file()
		for path in (files.embeddings, files.valid_tokens, files.metadata)
	):
		raise FileNotFoundError('center-trace embedding artifacts are incomplete')
	selected = Path(str(checkpoint['selected_path'])).resolve()
	if Path(str(embeddings_config['checkpoint'])).resolve() != selected:
		raise ValueError(
			'center-trace embedding extraction config does not bind selected checkpoint'
		)
	manifest_config = _mapping(
		extraction['manifests'], 'center-trace embedding manifests'
	)
	if Path(str(manifest_config['input'])).resolve() != _train_manifest_path(training):
		raise ValueError(
			'center-trace embedding extraction config does not bind training manifest'
		)
	metadata = _mapping(_json(files.metadata), 'center-trace embedding metadata')
	if Path(str(metadata.get('checkpoint_path'))).resolve() != selected:
		raise ValueError('center-trace embedding does not bind selected checkpoint')
	if metadata.get('checkpoint_sha256') != file_sha256(selected):
		raise ValueError('center-trace embedding checkpoint SHA-256 mismatch')
	_validate_embedding_extraction_identity(metadata, extraction)
	identity = _mapping(checkpoint['identity'], 'center-trace checkpoint identity')
	_validate_embedding_identity(metadata, identity, training)
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	if (
		embeddings.shape != (76, 113, 32, 384)
		or embeddings.dtype != np.float16
		or valid.shape != (76, 113, 32)
		or valid.dtype != np.bool_
		or not int(valid.sum())
		or not np.isfinite(embeddings[valid]).all()
	):
		raise ValueError('center-trace embedding shape/dtype/finite contract mismatch')
	valid_sha256 = file_sha256(files.valid_tokens)
	canonical = _canonical_valid_token_identities(config)
	if any(reference['sha256'] != valid_sha256 for reference in canonical.values()):
		raise ValueError(
			'center-trace valid-token mask differs from a canonical baseline'
		)
	return {
		'root': str(root),
		'metadata_path': str(files.metadata),
		'metadata_sha256': file_sha256(files.metadata),
		'embeddings_path': str(files.embeddings),
		'embeddings_sha256': file_sha256(files.embeddings),
		'valid_tokens_path': str(files.valid_tokens),
		'valid_tokens_sha256': valid_sha256,
		'embeddings_shape': list(embeddings.shape),
		'embeddings_dtype': str(embeddings.dtype),
		'valid_tokens_shape': list(valid.shape),
		'valid_tokens_dtype': str(valid.dtype),
		'finite_valid_count': int(valid.sum()),
		'canonical_valid_token_identities': canonical,
	}


def _validate_embedding_extraction_identity(
	metadata: Mapping[str, object], extraction: Mapping[str, object]
) -> None:
	"""Bind extraction settings to the metadata that produced the arrays."""
	embedding_config = _mapping(
		extraction['embedding'], 'center-trace embedding extraction settings'
	)
	for metadata_key, config_key in (
		('window_size', 'window_size'),
		('overlap', 'overlap'),
		('output_dtype', 'output_dtype'),
		('min_token_valid_fraction', 'min_token_valid_fraction'),
	):
		if metadata.get(metadata_key) != embedding_config.get(config_key):
			raise ValueError(
				'center-trace embedding extraction setting mismatch: '
				f'{config_key}'
			)
	precision = _mapping(metadata.get('precision'), 'center-trace embedding precision')
	for metadata_key, config_key in (
		('amp_requested', 'amp'),
		('amp_dtype_requested', 'amp_dtype'),
	):
		expected = embedding_config.get(
			config_key, False if config_key == 'amp' else 'auto'
		)
		if precision.get(metadata_key) != expected:
			raise ValueError(
				'center-trace embedding extraction precision mismatch: '
				f'{config_key}'
			)
	cache_config = _mapping(
		embedding_config.get('preprocessing_cache', {'mode': 'off'}),
		'center-trace preprocessing cache config',
	)
	cache_metadata = _mapping(
		metadata.get('preprocessing_cache'),
		'center-trace embedding preprocessing cache',
	)
	if cache_metadata.get('requested_mode') != cache_config.get('mode', 'off'):
		raise ValueError('center-trace embedding preprocessing cache mismatch')


def _validate_embedding_identity(
	metadata: Mapping[str, object],
	identity: Mapping[str, object],
	training: Mapping[str, object],
) -> None:
	stratigraphy = _mapping(metadata.get('stratigraphy_pretext'), 'embedding identity')
	head = _mapping(training['head'], 'embedding head')
	student = _mapping(training['student'], 'embedding student')
	loss = _mapping(training['loss'], 'embedding loss')
	for key, expected in (
		('method', 'strat_hmm_multi_head_pretext'),
		('base_objective', 'amp_mae3d'),
		('head_spec', _HEAD_SPEC),
		('head_ks', _HEAD_KS),
		('head_count', 3),
		('unfreeze_top_blocks', student['unfreeze_top_blocks']),
		('distillation_weight', loss['distillation_weight']),
		('prototype_weight', loss['prototype_weight']),
		('prototype_weight_semantics', 'mean_across_heads'),
		('usage_weight', loss['usage_weight']),
		('usage_weight_semantics', 'mean_across_heads'),
		('consistency_policy', _CONSISTENCY_POLICY),
		('consistency_weight', loss['consistency_weight']),
		('consistency_beta', loss['consistency_beta']),
		('model_tag', _MODEL_TAG),
		('scientific_identity_sha256', identity['scientific_identity_sha256']),
	):
		if stratigraphy.get(key) != expected:
			raise ValueError(f'center-trace embedding identity mismatch: {key}')
	for key in (
		'target_representation',
		'objective_semantics',
		'mask_semantics',
		'column_fraction',
		'selection_policy',
		'replacement',
		'replacement_initialization',
		'rng_policy',
		'masked_prototype_weight',
		'visible_prototype_weight',
		'distillation_scope',
		'supervised_loss',
	):
		if stratigraphy.get(key) != identity.get(key):
			raise ValueError(f'center-trace embedding identity mismatch: {key}')
	manifest = _mapping(identity['target_manifest'], 'checkpoint target manifest')
	if (
		stratigraphy.get('target_manifest_path') != manifest['path']
		or stratigraphy.get('target_manifest_sha256') != manifest['sha256']
	):
		raise ValueError('center-trace embedding target manifest identity mismatch')
	if stratigraphy.get('per_head_target_sha256') != identity['per_head_targets']:
		raise ValueError('center-trace embedding target head identity mismatch')
	if (
		stratigraphy.get('checkpoint_stratigraphy_state_sha256')
		!= identity['stratigraphy_state_sha256']
	):
		raise ValueError('center-trace embedding checkpoint state identity mismatch')
	if head.get('spec') != _HEAD_SPEC or head.get('ks') != _HEAD_KS:
		raise ValueError('center-trace embedding head config mismatch')


def _canonical_valid_token_identities(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
) -> dict[str, dict[str, str]]:
	result = {}
	for role, model_tag in (
		('mae', _MAE_MODEL_TAG),
		('current_k6', _CURRENT_K6_MODEL_TAG),
		('mh_nocons', _HARD_MODEL_TAG),
	):
		path = output_paths(
			config.artifact_root
			/ 'embeddings/f3/facies_benchmark_v1'
			/ model_tag
			/ 'overlap_x16',
			'f3_facies_benchmark',
		).valid_tokens
		if not path.is_file():
			raise FileNotFoundError(
				f'{role} canonical valid-token artifact is missing: {path}'
			)
		result[role] = {'path': str(path), 'sha256': file_sha256(path)}
	if len({reference['sha256'] for reference in result.values()}) != 1:
		raise ValueError('canonical valid-token masks are not bitwise identical')
	return result


def _handoff(evidence: Mapping[str, object]) -> dict[str, object]:
	identity = _mapping(evidence['identity'], 'candidate checkpoint identity')
	selection = _mapping(evidence['selection'], 'checkpoint selection')
	selected = _mapping(selection['selected'], 'selected checkpoint event')
	embedding = _mapping(evidence['embedding'], 'embedding evidence')
	targets = {
		'model_tag': _MODEL_TAG,
		'experiment_role': _ROLE,
		'variant': _VARIANT,
		'target_representation': _TARGET_REPRESENTATION,
		'target_manifest': evidence['target_manifest'],
		'per_head_target_hashes': evidence['per_head_target_hashes'],
		'objective_semantics': _OBJECTIVE,
		'mask_semantics': _MASK_SEMANTICS,
		'column_fraction': 0.10,
		'selection_policy': _SELECTION_POLICY,
		'replacement': _REPLACEMENT,
		'replacement_initialization': _REPLACEMENT_INITIALIZATION,
		'rng_policy': _RNG_POLICY,
		'masked_prototype_weight': 0.50,
		'visible_prototype_weight': 0.50,
		'distillation_scope': _DISTILLATION_SCOPE,
		'supervised_loss': _SUPERVISED_LOSS,
		'consistency_policy': _CONSISTENCY_POLICY,
		'scientific_identity_sha256': identity['scientific_identity_sha256'],
		'initial_student_state_sha256': identity['initial_student_state_sha256'],
		'initial_head_state_sha256': identity['initial_head_state_sha256'],
		'initial_spatial_context_state_sha256': identity[
			'initial_spatial_context_state_sha256'
		],
			'hard_baseline_config': evidence['hard_baseline_config'],
			'hard_baseline_handoff': evidence['hard_baseline_handoff'],
			'hard_baseline_config_parity': evidence['hard_baseline_config_parity'],
			'real_data_inputs': evidence['real_data_inputs'],
			'allowed_differences': list(_ALLOWED_CONFIG_DIFFERENCES),
		}
	trainability = _mapping(evidence['best']['trainability_summary'], 'trainability')
	return {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': _MODEL_TAG,
		'variant': _VARIANT,
		'targets': targets,
		'checkpoint': {
			'path': evidence['selected_path'],
			'sha256': evidence['selected_sha256'],
			'latest_path': evidence['latest_path'],
			'latest_sha256': evidence['latest_sha256'],
			'selected_checkpoint_kind': selected['checkpoint_kind'],
			'selected_epoch': selected['epoch'],
			'selected_global_step': selected['global_step'],
			'selected_loss': selected['loss'],
			'selection_history_sha256': selection['sha256'],
			'selection_history_event_count': selection['event_count'],
			'selection_history_schema_version': selection['schema_version'],
			'optimizer_group_identity': identity['optimizer_group_identity'],
			'trainability_summary': dict(trainability),
			'trainability_summary_sha256': scientific_identity_sha256(trainability),
			'schema_version': identity['schema_version'],
			'scientific_identity_sha256': identity['scientific_identity_sha256'],
		},
		'training_diagnostics': {
			'path': evidence['training_diagnostics_path'],
			'sha256': evidence['training_diagnostics_sha256'],
		},
		'embedding': dict(embedding),
		'execution': dict(evidence['execution']),
	}


def _publish_handoff(
	path: Path,
	handoff: Mapping[str, object],
	*,
	only_missing: bool,
	quarantine_invalid: bool,
) -> bool:
	if path.is_file():
		try:
			existing = load_f3_center_trace_masked_pretraining_handoff(path)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == handoff:
			# An exact PASS handoff is reusable even without --only-missing.
			return not only_missing
		if existing != handoff and not quarantine_invalid:
			raise ValueError(
				'existing center-trace handoff is stale or invalid; '
				'pass --quarantine-invalid to replace it'
			)
		if existing != handoff:
			hard_validation._quarantine(path)
	_atomic_json(path, handoff)
	return True


def _execution_evidence_path(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
) -> Path:
	"""Return the run-state sidecar shared by the validator phases."""
	return config.experiment_root / _EXECUTION_EVIDENCE_FILENAME


def _phase_evidence_path(
	config: F3CenterTraceMaskedPretrainingValidationConfig, phase: str
) -> Path:
	if phase not in {'inputs', 'smoke'}:
		raise ValueError('phase evidence is only available for inputs and smoke')
	return config.experiment_root / f'{_PHASE_EVIDENCE_PREFIX}_{phase}.json'


def _write_phase_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	phase: str,
	evidence: Mapping[str, object],
) -> Path:
	"""Persist a PASS report without publishing the future handoff."""
	path = _phase_evidence_path(config, phase)
	payload: dict[str, object] = {
		'artifact_type': 'f3_center_trace_masked_pretraining_validation',
		'schema_version': 1,
		'phase': phase,
		'status': 'PASS',
		'evidence': dict(evidence),
	}
	if phase == 'smoke':
		payload['binding'] = _smoke_phase_binding_from_evidence(config, evidence)
	_atomic_json(path, payload)
	return path


def _load_phase_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig, *, phase: str
) -> Mapping[str, object]:
	path = _phase_evidence_path(config, phase)
	if not path.is_file():
		raise FileNotFoundError(f'center-trace {phase} evidence is missing: {path}')
	payload = _mapping(_json(path), f'center-trace {phase} evidence')
	expected_keys = {'artifact_type', 'schema_version', 'phase', 'status', 'evidence'}
	if phase == 'smoke':
		expected_keys.add('binding')
	if set(payload) != expected_keys:
		raise ValueError(f'center-trace {phase} evidence keys mismatch')
	if (
		payload['artifact_type']
			!= 'f3_center_trace_masked_pretraining_validation'
		or payload['schema_version'] != 1
		or payload['phase'] != phase
		or payload['status'] != 'PASS'
	):
		raise ValueError(f'center-trace {phase} evidence identity mismatch')
	evidence = _mapping(payload['evidence'], f'center-trace {phase} evidence body')
	_validate_real_data_inputs_evidence(evidence['real_data_inputs'])
	if phase == 'smoke':
		smoke = _mapping(evidence['smoke'], 'center-trace smoke evidence body')
		if (
			smoke.get('device') != 'cpu'
			or smoke.get('max_steps_override') != 2
			or smoke.get('global_step') != 2
			or smoke.get('schema_version') != 7
		):
				raise ValueError('center-trace smoke evidence does not prove CPU two-step run')
	return payload


def _smoke_phase_binding_from_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	evidence: Mapping[str, object],
) -> dict[str, object]:
	"""Build the immutable binding stored beside the smoke phase report."""
	target_manifest = _mapping(evidence['target_manifest'], 'smoke target manifest')
	per_head_target_hashes = evidence['per_head_target_hashes']
	smoke = _mapping(evidence['smoke'], 'smoke evidence')
	return _smoke_phase_binding(
		config,
		target_manifest=target_manifest,
		per_head_target_hashes=per_head_target_hashes,
		output_root=Path(str(smoke['root'])).resolve(),
		latest_path=Path(str(smoke['latest_path'])).resolve(),
		latest_sha256=smoke['latest_sha256'],
	)


def _smoke_phase_binding(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	target_manifest: Mapping[str, object],
	per_head_target_hashes: object,
	output_root: Path,
	latest_path: Path,
	latest_sha256: object,
) -> dict[str, object]:
	return {
		'target_manifest': dict(target_manifest),
		'per_head_target_hashes': per_head_target_hashes,
		'hard_full_config': _reference(config.hard_full_config),
		'hard_handoff': _reference(config.hard_handoff),
		'center_trace_masked_full_config': _reference(
			config.center_trace_masked_full_config
		),
		'center_trace_masked_smoke_config': _reference(
			config.center_trace_masked_smoke_config
		),
		'output_root': str(output_root),
		'latest_checkpoint': {
			'path': str(latest_path),
			'sha256': latest_sha256,
		},
	}


def _validate_smoke_phase_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	phase_evidence: Mapping[str, object],
	target_evidence: Mapping[str, object],
	full: Mapping[str, object],
	smoke: Mapping[str, object],
) -> None:
	"""Bind the persisted smoke report to the current complete-run inputs."""
	evidence = _mapping(phase_evidence['evidence'], 'center-trace smoke evidence')
	for key, expected in target_evidence.items():
		if evidence.get(key) != expected:
			raise ValueError(
				f'center-trace smoke evidence is stale for target/config field: {key}'
			)
	smoke_evidence = _mapping(evidence['smoke'], 'center-trace smoke evidence body')
	full_root = _training_output_root(full, label='full output root')
	smoke_root = _training_output_root(smoke, label='smoke output root')
	if full_root == smoke_root:
		raise ValueError('center-trace smoke output root must differ from full root')
	latest_path = smoke_root / 'latest.pt'
	if not latest_path.is_file():
		raise FileNotFoundError(
			f'center-trace smoke checkpoint is missing: {latest_path}'
		)
	latest_sha256 = file_sha256(latest_path)
	if (
		Path(str(smoke_evidence.get('root'))).resolve() != smoke_root
		or Path(str(smoke_evidence.get('latest_path'))).resolve() != latest_path
		or smoke_evidence.get('latest_sha256') != latest_sha256
	):
			raise ValueError(
			'center-trace smoke evidence does not bind the current output root/checkpoint'
		)
	candidate_runtime = _mapping(
		_mapping(
			target_evidence['hard_baseline_config_parity'],
			'current hard baseline config parity',
		)['candidate_runtime'],
		'current candidate runtime',
	)
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'initial_spatial_context_state_sha256',
	):
		if smoke_evidence.get(key) != candidate_runtime.get(key):
			raise ValueError(f'center-trace smoke evidence initialization is stale: {key}')
	expected_binding = _smoke_phase_binding(
		config,
		target_manifest=_mapping(
			target_evidence['target_manifest'], 'current target manifest'
		),
		per_head_target_hashes=target_evidence['per_head_target_hashes'],
		output_root=smoke_root,
		latest_path=latest_path,
		latest_sha256=latest_sha256,
	)
	if phase_evidence.get('binding') != expected_binding:
		raise ValueError('center-trace smoke evidence binding is stale')


def _execution_binding(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
) -> dict[str, dict[str, str]]:
	"""Bind execution-state snapshots to the immutable validation inputs."""
	return {
		'target_manifest': _reference(config.target_manifest),
		'hard_full_config': _reference(config.hard_full_config),
		'hard_handoff': _reference(config.hard_handoff),
		'center_trace_masked_smoke_config': _reference(
			config.center_trace_masked_smoke_config
		),
			'center_trace_masked_full_config': _reference(
				config.center_trace_masked_full_config
			),
			'center_trace_masked_embedding_config': _reference(
				config.center_trace_masked_embedding_config
			),
		}


def _start_execution_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	dry_run: bool,
) -> Mapping[str, object]:
	"""Capture the repository state before the smoke execution begins."""
	before = _execution_identity()
	_validate_execution_state(before, 'execution.before')
	record: dict[str, object] = {
		'artifact_type': _EXECUTION_ARTIFACT_TYPE,
		'schema_version': 1,
		'phase': 'inputs',
		'binding': _execution_binding(config),
		'execution': {'before': before, 'after': None},
	}
	if not dry_run:
		_atomic_json(_execution_evidence_path(config), record)
	return _mapping(record['execution'], 'execution evidence')


def _update_execution_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
	*,
	phase: str,
	dry_run: bool,
) -> Mapping[str, object]:
	"""Record the repository state after smoke or complete execution."""
	record = _load_execution_evidence(config)
	previous_phase = record['phase']
	if phase == 'smoke' and previous_phase != 'inputs':
		raise ValueError(
			'center-trace smoke execution evidence must follow inputs evidence'
		)
	if phase == 'complete' and previous_phase not in {'smoke', 'complete'}:
		raise ValueError(
			'center-trace complete execution evidence requires smoke evidence'
		)
	after = _execution_identity()
	_validate_execution_state(after, f'execution.{phase}.after')
	execution = dict(_mapping(record['execution'], 'execution evidence'))
	execution['after'] = after
	updated = dict(record)
	updated['phase'] = phase
	updated['execution'] = execution
	if not dry_run and updated != record:
		_atomic_json(_execution_evidence_path(config), updated)
	return _mapping(updated['execution'], 'execution evidence')


def _load_execution_evidence(
	config: F3CenterTraceMaskedPretrainingValidationConfig,
) -> Mapping[str, object]:
	"""Load and validate the before/after state for this exact experiment."""
	path = _execution_evidence_path(config)
	if not path.is_file():
		raise FileNotFoundError(f'center-trace execution evidence is missing: {path}')
	record = _mapping(_json(path), 'center-trace execution evidence')
	if set(record) != {
		'artifact_type',
		'schema_version',
		'phase',
		'binding',
		'execution',
	}:
		raise ValueError('center-trace execution evidence keys mismatch')
	if (
		record['artifact_type'] != _EXECUTION_ARTIFACT_TYPE
		or record['schema_version'] != 1
		or record['phase'] not in {'inputs', 'smoke', 'complete'}
	):
		raise ValueError('center-trace execution evidence identity mismatch')
	if record['binding'] != _execution_binding(config):
		raise ValueError('center-trace execution evidence binding mismatch')
	execution = _mapping(record['execution'], 'center-trace execution evidence')
	if set(execution) != {'before', 'after'}:
		raise ValueError('center-trace execution evidence state keys mismatch')
	_validate_execution_state(execution['before'], 'execution evidence.before')
	if record['phase'] != 'inputs':
		_validate_execution_state(execution['after'], 'execution evidence.after')
	elif execution['after'] is not None:
		raise ValueError('center-trace inputs execution evidence has an after state')
	return record


def _validate_execution_state(value: object, label: str) -> None:
	"""Validate one immutable Git state snapshot."""
	state = _mapping(value, label)
	if set(state) != {'git_commit', 'git_status_short', 'git_diff_sha256'}:
		raise ValueError(f'{label} keys mismatch')
	if not _valid_git_commit(state['git_commit']):
		raise ValueError(f'{label}.git_commit is missing or invalid')
	if not isinstance(state['git_status_short'], list) or any(
		not isinstance(item, str) for item in state['git_status_short']
	):
		raise TypeError(f'{label}.git_status_short must be string list')
	_require_sha256(state['git_diff_sha256'], f'{label}.git_diff_sha256')


def _execution_identity() -> dict[str, object]:
	root = Path(__file__).resolve().parents[3]
	commit = _git_output(root, 'rev-parse', 'HEAD')
	if not _valid_git_commit(commit):
		raise RuntimeError('unable to collect the current Git commit')
	status = _git_output(root, 'status', '--short')
	if status is None:
		raise RuntimeError('unable to collect the current Git status')
	diff = _git_bytes(root, 'diff', '--binary', 'HEAD')
	if diff is None:
		raise RuntimeError('unable to collect the current Git diff')
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


def _valid_git_commit(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) in {40, 64}
		and bool(value)
		and all(character in '0123456789abcdef' for character in value.lower())
	)


def _validate_target_file_references(target: Mapping[str, object]) -> None:
	for head_k in _HEAD_KS:
		head = _mapping(
			_mapping(target['heads'], 'target heads').get(str(head_k)),
			f'target K={head_k}',
		)
		for survey_id, entry_value in _mapping(
			head['surveys'], f'target K={head_k} surveys'
		).items():
			entry = _mapping(entry_value, f'target K={head_k} survey {survey_id}')
			if 'boundary_weight' in entry:
				raise ValueError(
					'center-trace target must preserve the source valid-token boundary semantics'
				)
			for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
				reference = _mapping(entry[name], f'target {name}')
				path = Path(str(reference['path']))
				if not path.is_file() or file_sha256(path) != reference['sha256']:
					raise ValueError(f'target {name} reference hash mismatch')


def _validate_target_hashes(value: object) -> None:
	hashes = _mapping(value, 'target head hashes')
	if set(hashes) != {'6', '8', '10'}:
		raise ValueError('target head hashes must contain K=6/8/10')
	for head_k, surveys_value in hashes.items():
		surveys = _mapping(surveys_value, f'target head hashes K={head_k}')
		if not surveys:
			raise ValueError(f'target head hashes K={head_k} are empty')
		for survey_id, artifacts_value in surveys.items():
			artifacts = _mapping(
				artifacts_value, f'target hashes K={head_k} survey {survey_id}'
			)
			if set(artifacts) != {'labels', 'confidence', 'valid_tokens', 'metadata'}:
				raise ValueError('target artifact hash keys mismatch')
			for name, digest in artifacts.items():
				_require_sha256(digest, f'target hash {head_k}/{survey_id}/{name}')


def _validate_hard_target_representation(
	training: Mapping[str, object],
	label: str,
) -> None:
	pseudo_targets = _mapping(training['pseudo_targets'], f'{label} pseudo_targets')
	# The existing hard baseline config omits this field; the shared runner's
	# resolved default is the immutable hard-label representation.
	representation = pseudo_targets.get(
		'target_representation', _TARGET_REPRESENTATION
	)
	if representation != _TARGET_REPRESENTATION:
		raise ValueError(
			f'{label} pseudo_targets.target_representation must be '
			f'{_TARGET_REPRESENTATION!r}'
		)


def _quarantine_smoke_output(path: Path) -> Path:
	"""Move invalid smoke evidence to a timestamped sibling for recovery."""
	target = path.with_name(
		f'{path.name}.quarantine.{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
	)
	if target.exists():
		raise FileExistsError(f'smoke quarantine path already exists: {target}')
	path.replace(target)
	return target


def _validate_canonical_valid_token_identities(value: object, expected: str) -> None:
	identities = _mapping(value, 'canonical valid-token identities')
	if set(identities) != {'mae', 'current_k6', 'mh_nocons'}:
		raise ValueError('canonical valid-token identity roles mismatch')
	for role, identity_value in identities.items():
		identity = _mapping(identity_value, f'canonical valid-token identity {role}')
		if not isinstance(identity.get('path'), str) or not identity['path']:
			raise TypeError(f'canonical valid-token identity {role} path is missing')
		if identity.get('sha256') != expected:
			raise ValueError(f'canonical valid-token identity {role} hash mismatch')


def _validate_reference(value: object, label: str) -> None:
	reference = _mapping(value, label)
	if set(reference) != {'path', 'sha256'}:
		raise ValueError(f'{label} keys mismatch')
	if not isinstance(reference['path'], str) or not reference['path']:
		raise TypeError(f'{label}.path is missing')
	_require_sha256(reference['sha256'], f'{label}.sha256')


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path.resolve()), 'sha256': file_sha256(path)}


def _scientific(training: Mapping[str, object], label: str) -> Mapping[str, object]:
	return _mapping(
		_mapping(training['identity'], f'{label} identity')['scientific_identity'],
		f'{label} scientific identity',
	)


def _manifest_path(training: Mapping[str, object]) -> Path:
	return Path(
		str(_mapping(training['pseudo_targets'], 'pseudo_targets')['manifest'])
	).resolve()


def _train_manifest_path(training: Mapping[str, object]) -> Path:
	return Path(
		str(_mapping(training['manifests'], 'manifests')['train'])
	).resolve()


def _model_tag(training: Mapping[str, object], label: str) -> str:
	value = _mapping(training['identity'], f'{label} identity').get('model_tag')
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} model tag is invalid')
	return value


def _checkpoint_kind(payload: Mapping[str, object]) -> str:
	value = _mapping(payload['training_state'], 'training state').get('checkpoint_kind')
	if value not in {'step', 'epoch'}:
		raise ValueError('checkpoint kind is invalid')
	return str(value)


def _torch_mapping(path: Path) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	return _mapping(payload, f'checkpoint {path}')


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _json(path: Path) -> object:
	return json.loads(path.read_text(encoding='utf-8'))


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
	temporary = Path(temporary_name)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as handle:
			json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
			handle.write('\n')
			handle.flush()
			os.fsync(handle.fileno())
		temporary.replace(path)
	except BaseException:
		temporary.unlink(missing_ok=True)
		raise


def _require_sha256(value: object, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value.lower())
	):
		raise TypeError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _positive_int(value: object) -> bool:
	return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_int(value: object) -> bool:
	return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_number(value: object) -> bool:
	return (
		not isinstance(value, bool)
		and isinstance(value, int | float)
		and math.isfinite(float(value))
	)


__all__ = [
	'F3CenterTraceMaskedPretrainingValidationConfig',
	'F3CenterTraceMaskedPretrainingValidationResult',
	'f3_center_trace_masked_pretraining_validation_config_from_mapping',
	'load_f3_center_trace_masked_pretraining_handoff',
	'load_f3_center_trace_masked_pretraining_validation_config',
	'validate_f3_center_trace_masked_pretraining',
]
