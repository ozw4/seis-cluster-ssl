"""Validate the schema-6 F3 unanimous XY-neighbour pretraining run.

This is intentionally a separate validator from the 3-of-4 successor.  It
accepts only the immutable unanimous target representation and schema-6
checkpoints, while reusing the established hard-label checkpoint-selection
checks.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.config.pretraining import (
	_multi_head_target_hashes,
	_xy_neighbor_unanimous_smoothing_identity,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.f3.xy_neighbor_unanimous_target_audit import (
	load_f3_xy_neighbor_unanimous_target_audit,
	replay_f3_xy_neighbor_unanimous_target_audit,
)
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	load_multi_head_xy_neighbor_unanimous_target_manifest,
)
from seis_ssl_cluster.training.strat_hmm.components import build_strat_hmm_components
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	scientific_identity_sha256,
	validate_stratigraphy_checkpoint_payload,
)

_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'experiment_root',
		'target_manifest',
		'target_audit',
		'hard_full_config',
		'xy_neighbor_unanimous_smoke_config',
		'xy_neighbor_unanimous_full_config',
	}
)
_TARGET_REPRESENTATION = 'xy_neighbor_unanimous_hard_labels_v1'
_TARGET_SEMANTICS = 'xy_neighbor_unanimous_outlier_correction_v1'
_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1'
_VARIANT = 'xyunanim1_nocons'
_HANDOFF_TYPE = 'f3_xy_neighbor_unanimous_pretraining_handoff'
_CONSISTENCY_POLICY = 'disabled_for_xy_neighbor_unanimous_v1'
_EMBEDDING_FIELDS = frozenset(
	{
		'method',
		'base_objective',
		'head_spec',
		'head_ks',
		'head_count',
		'unfreeze_top_blocks',
		'distillation_weight',
		'prototype_weight',
		'prototype_weight_semantics',
		'usage_weight',
		'usage_weight_semantics',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'model_tag',
		'scientific_identity_sha256',
		'checkpoint_stratigraphy_state_sha256',
		'target_representation',
		'target_semantics',
		'xy_neighbor_unanimous_target_manifest_path',
		'xy_neighbor_unanimous_target_manifest_sha256',
		'per_head_xy_neighbor_unanimous_target_sha256',
		'source_hard_manifest_sha256',
		'xy_neighbor_unanimous_smoothing',
	}
)


@dataclass(frozen=True)
class F3XYNeighborUnanimousPretrainingValidationConfig:
	"""Closed inputs needed to validate the unanimous pretraining contract."""

	artifact_root: Path
	experiment_root: Path
	target_manifest: Path
	target_audit: Path
	hard_full_config: Path
	xy_neighbor_unanimous_smoke_config: Path
	xy_neighbor_unanimous_full_config: Path


@dataclass(frozen=True)
class F3XYNeighborUnanimousPretrainingValidationResult:
	"""Evidence for one validation phase and an optional final handoff."""

	phase: str
	evidence: Mapping[str, object]
	published_handoff: Path | None


def f3_xy_neighbor_unanimous_pretraining_validation_config_from_mapping(
	config: Mapping[str, object],
) -> F3XYNeighborUnanimousPretrainingValidationConfig:
	"""Resolve the intentionally non-extensible unanimous validation schema."""
	if not isinstance(config, Mapping):
		raise TypeError('unanimous pretraining validation config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown unanimous validation keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing unanimous validation keys: {sorted(missing)!r}')

	def path(name: str, *, must_exist: bool) -> Path:
		value = config[name]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{name} must be a non-empty path string')
		result = Path(value).resolve()
		if must_exist and not result.is_file():
			raise FileNotFoundError(f'{name} is missing: {result}')
		return result

	result = F3XYNeighborUnanimousPretrainingValidationConfig(
		artifact_root=path('artifact_root', must_exist=False),
		experiment_root=path('experiment_root', must_exist=False),
		target_manifest=path('target_manifest', must_exist=True),
		target_audit=path('target_audit', must_exist=True),
		hard_full_config=path('hard_full_config', must_exist=True),
		xy_neighbor_unanimous_smoke_config=path(
			'xy_neighbor_unanimous_smoke_config', must_exist=True
		),
		xy_neighbor_unanimous_full_config=path(
			'xy_neighbor_unanimous_full_config', must_exist=True
		),
	)
	if not result.artifact_root.is_dir() or not result.experiment_root.is_dir():
		raise FileNotFoundError('artifact_root and experiment_root must be directories')
	ensure_under_root(
		result.experiment_root, root=result.artifact_root, label='experiment_root'
	)
	for name, value in (
		('target_manifest', result.target_manifest),
		('target_audit', result.target_audit),
	):
		ensure_under_root(value, root=result.artifact_root, label=name)
	return result


def load_f3_xy_neighbor_unanimous_pretraining_validation_config(
	path: str | Path,
) -> F3XYNeighborUnanimousPretrainingValidationConfig:
	"""Load the unanimous validation configuration from YAML."""
	return f3_xy_neighbor_unanimous_pretraining_validation_config_from_mapping(
		load_config(path)
	)


def validate_f3_xy_neighbor_unanimous_pretraining(
	config: F3XYNeighborUnanimousPretrainingValidationConfig,
	*,
	phase: str,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3XYNeighborUnanimousPretrainingValidationResult:
	"""Validate targets, smoke, checkpoints, or the complete publication."""
	if phase not in {'targets', 'smoke', 'checkpoints', 'complete'}:
		raise ValueError('phase must be targets, smoke, checkpoints, or complete')
	try:
		target = load_multi_head_xy_neighbor_unanimous_target_manifest(
			config.target_manifest
		)
		audit = load_f3_xy_neighbor_unanimous_target_audit(config.target_audit)
		audit = replay_f3_xy_neighbor_unanimous_target_audit(
			config.target_audit,
			artifact_root=config.artifact_root,
		)
		full = _training_config(config.xy_neighbor_unanimous_full_config)
		hard = _training_config(
			config.hard_full_config,
			artifact_root=config.artifact_root,
		)
		target_evidence = _target_evidence(
			config, target=target, audit=audit, full=full, hard=hard
		)
		if phase == 'targets':
			return F3XYNeighborUnanimousPretrainingValidationResult(
				phase, {'status': 'PASS', **target_evidence}, None
			)

		smoke = _training_config(config.xy_neighbor_unanimous_smoke_config)
		_smoke_config_contract(config, full=full, smoke=smoke)
		if phase == 'smoke':
			smoke_evidence = _checkpoint_evidence(
				smoke, expected_global_step=2, require_best=False
			)
			_validate_initial_state_parity(
				smoke_evidence,
				hard,
				runtime_parity=_mapping(
					target_evidence['hard_baseline_config_parity'],
					'hard baseline parity',
				),
			)
			return F3XYNeighborUnanimousPretrainingValidationResult(
				phase,
				{'status': 'PASS', **target_evidence, 'smoke': smoke_evidence},
				None,
			)

		checkpoint = _checkpoint_evidence(
			full, expected_global_step=25600, require_best=True
		)
		_validate_initial_state_parity(
			checkpoint,
			hard,
			runtime_parity=_mapping(
				target_evidence['hard_baseline_config_parity'],
				'hard baseline parity',
			),
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
					/ 'xy_neighbor_unanimous_checkpoint_validation.json',
					{
						'artifact_type': 'f3_xy_neighbor_unanimous_validation',
						'schema_version': 1,
						'phase': phase,
						'status': 'PASS',
						'target_manifest_sha256': _mapping(
							target_evidence['target_manifest'], 'target manifest'
						)['sha256'],
					},
				)
			return F3XYNeighborUnanimousPretrainingValidationResult(
				phase, evidence, None
			)

		evidence['embedding'] = _embedding_evidence(config, checkpoint, training=full)
		handoff = _handoff(evidence)
		handoff_path = (
			Path(str(checkpoint['root']))
			/ 'preflight'
			/ 'xy_neighbor_unanimous_handoff.json'
		)
		if dry_run:
			return F3XYNeighborUnanimousPretrainingValidationResult(
				phase, evidence, None
			)
		published = _publish_handoff(
			handoff_path,
			handoff,
			only_missing=only_missing,
			quarantine_invalid=quarantine_invalid,
		)
		return F3XYNeighborUnanimousPretrainingValidationResult(
			phase, evidence, handoff_path if published else None
		)
	except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
		if not dry_run:
			raise
		return F3XYNeighborUnanimousPretrainingValidationResult(
			phase,
			{'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'},
			None,
		)


def load_f3_xy_neighbor_unanimous_pretraining_handoff(  # noqa: C901, PLR0912
	path: str | Path,
) -> Mapping[str, object]:
	"""Load a complete PASS handoff without accepting predecessor identities."""
	payload = _mapping(_json(Path(path)), 'unanimous pretraining handoff')
	if set(payload) != {
		'artifact_type',
		'schema_version',
		'status',
		'model_tag',
		'variant',
		'targets',
		'checkpoint',
		'embedding',
	}:
		raise ValueError('unanimous pretraining handoff keys mismatch')
	if (
		payload.get('artifact_type') != _HANDOFF_TYPE
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
		or payload.get('model_tag') != _MODEL_TAG
		or payload.get('variant') != _VARIANT
	):
		raise ValueError('unanimous pretraining handoff identity mismatch')
	targets = _mapping(payload['targets'], 'unanimous handoff targets')
	if set(targets) != {
		'target_representation',
		'target_semantics',
		'consistency_policy',
		'target_manifest',
		'target_audit',
		'xy_neighbor_unanimous_target_head_hashes',
		'source_hard_manifest',
		'xy_neighbor_unanimous_smoothing',
		'temporal_transition_counts',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	}:
		raise ValueError('unanimous pretraining handoff target keys mismatch')
	if (
		targets.get('target_representation') != _TARGET_REPRESENTATION
		or targets.get('target_semantics') != _TARGET_SEMANTICS
		or targets.get('consistency_policy') != _CONSISTENCY_POLICY
	):
		raise ValueError('unanimous pretraining handoff target contract mismatch')
	for key in ('target_manifest', 'target_audit', 'source_hard_manifest'):
		_reference(targets[key], f'unanimous handoff {key}')
	_validate_head_hashes(targets['xy_neighbor_unanimous_target_head_hashes'])
	_validate_transition_counts(targets['temporal_transition_counts'])
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if not _sha256(targets.get(key)):
			raise TypeError(f'unanimous handoff {key} is invalid')
	checkpoint = _mapping(payload['checkpoint'], 'unanimous handoff checkpoint')
	if set(checkpoint) != {
		'path',
		'sha256',
		'selected_checkpoint_kind',
		'selected_epoch',
		'selected_global_step',
		'selected_loss',
	}:
		raise ValueError('unanimous pretraining handoff checkpoint keys mismatch')
	if not _sha256(checkpoint.get('sha256')):
		raise TypeError('unanimous handoff checkpoint SHA-256 is invalid')
	if checkpoint.get('selected_checkpoint_kind') not in {'step', 'epoch'}:
		raise ValueError('unanimous handoff checkpoint kind is invalid')
	for key in ('selected_epoch', 'selected_global_step'):
		if not _positive_int(checkpoint.get(key)):
			raise TypeError(f'unanimous handoff checkpoint {key} is invalid')
	if not _finite_number(checkpoint.get('selected_loss')):
		raise TypeError('unanimous handoff selected loss is invalid')
	embedding = _mapping(payload['embedding'], 'unanimous handoff embedding')
	if set(embedding) != {
		'root',
		'metadata_path',
		'metadata_sha256',
		'embeddings_sha256',
		'valid_tokens_sha256',
		'valid_token_count',
	}:
		raise ValueError('unanimous pretraining handoff embedding keys mismatch')
	if not _positive_int(embedding.get('valid_token_count')):
		raise TypeError('unanimous handoff valid token count is invalid')
	for key in ('metadata_sha256', 'embeddings_sha256', 'valid_tokens_sha256'):
		if not _sha256(embedding.get(key)):
			raise TypeError(f'unanimous handoff {key} is invalid')
	return payload


def _target_evidence(  # noqa: C901, PLR0912
	config: F3XYNeighborUnanimousPretrainingValidationConfig,
	*,
	target: Mapping[str, object],
	audit: Mapping[str, object],
	full: Mapping[str, object],
	hard: Mapping[str, object],
) -> dict[str, object]:
	if target.get('head_ks') != [6, 8, 10]:
		raise ValueError('unanimous target K identity mismatch')
	if (
		target.get('target_representation') != _TARGET_REPRESENTATION
		or target.get('target_semantics') != _TARGET_SEMANTICS
	):
		raise ValueError('unanimous target representation or semantics mismatch')
	if audit.get('status') != 'XYUNANIM_TARGET_GO':
		raise ValueError('unanimous target audit must be XYUNANIM_TARGET_GO')
	if audit.get('xy_neighbor_unanimous_target_manifest') != _identity(
		config.target_manifest
	):
		raise ValueError('unanimous target audit target identity mismatch')
	if audit.get('source_hard_manifest') != target.get('source_hard_manifest'):
		raise ValueError('unanimous target audit source identity mismatch')
	identity = _training_identity(full, 'full')
	if _manifest_path(full) != config.target_manifest:
		raise ValueError('full training target manifest path mismatch')
	if _model_tag(full, 'full') != _MODEL_TAG:
		raise ValueError('full training model tag mismatch')
	if (
		Path(str(_mapping(full['paths'], 'full paths')['output_root'])).resolve()
		!= (config.experiment_root / _MODEL_TAG).resolve()
	):
		raise ValueError('full training output root mismatch')
	if identity.get('target_representation') != _TARGET_REPRESENTATION:
		raise ValueError('full training target representation mismatch')
	if identity.get('target_semantics') != _TARGET_SEMANTICS:
		raise ValueError('full training target semantics mismatch')
	if identity.get('consistency_policy') != _CONSISTENCY_POLICY:
		raise ValueError('full training consistency policy mismatch')
	if _mapping(full['loss'], 'full loss').get('consistency_weight') != 0.0:
		raise ValueError('unanimous consistency must be disabled')
	if identity.get('supervised_loss') != 'structured_hmm_hard_categorical_v1':
		raise ValueError('unanimous training must use the hard categorical loss')
	if _mapping(full['pseudo_targets'], 'pseudo targets').get('min_confidence') != 0.0:
		raise ValueError('unanimous min_confidence must be zero')
	manifest_sha256 = file_sha256(config.target_manifest)
	if identity.get('xy_neighbor_unanimous_target_manifest_sha256') != manifest_sha256:
		raise ValueError('unanimous target manifest SHA-256 mismatch')
	if identity.get(
		'xy_neighbor_unanimous_target_head_hashes'
	) != _multi_head_target_hashes(target):
		raise ValueError('unanimous target head hashes mismatch')
	source = _reference(target['source_hard_manifest'], 'source hard manifest')
	if identity.get('source_hard_manifest_sha256') != source['sha256']:
		raise ValueError('unanimous source hard manifest SHA-256 mismatch')
	if identity.get('xy_neighbor_unanimous_smoothing') != (
		_xy_neighbor_unanimous_smoothing_identity(target)
	):
		raise ValueError('unanimous smoothing policy identity mismatch')
	for forbidden in (
		'source_posterior_manifest_sha256',
		'lateral_smoothing',
		'xy_neighbor_consensus_target_manifest_sha256',
	):
		if forbidden in identity:
			raise ValueError(f'unanimous training must not carry {forbidden}')
	parity = _hard_config_parity(full, hard)
	return {
		'target_representation': _TARGET_REPRESENTATION,
		'target_semantics': _TARGET_SEMANTICS,
		'consistency_policy': _CONSISTENCY_POLICY,
		'target_manifest': _identity(config.target_manifest),
		'target_audit': _identity(config.target_audit),
		'xy_neighbor_unanimous_target_head_hashes': _multi_head_target_hashes(target),
		'source_hard_manifest': source,
		'xy_neighbor_unanimous_smoothing': _xy_neighbor_unanimous_smoothing_identity(
			target
		),
		'temporal_transition_counts': _target_temporal_transition_counts(target),
		'hard_baseline_config_parity': parity,
	}


def _hard_config_parity(
	full: Mapping[str, object], hard: Mapping[str, object]
) -> dict[str, object]:
	"""Prove all non-lineage training settings equal the hard baseline."""
	left = json.loads(json.dumps(full))
	right = json.loads(json.dumps(hard))
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		identity = _mapping(value['identity'], 'identity')
		identity.pop('model_tag', None)
		scientific = _mapping(identity['scientific_identity'], 'scientific identity')
		for key in (
			'experiment_role',
			'variant',
			'target_representation',
			'target_semantics',
			'target_manifest_sha256',
			'target_head_hashes',
			'xy_neighbor_unanimous_target_manifest_sha256',
			'xy_neighbor_unanimous_target_head_hashes',
			'source_hard_manifest_sha256',
			'xy_neighbor_unanimous_smoothing',
			'consistency_policy',
			'supervised_loss',
			'consistency_weight',
		):
			scientific.pop(key, None)
		pseudo = _mapping(value['pseudo_targets'], 'pseudo targets')
		pseudo.pop('manifest', None)
		pseudo.pop('target_representation', None)
	if left != right:
		raise ValueError('unanimous/full hard baseline config parity differs')
	hard_runtime = _runtime_contract(hard)
	candidate_runtime = _runtime_contract(full)
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if hard_runtime[key] != candidate_runtime[key]:
			raise ValueError(f'hard/candidate runtime parity differs: {key}')
	for key in ('trainability_summary', 'optimizer_group_identity'):
		if hard_runtime[key] != candidate_runtime[key]:
			raise ValueError(f'hard/candidate runtime parity differs: {key}')
	return {
		'status': 'PASS',
		'allowed_differences': (
			'paths.output_root',
			'identity.model_tag',
			'identity.scientific_identity lineage fields',
			'pseudo_targets.manifest',
			'pseudo_targets.target_representation',
		),
		'hard_runtime': hard_runtime,
		'candidate_runtime': candidate_runtime,
	}


def _runtime_contract(training: Mapping[str, object]) -> dict[str, object]:
	"""Capture the hard-route initialization and optimizer contract on CPU."""
	train = _mapping(training['train'], 'training train')
	seed = train.get('seed')
	if isinstance(seed, bool) or not isinstance(seed, int):
		raise TypeError('training seed must be an integer')
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(seed)
		components = build_strat_hmm_components(training, device='cpu')
	heads = getattr(components, 'heads', None)
	if not isinstance(heads, torch.nn.Module):
		raise TypeError('unanimous runtime contract requires multi-head components')
	parameter_names = {
		id(parameter): f'{prefix}.{name}'
		for prefix, module in (('student', components.student), ('head', heads))
		for name, parameter in module.named_parameters()
	}
	groups: list[dict[str, object]] = []
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
	return {
		'initial_student_state_sha256': hard_validation._state_sha256(  # noqa: SLF001
			components.student.state_dict()
		),
		'initial_head_state_sha256': hard_validation._state_sha256(  # noqa: SLF001
			heads.state_dict()
		),
		'trainability_summary': {
			'trainable_parameter_count': int(summary.trainable_parameter_count),
			'frozen_parameter_count': int(summary.frozen_parameter_count),
			'trainable_names': list(summary.trainable_names),
		},
		'optimizer_group_identity': groups,
	}


def _smoke_config_contract(
	config: F3XYNeighborUnanimousPretrainingValidationConfig,
	*,
	full: Mapping[str, object],
	smoke: Mapping[str, object],
) -> None:
	if _manifest_path(smoke) != config.target_manifest:
		raise ValueError('smoke target manifest path mismatch')
	if _model_tag(smoke, 'smoke') != _MODEL_TAG:
		raise ValueError('smoke model tag mismatch')
	if _mapping(smoke['train'], 'smoke train').get('device') != 'cpu':
		raise ValueError('smoke device must be cpu')
	if _mapping(smoke['train'], 'smoke train').get('max_steps') != 2:
		raise ValueError('smoke max_steps must be exactly two')
	if _mapping(smoke['loss'], 'smoke loss').get('consistency_weight') != 0.0:
		raise ValueError('smoke consistency must be disabled')
	if (
		Path(str(_mapping(smoke['paths'], 'smoke paths')['output_root'])).resolve()
		== Path(str(_mapping(full['paths'], 'full paths')['output_root'])).resolve()
	):
		raise ValueError('smoke and full output roots must differ')
	left, right = json.loads(json.dumps(full)), json.loads(json.dumps(smoke))
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		identity = _mapping(value['identity'], 'identity')
		runtime = identity.get('runtime_identity')
		if isinstance(runtime, Mapping):
			_mapping(runtime, 'runtime identity').pop('device', None)
		train = _mapping(value['train'], 'train')
		train.pop('device', None)
		train.pop('max_steps', None)
		_mapping(
			_mapping(identity['scientific_identity'], 'scientific identity')['train'],
			'scientific train identity',
		).pop('max_steps', None)
	if left != right:
		raise ValueError(
			'smoke/full unanimous config drift outside CPU two-step settings'
		)


def _checkpoint_evidence(
	training: Mapping[str, object], *, expected_global_step: int, require_best: bool
) -> dict[str, object]:
	root = Path(str(_mapping(training['paths'], 'paths')['output_root']))
	latest_path, best_path = root / 'latest.pt', root / 'best.pt'
	if not latest_path.is_file() or (require_best and not best_path.is_file()):
		raise FileNotFoundError('required unanimous checkpoint is missing')
	latest = _checkpoint(latest_path, expected_config=training)
	best = (
		_checkpoint(best_path, expected_config=training)
		if best_path.is_file()
		else None
	)
	for payload in (latest, best):
		if payload is not None:
			hard_validation._metrics_finite(payload)  # noqa: SLF001
			_validate_hard_label_metrics(payload)
	if latest.get('global_step') != expected_global_step:
		raise ValueError('unanimous checkpoint global step mismatch')
	if require_best:
		if latest.get('epoch') != 25 or _checkpoint_kind(latest) != 'epoch':
			raise ValueError('unanimous full run must end at epoch 25')
		if best is None:
			raise AssertionError('full unanimous validation requires best.pt')
		rows = hard_validation._epoch_rows(root / 'multi_head_epoch_metrics.csv')  # noqa: SLF001
		if [row['epoch'] for row in rows] != list(range(1, 26)) or rows[-1][
			'global_step'
		] != expected_global_step:
			raise ValueError('unanimous epoch metrics are incomplete')
		selection = hard_validation._validate_best_selection(  # noqa: SLF001
			best, latest, variant=_VARIANT
		)
		selected_path, selected = best_path, best
	else:
		if latest.get('epoch') != 1 or _checkpoint_kind(latest) != 'step':
			raise ValueError('unanimous smoke must end with a step checkpoint')
		rows, selection = [], None
		selected_path, selected = latest_path, latest
	identity = _mapping(selected['stratigraphy_checkpoint'], 'checkpoint identity')
	_scientific_checkpoint_contract(training, identity)
	return {
		'root': str(root),
		'latest_path': str(latest_path),
		'latest_sha256': file_sha256(latest_path),
		'best_path': None if best is None else str(best_path),
		'best_sha256': None if best is None else file_sha256(best_path),
		'selected_path': str(selected_path),
		'selected_sha256': file_sha256(selected_path),
		'selected_checkpoint_kind': _checkpoint_kind(selected),
		'selected_epoch': selected['epoch'],
		'selected_global_step': selected['global_step'],
		'selected_loss': _mapping(selected['metrics'], 'metrics')['loss'],
		'initial_student_state_sha256': identity['initial_student_state_sha256'],
		'initial_head_state_sha256': identity['initial_head_state_sha256'],
		'trainability_summary': _mapping(
			selected['trainability_summary'], 'checkpoint trainability summary'
		),
		'identity': identity,
		'epoch_rows': rows,
		'checkpoint_selection': selection,
	}


def _checkpoint(
	path: Path, *, expected_config: Mapping[str, object]
) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError(f'checkpoint must be a mapping: {path}')
	validate_stratigraphy_checkpoint_payload(payload, expected_config=expected_config)
	if (
		_mapping(payload.get('stratigraphy_config'), 'checkpoint config')
		!= expected_config
	):
		raise ValueError('unanimous checkpoint config differs from resolved config')
	_scientific_checkpoint_contract(
		expected_config,
		_mapping(payload.get('stratigraphy_checkpoint'), 'checkpoint identity'),
	)
	return payload


def _scientific_checkpoint_contract(
	training: Mapping[str, object], identity: Mapping[str, object]
) -> None:
	if identity.get('schema_version') != 6:
		raise ValueError('unanimous checkpoint requires schema_version 6')
	scientific = _training_identity(training, 'checkpoint')
	for key in (
		'target_representation',
		'target_semantics',
		'xy_neighbor_unanimous_target_manifest_sha256',
		'source_hard_manifest_sha256',
		'xy_neighbor_unanimous_smoothing',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
	):
		if identity.get(key) != scientific.get(key):
			raise ValueError(f'unanimous checkpoint identity mismatch: {key}')
	if identity.get('per_head_xy_neighbor_unanimous_targets') != scientific.get(
		'xy_neighbor_unanimous_target_head_hashes'
	):
		raise ValueError('unanimous checkpoint target head hashes mismatch')
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('unanimous checkpoint scientific identity SHA mismatch')


def _validate_hard_label_metrics(payload: Mapping[str, object]) -> None:
	metrics = _mapping(payload.get('metrics'), 'checkpoint metrics')
	if 'loss_consistency_contribution' not in metrics or any(
		'posterior' in str(key) for key in metrics
	):
		raise ValueError('unanimous checkpoint did not use only the hard-label route')
	consistency = metrics['loss_consistency_contribution']
	if (
		isinstance(consistency, bool)
		or not isinstance(consistency, int | float)
		or not math.isfinite(float(consistency))
		or float(consistency) != 0.0
	):
		raise ValueError('unanimous checkpoint consistency contribution is not zero')
	if (
		_mapping(payload['stratigraphy_checkpoint'], 'checkpoint identity').get(
			'consistency_weight'
		)
		!= 0.0
	):
		raise ValueError('unanimous checkpoint consistency contribution is not zero')


def _validate_initial_state_parity(  # noqa: C901
	checkpoint: Mapping[str, object],
	hard: Mapping[str, object],
	*,
	runtime_parity: Mapping[str, object],
) -> None:
	hard_runtime = _mapping(runtime_parity['hard_runtime'], 'hard runtime contract')
	candidate_runtime = _mapping(
		runtime_parity['candidate_runtime'], 'candidate runtime contract'
	)
	hard_root = Path(str(_mapping(hard['paths'], 'hard paths')['output_root']))
	hard_path = hard_root / 'best.pt'
	if not hard_path.is_file():
		raise FileNotFoundError(
			f'hard baseline best checkpoint is missing: {hard_path}'
		)
	hard_payload = torch.load(hard_path, map_location='cpu', weights_only=False)
	if not isinstance(hard_payload, Mapping):
		raise TypeError('hard baseline checkpoint must be a mapping')
	validate_stratigraphy_checkpoint_payload(hard_payload, expected_config=hard)
	if _mapping(hard_payload['stratigraphy_config'], 'hard checkpoint config') != hard:
		raise ValueError('hard baseline checkpoint config differs from resolved config')
	hard_identity = _mapping(hard_payload['stratigraphy_checkpoint'], 'hard identity')
	candidate_identity = _mapping(checkpoint['identity'], 'candidate identity')
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if hard_identity.get(key) != hard_runtime.get(key):
			raise ValueError(f'hard baseline runtime contract differs: {key}')
		if candidate_identity.get(key) != candidate_runtime.get(key):
			raise ValueError(f'unanimous runtime contract differs: {key}')
		if checkpoint.get(key) != hard_identity.get(key):
			raise ValueError(
				f'unanimous hard-baseline initialization parity differs: {key}'
			)
	if (
		_mapping(
			hard_payload['trainability_summary'], 'hard checkpoint trainability summary'
		)
		!= hard_runtime['trainability_summary']
	):
		raise ValueError('hard baseline checkpoint trainability contract differs')
	if (
		checkpoint.get('trainability_summary')
		!= candidate_runtime['trainability_summary']
	):
		raise ValueError('unanimous checkpoint trainability contract differs')
	if (
		hard_identity.get('optimizer_group_identity')
		!= hard_runtime['optimizer_group_identity']
	):
		raise ValueError('hard baseline checkpoint optimizer contract differs')
	if (
		candidate_identity.get('optimizer_group_identity')
		!= candidate_runtime['optimizer_group_identity']
	):
		raise ValueError('unanimous checkpoint optimizer contract differs')


def _embedding_evidence(
	config: F3XYNeighborUnanimousPretrainingValidationConfig,
	checkpoint: Mapping[str, object],
	*,
	training: Mapping[str, object],
) -> dict[str, object]:
	root = (
		config.artifact_root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ _MODEL_TAG
		/ 'overlap_x16'
	)
	files = output_paths(root, 'f3_facies_benchmark')
	if not all(
		path.is_file()
		for path in (files.embeddings, files.valid_tokens, files.metadata)
	):
		raise FileNotFoundError('unanimous embedding artifacts are incomplete')
	metadata = _mapping(_json(files.metadata), 'embedding metadata')
	selected = Path(str(checkpoint['selected_path']))
	if Path(str(metadata.get('checkpoint_path', ''))).resolve() != selected.resolve():
		raise ValueError('unanimous embedding checkpoint path mismatch')
	if metadata.get('checkpoint_sha256') != file_sha256(selected):
		raise ValueError('unanimous embedding checkpoint SHA mismatch')
	_validate_embedding_identity(
		metadata,
		_mapping(checkpoint['identity'], 'identity'),
		training=training,
	)
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
		raise ValueError('unanimous embedding array contract mismatch')
	return {
		'root': str(root),
		'metadata_path': str(files.metadata),
		'metadata_sha256': file_sha256(files.metadata),
		'embeddings_sha256': file_sha256(files.embeddings),
		'valid_tokens_sha256': file_sha256(files.valid_tokens),
		'valid_token_count': int(valid.sum()),
	}


def _validate_embedding_identity(
	metadata: Mapping[str, object],
	identity: Mapping[str, object],
	*,
	training: Mapping[str, object],
) -> None:
	stratigraphy = _mapping(metadata.get('stratigraphy_pretext'), 'embedding identity')
	if set(stratigraphy) != _EMBEDDING_FIELDS:
		raise ValueError('unanimous embedding stratigraphy fields mismatch')
	head = _mapping(training['head'], 'embedding training head')
	student = _mapping(training['student'], 'embedding training student')
	loss = _mapping(training['loss'], 'embedding training loss')
	head_ks = identity.get('head_ks')
	if not isinstance(head_ks, list) or head_ks != [6, 8, 10]:
		raise ValueError('unanimous checkpoint head K identity is invalid')
	for metadata_key, expected in (
		('method', 'strat_hmm_multi_head_pretext'),
		('base_objective', 'amp_mae3d'),
		('head_spec', identity.get('head_spec')),
		('head_ks', head_ks),
		('head_count', len(head_ks)),
		('unfreeze_top_blocks', student.get('unfreeze_top_blocks')),
		('distillation_weight', loss.get('distillation_weight')),
		('prototype_weight', loss.get('prototype_weight')),
		('prototype_weight_semantics', 'mean_across_heads'),
		('usage_weight', loss.get('usage_weight')),
		('usage_weight_semantics', 'mean_across_heads'),
		('consistency_policy', identity.get('consistency_policy')),
		('consistency_weight', identity.get('consistency_weight')),
		('consistency_beta', identity.get('consistency_beta')),
	):
		if stratigraphy.get(metadata_key) != expected:
			raise ValueError(f'unanimous embedding identity mismatch: {metadata_key}')
	if head.get('spec') != identity.get('head_spec'):
		raise ValueError('unanimous checkpoint head spec differs from training')
	manifest = _mapping(
		identity.get('xy_neighbor_unanimous_target_manifest'), 'target manifest'
	)
	for metadata_key, identity_key in (
		('model_tag', 'model_tag'),
		('target_representation', 'target_representation'),
		('target_semantics', 'target_semantics'),
		(
			'xy_neighbor_unanimous_target_manifest_sha256',
			'xy_neighbor_unanimous_target_manifest_sha256',
		),
		(
			'per_head_xy_neighbor_unanimous_target_sha256',
			'per_head_xy_neighbor_unanimous_targets',
		),
		('source_hard_manifest_sha256', 'source_hard_manifest_sha256'),
		('xy_neighbor_unanimous_smoothing', 'xy_neighbor_unanimous_smoothing'),
		('scientific_identity_sha256', 'scientific_identity_sha256'),
	):
		if stratigraphy.get(metadata_key) != identity.get(identity_key):
			raise ValueError(f'unanimous embedding identity mismatch: {metadata_key}')
	if stratigraphy.get('xy_neighbor_unanimous_target_manifest_path') != manifest.get(
		'path'
	):
		raise ValueError('unanimous embedding target manifest path mismatch')
	if stratigraphy.get('checkpoint_stratigraphy_state_sha256') != identity.get(
		'stratigraphy_state_sha256'
	):
		raise ValueError('unanimous embedding stratigraphy state mismatch')


def _target_temporal_transition_counts(
	target: Mapping[str, object],
) -> dict[str, dict[str, int]]:
	result: dict[str, dict[str, int]] = {}
	for k in (6, 8, 10):
		head = _mapping(_mapping(target['heads'], 'heads')[str(k)], f'head K={k}')
		diagnostics = _mapping(head['diagnostics'], f'diagnostics K={k}')
		aggregate = _mapping(diagnostics['aggregate'], f'aggregate K={k}')
		result[str(k)] = _transition_pair(aggregate['temporal_transition_counts'])
	return result


def _transition_pair(value: object) -> dict[str, int]:
	counts = _mapping(value, 'temporal transition counts')
	if set(counts) != {'source', 'output'}:
		raise ValueError('temporal transition counts keys mismatch')
	result = {}
	for key in ('source', 'output'):
		if (
			isinstance(counts[key], bool)
			or not isinstance(counts[key], int)
			or counts[key] < 0
		):
			raise TypeError(f'temporal transition {key} is invalid')
		result[key] = counts[key]
	return result


def _validate_transition_counts(value: object) -> None:
	counts = _mapping(value, 'handoff temporal transitions')
	if set(counts) != {'6', '8', '10'}:
		raise ValueError('handoff temporal transitions head keys mismatch')
	for pair in counts.values():
		_transition_pair(pair)


def _validate_head_hashes(value: object) -> None:
	heads = _mapping(value, 'handoff target head hashes')
	if set(heads) != {'6', '8', '10'}:
		raise ValueError('handoff target head hashes must contain K=6/8/10')
	for surveys in heads.values():
		for artifacts in _mapping(surveys, 'target head surveys').values():
			mapping = _mapping(artifacts, 'target artifacts')
			if set(mapping) != {'labels', 'confidence', 'valid_tokens', 'metadata'}:
				raise ValueError('target artifact hash keys mismatch')
			if not all(_sha256(item) for item in mapping.values()):
				raise TypeError('target artifact digest is invalid')


def _handoff(evidence: Mapping[str, object]) -> dict[str, object]:
	target = _mapping(evidence['target_manifest'], 'target evidence')
	return {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': _MODEL_TAG,
		'variant': _VARIANT,
		'targets': {
			'target_representation': evidence['target_representation'],
			'target_semantics': evidence['target_semantics'],
			'consistency_policy': evidence['consistency_policy'],
			'target_manifest': target,
			'target_audit': evidence['target_audit'],
			'xy_neighbor_unanimous_target_head_hashes': evidence[
				'xy_neighbor_unanimous_target_head_hashes'
			],
			'source_hard_manifest': evidence['source_hard_manifest'],
			'xy_neighbor_unanimous_smoothing': evidence[
				'xy_neighbor_unanimous_smoothing'
			],
			'temporal_transition_counts': evidence['temporal_transition_counts'],
			'initial_student_state_sha256': evidence['initial_student_state_sha256'],
			'initial_head_state_sha256': evidence['initial_head_state_sha256'],
		},
		'checkpoint': {
			'path': evidence['selected_path'],
			'sha256': evidence['selected_sha256'],
			'selected_checkpoint_kind': evidence['selected_checkpoint_kind'],
			'selected_epoch': evidence['selected_epoch'],
			'selected_global_step': evidence['selected_global_step'],
			'selected_loss': evidence['selected_loss'],
		},
		'embedding': evidence['embedding'],
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
			existing = load_f3_xy_neighbor_unanimous_pretraining_handoff(path)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == handoff:
			if only_missing:
				return False
			raise FileExistsError(f'unanimous handoff already exists: {path}')
		if not quarantine_invalid:
			raise ValueError(
				'existing unanimous handoff is stale; use --quarantine-invalid'
			)
		path.replace(path.with_name(f'{path.name}.quarantine'))
	_atomic_json(path, handoff)
	return True


def _training_config(
	path: Path,
	*,
	artifact_root: Path | None = None,
) -> Mapping[str, object]:
	"""Resolve one config, deriving only the frozen hard-baseline digest if absent."""
	try:
		return resolve_strat_hmm_pretext_config(load_config(path))
	except ValueError as error:
		variable = 'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256'
		if artifact_root is None or variable not in str(error):
			raise
		manifest = (
			artifact_root
			/ 'pseudo_targets/f3/facies_benchmark_v1'
			/ 'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1'
			/ 'multi_head_target_manifest.json'
		)
		if not manifest.is_file():
			raise FileNotFoundError(
				'frozen hard target manifest is missing for baseline parity: {manifest}'
			) from error
		previous = os.environ.get(variable)
		os.environ[variable] = file_sha256(manifest)
		try:
			return resolve_strat_hmm_pretext_config(load_config(path))
		finally:
			if previous is None:
				os.environ.pop(variable, None)
			else:
				os.environ[variable] = previous


def _training_identity(
	training: Mapping[str, object], label: str
) -> Mapping[str, object]:
	return _mapping(
		_mapping(training['identity'], f'{label} identity')['scientific_identity'],
		f'{label} scientific identity',
	)


def _manifest_path(training: Mapping[str, object]) -> Path:
	return Path(
		str(_mapping(training['pseudo_targets'], 'pseudo targets')['manifest'])
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


def _reference(value: object, label: str) -> dict[str, str]:
	mapping = _mapping(value, label)
	if set(mapping) != {'path', 'sha256'}:
		raise ValueError(f'{label} reference keys mismatch')
	path = Path(_string(mapping['path'], f'{label}.path'))
	sha = _string(mapping['sha256'], f'{label}.sha256')
	if not path.is_file() or file_sha256(path) != sha:
		raise ValueError(f'{label} reference identity mismatch')
	return {'path': str(path), 'sha256': sha}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _sha256(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _positive_int(value: object) -> bool:
	return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _finite_number(value: object) -> bool:
	return (
		not isinstance(value, bool)
		and isinstance(value, int | float)
		and math.isfinite(float(value))
	)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
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
			os.fsync(handle.fileno())
		temporary.replace(path)
	except BaseException:
		temporary.unlink(missing_ok=True)
		raise


__all__ = [
	'F3XYNeighborUnanimousPretrainingValidationConfig',
	'F3XYNeighborUnanimousPretrainingValidationResult',
	'f3_xy_neighbor_unanimous_pretraining_validation_config_from_mapping',
	'load_f3_xy_neighbor_unanimous_pretraining_handoff',
	'load_f3_xy_neighbor_unanimous_pretraining_validation_config',
	'validate_f3_xy_neighbor_unanimous_pretraining',
]
