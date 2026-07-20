"""Checkpoint helpers for stratigraphic HMM pretext training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

import seis_ssl_cluster
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.prototypes import (
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
)
from seis_ssl_cluster.training.checkpoint import capture_rng_state


@dataclass(frozen=True)
class StratRollingCheckpointResult:
	"""Result of a rolling strat HMM checkpoint write."""

	latest_path: Path
	best_path: Path
	best_score: float | None
	best_updated: bool


def save_strat_hmm_rolling_checkpoint(  # noqa: PLR0913
	checkpoint_dir: str | Path,
	*,
	student: torch.nn.Module,
	head: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch'],
	batch_index: int | None,
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	rng_state: Mapping[str, object] | None = None,
	best_score: float | None = None,
	trainability_summary: Mapping[str, object] | None = None,
	control_identity: Mapping[str, object] | None = None,
) -> StratRollingCheckpointResult:
	"""Write rolling ``latest.pt`` and update ``best.pt`` on lower loss."""
	checkpoint_root = Path(checkpoint_dir)
	latest_path = save_strat_hmm_checkpoint(
		checkpoint_root / 'latest.pt',
		student=student,
		head=head,
		optimizer=optimizer,
		epoch=epoch,
		mae_config=mae_config,
		stratigraphy_config=stratigraphy_config,
		metrics=metrics,
		global_step=global_step,
		amp_enabled=amp_enabled,
		scaler=scaler,
		checkpoint_kind=checkpoint_kind,
		batch_index=batch_index,
		rng_state=rng_state,
		trainability_summary=trainability_summary,
		control_identity=control_identity,
	)
	score = _loss_score(metrics)
	best_updated = _is_improved(score, best_score)
	resolved_best_score = best_score
	best_path = checkpoint_root / 'best.pt'
	if best_updated:
		_copy_checkpoint_atomic(latest_path, best_path)
		resolved_best_score = score
	return StratRollingCheckpointResult(
		latest_path=latest_path,
		best_path=best_path,
		best_score=resolved_best_score,
		best_updated=best_updated,
	)


def save_strat_hmm_checkpoint(  # noqa: PLR0913
	path: str | Path,
	*,
	student: torch.nn.Module,
	head: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch'],
	batch_index: int | None,
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	rng_state: Mapping[str, object] | None = None,
	trainability_summary: Mapping[str, object] | None = None,
	control_identity: Mapping[str, object] | None = None,
) -> Path:
	"""Atomically save an extraction-compatible strat HMM checkpoint."""
	checkpoint_path = Path(path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	model_state_dict = _state_dict_cpu(student)
	stratigraphy_state_dict = _state_dict_cpu(head)
	_validate_checkpoint_inputs(
		model_state_dict=model_state_dict,
		stratigraphy_state_dict=stratigraphy_state_dict,
		optimizer=optimizer,
		stratigraphy_config=stratigraphy_config,
		head=head,
	)
	payload = {
		'model_state_dict': model_state_dict,
		'stratigraphy_state_dict': stratigraphy_state_dict,
		'stratigraphy_config': _to_plain_value(stratigraphy_config),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': int(epoch),
		'global_step': int(global_step),
		'amp_enabled': bool(amp_enabled),
		'scaler_state_dict': None if scaler is None else scaler.state_dict(),
		'config': _to_plain_value(mae_config),
		'package_version': getattr(seis_ssl_cluster, '__version__', None),
		'metrics': dict(metrics),
		'trainability_summary': (
			{}
			if trainability_summary is None
			else _to_plain_value(trainability_summary)
		),
		'rng_state': dict(capture_rng_state() if rng_state is None else rng_state),
		'training_state': {
			'schema_version': 1,
			'stage': 'train_strat_hmm_pretext',
			'checkpoint_kind': checkpoint_kind,
			'batch_index': batch_index,
		},
	}
	if control_identity is not None:
		payload['control_identity'] = _to_plain_value(control_identity)
	if _is_multi_head_config(stratigraphy_config):
		payload['stratigraphy_checkpoint'] = _multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	return _atomic_torch_save(checkpoint_path, payload)


def validate_stratigraphy_checkpoint_payload(
	payload: Mapping[str, object],
	*,
	expected_config: Mapping[str, object] | None = None,
	expected_optimizer: torch.optim.Optimizer | None = None,
	expected_student: torch.nn.Module | None = None,
	expected_head: torch.nn.Module | None = None,
) -> None:
	"""Validate the versioned multi-head identity before loading any state."""
	identity = payload.get('stratigraphy_checkpoint')
	config = payload.get('stratigraphy_config')
	if identity is None:
		if isinstance(config, Mapping) and _is_multi_head_config(config):
			raise ValueError('multi-head checkpoint is missing versioned identity')
		return
	if not isinstance(identity, Mapping):
		raise TypeError('checkpoint stratigraphy_checkpoint must be a mapping')
	if identity.get('schema_version') != 2:
		raise ValueError('unsupported stratigraphy checkpoint schema_version')
	if identity.get('head_spec') != 'multi_resolution_ordered_prototypes_v1':
		raise ValueError('unsupported stratigraphy multi-head head_spec')
	state = payload.get('stratigraphy_state_dict')
	if not isinstance(config, Mapping) or not isinstance(state, Mapping):
		raise TypeError('multi-head checkpoint requires config and head state mappings')
	_validate_multi_head_identity(
		identity=identity,
		stratigraphy_config=config,
		stratigraphy_state_dict=state,
	)
	if not _optimizer_state_group_identity_matches(
		payload.get('optimizer_state_dict'),
		identity.get('optimizer_group_identity'),
	):
		raise ValueError(
			'checkpoint optimizer state groups do not match optimizer group identity'
		)
	if expected_config is not None:
		_validate_expected_multi_head_identity(identity, expected_config)
	if expected_optimizer is not None:
		_validate_expected_optimizer_group_identity(
			identity=identity,
			optimizer_state_dict=payload.get('optimizer_state_dict'),
			expected_optimizer=expected_optimizer,
			expected_student=expected_student,
			expected_head=expected_head,
		)


def inspect_stratigraphy_checkpoint(
	payload: Mapping[str, object],
	*,
	expected_config: Mapping[str, object] | None = None,
	expected_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
	"""Return a machine-readable checkpoint identity and state summary."""
	validate_stratigraphy_checkpoint_payload(payload)
	return {
		'stratigraphy_checkpoint': _to_plain_value(
			payload.get('stratigraphy_checkpoint', {'schema_version': 1}),
		),
		'model_state': _state_summary(payload.get('model_state_dict')),
		'stratigraphy_state': _state_summary(payload.get('stratigraphy_state_dict')),
		'optimizer_groups': _optimizer_group_summary(
			payload.get('optimizer_state_dict')
		),
		'best_selection_metric': 'metrics.loss',
		'embedding_compatible': isinstance(payload.get('model_state_dict'), Mapping),
		'resume_compatibility': _resume_compatibility_result(
			payload,
			expected_config=expected_config,
			expected_optimizer=expected_optimizer,
		),
	}


def _resume_compatibility_result(
	payload: Mapping[str, object],
	*,
	expected_config: Mapping[str, object] | None,
	expected_optimizer: torch.optim.Optimizer | None,
) -> dict[str, object]:
	"""Report resume compatibility without loading checkpoint state."""
	if expected_config is None:
		return {
			'checked': False,
			'compatible': None,
			'reason': 'expected_config was not provided',
		}
	try:
		_validate_resume_compatibility(
			payload,
			expected_config,
			expected_optimizer,
		)
	except (TypeError, ValueError) as exc:
		return {'checked': True, 'compatible': False, 'reason': str(exc)}
	return {'checked': True, 'compatible': True, 'reason': None}


def _validate_resume_compatibility(
	payload: Mapping[str, object],
	expected_config: Mapping[str, object],
	expected_optimizer: torch.optim.Optimizer | None,
) -> None:
	"""Apply the same compatibility checks used before resume state loading."""
	from seis_ssl_cluster.training.strat_hmm.resume import (  # noqa: PLC0415
		_first_compatibility_mismatch,
		_strat_resume_compatibility_view,
		_validate_stratigraphy_checkpoint_mode,
	)

	checkpoint_config = payload.get('stratigraphy_config')
	if not isinstance(checkpoint_config, Mapping):
		raise TypeError('resume checkpoint stratigraphy_config must be a mapping')
	checkpoint_view = _strat_resume_compatibility_view(checkpoint_config)
	expected_view = _strat_resume_compatibility_view(expected_config)
	if checkpoint_view != expected_view:
		label = _first_compatibility_mismatch(checkpoint_view, expected_view)
		msg = (
			'resume checkpoint stratigraphy_config is incompatible with '
			f'current resolved config at {label}'
		)
		raise ValueError(msg)
	_validate_stratigraphy_checkpoint_mode(
		payload,
		expected_config,
		optimizer=expected_optimizer,
	)


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
	return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _is_multi_head_config(config: Mapping[str, object]) -> bool:
	head = config.get('head')
	return isinstance(head, Mapping) and 'spec' in head


def _validate_checkpoint_inputs(
	*,
	model_state_dict: Mapping[str, torch.Tensor],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	optimizer: torch.optim.Optimizer,
	stratigraphy_config: Mapping[str, object],
	head: torch.nn.Module,
) -> None:
	_validate_finite_state_dict(model_state_dict, label='model_state_dict')
	_validate_finite_state_dict(
		stratigraphy_state_dict, label='stratigraphy_state_dict'
	)
	if _is_multi_head_config(stratigraphy_config):
		_validate_multi_head_config_and_state(
			stratigraphy_config, stratigraphy_state_dict
		)
		_validate_multi_head_module_and_state(
			stratigraphy_config,
			head,
			stratigraphy_state_dict,
		)
	_validate_finite_optimizer_state(optimizer.state_dict())
	for group in optimizer.param_groups:
		if not group.get('params'):
			raise ValueError('optimizer parameter group must not be empty')


def _validate_finite_optimizer_state(state: Mapping[str, object]) -> None:
	for value in _walk_values(state):
		if (
			isinstance(value, torch.Tensor)
			and value.is_floating_point()
			and not torch.isfinite(value).all()
		):
			raise ValueError('optimizer state contains non-finite values')


def _walk_values(value: object) -> Iterator[object]:
	if isinstance(value, Mapping):
		for child in value.values():
			yield from _walk_values(child)
	elif isinstance(value, list | tuple):
		for child in value:
			yield from _walk_values(child)
	else:
		yield value


def _validate_finite_state_dict(
	state: Mapping[str, torch.Tensor], *, label: str
) -> None:
	for key, value in state.items():
		if not isinstance(value, torch.Tensor):
			raise TypeError(f'{label}.{key} must be a tensor')
		if value.is_floating_point() and not torch.isfinite(value).all():
			raise ValueError(f'{label}.{key} contains non-finite values')


def _validate_multi_head_config_and_state(
	config: Mapping[str, object], state: Mapping[str, torch.Tensor]
) -> None:
	head = config.get('head')
	if not isinstance(head, Mapping):
		raise TypeError('multi-head stratigraphy_config.head must be a mapping')
	if head.get('spec') != 'multi_resolution_ordered_prototypes_v1':
		raise ValueError('multi-head checkpoint head spec mismatch')
	ks = _head_ks(head.get('ks'))
	prototype_keys = {f'heads.k{k}.prototypes' for k in ks}
	projection_keys = {
		f'heads.k{k}.projection.{parameter}'
		for k in ks
		for parameter in ('weight', 'bias')
	}
	state_keys = set(state)
	if 'projection_dim' in head:
		expected_keys = (
			prototype_keys | projection_keys
			if head.get('projection_dim') is not None
			else prototype_keys
		)
		valid_state_keys = {frozenset(expected_keys)}
	else:
		valid_state_keys = {
			frozenset(prototype_keys),
			frozenset(prototype_keys | projection_keys),
		}
	if frozenset(state_keys) not in valid_state_keys:
		raise ValueError('multi-head checkpoint head state keys do not match head.ks')
	_validate_multi_head_state_shapes(head, state)


def _validate_multi_head_module_and_state(
	config: Mapping[str, object],
	module: torch.nn.Module,
	state: Mapping[str, torch.Tensor],
) -> None:
	"""Require the saved module to implement the configured multi-head contract."""
	if type(module) is not MultiResolutionOrderedPrototypeHeads:
		raise TypeError(
			'multi-head checkpoint head module type must be '
			'MultiResolutionOrderedPrototypeHeads'
		)
	head = _required_mapping(config, 'head')
	ks = _head_ks(head.get('ks'))
	if module.head_ks != ks:
		raise ValueError(
			'multi-head checkpoint head module K order does not match head.ks'
		)
	expected_head_keys = tuple(f'k{k}' for k in ks)
	if tuple(module.heads) != expected_head_keys:
		raise ValueError('multi-head checkpoint head module keys do not match head.ks')
	for k in ks:
		per_head = module.heads[f'k{k}']
		if type(per_head) is not OrderedPrototypeHead:
			raise TypeError(
				'multi-head checkpoint per-head module type must be '
				'OrderedPrototypeHead'
			)
		if per_head.num_prototypes != k:
			raise ValueError(
				'multi-head checkpoint per-head prototype count does not match head.ks'
			)
		if per_head.feature_dim != module.feature_dim:
			raise ValueError(
				'multi-head checkpoint per-head feature dimension does not match '
				'module feature dimension'
			)
	_validate_multi_head_state_shapes(head, state)


def _validate_multi_head_state_shapes(
	head: Mapping[str, object], state: Mapping[str, torch.Tensor]
) -> None:
	"""Validate that every saved head tensor has its declared prototype shape."""
	ks = _head_ks(head.get('ks'))
	projection_dim: int | None = None
	feature_dim: int | None = None
	for k in ks:
		prototypes = state[f'heads.k{k}.prototypes']
		if prototypes.ndim != 2 or prototypes.shape[0] != k:
			raise ValueError(
				'multi-head checkpoint prototype tensor shape does not match head.ks'
			)
		current_projection_dim = prototypes.shape[1]
		if projection_dim is None:
			projection_dim = current_projection_dim
		elif current_projection_dim != projection_dim:
			raise ValueError(
				'multi-head checkpoint prototype tensor shapes must share projection '
				'dimension'
			)
		configured_projection_dim = head.get('projection_dim')
		if (
			configured_projection_dim is not None
			and current_projection_dim != configured_projection_dim
		):
			raise ValueError(
				'multi-head checkpoint prototype tensor shape does not match '
				'head.projection_dim'
			)
		projection_weight_key = f'heads.k{k}.projection.weight'
		if projection_weight_key not in state:
			continue
		weight = state[projection_weight_key]
		bias = state[f'heads.k{k}.projection.bias']
		if (
			weight.ndim != 2
			or bias.ndim != 1
			or weight.shape[0] != current_projection_dim
			or bias.shape[0] != current_projection_dim
		):
			raise ValueError(
				'multi-head checkpoint projection tensor shapes do not match '
				'prototype shape'
			)
		current_feature_dim = weight.shape[1]
		if feature_dim is None:
			feature_dim = current_feature_dim
		elif current_feature_dim != feature_dim:
			raise ValueError(
				'multi-head checkpoint projection tensor shapes must share feature '
				'dimension'
			)


def _multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	if scientific.get('target_manifest_sha256') != manifest_sha256:
		raise ValueError('target manifest SHA-256 does not match scientific identity')
	per_head_targets = _manifest_per_head_target_hashes(Path(manifest_path))
	if scientific.get('target_head_hashes') != per_head_targets:
		raise ValueError(
			'scientific identity per-head target hashes do not match target manifest'
		)
	result: dict[str, object] = {
		'schema_version': 2,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_manifest': {'path': manifest_path, 'sha256': manifest_sha256},
		'per_head_targets': per_head_targets,
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': _canonical_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
	}
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = control_identity.get('input_identities')
	if isinstance(inputs, Mapping):
		result['teacher_checkpoint_sha256'] = _identity_sha256(
			inputs.get('teacher_checkpoint')
		)
		result['student_init_checkpoint_sha256'] = _identity_sha256(
			inputs.get('student_init_checkpoint')
		)
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	result['initial_student_state_sha256'] = _required_sha256(
		initial_states.get('student'), 'initial_state_sha256.student'
	)
	result['initial_head_state_sha256'] = _required_sha256(
		initial_states.get('head'), 'initial_state_sha256.head'
	)
	return result


def _validate_multi_head_identity(
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	if list(_head_ks(head.get('ks'))) != identity.get('head_ks'):
		raise ValueError('checkpoint head_ks does not match stratigraphy config')
	if identity.get('head_spec') != head.get('spec'):
		raise ValueError('checkpoint head_spec does not match stratigraphy config')
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	_required_sha256(
		identity.get('initial_student_state_sha256'),
		'checkpoint initial_student_state_sha256',
	)
	_required_sha256(
		identity.get('initial_head_state_sha256'),
		'checkpoint initial_head_state_sha256',
	)
	target = identity.get('target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError('checkpoint target_manifest must be a mapping')
	path = _required_string(target.get('path'), 'checkpoint target_manifest.path')
	if target.get('sha256') != _file_sha256(Path(path)):
		raise ValueError('checkpoint target manifest SHA-256 mismatch')
	if identity.get('per_head_targets') != _manifest_per_head_target_hashes(
		Path(path)
	):
		raise ValueError(
			'checkpoint per-head target hashes do not match target manifest'
		)
	scientific = _required_mapping(
		_required_mapping(stratigraphy_config, 'identity'), 'scientific_identity'
	)
	for key in (
		'target_head_hashes',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
	):
		checkpoint_key = 'per_head_targets' if key == 'target_head_hashes' else key
		if identity.get(checkpoint_key) != scientific.get(key):
			raise ValueError(
				f'checkpoint {checkpoint_key} does not match scientific identity'
			)


def _validate_expected_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	head = _required_mapping(config, 'head')
	config_identity = _required_mapping(config, 'identity')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	teacher_path = Path(
		_required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
	)
	student_path = Path(
		student.get('init_checkpoint')
		or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
	)
	checks = {
		'head_spec': head.get('spec'),
		'head_ks': list(_head_ks(head.get('ks'))),
		'per_head_targets': scientific.get('target_head_hashes'),
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'scientific_identity_sha256': _canonical_sha256(scientific),
		'model_tag': config_identity.get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(teacher_path),
		'student_init_checkpoint_sha256': _file_sha256(student_path),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(f'checkpoint multi-head identity is incompatible at {key}')


def _validate_expected_optimizer_group_identity(
	*,
	identity: Mapping[str, object],
	optimizer_state_dict: object,
	expected_optimizer: torch.optim.Optimizer,
	expected_student: torch.nn.Module | None,
	expected_head: torch.nn.Module | None,
) -> None:
	if expected_student is None or expected_head is None:
		expected = _optimizer_group_summary(expected_optimizer.state_dict())
	else:
		expected = _optimizer_group_identity(
			expected_optimizer,
			parameter_names=_stratigraphy_parameter_names(
				expected_student, expected_head
			),
		)
	checkpoint_identity = identity.get('optimizer_group_identity')
	if expected_student is not None and expected_head is not None:
		if checkpoint_identity != expected:
			raise ValueError(
				'checkpoint optimizer group identity is incompatible with current '
				'optimizer'
			)
	elif not _optimizer_group_counts_match(checkpoint_identity, expected):
		raise ValueError(
			'checkpoint optimizer group identity is incompatible with current optimizer'
		)
	if not _optimizer_state_group_identity_matches(
		optimizer_state_dict, checkpoint_identity
	):
		raise ValueError(
			'checkpoint optimizer state groups do not match optimizer group identity'
		)


def _head_ks(value: object) -> tuple[int, ...]:
	if (
		not isinstance(value, list | tuple)
		or not value
		or any(isinstance(k, bool) or not isinstance(k, int) for k in value)
	):
		raise TypeError('multi-head checkpoint head.ks must be non-empty integers')
	return tuple(value)


def _manifest_per_head_target_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	"""Load target references and return their manifest-bound per-head hashes."""
	manifest = load_multi_head_target_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	head_ks = _head_ks(manifest.get('head_ks'))
	heads = _required_mapping(manifest, 'heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in head_ks:
		head = _required_mapping(heads, str(k))
		surveys = _required_mapping(head, 'surveys')
		result[str(k)] = {}
		for survey_id, entry in surveys.items():
			if not isinstance(entry, Mapping):
				raise TypeError(
					f'manifest heads.{k}.surveys.{survey_id} must be a mapping'
				)
			target = entry
			result[str(k)][str(survey_id)] = {
				name: _required_string(
					_required_mapping(target, name).get('sha256'),
					f'manifest heads.{k}.surveys.{survey_id}.{name}.sha256',
				)
				for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
			}
	return result


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return value


def _required_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _required_sha256(value: object, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _state_sha256(state: Mapping[str, object]) -> str:
	digest = hashlib.sha256()
	for key in sorted(state):
		value = state[key]
		if not isinstance(value, torch.Tensor):
			raise TypeError(f'state value {key!r} must be a tensor')
		cpu = value.detach().cpu().contiguous()
		digest.update(key.encode('utf-8'))
		digest.update(str(cpu.dtype).encode('utf-8'))
		digest.update(str(tuple(cpu.shape)).encode('utf-8'))
		digest.update(cpu.view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
	return hashlib.sha256(
		json.dumps(
			_to_plain_value(value),
			sort_keys=True,
			separators=(',', ':'),
			allow_nan=False,
		).encode('utf-8')
	).hexdigest()


def _identity_sha256(value: object) -> object:
	return value.get('sha256') if isinstance(value, Mapping) else None


def _optimizer_group_identity(
	optimizer: torch.optim.Optimizer,
	*,
	parameter_names: Mapping[int, str],
) -> list[dict[str, object]]:
	result: list[dict[str, object]] = []
	for group in optimizer.param_groups:
		parameters = group['params']
		try:
			names = [parameter_names[id(parameter)] for parameter in parameters]
		except KeyError as exc:
			raise ValueError(
				'optimizer contains a parameter outside the student/head modules'
			) from exc
		if len(names) != len(set(names)):
			raise ValueError('optimizer parameter group contains duplicate parameters')
		result.append({'name': group.get('name'), 'parameter_names': names})
	return result


def _stratigraphy_parameter_names(
	student: torch.nn.Module, head: torch.nn.Module
) -> dict[int, str]:
	result: dict[int, str] = {}
	for prefix, module in (('student', student), ('head', head)):
		for name, parameter in module.named_parameters():
			parameter_id = id(parameter)
			if parameter_id in result:
				raise ValueError(
					'student and head modules must not share trainable parameters'
				)
			result[parameter_id] = f'{prefix}.{name}'
	return result


def _optimizer_state_group_identity_matches(
	value: object, identity: object
) -> bool:
	if not isinstance(value, Mapping):
		return False
	groups = value.get('param_groups')
	if not isinstance(groups, list) or not isinstance(identity, list):
		return False
	if len(groups) != len(identity):
		return False
	next_parameter_id = 0
	for group, expected in zip(groups, identity, strict=True):
		if not isinstance(group, Mapping) or not isinstance(expected, Mapping):
			return False
		parameters = group.get('params')
		names = expected.get('parameter_names')
		if (
			group.get('name') != expected.get('name')
			or not isinstance(parameters, list)
			or not isinstance(names, list)
			or not all(isinstance(name, str) for name in names)
			or len(parameters) != len(names)
			or parameters
			!= list(range(next_parameter_id, next_parameter_id + len(parameters)))
		):
			return False
		next_parameter_id += len(parameters)
	return True


def _optimizer_group_counts_match(identity: object, summary: object) -> bool:
	if not isinstance(identity, list) or not isinstance(summary, list):
		return False
	if len(identity) != len(summary):
		return False
	return all(
		isinstance(recorded, Mapping)
		and isinstance(expected, Mapping)
		and recorded.get('name') == expected.get('name')
		and len(recorded.get('parameter_names', []))
		== len(expected.get('params', []))
		for recorded, expected in zip(identity, summary, strict=True)
	)


def _optimizer_group_summary(value: object) -> object:
	if not isinstance(value, Mapping):
		return None
	groups = value.get('param_groups')
	return groups if isinstance(groups, list) else None


def _state_summary(value: object) -> dict[str, object] | None:
	if not isinstance(value, Mapping):
		return None
	state = {
		str(key): tensor
		for key, tensor in value.items()
		if isinstance(tensor, torch.Tensor)
	}
	return {
		'keys': sorted(state),
		'shapes': {key: list(tensor.shape) for key, tensor in state.items()},
		'sha256': _state_sha256(state),
	}


def _loss_score(metrics: Mapping[str, float]) -> float | None:
	value = metrics.get('loss')
	if isinstance(value, int | float) and not isinstance(value, bool):
		score = float(value)
		if math.isfinite(score):
			return score
	return None


def _is_improved(score: float | None, best_score: float | None) -> bool:
	if score is None:
		return False
	if best_score is None:
		return True
	return score < best_score


def _copy_checkpoint_atomic(source: Path, target: Path) -> None:
	target.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = target.with_suffix('.pt.tmp')
	shutil.copy2(source, tmp_path)
	tmp_path.replace(target)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> Path:
	fd, tmp_name = tempfile.mkstemp(
		prefix=f'.{path.name}.',
		suffix='.tmp',
		dir=path.parent,
	)
	tmp_path = Path(tmp_name)
	try:
		with os.fdopen(fd, 'wb') as file_obj:
			torch.save(dict(payload), file_obj)
			file_obj.flush()
			os.fsync(file_obj.fileno())
		tmp_path.replace(path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()
	return path


def _to_plain_value(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _to_plain_value(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_to_plain_value(child) for child in value]
	if isinstance(value, Path):
		return str(value)
	return value


__all__ = [
	'StratRollingCheckpointResult',
	'save_strat_hmm_checkpoint',
	'save_strat_hmm_rolling_checkpoint',
]
