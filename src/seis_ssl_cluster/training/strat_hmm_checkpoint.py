"""Checkpoint helpers for stratigraphic HMM pretext training."""

from __future__ import annotations

import csv
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
from seis_ssl_cluster.stratigraphy.lateral_targets import (
	load_multi_head_lateral_target_manifest,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.prototypes import (
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
)
from seis_ssl_cluster.stratigraphy.state_posterior import (
	load_multi_head_state_posterior_manifest,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.training.checkpoint import capture_rng_state, load_checkpoint


@dataclass(frozen=True)
class StratRollingCheckpointResult:
	"""Result of a rolling strat HMM checkpoint write."""

	latest_path: Path
	best_path: Path
	best_score: float | None
	best_updated: bool
	checkpoint_selection: Mapping[str, object] | None = None


_CHECKPOINT_SELECTION_SCHEMA_VERSION = 1
_CHECKPOINT_SELECTION_CRITERION = 'metrics.loss'
_CHECKPOINT_SELECTION_POLICY = 'strictly_lower_loss_v1'
_CHECKPOINT_SELECTION_TRANSACTION_SCHEMA_VERSION = 1
_CHECKPOINT_SELECTION_TRANSACTION_NAME = '.checkpoint_selection_transaction.json'

_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION = 'xy_neighbor_consensus_hard_labels_v1'
_XY_NEIGHBOR_CONSENSUS_TARGET_SEMANTICS = (
	'xy_neighbor_consensus_hard_label_smoothing_v1'
)
_XY_NEIGHBOR_CONSENSUS_CONSISTENCY_POLICY = 'disabled_for_xy_neighbor_consensus_v1'
_XY_NEIGHBOR_CONSENSUS_EXPERIMENT_ROLE = (
	'multi_head_ordered_xy_neighbor_consensus_hard_pretext'
)

_XY_NEIGHBOR_CONSENSUS_CHECKPOINT_IDENTITY_FIELDS = frozenset(
	{
		'schema_version',
		'head_spec',
		'head_ks',
		'target_representation',
		'target_semantics',
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_consensus_target_manifest',
		'per_head_xy_neighbor_consensus_targets',
		'source_hard_manifest_sha256',
		'xy_neighbor_consensus_smoothing',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'model_tag',
		'output_root',
		'scientific_identity_sha256',
		'stratigraphy_state_sha256',
		'optimizer_group_identity',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	}
)

_XY_NEIGHBOR_CONSENSUS_SCIENTIFIC_IDENTITY_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'head_spec',
		'head_ks',
		'head_projection_dim',
		'head_temperature',
		'head_normalize',
		'target_representation',
		'target_semantics',
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_consensus_target_head_hashes',
		'source_hard_manifest_sha256',
		'xy_neighbor_consensus_smoothing',
		'supervised_loss',
		'consistency_policy',
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
		'teacher_checkpoint',
		'student_init_checkpoint',
		'student_unfreeze_top_blocks',
		'model',
		'data',
		'zero_mask',
		'train',
	}
)


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
	checkpoint_selection: Mapping[str, object] | None = None,
	trainability_summary: Mapping[str, object] | None = None,
	control_identity: Mapping[str, object] | None = None,
) -> StratRollingCheckpointResult:
	"""Write rolling ``latest.pt`` and update ``best.pt`` on lower loss."""
	checkpoint_root = Path(checkpoint_dir)
	checkpoint_root.mkdir(parents=True, exist_ok=True)
	is_multi_head = _is_multi_head_config(stratigraphy_config)
	if is_multi_head:
		recover_strat_hmm_rolling_checkpoint(checkpoint_root)
	selection_best_score = (
		float(selected_checkpoint_selection_event(checkpoint_selection)['loss'])
		if checkpoint_selection is not None
		else None
	)
	if is_multi_head and best_score != selection_best_score:
		raise ValueError('checkpoint selection best_score does not match history')
	selection = (
		_update_checkpoint_selection(
			checkpoint_selection,
			epoch=epoch,
			global_step=global_step,
			checkpoint_kind=checkpoint_kind,
			batch_index=batch_index,
			loss=_required_loss_score(metrics),
		)
		if is_multi_head
		else None
	)
	score = _loss_score(metrics)
	best_updated = (
		bool(selection['events'][-1]['best_updated'])
		if selection is not None
		else _is_improved(score, best_score)
	)
	if selection is not None and best_updated:
		_write_checkpoint_selection_transaction(checkpoint_root, selection)
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
		checkpoint_selection=selection,
	)
	resolved_best_score = (
		float(selection['selected']['loss']) if selection is not None else best_score
	)
	best_path = checkpoint_root / 'best.pt'
	if best_updated:
		_copy_checkpoint_atomic(latest_path, best_path)
		resolved_best_score = score
	if selection is not None:
		_write_checkpoint_selection_reports(checkpoint_root, selection)
	if selection is not None and best_updated:
		_checkpoint_selection_transaction_path(checkpoint_root).unlink()
	return StratRollingCheckpointResult(
		latest_path=latest_path,
		best_path=best_path,
		best_score=resolved_best_score,
		best_updated=best_updated,
		checkpoint_selection=selection,
	)


def recover_strat_hmm_rolling_checkpoint(checkpoint_dir: str | Path) -> None:
	"""Finish an interrupted multi-head ``latest.pt``/``best.pt`` update.

	A transaction record is only left behind while an improved latest checkpoint
	is waiting to be copied to ``best.pt``.  Recovery verifies that the latest
	payload is that recorded candidate before repairing the derived best file.
	"""
	checkpoint_root = Path(checkpoint_dir)
	transaction_path = _checkpoint_selection_transaction_path(checkpoint_root)
	if not transaction_path.is_file():
		return
	transaction = _load_checkpoint_selection_transaction(transaction_path)
	latest_path = checkpoint_root / 'latest.pt'
	if not latest_path.is_file():
		transaction_path.unlink()
		return
	latest = load_checkpoint(latest_path, map_location='cpu')
	selection = _validated_checkpoint_selection(latest.get('checkpoint_selection'))
	if checkpoint_selection_sha256(selection) != transaction['selection_sha256']:
		transaction_path.unlink()
		return
	selected = selected_checkpoint_selection_event(selection)
	if selected != transaction['selected']:
		raise ValueError('checkpoint selection transaction selected event mismatch')
	events = selection['events']
	if not isinstance(events, list) or not isinstance(events[-1], Mapping):
		raise TypeError('checkpoint selection transaction events are invalid')
	last = events[-1]
	if (
		_selection_event_identity_from_event(last) != selected
		or last.get('best_updated') is not True
	):
		raise ValueError('checkpoint selection transaction does not select latest')
	_validate_checkpoint_selection_payload_binding(latest, selection)
	best_path = checkpoint_root / 'best.pt'
	if not _checkpoint_payload_matches_selected_event(best_path, selected):
		_copy_checkpoint_atomic(latest_path, best_path)
	_write_checkpoint_selection_reports(checkpoint_root, selection)
	transaction_path.unlink()


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
	checkpoint_selection: Mapping[str, object] | None = None,
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
		student=student,
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
		if checkpoint_selection is None:
			checkpoint_selection = _update_checkpoint_selection(
				None,
				epoch=epoch,
				global_step=global_step,
				checkpoint_kind=checkpoint_kind,
				batch_index=batch_index,
				loss=_required_loss_score(metrics),
			)
		payload['checkpoint_selection'] = _validated_checkpoint_selection(
			checkpoint_selection
		)
		payload['stratigraphy_checkpoint'] = _multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	return _atomic_torch_save(checkpoint_path, payload)


def validate_stratigraphy_checkpoint_payload(  # noqa: C901
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
	if identity.get('schema_version') not in {2, 3, 4, 5}:
		raise ValueError('unsupported stratigraphy checkpoint schema_version')
	if identity.get('head_spec') != 'multi_resolution_ordered_prototypes_v1':
		raise ValueError('unsupported stratigraphy multi-head head_spec')
	state = payload.get('stratigraphy_state_dict')
	if not isinstance(config, Mapping) or not isinstance(state, Mapping):
		raise TypeError('multi-head checkpoint requires config and head state mappings')
	expected_schema_version = (
		5
		if _is_xy_neighbor_consensus_multi_head_config(config)
		else (
			4
			if _is_lateral_multi_head_config(config)
			else 3
			if _is_soft_multi_head_config(config)
			else 2
		)
	)
	if identity.get('schema_version') != expected_schema_version:
		raise ValueError(
			'multi-head checkpoint schema_version does not match target representation'
		)
	_validate_multi_head_identity(
		identity=identity,
		stratigraphy_config=config,
		stratigraphy_state_dict=state,
	)
	selection = _validated_checkpoint_selection(payload.get('checkpoint_selection'))
	_validate_checkpoint_selection_payload_binding(payload, selection)
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


def _validate_checkpoint_inputs(  # noqa: PLR0913
	*,
	model_state_dict: Mapping[str, torch.Tensor],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	optimizer: torch.optim.Optimizer,
	stratigraphy_config: Mapping[str, object],
	student: torch.nn.Module,
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
		_validate_multi_head_optimizer_layout(
			optimizer=optimizer,
			stratigraphy_config=stratigraphy_config,
			student=student,
			head=head,
		)
	_validate_finite_optimizer_state(optimizer.state_dict())
	for group in optimizer.param_groups:
		if not group.get('params'):
			raise ValueError('optimizer parameter group must not be empty')


def _validate_multi_head_optimizer_layout(  # noqa: C901
	*,
	optimizer: torch.optim.Optimizer,
	stratigraphy_config: Mapping[str, object],
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> None:
	"""Require the fixed multi-head encoder/head optimizer partition."""
	groups = optimizer.param_groups
	if len(groups) != 2 or [group.get('name') for group in groups] != [
		'head',
		'encoder',
	]:
		raise ValueError(
			'multi-head optimizer requires exactly head and encoder parameter groups'
		)
	train = _required_mapping(stratigraphy_config, 'train')
	expected_lrs = (
		_required_positive_finite_number(train.get('lr'), 'train.lr'),
		_required_positive_finite_number(train.get('encoder_lr'), 'train.encoder_lr'),
	)
	for group, expected_lr in zip(groups, expected_lrs, strict=True):
		actual_lr = _required_positive_finite_number(
			group.get('lr'), f'multi-head optimizer {group.get("name")!r} group lr'
		)
		if actual_lr != expected_lr:
			raise ValueError(
				'multi-head optimizer group learning rate does not match train config'
			)

	head_parameters = tuple(head.parameters())
	student_parameters = tuple(student.parameters())
	if not all(parameter.requires_grad for parameter in head_parameters):
		raise ValueError('all multi-head head parameters must be trainable')
	trainable_student_parameters = tuple(
		parameter for parameter in student_parameters if parameter.requires_grad
	)
	if not trainable_student_parameters:
		raise ValueError('multi-head optimizer requires trainable encoder parameters')

	known_parameters = {
		id(parameter): parameter
		for parameter in (*head_parameters, *student_parameters)
	}
	group_parameters = tuple(
		parameter for group in groups for parameter in group['params']
	)
	group_ids = tuple(id(parameter) for parameter in group_parameters)
	if len(group_ids) != len(set(group_ids)):
		raise ValueError('multi-head optimizer groups contain duplicate parameters')
	if any(parameter_id not in known_parameters for parameter_id in group_ids):
		raise ValueError(
			'multi-head optimizer contains a parameter outside the student/head modules'
		)
	if any(
		not known_parameters[parameter_id].requires_grad for parameter_id in group_ids
	):
		raise ValueError('multi-head optimizer contains frozen parameters')

	head_ids = {id(parameter) for parameter in head_parameters}
	encoder_ids = {id(parameter) for parameter in trainable_student_parameters}
	if {id(parameter) for parameter in groups[0]['params']} != head_ids:
		raise ValueError(
			'multi-head optimizer head group must contain every head parameter '
			'exactly once'
		)
	if {id(parameter) for parameter in groups[1]['params']} != encoder_ids:
		raise ValueError(
			'multi-head optimizer encoder group must contain every trainable encoder '
			'parameter exactly once'
		)


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
	_validate_multi_head_module_scientific_identity(config, module)
	_validate_multi_head_state_shapes(head, state)


def _validate_multi_head_module_scientific_identity(
	config: Mapping[str, object],
	module: MultiResolutionOrderedPrototypeHeads,
) -> None:
	"""Bind non-state-dict head settings and loss weights to provenance."""
	identity = config.get('identity')
	if not isinstance(identity, Mapping):
		return
	scientific = identity.get('scientific_identity')
	if not isinstance(scientific, Mapping):
		return
	head = _required_mapping(config, 'head')
	for scientific_key, config_key, module_key in (
		('head_temperature', 'temperature', 'temperature'),
		('head_normalize', 'normalize', 'normalize'),
	):
		if scientific_key not in scientific:
			continue
		expected = scientific[scientific_key]
		if head.get(config_key) != expected:
			raise ValueError(
				f'multi-head checkpoint head.{config_key} does not match '
				'scientific identity'
			)
		if any(
			getattr(per_head, module_key) != expected
			for per_head in module.heads.values()
		):
			raise ValueError(
				f'multi-head checkpoint module {module_key} does not match '
				'scientific identity'
			)

	for key in (
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
	):
		if key not in scientific:
			continue
		loss = _required_mapping(config, 'loss')
		if loss.get(key) != scientific[key]:
			raise ValueError(
				f'multi-head checkpoint loss.{key} does not match scientific identity'
			)


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
	if _is_soft_multi_head_config(stratigraphy_config):
		return _soft_multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	if _is_lateral_multi_head_config(stratigraphy_config):
		return _lateral_multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
	if _is_xy_neighbor_consensus_multi_head_config(stratigraphy_config):
		return _xy_neighbor_consensus_multi_head_checkpoint_identity(
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
			control_identity=control_identity,
			optimizer=optimizer,
			student=student,
			head=head,
		)
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
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
	}
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	result['teacher_checkpoint_sha256'] = _required_sha256(
		_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
		'input_identities.teacher_checkpoint.sha256',
	)
	result['student_init_checkpoint_sha256'] = _required_sha256(
		_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
		'input_identities.student_init_checkpoint.sha256',
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


def _soft_multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	"""Build schema-v3 identity for an immutable soft-posterior run."""
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	per_head_posteriors = _posterior_per_head_hashes(Path(manifest_path))
	if scientific.get('posterior_manifest_sha256') != manifest_sha256:
		raise ValueError(
			'posterior manifest SHA-256 does not match scientific identity'
		)
	if scientific.get('posterior_head_hashes') != per_head_posteriors:
		raise ValueError('scientific identity posterior hashes do not match manifest')
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	return {
		'schema_version': 3,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_representation': scientific['target_representation'],
		'posterior_semantics': scientific['posterior_semantics'],
		'posterior_cost_temperature': scientific['posterior_cost_temperature'],
		'posterior_manifest_sha256': manifest_sha256,
		'posterior_manifest': {'path': manifest_path, 'sha256': manifest_sha256},
		'per_head_posteriors': per_head_posteriors,
		'consistency_policy': scientific['consistency_policy'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _lateral_multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	"""Build schema-v4 identity for immutable M5-LS hard supervision."""
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	per_head_targets = _lateral_per_head_target_hashes(Path(manifest_path))
	if scientific.get('lateral_target_manifest_sha256') != manifest_sha256:
		raise ValueError('lateral manifest SHA-256 does not match scientific identity')
	if scientific.get('lateral_target_head_hashes') != per_head_targets:
		raise ValueError('lateral target hashes do not match scientific identity')
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	return {
		'schema_version': 4,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_representation': scientific['target_representation'],
		'target_semantics': scientific['target_semantics'],
		'lateral_target_manifest_sha256': manifest_sha256,
		'lateral_target_manifest': {'path': manifest_path, 'sha256': manifest_sha256},
		'per_head_lateral_targets': per_head_targets,
		'source_hard_manifest_sha256': scientific['source_hard_manifest_sha256'],
		'source_posterior_manifest_sha256': scientific[
			'source_posterior_manifest_sha256'
		],
		'lateral_smoothing': scientific['lateral_smoothing'],
		'consistency_policy': scientific['consistency_policy'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _xy_neighbor_consensus_multi_head_checkpoint_identity(  # noqa: PLR0913
	*,
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, torch.Tensor],
	control_identity: Mapping[str, object] | None,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module,
	head: torch.nn.Module,
) -> dict[str, object]:
	"""Build schema-v5 identity for source-hard XY consensus supervision."""
	head_config = _required_mapping(stratigraphy_config, 'head')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	identity = _required_mapping(stratigraphy_config, 'identity')
	scientific = _required_mapping(identity, 'scientific_identity')
	manifest_path = _required_string(
		pseudo_targets.get('manifest'), 'pseudo_targets.manifest'
	)
	manifest_sha256 = _file_sha256(Path(manifest_path))
	(
		per_head_targets,
		source_hard_manifest_sha256,
		smoothing,
	) = _xy_neighbor_consensus_target_provenance(Path(manifest_path))
	if (
		scientific.get('xy_neighbor_consensus_target_manifest_sha256')
		!= manifest_sha256
	):
		raise ValueError(
			'XY neighbor consensus manifest SHA-256 does not match scientific identity'
		)
	if scientific.get('xy_neighbor_consensus_target_head_hashes') != per_head_targets:
		raise ValueError(
			'XY neighbor consensus target hashes do not match scientific identity'
		)
	if scientific.get('source_hard_manifest_sha256') != source_hard_manifest_sha256:
		raise ValueError(
			'XY neighbor consensus source hard manifest does not match scientific '
			'identity'
		)
	if _to_plain_value(scientific.get('xy_neighbor_consensus_smoothing')) != smoothing:
		raise ValueError(
			'XY neighbor consensus smoothing policy does not match scientific identity'
		)
	if not isinstance(control_identity, Mapping):
		raise TypeError('multi-head checkpoint requires control identity')
	inputs = _required_mapping(control_identity, 'input_identities')
	initial_states = control_identity.get('initial_state_sha256')
	if not isinstance(initial_states, Mapping):
		raise TypeError('multi-head checkpoint requires initial state hashes')
	return {
		'schema_version': 5,
		'head_spec': head_config['spec'],
		'head_ks': list(_head_ks(head_config.get('ks'))),
		'target_representation': scientific['target_representation'],
		'target_semantics': scientific['target_semantics'],
		'xy_neighbor_consensus_target_manifest_sha256': manifest_sha256,
		'xy_neighbor_consensus_target_manifest': {
			'path': manifest_path,
			'sha256': manifest_sha256,
		},
		'per_head_xy_neighbor_consensus_targets': per_head_targets,
		'source_hard_manifest_sha256': source_hard_manifest_sha256,
		'xy_neighbor_consensus_smoothing': smoothing,
		'consistency_policy': scientific['consistency_policy'],
		'consistency_weight': scientific['consistency_weight'],
		'consistency_beta': scientific['consistency_beta'],
		'model_tag': identity.get('model_tag'),
		'output_root': _required_mapping(stratigraphy_config, 'paths').get(
			'output_root'
		),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'stratigraphy_state_sha256': _state_sha256(stratigraphy_state_dict),
		'optimizer_group_identity': _optimizer_group_identity(
			optimizer,
			parameter_names=_stratigraphy_parameter_names(student, head),
		),
		'teacher_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'teacher_checkpoint').get('sha256'),
			'input_identities.teacher_checkpoint.sha256',
		),
		'student_init_checkpoint_sha256': _required_sha256(
			_required_mapping(inputs, 'student_init_checkpoint').get('sha256'),
			'input_identities.student_init_checkpoint.sha256',
		),
		'initial_student_state_sha256': _required_sha256(
			initial_states.get('student'), 'initial_state_sha256.student'
		),
		'initial_head_state_sha256': _required_sha256(
			initial_states.get('head'), 'initial_state_sha256.head'
		),
	}


def _validate_multi_head_identity(  # noqa: C901
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	if _is_soft_multi_head_config(stratigraphy_config):
		_validate_soft_multi_head_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	if _is_lateral_multi_head_config(stratigraphy_config):
		_validate_lateral_multi_head_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	if _is_xy_neighbor_consensus_multi_head_config(stratigraphy_config):
		_validate_xy_neighbor_consensus_multi_head_identity(
			identity=identity,
			stratigraphy_config=stratigraphy_config,
			stratigraphy_state_dict=stratigraphy_state_dict,
		)
		return
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_consensus_fields_from_legacy_identity(identity, scientific)
	if list(_head_ks(head.get('ks'))) != identity.get('head_ks'):
		raise ValueError('checkpoint head_ks does not match stratigraphy config')
	if identity.get('head_spec') != head.get('spec'):
		raise ValueError('checkpoint head_spec does not match stratigraphy config')
	for key, expected in (
		('model_tag', config_identity.get('model_tag')),
		('output_root', paths.get('output_root')),
	):
		if identity.get(key) != expected:
			raise ValueError(f'checkpoint {key} does not match stratigraphy config')
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
	_required_sha256(
		identity.get('teacher_checkpoint_sha256'),
		'checkpoint teacher_checkpoint_sha256',
	)
	_required_sha256(
		identity.get('student_init_checkpoint_sha256'),
		'checkpoint student_init_checkpoint_sha256',
	)
	scientific = _validate_multi_head_target_manifest_identity(
		identity=identity,
		stratigraphy_config=stratigraphy_config,
	)
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
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


def _validate_multi_head_target_manifest_identity(
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
) -> Mapping[str, object]:
	"""Bind the checkpoint manifest to its scientific identity."""
	target = identity.get('target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError('checkpoint target_manifest must be a mapping')
	path = _required_string(target.get('path'), 'checkpoint target_manifest.path')
	manifest_sha256 = _required_sha256(
		target.get('sha256'), 'checkpoint target_manifest.sha256'
	)
	if manifest_sha256 != _file_sha256(Path(path)):
		raise ValueError('checkpoint target manifest SHA-256 mismatch')
	if identity.get('per_head_targets') != _manifest_per_head_target_hashes(Path(path)):
		raise ValueError(
			'checkpoint per-head target hashes do not match target manifest'
		)
	scientific = _required_mapping(
		_required_mapping(stratigraphy_config, 'identity'), 'scientific_identity'
	)
	if manifest_sha256 != _required_sha256(
		scientific.get('target_manifest_sha256'),
		'scientific identity target_manifest_sha256',
	):
		raise ValueError(
			'checkpoint target manifest SHA-256 does not match scientific identity'
		)
	return scientific


def _validate_expected_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	if _is_soft_multi_head_config(config):
		_validate_expected_soft_multi_head_identity(identity, config)
		return
	if _is_lateral_multi_head_config(config):
		_validate_expected_lateral_multi_head_identity(identity, config)
		return
	if _is_xy_neighbor_consensus_multi_head_config(config):
		_validate_expected_xy_neighbor_consensus_multi_head_identity(identity, config)
		return
	if identity.get('target_representation') not in {None, 'hard_viterbi_labels_v1'}:
		raise ValueError(
			'checkpoint multi-head identity is incompatible at target_representation'
		)
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
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
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


def _is_soft_multi_head_config(config: Mapping[str, object]) -> bool:
	pseudo_targets = config.get('pseudo_targets')
	return (
		isinstance(pseudo_targets, Mapping)
		and pseudo_targets.get('target_representation')
		== 'ordered_path_state_posterior_v1'
	)


def _is_lateral_multi_head_config(config: Mapping[str, object]) -> bool:
	pseudo_targets = config.get('pseudo_targets')
	return (
		isinstance(pseudo_targets, Mapping)
		and pseudo_targets.get('target_representation')
		== 'lateral_mean_field_hard_labels_v1'
	)


def _is_xy_neighbor_consensus_multi_head_config(
	config: Mapping[str, object],
) -> bool:
	pseudo_targets = config.get('pseudo_targets')
	return (
		isinstance(pseudo_targets, Mapping)
		and pseudo_targets.get('target_representation')
		== _XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION
	)


def _posterior_per_head_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	manifest = load_multi_head_state_posterior_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	heads = _required_mapping(manifest, 'heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in _head_ks(manifest.get('head_ks')):
		head = _required_mapping(heads, str(k))
		surveys = _required_mapping(head, 'surveys')
		result[str(k)] = {}
		for survey_id, entry in surveys.items():
			if not isinstance(entry, Mapping):
				raise TypeError(f'posterior manifest head {k} survey must be a mapping')
			result[str(k)][str(survey_id)] = {
				name: _required_string(
					_required_mapping(entry, name).get('sha256'),
					f'posterior manifest heads.{k}.surveys.{survey_id}.{name}.sha256',
				)
				for name in ('posterior', 'valid_tokens', 'metadata')
			}
	return result


def _lateral_per_head_target_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	manifest = load_multi_head_lateral_target_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	return _per_head_hard_target_hashes(manifest)


def _xy_neighbor_consensus_per_head_target_hashes(
	manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
	return _xy_neighbor_consensus_target_provenance(manifest_path)[0]


def _xy_neighbor_consensus_target_provenance(
	manifest_path: Path,
) -> tuple[
	dict[str, dict[str, dict[str, str]]],
	str,
	Mapping[str, object],
]:
	"""Read the source-hard-only identity from a strict consensus manifest."""
	manifest = load_multi_head_xy_neighbor_consensus_target_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	source_hard = _required_mapping(manifest, 'source_hard_manifest')
	source_hard_manifest_sha256 = _required_sha256(
		source_hard.get('sha256'),
		'XY neighbor consensus source_hard_manifest.sha256',
	)
	smoothing = _required_mapping(manifest, 'smoothing')
	return (
		_per_head_hard_target_hashes(manifest),
		source_hard_manifest_sha256,
		_to_plain_value(smoothing),
	)


def _per_head_hard_target_hashes(
	manifest: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	heads = _required_mapping(manifest, 'heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in _head_ks(manifest.get('head_ks')):
		head = _required_mapping(heads, str(k))
		surveys = _required_mapping(head, 'surveys')
		result[str(k)] = {}
		for survey_id, entry in surveys.items():
			if not isinstance(entry, Mapping):
				raise TypeError(f'lateral manifest head {k} survey must be a mapping')
			target = entry
			result[str(k)][str(survey_id)] = {
				name: _required_string(
					_required_mapping(target, name).get('sha256'),
					f'lateral manifest heads.{k}.surveys.{survey_id}.{name}.sha256',
				)
				for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
			}
	return result


def _validate_soft_multi_head_identity(  # noqa: C901
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate the schema-v3 soft identity without accepting hard artifacts."""
	if identity.get('schema_version') != 3:
		raise ValueError('soft posterior checkpoint requires schema_version 3')
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_consensus_fields_from_legacy_identity(identity, scientific)
	if identity.get('head_spec') != head.get('spec') or identity.get('head_ks') != list(
		_head_ks(head.get('ks'))
	):
		raise ValueError(
			'soft checkpoint head identity does not match stratigraphy config'
		)
	if identity.get('model_tag') != config_identity.get('model_tag') or identity.get(
		'output_root'
	) != paths.get('output_root'):
		raise ValueError(
			'soft checkpoint model identity does not match stratigraphy config'
		)
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	posterior = identity.get('posterior_manifest')
	if not isinstance(posterior, Mapping):
		raise TypeError('soft checkpoint posterior_manifest must be a mapping')
	path = Path(
		_required_string(posterior.get('path'), 'checkpoint posterior_manifest.path')
	)
	sha256 = _required_sha256(
		posterior.get('sha256'), 'checkpoint posterior_manifest.sha256'
	)
	if sha256 != _file_sha256(path):
		raise ValueError('checkpoint posterior manifest SHA-256 mismatch')
	if identity.get('per_head_posteriors') != _posterior_per_head_hashes(path):
		raise ValueError('checkpoint posterior hashes do not match posterior manifest')
	for key in (
		'target_representation',
		'posterior_semantics',
		'posterior_cost_temperature',
		'posterior_manifest_sha256',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
	):
		if identity.get(key) != scientific.get(key):
			raise ValueError(f'checkpoint {key} does not match scientific identity')
	if identity.get('per_head_posteriors') != scientific.get('posterior_head_hashes'):
		raise ValueError('checkpoint posterior hashes do not match scientific identity')
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		_required_sha256(identity.get(key), f'checkpoint {key}')


def _validate_expected_soft_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	checks = {
		'target_representation': scientific.get('target_representation'),
		'posterior_semantics': scientific.get('posterior_semantics'),
		'posterior_cost_temperature': scientific.get('posterior_cost_temperature'),
		'posterior_manifest_sha256': scientific.get('posterior_manifest_sha256'),
		'per_head_posteriors': scientific.get('posterior_head_hashes'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(
				f'checkpoint soft multi-head identity is incompatible at {key}'
			)


def _validate_lateral_multi_head_identity(  # noqa: C901
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate schema-v4 M5-LS identity before any model state is restored."""
	if identity.get('schema_version') != 4:
		raise ValueError('lateral hard checkpoint requires schema_version 4')
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_consensus_fields_from_legacy_identity(identity, scientific)
	if identity.get('head_spec') != head.get('spec') or identity.get('head_ks') != list(
		_head_ks(head.get('ks'))
	):
		raise ValueError('lateral checkpoint head identity does not match config')
	if identity.get('model_tag') != config_identity.get('model_tag') or identity.get(
		'output_root'
	) != paths.get('output_root'):
		raise ValueError('lateral checkpoint model identity does not match config')
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	target = identity.get('lateral_target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError('lateral checkpoint target manifest must be a mapping')
	path = Path(
		_required_string(target.get('path'), 'checkpoint lateral_target_manifest.path')
	)
	sha256 = _required_sha256(
		target.get('sha256'), 'checkpoint lateral_target_manifest.sha256'
	)
	if sha256 != _file_sha256(path):
		raise ValueError('checkpoint lateral target manifest SHA-256 mismatch')
	if identity.get('per_head_lateral_targets') != _lateral_per_head_target_hashes(
		path
	):
		raise ValueError('checkpoint lateral target hashes do not match manifest')
	for key in (
		'target_representation',
		'target_semantics',
		'lateral_target_manifest_sha256',
		'source_hard_manifest_sha256',
		'source_posterior_manifest_sha256',
		'lateral_smoothing',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
	):
		if identity.get(key) != scientific.get(key):
			raise ValueError(f'checkpoint {key} does not match scientific identity')
	if identity.get('per_head_lateral_targets') != scientific.get(
		'lateral_target_head_hashes'
	):
		raise ValueError('checkpoint lateral hashes do not match scientific identity')
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		_required_sha256(identity.get(key), f'checkpoint {key}')


def _validate_expected_lateral_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	checks = {
		'target_representation': scientific.get('target_representation'),
		'target_semantics': scientific.get('target_semantics'),
		'lateral_target_manifest_sha256': scientific.get(
			'lateral_target_manifest_sha256'
		),
		'per_head_lateral_targets': scientific.get('lateral_target_head_hashes'),
		'source_hard_manifest_sha256': scientific.get('source_hard_manifest_sha256'),
		'source_posterior_manifest_sha256': scientific.get(
			'source_posterior_manifest_sha256'
		),
		'lateral_smoothing': scientific.get('lateral_smoothing'),
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(
				f'checkpoint lateral multi-head identity is incompatible at {key}'
			)


def _validate_xy_neighbor_consensus_multi_head_identity(  # noqa: C901, PLR0912
	*,
	identity: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	stratigraphy_state_dict: Mapping[str, object],
) -> None:
	"""Validate schema-v5 XY-consensus identity before restoring state."""
	if identity.get('schema_version') != 5:
		raise ValueError('XY neighbor consensus checkpoint requires schema_version 5')
	_validate_multi_head_config_and_state(
		stratigraphy_config,
		{
			str(key): value
			for key, value in stratigraphy_state_dict.items()
			if isinstance(value, torch.Tensor)
		},
	)
	head = _required_mapping(stratigraphy_config, 'head')
	config_identity = _required_mapping(stratigraphy_config, 'identity')
	paths = _required_mapping(stratigraphy_config, 'paths')
	scientific = _required_mapping(config_identity, 'scientific_identity')
	_reject_xy_neighbor_consensus_legacy_fields(identity, scientific)
	if identity.get('head_spec') != head.get('spec') or identity.get('head_ks') != list(
		_head_ks(head.get('ks'))
	):
		raise ValueError(
			'XY neighbor consensus checkpoint head identity does not match config'
		)
	if identity.get('model_tag') != config_identity.get('model_tag') or identity.get(
		'output_root'
	) != paths.get('output_root'):
		raise ValueError(
			'XY neighbor consensus checkpoint model identity does not match config'
		)
	if identity.get('stratigraphy_state_sha256') != _state_sha256(
		stratigraphy_state_dict
	):
		raise ValueError('checkpoint stratigraphy state SHA-256 mismatch')
	target = identity.get('xy_neighbor_consensus_target_manifest')
	if not isinstance(target, Mapping):
		raise TypeError(
			'XY neighbor consensus checkpoint target manifest must be a mapping'
		)
	path = Path(
		_required_string(
			target.get('path'),
			'checkpoint xy_neighbor_consensus_target_manifest.path',
		)
	)
	sha256 = _required_sha256(
		target.get('sha256'),
		'checkpoint xy_neighbor_consensus_target_manifest.sha256',
	)
	if sha256 != _file_sha256(path):
		raise ValueError('checkpoint XY neighbor consensus manifest SHA-256 mismatch')
	(
		per_head_targets,
		source_hard_manifest_sha256,
		smoothing,
	) = _xy_neighbor_consensus_target_provenance(path)
	if identity.get('per_head_xy_neighbor_consensus_targets') != per_head_targets:
		raise ValueError(
			'checkpoint XY neighbor consensus target hashes do not match manifest'
		)
	for key, expected in (
		('target_representation', scientific.get('target_representation')),
		('target_semantics', scientific.get('target_semantics')),
		(
			'xy_neighbor_consensus_target_manifest_sha256',
			scientific.get('xy_neighbor_consensus_target_manifest_sha256'),
		),
		('source_hard_manifest_sha256', scientific.get('source_hard_manifest_sha256')),
		(
			'xy_neighbor_consensus_smoothing',
			_to_plain_value(scientific.get('xy_neighbor_consensus_smoothing')),
		),
		('consistency_policy', scientific.get('consistency_policy')),
		('consistency_weight', scientific.get('consistency_weight')),
		('consistency_beta', scientific.get('consistency_beta')),
	):
		if identity.get(key) != expected:
			raise ValueError(f'checkpoint {key} does not match scientific identity')
	if identity.get('xy_neighbor_consensus_target_manifest_sha256') != sha256:
		raise ValueError(
			'checkpoint XY neighbor consensus manifest hash does not match target '
			'manifest'
		)
	if identity.get('per_head_xy_neighbor_consensus_targets') != scientific.get(
		'xy_neighbor_consensus_target_head_hashes'
	):
		raise ValueError(
			'checkpoint XY neighbor consensus hashes do not match scientific identity'
		)
	if identity.get('source_hard_manifest_sha256') != source_hard_manifest_sha256:
		raise ValueError(
			'checkpoint XY neighbor consensus source hard manifest does not match '
			'target manifest'
		)
	if identity.get('xy_neighbor_consensus_smoothing') != smoothing:
		raise ValueError(
			'checkpoint XY neighbor consensus smoothing does not match target manifest'
		)
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('checkpoint scientific identity SHA-256 mismatch')
	for key in (
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
	):
		_required_sha256(identity.get(key), f'checkpoint {key}')


def _validate_expected_xy_neighbor_consensus_multi_head_identity(
	identity: Mapping[str, object], config: Mapping[str, object]
) -> None:
	"""Reject cross-resume and stale source-hard consensus provenance."""
	scientific = _required_mapping(
		_required_mapping(config, 'identity'), 'scientific_identity'
	)
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	checks = {
		'target_representation': scientific.get('target_representation'),
		'target_semantics': scientific.get('target_semantics'),
		'xy_neighbor_consensus_target_manifest_sha256': scientific.get(
			'xy_neighbor_consensus_target_manifest_sha256'
		),
		'per_head_xy_neighbor_consensus_targets': scientific.get(
			'xy_neighbor_consensus_target_head_hashes'
		),
		'source_hard_manifest_sha256': scientific.get('source_hard_manifest_sha256'),
		'xy_neighbor_consensus_smoothing': _to_plain_value(
			scientific.get('xy_neighbor_consensus_smoothing')
		),
		'consistency_policy': scientific.get('consistency_policy'),
		'consistency_weight': scientific.get('consistency_weight'),
		'consistency_beta': scientific.get('consistency_beta'),
		'scientific_identity_sha256': scientific_identity_sha256(scientific),
		'model_tag': _required_mapping(config, 'identity').get('model_tag'),
		'output_root': _required_mapping(config, 'paths').get('output_root'),
		'teacher_checkpoint_sha256': _file_sha256(
			Path(_required_string(teacher.get('checkpoint'), 'teacher.checkpoint'))
		),
		'student_init_checkpoint_sha256': _file_sha256(
			Path(
				student.get('init_checkpoint')
				or _required_string(teacher.get('checkpoint'), 'teacher.checkpoint')
			)
		),
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(
				'checkpoint XY neighbor consensus multi-head identity is '
				f'incompatible at {key}'
			)


def _reject_xy_neighbor_consensus_legacy_fields(
	identity: Mapping[str, object], scientific: Mapping[str, object]
) -> None:
	"""Reject every non-v5 field, including posterior and M5-LS carry-over."""
	unknown_identity = sorted(
		set(identity) - _XY_NEIGHBOR_CONSENSUS_CHECKPOINT_IDENTITY_FIELDS
	)
	if unknown_identity:
		raise ValueError(
			'XY neighbor consensus checkpoint must not contain legacy or unknown '
			f'identity fields: {unknown_identity!r}'
		)
	unknown_scientific = sorted(
		set(scientific) - _XY_NEIGHBOR_CONSENSUS_SCIENTIFIC_IDENTITY_FIELDS
	)
	if unknown_scientific:
		raise ValueError(
			'XY neighbor consensus scientific identity must not contain legacy or '
			f'unknown fields: {unknown_scientific!r}'
		)


def _reject_xy_neighbor_consensus_fields_from_legacy_identity(
	identity: Mapping[str, object], scientific: Mapping[str, object]
) -> None:
	"""Keep schema-v2 through v4 identities disjoint from the XY successor."""
	forbidden: list[str] = []
	for prefix, values in (
		('checkpoint', identity),
		('scientific_identity', scientific),
	):
		forbidden.extend(
			f'{prefix}.{key}'
			for key in values
			if isinstance(key, str) and key.startswith('xy_neighbor_consensus_')
		)
		forbidden.extend(
			f'{prefix}.{key}'
			for key, expected in (
				('target_representation', _XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION),
				('target_semantics', _XY_NEIGHBOR_CONSENSUS_TARGET_SEMANTICS),
				('consistency_policy', _XY_NEIGHBOR_CONSENSUS_CONSISTENCY_POLICY),
				('experiment_role', _XY_NEIGHBOR_CONSENSUS_EXPERIMENT_ROLE),
			)
			if values.get(key) == expected
		)
	if forbidden:
		raise ValueError(
			'legacy multi-head checkpoint identity must not contain XY neighbor '
			f'consensus fields: {sorted(forbidden)!r}'
		)


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


def _required_positive_finite_number(value: object, label: str) -> float:
	if (
		isinstance(value, bool)
		or not isinstance(value, int | float)
		or not math.isfinite(float(value))
		or value <= 0.0
	):
		raise ValueError(f'{label} must be a positive finite number')
	return float(value)


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


def scientific_identity_sha256(scientific_identity: Mapping[str, object]) -> str:
	"""Return the canonical checksum recorded for a scientific identity."""
	return _canonical_sha256(scientific_identity)


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
		result.append(
			{
				'name': group.get('name'),
				'parameter_names': names,
				'lr': _required_positive_finite_number(
					group.get('lr'),
					f'optimizer {group.get("name")!r} group lr',
				),
			}
		)
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


def _optimizer_state_group_identity_matches(value: object, identity: object) -> bool:
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
			or group.get('lr') != expected.get('lr')
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
		and recorded.get('lr') == expected.get('lr')
		and len(recorded.get('parameter_names', [])) == len(expected.get('params', []))
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


def checkpoint_selection_sha256(selection: Mapping[str, object]) -> str:
	"""Return the stable digest used to bind public selection evidence."""
	canonical = _validated_checkpoint_selection(selection)
	encoded = json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()
	return hashlib.sha256(encoded).hexdigest()


def selected_checkpoint_selection_event(
	selection: Mapping[str, object],
) -> Mapping[str, object]:
	"""Return a validated copy of the selected rolling-checkpoint event."""
	return _validated_checkpoint_selection(selection)['selected']


def _update_checkpoint_selection(  # noqa: PLR0913
	selection: Mapping[str, object] | None,
	*,
	epoch: int,
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch'],
	batch_index: int | None,
	loss: float,
) -> dict[str, object]:
	if selection is None:
		events: list[dict[str, object]] = []
		best_score: float | None = None
	else:
		validated = _validated_checkpoint_selection(selection)
		events = [dict(event) for event in validated['events']]
		selected = validated['selected']
		best_score = float(selected['loss'])
	event = {
		'sequence': len(events),
		'epoch': int(epoch),
		'global_step': int(global_step),
		'checkpoint_kind': checkpoint_kind,
		'batch_index': batch_index,
		'loss': loss,
		'previous_best_score': best_score,
		'best_updated': _is_improved(loss, best_score),
		'best_score_after': loss if _is_improved(loss, best_score) else best_score,
	}
	if any(
		_selection_event_key(existing) == _selection_event_key(event)
		for existing in events
	):
		raise ValueError('checkpoint selection history contains a duplicate event')
	events.append(event)
	selected = event if event['best_updated'] else selected
	return _validated_checkpoint_selection(
		{
			'schema_version': _CHECKPOINT_SELECTION_SCHEMA_VERSION,
			'criterion': _CHECKPOINT_SELECTION_CRITERION,
			'improvement_policy': _CHECKPOINT_SELECTION_POLICY,
			'events': events,
			'selected': _selection_event_identity_from_event(selected),
		}
	)


def _validated_checkpoint_selection(  # noqa: C901, PLR0912
	value: object,
) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('multi-head checkpoint_selection must be a mapping')
	if value.get('schema_version') != _CHECKPOINT_SELECTION_SCHEMA_VERSION:
		raise ValueError('unsupported checkpoint selection schema_version')
	if value.get('criterion') != _CHECKPOINT_SELECTION_CRITERION:
		raise ValueError('checkpoint selection criterion must be metrics.loss')
	if value.get('improvement_policy') != _CHECKPOINT_SELECTION_POLICY:
		raise ValueError('unsupported checkpoint selection improvement policy')
	raw_events = value.get('events')
	if not isinstance(raw_events, list) or not raw_events:
		raise ValueError('checkpoint selection events must be a non-empty list')
	events: list[dict[str, object]] = []
	best_score: float | None = None
	previous_event: Mapping[str, object] | None = None
	keys: set[tuple[int, int, str, int | None]] = set()
	selected: dict[str, object] | None = None
	for sequence, raw_event in enumerate(raw_events):
		if not isinstance(raw_event, Mapping):
			raise TypeError('checkpoint selection event must be a mapping')
		event = _validated_selection_event(raw_event, sequence)
		key = _selection_event_key(event)
		if key in keys:
			raise ValueError('checkpoint selection history contains a duplicate event')
		keys.add(key)
		if previous_event is not None:
			if event['epoch'] < previous_event['epoch']:
				raise ValueError('checkpoint selection event epochs must not regress')
			_selection_event_order(previous_event, event)
		previous_event = event
		if event['previous_best_score'] != best_score:
			raise ValueError('checkpoint selection previous_best_score is inconsistent')
		updated = _is_improved(float(event['loss']), best_score)
		if event['best_updated'] != updated:
			raise ValueError('checkpoint selection best_updated is inconsistent')
		best_score = float(event['loss']) if updated else best_score
		if event['best_score_after'] != best_score:
			raise ValueError('checkpoint selection best_score_after is inconsistent')
		if updated:
			selected = _selection_event_identity_from_event(event)
		events.append(event)
	if selected is None:
		raise ValueError('checkpoint selection history has no selected event')
	if _selection_event_identity(value.get('selected')) != selected:
		raise ValueError('checkpoint selection selected event is inconsistent')
	return {
		'schema_version': _CHECKPOINT_SELECTION_SCHEMA_VERSION,
		'criterion': _CHECKPOINT_SELECTION_CRITERION,
		'improvement_policy': _CHECKPOINT_SELECTION_POLICY,
		'events': events,
		'selected': selected,
	}


def _validate_checkpoint_selection_payload_binding(
	payload: Mapping[str, object], selection: Mapping[str, object]
) -> None:
	"""Require a checkpoint to identify the final event in its history."""
	events = selection['events']
	if not isinstance(events, list) or not events:
		raise ValueError('checkpoint selection events must be a non-empty list')
	last = events[-1]
	if not isinstance(last, Mapping):
		raise TypeError('checkpoint selection final event must be a mapping')
	training_state = payload.get('training_state')
	metrics = payload.get('metrics')
	if not isinstance(training_state, Mapping) or not isinstance(metrics, Mapping):
		raise TypeError(
			'checkpoint payload training_state and metrics must be mappings '
			'for checkpoint selection'
		)
	epoch, global_step = payload.get('epoch'), payload.get('global_step')
	if any(
		isinstance(value, bool) or not isinstance(value, int)
		for value in (epoch, global_step)
	):
		raise TypeError('checkpoint payload epoch/global_step must be integers')
	checkpoint_kind = training_state.get('checkpoint_kind')
	batch_index = training_state.get('batch_index')
	if checkpoint_kind not in {'step', 'epoch'}:
		raise ValueError('checkpoint payload checkpoint_kind must be step or epoch')
	if checkpoint_kind == 'step' and (
		isinstance(batch_index, bool)
		or not isinstance(batch_index, int)
		or batch_index < 0
	):
		raise TypeError(
			'checkpoint payload step batch_index must be a nonnegative integer'
		)
	if checkpoint_kind == 'epoch' and batch_index is not None:
		raise ValueError('checkpoint payload epoch batch_index must be null')
	loss = _finite_selection_number(metrics.get('loss'), 'payload metrics.loss')
	if (
		epoch != last['epoch']
		or global_step != last['global_step']
		or checkpoint_kind != last['checkpoint_kind']
		or batch_index != last['batch_index']
		or loss != last['loss']
	):
		raise ValueError(
			'checkpoint payload does not match final checkpoint selection event'
		)


def _validated_selection_event(
	value: Mapping[str, object], sequence: int
) -> dict[str, object]:
	if value.get('sequence') != sequence:
		raise ValueError('checkpoint selection event sequence is not contiguous')
	epoch, global_step = value.get('epoch'), value.get('global_step')
	if any(
		isinstance(item, bool) or not isinstance(item, int)
		for item in (epoch, global_step)
	):
		raise TypeError('checkpoint selection epoch/global_step must be integers')
	if epoch < 0 or global_step < 0:
		raise ValueError('checkpoint selection counters must be nonnegative')
	kind = value.get('checkpoint_kind')
	if kind not in {'step', 'epoch'}:
		raise ValueError('checkpoint selection checkpoint_kind must be step or epoch')
	batch_index = value.get('batch_index')
	if kind == 'step' and (
		isinstance(batch_index, bool)
		or not isinstance(batch_index, int)
		or batch_index < 0
	):
		raise TypeError('checkpoint selection step batch_index must be nonnegative')
	if kind == 'epoch' and batch_index is not None:
		raise ValueError('checkpoint selection epoch batch_index must be null')
	loss = _finite_selection_number(value.get('loss'), 'loss')
	previous = _nullable_selection_number(
		value.get('previous_best_score'), 'previous_best_score'
	)
	after = _nullable_selection_number(
		value.get('best_score_after'), 'best_score_after'
	)
	if not isinstance(value.get('best_updated'), bool):
		raise TypeError('checkpoint selection best_updated must be a boolean')
	return {
		'sequence': sequence,
		'epoch': epoch,
		'global_step': global_step,
		'checkpoint_kind': kind,
		'batch_index': batch_index,
		'loss': loss,
		'previous_best_score': previous,
		'best_updated': value['best_updated'],
		'best_score_after': after,
	}


def _finite_selection_number(value: object, label: str) -> float:
	if (
		isinstance(value, bool)
		or not isinstance(value, int | float)
		or not math.isfinite(float(value))
	):
		raise ValueError(f'checkpoint selection {label} must be finite')
	return float(value)


def _nullable_selection_number(value: object, label: str) -> float | None:
	return None if value is None else _finite_selection_number(value, label)


def _selection_event_key(
	event: Mapping[str, object],
) -> tuple[int, int, str, int | None]:
	return (
		int(event['epoch']),
		int(event['global_step']),
		str(event['checkpoint_kind']),
		event['batch_index'] if isinstance(event['batch_index'], int) else None,
	)


def _selection_event_order(
	previous: Mapping[str, object], event: Mapping[str, object]
) -> None:
	"""Require chronological checkpoints and the sole valid repeated-step pair."""
	previous_step = int(previous['global_step'])
	event_step = int(event['global_step'])
	if event_step < previous_step:
		raise ValueError('checkpoint selection events are not chronological')
	if event_step > previous_step:
		return
	if (
		event['epoch'] != previous['epoch']
		or previous['checkpoint_kind'] != 'step'
		or event['checkpoint_kind'] != 'epoch'
	):
		raise ValueError(
			'checkpoint selection repeated global_step must be a '
			'same-epoch step-to-epoch pair'
		)


def _selection_event_identity(value: object) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint selection selected event must be a mapping')
	keys = (
		'sequence',
		'epoch',
		'global_step',
		'checkpoint_kind',
		'batch_index',
		'loss',
	)
	if set(value) != set(keys):
		raise ValueError('checkpoint selection selected event identity is invalid')
	sequence, epoch, global_step = (
		value['sequence'],
		value['epoch'],
		value['global_step'],
	)
	if any(
		isinstance(item, bool) or not isinstance(item, int)
		for item in (sequence, epoch, global_step)
	):
		raise TypeError(
			'checkpoint selection selected event sequence/epoch/global_step '
			'must be integers'
		)
	if any(item < 0 for item in (sequence, epoch, global_step)):
		raise ValueError(
			'checkpoint selection selected event counters must be nonnegative'
		)
	kind = value['checkpoint_kind']
	if kind not in {'step', 'epoch'}:
		raise ValueError(
			'checkpoint selection selected event checkpoint_kind must be step or epoch'
		)
	batch_index = value['batch_index']
	if kind == 'step' and (
		isinstance(batch_index, bool)
		or not isinstance(batch_index, int)
		or batch_index < 0
	):
		raise TypeError(
			'checkpoint selection selected step batch_index must be a '
			'nonnegative integer'
		)
	if kind == 'epoch' and batch_index is not None:
		raise ValueError('checkpoint selection selected epoch batch_index must be null')
	return {
		'sequence': sequence,
		'epoch': epoch,
		'global_step': global_step,
		'checkpoint_kind': kind,
		'batch_index': batch_index,
		'loss': _finite_selection_number(value['loss'], 'selected event loss'),
	}


def _selection_event_identity_from_event(
	value: Mapping[str, object],
) -> dict[str, object]:
	keys = (
		'sequence',
		'epoch',
		'global_step',
		'checkpoint_kind',
		'batch_index',
		'loss',
	)
	return _selection_event_identity({key: value[key] for key in keys})


def _checkpoint_selection_transaction_path(checkpoint_root: Path) -> Path:
	return checkpoint_root / _CHECKPOINT_SELECTION_TRANSACTION_NAME


def _write_checkpoint_selection_transaction(
	checkpoint_root: Path, selection: Mapping[str, object]
) -> None:
	canonical = _validated_checkpoint_selection(selection)
	events = canonical['events']
	if not isinstance(events, list) or not isinstance(events[-1], Mapping):
		raise TypeError('checkpoint selection transaction events are invalid')
	selected = selected_checkpoint_selection_event(canonical)
	last = events[-1]
	if (
		_selection_event_identity_from_event(last) != selected
		or last.get('best_updated') is not True
	):
		raise ValueError('checkpoint selection transaction does not select latest')
	_atomic_json(
		_checkpoint_selection_transaction_path(checkpoint_root),
		{
			'schema_version': _CHECKPOINT_SELECTION_TRANSACTION_SCHEMA_VERSION,
			'selection_sha256': checkpoint_selection_sha256(canonical),
			'selected': selected,
		},
	)


def _load_checkpoint_selection_transaction(path: Path) -> dict[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError('checkpoint selection transaction is unreadable') from error
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint selection transaction must be a mapping')
	if value.get('schema_version') != _CHECKPOINT_SELECTION_TRANSACTION_SCHEMA_VERSION:
		raise ValueError('unsupported checkpoint selection transaction schema_version')
	digest = value.get('selection_sha256')
	if not isinstance(digest, str) or len(digest) != 64:
		raise TypeError('checkpoint selection transaction digest is invalid')
	if any(character not in '0123456789abcdef' for character in digest):
		raise ValueError('checkpoint selection transaction digest is invalid')
	return {
		'selection_sha256': digest,
		'selected': _selection_event_identity(value.get('selected')),
	}


def _checkpoint_payload_matches_selected_event(
	path: Path, selected: Mapping[str, object]
) -> bool:
	if not path.is_file():
		return False
	try:
		payload = load_checkpoint(path, map_location='cpu')
		selection = _validated_checkpoint_selection(payload.get('checkpoint_selection'))
		_validate_checkpoint_selection_payload_binding(payload, selection)
		events = selection['events']
		return (
			isinstance(events, list)
			and isinstance(events[-1], Mapping)
			and selected_checkpoint_selection_event(selection) == selected
			and _selection_event_identity_from_event(events[-1]) == selected
		)
	except (OSError, RuntimeError, TypeError, ValueError):
		return False


def _required_loss_score(metrics: Mapping[str, float]) -> float:
	score = _loss_score(metrics)
	if score is None:
		raise ValueError('checkpoint selection requires finite metrics.loss')
	return score


def _write_checkpoint_selection_reports(
	checkpoint_root: Path, selection: Mapping[str, object]
) -> None:
	canonical = _validated_checkpoint_selection(selection)
	_atomic_json(checkpoint_root / 'checkpoint_selection_summary.json', canonical)
	fieldnames = [
		'sequence',
		'epoch',
		'global_step',
		'checkpoint_kind',
		'batch_index',
		'loss',
		'previous_best_score',
		'best_updated',
		'best_score_after',
	]
	fd, name = tempfile.mkstemp(
		prefix='.checkpoint_selection_history.', suffix='.tmp', dir=checkpoint_root
	)
	temporary = Path(name)
	try:
		with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
			writer = csv.DictWriter(handle, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(canonical['events'])
			handle.flush()
			os.fsync(handle.fileno())
		temporary.replace(checkpoint_root / 'checkpoint_selection_history.csv')
	finally:
		if temporary.exists():
			temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
	fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
	temporary = Path(name)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as handle:
			json.dump(value, handle, sort_keys=True, indent=2)
			handle.write('\n')
			handle.flush()
			os.fsync(handle.fileno())
		temporary.replace(path)
	finally:
		if temporary.exists():
			temporary.unlink()


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
	'recover_strat_hmm_rolling_checkpoint',
	'save_strat_hmm_checkpoint',
	'save_strat_hmm_rolling_checkpoint',
]
