"""Component construction for stratigraphic HMM pretext training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from seis_ssl_cluster.embedding.extractor import build_model_from_config
from seis_ssl_cluster.stratigraphy import OrderedPrototypeHead
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm.runtime import (
	_bool_config,
	_float_config,
	_int_config,
	_mapping,
	_non_empty_string,
	_optional_int_config,
	_path_config,
)
from seis_ssl_cluster.training.strat_hmm.state import (
	StratHmmHeadOnlyComponents,
	TrainabilitySummary,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.models.mae import AmplitudeMAE3D

MODEL_GEOMETRY_KEYS = (
	'name',
	'in_channels',
	'out_channels',
	'patch_size',
	'encoder_dim',
	'encoder_depth',
	'encoder_heads',
	'decoder_dim',
	'decoder_depth',
	'decoder_heads',
)


def build_strat_hmm_head_only_components(
	config: Mapping[str, object],
	*,
	device: torch.device | str,
) -> StratHmmHeadOnlyComponents:
	"""Build student MAE, optional teacher, ordered prototype head, and optimizer."""
	resolved_device = torch.device(device)
	student_config = _mapping(config, 'student')
	loss_config = _mapping(config, 'loss')
	unfreeze_top_blocks = _int_config(
		student_config,
		'unfreeze_top_blocks',
		0,
	)
	distillation_weight = _float_config(loss_config, 'distillation_weight', 0.0)
	prototype_head_used = (
		_float_config(loss_config, 'prototype_weight', 1.0) > 0.0
		or _float_config(loss_config, 'usage_weight', 0.0) > 0.0
	)
	if unfreeze_top_blocks > 0 and distillation_weight <= 0.0:
		msg = (
			'loss.distillation_weight must be positive when '
			'student.unfreeze_top_blocks is greater than 0'
		)
		raise ValueError(msg)
	teacher_payload = load_checkpoint(
		_path_config(_mapping(config, 'teacher'), 'checkpoint'),
		map_location='cpu',
	)
	teacher_config = _checkpoint_config(teacher_payload)
	_verify_model_geometry(teacher_config, _mapping(config, 'model'))
	student = cast('AmplitudeMAE3D', build_model_from_config(teacher_config))
	init_checkpoint = _student_init_checkpoint(config)
	init_payload = load_checkpoint(init_checkpoint, map_location='cpu')
	student.load_state_dict(_model_state_dict(init_payload))
	trainability_summary = configure_student_trainability(
		student,
		unfreeze_top_blocks=unfreeze_top_blocks,
	)
	student.to(resolved_device)
	student.eval()

	teacher = None
	if distillation_weight > 0.0:
		teacher = cast('AmplitudeMAE3D', build_model_from_config(teacher_config))
		teacher.load_state_dict(_model_state_dict(teacher_payload))
		teacher.requires_grad_(requires_grad=False)
		teacher.to(resolved_device)
		teacher.eval()

	head_config = _mapping(config, 'head')
	head = OrderedPrototypeHead(
		feature_dim=student.encoder_dim,
		num_prototypes=_int_config(head_config, 'num_prototypes', 1),
		projection_dim=_optional_int_config(head_config, 'projection_dim'),
		temperature=_float_config(head_config, 'temperature', 0.1),
		normalize=_bool_config(head_config, 'normalize', default=True),
	).to(resolved_device)
	if not prototype_head_used:
		head.requires_grad_(requires_grad=False)
	param_groups = _optimizer_param_groups(
		student=student,
		head=head,
		train_config=_mapping(config, 'train'),
	)
	if not param_groups:
		msg = 'ordered prototype head has no trainable parameters'
		raise ValueError(msg)
	train_config = _mapping(config, 'train')
	optimizer = torch.optim.AdamW(
		param_groups,
		weight_decay=_float_config(train_config, 'weight_decay', 0.05),
	)
	return StratHmmHeadOnlyComponents(
		student=student,
		teacher=teacher,
		head=head,
		optimizer=optimizer,
		mae_checkpoint_config=_extraction_compatible_config(
			teacher_config,
			output_root=_path_config(_mapping(config, 'paths'), 'output_root'),
			strat_data_config=_mapping(config, 'data'),
			strat_zero_mask_config=_mapping(config, 'zero_mask'),
		),
		trainability_summary=trainability_summary,
	)


def configure_student_trainability(
	model: AmplitudeMAE3D,
	*,
	unfreeze_top_blocks: int,
) -> TrainabilitySummary:
	"""Freeze all student params, then unfreeze only the last N encoder blocks."""
	if isinstance(unfreeze_top_blocks, bool) or not isinstance(
		unfreeze_top_blocks,
		int,
	):
		msg = f'unfreeze_top_blocks must be an integer; got {unfreeze_top_blocks!r}'
		raise TypeError(msg)
	if unfreeze_top_blocks < 0:
		msg = f'unfreeze_top_blocks must be nonnegative; got {unfreeze_top_blocks!r}'
		raise ValueError(msg)
	if unfreeze_top_blocks > model.encoder.depth:
		msg = (
			'unfreeze_top_blocks must be less than or equal to '
			f'model.encoder.depth ({model.encoder.depth}); got {unfreeze_top_blocks}'
		)
		raise ValueError(msg)

	for parameter in model.parameters():
		parameter.requires_grad_(requires_grad=False)
	if unfreeze_top_blocks > 0:
		for layer in model.encoder.layers[-unfreeze_top_blocks:]:
			for parameter in layer.parameters():
				parameter.requires_grad_(requires_grad=True)

	trainable_names = tuple(
		name for name, parameter in model.named_parameters() if parameter.requires_grad
	)
	trainable_count = sum(
		parameter.numel() for parameter in model.parameters() if parameter.requires_grad
	)
	frozen_count = sum(
		parameter.numel()
		for parameter in model.parameters()
		if not parameter.requires_grad
	)
	return TrainabilitySummary(
		trainable_parameter_count=trainable_count,
		frozen_parameter_count=frozen_count,
		trainable_names=trainable_names,
	)


def _optimizer_param_groups(
	*,
	student: AmplitudeMAE3D,
	head: OrderedPrototypeHead,
	train_config: Mapping[str, object],
) -> list[dict[str, object]]:
	head_params = [
		parameter for parameter in head.parameters() if parameter.requires_grad
	]
	encoder_params = [
		parameter
		for parameter in student.encoder.parameters()
		if parameter.requires_grad
	]
	param_groups: list[dict[str, object]] = []
	if head_params:
		param_groups.append(
			{
				'params': head_params,
				'lr': _float_config(train_config, 'lr', 3.0e-4),
				'name': 'head',
			},
		)
	if encoder_params:
		param_groups.append(
			{
				'params': encoder_params,
				'lr': _float_config(train_config, 'encoder_lr', 3.0e-5),
				'name': 'encoder',
			},
		)
	_trainable_ids = {
		id(parameter)
		for module in (student, head)
		for parameter in module.parameters()
		if parameter.requires_grad
	}
	_group_ids = {
		id(parameter)
		for group in param_groups
		for parameter in cast('Sequence[torch.nn.Parameter]', group['params'])
	}
	if _group_ids != _trainable_ids:
		msg = 'optimizer parameter groups do not exactly match trainable parameters'
		raise ValueError(msg)
	return param_groups


def _trainability_metrics(summary: TrainabilitySummary) -> dict[str, float]:
	return {
		'trainable_parameter_count': float(summary.trainable_parameter_count),
		'frozen_parameter_count': float(summary.frozen_parameter_count),
	}


def _checkpoint_config(payload: Mapping[str, object]) -> Mapping[str, object]:
	value = payload.get('config')
	if not isinstance(value, Mapping):
		msg = 'teacher checkpoint is missing MAE resolved config'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _model_state_dict(payload: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
	value = payload.get('model_state_dict')
	if not isinstance(value, Mapping):
		msg = 'checkpoint is missing model_state_dict'
		raise TypeError(msg)
	return cast('Mapping[str, torch.Tensor]', value)


def _verify_model_geometry(
	teacher_config: Mapping[str, object],
	resolved_model_config: Mapping[str, object],
) -> None:
	teacher_model = _mapping(teacher_config, 'model')
	mismatches = [
		f'{key}: checkpoint={teacher_model.get(key)!r}, '
		f'resolved={resolved_model_config.get(key)!r}'
		for key in MODEL_GEOMETRY_KEYS
		if _geometry_value(teacher_model.get(key))
		!= _geometry_value(
			resolved_model_config.get(key),
		)
	]
	if mismatches:
		msg = 'teacher checkpoint model geometry does not match config: '
		msg += '; '.join(mismatches)
		raise ValueError(msg)


def _geometry_value(value: object) -> object:
	if isinstance(value, tuple):
		return list(value)
	return value


def _student_init_checkpoint(config: Mapping[str, object]) -> Path:
	student = _mapping(config, 'student')
	value = student.get('init_checkpoint')
	if value is None:
		return _path_config(_mapping(config, 'teacher'), 'checkpoint')
	return Path(_non_empty_string(value, 'student.init_checkpoint'))


def _extraction_compatible_config(
	teacher_config: Mapping[str, object],
	*,
	output_root: Path,
	strat_data_config: Mapping[str, object],
	strat_zero_mask_config: Mapping[str, object],
) -> Mapping[str, object]:
	result = deepcopy(dict(teacher_config))
	paths = dict(_mapping(result, 'paths'))
	paths['output_root'] = str(output_root)
	result['paths'] = paths
	data = dict(_mapping(result, 'data'))
	for key in ('normalized_clip_abs', 'amplitude_agc'):
		if key in strat_data_config:
			data[key] = deepcopy(strat_data_config[key])
	result['data'] = data
	result['zero_mask'] = deepcopy(dict(strat_zero_mask_config))
	return result


__all__ = [
	'StratHmmHeadOnlyComponents',
	'TrainabilitySummary',
	'build_strat_hmm_head_only_components',
	'configure_student_trainability',
]
