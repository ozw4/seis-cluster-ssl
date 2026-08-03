"""Component construction for stratigraphic HMM pretext training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from seis_ssl_cluster.embedding.extractor import build_model_from_config
from seis_ssl_cluster.models.mae import LearnedEncoderReplacementToken
from seis_ssl_cluster.stratigraphy import (
	MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1,
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm.masking import (
	center_trace_replacement_token_seed,
)
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
	StratHmmCenterTraceMaskedComponents,
	StratHmmHeadOnlyComponents,
	StratHmmMultiHeadComponents,
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
	loss_config = _mapping(config, 'loss')
	student, teacher, teacher_config, trainability_summary = _build_student_teacher(
		config,
		device=device,
	)
	prototype_head_used = (
		_float_config(loss_config, 'prototype_weight', 1.0) > 0.0
		or _float_config(loss_config, 'usage_weight', 0.0) > 0.0
	)
	head_config = _mapping(config, 'head')
	head = OrderedPrototypeHead(
		feature_dim=student.encoder_dim,
		num_prototypes=_int_config(head_config, 'num_prototypes', 1),
		projection_dim=_optional_int_config(head_config, 'projection_dim'),
		temperature=_float_config(head_config, 'temperature', 0.1),
		normalize=_bool_config(head_config, 'normalize', default=True),
	).to(device)
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


def build_strat_hmm_components(
	config: Mapping[str, object],
	*,
	device: torch.device | str,
) -> StratHmmHeadOnlyComponents | StratHmmMultiHeadComponents:
	"""Build the configured single-head or multi-resolution training components."""
	if isinstance(config.get('spatial_context'), Mapping):
		return build_strat_hmm_center_trace_masked_components(config, device=device)
	head_config = _mapping(config, 'head')
	if 'spec' not in head_config:
		return build_strat_hmm_head_only_components(config, device=device)
	if head_config.get('spec') != MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1:
		msg = f'unsupported strat HMM head.spec: {head_config.get("spec")!r}'
		raise ValueError(msg)
	return build_strat_hmm_multi_head_components(config, device=device)


def build_strat_hmm_multi_head_components(
	config: Mapping[str, object],
	*,
	device: torch.device | str,
) -> StratHmmMultiHeadComponents:
	"""Build shared MAE components and independent heads for every resolution."""
	student, teacher, teacher_config, trainability_summary = _build_student_teacher(
		config,
		device=device,
	)
	head_config = _mapping(config, 'head')
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=student.encoder_dim,
		ks=cast('Sequence[int]', head_config['ks']),
		projection_dim=_optional_int_config(head_config, 'projection_dim'),
		temperature=_float_config(head_config, 'temperature', 0.1),
		normalize=_bool_config(head_config, 'normalize', default=True),
	).to(device)
	train_config = _mapping(config, 'train')
	param_groups = _optimizer_param_groups(
		student=student,
		head=heads,
		train_config=train_config,
	)
	if not param_groups:
		raise ValueError(
			'multi-resolution ordered prototype heads have no trainable parameters'
		)
	optimizer = torch.optim.AdamW(
		param_groups,
		weight_decay=_float_config(train_config, 'weight_decay', 0.05),
	)
	return StratHmmMultiHeadComponents(
		student=student,
		teacher=teacher,
		heads=heads,
		optimizer=optimizer,
		mae_checkpoint_config=_extraction_compatible_config(
			teacher_config,
			output_root=_path_config(_mapping(config, 'paths'), 'output_root'),
			strat_data_config=_mapping(config, 'data'),
			strat_zero_mask_config=_mapping(config, 'zero_mask'),
		),
		trainability_summary=trainability_summary,
		head_spec=MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1,
		head_ks=heads.head_ks,
	)


def build_strat_hmm_center_trace_masked_components(
	config: Mapping[str, object],
	*,
	device: torch.device | str,
	replacement_token_seed: int | None = None,
) -> StratHmmCenterTraceMaskedComponents:
	"""Build the isolated center-trace masked multi-head training components."""
	student, teacher, teacher_config, trainability_summary = _build_student_teacher(
		config,
		device=device,
	)
	head_config = _mapping(config, 'head')
	if head_config.get('spec') != MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1:
		msg = f'unsupported strat HMM head.spec: {head_config.get("spec")!r}'
		raise ValueError(msg)
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=student.encoder_dim,
		ks=cast('Sequence[int]', head_config['ks']),
		projection_dim=_optional_int_config(head_config, 'projection_dim'),
		temperature=_float_config(head_config, 'temperature', 0.1),
		normalize=_bool_config(head_config, 'normalize', default=True),
	).to(device)
	if heads.head_ks != (6, 8, 10):
		raise ValueError(
			'center-trace masked training requires heads K=(6, 8, 10); '
			f'got {heads.head_ks!r}'
		)

	train_config = _mapping(config, 'train')
	training_seed = _int_config(train_config, 'seed', 42)
	if replacement_token_seed is None:
		replacement_token_seed = center_trace_replacement_token_seed(training_seed)
	replacement_token = LearnedEncoderReplacementToken(
		student.encoder_dim,
		seed=replacement_token_seed,
		device=device,
		dtype=_module_dtype(student),
	)
	param_groups = _center_trace_optimizer_param_groups(
		student=student,
		head=heads,
		replacement_token=replacement_token,
		train_config=train_config,
	)
	optimizer = torch.optim.AdamW(
		param_groups,
		weight_decay=_float_config(train_config, 'weight_decay', 0.05),
	)
	return StratHmmCenterTraceMaskedComponents(
		student=student,
		teacher=teacher,
		heads=heads,
		replacement_token=replacement_token,
		optimizer=optimizer,
		mae_checkpoint_config=_extraction_compatible_config(
			teacher_config,
			output_root=_path_config(_mapping(config, 'paths'), 'output_root'),
			strat_data_config=_mapping(config, 'data'),
			strat_zero_mask_config=_mapping(config, 'zero_mask'),
		),
		trainability_summary=trainability_summary,
		head_spec=MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1,
		head_ks=heads.head_ks,
	)


def _build_student_teacher(
	config: Mapping[str, object],
	*,
	device: torch.device | str,
) -> tuple[
	AmplitudeMAE3D, AmplitudeMAE3D | None, Mapping[str, object], TrainabilitySummary
]:
	"""Build the shared MAE teacher/student pair for either head mode."""
	resolved_device = torch.device(device)
	student_config = _mapping(config, 'student')
	distillation_weight = _float_config(
		_mapping(config, 'loss'), 'distillation_weight', 0.0
	)
	unfreeze_top_blocks = _int_config(student_config, 'unfreeze_top_blocks', 0)
	if unfreeze_top_blocks > 0 and distillation_weight <= 0.0:
		raise ValueError(
			'loss.distillation_weight must be positive when '
			'student.unfreeze_top_blocks is greater than 0',
		)
	teacher_payload = load_checkpoint(
		_path_config(_mapping(config, 'teacher'), 'checkpoint'), map_location='cpu'
	)
	teacher_config = _checkpoint_config(teacher_payload)
	_verify_model_geometry(teacher_config, _mapping(config, 'model'))
	student = cast('AmplitudeMAE3D', build_model_from_config(teacher_config))
	student.load_state_dict(
		_model_state_dict(
			load_checkpoint(_student_init_checkpoint(config), map_location='cpu')
		)
	)
	trainability_summary = configure_student_trainability(
		student, unfreeze_top_blocks=unfreeze_top_blocks
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
	return student, teacher, teacher_config, trainability_summary


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
	head: torch.nn.Module,
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
	group_parameters = [
		parameter
		for group in param_groups
		for parameter in cast('Sequence[torch.nn.Parameter]', group['params'])
	]
	_group_ids = {id(parameter) for parameter in group_parameters}
	if len(_group_ids) != len(group_parameters):
		raise ValueError('optimizer parameter groups contain duplicate parameters')
	if _group_ids != _trainable_ids:
		msg = 'optimizer parameter groups do not exactly match trainable parameters'
		raise ValueError(msg)
	encoder_ids = {id(parameter) for parameter in encoder_params}
	if any(id(parameter) in encoder_ids for parameter in head_params):
		raise ValueError('head and encoder optimizer parameters must not overlap')
	if any(
		parameter.requires_grad
		for parameter in student.parameters()
		if id(parameter) not in encoder_ids
	):
		raise ValueError('only encoder parameters may be trainable in the student')
	return param_groups


def _center_trace_optimizer_param_groups(
	*,
	student: AmplitudeMAE3D,
	head: torch.nn.Module,
	replacement_token: LearnedEncoderReplacementToken,
	train_config: Mapping[str, object],
) -> list[dict[str, object]]:
	"""Return the fixed head/encoder/spatial-context optimizer partition."""
	head_params = [
		parameter for parameter in head.parameters() if parameter.requires_grad
	]
	encoder_params = [
		parameter
		for parameter in student.encoder.parameters()
		if parameter.requires_grad
	]
	spatial_context_params = [
		parameter
		for parameter in replacement_token.parameters()
		if parameter.requires_grad
	]
	if not head_params:
		raise ValueError('center-trace masked heads have no trainable parameters')
	if not encoder_params:
		raise ValueError(
			'center-trace masked student has no trainable encoder parameters'
		)
	if not spatial_context_params:
		raise ValueError(
			'center-trace masked replacement token has no trainable parameters'
		)
	param_groups: list[dict[str, object]] = [
		{
			'params': head_params,
			'lr': _float_config(train_config, 'lr', 3.0e-4),
			'name': 'head',
		},
		{
			'params': encoder_params,
			'lr': _float_config(train_config, 'encoder_lr', 3.0e-5),
			'name': 'encoder',
		},
		{
			'params': spatial_context_params,
			'lr': _float_config(train_config, 'lr', 3.0e-4),
			'name': 'spatial_context',
		},
	]
	trainable_ids = {
		id(parameter)
		for module in (student, head, replacement_token)
		for parameter in module.parameters()
		if parameter.requires_grad
	}
	group_parameters = [
		parameter
		for group in param_groups
		for parameter in cast('Sequence[torch.nn.Parameter]', group['params'])
	]
	group_ids = {id(parameter) for parameter in group_parameters}
	if len(group_ids) != len(group_parameters):
		raise ValueError('center-trace optimizer parameter groups contain duplicates')
	if group_ids != trainable_ids:
		raise ValueError(
			'center-trace optimizer parameter groups do not exactly match trainable '
			'parameters'
		)
	return param_groups


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
	try:
		parameter = next(module.parameters())
	except StopIteration as exc:
		raise ValueError('student must contain at least one parameter') from exc
	if not parameter.dtype.is_floating_point:
		raise TypeError('student parameters must have a floating dtype')
	return parameter.dtype


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
	'StratHmmCenterTraceMaskedComponents',
	'StratHmmHeadOnlyComponents',
	'StratHmmMultiHeadComponents',
	'TrainabilitySummary',
	'build_strat_hmm_center_trace_masked_components',
	'build_strat_hmm_components',
	'build_strat_hmm_head_only_components',
	'build_strat_hmm_multi_head_components',
	'configure_student_trainability',
]
