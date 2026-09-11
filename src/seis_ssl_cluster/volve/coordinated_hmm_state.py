"""CPU-only model/AdamW contracts for the exact FP32 Volve Random HMM arms."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch

from seis_ssl_cluster.models.amplitude_encoder_factory import build_model_from_config
from seis_ssl_cluster.stratigraphy.prototypes import OrderedPrototypeHead
from seis_ssl_cluster.training.strat_hmm.components import _extraction_compatible_config
from seis_ssl_cluster.training.strat_hmm.runner import (
	_parameter_sha256,
	_state_dict_sha256,
)


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('coordinated Volve HMM state must be a mapping')
	return value


def _model_template(config: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
	# Preserve caller CPU RNG and never query/restore CUDA RNG while waiting.
	with torch.random.fork_rng(devices=[]):
		return build_model_from_config(config).state_dict()


def _initial_head(config: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
	"""Reproduce student, teacher, then prototype-head construction on CPU."""
	with torch.random.fork_rng(devices=[]), torch.device('cpu'):
		torch.random.default_generator.manual_seed(42)
		build_model_from_config(config)
		build_model_from_config(config)
		return OrderedPrototypeHead(
			feature_dim=384,
			num_prototypes=6,
			projection_dim=128,
			temperature=0.1,
			normalize=True,
		).state_dict()


def _tensor(value: object, shape: tuple[int, ...]) -> torch.Tensor:
	if (
		not isinstance(value, torch.Tensor)
		or value.device.type != 'cpu'
		or value.dtype != torch.float32
		or tuple(value.shape) != shape
		or not torch.isfinite(value).all()
	):
		raise ValueError('coordinated Volve HMM tensor shape/dtype/finiteness differs')
	return value


def _parent(config: Mapping[str, object]) -> Mapping[str, object]:
	path = Path(str(_mapping(config['student'])['init_checkpoint']))
	payload = _mapping(
		torch.load(path, map_location='cpu', mmap=True, weights_only=False)
	)
	metadata = _mapping(payload.get('metadata'))
	state = _mapping(payload.get('training_state'))
	if (
		payload.get('epoch') != 0
		or payload.get('global_step') != 0
		or metadata.get('random_encoder_baseline') is not True
		or metadata.get('pretrained_weights_loaded') is not False
		or metadata.get('seed') != 42
		or state.get('stage') != 'create_random_mae_checkpoint'
		or state.get('checkpoint_kind') != 'random_init'
		or _mapping(payload['config'])['model'] != config['model']
	):
		raise ValueError(
			'coordinated Volve HMM parent is not the declared Random encoder'
		)
	return payload


def _model_parameters(
	payload: Mapping[str, object],
	config: Mapping[str, object],
) -> list[torch.Tensor]:
	parent = _parent(config)
	parent_config = _mapping(parent['config'])
	expected_config = _extraction_compatible_config(
		parent_config,
		output_root=Path(str(_mapping(config['paths'])['output_root'])),
		strat_data_config=_mapping(config['data']),
		strat_zero_mask_config=_mapping(config['zero_mask']),
	)
	if payload['config'] != expected_config:
		raise ValueError('coordinated Volve HMM extraction-compatible config differs')
	template = _model_template(parent_config)
	model, original = (
		_mapping(payload['model_state_dict']),
		_mapping(parent['model_state_dict']),
	)
	if set(model) != set(template) or set(original) != set(template):
		raise ValueError(
			'coordinated Volve HMM model state keys differ from architecture'
		)
	trainable = []
	for name, expected in template.items():
		value = _tensor(model[name], tuple(expected.shape))
		initial = _tensor(original[name], tuple(expected.shape))
		if name.startswith('encoder.layers.7.'):
			trainable.append(name)
		elif not torch.equal(value, initial):
			raise ValueError('coordinated Volve HMM frozen model tensor changed')
	if len(trainable) != 12:
		raise ValueError('coordinated Volve HMM requires exactly the top encoder block')
	head_shapes = {
		'prototypes': (6, 128),
		'projection.weight': (128, 384),
		'projection.bias': (128,),
	}
	head = _mapping(payload['stratigraphy_state_dict'])
	if set(head) != set(head_shapes):
		raise ValueError('coordinated Volve HMM prototype head keys differ')
	parameters = [_tensor(head[name], shape) for name, shape in head_shapes.items()]
	parameters.extend(model[name] for name in trainable)
	expected_trainability = {
		'trainable_parameter_count': sum(model[name].numel() for name in trainable),
		'frozen_parameter_count': sum(
			model[name].numel() for name in model if name not in trainable
		),
		'trainable_names': trainable,
	}
	if payload.get('trainability_summary') != expected_trainability:
		raise ValueError('coordinated Volve HMM trainability summary differs')
	control = _mapping(payload['control_identity'])
	initial = _mapping(control.get('initial_parameter_sha256'))
	initial_state = _mapping(control.get('initial_state_sha256'))
	expected_head = _initial_head(parent_config)
	if (
		set(initial) != {'student_trainable', 'prototype_head'}
		or set(initial_state) != {'student', 'head'}
		or initial.get('student_trainable')
		!= _parameter_sha256((name, original[name]) for name in trainable)
		or initial_state.get('student') != _state_dict_sha256(original)
		or initial.get('prototype_head') != _parameter_sha256(expected_head.items())
		or initial_state.get('head') != _state_dict_sha256(expected_head)
	):
		raise ValueError('coordinated Volve HMM initial student provenance differs')
	return parameters


def validate_parent(config: Mapping[str, object]) -> None:
	"""Require a real finite Random encoder before starting a fresh GPU run."""
	parent = _parent(config)
	template = _model_template(_mapping(parent['config']))
	model = _mapping(parent['model_state_dict'])
	if set(model) != set(template):
		raise ValueError('coordinated Volve HMM Random model keys differ')
	for name, value in template.items():
		_tensor(model[name], tuple(value.shape))


def _optimizer(  # noqa: C901 - Validate each independent AdamW invariant explicitly.
	payload: Mapping[str, object],
	config: Mapping[str, object],
	parameters: list[torch.Tensor],
) -> None:
	optimizer = _mapping(payload['optimizer_state_dict'])
	groups = optimizer.get('param_groups')
	states = _mapping(optimizer.get('state'))
	if not isinstance(groups, list) or len(groups) != 2:
		raise ValueError('coordinated Volve HMM requires head and encoder AdamW groups')
	train = _mapping(config['train'])
	for group, name, indices, lr in zip(
		groups,
		('head', 'encoder'),
		(list(range(3)), list(range(3, 15))),
		(train['lr'], train['encoder_lr']),
		strict=True,
	):
		expected = {
			'name': name,
			'params': indices,
			'lr': lr,
			'weight_decay': train['weight_decay'],
			'betas': (0.9, 0.999),
			'eps': 1e-8,
			'amsgrad': False,
			'maximize': False,
			'capturable': False,
			'differentiable': False,
			'foreach': None,
			'fused': None,
		}
		# Older PyTorch AdamW implementations omit this otherwise explicit flag.
		if 'decoupled_weight_decay' in _mapping(group):
			if group['decoupled_weight_decay'] is not True:
				raise ValueError('coordinated Volve HMM requires decoupled AdamW decay')
			expected['decoupled_weight_decay'] = True
		if _mapping(group) != expected or any(
			type(i) is not int for i in group['params']
		):
			raise ValueError('coordinated Volve HMM AdamW parameter groups differ')
	if set(states) != set(range(15)) or any(type(index) is not int for index in states):
		raise ValueError('coordinated Volve HMM AdamW parameter identities differ')
	for index, parameter in enumerate(parameters):
		state = _mapping(states[index])
		if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
			raise ValueError('coordinated Volve HMM AdamW moment keys differ')
		step = _tensor(state['step'], ())
		if not math.isfinite(step.item()) or step.item() != payload['global_step']:
			raise ValueError(
				'coordinated Volve FP32 AdamW step differs from global step'
			)
		_tensor(state['exp_avg'], tuple(parameter.shape))
		variance = _tensor(state['exp_avg_sq'], tuple(parameter.shape))
		if torch.any(variance < 0):
			raise ValueError('coordinated Volve HMM AdamW variance must be nonnegative')


def validate_state(payload: Mapping[str, object], config: Mapping[str, object]) -> None:
	"""Reject malformed, foreign, non-FP32 or non-finite model and AdamW state."""
	_optimizer(payload, config, _model_parameters(payload, config))
