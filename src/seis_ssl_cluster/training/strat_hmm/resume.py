"""Checkpoint resume helpers for stratigraphic HMM pretext training."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from seis_ssl_cluster.training.checkpoint import restore_rng_state
from seis_ssl_cluster.training.strat_hmm.runtime import _to_json_safe
from seis_ssl_cluster.training.strat_hmm.state import StratHmmResumeState
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)


def restore_strat_hmm_training_checkpoint(  # noqa: PLR0913
	*,
	payload: Mapping[str, object],
	student: torch.nn.Module,
	head: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	amp_enabled: bool,
	config: Mapping[str, object],
	spatial_context: torch.nn.Module | None = None,
) -> StratHmmResumeState:
	"""Restore model, optimizer, AMP, RNG, and resume counters from a checkpoint."""
	_validate_strat_resume_payload(payload, amp_enabled=amp_enabled)
	checkpoint_identity = payload.get('stratigraphy_checkpoint')
	if (
		isinstance(checkpoint_identity, Mapping)
		and checkpoint_identity.get('schema_version') == 7
		and spatial_context is None
	):
		raise ValueError(
			'schema-7 resume requires the spatial_context replacement-token module'
		)
	_validate_stratigraphy_checkpoint_mode(
		payload,
		config,
		optimizer=optimizer,
		student=student,
		head=head,
		spatial_context=spatial_context,
	)
	_validate_strat_resume_config_compatibility(payload, config)
	try:
		student.load_state_dict(payload['model_state_dict'])
		head.load_state_dict(payload['stratigraphy_state_dict'])
		if spatial_context is not None:
			spatial_context.load_state_dict(payload['spatial_context_state_dict'])
	except RuntimeError as exc:
		msg = (
			'incompatible model/head/spatial context geometry for resume checkpoint: '
			f'{exc}'
		)
		raise ValueError(msg) from exc
	optimizer.load_state_dict(payload['optimizer_state_dict'])
	if amp_enabled:
		if scaler is None:
			msg = 'scaler is required when amp_enabled is true'
			raise ValueError(msg)
		scaler_state = payload['scaler_state_dict']
		if scaler_state is None:
			msg = (
				'resume checkpoint scaler_state_dict must not be null when '
				'AMP is enabled'
			)
			raise ValueError(msg)
		scaler.load_state_dict(scaler_state)
	restore_rng_state(payload)

	training_state = payload['training_state']
	if not isinstance(training_state, Mapping):
		msg = 'resume checkpoint training_state must be a mapping'
		raise TypeError(msg)
	checkpoint_kind = training_state['checkpoint_kind']
	batch_index = training_state['batch_index']
	if checkpoint_kind == 'step':
		return StratHmmResumeState(
			start_epoch=int(payload['epoch']),
			global_step=int(payload['global_step']),
			skip_batches=int(batch_index) + 1,
		)
	return StratHmmResumeState(
		start_epoch=int(payload['epoch']) + 1,
		global_step=int(payload['global_step']),
		skip_batches=0,
	)


def _validate_strat_resume_payload(
	payload: Mapping[str, object],
	*,
	amp_enabled: bool,
) -> None:
	for key in (
		'model_state_dict',
		'stratigraphy_state_dict',
		'optimizer_state_dict',
		'epoch',
		'global_step',
		'amp_enabled',
		'scaler_state_dict',
		'config',
		'stratigraphy_config',
		'package_version',
		'metrics',
		'rng_state',
		'training_state',
	):
		if key not in payload:
			msg = f'resume checkpoint is missing {key}'
			raise ValueError(msg)
	for key in (
		'model_state_dict',
		'stratigraphy_state_dict',
		'optimizer_state_dict',
		'config',
		'stratigraphy_config',
		'metrics',
		'training_state',
	):
		if not isinstance(payload[key], Mapping):
			msg = f'resume checkpoint {key} must be a mapping'
			raise TypeError(msg)
	if not isinstance(payload['epoch'], int) or isinstance(payload['epoch'], bool):
		msg = 'resume checkpoint epoch must be an integer'
		raise TypeError(msg)
	if not isinstance(payload['global_step'], int) or isinstance(
		payload['global_step'],
		bool,
	):
		msg = 'resume checkpoint global_step must be an integer'
		raise TypeError(msg)
	if payload['epoch'] < 0 or payload['global_step'] < 0:
		msg = 'resume checkpoint counters must be nonnegative'
		raise ValueError(msg)
	if not isinstance(payload['amp_enabled'], bool):
		msg = 'resume checkpoint amp_enabled must be a boolean'
		raise TypeError(msg)
	if bool(payload['amp_enabled']) != bool(amp_enabled):
		msg = (
			'resume checkpoint amp_enabled does not match current runtime: '
			f'checkpoint={payload["amp_enabled"]!r}, current={amp_enabled!r}'
		)
		raise ValueError(msg)
	_validate_strat_resume_training_state(payload)
	_validate_resume_rng_state(payload)


def _validate_strat_resume_training_state(payload: Mapping[str, object]) -> None:
	training_state = payload['training_state']
	if not isinstance(training_state, Mapping):
		msg = 'resume checkpoint training_state must be a mapping'
		raise TypeError(msg)
	for key in ('schema_version', 'stage', 'checkpoint_kind', 'batch_index'):
		if key not in training_state:
			msg = f'resume checkpoint training_state is missing {key}'
			raise ValueError(msg)
	if training_state['stage'] != 'train_strat_hmm_pretext':
		msg = (
			'resume checkpoint stage must be train_strat_hmm_pretext; '
			f'got {training_state["stage"]!r}'
		)
		raise ValueError(msg)
	if training_state['checkpoint_kind'] not in {'step', 'epoch'}:
		msg = 'resume checkpoint checkpoint_kind must be "step" or "epoch"'
		raise ValueError(msg)
	batch_index = training_state['batch_index']
	if training_state['checkpoint_kind'] == 'step':
		if not isinstance(batch_index, int) or isinstance(batch_index, bool):
			msg = (
				'resume checkpoint batch_index must be an integer for step checkpoints'
			)
			raise TypeError(msg)
		if batch_index < 0:
			msg = 'resume checkpoint batch_index must be nonnegative'
			raise ValueError(msg)
	elif batch_index is not None:
		msg = 'resume checkpoint batch_index must be null for epoch checkpoints'
		raise ValueError(msg)


def _validate_resume_rng_state(payload: Mapping[str, object]) -> None:
	rng_state = payload['rng_state']
	if not isinstance(rng_state, Mapping):
		msg = 'resume checkpoint rng_state must be a mapping'
		raise TypeError(msg)
	if 'dataloader_generator' not in rng_state:
		msg = 'resume checkpoint rng_state is missing dataloader_generator'
		raise ValueError(msg)
	if not isinstance(rng_state['dataloader_generator'], torch.Tensor):
		msg = 'resume checkpoint rng_state.dataloader_generator must be a tensor'
		raise TypeError(msg)


def _validate_strat_resume_config_compatibility(
	payload: Mapping[str, object],
	config: Mapping[str, object],
) -> None:
	checkpoint_config = payload['stratigraphy_config']
	if not isinstance(checkpoint_config, Mapping):
		msg = 'resume checkpoint stratigraphy_config must be a mapping'
		raise TypeError(msg)
	checkpoint_view = _strat_resume_compatibility_view(checkpoint_config)
	current_view = _strat_resume_compatibility_view(config)
	if checkpoint_view == current_view:
		return
	label = _first_compatibility_mismatch(checkpoint_view, current_view)
	msg = (
		'resume checkpoint stratigraphy_config is incompatible with current '
		f'resolved config at {label}'
	)
	raise ValueError(msg)


def _validate_stratigraphy_checkpoint_mode(  # noqa: PLR0913
	payload: Mapping[str, object],
	config: Mapping[str, object],
	*,
	optimizer: torch.optim.Optimizer,
	student: torch.nn.Module | None = None,
	head: torch.nn.Module | None = None,
	spatial_context: torch.nn.Module | None = None,
) -> None:
	head_config = config.get('head')
	is_multi_head = isinstance(head_config, Mapping) and 'spec' in head_config
	if is_multi_head:
		if 'stratigraphy_checkpoint' not in payload:
			raise ValueError(
				'multi-head resume requires a versioned multi-head checkpoint'
			)
		validate_stratigraphy_checkpoint_payload(
			payload,
			expected_config=config,
			expected_optimizer=optimizer,
			expected_student=student,
			expected_head=head,
			expected_spatial_context=spatial_context,
		)
	elif 'stratigraphy_checkpoint' in payload:
		raise ValueError('multi-head checkpoint cannot resume as a single-head run')
	elif spatial_context is not None:
		raise ValueError('spatial_context requires a versioned multi-head checkpoint')


def _strat_resume_compatibility_view(
	config: Mapping[str, object],
) -> dict[str, object]:
	view: dict[str, object] = {'stage': config.get('stage')}
	for section in (
		'manifests',
		'data',
		'model',
		'pseudo_targets',
		'teacher',
		'student',
		'head',
		'loss',
		'zero_mask',
	):
		value = config.get(section)
		if section == 'data' and isinstance(value, Mapping):
			view[section] = _to_json_safe(_data_resume_compatibility_view(value))
		else:
			view[section] = _to_json_safe(value)
	if 'spatial_context' in config:
		view['spatial_context'] = _to_json_safe(config['spatial_context'])
	train = config.get('train')
	if isinstance(train, Mapping):
		view['train'] = {
			str(key): _to_json_safe(value)
			for key, value in sorted(train.items(), key=lambda item: str(item[0]))
			if str(key)
			not in {
				'epochs',
				'max_steps',
				'checkpoint_every_steps',
				'allow_overwrite_output',
				'device',
				'num_workers',
			}
		}
	else:
		view['train'] = _to_json_safe(train)
	return view


def _data_resume_compatibility_view(data: Mapping[str, object]) -> dict[str, object]:
	view = dict(data)
	if 'amplitude_agc' not in view:
		view['amplitude_agc'] = {'enabled': False}
	return view


def _first_compatibility_mismatch(
	left: Mapping[str, object],
	right: Mapping[str, object],
	*,
	prefix: str = '',
) -> str:
	for key in sorted(set(left) | set(right)):
		label = f'{prefix}.{key}' if prefix else str(key)
		if key not in left or key not in right:
			return label
		left_value = left[key]
		right_value = right[key]
		if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
			child = _first_compatibility_mismatch(
				left_value,
				right_value,
				prefix=label,
			)
			if child:
				return child
		elif left_value != right_value:
			return label
	return ''


__all__ = ['restore_strat_hmm_training_checkpoint']
