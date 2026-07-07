"""MAE-specific checkpoint and resume helpers."""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

import seis_ssl_cluster
from seis_ssl_cluster.training.checkpoint import (
	capture_rng_state,
	restore_rng_state,
	save_checkpoint,
)

_RESUME_REQUIRED_KEYS = (
	'model_state_dict',
	'optimizer_state_dict',
	'epoch',
	'global_step',
	'amp_enabled',
	'scaler_state_dict',
	'config',
	'package_version',
	'metrics',
	'rng_state',
	'training_state',
)
_RESUME_MAPPING_KEYS = (
	'model_state_dict',
	'optimizer_state_dict',
	'config',
	'metrics',
	'training_state',
)
_RESUME_COMPATIBILITY_SECTIONS = (
	'manifests',
	'data',
	'zero_mask',
	'model',
	'masking',
	'loss',
)
_RESUME_ALLOWED_TRAIN_OVERRIDES = frozenset(
	{
		'epochs',
		'max_steps',
		'checkpoint_every_steps',
		'allow_overwrite_output',
		'diagnostics_dir',
		'device',
	},
)


@dataclass(frozen=True)
class ResumeState:
	"""Resolved checkpoint resume location."""

	start_epoch: int
	global_step: int
	skip_batches: int


def _save_mae_checkpoint(  # noqa: PLR0913
	path: Path,
	*,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	amp_enabled: bool,
	scaler: torch.amp.GradScaler | None,
	checkpoint_kind: Literal['step', 'epoch'],
	batch_index: int | None,
	rng_state: Mapping[str, object] | None = None,
) -> Path:
	checkpoint_path = save_checkpoint(
		path,
		model=model,
		optimizer=optimizer,
		epoch=epoch,
		config=config,
		package_version=getattr(seis_ssl_cluster, '__version__', None),
		metrics=metrics,
		global_step=global_step,
		amp_enabled=amp_enabled,
		scaler=scaler,
		training_state={
			'schema_version': 1,
			'stage': 'train_amp_mae',
			'checkpoint_kind': checkpoint_kind,
			'batch_index': batch_index,
		},
		rng_state=rng_state,
	)
	_latest_path = checkpoint_path.parent / 'mae_latest.pt'
	_tmp_latest = _latest_path.with_suffix('.pt.tmp')
	shutil.copy2(checkpoint_path, _tmp_latest)
	_tmp_latest.replace(_latest_path)
	return checkpoint_path


def _restore_mae_checkpoint(  # noqa: PLR0913
	*,
	payload: Mapping[str, object],
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	amp_enabled: bool,
	config: Mapping[str, object] | None = None,
) -> ResumeState:
	_validate_resume_payload(payload, amp_enabled=amp_enabled)
	if config is not None:
		_validate_resume_config_compatibility(payload, config)
	try:
		model.load_state_dict(payload['model_state_dict'])
	except RuntimeError as exc:
		msg = f'incompatible model geometry for resume checkpoint: {exc}'
		raise ValueError(msg) from exc
	optimizer.load_state_dict(payload['optimizer_state_dict'])
	if amp_enabled:
		if scaler is None:
			msg = 'scaler is required when amp_enabled is true'
			raise ValueError(msg)
		scaler.load_state_dict(payload['scaler_state_dict'])
	restore_rng_state(payload)

	training_state = payload['training_state']
	if not isinstance(training_state, Mapping):
		msg = 'resume checkpoint training_state must be a mapping'
		raise TypeError(msg)
	checkpoint_kind = training_state['checkpoint_kind']
	batch_index = training_state['batch_index']
	if checkpoint_kind == 'step':
		return ResumeState(
			start_epoch=int(payload['epoch']),
			global_step=int(payload.get('global_step', 0)),
			skip_batches=int(batch_index) + 1,
		)
	return ResumeState(
		start_epoch=int(payload['epoch']) + 1,
		global_step=int(payload.get('global_step', 0)),
		skip_batches=0,
	)


def _rng_state_for_step_checkpoint(
	*,
	dataloader: torch.utils.data.DataLoader,
	epoch_start_dataloader_rng_state: torch.Tensor,
	batch_index: int,
) -> dict[str, object]:
	if batch_index >= len(dataloader) - 1:
		return _rng_state_with_dataloader(dataloader)
	return _rng_state_with_dataloader(
		dataloader,
		dataloader_generator_state=epoch_start_dataloader_rng_state,
	)


def _rng_state_with_dataloader(
	dataloader: torch.utils.data.DataLoader,
	*,
	dataloader_generator_state: torch.Tensor | None = None,
) -> dict[str, object]:
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = (
		_dataloader_generator_state(dataloader)
		if dataloader_generator_state is None
		else dataloader_generator_state.clone()
	)
	return rng_state


def _dataloader_generator_state(
	dataloader: torch.utils.data.DataLoader,
) -> torch.Tensor:
	generator = getattr(dataloader, 'generator', None)
	if not isinstance(generator, torch.Generator):
		msg = 'MAE dataloader must expose a torch.Generator for deterministic resume'
		raise TypeError(msg)
	return generator.get_state().clone()


def _restore_dataloader_generator_state(
	*,
	payload: Mapping[str, object],
	dataloader: torch.utils.data.DataLoader,
) -> None:
	rng_state = payload['rng_state']
	if not isinstance(rng_state, Mapping):
		msg = 'resume checkpoint rng_state must be a mapping'
		raise TypeError(msg)
	generator_state = rng_state['dataloader_generator']
	if not isinstance(generator_state, torch.Tensor):
		msg = 'resume checkpoint rng_state.dataloader_generator must be a tensor'
		raise TypeError(msg)
	generator = getattr(dataloader, 'generator', None)
	if not isinstance(generator, torch.Generator):
		msg = 'MAE dataloader must expose a torch.Generator for deterministic resume'
		raise TypeError(msg)
	generator.set_state(generator_state.cpu())


def _validate_resume_payload(
	payload: Mapping[str, object],
	*,
	amp_enabled: bool,
) -> None:
	_require_resume_keys(payload)
	_validate_resume_mapping_fields(payload)
	_validate_resume_counters(payload)
	_validate_resume_rng_state(payload)
	_validate_resume_training_state(payload)
	_validate_resume_amp_state(payload, amp_enabled=amp_enabled)
	stage = _checkpoint_stage(payload)
	if stage is not None and stage != 'train_amp_mae':
		msg = f'resume checkpoint stage must be train_amp_mae; got {stage!r}'
		raise ValueError(msg)


def _require_resume_keys(payload: Mapping[str, object]) -> None:
	for key in _RESUME_REQUIRED_KEYS:
		if key not in payload:
			msg = f'resume checkpoint is missing {key}'
			raise ValueError(msg)


def _validate_resume_mapping_fields(payload: Mapping[str, object]) -> None:
	for key in _RESUME_MAPPING_KEYS:
		if not isinstance(payload[key], Mapping):
			msg = f'resume checkpoint {key} must be a mapping'
			raise TypeError(msg)


def _validate_resume_counters(payload: Mapping[str, object]) -> None:
	if not isinstance(payload['epoch'], int) or isinstance(payload['epoch'], bool):
		msg = 'resume checkpoint epoch must be an integer'
		raise TypeError(msg)
	if payload['epoch'] < 0:
		msg = 'resume checkpoint epoch must be nonnegative'
		raise ValueError(msg)
	if (
		not isinstance(payload['global_step'], int)
		or isinstance(payload['global_step'], bool)
	):
		msg = 'resume checkpoint global_step must be an integer'
		raise TypeError(msg)
	if payload['global_step'] < 0:
		msg = 'resume checkpoint global_step must be nonnegative'
		raise ValueError(msg)


def _validate_resume_amp_state(
	payload: Mapping[str, object],
	*,
	amp_enabled: bool,
) -> None:
	if not isinstance(payload['amp_enabled'], bool):
		msg = 'resume checkpoint amp_enabled must be a bool'
		raise TypeError(msg)
	if amp_enabled and not isinstance(payload['scaler_state_dict'], Mapping):
		msg = 'resume checkpoint is missing scaler_state_dict for AMP resume'
		raise ValueError(msg)


def _validate_resume_rng_state(payload: Mapping[str, object]) -> None:
	rng_state = payload['rng_state']
	if not isinstance(rng_state, Mapping):
		msg = 'resume checkpoint rng_state must be a mapping'
		raise TypeError(msg)
	for key in ('python', 'numpy', 'torch', 'dataloader_generator'):
		if key not in rng_state:
			msg = f'resume checkpoint rng_state is missing {key}'
			raise ValueError(msg)
	if not isinstance(rng_state['python'], tuple):
		msg = 'resume checkpoint rng_state.python must be a tuple'
		raise TypeError(msg)
	if not _is_numpy_rng_state(rng_state['numpy']):
		msg = 'resume checkpoint rng_state.numpy must be a NumPy RNG state tuple'
		raise TypeError(msg)
	if not isinstance(rng_state['torch'], torch.Tensor):
		msg = 'resume checkpoint rng_state.torch must be a tensor'
		raise TypeError(msg)
	if not isinstance(rng_state['dataloader_generator'], torch.Tensor):
		msg = 'resume checkpoint rng_state.dataloader_generator must be a tensor'
		raise TypeError(msg)
	cuda_state = rng_state.get('torch_cuda')
	if cuda_state is not None and not _is_cuda_rng_state(cuda_state):
		msg = 'resume checkpoint rng_state.torch_cuda must be a list of tensors'
		raise TypeError(msg)


def _validate_resume_training_state(payload: Mapping[str, object]) -> None:
	training_state = payload['training_state']
	if not isinstance(training_state, Mapping):
		msg = 'resume checkpoint training_state must be a mapping'
		raise TypeError(msg)
	for key in ('schema_version', 'stage', 'checkpoint_kind', 'batch_index'):
		if key not in training_state:
			msg = f'resume checkpoint training_state is missing {key}'
			raise ValueError(msg)
	if training_state['schema_version'] != 1:
		msg = (
			'resume checkpoint training_state.schema_version must be 1; '
			f"got {training_state['schema_version']!r}"
		)
		raise ValueError(msg)
	if training_state['stage'] != 'train_amp_mae':
		msg = (
			'resume checkpoint training_state.stage must be train_amp_mae; '
			f"got {training_state['stage']!r}"
		)
		raise ValueError(msg)

	checkpoint_kind = _validate_resume_checkpoint_kind(
		training_state['checkpoint_kind'],
	)
	_validate_resume_training_batch_index(
		checkpoint_kind=checkpoint_kind,
		batch_index=training_state['batch_index'],
	)


def _validate_resume_checkpoint_kind(value: object) -> Literal['epoch', 'step']:
	if value not in ('epoch', 'step'):
		msg = (
			'resume checkpoint training_state.checkpoint_kind must be '
			f"'epoch' or 'step'; got {value!r}"
		)
		raise ValueError(msg)
	return cast('Literal["epoch", "step"]', value)


def _validate_resume_training_batch_index(
	*,
	checkpoint_kind: Literal['epoch', 'step'],
	batch_index: object,
) -> None:
	if checkpoint_kind == 'epoch':
		if batch_index is not None:
			msg = (
				'resume checkpoint training_state.batch_index must be null '
				'for epoch checkpoints'
			)
			raise ValueError(msg)
		return
	if not isinstance(batch_index, int) or isinstance(batch_index, bool):
		msg = (
			'resume checkpoint training_state.batch_index must be an integer '
			'for step checkpoints'
		)
		raise TypeError(msg)
	if batch_index < 0:
		msg = 'resume checkpoint training_state.batch_index must be nonnegative'
		raise ValueError(msg)


def _is_numpy_rng_state(value: object) -> bool:
	return (
		isinstance(value, tuple)
		and len(value) == 5
		and isinstance(value[0], str)
		and isinstance(value[1], np.ndarray)
		and isinstance(value[2], int)
		and isinstance(value[3], int)
		and isinstance(value[4], float)
	)


def _is_cuda_rng_state(value: object) -> bool:
	return isinstance(value, list) and all(
		isinstance(child, torch.Tensor) for child in value
	)


def _checkpoint_stage(payload: Mapping[str, object]) -> object | None:
	training_state = payload.get('training_state')
	if isinstance(training_state, Mapping) and 'stage' in training_state:
		return training_state.get('stage')
	config = payload.get('config')
	if isinstance(config, Mapping):
		return config.get('stage')
	return None


def _validate_resume_config_compatibility(
	payload: Mapping[str, object],
	config: Mapping[str, object],
) -> None:
	checkpoint_config = payload['config']
	if not isinstance(checkpoint_config, Mapping):
		msg = 'resume checkpoint config must be a mapping'
		raise TypeError(msg)
	checkpoint_view = _resume_compatibility_view(checkpoint_config)
	current_view = _resume_compatibility_view(config)
	if checkpoint_view == current_view:
		return
	label = _first_compatibility_mismatch(checkpoint_view, current_view)
	msg = (
		'resume checkpoint config is incompatible with current resolved '
		f'config at {label}'
	)
	raise ValueError(msg)


def _resume_compatibility_view(config: Mapping[str, object]) -> dict[str, object]:
	view: dict[str, object] = {'stage': config.get('stage')}
	for section in _RESUME_COMPATIBILITY_SECTIONS:
		value = config.get(section)
		if section == 'loss' and isinstance(value, Mapping):
			view[section] = _json_safe(_loss_resume_compatibility_view(value))
		elif section == 'data' and isinstance(value, Mapping):
			view[section] = _json_safe(_data_resume_compatibility_view(value))
		else:
			view[section] = _json_safe(value)
	train = config.get('train')
	if isinstance(train, Mapping):
		view['train'] = {
			str(key): _json_safe(value)
			for key, value in sorted(train.items(), key=lambda item: str(item[0]))
			if str(key) not in _RESUME_ALLOWED_TRAIN_OVERRIDES
		}
	else:
		view['train'] = _json_safe(train)
	return view


def _loss_resume_compatibility_view(loss: Mapping[str, object]) -> dict[str, object]:
	view = dict(loss)
	if 'target_normalization' not in view:
		view['target_normalization'] = {'mode': 'none'}
	view.setdefault('visible_reconstruction_weight', 0.0)
	return view


def _data_resume_compatibility_view(data: Mapping[str, object]) -> dict[str, object]:
	view = dict(data)
	if 'amplitude_agc' not in view:
		view['amplitude_agc'] = {'enabled': False}
	return view


def _first_compatibility_mismatch(
	left: Mapping[str, object],
	right: Mapping[str, object],
) -> str:
	return _first_nested_mismatch(left, right, prefix='') or 'config'


def _first_nested_mismatch(
	left: Mapping[str, object],
	right: Mapping[str, object],
	*,
	prefix: str,
) -> str | None:
	for key in sorted(left.keys() | right.keys(), key=str):
		if left.get(key) == right.get(key):
			continue
		left_value = left.get(key)
		right_value = right.get(key)
		label = f'{prefix}.{key}' if prefix else str(key)
		if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
			nested = _first_nested_mismatch(
				left_value,
				right_value,
				prefix=label,
			)
			if nested is not None:
				return nested
		return label
	return None


def _json_safe(value: object) -> object:  # noqa: PLR0911
	if isinstance(value, torch.Tensor):
		if value.numel() > 4096:
			return _summarize_tensor(value)
		return _json_safe(value.detach().cpu().tolist())
	if isinstance(value, Mapping):
		return {str(key): _json_safe(child) for key, child in value.items()}
	if isinstance(value, tuple | list):
		return [_json_safe(child) for child in value]
	if isinstance(value, bool | str) or value is None:
		return value
	if isinstance(value, int):
		return int(value)
	if isinstance(value, float):
		return _json_safe_number(value)
	if isinstance(value, Path):
		return str(value)
	return repr(value)


def _json_safe_number(value: object) -> object:
	if isinstance(value, bool):
		return value
	if isinstance(value, int):
		return int(value)
	if isinstance(value, float):
		if math.isfinite(value):
			return float(value)
		return {'value': None, 'finite': False, 'repr': repr(value)}
	return value


def _summarize_tensor(value: object) -> dict[str, object]:
	if value is None:
		return {'present': False}
	if not isinstance(value, torch.Tensor):
		return {'present': True, 'type': type(value).__name__, 'repr': repr(value)}
	summary: dict[str, object] = {
		'present': True,
		'dtype': str(value.dtype),
		'shape': [int(dim) for dim in value.shape],
		'numel': int(value.numel()),
	}
	if value.numel() == 0:
		return summary
	detached = value.detach()
	if detached.dtype == torch.bool:
		true_count = int(detached.sum().cpu().item())
		summary['true_count'] = true_count
		summary['false_count'] = int(detached.numel() - true_count)
		return summary
	if torch.is_floating_point(detached):
		return _summarize_float_tensor(detached, summary)
	cpu = detached.cpu()
	summary['min'] = _json_safe_number(cpu.min().item())
	summary['max'] = _json_safe_number(cpu.max().item())
	return summary


def _summarize_float_tensor(
	value: torch.Tensor,
	summary: dict[str, object],
) -> dict[str, object]:
	finite = torch.isfinite(value)
	finite_count = int(finite.sum().cpu().item())
	summary['finite_count'] = finite_count
	summary['nonfinite_count'] = int(value.numel() - finite_count)
	if finite_count == 0:
		return summary
	finite_values = value[finite].cpu()
	summary['min'] = _json_safe_number(finite_values.min().item())
	summary['max'] = _json_safe_number(finite_values.max().item())
	summary['mean'] = _json_safe_number(finite_values.mean().item())
	return summary


__all__ = [
	'ResumeState',
	'_checkpoint_stage',
	'_data_resume_compatibility_view',
	'_dataloader_generator_state',
	'_first_compatibility_mismatch',
	'_first_nested_mismatch',
	'_is_cuda_rng_state',
	'_is_numpy_rng_state',
	'_loss_resume_compatibility_view',
	'_require_resume_keys',
	'_restore_dataloader_generator_state',
	'_restore_mae_checkpoint',
	'_resume_compatibility_view',
	'_rng_state_for_step_checkpoint',
	'_rng_state_with_dataloader',
	'_save_mae_checkpoint',
	'_validate_resume_amp_state',
	'_validate_resume_checkpoint_kind',
	'_validate_resume_config_compatibility',
	'_validate_resume_counters',
	'_validate_resume_mapping_fields',
	'_validate_resume_payload',
	'_validate_resume_rng_state',
	'_validate_resume_training_batch_index',
	'_validate_resume_training_state',
]
