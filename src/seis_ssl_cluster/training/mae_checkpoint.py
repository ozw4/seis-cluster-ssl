# ruff: noqa: CPY001
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
	load_checkpoint,
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
		'amp_dtype',
		'prefetch_factor',
		'persistent_workers',
		'runtime_check_mode',
		'stage_timing',
	},
)


@dataclass(frozen=True)
class ResumeState:
	"""Resolved checkpoint resume location."""

	start_epoch: int
	global_step: int
	skip_batches: int


@dataclass(frozen=True)
class RollingCheckpointResult:
	"""Result of a rolling MAE checkpoint write."""

	latest_path: Path
	best_path: Path
	best_score: float | None
	best_updated: bool
	best_metric_key: str | None


@dataclass(frozen=True)
class MaeCheckpointInspection:
	"""Validated, immutable evidence from one MAE checkpoint load."""

	schema_version: int
	stage: str
	checkpoint_kind: str
	batch_index: int | None
	epoch: int
	global_step: int
	resolved_precision: str
	amp_enabled: bool
	scaler_present: bool
	metrics: tuple[tuple[str, float], ...]
	best_metric_key: str | None
	best_metric_value: float | None

	def metrics_dict(self) -> dict[str, float]:
		"""Return a mutable serialization copy of the validated metrics."""
		return dict(self.metrics)


_BEST_METRIC_KEYS = (
	'val_loss',
	'validation_loss',
	'loss_val',
	'loss',
	'loss_reconstruction',
)

_RESOLVED_PRECISIONS = frozenset({'float32', 'bfloat16', 'float16'})
_TRAINING_STATE_SCHEMA_VERSION = 2
_SUPPORTED_TRAINING_STATE_SCHEMA_VERSIONS = (1, 2)


def inspect_mae_checkpoint(  # noqa: PLR0913
	path: str | Path,
	*,
	resolved_config: Mapping[str, object],
	model: torch.nn.Module,
	resolved_precision: str,
	amp_enabled: bool,
	scaler_present: bool,
) -> MaeCheckpointInspection:
	"""Load and fully inspect one MAE checkpoint against explicit run evidence."""
	checkpoint_path = Path(path)
	payload = cast(
		'Mapping[str, object]',
		load_checkpoint(checkpoint_path, map_location='cpu'),
	)
	_validate_resume_payload(
		payload,
		amp_enabled=amp_enabled,
		scaler_required=scaler_present,
		resolved_precision=resolved_precision,
	)
	if payload['config'] != resolved_config:
		raise ValueError(
			'MAE checkpoint config does not match the resolved config: '
			f'{checkpoint_path}'
		)
	try:
		model.load_state_dict(payload['model_state_dict'], strict=True)
	except RuntimeError as exc:
		msg = f'MAE checkpoint model geometry/state mismatch: {checkpoint_path}: {exc}'
		raise ValueError(msg) from exc
	_require_finite_checkpoint_tree(
		model.state_dict(), 'MAE checkpoint loaded model state'
	)
	_require_finite_checkpoint_tree(
		payload['optimizer_state_dict'], 'MAE checkpoint optimizer_state_dict'
	)
	_require_finite_checkpoint_tree(
		payload['scaler_state_dict'], 'MAE checkpoint scaler_state_dict'
	)
	metrics = _finite_checkpoint_metrics(payload['metrics'])
	best_metric_key, best_metric_value = _best_metric_from_metrics(dict(metrics))
	training_state = cast('Mapping[str, object]', payload['training_state'])
	return MaeCheckpointInspection(
		schema_version=cast('int', training_state['schema_version']),
		stage=cast('str', training_state['stage']),
		checkpoint_kind=cast('str', training_state['checkpoint_kind']),
		batch_index=cast('int | None', training_state['batch_index']),
		epoch=cast('int', payload['epoch']),
		global_step=cast('int', payload['global_step']),
		resolved_precision=_checkpoint_resolved_precision(payload),
		amp_enabled=cast('bool', payload['amp_enabled']),
		scaler_present=isinstance(payload['scaler_state_dict'], Mapping),
		metrics=metrics,
		best_metric_key=best_metric_key,
		best_metric_value=best_metric_value,
	)


def _save_mae_rolling_checkpoint(  # noqa: PLR0913
	checkpoint_dir: Path,
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
	best_score: float | None = None,
) -> RollingCheckpointResult:
	"""Write rolling ``latest.pt`` and update ``best.pt`` on metric improvement."""
	checkpoint_path = _save_mae_checkpoint(
		checkpoint_dir / 'latest.pt',
		model=model,
		optimizer=optimizer,
		epoch=epoch,
		config=config,
		metrics=metrics,
		global_step=global_step,
		amp_enabled=amp_enabled,
		scaler=scaler,
		checkpoint_kind=checkpoint_kind,
		batch_index=batch_index,
		rng_state=rng_state,
	)
	metric_key, metric_value = _best_metric_from_metrics(metrics)
	best_updated = _is_improved_best_metric(metric_value, best_score)
	resolved_best_score = best_score
	best_path = checkpoint_dir / 'best.pt'
	if best_updated:
		_copy_checkpoint_atomic(checkpoint_path, best_path)
		resolved_best_score = metric_value
	return RollingCheckpointResult(
		latest_path=checkpoint_path,
		best_path=best_path,
		best_score=resolved_best_score,
		best_updated=best_updated,
		best_metric_key=metric_key,
	)


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
	return save_checkpoint(
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
		scaler_required=scaler is not None,
		training_state={
			'schema_version': _TRAINING_STATE_SCHEMA_VERSION,
			'stage': 'train_amp_mae',
			'checkpoint_kind': checkpoint_kind,
			'batch_index': batch_index,
			'resolved_precision': _resolved_precision_from_amp_state(
				amp_enabled=amp_enabled,
				scaler_required=scaler is not None,
			),
		},
		rng_state=rng_state,
	)


def _best_metric_from_metrics(
	metrics: Mapping[str, float],
) -> tuple[str | None, float | None]:
	for key in _BEST_METRIC_KEYS:
		value = metrics.get(key)
		if isinstance(value, int | float) and not isinstance(value, bool):
			score = float(value)
			if math.isfinite(score):
				return key, score
	return None, None


def _finite_checkpoint_metrics(
	value: object,
) -> tuple[tuple[str, float], ...]:
	if not isinstance(value, Mapping):
		raise TypeError('MAE checkpoint metrics must be a mapping')
	metrics: list[tuple[str, float]] = []
	for key, metric in value.items():
		if not isinstance(key, str):
			raise TypeError('MAE checkpoint metric keys must be strings')
		if isinstance(metric, bool) or not isinstance(metric, int | float):
			raise TypeError(f'MAE checkpoint metric {key} must be numeric')
		floating = float(metric)
		if not math.isfinite(floating):
			raise ValueError(f'MAE checkpoint metric {key} must be finite')
		metrics.append((key, floating))
	if not metrics:
		raise ValueError('MAE checkpoint metrics must not be empty')
	return tuple(metrics)


def _require_finite_checkpoint_tree(value: object, label: str) -> None:
	if isinstance(value, torch.Tensor):
		if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
			torch.isfinite(value).all(),
		):
			raise ValueError(f'{label} contains a nonfinite tensor')
		return
	if isinstance(value, Mapping):
		for key, child in value.items():
			_require_finite_checkpoint_tree(child, f'{label}.{key}')
		return
	if isinstance(value, (list, tuple)):
		for index, child in enumerate(value):
			_require_finite_checkpoint_tree(child, f'{label}[{index}]')
		return
	if isinstance(
		value, float | complex | np.floating | np.complexfloating
	) and not np.isfinite(value):
		raise ValueError(f'{label} contains a nonfinite scalar')


def _is_improved_best_metric(
	score: float | None,
	best_score: float | None,
) -> bool:
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


def _restore_mae_checkpoint(  # noqa: PLR0913
	*,
	payload: Mapping[str, object],
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	amp_enabled: bool,
	scaler_required: bool | None = None,
	config: Mapping[str, object] | None = None,
) -> ResumeState:
	resolved_scaler_required = (
		amp_enabled if scaler_required is None else scaler_required
	)
	resolved_precision = _resolved_precision_from_amp_state(
		amp_enabled=amp_enabled,
		scaler_required=resolved_scaler_required,
	)
	_validate_resume_payload(
		payload,
		amp_enabled=amp_enabled,
		scaler_required=resolved_scaler_required,
		resolved_precision=resolved_precision,
	)
	if config is not None:
		_validate_resume_config_compatibility(payload, config)
	try:
		model.load_state_dict(payload['model_state_dict'])
	except RuntimeError as exc:
		msg = f'incompatible model geometry for resume checkpoint: {exc}'
		raise ValueError(msg) from exc
	optimizer.load_state_dict(payload['optimizer_state_dict'])
	if resolved_scaler_required:
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
	resolved_precision: str,
	scaler_required: bool | None = None,
) -> None:
	resolved_scaler_required = (
		amp_enabled if scaler_required is None else scaler_required
	)
	_require_resume_keys(payload)
	_validate_resume_mapping_fields(payload)
	_validate_resume_counters(payload)
	_validate_resume_rng_state(payload)
	_validate_resume_training_state(payload)
	_validate_resume_precision(payload, resolved_precision=resolved_precision)
	_validate_resume_amp_state(
		payload,
		amp_enabled=amp_enabled,
		scaler_required=resolved_scaler_required,
	)
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
	scaler_required: bool,
) -> None:
	if not isinstance(payload['amp_enabled'], bool):
		msg = 'resume checkpoint amp_enabled must be a bool'
		raise TypeError(msg)
	if payload['amp_enabled'] is not amp_enabled:
		msg = (
			'resume checkpoint amp_enabled does not match the current runtime: '
			f'checkpoint={payload["amp_enabled"]!r}, current={amp_enabled!r}'
		)
		raise ValueError(msg)
	if scaler_required and not isinstance(payload['scaler_state_dict'], Mapping):
		msg = 'resume checkpoint is missing scaler_state_dict for AMP resume'
		raise ValueError(msg)


def _validate_resume_precision(
	payload: Mapping[str, object],
	*,
	resolved_precision: str,
) -> None:
	training_state = payload['training_state']
	if not isinstance(training_state, Mapping):
		msg = 'resume checkpoint training_state must be a mapping'
		raise TypeError(msg)
	checkpoint_precision = _checkpoint_resolved_precision(payload)
	if checkpoint_precision != resolved_precision:
		msg = (
			'resume checkpoint resolved precision is incompatible with the '
			'current runtime: '
			f'checkpoint={checkpoint_precision!r}, current={resolved_precision!r}'
		)
		raise ValueError(msg)
	expected_amp_enabled = checkpoint_precision != 'float32'
	if payload['amp_enabled'] is not expected_amp_enabled:
		msg = (
			'resume checkpoint amp_enabled is inconsistent with '
			f'resolved_precision={checkpoint_precision!r}'
		)
		raise ValueError(msg)
	expected_scaler = checkpoint_precision == 'float16'
	has_scaler = isinstance(payload['scaler_state_dict'], Mapping)
	if has_scaler is not expected_scaler:
		msg = (
			'resume checkpoint scaler_state_dict is inconsistent with '
			f'resolved_precision={checkpoint_precision!r}'
		)
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
	schema_version = training_state['schema_version']
	if not isinstance(schema_version, int) or isinstance(schema_version, bool):
		raise TypeError(
			'resume checkpoint training_state.schema_version must be an integer'
		)
	if schema_version not in _SUPPORTED_TRAINING_STATE_SCHEMA_VERSIONS:
		msg = (
			'resume checkpoint training_state.schema_version must be one of '
			f'{sorted(_SUPPORTED_TRAINING_STATE_SCHEMA_VERSIONS)!r}; '
			f'got {schema_version!r}'
		)
		raise ValueError(msg)
	if training_state['stage'] != 'train_amp_mae':
		msg = (
			'resume checkpoint training_state.stage must be train_amp_mae; '
			f"got {training_state['stage']!r}"
		)
		raise ValueError(msg)
	if (
		schema_version == _TRAINING_STATE_SCHEMA_VERSION
		and 'resolved_precision' not in training_state
	):
		msg = 'resume checkpoint training_state is missing resolved_precision'
		raise ValueError(msg)
	if (
		'resolved_precision' in training_state
		and training_state['resolved_precision'] not in _RESOLVED_PRECISIONS
	):
		msg = (
			'resume checkpoint training_state.resolved_precision must be one of '
			f'{sorted(_RESOLVED_PRECISIONS)!r}; '
			f"got {training_state['resolved_precision']!r}"
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


def _resolved_precision_from_amp_state(
	*,
	amp_enabled: bool,
	scaler_required: bool,
) -> str:
	if not amp_enabled:
		return 'float32'
	return 'float16' if scaler_required else 'bfloat16'


def _checkpoint_resolved_precision(payload: Mapping[str, object]) -> str:
	training_state = cast('Mapping[str, object]', payload['training_state'])
	precision = training_state.get('resolved_precision')
	if isinstance(precision, str):
		return precision
	# Schema 1 predates BF16 support: AMP always meant FP16 with a scaler.
	return 'float16' if payload['amp_enabled'] else 'float32'


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
	'MaeCheckpointInspection',
	'ResumeState',
	'RollingCheckpointResult',
	'_best_metric_from_metrics',
	'_checkpoint_stage',
	'_data_resume_compatibility_view',
	'_dataloader_generator_state',
	'_first_compatibility_mismatch',
	'_first_nested_mismatch',
	'_is_cuda_rng_state',
	'_is_improved_best_metric',
	'_is_numpy_rng_state',
	'_loss_resume_compatibility_view',
	'_require_resume_keys',
	'_restore_dataloader_generator_state',
	'_restore_mae_checkpoint',
	'_resume_compatibility_view',
	'_rng_state_for_step_checkpoint',
	'_rng_state_with_dataloader',
	'_save_mae_checkpoint',
	'_save_mae_rolling_checkpoint',
	'_validate_resume_amp_state',
	'_validate_resume_checkpoint_kind',
	'_validate_resume_config_compatibility',
	'_validate_resume_counters',
	'_validate_resume_mapping_fields',
	'_validate_resume_payload',
	'_validate_resume_rng_state',
	'_validate_resume_training_batch_index',
	'_validate_resume_training_state',
	'inspect_mae_checkpoint',
]
