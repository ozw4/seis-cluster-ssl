"""Config-driven 3D Barlow Twins pretraining."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from seis_ssl_cluster.data.amplitude_dataset import AmplitudePretrainDataset
from seis_ssl_cluster.data.barlow_twins_dataset import BarlowTwinsPretrainDataset
from seis_ssl_cluster.data.schema import read_manifest_json
from seis_ssl_cluster.data.zero_mask import ZeroMaskConfig
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D, BarlowTwinsLoss
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	BarlowTwinsResumeState,
	load_barlow_twins_checkpoint,
	restore_barlow_twins_checkpoint,
	save_barlow_twins_checkpoint,
	update_best_checkpoint,
)
from seis_ssl_cluster.training.barlow_twins_continuation import (
	configure_barlow_twins_continuation_trainability,
	load_barlow_twins_continuation_weights,
)
from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.dataloaders import build_barlow_twins_dataloader
from seis_ssl_cluster.training.logging import print_epoch_metrics

_LOSS_METRICS = (
	'training_loss',
	'on_diag',
	'off_diag',
	'projection_std_mean',
	'projection_std_min',
	'projection_norm_mean',
	'cross_correlation_diag_mean',
	'cross_correlation_offdiag_rms',
	'weighted_off_diag',
)


@dataclass(frozen=True)
class BarlowTwinsTrainingState:
	"""Training progress and averaged metrics for one epoch invocation."""

	epoch: int
	global_step: int
	metrics: dict[str, float]
	completed_epoch: bool


@dataclass(frozen=True)
class _Precision:
	amp_enabled: bool
	autocast_dtype: torch.dtype | None
	scaler_required: bool


def train_barlow_twins_one_epoch(  # noqa: PLR0913
	*,
	model: BarlowTwins3D,
	loss_fn: BarlowTwinsLoss,
	dataloader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	global_step: int = 0,
	max_steps: int | None = None,
	amp_enabled: bool = False,
	amp_dtype: torch.dtype | None = None,
	scaler: torch.amp.GradScaler | None = None,
	grad_clip_norm: float | None = None,
) -> BarlowTwinsTrainingState:
	"""Train for one epoch or until the supplied per-call step limit."""
	model.train()
	totals = dict.fromkeys(
		(*_LOSS_METRICS, 'gradient_norm', 'learning_rate', 'step_time_seconds'),
		0.0,
	)
	parameters = tuple(
		parameter
		for parameter in model.pretraining_parameters()
		if parameter.requires_grad
	)
	if device.type == 'cuda':
		torch.cuda.reset_peak_memory_stats(device)
	batches = 0
	for raw_batch in dataloader:
		if max_steps is not None and batches >= max_steps:
			break
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		step_started = time.perf_counter()
		batch = move_batch_to_device(
			raw_batch,
			device,
			non_blocking=device.type == 'cuda',
		)
		optimizer.zero_grad(set_to_none=True)
		with torch.autocast(
			device_type=device.type,
			dtype=amp_dtype,
			enabled=amp_enabled,
		):
			outputs = model(
				_tensor(batch, 'view_a'),
				_tensor(batch, 'view_b'),
				valid_mask_a=_tensor(batch, 'valid_mask_a'),
				valid_mask_b=_tensor(batch, 'valid_mask_b'),
			)
			losses = loss_fn(outputs['z_a'], outputs['z_b'])
			loss = losses['loss']
		if not bool(torch.isfinite(loss).item()):
			msg = (
				'non-finite Barlow Twins loss '
				f'at epoch={epoch}, global_step={global_step}'
			)
			raise FloatingPointError(msg)
		if scaler is None:
			loss.backward()
			grad_norm = _clip_and_check_gradients(parameters, grad_clip_norm)
			optimizer.step()
		else:
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			grad_norm = _clip_and_check_gradients(parameters, grad_clip_norm)
			scaler.step(optimizer)
			scaler.update()
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		step_time_seconds = time.perf_counter() - step_started
		global_step += 1
		batches += 1
		totals['training_loss'] += float(loss.detach().cpu())
		totals['on_diag'] += float(losses['on_diag'].detach().cpu())
		totals['off_diag'] += float(losses['off_diag'].detach().cpu())
		for key in _LOSS_METRICS[3:]:
			totals[key] += float(losses[key].detach().cpu())
		totals['gradient_norm'] += float(grad_norm.detach().cpu())
		totals['learning_rate'] += float(optimizer.param_groups[0]['lr'])
		totals['step_time_seconds'] += step_time_seconds
	if batches == 0:
		raise ValueError('Barlow Twins dataloader produced no batches')
	metrics = {key: value / batches for key, value in totals.items()}
	metrics['peak_cuda_memory_mib'] = (
		float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
		if device.type == 'cuda'
		else 0.0
	)
	return BarlowTwinsTrainingState(
		epoch=epoch,
		global_step=global_step,
		metrics=metrics,
		completed_epoch=batches == len(dataloader),
	)


def run_barlow_twins_pretraining(
	config: Mapping[str, object],
	*,
	resume: str | Path | None = None,
) -> Path:
	"""Run Barlow Twins pretraining from a resolved stage config."""
	if config.get('stage') != 'barlow_twins_training':
		raise ValueError('config must be resolved for barlow_twins_training')
	train = _mapping(config, 'train')
	model_config = _mapping(config, 'model')
	data = _mapping(config, 'data')
	barlow = _mapping(config, 'barlow_twins')
	augmentations = _mapping(config, 'augmentations')
	continuation = (
		_mapping(config, 'continuation') if 'continuation' in config else None
	)
	device = _resolve_device(train)
	seed = _integer(train, 'seed')
	_seed_everything(seed, device=device)
	precision = _resolve_precision(train, device=device)
	scaler = (
		torch.amp.GradScaler('cuda', enabled=True)
		if precision.scaler_required
		else None
	)

	output_root = Path(_string(_mapping(config, 'paths'), 'output_root'))
	manifests = read_manifest_json(
		Path(_string(_mapping(config, 'manifests'), 'train'))
	)

	base_dataset = AmplitudePretrainDataset(
		manifests,
		local_crop_size_xyz=_xyz(data, 'local_crop_size'),
		patch_size_xyz=_xyz(model_config, 'patch_size'),
		emit_spatial_mask=False,
		seed=seed,
		samples_per_epoch=_integer(train, 'samples_per_epoch'),
		zero_mask=_zero_mask(config),
		min_valid_fraction=_floating(data, 'min_valid_fraction'),
		max_resample_attempts=_integer(data, 'max_resample_attempts'),
		normalized_clip_abs=_optional_float(data, 'normalized_clip_abs'),
		amplitude_agc=cast('Mapping[str, object]', data['amplitude_agc']),
		finite_check_mode=cast('Any', data['finite_check_mode']),
	)
	dataset = BarlowTwinsPretrainDataset(
		base_dataset,
		horizontal_flip_probability=_floating(
			augmentations,
			'horizontal_flip_probability',
		),
	)
	dataloader = build_barlow_twins_dataloader(
		dataset,
		batch_size=_integer(train, 'batch_size'),
		num_workers=_integer(train, 'num_workers'),
		shuffle=_boolean(train, 'shuffle'),
		seed=seed,
		device=device,
		prefetch_factor=_optional_int(train, 'prefetch_factor'),
		persistent_workers=_boolean(train, 'persistent_workers'),
	)
	backbone = _build_backbone(model_config).to(device)
	model = BarlowTwins3D(
		backbone,
		projector_dim=_integer(barlow, 'projector_dim'),
	).to(device)
	loss_fn = BarlowTwinsLoss(
		redundancy_weight=_floating(barlow, 'redundancy_weight'),
		normalization_eps=_floating(barlow, 'normalization_eps'),
	)
	optimizer_parameters = _initialize_barlow_twins_model(
		model,
		continuation=continuation,
		resume=resume,
		model_config=model_config,
		barlow_twins_config=barlow,
	)
	optimizer = torch.optim.AdamW(
		optimizer_parameters,
		lr=_floating(train, 'lr'),
		weight_decay=_floating(train, 'weight_decay'),
	)
	resume_state = BarlowTwinsResumeState(start_epoch=1, global_step=0)
	if resume is not None:
		resume_state = restore_barlow_twins_checkpoint(
			load_barlow_twins_checkpoint(resume, map_location=device),
			backbone=backbone,
			projector=model.projector,
			optimizer=optimizer,
			scaler=scaler,
			scaler_required=precision.scaler_required,
			config=config,
			dataloader_generator=dataloader.generator,
		)

	_prepare_run_directory(
		output_root,
		resume=resume,
		allow_overwrite=_boolean(train, 'allow_overwrite_output'),
	)
	_write_json(output_root / 'resolved_config.json', config)
	history = _read_history(output_root / 'history.json') if resume else []
	best_loss = _existing_best_loss(output_root / 'best.pt')
	global_step = resume_state.global_step
	checkpoint_path: Path | None = None
	max_steps = _optional_int(train, 'max_steps')
	for epoch in range(resume_state.start_epoch, _integer(train, 'epochs') + 1):
		remaining_steps = None if max_steps is None else max_steps - global_step
		if remaining_steps is not None and remaining_steps <= 0:
			break
		dataset.set_epoch(epoch - 1)
		state = train_barlow_twins_one_epoch(
			model=model,
			loss_fn=loss_fn,
			dataloader=dataloader,
			optimizer=optimizer,
			device=device,
			epoch=epoch,
			global_step=global_step,
			max_steps=remaining_steps,
			amp_enabled=precision.amp_enabled,
			amp_dtype=precision.autocast_dtype,
			scaler=scaler,
			grad_clip_norm=_floating(train, 'grad_clip_norm'),
		)
		global_step = state.global_step
		print_epoch_metrics(epoch, state.metrics)
		history.append(
			{'epoch': epoch, 'global_step': global_step, **state.metrics}
		)
		_write_json(output_root / 'history.json', history)
		checkpoint_path = save_barlow_twins_checkpoint(
			output_root / 'latest.pt',
			backbone=backbone,
			projector=model.projector,
			optimizer=optimizer,
			epoch=epoch,
			global_step=global_step,
			config=config,
			metrics=state.metrics,
			amp_enabled=precision.amp_enabled,
			scaler=scaler,
			scaler_required=precision.scaler_required,
			dataset_epoch=dataset.epoch,
			completed_epoch=state.completed_epoch,
			dataloader_generator=dataloader.generator,
		)
		best_loss = update_best_checkpoint(
			checkpoint_path,
			output_root / 'best.pt',
			loss=state.metrics['training_loss'],
			best_loss=best_loss,
		)
		if max_steps is not None and global_step >= max_steps:
			break
	if checkpoint_path is None:
		raise ValueError('no Barlow Twins training epochs were run')
	return checkpoint_path


def _initialize_barlow_twins_model(
	model: BarlowTwins3D,
	*,
	continuation: Mapping[str, object] | None,
	resume: str | Path | None,
	model_config: Mapping[str, object],
	barlow_twins_config: Mapping[str, object],
) -> tuple[torch.nn.Parameter, ...]:
	if continuation is None:
		return tuple(model.pretraining_parameters())
	if resume is None:
		load_barlow_twins_continuation_weights(
			model,
			_string(continuation, 'init_checkpoint'),
			expected_model_config=model_config,
			expected_barlow_twins_config=barlow_twins_config,
		)
	return configure_barlow_twins_continuation_trainability(
		model,
		unfreeze_top_blocks=_integer(continuation, 'unfreeze_top_blocks'),
	)


def _clip_and_check_gradients(
	parameters: tuple[torch.nn.Parameter, ...],
	grad_clip_norm: float | None,
) -> torch.Tensor:
	return torch.nn.utils.clip_grad_norm_(
		parameters,
		float('inf') if grad_clip_norm is None else grad_clip_norm,
		error_if_nonfinite=True,
	)


def _build_backbone(config: Mapping[str, object]) -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		in_channels=_integer(config, 'in_channels'),
		out_channels=_integer(config, 'out_channels'),
		patch_size_xyz=_xyz(config, 'patch_size'),
		encoder_dim=_integer(config, 'encoder_dim'),
		encoder_depth=_integer(config, 'encoder_depth'),
		encoder_heads=_integer(config, 'encoder_heads'),
		decoder_dim=_integer(config, 'decoder_dim'),
		decoder_depth=_integer(config, 'decoder_depth'),
		decoder_heads=_integer(config, 'decoder_heads'),
	)


def _resolve_device(train: Mapping[str, object]) -> torch.device:
	name = train.get('device')
	if name == 'auto':
		name = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = torch.device(cast('str', name))
	if device.type == 'cuda' and not torch.cuda.is_available():
		raise ValueError('train.device requested CUDA, but CUDA is not available')
	return device


def _resolve_precision(
	train: Mapping[str, object],
	*,
	device: torch.device,
) -> _Precision:
	if not _boolean(train, 'amp') or device.type != 'cuda':
		return _Precision(
			amp_enabled=False,
			autocast_dtype=None,
			scaler_required=False,
		)
	requested = _string(train, 'amp_dtype')
	use_bfloat16 = requested == 'bfloat16' or (
		requested == 'auto' and torch.cuda.is_bf16_supported()
	)
	if requested == 'bfloat16' and not torch.cuda.is_bf16_supported():
		raise ValueError('train.amp_dtype=bfloat16 is not supported by the device')
	return _Precision(
		amp_enabled=True,
		autocast_dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
		scaler_required=not use_bfloat16,
	)


def _prepare_run_directory(
	output_root: Path,
	*,
	resume: str | Path | None,
	allow_overwrite: bool,
) -> None:
	output_root.mkdir(parents=True, exist_ok=True)
	if resume is None and not allow_overwrite and any(output_root.iterdir()):
		raise FileExistsError(f'output_root is nonempty: {output_root}')


def _zero_mask(config: Mapping[str, object]) -> ZeroMaskConfig:
	value = _mapping(config, 'zero_mask')
	return ZeroMaskConfig(
		enabled=_boolean(value, 'enabled'),
		zero_atol=_floating(value, 'zero_atol'),
		z_sample_influence_radius=_integer(value, 'z_sample_influence_radius'),
		xy_trace_influence_radius=_integer(value, 'xy_trace_influence_radius'),
	)


def _seed_everything(seed: int, *, device: torch.device) -> None:
	random.seed(seed)
	np.random.seed(seed)  # noqa: NPY002
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed_all(seed)


def _existing_best_loss(path: Path) -> float | None:
	if not path.is_file():
		return None
	payload = load_barlow_twins_checkpoint(path, map_location='cpu')
	metrics = payload.get('metrics')
	if not isinstance(metrics, Mapping):
		return None
	value = metrics.get('training_loss')
	return float(value) if isinstance(value, int | float) else None


def _read_history(path: Path) -> list[dict[str, object]]:
	if not path.is_file():
		return []
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, list):
		raise TypeError('history.json must contain a list')
	return cast('list[dict[str, object]]', value)


def _write_json(path: Path, value: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		f'{json.dumps(value, indent=2, sort_keys=True, allow_nan=False)}\n',
		encoding='utf-8',
	)


def _tensor(config: Mapping[str, object], key: str) -> torch.Tensor:
	value = config.get(key)
	if not isinstance(value, torch.Tensor):
		raise TypeError(f'{key} must be a tensor')
	return value


def _mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = config.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return value


def _string(config: Mapping[str, object], key: str) -> str:
	value = config.get(key)
	if not isinstance(value, str):
		raise TypeError(f'{key} must be a string')
	return value


def _integer(config: Mapping[str, object], key: str) -> int:
	value = config.get(key)
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{key} must be an integer')
	return value


def _optional_int(config: Mapping[str, object], key: str) -> int | None:
	value = config.get(key)
	return None if value is None else _integer(config, key)


def _floating(config: Mapping[str, object], key: str) -> float:
	value = config.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{key} must be numeric')
	return float(value)


def _optional_float(config: Mapping[str, object], key: str) -> float | None:
	value = config.get(key)
	return None if value is None else _floating(config, key)


def _boolean(config: Mapping[str, object], key: str) -> bool:
	value = config.get(key)
	if not isinstance(value, bool):
		raise TypeError(f'{key} must be a bool')
	return value


def _xyz(config: Mapping[str, object], key: str) -> tuple[int, int, int]:
	value = config.get(key)
	if not isinstance(value, list | tuple) or len(value) != 3:
		raise TypeError(f'{key} must contain three integers')
	return cast('tuple[int, int, int]', tuple(value))


__all__ = [
	'BarlowTwinsTrainingState',
	'run_barlow_twins_pretraining',
	'train_barlow_twins_one_epoch',
]
