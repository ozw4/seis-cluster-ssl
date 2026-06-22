"""Amplitude MAE pretraining engine."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

import seis_ssl_cluster
from seis_ssl_cluster.config.schema import (
	DEFAULT_MAE_DATA_OPTIONS,
	DEFAULT_MAE_LOSS_OPTIONS,
	DEFAULT_MAE_TRAIN_OPTIONS,
	DEFAULT_ZERO_MASK_CONTRACT,
	FIXED_DATA_CONTRACT,
	FIXED_LOSS_CONTRACT,
	FIXED_MASKING_CONTRACT,
	FIXED_MODEL_CONTRACT,
	STAGE_MAE_TRAINING,
)
from seis_ssl_cluster.data import NopimsAmplitudePretrainDataset, read_manifest_json
from seis_ssl_cluster.losses import mae_pretraining_loss
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.models.mae.patching import unpatchify_3d
from seis_ssl_cluster.training.checkpoint import (
	capture_rng_state,
	load_checkpoint,
	restore_rng_state,
	save_checkpoint,
)
from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.dataloaders import build_mae_dataloader
from seis_ssl_cluster.training.logging import print_epoch_metrics
from seis_ssl_cluster.visualization.mae_debug import (
	MaeDebugVisualizationConfig,
	save_mae_debug_visualization_pngs,
)

_MANIFEST_BUILD_HINT = (
	'Build NOPIMS manifests with '
	'`python proc/seis_ssl_cluster/build_nopims_manifests.py --config '
	'proc/configs/seis_ssl_cluster/build_nopims_manifests.yaml`.'
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
class MaeTrainingState:
	"""Summary state returned from one MAE training epoch."""

	epoch: int
	global_step: int
	metrics: dict[str, float]
	amp_enabled: bool
	last_batch_index: int
	completed_epoch: bool


@dataclass(frozen=True)
class MaeStepState:
	"""State captured immediately after one MAE optimizer step."""

	epoch: int
	batch_index: int
	global_step: int
	metrics: dict[str, float]
	amp_enabled: bool


@dataclass(frozen=True)
class ResumeState:
	"""Resolved checkpoint resume location."""

	start_epoch: int
	global_step: int
	skip_batches: int


StepCallback = Callable[[MaeStepState], None]


def train_mae_one_epoch(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	patch_size_xyz: tuple[int, int, int],
	loss_config: Mapping[str, object],
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	global_step: int = 0,
	max_steps: int | None = None,
	diagnostics_dir: Path | None = None,
	grad_clip_norm: float | None = None,
	skip_batches: int = 0,
	visualization_config: MaeDebugVisualizationConfig | None = None,
	step_callback: StepCallback | None = None,
	run_config: Mapping[str, object] | None = None,
) -> MaeTrainingState:
	"""Train ``model`` for one epoch and return averaged loss metrics."""
	model.train()
	totals: dict[str, float] = {}
	batches = 0
	last_batch_index = -1
	epoch_visualization_triggered = False

	for batch_index, raw_batch in enumerate(dataloader):
		if batch_index < skip_batches:
			continue
		if max_steps is not None and batches >= max_steps:
			break
		batch = move_batch_to_device(raw_batch, device)
		optimizer.zero_grad(set_to_none=True)

		with torch.amp.autocast('cuda', enabled=amp_enabled):
			output = model(cast('Mapping[str, torch.Tensor]', batch))
			losses = mae_pretraining_loss(
				pred_patches=_required_tensor(output, 'pred_patches'),
				target=_required_tensor(batch, 'target'),
				spatial_mask=_required_tensor(batch, 'spatial_mask'),
				local_valid_mask=_required_tensor(batch, 'local_valid_mask'),
				patch_size_xyz=patch_size_xyz,
				reconstruction=_loss_mode(loss_config.get('reconstruction', 'huber')),
				huber_delta=_float_config(loss_config, 'huber_delta', 1.0),
				gradient_weight=_float_config(loss_config, 'gradient_weight', 0.05),
			)
			loss = losses['loss']

		if not torch.isfinite(loss).all():
			_raise_nonfinite(
				kind='loss',
				global_step=global_step,
				epoch=epoch,
				batch_index=batch_index,
				batch=batch,
				output=output,
				losses=losses,
				amp_enabled=amp_enabled,
				diagnostics_dir=diagnostics_dir,
				patch_size_xyz=patch_size_xyz,
				run_config=run_config,
			)

		current_grad_norm: float | None = None
		if amp_enabled:
			if scaler is None:
				msg = 'scaler is required when amp_enabled is true'
				raise ValueError(msg)
			scaler.scale(loss).backward()
			if grad_clip_norm is not None:
				scaler.unscale_(optimizer)
				current_grad_norm = _clip_and_check_gradients(
					model=model,
					grad_clip_norm=grad_clip_norm,
					global_step=global_step,
					epoch=epoch,
					batch_index=batch_index,
					batch=batch,
					output=output,
					losses=losses,
					amp_enabled=amp_enabled,
					diagnostics_dir=diagnostics_dir,
					patch_size_xyz=patch_size_xyz,
					run_config=run_config,
				)
				totals['grad_norm'] = totals.get('grad_norm', 0.0) + current_grad_norm
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			if grad_clip_norm is not None:
				current_grad_norm = _clip_and_check_gradients(
					model=model,
					grad_clip_norm=grad_clip_norm,
					global_step=global_step,
					epoch=epoch,
					batch_index=batch_index,
					batch=batch,
					output=output,
					losses=losses,
					amp_enabled=amp_enabled,
					diagnostics_dir=diagnostics_dir,
					patch_size_xyz=patch_size_xyz,
					run_config=run_config,
				)
				totals['grad_norm'] = totals.get('grad_norm', 0.0) + current_grad_norm
			optimizer.step()

		step_metrics: dict[str, float] = {}
		for key, value in losses.items():
			metric = float(value.detach().cpu().item())
			step_metrics[key] = metric
			totals[key] = totals.get(key, 0.0) + metric
		if current_grad_norm is not None:
			step_metrics['grad_norm'] = current_grad_norm
		batches += 1
		global_step += 1
		last_batch_index = batch_index
		if visualization_config is not None:
			epoch_triggered = _mae_debug_epoch_triggered(
				config=visualization_config,
				epoch=epoch,
				already_triggered=epoch_visualization_triggered,
			)
			step_triggered = _mae_debug_step_triggered(
				config=visualization_config,
				global_step=global_step,
			)
			if epoch_triggered or step_triggered:
				_save_mae_debug_visualization(
					batch=batch,
					model_output=output,
					patch_size_xyz=patch_size_xyz,
					epoch=epoch,
					global_step=global_step,
					config=visualization_config,
					metrics=step_metrics,
				)
				epoch_visualization_triggered = (
					epoch_visualization_triggered or epoch_triggered
				)
		if step_callback is not None:
			step_callback(
				MaeStepState(
					epoch=epoch,
					batch_index=batch_index,
					global_step=global_step,
					metrics=step_metrics,
					amp_enabled=amp_enabled,
				),
			)

	if batches == 0:
		msg = 'dataloader produced no batches'
		raise ValueError(msg)

	return MaeTrainingState(
		epoch=epoch,
		global_step=global_step,
		metrics={key: total / batches for key, total in totals.items()},
		amp_enabled=amp_enabled,
		last_batch_index=last_batch_index,
		completed_epoch=last_batch_index >= len(dataloader) - 1,
	)


def run_mae_pretraining(  # noqa: C901, PLR0915
	config: Mapping[str, object],
	*,
	resume: str | Path | None = None,
) -> Path:
	"""Run amplitude-only MAE pretraining from ``config``."""
	config = _complete_mae_training_config(config)
	manifests = read_manifest_json(_manifest_train_path(config))
	train_config = _mapping(config, 'train')
	model_config = _mapping(config, 'model')
	paths_config = _mapping(config, 'paths')
	loss_config = _mapping(config, 'loss')

	device = _resolve_device(train_config)
	seed = _int_config(train_config, 'seed', 42)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed_all(seed)

	output_root = _resolve_output_root(paths_config)
	allow_overwrite_output = _bool_config(
		train_config,
		'allow_overwrite_output',
		default=False,
	)
	prepare_run_directory(
		output_root=output_root,
		resume=resume,
		allow_overwrite=allow_overwrite_output,
	)
	_snapshot_run_inputs(
		output_root=output_root,
		config=config,
		overwrite=allow_overwrite_output and resume is None,
	)

	samples_per_epoch = _optional_int_config(train_config, 'samples_per_epoch')
	dataset = NopimsAmplitudePretrainDataset.from_config(
		manifests,
		config,
		samples_per_epoch=samples_per_epoch,
	)
	dataloader = build_mae_dataloader(
		dataset,
		batch_size=_int_config(train_config, 'batch_size', 4),
		num_workers=_nonnegative_int_config(train_config, 'num_workers', 0),
		shuffle=_bool_config(train_config, 'shuffle', default=True),
		seed=seed,
		device=device,
	)

	model = _build_model(model_config).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=_float_config(train_config, 'lr', 3.0e-5),
		weight_decay=_float_config(train_config, 'weight_decay', 0.05),
	)
	amp_enabled = (
		_bool_config(train_config, 'amp', default=False)
		and device.type == 'cuda'
		and torch.cuda.is_available()
	)
	scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
	resume_state = ResumeState(start_epoch=1, global_step=0, skip_batches=0)
	if resume is not None:
		payload = load_checkpoint(resume, map_location=device)
		resume_state = _restore_mae_checkpoint(
			payload=payload,
			model=model,
			optimizer=optimizer,
			scaler=scaler,
			amp_enabled=amp_enabled,
			config=config,
		)
		_restore_dataloader_generator_state(payload=payload, dataloader=dataloader)
		if resume_state.skip_batches >= len(dataloader):
			resume_state = ResumeState(
				start_epoch=resume_state.start_epoch + 1,
				global_step=resume_state.global_step,
				skip_batches=0,
			)

	epochs = _int_config(train_config, 'epochs', 100)
	max_steps = _optional_int_config(train_config, 'max_steps')
	checkpoint_every_steps = _optional_int_config(
		train_config,
		'checkpoint_every_steps',
	)
	diagnostics_dir = _resolve_diagnostics_dir(train_config, output_root)
	grad_clip_norm = _optional_positive_float_config(train_config, 'grad_clip_norm')
	visualization_config = _mae_debug_visualization_config(config, output_root)
	state: MaeTrainingState = MaeTrainingState(
		epoch=resume_state.start_epoch - 1,
		global_step=resume_state.global_step,
		metrics={},
		amp_enabled=amp_enabled,
		last_batch_index=-1,
		completed_epoch=True,
	)
	checkpoint_path: Path | None = None
	for epoch in range(resume_state.start_epoch, epochs + 1):
		set_epoch = getattr(dataset, 'set_epoch', None)
		if callable(set_epoch):
			set_epoch(epoch - 1)
		epoch_start_dataloader_rng_state = _dataloader_generator_state(dataloader)
		remaining_steps = None
		if max_steps is not None:
			remaining_steps = max_steps - state.global_step
			if remaining_steps <= 0:
				break
		skip_batches = (
			resume_state.skip_batches
			if epoch == resume_state.start_epoch
			else 0
		)

		def save_step_checkpoint(
			step_state: MaeStepState,
			epoch_start_rng_state: torch.Tensor = epoch_start_dataloader_rng_state,
		) -> None:
			nonlocal checkpoint_path
			if (
				checkpoint_every_steps is None
				or step_state.global_step % checkpoint_every_steps != 0
			):
				return
			checkpoint_path = _save_mae_checkpoint(
				output_root / f'mae_step_{step_state.global_step:08d}.pt',
				model=model,
				optimizer=optimizer,
				epoch=step_state.epoch,
				config=config,
				metrics=step_state.metrics,
				global_step=step_state.global_step,
				amp_enabled=step_state.amp_enabled,
				scaler=scaler,
				checkpoint_kind='step',
				batch_index=step_state.batch_index,
				rng_state=_rng_state_for_step_checkpoint(
					dataloader=dataloader,
					epoch_start_dataloader_rng_state=epoch_start_rng_state,
					batch_index=step_state.batch_index,
				),
			)

		state = train_mae_one_epoch(
			model=model,
			dataloader=dataloader,
			optimizer=optimizer,
			device=device,
			epoch=epoch,
			patch_size_xyz=_xyz_config(model_config, 'patch_size'),
			loss_config=loss_config,
			amp_enabled=amp_enabled,
			scaler=scaler,
			global_step=state.global_step,
			max_steps=remaining_steps,
			diagnostics_dir=diagnostics_dir,
			grad_clip_norm=grad_clip_norm,
			skip_batches=skip_batches,
			visualization_config=visualization_config,
			step_callback=save_step_checkpoint,
			run_config=config,
		)
		print_epoch_metrics(epoch, state.metrics)
		checkpoint_kind: Literal['step', 'epoch'] = (
			'epoch' if state.completed_epoch else 'step'
		)
		checkpoint_path = _save_mae_checkpoint(
			output_root / f'mae_epoch_{epoch:04d}.pt',
			model=model,
			optimizer=optimizer,
			epoch=epoch,
			config=config,
			metrics={**state.metrics, 'amp_enabled': float(state.amp_enabled)},
			global_step=state.global_step,
			amp_enabled=state.amp_enabled,
			scaler=scaler,
			checkpoint_kind=checkpoint_kind,
			batch_index=None if state.completed_epoch else state.last_batch_index,
			rng_state=(
				_rng_state_with_dataloader(dataloader)
				if state.completed_epoch
				else _rng_state_for_step_checkpoint(
					dataloader=dataloader,
					epoch_start_dataloader_rng_state=epoch_start_dataloader_rng_state,
					batch_index=state.last_batch_index,
				)
			),
		)
		if max_steps is not None and state.global_step >= max_steps:
			break

	if checkpoint_path is None:
		msg = 'no MAE training epochs were run'
		raise ValueError(msg)
	return checkpoint_path


def prepare_run_directory(
	*,
	output_root: Path,
	resume: str | Path | None,
	allow_overwrite: bool,
) -> None:
	"""Create or validate the run output directory."""
	output_root.mkdir(parents=True, exist_ok=True)
	if resume is not None or allow_overwrite:
		return
	entries = list(output_root.iterdir())
	if entries:
		msg = (
			f'output_root is nonempty: {output_root}. Set '
			'train.allow_overwrite_output=true or use --resume.'
		)
		raise FileExistsError(msg)


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


def _mae_debug_visualization_config(
	config: Mapping[str, object],
	output_root: Path,
) -> MaeDebugVisualizationConfig | None:
	visualization = config.get('visualization')
	if visualization is None:
		return None
	if not isinstance(visualization, Mapping):
		msg = f'visualization must be a mapping; got {visualization!r}'
		raise TypeError(msg)
	mae_debug = visualization.get('mae_debug')
	if mae_debug is None:
		return None
	if not isinstance(mae_debug, Mapping):
		msg = f'visualization.mae_debug must be a mapping; got {mae_debug!r}'
		raise TypeError(msg)
	if not _bool_config(mae_debug, 'enabled', default=False):
		return None

	output_dir_value = mae_debug.get('output_dir')
	if output_dir_value is None:
		output_dir = output_root / 'visualizations' / 'mae_debug'
	elif isinstance(output_dir_value, str):
		output_dir = Path(output_dir_value)
	else:
		msg = (
			'visualization.mae_debug.output_dir must be a string or null; '
			f'got {output_dir_value!r}'
		)
		raise TypeError(msg)

	return MaeDebugVisualizationConfig(
		output_dir=output_dir,
		every_steps=_optional_int_config(mae_debug, 'every_steps'),
		every_epochs=_optional_int_config(mae_debug, 'every_epochs'),
		max_samples=_int_config(mae_debug, 'max_samples', 1),
		xy_slice_index=_optional_any_int_config(mae_debug, 'xy_slice_index'),
		xz_slice_y_index=_optional_any_int_config(mae_debug, 'xz_slice_y_index'),
		dpi=_int_config(mae_debug, 'dpi', 160),
		clip_percentiles=_float_pair_config(
			mae_debug,
			'clip_percentiles',
			default=(1.0, 99.0),
		),
		columns=_string_tuple_config(
			mae_debug,
			'columns',
			default=(
				'input',
				'masked_input',
				'target',
				'prediction',
				'abs_error',
				'valid_mask',
			),
		),
		panel_width=_float_config(mae_debug, 'panel_width', 2.6),
		panel_height=_float_config(mae_debug, 'panel_height', 2.4),
		invalid_color=_str_config_with_default(
			mae_debug,
			'invalid_color',
			'lightgray',
		),
	)


def _mae_debug_epoch_triggered(
	*,
	config: MaeDebugVisualizationConfig,
	epoch: int,
	already_triggered: bool,
) -> bool:
	if config.every_epochs is None or already_triggered:
		return False
	return epoch % config.every_epochs == 0


def _mae_debug_step_triggered(
	*,
	config: MaeDebugVisualizationConfig,
	global_step: int,
) -> bool:
	if config.every_steps is None:
		return False
	return global_step % config.every_steps == 0


def _save_mae_debug_visualization(  # noqa: PLR0913
	*,
	batch: Mapping[str, object],
	model_output: Mapping[str, object],
	patch_size_xyz: tuple[int, int, int],
	epoch: int,
	global_step: int,
	config: MaeDebugVisualizationConfig,
	metrics: Mapping[str, float],
) -> None:
	save_mae_debug_visualization_pngs(
		batch=batch,
		model_output=model_output,
		patch_size_xyz=patch_size_xyz,
		epoch=epoch,
		global_step=global_step,
		config=config,
		metrics=metrics,
	)


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
		view[section] = _json_safe(config.get(section))
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


def _first_compatibility_mismatch(
	left: Mapping[str, object],
	right: Mapping[str, object],
) -> str:
	for key in left:
		if left.get(key) == right.get(key):
			continue
		left_value = left.get(key)
		right_value = right.get(key)
		if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
			child_keys = sorted(
				left_value.keys() | right_value.keys(),
				key=str,
			)
			for child_key in child_keys:
				if left_value.get(child_key) != right_value.get(child_key):
					return f'{key}.{child_key}'
		return str(key)
	return 'config'


def _complete_mae_training_config(config: Mapping[str, object]) -> dict[str, object]:
	if not isinstance(config, Mapping):
		msg = f'config must be a mapping; got {config!r}'
		raise TypeError(msg)
	resolved = deepcopy(dict(config))
	stage = resolved.get('stage', STAGE_MAE_TRAINING)
	if stage != STAGE_MAE_TRAINING:
		msg = f'config.stage must be train_amp_mae; got {stage!r}'
		raise ValueError(msg)
	resolved['stage'] = STAGE_MAE_TRAINING
	for section in ('paths', 'manifests', 'data', 'model', 'masking', 'loss', 'train'):
		_runtime_mapping(resolved, section)
	_merge_runtime_defaults(resolved, 'data', DEFAULT_MAE_DATA_OPTIONS)
	_merge_runtime_defaults(resolved, 'loss', DEFAULT_MAE_LOSS_OPTIONS)
	_merge_runtime_defaults(resolved, 'train', DEFAULT_MAE_TRAIN_OPTIONS)
	_merge_runtime_defaults(resolved, 'zero_mask', DEFAULT_ZERO_MASK_CONTRACT)
	_merge_runtime_fixed(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_runtime_fixed(resolved, 'model', FIXED_MODEL_CONTRACT)
	_merge_runtime_fixed(resolved, 'masking', FIXED_MASKING_CONTRACT)
	_merge_runtime_fixed(resolved, 'loss', FIXED_LOSS_CONTRACT)
	return resolved


def _merge_runtime_defaults(
	config: dict[str, object],
	section: str,
	defaults: Mapping[str, object],
) -> None:
	current = config.get(section)
	if current is None:
		config[section] = deepcopy(dict(defaults))
		return
	if not isinstance(current, Mapping):
		msg = f'{section} must be a mapping'
		raise TypeError(msg)
	config[section] = {**deepcopy(dict(defaults)), **dict(current)}


def _merge_runtime_fixed(
	config: dict[str, object],
	section: str,
	fixed_values: Mapping[str, object],
) -> None:
	current = _runtime_mapping(config, section)
	for key, fixed_value in fixed_values.items():
		if key in current and current[key] != fixed_value:
			msg = (
				f'{section}.{key} is fixed by the amplitude-only training '
				f'contract; got {current[key]!r}'
			)
			raise ValueError(msg)
	config[section] = {**deepcopy(dict(fixed_values)), **dict(current)}


def _runtime_mapping(
	config: Mapping[str, object],
	section: str,
) -> Mapping[str, object]:
	value = config.get(section)
	if not isinstance(value, Mapping):
		msg = f'{section} must be a mapping'
		raise TypeError(msg)
	return value


def _build_model(model_config: Mapping[str, object]) -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		in_channels=_int_config(model_config, 'in_channels', 1),
		out_channels=_int_config(model_config, 'out_channels', 1),
		patch_size_xyz=_xyz_config(model_config, 'patch_size'),
		encoder_dim=_int_config(model_config, 'encoder_dim', 384),
		encoder_depth=_int_config(model_config, 'encoder_depth', 8),
		encoder_heads=_int_config(model_config, 'encoder_heads', 6),
		decoder_dim=_int_config(model_config, 'decoder_dim', 256),
		decoder_depth=_int_config(model_config, 'decoder_depth', 4),
		decoder_heads=_int_config(model_config, 'decoder_heads', 4),
	)


def _snapshot_run_inputs(
	*,
	output_root: Path,
	config: Mapping[str, object],
	overwrite: bool = False,
) -> None:
	_write_json_snapshot(
		output_root / 'resolved_config.json',
		_to_json_safe(config),
		overwrite=overwrite,
	)
	manifest_path = _manifest_train_path(config)
	_copy_snapshot(manifest_path, output_root / 'manifest.json', overwrite=overwrite)
	path_list = _configured_path_list(config)
	if path_list is not None:
		inputs_dir = output_root / 'inputs'
		inputs_dir.mkdir(parents=True, exist_ok=True)
		_copy_snapshot(path_list, inputs_dir / path_list.name, overwrite=overwrite)
	_write_json_snapshot(
		output_root / 'run_metadata.json',
		{
			'created_at_utc': datetime.now(timezone.utc).isoformat(),
			'git_commit': _git_commit(),
			'package_version': getattr(seis_ssl_cluster, '__version__', None),
		},
		overwrite=overwrite,
	)


def _write_json_snapshot(path: Path, payload: object, *, overwrite: bool) -> None:
	if path.exists() and not overwrite:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
	path.write_text(f'{text}\n', encoding='utf-8')


def _copy_snapshot(source: Path, target: Path, *, overwrite: bool) -> None:
	if target.exists() and not overwrite:
		return
	target.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(source, target)


def _configured_path_list(config: Mapping[str, object]) -> Path | None:
	manifests = config.get('manifests')
	if not isinstance(manifests, Mapping):
		return None
	path_value = manifests.get('train_path_list')
	if path_value is None:
		return None
	if not isinstance(path_value, str) or not path_value:
		msg = (
			'manifests.train_path_list must be a non-empty string when '
			f'configured; got {path_value!r}'
		)
		raise ValueError(msg)
	path = Path(path_value)
	if not path.is_file():
		msg = _manifest_path_error(
			f'manifests.train_path_list does not exist: {path}',
		)
		raise FileNotFoundError(msg)
	return path


def _git_commit() -> str | None:
	git = shutil.which('git')
	if git is None:
		return None
	try:
		return subprocess.check_output(  # noqa: S603
			[git, 'rev-parse', 'HEAD'],
			cwd=Path(__file__).resolve().parents[3],
			text=True,
			stderr=subprocess.DEVNULL,
		).strip()
	except (OSError, subprocess.CalledProcessError):
		return None


def _manifest_train_path(config: Mapping[str, object]) -> Path:
	manifests = config.get('manifests')
	if not isinstance(manifests, Mapping):
		msg = _manifest_path_error('manifests.train is required')
		raise TypeError(msg)
	if 'train' not in manifests:
		msg = _manifest_path_error('manifests.train is required')
		raise ValueError(msg)
	path_value = manifests.get('train')
	if not isinstance(path_value, str) or not path_value:
		msg = _manifest_path_error(
			f'manifests.train must be a non-empty string; got {path_value!r}',
		)
		raise ValueError(msg)
	path = Path(path_value)
	if not path.is_file():
		msg = _manifest_path_error(f'manifests.train does not exist: {path}')
		raise FileNotFoundError(msg)
	return path


def _manifest_path_error(reason: str) -> str:
	return f'{reason}. {_MANIFEST_BUILD_HINT}'


def _resolve_output_root(paths_config: Mapping[str, object]) -> Path:
	value = paths_config.get('output_root')
	if isinstance(value, str) and value:
		return Path(value)
	artifact_root = paths_config.get('artifact_root')
	if not isinstance(artifact_root, str) or not artifact_root:
		msg = 'paths.artifact_root must be a non-empty string'
		raise TypeError(msg)
	return Path(artifact_root) / 'runs' / 'train_amp_mae'


def _resolve_device(train_config: Mapping[str, object]) -> torch.device:
	device_name = train_config.get('device')
	if device_name is None or device_name == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if not isinstance(device_name, str):
		msg = f'train.device must be a string; got {device_name!r}'
		raise TypeError(msg)
	if device_name not in {'cpu', 'cuda'}:
		msg = 'train.device must be "auto", "cpu", or "cuda"'
		raise ValueError(msg)
	device = torch.device(device_name)
	if device.type == 'cuda' and not torch.cuda.is_available():
		msg = 'train.device requested CUDA, but CUDA is not available'
		raise ValueError(msg)
	return device


def _resolve_diagnostics_dir(
	train_config: Mapping[str, object],
	output_root: Path,
) -> Path:
	value = train_config.get('diagnostics_dir')
	if value is None:
		return output_root / 'diagnostics'
	if not isinstance(value, str):
		msg = f'train.diagnostics_dir must be a string; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if path.is_absolute():
		return path
	return output_root / path


def _clip_and_check_gradients(  # noqa: PLR0913
	*,
	model: torch.nn.Module,
	grad_clip_norm: float,
	global_step: int,
	epoch: int,
	batch_index: int,
	batch: Mapping[str, object],
	output: Mapping[str, object],
	losses: Mapping[str, torch.Tensor],
	amp_enabled: bool,
	diagnostics_dir: Path | None,
	patch_size_xyz: tuple[int, int, int],
	run_config: Mapping[str, object] | None,
) -> float:
	grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
	if torch.isfinite(grad_norm.detach()).all():
		return float(grad_norm.detach().float().cpu().item())
	_raise_nonfinite(
		kind='gradient norm',
		global_step=global_step,
		epoch=epoch,
		batch_index=batch_index,
		batch=batch,
		output=output,
		losses=losses,
		amp_enabled=amp_enabled,
		diagnostics_dir=diagnostics_dir,
		patch_size_xyz=patch_size_xyz,
		grad_norm=grad_norm,
		run_config=run_config,
	)
	raise AssertionError('unreachable after non-finite gradient diagnostic')


def _raise_nonfinite(  # noqa: PLR0913
	*,
	kind: str,
	global_step: int,
	epoch: int,
	batch_index: int,
	batch: Mapping[str, object],
	output: Mapping[str, object],
	losses: Mapping[str, torch.Tensor],
	amp_enabled: bool,
	diagnostics_dir: Path | None,
	patch_size_xyz: tuple[int, int, int],
	grad_norm: torch.Tensor | None = None,
	run_config: Mapping[str, object] | None = None,
) -> None:
	if diagnostics_dir is not None:
		diagnostic_path = _write_json_diagnostic(
			_build_nonfinite_diagnostic(
				global_step=global_step,
				epoch=epoch,
				batch_index=batch_index,
				batch=batch,
				output=output,
				losses=losses,
				amp_enabled=amp_enabled,
				patch_size_xyz=patch_size_xyz,
				grad_norm=grad_norm,
				run_config=run_config,
			),
			diagnostics_dir / f'nonfinite_mae_step_{global_step:08d}.json',
		)
		msg = (
			f'non-finite MAE {kind} at epoch {epoch}, step {global_step}; '
			f'diagnostic written to {diagnostic_path}'
		)
	else:
		msg = f'non-finite MAE {kind} at epoch {epoch}, step {global_step}'
	raise FloatingPointError(msg)


def _build_nonfinite_diagnostic(  # noqa: PLR0913
	*,
	global_step: int,
	epoch: int,
	batch_index: int,
	batch: Mapping[str, object],
	output: Mapping[str, object],
	losses: Mapping[str, torch.Tensor],
	amp_enabled: bool,
	patch_size_xyz: tuple[int, int, int],
	grad_norm: torch.Tensor | None,
	run_config: Mapping[str, object] | None,
) -> dict[str, object]:
	coords = _json_safe(batch.get('coords'))
	coord_items = coords if isinstance(coords, list) else []
	diagnostic = {
		'global_step': int(global_step),
		'epoch': int(epoch),
		'batch_index': int(batch_index),
		'survey_id': _coord_values(coord_items, 'survey_id'),
		'local_start_xyz': _coord_values(coord_items, 'local_start_xyz'),
		'coords': coords,
		'losses': _summarize_loss_components(losses),
		'tensors': {
			'x': _summarize_tensor(batch.get('x')),
			'target': _summarize_tensor(batch.get('target')),
			'prediction': _summarize_prediction(
				output.get('pred_patches'),
				batch.get('target'),
				patch_size_xyz,
			),
			'pred_patches': _summarize_tensor(output.get('pred_patches')),
			'local_valid_mask': _summarize_tensor(batch.get('local_valid_mask')),
			'spatial_mask': _summarize_tensor(batch.get('spatial_mask')),
			'visible_spatial_mask': _summarize_tensor(
				batch.get('visible_spatial_mask'),
			),
		},
		'valid_voxel_count': _valid_voxel_count(batch.get('local_valid_mask')),
		'grad_norm': _summarize_tensor(grad_norm),
		'torch_amp_enabled': bool(amp_enabled),
	}
	if run_config is not None:
		diagnostic['config'] = _json_safe(run_config)
	return diagnostic


def _summarize_prediction(
	pred_patches: object,
	target: object,
	patch_size_xyz: tuple[int, int, int],
) -> dict[str, object]:
	if not isinstance(pred_patches, torch.Tensor):
		return {'present': False, 'reason': 'pred_patches is not a tensor'}
	if not isinstance(target, torch.Tensor):
		return {'present': False, 'reason': 'target is not a tensor'}
	if target.ndim != 5:
		return {
			'present': False,
			'reason': f'target shape is not [B, C, X, Y, Z]: {tuple(target.shape)!r}',
		}

	spatial_shape = tuple(int(dim) for dim in target.shape[2:])
	if any(
		size % patch != 0
		for size, patch in zip(spatial_shape, patch_size_xyz, strict=True)
	):
		return {
			'present': False,
			'reason': (
				'target spatial shape is not divisible by patch_size_xyz: '
				f'{spatial_shape!r} vs {patch_size_xyz!r}'
			),
		}

	grid_size_xyz = tuple(
		size // patch for size, patch in zip(spatial_shape, patch_size_xyz, strict=True)
	)
	try:
		prediction = unpatchify_3d(pred_patches, patch_size_xyz, grid_size_xyz)
	except ValueError as exc:
		return {'present': False, 'reason': str(exc)}
	return _summarize_tensor(prediction)


def _write_json_diagnostic(payload: Mapping[str, object], path: str | Path) -> Path:
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
	output_path.write_text(f'{text}\n', encoding='utf-8')
	return output_path


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
	summary.update(
		{
			'finite_count': finite_count,
			'nan_count': int(torch.isnan(value).sum().cpu().item()),
			'posinf_count': int(torch.isposinf(value).sum().cpu().item()),
			'neginf_count': int(torch.isneginf(value).sum().cpu().item()),
			'all_finite': finite_count == value.numel(),
		},
	)
	if finite_count == 0:
		summary.update({'min': None, 'max': None, 'mean': None})
		return summary
	finite_values = value[finite].float().cpu()
	summary.update(
		{
			'min': _json_safe_number(finite_values.min().item()),
			'max': _json_safe_number(finite_values.max().item()),
			'mean': _json_safe_number(finite_values.mean().item()),
		},
	)
	return summary


def _summarize_loss_components(
	losses: Mapping[str, torch.Tensor],
) -> dict[str, object]:
	return {key: _summarize_loss_value(value) for key, value in losses.items()}


def _summarize_loss_value(value: torch.Tensor) -> dict[str, object]:
	if not isinstance(value, torch.Tensor):
		return {'present': True, 'type': type(value).__name__, 'repr': repr(value)}
	if value.numel() != 1:
		return _summarize_tensor(value)
	item = value.detach().cpu().item()
	if isinstance(item, float) and not math.isfinite(item):
		return {'value': None, 'finite': False, 'repr': repr(item)}
	return {'value': _json_safe_number(item), 'finite': True}


def _valid_voxel_count(value: object) -> int | None:
	if not isinstance(value, torch.Tensor):
		return None
	if value.dtype != torch.bool:
		return None
	return int(value.detach().sum().cpu().item())


def _coord_values(coords: Sequence[object], key: str) -> list[object]:
	values: list[object] = []
	for coord in coords:
		if isinstance(coord, Mapping):
			values.append(_json_safe(coord.get(key)))
		else:
			values.append(None)
	return values


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


def _to_json_safe(value: object) -> object:
	return _json_safe(value)


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


def _required_tensor(
	mapping: Mapping[str, object],
	key: str,
) -> torch.Tensor:
	value = mapping[key]
	if not isinstance(value, torch.Tensor):
		msg = f'{key} must be a torch.Tensor; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _int_config(parent: Mapping[str, object], key: str, default: int) -> int:
	value = parent.get(key, default)
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{key} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _nonnegative_int_config(
	parent: Mapping[str, object],
	key: str,
	default: int,
) -> int:
	value = parent.get(key, default)
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value < 0:
		msg = f'{key} must be nonnegative; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_int_config(parent: Mapping[str, object], key: str) -> int | None:
	value = parent.get(key)
	if value is None:
		return None
	return _int_config(parent, key, 1)


def _optional_any_int_config(parent: Mapping[str, object], key: str) -> int | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer or null; got {value!r}'
		raise TypeError(msg)
	return value


def _float_config(parent: Mapping[str, object], key: str, default: float) -> float:
	value = parent.get(key, default)
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	return float(value)


def _float_pair_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: tuple[float, float],
) -> tuple[float, float]:
	value = parent.get(key, default)
	if not isinstance(value, list | tuple) or len(value) != 2:
		msg = f'{key} must be a length-2 float sequence; got {value!r}'
		raise TypeError(msg)
	low, high = value
	if (
		not isinstance(low, float | int)
		or isinstance(low, bool)
		or not isinstance(high, float | int)
		or isinstance(high, bool)
	):
		msg = f'{key} must contain numeric percentile values; got {value!r}'
		raise TypeError(msg)
	resolved = (float(low), float(high))
	if not 0.0 <= resolved[0] < resolved[1] <= 100.0:
		msg = f'{key} must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)
	return resolved


def _optional_positive_float_config(
	parent: Mapping[str, object],
	key: str,
) -> float | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not math.isfinite(number) or number <= 0.0:
		msg = f'{key} must be finite and positive; got {value!r}'
		raise ValueError(msg)
	return number


def _bool_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: bool,
) -> bool:
	value = parent.get(key, default)
	if not isinstance(value, bool):
		msg = f'{key} must be a bool; got {value!r}'
		raise TypeError(msg)
	return value


def _str_config_with_default(
	parent: Mapping[str, object],
	key: str,
	default: str,
) -> str:
	value = parent.get(key, default)
	if not isinstance(value, str) or not value:
		msg = f'{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _string_tuple_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: tuple[str, ...],
) -> tuple[str, ...]:
	value = parent.get(key, default)
	if not isinstance(value, list | tuple) or not value:
		msg = f'{key} must be a non-empty string sequence; got {value!r}'
		raise TypeError(msg)
	resolved: list[str] = []
	for item in value:
		if not isinstance(item, str) or not item:
			msg = f'{key} must contain non-empty strings; got {value!r}'
			raise TypeError(msg)
		resolved.append(item)
	return tuple(resolved)


def _xyz_config(parent: Mapping[str, object], key: str) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list | tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
	):
		msg = f'{key} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(cast('tuple[int, int, int]', value))
	if any(item <= 0 for item in xyz):
		msg = f'{key} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _loss_mode(value: object) -> Literal['huber', 'l1', 'mse']:
	if value not in {'huber', 'l1', 'mse'}:
		msg = f'reconstruction must be "huber", "l1", or "mse"; got {value!r}'
		raise ValueError(msg)
	return cast('Literal["huber", "l1", "mse"]', value)


__all__ = [
	'MaeStepState',
	'MaeTrainingState',
	'ResumeState',
	'prepare_run_directory',
	'run_mae_pretraining',
	'train_mae_one_epoch',
]
