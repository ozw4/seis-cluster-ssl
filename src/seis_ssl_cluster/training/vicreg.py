"""Config-driven local VICReg pretraining for 3D amplitude encoders."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from seis_ssl_cluster.config.schema import (
	HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY,
	IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
)
from seis_ssl_cluster.config.vicreg import resolve_vicreg_pretraining_method
from seis_ssl_cluster.data.amplitude_dataset import AmplitudePretrainDataset
from seis_ssl_cluster.data.barlow_twins_dataset import (
	LocalBarlowTwinsD4TraceDropPretrainDataset,
	LocalBarlowTwinsPretrainDataset,
)
from seis_ssl_cluster.data.schema import read_manifest_json
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D
from seis_ssl_cluster.models.vicreg import VICRegLoss
from seis_ssl_cluster.training.barlow_twins import (
	_boolean,
	_build_backbone,
	_clip_and_check_gradients,
	_d4_augmentation_metrics,
	_floating,
	_integer,
	_mapping,
	_new_d4_augmentation_totals,
	_optional_float,
	_optional_int,
	_prepare_run_directory,
	_read_history,
	_resolve_device,
	_resolve_precision,
	_seed_everything,
	_string,
	_tensor,
	_update_d4_augmentation_totals,
	_write_json,
	_xyz,
	_zero_mask,
)
from seis_ssl_cluster.training.barlow_twins_checkpoint import update_best_checkpoint
from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.dataloaders import build_barlow_twins_dataloader
from seis_ssl_cluster.training.logging import print_epoch_metrics
from seis_ssl_cluster.training.vicreg_checkpoint import (
	VICRegResumeState,
	load_vicreg_checkpoint,
	restore_vicreg_checkpoint,
	save_vicreg_checkpoint,
)
from seis_ssl_cluster.training.vicreg_continuation import (
	configure_vicreg_continuation_trainability,
	load_vicreg_continuation_weights,
)

_LOSS_METRICS = (
	'training_loss',
	'invariance_loss',
	'variance_loss',
	'covariance_loss',
	'projection_std_mean',
	'projection_std_min',
	'covariance_offdiag_rms',
	'weighted_invariance',
	'weighted_variance',
	'weighted_covariance',
)


@dataclass(frozen=True)
class VICRegTrainingState:
	"""Training progress and averaged metrics for one epoch invocation."""

	epoch: int
	global_step: int
	metrics: dict[str, float]
	completed_epoch: bool


def train_vicreg_one_epoch(  # noqa: PLR0913
	*,
	model: BarlowTwins3D,
	loss_fn: VICRegLoss,
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
	augmentation_policy: str | None = None,
) -> VICRegTrainingState:
	"""Train local VICReg for one epoch or a per-call step limit."""
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
	d4_totals = _new_d4_augmentation_totals(augmentation_policy, device=device)
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
		_update_d4_augmentation_totals(d4_totals, batch)
		optimizer.zero_grad(set_to_none=True)
		with torch.autocast(
			device_type=device.type,
			dtype=amp_dtype,
			enabled=amp_enabled,
		):
			outputs = model.forward_local(
				_tensor(batch, 'view_a'),
				_tensor(batch, 'view_b'),
				valid_mask_a=_tensor(batch, 'valid_mask_a'),
				valid_mask_b=_tensor(batch, 'valid_mask_b'),
				local_pair_indices_a=_tensor(batch, 'local_pair_indices_a'),
				local_pair_indices_b=_tensor(batch, 'local_pair_indices_b'),
			)
			losses = loss_fn(outputs['z_a'], outputs['z_b'])
			loss = losses['loss']
		if not bool(torch.isfinite(loss).item()):
			raise FloatingPointError(
				'non-finite VICReg loss '
				f'at epoch={epoch}, global_step={global_step}'
			)
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
		global_step += 1
		batches += 1
		totals['training_loss'] += float(loss.detach().cpu())
		for key in _LOSS_METRICS[1:]:
			totals[key] += float(losses[key].detach().cpu())
		totals['gradient_norm'] += float(grad_norm.detach().cpu())
		totals['learning_rate'] += float(optimizer.param_groups[0]['lr'])
		totals['step_time_seconds'] += time.perf_counter() - step_started
	if batches == 0:
		raise ValueError('VICReg dataloader produced no batches')
	metrics = {key: value / batches for key, value in totals.items()}
	metrics.update(_d4_augmentation_metrics(d4_totals))
	metrics['peak_cuda_memory_mib'] = (
		float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
		if device.type == 'cuda'
		else 0.0
	)
	return VICRegTrainingState(
		epoch=epoch,
		global_step=global_step,
		metrics=metrics,
		completed_epoch=batches == len(dataloader),
	)


def run_vicreg_pretraining(  # noqa: C901, PLR0915
	config: Mapping[str, object],
	*,
	resume: str | Path | None = None,
) -> Path:
	"""Run local VICReg pretraining from a resolved stage config."""
	if config.get('stage') != 'vicreg_training':
		raise ValueError('config must be resolved for vicreg_training')
	train = _mapping(config, 'train')
	model_config = _mapping(config, 'model')
	data = _mapping(config, 'data')
	vicreg = _mapping(config, 'vicreg')
	resolve_vicreg_pretraining_method(config)
	local_pairs_per_crop = _integer(vicreg, 'local_pairs_per_crop')
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
		min_valid_token_count=local_pairs_per_crop,
	)
	augmentation_policy = (
		_string(augmentations, 'policy') if 'policy' in augmentations else None
	)
	if augmentation_policy == IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY:
		dataset = LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=0.0,
			gaussian_noise_std=_floating(augmentations, 'gaussian_noise_std'),
			require_distinct_horizontal_views=False,
		)
	elif augmentation_policy == HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY:
		dataset = LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=_floating(
				augmentations, 'horizontal_flip_probability'
			),
			gaussian_noise_std=_floating(augmentations, 'gaussian_noise_std'),
		)
	elif augmentation_policy == HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY:
		dataset = LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=_floating(
				augmentations, 'horizontal_flip_probability'
			),
			trace_drop_probability=_floating(
				augmentations, 'trace_drop_probability'
			),
		)
	elif (
		augmentation_policy
		== HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY
	):
		dataset = LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=_floating(
				augmentations, 'horizontal_flip_probability'
			),
			z_filter_side_weight=_floating(augmentations, 'z_filter_side_weight'),
		)
	elif augmentation_policy == XY_D4_TRACE_DROP_AUGMENTATION_POLICY:
		dataset = LocalBarlowTwinsD4TraceDropPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			reflection_probability=_floating(
				augmentations, 'reflection_probability'
			),
			trace_drop_probability=_floating(
				augmentations, 'trace_drop_probability'
			),
		)
	else:
		dataset = LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
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
		projector_dim=_integer(vicreg, 'projector_dim'),
	).to(device)
	loss_fn = VICRegLoss(
		invariance_weight=_floating(vicreg, 'invariance_weight'),
		variance_weight=_floating(vicreg, 'variance_weight'),
		covariance_weight=_floating(vicreg, 'covariance_weight'),
		variance_target_std=_floating(vicreg, 'variance_target_std'),
		variance_eps=_floating(vicreg, 'variance_eps'),
	)
	optimizer_parameters, continuation_lineage = _initialize_vicreg_model(
		model,
		continuation=continuation,
		resume=resume,
		model_config=model_config,
		vicreg_config=vicreg,
	)
	optimizer = torch.optim.AdamW(
		optimizer_parameters,
		lr=_floating(train, 'lr'),
		weight_decay=_floating(train, 'weight_decay'),
	)
	resume_state = VICRegResumeState(start_epoch=1, global_step=0, resume_count=0)
	if resume is not None:
		resume_payload = load_vicreg_checkpoint(resume, map_location=device)
		resume_state = restore_vicreg_checkpoint(
			resume_payload,
			backbone=backbone,
			projector=model.projector,
			optimizer=optimizer,
			scaler=scaler,
			scaler_required=precision.scaler_required,
			config=config,
			dataloader_generator=dataloader.generator,
		)
		continuation_lineage = _resumed_continuation_lineage(
			resume_payload,
			continuation=continuation,
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
		state = train_vicreg_one_epoch(
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
			augmentation_policy=augmentation_policy,
		)
		global_step = state.global_step
		print_epoch_metrics(epoch, state.metrics)
		history.append({'epoch': epoch, 'global_step': global_step, **state.metrics})
		_write_json(output_root / 'history.json', history)
		checkpoint_path = save_vicreg_checkpoint(
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
			continuation_lineage=continuation_lineage,
			resume_count=resume_state.resume_count,
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
		raise ValueError('no VICReg training epochs were run')
	return checkpoint_path


def _initialize_vicreg_model(
	model: BarlowTwins3D,
	*,
	continuation: Mapping[str, object] | None,
	resume: str | Path | None,
	model_config: Mapping[str, object],
	vicreg_config: Mapping[str, object],
) -> tuple[tuple[torch.nn.Parameter, ...], dict[str, object] | None]:
	if continuation is None:
		return tuple(model.pretraining_parameters()), None
	lineage: dict[str, object] | None = None
	if resume is None:
		init_checkpoint = _string(continuation, 'init_checkpoint')
		init_sha256 = load_vicreg_continuation_weights(
			model,
			init_checkpoint,
			expected_model_config=model_config,
			expected_vicreg_config=vicreg_config,
		)
		lineage = {
			'schema_version': 1,
			'init_checkpoint': init_checkpoint,
			'init_checkpoint_sha256': init_sha256,
			'resume_count': 0,
		}
	return (
		configure_vicreg_continuation_trainability(
			model,
			unfreeze_top_blocks=_integer(continuation, 'unfreeze_top_blocks'),
		),
		lineage,
	)


def _resumed_continuation_lineage(
	payload: Mapping[str, Any],
	*,
	continuation: Mapping[str, object] | None,
) -> dict[str, object] | None:
	value = payload.get('continuation_lineage')
	if continuation is None:
		if value is not None:
			raise ValueError(
				'base VICReg checkpoint must not have continuation lineage'
			)
		return None
	if not isinstance(value, Mapping):
		raise TypeError('continued VICReg resume checkpoint is missing lineage')
	if value.get('schema_version') != 1:
		raise ValueError('continuation lineage schema_version must be 1')
	checkpoint = _string(continuation, 'init_checkpoint')
	if value.get('init_checkpoint') != checkpoint:
		raise ValueError('continuation lineage init_checkpoint does not match config')
	sha256 = value.get('init_checkpoint_sha256')
	if not isinstance(sha256, str) or len(sha256) != 64 or any(
		character not in '0123456789abcdef' for character in sha256
	):
		raise ValueError('continuation lineage SHA-256 is invalid')
	resume_count = value.get('resume_count')
	if isinstance(resume_count, bool) or not isinstance(resume_count, int):
		raise TypeError('continuation lineage resume_count must be an integer')
	if resume_count < 0:
		raise ValueError('continuation lineage resume_count must be non-negative')
	return {
		'schema_version': 1,
		'init_checkpoint': checkpoint,
		'init_checkpoint_sha256': sha256,
		'resume_count': resume_count + 1,
	}


def _existing_best_loss(path: Path) -> float | None:
	if not path.is_file():
		return None
	payload = load_vicreg_checkpoint(path, map_location='cpu')
	metrics = payload.get('metrics')
	if not isinstance(metrics, Mapping):
		return None
	value = metrics.get('training_loss')
	return float(value) if isinstance(value, int | float) else None


__all__ = [
	'VICRegTrainingState',
	'run_vicreg_pretraining',
	'train_vicreg_one_epoch',
]
