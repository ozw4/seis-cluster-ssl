"""Head-only stratigraphic HMM pretext training."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import torch

from seis_ssl_cluster.data import (
	NopimsStratPseudoTargetDataset,
	read_manifest_json,
)
from seis_ssl_cluster.embedding.extractor import build_model_from_config
from seis_ssl_cluster.stratigraphy import (
	OrderedPrototypeHead,
	discover_pseudo_target_inputs,
	feature_distillation_loss,
	structured_hmm_prototype_loss,
	usage_entropy_floor_loss,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint, restore_rng_state
from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.dataloaders import build_strat_pseudo_target_dataloader
from seis_ssl_cluster.training.mae import prepare_run_directory
from seis_ssl_cluster.training.strat_hmm.runtime import (
	_bool_config,
	_dataloader_generator_state,
	_float_config,
	_int_config,
	_load_existing_best_score,
	_mapping,
	_non_empty_string,
	_optional_float_config,
	_optional_int_config,
	_optional_positive_int_config,
	_path_config,
	_required_tensor,
	_resolve_device,
	_restore_dataloader_generator_state,
	_rng_state_for_step_checkpoint,
	_rng_state_with_dataloader,
	_snapshot_run_inputs,
	_to_json_safe,
	_trainability_summary_payload,
	_write_run_metadata,
	_xyz_config,
	_zero_mask_from_config,
)
from seis_ssl_cluster.training.strat_hmm.state import (
	StratHmmHeadOnlyComponents,
	StratHmmResumeState,
	StratHmmTrainingState,
	TrainabilitySummary,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	save_strat_hmm_rolling_checkpoint,
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


def run_strat_hmm_pretext_training(  # noqa: C901, PLR0915
	config: Mapping[str, object],
	*,
	resume: str | Path | None = None,
) -> Path:
	"""Run strat HMM pretext training from ``config``."""
	train_config = _mapping(config, 'train')
	paths_config = _mapping(config, 'paths')
	data_config = _mapping(config, 'data')
	model_config = _mapping(config, 'model')
	pseudo_config = _mapping(config, 'pseudo_targets')
	device = _resolve_device(train_config)
	seed = _int_config(train_config, 'seed', 42)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed_all(seed)

	output_root = _path_config(paths_config, 'output_root')
	prepare_run_directory(
		output_root=output_root,
		resume=resume,
		allow_overwrite=_bool_config(
			train_config,
			'allow_overwrite_output',
			default=False,
		),
	)
	_snapshot_run_inputs(
		output_root=output_root,
		config=config,
		overwrite=(
			_bool_config(train_config, 'allow_overwrite_output', default=False)
			and resume is None
		),
	)

	manifests = read_manifest_json(_path_config(_mapping(config, 'manifests'), 'train'))
	pseudo_inputs = discover_pseudo_target_inputs(
		_path_config(pseudo_config, 'input_dir'),
		k=_int_config(pseudo_config, 'k', 1),
	)
	dataset = NopimsStratPseudoTargetDataset(
		manifests,
		pseudo_inputs,
		local_crop_size_xyz=_xyz_config(data_config, 'local_crop_size'),
		patch_size_xyz=_xyz_config(model_config, 'patch_size'),
		seed=seed,
		samples_per_epoch=_int_config(train_config, 'samples_per_epoch', 1),
		zero_mask=_zero_mask_from_config(config),
		min_valid_fraction=_float_config(data_config, 'min_valid_fraction', 0.0),
		max_resample_attempts=_int_config(
			data_config,
			'max_resample_attempts',
			16,
		),
		normalized_clip_abs=_optional_float_config(data_config, 'normalized_clip_abs'),
		amplitude_agc=data_config.get('amplitude_agc'),
		min_confidence=_float_config(pseudo_config, 'min_confidence', 0.0),
	)
	dataloader = build_strat_pseudo_target_dataloader(
		dataset,
		batch_size=_int_config(train_config, 'batch_size', 1),
		num_workers=_int_config(train_config, 'num_workers', 0),
		shuffle=_bool_config(train_config, 'shuffle', default=True),
		seed=seed,
		device=device,
	)
	components = build_strat_hmm_head_only_components(config, device=device)
	_write_run_metadata(
		output_root=output_root,
		trainability_summary=components.trainability_summary,
		overwrite=True,
	)
	amp_enabled = (
		_bool_config(train_config, 'amp', default=False)
		and device.type == 'cuda'
		and torch.cuda.is_available()
	)
	scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
	resume_state = StratHmmResumeState(start_epoch=1, global_step=0, skip_batches=0)
	if resume is not None:
		payload = load_checkpoint(resume, map_location=device)
		resume_state = _restore_strat_hmm_checkpoint(
			payload=payload,
			student=components.student,
			head=components.head,
			optimizer=components.optimizer,
			scaler=scaler,
			amp_enabled=amp_enabled,
			config=config,
		)
		_restore_dataloader_generator_state(payload=payload, dataloader=dataloader)
		if resume_state.skip_batches >= len(dataloader):
			resume_state = StratHmmResumeState(
				start_epoch=resume_state.start_epoch + 1,
				global_step=resume_state.global_step,
				skip_batches=0,
			)
	epochs = _int_config(train_config, 'epochs', 1)
	max_steps = _optional_int_config(train_config, 'max_steps')
	checkpoint_every_steps = _optional_positive_int_config(
		train_config,
		'checkpoint_every_steps',
	)
	grad_clip_norm = _optional_float_config(train_config, 'grad_clip_norm')

	state = StratHmmTrainingState(
		epoch=resume_state.start_epoch - 1,
		global_step=resume_state.global_step,
		metrics={},
		last_batch_index=-1,
		completed_epoch=True,
	)
	checkpoint_path: Path | None = None
	best_score = _load_existing_best_score(output_root) if resume is not None else None
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
			step_state: StratHmmTrainingState,
			epoch_start_rng_state: torch.Tensor = epoch_start_dataloader_rng_state,
		) -> None:
			nonlocal best_score, checkpoint_path
			if (
				checkpoint_every_steps is None
				or step_state.global_step % checkpoint_every_steps != 0
			):
				return
			result = save_strat_hmm_rolling_checkpoint(
				output_root,
				student=components.student,
				head=components.head,
				optimizer=components.optimizer,
				epoch=step_state.epoch,
				mae_config=components.mae_checkpoint_config,
				stratigraphy_config=config,
				metrics={
					**step_state.metrics,
					**_trainability_metrics(components.trainability_summary),
				},
				global_step=step_state.global_step,
				checkpoint_kind='step',
				batch_index=step_state.last_batch_index,
				amp_enabled=amp_enabled,
				scaler=scaler,
				rng_state=_rng_state_for_step_checkpoint(
					dataloader=dataloader,
					epoch_start_dataloader_rng_state=epoch_start_rng_state,
					batch_index=step_state.last_batch_index,
				),
				trainability_summary=_trainability_summary_payload(
					components.trainability_summary,
				),
				best_score=best_score,
			)
			best_score = result.best_score
			checkpoint_path = result.latest_path

		state = train_strat_hmm_head_only_one_epoch(
			student=components.student,
			teacher=components.teacher,
			head=components.head,
			dataloader=dataloader,
			optimizer=components.optimizer,
			device=device,
			epoch=epoch,
			loss_config=_mapping(config, 'loss'),
			pseudo_target_config=pseudo_config,
			amp_enabled=amp_enabled,
			scaler=scaler,
			global_step=state.global_step,
			max_steps=remaining_steps,
			grad_clip_norm=grad_clip_norm,
			skip_batches=skip_batches,
			step_callback=save_step_checkpoint,
		)
		trainability_metrics = _trainability_metrics(components.trainability_summary)
		checkpoint_kind: Literal['step', 'epoch'] = (
			'epoch' if state.completed_epoch else 'step'
		)
		result = save_strat_hmm_rolling_checkpoint(
			output_root,
			student=components.student,
			head=components.head,
			optimizer=components.optimizer,
			epoch=epoch,
			mae_config=components.mae_checkpoint_config,
			stratigraphy_config=config,
			metrics={
				**state.metrics,
				**trainability_metrics,
				'amp_enabled': float(amp_enabled),
			},
			global_step=state.global_step,
			checkpoint_kind=checkpoint_kind,
			batch_index=None if state.completed_epoch else state.last_batch_index,
			amp_enabled=amp_enabled,
			scaler=scaler,
			rng_state=(
				_rng_state_with_dataloader(dataloader)
				if state.completed_epoch
				else _rng_state_for_step_checkpoint(
					dataloader=dataloader,
					epoch_start_dataloader_rng_state=epoch_start_dataloader_rng_state,
					batch_index=state.last_batch_index,
				)
			),
			trainability_summary=_trainability_summary_payload(
				components.trainability_summary,
			),
			best_score=best_score,
		)
		best_score = result.best_score
		checkpoint_path = result.latest_path
		if max_steps is not None and state.global_step >= max_steps:
			break

	if checkpoint_path is None:
		msg = 'no strat HMM pretext training steps were run'
		raise ValueError(msg)
	return checkpoint_path


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


def train_strat_hmm_head_only_one_epoch(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	student: AmplitudeMAE3D,
	teacher: AmplitudeMAE3D | None = None,
	head: OrderedPrototypeHead,
	dataloader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	loss_config: Mapping[str, object],
	pseudo_target_config: Mapping[str, object],
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	global_step: int = 0,
	max_steps: int | None = None,
	grad_clip_norm: float | None = None,
	skip_batches: int = 0,
	step_callback: (
		Callable[[StratHmmTrainingState], None] | None
	) = None,
) -> StratHmmTrainingState:
	"""Train the ordered prototype head for one epoch."""
	student.eval()
	head.train()
	totals: dict[str, float] = {}
	batches = 0
	last_batch_index = -1
	for batch_index, raw_batch in enumerate(dataloader):
		if batch_index < skip_batches:
			continue
		if max_steps is not None and batches >= max_steps:
			break
		batch = move_batch_to_device(raw_batch, device)
		optimizer.zero_grad(set_to_none=True)

		student_grad_enabled = any(
			parameter.requires_grad for parameter in student.parameters()
		)
		with torch.set_grad_enabled(student_grad_enabled):
			encoded = student.encode_tokens(
				_required_tensor(batch, 'x'),
				valid_mask=_required_tensor(batch, 'local_valid_mask'),
			)
		teacher_encoded = None
		if _float_config(loss_config, 'distillation_weight', 0.0) > 0.0:
			if teacher is None:
				msg = 'teacher is required when loss.distillation_weight is positive'
				raise ValueError(msg)
			teacher.eval()
			with torch.no_grad():
				teacher_encoded = teacher.encode_tokens(
					_required_tensor(batch, 'x'),
					valid_mask=_required_tensor(batch, 'local_valid_mask'),
				)
		with torch.amp.autocast('cuda', enabled=amp_enabled):
			losses = _strat_head_losses(
				head=head,
				encoded=encoded,
				teacher_encoded=teacher_encoded,
				batch=batch,
				loss_config=loss_config,
				pseudo_target_config=pseudo_target_config,
			)
			loss = losses['loss']

		if not torch.isfinite(loss).all():
			msg = (
				'non-finite strat HMM pretext loss at '
				f'epoch {epoch}, step {global_step}, batch {batch_index}'
			)
			raise FloatingPointError(msg)

		if amp_enabled:
			if scaler is None:
				msg = 'scaler is required when amp_enabled is true'
				raise ValueError(msg)
			scaler.scale(loss).backward()
			if grad_clip_norm is not None:
				scaler.unscale_(optimizer)
				_clip_gradients(
					student,
					head,
					grad_clip_norm,
					epoch=epoch,
					global_step=global_step,
					batch_index=batch_index,
				)
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			if grad_clip_norm is not None:
				_clip_gradients(
					student,
					head,
					grad_clip_norm,
					epoch=epoch,
					global_step=global_step,
					batch_index=batch_index,
				)
			optimizer.step()

		step_metrics = {
			key: float(value.detach().cpu().item())
			for key, value in losses.items()
		}
		for key, value in step_metrics.items():
			totals[key] = totals.get(key, 0.0) + value
		batches += 1
		global_step += 1
		last_batch_index = batch_index
		if step_callback is not None:
			step_callback(
				StratHmmTrainingState(
					epoch=epoch,
					global_step=global_step,
					metrics=step_metrics,
					last_batch_index=batch_index,
					completed_epoch=batch_index >= len(dataloader) - 1,
				),
			)

	if batches == 0:
		msg = 'dataloader produced no batches'
		raise ValueError(msg)
	return StratHmmTrainingState(
		epoch=epoch,
		global_step=global_step,
		metrics={key: total / batches for key, total in totals.items()},
		last_batch_index=last_batch_index,
		completed_epoch=last_batch_index >= len(dataloader) - 1,
	)


def _strat_head_losses(  # noqa: PLR0913
	*,
	head: OrderedPrototypeHead,
	encoded: Mapping[str, object],
	teacher_encoded: Mapping[str, object] | None,
	batch: Mapping[str, object],
	loss_config: Mapping[str, object],
	pseudo_target_config: Mapping[str, object],
) -> dict[str, torch.Tensor]:
	tokens = _encoded_tokens(encoded)
	logits = head(tokens).logits
	labels = _flatten_token_tensor(
		_required_tensor(batch, 'strat_labels'),
		logits,
		'strat_labels',
	).long()
	confidence = _flatten_token_tensor(
		_required_tensor(batch, 'strat_confidence'),
		logits,
		'strat_confidence',
	).to(dtype=logits.dtype)
	valid_mask = _flatten_token_tensor(
		_required_tensor(batch, 'strat_valid_mask'),
		logits,
		'strat_valid_mask',
	).bool()
	distillation_valid_mask = valid_mask
	token_valid_mask = encoded.get('token_valid_mask')
	if token_valid_mask is not None:
		student_valid_mask = _encoded_token_valid_mask(token_valid_mask, logits)
		valid_mask = valid_mask & student_valid_mask
		distillation_valid_mask = distillation_valid_mask & student_valid_mask
	if teacher_encoded is not None:
		teacher_token_valid_mask = teacher_encoded.get('token_valid_mask')
		if teacher_token_valid_mask is not None:
			teacher_valid_mask = _encoded_token_valid_mask(
				teacher_token_valid_mask,
				logits,
			)
			distillation_valid_mask = distillation_valid_mask & teacher_valid_mask
	min_confidence = _float_config(pseudo_target_config, 'min_confidence', 0.0)
	if min_confidence > 0.0:
		valid_mask = valid_mask & confidence.ge(min_confidence)

	prototype_loss = structured_hmm_prototype_loss(
		logits,
		labels,
		valid_mask=valid_mask,
		confidence=confidence,
	)
	usage_weight = _float_config(loss_config, 'usage_weight', 0.0)
	if usage_weight > 0.0:
		probs = torch.nn.functional.softmax(logits, dim=-1)
		entropy_floor = loss_config.get('entropy_floor')
		if entropy_floor is None:
			# Prompt-07 default: a weak half-uniform floor if usage loss is enabled.
			entropy_floor_value = 0.5 * math.log(logits.shape[-1])
		else:
			entropy_floor_value = float(entropy_floor)
		usage_loss = usage_entropy_floor_loss(
			probs,
			valid_mask=valid_mask,
			entropy_floor=entropy_floor_value,
		)
	else:
		probs = torch.nn.functional.softmax(logits, dim=-1)
		usage_loss = logits.new_zeros(())
	prototype_weight = _float_config(loss_config, 'prototype_weight', 1.0)
	distillation_weight = _float_config(loss_config, 'distillation_weight', 0.0)
	if distillation_weight > 0.0:
		if teacher_encoded is None:
			msg = 'teacher encoded tokens are required for feature distillation'
			raise ValueError(msg)
		distillation_loss = feature_distillation_loss(
			tokens,
			_encoded_tokens(teacher_encoded),
			valid_mask=distillation_valid_mask,
		)
	else:
		distillation_loss = logits.new_zeros(())
	total_loss = (
		prototype_weight * prototype_loss
		+ usage_weight * usage_loss
		+ distillation_weight * distillation_loss
	)
	return {
		'loss': total_loss,
		'loss_prototype': prototype_loss,
		'loss_usage': usage_loss,
		'loss_distillation': distillation_loss,
		'valid_supervised_token_fraction': valid_mask.float().mean(),
		'valid_distillation_token_fraction': distillation_valid_mask.float().mean(),
		'target_usage_entropy': _target_usage_entropy(
			labels,
			valid_mask,
			num_prototypes=logits.shape[-1],
		),
		'prototype_usage_entropy': _prototype_usage_entropy(probs, valid_mask),
	}


def _encoded_tokens(encoded: Mapping[str, object]) -> torch.Tensor:
	value = encoded.get('tokens')
	if not isinstance(value, torch.Tensor):
		msg = 'encoded output is missing tensor key "tokens"'
		raise TypeError(msg)
	return value


def _encoded_token_valid_mask(value: object, logits: torch.Tensor) -> torch.Tensor:
	if not isinstance(value, torch.Tensor):
		msg = 'encoded token_valid_mask must be a tensor or None'
		raise TypeError(msg)
	mask = value.bool()
	if tuple(mask.shape) != tuple(logits.shape[:-1]):
		msg = (
			'encoded token_valid_mask shape must match token logits prefix; '
			f'got {tuple(mask.shape)!r}, expected {tuple(logits.shape[:-1])!r}'
		)
		raise ValueError(msg)
	return mask


def _flatten_token_tensor(
	tensor: torch.Tensor,
	logits: torch.Tensor,
	name: str,
) -> torch.Tensor:
	if tensor.shape[0] != logits.shape[0]:
		msg = f'{name} batch dimension must match logits'
		raise ValueError(msg)
	return tensor.reshape(logits.shape[0], -1)


def _target_usage_entropy(
	labels: torch.Tensor,
	valid_mask: torch.Tensor,
	*,
	num_prototypes: int,
) -> torch.Tensor:
	selected = labels[valid_mask]
	if selected.numel() == 0:
		return labels.new_tensor(0.0, dtype=torch.float32)
	counts = torch.bincount(
		selected.clamp_min(0),
		minlength=num_prototypes,
	).to(dtype=torch.float32, device=labels.device)
	probs = counts / counts.sum().clamp_min(1.0)
	return -(probs * (probs + 1.0e-8).log()).sum()


def _prototype_usage_entropy(
	probs: torch.Tensor,
	valid_mask: torch.Tensor,
) -> torch.Tensor:
	selected = probs.reshape(-1, probs.shape[-1])[valid_mask.reshape(-1)]
	if selected.numel() == 0:
		return probs.new_zeros(())
	q_bar = selected.mean(dim=0)
	return -(q_bar * (q_bar + 1.0e-8).log()).sum()


def _clip_gradients(  # noqa: PLR0913
	student: torch.nn.Module,
	head: torch.nn.Module,
	grad_clip_norm: float,
	*,
	epoch: int,
	global_step: int,
	batch_index: int,
) -> None:
	parameters = [
		parameter
		for module in (student, head)
		for parameter in module.parameters()
		if parameter.requires_grad
	]
	grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
	if torch.isfinite(grad_norm.detach()).all():
		return
	msg = (
		'non-finite strat HMM pretext gradient norm at '
		f'epoch {epoch}, step {global_step}, batch {batch_index}'
	)
	raise FloatingPointError(msg)


def _trainability_metrics(summary: TrainabilitySummary) -> dict[str, float]:
	return {
		'trainable_parameter_count': float(summary.trainable_parameter_count),
		'frozen_parameter_count': float(summary.frozen_parameter_count),
	}


def _restore_strat_hmm_checkpoint(  # noqa: PLR0913
	*,
	payload: Mapping[str, object],
	student: torch.nn.Module,
	head: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	amp_enabled: bool,
	config: Mapping[str, object],
) -> StratHmmResumeState:
	_validate_strat_resume_payload(payload, amp_enabled=amp_enabled)
	_validate_strat_resume_config_compatibility(payload, config)
	try:
		student.load_state_dict(payload['model_state_dict'])
		head.load_state_dict(payload['stratigraphy_state_dict'])
	except RuntimeError as exc:
		msg = f'incompatible model/head geometry for resume checkpoint: {exc}'
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
			f"checkpoint={payload['amp_enabled']!r}, current={amp_enabled!r}"
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
			f"got {training_state['stage']!r}"
		)
		raise ValueError(msg)
	if training_state['checkpoint_kind'] not in {'step', 'epoch'}:
		msg = 'resume checkpoint checkpoint_kind must be "step" or "epoch"'
		raise ValueError(msg)
	batch_index = training_state['batch_index']
	if training_state['checkpoint_kind'] == 'step':
		if not isinstance(batch_index, int) or isinstance(batch_index, bool):
			msg = (
				'resume checkpoint batch_index must be an integer for '
				'step checkpoints'
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
	'StratHmmResumeState',
	'StratHmmTrainingState',
	'TrainabilitySummary',
	'build_strat_hmm_head_only_components',
	'configure_student_trainability',
	'run_strat_hmm_pretext_training',
	'train_strat_hmm_head_only_one_epoch',
]
