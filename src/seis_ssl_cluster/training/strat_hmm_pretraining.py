"""Head-only stratigraphic HMM pretext training."""

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
from typing import TYPE_CHECKING, Literal, cast

import torch

import seis_ssl_cluster
from seis_ssl_cluster.data import (
	NopimsStratPseudoTargetDataset,
	ZeroMaskConfig,
	read_manifest_json,
)
from seis_ssl_cluster.embedding.extractor import build_model_from_config
from seis_ssl_cluster.stratigraphy import (
	OrderedPrototypeHead,
	discover_pseudo_target_inputs,
	structured_hmm_prototype_loss,
	usage_entropy_floor_loss,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.dataloaders import build_strat_pseudo_target_dataloader
from seis_ssl_cluster.training.mae import prepare_run_directory
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


@dataclass(frozen=True)
class StratHmmTrainingState:
	"""Summary state returned from one strat HMM training epoch."""

	epoch: int
	global_step: int
	metrics: dict[str, float]
	last_batch_index: int
	completed_epoch: bool


@dataclass(frozen=True)
class StratHmmHeadOnlyComponents:
	"""Trainable components for prompt-07 strat HMM pretext training."""

	student: AmplitudeMAE3D
	head: OrderedPrototypeHead
	optimizer: torch.optim.Optimizer
	mae_checkpoint_config: Mapping[str, object]


def run_strat_hmm_pretext_training(  # noqa: C901, PLR0915
	config: Mapping[str, object],
	*,
	resume: str | Path | None = None,
) -> Path:
	"""Run frozen-encoder strat HMM pretext training from ``config``."""
	if resume is not None:
		msg = 'resume for strat HMM pretext training is not implemented yet'
		raise NotImplementedError(msg)
	_validate_head_only_scope(config)

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
		resume=None,
		allow_overwrite=_bool_config(
			train_config,
			'allow_overwrite_output',
			default=False,
		),
	)
	_snapshot_run_inputs(output_root=output_root, config=config)

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
	amp_enabled = (
		_bool_config(train_config, 'amp', default=False)
		and device.type == 'cuda'
		and torch.cuda.is_available()
	)
	scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
	epochs = _int_config(train_config, 'epochs', 1)
	max_steps = _optional_int_config(train_config, 'max_steps')
	checkpoint_every_steps = _optional_int_config(
		train_config,
		'checkpoint_every_steps',
	)
	grad_clip_norm = _optional_float_config(train_config, 'grad_clip_norm')

	state = StratHmmTrainingState(
		epoch=0,
		global_step=0,
		metrics={},
		last_batch_index=-1,
		completed_epoch=True,
	)
	checkpoint_path: Path | None = None
	best_score: float | None = None
	for epoch in range(1, epochs + 1):
		set_epoch = getattr(dataset, 'set_epoch', None)
		if callable(set_epoch):
			set_epoch(epoch - 1)
		remaining_steps = None
		if max_steps is not None:
			remaining_steps = max_steps - state.global_step
			if remaining_steps <= 0:
				break

		def save_step_checkpoint(
			step_state: StratHmmTrainingState,
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
				metrics=step_state.metrics,
				global_step=step_state.global_step,
				checkpoint_kind='step',
				batch_index=step_state.last_batch_index,
				amp_enabled=amp_enabled,
				scaler=scaler,
				best_score=best_score,
			)
			best_score = result.best_score
			checkpoint_path = result.latest_path

		state = train_strat_hmm_head_only_one_epoch(
			student=components.student,
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
			step_callback=save_step_checkpoint,
		)
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
			metrics={**state.metrics, 'amp_enabled': float(amp_enabled)},
			global_step=state.global_step,
			checkpoint_kind=checkpoint_kind,
			batch_index=None if state.completed_epoch else state.last_batch_index,
			amp_enabled=amp_enabled,
			scaler=scaler,
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
	"""Build frozen student MAE, ordered prototype head, and optimizer."""
	_validate_head_only_scope(config)
	resolved_device = torch.device(device)
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
	for parameter in student.parameters():
		parameter.requires_grad_(requires_grad=False)
	student.to(resolved_device)
	student.eval()

	head_config = _mapping(config, 'head')
	head = OrderedPrototypeHead(
		feature_dim=student.encoder_dim,
		num_prototypes=_int_config(head_config, 'num_prototypes', 1),
		projection_dim=_optional_int_config(head_config, 'projection_dim'),
		temperature=_float_config(head_config, 'temperature', 0.1),
		normalize=_bool_config(head_config, 'normalize', default=True),
	).to(resolved_device)
	trainable_parameters = [
		parameter for parameter in head.parameters() if parameter.requires_grad
	]
	if not trainable_parameters:
		msg = 'ordered prototype head has no trainable parameters'
		raise ValueError(msg)
	train_config = _mapping(config, 'train')
	optimizer = torch.optim.AdamW(
		trainable_parameters,
		lr=_float_config(train_config, 'lr', 3.0e-4),
		weight_decay=_float_config(train_config, 'weight_decay', 0.05),
	)
	return StratHmmHeadOnlyComponents(
		student=student,
		head=head,
		optimizer=optimizer,
		mae_checkpoint_config=_extraction_compatible_config(
			teacher_config,
			output_root=_path_config(_mapping(config, 'paths'), 'output_root'),
		),
	)


def train_strat_hmm_head_only_one_epoch(  # noqa: C901, PLR0913
	*,
	student: AmplitudeMAE3D,
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
		if max_steps is not None and batches >= max_steps:
			break
		batch = move_batch_to_device(raw_batch, device)
		optimizer.zero_grad(set_to_none=True)

		with torch.no_grad():
			encoded = student.encode_tokens(
				_required_tensor(batch, 'x'),
				valid_mask=_required_tensor(batch, 'local_valid_mask'),
			)
		with torch.amp.autocast('cuda', enabled=amp_enabled):
			losses = _strat_head_losses(
				head=head,
				encoded=encoded,
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


def _strat_head_losses(
	*,
	head: OrderedPrototypeHead,
	encoded: Mapping[str, object],
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
	token_valid_mask = encoded.get('token_valid_mask')
	if token_valid_mask is not None:
		if not isinstance(token_valid_mask, torch.Tensor):
			msg = 'encoded token_valid_mask must be a tensor or None'
			raise TypeError(msg)
		valid_mask = valid_mask & token_valid_mask.bool()
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
	total_loss = prototype_weight * prototype_loss + usage_weight * usage_loss
	return {
		'loss': total_loss,
		'loss_prototype': prototype_loss,
		'loss_usage': usage_loss,
		'valid_supervised_token_fraction': valid_mask.float().mean(),
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


def _clip_gradients(
	head: torch.nn.Module,
	grad_clip_norm: float,
	*,
	epoch: int,
	global_step: int,
	batch_index: int,
) -> None:
	grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
	if torch.isfinite(grad_norm.detach()).all():
		return
	msg = (
		'non-finite strat HMM pretext gradient norm at '
		f'epoch {epoch}, step {global_step}, batch {batch_index}'
	)
	raise FloatingPointError(msg)


def _validate_head_only_scope(config: Mapping[str, object]) -> None:
	student = _mapping(config, 'student')
	loss = _mapping(config, 'loss')
	if _int_config(student, 'unfreeze_top_blocks', 0) != 0:
		msg = 'only student.unfreeze_top_blocks == 0 is implemented in this prompt'
		raise NotImplementedError(msg)
	if _float_config(loss, 'distillation_weight', 0.0) != 0.0:
		msg = 'only loss.distillation_weight == 0 is implemented in this prompt'
		raise NotImplementedError(msg)


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
) -> Mapping[str, object]:
	result = deepcopy(dict(teacher_config))
	paths = dict(_mapping(result, 'paths'))
	paths['output_root'] = str(output_root)
	result['paths'] = paths
	return result


def _zero_mask_from_config(config: Mapping[str, object]) -> ZeroMaskConfig:
	value = _mapping(config, 'zero_mask')
	zero_mask = ZeroMaskConfig(
		enabled=_bool_config(value, 'enabled', default=True),
		zero_atol=_float_config(value, 'zero_atol', 0.0),
		z_sample_influence_radius=_int_config(
			value,
			'z_sample_influence_radius',
			16,
		),
		xy_trace_influence_radius=_int_config(
			value,
			'xy_trace_influence_radius',
			1,
		),
	)
	zero_mask.validate()
	return zero_mask


def _snapshot_run_inputs(
	*,
	output_root: Path,
	config: Mapping[str, object],
) -> None:
	_write_json(
		output_root / 'resolved_config.json',
		_to_json_safe(config),
	)
	_write_json(
		output_root / 'run_metadata.json',
		{
			'created_at_utc': datetime.now(timezone.utc).isoformat(),
			'git_commit': _git_commit(),
			'package_version': getattr(seis_ssl_cluster, '__version__', None),
		},
	)


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
	path.write_text(f'{text}\n', encoding='utf-8')


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


def _required_tensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
	value = batch.get(key)
	if not isinstance(value, torch.Tensor):
		msg = f'batch key {key!r} must be a tensor'
		raise TypeError(msg)
	return value


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _non_empty_string(value: object, name: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{name} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _path_config(config: Mapping[str, object], key: str) -> Path:
	return Path(_non_empty_string(config.get(key), key))


def _int_config(config: Mapping[str, object], key: str, default: int) -> int:
	value = config.get(key, default)
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return int(value)


def _optional_int_config(config: Mapping[str, object], key: str) -> int | None:
	value = config.get(key)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{key} must be an integer or None; got {value!r}'
		raise TypeError(msg)
	return int(value)


def _float_config(config: Mapping[str, object], key: str, default: float) -> float:
	value = config.get(key, default)
	if isinstance(value, bool) or not isinstance(value, int | float):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	result = float(value)
	if not math.isfinite(result):
		msg = f'{key} must be finite; got {value!r}'
		raise ValueError(msg)
	return result


def _optional_float_config(
	config: Mapping[str, object],
	key: str,
) -> float | None:
	value = config.get(key)
	if value is None:
		return None
	return _float_config(config, key, 0.0)


def _bool_config(
	config: Mapping[str, object],
	key: str,
	*,
	default: bool,
) -> bool:
	value = config.get(key, default)
	if not isinstance(value, bool):
		msg = f'{key} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _xyz_config(
	config: Mapping[str, object],
	key: str,
) -> tuple[int, int, int]:
	value = config.get(key)
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in value)
	):
		msg = f'{key} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(int(axis) for axis in value)
	if any(axis <= 0 for axis in xyz):
		msg = f'{key} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return cast('tuple[int, int, int]', xyz)


def _to_json_safe(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _to_json_safe(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_to_json_safe(child) for child in value]
	if isinstance(value, Path):
		return str(value)
	return value


__all__ = [
	'StratHmmHeadOnlyComponents',
	'StratHmmTrainingState',
	'build_strat_hmm_head_only_components',
	'run_strat_hmm_pretext_training',
	'train_strat_hmm_head_only_one_epoch',
]
