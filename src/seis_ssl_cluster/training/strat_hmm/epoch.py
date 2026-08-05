"""One-epoch loop for stratigraphic HMM pretext training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch

from seis_ssl_cluster.models.mae.patching import compute_num_patches, patchify_3d
from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.strat_hmm.losses import (
	compute_strat_hmm_center_trace_masked_losses,
	compute_strat_hmm_multi_head_losses,
	compute_strat_hmm_multi_head_posterior_losses,
	compute_strat_hmm_pretext_losses,
)
from seis_ssl_cluster.training.strat_hmm.masking import (
	COMMON_HARD_TARGET_HEAD_KS,
	plan_xy_token_column_mask,
	validate_common_hard_target_valid_masks,
)
from seis_ssl_cluster.training.strat_hmm.runtime import (
	_float_config,
	_required_tensor,
)
from seis_ssl_cluster.training.strat_hmm.state import StratHmmTrainingState

if TYPE_CHECKING:
	from collections.abc import Callable

	from seis_ssl_cluster.models.mae import (
		AmplitudeMAE3D,
		LearnedEncoderReplacementToken,
	)
	from seis_ssl_cluster.stratigraphy import (
		MultiResolutionOrderedPrototypeHeads,
		OrderedPrototypeHead,
	)


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
	step_callback: (Callable[[StratHmmTrainingState], None] | None) = None,
) -> StratHmmTrainingState:
	"""Train the ordered prototype head for one epoch."""
	_set_student_training_mode(student)
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
			losses = compute_strat_hmm_pretext_losses(
				head=head,
				encoded=encoded,
				teacher_encoded=teacher_encoded,
				batch=batch,
				loss_config=loss_config,
				pseudo_target_config=pseudo_target_config,
			)
			loss = losses['loss']

		_ensure_finite_losses(
			losses,
			epoch=epoch,
			global_step=global_step,
			batch_index=batch_index,
		)
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
			scaler.unscale_(optimizer)
			_ensure_finite_gradients(
				(student, head),
				epoch=epoch,
				global_step=global_step,
				batch_index=batch_index,
			)
			if grad_clip_norm is not None:
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
			_ensure_finite_gradients(
				(student, head),
				epoch=epoch,
				global_step=global_step,
				batch_index=batch_index,
			)
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
			key: float(value.detach().cpu().item()) for key, value in losses.items()
		}
		if batches > 0 and set(step_metrics) != set(totals):
			raise ValueError('strat HMM epoch metric keys changed between batches')
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


def train_strat_hmm_multi_head_one_epoch(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	student: AmplitudeMAE3D,
	teacher: AmplitudeMAE3D | None = None,
	heads: MultiResolutionOrderedPrototypeHeads,
	dataloader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	loss_config: Mapping[str, object],
	pseudo_target_config: Mapping[str, object],
	target_representation: str = 'hard_viterbi_labels_v1',
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	global_step: int = 0,
	max_steps: int | None = None,
	grad_clip_norm: float | None = None,
	skip_batches: int = 0,
	step_callback: Callable[[StratHmmTrainingState], None] | None = None,
) -> StratHmmTrainingState:
	"""Train all ordered-prototype resolutions from one shared token encoding."""
	_set_student_training_mode(student)
	heads.train()
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
				raise ValueError(
					'teacher is required when loss.distillation_weight is positive',
				)
			teacher.eval()
			with torch.no_grad():
				teacher_encoded = teacher.encode_tokens(
					_required_tensor(batch, 'x'),
					valid_mask=_required_tensor(batch, 'local_valid_mask'),
				)
		with torch.amp.autocast('cuda', enabled=amp_enabled):
			if target_representation == 'ordered_path_state_posterior_v1':
				losses = compute_strat_hmm_multi_head_posterior_losses(
					heads=heads,
					encoded=encoded,
					teacher_encoded=teacher_encoded,
					batch=batch,
					loss_config=loss_config,
				)
			elif target_representation in {
				'hard_viterbi_labels_v1',
				'lateral_mean_field_hard_labels_v1',
				'xy_neighbor_consensus_hard_labels_v1',
				'xy_neighbor_unanimous_hard_labels_v1',
			}:
				losses = compute_strat_hmm_multi_head_losses(
					heads=heads,
					encoded=encoded,
					teacher_encoded=teacher_encoded,
					batch=batch,
					loss_config=loss_config,
					pseudo_target_config=pseudo_target_config,
				)
			else:
				raise ValueError(
					'unsupported multi-head target representation: '
					f'{target_representation!r}'
				)
			if target_representation == 'xy_neighbor_unanimous_hard_labels_v1':
				# ``loss_consistency`` remains the diagnostic raw pair loss used by
				# every hard multi-head route. Schema-6 additionally records its
				# actual weighted contribution, which the fixed unanimous policy
				# requires to be exactly zero.
				consistency_weight = _float_config(
					loss_config,
					'consistency_weight',
					0.0,
				)
				if consistency_weight != 0.0:
					raise ValueError(
						'unanimous hard-label training requires zero consistency weight'
					)
				losses['loss_consistency_contribution'] = (
					losses['loss_consistency'] * consistency_weight
				)
			loss = losses['loss']
		_ensure_finite_losses(
			losses,
			epoch=epoch,
			global_step=global_step,
			batch_index=batch_index,
		)
		if amp_enabled:
			if scaler is None:
				raise ValueError('scaler is required when amp_enabled is true')
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			_ensure_finite_gradients(
				(student, heads),
				epoch=epoch,
				global_step=global_step,
				batch_index=batch_index,
			)
			if grad_clip_norm is not None:
				_clip_gradients(
					student,
					heads,
					grad_clip_norm,
					epoch=epoch,
					global_step=global_step,
					batch_index=batch_index,
				)
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			_ensure_finite_gradients(
				(student, heads),
				epoch=epoch,
				global_step=global_step,
				batch_index=batch_index,
			)
			if grad_clip_norm is not None:
				_clip_gradients(
					student,
					heads,
					grad_clip_norm,
					epoch=epoch,
					global_step=global_step,
					batch_index=batch_index,
				)
			optimizer.step()
		step_metrics = {
			key: float(value.detach().cpu().item()) for key, value in losses.items()
		}
		if batches > 0 and set(step_metrics) != set(totals):
			raise ValueError('strat HMM epoch metric keys changed between batches')
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
				)
			)
	if batches == 0:
		raise ValueError('dataloader produced no batches')
	return StratHmmTrainingState(
		epoch=epoch,
		global_step=global_step,
		metrics={key: total / batches for key, total in totals.items()},
		last_batch_index=last_batch_index,
		completed_epoch=last_batch_index >= len(dataloader) - 1,
	)


def train_strat_hmm_center_trace_masked_one_epoch(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	student: AmplitudeMAE3D,
	teacher: AmplitudeMAE3D | None = None,
	heads: MultiResolutionOrderedPrototypeHeads,
	replacement_token: LearnedEncoderReplacementToken,
	dataloader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	loss_config: Mapping[str, object],
	pseudo_target_config: Mapping[str, object],
	training_seed: int = 42,
	column_fraction: float = 0.10,
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	global_step: int = 0,
	max_steps: int | None = None,
	grad_clip_norm: float | None = None,
	skip_batches: int = 0,
	step_callback: Callable[[StratHmmTrainingState], None] | None = None,
	initial_epoch_metric_totals: Mapping[str, float] | None = None,
	initial_epoch_batch_count: int = 0,
) -> StratHmmTrainingState:
	"""Train the center-trace masked hard multi-head objective for one epoch."""
	if heads.head_ks != COMMON_HARD_TARGET_HEAD_KS:
		raise ValueError(
			'center-trace masked training requires heads K=(6, 8, 10); '
			f'got {heads.head_ks!r}'
		)
	_validate_center_trace_optimizer(optimizer, replacement_token)
	_set_student_training_mode(student)
	heads.train()
	replacement_token.train()
	if initial_epoch_batch_count < 0:
		raise ValueError('initial epoch batch count must be nonnegative')
	if initial_epoch_batch_count and initial_epoch_metric_totals is None:
		raise ValueError('initial epoch metric totals are required for resumed batches')
	if initial_epoch_metric_totals is not None and not initial_epoch_batch_count:
		raise ValueError('initial epoch metric totals require resumed batches')
	totals = {
		str(key): float(value)
		for key, value in (initial_epoch_metric_totals or {}).items()
	}
	if not all(torch.isfinite(torch.tensor(value)) for value in totals.values()):
		raise ValueError('initial epoch metric totals must be finite')
	batches = initial_epoch_batch_count
	last_batch_index = -1
	for batch_index, raw_batch in enumerate(dataloader):
		if batch_index < skip_batches:
			continue
		if max_steps is not None and batches >= max_steps:
			break
		batch = move_batch_to_device(raw_batch, device)
		x = _required_tensor(batch, 'x')
		local_valid_mask = _required_tensor(batch, 'local_valid_mask')
		common_target_valid_mask = _center_trace_common_target_valid_mask(batch)
		student_token_valid_mask = _center_trace_student_token_valid_mask(
			student,
			x,
			local_valid_mask,
			common_target_valid_mask,
		)
		plan = plan_xy_token_column_mask(
			common_target_valid_mask,
			student_token_valid_mask,
			column_fraction=column_fraction,
			training_seed=training_seed,
			epoch=epoch,
			global_step=global_step,
			batch_index=batch_index,
			sample_indices=_center_trace_sample_indices(batch),
		)

		optimizer.zero_grad(set_to_none=True)
		student_grad_enabled = any(
			parameter.requires_grad
			for module in (student, replacement_token)
			for parameter in module.parameters()
		)
		with torch.set_grad_enabled(student_grad_enabled):
			encoded = student.encode_tokens(
				x,
				valid_mask=local_valid_mask,
				replacement_mask=plan.mask,
				replacement_token=replacement_token(),
			)
		teacher_encoded = None
		if teacher is not None:
			teacher.eval()
			with torch.no_grad():
				teacher_encoded = teacher.encode_tokens(
					x,
					valid_mask=local_valid_mask,
				)
		elif _float_config(loss_config, 'distillation_weight', 0.0) > 0.0:
			raise ValueError(
				'teacher is required when loss.distillation_weight is positive'
			)
		with torch.amp.autocast('cuda', enabled=amp_enabled):
			losses = compute_strat_hmm_center_trace_masked_losses(
				heads=heads,
				encoded=encoded,
				teacher_encoded=teacher_encoded,
				batch=batch,
				replacement_mask=plan.mask,
				loss_config=loss_config,
				pseudo_target_config=pseudo_target_config,
			)
			losses['eligible_xy_column_count'] = plan.eligible_counts.to(
				dtype=_required_tensor(encoded, 'tokens').dtype
			).mean()
			losses['selected_xy_column_count'] = plan.selected_counts.to(
				dtype=_required_tensor(encoded, 'tokens').dtype
			).mean()
			loss = losses['loss']
		_ensure_finite_losses(
			losses,
			epoch=epoch,
			global_step=global_step,
			batch_index=batch_index,
		)
		if amp_enabled:
			if scaler is None:
				raise ValueError('scaler is required when amp_enabled is true')
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			_ensure_finite_gradients(
				(student, heads, replacement_token),
				epoch=epoch,
				global_step=global_step,
				batch_index=batch_index,
			)
			if grad_clip_norm is not None:
				_clip_gradients_for_modules(
					(student, heads, replacement_token),
					grad_clip_norm,
					epoch=epoch,
					global_step=global_step,
					batch_index=batch_index,
				)
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			_ensure_finite_gradients(
				(student, heads, replacement_token),
				epoch=epoch,
				global_step=global_step,
				batch_index=batch_index,
			)
			if grad_clip_norm is not None:
				_clip_gradients_for_modules(
					(student, heads, replacement_token),
					grad_clip_norm,
					epoch=epoch,
					global_step=global_step,
					batch_index=batch_index,
				)
			optimizer.step()

		step_metrics = {
			key: float(value.detach().cpu().item()) for key, value in losses.items()
		}
		if batches > 0 and set(step_metrics) != set(totals):
			raise ValueError('center-trace epoch metric keys changed between batches')
		for key, value in step_metrics.items():
			totals[key] = totals.get(key, 0.0) + value
		batches += 1
		global_step += 1
		last_batch_index = batch_index
		if step_callback is not None:
			completed_epoch = batch_index >= len(dataloader) - 1
			step_callback(
				StratHmmTrainingState(
					epoch=epoch,
					global_step=global_step,
					metrics=step_metrics,
					last_batch_index=batch_index,
					completed_epoch=completed_epoch,
					epoch_metrics=(
						{key: total / batches for key, total in totals.items()}
						if completed_epoch
						else None
					),
					epoch_metric_totals=dict(totals),
					epoch_batch_count=batches,
				)
			)
	if batches == 0:
		raise ValueError('dataloader produced no batches')
	return StratHmmTrainingState(
		epoch=epoch,
		global_step=global_step,
		metrics={key: total / batches for key, total in totals.items()},
		last_batch_index=last_batch_index,
		completed_epoch=last_batch_index >= len(dataloader) - 1,
		epoch_metric_totals=dict(totals),
		epoch_batch_count=batches,
	)


def _center_trace_common_target_valid_mask(
	batch: Mapping[str, object],
) -> torch.Tensor:
	targets = batch.get('strat_multi_targets')
	if not isinstance(targets, Mapping):
		raise TypeError('strat_multi_targets must be a mapping')
	if set(targets) != {f'k{k}' for k in COMMON_HARD_TARGET_HEAD_KS}:
		raise ValueError(
			'center-trace masked training requires target heads k6, k8, and k10'
		)
	valid_masks: dict[int, torch.Tensor] = {}
	for k in COMMON_HARD_TARGET_HEAD_KS:
		target = targets[f'k{k}']
		if not isinstance(target, Mapping):
			raise TypeError(f'strat_multi_targets["k{k}"] must be a mapping')
		valid_masks[k] = _required_tensor(
			cast('Mapping[str, object]', target),
			'valid_mask',
		)
	return validate_common_hard_target_valid_masks(valid_masks)


def _center_trace_student_token_valid_mask(
	student: torch.nn.Module,
	x: torch.Tensor,
	local_valid_mask: torch.Tensor,
	common_target_valid_mask: torch.Tensor,
) -> torch.Tensor:
	batch_size = int(x.shape[0])
	if local_valid_mask.ndim == 4 and tuple(local_valid_mask.shape) == tuple(
		common_target_valid_mask.shape
	):
		return local_valid_mask
	if local_valid_mask.ndim == 2 and tuple(local_valid_mask.shape) == (
		batch_size,
		int(common_target_valid_mask[0].numel()),
	):
		return local_valid_mask.reshape(common_target_valid_mask.shape)
	patch_size = getattr(student, 'patch_size_xyz', None)
	if not isinstance(patch_size, tuple) or len(patch_size) != 3:
		raise ValueError(
			'center-trace masking requires local_valid_mask in token-grid shape or '
			'student.patch_size_xyz'
		)
	if local_valid_mask.ndim != 4 or local_valid_mask.shape[0] != batch_size:
		raise ValueError(
			'local_valid_mask must have shape [B, X, Y, Z] or token-grid shape'
		)
	grid_shape = compute_num_patches(
		(int(x.shape[2]), int(x.shape[3]), int(x.shape[4])),
		patch_size,
	)[:3]
	if tuple(grid_shape) != tuple(common_target_valid_mask.shape[1:]):
		raise ValueError(
			'common hard-target valid mask shape must match the student token '
			'grid; '
			f'got {tuple(common_target_valid_mask.shape[1:])!r}, '
			f'expected {grid_shape!r}'
		)
	mask_patches = patchify_3d(
		local_valid_mask.unsqueeze(1),
		patch_size,
	)
	return mask_patches.squeeze(2).all(dim=-1).reshape(common_target_valid_mask.shape)


def _center_trace_sample_indices(batch: Mapping[str, object]) -> torch.Tensor | None:
	value = batch.get('sample_indices')
	if value is not None:
		if not isinstance(value, torch.Tensor):
			raise TypeError('sample_indices must be a tensor when provided')
		return value
	coords = batch.get('coords')
	if isinstance(coords, list) and all(isinstance(item, Mapping) for item in coords):
		indices = [item.get('sample_index') for item in coords]
		if all(
			isinstance(item, int) and not isinstance(item, bool) for item in indices
		):
			return torch.tensor(indices, dtype=torch.int64)
	return None


def _validate_center_trace_optimizer(
	optimizer: torch.optim.Optimizer,
	replacement_token: torch.nn.Module,
) -> None:
	required_ids = {
		id(parameter)
		for parameter in replacement_token.parameters()
		if parameter.requires_grad
	}
	optimizer_ids = [
		id(parameter)
		for group in optimizer.param_groups
		for parameter in group['params']
	]
	if len(set(optimizer_ids)) != len(optimizer_ids):
		raise ValueError('center-trace optimizer contains duplicate parameters')
	if not required_ids <= set(optimizer_ids):
		raise ValueError(
			'center-trace optimizer must include all replacement-token parameters'
		)


def _set_student_training_mode(student: torch.nn.Module) -> None:
	"""Keep the frozen MAE in evaluation mode while training top blocks only."""
	student.eval()
	encoder = getattr(student, 'encoder', None)
	layers = getattr(encoder, 'layers', ())
	for layer in layers:
		if any(parameter.requires_grad for parameter in layer.parameters()):
			layer.train()


def _ensure_finite_losses(
	losses: Mapping[str, torch.Tensor],
	*,
	epoch: int,
	global_step: int,
	batch_index: int,
) -> None:
	if all(torch.isfinite(value).all() for value in losses.values()):
		return
	raise FloatingPointError(
		'non-finite strat HMM pretext loss at '
		f'epoch {epoch}, step {global_step}, batch {batch_index}',
	)


def _ensure_finite_gradients(
	modules: tuple[torch.nn.Module, ...],
	*,
	epoch: int,
	global_step: int,
	batch_index: int,
) -> None:
	gradients = [
		parameter.grad
		for module in modules
		for parameter in module.parameters()
		if parameter.requires_grad and parameter.grad is not None
	]
	if all(torch.isfinite(gradient).all() for gradient in gradients):
		return
	raise FloatingPointError(
		'non-finite strat HMM pretext gradient at '
		f'epoch {epoch}, step {global_step}, batch {batch_index}',
	)


def _clip_gradients(  # noqa: PLR0913
	student: torch.nn.Module,
	head: torch.nn.Module,
	grad_clip_norm: float,
	*,
	epoch: int,
	global_step: int,
	batch_index: int,
) -> None:
	_clip_gradients_for_modules(
		(student, head),
		grad_clip_norm,
		epoch=epoch,
		global_step=global_step,
		batch_index=batch_index,
	)


def _clip_gradients_for_modules(
	modules: tuple[torch.nn.Module, ...],
	grad_clip_norm: float,
	*,
	epoch: int,
	global_step: int,
	batch_index: int,
) -> None:
	parameters = [
		parameter
		for module in modules
		for parameter in module.parameters()
		if parameter.requires_grad
	]
	grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
	gradients = [
		parameter.grad.detach()
		for parameter in parameters
		if parameter.grad is not None
	]
	if not torch.isfinite(grad_norm.detach()).all():
		msg = (
			'non-finite strat HMM pretext gradient norm at '
			f'epoch {epoch}, step {global_step}, batch {batch_index}'
		)
		raise FloatingPointError(msg)
	if not gradients:
		return
	post_clip_grad_norm = torch.linalg.vector_norm(
		torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients]),
	)
	if torch.isfinite(post_clip_grad_norm).all():
		return
	msg = (
		'non-finite strat HMM pretext gradient norm at '
		f'epoch {epoch}, step {global_step}, batch {batch_index}'
	)
	raise FloatingPointError(msg)


__all__ = [
	'train_strat_hmm_center_trace_masked_one_epoch',
	'train_strat_hmm_head_only_one_epoch',
	'train_strat_hmm_multi_head_one_epoch',
]
