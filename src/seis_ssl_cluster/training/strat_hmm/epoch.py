"""One-epoch loop for stratigraphic HMM pretext training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.strat_hmm.losses import (
	compute_strat_hmm_multi_head_losses,
	compute_strat_hmm_multi_head_posterior_losses,
	compute_strat_hmm_pretext_losses,
)
from seis_ssl_cluster.training.strat_hmm.runtime import (
	_float_config,
	_required_tensor,
)
from seis_ssl_cluster.training.strat_hmm.state import StratHmmTrainingState

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping

	from seis_ssl_cluster.models.mae import AmplitudeMAE3D
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
	parameters = [
		parameter
		for module in (student, head)
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
	'train_strat_hmm_head_only_one_epoch',
	'train_strat_hmm_multi_head_one_epoch',
]
