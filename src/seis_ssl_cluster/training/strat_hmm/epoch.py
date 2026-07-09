"""One-epoch loop for stratigraphic HMM pretext training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.strat_hmm.losses import (
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
	from seis_ssl_cluster.stratigraphy import OrderedPrototypeHead


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
			losses = compute_strat_hmm_pretext_losses(
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


__all__ = [
	'train_strat_hmm_head_only_one_epoch',
]
