'''Shared optimization lifecycle for Volve horizon training.'''

from __future__ import annotations

import torch


def backward_and_step_horizon_optimizer(
	*,
	loss: torch.Tensor,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	gradient_clip_norm: float,
) -> torch.Tensor:
	'''Backpropagate, reject non-finite gradients, then update parameters.'''
	if scaler is None:
		loss.backward()
	else:
		scaler.scale(loss).backward()
		scaler.unscale_(optimizer)
	gradient_norm = torch.nn.utils.clip_grad_norm_(
		model.parameters(), gradient_clip_norm
	)
	if not bool(torch.isfinite(gradient_norm.detach()).all()):
		raise FloatingPointError('non-finite Volve horizon gradient norm')
	if scaler is None:
		optimizer.step()
	else:
		scaler.step(optimizer)
		scaler.update()
	return gradient_norm.detach()
