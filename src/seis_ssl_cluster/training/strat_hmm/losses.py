"""Batch loss orchestration for stratigraphic HMM pretext training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from seis_ssl_cluster.stratigraphy import (
	OrderedPrototypeHead,
	feature_distillation_loss,
	structured_hmm_prototype_loss,
	usage_entropy_floor_loss,
)
from seis_ssl_cluster.training.strat_hmm.runtime import (
	_float_config,
	_required_tensor,
)

if TYPE_CHECKING:
	from collections.abc import Mapping


def compute_strat_hmm_pretext_losses(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	head: OrderedPrototypeHead,
	encoded: Mapping[str, object],
	teacher_encoded: Mapping[str, object] | None,
	batch: Mapping[str, object],
	loss_config: Mapping[str, object],
	pseudo_target_config: Mapping[str, object],
) -> dict[str, torch.Tensor]:
	"""Compute training-batch losses and metrics for strat HMM pretext learning."""
	tokens = _encoded_tokens(encoded)
	prototype_weight = _float_config(loss_config, 'prototype_weight', 1.0)
	usage_weight = _float_config(loss_config, 'usage_weight', 0.0)
	logits = (
		head(tokens).logits
		if prototype_weight > 0.0 or usage_weight > 0.0
		else None
	)
	reference = logits if logits is not None else tokens
	labels = _flatten_token_tensor(
		_required_tensor(batch, 'strat_labels'),
		reference,
		'strat_labels',
	).long()
	confidence = _flatten_token_tensor(
		_required_tensor(batch, 'strat_confidence'),
		reference,
		'strat_confidence',
	).to(dtype=reference.dtype)
	valid_mask = _flatten_token_tensor(
		_required_tensor(batch, 'strat_valid_mask'),
		reference,
		'strat_valid_mask',
	).bool()
	distillation_valid_mask = valid_mask
	token_valid_mask = encoded.get('token_valid_mask')
	if token_valid_mask is not None:
		student_valid_mask = _encoded_token_valid_mask(token_valid_mask, reference)
		valid_mask = valid_mask & student_valid_mask
		distillation_valid_mask = distillation_valid_mask & student_valid_mask
	if teacher_encoded is not None:
		teacher_token_valid_mask = teacher_encoded.get('token_valid_mask')
		if teacher_token_valid_mask is not None:
			teacher_valid_mask = _encoded_token_valid_mask(
				teacher_token_valid_mask,
				reference,
			)
			distillation_valid_mask = distillation_valid_mask & teacher_valid_mask
	min_confidence = _float_config(pseudo_target_config, 'min_confidence', 0.0)
	if min_confidence > 0.0:
		valid_mask = valid_mask & confidence.ge(min_confidence)

	if prototype_weight > 0.0:
		if logits is None:  # pragma: no cover - guarded by logits construction
			raise AssertionError('prototype logits were not computed')
		prototype_loss = structured_hmm_prototype_loss(
			logits,
			labels,
			valid_mask=valid_mask,
			confidence=confidence,
		)
	else:
		prototype_loss = tokens.new_zeros(())
	if usage_weight > 0.0:
		if logits is None:  # pragma: no cover - guarded by logits construction
			raise AssertionError('prototype logits were not computed')
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
		probs = (
			torch.nn.functional.softmax(logits, dim=-1)
			if logits is not None
			else None
		)
		usage_loss = tokens.new_zeros(())
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
		distillation_loss = tokens.new_zeros(())
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
			num_prototypes=head.num_prototypes,
		),
		'prototype_usage_entropy': (
			_prototype_usage_entropy(probs, valid_mask)
			if probs is not None
			else tokens.new_zeros(())
		),
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
	flattened = tensor.reshape(logits.shape[0], -1)
	if flattened.shape[1] != logits.shape[1]:
		msg = (
			f'{name} flattened token count must match logits token dimension; '
			f'got {flattened.shape[1]}, expected {logits.shape[1]}'
		)
		raise ValueError(msg)
	return flattened


def _target_usage_entropy(
	labels: torch.Tensor,
	valid_mask: torch.Tensor,
	*,
	num_prototypes: int,
) -> torch.Tensor:
	selected = labels[
		valid_mask & labels.ge(0) & labels.lt(num_prototypes)
	]
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


_strat_head_losses = compute_strat_hmm_pretext_losses


__all__ = [
	'compute_strat_hmm_pretext_losses',
]
