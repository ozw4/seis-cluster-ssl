"""Losses for supervised training of the frozen-embedding voxel decoder."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import torch
from torch.nn.functional import cross_entropy

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


def balanced_class_weights_from_counts(
	class_counts: Sequence[int],
	*,
	device: torch.device | str | None = None,
) -> torch.Tensor:
	"""Return ``N / (C * count_c)`` weights for common train counts."""
	counts = tuple(class_counts)
	if not counts:
		raise ValueError('class_counts must contain at least one class')
	if any(
		not isinstance(count, Integral) or isinstance(count, bool) or count < 0
		for count in counts
	):
		raise ValueError('class_counts must contain non-negative integers')
	zero_classes = [index for index, count in enumerate(counts) if count == 0]
	if zero_classes:
		raise ValueError(
			f'train class counts must be positive; zero-count classes: {zero_classes!r}'
		)
	values = torch.as_tensor(counts, dtype=torch.float32, device=device)
	return values.sum() / (len(counts) * values)


def masked_weighted_voxel_cross_entropy(
	logits: torch.Tensor,
	labels: torch.Tensor,
	mask: torch.Tensor,
	class_weights: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, float]]:
	"""Compute class-weight-normalized cross entropy on selected voxels."""
	_validate_inputs(logits, labels, mask, class_weights)
	selected_labels = labels[mask]
	if selected_labels.numel() == 0:
		raise ValueError('masked voxel cross entropy requires a positive denominator')
	class_count = logits.shape[1]
	invalid = (selected_labels < 0) | (selected_labels >= class_count)
	if invalid.any():
		unknown = torch.unique(selected_labels[invalid]).detach().cpu().tolist()
		raise ValueError(f'masked labels contain unknown classes: {unknown!r}')

	selected_logits = logits.movedim(1, -1)[mask].float()
	weights = class_weights.float()
	if not torch.isfinite(selected_logits).all():
		raise FloatingPointError('masked logits must be finite')
	per_voxel_ce = cross_entropy(
		selected_logits,
		selected_labels.to(dtype=torch.long),
		reduction='none',
	)
	selected_weights = weights[selected_labels.to(dtype=torch.long)]
	weight_sum = selected_weights.sum(dtype=torch.float32)
	if not torch.isfinite(per_voxel_ce).all() or not torch.isfinite(weight_sum):
		raise FloatingPointError('voxel cross entropy and denominator must be finite')
	if weight_sum.item() <= 0.0:
		raise ValueError('masked voxel cross entropy requires a positive denominator')
	weighted_ce = (
		(per_voxel_ce * selected_weights).sum(dtype=torch.float32) / weight_sum
	)
	unweighted_ce = per_voxel_ce.sum(dtype=torch.float32) / selected_labels.numel()
	if not torch.isfinite(weighted_ce) or not torch.isfinite(unweighted_ce):
		raise FloatingPointError('voxel cross entropy must be finite')

	summary = {
		'supervised_voxel_count': float(selected_labels.numel()),
		'class_weight_sum': float(weight_sum.detach().cpu().item()),
		'unweighted_cross_entropy': float(unweighted_ce.detach().cpu().item()),
		'weighted_cross_entropy': float(weighted_ce.detach().cpu().item()),
	}
	class_counts = torch.bincount(selected_labels, minlength=class_count)
	for class_index, count in enumerate(class_counts.detach().cpu().tolist()):
		summary[f'class_{class_index}_count'] = float(count)
	return weighted_ce, summary


def _validate_inputs(  # noqa: C901
	logits: torch.Tensor,
	labels: torch.Tensor,
	mask: torch.Tensor,
	class_weights: torch.Tensor,
) -> None:
	if logits.ndim != 5:
		raise ValueError('logits must have shape [B,C,X,Y,Z]')
	if not logits.is_floating_point():
		raise TypeError('logits must have a floating dtype')
	expected = (logits.shape[0], *logits.shape[2:])
	if labels.shape != expected or mask.shape != expected:
		raise ValueError(
			'labels and mask must match logits batch and spatial dimensions; '
			f'expected {expected!r}, got {tuple(labels.shape)!r} '
			f'and {tuple(mask.shape)!r}'
		)
	if labels.dtype not in (
		torch.int8,
		torch.int16,
		torch.int32,
		torch.int64,
		torch.uint8,
	):
		raise TypeError('labels must have an integer dtype')
	if mask.dtype != torch.bool:
		raise TypeError('mask must have dtype bool')
	if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
		raise ValueError('class_weights must have shape [C]')
	if not class_weights.is_floating_point():
		raise TypeError('class_weights must have a floating dtype')
	if logits.device != labels.device or logits.device != mask.device:
		raise ValueError('logits, labels, and mask must be on the same device')
	if logits.device != class_weights.device:
		raise ValueError('logits and class_weights must be on the same device')
	if not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
		raise ValueError('class_weights must contain finite positive values')


__all__ = [
	'balanced_class_weights_from_counts',
	'masked_weighted_voxel_cross_entropy',
]
