'''Fractional-sample supervision for the shared Volve horizon decoder.'''

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn.functional import log_softmax

from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_tiles import (
	HORIZON_WINDOW_START,
	HORIZON_WINDOW_STOP,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


def fractional_horizon_cross_entropy(
	logits: torch.Tensor,
	target_sample_float: torch.Tensor,
	supervision_mask: torch.Tensor,
	*,
	window_start: int = HORIZON_WINDOW_START,
) -> tuple[torch.Tensor, Mapping[str, object]]:
	'''Macro-average two-bin cross-entropy over active horizons in a tile batch.'''
	_validate_loss_inputs(logits, target_sample_float, supervision_mask)
	window_stop = window_start + logits.shape[-1]
	(
		valid_target,
		floor_index,
		ceil_index,
		floor_weight,
		ceil_weight,
	) = fractional_target_weights(
		target_sample_float,
		window_start=window_start,
		window_stop=window_stop,
	)
	selected = supervision_mask & valid_target
	log_probabilities = log_softmax(logits.float(), dim=-1)
	floor_log_probability = torch.gather(
		log_probabilities, -1, floor_index.unsqueeze(-1)
	).squeeze(-1)
	ceil_log_probability = torch.gather(
		log_probabilities, -1, ceil_index.unsqueeze(-1)
	).squeeze(-1)
	point_loss = -(
		floor_weight.float() * floor_log_probability
		+ ceil_weight.float() * ceil_log_probability
	)
	per_horizon_losses: list[torch.Tensor] = []
	per_horizon_counts: list[int] = []
	per_horizon_summary: dict[str, dict[str, float | int | None]] = {}
	for horizon_index, horizon_name in enumerate(HORIZON_NAMES):
		horizon_mask = selected[:, horizon_index]
		count = int(torch.count_nonzero(horizon_mask).detach().cpu().item())
		per_horizon_counts.append(count)
		if count:
			value = point_loss[:, horizon_index][horizon_mask].mean(dtype=torch.float32)
			per_horizon_losses.append(value)
			summary_value: float | None = float(value.detach().cpu().item())
		else:
			summary_value = None
		per_horizon_summary[horizon_name] = {
			'count': count,
			'cross_entropy': summary_value,
		}
	if not per_horizon_losses:
		raise ValueError('horizon loss requires at least one supervised observation')
	loss = torch.stack(per_horizon_losses).mean()
	if not torch.isfinite(loss):
		raise FloatingPointError('fractional horizon loss must be finite')
	return loss, {
		'active_horizon_count': len(per_horizon_losses),
		'supervised_observation_count': sum(per_horizon_counts),
		'per_horizon': per_horizon_summary,
		'macro_cross_entropy': float(loss.detach().cpu().item()),
	}


def fractional_target_weights(
	sample_float: torch.Tensor,
	*,
	window_start: int = HORIZON_WINDOW_START,
	window_stop: int = HORIZON_WINDOW_STOP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	'''Return valid mask, floor/ceil indices, and their linear weights.'''
	if not isinstance(sample_float, torch.Tensor):
		raise TypeError('sample_float must be a tensor')
	if not sample_float.is_floating_point():
		raise TypeError('sample_float must have a floating dtype')
	if window_stop <= window_start:
		raise ValueError('window_stop must be greater than window_start')
	valid = (
		torch.isfinite(sample_float)
		& (sample_float >= window_start)
		& (sample_float < window_stop)
	)
	relative = torch.where(
		valid,
		sample_float - float(window_start),
		torch.zeros_like(sample_float),
	)
	floor_index = torch.floor(relative).to(dtype=torch.long)
	ceil_index = torch.clamp(floor_index + 1, max=window_stop - window_start - 1)
	ceil_weight = relative - floor_index.to(dtype=relative.dtype)
	floor_weight = 1.0 - ceil_weight
	return valid, floor_index, ceil_index, floor_weight, ceil_weight


def training_horizon_observation_counts(
	supervision_mask: np.ndarray | torch.Tensor,
) -> tuple[int, ...]:
	'''Count job-level observations for each horizon.'''
	if isinstance(supervision_mask, torch.Tensor):
		mask = supervision_mask.detach()
		if mask.dtype != torch.bool:
			raise TypeError('training supervision mask must have dtype bool')
		if mask.ndim == 3 and mask.shape[0] == len(HORIZON_NAMES):
			return tuple(
				int(torch.count_nonzero(mask[index]).cpu().item())
				for index in range(len(HORIZON_NAMES))
			)
		if mask.ndim == 4 and mask.shape[1] == len(HORIZON_NAMES):
			return tuple(
				int(torch.count_nonzero(mask[:, index]).cpu().item())
				for index in range(len(HORIZON_NAMES))
			)
	else:
		mask = np.asarray(supervision_mask)
		if mask.dtype != np.bool_:
			raise TypeError('training supervision mask must have dtype bool')
		if mask.ndim == 3 and mask.shape[0] == len(HORIZON_NAMES):
			return tuple(
				int(np.count_nonzero(mask[index]))
				for index in range(len(HORIZON_NAMES))
			)
		if mask.ndim == 4 and mask.shape[1] == len(HORIZON_NAMES):
			return tuple(
				int(np.count_nonzero(mask[:, index]))
				for index in range(len(HORIZON_NAMES))
			)
		raise ValueError(
			'training supervision mask must have shape [5,X,Y] or [B,5,X,Y]'
		)
	raise ValueError(
		'training supervision mask must have shape [5,X,Y] or [B,5,X,Y]'
	)


def validate_training_horizon_coverage(
	counts_or_mask: Sequence[int] | np.ndarray | torch.Tensor,
) -> tuple[int, ...]:
	'''Fail preflight when any job-level horizon has zero observations.'''
	if isinstance(counts_or_mask, (np.ndarray, torch.Tensor)):
		counts = training_horizon_observation_counts(counts_or_mask)
	else:
		counts = tuple(counts_or_mask)
		if len(counts) != len(HORIZON_NAMES) or any(
			not isinstance(count, Integral)
			or isinstance(count, bool)
			or count < 0
			for count in counts
		):
			raise ValueError('training counts must contain five non-negative integers')
	zero = [
		HORIZON_NAMES[index]
		for index, count in enumerate(counts)
		if int(count) == 0
	]
	if zero:
		raise ValueError(
			'training observation count must be positive for every horizon; '
			f'zero coverage: {zero!r}'
		)
	return tuple(int(count) for count in counts)


def _validate_loss_inputs(  # noqa: C901
	logits: torch.Tensor,
	target_sample_float: torch.Tensor,
	supervision_mask: torch.Tensor,
) -> None:
	if not isinstance(logits, torch.Tensor) or logits.ndim != 5:
		raise ValueError('logits must have shape [B,5,X,Y,Z]')
	if logits.shape[1] != len(HORIZON_NAMES):
		raise ValueError('logits must contain exactly five horizon channels')
	if logits.shape[-1] != HORIZON_WINDOW_STOP - HORIZON_WINDOW_START:
		raise ValueError('logits vertical dimension must be 216 samples')
	if not logits.is_floating_point():
		raise TypeError('logits must have a floating dtype')
	expected = logits.shape[:-1]
	if tuple(target_sample_float.shape) != tuple(expected):
		raise ValueError('target_sample_float must have shape [B,5,X,Y]')
	if tuple(supervision_mask.shape) != tuple(expected):
		raise ValueError('supervision_mask must have shape [B,5,X,Y]')
	if not target_sample_float.is_floating_point():
		raise TypeError('target_sample_float must have a floating dtype')
	if supervision_mask.dtype != torch.bool:
		raise TypeError('supervision_mask must have dtype bool')
	if (
		logits.device != target_sample_float.device
		or logits.device != supervision_mask.device
	):
		raise ValueError('logits, targets, and supervision mask must share a device')
	if not torch.isfinite(logits).all():
		raise FloatingPointError('horizon logits must be finite')


__all__ = [
	'fractional_horizon_cross_entropy',
	'fractional_target_weights',
	'training_horizon_observation_counts',
	'validate_training_horizon_coverage',
]
