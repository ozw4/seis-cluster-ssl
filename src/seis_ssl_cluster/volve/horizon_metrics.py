'''Horizon-first metrics for fractional Volve TWT predictions.'''

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_tiles import (
	HORIZON_WINDOW_LENGTH,
	HORIZON_WINDOW_START,
	HORIZON_WINDOW_STOP,
)

if TYPE_CHECKING:
	from collections.abc import Mapping


def soft_argmax_global_sample(
	logits: torch.Tensor,
	*,
	window_start: int = HORIZON_WINDOW_START,
) -> torch.Tensor:
	'''Return fractional global samples from a Z-axis softmax expectation.'''
	_validate_logits(logits)
	probabilities = torch.softmax(logits.float(), dim=-1)
	indices = torch.arange(
		logits.shape[-1],
		device=logits.device,
		dtype=probabilities.dtype,
	)
	return float(window_start) + torch.sum(probabilities * indices, dim=-1)


def hard_argmax_global_sample(
	logits: torch.Tensor,
	*,
	window_start: int = HORIZON_WINDOW_START,
) -> torch.Tensor:
	'''Return integer global samples as a diagnostic prediction.'''
	_validate_logits(logits)
	return torch.argmax(logits, dim=-1) + window_start


def compute_horizon_metrics(
	predicted_global_sample: np.ndarray | torch.Tensor,
	target_sample_float: np.ndarray | torch.Tensor,
	evaluation_mask: np.ndarray | torch.Tensor,
	*,
	sample_interval_ms: float = 4.0,
) -> Mapping[str, object]:
	'''Compute per-horizon errors first, then macro-average horizon metrics.'''
	predicted = _numpy(predicted_global_sample)
	target = _numpy(target_sample_float)
	mask = _bool_numpy(evaluation_mask)
	if predicted.shape != target.shape or mask.shape != target.shape:
		raise ValueError('predictions, targets, and mask must have matching shapes')
	if target.ndim != 4 or target.shape[1] != len(HORIZON_NAMES):
		raise ValueError('horizon metric inputs must have shape [B,5,X,Y]')
	if not np.isfinite(sample_interval_ms) or sample_interval_ms <= 0.0:
		raise ValueError('sample_interval_ms must be finite and positive')

	eligible = (
		mask
		& np.isfinite(target)
		& (target >= HORIZON_WINDOW_START)
		& (target < HORIZON_WINDOW_STOP)
	)
	predicted_finite = np.isfinite(predicted)
	per_horizon: dict[str, dict[str, float | int | None]] = {}
	macro_values: dict[str, list[float]] = {
		'mae_samples': [],
		'within_1': [],
		'within_2': [],
		'within_4': [],
	}
	total_eligible = 0
	total_predicted = 0
	for horizon_index, horizon_name in enumerate(HORIZON_NAMES):
		target_mask = eligible[:, horizon_index]
		prediction_mask = target_mask & predicted_finite[:, horizon_index]
		count = int(np.count_nonzero(target_mask))
		prediction_count = int(np.count_nonzero(prediction_mask))
		missing = count - prediction_count
		total_eligible += count
		total_predicted += prediction_count
		if prediction_count:
			error = np.abs(
				predicted[:, horizon_index][prediction_mask]
				- target[:, horizon_index][prediction_mask]
			).astype(np.float64, copy=False)
			mae = float(np.mean(error))
			within_1 = float(np.mean(error <= 1.0))
			within_2 = float(np.mean(error <= 2.0))
			within_4 = float(np.mean(error <= 4.0))
			median = float(np.median(error))
			p95 = float(np.percentile(error, 95))
			macro_values['mae_samples'].append(mae)
			macro_values['within_1'].append(within_1)
			macro_values['within_2'].append(within_2)
			macro_values['within_4'].append(within_4)
		else:
			mae = within_1 = within_2 = within_4 = median = p95 = None
		per_horizon[horizon_name] = {
			'count': count,
			'predicted_count': prediction_count,
			'missing_prediction_count': missing,
			'mae_samples': mae,
			'mae_ms': None if mae is None else mae * sample_interval_ms,
			'median_abs_error_samples': median,
			'p95_abs_error_samples': p95,
			'within_1': within_1,
			'within_2': within_2,
			'within_4': within_4,
		}
	macro = {
		name: _mean_or_none(values) for name, values in macro_values.items()
	}
	order = predicted_adjacent_order_violation(
		predicted,
		eligible & predicted_finite,
	)
	return {
		'macro_mae_samples': macro['mae_samples'],
		'macro_within_2_samples': macro['within_2'],
		'macro': {
			'within_1': macro['within_1'],
			'within_4': macro['within_4'],
		},
		'per_horizon': per_horizon,
		'coverage': {
			'eligible_count': total_eligible,
			'predicted_count': total_predicted,
			'fraction': (
				float(total_predicted / total_eligible) if total_eligible else None
			),
		},
		'missing_prediction_count': total_eligible - total_predicted,
		'predicted_adjacent_order_violation_rate': order['rate'],
		'predicted_adjacent_order_pair_count': order['pair_count'],
	}


def predicted_adjacent_order_violation(
	predicted_global_sample: np.ndarray | torch.Tensor,
	valid_prediction_mask: np.ndarray | torch.Tensor,
) -> Mapping[str, float | int | None]:
	'''Measure adjacent pair violations without sorting predictions.'''
	predicted = _numpy(predicted_global_sample)
	valid = _bool_numpy(valid_prediction_mask)
	if predicted.shape != valid.shape:
		raise ValueError('prediction and order-valid mask shapes must match')
	if predicted.ndim != 4 or predicted.shape[1] != len(HORIZON_NAMES):
		raise ValueError('order metric inputs must have shape [B,5,X,Y]')
	pair_valid = valid[:, :-1] & valid[:, 1:]
	pair_count = int(np.count_nonzero(pair_valid))
	violations = (predicted[:, :-1] >= predicted[:, 1:]) & pair_valid
	violation_count = int(np.count_nonzero(violations))
	return {
		'pair_count': pair_count,
		'violation_count': violation_count,
		'rate': float(violation_count / pair_count) if pair_count else None,
	}


def _validate_logits(logits: torch.Tensor) -> None:
	if not isinstance(logits, torch.Tensor) or logits.ndim != 5:
		raise ValueError('logits must have shape [B,5,X,Y,216]')
	if logits.shape[1] != len(HORIZON_NAMES):
		raise ValueError('logits must contain exactly five horizon channels')
	if logits.shape[-1] != HORIZON_WINDOW_LENGTH:
		raise ValueError('logits vertical dimension must be 216 samples')
	if not logits.is_floating_point():
		raise TypeError('logits must have a floating dtype')
	if torch.isnan(logits).any() or torch.isposinf(logits).any():
		raise FloatingPointError('horizon logits must not contain NaN or +Inf')
	if not torch.isfinite(logits).any(dim=-1).all():
		raise FloatingPointError('every horizon distribution needs a finite logit')


def _numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().numpy()
	return np.asarray(value)


def _bool_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
	array = _numpy(value)
	if array.dtype != np.bool_:
		raise TypeError('metric masks must have dtype bool')
	return array


def _mean_or_none(values: list[float]) -> float | None:
	return float(np.mean(values)) if values else None


__all__ = [
	'compute_horizon_metrics',
	'hard_argmax_global_sample',
	'predicted_adjacent_order_violation',
	'soft_argmax_global_sample',
]
