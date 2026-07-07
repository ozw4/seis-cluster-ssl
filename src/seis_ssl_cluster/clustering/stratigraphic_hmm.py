"""Stratigraphic HMM clustering backend scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, NoReturn

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Mapping


@dataclass(frozen=True)
class HMMTransitionSettings:
	"""Transition penalties for ordered stratigraphic HMM decoding."""

	same_cost: float
	advance_cost: float
	jump_cost: float
	reverse_cost: float
	forbid_reverse: bool
	max_jump: int | None


def build_ordered_transition_costs(
	k: int,
	settings: HMMTransitionSettings,
) -> np.ndarray:
	"""Build ordered-state transition costs with rows as previous states."""
	_validate_positive_int(k, 'k')
	_validate_transition_settings(settings)

	previous = np.arange(k)[:, np.newaxis]
	next_state = np.arange(k)[np.newaxis, :]
	delta = next_state - previous

	costs = np.full((k, k), np.inf, dtype=np.float32)
	same = delta == 0
	forward = delta > 0
	reverse = delta < 0

	costs[same] = np.float32(settings.same_cost)
	costs[forward] = np.float32(
		settings.advance_cost + settings.jump_cost * (delta[forward] - 1),
	)
	if not settings.forbid_reverse:
		costs[reverse] = np.float32(
			settings.reverse_cost + settings.jump_cost * (-delta[reverse] - 1),
		)

	if settings.max_jump is not None:
		too_far = np.abs(delta) > settings.max_jump
		costs[too_far & ~same] = np.inf

	return costs


def viterbi_decode_costs(
	emission_costs: np.ndarray,
	transition_costs: np.ndarray,
) -> np.ndarray:
	"""Decode the minimum-cost state path with deterministic tie-breaking."""
	emissions = _as_float_matrix(emission_costs, 'emission_costs')
	transitions = _as_float_matrix(transition_costs, 'transition_costs')
	if emissions.shape[0] == 0 or emissions.shape[1] == 0:
		raise ValueError('emission_costs must be non-empty in both dimensions')
	if not np.all(np.isfinite(emissions)):
		raise ValueError('emission_costs must contain only finite values')

	k = emissions.shape[1]
	if transitions.shape != (k, k):
		raise ValueError(
			'transition_costs must have shape '
			f'({k}, {k}); got {transitions.shape}'
		)
	if np.isnan(transitions).any():
		raise ValueError('transition_costs must not contain NaN values')

	t_count = emissions.shape[0]
	dp = np.empty((t_count, k), dtype=np.float64)
	backpointers = np.zeros((t_count, k), dtype=np.int32)
	dp[0] = emissions[0]

	for t_index in range(1, t_count):
		candidates = dp[t_index - 1, :, np.newaxis] + transitions
		previous = np.argmin(candidates, axis=0)
		best_transition_costs = candidates[previous, np.arange(k)]
		dp[t_index] = best_transition_costs + emissions[t_index]
		backpointers[t_index] = previous.astype(np.int32)

	final_state = int(np.argmin(dp[-1]))
	if not np.isfinite(dp[-1, final_state]):
		raise ValueError(
			'no finite path exists for emission_costs and transition_costs',
		)

	path = np.empty(t_count, dtype=np.int32)
	path[-1] = final_state
	for t_index in range(t_count - 1, 0, -1):
		path[t_index - 1] = backpointers[t_index, path[t_index]]
	return path


def contiguous_true_segments(mask: np.ndarray) -> tuple[slice, ...]:
	"""Return contiguous true spans from a one-dimensional boolean mask."""
	mask_array = np.asarray(mask)
	if mask_array.ndim != 1:
		raise ValueError(f'mask must be 1D; got shape {mask_array.shape}')
	if mask_array.dtype != np.bool_:
		raise TypeError('mask must have boolean dtype')

	segments: list[slice] = []
	start: int | None = None
	for index, is_valid in enumerate(mask_array):
		if bool(is_valid) and start is None:
			start = index
		elif not bool(is_valid) and start is not None:
			segments.append(slice(start, index))
			start = None
	if start is not None:
		segments.append(slice(start, mask_array.size))
	return tuple(segments)


def decode_trace_segments(
	emission_costs: np.ndarray,
	valid_mask: np.ndarray,
	transition_costs: np.ndarray,
) -> np.ndarray:
	"""Decode valid vertical trace segments independently."""
	emissions = _as_float_matrix(emission_costs, 'emission_costs')
	if emissions.shape[0] == 0 or emissions.shape[1] == 0:
		raise ValueError('emission_costs must be non-empty in both dimensions')
	if not np.all(np.isfinite(emissions)):
		raise ValueError('emission_costs must contain only finite values')

	mask = np.asarray(valid_mask)
	if mask.ndim != 1:
		raise ValueError(f'valid_mask must be 1D; got shape {mask.shape}')
	if mask.dtype != np.bool_:
		raise TypeError('valid_mask must have boolean dtype')
	if mask.shape != (emissions.shape[0],):
		raise ValueError(
			'valid_mask must have shape '
			f'({emissions.shape[0]},); got {mask.shape}'
		)

	transitions = _as_float_matrix(transition_costs, 'transition_costs')
	k = emissions.shape[1]
	if transitions.shape != (k, k):
		raise ValueError(
			'transition_costs must have shape '
			f'({k}, {k}); got {transitions.shape}'
		)
	if np.isnan(transitions).any():
		raise ValueError('transition_costs must not contain NaN values')

	labels = np.full(emissions.shape[0], -1, dtype=np.int32)
	for segment in contiguous_true_segments(mask):
		labels[segment] = viterbi_decode_costs(emissions[segment], transitions)
	return labels


def run_stratigraphic_hmm_clustering(config: Mapping[str, object]) -> NoReturn:
	"""Run stratigraphic HMM clustering from a validated config mapping."""
	raise NotImplementedError('stratigraphic_hmm_kmeans is not implemented yet')


def _validate_transition_settings(settings: HMMTransitionSettings) -> None:
	for field in ('same_cost', 'advance_cost', 'jump_cost', 'reverse_cost'):
		_validate_nonnegative_finite_cost(getattr(settings, field), field)
	if not isinstance(settings.forbid_reverse, bool):
		raise TypeError('forbid_reverse must be bool')
	if settings.max_jump is not None:
		_validate_positive_int(settings.max_jump, 'max_jump')


def _validate_nonnegative_finite_cost(value: object, name: str) -> None:
	if isinstance(value, bool) or not isinstance(value, Real):
		raise TypeError(f'{name} must be a finite non-negative number')
	if not np.isfinite(float(value)) or float(value) < 0.0:
		raise ValueError(f'{name} must be a finite non-negative number')


def _validate_positive_int(value: object, name: str) -> None:
	if isinstance(value, bool) or not isinstance(value, Integral):
		raise TypeError(f'{name} must be a positive integer')
	if int(value) <= 0:
		raise ValueError(f'{name} must be a positive integer')


def _as_float_matrix(value: np.ndarray, name: str) -> np.ndarray:
	array = np.asarray(value, dtype=np.float64)
	if array.ndim != 2:
		raise ValueError(f'{name} must be 2D; got shape {array.shape}')
	return array


__all__ = [
	'HMMTransitionSettings',
	'build_ordered_transition_costs',
	'contiguous_true_segments',
	'decode_trace_segments',
	'run_stratigraphic_hmm_clustering',
	'viterbi_decode_costs',
]
