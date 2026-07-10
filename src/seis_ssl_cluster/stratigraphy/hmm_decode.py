"""Decode ordered prototype logits into stratigraphic HMM pseudo-targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	HMMExpectedBoundariesSettings,
	HMMPathPriorSettings,
	HMMTransitionSettings,
	_resolve_expected_boundary_count,
	build_initial_state_costs,
	build_ordered_transition_costs,
	build_terminal_state_costs,
	decode_trace_segments,
	edge_margin_mask_for_shape,
)
from seis_ssl_cluster.stratigraphy.boundary_weights import boundary_weight_tokens


@dataclass(frozen=True)
class LogitHMMPseudoTarget:
	"""In-memory HMM pseudo-target arrays decoded from prototype logits."""

	labels: np.ndarray
	confidence: np.ndarray
	valid_tokens: np.ndarray
	boundary_weight: np.ndarray
	metadata: dict[str, object]


def emission_costs_from_logits(
	logits: np.ndarray,
	*,
	eps: float = 1.0e-8,
) -> np.ndarray:
	"""Return stable ``-log_softmax`` emission costs for ``[TX, TY, TZ, K]`` logits."""
	_validate_eps(eps)
	logit_array = _validate_logits(logits)
	max_logits = np.max(logit_array, axis=-1, keepdims=True)
	log_sum_exp = max_logits + np.log(
		np.sum(np.exp(logit_array - max_logits), axis=-1, keepdims=True),
	)
	return log_sum_exp - logit_array


def decode_ordered_logits_survey(  # noqa: PLR0913
	logits: np.ndarray,
	valid_tokens: np.ndarray,
	*,
	transition: HMMTransitionSettings,
	path_prior: HMMPathPriorSettings | None = None,
	edge_margin_tokens: tuple[int, int, int] = (0, 0, 0),
	expected_boundaries: HMMExpectedBoundariesSettings | None = None,
	boundary_alpha: float = 0.0,
	boundary_tau: float = 1.0,
	eps: float = 1.0e-8,
) -> LogitHMMPseudoTarget:
	"""Decode one survey's prototype logits into ordered HMM pseudo-target arrays."""
	costs = emission_costs_from_logits(logits, eps=eps)
	logit_array = _validate_logits(logits)
	valid = _validate_valid_tokens(valid_tokens, token_grid_shape=logit_array.shape[:3])
	edge_margin = _validate_edge_margin_tokens(edge_margin_tokens)
	effective_valid = valid & edge_margin_mask_for_shape(valid.shape, edge_margin)
	if not np.any(effective_valid):
		msg = 'effective valid_tokens must contain at least one token'
		raise ValueError(msg)

	k = int(logit_array.shape[-1])
	transition_costs = build_ordered_transition_costs(k, transition)
	initial_state_costs = (
		None if path_prior is None else build_initial_state_costs(k, path_prior)
	)
	terminal_state_costs = (
		None if path_prior is None else build_terminal_state_costs(k, path_prior)
	)

	labels = np.full(valid.shape, -1, dtype=np.int32)
	x_count, y_count, _ = valid.shape
	for x_index in range(x_count):
		for y_index in range(y_count):
			trace_valid = effective_valid[x_index, y_index, :]
			valid_trace_length = int(np.count_nonzero(trace_valid))
			if valid_trace_length == 0:
				continue
			expected_boundary_count = _resolve_expected_boundary_count(
				expected_boundaries,
				k=k,
				valid_trace_length=valid_trace_length,
			)
			if expected_boundaries is None:
				boundary_count_weight = 0.0
			else:
				boundary_count_weight = float(expected_boundaries.weight)
			labels[x_index, y_index, :] = decode_trace_segments(
				costs[x_index, y_index, :, :],
				trace_valid,
				transition_costs,
				initial_state_costs=initial_state_costs,
				terminal_state_costs=terminal_state_costs,
				expected_boundary_count=expected_boundary_count,
				boundary_count_weight=boundary_count_weight,
			)

	probabilities = _softmax_probabilities(logit_array)
	confidence = np.zeros(valid.shape, dtype=np.float32)
	decoded = effective_valid
	x_coords, y_coords, z_coords = np.nonzero(decoded)
	decoded_labels = labels[decoded]
	confidence[decoded] = probabilities[
		x_coords,
		y_coords,
		z_coords,
		decoded_labels,
	].astype(np.float32, copy=False)
	boundary_weight = boundary_weight_tokens(
		labels,
		effective_valid,
		alpha=boundary_alpha,
		tau=boundary_tau,
	)

	return LogitHMMPseudoTarget(
		labels=labels,
		confidence=confidence,
		valid_tokens=effective_valid.astype(np.bool_, copy=False),
		boundary_weight=boundary_weight,
		metadata=_metadata(
			k=k,
			labels=labels,
			token_grid_shape=valid.shape,
			transition=transition,
			path_prior=path_prior,
			edge_margin_tokens=edge_margin,
			valid_tokens=valid,
			effective_valid_tokens=effective_valid,
			confidence=confidence,
			expected_boundaries=expected_boundaries,
			boundary_weight=boundary_weight,
			boundary_alpha=float(boundary_alpha),
			boundary_tau=float(boundary_tau),
		),
	)


def _softmax_probabilities(logits: np.ndarray) -> np.ndarray:
	max_logits = np.max(logits, axis=-1, keepdims=True)
	shifted = logits - max_logits
	exp_logits = np.exp(shifted)
	return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _validate_logits(logits: np.ndarray) -> np.ndarray:
	array = np.asarray(logits, dtype=np.float64)
	if array.ndim != 4:
		msg = f'logits must be 4D [TX, TY, TZ, K]; got shape {array.shape!r}'
		raise ValueError(msg)
	if array.shape[-1] < 1:
		msg = f'logits must have K >= 1; got shape {array.shape!r}'
		raise ValueError(msg)
	if not np.all(np.isfinite(array)):
		msg = 'logits must contain only finite values'
		raise ValueError(msg)
	return array


def _validate_valid_tokens(
	valid_tokens: np.ndarray,
	*,
	token_grid_shape: tuple[int, int, int],
) -> np.ndarray:
	array = np.asarray(valid_tokens)
	if array.ndim != 3:
		msg = f'valid_tokens must be 3D [TX, TY, TZ]; got shape {array.shape!r}'
		raise ValueError(msg)
	if array.shape != token_grid_shape:
		msg = (
			'valid_tokens shape must match logits token grid shape; '
			f'got {array.shape!r} and {token_grid_shape!r}'
		)
		raise ValueError(msg)
	if array.dtype != np.bool_:
		msg = f'valid_tokens dtype must be bool; got {array.dtype}'
		raise TypeError(msg)
	return array


def _validate_edge_margin_tokens(
	edge_margin_tokens: tuple[int, int, int],
) -> tuple[int, int, int]:
	try:
		values = tuple(edge_margin_tokens)
	except TypeError as exc:
		msg = (
			'edge_margin_tokens must contain exactly three integers; '
			f'got {edge_margin_tokens!r}'
		)
		raise TypeError(msg) from exc
	if len(values) != 3:
		msg = (
			'edge_margin_tokens must contain exactly three integers; '
			f'got {edge_margin_tokens!r}'
		)
		raise ValueError(msg)
	for value in values:
		if isinstance(value, bool) or not isinstance(value, Integral):
			msg = (
				'edge_margin_tokens must contain integers; '
				f'got {edge_margin_tokens!r}'
			)
			raise TypeError(msg)
		if int(value) < 0:
			msg = f'edge_margin_tokens must be non-negative; got {edge_margin_tokens!r}'
			raise ValueError(msg)
	return (int(values[0]), int(values[1]), int(values[2]))


def _validate_eps(eps: float) -> None:
	if isinstance(eps, bool) or not isinstance(eps, Real):
		msg = f'eps must be a positive finite number; got {eps!r}'
		raise TypeError(msg)
	if not np.isfinite(float(eps)) or float(eps) <= 0.0:
		msg = f'eps must be a positive finite number; got {eps!r}'
		raise ValueError(msg)


def _metadata(  # noqa: PLR0913
	*,
	k: int,
	labels: np.ndarray,
	token_grid_shape: tuple[int, int, int],
	transition: HMMTransitionSettings,
	path_prior: HMMPathPriorSettings | None,
	edge_margin_tokens: tuple[int, int, int],
	valid_tokens: np.ndarray,
	effective_valid_tokens: np.ndarray,
	confidence: np.ndarray,
	expected_boundaries: HMMExpectedBoundariesSettings | None,
	boundary_weight: np.ndarray,
	boundary_alpha: float,
	boundary_tau: float,
) -> dict[str, object]:
	decoded_confidence = confidence[effective_valid_tokens]
	valid_boundary_weight = boundary_weight[effective_valid_tokens]
	transition_count = int(
		np.count_nonzero(
			effective_valid_tokens[..., :-1]
			& effective_valid_tokens[..., 1:]
			& (labels[..., :-1] != labels[..., 1:]),
		),
	)
	return {
		'boundary_weight_summary': {
			'downweighted_valid_token_count': int(
				np.count_nonzero(valid_boundary_weight < 1.0),
			),
			'max': float(np.max(valid_boundary_weight)),
			'mean': float(np.mean(valid_boundary_weight, dtype=np.float64)),
			'min': float(np.min(valid_boundary_weight)),
			'transition_boundary_count': transition_count,
			'zero_weight_valid_token_count': int(
				np.count_nonzero(valid_boundary_weight == 0.0),
			),
		},
		'boundary_weighting': {
			'adjacent_transition_distance': 0,
			'alpha': boundary_alpha,
			'invalid_gap_crossing': False,
			'method': 'transition_distance_exponential',
			'tau': boundary_tau,
		},
		'confidence_note': (
			'confidence is the decoded-label softmax probability and should be '
			'treated as a training weight, not a calibrated probability'
		),
		'confidence_summary': {
			'max': float(np.max(decoded_confidence)),
			'mean': float(np.mean(decoded_confidence)),
			'min': float(np.min(decoded_confidence)),
		},
		'edge_margin_excluded_valid_token_count': int(
			np.count_nonzero(valid_tokens & ~effective_valid_tokens),
		),
		'edge_margin_tokens': [int(value) for value in edge_margin_tokens],
		'effective_valid_token_count': int(np.count_nonzero(effective_valid_tokens)),
		'expected_boundaries': (
			None if expected_boundaries is None else asdict(expected_boundaries)
		),
		'expected_boundaries_enabled': bool(
			expected_boundaries is not None and expected_boundaries.enabled,
		),
		'k': int(k),
		'path_prior': None if path_prior is None else asdict(path_prior),
		'path_prior_enabled': bool(path_prior is not None and path_prior.enabled),
		'token_grid_shape': [int(size) for size in token_grid_shape],
		'transition': asdict(transition),
		'valid_token_count': int(np.count_nonzero(valid_tokens)),
	}


__all__ = [
	'LogitHMMPseudoTarget',
	'decode_ordered_logits_survey',
	'emission_costs_from_logits',
]
