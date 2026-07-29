"""One-step XY lateral messages followed by ordered-HMM hard redecoding.

This module deliberately works on one vertical trace and at most its four XY
neighbours.  It does not construct survey-sized arrays or iterate messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LATERAL_SMOOTHING_SEMANTICS = 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1'


@dataclass(frozen=True)
class LateralMessageResult:
	"""Per-token diagnostics for one normalized lateral message."""

	distance: np.ndarray
	affinity: np.ndarray
	neighbor_count: np.ndarray
	weight_sum: np.ndarray
	message: np.ndarray
	message_entropy: np.ndarray


@dataclass(frozen=True)
class LateralCostUpdateResult:
	"""Emission costs after the one-step lateral update."""

	emission_costs: np.ndarray
	changed_cost_magnitude: np.ndarray


@dataclass(frozen=True)
class LateralSmoothingResult:
	"""Complete one-trace M5-LS result, including hard ordered labels."""

	emission_costs: np.ndarray
	labels: np.ndarray
	diagnostics: LateralMessageResult
	changed_cost_magnitude: np.ndarray


def enumerate_xy_four_neighbors(
	coordinate: tuple[int, int, int],
	grid_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
	"""Return in-bounds XY neighbours in fixed x-, x+, y-, y+ order."""
	if len(coordinate) != 3 or len(grid_shape) != 3:
		raise ValueError('coordinate and grid_shape must each contain three values')
	x, y, z = coordinate
	x_count, y_count, z_count = grid_shape
	values = (*coordinate, *grid_shape)
	if any(
		isinstance(value, bool) or not isinstance(value, (int, np.integer))
		for value in values
	):
		raise TypeError('coordinate and grid_shape values must be integers')
	if x_count <= 0 or y_count <= 0 or z_count <= 0:
		raise ValueError('grid_shape values must be positive')
	if not (0 <= x < x_count and 0 <= y < y_count and 0 <= z < z_count):
		raise ValueError('coordinate must be within grid_shape')
	candidates = ((x - 1, y, z), (x + 1, y, z), (x, y - 1, z), (x, y + 1, z))
	return tuple(
		item
		for item in candidates
		if 0 <= item[0] < x_count and 0 <= item[1] < y_count
	)


def cosine_rbf_affinities(
	center_embedding: np.ndarray,
	neighbor_embeddings: np.ndarray,
	center_valid_mask: np.ndarray,
	neighbor_valid_masks: np.ndarray,
	*,
	affinity_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return distance and cosine-RBF affinity for valid XY edge endpoints."""
	center, neighbors, center_valid, neighbor_valid = _validate_embedding_inputs(
		center_embedding,
		neighbor_embeddings,
		center_valid_mask,
		neighbor_valid_masks,
	)
	_validate_positive_finite(affinity_scale, 'affinity_scale')
	n_count, t_count, _ = neighbors.shape
	distance = np.zeros((n_count, t_count), dtype=np.float64)
	affinity = np.zeros((n_count, t_count), dtype=np.float64)
	center_norm = np.linalg.norm(center, axis=1)
	neighbor_norm = np.linalg.norm(neighbors, axis=2)
	if np.any(center_valid & (center_norm == 0.0)):
		raise ValueError('valid center_embedding rows must have nonzero norm')
	if np.any(neighbor_valid & (neighbor_norm == 0.0)):
		raise ValueError('valid neighbor_embeddings rows must have nonzero norm')
	available = neighbor_valid & center_valid[np.newaxis, :]
	if not np.any(available):
		return distance, affinity
	dot = np.einsum('td,ntd->nt', center, neighbors, optimize=True)
	with np.errstate(invalid='ignore', divide='ignore'):
		cosine = dot / (neighbor_norm * center_norm[np.newaxis, :])
	distance[available] = np.clip(1.0 - cosine[available], 0.0, 2.0)
	affinity[available] = np.exp(-distance[available] / float(affinity_scale))
	return distance, affinity


def normalized_lateral_message(  # noqa: C901
	neighbor_posterior: np.ndarray,
	affinity: np.ndarray,
	center_valid_mask: np.ndarray,
	neighbor_valid_masks: np.ndarray,
) -> LateralMessageResult:
	"""Average neighbour state posteriors with normalized positive affinities."""
	posterior = np.asarray(neighbor_posterior, dtype=np.float64)
	weights = np.asarray(affinity, dtype=np.float64)
	center_valid = _as_bool_vector(center_valid_mask, 'center_valid_mask')
	neighbor_valid = _as_bool_matrix(neighbor_valid_masks, 'neighbor_valid_masks')
	if posterior.ndim != 3:
		raise ValueError(
			'neighbor_posterior must have shape [N, T, K]; '
			f'got {posterior.shape!r}'
		)
	n_count, t_count, k = posterior.shape
	if k < 2:
		raise ValueError('neighbor_posterior K must be at least 2')
	if weights.shape != (n_count, t_count):
		raise ValueError(
			'affinity must have shape [N, T] matching neighbor_posterior; '
			f'got {weights.shape!r}'
		)
	if center_valid.shape != (t_count,) or neighbor_valid.shape != (n_count, t_count):
		raise ValueError('valid mask shapes must match neighbor_posterior')
	if n_count > 4:
		raise ValueError('at most four XY neighbours are supported')
	_validate_posterior(posterior, neighbor_valid)
	if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
		raise ValueError('affinity must be finite and non-negative')
	available = neighbor_valid & center_valid[np.newaxis, :] & (weights > 0.0)
	if np.any((~available) & (weights != 0.0)):
		raise ValueError('affinity must be zero for invalid edge endpoints')
	masked_weights = np.where(available, weights, 0.0)
	weight_sum = np.sum(masked_weights, axis=0, dtype=np.float64)
	message = np.zeros((t_count, k), dtype=np.float64)
	has_neighbors = weight_sum > 0.0
	if np.any(has_neighbors):
		message[has_neighbors] = (
			np.einsum('nt,ntk->tk', masked_weights, posterior, optimize=True)[
				has_neighbors
			]
			/ weight_sum[has_neighbors, np.newaxis]
		)
	if np.any(has_neighbors) and not np.allclose(
		np.sum(message[has_neighbors], axis=1),
		1.0,
		rtol=1.0e-10,
		atol=1.0e-12,
	):
		raise ValueError('normalized lateral message rows must sum to one')
	entropy = np.zeros(t_count, dtype=np.float64)
	if np.any(has_neighbors):
		active_message = message[has_neighbors]
		with np.errstate(divide='ignore', invalid='ignore'):
			entropy[has_neighbors] = -np.sum(
				np.where(
					active_message > 0.0,
					active_message * np.log(active_message),
					0.0,
				),
				axis=1,
			)
	return LateralMessageResult(
		distance=np.zeros((n_count, t_count), dtype=np.float64),
		affinity=weights.copy(),
		neighbor_count=np.sum(available, axis=0, dtype=np.int32),
		weight_sum=weight_sum,
		message=message,
		message_entropy=entropy,
	)


def apply_lateral_message_to_emission_costs(  # noqa: PLR0913
	emission_costs: np.ndarray,
	message: np.ndarray,
	center_valid_mask: np.ndarray,
	weight_sum: np.ndarray,
	*,
	emission_gap_scale: float,
	pairwise_strength_ratio: float,
) -> LateralCostUpdateResult:
	"""Apply one dimensionless lateral message update to emission costs."""
	costs = _as_finite_cost_matrix(emission_costs)
	message_array = np.asarray(message, dtype=np.float64)
	valid = _as_bool_vector(center_valid_mask, 'center_valid_mask')
	weights = np.asarray(weight_sum, dtype=np.float64)
	if message_array.shape != costs.shape:
		raise ValueError('message must have the same shape as emission_costs')
	if valid.shape != (costs.shape[0],) or weights.shape != (costs.shape[0],):
		raise ValueError(
			'center_valid_mask and weight_sum must have one value per token'
		)
	if not np.all(np.isfinite(message_array)) or np.any(message_array < 0.0):
		raise ValueError('message must be finite and non-negative')
	if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
		raise ValueError('weight_sum must be finite and non-negative')
	has_neighbors = weights > 0.0
	if np.any(~has_neighbors & np.any(message_array != 0.0, axis=1)):
		raise ValueError('message must be zero when weight_sum is zero')
	if np.any(has_neighbors) and not np.allclose(
		np.sum(message_array[has_neighbors], axis=1),
		1.0,
		rtol=1.0e-10,
		atol=1.0e-12,
	):
		raise ValueError('message rows with neighbours must sum to one')
	_validate_positive_finite(emission_gap_scale, 'emission_gap_scale')
	_validate_nonnegative_finite(pairwise_strength_ratio, 'pairwise_strength_ratio')
	updated = costs.copy()
	active = valid & has_neighbors & (pairwise_strength_ratio > 0.0)
	if np.any(active):
		updated[active] -= (
			float(pairwise_strength_ratio)
			* float(emission_gap_scale)
			* message_array[active]
		)
	return LateralCostUpdateResult(
		emission_costs=updated,
		changed_cost_magnitude=np.max(np.abs(updated - costs), axis=1),
	)


def redecode_ordered_lateral_trace(  # noqa: PLR0913
	emission_costs: np.ndarray,
	valid_mask: np.ndarray,
	transition_costs: np.ndarray,
	*,
	initial_state_costs: np.ndarray | None = None,
	terminal_state_costs: np.ndarray | None = None,
	expected_boundary_count: int | None = None,
	boundary_count_weight: float = 0.0,
) -> np.ndarray:
	"""Viterbi-decode compacted valid trace positions as one ordered sequence."""
	costs = _as_finite_cost_matrix(emission_costs)
	valid = _as_bool_vector(valid_mask, 'valid_mask')
	if valid.shape != (costs.shape[0],):
		raise ValueError('valid_mask must have one value per emission-cost row')
	labels = np.full(costs.shape[0], -1, dtype=np.int32)
	valid_positions = np.flatnonzero(valid)
	if valid_positions.size == 0:
		return labels
	# The ordered-HMM implementation imports optional clustering dependencies.
	# Keep this one-trace numerical module importable by lightweight commands
	# that do not request hard redecoding.
	from seis_ssl_cluster.clustering.stratigraphic_hmm import (  # noqa: PLC0415
		viterbi_decode_costs,
	)

	path = viterbi_decode_costs(
		costs[valid_positions],
		transition_costs,
		initial_state_costs=initial_state_costs,
		terminal_state_costs=terminal_state_costs,
		expected_boundary_count=expected_boundary_count,
		boundary_count_weight=boundary_count_weight,
	)
	if path.shape != (valid_positions.size,) or np.any(
		(path < 0) | (path >= costs.shape[1])
	):
		raise ValueError('Viterbi returned an invalid path')
	labels[valid_positions] = path.astype(np.int32, copy=False)
	return labels


def smooth_and_redecode_ordered_trace(  # noqa: PLR0913
	center_embedding: np.ndarray,
	neighbor_embeddings: np.ndarray,
	neighbor_posterior: np.ndarray,
	center_valid_mask: np.ndarray,
	neighbor_valid_masks: np.ndarray,
	emission_costs: np.ndarray,
	transition_costs: np.ndarray,
	*,
	affinity_scale: float,
	emission_gap_scale: float,
	pairwise_strength_ratio: float,
	initial_state_costs: np.ndarray | None = None,
	terminal_state_costs: np.ndarray | None = None,
	expected_boundary_count: int | None = None,
	boundary_count_weight: float = 0.0,
) -> LateralSmoothingResult:
	"""Compute one M5-LS message, update costs, and reproject to hard labels."""
	distance, affinity = cosine_rbf_affinities(
		center_embedding,
		neighbor_embeddings,
		center_valid_mask,
		neighbor_valid_masks,
		affinity_scale=affinity_scale,
	)
	diagnostics = normalized_lateral_message(
		neighbor_posterior,
		affinity,
		center_valid_mask,
		neighbor_valid_masks,
	)
	diagnostics = LateralMessageResult(
		distance=distance,
		affinity=diagnostics.affinity,
		neighbor_count=diagnostics.neighbor_count,
		weight_sum=diagnostics.weight_sum,
		message=diagnostics.message,
		message_entropy=diagnostics.message_entropy,
	)
	update = apply_lateral_message_to_emission_costs(
		emission_costs,
		diagnostics.message,
		center_valid_mask,
		diagnostics.weight_sum,
		emission_gap_scale=emission_gap_scale,
		pairwise_strength_ratio=pairwise_strength_ratio,
	)
	return LateralSmoothingResult(
		emission_costs=update.emission_costs,
		labels=redecode_ordered_lateral_trace(
			update.emission_costs,
			center_valid_mask,
			transition_costs,
			initial_state_costs=initial_state_costs,
			terminal_state_costs=terminal_state_costs,
			expected_boundary_count=expected_boundary_count,
			boundary_count_weight=boundary_count_weight,
		),
		diagnostics=diagnostics,
		changed_cost_magnitude=update.changed_cost_magnitude,
	)


def median_scale_with_floor(values: np.ndarray, *, floor: float = 1.0e-6) -> float:
	"""Return ``max(median(values), floor)`` for a small finite vector."""
	array = np.asarray(values, dtype=np.float64)
	if array.ndim != 1:
		raise ValueError(f'values must be 1D; got shape {array.shape!r}')
	if not np.all(np.isfinite(array)):
		raise ValueError('values must be finite')
	_validate_positive_finite(floor, 'floor')
	if array.size == 0:
		return float(floor)
	return max(float(np.median(array)), float(floor))


def _validate_embedding_inputs(
	center_embedding: np.ndarray,
	neighbor_embeddings: np.ndarray,
	center_valid_mask: np.ndarray,
	neighbor_valid_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	center = np.asarray(center_embedding, dtype=np.float64)
	neighbors = np.asarray(neighbor_embeddings, dtype=np.float64)
	center_valid = _as_bool_vector(center_valid_mask, 'center_valid_mask')
	neighbor_valid = _as_bool_matrix(neighbor_valid_masks, 'neighbor_valid_masks')
	if center.ndim != 2 or center.shape[1] == 0:
		raise ValueError('center_embedding must have shape [T, D] with D > 0')
	if neighbors.ndim != 3 or neighbors.shape[2] != center.shape[1]:
		raise ValueError(
			'neighbor_embeddings must have shape [N, T, D] '
			'matching center_embedding'
		)
	if neighbors.shape[0] > 4:
		raise ValueError('at most four XY neighbours are supported')
	if (
		center_valid.shape != (center.shape[0],)
		or neighbor_valid.shape != neighbors.shape[:2]
	):
		raise ValueError('valid mask shapes must match embedding arrays')
	if not np.all(np.isfinite(center)) or not np.all(np.isfinite(neighbors)):
		raise ValueError('embedding arrays must contain only finite values')
	return center, neighbors, center_valid, neighbor_valid


def _validate_posterior(posterior: np.ndarray, valid_mask: np.ndarray) -> None:
	if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
		raise ValueError('neighbor_posterior must be finite and non-negative')
	valid_rows = posterior[valid_mask]
	if valid_rows.size and not np.allclose(
		np.sum(valid_rows, axis=1),
		1.0,
		rtol=1.0e-10,
		atol=1.0e-12,
	):
		raise ValueError('valid neighbor_posterior rows must sum to one')
	if np.any(posterior[~valid_mask] != 0.0):
		raise ValueError('invalid neighbor_posterior rows must be all zero')


def _as_finite_cost_matrix(values: np.ndarray) -> np.ndarray:
	array = np.asarray(values, dtype=np.float64)
	if array.ndim != 2 or array.shape[1] < 2:
		raise ValueError('emission_costs must have shape [T, K] with K at least 2')
	if not np.all(np.isfinite(array)):
		raise ValueError('emission_costs must contain only finite values')
	return array


def _as_bool_vector(values: np.ndarray, name: str) -> np.ndarray:
	array = np.asarray(values)
	if array.dtype != np.bool_ or array.ndim != 1:
		raise ValueError(f'{name} must be a boolean 1D array')
	return array


def _as_bool_matrix(values: np.ndarray, name: str) -> np.ndarray:
	array = np.asarray(values)
	if array.dtype != np.bool_ or array.ndim != 2:
		raise ValueError(f'{name} must be a boolean 2D array')
	return array


def _validate_positive_finite(value: float, name: str) -> None:
	if not np.isfinite(value) or value <= 0.0:
		raise ValueError(f'{name} must be positive and finite')


def _validate_nonnegative_finite(value: float, name: str) -> None:
	if not np.isfinite(value) or value < 0.0:
		raise ValueError(f'{name} must be non-negative and finite')
