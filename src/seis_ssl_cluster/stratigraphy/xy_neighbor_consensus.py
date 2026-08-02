"""Single-pass source-label XY-neighbour consensus for hard stratigraphic targets.

This module is intentionally independent from the M5-LS posterior and HMM
redecoding path.  It only reads frozen hard labels and their valid-token mask,
then makes one synchronous XY consensus proposal at each same-``z`` position.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_SEMANTICS = (
	'xy_neighbor_consensus_hard_label_smoothing_v1'
)
"""Immutable semantic identifier for the source-label consensus operation."""

XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_POLICY = MappingProxyType(
	{
		'neighborhood': 'same_z_xy_four_neighbors',
		'neighbor_order': ('x_minus', 'x_plus', 'y_minus', 'y_plus'),
		'four_valid_neighbors_minimum_agreement': 3,
		'three_valid_neighbors_minimum_agreement': 3,
		'fewer_than_three_valid_neighbors': 'unchanged',
		'tied_or_nonunique_consensus': 'unchanged',
		'center_matching_consensus': 'unchanged',
		'temporal_guard': 'internal_valid_token_source_label_bounds',
		'application': 'single_pass_synchronous_source_labels',
	}
)
"""Fixed, JSON-compatible policy values for manifest and checkpoint identity."""


XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_SEMANTICS = (
	'xy_neighbor_unanimous_outlier_correction_v1'
)
"""Immutable semantic identifier for unanimous source-label correction."""

XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_POLICY = MappingProxyType(
	{
		'neighborhood': 'same_z_xy_four_neighbors',
		'neighbor_order': ('x_minus', 'x_plus', 'y_minus', 'y_plus'),
		'four_valid_neighbors_minimum_agreement': 4,
		'three_valid_neighbors_minimum_agreement': 3,
		'fewer_than_three_valid_neighbors': 'unchanged',
		'tied_or_nonunique_consensus': 'unchanged',
		'center_matching_consensus': 'unchanged',
		'temporal_guard': 'internal_valid_token_source_label_bounds',
		'application': 'single_pass_synchronous_source_labels',
	}
)
"""Fixed, JSON-compatible unanimous correction policy values."""


@dataclass(frozen=True)
class XYNeighborConsensusDiagnostics:
	"""Per-token decisions made by the fixed XY consensus policy.

	``proposal_labels`` is meaningful where ``consensus_mask`` is true.  The
	other masks intentionally describe progressively stricter stages: a raw
	neighbour consensus, a candidate differing from the center, an internal
	valid-token position, an order-compatible candidate, and a final change.
	"""

	neighbor_count: np.ndarray
	proposal_labels: np.ndarray
	consensus_mask: np.ndarray
	change_candidate_mask: np.ndarray
	internal_valid_mask: np.ndarray
	order_compatible_mask: np.ndarray
	changed_mask: np.ndarray


@dataclass(frozen=True)
class XYNeighborConsensusResult:
	"""Synchronous hard-label consensus output and its diagnostics."""

	labels: np.ndarray
	diagnostics: XYNeighborConsensusDiagnostics


def smooth_xy_neighbor_consensus_hard_labels(
	source_labels: np.ndarray,
	valid_mask: np.ndarray,
) -> XYNeighborConsensusResult:
	"""Apply the fixed same-``z`` XY consensus policy once to source labels.

	The three grid axes are ``x``, ``y``, and ``z``.  Every decision reads only
	``source_labels``; no proposal can affect another proposal in this pass.
	Invalid positions are never considered as centers or neighbours and their
	label values are copied to the result exactly as supplied.

	For a valid center, exactly three valid neighbours must agree, while four
	valid neighbours require an agreement count of at least three.  A consensus
	that equals the center remains unchanged.  A distinct proposal is accepted
	only when the center is an internal member of its trace's *valid-token*
	sequence and is bounded by the labels of the preceding and following valid
	tokens.  Physical ``z`` gaps do not reset that sequence.
	"""
	return _smooth_xy_neighbor_hard_labels(
		source_labels,
		valid_mask,
		four_valid_neighbors_minimum_agreement=3,
	)


def smooth_xy_neighbor_unanimous_hard_labels(
	source_labels: np.ndarray,
	valid_mask: np.ndarray,
) -> XYNeighborConsensusResult:
	"""Apply the fixed conservative unanimous XY correction once.

	Four valid neighbours must all agree on one label.  Three valid neighbours
	still require all three labels to agree; fewer valid neighbours never make a
	proposal.  The ordered-trace guard and synchronous source-label semantics are
	exactly the same as :func:`smooth_xy_neighbor_consensus_hard_labels`.
	"""
	return _smooth_xy_neighbor_hard_labels(
		source_labels,
		valid_mask,
		four_valid_neighbors_minimum_agreement=4,
	)


def _smooth_xy_neighbor_hard_labels(
	source_labels: np.ndarray,
	valid_mask: np.ndarray,
	*,
	four_valid_neighbors_minimum_agreement: int,
) -> XYNeighborConsensusResult:
	"""Apply one source-only XY policy with a fixed four-neighbour threshold."""
	if four_valid_neighbors_minimum_agreement not in {3, 4}:
		raise ValueError(
			'four_valid_neighbors_minimum_agreement must be either 3 or 4'
		)
	labels = _as_integer_grid(source_labels, 'source_labels')
	valid = _as_bool_grid(valid_mask, 'valid_mask')
	if labels.shape != valid.shape:
		raise ValueError('source_labels and valid_mask must have the same shape')
	if np.any(labels[valid] < 0):
		raise ValueError('valid source_labels must be non-negative')

	output = labels.copy()
	neighbor_count = np.zeros(labels.shape, dtype=np.int32)
	proposal_labels = labels.copy()
	consensus_mask = np.zeros(labels.shape, dtype=bool)
	change_candidate_mask = np.zeros(labels.shape, dtype=bool)
	internal_valid_mask = np.zeros(labels.shape, dtype=bool)
	order_compatible_mask = np.zeros(labels.shape, dtype=bool)
	changed_mask = np.zeros(labels.shape, dtype=bool)

	x_count, _, z_count = labels.shape
	for x in range(x_count):
		center_labels = labels[x]
		center_valid = valid[x]
		(
			neighbor_labels,
			neighbor_valid,
		) = _xy_neighbor_planes(labels, valid, x)
		plane_neighbor_count = np.sum(neighbor_valid, axis=0, dtype=np.int32)
		plane_neighbor_count = np.where(center_valid, plane_neighbor_count, 0)
		neighbor_count[x] = plane_neighbor_count

		plane_proposals, plane_consensus = _consensus_proposals(
			neighbor_labels,
			neighbor_valid,
			plane_neighbor_count,
			center_valid,
			center_labels,
			four_valid_neighbors_minimum_agreement,
		)
		proposal_labels[x] = plane_proposals
		consensus_mask[x] = plane_consensus
		plane_change_candidate = plane_consensus & (plane_proposals != center_labels)
		change_candidate_mask[x] = plane_change_candidate

		(
			plane_internal,
			previous_labels,
			next_labels,
		) = _internal_valid_token_bounds(center_labels, center_valid, z_count)
		internal_valid_mask[x] = plane_internal
		plane_order_compatible = (
			plane_change_candidate
			& plane_internal
			& (previous_labels <= plane_proposals)
			& (plane_proposals <= next_labels)
		)
		order_compatible_mask[x] = plane_order_compatible
		changed_mask[x] = plane_order_compatible
		output[x, plane_order_compatible] = plane_proposals[plane_order_compatible]

	return XYNeighborConsensusResult(
		labels=output,
		diagnostics=XYNeighborConsensusDiagnostics(
			neighbor_count=neighbor_count,
			proposal_labels=proposal_labels,
			consensus_mask=consensus_mask,
			change_candidate_mask=change_candidate_mask,
			internal_valid_mask=internal_valid_mask,
			order_compatible_mask=order_compatible_mask,
			changed_mask=changed_mask,
		),
	)


def _xy_neighbor_planes(
	labels: np.ndarray,
	valid: np.ndarray,
	x: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return same-``z`` XY neighbour values in fixed x-, x+, y-, y+ order."""
	_, y_count, z_count = labels.shape
	neighbor_labels = np.zeros((4, y_count, z_count), dtype=labels.dtype)
	neighbor_valid = np.zeros((4, y_count, z_count), dtype=bool)
	if x > 0:
		neighbor_labels[0] = labels[x - 1]
		neighbor_valid[0] = valid[x - 1]
	if x + 1 < labels.shape[0]:
		neighbor_labels[1] = labels[x + 1]
		neighbor_valid[1] = valid[x + 1]
	if y_count > 1:
		neighbor_labels[2, 1:] = labels[x, :-1]
		neighbor_valid[2, 1:] = valid[x, :-1]
		neighbor_labels[3, :-1] = labels[x, 1:]
		neighbor_valid[3, :-1] = valid[x, 1:]
	return neighbor_labels, neighbor_valid


def _consensus_proposals(  # noqa: PLR0913
	neighbor_labels: np.ndarray,
	neighbor_valid: np.ndarray,
	neighbor_count: np.ndarray,
	center_valid: np.ndarray,
	center_labels: np.ndarray,
	four_valid_neighbors_minimum_agreement: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute threshold-qualified consensus labels from frozen neighbour values."""
	proposals = center_labels.copy()
	best_count = np.zeros(center_labels.shape, dtype=np.int8)
	for index in range(neighbor_labels.shape[0]):
		matches = neighbor_valid & (neighbor_labels == neighbor_labels[index])
		count = np.sum(matches, axis=0, dtype=np.int8)
		is_better = count > best_count
		proposals[is_better] = neighbor_labels[index, is_better]
		best_count = np.maximum(best_count, count)
	# The three-neighbour rule is always unanimous.  The policy only varies for
	# four valid neighbours: 3/4 for the legacy consensus route and 4/4 for the
	# conservative unanimous route.
	required_agreement = np.where(
		neighbor_count == 4,
		four_valid_neighbors_minimum_agreement,
		np.where(neighbor_count == 3, 3, 5),
	)
	consensus = center_valid & (best_count >= required_agreement)
	return proposals, consensus


def _internal_valid_token_bounds(
	labels: np.ndarray,
	valid: np.ndarray,
	z_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Find prior/next source labels in each trace's valid-token sequence."""
	y_count = labels.shape[0]
	indices = np.broadcast_to(np.arange(z_count, dtype=np.int64), (y_count, z_count))
	previous_seen = np.maximum.accumulate(np.where(valid, indices, -1), axis=1)
	next_seen = np.minimum.accumulate(
		np.where(valid, indices, z_count)[:, ::-1], axis=1
	)[:, ::-1]
	previous_indices = np.full((y_count, z_count), -1, dtype=np.int64)
	next_indices = np.full((y_count, z_count), z_count, dtype=np.int64)
	if z_count > 1:
		previous_indices[:, 1:] = previous_seen[:, :-1]
		next_indices[:, :-1] = next_seen[:, 1:]
	internal = valid & (previous_indices >= 0) & (next_indices < z_count)
	safe_previous = np.maximum(previous_indices, 0)
	safe_next = np.minimum(next_indices, max(z_count - 1, 0))
	rows = np.arange(y_count)[:, np.newaxis]
	return (
		internal,
		labels[rows, safe_previous],
		labels[rows, safe_next],
	)


def _as_integer_grid(values: np.ndarray, name: str) -> np.ndarray:
	array = np.asarray(values)
	if array.ndim != 3 or not np.issubdtype(array.dtype, np.integer):
		raise ValueError(f'{name} must be a three-dimensional integer array')
	if array.dtype == np.bool_:
		raise ValueError(f'{name} must be a three-dimensional integer array')
	return array


def _as_bool_grid(values: np.ndarray, name: str) -> np.ndarray:
	array = np.asarray(values)
	if array.ndim != 3 or array.dtype != np.bool_:
		raise ValueError(f'{name} must be a three-dimensional boolean array')
	return array
