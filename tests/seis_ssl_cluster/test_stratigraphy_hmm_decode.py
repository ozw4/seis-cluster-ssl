from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.clustering.stratigraphic_hmm import HMMTransitionSettings
from seis_ssl_cluster.stratigraphy import (
	decode_ordered_logits_survey,
	emission_costs_from_logits,
	validate_pseudo_target_arrays,
)


def test_emission_costs_prefer_larger_logits_and_are_finite() -> None:
	logits = np.array([[[[0.0, 3.0, -2.0]]]], dtype=np.float32)

	costs = emission_costs_from_logits(logits)

	assert costs.shape == logits.shape
	assert np.all(np.isfinite(costs))
	assert costs[0, 0, 0, 1] < costs[0, 0, 0, 0]
	assert costs[0, 0, 0, 0] < costs[0, 0, 0, 2]


def test_emission_costs_do_not_clip_low_probability_tail() -> None:
	logits = np.array([[[[0.0, -100.0, -1000.0]]]], dtype=np.float32)

	costs = emission_costs_from_logits(logits)

	assert costs[0, 0, 0, 2] - costs[0, 0, 0, 1] == pytest.approx(900.0)


def test_simple_synthetic_logits_decode_to_ordered_labels_along_z() -> None:
	target = decode_ordered_logits_survey(
		_logits_for_labels([0, 1, 2], k=3),
		np.ones((1, 1, 3), dtype=np.bool_),
		transition=_transition(forbid_reverse=True),
	)

	np.testing.assert_array_equal(
		target.labels,
		np.array([[[0, 1, 2]]], dtype=np.int32),
	)
	validate_pseudo_target_arrays(
		target.labels,
		target.confidence,
		target.valid_tokens,
		k=3,
	)


def test_reverse_transitions_are_forbidden_when_configured() -> None:
	logits = np.array([[[[0.0, 10.0], [9.0, 0.0]]]], dtype=np.float32)

	target = decode_ordered_logits_survey(
		logits,
		np.ones((1, 1, 2), dtype=np.bool_),
		transition=_transition(k=2, forbid_reverse=True),
	)

	np.testing.assert_array_equal(
		target.labels,
		np.array([[[1, 1]]], dtype=np.int32),
	)


def test_edge_margins_exclude_expected_token_borders() -> None:
	logits = np.zeros((3, 3, 4, 2), dtype=np.float32)

	target = decode_ordered_logits_survey(
		logits,
		np.ones((3, 3, 4), dtype=np.bool_),
		transition=_transition(k=2),
		edge_margin_tokens=(1, 1, 1),
	)

	expected_valid = np.zeros((3, 3, 4), dtype=np.bool_)
	expected_valid[1, 1, 1:3] = True
	np.testing.assert_array_equal(target.valid_tokens, expected_valid)
	assert np.all(target.labels[~expected_valid] == -1)
	assert target.metadata['edge_margin_excluded_valid_token_count'] == 34


def test_invalid_trace_gaps_are_preserved_while_valid_segments_decode() -> None:
	valid = np.array([[[True, False, True, True]]], dtype=np.bool_)

	target = decode_ordered_logits_survey(
		_logits_for_labels([0, 0, 1, 2], k=3),
		valid,
		transition=_transition(forbid_reverse=True),
	)

	np.testing.assert_array_equal(
		target.labels,
		np.array([[[0, -1, 1, 2]]], dtype=np.int32),
	)
	np.testing.assert_array_equal(target.valid_tokens, valid)
	assert target.confidence[0, 0, 1] == 0.0


def test_confidence_equals_assigned_softmax_probability_for_decoded_tokens() -> None:
	logits = np.array([[[[2.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)

	target = decode_ordered_logits_survey(
		logits,
		np.ones((1, 1, 2), dtype=np.bool_),
		transition=_transition(k=2, forbid_reverse=True),
	)

	probabilities = _softmax(logits)
	for z_index in range(2):
		label = target.labels[0, 0, z_index]
		assert target.confidence[0, 0, z_index] == pytest.approx(
			probabilities[0, 0, z_index, label],
		)


@pytest.mark.parametrize(
	('logits', 'valid_tokens', 'error'),
	[
		(np.zeros((1, 1, 2)), np.ones((1, 1, 2), dtype=np.bool_), 'logits must be 4D'),
		(
			np.zeros((1, 1, 2, 0)),
			np.ones((1, 1, 2), dtype=np.bool_),
			'K >= 1',
		),
		(
			np.zeros((1, 1, 2, 2)),
			np.ones((1, 1, 3), dtype=np.bool_),
			'shape must match',
		),
		(
			np.array([[[[np.nan, 0.0]]]]),
			np.ones((1, 1, 1), dtype=np.bool_),
			'finite',
		),
		(
			np.zeros((1, 1, 2, 2)),
			np.ones((1, 1, 2), dtype=np.int32),
			'dtype must be bool',
		),
	],
)
def test_input_validation_rejects_malformed_shapes_and_non_finite_logits(
	logits: np.ndarray,
	valid_tokens: np.ndarray,
	error: str,
) -> None:
	with pytest.raises((TypeError, ValueError), match=error):
		decode_ordered_logits_survey(
			logits,
			valid_tokens,
			transition=_transition(k=2),
		)


def test_input_validation_rejects_bad_margin_and_empty_effective_mask() -> None:
	with pytest.raises(ValueError, match='non-negative'):
		decode_ordered_logits_survey(
			np.zeros((1, 1, 2, 2), dtype=np.float32),
			np.ones((1, 1, 2), dtype=np.bool_),
			transition=_transition(k=2),
			edge_margin_tokens=(-1, 0, 0),
		)

	with pytest.raises(ValueError, match='effective valid_tokens'):
		decode_ordered_logits_survey(
			np.zeros((1, 1, 2, 2), dtype=np.float32),
			np.zeros((1, 1, 2), dtype=np.bool_),
			transition=_transition(k=2),
		)


def _transition(
	k: int = 3,
	*,
	forbid_reverse: bool = True,
) -> HMMTransitionSettings:
	return HMMTransitionSettings(
		same_cost=0.0,
		advance_cost=0.0,
		jump_cost=10.0,
		reverse_cost=10.0,
		forbid_reverse=forbid_reverse,
		max_jump=1 if k > 1 else None,
	)


def _logits_for_labels(labels: list[int], *, k: int) -> np.ndarray:
	logits = np.full((1, 1, len(labels), k), -8.0, dtype=np.float32)
	for z_index, label in enumerate(labels):
		logits[0, 0, z_index, label] = 8.0
	return logits


def _softmax(logits: np.ndarray) -> np.ndarray:
	shifted = logits - np.max(logits, axis=-1, keepdims=True)
	exp_logits = np.exp(shifted)
	return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
