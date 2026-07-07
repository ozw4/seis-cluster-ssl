from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	HMMTransitionSettings,
	build_ordered_transition_costs,
	contiguous_true_segments,
	decode_trace_segments,
	stratigraphic_hmm_settings_from_config,
	viterbi_decode_costs,
)


def test_build_ordered_transition_costs_forbids_reverse_transitions() -> None:
	costs = build_ordered_transition_costs(
		3,
		HMMTransitionSettings(
			same_cost=0.0,
			advance_cost=0.5,
			jump_cost=2.0,
			reverse_cost=7.0,
			forbid_reverse=True,
			max_jump=None,
		),
	)

	assert costs.dtype == np.float32
	assert costs.shape == (3, 3)
	assert np.isinf(costs[1, 0])
	assert np.isinf(costs[2, 0])
	assert costs[0, 0] == 0.0
	assert costs[0, 1] == 0.5


def test_build_ordered_transition_costs_applies_jump_penalty() -> None:
	costs = build_ordered_transition_costs(
		4,
		HMMTransitionSettings(
			same_cost=0.0,
			advance_cost=1.0,
			jump_cost=3.0,
			reverse_cost=10.0,
			forbid_reverse=False,
			max_jump=2,
		),
	)

	assert costs[0, 2] == 4.0
	assert costs[3, 1] == 13.0
	assert np.isinf(costs[0, 3])
	assert costs[2, 2] == 0.0


def test_viterbi_decode_costs_is_monotone_when_reverse_is_forbidden() -> None:
	transitions = build_ordered_transition_costs(
		3,
		HMMTransitionSettings(
			same_cost=0.0,
			advance_cost=0.1,
			jump_cost=5.0,
			reverse_cost=100.0,
			forbid_reverse=True,
			max_jump=None,
		),
	)
	emissions = np.array(
		[
			[0.0, 4.0, 8.0],
			[4.0, 0.0, 8.0],
			[0.0, 3.0, 8.0],
			[8.0, 4.0, 0.0],
		],
		dtype=np.float32,
	)

	labels = viterbi_decode_costs(emissions, transitions)

	np.testing.assert_array_equal(labels, np.array([0, 1, 1, 2], dtype=np.int32))


def test_viterbi_decode_costs_ties_choose_smaller_previous_state() -> None:
	labels = viterbi_decode_costs(
		np.array(
			[
				[0.0, 0.0, 0.0],
				[1.0, 0.0, 1.0],
			],
			dtype=np.float32,
		),
		np.zeros((3, 3), dtype=np.float32),
	)

	np.testing.assert_array_equal(labels, np.array([0, 1], dtype=np.int32))


def test_decode_trace_segments_preserves_invalid_positions_and_decodes_gaps() -> None:
	transitions = build_ordered_transition_costs(
		2,
		HMMTransitionSettings(
			same_cost=0.0,
			advance_cost=0.0,
			jump_cost=1.0,
			reverse_cost=100.0,
			forbid_reverse=True,
			max_jump=None,
		),
	)
	emissions = np.array(
		[
			[0.0, 5.0],
			[5.0, 0.0],
			[9.0, 9.0],
			[0.0, 5.0],
			[5.0, 0.0],
		],
		dtype=np.float32,
	)
	valid_mask = np.array([True, True, False, True, True])

	labels = decode_trace_segments(emissions, valid_mask, transitions)

	np.testing.assert_array_equal(labels, np.array([0, 1, -1, 0, 1], dtype=np.int32))


def test_contiguous_true_segments_returns_valid_spans() -> None:
	segments = contiguous_true_segments(
		np.array([False, True, True, False, True], dtype=np.bool_),
	)

	assert segments == (slice(1, 3), slice(4, 5))


def test_stratigraphic_hmm_settings_from_config_parses_resolved_config() -> None:
	settings = stratigraphic_hmm_settings_from_config(
		{
			'clustering': {
				'stratigraphic_hmm': {
					'iterations': 3,
					'z_axis': 2,
					'z_direction': 'increasing_downward',
					'transition': {
						'same_cost': 0.0,
						'advance_cost': 0.25,
						'jump_cost': 1.0,
						'reverse_cost': 1000000.0,
						'forbid_reverse': True,
						'max_jump': None,
					},
					'init': {'order_by': 'mean_z'},
					'update': {'empty_cluster_policy': 'keep_previous'},
				},
			},
		},
	)

	assert settings.iterations == 3
	assert settings.z_axis == 2
	assert settings.z_direction == 'increasing_downward'
	assert settings.transition.advance_cost == 0.25
	assert settings.transition.max_jump is None
	assert settings.init_order_by == 'mean_z'
	assert settings.empty_cluster_policy == 'keep_previous'


@pytest.mark.parametrize(
	('emissions', 'transitions', 'message'),
	[
		(np.ones((2, 2, 1), dtype=np.float32), np.zeros((2, 2)), '2D'),
		(np.ones((2, 2), dtype=np.float32), np.zeros((3, 3)), 'shape'),
		(np.array([[0.0, np.nan]], dtype=np.float32), np.zeros((2, 2)), 'finite'),
	],
)
def test_viterbi_decode_costs_rejects_invalid_inputs(
	emissions: np.ndarray,
	transitions: np.ndarray,
	message: str,
) -> None:
	with pytest.raises(ValueError, match=message):
		viterbi_decode_costs(emissions, transitions)


def test_viterbi_decode_costs_rejects_no_finite_path() -> None:
	with pytest.raises(ValueError, match='finite path'):
		viterbi_decode_costs(
			np.zeros((2, 2), dtype=np.float32),
			np.full((2, 2), np.inf, dtype=np.float32),
		)


def test_decode_trace_segments_rejects_invalid_mask_shape() -> None:
	with pytest.raises(ValueError, match='valid_mask'):
		decode_trace_segments(
			np.zeros((2, 2), dtype=np.float32),
			np.ones((2, 1), dtype=np.bool_),
			np.zeros((2, 2), dtype=np.float32),
		)
