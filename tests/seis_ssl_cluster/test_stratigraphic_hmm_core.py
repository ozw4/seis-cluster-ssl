from __future__ import annotations

import inspect
import itertools

import numpy as np
import pytest

from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	HMMAnchorPriorSettings,
	HMMExpectedBoundariesSettings,
	HMMPathPriorSettings,
	HMMTransitionSettings,
	_decode_compacted_trace,
	_resolve_expected_boundary_count,
	_squared_euclidean_emission_costs_with_center_norms,
	build_initial_state_costs,
	build_ordered_transition_costs,
	build_terminal_state_costs,
	decode_trace_segments,
	forward_backward_state_posteriors,
	squared_euclidean_emission_costs,
	stratigraphic_hmm_settings_from_config,
	viterbi_decode_costs,
)
from seis_ssl_cluster.clustering.writer import write_json


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_squared_euclidean_emissions_match_broadcast_reference(dtype) -> None:
	features = np.array(
		[[0.0, 0.0, 0.0], [1.5, -2.0, 3.0], [1.0e8, -2.0e8, 3.0e8]],
		dtype=dtype,
	)
	centers = np.array(
		[[0.0, 0.0, 0.0], [-4.0, 5.0, -6.0], [-1.0e8, 2.0e8, -3.0e8]],
		dtype=dtype,
	)
	reference = np.sum(
		(features[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
		axis=2,
	)

	costs = squared_euclidean_emission_costs(features, centers)

	assert costs.dtype == dtype
	np.testing.assert_allclose(costs, reference, rtol=2e-6, atol=1e-12)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_squared_euclidean_emissions_preserve_nearby_large_magnitude_distance(
	dtype,
) -> None:
	features = np.array([[1.0e8, 1.0]], dtype=dtype)
	centers = np.array([[1.0e8, 0.0]], dtype=dtype)

	costs = squared_euclidean_emission_costs(features, centers)

	assert costs[0, 0] == 1.0


def test_squared_euclidean_emissions_correct_cancellation_roundoff() -> None:
	features = np.array(
		[[-0.6562850109290311, -0.6720146806586559]],
		dtype=np.float64,
	)
	centers = np.array(
		[[-0.6562850071271402, -0.6720146817592428]],
		dtype=np.float64,
	)

	costs = squared_euclidean_emission_costs(features, centers)
	reference = np.sum((features[0] - centers[0]) ** 2)

	assert costs[0, 0] == reference


def test_squared_euclidean_emissions_preserve_near_tie_winner() -> None:
	features = np.array(
		[[1.012420654296875, 0.12154181301593781]],
		dtype=np.float32,
	)
	centers = np.array(
		[
			[1.0124208927154541, 0.12154107540845871],
			[1.0124211311340332, 0.12154120206832886],
		],
		dtype=np.float32,
	)
	reference = np.sum(
		(features[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
		axis=2,
	)

	costs = squared_euclidean_emission_costs(features, centers)

	assert int(np.argmin(reference[0])) == 1
	assert int(np.argmin(costs[0])) == 1
	np.testing.assert_array_equal(
		viterbi_decode_costs(costs, np.zeros((2, 2), dtype=np.float32)),
		viterbi_decode_costs(reference, np.zeros((2, 2), dtype=np.float32)),
	)


def test_float32_emissions_preserve_legacy_complete_tie() -> None:
	features = np.array([[-0.46287498, 0.9438414]], dtype=np.float32)
	centers = np.array(
		[
			[0.3468315, 0.84368396],
			[0.34683147, 0.84368384],
		],
		dtype=np.float32,
	)
	reference = np.sum(
		(features[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
		axis=2,
	)

	costs = squared_euclidean_emission_costs(features, centers)

	assert reference[0, 0] == reference[0, 1]
	np.testing.assert_array_equal(costs, reference)
	assert int(np.argmin(costs[0])) == 0


def test_float32_emissions_preserve_nonminimum_tie_for_viterbi() -> None:
	features = np.array([[-0.99506694, 2.0499198]], dtype=np.float32)
	centers = np.array(
		[
			[-0.99506694, 2.0499198],
			[0.49659768, 0.81159914],
			[0.49659854, 0.8116002],
		],
		dtype=np.float32,
	)
	reference = np.sum(
		(features[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
		axis=2,
	)

	costs = squared_euclidean_emission_costs(features, centers)
	initial_costs = np.array([100.0, 0.0, 0.0], dtype=np.float32)
	transitions = np.zeros((3, 3), dtype=np.float32)

	assert reference[0, 1] == reference[0, 2]
	np.testing.assert_array_equal(costs, reference)
	np.testing.assert_array_equal(
		viterbi_decode_costs(
			costs,
			transitions,
			initial_state_costs=initial_costs,
		),
		viterbi_decode_costs(
			reference,
			transitions,
			initial_state_costs=initial_costs,
		),
	)


def test_emission_kernel_does_not_form_three_dimensional_pairwise_deltas() -> None:
	source = inspect.getsource(
		_squared_euclidean_emission_costs_with_center_norms,
	)

	assert 'features @ centers.T' in source
	assert 'features[:, np.newaxis, :]' not in source
	assert 'centers[np.newaxis, :, :]' not in source


def test_write_json_rejects_non_finite_floats(tmp_path) -> None:
	with pytest.raises(ValueError, match='Out of range'):
		write_json(tmp_path / 'metadata.json', {'x': float('inf')})


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


def test_viterbi_decode_costs_accepts_single_sample_and_state() -> None:
	labels = viterbi_decode_costs(
		np.array([[0.0]], dtype=np.float32),
		np.array([[0.0]], dtype=np.float32),
	)

	np.testing.assert_array_equal(labels, np.array([0], dtype=np.int32))


def test_compacted_trace_decode_preserves_transitions_and_trace_priors_at_gaps(
) -> None:
	z_indices = np.array([0, 1, 3, 4], dtype=np.int64)
	transition_emissions = np.array(
		[[0.0, 5.0], [5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]
	)
	transitions = np.array([[0.0, 0.0], [np.inf, 0.0]])
	transition_labels = _decode_compacted_trace(
		transition_emissions,
		z_indices,
		transitions,
		initial_state_costs=None,
		terminal_state_costs=None,
		expected_boundaries=None,
	)
	prior_emissions = np.array([[100.0, 0.0]] * 4)
	prior_transitions = np.array([[0.0, np.inf], [np.inf, 0.0]])
	initial_costs = np.array([0.0, 250.0])
	prior_labels = _decode_compacted_trace(
		prior_emissions,
		z_indices,
		prior_transitions,
		initial_state_costs=initial_costs,
		terminal_state_costs=None,
		expected_boundaries=None,
	)
	terminal_labels = _decode_compacted_trace(
		np.array([[0.0, 100.0]] * 4),
		z_indices,
		prior_transitions,
		initial_state_costs=None,
		terminal_state_costs=np.array([250.0, 0.0]),
		expected_boundaries=None,
	)

	np.testing.assert_array_equal(transition_labels, np.array([0, 0, 0, 1]))
	np.testing.assert_array_equal(prior_labels, np.ones(4, dtype=np.int32))
	np.testing.assert_array_equal(terminal_labels, np.zeros(4, dtype=np.int32))


def test_compacted_trace_decode_resolves_boundary_target_for_complete_trace(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	calls: list[tuple[int, object]] = []

	def record_target(
		emission_costs: np.ndarray,
		*_args: object,
		**kwargs: object,
	) -> np.ndarray:
		calls.append((emission_costs.shape[0], kwargs['expected_boundary_count']))
		return np.zeros(emission_costs.shape[0], dtype=np.int32)

	monkeypatch.setattr(
		'seis_ssl_cluster.clustering.stratigraphic_hmm.viterbi_decode_costs',
		record_target,
	)
	_decode_compacted_trace(
		np.zeros((5, 6), dtype=np.float32),
		np.array([0, 1, 3, 4, 5], dtype=np.int64),
		np.zeros((6, 6), dtype=np.float32),
		initial_state_costs=None,
		terminal_state_costs=None,
		expected_boundaries=HMMExpectedBoundariesSettings(
			enabled=True,
			target='auto_k_minus_1',
			weight=1.0,
		),
	)

	assert calls == [(5, 4)]


def test_viterbi_decode_costs_accepts_omitted_path_prior_costs() -> None:
	labels = viterbi_decode_costs(
		np.array(
			[
				[0.0, 2.0],
				[2.0, 0.0],
				[0.0, 2.0],
			],
			dtype=np.float32,
		),
		np.zeros((2, 2), dtype=np.float32),
	)

	np.testing.assert_array_equal(labels, np.array([0, 1, 0], dtype=np.int32))


def test_initial_state_costs_can_shift_first_state_shallower() -> None:
	emissions = np.array(
		[
			[0.01, 0.0],
			[0.0, 0.0],
		],
		dtype=np.float32,
	)
	transitions = np.zeros((2, 2), dtype=np.float32)

	without_prior = viterbi_decode_costs(emissions, transitions)
	with_prior = viterbi_decode_costs(
		emissions,
		transitions,
		initial_state_costs=np.array([0.0, 1.0], dtype=np.float32),
	)

	assert without_prior[0] == 1
	assert with_prior[0] == 0


def test_terminal_state_costs_can_shift_final_state_deeper() -> None:
	emissions = np.array(
		[
			[0.0, 0.0],
			[0.0, 0.01],
		],
		dtype=np.float32,
	)
	transitions = np.zeros((2, 2), dtype=np.float32)

	without_prior = viterbi_decode_costs(emissions, transitions)
	with_prior = viterbi_decode_costs(
		emissions,
		transitions,
		terminal_state_costs=np.array([1.0, 0.0], dtype=np.float32),
	)

	assert without_prior[-1] == 0
	assert with_prior[-1] == 1


def test_expected_boundary_count_prior_can_add_boundaries() -> None:
	transitions = build_ordered_transition_costs(
		3,
		HMMTransitionSettings(
			same_cost=0.0,
			advance_cost=0.1,
			jump_cost=100.0,
			reverse_cost=100.0,
			forbid_reverse=True,
			max_jump=1,
		),
	)
	emissions = np.array(
		[
			[0.0, 0.2, 0.2],
			[0.0, 0.2, 0.2],
			[0.0, 0.2, 0.2],
			[0.0, 0.2, 0.2],
			[0.0, 0.2, 0.2],
		],
		dtype=np.float32,
	)

	without_prior = viterbi_decode_costs(emissions, transitions)
	with_prior = viterbi_decode_costs(
		emissions,
		transitions,
		expected_boundary_count=2,
		boundary_count_weight=10.0,
	)

	assert _boundary_count(without_prior) == 0
	assert _boundary_count(with_prior) == 2
	assert abs(_boundary_count(with_prior) - 2) < abs(
		_boundary_count(without_prior) - 2,
	)


def test_expected_boundary_count_clamps_to_trace_length_minus_one() -> None:
	labels = viterbi_decode_costs(
		np.zeros((3, 5), dtype=np.float32),
		build_ordered_transition_costs(
			5,
			HMMTransitionSettings(
				same_cost=0.0,
				advance_cost=0.0,
				jump_cost=0.0,
				reverse_cost=100.0,
				forbid_reverse=True,
				max_jump=1,
			),
		),
		expected_boundary_count=99,
		boundary_count_weight=1.0,
	)

	assert _boundary_count(labels) == 2


def test_expected_boundary_count_target_zero_is_valid() -> None:
	assert (
		_resolve_expected_boundary_count(
			HMMExpectedBoundariesSettings(enabled=True, target=0, weight=0.1),
			k=3,
			valid_trace_length=5,
		)
		== 0
	)


def test_disabled_expected_boundary_count_prior_uses_fast_path_output() -> None:
	emissions = np.array(
		[
			[0.0, 3.0, 6.0],
			[3.0, 0.0, 6.0],
			[0.0, 3.0, 6.0],
			[6.0, 3.0, 0.0],
		],
		dtype=np.float32,
	)
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

	fast = viterbi_decode_costs(emissions, transitions)
	zero_weight = viterbi_decode_costs(
		emissions,
		transitions,
		expected_boundary_count=2,
		boundary_count_weight=0.0,
	)
	no_target = viterbi_decode_costs(
		emissions,
		transitions,
		expected_boundary_count=None,
		boundary_count_weight=10.0,
	)

	np.testing.assert_array_equal(zero_weight, fast)
	np.testing.assert_array_equal(no_target, fast)


def test_disabled_path_prior_builders_return_zero_vectors() -> None:
	settings = HMMPathPriorSettings(
		enabled=False,
		initial_state=HMMAnchorPriorSettings(mode='shallow_anchor', weight=0.5),
		terminal_state=HMMAnchorPriorSettings(mode='deep_anchor', weight=0.5),
		expected_boundaries=HMMExpectedBoundariesSettings(
			enabled=False,
			target='auto_k_minus_1',
			weight=0.0,
		),
	)

	np.testing.assert_array_equal(
		build_initial_state_costs(3, settings),
		np.zeros(3, dtype=np.float32),
	)
	np.testing.assert_array_equal(
		build_terminal_state_costs(3, settings),
		np.zeros(3, dtype=np.float32),
	)
	np.testing.assert_array_equal(
		build_initial_state_costs(1, settings),
		np.zeros(1, dtype=np.float32),
	)


@pytest.mark.parametrize(
	(
		'emissions',
		'transitions',
		'initial_costs',
		'terminal_costs',
		'expected_boundary_count',
		'boundary_count_weight',
	),
	[
		(
			np.array([[0.0, 0.0]]),
			np.zeros((2, 2)),
			np.array([0.2, 0.0]),
			np.array([0.0, 0.3]),
			None,
			0.0,
		),
		(
			np.array([[0.0, 0.4], [0.2, 0.0], [0.3, 0.1]]),
			np.array([[0.0, 0.2], [np.inf, 0.0]]),
			np.array([0.1, 0.2]),
			np.array([0.3, 0.0]),
			None,
			0.0,
		),
		(
			np.array(
				[
					[0.0, 0.0, 1.0],
					[0.1, 0.0, 0.1],
					[0.2, 0.1, 0.0],
					[0.3, 0.2, 0.0],
				],
			),
			np.array(
				[
					[0.0, 0.2, np.inf],
					[np.inf, 0.0, 0.2],
					[np.inf, np.inf, 0.0],
				],
			),
			np.array([0.0, 0.1, 0.2]),
			np.array([0.2, 0.1, 0.0]),
			2,
			0.7,
		),
	],
)
def test_forward_backward_state_posteriors_matches_path_enumeration(  # noqa: PLR0913, PLR0917
	emissions: np.ndarray,
	transitions: np.ndarray,
	initial_costs: np.ndarray,
	terminal_costs: np.ndarray,
	expected_boundary_count: int | None,
	boundary_count_weight: float,
) -> None:
	result = forward_backward_state_posteriors(
		emissions,
		transitions,
		initial_state_costs=initial_costs,
		terminal_state_costs=terminal_costs,
		expected_boundary_count=expected_boundary_count,
		boundary_count_weight=boundary_count_weight,
	)
	reference_posterior, reference_log_partition = _enumerated_state_posterior(
		emissions,
		transitions,
		initial_costs=initial_costs,
		terminal_costs=terminal_costs,
		expected_boundary_count=expected_boundary_count,
		boundary_count_weight=boundary_count_weight,
	)

	np.testing.assert_allclose(result.posterior, reference_posterior)
	assert result.log_partition == pytest.approx(reference_log_partition)
	np.testing.assert_allclose(
		result.expected_normalized_order,
		reference_posterior @ np.linspace(0.0, 1.0, emissions.shape[1]),
	)


def test_forward_backward_state_posteriors_respects_unique_finite_path() -> None:
	result = forward_backward_state_posteriors(
		np.array([[0.0, np.inf], [0.0, np.inf], [0.0, np.inf]]),
		np.array([[0.0, np.inf], [np.inf, 0.0]]),
	)

	np.testing.assert_array_equal(
		result.posterior,
		np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
	)


def test_forward_backward_state_posteriors_can_differ_from_viterbi_map() -> None:
	emissions = np.zeros((2, 2))
	transitions = np.array([[0.0, 0.0], [1.0, 0.0]])

	viterbi = viterbi_decode_costs(emissions, transitions)
	posterior_argmax = np.argmax(
		forward_backward_state_posteriors(emissions, transitions).posterior,
		axis=1,
	)

	np.testing.assert_array_equal(viterbi, np.array([0, 0], dtype=np.int32))
	np.testing.assert_array_equal(posterior_argmax, np.array([0, 1]))


def test_forward_backward_cost_fixture_has_the_existing_viterbi_map() -> None:
	emissions = np.array(
		[
			[0.0, 3.0, 6.0],
			[3.0, 0.0, 3.0],
			[6.0, 3.0, 0.0],
			[6.0, 3.0, 0.0],
		],
	)
	transitions = build_ordered_transition_costs(
		3,
		HMMTransitionSettings(
			same_cost=0.0,
			advance_cost=0.1,
			jump_cost=5.0,
			reverse_cost=100.0,
			forbid_reverse=True,
			max_jump=1,
		),
	)

	forward_backward_state_posteriors(
		emissions,
		transitions,
		expected_boundary_count=2,
		boundary_count_weight=1.0,
	)
	viterbi = viterbi_decode_costs(
		emissions,
		transitions,
		expected_boundary_count=2,
		boundary_count_weight=1.0,
	)

	np.testing.assert_array_equal(
		viterbi,
		_enumerated_minimum_cost_path(
			emissions,
			transitions,
			expected_boundary_count=2,
			boundary_count_weight=1.0,
		),
	)


def test_forward_backward_state_posteriors_preserves_ordered_expectation() -> None:
	result = forward_backward_state_posteriors(
		np.zeros((4, 3)),
		build_ordered_transition_costs(
			3,
			HMMTransitionSettings(
				same_cost=0.0,
				advance_cost=0.0,
				jump_cost=0.0,
				reverse_cost=1.0,
				forbid_reverse=True,
				max_jump=1,
			),
		),
	)

	assert np.all(np.diff(result.expected_normalized_order) >= -1.0e-12)


@pytest.mark.parametrize(
	('kwargs', 'message'),
	[
		({'cost_temperature': 0.0}, 'cost_temperature'),
		({'cost_temperature': np.inf}, 'cost_temperature'),
		({'expected_boundary_count': -1}, 'expected_boundary_count'),
		({'boundary_count_weight': np.nan}, 'boundary_count_weight'),
	],
)
def test_forward_backward_state_posteriors_rejects_invalid_parameters(
	kwargs: dict[str, object],
	message: str,
) -> None:
	with pytest.raises(ValueError, match=message):
		forward_backward_state_posteriors(
			np.zeros((2, 2)),
			np.zeros((2, 2)),
			**kwargs,
		)


def test_forward_backward_state_posteriors_rejects_invalid_costs_and_no_path() -> None:
	with pytest.raises(ValueError, match='non-empty'):
		forward_backward_state_posteriors(
			np.empty((0, 2)),
			np.zeros((2, 2)),
		)
	with pytest.raises(ValueError, match='shape'):
		forward_backward_state_posteriors(
			np.zeros((2, 2)),
			np.zeros((3, 3)),
		)
	with pytest.raises(ValueError, match='NaN'):
		forward_backward_state_posteriors(
			np.array([[0.0, np.nan]]),
			np.zeros((2, 2)),
		)
	with pytest.raises(ValueError, match='finite path'):
		forward_backward_state_posteriors(
			np.full((2, 2), np.inf),
			np.zeros((2, 2)),
		)


def test_forward_backward_state_posteriors_uses_bounded_dynamic_programming() -> None:
	source = inspect.getsource(forward_backward_state_posteriors)
	boundary_source = inspect.getsource(
		forward_backward_state_posteriors.__globals__[
			'_state_posteriors_with_boundary_count'
		],
	)

	assert 'itertools' not in source
	assert 'itertools' not in boundary_source
	assert 'k, k, max_boundaries' not in boundary_source


def test_viterbi_decode_costs_rejects_invalid_path_prior_cost_shape() -> None:
	with pytest.raises(ValueError, match='initial_state_costs'):
		viterbi_decode_costs(
			np.zeros((2, 2), dtype=np.float32),
			np.zeros((2, 2), dtype=np.float32),
			initial_state_costs=np.zeros(3, dtype=np.float32),
		)


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

	np.testing.assert_array_equal(labels, np.array([0, 0, -1, 0, 1], dtype=np.int32))
	np.testing.assert_array_equal(labels[~valid_mask], np.array([-1], dtype=np.int32))


def test_decode_trace_segments_applies_initial_prior_once_across_gap() -> None:
	emissions = np.array(
		[
			[100.0, 0.0],
			[100.0, 0.0],
			[9.0, 9.0],
			[100.0, 0.0],
			[100.0, 0.0],
		],
		dtype=np.float32,
	)
	valid_mask = np.array([True, True, False, True, True])

	labels = decode_trace_segments(
		emissions,
		valid_mask,
		np.array([[0.0, np.inf], [np.inf, 0.0]]),
		initial_state_costs=np.array([0.0, 250.0]),
	)

	np.testing.assert_array_equal(labels, np.array([1, 1, -1, 1, 1]))


def test_decode_trace_segments_returns_invalid_labels_when_no_valid_tokens() -> None:
	labels = decode_trace_segments(
		np.zeros((3, 2), dtype=np.float32),
		np.zeros(3, dtype=np.bool_),
		np.zeros((2, 2), dtype=np.float32),
	)

	np.testing.assert_array_equal(labels, np.full(3, -1, dtype=np.int32))


def test_stratigraphic_hmm_settings_from_config_parses_resolved_config() -> None:
	settings = stratigraphic_hmm_settings_from_config(_hmm_settings_config())

	assert settings.iterations == 3
	assert settings.z_axis == 2
	assert settings.z_direction == 'increasing_downward'
	assert settings.transition.advance_cost == 0.25
	assert settings.transition.max_jump is None
	assert settings.init_order_by == 'mean_z'
	assert settings.empty_cluster_policy == 'keep_previous'
	assert settings.edge_margin_tokens == (0, 0, 0)
	assert settings.path_prior == HMMPathPriorSettings(
		enabled=False,
		initial_state=HMMAnchorPriorSettings(mode='none', weight=0.0),
		terminal_state=HMMAnchorPriorSettings(mode='none', weight=0.0),
		expected_boundaries=HMMExpectedBoundariesSettings(
			enabled=False,
			target='auto_k_minus_1',
			weight=0.0,
		),
	)


def test_stratigraphic_hmm_settings_from_config_preserves_path_prior() -> None:
	cfg = _hmm_settings_config()
	hmm = cfg['clustering']['stratigraphic_hmm']
	hmm['edge_margin_tokens'] = [8, 8, 0]
	hmm['path_prior'] = {
		'enabled': True,
		'initial_state': {'mode': 'shallow_anchor', 'weight': 0.5},
		'terminal_state': {'mode': 'deep_anchor', 'weight': 0.25},
		'expected_boundaries': {
			'enabled': True,
			'target': 3,
			'weight': 0.1,
		},
	}

	settings = stratigraphic_hmm_settings_from_config(cfg)

	assert settings.edge_margin_tokens == (8, 8, 0)
	assert settings.path_prior == HMMPathPriorSettings(
		enabled=True,
		initial_state=HMMAnchorPriorSettings(mode='shallow_anchor', weight=0.5),
		terminal_state=HMMAnchorPriorSettings(mode='deep_anchor', weight=0.25),
		expected_boundaries=HMMExpectedBoundariesSettings(
			enabled=True,
			target=3,
			weight=0.1,
		),
	)


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


def _boundary_count(labels: np.ndarray) -> int:
	return int(np.count_nonzero(np.diff(labels) > 0))


def _enumerated_state_posterior(  # noqa: PLR0913
	emissions: np.ndarray,
	transitions: np.ndarray,
	*,
	initial_costs: np.ndarray,
	terminal_costs: np.ndarray,
	expected_boundary_count: int | None,
	boundary_count_weight: float,
) -> tuple[np.ndarray, float]:
	t_count, k = emissions.shape
	max_boundaries = min(k - 1, t_count - 1)
	target = (
		None
		if expected_boundary_count is None
		else min(expected_boundary_count, max_boundaries)
	)
	paths: list[tuple[int, ...]] = []
	log_weights: list[float] = []
	for path in itertools.product(range(k), repeat=t_count):
		cost = float(emissions[np.arange(t_count), path].sum())
		cost += float(initial_costs[path[0]] + terminal_costs[path[-1]])
		for previous, next_state in itertools.pairwise(path):
			transition = transitions[previous, next_state]
			if not np.isfinite(transition):
				break
			cost += float(transition)
		else:
			boundaries = sum(
				next_state > previous
				for previous, next_state in itertools.pairwise(path)
			)
			if boundaries > max_boundaries:
				continue
			if target is not None and boundary_count_weight > 0.0:
				cost += boundary_count_weight * (boundaries - target) ** 2
			paths.append(path)
			log_weights.append(-cost)
	log_partition = float(np.logaddexp.reduce(log_weights))
	posterior = np.zeros((t_count, k))
	for path, log_weight in zip(paths, log_weights, strict=True):
		posterior[np.arange(t_count), path] += np.exp(log_weight - log_partition)
	return posterior, log_partition


def _enumerated_minimum_cost_path(
	emissions: np.ndarray,
	transitions: np.ndarray,
	*,
	expected_boundary_count: int,
	boundary_count_weight: float,
) -> np.ndarray:
	t_count, k = emissions.shape
	max_boundaries = min(k - 1, t_count - 1)
	target = min(expected_boundary_count, max_boundaries)
	best_path: tuple[int, ...] | None = None
	best_cost = np.inf
	for path in itertools.product(range(k), repeat=t_count):
		cost = float(emissions[np.arange(t_count), path].sum())
		for previous, next_state in itertools.pairwise(path):
			transition = transitions[previous, next_state]
			if not np.isfinite(transition):
				break
			cost += float(transition)
		else:
			boundaries = sum(
				next_state > previous
				for previous, next_state in itertools.pairwise(path)
			)
			if boundaries > max_boundaries:
				continue
			cost += boundary_count_weight * (boundaries - target) ** 2
			if cost < best_cost:
				best_path = path
				best_cost = cost
	if best_path is None:
		raise AssertionError('test fixture must have a finite path')
	return np.asarray(best_path, dtype=np.int32)


def _hmm_settings_config() -> dict[str, object]:
	return {
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
	}
