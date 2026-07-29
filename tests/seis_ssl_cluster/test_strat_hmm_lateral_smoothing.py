from __future__ import annotations

import inspect

import numpy as np
import pytest

from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	HMMTransitionSettings,
	build_ordered_transition_costs,
	viterbi_decode_costs,
)
from seis_ssl_cluster.stratigraphy.lateral_smoothing import (
	apply_lateral_message_to_emission_costs,
	cosine_rbf_affinities,
	enumerate_xy_four_neighbors,
	median_scale_with_floor,
	normalized_lateral_message,
	redecode_ordered_lateral_trace,
	smooth_and_redecode_ordered_trace,
)


def test_xy_four_neighbors_are_fixed_and_exclude_z_and_diagonals() -> None:
	assert enumerate_xy_four_neighbors((1, 1, 2), (3, 3, 5)) == (
		(0, 1, 2),
		(2, 1, 2),
		(1, 0, 2),
		(1, 2, 2),
	)
	assert enumerate_xy_four_neighbors((0, 0, 0), (2, 2, 2)) == (
		(1, 0, 0),
		(0, 1, 0),
	)


def test_cosine_affinity_excludes_invalid_edges_and_prefers_similarity() -> None:
	center = np.array([[1.0, 0.0], [1.0, 0.0]])
	neighbors = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]])
	distance, affinity = cosine_rbf_affinities(
		center,
		neighbors,
		np.array([True, True]),
		np.array([[True, True], [True, False]]),
		affinity_scale=1.0,
	)
	assert affinity[0, 0] > affinity[0, 1]
	assert distance[1, 1] == 0.0
	assert affinity[1, 1] == 0.0


def test_cosine_affinity_rejects_zero_norm_valid_embedding_without_edge() -> None:
	with pytest.raises(ValueError, match='nonzero norm'):
		cosine_rbf_affinities(
			np.array([[0.0, 0.0]]),
			np.empty((0, 1, 2)),
			np.array([True]),
			np.empty((0, 1), dtype=bool),
			affinity_scale=1.0,
		)


def test_lateral_message_is_weighted_normalized_posterior_average() -> None:
	posterior = np.array(
		[
			[[1.0, 0.0], [0.25, 0.75]],
			[[0.0, 1.0], [0.0, 0.0]],
			[[0.5, 0.5], [0.75, 0.25]],
			[[0.25, 0.75], [0.0, 0.0]],
		]
	)
	result = normalized_lateral_message(
		posterior,
		np.array([[1.0, 2.0], [2.0, 0.0], [3.0, 4.0], [4.0, 0.0]]),
		np.array([True, True]),
		np.array([[True, True], [True, False], [True, True], [True, False]]),
	)
	np.testing.assert_allclose(result.message[0], [3.5 / 10.0, 6.5 / 10.0])
	np.testing.assert_allclose(result.message[1], [3.5 / 6.0, 2.5 / 6.0])
	np.testing.assert_allclose(result.message.sum(axis=1), 1.0)
	np.testing.assert_array_equal(result.neighbor_count, [4, 2])


def test_no_neighbor_and_beta_zero_preserve_source_costs_and_labels() -> None:
	kwargs = _trace_inputs()
	no_neighbors = smooth_and_redecode_ordered_trace(
		**{
			**kwargs,
			'neighbor_embeddings': np.empty((0, 3, 2)),
			'neighbor_posterior': np.empty((0, 3, 2)),
			'neighbor_valid_masks': np.empty((0, 3), dtype=bool),
		},
		affinity_scale=1.0,
		emission_gap_scale=2.0,
		pairwise_strength_ratio=1.0,
	)
	assert np.array_equal(no_neighbors.diagnostics.message, np.zeros((3, 2)))
	assert np.array_equal(no_neighbors.emission_costs, kwargs['emission_costs'])

	zero_beta = smooth_and_redecode_ordered_trace(
		**kwargs,
		affinity_scale=1.0,
		emission_gap_scale=2.0,
		pairwise_strength_ratio=0.0,
	)
	assert np.array_equal(zero_beta.emission_costs, kwargs['emission_costs'])
	np.testing.assert_array_equal(
		zero_beta.labels,
		redecode_ordered_lateral_trace(
			kwargs['emission_costs'],
			kwargs['center_valid_mask'],
			kwargs['transition_costs'],
		),
	)


def test_message_lowers_matching_state_cost_without_mutating_inputs() -> None:
	costs = np.array([[2.0, 2.0]], dtype=np.float64)
	message = np.array([[0.1, 0.9]], dtype=np.float64)
	result = apply_lateral_message_to_emission_costs(
		costs,
		message,
		np.array([True]),
		np.array([1.0]),
		emission_gap_scale=2.0,
		pairwise_strength_ratio=1.0,
	)
	np.testing.assert_allclose(result.emission_costs, [[1.8, 0.2]])
	np.testing.assert_array_equal(costs, [[2.0, 2.0]])


def test_redecode_matches_existing_boundary_prior_and_compacts_invalid_gap() -> None:
	transitions = build_ordered_transition_costs(
		3,
		HMMTransitionSettings(
			same_cost=0.0,
			advance_cost=0.1,
			jump_cost=2.0,
			reverse_cost=9.0,
			forbid_reverse=True,
			max_jump=None,
		),
	)
	costs = np.array([[0.0, 3.0, 7.0], [4.0, 0.0, 4.0], [8.0, 4.0, 0.0]])
	valid = np.array([True, False, True])
	labels = redecode_ordered_lateral_trace(
		costs,
		valid,
		transitions,
		expected_boundary_count=1,
		boundary_count_weight=3.0,
	)
	expected = viterbi_decode_costs(
		costs[valid],
		transitions,
		expected_boundary_count=1,
		boundary_count_weight=3.0,
	)
	np.testing.assert_array_equal(labels[valid], expected)
	assert labels[1] == -1
	assert np.all(np.diff(labels[valid]) >= 0)


def test_redecode_rejects_trace_with_no_finite_path() -> None:
	with pytest.raises(ValueError, match='no finite path'):
		redecode_ordered_lateral_trace(
			np.zeros((2, 2)),
			np.array([True, True]),
			np.full((2, 2), np.inf),
		)


@pytest.mark.parametrize(
	('center', 'neighbors', 'posterior', 'valid', 'neighbor_valid', 'costs'),
	[
		(
			np.empty((0, 2)),
			np.empty((0, 0, 2)),
			np.empty((0, 0, 2)),
			np.empty(0, dtype=bool),
			np.empty((0, 0), dtype=bool),
			np.empty((0, 2)),
		),
		(
			np.array([[1.0, 0.0]]),
			np.array([[[1.0, 0.0]]]),
			np.array([[[1.0, 0.0]]]),
			np.array([True]),
			np.array([[True]]),
			np.array([[1.0, 2.0]]),
		),
	],
)
def test_empty_and_single_token_traces(  # noqa: PLR0913
	center, neighbors, posterior, valid, neighbor_valid, costs
) -> None:
	result = smooth_and_redecode_ordered_trace(
		center,
		neighbors,
		posterior,
		valid,
		neighbor_valid,
		costs,
		np.zeros((2, 2)),
		affinity_scale=1.0,
		emission_gap_scale=1.0,
		pairwise_strength_ratio=1.0,
	)
	assert result.labels.shape == (costs.shape[0],)


@pytest.mark.parametrize(
	'change',
	[
		lambda data: data.update(center_embedding=np.array([[0.0, 0.0]] * 3)),
		lambda data: data.update(affinity_scale=0.0),
		lambda data: data.update(pairwise_strength_ratio=-1.0),
		lambda data: data.update(emission_costs=np.array([[np.nan, 0.0]] * 3)),
		lambda data: data.update(neighbor_posterior=np.array([[[2.0, 0.0]] * 3])),
	],
)
def test_invalid_inputs_are_rejected(change) -> None:
	data = _trace_inputs()
	data.update(affinity_scale=1.0, emission_gap_scale=1.0, pairwise_strength_ratio=1.0)
	change(data)
	with pytest.raises(ValueError, match=r'.+'):
		smooth_and_redecode_ordered_trace(**data)


def test_scale_floor_and_no_full_grid_or_iteration_structure() -> None:
	assert median_scale_with_floor(np.array([])) == 1.0e-6
	assert median_scale_with_floor(np.array([0.0, 3.0, 9.0])) == 3.0
	source = inspect.getsource(smooth_and_redecode_ordered_trace)
	assert 'for ' not in source


def _trace_inputs() -> dict[str, np.ndarray]:
	return {
		'center_embedding': np.array([[1.0, 0.0]] * 3),
		'neighbor_embeddings': np.array([[[1.0, 0.0]] * 3]),
		'neighbor_posterior': np.array([[[0.0, 1.0]] * 3]),
		'center_valid_mask': np.array([True, True, True]),
		'neighbor_valid_masks': np.array([[True, True, True]]),
		'emission_costs': np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
		'transition_costs': np.zeros((2, 2)),
	}
