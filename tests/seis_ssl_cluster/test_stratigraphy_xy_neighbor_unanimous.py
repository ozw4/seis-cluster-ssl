"""Numerical invariants for the conservative unanimous XY policy."""

from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus import (
	XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_POLICY,
	XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_POLICY,
	XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_SEMANTICS,
	smooth_xy_neighbor_consensus_hard_labels,
	smooth_xy_neighbor_unanimous_hard_labels,
)


def test_three_of_four_changes_but_unanimous_leaves_ambiguous_center() -> None:
	labels, valid = _three_level_grid()
	_set_neighbors(labels, 1, 1, 1, (3, 3, 3, 1))

	consensus = smooth_xy_neighbor_consensus_hard_labels(labels, valid)
	unanimous = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

	assert consensus.labels[1, 1, 1] == 3
	assert consensus.diagnostics.changed_mask[1, 1, 1]
	assert unanimous.labels[1, 1, 1] == 2
	assert not unanimous.diagnostics.consensus_mask[1, 1, 1]
	assert not unanimous.diagnostics.changed_mask[1, 1, 1]


def test_four_unanimous_neighbours_change_internal_center() -> None:
	labels, valid = _three_level_grid()
	_set_neighbors(labels, 1, 1, 1, (3, 3, 3, 3))

	result = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

	assert result.labels[1, 1, 1] == 3
	assert result.diagnostics.neighbor_count[1, 1, 1] == 4
	assert result.diagnostics.consensus_mask[1, 1, 1]
	assert result.diagnostics.changed_mask[1, 1, 1]


@pytest.mark.parametrize(
	('neighbor_labels', 'neighbor_valid', 'expected_label'),
	[
		((3, 3, 3, 9), (True, True, True, False), 3),
		((3, 3, 1, 9), (True, True, True, False), 2),
		((3, 3, 9, 9), (True, True, False, False), 2),
	],
)
def test_three_neighbour_rule_is_unanimous_and_two_neighbours_never_change(
	neighbor_labels: tuple[int, int, int, int],
	neighbor_valid: tuple[bool, bool, bool, bool],
	expected_label: int,
) -> None:
	labels, valid = _three_level_grid()
	_set_neighbors(labels, 1, 1, 1, neighbor_labels)
	_set_neighbor_validity(valid, 1, 1, 1, neighbor_valid)

	result = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

	assert result.diagnostics.neighbor_count[1, 1, 1] == sum(neighbor_valid)
	assert bool(result.diagnostics.changed_mask[1, 1, 1]) is (expected_label == 3)
	assert result.labels[1, 1, 1] == expected_label


def test_center_matching_consensus_is_unchanged() -> None:
	labels, valid = _three_level_grid(center_label=3)
	_set_neighbors(labels, 1, 1, 1, (3, 3, 3, 3))

	result = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

	assert result.diagnostics.consensus_mask[1, 1, 1]
	assert not result.diagnostics.change_candidate_mask[1, 1, 1]
	assert result.labels[1, 1, 1] == 3


def test_invalid_center_is_excluded_without_mutating_the_source_mask() -> None:
	labels, valid = _three_level_grid()
	labels[1, 1, 1] = -99
	valid[1, 1, 1] = False
	_set_neighbors(labels, 1, 1, 1, (3, 3, 3, 3))
	labels_before = labels.copy()
	valid_before = valid.copy()

	result = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

	assert result.labels[1, 1, 1] == -99
	assert result.diagnostics.neighbor_count[1, 1, 1] == 0
	assert not result.diagnostics.consensus_mask[1, 1, 1]
	np.testing.assert_array_equal(labels, labels_before)
	np.testing.assert_array_equal(valid, valid_before)


def test_invalid_values_and_valid_token_gaps_are_preserved_exactly() -> None:
	labels, valid = _three_level_grid(shape=(3, 3, 5), center_label=2)
	labels[:, :, 0] = 0
	labels[:, :, 1] = -31
	labels[:, :, 2] = 2
	labels[:, :, 3] = -37
	labels[:, :, 4] = 4
	valid[:, :, 1] = False
	valid[:, :, 3] = False
	_set_neighbors(labels, 1, 1, 2, (3, 3, 3, 3))

	result = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

	assert result.diagnostics.internal_valid_mask[1, 1, 2]
	assert result.labels[1, 1, 2] == 3
	np.testing.assert_array_equal(result.labels[~valid], labels[~valid])
	np.testing.assert_array_equal(result.diagnostics.neighbor_count[~valid], 0)


def test_order_guard_rejects_invalid_positions_and_out_of_bounds_proposals() -> None:
	labels, valid = _three_level_grid()
	labels[:, :, 0] = 1
	labels[:, :, 1] = 2
	labels[:, :, 2] = 3
	_set_neighbors(labels, 1, 1, 1, (4, 4, 4, 4))
	_set_neighbors(labels, 1, 1, 0, (2, 2, 2, 2))

	result = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

	assert result.diagnostics.change_candidate_mask[1, 1, 1]
	assert not result.diagnostics.order_compatible_mask[1, 1, 1]
	assert result.labels[1, 1, 1] == 2
	assert result.diagnostics.change_candidate_mask[1, 1, 0]
	assert not result.diagnostics.internal_valid_mask[1, 1, 0]
	assert result.labels[1, 1, 0] == 1

	labels, valid = _three_level_grid(shape=(3, 3, 5), center_label=2)
	labels[:, :, 0] = 0
	labels[:, :, 1:] = -8
	valid[:, :, 1:] = False
	labels[1, 1, 2] = 2
	valid[1, 1, 2] = True
	_set_neighbors(labels, 1, 1, 2, (3, 3, 3, 3))
	_set_neighbor_validity(valid, 1, 1, 2, (True, True, True, True))

	one_sided = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

	assert one_sided.diagnostics.change_candidate_mask[1, 1, 2]
	assert not one_sided.diagnostics.internal_valid_mask[1, 1, 2]
	assert one_sided.labels[1, 1, 2] == 2


def test_unanimous_application_is_a_single_synchronous_pass() -> None:
	labels, valid = _three_level_grid()
	labels[1, 1, 1] = 0
	_set_neighbors(labels, 1, 1, 1, (2, 2, 2, 2))

	first = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)
	second = smooth_xy_neighbor_unanimous_hard_labels(first.labels, valid)

	assert first.labels[1, 1, 1] == 2
	assert first.diagnostics.changed_mask[1, 1, 1]
	np.testing.assert_array_equal(second.labels, first.labels)
	assert not np.any(second.diagnostics.changed_mask)


def test_unanimous_changes_are_a_strict_policy_subset_of_three_of_four() -> None:
	rng = np.random.default_rng(731)
	for _ in range(80):
		shape = (5, 4, 7)
		valid = rng.random(shape) > 0.2
		labels = np.sort(rng.integers(0, 6, size=shape), axis=2).astype(np.int32)
		labels[~valid] = rng.integers(-100, -1, size=np.count_nonzero(~valid))

		consensus = smooth_xy_neighbor_consensus_hard_labels(labels, valid)
		unanimous = smooth_xy_neighbor_unanimous_hard_labels(labels, valid)

		assert np.all(
			~unanimous.diagnostics.changed_mask
			| consensus.diagnostics.changed_mask
		)
		np.testing.assert_array_equal(
			unanimous.labels[unanimous.diagnostics.changed_mask],
			consensus.labels[unanimous.diagnostics.changed_mask],
		)
		np.testing.assert_array_equal(unanimous.labels[~valid], labels[~valid])
		for x in range(shape[0]):
			for y in range(shape[1]):
				trace = unanimous.labels[x, y, valid[x, y]]
				assert np.all(np.diff(trace) >= 0)


def test_unanimous_identity_and_policy_leave_legacy_policy_unchanged() -> None:
	assert (
		XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_SEMANTICS
		== 'xy_neighbor_unanimous_outlier_correction_v1'
	)
	assert dict(XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_POLICY) == {
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
	assert dict(XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_POLICY) == {
		**dict(XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_POLICY),
		'four_valid_neighbors_minimum_agreement': 4,
	}


def _three_level_grid(
	*,
	shape: tuple[int, int, int] = (3, 3, 3),
	center_label: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
	labels = np.full(shape, center_label, dtype=np.int32)
	labels[:, :, 0] = 0
	labels[:, :, -1] = 4
	valid = np.ones(shape, dtype=bool)
	return labels, valid


def _set_neighbors(
	labels: np.ndarray,
	x: int,
	y: int,
	z: int,
	values: tuple[int, int, int, int],
) -> None:
	for (xx, yy), value in zip(
		((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)),
		values,
		strict=True,
	):
		labels[xx, yy, z] = value


def _set_neighbor_validity(
	valid: np.ndarray,
	x: int,
	y: int,
	z: int,
	values: tuple[bool, bool, bool, bool],
) -> None:
	for (xx, yy), value in zip(
		((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)),
		values,
		strict=True,
	):
		valid[xx, yy, z] = value
