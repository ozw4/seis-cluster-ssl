from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus import (
	XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_SEMANTICS,
	smooth_xy_neighbor_consensus_hard_labels,
)


def test_four_neighbor_three_of_four_consensus_changes_internal_center() -> None:
	labels, valid = _three_level_grid()
	_set_neighbors(labels, 1, 1, 1, (3, 3, 3, 1))

	result = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert result.labels[1, 1, 1] == 3
	assert result.diagnostics.neighbor_count[1, 1, 1] == 4
	assert result.diagnostics.proposal_labels[1, 1, 1] == 3
	assert result.diagnostics.changed_mask[1, 1, 1]


def test_consensus_uses_same_z_xy_neighbors_not_other_depths() -> None:
	labels, valid = _three_level_grid()
	_set_neighbors(labels, 1, 1, 1, (1, 1, 3, 3))
	_set_neighbors(labels, 1, 1, 2, (3, 3, 3, 3))

	result = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert result.diagnostics.neighbor_count[1, 1, 1] == 4
	assert not result.diagnostics.consensus_mask[1, 1, 1]
	assert result.labels[1, 1, 1] == 2


@pytest.mark.parametrize(
	('neighbor_labels', 'neighbor_valid', 'expected_label'),
	[
		((3, 3, 3, 1), (True, True, True, True), 3),
		((3, 3, 1, 1), (True, True, True, True), 2),
		((3, 3, 3, 9), (True, True, True, False), 3),
		((3, 3, 1, 9), (True, True, True, False), 2),
		((3, 3, 9, 9), (True, True, False, False), 2),
	],
)
def test_neighbor_count_thresholds_and_invalid_neighbor_exclusion(
	neighbor_labels: tuple[int, int, int, int],
	neighbor_valid: tuple[bool, bool, bool, bool],
	expected_label: int,
) -> None:
	labels, valid = _three_level_grid()
	_set_neighbors(labels, 1, 1, 1, neighbor_labels)
	_set_neighbor_validity(valid, 1, 1, 1, neighbor_valid)

	result = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert result.labels[1, 1, 1] == expected_label
	assert bool(result.diagnostics.changed_mask[1, 1, 1]) is (expected_label == 3)
	assert result.diagnostics.neighbor_count[1, 1, 1] == sum(neighbor_valid)


def test_center_matching_consensus_and_two_to_two_tie_are_unchanged() -> None:
	labels, valid = _three_level_grid(center_label=3)
	_set_neighbors(labels, 1, 1, 1, (3, 3, 3, 1))

	matching = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert matching.labels[1, 1, 1] == 3
	assert matching.diagnostics.consensus_mask[1, 1, 1]
	assert not matching.diagnostics.change_candidate_mask[1, 1, 1]
	assert not matching.diagnostics.changed_mask[1, 1, 1]

	labels, valid = _three_level_grid()
	_set_neighbors(labels, 1, 1, 1, (1, 1, 3, 3))

	tied = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert tied.labels[1, 1, 1] == 2
	assert not tied.diagnostics.consensus_mask[1, 1, 1]
	assert not tied.diagnostics.changed_mask[1, 1, 1]


def test_invalid_center_is_never_a_candidate_and_its_value_is_preserved() -> None:
	labels, valid = _three_level_grid()
	labels[1, 1, 1] = -99
	valid[1, 1, 1] = False
	_set_neighbors(labels, 1, 1, 1, (3, 3, 3, 1))

	result = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert result.labels[1, 1, 1] == -99
	assert result.diagnostics.neighbor_count[1, 1, 1] == 0
	assert not result.diagnostics.consensus_mask[1, 1, 1]
	assert not result.diagnostics.changed_mask[1, 1, 1]


def test_updates_are_synchronous_against_frozen_source_labels() -> None:
	labels, valid = _three_level_grid(shape=(4, 3, 3), center_label=2)
	# A changes from 0 to 2.  B would obtain a three-of-four vote for 2 only if
	# it were allowed to consume A's just-updated output instead of source A=0.
	labels[1, 1, 1] = 0
	labels[2, 1, 1] = 1
	_set_neighbors(labels, 1, 1, 1, (2, 1, 2, 2))
	_set_neighbors(labels, 2, 1, 1, (0, 2, 2, 3))

	result = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert result.labels[1, 1, 1] == 2
	assert result.labels[2, 1, 1] == 1
	assert not result.diagnostics.consensus_mask[2, 1, 1]


def test_order_guard_accepts_internal_proposal_with_valid_z_gaps() -> None:
	labels, valid = _three_level_grid(shape=(3, 3, 5), center_label=2)
	labels[:, :, 0] = 0
	labels[:, :, 1] = -7
	labels[:, :, 2] = 2
	labels[:, :, 3] = -7
	labels[:, :, 4] = 4
	valid[:, :, 1] = False
	valid[:, :, 3] = False
	_set_neighbors(labels, 1, 1, 2, (3, 3, 3, 1))

	result = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert result.diagnostics.internal_valid_mask[1, 1, 2]
	assert result.labels[1, 1, 2] == 3
	assert result.labels[1, 1, 1] == -7
	assert result.labels[1, 1, 3] == -7


def test_order_guard_rejects_out_of_range_endpoints_and_one_sided_valid_tokens() -> (
	None
):
	labels, valid = _three_level_grid()
	labels[:, :, 0] = 1
	labels[:, :, 1] = 2
	labels[:, :, 2] = 3
	_set_neighbors(labels, 1, 1, 1, (4, 4, 4, 0))
	_set_neighbors(labels, 1, 1, 0, (2, 2, 2, 0))

	result = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert result.diagnostics.change_candidate_mask[1, 1, 1]
	assert not result.diagnostics.order_compatible_mask[1, 1, 1]
	assert result.labels[1, 1, 1] == 2
	assert result.diagnostics.change_candidate_mask[1, 1, 0]
	assert not result.diagnostics.internal_valid_mask[1, 1, 0]
	assert result.labels[1, 1, 0] == 1

	labels, valid = _three_level_grid(shape=(3, 3, 5), center_label=2)
	labels[:, :, 0] = 0
	labels[:, :, 1] = -8
	labels[:, :, 2] = 2
	labels[:, :, 3] = -8
	labels[:, :, 4] = -8
	valid[:, :, 1:] = False
	valid[:, :, 2] = True
	_set_neighbors(labels, 1, 1, 2, (3, 3, 3, 1))

	one_sided = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	assert one_sided.diagnostics.change_candidate_mask[1, 1, 2]
	assert not one_sided.diagnostics.internal_valid_mask[1, 1, 2]
	assert one_sided.labels[1, 1, 2] == 2


def test_invalid_values_are_preserved_and_valid_traces_remain_non_decreasing() -> None:
	labels, valid = _three_level_grid(shape=(4, 3, 5), center_label=2)
	labels[:, :, 0] = 0
	labels[:, :, 1] = -5
	labels[:, :, 2] = 2
	labels[:, :, 3] = -6
	labels[:, :, 4] = 4
	valid[:, :, 1] = False
	valid[:, :, 3] = False
	_set_neighbors(labels, 1, 1, 2, (3, 3, 3, 1))

	result = smooth_xy_neighbor_consensus_hard_labels(labels, valid)

	np.testing.assert_array_equal(result.labels[~valid], labels[~valid])
	for x in range(labels.shape[0]):
		for y in range(labels.shape[1]):
			trace = result.labels[x, y, valid[x, y]]
			assert np.all(np.diff(trace) >= 0)


def test_semantic_identifier_is_public_contract_value() -> None:
	assert (
		XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_SEMANTICS
		== 'xy_neighbor_consensus_hard_label_smoothing_v1'
	)


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
		((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)), values, strict=True
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
		((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)), values, strict=True
	):
		valid[xx, yy, z] = value
