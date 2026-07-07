from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.clustering.ordered_diagnostics import (
	ordered_boundary_summary,
	ordered_label_diagnostics,
)


def test_ordered_label_diagnostics_monotone_has_no_reverse_transitions() -> None:
	labels = np.array(
		[
			[[0, 0, 1, 1]],
			[[0, 1, 2, 2]],
		],
		dtype=np.int32,
	)

	diagnostics = ordered_label_diagnostics(labels, k=3)

	assert diagnostics['vertical_adjacent_pair_count'] == 6
	assert diagnostics['reverse_transition_count'] == 0
	assert diagnostics['reverse_transition_rate'] == 0.0
	assert diagnostics['mean_boundaries_per_valid_trace'] == 1.5
	assert diagnostics['max_boundaries_per_valid_trace'] == 2


def test_ordered_label_diagnostics_counts_reverse_transition() -> None:
	labels = np.array([[[0, 1, 0]]], dtype=np.int32)

	diagnostics = ordered_label_diagnostics(labels, k=2)

	assert diagnostics['vertical_adjacent_pair_count'] == 2
	assert diagnostics['reverse_transition_count'] == 1
	assert diagnostics['reverse_transition_rate'] == pytest.approx(0.5)


def test_ordered_label_diagnostics_invalid_gaps_do_not_bridge_pairs() -> None:
	labels = np.array([[[0, -1, 1]]], dtype=np.int32)

	diagnostics = ordered_label_diagnostics(labels, k=2)

	assert diagnostics['valid_token_count'] == 2
	assert diagnostics['invalid_token_count'] == 1
	assert diagnostics['vertical_adjacent_pair_count'] == 0
	assert diagnostics['forward_transition_count'] == 0
	assert diagnostics['mean_boundaries_per_valid_trace'] == 0.0


def test_ordered_label_diagnostics_counts_jump_transitions() -> None:
	labels = np.array([[[0, 2, 1]]], dtype=np.int32)

	diagnostics = ordered_label_diagnostics(labels, k=3)

	assert diagnostics['vertical_adjacent_pair_count'] == 2
	assert diagnostics['jump_transition_count'] == 1
	assert diagnostics['jump_transition_rate'] == pytest.approx(0.5)
	assert diagnostics['reverse_transition_count'] == 1


def test_ordered_boundary_summary_reports_expected_z_statistics() -> None:
	labels = np.array(
		[
			[[0, 1, 1, 2], [0, 0, 1, 2]],
			[[-1, 0, 0, 1], [0, 0, 0, 0]],
		],
		dtype=np.int32,
	)

	summary = ordered_boundary_summary(labels, k=3)

	assert summary['0_to_1']['observed_trace_count'] == 3
	assert summary['0_to_1']['mean_z'] == pytest.approx(2.0)
	assert summary['0_to_1']['min_z'] == 1
	assert summary['0_to_1']['max_z'] == 3
	assert summary['1_to_2']['observed_trace_count'] == 2
	assert summary['1_to_2']['mean_z'] == pytest.approx(3.0)
	assert summary['1_to_2']['min_z'] == 3
	assert summary['1_to_2']['max_z'] == 3


def test_ordered_label_diagnostics_rejects_invalid_label_values() -> None:
	labels = np.array([[[0, 2]]], dtype=np.int32)

	with pytest.raises(ValueError, match=r'outside valid range.*0\.\.1.*2'):
		ordered_label_diagnostics(labels, k=2)
