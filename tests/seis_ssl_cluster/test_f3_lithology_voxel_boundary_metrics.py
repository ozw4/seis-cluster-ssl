from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.voxel_boundary_metrics import (
	build_vertical_boundary_region_masks,
	compute_vertical_boundary_metrics,
)


def test_exact_boundary_has_perfect_metrics_and_zero_error() -> None:
	true = _trace(8, (3,))
	metrics = _metrics(true, true.copy(), tolerances=(0, 1))

	assert metrics['vertical_boundary_precision_at_0'] == 1.0
	assert metrics['vertical_boundary_recall_at_0'] == 1.0
	assert metrics['vertical_boundary_f1_at_0'] == 1.0
	assert metrics['vertical_boundary_position_mae_at_1'] == 0.0
	assert metrics['vertical_boundary_position_median_ae_at_1'] == 0.0


def test_shift_tolerance_extra_prediction_and_missing_prediction() -> None:
	true = _trace(12, (2, 8))
	pred = _trace(12, (3, 5))
	metrics = _metrics(true, pred, tolerances=(0, 1, 2))

	assert metrics['vertical_boundary_matched_count_at_0'] == 0
	assert metrics['vertical_boundary_matched_count_at_1'] == 1
	assert metrics['vertical_boundary_matched_count_at_2'] == 1
	assert metrics['vertical_boundary_true_count'] == 2
	assert metrics['vertical_boundary_pred_count'] == 2
	assert metrics['vertical_boundary_f1_at_1'] == pytest.approx(0.5)
	assert metrics['vertical_boundary_miss_rate_at_2'] == pytest.approx(0.5)


def test_dp_maximises_cardinality_before_minimising_distance() -> None:
	# Nearest-first greedy would consume prediction 3 for GT 2 and miss GT 4.
	true = _trace(7, (2, 4))
	pred = _trace(7, (0, 3))
	metrics = _metrics(true, pred, tolerances=(2,))

	assert metrics['vertical_boundary_matched_count_at_2'] == 2
	assert metrics['vertical_boundary_position_mae_at_2'] == pytest.approx(1.5)


def test_evaluation_and_prediction_gaps_do_not_create_boundaries() -> None:
	true = _trace(7, (2,))
	pred = true.copy()
	mask = np.ones_like(true, dtype=bool)
	mask[..., 3] = False
	metrics = compute_vertical_boundary_metrics(
		true,
		pred,
		evaluation_mask=mask,
		prediction_valid_mask=mask,
		class_ids=(0, 1),
		tolerances=(1,),
	)
	assert metrics['vertical_boundary_true_count'] == 0
	assert metrics['vertical_boundary_pred_count'] == 0
	assert metrics['vertical_boundary_f1_at_1'] is None
	assert metrics['vertical_boundary_position_mae_at_1'] is None


def test_boundary_regions_respect_radius_and_mask_gaps() -> None:
	true = _trace(9, (3,))
	mask = np.ones_like(true, dtype=bool)
	mask[..., 1] = False
	regions, interior = build_vertical_boundary_region_masks(
		true, evaluation_mask=mask, class_ids=(0, 1), radii=(0, 2)
	)

	assert np.flatnonzero(regions[0]).tolist() == [3]
	assert np.flatnonzero(regions[2]).tolist() == [2, 3, 4, 5]
	assert not regions[2][0, 0, 1]
	assert np.array_equal(interior, mask & ~regions[2])


def test_monitored_class_requires_class_on_both_matched_interfaces() -> None:
	true = np.asarray([[[0, 0, 3, 3, 5, 5]]])
	pred = np.asarray([[[0, 0, 1, 1, 5, 5]]])
	metrics = compute_vertical_boundary_metrics(
		true,
		pred,
		evaluation_mask=np.ones_like(true, dtype=bool),
		prediction_valid_mask=np.ones_like(true, dtype=bool),
		class_ids=(0, 1, 3, 5),
		tolerances=(0,),
		monitored_class_ids=(3, 5),
	)

	assert metrics['vertical_boundary_class_3_true_count'] == 2
	assert metrics['vertical_boundary_class_3_recall_at_0'] == 0.0
	assert metrics['vertical_boundary_class_5_true_count'] == 1
	assert metrics['vertical_boundary_class_5_recall_at_0'] == 1.0


def test_no_boundary_volume_has_json_safe_undefined_metrics() -> None:
	true = np.zeros((2, 2, 5), dtype=np.int64)
	metrics = _metrics(true, true, tolerances=(1,))
	assert metrics['vertical_boundary_true_count'] == 0
	assert metrics['vertical_boundary_precision_at_1'] is None
	assert metrics['vertical_boundary_recall_at_1'] is None
	assert metrics['vertical_boundary_miss_rate_at_1'] is None


def test_rejects_unknown_classes_and_shape_mismatch_without_mutation() -> None:
	true = _trace(5, (2,))
	pred = true.copy()
	true_before = true.copy()
	pred_before = pred.copy()
	with pytest.raises(ValueError, match='unknown class ids'):
		_metrics(true + 8, pred, tolerances=(1,))
	with pytest.raises(ValueError, match='matching shapes'):
		compute_vertical_boundary_metrics(
			true,
			pred[..., :-1],
			evaluation_mask=np.ones_like(true, dtype=bool),
			prediction_valid_mask=np.ones_like(true, dtype=bool),
			class_ids=(0, 1),
		)
	assert np.array_equal(true, true_before)
	assert np.array_equal(pred, pred_before)


def _trace(length: int, boundaries: tuple[int, ...]) -> np.ndarray:
	labels = np.zeros((1, 1, length), dtype=np.int64)
	value = 0
	for z in range(length):
		labels[0, 0, z] = value
		if z in boundaries:
			value = 1 - value
	return labels


def _metrics(
	true: np.ndarray, pred: np.ndarray, *, tolerances: tuple[int, ...]
) -> dict[str, float | int | None]:
	return compute_vertical_boundary_metrics(
		true,
		pred,
		evaluation_mask=np.ones_like(true, dtype=bool),
		prediction_valid_mask=np.ones_like(pred, dtype=bool),
		class_ids=(0, 1),
		tolerances=tolerances,
	)
