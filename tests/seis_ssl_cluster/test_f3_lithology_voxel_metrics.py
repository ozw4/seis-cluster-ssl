from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.f3.labels import F3ClassInfo
from seis_ssl_cluster.f3.lithology.metrics import (
	compute_lithology_metrics,
	lithology_metrics_from_confusion_matrix,
)
from seis_ssl_cluster.f3.lithology.voxel_metrics import update_confusion_matrix


def test_chunked_metrics_match_direct_metrics_independent_of_order() -> None:
	true = np.asarray([0, 1, 5, 0, 5, 1, 0, 5])
	pred = np.asarray([0, 5, 5, 1, 0, 1, 0, 5])
	direct = compute_lithology_metrics(true, pred, _classes())

	forward = _chunked_matrix(true, pred, (slice(0, 3), slice(3, None)))
	reverse = _chunked_matrix(true, pred, (slice(3, None), slice(0, 3)))

	assert np.array_equal(forward, reverse)
	assert lithology_metrics_from_confusion_matrix(forward, _classes()) == direct


def test_masked_values_are_ignored_and_unknown_selected_values_are_rejected() -> None:
	matrix = np.zeros((3, 3), dtype=np.int64)
	update_confusion_matrix(
		matrix,
		np.asarray([[0, 99], [5, 1]]),
		np.asarray([[0, -1], [1, 1]]),
		valid_mask=np.asarray([[True, False], [True, True]]),
		class_ids=(0, 1, 5),
	)
	assert matrix.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]

	with pytest.raises(ValueError, match='unknown class ids'):
		update_confusion_matrix(
			matrix,
			np.asarray([0, 99]),
			np.asarray([0, 1]),
			valid_mask=np.ones(2, dtype=bool),
			class_ids=(0, 1, 5),
		)
	with pytest.raises(ValueError, match='unknown class ids'):
		update_confusion_matrix(
			matrix,
			np.asarray([0, 1]),
			np.asarray([0, 99]),
			valid_mask=np.ones(2, dtype=bool),
			class_ids=(0, 1, 5),
		)


def test_zero_support_class_is_retained_with_zero_division_semantics() -> None:
	matrix = np.asarray([[2, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.int64)
	metrics = lithology_metrics_from_confusion_matrix(matrix, _classes())

	assert metrics['per_class_support'] == {'0': 2, '1': 2, '5': 0}
	assert metrics['per_class_precision']['5'] == 0.0
	assert metrics['per_class_recall']['5'] == 0.0
	assert metrics['per_class_f1']['5'] == 0.0
	assert metrics['per_class_iou']['5'] == 0.0
	assert metrics['balanced_accuracy'] == pytest.approx(0.75)


def test_empty_chunk_is_ignored_but_empty_matrix_is_rejected() -> None:
	matrix = np.zeros((3, 3), dtype=np.int64)
	update_confusion_matrix(
		matrix,
		np.asarray([99]),
		np.asarray([-1]),
		valid_mask=np.zeros(1, dtype=bool),
		class_ids=(0, 1, 5),
	)
	assert not matrix.any()

	with pytest.raises(ValueError, match='at least one labeled evaluation voxel'):
		lithology_metrics_from_confusion_matrix(matrix, _classes())


@pytest.mark.parametrize('class_ids', [(0, 1.9), (0, '1'), (0, True)])
def test_class_ids_reject_non_integer_values(class_ids: tuple[object, ...]) -> None:
	matrix = np.zeros((2, 2), dtype=np.int64)
	with pytest.raises(TypeError, match='class_ids must contain integers'):
		update_confusion_matrix(
			matrix,
			np.asarray([0]),
			np.asarray([0]),
			valid_mask=np.ones(1, dtype=bool),
			class_ids=class_ids,  # type: ignore[arg-type]
		)


def test_confusion_matrix_preserves_counts_larger_than_int32() -> None:
	large = 2**31 + 17
	matrix = np.asarray([[large, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
	update_confusion_matrix(
		matrix,
		np.asarray([0, 5]),
		np.asarray([0, 5]),
		valid_mask=np.ones(2, dtype=bool),
		class_ids=(0, 1, 5),
	)

	assert matrix.dtype == np.int64
	assert matrix[0, 0] == large + 1
	metrics = lithology_metrics_from_confusion_matrix(matrix, _classes())
	assert metrics['confusion_matrix'][0][0] == large + 1


def test_metric_aggregates_do_not_overflow_int64() -> None:
	maximum = np.iinfo(np.int64).max
	matrix = np.asarray(
		[[maximum, maximum, 0], [0, 1, 0], [0, 0, 1]],
		dtype=np.int64,
	)

	metrics = lithology_metrics_from_confusion_matrix(matrix, _classes())

	assert metrics['per_class_support']['0'] == 2 * maximum
	assert metrics['accuracy'] == pytest.approx(0.5)
	assert metrics['per_class_recall']['0'] == pytest.approx(0.5)
	assert metrics['per_class_f1']['0'] == pytest.approx(2 / 3)
	assert metrics['per_class_iou']['0'] == pytest.approx(0.5)
	assert metrics['confusion_matrix_row_normalized'][0] == pytest.approx([0.5, 0.5, 0])


def _chunked_matrix(
	true: np.ndarray,
	pred: np.ndarray,
	chunks: tuple[slice, ...],
) -> np.ndarray:
	matrix = np.zeros((3, 3), dtype=np.int64)
	for chunk in chunks:
		update_confusion_matrix(
			matrix,
			true[chunk],
			pred[chunk],
			valid_mask=np.ones(true[chunk].shape, dtype=bool),
			class_ids=(0, 1, 5),
		)
	return matrix


def _classes() -> tuple[F3ClassInfo, ...]:
	return (
		F3ClassInfo(class_id=0, class_name='Upper North Sea', rgb=(0, 0, 0)),
		F3ClassInfo(class_id=1, class_name='Middle North Sea', rgb=(1, 1, 1)),
		F3ClassInfo(class_id=5, class_name='Zechstein', rgb=(5, 5, 5)),
	)
