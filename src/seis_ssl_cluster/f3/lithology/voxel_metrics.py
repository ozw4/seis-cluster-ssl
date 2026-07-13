"""Streaming classification metrics for F3 lithology voxel volumes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Sequence

	from numpy.typing import NDArray


def update_confusion_matrix(
	matrix: NDArray[np.int64],
	y_true: NDArray[np.generic],
	y_pred: NDArray[np.generic],
	*,
	valid_mask: NDArray[np.bool_],
	class_ids: Sequence[int],
) -> None:
	"""Accumulate one masked chunk into a fixed-order confusion matrix."""
	class_id_list = _validated_class_ids(class_ids)
	_validate_matrix(matrix, len(class_id_list))
	true, pred, mask = _validated_chunk_arrays(y_true, y_pred, valid_mask)
	if not np.any(mask):
		return

	masked_true = _integer_labels(true[mask], 'y_true')
	masked_pred = _integer_labels(pred[mask], 'y_pred')
	class_id_array = np.asarray(class_id_list, dtype=np.int64)
	true_indices = _class_indices(masked_true, class_id_array, 'y_true')
	pred_indices = _class_indices(masked_pred, class_id_array, 'y_pred')
	chunk_counts = np.bincount(
		true_indices * len(class_id_list) + pred_indices,
		minlength=len(class_id_list) ** 2,
	).reshape(matrix.shape)
	if np.any(matrix > np.iinfo(np.int64).max - chunk_counts):
		msg = 'confusion matrix count would overflow int64'
		raise OverflowError(msg)
	matrix += chunk_counts


def _validate_matrix(matrix: NDArray[np.int64], n_classes: int) -> None:
	if not isinstance(matrix, np.ndarray) or matrix.dtype != np.dtype(np.int64):
		msg = 'matrix must be a NumPy array with dtype int64'
		raise TypeError(msg)
	expected_shape = (n_classes, n_classes)
	if matrix.shape != expected_shape:
		msg = f'matrix must have shape {expected_shape}; got {matrix.shape}'
		raise ValueError(msg)
	if np.any(matrix < 0):
		msg = 'matrix counts must be non-negative'
		raise ValueError(msg)


def _validated_chunk_arrays(
	y_true: NDArray[np.generic],
	y_pred: NDArray[np.generic],
	valid_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.bool_]]:
	true = np.asarray(y_true)
	pred = np.asarray(y_pred)
	mask = np.asarray(valid_mask)
	if true.shape != pred.shape or true.shape != mask.shape:
		msg = (
			'y_true, y_pred, and valid_mask must have matching shapes; '
			f'got {true.shape}, {pred.shape}, and {mask.shape}'
		)
		raise ValueError(msg)
	if mask.dtype != np.dtype(bool):
		msg = 'valid_mask must have boolean dtype'
		raise TypeError(msg)
	return true, pred, mask


def _class_indices(
	values: NDArray[np.int64],
	class_ids: NDArray[np.int64],
	label: str,
) -> NDArray[np.intp]:
	order = np.argsort(class_ids)
	sorted_ids = class_ids[order]
	positions = np.searchsorted(sorted_ids, values)
	known = positions < sorted_ids.size
	known[known] &= sorted_ids[positions[known]] == values[known]
	if not np.all(known):
		unknown = np.unique(values[~known]).tolist()
		msg = f'masked labels contain unknown class ids: {label}={unknown!r}'
		raise ValueError(msg)
	return np.asarray(order[positions], dtype=np.intp)


def _validated_class_ids(class_ids: Sequence[int]) -> tuple[int, ...]:
	values = tuple(int(class_id) for class_id in class_ids)
	if not values:
		msg = 'class_ids must contain at least one class'
		raise ValueError(msg)
	if len(set(values)) != len(values):
		msg = f'class_ids must be unique; got {values!r}'
		raise ValueError(msg)
	return values


def _integer_labels(values: NDArray[np.generic], label: str) -> NDArray[np.int64]:
	if not np.issubdtype(values.dtype, np.integer):
		if not np.issubdtype(values.dtype, np.floating):
			msg = f'{label} must contain integer class ids'
			raise ValueError(msg)
		rounded = np.rint(values)
		if not np.all(np.isfinite(values)) or not np.array_equal(values, rounded):
			msg = f'{label} must contain integer class ids'
			raise ValueError(msg)
		values = rounded
	return np.asarray(values, dtype=np.int64)


__all__ = ['update_confusion_matrix']
