"""Vertical-trace boundary metrics for F3 lithology voxel volumes."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Sequence

	from numpy.typing import NDArray


def compute_vertical_boundary_metrics(  # noqa: PLR0913
	y_true: NDArray[np.generic],
	y_pred: NDArray[np.generic],
	*,
	evaluation_mask: NDArray[np.bool_],
	prediction_valid_mask: NDArray[np.bool_],
	class_ids: Sequence[int],
	tolerances: Sequence[int] = (1, 2, 4, 8),
	monitored_class_ids: Sequence[int] = (),
) -> dict[str, float | int | None]:
	"""Compute one-to-one vertical boundary metrics over an XYZ volume."""
	true, pred, evaluation, prediction_valid = _validated_inputs(
		y_true, y_pred, evaluation_mask, prediction_valid_mask
	)
	classes = _validated_class_ids(class_ids, 'class_ids')
	monitored = _validated_class_ids(
		monitored_class_ids, 'monitored_class_ids', allow_empty=True
	)
	unknown_monitored = sorted(set(monitored).difference(classes))
	if unknown_monitored:
		raise ValueError(
			'monitored_class_ids contain unknown class ids: '
			f'{unknown_monitored!r}'
		)
	tolerance_values = _validated_nonnegative_values(tolerances, 'tolerances')
	true_labels = _validated_labels(true, evaluation, classes, 'y_true')
	pred_labels = _validated_labels(pred, prediction_valid, classes, 'y_pred')

	true_boundaries = _boundaries_by_trace(true_labels, evaluation)
	pred_boundaries = _boundaries_by_trace(pred_labels, prediction_valid)
	true_count = sum(len(boundaries) for boundaries in true_boundaries)
	pred_count = sum(len(boundaries) for boundaries in pred_boundaries)
	metrics: dict[str, float | int | None] = {
		'vertical_boundary_true_count': true_count,
		'vertical_boundary_pred_count': pred_count,
	}
	distances_at_max: list[int] = []
	for tolerance in tolerance_values:
		distances = _match_volume_boundaries(
			true_boundaries, pred_boundaries, tolerance
		)
		matched = len(distances)
		metrics[f'vertical_boundary_matched_count_at_{tolerance}'] = matched
		metrics[f'vertical_boundary_precision_at_{tolerance}'] = _ratio(
			matched, pred_count
		)
		metrics[f'vertical_boundary_recall_at_{tolerance}'] = _ratio(
			matched, true_count
		)
		metrics[f'vertical_boundary_f1_at_{tolerance}'] = (
			None
			if true_count + pred_count == 0
			else 2.0 * matched / (true_count + pred_count)
		)
		if tolerance == max(tolerance_values):
			distances_at_max = distances

	max_tolerance = max(tolerance_values)
	metrics[f'vertical_boundary_position_mae_at_{max_tolerance}'] = (
		float(np.mean(distances_at_max)) if distances_at_max else None
	)
	metrics[f'vertical_boundary_position_median_ae_at_{max_tolerance}'] = (
		float(np.median(distances_at_max)) if distances_at_max else None
	)
	metrics[f'vertical_boundary_miss_rate_at_{max_tolerance}'] = (
		None
		if true_count == 0
		else (true_count - len(distances_at_max)) / true_count
	)

	for class_id in monitored:
		class_true = _class_boundaries_by_trace(
			true_boundaries, true_labels, class_id
		)
		class_pred = _class_boundaries_by_trace(
			pred_boundaries, pred_labels, class_id
		)
		class_true_count = sum(len(boundaries) for boundaries in class_true)
		metrics[f'vertical_boundary_class_{class_id}_true_count'] = class_true_count
		for tolerance in tolerance_values:
			matched = len(
				_match_volume_boundaries(class_true, class_pred, tolerance)
			)
			metrics[
				f'vertical_boundary_class_{class_id}_matched_count_at_{tolerance}'
			] = matched
			metrics[f'vertical_boundary_class_{class_id}_recall_at_{tolerance}'] = (
				_ratio(matched, class_true_count)
			)
	return metrics


def build_vertical_boundary_region_masks(
	y_true: NDArray[np.generic],
	*,
	evaluation_mask: NDArray[np.bool_],
	class_ids: Sequence[int],
	radii: Sequence[int] = (1, 2, 4, 8),
) -> tuple[dict[int, NDArray[np.bool_]], NDArray[np.bool_]]:
	"""Return gap-aware GT boundary masks by radius and max-radius interior."""
	true = np.asarray(y_true)
	evaluation = np.asarray(evaluation_mask)
	if true.ndim != 3:
		raise ValueError(f'y_true must be a 3D XYZ array; got shape {true.shape}')
	if evaluation.shape != true.shape:
		raise ValueError(
			'y_true and evaluation_mask must have matching shapes; '
			f'got {true.shape} and {evaluation.shape}'
		)
	if evaluation.dtype != np.dtype(bool):
		raise TypeError('evaluation_mask must have boolean dtype')
	classes = _validated_class_ids(class_ids, 'class_ids')
	radius_values = _validated_nonnegative_values(radii, 'radii')
	labels = _validated_labels(true, evaluation, classes, 'y_true')
	boundaries = _boundaries_by_trace(labels, evaluation)
	regions = {
		radius: _boundary_region_mask(boundaries, evaluation, radius)
		for radius in radius_values
	}
	interior = evaluation & ~regions[max(radius_values)]
	return regions, interior


def _validated_inputs(
	y_true: NDArray[np.generic],
	y_pred: NDArray[np.generic],
	evaluation_mask: NDArray[np.bool_],
	prediction_valid_mask: NDArray[np.bool_],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	arrays = tuple(
		np.asarray(value)
		for value in (y_true, y_pred, evaluation_mask, prediction_valid_mask)
	)
	true, pred, evaluation, prediction_valid = arrays
	if true.ndim != 3:
		raise ValueError(f'inputs must be 3D XYZ arrays; got shape {true.shape}')
	if any(value.shape != true.shape for value in arrays[1:]):
		raise ValueError(
			'y_true, y_pred, evaluation_mask, and prediction_valid_mask must '
			f'have matching shapes; got {[value.shape for value in arrays]!r}'
		)
	if evaluation.dtype != np.dtype(bool):
		raise TypeError('evaluation_mask must have boolean dtype')
	if prediction_valid.dtype != np.dtype(bool):
		raise TypeError('prediction_valid_mask must have boolean dtype')
	return true, pred, evaluation, prediction_valid


def _validated_class_ids(
	values: Sequence[int], label: str, *, allow_empty: bool = False
) -> tuple[int, ...]:
	result = tuple(
		int(value) if isinstance(value, Integral) else value for value in values
	)
	if not result and not allow_empty:
		raise ValueError(f'{label} must contain at least one class')
	if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
		raise TypeError(f'{label} must contain integers')
	if len(set(result)) != len(result):
		raise ValueError(f'{label} must not contain duplicates')
	return result


def _validated_nonnegative_values(
	values: Sequence[int], label: str
) -> tuple[int, ...]:
	result = tuple(
		int(value) if isinstance(value, Integral) else value for value in values
	)
	if not result:
		raise ValueError(f'{label} must not be empty')
	if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
		raise TypeError(f'{label} must contain integers')
	if any(value < 0 for value in result):
		raise ValueError(f'{label} must contain non-negative values')
	if len(set(result)) != len(result):
		raise ValueError(f'{label} must not contain duplicates')
	return result


def _validated_labels(
	values: np.ndarray,
	mask: np.ndarray,
	class_ids: tuple[int, ...],
	label: str,
) -> np.ndarray:
	selected = values[mask]
	if not np.issubdtype(values.dtype, np.integer):
		if not np.issubdtype(values.dtype, np.floating):
			raise ValueError(f'{label} must contain integer class ids')
		rounded = np.rint(selected)
		if not np.all(np.isfinite(selected)) or not np.array_equal(selected, rounded):
			raise ValueError(f'{label} must contain integer class ids')
	result = np.zeros(values.shape, dtype=np.int64)
	result[mask] = selected
	unknown = np.setdiff1d(np.unique(result[mask]), np.asarray(class_ids))
	if unknown.size:
		raise ValueError(
			f'{label} contains unknown class ids in its valid mask: '
			f'{unknown.tolist()!r}'
		)
	return result


def _boundaries_by_trace(
	labels: np.ndarray, mask: np.ndarray
) -> list[NDArray[np.int64]]:
	interfaces = mask[..., :-1] & mask[..., 1:]
	interfaces &= labels[..., :-1] != labels[..., 1:]
	return [
		np.flatnonzero(interfaces[x, y]).astype(np.int64, copy=False)
		for x in range(labels.shape[0])
		for y in range(labels.shape[1])
	]


def _class_boundaries_by_trace(
	boundaries: list[NDArray[np.int64]], labels: np.ndarray, class_id: int
) -> list[NDArray[np.int64]]:
	result: list[NDArray[np.int64]] = []
	for trace_index, trace_boundaries in enumerate(boundaries):
		x, y = divmod(trace_index, labels.shape[1])
		selected = [
			int(z)
			for z in trace_boundaries
			if labels[x, y, z] == class_id or labels[x, y, z + 1] == class_id
		]
		result.append(np.asarray(selected, dtype=np.int64))
	return result


def _match_volume_boundaries(
	true_boundaries: list[NDArray[np.int64]],
	pred_boundaries: list[NDArray[np.int64]],
	tolerance: int,
) -> list[int]:
	distances: list[int] = []
	for true_trace, pred_trace in zip(
		true_boundaries, pred_boundaries, strict=True
	):
		distances.extend(_ordered_match_distances(true_trace, pred_trace, tolerance))
	return distances


def _ordered_match_distances(
	true_boundaries: NDArray[np.int64],
	pred_boundaries: NDArray[np.int64],
	tolerance: int,
) -> list[int]:
	"""Maximise cardinality, then minimise total distance, for ordered points."""
	n_true, n_pred = len(true_boundaries), len(pred_boundaries)
	counts = np.zeros((n_true + 1, n_pred + 1), dtype=np.int64)
	costs = np.zeros((n_true + 1, n_pred + 1), dtype=np.int64)
	actions = np.zeros((n_true, n_pred), dtype=np.uint8)
	for true_index in range(n_true - 1, -1, -1):
		for pred_index in range(n_pred - 1, -1, -1):
			options = [
				(
					counts[true_index + 1, pred_index],
					costs[true_index + 1, pred_index],
					2,
				),
				(
					counts[true_index, pred_index + 1],
					costs[true_index, pred_index + 1],
					3,
				),
			]
			distance = abs(
				int(true_boundaries[true_index]) - int(pred_boundaries[pred_index])
			)
			if distance <= tolerance:
				options.insert(
					0,
					(
						counts[true_index + 1, pred_index + 1] + 1,
						costs[true_index + 1, pred_index + 1] + distance,
						1,
					),
				)
			best = min(options, key=lambda item: (-int(item[0]), int(item[1])))
			counts[true_index, pred_index] = best[0]
			costs[true_index, pred_index] = best[1]
			actions[true_index, pred_index] = best[2]

	distances: list[int] = []
	true_index = pred_index = 0
	while true_index < n_true and pred_index < n_pred:
		action = actions[true_index, pred_index]
		if action == 1:
			distances.append(
				abs(
					int(true_boundaries[true_index])
					- int(pred_boundaries[pred_index])
				)
			)
			true_index += 1
			pred_index += 1
		elif action == 2:
			true_index += 1
		else:
			pred_index += 1
	return distances


def _boundary_region_mask(
	boundaries: list[NDArray[np.int64]], evaluation: np.ndarray, radius: int
) -> NDArray[np.bool_]:
	region = np.zeros(evaluation.shape, dtype=bool)
	for trace_index, trace_boundaries in enumerate(boundaries):
		x, y = divmod(trace_index, evaluation.shape[1])
		trace_mask = evaluation[x, y]
		for boundary in trace_boundaries:
			z = int(boundary)
			start = max(0, z - radius)
			stop = min(len(trace_mask), z + radius + 1)
			for sample in range(start, stop):
				path = trace_mask[min(sample, z) : max(sample, z + 1) + 1]
				if trace_mask[sample] and np.all(path):
					region[x, y, sample] = True
	return region


def _ratio(numerator: int, denominator: int) -> float | None:
	return None if denominator == 0 else numerator / denominator


__all__ = [
	'build_vertical_boundary_region_masks',
	'compute_vertical_boundary_metrics',
]
