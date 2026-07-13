"""Metrics and report writers for F3 token-level lithology probes."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Sequence

	from numpy.typing import NDArray

	from seis_ssl_cluster.f3.labels import F3ClassInfo


REQUIRED_LITHOLOGY_METRICS = (
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'per_class_precision',
	'per_class_recall',
	'per_class_f1',
	'per_class_iou',
	'mean_iou',
	'confusion_matrix',
)


def compute_lithology_metrics(
	y_true: NDArray[np.generic],
	y_pred: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
) -> dict[str, object]:
	"""Compute F3 lithology classification metrics using fixed class ordering."""
	true = _label_vector(y_true, 'y_true')
	pred = _label_vector(y_pred, 'y_pred')
	if true.shape != pred.shape:
		msg = (
			'y_true and y_pred must have matching shapes; '
			f'got {true.shape} and {pred.shape}'
		)
		raise ValueError(msg)
	if true.size == 0:
		msg = 'metrics require at least one labeled validation token'
		raise ValueError(msg)
	class_ids = _class_ids(classes)
	matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
	from seis_ssl_cluster.f3.lithology.voxel_metrics import (  # noqa: PLC0415
		update_confusion_matrix,
	)

	update_confusion_matrix(
		matrix,
		true,
		pred,
		valid_mask=np.ones(true.shape, dtype=bool),
		class_ids=class_ids,
	)
	return lithology_metrics_from_confusion_matrix(matrix, classes)


def lithology_metrics_from_confusion_matrix(
	matrix: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
) -> dict[str, object]:
	"""Return lithology metrics from a fixed-class-order confusion matrix."""
	class_ids = _class_ids(classes)
	counts = _validated_confusion_matrix(matrix, len(class_ids))
	exact_counts = counts.astype(object)
	support = exact_counts.sum(axis=1)
	predicted = exact_counts.sum(axis=0)
	true_positive = np.diag(exact_counts)
	precision = _safe_ratio(true_positive, predicted)
	recall = _safe_ratio(true_positive, support)
	f1 = _safe_ratio(2 * true_positive, support + predicted)
	per_class_iou = _safe_ratio(true_positive, support + predicted - true_positive)
	row_normalized = _row_normalized_confusion_matrix(counts, support)
	total = sum(int(value) for value in support)
	supported = support != 0
	return {
		'accuracy': float(sum(int(value) for value in true_positive) / total),
		'balanced_accuracy': float(np.mean(recall[supported])),
		'macro_f1': float(np.mean(f1)),
		'weighted_f1': float(np.average(f1, weights=_as_float_array(support))),
		'per_class_precision': _per_class_metric(class_ids, precision),
		'per_class_recall': _per_class_metric(class_ids, recall),
		'per_class_f1': _per_class_metric(class_ids, f1),
		'per_class_iou': _per_class_metric(class_ids, per_class_iou),
		'per_class_support': {
			str(class_id): int(value)
			for class_id, value in zip(class_ids, support, strict=True)
		},
		'mean_iou': float(np.mean(per_class_iou)),
		'confusion_matrix': counts.tolist(),
		'confusion_matrix_row_normalized': row_normalized.tolist(),
		'class_ids': [int(class_id) for class_id in class_ids],
		'class_names': {
			str(class_info.class_id): class_info.class_name for class_info in classes
		},
	}


def write_metrics_csv(
	path: str | Path,
	metrics: Mapping[str, object],
	classes: Sequence[F3ClassInfo],
) -> None:
	"""Write overall and per-class metrics in a long CSV table."""
	rows: list[dict[str, object]] = [
		{
			'metric': metric,
			'class_id': '',
			'class_name': '',
			'value': _float_metric(metrics[metric]),
		}
		for metric in (
			'accuracy',
			'balanced_accuracy',
			'macro_f1',
			'weighted_f1',
			'mean_iou',
		)
	]
	for metric in (
		'per_class_precision',
		'per_class_recall',
		'per_class_f1',
		'per_class_iou',
	):
		values = _metric_mapping(metrics[metric], metric)
		rows.extend(
			[
				{
					'metric': metric,
					'class_id': class_info.class_id,
					'class_name': class_info.class_name,
					'value': _float_metric(values[str(class_info.class_id)]),
				}
				for class_info in classes
			],
		)
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(
			file_obj,
			fieldnames=('metric', 'class_id', 'class_name', 'value'),
		)
		writer.writeheader()
		writer.writerows(rows)


def write_confusion_matrix_csv(
	path: str | Path,
	metrics: Mapping[str, object],
	classes: Sequence[F3ClassInfo],
) -> None:
	"""Write the raw validation confusion matrix with class-id columns."""
	matrix = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
	class_ids = [class_info.class_id for class_info in classes]
	if matrix.shape != (len(class_ids), len(class_ids)):
		msg = (
			'confusion_matrix shape must match classes; '
			f'got {matrix.shape}, expected={(len(class_ids), len(class_ids))}'
		)
		raise ValueError(msg)
	fieldnames = (
		'true_class_id',
		'true_class_name',
		*(f'pred_{class_id}' for class_id in class_ids),
		'total',
	)
	rows = []
	for row_index, class_info in enumerate(classes):
		row = {
			'true_class_id': class_info.class_id,
			'true_class_name': class_info.class_name,
			'total': int(matrix[row_index].sum()),
		}
		row.update(
			{
				f'pred_{class_id}': int(value)
				for class_id, value in zip(class_ids, matrix[row_index], strict=True)
			},
		)
		rows.append(row)
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def render_classification_report_markdown(
	metrics: Mapping[str, object],
	classes: Sequence[F3ClassInfo],
) -> str:
	"""Render a compact validation report emphasizing macro F1 and Zechstein."""
	precision = _metric_mapping(metrics['per_class_precision'], 'per_class_precision')
	recall = _metric_mapping(metrics['per_class_recall'], 'per_class_recall')
	f1 = _metric_mapping(metrics['per_class_f1'], 'per_class_f1')
	iou = _metric_mapping(metrics['per_class_iou'], 'per_class_iou')
	support = _metric_mapping(metrics['per_class_support'], 'per_class_support')
	lines = [
		'# F3 lithology probe validation report',
		'',
		'## Highlights',
		'',
		f'- macro F1: {_format_float(_float_metric(metrics["macro_f1"]))}',
		f'- mean IoU: {_format_float(_float_metric(metrics["mean_iou"]))}',
		(
			'- balanced accuracy: '
			f'{_format_float(_float_metric(metrics["balanced_accuracy"]))}'
		),
	]
	class_five = next(
		(class_info for class_info in classes if class_info.class_id == 5),
		None,
	)
	if class_five is not None:
		key = str(class_five.class_id)
		lines.extend(
			[
				(
					f'- class 5 {class_five.class_name} recall: '
					f'{_format_float(_float_metric(recall[key]))}'
				),
				(
					f'- class 5 {class_five.class_name} F1: '
					f'{_format_float(_float_metric(f1[key]))}'
				),
			],
		)
	lines.extend(
		[
			'',
			'## Per-class metrics',
			'',
			'| class_id | class_name | precision | recall | F1 | IoU | support |',
			'|---:|---|---:|---:|---:|---:|---:|',
		],
	)
	for class_info in classes:
		key = str(class_info.class_id)
		lines.append(
			f'| {class_info.class_id} | {class_info.class_name} | '
			f'{_format_float(_float_metric(precision[key]))} | '
			f'{_format_float(_float_metric(recall[key]))} | '
			f'{_format_float(_float_metric(f1[key]))} | '
			f'{_format_float(_float_metric(iou[key]))} | '
			f'{int(support[key])} |',
		)
	return '\n'.join(lines) + '\n'


def _label_vector(values: NDArray[np.generic], label: str) -> NDArray[np.int64]:
	array = np.asarray(values)
	if array.ndim != 1:
		msg = f'{label} must be a 1D label vector; got {array.shape}'
		raise ValueError(msg)
	if not np.issubdtype(array.dtype, np.integer):
		rounded = np.rint(array)
		if not np.array_equal(array, rounded):
			msg = f'{label} must contain integer class ids'
			raise ValueError(msg)
		array = rounded
	return np.asarray(array, dtype=np.int64)


def _class_ids(classes: Sequence[F3ClassInfo]) -> list[int]:
	class_ids = [int(class_info.class_id) for class_info in classes]
	if not class_ids:
		msg = 'classes must contain at least one class'
		raise ValueError(msg)
	if len(set(class_ids)) != len(class_ids):
		msg = f'classes must have unique class ids; got {class_ids!r}'
		raise ValueError(msg)
	return class_ids


def _validated_confusion_matrix(
	matrix: NDArray[np.generic],
	n_classes: int,
) -> NDArray[np.int64]:
	counts = np.asarray(matrix)
	expected_shape = (n_classes, n_classes)
	if counts.shape != expected_shape:
		msg = f'confusion matrix must have shape {expected_shape}; got {counts.shape}'
		raise ValueError(msg)
	if counts.dtype != np.dtype(np.int64):
		msg = 'confusion matrix must have dtype int64'
		raise TypeError(msg)
	if np.any(counts < 0):
		msg = 'confusion matrix counts must be non-negative'
		raise ValueError(msg)
	if not np.any(counts):
		msg = 'metrics require at least one labeled evaluation voxel'
		raise ValueError(msg)
	return counts


def _safe_ratio(
	numerator: NDArray[np.generic],
	denominator: NDArray[np.generic],
) -> NDArray[np.float64]:
	float_numerator = _as_float_array(numerator)
	float_denominator = _as_float_array(denominator)
	return np.divide(
		float_numerator,
		float_denominator,
		out=np.zeros(float_numerator.shape, dtype=np.float64),
		where=float_denominator != 0,
	)


def _as_float_array(values: NDArray[np.generic]) -> NDArray[np.float64]:
	return np.asarray(values, dtype=np.float64)


def _row_normalized_confusion_matrix(
	matrix: NDArray[np.int64],
	exact_row_totals: NDArray[np.generic],
) -> NDArray[np.float64]:
	row_totals = _as_float_array(exact_row_totals)[:, None]
	return np.divide(
		matrix.astype(np.float64),
		row_totals,
		out=np.zeros(matrix.shape, dtype=np.float64),
		where=row_totals != 0,
	)


def _per_class_metric(
	class_ids: Sequence[int],
	values: NDArray[np.float64],
) -> dict[str, float]:
	return {
		str(class_id): float(value)
		for class_id, value in zip(class_ids, values, strict=True)
	}


def _metric_mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		msg = f'{label} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _float_metric(value: object) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'metric value must be numeric; got {value!r}'
		raise TypeError(msg)
	return float(value)


def _format_float(value: float) -> str:
	return f'{value:.4f}'


__all__ = [
	'REQUIRED_LITHOLOGY_METRICS',
	'compute_lithology_metrics',
	'lithology_metrics_from_confusion_matrix',
	'render_classification_report_markdown',
	'write_confusion_matrix_csv',
	'write_metrics_csv',
]
