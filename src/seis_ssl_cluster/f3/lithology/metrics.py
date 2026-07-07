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
	from sklearn.metrics import (  # noqa: PLC0415
		accuracy_score,
		balanced_accuracy_score,
		confusion_matrix,
		f1_score,
		precision_recall_fscore_support,
	)

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
	precision, recall, f1, support = precision_recall_fscore_support(
		true,
		pred,
		labels=class_ids,
		zero_division=0,
	)
	matrix = confusion_matrix(true, pred, labels=class_ids)
	per_class_iou = _per_class_iou(matrix)
	row_normalized = _row_normalized_confusion_matrix(matrix)
	return {
		'accuracy': float(accuracy_score(true, pred)),
		'balanced_accuracy': float(balanced_accuracy_score(true, pred)),
		'macro_f1': float(
			f1_score(true, pred, labels=class_ids, average='macro', zero_division=0),
		),
		'weighted_f1': float(
			f1_score(
				true,
				pred,
				labels=class_ids,
				average='weighted',
				zero_division=0,
			),
		),
		'per_class_precision': _per_class_metric(class_ids, precision),
		'per_class_recall': _per_class_metric(class_ids, recall),
		'per_class_f1': _per_class_metric(class_ids, f1),
		'per_class_iou': _per_class_metric(class_ids, per_class_iou),
		'per_class_support': {
			str(class_id): int(value)
			for class_id, value in zip(class_ids, support, strict=True)
		},
		'mean_iou': float(np.mean(per_class_iou)),
		'confusion_matrix': matrix.astype(int).tolist(),
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


def _per_class_iou(matrix: NDArray[np.int64]) -> NDArray[np.float64]:
	true_positive = np.diag(matrix).astype(np.float64)
	false_positive = matrix.sum(axis=0).astype(np.float64) - true_positive
	false_negative = matrix.sum(axis=1).astype(np.float64) - true_positive
	denominator = true_positive + false_positive + false_negative
	return np.divide(
		true_positive,
		denominator,
		out=np.zeros_like(true_positive, dtype=np.float64),
		where=denominator != 0,
	)


def _row_normalized_confusion_matrix(matrix: NDArray[np.int64]) -> NDArray[np.float64]:
	row_totals = matrix.sum(axis=1, keepdims=True).astype(np.float64)
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
	'render_classification_report_markdown',
	'write_confusion_matrix_csv',
	'write_metrics_csv',
]
