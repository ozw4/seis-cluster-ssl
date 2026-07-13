"""Numeric evaluation of common V0/V1 F3 voxel prediction artifacts."""

from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.metrics import (
	lithology_metrics_from_confusion_matrix,
	render_classification_report_markdown,
	write_confusion_matrix_csv,
	write_metrics_csv,
)
from seis_ssl_cluster.f3.lithology.voxel_boundary_metrics import (
	build_vertical_boundary_region_masks,
	compute_vertical_boundary_metrics,
)
from seis_ssl_cluster.f3.lithology.voxel_dataset import GRID_NAME, METADATA_NAME
from seis_ssl_cluster.f3.lithology.voxel_metrics import update_confusion_matrix
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	F3VoxelPredictionArtifact,
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_split import VALIDATION_VOXEL_SPLIT
from seis_ssl_cluster.f3.splits import (
	F3LineGeometry,
	F3SliceSplitRecord,
	load_f3_slice_split_records,
	read_f3_line_geometry,
	resolve_f3_slice_array_index,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

	from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
		F3LithologyVoxelEvaluationConfig,
	)
	from seis_ssl_cluster.f3.labels import F3ClassInfo

METRICS_JSON = 'metrics.json'
METRICS_CSV = 'metrics.csv'
CONFUSION_MATRIX_CSV = 'confusion_matrix.csv'
CLASSIFICATION_REPORT_MD = 'classification_report.md'
BOUNDARY_METRICS_JSON = 'boundary_metrics.json'
BOUNDARY_METRICS_CSV = 'boundary_metrics.csv'
BOUNDARY_REGION_METRICS_CSV = 'boundary_region_metrics.csv'
VALIDATION_SLICE_METRICS_CSV = 'validation_slice_metrics.csv'
VALIDATION_TRACE_METRICS_CSV = 'validation_trace_metrics.csv'
EVALUATION_METADATA_JSON = 'evaluation_metadata.json'


@dataclass(frozen=True)
class F3LithologyVoxelEvaluationInspection:
	"""Mmap-backed, identity-validated inputs for an evaluation."""

	prediction_artifact: F3VoxelPredictionArtifact
	split_grid: NDArray[np.integer]
	label_volume: NDArray[np.generic]
	voxel_dataset_metadata: Mapping[str, object]
	classes: tuple[F3ClassInfo, ...]
	geometry: F3LineGeometry
	validation_records: tuple[F3SliceSplitRecord, ...]
	validation_voxel_count: int


@dataclass(frozen=True)
class F3LithologyVoxelEvaluationResult:
	"""Paths and primary count for one completed evaluation."""

	output_dir: Path
	validation_voxel_count: int
	metrics_json: Path
	boundary_metrics_json: Path
	validation_slice_metrics_csv: Path
	validation_trace_metrics_csv: Path
	evaluation_metadata_json: Path


def inspect_f3_lithology_voxel_evaluation(  # noqa: C901
	config: F3LithologyVoxelEvaluationConfig,
) -> F3LithologyVoxelEvaluationInspection:
	"""Load inputs with mmap and reject geometry or source-identity drift."""
	grid_path = config.voxel_dataset_input_dir / GRID_NAME
	metadata_path = config.voxel_dataset_input_dir / METADATA_NAME
	for path in (
		grid_path,
		metadata_path,
		config.source_label_volume,
		config.source_label_segy,
		config.png_label_inventory,
		config.segy_geometry_json,
		config.class_info,
	):
		if not path.is_file():
			raise FileNotFoundError(f'missing voxel evaluation input: {path}')
	prediction = validate_f3_voxel_prediction_artifact(
		config.prediction_input_dir, mmap_mode='r'
	)
	metadata = _read_json_object(metadata_path, 'voxel dataset metadata')
	_validate_supervision_metadata(metadata, config=config)
	grid = np.load(grid_path, mmap_mode='r', allow_pickle=False)
	labels = np.load(config.source_label_volume, mmap_mode='r', allow_pickle=False)
	classes = _read_classes(config.class_info)
	geometry = read_f3_line_geometry(config.segy_geometry_json)
	shape = tuple(cast('Sequence[int]', prediction.metadata['volume_shape_xyz']))
	if grid.ndim != 3 or tuple(grid.shape) != shape:
		raise ValueError(
			'supervision split grid shape does not match prediction volume'
		)
	if not np.issubdtype(grid.dtype, np.integer):
		raise TypeError('supervision split grid must have integer dtype')
	if labels.ndim != 3 or tuple(labels.shape) != shape:
		raise ValueError('source label volume shape does not match prediction volume')
	if tuple(geometry.shape_xyz) != shape:
		raise ValueError('SEGY geometry shape does not match prediction volume')
	_validate_class_and_patch_identity(
		prediction, metadata=metadata, classes=classes
	)
	unknown_monitored = sorted(
		set(config.monitored_class_ids).difference(item.class_id for item in classes)
	)
	if unknown_monitored:
		raise ValueError(
			'monitored_class_ids contain unknown class ids: '
			f'{unknown_monitored!r}'
		)
	_validate_source_identities(
		prediction,
		metadata=metadata,
		grid_path=grid_path,
		metadata_path=metadata_path,
		config=config,
	)
	records = tuple(
		record
		for record in load_f3_slice_split_records(config.png_label_inventory)
		if record.split == 'validation'
	)
	for record in records:
		resolve_f3_slice_array_index(record, geometry)
	validation_count = _validate_prediction_coverage(
		grid,
		prediction.arrays.valid_mask,
		chunk_size_x=config.chunk_size_x,
	)
	if validation_count == 0:
		raise ValueError('voxel evaluation requires at least one validation voxel')
	unknown_codes = _stream_unique_values(grid, chunk_size_x=config.chunk_size_x)
	if not unknown_codes.issubset({0, 1, VALIDATION_VOXEL_SPLIT}):
		raise ValueError(
			f'supervision split grid contains unknown codes: {unknown_codes}'
		)
	return F3LithologyVoxelEvaluationInspection(
		prediction_artifact=prediction,
		split_grid=grid,
		label_volume=labels,
		voxel_dataset_metadata=metadata,
		classes=classes,
		geometry=geometry,
		validation_records=records,
		validation_voxel_count=validation_count,
	)


def evaluate_f3_lithology_voxels(
	config: F3LithologyVoxelEvaluationConfig,
) -> F3LithologyVoxelEvaluationResult:
	"""Evaluate unique validation voxels and write all numeric artifacts."""
	inspection = inspect_f3_lithology_voxel_evaluation(config)
	if config.output_dir.exists() and not config.overwrite:
		raise FileExistsError(
			f'refusing to overwrite existing output: {config.output_dir}'
		)
	config.output_dir.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix=f'.{config.output_dir.name}.staging-', dir=config.output_dir.parent
		)
	)
	try:
		_write_evaluation(staging, config=config, inspection=inspection)
		_commit_directory(staging, config.output_dir, overwrite=config.overwrite)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	return F3LithologyVoxelEvaluationResult(
		output_dir=config.output_dir,
		validation_voxel_count=inspection.validation_voxel_count,
		metrics_json=config.output_dir / METRICS_JSON,
		boundary_metrics_json=config.output_dir / BOUNDARY_METRICS_JSON,
		validation_slice_metrics_csv=(
			config.output_dir / VALIDATION_SLICE_METRICS_CSV
		),
		validation_trace_metrics_csv=(
			config.output_dir / VALIDATION_TRACE_METRICS_CSV
		),
		evaluation_metadata_json=config.output_dir / EVALUATION_METADATA_JSON,
	)


def _write_evaluation(
	output_dir: Path,
	*,
	config: F3LithologyVoxelEvaluationConfig,
	inspection: F3LithologyVoxelEvaluationInspection,
) -> None:
	class_ids = tuple(item.class_id for item in inspection.classes)
	matrix = _aggregate_confusion(
		inspection, class_ids=class_ids, chunk_size_x=config.chunk_size_x
	)
	metrics = lithology_metrics_from_confusion_matrix(matrix, inspection.classes)
	metrics['evaluation_voxel_count'] = inspection.validation_voxel_count
	metrics['aggregation_unit'] = 'unique_validation_voxel'
	_write_json(output_dir / METRICS_JSON, metrics)
	write_metrics_csv(output_dir / METRICS_CSV, metrics, inspection.classes)
	write_confusion_matrix_csv(
		output_dir / CONFUSION_MATRIX_CSV, metrics, inspection.classes
	)
	(output_dir / CLASSIFICATION_REPORT_MD).write_text(
		render_classification_report_markdown(metrics, inspection.classes),
		encoding='utf-8',
	)
	boundary = _aggregate_boundary_metrics(
		inspection,
		tolerances=config.boundary_tolerances,
		monitored_class_ids=config.monitored_class_ids,
	)
	boundary_payload = _with_undefined_reasons(boundary)
	_write_json(output_dir / BOUNDARY_METRICS_JSON, boundary_payload)
	_write_long_metrics_csv(output_dir / BOUNDARY_METRICS_CSV, boundary)
	region_rows = _boundary_region_rows(
		inspection,
		class_ids=class_ids,
		radii=config.boundary_region_radii,
	)
	_write_rows(output_dir / BOUNDARY_REGION_METRICS_CSV, region_rows)
	slice_rows, trace_rows = _slice_and_trace_rows(
		inspection,
		class_ids=class_ids,
		tolerances=config.boundary_tolerances,
		monitored_class_ids=config.monitored_class_ids,
	)
	_write_rows(output_dir / VALIDATION_SLICE_METRICS_CSV, slice_rows)
	_write_rows(output_dir / VALIDATION_TRACE_METRICS_CSV, trace_rows)
	_write_json(
		output_dir / EVALUATION_METADATA_JSON,
		_evaluation_metadata(
			config,
			inspection=inspection,
			slice_count=len(slice_rows),
			trace_count=len(trace_rows),
		),
	)


def _aggregate_confusion(
	inspection: F3LithologyVoxelEvaluationInspection,
	*,
	class_ids: Sequence[int],
	chunk_size_x: int,
) -> NDArray[np.int64]:
	matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
	for start in range(0, inspection.split_grid.shape[0], chunk_size_x):
		stop = min(start + chunk_size_x, inspection.split_grid.shape[0])
		mask = inspection.split_grid[start:stop] == VALIDATION_VOXEL_SPLIT
		update_confusion_matrix(
			matrix,
			inspection.label_volume[start:stop],
			inspection.prediction_artifact.arrays.predictions[start:stop],
			valid_mask=mask,
			class_ids=class_ids,
		)
	return matrix


def _aggregate_boundary_metrics(  # noqa: C901
	inspection: F3LithologyVoxelEvaluationInspection,
	*,
	tolerances: Sequence[int],
	monitored_class_ids: Sequence[int],
) -> dict[str, int | float | None]:
	true_count = pred_count = 0
	matched = dict.fromkeys(tolerances, 0)
	distances_at_max: list[int] = []
	class_true = dict.fromkeys(monitored_class_ids, 0)
	class_matched = {
		(class_id, tolerance): 0
		for class_id in monitored_class_ids
		for tolerance in tolerances
	}
	maximum = max(tolerances)
	for x in range(inspection.split_grid.shape[0]):
		for y in range(inspection.split_grid.shape[1]):
			mask = inspection.split_grid[x, y] == VALIDATION_VOXEL_SPLIT
			if not np.any(mask):
				continue
			true = np.asarray(inspection.label_volume[x, y])
			pred = np.asarray(inspection.prediction_artifact.arrays.predictions[x, y])
			true_boundaries = _trace_boundaries(true, mask)
			pred_boundaries = _trace_boundaries(pred, mask)
			true_count += len(true_boundaries)
			pred_count += len(pred_boundaries)
			for tolerance in tolerances:
				distances = _ordered_match_distances(
					true_boundaries, pred_boundaries, tolerance
				)
				matched[tolerance] += len(distances)
				if tolerance == maximum:
					distances_at_max.extend(distances)
			for class_id in monitored_class_ids:
				selected_true = _class_boundaries(true_boundaries, true, class_id)
				selected_pred = _class_boundaries(pred_boundaries, pred, class_id)
				class_true[class_id] += len(selected_true)
				for tolerance in tolerances:
					class_matched[(class_id, tolerance)] += len(
						_ordered_match_distances(
							selected_true, selected_pred, tolerance
						)
					)
	result: dict[str, int | float | None] = {
		'vertical_boundary_true_count': true_count,
		'vertical_boundary_pred_count': pred_count,
	}
	for tolerance in tolerances:
		count = matched[tolerance]
		result[f'vertical_boundary_matched_count_at_{tolerance}'] = count
		result[f'vertical_boundary_precision_at_{tolerance}'] = _ratio(
			count, pred_count
		)
		result[f'vertical_boundary_recall_at_{tolerance}'] = _ratio(
			count, true_count
		)
		result[f'vertical_boundary_f1_at_{tolerance}'] = (
			None
			if true_count + pred_count == 0
			else 2.0 * count / (true_count + pred_count)
		)
	result[f'vertical_boundary_position_mae_at_{maximum}'] = (
		float(np.mean(distances_at_max)) if distances_at_max else None
	)
	result[f'vertical_boundary_position_median_ae_at_{maximum}'] = (
		float(np.median(distances_at_max)) if distances_at_max else None
	)
	result[f'vertical_boundary_miss_rate_at_{maximum}'] = (
		None
		if true_count == 0
		else (true_count - len(distances_at_max)) / true_count
	)
	for class_id in monitored_class_ids:
		count = class_true[class_id]
		result[f'vertical_boundary_class_{class_id}_true_count'] = count
		for tolerance in tolerances:
			value = class_matched[(class_id, tolerance)]
			result[
				f'vertical_boundary_class_{class_id}_matched_count_at_{tolerance}'
			] = value
			result[
				f'vertical_boundary_class_{class_id}_recall_at_{tolerance}'
			] = _ratio(value, count)
	return result


def _boundary_region_rows(
	inspection: F3LithologyVoxelEvaluationInspection,
	*,
	class_ids: Sequence[int],
	radii: Sequence[int],
) -> list[dict[str, object]]:
	matrices = {
		radius: np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
		for radius in radii
	}
	interior = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
	for x in range(inspection.split_grid.shape[0]):
		for y in range(inspection.split_grid.shape[1]):
			mask = inspection.split_grid[x, y] == VALIDATION_VOXEL_SPLIT
			if not np.any(mask):
				continue
			true = np.asarray(inspection.label_volume[x, y])[None, None, :]
			pred = np.asarray(
				inspection.prediction_artifact.arrays.predictions[x, y]
			)[None, None, :]
			regions, interior_mask = build_vertical_boundary_region_masks(
				true,
				evaluation_mask=mask[None, None, :],
				class_ids=class_ids,
				radii=radii,
			)
			for radius, region in regions.items():
				update_confusion_matrix(
					matrices[radius], true, pred, valid_mask=region, class_ids=class_ids
				)
			update_confusion_matrix(
				interior, true, pred, valid_mask=interior_mask, class_ids=class_ids
			)
	rows = [
		_classification_row(
			matrix, class_ids=class_ids, classes=inspection.classes,
			base={'region': 'boundary', 'radius': radius},
		)
		for radius, matrix in matrices.items()
	]
	rows.append(
		_classification_row(
			interior,
			class_ids=class_ids,
			classes=inspection.classes,
			base={'region': 'interior', 'radius': max(radii)},
		)
	)
	return rows


def _slice_and_trace_rows(
	inspection: F3LithologyVoxelEvaluationInspection,
	*,
	class_ids: Sequence[int],
	tolerances: Sequence[int],
	monitored_class_ids: Sequence[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
	slice_rows: list[dict[str, object]] = []
	trace_rows: list[dict[str, object]] = []
	primary = max(tolerances)
	for record in inspection.validation_records:
		array_index = resolve_f3_slice_array_index(record, inspection.geometry)
		true, pred, split = _slice_arrays(inspection, record, array_index)
		mask = split == VALIDATION_VOXEL_SPLIT
		matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
		update_confusion_matrix(
			matrix, true, pred, valid_mask=mask, class_ids=class_ids
		)
		base = {
			'slice_type': record.slice_type,
			'slice_index': record.slice_index,
			'array_index': array_index,
		}
		row = _classification_row(
			matrix, class_ids=class_ids, classes=inspection.classes, base=base
		)
		boundary = compute_vertical_boundary_metrics(
			true,
			pred,
			evaluation_mask=mask,
			prediction_valid_mask=mask,
			class_ids=class_ids,
			tolerances=tolerances,
			monitored_class_ids=monitored_class_ids,
		)
		row.update(boundary)
		slice_rows.append(row)
		trace_count = (
			true.shape[1] if record.slice_type == 'inline' else true.shape[0]
		)
		for trace_index in range(trace_count):
			trace_true, trace_pred, trace_mask, x, y = _trace_arrays(
				inspection, record, array_index, trace_index
			)
			voxel_count = int(np.count_nonzero(trace_mask))
			if voxel_count == 0:
				continue
			trace_boundary = compute_vertical_boundary_metrics(
				trace_true[None, None, :],
				trace_pred[None, None, :],
				evaluation_mask=trace_mask[None, None, :],
				prediction_valid_mask=trace_mask[None, None, :],
				class_ids=class_ids,
				tolerances=(primary,),
			)
			trace_rows.append(
				{
					'slice_type': record.slice_type,
					'slice_index': record.slice_index,
					'x': inspection.geometry.inline_min + x,
					'y': inspection.geometry.crossline_min + y,
					'voxel_count': voxel_count,
					'accuracy': float(
						np.mean(trace_true[trace_mask] == trace_pred[trace_mask])
					),
					'true_boundary_count': trace_boundary[
						'vertical_boundary_true_count'
					],
					'pred_boundary_count': trace_boundary[
						'vertical_boundary_pred_count'
					],
					'primary_tolerance': primary,
					'matched_boundary_count': trace_boundary[
						f'vertical_boundary_matched_count_at_{primary}'
					],
					'boundary_position_mae': trace_boundary[
						f'vertical_boundary_position_mae_at_{primary}'
					],
					'boundary_position_median_ae': trace_boundary[
						f'vertical_boundary_position_median_ae_at_{primary}'
					],
				}
			)
	return slice_rows, trace_rows


def _classification_row(
	matrix: NDArray[np.int64],
	*,
	class_ids: Sequence[int],
	classes: Sequence[F3ClassInfo],
	base: Mapping[str, object],
) -> dict[str, object]:
	row = dict(base)
	count = int(matrix.sum())
	row['voxel_count'] = count
	if count == 0:
		row.update(
			{
				'accuracy': None,
				'balanced_accuracy': None,
				'macro_f1': None,
				'weighted_f1': None,
				'mean_iou': None,
				'undefined_reason': 'region contains no validation voxels',
			}
		)
		for class_id in class_ids:
			for name in ('precision', 'recall', 'f1', 'iou', 'support'):
				row[f'class_{class_id}_{name}'] = None
		return row
	metrics = lithology_metrics_from_confusion_matrix(matrix, classes)
	for name in (
		'accuracy',
		'balanced_accuracy',
		'macro_f1',
		'weighted_f1',
		'mean_iou',
	):
		row[name] = metrics[name]
	for class_id in class_ids:
		key = str(class_id)
		for name, source in (
			('precision', 'per_class_precision'),
			('recall', 'per_class_recall'),
			('f1', 'per_class_f1'),
			('iou', 'per_class_iou'),
			('support', 'per_class_support'),
		):
			values = cast('Mapping[str, object]', metrics[source])
			row[f'class_{class_id}_{name}'] = values[key]
	row['undefined_reason'] = ''
	return row


def _slice_arrays(
	inspection: F3LithologyVoxelEvaluationInspection,
	record: F3SliceSplitRecord,
	array_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if record.slice_type == 'inline':
		selection = (slice(array_index, array_index + 1), slice(None), slice(None))
	else:
		selection = (slice(None), slice(array_index, array_index + 1), slice(None))
	return (
		np.asarray(inspection.label_volume[selection]),
		np.asarray(inspection.prediction_artifact.arrays.predictions[selection]),
		np.asarray(inspection.split_grid[selection]),
	)


def _trace_arrays(
	inspection: F3LithologyVoxelEvaluationInspection,
	record: F3SliceSplitRecord,
	array_index: int,
	trace_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
	if record.slice_type == 'inline':
		x, y = array_index, trace_index
	else:
		x, y = trace_index, array_index
	return (
		np.asarray(inspection.label_volume[x, y]),
		np.asarray(inspection.prediction_artifact.arrays.predictions[x, y]),
		np.asarray(inspection.split_grid[x, y]) == VALIDATION_VOXEL_SPLIT,
		x,
		y,
	)


def _trace_boundaries(labels: np.ndarray, mask: np.ndarray) -> NDArray[np.int64]:
	interfaces = mask[:-1] & mask[1:] & (labels[:-1] != labels[1:])
	return np.flatnonzero(interfaces).astype(np.int64, copy=False)


def _class_boundaries(
	boundaries: NDArray[np.int64], labels: np.ndarray, class_id: int
) -> NDArray[np.int64]:
	return np.asarray(
		[
			int(z)
			for z in boundaries
			if labels[int(z)] == class_id or labels[int(z) + 1] == class_id
		],
		dtype=np.int64,
	)


def _ordered_match_distances(
	true: NDArray[np.int64], pred: NDArray[np.int64], tolerance: int
) -> list[int]:
	"""Maximize ordered match count, then minimize total absolute error."""
	n_true, n_pred = len(true), len(pred)
	counts = np.zeros((n_true + 1, n_pred + 1), dtype=np.int64)
	costs = np.zeros_like(counts)
	actions = np.zeros((n_true, n_pred), dtype=np.uint8)
	for i in range(n_true - 1, -1, -1):
		for j in range(n_pred - 1, -1, -1):
			options = [
				(counts[i + 1, j], costs[i + 1, j], 2),
				(counts[i, j + 1], costs[i, j + 1], 3),
			]
			distance = abs(int(true[i]) - int(pred[j]))
			if distance <= tolerance:
				options.insert(
					0,
					(counts[i + 1, j + 1] + 1, costs[i + 1, j + 1] + distance, 1)
				)
			best = min(options, key=lambda item: (-int(item[0]), int(item[1])))
			counts[i, j], costs[i, j], actions[i, j] = best
	distances: list[int] = []
	i = j = 0
	while i < n_true and j < n_pred:
		if actions[i, j] == 1:
			distances.append(abs(int(true[i]) - int(pred[j])))
			i += 1
			j += 1
		elif actions[i, j] == 2:
			i += 1
		else:
			j += 1
	return distances


def _validate_prediction_coverage(
	grid: np.ndarray, prediction_valid: np.ndarray, *, chunk_size_x: int
) -> int:
	count = 0
	for start in range(0, grid.shape[0], chunk_size_x):
		stop = min(start + chunk_size_x, grid.shape[0])
		validation = grid[start:stop] == VALIDATION_VOXEL_SPLIT
		missing = validation & ~prediction_valid[start:stop]
		if np.any(missing):
			local = np.argwhere(missing)[0]
			coordinate = (int(local[0]) + start, int(local[1]), int(local[2]))
			raise ValueError(
				'validation supervised voxel is outside the prediction valid mask: '
				f'{coordinate}'
			)
		count += int(np.count_nonzero(validation))
	return count


def _validate_supervision_metadata(
	metadata: Mapping[str, object], *, config: F3LithologyVoxelEvaluationConfig
) -> None:
	if metadata.get('artifact_type') != 'f3_lithology_voxel_supervision':
		raise ValueError('invalid voxel supervision artifact_type')
	if metadata.get('schema_version') != 1:
		raise ValueError('unsupported voxel supervision schema_version')
	if metadata.get('dataset') != dict(config.dataset):
		raise ValueError('prediction/supervision dataset identity mismatch')
	if metadata.get('split_codes') != {
		'unsupervised': 0,
		'train': 1,
		'validation': VALIDATION_VOXEL_SPLIT,
	}:
		raise ValueError('voxel supervision split-code contract mismatch')


def _validate_class_and_patch_identity(
	prediction: F3VoxelPredictionArtifact,
	*,
	metadata: Mapping[str, object],
	classes: Sequence[F3ClassInfo],
) -> None:
	expected_ids = tuple(item.class_id for item in classes)
	prediction_ids = tuple(
		cast('Sequence[int]', prediction.metadata['class_probability_order'])
	)
	if prediction_ids != expected_ids:
		raise ValueError('prediction/supervision class identity mismatch')
	supervision_classes = metadata.get('classes')
	if not isinstance(supervision_classes, Sequence):
		raise TypeError('voxel supervision classes must be a sequence')
	if tuple(_class_id(item) for item in supervision_classes) != expected_ids:
		raise ValueError('prediction/supervision class identity mismatch')
	if tuple(supervision_classes) != tuple(item.to_dict() for item in classes):
		raise ValueError('class_info does not match voxel supervision classes')
	prediction_classes = cast('Sequence[object]', prediction.metadata['classes'])
	for entry, class_info in zip(prediction_classes, classes, strict=True):
		class_entry = _mapping(entry, 'prediction class entry')
		if class_entry.get('class_name') != class_info.class_name:
			raise ValueError('prediction/supervision class identity mismatch')
	reference = _mapping(metadata.get('reference_embedding'), 'reference_embedding')
	patch = reference.get('patch_size')
	if prediction.metadata['patch_size_xyz'] != patch:
		raise ValueError('prediction/supervision patch identity mismatch')
	if prediction.metadata['volume_shape_xyz'] != reference.get('volume_shape_xyz'):
		raise ValueError('prediction/supervision volume identity mismatch')


def _validate_source_identities(
	prediction: F3VoxelPredictionArtifact,
	*,
	metadata: Mapping[str, object],
	grid_path: Path,
	metadata_path: Path,
	config: F3LithologyVoxelEvaluationConfig,
) -> None:
	for key, selected in (
		('label_volume', config.source_label_volume),
		('inventory', config.png_label_inventory),
	):
		_validate_identity(_mapping(metadata.get(key), key), selected, label=key)
	labels = _mapping(metadata.get('labels'), 'labels')
	_assert_same_path(
		labels.get('source_label_segy'),
		config.source_label_segy,
		'source label SEGY',
	)
	_assert_same_path(labels.get('class_info'), config.class_info, 'class info')
	if _mapping(metadata.get('geometry'), 'geometry') != config_geometry(config):
		raise ValueError('prediction/supervision SEGY geometry identity mismatch')

	source_identity = _mapping(
		prediction.metadata.get('source_identity'), 'source_identity'
	)
	artifact_identities = source_identity.get('artifact_identities')
	if isinstance(artifact_identities, Mapping):
		for key, selected in (
			('voxel_dataset_metadata', metadata_path),
			('voxel_split_grid', grid_path),
			('label_volume', config.source_label_volume),
		):
			_validate_identity(
				_mapping(artifact_identities.get(key), key), selected, label=key
			)
		class_identity = source_identity.get('class_info')
		if isinstance(class_identity, Mapping):
			_validate_identity(
				class_identity, config.class_info, label='prediction class_info'
			)
		return
	_token_projection_source_check(prediction, metadata=metadata, config=config)


def _token_projection_source_check(
	prediction: F3VoxelPredictionArtifact,
	*,
	metadata: Mapping[str, object],
	config: F3LithologyVoxelEvaluationConfig,
) -> None:
	files = _mapping(
		_mapping(prediction.metadata.get('source_identity'), 'source_identity').get(
			'token_artifact_files'
		),
		'token_artifact_files',
	)
	for name, value in files.items():
		identity = _mapping(value, f'token artifact {name}')
		path_value = identity.get('path')
		if not isinstance(path_value, str):
			raise TypeError(f'token artifact {name} identity requires a path')
		_validate_identity(identity, Path(path_value), label=f'token artifact {name}')
	prediction_metadata = _mapping(
		files.get('prediction_metadata'), 'prediction_metadata'
	)
	token_metadata = _read_json_object(
		Path(cast('str', prediction_metadata['path'])), 'token prediction metadata'
	)
	if token_metadata.get('dataset') != dict(config.dataset):
		raise ValueError('prediction/supervision source dataset identity mismatch')
	inputs = _mapping(token_metadata.get('inputs'), 'token prediction inputs')
	for key, selected in (
		('label_volume', config.source_label_volume),
		('class_info', config.class_info),
		('png_label_inventory', config.png_label_inventory),
		('segy_geometry_json', config.segy_geometry_json),
		('source_label_segy', config.source_label_segy),
	):
		_assert_same_path(inputs.get(key), selected, f'token source {key}')
	reference = _mapping(metadata.get('reference_embedding'), 'reference_embedding')
	valid_tokens = _mapping(
		metadata.get('reference_valid_tokens'), 'reference_valid_tokens'
	)
	for key, identity in (
		('embedding_metadata_json', reference),
		('valid_tokens_path', valid_tokens),
	):
		_assert_same_path(
			inputs.get(key), Path(cast('str', identity['path'])), key
		)


def config_geometry(config: F3LithologyVoxelEvaluationConfig) -> Mapping[str, object]:
	"""Return the canonical selected line geometry mapping."""
	return read_f3_line_geometry(config.segy_geometry_json).to_dict()


def _read_classes(path: Path) -> tuple[F3ClassInfo, ...]:
	from seis_ssl_cluster.f3.lithology.tokens import (  # noqa: PLC0415
		read_f3_lithology_class_info,
	)

	return read_f3_lithology_class_info(path)


def _class_id(value: object) -> int:
	entry = _mapping(value, 'class entry')
	class_id = entry.get('class_id')
	if not isinstance(class_id, int) or isinstance(class_id, bool):
		raise TypeError('class entry class_id must be an integer')
	return class_id


def _validate_identity(
	identity: Mapping[str, object], path: Path, *, label: str
) -> None:
	_assert_same_path(identity.get('path'), path, label)
	sha256 = identity.get('sha256')
	if not isinstance(sha256, str) or file_sha256(path) != sha256:
		raise ValueError(
			f'prediction/supervision source identity mismatch: {label} hash'
		)


def _assert_same_path(value: object, path: Path, label: str) -> None:
	mismatch = not isinstance(value, str) or Path(value).resolve(
		strict=False
	) != path.resolve(strict=False)
	if mismatch:
		raise ValueError(
			f'prediction/supervision source identity mismatch: {label} path'
		)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return cast('Mapping[str, object]', value)


def _read_json_object(path: Path, label: str) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj, parse_constant=_reject_json_constant)
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a JSON object')
	return cast('Mapping[str, object]', payload)


def _stream_unique_values(grid: np.ndarray, *, chunk_size_x: int) -> set[int]:
	values: set[int] = set()
	for start in range(0, grid.shape[0], chunk_size_x):
		chunk = grid[start : start + chunk_size_x]
		values.update(int(value) for value in np.unique(chunk))
	return values


def _with_undefined_reasons(
	metrics: Mapping[str, int | float | None],
) -> dict[str, object]:
	payload = dict(metrics)
	reasons = {}
	for key, value in metrics.items():
		if value is None:
			reasons[key] = (
				'undefined because the required boundary denominator or '
				'matched-boundary '
				'set is empty'
			)
	payload['undefined_reasons'] = reasons
	return payload


def _evaluation_metadata(
	config: F3LithologyVoxelEvaluationConfig,
	*,
	inspection: F3LithologyVoxelEvaluationInspection,
	slice_count: int,
	trace_count: int,
) -> dict[str, object]:
	return {
		'artifact_type': 'f3_lithology_voxel_evaluation',
		'schema_version': 1,
		'dataset': dict(config.dataset),
		'prediction_kind': inspection.prediction_artifact.metadata['prediction_kind'],
		'model_tag': inspection.prediction_artifact.metadata['model_tag'],
		'aggregation': {
			'primary_unit': 'unique_validation_voxel',
			'split_code': int(VALIDATION_VOXEL_SPLIT),
			'intersection_voxels_counted_once': True,
			'per_slice_planes_evaluated_independently': True,
			'voxel_independence_p_values_computed': False,
		},
		'policy': {
			'monitored_class_ids': list(config.monitored_class_ids),
			'boundary_tolerances': list(config.boundary_tolerances),
			'boundary_region_radii': list(config.boundary_region_radii),
			'primary_trace_boundary_tolerance': max(config.boundary_tolerances),
			'chunk_size_x': config.chunk_size_x,
		},
		'inputs': {
			'prediction_metadata': _identity(
				inspection.prediction_artifact.paths.metadata
			),
			'voxel_dataset_metadata': _identity(
				config.voxel_dataset_input_dir / METADATA_NAME
			),
			'voxel_split_grid': _identity(config.voxel_dataset_input_dir / GRID_NAME),
			'label_volume': _identity(config.source_label_volume),
			'png_label_inventory': _identity(config.png_label_inventory),
			'source_label_segy': _identity(config.source_label_segy),
			'segy_geometry_json': _identity(config.segy_geometry_json),
			'class_info': _identity(config.class_info),
		},
		'summary': {
			'unique_validation_voxel_count': inspection.validation_voxel_count,
			'validation_slice_row_count': slice_count,
			'validation_trace_row_count': trace_count,
		},
		'outputs': {
			name: str(config.output_dir / name)
			for name in (
				METRICS_JSON,
				METRICS_CSV,
				CONFUSION_MATRIX_CSV,
				CLASSIFICATION_REPORT_MD,
				BOUNDARY_METRICS_JSON,
				BOUNDARY_METRICS_CSV,
				BOUNDARY_REGION_METRICS_CSV,
				VALIDATION_SLICE_METRICS_CSV,
				VALIDATION_TRACE_METRICS_CSV,
				EVALUATION_METADATA_JSON,
			)
		},
	}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	_validate_finite_json(payload)
	path.write_text(
		json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _validate_finite_json(value: object) -> None:
	if isinstance(value, float) and not math.isfinite(value):
		raise ValueError('evaluation output contains NaN or Infinity')
	if isinstance(value, Mapping):
		for item in value.values():
			_validate_finite_json(item)
	elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
		for item in value:
			_validate_finite_json(item)


def _write_long_metrics_csv(
	path: Path, metrics: Mapping[str, int | float | None]
) -> None:
	rows = [
		{
			'metric': key,
			'value': '' if value is None else value,
			'undefined_reason': (
				''
				if value is not None
				else 'empty boundary denominator or matched-boundary set'
			),
		}
		for key, value in metrics.items()
	]
	_write_rows(path, rows)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	if not rows:
		path.write_text('', encoding='utf-8')
		return
	fieldnames: list[str] = []
	for row in rows:
		for key in row:
			if key not in fieldnames:
				fieldnames.append(key)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction='raise')
		writer.writeheader()
		writer.writerows(rows)


def _commit_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
	if not target.exists():
		staging.replace(target)
		return
	if not overwrite:
		raise FileExistsError(f'refusing to overwrite existing output: {target}')
	backup = target.with_name(f'.{target.name}.backup')
	if backup.exists():
		shutil.rmtree(backup)
	target.replace(backup)
	try:
		staging.replace(target)
	except BaseException:
		backup.replace(target)
		raise
	shutil.rmtree(backup)


def _ratio(numerator: int, denominator: int) -> float | None:
	return None if denominator == 0 else numerator / denominator


def _reject_json_constant(value: str) -> None:
	raise ValueError(f'non-standard JSON constant: {value}')


__all__ = [
	'F3LithologyVoxelEvaluationInspection',
	'F3LithologyVoxelEvaluationResult',
	'evaluate_f3_lithology_voxels',
	'inspect_f3_lithology_voxel_evaluation',
]
