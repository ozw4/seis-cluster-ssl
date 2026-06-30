"""Apply trained F3 lithology probes to full token embedding volumes."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from seis_ssl_cluster.embedding.sliding_window import token_grid_shape_xyz
from seis_ssl_cluster.f3.lithology_tokens import (
	F3EmbeddingArtifact,
	F3LithologyTokenPolicy,
	load_f3_embedding_artifacts,
	read_f3_lithology_class_info,
	tokenize_f3_lithology_slice,
)
from seis_ssl_cluster.f3.metrics import compute_lithology_metrics
from seis_ssl_cluster.f3.splits import (
	F3LineGeometry,
	F3SliceSplitRecord,
	load_f3_slice_split_records,
	read_f3_line_geometry,
)

if TYPE_CHECKING:
	from pathlib import Path

	from numpy.typing import NDArray

	from seis_ssl_cluster.f3.labels import F3ClassInfo


VALIDATION_SLICE_METRIC_FIELDNAMES = (
	'split',
	'slice_type',
	'slice_index',
	'array_index',
	'token_count',
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'mean_iou',
)


@dataclass(frozen=True)
class F3LithologyPredictionInputs:
	"""Input artifacts for full-volume F3 lithology token prediction."""

	embeddings_dir: Path
	probe_joblib: Path
	scaler_joblib: Path
	label_volume: Path
	class_info: Path
	png_label_inventory: Path
	segy_geometry_json: Path
	source_label_segy: Path | None = None


@dataclass(frozen=True)
class F3LithologyPredictionOutputs:
	"""Output artifacts for full-volume F3 lithology token prediction."""

	output_dir: Path
	token_predictions: Path
	probability_volume: Path
	valid_token_grid: Path
	metadata_json: Path
	validation_slice_metrics_csv: Path


@dataclass(frozen=True)
class F3LithologyPredictionConfig:
	"""Complete F3 lithology token prediction configuration."""

	inputs: F3LithologyPredictionInputs
	outputs: F3LithologyPredictionOutputs
	classes: tuple[F3ClassInfo, ...]
	token_policy: F3LithologyTokenPolicy
	dataset: Mapping[str, object]
	model: Mapping[str, object]
	embeddings: Mapping[str, object]
	labels: Mapping[str, object]
	lithology: Mapping[str, object]
	probe: Mapping[str, object]
	batch_size: int = 4096

	def __post_init__(self) -> None:
		"""Validate runtime controls."""
		if (
			not isinstance(self.batch_size, int)
			or isinstance(self.batch_size, bool)
			or self.batch_size <= 0
		):
			msg = f'batch_size must be a positive integer; got {self.batch_size!r}'
			raise ValueError(msg)


@dataclass(frozen=True)
class F3LithologyPredictionResult:
	"""Paths and counts written by full-volume F3 lithology prediction."""

	token_predictions: Path
	probability_volume: Path
	valid_token_grid: Path
	metadata_json: Path
	validation_slice_metrics_csv: Path
	token_grid_shape_xyz: tuple[int, int, int]
	valid_token_count: int
	invalid_token_count: int
	validation_slice_count: int


def predict_f3_lithology_tokens(
	config: F3LithologyPredictionConfig,
) -> F3LithologyPredictionResult:
	"""Apply a trained probe to every valid F3 token embedding."""
	classes = tuple(config.classes)
	if not classes:
		msg = 'classes must contain at least one F3 class'
		raise ValueError(msg)
	embedding = _single_embedding_artifact(config.inputs.embeddings_dir)
	label_volume = np.load(config.inputs.label_volume, mmap_mode='r')
	geometry = read_f3_line_geometry(config.inputs.segy_geometry_json)
	_validate_prediction_inputs(
		label_volume=label_volume,
		geometry=geometry,
		embedding=embedding,
	)
	probe_model = joblib.load(config.inputs.probe_joblib)
	scaler = joblib.load(config.inputs.scaler_joblib)
	class_ids = _class_ids(classes)
	probe_class_ids = _probe_class_ids(probe_model)
	if probe_class_ids != class_ids:
		msg = (
			'probe class order must match class_info order; '
			f'probe={probe_class_ids!r}, class_info={class_ids!r}'
		)
		raise ValueError(msg)

	predictions, probabilities = _predict_token_grids(
		embedding,
		probe_model=probe_model,
		scaler=scaler,
		class_count=len(classes),
		batch_size=config.batch_size,
	)
	metrics_rows = _validation_slice_metric_rows(
		config,
		label_volume=label_volume,
		geometry=geometry,
		embedding=embedding,
		predictions=predictions,
		classes=classes,
	)
	_write_outputs(
		config,
		embedding=embedding,
		geometry=geometry,
		predictions=predictions,
		probabilities=probabilities,
		metrics_rows=metrics_rows,
	)
	valid_count = int(np.count_nonzero(embedding.valid_tokens))
	return F3LithologyPredictionResult(
		token_predictions=config.outputs.token_predictions,
		probability_volume=config.outputs.probability_volume,
		valid_token_grid=config.outputs.valid_token_grid,
		metadata_json=config.outputs.metadata_json,
		validation_slice_metrics_csv=config.outputs.validation_slice_metrics_csv,
		token_grid_shape_xyz=embedding.token_grid_shape_xyz,
		valid_token_count=valid_count,
		invalid_token_count=int(embedding.valid_tokens.size - valid_count),
		validation_slice_count=len(metrics_rows),
	)


def read_f3_lithology_prediction_classes(
	path: str | Path,
) -> tuple[F3ClassInfo, ...]:
	"""Read class metadata for prediction configs."""
	return read_f3_lithology_class_info(path)


def _single_embedding_artifact(input_dir: Path) -> F3EmbeddingArtifact:
	artifacts = load_f3_embedding_artifacts(input_dir)
	if len(artifacts) != 1:
		msg = (
			'F3 lithology prediction expects exactly one embedding volume; '
			f'got {len(artifacts)} under {input_dir}'
		)
		raise ValueError(msg)
	return artifacts[0]


def _validate_prediction_inputs(
	*,
	label_volume: NDArray[np.generic],
	geometry: F3LineGeometry,
	embedding: F3EmbeddingArtifact,
) -> None:
	if label_volume.ndim != 3:
		msg = f'label volume must be 3D XYZ; got {label_volume.shape!r}'
		raise ValueError(msg)
	if tuple(int(axis) for axis in label_volume.shape) != geometry.shape_xyz:
		msg = (
			'F3 geometry shape does not match label volume; '
			f'geometry={geometry.shape_xyz!r}, label={label_volume.shape!r}'
		)
		raise ValueError(msg)
	expected_grid = token_grid_shape_xyz(geometry.shape_xyz, embedding.patch_size_xyz)
	if expected_grid != embedding.token_grid_shape_xyz:
		msg = (
			'embedding token grid does not match label volume and patch size; '
			f'expected={expected_grid!r}, got={embedding.token_grid_shape_xyz!r}'
		)
		raise ValueError(msg)


def _class_ids(classes: Sequence[F3ClassInfo]) -> tuple[int, ...]:
	return tuple(int(class_info.class_id) for class_info in classes)


def _probe_class_ids(probe_model: object) -> tuple[int, ...]:
	classes = getattr(probe_model, 'classes_', None)
	if classes is None:
		msg = 'probe artifact must expose classes_ for probability column order'
		raise TypeError(msg)
	return tuple(int(value) for value in np.asarray(classes).ravel())


def _predict_token_grids(
	embedding: F3EmbeddingArtifact,
	*,
	probe_model: object,
	scaler: object,
	class_count: int,
	batch_size: int,
) -> tuple[NDArray[np.int16], NDArray[np.float32]]:
	valid_indices = np.argwhere(embedding.valid_tokens)
	predictions = np.full(embedding.token_grid_shape_xyz, -1, dtype=np.int16)
	probabilities = np.full(
		(*embedding.token_grid_shape_xyz, class_count),
		np.nan,
		dtype=np.float32,
	)
	if valid_indices.size == 0:
		return predictions, probabilities
	if not hasattr(scaler, 'transform'):
		msg = 'scaler artifact must expose transform(features)'
		raise TypeError(msg)
	if not hasattr(probe_model, 'predict'):
		msg = 'probe artifact must expose predict(features)'
		raise TypeError(msg)
	if not hasattr(probe_model, 'predict_proba'):
		msg = 'probe artifact must expose predict_proba(features)'
		raise TypeError(msg)

	for start in range(0, int(valid_indices.shape[0]), batch_size):
		batch_indices = valid_indices[start : start + batch_size]
		features = np.asarray(
			embedding.embeddings[
				batch_indices[:, 0],
				batch_indices[:, 1],
				batch_indices[:, 2],
			],
			dtype=np.float32,
		)
		scaled = np.asarray(scaler.transform(features), dtype=np.float32)
		batch_pred = np.asarray(probe_model.predict(scaled), dtype=np.int16)
		batch_prob = np.asarray(probe_model.predict_proba(scaled), dtype=np.float32)
		if batch_prob.shape != (batch_indices.shape[0], class_count):
			msg = (
				'probe probability shape does not match class count; '
				f'got {batch_prob.shape!r}, '
				f'expected={(batch_indices.shape[0], class_count)!r}'
			)
			raise ValueError(msg)
		predictions[
			batch_indices[:, 0],
			batch_indices[:, 1],
			batch_indices[:, 2],
		] = batch_pred
		probabilities[
			batch_indices[:, 0],
			batch_indices[:, 1],
			batch_indices[:, 2],
			:,
		] = batch_prob
	return predictions, probabilities


def _validation_slice_metric_rows(  # noqa: PLR0913
	config: F3LithologyPredictionConfig,
	*,
	label_volume: NDArray[np.generic],
	geometry: F3LineGeometry,
	embedding: F3EmbeddingArtifact,
	predictions: NDArray[np.int16],
	classes: Sequence[F3ClassInfo],
) -> list[dict[str, object]]:
	records = [
		record
		for record in load_f3_slice_split_records(config.inputs.png_label_inventory)
		if record.split == 'validation'
	]
	rows: list[dict[str, object]] = []
	for record in records:
		tokenization = tokenize_f3_lithology_slice(
			record,
			label_volume=label_volume,
			valid_tokens=embedding.valid_tokens,
			geometry=geometry,
			patch_size_xyz=embedding.patch_size_xyz,
			policy=config.token_policy,
			classes=classes,
		)
		pred_plane = _prediction_plane(
			predictions,
			slice_type=tokenization.record.slice_type,
			fixed_token_index=tokenization.tokenization.plane.fixed_token_index,
		)
		if pred_plane.shape != tokenization.usable_mask.shape:
			msg = (
				'prediction plane shape does not match validation tokenization; '
				f'prediction={pred_plane.shape!r}, '
				f'tokenization={tokenization.usable_mask.shape!r}'
			)
			raise ValueError(msg)
		metric_mask = tokenization.usable_mask & (pred_plane >= 0)
		y_true = tokenization.tokenization.majority_class_ids[metric_mask]
		y_pred = pred_plane[metric_mask]
		rows.append(
			_validation_metric_row(
				record,
				array_index=tokenization.array_index,
				y_true=y_true,
				y_pred=y_pred,
				classes=classes,
			),
		)
	return rows


def _prediction_plane(
	predictions: NDArray[np.int16],
	*,
	slice_type: str,
	fixed_token_index: int,
) -> NDArray[np.int16]:
	if slice_type == 'inline':
		return predictions[fixed_token_index, :, :]
	if slice_type == 'crossline':
		return predictions[:, fixed_token_index, :]
	msg = f'slice_type must be inline or crossline; got {slice_type!r}'
	raise ValueError(msg)


def _validation_metric_row(
	record: F3SliceSplitRecord,
	*,
	array_index: int,
	y_true: NDArray[np.generic],
	y_pred: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
) -> dict[str, object]:
	row: dict[str, object] = {
		'split': record.split,
		'slice_type': record.slice_type,
		'slice_index': record.slice_index,
		'array_index': array_index,
		'token_count': int(np.asarray(y_true).shape[0]),
	}
	if row['token_count'] == 0:
		row.update(
			{
				'accuracy': np.nan,
				'balanced_accuracy': np.nan,
				'macro_f1': np.nan,
				'weighted_f1': np.nan,
				'mean_iou': np.nan,
			},
		)
		return row
	metrics = compute_lithology_metrics(y_true, y_pred, classes)
	row.update(
		{
			'accuracy': metrics['accuracy'],
			'balanced_accuracy': metrics['balanced_accuracy'],
			'macro_f1': metrics['macro_f1'],
			'weighted_f1': metrics['weighted_f1'],
			'mean_iou': metrics['mean_iou'],
		},
	)
	return row


def _write_outputs(  # noqa: PLR0913
	config: F3LithologyPredictionConfig,
	*,
	embedding: F3EmbeddingArtifact,
	geometry: F3LineGeometry,
	predictions: NDArray[np.int16],
	probabilities: NDArray[np.float32],
	metrics_rows: Sequence[Mapping[str, object]],
) -> None:
	outputs = config.outputs
	outputs.output_dir.mkdir(parents=True, exist_ok=True)
	np.save(outputs.token_predictions, predictions)
	np.save(outputs.probability_volume, probabilities)
	np.save(
		outputs.valid_token_grid,
		np.asarray(embedding.valid_tokens, dtype=np.bool_),
	)
	_write_validation_metrics_csv(outputs.validation_slice_metrics_csv, metrics_rows)
	_write_json(
		outputs.metadata_json,
		_metadata_payload(
			config,
			embedding=embedding,
			geometry=geometry,
			predictions=predictions,
			probabilities=probabilities,
			metrics_rows=metrics_rows,
		),
	)


def _write_validation_metrics_csv(
	path: Path,
	rows: Sequence[Mapping[str, object]],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=VALIDATION_SLICE_METRIC_FIELDNAMES)
		writer.writeheader()
		writer.writerows(rows)


def _metadata_payload(  # noqa: PLR0913
	config: F3LithologyPredictionConfig,
	*,
	embedding: F3EmbeddingArtifact,
	geometry: F3LineGeometry,
	predictions: NDArray[np.int16],
	probabilities: NDArray[np.float32],
	metrics_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	valid_count = int(np.count_nonzero(embedding.valid_tokens))
	return {
		'artifact_type': 'f3_lithology_token_predictions',
		'dataset': dict(config.dataset),
		'model': dict(config.model),
		'embeddings': dict(config.embeddings),
		'labels': dict(config.labels),
		'lithology': dict(config.lithology),
		'probe': dict(config.probe),
		'label_source_of_truth': 'segy_label_volume',
		'png_label_role': 'train_validation_slice_selection_and_visual_qc',
		'tokenization': config.token_policy.to_dict(),
		'classes': [class_info.to_dict() for class_info in config.classes],
		'class_probability_order': _class_ids(config.classes),
		'invalid_prediction_class_id': -1,
		'invalid_probability_value': 'nan',
		'inputs': {
			'embeddings_dir': str(config.inputs.embeddings_dir),
			'embeddings_path': str(embedding.embeddings_path),
			'valid_tokens_path': str(embedding.valid_tokens_path),
			'embedding_metadata_json': str(embedding.metadata_path),
			'probe_joblib': str(config.inputs.probe_joblib),
			'scaler_joblib': str(config.inputs.scaler_joblib),
			'label_volume': str(config.inputs.label_volume),
			'class_info': str(config.inputs.class_info),
			'png_label_inventory': str(config.inputs.png_label_inventory),
			'segy_geometry_json': str(config.inputs.segy_geometry_json),
			'source_label_segy': (
				None
				if config.inputs.source_label_segy is None
				else str(config.inputs.source_label_segy)
			),
		},
		'embedding': {
			'survey_id': embedding.survey_id,
			'patch_size_xyz': list(embedding.patch_size_xyz),
			'token_grid_shape_xyz': list(embedding.token_grid_shape_xyz),
			'embedding_dim': embedding.embedding_dim,
		},
		'geometry': geometry.to_dict(),
		'outputs': {
			'token_predictions': str(config.outputs.token_predictions),
			'probability_volume': str(config.outputs.probability_volume),
			'valid_token_grid': str(config.outputs.valid_token_grid),
			'metadata_json': str(config.outputs.metadata_json),
			'validation_slice_metrics_csv': str(
				config.outputs.validation_slice_metrics_csv,
			),
		},
		'summary': {
			'token_grid_shape_xyz': [int(axis) for axis in predictions.shape],
			'probability_grid_shape': [int(axis) for axis in probabilities.shape],
			'valid_token_count': valid_count,
			'invalid_token_count': int(embedding.valid_tokens.size - valid_count),
			'validation_slice_count': len(metrics_rows),
		},
		'validation_slice_metrics': [dict(row) for row in metrics_rows],
	}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(
			_json_safe(payload),
			indent=2,
			sort_keys=True,
			allow_nan=False,
		)
		+ '\n',
		encoding='utf-8',
	)


def _json_safe(value: object) -> object:  # noqa: PLR0911
	if isinstance(value, Mapping):
		return {str(key): _json_safe(child) for key, child in value.items()}
	if isinstance(value, tuple | list):
		return [_json_safe(child) for child in value]
	if isinstance(value, np.ndarray):
		return _json_safe(value.tolist())
	if isinstance(value, np.bool_):
		return bool(value)
	if isinstance(value, np.integer):
		return int(value)
	if isinstance(value, np.floating):
		number = float(value)
		return number if np.isfinite(number) else None
	if isinstance(value, float):
		return value if np.isfinite(value) else None
	return value
