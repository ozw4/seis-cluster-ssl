"""Visualize F3 lithology token predictions on seismic slices."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from seis_ssl_cluster.f3.lithology_tokens import read_f3_lithology_class_info
from seis_ssl_cluster.f3.metrics import compute_lithology_metrics
from seis_ssl_cluster.f3.splits import (
	F3LineGeometry,
	F3SliceSplitRecord,
	load_f3_slice_split_records,
	read_f3_line_geometry,
	resolve_f3_slice_array_index,
)
from seis_ssl_cluster.f3.visualization import class_id_image_to_rgb

if TYPE_CHECKING:
	from pathlib import Path

	from numpy.typing import NDArray

	from seis_ssl_cluster.f3.labels import F3ClassInfo


_INVALID_LABEL_RGB = (226, 226, 226)
_ERROR_INVALID_RGB = (226, 226, 226)
_ERROR_CORRECT_RGB = (0, 0, 0)
_ERROR_WRONG_RGB = (213, 94, 0)


@dataclass(frozen=True)
class F3LithologyVisualizationInputs:
	"""Input artifacts for F3 lithology prediction visualization."""

	seismic_volume: Path
	label_volume: Path
	class_info: Path
	png_label_inventory: Path
	segy_geometry_json: Path
	token_predictions: Path
	probability_volume: Path
	prediction_metadata_json: Path
	validation_slice_metrics_csv: Path | None = None


@dataclass(frozen=True)
class F3LithologyVisualizationOutputs:
	"""Output locations for F3 lithology prediction figures."""

	output_dir: Path
	metadata_json: Path
	selected_slices_dir: Path


@dataclass(frozen=True)
class F3LithologyVisualizationFigureConfig:
	"""Rendering controls for publication-oriented lithology figures."""

	dpi: int = 300
	background: str = 'white'
	z_axis: str = 'down'
	include_legend: bool = True
	include_confidence: bool = False
	amplitude_clip_percentiles: tuple[float, float] = (1.0, 99.0)

	def __post_init__(self) -> None:
		"""Validate figure settings."""
		if not isinstance(self.dpi, int) or isinstance(self.dpi, bool) or self.dpi <= 0:
			msg = f'dpi must be a positive integer; got {self.dpi!r}'
			raise ValueError(msg)
		if self.background != 'white':
			msg = 'background must be "white" for F3 lithology figures'
			raise ValueError(msg)
		if self.z_axis != 'down':
			msg = 'z_axis must be "down" for F3 lithology figures'
			raise ValueError(msg)
		if not isinstance(self.include_legend, bool):
			msg = f'include_legend must be boolean; got {self.include_legend!r}'
			raise TypeError(msg)
		if not isinstance(self.include_confidence, bool):
			msg = (
				'include_confidence must be boolean; '
				f'got {self.include_confidence!r}'
			)
			raise TypeError(msg)
		_validate_percentiles(self.amplitude_clip_percentiles)

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable figure config."""
		return {
			'dpi': self.dpi,
			'background': self.background,
			'z_axis': self.z_axis,
			'include_legend': self.include_legend,
			'include_confidence': self.include_confidence,
			'amplitude_clip_percentiles': list(self.amplitude_clip_percentiles),
			'output_formats': ['png'],
		}


@dataclass(frozen=True)
class F3LithologyVisualizationConfig:
	"""Complete F3 lithology prediction visualization configuration."""

	inputs: F3LithologyVisualizationInputs
	outputs: F3LithologyVisualizationOutputs
	classes: tuple[F3ClassInfo, ...]
	dataset: Mapping[str, object]
	model: Mapping[str, object]
	labels: Mapping[str, object]
	lithology: Mapping[str, object]
	probe: Mapping[str, object]
	predictions: Mapping[str, object]
	selected_slices: Mapping[str, tuple[int, ...]]
	figure: F3LithologyVisualizationFigureConfig = field(
		default_factory=F3LithologyVisualizationFigureConfig,
	)


@dataclass(frozen=True)
class F3LithologyVisualizationResult:
	"""Paths written by F3 lithology prediction visualization."""

	png_paths: tuple[Path, ...]
	sidecar_paths: tuple[Path, ...]
	metadata_json: Path


@dataclass(frozen=True)
class F3LithologySliceFigure:
	"""Display-ready arrays and metadata for one lithology prediction slice."""

	slice_type: str
	slice_index: int
	array_index: int
	fixed_token_index: int
	seismic: NDArray[np.generic]
	labels: NDArray[np.int32]
	predictions: NDArray[np.int32]
	confidence: NDArray[np.float32]
	origin: str
	aspect: str
	horizontal_axis: str
	vertical_axis: str

	@property
	def valid_comparison_mask(self) -> NDArray[np.bool_]:
		"""Return pixels with both ground-truth and predicted facies."""
		return (self.labels >= 0) & (self.predictions >= 0)

	@property
	def error_mask(self) -> NDArray[np.bool_]:
		"""Return valid pixels where prediction disagrees with ground truth."""
		return self.valid_comparison_mask & (self.labels != self.predictions)

	def display_metadata(self) -> dict[str, object]:
		"""Return JSON-safe display metadata."""
		return {
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'array_index': self.array_index,
			'fixed_token_index': self.fixed_token_index,
			'shape': [int(axis) for axis in self.seismic.shape],
			'origin': self.origin,
			'aspect': self.aspect,
			'horizontal_axis': self.horizontal_axis,
			'vertical_axis': self.vertical_axis,
			'z_axis': 'down' if 'sample/time' in self.vertical_axis else None,
		}


def visualize_f3_lithology_predictions(
	config: F3LithologyVisualizationConfig,
) -> F3LithologyVisualizationResult:
	"""Write publication-oriented slice figures for F3 lithology predictions."""
	classes = tuple(config.classes)
	if not classes:
		msg = 'classes must contain at least one F3 class'
		raise ValueError(msg)
	seismic = np.load(config.inputs.seismic_volume, mmap_mode='r')
	labels = np.load(config.inputs.label_volume, mmap_mode='r')
	predictions = np.load(config.inputs.token_predictions, mmap_mode='r')
	probabilities = np.load(config.inputs.probability_volume, mmap_mode='r')
	metadata = _read_json(config.inputs.prediction_metadata_json)
	metrics_by_slice = _read_validation_metrics(
		config.inputs.validation_slice_metrics_csv,
	)
	geometry = read_f3_line_geometry(config.inputs.segy_geometry_json)
	patch_size = _metadata_patch_size(metadata)
	_validate_inputs(
		seismic=seismic,
		labels=labels,
		predictions=predictions,
		probabilities=probabilities,
		geometry=geometry,
		patch_size=patch_size,
	)

	records = _validation_records(config.inputs.png_label_inventory)
	entries: list[dict[str, object]] = []
	png_paths: list[Path] = []
	sidecar_paths: list[Path] = []

	config.outputs.output_dir.mkdir(parents=True, exist_ok=True)
	config.outputs.selected_slices_dir.mkdir(parents=True, exist_ok=True)
	for record in records:
		path = config.outputs.output_dir / (
			f'validation_{record.slice_type}_{record.slice_index:04d}_prediction.png'
		)
		figure = _make_slice_figure(
			record.slice_type,
			record.slice_index,
			seismic=seismic,
			labels=labels,
			predictions=predictions,
			probabilities=probabilities,
			geometry=geometry,
			patch_size=patch_size,
		)
		payload = _sidecar_payload(
			config,
			figure,
			path=path,
			group='validation',
			token_metrics=metrics_by_slice.get(
				(record.slice_type, record.slice_index),
			),
		)
		_save_slice_figure(
			figure,
			path,
			payload=payload,
			classes=classes,
			config=config.figure,
		)
		png_paths.append(path)
		sidecar_paths.append(path.with_suffix('.json'))
		entries.append(_metadata_entry(path, path.with_suffix('.json'), payload))

	for slice_type, indices in _selected_slice_items(config.selected_slices):
		for slice_index in indices:
			path = config.outputs.selected_slices_dir / (
				f'selected_{slice_type}_{slice_index:04d}_prediction.png'
			)
			figure = _make_slice_figure(
				slice_type,
				slice_index,
				seismic=seismic,
				labels=labels,
				predictions=predictions,
				probabilities=probabilities,
				geometry=geometry,
				patch_size=patch_size,
			)
			payload = _sidecar_payload(
				config,
				figure,
				path=path,
				group='selected',
				token_metrics=None,
			)
			_save_slice_figure(
				figure,
				path,
				payload=payload,
				classes=classes,
				config=config.figure,
			)
			png_paths.append(path)
			sidecar_paths.append(path.with_suffix('.json'))
			entries.append(_metadata_entry(path, path.with_suffix('.json'), payload))

	_write_json(
		config.outputs.metadata_json,
		_summary_payload(config, prediction_metadata=metadata, entries=entries),
	)
	return F3LithologyVisualizationResult(
		png_paths=tuple(png_paths),
		sidecar_paths=tuple(sidecar_paths),
		metadata_json=config.outputs.metadata_json,
	)


def read_f3_lithology_visualization_classes(
	path: str | Path,
) -> tuple[F3ClassInfo, ...]:
	"""Read class metadata for visualization configs."""
	return read_f3_lithology_class_info(path)


def _validation_records(path: Path) -> tuple[F3SliceSplitRecord, ...]:
	return tuple(
		record
		for record in load_f3_slice_split_records(path)
		if record.split == 'validation'
	)


def _selected_slice_items(
	selected_slices: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
	return tuple(
		(slice_type, tuple(indices))
		for slice_type, indices in selected_slices.items()
		if slice_type in {'inline', 'crossline', 'z'} and indices
	)


def _make_slice_figure(  # noqa: PLR0913
	slice_type: str,
	slice_index: int,
	*,
	seismic: NDArray[np.generic],
	labels: NDArray[np.generic],
	predictions: NDArray[np.generic],
	probabilities: NDArray[np.generic],
	geometry: F3LineGeometry,
	patch_size: tuple[int, int, int],
) -> F3LithologySliceFigure:
	if slice_type == 'inline':
		record = _slice_record(slice_type, slice_index)
		array_index = resolve_f3_slice_array_index(record, geometry)
		fixed_token_index = array_index // patch_size[0]
		prediction_slice = _expand_token_plane(
			predictions[fixed_token_index, :, :],
			row_patch_size=patch_size[1],
			column_patch_size=patch_size[2],
			target_shape=labels[array_index, :, :].shape,
		).T
		confidence_slice = _expand_token_plane(
			_max_probability(probabilities[fixed_token_index, :, :, :]),
			row_patch_size=patch_size[1],
			column_patch_size=patch_size[2],
			target_shape=labels[array_index, :, :].shape,
		).T
		return F3LithologySliceFigure(
			slice_type=slice_type,
			slice_index=slice_index,
			array_index=array_index,
			fixed_token_index=fixed_token_index,
			seismic=np.asarray(seismic[array_index, :, :]).T,
			labels=_normalize_class_ids(labels[array_index, :, :]).T,
			predictions=_normalize_class_ids(prediction_slice),
			confidence=np.asarray(confidence_slice, dtype=np.float32),
			origin='upper',
			aspect='auto',
			horizontal_axis='crossline index',
			vertical_axis='sample/time index down',
		)
	if slice_type == 'crossline':
		record = _slice_record(slice_type, slice_index)
		array_index = resolve_f3_slice_array_index(record, geometry)
		fixed_token_index = array_index // patch_size[1]
		prediction_slice = _expand_token_plane(
			predictions[:, fixed_token_index, :],
			row_patch_size=patch_size[0],
			column_patch_size=patch_size[2],
			target_shape=labels[:, array_index, :].shape,
		).T
		confidence_slice = _expand_token_plane(
			_max_probability(probabilities[:, fixed_token_index, :, :]),
			row_patch_size=patch_size[0],
			column_patch_size=patch_size[2],
			target_shape=labels[:, array_index, :].shape,
		).T
		return F3LithologySliceFigure(
			slice_type=slice_type,
			slice_index=slice_index,
			array_index=array_index,
			fixed_token_index=fixed_token_index,
			seismic=np.asarray(seismic[:, array_index, :]).T,
			labels=_normalize_class_ids(labels[:, array_index, :]).T,
			predictions=_normalize_class_ids(prediction_slice),
			confidence=np.asarray(confidence_slice, dtype=np.float32),
			origin='upper',
			aspect='auto',
			horizontal_axis='inline index',
			vertical_axis='sample/time index down',
		)
	if slice_type == 'z':
		array_index = _validate_z_slice_index(slice_index, labels.shape[2])
		fixed_token_index = array_index // patch_size[2]
		prediction_slice = _expand_token_plane(
			predictions[:, :, fixed_token_index],
			row_patch_size=patch_size[0],
			column_patch_size=patch_size[1],
			target_shape=labels[:, :, array_index].shape,
		).T
		confidence_slice = _expand_token_plane(
			_max_probability(probabilities[:, :, fixed_token_index, :]),
			row_patch_size=patch_size[0],
			column_patch_size=patch_size[1],
			target_shape=labels[:, :, array_index].shape,
		).T
		return F3LithologySliceFigure(
			slice_type=slice_type,
			slice_index=slice_index,
			array_index=array_index,
			fixed_token_index=fixed_token_index,
			seismic=np.asarray(seismic[:, :, array_index]).T,
			labels=_normalize_class_ids(labels[:, :, array_index]).T,
			predictions=_normalize_class_ids(prediction_slice),
			confidence=np.asarray(confidence_slice, dtype=np.float32),
			origin='lower',
			aspect='equal',
			horizontal_axis='inline index',
			vertical_axis='crossline index',
		)
	msg = f'slice type must be inline, crossline, or z; got {slice_type!r}'
	raise ValueError(msg)


def _slice_record(slice_type: str, slice_index: int) -> F3SliceSplitRecord:
	return F3SliceSplitRecord(
		relative_path=f'selected/{slice_type}_{slice_index:04d}.png',
		split='validation',
		slice_type=slice_type,
		slice_index=slice_index,
	)


def _validate_z_slice_index(slice_index: int, z_size: int) -> int:
	index = int(slice_index)
	if index < 0 or index >= z_size:
		msg = f'z slice index out of range: {index}; valid=[0, {z_size - 1}]'
		raise ValueError(msg)
	return index


def _max_probability(probability_plane: NDArray[np.generic]) -> NDArray[np.float32]:
	plane = np.asarray(probability_plane, dtype=np.float32)
	confidence = np.full(plane.shape[:-1], np.nan, dtype=np.float32)
	valid = np.isfinite(plane).any(axis=-1)
	if np.any(valid):
		confidence[valid] = np.nanmax(plane[valid], axis=-1)
	return confidence


def _expand_token_plane(
	token_values: NDArray[np.generic],
	*,
	row_patch_size: int,
	column_patch_size: int,
	target_shape: tuple[int, int],
) -> NDArray[np.generic]:
	values = np.asarray(token_values)
	expanded = np.full(target_shape, _invalid_fill_value(values), dtype=values.dtype)
	for row_token in range(values.shape[0]):
		row_start = row_token * row_patch_size
		row_stop = min(row_start + row_patch_size, target_shape[0])
		for column_token in range(values.shape[1]):
			column_start = column_token * column_patch_size
			column_stop = min(column_start + column_patch_size, target_shape[1])
			expanded[row_start:row_stop, column_start:column_stop] = values[
				row_token,
				column_token,
			]
	return expanded


def _invalid_fill_value(values: NDArray[np.generic]) -> float | int:
	if np.issubdtype(values.dtype, np.floating):
		return np.nan
	return -1


def _save_slice_figure(
	figure: F3LithologySliceFigure,
	path: Path,
	*,
	payload: Mapping[str, object],
	classes: Sequence[F3ClassInfo],
	config: F3LithologyVisualizationFigureConfig,
) -> None:
	plt = _matplotlib_pyplot()
	path.parent.mkdir(parents=True, exist_ok=True)
	panel_count = 5 if config.include_confidence else 4
	fig_width = 4.0 * panel_count
	fig, axes = plt.subplots(
		1,
		panel_count,
		figsize=(fig_width, 4.6),
		dpi=config.dpi,
		sharex=True,
		sharey=True,
		constrained_layout=True,
	)
	axes = np.ravel(axes)
	vmin, vmax = _amplitude_limits(
		figure.seismic,
		config.amplitude_clip_percentiles,
	)
	seismic_image = axes[0].imshow(
		figure.seismic,
		cmap='gray',
		origin=figure.origin,
		aspect=figure.aspect,
		interpolation='none',
		vmin=vmin,
		vmax=vmax,
	)
	axes[0].set_title('seismic amplitude')
	axes[1].imshow(
		class_id_image_to_rgb(
			figure.labels,
			classes,
			invalid_rgb=_INVALID_LABEL_RGB,
		),
		origin=figure.origin,
		aspect=figure.aspect,
		interpolation='nearest',
	)
	axes[1].set_title('ground truth facies')
	axes[2].imshow(
		class_id_image_to_rgb(
			figure.predictions,
			classes,
			invalid_rgb=_INVALID_LABEL_RGB,
		),
		origin=figure.origin,
		aspect=figure.aspect,
		interpolation='nearest',
	)
	axes[2].set_title('predicted facies')
	axes[3].imshow(
		_error_mask_rgb(figure),
		origin=figure.origin,
		aspect=figure.aspect,
		interpolation='nearest',
	)
	axes[3].set_title('error mask')
	if config.include_confidence:
		confidence_image = axes[4].imshow(
			np.ma.masked_invalid(figure.confidence),
			cmap='viridis',
			origin=figure.origin,
			aspect=figure.aspect,
			interpolation='nearest',
			vmin=0.0,
			vmax=1.0,
		)
		axes[4].set_title('max probability')
		confidence_colorbar = fig.colorbar(
			confidence_image,
			ax=axes[4],
			fraction=0.046,
			pad=0.04,
		)
		confidence_colorbar.set_label('probability')
	for ax in axes:
		_configure_axes(ax, figure)
	amplitude_colorbar = fig.colorbar(
		seismic_image,
		ax=axes[0],
		fraction=0.046,
		pad=0.04,
	)
	amplitude_colorbar.set_label('amplitude')
	if config.include_legend:
		axes[2].legend(
			handles=_class_legend_handles(classes),
			loc='center left',
			bbox_to_anchor=(1.02, 0.5),
			frameon=False,
			fontsize=7,
			title='class',
			title_fontsize=8,
		)
		axes[3].legend(
			handles=_error_legend_handles(),
			loc='lower left',
			frameon=False,
			fontsize=7,
		)
	fig.suptitle(f'{figure.slice_type} {figure.slice_index}', fontsize=10)
	fig.savefig(path, facecolor=config.background, bbox_inches='tight')
	plt.close(fig)
	_write_json(path.with_suffix('.json'), payload)


def _sidecar_payload(
	config: F3LithologyVisualizationConfig,
	figure: F3LithologySliceFigure,
	*,
	path: Path,
	group: str,
	token_metrics: Mapping[str, object] | None,
) -> dict[str, object]:
	return {
		'figure_type': 'f3_lithology_prediction_slice',
		'output_path': str(path),
		'group': group,
		'slice_type': figure.slice_type,
		'slice_index': figure.slice_index,
		'array_index': figure.array_index,
		'fixed_token_index': figure.fixed_token_index,
		'display': figure.display_metadata(),
		'probe': dict(config.probe),
		'inputs': {
			'seismic_volume': str(config.inputs.seismic_volume),
			'label_volume': str(config.inputs.label_volume),
			'token_predictions': str(config.inputs.token_predictions),
			'probability_volume': str(config.inputs.probability_volume),
			'prediction_metadata_json': str(config.inputs.prediction_metadata_json),
			'png_label_inventory': str(config.inputs.png_label_inventory),
			'class_info': str(config.inputs.class_info),
		},
		'figure_config': config.figure.to_dict(),
		'class_legend': [
			{
				'class_id': class_info.class_id,
				'class_name': class_info.class_name,
				'rgb': list(class_info.rgb),
			}
			for class_info in config.classes
		],
		'error_legend': {
			'correct_rgb': list(_ERROR_CORRECT_RGB),
			'error_rgb': list(_ERROR_WRONG_RGB),
			'invalid_rgb': list(_ERROR_INVALID_RGB),
		},
		'token_metrics': None if token_metrics is None else dict(token_metrics),
		'voxel_projection_metrics': _voxel_projection_metrics(figure, config.classes),
	}


def _voxel_projection_metrics(
	figure: F3LithologySliceFigure,
	classes: Sequence[F3ClassInfo],
) -> dict[str, object]:
	mask = figure.valid_comparison_mask
	if not np.any(mask):
		return {
			'pixel_count': 0,
			'accuracy': np.nan,
			'balanced_accuracy': np.nan,
			'macro_f1': np.nan,
			'weighted_f1': np.nan,
			'mean_iou': np.nan,
		}
	metrics = compute_lithology_metrics(
		figure.labels[mask],
		figure.predictions[mask],
		classes,
	)
	return {
		'pixel_count': int(np.count_nonzero(mask)),
		'accuracy': metrics['accuracy'],
		'balanced_accuracy': metrics['balanced_accuracy'],
		'macro_f1': metrics['macro_f1'],
		'weighted_f1': metrics['weighted_f1'],
		'mean_iou': metrics['mean_iou'],
	}


def _summary_payload(
	config: F3LithologyVisualizationConfig,
	*,
	prediction_metadata: Mapping[str, object],
	entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	return {
		'artifact_type': 'f3_lithology_prediction_visualizations',
		'dataset': dict(config.dataset),
		'model': dict(config.model),
		'labels': dict(config.labels),
		'lithology': dict(config.lithology),
		'probe': dict(config.probe),
		'predictions': dict(config.predictions),
		'figure_config': config.figure.to_dict(),
		'prediction_metadata_json': str(config.inputs.prediction_metadata_json),
		'prediction_summary': prediction_metadata.get('summary'),
		'outputs': {
			'output_dir': str(config.outputs.output_dir),
			'selected_slices_dir': str(config.outputs.selected_slices_dir),
			'metadata_json': str(config.outputs.metadata_json),
		},
		'figures': list(entries),
	}


def _metadata_entry(
	path: Path,
	sidecar_path: Path,
	payload: Mapping[str, object],
) -> dict[str, object]:
	return {
		'path': str(path),
		'sidecar_path': str(sidecar_path),
		'group': payload['group'],
		'slice_type': payload['slice_type'],
		'slice_index': payload['slice_index'],
	}


def _error_mask_rgb(figure: F3LithologySliceFigure) -> NDArray[np.uint8]:
	rgb = np.full((*figure.labels.shape, 3), _ERROR_INVALID_RGB, dtype=np.uint8)
	rgb[figure.valid_comparison_mask] = _ERROR_CORRECT_RGB
	rgb[figure.error_mask] = _ERROR_WRONG_RGB
	return rgb


def _class_legend_handles(classes: Sequence[F3ClassInfo]) -> list[object]:
	from matplotlib.patches import Patch  # noqa: PLC0415

	return [
		Patch(
			facecolor=np.asarray(class_info.rgb, dtype=np.float32) / 255.0,
			label=f'{class_info.class_id}: {class_info.class_name}',
		)
		for class_info in classes
	]


def _error_legend_handles() -> list[object]:
	from matplotlib.patches import Patch  # noqa: PLC0415

	return [
		Patch(
			facecolor=np.asarray(_ERROR_CORRECT_RGB, dtype=np.float32) / 255.0,
			label='correct',
		),
		Patch(
			facecolor=np.asarray(_ERROR_WRONG_RGB, dtype=np.float32) / 255.0,
			label='error',
		),
	]


def _configure_axes(ax: object, figure: F3LithologySliceFigure) -> None:
	ax.set_xlabel(figure.horizontal_axis)
	ax.set_ylabel(figure.vertical_axis)
	ax.tick_params(labelsize=7)


def _validate_inputs(  # noqa: PLR0913
	*,
	seismic: NDArray[np.generic],
	labels: NDArray[np.generic],
	predictions: NDArray[np.generic],
	probabilities: NDArray[np.generic],
	geometry: F3LineGeometry,
	patch_size: tuple[int, int, int],
) -> None:
	if seismic.ndim != 3:
		msg = f'seismic volume must be 3D XYZ; got {seismic.shape!r}'
		raise ValueError(msg)
	if labels.shape != seismic.shape:
		msg = (
			'seismic and label volumes must have matching XYZ shapes; '
			f'seismic={seismic.shape!r}, label={labels.shape!r}'
		)
		raise ValueError(msg)
	if tuple(int(axis) for axis in labels.shape) != geometry.shape_xyz:
		msg = (
			'F3 geometry shape does not match label volume; '
			f'geometry={geometry.shape_xyz!r}, label={labels.shape!r}'
		)
		raise ValueError(msg)
	expected_grid = tuple(
		int(np.ceil(size / patch))
		for size, patch in zip(geometry.shape_xyz, patch_size, strict=True)
	)
	if tuple(int(axis) for axis in predictions.shape) != expected_grid:
		msg = (
			'prediction token grid does not match volume shape and patch size; '
			f'prediction={predictions.shape!r}, expected={expected_grid!r}'
		)
		raise ValueError(msg)
	if probabilities.shape[:3] != predictions.shape:
		msg = (
			'probability grid spatial shape must match predictions; '
			f'probabilities={probabilities.shape!r}, predictions={predictions.shape!r}'
		)
		raise ValueError(msg)


def _metadata_patch_size(metadata: Mapping[str, object]) -> tuple[int, int, int]:
	embedding = metadata.get('embedding')
	if not isinstance(embedding, Mapping):
		msg = 'prediction metadata must contain an embedding object'
		raise TypeError(msg)
	value = embedding.get('patch_size_xyz')
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		msg = (
			'prediction metadata embedding.patch_size_xyz must be length 3; '
			f'got {value!r}'
		)
		raise TypeError(msg)
	patch = tuple(int(item) for item in value)
	if any(axis <= 0 for axis in patch):
		msg = f'prediction metadata patch sizes must be positive; got {patch!r}'
		raise ValueError(msg)
	return patch


def _normalize_class_ids(values: NDArray[np.generic]) -> NDArray[np.int32]:
	array = np.asarray(values)
	finite = np.isfinite(array)
	rounded = np.rint(array)
	if not np.array_equal(array[finite], rounded[finite]):
		msg = 'class id slices must contain integer-like values'
		raise ValueError(msg)
	ids = np.full(array.shape, -1, dtype=np.int32)
	ids[finite] = rounded[finite].astype(np.int32)
	ids[ids < 0] = -1
	return ids


def _amplitude_limits(
	image: NDArray[np.generic],
	percentiles: tuple[float, float],
) -> tuple[float | None, float | None]:
	_validate_percentiles(percentiles)
	values = np.asarray(image, dtype=np.float64)
	values = values[np.isfinite(values)]
	if values.size == 0:
		return None, None
	vmin, vmax = np.percentile(values, percentiles)
	if np.isclose(vmin, vmax):
		return None, None
	return float(vmin), float(vmax)


def _validate_percentiles(value: tuple[float, float]) -> None:
	if len(value) != 2:
		msg = f'percentiles must contain two values; got {value!r}'
		raise ValueError(msg)
	low, high = (float(value[0]), float(value[1]))
	if not 0.0 <= low < high <= 100.0:
		msg = f'percentiles must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)


def _read_validation_metrics(
	path: Path | None,
) -> dict[tuple[str, int], dict[str, object]]:
	if path is None or not path.is_file():
		return {}
	with path.open(encoding='utf-8', newline='') as file_obj:
		rows = list(csv.DictReader(file_obj))
	return {
		(row['slice_type'], int(row['slice_index'])): dict(row)
		for row in rows
	}


def _read_json(path: Path) -> Mapping[str, Any]:
	with path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if not isinstance(payload, Mapping):
		msg = f'JSON file must contain an object: {path}'
		raise TypeError(msg)
	return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + '\n',
		encoding='utf-8',
	)


def _matplotlib_pyplot() -> object:
	import matplotlib.pyplot as plt  # noqa: PLC0415

	return plt
