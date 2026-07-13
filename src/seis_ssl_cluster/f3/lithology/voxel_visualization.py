"""Selected-slice figures for F3 voxel lithology predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.f3.lithology.voxel_split import VALIDATION_VOXEL_SPLIT
from seis_ssl_cluster.f3.splits import (
	F3LineGeometry,
	F3SliceSplitRecord,
	resolve_f3_slice_array_index,
)
from seis_ssl_cluster.visualization.facies import (
	facies_legend_handles,
	label_imshow,
)
from seis_ssl_cluster.visualization.seismic import seismic_imshow
from seis_ssl_cluster.visualization.style import aspect_for_view, origin_for_view

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence
	from pathlib import Path

	from numpy.typing import NDArray

	from seis_ssl_cluster.f3.labels import F3ClassInfo

_UNSUPERVISED_RGB = (226, 226, 226)
_INVALID_RGB = (240, 191, 0)
_CORRECT_RGB = (0, 0, 0)
_ERROR_RGB = (213, 94, 0)


@dataclass(frozen=True)
class F3LithologyVoxelFigureConfig:
	"""Rendering controls shared by all selected voxel slices."""

	dpi: int = 150
	include_confidence: bool = False
	amplitude_clip_percentiles: tuple[float, float] = (1.0, 99.0)

	def __post_init__(self) -> None:
		"""Reject settings that violate the common F3 figure conventions."""
		if not isinstance(self.dpi, int) or isinstance(self.dpi, bool) or self.dpi <= 0:
			raise ValueError('dpi must be a positive integer')
		if not isinstance(self.include_confidence, bool):
			raise TypeError('include_confidence must be boolean')
		low, high = self.amplitude_clip_percentiles
		if not 0.0 <= float(low) < float(high) <= 100.0:
			raise ValueError(
				'amplitude_clip_percentiles must satisfy 0 <= low < high <= 100'
			)


_DEFAULT_FIGURE_CONFIG = F3LithologyVoxelFigureConfig()


@dataclass(frozen=True)
class F3LithologyVoxelSlice:
	"""Display-only views for one inline or crossline voxel section."""

	slice_type: str
	slice_index: int
	array_index: int
	seismic: NDArray[np.generic]
	labels: NDArray[np.generic]
	predictions: NDArray[np.generic]
	confidence: NDArray[np.generic]
	supervision_mask: NDArray[np.bool_]
	prediction_valid_mask: NDArray[np.bool_]
	origin: str
	aspect: str
	horizontal_axis: str

	@property
	def evaluation_mask(self) -> NDArray[np.bool_]:
		"""Pixels used by the slice metrics."""
		return self.supervision_mask & self.prediction_valid_mask

	@property
	def error_mask(self) -> NDArray[np.bool_]:
		"""Evaluated pixels whose unsmoothed class prediction is wrong."""
		return self.evaluation_mask & (self.labels != self.predictions)


@dataclass(frozen=True)
class F3LithologyVoxelVisualizationResult:
	"""Selected-slice PNGs produced by voxel visualization."""

	png_paths: tuple[Path, ...]


def prepare_f3_lithology_voxel_slice(  # noqa: PLR0913
	slice_type: str,
	slice_index: int,
	*,
	seismic: NDArray[np.generic],
	labels: NDArray[np.generic],
	predictions: NDArray[np.generic],
	confidence: NDArray[np.generic],
	split_grid: NDArray[np.generic],
	prediction_valid_mask: NDArray[np.generic],
	geometry: F3LineGeometry,
) -> F3LithologyVoxelSlice:
	"""Return orientation-correct views without smoothing or mutating inputs."""
	arrays = tuple(
		np.asarray(value)
		for value in (
			seismic,
			labels,
			predictions,
			confidence,
			split_grid,
			prediction_valid_mask,
		)
	)
	if any(array.ndim != 3 for array in arrays):
		raise ValueError('voxel visualization inputs must be 3D XYZ arrays')
	if any(array.shape != arrays[0].shape for array in arrays[1:]):
		raise ValueError('voxel visualization input shapes must match')
	if tuple(geometry.shape_xyz) != arrays[0].shape:
		raise ValueError('voxel visualization geometry shape must match inputs')
	if arrays[-1].dtype != np.dtype(bool):
		raise TypeError('prediction_valid_mask must have boolean dtype')
	record = F3SliceSplitRecord(
		relative_path=f'{slice_type}_{slice_index}.png',
		split='validation',
		slice_type=slice_type,
		slice_index=slice_index,
	)
	array_index = resolve_f3_slice_array_index(record, geometry)
	if slice_type == 'inline':
		selection = (array_index, slice(None), slice(None))
		horizontal_axis = 'crossline index'
	elif slice_type == 'crossline':
		selection = (slice(None), array_index, slice(None))
		horizontal_axis = 'inline index'
	else:
		raise ValueError('slice_type must be inline or crossline')

	def display(array: np.ndarray) -> np.ndarray:
		# The copy prevents plotting helpers or callers from changing mmap inputs.
		return np.array(array[selection], copy=True).T

	return F3LithologyVoxelSlice(
		slice_type=slice_type,
		slice_index=slice_index,
		array_index=array_index,
		seismic=display(arrays[0]),
		labels=display(arrays[1]),
		predictions=display(arrays[2]),
		confidence=display(arrays[3]),
		supervision_mask=(display(arrays[4]) == VALIDATION_VOXEL_SPLIT),
		prediction_valid_mask=display(arrays[5]).astype(bool, copy=False),
		origin=origin_for_view(slice_type),
		aspect=aspect_for_view(slice_type),
		horizontal_axis=horizontal_axis,
	)


def save_f3_lithology_voxel_slice_figure(
	figure: F3LithologyVoxelSlice,
	path: Path,
	*,
	classes: Sequence[F3ClassInfo | Mapping[str, object]],
	config: F3LithologyVoxelFigureConfig = _DEFAULT_FIGURE_CONFIG,
	slice_metrics: Mapping[str, object] | None = None,
) -> Path:
	"""Write seismic, labels, raw prediction, errors, and boundary overlays."""
	import matplotlib.pyplot as plt  # noqa: PLC0415
	from matplotlib.patches import Patch  # noqa: PLC0415

	panel_count = 6 if config.include_confidence else 5
	canvas, axes = plt.subplots(
		1,
		panel_count,
		figsize=(3.25 * panel_count, 5.2),
		constrained_layout=True,
	)
	seismic_imshow(
		axes[0],
		figure.seismic,
		view=figure.slice_type,
		clip_percentiles=config.amplitude_clip_percentiles,
	)
	axes[0].set_title('seismic amplitude')
	_masked_label_panel(
		axes[1], figure.labels, figure.supervision_mask, classes, figure.slice_type
	)
	axes[1].set_title('SEG-Y ground truth\n(validation supervision)')
	_masked_label_panel(
		axes[2],
		figure.predictions,
		figure.prediction_valid_mask,
		classes,
		figure.slice_type,
	)
	axes[2].set_title('voxel prediction\n(no smoothing)')
	axes[3].imshow(
		_error_rgb(figure),
		origin=figure.origin,
		aspect=figure.aspect,
		interpolation='nearest',
	)
	axes[3].set_title('error mask')
	axes[3].legend(
		handles=[
			Patch(facecolor=np.asarray(rgb) / 255.0, label=label)
			for rgb, label in (
				(_UNSUPERVISED_RGB, 'unsupervised'),
				(_INVALID_RGB, 'invalid prediction'),
				(_CORRECT_RGB, 'correct'),
				(_ERROR_RGB, 'error'),
			)
		],
		loc='upper right',
		fontsize='x-small',
	)
	seismic_imshow(
		axes[4],
		figure.seismic,
		view=figure.slice_type,
		clip_percentiles=config.amplitude_clip_percentiles,
	)
	_overlay_vertical_boundaries(axes[4], figure)
	axes[4].set_title('vertical boundaries\nGT cyan / prediction magenta')
	if config.include_confidence:
		confidence_image = np.ma.masked_where(
			~figure.prediction_valid_mask, figure.confidence
		)
		image = axes[5].imshow(
			confidence_image,
			origin=figure.origin,
			aspect=figure.aspect,
			interpolation='nearest',
			vmin=0.0,
			vmax=1.0,
			cmap='viridis',
		)
		canvas.colorbar(image, ax=axes[5], fraction=0.046)
		axes[5].set_title('confidence')
	for axis in axes:
		axis.set_xlabel(figure.horizontal_axis)
		axis.set_ylabel('sample/time index down')
	if classes:
		canvas.legend(
			handles=facies_legend_handles(classes),
			loc='lower center',
			bbox_to_anchor=(0.5, -0.02),
			ncol=min(4, len(classes)),
		)
	metric_text = _slice_metric_text(slice_metrics)
	canvas.suptitle(
		f'{figure.slice_type} {figure.slice_index} — slice metrics: {metric_text}'
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	canvas.savefig(path, dpi=config.dpi, facecolor='white')
	plt.close(canvas)
	return path


def visualize_f3_lithology_voxel_slices(  # noqa: PLR0913
	*,
	seismic: NDArray[np.generic],
	labels: NDArray[np.generic],
	predictions: NDArray[np.generic],
	confidence: NDArray[np.generic],
	split_grid: NDArray[np.generic],
	prediction_valid_mask: NDArray[np.generic],
	geometry: F3LineGeometry,
	classes: Sequence[F3ClassInfo | Mapping[str, object]],
	slices: Sequence[tuple[str, int]],
	output_dir: Path,
	config: F3LithologyVoxelFigureConfig = _DEFAULT_FIGURE_CONFIG,
	metrics_by_slice: Mapping[tuple[str, int], Mapping[str, object]] | None = None,
) -> F3LithologyVoxelVisualizationResult:
	"""Prepare and save all explicitly selected inline/crossline figures."""
	paths: list[Path] = []
	for slice_type, slice_index in slices:
		figure = prepare_f3_lithology_voxel_slice(
			slice_type,
			slice_index,
			seismic=seismic,
			labels=labels,
			predictions=predictions,
			confidence=confidence,
			split_grid=split_grid,
			prediction_valid_mask=prediction_valid_mask,
			geometry=geometry,
		)
		path = output_dir / f'{slice_type}_{slice_index:04d}_overview.png'
		save_f3_lithology_voxel_slice_figure(
			figure,
			path,
			classes=classes,
			config=config,
			slice_metrics=(metrics_by_slice or {}).get((slice_type, slice_index)),
		)
		paths.append(path)
	return F3LithologyVoxelVisualizationResult(tuple(paths))


def _masked_label_panel(
	ax: object,
	values: np.ndarray,
	mask: np.ndarray,
	classes: Sequence[F3ClassInfo | Mapping[str, object]],
	view: str,
) -> None:
	display = np.where(mask, values, -1)
	label_imshow(ax, display, classes=classes, view=view)


def _error_rgb(figure: F3LithologyVoxelSlice) -> np.ndarray:
	rgb = np.full((*figure.labels.shape, 3), _UNSUPERVISED_RGB, dtype=np.uint8)
	invalid = figure.supervision_mask & ~figure.prediction_valid_mask
	rgb[invalid] = _INVALID_RGB
	rgb[figure.evaluation_mask] = _CORRECT_RGB
	rgb[figure.error_mask] = _ERROR_RGB
	return rgb


def _vertical_boundaries(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
	boundaries = np.zeros(values.shape, dtype=bool)
	boundaries[:-1] = mask[:-1] & mask[1:] & (values[:-1] != values[1:])
	return boundaries


def _overlay_vertical_boundaries(ax: object, figure: F3LithologyVoxelSlice) -> None:
	gt = _vertical_boundaries(figure.labels, figure.supervision_mask)
	pred = _vertical_boundaries(
		figure.predictions, figure.prediction_valid_mask & figure.supervision_mask
	)
	for boundary, color, marker, label in (
		(gt, '#00bcd4', '_', 'GT'),
		(pred, '#d81b60', '.', 'prediction'),
	):
		row, column = np.nonzero(boundary)
		ax.scatter(
			column,
			row + 0.5,
			s=7,
			c=color,
			marker=marker,
			linewidths=0.6,
			label=label,
		)
	ax.legend(loc='upper right', fontsize='small')


def _slice_metric_text(metrics: Mapping[str, object] | None) -> str:
	if metrics is None:
		return 'not available'
	parts = []
	for key, label in (('accuracy', 'accuracy'), ('macro_f1', 'macro F1')):
		value = metrics.get(key)
		parts.append(
			f'{label}={float(value):.3f}' if value not in {None, ''} else f'{label}=n/a'
		)
	return ', '.join(parts)


__all__ = [
	'F3LithologyVoxelFigureConfig',
	'F3LithologyVoxelSlice',
	'F3LithologyVoxelVisualizationResult',
	'prepare_f3_lithology_voxel_slice',
	'save_f3_lithology_voxel_slice_figure',
	'visualize_f3_lithology_voxel_slices',
]
