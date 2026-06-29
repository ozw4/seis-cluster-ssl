"""Quicklook visualization for the F3 facies benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.f3.labels import VALID_LABEL_SPLITS, F3ClassInfo
from seis_ssl_cluster.f3.png_labels import (
	F3PngLabelInspection,
	PngLabelFileInspection,
	read_png_rgb,
	rgb_to_class_id_map,
)
from seis_ssl_cluster.f3.segy import (
	F3SegyGeometry,
	F3SegyInspection,
	axis_assumption_metadata,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from numpy.typing import NDArray

ORTHOGONAL_VIEWS = ('xy', 'xz', 'yz')

_INVALID_LABEL_RGB = (226, 226, 226)
_ORTHOGONAL_MID_AXIS = {
	'xy': 'z',
	'xz': 'y',
	'yz': 'x',
}
_SLICE_TYPE_ORDER = {'inline': 0, 'crossline': 1}


@dataclass(frozen=True)
class F3QuicklookFigureConfig:
	"""Rendering controls for F3 quicklook figures."""

	dpi: int = 300
	seismic_cmap: str = 'gray'
	amplitude_clip_percentiles: tuple[float, float] = (1.0, 99.0)
	overlay_alpha: float = 0.45
	xz_yz_origin: str = 'upper'

	def __post_init__(self) -> None:
		"""Validate figure settings."""
		if not isinstance(self.dpi, int) or isinstance(self.dpi, bool) or self.dpi <= 0:
			msg = f'dpi must be a positive integer; got {self.dpi!r}'
			raise ValueError(msg)
		_validate_percentiles(self.amplitude_clip_percentiles)
		_validate_alpha(self.overlay_alpha)
		if self.xz_yz_origin != 'upper':
			msg = 'xz_yz_origin must be "upper" for F3 seismic section convention'
			raise ValueError(msg)

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable config payload."""
		return {
			'dpi': self.dpi,
			'seismic_cmap': self.seismic_cmap,
			'amplitude_clip_percentiles': list(self.amplitude_clip_percentiles),
			'overlay_alpha': self.overlay_alpha,
			'xz_yz_origin': self.xz_yz_origin,
		}


@dataclass(frozen=True)
class F3QuicklookOutputConfig:
	"""Destination paths for F3 quicklook artifacts."""

	quicklook_dir: Path
	seismic_dir: Path
	labels_dir: Path
	overlays_dir: Path
	metadata_json: Path


@dataclass(frozen=True)
class F3DisplaySlice:
	"""A display-oriented 2D slice and its axes metadata."""

	image: NDArray[np.generic]
	view: str
	slice_index: int
	origin: str
	aspect: str
	horizontal_axis: str
	vertical_axis: str

	def to_metadata(self) -> dict[str, object]:
		"""Return display metadata for this slice."""
		return {
			'view': self.view,
			'slice_index': self.slice_index,
			'shape': [int(axis) for axis in self.image.shape[:2]],
			'origin': self.origin,
			'aspect': self.aspect,
			'horizontal_axis': self.horizontal_axis,
			'vertical_axis': self.vertical_axis,
		}


@dataclass(frozen=True)
class F3ResolvedLineIndex:
	"""Mapping from an F3 PNG line number to a SEGY cube array index."""

	slice_type: str
	slice_index: int
	array_index: int
	axis_name: str
	axis_count: int
	coordinate_min: int | None
	coordinate_max: int | None
	resolution: str

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable index mapping."""
		return {
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'array_index': self.array_index,
			'axis_name': self.axis_name,
			'axis_count': self.axis_count,
			'coordinate_min': self.coordinate_min,
			'coordinate_max': self.coordinate_max,
			'resolution': self.resolution,
		}


@dataclass(frozen=True)
class F3PngLabelAlignment:
	"""Shape alignment applied to a teacher PNG label before overlay."""

	rgb: NDArray[np.uint8]
	source_shape: tuple[int, int]
	target_shape: tuple[int, int]
	transform: str
	transpose: bool
	flip_vertical: bool = False
	flip_horizontal: bool = False

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable alignment payload."""
		return {
			'source_shape': list(self.source_shape),
			'target_shape': list(self.target_shape),
			'output_shape': [int(axis) for axis in self.rgb.shape[:2]],
			'transform': self.transform,
			'transpose': self.transpose,
			'flip_vertical': self.flip_vertical,
			'flip_horizontal': self.flip_horizontal,
		}


@dataclass(frozen=True)
class F3QuicklookResult:
	"""Paths written by the F3 quicklook generator."""

	png_paths: tuple[Path, ...]
	sidecar_paths: tuple[Path, ...]
	metadata_json: Path


def make_orthogonal_display_slice(
	cube: NDArray[np.generic],
	view: str,
	*,
	slice_index: int | None = None,
	xz_yz_origin: str = 'upper',
) -> F3DisplaySlice:
	"""Return an F3 cube slice in display coordinates."""
	array = _validate_cube(cube, label='cube')
	if view not in ORTHOGONAL_VIEWS:
		msg = f'view must be one of {ORTHOGONAL_VIEWS!r}; got {view!r}'
		raise ValueError(msg)
	if xz_yz_origin != 'upper':
		msg = 'xz_yz_origin must be "upper" for F3 seismic section convention'
		raise ValueError(msg)

	if view == 'xy':
		index = _resolve_default_index(
			slice_index,
			axis_size=array.shape[2],
			label='z',
		)
		return F3DisplaySlice(
			image=np.asarray(array[:, :, index]).T,
			view=view,
			slice_index=index,
			origin='lower',
			aspect='equal',
			horizontal_axis='inline index',
			vertical_axis='crossline index',
		)
	if view == 'xz':
		index = _resolve_default_index(
			slice_index,
			axis_size=array.shape[1],
			label='y',
		)
		return F3DisplaySlice(
			image=np.asarray(array[:, index, :]).T,
			view=view,
			slice_index=index,
			origin=xz_yz_origin,
			aspect='auto',
			horizontal_axis='inline index',
			vertical_axis='sample/time index down',
		)
	index = _resolve_default_index(
		slice_index,
		axis_size=array.shape[0],
		label='x',
	)
	return F3DisplaySlice(
		image=np.asarray(array[index, :, :]).T,
		view=view,
		slice_index=index,
		origin=xz_yz_origin,
		aspect='auto',
		horizontal_axis='crossline index',
		vertical_axis='sample/time index down',
	)


def make_teacher_seismic_display_slice(
	cube: NDArray[np.generic],
	geometry: F3SegyGeometry,
	*,
	slice_type: str,
	slice_index: int,
) -> tuple[F3DisplaySlice, F3ResolvedLineIndex]:
	"""Return the display-oriented seismic slice matching one teacher PNG."""
	array = _validate_cube(cube, label='seismic cube')
	resolved = resolve_teacher_line_index(
		geometry,
		slice_type=slice_type,
		slice_index=slice_index,
	)
	if slice_type == 'inline':
		return (
			F3DisplaySlice(
				image=np.asarray(array[resolved.array_index, :, :]).T,
				view='inline',
				slice_index=slice_index,
				origin='upper',
				aspect='auto',
				horizontal_axis='crossline index',
				vertical_axis='sample/time index down',
			),
			resolved,
		)
	if slice_type == 'crossline':
		return (
			F3DisplaySlice(
				image=np.asarray(array[:, resolved.array_index, :]).T,
				view='crossline',
				slice_index=slice_index,
				origin='upper',
				aspect='auto',
				horizontal_axis='inline index',
				vertical_axis='sample/time index down',
			),
			resolved,
		)
	msg = f'slice_type must be inline or crossline; got {slice_type!r}'
	raise ValueError(msg)


def resolve_teacher_line_index(
	geometry: F3SegyGeometry,
	*,
	slice_type: str,
	slice_index: int,
) -> F3ResolvedLineIndex:
	"""Resolve a teacher PNG inline/crossline number to an array index."""
	if slice_type == 'inline':
		array_index, resolution = _resolve_axis_coordinate(
			slice_index,
			axis_count=geometry.iline_count,
			coordinate_min=geometry.iline_min,
			coordinate_max=geometry.iline_max,
			axis_name='inline',
		)
		return F3ResolvedLineIndex(
			slice_type=slice_type,
			slice_index=slice_index,
			array_index=array_index,
			axis_name='inline',
			axis_count=geometry.iline_count,
			coordinate_min=geometry.iline_min,
			coordinate_max=geometry.iline_max,
			resolution=resolution,
		)
	if slice_type == 'crossline':
		array_index, resolution = _resolve_axis_coordinate(
			slice_index,
			axis_count=geometry.xline_count,
			coordinate_min=geometry.xline_min,
			coordinate_max=geometry.xline_max,
			axis_name='crossline',
		)
		return F3ResolvedLineIndex(
			slice_type=slice_type,
			slice_index=slice_index,
			array_index=array_index,
			axis_name='crossline',
			axis_count=geometry.xline_count,
			coordinate_min=geometry.xline_min,
			coordinate_max=geometry.xline_max,
			resolution=resolution,
		)
	msg = f'slice_type must be inline or crossline; got {slice_type!r}'
	raise ValueError(msg)


def align_png_label_to_seismic_slice(
	rgb: NDArray[np.generic],
	*,
	seismic_shape: tuple[int, int],
) -> F3PngLabelAlignment:
	"""Align a PNG label to a seismic display slice by shape only."""
	label_rgb = np.asarray(rgb)
	if label_rgb.ndim != 3 or label_rgb.shape[2] != 3:
		msg = f'rgb label must have shape H x W x 3; got {label_rgb.shape!r}'
		raise ValueError(msg)
	if len(seismic_shape) != 2:
		msg = f'seismic_shape must contain two axes; got {seismic_shape!r}'
		raise ValueError(msg)
	source_shape = (int(label_rgb.shape[0]), int(label_rgb.shape[1]))
	target_shape = (int(seismic_shape[0]), int(seismic_shape[1]))
	if source_shape == target_shape:
		return F3PngLabelAlignment(
			rgb=label_rgb.astype(np.uint8, copy=False),
			source_shape=source_shape,
			target_shape=target_shape,
			transform='none',
			transpose=False,
		)
	if source_shape == (target_shape[1], target_shape[0]):
		return F3PngLabelAlignment(
			rgb=np.transpose(label_rgb, (1, 0, 2)).astype(np.uint8, copy=False),
			source_shape=source_shape,
			target_shape=target_shape,
			transform='transpose',
			transpose=True,
		)
	msg = (
		'PNG label shape does not match seismic display slice shape; '
		f'png_shape={source_shape!r}, seismic_shape={target_shape!r}'
	)
	raise ValueError(msg)


def class_id_image_to_rgb(
	class_ids: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
	*,
	invalid_rgb: tuple[int, int, int] = _INVALID_LABEL_RGB,
) -> NDArray[np.uint8]:
	"""Render a 2D integer class-ID image using fixed class-info RGB colors."""
	array = np.asarray(class_ids)
	if array.ndim != 2:
		msg = f'class_ids must be 2D; got shape={array.shape!r}'
		raise ValueError(msg)
	ids = _normalize_class_id_image(array)
	rgb = np.full((*ids.shape, 3), invalid_rgb, dtype=np.uint8)
	for class_info in classes:
		rgb[ids == class_info.class_id] = class_info.rgb
	return rgb


def facies_legend_labels(classes: Sequence[F3ClassInfo]) -> tuple[str, ...]:
	"""Return stable legend labels for facies classes."""
	return tuple(f'{item.class_id}: {item.class_name}' for item in classes)


def write_f3_quicklook_outputs(
	segy: F3SegyInspection,
	png_labels: F3PngLabelInspection,
	outputs: F3QuicklookOutputConfig,
	figure_config: F3QuicklookFigureConfig | None = None,
) -> F3QuicklookResult:
	"""Write F3 seismic, label, contact-sheet, and overlay quicklook figures."""
	config = figure_config or F3QuicklookFigureConfig()
	_validate_matching_shapes(segy)
	paths: list[Path] = []
	sidecars: list[Path] = []
	entries: list[dict[str, object]] = []

	outputs.quicklook_dir.mkdir(parents=True, exist_ok=True)
	outputs.seismic_dir.mkdir(parents=True, exist_ok=True)
	outputs.labels_dir.mkdir(parents=True, exist_ok=True)
	outputs.overlays_dir.mkdir(parents=True, exist_ok=True)

	for view in ORTHOGONAL_VIEWS:
		display_slice = make_orthogonal_display_slice(
			segy.seismic.cube,
			view,
			xz_yz_origin=config.xz_yz_origin,
		)
		path = (
			outputs.seismic_dir / f'seismic_{view}_{_ORTHOGONAL_MID_AXIS[view]}_mid.png'
		)
		payload = _seismic_sidecar_payload(
			segy,
			display_slice,
			path=path,
			figure_config=config,
			figure_type='seismic_orthogonal_quicklook',
		)
		_save_seismic_png(display_slice, path, payload=payload, config=config)
		paths.append(path)
		sidecars.append(path.with_suffix('.json'))
		entries.append(_metadata_entry(path, path.with_suffix('.json'), payload))

	for view in ORTHOGONAL_VIEWS:
		display_slice = make_orthogonal_display_slice(
			segy.label.cube,
			view,
			xz_yz_origin=config.xz_yz_origin,
		)
		path = outputs.labels_dir / f'label_{view}_{_ORTHOGONAL_MID_AXIS[view]}_mid.png'
		payload = _label_sidecar_payload(
			segy,
			display_slice,
			path=path,
			figure_config=config,
			figure_type='label_orthogonal_quicklook',
		)
		label_rgb = class_id_image_to_rgb(display_slice.image, segy.classes)
		_save_label_png(
			display_slice,
			label_rgb,
			path,
			payload=payload,
			classes=segy.classes,
			config=config,
		)
		paths.append(path)
		sidecars.append(path.with_suffix('.json'))
		entries.append(_metadata_entry(path, path.with_suffix('.json'), payload))

	for split in sorted(VALID_LABEL_SPLITS):
		path = outputs.labels_dir / f'label_slices_{split}_contact_sheet.png'
		payload = _contact_sheet_payload(
			segy,
			png_labels,
			split=split,
			path=path,
			figure_config=config,
		)
		_save_contact_sheet_png(
			png_labels,
			split=split,
			path=path,
			payload=payload,
			config=config,
		)
		paths.append(path)
		sidecars.append(path.with_suffix('.json'))
		entries.append(_metadata_entry(path, path.with_suffix('.json'), payload))

	for file_result in _sorted_png_label_files(png_labels.files):
		path, payload = _save_teacher_overlay_png(
			segy,
			file_result,
			output_dir=outputs.overlays_dir,
			config=config,
		)
		paths.append(path)
		sidecars.append(path.with_suffix('.json'))
		entries.append(_metadata_entry(path, path.with_suffix('.json'), payload))

	_write_json(
		outputs.metadata_json,
		_summary_payload(segy, png_labels, config, entries),
	)
	return F3QuicklookResult(
		png_paths=tuple(paths),
		sidecar_paths=tuple(sidecars),
		metadata_json=outputs.metadata_json,
	)


def _save_teacher_overlay_png(
	segy: F3SegyInspection,
	file_result: PngLabelFileInspection,
	*,
	output_dir: Path,
	config: F3QuicklookFigureConfig,
) -> tuple[Path, dict[str, object]]:
	if file_result.slice_type is None or file_result.slice_index is None:
		msg = (
			'teacher PNG overlay requires parseable slice_type and slice_index: '
			f'{file_result.relative_path}'
		)
		raise ValueError(msg)
	display_slice, resolved = make_teacher_seismic_display_slice(
		segy.seismic.cube,
		segy.seismic.geometry,
		slice_type=file_result.slice_type,
		slice_index=file_result.slice_index,
	)
	label_rgb = read_png_rgb(file_result.absolute_path)
	alignment = align_png_label_to_seismic_slice(
		label_rgb,
		seismic_shape=tuple(int(axis) for axis in display_slice.image.shape),
	)
	rgb_to_class_id_map(alignment.rgb, segy.classes)
	path = output_dir / (
		f'{file_result.split}_{file_result.slice_type}_'
		f'{file_result.slice_index:04d}_overlay.png'
	)
	payload = _overlay_sidecar_payload(
		segy,
		file_result,
		display_slice,
		resolved,
		alignment,
		path=path,
		figure_config=config,
	)
	_save_overlay_png(
		display_slice,
		alignment.rgb,
		path,
		payload=payload,
		classes=segy.classes,
		config=config,
	)
	return path, payload


def _save_seismic_png(
	display_slice: F3DisplaySlice,
	path: Path,
	*,
	payload: Mapping[str, object],
	config: F3QuicklookFigureConfig,
) -> None:
	plt = _matplotlib_pyplot()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(
		figsize=_single_panel_figsize(display_slice),
		dpi=config.dpi,
		constrained_layout=True,
	)
	vmin, vmax = _amplitude_limits(
		display_slice.image,
		config.amplitude_clip_percentiles,
	)
	image = ax.imshow(
		display_slice.image,
		cmap=config.seismic_cmap,
		origin=display_slice.origin,
		aspect=display_slice.aspect,
		interpolation='none',
		vmin=vmin,
		vmax=vmax,
	)
	ax.set_title(_short_slice_title(display_slice))
	_configure_axes(ax, display_slice)
	colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
	colorbar.set_label('amplitude')
	fig.savefig(path, facecolor='white', bbox_inches='tight')
	plt.close(fig)
	_write_json(path.with_suffix('.json'), payload)


def _save_label_png(  # noqa: PLR0913
	display_slice: F3DisplaySlice,
	label_rgb: NDArray[np.uint8],
	path: Path,
	*,
	payload: Mapping[str, object],
	classes: Sequence[F3ClassInfo],
	config: F3QuicklookFigureConfig,
) -> None:
	plt = _matplotlib_pyplot()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(
		figsize=_single_panel_figsize(display_slice),
		dpi=config.dpi,
		constrained_layout=True,
	)
	ax.imshow(
		label_rgb,
		origin=display_slice.origin,
		aspect=display_slice.aspect,
		interpolation='nearest',
	)
	ax.set_title(_short_slice_title(display_slice))
	_configure_axes(ax, display_slice)
	ax.legend(
		handles=_class_legend_handles(classes),
		loc='center left',
		bbox_to_anchor=(1.02, 0.5),
		frameon=False,
		fontsize=7,
		title='class',
		title_fontsize=8,
	)
	fig.savefig(path, facecolor='white', bbox_inches='tight')
	plt.close(fig)
	_write_json(path.with_suffix('.json'), payload)


def _save_contact_sheet_png(
	png_labels: F3PngLabelInspection,
	*,
	split: str,
	path: Path,
	payload: Mapping[str, object],
	config: F3QuicklookFigureConfig,
) -> None:
	plt = _matplotlib_pyplot()
	files = png_labels.files_for_split(split)
	columns = min(4, max(1, len(files)))
	rows = max(1, math.ceil(len(files) / columns))
	fig_width = max(5.5, columns * 3.0)
	fig_height = max(3.2, rows * 2.7)
	fig, axes = plt.subplots(
		rows,
		columns,
		figsize=(fig_width, fig_height),
		dpi=config.dpi,
		squeeze=False,
	)
	for ax in axes.ravel():
		ax.set_axis_off()
	for ax, file_result in zip(axes.ravel(), files, strict=False):
		rgb = read_png_rgb(file_result.absolute_path)
		ax.imshow(rgb, interpolation='nearest')
		ax.set_title(_png_label_title(file_result), fontsize=8)
		ax.set_xlabel('horizontal pixel')
		ax.set_ylabel('vertical pixel')
		ax.tick_params(labelsize=6)
		ax.set_axis_on()
	fig.suptitle(f'{split} PNG labels', fontsize=10)
	fig.legend(
		handles=_class_legend_handles(png_labels.classes),
		labels=facies_legend_labels(png_labels.classes),
		loc='center left',
		bbox_to_anchor=(1.01, 0.5),
		frameon=False,
		fontsize=7,
		title='class',
		title_fontsize=8,
	)
	fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.96))
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, facecolor='white', bbox_inches='tight')
	plt.close(fig)
	_write_json(path.with_suffix('.json'), payload)


def _save_overlay_png(  # noqa: PLR0913
	display_slice: F3DisplaySlice,
	label_rgb: NDArray[np.uint8],
	path: Path,
	*,
	payload: Mapping[str, object],
	classes: Sequence[F3ClassInfo],
	config: F3QuicklookFigureConfig,
) -> None:
	plt = _matplotlib_pyplot()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(
		1,
		3,
		figsize=(12.0, 4.2),
		dpi=config.dpi,
		sharex=True,
		sharey=True,
		constrained_layout=True,
	)
	vmin, vmax = _amplitude_limits(
		display_slice.image,
		config.amplitude_clip_percentiles,
	)
	seismic_ax, label_ax, overlay_ax = axes
	seismic_image = seismic_ax.imshow(
		display_slice.image,
		cmap=config.seismic_cmap,
		origin=display_slice.origin,
		aspect=display_slice.aspect,
		interpolation='none',
		vmin=vmin,
		vmax=vmax,
	)
	seismic_ax.set_title('seismic')
	label_ax.imshow(
		label_rgb,
		origin=display_slice.origin,
		aspect=display_slice.aspect,
		interpolation='nearest',
	)
	label_ax.set_title('label')
	overlay_ax.imshow(
		display_slice.image,
		cmap=config.seismic_cmap,
		origin=display_slice.origin,
		aspect=display_slice.aspect,
		interpolation='none',
		vmin=vmin,
		vmax=vmax,
	)
	overlay_ax.imshow(
		label_rgb,
		origin=display_slice.origin,
		aspect=display_slice.aspect,
		interpolation='nearest',
		alpha=config.overlay_alpha,
	)
	overlay_ax.set_title('overlay')
	for ax in axes:
		_configure_axes(ax, display_slice)
	colorbar = fig.colorbar(seismic_image, ax=seismic_ax, fraction=0.046, pad=0.04)
	colorbar.set_label('amplitude')
	label_ax.legend(
		handles=_class_legend_handles(classes),
		loc='center left',
		bbox_to_anchor=(1.02, 0.5),
		frameon=False,
		fontsize=7,
		title='class',
		title_fontsize=8,
	)
	fig.suptitle(_short_slice_title(display_slice), fontsize=10)
	fig.savefig(path, facecolor='white', bbox_inches='tight')
	plt.close(fig)
	_write_json(path.with_suffix('.json'), payload)


def _seismic_sidecar_payload(
	segy: F3SegyInspection,
	display_slice: F3DisplaySlice,
	*,
	path: Path,
	figure_config: F3QuicklookFigureConfig,
	figure_type: str,
) -> dict[str, object]:
	return {
		'figure_type': figure_type,
		'output_path': str(path),
		'source_segy': str(segy.seismic.geometry.path),
		'source_png_label': None,
		'slice_type': display_slice.view,
		'slice_index': display_slice.slice_index,
		'axis_mapping': axis_assumption_metadata(),
		'display': display_slice.to_metadata(),
		'origin': display_slice.origin,
		'amplitude_clip_percentiles': list(
			figure_config.amplitude_clip_percentiles,
		),
		'amplitude_clip_values': list(
			_amplitude_limits(
				display_slice.image,
				figure_config.amplitude_clip_percentiles,
			),
		),
		'overlay_alpha': None,
		'class_info_version': 'class_info.json exact RGB',
		'class_info_path': str(segy.class_info_path),
		'dpi': figure_config.dpi,
	}


def _label_sidecar_payload(
	segy: F3SegyInspection,
	display_slice: F3DisplaySlice,
	*,
	path: Path,
	figure_config: F3QuicklookFigureConfig,
	figure_type: str,
) -> dict[str, object]:
	return {
		'figure_type': figure_type,
		'output_path': str(path),
		'source_segy': str(segy.label.geometry.path),
		'source_png_label': None,
		'slice_type': display_slice.view,
		'slice_index': display_slice.slice_index,
		'axis_mapping': axis_assumption_metadata(),
		'display': display_slice.to_metadata(),
		'origin': display_slice.origin,
		'amplitude_clip_percentiles': list(
			figure_config.amplitude_clip_percentiles,
		),
		'overlay_alpha': None,
		'class_info_version': 'class_info.json exact RGB',
		'class_info_path': str(segy.class_info_path),
		'dpi': figure_config.dpi,
	}


def _contact_sheet_payload(
	segy: F3SegyInspection,
	png_labels: F3PngLabelInspection,
	*,
	split: str,
	path: Path,
	figure_config: F3QuicklookFigureConfig,
) -> dict[str, object]:
	files = png_labels.files_for_split(split)
	return {
		'figure_type': 'png_label_contact_sheet',
		'output_path': str(path),
		'source_segy': None,
		'source_png_label': None,
		'source_png_labels': [str(item.absolute_path) for item in files],
		'split': split,
		'slice_type': None,
		'slice_index': None,
		'axis_mapping': axis_assumption_metadata(),
		'origin': 'upper',
		'amplitude_clip_percentiles': list(
			figure_config.amplitude_clip_percentiles,
		),
		'overlay_alpha': None,
		'class_info_version': 'class_info.json exact RGB',
		'class_info_path': str(segy.class_info_path),
		'dpi': figure_config.dpi,
		'class_legend': list(facies_legend_labels(png_labels.classes)),
	}


def _overlay_sidecar_payload(  # noqa: PLR0913
	segy: F3SegyInspection,
	file_result: PngLabelFileInspection,
	display_slice: F3DisplaySlice,
	resolved: F3ResolvedLineIndex,
	alignment: F3PngLabelAlignment,
	*,
	path: Path,
	figure_config: F3QuicklookFigureConfig,
) -> dict[str, object]:
	return {
		'figure_type': 'teacher_slice_seismic_label_overlay',
		'output_path': str(path),
		'source_segy': str(segy.seismic.geometry.path),
		'source_png_label': str(file_result.absolute_path),
		'source_png_label_relative_path': str(file_result.relative_path),
		'split': file_result.split,
		'slice_type': file_result.slice_type,
		'slice_index': file_result.slice_index,
		'segy_line_mapping': resolved.to_dict(),
		'axis_mapping': axis_assumption_metadata(),
		'display': display_slice.to_metadata(),
		'origin': display_slice.origin,
		'amplitude_clip_percentiles': list(
			figure_config.amplitude_clip_percentiles,
		),
		'amplitude_clip_values': list(
			_amplitude_limits(
				display_slice.image,
				figure_config.amplitude_clip_percentiles,
			),
		),
		'overlay_alpha': figure_config.overlay_alpha,
		'class_info_version': 'class_info.json exact RGB',
		'class_info_path': str(segy.class_info_path),
		'label_shape_alignment': alignment.to_dict(),
		'dpi': figure_config.dpi,
	}


def _summary_payload(
	segy: F3SegyInspection,
	png_labels: F3PngLabelInspection,
	config: F3QuicklookFigureConfig,
	entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	return {
		'f3_root': str(segy.f3_root),
		'source_segy': {
			'seismic': str(segy.seismic.geometry.path),
			'label': str(segy.label.geometry.path),
		},
		'class_info_path': str(segy.class_info_path),
		'png_label_file_count': len(png_labels.files),
		'axis_mapping': axis_assumption_metadata(),
		'figure_config': config.to_dict(),
		'outputs': list(entries),
	}


def _metadata_entry(
	path: Path,
	sidecar_path: Path,
	payload: Mapping[str, object],
) -> dict[str, object]:
	return {
		'path': str(path),
		'sidecar_path': str(sidecar_path),
		'figure_type': payload['figure_type'],
		'slice_type': payload.get('slice_type'),
		'slice_index': payload.get('slice_index'),
	}


def _validate_cube(cube: NDArray[np.generic], *, label: str) -> NDArray[np.generic]:
	array = np.asarray(cube)
	if array.ndim != 3:
		msg = (
			f'{label} must be 3D with shape inline x crossline x sample; '
			f'got {array.shape!r}'
		)
		raise ValueError(msg)
	return array


def _validate_matching_shapes(segy: F3SegyInspection) -> None:
	if segy.seismic.cube.shape != segy.label.cube.shape:
		msg = (
			'F3 seismic and dense label cubes must have the same shape; '
			f'seismic={segy.seismic.cube.shape!r}, label={segy.label.cube.shape!r}'
		)
		raise ValueError(msg)


def _resolve_default_index(
	slice_index: int | None,
	*,
	axis_size: int,
	label: str,
) -> int:
	index = axis_size // 2 if slice_index is None else int(slice_index)
	if index < 0 or index >= axis_size:
		msg = f'{label} slice index out of range: {index}; valid=[0, {axis_size - 1}]'
		raise ValueError(msg)
	return index


def _resolve_axis_coordinate(
	value: int,
	*,
	axis_count: int,
	coordinate_min: int | None,
	coordinate_max: int | None,
	axis_name: str,
) -> tuple[int, str]:
	if (
		coordinate_min is not None
		and coordinate_max is not None
		and coordinate_max - coordinate_min + 1 == axis_count
	):
		if coordinate_min <= value <= coordinate_max:
			return value - coordinate_min, 'contiguous_coordinate'
		msg = (
			f'{axis_name} coordinate out of range: {value}; '
			f'valid=[{coordinate_min}, {coordinate_max}]'
		)
		raise ValueError(msg)
	if 0 <= value < axis_count:
		return value, 'array_index'
	msg = f'{axis_name} array index out of range: {value}; valid=[0, {axis_count - 1}]'
	raise ValueError(msg)


def _normalize_class_id_image(values: NDArray[np.generic]) -> NDArray[np.int64]:
	array = np.asarray(values)
	finite_mask = np.isfinite(array)
	rounded = np.rint(array)
	if not np.allclose(array[finite_mask], rounded[finite_mask]):
		msg = 'label class image must contain integer-like values'
		raise ValueError(msg)
	ids = np.full(array.shape, -1, dtype=np.int64)
	ids[finite_mask] = rounded[finite_mask].astype(np.int64)
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


def _validate_alpha(value: float) -> float:
	alpha = float(value)
	if not 0.0 <= alpha <= 1.0:
		msg = f'alpha must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return alpha


def _single_panel_figsize(display_slice: F3DisplaySlice) -> tuple[float, float]:
	if display_slice.view == 'xy':
		return (6.0, 5.6)
	return (6.0, 7.2)


def _short_slice_title(display_slice: F3DisplaySlice) -> str:
	if display_slice.view in ORTHOGONAL_VIEWS:
		axis = _ORTHOGONAL_MID_AXIS[display_slice.view]
		return f'{display_slice.view.upper()} {axis}={display_slice.slice_index}'
	return f'{display_slice.view} {display_slice.slice_index}'


def _configure_axes(ax: object, display_slice: F3DisplaySlice) -> None:
	ax.set_xlabel(display_slice.horizontal_axis)
	ax.set_ylabel(display_slice.vertical_axis)
	ax.tick_params(labelsize=7)


def _png_label_title(file_result: PngLabelFileInspection) -> str:
	slice_type = '?' if file_result.slice_type is None else file_result.slice_type
	slice_index = (
		'?' if file_result.slice_index is None else str(file_result.slice_index)
	)
	return f'{slice_type} {slice_index}'


def _class_legend_handles(classes: Sequence[F3ClassInfo]) -> list[object]:
	patches = __import__('matplotlib.patches', fromlist=['patches'])
	return [
		patches.Patch(
			facecolor=class_info.hex_color,
			edgecolor='#262626',
			linewidth=0.4,
			label=f'{class_info.class_id}: {class_info.class_name}',
		)
		for class_info in classes
	]


def _sorted_png_label_files(
	files: Sequence[PngLabelFileInspection],
) -> tuple[PngLabelFileInspection, ...]:
	return tuple(
		sorted(
			files,
			key=lambda item: (
				0 if item.split == 'train' else 1,
				_SLICE_TYPE_ORDER.get(item.slice_type or '', 2),
				item.slice_index if item.slice_index is not None else 10**12,
				item.relative_path,
			),
		),
	)


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _matplotlib_pyplot() -> object:
	try:
		return __import__('matplotlib.pyplot', fromlist=['pyplot'])
	except ImportError as exc:
		msg = (
			'F3 quicklook visualization requires matplotlib; '
			'install seis-cluster-ssl[visualization].'
		)
		raise ImportError(msg) from exc


__all__ = [
	'ORTHOGONAL_VIEWS',
	'F3DisplaySlice',
	'F3PngLabelAlignment',
	'F3QuicklookFigureConfig',
	'F3QuicklookOutputConfig',
	'F3QuicklookResult',
	'F3ResolvedLineIndex',
	'align_png_label_to_seismic_slice',
	'class_id_image_to_rgb',
	'facies_legend_labels',
	'make_orthogonal_display_slice',
	'make_teacher_seismic_display_slice',
	'resolve_teacher_line_index',
	'write_f3_quicklook_outputs',
]
