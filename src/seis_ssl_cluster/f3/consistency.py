"""Consistency checks between F3 PNG teacher labels and dense SEGY labels."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
from seis_ssl_cluster.f3.visualization import (
	F3ResolvedLineIndex,
	class_id_image_to_rgb,
	resolve_teacher_line_index,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from numpy.typing import NDArray

	from seis_ssl_cluster.f3.labels import F3ClassInfo

CONSISTENCY_CSV_FIELDNAMES = (
	'relative_path',
	'absolute_path',
	'split',
	'slice_type',
	'slice_index',
	'png_label_shape',
	'segy_slice_shape',
	'orientation',
	'matched_pixel_count',
	'mismatch_pixel_count',
	'mismatch_rate',
	'border_mismatch_pixel_count',
	'interior_mismatch_pixel_count',
	'effective_mismatch_rate',
	'border_only_mismatch',
	'unknown_png_pixel_count',
	'unexpected_segy_label_values',
	'exceeds_threshold',
)

_SLICE_TYPE_ORDER = {'inline': 0, 'crossline': 1}
_INVALID_LABEL_RGB = (226, 226, 226)
_MISMATCH_RGB = (215, 25, 28)


@dataclass(frozen=True)
class F3LabelConsistencyFigureConfig:
	"""Rendering controls for F3 label consistency QC figures."""

	dpi: int = 300
	mismatch_rgb: tuple[int, int, int] = _MISMATCH_RGB

	def __post_init__(self) -> None:
		"""Validate figure settings."""
		if not isinstance(self.dpi, int) or isinstance(self.dpi, bool) or self.dpi <= 0:
			msg = f'dpi must be a positive integer; got {self.dpi!r}'
			raise ValueError(msg)
		_normalize_rgb(self.mismatch_rgb, label='mismatch_rgb')

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable figure config."""
		return {
			'dpi': self.dpi,
			'mismatch_rgb': list(self.mismatch_rgb),
			'background': 'white',
		}


@dataclass(frozen=True)
class F3LabelConsistencyOutputConfig:
	"""Destination paths for F3 label consistency artifacts."""

	consistency_dir: Path
	output_json: Path
	output_csv: Path
	report_path: Path


@dataclass(frozen=True)
class F3LabelConsistencyOutputResult:
	"""Paths written by the F3 label consistency artifact writer."""

	metadata_json: Path
	report_csv: Path
	report_markdown: Path
	per_slice_json_paths: tuple[Path, ...]
	figure_paths: tuple[Path, ...]


@dataclass(frozen=True)
class F3LabelSlice:
	"""A raw dense-label slice and its line-index mapping."""

	values: NDArray[np.generic]
	resolved_line: F3ResolvedLineIndex

	def to_metadata(self) -> dict[str, object]:
		"""Return JSON metadata for the extracted dense-label slice."""
		return {
			'shape': [int(axis) for axis in self.values.shape],
			'segy_line_mapping': self.resolved_line.to_dict(),
		}


@dataclass(frozen=True)
class F3LabelConsistencyAlignment:
	"""Shape alignment applied to PNG class IDs before SEGY comparison."""

	class_id_map: NDArray[np.int32]
	source_shape: tuple[int, int]
	target_shape: tuple[int, int]
	transform: str
	transpose: bool

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable alignment payload."""
		return {
			'source_shape': list(self.source_shape),
			'target_shape': list(self.target_shape),
			'output_shape': [int(axis) for axis in self.class_id_map.shape],
			'transform': self.transform,
			'transpose': self.transpose,
		}


@dataclass(frozen=True)
class F3LabelConsistencyRecord:
	"""Per-PNG consistency result against the dense label volume."""

	relative_path: str
	absolute_path: str
	split: str
	slice_type: str
	slice_index: int
	png_label_shape: tuple[int, int]
	segy_slice_shape: tuple[int, int]
	alignment: F3LabelConsistencyAlignment | None
	segy_line_mapping: F3ResolvedLineIndex
	matched_pixel_count: int
	mismatch_pixel_count: int
	mismatch_rate: float | None
	border_mismatch_pixel_count: int
	interior_mismatch_pixel_count: int
	effective_mismatch_rate: float | None
	border_only_mismatch: bool
	unknown_png_pixel_count: int
	unexpected_segy_label_values: tuple[int | float | str, ...]
	exceeds_threshold: bool

	@property
	def orientation(self) -> str:
		"""Return the adopted PNG-to-SEGY orientation decision."""
		if self.alignment is None:
			return 'incompatible_shape'
		return self.alignment.transform

	@property
	def comparable(self) -> bool:
		"""Return whether pixel-wise comparison was possible."""
		return self.alignment is not None

	@property
	def output_prefix(self) -> str:
		"""Return the stable basename prefix used for per-slice artifacts."""
		return f'{self.split}_{self.slice_type}_{self.slice_index:04d}'

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable per-PNG result."""
		return {
			'relative_path': self.relative_path,
			'absolute_path': self.absolute_path,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'png_label_shape': list(self.png_label_shape),
			'segy_slice_shape': list(self.segy_slice_shape),
			'orientation': self.orientation,
			'comparable': self.comparable,
			'label_shape_alignment': (
				None if self.alignment is None else self.alignment.to_dict()
			),
			'segy_line_mapping': self.segy_line_mapping.to_dict(),
			'matched_pixel_count': self.matched_pixel_count,
			'mismatch_pixel_count': self.mismatch_pixel_count,
			'mismatch_rate': self.mismatch_rate,
			'border_mismatch_pixel_count': self.border_mismatch_pixel_count,
			'interior_mismatch_pixel_count': self.interior_mismatch_pixel_count,
			'effective_mismatch_rate': self.effective_mismatch_rate,
			'border_only_mismatch': self.border_only_mismatch,
			'unknown_png_pixel_count': self.unknown_png_pixel_count,
			'unexpected_segy_label_values': list(
				self.unexpected_segy_label_values,
			),
			'exceeds_threshold': self.exceeds_threshold,
		}

	def to_csv_row(self) -> dict[str, object]:
		"""Return the CSV report row for this consistency result."""
		return {
			'relative_path': self.relative_path,
			'absolute_path': self.absolute_path,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'png_label_shape': json.dumps(list(self.png_label_shape)),
			'segy_slice_shape': json.dumps(list(self.segy_slice_shape)),
			'orientation': self.orientation,
			'matched_pixel_count': self.matched_pixel_count,
			'mismatch_pixel_count': self.mismatch_pixel_count,
			'mismatch_rate': '' if self.mismatch_rate is None else self.mismatch_rate,
			'border_mismatch_pixel_count': self.border_mismatch_pixel_count,
			'interior_mismatch_pixel_count': self.interior_mismatch_pixel_count,
			'effective_mismatch_rate': (
				''
				if self.effective_mismatch_rate is None
				else self.effective_mismatch_rate
			),
			'border_only_mismatch': self.border_only_mismatch,
			'unknown_png_pixel_count': self.unknown_png_pixel_count,
			'unexpected_segy_label_values': json.dumps(
				list(self.unexpected_segy_label_values),
			),
			'exceeds_threshold': self.exceeds_threshold,
		}


@dataclass(frozen=True)
class F3LabelConsistencyReport:
	"""Complete PNG-vs-SEGY label consistency report."""

	f3_root: Path
	class_info_path: Path
	source_segy_label: Path
	classes: tuple[F3ClassInfo, ...]
	label_cube: NDArray[np.generic]
	label_geometry: F3SegyGeometry
	max_mismatch_rate: float
	ignore_border_samples_z: int
	records: tuple[F3LabelConsistencyRecord, ...]
	warnings: tuple[str, ...]

	@property
	def passed(self) -> bool:
		"""Return whether every comparable slice is within threshold."""
		return not any(record.exceeds_threshold for record in self.records)

	def total_mismatch_pixel_count(self) -> int:
		"""Return total mismatched pixels across comparable records."""
		return int(sum(record.mismatch_pixel_count for record in self.records))

	def max_observed_mismatch_rate(self) -> float | None:
		"""Return the largest non-null mismatch rate in the report."""
		rates = [
			record.mismatch_rate
			for record in self.records
			if record.mismatch_rate is not None
		]
		if not rates:
			return None
		return float(max(rates))

	def max_observed_effective_mismatch_rate(self) -> float | None:
		"""Return the largest non-null threshold mismatch rate in the report."""
		rates = [
			record.effective_mismatch_rate
			for record in self.records
			if record.effective_mismatch_rate is not None
		]
		if not rates:
			return None
		return float(max(rates))


def extract_teacher_label_slice(
	cube: NDArray[np.generic],
	geometry: F3SegyGeometry,
	*,
	slice_type: str,
	slice_index: int,
) -> F3LabelSlice:
	"""Extract the raw dense-label slice for one teacher PNG line."""
	array = _validate_label_cube(cube)
	resolved = resolve_teacher_line_index(
		geometry,
		slice_type=slice_type,
		slice_index=slice_index,
	)
	if slice_type == 'inline':
		values = np.asarray(array[resolved.array_index, :, :])
	elif slice_type == 'crossline':
		values = np.asarray(array[:, resolved.array_index, :])
	else:
		msg = f'slice_type must be inline or crossline; got {slice_type!r}'
		raise ValueError(msg)
	return F3LabelSlice(values=values, resolved_line=resolved)


def align_png_class_ids_to_segy_slice(
	class_id_map: NDArray[np.integer],
	*,
	segy_slice_shape: tuple[int, int],
) -> F3LabelConsistencyAlignment:
	"""Align PNG class IDs to a raw dense-label slice by shape only."""
	array = np.asarray(class_id_map)
	if array.ndim != 2:
		msg = f'class_id_map must be 2D; got shape={array.shape!r}'
		raise ValueError(msg)
	source_shape = (int(array.shape[0]), int(array.shape[1]))
	target_shape = _normalize_2d_shape(segy_slice_shape, label='segy_slice_shape')
	if source_shape == target_shape:
		return F3LabelConsistencyAlignment(
			class_id_map=array.astype(np.int32, copy=False),
			source_shape=source_shape,
			target_shape=target_shape,
			transform='none',
			transpose=False,
		)
	if source_shape == (target_shape[1], target_shape[0]):
		return F3LabelConsistencyAlignment(
			class_id_map=np.asarray(array.T, dtype=np.int32),
			source_shape=source_shape,
			target_shape=target_shape,
			transform='transpose_png_to_segy',
			transpose=True,
		)
	msg = (
		'PNG label shape does not match dense SEGY label slice shape; '
		f'png_shape={source_shape!r}, segy_slice_shape={target_shape!r}'
	)
	raise ValueError(msg)


def check_f3_label_consistency(
	segy: F3SegyInspection,
	png_labels: F3PngLabelInspection,
	*,
	max_mismatch_rate: float = 0.001,
	ignore_border_samples_z: int = 0,
) -> F3LabelConsistencyReport:
	"""Compare every F3 teacher PNG label with the dense SEGY label volume."""
	threshold = _validate_mismatch_rate_threshold(max_mismatch_rate)
	border_samples = _validate_ignore_border_samples_z(ignore_border_samples_z)
	_validate_class_info_matches(segy.classes, png_labels.classes)
	records = tuple(
		_compare_png_label_file(
			segy,
			file_result,
			max_mismatch_rate=threshold,
			ignore_border_samples_z=border_samples,
		)
		for file_result in _sorted_png_label_files(png_labels.files)
	)
	return F3LabelConsistencyReport(
		f3_root=segy.f3_root,
		class_info_path=segy.class_info_path,
		source_segy_label=segy.label.geometry.path,
		classes=tuple(segy.classes),
		label_cube=segy.label.cube,
		label_geometry=segy.label.geometry,
		max_mismatch_rate=threshold,
		ignore_border_samples_z=border_samples,
		records=records,
		warnings=(*png_labels.warnings, *_report_warnings(records)),
	)


def label_consistency_report_to_dict(
	report: F3LabelConsistencyReport,
) -> dict[str, object]:
	"""Return a machine-readable label consistency report."""
	return {
		'f3_root': str(report.f3_root),
		'class_info_path': str(report.class_info_path),
		'source_segy_label': str(report.source_segy_label),
		'axis_mapping': axis_assumption_metadata(),
		'max_mismatch_rate': report.max_mismatch_rate,
		'ignore_border_samples_z': report.ignore_border_samples_z,
		'passed': report.passed,
		'png_label_file_count': len(report.records),
		'total_mismatch_pixel_count': report.total_mismatch_pixel_count(),
		'max_observed_mismatch_rate': report.max_observed_mismatch_rate(),
		'max_observed_effective_mismatch_rate': (
			report.max_observed_effective_mismatch_rate()
		),
		'warnings': list(report.warnings),
		'files': [record.to_dict() for record in report.records],
	}


def render_label_consistency_markdown(
	report: F3LabelConsistencyReport,
) -> str:
	"""Render a Markdown summary for the F3 label consistency report."""
	status = 'PASS' if report.passed else 'FAIL'
	max_rate = report.max_observed_mismatch_rate()
	lines = [
		'# F3 label consistency report',
		'',
		f'- status: {status}',
		f'- F3 root: `{report.f3_root}`',
		f'- class_info: `{report.class_info_path}`',
		f'- SEGY label volume: `{report.source_segy_label}`',
		f'- PNG labels: {len(report.records)}',
		f'- max mismatch threshold: {report.max_mismatch_rate}',
		f'- ignored z-border samples: {report.ignore_border_samples_z}',
		f'- max observed mismatch rate: {max_rate}',
		(
			'- max observed effective mismatch rate: '
			f'{report.max_observed_effective_mismatch_rate()}'
		),
		f'- total mismatch pixels: {report.total_mismatch_pixel_count()}',
		'',
		'## Per-slice results',
		'',
		(
			'| split | slice | orientation | png_shape | segy_shape | '
			'matched | mismatched | raw_mismatch_rate | effective_mismatch_rate | '
			'border_only_mismatch | border_mismatch_pixel_count | '
			'interior_mismatch_pixel_count | '
			'unknown_png | unexpected_segy | threshold |'
		),
		'|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|',
	]
	lines.extend(_render_record_row(record) for record in report.records)
	lines.extend(['', '## Warnings', ''])
	if report.warnings:
		lines.extend(f'- {warning}' for warning in report.warnings)
	else:
		lines.append('- none')
	return '\n'.join(lines) + '\n'


def write_f3_label_consistency_outputs(
	report: F3LabelConsistencyReport,
	outputs: F3LabelConsistencyOutputConfig,
	figure_config: F3LabelConsistencyFigureConfig | None = None,
) -> F3LabelConsistencyOutputResult:
	"""Write CSV, JSON, Markdown, and PNG consistency artifacts."""
	config = figure_config or F3LabelConsistencyFigureConfig()
	outputs.consistency_dir.mkdir(parents=True, exist_ok=True)
	_write_json(outputs.output_json, label_consistency_report_to_dict(report))
	_write_csv(outputs.output_csv, report.records)
	_write_text(outputs.report_path, render_label_consistency_markdown(report))

	per_slice_json_paths: list[Path] = []
	figure_paths: list[Path] = []
	for record in report.records:
		paths = _write_record_artifacts(
			report,
			record,
			output_dir=outputs.consistency_dir,
			figure_config=config,
		)
		per_slice_json_paths.append(paths['json'])
		figure_paths.extend(
			(paths['png_label'], paths['segy_label'], paths['mismatch']),
		)
	return F3LabelConsistencyOutputResult(
		metadata_json=outputs.output_json,
		report_csv=outputs.output_csv,
		report_markdown=outputs.report_path,
		per_slice_json_paths=tuple(per_slice_json_paths),
		figure_paths=tuple(figure_paths),
	)


def _compare_png_label_file(
	segy: F3SegyInspection,
	file_result: PngLabelFileInspection,
	*,
	max_mismatch_rate: float,
	ignore_border_samples_z: int,
) -> F3LabelConsistencyRecord:
	if file_result.slice_type is None or file_result.slice_index is None:
		msg = (
			'label consistency requires parseable slice_type and slice_index: '
			f'{file_result.relative_path}'
		)
		raise ValueError(msg)
	label_slice = extract_teacher_label_slice(
		segy.label.cube,
		segy.label.geometry,
		slice_type=file_result.slice_type,
		slice_index=file_result.slice_index,
	)
	rgb = read_png_rgb(file_result.absolute_path)
	png_map = rgb_to_class_id_map(
		rgb,
		segy.classes,
		allow_unknown_colors=True,
	)
	normalized_segy = _normalize_segy_label_values(label_slice.values, segy.classes)
	png_shape = _shape_2d(png_map.class_id_map)
	segy_shape = _shape_2d(label_slice.values)
	try:
		alignment = align_png_class_ids_to_segy_slice(
			png_map.class_id_map,
			segy_slice_shape=segy_shape,
		)
	except ValueError:
		return F3LabelConsistencyRecord(
			relative_path=file_result.relative_path,
			absolute_path=file_result.absolute_path,
			split=file_result.split,
			slice_type=file_result.slice_type,
			slice_index=file_result.slice_index,
			png_label_shape=png_shape,
			segy_slice_shape=segy_shape,
			alignment=None,
			segy_line_mapping=label_slice.resolved_line,
			matched_pixel_count=0,
			mismatch_pixel_count=0,
			mismatch_rate=None,
			border_mismatch_pixel_count=0,
			interior_mismatch_pixel_count=0,
			effective_mismatch_rate=None,
			border_only_mismatch=False,
			unknown_png_pixel_count=png_map.unknown_pixel_count,
			unexpected_segy_label_values=normalized_segy.unexpected_values,
			exceeds_threshold=True,
		)

	mismatch_mask = alignment.class_id_map != normalized_segy.values
	mismatch_count = int(np.count_nonzero(mismatch_mask))
	total_count = int(mismatch_mask.size)
	mismatch_rate = _fraction(mismatch_count, total_count)
	interior_mask = _z_interior_mask(
		mismatch_mask.shape,
		ignore_border_samples_z=ignore_border_samples_z,
	)
	interior_mismatch_count = int(np.count_nonzero(mismatch_mask & interior_mask))
	interior_total_count = int(np.count_nonzero(interior_mask))
	border_mismatch_count = mismatch_count - interior_mismatch_count
	effective_mismatch_rate = _fraction(
		interior_mismatch_count,
		interior_total_count,
	)
	border_only_mismatch = mismatch_count > 0 and interior_mismatch_count == 0
	return F3LabelConsistencyRecord(
		relative_path=file_result.relative_path,
		absolute_path=file_result.absolute_path,
		split=file_result.split,
		slice_type=file_result.slice_type,
		slice_index=file_result.slice_index,
		png_label_shape=png_shape,
		segy_slice_shape=segy_shape,
		alignment=alignment,
		segy_line_mapping=label_slice.resolved_line,
		matched_pixel_count=total_count - mismatch_count,
		mismatch_pixel_count=mismatch_count,
		mismatch_rate=mismatch_rate,
		border_mismatch_pixel_count=border_mismatch_count,
		interior_mismatch_pixel_count=interior_mismatch_count,
		effective_mismatch_rate=effective_mismatch_rate,
		border_only_mismatch=border_only_mismatch,
		unknown_png_pixel_count=png_map.unknown_pixel_count,
		unexpected_segy_label_values=normalized_segy.unexpected_values,
		exceeds_threshold=effective_mismatch_rate > max_mismatch_rate,
	)


@dataclass(frozen=True)
class _NormalizedSegyLabels:
	values: NDArray[np.generic]
	unexpected_values: tuple[int | float | str, ...]


def _normalize_segy_label_values(
	values: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
) -> _NormalizedSegyLabels:
	array = np.asarray(values)
	if array.ndim != 2:
		msg = f'SEGY label slice must be 2D; got shape={array.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(array.dtype, np.number):
		msg = f'SEGY label slice must be numeric; got dtype={array.dtype}'
		raise TypeError(msg)
	finite_mask = np.isfinite(array)
	rounded = np.rint(array)
	integer_like = bool(np.all(np.equal(array[finite_mask], rounded[finite_mask])))
	if integer_like:
		normalized = np.full(array.shape, np.iinfo(np.int64).min, dtype=np.int64)
		normalized[finite_mask] = rounded[finite_mask].astype(np.int64)
	else:
		normalized = array.astype(np.float64, copy=False)
	class_ids = {item.class_id for item in classes}
	unexpected = _unexpected_segy_values(
		array,
		class_ids=class_ids,
		integer_like=integer_like,
		finite_mask=finite_mask,
	)
	return _NormalizedSegyLabels(
		values=normalized,
		unexpected_values=unexpected,
	)


def _unexpected_segy_values(
	array: NDArray[np.generic],
	*,
	class_ids: set[int],
	integer_like: bool,
	finite_mask: NDArray[np.bool_],
) -> tuple[int | float | str, ...]:
	finite_values = array[finite_mask]
	if finite_values.size == 0:
		unexpected: list[int | float | str] = []
	elif integer_like:
		unique_values = sorted({int(value) for value in np.rint(finite_values)})
		unexpected = [value for value in unique_values if value not in class_ids]
	else:
		unique_values = sorted({float(value) for value in finite_values})
		unexpected = [value for value in unique_values if value not in class_ids]
	if int(np.count_nonzero(~finite_mask)) > 0:
		unexpected.append('nonfinite')
	return tuple(unexpected)


def _write_record_artifacts(
	report: F3LabelConsistencyReport,
	record: F3LabelConsistencyRecord,
	*,
	output_dir: Path,
	figure_config: F3LabelConsistencyFigureConfig,
) -> dict[str, Path]:
	paths = {
		'png_label': output_dir / f'{record.output_prefix}_png_label.png',
		'segy_label': output_dir / f'{record.output_prefix}_segy_label.png',
		'mismatch': output_dir / f'{record.output_prefix}_mismatch.png',
		'json': output_dir / f'{record.output_prefix}_consistency.json',
	}
	artifact_payload = _record_artifact_payload(
		report,
		record,
		paths=paths,
		figure_config=figure_config,
	)
	_write_json(paths['json'], artifact_payload)
	_save_record_figures(
		report,
		record,
		paths=paths,
		payload=artifact_payload,
		figure_config=figure_config,
	)
	return paths


def _record_artifact_payload(
	report: F3LabelConsistencyReport,
	record: F3LabelConsistencyRecord,
	*,
	paths: Mapping[str, Path],
	figure_config: F3LabelConsistencyFigureConfig,
) -> dict[str, object]:
	return {
		'artifact_type': 'f3_label_consistency',
		'f3_root': str(report.f3_root),
		'class_info_path': str(report.class_info_path),
		'source_segy_label': str(report.source_segy_label),
		'axis_mapping': axis_assumption_metadata(),
		'max_mismatch_rate': report.max_mismatch_rate,
		'figure_config': figure_config.to_dict(),
		'outputs': {key: str(path) for key, path in paths.items()},
		'result': record.to_dict(),
	}


def _save_record_figures(
	report: F3LabelConsistencyReport,
	record: F3LabelConsistencyRecord,
	*,
	paths: Mapping[str, Path],
	payload: Mapping[str, object],
	figure_config: F3LabelConsistencyFigureConfig,
) -> None:
	png_ids, segy_ids, mismatch_mask = _load_record_comparison_arrays(
		report,
		record,
	)
	_save_label_panel(
		png_ids,
		record,
		paths['png_label'],
		title='PNG label',
		classes=report.classes,
		figure_config=figure_config,
	)
	_save_label_panel(
		segy_ids,
		record,
		paths['segy_label'],
		title='SEGY label',
		classes=report.classes,
		figure_config=figure_config,
	)
	_save_mismatch_figure(
		png_ids,
		segy_ids,
		mismatch_mask,
		record,
		paths['mismatch'],
		classes=report.classes,
		figure_config=figure_config,
	)
	_write_json(paths['mismatch'].with_suffix('.json'), payload)


def _load_record_comparison_arrays(
	report: F3LabelConsistencyReport,
	record: F3LabelConsistencyRecord,
) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.bool_]]:
	classes = report.classes
	rgb = read_png_rgb(record.absolute_path)
	png_map = rgb_to_class_id_map(rgb, classes, allow_unknown_colors=True)
	label_slice = extract_teacher_label_slice(
		report.label_cube,
		report.label_geometry,
		slice_type=record.slice_type,
		slice_index=record.slice_index,
	)
	segy_labels = _normalize_segy_label_values(label_slice.values, classes).values
	if record.alignment is None:
		png_ids = png_map.class_id_map
		target_shape = png_ids.shape
		segy_ids = _resize_for_incompatible_display(segy_labels, target_shape)
	else:
		png_ids = record.alignment.class_id_map
		segy_ids = segy_labels
	mismatch_mask = png_ids != segy_ids
	return png_ids, segy_ids, np.asarray(mismatch_mask, dtype=np.bool_)


def _save_label_panel(  # noqa: PLR0913
	class_ids: NDArray[np.generic],
	record: F3LabelConsistencyRecord,
	path: Path,
	*,
	title: str,
	classes: Sequence[F3ClassInfo],
	figure_config: F3LabelConsistencyFigureConfig,
) -> None:
	plt = _matplotlib_pyplot()
	path.parent.mkdir(parents=True, exist_ok=True)
	display_ids = _display_image(class_ids)
	fig, ax = plt.subplots(
		figsize=(5.2, 4.2),
		dpi=figure_config.dpi,
		constrained_layout=True,
	)
	ax.imshow(
		_label_rgb(display_ids, classes),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	ax.set_title(title)
	_configure_slice_axes(ax, record)
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


def _save_mismatch_figure(  # noqa: PLR0913
	png_ids: NDArray[np.generic],
	segy_ids: NDArray[np.generic],
	mismatch_mask: NDArray[np.bool_],
	record: F3LabelConsistencyRecord,
	path: Path,
	*,
	classes: Sequence[F3ClassInfo],
	figure_config: F3LabelConsistencyFigureConfig,
) -> None:
	plt = _matplotlib_pyplot()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(
		1,
		3,
		figsize=(11.2, 4.0),
		dpi=figure_config.dpi,
		sharex=True,
		sharey=True,
		constrained_layout=True,
	)
	png_display = _display_image(png_ids)
	segy_display = _display_image(segy_ids)
	mismatch_display = _display_image(mismatch_mask)
	axes[0].imshow(
		_label_rgb(png_display, classes),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[0].set_title('PNG label')
	axes[1].imshow(
		_label_rgb(segy_display, classes),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[1].set_title('SEGY label')
	axes[2].imshow(
		_mismatch_rgb(mismatch_display, figure_config.mismatch_rgb),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[2].set_title('mismatch')
	for ax in axes:
		_configure_slice_axes(ax, record)
	axes[1].legend(
		handles=_class_legend_handles(classes),
		loc='center left',
		bbox_to_anchor=(1.02, 0.5),
		frameon=False,
		fontsize=7,
		title='class',
		title_fontsize=8,
	)
	fig.suptitle(_record_title(record), fontsize=10)
	fig.savefig(path, facecolor='white', bbox_inches='tight')
	plt.close(fig)


def _display_image(values: NDArray[np.generic]) -> NDArray[np.generic]:
	array = np.asarray(values)
	if array.ndim != 2:
		msg = f'display image must be 2D; got shape={array.shape!r}'
		raise ValueError(msg)
	return np.asarray(array.T)


def _label_rgb(
	class_ids: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
) -> NDArray[np.uint8]:
	array = np.asarray(class_ids)
	ids = np.full(array.shape, -1, dtype=np.int64)
	finite_mask = np.isfinite(array)
	rounded = np.rint(array)
	integer_mask = finite_mask & np.equal(array, rounded)
	ids[integer_mask] = rounded[integer_mask].astype(np.int64)
	return class_id_image_to_rgb(ids, classes, invalid_rgb=_INVALID_LABEL_RGB)


def _mismatch_rgb(
	mask: NDArray[np.bool_],
	mismatch_rgb: tuple[int, int, int],
) -> NDArray[np.uint8]:
	array = np.asarray(mask, dtype=np.bool_)
	rgb = np.full((*array.shape, 3), 255, dtype=np.uint8)
	rgb[array] = mismatch_rgb
	return rgb


def _configure_slice_axes(ax: object, record: F3LabelConsistencyRecord) -> None:
	if record.slice_type == 'inline':
		ax.set_xlabel('crossline index')
	else:
		ax.set_xlabel('inline index')
	ax.set_ylabel('sample/time index down')
	ax.tick_params(labelsize=7)


def _record_title(record: F3LabelConsistencyRecord) -> str:
	rate = 'n/a' if record.mismatch_rate is None else f'{record.mismatch_rate:.6g}'
	effective = (
		'n/a'
		if record.effective_mismatch_rate is None
		else f'{record.effective_mismatch_rate:.6g}'
	)
	return (
		f'{record.split} {record.slice_type} {record.slice_index} '
		f'mismatch={rate} effective={effective}'
	)


def _render_record_row(record: F3LabelConsistencyRecord) -> str:
	rate = '' if record.mismatch_rate is None else f'{record.mismatch_rate:.6g}'
	effective = (
		''
		if record.effective_mismatch_rate is None
		else f'{record.effective_mismatch_rate:.6g}'
	)
	border_only = 'yes' if record.border_only_mismatch else 'no'
	unexpected = json.dumps(list(record.unexpected_segy_label_values))
	threshold = 'FAIL' if record.exceeds_threshold else 'PASS'
	return (
		f'| {record.split} | {record.slice_type} {record.slice_index} | '
		f'{record.orientation} | {list(record.png_label_shape)} | '
		f'{list(record.segy_slice_shape)} | {record.matched_pixel_count} | '
		f'{record.mismatch_pixel_count} | {rate} | {effective} | '
		f'{border_only} | {record.border_mismatch_pixel_count} | '
		f'{record.interior_mismatch_pixel_count} | '
		f'{record.unknown_png_pixel_count} | `{unexpected}` | {threshold} |'
	)


def _write_csv(
	path: str | Path,
	records: Sequence[F3LabelConsistencyRecord],
) -> None:
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=CONSISTENCY_CSV_FIELDNAMES)
		writer.writeheader()
		writer.writerows(record.to_csv_row() for record in records)


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _write_text(path: str | Path, content: str) -> None:
	text_path = Path(path)
	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text(content, encoding='utf-8')


def _validate_label_cube(cube: NDArray[np.generic]) -> NDArray[np.generic]:
	array = np.asarray(cube)
	if array.ndim != 3:
		msg = (
			'label cube must be 3D with shape inline x crossline x sample; '
			f'got {array.shape!r}'
		)
		raise ValueError(msg)
	return array


def _shape_2d(values: NDArray[np.generic]) -> tuple[int, int]:
	array = np.asarray(values)
	if array.ndim != 2:
		msg = f'expected a 2D image; got shape={array.shape!r}'
		raise ValueError(msg)
	return int(array.shape[0]), int(array.shape[1])


def _normalize_2d_shape(
	shape: tuple[int, ...],
	*,
	label: str,
) -> tuple[int, int]:
	if len(shape) != 2:
		msg = f'{label} must contain two axes; got {shape!r}'
		raise ValueError(msg)
	return int(shape[0]), int(shape[1])


def _z_interior_mask(
	shape: tuple[int, ...],
	*,
	ignore_border_samples_z: int,
) -> NDArray[np.bool_]:
	rows, columns = _normalize_2d_shape(shape, label='mismatch_mask.shape')
	mask = np.ones((rows, columns), dtype=np.bool_)
	border = ignore_border_samples_z
	if border <= 0:
		return mask
	if border * 2 >= columns:
		msg = (
			'ignore_border_samples_z is too large for z/sample axis; '
			f'border={border}, z_size={columns}'
		)
		raise ValueError(msg)
	mask[:, :border] = False
	mask[:, -border:] = False
	return mask


def _fraction(numerator: int, denominator: int) -> float:
	if denominator <= 0:
		return 0.0
	return float(numerator / denominator)


def _validate_mismatch_rate_threshold(value: float) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'max_mismatch_rate must be a number in [0, 1]; got {value!r}'
		raise TypeError(msg)
	threshold = float(value)
	if not 0.0 <= threshold <= 1.0:
		msg = f'max_mismatch_rate must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return threshold


def _validate_ignore_border_samples_z(value: int) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		msg = f'ignore_border_samples_z must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_class_info_matches(
	segy_classes: Sequence[F3ClassInfo],
	png_classes: Sequence[F3ClassInfo],
) -> None:
	if tuple(segy_classes) != tuple(png_classes):
		msg = 'SEGY and PNG class_info records must match exactly'
		raise ValueError(msg)


def _report_warnings(
	records: Sequence[F3LabelConsistencyRecord],
) -> tuple[str, ...]:
	warnings: list[str] = []
	if any(record.alignment is None for record in records):
		warnings.append('one or more PNG labels could not be shape-aligned to SEGY')
	transposed = [record for record in records if record.orientation != 'none']
	if transposed:
		warnings.append(
			'one or more PNG labels required non-default orientation; '
			'see per-slice JSON metadata',
		)
	if any(record.border_only_mismatch for record in records):
		warnings.append(
			'one or more PNG/SEGY label mismatches are confined to ignored '
			'z-border samples',
		)
	if any(record.exceeds_threshold for record in records):
		warnings.append('one or more slices exceed max_mismatch_rate')
	return tuple(warnings)


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


def _resize_for_incompatible_display(
	values: NDArray[np.generic],
	shape: tuple[int, ...],
) -> NDArray[np.generic]:
	array = np.asarray(values)
	result = np.full(shape, np.iinfo(np.int64).min, dtype=np.int64)
	rows = min(result.shape[0], array.shape[0])
	columns = min(result.shape[1], array.shape[1])
	result[:rows, :columns] = _label_rgb_ids(array[:rows, :columns])
	return result


def _label_rgb_ids(values: NDArray[np.generic]) -> NDArray[np.int64]:
	array = np.asarray(values)
	result = np.full(array.shape, np.iinfo(np.int64).min, dtype=np.int64)
	finite_mask = np.isfinite(array)
	rounded = np.rint(array)
	integer_mask = finite_mask & np.equal(array, rounded)
	result[integer_mask] = rounded[integer_mask].astype(np.int64)
	return result


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


def _normalize_rgb(
	rgb: tuple[int, int, int],
	*,
	label: str,
) -> tuple[int, int, int]:
	if len(rgb) != 3:
		msg = f'{label} must contain exactly three channels; got {rgb!r}'
		raise ValueError(msg)
	for channel in rgb:
		if not isinstance(channel, int) or isinstance(channel, bool):
			msg = f'{label} channels must be integers; got {rgb!r}'
			raise TypeError(msg)
		if channel < 0 or channel > 255:
			msg = f'{label} channels must be in [0, 255]; got {rgb!r}'
			raise ValueError(msg)
	return rgb


def _matplotlib_pyplot() -> object:
	try:
		return __import__('matplotlib.pyplot', fromlist=['pyplot'])
	except ImportError as exc:
		msg = (
			'F3 label consistency visualization requires matplotlib; '
			'install seis-cluster-ssl[visualization].'
		)
		raise ImportError(msg) from exc


__all__ = [
	'CONSISTENCY_CSV_FIELDNAMES',
	'F3LabelConsistencyAlignment',
	'F3LabelConsistencyFigureConfig',
	'F3LabelConsistencyOutputConfig',
	'F3LabelConsistencyOutputResult',
	'F3LabelConsistencyRecord',
	'F3LabelConsistencyReport',
	'F3LabelSlice',
	'align_png_class_ids_to_segy_slice',
	'check_f3_label_consistency',
	'extract_teacher_label_slice',
	'label_consistency_report_to_dict',
	'render_label_consistency_markdown',
	'write_f3_label_consistency_outputs',
]
