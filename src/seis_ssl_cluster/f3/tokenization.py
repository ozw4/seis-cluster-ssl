"""Tokenization preview helpers for F3 teacher label slices."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
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
from seis_ssl_cluster.f3.visualization import class_id_image_to_rgb

if TYPE_CHECKING:
	from numpy.typing import NDArray

	from seis_ssl_cluster.f3.labels import F3ClassInfo

TOKENIZATION_SUMMARY_FIELDNAMES = (
	'split',
	'relative_path',
	'slice_type',
	'slice_index',
	'total_tokens',
	'retained_tokens',
	'dropped_tokens',
	'retained_fraction',
	'class_counts_retained',
	'ambiguous_token_count',
	'empty_token_count',
)

_INVALID_LABEL_RGB = (226, 226, 226)
_RETAINED_RGB = (0, 114, 178)
_DROPPED_RGB = (213, 94, 0)
_GRID_COLOR = '#4D4D4D'
_SLICE_TYPE_ORDER = {'inline': 0, 'crossline': 1}


@dataclass(frozen=True)
class F3TokenizationConfig:
	"""Patch and threshold controls for label-to-token aggregation."""

	patch_size_xyz: tuple[int, int, int]
	min_labeled_fraction: float = 0.5
	min_majority_fraction: float = 0.7

	def __post_init__(self) -> None:
		"""Validate tokenization settings."""
		_validate_positive_int_triplet(self.patch_size_xyz, label='patch_size_xyz')
		_validate_fraction(
			self.min_labeled_fraction,
			label='min_labeled_fraction',
		)
		_validate_fraction(
			self.min_majority_fraction,
			label='min_majority_fraction',
		)

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable tokenization config."""
		return {
			'patch_size_xyz': list(self.patch_size_xyz),
			'min_labeled_fraction': self.min_labeled_fraction,
			'min_majority_fraction': self.min_majority_fraction,
		}


@dataclass(frozen=True)
class F3TokenizationFigureConfig:
	"""Rendering controls for F3 tokenization preview figures."""

	dpi: int = 300
	background: str = 'white'

	def __post_init__(self) -> None:
		"""Validate figure settings."""
		if not isinstance(self.dpi, int) or isinstance(self.dpi, bool) or self.dpi <= 0:
			msg = f'dpi must be a positive integer; got {self.dpi!r}'
			raise ValueError(msg)
		if self.background != 'white':
			msg = 'background must be "white" for F3 inspection figures'
			raise ValueError(msg)

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable figure config."""
		return {
			'dpi': self.dpi,
			'background': self.background,
			'output_formats': ['png'],
		}


@dataclass(frozen=True)
class F3TokenizationOutputConfig:
	"""Destination paths for F3 tokenization preview artifacts."""

	tokenization_dir: Path
	metadata_json: Path
	summary_csv: Path
	summary_markdown: Path


@dataclass(frozen=True)
class F3TokenizationOutputResult:
	"""Paths written by the F3 tokenization preview writer."""

	png_paths: tuple[Path, ...]
	sidecar_paths: tuple[Path, ...]
	metadata_json: Path
	summary_csv: Path
	summary_markdown: Path


@dataclass(frozen=True)
class F3TokenPlaneSpec:
	"""Axis mapping from a 2D teacher slice into 3D token coordinates."""

	slice_type: str
	slice_index: int
	fixed_axis: str
	fixed_token_index: int
	row_axis: str
	column_axis: str
	row_patch_size: int
	column_patch_size: int
	horizontal_axis_label: str
	vertical_axis_label: str

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable token-plane mapping."""
		return {
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'fixed_axis': self.fixed_axis,
			'fixed_token_index': self.fixed_token_index,
			'row_axis': self.row_axis,
			'column_axis': self.column_axis,
			'row_patch_size': self.row_patch_size,
			'column_patch_size': self.column_patch_size,
			'horizontal_axis_label': self.horizontal_axis_label,
			'vertical_axis_label': self.vertical_axis_label,
		}


@dataclass(frozen=True)
class F3TokenizationAlignment:
	"""PNG class-ID alignment used before token aggregation."""

	class_id_map: NDArray[np.int32]
	source_shape: tuple[int, int]
	tokenization_shape: tuple[int, int]
	transform: str
	transpose: bool

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable alignment payload."""
		return {
			'source_shape': list(self.source_shape),
			'tokenization_shape': list(self.tokenization_shape),
			'output_shape': [int(axis) for axis in self.class_id_map.shape],
			'transform': self.transform,
			'transpose': self.transpose,
		}


@dataclass(frozen=True)
class F3TokenizationSliceResult:
	"""Token aggregation result for one aligned 2D class-ID slice."""

	class_id_map: NDArray[np.int32]
	plane: F3TokenPlaneSpec
	majority_class_ids: NDArray[np.int32]
	retained_mask: NDArray[np.bool_]
	dropped_mask: NDArray[np.bool_]
	empty_mask: NDArray[np.bool_]
	labeled_fraction: NDArray[np.float64]
	majority_fraction: NDArray[np.float64]
	class_counts_retained: dict[int, int]

	@property
	def total_tokens(self) -> int:
		"""Return token count in this 2D token plane."""
		return int(self.retained_mask.size)

	@property
	def retained_tokens(self) -> int:
		"""Return number of tokens satisfying all adoption thresholds."""
		return int(np.count_nonzero(self.retained_mask))

	@property
	def dropped_tokens(self) -> int:
		"""Return number of tokens not retained."""
		return int(np.count_nonzero(self.dropped_mask))

	@property
	def empty_token_count(self) -> int:
		"""Return number of tokens with no labeled pixels."""
		return int(np.count_nonzero(self.empty_mask))

	@property
	def ambiguous_token_count(self) -> int:
		"""Return non-empty dropped tokens."""
		return int(np.count_nonzero(self.dropped_mask & ~self.empty_mask))

	@property
	def retained_fraction(self) -> float:
		"""Return retained token fraction."""
		return _fraction(self.retained_tokens, self.total_tokens)

	def to_summary_dict(self) -> dict[str, object]:
		"""Return JSON-safe summary metrics for this tokenized slice."""
		return {
			'token_grid_shape': [int(axis) for axis in self.retained_mask.shape],
			'total_tokens': self.total_tokens,
			'retained_tokens': self.retained_tokens,
			'dropped_tokens': self.dropped_tokens,
			'retained_fraction': self.retained_fraction,
			'class_counts_retained': {
				str(class_id): count
				for class_id, count in sorted(self.class_counts_retained.items())
			},
			'ambiguous_token_count': self.ambiguous_token_count,
			'empty_token_count': self.empty_token_count,
		}


@dataclass(frozen=True)
class F3TokenizationPreviewRecord:
	"""Per-PNG tokenization preview result and artifact paths."""

	relative_path: str
	absolute_path: str
	split: str
	slice_type: str
	slice_index: int
	alignment: F3TokenizationAlignment
	tokenization: F3TokenizationSliceResult
	output_path: Path
	sidecar_path: Path

	@property
	def output_prefix(self) -> str:
		"""Return the stable basename prefix used for this slice."""
		return f'{self.split}_{self.slice_type}_{self.slice_index:04d}'

	def to_csv_row(self) -> dict[str, object]:
		"""Return the CSV summary row for this preview record."""
		summary = self.tokenization.to_summary_dict()
		return {
			'split': self.split,
			'relative_path': self.relative_path,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'total_tokens': summary['total_tokens'],
			'retained_tokens': summary['retained_tokens'],
			'dropped_tokens': summary['dropped_tokens'],
			'retained_fraction': summary['retained_fraction'],
			'class_counts_retained': json.dumps(
				summary['class_counts_retained'],
				sort_keys=True,
			),
			'ambiguous_token_count': summary['ambiguous_token_count'],
			'empty_token_count': summary['empty_token_count'],
		}

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable per-slice preview record."""
		return {
			'relative_path': self.relative_path,
			'absolute_path': self.absolute_path,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'output_path': str(self.output_path),
			'sidecar_path': str(self.sidecar_path),
			'label_shape_alignment': self.alignment.to_dict(),
			'token_plane': self.tokenization.plane.to_dict(),
			'summary': self.tokenization.to_summary_dict(),
		}


def token_plane_spec(
	*,
	slice_type: str,
	slice_index: int,
	patch_size_xyz: Sequence[int],
) -> F3TokenPlaneSpec:
	"""Return the fixed-axis token-plane mapping for an F3 teacher slice."""
	patch_x, patch_y, patch_z = _validate_positive_int_triplet(
		patch_size_xyz,
		label='patch_size_xyz',
	)
	if not isinstance(slice_index, int) or isinstance(slice_index, bool):
		msg = f'slice_index must be an integer; got {slice_index!r}'
		raise TypeError(msg)
	if slice_index < 0:
		msg = f'slice_index must be non-negative; got {slice_index!r}'
		raise ValueError(msg)
	if slice_type == 'inline':
		return F3TokenPlaneSpec(
			slice_type=slice_type,
			slice_index=slice_index,
			fixed_axis='x',
			fixed_token_index=slice_index // patch_x,
			row_axis='y',
			column_axis='z',
			row_patch_size=patch_y,
			column_patch_size=patch_z,
			horizontal_axis_label='crossline index',
			vertical_axis_label='sample/time index down',
		)
	if slice_type == 'crossline':
		return F3TokenPlaneSpec(
			slice_type=slice_type,
			slice_index=slice_index,
			fixed_axis='y',
			fixed_token_index=slice_index // patch_y,
			row_axis='x',
			column_axis='z',
			row_patch_size=patch_x,
			column_patch_size=patch_z,
			horizontal_axis_label='inline index',
			vertical_axis_label='sample/time index down',
		)
	msg = f'slice_type must be inline or crossline; got {slice_type!r}'
	raise ValueError(msg)


def tokenize_label_slice(
	class_id_map: NDArray[np.generic],
	*,
	slice_type: str,
	slice_index: int,
	config: F3TokenizationConfig,
	classes: Sequence[F3ClassInfo] = (),
) -> F3TokenizationSliceResult:
	"""Aggregate 2D pixel labels into token labels by majority vote."""
	class_ids = _normalize_class_id_map(class_id_map)
	plane = token_plane_spec(
		slice_type=slice_type,
		slice_index=slice_index,
		patch_size_xyz=config.patch_size_xyz,
	)
	row_tokens = _ceil_div(class_ids.shape[0], plane.row_patch_size)
	column_tokens = _ceil_div(class_ids.shape[1], plane.column_patch_size)
	majority_class_ids = np.full((row_tokens, column_tokens), -1, dtype=np.int32)
	retained_mask = np.zeros((row_tokens, column_tokens), dtype=np.bool_)
	empty_mask = np.zeros((row_tokens, column_tokens), dtype=np.bool_)
	labeled_fraction = np.zeros((row_tokens, column_tokens), dtype=np.float64)
	majority_fraction = np.zeros((row_tokens, column_tokens), dtype=np.float64)

	for row_token in range(row_tokens):
		row_start = row_token * plane.row_patch_size
		row_stop = min(row_start + plane.row_patch_size, class_ids.shape[0])
		for column_token in range(column_tokens):
			column_start = column_token * plane.column_patch_size
			column_stop = min(
				column_start + plane.column_patch_size,
				class_ids.shape[1],
			)
			block = class_ids[row_start:row_stop, column_start:column_stop]
			labeled = block[block >= 0]
			labeled_fraction[row_token, column_token] = _fraction(
				int(labeled.size),
				int(block.size),
			)
			if labeled.size == 0:
				empty_mask[row_token, column_token] = True
				continue
			values, counts = np.unique(labeled, return_counts=True)
			winner_index = int(np.argmax(counts))
			winner = int(values[winner_index])
			majority_class_ids[row_token, column_token] = winner
			majority_fraction[row_token, column_token] = _fraction(
				int(counts[winner_index]),
				int(labeled.size),
			)
			retained_mask[row_token, column_token] = (
				labeled_fraction[row_token, column_token]
				>= config.min_labeled_fraction
				and majority_fraction[row_token, column_token]
				>= config.min_majority_fraction
			)

	dropped_mask = ~retained_mask
	class_counts = _retained_class_counts(
		majority_class_ids,
		retained_mask,
		classes=classes,
	)
	return F3TokenizationSliceResult(
		class_id_map=class_ids,
		plane=plane,
		majority_class_ids=majority_class_ids,
		retained_mask=retained_mask,
		dropped_mask=dropped_mask,
		empty_mask=empty_mask,
		labeled_fraction=labeled_fraction,
		majority_fraction=majority_fraction,
		class_counts_retained=class_counts,
	)


def load_f3_label_consistency_alignments(
	path: str | Path,
) -> dict[str, F3TokenizationAlignment]:
	"""Load PNG-to-SEGY shape alignments from label consistency JSON."""
	json_path = Path(path)
	with json_path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if not isinstance(payload, Mapping):
		msg = f'label consistency JSON must contain an object: {json_path}'
		raise TypeError(msg)
	files = payload.get('files')
	if not isinstance(files, list):
		msg = f'label consistency JSON must contain a files list: {json_path}'
		raise TypeError(msg)

	alignments: dict[str, F3TokenizationAlignment] = {}
	for raw_record in files:
		if not isinstance(raw_record, Mapping):
			msg = f'label consistency file record must be an object: {raw_record!r}'
			raise TypeError(msg)
		relative_path = _required_str(raw_record, 'relative_path')
		shape_alignment = raw_record.get('label_shape_alignment')
		if not isinstance(shape_alignment, Mapping):
			msg = (
				'label consistency record is not shape-aligned: '
				f'{relative_path}'
			)
			raise TypeError(msg)
		alignments[relative_path] = _alignment_from_consistency_record(
			shape_alignment,
			source=relative_path,
		)
	return alignments


def write_f3_tokenization_preview_outputs(
	png_labels: F3PngLabelInspection,
	outputs: F3TokenizationOutputConfig,
	config: F3TokenizationConfig,
	alignments: Mapping[str, F3TokenizationAlignment],
	figure_config: F3TokenizationFigureConfig | None = None,
) -> F3TokenizationOutputResult:
	"""Write tokenization preview figures, sidecars, and summary tables."""
	figure = figure_config or F3TokenizationFigureConfig()
	outputs.tokenization_dir.mkdir(parents=True, exist_ok=True)
	outputs.summary_csv.parent.mkdir(parents=True, exist_ok=True)
	outputs.summary_markdown.parent.mkdir(parents=True, exist_ok=True)
	outputs.metadata_json.parent.mkdir(parents=True, exist_ok=True)

	records: list[F3TokenizationPreviewRecord] = []
	for file_result in _sorted_png_label_files(png_labels.files):
		record = _tokenize_png_label_file(
			file_result,
			classes=png_labels.classes,
			config=config,
			alignments=alignments,
			output_dir=outputs.tokenization_dir,
		)
		_save_tokenization_preview_png(
			record,
			classes=png_labels.classes,
			figure_config=figure,
		)
		_write_json(record.sidecar_path, _sidecar_payload(record, config, figure))
		records.append(record)

	_write_summary_csv(outputs.summary_csv, records)
	_write_text(outputs.summary_markdown, render_tokenization_summary_markdown(records))
	_write_json(
		outputs.metadata_json,
		_tokenization_preview_payload(png_labels, records, config, figure),
	)
	return F3TokenizationOutputResult(
		png_paths=tuple(record.output_path for record in records),
		sidecar_paths=tuple(record.sidecar_path for record in records),
		metadata_json=outputs.metadata_json,
		summary_csv=outputs.summary_csv,
		summary_markdown=outputs.summary_markdown,
	)


def render_tokenization_summary_markdown(
	records: Sequence[F3TokenizationPreviewRecord],
) -> str:
	"""Render a Markdown summary for F3 tokenization previews."""
	total_tokens = sum(record.tokenization.total_tokens for record in records)
	retained_tokens = sum(record.tokenization.retained_tokens for record in records)
	dropped_tokens = sum(record.tokenization.dropped_tokens for record in records)
	ambiguous_tokens = sum(
		record.tokenization.ambiguous_token_count for record in records
	)
	empty_tokens = sum(record.tokenization.empty_token_count for record in records)
	lines = [
		'# F3 tokenization preview',
		'',
		f'- PNG labels: {len(records)}',
		f'- total tokens: {total_tokens}',
		f'- retained tokens: {retained_tokens}',
		f'- dropped tokens: {dropped_tokens}',
		f'- retained fraction: {_fraction(retained_tokens, total_tokens):.6g}',
		f'- ambiguous tokens: {ambiguous_tokens}',
		f'- empty tokens: {empty_tokens}',
		'',
		'## Per-slice results',
		'',
		(
			'| split | slice | total_tokens | retained_tokens | '
			'dropped_tokens | retained_fraction | ambiguous | empty |'
		),
		'|---|---|---:|---:|---:|---:|---:|---:|',
	]
	lines.extend(
		(
			f'| {record.split} | {record.slice_type} {record.slice_index} | '
			f'{record.tokenization.total_tokens} | '
			f'{record.tokenization.retained_tokens} | '
			f'{record.tokenization.dropped_tokens} | '
			f'{record.tokenization.retained_fraction:.6g} | '
			f'{record.tokenization.ambiguous_token_count} | '
			f'{record.tokenization.empty_token_count} |'
		)
		for record in records
	)
	lines.extend(['', '## Retained class counts', ''])
	class_counts = _overall_class_counts(records)
	if class_counts:
		lines.extend(
			f'- class {class_id}: {count}'
			for class_id, count in sorted(class_counts.items())
		)
	else:
		lines.append('- none')
	return '\n'.join(lines) + '\n'


def _tokenize_png_label_file(
	file_result: PngLabelFileInspection,
	*,
	classes: Sequence[F3ClassInfo],
	config: F3TokenizationConfig,
	alignments: Mapping[str, F3TokenizationAlignment],
	output_dir: Path,
) -> F3TokenizationPreviewRecord:
	if file_result.slice_type is None or file_result.slice_index is None:
		msg = (
			'tokenization preview requires parseable slice_type and slice_index: '
			f'{file_result.relative_path}'
		)
		raise ValueError(msg)
	try:
		alignment_template = alignments[file_result.relative_path]
	except KeyError as exc:
		msg = (
			'missing label consistency alignment for PNG label: '
			f'{file_result.relative_path}'
		)
		raise KeyError(msg) from exc
	rgb = read_png_rgb(file_result.absolute_path)
	png_map = rgb_to_class_id_map(rgb, classes, allow_unknown_colors=True)
	alignment = apply_tokenization_alignment(
		png_map.class_id_map,
		alignment_template,
	)
	tokenization = tokenize_label_slice(
		alignment.class_id_map,
		slice_type=file_result.slice_type,
		slice_index=file_result.slice_index,
		config=config,
		classes=classes,
	)
	output_path = output_dir / (
		f'{file_result.split}_{file_result.slice_type}_'
		f'{file_result.slice_index:04d}_tokenization.png'
	)
	return F3TokenizationPreviewRecord(
		relative_path=file_result.relative_path,
		absolute_path=file_result.absolute_path,
		split=file_result.split,
		slice_type=file_result.slice_type,
		slice_index=file_result.slice_index,
		alignment=alignment,
		tokenization=tokenization,
		output_path=output_path,
		sidecar_path=output_path.with_suffix('.json'),
	)


def apply_tokenization_alignment(
	class_id_map: NDArray[np.generic],
	alignment: F3TokenizationAlignment,
) -> F3TokenizationAlignment:
	"""Apply a stored PNG alignment transform to a fresh class-ID map."""
	array = _normalize_class_id_map(class_id_map)
	source_shape = _shape_2d(array)
	if source_shape != alignment.source_shape:
		msg = (
			'PNG label shape changed since label consistency check; '
			f'expected {alignment.source_shape!r}, got {source_shape!r}'
		)
		raise ValueError(msg)
	if alignment.transform == 'none':
		aligned = array
	elif alignment.transform == 'transpose_png_to_segy':
		aligned = np.asarray(array.T, dtype=np.int32)
	else:
		msg = f'unsupported PNG alignment transform: {alignment.transform!r}'
		raise ValueError(msg)
	if _shape_2d(aligned) != alignment.tokenization_shape:
		msg = (
			'aligned PNG label shape does not match tokenization shape; '
			f'expected {alignment.tokenization_shape!r}, got {_shape_2d(aligned)!r}'
		)
		raise ValueError(msg)
	return F3TokenizationAlignment(
		class_id_map=aligned,
		source_shape=alignment.source_shape,
		tokenization_shape=alignment.tokenization_shape,
		transform=alignment.transform,
		transpose=alignment.transpose,
	)


def _alignment_from_consistency_record(
	alignment: Mapping[str, object],
	*,
	source: str,
) -> F3TokenizationAlignment:
	transform = _required_str(alignment, 'transform')
	transpose = _required_bool(alignment, 'transpose')
	source_shape = _shape_from_json(alignment.get('source_shape'), label=source)
	output_shape = _shape_from_json(alignment.get('output_shape'), label=source)
	empty = np.empty(output_shape, dtype=np.int32)
	return F3TokenizationAlignment(
		class_id_map=empty,
		source_shape=source_shape,
		tokenization_shape=output_shape,
		transform=transform,
		transpose=transpose,
	)


def _save_tokenization_preview_png(
	record: F3TokenizationPreviewRecord,
	*,
	classes: Sequence[F3ClassInfo],
	figure_config: F3TokenizationFigureConfig,
) -> None:
	plt = _matplotlib_pyplot()
	record.output_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(
		1,
		4,
		figsize=(13.6, 4.0),
		dpi=figure_config.dpi,
		sharex=True,
		sharey=True,
	)
	original = _display_image(record.tokenization.class_id_map)
	majority = _display_image(_expand_token_values(record.tokenization))
	retained = _display_image(
		_expand_token_mask(
			record.tokenization.retained_mask,
			record.tokenization,
		),
	)
	dropped = _display_image(
		_expand_token_mask(
			record.tokenization.dropped_mask,
			record.tokenization,
		),
	)

	axes[0].imshow(
		class_id_image_to_rgb(original, classes, invalid_rgb=_INVALID_LABEL_RGB),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[0].set_title('original label')
	axes[1].imshow(
		class_id_image_to_rgb(majority, classes, invalid_rgb=_INVALID_LABEL_RGB),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[1].set_title('token majority')
	axes[2].imshow(
		_binary_mask_rgb(retained, true_rgb=_RETAINED_RGB),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	axes[2].set_title(
		f'retained\n{record.tokenization.retained_fraction:.1%}',
		fontsize=9,
	)
	axes[3].imshow(
		_binary_mask_rgb(dropped, true_rgb=_DROPPED_RGB),
		origin='upper',
		aspect='auto',
		interpolation='nearest',
	)
	dropped_fraction = _fraction(
		record.tokenization.dropped_tokens,
		record.tokenization.total_tokens,
	)
	axes[3].set_title(
		f'dropped\n{dropped_fraction:.1%}',
		fontsize=9,
	)
	for ax in axes:
		_configure_token_axes(ax, record)
		_draw_token_grid(ax, record.tokenization)
	fig.legend(
		handles=_class_legend_handles(classes),
		loc='center right',
		bbox_to_anchor=(0.995, 0.5),
		frameon=False,
		fontsize=6,
		title='class',
		title_fontsize=7,
	)
	fig.suptitle(
		f'{record.split} {record.slice_type} {record.slice_index}',
		fontsize=10,
	)
	fig.tight_layout(rect=(0.0, 0.0, 0.90, 0.94))
	fig.savefig(
		record.output_path,
		facecolor=figure_config.background,
		bbox_inches='tight',
	)
	plt.close(fig)


def _expand_token_values(
	tokenization: F3TokenizationSliceResult,
) -> NDArray[np.int32]:
	expanded = np.full(tokenization.class_id_map.shape, -1, dtype=np.int32)
	_fill_token_blocks(expanded, tokenization.majority_class_ids, tokenization)
	return expanded


def _expand_token_mask(
	mask: NDArray[np.bool_],
	tokenization: F3TokenizationSliceResult,
) -> NDArray[np.bool_]:
	expanded = np.zeros(tokenization.class_id_map.shape, dtype=np.bool_)
	_fill_token_blocks(expanded, np.asarray(mask, dtype=np.bool_), tokenization)
	return expanded


def _fill_token_blocks(
	destination: NDArray[np.generic],
	token_values: NDArray[np.generic],
	tokenization: F3TokenizationSliceResult,
) -> None:
	plane = tokenization.plane
	for row_token in range(token_values.shape[0]):
		row_start = row_token * plane.row_patch_size
		row_stop = min(row_start + plane.row_patch_size, destination.shape[0])
		for column_token in range(token_values.shape[1]):
			column_start = column_token * plane.column_patch_size
			column_stop = min(
				column_start + plane.column_patch_size,
				destination.shape[1],
			)
			destination[row_start:row_stop, column_start:column_stop] = token_values[
				row_token,
				column_token,
			]


def _draw_token_grid(ax: object, tokenization: F3TokenizationSliceResult) -> None:
	plane = tokenization.plane
	line_count = tokenization.class_id_map.shape[0]
	sample_count = tokenization.class_id_map.shape[1]
	for boundary in range(plane.row_patch_size, line_count, plane.row_patch_size):
		ax.axvline(boundary - 0.5, color=_GRID_COLOR, linewidth=0.35, alpha=0.35)
	for boundary in range(
		plane.column_patch_size,
		sample_count,
		plane.column_patch_size,
	):
		ax.axhline(boundary - 0.5, color=_GRID_COLOR, linewidth=0.35, alpha=0.35)


def _configure_token_axes(ax: object, record: F3TokenizationPreviewRecord) -> None:
	ax.set_xlabel(record.tokenization.plane.horizontal_axis_label)
	ax.set_ylabel(record.tokenization.plane.vertical_axis_label)
	ax.tick_params(labelsize=6)


def _binary_mask_rgb(
	mask: NDArray[np.bool_],
	*,
	true_rgb: tuple[int, int, int],
) -> NDArray[np.uint8]:
	array = np.asarray(mask, dtype=np.bool_)
	rgb = np.full((*array.shape, 3), 255, dtype=np.uint8)
	rgb[array] = true_rgb
	return rgb


def _display_image(values: NDArray[np.generic]) -> NDArray[np.generic]:
	array = np.asarray(values)
	if array.ndim != 2:
		msg = f'display image must be 2D; got shape={array.shape!r}'
		raise ValueError(msg)
	return np.asarray(array.T)


def _sidecar_payload(
	record: F3TokenizationPreviewRecord,
	config: F3TokenizationConfig,
	figure: F3TokenizationFigureConfig,
) -> dict[str, object]:
	return {
		'figure_type': 'teacher_slice_tokenization_preview',
		'tokenization_config': config.to_dict(),
		'figure_config': figure.to_dict(),
		'result': record.to_dict(),
	}


def _tokenization_preview_payload(
	png_labels: F3PngLabelInspection,
	records: Sequence[F3TokenizationPreviewRecord],
	config: F3TokenizationConfig,
	figure: F3TokenizationFigureConfig,
) -> dict[str, object]:
	return {
		'f3_root': str(png_labels.f3_root),
		'class_info_path': str(png_labels.class_info_path),
		'png_label_file_count': len(records),
		'tokenization_config': config.to_dict(),
		'figure_config': figure.to_dict(),
		'overall_summary': {
			'total_tokens': sum(item.tokenization.total_tokens for item in records),
			'retained_tokens': sum(
				item.tokenization.retained_tokens for item in records
			),
			'dropped_tokens': sum(item.tokenization.dropped_tokens for item in records),
			'ambiguous_token_count': sum(
				item.tokenization.ambiguous_token_count for item in records
			),
			'empty_token_count': sum(
				item.tokenization.empty_token_count for item in records
			),
			'class_counts_retained': {
				str(class_id): count
				for class_id, count in sorted(_overall_class_counts(records).items())
			},
		},
		'outputs': [record.to_dict() for record in records],
	}


def _write_summary_csv(
	path: str | Path,
	records: Sequence[F3TokenizationPreviewRecord],
) -> None:
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=TOKENIZATION_SUMMARY_FIELDNAMES)
		writer.writeheader()
		for record in records:
			writer.writerow(record.to_csv_row())


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _write_text(path: str | Path, text: str) -> None:
	text_path = Path(path)
	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text(text, encoding='utf-8')


def _retained_class_counts(
	majority_class_ids: NDArray[np.int32],
	retained_mask: NDArray[np.bool_],
	*,
	classes: Sequence[F3ClassInfo],
) -> dict[int, int]:
	counts = Counter(int(value) for value in majority_class_ids[retained_mask])
	if classes:
		return {
			class_info.class_id: int(counts.get(class_info.class_id, 0))
			for class_info in classes
		}
	return {class_id: int(count) for class_id, count in sorted(counts.items())}


def _overall_class_counts(
	records: Sequence[F3TokenizationPreviewRecord],
) -> dict[int, int]:
	counts: Counter[int] = Counter()
	for record in records:
		counts.update(record.tokenization.class_counts_retained)
	return {class_id: int(count) for class_id, count in sorted(counts.items())}


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


def _normalize_class_id_map(values: NDArray[np.generic]) -> NDArray[np.int32]:
	array = np.asarray(values)
	if array.ndim != 2:
		msg = f'class_id_map must be 2D; got shape={array.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(array.dtype, np.number):
		msg = f'class_id_map must be numeric; got dtype={array.dtype}'
		raise TypeError(msg)
	finite_mask = np.isfinite(array)
	rounded = np.rint(array)
	if not np.all(np.equal(array[finite_mask], rounded[finite_mask])):
		msg = 'class_id_map must contain integer-like labels'
		raise ValueError(msg)
	class_ids = np.full(array.shape, -1, dtype=np.int32)
	class_ids[finite_mask] = rounded[finite_mask].astype(np.int32)
	return class_ids


def _shape_2d(values: NDArray[np.generic]) -> tuple[int, int]:
	array = np.asarray(values)
	if array.ndim != 2:
		msg = f'array must be 2D; got shape={array.shape!r}'
		raise ValueError(msg)
	return int(array.shape[0]), int(array.shape[1])


def _shape_from_json(value: object, *, label: str) -> tuple[int, int]:
	if not isinstance(value, list | tuple) or len(value) != 2:
		msg = f'{label} shape must be a two-item sequence; got {value!r}'
		raise TypeError(msg)
	shape: list[int] = []
	for item in value:
		if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
			msg = f'{label} shape values must be positive integers; got {value!r}'
			raise ValueError(msg)
		shape.append(item)
	return shape[0], shape[1]


def _validate_positive_int_triplet(
	value: Sequence[int],
	*,
	label: str,
) -> tuple[int, int, int]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a three-item sequence; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if len(values) != 3:
		msg = f'{label} must contain three values; got {value!r}'
		raise ValueError(msg)
	for item in values:
		if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
			msg = f'{label} values must be positive integers; got {value!r}'
			raise ValueError(msg)
	return int(values[0]), int(values[1]), int(values[2])


def _validate_fraction(value: float, *, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in [0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{label} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _required_str(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _required_bool(parent: Mapping[str, object], key: str) -> bool:
	value = parent.get(key)
	if not isinstance(value, bool):
		msg = f'{key} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _ceil_div(numerator: int, denominator: int) -> int:
	return int((numerator + denominator - 1) // denominator)


def _fraction(numerator: int, denominator: int) -> float:
	if denominator == 0:
		return 0.0
	return float(numerator) / float(denominator)


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


def _matplotlib_pyplot() -> object:
	try:
		return __import__('matplotlib.pyplot', fromlist=['pyplot'])
	except ImportError as exc:
		msg = (
			'F3 tokenization preview requires matplotlib; '
			'install seis-cluster-ssl[visualization].'
		)
		raise ImportError(msg) from exc


__all__ = [
	'TOKENIZATION_SUMMARY_FIELDNAMES',
	'F3TokenPlaneSpec',
	'F3TokenizationAlignment',
	'F3TokenizationConfig',
	'F3TokenizationFigureConfig',
	'F3TokenizationOutputConfig',
	'F3TokenizationOutputResult',
	'F3TokenizationPreviewRecord',
	'F3TokenizationSliceResult',
	'apply_tokenization_alignment',
	'load_f3_label_consistency_alignments',
	'render_tokenization_summary_markdown',
	'token_plane_spec',
	'tokenize_label_slice',
	'write_f3_tokenization_preview_outputs',
]
