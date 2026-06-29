"""RGB class inspection for F3 facies benchmark PNG labels."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.f3.inspection import find_class_info_file
from seis_ssl_cluster.f3.labels import (
	VALID_LABEL_SPLITS,
	F3ClassInfo,
	extract_label_split,
	parse_label_png_name,
	read_class_info,
	rgb_to_hex,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from numpy.typing import NDArray

RGB = tuple[int, int, int]

PNG_LABEL_INVENTORY_FIELDNAMES = (
	'relative_path',
	'absolute_path',
	'split',
	'slice_type',
	'slice_index',
	'width',
	'height',
	'pixel_count',
	'unknown_pixel_count',
	'unknown_color_count',
	'unknown_colors',
)
PNG_LABEL_CLASS_COUNT_FIELDNAMES = (
	'scope',
	'relative_path',
	'split',
	'slice_type',
	'slice_index',
	'class_id',
	'class_name',
	'rgb',
	'hex_color',
	'pixel_count',
	'total_pixels',
	'unknown_pixel_count',
	'fraction',
)
_SLICE_TYPE_ORDER = {'inline': 0, 'crossline': 1, None: 2}


@dataclass(frozen=True)
class PngLabelUnknownColor:
	"""One RGB color present in PNG labels but absent from class-info."""

	rgb: RGB
	pixel_count: int

	@property
	def hex_color(self) -> str:
		"""Return the unknown RGB color as `#RRGGBB`."""
		return rgb_to_hex(self.rgb)

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable unknown-color record."""
		return {
			'rgb': list(self.rgb),
			'hex_color': self.hex_color,
			'pixel_count': self.pixel_count,
		}


@dataclass(frozen=True)
class PngLabelClassCount:
	"""Pixel count for one class in one PNG-label scope."""

	class_id: int
	pixel_count: int

	def to_dict(
		self,
		classes_by_id: Mapping[int, F3ClassInfo],
		*,
		total_pixels: int,
	) -> dict[str, object]:
		"""Return a JSON-serializable class-count record."""
		class_info = classes_by_id[self.class_id]
		return {
			'class_id': class_info.class_id,
			'class_name': class_info.class_name,
			'rgb': list(class_info.rgb),
			'hex_color': class_info.hex_color,
			'pixel_count': self.pixel_count,
			'fraction': _fraction(self.pixel_count, total_pixels),
		}


@dataclass(frozen=True)
class PngLabelMap:
	"""Class-ID map converted from one RGB PNG label image."""

	class_id_map: NDArray[np.int32]
	unknown_colors: tuple[PngLabelUnknownColor, ...]

	@property
	def unknown_pixel_count(self) -> int:
		"""Return the total number of unknown-color pixels."""
		return int(sum(item.pixel_count for item in self.unknown_colors))


@dataclass(frozen=True)
class PngLabelFileInspection:
	"""Per-file PNG label inspection result."""

	relative_path: str
	absolute_path: str
	split: str
	slice_type: str | None
	slice_index: int | None
	width: int
	height: int
	pixel_count: int
	unknown_pixel_count: int
	class_counts: tuple[PngLabelClassCount, ...]
	unknown_colors: tuple[PngLabelUnknownColor, ...]

	def class_count_by_id(self) -> dict[int, int]:
		"""Return class pixel counts keyed by class ID."""
		return {item.class_id: item.pixel_count for item in self.class_counts}

	def to_inventory_row(self) -> dict[str, object]:
		"""Return the CSV inventory row for this PNG label."""
		return {
			'relative_path': self.relative_path,
			'absolute_path': self.absolute_path,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'width': self.width,
			'height': self.height,
			'pixel_count': self.pixel_count,
			'unknown_pixel_count': self.unknown_pixel_count,
			'unknown_color_count': len(self.unknown_colors),
			'unknown_colors': _format_unknown_colors(self.unknown_colors),
		}

	def to_inventory_dict(self) -> dict[str, object]:
		"""Return the JSON inventory record for this PNG label."""
		return {
			'relative_path': self.relative_path,
			'absolute_path': self.absolute_path,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'width': self.width,
			'height': self.height,
			'pixel_count': self.pixel_count,
			'unknown_pixel_count': self.unknown_pixel_count,
			'unknown_color_count': len(self.unknown_colors),
			'unknown_colors': [item.to_dict() for item in self.unknown_colors],
		}

	def to_dict(self, classes_by_id: Mapping[int, F3ClassInfo]) -> dict[str, object]:
		"""Return a JSON-serializable per-file inspection record."""
		return {
			'relative_path': self.relative_path,
			'absolute_path': self.absolute_path,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
			'width': self.width,
			'height': self.height,
			'pixel_count': self.pixel_count,
			'unknown_pixel_count': self.unknown_pixel_count,
			'class_counts': [
				item.to_dict(classes_by_id, total_pixels=self.pixel_count)
				for item in self.class_counts
			],
			'unknown_colors': [item.to_dict() for item in self.unknown_colors],
		}


@dataclass(frozen=True)
class F3PngLabelInspection:
	"""Complete F3 PNG label inspection and class distribution result."""

	f3_root: Path
	class_info_path: Path
	classes: tuple[F3ClassInfo, ...]
	files: tuple[PngLabelFileInspection, ...]
	allow_unknown_colors: bool
	warnings: tuple[str, ...]

	@property
	def classes_by_id(self) -> dict[int, F3ClassInfo]:
		"""Return class-info records keyed by class ID."""
		return {item.class_id: item for item in self.classes}

	def files_for_split(self, split: str) -> tuple[PngLabelFileInspection, ...]:
		"""Return inspected PNG files for one split."""
		return tuple(item for item in self.files if item.split == split)

	def total_pixel_count(self) -> int:
		"""Return the total pixel count across all inspected PNG labels."""
		return int(sum(item.pixel_count for item in self.files))

	def total_unknown_pixel_count(self) -> int:
		"""Return the unknown-color pixel count across all PNG labels."""
		return int(sum(item.unknown_pixel_count for item in self.files))

	def class_counts_for_files(
		self,
		files: Sequence[PngLabelFileInspection],
	) -> tuple[PngLabelClassCount, ...]:
		"""Aggregate class counts for the supplied PNG files."""
		accumulator: Counter[int] = Counter()
		for file_result in files:
			accumulator.update(file_result.class_count_by_id())
		return tuple(
			PngLabelClassCount(
				class_id=item.class_id,
				pixel_count=int(accumulator.get(item.class_id, 0)),
			)
			for item in self.classes
		)

	def overall_class_counts(self) -> tuple[PngLabelClassCount, ...]:
		"""Return class counts across all inspected PNG labels."""
		return self.class_counts_for_files(self.files)

	def unknown_colors(self) -> tuple[PngLabelUnknownColor, ...]:
		"""Return unknown colors aggregated across all inspected PNG labels."""
		counter: Counter[RGB] = Counter()
		for file_result in self.files:
			for unknown in file_result.unknown_colors:
				counter[unknown.rgb] += unknown.pixel_count
		return tuple(
			PngLabelUnknownColor(rgb=rgb, pixel_count=count)
			for rgb, count in sorted(
				counter.items(),
				key=lambda item: (-item[1], item[0]),
			)
		)


@dataclass(frozen=True)
class F3PngLabelOutputConfig:
	"""Destination paths for F3 PNG label inspection artifacts."""

	inventory_csv: Path
	inventory_json: Path
	palette_json: Path
	class_counts_csv: Path
	summary_json: Path
	summary_markdown: Path
	class_distribution_train_png: Path
	class_distribution_validation_png: Path
	class_distribution_per_slice_png: Path
	dpi: int = 300


def rgb_to_class_id_map(
	image: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
	*,
	allow_unknown_colors: bool = False,
) -> PngLabelMap:
	"""Convert an RGB image to an integer class-ID map by exact RGB matching."""
	rgb_image = normalize_png_rgb(image)
	_lookup = _rgb_class_lookup(classes)
	flat_codes = _pack_rgb_codes(rgb_image.reshape(-1, 3))
	flat_class_ids = np.full(flat_codes.shape, -1, dtype=np.int32)
	for rgb_code, class_id in _lookup.items():
		flat_class_ids[flat_codes == rgb_code] = class_id
	unknown_colors = _unknown_colors_from_codes(
		flat_codes[flat_class_ids < 0],
	)
	if unknown_colors and not allow_unknown_colors:
		msg = (
			'PNG label contains RGB colors absent from class_info: '
			f'{_format_unknown_colors(unknown_colors)}'
		)
		raise ValueError(msg)
	return PngLabelMap(
		class_id_map=flat_class_ids.reshape(rgb_image.shape[:2]),
		unknown_colors=unknown_colors,
	)


def normalize_png_rgb(
	image: NDArray[np.generic],
	*,
	source: str | Path = 'PNG label',
) -> NDArray[np.uint8]:
	"""Normalize a PNG image array to an `H x W x RGB uint8` array."""
	array = np.asarray(image)
	if array.ndim != 3 or array.shape[2] < 3:
		msg = f'{source} must be an RGB or RGBA image; got shape={array.shape!r}'
		raise ValueError(msg)
	rgb = array[:, :, :3]
	if np.issubdtype(rgb.dtype, np.floating):
		if not np.isfinite(rgb).all() or rgb.min() < 0.0 or rgb.max() > 1.0:
			msg = f'{source} floating RGB values must be finite and within [0, 1]'
			raise ValueError(msg)
		return np.rint(rgb * 255.0).astype(np.uint8)
	if np.issubdtype(rgb.dtype, np.integer):
		if rgb.min() < 0 or rgb.max() > 255:
			msg = f'{source} integer RGB values must be within [0, 255]'
			raise ValueError(msg)
		return rgb.astype(np.uint8, copy=False)
	msg = f'{source} RGB array must use integer or floating dtype; got {rgb.dtype}'
	raise TypeError(msg)


def count_class_pixels(
	class_id_map: NDArray[np.integer],
	classes: Sequence[F3ClassInfo],
) -> tuple[PngLabelClassCount, ...]:
	"""Count pixels for every class-info class, including class ID 0."""
	return tuple(
		PngLabelClassCount(
			class_id=item.class_id,
			pixel_count=int(np.count_nonzero(class_id_map == item.class_id)),
		)
		for item in classes
	)


def read_png_rgb(path: str | Path) -> NDArray[np.uint8]:
	"""Read a PNG label image and return its RGB channels as `uint8`."""
	image_path = Path(path)
	image_module = _matplotlib_image()
	return normalize_png_rgb(image_module.imread(image_path), source=image_path)


def inspect_f3_png_labels(
	f3_root: str | Path,
	*,
	candidate_extensions: Sequence[str] = ('.png',),
	allow_unknown_colors: bool = False,
) -> F3PngLabelInspection:
	"""Inspect all F3 train/validation PNG labels and class distributions."""
	root = Path(f3_root)
	if not root.is_dir():
		msg = f'F3 root directory does not exist: {root}'
		raise FileNotFoundError(msg)
	class_info_path, class_info_warnings = find_class_info_file(root)
	classes = read_class_info(class_info_path)
	label_paths = _find_label_png_paths(
		root,
		candidate_extensions=candidate_extensions,
	)
	if not label_paths:
		msg = f'missing F3 train/validation PNG labels under {root}'
		raise FileNotFoundError(msg)
	files = tuple(
		_inspect_png_file(
			root,
			path,
			classes=classes,
			allow_unknown_colors=allow_unknown_colors,
		)
		for path in label_paths
	)
	warnings = (
		*class_info_warnings,
		*_metadata_warnings(files),
		*_unknown_color_warnings(files),
	)
	return F3PngLabelInspection(
		f3_root=root.resolve(strict=False),
		class_info_path=class_info_path.resolve(strict=False),
		classes=classes,
		files=files,
		allow_unknown_colors=allow_unknown_colors,
		warnings=warnings,
	)


def png_label_inspection_to_dict(
	inspection: F3PngLabelInspection,
) -> dict[str, object]:
	"""Return the machine-readable PNG label inspection summary."""
	classes_by_id = inspection.classes_by_id
	total_pixels = inspection.total_pixel_count()
	overall_counts = inspection.overall_class_counts()
	return {
		'f3_root': str(inspection.f3_root),
		'class_info_path': str(inspection.class_info_path),
		'allow_unknown_colors': inspection.allow_unknown_colors,
		'file_count': len(inspection.files),
		'total_pixels': total_pixels,
		'total_unknown_pixels': inspection.total_unknown_pixel_count(),
		'class_info': {
			'class_count': len(inspection.classes),
			'classes': [item.to_dict() for item in inspection.classes],
		},
		'overall_class_counts': [
			item.to_dict(classes_by_id, total_pixels=total_pixels)
			for item in overall_counts
		],
		'splits': {
			split: _split_summary(inspection, split)
			for split in sorted(VALID_LABEL_SPLITS)
		},
		'unknown_colors': [
			item.to_dict()
			for item in inspection.unknown_colors()
		],
		'warnings': list(inspection.warnings),
		'files': [
			file_result.to_dict(classes_by_id)
			for file_result in inspection.files
		],
	}


def render_png_label_summary_markdown(
	inspection: F3PngLabelInspection,
) -> str:
	"""Render a Markdown summary for the F3 PNG label inspection."""
	classes_by_id = inspection.classes_by_id
	total_pixels = inspection.total_pixel_count()
	lines = [
		'# F3 PNG label inspection',
		'',
		f'- F3 root: `{inspection.f3_root}`',
		f'- class_info: `{inspection.class_info_path}`',
		f'- PNG files: {len(inspection.files)}',
		f'- total pixels: {total_pixels}',
		f'- unknown pixels: {inspection.total_unknown_pixel_count()}',
		'',
		'## Overall class distribution',
		'',
		'| class_id | class_name | color | pixel_count | fraction |',
		'|---:|---|---|---:|---:|',
	]
	lines.extend(
		_render_count_row(
			count,
			classes_by_id=classes_by_id,
			total_pixels=total_pixels,
		)
		for count in inspection.overall_class_counts()
	)
	lines.extend(
		[
			'',
			'## Split summary',
			'',
			'| split | files | total_pixels | unknown_pixels |',
			'|---|---:|---:|---:|',
		],
	)
	for split in sorted(VALID_LABEL_SPLITS):
		files = inspection.files_for_split(split)
		lines.append(
			f'| {split} | {len(files)} | '
			f'{sum(item.pixel_count for item in files)} | '
			f'{sum(item.unknown_pixel_count for item in files)} |',
		)
	lines.extend(['', '## Unknown colors', ''])
	if inspection.unknown_colors():
		lines.extend(
			f'- `{item.hex_color}` {list(item.rgb)}: {item.pixel_count}'
			for item in inspection.unknown_colors()
		)
	else:
		lines.append('- none')
	lines.extend(['', '## Warnings', ''])
	if inspection.warnings:
		lines.extend(f'- {warning}' for warning in inspection.warnings)
	else:
		lines.append('- none')
	return '\n'.join(lines) + '\n'


def write_f3_png_label_inspection_outputs(
	inspection: F3PngLabelInspection,
	outputs: F3PngLabelOutputConfig,
) -> None:
	"""Write CSV, JSON, Markdown, and PNG figure artifacts."""
	_write_inventory_csv(outputs.inventory_csv, inspection.files)
	_write_json(outputs.inventory_json, png_label_inventory_to_dict(inspection))
	_write_json(outputs.palette_json, facies_palette_to_dict(inspection))
	_write_class_counts_csv(outputs.class_counts_csv, inspection)
	_write_json(outputs.summary_json, png_label_inspection_to_dict(inspection))
	_write_text(
		outputs.summary_markdown,
		render_png_label_summary_markdown(inspection),
	)
	save_png_label_distribution_figures(inspection, outputs)


def png_label_inventory_to_dict(
	inspection: F3PngLabelInspection,
) -> dict[str, object]:
	"""Return the dedicated machine-readable PNG-label inventory."""
	return {
		'f3_root': str(inspection.f3_root),
		'class_info_path': str(inspection.class_info_path),
		'file_count': len(inspection.files),
		'total_pixels': inspection.total_pixel_count(),
		'total_unknown_pixels': inspection.total_unknown_pixel_count(),
		'files': [
			file_result.to_inventory_dict()
			for file_result in inspection.files
		],
		'warnings': list(inspection.warnings),
	}


def facies_palette_to_dict(
	inspection: F3PngLabelInspection,
) -> dict[str, object]:
	"""Return the fixed facies label palette derived from class-info RGBs."""
	return {
		'class_info_path': str(inspection.class_info_path),
		'palette_source': 'interpretation/class_info.json exact RGB values',
		'class_count': len(inspection.classes),
		'classes': [item.to_dict() for item in inspection.classes],
	}


def save_png_label_distribution_figures(
	inspection: F3PngLabelInspection,
	outputs: F3PngLabelOutputConfig,
) -> None:
	"""Save train, validation, and per-slice class distribution figures."""
	_save_split_distribution_png(
		inspection,
		'train',
		outputs.class_distribution_train_png,
		dpi=outputs.dpi,
	)
	_save_split_distribution_png(
		inspection,
		'validation',
		outputs.class_distribution_validation_png,
		dpi=outputs.dpi,
	)
	_save_per_slice_distribution_png(
		inspection,
		outputs.class_distribution_per_slice_png,
		dpi=outputs.dpi,
	)


def _inspect_png_file(
	root: Path,
	path: Path,
	*,
	classes: Sequence[F3ClassInfo],
	allow_unknown_colors: bool,
) -> PngLabelFileInspection:
	rgb_image = read_png_rgb(path)
	label_map = rgb_to_class_id_map(
		rgb_image,
		classes,
		allow_unknown_colors=allow_unknown_colors,
	)
	relative = path.relative_to(root)
	split = extract_label_split(relative)
	if split not in VALID_LABEL_SPLITS:
		msg = f'PNG label path must contain train or validation split: {relative}'
		raise ValueError(msg)
	name_parts = parse_label_png_name(path.name)
	height, width = rgb_image.shape[:2]
	return PngLabelFileInspection(
		relative_path=relative.as_posix(),
		absolute_path=str(path.resolve(strict=False)),
		split=split,
		slice_type=name_parts.slice_type,
		slice_index=name_parts.slice_index,
		width=int(width),
		height=int(height),
		pixel_count=int(width * height),
		unknown_pixel_count=label_map.unknown_pixel_count,
		class_counts=count_class_pixels(label_map.class_id_map, classes),
		unknown_colors=label_map.unknown_colors,
	)


def _find_label_png_paths(
	root: Path,
	*,
	candidate_extensions: Sequence[str],
) -> tuple[Path, ...]:
	suffixes = _normalize_suffixes(candidate_extensions)
	paths: list[Path] = []
	for split in sorted(VALID_LABEL_SPLITS):
		split_dir = root / 'interpretation' / split
		if not split_dir.is_dir():
			continue
		paths.extend(
			path
			for path in split_dir.iterdir()
			if path.is_file() and path.suffix.lower() in suffixes
		)
	return tuple(sorted(paths, key=lambda path: _label_path_sort_key(root, path)))


def _normalize_suffixes(values: Sequence[str]) -> tuple[str, ...]:
	suffixes: list[str] = []
	for value in values:
		if not isinstance(value, str) or not value:
			msg = f'candidate_extensions must contain non-empty strings: {values!r}'
			raise TypeError(msg)
		suffix = value.lower()
		if not suffix.startswith('.'):
			suffix = f'.{suffix}'
		suffixes.append(suffix)
	return tuple(dict.fromkeys(suffixes))


def _label_path_sort_key(root: Path, path: Path) -> tuple[int, int, int, str]:
	relative = path.relative_to(root)
	split = extract_label_split(relative)
	name_parts = parse_label_png_name(path.name)
	return (
		0 if split == 'train' else 1,
		_SLICE_TYPE_ORDER[name_parts.slice_type],
		name_parts.slice_index if name_parts.slice_index is not None else 10**12,
		relative.as_posix().lower(),
	)


def _metadata_warnings(
	files: Sequence[PngLabelFileInspection],
) -> tuple[str, ...]:
	return tuple(
		'could not parse slice_type/slice_index from PNG filename: '
		f'{file_result.relative_path}'
		for file_result in files
		if file_result.slice_type is None or file_result.slice_index is None
	)


def _unknown_color_warnings(
	files: Sequence[PngLabelFileInspection],
) -> tuple[str, ...]:
	total = sum(item.unknown_pixel_count for item in files)
	if total == 0:
		return ()
	return (f'unknown PNG label colors detected: {total} pixels',)


def _rgb_class_lookup(classes: Sequence[F3ClassInfo]) -> dict[int, int]:
	lookup: dict[int, int] = {}
	for item in classes:
		code = _pack_rgb(item.rgb)
		if code in lookup:
			msg = (
				'class_info contains duplicate RGB colors: '
				f'{rgb_to_hex(item.rgb)}'
			)
			raise ValueError(msg)
		lookup[code] = item.class_id
	return lookup


def _pack_rgb_codes(rgb: NDArray[np.uint8]) -> NDArray[np.uint32]:
	values = rgb.astype(np.uint32, copy=False)
	return (
		(values[:, 0] << np.uint32(16))
		| (values[:, 1] << np.uint32(8))
		| values[:, 2]
	)


def _pack_rgb(rgb: RGB) -> int:
	red, green, blue = rgb
	return (red << 16) | (green << 8) | blue


def _unpack_rgb(code: int) -> RGB:
	return (
		int((code >> 16) & 0xFF),
		int((code >> 8) & 0xFF),
		int(code & 0xFF),
	)


def _unknown_colors_from_codes(
	codes: NDArray[np.uint32],
) -> tuple[PngLabelUnknownColor, ...]:
	if codes.size == 0:
		return ()
	unique_codes, counts = np.unique(codes, return_counts=True)
	items = [
		PngLabelUnknownColor(
			rgb=_unpack_rgb(int(code)),
			pixel_count=int(count),
		)
		for code, count in zip(unique_codes, counts, strict=True)
	]
	return tuple(sorted(items, key=lambda item: (-item.pixel_count, item.rgb)))


def _split_summary(
	inspection: F3PngLabelInspection,
	split: str,
) -> dict[str, object]:
	files = inspection.files_for_split(split)
	total_pixels = int(sum(item.pixel_count for item in files))
	unknown_pixels = int(sum(item.unknown_pixel_count for item in files))
	return {
		'file_count': len(files),
		'total_pixels': total_pixels,
		'unknown_pixel_count': unknown_pixels,
		'class_counts': [
			item.to_dict(inspection.classes_by_id, total_pixels=total_pixels)
			for item in inspection.class_counts_for_files(files)
		],
	}


def _write_inventory_csv(
	path: str | Path,
	files: Sequence[PngLabelFileInspection],
) -> None:
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=PNG_LABEL_INVENTORY_FIELDNAMES)
		writer.writeheader()
		for file_result in files:
			writer.writerow(file_result.to_inventory_row())


def _write_class_counts_csv(
	path: str | Path,
	inspection: F3PngLabelInspection,
) -> None:
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(
			file_obj,
			fieldnames=PNG_LABEL_CLASS_COUNT_FIELDNAMES,
		)
		writer.writeheader()
		for row in _iter_class_count_rows(inspection):
			writer.writerow(row)


def _iter_class_count_rows(
	inspection: F3PngLabelInspection,
) -> tuple[dict[str, object], ...]:
	rows: list[dict[str, object]] = []
	for file_result in inspection.files:
		rows.extend(
			_class_count_rows(
				scope='per_png_file',
				relative_path=file_result.relative_path,
				split=file_result.split,
				slice_type=file_result.slice_type,
				slice_index=file_result.slice_index,
				counts=file_result.class_counts,
				total_pixels=file_result.pixel_count,
				unknown_pixel_count=file_result.unknown_pixel_count,
				classes_by_id=inspection.classes_by_id,
			),
		)
	for split in sorted(VALID_LABEL_SPLITS):
		files = inspection.files_for_split(split)
		total_pixels = int(sum(item.pixel_count for item in files))
		unknown_pixel_count = int(sum(item.unknown_pixel_count for item in files))
		rows.extend(
			_class_count_rows(
				scope='per_split',
				relative_path='',
				split=split,
				slice_type=None,
				slice_index=None,
				counts=inspection.class_counts_for_files(files),
				total_pixels=total_pixels,
				unknown_pixel_count=unknown_pixel_count,
				classes_by_id=inspection.classes_by_id,
			),
		)
	rows.extend(
		_class_count_rows(
			scope='overall',
			relative_path='',
			split='',
			slice_type=None,
			slice_index=None,
			counts=inspection.overall_class_counts(),
			total_pixels=inspection.total_pixel_count(),
			unknown_pixel_count=inspection.total_unknown_pixel_count(),
			classes_by_id=inspection.classes_by_id,
		),
	)
	return tuple(rows)


def _class_count_rows(  # noqa: PLR0913
	*,
	scope: str,
	relative_path: str,
	split: str,
	slice_type: str | None,
	slice_index: int | None,
	counts: Sequence[PngLabelClassCount],
	total_pixels: int,
	unknown_pixel_count: int,
	classes_by_id: Mapping[int, F3ClassInfo],
) -> tuple[dict[str, object], ...]:
	rows: list[dict[str, object]] = []
	for count in counts:
		class_info = classes_by_id[count.class_id]
		rows.append(
			{
				'scope': scope,
				'relative_path': relative_path,
				'split': split,
				'slice_type': slice_type,
				'slice_index': slice_index,
				'class_id': class_info.class_id,
				'class_name': class_info.class_name,
				'rgb': _format_rgb(class_info.rgb),
				'hex_color': class_info.hex_color,
				'pixel_count': count.pixel_count,
				'total_pixels': total_pixels,
				'unknown_pixel_count': unknown_pixel_count,
				'fraction': _fraction(count.pixel_count, total_pixels),
			},
		)
	return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _write_text(path: str | Path, content: str) -> None:
	text_path = Path(path)
	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text(content, encoding='utf-8')


def _render_count_row(
	count: PngLabelClassCount,
	*,
	classes_by_id: Mapping[int, F3ClassInfo],
	total_pixels: int,
) -> str:
	class_info = classes_by_id[count.class_id]
	return (
		f'| {class_info.class_id} | {class_info.class_name} | '
		f'`{class_info.hex_color}` | {count.pixel_count} | '
		f'{_fraction(count.pixel_count, total_pixels):.6f} |'
	)


def _format_rgb(rgb: RGB) -> str:
	return ','.join(str(value) for value in rgb)


def _format_unknown_colors(colors: Sequence[PngLabelUnknownColor]) -> str:
	return ';'.join(
		f'{item.hex_color}:{item.pixel_count}'
		for item in colors
	)


def _fraction(count: int, total: int) -> float:
	if total <= 0:
		return 0.0
	return float(count / total)


def _save_split_distribution_png(
	inspection: F3PngLabelInspection,
	split: str,
	path: str | Path,
	*,
	dpi: int,
) -> None:
	plt = _matplotlib_pyplot()
	files = inspection.files_for_split(split)
	total_pixels = int(sum(item.pixel_count for item in files))
	counts = inspection.class_counts_for_files(files)
	values = np.asarray([item.pixel_count for item in counts], dtype=np.int64)
	max_value = int(values.max()) if values.size else 0
	figure_path = Path(path)
	figure_path.parent.mkdir(parents=True, exist_ok=True)
	fig_height = max(3.2, 0.45 * len(inspection.classes) + 1.6)
	fig, ax = plt.subplots(figsize=(7.4, fig_height), dpi=dpi)
	y_positions = np.arange(len(inspection.classes))
	ax.barh(
		y_positions,
		values,
		color=[item.hex_color for item in inspection.classes],
		edgecolor='#262626',
		linewidth=0.4,
	)
	ax.set_yticks(y_positions)
	ax.set_yticklabels(_class_axis_labels(inspection.classes))
	ax.invert_yaxis()
	ax.set_xlabel('pixel count')
	ax.set_ylabel('class')
	ax.set_title(f'{split} class distribution')
	ax.grid(axis='x', color='#D9D9D9', linewidth=0.6)
	ax.set_axisbelow(True)
	if max_value == 0:
		ax.set_xlim(0, 1)
	else:
		ax.set_xlim(0, max_value * 1.18)
	for row_index, count in enumerate(counts):
		fraction = _fraction(count.pixel_count, total_pixels)
		label = f'{count.pixel_count:,} ({fraction:.1%})'
		x_position = count.pixel_count + max(max_value * 0.015, 1.0)
		ax.text(x_position, row_index, label, va='center', fontsize=8)
	fig.tight_layout()
	fig.savefig(figure_path, facecolor='white')
	plt.close(fig)


def _save_per_slice_distribution_png(
	inspection: F3PngLabelInspection,
	path: str | Path,
	*,
	dpi: int,
) -> None:
	plt = _matplotlib_pyplot()
	ticker = __import__('matplotlib.ticker', fromlist=['ticker'])
	figure_path = Path(path)
	figure_path.parent.mkdir(parents=True, exist_ok=True)
	max_files = max(
		(len(inspection.files_for_split(split)) for split in VALID_LABEL_SPLITS),
		default=1,
	)
	fig_width = min(18.0, max(8.0, 0.34 * max_files + 3.0))
	fig, axes = plt.subplots(2, 1, figsize=(fig_width, 6.6), dpi=dpi, sharey=True)
	for axis_index, split in enumerate(sorted(VALID_LABEL_SPLITS)):
		ax = axes[axis_index]
		files = inspection.files_for_split(split)
		_draw_split_slice_distribution(ax, files, inspection.classes)
		ax.set_title(split)
		ax.set_ylabel('class fraction')
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
		ax.grid(axis='y', color='#D9D9D9', linewidth=0.6)
		ax.set_axisbelow(True)
	axes[-1].set_xlabel('slice (I=inline, X=crossline)')
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(
		handles,
		labels,
		loc='lower center',
		ncol=min(4, max(1, len(labels))),
		frameon=False,
		fontsize=8,
	)
	fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
	fig.savefig(figure_path, facecolor='white')
	plt.close(fig)


def _draw_split_slice_distribution(
	ax: object,
	files: Sequence[PngLabelFileInspection],
	classes: Sequence[F3ClassInfo],
) -> None:
	if not files:
		ax.text(
			0.5,
			0.5,
			'No PNG labels',
			ha='center',
			va='center',
			transform=ax.transAxes,
		)
		ax.set_xticks([])
		ax.set_ylim(0.0, 1.0)
		return
	x_positions = np.arange(len(files))
	bottoms = np.zeros(len(files), dtype=np.float64)
	for class_info in classes:
		values = np.asarray(
			[
				_fraction(
					file_result.class_count_by_id()[class_info.class_id],
					file_result.pixel_count,
				)
				for file_result in files
			],
			dtype=np.float64,
		)
		ax.bar(
			x_positions,
			values,
			bottom=bottoms,
			color=class_info.hex_color,
			edgecolor='#FFFFFF',
			linewidth=0.25,
			label=f'{class_info.class_id}: {class_info.class_name}',
		)
		bottoms += values
	ax.set_ylim(0.0, 1.0)
	tick_step = max(1, math.ceil(len(files) / 24))
	tick_positions = x_positions[::tick_step]
	ax.set_xticks(tick_positions)
	ax.set_xticklabels(
		[_slice_tick_label(files[index]) for index in tick_positions],
		rotation=45,
		ha='right',
		fontsize=8,
	)


def _class_axis_labels(classes: Sequence[F3ClassInfo]) -> list[str]:
	return [f'{item.class_id}: {item.class_name}' for item in classes]


def _slice_tick_label(file_result: PngLabelFileInspection) -> str:
	prefix = 'I' if file_result.slice_type == 'inline' else 'X'
	if file_result.slice_index is None:
		return f'{prefix}?'
	return f'{prefix}{file_result.slice_index}'


def _matplotlib_image() -> object:
	try:
		return __import__('matplotlib.image', fromlist=['image'])
	except ImportError as exc:
		msg = (
			'F3 PNG label inspection requires matplotlib; '
			'install seis-cluster-ssl[visualization].'
		)
		raise ImportError(msg) from exc


def _matplotlib_pyplot() -> object:
	try:
		return __import__('matplotlib.pyplot', fromlist=['pyplot'])
	except ImportError as exc:
		msg = (
			'F3 PNG label distribution figures require matplotlib; '
			'install seis-cluster-ssl[visualization].'
		)
		raise ImportError(msg) from exc


__all__ = [
	'PNG_LABEL_CLASS_COUNT_FIELDNAMES',
	'PNG_LABEL_INVENTORY_FIELDNAMES',
	'F3PngLabelInspection',
	'F3PngLabelOutputConfig',
	'PngLabelClassCount',
	'PngLabelFileInspection',
	'PngLabelMap',
	'PngLabelUnknownColor',
	'count_class_pixels',
	'inspect_f3_png_labels',
	'normalize_png_rgb',
	'png_label_inspection_to_dict',
	'read_png_rgb',
	'render_png_label_summary_markdown',
	'rgb_to_class_id_map',
	'save_png_label_distribution_figures',
	'write_f3_png_label_inspection_outputs',
]
