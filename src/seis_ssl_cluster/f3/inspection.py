"""File inventory inspection for the F3 facies benchmark raw directory."""

from __future__ import annotations

import csv
import fnmatch
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.f3.labels import (
	VALID_LABEL_SPLITS,
	F3ClassInfo,
	extract_label_split,
	parse_label_png_name,
	read_class_info,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

CATEGORY_SEISMIC_SEGY = 'seismic_segy'
CATEGORY_LABEL_SEGY = 'label_segy'
CATEGORY_CLASS_INFO = 'class_info'
CATEGORY_LABEL_PNG = 'label_png'
CATEGORY_OTHER = 'other'

INVENTORY_FIELDNAMES = (
	'relative_path',
	'absolute_path',
	'suffix',
	'size_bytes',
	'category',
	'split',
	'slice_type',
	'slice_index',
)

_SEGY_SUFFIXES = frozenset({'.sgy', '.segy'})
_INVENTORY_CATEGORIES = (
	CATEGORY_SEISMIC_SEGY,
	CATEGORY_LABEL_SEGY,
	CATEGORY_CLASS_INFO,
	CATEGORY_LABEL_PNG,
	CATEGORY_OTHER,
)
_SLICE_TYPES = ('inline', 'crossline')


@dataclass(frozen=True)
class F3FileRecord:
	"""One file record in the F3 raw-directory inventory."""

	relative_path: str
	absolute_path: str
	suffix: str
	size_bytes: int
	category: str
	split: str | None
	slice_type: str | None
	slice_index: int | None

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON/CSV-serializable file record."""
		return {
			'relative_path': self.relative_path,
			'absolute_path': self.absolute_path,
			'suffix': self.suffix,
			'size_bytes': self.size_bytes,
			'category': self.category,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
		}


@dataclass(frozen=True)
class F3FileInventory:
	"""Complete F3 file inventory with normalized class information."""

	f3_root: Path
	class_info_path: Path
	files: tuple[F3FileRecord, ...]
	classes: tuple[F3ClassInfo, ...]
	warnings: tuple[str, ...]

	def label_png_files(self) -> tuple[F3FileRecord, ...]:
		"""Return label PNG records only."""
		return tuple(
			record
			for record in self.files
			if record.category == CATEGORY_LABEL_PNG
		)

	def category_counts(self) -> dict[str, int]:
		"""Return file counts for every inventory category."""
		counts = Counter(record.category for record in self.files)
		return {
			category: int(counts.get(category, 0))
			for category in _INVENTORY_CATEGORIES
		}

	def split_counts(self) -> dict[str, int]:
		"""Return label PNG counts by split."""
		counts = Counter(
			record.split
			for record in self.label_png_files()
			if record.split in VALID_LABEL_SPLITS
		)
		return {
			split: int(counts.get(split, 0))
			for split in sorted(VALID_LABEL_SPLITS)
		}


@dataclass(frozen=True)
class F3InventoryOutputConfig:
	"""Destination paths for F3 file-inventory artifacts."""

	file_inventory_json: Path
	file_inventory_markdown: Path
	class_info_json: Path
	label_png_inventory_csv: Path
	file_inventory_csv: Path | None = None


def scan_f3_file_inventory(
	f3_root: str | Path,
	*,
	include_globs: Sequence[str] = ('**/*',),
	exclude_globs: Sequence[str] = (),
) -> F3FileInventory:
	"""Scan an F3 raw directory and parse class-info plus label PNG metadata."""
	root = Path(f3_root)
	if not root.is_dir():
		msg = f'F3 root directory does not exist: {root}'
		raise FileNotFoundError(msg)

	files = tuple(_iter_inventory_files(root, include_globs, exclude_globs))
	class_info_path, class_info_warnings = find_class_info_file(root, files)
	classes = read_class_info(class_info_path)
	records = tuple(_make_file_record(root, path) for path in files)
	warnings = (
		*class_info_warnings,
		*_inventory_warnings(records),
	)
	return F3FileInventory(
		f3_root=root.resolve(strict=False),
		class_info_path=class_info_path.resolve(strict=False),
		files=records,
		classes=classes,
		warnings=warnings,
	)


def find_class_info_file(
	f3_root: str | Path,
	files: Sequence[Path] | None = None,
) -> tuple[Path, tuple[str, ...]]:
	"""Find `class_info.json` under an F3 root using case-insensitive matching."""
	root = Path(f3_root)
	candidate_files = tuple(root.rglob('*')) if files is None else tuple(files)
	candidates = sorted(
		(
			path
			for path in candidate_files
			if path.is_file() and path.name.lower() == 'class_info.json'
		),
		key=lambda path: path.relative_to(root).as_posix().lower(),
	)
	if not candidates:
		msg = f'missing F3 class_info.json under {root}'
		raise FileNotFoundError(msg)
	warnings: tuple[str, ...] = ()
	if len(candidates) > 1:
		used = candidates[0].relative_to(root).as_posix()
		warnings = (
			f'class_info.json が複数見つかりました。使用: {used}',
		)
	return candidates[0], warnings


def write_f3_file_inventory_outputs(
	inventory: F3FileInventory,
	outputs: F3InventoryOutputConfig,
) -> None:
	"""Write JSON, Markdown, class-info JSON, and label PNG CSV artifacts."""
	_write_json(outputs.file_inventory_json, file_inventory_to_dict(inventory))
	if outputs.file_inventory_csv is not None:
		_write_file_records_csv(outputs.file_inventory_csv, inventory.files)
	_write_text(
		outputs.file_inventory_markdown,
		render_file_inventory_markdown(inventory),
	)
	_write_json(outputs.class_info_json, class_info_inventory_to_dict(inventory))
	_write_file_records_csv(
		outputs.label_png_inventory_csv,
		inventory.label_png_files(),
	)


def file_inventory_to_dict(inventory: F3FileInventory) -> dict[str, object]:
	"""Return the machine-readable file inventory payload."""
	return {
		'f3_root': str(inventory.f3_root),
		'class_info_path': str(inventory.class_info_path),
		'file_count': len(inventory.files),
		'category_counts': inventory.category_counts(),
		'split_counts': inventory.split_counts(),
		'warnings': list(inventory.warnings),
		'files': [record.to_dict() for record in inventory.files],
	}


def class_info_inventory_to_dict(
	inventory: F3FileInventory,
) -> dict[str, object]:
	"""Return the normalized class-info artifact payload."""
	return {
		'source_path': str(inventory.class_info_path),
		'class_count': len(inventory.classes),
		'classes': [item.to_dict() for item in inventory.classes],
	}


def render_file_inventory_markdown(inventory: F3FileInventory) -> str:
	"""Render a Japanese human-readable F3 file-inventory summary."""
	lines = [
		'# F3ファイルインベントリ',
		'',
		f'- F3 root: `{inventory.f3_root}`',
		f'- ファイル総数: {len(inventory.files)}',
		'',
		'## SEGYファイル',
		'',
		*_render_segy_rows(inventory.files),
		'',
		'## class_infoのclass一覧',
		'',
		'| class_id | class_name | rgb | hex_color |',
		'|---:|---|---|---|',
	]
	lines.extend(
		f'| {item.class_id} | {item.class_name} | '
		f'{list(item.rgb)} | `{item.hex_color}` |'
		for item in inventory.classes
	)
	lines.extend(
		[
			'',
			'## PNG枚数',
			'',
			f'- train PNG枚数: {inventory.split_counts()["train"]}',
			f'- validation PNG枚数: {inventory.split_counts()["validation"]}',
			'',
			'## inline/crossline slice index一覧',
			'',
			*_render_slice_index_rows(inventory.label_png_files()),
			'',
			'## 注意・警告',
			'',
		],
	)
	if inventory.warnings:
		lines.extend(f'- {warning}' for warning in inventory.warnings)
	else:
		lines.append('- 警告なし')
	return '\n'.join(lines) + '\n'


def _iter_inventory_files(
	root: Path,
	include_globs: Sequence[str],
	exclude_globs: Sequence[str],
) -> tuple[Path, ...]:
	files: list[Path] = []
	for path in root.rglob('*'):
		if not path.is_file():
			continue
		relative_path = path.relative_to(root).as_posix()
		if not any(_matches_glob(relative_path, pattern) for pattern in include_globs):
			continue
		if any(_matches_glob(relative_path, pattern) for pattern in exclude_globs):
			continue
		files.append(path)
	return tuple(
		sorted(
			files,
			key=lambda path: path.relative_to(root).as_posix().lower(),
		),
	)


def _matches_glob(relative_path: str, pattern: str) -> bool:
	normalized = pattern.replace('\\', '/')
	if normalized in {'**', '**/*'}:
		return True
	if fnmatch.fnmatchcase(relative_path, normalized):
		return True
	if normalized.startswith('**/'):
		return fnmatch.fnmatchcase(relative_path, normalized[3:])
	return False


def _make_file_record(root: Path, path: Path) -> F3FileRecord:
	relative = path.relative_to(root)
	suffix = path.suffix.lower()
	split = extract_label_split(relative)
	label_parts = parse_label_png_name(path.name)
	category = _classify_file(path, suffix=suffix, split=split)
	return F3FileRecord(
		relative_path=relative.as_posix(),
		absolute_path=str(path.resolve(strict=False)),
		suffix=suffix,
		size_bytes=path.stat().st_size,
		category=category,
		split=split if category == CATEGORY_LABEL_PNG else None,
		slice_type=(
			label_parts.slice_type
			if category == CATEGORY_LABEL_PNG
			else None
		),
		slice_index=(
			label_parts.slice_index
			if category == CATEGORY_LABEL_PNG
			else None
		),
	)


def _classify_file(path: Path, *, suffix: str, split: str | None) -> str:
	name = path.name.lower()
	stem = path.stem.lower()
	if suffix == '.json' and name == 'class_info.json':
		return CATEGORY_CLASS_INFO
	if suffix in _SEGY_SUFFIXES:
		if 'label' in stem:
			return CATEGORY_LABEL_SEGY
		if 'seismic' in stem:
			return CATEGORY_SEISMIC_SEGY
	if suffix == '.png' and split in VALID_LABEL_SPLITS:
		return CATEGORY_LABEL_PNG
	return CATEGORY_OTHER


def _inventory_warnings(records: Sequence[F3FileRecord]) -> tuple[str, ...]:
	counts = Counter(record.category for record in records)
	warnings: list[str] = []
	if counts.get(CATEGORY_SEISMIC_SEGY, 0) == 0:
		warnings.append('seismic SEGYファイルが見つかりません。')
	if counts.get(CATEGORY_LABEL_SEGY, 0) == 0:
		warnings.append('label SEGYファイルが見つかりません。')
	if counts.get(CATEGORY_LABEL_PNG, 0) == 0:
		warnings.append('train/validation label PNGが見つかりません。')
	for record in records:
		if record.category != CATEGORY_LABEL_PNG:
			continue
		if record.slice_type is None or record.slice_index is None:
			warnings.append(
				'label PNG名からslice_type/slice_indexを抽出できません: '
				f'{record.relative_path}',
			)
	return tuple(warnings)


def _render_segy_rows(records: Sequence[F3FileRecord]) -> list[str]:
	segy_records = [
		record
		for record in records
		if record.category in {CATEGORY_SEISMIC_SEGY, CATEGORY_LABEL_SEGY}
	]
	if not segy_records:
		return ['SEGYファイルは見つかりませんでした。']
	lines = [
		'| category | relative_path | size_bytes |',
		'|---|---|---:|',
	]
	lines.extend(
		f'| {record.category} | `{record.relative_path}` | {record.size_bytes} |'
		for record in segy_records
	)
	return lines


def _render_slice_index_rows(records: Sequence[F3FileRecord]) -> list[str]:
	lines: list[str] = []
	for split in sorted(VALID_LABEL_SPLITS):
		for slice_type in _SLICE_TYPES:
			indices = sorted(
				{
					record.slice_index
					for record in records
					if record.split == split
					and record.slice_type == slice_type
					and record.slice_index is not None
				},
			)
			lines.append(
				f'- {split} {slice_type}: {_format_indices(indices)}',
			)
	return lines


def _format_indices(indices: Sequence[int]) -> str:
	if not indices:
		return 'なし'
	return ', '.join(str(index) for index in indices)


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


def _write_file_records_csv(
	path: str | Path,
	records: Sequence[F3FileRecord],
) -> None:
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=INVENTORY_FIELDNAMES)
		writer.writeheader()
		for record in records:
			writer.writerow(record.to_dict())


__all__ = [
	'CATEGORY_CLASS_INFO',
	'CATEGORY_LABEL_PNG',
	'CATEGORY_LABEL_SEGY',
	'CATEGORY_OTHER',
	'CATEGORY_SEISMIC_SEGY',
	'F3FileInventory',
	'F3FileRecord',
	'F3InventoryOutputConfig',
	'class_info_inventory_to_dict',
	'file_inventory_to_dict',
	'find_class_info_file',
	'render_file_inventory_markdown',
	'scan_f3_file_inventory',
	'write_f3_file_inventory_outputs',
]
