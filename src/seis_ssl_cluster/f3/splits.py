"""F3 train/validation slice split helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from seis_ssl_cluster.f3.labels import VALID_LABEL_SPLITS, VALID_SLICE_TYPES

_SLICE_TYPE_ORDER = {'inline': 0, 'crossline': 1}


@dataclass(frozen=True)
class F3SliceSplitRecord:
	"""One supervised F3 slice selected by the PNG label inventory."""

	relative_path: str
	split: str
	slice_type: str
	slice_index: int
	absolute_path: str | None = None

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable split record."""
		payload: dict[str, object] = {
			'relative_path': self.relative_path,
			'split': self.split,
			'slice_type': self.slice_type,
			'slice_index': self.slice_index,
		}
		if self.absolute_path is not None:
			payload['absolute_path'] = self.absolute_path
		return payload


@dataclass(frozen=True)
class F3LineGeometry:
	"""Inline/crossline coordinate bounds for a prepared F3 cube."""

	shape_xyz: tuple[int, int, int]
	inline_min: int
	inline_max: int
	crossline_min: int
	crossline_max: int

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable geometry record."""
		return {
			'shape_xyz': list(self.shape_xyz),
			'inline_min': self.inline_min,
			'inline_max': self.inline_max,
			'crossline_min': self.crossline_min,
			'crossline_max': self.crossline_max,
		}


def load_f3_slice_split_records(path: str | Path) -> tuple[F3SliceSplitRecord, ...]:
	"""Load supervised slice records from a PNG-label inventory CSV or JSON."""
	inventory_path = Path(path)
	if inventory_path.suffix.lower() == '.csv':
		records = _records_from_csv(inventory_path)
	elif inventory_path.suffix.lower() == '.json':
		records = _records_from_json(inventory_path)
	else:
		msg = f'PNG label inventory must be CSV or JSON: {inventory_path}'
		raise ValueError(msg)
	return _validate_unique_slice_records(records)


def read_f3_line_geometry(path: str | Path) -> F3LineGeometry:
	"""Read F3 inline/crossline geometry from an inspection geometry JSON."""
	geometry_path = Path(path)
	with geometry_path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if not isinstance(payload, Mapping):
		msg = f'F3 geometry JSON must contain an object: {geometry_path}'
		raise TypeError(msg)
	return f3_line_geometry_from_mapping(payload)


def f3_line_geometry_from_mapping(payload: Mapping[str, object]) -> F3LineGeometry:
	"""Parse F3 line geometry from a SEGY inspection payload."""
	source = payload
	segy_files = payload.get('segy_files')
	if isinstance(segy_files, Mapping):
		label_geometry = segy_files.get('label')
		if not isinstance(label_geometry, Mapping):
			msg = 'segy_files.label must be a mapping'
			raise TypeError(msg)
		source = cast('Mapping[str, object]', label_geometry)
	elif isinstance(payload.get('label'), Mapping):
		source = cast('Mapping[str, object]', payload['label'])

	shape = _int_triplet(source.get('cube_shape'), label='label cube_shape')
	return F3LineGeometry(
		shape_xyz=shape,
		inline_min=_required_int(source, 'iline_min'),
		inline_max=_required_int(source, 'iline_max'),
		crossline_min=_required_int(source, 'xline_min'),
		crossline_max=_required_int(source, 'xline_max'),
	)


def resolve_f3_slice_array_index(
	record: F3SliceSplitRecord,
	geometry: F3LineGeometry,
) -> int:
	"""Convert an F3 inline/crossline number to a zero-based cube index."""
	if record.slice_type == 'inline':
		index = record.slice_index - geometry.inline_min
		_validate_array_index(
			index,
			axis_size=geometry.shape_xyz[0],
			label=f'inline {record.slice_index}',
		)
		return index
	if record.slice_type == 'crossline':
		index = record.slice_index - geometry.crossline_min
		_validate_array_index(
			index,
			axis_size=geometry.shape_xyz[1],
			label=f'crossline {record.slice_index}',
		)
		return index
	msg = f'slice_type must be inline or crossline; got {record.slice_type!r}'
	raise ValueError(msg)


def f3_slice_split_manifest(
	records: Sequence[F3SliceSplitRecord],
) -> dict[str, object]:
	"""Return a deterministic slice-level split manifest without random splits."""
	sorted_records = _sort_records(records)
	return {
		'split_source': 'png_label_inventory',
		'split_unit': 'slice',
		'strategy': 'inventory_split_no_random_token_split',
		'no_random_split': True,
		'splits': {
			split: [
				record.to_dict()
				for record in sorted_records
				if record.split == split
			]
			for split in sorted(VALID_LABEL_SPLITS)
		},
	}


def _records_from_csv(path: Path) -> tuple[F3SliceSplitRecord, ...]:
	with path.open(encoding='utf-8', newline='') as file_obj:
		reader = csv.DictReader(file_obj)
		return tuple(_record_from_mapping(row, source=path) for row in reader)


def _records_from_json(path: Path) -> tuple[F3SliceSplitRecord, ...]:
	with path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if isinstance(payload, Mapping):
		files = payload.get('files')
	elif isinstance(payload, list):
		files = payload
	else:
		msg = f'PNG label inventory JSON must contain an object or list: {path}'
		raise TypeError(msg)
	if not isinstance(files, list):
		msg = f'PNG label inventory JSON must contain a files list: {path}'
		raise TypeError(msg)
	return tuple(
		_record_from_mapping(_mapping(item, label='inventory file'), source=path)
		for item in files
	)


def _record_from_mapping(
	row: Mapping[str, object],
	*,
	source: Path,
) -> F3SliceSplitRecord:
	split = _required_str(row, 'split', source=source).lower()
	slice_type = _required_str(row, 'slice_type', source=source).lower()
	if split not in VALID_LABEL_SPLITS:
		msg = f'invalid split {split!r} in {source}'
		raise ValueError(msg)
	if slice_type not in VALID_SLICE_TYPES:
		msg = f'invalid slice_type {slice_type!r} in {source}'
		raise ValueError(msg)
	return F3SliceSplitRecord(
		relative_path=_required_str(row, 'relative_path', source=source),
		absolute_path=_optional_str(row.get('absolute_path')),
		split=split,
		slice_type=slice_type,
		slice_index=_required_int(row, 'slice_index'),
	)


def _validate_unique_slice_records(
	records: Sequence[F3SliceSplitRecord],
) -> tuple[F3SliceSplitRecord, ...]:
	seen: dict[tuple[str, int], F3SliceSplitRecord] = {}
	for record in records:
		key = (record.slice_type, record.slice_index)
		previous = seen.get(key)
		if previous is not None:
			msg = (
				'duplicate F3 supervised slice in PNG inventory: '
				f'{record.slice_type} {record.slice_index} '
				f'({previous.split}, {record.split})'
			)
			raise ValueError(msg)
		seen[key] = record
	return _sort_records(records)


def _sort_records(
	records: Sequence[F3SliceSplitRecord],
) -> tuple[F3SliceSplitRecord, ...]:
	return tuple(
		sorted(
			records,
			key=lambda record: (
				0 if record.split == 'train' else 1,
				_SLICE_TYPE_ORDER[record.slice_type],
				record.slice_index,
				record.relative_path,
			),
		),
	)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		msg = f'{label} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _required_str(
	parent: Mapping[str, object],
	key: str,
	*,
	source: Path,
) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{source} field {key!r} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_str(value: object) -> str | None:
	if value is None or value == '':
		return None
	if not isinstance(value, str):
		msg = f'optional path field must be a string; got {value!r}'
		raise TypeError(msg)
	return value


def _required_int(parent: Mapping[str, object], key: str) -> int:
	value = parent.get(key)
	if isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	try:
		return int(cast('object', value))
	except (TypeError, ValueError) as exc:
		msg = f'{key} must be an integer; got {value!r}'
		raise ValueError(msg) from exc


def _int_triplet(value: object, *, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		msg = f'{label} must be a length-3 sequence; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
		msg = f'{label} values must be integers; got {value!r}'
		raise TypeError(msg)
	shape = cast('tuple[int, int, int]', values)
	if any(axis <= 0 for axis in shape):
		msg = f'{label} values must be positive; got {shape!r}'
		raise ValueError(msg)
	return shape


def _validate_array_index(index: int, *, axis_size: int, label: str) -> None:
	if index < 0 or index >= axis_size:
		msg = (
			f'{label} resolves outside F3 cube axis; '
			f'array_index={index}, axis_size={axis_size}'
		)
		raise ValueError(msg)


__all__ = [
	'F3LineGeometry',
	'F3SliceSplitRecord',
	'f3_line_geometry_from_mapping',
	'f3_slice_split_manifest',
	'load_f3_slice_split_records',
	'read_f3_line_geometry',
	'resolve_f3_slice_array_index',
]
