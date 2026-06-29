"""F3 facies label and class-info parsing helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_LABEL_SPLITS = frozenset({'train', 'validation'})
VALID_SLICE_TYPES = frozenset({'inline', 'crossline'})

_LABEL_PNG_NAME_RE = re.compile(
	r'^(?:.*_)?labels_(?P<slice_type>inline|crossline)_(?P<slice_index>\d+)\.png$',
	re.IGNORECASE,
)


@dataclass(frozen=True)
class F3ClassInfo:
	"""One facies class entry from `interpretation/class_info.json`."""

	class_id: int
	class_name: str
	rgb: tuple[int, int, int]

	@property
	def hex_color(self) -> str:
		"""Return the class RGB value as a stable uppercase `#RRGGBB` string."""
		return rgb_to_hex(self.rgb)

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable class-info record."""
		return {
			'class_id': self.class_id,
			'class_name': self.class_name,
			'rgb': list(self.rgb),
			'hex_color': self.hex_color,
		}


@dataclass(frozen=True)
class LabelPngNameParts:
	"""Parsed slice metadata from an F3 label PNG filename."""

	slice_type: str | None
	slice_index: int | None


def read_class_info(path: str | Path) -> tuple[F3ClassInfo, ...]:
	"""Read and normalize an F3 `class_info.json` file."""
	class_info_path = Path(path)
	with class_info_path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	return parse_class_info_payload(payload, source=class_info_path)


def parse_class_info_payload(
	payload: object,
	*,
	source: str | Path = 'class_info.json',
) -> tuple[F3ClassInfo, ...]:
	"""Normalize the F3 class-info JSON object into sorted class records."""
	if not isinstance(payload, Mapping):
		msg = f'{source} must contain a JSON object keyed by class id'
		raise TypeError(msg)

	classes: list[F3ClassInfo] = []
	seen_class_ids: set[int] = set()
	for raw_class_id, raw_class in payload.items():
		class_id = _parse_class_id(raw_class_id, source=source)
		if class_id in seen_class_ids:
			msg = (
				f'{source} contains duplicate class_id after int conversion: '
				f'{class_id}'
			)
			raise ValueError(msg)
		seen_class_ids.add(class_id)
		if not isinstance(raw_class, Mapping):
			msg = f'{source} class {raw_class_id!r} must be a JSON object'
			raise TypeError(msg)
		class_name = _parse_class_name(raw_class, raw_class_id, source=source)
		rgb = _parse_class_rgb(raw_class, raw_class_id, source=source)
		classes.append(
			F3ClassInfo(
				class_id=class_id,
				class_name=class_name,
				rgb=rgb,
			),
		)

	if not classes:
		msg = f'{source} must contain at least one class'
		raise ValueError(msg)
	return tuple(sorted(classes, key=lambda item: item.class_id))


def parse_label_png_name(filename: str | Path) -> LabelPngNameParts:
	"""Parse `inline`/`crossline` and numeric slice index from a label PNG name."""
	match = _LABEL_PNG_NAME_RE.match(Path(filename).name)
	if match is None:
		return LabelPngNameParts(slice_type=None, slice_index=None)
	return LabelPngNameParts(
		slice_type=match.group('slice_type').lower(),
		slice_index=int(match.group('slice_index')),
	)


def extract_label_split(relative_path: str | Path) -> str | None:
	"""Return `train` or `validation` when either appears in path components."""
	for part in Path(relative_path).parts:
		normalized = part.lower()
		if normalized in VALID_LABEL_SPLITS:
			return normalized
	return None


def rgb_to_hex(rgb: Sequence[int]) -> str:
	"""Convert an RGB triplet to a stable uppercase `#RRGGBB` string."""
	red, green, blue = _normalize_rgb(rgb)
	return f'#{red:02X}{green:02X}{blue:02X}'


def _parse_class_id(raw_class_id: object, *, source: str | Path) -> int:
	try:
		return int(raw_class_id)
	except (TypeError, ValueError) as exc:
		msg = f'{source} class id must be an integer-compatible key: {raw_class_id!r}'
		raise ValueError(msg) from exc


def _parse_class_name(
	raw_class: Mapping[str, Any],
	raw_class_id: object,
	*,
	source: str | Path,
) -> str:
	class_name = raw_class.get('name')
	if not isinstance(class_name, str) or not class_name:
		msg = f'{source} class {raw_class_id!r} must contain a non-empty string name'
		raise TypeError(msg)
	return class_name


def _parse_class_rgb(
	raw_class: Mapping[str, Any],
	raw_class_id: object,
	*,
	source: str | Path,
) -> tuple[int, int, int]:
	raw_rgb = raw_class.get('color')
	if raw_rgb is None:
		raw_rgb = raw_class.get('rgb')
	if not isinstance(raw_rgb, Sequence) or isinstance(raw_rgb, str | bytes):
		msg = f'{source} class {raw_class_id!r} must contain an RGB color list'
		raise TypeError(msg)
	try:
		return _normalize_rgb(raw_rgb)
	except (TypeError, ValueError) as exc:
		msg = f'{source} class {raw_class_id!r} has invalid RGB color: {raw_rgb!r}'
		raise ValueError(msg) from exc


def _normalize_rgb(rgb: Sequence[int]) -> tuple[int, int, int]:
	if len(rgb) != 3:
		msg = f'RGB values must have exactly three channels; got {rgb!r}'
		raise ValueError(msg)
	channels: list[int] = []
	for value in rgb:
		if not isinstance(value, int) or isinstance(value, bool):
			msg = f'RGB channels must be integers; got {rgb!r}'
			raise TypeError(msg)
		if value < 0 or value > 255:
			msg = f'RGB channels must be in [0, 255]; got {rgb!r}'
			raise ValueError(msg)
		channels.append(value)
	return channels[0], channels[1], channels[2]


__all__ = [
	'VALID_LABEL_SPLITS',
	'VALID_SLICE_TYPES',
	'F3ClassInfo',
	'LabelPngNameParts',
	'extract_label_split',
	'parse_class_info_payload',
	'parse_label_png_name',
	'read_class_info',
	'rgb_to_hex',
]
