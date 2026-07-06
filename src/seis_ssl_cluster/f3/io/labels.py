"""F3 facies label and class-info parsing helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
	from numpy.typing import NDArray

VALID_LABEL_SPLITS = frozenset({'train', 'validation'})
VALID_SLICE_TYPES = frozenset({'inline', 'crossline'})
RGB = tuple[int, int, int]

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
class PngLabelMap:
	"""Class-ID map converted from one RGB PNG label image."""

	class_id_map: NDArray[np.int32]
	unknown_colors: tuple[PngLabelUnknownColor, ...]

	@property
	def unknown_pixel_count(self) -> int:
		"""Return the total number of unknown-color pixels."""
		return int(sum(item.pixel_count for item in self.unknown_colors))


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


def rgb_to_class_id_map(
	image: NDArray[np.generic],
	classes: Sequence[F3ClassInfo],
	*,
	allow_unknown_colors: bool = False,
) -> PngLabelMap:
	"""Convert an RGB image to an integer class-ID map by exact RGB matching."""
	rgb_image = normalize_png_rgb(image)
	lookup = _rgb_class_lookup(classes)
	flat_codes = _pack_rgb_codes(rgb_image.reshape(-1, 3))
	flat_class_ids = np.full(flat_codes.shape, -1, dtype=np.int32)
	for rgb_code, class_id in lookup.items():
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


def read_png_rgb(path: str | Path) -> NDArray[np.uint8]:
	"""Read a PNG label image and return its RGB channels as `uint8`."""
	image_path = Path(path)
	image_module = _matplotlib_image()
	return normalize_png_rgb(image_module.imread(image_path), source=image_path)


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


def _format_unknown_colors(colors: Sequence[PngLabelUnknownColor]) -> str:
	return ';'.join(
		f'{item.hex_color}:{item.pixel_count}'
		for item in colors
	)


def _matplotlib_image() -> object:
	try:
		return __import__('matplotlib.image', fromlist=['image'])
	except ImportError as exc:
		msg = (
			'F3 PNG label inspection requires matplotlib; '
			'install seis-cluster-ssl[visualization].'
		)
		raise ImportError(msg) from exc


__all__ = [
	'RGB',
	'VALID_LABEL_SPLITS',
	'VALID_SLICE_TYPES',
	'F3ClassInfo',
	'LabelPngNameParts',
	'PngLabelMap',
	'PngLabelUnknownColor',
	'extract_label_split',
	'normalize_png_rgb',
	'parse_class_info_payload',
	'parse_label_png_name',
	'read_class_info',
	'read_png_rgb',
	'rgb_to_class_id_map',
	'rgb_to_hex',
]
