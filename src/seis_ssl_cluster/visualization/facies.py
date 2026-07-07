"""Shared facies label rendering helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from seis_ssl_cluster.visualization.style import aspect_for_view, origin_for_view

_INVALID_LABEL_RGB = (226, 226, 226)


def facies_palette(classes: Sequence[object]) -> dict[int, tuple[float, float, float]]:
	"""Return a class-id to RGB-float palette for facies class records."""
	return {
		_class_id(class_info): _rgb_float(_rgb(class_info))
		for class_info in classes
	}


def label_imshow(
	ax: object,
	labels: object,
	*,
	classes: Sequence[object],
	view: str,
	**imshow_kwargs: object,
) -> object:
	"""Render integer facies labels on an axes using shared view conventions."""
	rgb = class_id_image_to_rgb(labels, classes)
	options = {
		'origin': origin_for_view(view),
		'aspect': aspect_for_view(view),
		'interpolation': 'nearest',
	}
	options.update(imshow_kwargs)
	return ax.imshow(rgb, **options)


def class_id_image_to_rgb(
	class_ids: object,
	classes: Sequence[object],
	*,
	invalid_rgb: tuple[int, int, int] = _INVALID_LABEL_RGB,
) -> np.ndarray:
	"""Render a 2D integer class-ID image using class-info RGB colors."""
	array = np.asarray(class_ids)
	if array.ndim != 2:
		msg = f'class_ids must be 2D; got shape={array.shape!r}'
		raise ValueError(msg)
	ids = _normalize_class_ids(array)
	rgb = np.full((*ids.shape, 3), invalid_rgb, dtype=np.uint8)
	for class_info in classes:
		rgb[ids == _class_id(class_info)] = _rgb(class_info)
	return rgb


def facies_legend_handles(classes: Sequence[object]) -> list[object]:
	"""Return matplotlib legend handles for facies classes."""
	from matplotlib.patches import Patch  # noqa: PLC0415

	return [
		Patch(
			facecolor=_rgb_float(_rgb(class_info)),
			label=f'{_class_id(class_info)}: {_class_name(class_info)}',
		)
		for class_info in classes
	]


def _normalize_class_ids(array: np.ndarray) -> np.ndarray:
	values = np.asarray(array, dtype=np.float64)
	finite = np.isfinite(values)
	rounded = np.rint(values)
	if not np.allclose(values[finite], rounded[finite]):
		msg = 'label class image must contain integer-like values'
		raise ValueError(msg)
	ids = np.full(values.shape, -1, dtype=np.int64)
	ids[finite] = rounded[finite].astype(np.int64)
	return ids


def _class_id(class_info: object) -> int:
	value = _field(class_info, 'class_id')
	return int(value)


def _class_name(class_info: object) -> str:
	return str(_field(class_info, 'class_name'))


def _rgb(class_info: object) -> tuple[int, int, int]:
	value = _field(class_info, 'rgb')
	if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
		msg = f'class rgb must be a 3-item sequence; got {value!r}'
		raise ValueError(msg)
	return tuple(int(channel) for channel in value)


def _rgb_float(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
	return tuple(float(channel) / 255.0 for channel in rgb)


def _field(class_info: object, name: str) -> object:
	if isinstance(class_info, Mapping):
		return class_info[name]
	return getattr(class_info, name)


__all__ = [
	'class_id_image_to_rgb',
	'facies_legend_handles',
	'facies_palette',
	'label_imshow',
]
