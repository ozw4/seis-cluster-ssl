"""Shared seismic amplitude rendering helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from seis_ssl_cluster.visualization.style import aspect_for_view, origin_for_view

ConstantAmplitudePolicy = Literal['expand', 'unit', 'none']
ConstantAmplitudeTolerance = Literal['exact', 'isclose']


def amplitude_clip_limits(
	values: object,
	*,
	clip_percentiles: tuple[float, float] = (1.0, 99.0),
	constant_policy: ConstantAmplitudePolicy = 'expand',
	constant_tolerance: ConstantAmplitudeTolerance = 'isclose',
) -> tuple[float | None, float | None]:
	"""Return finite-value percentile display limits for seismic amplitudes."""
	_validate_clip_percentiles(clip_percentiles)
	_validate_constant_policy(constant_policy)
	_validate_constant_tolerance(constant_tolerance)
	array = np.asarray(values, dtype=np.float64)
	finite = array[np.isfinite(array)]
	if finite.size == 0:
		if constant_policy == 'none':
			return None, None
		return 0.0, 1.0
	vmin, vmax = np.percentile(finite, clip_percentiles)
	if _is_constant_limit(vmin, vmax, constant_tolerance=constant_tolerance):
		if constant_policy == 'none':
			return None, None
		if constant_policy == 'unit':
			return float(vmin) - 1.0, float(vmax) + 1.0
		center = float(np.mean(finite))
		half_width = float(np.std(finite)) or 1.0
		return center - half_width, center + half_width
	return float(vmin), float(vmax)


def seismic_imshow(  # noqa: PLR0913
	ax: object,
	image: object,
	*,
	view: str,
	clip_percentiles: tuple[float, float] = (1.0, 99.0),
	constant_policy: ConstantAmplitudePolicy = 'expand',
	constant_tolerance: ConstantAmplitudeTolerance = 'isclose',
	cmap: str = 'gray',
	**imshow_kwargs: object,
) -> object:
	"""Render a seismic image on an axes using shared view conventions."""
	vmin, vmax = amplitude_clip_limits(
		image,
		clip_percentiles=clip_percentiles,
		constant_policy=constant_policy,
		constant_tolerance=constant_tolerance,
	)
	options = {
		'cmap': cmap,
		'origin': origin_for_view(view),
		'aspect': aspect_for_view(view),
		'vmin': vmin,
		'vmax': vmax,
	}
	options.update(imshow_kwargs)
	return ax.imshow(image, **options)


def _validate_clip_percentiles(value: tuple[float, float]) -> None:
	if len(value) != 2:
		msg = f'clip_percentiles must contain two values; got {value!r}'
		raise ValueError(msg)
	low, high = (float(value[0]), float(value[1]))
	if not 0.0 <= low < high <= 100.0:
		msg = f'clip_percentiles must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)


def _validate_constant_policy(value: str) -> None:
	if value not in {'expand', 'unit', 'none'}:
		msg = f'constant_policy must be expand, unit, or none; got {value!r}'
		raise ValueError(msg)


def _validate_constant_tolerance(value: str) -> None:
	if value not in {'exact', 'isclose'}:
		msg = f'constant_tolerance must be exact or isclose; got {value!r}'
		raise ValueError(msg)


def _is_constant_limit(
	vmin: float,
	vmax: float,
	*,
	constant_tolerance: ConstantAmplitudeTolerance,
) -> bool:
	if constant_tolerance == 'exact':
		return float(vmin) == float(vmax)
	return bool(np.isclose(vmin, vmax))


__all__ = [
	'amplitude_clip_limits',
	'seismic_imshow',
]
