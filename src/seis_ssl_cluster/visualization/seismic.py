"""Shared seismic amplitude rendering helpers."""

from __future__ import annotations

import numpy as np

from seis_ssl_cluster.visualization.style import aspect_for_view, origin_for_view


def amplitude_clip_limits(
	values: object,
	*,
	clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
	"""Return finite-value percentile display limits for seismic amplitudes."""
	_validate_clip_percentiles(clip_percentiles)
	array = np.asarray(values, dtype=np.float64)
	finite = array[np.isfinite(array)]
	if finite.size == 0:
		return 0.0, 1.0
	vmin, vmax = np.percentile(finite, clip_percentiles)
	if np.isclose(vmin, vmax):
		center = float(np.mean(finite))
		half_width = float(np.std(finite)) or 1.0
		return center - half_width, center + half_width
	return float(vmin), float(vmax)


def seismic_imshow(
	ax: object,
	image: object,
	*,
	view: str,
	clip_percentiles: tuple[float, float] = (1.0, 99.0),
	cmap: str = 'gray',
	**imshow_kwargs: object,
) -> object:
	"""Render a seismic image on an axes using shared view conventions."""
	vmin, vmax = amplitude_clip_limits(
		image,
		clip_percentiles=clip_percentiles,
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


__all__ = [
	'amplitude_clip_limits',
	'seismic_imshow',
]
