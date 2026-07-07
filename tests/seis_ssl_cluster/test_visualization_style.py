from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from seis_ssl_cluster.visualization.facies import facies_palette, label_imshow
from seis_ssl_cluster.visualization.seismic import (
	amplitude_clip_limits,
	seismic_imshow,
)
from seis_ssl_cluster.visualization.style import aspect_for_view, origin_for_view

plt = pytest.importorskip('matplotlib.pyplot')


@dataclass(frozen=True)
class _ClassInfo:
	class_id: int
	class_name: str
	rgb: tuple[int, int, int]


def test_origin_and_aspect_for_representative_views() -> None:
	assert origin_for_view('xy') == 'lower'
	assert aspect_for_view('xy') == 'equal'
	assert origin_for_view('timeslice') == 'lower'
	assert aspect_for_view('depth_slice') == 'equal'
	assert origin_for_view('xz') == 'upper'
	assert aspect_for_view('xz') == 'auto'
	assert origin_for_view('yz') == 'upper'
	assert aspect_for_view('inline') == 'auto'
	assert origin_for_view('crossline') == 'upper'


def test_origin_for_f3_lithology_slice_views() -> None:
	assert origin_for_view('inline') == 'upper'
	assert origin_for_view('crossline') == 'upper'
	assert origin_for_view('z') == 'lower'


def test_unknown_view_raises_value_error() -> None:
	with pytest.raises(ValueError, match='unknown view'):
		origin_for_view('diagonal')
	with pytest.raises(ValueError, match='unknown view'):
		aspect_for_view('diagonal')


def test_amplitude_clip_limits_ignores_nan_and_inf() -> None:
	values = np.asarray([np.nan, -np.inf, -10.0, 0.0, 10.0, np.inf])

	assert amplitude_clip_limits(values, clip_percentiles=(0.0, 100.0)) == (
		-10.0,
		10.0,
	)


def test_amplitude_clip_limits_returns_safe_default_for_no_finite_values() -> None:
	values = np.asarray([np.nan, np.inf, -np.inf])

	assert amplitude_clip_limits(values) == (0.0, 1.0)


def test_amplitude_clip_limits_supports_f3_constant_policies() -> None:
	values = np.asarray([2.0, 2.0, 2.0])

	assert amplitude_clip_limits(
		values,
		constant_policy='unit',
		constant_tolerance='exact',
	) == (1.0, 3.0)
	assert amplitude_clip_limits(values, constant_policy='none') == (None, None)


def test_seismic_imshow_and_label_imshow_render_small_arrays() -> None:
	fig, axes = plt.subplots(1, 2)
	try:
		seismic_image = seismic_imshow(
			axes[0],
			np.asarray([[0.0, 1.0], [np.nan, 2.0]]),
			view='xz',
		)
		label_image = label_imshow(
			axes[1],
			np.asarray([[0, 1], [1, 99]]),
			classes=(
				_ClassInfo(0, 'zero', (255, 0, 0)),
				_ClassInfo(1, 'one', (0, 255, 0)),
			),
			view='xy',
		)

		assert seismic_image.origin == 'upper'
		assert label_image.origin == 'lower'
		assert facies_palette((_ClassInfo(1, 'one', (0, 128, 255)),)) == {
			1: (0.0, 128.0 / 255.0, 1.0),
		}
	finally:
		plt.close(fig)
