from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.voxel_visualization import (
	F3LithologyVoxelFigureConfig,
	prepare_f3_lithology_voxel_slice,
	save_f3_lithology_voxel_slice_figure,
)
from seis_ssl_cluster.f3.splits import F3LineGeometry

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.parametrize(
	('slice_type', 'slice_index', 'expected'),
	[('inline', 100, (4, 3)), ('crossline', 200, (4, 2))],
)
def test_selected_voxel_slice_is_nonempty_and_does_not_mutate_prediction(
	tmp_path: Path,
	slice_type: str,
	slice_index: int,
	expected: tuple[int, int],
) -> None:
	pytest.importorskip('matplotlib.pyplot')
	shape = (2, 3, 4)
	seismic = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
	labels = np.zeros(shape, dtype=np.int16)
	labels[..., 2:] = 3
	predictions = labels.copy()
	predictions[0, 0, 0] = 3
	before = predictions.copy()
	confidence = np.full(shape, 0.75, dtype=np.float32)
	split = np.zeros(shape, dtype=np.uint8)
	split[0, :, :] = 2
	split[:, 0, :] = 2
	valid = np.ones(shape, dtype=bool)
	figure = prepare_f3_lithology_voxel_slice(
		slice_type,
		slice_index,
		seismic=seismic,
		labels=labels,
		predictions=predictions,
		confidence=confidence,
		split_grid=split,
		prediction_valid_mask=valid,
		geometry=F3LineGeometry(shape, 100, 101, 200, 202),
	)
	path = tmp_path / f'{slice_type}.png'
	save_f3_lithology_voxel_slice_figure(
		figure,
		path,
		classes=(
			{'class_id': 0, 'class_name': 'zero', 'rgb': [0, 0, 0]},
			{'class_id': 3, 'class_name': 'three', 'rgb': [255, 0, 0]},
		),
		config=F3LithologyVoxelFigureConfig(dpi=35, include_confidence=True),
		slice_metrics={'accuracy': '0.9', 'macro_f1': '0.8'},
	)

	assert figure.predictions.shape == expected
	assert figure.origin == 'upper'
	assert figure.aspect == 'auto'
	assert path.stat().st_size > 0
	assert np.array_equal(predictions, before)
	assert np.array_equal(
		figure.predictions, before[0].T if slice_type == 'inline' else before[:, 0, :].T
	)
