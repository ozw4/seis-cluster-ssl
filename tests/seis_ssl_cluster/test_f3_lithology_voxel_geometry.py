from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.voxel_geometry import (
	project_token_grid_nearest,
	valid_tokens_to_voxel_mask,
)


def test_project_token_grid_nearest_preserves_trailing_axis_and_crops() -> None:
	values = np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3)
	original = values.copy()

	actual = project_token_grid_nearest(
		values,
		patch_size_xyz=(2, 3, 2),
		volume_shape_xyz=(3, 5, 4),
	)

	assert actual.shape == (3, 5, 4, 3)
	np.testing.assert_array_equal(actual[2, 4, 3], values[1, 1, 1])
	np.testing.assert_array_equal(values, original)


def test_project_token_grid_nearest_divisible_volume() -> None:
	values = np.arange(8).reshape(2, 2, 2)
	actual = project_token_grid_nearest(
		values,
		patch_size_xyz=(2, 2, 2),
		volume_shape_xyz=(4, 4, 4),
	)
	assert actual.shape == (4, 4, 4)
	assert np.all(actual[:2, :2, :2] == values[0, 0, 0])


def test_valid_tokens_to_voxel_mask_preserves_bool() -> None:
	valid = np.array([[[True, False]]], dtype=np.bool_)
	actual = valid_tokens_to_voxel_mask(
		valid,
		patch_size_xyz=(2, 2, 2),
		volume_shape_xyz=(2, 2, 3),
	)
	assert actual.dtype == np.bool_
	assert actual[:, :, :2].all()
	assert not actual[:, :, 2].any()


@pytest.mark.parametrize(
	('kwargs', 'error'),
	[
		(
			{'patch_size_xyz': (0, 2, 2), 'volume_shape_xyz': (2, 2, 2)},
			'positive integer triple',
		),
		(
			{'patch_size_xyz': (1, 1, 1), 'volume_shape_xyz': (3, 2, 2)},
			'does not cover',
		),
	],
)
def test_project_token_grid_nearest_rejects_invalid_geometry(
	kwargs: dict[str, tuple[int, int, int]],
	error: str,
) -> None:
	with pytest.raises(ValueError, match=error):
		project_token_grid_nearest(np.ones((2, 2, 2)), **kwargs)


def test_valid_tokens_to_voxel_mask_rejects_non_bool() -> None:
	with pytest.raises(TypeError, match='dtype must be bool'):
		valid_tokens_to_voxel_mask(
			np.ones((1, 1, 1), dtype=np.uint8),
			patch_size_xyz=(1, 1, 1),
			volume_shape_xyz=(1, 1, 1),
		)
