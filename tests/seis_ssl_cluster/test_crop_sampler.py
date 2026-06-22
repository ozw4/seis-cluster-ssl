from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.data import (
	expand_request_with_margin,
	required_zero_mask_margin_xyz,
	rng_for_sample,
	sample_random_local_crop,
	select_round_robin_index,
)
from seis_ssl_cluster.data.schema import CropRequest


def test_sample_random_local_crop_is_in_bounds_and_deterministic() -> None:
	first = sample_random_local_crop(
		(16, 17, 18),
		(8, 9, 10),
		rng_for_sample(42, 0, 3),
		survey_id='survey',
	)
	second = sample_random_local_crop(
		(16, 17, 18),
		(8, 9, 10),
		rng_for_sample(42, 0, 3),
		survey_id='survey',
	)

	assert first == second
	assert first.survey_id == 'survey'
	assert first.size_xyz == (8, 9, 10)
	assert all(
		0 <= start_axis <= shape_axis - size_axis
		for start_axis, shape_axis, size_axis in zip(
			first.start_xyz,
			(16, 17, 18),
			first.size_xyz,
			strict=True,
		)
	)


def test_sample_random_local_crop_rejects_crop_larger_than_volume() -> None:
	with pytest.raises(ValueError, match='does not fit'):
		sample_random_local_crop(
			(4, 5, 6),
			(4, 6, 6),
			np.random.default_rng(0),
		)


def test_round_robin_selection_uses_manifest_order() -> None:
	assert [select_round_robin_index(3, index) for index in range(7)] == [
		0,
		1,
		2,
		0,
		1,
		2,
		0,
	]


def test_expand_request_with_margin_returns_payload_slices() -> None:
	request = CropRequest(
		survey_id='survey',
		start_xyz=(10, 20, 30),
		size_xyz=(4, 5, 6),
	)

	expanded, payload_slices = expand_request_with_margin(request, (1, 2, 3))

	assert expanded == CropRequest(
		survey_id='survey',
		start_xyz=(9, 18, 27),
		size_xyz=(6, 9, 12),
	)
	assert payload_slices == (slice(1, 5), slice(2, 7), slice(3, 9))


def test_required_zero_mask_margin_uses_xy_and_z_influence_radii() -> None:
	assert required_zero_mask_margin_xyz(
		z_sample_influence_radius=16,
		xy_trace_influence_radius=1,
	) == (1, 1, 16)
