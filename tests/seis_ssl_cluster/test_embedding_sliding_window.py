from __future__ import annotations

import pytest

from seis_ssl_cluster.embedding.sliding_window import (
	SlidingWindow,
	compute_stride_xyz,
	iter_sliding_windows,
	padded_volume_shape_xyz,
	token_grid_shape_xyz,
)


def test_sliding_windows_cover_nondivisible_volume_with_patch_padding() -> None:
	windows = list(
		iter_sliding_windows(
			(10, 11, 12),
			window_size_xyz=(8, 8, 8),
			overlap_xyz=(4, 4, 4),
			patch_size_xyz=(4, 4, 4),
		),
	)

	assert padded_volume_shape_xyz((10, 11, 12), (4, 4, 4)) == (12, 12, 12)
	assert token_grid_shape_xyz((10, 11, 12), (4, 4, 4)) == (3, 3, 3)
	assert windows == [
		SlidingWindow(start_xyz=(x, y, z), size_xyz=(8, 8, 8))
		for x in (0, 4)
		for y in (0, 4)
		for z in (0, 4)
	]


def test_sliding_window_single_window_when_padded_volume_is_smaller() -> None:
	windows = list(
		iter_sliding_windows(
			(5, 6, 7),
			window_size_xyz=(16, 16, 16),
			overlap_xyz=(8, 8, 8),
			patch_size_xyz=(4, 4, 4),
		),
	)

	assert windows == [SlidingWindow(start_xyz=(0, 0, 0), size_xyz=(16, 16, 16))]


def test_sliding_window_rejects_unaligned_overlap() -> None:
	with pytest.raises(ValueError, match='overlap_xyz'):
		list(
			iter_sliding_windows(
				(16, 16, 16),
				window_size_xyz=(8, 8, 8),
				overlap_xyz=(2, 4, 4),
				patch_size_xyz=(4, 4, 4),
			),
		)


def test_stride_must_be_positive() -> None:
	with pytest.raises(ValueError, match='smaller than window_size_xyz'):
		compute_stride_xyz((8, 8, 8), (8, 4, 4))
