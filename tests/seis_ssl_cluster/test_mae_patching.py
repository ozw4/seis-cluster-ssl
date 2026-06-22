from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.models.mae.patching import (
	compute_num_patches,
	patchify_3d,
	unpatchify_3d,
)


def test_patchify_unpatchify_round_trip_reconstructs_input() -> None:
	patch_size = (2, 3, 4)
	x = torch.arange(2 * 3 * 8 * 12 * 16).reshape(2, 3, 8, 12, 16)
	grid_size = compute_num_patches((8, 12, 16), patch_size)[:3]

	x2 = unpatchify_3d(patchify_3d(x, patch_size), patch_size, grid_size)

	assert torch.equal(x, x2)


def test_patchify_3d_returns_expected_shape_for_small_tensor() -> None:
	x = torch.zeros((2, 3, 8, 12, 16))

	patches = patchify_3d(x, (2, 3, 4))

	assert patches.shape == (2, 64, 3, 24)


def test_patchify_3d_rejects_non_divisible_dimensions() -> None:
	x = torch.zeros((1, 1, 7, 12, 16))

	with pytest.raises(ValueError, match=r'volume_size_xyz=.*patch_size_xyz='):
		patchify_3d(x, (2, 3, 4))


def test_unpatchify_3d_rejects_mismatched_patch_shape() -> None:
	patches = torch.zeros((1, 3, 2, 8))

	with pytest.raises(ValueError, match='expected_num_patches=4'):
		unpatchify_3d(patches, (2, 2, 2), (2, 1, 2))


def test_patchify_3d_preserves_xyz_grid_order() -> None:
	x = torch.empty((1, 1, 4, 4, 4), dtype=torch.int64)
	for x_index in range(4):
		for y_index in range(4):
			for z_index in range(4):
				x[0, 0, x_index, y_index, z_index] = (
					100 * x_index + 10 * y_index + z_index
				)

	patches = patchify_3d(x, (2, 2, 2))

	assert torch.equal(
		patches[0, 0, 0],
		torch.tensor([0, 1, 10, 11, 100, 101, 110, 111]),
	)
	assert torch.equal(
		patches[0, 7, 0],
		torch.tensor([222, 223, 232, 233, 322, 323, 332, 333]),
	)
