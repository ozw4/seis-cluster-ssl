from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.losses import (
	gradient_loss_xyz,
	mae_pretraining_loss,
	masked_patch_reconstruction_loss,
)
from seis_ssl_cluster.models.mae.patching import patchify_3d

PATCH_SIZE_XYZ = (2, 2, 2)


def _spatial_mask() -> torch.Tensor:
	mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	mask[0, 0, 0, 0] = True
	return mask


def test_invalid_voxels_do_not_change_reconstruction_loss() -> None:
	target = torch.zeros((1, 1, 4, 4, 4))
	local_valid_mask = torch.ones((1, 4, 4, 4), dtype=torch.bool)
	local_valid_mask[:, 0, 0, 0] = False
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ) + 1.0
	invalid_error_pred = pred_patches.clone()
	valid_voxels = patchify_3d(local_valid_mask.unsqueeze(1), PATCH_SIZE_XYZ)
	invalid_error_pred[~valid_voxels] = 100.0

	loss = masked_patch_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		local_valid_mask=local_valid_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)
	loss_with_invalid_error = masked_patch_reconstruction_loss(
		pred_patches=invalid_error_pred,
		target=target,
		spatial_mask=_spatial_mask(),
		local_valid_mask=local_valid_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)

	assert loss_with_invalid_error == loss


def test_valid_masked_voxel_changes_reconstruction_loss() -> None:
	target = torch.zeros((1, 1, 4, 4, 4))
	local_valid_mask = torch.ones((1, 4, 4, 4), dtype=torch.bool)
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ)
	changed_pred = pred_patches.clone()
	changed_pred[:, 0, 0, 0] = 2.0

	base_loss = masked_patch_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		local_valid_mask=local_valid_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)
	changed_loss = masked_patch_reconstruction_loss(
		pred_patches=changed_pred,
		target=target,
		spatial_mask=_spatial_mask(),
		local_valid_mask=local_valid_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)

	assert base_loss == torch.tensor(0.0)
	assert changed_loss > base_loss


def test_visible_tokens_do_not_contribute_to_reconstruction_loss() -> None:
	target = torch.zeros((1, 1, 4, 4, 4))
	local_valid_mask = torch.ones((1, 4, 4, 4), dtype=torch.bool)
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ)
	pred_patches[:, 1] = 10.0

	loss = masked_patch_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		local_valid_mask=local_valid_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)

	assert loss == torch.tensor(0.0)


def test_valid_reconstruction_voxels_are_reported() -> None:
	target = torch.zeros((1, 1, 4, 4, 4))
	local_valid_mask = torch.ones((1, 4, 4, 4), dtype=torch.bool)
	local_valid_mask[:, 0, 0, 0] = False
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ)

	losses = mae_pretraining_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		local_valid_mask=local_valid_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		gradient_weight=0.0,
	)

	assert losses['valid_reconstruction_voxels'] == torch.tensor(7)
	assert losses['loss'] == torch.tensor(0.0)


def test_reconstruction_loss_raises_when_no_valid_masked_voxels() -> None:
	target = torch.zeros((1, 1, 4, 4, 4))
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ)
	local_valid_mask = torch.zeros((1, 4, 4, 4), dtype=torch.bool)

	with pytest.raises(ValueError, match='no valid masked voxels'):
		masked_patch_reconstruction_loss(
			pred_patches=pred_patches,
			target=target,
			spatial_mask=_spatial_mask(),
			local_valid_mask=local_valid_mask,
			patch_size_xyz=PATCH_SIZE_XYZ,
		)


def test_gradient_loss_excludes_pairs_crossing_invalid_voxels() -> None:
	target = torch.zeros((1, 1, 2, 2, 2))
	pred_patches = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
	local_valid_mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	local_valid_mask[:, 0, 0, 0] = True

	loss = gradient_loss_xyz(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=torch.ones((1, 1, 1, 1), dtype=torch.bool),
		local_valid_mask=local_valid_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)

	assert loss == torch.tensor(0.0)


@pytest.mark.parametrize('reconstruction', ['huber', 'l1', 'mse'])
def test_reconstruction_modes_work(reconstruction: str) -> None:
	target = torch.zeros((1, 1, 2, 2, 2))
	pred_patches = torch.ones((1, 1, 1, 8))
	local_valid_mask = torch.ones((1, 2, 2, 2), dtype=torch.bool)

	loss = masked_patch_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=torch.ones((1, 1, 1, 1), dtype=torch.bool),
		local_valid_mask=local_valid_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction=reconstruction,
	)

	assert loss > 0
