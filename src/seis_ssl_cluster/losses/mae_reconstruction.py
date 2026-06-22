"""Voxel-valid-mask-aware MAE reconstruction losses."""

from __future__ import annotations

from typing import Literal

import torch

from seis_ssl_cluster.losses.gradient import gradient_loss_xyz
from seis_ssl_cluster.models.mae.patching import patchify_3d

LossMode = Literal['huber', 'l1', 'mse']


def masked_patch_reconstruction_loss(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	spatial_mask: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
) -> torch.Tensor:
	"""Return reconstruction loss over valid voxels in masked spatial patches."""
	loss, _valid_voxels = _masked_reconstruction_loss_and_count(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=spatial_mask,
		local_valid_mask=local_valid_mask,
		patch_size_xyz=patch_size_xyz,
		reconstruction=reconstruction,
		huber_delta=huber_delta,
	)
	return loss


def mae_pretraining_loss(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	spatial_mask: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
	gradient_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
	"""Return total amplitude-only MAE loss and component scalars."""
	if gradient_weight < 0:
		msg = f'gradient_weight must be nonnegative; got {gradient_weight!r}'
		raise ValueError(msg)

	loss_reconstruction, valid_reconstruction_voxels = (
		_masked_reconstruction_loss_and_count(
			pred_patches=pred_patches,
			target=target,
			spatial_mask=spatial_mask,
			local_valid_mask=local_valid_mask,
			patch_size_xyz=patch_size_xyz,
			reconstruction=reconstruction,
			huber_delta=huber_delta,
		)
	)
	loss_gradient = gradient_loss_xyz(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=spatial_mask,
		local_valid_mask=local_valid_mask,
		patch_size_xyz=patch_size_xyz,
		reconstruction=reconstruction,
		huber_delta=huber_delta,
	)
	loss = loss_reconstruction + gradient_weight * loss_gradient
	return {
		'loss': loss,
		'loss_reconstruction': loss_reconstruction,
		'loss_gradient': loss_gradient,
		'valid_reconstruction_voxels': valid_reconstruction_voxels,
	}


def _masked_reconstruction_loss_and_count(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	spatial_mask: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode,
	huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
	target_patches = _aligned_target_patches(pred_patches, target, patch_size_xyz)
	local_valid_patch_voxels = _local_valid_patch_voxels(
		local_valid_mask=local_valid_mask,
		pred_patches=pred_patches,
		target=target,
		patch_size_xyz=patch_size_xyz,
	)
	_validate_spatial_mask(spatial_mask, pred_patches)
	_validate_same_device(pred_patches, target, spatial_mask, local_valid_mask)

	selection = (
		spatial_mask.reshape(pred_patches.shape[0], pred_patches.shape[1])
		.unsqueeze(-1)
		.unsqueeze(-1)
		& local_valid_patch_voxels
	)
	valid_voxels = selection.sum()
	if bool(valid_voxels.detach().eq(0).item()):
		msg = (
			'no valid masked voxels for reconstruction loss; check spatial_mask '
			'and local_valid_mask'
		)
		raise ValueError(msg)

	loss = _elementwise_loss(
		pred_patches,
		target_patches.to(dtype=pred_patches.dtype),
		reconstruction,
		huber_delta,
	)
	return (
		loss.masked_select(selection).mean(),
		valid_voxels,
	)


def _aligned_target_patches(
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	_validate_prediction_and_target(pred_patches, target)
	target_patches = patchify_3d(target, patch_size_xyz)
	if target_patches.shape != pred_patches.shape:
		msg = (
			'patchified target must match pred_patches shape; '
			f'got {tuple(target_patches.shape)!r} and {tuple(pred_patches.shape)!r}'
		)
		raise ValueError(msg)
	return target_patches


def _local_valid_patch_voxels(
	*,
	local_valid_mask: torch.Tensor,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	_validate_local_valid_mask(local_valid_mask, target)
	patch_voxels = patchify_3d(local_valid_mask.unsqueeze(1), patch_size_xyz)
	expected_shape = (
		pred_patches.shape[0],
		pred_patches.shape[1],
		1,
		pred_patches.shape[3],
	)
	if tuple(patch_voxels.shape) != expected_shape:
		msg = (
			'patchified local_valid_mask must match pred_patches patch layout; '
			f'got {tuple(patch_voxels.shape)!r} and {expected_shape!r}'
		)
		raise ValueError(msg)
	return patch_voxels


def _elementwise_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	reconstruction: LossMode,
	huber_delta: float,
) -> torch.Tensor:
	if reconstruction == 'mse':
		return (pred - target).square()
	if reconstruction == 'l1':
		return (pred - target).abs()
	if reconstruction == 'huber':
		if huber_delta <= 0:
			msg = f'huber_delta must be positive; got {huber_delta!r}'
			raise ValueError(msg)
		return torch.nn.functional.huber_loss(
			pred,
			target,
			reduction='none',
			delta=huber_delta,
		)
	msg = f'reconstruction must be "huber", "l1", or "mse"; got {reconstruction!r}'
	raise ValueError(msg)


def _validate_prediction_and_target(
	pred_patches: torch.Tensor,
	target: torch.Tensor,
) -> None:
	if pred_patches.ndim != 4:
		msg = (
			'pred_patches must be a 4D tensor with shape '
			f'[B, N, 1, patch_volume]; got {tuple(pred_patches.shape)!r}'
		)
		raise ValueError(msg)
	if target.ndim != 5:
		msg = (
			'target must be a 5D tensor with shape [B, 1, X, Y, Z]; '
			f'got {tuple(target.shape)!r}'
		)
		raise ValueError(msg)
	if pred_patches.shape[2] != 1 or target.shape[1] != 1:
		msg = (
			'amplitude MAE losses require one channel; got '
			f'pred_channels={pred_patches.shape[2]}, target_channels={target.shape[1]}'
		)
		raise ValueError(msg)
	if pred_patches.shape[0] != target.shape[0]:
		msg = (
			'pred_patches and target batch dimensions must match; '
			f'got {pred_patches.shape[0]} and {target.shape[0]}'
		)
		raise ValueError(msg)


def _validate_local_valid_mask(
	local_valid_mask: torch.Tensor,
	target: torch.Tensor,
) -> None:
	if local_valid_mask.dtype != torch.bool:
		msg = (
			'local_valid_mask must have dtype torch.bool; '
			f'got {local_valid_mask.dtype!r}'
		)
		raise TypeError(msg)
	expected_shape = (
		target.shape[0],
		target.shape[2],
		target.shape[3],
		target.shape[4],
	)
	if tuple(local_valid_mask.shape) != expected_shape:
		msg = (
			f'local_valid_mask shape must be {expected_shape!r}; '
			f'got {tuple(local_valid_mask.shape)!r}'
		)
		raise ValueError(msg)


def _validate_spatial_mask(
	spatial_mask: torch.Tensor,
	pred_patches: torch.Tensor,
) -> None:
	if spatial_mask.dtype != torch.bool:
		msg = f'spatial_mask must have dtype torch.bool; got {spatial_mask.dtype!r}'
		raise TypeError(msg)
	if spatial_mask.ndim != 4:
		msg = (
			'spatial_mask must be a 4D tensor with shape [B, TX, TY, TZ]; '
			f'got {tuple(spatial_mask.shape)!r}'
		)
		raise ValueError(msg)
	if spatial_mask.shape[0] != pred_patches.shape[0]:
		msg = (
			'spatial_mask batch dimension must match pred_patches; '
			f'got {spatial_mask.shape[0]} and {pred_patches.shape[0]}'
		)
		raise ValueError(msg)
	num_spatial_patches = spatial_mask.reshape(spatial_mask.shape[0], -1).shape[1]
	if num_spatial_patches != pred_patches.shape[1]:
		msg = (
			'spatial_mask grid must match pred_patches patch count; '
			f'got {tuple(spatial_mask.shape[1:])!r} and {pred_patches.shape[1]}'
		)
		raise ValueError(msg)


def _validate_same_device(*tensors: torch.Tensor) -> None:
	devices = {tensor.device for tensor in tensors}
	if len(devices) != 1:
		device_names = sorted(map(str, devices))
		msg = f'all tensors must be on the same device; got {device_names!r}'
		raise ValueError(msg)


__all__ = [
	'mae_pretraining_loss',
	'masked_patch_reconstruction_loss',
]
