"""Gradient-domain losses for amplitude MAE reconstruction targets."""

from __future__ import annotations

from typing import Literal

import torch

from seis_ssl_cluster.models.mae.patching import unpatchify_3d

LossMode = Literal['huber', 'l1', 'mse']


def gradient_loss_xyz(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	spatial_mask: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
) -> torch.Tensor:
	"""Return finite-difference XYZ loss over valid masked reconstruction regions."""
	_validate_prediction_and_target(pred_patches, target)
	_validate_spatial_mask(spatial_mask, pred_patches)
	_validate_local_valid_mask(local_valid_mask, target)
	_validate_same_device(pred_patches, target, spatial_mask, local_valid_mask)

	grid_size_xyz = _grid_size_from_mask(spatial_mask, pred_patches.shape[1])
	pred_volume = unpatchify_3d(pred_patches, patch_size_xyz, grid_size_xyz)
	if pred_volume.shape != target.shape:
		msg = (
			'unpatchified pred_patches must match target shape; '
			f'got {tuple(pred_volume.shape)!r} and {tuple(target.shape)!r}'
		)
		raise ValueError(msg)

	volume_mask = _spatial_mask_to_volume(spatial_mask, patch_size_xyz)
	if tuple(volume_mask.shape) != tuple(local_valid_mask.shape):
		msg = (
			'expanded spatial_mask must match local_valid_mask shape; '
			f'got {tuple(volume_mask.shape)!r} and {tuple(local_valid_mask.shape)!r}'
		)
		raise ValueError(msg)
	volume_mask = volume_mask & local_valid_mask

	numerator = pred_patches.sum() * 0.0
	denominator = pred_patches.new_tensor(0.0)
	for dim in (2, 3, 4):
		pred_grad = pred_volume.diff(dim=dim)
		target_grad = target.diff(dim=dim)
		loss = _elementwise_loss(
			pred_grad,
			target_grad.to(dtype=pred_patches.dtype),
			reconstruction,
			huber_delta,
		)
		selected = _neighbor_mask(volume_mask, dim).unsqueeze(1)
		weight = selected.to(dtype=loss.dtype)
		numerator = numerator + (loss * weight).sum()
		denominator = denominator + weight.sum()

	if bool(denominator.detach().eq(0).item()):
		return pred_patches.sum() * 0.0
	return numerator / denominator


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


def _grid_size_from_mask(
	spatial_mask: torch.Tensor,
	num_patches: int,
) -> tuple[int, int, int]:
	grid_size_xyz = (
		int(spatial_mask.shape[1]),
		int(spatial_mask.shape[2]),
		int(spatial_mask.shape[3]),
	)
	if grid_size_xyz[0] * grid_size_xyz[1] * grid_size_xyz[2] != num_patches:
		msg = (
			'spatial_mask grid must match pred_patches patch count; '
			f'got grid_size_xyz={grid_size_xyz!r}, num_patches={num_patches}'
		)
		raise ValueError(msg)
	return grid_size_xyz


def _spatial_mask_to_volume(
	spatial_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	px_size, py_size, pz_size = patch_size_xyz
	return (
		spatial_mask.repeat_interleave(px_size, dim=1)
		.repeat_interleave(py_size, dim=2)
		.repeat_interleave(pz_size, dim=3)
	)


def _neighbor_mask(mask: torch.Tensor, dim: int) -> torch.Tensor:
	head = [slice(None)] * mask.ndim
	tail = [slice(None)] * mask.ndim
	head[dim - 1] = slice(1, None)
	tail[dim - 1] = slice(None, -1)
	return mask[tuple(head)] & mask[tuple(tail)]


def _validate_same_device(*tensors: torch.Tensor) -> None:
	devices = {tensor.device for tensor in tensors}
	if len(devices) != 1:
		device_names = sorted(map(str, devices))
		msg = f'all tensors must be on the same device; got {device_names!r}'
		raise ValueError(msg)


__all__ = ['gradient_loss_xyz']
