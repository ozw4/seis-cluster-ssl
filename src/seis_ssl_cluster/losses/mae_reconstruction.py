"""Voxel-valid-mask-aware MAE reconstruction losses."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import torch

from seis_ssl_cluster.losses.gradient import gradient_loss_xyz
from seis_ssl_cluster.losses.target_normalization import (
	PatchTargetNormalizationResult,
	TargetNormalizationMode,
	normalize_target_patches,
)
from seis_ssl_cluster.models.mae.patching import compute_num_patches
from seis_ssl_cluster.runtime_checks import RuntimeCheckMode, RuntimeChecks

LossMode = Literal['huber', 'l1', 'mse']


def masked_patch_reconstruction_loss(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target_patches: torch.Tensor,
	spatial_mask: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
	target_normalization_mode: TargetNormalizationMode = 'none',
	target_normalization_eps: float | None = None,
	target_normalization_min_std: float | None = None,
	runtime_check_mode: RuntimeCheckMode = 'strict',
	runtime_checks: RuntimeChecks | None = None,
) -> torch.Tensor:
	"""Return reconstruction loss over valid voxels in masked spatial patches."""
	runtime_checks = runtime_checks or RuntimeChecks(runtime_check_mode)
	loss, _valid_voxels, _normalization_result, _patch_selection = (
		_masked_reconstruction_loss_and_count(
			pred_patches=pred_patches,
			target_patches=target_patches,
			spatial_mask=spatial_mask,
			local_valid_mask=local_valid_mask,
			patch_size_xyz=patch_size_xyz,
			reconstruction=reconstruction,
			huber_delta=huber_delta,
			target_normalization_mode=target_normalization_mode,
			target_normalization_eps=target_normalization_eps,
			target_normalization_min_std=target_normalization_min_std,
			runtime_checks=runtime_checks,
		)
	)
	return loss


def reconstruction_target_patches_for_loss(  # noqa: PLR0913
	*,
	target_patches: torch.Tensor,
	pred_patches: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	target_normalization_mode: TargetNormalizationMode = 'none',
	target_normalization_eps: float | None = None,
	target_normalization_min_std: float | None = None,
) -> PatchTargetNormalizationResult:
	"""Return target patches as used by reconstruction loss."""
	target_patches = _aligned_target_patches(pred_patches, target_patches)
	local_valid_patch_voxels = _local_valid_patch_voxels(
		local_valid_mask=local_valid_mask,
		pred_patches=pred_patches,
		spatial_mask=None,
		patch_size_xyz=patch_size_xyz,
	)
	return normalize_target_patches(
		target_patches.detach().to(dtype=pred_patches.dtype),
		local_valid_patch_voxels,
		mode=target_normalization_mode,
		eps=target_normalization_eps,
		min_std=target_normalization_min_std,
	)


def mae_pretraining_loss(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target_patches: torch.Tensor,
	x: torch.Tensor,
	spatial_mask: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
	gradient_weight: float = 0.05,
	visible_reconstruction_weight: float = 0.0,
	target_normalization_mode: TargetNormalizationMode = 'none',
	target_normalization_eps: float | None = None,
	target_normalization_min_std: float | None = None,
	runtime_check_mode: RuntimeCheckMode = 'strict',
	runtime_checks: RuntimeChecks | None = None,
) -> dict[str, torch.Tensor]:
	"""Return total amplitude-only MAE loss and component scalars."""
	runtime_checks = runtime_checks or RuntimeChecks(runtime_check_mode)
	if gradient_weight < 0:
		msg = f'gradient_weight must be nonnegative; got {gradient_weight!r}'
		raise ValueError(msg)
	_validate_visible_reconstruction_weight(visible_reconstruction_weight)
	_validate_target_normalization_mode(target_normalization_mode)
	_validate_target_normalization_gradient_contract(
		target_normalization_mode,
		gradient_weight,
	)

	(
		loss_reconstruction_masked,
		loss_reconstruction_visible,
		valid_reconstruction_voxels,
		valid_visible_reconstruction_voxels,
		normalization_result,
		patch_selection,
	) = _reconstruction_loss_components(
		pred_patches=pred_patches,
		target_patches=target_patches,
		spatial_mask=spatial_mask,
		local_valid_mask=local_valid_mask,
		patch_size_xyz=patch_size_xyz,
		reconstruction=reconstruction,
		huber_delta=huber_delta,
		target_normalization_mode=target_normalization_mode,
		target_normalization_eps=target_normalization_eps,
		target_normalization_min_std=target_normalization_min_std,
		runtime_checks=runtime_checks,
	)
	visible_weight = loss_reconstruction_masked.new_tensor(
		float(visible_reconstruction_weight),
	)
	loss_reconstruction = (
		loss_reconstruction_masked + visible_weight * loss_reconstruction_visible
	)
	if gradient_weight == 0.0:
		loss_gradient = loss_reconstruction.detach().new_tensor(0.0)
	else:
		loss_gradient = gradient_loss_xyz(
			pred_patches=pred_patches,
			target=x.detach(),
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
		'loss_reconstruction_masked': loss_reconstruction_masked,
		'loss_reconstruction_visible': loss_reconstruction_visible,
		'visible_reconstruction_weight': visible_weight,
		'valid_reconstruction_voxels': valid_reconstruction_voxels,
		'valid_visible_reconstruction_voxels': valid_visible_reconstruction_voxels,
		**_target_normalization_metrics(normalization_result, patch_selection),
	}


def _reconstruction_loss_components(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target_patches: torch.Tensor,
	spatial_mask: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode,
	huber_delta: float,
	target_normalization_mode: TargetNormalizationMode,
	target_normalization_eps: float | None,
	target_normalization_min_std: float | None,
	runtime_checks: RuntimeChecks,
) -> tuple[
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	PatchTargetNormalizationResult,
	torch.Tensor,
]:
	target_patches = _aligned_target_patches(pred_patches, target_patches)
	local_valid_patch_voxels = _local_valid_patch_voxels(
		local_valid_mask=local_valid_mask,
		pred_patches=pred_patches,
		spatial_mask=spatial_mask,
		patch_size_xyz=patch_size_xyz,
	)
	_validate_spatial_mask(spatial_mask, pred_patches)
	_validate_same_device(
		pred_patches,
		target_patches,
		spatial_mask,
		local_valid_mask,
	)

	spatial_patch_mask = (
		spatial_mask.reshape(pred_patches.shape[0], pred_patches.shape[1])
		.unsqueeze(-1)
		.unsqueeze(-1)
	)
	masked_selection = spatial_patch_mask & local_valid_patch_voxels
	visible_selection = ~spatial_patch_mask & local_valid_patch_voxels
	valid_reconstruction_voxels = masked_selection.sum()
	runtime_checks.check(
		'valid_masked_reconstruction_voxels',
		lambda: valid_reconstruction_voxels.ne(0),
		error=ValueError(
			'no valid masked voxels for reconstruction loss; check spatial_mask '
			'and local_valid_mask',
		),
	)

	normalization_result = normalize_target_patches(
		target_patches.detach().to(dtype=pred_patches.dtype),
		local_valid_patch_voxels,
		mode=target_normalization_mode,
		eps=target_normalization_eps,
		min_std=target_normalization_min_std,
	)
	loss = _elementwise_loss(
		pred_patches,
		normalization_result.normalized_target,
		reconstruction,
		huber_delta,
	)
	loss_reconstruction_masked = (
		loss.masked_select(masked_selection).sum()
		/ valid_reconstruction_voxels.clamp_min(1)
	)
	valid_visible_reconstruction_voxels = visible_selection.sum()
	loss_reconstruction_visible = _mean_or_zero(
		loss,
		visible_selection,
		valid_visible_reconstruction_voxels,
	)
	patch_selection = spatial_patch_mask & normalization_result.valid_count.gt(0)
	return (
		loss_reconstruction_masked,
		loss_reconstruction_visible,
		valid_reconstruction_voxels,
		valid_visible_reconstruction_voxels,
		normalization_result,
		patch_selection,
	)


def _masked_reconstruction_loss_and_count(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target_patches: torch.Tensor,
	spatial_mask: torch.Tensor,
	local_valid_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode,
	huber_delta: float,
	target_normalization_mode: TargetNormalizationMode,
	target_normalization_eps: float | None,
	target_normalization_min_std: float | None,
	runtime_checks: RuntimeChecks,
) -> tuple[
	torch.Tensor,
	torch.Tensor,
	PatchTargetNormalizationResult,
	torch.Tensor,
]:
	(
		loss_reconstruction_masked,
		_loss_reconstruction_visible,
		valid_voxels,
		_valid_visible_voxels,
		normalization_result,
		patch_selection,
	) = _reconstruction_loss_components(
		pred_patches=pred_patches,
		target_patches=target_patches,
		spatial_mask=spatial_mask,
		local_valid_mask=local_valid_mask,
		patch_size_xyz=patch_size_xyz,
		reconstruction=reconstruction,
		huber_delta=huber_delta,
		target_normalization_mode=target_normalization_mode,
		target_normalization_eps=target_normalization_eps,
		target_normalization_min_std=target_normalization_min_std,
		runtime_checks=runtime_checks,
	)
	return (
		loss_reconstruction_masked,
		valid_voxels,
		normalization_result,
		patch_selection,
	)


def _mean_or_zero(
	loss: torch.Tensor,
	selection: torch.Tensor,
	valid_voxels: torch.Tensor,
) -> torch.Tensor:
	return loss.masked_select(selection).sum() / valid_voxels.clamp_min(1)


def _target_normalization_metrics(
	result: PatchTargetNormalizationResult,
	patch_selection: torch.Tensor,
) -> dict[str, torch.Tensor]:
	selected_std = result.patch_std.masked_select(patch_selection)
	if selected_std.numel() == 0:
		zero = result.patch_std.new_tensor(0.0)
		return {
			'target_patch_std_mean': zero,
			'target_patch_std_min': zero,
			'target_patch_std_max': zero,
			'target_patch_low_std_fraction': zero,
		}
	low_std = result.low_std_mask.masked_select(patch_selection)
	return {
		'target_patch_std_mean': selected_std.mean(),
		'target_patch_std_min': selected_std.min(),
		'target_patch_std_max': selected_std.max(),
		'target_patch_low_std_fraction': low_std.to(dtype=selected_std.dtype).mean(),
	}


def _validate_target_normalization_gradient_contract(
	mode: TargetNormalizationMode,
	gradient_weight: float,
) -> None:
	if mode == 'patch_zscore' and gradient_weight != 0.0:
		msg = (
			'loss.gradient_weight must be 0.0 when '
			"loss.target_normalization.mode is 'patch_zscore'; "
			'the current gradient loss operates in survey-normalized amplitude space'
		)
		raise ValueError(msg)


def _validate_visible_reconstruction_weight(value: object) -> None:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = (
			'visible_reconstruction_weight must be a finite nonnegative real '
			f'number; got {value!r}'
		)
		raise TypeError(msg)
	weight = float(value)
	if not math.isfinite(weight) or weight < 0.0:
		msg = (
			'visible_reconstruction_weight must be a finite nonnegative real '
			f'number; got {value!r}'
		)
		raise ValueError(msg)


def _validate_target_normalization_mode(mode: TargetNormalizationMode) -> None:
	if mode not in ('none', 'patch_zscore'):
		msg = (
			'target_normalization_mode must be "none" or "patch_zscore"; '
			f'got {mode!r}'
		)
		raise ValueError(msg)


def _aligned_target_patches(
	pred_patches: torch.Tensor,
	target_patches: torch.Tensor,
) -> torch.Tensor:
	_validate_prediction_and_target_patches(pred_patches, target_patches)
	if target_patches.shape != pred_patches.shape:
		msg = (
			'target_patches must match pred_patches shape; '
			f'got {tuple(target_patches.shape)!r} and {tuple(pred_patches.shape)!r}'
		)
		raise ValueError(msg)
	return target_patches


def _local_valid_patch_voxels(
	*,
	local_valid_mask: torch.Tensor,
	pred_patches: torch.Tensor,
	spatial_mask: torch.Tensor | None,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	patch_voxels = _reshape_local_valid_mask_to_patches(
		local_valid_mask,
		pred_patches,
		spatial_mask,
		patch_size_xyz,
	)
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


def _reshape_local_valid_mask_to_patches(
	local_valid_mask: torch.Tensor,
	pred_patches: torch.Tensor,
	spatial_mask: torch.Tensor | None,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	if local_valid_mask.dtype != torch.bool:
		msg = (
			'local_valid_mask must have dtype torch.bool; '
			f'got {local_valid_mask.dtype!r}'
		)
		raise TypeError(msg)
	if local_valid_mask.ndim != 4:
		msg = (
			'local_valid_mask must have shape [B, X, Y, Z]; '
			f'got {tuple(local_valid_mask.shape)!r}'
		)
		raise ValueError(msg)
	px_size, py_size, pz_size = patch_size_xyz
	batch_size, x_size, y_size, z_size = local_valid_mask.shape
	token_grid_shape = compute_num_patches(
		(int(x_size), int(y_size), int(z_size)),
		patch_size_xyz,
	)[:3]
	if spatial_mask is not None and tuple(spatial_mask.shape[1:]) != token_grid_shape:
		msg = (
			'local_valid_mask patch grid must match spatial_mask; '
			f'got {token_grid_shape!r} and {tuple(spatial_mask.shape[1:])!r}'
		)
		raise ValueError(msg)
	if batch_size != pred_patches.shape[0]:
		msg = (
			'local_valid_mask batch dimension must match pred_patches; '
			f'got {batch_size} and {pred_patches.shape[0]}'
		)
		raise ValueError(msg)
	tx_size, ty_size, tz_size = token_grid_shape
	return (
		local_valid_mask.reshape(
			batch_size,
			tx_size,
			px_size,
			ty_size,
			py_size,
			tz_size,
			pz_size,
		)
		.permute(0, 1, 3, 5, 2, 4, 6)
		.contiguous()
		.reshape(batch_size, tx_size * ty_size * tz_size, 1, -1)
	)


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


def _validate_prediction_and_target_patches(
	pred_patches: torch.Tensor,
	target_patches: torch.Tensor,
) -> None:
	if pred_patches.ndim != 4:
		msg = (
			'pred_patches must be a 4D tensor with shape '
			f'[B, N, 1, patch_volume]; got {tuple(pred_patches.shape)!r}'
		)
		raise ValueError(msg)
	if target_patches.ndim != 4:
		msg = (
			'target_patches must be a 4D tensor with shape '
			f'[B, N, 1, patch_volume]; got {tuple(target_patches.shape)!r}'
		)
		raise ValueError(msg)
	if pred_patches.shape[2] != 1 or target_patches.shape[2] != 1:
		msg = (
			'amplitude MAE losses require one channel; got '
			f'pred_channels={pred_patches.shape[2]}, '
			f'target_channels={target_patches.shape[2]}'
		)
		raise ValueError(msg)
	if pred_patches.shape[0] != target_patches.shape[0]:
		msg = (
			'pred_patches and target_patches batch dimensions must match; '
			f'got {pred_patches.shape[0]} and {target_patches.shape[0]}'
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
	'reconstruction_target_patches_for_loss',
]
