"""Target-only normalization helpers for MAE reconstruction losses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

TargetNormalizationMode = Literal['none', 'patch_zscore']


@dataclass(frozen=True)
class PatchTargetNormalizationResult:
	"""Patch-wise target normalization tensors and diagnostics."""

	normalized_target: torch.Tensor
	patch_mean: torch.Tensor
	patch_std: torch.Tensor
	patch_std_eff: torch.Tensor
	valid_count: torch.Tensor
	low_std_mask: torch.Tensor


def normalize_target_patches(
	target_patches: torch.Tensor,
	valid_patch_voxels: torch.Tensor,
	*,
	mode: TargetNormalizationMode,
	eps: float | None = None,
	min_std: float | None = None,
) -> PatchTargetNormalizationResult:
	"""Normalize target patches for loss computation without changing inputs."""
	_validate_inputs(target_patches, valid_patch_voxels)
	valid_f = valid_patch_voxels.to(dtype=target_patches.dtype)
	valid_count = valid_f.sum(dim=-1, keepdim=True)
	nonempty = valid_count > 0

	if mode == 'none':
		zeros = torch.zeros_like(valid_count, dtype=target_patches.dtype)
		ones = torch.ones_like(valid_count, dtype=target_patches.dtype)
		return PatchTargetNormalizationResult(
			normalized_target=target_patches,
			patch_mean=zeros,
			patch_std=ones,
			patch_std_eff=ones,
			valid_count=valid_count,
			low_std_mask=torch.zeros_like(valid_count, dtype=torch.bool),
		)
	if mode != 'patch_zscore':
		msg = f'target normalization mode must be "none" or "patch_zscore"; got {mode!r}'
		raise ValueError(msg)

	eps_value = _positive_finite(eps, 'eps')
	min_std_value = _positive_finite(min_std, 'min_std')
	safe_count = valid_count.clamp_min(1.0)
	patch_mean = (target_patches * valid_f).sum(dim=-1, keepdim=True) / safe_count
	centered = (target_patches - patch_mean) * valid_f
	patch_var = centered.square().sum(dim=-1, keepdim=True) / safe_count
	patch_std = torch.sqrt(patch_var + eps_value)
	patch_std_eff = patch_std.clamp_min(min_std_value)
	ones = torch.ones_like(patch_std_eff)
	patch_std = torch.where(nonempty, patch_std, ones)
	patch_std_eff = torch.where(nonempty, patch_std_eff, ones)
	patch_mean = torch.where(nonempty, patch_mean, torch.zeros_like(patch_mean))
	low_std_mask = (patch_std < min_std_value) & nonempty
	normalized = (target_patches - patch_mean) / patch_std_eff
	normalized = torch.where(
		valid_patch_voxels,
		normalized,
		torch.zeros_like(normalized),
	)
	return PatchTargetNormalizationResult(
		normalized_target=normalized,
		patch_mean=patch_mean,
		patch_std=patch_std,
		patch_std_eff=patch_std_eff,
		valid_count=valid_count,
		low_std_mask=low_std_mask,
	)


def denormalize_predicted_patches(
	pred_patches: torch.Tensor,
	*,
	patch_mean: torch.Tensor,
	patch_std_eff: torch.Tensor,
) -> torch.Tensor:
	"""Map patch-z-score predictions back with target-owned patch statistics."""
	return pred_patches * patch_std_eff + patch_mean


def _validate_inputs(
	target_patches: torch.Tensor,
	valid_patch_voxels: torch.Tensor,
) -> None:
	if target_patches.ndim != 4:
		msg = (
			'target_patches must be [B, N, 1, P]; '
			f'got {tuple(target_patches.shape)!r}'
		)
		raise ValueError(msg)
	if valid_patch_voxels.shape != target_patches.shape:
		msg = (
			'valid_patch_voxels shape must match target_patches; '
			f'got {tuple(valid_patch_voxels.shape)!r} and '
			f'{tuple(target_patches.shape)!r}'
		)
		raise ValueError(msg)
	if valid_patch_voxels.dtype != torch.bool:
		msg = f'valid_patch_voxels must be bool; got {valid_patch_voxels.dtype!r}'
		raise TypeError(msg)
	if valid_patch_voxels.device != target_patches.device:
		msg = 'target_patches and valid_patch_voxels must be on the same device'
		raise ValueError(msg)


def _positive_finite(value: float | None, name: str) -> float:
	if value is None or not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{name} must be a finite positive number'
		raise ValueError(msg)
	resolved = float(value)
	if not math.isfinite(resolved) or resolved <= 0.0:
		msg = f'{name} must be a finite positive number; got {value!r}'
		raise ValueError(msg)
	return resolved


__all__ = [
	'PatchTargetNormalizationResult',
	'TargetNormalizationMode',
	'denormalize_predicted_patches',
	'normalize_target_patches',
]
