"""Cross-correlation objective for Barlow Twins pretraining."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn


class BarlowTwinsLoss(nn.Module):
	"""Standard Barlow Twins invariance and redundancy-reduction loss."""

	def __init__(
		self,
		*,
		redundancy_weight: float = 0.005,
		normalization_eps: float = 1e-12,
	) -> None:
		"""Store objective weights and feature-normalization epsilon."""
		super().__init__()
		self.redundancy_weight = _validate_nonnegative_float(
			redundancy_weight,
			'redundancy_weight',
		)
		self.normalization_eps = _validate_positive_float(
			normalization_eps,
			'normalization_eps',
		)

	def forward(
		self,
		z_a: torch.Tensor,
		z_b: torch.Tensor,
	) -> dict[str, torch.Tensor]:
		"""Return the objective terms and representation diagnostics."""
		return barlow_twins_loss(
			z_a,
			z_b,
			redundancy_weight=self.redundancy_weight,
			normalization_eps=self.normalization_eps,
		)


def barlow_twins_loss(
	z_a: torch.Tensor,
	z_b: torch.Tensor,
	*,
	redundancy_weight: float = 0.005,
	normalization_eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
	"""Return the Barlow Twins objective and diagnostics for two projections."""
	_validate_projections(z_a, z_b)
	redundancy_weight = _validate_nonnegative_float(
		redundancy_weight,
		'redundancy_weight',
	)
	normalization_eps = _validate_positive_float(
		normalization_eps,
		'normalization_eps',
	)

	with torch.autocast(device_type=z_a.device.type, enabled=False):
		projection_a = z_a.float()
		projection_b = z_b.float()
		normalized_a = _normalize_features(projection_a, normalization_eps)
		normalized_b = _normalize_features(projection_b, normalization_eps)
		cross_correlation = normalized_a.transpose(0, 1) @ normalized_b
		cross_correlation = cross_correlation / z_a.shape[0]

		diagonal = cross_correlation.diagonal()
		on_diag = (1.0 - diagonal).square().sum()
		off_diagonal_mask = ~torch.eye(
			cross_correlation.shape[0],
			dtype=torch.bool,
			device=cross_correlation.device,
		)
		off_diagonal = cross_correlation[off_diagonal_mask]
		off_diag = off_diagonal.square().sum()
		weighted_off_diag = redundancy_weight * off_diag
		loss = on_diag + weighted_off_diag
		with torch.no_grad():
			projection_std = torch.cat(
				(
					projection_a.std(dim=0, unbiased=False),
					projection_b.std(dim=0, unbiased=False),
				)
			)
			projection_norm_mean = torch.cat(
				(projection_a.norm(dim=1), projection_b.norm(dim=1))
			).mean()
			off_diagonal_rms = (
				off_diagonal.square().mean().sqrt()
				if off_diagonal.numel()
				else off_diag.new_zeros(())
			)
	return {
		'loss': loss,
		'on_diag': on_diag,
		'off_diag': off_diag,
		'projection_std_mean': projection_std.mean(),
		'projection_std_min': projection_std.min(),
		'projection_norm_mean': projection_norm_mean,
		'cross_correlation_diag_mean': diagonal.mean().detach(),
		'cross_correlation_offdiag_rms': off_diagonal_rms,
		'weighted_off_diag': weighted_off_diag.detach(),
	}


def _normalize_features(z: torch.Tensor, eps: float) -> torch.Tensor:
	centered = z - z.mean(dim=0)
	variance = centered.square().mean(dim=0)
	return centered * torch.rsqrt(variance + eps)


def _validate_projections(z_a: torch.Tensor, z_b: torch.Tensor) -> None:
	if not isinstance(z_a, torch.Tensor) or not isinstance(z_b, torch.Tensor):
		raise TypeError('z_a and z_b must be tensors')
	if z_a.ndim != 2 or z_b.ndim != 2:
		msg = (
			'z_a and z_b must have shape [batch, feature]; '
			f'got z_a={tuple(z_a.shape)!r}, z_b={tuple(z_b.shape)!r}'
		)
		raise ValueError(msg)
	if z_a.shape != z_b.shape:
		msg = (
			'z_a and z_b must have the same shape; '
			f'got z_a={tuple(z_a.shape)!r}, z_b={tuple(z_b.shape)!r}'
		)
		raise ValueError(msg)
	if z_a.shape[0] < 2:
		raise ValueError('Barlow Twins loss requires batch size at least 2')
	if z_a.shape[1] < 1:
		raise ValueError('projections must contain at least one feature')
	if not z_a.dtype.is_floating_point or not z_b.dtype.is_floating_point:
		raise TypeError('z_a and z_b must have floating-point dtypes')
	if z_a.device != z_b.device:
		msg = (
			'z_a and z_b must be on the same device; '
			f'got z_a={z_a.device}, z_b={z_b.device}'
		)
		raise ValueError(msg)


def _validate_nonnegative_float(value: float, name: str) -> float:
	value = _validate_finite_real(value, name)
	if value < 0.0:
		msg = f'{name} must be nonnegative; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_positive_float(value: float, name: str) -> float:
	value = _validate_finite_real(value, name)
	if value <= 0.0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_finite_real(value: float, name: str) -> float:
	if not isinstance(value, Real) or isinstance(value, bool):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	value = float(value)
	if not math.isfinite(value):
		msg = f'{name} must be finite; got {value!r}'
		raise ValueError(msg)
	return value


__all__ = ['BarlowTwinsLoss', 'barlow_twins_loss']
