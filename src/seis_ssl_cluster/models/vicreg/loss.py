"""Variance-invariance-covariance objective for VICReg pretraining."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn


class VICRegLoss(nn.Module):
	"""VICReg invariance, variance, and covariance loss."""

	def __init__(
		self,
		*,
		invariance_weight: float = 25.0,
		variance_weight: float = 25.0,
		covariance_weight: float = 1.0,
		variance_target_std: float = 1.0,
		variance_eps: float = 1.0e-4,
	) -> None:
		"""Store validated objective weights and variance parameters."""
		super().__init__()
		self.invariance_weight = _validate_nonnegative_float(
			invariance_weight,
			'invariance_weight',
		)
		self.variance_weight = _validate_nonnegative_float(
			variance_weight,
			'variance_weight',
		)
		self.covariance_weight = _validate_nonnegative_float(
			covariance_weight,
			'covariance_weight',
		)
		self.variance_target_std = _validate_positive_float(
			variance_target_std,
			'variance_target_std',
		)
		self.variance_eps = _validate_positive_float(
			variance_eps,
			'variance_eps',
		)

	def forward(
		self,
		z_a: torch.Tensor,
		z_b: torch.Tensor,
	) -> dict[str, torch.Tensor]:
		"""Return the VICReg objective terms and projection diagnostics."""
		return vicreg_loss(
			z_a,
			z_b,
			invariance_weight=self.invariance_weight,
			variance_weight=self.variance_weight,
			covariance_weight=self.covariance_weight,
			variance_target_std=self.variance_target_std,
			variance_eps=self.variance_eps,
		)


def vicreg_loss(  # noqa: PLR0913
	z_a: torch.Tensor,
	z_b: torch.Tensor,
	*,
	invariance_weight: float = 25.0,
	variance_weight: float = 25.0,
	covariance_weight: float = 1.0,
	variance_target_std: float = 1.0,
	variance_eps: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
	"""Return the FP32 VICReg objective and projection diagnostics."""
	_validate_projections(z_a, z_b)
	invariance_weight = _validate_nonnegative_float(
		invariance_weight,
		'invariance_weight',
	)
	variance_weight = _validate_nonnegative_float(
		variance_weight,
		'variance_weight',
	)
	covariance_weight = _validate_nonnegative_float(
		covariance_weight,
		'covariance_weight',
	)
	variance_target_std = _validate_positive_float(
		variance_target_std,
		'variance_target_std',
	)
	variance_eps = _validate_positive_float(variance_eps, 'variance_eps')

	with torch.autocast(device_type=z_a.device.type, enabled=False):
		projection_a = z_a.float()
		projection_b = z_b.float()
		invariance_loss = (projection_a - projection_b).square().mean()

		centered_a = projection_a - projection_a.mean(dim=0)
		centered_b = projection_b - projection_b.mean(dim=0)
		std_a = _sample_std(centered_a, variance_eps)
		std_b = _sample_std(centered_b, variance_eps)
		variance_loss_a = torch.relu(variance_target_std - std_a).mean()
		variance_loss_b = torch.relu(variance_target_std - std_b).mean()
		variance_loss = 0.5 * (variance_loss_a + variance_loss_b)

		covariance_a = centered_a.transpose(0, 1) @ centered_a
		covariance_b = centered_b.transpose(0, 1) @ centered_b
		covariance_a = covariance_a / (projection_a.shape[0] - 1)
		covariance_b = covariance_b / (projection_b.shape[0] - 1)
		off_diagonal_a = _off_diagonal(covariance_a)
		off_diagonal_b = _off_diagonal(covariance_b)
		feature_count = projection_a.shape[1]
		covariance_loss_a = off_diagonal_a.square().sum() / feature_count
		covariance_loss_b = off_diagonal_b.square().sum() / feature_count
		covariance_loss = covariance_loss_a + covariance_loss_b

		weighted_invariance = invariance_weight * invariance_loss
		weighted_variance = variance_weight * variance_loss
		weighted_covariance = covariance_weight * covariance_loss
		loss = weighted_invariance + weighted_variance + weighted_covariance

		with torch.no_grad():
			projection_std = torch.cat((std_a, std_b))
			off_diagonal = torch.cat((off_diagonal_a, off_diagonal_b))
			covariance_offdiag_rms = (
				off_diagonal.square().mean().sqrt()
				if off_diagonal.numel()
				else covariance_loss.new_zeros(())
			)
	return {
		'loss': loss,
		'invariance_loss': invariance_loss,
		'variance_loss': variance_loss,
		'covariance_loss': covariance_loss,
		'variance_loss_a': variance_loss_a,
		'variance_loss_b': variance_loss_b,
		'covariance_loss_a': covariance_loss_a,
		'covariance_loss_b': covariance_loss_b,
		'projection_std_mean': projection_std.mean(),
		'projection_std_min': projection_std.min(),
		'covariance_offdiag_rms': covariance_offdiag_rms,
		'weighted_invariance': weighted_invariance,
		'weighted_variance': weighted_variance,
		'weighted_covariance': weighted_covariance,
	}


def _sample_std(centered: torch.Tensor, eps: float) -> torch.Tensor:
	variance = centered.square().sum(dim=0) / (centered.shape[0] - 1)
	return torch.sqrt(variance + eps)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
	mask = ~torch.eye(
		matrix.shape[0],
		dtype=torch.bool,
		device=matrix.device,
	)
	return matrix[mask]


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
		raise ValueError('VICReg loss requires batch size at least 2')
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


__all__ = ['VICRegLoss', 'vicreg_loss']
