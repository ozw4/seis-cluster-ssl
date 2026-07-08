"""Ordered prototype head for stratigraphic token pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional
from torch import nn


@dataclass(frozen=True)
class OrderedPrototypeOutput:
	"""Output tensors from :class:`OrderedPrototypeHead`."""

	logits: torch.Tensor
	projected_features: torch.Tensor


class OrderedPrototypeHead(nn.Module):
	"""Single ordered prototype head over token feature vectors."""

	def __init__(
		self,
		*,
		feature_dim: int,
		num_prototypes: int,
		projection_dim: int | None = None,
		temperature: float = 0.1,
		normalize: bool = True,
	) -> None:
		"""Initialize projection and ordered prototype slots."""
		super().__init__()
		self.feature_dim = _validate_positive_int(feature_dim, 'feature_dim')
		self.num_prototypes = _validate_positive_int(
			num_prototypes,
			'num_prototypes',
		)
		self.temperature = _validate_positive_finite_float(
			temperature,
			'temperature',
		)
		if not isinstance(normalize, bool):
			msg = f'normalize must be a bool; got {normalize!r}'
			raise TypeError(msg)
		self.normalize = normalize

		if projection_dim is None:
			self.projection_dim = self.feature_dim
			self.projection = nn.Identity()
		else:
			self.projection_dim = _validate_positive_int(
				projection_dim,
				'projection_dim',
			)
			self.projection = nn.Linear(self.feature_dim, self.projection_dim)
		self.prototypes = nn.Parameter(
			torch.empty(self.num_prototypes, self.projection_dim),
		)
		nn.init.normal_(self.prototypes, mean=0.0, std=1.0)

	def forward(self, features: torch.Tensor) -> OrderedPrototypeOutput:
		"""Return prototype logits for any tensor ending in ``feature_dim``."""
		_validate_features(features, self.feature_dim)
		projected_features = self.projection(features)
		prototypes = self.prototypes
		if projected_features.device != prototypes.device:
			msg = (
				'features and prototypes must be on the same device; '
				f'got features_device={projected_features.device}, '
				f'prototypes_device={prototypes.device}'
			)
			raise ValueError(msg)
		if projected_features.dtype != prototypes.dtype:
			prototypes = prototypes.to(dtype=projected_features.dtype)

		logit_features = projected_features
		logit_prototypes = prototypes
		if self.normalize:
			logit_features = torch.nn.functional.normalize(
				logit_features,
				p=2,
				dim=-1,
			)
			logit_prototypes = torch.nn.functional.normalize(
				logit_prototypes,
				p=2,
				dim=-1,
			)
		logits = torch.matmul(logit_features, logit_prototypes.transpose(0, 1))
		return OrderedPrototypeOutput(
			logits=logits / self.temperature,
			projected_features=projected_features,
		)


def _validate_positive_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_positive_finite_float(value: float, name: str) -> float:
	if not isinstance(value, (float, int)) or isinstance(value, bool):
		msg = f'{name} must be a float; got {value!r}'
		raise TypeError(msg)
	value = float(value)
	if not math.isfinite(value) or value <= 0.0:
		msg = f'{name} must be positive and finite; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_features(features: torch.Tensor, feature_dim: int) -> None:
	if not isinstance(features, torch.Tensor):
		msg = f'features must be a torch.Tensor; got {type(features)!r}'
		raise TypeError(msg)
	if features.ndim < 1:
		msg = 'features must have at least one dimension ending in feature_dim'
		raise ValueError(msg)
	if features.shape[-1] != feature_dim:
		msg = (
			'features last dimension must equal feature_dim; '
			f'got shape={tuple(features.shape)!r}, feature_dim={feature_dim!r}'
		)
		raise ValueError(msg)
	if not torch.is_floating_point(features):
		msg = f'features must be floating point; got {features.dtype}'
		raise TypeError(msg)


__all__ = ['OrderedPrototypeHead', 'OrderedPrototypeOutput']
