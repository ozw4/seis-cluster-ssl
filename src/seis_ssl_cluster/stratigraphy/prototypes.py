"""Ordered prototype head for stratigraphic token pretraining."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional
from torch import nn

MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1 = 'multi_resolution_ordered_prototypes_v1'


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
		return self.forward_validated_features(features)

	def forward_validated_features(
		self,
		features: torch.Tensor,
	) -> OrderedPrototypeOutput:
		"""Return logits after the common feature validation has completed."""
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


@dataclass(frozen=True)
class MultiResolutionOrderedPrototypeOutput:
	"""Outputs from independently parameterized ordered prototype heads."""

	outputs: Mapping[str, OrderedPrototypeOutput]
	head_ks: tuple[int, ...]


class MultiResolutionOrderedPrototypeHeads(nn.Module):
	"""Apply independent ordered prototype heads to shared encoder features."""

	def __init__(
		self,
		*,
		feature_dim: int,
		ks: Sequence[int],
		projection_dim: int | None,
		temperature: float,
		normalize: bool,
	) -> None:
		"""Initialize one independent head for each ordered resolution."""
		super().__init__()
		self.feature_dim = _validate_positive_int(feature_dim, 'feature_dim')
		self.head_ks = validate_multi_resolution_head_ks(ks, prefix='ks')
		self.heads = nn.ModuleDict(
			{
				_head_key(k): OrderedPrototypeHead(
					feature_dim=self.feature_dim,
					num_prototypes=k,
					projection_dim=projection_dim,
					temperature=temperature,
					normalize=normalize,
				)
				for k in self.head_ks
			}
		)

	def forward(self, features: torch.Tensor) -> MultiResolutionOrderedPrototypeOutput:
		"""Return ordered prototype outputs for every configured resolution."""
		_validate_features(features, self.feature_dim)
		outputs = {
			_head_key(k): self.heads[_head_key(k)].forward_validated_features(
				features,
			)
			for k in self.head_ks
		}
		return MultiResolutionOrderedPrototypeOutput(
			outputs=outputs,
			head_ks=self.head_ks,
		)


def expected_normalized_order_coordinate(logits: torch.Tensor) -> torch.Tensor:
	"""Return the softmax expected normalized ordered prototype coordinate."""
	_validate_ordered_logits(logits)
	num_prototypes = logits.shape[-1]
	ranks = torch.linspace(
		0.0,
		1.0,
		steps=num_prototypes,
		device=logits.device,
		dtype=logits.dtype,
	)
	return (torch.softmax(logits, dim=-1) * ranks).sum(dim=-1)


def validate_multi_resolution_head_ks(
	value: object,
	*,
	prefix: str,
) -> tuple[int, ...]:
	"""Validate the canonical ordered-prototype multi-resolution K sequence."""
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{prefix}.ks must be a sequence of integers')
	if len(value) < 2:
		raise ValueError(f'{prefix}.ks must contain at least two heads')
	if any(isinstance(k, bool) or not isinstance(k, int) for k in value):
		raise TypeError(f'{prefix}.ks must contain integers and not bools')
	ks = tuple(value)
	if any(k < 2 for k in ks):
		raise ValueError(f'{prefix}.ks values must be at least 2')
	if tuple(sorted(ks)) != ks:
		raise ValueError(f'{prefix}.ks must be strictly increasing')
	if len(set(ks)) != len(ks):
		raise ValueError(f'{prefix}.ks must not contain duplicates')
	return ks


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


def _head_key(k: int) -> str:
	return f'k{k}'


def _validate_ordered_logits(logits: torch.Tensor) -> None:
	if not isinstance(logits, torch.Tensor):
		msg = f'logits must be a torch.Tensor; got {type(logits)!r}'
		raise TypeError(msg)
	if logits.ndim < 1:
		msg = 'logits must have at least one prototype dimension'
		raise ValueError(msg)
	if logits.shape[-1] < 2:
		msg = 'logits must contain at least two ordered prototypes'
		raise ValueError(msg)
	if not torch.is_floating_point(logits):
		msg = f'logits must be floating point; got {logits.dtype}'
		raise TypeError(msg)
	if not bool(torch.isfinite(logits).all().item()):
		msg = 'logits must be finite'
		raise ValueError(msg)


__all__ = [
	'MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1',
	'MultiResolutionOrderedPrototypeHeads',
	'MultiResolutionOrderedPrototypeOutput',
	'OrderedPrototypeHead',
	'OrderedPrototypeOutput',
	'expected_normalized_order_coordinate',
	'validate_multi_resolution_head_ks',
]
