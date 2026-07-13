"""Lightweight voxel decoder for precomputed 3D token embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn.functional import interpolate


class _SingletonSafeGroupNorm(nn.GroupNorm):
	"""Apply GroupNorm when a group contains only one scalar value."""

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return the mathematically defined affine output for singleton groups."""
		values_per_group = (x.shape[1] // self.num_groups) * _product(x.shape[2:])
		if values_per_group != 1:
			return super().forward(x)

		output = x * 0.0
		if self.affine:
			broadcast_shape = (1, -1, *(1 for _ in x.shape[2:]))
			weight = self.weight.to(dtype=x.dtype).view(broadcast_shape)
			bias = self.bias.to(dtype=x.dtype).view(broadcast_shape)
			output = output * weight + bias
		return output


class _UpsampleBlock(nn.Module):
	"""Upsample a feature grid and refine it with a convolution."""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		upsample_factor: tuple[int, int, int],
		group_count: int,
	) -> None:
		super().__init__()
		self.upsample_factor = upsample_factor
		self.convolution = nn.Conv3d(
			in_channels,
			out_channels,
			kernel_size=3,
			padding=1,
		)
		self.normalization = _SingletonSafeGroupNorm(group_count, out_channels)
		self.activation = nn.GELU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return the upsampled and refined feature grid."""
		x = interpolate(
			x,
			scale_factor=self.upsample_factor,
			mode='trilinear',
			align_corners=False,
		)
		return self.activation(self.normalization(self.convolution(x)))


class VoxelDecoder3D(nn.Module):
	"""Decode frozen encoder embeddings into dense voxel-class logits."""

	def __init__(  # noqa: PLR0913
		self,
		*,
		embedding_dim: int = 384,
		class_count: int = 6,
		hidden_channels: Sequence[int] = (128, 64, 32),
		upsample_factors: Sequence[Sequence[int]] = (
			(2, 2, 2),
			(2, 2, 2),
			(2, 2, 2),
		),
		patch_size_xyz: Sequence[int] = (8, 8, 8),
		max_group_count: int = 8,
	) -> None:
		"""Initialize the projection, upsampling blocks, and logits head."""
		super().__init__()
		self.embedding_dim = _positive_int(embedding_dim, 'embedding_dim')
		self.class_count = _positive_int(class_count, 'class_count')
		self.hidden_channels = _positive_int_sequence(
			hidden_channels,
			'hidden_channels',
		)
		self.upsample_factors = _factor_sequence(upsample_factors)
		self.patch_size_xyz = _positive_int_triple(
			patch_size_xyz,
			'patch_size_xyz',
		)
		self.max_group_count = _positive_int(max_group_count, 'max_group_count')

		if len(self.hidden_channels) != len(self.upsample_factors):
			msg = (
				'hidden_channels and upsample_factors must have the same length; '
				f'got {len(self.hidden_channels)} and {len(self.upsample_factors)}'
			)
			raise ValueError(msg)
		_factor_product = tuple(
			_product(factor[axis] for factor in self.upsample_factors)
			for axis in range(3)
		)
		if _factor_product != self.patch_size_xyz:
			msg = (
				'upsample factor products must equal patch_size_xyz; '
				f'got products={_factor_product!r}, '
				f'patch_size_xyz={self.patch_size_xyz!r}'
			)
			raise ValueError(msg)

		self.group_norm_groups = tuple(
			_resolve_group_count(channels, self.max_group_count)
			for channels in self.hidden_channels
		)
		self.token_normalization = nn.LayerNorm(self.embedding_dim)
		self.input_projection = nn.Conv3d(
			self.embedding_dim,
			self.hidden_channels[0],
			kernel_size=1,
		)
		stage_inputs = (self.hidden_channels[0], *self.hidden_channels[:-1])
		self.upsample_blocks = nn.ModuleList(
			_UpsampleBlock(in_channels, out_channels, factor, group_count)
			for in_channels, out_channels, factor, group_count in zip(
				stage_inputs,
				self.hidden_channels,
				self.upsample_factors,
				self.group_norm_groups,
				strict=True,
			)
		)
		self.logits_head = nn.Conv3d(
			self.hidden_channels[-1],
			self.class_count,
			kernel_size=1,
		)

	def forward(
		self,
		embeddings: torch.Tensor,
		token_valid_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return voxel logits for a channel-first token embedding grid."""
		_validate_embeddings(embeddings, self.embedding_dim)
		_validate_token_valid_mask(token_valid_mask, embeddings)

		if token_valid_mask is not None:
			invalid_tokens = ~token_valid_mask.unsqueeze(1)
			embeddings = embeddings.masked_fill(invalid_tokens, 0.0)
		x = self.token_normalization(embeddings.movedim(1, -1)).movedim(-1, 1)
		if token_valid_mask is not None:
			x = x.masked_fill(invalid_tokens, 0.0)
		x = self.input_projection(x)
		for block in self.upsample_blocks:
			x = block(x)
		return self.logits_head(x)


def _validate_embeddings(embeddings: torch.Tensor, embedding_dim: int) -> None:
	if not isinstance(embeddings, torch.Tensor):
		msg = f'embeddings must be a torch.Tensor; got {type(embeddings).__name__}'
		raise TypeError(msg)
	if embeddings.ndim != 5:
		msg = f'embeddings must have shape [B,D,TX,TY,TZ]; got {embeddings.shape}'
		raise ValueError(msg)
	if embeddings.shape[1] != embedding_dim:
		msg = (
			f'embeddings channel dimension must equal embedding_dim={embedding_dim}; '
			f'got {embeddings.shape[1]}'
		)
		raise ValueError(msg)
	if not embeddings.is_floating_point():
		msg = f'embeddings must have a floating-point dtype; got {embeddings.dtype}'
		raise TypeError(msg)


def _validate_token_valid_mask(
	mask: torch.Tensor | None,
	embeddings: torch.Tensor,
) -> None:
	if mask is None:
		return
	if not isinstance(mask, torch.Tensor):
		msg = f'token_valid_mask must be a torch.Tensor; got {type(mask).__name__}'
		raise TypeError(msg)
	if mask.dtype != torch.bool:
		msg = f'token_valid_mask must have dtype bool; got {mask.dtype}'
		raise TypeError(msg)
	expected_shape = (embeddings.shape[0], *embeddings.shape[2:])
	if tuple(mask.shape) != expected_shape:
		msg = (
			'token_valid_mask must have shape [B,TX,TY,TZ] matching embeddings; '
			f'expected {expected_shape}, got {tuple(mask.shape)}'
		)
		raise ValueError(msg)
	if mask.device != embeddings.device:
		msg = (
			'token_valid_mask must be on the same device as embeddings; '
			f'got {mask.device} and {embeddings.device}'
		)
		raise ValueError(msg)


def _positive_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _positive_int_sequence(value: Sequence[int], name: str) -> tuple[int, ...]:
	if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
		msg = f'{name} must be a non-empty sequence of positive integers; got {value!r}'
		raise ValueError(msg)
	return tuple(_positive_int(item, name) for item in value)


def _positive_int_triple(value: Sequence[int], name: str) -> tuple[int, int, int]:
	if (
		isinstance(value, (str, bytes))
		or not isinstance(value, Sequence)
		or len(value) != 3
	):
		msg = f'{name} must be a positive integer triple; got {value!r}'
		raise ValueError(msg)
	return (
		_positive_int(value[0], name),
		_positive_int(value[1], name),
		_positive_int(value[2], name),
	)


def _factor_sequence(
	value: Sequence[Sequence[int]],
) -> tuple[tuple[int, int, int], ...]:
	if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
		msg = f'upsample_factors must be a non-empty sequence; got {value!r}'
		raise ValueError(msg)
	return tuple(
		_positive_int_triple(factor, 'upsample_factors') for factor in value
	)


def _resolve_group_count(channels: int, max_group_count: int) -> int:
	return next(
		group_count
		for group_count in range(min(channels, max_group_count), 0, -1)
		if channels % group_count == 0
	)


def _product(values: Iterable[int]) -> int:
	result = 1
	for value in values:
		result *= value
	return result


__all__ = ['VoxelDecoder3D']
