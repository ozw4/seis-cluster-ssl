"""Lightweight voxel decoder for precomputed 3D token embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn.functional import interpolate

from seis_ssl_cluster.models.voxel_decoder.spec import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
	validate_voxel_decoder_implementation,
	voxel_decoder_architecture_mapping,
)


class _VoxelwiseLayerNorm(nn.LayerNorm):
	"""Normalize channels independently at each voxel location."""

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Apply layer normalization without spatially coupling tiles."""
		return super().forward(x.movedim(1, -1)).movedim(-1, 1)


class _UpsampleBlock(nn.Module):
	"""Upsample a feature grid and refine it with a convolution."""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		upsample_factor: tuple[int, int, int],
		*,
		upsample_mode: str,
		normalization: str,
	) -> None:
		super().__init__()
		validate_voxel_decoder_implementation(
			spec=VOXEL_DECODER_SPEC,
			upsample_mode=upsample_mode,
			normalization=normalization,
		)
		self.upsample_factor = upsample_factor
		self.upsample_mode = upsample_mode
		self.normalization_name = normalization
		self.convolution = nn.Conv3d(
			in_channels,
			out_channels,
			kernel_size=3,
			padding=1,
		)
		self.normalization = _VoxelwiseLayerNorm(out_channels)
		self.activation = nn.GELU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return the upsampled and refined feature grid."""
		x = interpolate(
			x,
			scale_factor=self.upsample_factor,
			mode=self.upsample_mode,
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
		spec: str = VOXEL_DECODER_SPEC,
		upsample_mode: str = VOXEL_DECODER_UPSAMPLE_MODE,
		normalization: str = VOXEL_DECODER_NORMALIZATION,
	) -> None:
		"""Initialize the projection, upsampling blocks, and logits head."""
		super().__init__()
		validate_voxel_decoder_implementation(
			spec=spec,
			upsample_mode=upsample_mode,
			normalization=normalization,
		)
		self.spec = spec
		self.upsample_mode = upsample_mode
		self.normalization = normalization
		self.embedding_dim = _positive_int(embedding_dim, 'embedding_dim')
		self.class_count = _positive_int(class_count, 'class_count')
		(
			self.hidden_channels,
			self.upsample_factors,
			self.patch_size_xyz,
		) = _validated_voxel_decoder_architecture(
			hidden_channels,
			upsample_factors,
			patch_size_xyz,
		)

		self.token_normalization = nn.LayerNorm(self.embedding_dim)
		self.input_projection = nn.Conv3d(
			self.embedding_dim,
			self.hidden_channels[0],
			kernel_size=1,
		)
		stage_inputs = (self.hidden_channels[0], *self.hidden_channels[:-1])
		self.upsample_blocks = nn.ModuleList(
			_UpsampleBlock(
				in_channels,
				out_channels,
				factor,
				upsample_mode=self.upsample_mode,
				normalization=self.normalization,
			)
			for in_channels, out_channels, factor in zip(
				stage_inputs,
				self.hidden_channels,
				self.upsample_factors,
				strict=True,
			)
		)
		self.logits_head = nn.Conv3d(
			self.hidden_channels[-1],
			self.class_count,
			kernel_size=1,
		)

	@property
	def architecture(self) -> dict[str, object]:
		"""Return this decoder's canonical JSON-compatible architecture."""
		return voxel_decoder_architecture_mapping(
			spec=self.spec,
			embedding_dim=self.embedding_dim,
			class_count=self.class_count,
			hidden_channels=self.hidden_channels,
			upsample_factors=self.upsample_factors,
			upsample_mode=self.upsample_mode,
			normalization=self.normalization,
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
			valid_voxels = token_valid_mask.unsqueeze(1)
		else:
			valid_voxels = None
		x = self.token_normalization(embeddings.movedim(1, -1)).movedim(-1, 1)
		if token_valid_mask is not None:
			x = x.masked_fill(invalid_tokens, 0.0)
		x = self.input_projection(x)
		if token_valid_mask is not None:
			x = x.masked_fill(invalid_tokens, 0.0)
		for block in self.upsample_blocks:
			x = block(x)
			if valid_voxels is not None:
				valid_voxels = interpolate(
					valid_voxels.to(dtype=x.dtype),
					scale_factor=block.upsample_factor,
					mode=self.upsample_mode,
				).to(dtype=torch.bool)
				x = x.masked_fill(~valid_voxels, 0.0)
		return self.logits_head(x)


def required_context_halo_tokens(
	upsample_factors: Sequence[Sequence[int]],
) -> tuple[int, int, int]:
	"""Return the token halo needed by the decoder's full receptive field."""
	factors = _factor_sequence(upsample_factors)
	required: list[int] = []
	for axis in range(3):
		voxel_width = _product(factor[axis] for factor in factors)
		lower = 0
		upper = voxel_width - 1
		for factor in reversed(factors):
			lower = (lower - 1) // factor[axis]
			upper = (upper + 1) // factor[axis]
		required.append(max(-lower, upper))
	return (required[0], required[1], required[2])


def validate_voxel_decoder_architecture(
	*,
	hidden_channels: Sequence[int],
	upsample_factors: Sequence[Sequence[int]],
	patch_size_xyz: Sequence[int],
) -> None:
	"""Validate decoder stage counts and aggregate upsampling geometry."""
	_validated_voxel_decoder_architecture(
		hidden_channels,
		upsample_factors,
		patch_size_xyz,
	)


def validate_context_halo_tokens(
	*,
	context_halo_tokens: tuple[int, int, int],
	core_size_tokens: tuple[int, int, int],
	token_grid_shape_xyz: tuple[int, int, int],
	upsample_factors: Sequence[Sequence[int]],
) -> None:
	"""Require enough halo on every axis partitioned into multiple core tiles."""
	required = required_context_halo_tokens(upsample_factors)
	insufficient = tuple(
		axis
		for axis in range(3)
		if token_grid_shape_xyz[axis] > core_size_tokens[axis]
		and context_halo_tokens[axis] < required[axis]
	)
	if insufficient:
		raise ValueError(
			'tiles.context_halo_tokens does not cover the decoder receptive field; '
			f'required={required!r}, configured={context_halo_tokens!r}, '
			f'tiled_axes={insufficient!r}'
		)


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


def _validated_voxel_decoder_architecture(
	hidden_channels: Sequence[int],
	upsample_factors: Sequence[Sequence[int]],
	patch_size_xyz: Sequence[int],
) -> tuple[
	tuple[int, ...],
	tuple[tuple[int, int, int], ...],
	tuple[int, int, int],
]:
	channels = _positive_int_sequence(hidden_channels, 'hidden_channels')
	factors = _factor_sequence(upsample_factors)
	patch_size = _positive_int_triple(patch_size_xyz, 'patch_size_xyz')
	if len(channels) != len(factors):
		msg = (
			'hidden_channels and upsample_factors must have the same length; '
			f'got {len(channels)} and {len(factors)}'
		)
		raise ValueError(msg)
	factor_product = tuple(
		_product(factor[axis] for factor in factors) for axis in range(3)
	)
	if factor_product != patch_size:
		msg = (
			'upsample factor products must equal patch_size_xyz; '
			f'got products={factor_product!r}, patch_size_xyz={patch_size!r}'
		)
		raise ValueError(msg)
	return channels, factors, patch_size


def _product(values: Iterable[int]) -> int:
	result = 1
	for value in values:
		result *= value
	return result


__all__ = [
	'VoxelDecoder3D',
	'required_context_halo_tokens',
	'validate_context_halo_tokens',
	'validate_voxel_decoder_architecture',
]
