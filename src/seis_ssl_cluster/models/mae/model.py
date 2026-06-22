"""Amplitude-only 3D masked autoencoder model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from seis_ssl_cluster.models.common import TransformerStack
from seis_ssl_cluster.models.mae.patching import compute_num_patches, patchify_3d
from seis_ssl_cluster.models.mae.positional_encoding import (
	build_3d_sincos_position_embedding,
	restore_decoder_sequence,
	select_visible_tokens,
)

if TYPE_CHECKING:
	from collections.abc import Mapping


class AmplitudeMAE3D(nn.Module):
	"""Masked autoencoder for single-channel 3D amplitude volumes."""

	def __init__(  # noqa: PLR0913
		self,
		*,
		in_channels: int = 1,
		out_channels: int = 1,
		patch_size_xyz: tuple[int, int, int] = (8, 8, 8),
		encoder_dim: int = 384,
		encoder_depth: int = 8,
		encoder_heads: int = 6,
		decoder_dim: int = 256,
		decoder_depth: int = 4,
		decoder_heads: int = 4,
	) -> None:
		"""Initialize patch projection, transformer stacks, and prediction head."""
		super().__init__()
		self.in_channels = _validate_positive_int(in_channels, 'in_channels')
		self.out_channels = _validate_positive_int(out_channels, 'out_channels')
		self.patch_size_xyz = _validate_patch_size(patch_size_xyz)
		self.encoder_dim = _validate_positive_int(encoder_dim, 'encoder_dim')
		self.decoder_dim = _validate_positive_int(decoder_dim, 'decoder_dim')
		self.patch_volume = (
			self.patch_size_xyz[0] * self.patch_size_xyz[1] * self.patch_size_xyz[2]
		)

		self.patch_projection = nn.Linear(
			self.in_channels * self.patch_volume,
			self.encoder_dim,
		)
		self.encoder = TransformerStack(
			embed_dim=self.encoder_dim,
			num_heads=encoder_heads,
			depth=encoder_depth,
		)
		self.encoder_to_decoder = nn.Linear(self.encoder_dim, self.decoder_dim)
		self.mask_token = nn.Parameter(torch.empty(self.decoder_dim))
		self.decoder = TransformerStack(
			embed_dim=self.decoder_dim,
			num_heads=decoder_heads,
			depth=decoder_depth,
		)
		self.prediction_head = nn.Linear(
			self.decoder_dim,
			self.out_channels * self.patch_volume,
		)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		"""Initialize learned mask token."""
		nn.init.normal_(self.mask_token, std=0.02)

	def forward(
		self,
		batch: Mapping[str, torch.Tensor],
	) -> dict[str, torch.Tensor | tuple[int, int, int]]:
		"""Return full-grid MAE patch predictions for the provided batch."""
		x = _required_tensor(batch, 'x')
		spatial_mask = _required_tensor(batch, 'spatial_mask')
		visible_spatial_mask = _required_tensor(batch, 'visible_spatial_mask')

		local_tokens, token_grid_shape = self._project_patches(x)
		_validate_spatial_masks(
			spatial_mask,
			visible_spatial_mask,
			local_tokens.shape[0],
			token_grid_shape,
			local_tokens.device,
		)

		encoder_pos = build_3d_sincos_position_embedding(
			token_grid_shape,
			self.encoder_dim,
		).to(device=local_tokens.device, dtype=local_tokens.dtype)
		visible_tokens, visible_pos, visible_valid_mask = select_visible_tokens(
			local_tokens,
			encoder_pos,
			visible_spatial_mask,
		)
		encoded_visible_tokens = self.encoder(
			visible_tokens + visible_pos,
			~visible_valid_mask,
		)

		decoder_visible = self.encoder_to_decoder(encoded_visible_tokens)
		decoder_pos = build_3d_sincos_position_embedding(
			token_grid_shape,
			self.decoder_dim,
		).to(device=decoder_visible.device, dtype=decoder_visible.dtype)
		decoder_tokens, _masked_token_mask = restore_decoder_sequence(
			decoder_visible,
			decoder_pos,
			visible_spatial_mask,
			self.mask_token.to(dtype=decoder_visible.dtype),
		)
		decoded = self.decoder(decoder_tokens)
		pred_patches = self.prediction_head(decoded).reshape(
			x.shape[0],
			decoder_tokens.shape[1],
			self.out_channels,
			self.patch_volume,
		)
		return {
			'pred_patches': pred_patches,
			'encoded_visible_tokens': encoded_visible_tokens,
			'token_grid_shape': token_grid_shape,
			'spatial_mask': spatial_mask,
		}

	def encode_tokens(
		self,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> dict[str, torch.Tensor | tuple[int, int, int] | None]:
		"""Encode all spatial tokens without MAE masking."""
		tokens, token_grid_shape = self._project_patches(x)
		pos = build_3d_sincos_position_embedding(
			token_grid_shape,
			self.encoder_dim,
		).to(device=tokens.device, dtype=tokens.dtype)
		token_valid_mask = _token_valid_mask(
			valid_mask,
			x,
			self.patch_size_xyz,
			token_grid_shape,
		)
		key_padding_mask = None
		if token_valid_mask is not None:
			key_padding_mask = ~token_valid_mask

		encoded = self.encoder(tokens + pos.unsqueeze(0), key_padding_mask)
		return {
			'tokens': encoded,
			'token_grid_shape': token_grid_shape,
			'token_valid_mask': token_valid_mask,
		}

	def _project_patches(
		self,
		x: torch.Tensor,
	) -> tuple[torch.Tensor, tuple[int, int, int]]:
		_validate_input_volume(x, self.in_channels)
		batch_size, _channels, x_size, y_size, z_size = x.shape
		token_grid_shape = compute_num_patches(
			(int(x_size), int(y_size), int(z_size)),
			self.patch_size_xyz,
		)[:3]
		patches = patchify_3d(x, self.patch_size_xyz).reshape(
			batch_size,
			-1,
			self.in_channels * self.patch_volume,
		)
		return self.patch_projection(patches), token_grid_shape


def _required_tensor(
	batch: Mapping[str, torch.Tensor],
	key: str,
) -> torch.Tensor:
	try:
		value = batch[key]
	except KeyError as exc:
		msg = f'batch is missing required key {key!r}'
		raise KeyError(msg) from exc
	if not isinstance(value, torch.Tensor):
		msg = f'batch key {key!r} must be a tensor; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _validate_spatial_masks(
	spatial_mask: torch.Tensor,
	visible_spatial_mask: torch.Tensor,
	batch_size: int,
	token_grid_shape: tuple[int, int, int],
	device: torch.device,
) -> None:
	expected_shape = (batch_size, *token_grid_shape)
	for name, mask in (
		('spatial_mask', spatial_mask),
		('visible_spatial_mask', visible_spatial_mask),
	):
		if mask.ndim != 4 or tuple(mask.shape) != expected_shape:
			msg = (
				f'{name} must have shape [B, TX, TY, TZ]; '
				f'got shape={tuple(mask.shape)!r}, expected={expected_shape!r}'
			)
			raise ValueError(msg)
		if mask.dtype != torch.bool:
			msg = f'{name} dtype must be bool; got {mask.dtype}'
			raise TypeError(msg)
		if mask.device != device:
			msg = (
				f'{name} must be on the same device as x; '
				f'got mask_device={mask.device}, x_device={device}'
			)
			raise ValueError(msg)
	if not torch.equal(visible_spatial_mask, ~spatial_mask):
		msg = 'visible_spatial_mask must equal ~spatial_mask'
		raise ValueError(msg)


def _token_valid_mask(
	valid_mask: torch.Tensor | None,
	x: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	token_grid_shape: tuple[int, int, int],
) -> torch.Tensor | None:
	if valid_mask is None:
		return None
	_validate_bool_mask(valid_mask, x.shape[0], x.device)
	token_shape = (x.shape[0], *token_grid_shape)
	if tuple(valid_mask.shape) == token_shape:
		return valid_mask.reshape(x.shape[0], -1)
	if tuple(valid_mask.shape) != (x.shape[0], x.shape[2], x.shape[3], x.shape[4]):
		msg = (
			'valid_mask must have shape [B, X, Y, Z] or [B, TX, TY, TZ]; '
			f'got shape={tuple(valid_mask.shape)!r}'
		)
		raise ValueError(msg)

	mask_patches = patchify_3d(
		valid_mask.unsqueeze(1).to(dtype=x.dtype),
		patch_size_xyz,
	)
	token_valid_mask = mask_patches.squeeze(2).all(dim=-1)
	if not token_valid_mask.any(dim=1).all():
		msg = 'each sample must contain at least one valid token'
		raise ValueError(msg)
	return token_valid_mask


def _validate_bool_mask(
	mask: torch.Tensor,
	batch_size: int,
	device: torch.device,
) -> None:
	if mask.ndim != 4:
		msg = (
			'valid_mask must be a 4D tensor with shape [B, X, Y, Z] '
			f'or [B, TX, TY, TZ]; got shape={tuple(mask.shape)!r}'
		)
		raise ValueError(msg)
	if mask.shape[0] != batch_size:
		msg = (
			'valid_mask batch dimension must match x; '
			f'got shape={tuple(mask.shape)!r}, batch_size={batch_size!r}'
		)
		raise ValueError(msg)
	if mask.dtype != torch.bool:
		msg = f'valid_mask dtype must be bool; got {mask.dtype}'
		raise TypeError(msg)
	if mask.device != device:
		msg = (
			'valid_mask must be on the same device as x; '
			f'got mask_device={mask.device}, x_device={device}'
		)
		raise ValueError(msg)


def _validate_input_volume(x: torch.Tensor, in_channels: int) -> None:
	if x.ndim != 5:
		msg = (
			'x must be a 5D tensor with shape [B, C, X, Y, Z]; '
			f'got shape={tuple(x.shape)!r}'
		)
		raise ValueError(msg)
	if x.shape[1] != in_channels:
		msg = (
			'x channel dimension must match in_channels; '
			f'got shape={tuple(x.shape)!r}, in_channels={in_channels!r}'
		)
		raise ValueError(msg)


def _validate_positive_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_patch_size(value: tuple[int, int, int]) -> tuple[int, int, int]:
	if (
		not isinstance(value, tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
		or any(item <= 0 for item in value)
	):
		msg = f'patch_size_xyz must be a positive integer triple; got {value!r}'
		raise ValueError(msg)
	return value


__all__ = ['AmplitudeMAE3D']
