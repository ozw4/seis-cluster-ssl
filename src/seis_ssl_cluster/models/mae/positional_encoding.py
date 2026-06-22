"""3D positional encoding helpers for amplitude MAE token grids."""

from __future__ import annotations

import math

import torch


def build_3d_sincos_position_embedding(
	grid_shape_xyz: tuple[int, int, int],
	embed_dim: int,
) -> torch.Tensor:
	"""Return fixed 3D sin-cos embeddings with shape ``[TX * TY * TZ, D]``."""
	tx_size, ty_size, tz_size = _validate_positive_int_triple(
		grid_shape_xyz,
		'grid_shape_xyz',
	)
	embed_dim = _validate_positive_int(embed_dim, 'embed_dim')
	axis_embed_dim = 2 * math.ceil(math.ceil(embed_dim / 3) / 2)

	x_coords, y_coords, z_coords = torch.meshgrid(
		torch.arange(tx_size, dtype=torch.float32),
		torch.arange(ty_size, dtype=torch.float32),
		torch.arange(tz_size, dtype=torch.float32),
		indexing='ij',
	)
	embeddings = torch.cat(
		[
			_sincos_1d(x_coords.reshape(-1), axis_embed_dim),
			_sincos_1d(y_coords.reshape(-1), axis_embed_dim),
			_sincos_1d(z_coords.reshape(-1), axis_embed_dim),
		],
		dim=1,
	)
	return embeddings[:, :embed_dim].contiguous()


def select_visible_tokens(
	tokens: torch.Tensor,
	pos: torch.Tensor,
	visible_spatial_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Pack visible spatial tokens and positions, padding to the batch maximum."""
	batch_size, num_tokens, embed_dim = _validate_tokens(tokens, 'tokens')
	_validate_positions(pos, num_tokens, embed_dim, tokens.device)
	flat_visible_mask = _flatten_visible_spatial_mask(
		visible_spatial_mask,
		batch_size,
		num_tokens,
		tokens.device,
	)
	visible_counts = flat_visible_mask.sum(dim=1)
	if not (visible_counts > 0).all():
		msg = 'each sample must contain at least one visible spatial token'
		raise ValueError(msg)

	max_visible_tokens = int(visible_counts.max().item())
	visible_tokens = tokens.new_zeros((batch_size, max_visible_tokens, embed_dim))
	visible_pos = pos.new_zeros((batch_size, max_visible_tokens, embed_dim))
	valid_mask = torch.zeros(
		(batch_size, max_visible_tokens),
		dtype=torch.bool,
		device=tokens.device,
	)

	for batch_index in range(batch_size):
		visible_indices = flat_visible_mask[batch_index].nonzero(
			as_tuple=False,
		).squeeze(1)
		count = int(visible_indices.numel())
		visible_tokens[batch_index, :count] = tokens[batch_index, visible_indices]
		visible_pos[batch_index, :count] = pos[visible_indices]
		valid_mask[batch_index, :count] = True

	return visible_tokens, visible_pos, valid_mask


def restore_decoder_sequence(
	visible_tokens: torch.Tensor,
	pos: torch.Tensor,
	visible_spatial_mask: torch.Tensor,
	mask_token: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Scatter packed visible tokens into full XYZ order for the decoder."""
	batch_size, max_visible_tokens, embed_dim = _validate_tokens(
		visible_tokens,
		'visible_tokens',
	)
	num_tokens = _validate_positions_only(pos, embed_dim, visible_tokens.device)
	flat_visible_mask = _flatten_visible_spatial_mask(
		visible_spatial_mask,
		batch_size,
		num_tokens,
		visible_tokens.device,
	)
	visible_counts = flat_visible_mask.sum(dim=1)
	if not (visible_counts > 0).all():
		msg = 'each sample must contain at least one visible spatial token'
		raise ValueError(msg)
	if int(visible_counts.max().item()) > max_visible_tokens:
		msg = (
			'visible_tokens does not contain enough packed tokens; '
			f'got max_visible_tokens={max_visible_tokens!r}, '
			f'required={int(visible_counts.max().item())!r}'
		)
		raise ValueError(msg)

	normalized_mask_token = _normalize_mask_token(
		mask_token,
		embed_dim,
		visible_tokens,
	)
	decoder_tokens = normalized_mask_token.view(1, 1, embed_dim).expand(
		batch_size,
		num_tokens,
		embed_dim,
	).clone()
	for batch_index in range(batch_size):
		visible_indices = flat_visible_mask[batch_index].nonzero(
			as_tuple=False,
		).squeeze(1)
		count = int(visible_indices.numel())
		decoder_tokens[batch_index, visible_indices] = visible_tokens[
			batch_index,
			:count,
		]

	return decoder_tokens + pos.unsqueeze(0), ~flat_visible_mask


def _sincos_1d(positions: torch.Tensor, embed_dim: int) -> torch.Tensor:
	half_dim = embed_dim // 2
	frequencies = torch.arange(half_dim, dtype=torch.float32)
	frequencies = 1.0 / (10_000 ** (frequencies / half_dim))
	angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
	return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def _validate_positive_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_positive_int_triple(
	value: tuple[int, int, int],
	name: str,
) -> tuple[int, int, int]:
	if (
		not isinstance(value, tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
		or any(item <= 0 for item in value)
	):
		msg = f'{name} must be a positive integer triple; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_tokens(tokens: torch.Tensor, name: str) -> tuple[int, int, int]:
	if tokens.ndim != 3:
		msg = (
			f'{name} must be a 3D tensor with shape [B, N, D]; '
			f'got shape={tuple(tokens.shape)!r}'
		)
		raise ValueError(msg)
	batch_size, num_tokens, embed_dim = tokens.shape
	if num_tokens <= 0:
		msg = f'{name} must contain at least one token'
		raise ValueError(msg)
	if embed_dim <= 0:
		msg = f'{name} must have a positive embedding dimension'
		raise ValueError(msg)
	return int(batch_size), int(num_tokens), int(embed_dim)


def _validate_positions(
	pos: torch.Tensor,
	num_tokens: int,
	embed_dim: int,
	device: torch.device,
) -> None:
	if pos.ndim != 2 or tuple(pos.shape) != (num_tokens, embed_dim):
		msg = (
			'pos must have shape [N, D] matching tokens; '
			f'got shape={tuple(pos.shape)!r}, '
			f'expected={(num_tokens, embed_dim)!r}'
		)
		raise ValueError(msg)
	if pos.device != device:
		msg = (
			'pos must be on the same device as tokens; '
			f'got pos_device={pos.device}, tokens_device={device}'
		)
		raise ValueError(msg)


def _validate_positions_only(
	pos: torch.Tensor,
	embed_dim: int,
	device: torch.device,
) -> int:
	if pos.ndim != 2:
		msg = f'pos must have shape [N, D]; got shape={tuple(pos.shape)!r}'
		raise ValueError(msg)
	num_tokens, pos_embed_dim = pos.shape
	if pos_embed_dim != embed_dim:
		msg = (
			'pos last dimension must match visible_tokens; '
			f'got shape={tuple(pos.shape)!r}, embed_dim={embed_dim!r}'
		)
		raise ValueError(msg)
	if pos.device != device:
		msg = (
			'pos must be on the same device as visible_tokens; '
			f'got pos_device={pos.device}, tokens_device={device}'
		)
		raise ValueError(msg)
	return int(num_tokens)


def _flatten_visible_spatial_mask(
	visible_spatial_mask: torch.Tensor,
	batch_size: int,
	num_tokens: int,
	device: torch.device,
) -> torch.Tensor:
	if visible_spatial_mask.ndim != 4:
		msg = (
			'visible_spatial_mask must have shape [B, TX, TY, TZ]; '
			f'got shape={tuple(visible_spatial_mask.shape)!r}'
		)
		raise ValueError(msg)
	if visible_spatial_mask.shape[0] != batch_size:
		msg = (
			'visible_spatial_mask batch dimension must match tokens; '
			f'got shape={tuple(visible_spatial_mask.shape)!r}, '
			f'batch_size={batch_size!r}'
		)
		raise ValueError(msg)
	if visible_spatial_mask.dtype != torch.bool:
		msg = (
			'visible_spatial_mask dtype must be bool; '
			f'got dtype={visible_spatial_mask.dtype}'
		)
		raise TypeError(msg)
	if visible_spatial_mask.device != device:
		msg = (
			'visible_spatial_mask must be on the same device as tokens; '
			f'got mask_device={visible_spatial_mask.device}, tokens_device={device}'
		)
		raise ValueError(msg)
	flat_visible_mask = visible_spatial_mask.reshape(batch_size, -1)
	if flat_visible_mask.shape[1] != num_tokens:
		msg = (
			'visible_spatial_mask spatial dimensions must flatten to N; '
			f'got shape={tuple(visible_spatial_mask.shape)!r}, '
			f'num_tokens={num_tokens!r}'
		)
		raise ValueError(msg)
	return flat_visible_mask


def _normalize_mask_token(
	mask_token: torch.Tensor | None,
	embed_dim: int,
	reference: torch.Tensor,
) -> torch.Tensor:
	if mask_token is None:
		return reference.new_zeros((embed_dim,))
	if mask_token.device != reference.device:
		msg = (
			'mask_token must be on the same device as visible_tokens; '
			f'got mask_token_device={mask_token.device}, '
			f'tokens_device={reference.device}'
		)
		raise ValueError(msg)
	if mask_token.dtype != reference.dtype:
		msg = (
			'mask_token dtype must match visible_tokens; '
			f'got mask_token_dtype={mask_token.dtype}, tokens_dtype={reference.dtype}'
		)
		raise TypeError(msg)
	if tuple(mask_token.shape) == (embed_dim,):
		return mask_token
	if tuple(mask_token.shape) == (1, embed_dim):
		return mask_token.squeeze(0)
	if tuple(mask_token.shape) == (1, 1, embed_dim):
		return mask_token.squeeze(0).squeeze(0)
	msg = (
		'mask_token must have shape [D], [1, D], or [1, 1, D]; '
		f'got shape={tuple(mask_token.shape)!r}'
	)
	raise ValueError(msg)


__all__ = [
	'build_3d_sincos_position_embedding',
	'restore_decoder_sequence',
	'select_visible_tokens',
]
