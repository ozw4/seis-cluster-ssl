"""Reusable transformer blocks for amplitude MAE token sequences."""

from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
	"""Pre-norm transformer block over ``[B, N, D]`` token sequences."""

	def __init__(
		self,
		*,
		embed_dim: int,
		num_heads: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
	) -> None:
		"""Initialize self-attention, MLP, and normalization layers."""
		super().__init__()
		self.embed_dim = _validate_positive_int(embed_dim, 'embed_dim')
		self.num_heads = _validate_positive_int(num_heads, 'num_heads')
		if self.embed_dim % self.num_heads != 0:
			msg = (
				'embed_dim must be divisible by num_heads; '
				f'got embed_dim={self.embed_dim!r}, num_heads={self.num_heads!r}'
			)
			raise ValueError(msg)

		self.mlp_ratio = _validate_positive_float(mlp_ratio, 'mlp_ratio')
		self.dropout_prob = _validate_dropout(dropout)
		hidden_dim = int(self.embed_dim * self.mlp_ratio)
		if hidden_dim <= 0:
			msg = (
				'mlp_ratio produces an empty hidden dimension; '
				f'got embed_dim={self.embed_dim!r}, mlp_ratio={self.mlp_ratio!r}'
			)
			raise ValueError(msg)

		self.norm1 = nn.LayerNorm(self.embed_dim)
		self.attention = nn.MultiheadAttention(
			self.embed_dim,
			self.num_heads,
			dropout=self.dropout_prob,
			batch_first=True,
		)
		self.dropout1 = nn.Dropout(self.dropout_prob)
		self.norm2 = nn.LayerNorm(self.embed_dim)
		self.mlp = nn.Sequential(
			nn.Linear(self.embed_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(self.dropout_prob),
			nn.Linear(hidden_dim, self.embed_dim),
			nn.Dropout(self.dropout_prob),
		)

	def forward(
		self,
		tokens: torch.Tensor,
		key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return transformed tokens with the same shape as ``tokens``."""
		batch_size, num_tokens = _validate_tokens(tokens, self.embed_dim)
		_validate_key_padding_mask(
			key_padding_mask,
			batch_size,
			num_tokens,
			tokens.device,
		)

		attention_input = self.norm1(tokens)
		attention_output, _attention_weights = self.attention(
			attention_input,
			attention_input,
			attention_input,
			key_padding_mask=key_padding_mask,
			need_weights=False,
		)
		tokens = tokens + self.dropout1(attention_output)
		return tokens + self.mlp(self.norm2(tokens))


class TransformerStack(nn.Module):
	"""Stack of pre-norm transformer blocks."""

	def __init__(
		self,
		*,
		embed_dim: int,
		num_heads: int,
		depth: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
	) -> None:
		"""Initialize ``depth`` identical transformer blocks."""
		super().__init__()
		self.embed_dim = _validate_positive_int(embed_dim, 'embed_dim')
		self.num_heads = _validate_positive_int(num_heads, 'num_heads')
		self.depth = _validate_positive_int(depth, 'depth')
		self.layers = nn.ModuleList(
			[
				TransformerBlock(
					embed_dim=self.embed_dim,
					num_heads=self.num_heads,
					mlp_ratio=mlp_ratio,
					dropout=dropout,
				)
				for _layer_index in range(self.depth)
			],
		)

	def forward(
		self,
		tokens: torch.Tensor,
		key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Apply each transformer block in sequence."""
		for layer in self.layers:
			tokens = layer(tokens, key_padding_mask)
		return tokens


def _validate_positive_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_positive_float(value: float, name: str) -> float:
	if not isinstance(value, (float, int)) or isinstance(value, bool):
		msg = f'{name} must be a float; got {value!r}'
		raise TypeError(msg)
	value = float(value)
	if value <= 0.0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_dropout(value: float) -> float:
	if not isinstance(value, (float, int)) or isinstance(value, bool):
		msg = f'dropout must be a float; got {value!r}'
		raise TypeError(msg)
	value = float(value)
	if not 0.0 <= value <= 1.0:
		msg = f'dropout must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_tokens(tokens: torch.Tensor, embed_dim: int) -> tuple[int, int]:
	if tokens.ndim != 3:
		msg = (
			'tokens must be a 3D tensor with shape [B, N, D]; '
			f'got shape={tuple(tokens.shape)!r}'
		)
		raise ValueError(msg)
	batch_size, num_tokens, token_embed_dim = tokens.shape
	if token_embed_dim != embed_dim:
		msg = (
			'tokens last dimension must equal embed_dim; '
			f'got shape={tuple(tokens.shape)!r}, embed_dim={embed_dim!r}'
		)
		raise ValueError(msg)
	if num_tokens <= 0:
		msg = 'tokens must contain at least one token'
		raise ValueError(msg)
	return int(batch_size), int(num_tokens)


def _validate_key_padding_mask(
	key_padding_mask: torch.Tensor | None,
	batch_size: int,
	num_tokens: int,
	device: torch.device,
) -> None:
	if key_padding_mask is None:
		return
	if key_padding_mask.ndim != 2 or tuple(key_padding_mask.shape) != (
		batch_size,
		num_tokens,
	):
		msg = (
			'key_padding_mask must have shape [B, N]; '
			f'got shape={tuple(key_padding_mask.shape)!r}, '
			f'expected={(batch_size, num_tokens)!r}'
		)
		raise ValueError(msg)
	if key_padding_mask.dtype != torch.bool:
		msg = f'key_padding_mask dtype must be bool; got {key_padding_mask.dtype}'
		raise TypeError(msg)
	if key_padding_mask.device != device:
		msg = (
			'key_padding_mask must be on the same device as tokens; '
			f'got mask_device={key_padding_mask.device}, tokens_device={device}'
		)
		raise ValueError(msg)
	if key_padding_mask.all(dim=1).any():
		msg = 'each sample must contain at least one unmasked token'
		raise ValueError(msg)


__all__ = ['TransformerBlock', 'TransformerStack']
