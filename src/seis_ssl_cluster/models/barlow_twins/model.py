"""Barlow Twins wrapper around the amplitude MAE encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from seis_ssl_cluster.models.mae import AmplitudeMAE3D

if TYPE_CHECKING:
	from collections.abc import Iterator


class BarlowTwins3D(nn.Module):
	"""Encode two 3D views with one shared MAE encoder and projector."""

	def __init__(
		self,
		backbone: AmplitudeMAE3D,
		*,
		projector_dim: int,
	) -> None:
		"""Initialize the shared backbone and three-layer projector."""
		super().__init__()
		if not isinstance(backbone, AmplitudeMAE3D):
			msg = (
				'backbone must be an AmplitudeMAE3D; '
				f'got {type(backbone).__name__}'
			)
			raise TypeError(msg)
		if not isinstance(projector_dim, int) or isinstance(projector_dim, bool):
			msg = f'projector_dim must be an integer; got {projector_dim!r}'
			raise TypeError(msg)
		if projector_dim <= 0:
			msg = f'projector_dim must be positive; got {projector_dim!r}'
			raise ValueError(msg)

		self.backbone = backbone
		self.projector_dim = projector_dim
		self.projector = nn.Sequential(
			nn.Linear(backbone.encoder_dim, projector_dim),
			nn.BatchNorm1d(projector_dim),
			nn.ReLU(),
			nn.Linear(projector_dim, projector_dim),
			nn.BatchNorm1d(projector_dim),
			nn.ReLU(),
			nn.Linear(projector_dim, projector_dim),
		)

		self.backbone.encoder_to_decoder.requires_grad_(requires_grad=False)
		self.backbone.mask_token.requires_grad_(requires_grad=False)
		self.backbone.decoder.requires_grad_(requires_grad=False)
		self.backbone.prediction_head.requires_grad_(requires_grad=False)

	def forward(
		self,
		view_a: torch.Tensor,
		view_b: torch.Tensor,
		*,
		valid_mask_a: torch.Tensor | None = None,
		valid_mask_b: torch.Tensor | None = None,
	) -> dict[str, torch.Tensor]:
		"""Return projected crop representations for both views."""
		return {
			'z_a': self.encode_view(view_a, valid_mask=valid_mask_a),
			'z_b': self.encode_view(view_b, valid_mask=valid_mask_b),
		}

	def encode_view(
		self,
		view: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Encode, valid-token mean pool, and project one crop."""
		encoded = self.backbone.encode_tokens(view, valid_mask=valid_mask)
		tokens = encoded['tokens']
		token_valid_mask = encoded['token_valid_mask']
		if not isinstance(tokens, torch.Tensor):
			raise TypeError('AmplitudeMAE3D.encode_tokens() returned invalid tokens')
		if token_valid_mask is not None and not isinstance(
			token_valid_mask,
			torch.Tensor,
		):
			raise RuntimeError(
				'AmplitudeMAE3D.encode_tokens() returned an invalid token-valid mask'
			)
		pooled = mean_pool_encoded_tokens(tokens, token_valid_mask)
		return self.projector(pooled)

	def pretraining_parameters(self) -> Iterator[nn.Parameter]:
		"""Yield only patch projection, encoder, and projector parameters."""
		yield from self.backbone.patch_projection.parameters()
		yield from self.backbone.encoder.parameters()
		yield from self.projector.parameters()


def mean_pool_encoded_tokens(
	tokens: torch.Tensor,
	token_valid_mask: torch.Tensor | None,
) -> torch.Tensor:
	"""Mean pool ``[B, N, D]`` tokens, excluding invalid token positions."""
	if tokens.ndim != 3:
		msg = (
			'tokens must have shape [B, N, D]; '
			f'got shape={tuple(tokens.shape)!r}'
		)
		raise ValueError(msg)
	if tokens.shape[1] == 0:
		raise ValueError('tokens must contain at least one token')
	if token_valid_mask is None:
		return tokens.mean(dim=1)
	if token_valid_mask.ndim != 2 or tuple(token_valid_mask.shape) != tuple(
		tokens.shape[:2]
	):
		msg = (
			'token_valid_mask must have shape [B, N]; '
			f'got shape={tuple(token_valid_mask.shape)!r}, '
			f'expected={tuple(tokens.shape[:2])!r}'
		)
		raise ValueError(msg)
	if token_valid_mask.dtype != torch.bool:
		msg = f'token_valid_mask dtype must be bool; got {token_valid_mask.dtype}'
		raise TypeError(msg)
	if token_valid_mask.device != tokens.device:
		msg = (
			'token_valid_mask must be on the same device as tokens; '
			f'got mask_device={token_valid_mask.device}, '
			f'tokens_device={tokens.device}'
		)
		raise ValueError(msg)

	valid_counts = token_valid_mask.sum(dim=1)
	if not bool(valid_counts.gt(0).all().item()):
		raise ValueError('each sample must contain at least one valid token')
	masked_tokens = tokens.masked_fill(~token_valid_mask.unsqueeze(-1), 0)
	return masked_tokens.sum(dim=1) / valid_counts.unsqueeze(-1).to(tokens.dtype)


__all__ = ['BarlowTwins3D', 'mean_pool_encoded_tokens']
