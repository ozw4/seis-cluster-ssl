"""Barlow Twins wrapper around the amplitude MAE encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

	def forward_local(  # noqa: PLR0913
		self,
		view_a: torch.Tensor,
		view_b: torch.Tensor,
		*,
		valid_mask_a: torch.Tensor,
		valid_mask_b: torch.Tensor,
		local_pair_indices_a: torch.Tensor,
		local_pair_indices_b: torch.Tensor,
	) -> dict[str, torch.Tensor]:
		"""Return projections for corresponding valid tokens in two views."""
		_validate_local_input_tensors(
			valid_mask_a=valid_mask_a,
			valid_mask_b=valid_mask_b,
			local_pair_indices_a=local_pair_indices_a,
			local_pair_indices_b=local_pair_indices_b,
		)
		encoded_a = self.backbone.encode_tokens(view_a, valid_mask=valid_mask_a)
		encoded_b = self.backbone.encode_tokens(view_b, valid_mask=valid_mask_b)
		tokens_a = cast('torch.Tensor', encoded_a['tokens'])
		tokens_b = cast('torch.Tensor', encoded_b['tokens'])
		selected_a = _gather_local_tokens(
			tokens_a,
			local_pair_indices_a,
			name='local_pair_indices_a',
		)
		selected_b = _gather_local_tokens(
			tokens_b,
			local_pair_indices_b,
			name='local_pair_indices_b',
		)
		return {
			'z_a': self.projector(selected_a.flatten(0, 1)),
			'z_b': self.projector(selected_b.flatten(0, 1)),
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


def _validate_local_input_tensors(
	*,
	valid_mask_a: object,
	valid_mask_b: object,
	local_pair_indices_a: object,
	local_pair_indices_b: object,
) -> None:
	for name, valid_mask in (
		('valid_mask_a', valid_mask_a),
		('valid_mask_b', valid_mask_b),
	):
		if not isinstance(valid_mask, torch.Tensor):
			msg = f'{name} must be a tensor; got {type(valid_mask).__name__}'
			raise TypeError(msg)

	for name, indices in (
		('local_pair_indices_a', local_pair_indices_a),
		('local_pair_indices_b', local_pair_indices_b),
	):
		if not isinstance(indices, torch.Tensor):
			msg = f'{name} must be a tensor; got {type(indices).__name__}'
			raise TypeError(msg)
		if indices.dtype != torch.int64:
			msg = f'{name} dtype must be torch.int64; got {indices.dtype}'
			raise TypeError(msg)
		if indices.ndim != 2:
			msg = f'{name} must have shape [B, K]; got shape={tuple(indices.shape)!r}'
			raise ValueError(msg)

	if local_pair_indices_a.shape != local_pair_indices_b.shape:
		msg = (
			'local pair index tensors must have matching [B, K] shapes; '
			f'got local_pair_indices_a={tuple(local_pair_indices_a.shape)!r}, '
			f'local_pair_indices_b={tuple(local_pair_indices_b.shape)!r}'
		)
		raise ValueError(msg)


def _gather_local_tokens(
	tokens: torch.Tensor,
	indices: torch.Tensor,
	*,
	name: str,
) -> torch.Tensor:
	if indices.shape[0] != tokens.shape[0]:
		msg = (
			f'{name} batch dimension must match encoded token batch; '
			f'got indices_batch={indices.shape[0]!r}, '
			f'tokens_batch={tokens.shape[0]!r}'
		)
		raise ValueError(msg)
	gather_indices = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[2])
	return torch.gather(tokens, dim=1, index=gather_indices)


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
