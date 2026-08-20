"""Barlow Twins wrapper around the amplitude MAE encoder."""

from __future__ import annotations

from collections.abc import Mapping
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
			view_a,
			view_b,
			valid_mask_a=valid_mask_a,
			valid_mask_b=valid_mask_b,
			local_pair_indices_a=local_pair_indices_a,
			local_pair_indices_b=local_pair_indices_b,
		)
		encoded_a = self.backbone.encode_tokens(view_a, valid_mask=valid_mask_a)
		encoded_b = self.backbone.encode_tokens(view_b, valid_mask=valid_mask_b)
		tokens_a, token_grid_a, token_valid_mask_a = _local_encoded_components(
			encoded_a,
			view=view_a,
			view_name='view_a',
		)
		tokens_b, token_grid_b, token_valid_mask_b = _local_encoded_components(
			encoded_b,
			view=view_b,
			view_name='view_b',
		)
		_validate_local_encoded_pair(
			tokens_a,
			tokens_b,
			token_grid_a=token_grid_a,
			token_grid_b=token_grid_b,
			expected_feature_dim=self.backbone.encoder_dim,
		)

		pair_count = _validate_local_pair_indices(
			local_pair_indices_a,
			name='local_pair_indices_a',
			tokens=tokens_a,
		)
		other_pair_count = _validate_local_pair_indices(
			local_pair_indices_b,
			name='local_pair_indices_b',
			tokens=tokens_b,
		)
		if pair_count != other_pair_count:
			msg = (
				'local pair index tensors must contain the same number of pairs; '
				f'got local_pair_indices_a K={pair_count!r}, '
				f'local_pair_indices_b K={other_pair_count!r}'
			)
			raise ValueError(msg)

		selected_a = _gather_valid_local_tokens(
			tokens_a,
			token_valid_mask_a,
			local_pair_indices_a,
			name='local_pair_indices_a',
		)
		selected_b = _gather_valid_local_tokens(
			tokens_b,
			token_valid_mask_b,
			local_pair_indices_b,
			name='local_pair_indices_b',
		)
		feature_dim = tokens_a.shape[2]
		return {
			'z_a': self.projector(selected_a.reshape(-1, feature_dim)),
			'z_b': self.projector(selected_b.reshape(-1, feature_dim)),
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


def _validate_local_input_tensors(  # noqa: PLR0913
	view_a: object,
	view_b: object,
	*,
	valid_mask_a: object,
	valid_mask_b: object,
	local_pair_indices_a: object,
	local_pair_indices_b: object,
) -> None:
	named_tensors = (
		('view_a', view_a),
		('view_b', view_b),
		('valid_mask_a', valid_mask_a),
		('valid_mask_b', valid_mask_b),
		('local_pair_indices_a', local_pair_indices_a),
		('local_pair_indices_b', local_pair_indices_b),
	)
	for name, tensor in named_tensors:
		if not isinstance(tensor, torch.Tensor):
			msg = f'{name} must be a tensor; got {type(tensor).__name__}'
			raise TypeError(msg)
	for name, indices in named_tensors[-2:]:
		if indices.dtype != torch.int64:
			msg = f'{name} dtype must be torch.int64; got {indices.dtype}'
			raise TypeError(msg)

	reference_device = view_a.device
	for name, tensor in named_tensors[1:]:
		if tensor.device != reference_device:
			msg = (
				'all local forward inputs must be on the same device as view_a; '
				f'got {name}_device={tensor.device}, '
				f'view_a_device={reference_device}'
			)
			raise ValueError(msg)


def _local_encoded_components(
	encoded: object,
	*,
	view: torch.Tensor,
	view_name: str,
) -> tuple[torch.Tensor, tuple[int, int, int], torch.Tensor]:
	if not isinstance(encoded, Mapping):
		msg = (
			'AmplitudeMAE3D.encode_tokens() must return a mapping for '
			f'{view_name}; got {type(encoded).__name__}'
		)
		raise TypeError(msg)
	tokens = _validated_local_encoded_tokens(
		encoded.get('tokens'),
		view=view,
		view_name=view_name,
	)
	token_grid = _validated_local_token_grid(
		encoded.get('token_grid_shape'),
		num_tokens=tokens.shape[1],
		view_name=view_name,
	)
	token_valid_mask = _validated_local_token_mask(
		encoded.get('token_valid_mask'),
		tokens=tokens,
		view_name=view_name,
	)
	return tokens, token_grid, token_valid_mask


def _validated_local_encoded_tokens(
	tokens: object,
	*,
	view: torch.Tensor,
	view_name: str,
) -> torch.Tensor:
	if not isinstance(tokens, torch.Tensor):
		raise TypeError(
			'AmplitudeMAE3D.encode_tokens() returned invalid tokens '
			f'for {view_name}'
		)
	if tokens.ndim != 3:
		msg = (
			'encoded tokens must have shape [B, N, D]; '
			f'got {view_name} shape={tuple(tokens.shape)!r}'
		)
		raise ValueError(msg)
	if tokens.shape[1] <= 0 or tokens.shape[2] <= 0:
		msg = (
			'encoded tokens must contain at least one token and feature; '
			f'got {view_name} shape={tuple(tokens.shape)!r}'
		)
		raise ValueError(msg)
	if tokens.shape[0] != view.shape[0]:
		msg = (
			'encoded token batch size must match its input view; '
			f'got {view_name} tokens_batch={tokens.shape[0]!r}, '
			f'view_batch={view.shape[0]!r}'
		)
		raise RuntimeError(msg)
	if tokens.device != view.device:
		msg = (
			'encoded tokens must be on the same device as their input view; '
			f'got {view_name}_tokens_device={tokens.device}, '
			f'{view_name}_device={view.device}'
		)
		raise RuntimeError(msg)
	return tokens


def _validated_local_token_grid(
	token_grid: object,
	*,
	num_tokens: int,
	view_name: str,
) -> tuple[int, int, int]:
	if (
		not isinstance(token_grid, tuple)
		or len(token_grid) != 3
		or any(
			not isinstance(size, int) or isinstance(size, bool) or size <= 0
			for size in token_grid
		)
	):
		msg = (
			'AmplitudeMAE3D.encode_tokens() returned an invalid token grid '
			f'for {view_name}: {token_grid!r}'
		)
		raise RuntimeError(msg)
	if token_grid[0] * token_grid[1] * token_grid[2] != num_tokens:
		msg = (
			'encoded token count must match token_grid_shape; '
			f'got {view_name} tokens={num_tokens!r}, '
			f'token_grid_shape={token_grid!r}'
		)
		raise RuntimeError(msg)
	return token_grid


def _validated_local_token_mask(
	token_valid_mask: object,
	*,
	tokens: torch.Tensor,
	view_name: str,
) -> torch.Tensor:
	if token_valid_mask is None:
		raise RuntimeError(
			'local forward requires encode_tokens() to return a token-valid mask '
			f'for {view_name}'
		)
	if not isinstance(token_valid_mask, torch.Tensor):
		raise TypeError(
			'AmplitudeMAE3D.encode_tokens() returned an invalid token-valid mask '
			f'for {view_name}'
		)
	if token_valid_mask.ndim != 2 or tuple(token_valid_mask.shape) != tuple(
		tokens.shape[:2]
	):
		msg = (
			'encoded token-valid mask must have shape [B, N]; '
			f'got {view_name} mask_shape={tuple(token_valid_mask.shape)!r}, '
			f'expected={tuple(tokens.shape[:2])!r}'
		)
		raise RuntimeError(msg)
	if token_valid_mask.dtype != torch.bool:
		msg = (
			'encoded token-valid mask dtype must be bool; '
			f'got {view_name} dtype={token_valid_mask.dtype}'
		)
		raise RuntimeError(msg)
	if token_valid_mask.device != tokens.device:
		msg = (
			'encoded token-valid mask must be on the same device as tokens; '
			f'got {view_name} mask_device={token_valid_mask.device}, '
			f'tokens_device={tokens.device}'
		)
		raise RuntimeError(msg)
	return token_valid_mask


def _validate_local_encoded_pair(
	tokens_a: torch.Tensor,
	tokens_b: torch.Tensor,
	*,
	token_grid_a: tuple[int, int, int],
	token_grid_b: tuple[int, int, int],
	expected_feature_dim: int,
) -> None:
	if token_grid_a != token_grid_b:
		msg = (
			'local view token grids must match; '
			f'got view_a={token_grid_a!r}, view_b={token_grid_b!r}'
		)
		raise ValueError(msg)
	if tokens_a.shape[0] != tokens_b.shape[0]:
		msg = (
			'local view batch sizes must match; '
			f'got view_a={tokens_a.shape[0]!r}, view_b={tokens_b.shape[0]!r}'
		)
		raise ValueError(msg)
	if tokens_a.shape[1] != tokens_b.shape[1]:
		msg = (
			'local view token counts must match; '
			f'got view_a={tokens_a.shape[1]!r}, view_b={tokens_b.shape[1]!r}'
		)
		raise ValueError(msg)
	if tokens_a.shape[2] != tokens_b.shape[2]:
		msg = (
			'local view feature dimensions must match; '
			f'got view_a={tokens_a.shape[2]!r}, view_b={tokens_b.shape[2]!r}'
		)
		raise ValueError(msg)
	if tokens_a.shape[2] != expected_feature_dim:
		msg = (
			'local view feature dimension must match backbone.encoder_dim; '
			f'got feature_dim={tokens_a.shape[2]!r}, '
			f'encoder_dim={expected_feature_dim!r}'
		)
		raise RuntimeError(msg)
	if tokens_a.device != tokens_b.device:
		msg = (
			'local view encoded tokens must be on the same device; '
			f'got view_a={tokens_a.device}, view_b={tokens_b.device}'
		)
		raise ValueError(msg)


def _validate_local_pair_indices(
	indices: torch.Tensor,
	*,
	name: str,
	tokens: torch.Tensor,
) -> int:
	expected_batch_size, num_tokens = tokens.shape[:2]
	if indices.ndim != 2 or indices.shape[0] != expected_batch_size:
		msg = (
			f'{name} must have shape [B, K]; '
			f'got shape={tuple(indices.shape)!r}, '
			f'expected_batch_size={expected_batch_size!r}'
		)
		raise ValueError(msg)
	pair_count = int(indices.shape[1])
	if pair_count <= 0:
		msg = f'{name} must contain at least one local pair'
		raise ValueError(msg)
	if indices.device != tokens.device:
		msg = (
			f'{name} must be on the same device as encoded tokens; '
			f'got indices_device={indices.device}, tokens_device={tokens.device}'
		)
		raise ValueError(msg)
	if bool(((indices < 0) | (indices >= num_tokens)).any().item()):
		msg = (
			f'{name} values must be in [0, {num_tokens}); '
			f'got minimum={int(indices.min().item())!r}, '
			f'maximum={int(indices.max().item())!r}'
		)
		raise ValueError(msg)
	return pair_count


def _gather_valid_local_tokens(
	tokens: torch.Tensor,
	token_valid_mask: torch.Tensor,
	indices: torch.Tensor,
	*,
	name: str,
) -> torch.Tensor:
	selected_valid_mask = torch.gather(token_valid_mask, dim=1, index=indices)
	if not bool(selected_valid_mask.all().item()):
		invalid_count = int((~selected_valid_mask).sum().item())
		msg = f'{name} selected {invalid_count} invalid token(s)'
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
