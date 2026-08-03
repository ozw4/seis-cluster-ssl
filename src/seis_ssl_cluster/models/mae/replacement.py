"""Training-only learned replacement tokens for MAE encoder inputs."""

from __future__ import annotations

import torch
from torch import nn


class LearnedEncoderReplacementToken(nn.Module):
	"""A seeded, standalone learned token in encoder embedding space."""

	def __init__(
		self,
		encoder_dim: int,
		*,
		seed: int,
		device: torch.device | str | None = None,
		dtype: torch.dtype = torch.float32,
	) -> None:
		"""Initialize one finite ``[encoder_dim]`` parameter without global RNG."""
		super().__init__()
		_validate_positive_int(encoder_dim, 'encoder_dim')
		_validate_seed(seed)
		if not dtype.is_floating_point:
			raise TypeError(
				f'dtype must be a floating dtype for the replacement token; got {dtype}'
			)
		resolved_device = torch.device('cpu' if device is None else device)
		generator = torch.Generator(device='cpu')
		generator.manual_seed(seed)
		initial = torch.randn(
			(encoder_dim,),
			generator=generator,
			dtype=torch.float32,
		).mul(0.02)
		initial = initial.to(device=resolved_device, dtype=dtype)
		if not bool(torch.isfinite(initial).all().item()):
			raise ValueError('replacement token initialization must be finite')
		self.replacement_token = nn.Parameter(initial)

	@property
	def token(self) -> nn.Parameter:
		"""Return the learned encoder-space token parameter."""
		return self.replacement_token

	def forward(self) -> torch.Tensor:
		"""Return the learned token for use by ``AmplitudeMAE3D.encode_tokens``."""
		return self.replacement_token


def _validate_positive_int(value: int, name: str) -> None:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{name} must be an integer; got {value!r}')
	if value <= 0:
		raise ValueError(f'{name} must be positive; got {value!r}')


def _validate_seed(value: int) -> None:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'seed must be a nonnegative integer; got {value!r}')
	if value < 0:
		raise ValueError(f'seed must be nonnegative; got {value!r}')


__all__ = ['LearnedEncoderReplacementToken']
