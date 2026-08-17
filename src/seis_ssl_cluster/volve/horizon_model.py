'''Shared five-horizon decoder for frozen and end-to-end Volve inputs.'''

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

HORIZON_DECODER_SEED = 42000
HORIZON_EMBEDDING_DIM = 384
HORIZON_PATCH_SIZE = (8, 8, 8)
HORIZON_CORE_SIZE_TOKENS = (8, 8, 27)
HORIZON_CONTEXT_HALO_TOKENS = (1, 1, 0)
HORIZON_INPUT_SIZE_TOKENS = (10, 10, 27)
HORIZON_CORE_SIZE_VOXELS = (64, 64, 216)
HORIZON_INPUT_SIZE_VOXELS = (80, 80, 216)
HORIZON_HIDDEN_CHANNELS = (128, 64, 32)
HORIZON_UPSAMPLE_FACTORS = ((2, 2, 2),) * 3


@dataclass(frozen=True)
class HorizonDecoderGeometry:
	'''Token input and supervised-core geometry for one decoder call.'''

	patch_size_xyz: tuple[int, int, int] = HORIZON_PATCH_SIZE
	core_size_tokens: tuple[int, int, int] = HORIZON_CORE_SIZE_TOKENS
	context_halo_tokens: tuple[int, int, int] = HORIZON_CONTEXT_HALO_TOKENS

	@property
	def input_size_tokens(self) -> tuple[int, int, int]:
		'''Return the halo-padded decoder input size in tokens.'''
		return cast(
			'tuple[int, int, int]',
			tuple(
				self.core_size_tokens[axis]
				+ 2 * self.context_halo_tokens[axis]
				for axis in range(3)
			),
		)

	@property
	def core_size_voxels(self) -> tuple[int, int, int]:
		'''Return the supervised output size in voxels.'''
		return cast(
			'tuple[int, int, int]',
			tuple(
				self.core_size_tokens[axis] * self.patch_size_xyz[axis]
				for axis in range(3)
			),
		)

	def validate(self) -> None:
		'''Require the one fixed Volve horizon decoder geometry.'''
		for name, value in (
			('patch_size_xyz', self.patch_size_xyz),
			('core_size_tokens', self.core_size_tokens),
		):
			if len(value) != 3 or any(
				not isinstance(axis, Integral)
				or isinstance(axis, bool)
				or axis <= 0
				for axis in value
			):
				raise ValueError(f'{name} must be a positive integer triple')
		if len(self.context_halo_tokens) != 3 or any(
			not isinstance(axis, Integral)
			or isinstance(axis, bool)
			or axis < 0
			for axis in self.context_halo_tokens
		):
			raise ValueError(
				'context_halo_tokens must be a non-negative integer triple'
			)
		if self.patch_size_xyz != HORIZON_PATCH_SIZE:
			raise ValueError(f'patch_size_xyz must be {HORIZON_PATCH_SIZE!r}')
		if self.core_size_tokens != HORIZON_CORE_SIZE_TOKENS:
			raise ValueError(
				f'core_size_tokens must be {HORIZON_CORE_SIZE_TOKENS!r}'
			)
		if self.context_halo_tokens != HORIZON_CONTEXT_HALO_TOKENS:
			raise ValueError(
				'context_halo_tokens must be '
				f'{HORIZON_CONTEXT_HALO_TOKENS!r}'
			)


DEFAULT_HORIZON_DECODER_GEOMETRY = HorizonDecoderGeometry()


class VolveHorizonDecoder(nn.Module):
	'''Decode a halo-padded token grid and return central five-horizon logits.'''

	def __init__(
		self,
		*,
		embedding_dim: int = HORIZON_EMBEDDING_DIM,
		hidden_channels: Sequence[int] = HORIZON_HIDDEN_CHANNELS,
		geometry: HorizonDecoderGeometry = DEFAULT_HORIZON_DECODER_GEOMETRY,
	) -> None:
		'''Build the fixed nearest-upsampling decoder and central-core crop.'''
		super().__init__()
		geometry.validate()
		self.geometry = geometry
		self.voxel_decoder = VoxelDecoder3D(
			embedding_dim=embedding_dim,
			class_count=len(HORIZON_NAMES),
			hidden_channels=hidden_channels,
			upsample_factors=HORIZON_UPSAMPLE_FACTORS,
			patch_size_xyz=geometry.patch_size_xyz,
		)

	@property
	def embedding_dim(self) -> int:
		'''Return the required encoder embedding width.'''
		return self.voxel_decoder.embedding_dim

	@property
	def architecture(self) -> dict[str, object]:
		'''Return the decoder and supervised-core architecture identity.'''
		return {
			**self.voxel_decoder.architecture,
			'horizon_names': list(HORIZON_NAMES),
			'patch_size_xyz': list(self.geometry.patch_size_xyz),
			'core_size_tokens': list(self.geometry.core_size_tokens),
			'context_halo_tokens': list(self.geometry.context_halo_tokens),
			'input_size_tokens': list(self.geometry.input_size_tokens),
		}

	def forward(
		self,
		embeddings: torch.Tensor,
		token_valid_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		'''Return `[B,5,core_x,core_y,216]` logits from frozen embeddings.'''
		self._validate_input_grid(embeddings, token_valid_mask)
		context_logits = self.voxel_decoder(embeddings, token_valid_mask)
		halo_voxels = tuple(
			self.geometry.context_halo_tokens[axis]
			* self.geometry.patch_size_xyz[axis]
			for axis in range(3)
		)
		core_voxels = self.geometry.core_size_voxels
		x_slice = slice(halo_voxels[0], halo_voxels[0] + core_voxels[0])
		y_slice = slice(halo_voxels[1], halo_voxels[1] + core_voxels[1])
		z_slice = slice(halo_voxels[2], halo_voxels[2] + core_voxels[2])
		return context_logits[:, :, x_slice, y_slice, z_slice]

	def forward_encoder_output(
		self,
		encoded: Mapping[str, object],
	) -> torch.Tensor:
		'''Decode the mapping returned by `AmplitudeMAE3D.encode_tokens`.'''
		tokens = encoded.get('tokens')
		grid_value = encoded.get('token_grid_shape')
		valid = encoded.get('token_valid_mask')
		if not isinstance(tokens, torch.Tensor):
			raise TypeError('encoder output tokens must be a tensor')
		if not _is_int_triplet(grid_value):
			raise TypeError('encoder token_grid_shape must be an integer triple')
		grid = cast('tuple[int, int, int]', grid_value)
		if grid != self.geometry.input_size_tokens:
			raise ValueError(
				'encoder token grid does not match Volve horizon decoder input; '
				f'expected {self.geometry.input_size_tokens!r}, got {grid!r}'
			)
		expected_tokens = (tokens.shape[0], _product(grid), self.embedding_dim)
		if tuple(tokens.shape) != expected_tokens:
			raise ValueError(
				'encoder tokens must have shape [B,TX*TY*TZ,D]; '
				f'expected {expected_tokens!r}, got {tuple(tokens.shape)!r}'
			)
		if not isinstance(valid, torch.Tensor):
			raise TypeError('encoder output token_valid_mask must be a tensor')
		expected_valid = (tokens.shape[0], _product(grid))
		if tuple(valid.shape) != expected_valid:
			raise ValueError(
				'encoder token_valid_mask does not match the reported token grid'
			)
		embeddings = tokens.reshape(tokens.shape[0], *grid, self.embedding_dim)
		embeddings = embeddings.movedim(-1, 1)
		return self(embeddings, valid.reshape(tokens.shape[0], *grid))

	def _validate_input_grid(
		self,
		embeddings: torch.Tensor,
		token_valid_mask: torch.Tensor | None,
	) -> None:
		if embeddings.ndim != 5:
			return
		grid = tuple(embeddings.shape[2:])
		if grid != self.geometry.input_size_tokens:
			raise ValueError(
				'Volve horizon embeddings must use the fixed halo-padded token grid; '
				f'expected {self.geometry.input_size_tokens!r}, got {grid!r}'
			)
		if token_valid_mask is not None and tuple(token_valid_mask.shape[1:]) != grid:
			raise ValueError('token-valid mask grid must match horizon embeddings')


def create_volve_horizon_decoder(
	*,
	seed: int = HORIZON_DECODER_SEED,
	embedding_dim: int = HORIZON_EMBEDDING_DIM,
	hidden_channels: Sequence[int] = HORIZON_HIDDEN_CHANNELS,
	geometry: HorizonDecoderGeometry = DEFAULT_HORIZON_DECODER_GEOMETRY,
) -> VolveHorizonDecoder:
	'''Create a deterministically initialized decoder without changing caller RNG.'''
	if not isinstance(seed, Integral) or isinstance(seed, bool):
		raise TypeError('decoder seed must be an integer')
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(int(seed))
		return VolveHorizonDecoder(
			embedding_dim=embedding_dim,
			hidden_channels=hidden_channels,
			geometry=geometry,
		)


def _is_int_triplet(value: object) -> bool:
	return (
		isinstance(value, tuple)
		and len(value) == 3
		and all(
			isinstance(axis, Integral) and not isinstance(axis, bool) and axis > 0
			for axis in value
		)
	)


def _product(values: Sequence[int]) -> int:
	result = 1
	for value in values:
		result *= value
	return result


__all__ = [
	'DEFAULT_HORIZON_DECODER_GEOMETRY',
	'HORIZON_CONTEXT_HALO_TOKENS',
	'HORIZON_CORE_SIZE_TOKENS',
	'HORIZON_CORE_SIZE_VOXELS',
	'HORIZON_DECODER_SEED',
	'HORIZON_EMBEDDING_DIM',
	'HORIZON_HIDDEN_CHANNELS',
	'HORIZON_INPUT_SIZE_TOKENS',
	'HORIZON_INPUT_SIZE_VOXELS',
	'HORIZON_PATCH_SIZE',
	'HORIZON_UPSAMPLE_FACTORS',
	'HorizonDecoderGeometry',
	'VolveHorizonDecoder',
	'create_volve_horizon_decoder',
]
