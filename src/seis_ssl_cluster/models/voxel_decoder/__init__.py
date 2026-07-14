"""Voxel decoder models for frozen seismic embeddings."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from seis_ssl_cluster.models.voxel_decoder.spec import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
	validate_voxel_decoder_architecture_mapping,
	validate_voxel_decoder_implementation,
	voxel_decoder_architecture_mapping,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.models.voxel_decoder.model import (
		VoxelDecoder3D,
		required_context_halo_tokens,
		validate_context_halo_tokens,
		validate_voxel_decoder_architecture,
	)

_MODEL_EXPORTS = frozenset(
	{
		'VoxelDecoder3D',
		'required_context_halo_tokens',
		'validate_context_halo_tokens',
		'validate_voxel_decoder_architecture',
	}
)


def __getattr__(name: str) -> object:
	"""Load torch-backed model exports only when they are requested."""
	if name not in _MODEL_EXPORTS:
		raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
	model_module = import_module('seis_ssl_cluster.models.voxel_decoder.model')
	value = getattr(model_module, name)
	globals()[name] = value
	return value


__all__ = [
	'VOXEL_DECODER_NORMALIZATION',
	'VOXEL_DECODER_SPEC',
	'VOXEL_DECODER_UPSAMPLE_MODE',
	'VoxelDecoder3D',
	'required_context_halo_tokens',
	'validate_context_halo_tokens',
	'validate_voxel_decoder_architecture',
	'validate_voxel_decoder_architecture_mapping',
	'validate_voxel_decoder_implementation',
	'voxel_decoder_architecture_mapping',
]
