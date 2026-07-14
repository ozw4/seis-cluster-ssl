"""Canonical identity for the frozen-embedding voxel decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence

VOXEL_DECODER_SPEC = 'frozen_embedding_decoder_nearest_voxel_ln_v1'
VOXEL_DECODER_UPSAMPLE_MODE = 'nearest'
VOXEL_DECODER_NORMALIZATION = 'voxelwise_layer_norm'


def validate_voxel_decoder_implementation(
	*,
	spec: object,
	upsample_mode: object,
	normalization: object,
	field_prefix: str = '',
) -> None:
	"""Require the one decoder implementation supported by this package."""
	prefix = f'{field_prefix}.' if field_prefix else ''
	for field, value, expected in (
		('spec', spec, VOXEL_DECODER_SPEC),
		('upsample_mode', upsample_mode, VOXEL_DECODER_UPSAMPLE_MODE),
		('normalization', normalization, VOXEL_DECODER_NORMALIZATION),
	):
		if value != expected:
			raise ValueError(
				f'{prefix}{field} must be {expected!r}; got {value!r}'
			)


def voxel_decoder_architecture_mapping(  # noqa: PLR0913
	*,
	embedding_dim: int,
	class_count: int,
	hidden_channels: Sequence[int],
	upsample_factors: Sequence[Sequence[int]],
	spec: str = VOXEL_DECODER_SPEC,
	upsample_mode: str = VOXEL_DECODER_UPSAMPLE_MODE,
	normalization: str = VOXEL_DECODER_NORMALIZATION,
) -> dict[str, object]:
	"""Return the canonical JSON-compatible decoder architecture mapping."""
	validate_voxel_decoder_implementation(
		spec=spec,
		upsample_mode=upsample_mode,
		normalization=normalization,
	)
	return {
		'spec': spec,
		'embedding_dim': embedding_dim,
		'class_count': class_count,
		'hidden_channels': list(hidden_channels),
		'upsample_factors': [list(factor) for factor in upsample_factors],
		'upsample_mode': upsample_mode,
		'normalization': normalization,
	}


__all__ = [
	'VOXEL_DECODER_NORMALIZATION',
	'VOXEL_DECODER_SPEC',
	'VOXEL_DECODER_UPSAMPLE_MODE',
	'validate_voxel_decoder_implementation',
	'voxel_decoder_architecture_mapping',
]
