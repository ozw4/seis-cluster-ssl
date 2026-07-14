"""Canonical identity for the frozen-embedding voxel decoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

VOXEL_DECODER_SPEC = 'frozen_embedding_decoder_nearest_voxel_ln_v1'
VOXEL_DECODER_UPSAMPLE_MODE = 'nearest'
VOXEL_DECODER_NORMALIZATION = 'voxelwise_layer_norm'
_ARCHITECTURE_FIELDS = frozenset(
	{
		'spec',
		'embedding_dim',
		'class_count',
		'hidden_channels',
		'upsample_factors',
		'upsample_mode',
		'normalization',
	}
)


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


def validate_voxel_decoder_architecture_mapping(
	value: object, *, field_prefix: str = 'decoder'
) -> dict[str, object]:
	"""Validate and return the exact canonical decoder identity mapping."""
	if not isinstance(value, Mapping):
		raise TypeError(f'{field_prefix} must be a mapping')
	fields = set(value)
	if fields != _ARCHITECTURE_FIELDS:
		missing = sorted(_ARCHITECTURE_FIELDS - fields)
		extra = sorted(repr(field) for field in fields - _ARCHITECTURE_FIELDS)
		raise ValueError(
			f'{field_prefix} must contain exactly the canonical architecture fields; '
			f'missing={missing!r}, extra={extra!r}'
		)
	validate_voxel_decoder_implementation(
		spec=value['spec'],
		upsample_mode=value['upsample_mode'],
		normalization=value['normalization'],
		field_prefix=field_prefix,
	)
	embedding_dim = _positive_int(
		value['embedding_dim'], f'{field_prefix}.embedding_dim'
	)
	class_count = _positive_int(value['class_count'], f'{field_prefix}.class_count')
	hidden_channels = _positive_int_list(
		value['hidden_channels'], f'{field_prefix}.hidden_channels'
	)
	factors_value = value['upsample_factors']
	if not isinstance(factors_value, list) or not factors_value:
		raise TypeError(
			f'{field_prefix}.upsample_factors must be a non-empty list'
		)
	upsample_factors = [
		_positive_int_list(factor, f'{field_prefix}.upsample_factors')
		for factor in factors_value
	]
	if any(len(factor) != 3 for factor in upsample_factors):
		raise ValueError(
			f'{field_prefix}.upsample_factors entries must contain three integers'
		)
	canonical = voxel_decoder_architecture_mapping(
		spec=VOXEL_DECODER_SPEC,
		embedding_dim=embedding_dim,
		class_count=class_count,
		hidden_channels=hidden_channels,
		upsample_factors=upsample_factors,
		upsample_mode=VOXEL_DECODER_UPSAMPLE_MODE,
		normalization=VOXEL_DECODER_NORMALIZATION,
	)
	if dict(value) != canonical:
		raise ValueError(f'{field_prefix} is not the canonical architecture mapping')
	return canonical


def _positive_int(value: object, field: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError(f'{field} must be a positive integer')
	return value


def _positive_int_list(value: object, field: str) -> list[int]:
	if not isinstance(value, list) or not value:
		raise TypeError(f'{field} must be a non-empty list')
	return [_positive_int(item, field) for item in value]


__all__ = [
	'VOXEL_DECODER_NORMALIZATION',
	'VOXEL_DECODER_SPEC',
	'VOXEL_DECODER_UPSAMPLE_MODE',
	'validate_voxel_decoder_architecture_mapping',
	'validate_voxel_decoder_implementation',
	'voxel_decoder_architecture_mapping',
]
