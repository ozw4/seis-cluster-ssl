"""Closed section-layout contract for the F3 voxel benchmark."""
# ruff: noqa: CPY001

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_str,
	_validate_allowed_keys,
)

CONTRACT_SCHEMA_VERSION = 'f3_voxel_section_layout_contract_v1'
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZES = ('small', 'medium', 'large')
LINE_COUNTS = {
	'small': (1, 1),
	'medium': (2, 2),
	'large': (4, 4),
}
STATISTICAL_UNIT = 'layout_id'
NESTING_SEMANTICS = 'strict_small_medium_large'
VALIDATION_MASK_SEMANTICS = 'shared_across_all_layouts_sizes_and_models'
STABLE_SELECTION_SEMANTICS = 'stable_sha256_voxel_rank_v1'
PATCH_SIZE = (8, 8, 8)
DECODER_SEED = 42000

FIXED_DECODER_CONTRACT: Mapping[str, object] = MappingProxyType({
	'spec': 'frozen_embedding_decoder_nearest_voxel_ln_v1',
	'embedding_dim': 384,
	'class_count': 6,
	'hidden_channels': (128, 64, 32),
	'upsample_factors': ((2, 2, 2), (2, 2, 2), (2, 2, 2)),
	'upsample_mode': 'nearest',
	'normalization': 'voxelwise_layer_norm',
	'epochs': 50,
	'batch_size': 1,
	'learning_rate': 0.001,
	'weight_decay': 0.0001,
	'class_weight': 'balanced',
	'sampling_mode': 'uniform_tiles_with_replacement',
	'steps_per_epoch': 440,
	'amp': True,
	'gradient_clip_norm': 1.0,
	'write_probabilities': False,
	'seed': DECODER_SEED,
})


@dataclass(frozen=True)
class F3SectionLayoutSizeSpec:
	"""One nested teacher-line selection and its calibrated voxel target."""

	data_size: str
	inline_lines: tuple[int, ...]
	crossline_lines: tuple[int, ...]
	target_train_voxel_count: int


@dataclass(frozen=True)
class F3SectionLayoutSpec:
	"""The small, medium, and large selections for one statistical unit."""

	layout_id: str
	sizes: tuple[F3SectionLayoutSizeSpec, ...]

	@property
	def size_by_name(self) -> Mapping[str, F3SectionLayoutSizeSpec]:
		"""Index this layout's size specifications by canonical name."""
		return {size.data_size: size for size in self.sizes}


@dataclass(frozen=True)
class F3SectionLayoutContract:
	"""Resolved scientific identity shared by builders and runners."""

	layouts: tuple[F3SectionLayoutSpec, ...]
	patch_size: tuple[int, int, int]
	allowed_relative_error: float
	statistical_unit: str
	nesting_semantics: str
	validation_mask_semantics: str
	stable_selection_semantics: str
	decoder_seed: int
	decoder: Mapping[str, object]

	@property
	def layout_by_id(self) -> Mapping[str, F3SectionLayoutSpec]:
		"""Index the five statistical units by canonical layout ID."""
		return {layout.layout_id: layout for layout in self.layouts}


def f3_lithology_voxel_section_layout_contract_from_mapping(
	config: Mapping[str, object],
	*,
	line_inventory: Sequence[Mapping[str, object]]
	| Mapping[str, Sequence[int]]
	| None = None,
) -> F3SectionLayoutContract:
	"""Resolve a generated contract using only supplied in-memory mappings.

	When ``line_inventory`` is a sequence, rows use the existing F3 inventory
	fields ``split``, ``slice_type``, and ``slice_index``. A mapping with exact
	``inline`` and ``crossline`` keys may instead supply validation lines directly.
	No path is opened by this resolver.
	"""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'schema_version',
				'statistical_unit',
				'nesting_semantics',
				'validation_mask_semantics',
				'stable_selection_semantics',
				'patch_size',
				'allowed_relative_error',
				'decoder_seed',
				'layouts',
				'decoder',
			}
		),
		prefix='config',
	)
	_exact_string(config, 'schema_version', CONTRACT_SCHEMA_VERSION)
	statistical_unit = _exact_string(
		config, 'statistical_unit', STATISTICAL_UNIT
	)
	nesting = _exact_string(config, 'nesting_semantics', NESTING_SEMANTICS)
	validation_mask = _exact_string(
		config, 'validation_mask_semantics', VALIDATION_MASK_SEMANTICS
	)
	stable_selection = _exact_string(
		config,
		'stable_selection_semantics',
		STABLE_SELECTION_SEMANTICS,
	)
	patch_size = _integer_tuple(config.get('patch_size'), 'patch_size')
	if patch_size != PATCH_SIZE:
		raise ValueError(f'patch_size must be exactly {list(PATCH_SIZE)!r}')
	allowed_relative_error = _relative_error(config.get('allowed_relative_error'))
	decoder_seed = _integer(config.get('decoder_seed'), 'decoder_seed')
	if decoder_seed != DECODER_SEED:
		raise ValueError(f'decoder_seed must be exactly {DECODER_SEED}')
	layouts = _resolve_layouts(config.get('layouts'))
	_validate_target_counts(layouts)
	_validate_validation_disjoint(layouts, line_inventory=line_inventory)
	decoder = _resolve_fixed_decoder(config.get('decoder'))
	return F3SectionLayoutContract(
		layouts=layouts,
		patch_size=PATCH_SIZE,
		allowed_relative_error=allowed_relative_error,
		statistical_unit=statistical_unit,
		nesting_semantics=nesting,
		validation_mask_semantics=validation_mask,
		stable_selection_semantics=stable_selection,
		decoder_seed=decoder_seed,
		decoder=decoder,
	)


resolve_f3_lithology_voxel_section_layout_contract = (
	f3_lithology_voxel_section_layout_contract_from_mapping
)
f3_section_layout_contract_from_mapping = (
	f3_lithology_voxel_section_layout_contract_from_mapping
)
f3_lithology_voxel_section_layout_config_from_mapping = (
	f3_lithology_voxel_section_layout_contract_from_mapping
)


def _resolve_layouts(value: object) -> tuple[F3SectionLayoutSpec, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError('layouts must be a list of layout mappings')
	if len(value) != len(LAYOUT_IDS):
		raise ValueError(f'layouts must contain exactly {len(LAYOUT_IDS)} entries')
	layouts = tuple(
		_resolve_layout(item, index=index) for index, item in enumerate(value)
	)
	ids = tuple(layout.layout_id for layout in layouts)
	if len(set(ids)) != len(ids):
		raise ValueError('layout IDs must be unique')
	if set(ids) != set(LAYOUT_IDS):
		raise ValueError(f'layout IDs must be exactly {list(LAYOUT_IDS)!r}')
	by_id = {layout.layout_id: layout for layout in layouts}
	return tuple(by_id[layout_id] for layout_id in LAYOUT_IDS)


def _resolve_layout(value: object, *, index: int) -> F3SectionLayoutSpec:
	if not isinstance(value, Mapping):
		raise TypeError(f'layouts[{index}] must be a mapping; got {value!r}')
	prefix = f'layouts[{index}]'
	_validate_allowed_keys(
		value, frozenset({'layout_id', 'sizes'}), prefix=prefix
	)
	layout_id = _required_str(value, 'layout_id', prefix=prefix)
	raw_sizes = value.get('sizes')
	if not isinstance(raw_sizes, Mapping):
		raise TypeError(f'{prefix}.sizes must be a mapping')
	_validate_allowed_keys(raw_sizes, frozenset(DATA_SIZES), prefix=f'{prefix}.sizes')
	if set(raw_sizes) != set(DATA_SIZES):
		raise ValueError(
			f'{prefix}.sizes must define exactly {list(DATA_SIZES)!r}'
		)
	sizes = tuple(
		_resolve_size(raw_sizes[data_size], data_size=data_size, prefix=prefix)
		for data_size in DATA_SIZES
	)
	_validate_nesting(layout_id, sizes)
	return F3SectionLayoutSpec(layout_id=layout_id, sizes=sizes)


def _resolve_size(
	value: object, *, data_size: str, prefix: str
) -> F3SectionLayoutSizeSpec:
	if not isinstance(value, Mapping):
		raise TypeError(f'{prefix}.sizes.{data_size} must be a mapping')
	label = f'{prefix}.sizes.{data_size}'
	_validate_allowed_keys(
		value,
		frozenset(
			{'inline_lines', 'crossline_lines', 'target_train_voxel_count'}
		),
		prefix=label,
	)
	inlines = _line_numbers(value.get('inline_lines'), f'{label}.inline_lines')
	crosslines = _line_numbers(
		value.get('crossline_lines'), f'{label}.crossline_lines'
	)
	expected_inline_count, expected_crossline_count = LINE_COUNTS[data_size]
	if (
		len(inlines) != expected_inline_count
		or len(crosslines) != expected_crossline_count
	):
		raise ValueError(
			f'{label} must contain exactly {expected_inline_count} inline and '
			f'{expected_crossline_count} crossline lines'
		)
	target = _positive_integer(
		value.get('target_train_voxel_count'),
		f'{label}.target_train_voxel_count',
	)
	return F3SectionLayoutSizeSpec(
		data_size=data_size,
		inline_lines=inlines,
		crossline_lines=crosslines,
		target_train_voxel_count=target,
	)


def _validate_nesting(
	layout_id: str, sizes: tuple[F3SectionLayoutSizeSpec, ...]
) -> None:
	by_name = {size.data_size: size for size in sizes}
	for axis in ('inline_lines', 'crossline_lines'):
		small = set(getattr(by_name['small'], axis))
		medium = set(getattr(by_name['medium'], axis))
		large = set(getattr(by_name['large'], axis))
		if not small < medium < large:
			raise ValueError(
				f'{layout_id} {axis} must be strictly nested small < medium < large'
			)


def _validate_target_counts(layouts: tuple[F3SectionLayoutSpec, ...]) -> None:
	for data_size in DATA_SIZES:
		counts = {
			layout.size_by_name[data_size].target_train_voxel_count
			for layout in layouts
		}
		if len(counts) != 1:
			raise ValueError(
				f'{data_size} target_train_voxel_count must be identical across layouts'
			)


def _validate_validation_disjoint(
	layouts: tuple[F3SectionLayoutSpec, ...],
	*,
	line_inventory: Sequence[Mapping[str, object]]
	| Mapping[str, Sequence[int]]
	| None,
) -> None:
	if line_inventory is None:
		return
	validation_lines = _validation_lines(line_inventory)
	for layout in layouts:
		for size in layout.sizes:
			for slice_type, selected in (
				('inline', size.inline_lines),
				('crossline', size.crossline_lines),
			):
				overlap = sorted(set(selected) & validation_lines[slice_type])
				if overlap:
					raise ValueError(
						f'{layout.layout_id}/{size.data_size} selects validation '
						f'{slice_type} lines {overlap!r}'
					)


def _validation_lines(
	line_inventory: Sequence[Mapping[str, object]]
	| Mapping[str, Sequence[int]],
) -> Mapping[str, set[int]]:
	if isinstance(line_inventory, Mapping):
		_validate_allowed_keys(
			line_inventory,
			frozenset({'inline', 'crossline'}),
			prefix='line_inventory',
		)
		if set(line_inventory) != {'inline', 'crossline'}:
			raise ValueError(
				'line_inventory must define exactly inline and crossline'
			)
		return {
			slice_type: set(
				_line_numbers(
					line_inventory[slice_type],
					f'line_inventory.{slice_type}',
				)
			)
			for slice_type in ('inline', 'crossline')
		}
	if isinstance(line_inventory, str | bytes) or not isinstance(
		line_inventory, Sequence
	):
		raise TypeError('line_inventory must be a mapping or list of mappings')
	result = {'inline': set(), 'crossline': set()}
	for index, row in enumerate(line_inventory):
		if not isinstance(row, Mapping):
			raise TypeError(f'line_inventory[{index}] must be a mapping')
		if row.get('split') != 'validation':
			continue
		slice_type = row.get('slice_type')
		if slice_type not in result:
			raise ValueError(
				f'line_inventory[{index}].slice_type must be inline or crossline'
			)
		slice_index = _positive_integer(
			row.get('slice_index'), f'line_inventory[{index}].slice_index'
		)
		result[slice_type].add(slice_index)
	return result


def _resolve_fixed_decoder(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('decoder must be a mapping')
	_validate_allowed_keys(value, frozenset(FIXED_DECODER_CONTRACT), prefix='decoder')
	if set(value) != set(FIXED_DECODER_CONTRACT):
		raise ValueError('decoder must define every field in the fixed contract')
	resolved: dict[str, object] = {}
	for key in (
		'spec',
		'upsample_mode',
		'normalization',
		'class_weight',
		'sampling_mode',
	):
		resolved[key] = _required_str(value, key, prefix='decoder')
	for key in (
		'embedding_dim',
		'class_count',
		'epochs',
		'batch_size',
		'steps_per_epoch',
		'seed',
	):
		resolved[key] = _positive_integer(value.get(key), f'decoder.{key}')
	resolved['hidden_channels'] = _integer_tuple(
		value.get('hidden_channels'), 'decoder.hidden_channels'
	)
	resolved['upsample_factors'] = _nested_integer_tuple(
		value.get('upsample_factors'), 'decoder.upsample_factors'
	)
	for key in ('learning_rate', 'weight_decay', 'gradient_clip_norm'):
		resolved[key] = _number(value.get(key), f'decoder.{key}')
	for key in ('amp', 'write_probabilities'):
		item = value.get(key)
		if not isinstance(item, bool):
			raise TypeError(f'decoder.{key} must be a boolean; got {item!r}')
		resolved[key] = item
	if resolved != FIXED_DECODER_CONTRACT:
		raise ValueError('decoder must match the fixed section-layout decoder contract')
	return FIXED_DECODER_CONTRACT


def _exact_string(
	parent: Mapping[str, object], key: str, expected: str
) -> str:
	value = _required_str(parent, key, prefix='config')
	if value != expected:
		raise ValueError(f'{key} must be exactly {expected!r}')
	return value


def _line_numbers(value: object, label: str) -> tuple[int, ...]:
	items = _integer_tuple(value, label)
	if any(item <= 0 for item in items):
		raise ValueError(f'{label} entries must be positive integers')
	if len(set(items)) != len(items):
		raise ValueError(f'{label} must not contain duplicate lines')
	return items


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a list of integers')
	return tuple(_integer(item, f'{label} entry') for item in value)


def _nested_integer_tuple(
	value: object, label: str
) -> tuple[tuple[int, ...], ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a list of integer lists')
	return tuple(
		_integer_tuple(item, f'{label} entry') for item in value
	)


def _integer(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{label} must be an integer; got {value!r}')
	return value


def _positive_integer(value: object, label: str) -> int:
	result = _integer(value, label)
	if result <= 0:
		raise ValueError(f'{label} must be a positive integer; got {value!r}')
	return result


def _number(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{label} must be a number; got {value!r}')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label} must be finite; got {value!r}')
	return result


def _relative_error(value: object) -> float:
	result = _number(value, 'allowed_relative_error')
	if not 0.0 < result <= 0.1:
		raise ValueError('allowed_relative_error must be in (0, 0.1]')
	return result


__all__ = [
	'CONTRACT_SCHEMA_VERSION',
	'DATA_SIZES',
	'DECODER_SEED',
	'FIXED_DECODER_CONTRACT',
	'LAYOUT_IDS',
	'LINE_COUNTS',
	'NESTING_SEMANTICS',
	'PATCH_SIZE',
	'STABLE_SELECTION_SEMANTICS',
	'STATISTICAL_UNIT',
	'VALIDATION_MASK_SEMANTICS',
	'F3SectionLayoutContract',
	'F3SectionLayoutSizeSpec',
	'F3SectionLayoutSpec',
	'f3_lithology_voxel_section_layout_config_from_mapping',
	'f3_lithology_voxel_section_layout_contract_from_mapping',
	'f3_section_layout_contract_from_mapping',
	'resolve_f3_lithology_voxel_section_layout_contract',
]
