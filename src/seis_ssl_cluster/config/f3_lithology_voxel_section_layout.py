"""Closed section-layout contract for the F3 voxel benchmark."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

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
STABLE_SELECTION_SEMANTICS = 'stable_hash_partial_section_token_footprints_v1'
CLASS_BALANCED_SELECTION_SEMANTICS = (
	'seeded_nested_class_balanced_section_token_rows_v1'
)
CONTRACT_ARTIFACT_TYPE = 'f3_lithology_voxel_section_layout_contract'
# Existing contracts use the largest common target reachable by all layouts.
# Versioned experiment configs may instead provide fixed reachable targets.
TARGET_CALIBRATION_RULE = 'max_common_reachable_active_pool_v1'
FIXED_TRAIN_VOXEL_COUNTS_RULE = 'fixed_train_voxel_counts_v1'
FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE = (
	'fixed_per_class_token_row_caps_v1'  # noqa: S105
)
PATCH_SIZE = (8, 8, 8)
DECODER_SEED = 42000
TOKEN_ROW_VALIDATION_PRECEDENCE = (
	'validation_token_xyz_remove_all_train_rows_v1'  # noqa: S105
)
TOKEN_ROW_VOXEL_MATERIALIZATION = (
	'partial_active_section_token_footprints_v1'  # noqa: S105
)
TOKENIZATION_POLICY: Mapping[str, object] = MappingProxyType({
	'min_labeled_fraction': 0.5,
	'min_majority_fraction': 0.7,
	'ignore_z_border_samples': 1,
})

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
	subsample_seed: int | None = None
	per_class_token_row_cap: int | None = None
	selected_token_row_count: int | None = None
	selected_token_row_identity_sha256: str | None = None
	per_class_selected_token_row_counts: Mapping[str, int] | None = None
	active_pool_per_class_token_row_counts: Mapping[str, int] | None = None
	per_line_selected_token_row_counts: Mapping[str, int] | None = None


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
	selection_semantics: str
	stable_selection_semantics: str | None
	class_balanced_selection: Mapping[str, object] | None
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
				'artifact_type',
				'schema_version',
				'selection_semantics',
				'statistical_unit',
				'nesting_semantics',
				'validation_mask_semantics',
				'stable_selection_semantics',
				'patch_size',
				'patch_size_xyz',
				'allowed_relative_error',
				'target_train_voxel_counts',
				'target_calibration',
				'active_prefix_counts',
				'decoder_seed',
				'layouts',
				'decoder',
				'validation_identity',
				'source_file_identities',
				'legacy_budget_source_identity',
				'class_balanced_selection',
			}
		),
		prefix='config',
	)
	if 'artifact_type' in config:
		_exact_string(config, 'artifact_type', CONTRACT_ARTIFACT_TYPE)
	_exact_string(config, 'schema_version', CONTRACT_SCHEMA_VERSION)
	statistical_unit = _exact_string(
		config, 'statistical_unit', STATISTICAL_UNIT
	)
	nesting = _exact_string(config, 'nesting_semantics', NESTING_SEMANTICS)
	validation_mask = _exact_string(
		config, 'validation_mask_semantics', VALIDATION_MASK_SEMANTICS
	)
	selection_semantics = _selection_semantics(config)
	stable_selection: str | None
	class_balanced_selection: Mapping[str, object] | None
	if selection_semantics == STABLE_SELECTION_SEMANTICS:
		stable_selection = _exact_string(
			config,
			'stable_selection_semantics',
			STABLE_SELECTION_SEMANTICS,
		)
		if 'class_balanced_selection' in config:
			raise ValueError(
				'class_balanced_selection is only valid for '
				f'{CLASS_BALANCED_SELECTION_SEMANTICS!r}'
			)
		class_balanced_selection = None
	else:
		if 'stable_selection_semantics' in config:
			raise ValueError(
				'stable_selection_semantics must not be specified for '
				f'{CLASS_BALANCED_SELECTION_SEMANTICS!r}'
			)
		stable_selection = None
		class_balanced_selection = _class_balanced_selection(
			config.get('class_balanced_selection')
		)
	patch_size = _integer_tuple(config.get('patch_size'), 'patch_size')
	if patch_size != PATCH_SIZE:
		raise ValueError(f'patch_size must be exactly {list(PATCH_SIZE)!r}')
	if 'patch_size_xyz' in config:
		patch_size_xyz = _integer_tuple(config.get('patch_size_xyz'), 'patch_size_xyz')
		if patch_size_xyz != PATCH_SIZE:
			raise ValueError(f'patch_size_xyz must be exactly {list(PATCH_SIZE)!r}')
	allowed_relative_error = _relative_error(config.get('allowed_relative_error'))
	decoder_seed = _integer(config.get('decoder_seed'), 'decoder_seed')
	if decoder_seed != DECODER_SEED:
		raise ValueError(f'decoder_seed must be exactly {DECODER_SEED}')
	layouts = _resolve_layouts(
		config.get('layouts'), selection_semantics=selection_semantics
	)
	_validate_target_counts(layouts)
	_validate_generated_summary(config, layouts)
	_validate_target_calibration(
		config,
		layouts,
		selection_semantics=selection_semantics,
		class_balanced_selection=class_balanced_selection,
	)
	_validate_validation_disjoint(layouts, line_inventory=line_inventory)
	decoder = _resolve_fixed_decoder(config.get('decoder'))
	return F3SectionLayoutContract(
		layouts=layouts,
		patch_size=PATCH_SIZE,
		allowed_relative_error=allowed_relative_error,
		statistical_unit=statistical_unit,
		nesting_semantics=nesting,
		validation_mask_semantics=validation_mask,
		selection_semantics=selection_semantics,
		stable_selection_semantics=stable_selection,
		class_balanced_selection=class_balanced_selection,
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


def _resolve_layouts(
	value: object, *, selection_semantics: str
) -> tuple[F3SectionLayoutSpec, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError('layouts must be a list of layout mappings')
	if len(value) != len(LAYOUT_IDS):
		raise ValueError(f'layouts must contain exactly {len(LAYOUT_IDS)} entries')
	layouts = tuple(
		_resolve_layout(
			item, index=index, selection_semantics=selection_semantics
		)
		for index, item in enumerate(value)
	)
	ids = tuple(layout.layout_id for layout in layouts)
	if len(set(ids)) != len(ids):
		raise ValueError('layout IDs must be unique')
	if set(ids) != set(LAYOUT_IDS):
		raise ValueError(f'layout IDs must be exactly {list(LAYOUT_IDS)!r}')
	by_id = {layout.layout_id: layout for layout in layouts}
	return tuple(by_id[layout_id] for layout_id in LAYOUT_IDS)


def _resolve_layout(
	value: object, *, index: int, selection_semantics: str
) -> F3SectionLayoutSpec:
	if not isinstance(value, Mapping):
		raise TypeError(f'layouts[{index}] must be a mapping; got {value!r}')
	prefix = f'layouts[{index}]'
	_validate_allowed_keys(
		value,
		frozenset({'layout_id', 'ordered_inlines', 'ordered_crosslines', 'sizes'}),
		prefix=prefix,
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
		_resolve_size(
			raw_sizes[data_size],
			data_size=data_size,
			prefix=prefix,
			selection_semantics=selection_semantics,
		)
		for data_size in DATA_SIZES
	)
	_validate_nesting(layout_id, sizes)
	if 'ordered_inlines' in value:
		ordered = _line_numbers(
			value.get('ordered_inlines'), f'{prefix}.ordered_inlines'
		)
		if ordered != sizes[-1].inline_lines:
			raise ValueError(f'{prefix}.ordered_inlines must match large inline_lines')
	if 'ordered_crosslines' in value:
		ordered = _line_numbers(
			value.get('ordered_crosslines'), f'{prefix}.ordered_crosslines'
		)
		if ordered != sizes[-1].crossline_lines:
			raise ValueError(
				f'{prefix}.ordered_crosslines must match large crossline_lines'
			)
	return F3SectionLayoutSpec(layout_id=layout_id, sizes=sizes)


def _resolve_size(
	value: object, *, data_size: str, prefix: str, selection_semantics: str
) -> F3SectionLayoutSizeSpec:
	if not isinstance(value, Mapping):
		raise TypeError(f'{prefix}.sizes.{data_size} must be a mapping')
	label = f'{prefix}.sizes.{data_size}'
	allowed = {
		'inline_lines',
		'crossline_lines',
		'target_train_voxel_count',
		'preview_actual_train_voxel_count',
		'preview_count_error',
		'preview_relative_count_error',
		'selected_token_xyz',
		'per_line_contributions',
		'per_class_voxel_counts',
	}
	if selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
		allowed.update({
			'subsample_seed',
			'per_class_token_row_cap',
			'selected_token_row_count',
			'selected_token_row_identity_sha256',
			'per_class_selected_token_row_counts',
			'active_pool_per_class_token_row_counts',
			'per_line_selected_token_row_counts',
		})
	_validate_allowed_keys(
		value,
		frozenset(allowed),
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
	class_cap_fields: dict[str, object] = {}
	if selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
		class_cap_fields = _resolve_class_cap_size_fields(
			value,
			label=label,
			inline_lines=inlines,
			crossline_lines=crosslines,
		)
	return F3SectionLayoutSizeSpec(
		data_size=data_size,
		inline_lines=inlines,
		crossline_lines=crosslines,
		target_train_voxel_count=target,
		**class_cap_fields,
	)


def _selection_semantics(config: Mapping[str, object]) -> str:
	if 'selection_semantics' not in config:
		return STABLE_SELECTION_SEMANTICS
	value = _required_str(config, 'selection_semantics', prefix='config')
	if value not in {
		STABLE_SELECTION_SEMANTICS,
		CLASS_BALANCED_SELECTION_SEMANTICS,
	}:
		raise ValueError(
			'selection_semantics must be one of '
			f'{[STABLE_SELECTION_SEMANTICS, CLASS_BALANCED_SELECTION_SEMANTICS]!r}'
		)
	return value


def _class_balanced_selection(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('class_balanced_selection must be a mapping')
	_validate_allowed_keys(
		value,
		frozenset({
			'per_class_token_row_caps',
			'layout_subsample_seeds',
			'tokenization_policy',
			'validation_precedence',
			'voxel_materialization',
			'token_row_pool_provenance',
		}),
		prefix='class_balanced_selection',
	)
	if set(value) != {
		'per_class_token_row_caps',
		'layout_subsample_seeds',
		'tokenization_policy',
		'validation_precedence',
		'voxel_materialization',
		'token_row_pool_provenance',
	}:
		raise ValueError('class_balanced_selection must define every field')
	caps = _size_counts(
		value.get('per_class_token_row_caps'),
		label='class_balanced_selection.per_class_token_row_caps',
	)
	seeds = _layout_seeds(
		value.get('layout_subsample_seeds'),
		label='class_balanced_selection.layout_subsample_seeds',
	)
	policy = _tokenization_policy(
		value.get('tokenization_policy'),
		label='class_balanced_selection.tokenization_policy',
	)
	validation = value.get('validation_precedence')
	if validation != TOKEN_ROW_VALIDATION_PRECEDENCE:
		raise ValueError(
			'class_balanced_selection.validation_precedence must be exactly '
			f'{TOKEN_ROW_VALIDATION_PRECEDENCE!r}'
		)
	materialization = value.get('voxel_materialization')
	if materialization != TOKEN_ROW_VOXEL_MATERIALIZATION:
		raise ValueError(
			'class_balanced_selection.voxel_materialization must be exactly '
			f'{TOKEN_ROW_VOXEL_MATERIALIZATION!r}'
		)
	pool_provenance = _token_row_pool_provenance(
		value.get('token_row_pool_provenance')
	)
	return MappingProxyType({
		'per_class_token_row_caps': MappingProxyType(caps),
		'layout_subsample_seeds': MappingProxyType(seeds),
		'tokenization_policy': MappingProxyType(policy),
		'validation_precedence': TOKEN_ROW_VALIDATION_PRECEDENCE,
		'voxel_materialization': TOKEN_ROW_VOXEL_MATERIALIZATION,
		'token_row_pool_provenance': MappingProxyType(pool_provenance),
	})


def _token_row_pool_provenance(value: object) -> dict[str, object]:
	label = 'class_balanced_selection.token_row_pool_provenance'
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	keys = {
		'train_token_row_count',
		'train_row_count_before_validation_precedence',
		'train_rows_removed_by_validation_precedence',
		'validation_token_xyz_count',
		'validation_token_xyz_sha256',
	}
	_validate_allowed_keys(value, frozenset(keys), prefix=label)
	if set(value) != keys:
		raise ValueError(f'{label} must define every field')
	train_count = _positive_integer(
		value.get('train_token_row_count'), f'{label}.train_token_row_count'
	)
	before_count = _positive_integer(
		value.get('train_row_count_before_validation_precedence'),
		f'{label}.train_row_count_before_validation_precedence',
	)
	removed_count = _nonnegative_integer(
		value.get('train_rows_removed_by_validation_precedence'),
		f'{label}.train_rows_removed_by_validation_precedence',
	)
	if before_count - removed_count != train_count:
		raise ValueError(
			f'{label} train-row counts do not satisfy before - removed = retained'
		)
	return {
		'train_token_row_count': train_count,
		'train_row_count_before_validation_precedence': before_count,
		'train_rows_removed_by_validation_precedence': removed_count,
		'validation_token_xyz_count': _positive_integer(
			value.get('validation_token_xyz_count'),
			f'{label}.validation_token_xyz_count',
		),
		'validation_token_xyz_sha256': _sha256_string(
			value.get('validation_token_xyz_sha256'),
			f'{label}.validation_token_xyz_sha256',
		),
	}


def _resolve_class_cap_size_fields(
	value: Mapping[str, object],
	*,
	label: str,
	inline_lines: tuple[int, ...],
	crossline_lines: tuple[int, ...],
) -> dict[str, object]:
	required = {
		'subsample_seed',
		'per_class_token_row_cap',
		'selected_token_row_count',
		'selected_token_row_identity_sha256',
		'per_class_selected_token_row_counts',
		'active_pool_per_class_token_row_counts',
		'per_line_selected_token_row_counts',
	}
	missing = sorted(required - set(value))
	if missing:
		raise ValueError(f'{label} must define class-cap fields; missing={missing!r}')
	cap = _positive_integer(
		value.get('per_class_token_row_cap'), f'{label}.per_class_token_row_cap'
	)
	selected_by_class = _class_counts(
		value.get('per_class_selected_token_row_counts'),
		label=f'{label}.per_class_selected_token_row_counts',
	)
	if any(count != cap for count in selected_by_class.values()):
		raise ValueError(
			f'{label} must select exactly the cap from every class'  # noqa: S608
		)
	selected_count = _positive_integer(
		value.get('selected_token_row_count'), f'{label}.selected_token_row_count'
	)
	if selected_count != len(selected_by_class) * cap:
		raise ValueError(
			f'{label}.selected_token_row_count must equal six times the class cap'
		)
	active_pool = _class_counts(
		value.get('active_pool_per_class_token_row_counts'),
		label=f'{label}.active_pool_per_class_token_row_counts',
	)
	if any(active_pool[key] < cap for key in active_pool):
		raise ValueError(f'{label} class-cap target exceeds an active token-row pool')
	line_counts = _positive_string_mapping(
		value.get('per_line_selected_token_row_counts'),
		label=f'{label}.per_line_selected_token_row_counts',
	)
	expected_line_keys = {
		*(f'inline:{line}' for line in inline_lines),
		*(f'crossline:{line}' for line in crossline_lines),
	}
	if set(line_counts) != expected_line_keys:
		raise ValueError(
			f'{label}.per_line_selected_token_row_counts must define exactly '
			'the active lines'
		)
	if sum(line_counts.values()) != selected_count:
		raise ValueError(f'{label} per-line selected rows must sum to selected rows')
	return {
		'subsample_seed': _nonnegative_integer(
			value.get('subsample_seed'), f'{label}.subsample_seed'
		),
		'per_class_token_row_cap': cap,
		'selected_token_row_count': selected_count,
		'selected_token_row_identity_sha256': _sha256_string(
			value.get('selected_token_row_identity_sha256'),
			f'{label}.selected_token_row_identity_sha256',
		),
		'per_class_selected_token_row_counts': MappingProxyType(selected_by_class),
		'active_pool_per_class_token_row_counts': MappingProxyType(active_pool),
		'per_line_selected_token_row_counts': MappingProxyType(line_counts),
	}


def _validate_generated_summary(
	config: Mapping[str, object], layouts: tuple[F3SectionLayoutSpec, ...]
) -> None:
	if 'target_train_voxel_counts' in config:
		targets = config.get('target_train_voxel_counts')
		if not isinstance(targets, Mapping) or set(targets) != set(DATA_SIZES):
			raise ValueError('target_train_voxel_counts must define exactly all sizes')
		for size in DATA_SIZES:
			expected = layouts[0].size_by_name[size].target_train_voxel_count
			if (
				_positive_integer(targets[size], f'target_train_voxel_counts.{size}')
				!= expected
			):
				raise ValueError(
					f'target_train_voxel_counts.{size} does not match layouts'
				)
	if 'active_prefix_counts' in config:
		prefixes = config.get('active_prefix_counts')
		if not isinstance(prefixes, Mapping) or set(prefixes) != set(DATA_SIZES):
			raise ValueError('active_prefix_counts must define exactly all sizes')
		for size in DATA_SIZES:
			value = prefixes[size]
			if not isinstance(value, Mapping) or set(value) != {'inline', 'crossline'}:
				raise ValueError(
					f'active_prefix_counts.{size} must define inline and crossline'
				)
			expected = LINE_COUNTS[size]
			actual = (
				_integer(value['inline'], f'active_prefix_counts.{size}.inline'),
				_integer(value['crossline'], f'active_prefix_counts.{size}.crossline'),
			)
			if actual != expected:
				raise ValueError(
					f'active_prefix_counts.{size} must be exactly {expected!r}'
				)


def _validate_target_calibration(
	config: Mapping[str, object],
	layouts: tuple[F3SectionLayoutSpec, ...],
	*,
	selection_semantics: str,
	class_balanced_selection: Mapping[str, object] | None,
) -> None:
	if 'target_calibration' not in config:
		if selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
			raise ValueError(
				'target_calibration is required for class-balanced selection'
			)
		return
	calibration = config.get('target_calibration')
	if not isinstance(calibration, Mapping):
		raise TypeError('target_calibration must be a mapping')
	rule = _required_str(calibration, 'rule', prefix='target_calibration')
	if selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
		if rule != FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE:
			raise ValueError(
				'target_calibration.rule must be exactly '
				f'{FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE!r} for class-balanced selection'
			)
		_validate_class_cap_targets(
			config,
			layouts,
			calibration=calibration,
			selection=class_balanced_selection,
		)
		return
	if rule == FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE:
		raise ValueError(
			f'{FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE!r} requires '
			f'{CLASS_BALANCED_SELECTION_SEMANTICS!r}'
		)
	if rule == TARGET_CALIBRATION_RULE:
		_validate_allowed_keys(
			calibration,
			frozenset({'rule', 'active_pool_train_voxel_counts'}),
			prefix='target_calibration',
		)
		_validate_max_common_targets(
			layouts, active_pools=_resolve_active_pool_counts(calibration)
		)
		return
	if rule == FIXED_TRAIN_VOXEL_COUNTS_RULE:
		_validate_allowed_keys(
			calibration,
			frozenset(
				{
					'rule',
					'fixed_target_train_voxel_counts',
					'active_pool_train_voxel_counts',
				}
			),
			prefix='target_calibration',
		)
		_validate_fixed_targets(
			config,
			layouts,
			calibration=calibration,
			active_pools=_resolve_active_pool_counts(calibration),
		)
		return
	raise ValueError(
		'target_calibration.rule must be one of '
		f'{[TARGET_CALIBRATION_RULE, FIXED_TRAIN_VOXEL_COUNTS_RULE, FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE]!r}'  # noqa: E501
	)


def _resolve_active_pool_counts(
	calibration: Mapping[str, object],
) -> dict[str, dict[str, int]]:
	pools = calibration.get('active_pool_train_voxel_counts')
	if not isinstance(pools, Mapping) or set(pools) != set(DATA_SIZES):
		raise ValueError(
			'target_calibration.active_pool_train_voxel_counts must define '
			'exactly all sizes'
		)
	resolved_pools: dict[str, dict[str, int]] = {}
	for size in DATA_SIZES:
		by_layout = pools[size]
		if not isinstance(by_layout, Mapping) or set(by_layout) != set(LAYOUT_IDS):
			raise ValueError(
				f'target_calibration.active_pool_train_voxel_counts.{size} must '
				'define exactly all layouts'
			)
		resolved_pools[size] = {
			layout_id: _positive_integer(
				by_layout[layout_id],
				f'target_calibration.active_pool_train_voxel_counts.{size}.'
				f'{layout_id}',
			)
			for layout_id in LAYOUT_IDS
		}
	return resolved_pools


def _validate_max_common_targets(
	layouts: tuple[F3SectionLayoutSpec, ...],
	*,
	active_pools: Mapping[str, Mapping[str, int]],
) -> None:
	for size in DATA_SIZES:
		common_target = min(active_pools[size].values())
		# Stored targets must replay the rule from the stored active pools.
		for layout in layouts:
			if layout.size_by_name[size].target_train_voxel_count != common_target:
				raise ValueError(
					f'{layout.layout_id}/{size} target must equal the minimum '
					f'active pool {common_target} under {TARGET_CALIBRATION_RULE!r}'
				)


def _validate_fixed_targets(
	config: Mapping[str, object],
	layouts: tuple[F3SectionLayoutSpec, ...],
	*,
	calibration: Mapping[str, object],
	active_pools: Mapping[str, Mapping[str, int]],
) -> None:
	fixed = calibration.get('fixed_target_train_voxel_counts')
	if not isinstance(fixed, Mapping) or set(fixed) != set(DATA_SIZES):
		raise ValueError(
			'target_calibration.fixed_target_train_voxel_counts must define '
			'exactly all sizes'
		)
	fixed_targets = {
		size: _positive_integer(
			fixed[size],
			f'target_calibration.fixed_target_train_voxel_counts.{size}',
		)
		for size in DATA_SIZES
	}
	if not (
		fixed_targets['small']
		< fixed_targets['medium']
		< fixed_targets['large']
	):
		raise ValueError(
			'target_calibration.fixed_target_train_voxel_counts must strictly '
			'increase small < medium < large'
		)
	top_level = config.get('target_train_voxel_counts')
	if not isinstance(top_level, Mapping) or set(top_level) != set(DATA_SIZES):
		raise ValueError(
			'fixed target calibration requires top-level '
			'target_train_voxel_counts for exactly all sizes'
		)
	for size in DATA_SIZES:
		top_level_target = _positive_integer(
			top_level[size], f'target_train_voxel_counts.{size}'
		)
		if top_level_target != fixed_targets[size]:
			raise ValueError(
				f'target_train_voxel_counts.{size} must match fixed target '
				f'{fixed_targets[size]}'
			)
		for layout in layouts:
			layout_target = layout.size_by_name[size].target_train_voxel_count
			if layout_target != fixed_targets[size]:
				raise ValueError(
					f'{layout.layout_id}/{size} target must match fixed target '
					f'{fixed_targets[size]}'
				)
			pool = active_pools[size][layout.layout_id]
			if fixed_targets[size] > pool:
				raise ValueError(
					f'{layout.layout_id}/{size} fixed target '
					f'{fixed_targets[size]} exceeds active pool {pool}'
				)


def _validate_class_cap_targets(  # noqa: C901, PLR0912
	config: Mapping[str, object],
	layouts: tuple[F3SectionLayoutSpec, ...],
	*,
	calibration: Mapping[str, object],
	selection: Mapping[str, object] | None,
) -> None:
	if selection is None:
		raise AssertionError('class-balanced selection metadata was not resolved')
	_validate_allowed_keys(
		calibration,
		frozenset({
			'rule',
			'nominal_target_train_voxel_counts',
			'active_pool_token_row_counts',
		}),
		prefix='target_calibration',
	)
	if set(calibration) != {
		'rule',
		'nominal_target_train_voxel_counts',
		'active_pool_token_row_counts',
	}:
		raise ValueError('class-cap target_calibration must define every field')
	nominal = _size_counts(
		calibration.get('nominal_target_train_voxel_counts'),
		label='target_calibration.nominal_target_train_voxel_counts',
	)
	top_level = _size_counts(
		config.get('target_train_voxel_counts'),
		label='target_train_voxel_counts',
	)
	if nominal != top_level:
		raise ValueError('top-level targets must match nominal class-cap targets')
	caps = dict(cast('Mapping[str, int]', selection['per_class_token_row_caps']))
	for size in DATA_SIZES:
		expected = len(range(6)) * caps[size] * PATCH_SIZE[1] * PATCH_SIZE[2]
		if nominal[size] != expected:
			raise ValueError(
				f'{size} nominal target must equal six classes * cap * 8 * 8 '
				f'({expected})'
			)
	pools = _active_token_row_pools(
		calibration.get('active_pool_token_row_counts')
	)
	seeds = cast('Mapping[str, int]', selection['layout_subsample_seeds'])
	for layout in layouts:
		for size in DATA_SIZES:
			spec = layout.size_by_name[size]
			if spec.target_train_voxel_count != nominal[size]:
				raise ValueError(
					f'{layout.layout_id}/{size} target must match nominal target '
					f'{nominal[size]}'
				)
			if spec.per_class_token_row_cap != caps[size]:
				raise ValueError(
					f'{layout.layout_id}/{size} token-row cap drift'
				)
			if spec.subsample_seed != seeds[layout.layout_id]:
				raise ValueError(
					f'{layout.layout_id}/{size} subsample seed drift'
				)
			stored_pool = dict(spec.active_pool_per_class_token_row_counts or {})
			if stored_pool != pools[size][layout.layout_id]:
				raise ValueError(
					f'{layout.layout_id}/{size} active token-row pool drift'
				)
			for class_id, count in stored_pool.items():
				if count < caps[size]:
					raise ValueError(
						f'{layout.layout_id}/{size}/class {class_id} pool {count} '
						f'cannot reach cap {caps[size]}'
					)
			line_counts = dict(spec.per_line_selected_token_row_counts or {})
			if len(line_counts) != sum(LINE_COUNTS[size]):
				raise ValueError(
					f'{layout.layout_id}/{size} selected-row line inventory drift'
				)


def _active_token_row_pools(
	value: object,
) -> dict[str, dict[str, dict[str, int]]]:
	if not isinstance(value, Mapping) or set(value) != set(DATA_SIZES):
		raise ValueError(
			'target_calibration.active_pool_token_row_counts must define all sizes'
		)
	result: dict[str, dict[str, dict[str, int]]] = {}
	for size in DATA_SIZES:
		by_layout = value[size]
		if not isinstance(by_layout, Mapping) or set(by_layout) != set(LAYOUT_IDS):
			raise ValueError(
				'target_calibration.active_pool_token_row_counts.'
				f'{size} must define all layouts'
			)
		result[size] = {
			layout_id: _class_counts(
				by_layout[layout_id],
				label=(
					'target_calibration.active_pool_token_row_counts.'
					f'{size}.{layout_id}'
				),
			)
			for layout_id in LAYOUT_IDS
		}
	return result


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


def _size_counts(value: object, *, label: str) -> dict[str, int]:
	if not isinstance(value, Mapping) or set(value) != set(DATA_SIZES):
		raise ValueError(f'{label} must define exactly {list(DATA_SIZES)!r}')
	result = {
		size: _positive_integer(value[size], f'{label}.{size}')
		for size in DATA_SIZES
	}
	if not result['small'] < result['medium'] < result['large']:
		raise ValueError(f'{label} must strictly increase small < medium < large')
	return result


def _layout_seeds(value: object, *, label: str) -> dict[str, int]:
	if not isinstance(value, Mapping) or set(value) != set(LAYOUT_IDS):
		raise ValueError(f'{label} must define exactly {list(LAYOUT_IDS)!r}')
	result = {
		layout_id: _nonnegative_integer(value[layout_id], f'{label}.{layout_id}')
		for layout_id in LAYOUT_IDS
	}
	if len(set(result.values())) != len(result):
		raise ValueError(f'{label} must contain unique seeds')
	for layout_id, seed in result.items():
		if seed != int(layout_id.removeprefix('layout_')):
			raise ValueError(f'{label}.{layout_id} must equal its layout suffix')
	return result


def _tokenization_policy(value: object, *, label: str) -> dict[str, object]:
	if not isinstance(value, Mapping) or set(value) != set(TOKENIZATION_POLICY):
		raise ValueError(
			f'{label} must define exactly {list(TOKENIZATION_POLICY)!r}'
		)
	resolved = {
		'min_labeled_fraction': _number(
			value['min_labeled_fraction'], f'{label}.min_labeled_fraction'
		),
		'min_majority_fraction': _number(
			value['min_majority_fraction'], f'{label}.min_majority_fraction'
		),
		'ignore_z_border_samples': _nonnegative_integer(
			value['ignore_z_border_samples'], f'{label}.ignore_z_border_samples'
		),
	}
	if resolved != dict(TOKENIZATION_POLICY):
		raise ValueError(f'{label} must match the fixed v1 tokenization policy')
	return resolved


def _class_counts(value: object, *, label: str) -> dict[str, int]:
	expected = {str(class_id) for class_id in range(6)}
	if not isinstance(value, Mapping) or set(value) != expected:
		raise ValueError(f'{label} must define exactly classes 0 through 5')
	return {
		str(class_id): _positive_integer(
			value[str(class_id)], f'{label}.{class_id}'
		)
		for class_id in range(6)
	}


def _positive_string_mapping(value: object, *, label: str) -> dict[str, int]:
	if not isinstance(value, Mapping) or not value:
		raise ValueError(f'{label} must be a non-empty mapping')
	result: dict[str, int] = {}
	for key, count in value.items():
		if not isinstance(key, str) or not key:
			raise TypeError(f'{label} keys must be non-empty strings')
		result[key] = _positive_integer(count, f'{label}.{key}')
	return result


def _sha256_string(value: object, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 hex digest')
	return value


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


def _nonnegative_integer(value: object, label: str) -> int:
	result = _integer(value, label)
	if result < 0:
		raise ValueError(f'{label} must be a nonnegative integer; got {value!r}')
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
	'CLASS_BALANCED_SELECTION_SEMANTICS',
	'CONTRACT_ARTIFACT_TYPE',
	'CONTRACT_SCHEMA_VERSION',
	'DATA_SIZES',
	'DECODER_SEED',
	'FIXED_DECODER_CONTRACT',
	'FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE',
	'FIXED_TRAIN_VOXEL_COUNTS_RULE',
	'LAYOUT_IDS',
	'LINE_COUNTS',
	'NESTING_SEMANTICS',
	'PATCH_SIZE',
	'STABLE_SELECTION_SEMANTICS',
	'STATISTICAL_UNIT',
	'TARGET_CALIBRATION_RULE',
	'TOKENIZATION_POLICY',
	'TOKEN_ROW_VALIDATION_PRECEDENCE',
	'TOKEN_ROW_VOXEL_MATERIALIZATION',
	'VALIDATION_MASK_SEMANTICS',
	'F3SectionLayoutContract',
	'F3SectionLayoutSizeSpec',
	'F3SectionLayoutSpec',
	'f3_lithology_voxel_section_layout_config_from_mapping',
	'f3_lithology_voxel_section_layout_contract_from_mapping',
	'f3_section_layout_contract_from_mapping',
	'resolve_f3_lithology_voxel_section_layout_contract',
]
