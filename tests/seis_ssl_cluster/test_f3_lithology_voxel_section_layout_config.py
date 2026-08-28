
from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config import (
	f3_lithology_voxel_section_layout_contract_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	CLASS_BALANCED_SELECTION_SEMANTICS,
	CONTRACT_SCHEMA_VERSION,
	DECODER_SEED,
	FIXED_DECODER_CONTRACT,
	FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE,
	FIXED_TRAIN_VOXEL_COUNTS_RULE,
	NESTING_SEMANTICS,
	STABLE_SELECTION_SEMANTICS,
	TARGET_CALIBRATION_RULE,
	TOKEN_ROW_VALIDATION_PRECEDENCE,
	TOKEN_ROW_VOXEL_MATERIALIZATION,
	VALIDATION_MASK_SEMANTICS,
)


def test_exact_five_layout_three_size_contract_resolves() -> None:
	contract = f3_lithology_voxel_section_layout_contract_from_mapping(
		_contract_mapping()
	)

	assert tuple(contract.layout_by_id) == tuple(
		f'layout_{index:03d}' for index in range(5)
	)
	assert tuple(contract.layouts[0].size_by_name) == ('small', 'medium', 'large')
	assert [
		(
			len(size.inline_lines),
			len(size.crossline_lines),
			size.target_train_voxel_count,
		)
		for size in contract.layouts[0].sizes
	] == [(1, 1, 1000), (2, 2, 2000), (4, 4, 4000)]
	assert contract.patch_size == (8, 8, 8)
	assert contract.decoder_seed == DECODER_SEED
	assert contract.decoder == FIXED_DECODER_CONTRACT


@pytest.mark.parametrize(
	('mutation', 'match'),
	[
		(('layouts', 0, 'sizes', 'small', 'inline_lines'), 'exactly 1 inline'),
		(('layouts', 0, 'sizes', 'medium', 'crossline_lines'), 'exactly 2 inline'),
	],
)
def test_contract_rejects_line_count_drift(
	mutation: tuple[object, ...], match: str
) -> None:
	raw = _contract_mapping()
	value = raw
	for key in mutation[:-1]:
		value = value[key]
	value[mutation[-1]] = []

	with pytest.raises(ValueError, match=match):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_contract_rejects_duplicate_line() -> None:
	raw = _contract_mapping()
	raw['layouts'][0]['sizes']['medium']['inline_lines'] = [100, 100]

	with pytest.raises(ValueError, match='duplicate lines'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_contract_rejects_unknown_key_and_bool_as_int() -> None:
	unknown = _contract_mapping()
	unknown['layouts'][0]['sizes']['small']['cap'] = 25
	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_section_layout_contract_from_mapping(unknown)

	boolean = _contract_mapping()
	boolean['layouts'][0]['sizes']['small']['target_train_voxel_count'] = True
	with pytest.raises(TypeError, match='must be an integer'):
		f3_lithology_voxel_section_layout_contract_from_mapping(boolean)


def test_contract_rejects_validation_line_from_supplied_inventory() -> None:
	raw = _contract_mapping()
	inventory = [
		{'split': 'train', 'slice_type': 'inline', 'slice_index': 999},
		{'split': 'validation', 'slice_type': 'inline', 'slice_index': 100},
	]

	with pytest.raises(ValueError, match='selects validation inline'):
		f3_lithology_voxel_section_layout_contract_from_mapping(
			raw, line_inventory=inventory
		)


def test_contract_rejects_fixed_decoder_one_field_drift() -> None:
	raw = _contract_mapping()
	raw['decoder']['steps_per_epoch'] = 441

	with pytest.raises(ValueError, match='fixed section-layout decoder'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_contract_rejects_unsupported_selection_semantics() -> None:
	raw = _contract_mapping()
	raw['stable_selection_semantics'] = 'random_choice_v1'

	with pytest.raises(ValueError, match='stable_selection_semantics'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_resolves_exact_caps_seeds_and_nominal_targets(
) -> None:
	contract = f3_lithology_voxel_section_layout_contract_from_mapping(
		_class_balanced_contract_mapping()
	)

	assert contract.selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS
	assert contract.stable_selection_semantics is None
	assert contract.class_balanced_selection is not None
	assert dict(
		contract.class_balanced_selection['per_class_token_row_caps']
	) == {'small': 25, 'medium': 50, 'large': 100}
	assert dict(contract.class_balanced_selection['layout_subsample_seeds']) == {
		f'layout_{index:03d}': index for index in range(5)
	}
	assert [
		(
			size.target_train_voxel_count,
			size.per_class_token_row_cap,
			size.subsample_seed,
			size.selected_token_row_count,
		)
		for size in contract.layouts[3].sizes
	] == [
		(9_600, 25, 3, 150),
		(19_200, 50, 3, 300),
		(38_400, 100, 3, 600),
	]


@pytest.mark.parametrize(
	'field',
	[
		'per_class_token_row_caps',
		'layout_subsample_seeds',
		'tokenization_policy',
	],
)
def test_class_balanced_contract_rejects_missing_selection_field(
	field: str,
) -> None:
	raw = _class_balanced_contract_mapping()
	del raw['class_balanced_selection'][field]

	with pytest.raises(ValueError, match='must define every field'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_missing_tokenization_policy_field() -> None:
	raw = _class_balanced_contract_mapping()
	del raw['class_balanced_selection']['tokenization_policy'][
		'ignore_z_border_samples'
	]

	with pytest.raises(ValueError, match='must define exactly'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_unknown_cap_size() -> None:
	raw = _class_balanced_contract_mapping()
	raw['class_balanced_selection']['per_class_token_row_caps']['tiny'] = 1

	with pytest.raises(ValueError, match='must define exactly'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


@pytest.mark.parametrize(
	('caps', 'error', 'match'),
	[
		(
			{'small': True, 'medium': 50, 'large': 100},
			TypeError,
			'must be an integer',
		),
		(
			{'small': 0, 'medium': 50, 'large': 100},
			ValueError,
			'must be a positive integer',
		),
		(
			{'small': 25.5, 'medium': 50, 'large': 100},
			TypeError,
			'must be an integer',
		),
		(
			{'small': 25, 'medium': 25, 'large': 100},
			ValueError,
			'must strictly increase',
		),
	],
)
def test_class_balanced_contract_rejects_invalid_caps(
	caps: dict[str, object],
	error: type[Exception],
	match: str,
) -> None:
	raw = _class_balanced_contract_mapping()
	raw['class_balanced_selection']['per_class_token_row_caps'] = caps

	with pytest.raises(error, match=match):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


@pytest.mark.parametrize(
	('layout_id', 'seed', 'error', 'match'),
	[
		('layout_001', 0, ValueError, 'must contain unique seeds'),
		('layout_000', True, TypeError, 'must be an integer'),
	],
)
def test_class_balanced_contract_rejects_invalid_layout_seeds(
	layout_id: str,
	seed: object,
	error: type[Exception],
	match: str,
) -> None:
	raw = _class_balanced_contract_mapping()
	raw['class_balanced_selection']['layout_subsample_seeds'][layout_id] = seed

	with pytest.raises(error, match=match):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_nominal_target_formula_drift() -> None:
	raw = _class_balanced_contract_mapping()
	raw['target_train_voxel_counts']['small'] = 9_601
	raw['target_calibration']['nominal_target_train_voxel_counts']['small'] = 9_601
	for layout in raw['layouts']:
		layout['sizes']['small']['target_train_voxel_count'] = 9_601

	with pytest.raises(
		ValueError,
		match=r'small nominal target must equal six classes \* cap \* 8 \* 8',
	):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_active_pool_shortage() -> None:
	raw = _class_balanced_contract_mapping()
	raw['layouts'][0]['sizes']['small'][
		'active_pool_per_class_token_row_counts'
	]['5'] = 24
	raw['target_calibration']['active_pool_token_row_counts']['small'][
		'layout_000'
	]['5'] = 24

	with pytest.raises(ValueError, match='class-cap target exceeds an active'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_stored_cap_drift() -> None:
	raw = _class_balanced_contract_mapping()
	small = raw['layouts'][0]['sizes']['small']
	small['per_class_token_row_cap'] = 24
	small['per_class_selected_token_row_counts'] = _class_counts(24)
	small['selected_token_row_count'] = 144
	small['per_line_selected_token_row_counts'] = {
		'inline:100': 72,
		'crossline:200': 72,
	}

	with pytest.raises(ValueError, match='token-row cap drift'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_stored_seed_drift() -> None:
	raw = _class_balanced_contract_mapping()
	raw['layouts'][0]['sizes']['small']['subsample_seed'] = 1

	with pytest.raises(ValueError, match='subsample seed drift'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_stored_pool_drift() -> None:
	raw = _class_balanced_contract_mapping()
	raw['layouts'][0]['sizes']['small'][
		'active_pool_per_class_token_row_counts'
	]['0'] = 36

	with pytest.raises(ValueError, match='active token-row pool drift'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_selected_row_line_key_drift() -> None:
	raw = _class_balanced_contract_mapping()
	line_counts = raw['layouts'][0]['sizes']['small'][
		'per_line_selected_token_row_counts'
	]
	line_counts['inline:1000'] = line_counts.pop('inline:100')

	with pytest.raises(ValueError, match='must define exactly the active lines'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


@pytest.mark.parametrize('sha', ['0' * 63, 'G' * 64])
def test_class_balanced_contract_rejects_row_sha_format_drift(sha: str) -> None:
	raw = _class_balanced_contract_mapping()
	raw['layouts'][0]['sizes']['small'][
		'selected_token_row_identity_sha256'
	] = sha

	with pytest.raises(ValueError, match='lowercase SHA-256 hex digest'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


def test_class_balanced_contract_rejects_missing_row_sha() -> None:
	raw = _class_balanced_contract_mapping()
	del raw['layouts'][0]['sizes']['small'][
		'selected_token_row_identity_sha256'
	]

	with pytest.raises(ValueError, match='must define class-cap fields'):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


@pytest.mark.parametrize(
	('mutation', 'match'),
	[
		('missing', 'must define every field'),
		('unknown', 'not allowed'),
		('count_drift', 'before - removed = retained'),
	],
)
def test_class_balanced_contract_rejects_pool_provenance_drift(
	mutation: str,
	match: str,
) -> None:
	raw = _class_balanced_contract_mapping()
	provenance = raw['class_balanced_selection']['token_row_pool_provenance']
	if mutation == 'missing':
		del provenance['validation_token_xyz_sha256']
	elif mutation == 'unknown':
		provenance['extra'] = 1
	else:
		provenance['train_rows_removed_by_validation_precedence'] = 9

	with pytest.raises(ValueError, match=match):
		f3_lithology_voxel_section_layout_contract_from_mapping(raw)


@pytest.mark.parametrize(
	'rule',
	[TARGET_CALIBRATION_RULE, FIXED_TRAIN_VOXEL_COUNTS_RULE],
)
def test_class_balanced_support_preserves_stable_calibration_rules(
	rule: str,
) -> None:
	contract = f3_lithology_voxel_section_layout_contract_from_mapping(
		_stable_calibrated_contract_mapping(rule)
	)

	assert contract.selection_semantics == STABLE_SELECTION_SEMANTICS
	assert contract.stable_selection_semantics == STABLE_SELECTION_SEMANTICS
	assert contract.class_balanced_selection is None


def _class_balanced_contract_mapping() -> dict[str, object]:
	caps = {'small': 25, 'medium': 50, 'large': 100}
	targets = {'small': 9_600, 'medium': 19_200, 'large': 38_400}
	active_pools: dict[str, dict[str, dict[str, int]]] = {
		size: {} for size in caps
	}
	layouts = []
	for layout_index in range(5):
		layout_id = f'layout_{layout_index:03d}'
		inline = 100 + layout_index * 10
		crossline = 200 + layout_index * 10
		line_sets = {
			'small': ([inline], [crossline]),
			'medium': ([inline, inline + 1], [crossline, crossline + 1]),
			'large': (
				[inline + offset for offset in range(4)],
				[crossline + offset for offset in range(4)],
			),
		}
		sizes = {}
		for size_index, size in enumerate(('small', 'medium', 'large')):
			cap = caps[size]
			pool = _class_counts(cap + 10)
			active_pools[size][layout_id] = dict(pool)
			inlines, crosslines = line_sets[size]
			line_keys = [
				*(f'inline:{line}' for line in inlines),
				*(f'crossline:{line}' for line in crosslines),
			]
			selected_count = 6 * cap
			per_line_count = selected_count // len(line_keys)
			sizes[size] = {
				'inline_lines': inlines,
				'crossline_lines': crosslines,
				'target_train_voxel_count': targets[size],
				'subsample_seed': layout_index,
				'per_class_token_row_cap': cap,
				'selected_token_row_count': selected_count,
				'selected_token_row_identity_sha256': (
					f'{layout_index * 3 + size_index + 1:064x}'
				),
				'per_class_selected_token_row_counts': _class_counts(cap),
				'active_pool_per_class_token_row_counts': pool,
				'per_line_selected_token_row_counts': dict.fromkeys(
					line_keys, per_line_count
				),
			}
		layouts.append({'layout_id': layout_id, 'sizes': sizes})
	return {
		'schema_version': CONTRACT_SCHEMA_VERSION,
		'selection_semantics': CLASS_BALANCED_SELECTION_SEMANTICS,
		'statistical_unit': 'layout_id',
		'nesting_semantics': NESTING_SEMANTICS,
		'validation_mask_semantics': VALIDATION_MASK_SEMANTICS,
		'patch_size': [8, 8, 8],
		'allowed_relative_error': 0.05,
		'target_train_voxel_counts': targets,
		'target_calibration': {
			'rule': FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE,
			'nominal_target_train_voxel_counts': dict(targets),
			'active_pool_token_row_counts': active_pools,
		},
		'class_balanced_selection': {
			'per_class_token_row_caps': caps,
			'layout_subsample_seeds': {
				f'layout_{index:03d}': index for index in range(5)
			},
			'tokenization_policy': {
				'min_labeled_fraction': 0.5,
				'min_majority_fraction': 0.7,
				'ignore_z_border_samples': 1,
			},
			'validation_precedence': TOKEN_ROW_VALIDATION_PRECEDENCE,
			'voxel_materialization': TOKEN_ROW_VOXEL_MATERIALIZATION,
			'token_row_pool_provenance': {
				'train_token_row_count': 1_000,
				'train_row_count_before_validation_precedence': 1_010,
				'train_rows_removed_by_validation_precedence': 10,
				'validation_token_xyz_count': 20,
				'validation_token_xyz_sha256': 'a' * 64,
			},
		},
		'decoder_seed': DECODER_SEED,
		'layouts': layouts,
		'decoder': deepcopy(dict(FIXED_DECODER_CONTRACT)),
	}


def _class_counts(count: int) -> dict[str, int]:
	return {str(class_id): count for class_id in range(6)}


def _stable_calibrated_contract_mapping(rule: str) -> dict[str, object]:
	raw = _contract_mapping()
	targets = {'small': 1_000, 'medium': 2_000, 'large': 4_000}
	pools = {
		size: {
			f'layout_{index:03d}': target + index * 10
			for index in range(5)
		}
		for size, target in targets.items()
	}
	raw['target_train_voxel_counts'] = targets
	if rule == TARGET_CALIBRATION_RULE:
		raw['target_calibration'] = {
			'rule': TARGET_CALIBRATION_RULE,
			'active_pool_train_voxel_counts': pools,
		}
	else:
		raw['target_calibration'] = {
			'rule': FIXED_TRAIN_VOXEL_COUNTS_RULE,
			'fixed_target_train_voxel_counts': dict(targets),
			'active_pool_train_voxel_counts': pools,
		}
	return raw


def _contract_mapping() -> dict[str, object]:
	layouts = []
	for index in range(5):
		inline = 100 + index * 10
		crossline = 200 + index * 10
		layouts.append(
			{
				'layout_id': f'layout_{index:03d}',
				'sizes': {
					'small': {
						'inline_lines': [inline],
						'crossline_lines': [crossline],
						'target_train_voxel_count': 1000,
					},
					'medium': {
						'inline_lines': [inline, inline + 1],
						'crossline_lines': [crossline, crossline + 1],
						'target_train_voxel_count': 2000,
					},
					'large': {
						'inline_lines': [inline + offset for offset in range(4)],
						'crossline_lines': [
							crossline + offset for offset in range(4)
						],
						'target_train_voxel_count': 4000,
					},
				},
			}
		)
	return {
		'schema_version': CONTRACT_SCHEMA_VERSION,
		'statistical_unit': 'layout_id',
		'nesting_semantics': NESTING_SEMANTICS,
		'validation_mask_semantics': VALIDATION_MASK_SEMANTICS,
		'stable_selection_semantics': STABLE_SELECTION_SEMANTICS,
		'patch_size': [8, 8, 8],
		'allowed_relative_error': 0.05,
		'decoder_seed': DECODER_SEED,
		'layouts': layouts,
		'decoder': deepcopy(dict(FIXED_DECODER_CONTRACT)),
	}
