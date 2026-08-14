
from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config import (
	f3_lithology_voxel_section_layout_contract_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	CONTRACT_SCHEMA_VERSION,
	DECODER_SEED,
	FIXED_DECODER_CONTRACT,
	NESTING_SEMANTICS,
	STABLE_SELECTION_SEMANTICS,
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
