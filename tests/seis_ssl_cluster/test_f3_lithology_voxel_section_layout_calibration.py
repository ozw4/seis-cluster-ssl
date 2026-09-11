from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	CLASS_BALANCED_SELECTION_SEMANTICS,
	FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE,
	FIXED_TRAIN_VOXEL_COUNTS_RULE,
	LAYOUT_IDS,
	TARGET_CALIBRATION_RULE,
	TOKEN_ROW_VALIDATION_PRECEDENCE,
	TOKEN_ROW_VOXEL_MATERIALIZATION,
	TOKENIZATION_POLICY,
	f3_lithology_voxel_section_layout_contract_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	REQUIRED_CONDITION_FILES,
	validate_f3_lithology_voxel_section_layout_condition,
	validate_f3_lithology_voxel_section_layout_manifest,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_calibration import (
	CLASS_CAP_TARGET_RULE,
	FIXED_TARGET_RULE,
	TARGET_RULE,
	F3SectionLayoutCalibrationConfig,
	SectionTokenRowPool,
	active_pool_train_voxel_counts,
	build_section_layout_contract,
	calibrate_target_train_voxel_counts,
	f3_section_layout_calibration_config_from_mapping,
	inspect_section_candidates,
	load_section_lines,
	preview_class_balanced_layout_selection,
	run_section_layout_calibration,
	validate_layout_lines,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_selection import (
	CLASS_IDS,
	SELECTION_SEMANTICS,
	LayoutLines,
	SectionLine,
	SelectionPreview,
	TokenRow,
	preview_nested_selection,
)
from tests.helpers import run_python_proc

PROC = Path('proc/seis_ssl_cluster')
SHAPE = (32, 32, 8)
TRAIN_INLINE_INDICES = (0, 8, 16, 24, 30)
TRAIN_CROSSLINE_INDICES = (0, 8, 16, 24)
VALIDATION_CROSSLINE_INDEX = 31
INLINE_MIN = 100
CROSSLINE_MIN = 200
FIXED_SYNTHETIC_TARGETS = {'small': 124, 'medium': 548, 'large': 1200}
CLASS_CAPS = {'small': 25, 'medium': 50, 'large': 100}
CLASS_CAP_TARGETS = {'small': 9600, 'medium': 19200, 'large': 38400}
LAYOUT_SEEDS = {layout_id: index for index, layout_id in enumerate(LAYOUT_IDS)}
PORTABLE_SHAPE = (80, 80, 8)
PORTABLE_TRAIN_LINE_INDICES = (0, 8, 16, 24)
PORTABLE_VALIDATION_CROSSLINE_INDEX = 64


def test_candidates_report_train_counts_classes_and_validation_flag() -> None:
	grid, labels, lines = _volume_fixture()
	candidates = inspect_section_candidates(grid, labels, lines)
	by_key = {item.line.key: item for item in candidates}
	assert len(candidates) == len(lines)
	validation = by_key[('crossline', CROSSLINE_MIN + VALIDATION_CROSSLINE_INDEX)]
	assert validation.line.is_validation_line
	assert validation.canonical_train_voxel_count == 0
	inline = by_key[('inline', INLINE_MIN)]
	# 32 crossline traces on the inline minus the validation trace, 8 samples each.
	assert inline.canonical_train_voxel_count == 31 * 8
	assert all(inline.per_class_voxel_counts[str(item)] > 0 for item in CLASS_IDS)
	row = inline.to_dict()
	assert row['class_3_voxel_count'] == inline.per_class_voxel_counts['3']
	assert row['is_validation_line'] is False


def test_layout_validation_rejects_count_duplicate_unknown_and_validation() -> None:
	grid, labels, lines = _volume_fixture()
	candidates = inspect_section_candidates(grid, labels, lines)
	assert len(validate_layout_lines(_layout_mapping(), candidates)) == 5
	wrong_count = _layout_mapping()
	wrong_count['layouts'].pop()
	with pytest.raises(ValueError, match='exactly 5'):
		validate_layout_lines(wrong_count, candidates)
	duplicate = _layout_mapping()
	duplicate['layouts'][0]['ordered_inlines'][1] = INLINE_MIN
	with pytest.raises(ValueError, match='duplicate'):
		validate_layout_lines(duplicate, candidates)
	unknown = _layout_mapping()
	unknown['layouts'][0]['ordered_inlines'][-1] = INLINE_MIN + 1
	with pytest.raises(ValueError, match='unknown'):
		validate_layout_lines(unknown, candidates)
	validation = _layout_mapping()
	validation['layouts'][0]['ordered_crosslines'][-1] = (
		CROSSLINE_MIN + VALIDATION_CROSSLINE_INDEX
	)
	with pytest.raises(ValueError, match='validation'):
		validate_layout_lines(validation, candidates)
	three_lines = _layout_mapping()
	three_lines['layouts'][0]['ordered_inlines'].pop()
	with pytest.raises(ValueError, match='exactly 4'):
		validate_layout_lines(three_lines, candidates)


def test_active_pools_are_prefix_unions_and_targets_are_the_common_minimum() -> None:
	grid, labels, lines = _volume_fixture()
	layouts = validate_layout_lines(
		_layout_mapping(), inspect_section_candidates(grid, labels, lines)
	)
	pools = active_pool_train_voxel_counts(layouts, grid, labels, lines)
	assert set(pools) == {'small', 'medium', 'large'}
	for size in pools:
		assert set(pools[size]) == {f'layout_{index:03d}' for index in range(5)}
	# One inline (31 train traces) plus one crossline (32 traces) minus the shared
	# trace, times 8 samples.
	assert pools['small']['layout_000'] == (31 + 32 - 1) * 8
	assert pools['medium']['layout_000'] == (2 * 31 + 2 * 32 - 4) * 8
	assert pools['large']['layout_000'] == (4 * 31 + 4 * 32 - 16) * 8
	targets = calibrate_target_train_voxel_counts(pools)
	assert targets == {
		size: min(pools[size].values()) for size in ('small', 'medium', 'large')
	}
	skewed = {size: dict(values) for size, values in pools.items()}
	skewed['small']['layout_002'] = 7
	assert calibrate_target_train_voxel_counts(skewed)['small'] == 7
	flat = {size: dict(values) for size, values in pools.items()}
	flat['medium'] = dict(flat['small'])
	with pytest.raises(ValueError, match='strictly increase'):
		calibrate_target_train_voxel_counts(flat)
	with pytest.raises(ValueError, match='exactly'):
		calibrate_target_train_voxel_counts({'small': pools['small']})


def test_preview_with_calibrated_targets_is_nested_and_passes_gates() -> None:
	grid, labels, lines = _volume_fixture()
	layouts = validate_layout_lines(
		_layout_mapping(), inspect_section_candidates(grid, labels, lines)
	)
	targets = calibrate_target_train_voxel_counts(
		active_pool_train_voxel_counts(layouts, grid, labels, lines)
	)
	previews = preview_nested_selection(
		layouts[0], targets, grid, labels, lines, allowed_relative_error=0.05
	)
	selected = [set(item.selected_token_xyz) for item in previews]
	assert selected[0] < selected[1] < selected[2]
	for preview in previews:
		assert preview.actual_train_voxel_count == targets[preview.data_size]
		assert preview.relative_count_error == 0.0
		assert all(preview.per_class_voxel_counts[str(item)] > 0 for item in CLASS_IDS)
		assert all(value > 0 for value in preview.per_line_contributions.values())


def test_contract_gates_target_class_monitored_class_line_and_nesting() -> None:
	base = _manual_preview(relative_error=0.0)
	contract = _preview_contract(base)
	assert contract['selection_semantics'] == SELECTION_SEMANTICS
	assert contract['target_calibration']['rule'] == TARGET_RULE
	assert TARGET_RULE == TARGET_CALIBRATION_RULE
	assert TARGET_RULE == 'max_common_reachable_active_pool_v1'
	assert 'legacy_budget_source_identity' not in contract
	outside = replace(base, relative_count_error=0.100001)
	with pytest.raises(ValueError, match='relative error'):
		_preview_contract(outside, allowed_relative_error=0.1)
	missing = dict(base.per_class_voxel_counts)
	missing['1'] = 0
	with pytest.raises(ValueError, match='missing classes'):
		_preview_contract(replace(base, per_class_voxel_counts=missing))
	drift = replace(base, target_train_voxel_count=99)
	with pytest.raises(ValueError, match='calibrated target'):
		_preview_contract(drift)
	zero_line = dict(base.per_line_contributions)
	zero_line['crossline:200'] = 0
	with pytest.raises(ValueError, match='zero teacher voxels'):
		_preview_contract(replace(base, per_line_contributions=zero_line))
	previews = list(_preview_matrix(base))
	previews[0] = replace(previews[0], selected_token_xyz=((1, 1, 1),))
	with pytest.raises(ValueError, match='strict_small_medium_large'):
		build_section_layout_contract(
			_layouts(),
			{'small': 100, 'medium': 200, 'large': 400},
			previews,
			allowed_relative_error=0.1,
			validation_identity={'unchanged_by_preview': True},
			source_file_identities={},
			target_calibration=_fixture_calibration(),
		)
	with pytest.raises(ValueError, match='validation mask'):
		build_section_layout_contract(
			_layouts(),
			{'small': 100, 'medium': 200, 'large': 400},
			_preview_matrix(base),
			allowed_relative_error=0.1,
			validation_identity={'unchanged_by_preview': False},
			source_file_identities={},
			target_calibration=_fixture_calibration(),
		)


def test_contract_resolver_replays_targets_from_stored_pools() -> None:
	contract = _preview_contract(_manual_preview(relative_error=0.0))
	f3_lithology_voxel_section_layout_contract_from_mapping(contract)
	# One layout with a larger pool leaves the common minimum unchanged.
	pools = _fixture_pools()
	pools['large']['layout_003'] = 401
	f3_lithology_voxel_section_layout_contract_from_mapping(
		{**contract, 'target_calibration': _fixture_calibration(pools)}
	)
	pools['large']['layout_003'] = 399
	with pytest.raises(ValueError, match='must equal the minimum active pool 399'):
		f3_lithology_voxel_section_layout_contract_from_mapping(
			{**contract, 'target_calibration': _fixture_calibration(pools)}
		)
	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_section_layout_contract_from_mapping(
			{**contract, 'target_calibration': {**_fixture_calibration(), 'extra': 1}}
		)
	legacy = dict(contract)
	del legacy['target_calibration']
	f3_lithology_voxel_section_layout_contract_from_mapping(legacy)


def test_contract_resolver_validates_fixed_targets_and_reachability() -> None:
	contract = _preview_contract(_manual_preview(relative_error=0.0))
	contract['target_calibration'] = _fixture_fixed_calibration()
	f3_lithology_voxel_section_layout_contract_from_mapping(contract)

	fixed_drift = json.loads(json.dumps(contract))
	fixed_drift['target_calibration']['fixed_target_train_voxel_counts'][
		'small'
	] = 99
	with pytest.raises(ValueError, match='must match fixed target 99'):
		f3_lithology_voxel_section_layout_contract_from_mapping(fixed_drift)

	top_level_drift = json.loads(json.dumps(contract))
	top_level_drift['target_train_voxel_counts']['small'] = 99
	with pytest.raises(ValueError, match='does not match layouts'):
		f3_lithology_voxel_section_layout_contract_from_mapping(top_level_drift)

	unreachable = json.loads(json.dumps(contract))
	unreachable['target_calibration']['active_pool_train_voxel_counts']['small'][
		'layout_004'
	] = 99
	with pytest.raises(ValueError, match='fixed target 100 exceeds active pool 99'):
		f3_lithology_voxel_section_layout_contract_from_mapping(unreachable)

	missing_fixed = json.loads(json.dumps(contract))
	del missing_fixed['target_calibration']['fixed_target_train_voxel_counts']
	with pytest.raises(ValueError, match='fixed_target_train_voxel_counts must'):
		f3_lithology_voxel_section_layout_contract_from_mapping(missing_fixed)


def test_contract_resolver_rejects_unrelated_target_rule() -> None:
	contract = _preview_contract(_manual_preview(relative_error=0.0))
	with pytest.raises(ValueError, match=r'target_calibration\.rule must be one of'):
		f3_lithology_voxel_section_layout_contract_from_mapping(
			{
				**contract,
				'target_calibration': {
					**_fixture_calibration(),
					'rule': 'unrelated_rule',
				},
			}
		)


def test_contract_resolver_rejects_omitted_active_pools() -> None:
	contract = _preview_contract(_manual_preview(relative_error=0.0))
	with pytest.raises(ValueError, match='active_pool_train_voxel_counts must'):
		f3_lithology_voxel_section_layout_contract_from_mapping(
			{**contract, 'target_calibration': {'rule': TARGET_RULE}}
		)


def test_contract_resolver_rejects_target_below_common_minimum_pool() -> None:
	contract = _preview_contract(_manual_preview(relative_error=0.0))
	pools = _fixture_pools()
	pools['small'] = {
		layout_id: 1_000 + 10 * index for index, layout_id in enumerate(LAYOUT_IDS)
	}
	with pytest.raises(ValueError, match='layout_000/small target must equal'):
		f3_lithology_voxel_section_layout_contract_from_mapping(
			{**contract, 'target_calibration': _fixture_calibration(pools)}
		)


def test_config_rejects_unknown_key_legacy_input_and_other_rule(
	tmp_path: Path,
) -> None:
	raw = _config_mapping(tmp_path)
	raw['selection']['unknown'] = 1
	with pytest.raises(ValueError, match='not allowed'):
		f3_section_layout_calibration_config_from_mapping(raw)
	raw = _config_mapping(tmp_path)
	raw['targets']['unknown'] = 1
	with pytest.raises(ValueError, match='not allowed'):
		f3_section_layout_calibration_config_from_mapping(raw)
	raw = _config_mapping(tmp_path)
	raw['inputs']['legacy_budget_manifest'] = str(tmp_path / 'legacy.json')
	with pytest.raises(ValueError, match='not allowed'):
		f3_section_layout_calibration_config_from_mapping(raw)
	raw = _config_mapping(tmp_path)
	raw['targets']['rule'] = 'legacy_budget_median'
	with pytest.raises(ValueError, match=r'targets\.rule'):
		f3_section_layout_calibration_config_from_mapping(raw)
	raw = _config_mapping(tmp_path)
	raw['selection']['allowed_relative_error'] = True
	with pytest.raises(TypeError, match='number'):
		f3_section_layout_calibration_config_from_mapping(raw)


def test_config_accepts_existing_and_fixed_target_rules(tmp_path: Path) -> None:
	maximum = f3_section_layout_calibration_config_from_mapping(
		_config_mapping(tmp_path)
	)
	assert maximum.target_rule == TARGET_RULE
	assert maximum.target_train_voxel_counts is None
	assert TARGET_RULE == TARGET_CALIBRATION_RULE
	assert TARGET_RULE == 'max_common_reachable_active_pool_v1'

	fixed = f3_section_layout_calibration_config_from_mapping(
		_fixed_config_mapping(tmp_path)
	)
	assert fixed.target_rule == FIXED_TARGET_RULE
	assert fixed.target_rule == FIXED_TRAIN_VOXEL_COUNTS_RULE
	assert dict(fixed.target_train_voxel_counts or {}) == FIXED_SYNTHETIC_TARGETS


def test_class_balanced_config_parses_complete_fixed_contract(tmp_path: Path) -> None:
	resolved = f3_section_layout_calibration_config_from_mapping(
		_class_balanced_config_mapping(tmp_path)
	)

	assert resolved.selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS
	assert resolved.target_rule == CLASS_CAP_TARGET_RULE
	assert resolved.target_rule == FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE
	assert dict(resolved.per_class_token_row_caps or {}) == CLASS_CAPS
	assert dict(resolved.target_train_voxel_counts or {}) == CLASS_CAP_TARGETS
	assert dict(resolved.layout_subsample_seeds or {}) == LAYOUT_SEEDS
	assert dict(resolved.tokenization_policy or {}) == dict(TOKENIZATION_POLICY)
	assert resolved.reference_valid_tokens == tmp_path / 'valid.npy'


@pytest.mark.parametrize(
	('section', 'field', 'message'),
	[
		('targets', 'per_class_token_row_caps', 'must be a mapping'),
		('targets', 'nominal_train_voxel_counts', 'must be a mapping'),
		('selection', 'layout_subsample_seeds', 'must define exactly'),
		('selection', 'tokenization_policy', 'must define exactly'),
		('inputs', 'reference_valid_tokens', 'missing'),
	],
)
def test_class_balanced_config_rejects_missing_required_fields(
	tmp_path: Path, section: str, field: str, message: str
) -> None:
	raw = _class_balanced_config_mapping(tmp_path)
	del raw[section][field]

	with pytest.raises((TypeError, ValueError), match=message):
		f3_section_layout_calibration_config_from_mapping(raw)


@pytest.mark.parametrize(
	'caps',
	[
		{'small': 25, 'medium': 50},
		{'small': 25, 'medium': 50, 'large': 100, 'tiny': 1},
		{'small': True, 'medium': 50, 'large': 100},
		{'small': 0, 'medium': 50, 'large': 100},
		{'small': -1, 'medium': 50, 'large': 100},
		{'small': 25.0, 'medium': 50, 'large': 100},
		{'small': '25', 'medium': 50, 'large': 100},
		{'small': 25, 'medium': 25, 'large': 100},
		{'small': 25, 'medium': 100, 'large': 50},
	],
)
def test_class_balanced_config_rejects_invalid_per_class_caps(
	tmp_path: Path, caps: dict[str, object]
) -> None:
	raw = _class_balanced_config_mapping(tmp_path)
	raw['targets']['per_class_token_row_caps'] = caps

	with pytest.raises((TypeError, ValueError)):
		f3_section_layout_calibration_config_from_mapping(raw)


@pytest.mark.parametrize('invalid_seed', [True, 1.0, '0'])
def test_class_balanced_config_rejects_noninteger_layout_seed(
	tmp_path: Path, invalid_seed: object
) -> None:
	raw = _class_balanced_config_mapping(tmp_path)
	raw['selection']['layout_subsample_seeds']['layout_000'] = invalid_seed

	with pytest.raises(TypeError, match=r'layout_000.*integer'):
		f3_section_layout_calibration_config_from_mapping(raw)


def test_class_balanced_config_rejects_duplicate_or_suffix_drifted_seeds(
	tmp_path: Path,
) -> None:
	duplicate = _class_balanced_config_mapping(tmp_path)
	duplicate['selection']['layout_subsample_seeds']['layout_001'] = 0
	with pytest.raises(ValueError, match='unique seeds'):
		f3_section_layout_calibration_config_from_mapping(duplicate)

	drifted = _class_balanced_config_mapping(tmp_path)
	drifted['selection']['layout_subsample_seeds']['layout_000'] = 5
	with pytest.raises(ValueError, match=r'unique seeds|layout suffix'):
		f3_section_layout_calibration_config_from_mapping(drifted)


def test_class_balanced_config_rejects_nominal_formula_and_policy_drift(
	tmp_path: Path,
) -> None:
	nominal = _class_balanced_config_mapping(tmp_path)
	nominal['targets']['nominal_train_voxel_counts']['small'] += 1
	with pytest.raises(ValueError, match=r'six classes \* cap \* 8 \* 8'):
		f3_section_layout_calibration_config_from_mapping(nominal)

	policy = _class_balanced_config_mapping(tmp_path)
	policy['selection']['tokenization_policy']['min_labeled_fraction'] = 0.6
	with pytest.raises(ValueError, match='fixed v1 tokenization policy'):
		f3_section_layout_calibration_config_from_mapping(policy)


def test_class_balanced_active_pool_shortage_reports_layout_size_and_class() -> None:
	grid, labels, lines = _volume_fixture()
	layout = validate_layout_lines(
		_layout_mapping(), inspect_section_candidates(grid, labels, lines)
	)[0]
	rows = tuple(
		TokenRow('inline', INLINE_MIN, (class_id % 4, class_id // 4, 0), class_id)
		for class_id in range(5)
	)
	pool = SectionTokenRowPool(
		train_rows=rows,
		validation_token_xyz=(),
		train_row_count_before_validation_precedence=len(rows),
		train_rows_removed_by_validation_precedence=0,
	)

	with pytest.raises(
		ValueError,
		match=(
			r'layout_000/small class 5 token-row pool 0 is below cap 1 '
			r'after validation precedence'
		),
	):
		preview_class_balanced_layout_selection(
			layout,
			pool,
			grid,
			labels,
			lines,
			subsample_seed=0,
			per_class_token_row_caps={'small': 1, 'medium': 2, 'large': 3},
			target_train_voxel_counts={
				'small': 384,
				'medium': 768,
				'large': 1152,
			},
		)


def test_class_balanced_contract_builder_records_replayable_provenance() -> None:
	previews = _class_balanced_preview_matrix()
	selection = _class_balanced_selection_provenance()
	calibration = _class_balanced_target_calibration()

	contract = build_section_layout_contract(
		_layouts(),
		CLASS_CAP_TARGETS,
		previews,
		allowed_relative_error=0.05,
		validation_identity={'unchanged_by_preview': True},
		source_file_identities={},
		target_calibration=calibration,
		selection_semantics=CLASS_BALANCED_SELECTION_SEMANTICS,
		class_balanced_selection=selection,
	)
	resolved = f3_lithology_voxel_section_layout_contract_from_mapping(contract)

	assert resolved.selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS
	assert resolved.stable_selection_semantics is None
	assert dict(resolved.class_balanced_selection or {})
	assert contract['target_train_voxel_counts'] == CLASS_CAP_TARGETS
	assert contract['target_calibration'] == calibration
	assert contract['class_balanced_selection'] == selection
	first = contract['layouts'][0]['sizes']['small']
	assert first['subsample_seed'] == 0
	assert first['per_class_token_row_cap'] == 25
	assert first['selected_token_row_count'] == 150
	assert first['selected_token_row_identity_sha256'] == f'{0:064x}'
	assert first['active_pool_per_class_token_row_counts'] == {
		str(class_id): 30 for class_id in CLASS_IDS
	}


def test_config_requires_rule_specific_target_counts(tmp_path: Path) -> None:
	missing = _config_mapping(tmp_path)
	missing['targets'] = {'rule': FIXED_TARGET_RULE}
	with pytest.raises(ValueError, match='train_voxel_counts is required'):
		f3_section_layout_calibration_config_from_mapping(missing)

	unexpected = _config_mapping(tmp_path)
	unexpected['targets']['train_voxel_counts'] = dict(FIXED_SYNTHETIC_TARGETS)
	with pytest.raises(ValueError, match='must not be specified'):
		f3_section_layout_calibration_config_from_mapping(unexpected)


@pytest.mark.parametrize(
	'counts',
	[
		{'small': 124, 'medium': 548},
		{'small': 124, 'medium': 548, 'large': 1200, 'tiny': 1},
		{'small': True, 'medium': 548, 'large': 1200},
		{'small': 0, 'medium': 548, 'large': 1200},
		{'small': -1, 'medium': 548, 'large': 1200},
		{'small': 124.0, 'medium': 548, 'large': 1200},
		{'small': '124', 'medium': 548, 'large': 1200},
		{'small': 124, 'medium': 124, 'large': 1200},
		{'small': 124, 'medium': 1200, 'large': 548},
	],
)
def test_config_rejects_invalid_fixed_target_counts(
	tmp_path: Path, counts: dict[str, object]
) -> None:
	raw = _fixed_config_mapping(tmp_path)
	raw['targets']['train_voxel_counts'] = counts
	with pytest.raises((TypeError, ValueError)):
		f3_section_layout_calibration_config_from_mapping(raw)


def test_finalize_rejects_fixed_target_above_active_pool(tmp_path: Path) -> None:
	config = _write_cli_fixture(
		tmp_path,
		fixed_targets={'small': 497, 'medium': 548, 'large': 1200},
	)
	with pytest.raises(ValueError, match='small fixed target 497 exceeds'):
		run_section_layout_calibration(config, mode='finalize', dry_run=True)


@pytest.mark.parametrize('mode', ['inspect', 'finalize'])
def test_dry_run_never_writes_outputs(tmp_path: Path, mode: str) -> None:
	config = _write_cli_fixture(tmp_path)
	result = run_section_layout_calibration(config, mode=mode, dry_run=True)
	assert result
	assert not config.candidate_statistics_csv.exists()
	assert not config.candidate_statistics_json.exists()
	assert not config.canonical_contract.exists()


@pytest.mark.integration
def test_thin_clis_inspect_finalize_and_build_fifteen_conditions(  # noqa: PLR0915
	tmp_path: Path,
) -> None:
	config = _write_cli_fixture(
		tmp_path, fixed_targets=FIXED_SYNTHETIC_TARGETS
	)
	calibration_yaml = tmp_path / 'calibration.yaml'
	calibration_yaml.write_text(
		yaml.safe_dump(_fixed_config_mapping(tmp_path)), encoding='utf-8'
	)
	prepare = PROC / 'prepare_f3_lithology_voxel_section_layout_contract.py'
	for mode in ('inspect', 'finalize'):
		dry = run_python_proc(
			prepare, '--config', calibration_yaml, '--mode', mode, '--dry-run'
		)
		assert dry.returncode == 0, dry.stderr
		assert 'dry-run' in dry.stdout
		live = run_python_proc(prepare, '--config', calibration_yaml, '--mode', mode)
		assert live.returncode == 0, live.stderr
	rows = list(csv.DictReader(config.candidate_statistics_csv.open(encoding='utf-8')))
	assert len(rows) == len(TRAIN_INLINE_INDICES) + len(TRAIN_CROSSLINE_INDICES) + 1
	contract = json.loads(config.canonical_contract.read_text(encoding='utf-8'))
	resolved = f3_lithology_voxel_section_layout_contract_from_mapping(
		contract, line_inventory=_inventory_rows()
	)
	assert [layout.layout_id for layout in resolved.layouts] == [
		f'layout_{index:03d}' for index in range(5)
	]
	assert contract['target_calibration']['rule'] == FIXED_TARGET_RULE
	assert contract['target_calibration']['fixed_target_train_voxel_counts'] == (
		FIXED_SYNTHETIC_TARGETS
	)
	assert contract['target_train_voxel_counts'] == FIXED_SYNTHETIC_TARGETS
	for size in ('small', 'medium', 'large'):
		pools = contract['target_calibration']['active_pool_train_voxel_counts'][size]
		assert contract['target_train_voxel_counts'][size] <= min(pools.values())

	canonical_root = tmp_path / 'canonical'
	output_root = tmp_path / 'section_layout_v2'
	build_yaml = tmp_path / 'build.yaml'
	build_yaml.write_text(
		yaml.safe_dump(
			{
				'inputs': {
					'section_layout_contract': str(config.canonical_contract),
					'canonical_voxel_dataset': str(canonical_root),
					'source_label_volume': str(config.label_volume),
					'png_label_inventory': str(config.line_inventory),
					'segy_geometry_json': str(config.segy_geometry_json),
					'class_info': str(tmp_path / 'class_info.json'),
					'reference_valid_tokens': str(tmp_path / 'valid.npy'),
				},
				'outputs': {'output_root': str(output_root)},
			}
		),
		encoding='utf-8',
	)
	build = PROC / 'build_f3_lithology_voxel_section_layout_datasets.py'
	dry = run_python_proc(build, '--config', build_yaml, '--dry-run')
	assert dry.returncode == 0, dry.stderr
	assert 'condition_count: 15' in dry.stdout
	assert not output_root.exists()
	live = run_python_proc(build, '--config', build_yaml)
	assert live.returncode == 0, live.stderr
	manifest_path = output_root / 'section_layout_dataset_manifest.json'
	manifest = validate_f3_lithology_voxel_section_layout_manifest(manifest_path)
	assert manifest['condition_count'] == 15
	validation_hashes = set()
	train_masks: dict[tuple[str, str], np.ndarray] = {}
	has_token_granularity_error = False
	for row in manifest['rows']:
		root = Path(str(row['voxel_dataset_root']))
		assert root == (
			output_root
			/ 'datasets'
			/ f'layout={row["layout_id"]}'
			/ f'size={row["data_size"]}'
			/ 'voxel_supervision'
		)
		assert {path.name for path in root.iterdir()} == set(REQUIRED_CONDITION_FILES)
		validate_f3_lithology_voxel_section_layout_condition(root)
		target = FIXED_SYNTHETIC_TARGETS[row['data_size']]
		actual = row['actual_train_voxel_count']
		assert row['target_train_voxel_count'] == target
		assert row['relative_count_error'] == abs(actual - target) / target
		assert row['relative_count_error'] <= 0.05
		has_token_granularity_error |= actual != target
		pool = contract['target_calibration']['active_pool_train_voxel_counts'][
			row['data_size']
		][row['layout_id']]
		assert actual < pool
		assert all(int(v) > 0 for v in row['per_class_train_voxel_counts'].values())
		assert row['per_class_train_voxel_counts']['3'] > 0
		assert row['per_class_train_voxel_counts']['5'] > 0
		assert all(int(v) > 0 for v in row['per_line_contributions'].values())
		validation_hashes.add(row['validation_mask_sha256'])
		train_masks[(row['layout_id'], row['data_size'])] = np.load(
			root / 'supervision_split_grid.npy', allow_pickle=False
		) == 1
	assert has_token_granularity_error
	for layout_id in LAYOUT_IDS:
		small = train_masks[(layout_id, 'small')]
		medium = train_masks[(layout_id, 'medium')]
		large = train_masks[(layout_id, 'large')]
		assert np.all(~small | medium)
		assert np.all(~medium | large)
		assert np.any(medium & ~small)
		assert np.any(large & ~medium)
	assert len(validation_hashes) == 1
	assert not (output_root / 'datasets_v1').exists()


def test_class_balanced_portable_calibration_and_builder_e2e(  # noqa: PLR0915
	tmp_path: Path,
) -> None:
	config, raw_config = _write_class_balanced_cli_fixture(tmp_path)
	inspection = run_section_layout_calibration(
		config, mode='inspect', dry_run=False
	)
	contract = run_section_layout_calibration(
		config, mode='finalize', dry_run=False
	)
	serialized = json.loads(config.canonical_contract.read_text(encoding='utf-8'))
	assert serialized == contract
	provenance = inspection['token_row_pool_provenance']
	assert {
		key: provenance[key]
		for key in (
			'train_token_row_count',
			'train_row_count_before_validation_precedence',
			'train_rows_removed_by_validation_precedence',
			'validation_token_xyz_count',
		)
	} == {
		'train_token_row_count': 24,
		'train_row_count_before_validation_precedence': 25,
		'train_rows_removed_by_validation_precedence': 1,
		'validation_token_xyz_count': 10,
	}
	assert len(provenance['validation_token_xyz_sha256']) == 64
	assert set(provenance['validation_token_xyz_sha256']) <= set('0123456789abcdef')
	resolved = f3_lithology_voxel_section_layout_contract_from_mapping(
		serialized,
		line_inventory=_portable_inventory_rows(),
	)
	assert resolved.selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS
	assert len(resolved.layouts) == 5

	caps = {'small': 1, 'medium': 2, 'large': 4}
	targets = {'small': 384, 'medium': 768, 'large': 1536}
	for layout in serialized['layouts']:
		selected_by_size = []
		for size in ('small', 'medium', 'large'):
			spec = layout['sizes'][size]
			assert spec['target_train_voxel_count'] == targets[size]
			assert spec['per_class_token_row_cap'] == caps[size]
			assert spec['selected_token_row_count'] == 6 * caps[size]
			assert set(spec['per_class_selected_token_row_counts'].values()) == {
				caps[size]
			}
			assert set(spec['active_pool_per_class_token_row_counts'].values()) == {
				caps[size]
			}
			assert all(
				count > 0
				for count in spec['per_line_selected_token_row_counts'].values()
			)
			selected_by_size.append({tuple(row) for row in spec['selected_token_xyz']})
		assert selected_by_size[0] < selected_by_size[1] < selected_by_size[2]

	output_root = tmp_path / 'section_layout_v3'
	build_yaml = tmp_path / 'build_v3.yaml'
	build_yaml.write_text(
		yaml.safe_dump({
			'inputs': {
				'section_layout_contract': str(config.canonical_contract),
				'canonical_voxel_dataset': str(tmp_path / 'canonical'),
				'source_label_volume': str(config.label_volume),
				'png_label_inventory': str(config.line_inventory),
				'segy_geometry_json': str(config.segy_geometry_json),
				'class_info': str(tmp_path / 'class_info.json'),
				'reference_valid_tokens': str(config.reference_valid_tokens),
			},
			'outputs': {'output_root': str(output_root)},
		}),
		encoding='utf-8',
	)
	result = run_python_proc(
		PROC / 'build_f3_lithology_voxel_section_layout_datasets.py',
		'--config',
		build_yaml,
	)
	assert result.returncode == 0, result.stderr
	manifest = validate_f3_lithology_voxel_section_layout_manifest(
		output_root / 'section_layout_dataset_manifest.json'
	)
	assert manifest['condition_count'] == 15
	validation_hashes = set()
	train_masks: dict[tuple[str, str], np.ndarray] = {}
	for row in manifest['rows']:
		size = row['data_size']
		actual = row['actual_train_voxel_count']
		assert row['target_train_voxel_count'] == targets[size]
		assert actual == targets[size]
		assert row['relative_count_error'] <= raw_config['selection'][
			'allowed_relative_error'
		]
		assert all(
			int(value) > 0
			for value in row['per_class_train_voxel_counts'].values()
		)
		assert row['per_class_train_voxel_counts']['3'] > 0
		assert row['per_class_train_voxel_counts']['5'] > 0
		assert all(
			int(value) > 0 for value in row['per_line_contributions'].values()
		)
		validation_hashes.add(row['validation_mask_sha256'])
		condition_root = Path(str(row['voxel_dataset_root']))
		train_masks[(row['layout_id'], size)] = np.load(
			condition_root / 'supervision_split_grid.npy', allow_pickle=False
		) == 1
	for layout_id in LAYOUT_IDS:
		small = train_masks[(layout_id, 'small')]
		medium = train_masks[(layout_id, 'medium')]
		large = train_masks[(layout_id, 'large')]
		assert np.all(~small | medium)
		assert np.all(~medium | large)
		assert np.any(medium & ~small)
		assert np.any(large & ~medium)
	assert len(validation_hashes) == 1


def _volume_fixture() -> tuple[np.ndarray, np.ndarray, tuple[SectionLine, ...]]:
	labels = np.broadcast_to(np.arange(8, dtype=np.int16) % 6, SHAPE).copy()
	grid = np.zeros(SHAPE, dtype=np.uint8)
	for index in TRAIN_INLINE_INDICES:
		grid[index, :, :] = 1
	for index in TRAIN_CROSSLINE_INDICES:
		grid[:, index, :] = 1
	grid[:, VALIDATION_CROSSLINE_INDEX, :] = 2
	return grid, labels, _section_lines()


def _section_lines() -> tuple[SectionLine, ...]:
	return (
		*(
			SectionLine('inline', INLINE_MIN + index, index, is_validation_line=False)
			for index in TRAIN_INLINE_INDICES
		),
		*(
			SectionLine(
				'crossline', CROSSLINE_MIN + index, index, is_validation_line=False
			)
			for index in TRAIN_CROSSLINE_INDICES
		),
		SectionLine(
			'crossline',
			CROSSLINE_MIN + VALIDATION_CROSSLINE_INDEX,
			VALIDATION_CROSSLINE_INDEX,
			is_validation_line=True,
		),
	)


def _inventory_rows() -> list[dict[str, object]]:
	return [
		{
			'relative_path': f'{line.slice_type}_{line.slice_index}.png',
			'split': 'validation' if line.is_validation_line else 'train',
			'slice_type': line.slice_type,
			'slice_index': line.slice_index,
		}
		for line in _section_lines()
	]


def _layout_mapping() -> dict[str, list[dict[str, object]]]:
	inlines = [INLINE_MIN + index for index in TRAIN_INLINE_INDICES[:4]]
	crosslines = [CROSSLINE_MIN + index for index in TRAIN_CROSSLINE_INDICES]
	return {
		'layouts': [
			{
				'layout_id': f'layout_{index:03d}',
				'ordered_inlines': inlines[index:] + inlines[:index],
				'ordered_crosslines': crosslines[index:] + crosslines[:index],
			}
			for index in range(5)
		]
	}


def _layouts() -> tuple[LayoutLines, ...]:
	return tuple(
		LayoutLines(f'layout_{index:03d}', (100, 101, 102, 103), (200, 201, 202, 203))
		for index in range(5)
	)


def _manual_preview(*, relative_error: float) -> SelectionPreview:
	return SelectionPreview(
		layout_id='layout_000',
		data_size='small',
		inline_lines=(100,),
		crossline_lines=(200,),
		target_train_voxel_count=100,
		actual_train_voxel_count=110,
		count_error=10,
		relative_count_error=relative_error,
		selected_token_xyz=((0, 0, 0),),
		selected_flat_voxel_indices=tuple(range(110)),
		per_line_contributions={'inline:100': 55, 'crossline:200': 55},
		per_class_voxel_counts={str(item): 1 for item in CLASS_IDS},
	)


def _preview_matrix(base: SelectionPreview) -> tuple[SelectionPreview, ...]:
	result = []
	for layout in _layouts():
		for size, count, lines in (
			('small', 100, 1),
			('medium', 200, 2),
			('large', 400, 4),
		):
			line_counts = {
				**{f'inline:{100 + index}': 1 for index in range(lines)},
				**{f'crossline:{200 + index}': 1 for index in range(lines)},
			}
			if any(value <= 0 for value in base.per_line_contributions.values()):
				line_counts[f'crossline:{layout.ordered_crosslines[0]}'] = 0
			result.append(
				replace(
					base,
					layout_id=layout.layout_id,
					data_size=size,
					inline_lines=layout.ordered_inlines[:lines],
					crossline_lines=layout.ordered_crosslines[:lines],
					target_train_voxel_count=(
						count if base.target_train_voxel_count == 100 else 99
					),
					per_line_contributions=line_counts,
				)
			)
	return tuple(result)


def _class_balanced_preview_matrix() -> tuple[SelectionPreview, ...]:
	result = []
	for layout_index, layout in enumerate(_layouts()):
		for size_index, size in enumerate(('small', 'medium', 'large')):
			line_count = (1, 2, 4)[size_index]
			cap = CLASS_CAPS[size]
			target = CLASS_CAP_TARGETS[size]
			inline_lines = layout.ordered_inlines[:line_count]
			crossline_lines = layout.ordered_crosslines[:line_count]
			line_keys = (
				*(f'inline:{line}' for line in inline_lines),
				*(f'crossline:{line}' for line in crossline_lines),
			)
			result.append(
				SelectionPreview(
					layout_id=layout.layout_id,
					data_size=size,
					inline_lines=inline_lines,
					crossline_lines=crossline_lines,
					target_train_voxel_count=target,
					actual_train_voxel_count=target,
					count_error=0,
					relative_count_error=0.0,
					selected_token_xyz=tuple(
						(layout_index, token_index, 0)
						for token_index in range(size_index + 1)
					),
					selected_flat_voxel_indices=tuple(range(target)),
					per_line_contributions=_positive_distribution(
						target, line_keys
					),
					per_class_voxel_counts={
						str(class_id): target // len(CLASS_IDS)
						for class_id in CLASS_IDS
					},
					selection_semantics=CLASS_BALANCED_SELECTION_SEMANTICS,
					subsample_seed=layout_index,
					per_class_token_row_cap=cap,
					selected_token_row_count=len(CLASS_IDS) * cap,
					selected_token_row_identity_sha256=(
						f'{layout_index * 3 + size_index:064x}'
					),
					per_class_selected_token_row_counts={
						str(class_id): cap for class_id in CLASS_IDS
					},
					active_pool_per_class_token_row_counts={
						str(class_id): cap + 5 for class_id in CLASS_IDS
					},
					per_line_selected_token_row_counts=_positive_distribution(
						len(CLASS_IDS) * cap, line_keys
					),
				)
			)
	return tuple(result)


def _positive_distribution(total: int, keys: tuple[str, ...]) -> dict[str, int]:
	quotient, remainder = divmod(total, len(keys))
	return {
		key: quotient + (index < remainder) for index, key in enumerate(keys)
	}


def _class_balanced_selection_provenance() -> dict[str, object]:
	return {
		'per_class_token_row_caps': dict(CLASS_CAPS),
		'layout_subsample_seeds': dict(LAYOUT_SEEDS),
		'tokenization_policy': dict(TOKENIZATION_POLICY),
		'validation_precedence': TOKEN_ROW_VALIDATION_PRECEDENCE,
		'voxel_materialization': TOKEN_ROW_VOXEL_MATERIALIZATION,
		'token_row_pool_provenance': {
			'train_token_row_count': 1000,
			'train_row_count_before_validation_precedence': 1010,
			'train_rows_removed_by_validation_precedence': 10,
			'validation_token_xyz_count': 1,
			'validation_token_xyz_sha256': 'a' * 64,
		},
	}


def _class_balanced_target_calibration() -> dict[str, object]:
	return {
		'rule': CLASS_CAP_TARGET_RULE,
		'nominal_target_train_voxel_counts': dict(CLASS_CAP_TARGETS),
		'active_pool_token_row_counts': {
			size: {
				layout_id: {
					str(class_id): CLASS_CAPS[size] + 5
					for class_id in CLASS_IDS
				}
				for layout_id in LAYOUT_IDS
			}
			for size in ('small', 'medium', 'large')
		},
	}


def _fixture_pools() -> dict[str, dict[str, int]]:
	return {
		size: dict.fromkeys(LAYOUT_IDS, count)
		for size, count in (('small', 100), ('medium', 200), ('large', 400))
	}


def _fixture_calibration(
	pools: dict[str, dict[str, int]] | None = None,
) -> dict[str, object]:
	return {
		'rule': TARGET_RULE,
		'active_pool_train_voxel_counts': (
			_fixture_pools() if pools is None else pools
		),
	}


def _fixture_fixed_calibration() -> dict[str, object]:
	targets = {'small': 100, 'medium': 200, 'large': 400}
	return {
		'rule': FIXED_TARGET_RULE,
		'fixed_target_train_voxel_counts': targets,
		'active_pool_train_voxel_counts': {
			size: dict.fromkeys(LAYOUT_IDS, target + 20)
			for size, target in targets.items()
		},
	}


def _preview_contract(
	preview: SelectionPreview, *, allowed_relative_error: float = 0.1
) -> dict[str, object]:
	return build_section_layout_contract(
		_layouts(),
		{'small': 100, 'medium': 200, 'large': 400},
		_preview_matrix(preview),
		allowed_relative_error=allowed_relative_error,
		validation_identity={'unchanged_by_preview': True},
		source_file_identities={},
		target_calibration=_fixture_calibration(),
	)


def _config_mapping(tmp_path: Path) -> dict[str, dict[str, object]]:
	return {
		'inputs': {
			'canonical_split_grid': str(
				tmp_path / 'canonical' / 'supervision_split_grid.npy'
			),
			'label_volume': str(tmp_path / 'labels.npy'),
			'line_inventory': str(tmp_path / 'inventory.csv'),
			'segy_geometry_json': str(tmp_path / 'geometry.json'),
			'layout_lines': str(tmp_path / 'layouts.yaml'),
		},
		'selection': {
			'semantics': SELECTION_SEMANTICS,
			'patch_size_xyz': [8, 8, 8],
			'allowed_relative_error': 0.05,
		},
		'targets': {'rule': TARGET_RULE},
		'outputs': {
			'candidate_statistics_csv': str(tmp_path / 'out' / 'candidates.csv'),
			'candidate_statistics_json': str(tmp_path / 'out' / 'candidates.json'),
			'canonical_contract': str(tmp_path / 'out' / 'contract.json'),
		},
	}


def _fixed_config_mapping(tmp_path: Path) -> dict[str, dict[str, object]]:
	raw = _config_mapping(tmp_path)
	raw['targets'] = {
		'rule': FIXED_TARGET_RULE,
		'train_voxel_counts': dict(FIXED_SYNTHETIC_TARGETS),
	}
	return raw


def _class_balanced_config_mapping(
	tmp_path: Path,
) -> dict[str, dict[str, object]]:
	raw = _config_mapping(tmp_path)
	raw['inputs']['reference_valid_tokens'] = str(tmp_path / 'valid.npy')
	raw['selection'] = {
		'semantics': CLASS_BALANCED_SELECTION_SEMANTICS,
		'patch_size_xyz': [8, 8, 8],
		'allowed_relative_error': 0.05,
		'layout_subsample_seeds': dict(LAYOUT_SEEDS),
		'tokenization_policy': dict(TOKENIZATION_POLICY),
	}
	raw['targets'] = {
		'rule': CLASS_CAP_TARGET_RULE,
		'per_class_token_row_caps': dict(CLASS_CAPS),
		'nominal_train_voxel_counts': dict(CLASS_CAP_TARGETS),
	}
	return raw


def _write_cli_fixture(
	tmp_path: Path, *, fixed_targets: dict[str, int] | None = None
) -> F3SectionLayoutCalibrationConfig:
	grid, labels, _lines = _volume_fixture()
	canonical_root = tmp_path / 'canonical'
	canonical_root.mkdir(parents=True, exist_ok=True)
	grid_path = canonical_root / 'supervision_split_grid.npy'
	np.save(grid_path, grid)
	label_path = tmp_path / 'labels.npy'
	np.save(label_path, labels)
	valid_path = tmp_path / 'valid.npy'
	np.save(valid_path, np.ones((4, 4, 1), dtype=np.bool_))
	inventory_path = tmp_path / 'inventory.csv'
	with inventory_path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(
			handle, fieldnames=('relative_path', 'split', 'slice_type', 'slice_index')
		)
		writer.writeheader()
		writer.writerows(_inventory_rows())
	geometry_path = tmp_path / 'geometry.json'
	_write_json(
		geometry_path,
		{
			'segy_files': {
				'label': {
					'cube_shape': list(SHAPE),
					'iline_min': INLINE_MIN,
					'iline_max': INLINE_MIN + SHAPE[0] - 1,
					'xline_min': CROSSLINE_MIN,
					'xline_max': CROSSLINE_MIN + SHAPE[1] - 1,
				}
			}
		},
	)
	class_path = tmp_path / 'class_info.json'
	_write_json(
		class_path,
		{
			str(class_id): {
				'name': f'class_{class_id}',
				'color': [class_id * 10, class_id * 10, class_id * 10],
			}
			for class_id in range(6)
		},
	)
	(tmp_path / 'layouts.yaml').write_text(
		yaml.safe_dump(_layout_mapping()), encoding='utf-8'
	)
	classes = read_f3_lithology_class_info(class_path)
	_write_json(
		canonical_root / 'split_manifest.json',
		{
			'split_source': 'png_label_inventory',
			'split_unit': 'slice',
			'strategy': 'inventory_split_no_random_token_split',
			'no_random_split': True,
			'splits': {},
		},
	)
	(canonical_root / 'class_counts.csv').write_text(
		'split,class_id,class_name,count,fraction\n', encoding='utf-8'
	)
	(canonical_root / 'voxel_dataset_summary.md').write_text(
		'# synthetic canonical\n', encoding='utf-8'
	)
	_write_json(
		canonical_root / 'voxel_dataset_metadata.json',
		{
			'artifact_type': 'f3_lithology_voxel_supervision',
			'schema_version': 1,
			'dataset': {
				'name': 'f3_facies_benchmark',
				'version': 'facies_benchmark_v2',
			},
			'classes': [item.to_dict() for item in classes],
			'split_codes': {'unsupervised': 0, 'train': 1, 'validation': 2},
			'validation_precedence': True,
			'reference_embedding': {
				'patch_size': [8, 8, 8],
				'token_grid_shape': [4, 4, 1],
				'volume_shape_xyz': list(SHAPE),
			},
			'label_volume': _identity(label_path),
			'inventory': _identity(inventory_path),
			'reference_valid_tokens': _identity(valid_path),
			'source_identities': {
				'class_info': _identity(class_path),
				'segy_geometry_json': _identity(geometry_path),
			},
			'outputs': {'supervision_split_grid': str(grid_path)},
			'summary': {},
		},
	)
	raw_config = _config_mapping(tmp_path)
	if fixed_targets is not None:
		raw_config['targets'] = {
			'rule': FIXED_TARGET_RULE,
			'train_voxel_counts': dict(fixed_targets),
		}
	config = f3_section_layout_calibration_config_from_mapping(raw_config)
	lines = load_section_lines(config.line_inventory, config.segy_geometry_json)
	assert len(lines) == 10
	return config


def _write_class_balanced_cli_fixture(
	tmp_path: Path,
) -> tuple[F3SectionLayoutCalibrationConfig, dict[str, dict[str, object]]]:
	grid, labels = _portable_class_balanced_volume_fixture()
	canonical_root = tmp_path / 'canonical'
	canonical_root.mkdir(parents=True, exist_ok=True)
	grid_path = canonical_root / 'supervision_split_grid.npy'
	label_path = tmp_path / 'labels.npy'
	valid_path = tmp_path / 'valid.npy'
	np.save(grid_path, grid)
	np.save(label_path, labels)
	np.save(valid_path, np.ones((10, 10, 1), dtype=np.bool_))

	inventory_path = tmp_path / 'inventory.csv'
	with inventory_path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=('relative_path', 'split', 'slice_type', 'slice_index'),
		)
		writer.writeheader()
		writer.writerows(_portable_inventory_rows())
	geometry_path = tmp_path / 'geometry.json'
	_write_json(
		geometry_path,
		{
			'segy_files': {
				'label': {
					'cube_shape': list(PORTABLE_SHAPE),
					'iline_min': INLINE_MIN,
					'iline_max': INLINE_MIN + PORTABLE_SHAPE[0] - 1,
					'xline_min': CROSSLINE_MIN,
					'xline_max': CROSSLINE_MIN + PORTABLE_SHAPE[1] - 1,
				}
			}
		},
	)
	class_path = tmp_path / 'class_info.json'
	_write_json(
		class_path,
		{
			str(class_id): {
				'name': f'class_{class_id}',
				'color': [class_id * 10, class_id * 10, class_id * 10],
			}
			for class_id in CLASS_IDS
		},
	)
	(tmp_path / 'layouts.yaml').write_text(
		yaml.safe_dump(_portable_layout_mapping()), encoding='utf-8'
	)
	classes = read_f3_lithology_class_info(class_path)
	_write_json(
		canonical_root / 'split_manifest.json',
		{
			'split_source': 'png_label_inventory',
			'split_unit': 'slice',
			'strategy': 'inventory_split_no_random_token_split',
			'no_random_split': True,
			'splits': {},
		},
	)
	(canonical_root / 'class_counts.csv').write_text(
		'split,class_id,class_name,count,fraction\n', encoding='utf-8'
	)
	(canonical_root / 'voxel_dataset_summary.md').write_text(
		'# portable class-balanced canonical\n', encoding='utf-8'
	)
	_write_json(
		canonical_root / 'voxel_dataset_metadata.json',
		{
			'artifact_type': 'f3_lithology_voxel_supervision',
			'schema_version': 1,
			'dataset': {
				'name': 'f3_facies_benchmark',
				'version': 'facies_benchmark_v2',
			},
			'classes': [item.to_dict() for item in classes],
			'split_codes': {'unsupervised': 0, 'train': 1, 'validation': 2},
			'validation_precedence': True,
			'reference_embedding': {
				'patch_size': [8, 8, 8],
				'token_grid_shape': [10, 10, 1],
				'volume_shape_xyz': list(PORTABLE_SHAPE),
			},
			'label_volume': _identity(label_path),
			'inventory': _identity(inventory_path),
			'reference_valid_tokens': _identity(valid_path),
			'source_identities': {
				'class_info': _identity(class_path),
				'segy_geometry_json': _identity(geometry_path),
			},
			'outputs': {'supervision_split_grid': str(grid_path)},
			'summary': {},
		},
	)
	raw = _class_balanced_config_mapping(tmp_path)
	raw['targets']['per_class_token_row_caps'] = {
		'small': 1,
		'medium': 2,
		'large': 4,
	}
	raw['targets']['nominal_train_voxel_counts'] = {
		'small': 384,
		'medium': 768,
		'large': 1536,
	}
	return f3_section_layout_calibration_config_from_mapping(raw), raw


def _portable_class_balanced_volume_fixture() -> tuple[np.ndarray, np.ndarray]:
	grid = np.zeros(PORTABLE_SHAPE, dtype=np.uint8)
	labels = np.full(PORTABLE_SHAPE, -1, dtype=np.int16)
	specs = (
		('inline', 0, ((4, 0), (5, 1), (6, 2))),
		('crossline', 0, ((4, 3), (5, 4), (6, 5))),
		('inline', 8, ((4, 3), (5, 4), (6, 5))),
		('crossline', 8, ((4, 0), (5, 1), (6, 2))),
		('inline', 16, ((4, 0), (5, 1), (6, 2))),
		('inline', 24, ((4, 3), (5, 4), (6, 5))),
		('crossline', 16, ((4, 0), (5, 1), (6, 2))),
		('crossline', 24, ((4, 3), (5, 4), (6, 5))),
	)
	for slice_type, array_index, token_classes in specs:
		for variable_token_index, class_id in token_classes:
			start = variable_token_index * 8
			stop = start + 8
			if slice_type == 'inline':
				labels[array_index, start:stop, :] = class_id
				grid[array_index, start:stop, :] = 1
			else:
				labels[start:stop, array_index, :] = class_id
				grid[start:stop, array_index, :] = 1

	# This train row is usable before validation precedence. The validation line
	# produces the same token_xyz and therefore removes it from every layout pool.
	labels[0, 64:72, :] = 0
	grid[0, 64:72, :] = 1
	labels[:, PORTABLE_VALIDATION_CROSSLINE_INDEX, :] = 0
	grid[:, PORTABLE_VALIDATION_CROSSLINE_INDEX, :] = 2
	return grid, labels


def _portable_inventory_rows() -> list[dict[str, object]]:
	return [
		*[
			{
				'relative_path': f'inline_{INLINE_MIN + index}.png',
				'split': 'train',
				'slice_type': 'inline',
				'slice_index': INLINE_MIN + index,
			}
			for index in PORTABLE_TRAIN_LINE_INDICES
		],
		*[
			{
				'relative_path': f'crossline_{CROSSLINE_MIN + index}.png',
				'split': 'train',
				'slice_type': 'crossline',
				'slice_index': CROSSLINE_MIN + index,
			}
			for index in PORTABLE_TRAIN_LINE_INDICES
		],
		{
			'relative_path': (
				f'crossline_{CROSSLINE_MIN + PORTABLE_VALIDATION_CROSSLINE_INDEX}.png'
			),
			'split': 'validation',
			'slice_type': 'crossline',
			'slice_index': CROSSLINE_MIN + PORTABLE_VALIDATION_CROSSLINE_INDEX,
		},
	]


def _portable_layout_mapping() -> dict[str, list[dict[str, object]]]:
	inlines = [INLINE_MIN + index for index in PORTABLE_TRAIN_LINE_INDICES]
	crosslines = [CROSSLINE_MIN + index for index in PORTABLE_TRAIN_LINE_INDICES]
	return {
		'layouts': [
			{
				'layout_id': layout_id,
				'ordered_inlines': inlines,
				'ordered_crosslines': crosslines,
			}
			for layout_id in LAYOUT_IDS
		]
	}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
