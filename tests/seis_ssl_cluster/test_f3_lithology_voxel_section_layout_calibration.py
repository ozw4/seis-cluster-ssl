
from __future__ import annotations

import csv
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.voxel_section_layout_calibration import (
	F3SectionLayoutCalibrationConfig,
	build_section_layout_contract,
	f3_section_layout_calibration_config_from_mapping,
	inspect_section_candidates,
	load_legacy_budget_counts,
	median_target_counts,
	run_section_layout_calibration,
	validate_layout_lines,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_selection import (
	CLASS_IDS,
	SELECTION_SEMANTICS,
	LayoutLines,
	SectionLine,
	SelectionPreview,
	candidate_token_footprints,
	per_line_contributions,
	preview_nested_selection,
	replay_selected_teacher_mask,
	stable_token_order,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_requires_exact_legacy_budget_seed_matrix() -> None:
	payload = _legacy_manifest()
	rows = load_legacy_budget_counts(payload)
	assert len(rows) == 15
	duplicate = _legacy_manifest()
	duplicate['rows'][-1] = dict(duplicate['rows'][0])
	with pytest.raises(ValueError, match=r'duplicate|matrix'):
		load_legacy_budget_counts(duplicate)
	short = _legacy_manifest()
	short['rows'].pop()
	with pytest.raises(ValueError, match='exactly 15'):
		load_legacy_budget_counts(short)


def test_median_targets_use_actual_voxel_counts_not_cap_names() -> None:
	rows = load_legacy_budget_counts(_legacy_manifest())
	assert median_target_counts(rows) == {
		'small': 120,
		'medium': 332,
		'large': 712,
	}


def test_legacy_count_accepts_current_canonical_synonym() -> None:
	payload = _legacy_manifest()
	for row in payload['rows']:
		row['train_voxel_count'] = row.pop('actual_train_voxel_count')
	assert median_target_counts(load_legacy_budget_counts(payload))['small'] == 120


def test_layout_validation_rejects_count_duplicate_and_validation() -> None:
	grid, labels, lines = _volume_fixture()
	candidates = inspect_section_candidates(
		grid, labels, lines, patch_size_xyz=(8, 8, 8)
	)
	raw = _layout_mapping()
	assert len(validate_layout_lines(raw, candidates)) == 5
	wrong_count = _layout_mapping()
	wrong_count['layouts'].pop()
	with pytest.raises(ValueError, match='exactly 5'):
		validate_layout_lines(wrong_count, candidates)
	duplicate = _layout_mapping()
	duplicate['layouts'][0]['ordered_inlines'][1] = 100
	with pytest.raises(ValueError, match='duplicate'):
		validate_layout_lines(duplicate, candidates)
	validation = _layout_mapping()
	validation['layouts'][0]['ordered_crosslines'][-1] = 231
	with pytest.raises(ValueError, match='validation'):
		validate_layout_lines(validation, candidates)


def test_stable_token_order_is_repeatable_and_semantics_bound() -> None:
	grid, labels, lines = _volume_fixture()
	footprints = candidate_token_footprints(
		grid, labels, lines[:2], patch_size_xyz=(8, 8, 8)
	)
	first = [
		item.token_xyz
		for item in stable_token_order(footprints, layout_id='layout_000')
	]
	second = [
		item.token_xyz
		for item in stable_token_order(
			tuple(reversed(footprints)), layout_id='layout_000'
		)
	]
	assert first == second
	with pytest.raises(ValueError, match='semantics'):
		stable_token_order(
			footprints, layout_id='layout_000', semantics_version='drift'
		)


def test_partial_footprints_stay_on_active_planes_and_deduplicate_intersection() -> (
	None
):
	shape = (8, 8, 8)
	grid = np.ones(shape, dtype=np.uint8)
	labels = np.indices(shape).sum(axis=0).astype(np.int16) % 6
	lines = (
		SectionLine('inline', 100, 0, is_validation_line=False),
		SectionLine('crossline', 200, 0, is_validation_line=False),
	)
	footprints = candidate_token_footprints(
		grid, labels, lines, patch_size_xyz=(8, 8, 8)
	)
	assert len(footprints) == 1
	assert footprints[0].voxel_count == 8 * 8 + 8 * 8 - 8
	owned = footprints[0].per_line_flat_voxel_indices
	assert len(owned[('inline', 100)]) == 8 * 8
	assert len(owned[('crossline', 200)]) == 8 * 8 - 8
	assert set(owned[('inline', 100)]).isdisjoint(owned[('crossline', 200)])
	assert set(owned[('inline', 100)]) | set(owned[('crossline', 200)]) == set(
		footprints[0].flat_voxel_indices
	)
	assert per_line_contributions(footprints, lines) == {
		'inline:100': 8 * 8,
		'crossline:200': (8 * 8) - 8,
	}
	replayed = replay_selected_teacher_mask(
		grid, labels, lines, [footprints[0].token_xyz]
	)
	coordinates = np.indices(shape)
	assert np.array_equal(
		replayed, (coordinates[0] == 0) | (coordinates[1] == 0)
	)
	for flat in footprints[0].flat_voxel_indices:
		x, y, _ = np.unravel_index(flat, shape)
		assert x == 0 or y == 0


def test_coverage_uses_owned_teacher_voxels_not_geometric_token_intersection() -> (
	None
):
	shape = (8, 8, 8)
	grid = np.zeros(shape, dtype=np.uint8)
	grid[0, 0, :] = 1
	labels = np.indices(shape).sum(axis=0).astype(np.int16) % 6
	lines = (
		SectionLine('inline', 100, 0, is_validation_line=False),
		SectionLine('crossline', 200, 0, is_validation_line=False),
	)
	footprint = candidate_token_footprints(grid, labels, lines)[0]
	assert footprint.line_voxel_count(('inline', 100)) == 8
	assert footprint.line_voxel_count(('crossline', 200)) == 0
	with pytest.raises(ValueError, match=r'crossline.*contributes no teacher voxels'):
		preview_nested_selection(
			LayoutLines('layout_000', (100,), (200,)),
			{'small': 8, 'medium': 8, 'large': 8},
			grid,
			labels,
			lines,
		)


def test_preview_is_nested_and_satisfies_all_finalize_gates() -> None:
	grid, labels, lines = _volume_fixture()
	candidates = inspect_section_candidates(
		grid, labels, lines, patch_size_xyz=(8, 8, 8)
	)
	layout = validate_layout_lines(_layout_mapping(), candidates)[0]
	previews = preview_nested_selection(
		layout,
		{'small': 120, 'medium': 332, 'large': 712},
		grid,
		labels,
		lines,
		patch_size_xyz=(8, 8, 8),
		allowed_relative_error=0.1,
	)
	selected = [set(item.selected_token_xyz) for item in previews]
	assert selected[0] <= selected[1] <= selected[2]
	voxels = [set(item.selected_flat_voxel_indices) for item in previews]
	assert voxels[0] <= voxels[1] <= voxels[2]
	for preview in previews:
		assert set(preview.per_class_voxel_counts) == {str(item) for item in CLASS_IDS}
		assert all(preview.per_class_voxel_counts[str(item)] > 0 for item in CLASS_IDS)
		assert all(value > 0 for value in preview.per_line_contributions.values())


def test_target_tolerance_boundary_is_inclusive() -> None:
	preview = _manual_preview(relative_error=0.1)
	contract = build_section_layout_contract(
		_layouts(),
		{'small': 100, 'medium': 200, 'large': 400},
		_preview_matrix(preview),
		allowed_relative_error=0.1,
		validation_identity={'mask_sha256': 'a', 'unchanged_by_preview': True},
		source_file_identities={'grid': {'path': '/x', 'sha256': 'b'}},
		legacy_budget_source_identity={'path': '/legacy', 'sha256': 'c'},
	)
	assert contract['selection_semantics'] == SELECTION_SEMANTICS
	outside = replace(preview, relative_count_error=0.100001)
	with pytest.raises(ValueError, match='relative error'):
		build_section_layout_contract(
			_layouts(),
			{'small': 100, 'medium': 200, 'large': 400},
			_preview_matrix(outside),
			allowed_relative_error=0.1,
			validation_identity={'unchanged_by_preview': True},
			source_file_identities={},
			legacy_budget_source_identity={},
		)


def test_finalize_rejects_missing_class_and_zero_line_contribution() -> None:
	missing = _manual_preview(relative_error=0.0)
	missing_counts = dict(missing.per_class_voxel_counts)
	missing_counts['5'] = 0
	missing = replace(missing, per_class_voxel_counts=missing_counts)
	with pytest.raises(ValueError, match='missing classes'):
		_preview_contract(missing)
	zero = _manual_preview(relative_error=0.0)
	line_counts = dict(zero.per_line_contributions)
	line_counts['crossline:200'] = 0
	zero = replace(zero, per_line_contributions=line_counts)
	with pytest.raises(ValueError, match='zero teacher voxels'):
		_preview_contract(zero)


def test_finalize_rejects_changed_validation_and_non_nested_tokens() -> None:
	preview = _manual_preview(relative_error=0.0)
	with pytest.raises(ValueError, match='validation mask'):
		build_section_layout_contract(
			_layouts(),
			{'small': 100, 'medium': 200, 'large': 400},
			_preview_matrix(preview),
			allowed_relative_error=0.1,
			validation_identity={'unchanged_by_preview': False},
			source_file_identities={},
			legacy_budget_source_identity={},
		)
	previews = list(_preview_matrix(preview))
	previews[0] = replace(previews[0], selected_token_xyz=((1, 1, 1),))
	with pytest.raises(ValueError, match='nested'):
		build_section_layout_contract(
			_layouts(),
			{'small': 100, 'medium': 200, 'large': 400},
			previews,
			allowed_relative_error=0.1,
			validation_identity={'unchanged_by_preview': True},
			source_file_identities={},
			legacy_budget_source_identity={},
		)


def test_config_rejects_unknown_key_and_bool_as_number(tmp_path: Path) -> None:
	raw = _config_mapping(tmp_path)
	raw['selection']['unknown'] = 1
	with pytest.raises(ValueError, match='not allowed'):
		f3_section_layout_calibration_config_from_mapping(raw)
	raw = _config_mapping(tmp_path)
	raw['selection']['allowed_relative_error'] = True
	with pytest.raises(TypeError, match='number'):
		f3_section_layout_calibration_config_from_mapping(raw)


@pytest.mark.parametrize('mode', ['inspect', 'finalize'])
def test_dry_run_never_writes_outputs(tmp_path: Path, mode: str) -> None:
	config = _write_cli_fixture(tmp_path)
	result = run_section_layout_calibration(config, mode=mode, dry_run=True)
	assert result
	assert not config.candidate_statistics_csv.exists()
	assert not config.candidate_statistics_json.exists()
	assert not config.canonical_contract.exists()


def _volume_fixture() -> tuple[np.ndarray, np.ndarray, tuple[SectionLine, ...]]:
	shape = (32, 32, 8)
	grid = np.ones(shape, dtype=np.uint8)
	grid[:, 31, :] = 2
	labels = np.broadcast_to(np.arange(8, dtype=np.int16) % 6, shape).copy()
	lines = (
		tuple(
			SectionLine('inline', 100 + index, index, is_validation_line=False)
			for index in (0, 8, 16, 24)
		)
		+ tuple(
			SectionLine('crossline', 200 + index, index, is_validation_line=False)
			for index in (0, 8, 16, 24)
		)
		+ (SectionLine('crossline', 231, 31, is_validation_line=True),)
	)
	return grid, labels, lines


def _legacy_manifest() -> dict[str, list[dict[str, object]]]:
	medians = {'cap25': 120, 'cap50': 332, 'cap100': 712}
	return {
		'rows': [
			{
				'budget_id': budget,
				'subsample_seed': seed,
				'actual_train_voxel_count': median + seed - 2,
			}
			for budget, median in medians.items()
			for seed in range(5)
		]
	}


def _layout_mapping() -> dict[str, list[dict[str, object]]]:
	inlines = [100, 108, 116, 124]
	crosslines = [200, 208, 216, 224]
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
					target_train_voxel_count=count,
					per_line_contributions=line_counts,
				)
			)
	return tuple(result)


def _preview_contract(preview: SelectionPreview) -> dict[str, object]:
	return build_section_layout_contract(
		_layouts(),
		{'small': 100, 'medium': 200, 'large': 400},
		_preview_matrix(preview),
		allowed_relative_error=0.1,
		validation_identity={'unchanged_by_preview': True},
		source_file_identities={},
		legacy_budget_source_identity={},
	)


def _config_mapping(tmp_path: Path) -> dict[str, dict[str, object]]:
	return {
		'inputs': {
			'legacy_budget_manifest': str(tmp_path / 'legacy.json'),
			'canonical_split_grid': str(tmp_path / 'grid.npy'),
			'label_volume': str(tmp_path / 'labels.npy'),
			'line_inventory': str(tmp_path / 'inventory.csv'),
			'segy_geometry_json': str(tmp_path / 'geometry.json'),
			'layout_lines': str(tmp_path / 'layouts.yaml'),
		},
		'selection': {
			'semantics': SELECTION_SEMANTICS,
			'patch_size_xyz': [8, 8, 8],
			'allowed_relative_error': 0.1,
		},
		'outputs': {
			'candidate_statistics_csv': str(tmp_path / 'out' / 'candidates.csv'),
			'candidate_statistics_json': str(tmp_path / 'out' / 'candidates.json'),
			'canonical_contract': str(tmp_path / 'out' / 'contract.json'),
		},
	}


def _write_cli_fixture(tmp_path: Path) -> F3SectionLayoutCalibrationConfig:
	grid, labels, lines = _volume_fixture()
	(tmp_path / 'legacy.json').write_text(
		json.dumps(_legacy_manifest()), encoding='utf-8'
	)
	np.save(tmp_path / 'grid.npy', grid)
	np.save(tmp_path / 'labels.npy', labels)
	with (tmp_path / 'inventory.csv').open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(
			handle, fieldnames=('split', 'slice_type', 'slice_index', 'array_index')
		)
		writer.writeheader()
		for line in lines:
			writer.writerow(
				{
					'split': 'validation' if line.is_validation_line else 'train',
					'slice_type': line.slice_type,
					'slice_index': line.slice_index,
					'array_index': line.array_index,
				}
			)
	(tmp_path / 'geometry.json').write_text(
		json.dumps({'iline_min': 100, 'xline_min': 200, 'cube_shape': [32, 32, 8]}),
		encoding='utf-8',
	)
	(tmp_path / 'layouts.yaml').write_text(
		json.dumps(_layout_mapping()), encoding='utf-8'
	)
	return f3_section_layout_calibration_config_from_mapping(_config_mapping(tmp_path))
