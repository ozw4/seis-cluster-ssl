from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.volve.horizon_data import (
	HORIZON_NAMES,
	SECTION_STATISTICS_FIELDS,
	VolveHorizonData,
	load_volve_horizon_data,
	resolve_volve_horizon_inspection_config,
	section_statistics,
	write_section_statistics_csv,
)
from seis_ssl_cluster.volve.horizon_layouts import (
	DATA_SIZE_PREFIX,
	LAYOUT_IDS,
	SELECTION_SEMANTICS,
	build_all_horizon_split_plans,
	build_horizon_split_plan,
	load_volve_horizon_layouts,
	reserved_large_lines,
	selected_training_lines,
)
from tests.seis_ssl_cluster.helpers_volve import (
	write_synthetic_volve_horizon_root,
)


@pytest.fixture
def horizon_fixture(tmp_path: Path) -> tuple[VolveHorizonData, Path]:
	root, geometry = write_synthetic_volve_horizon_root(tmp_path)
	data = load_volve_horizon_data(root, geometry=geometry)
	layout_path = _write_layout(tmp_path)
	return data, layout_path


def test_binding_and_visual_review_contract_is_validated(tmp_path: Path) -> None:
	root, geometry = write_synthetic_volve_horizon_root(tmp_path)
	data = load_volve_horizon_data(root, geometry=geometry)
	assert data.horizon_names == HORIZON_NAMES
	assert isinstance(data.inline_values, np.memmap)
	assert isinstance(data.valid_trace_mask, np.memmap)
	manual_path = root / 'qc/volve_binding_visual_qc_v1/manual_review.json'
	manual = json.loads(manual_path.read_text(encoding='utf-8'))
	manual['horizon_visual_qc'] = 'FAIL'
	manual_path.write_text(json.dumps(manual), encoding='utf-8')
	with pytest.raises(ValueError, match='horizon visual QC must be PASS'):
		load_volve_horizon_data(root, geometry=geometry)


@pytest.mark.parametrize(
	('field', 'message'),
	[
		('status', 'manual review status must be PASS'),
		('fault_visual_qc', 'fault visual QC must be PASS'),
	],
)
def test_manual_review_failure_is_rejected(
	tmp_path: Path, field: str, message: str
) -> None:
	root, geometry = write_synthetic_volve_horizon_root(tmp_path)
	path = root / 'qc/volve_binding_visual_qc_v1/manual_review.json'
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload[field] = 'FAIL'
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match=message):
		load_volve_horizon_data(root, geometry=geometry)


def test_binding_schema_and_status_are_required(tmp_path: Path) -> None:
	root, geometry = write_synthetic_volve_horizon_root(tmp_path)
	path = root / 'manifests/volve_binding_v2/volve_grid_binding_summary_v2.json'
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['schema_version'] = 1
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='schema_version must be 2'):
		load_volve_horizon_data(root, geometry=geometry)


def test_binding_failure_status_is_rejected(tmp_path: Path) -> None:
	root, geometry = write_synthetic_volve_horizon_root(tmp_path)
	path = root / 'manifests/volve_binding_v2/volve_grid_binding_summary_v2.json'
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['status'] = 'FAIL'
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='binding status must be PASS'):
		load_volve_horizon_data(root, geometry=geometry)


def test_binding_npz_summary_hash_is_required(tmp_path: Path) -> None:
	root, geometry = write_synthetic_volve_horizon_root(tmp_path)
	path = root / 'manifests/volve_binding_v2/volve_horizon_binding_v2.npz'
	with path.open('ab') as file_obj:
		file_obj.write(b'changed')
	with pytest.raises(ValueError, match='NPZ SHA-256'):
		load_volve_horizon_data(root, geometry=geometry)


def test_exact_float_physical_axes_are_accepted(tmp_path: Path) -> None:
	root, geometry = write_synthetic_volve_horizon_root(tmp_path)
	canonical = root / 'canonical/volve_st10010_full_t_v1'
	np.save(canonical / 'inline_values.npy', np.arange(100, 106, dtype=np.float64))
	np.save(
		canonical / 'crossline_values.npy',
		np.arange(200, 207, dtype=np.float32),
	)
	data = load_volve_horizon_data(root, geometry=geometry)
	assert data.inline_values.dtype == np.float64
	assert data.crossline_values.dtype == np.float32


def test_physical_lines_map_to_canonical_indices(
	horizon_fixture: tuple[VolveHorizonData, Path],
) -> None:
	data, layout_path = horizon_fixture
	layouts = load_volve_horizon_layouts(layout_path, data)
	plan = build_horizon_split_plan(data, layouts, 'layout_000', 'medium')
	assert plan.selected_physical_lines.inline == (100, 101)
	assert plan.selected_physical_lines.crossline == (200, 201)
	assert plan.selected_indices.inline == (0, 1)
	assert plan.selected_indices.crossline == (0, 1)
	assert plan.validation_indices.inline == (5,)
	assert plan.validation_indices.crossline == (6,)


def test_five_layouts_and_nested_sizes_are_exact(
	horizon_fixture: tuple[VolveHorizonData, Path],
) -> None:
	data, layout_path = horizon_fixture
	layouts = load_volve_horizon_layouts(layout_path, data)
	assert tuple(layouts.layouts) == LAYOUT_IDS
	for lines in layouts.layouts.values():
		assert len(lines.inline) == 4
		assert len(lines.crossline) == 4
	for layout_id in LAYOUT_IDS:
		small = selected_training_lines(layouts, layout_id, 'small')
		medium = selected_training_lines(layouts, layout_id, 'medium')
		large = selected_training_lines(layouts, layout_id, 'large')
		assert small.inline == large.inline[:1]
		assert medium.inline == large.inline[:2]
		assert small.crossline == large.crossline[:1]
		assert medium.crossline == large.crossline[:2]
	for size, prefix in DATA_SIZE_PREFIX.items():
		selections = {
			(
				selected_training_lines(layouts, layout_id, size).inline,
				selected_training_lines(layouts, layout_id, size).crossline,
			)
			for layout_id in LAYOUT_IDS
		}
		assert len(selections) == 5
		assert all(len(item[0]) == prefix for item in selections)


def test_validation_is_disjoint_from_same_orientation_training(
	horizon_fixture: tuple[VolveHorizonData, Path],
) -> None:
	data, layout_path = horizon_fixture
	layouts = load_volve_horizon_layouts(layout_path, data)
	for lines in layouts.layouts.values():
		assert not set(lines.inline) & set(layouts.validation.inline)
		assert not set(lines.crossline) & set(layouts.validation.crossline)


def test_train_mask_uses_all_available_points_and_deduplicates_intersection(
	horizon_fixture: tuple[VolveHorizonData, Path],
) -> None:
	data, layout_path = horizon_fixture
	layouts = load_volve_horizon_layouts(layout_path, data)
	plan = build_horizon_split_plan(data, layouts, 'layout_000', 'small')
	selected = np.zeros(data.shape_xy, dtype=np.bool_)
	selected[0, :] = True
	selected[:, 0] = True
	validation = np.zeros(data.shape_xy, dtype=np.bool_)
	validation[5, :] = True
	validation[:, 6] = True
	expected = (
		data.bound_valid_mask
		& selected[np.newaxis, :, :]
		& ~validation[np.newaxis, :, :]
	)
	assert np.array_equal(plan.train_mask, expected)
	assert np.count_nonzero(plan.train_mask[1]) == 9
	assert not plan.train_mask[0, 1, 1]
	assert not np.any(plan.train_mask[:, 5, :])
	assert not np.any(plan.train_mask[:, :, 6])


def test_fixed_common_test_is_invariant_and_keeps_unused_candidates_reserved(
	horizon_fixture: tuple[VolveHorizonData, Path],
) -> None:
	data, layout_path = horizon_fixture
	layouts = load_volve_horizon_layouts(layout_path, data)
	plans = build_all_horizon_split_plans(data, layouts)
	assert len(plans) == 15
	primary_hashes = {
		plan.identity()['mask_sha256']['test_primary_common'] for plan in plans
	}
	secondary_hashes = {
		plan.identity()['mask_sha256']['test_secondary_per_horizon']
		for plan in plans
	}
	assert len(primary_hashes) == 1
	assert len(secondary_hashes) == 1
	reserved = reserved_large_lines(layouts)
	first = plans[0]
	for physical in reserved.inline:
		index = int(np.flatnonzero(data.inline_values == physical)[0])
		assert not np.any(first.test_per_horizon_mask[:, index, :])
	for physical in reserved.crossline:
		index = int(np.flatnonzero(data.crossline_values == physical)[0])
		assert not np.any(first.test_per_horizon_mask[:, :, index])


def test_per_horizon_missing_observations_are_excluded_from_loss_masks(
	horizon_fixture: tuple[VolveHorizonData, Path],
) -> None:
	data, layout_path = horizon_fixture
	layouts = load_volve_horizon_layouts(layout_path, data)
	plan = build_horizon_split_plan(data, layouts, 'layout_001', 'large')
	assert not plan.train_mask[0, 1, 1]
	assert not plan.train_mask[4, 2, 2]
	assert np.all(plan.train_mask <= data.bound_valid_mask)
	assert np.all(plan.validation_mask <= data.bound_valid_mask)
	assert np.all(plan.test_per_horizon_mask <= data.bound_valid_mask)


def test_twt_window_and_plan_identity_are_fixed(
	horizon_fixture: tuple[VolveHorizonData, Path],
) -> None:
	data, layout_path = horizon_fixture
	layouts = load_volve_horizon_layouts(layout_path, data)
	plans = build_all_horizon_split_plans(data, layouts)
	assert {
		(
			plan.twt_window.start_index,
			plan.twt_window.stop_index_exclusive,
			plan.twt_window.length_samples,
		)
		for plan in plans
	} == {(552, 768, 216)}
	assert len({plan.scientific_identity_sha256 for plan in plans}) == 15
	assert all(
		plan.identity()['layout_config_sha256'] == layouts.config_sha256
		for plan in plans
	)


def test_section_csv_is_deterministic(
	horizon_fixture: tuple[VolveHorizonData, Path], tmp_path: Path
) -> None:
	data, _ = horizon_fixture
	first = tmp_path / 'first.csv'
	second = tmp_path / 'second.csv'
	assert write_section_statistics_csv(data, first) == 13
	assert write_section_statistics_csv(data, second) == 13
	assert first.read_bytes() == second.read_bytes()
	with first.open(encoding='utf-8', newline='') as file_obj:
		rows = list(csv.DictReader(file_obj))
	assert tuple(rows[0]) == SECTION_STATISTICS_FIELDS
	assert rows[0]['orientation'] == 'inline'
	assert rows[0]['physical_line_number'] == '100'
	assert rows[-1]['orientation'] == 'crossline'
	assert rows[-1]['physical_line_number'] == '206'
	assert section_statistics(data)[0]['array_index'] == 0


@pytest.mark.parametrize(
	('mutation', 'message'),
	[
		(lambda value: value.pop('selection'), 'must contain exactly'),
		(
			lambda value: value['selection'].__setitem__(
				'semantics', 'stable_hash_subsampling'
			),
			'selection.semantics',
		),
		(
			lambda value: value['layouts']['layout_000'].__setitem__(
				'inline', [100, 101]
			),
			'exactly 4',
		),
		(
			lambda value: value['validation'].__setitem__('inline', [100]),
			'overlaps validation',
		),
	],
)
def test_invalid_layout_config_is_rejected(
	horizon_fixture: tuple[VolveHorizonData, Path], mutation: object, message: str
) -> None:
	data, layout_path = horizon_fixture
	payload = yaml.safe_load(layout_path.read_text(encoding='utf-8'))
	assert callable(mutation)
	mutation(payload)
	layout_path.write_text(yaml.safe_dump(payload), encoding='utf-8')
	with pytest.raises((TypeError, ValueError), match=message):
		load_volve_horizon_layouts(layout_path, data)


def test_inspection_config_keeps_outputs_below_artifact_root(
	tmp_path: Path,
) -> None:
	artifact_root = (tmp_path / 'artifacts').resolve()
	config = resolve_volve_horizon_inspection_config(
		{
			'paths': {
				'volve_root': str((tmp_path / 'public').resolve()),
				'artifact_root': str(artifact_root),
				'layout_config': str(tmp_path / 'layouts.yaml'),
			},
			'outputs': {
				'section_statistics_csv': 'data/sections.csv',
				'split_plans_json': 'data/plans.json',
			},
		}
	)
	assert config.section_statistics_csv == artifact_root / 'data/sections.csv'
	assert config.split_plans_json == artifact_root / 'data/plans.json'


def test_concrete_experiment_layout_has_twenty_reserved_lines_per_orientation(
) -> None:
	path = (
		Path(__file__).resolve().parents[2]
		/ 'experiments/volve/horizon_benchmark_v1/20_horizon_supervision'
		/ '01_layouts.yaml'
	)
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	assert payload['selection'] == {'semantics': SELECTION_SEMANTICS}
	assert tuple(payload['layouts']) == LAYOUT_IDS
	inline = [
		value
		for lines in payload['layouts'].values()
		for value in lines['inline']
	]
	crossline = [
		value
		for lines in payload['layouts'].values()
		for value in lines['crossline']
	]
	assert len(inline) == len(set(inline)) == 20
	assert len(crossline) == len(set(crossline)) == 20
	assert payload['validation']['inline'][0] not in inline
	assert payload['validation']['crossline'][0] not in crossline


def _write_layout(tmp_path: Path) -> Path:
	inline = [100, 101, 102, 103, 104]
	crossline = [200, 201, 202, 203, 204]
	payload = {
		'selection': {'semantics': SELECTION_SEMANTICS},
		'validation': {'inline': [105], 'crossline': [206]},
		'layouts': {
			layout_id: {
				'inline': (inline[index:] + inline[:index])[:4],
				'crossline': (crossline[index:] + crossline[:index])[:4],
			}
			for index, layout_id in enumerate(LAYOUT_IDS)
		},
	}
	path = tmp_path / 'layouts.yaml'
	path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
	return path
