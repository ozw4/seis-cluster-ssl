from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
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
	TARGET_RULE,
	F3SectionLayoutCalibrationConfig,
	active_pool_train_voxel_counts,
	build_section_layout_contract,
	calibrate_target_train_voxel_counts,
	f3_section_layout_calibration_config_from_mapping,
	inspect_section_candidates,
	load_section_lines,
	run_section_layout_calibration,
	validate_layout_lines,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_selection import (
	CLASS_IDS,
	SELECTION_SEMANTICS,
	LayoutLines,
	SectionLine,
	SelectionPreview,
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
	with pytest.raises(ValueError, match='nested'):
		build_section_layout_contract(
			_layouts(),
			{'small': 100, 'medium': 200, 'large': 400},
			previews,
			allowed_relative_error=0.1,
			validation_identity={'unchanged_by_preview': True},
			source_file_identities={},
			target_calibration={'rule': TARGET_RULE},
		)
	with pytest.raises(ValueError, match='validation mask'):
		build_section_layout_contract(
			_layouts(),
			{'small': 100, 'medium': 200, 'large': 400},
			_preview_matrix(base),
			allowed_relative_error=0.1,
			validation_identity={'unchanged_by_preview': False},
			source_file_identities={},
			target_calibration={'rule': TARGET_RULE},
		)


def test_contract_resolver_rejects_target_above_active_pool() -> None:
	contract = _preview_contract(_manual_preview(relative_error=0.0))
	pools = {
		size: {f'layout_{index:03d}': 1_000 for index in range(5)}
		for size in ('small', 'medium', 'large')
	}
	accepted = {
		**contract,
		'target_calibration': {
			'rule': TARGET_RULE,
			'active_pool_train_voxel_counts': pools,
		},
	}
	f3_lithology_voxel_section_layout_contract_from_mapping(accepted)
	pools['large']['layout_003'] = 399
	with pytest.raises(ValueError, match='exceeds its active pool'):
		f3_lithology_voxel_section_layout_contract_from_mapping(accepted)
	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_section_layout_contract_from_mapping(
			{**contract, 'target_calibration': {'rule': TARGET_RULE, 'extra': 1}}
		)


def test_config_rejects_unknown_key_legacy_input_and_other_rule(
	tmp_path: Path,
) -> None:
	raw = _config_mapping(tmp_path)
	raw['selection']['unknown'] = 1
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


@pytest.mark.parametrize('mode', ['inspect', 'finalize'])
def test_dry_run_never_writes_outputs(tmp_path: Path, mode: str) -> None:
	config = _write_cli_fixture(tmp_path)
	result = run_section_layout_calibration(config, mode=mode, dry_run=True)
	assert result
	assert not config.candidate_statistics_csv.exists()
	assert not config.candidate_statistics_json.exists()
	assert not config.canonical_contract.exists()


@pytest.mark.integration
def test_thin_clis_inspect_finalize_and_build_fifteen_conditions(
	tmp_path: Path,
) -> None:
	config = _write_cli_fixture(tmp_path)
	calibration_yaml = tmp_path / 'calibration.yaml'
	calibration_yaml.write_text(
		yaml.safe_dump(_config_mapping(tmp_path)), encoding='utf-8'
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
	assert contract['target_calibration']['rule'] == TARGET_RULE
	for size in ('small', 'medium', 'large'):
		pools = contract['target_calibration']['active_pool_train_voxel_counts'][size]
		assert contract['target_train_voxel_counts'][size] == min(pools.values())

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
		assert row['target_train_voxel_count'] == row['actual_train_voxel_count']
		assert all(int(v) > 0 for v in row['per_class_train_voxel_counts'].values())
		assert all(int(v) > 0 for v in row['per_line_contributions'].values())
		validation_hashes.add(row['validation_mask_sha256'])
	assert len(validation_hashes) == 1
	assert not (output_root / 'datasets_v1').exists()


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
		target_calibration={'rule': TARGET_RULE},
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


def _write_cli_fixture(tmp_path: Path) -> F3SectionLayoutCalibrationConfig:
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
	config = f3_section_layout_calibration_config_from_mapping(
		_config_mapping(tmp_path)
	)
	lines = load_section_lines(config.line_inventory, config.segy_geometry_json)
	assert len(lines) == 10
	return config


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
