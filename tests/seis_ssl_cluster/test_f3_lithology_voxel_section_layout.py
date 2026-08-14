
from __future__ import annotations

import csv
import hashlib
import json
import weakref
from typing import TYPE_CHECKING

import numpy as np
import pytest

import seis_ssl_cluster.f3.lithology.voxel_section_layout as builder_module
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	CONTRACT_ARTIFACT_TYPE,
	CONTRACT_SCHEMA_VERSION,
	DECODER_SEED,
	FIXED_DECODER_CONTRACT,
	NESTING_SEMANTICS,
	PATCH_SIZE,
	STABLE_SELECTION_SEMANTICS,
	STATISTICAL_UNIT,
	VALIDATION_MASK_SEMANTICS,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_dataset import (
	F3SectionLayoutDatasetConfig,
	f3_lithology_voxel_section_layout_dataset_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	COUNTS_NAME,
	GRID_NAME,
	REQUIRED_CONDITION_FILES,
	TOKEN_NAME,
	build_f3_lithology_voxel_section_layout_datasets,
	inspect_f3_lithology_voxel_section_layout_datasets,
	validate_f3_lithology_voxel_section_layout_condition,
	validate_f3_lithology_voxel_section_layout_manifest,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_selection import (
	LayoutLines,
	SectionLine,
	SelectionPreview,
	preview_nested_selection,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_inspection_derives_exact_common_condition_matrix(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	inspection = inspect_f3_lithology_voxel_section_layout_datasets(config)
	assert len(inspection.conditions) == 15
	assert [(item.layout_id, item.data_size) for item in inspection.conditions] == [
		(f'layout_{layout:03d}', size)
		for layout in range(5)
		for size in ('small', 'medium', 'large')
	]
	for condition in inspection.conditions:
		assert not hasattr(condition, 'grid')
		expected = {'small': 1, 'medium': 2, 'large': 4}[condition.data_size]
		assert len(condition.active_inlines) == expected
		assert len(condition.active_crosslines) == expected
		assert condition.relative_count_error <= 0.1
		assert all(
			condition.per_class_train_voxel_counts[str(item)] > 0 for item in range(6)
		)
		assert all(value > 0 for value in condition.per_line_contributions.values())
		assert (
			sum(condition.per_line_contributions.values())
			== condition.actual_train_voxel_count
		)


def test_live_grids_use_only_partial_active_plane_footprints(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	inspection = inspect_f3_lithology_voxel_section_layout_datasets(config)
	result = build_f3_lithology_voxel_section_layout_datasets(config)
	for condition, root in zip(
		inspection.conditions, result.condition_roots, strict=True
	):
		grid = np.load(root / GRID_NAME, mmap_mode='r', allow_pickle=False)
		train = grid == 1
		active = np.zeros(train.shape, dtype=np.bool_)
		for line in condition.active_inlines:
			active[line - 100, :, :] = True
		for line in condition.active_crosslines:
			active[:, line - 200, :] = True
		assert not np.any(train & ~active)
		assert np.any(active & (inspection.canonical_grid == 1) & ~train)
		assert np.any((inspection.canonical_grid == 1) & ~train)
		assert np.array_equal(grid == 2, inspection.canonical_grid == 2)


def test_build_writes_exact_files_and_preserves_dense_labels(tmp_path: Path) -> None:
	config = _fixture(tmp_path)
	label_hash = file_sha256(config.source_label_volume)
	result = build_f3_lithology_voxel_section_layout_datasets(config)
	assert len(result.rows) == 15
	assert file_sha256(config.source_label_volume) == label_hash
	manifest = validate_f3_lithology_voxel_section_layout_manifest(result.manifest_json)
	assert manifest['condition_count'] == 15
	assert [(row['layout_id'], row['data_size']) for row in manifest['rows']] == [
		(f'layout_{layout:03d}', size)
		for layout in range(5)
		for size in ('small', 'medium', 'large')
	]
	validation_arrays = []
	for root in result.condition_roots:
		assert {path.name for path in root.iterdir()} == set(REQUIRED_CONDITION_FILES)
		validate_f3_lithology_voxel_section_layout_condition(root)
		grid = np.load(root / GRID_NAME, allow_pickle=False)
		tokens = np.load(root / TOKEN_NAME, allow_pickle=False)
		assert tokens.dtype == np.int64
		validation_arrays.append(grid == 2)
	assert all(
		np.array_equal(validation_arrays[0], mask) for mask in validation_arrays[1:]
	)


def test_train_masks_and_selected_tokens_are_nested_and_deterministic(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path)
	first = inspect_f3_lithology_voxel_section_layout_datasets(config)
	second = inspect_f3_lithology_voxel_section_layout_datasets(config)
	for left, right in zip(first.conditions, second.conditions, strict=True):
		assert np.array_equal(left.selected_token_xyz, right.selected_token_xyz)
		assert left.train_mask_sha256 == right.train_mask_sha256
		assert left.grid_array_sha256 == right.grid_array_sha256
	by_key = {(item.layout_id, item.data_size): item for item in first.conditions}
	for layout_id in (f'layout_{index:03d}' for index in range(5)):
		small = {
			tuple(row) for row in by_key[(layout_id, 'small')].selected_token_xyz
		}
		medium = {
			tuple(row) for row in by_key[(layout_id, 'medium')].selected_token_xyz
		}
		large = {
			tuple(row) for row in by_key[(layout_id, 'large')].selected_token_xyz
		}
		assert small < medium
		assert medium < large
	build = build_f3_lithology_voxel_section_layout_datasets(config)
	manifest_before = build.manifest_json.read_bytes()
	reused = build_f3_lithology_voxel_section_layout_datasets(config, only_missing=True)
	assert {row['action'] for row in reused.rows} == {'REUSED'}
	assert reused.manifest_json.read_bytes() == manifest_before


def test_inspection_is_compact_and_build_releases_each_materialized_grid(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _fixture(tmp_path)
	original = builder_module._materialize_condition_grid  # noqa: SLF001
	monkeypatch.setattr(
		builder_module,
		'_materialize_condition_grid',
		lambda *_args, **_kwargs: pytest.fail('inspection materialized a grid'),
	)
	inspection = inspect_f3_lithology_voxel_section_layout_datasets(config)
	assert len(inspection.conditions) == 15
	assert all(not hasattr(condition, 'grid') for condition in inspection.conditions)

	references: list[weakref.ReferenceType[np.ndarray]] = []

	def track_materialization(*args: object, **kwargs: object) -> np.ndarray:
		assert all(reference() is None for reference in references)
		grid = original(*args, **kwargs)  # type: ignore[arg-type]
		references.append(weakref.ref(grid))
		return grid

	monkeypatch.setattr(
		builder_module, '_materialize_condition_grid', track_materialization
	)
	result = build_f3_lithology_voxel_section_layout_datasets(config)
	assert len(result.rows) == 15
	assert len(references) == 15


def test_source_shape_dtype_hash_class_and_validation_drift_fail_closed(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path)
	labels = np.load(config.source_label_volume)
	np.save(config.source_label_volume, labels.astype(np.float32))
	with pytest.raises(TypeError, match='integer'):
		inspect_f3_lithology_voxel_section_layout_datasets(config)
	config = _fixture(tmp_path / 'hash')
	with config.png_label_inventory.open('a', encoding='utf-8') as handle:
		handle.write('\n')
	with pytest.raises(ValueError, match='SHA-256'):
		inspect_f3_lithology_voxel_section_layout_datasets(config)
	config = _fixture(tmp_path / 'validation')
	contract = json.loads(config.section_layout_contract.read_text())
	contract['validation_identity']['mask_sha256'] = 'drift'
	_write_json(config.section_layout_contract, contract)
	with pytest.raises(ValueError, match='validation identity drift'):
		inspect_f3_lithology_voxel_section_layout_datasets(config)


def test_staging_failure_leaves_no_partial_final(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _fixture(tmp_path)
	original = builder_module._write_condition_files  # noqa: SLF001
	calls = 0

	def fail_second(*args: object, **kwargs: object) -> None:
		nonlocal calls
		calls += 1
		if calls == 2:
			raise RuntimeError('synthetic staging failure')
		original(*args, **kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(builder_module, '_write_condition_files', fail_second)
	with pytest.raises(RuntimeError, match='synthetic staging failure'):
		build_f3_lithology_voxel_section_layout_datasets(config)
	assert not config.output_root.exists()


def test_stale_reuse_rejected_and_explicit_quarantine_rebuilds(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path)
	result = build_f3_lithology_voxel_section_layout_datasets(config)
	stale_root = result.condition_roots[0]
	(stale_root / COUNTS_NAME).unlink()
	with pytest.raises(ValueError, match='stale or partial'):
		build_f3_lithology_voxel_section_layout_datasets(config, only_missing=True)
	rebuilt = build_f3_lithology_voxel_section_layout_datasets(
		config, only_missing=True, quarantine_invalid=True
	)
	assert len(rebuilt.quarantines) == 2
	assert rebuilt.rows[0]['action'] == 'REBUILT_AFTER_QUARANTINE'
	assert all(path.exists() for path in rebuilt.quarantines)
	assert (stale_root / COUNTS_NAME).is_file()


def test_config_unknown_key_and_quarantine_without_reuse_rejected(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path)
	raw = _config_mapping(config)
	raw['inputs']['unknown'] = str(tmp_path / 'x')
	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_section_layout_dataset_config_from_mapping(raw)
	with pytest.raises(ValueError, match='requires --only-missing'):
		build_f3_lithology_voxel_section_layout_datasets(
			config, quarantine_invalid=True
		)


def _fixture(tmp_path: Path) -> F3SectionLayoutDatasetConfig:
	tmp_path.mkdir(parents=True, exist_ok=True)
	canonical_root = tmp_path / 'canonical'
	canonical_root.mkdir()
	shape = (32, 32, 8)
	labels = np.broadcast_to(np.arange(8, dtype=np.int16) % 6, shape).copy()
	grid = np.zeros(shape, dtype=np.uint8)
	for index in (0, 8, 16, 24, 30):
		grid[index, :, :] = 1
	for index in (0, 8, 16, 24):
		grid[:, index, :] = 1
	grid[:, 31, :] = 2
	label_path = tmp_path / 'labels.npy'
	valid_path = tmp_path / 'valid.npy'
	inventory_path = tmp_path / 'inventory.csv'
	geometry_path = tmp_path / 'geometry.json'
	class_path = tmp_path / 'class_info.json'
	contract_path = tmp_path / 'contract.json'
	layout_source = tmp_path / 'layout_lines.yaml'
	np.save(label_path, labels)
	np.save(valid_path, np.ones((4, 4, 1), dtype=np.bool_))
	np.save(canonical_root / GRID_NAME, grid)
	_write_inventory(inventory_path)
	_write_json(
		geometry_path,
		{
			'segy_files': {
				'label': {
					'cube_shape': list(shape),
					'iline_min': 100,
					'iline_max': 131,
					'xline_min': 200,
					'xline_max': 231,
				}
			}
		},
	)
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
	layout_source.write_text('synthetic layout source\n', encoding='utf-8')
	classes = read_f3_lithology_class_info(class_path)
	canonical_manifest = {
		'split_source': 'png_label_inventory',
		'split_unit': 'slice',
		'strategy': 'inventory_split_no_random_token_split',
		'no_random_split': True,
		'splits': {},
	}
	_write_json(canonical_root / 'split_manifest.json', canonical_manifest)
	(canonical_root / 'class_counts.csv').write_text(
		'split,class_id,class_name,count,fraction\n', encoding='utf-8'
	)
	(canonical_root / 'voxel_dataset_summary.md').write_text(
		'# synthetic canonical\n', encoding='utf-8'
	)
	metadata = {
		'artifact_type': 'f3_lithology_voxel_supervision',
		'schema_version': 1,
		'classes': [item.to_dict() for item in classes],
		'split_codes': {'unsupervised': 0, 'train': 1, 'validation': 2},
		'validation_precedence': True,
		'reference_embedding': {
			'patch_size': [8, 8, 8],
			'token_grid_shape': [4, 4, 1],
			'volume_shape_xyz': list(shape),
		},
		'label_volume': _identity(label_path),
		'inventory': _identity(inventory_path),
		'reference_valid_tokens': _identity(valid_path),
		'source_identities': {
			'class_info': _identity(class_path),
			'segy_geometry_json': _identity(geometry_path),
		},
		'outputs': {
			'supervision_split_grid': str(canonical_root / GRID_NAME),
		},
		'summary': {},
	}
	_write_json(canonical_root / 'voxel_dataset_metadata.json', metadata)
	lines = _section_lines()
	layouts = _layouts()
	targets = {'small': 120, 'medium': 332, 'large': 712}
	previews = tuple(
		preview
		for layout in layouts
		for preview in preview_nested_selection(
			layout,
			targets,
			grid,
			labels,
			lines,
			allowed_relative_error=0.1,
		)
	)
	validation = grid == 2
	contract = _section_layout_contract(
		layouts,
		targets,
		previews,
		allowed_relative_error=0.1,
		validation_identity={
			'mask_sha256': hashlib.sha256(
				np.ascontiguousarray(validation).tobytes()
			).hexdigest(),
			'voxel_count': int(np.count_nonzero(validation)),
			'source_path': str(canonical_root / GRID_NAME),
			'source_sha256': file_sha256(canonical_root / GRID_NAME),
			'unchanged_by_preview': True,
		},
		source_file_identities={
			'canonical_split_grid': _identity(canonical_root / GRID_NAME),
			'label_volume': _identity(label_path),
			'line_inventory': _identity(inventory_path),
			'segy_geometry_json': _identity(geometry_path),
			'layout_lines': _identity(layout_source),
		},
	)
	_write_json(contract_path, contract)
	return F3SectionLayoutDatasetConfig(
		section_layout_contract=contract_path,
		canonical_voxel_dataset=canonical_root,
		source_label_volume=label_path,
		png_label_inventory=inventory_path,
		segy_geometry_json=geometry_path,
		class_info=class_path,
		reference_valid_tokens=valid_path,
		output_root=tmp_path / 'output',
	)


def _section_layout_contract(  # noqa: PLR0913
	layouts: tuple[LayoutLines, ...],
	targets: dict[str, int],
	previews: tuple[SelectionPreview, ...],
	*,
	allowed_relative_error: float,
	validation_identity: dict[str, object],
	source_file_identities: dict[str, dict[str, str]],
) -> dict[str, object]:
	preview_by_key = {
		(preview.layout_id, preview.data_size): preview for preview in previews
	}
	contract_layouts = []
	for layout in layouts:
		sizes = {}
		for data_size in ('small', 'medium', 'large'):
			preview = preview_by_key[(layout.layout_id, data_size)]
			sizes[data_size] = {
				'inline_lines': list(preview.inline_lines),
				'crossline_lines': list(preview.crossline_lines),
				'target_train_voxel_count': preview.target_train_voxel_count,
				'preview_actual_train_voxel_count': preview.actual_train_voxel_count,
				'preview_count_error': preview.count_error,
				'preview_relative_count_error': preview.relative_count_error,
				'selected_token_xyz': [
					list(value) for value in preview.selected_token_xyz
				],
				'per_line_contributions': dict(preview.per_line_contributions),
				'per_class_voxel_counts': dict(preview.per_class_voxel_counts),
			}
		contract_layouts.append(
			{
				'layout_id': layout.layout_id,
				'ordered_inlines': list(layout.ordered_inlines),
				'ordered_crosslines': list(layout.ordered_crosslines),
				'sizes': sizes,
			}
		)
	return {
		'artifact_type': CONTRACT_ARTIFACT_TYPE,
		'schema_version': CONTRACT_SCHEMA_VERSION,
		'selection_semantics': STABLE_SELECTION_SEMANTICS,
		'stable_selection_semantics': STABLE_SELECTION_SEMANTICS,
		'statistical_unit': STATISTICAL_UNIT,
		'nesting_semantics': NESTING_SEMANTICS,
		'validation_mask_semantics': VALIDATION_MASK_SEMANTICS,
		'patch_size': list(PATCH_SIZE),
		'patch_size_xyz': list(PATCH_SIZE),
		'allowed_relative_error': allowed_relative_error,
		'target_train_voxel_counts': targets,
		'active_prefix_counts': {
			'small': {'inline': 1, 'crossline': 1},
			'medium': {'inline': 2, 'crossline': 2},
			'large': {'inline': 4, 'crossline': 4},
		},
		'decoder_seed': DECODER_SEED,
		'decoder': dict(FIXED_DECODER_CONTRACT),
		'layouts': contract_layouts,
		'validation_identity': validation_identity,
		'source_file_identities': source_file_identities,
	}


def _section_lines() -> tuple[SectionLine, ...]:
	return (
		tuple(
			SectionLine(
				'inline', 100 + index, index, is_validation_line=False
			)
			for index in (0, 8, 16, 24, 30)
		)
		+ tuple(
			SectionLine(
				'crossline', 200 + index, index, is_validation_line=False
			)
			for index in (0, 8, 16, 24)
		)
		+ (SectionLine('crossline', 231, 31, is_validation_line=True),)
	)


def _layouts() -> tuple[LayoutLines, ...]:
	inlines = (100, 108, 116, 124)
	crosslines = (200, 208, 216, 224)
	return tuple(
		LayoutLines(
			f'layout_{index:03d}',
			inlines[index:] + inlines[:index],
			crosslines[index:] + crosslines[:index],
		)
		for index in range(5)
	)


def _write_inventory(path: Path) -> None:
	with path.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=('relative_path', 'split', 'slice_type', 'slice_index'),
		)
		writer.writeheader()
		for line in _section_lines():
			writer.writerow(
				{
					'relative_path': f'{line.slice_type}_{line.slice_index}.png',
					'split': 'validation' if line.is_validation_line else 'train',
					'slice_type': line.slice_type,
					'slice_index': line.slice_index,
				}
			)


def _config_mapping(
	config: F3SectionLayoutDatasetConfig,
) -> dict[str, dict[str, str]]:
	return {
		'inputs': {
			'section_layout_contract': str(config.section_layout_contract),
			'canonical_voxel_dataset': str(config.canonical_voxel_dataset),
			'source_label_volume': str(config.source_label_volume),
			'png_label_inventory': str(config.png_label_inventory),
			'segy_geometry_json': str(config.segy_geometry_json),
			'class_info': str(config.class_info),
			'reference_valid_tokens': str(config.reference_valid_tokens),
		},
		'outputs': {'output_root': str(config.output_root)},
	}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
