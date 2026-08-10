"""Build model-independent F3 section-layout voxel supervision datasets."""
# ruff: noqa: CPY001

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	CONTRACT_ARTIFACT_TYPE,
	DATA_SIZES,
	LAYOUT_IDS,
	LINE_COUNTS,
	PATCH_SIZE,
	f3_lithology_voxel_section_layout_contract_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info
from seis_ssl_cluster.f3.lithology.voxel_section_layout_selection import (
	CLASS_IDS,
	SELECTION_SEMANTICS,
	LayoutLines,
	SectionLine,
	SelectionPreview,
	preview_nested_selection,
	replay_selected_teacher_mask,
)
from seis_ssl_cluster.f3.lithology.voxel_split import (
	TRAIN_VOXEL_SPLIT,
	UNSUPERVISED_VOXEL_SPLIT,
	VALIDATION_VOXEL_SPLIT,
)
from seis_ssl_cluster.f3.splits import (
	F3LineGeometry,
	load_f3_slice_split_records,
	read_f3_line_geometry,
	resolve_f3_slice_array_index,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

	from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_dataset import (
		F3SectionLayoutDatasetConfig,
	)

DATASET_MANIFEST_NAME = 'section_layout_dataset_manifest.json'
GRID_NAME = 'supervision_split_grid.npy'
TOKEN_NAME = 'selected_token_xyz.npy'  # noqa: S105
VOXEL_METADATA_NAME = 'voxel_dataset_metadata.json'
LAYOUT_METADATA_NAME = 'section_layout_metadata.json'
COUNTS_NAME = 'class_counts.csv'
SPLIT_MANIFEST_NAME = 'split_manifest.json'
SUMMARY_NAME = 'summary.md'
ARTIFACT_TYPE = 'f3_lithology_voxel_section_layout_dataset'
MANIFEST_ARTIFACT_TYPE = 'f3_lithology_voxel_section_layout_dataset_manifest'
SCHEMA_VERSION = 1
_STREAM_CHUNK_VOXELS = 1_048_576
REQUIRED_CONDITION_FILES = (
	GRID_NAME,
	TOKEN_NAME,
	VOXEL_METADATA_NAME,
	LAYOUT_METADATA_NAME,
	COUNTS_NAME,
	SPLIT_MANIFEST_NAME,
	SUMMARY_NAME,
)


@dataclass(frozen=True)
class _ConditionPlan:
	layout_id: str
	data_size: str
	output_dir: Path
	ordered_inlines: tuple[int, ...]
	ordered_crosslines: tuple[int, ...]
	active_inlines: tuple[int, ...]
	active_crosslines: tuple[int, ...]
	target_train_voxel_count: int
	actual_train_voxel_count: int
	relative_count_error: float
	selected_token_xyz: NDArray[np.int64]
	per_line_contributions: Mapping[str, int]
	per_class_train_voxel_counts: Mapping[str, int]
	per_class_validation_voxel_counts: Mapping[str, int]
	train_mask_sha256: str
	validation_mask_sha256: str
	grid_array_sha256: str
	parent_size: str | None


@dataclass(frozen=True)
class _Inspection:
	config: F3SectionLayoutDatasetConfig
	contract_payload: Mapping[str, object]
	canonical_metadata: Mapping[str, object]
	canonical_grid: NDArray[np.integer]
	label_volume: NDArray[np.integer]
	valid_tokens: NDArray[np.bool_]
	geometry: F3LineGeometry
	class_ids: tuple[int, ...]
	class_names: tuple[str, ...]
	conditions: tuple[_ConditionPlan, ...]
	line_array_indices: Mapping[tuple[str, int], int]
	source_identities: Mapping[str, Mapping[str, str]]
	validation_mask_sha256: str
	validation_voxel_count: int


@dataclass(frozen=True)
class _BuildResult:
	manifest_json: Path
	condition_roots: tuple[Path, ...]
	rows: tuple[Mapping[str, object], ...]
	quarantines: tuple[Path, ...]


def inspect_f3_lithology_voxel_section_layout_datasets(  # noqa: C901, PLR0912, PLR0915
	config: F3SectionLayoutDatasetConfig,
) -> _Inspection:
	"""Validate every source and derive the exact ordered 15-condition plan."""
	paths = _input_paths(config)
	for label, path in paths.items():
		if label == 'canonical_voxel_dataset':
			if not path.is_dir():
				raise FileNotFoundError(f'missing {label}: {path}')
		elif not path.is_file():
			raise FileNotFoundError(f'missing {label}: {path}')
	canonical_files = {
		'canonical_split_grid': config.canonical_voxel_dataset / GRID_NAME,
		'canonical_voxel_metadata': config.canonical_voxel_dataset
		/ VOXEL_METADATA_NAME,
		'canonical_class_counts': config.canonical_voxel_dataset / COUNTS_NAME,
		'canonical_split_manifest': config.canonical_voxel_dataset
		/ SPLIT_MANIFEST_NAME,
	}
	for label, path in canonical_files.items():
		if not path.is_file():
			raise FileNotFoundError(f'missing {label}: {path}')
	contract_payload = _read_json(config.section_layout_contract)
	canonical_metadata = _read_json(canonical_files['canonical_voxel_metadata'])
	_validate_canonical_metadata(canonical_metadata)
	grid = _load_integer_volume(
		canonical_files['canonical_split_grid'], label='canonical split grid'
	)
	labels = _load_integer_volume(config.source_label_volume, label='label volume')
	if grid.shape != labels.shape:
		raise ValueError('canonical split grid and label volume shape mismatch')
	if grid.dtype != np.dtype(np.uint8):
		raise TypeError('canonical split grid dtype must be uint8')
	_validate_split_codes(grid)
	geometry = read_f3_line_geometry(config.segy_geometry_json)
	if tuple(geometry.shape_xyz) != tuple(grid.shape):
		raise ValueError('SEGY geometry shape does not match canonical grid')
	classes = read_f3_lithology_class_info(config.class_info)
	class_ids = tuple(item.class_id for item in classes)
	class_names = tuple(item.class_name for item in classes)
	if class_ids != CLASS_IDS:
		raise ValueError(f'class order must be exactly {list(CLASS_IDS)!r}')
	if [item.to_dict() for item in classes] != canonical_metadata.get('classes'):
		raise ValueError('class info/order differs from canonical voxel metadata')
	reference = _mapping(
		canonical_metadata.get('reference_embedding'), 'reference_embedding'
	)
	patch = _positive_triplet(reference.get('patch_size'), 'reference.patch_size')
	if patch != PATCH_SIZE:
		raise ValueError(f'canonical patch size must be exactly {list(PATCH_SIZE)!r}')
	volume_shape = _positive_triplet(
		reference.get('volume_shape_xyz'), 'reference.volume_shape_xyz'
	)
	token_shape = _positive_triplet(
		reference.get('token_grid_shape'), 'reference.token_grid_shape'
	)
	if volume_shape != tuple(grid.shape):
		raise ValueError('canonical metadata volume shape differs from grid')
	expected_token_shape = tuple(
		(size + step - 1) // step
		for size, step in zip(volume_shape, patch, strict=True)
	)
	if token_shape != expected_token_shape:
		raise ValueError('canonical token geometry is inconsistent')
	valid_tokens = np.load(
		config.reference_valid_tokens, mmap_mode='r', allow_pickle=False
	)
	if valid_tokens.dtype != np.dtype(np.bool_):
		raise TypeError('reference valid-token mask dtype must be bool')
	if tuple(valid_tokens.shape) != token_shape:
		raise ValueError('reference valid-token mask shape mismatch')
	_validate_canonical_source_identities(
		config,
		canonical_metadata=canonical_metadata,
		canonical_grid=canonical_files['canonical_split_grid'],
	)
	records = load_f3_slice_split_records(config.png_label_inventory)
	inventory_rows = tuple(record.to_dict() for record in records)
	contract = f3_lithology_voxel_section_layout_contract_from_mapping(
		contract_payload, line_inventory=inventory_rows
	)
	_validate_contract_source_identities(
		contract_payload,
		config=config,
		canonical_grid=canonical_files['canonical_split_grid'],
	)
	validation_hash, validation_count = _split_mask_identity(
		grid, VALIDATION_VOXEL_SPLIT
	)
	_validate_contract_validation_identity(
		contract_payload,
		mask_sha256=validation_hash,
		voxel_count=validation_count,
		canonical_grid=canonical_files['canonical_split_grid'],
	)
	_validate_reference_validity(grid, valid_tokens=valid_tokens, patch=patch)
	known = np.asarray(class_ids, dtype=labels.dtype)
	if np.any((grid > UNSUPERVISED_VOXEL_SPLIT) & ~np.isin(labels, known)):
		raise ValueError(
			'canonical supervised grid contains a label outside class order'
		)
	lines = tuple(
		SectionLine(
			record.slice_type,
			record.slice_index,
			resolve_f3_slice_array_index(record, geometry),
			record.split == 'validation',
		)
		for record in records
	)
	line_array_indices = {line.key: line.array_index for line in lines}
	targets = {
		size: contract.layouts[0].size_by_name[size].target_train_voxel_count
		for size in DATA_SIZES
	}
	validation_class_counts = _split_class_counts(
		labels,
		grid,
		split_code=VALIDATION_VOXEL_SPLIT,
		class_ids=class_ids,
	)
	conditions: list[_ConditionPlan] = []
	for layout in contract.layouts:
		layout_lines = LayoutLines(
			layout.layout_id,
			layout.size_by_name['large'].inline_lines,
			layout.size_by_name['large'].crossline_lines,
		)
		previews = preview_nested_selection(
			layout_lines,
			targets,
			grid,
			labels,
			lines,
			patch_size_xyz=patch,
			allowed_relative_error=contract.allowed_relative_error,
		)
		for preview in previews:
			_validate_preview_matches_contract(contract_payload, preview=preview)
			conditions.append(
				_condition_plan_from_preview(
					preview,
					layout=layout_lines,
					canonical_grid=grid,
					labels=labels,
					valid_tokens=cast('NDArray[np.bool_]', valid_tokens),
					class_ids=class_ids,
					line_array_indices=line_array_indices,
					output_root=config.output_root,
					allowed_relative_error=contract.allowed_relative_error,
					validation_mask_sha256=validation_hash,
					validation_class_counts=validation_class_counts,
				)
			)
	_validate_condition_plan_matrix(conditions)
	source_identities = {
		'section_layout_contract': _identity(config.section_layout_contract),
		'canonical_voxel_dataset_metadata': _identity(
			canonical_files['canonical_voxel_metadata']
		),
		'canonical_split_grid': _identity(canonical_files['canonical_split_grid']),
		'canonical_class_counts': _identity(canonical_files['canonical_class_counts']),
		'canonical_split_manifest': _identity(
			canonical_files['canonical_split_manifest']
		),
		'label_volume': _identity(config.source_label_volume),
		'png_label_inventory': _identity(config.png_label_inventory),
		'segy_geometry_json': _identity(config.segy_geometry_json),
		'class_info': _identity(config.class_info),
		'reference_valid_tokens': _identity(config.reference_valid_tokens),
	}
	return _Inspection(
		config=config,
		contract_payload=contract_payload,
		canonical_metadata=canonical_metadata,
		canonical_grid=grid,
		label_volume=labels,
		valid_tokens=cast('NDArray[np.bool_]', valid_tokens),
		geometry=geometry,
		class_ids=class_ids,
		class_names=class_names,
		conditions=tuple(conditions),
		line_array_indices=line_array_indices,
		source_identities=source_identities,
		validation_mask_sha256=validation_hash,
		validation_voxel_count=validation_count,
	)


def build_f3_lithology_voxel_section_layout_datasets(
	config: F3SectionLayoutDatasetConfig,
	*,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> _BuildResult:
	"""Stage, reload, and commit all 15 conditions or strictly reuse them."""
	if quarantine_invalid and not only_missing:
		raise ValueError('--quarantine-invalid requires --only-missing')
	inspection = inspect_f3_lithology_voxel_section_layout_datasets(config)
	if not only_missing:
		return _build_new_suite(inspection)
	return _build_only_missing(inspection, quarantine_invalid=quarantine_invalid)


def validate_f3_lithology_voxel_section_layout_condition(  # noqa: C901
	root: str | Path,
	*,
	inspection: _Inspection | None = None,
	condition: _ConditionPlan | None = None,
) -> Mapping[str, object]:
	"""Reload and validate one exact seven-file condition artifact."""
	condition_root = Path(root)
	actual_names = (
		{path.name for path in condition_root.iterdir()}
		if condition_root.is_dir()
		else set()
	)
	if actual_names != set(REQUIRED_CONDITION_FILES):
		missing = sorted(set(REQUIRED_CONDITION_FILES) - actual_names)
		extra = sorted(actual_names - set(REQUIRED_CONDITION_FILES))
		raise FileNotFoundError(
			f'condition file inventory mismatch; missing={missing!r}, extra={extra!r}'
		)
	metadata = _read_json(condition_root / LAYOUT_METADATA_NAME)
	if (
		metadata.get('artifact_type') != ARTIFACT_TYPE
		or metadata.get('schema_version') != SCHEMA_VERSION
	):
		raise ValueError('invalid section-layout condition metadata schema')
	outputs = _mapping(metadata.get('outputs'), 'section-layout outputs')
	expected_output_names = set(REQUIRED_CONDITION_FILES) - {LAYOUT_METADATA_NAME}
	if set(outputs) != expected_output_names:
		raise ValueError('section-layout condition output inventory mismatch')
	for name in sorted(expected_output_names):
		_validate_output_identity(
			outputs[name],
			condition_root / name,
			recorded_root=condition.output_dir if condition else condition_root,
		)
	grid = np.load(condition_root / GRID_NAME, mmap_mode='r', allow_pickle=False)
	tokens = np.load(condition_root / TOKEN_NAME, mmap_mode='r', allow_pickle=False)
	_validate_token_array(tokens)
	identity = _mapping(metadata.get('identity'), 'section-layout identity')
	if tuple(grid.shape) != _positive_triplet(
		identity.get('volume_shape_xyz'), 'identity.volume_shape_xyz'
	):
		raise ValueError('condition grid volume shape mismatch')
	if not np.issubdtype(grid.dtype, np.integer) or grid.dtype == np.dtype(np.bool_):
		raise TypeError('condition grid must use an integer dtype')
	_validate_split_codes(cast('NDArray[np.integer]', grid))
	train_mask_sha256, actual_train_voxel_count = _split_mask_identity(
		cast('NDArray[np.integer]', grid), TRAIN_VOXEL_SPLIT
	)
	validation_mask_sha256, validation_voxel_count = _split_mask_identity(
		cast('NDArray[np.integer]', grid), VALIDATION_VOXEL_SPLIT
	)
	checks = {
		'grid_array_sha256': _array_sha256(grid),
		'train_mask_sha256': train_mask_sha256,
		'validation_mask_sha256': validation_mask_sha256,
		'actual_train_voxel_count': actual_train_voxel_count,
		'validation_voxel_count': validation_voxel_count,
		'selected_token_count': int(tokens.shape[0]),
		'selected_token_identity_sha256': _array_sha256(tokens),
	}
	for key, value in checks.items():
		if identity.get(key) != value:
			raise ValueError(f'condition identity mismatch: {key}')
	if not (condition_root / SUMMARY_NAME).read_text(encoding='utf-8').strip():
		raise ValueError('condition summary is empty')
	_read_json(condition_root / VOXEL_METADATA_NAME)
	_read_json(condition_root / SPLIT_MANIFEST_NAME)
	_validate_class_counts_csv(condition_root / COUNTS_NAME)
	if inspection is not None or condition is not None:
		if inspection is None or condition is None:
			raise ValueError(
				'deep condition validation requires inspection and condition'
			)
		_validate_committed_condition(
			condition_root,
			metadata=metadata,
			grid=cast('NDArray[np.integer]', grid),
			tokens=cast('NDArray[np.integer]', tokens),
			inspection=inspection,
			condition=condition,
		)
	return _condition_manifest_row(
		condition_root,
		metadata=metadata,
		recorded_root=condition.output_dir if condition else condition_root,
	)


def validate_f3_lithology_voxel_section_layout_manifest(  # noqa: C901, PLR0912
	source: str | Path | Mapping[str, object],
	*,
	inspection: _Inspection | None = None,
) -> Mapping[str, object]:
	"""Validate exact row identities and canonical layout/size ordering."""
	payload = source if isinstance(source, Mapping) else _read_json(Path(source))
	expected_top_keys = {
		'artifact_type',
		'schema_version',
		'condition_count',
		'row_order',
		'selection_semantics',
		'statistical_unit',
		'source_identities',
		'validation_identity',
		'rows',
	}
	if set(payload) != expected_top_keys:
		raise ValueError('section-layout dataset manifest key inventory mismatch')
	if (
		payload.get('artifact_type') != MANIFEST_ARTIFACT_TYPE
		or payload.get('schema_version') != SCHEMA_VERSION
	):
		raise ValueError('invalid section-layout dataset manifest schema')
	if payload.get('condition_count') != 15:
		raise ValueError('section-layout dataset manifest condition_count must be 15')
	if payload.get('row_order') != 'layout_id_then_small_medium_large':
		raise ValueError('section-layout dataset manifest row-order drift')
	if payload.get('selection_semantics') != SELECTION_SEMANTICS:
		raise ValueError('section-layout dataset manifest selection semantics drift')
	if payload.get('statistical_unit') != 'layout_id':
		raise ValueError('section-layout dataset manifest statistical unit drift')
	validation_identity = _mapping(
		payload.get('validation_identity'), 'manifest validation identity'
	)
	if set(validation_identity) != {'mask_sha256', 'voxel_count'}:
		raise ValueError('manifest validation identity key inventory mismatch')
	if not _is_nonempty_string(validation_identity.get('mask_sha256')):
		raise TypeError('manifest validation mask SHA-256 must be non-empty')
	_positive_integer(
		validation_identity.get('voxel_count'),
		'manifest validation voxel_count',
	)
	_validate_manifest_sources(payload.get('source_identities'))
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError('section-layout dataset manifest rows must be a list')
	if len(rows) != len(LAYOUT_IDS) * len(DATA_SIZES):
		raise ValueError('section-layout dataset manifest must contain exactly 15 rows')
	expected_order = tuple(
		(layout_id, size) for layout_id in LAYOUT_IDS for size in DATA_SIZES
	)
	actual_order: list[tuple[object, object]] = []
	for index, row in enumerate(rows):
		if not isinstance(row, Mapping):
			raise TypeError(f'manifest row {index} must be a mapping')
		_validate_manifest_row(
			row,
			index=index,
			validation_mask_sha256=cast('str', validation_identity['mask_sha256']),
		)
		actual_order.append((row.get('layout_id'), row.get('data_size')))
	if tuple(actual_order) != expected_order:
		raise ValueError('manifest rows must be ordered by layout ID then data size')
	if len(set(actual_order)) != len(actual_order):
		raise ValueError('manifest condition rows must be unique')
	if inspection is not None:
		expected = _manifest_payload(
			inspection,
			[
				_condition_manifest_row(
					condition.output_dir,
					metadata=_read_json(condition.output_dir / LAYOUT_METADATA_NAME),
					recorded_root=condition.output_dir,
				)
				for condition in inspection.conditions
			],
		)
		if payload != expected:
			raise ValueError('section-layout dataset manifest content mismatch')
	return cast('Mapping[str, object]', payload)


def _condition_plan_from_preview(  # noqa: PLR0913
	preview: SelectionPreview,
	*,
	layout: LayoutLines,
	canonical_grid: NDArray[np.integer],
	labels: NDArray[np.integer],
	valid_tokens: NDArray[np.bool_],
	class_ids: tuple[int, ...],
	line_array_indices: Mapping[tuple[str, int], int],
	output_root: Path,
	allowed_relative_error: float,
	validation_mask_sha256: str,
	validation_class_counts: Mapping[str, int],
) -> _ConditionPlan:
	tokens = np.asarray(preview.selected_token_xyz, dtype=np.int64)
	_validate_token_array(tokens)
	if np.any(tokens >= np.asarray(valid_tokens.shape, dtype=np.int64)):
		raise ValueError('selected token coordinate lies outside token grid')
	if np.any(~valid_tokens[tuple(tokens.T)]):
		raise ValueError('selected token coordinate is invalid in reference mask')
	selected_flat = _validate_preview_flat_voxel_indices(
		preview,
		tokens=tokens,
		canonical_grid=canonical_grid,
		labels=labels,
		valid_token_shape=cast('tuple[int, int, int]', valid_tokens.shape),
		class_ids=class_ids,
		line_array_indices=line_array_indices,
	)
	actual = int(selected_flat.size)
	if actual != preview.actual_train_voxel_count:
		raise ValueError('teacher voxel count differs from selection preview')
	if actual == _count_split_code(canonical_grid, TRAIN_VOXEL_SPLIT):
		raise ValueError('section-layout condition must not select full train grid')
	relative = (
		abs(actual - preview.target_train_voxel_count)
		/ preview.target_train_voxel_count
	)
	if relative > allowed_relative_error + 1e-15:
		raise ValueError('live train voxel count exceeds contract tolerance')
	train_counts = dict(preview.per_class_voxel_counts)
	if any(train_counts[str(class_id)] <= 0 for class_id in class_ids):
		raise ValueError('selection preview is missing a required class')
	if sum(train_counts.values()) != actual:
		raise ValueError('selection preview class counts do not sum to actual count')
	if sum(preview.per_line_contributions.values()) != actual:
		raise ValueError('per-line contributions double-count intersections')
	if any(value <= 0 for value in preview.per_line_contributions.values()):
		raise ValueError('an active line contributes zero teacher voxels')
	data_index = DATA_SIZES.index(preview.data_size)
	parent_size = None if data_index == 0 else DATA_SIZES[data_index - 1]
	train_hash, grid_hash = _planned_grid_identities(
		canonical_grid, selected_train_flat_indices=selected_flat
	)
	return _ConditionPlan(
		layout_id=preview.layout_id,
		data_size=preview.data_size,
		output_dir=(
			output_root
			/ 'datasets'
			/ f'layout={preview.layout_id}'
			/ f'size={preview.data_size}'
			/ 'voxel_supervision'
		),
		ordered_inlines=layout.ordered_inlines,
		ordered_crosslines=layout.ordered_crosslines,
		active_inlines=preview.inline_lines,
		active_crosslines=preview.crossline_lines,
		target_train_voxel_count=preview.target_train_voxel_count,
		actual_train_voxel_count=actual,
		relative_count_error=relative,
		selected_token_xyz=tokens,
		per_line_contributions=dict(preview.per_line_contributions),
		per_class_train_voxel_counts=train_counts,
		per_class_validation_voxel_counts=dict(validation_class_counts),
		train_mask_sha256=train_hash,
		validation_mask_sha256=validation_mask_sha256,
		grid_array_sha256=grid_hash,
		parent_size=parent_size,
	)


def _validate_condition_plan_matrix(
	conditions: Sequence[_ConditionPlan],
) -> None:
	expected = tuple((layout, size) for layout in LAYOUT_IDS for size in DATA_SIZES)
	actual = tuple((item.layout_id, item.data_size) for item in conditions)
	if actual != expected:
		raise ValueError('conditions must be the exact ordered 5 by 3 matrix')
	by_key = {(item.layout_id, item.data_size): item for item in conditions}
	for layout_id in LAYOUT_IDS:
		small, medium, large = (by_key[(layout_id, size)] for size in DATA_SIZES)
		if (
			medium.active_inlines[: len(small.active_inlines)]
			!= small.active_inlines
			or large.active_inlines[: len(medium.active_inlines)]
			!= medium.active_inlines
			or medium.active_crosslines[: len(small.active_crosslines)]
			!= small.active_crosslines
			or large.active_crosslines[: len(medium.active_crosslines)]
			!= medium.active_crosslines
		):
			raise ValueError(f'{layout_id} active line prefixes are not nested')
		if not _sorted_token_rows_are_subset(
			small.selected_token_xyz, medium.selected_token_xyz
		) or not _sorted_token_rows_are_subset(
			medium.selected_token_xyz, large.selected_token_xyz
		):
			raise ValueError(f'{layout_id} selected token plans are not nested')


def _build_new_suite(inspection: _Inspection) -> _BuildResult:
	root = inspection.config.output_root
	if root.exists():
		raise FileExistsError(f'refusing existing section-layout output: {root}')
	root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(tempfile.mkdtemp(prefix=f'.{root.name}.staging-', dir=root.parent))
	try:
		rows = []
		for condition in inspection.conditions:
			stage_root = _stage_condition_root(staging, root, condition.output_dir)
			grid = _materialize_condition_grid(condition, inspection=inspection)
			try:
				_write_condition_files(
					stage_root,
					recorded_root=condition.output_dir,
					condition=condition,
					grid=grid,
					inspection=inspection,
				)
			finally:
				del grid
			rows.append(
				validate_f3_lithology_voxel_section_layout_condition(
					stage_root, inspection=inspection, condition=condition
				)
			)
		payload = _manifest_payload(inspection, rows)
		_write_json(staging / DATASET_MANIFEST_NAME, payload)
		validate_f3_lithology_voxel_section_layout_manifest(
			staging / DATASET_MANIFEST_NAME
		)
		staging.replace(root)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	return _BuildResult(
		manifest_json=root / DATASET_MANIFEST_NAME,
		condition_roots=tuple(item.output_dir for item in inspection.conditions),
		rows=tuple({**row, 'action': 'NEW'} for row in rows),
		quarantines=(),
	)


def _build_only_missing(  # noqa: C901, PLR0912, PLR0915
	inspection: _Inspection, *, quarantine_invalid: bool
) -> _BuildResult:
	root = inspection.config.output_root
	manifest_path = root / DATASET_MANIFEST_NAME
	statuses: list[
		tuple[_ConditionPlan, Mapping[str, object] | None, BaseException | None]
	] = []
	for condition in inspection.conditions:
		if not condition.output_dir.exists():
			statuses.append((condition, None, None))
			continue
		try:
			row = validate_f3_lithology_voxel_section_layout_condition(
				condition.output_dir, inspection=inspection, condition=condition
			)
		except (OSError, TypeError, ValueError) as error:
			statuses.append((condition, None, error))
		else:
			statuses.append((condition, row, None))
	manifest_error: BaseException | None = None
	if manifest_path.exists():
		try:
			_validate_existing_manifest(
				manifest_path, inspection=inspection, statuses=statuses
			)
		except (OSError, TypeError, ValueError) as error:
			manifest_error = error
	invalid = [item for item in statuses if item[2] is not None]
	if (invalid or manifest_error is not None) and not quarantine_invalid:
		error = invalid[0][2] if invalid else manifest_error
		raise ValueError(f'stale or partial section-layout output: {error}')
	quarantines: list[Path] = []
	if manifest_error is not None:
		quarantines.append(_quarantine(manifest_path, reason='manifest_invalid'))
	for condition, _row, error in invalid:
		quarantines.append(
			_quarantine(condition.output_dir, reason=type(error).__name__)
		)
	root.mkdir(parents=True, exist_ok=True)
	rows: list[Mapping[str, object]] = []
	for condition, row, error in statuses:
		if row is not None and error is None:
			rows.append(row)
			continue
		condition.output_dir.parent.mkdir(parents=True, exist_ok=True)
		staging = Path(
			tempfile.mkdtemp(
				prefix=f'.{condition.output_dir.name}.staging-',
				dir=condition.output_dir.parent,
			)
		)
		try:
			grid = _materialize_condition_grid(condition, inspection=inspection)
			try:
				_write_condition_files(
					staging,
					recorded_root=condition.output_dir,
					condition=condition,
					grid=grid,
					inspection=inspection,
				)
			finally:
				del grid
			built_row = validate_f3_lithology_voxel_section_layout_condition(
				staging, inspection=inspection, condition=condition
			)
			staging.replace(condition.output_dir)
		except BaseException:
			shutil.rmtree(staging, ignore_errors=True)
			raise
		rows.append(built_row)
	payload = _manifest_payload(inspection, rows)
	if manifest_path.exists():
		current = _read_json(manifest_path)
		if current != payload:
			if not quarantine_invalid:
				raise ValueError('existing suite manifest is stale')
			quarantines.append(_quarantine(manifest_path, reason='manifest_stale'))
	if not manifest_path.exists():
		_write_json_atomic(manifest_path, payload)
	validate_f3_lithology_voxel_section_layout_manifest(
		manifest_path, inspection=inspection
	)
	actions = []
	for (_condition, previous, error), row in zip(statuses, rows, strict=True):
		if previous is not None:
			action = 'REUSED'
		elif error is not None:
			action = 'REBUILT_AFTER_QUARANTINE'
		else:
			action = 'NEW'
		actions.append({**row, 'action': action})
	return _BuildResult(
		manifest_json=manifest_path,
		condition_roots=tuple(item.output_dir for item in inspection.conditions),
		rows=tuple(actions),
		quarantines=tuple(quarantines),
	)


def _write_condition_files(
	root: Path,
	*,
	recorded_root: Path,
	condition: _ConditionPlan,
	grid: NDArray[np.integer],
	inspection: _Inspection,
) -> None:
	if root.exists() and any(root.iterdir()):
		raise FileExistsError(f'non-empty staging condition root: {root}')
	root.mkdir(parents=True, exist_ok=True)
	np.save(root / GRID_NAME, grid, allow_pickle=False)
	np.save(root / TOKEN_NAME, condition.selected_token_xyz, allow_pickle=False)
	shutil.copyfile(
		Path(inspection.source_identities['canonical_split_manifest']['path']),
		root / SPLIT_MANIFEST_NAME,
	)
	_write_class_counts(root / COUNTS_NAME, condition=condition, inspection=inspection)
	voxel_metadata = _voxel_metadata(
		condition, inspection=inspection, recorded_root=recorded_root
	)
	_write_json(root / VOXEL_METADATA_NAME, voxel_metadata)
	(root / SUMMARY_NAME).write_text(_render_summary(condition), encoding='utf-8')
	metadata = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'identity': _condition_identity(condition, inspection=inspection),
		'layout_id': condition.layout_id,
		'data_size': condition.data_size,
		'ordered_lines': {
			'inline': list(condition.ordered_inlines),
			'crossline': list(condition.ordered_crosslines),
		},
		'active_lines': {
			'inline': list(condition.active_inlines),
			'crossline': list(condition.active_crosslines),
		},
		'target_train_voxel_count': condition.target_train_voxel_count,
		'actual_train_voxel_count': condition.actual_train_voxel_count,
		'relative_count_error': condition.relative_count_error,
		'selected_token_count': int(condition.selected_token_xyz.shape[0]),
		'selected_token_identity_sha256': _array_sha256(condition.selected_token_xyz),
		'per_line_contributions': dict(condition.per_line_contributions),
		'per_class_counts': dict(condition.per_class_train_voxel_counts),
		'train_mask_identity': {
			'sha256': condition.train_mask_sha256,
			'voxel_count': condition.actual_train_voxel_count,
		},
		'validation_mask_identity': {
			'sha256': condition.validation_mask_sha256,
			'voxel_count': inspection.validation_voxel_count,
		},
		'parent_size': condition.parent_size,
		'source_identities': {
			key: dict(value) for key, value in inspection.source_identities.items()
		},
		'selection_semantics': SELECTION_SEMANTICS,
		'outputs': {
			name: _identity(root / name, recorded_path=recorded_root / name)
			for name in REQUIRED_CONDITION_FILES
			if name != LAYOUT_METADATA_NAME
		},
	}
	_write_json(root / LAYOUT_METADATA_NAME, metadata)


def _validate_committed_condition(  # noqa: PLR0913
	root: Path,
	*,
	metadata: Mapping[str, object],
	grid: NDArray[np.integer],
	tokens: NDArray[np.integer],
	inspection: _Inspection,
	condition: _ConditionPlan,
) -> None:
	_validate_materialized_grid(grid, condition=condition, inspection=inspection)
	if not np.array_equal(tokens, condition.selected_token_xyz):
		raise ValueError('committed selected token coordinates mismatch')
	if metadata.get('identity') != _condition_identity(
		condition, inspection=inspection
	):
		raise ValueError('committed condition identity mismatch')
	if metadata.get('source_identities') != inspection.source_identities:
		raise ValueError('committed source identities mismatch')
	if metadata.get('selection_semantics') != SELECTION_SEMANTICS:
		raise ValueError('committed selection semantics mismatch')
	if _read_json(root / VOXEL_METADATA_NAME) != _voxel_metadata(
		condition, inspection=inspection, recorded_root=condition.output_dir
	):
		raise ValueError('committed voxel metadata content mismatch')
	_validate_condition_class_counts(
		root / COUNTS_NAME, condition=condition, inspection=inspection
	)
	if (
		file_sha256(root / SPLIT_MANIFEST_NAME)
		!= inspection.source_identities['canonical_split_manifest']['sha256']
	):
		raise ValueError('condition split manifest differs from canonical source')
	if (root / SUMMARY_NAME).read_text(encoding='utf-8') != _render_summary(condition):
		raise ValueError('committed condition summary mismatch')


def _condition_identity(
	condition: _ConditionPlan, *, inspection: _Inspection
) -> dict[str, object]:
	return {
		'layout_id': condition.layout_id,
		'data_size': condition.data_size,
		'parent_size': condition.parent_size,
		'patch_size_xyz': list(PATCH_SIZE),
		'volume_shape_xyz': list(inspection.canonical_grid.shape),
		'class_order': list(inspection.class_ids),
		'target_train_voxel_count': condition.target_train_voxel_count,
		'actual_train_voxel_count': condition.actual_train_voxel_count,
		'relative_count_error': condition.relative_count_error,
		'selected_token_count': int(condition.selected_token_xyz.shape[0]),
		'selected_token_identity_sha256': _array_sha256(condition.selected_token_xyz),
		'train_mask_sha256': condition.train_mask_sha256,
		'validation_mask_sha256': condition.validation_mask_sha256,
		'validation_voxel_count': inspection.validation_voxel_count,
		'grid_array_sha256': condition.grid_array_sha256,
		'per_line_contributions': dict(condition.per_line_contributions),
		'per_class_train_voxel_counts': dict(condition.per_class_train_voxel_counts),
		'per_class_validation_voxel_counts': dict(
			condition.per_class_validation_voxel_counts
		),
	}


def _voxel_metadata(
	condition: _ConditionPlan, *, inspection: _Inspection, recorded_root: Path
) -> dict[str, object]:
	payload = json.loads(json.dumps(inspection.canonical_metadata))
	payload['outputs'] = {
		'supervision_split_grid': str(recorded_root / GRID_NAME),
		'metadata_json': str(recorded_root / VOXEL_METADATA_NAME),
		'class_counts_csv': str(recorded_root / COUNTS_NAME),
		'split_manifest_json': str(recorded_root / SPLIT_MANIFEST_NAME),
		'summary_markdown': str(recorded_root / SUMMARY_NAME),
	}
	payload['summary'] = {
		'final_train_voxels': condition.actual_train_voxel_count,
		'final_validation_voxels': inspection.validation_voxel_count,
		'selected_token_count': int(condition.selected_token_xyz.shape[0]),
	}
	payload['section_layout'] = {
		'layout_id': condition.layout_id,
		'data_size': condition.data_size,
		'parent_size': condition.parent_size,
		'selection_semantics': SELECTION_SEMANTICS,
		'dense_voxel_labels_preserved': True,
		'partial_active_plane_footprints_only': True,
		'validation_reuse': 'canonical_validation_bitwise',
	}
	return cast('dict[str, object]', payload)


def _condition_manifest_row(
	root: Path, *, metadata: Mapping[str, object], recorded_root: Path
) -> dict[str, object]:
	identity = _mapping(metadata.get('identity'), 'condition identity')
	return {
		'layout_id': identity['layout_id'],
		'data_size': identity['data_size'],
		'parent_size': identity['parent_size'],
		'voxel_dataset_root': str(recorded_root),
		'target_train_voxel_count': identity['target_train_voxel_count'],
		'actual_train_voxel_count': identity['actual_train_voxel_count'],
		'relative_count_error': identity['relative_count_error'],
		'selected_token_count': identity['selected_token_count'],
		'selected_token_identity_sha256': identity['selected_token_identity_sha256'],
		'train_mask_sha256': identity['train_mask_sha256'],
		'validation_mask_sha256': identity['validation_mask_sha256'],
		'per_line_contributions': identity['per_line_contributions'],
		'per_class_train_voxel_counts': identity['per_class_train_voxel_counts'],
		'outputs': {
			name: _identity(root / name, recorded_path=recorded_root / name)
			for name in REQUIRED_CONDITION_FILES
		},
	}


def _manifest_payload(
	inspection: _Inspection, rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
	return {
		'artifact_type': MANIFEST_ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'condition_count': len(rows),
		'row_order': 'layout_id_then_small_medium_large',
		'selection_semantics': SELECTION_SEMANTICS,
		'statistical_unit': 'layout_id',
		'source_identities': {
			key: dict(value) for key, value in inspection.source_identities.items()
		},
		'validation_identity': {
			'mask_sha256': inspection.validation_mask_sha256,
			'voxel_count': inspection.validation_voxel_count,
		},
		'rows': list(rows),
	}


def _validate_existing_manifest(
	path: Path,
	*,
	inspection: _Inspection,
	statuses: Sequence[
		tuple[_ConditionPlan, Mapping[str, object] | None, BaseException | None]
	],
) -> None:
	if any(row is None or error is not None for _condition, row, error in statuses):
		raise ValueError('manifest exists while one or more conditions are incomplete')
	rows = [cast('Mapping[str, object]', row) for _condition, row, _error in statuses]
	payload = _read_json(path)
	if payload != _manifest_payload(inspection, rows):
		raise ValueError('existing suite manifest content mismatch')
	validate_f3_lithology_voxel_section_layout_manifest(payload)


def _validate_manifest_sources(value: object) -> None:
	expected = {
		'section_layout_contract',
		'canonical_voxel_dataset_metadata',
		'canonical_split_grid',
		'canonical_class_counts',
		'canonical_split_manifest',
		'label_volume',
		'png_label_inventory',
		'segy_geometry_json',
		'class_info',
		'reference_valid_tokens',
	}
	sources = _mapping(value, 'manifest source identities')
	if set(sources) != expected:
		raise ValueError('manifest source identity inventory mismatch')
	for name, record in sources.items():
		identity = _mapping(record, f'manifest source {name}')
		if set(identity) != {'path', 'sha256'}:
			raise ValueError(f'manifest source {name} key inventory mismatch')
		path_value = identity.get('path')
		if (
			not _is_nonempty_string(path_value)
			or not Path(cast('str', path_value)).is_absolute()
		):
			raise ValueError(f'manifest source {name} path must be absolute')
		if not _is_nonempty_string(identity.get('sha256')):
			raise ValueError(f'manifest source {name} SHA-256 must be non-empty')


def _validate_manifest_row(  # noqa: C901, PLR0912
	row: Mapping[str, object],
	*,
	index: int,
	validation_mask_sha256: str,
) -> None:
	expected_keys = {
		'layout_id',
		'data_size',
		'parent_size',
		'voxel_dataset_root',
		'target_train_voxel_count',
		'actual_train_voxel_count',
		'relative_count_error',
		'selected_token_count',
		'selected_token_identity_sha256',
		'train_mask_sha256',
		'validation_mask_sha256',
		'per_line_contributions',
		'per_class_train_voxel_counts',
		'outputs',
	}
	if set(row) != expected_keys:
		raise ValueError(f'manifest row {index} key inventory mismatch')
	layout_id = row.get('layout_id')
	data_size = row.get('data_size')
	if layout_id not in LAYOUT_IDS or data_size not in DATA_SIZES:
		raise ValueError(f'manifest row {index} condition identity is invalid')
	size_index = DATA_SIZES.index(cast('str', data_size))
	expected_parent = None if size_index == 0 else DATA_SIZES[size_index - 1]
	if row.get('parent_size') != expected_parent:
		raise ValueError(f'manifest row {index} parent size mismatch')
	root_value = row.get('voxel_dataset_root')
	if (
		not _is_nonempty_string(root_value)
		or not Path(cast('str', root_value)).is_absolute()
	):
		raise ValueError(f'manifest row {index} dataset root must be absolute')
	target = _positive_integer(
		row.get('target_train_voxel_count'), f'manifest row {index} target count'
	)
	actual = _positive_integer(
		row.get('actual_train_voxel_count'), f'manifest row {index} actual count'
	)
	_positive_integer(
		row.get('selected_token_count'), f'manifest row {index} selected tokens'
	)
	relative = row.get('relative_count_error')
	if (
		not isinstance(relative, int | float)
		or isinstance(relative, bool)
		or not math.isfinite(float(relative))
		or float(relative) < 0.0
		or float(relative) > 0.1
	):
		raise ValueError(f'manifest row {index} relative error is invalid')
	if float(relative) != abs(actual - target) / target:
		raise ValueError(f'manifest row {index} relative error mismatch')
	for name in (
		'selected_token_identity_sha256',
		'train_mask_sha256',
		'validation_mask_sha256',
	):
		if not _is_nonempty_string(row.get(name)):
			raise TypeError(f'manifest row {index} {name} must be non-empty')
	if row.get('validation_mask_sha256') != validation_mask_sha256:
		raise ValueError(f'manifest row {index} validation identity mismatch')
	line_counts = _mapping(
		row.get('per_line_contributions'), f'manifest row {index} line counts'
	)
	expected_line_count = sum(LINE_COUNTS[cast('str', data_size)])
	if len(line_counts) != expected_line_count:
		raise ValueError(f'manifest row {index} active line count mismatch')
	resolved_line_counts = [
		_positive_integer(value, f'manifest row {index} line contribution')
		for value in line_counts.values()
	]
	if sum(resolved_line_counts) != actual:
		raise ValueError(f'manifest row {index} line contributions double-count')
	class_counts = _mapping(
		row.get('per_class_train_voxel_counts'),
		f'manifest row {index} class counts',
	)
	if set(class_counts) != {str(class_id) for class_id in CLASS_IDS}:
		raise ValueError(f'manifest row {index} class order mismatch')
	if (
		sum(
			_positive_integer(value, f'manifest row {index} class count')
			for value in class_counts.values()
		)
		!= actual
	):
		raise ValueError(f'manifest row {index} class count total mismatch')
	_validate_manifest_row_outputs(
		row.get('outputs'), root=Path(cast('str', root_value)), index=index
	)


def _validate_manifest_row_outputs(value: object, *, root: Path, index: int) -> None:
	outputs = _mapping(value, f'manifest row {index} outputs')
	if set(outputs) != set(REQUIRED_CONDITION_FILES):
		raise ValueError(f'manifest row {index} output inventory mismatch')
	for name, record in outputs.items():
		identity = _mapping(record, f'manifest row {index} output {name}')
		if set(identity) != {'path', 'sha256'}:
			raise ValueError(f'manifest row {index} output {name} identity mismatch')
		path = root / name
		if Path(str(identity.get('path'))).resolve(strict=False) != path.resolve(
			strict=False
		):
			raise ValueError(f'manifest row {index} output {name} path mismatch')
		if not _is_nonempty_string(identity.get('sha256')):
			raise ValueError(f'manifest row {index} output {name} SHA-256 is empty')
		if path.is_file() and identity.get('sha256') != file_sha256(path):
			raise ValueError(f'manifest row {index} output {name} SHA-256 mismatch')


def _validate_preview_matches_contract(
	payload: Mapping[str, object], *, preview: SelectionPreview
) -> None:
	layouts = cast('Sequence[Mapping[str, object]]', payload['layouts'])
	layout = next(
		item for item in layouts if item.get('layout_id') == preview.layout_id
	)
	sizes = _mapping(layout.get('sizes'), f'{preview.layout_id}.sizes')
	recorded = _mapping(
		sizes.get(preview.data_size),
		f'{preview.layout_id}.{preview.data_size}',
	)
	expected = {
		'preview_actual_train_voxel_count': preview.actual_train_voxel_count,
		'preview_count_error': preview.count_error,
		'preview_relative_count_error': preview.relative_count_error,
		'selected_token_xyz': [list(item) for item in preview.selected_token_xyz],
		'per_line_contributions': dict(preview.per_line_contributions),
		'per_class_voxel_counts': dict(preview.per_class_voxel_counts),
	}
	for key, value in expected.items():
		if recorded.get(key) != value:
			raise ValueError(
				f'contract preview drift for {preview.layout_id}/'
				f'{preview.data_size}: {key}'
			)


def _validate_canonical_metadata(metadata: Mapping[str, object]) -> None:
	if metadata.get('artifact_type') != 'f3_lithology_voxel_supervision':
		raise ValueError('invalid canonical voxel supervision artifact type')
	if metadata.get('schema_version') != 1:
		raise ValueError('unsupported canonical voxel supervision schema')
	if metadata.get('split_codes') != {
		'unsupervised': int(UNSUPERVISED_VOXEL_SPLIT),
		'train': int(TRAIN_VOXEL_SPLIT),
		'validation': int(VALIDATION_VOXEL_SPLIT),
	}:
		raise ValueError('canonical voxel split-code contract mismatch')
	if metadata.get('validation_precedence') is not True:
		raise ValueError('canonical validation precedence must be true')


def _validate_canonical_source_identities(
	config: F3SectionLayoutDatasetConfig,
	*,
	canonical_metadata: Mapping[str, object],
	canonical_grid: Path,
) -> None:
	declared = {
		'label_volume': (
			canonical_metadata.get('label_volume'),
			config.source_label_volume,
		),
		'inventory': (
			canonical_metadata.get('inventory'),
			config.png_label_inventory,
		),
		'reference_valid_tokens': (
			canonical_metadata.get('reference_valid_tokens'),
			config.reference_valid_tokens,
		),
	}
	sources = _mapping(canonical_metadata.get('source_identities'), 'source_identities')
	declared.update(
		{
			'class_info': (sources.get('class_info'), config.class_info),
			'segy_geometry_json': (
				sources.get('segy_geometry_json'),
				config.segy_geometry_json,
			),
		}
	)
	for label, (record, path) in declared.items():
		_validate_source_identity(record, path, label=f'canonical {label}')
	outputs = _mapping(canonical_metadata.get('outputs'), 'canonical outputs')
	if Path(str(outputs.get('supervision_split_grid'))).resolve(
		strict=False
	) != canonical_grid.resolve(strict=False):
		raise ValueError('canonical split grid output path drift')


def _validate_contract_source_identities(
	payload: Mapping[str, object],
	*,
	config: F3SectionLayoutDatasetConfig,
	canonical_grid: Path,
) -> None:
	if payload.get('artifact_type') != CONTRACT_ARTIFACT_TYPE:
		raise ValueError(
			'builder requires a generated canonical section-layout contract'
		)
	sources = _mapping(payload.get('source_file_identities'), 'contract sources')
	expected = {
		'canonical_split_grid': canonical_grid,
		'label_volume': config.source_label_volume,
		'line_inventory': config.png_label_inventory,
		'segy_geometry_json': config.segy_geometry_json,
	}
	for label, path in expected.items():
		_validate_source_identity(sources.get(label), path, label=f'contract {label}')


def _validate_contract_validation_identity(
	payload: Mapping[str, object],
	*,
	mask_sha256: str,
	voxel_count: int,
	canonical_grid: Path,
) -> None:
	identity = _mapping(payload.get('validation_identity'), 'validation_identity')
	checks = {
		'mask_sha256': mask_sha256,
		'voxel_count': voxel_count,
		'source_sha256': file_sha256(canonical_grid),
		'unchanged_by_preview': True,
	}
	for key, expected in checks.items():
		if identity.get(key) != expected:
			raise ValueError(f'contract validation identity drift: {key}')
	if Path(str(identity.get('source_path'))).resolve(
		strict=False
	) != canonical_grid.resolve(strict=False):
		raise ValueError('contract validation source path drift')


def _validate_reference_validity(
	grid: NDArray[np.integer],
	*,
	valid_tokens: NDArray[np.bool_],
	patch: tuple[int, int, int],
) -> None:
	voxels = np.argwhere(grid > UNSUPERVISED_VOXEL_SPLIT)
	coordinates = np.unique(voxels // np.asarray(patch, dtype=np.int64), axis=0)
	if coordinates.size and np.any(~valid_tokens[tuple(coordinates.T)]):
		raise ValueError('canonical supervision intersects an invalid reference token')


def _validate_preview_flat_voxel_indices(  # noqa: C901, PLR0913
	preview: SelectionPreview,
	*,
	tokens: NDArray[np.int64],
	canonical_grid: NDArray[np.integer],
	labels: NDArray[np.integer],
	valid_token_shape: tuple[int, int, int],
	class_ids: tuple[int, ...],
	line_array_indices: Mapping[tuple[str, int], int],
) -> NDArray[np.int64]:
	flat = np.asarray(preview.selected_flat_voxel_indices, dtype=np.int64)
	if flat.ndim != 1 or flat.size == 0:
		raise ValueError('selection preview must contain flat teacher voxel indices')
	flat = np.sort(flat)
	if flat[0] < 0 or flat[-1] >= canonical_grid.size:
		raise ValueError('selection preview teacher voxel lies outside volume')
	if np.any(flat[1:] == flat[:-1]):
		raise ValueError('selection preview double-counts a teacher voxel')
	inline_indices = np.asarray(
		[line_array_indices[('inline', value)] for value in preview.inline_lines],
		dtype=np.int64,
	)
	crossline_indices = np.asarray(
		[
			line_array_indices[('crossline', value)]
			for value in preview.crossline_lines
		],
		dtype=np.int64,
	)
	token_linear = np.ravel_multi_index(tokens.T, valid_token_shape)
	known = np.asarray(class_ids, dtype=labels.dtype)
	class_counts = {str(class_id): 0 for class_id in class_ids}
	grid_flat = canonical_grid.reshape(-1)
	label_flat = labels.reshape(-1)
	for start in range(0, flat.size, _STREAM_CHUNK_VOXELS):
		selected = flat[start : start + _STREAM_CHUNK_VOXELS]
		xyz = np.unravel_index(selected, canonical_grid.shape)
		active = np.isin(xyz[0], inline_indices) | np.isin(
			xyz[1], crossline_indices
		)
		if not np.all(active):
			raise ValueError(
				'selection preview teacher voxel lies outside active lines'
			)
		voxel_tokens = tuple(
			xyz[axis] // PATCH_SIZE[axis] for axis in range(3)
		)
		selected_token_linear = np.ravel_multi_index(
			voxel_tokens, valid_token_shape
		)
		positions = np.searchsorted(token_linear, selected_token_linear)
		if np.any(positions >= token_linear.size) or np.any(
			token_linear[np.minimum(positions, token_linear.size - 1)]
			!= selected_token_linear
		):
			raise ValueError(
				'selection preview teacher voxel lies outside selected token blocks'
			)
		if np.any(grid_flat[selected] != TRAIN_VOXEL_SPLIT):
			raise ValueError('selection preview teacher voxel is not canonical train')
		selected_labels = label_flat[selected]
		if np.any(~np.isin(selected_labels, known)):
			raise ValueError('selection preview teacher voxel has an unknown class')
		for class_id in class_ids:
			class_counts[str(class_id)] += int(
				np.count_nonzero(selected_labels == class_id)
			)
	if class_counts != dict(preview.per_class_voxel_counts):
		raise ValueError('selection preview teacher class counts are inconsistent')
	return flat


def _planned_grid_identities(
	canonical_grid: NDArray[np.integer],
	*,
	selected_train_flat_indices: NDArray[np.int64],
) -> tuple[str, str]:
	train_hasher = hashlib.sha256()
	grid_hasher = hashlib.sha256()
	grid_hasher.update(canonical_grid.dtype.str.encode('ascii'))
	grid_hasher.update(
		json.dumps(list(canonical_grid.shape), separators=(',', ':')).encode('ascii')
	)
	canonical_flat = canonical_grid.reshape(-1)
	for start in range(0, canonical_grid.size, _STREAM_CHUNK_VOXELS):
		stop = min(start + _STREAM_CHUNK_VOXELS, canonical_grid.size)
		train = np.zeros(stop - start, dtype=np.bool_)
		left = int(np.searchsorted(selected_train_flat_indices, start))
		right = int(np.searchsorted(selected_train_flat_indices, stop))
		train[selected_train_flat_indices[left:right] - start] = True
		validation = canonical_flat[start:stop] == VALIDATION_VOXEL_SPLIT
		if np.any(train & validation):
			raise ValueError('planned train and validation voxels overlap')
		grid = np.full(
			stop - start,
			UNSUPERVISED_VOXEL_SPLIT,
			dtype=canonical_grid.dtype,
		)
		grid[train] = TRAIN_VOXEL_SPLIT
		grid[validation] = VALIDATION_VOXEL_SPLIT
		train_hasher.update(train.view(np.uint8))
		grid_hasher.update(grid.view(np.uint8))
	return train_hasher.hexdigest(), grid_hasher.hexdigest()


def _materialize_condition_grid(
	condition: _ConditionPlan, *, inspection: _Inspection
) -> NDArray[np.integer]:
	grid = np.full(
		inspection.canonical_grid.shape,
		UNSUPERVISED_VOXEL_SPLIT,
		dtype=inspection.canonical_grid.dtype,
	)
	grid_flat = grid.reshape(-1)
	canonical_flat = inspection.canonical_grid.reshape(-1)
	for start in range(0, grid.size, _STREAM_CHUNK_VOXELS):
		stop = min(start + _STREAM_CHUNK_VOXELS, grid.size)
		validation = canonical_flat[start:stop] == VALIDATION_VOXEL_SPLIT
		grid_flat[start:stop][validation] = VALIDATION_VOXEL_SPLIT
	active_lines = tuple(
		SectionLine(
			slice_type='inline',
			slice_index=value,
			array_index=inspection.line_array_indices[('inline', value)],
			is_validation_line=False,
		)
		for value in condition.active_inlines
	) + tuple(
		SectionLine(
			slice_type='crossline',
			slice_index=value,
			array_index=inspection.line_array_indices[('crossline', value)],
			is_validation_line=False,
		)
		for value in condition.active_crosslines
	)
	teacher = replay_selected_teacher_mask(
		inspection.canonical_grid,
		inspection.label_volume,
		active_lines,
		condition.selected_token_xyz,
		patch_size_xyz=PATCH_SIZE,
		class_ids=inspection.class_ids,
	)
	grid[teacher] = TRAIN_VOXEL_SPLIT
	_validate_materialized_grid(grid, condition=condition, inspection=inspection)
	return cast('NDArray[np.integer]', grid)


def _validate_materialized_grid(
	grid: NDArray[np.integer],
	*,
	condition: _ConditionPlan,
	inspection: _Inspection,
) -> None:
	if grid.shape != inspection.canonical_grid.shape:
		raise ValueError('materialized condition grid shape mismatch')
	if grid.dtype != inspection.canonical_grid.dtype:
		raise TypeError('materialized condition grid dtype mismatch')
	_validate_split_codes(grid)
	train_hash, actual = _split_mask_identity(grid, TRAIN_VOXEL_SPLIT)
	validation_hash, validation_count = _split_mask_identity(
		grid, VALIDATION_VOXEL_SPLIT
	)
	checks = {
		'train mask identity': (train_hash, condition.train_mask_sha256),
		'validation mask identity': (
			validation_hash,
			condition.validation_mask_sha256,
		),
		'grid array identity': (_array_sha256(grid), condition.grid_array_sha256),
	}
	for label, (actual_value, expected_value) in checks.items():
		if actual_value != expected_value:
			raise ValueError(f'materialized condition {label} mismatch')
	if actual != condition.actual_train_voxel_count:
		raise ValueError('materialized condition train voxel count mismatch')
	if validation_count != inspection.validation_voxel_count:
		raise ValueError('materialized condition validation voxel count mismatch')
	train_counts = _split_class_counts(
		inspection.label_volume,
		grid,
		split_code=TRAIN_VOXEL_SPLIT,
		class_ids=inspection.class_ids,
	)
	if train_counts != dict(condition.per_class_train_voxel_counts):
		raise ValueError('materialized condition train class counts mismatch')


def _split_class_counts(
	labels: NDArray[np.integer],
	grid: NDArray[np.integer],
	*,
	split_code: int,
	class_ids: Sequence[int],
) -> dict[str, int]:
	counts = {str(class_id): 0 for class_id in class_ids}
	label_flat = labels.reshape(-1)
	grid_flat = grid.reshape(-1)
	for start in range(0, grid.size, _STREAM_CHUNK_VOXELS):
		stop = min(start + _STREAM_CHUNK_VOXELS, grid.size)
		selected_labels = label_flat[start:stop][
			grid_flat[start:stop] == split_code
		]
		for class_id in class_ids:
			counts[str(class_id)] += int(
				np.count_nonzero(selected_labels == class_id)
			)
	return counts


def _sorted_token_rows_are_subset(
	smaller: NDArray[np.int64], larger: NDArray[np.int64]
) -> bool:
	shape = tuple(
		int(value) + 1
		for value in np.maximum(smaller.max(axis=0), larger.max(axis=0))
	)
	smaller_linear = np.ravel_multi_index(smaller.T, shape)
	larger_linear = np.ravel_multi_index(larger.T, shape)
	positions = np.searchsorted(larger_linear, smaller_linear)
	return bool(
		np.all(positions < larger_linear.size)
		and np.all(
			larger_linear[np.minimum(positions, larger_linear.size - 1)]
			== smaller_linear
		)
	)


def _write_class_counts(
	path: Path, *, condition: _ConditionPlan, inspection: _Inspection
) -> None:
	rows = _class_count_rows(condition=condition, inspection=inspection)
	with path.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=('split', 'class_id', 'class_name', 'count', 'fraction'),
		)
		writer.writeheader()
		writer.writerows(rows)


def _class_count_rows(
	*, condition: _ConditionPlan, inspection: _Inspection
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for split, counts in (
		('train', condition.per_class_train_voxel_counts),
		('validation', condition.per_class_validation_voxel_counts),
	):
		total = sum(counts.values())
		for class_id, name in zip(
			inspection.class_ids, inspection.class_names, strict=True
		):
			count = counts[str(class_id)]
			rows.append(
				{
					'split': split,
					'class_id': class_id,
					'class_name': name,
					'count': count,
					'fraction': 0.0 if total == 0 else count / total,
				}
			)
	return rows


def _validate_condition_class_counts(
	path: Path, *, condition: _ConditionPlan, inspection: _Inspection
) -> None:
	expected = _class_count_rows(condition=condition, inspection=inspection)
	with path.open(newline='', encoding='utf-8') as handle:
		actual = list(csv.DictReader(handle))
	if len(actual) != len(expected):
		raise ValueError('condition class-count row count mismatch')
	for actual_row, expected_row in zip(actual, expected, strict=True):
		for key in ('split', 'class_id', 'class_name', 'count'):
			if actual_row.get(key) != str(expected_row[key]):
				raise ValueError(f'condition class-count mismatch: {key}')
		if float(actual_row['fraction']) != float(expected_row['fraction']):
			raise ValueError('condition class-count fraction mismatch')


def _validate_class_counts_csv(path: Path) -> None:
	with path.open(newline='', encoding='utf-8') as handle:
		reader = csv.DictReader(handle)
		if tuple(reader.fieldnames or ()) != (
			'split',
			'class_id',
			'class_name',
			'count',
			'fraction',
		):
			raise ValueError('condition class-count CSV header mismatch')
		if not list(reader):
			raise ValueError('condition class-count CSV is empty')


def _render_summary(condition: _ConditionPlan) -> str:
	return '\n'.join(
		[
			'# F3 section-layout voxel supervision',
			'',
			f'- layout: {condition.layout_id}',
			f'- data size: {condition.data_size}',
			f'- target train voxels: {condition.target_train_voxel_count}',
			f'- actual train voxels: {condition.actual_train_voxel_count}',
			f'- relative count error: {condition.relative_count_error:.12g}',
			f'- selected tokens: {condition.selected_token_xyz.shape[0]}',
			'- teacher voxels are partial active-plane token footprints only.',
			'- canonical validation is preserved bitwise.',
			'- dense label values are unchanged.',
			'',
		]
	)


def _input_paths(config: F3SectionLayoutDatasetConfig) -> dict[str, Path]:
	return {
		'section_layout_contract': config.section_layout_contract,
		'canonical_voxel_dataset': config.canonical_voxel_dataset,
		'source_label_volume': config.source_label_volume,
		'png_label_inventory': config.png_label_inventory,
		'segy_geometry_json': config.segy_geometry_json,
		'class_info': config.class_info,
		'reference_valid_tokens': config.reference_valid_tokens,
	}


def _stage_condition_root(staging: Path, final_root: Path, target: Path) -> Path:
	return staging / target.relative_to(final_root)


def _validate_source_identity(value: object, path: Path, *, label: str) -> None:
	identity = _mapping(value, label)
	if Path(str(identity.get('path'))).resolve(strict=False) != path.resolve(
		strict=False
	):
		raise ValueError(f'{label} path drift')
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} SHA-256 drift')


def _validate_output_identity(
	value: object, path: Path, *, recorded_root: Path
) -> None:
	identity = _mapping(value, f'output {path.name}')
	if Path(str(identity.get('path'))).resolve(strict=False) != (
		recorded_root / path.name
	).resolve(strict=False):
		raise ValueError(f'output {path.name} recorded path mismatch')
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'output {path.name} SHA-256 mismatch')


def _load_integer_volume(path: Path, *, label: str) -> NDArray[np.integer]:
	array = np.load(path, mmap_mode='r', allow_pickle=False)
	if (
		array.ndim != 3
		or not np.issubdtype(array.dtype, np.integer)
		or array.dtype == np.dtype(np.bool_)
	):
		raise TypeError(f'{label} must be a 3D integer array')
	return cast('NDArray[np.integer]', array)


def _validate_split_codes(grid: NDArray[np.integer]) -> None:
	values = {int(value) for value in np.unique(grid)}
	allowed = {
		int(UNSUPERVISED_VOXEL_SPLIT),
		int(TRAIN_VOXEL_SPLIT),
		int(VALIDATION_VOXEL_SPLIT),
	}
	if (
		not values <= allowed
		or not {
			int(TRAIN_VOXEL_SPLIT),
			int(VALIDATION_VOXEL_SPLIT),
		}
		<= values
	):
		raise ValueError(f'invalid split grid codes: {sorted(values)!r}')


def _validate_token_array(array: NDArray[np.generic]) -> None:
	if array.ndim != 2 or array.shape[1:] != (3,):
		raise ValueError('selected_token_xyz must have shape [N, 3]')
	if array.dtype != np.dtype(np.int64):
		raise TypeError('selected_token_xyz dtype must be int64')
	if array.shape[0] <= 0 or np.any(array < 0):
		raise ValueError('selected_token_xyz must contain nonnegative rows')
	if not np.array_equal(array, np.unique(array, axis=0)):
		raise ValueError(
			'selected_token_xyz must be unique and lexicographically sorted'
		)


def _split_mask_identity(
	grid: NDArray[np.integer], split_code: int
) -> tuple[str, int]:
	hasher = hashlib.sha256()
	count = 0
	flat = grid.reshape(-1)
	for start in range(0, grid.size, _STREAM_CHUNK_VOXELS):
		stop = min(start + _STREAM_CHUNK_VOXELS, grid.size)
		mask = np.ascontiguousarray(flat[start:stop] == split_code)
		hasher.update(mask.view(np.uint8))
		count += int(np.count_nonzero(mask))
	return hasher.hexdigest(), count


def _count_split_code(grid: NDArray[np.integer], split_code: int) -> int:
	return _split_mask_identity(grid, split_code)[1]


def _array_sha256(array: NDArray[np.generic]) -> str:
	value = np.ascontiguousarray(array)
	hasher = hashlib.sha256()
	hasher.update(value.dtype.str.encode('ascii'))
	hasher.update(json.dumps(list(value.shape), separators=(',', ':')).encode('ascii'))
	hasher.update(value.view(np.uint8))
	return hasher.hexdigest()


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'{label} must be a positive integer triple')
	items = tuple(value)
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in items
	):
		raise ValueError(f'{label} must be a positive integer triple')
	return cast('tuple[int, int, int]', items)


def _positive_integer(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{label} must be an integer')
	if value <= 0:
		raise ValueError(f'{label} must be positive')
	return value


def _is_nonempty_string(value: object) -> bool:
	return isinstance(value, str) and bool(value)


def _identity(path: Path, *, recorded_path: Path | None = None) -> dict[str, str]:
	return {
		'path': str(path if recorded_path is None else recorded_path),
		'sha256': file_sha256(path),
	}


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON must contain an object: {path}')
	return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary_name = tempfile.mkstemp(
		prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
	)
	try:
		with open(fd, 'w', encoding='utf-8') as handle:  # noqa: PTH123
			json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
			handle.write('\n')
		Path(temporary_name).replace(path)
	except BaseException:
		Path(temporary_name).unlink(missing_ok=True)
		raise


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _quarantine(path: Path, *, reason: str) -> Path:
	stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
	safe_reason = ''.join(
		character if character.isalnum() else '_' for character in reason
	)
	candidate = path.with_name(f'{path.name}.quarantine_{stamp}_{safe_reason}')
	counter = 1
	while candidate.exists():
		candidate = path.with_name(
			f'{path.name}.quarantine_{stamp}_{safe_reason}_{counter}'
		)
		counter += 1
	path.replace(candidate)
	return candidate


__all__ = [
	'build_f3_lithology_voxel_section_layout_datasets',
	'inspect_f3_lithology_voxel_section_layout_datasets',
	'validate_f3_lithology_voxel_section_layout_condition',
	'validate_f3_lithology_voxel_section_layout_manifest',
]
