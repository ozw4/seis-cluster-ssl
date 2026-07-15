"""Encoder-independent low-label voxel supervision for the F3 benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.robustness import (
	assert_same_token_identity,
	load_token_dataset_npz,
	paired_token_identity_hash,
)
from seis_ssl_cluster.f3.lithology.voxel_dataset import (
	COUNTS_NAME,
	GRID_NAME,
	MANIFEST_NAME,
	METADATA_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_split import (
	TRAIN_VOXEL_SPLIT,
	UNSUPERVISED_VOXEL_SPLIT,
	VALIDATION_VOXEL_SPLIT,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget import (
		F3VoxelLabelBudgetDatasetConfig,
	)
	from seis_ssl_cluster.f3.lithology.tokens import F3LithologyTokenDataset

DATASET_MANIFEST_NAME = 'voxel_label_budget_dataset_manifest.json'
LABEL_BUDGET_METADATA_NAME = 'voxel_label_budget_metadata.json'
LABEL_BUDGET_SUMMARY_NAME = 'voxel_label_budget_summary.md'
ARTIFACT_TYPE = 'f3_lithology_voxel_label_budget_dataset'
MANIFEST_ARTIFACT_TYPE = 'f3_lithology_voxel_label_budget_dataset_manifest'
SCHEMA_VERSION = 1
EXPECTED_TOKEN_MANIFEST_TYPE = 'f3_lithology_label_budget_suite_manifest'  # noqa: S105
EXPECTED_TOKEN_DATASET_TYPE = 'f3_lithology_label_budget_token_dataset'  # noqa: S105
REQUIRED_CONDITION_FILES = (
	GRID_NAME,
	METADATA_NAME,
	COUNTS_NAME,
	MANIFEST_NAME,
	LABEL_BUDGET_METADATA_NAME,
	LABEL_BUDGET_SUMMARY_NAME,
)


@dataclass(frozen=True)
class VoxelLabelBudgetCondition:
	"""Identity-validated inputs and the derived grid for one budget/seed."""

	budget_id: str
	per_class_cap: int
	subsample_seed: int
	output_dir: Path
	selected_token_rows: int
	unique_token_xyz: NDArray[np.int64]
	duplicate_selected_rows: int
	selected_token_identity_sha256: str
	validation_token_identity_sha256: str
	paired_token_identity_sha256: str
	selected_token_sources: Mapping[str, Mapping[str, str]]
	train_mask_sha256: str
	validation_mask_sha256: str
	grid_array_sha256: str
	train_voxel_count: int
	validation_voxel_count: int
	per_class_train_voxel_counts: Mapping[str, int]
	per_class_validation_voxel_counts: Mapping[str, int]


@dataclass(frozen=True)
class VoxelLabelBudgetDatasetInspection:
	"""Fully validated common inputs and all requested derived conditions."""

	conditions: tuple[VoxelLabelBudgetCondition, ...]
	common_metadata: Mapping[str, object]
	common_grid: NDArray[np.integer]
	label_volume: NDArray[np.integer]
	class_ids: tuple[int, ...]
	class_names: tuple[str, ...]
	token_grid_shape_xyz: tuple[int, int, int]
	volume_shape_xyz: tuple[int, int, int]
	patch_size_xyz: tuple[int, int, int]
	suite_name: str
	output_root: Path
	source_identities: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class VoxelLabelBudgetDatasetBuildResult:
	"""Committed suite manifest and the condition actions taken."""

	manifest_json: Path
	condition_roots: tuple[Path, ...]
	rows: tuple[Mapping[str, object], ...]
	quarantines: tuple[Path, ...]


def expand_selected_token_blocks(
	token_xyz: NDArray[np.integer] | Sequence[Sequence[int]],
	*,
	patch_size_xyz: Sequence[int],
	volume_shape_xyz: Sequence[int],
) -> NDArray[np.bool_]:
	"""Expand unique token coordinates into clipped dense voxel blocks."""
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	shape = _positive_triplet(volume_shape_xyz, 'volume_shape_xyz')
	coordinates = _token_coordinates(token_xyz)
	if np.any(coordinates < 0):
		raise ValueError('selected token coordinates must be non-negative')
	mask = np.zeros(shape, dtype=np.bool_)
	for coordinate in np.unique(coordinates, axis=0):
		start = tuple(int(coordinate[axis]) * patch[axis] for axis in range(3))
		stop = tuple(
			min(start[axis] + patch[axis], shape[axis]) for axis in range(3)
		)
		if any(start[axis] >= shape[axis] for axis in range(3)):
			raise ValueError(
				'selected token coordinate is outside the volume: '
				f'{coordinate.tolist()}'
			)
		mask[_slices(start, stop)] = True
	return mask


def build_low_label_supervision_grid(
	full_grid: NDArray[np.integer],
	token_xyz: NDArray[np.integer] | Sequence[Sequence[int]],
	*,
	patch_size_xyz: Sequence[int],
) -> NDArray[np.integer]:
	"""Intersect expanded token blocks with train and preserve full validation."""
	grid = np.asarray(full_grid)
	_validate_full_grid(grid)
	coordinates = np.unique(_token_coordinates(token_xyz), axis=0)
	if np.any(coordinates < 0):
		raise ValueError('selected token coordinates must be non-negative')
	result = np.zeros_like(grid)
	for coordinate in coordinates:
		block = _token_block(
			coordinate, patch_size_xyz=patch_size_xyz, volume_shape_xyz=grid.shape
		)
		full_block = np.asarray(grid[block])
		block_train = full_block == TRAIN_VOXEL_SPLIT
		if not np.any(block_train):
			raise ValueError(
				'selected unique token has no canonical train voxel: '
				f'{coordinate.tolist()}'
			)
		result_block = result[block]
		result_block[block_train] = TRAIN_VOXEL_SPLIT
	result[grid == VALIDATION_VOXEL_SPLIT] = VALIDATION_VOXEL_SPLIT
	_validate_derived_grid(result, full_grid=grid)
	return result


def inspect_f3_lithology_voxel_label_budget_datasets(
	config: F3VoxelLabelBudgetDatasetConfig,
) -> VoxelLabelBudgetDatasetInspection:
	"""Validate both token suites and derive all 15 low-label voxel grids."""
	common_paths = {
		'common_grid': config.common_voxel_dataset / GRID_NAME,
		'common_metadata': config.common_voxel_dataset / METADATA_NAME,
		'common_class_counts': config.common_voxel_dataset / COUNTS_NAME,
		'common_split_manifest': config.common_voxel_dataset / MANIFEST_NAME,
		'mae_m1_token_manifest': config.mae_m1_label_budget_manifest,
		'm1_m2a_token_manifest': config.m1_m2a_label_budget_manifest,
	}
	for label, path in common_paths.items():
		if not path.is_file():
			raise FileNotFoundError(f'missing {label}: {path}')
	common_metadata = _read_json(common_paths['common_metadata'])
	_validate_common_metadata(common_metadata)
	for source_key in (
		'reference_embedding',
		'reference_valid_tokens',
		'label_volume',
		'inventory',
	):
		_validate_declared_common_source(common_metadata, source_key)
	common_grid = np.load(
		common_paths['common_grid'], mmap_mode='r', allow_pickle=False
	)
	_validate_full_grid(common_grid)
	label_identity = _mapping(common_metadata.get('label_volume'), 'label_volume')
	label_path = _identity_path(label_identity, label='label_volume')
	if label_identity.get('sha256') != file_sha256(label_path):
		raise ValueError('canonical label-volume hash mismatch')
	labels = np.load(label_path, mmap_mode='r', allow_pickle=False)
	if (
		labels.shape != common_grid.shape
		or not np.issubdtype(labels.dtype, np.integer)
		or labels.dtype == np.bool_
	):
		raise ValueError('canonical label volume must match the full split grid')
	classes = cast('Sequence[Mapping[str, object]]', common_metadata['classes'])
	class_ids = tuple(int(item['class_id']) for item in classes)
	class_names = tuple(str(item['class_name']) for item in classes)
	reference = _mapping(
		common_metadata.get('reference_embedding'), 'reference_embedding'
	)
	token_shape = _positive_triplet(
		reference.get('token_grid_shape'), 'reference_embedding.token_grid_shape'
	)
	volume_shape = _positive_triplet(
		reference.get('volume_shape_xyz'), 'reference_embedding.volume_shape_xyz'
	)
	if tuple(common_grid.shape) != volume_shape:
		raise ValueError('canonical grid shape does not match reference embedding')
	if tuple(config.patch_size_xyz) != _positive_triplet(
		reference.get('patch_size'), 'reference_embedding.patch_size'
	):
		raise ValueError('configured patch size does not match canonical voxel dataset')
	expected_token_shape = tuple(
		(size + patch - 1) // patch
		for size, patch in zip(
			volume_shape, config.patch_size_xyz, strict=True
		)
	)
	if expected_token_shape != token_shape:
		raise ValueError('canonical voxel/token geometry is inconsistent')
	manifests = {
		'mae_m1': _read_token_manifest(config.mae_m1_label_budget_manifest),
		'm1_m2a': _read_token_manifest(config.m1_m2a_label_budget_manifest),
	}
	rows_by_suite = {
		name: _token_rows_by_key(payload, source=name)
		for name, payload in manifests.items()
	}
	source_identities = {
		label: _identity(path) for label, path in common_paths.items()
	}
	conditions = [
		_condition(
			config,
			budget_id=budget_id,
			seed=seed,
			rows_by_suite=rows_by_suite,
			common_grid=common_grid,
			labels=labels,
			class_ids=class_ids,
			token_grid_shape_xyz=token_shape,
		)
		for budget_id in config.budgets
		for seed in config.subsample_seeds
	]
	validation_hashes = {item.validation_mask_sha256 for item in conditions}
	validation_counts = {item.validation_voxel_count for item in conditions}
	if len(validation_hashes) != 1 or len(validation_counts) != 1:
		raise ValueError('derived validation identity changed across conditions')
	return VoxelLabelBudgetDatasetInspection(
		conditions=tuple(conditions),
		common_metadata=common_metadata,
		common_grid=common_grid,
		label_volume=labels,
		class_ids=class_ids,
		class_names=class_names,
		token_grid_shape_xyz=token_shape,
		volume_shape_xyz=volume_shape,
		patch_size_xyz=config.patch_size_xyz,
		suite_name=config.suite_name,
		output_root=config.output_root,
		source_identities=source_identities,
	)


def build_f3_lithology_voxel_label_budget_datasets(  # noqa: C901, PLR0912
	config: F3VoxelLabelBudgetDatasetConfig,
	*,
	only_missing: bool = False,
) -> VoxelLabelBudgetDatasetBuildResult:
	"""Commit or strictly reuse all requested low-label voxel datasets."""
	inspection = inspect_f3_lithology_voxel_label_budget_datasets(config)
	manifest_path = config.output_root / DATASET_MANIFEST_NAME
	if not only_missing and not config.overwrite:
		existing = [
			path
			for path in (
				manifest_path,
				*(condition.output_dir for condition in inspection.conditions),
			)
			if path.exists()
		]
		if existing:
			raise FileExistsError(
				'refusing existing voxel label-budget output path: '
				f'{existing[0]}'
			)
	config.output_root.mkdir(parents=True, exist_ok=True)
	rows: list[Mapping[str, object]] = []
	quarantines: list[Path] = []
	for condition in inspection.conditions:
		action = 'NEW'
		if condition.output_dir.exists():
			if only_missing:
				try:
					row = validate_voxel_label_budget_condition(
						condition, inspection=inspection
					)
				except (OSError, TypeError, ValueError) as error:
					quarantine = _quarantine(
						condition.output_dir, reason=type(error).__name__
					)
					quarantines.append(quarantine)
					action = 'REBUILT_AFTER_QUARANTINE'
				else:
					rows.append({**row, 'action': 'REUSED'})
					continue
			elif config.overwrite:
				quarantines.append(
					_quarantine(condition.output_dir, reason='overwrite')
				)
				action = 'REBUILT_AFTER_QUARANTINE'
			else:  # pragma: no cover - protected by the suite-wide preflight
				raise FileExistsError(condition.output_dir)
		_write_condition(condition, inspection=inspection, config=config)
		row = validate_voxel_label_budget_condition(
			condition, inspection=inspection
		)
		rows.append({**row, 'action': action})
	payload = _suite_manifest_payload(config, inspection=inspection, rows=rows)
	if manifest_path.exists():
		try:
			current = _read_json(manifest_path)
		except (OSError, TypeError, ValueError):
			current = None
		if current is not None and _without_actions(current) == _without_actions(
			payload
		):
			return VoxelLabelBudgetDatasetBuildResult(
				manifest_json=manifest_path,
				condition_roots=tuple(
					item.output_dir for item in inspection.conditions
				),
				rows=tuple(rows),
				quarantines=tuple(quarantines),
			)
		if only_missing or config.overwrite:
			quarantines.append(_quarantine(manifest_path, reason='manifest_mismatch'))
			_write_json_atomic(manifest_path, payload)
		else:  # pragma: no cover - protected by the suite-wide preflight
			raise FileExistsError(manifest_path)
	else:
		_write_json_atomic(manifest_path, payload)
	return VoxelLabelBudgetDatasetBuildResult(
		manifest_json=manifest_path,
		condition_roots=tuple(item.output_dir for item in inspection.conditions),
		rows=tuple(rows),
		quarantines=tuple(quarantines),
	)


def validate_voxel_label_budget_condition(
	condition: VoxelLabelBudgetCondition,
	*,
	inspection: VoxelLabelBudgetDatasetInspection,
) -> dict[str, object]:
	"""Fully validate a committed condition before reuse or manifest inclusion."""
	root = condition.output_dir
	metadata = validate_voxel_label_budget_condition_artifact(root)
	grid = np.load(root / GRID_NAME, mmap_mode='r', allow_pickle=False)
	if (
		grid.shape != inspection.common_grid.shape
		or grid.dtype != inspection.common_grid.dtype
	):
		raise ValueError('committed low-label grid shape/dtype mismatch')
	expected_grid = _derived_condition_grid(condition, inspection=inspection)
	if not np.array_equal(grid, expected_grid):
		raise ValueError('committed low-label grid content mismatch')
	expected_suite = {
		'name': inspection.suite_name,
		'output_root': str(inspection.output_root),
	}
	if metadata.get('suite') != expected_suite:
		raise ValueError('voxel label-budget suite identity mismatch')
	identity = _mapping(metadata.get('identity'), 'identity')
	expected_identity = _condition_identity(condition, inspection=inspection)
	if identity != expected_identity:
		raise ValueError('voxel label-budget condition identity mismatch')
	expected_sources = {
		**inspection.source_identities,
		'selected_token_artifacts': condition.selected_token_sources,
	}
	if metadata.get('sources') != expected_sources:
		raise ValueError('voxel label-budget source identity mismatch')
	voxel_metadata = _read_json(root / METADATA_NAME)
	expected_voxel_metadata = _voxel_dataset_metadata(
		condition,
		inspection=inspection,
		output_dir=root,
		suite_name=inspection.suite_name,
	)
	if voxel_metadata != expected_voxel_metadata:
		raise ValueError('voxel dataset metadata content mismatch')
	_validate_condition_class_counts(
		root / COUNTS_NAME,
		condition=condition,
		class_ids=inspection.class_ids,
		class_names=inspection.class_names,
	)
	if file_sha256(root / MANIFEST_NAME) != inspection.source_identities[
		'common_split_manifest'
	]['sha256']:
		raise ValueError('condition split manifest differs from canonical source')
	if (root / LABEL_BUDGET_SUMMARY_NAME).read_text(
		encoding='utf-8'
	) != _render_condition_summary(condition):
		raise ValueError('voxel label-budget summary content mismatch')
	return _condition_manifest_row(condition, root=root)


def validate_voxel_label_budget_condition_artifact(  # noqa: C901
	root: str | Path,
) -> Mapping[str, object]:
	"""Validate the self-contained six-file condition artifact contract.

	This is the shared committed-artifact boundary used both by the builder's
	deep source-aware validator and by downstream consumers that do not own the
	original token-suite configuration.
	"""
	condition_root = Path(root)
	missing = [
		name
		for name in REQUIRED_CONDITION_FILES
		if not (condition_root / name).is_file()
	]
	if missing:
		raise FileNotFoundError(
			f'condition is missing required files: {missing!r}'
		)
	metadata = _read_json(condition_root / LABEL_BUDGET_METADATA_NAME)
	if metadata.get('artifact_type') != ARTIFACT_TYPE or metadata.get(
		'schema_version'
	) != SCHEMA_VERSION:
		raise ValueError('invalid voxel label-budget metadata schema')
	outputs = _mapping(metadata.get('outputs'), 'outputs')
	expected_outputs = set(REQUIRED_CONDITION_FILES) - {
		LABEL_BUDGET_METADATA_NAME
	}
	if set(outputs) != expected_outputs:
		raise ValueError('voxel label-budget output inventory mismatch')
	for name in sorted(expected_outputs):
		_validate_identity_record(
			outputs.get(name),
			condition_root / name,
			label=f'condition output {name}',
		)

	identity = _mapping(metadata.get('identity'), 'identity')
	grid = np.load(condition_root / GRID_NAME, mmap_mode='r', allow_pickle=False)
	if (
		not np.issubdtype(grid.dtype, np.integer)
		or grid.dtype == np.dtype(np.bool_)
	):
		raise TypeError('committed low-label grid must use an integer dtype')
	shape = _positive_triplet(identity.get('volume_shape_xyz'), 'volume_shape_xyz')
	if tuple(grid.shape) != shape:
		raise ValueError('committed low-label grid volume shape mismatch')
	flat_grid = grid.reshape(-1)
	allowed_codes = np.asarray(
		[
			UNSUPERVISED_VOXEL_SPLIT,
			TRAIN_VOXEL_SPLIT,
			VALIDATION_VOXEL_SPLIT,
		],
		dtype=grid.dtype,
	)
	for start in range(0, flat_grid.size, 1_000_000):
		stop = min(start + 1_000_000, flat_grid.size)
		if not np.all(np.isin(flat_grid[start:stop], allowed_codes)):
			raise ValueError(
				'committed low-label grid contains an unknown split code'
			)
	train_hash, train_count = _split_mask_identity(grid, TRAIN_VOXEL_SPLIT)
	validation_hash, validation_count = _split_mask_identity(
		grid, VALIDATION_VOXEL_SPLIT
	)
	checks = (
		('grid_array_sha256', array_sha256(grid)),
		('train_mask_sha256', train_hash),
		('validation_mask_sha256', validation_hash),
		('actual_train_voxel_count', train_count),
		('validation_voxel_count', validation_count),
	)
	for field, expected in checks:
		if identity.get(field) != expected:
			raise ValueError(f'committed low-label grid {field} mismatch')

	# Parse every non-array contract file as part of completeness validation.
	_read_json(condition_root / METADATA_NAME)
	_read_json(condition_root / MANIFEST_NAME)
	with (condition_root / COUNTS_NAME).open(
		newline='', encoding='utf-8'
	) as handle:
		reader = csv.DictReader(handle)
		if reader.fieldnames is None or not list(reader):
			raise ValueError('condition class_counts.csv is empty')
	if not (condition_root / LABEL_BUDGET_SUMMARY_NAME).read_text(
		encoding='utf-8'
	).strip():
		raise ValueError('condition voxel_label_budget_summary.md is empty')
	return metadata


def _condition(  # noqa: C901, PLR0912, PLR0913
	config: F3VoxelLabelBudgetDatasetConfig,
	*,
	budget_id: str,
	seed: int,
	rows_by_suite: Mapping[str, Mapping[tuple[str, str, int], Mapping[str, object]]],
	common_grid: NDArray[np.integer],
	labels: NDArray[np.integer],
	class_ids: tuple[int, ...],
	token_grid_shape_xyz: tuple[int, int, int],
) -> VoxelLabelBudgetCondition:
	roles = {
		'mae_m1_mae': ('mae_m1', config.models['mae'], 'baseline'),
		'mae_m1_m1': ('mae_m1', config.models['m1'], 'candidate'),
		'm1_m2a_m1': ('m1_m2a', config.models['m1'], 'baseline'),
		'm1_m2a_m2a': ('m1_m2a', config.models['m2a'], 'candidate'),
	}
	selected_rows: dict[str, Mapping[str, object]] = {}
	for role, (suite, model_tag, _model_role) in roles.items():
		key = (model_tag, budget_id, seed)
		try:
			selected_rows[role] = rows_by_suite[suite][key]
		except KeyError as error:
			raise ValueError(
				f'missing required token label-budget row: {key!r}'
			) from error
	per_class_cap = int(budget_id[3:])
	datasets: dict[
		str, tuple[F3LithologyTokenDataset, F3LithologyTokenDataset]
	] = {}
	sources: dict[str, Mapping[str, str]] = {}
	for role, row in selected_rows.items():
		_validate_token_row(
			row,
			budget_id=budget_id,
			per_class_cap=per_class_cap,
			seed=seed,
			expected_model_tag=roles[role][1],
			expected_model_role=roles[role][2],
		)
		train_path = Path(str(row['train_tokens']))
		validation_path = Path(str(row['validation_tokens']))
		metadata_path = Path(str(row['metadata_json']))
		for path in (train_path, validation_path, metadata_path):
			if not path.is_file():
				raise FileNotFoundError(path)
		metadata = _read_json(metadata_path)
		_validate_token_metadata(
			metadata,
			row=row,
			budget_id=budget_id,
			per_class_cap=per_class_cap,
			seed=seed,
			expected_model_tag=roles[role][1],
			expected_model_role=roles[role][2],
		)
		train_dataset = load_token_dataset_npz(train_path)
		validation_dataset = load_token_dataset_npz(validation_path)
		_validate_loaded_token_datasets(
			train_dataset,
			validation_dataset,
			row=row,
			metadata=metadata,
			class_ids=class_ids,
			per_class_cap=per_class_cap,
			require_all_classes=config.require_all_classes,
		)
		actual_paired_hash = paired_token_identity_hash(
			train_dataset, validation_dataset
		)
		if row.get('paired_identity_hash') != actual_paired_hash:
			raise ValueError('token suite paired identity hash mismatch')
		datasets[role] = (train_dataset, validation_dataset)
		sources[role] = {
			'train_tokens_path': str(train_path),
			'train_tokens_sha256': file_sha256(train_path),
			'validation_tokens_path': str(validation_path),
			'validation_tokens_sha256': file_sha256(validation_path),
			'metadata_path': str(metadata_path),
			'metadata_sha256': file_sha256(metadata_path),
		}
	reference_train, reference_validation = datasets['mae_m1_mae']
	for role, (train, validation) in datasets.items():
		if role != 'mae_m1_mae':
			assert_same_token_identity(
				reference_train,
				train,
				reference_label='MAE selected train',
				candidate_label=f'{role} selected train',
			)
			assert_same_token_identity(
				reference_validation,
				validation,
				reference_label='MAE validation',
				candidate_label=f'{role} validation',
			)
	paired_hash = paired_token_identity_hash(reference_train, reference_validation)
	if any(
		paired_token_identity_hash(train, validation) != paired_hash
		for train, validation in datasets.values()
	):
		raise ValueError('three-model selected token paired identity mismatch')
	coordinates = _token_coordinates(reference_train.token_xyz)
	if np.any(coordinates < 0) or np.any(
		coordinates >= np.asarray(token_grid_shape_xyz)[None, :]
	):
		raise ValueError('selected token coordinate is outside canonical token grid')
	unique = np.unique(coordinates, axis=0).astype(np.int64, copy=False)
	grid = build_low_label_supervision_grid(
		common_grid, unique, patch_size_xyz=config.patch_size_xyz
	)
	train = grid == TRAIN_VOXEL_SPLIT
	validation = grid == VALIDATION_VOXEL_SPLIT
	train_counts = _class_counts(labels, train, class_ids)
	validation_counts = _class_counts(labels, validation, class_ids)
	train_voxel_count = int(np.count_nonzero(train))
	validation_voxel_count = int(np.count_nonzero(validation))
	if sum(train_counts.values()) != train_voxel_count:
		raise ValueError('low-label train contains a label outside the class order')
	if sum(validation_counts.values()) != validation_voxel_count:
		raise ValueError('validation contains a label outside the class order')
	if config.require_all_classes and any(
		train_counts[str(item)] <= 0 for item in class_ids
	):
		raise ValueError('low-label voxel supervision is missing a required class')
	return VoxelLabelBudgetCondition(
		budget_id=budget_id,
		per_class_cap=per_class_cap,
		subsample_seed=seed,
		output_dir=(
			config.output_root
			/ 'datasets'
			/ f'budget={budget_id}'
			/ f'subsample_seed={seed}'
			/ 'voxel_supervision'
		),
		selected_token_rows=reference_train.count,
		unique_token_xyz=unique,
		duplicate_selected_rows=reference_train.count - unique.shape[0],
		selected_token_identity_sha256=paired_token_identity_hash(reference_train),
		validation_token_identity_sha256=paired_token_identity_hash(
			reference_validation
		),
		paired_token_identity_sha256=paired_hash,
		selected_token_sources=sources,
		train_mask_sha256=array_sha256(train),
		validation_mask_sha256=array_sha256(validation),
		grid_array_sha256=array_sha256(grid),
		train_voxel_count=train_voxel_count,
		validation_voxel_count=validation_voxel_count,
		per_class_train_voxel_counts=train_counts,
		per_class_validation_voxel_counts=validation_counts,
	)


def _write_condition(
	condition: VoxelLabelBudgetCondition,
	*,
	inspection: VoxelLabelBudgetDatasetInspection,
	config: F3VoxelLabelBudgetDatasetConfig,
) -> None:
	root = condition.output_dir
	if root.exists():
		raise FileExistsError(root)
	root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(tempfile.mkdtemp(prefix=f'.{root.name}.staging-', dir=root.parent))
	try:
		grid = _derived_condition_grid(condition, inspection=inspection)
		np.save(staging / GRID_NAME, grid, allow_pickle=False)
		shutil.copyfile(
			Path(inspection.source_identities['common_split_manifest']['path']),
			staging / MANIFEST_NAME,
		)
		_write_class_counts(
			staging / COUNTS_NAME,
			condition=condition,
			class_ids=inspection.class_ids,
			class_names=inspection.class_names,
		)
		voxel_metadata = _voxel_dataset_metadata(
			condition,
			inspection=inspection,
			output_dir=root,
			suite_name=config.suite_name,
		)
		_write_json(staging / METADATA_NAME, voxel_metadata)
		(staging / LABEL_BUDGET_SUMMARY_NAME).write_text(
			_render_condition_summary(condition), encoding='utf-8'
		)
		metadata = {
			'artifact_type': ARTIFACT_TYPE,
			'schema_version': SCHEMA_VERSION,
			'suite': {
				'name': config.suite_name,
				'output_root': str(config.output_root),
			},
			'identity': _condition_identity(condition, inspection=inspection),
			'sources': {
				**inspection.source_identities,
				'selected_token_artifacts': condition.selected_token_sources,
			},
			'outputs': {
				name: _identity(staging / name, recorded_path=root / name)
				for name in REQUIRED_CONDITION_FILES
				if name != LABEL_BUDGET_METADATA_NAME
			},
		}
		_write_json(staging / LABEL_BUDGET_METADATA_NAME, metadata)
		staging.replace(root)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise


def _voxel_dataset_metadata(
	condition: VoxelLabelBudgetCondition,
	*,
	inspection: VoxelLabelBudgetDatasetInspection,
	output_dir: Path,
	suite_name: str,
) -> dict[str, object]:
	metadata = dict(inspection.common_metadata)
	source_summary = metadata.get('summary', {})
	metadata['outputs'] = {
		'supervision_split_grid': str(output_dir / GRID_NAME),
		'metadata_json': str(output_dir / METADATA_NAME),
		'class_counts_csv': str(output_dir / COUNTS_NAME),
		'split_manifest_json': str(output_dir / MANIFEST_NAME),
		'summary_markdown': str(output_dir / LABEL_BUDGET_SUMMARY_NAME),
	}
	metadata['summary'] = {
		'final_train_voxels': condition.train_voxel_count,
		'final_validation_voxels': condition.validation_voxel_count,
		'selected_token_row_count': condition.selected_token_rows,
		'unique_selected_token_xyz_count': int(condition.unique_token_xyz.shape[0]),
		'duplicate_selected_row_count': condition.duplicate_selected_rows,
		'source_full_summary': source_summary,
	}
	identity = _condition_identity(condition, inspection=inspection)
	metadata['voxel_label_budget'] = {
		'suite_name': suite_name,
		**identity,
		'budget_semantics': 'per_class_selected_token_row_cap',
		'dense_voxel_labels_preserved': True,
		'validation_reuse': 'canonical_full_validation_bitwise',
	}
	return metadata


def _condition_identity(
	condition: VoxelLabelBudgetCondition,
	*,
	inspection: VoxelLabelBudgetDatasetInspection,
) -> dict[str, object]:
	return {
		'budget_id': condition.budget_id,
		'per_class_cap': condition.per_class_cap,
		'subsample_seed': condition.subsample_seed,
		'patch_size_xyz': list(inspection.patch_size_xyz),
		'token_grid_shape_xyz': list(inspection.token_grid_shape_xyz),
		'volume_shape_xyz': list(inspection.volume_shape_xyz),
		'class_order': list(inspection.class_ids),
		'selected_token_row_identity_sha256': condition.selected_token_identity_sha256,
		'selected_token_identity_sha256': condition.selected_token_identity_sha256,
		'validation_token_identity_sha256': condition.validation_token_identity_sha256,
		'three_model_paired_token_identity_sha256': (
			condition.paired_token_identity_sha256
		),
		'unique_token_xyz_sha256': array_sha256(condition.unique_token_xyz),
		'selected_token_row_count': condition.selected_token_rows,
		'unique_selected_token_xyz_count': int(condition.unique_token_xyz.shape[0]),
		'duplicate_selected_row_count': condition.duplicate_selected_rows,
		'actual_train_voxel_count': condition.train_voxel_count,
		'validation_voxel_count': condition.validation_voxel_count,
		'per_class_train_voxel_counts': dict(condition.per_class_train_voxel_counts),
		'per_class_validation_voxel_counts': dict(
			condition.per_class_validation_voxel_counts
		),
		'train_mask_sha256': condition.train_mask_sha256,
		'validation_mask_sha256': condition.validation_mask_sha256,
		'grid_array_sha256': condition.grid_array_sha256,
	}


def _derived_condition_grid(
	condition: VoxelLabelBudgetCondition,
	*,
	inspection: VoxelLabelBudgetDatasetInspection,
) -> NDArray[np.integer]:
	grid = build_low_label_supervision_grid(
		inspection.common_grid,
		condition.unique_token_xyz,
		patch_size_xyz=inspection.patch_size_xyz,
	)
	if array_sha256(grid) != condition.grid_array_sha256:
		raise ValueError('derived low-label grid identity changed')
	return grid


def _condition_manifest_row(
	condition: VoxelLabelBudgetCondition, *, root: Path
) -> dict[str, object]:
	metadata = _read_json(root / LABEL_BUDGET_METADATA_NAME)
	identity = _mapping(metadata.get('identity'), 'identity')
	return {
		'budget_id': condition.budget_id,
		'per_class_cap': condition.per_class_cap,
		'subsample_seed': condition.subsample_seed,
		'voxel_dataset_root': str(root),
		'train_voxel_count': condition.train_voxel_count,
		'validation_voxel_count': condition.validation_voxel_count,
		'class_order': identity['class_order'],
		'per_class_train_voxel_counts': dict(condition.per_class_train_voxel_counts),
		'per_class_validation_voxel_counts': dict(
			condition.per_class_validation_voxel_counts
		),
		'selected_token_row_count': condition.selected_token_rows,
		'unique_selected_token_xyz_count': int(condition.unique_token_xyz.shape[0]),
		'duplicate_selected_row_count': condition.duplicate_selected_rows,
		'selected_token_identity_sha256': condition.selected_token_identity_sha256,
		'unique_token_xyz_sha256': identity['unique_token_xyz_sha256'],
		'train_mask_sha256': condition.train_mask_sha256,
		'validation_mask_sha256': condition.validation_mask_sha256,
		'supervision_split_grid': _identity(root / GRID_NAME),
		'voxel_dataset_metadata': _identity(root / METADATA_NAME),
		'class_counts': _identity(root / COUNTS_NAME),
		'split_manifest': _identity(root / MANIFEST_NAME),
		'voxel_label_budget_metadata': _identity(
			root / LABEL_BUDGET_METADATA_NAME
		),
		'voxel_label_budget_summary': _identity(root / LABEL_BUDGET_SUMMARY_NAME),
	}


def _suite_manifest_payload(
	config: F3VoxelLabelBudgetDatasetConfig,
	*,
	inspection: VoxelLabelBudgetDatasetInspection,
	rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	return {
		'artifact_type': MANIFEST_ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'suite': {
			'name': config.suite_name,
			'output_root': str(config.output_root),
			'budget_semantics': 'per_class_selected_token_row_cap',
		},
		'contract': {
			'budgets': list(config.budgets),
			'subsample_seeds': list(config.subsample_seeds),
			'patch_size_xyz': list(config.patch_size_xyz),
			'require_all_classes': config.require_all_classes,
			'validation': 'canonical_full_validation_bitwise',
		},
		'models': dict(config.models),
		'sources': dict(inspection.source_identities),
		'common_validation_mask_sha256': inspection.conditions[
			0
		].validation_mask_sha256,
		'condition_count': len(rows),
		'rows': list(rows),
	}


def _write_class_counts(
	path: Path,
	*,
	condition: VoxelLabelBudgetCondition,
	class_ids: Sequence[int],
	class_names: Sequence[str],
) -> None:
	rows = _condition_class_count_rows(
		condition,
		class_ids=class_ids,
		class_names=class_names,
	)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(
			file_obj,
			fieldnames=('split', 'class_id', 'class_name', 'count', 'fraction'),
		)
		writer.writeheader()
		writer.writerows(rows)


def _condition_class_count_rows(
	condition: VoxelLabelBudgetCondition,
	*,
	class_ids: Sequence[int],
	class_names: Sequence[str],
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for split, counts in (
		('train', condition.per_class_train_voxel_counts),
		('validation', condition.per_class_validation_voxel_counts),
	):
		total = sum(counts.values())
		for class_id, class_name in zip(class_ids, class_names, strict=True):
			count = int(counts[str(class_id)])
			rows.append(
				{
					'split': split,
					'class_id': class_id,
					'class_name': class_name,
					'count': count,
					'fraction': count / total if total else 0.0,
				}
			)
	all_counts = {
		str(class_id): condition.per_class_train_voxel_counts[str(class_id)]
		+ condition.per_class_validation_voxel_counts[str(class_id)]
		for class_id in class_ids
	}
	total = sum(all_counts.values())
	for class_id, class_name in zip(class_ids, class_names, strict=True):
		count = int(all_counts[str(class_id)])
		rows.append(
			{
				'split': 'all_supervised',
				'class_id': class_id,
				'class_name': class_name,
				'count': count,
				'fraction': count / total if total else 0.0,
			}
		)
	return rows


def _validate_condition_class_counts(
	path: Path,
	*,
	condition: VoxelLabelBudgetCondition,
	class_ids: Sequence[int],
	class_names: Sequence[str],
) -> None:
	expected = _condition_class_count_rows(
		condition,
		class_ids=class_ids,
		class_names=class_names,
	)
	with path.open(encoding='utf-8', newline='') as file_obj:
		reader = csv.DictReader(file_obj)
		if tuple(reader.fieldnames or ()) != (
			'split',
			'class_id',
			'class_name',
			'count',
			'fraction',
		):
			raise ValueError('condition class-count CSV header mismatch')
		actual = list(reader)
	if len(actual) != len(expected):
		raise ValueError('condition class-count CSV row count mismatch')
	for actual_row, expected_row in zip(actual, expected, strict=True):
		for key in ('split', 'class_id', 'class_name', 'count'):
			if actual_row.get(key) != str(expected_row[key]):
				raise ValueError(
					f'condition class-count CSV {key} mismatch'
				)
		try:
			fraction = float(actual_row['fraction'])
		except (KeyError, TypeError, ValueError) as error:
			raise ValueError(
				'condition class-count CSV fraction is invalid'
			) from error
		if fraction != float(expected_row['fraction']):
			raise ValueError('condition class-count CSV fraction mismatch')


def _render_condition_summary(condition: VoxelLabelBudgetCondition) -> str:
	return '\n'.join(
		[
			'# F3 voxel label-budget supervision',
			'',
			f'- budget: {condition.budget_id} (per-class selected token row cap)',
			f'- subsample seed: {condition.subsample_seed}',
			f'- selected token rows: {condition.selected_token_rows}',
			f'- unique selected token_xyz: {condition.unique_token_xyz.shape[0]}',
			f'- duplicate selected rows: {condition.duplicate_selected_rows}',
			f'- actual supervised train voxels: {condition.train_voxel_count}',
			f'- fixed full validation voxels: {condition.validation_voxel_count}',
			'- dense voxel labels inside selected blocks are preserved.',
			'',
		]
	)


def _validate_token_row(  # noqa: PLR0913
	row: Mapping[str, object],
	*,
	budget_id: str,
	per_class_cap: int,
	seed: int,
	expected_model_tag: str,
	expected_model_role: str,
) -> None:
	expected = {
		'budget_id': budget_id,
		'per_class_cap': per_class_cap,
		'subsample_seed': seed,
		'model_tag': expected_model_tag,
		'model_role': expected_model_role,
	}
	for key, value in expected.items():
		if row.get(key) != value:
			raise ValueError(f'token suite row identity mismatch: {key}')
	for key in ('selected_train_token_count', 'validation_token_count'):
		value = row.get(key)
		if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
			raise ValueError(f'token suite row {key} must be positive')
	paired_hash = row.get('paired_identity_hash')
	if not isinstance(paired_hash, str) or not paired_hash:
		raise ValueError('token suite row paired_identity_hash is required')
	for key in ('train_tokens', 'validation_tokens', 'metadata_json'):
		path_value = row.get(key)
		if not isinstance(path_value, str) or not Path(path_value).is_absolute():
			raise ValueError(f'token suite row {key} must be an absolute path')


def _validate_token_metadata(  # noqa: PLR0913
	metadata: Mapping[str, object],
	*,
	row: Mapping[str, object],
	budget_id: str,
	per_class_cap: int,
	seed: int,
	expected_model_tag: str,
	expected_model_role: str,
) -> None:
	if metadata.get('artifact_type') != EXPECTED_TOKEN_DATASET_TYPE:
		raise ValueError('invalid token label-budget dataset artifact type')
	budget = _mapping(metadata.get('label_budget'), 'token label_budget')
	model = _mapping(metadata.get('model'), 'token model')
	for actual, expected, label in (
		(budget.get('budget_id'), budget_id, 'budget_id'),
		(budget.get('per_class_cap'), per_class_cap, 'per_class_cap'),
		(budget.get('subsample_seed'), seed, 'subsample_seed'),
		(model.get('model_tag'), expected_model_tag, 'model_tag'),
		(model.get('role'), expected_model_role, 'model role'),
		(
			metadata.get('paired_identity_hash'),
			row.get('paired_identity_hash'),
			'paired hash',
		),
		(
			metadata.get('selected_train_token_count'),
			row.get('selected_train_token_count'),
			'selected train token count',
		),
		(
			metadata.get('validation_token_count'),
			row.get('validation_token_count'),
			'validation token count',
		),
		(
			metadata.get('selected_class_counts'),
			row.get('selected_class_counts'),
			'selected class counts',
		),
		(
			metadata.get('validation_class_counts'),
			row.get('validation_class_counts'),
			'validation class counts',
		),
	):
		if actual != expected:
			raise ValueError(f'token label-budget metadata mismatch: {label}')
	validation = _mapping(metadata.get('validation'), 'token validation')
	if validation.get('reuse_full_validation') is not True:
		raise ValueError('token label-budget validation must reuse full validation')


def _validate_loaded_token_datasets(  # noqa: C901, PLR0913
	train: F3LithologyTokenDataset,
	validation: F3LithologyTokenDataset,
	*,
	row: Mapping[str, object],
	metadata: Mapping[str, object],
	class_ids: Sequence[int],
	per_class_cap: int,
	require_all_classes: bool,
) -> None:
	if not np.all(np.asarray(train.split, dtype=str) == 'train'):
		raise ValueError('selected token dataset contains non-train rows')
	if not np.all(np.asarray(validation.split, dtype=str) == 'validation'):
		raise ValueError('validation token dataset contains non-validation rows')
	if train.count != row.get('selected_train_token_count'):
		raise ValueError('selected token row count differs from manifest')
	if validation.count != row.get('validation_token_count'):
		raise ValueError('validation token row count differs from manifest')
	train_counts = _token_class_counts(train.labels, class_ids=class_ids)
	validation_counts = _token_class_counts(
		validation.labels, class_ids=class_ids
	)
	if train_counts != row.get('selected_class_counts'):
		raise ValueError('selected token class counts differ from manifest')
	if validation_counts != row.get('validation_class_counts'):
		raise ValueError('validation token class counts differ from manifest')
	if train_counts != metadata.get('selected_class_counts'):
		raise ValueError('selected token class counts differ from metadata')
	if validation_counts != metadata.get('validation_class_counts'):
		raise ValueError('validation token class counts differ from metadata')
	if any(count > per_class_cap for count in train_counts.values()):
		raise ValueError('selected token class count exceeds per-class cap')
	if require_all_classes and any(count <= 0 for count in train_counts.values()):
		raise ValueError('selected token dataset is missing a required class')


def _token_class_counts(
	labels: NDArray[np.integer], *, class_ids: Sequence[int]
) -> dict[str, int]:
	values = np.asarray(labels)
	known = np.asarray(tuple(class_ids), dtype=np.int64)
	if np.any(~np.isin(values, known)):
		raise ValueError('token dataset contains a label outside the class order')
	return {
		str(class_id): int(np.count_nonzero(values == class_id))
		for class_id in class_ids
	}


def _read_token_manifest(path: Path) -> Mapping[str, object]:
	payload = _read_json(path)
	if payload.get('artifact_type') != EXPECTED_TOKEN_MANIFEST_TYPE:
		raise ValueError(f'invalid token label-budget manifest: {path}')
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes) or not rows:
		raise ValueError(f'token label-budget manifest rows are invalid: {path}')
	return payload


def _token_rows_by_key(
	payload: Mapping[str, object], *, source: str
) -> dict[tuple[str, str, int], Mapping[str, object]]:
	result: dict[tuple[str, str, int], Mapping[str, object]] = {}
	for index, value in enumerate(cast('Sequence[object]', payload['rows'])):
		if not isinstance(value, Mapping):
			raise TypeError(f'{source} token manifest row {index} must be a mapping')
		model_tag = value.get('model_tag')
		budget_id = value.get('budget_id')
		seed = value.get('subsample_seed')
		if (
			not isinstance(model_tag, str)
			or not isinstance(budget_id, str)
			or not isinstance(seed, int)
			or isinstance(seed, bool)
		):
			raise TypeError(f'{source} token manifest row identity is invalid')
		key = (model_tag, budget_id, seed)
		if key in result:
			raise ValueError(f'duplicate token label-budget manifest row: {key!r}')
		result[key] = value
	return result


def _validate_declared_common_source(
	metadata: Mapping[str, object], key: str
) -> None:
	identity = _mapping(metadata.get(key), f'canonical voxel {key}')
	path = _identity_path(identity, label=f'canonical voxel {key}')
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'canonical voxel {key} SHA-256 mismatch')


def _validate_common_metadata(metadata: Mapping[str, object]) -> None:
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
	classes = metadata.get('classes')
	if (
		not isinstance(classes, Sequence)
		or isinstance(classes, str | bytes)
		or not classes
	):
		raise ValueError('canonical voxel classes are invalid')
	class_ids: list[int] = []
	for index, item in enumerate(classes):
		if not isinstance(item, Mapping):
			raise TypeError(f'canonical voxel class {index} must be a mapping')
		class_id = item.get('class_id')
		class_name = item.get('class_name')
		if not isinstance(class_id, int) or isinstance(class_id, bool):
			raise TypeError(f'canonical voxel class {index} ID must be an integer')
		if not isinstance(class_name, str) or not class_name:
			raise TypeError(
				f'canonical voxel class {index} name must be non-empty'
			)
		class_ids.append(class_id)
	if len(class_ids) != len(set(class_ids)):
		raise ValueError('canonical voxel class IDs must be unique')


def _validate_full_grid(grid: NDArray[np.integer]) -> None:
	if grid.ndim != 3 or not np.issubdtype(grid.dtype, np.integer):
		raise TypeError('canonical split grid must be a 3D integer array')
	values = {int(item) for item in np.unique(grid)}
	if not values.issubset({0, 1, 2}) or not {1, 2}.issubset(values):
		raise ValueError(f'canonical split grid codes are invalid: {sorted(values)!r}')


def _validate_derived_grid(
	grid: NDArray[np.integer], *, full_grid: NDArray[np.integer]
) -> None:
	if grid.shape != full_grid.shape or grid.dtype != full_grid.dtype:
		raise ValueError('low-label grid must preserve canonical shape and dtype')
	train = grid == TRAIN_VOXEL_SPLIT
	validation = grid == VALIDATION_VOXEL_SPLIT
	if np.any(train & (full_grid != TRAIN_VOXEL_SPLIT)):
		raise ValueError('low-label train is not a subset of canonical full train')
	if not np.array_equal(validation, full_grid == VALIDATION_VOXEL_SPLIT):
		raise ValueError('low-label validation does not preserve canonical validation')
	if np.any(train & validation):
		raise ValueError('low-label train and validation overlap')


def _class_counts(
	labels: NDArray[np.integer], mask: NDArray[np.bool_], class_ids: Sequence[int]
) -> dict[str, int]:
	selected = np.asarray(labels[mask])
	return {
		str(class_id): int(np.count_nonzero(selected == class_id))
		for class_id in class_ids
	}


def _token_coordinates(
	value: NDArray[np.integer] | Sequence[Sequence[int]],
) -> NDArray[np.int64]:
	array = np.asarray(value)
	if array.ndim != 2 or array.shape[1] != 3:
		raise ValueError('token_xyz must have shape [N,3]')
	if array.shape[0] == 0:
		raise ValueError('token_xyz must contain at least one row')
	if not np.issubdtype(array.dtype, np.integer):
		raise TypeError('token_xyz must contain integers')
	return np.asarray(array, dtype=np.int64)


def _token_block(
	coordinate: Sequence[int],
	*,
	patch_size_xyz: Sequence[int],
	volume_shape_xyz: Sequence[int],
) -> tuple[slice, slice, slice]:
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	shape = _positive_triplet(volume_shape_xyz, 'volume_shape_xyz')
	start = tuple(int(coordinate[axis]) * patch[axis] for axis in range(3))
	stop = tuple(min(start[axis] + patch[axis], shape[axis]) for axis in range(3))
	return _slices(start, stop)


def array_sha256(array: NDArray[np.generic]) -> str:
	"""Hash dtype, shape, and C-order bytes for a stable ndarray identity."""
	value = np.asarray(array)
	hasher = hashlib.sha256()
	hasher.update(value.dtype.str.encode('ascii'))
	hasher.update(json.dumps(list(value.shape), separators=(',', ':')).encode('ascii'))
	if value.flags.c_contiguous:
		flat = value.reshape(-1)
		for start in range(0, flat.size, 1_000_000):
			stop = min(start + 1_000_000, flat.size)
			hasher.update(flat[start:stop].view(np.uint8))
	else:
		hasher.update(np.ascontiguousarray(value).view(np.uint8))
	return hasher.hexdigest()


def _split_mask_identity(
	grid: NDArray[np.integer], split_code: int
) -> tuple[str, int]:
	"""Hash and count one boolean split mask without materializing the volume."""
	hasher = hashlib.sha256()
	hasher.update(np.dtype(np.bool_).str.encode('ascii'))
	hasher.update(json.dumps(list(grid.shape), separators=(',', ':')).encode('ascii'))
	count = 0
	flat = grid.reshape(-1)
	for start in range(0, flat.size, 1_000_000):
		stop = min(start + 1_000_000, flat.size)
		mask = np.asarray(flat[start:stop] == split_code, dtype=np.bool_)
		hasher.update(mask.view(np.uint8))
		count += int(np.count_nonzero(mask))
	return hasher.hexdigest(), count


def _identity(path: Path, *, recorded_path: Path | None = None) -> dict[str, str]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path if recorded_path is None else recorded_path),
		'sha256': file_sha256(path),
	}


def _validate_identity_record(value: object, path: Path, *, label: str) -> None:
	identity = _mapping(value, label)
	if Path(str(identity.get('path'))).resolve(strict=False) != path.resolve(
		strict=False
	):
		raise ValueError(f'{label} path mismatch')
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} SHA-256 mismatch')


def _identity_path(value: Mapping[str, object], *, label: str) -> Path:
	path_value = value.get('path')
	if not isinstance(path_value, str) or not path_value:
		raise ValueError(f'{label}.path is required')
	path = Path(path_value)
	if not path.is_file():
		raise FileNotFoundError(path)
	return path


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


def _without_actions(value: object) -> object:
	if isinstance(value, Mapping):
		return {
			key: _without_actions(child)
			for key, child in value.items()
			if key != 'action'
		}
	if isinstance(value, list):
		return [_without_actions(child) for child in value]
	return value


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON document must contain an object: {path}')
	return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.tmp')
	_write_json(temporary, payload)
	temporary.replace(path)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'{label} must be an integer triple')
	items = tuple(value)
	if any(not isinstance(item, int | np.integer) or int(item) <= 0 for item in items):
		raise ValueError(f'{label} must contain positive integers')
	return (int(items[0]), int(items[1]), int(items[2]))


def _slices(start: Sequence[int], stop: Sequence[int]) -> tuple[slice, slice, slice]:
	return tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))  # type: ignore[return-value]


__all__ = [
	'ARTIFACT_TYPE',
	'DATASET_MANIFEST_NAME',
	'LABEL_BUDGET_METADATA_NAME',
	'MANIFEST_ARTIFACT_TYPE',
	'VoxelLabelBudgetCondition',
	'VoxelLabelBudgetDatasetBuildResult',
	'VoxelLabelBudgetDatasetInspection',
	'array_sha256',
	'build_f3_lithology_voxel_label_budget_datasets',
	'build_low_label_supervision_grid',
	'expand_selected_token_blocks',
	'inspect_f3_lithology_voxel_label_budget_datasets',
	'validate_voxel_label_budget_condition',
	'validate_voxel_label_budget_condition_artifact',
]
