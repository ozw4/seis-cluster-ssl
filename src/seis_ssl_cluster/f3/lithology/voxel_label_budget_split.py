"""Build the encoder-independent six-split low-label voxel datasets.

The split inventory is strictly read-only.  Token labels are used only to draw
the canonical MAE rows; the dense labels in each voxel dataset remain the
supervision source.
"""
# ruff: noqa: C901, D101, E501, PLR0911, PLR0913, TRY004

from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.robustness import (
	assert_same_token_identity,
	class_stratified_subset_indices,
	load_token_dataset_npz,
	paired_token_identity_hash,
	subset_token_dataset,
)
from seis_ssl_cluster.f3.lithology.voxel_dataset import GRID_NAME, METADATA_NAME
from seis_ssl_cluster.f3.lithology.voxel_label_budget import (
	array_sha256,
	build_low_label_supervision_grid,
	token_block,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import _json_sha256
from seis_ssl_cluster.f3.lithology.voxel_split import (
	TRAIN_VOXEL_SPLIT,
	VALIDATION_VOXEL_SPLIT,
)
from seis_ssl_cluster.f3.splits import read_f3_line_geometry

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_split import (
		F3VoxelLabelBudgetSplitConfig,
	)

MANIFEST_NAME = 'low_label_split_dataset_manifest.json'


@dataclass(frozen=True)
class LowLabelSplitCondition:
	split_id: str
	budget_id: str
	output_root: Path
	row: Mapping[str, object]
	selected_xyz: np.ndarray
	metadata: Mapping[str, object]


@dataclass(frozen=True)
class LowLabelSplitInspection:
	conditions: tuple[LowLabelSplitCondition, ...]


def inspect_f3_lithology_voxel_label_budget_split_datasets(
	config: F3VoxelLabelBudgetSplitConfig,
) -> LowLabelSplitInspection:
	"""Validate immutable inputs and derive the complete 6 x 2 matrix."""
	inventory = _manifest(config.split_inventory_manifest)
	if [item.get('split_id') for item in inventory['rows']] != list(config.split_ids):
		raise ValueError(
			'split inventory must contain the canonical ordered six splits'
		)
	if inventory['rows'][0].get('random_seed') is not None:
		raise ValueError('split_000 must be the original/base split')
	token_manifest = _manifest(config.split_dataset_manifest)
	voxel_manifest = _manifest(config.voxel_dataset_manifest)
	_validate_full_voxel_sources(config, inventory, token_manifest, voxel_manifest)
	tokens = _rows_by_split_model(token_manifest['rows'])
	voxels = {str(row['split_id']): row for row in voxel_manifest['rows']}
	conditions = []
	for split_id in config.split_ids:
		baseline = tokens[(split_id, 'baseline')]
		historical = tokens[(split_id, 'candidate')]
		train = load_token_dataset_npz(Path(str(baseline['train_tokens'])))
		other = load_token_dataset_npz(Path(str(historical['train_tokens'])))
		assert_same_token_identity(
			train,
			other,
			reference_label='MAE train',
			candidate_label='historical M1 train',
		)
		if split_id not in voxels:
			raise ValueError(f'missing full-label voxel source for {split_id}')
		for budget_id, cap in zip(config.budgets, (25, 50), strict=True):
			selected = class_stratified_subset_indices(
				train.labels, per_class_cap=cap, seed=0, require_all_classes=True
			)
			conditions.append(
				_condition(
					split_id, budget_id, cap, train, selected, voxels[split_id], config
				)
			)
	return LowLabelSplitInspection(tuple(conditions))


def build_f3_lithology_voxel_label_budget_split_datasets(
	config: F3VoxelLabelBudgetSplitConfig, *, only_missing: bool = False
) -> tuple[Path, tuple[Mapping[str, object], ...]]:
	"""Build or validate the twelve datasets and atomically publish a manifest."""
	inspection = inspect_f3_lithology_voxel_label_budget_split_datasets(config)
	# The base split is a publication gate, not a post-write diagnostic.
	_parity_gate([condition.row for condition in inspection.conditions], config)
	rows = []
	for condition in inspection.conditions:
		quarantine_path = None
		quarantine_reason = None
		if condition.output_root.exists():
			if not only_missing:
				raise FileExistsError(condition.output_root)
			if _complete(condition.output_root, condition.row):
				rows.append({
					**condition.row,
					'action': 'REUSED',
					'quarantine_path': None,
					'quarantine_reason': None,
				})
				continue
			quarantine_reason = _quarantine_reason(condition.output_root)
			quarantine = condition.output_root.with_name(
				condition.output_root.name + f'.quarantine-{_timestamp()}'
			)
			condition.output_root.replace(quarantine)
			quarantine_path = str(quarantine)
		_write_condition(condition)
		rows.append({
			**condition.row,
			'action': 'NEW',
			'quarantine_path': quarantine_path,
			'quarantine_reason': quarantine_reason,
		})
	manifest = config.output_root / MANIFEST_NAME
	_write_json(
		manifest,
		{
			'artifact_type': 'f3_lithology_voxel_label_budget_split_dataset_manifest',
			'schema_version': 1,
			'contract': {
				'split_ids': list(config.split_ids),
				'budgets': list(config.budgets),
				'label_subset_seed': 0,
			},
			'sources': {
				name: _identity(path)
				for name, path in {
					'split_inventory_manifest': config.split_inventory_manifest,
					'split_dataset_manifest': config.split_dataset_manifest,
					'voxel_dataset_manifest': config.voxel_dataset_manifest,
				}.items()
			},
			'rows': rows,
		},
	)
	return manifest, tuple(rows)


def _condition(
	split_id: str,
	budget_id: str,
	cap: int,
	train: object,
	selected: np.ndarray,
	voxel_row: Mapping[str, object],
	config: F3VoxelLabelBudgetSplitConfig,
) -> LowLabelSplitCondition:
	coordinates = np.asarray(train.token_xyz)[selected].astype(np.int64, copy=False)
	unique = np.unique(coordinates, axis=0)
	grid_path = Path(str(_mapping(voxel_row['split_grid'])['path']))
	full_grid = np.load(grid_path, mmap_mode='r', allow_pickle=False)
	_require_selected_tokens_cover_train_voxels(
		unique, full_grid, split_id=split_id, budget_id=budget_id
	)
	grid = build_low_label_supervision_grid(full_grid, unique, patch_size_xyz=(8, 8, 8))
	del full_grid
	metadata = dict(_read_json(Path(str(_mapping(voxel_row['metadata'])['path']))))
	source_identities = _normalized_source_identities(
		config, metadata, split_id=split_id
	)
	metadata['source_identities'] = source_identities
	labels = np.load(
		str(_mapping(metadata['label_volume'])['path']),
		mmap_mode='r',
		allow_pickle=False,
	)
	classes = [int(item['class_id']) for item in metadata['classes']]
	train_mask_sha256, train_voxel_count, per_class_train_voxel_counts = (
		_split_mask_identity_and_counts(grid, labels, TRAIN_VOXEL_SPLIT, classes)
	)
	(
		validation_mask_sha256,
		validation_voxel_count,
		per_class_validation_voxel_counts,
	) = _split_mask_identity_and_counts(
		grid, labels, VALIDATION_VOXEL_SPLIT, classes
	)
	selected_dataset = subset_token_dataset(train, selected)
	selected_identity = paired_token_identity_hash(selected_dataset)
	row = {
		'split_id': split_id,
		'budget_id': budget_id,
		'per_class_cap': cap,
		'label_subset_seed': 0,
		'voxel_dataset_root': str(
			config.output_root / 'datasets' / split_id / budget_id / 'voxel_supervision'
		),
		'selected_token_row_count': int(selected.size),
		'unique_selected_token_xyz_count': int(unique.shape[0]),
		'duplicate_selected_row_count': int(selected.size - unique.shape[0]),
		'selected_token_identity_sha256': selected_identity,
		'unique_token_xyz_sha256': array_sha256(unique),
		'train_voxel_count': train_voxel_count,
		'actual_train_voxel_count': train_voxel_count,
		'validation_voxel_count': validation_voxel_count,
		'per_class_train_voxel_counts': per_class_train_voxel_counts,
		'per_class_validation_voxel_counts': per_class_validation_voxel_counts,
		'train_mask_sha256': train_mask_sha256,
		'validation_mask_sha256': validation_mask_sha256,
		'grid_array_sha256': array_sha256(grid),
		'supervision_split_grid': {
			'path': str(
				config.output_root
				/ 'datasets'
				/ split_id
				/ budget_id
				/ 'voxel_supervision'
				/ GRID_NAME
			),
			'sha256': _npy_sha256(grid),
		},
		'canonical_valid_tokens_sha256': _mapping(voxel_row['reference_valid_tokens'])[
			'sha256'
		],
		'class_order': classes,
		'patch_size_xyz': [8, 8, 8],
		'source_identities': source_identities,
		'source_identities_sha256': _json_sha256(source_identities),
		'source_full_voxel_dataset': voxel_row,
	}
	del grid
	gc.collect()
	return LowLabelSplitCondition(
		split_id,
		budget_id,
		Path(str(row['voxel_dataset_root'])),
		row,
		unique,
		metadata,
	)


def _write_condition(condition: LowLabelSplitCondition) -> None:
	root = condition.output_root
	root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(tempfile.mkdtemp(prefix='.low-label-', dir=root.parent))
	try:
		source = _mapping(condition.row['source_full_voxel_dataset'])
		source_grid = np.load(
			str(_mapping(source['split_grid'])['path']),
			mmap_mode='r',
			allow_pickle=False,
		)
		grid = build_low_label_supervision_grid(
			source_grid, condition.selected_xyz, patch_size_xyz=(8, 8, 8)
		)
		_assert_grid_identity(grid, condition.row)
		np.save(staging / GRID_NAME, grid, allow_pickle=False)
		np.save(
			staging / 'selected_token_xyz.npy',
			condition.selected_xyz,
			allow_pickle=False,
		)
		metadata = dict(condition.metadata)
		metadata['outputs'] = {
			'supervision_split_grid': str(root / GRID_NAME),
			'metadata_json': str(root / METADATA_NAME),
		}
		metadata_identity = {
			key: value
			for key, value in condition.row.items()
			if key not in {'source_full_voxel_dataset', 'source_identities_sha256'}
		}
		metadata_identity['source_identities_sha256'] = condition.row[
			'source_identities_sha256'
		]
		metadata['voxel_label_budget_split'] = {
			'split_id': condition.split_id,
			'budget_id': condition.budget_id,
			'dense_voxel_labels_preserved': True,
			'validation_reuse': 'canonical_full_validation_bitwise',
			'identity': metadata_identity,
		}
		_write_json(staging / METADATA_NAME, metadata)
		shutil.copyfile(
			str(_mapping(source['slice_split_manifest'])['path']),
			staging / 'split_manifest.json',
		)
		with (staging / 'class_counts.csv').open(
			'w', encoding='utf-8', newline=''
		) as handle:
			writer = csv.DictWriter(handle, fieldnames=('split', 'class_id', 'count'))
			writer.writeheader()
			for split, counts in (
				('train', condition.row['per_class_train_voxel_counts']),
				('validation', condition.row['per_class_validation_voxel_counts']),
			):
				for class_id, count in _mapping(counts).items():
					writer.writerow(
						{'split': split, 'class_id': class_id, 'count': count}
					)
		_write_json(
			staging / 'low_label_split_metadata.json',
			{
				'artifact_type': 'f3_lithology_voxel_label_budget_split_dataset',
				'identity': {
					key: value
					for key, value in condition.row.items()
					if key != 'source_full_voxel_dataset'
				},
				'sources': {
					'full_voxel_dataset': source,
					'normalized_source_identities': condition.row['source_identities'],
				},
			},
		)
		staging.replace(root)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise


def _assert_grid_identity(grid: np.ndarray, row: Mapping[str, object]) -> None:
	if array_sha256(grid) != row['grid_array_sha256']:
		raise ValueError('derived grid identity changed before commit')


def _complete(root: Path, row: Mapping[str, object]) -> bool:
	"""Return whether every published dataset artifact matches its identity."""
	try:
		grid_path = root / GRID_NAME
		selected_path = root / 'selected_token_xyz.npy'
		metadata_path = root / METADATA_NAME
		provenance_path = root / 'low_label_split_metadata.json'
		class_counts_path = root / 'class_counts.csv'
		split_manifest_path = root / 'split_manifest.json'
		if not all(
			path.is_file()
			for path in (
				grid_path,
				selected_path,
				metadata_path,
				provenance_path,
				class_counts_path,
				split_manifest_path,
			)
		):
			return False
		if (
			array_sha256(np.load(grid_path, mmap_mode='r', allow_pickle=False))
			!= row['grid_array_sha256']
		):
			return False
		if (
			array_sha256(np.load(selected_path, mmap_mode='r', allow_pickle=False))
			!= row['unique_token_xyz_sha256']
		):
			return False
		if file_sha256(grid_path) != _mapping(row['supervision_split_grid'])['sha256']:
			return False
		expected_identity = {
			key: value
			for key, value in row.items()
			if key
			not in {
				'source_full_voxel_dataset',
				'action',
				'quarantine_path',
				'quarantine_reason',
			}
		}
		provenance = _read_json(provenance_path)
		if (
			provenance.get('artifact_type')
			!= 'f3_lithology_voxel_label_budget_split_dataset'
			or provenance.get('identity') != expected_identity
			or provenance.get('sources')
			!= {
				'full_voxel_dataset': row['source_full_voxel_dataset'],
				'normalized_source_identities': row['source_identities'],
			}
		):
			return False
		metadata = _read_json(metadata_path)
		if not _source_identities_complete(
			_mapping(metadata.get('source_identities')), row
		):
			return False
		metadata_split = _mapping(metadata.get('voxel_label_budget_split'))
		metadata_identity = _mapping(metadata_split.get('identity'))
		if (
			metadata_identity.get('source_identities_sha256')
			!= row['source_identities_sha256']
		):
			return False
		if metadata_split != {
			'split_id': row['split_id'],
			'budget_id': row['budget_id'],
			'dense_voxel_labels_preserved': True,
			'validation_reuse': 'canonical_full_validation_bitwise',
			'identity': expected_identity,
		}:
			return False
		if metadata.get('outputs') != {
			'supervision_split_grid': str(grid_path),
			'metadata_json': str(metadata_path),
		}:
			return False
		source = _mapping(row['source_full_voxel_dataset'])
		if (
			file_sha256(split_manifest_path)
			!= _mapping(source['slice_split_manifest'])['sha256']
		):
			return False
		with class_counts_path.open(encoding='utf-8', newline='') as handle:
			class_counts = list(csv.DictReader(handle))
		expected_counts = [
			{'split': split, 'class_id': str(class_id), 'count': str(count)}
			for split, counts in (
				('train', _mapping(row['per_class_train_voxel_counts'])),
				('validation', _mapping(row['per_class_validation_voxel_counts'])),
			)
			for class_id, count in counts.items()
		]
	except (OSError, TypeError, ValueError, json.JSONDecodeError):
		return False
	else:
		return class_counts == expected_counts


def _quarantine_reason(root: Path) -> str:
	"""Return an audit reason for a dataset that failed completion validation."""
	try:
		metadata = _read_json(root / METADATA_NAME)
		provenance = _read_json(root / 'low_label_split_metadata.json')
		if (
			not isinstance(metadata.get('source_identities'), Mapping)
			or not isinstance(
				_mapping(provenance.get('sources')).get(
					'normalized_source_identities'
				),
				Mapping,
			)
		):
			return 'legacy_missing_normalized_source_identity'
	except (OSError, TypeError, ValueError, json.JSONDecodeError):
		pass
	return 'incomplete_or_mismatched_dataset'


def _source_identities_complete(
	sources: Mapping[str, object], row: Mapping[str, object]
) -> bool:
	"""Require the committed normalized source identities to be live and exact."""
	expected = _mapping(row.get('source_identities'))
	expected_sha256 = row.get('source_identities_sha256')
	if sources != expected or not isinstance(expected_sha256, str):
		return False
	if _json_sha256(expected) != expected_sha256:
		return False
	for name in (
		'class_info',
		'source_label_segy',
		'segy_geometry_json',
		'seismic_volume',
	):
		identity = _mapping(expected.get(name))
		path = identity.get('path')
		sha256 = identity.get('sha256')
		if (
			not isinstance(path, str)
			or not Path(path).is_absolute()
			or not Path(path).is_file()
			or not isinstance(sha256, str)
			or file_sha256(Path(path)) != sha256
		):
			return False
	return set(expected) == {
		'class_info',
		'source_label_segy',
		'segy_geometry_json',
		'seismic_volume',
	}


def _parity_gate(
	rows: list[Mapping[str, object]], config: F3VoxelLabelBudgetSplitConfig
) -> None:
	original = _manifest(config.original_dataset_manifest)
	by_budget = {
		str(row['budget_id']): row
		for row in original['rows']
		if row.get('subsample_seed') == 0
	}
	for row in rows:
		if row['split_id'] != 'split_000':
			continue
		prior = by_budget.get(str(row['budget_id']))
		if prior is None:
			raise ValueError('original-split parity reference is missing')
		comparisons = {
			'selected_token_identity_sha256': row['selected_token_identity_sha256'],
			'unique_token_xyz_sha256': row['unique_token_xyz_sha256'],
			'train_mask_sha256': row['train_mask_sha256'],
			'validation_mask_sha256': row['validation_mask_sha256'],
			'actual_train_voxel_count': row['actual_train_voxel_count'],
			'validation_voxel_count': row['validation_voxel_count'],
			'per_class_train_voxel_counts': row['per_class_train_voxel_counts'],
			'per_class_validation_voxel_counts': row['per_class_validation_voxel_counts'],
			'class_order': row['class_order'],
		}
		prior_identity = _original_identity(prior)
		for key, expected in comparisons.items():
			if prior_identity.get(key, prior.get(key)) != expected:
				raise ValueError(f'split_000 parity mismatch: {key}')
		prior_grid_array = prior_identity.get('grid_array_sha256')
		if prior_grid_array is None:
			prior_grid = np.load(
				str(_mapping(prior['supervision_split_grid'])['path']),
				mmap_mode='r', allow_pickle=False,
			)
			prior_grid_array = array_sha256(prior_grid)
		if prior_grid_array != row['grid_array_sha256']:
			raise ValueError('split_000 parity mismatch: grid_array_sha256')


def _manifest(path: Path) -> Mapping[str, object]:
	payload = _read_json(path)
	if not isinstance(payload.get('rows'), list):
		raise ValueError(f'manifest lacks rows: {path}')
	return payload


def _rows_by_split_model(
	rows: object,
) -> Mapping[tuple[str, str], Mapping[str, object]]:
	if not isinstance(rows, list):
		raise TypeError('token manifest rows must be a list')
	return {
		(str(row['split_id']), str(row['model_role'])): _mapping(row) for row in rows
	}


def _validate_full_voxel_sources(
	config: F3VoxelLabelBudgetSplitConfig,
	inventory_manifest: Mapping[str, object],
	token_manifest: Mapping[str, object],
	voxel_manifest: Mapping[str, object],
) -> None:
	"""Require the read-only voxel suite to match the selected split sources."""
	if (
		voxel_manifest.get('artifact_type')
		!= 'f3_lithology_voxel_split_dataset_manifest'
	):
		raise ValueError('unexpected full-label voxel dataset manifest type')
	_validate_identity(
		_mapping(voxel_manifest.get('source_split_inventory_manifest')),
		config.split_inventory_manifest,
		label='full-label voxel split inventory',
	)
	token_suite = _mapping(token_manifest.get('suite'))
	if Path(str(token_suite.get('split_inventory_manifest'))).resolve(
		strict=False
	) != config.split_inventory_manifest.resolve(strict=False):
		raise ValueError('token and configured split inventories differ')
	canonical_valid = _mapping(voxel_manifest.get('canonical_reference_valid_tokens'))
	_validate_identity(
		canonical_valid,
		config.embeddings['mae'] / Path(str(canonical_valid.get('path'))).name,
		label='full-label voxel canonical valid tokens',
	)
	inventory_rows = {
		str(row['split_id']): _mapping(row) for row in inventory_manifest['rows']
	}
	baseline_token_rows = _rows_by_split_model(token_manifest['rows'])
	dense_label_identity: Mapping[str, object] | None = None
	for raw_row in voxel_manifest['rows']:
		row = _mapping(raw_row)
		split_id = str(row.get('split_id'))
		if split_id not in inventory_rows:
			raise ValueError(f'full-label voxel source has unknown split: {split_id}')
		dense_label = _validate_full_voxel_row(
			row,
			split_id=split_id,
			inventory_row=inventory_rows[split_id],
			baseline_token_row=baseline_token_rows[(split_id, 'baseline')],
			canonical_valid=canonical_valid,
		)
		if dense_label_identity is None:
			dense_label_identity = dense_label
			_validate_identity(
				dense_label,
				Path(str(dense_label.get('path'))),
				label='full-label dense label volume',
			)
		elif dense_label != dense_label_identity:
			raise ValueError(
				f'{split_id} dense label source differs from canonical source'
			)


def _validate_full_voxel_row(
	row: Mapping[str, object],
	*,
	split_id: str,
	inventory_row: Mapping[str, object],
	baseline_token_row: Mapping[str, object],
	canonical_valid: Mapping[str, object],
) -> Mapping[str, object]:
	grid_identity = _mapping(row.get('split_grid'))
	metadata_identity = _mapping(row.get('metadata'))
	_validate_identity(
		grid_identity,
		Path(str(grid_identity.get('path'))),
		label=f'{split_id} full-label split grid',
	)
	_validate_identity(
		metadata_identity,
		Path(str(metadata_identity.get('path'))),
		label=f'{split_id} full-label metadata',
	)
	slice_manifest = _mapping(row.get('slice_split_manifest'))
	_validate_identity(
		slice_manifest,
		Path(str(slice_manifest.get('path'))),
		label=f'{split_id} full-label slice split manifest',
	)
	if _mapping(row.get('reference_valid_tokens')) != canonical_valid:
		raise ValueError(
			f'{split_id} full-label valid-token identity differs from canonical reference'
		)
	metadata = _read_json(Path(str(metadata_identity.get('path'))))
	_validate_identity(
		_mapping(metadata.get('inventory')),
		Path(str(inventory_row.get('png_label_inventory'))),
		label=f'{split_id} full-label inventory',
	)
	if _mapping(metadata.get('reference_valid_tokens')) != canonical_valid:
		raise ValueError(
			f'{split_id} metadata valid-token identity differs from canonical reference'
		)
	if Path(
		str(_mapping(metadata.get('outputs')).get('supervision_split_grid'))
	).resolve(strict=False) != Path(str(grid_identity.get('path'))).resolve(
		strict=False
	):
		raise ValueError(f'{split_id} metadata split-grid path differs from manifest')
	dense_label = _mapping(metadata.get('label_volume'))
	baseline_metadata = _read_json(Path(str(baseline_token_row.get('metadata_json'))))
	if Path(str(_mapping(baseline_metadata.get('inputs')).get('label_volume'))).resolve(
		strict=False
	) != Path(str(dense_label.get('path'))).resolve(strict=False):
		raise ValueError(
			f'{split_id} dense label source differs from paired token dataset'
		)
	grid = np.load(str(grid_identity.get('path')), mmap_mode='r', allow_pickle=False)
	labels = np.load(str(dense_label.get('path')), mmap_mode='r', allow_pickle=False)
	if grid.shape != labels.shape or list(grid.shape) != _mapping(
		metadata.get('geometry')
	).get('shape_xyz'):
		raise ValueError(
			f'{split_id} full-label split-grid geometry differs from dense labels'
		)
	return dense_label


def _split_mask_identity_and_counts(
	grid: np.ndarray,
	labels: np.ndarray,
	split_code: int,
	classes: list[int],
) -> tuple[str, int, Mapping[str, int]]:
	"""Build one canonical boolean mask on disk to bound inspection memory."""
	with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as handle:
		mask_path = Path(handle.name)
	try:
		mask = np.lib.format.open_memmap(
			mask_path, mode='w+', dtype=np.bool_, shape=grid.shape
		)
		counts = {str(value): 0 for value in classes}
		count = 0
		for start in range(0, grid.shape[0], 8):
			stop = min(start + 8, grid.shape[0])
			chunk_mask = np.asarray(grid[start:stop] == split_code, dtype=np.bool_)
			mask[start:stop] = chunk_mask
			count += int(np.count_nonzero(chunk_mask))
			label_chunk = labels[start:stop]
			for value in classes:
				counts[str(value)] += int(
					np.count_nonzero(label_chunk[chunk_mask] == value)
				)
		mask.flush()
		return array_sha256(mask), count, counts
	finally:
		with suppress(UnboundLocalError):
			del mask
		mask_path.unlink(missing_ok=True)


def _require_selected_tokens_cover_train_voxels(
	coordinates: np.ndarray,
	full_grid: np.ndarray,
	*,
	split_id: str = 'unknown',
	budget_id: str = 'unknown',
) -> None:
	for coordinate in coordinates:
		block = token_block(
			coordinate, patch_size_xyz=(8, 8, 8), volume_shape_xyz=full_grid.shape
		)
		if not np.any(np.asarray(full_grid[block]) == TRAIN_VOXEL_SPLIT):
			raise ValueError(
				'selected unique token has no canonical train voxel: '
				f'{split_id}/{budget_id}/{coordinate.tolist()}'
			)


def _normalized_source_identities(
	config: F3VoxelLabelBudgetSplitConfig,
	metadata: Mapping[str, object],
	*,
	split_id: str,
) -> Mapping[str, Mapping[str, str]]:
	"""Validate current or legacy full-label provenance and normalize it."""
	strict_sources = _strict_source_identities(config, split_id=split_id)
	labels = _mapping(metadata.get('labels'))
	reference = _mapping(metadata.get('reference_embedding'))
	reference_metadata = _mapping(reference.get('metadata'))
	expected = {
		'class_info': (config.class_info, labels.get('class_info')),
		'source_label_segy': (config.source_label_segy, labels.get('source_label_segy')),
		'seismic_volume': (config.seismic_volume, reference_metadata.get('source_amplitude_path')),
	}
	for name, (path, recorded_path) in expected.items():
		if not path.is_file():
			raise FileNotFoundError(f'{split_id} missing strict {name}: {path}')
		if Path(str(recorded_path)).resolve(strict=False) != path.resolve(strict=False):
			raise ValueError(f'{split_id} legacy {name} path differs from strict config')
	if not config.segy_geometry_json.is_file():
		raise FileNotFoundError(
			f'{split_id} missing strict segy_geometry_json: {config.segy_geometry_json}'
		)
	if dict(read_f3_line_geometry(config.segy_geometry_json).to_dict()) != dict(
		_mapping(metadata.get('geometry'))
	):
		raise ValueError(f'{split_id} legacy SEGY geometry differs from strict config')
	sources = metadata.get('source_identities')
	if sources is not None:
		current = _mapping(sources)
		if current != strict_sources:
			raise ValueError(
				f'{split_id} full-label source identities differ from strict config'
			)
	return strict_sources


def _strict_source_identities(
	config: F3VoxelLabelBudgetSplitConfig, *, split_id: str
) -> Mapping[str, Mapping[str, str]]:
	"""Validate the configured immutable source identity anchor before reuse."""
	configured = _mapping(config.source_identities)
	paths = {
		'class_info': config.class_info,
		'source_label_segy': config.source_label_segy,
		'segy_geometry_json': config.segy_geometry_json,
		'seismic_volume': config.seismic_volume,
	}
	if set(configured) != set(paths):
		raise ValueError(f'{split_id} strict source identity keys are invalid')
	for name, path in paths.items():
		_validate_identity(
			_mapping(configured.get(name)),
			path,
			label=f'{split_id} strict source {name}',
		)
	return {
		name: {
			'path': str(_mapping(configured[name])['path']),
			'sha256': str(_mapping(configured[name])['sha256']),
		}
		for name in paths
	}


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('expected mapping')
	return value


def _validate_identity(value: Mapping[str, object], path: Path, *, label: str) -> None:
	recorded_path = value.get('path')
	if not isinstance(recorded_path, str) or Path(recorded_path).resolve(
		strict=False
	) != path.resolve(strict=False):
		raise ValueError(f'{label} path identity mismatch')
	if value.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} hash identity mismatch')


def _read_json(path: Path) -> Mapping[str, object]:
	return _mapping(json.loads(path.read_text(encoding='utf-8')))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


def _identity(path: Path) -> Mapping[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _array_sha(value: np.ndarray) -> str:
	"""Compatibility alias for the canonical array identity helper."""
	return array_sha256(value)


def _npy_sha256(value: np.ndarray) -> str:
	"""Return the SHA-256 of the exact .npy bytes written by ``np.save``."""
	buffer = io.BytesIO()
	np.save(buffer, value, allow_pickle=False)
	return file_sha256_bytes(buffer.getvalue())


def file_sha256_bytes(value: bytes) -> str:
	"""Hash a small in-memory committed artifact."""
	return hashlib.sha256(value).hexdigest()


def _original_identity(row: Mapping[str, object]) -> Mapping[str, object]:
	metadata_record = row.get('voxel_label_budget_metadata')
	if not isinstance(metadata_record, Mapping):
		return {}
	metadata_path = Path(str(metadata_record.get('path', '')))
	if not metadata_path.is_file() or metadata_record.get('sha256') != file_sha256(metadata_path):
		raise ValueError('original-split parity metadata identity mismatch')
	identity = _read_json(metadata_path).get('identity')
	return _mapping(identity) if isinstance(identity, Mapping) else {}


def _timestamp() -> str:
	return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


__all__ = [
	'LowLabelSplitCondition',
	'LowLabelSplitInspection',
	'build_f3_lithology_voxel_label_budget_split_datasets',
	'inspect_f3_lithology_voxel_label_budget_split_datasets',
]
