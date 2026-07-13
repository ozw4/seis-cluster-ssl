"""Build the canonical encoder-independent F3 voxel supervision artifact."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info
from seis_ssl_cluster.f3.lithology.voxel_split import (
	TRAIN_VOXEL_SPLIT,
	VALIDATION_VOXEL_SPLIT,
	build_f3_voxel_supervision_split,
)
from seis_ssl_cluster.f3.splits import (
	f3_slice_split_manifest,
	load_f3_slice_split_records,
	read_f3_line_geometry,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_dataset import (
		F3LithologyVoxelDatasetConfig,
	)
	from seis_ssl_cluster.f3.io.labels import F3ClassInfo
	from seis_ssl_cluster.f3.lithology.voxel_split import F3VoxelSupervisionSplit
	from seis_ssl_cluster.f3.splits import F3LineGeometry, F3SliceSplitRecord

GRID_NAME = 'supervision_split_grid.npy'
METADATA_NAME = 'voxel_dataset_metadata.json'
COUNTS_NAME = 'class_counts.csv'
MANIFEST_NAME = 'split_manifest.json'
SUMMARY_NAME = 'voxel_dataset_summary.md'


@dataclass(frozen=True)
class F3LithologyVoxelDatasetInspection:
	"""Validated source geometry used by dry-runs and builds."""

	patch_size_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	volume_shape_xyz: tuple[int, int, int]
	label_volume: np.ndarray
	valid_tokens: np.ndarray
	metadata: Mapping[str, object]
	geometry: F3LineGeometry
	classes: tuple[F3ClassInfo, ...]
	records: tuple[F3SliceSplitRecord, ...]
	split: F3VoxelSupervisionSplit


@dataclass(frozen=True)
class F3LithologyVoxelDatasetResult:
	"""Committed output paths and primary voxel counts."""

	output_dir: Path
	split_grid: Path
	metadata_json: Path
	class_counts_csv: Path
	split_manifest_json: Path
	summary_markdown: Path
	train_voxel_count: int
	validation_voxel_count: int


def inspect_f3_lithology_voxel_dataset(
	config: F3LithologyVoxelDatasetConfig,
) -> F3LithologyVoxelDatasetInspection:
	"""Load all sources and validate metadata/array/geometry consistency."""
	for path in (
		config.source_label_volume,
		config.source_label_segy,
		config.png_label_inventory,
		config.class_info,
		config.segy_geometry_json,
		config.reference_metadata_json,
		config.reference_valid_tokens,
	):
		if not path.is_file():
			raise FileNotFoundError(f'missing voxel dataset input: {path}')
	with config.reference_metadata_json.open(encoding='utf-8') as file_obj:
		metadata = json.load(file_obj)
	if not isinstance(metadata, Mapping):
		raise TypeError('reference embedding metadata must contain an object')
	patch = _positive_triplet(metadata.get('patch_size'), 'patch_size')
	token_shape = _positive_triplet(
		metadata.get('token_grid_shape'), 'token_grid_shape'
	)
	volume_shape = _positive_triplet(
		metadata.get('volume_shape_xyz'), 'volume_shape_xyz'
	)
	labels = np.load(config.source_label_volume, mmap_mode='r')
	valid_tokens = np.load(config.reference_valid_tokens, mmap_mode='r')
	geometry = read_f3_line_geometry(config.segy_geometry_json)
	if labels.ndim != 3 or tuple(labels.shape) != volume_shape:
		raise ValueError(
			'embedding metadata volume_shape_xyz does not match label volume; '
			f'metadata={volume_shape!r}, array={labels.shape!r}'
		)
	if tuple(geometry.shape_xyz) != volume_shape:
		raise ValueError(
			'SEGY geometry shape does not match embedding volume_shape_xyz'
		)
	if valid_tokens.dtype != np.bool_:
		raise TypeError(
			f'reference valid_tokens dtype must be bool; got {valid_tokens.dtype}'
		)
	if tuple(valid_tokens.shape) != token_shape:
		raise ValueError(
			'embedding metadata token_grid_shape does not match valid-token array; '
			f'metadata={token_shape!r}, array={valid_tokens.shape!r}'
		)
	expected = tuple(
		(size + step - 1) // step
		for size, step in zip(volume_shape, patch, strict=True)
	)
	if expected != token_shape:
		raise ValueError(
			'embedding patch_size/token_grid_shape/volume_shape_xyz are inconsistent; '
			f'expected token grid {expected!r}'
		)
	classes = read_f3_lithology_class_info(config.class_info)
	records = load_f3_slice_split_records(config.png_label_inventory)
	split = build_f3_voxel_supervision_split(
		records,
		geometry=geometry,
		label_volume=labels,
		class_ids=tuple(item.class_id for item in classes),
		valid_tokens=valid_tokens,
		patch_size_xyz=patch,
		ignore_z_border_samples=config.ignore_z_border_samples,
	)
	return F3LithologyVoxelDatasetInspection(
		patch_size_xyz=patch,
		token_grid_shape_xyz=token_shape,
		volume_shape_xyz=volume_shape,
		label_volume=labels,
		valid_tokens=valid_tokens,
		metadata=cast('Mapping[str, object]', metadata),
		geometry=geometry,
		classes=classes,
		records=records,
		split=split,
	)


def build_f3_lithology_voxel_dataset(
	config: F3LithologyVoxelDatasetConfig,
) -> F3LithologyVoxelDatasetResult:
	"""Validate, stage, and atomically commit the voxel supervision artifact."""
	inspection = inspect_f3_lithology_voxel_dataset(config)
	if config.output_dir.exists() and not config.overwrite:
		raise FileExistsError(
			f'refusing to overwrite existing output: {config.output_dir}'
		)
	split = inspection.split
	config.output_dir.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix=f'.{config.output_dir.name}.staging-', dir=config.output_dir.parent
		)
	)
	try:
		_write_artifact(staging, config=config, inspection=inspection, split=split)
		_commit_directory(staging, config.output_dir, overwrite=config.overwrite)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	return F3LithologyVoxelDatasetResult(
		output_dir=config.output_dir,
		split_grid=config.output_dir / GRID_NAME,
		metadata_json=config.output_dir / METADATA_NAME,
		class_counts_csv=config.output_dir / COUNTS_NAME,
		split_manifest_json=config.output_dir / MANIFEST_NAME,
		summary_markdown=config.output_dir / SUMMARY_NAME,
		train_voxel_count=split.summary.final_train_voxels,
		validation_voxel_count=split.summary.final_validation_voxels,
	)


def _write_artifact(
	staging: Path,
	*,
	config: F3LithologyVoxelDatasetConfig,
	inspection: F3LithologyVoxelDatasetInspection,
	split: F3VoxelSupervisionSplit,
) -> None:
	grid_path = staging / GRID_NAME
	np.save(grid_path, split.split_grid, allow_pickle=False)
	rows = _class_count_rows(
		split.split_grid, inspection.label_volume, inspection.classes
	)
	with (staging / COUNTS_NAME).open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(
			file_obj,
			fieldnames=('split', 'class_id', 'class_name', 'count', 'fraction'),
		)
		writer.writeheader()
		writer.writerows(rows)
	manifest = f3_slice_split_manifest(inspection.records)
	manifest['validation_precedence'] = True
	manifest['precedence_contract'] = (
		'validation voxels replace overlapping train voxels'
	)
	_write_json(staging / MANIFEST_NAME, manifest)
	identities = {
		'reference_embedding': _identity(config.reference_metadata_json),
		'reference_valid_tokens': _identity(config.reference_valid_tokens),
		'label_volume': _identity(config.source_label_volume),
		'inventory': _identity(config.png_label_inventory),
	}
	metadata = {
		'artifact_type': 'f3_lithology_voxel_supervision',
		'schema_version': 1,
		'dataset': dict(config.dataset),
		'labels': {
			'source_label_segy': str(config.source_label_segy),
			'class_info': str(config.class_info),
		},
		'classes': [item.to_dict() for item in inspection.classes],
		'geometry': inspection.geometry.to_dict(),
		'split_codes': {'unsupervised': 0, 'train': 1, 'validation': 2},
		'split_strategy': 'png_label_inventory_slice_split_no_random_voxel_split',
		'no_random_split': True,
		'validation_precedence': True,
		'ignore_z_border_samples': config.ignore_z_border_samples,
		'reference_embedding': {
			**identities['reference_embedding'],
			'metadata': dict(inspection.metadata),
			'patch_size': list(inspection.patch_size_xyz),
			'token_grid_shape': list(inspection.token_grid_shape_xyz),
			'volume_shape_xyz': list(inspection.volume_shape_xyz),
		},
		'reference_valid_tokens': identities['reference_valid_tokens'],
		'label_volume': identities['label_volume'],
		'inventory': identities['inventory'],
		'summary': asdict(split.summary),
		'outputs': {
			'supervision_split_grid': str(config.output_dir / GRID_NAME),
			'metadata_json': str(config.output_dir / METADATA_NAME),
			'class_counts_csv': str(config.output_dir / COUNTS_NAME),
			'split_manifest_json': str(config.output_dir / MANIFEST_NAME),
			'summary_markdown': str(config.output_dir / SUMMARY_NAME),
		},
	}
	_write_json(staging / METADATA_NAME, metadata)
	(staging / SUMMARY_NAME).write_text(
		_summary_markdown(metadata, rows), encoding='utf-8'
	)


def _class_count_rows(
	grid: np.ndarray, labels: np.ndarray, classes: Sequence[F3ClassInfo]
) -> list[dict[str, object]]:
	rows = []
	for split_name, code in (
		('train', TRAIN_VOXEL_SPLIT),
		('validation', VALIDATION_VOXEL_SPLIT),
		('all_supervised', None),
	):
		mask = grid > 0 if code is None else grid == code
		counts = Counter(int(value) for value in np.asarray(labels[mask]))
		total = int(np.count_nonzero(mask))
		for item in classes:
			count = counts.get(item.class_id, 0)
			rows.append(
				{
					'split': split_name,
					'class_id': item.class_id,
					'class_name': item.class_name,
					'count': count,
					'fraction': 0.0 if total == 0 else count / total,
				}
			)
	return rows


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'embedding metadata {label} must be a positive integer triple')
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in value
	):
		raise ValueError(
			f'embedding metadata {label} must be a positive integer triple'
		)
	return cast('tuple[int, int, int]', tuple(value))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


def _summary_markdown(
	metadata: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> str:
	summary = cast('Mapping[str, object]', metadata['summary'])
	lines = [
		'# F3 voxel supervision dataset',
		'',
		f'- Train voxels: {summary["final_train_voxels"]}',
		f'- Validation voxels: {summary["final_validation_voxels"]}',
		'',
		'| split | class id | class | count |',
		'|---|---:|---|---:|',
	]
	lines.extend(
		f'| {row["split"]} | {row["class_id"]} | {row["class_name"]} | {row["count"]} |'
		for row in rows
	)
	return '\n'.join(lines) + '\n'


def _commit_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
	if not target.exists():
		staging.replace(target)
		return
	if not overwrite:
		raise FileExistsError(f'refusing to overwrite existing output: {target}')
	backup = target.with_name(f'.{target.name}.backup')
	if backup.exists():
		shutil.rmtree(backup)
	target.replace(backup)
	try:
		staging.replace(target)
	except BaseException:
		backup.replace(target)
		raise
	shutil.rmtree(backup)


__all__ = [
	'F3LithologyVoxelDatasetInspection',
	'F3LithologyVoxelDatasetResult',
	'build_f3_lithology_voxel_dataset',
	'inspect_f3_lithology_voxel_dataset',
]
