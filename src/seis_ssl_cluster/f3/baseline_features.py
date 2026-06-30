"""Build baseline feature token datasets for the F3 lithology probe."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
	from numpy.typing import NDArray

BASELINE_FEATURE_KINDS = frozenset({'z_only', 'amplitude_stats'})
AMPLITUDE_STATISTICS = (
	'mean',
	'std',
	'rms',
	'abs_mean',
	'min',
	'max',
	'p10',
	'p50',
	'p90',
)
_CLASS_COUNT_FIELDNAMES = (
	'split',
	'class_id',
	'class_name',
	'count',
	'fraction',
)


def _stat_rms(block: NDArray[np.float64]) -> float:
	return float(np.sqrt(np.mean(np.square(block))))


def _stat_abs_mean(block: NDArray[np.float64]) -> float:
	return float(np.mean(np.abs(block)))


def _stat_p10(block: NDArray[np.float64]) -> float:
	return float(np.percentile(block, 10))


def _stat_p50(block: NDArray[np.float64]) -> float:
	return float(np.percentile(block, 50))


def _stat_p90(block: NDArray[np.float64]) -> float:
	return float(np.percentile(block, 90))


_AMPLITUDE_STATISTIC_FUNCTIONS = {
	'mean': np.mean,
	'std': np.std,
	'rms': _stat_rms,
	'abs_mean': _stat_abs_mean,
	'min': np.min,
	'max': np.max,
	'p10': _stat_p10,
	'p50': _stat_p50,
	'p90': _stat_p90,
}


@dataclass(frozen=True)
class F3BaselineReferenceTokenDataset:
	"""Input token dataset whose split and labels are reused."""

	train_tokens: Path
	validation_tokens: Path
	metadata_json: Path
	split_manifest: Path | None = None
	root: Path | None = None


@dataclass(frozen=True)
class F3BaselineTokenDatasetOutputs:
	"""Output paths for a baseline token dataset."""

	output_dir: Path
	metadata_json: Path
	feature_summary_json: Path
	feature_summary_markdown: Path
	split_manifest_json: Path | None = None
	class_counts_csv: Path | None = None
	summary_markdown: Path | None = None

	@property
	def train_npz(self) -> Path:
		"""Return train split token dataset path."""
		return self.output_dir / 'train_tokens.npz'

	@property
	def validation_npz(self) -> Path:
		"""Return validation split token dataset path."""
		return self.output_dir / 'validation_tokens.npz'


@dataclass(frozen=True)
class F3BaselineFeatureConfig:
	"""Baseline feature generation settings."""

	kind: str
	statistics: tuple[str, ...] = AMPLITUDE_STATISTICS
	polynomial_degree: int = 1
	normalization: str = 'minmax'
	seismic_path: Path | None = None
	feature_space: str | None = None

	def __post_init__(self) -> None:
		"""Validate baseline feature settings."""
		if self.kind not in BASELINE_FEATURE_KINDS:
			msg = (
				f'baseline kind must be one of {sorted(BASELINE_FEATURE_KINDS)!r}; '
				f'got {self.kind!r}'
			)
			raise ValueError(msg)
		if (
			not isinstance(self.polynomial_degree, int)
			or isinstance(self.polynomial_degree, bool)
			or self.polynomial_degree <= 0
		):
			msg = (
				'baseline z_only polynomial_degree must be a positive integer; '
				f'got {self.polynomial_degree!r}'
			)
			raise ValueError(msg)
		unknown_statistics = sorted(set(self.statistics) - set(AMPLITUDE_STATISTICS))
		if unknown_statistics:
			msg = (
				f'unsupported amplitude statistic(s): {unknown_statistics!r}; '
				f'supported statistics are {list(AMPLITUDE_STATISTICS)!r}'
			)
			raise ValueError(msg)
		if self.kind == 'amplitude_stats' and self.seismic_path is None:
			msg = 'baseline amplitude_stats requires a seismic_path/source_volume'
			raise ValueError(msg)


@dataclass(frozen=True)
class F3LithologyBaselineTokenDatasetConfig:
	"""Complete F3 baseline token dataset build configuration."""

	reference: F3BaselineReferenceTokenDataset
	outputs: F3BaselineTokenDatasetOutputs
	features: F3BaselineFeatureConfig
	dataset: Mapping[str, object]
	model: Mapping[str, object]
	labels: Mapping[str, object]
	token_dataset: Mapping[str, object]
	feature_source: Mapping[str, object] | None = None


@dataclass(frozen=True)
class F3LithologyBaselineTokenDatasetResult:
	"""Paths and counts written by a baseline token dataset build."""

	train_npz: Path
	validation_npz: Path
	metadata_json: Path
	feature_summary_json: Path
	feature_summary_markdown: Path
	split_manifest_json: Path | None
	class_counts_csv: Path | None
	summary_markdown: Path | None
	train_token_count: int
	validation_token_count: int
	feature_dim: int


@dataclass(frozen=True)
class _TokenDataset:
	path: Path
	arrays: Mapping[str, NDArray[np.generic]]

	@property
	def labels(self) -> NDArray[np.int64]:
		return np.asarray(self.arrays['labels'], dtype=np.int64)

	@property
	def token_xyz(self) -> NDArray[np.int64]:
		return np.asarray(self.arrays['token_xyz'], dtype=np.int64)

	@property
	def count(self) -> int:
		return int(self.labels.shape[0])


@dataclass(frozen=True)
class _BuiltFeatures:
	features_by_split: Mapping[str, NDArray[np.float32]]
	feature_names: tuple[str, ...]
	feature_parameters: Mapping[str, object]


def build_f3_lithology_baseline_token_dataset(
	config: F3LithologyBaselineTokenDatasetConfig,
) -> F3LithologyBaselineTokenDatasetResult:
	"""Build a baseline F3 lithology token dataset from a reference dataset."""
	reference_metadata = _read_json(config.reference.metadata_json)
	train = _load_token_dataset(config.reference.train_tokens, label='train_tokens')
	validation = _load_token_dataset(
		config.reference.validation_tokens,
		label='validation_tokens',
	)
	_validate_reference_split(train, validation)
	features_by_split, feature_names, feature_parameters = _build_features(
		config.features,
		train=train,
		validation=validation,
		reference_metadata=reference_metadata,
	)
	built_features = _BuiltFeatures(
		features_by_split=features_by_split,
		feature_names=feature_names,
		feature_parameters=feature_parameters,
	)
	feature_dim = int(built_features.features_by_split['train'].shape[1])
	_write_outputs(
		config,
		train=train,
		validation=validation,
		built_features=built_features,
		reference_metadata=reference_metadata,
	)
	return F3LithologyBaselineTokenDatasetResult(
		train_npz=config.outputs.train_npz,
		validation_npz=config.outputs.validation_npz,
		metadata_json=config.outputs.metadata_json,
		feature_summary_json=config.outputs.feature_summary_json,
		feature_summary_markdown=config.outputs.feature_summary_markdown,
		split_manifest_json=config.outputs.split_manifest_json,
		class_counts_csv=config.outputs.class_counts_csv,
		summary_markdown=config.outputs.summary_markdown,
		train_token_count=train.count,
		validation_token_count=validation.count,
		feature_dim=feature_dim,
	)


def f3_lithology_baseline_token_dataset_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyBaselineTokenDatasetConfig:
	"""Validate and normalize a baseline token dataset config mapping."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'labels',
				'registry',
				'reference_token_dataset',
				'source_token_dataset',
				'lithology',
				'token_dataset',
				'baseline',
				'features',
			},
		),
		prefix='config',
	)
	paths = _optional_mapping(config, 'paths')
	artifact_root = _optional_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _optional_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _mapping_or_empty(config.get('dataset'))
	model = _mapping_or_empty(config.get('model'))
	labels = _mapping_or_empty(config.get('labels'))
	registry = _mapping_or_empty(config.get('registry'))
	token_dataset = _mapping_or_empty(config.get('token_dataset'))
	reference = _reference_from_mapping(config)
	outputs = _outputs_from_mapping(config)
	for label, path in _output_paths(outputs):
		_validate_artifact_output_path(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	features = _features_from_mapping(config, registry=registry)
	return F3LithologyBaselineTokenDatasetConfig(
		reference=reference,
		outputs=outputs,
		features=features,
		dataset=dataset,
		model=model,
		labels=labels,
		token_dataset=token_dataset,
		feature_source=_feature_source(token_dataset, features),
	)


def _build_features(
	features: F3BaselineFeatureConfig,
	*,
	train: _TokenDataset,
	validation: _TokenDataset,
	reference_metadata: Mapping[str, object],
) -> tuple[dict[str, NDArray[np.float32]], tuple[str, ...], dict[str, object]]:
	if features.kind == 'z_only':
		return _build_z_only_features(
			features,
			train=train,
			validation=validation,
			reference_metadata=reference_metadata,
		)
	return _build_amplitude_features(
		features,
		train=train,
		validation=validation,
		reference_metadata=reference_metadata,
	)


def _build_z_only_features(
	features: F3BaselineFeatureConfig,
	*,
	train: _TokenDataset,
	validation: _TokenDataset,
	reference_metadata: Mapping[str, object],
) -> tuple[dict[str, NDArray[np.float32]], tuple[str, ...], dict[str, object]]:
	train_z = _z_coordinates(train)
	validation_z = _z_coordinates(validation)
	if features.normalization == 'minmax':
		all_z = np.concatenate([train_z, validation_z])
		z_min = float(np.min(all_z))
		z_max = float(np.max(all_z))
	elif features.normalization == 'volume_z_extent':
		shape_xyz = _volume_shape_xyz(reference_metadata)
		z_min = 0.0
		z_max = float(shape_xyz[2] - 1)
	else:
		msg = (
			'z_only normalization must be "minmax" or "volume_z_extent"; '
			f'got {features.normalization!r}'
		)
		raise ValueError(msg)
	denominator = z_max - z_min
	if denominator <= 0.0:
		msg = f'z normalization range must be positive; got min={z_min}, max={z_max}'
		raise ValueError(msg)
	names = tuple(
		'z_norm' if degree == 1 else f'z_norm_power_{degree}'
		for degree in range(1, features.polynomial_degree + 1)
	)
	return (
		{
			'train': _z_polynomial_features(
				train_z,
				z_min=z_min,
				denominator=denominator,
				degree=features.polynomial_degree,
			),
			'validation': _z_polynomial_features(
				validation_z,
				z_min=z_min,
				denominator=denominator,
				degree=features.polynomial_degree,
			),
		},
		names,
		{
			'normalization': features.normalization,
			'z_min': z_min,
			'z_max': z_max,
			'polynomial_degree': features.polynomial_degree,
		},
	)


def _build_amplitude_features(
	features: F3BaselineFeatureConfig,
	*,
	train: _TokenDataset,
	validation: _TokenDataset,
	reference_metadata: Mapping[str, object],
) -> tuple[dict[str, NDArray[np.float32]], tuple[str, ...], dict[str, object]]:
	if features.seismic_path is None:
		msg = 'amplitude_stats requires seismic_path'
		raise ValueError(msg)
	seismic = np.load(features.seismic_path, mmap_mode='r')
	patch_size_xyz = _patch_size_xyz(reference_metadata)
	if seismic.ndim != 3:
		msg = f'amplitude seismic volume must be 3D; got shape={seismic.shape!r}'
		raise ValueError(msg)
	statistics = features.statistics
	return (
		{
			'train': _amplitude_statistics_for_tokens(
				seismic,
				train.token_xyz,
				patch_size_xyz=patch_size_xyz,
				statistics=statistics,
				label='train_tokens',
			),
			'validation': _amplitude_statistics_for_tokens(
				seismic,
				validation.token_xyz,
				patch_size_xyz=patch_size_xyz,
				statistics=statistics,
				label='validation_tokens',
			),
		},
		statistics,
		{
			'seismic_path': str(features.seismic_path),
			'patch_size_xyz': list(patch_size_xyz),
			'feature_space': features.feature_space,
			'statistics': list(statistics),
		},
	)


def _load_token_dataset(path: Path, *, label: str) -> _TokenDataset:
	if not path.is_file():
		msg = f'{label} does not exist: {path}'
		raise FileNotFoundError(msg)
	with np.load(path) as payload:
		arrays = {key: np.asarray(payload[key]) for key in payload.files}
	for key in ('features', 'labels', 'token_xyz'):
		if key not in arrays:
			msg = f'{label} must contain {key!r}: {path}'
			raise KeyError(msg)
	features = np.asarray(arrays['features'])
	labels = np.asarray(arrays['labels'])
	token_xyz = np.asarray(arrays['token_xyz'])
	if features.ndim != 2:
		msg = f'{label}.features must be 2D; got shape={features.shape!r}'
		raise ValueError(msg)
	if labels.ndim != 1:
		msg = f'{label}.labels must be 1D; got shape={labels.shape!r}'
		raise ValueError(msg)
	if token_xyz.shape != (labels.shape[0], 3):
		msg = (
			f'{label}.token_xyz must have shape {(labels.shape[0], 3)!r}; '
			f'got {token_xyz.shape!r}'
		)
		raise ValueError(msg)
	for key, array in arrays.items():
		if array.shape[:1] != labels.shape[:1]:
			msg = (
				f'{label}.{key} row count must match labels; '
				f'got {array.shape[:1]!r}, expected {labels.shape[:1]!r}'
			)
			raise ValueError(msg)
	return _TokenDataset(path=path, arrays=arrays)


def _validate_reference_split(train: _TokenDataset, validation: _TokenDataset) -> None:
	if train.count == 0:
		msg = 'train_tokens must contain at least one token'
		raise ValueError(msg)
	if validation.count == 0:
		msg = 'validation_tokens must contain at least one token'
		raise ValueError(msg)


def _z_coordinates(dataset: _TokenDataset) -> NDArray[np.float64]:
	if 'voxel_center_xyz' in dataset.arrays:
		centers = np.asarray(dataset.arrays['voxel_center_xyz'], dtype=np.float64)
		if centers.shape != (dataset.count, 3):
			msg = (
				f'{dataset.path}.voxel_center_xyz must have shape '
				f'{(dataset.count, 3)!r}; got {centers.shape!r}'
			)
			raise ValueError(msg)
		z_values = centers[:, 2]
	else:
		z_values = np.asarray(dataset.token_xyz[:, 2], dtype=np.float64)
	if not np.all(np.isfinite(z_values)):
		msg = f'{dataset.path} contains non-finite z coordinates'
		raise ValueError(msg)
	return z_values


def _z_polynomial_features(
	z_values: NDArray[np.float64],
	*,
	z_min: float,
	denominator: float,
	degree: int,
) -> NDArray[np.float32]:
	z_norm = (z_values - z_min) / denominator
	matrix = np.column_stack([z_norm**power for power in range(1, degree + 1)])
	return _finite_float32_features(matrix, label='z_only.features')


def _amplitude_statistics_for_tokens(
	seismic: NDArray[np.generic],
	token_xyz: NDArray[np.int64],
	*,
	patch_size_xyz: tuple[int, int, int],
	statistics: Sequence[str],
	label: str,
) -> NDArray[np.float32]:
	rows = np.empty((token_xyz.shape[0], len(statistics)), dtype=np.float32)
	volume_shape = tuple(int(axis) for axis in seismic.shape)
	for row_index, token in enumerate(token_xyz):
		block = _token_block(
			seismic,
			token,
			patch_size_xyz=patch_size_xyz,
			volume_shape=volume_shape,
			label=f'{label}.token_xyz[{row_index}]',
		)
		if not np.all(np.isfinite(block)):
			msg = f'{label} amplitude block contains NaN or Inf at row {row_index}'
			raise ValueError(msg)
		rows[row_index] = _amplitude_statistics(block, statistics)
	return _finite_float32_features(rows, label=f'{label}.features')


def _token_block(
	seismic: NDArray[np.generic],
	token: NDArray[np.int64],
	*,
	patch_size_xyz: tuple[int, int, int],
	volume_shape: tuple[int, int, int],
	label: str,
) -> NDArray[np.float64]:
	slices = []
	for axis, (token_index, patch_size, axis_size) in enumerate(
		zip(token, patch_size_xyz, volume_shape, strict=True),
	):
		if token_index < 0:
			msg = f'{label} axis {axis} must be nonnegative; got {int(token_index)}'
			raise ValueError(msg)
		start = int(token_index) * patch_size
		stop = min(start + patch_size, axis_size)
		if start >= stop:
			msg = (
				f'{label} axis {axis} block is outside seismic volume; '
				f'start={start}, stop={stop}, shape={volume_shape!r}'
			)
			raise ValueError(msg)
		slices.append(slice(start, stop))
	return np.asarray(seismic[tuple(slices)], dtype=np.float64)


def _amplitude_statistics(
	block: NDArray[np.float64],
	statistics: Sequence[str],
) -> NDArray[np.float32]:
	values = []
	for statistic in statistics:
		function = _AMPLITUDE_STATISTIC_FUNCTIONS.get(statistic)
		if function is None:
			msg = f'unsupported amplitude statistic: {statistic!r}'
			raise ValueError(msg)
		values.append(float(function(block)))
	return np.asarray(values, dtype=np.float32)


def _write_outputs(
	config: F3LithologyBaselineTokenDatasetConfig,
	*,
	train: _TokenDataset,
	validation: _TokenDataset,
	built_features: _BuiltFeatures,
	reference_metadata: Mapping[str, object],
) -> None:
	outputs = config.outputs
	outputs.output_dir.mkdir(parents=True, exist_ok=True)
	for path in (
		outputs.metadata_json,
		outputs.feature_summary_json,
		outputs.feature_summary_markdown,
		outputs.split_manifest_json,
		outputs.class_counts_csv,
		outputs.summary_markdown,
	):
		if path is not None:
			path.parent.mkdir(parents=True, exist_ok=True)
	_save_npz_with_features(
		outputs.train_npz,
		train,
		built_features.features_by_split['train'],
		label='train_tokens',
	)
	_save_npz_with_features(
		outputs.validation_npz,
		validation,
		built_features.features_by_split['validation'],
		label='validation_tokens',
	)
	if outputs.split_manifest_json is not None and config.reference.split_manifest:
		shutil.copyfile(config.reference.split_manifest, outputs.split_manifest_json)
	if outputs.class_counts_csv is not None:
		_write_class_counts_csv(
			outputs.class_counts_csv,
			classes=_classes_from_metadata(reference_metadata),
			train_labels=train.labels,
			validation_labels=validation.labels,
		)
	if outputs.summary_markdown is not None:
		_write_text(
			outputs.summary_markdown,
			_render_token_dataset_summary(
				config,
				train=train,
			validation=validation,
			feature_names=built_features.feature_names,
		),
	)
	feature_summary = _feature_summary_payload(
		config,
		train_features=built_features.features_by_split['train'],
		validation_features=built_features.features_by_split['validation'],
		feature_names=built_features.feature_names,
		feature_parameters=built_features.feature_parameters,
	)
	_write_json(outputs.feature_summary_json, feature_summary)
	_write_text(
		outputs.feature_summary_markdown,
		_render_feature_summary_markdown(feature_summary),
	)
	_write_json(
		outputs.metadata_json,
		_metadata_payload(
			config,
			train=train,
			validation=validation,
			reference_metadata=reference_metadata,
			built_features=built_features,
		),
	)


def _save_npz_with_features(
	path: Path,
	dataset: _TokenDataset,
	features: NDArray[np.float32],
	*,
	label: str,
) -> None:
	if features.shape[0] != dataset.count:
		msg = (
			f'{label} baseline features row count must match source labels; '
			f'features={features.shape[0]}, labels={dataset.count}'
		)
		raise ValueError(msg)
	arrays = dict(dataset.arrays)
	arrays['features'] = _finite_float32_features(features, label=f'{label}.features')
	np.savez_compressed(path, **arrays)


def _metadata_payload(
	config: F3LithologyBaselineTokenDatasetConfig,
	*,
	train: _TokenDataset,
	validation: _TokenDataset,
	reference_metadata: Mapping[str, object],
	built_features: _BuiltFeatures,
) -> dict[str, object]:
	reference_embedding = _mapping_or_none(reference_metadata.get('embedding')) or {}
	feature_dim = len(built_features.feature_names)
	embedding = dict(reference_embedding)
	embedding['embedding_dim'] = feature_dim
	embedding['baseline_feature_kind'] = config.features.kind
	payload: dict[str, object] = {
		'artifact_type': 'f3_lithology_token_dataset',
		'dataset': dict(config.dataset) or dict(
			_mapping_or_none(reference_metadata.get('dataset')) or {},
		),
		'model': dict(config.model),
		'label_source_of_truth': reference_metadata.get(
			'label_source_of_truth',
			'segy_label_volume',
		),
		'png_label_role': reference_metadata.get(
			'png_label_role',
			'train_validation_slice_selection_and_visual_qc',
		),
		'split_strategy': reference_metadata.get(
			'split_strategy',
			'reference_token_dataset_train_validation_split',
		),
		'no_random_split': bool(reference_metadata.get('no_random_split', True)),
		'feature_source': dict(config.feature_source or {}),
		'baseline': {
			'kind': config.features.kind,
			'feature_names': list(built_features.feature_names),
			'feature_dim': feature_dim,
			'parameters': dict(built_features.feature_parameters),
		},
		'reference_token_dataset': {
			'root': (
				None
				if config.reference.root is None
				else str(config.reference.root)
			),
			'train_tokens': str(config.reference.train_tokens),
			'validation_tokens': str(config.reference.validation_tokens),
			'metadata_json': str(config.reference.metadata_json),
			'split_manifest': (
				None
				if config.reference.split_manifest is None
				else str(config.reference.split_manifest)
			),
		},
		'labels': dict(config.labels),
		'embedding': embedding,
		'geometry': dict(_mapping_or_none(reference_metadata.get('geometry')) or {}),
		'tokenization': dict(
			_mapping_or_none(reference_metadata.get('tokenization')) or {},
		),
		'classes': list(reference_metadata.get('classes', [])),
		'outputs': {
			'train_tokens': str(config.outputs.train_npz),
			'validation_tokens': str(config.outputs.validation_npz),
			'metadata_json': str(config.outputs.metadata_json),
			'feature_summary_json': str(config.outputs.feature_summary_json),
			'feature_summary_markdown': str(
				config.outputs.feature_summary_markdown,
			),
			'split_manifest_json': (
				None
				if config.outputs.split_manifest_json is None
				else str(config.outputs.split_manifest_json)
			),
			'class_counts_csv': (
				None
				if config.outputs.class_counts_csv is None
				else str(config.outputs.class_counts_csv)
			),
			'summary_markdown': (
				None
				if config.outputs.summary_markdown is None
				else str(config.outputs.summary_markdown)
			),
		},
		'summary': {
			'train_tokens': train.count,
			'validation_tokens': validation.count,
			'feature_dim': feature_dim,
			'train_class_counts': _class_counts(train.labels),
			'validation_class_counts': _class_counts(validation.labels),
		},
	}
	if 'cross_split_token_overlap_resolution' in reference_metadata:
		payload['cross_split_token_overlap_resolution'] = reference_metadata[
			'cross_split_token_overlap_resolution'
		]
	return payload


def _feature_summary_payload(
	config: F3LithologyBaselineTokenDatasetConfig,
	*,
	train_features: NDArray[np.float32],
	validation_features: NDArray[np.float32],
	feature_names: Sequence[str],
	feature_parameters: Mapping[str, object],
) -> dict[str, object]:
	all_features = np.concatenate([train_features, validation_features], axis=0)
	return {
		'artifact_type': 'f3_lithology_baseline_feature_summary',
		'feature_source': dict(config.feature_source or {}),
		'baseline': {
			'kind': config.features.kind,
			'feature_names': list(feature_names),
			'feature_dim': len(feature_names),
			'parameters': dict(feature_parameters),
		},
		'splits': {
			'train': _matrix_summary(train_features, feature_names),
			'validation': _matrix_summary(validation_features, feature_names),
			'all': _matrix_summary(all_features, feature_names),
		},
	}


def _matrix_summary(
	features: NDArray[np.float32],
	feature_names: Sequence[str],
) -> dict[str, object]:
	return {
		'token_count': int(features.shape[0]),
		'feature_dim': int(features.shape[1]),
		'features': {
			name: {
				'min': float(np.min(features[:, index])),
				'max': float(np.max(features[:, index])),
				'mean': float(np.mean(features[:, index])),
				'std': float(np.std(features[:, index])),
			}
			for index, name in enumerate(feature_names)
		},
	}


def _render_feature_summary_markdown(payload: Mapping[str, object]) -> str:
	baseline = _mapping_or_none(payload.get('baseline')) or {}
	splits = _mapping_or_none(payload.get('splits')) or {}
	feature_names = ', '.join(str(v) for v in baseline.get('feature_names', []))
	lines = [
		'# F3 lithology baseline feature summary',
		'',
		f'- baseline kind: {baseline.get("kind")}',
		f'- feature dim: {baseline.get("feature_dim")}',
		f'- feature names: {feature_names}',
		'',
		'| split | tokens |',
		'| --- | ---: |',
	]
	for split in ('train', 'validation', 'all'):
		summary = _mapping_or_none(splits.get(split)) or {}
		lines.append(f'| {split} | {summary.get("token_count", 0)} |')
	lines.append('')
	return '\n'.join(lines)


def _render_token_dataset_summary(
	config: F3LithologyBaselineTokenDatasetConfig,
	*,
	train: _TokenDataset,
	validation: _TokenDataset,
	feature_names: Sequence[str],
) -> str:
	return '\n'.join(
		[
			'# F3 lithology baseline token dataset',
			'',
			f'- baseline kind: {config.features.kind}',
			f'- train tokens: {train.count}',
			f'- validation tokens: {validation.count}',
			f'- feature dim: {len(feature_names)}',
			f'- feature names: {", ".join(feature_names)}',
			'- split source: reference token dataset',
			'',
		],
	)


def _write_class_counts_csv(
	path: Path,
	*,
	classes: Sequence[Mapping[str, object]],
	train_labels: NDArray[np.int64],
	validation_labels: NDArray[np.int64],
) -> None:
	rows = []
	rows.extend(_class_count_rows('train', classes, train_labels))
	rows.extend(_class_count_rows('validation', classes, validation_labels))
	rows.extend(
		_class_count_rows(
			'all_labeled',
			classes,
			np.concatenate([train_labels, validation_labels]),
		),
	)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=_CLASS_COUNT_FIELDNAMES)
		writer.writeheader()
		writer.writerows(rows)


def _class_count_rows(
	split: str,
	classes: Sequence[Mapping[str, object]],
	labels: NDArray[np.int64],
) -> list[dict[str, object]]:
	counts = Counter(int(label) for label in labels)
	total = int(labels.shape[0])
	return [
		{
			'split': split,
			'class_id': int(class_info['class_id']),
			'class_name': str(class_info.get('class_name', '')),
			'count': int(counts.get(int(class_info['class_id']), 0)),
			'fraction': _fraction(
				int(counts.get(int(class_info['class_id']), 0)),
				total,
			),
		}
		for class_info in classes
	]


def _classes_from_metadata(
	metadata: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
	classes = metadata.get('classes')
	if isinstance(classes, Sequence) and not isinstance(classes, str | bytes):
		parsed = [
			dict(item)
			for item in classes
			if isinstance(item, Mapping) and 'class_id' in item
		]
		if parsed:
			return tuple(parsed)
	return ()


def _class_counts(labels: NDArray[np.int64]) -> dict[str, int]:
	return {str(label): int(count) for label, count in Counter(labels.tolist()).items()}


def _fraction(count: int, total: int) -> float:
	return 0.0 if total == 0 else float(count / total)


def _finite_float32_features(
	features: NDArray[np.generic],
	*,
	label: str,
) -> NDArray[np.float32]:
	matrix = np.asarray(features, dtype=np.float32)
	if matrix.ndim != 2:
		msg = f'{label} must be a 2D matrix; got shape={matrix.shape!r}'
		raise ValueError(msg)
	if not np.all(np.isfinite(matrix)):
		msg = f'{label} contains NaN or Inf'
		raise ValueError(msg)
	return matrix


def _patch_size_xyz(metadata: Mapping[str, object]) -> tuple[int, int, int]:
	embedding = _mapping_or_none(metadata.get('embedding')) or {}
	patch_size = embedding.get('patch_size_xyz') or embedding.get('patch_size')
	return _positive_xyz(patch_size, 'reference metadata embedding.patch_size_xyz')


def _volume_shape_xyz(metadata: Mapping[str, object]) -> tuple[int, int, int]:
	geometry = _mapping_or_none(metadata.get('geometry')) or {}
	return _positive_xyz(
		geometry.get('shape_xyz'),
		'reference metadata geometry.shape_xyz',
	)


def _positive_xyz(value: object, label: str) -> tuple[int, int, int]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a sequence of three positive integers; got {value!r}'
		raise TypeError(msg)
	if len(value) != 3:
		msg = f'{label} must contain three entries; got {value!r}'
		raise ValueError(msg)
	xyz = tuple(int(axis) for axis in value)
	if any(axis <= 0 for axis in xyz):
		msg = f'{label} entries must be positive; got {value!r}'
		raise ValueError(msg)
	return xyz


def _reference_from_mapping(
	config: Mapping[str, object],
) -> F3BaselineReferenceTokenDataset:
	if 'reference_token_dataset' in config:
		raw = _required_mapping(config, 'reference_token_dataset')
	elif 'source_token_dataset' in config:
		raw = _required_mapping(config, 'source_token_dataset')
	else:
		msg = 'config requires reference_token_dataset or source_token_dataset'
		raise KeyError(msg)
	root = _optional_absolute_path(raw, 'root', prefix='reference_token_dataset')
	directory = _optional_absolute_path(
		raw,
		'directory',
		prefix='source_token_dataset',
	)
	base = root or directory
	train_tokens = _optional_absolute_path(
		raw,
		'train_tokens',
		prefix='reference_token_dataset',
	)
	validation_tokens = _optional_absolute_path(
		raw,
		'validation_tokens',
		prefix='reference_token_dataset',
	)
	metadata_json = _optional_absolute_path(
		raw,
		'metadata_json',
		prefix='reference_token_dataset',
	)
	split_manifest = _optional_absolute_path(
		raw,
		'split_manifest',
		prefix='reference_token_dataset',
	)
	if base is not None:
		train_tokens = train_tokens or base / 'train_tokens.npz'
		validation_tokens = validation_tokens or base / 'validation_tokens.npz'
		metadata_json = metadata_json or base / 'token_dataset_metadata.json'
		split_manifest = split_manifest or base / 'splits.json'
	if train_tokens is None or validation_tokens is None or metadata_json is None:
		msg = (
			'reference token dataset requires train_tokens, validation_tokens, '
			'and metadata_json, or a root/directory containing the standard files'
		)
		raise KeyError(msg)
	return F3BaselineReferenceTokenDataset(
		train_tokens=train_tokens,
		validation_tokens=validation_tokens,
		metadata_json=metadata_json,
		split_manifest=split_manifest,
		root=base,
	)


def _outputs_from_mapping(
	config: Mapping[str, object],
) -> F3BaselineTokenDatasetOutputs:
	baseline = _mapping_or_empty(config.get('baseline'))
	token_dataset = _mapping_or_empty(config.get('token_dataset'))
	output_dir = _optional_absolute_path(
		baseline,
		'output_dir',
		prefix='baseline',
	) or _required_absolute_path(
		token_dataset,
		'output_dir',
		prefix='token_dataset',
	)
	metadata_json = _optional_absolute_path(
		token_dataset,
		'metadata_json',
		prefix='token_dataset',
	) or output_dir / 'token_dataset_metadata.json'
	return F3BaselineTokenDatasetOutputs(
		output_dir=output_dir,
		metadata_json=metadata_json,
		feature_summary_json=output_dir / 'feature_summary.json',
		feature_summary_markdown=output_dir / 'feature_summary.md',
		split_manifest_json=_optional_absolute_path(
			token_dataset,
			'split_manifest',
			prefix='token_dataset',
		),
		class_counts_csv=_optional_absolute_path(
			token_dataset,
			'class_counts_csv',
			prefix='token_dataset',
		),
		summary_markdown=_optional_absolute_path(
			token_dataset,
			'summary_markdown',
			prefix='token_dataset',
		),
	)


def _features_from_mapping(
	config: Mapping[str, object],
	*,
	registry: Mapping[str, object],
) -> F3BaselineFeatureConfig:
	baseline = _mapping_or_empty(config.get('baseline'))
	raw_features = _mapping_or_empty(config.get('features'))
	kind = _optional_str(baseline, 'kind') or _required_str(
		raw_features,
		'kind',
		prefix='features',
	)
	kind_options = _mapping_or_empty(baseline.get(kind))
	if kind == 'z_only':
		return F3BaselineFeatureConfig(
			kind=kind,
			polynomial_degree=_optional_positive_int(
				kind_options.get(
					'polynomial_degree',
					raw_features.get('polynomial_degree', 1),
				),
				'baseline.z_only.polynomial_degree',
			),
			normalization=str(
				kind_options.get(
					'normalize',
					kind_options.get(
						'normalization',
						raw_features.get(
							'normalize',
							raw_features.get('normalization', 'minmax'),
						),
					),
				),
			),
		)
	if kind == 'amplitude_stats':
		return F3BaselineFeatureConfig(
			kind=kind,
			statistics=_statistics_from_mapping(kind_options, raw_features),
			seismic_path=(
				_optional_absolute_path(
					kind_options,
					'seismic_path',
					prefix='baseline.amplitude_stats',
				)
				or _optional_absolute_path(
					raw_features,
					'source_volume',
					prefix='features',
				)
				or _optional_absolute_path(
					registry,
					'seismic_volume',
					prefix='registry',
				)
			),
			feature_space=_optional_str(kind_options, 'feature_space')
			or _optional_str(raw_features, 'feature_space'),
		)
	msg = (
		f'baseline kind must be one of {sorted(BASELINE_FEATURE_KINDS)!r}; '
		f'got {kind!r}'
	)
	raise ValueError(msg)


def _statistics_from_mapping(
	kind_options: Mapping[str, object],
	raw_features: Mapping[str, object],
) -> tuple[str, ...]:
	value = kind_options.get(
		'statistics',
		raw_features.get('statistics', AMPLITUDE_STATISTICS),
	)
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'amplitude statistics must be a sequence; got {value!r}'
		raise TypeError(msg)
	return tuple(str(item) for item in value)


def _feature_source(
	token_dataset: Mapping[str, object],
	features: F3BaselineFeatureConfig,
) -> Mapping[str, object]:
	value = token_dataset.get('feature_source')
	if isinstance(value, Mapping):
		return dict(value)
	return {
		'kind': features.kind,
		'reference_model_tag': None,
		'embedding_spec': None,
		'description': f'{features.kind} baseline features',
	}


def _output_paths(
	outputs: F3BaselineTokenDatasetOutputs,
) -> tuple[tuple[str, Path], ...]:
	paths = [
		('token_dataset.output_dir', outputs.output_dir),
		('token_dataset.metadata_json', outputs.metadata_json),
		('token_dataset.feature_summary_json', outputs.feature_summary_json),
		('token_dataset.feature_summary_markdown', outputs.feature_summary_markdown),
	]
	if outputs.split_manifest_json is not None:
		paths.append(('token_dataset.split_manifest', outputs.split_manifest_json))
	if outputs.class_counts_csv is not None:
		paths.append(('token_dataset.class_counts_csv', outputs.class_counts_csv))
	if outputs.summary_markdown is not None:
		paths.append(('token_dataset.summary_markdown', outputs.summary_markdown))
	return tuple(paths)


def _validate_artifact_output_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path | None,
	f3_root: Path | None,
) -> None:
	if 'runs' in path.parts:
		msg = f'{label} must not use runs/ paths; got {path}'
		raise ValueError(msg)
	if f3_root is not None and _is_relative_to(path, f3_root):
		msg = f'{label} must not be under paths.f3_root; got {path}'
		raise ValueError(msg)
	if artifact_root is not None and not _is_relative_to(path, artifact_root):
		msg = (
			f'{label} must be under paths.artifact_root '
			f'({artifact_root}); got {path}'
		)
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _read_json(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		msg = f'JSON file does not exist: {path}'
		raise FileNotFoundError(msg)
	with path.open(encoding='utf-8') as file_obj:
		value = json.load(file_obj)
	if not isinstance(value, Mapping):
		msg = f'JSON root must be a mapping: {path}'
		raise TypeError(msg)
	return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _write_text(path: Path, text: str) -> None:
	path.write_text(text, encoding='utf-8')


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'expected a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _mapping_or_none(value: object) -> Mapping[str, object] | None:
	return value if isinstance(value, Mapping) else None


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		msg = (
			f'{prefix} key(s) not allowed: {unexpected!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _required_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = Path(_required_str(parent, key, prefix=prefix))
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _optional_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _required_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_str(parent: Mapping[str, object], key: str) -> str | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value
