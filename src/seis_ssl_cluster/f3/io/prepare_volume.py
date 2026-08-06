"""Prepare F3 facies SEGY cubes as registry NPY volumes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import numpy as np

from seis_ssl_cluster.config.common import _validate_distinct_paths
from seis_ssl_cluster.config.schema import (
	F3_FACIES_DATASET_NAME,
	F3_FACIES_DATASET_VERSION,
)
from seis_ssl_cluster.data.normalization import (
	compute_normalization_stats,
	write_normalization_stats,
)
from seis_ssl_cluster.data.schema import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	write_manifest_json,
)
from seis_ssl_cluster.data.volume_store import inspect_npy_volume
from seis_ssl_cluster.f3.io.labels import F3ClassInfo, read_class_info
from seis_ssl_cluster.f3.io.segy import (
	axis_assumption_metadata,
	read_f3_segy_file,
)

F3_SEISMIC_NPY_NAME = 'f3_seismic.npy'
F3_LABEL_NPY_NAME = 'f3_facies_labels.npy'
F3_METADATA_NAME = 'f3_metadata.json'
F3_MANIFEST_NAME = 'f3_amplitude_manifest.json'
F3_SPLIT_NAME = 'f3_npy_paths.txt'
F3_NORMALIZATION_STATS_NAME = 'f3_seismic.normalization_stats.json'
F3_SURVEY_ID = F3_FACIES_DATASET_NAME


@dataclass(frozen=True)
class F3PrepareRootPaths:
	"""Root directories used by the F3 preparation stage."""

	f3_root: Path
	artifact_root: Path


@dataclass(frozen=True)
class F3PrepareInputPaths:
	"""Raw and inspection inputs for F3 volume preparation."""

	seismic_segy: Path
	label_segy: Path
	class_info: Path
	inspection_report: Path


@dataclass(frozen=True)
class F3PrepareOutputPaths:
	"""Registry outputs for prepared F3 facies volumes."""

	volume_dir: Path
	manifest_path: Path
	split_path: Path
	normalization_stats_path: Path
	metadata_path: Path

	@property
	def seismic_npy(self) -> Path:
		"""Prepared seismic amplitude NPY path."""
		return self.volume_dir / F3_SEISMIC_NPY_NAME

	@property
	def label_npy(self) -> Path:
		"""Prepared facies-label NPY path."""
		return self.volume_dir / F3_LABEL_NPY_NAME


@dataclass(frozen=True)
class F3PrepareDatasetConfig:
	"""Fixed F3 facies dataset identity."""

	name: str
	version: str
	survey_id: str


@dataclass(frozen=True)
class F3PrepareNormalizationConfig:
	"""Normalization-stat settings for the prepared F3 seismic volume."""

	clip_low_percentile: float
	clip_high_percentile: float
	eps: float
	max_samples: int | None
	seed: int


@dataclass(frozen=True)
class F3PrepareVolumeConfig:
	"""Complete F3 volume preparation configuration."""

	paths: F3PrepareRootPaths
	inputs: F3PrepareInputPaths
	outputs: F3PrepareOutputPaths
	dataset: F3PrepareDatasetConfig
	normalization: F3PrepareNormalizationConfig


@dataclass(frozen=True)
class F3PrepareVolumeResult:
	"""Paths and key metadata written by an F3 preparation run."""

	seismic_npy: Path
	label_npy: Path
	metadata_path: Path
	manifest_path: Path
	split_path: Path
	normalization_stats_path: Path
	shape_xyz: tuple[int, int, int]
	label_dtype: str


def prepare_f3_facies_volume(
	config: F3PrepareVolumeConfig,
	*,
	overwrite: bool = False,
) -> F3PrepareVolumeResult:
	"""Convert F3 SEGY inputs into NPY registry artifacts."""
	_validate_prepare_file_paths(inputs=config.inputs, outputs=config.outputs)
	_validate_source_files(config.inputs)
	_validate_writable_outputs(config.outputs, overwrite=overwrite)

	seismic = _load_seismic_cube(config.inputs.seismic_segy)
	labels = _load_label_cube(config.inputs.label_segy)
	if seismic.shape != labels.shape:
		msg = (
			'F3 seismic and label SEGY cube shapes must match; '
			f'got seismic={seismic.shape!r}, label={labels.shape!r}'
		)
		raise ValueError(msg)

	config.outputs.volume_dir.mkdir(parents=True, exist_ok=True)
	np.save(config.outputs.seismic_npy, seismic)
	np.save(config.outputs.label_npy, labels)

	seismic_info = inspect_npy_volume(config.outputs.seismic_npy)
	label_info = inspect_npy_volume(config.outputs.label_npy)
	manifest = _build_manifest(
		config,
		shape_xyz=seismic_info.shape_xyz,
		dtype=seismic_info.dtype,
	)
	config.outputs.manifest_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json([manifest], config.outputs.manifest_path)
	_write_split_path(config.outputs.split_path, config.outputs.seismic_npy)

	stats = compute_normalization_stats(
		config.outputs.seismic_npy,
		survey_id=config.dataset.survey_id,
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=config.normalization.clip_low_percentile,
		clip_high_percentile=config.normalization.clip_high_percentile,
		max_samples=config.normalization.max_samples,
		seed=config.normalization.seed,
		eps=config.normalization.eps,
	)
	write_normalization_stats(stats, config.outputs.normalization_stats_path)

	classes = read_class_info(config.inputs.class_info)
	inspection_report = _read_json_mapping(
		config.inputs.inspection_report,
		label='F3 inspection report',
	)
	_write_json(
		config.outputs.metadata_path,
		_prepare_metadata(
			config,
			seismic_shape=seismic_info.shape_xyz,
			seismic_dtype=seismic_info.dtype,
			label_shape=label_info.shape_xyz,
			label_dtype=label_info.dtype,
			labels=labels,
			classes=classes,
			inspection_report=inspection_report,
		),
	)

	return F3PrepareVolumeResult(
		seismic_npy=config.outputs.seismic_npy,
		label_npy=config.outputs.label_npy,
		metadata_path=config.outputs.metadata_path,
		manifest_path=config.outputs.manifest_path,
		split_path=config.outputs.split_path,
		normalization_stats_path=config.outputs.normalization_stats_path,
		shape_xyz=seismic_info.shape_xyz,
		label_dtype=label_info.dtype,
	)


def f3_prepare_volume_config_from_mapping(
	config: Mapping[str, object],
) -> F3PrepareVolumeConfig:
	"""Validate and normalize an F3 volume preparation config mapping."""
	_validate_allowed_keys(
		config,
		frozenset({'paths', 'inputs', 'outputs', 'dataset', 'normalization'}),
		prefix='config',
	)
	paths = _parse_paths(_required_mapping(config, 'paths'))
	dataset = _parse_dataset(_required_mapping(config, 'dataset'))
	inputs = _parse_inputs(_required_mapping(config, 'inputs'))
	outputs = _parse_outputs(
		_required_mapping(config, 'outputs'),
	)
	_validate_prepare_file_paths(inputs=inputs, outputs=outputs)
	_validate_prepare_output_locations(outputs=outputs, raw_root=paths.f3_root)
	normalization = _parse_normalization(_required_mapping(config, 'normalization'))
	return F3PrepareVolumeConfig(
		paths=paths,
		inputs=inputs,
		outputs=outputs,
		dataset=dataset,
		normalization=normalization,
	)


def _load_seismic_cube(path: Path) -> np.ndarray:
	inspection = read_f3_segy_file(path, role='seismic')
	array = np.asarray(inspection.cube)
	if not np.issubdtype(array.dtype, np.number):
		msg = f'F3 seismic SEGY cube must be numeric; got {array.dtype}: {path}'
		raise TypeError(msg)
	return array.astype(np.float32, copy=False)


def _load_label_cube(path: Path) -> np.ndarray:
	inspection = read_f3_segy_file(path, role='label')
	array = np.asarray(inspection.cube)
	if not np.issubdtype(array.dtype, np.number):
		msg = f'F3 label SEGY cube must be numeric; got {array.dtype}: {path}'
		raise TypeError(msg)
	if not np.isfinite(array).all():
		msg = f'F3 label SEGY cube contains non-finite values: {path}'
		raise ValueError(msg)
	if np.issubdtype(array.dtype, np.integer):
		label_min = int(np.min(array))
		label_max = int(np.max(array))
		integer_labels = array
	else:
		rounded = np.rint(array)
		if not np.array_equal(array, rounded):
			msg = f'F3 label SEGY cube must contain integer class ids: {path}'
			raise ValueError(msg)
		label_min = int(np.min(rounded))
		label_max = int(np.max(rounded))
		integer_labels = rounded
	if label_min < np.iinfo(np.int32).min or label_max > np.iinfo(np.int32).max:
		msg = (
			'F3 label SEGY class ids must fit int32; '
			f'got min={label_min}, max={label_max}: {path}'
		)
		raise ValueError(msg)
	dtype = (
		np.int16
		if label_min >= np.iinfo(np.int16).min
		and label_max <= np.iinfo(np.int16).max
		else np.int32
	)
	return integer_labels.astype(dtype, copy=False)


def _build_manifest(
	config: F3PrepareVolumeConfig,
	*,
	shape_xyz: tuple[int, int, int],
	dtype: str,
) -> SurveyManifest:
	record = AmplitudeVolumeRecord(
		survey_id=config.dataset.survey_id,
		path=config.outputs.seismic_npy,
		shape_xyz=shape_xyz,
		dtype=dtype,
		grid_order=GRID_ORDER_XYZ,
		normalization_stats_path=config.outputs.normalization_stats_path,
	)
	manifest = SurveyManifest(
		survey_id=config.dataset.survey_id,
		root=config.outputs.volume_dir,
		amplitude=record,
	)
	manifest.validate()
	return manifest


def _prepare_metadata(  # noqa: PLR0913
	config: F3PrepareVolumeConfig,
	*,
	seismic_shape: tuple[int, int, int],
	seismic_dtype: str,
	label_shape: tuple[int, int, int],
	label_dtype: str,
	labels: np.ndarray,
	classes: Sequence[F3ClassInfo],
	inspection_report: Mapping[str, object],
) -> dict[str, object]:
	unique_labels, counts = np.unique(labels, return_counts=True)
	label_counts = {
		str(int(value)): int(count)
		for value, count in zip(unique_labels, counts, strict=True)
	}
	label_zero_is_valid = 0 in {int(value) for value in unique_labels}
	return {
		'dataset': {
			'name': config.dataset.name,
			'version': config.dataset.version,
			'survey_id': config.dataset.survey_id,
		},
		'axis_assumption': axis_assumption_metadata(),
		'grid_order': list(GRID_ORDER_XYZ),
		'sources': {
			'f3_root': str(config.paths.f3_root),
			'seismic_segy': str(config.inputs.seismic_segy),
			'label_segy': str(config.inputs.label_segy),
			'class_info': str(config.inputs.class_info),
			'inspection_report': str(config.inputs.inspection_report),
		},
		'outputs': {
			'seismic_npy': str(config.outputs.seismic_npy),
			'label_npy': str(config.outputs.label_npy),
			'manifest': str(config.outputs.manifest_path),
			'split': str(config.outputs.split_path),
			'normalization_stats': str(config.outputs.normalization_stats_path),
			'metadata': str(config.outputs.metadata_path),
		},
		'volumes': {
			'seismic': {
				'shape_xyz': list(seismic_shape),
				'dtype': seismic_dtype,
				'grid_order': list(GRID_ORDER_XYZ),
			},
			'label': {
				'shape_xyz': list(label_shape),
				'dtype': label_dtype,
				'grid_order': list(GRID_ORDER_XYZ),
				'unique_values': [int(value) for value in unique_labels],
				'counts_by_value': label_counts,
				'label_zero_is_valid_class': label_zero_is_valid,
			},
		},
		'facies_classes': [item.to_dict() for item in classes],
		'inspection_report': {
			'path': str(config.inputs.inspection_report),
			'downstream_readiness': inspection_report.get('downstream_readiness'),
		},
	}


def _write_split_path(path: Path, seismic_npy: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(f'{seismic_npy}\n', encoding='utf-8')


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
	if not path.is_file():
		msg = f'{label} does not exist: {path}'
		raise FileNotFoundError(msg)
	with path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if not isinstance(payload, Mapping):
		msg = f'{label} must contain a JSON object: {path}'
		raise TypeError(msg)
	return cast('Mapping[str, object]', payload)


def _parse_paths(paths: Mapping[str, object]) -> F3PrepareRootPaths:
	_validate_allowed_keys(
		paths,
		frozenset({'f3_root', 'artifact_root'}),
		prefix='paths',
	)
	_validate_required_keys(
		paths,
		frozenset({'f3_root', 'artifact_root'}),
		prefix='paths',
	)
	return F3PrepareRootPaths(
		f3_root=_required_absolute_path(paths, 'f3_root', prefix='paths'),
		artifact_root=_required_absolute_path(paths, 'artifact_root', prefix='paths'),
	)


def _parse_inputs(inputs: Mapping[str, object]) -> F3PrepareInputPaths:
	_validate_allowed_keys(
		inputs,
		frozenset({'seismic_segy', 'label_segy', 'class_info', 'inspection_report'}),
		prefix='inputs',
	)
	_validate_required_keys(
		inputs,
		frozenset({'seismic_segy', 'label_segy', 'class_info', 'inspection_report'}),
		prefix='inputs',
	)
	seismic_segy = _required_absolute_path(inputs, 'seismic_segy', prefix='inputs')
	label_segy = _required_absolute_path(inputs, 'label_segy', prefix='inputs')
	class_info = _required_absolute_path(inputs, 'class_info', prefix='inputs')
	inspection_report = _required_absolute_path(
		inputs,
		'inspection_report',
		prefix='inputs',
	)
	return F3PrepareInputPaths(
		seismic_segy=seismic_segy,
		label_segy=label_segy,
		class_info=class_info,
		inspection_report=inspection_report,
	)


def _parse_outputs(
	outputs: Mapping[str, object],
) -> F3PrepareOutputPaths:
	_validate_allowed_keys(
		outputs,
		frozenset(
			{
				'volume_dir',
				'manifest_path',
				'split_path',
				'normalization_stats_path',
				'metadata_path',
			},
		),
		prefix='outputs',
	)
	_validate_required_keys(
		outputs,
		frozenset(
			{
				'volume_dir',
				'manifest_path',
				'split_path',
				'normalization_stats_path',
				'metadata_path',
			},
		),
		prefix='outputs',
	)
	return F3PrepareOutputPaths(
		volume_dir=_required_absolute_path(outputs, 'volume_dir', prefix='outputs'),
		manifest_path=_required_absolute_path(
			outputs,
			'manifest_path',
			prefix='outputs',
		),
		split_path=_required_absolute_path(outputs, 'split_path', prefix='outputs'),
		normalization_stats_path=_required_absolute_path(
			outputs,
			'normalization_stats_path',
			prefix='outputs',
		),
		metadata_path=_required_absolute_path(
			outputs,
			'metadata_path',
			prefix='outputs',
		),
	)


def _validate_prepare_file_paths(
	*,
	inputs: F3PrepareInputPaths,
	outputs: F3PrepareOutputPaths,
) -> None:
	source_files = (
		(inputs.seismic_segy, 'inputs.seismic_segy'),
		(inputs.label_segy, 'inputs.label_segy'),
		(inputs.class_info, 'inputs.class_info'),
		(inputs.inspection_report, 'inputs.inspection_report'),
	)
	output_files = (
		(outputs.seismic_npy, 'outputs.seismic_npy'),
		(outputs.label_npy, 'outputs.label_npy'),
		(outputs.metadata_path, 'outputs.metadata_path'),
		(outputs.manifest_path, 'outputs.manifest_path'),
		(outputs.split_path, 'outputs.split_path'),
		(
			outputs.normalization_stats_path,
			'outputs.normalization_stats_path',
		),
	)
	for output, output_label in output_files:
		for source, source_label in source_files:
			_validate_distinct_paths(output, output_label, source, source_label)
	for index, (left, left_label) in enumerate(output_files):
		for right, right_label in output_files[index + 1 :]:
			_validate_distinct_paths(left, left_label, right, right_label)


def _validate_prepare_output_locations(
	*,
	outputs: F3PrepareOutputPaths,
	raw_root: Path,
) -> None:
	for label, path in (
		('outputs.volume_dir', outputs.volume_dir),
		('outputs.manifest_path', outputs.manifest_path),
		('outputs.split_path', outputs.split_path),
		('outputs.normalization_stats_path', outputs.normalization_stats_path),
		('outputs.metadata_path', outputs.metadata_path),
	):
		_validate_output_not_in_raw_root(
			path,
			label,
			raw_root=raw_root,
		)


def _parse_dataset(dataset: Mapping[str, object]) -> F3PrepareDatasetConfig:
	_validate_allowed_keys(
		dataset,
		frozenset({'name', 'version', 'survey_id'}),
		prefix='dataset',
	)
	_validate_required_keys(
		dataset,
		frozenset({'name', 'version', 'survey_id'}),
		prefix='dataset',
	)
	name = _required_str(dataset, 'name', prefix='dataset')
	version = _required_str(dataset, 'version', prefix='dataset')
	survey_id = _required_str(dataset, 'survey_id', prefix='dataset')
	if name != F3_FACIES_DATASET_NAME:
		msg = f'dataset.name must be {F3_FACIES_DATASET_NAME!r}; got {name!r}'
		raise ValueError(msg)
	if version != F3_FACIES_DATASET_VERSION:
		msg = (
			f'dataset.version must be {F3_FACIES_DATASET_VERSION!r}; '
			f'got {version!r}'
		)
		raise ValueError(msg)
	if survey_id != F3_SURVEY_ID:
		msg = f'dataset.survey_id must be {F3_SURVEY_ID!r}; got {survey_id!r}'
		raise ValueError(msg)
	return F3PrepareDatasetConfig(
		name=name,
		version=version,
		survey_id=survey_id,
	)


def _parse_normalization(
	normalization: Mapping[str, object],
) -> F3PrepareNormalizationConfig:
	_validate_allowed_keys(
		normalization,
		frozenset({'clipping_percentiles', 'epsilon', 'max_samples', 'seed'}),
		prefix='normalization',
	)
	_validate_required_keys(
		normalization,
		frozenset({'clipping_percentiles', 'epsilon', 'max_samples', 'seed'}),
		prefix='normalization',
	)
	clip_low, clip_high = _required_percentiles(
		normalization,
		'clipping_percentiles',
		prefix='normalization',
	)
	return F3PrepareNormalizationConfig(
		clip_low_percentile=clip_low,
		clip_high_percentile=clip_high,
		eps=_required_positive_real(normalization, 'epsilon', prefix='normalization'),
		max_samples=_optional_positive_int_or_none(
			normalization,
			'max_samples',
			prefix='normalization',
		),
		seed=_required_int(normalization, 'seed', prefix='normalization'),
	)


def _validate_source_files(inputs: F3PrepareInputPaths) -> None:
	for label, path in (
		('F3 seismic SEGY file', inputs.seismic_segy),
		('F3 label SEGY file', inputs.label_segy),
		('F3 class_info JSON', inputs.class_info),
		('F3 inspection report JSON', inputs.inspection_report),
	):
		if not path.is_file():
			msg = f'{label} does not exist: {path}'
			raise FileNotFoundError(msg)


def _validate_writable_outputs(
	outputs: F3PrepareOutputPaths,
	*,
	overwrite: bool,
) -> None:
	if overwrite:
		return
	for path in (
		outputs.seismic_npy,
		outputs.label_npy,
		outputs.metadata_path,
		outputs.manifest_path,
		outputs.split_path,
		outputs.normalization_stats_path,
	):
		if path.exists():
			msg = f'output already exists; pass overwrite=True to replace: {path}'
			raise FileExistsError(msg)


def _validate_output_not_in_raw_root(
	path: Path,
	label: str,
	*,
	raw_root: Path,
) -> None:
	if _is_relative_to(path, raw_root):
		msg = f'{label} must not be under paths.f3_root; got {path}'
		raise ValueError(msg)


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


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


def _validate_required_keys(
	parent: Mapping[str, object],
	keys: frozenset[str],
	*,
	prefix: str,
) -> None:
	for key in sorted(keys):
		if key not in parent:
			msg = f'{prefix}.{key} is required'
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


def _required_str(parent: Mapping[str, object], key: str, *, prefix: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _required_percentiles(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[float, float]:
	value = parent.get(key)
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 2
		or any(not _is_real(item) for item in value)
	):
		msg = f'{prefix}.{key} must contain two numbers; got {value!r}'
		raise TypeError(msg)
	low, high = (float(value[0]), float(value[1]))
	if not 0.0 <= low < high <= 100.0:
		msg = f'{prefix}.{key} must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)
	return low, high


def _required_positive_real(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> float:
	value = parent.get(key)
	if not _is_real(value):
		msg = f'{prefix}.{key} must be numeric; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not np.isfinite(number) or number <= 0.0:
		msg = f'{prefix}.{key} must be finite and positive; got {value!r}'
		raise ValueError(msg)
	return number


def _optional_positive_int_or_none(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> int | None:
	value = parent.get(key)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{prefix}.{key} must be an integer or null; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{prefix}.{key} must be positive when provided; got {value!r}'
		raise ValueError(msg)
	return integer


def _required_int(parent: Mapping[str, object], key: str, *, prefix: str) -> int:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{prefix}.{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return int(value)


def _is_real(value: object) -> bool:
	return isinstance(value, Real) and not isinstance(value, bool)




def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


__all__ = [
	'F3_LABEL_NPY_NAME',
	'F3_MANIFEST_NAME',
	'F3_METADATA_NAME',
	'F3_NORMALIZATION_STATS_NAME',
	'F3_SEISMIC_NPY_NAME',
	'F3_SPLIT_NAME',
	'F3_SURVEY_ID',
	'F3PrepareDatasetConfig',
	'F3PrepareInputPaths',
	'F3PrepareNormalizationConfig',
	'F3PrepareOutputPaths',
	'F3PrepareRootPaths',
	'F3PrepareVolumeConfig',
	'F3PrepareVolumeResult',
	'f3_prepare_volume_config_from_mapping',
	'prepare_f3_facies_volume',
]
