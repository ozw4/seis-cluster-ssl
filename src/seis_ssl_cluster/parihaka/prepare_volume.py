"""Prepare the Parihaka amplitude NPZ as a bounded-memory XYZ NPY."""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

import numpy as np

from seis_ssl_cluster.config.common import (
	_validate_disjoint_directories,
	_validate_distinct_paths,
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

PARIHAKA_AMPLITUDE_NPY_NAME = 'parihaka_amplitude.npy'
PARIHAKA_MANIFEST_NAME = 'parihaka_amplitude_manifest.json'
PARIHAKA_PATH_LIST_NAME = 'parihaka_npy_paths.txt'
PARIHAKA_NORMALIZATION_STATS_NAME = 'parihaka_amplitude.normalization_stats.json'
PARIHAKA_METADATA_NAME = 'parihaka_prepare_metadata.json'
_STAGING_PREFIX = '.parihaka_prepare_staging-'
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_STAT_ABS_TOL = 1.0e-6
_WINDOWS_ABSOLUTE_PATH = re.compile(r'^[A-Za-z]:')

_FIXED_DATASET = {
	'name': 'parihaka',
	'version': 'facies_benchmark_v1',
	'survey_id': 'parihaka',
}
_FIXED_SOURCE = {
	'direct_distributor': 'Mendeley Data',
	'dataset_title': (
		'Parihaka + Netherlands F3 (raw volumes + labels) for seismic facies '
		'segmentation'
	),
	'contributor': 'jiang zishuo',
	'version': 1,
	'doi': '10.17632/gnvyh3msrj.1',
	'upstream': 'AIcrowd Seismic Facies Identification Challenge',
	'displayed_license': 'CC BY 4.0',
	'archive_name': 'parihaka_Data.zip',
	'upstream_member': 'data_train.npz',
	'local_filename': 'parihaka_data_train.npz',
	'local_modification': 'filename only',
	'aicrowd_byte_identity': 'unverified',
	'redistribution_transformation': 'unverified',
	'array_key': 'data',
	'member_name': 'data.npy',
	'shape_zxy': [1006, 782, 590],
	'dtype': 'float32',
	'fortran_order': True,
	'element_count': 464_148_280,
	'finite_count': 464_148_280,
	'nonfinite_count': 0,
	'min': -5195.5234375,
	'max': 5151.71875,
	'mean': 0.6766075433795379,
	'population_std': 390.30892519280377,
}
_FIXED_CONVERSION = {'transpose_axes': [1, 2, 0]}
_FIXED_NORMALIZATION = {
	'clipping_percentiles': [0.5, 99.5],
	'epsilon': 1.0e-6,
	'max_samples': 1_000_000,
	'seed': 42,
}


@dataclass(frozen=True)
class ArrayStatistics:
	"""Full-volume scalar statistics."""

	element_count: int
	finite_count: int
	nonfinite_count: int
	minimum: float
	maximum: float
	mean: float
	population_std: float

	def to_dict(self) -> dict[str, int | float]:
		"""Return JSON-compatible statistics."""
		return {
			'element_count': self.element_count,
			'finite_count': self.finite_count,
			'nonfinite_count': self.nonfinite_count,
			'min': self.minimum,
			'max': self.maximum,
			'mean': self.mean,
			'population_std': self.population_std,
		}


@dataclass(frozen=True)
class ParihakaPrepareRootPaths:
	"""Raw-data and artifact roots."""

	artifact_root: Path
	parihaka_root: Path


@dataclass(frozen=True)
class ParihakaPrepareInputPaths:
	"""Amplitude-only source inputs."""

	amplitude_npz: Path


@dataclass(frozen=True)
class ParihakaPrepareOutputPaths:
	"""Explicit direct preparation outputs."""

	data_dir: Path
	amplitude_npy: Path
	manifest: Path
	path_list: Path
	normalization_stats: Path
	metadata: Path

	def files(self) -> tuple[Path, ...]:
		"""Return every final output, with completion metadata last."""
		return (
			self.amplitude_npy,
			self.manifest,
			self.path_list,
			self.normalization_stats,
			self.metadata,
		)


@dataclass(frozen=True)
class ParihakaPrepareDatasetConfig:
	"""Prepared dataset identity."""

	name: str
	version: str
	survey_id: str


@dataclass(frozen=True)
class ParihakaPrepareSourceConfig:
	"""Provenance and exact source-array contract."""

	direct_distributor: str
	dataset_title: str
	contributor: str
	version: int
	doi: str
	upstream: str
	displayed_license: str
	archive_name: str
	upstream_member: str
	local_filename: str
	local_modification: str
	aicrowd_byte_identity: str
	redistribution_transformation: str
	acquisition_date: str | None
	acquisition_by: str | None
	array_key: str
	member_name: str
	shape_zxy: tuple[int, int, int]
	dtype: str
	fortran_order: bool
	expected_statistics: ArrayStatistics


@dataclass(frozen=True)
class ParihakaPrepareConversionConfig:
	"""Bounded-memory transpose settings."""

	transpose_axes: tuple[int, int, int]
	chunk_size_z: int


@dataclass(frozen=True)
class ParihakaPrepareNormalizationConfig:
	"""Existing normalization-stat settings."""

	clip_low_percentile: float
	clip_high_percentile: float
	eps: float
	max_samples: int
	seed: int


@dataclass(frozen=True)
class ParihakaPrepareVolumeConfig:
	"""Complete Parihaka preparation configuration."""

	paths: ParihakaPrepareRootPaths
	inputs: ParihakaPrepareInputPaths
	outputs: ParihakaPrepareOutputPaths
	dataset: ParihakaPrepareDatasetConfig
	source: ParihakaPrepareSourceConfig
	conversion: ParihakaPrepareConversionConfig
	normalization: ParihakaPrepareNormalizationConfig


@dataclass(frozen=True)
class ParihakaSourceInspection:
	"""Validated ZIP inventory and embedded NPY header."""

	member_name: str
	shape_zxy: tuple[int, int, int]
	dtype: str
	fortran_order: bool


@dataclass(frozen=True)
class ParihakaPrepareVolumeResult:
	"""Published output identities from one completed preparation."""

	amplitude_npy: Path
	manifest: Path
	path_list: Path
	normalization_stats: Path
	metadata: Path
	shape_xyz: tuple[int, int, int]
	dtype: str
	order: str
	source_sha256: str
	output_sha256: str
	source_statistics: ArrayStatistics
	output_statistics: ArrayStatistics


def inspect_parihaka_preparation(
	config: ParihakaPrepareVolumeConfig,
	*,
	overwrite: bool = False,
) -> ParihakaSourceInspection:
	"""Perform the complete read-only validation used by ``--dry-run``."""
	_validate_runtime_config(config)
	_validate_source_path(config)
	_validate_existing_outputs(config.outputs, overwrite=overwrite)
	return _inspect_source_archive(config)


def prepare_parihaka_volume(
	config: ParihakaPrepareVolumeConfig,
	*,
	overwrite: bool = False,
) -> ParihakaPrepareVolumeResult:
	"""Prepare and publish the amplitude-only Parihaka XYZ volume."""
	inspection = inspect_parihaka_preparation(config, overwrite=overwrite)
	config.outputs.data_dir.mkdir(parents=True, exist_ok=True)
	staging_dir = Path(
		tempfile.mkdtemp(
			prefix=_STAGING_PREFIX,
			dir=config.outputs.data_dir,
		),
	)
	try:
		return _prepare_in_staging(
			config,
			inspection=inspection,
			staging_dir=staging_dir,
			overwrite=overwrite,
		)
	finally:
		shutil.rmtree(staging_dir, ignore_errors=True)


def _prepare_in_staging(
	config: ParihakaPrepareVolumeConfig,
	*,
	inspection: ParihakaSourceInspection,
	staging_dir: Path,
	overwrite: bool,
) -> ParihakaPrepareVolumeResult:
	extracted_npy = staging_dir / 'source_data.npy'
	staged = _staged_outputs(config.outputs, staging_dir)
	source_sha256 = _file_sha256(config.inputs.amplitude_npz)
	source_size = config.inputs.amplitude_npz.stat().st_size
	_extract_member(
		config.inputs.amplitude_npz,
		member_name=inspection.member_name,
		destination=extracted_npy,
	)
	extracted_header = _inspect_npy_file_header(extracted_npy)
	_validate_header(config.source, extracted_header, label='extracted NPY')

	source = np.load(extracted_npy, mmap_mode='r', allow_pickle=False)
	destination_shape = _destination_shape(config.source.shape_zxy)
	destination = np.lib.format.open_memmap(
		staged.amplitude_npy,
		mode='w+',
		dtype=np.float32,
		shape=destination_shape,
		fortran_order=False,
	)
	source_accumulator = _StatisticsAccumulator()
	for z0, z1 in _z_chunks(config.source.shape_zxy[0], config.conversion.chunk_size_z):
		source_chunk = source[z0:z1, :, :]
		source_accumulator.update(source_chunk)
		destination[:, :, z0:z1] = source_chunk.transpose(
			config.conversion.transpose_axes,
		)
	destination.flush()
	del destination
	source_statistics = source_accumulator.finalize()
	_validate_statistics(
		source_statistics,
		config.source.expected_statistics,
		label='source amplitude',
	)

	output_statistics = _verify_destination(
		extracted_npy,
		staged.amplitude_npy,
		config=config,
	)
	_validate_statistics(
		output_statistics,
		config.source.expected_statistics,
		label='output amplitude',
	)
	output_sha256 = _file_sha256(staged.amplitude_npy)
	output_size = staged.amplitude_npy.stat().st_size

	_write_companion_outputs(config, staged)
	companion_hashes = {
		'manifest': _file_sha256(staged.manifest),
		'path_list': _file_sha256(staged.path_list),
		'normalization_stats': _file_sha256(staged.normalization_stats),
	}
	_write_json(
		staged.metadata,
		_build_metadata(
			config,
			source_sha256=source_sha256,
			source_size=source_size,
			source_statistics=source_statistics,
			output_sha256=output_sha256,
			output_size=output_size,
			output_statistics=output_statistics,
			companion_hashes=companion_hashes,
		),
	)
	_publish_staged_outputs(staged, config.outputs, overwrite=overwrite)
	return ParihakaPrepareVolumeResult(
		amplitude_npy=config.outputs.amplitude_npy,
		manifest=config.outputs.manifest,
		path_list=config.outputs.path_list,
		normalization_stats=config.outputs.normalization_stats,
		metadata=config.outputs.metadata,
		shape_xyz=destination_shape,
		dtype='float32',
		order='C',
		source_sha256=source_sha256,
		output_sha256=output_sha256,
		source_statistics=source_statistics,
		output_statistics=output_statistics,
	)


def parihaka_prepare_volume_config_from_mapping(
	config: Mapping[str, object],
) -> ParihakaPrepareVolumeConfig:
	"""Parse a closed, fixed Parihaka preparation mapping."""
	_validate_keys(
		config,
		{
			'paths',
			'inputs',
			'outputs',
			'dataset',
			'source',
			'conversion',
			'normalization',
		},
		prefix='config',
	)
	paths = _parse_paths(_mapping(config, 'paths'))
	inputs = _parse_inputs(_mapping(config, 'inputs'))
	outputs = _parse_outputs(_mapping(config, 'outputs'))
	dataset = _parse_dataset(_mapping(config, 'dataset'))
	source = _parse_source(_mapping(config, 'source'))
	conversion = _parse_conversion(_mapping(config, 'conversion'), source.shape_zxy[0])
	normalization = _parse_normalization(_mapping(config, 'normalization'))
	resolved = ParihakaPrepareVolumeConfig(
		paths=paths,
		inputs=inputs,
		outputs=outputs,
		dataset=dataset,
		source=source,
		conversion=conversion,
		normalization=normalization,
	)
	_validate_fixed_mapping(config)
	_validate_runtime_config(resolved)
	return resolved


@dataclass(frozen=True)
class _NpyHeader:
	shape: tuple[int, ...]
	dtype: np.dtype[object]
	fortran_order: bool


class _StatisticsAccumulator:
	def __init__(self) -> None:
		self.element_count = 0
		self.finite_count = 0
		self.minimum = math.inf
		self.maximum = -math.inf
		self.total = 0.0
		self.total_squares = 0.0

	def update(self, values: np.ndarray) -> None:
		array = np.asarray(values)
		self.element_count += int(array.size)
		finite_mask = np.isfinite(array)
		finite_count = int(np.count_nonzero(finite_mask))
		self.finite_count += finite_count
		if finite_count == 0:
			return
		finite_values = (
			array if finite_count == array.size else np.asarray(array[finite_mask])
		)
		self.minimum = min(self.minimum, float(np.min(finite_values)))
		self.maximum = max(self.maximum, float(np.max(finite_values)))
		self.total += float(np.sum(finite_values, dtype=np.float64))
		self.total_squares += float(
			np.sum(np.square(finite_values, dtype=np.float64), dtype=np.float64),
		)

	def finalize(self) -> ArrayStatistics:
		if self.finite_count == 0:
			minimum = maximum = mean = population_std = math.nan
		else:
			minimum = self.minimum
			maximum = self.maximum
			mean = self.total / self.finite_count
			variance = self.total_squares / self.finite_count - mean * mean
			population_std = math.sqrt(max(0.0, variance))
		return ArrayStatistics(
			element_count=self.element_count,
			finite_count=self.finite_count,
			nonfinite_count=self.element_count - self.finite_count,
			minimum=minimum,
			maximum=maximum,
			mean=mean,
			population_std=population_std,
		)


def _inspect_source_archive(
	config: ParihakaPrepareVolumeConfig,
) -> ParihakaSourceInspection:
	try:
		with zipfile.ZipFile(config.inputs.amplitude_npz) as archive:
			info = _validate_zip_inventory(archive, config.source)
			with archive.open(info, 'r') as member:
				header = _read_npy_header(member)
	except (OSError, EOFError, zipfile.BadZipFile) as exc:
		msg = f'invalid or truncated source NPZ: {config.inputs.amplitude_npz}: {exc}'
		raise ValueError(msg) from exc
	_validate_header(config.source, header, label='source NPZ member')
	return ParihakaSourceInspection(
		member_name=config.source.member_name,
		shape_zxy=cast('tuple[int, int, int]', header.shape),
		dtype=str(header.dtype),
		fortran_order=header.fortran_order,
	)


def _validate_zip_inventory(
	archive: zipfile.ZipFile,
	source: ParihakaPrepareSourceConfig,
) -> zipfile.ZipInfo:
	infos = archive.infolist()
	names = [info.filename for info in infos]
	duplicates = sorted({name for name in names if names.count(name) > 1})
	if duplicates:
		msg = f'source NPZ contains duplicate ZIP member(s): {duplicates!r}'
		raise ValueError(msg)
	unsafe = [name for name in names if _unsafe_zip_member(name)]
	if unsafe:
		msg = f'source NPZ contains unsafe ZIP member path(s): {unsafe!r}'
		raise ValueError(msg)
	expected = [source.member_name]
	if names != expected:
		msg = (
			f'source NPZ keys must be exactly [{source.array_key}]; '
			f'ZIP members={names!r}'
		)
		raise ValueError(msg)
	return infos[0]


def _unsafe_zip_member(name: str) -> bool:
	path = PurePosixPath(name)
	return (
		not name
		or name.endswith('/')
		or name.startswith('/')
		or '\\' in name
		or _WINDOWS_ABSOLUTE_PATH.match(name) is not None
		or '..' in path.parts
	)


def _read_npy_header(file_obj: BinaryIO) -> _NpyHeader:
	version = np.lib.format.read_magic(file_obj)
	if version == (1, 0):
		shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file_obj)
	elif version in {(2, 0), (3, 0)}:
		shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file_obj)
	else:
		msg = f'unsupported NPY format version: {version!r}'
		raise ValueError(msg)
	return _NpyHeader(
		shape=tuple(int(axis) for axis in shape),
		dtype=np.dtype(dtype),
		fortran_order=bool(fortran_order),
	)


def _inspect_npy_file_header(path: Path) -> _NpyHeader:
	with path.open('rb') as file_obj:
		return _read_npy_header(file_obj)


def _validate_header(
	source: ParihakaPrepareSourceConfig,
	header: _NpyHeader,
	*,
	label: str,
) -> None:
	if header.dtype.hasobject:
		msg = f'{label} object dtype is forbidden'
		raise TypeError(msg)
	if header.shape != source.shape_zxy:
		msg = f'{label} shape must be {source.shape_zxy!r}; got {header.shape!r}'
		raise ValueError(msg)
	if header.dtype != np.dtype(source.dtype):
		msg = f'{label} dtype must be {source.dtype}; got {header.dtype}'
		raise TypeError(msg)
	if header.fortran_order is not source.fortran_order:
		msg = (
			f'{label} fortran_order must be {source.fortran_order}; '
			f'got {header.fortran_order}'
		)
		raise ValueError(msg)


def _extract_member(source_npz: Path, *, member_name: str, destination: Path) -> None:
	try:
		with (
			zipfile.ZipFile(source_npz) as archive,
			archive.open(member_name, 'r') as source,
			destination.open('xb') as output,
		):
			shutil.copyfileobj(source, output, length=_HASH_CHUNK_BYTES)
	except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
		msg = f'failed CRC-checked extraction from source NPZ {source_npz}: {exc}'
		raise ValueError(msg) from exc


def _verify_destination(
	extracted_npy: Path,
	destination_npy: Path,
	*,
	config: ParihakaPrepareVolumeConfig,
) -> ArrayStatistics:
	info = inspect_npy_volume(destination_npy)
	expected_shape = _destination_shape(config.source.shape_zxy)
	if info.shape_xyz != expected_shape or info.dtype != config.source.dtype:
		msg = (
			'prepared NPY inspection mismatch: '
			f'shape={info.shape_xyz!r}, dtype={info.dtype!r}'
		)
		raise ValueError(msg)
	header = _inspect_npy_file_header(destination_npy)
	if header.fortran_order or header.shape != expected_shape:
		msg = 'prepared NPY must have a C-order XYZ header'
		raise ValueError(msg)
	source = np.load(extracted_npy, mmap_mode='r', allow_pickle=False)
	destination = np.load(destination_npy, mmap_mode='r', allow_pickle=False)
	if not isinstance(destination, np.memmap) or not destination.flags.c_contiguous:
		msg = f'prepared NPY is not a C-contiguous readable memmap: {destination_npy}'
		raise ValueError(msg)
	accumulator = _StatisticsAccumulator()
	for z0, z1 in _z_chunks(config.source.shape_zxy[0], config.conversion.chunk_size_z):
		expected = source[z0:z1, :, :].transpose(config.conversion.transpose_axes)
		actual = destination[:, :, z0:z1]
		if not np.array_equal(expected.view(np.uint32), actual.view(np.uint32)):
			msg = f'full chunkwise bitwise verification failed for z=[{z0}, {z1})'
			raise ValueError(msg)
		accumulator.update(actual)
	return accumulator.finalize()


def _write_companion_outputs(
	config: ParihakaPrepareVolumeConfig,
	staged: ParihakaPrepareOutputPaths,
) -> None:
	manifest = SurveyManifest(
		survey_id=config.dataset.survey_id,
		root=config.outputs.data_dir,
		amplitude=AmplitudeVolumeRecord(
			survey_id=config.dataset.survey_id,
			path=config.outputs.amplitude_npy,
			shape_xyz=_destination_shape(config.source.shape_zxy),
			dtype=config.source.dtype,
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=config.outputs.normalization_stats,
		),
	)
	write_manifest_json([manifest], staged.manifest)
	staged.path_list.write_text(f'{config.outputs.amplitude_npy}\n', encoding='utf-8')
	stats = compute_normalization_stats(
		staged.amplitude_npy,
		survey_id=config.dataset.survey_id,
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=config.normalization.clip_low_percentile,
		clip_high_percentile=config.normalization.clip_high_percentile,
		max_samples=config.normalization.max_samples,
		seed=config.normalization.seed,
		eps=config.normalization.eps,
	)
	write_normalization_stats(
		replace(stats, source_path=config.outputs.amplitude_npy),
		staged.normalization_stats,
	)


def _build_metadata(  # noqa: PLR0913
	config: ParihakaPrepareVolumeConfig,
	*,
	source_sha256: str,
	source_size: int,
	source_statistics: ArrayStatistics,
	output_sha256: str,
	output_size: int,
	output_statistics: ArrayStatistics,
	companion_hashes: Mapping[str, str],
) -> dict[str, object]:
	source = config.source
	return {
		'artifact_type': 'parihaka_amplitude_preparation',
		'schema_version': 1,
		'status': 'complete',
		'dataset': {
			'name': config.dataset.name,
			'version': config.dataset.version,
			'survey_id': config.dataset.survey_id,
		},
		'provenance': {
			'direct_distributor': source.direct_distributor,
			'dataset_title': source.dataset_title,
			'contributor': source.contributor,
			'version': source.version,
			'doi': source.doi,
			'upstream': source.upstream,
			'displayed_license': source.displayed_license,
			'archive_name': source.archive_name,
			'upstream_member': source.upstream_member,
			'local_filename': source.local_filename,
			'local_modification': source.local_modification,
			'aicrowd_byte_identity': source.aicrowd_byte_identity,
			'redistribution_transformation': source.redistribution_transformation,
			'acquisition_date': source.acquisition_date,
			'acquisition_by': source.acquisition_by,
		},
		'source': {
			'npz_path': str(config.inputs.amplitude_npz),
			'npz_sha256': source_sha256,
			'npz_size_bytes': source_size,
			'array_key': source.array_key,
			'member_name': source.member_name,
			'npy_header': {
				'shape': list(source.shape_zxy),
				'dtype': source.dtype,
				'fortran_order': source.fortran_order,
			},
			'logical_axes': ['Z', 'X', 'Y'],
			'shape_zxy': list(source.shape_zxy),
			'dtype': source.dtype,
			'fortran_order': source.fortran_order,
			'statistics': source_statistics.to_dict(),
		},
		'conversion': {
			'axis_mapping': 'ZXY -> XYZ',
			'transpose_axes': list(config.conversion.transpose_axes),
			'chunk_size_z': config.conversion.chunk_size_z,
			'verification': 'full_chunkwise_bitwise',
		},
		'outputs': {
			'amplitude_npy': {
				'path': str(config.outputs.amplitude_npy),
				'sha256': output_sha256,
				'size_bytes': output_size,
				'logical_axes': ['X', 'Y', 'Z'],
				'shape_xyz': list(_destination_shape(source.shape_zxy)),
				'dtype': source.dtype,
				'order': 'C',
				'statistics': output_statistics.to_dict(),
			},
			'manifest': _output_reference(
				config.outputs.manifest, companion_hashes['manifest']
			),
			'path_list': _output_reference(
				config.outputs.path_list, companion_hashes['path_list']
			),
			'normalization_stats': _output_reference(
				config.outputs.normalization_stats,
				companion_hashes['normalization_stats'],
			),
			'metadata': {'path': str(config.outputs.metadata)},
		},
	}


def _output_reference(path: Path, digest: str) -> dict[str, str]:
	return {'path': str(path), 'sha256': digest}


def _publish_staged_outputs(
	staged: ParihakaPrepareOutputPaths,
	final: ParihakaPrepareOutputPaths,
	*,
	overwrite: bool,
) -> None:
	staged_files = staged.files()
	final_files = final.files()
	backups: dict[Path, Path] = {}
	published: list[Path] = []
	try:
		if overwrite:
			for index, path in enumerate(final_files):
				backup = staged.data_dir / f'.backup-{index}-{path.name}'
				path.replace(backup)
				backups[path] = backup
		for source, destination in zip(staged_files, final_files, strict=True):
			source.replace(destination)
			published.append(destination)
	except BaseException:
		for path in reversed(published):
			path.unlink(missing_ok=True)
		for path, backup in backups.items():
			if backup.exists():
				backup.replace(path)
		raise


def _staged_outputs(
	final: ParihakaPrepareOutputPaths,
	staging_dir: Path,
) -> ParihakaPrepareOutputPaths:
	return ParihakaPrepareOutputPaths(
		data_dir=staging_dir,
		amplitude_npy=staging_dir / final.amplitude_npy.name,
		manifest=staging_dir / final.manifest.name,
		path_list=staging_dir / final.path_list.name,
		normalization_stats=staging_dir / final.normalization_stats.name,
		metadata=staging_dir / final.metadata.name,
	)


def _validate_runtime_config(config: ParihakaPrepareVolumeConfig) -> None:
	_validate_paths(config)
	if config.conversion.transpose_axes != (1, 2, 0):
		msg = 'conversion.transpose_axes must be [1, 2, 0]'
		raise ValueError(msg)
	z_length = config.source.shape_zxy[0]
	if (
		isinstance(config.conversion.chunk_size_z, bool)
		or not isinstance(config.conversion.chunk_size_z, int)
		or not 1 <= config.conversion.chunk_size_z <= z_length
	):
		msg = f'conversion.chunk_size_z must be in [1, {z_length}]'
		raise ValueError(msg)
	if np.dtype(config.source.dtype).hasobject:
		msg = 'source.dtype must not be an object dtype'
		raise TypeError(msg)


def _validate_paths(config: ParihakaPrepareVolumeConfig) -> None:
	paths = config.paths
	_validate_root_paths(paths)
	if not _is_relative_to(config.inputs.amplitude_npz, paths.parihaka_root):
		msg = 'inputs.amplitude_npz must be under paths.parihaka_root'
		raise ValueError(msg)
	output_files = config.outputs.files()
	for output in (config.outputs.data_dir, *output_files):
		_validate_output_root(output, paths)
	for output in output_files:
		if output.parent != config.outputs.data_dir:
			msg = f'direct output must be inside outputs.data_dir: {output}'
			raise ValueError(msg)
		_validate_distinct_paths(
			output,
			'output path',
			config.inputs.amplitude_npz,
			'inputs.amplitude_npz',
		)
	if len({path.resolve(strict=False) for path in output_files}) != len(output_files):
		msg = 'all final output paths must differ'
		raise ValueError(msg)


def _validate_root_paths(paths: ParihakaPrepareRootPaths) -> None:
	for label, root in (
		('paths.artifact_root', paths.artifact_root),
		('paths.parihaka_root', paths.parihaka_root),
	):
		if not root.is_absolute():
			msg = f'{label} must be an absolute path: {root}'
			raise ValueError(msg)
	_validate_disjoint_directories(
		paths.artifact_root,
		'paths.artifact_root',
		paths.parihaka_root,
		'paths.parihaka_root',
	)


def _validate_output_root(output: Path, paths: ParihakaPrepareRootPaths) -> None:
	if not output.is_absolute():
		msg = f'output path must be absolute: {output}'
		raise ValueError(msg)
	if not _is_relative_to(output, paths.artifact_root):
		msg = f'output path must be under paths.artifact_root: {output}'
		raise ValueError(msg)
	if _is_relative_to(output, paths.parihaka_root):
		msg = f'output path must be outside paths.parihaka_root: {output}'
		raise ValueError(msg)


def _validate_source_path(config: ParihakaPrepareVolumeConfig) -> None:
	source = config.inputs.amplitude_npz
	if source.name != config.source.local_filename:
		msg = f'source NPZ filename must be {config.source.local_filename}: {source}'
		raise ValueError(msg)
	if not source.is_file():
		msg = f'source amplitude NPZ does not exist: {source}'
		raise FileNotFoundError(msg)


def _validate_existing_outputs(
	outputs: ParihakaPrepareOutputPaths,
	*,
	overwrite: bool,
) -> None:
	if outputs.data_dir.exists() and not outputs.data_dir.is_dir():
		msg = f'outputs.data_dir exists but is not a directory: {outputs.data_dir}'
		raise ValueError(msg)
	if outputs.data_dir.is_dir():
		stale = sorted(
			path
			for path in outputs.data_dir.iterdir()
			if path.name.startswith(_STAGING_PREFIX)
		)
		if stale:
			msg = f'stale Parihaka staging path(s) must not be reused: {stale!r}'
			raise FileExistsError(msg)
	exists = [path.exists() for path in outputs.files()]
	if not any(exists):
		return
	if not all(exists):
		present = [
			str(path)
			for path, found in zip(outputs.files(), exists, strict=True)
			if found
		]
		missing = [
			str(path)
			for path, found in zip(outputs.files(), exists, strict=True)
			if not found
		]
		msg = (
			'partial final Parihaka output state; '
			f'present={present!r}, missing={missing!r}'
		)
		raise FileExistsError(msg)
	_validate_complete_metadata(outputs)
	if not overwrite:
		msg = (
			'complete Parihaka outputs already exist; use --overwrite: '
			f'{outputs.data_dir}'
		)
		raise FileExistsError(msg)


def _validate_complete_metadata(outputs: ParihakaPrepareOutputPaths) -> None:
	try:
		payload = json.loads(outputs.metadata.read_text(encoding='utf-8'))
		if (
			payload.get('artifact_type') != 'parihaka_amplitude_preparation'
			or payload.get('schema_version') != 1
			or payload.get('status') != 'complete'
		):
			raise ValueError('invalid completion identity')
		output_records = cast('Mapping[str, object]', payload['outputs'])
		metadata_record = cast('Mapping[str, object]', output_records['metadata'])
		if metadata_record.get('path') != str(outputs.metadata):
			msg = f'existing completion metadata path drift for {outputs.metadata}'
			raise ValueError(msg)
		references = {
			'amplitude_npy': outputs.amplitude_npy,
			'manifest': outputs.manifest,
			'path_list': outputs.path_list,
			'normalization_stats': outputs.normalization_stats,
		}
		for key, path in references.items():
			record = cast('Mapping[str, object]', output_records[key])
			if record.get('path') != str(path) or record.get('sha256') != _file_sha256(
				path
			):
				msg = f'existing completion metadata hash/path drift for {path}'
				raise ValueError(msg)
	except (KeyError, TypeError, json.JSONDecodeError) as exc:
		msg = f'invalid existing completion metadata: {outputs.metadata}: {exc}'
		raise ValueError(msg) from exc


def _validate_statistics(
	actual: ArrayStatistics,
	expected: ArrayStatistics,
	*,
	label: str,
) -> None:
	for field in ('element_count', 'finite_count', 'nonfinite_count'):
		if getattr(actual, field) != getattr(expected, field):
			msg = _statistics_mismatch_message(actual, expected, label, field)
			raise ValueError(msg)
	for field in ('minimum', 'maximum'):
		if (
			np.float32(getattr(actual, field)).tobytes()
			!= np.float32(getattr(expected, field)).tobytes()
		):
			msg = _statistics_mismatch_message(actual, expected, label, field)
			raise ValueError(msg)
	for field in ('mean', 'population_std'):
		if not math.isclose(
			getattr(actual, field),
			getattr(expected, field),
			rel_tol=0.0,
			abs_tol=_STAT_ABS_TOL,
		):
			msg = _statistics_mismatch_message(actual, expected, label, field)
			raise ValueError(msg)


def _statistics_mismatch_message(
	actual: ArrayStatistics,
	expected: ArrayStatistics,
	label: str,
	field: str,
) -> str:
	return (
		f'{label} {field} mismatch: expected {getattr(expected, field)!r}, '
		f'got {getattr(actual, field)!r}'
	)


def _destination_shape(shape_zxy: tuple[int, int, int]) -> tuple[int, int, int]:
	return (shape_zxy[1], shape_zxy[2], shape_zxy[0])


def _z_chunks(length: int, chunk_size: int) -> Sequence[tuple[int, int]]:
	return tuple(
		(start, min(start + chunk_size, length))
		for start in range(0, length, chunk_size)
	)


def _file_sha256(path: Path) -> str:
	digest = sha256()
	with path.open('rb') as file_obj:
		for chunk in iter(lambda: file_obj.read(_HASH_CHUNK_BYTES), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


def _parse_paths(data: Mapping[str, object]) -> ParihakaPrepareRootPaths:
	_validate_keys(data, {'artifact_root', 'parihaka_root'}, prefix='paths')
	return ParihakaPrepareRootPaths(
		artifact_root=_absolute_path(data, 'artifact_root', 'paths'),
		parihaka_root=_absolute_path(data, 'parihaka_root', 'paths'),
	)


def _parse_inputs(data: Mapping[str, object]) -> ParihakaPrepareInputPaths:
	_validate_keys(data, {'amplitude_npz'}, prefix='inputs')
	return ParihakaPrepareInputPaths(
		amplitude_npz=_absolute_path(data, 'amplitude_npz', 'inputs'),
	)


def _parse_outputs(data: Mapping[str, object]) -> ParihakaPrepareOutputPaths:
	keys = {
		'data_dir',
		'amplitude_npy',
		'manifest',
		'path_list',
		'normalization_stats',
		'metadata',
	}
	_validate_keys(data, keys, prefix='outputs')
	return ParihakaPrepareOutputPaths(
		data_dir=_absolute_path(data, 'data_dir', 'outputs'),
		amplitude_npy=_absolute_path(data, 'amplitude_npy', 'outputs'),
		manifest=_absolute_path(data, 'manifest', 'outputs'),
		path_list=_absolute_path(data, 'path_list', 'outputs'),
		normalization_stats=_absolute_path(data, 'normalization_stats', 'outputs'),
		metadata=_absolute_path(data, 'metadata', 'outputs'),
	)


def _parse_dataset(data: Mapping[str, object]) -> ParihakaPrepareDatasetConfig:
	_validate_keys(data, set(_FIXED_DATASET), prefix='dataset')
	return ParihakaPrepareDatasetConfig(
		name=_string(data, 'name', 'dataset'),
		version=_string(data, 'version', 'dataset'),
		survey_id=_string(data, 'survey_id', 'dataset'),
	)


def _parse_source(data: Mapping[str, object]) -> ParihakaPrepareSourceConfig:
	keys = set(_FIXED_SOURCE) | {'acquisition_date', 'acquisition_by'}
	_validate_keys(data, keys, prefix='source')
	shape = _int_triplet(data, 'shape_zxy', 'source')
	statistics = ArrayStatistics(
		element_count=_integer(data, 'element_count', 'source'),
		finite_count=_integer(data, 'finite_count', 'source'),
		nonfinite_count=_integer(data, 'nonfinite_count', 'source'),
		minimum=_number(data, 'min', 'source'),
		maximum=_number(data, 'max', 'source'),
		mean=_number(data, 'mean', 'source'),
		population_std=_number(data, 'population_std', 'source'),
	)
	return ParihakaPrepareSourceConfig(
		direct_distributor=_string(data, 'direct_distributor', 'source'),
		dataset_title=_string(data, 'dataset_title', 'source'),
		contributor=_string(data, 'contributor', 'source'),
		version=_integer(data, 'version', 'source'),
		doi=_string(data, 'doi', 'source'),
		upstream=_string(data, 'upstream', 'source'),
		displayed_license=_string(data, 'displayed_license', 'source'),
		archive_name=_string(data, 'archive_name', 'source'),
		upstream_member=_string(data, 'upstream_member', 'source'),
		local_filename=_string(data, 'local_filename', 'source'),
		local_modification=_string(data, 'local_modification', 'source'),
		aicrowd_byte_identity=_string(data, 'aicrowd_byte_identity', 'source'),
		redistribution_transformation=_string(
			data, 'redistribution_transformation', 'source'
		),
		acquisition_date=_optional_string(data, 'acquisition_date', 'source'),
		acquisition_by=_optional_string(data, 'acquisition_by', 'source'),
		array_key=_string(data, 'array_key', 'source'),
		member_name=_string(data, 'member_name', 'source'),
		shape_zxy=shape,
		dtype=_string(data, 'dtype', 'source'),
		fortran_order=_boolean(data, 'fortran_order', 'source'),
		expected_statistics=statistics,
	)


def _parse_conversion(
	data: Mapping[str, object],
	z_length: int,
) -> ParihakaPrepareConversionConfig:
	_validate_keys(data, {'transpose_axes', 'chunk_size_z'}, prefix='conversion')
	transpose_axes = _int_triplet(data, 'transpose_axes', 'conversion')
	chunk_size = _integer(data, 'chunk_size_z', 'conversion')
	if not 1 <= chunk_size <= z_length:
		msg = f'conversion.chunk_size_z must be in [1, {z_length}]'
		raise ValueError(msg)
	return ParihakaPrepareConversionConfig(transpose_axes, chunk_size)


def _parse_normalization(
	data: Mapping[str, object],
) -> ParihakaPrepareNormalizationConfig:
	_validate_keys(data, set(_FIXED_NORMALIZATION), prefix='normalization')
	percentiles = data.get('clipping_percentiles')
	if (
		not isinstance(percentiles, list)
		or len(percentiles) != 2
		or any(
			isinstance(value, bool) or not isinstance(value, Real)
			for value in percentiles
		)
	):
		msg = 'normalization.clipping_percentiles must be a two-number list'
		raise TypeError(msg)
	return ParihakaPrepareNormalizationConfig(
		clip_low_percentile=float(percentiles[0]),
		clip_high_percentile=float(percentiles[1]),
		eps=_number(data, 'epsilon', 'normalization'),
		max_samples=_integer(data, 'max_samples', 'normalization'),
		seed=_integer(data, 'seed', 'normalization'),
	)


def _validate_fixed_mapping(config: Mapping[str, object]) -> None:
	for section, expected in (
		('dataset', _FIXED_DATASET),
		('source', _FIXED_SOURCE),
		('normalization', _FIXED_NORMALIZATION),
	):
		actual = _mapping(config, section)
		for key, value in expected.items():
			if actual.get(key) != value:
				msg = (
					f'{section}.{key} must be fixed to {value!r}; '
					f'got {actual.get(key)!r}'
				)
				raise ValueError(msg)
	conversion = _mapping(config, 'conversion')
	if conversion.get('transpose_axes') != _FIXED_CONVERSION['transpose_axes']:
		msg = 'conversion.transpose_axes must be fixed to [1, 2, 0]'
		raise ValueError(msg)
	paths = _parse_paths(_mapping(config, 'paths'))
	inputs = _parse_inputs(_mapping(config, 'inputs'))
	outputs = _parse_outputs(_mapping(config, 'outputs'))
	expected_dir = paths.artifact_root / 'data' / 'parihaka' / 'facies_benchmark_v1'
	expected_outputs = {
		'data_dir': expected_dir,
		'amplitude_npy': expected_dir / PARIHAKA_AMPLITUDE_NPY_NAME,
		'manifest': expected_dir / PARIHAKA_MANIFEST_NAME,
		'path_list': expected_dir / PARIHAKA_PATH_LIST_NAME,
		'normalization_stats': expected_dir / PARIHAKA_NORMALIZATION_STATS_NAME,
		'metadata': expected_dir / PARIHAKA_METADATA_NAME,
	}
	for key, expected in expected_outputs.items():
		if getattr(outputs, key) != expected:
			msg = f'outputs.{key} must be the direct path {expected}'
			raise ValueError(msg)
	if inputs.amplitude_npz != paths.parihaka_root / _FIXED_SOURCE['local_filename']:
		msg = 'inputs.amplitude_npz must be the direct file under paths.parihaka_root'
		raise ValueError(msg)


def _validate_keys(
	data: Mapping[str, object], expected: set[str], *, prefix: str
) -> None:
	missing = sorted(expected - set(data))
	unexpected = sorted(set(data) - expected)
	if missing:
		msg = f'{prefix} missing required key(s): {missing!r}'
		raise ValueError(msg)
	if unexpected:
		msg = f'{prefix} key(s) not allowed: {unexpected!r}'
		raise ValueError(msg)


def _mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = data.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _absolute_path(data: Mapping[str, object], key: str, prefix: str) -> Path:
	path = Path(_string(data, key, prefix))
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path: {path}'
		raise ValueError(msg)
	return path


def _string(data: Mapping[str, object], key: str, prefix: str) -> str:
	value = data.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_string(data: Mapping[str, object], key: str, prefix: str) -> str | None:
	value = data.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be null or a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _integer(data: Mapping[str, object], key: str, prefix: str) -> int:
	value = data.get(key)
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{prefix}.{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return int(value)


def _number(data: Mapping[str, object], key: str, prefix: str) -> float:
	value = data.get(key)
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{prefix}.{key} must be a number; got {value!r}'
		raise TypeError(msg)
	return float(value)


def _boolean(data: Mapping[str, object], key: str, prefix: str) -> bool:
	value = data.get(key)
	if not isinstance(value, bool):
		msg = f'{prefix}.{key} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _int_triplet(
	data: Mapping[str, object], key: str, prefix: str
) -> tuple[int, int, int]:
	value = data.get(key)
	if (
		not isinstance(value, list)
		or len(value) != 3
		or any(
			isinstance(item, bool) or not isinstance(item, Integral) for item in value
		)
	):
		msg = f'{prefix}.{key} must be a three-integer list'
		raise TypeError(msg)
	return cast('tuple[int, int, int]', tuple(int(item) for item in value))


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


__all__ = [
	'PARIHAKA_AMPLITUDE_NPY_NAME',
	'PARIHAKA_MANIFEST_NAME',
	'PARIHAKA_METADATA_NAME',
	'PARIHAKA_NORMALIZATION_STATS_NAME',
	'PARIHAKA_PATH_LIST_NAME',
	'ArrayStatistics',
	'ParihakaPrepareConversionConfig',
	'ParihakaPrepareDatasetConfig',
	'ParihakaPrepareInputPaths',
	'ParihakaPrepareNormalizationConfig',
	'ParihakaPrepareOutputPaths',
	'ParihakaPrepareRootPaths',
	'ParihakaPrepareSourceConfig',
	'ParihakaPrepareVolumeConfig',
	'ParihakaPrepareVolumeResult',
	'inspect_parihaka_preparation',
	'parihaka_prepare_volume_config_from_mapping',
	'prepare_parihaka_volume',
]
