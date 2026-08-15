'''Validate and register the read-only Volve canonical amplitude volume.'''

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, cast

import numpy as np

from seis_ssl_cluster.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
)
from seis_ssl_cluster.data.schema import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	read_manifest_json,
	survey_manifest_to_dict,
)

VOLVE_DEFAULT_ROOT = Path('/home/dcuser/public_data/field/volve')
VOLVE_CANONICAL_RELATIVE_ROOT = Path('canonical/volve_st10010_full_t_v1')
VOLVE_OUTPUT_RELATIVE_ROOT = Path('data/volve/horizon_benchmark_v1')
VOLVE_CANONICAL_DATASET_ID = 'volve_st10010_full_t_v1'
VOLVE_SURVEY_ID = 'volve_st10010'
VOLVE_SOURCE_SEGY_SHA256 = (
	'f902e2bdaa277caf93a32e5f35eae653eb8b923138db0efc1e91918ef6757b2e'
)
VOLVE_AMPLITUDE_SHA256 = (
	'8e6a66c671658b2b24b9a961652972802e2735eaad3f7166642e52064bf46567'
)
VOLVE_AMPLITUDE_SIZE_BYTES = 981_648_128
VOLVE_SHAPE_XYZ = (401, 720, 850)
VOLVE_VALID_TRACE_COUNT = 288_694
VOLVE_MISSING_TRACE_COUNT = 26

_AMPLITUDE_NAME = 'amplitude.npy'
_VALID_MASK_NAME = 'valid_trace_mask.npy'
_INLINE_VALUES_NAME = 'inline_values.npy'
_CROSSLINE_VALUES_NAME = 'crossline_values.npy'
_TIME_NAME = 'time_ms.npy'
_CANONICAL_MANIFEST_NAME = 'canonical_volume_manifest.json'
_SOURCE_STATS_NAME = 'normalization_stats.json'
_TRACE_PARITY_NAME = 'trace_parity.json'
_REQUIRED_INPUT_NAMES = (
	_AMPLITUDE_NAME,
	_VALID_MASK_NAME,
	_INLINE_VALUES_NAME,
	_CROSSLINE_VALUES_NAME,
	_TIME_NAME,
	_CANONICAL_MANIFEST_NAME,
	_SOURCE_STATS_NAME,
	_TRACE_PARITY_NAME,
)
_MANIFEST_NAME = 'volve_amplitude_manifest.json'
_PATH_LIST_NAME = 'volve_npy_paths.txt'
_STATS_NAME = 'volve.normalization_stats.json'
_METADATA_NAME = 'volve_canonical_input_metadata.json'
_SCAN_CHUNK_INLINE_COUNT = 8


@dataclass(frozen=True)
class VolveCanonicalIdentity:
	'''Expected scientific identity of one canonical Volve-like input.'''

	dataset_id: str = VOLVE_CANONICAL_DATASET_ID
	survey_id: str = VOLVE_SURVEY_ID
	shape_xyz: tuple[int, int, int] = VOLVE_SHAPE_XYZ
	dtype: str = 'float32'
	valid_trace_count: int = VOLVE_VALID_TRACE_COUNT
	missing_trace_count: int = VOLVE_MISSING_TRACE_COUNT
	source_segy_sha256: str = VOLVE_SOURCE_SEGY_SHA256
	amplitude_sha256: str = VOLVE_AMPLITUDE_SHA256
	amplitude_size_bytes: int = VOLVE_AMPLITUDE_SIZE_BYTES


@dataclass(frozen=True)
class VolveCanonicalInputPaths:
	'''Resolved public inputs and small artifact outputs.'''

	canonical_root: Path
	output_dir: Path
	manifest_path: Path
	path_list_path: Path
	normalization_stats_path: Path
	metadata_path: Path


@dataclass(frozen=True)
class VolveCanonicalInputConfig:
	'''Configuration for validating and registering canonical inputs.'''

	volve_root: Path
	artifact_root: Path
	identity: VolveCanonicalIdentity = VolveCanonicalIdentity()

	@property
	def paths(self) -> VolveCanonicalInputPaths:
		'''Resolve fixed canonical and output locations from configured roots.'''
		canonical_root = self.volve_root / VOLVE_CANONICAL_RELATIVE_ROOT
		output_dir = self.artifact_root / VOLVE_OUTPUT_RELATIVE_ROOT
		return VolveCanonicalInputPaths(
			canonical_root=canonical_root,
			output_dir=output_dir,
			manifest_path=output_dir / _MANIFEST_NAME,
			path_list_path=output_dir / _PATH_LIST_NAME,
			normalization_stats_path=output_dir / _STATS_NAME,
			metadata_path=output_dir / _METADATA_NAME,
		)


@dataclass(frozen=True)
class VolveCanonicalInputResult:
	'''Outcome and paths for one input registration attempt.'''

	paths: VolveCanonicalInputPaths
	action: Literal['DRY_RUN', 'WROTE', 'REUSE']
	manifest: SurveyManifest
	normalization_stats: SurveyNormalizationStats
	scientific_identity_sha256: str


@dataclass(frozen=True)
class _ValidatedCanonicalInputs:
	amplitude_path: Path
	valid_mask_path: Path
	canonical_manifest_path: Path
	source_stats_path: Path
	trace_parity_path: Path
	source_stats: Mapping[str, object]
	canonical_manifest_sha256: str
	valid_mask_sha256: str
	inline_values_sha256: str
	crossline_values_sha256: str
	time_axis_sha256: str
	source_stats_sha256: str
	trace_parity_sha256: str
	valid_trace_count: int
	missing_trace_count: int


def resolve_volve_canonical_input_config(
	config: Mapping[str, object],
) -> VolveCanonicalInputConfig:
	'''Validate a config mapping while keeping the dataset identity fixed.'''
	_validate_allowed_keys(config, frozenset({'paths'}), label='config')
	paths = _required_mapping(config, 'paths', label='config')
	_validate_allowed_keys(
		paths,
		frozenset({'volve_root', 'artifact_root'}),
		label='paths',
	)
	volve_root = _required_absolute_path(paths, 'volve_root', label='paths')
	artifact_root = _required_absolute_path(paths, 'artifact_root', label='paths')
	resolved = VolveCanonicalInputConfig(
		volve_root=volve_root,
		artifact_root=artifact_root,
	)
	_validate_output_location(resolved)
	return resolved


def prepare_volve_canonical_inputs(
	config: VolveCanonicalInputConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
) -> VolveCanonicalInputResult:
	'''Validate public canonical inputs and emit only small registry artifacts.'''
	_validate_config(config)
	validated = _validate_canonical_inputs(config)
	manifest, stats, metadata = _build_outputs(config, validated)
	identity_hash = cast('str', metadata['scientific_identity_sha256'])

	if dry_run:
		return VolveCanonicalInputResult(
			paths=config.paths,
			action='DRY_RUN',
			manifest=manifest,
			normalization_stats=stats,
			scientific_identity_sha256=identity_hash,
		)

	output_paths = _output_files(config.paths)
	existing = tuple(path.is_file() for path in output_paths)
	if any(existing):
		if not only_missing:
			msg = (
				'Volve canonical input output already exists; use --only-missing '
				'to validate and reuse a complete matching output set'
			)
			raise FileExistsError(msg)
		if not all(existing):
			missing = [
				str(path)
				for path, exists in zip(output_paths, existing, strict=True)
				if not exists
			]
			msg = (
				'incomplete Volve canonical input output set; '
				f'missing: {missing!r}'
			)
			raise FileExistsError(msg)
		_validate_existing_outputs(config.paths, manifest, stats, metadata)
		return VolveCanonicalInputResult(
			paths=config.paths,
			action='REUSE',
			manifest=manifest,
			normalization_stats=stats,
			scientific_identity_sha256=identity_hash,
		)

	config.paths.output_dir.mkdir(parents=True, exist_ok=True)
	_write_json(config.paths.manifest_path, [survey_manifest_to_dict(manifest)])
	_write_json(config.paths.normalization_stats_path, stats.to_dict())
	config.paths.path_list_path.write_text(
		f'{validated.amplitude_path}\n',
		encoding='utf-8',
	)
	_write_json(config.paths.metadata_path, metadata)
	return VolveCanonicalInputResult(
		paths=config.paths,
		action='WROTE',
		manifest=manifest,
		normalization_stats=stats,
		scientific_identity_sha256=identity_hash,
	)


def _validate_config(config: VolveCanonicalInputConfig) -> None:
	if not isinstance(config, VolveCanonicalInputConfig):
		msg = f'config must be VolveCanonicalInputConfig; got {config!r}'
		raise TypeError(msg)
	for path, label in (
		(config.volve_root, 'volve_root'),
		(config.artifact_root, 'artifact_root'),
	):
		if not path.is_absolute():
			msg = f'{label} must be an absolute path; got {path}'
			raise ValueError(msg)
	_validate_identity(config.identity)
	_validate_output_location(config)


def _validate_identity(identity: VolveCanonicalIdentity) -> None:
	if not identity.dataset_id or not identity.survey_id:
		msg = 'canonical dataset_id and survey_id must be non-empty'
		raise ValueError(msg)
	if len(identity.shape_xyz) != 3 or any(
		axis <= 0 for axis in identity.shape_xyz
	):
		msg = (
			'canonical shape_xyz must contain three positive values: '
			f'{identity.shape_xyz!r}'
		)
		raise ValueError(msg)
	if np.dtype(identity.dtype) != np.dtype(np.float32):
		msg = f'canonical dtype must be float32; got {identity.dtype!r}'
		raise ValueError(msg)
	trace_count = identity.shape_xyz[0] * identity.shape_xyz[1]
	if identity.valid_trace_count + identity.missing_trace_count != trace_count:
		msg = 'canonical valid and missing trace counts must cover the XY grid'
		raise ValueError(msg)
	_validate_sha256(identity.source_segy_sha256, 'source SEG-Y SHA-256')
	_validate_sha256(identity.amplitude_sha256, 'amplitude SHA-256')
	if identity.amplitude_size_bytes <= 0:
		msg = 'canonical amplitude size must be positive'
		raise ValueError(msg)


def _validate_output_location(config: VolveCanonicalInputConfig) -> None:
	canonical_root = config.paths.canonical_root.resolve(strict=False)
	output_dir = config.paths.output_dir.resolve(strict=False)
	if _is_relative_to(output_dir, config.volve_root.resolve(strict=False)):
		msg = f'Volve outputs must not be under the public Volve root: {output_dir}'
		raise ValueError(msg)
	if output_dir == canonical_root:
		msg = 'Volve canonical input and output directories must differ'
		raise ValueError(msg)


def _validate_canonical_inputs(
	config: VolveCanonicalInputConfig,
) -> _ValidatedCanonicalInputs:
	root = config.paths.canonical_root
	for name in _REQUIRED_INPUT_NAMES:
		path = root / name
		if not path.is_file():
			msg = f'required Volve canonical input does not exist: {path}'
			raise FileNotFoundError(msg)

	manifest_path = root / _CANONICAL_MANIFEST_NAME
	canonical = _read_json_mapping(manifest_path, 'canonical volume manifest')
	_validate_canonical_manifest(canonical, config.identity)
	amplitude_path = (root / _AMPLITUDE_NAME).resolve(strict=True)
	_validate_amplitude_header(amplitude_path, config.identity)
	_validate_amplitude_identity(amplitude_path, canonical, config.identity)
	mask_path = (root / _VALID_MASK_NAME).resolve(strict=True)
	valid_count, missing_count = _validate_arrays_and_voxels(
		root,
		amplitude_path,
		mask_path,
		config.identity,
	)
	valid_mask_sha256 = _validate_manifest_artifact(
		canonical,
		_VALID_MASK_NAME,
		mask_path,
	)
	inline_values_sha256 = _validate_manifest_artifact(
		canonical,
		_INLINE_VALUES_NAME,
		root / _INLINE_VALUES_NAME,
	)
	crossline_values_sha256 = _validate_manifest_artifact(
		canonical,
		_CROSSLINE_VALUES_NAME,
		root / _CROSSLINE_VALUES_NAME,
	)
	time_axis_sha256 = _validate_manifest_artifact(
		canonical,
		_TIME_NAME,
		root / _TIME_NAME,
	)

	source_stats_path = root / _SOURCE_STATS_NAME
	source_stats = _read_json_mapping(
		source_stats_path,
		'canonical normalization stats',
	)
	_validate_source_stats(source_stats)
	source_stats_sha256 = _file_sha256(source_stats_path)
	_validate_manifest_artifact(canonical, _SOURCE_STATS_NAME, source_stats_path)

	parity_path = root / _TRACE_PARITY_NAME
	parity = _read_json_mapping(parity_path, 'canonical trace parity')
	_validate_trace_parity(parity, config.identity)
	_validate_manifest_artifact(canonical, _TRACE_PARITY_NAME, parity_path)

	return _ValidatedCanonicalInputs(
		amplitude_path=amplitude_path,
		valid_mask_path=mask_path,
		canonical_manifest_path=manifest_path.resolve(strict=True),
		source_stats_path=source_stats_path.resolve(strict=True),
		trace_parity_path=parity_path.resolve(strict=True),
		source_stats=source_stats,
		canonical_manifest_sha256=_file_sha256(manifest_path),
		valid_mask_sha256=valid_mask_sha256,
		inline_values_sha256=inline_values_sha256,
		crossline_values_sha256=crossline_values_sha256,
		time_axis_sha256=time_axis_sha256,
		source_stats_sha256=source_stats_sha256,
		trace_parity_sha256=_file_sha256(parity_path),
		valid_trace_count=valid_count,
		missing_trace_count=missing_count,
	)


def _validate_canonical_manifest(
	payload: Mapping[str, object],
	identity: VolveCanonicalIdentity,
) -> None:
	_require_equal(payload, 'schema_version', 1, 'canonical manifest')
	_require_equal(payload, 'status', 'PASS', 'canonical manifest')
	_require_equal(payload, 'dataset_id', identity.dataset_id, 'canonical manifest')
	_require_equal(payload, 'shape', list(identity.shape_xyz), 'canonical manifest')
	_require_equal(
		payload,
		'axis_order',
		['inline', 'crossline', 'twt'],
		'canonical manifest',
	)
	_require_equal(payload, 'dtype', identity.dtype, 'canonical manifest')
	geometry = _required_mapping(payload, 'geometry', label='canonical manifest')
	_require_equal(
		geometry,
		'valid_trace_count',
		identity.valid_trace_count,
		'canonical geometry',
	)
	_require_equal(
		geometry,
		'missing_trace_count',
		identity.missing_trace_count,
		'canonical geometry',
	)
	sources = _required_mapping(payload, 'sources', label='canonical manifest')
	segy = _required_mapping(sources, 'segy', label='canonical manifest sources')
	_require_equal(
		segy,
		'sha256',
		identity.source_segy_sha256,
		'canonical source SEG-Y',
	)


def _validate_amplitude_identity(
	path: Path,
	canonical: Mapping[str, object],
	identity: VolveCanonicalIdentity,
) -> None:
	if path.stat().st_size != identity.amplitude_size_bytes:
		msg = (
			'canonical amplitude file size mismatch: '
			f'expected {identity.amplitude_size_bytes}, got {path.stat().st_size}'
		)
		raise ValueError(msg)
	digest = _file_sha256(path)
	_validate_manifest_artifact(
		canonical,
		_AMPLITUDE_NAME,
		path,
		actual_sha256=digest,
	)
	if digest != identity.amplitude_sha256:
		msg = (
			'canonical amplitude SHA-256 mismatch: '
			f'expected {identity.amplitude_sha256}, got {digest}'
		)
		raise ValueError(msg)


def _validate_amplitude_header(
	path: Path,
	identity: VolveCanonicalIdentity,
) -> None:
	try:
		amplitude = np.load(path, mmap_mode='r', allow_pickle=False)
	except ValueError as exc:
		msg = f'failed to memory-map canonical amplitude: {exc}'
		raise ValueError(msg) from exc
	if tuple(amplitude.shape) != identity.shape_xyz:
		msg = (
			'canonical amplitude shape mismatch: '
			f'expected {identity.shape_xyz!r}, got {amplitude.shape!r}'
		)
		raise ValueError(msg)
	if amplitude.dtype != np.dtype(identity.dtype):
		msg = (
			'canonical amplitude dtype mismatch: '
			f'expected {identity.dtype}, got {amplitude.dtype}'
		)
		raise TypeError(msg)


def _validate_arrays_and_voxels(  # noqa: C901
	root: Path,
	amplitude_path: Path,
	mask_path: Path,
	identity: VolveCanonicalIdentity,
) -> tuple[int, int]:
	try:
		amplitude = np.load(amplitude_path, mmap_mode='r', allow_pickle=False)
		mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
	except ValueError as exc:
		msg = f'failed to memory-map canonical amplitude or mask: {exc}'
		raise ValueError(msg) from exc
	if tuple(amplitude.shape) != identity.shape_xyz:
		msg = f'canonical amplitude shape mismatch: {amplitude.shape!r}'
		raise ValueError(msg)
	if amplitude.dtype != np.dtype(identity.dtype):
		msg = f'canonical amplitude dtype mismatch: {amplitude.dtype}'
		raise TypeError(msg)
	expected_mask_shape = identity.shape_xyz[:2]
	if tuple(mask.shape) != expected_mask_shape:
		msg = f'canonical valid mask shape mismatch: {mask.shape!r}'
		raise ValueError(msg)
	if mask.dtype != np.dtype(bool):
		msg = f'canonical valid mask dtype must be bool; got {mask.dtype}'
		raise TypeError(msg)
	valid_count = 0
	missing_count = 0
	for start_x in range(0, identity.shape_xyz[0], _SCAN_CHUNK_INLINE_COUNT):
		stop_x = min(start_x + _SCAN_CHUNK_INLINE_COUNT, identity.shape_xyz[0])
		mask_chunk = np.asarray(mask[start_x:stop_x])
		amplitude_chunk = amplitude[start_x:stop_x]
		valid_count += int(np.count_nonzero(mask_chunk))
		missing_count += int(mask_chunk.size - np.count_nonzero(mask_chunk))
		if not np.isfinite(amplitude_chunk[mask_chunk]).all():
			msg = 'canonical amplitude contains non-finite samples on a valid trace'
			raise ValueError(msg)
		if not np.isnan(amplitude_chunk[~mask_chunk]).all():
			msg = 'canonical invalid traces must contain only NaN samples'
			raise ValueError(msg)
	if valid_count != identity.valid_trace_count:
		msg = f'canonical valid trace count mismatch: {valid_count}'
		raise ValueError(msg)
	if missing_count != identity.missing_trace_count:
		msg = f'canonical missing trace count mismatch: {missing_count}'
		raise ValueError(msg)
	_validate_axis_array(root / _INLINE_VALUES_NAME, identity.shape_xyz[0], 'inline')
	_validate_axis_array(
		root / _CROSSLINE_VALUES_NAME,
		identity.shape_xyz[1],
		'crossline',
	)
	_validate_time_axis(root / _TIME_NAME, identity.shape_xyz[2])
	return valid_count, missing_count


def _validate_axis_array(path: Path, expected_size: int, label: str) -> None:
	array = _load_npy_readonly(path, label)
	if array.ndim != 1 or array.size != expected_size:
		msg = f'canonical {label} axis shape mismatch: {array.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
		msg = f'canonical {label} axis must be finite numeric values'
		raise TypeError(msg)
	if array.size > 1 and not np.all(np.diff(array) > 0):
		msg = f'canonical {label} axis must be strictly increasing'
		raise ValueError(msg)


def _validate_time_axis(path: Path, expected_size: int) -> None:
	array = _load_npy_readonly(path, 'time')
	expected = np.arange(4.0, 4.0 * expected_size + 1.0, 4.0, dtype=np.float64)
	if array.ndim != 1 or array.size != expected_size:
		msg = f'canonical time axis shape mismatch: {array.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(array.dtype, np.number):
		msg = f'canonical time axis must be numeric; got {array.dtype}'
		raise TypeError(msg)
	if not np.array_equal(np.asarray(array, dtype=np.float64), expected):
		msg = 'canonical time axis must span 4 to 3400 ms at 4 ms intervals'
		raise ValueError(msg)


def _validate_source_stats(payload: Mapping[str, object]) -> None:
	_require_equal(payload, 'schema_version', 1, 'canonical normalization stats')
	_require_equal(payload, 'status', 'PASS', 'canonical normalization stats')
	deterministic = _required_mapping(
		payload,
		'deterministic_sample',
		label='canonical normalization stats',
	)
	policy = deterministic.get('policy')
	if not isinstance(policy, str) or not policy:
		msg = 'canonical normalization deterministic sample policy must be non-empty'
		raise ValueError(msg)
	quantiles = _required_mapping(
		deterministic,
		'value_quantiles',
		label='deterministic sample',
	)
	for key in ('p1', 'p25', 'p50', 'p75', 'p99'):
		_required_finite_real(quantiles, key, label='deterministic sample quantiles')


def _validate_trace_parity(
	payload: Mapping[str, object],
	identity: VolveCanonicalIdentity,
) -> None:
	_require_equal(payload, 'schema_version', 1, 'canonical trace parity')
	_require_equal(payload, 'status', 'PASS', 'canonical trace parity')
	_require_equal(
		payload,
		'all_exact',
		True,  # noqa: FBT003
		'canonical trace parity',
	)
	checks = _required_mapping(
		payload,
		'full_volume_checks',
		label='canonical trace parity',
	)
	valid_voxels = identity.valid_trace_count * identity.shape_xyz[2]
	missing_voxels = identity.missing_trace_count * identity.shape_xyz[2]
	_require_equal(
		checks,
		'valid_finite_voxel_count',
		valid_voxels,
		'trace parity checks',
	)
	_require_equal(
		checks,
		'missing_nan_voxel_count',
		missing_voxels,
		'trace parity checks',
	)


def _build_outputs(
	config: VolveCanonicalInputConfig,
	validated: _ValidatedCanonicalInputs,
) -> tuple[SurveyManifest, SurveyNormalizationStats, dict[str, object]]:
	identity = config.identity
	stats = _derive_normalization_stats(identity, validated)
	manifest = SurveyManifest(
		survey_id=identity.survey_id,
		root=config.paths.canonical_root.resolve(strict=True),
		amplitude=AmplitudeVolumeRecord(
			survey_id=identity.survey_id,
			path=Path(_AMPLITUDE_NAME),
			shape_xyz=identity.shape_xyz,
			dtype=identity.dtype,
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=config.paths.normalization_stats_path,
			valid_mask_path=Path(_VALID_MASK_NAME),
		),
	)
	manifest.validate()
	scientific_identity = {
		'dataset_id': identity.dataset_id,
		'survey_id': identity.survey_id,
		'shape_xyz': list(identity.shape_xyz),
		'grid_order': list(GRID_ORDER_XYZ),
		'dtype': identity.dtype,
		'valid_trace_count': identity.valid_trace_count,
		'missing_trace_count': identity.missing_trace_count,
		'source_segy_sha256': identity.source_segy_sha256,
		'canonical_amplitude_sha256': identity.amplitude_sha256,
		'valid_trace_mask_sha256': validated.valid_mask_sha256,
		'inline_values_sha256': validated.inline_values_sha256,
		'crossline_values_sha256': validated.crossline_values_sha256,
		'time_axis_sha256': validated.time_axis_sha256,
		'canonical_normalization_stats_sha256': validated.source_stats_sha256,
		'normalization': {
			'clip_low_percentile': stats.clip_low_percentile,
			'clip_high_percentile': stats.clip_high_percentile,
			'clip_low': stats.clip_low,
			'clip_high': stats.clip_high,
			'median': stats.median,
			'iqr': stats.iqr,
			'eps': stats.eps,
		},
	}
	identity_sha256 = _mapping_sha256(scientific_identity)
	deterministic = _required_mapping(
		validated.source_stats,
		'deterministic_sample',
		label='canonical normalization stats',
	)
	metadata: dict[str, object] = {
		'schema_version': 1,
		'artifact_type': 'volve_canonical_input_registration',
		'status': 'PASS',
		'scientific_identity': scientific_identity,
		'scientific_identity_sha256': identity_sha256,
		'provenance': {
			'canonical_root': str(config.paths.canonical_root.resolve(strict=True)),
			'public_inputs': {
				name: str((config.paths.canonical_root / name).resolve(strict=True))
				for name in _REQUIRED_INPUT_NAMES
			},
			'canonical_manifest': {
				'path': str(validated.canonical_manifest_path),
				'sha256': validated.canonical_manifest_sha256,
			},
			'amplitude': {
				'path': str(validated.amplitude_path),
				'sha256': identity.amplitude_sha256,
			},
			'valid_trace_mask': {'path': str(validated.valid_mask_path)},
			'trace_parity': {
				'path': str(validated.trace_parity_path),
				'sha256': validated.trace_parity_sha256,
			},
		},
		'normalization_provenance': {
			'derivation': 'canonical builder deterministic sample value quantiles',
			'sample_policy': deterministic['policy'],
			'source_stats_path': str(validated.source_stats_path),
			'source_stats_sha256': validated.source_stats_sha256,
		},
		'outputs': {
			'manifest': str(config.paths.manifest_path),
			'npy_paths': str(config.paths.path_list_path),
			'normalization_stats': str(config.paths.normalization_stats_path),
			'metadata': str(config.paths.metadata_path),
		},
		'validation': {
			'amplitude_scan': 'memory-mapped inline chunks',
			'chunk_inline_count': _SCAN_CHUNK_INLINE_COUNT,
			'valid_trace_count': validated.valid_trace_count,
			'missing_trace_count': validated.missing_trace_count,
			'valid_traces_finite': True,
			'invalid_traces_all_nan': True,
		},
	}
	return manifest, stats, metadata


def _derive_normalization_stats(
	identity: VolveCanonicalIdentity,
	validated: _ValidatedCanonicalInputs,
) -> SurveyNormalizationStats:
	deterministic = _required_mapping(
		validated.source_stats,
		'deterministic_sample',
		label='canonical normalization stats',
	)
	quantiles = _required_mapping(
		deterministic,
		'value_quantiles',
		label='deterministic sample',
	)
	quantile_label = 'deterministic sample quantiles'
	p1 = _required_finite_real(quantiles, 'p1', label=quantile_label)
	p25 = _required_finite_real(quantiles, 'p25', label=quantile_label)
	p50 = _required_finite_real(quantiles, 'p50', label=quantile_label)
	p75 = _required_finite_real(quantiles, 'p75', label=quantile_label)
	p99 = _required_finite_real(quantiles, 'p99', label=quantile_label)
	stats = SurveyNormalizationStats(
		survey_id=identity.survey_id,
		source_path=validated.amplitude_path,
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=1.0,
		clip_high_percentile=99.0,
		clip_low=p1,
		clip_high=p99,
		median=p50,
		iqr=p75 - p25,
		eps=1.0e-6,
	)
	stats.validate()
	return stats


def _validate_existing_outputs(
	paths: VolveCanonicalInputPaths,
	expected_manifest: SurveyManifest,
	expected_stats: SurveyNormalizationStats,
	expected_metadata: Mapping[str, object],
) -> None:
	try:
		manifests = read_manifest_json(paths.manifest_path)
		stats = load_normalization_stats(paths.normalization_stats_path)
	except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
		msg = f'existing Volve canonical outputs are invalid: {exc}'
		raise ValueError(msg) from exc
	if manifests != [expected_manifest]:
		msg = (
			'existing Volve amplitude manifest identity does not match '
			'canonical inputs'
		)
		raise ValueError(msg)
	if stats != expected_stats:
		msg = 'existing Volve normalization stats do not match canonical inputs'
		raise ValueError(msg)
	expected_amplitude_path = (
		expected_manifest.root / expected_manifest.amplitude.path
	)
	expected_path_list = f'{expected_amplitude_path}\n'
	if paths.path_list_path.read_text(encoding='utf-8') != expected_path_list:
		msg = 'existing Volve NPY path list does not match canonical inputs'
		raise ValueError(msg)
	metadata = _read_json_mapping(paths.metadata_path, 'existing Volve metadata')
	if metadata != expected_metadata:
		msg = 'existing Volve metadata identity does not match canonical inputs'
		raise ValueError(msg)


def _validate_manifest_artifact(
	canonical: Mapping[str, object],
	name: str,
	path: Path,
	*,
	actual_sha256: str | None = None,
) -> str:
	artifacts = _required_mapping(
		canonical,
		'artifacts',
		label='canonical manifest',
	)
	record = _required_mapping(
		artifacts,
		name,
		label='canonical manifest artifacts',
	)
	_require_equal(record, 'relative_path', name, f'canonical artifact {name}')
	_require_equal(
		record,
		'size_bytes',
		path.stat().st_size,
		f'canonical artifact {name}',
	)
	expected_hash = record.get('sha256')
	_validate_sha256(expected_hash, f'canonical artifact {name} SHA-256')
	actual_hash = _file_sha256(path) if actual_sha256 is None else actual_sha256
	if actual_hash != expected_hash:
		msg = (
			f'canonical artifact {name} SHA-256 mismatch: '
			f'expected {expected_hash}, got {actual_hash}'
		)
		raise ValueError(msg)
	return actual_hash


def _load_npy_readonly(path: Path, label: str) -> np.ndarray:
	try:
		return np.load(path, mmap_mode='r', allow_pickle=False)
	except ValueError as exc:
		msg = f'failed to memory-map canonical {label} axis: {exc}'
		raise ValueError(msg) from exc


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		msg = f'{label} must be valid JSON: {path}'
		raise ValueError(msg) from exc
	if not isinstance(payload, Mapping):
		msg = f'{label} must be a JSON object: {path}'
		raise TypeError(msg)
	return cast('Mapping[str, object]', payload)


def _write_json(path: Path, payload: object) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _mapping_sha256(payload: Mapping[str, object]) -> str:
	encoded = json.dumps(
		payload,
		sort_keys=True,
		separators=(',', ':'),
		allow_nan=False,
	).encode()
	return hashlib.sha256(encoded).hexdigest()


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
	*,
	label: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{label}.{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _required_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	label: str,
) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{label}.{key} must be a non-empty string'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{label}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _required_finite_real(
	parent: Mapping[str, object],
	key: str,
	*,
	label: str,
) -> float:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{label}.{key} must be numeric; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not np.isfinite(number):
		msg = f'{label}.{key} must be finite; got {value!r}'
		raise ValueError(msg)
	return number


def _require_equal(
	parent: Mapping[str, object],
	key: str,
	expected: object,
	label: str,
) -> None:
	actual = parent.get(key)
	if actual != expected:
		msg = f'{label}.{key} mismatch: expected {expected!r}, got {actual!r}'
		raise ValueError(msg)


def _validate_sha256(value: object, label: str) -> None:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		msg = f'{label} must be a lowercase hexadecimal SHA-256 digest'
		raise ValueError(msg)


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	label: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		msg = f'{label} key(s) not allowed: {unexpected!r}'
		raise ValueError(msg)


def _output_files(paths: VolveCanonicalInputPaths) -> tuple[Path, ...]:
	return (
		paths.manifest_path,
		paths.path_list_path,
		paths.normalization_stats_path,
		paths.metadata_path,
	)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.relative_to(root)
	except ValueError:
		return False
	return True


__all__ = [
	'VOLVE_AMPLITUDE_SHA256',
	'VOLVE_CANONICAL_DATASET_ID',
	'VOLVE_CANONICAL_RELATIVE_ROOT',
	'VOLVE_DEFAULT_ROOT',
	'VOLVE_SOURCE_SEGY_SHA256',
	'VOLVE_SURVEY_ID',
	'VolveCanonicalIdentity',
	'VolveCanonicalInputConfig',
	'VolveCanonicalInputPaths',
	'VolveCanonicalInputResult',
	'prepare_volve_canonical_inputs',
	'resolve_volve_canonical_input_config',
]
