'''Read and inspect the read-only Volve binding-v2 horizon surfaces.'''

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_ssl_cluster.volve.canonical_inputs import (
	VOLVE_CROSSLINE_MAX,
	VOLVE_CROSSLINE_MIN,
	VOLVE_FIRST_TWT_MS,
	VOLVE_INLINE_MAX,
	VOLVE_INLINE_MIN,
	VOLVE_SAMPLE_INTERVAL_MS,
	VOLVE_SHAPE_XYZ,
)

HORIZON_NAMES = (
	'ty_top',
	'shetland_top',
	'bcu',
	'hugin_top',
	'hugin_base',
)
BINDING_SCHEMA_VERSION = 2
BINDING_RELATIVE_ROOT = Path('manifests/volve_binding_v2')
VISUAL_QC_RELATIVE_ROOT = Path('qc/volve_binding_visual_qc_v1')
CANONICAL_RELATIVE_ROOT = Path('canonical/volve_st10010_full_t_v1')
HORIZON_BINDING_NAME = 'volve_horizon_binding_v2.npz'
HORIZON_SUMMARY_NAME = 'volve_horizon_binding_summary_v2.json'
GRID_SUMMARY_NAME = 'volve_grid_binding_summary_v2.json'
MANUAL_REVIEW_NAME = 'manual_review.json'
SECTION_STATISTICS_FIELDS = (
	'orientation',
	'physical_line_number',
	'array_index',
	'valid_trace_count',
	'five_horizon_common_count',
	'sample_strict_order_count',
	'ty_top_count',
	'shetland_top_count',
	'bcu_count',
	'hugin_top_count',
	'hugin_base_count',
	'minimum_horizon_count',
	'total_horizon_observation_count',
)
_NPZ_KEYS = frozenset(
	{
		'horizon_names',
		'twt_ms',
		'sample_float',
		'sample_index',
		'xy_residual_m',
		'source_present_mask',
		'bound_valid_mask',
		'common_bound_mask',
		'continuous_strict_order_mask',
		'sample_strict_order_mask',
	}
)


@dataclass(frozen=True)
class VolveHorizonGeometry:
	'''Fixed physical geometry expected by one Volve horizon input.'''

	shape_xyz: tuple[int, int, int] = VOLVE_SHAPE_XYZ
	inline_min: int = VOLVE_INLINE_MIN
	inline_max: int = VOLVE_INLINE_MAX
	crossline_min: int = VOLVE_CROSSLINE_MIN
	crossline_max: int = VOLVE_CROSSLINE_MAX
	first_twt_ms: float = VOLVE_FIRST_TWT_MS
	sample_interval_ms: float = VOLVE_SAMPLE_INTERVAL_MS


VOLVE_HORIZON_GEOMETRY = VolveHorizonGeometry()


@dataclass(frozen=True)
class VolveHorizonPaths:
	'''Resolved binding, visual-QC, and canonical paths.'''

	binding_npz: Path
	horizon_summary: Path
	grid_summary: Path
	manual_review: Path
	inline_values: Path
	crossline_values: Path
	time_ms: Path
	valid_trace_mask: Path


@dataclass(frozen=True)
class VolveHorizonData:
	'''Validated native Volve horizon arrays and their input identity.'''

	paths: VolveHorizonPaths
	inline_values: np.ndarray
	crossline_values: np.ndarray
	time_ms: np.ndarray
	valid_trace_mask: np.ndarray
	horizon_names: tuple[str, ...]
	twt_ms: np.ndarray
	sample_float: np.ndarray
	sample_index: np.ndarray
	xy_residual_m: np.ndarray
	source_present_mask: np.ndarray
	bound_valid_mask: np.ndarray
	common_bound_mask: np.ndarray
	continuous_strict_order_mask: np.ndarray
	sample_strict_order_mask: np.ndarray
	binding_npz_sha256: str
	horizon_summary_sha256: str
	grid_summary_sha256: str
	manual_review_sha256: str

	@property
	def shape_xy(self) -> tuple[int, int]:
		'''Return the canonical lateral shape.'''
		return (len(self.inline_values), len(self.crossline_values))

	def input_identity(self) -> dict[str, object]:
		'''Return the small scientific identity required by downstream plans.'''
		return {
			'binding_schema_version': BINDING_SCHEMA_VERSION,
			'binding_npz_sha256': self.binding_npz_sha256,
			'horizon_summary_sha256': self.horizon_summary_sha256,
			'grid_summary_sha256': self.grid_summary_sha256,
			'manual_review_sha256': self.manual_review_sha256,
			'horizon_names': list(self.horizon_names),
			'shape_xy': list(self.shape_xy),
			'inline_min': int(self.inline_values[0]),
			'inline_max': int(self.inline_values[-1]),
			'crossline_min': int(self.crossline_values[0]),
			'crossline_max': int(self.crossline_values[-1]),
		}


@dataclass(frozen=True)
class VolveHorizonInspectionConfig:
	'''Resolved paths for section inspection and split-plan construction.'''

	volve_root: Path
	artifact_root: Path
	layout_config: Path
	section_statistics_csv: Path
	split_plans_json: Path


def resolve_volve_horizon_inspection_config(
	config: Mapping[str, object],
) -> VolveHorizonInspectionConfig:
	'''Resolve the small inspection config and enforce artifact-only outputs.'''
	if set(config) != {'paths', 'outputs'}:
		raise ValueError('config must contain exactly paths and outputs')
	paths = _required_mapping(config, 'paths', 'config')
	outputs = _required_mapping(config, 'outputs', 'config')
	if set(paths) != {'volve_root', 'artifact_root', 'layout_config'}:
		raise ValueError(
			'paths must contain exactly volve_root, artifact_root, and layout_config'
		)
	if set(outputs) != {'section_statistics_csv', 'split_plans_json'}:
		raise ValueError(
			'outputs must contain exactly section_statistics_csv and split_plans_json'
		)
	volve_root = _absolute_path(paths.get('volve_root'), 'paths.volve_root')
	artifact_root = _absolute_path(
		paths.get('artifact_root'), 'paths.artifact_root'
	)
	layout_config = _path(paths.get('layout_config'), 'paths.layout_config').resolve()
	section_csv = _artifact_output(
		artifact_root,
		outputs.get('section_statistics_csv'),
		'outputs.section_statistics_csv',
	)
	plans_json = _artifact_output(
		artifact_root,
		outputs.get('split_plans_json'),
		'outputs.split_plans_json',
	)
	if _is_relative_to(section_csv, volve_root) or _is_relative_to(
		plans_json, volve_root
	):
		raise ValueError('inspection outputs must not be below public volve_root')
	return VolveHorizonInspectionConfig(
		volve_root=volve_root,
		artifact_root=artifact_root,
		layout_config=layout_config,
		section_statistics_csv=section_csv,
		split_plans_json=plans_json,
	)


def resolve_volve_horizon_paths(volve_root: str | Path) -> VolveHorizonPaths:
	'''Resolve the fixed binding-v2 input contract below a public root.'''
	root = Path(volve_root)
	if not root.is_absolute():
		raise ValueError('volve_root must be absolute')
	binding = root / BINDING_RELATIVE_ROOT
	canonical = root / CANONICAL_RELATIVE_ROOT
	return VolveHorizonPaths(
		binding_npz=binding / HORIZON_BINDING_NAME,
		horizon_summary=binding / HORIZON_SUMMARY_NAME,
		grid_summary=binding / GRID_SUMMARY_NAME,
		manual_review=root / VISUAL_QC_RELATIVE_ROOT / MANUAL_REVIEW_NAME,
		inline_values=canonical / 'inline_values.npy',
		crossline_values=canonical / 'crossline_values.npy',
		time_ms=canonical / 'time_ms.npy',
		valid_trace_mask=canonical / 'valid_trace_mask.npy',
	)


def load_volve_horizon_data(
	volve_root: str | Path,
	*,
	geometry: VolveHorizonGeometry = VOLVE_HORIZON_GEOMETRY,
) -> VolveHorizonData:
	'''Load and validate binding-v2 without modifying public inputs.'''
	paths = resolve_volve_horizon_paths(volve_root)
	for path in paths.__dict__.values():
		if not path.is_file():
			raise FileNotFoundError(f'missing Volve horizon input: {path}')

	horizon_summary = _read_json(paths.horizon_summary, 'horizon summary')
	grid_summary = _read_json(paths.grid_summary, 'grid summary')
	manual_review = _read_json(paths.manual_review, 'manual review')
	_validate_summaries(paths, horizon_summary, grid_summary, manual_review)

	inline_values = np.load(paths.inline_values, mmap_mode='r', allow_pickle=False)
	crossline_values = np.load(
		paths.crossline_values, mmap_mode='r', allow_pickle=False
	)
	time_ms = np.load(paths.time_ms, mmap_mode='r', allow_pickle=False)
	valid_trace_mask = np.load(
		paths.valid_trace_mask, mmap_mode='r', allow_pickle=False
	)
	_validate_canonical_arrays(
		inline_values,
		crossline_values,
		time_ms,
		valid_trace_mask,
		geometry,
		grid_summary,
	)

	with np.load(paths.binding_npz, allow_pickle=False) as archive:
		if set(archive.files) != _NPZ_KEYS:
			raise ValueError(
				'binding NPZ keys must be exactly '
				f'{sorted(_NPZ_KEYS)!r}; got {sorted(archive.files)!r}'
			)
		arrays = {name: archive[name] for name in archive.files}
	_validate_binding_arrays(arrays, geometry, horizon_summary, valid_trace_mask)
	return VolveHorizonData(
		paths=paths,
		inline_values=inline_values,
		crossline_values=crossline_values,
		time_ms=time_ms,
		valid_trace_mask=valid_trace_mask,
		horizon_names=tuple(str(value) for value in arrays['horizon_names']),
		twt_ms=arrays['twt_ms'],
		sample_float=arrays['sample_float'],
		sample_index=arrays['sample_index'],
		xy_residual_m=arrays['xy_residual_m'],
		source_present_mask=arrays['source_present_mask'],
		bound_valid_mask=arrays['bound_valid_mask'],
		common_bound_mask=arrays['common_bound_mask'],
		continuous_strict_order_mask=arrays['continuous_strict_order_mask'],
		sample_strict_order_mask=arrays['sample_strict_order_mask'],
		binding_npz_sha256=_file_sha256(paths.binding_npz),
		horizon_summary_sha256=_file_sha256(paths.horizon_summary),
		grid_summary_sha256=_file_sha256(paths.grid_summary),
		manual_review_sha256=_file_sha256(paths.manual_review),
	)


def section_statistics(data: VolveHorizonData) -> list[dict[str, object]]:
	'''Return deterministic statistics for every physical IL and XL section.'''
	rows: list[dict[str, object]] = []
	for orientation, values in (
		('inline', data.inline_values),
		('crossline', data.crossline_values),
	):
		for index, physical in enumerate(values):
			if orientation == 'inline':
				valid = data.valid_trace_mask[index, :]
				common = data.common_bound_mask[index, :]
				ordered = data.sample_strict_order_mask[index, :]
				horizon = data.bound_valid_mask[:, index, :]
			else:
				valid = data.valid_trace_mask[:, index]
				common = data.common_bound_mask[:, index]
				ordered = data.sample_strict_order_mask[:, index]
				horizon = data.bound_valid_mask[:, :, index]
			counts = tuple(int(np.count_nonzero(item)) for item in horizon)
			row: dict[str, object] = {
				'orientation': orientation,
				'physical_line_number': int(physical),
				'array_index': index,
				'valid_trace_count': int(np.count_nonzero(valid)),
				'five_horizon_common_count': int(np.count_nonzero(common)),
				'sample_strict_order_count': int(np.count_nonzero(ordered)),
			}
			row.update(
				{
					f'{name}_count': count
					for name, count in zip(HORIZON_NAMES, counts, strict=True)
				}
			)
			row['minimum_horizon_count'] = min(counts)
			row['total_horizon_observation_count'] = sum(counts)
			rows.append(row)
	return rows


def write_section_statistics_csv(
	data: VolveHorizonData, output_path: str | Path
) -> int:
	'''Write section statistics as a stable CSV artifact.'''
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	rows = section_statistics(data)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=SECTION_STATISTICS_FIELDS)
		writer.writeheader()
		writer.writerows(rows)
	return len(rows)


def array_sha256(array: np.ndarray) -> str:
	'''Hash dtype, shape, and C-order values using the binding-v2 convention.'''
	contiguous = np.ascontiguousarray(array)
	digest = hashlib.sha256()
	digest.update(str(contiguous.dtype).encode('ascii'))
	digest.update(json.dumps(list(contiguous.shape)).encode('ascii'))
	digest.update(contiguous.view(np.uint8))
	return digest.hexdigest()


def _validate_summaries(  # noqa: C901
	paths: VolveHorizonPaths,
	horizon: Mapping[str, object],
	grid: Mapping[str, object],
	manual: Mapping[str, object],
) -> None:
	if grid.get('schema_version') != BINDING_SCHEMA_VERSION:
		raise ValueError('binding schema_version must be 2')
	if grid.get('status') != 'PASS':
		raise ValueError('binding status must be PASS')
	if manual.get('status') != 'PASS':
		raise ValueError('manual review status must be PASS')
	if manual.get('horizon_visual_qc') != 'PASS':
		raise ValueError('horizon visual QC must be PASS')
	if manual.get('fault_visual_qc') != 'PASS':
		raise ValueError('fault visual QC must be PASS')
	if manual.get('source_binding_summary_sha256') != _file_sha256(
		paths.grid_summary
	):
		raise ValueError('manual review binding-summary SHA-256 mismatch')
	if grid.get('horizons') != horizon:
		raise ValueError('standalone horizon summary differs from grid summary')
	if horizon.get('horizon_order') != list(HORIZON_NAMES):
		raise ValueError(f'horizon order must be {HORIZON_NAMES!r}')
	if horizon.get('artifact_sha256') != _file_sha256(paths.binding_npz):
		raise ValueError('binding NPZ SHA-256 differs from horizon summary')
	for label, checks in (
		('trace grid', _nested_mapping(grid, 'acceptance_checks', 'trace_grid')),
		('horizons', _nested_mapping(grid, 'acceptance_checks', 'horizons')),
		('faults', _nested_mapping(grid, 'acceptance_checks', 'faults')),
	):
		if not checks or any(value is not True for value in checks.values()):
			raise ValueError(f'{label} binding acceptance checks must all pass')


def _validate_canonical_arrays(  # noqa: PLR0913, PLR0917
	inline: np.ndarray,
	crossline: np.ndarray,
	time_ms: np.ndarray,
	valid: np.ndarray,
	geometry: VolveHorizonGeometry,
	grid_summary: Mapping[str, object],
) -> None:
	x_size, y_size, z_size = geometry.shape_xyz
	if inline.shape != (x_size,) or not np.issubdtype(inline.dtype, np.number):
		raise TypeError('inline_values must be a numeric 1D canonical axis')
	if crossline.shape != (y_size,) or not np.issubdtype(
		crossline.dtype, np.number
	):
		raise TypeError('crossline_values must be a numeric 1D canonical axis')
	expected_inline = np.arange(geometry.inline_min, geometry.inline_max + 1)
	expected_crossline = np.arange(
		geometry.crossline_min, geometry.crossline_max + 1
	)
	if not np.array_equal(inline, expected_inline):
		raise ValueError('inline_values do not exactly match physical geometry')
	if not np.array_equal(crossline, expected_crossline):
		raise ValueError('crossline_values do not exactly match physical geometry')
	expected_time = geometry.first_twt_ms + geometry.sample_interval_ms * np.arange(
		z_size
	)
	if time_ms.shape != (z_size,) or not np.array_equal(time_ms, expected_time):
		raise ValueError('time_ms does not exactly match canonical sampling')
	if valid.shape != (x_size, y_size) or valid.dtype != np.bool_:
		raise TypeError('valid_trace_mask must be a bool canonical XY array')
	trace_grid = _required_mapping(grid_summary, 'trace_grid', 'grid summary')
	if trace_grid.get('valid_trace_mask_sha256') != array_sha256(valid):
		raise ValueError('canonical valid_trace_mask differs from binding summary')


def _validate_binding_arrays(  # noqa: C901, PLR0912, PLR0915
	arrays: Mapping[str, np.ndarray],
	geometry: VolveHorizonGeometry,
	summary: Mapping[str, object],
	valid_trace_mask: np.ndarray,
) -> None:
	shape_xy = geometry.shape_xyz[:2]
	shape_hxy = (len(HORIZON_NAMES), *shape_xy)
	names = arrays['horizon_names']
	if names.shape != (len(HORIZON_NAMES),) or names.dtype.kind != 'U':
		raise TypeError('horizon_names must be a 5-element Unicode array')
	if tuple(str(value) for value in names) != HORIZON_NAMES:
		raise ValueError(f'horizon order must be {HORIZON_NAMES!r}')
	for name in ('twt_ms', 'sample_float', 'xy_residual_m'):
		array = arrays[name]
		if array.shape != shape_hxy or array.dtype != np.float32:
			raise TypeError(f'{name} must be float32 with shape {shape_hxy!r}')
	if arrays['sample_index'].shape != shape_hxy or arrays[
		'sample_index'
	].dtype != np.int16:
		raise TypeError(f'sample_index must be int16 with shape {shape_hxy!r}')
	for name in (
		'source_present_mask',
		'bound_valid_mask',
	):
		array = arrays[name]
		if array.shape != shape_hxy or array.dtype != np.bool_:
			raise TypeError(f'{name} must be bool with shape {shape_hxy!r}')
	for name in (
		'common_bound_mask',
		'continuous_strict_order_mask',
		'sample_strict_order_mask',
	):
		array = arrays[name]
		if array.shape != shape_xy or array.dtype != np.bool_:
			raise TypeError(f'{name} must be bool with shape {shape_xy!r}')
	bound = arrays['bound_valid_mask']
	if np.any(bound & ~arrays['source_present_mask']):
		raise ValueError('bound_valid_mask must be within source_present_mask')
	if np.any(bound & ~valid_trace_mask[np.newaxis, :, :]):
		raise ValueError('bound_valid_mask must be within valid_trace_mask')
	if not np.array_equal(arrays['common_bound_mask'], np.all(bound, axis=0)):
		raise ValueError('common_bound_mask must equal all five bound masks')
	continuous_order = arrays['common_bound_mask'] & np.all(
		np.diff(arrays['twt_ms'], axis=0) > 0,
		axis=0,
	)
	if not np.array_equal(
		arrays['continuous_strict_order_mask'], continuous_order
	):
		raise ValueError('continuous_strict_order_mask differs from native TWT order')
	sample_order = arrays['common_bound_mask'] & np.all(
		np.diff(arrays['sample_index'].astype(np.int32), axis=0) > 0,
		axis=0,
	)
	if not np.array_equal(arrays['sample_strict_order_mask'], sample_order):
		raise ValueError('sample_strict_order_mask differs from native sample order')
	for name in ('twt_ms', 'sample_float', 'xy_residual_m'):
		array = arrays[name]
		if not np.all(np.isfinite(array[bound])) or not np.all(np.isnan(array[~bound])):
			raise ValueError(f'{name} finite/NaN values disagree with bound mask')
	index = arrays['sample_index']
	if np.any(index[bound] < 0) or np.any(index[bound] >= geometry.shape_xyz[2]):
		raise ValueError('bound sample_index is outside the canonical time axis')
	if np.any(index[~bound] != -1):
		raise ValueError('unbound sample_index values must be -1')
	common_count = int(np.count_nonzero(arrays['common_bound_mask']))
	if summary.get('common_bound_count') != common_count:
		raise ValueError('common-bound count differs from horizon summary')
	if summary.get('sample_strict_order_count') != int(
		np.count_nonzero(arrays['sample_strict_order_mask'])
	):
		raise ValueError('sample-order count differs from horizon summary')
	if summary.get('continuous_strict_order_count') != int(
		np.count_nonzero(arrays['continuous_strict_order_mask'])
	):
		raise ValueError('continuous-order count differs from horizon summary')
	for name in ('common_bound_mask', 'sample_strict_order_mask'):
		if summary.get(f'{name}_sha256') != array_sha256(arrays[name]):
			raise ValueError(f'{name} SHA-256 differs from horizon summary')
	per_horizon = _required_mapping(summary, 'per_horizon', 'horizon summary')
	for horizon_index, horizon_name in enumerate(HORIZON_NAMES):
		record = _required_mapping(per_horizon, horizon_name, 'per_horizon')
		count = int(np.count_nonzero(bound[horizon_index]))
		if record.get('bound_valid_count') != count:
			raise ValueError(f'{horizon_name} count differs from horizon summary')


def _read_json(path: Path, label: str) -> Mapping[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'invalid {label} JSON: {path}') from exc
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a JSON object')
	return value


def _required_mapping(
	value: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{label}.{key} must be a mapping')
	return child


def _nested_mapping(
	value: Mapping[str, object], parent: str, child: str
) -> Mapping[str, object]:
	parent_value = _required_mapping(value, parent, 'grid summary')
	return _required_mapping(parent_value, child, parent)


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file_obj:
		for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	return Path(value)


def _absolute_path(value: object, label: str) -> Path:
	path = _path(value, label)
	if not path.is_absolute():
		raise ValueError(f'{label} must be absolute')
	return path.resolve()


def _artifact_output(root: Path, value: object, label: str) -> Path:
	relative = _path(value, label)
	if relative.is_absolute() or '..' in relative.parts:
		raise ValueError(f'{label} must be a safe artifact-root-relative path')
	return (root / relative).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.relative_to(root)
	except ValueError:
		return False
	return True


__all__ = [
	'BINDING_SCHEMA_VERSION',
	'HORIZON_NAMES',
	'SECTION_STATISTICS_FIELDS',
	'VOLVE_HORIZON_GEOMETRY',
	'VolveHorizonData',
	'VolveHorizonGeometry',
	'VolveHorizonInspectionConfig',
	'VolveHorizonPaths',
	'array_sha256',
	'load_volve_horizon_data',
	'resolve_volve_horizon_inspection_config',
	'resolve_volve_horizon_paths',
	'section_statistics',
	'write_section_statistics_csv',
]
