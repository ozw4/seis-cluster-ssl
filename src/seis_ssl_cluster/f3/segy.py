"""SEGY geometry and statistics inspection for the F3 facies benchmark."""

from __future__ import annotations

import csv
import importlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.f3.inspection import find_class_info_file
from seis_ssl_cluster.f3.labels import F3ClassInfo, read_class_info

if TYPE_CHECKING:
	from types import ModuleType

AXIS_ASSUMPTION = {
	'cube_axis_0': {
		'repo_axis': 'x',
		'domain_axis': 'inline',
		'description': 'segyio.tools.cube axis 0 -> x / inline',
	},
	'cube_axis_1': {
		'repo_axis': 'y',
		'domain_axis': 'crossline',
		'description': 'segyio.tools.cube axis 1 -> y / crossline',
	},
	'cube_axis_2': {
		'repo_axis': 'z',
		'domain_axis': 'sample/time',
		'description': 'segyio.tools.cube axis 2 -> z / sample/time',
	},
}

GEOMETRY_CSV_FIELDNAMES = (
	'role',
	'path',
	'file_size',
	'iline_count',
	'xline_count',
	'sample_count',
	'iline_min',
	'iline_max',
	'xline_min',
	'xline_max',
	'sample_min',
	'sample_max',
	'cube_shape',
	'dtype',
)

_SEGY_SUFFIXES = frozenset({'.sgy', '.segy'})
_PERCENTILE_SPECS = (
	('p0.1', 0.1),
	('p1', 1.0),
	('p5', 5.0),
	('p50', 50.0),
	('p95', 95.0),
	('p99', 99.0),
	('p99.9', 99.9),
)


@dataclass(frozen=True)
class F3SegyPaths:
	"""Resolved seismic and label SEGY paths under an F3 root."""

	seismic: Path
	label: Path


@dataclass(frozen=True)
class F3SegyGeometry:
	"""Geometry metadata for one structured F3 SEGY cube."""

	role: str
	path: Path
	file_size: int
	iline_count: int
	xline_count: int
	sample_count: int
	iline_min: int | None
	iline_max: int | None
	xline_min: int | None
	xline_max: int | None
	sample_min: int | float | None
	sample_max: int | float | None
	cube_shape: tuple[int, ...]
	dtype: str

	def to_dict(self) -> dict[str, object]:
		"""Return a JSON-serializable geometry record."""
		return {
			'role': self.role,
			'path': str(self.path),
			'file_size': self.file_size,
			'iline_count': self.iline_count,
			'xline_count': self.xline_count,
			'sample_count': self.sample_count,
			'iline_min': self.iline_min,
			'iline_max': self.iline_max,
			'xline_min': self.xline_min,
			'xline_max': self.xline_max,
			'sample_min': self.sample_min,
			'sample_max': self.sample_max,
			'cube_shape': list(self.cube_shape),
			'dtype': self.dtype,
		}


@dataclass(frozen=True)
class F3SegyFileInspection:
	"""Geometry and loaded cube for one F3 SEGY role."""

	geometry: F3SegyGeometry
	cube: np.ndarray


@dataclass(frozen=True)
class F3SegyInspection:
	"""Complete F3 SEGY inspection result."""

	f3_root: Path
	class_info_path: Path
	classes: tuple[F3ClassInfo, ...]
	seismic: F3SegyFileInspection
	label: F3SegyFileInspection
	seismic_amplitude_stats: dict[str, object]
	label_unique_values: dict[str, object]


@dataclass(frozen=True)
class F3SegyInspectionOutputConfig:
	"""Destination paths for F3 SEGY inspection artifacts."""

	metadata_json: Path
	summary_markdown: Path
	seismic_amplitude_stats_json: Path
	label_unique_values_json: Path
	geometry_json: Path
	geometry_csv: Path


def axis_assumption_metadata() -> dict[str, object]:
	"""Return the repo XYZ mapping assumed for `segyio.tools.cube` output."""
	return {
		'description': (
			'segyio.tools.cube() のshapeは '
			'(inline, crossline, sample/time) として扱う。'
		),
		'cube_to_repo_axes': AXIS_ASSUMPTION,
	}


def calculate_seismic_amplitude_stats(values: np.ndarray) -> dict[str, object]:
	"""Calculate finite-value amplitude statistics for a seismic cube."""
	array = _as_numeric_array(values, label='seismic amplitudes')
	total_count = int(array.size)
	finite_mask = np.isfinite(array)
	finite_values = array[finite_mask].astype(np.float64, copy=False)
	finite_count = int(finite_values.size)
	nonfinite_count = total_count - finite_count
	zero_count = int(np.count_nonzero(finite_values == 0.0))

	stats: dict[str, object] = {
		'finite_count': finite_count,
		'nonfinite_count': nonfinite_count,
		'zero_count': zero_count,
	}
	if finite_count == 0:
		stats.update(
			{
				'min': None,
				'p0.1': None,
				'p1': None,
				'p5': None,
				'p50': None,
				'p95': None,
				'p99': None,
				'p99.9': None,
				'max': None,
				'mean': None,
				'std': None,
			},
		)
		return stats

	percentiles = np.percentile(
		finite_values,
		[percentile for _, percentile in _PERCENTILE_SPECS],
	)
	stats['min'] = float(np.min(finite_values))
	for (key, _), value in zip(_PERCENTILE_SPECS, percentiles, strict=True):
		stats[key] = float(value)
	stats['max'] = float(np.max(finite_values))
	stats['mean'] = float(np.mean(finite_values))
	stats['std'] = float(np.std(finite_values))
	return stats


def calculate_label_unique_values(
	values: np.ndarray,
	classes: Sequence[F3ClassInfo],
) -> dict[str, object]:
	"""Calculate label unique counts and compare them with class information."""
	array = _as_numeric_array(values, label='label values')
	flat_values = array.ravel()
	finite_mask = np.isfinite(flat_values)
	finite_values = flat_values[finite_mask]
	nonfinite_count = int(flat_values.size - finite_values.size)
	integer_like = _is_integer_like(finite_values, nonfinite_count=nonfinite_count)
	normalized = _normalize_label_values(finite_values, integer_like=integer_like)
	counts = Counter(normalized)
	unique_values = sorted(counts)
	class_ids = {item.class_id for item in classes}
	missing_class_ids = sorted(
		class_id for class_id in class_ids if class_id not in counts
	)
	unexpected_values = [
		value for value in unique_values if value not in class_ids
	]

	return {
		'unique_values': unique_values,
		'counts_by_value': {str(value): int(count) for value, count in counts.items()},
		'min': unique_values[0] if unique_values else None,
		'max': unique_values[-1] if unique_values else None,
		'integer_like': integer_like,
		'nonfinite_count': nonfinite_count,
		'class_info': {
			'class_count': len(classes),
			'classes': [
				{
					'class_id': item.class_id,
					'class_name': item.class_name,
					'rgb': list(item.rgb),
					'hex_color': item.hex_color,
					'present_in_label': item.class_id in counts,
					'count': int(counts.get(item.class_id, 0)),
				}
				for item in classes
			],
			'missing_class_ids': missing_class_ids,
		},
		'unexpected_label_values': unexpected_values,
	}


def find_f3_segy_paths(
	f3_root: str | Path,
	*,
	candidate_extensions: Sequence[str] = ('.segy', '.sgy'),
) -> F3SegyPaths:
	"""Find exactly one seismic and one label SEGY file under an F3 root."""
	root = Path(f3_root)
	if not root.is_dir():
		msg = f'F3 root directory does not exist: {root}'
		raise FileNotFoundError(msg)

	suffixes = _normalize_segy_suffixes(candidate_extensions)
	candidates = sorted(
		(
			path
			for path in root.rglob('*')
			if path.is_file() and path.suffix.lower() in suffixes
		),
		key=lambda path: path.relative_to(root).as_posix().lower(),
	)
	seismic = _select_single_segy_path(
		root,
		candidates,
		role='seismic',
		name_fragment='seismic',
	)
	label = _select_single_segy_path(
		root,
		candidates,
		role='label',
		name_fragment='label',
	)
	return F3SegyPaths(seismic=seismic, label=label)


def read_f3_segy_file(path: str | Path, *, role: str) -> F3SegyFileInspection:
	"""Read one F3 SEGY file into a cube and collect geometry metadata."""
	segy_path = Path(path)
	if not segy_path.is_file():
		msg = f'missing F3 {role} SEGY file: {segy_path}'
		raise FileNotFoundError(msg)

	segyio = _import_segyio()
	with segyio.open(str(segy_path), 'r') as segy_file:
		ilines = np.asarray(segy_file.ilines)
		xlines = np.asarray(segy_file.xlines)
		samples = np.asarray(segy_file.samples)
		cube = np.asarray(segyio.tools.cube(segy_file))

	geometry = F3SegyGeometry(
		role=role,
		path=segy_path.resolve(strict=False),
		file_size=segy_path.stat().st_size,
		iline_count=int(ilines.size),
		xline_count=int(xlines.size),
		sample_count=int(samples.size),
		iline_min=_int_min(ilines),
		iline_max=_int_max(ilines),
		xline_min=_int_min(xlines),
		xline_max=_int_max(xlines),
		sample_min=_numeric_min(samples),
		sample_max=_numeric_max(samples),
		cube_shape=tuple(int(axis) for axis in cube.shape),
		dtype=str(cube.dtype),
	)
	return F3SegyFileInspection(geometry=geometry, cube=cube)


def inspect_f3_segy_files(
	f3_root: str | Path,
	*,
	candidate_extensions: Sequence[str] = ('.segy', '.sgy'),
) -> F3SegyInspection:
	"""Inspect F3 seismic and label SEGY files plus class-info correspondence."""
	root = Path(f3_root)
	paths = find_f3_segy_paths(root, candidate_extensions=candidate_extensions)
	class_info_path, _warnings = find_class_info_file(root)
	classes = read_class_info(class_info_path)
	seismic = read_f3_segy_file(paths.seismic, role='seismic')
	label = read_f3_segy_file(paths.label, role='label')
	return F3SegyInspection(
		f3_root=root.resolve(strict=False),
		class_info_path=class_info_path.resolve(strict=False),
		classes=classes,
		seismic=seismic,
		label=label,
		seismic_amplitude_stats=calculate_seismic_amplitude_stats(seismic.cube),
		label_unique_values=calculate_label_unique_values(label.cube, classes),
	)


def segy_inspection_metadata_to_dict(
	inspection: F3SegyInspection,
) -> dict[str, object]:
	"""Return machine-readable F3 SEGY geometry metadata."""
	return {
		'f3_root': str(inspection.f3_root),
		'class_info_path': str(inspection.class_info_path),
		'axis_assumption': axis_assumption_metadata(),
		'shape_consistency': {
			'seismic_cube_shape': list(inspection.seismic.geometry.cube_shape),
			'label_cube_shape': list(inspection.label.geometry.cube_shape),
			'matches': (
				inspection.seismic.geometry.cube_shape
				== inspection.label.geometry.cube_shape
			),
		},
		'segy_files': {
			'seismic': inspection.seismic.geometry.to_dict(),
			'label': inspection.label.geometry.to_dict(),
		},
	}


def seismic_amplitude_stats_to_dict(
	inspection: F3SegyInspection,
) -> dict[str, object]:
	"""Return the seismic amplitude statistics artifact payload."""
	return {
		'source_path': str(inspection.seismic.geometry.path),
		'axis_assumption': axis_assumption_metadata(),
		'stats': inspection.seismic_amplitude_stats,
	}


def label_unique_values_to_dict(
	inspection: F3SegyInspection,
) -> dict[str, object]:
	"""Return the label unique-values artifact payload."""
	return {
		'source_path': str(inspection.label.geometry.path),
		'class_info_path': str(inspection.class_info_path),
		'axis_assumption': axis_assumption_metadata(),
		'stats': inspection.label_unique_values,
	}


def render_f3_segy_summary_markdown(inspection: F3SegyInspection) -> str:
	"""Render a Japanese Markdown summary for F3 SEGY inspection."""
	metadata = segy_inspection_metadata_to_dict(inspection)
	seismic_stats = inspection.seismic_amplitude_stats
	label_stats = inspection.label_unique_values
	lines = [
		'# F3 SEGY geometry inspection',
		'',
		f'- F3 root: `{inspection.f3_root}`',
		f'- class_info: `{inspection.class_info_path}`',
		'- XYZ仮定: cube axis 0 -> x / inline, '
		'axis 1 -> y / crossline, axis 2 -> z / sample/time',
		'- label値0は有効classとして扱う。',
		'',
		'## Geometry',
		'',
		'| role | cube_shape | iline | xline | sample/time | dtype |',
		'|---|---|---|---|---|---|',
		_render_geometry_row(inspection.seismic.geometry),
		_render_geometry_row(inspection.label.geometry),
		'',
		'## Shape対応',
		'',
		f'- seismic shape: {metadata["shape_consistency"]["seismic_cube_shape"]}',
		f'- label shape: {metadata["shape_consistency"]["label_cube_shape"]}',
		f'- 一致: {metadata["shape_consistency"]["matches"]}',
		'',
		'## Seismic amplitude統計',
		'',
		(
			f'- finite_count: {seismic_stats["finite_count"]}, '
			f'nonfinite_count: {seismic_stats["nonfinite_count"]}, '
			f'zero_count: {seismic_stats["zero_count"]}'
		),
		(
			f'- min/p50/max: {seismic_stats["min"]} / '
			f'{seismic_stats["p50"]} / {seismic_stats["max"]}'
		),
		(
			f'- p1/p99: {seismic_stats["p1"]} / '
			f'{seismic_stats["p99"]}'
		),
		(
			f'- mean/std: {seismic_stats["mean"]} / '
			f'{seismic_stats["std"]}'
		),
		'',
		'## Label unique値',
		'',
		f'- integer-like: {label_stats["integer_like"]}',
		f'- unique values: {label_stats["unique_values"]}',
		f'- unexpected label values: {label_stats["unexpected_label_values"]}',
		'',
		'| class_id | class_name | present | count | color |',
		'|---:|---|---|---:|---|',
	]
	lines.extend(_render_class_rows(label_stats))
	lines.append('')
	return '\n'.join(lines)


def write_f3_segy_inspection_outputs(
	inspection: F3SegyInspection,
	outputs: F3SegyInspectionOutputConfig,
) -> None:
	"""Write F3 SEGY geometry, statistics, and summary artifacts."""
	metadata = segy_inspection_metadata_to_dict(inspection)
	_write_json(outputs.metadata_json, metadata)
	_write_json(outputs.geometry_json, metadata)
	_write_geometry_csv(
		outputs.geometry_csv,
		(inspection.seismic.geometry, inspection.label.geometry),
	)
	_write_json(
		outputs.seismic_amplitude_stats_json,
		seismic_amplitude_stats_to_dict(inspection),
	)
	_write_json(
		outputs.label_unique_values_json,
		label_unique_values_to_dict(inspection),
	)
	_write_text(
		outputs.summary_markdown,
		render_f3_segy_summary_markdown(inspection),
	)


def _as_numeric_array(values: np.ndarray, *, label: str) -> np.ndarray:
	array = np.asarray(values)
	if not np.issubdtype(array.dtype, np.number):
		msg = f'{label} must be numeric; got dtype {array.dtype}'
		raise TypeError(msg)
	return array


def _is_integer_like(
	finite_values: np.ndarray,
	*,
	nonfinite_count: int,
) -> bool:
	if nonfinite_count > 0:
		return False
	if finite_values.size == 0:
		return True
	if np.issubdtype(finite_values.dtype, np.integer):
		return True
	return bool(np.all(np.equal(finite_values, np.round(finite_values))))


def _normalize_label_values(
	values: np.ndarray,
	*,
	integer_like: bool,
) -> list[int | float]:
	if integer_like:
		return [int(value) for value in values]
	return [float(value) for value in values]


def _normalize_segy_suffixes(candidate_extensions: Sequence[str]) -> frozenset[str]:
	if not candidate_extensions:
		msg = 'candidate_extensions must contain at least one suffix'
		raise ValueError(msg)
	suffixes = []
	for suffix in candidate_extensions:
		if not isinstance(suffix, str):
			msg = f'candidate_extensions must be strings; got {suffix!r}'
			raise TypeError(msg)
		normalized = suffix.lower()
		if not normalized.startswith('.'):
			msg = f'SEGY extension must start with ".": {suffix!r}'
			raise ValueError(msg)
		suffixes.append(normalized)
	return frozenset(suffixes)


def _select_single_segy_path(
	root: Path,
	candidates: Sequence[Path],
	*,
	role: str,
	name_fragment: str,
) -> Path:
	matches = [
		path
		for path in candidates
		if name_fragment in path.stem.lower()
	]
	if not matches:
		msg = f'missing F3 {role} SEGY file under {root}'
		raise FileNotFoundError(msg)
	if len(matches) > 1:
		relative_paths = [
			path.relative_to(root).as_posix()
			for path in matches
		]
		msg = f'multiple F3 {role} SEGY files found: {relative_paths!r}'
		raise ValueError(msg)
	return matches[0]


def _import_segyio() -> ModuleType:
	try:
		module = importlib.import_module('segyio')
	except ModuleNotFoundError as exc:
		msg = (
			'segyio is required to inspect F3 SEGY files; '
			'install the segy extra.'
		)
		raise ModuleNotFoundError(msg) from exc
	return module


def _int_min(values: np.ndarray) -> int | None:
	if values.size == 0:
		return None
	return int(np.min(values))


def _int_max(values: np.ndarray) -> int | None:
	if values.size == 0:
		return None
	return int(np.max(values))


def _numeric_min(values: np.ndarray) -> int | float | None:
	if values.size == 0:
		return None
	return _json_numeric(np.min(values))


def _numeric_max(values: np.ndarray) -> int | float | None:
	if values.size == 0:
		return None
	return _json_numeric(np.max(values))


def _json_numeric(value: object) -> int | float:
	if isinstance(value, np.integer):
		return int(value)
	if isinstance(value, np.floating):
		as_float = float(value)
		if as_float.is_integer():
			return int(as_float)
		return as_float
	if isinstance(value, int | float):
		if isinstance(value, float) and value.is_integer():
			return int(value)
		return value
	msg = f'expected numeric value; got {value!r}'
	raise TypeError(msg)


def _render_geometry_row(geometry: F3SegyGeometry) -> str:
	return (
		f'| {geometry.role} | {list(geometry.cube_shape)} | '
		f'{geometry.iline_min}-{geometry.iline_max} '
		f'({geometry.iline_count}) | '
		f'{geometry.xline_min}-{geometry.xline_max} '
		f'({geometry.xline_count}) | '
		f'{geometry.sample_min}-{geometry.sample_max} '
		f'({geometry.sample_count}) | {geometry.dtype} |'
	)


def _render_class_rows(label_stats: Mapping[str, object]) -> list[str]:
	class_info = label_stats['class_info']
	if not isinstance(class_info, Mapping):
		msg = 'label_stats.class_info must be a mapping'
		raise TypeError(msg)
	classes = class_info['classes']
	if not isinstance(classes, Sequence):
		msg = 'label_stats.class_info.classes must be a sequence'
		raise TypeError(msg)
	lines: list[str] = []
	for item in classes:
		if not isinstance(item, Mapping):
			msg = 'label_stats.class_info.classes entries must be mappings'
			raise TypeError(msg)
		lines.append(
			'| '
			f'{item["class_id"]} | {item["class_name"]} | '
			f'{item["present_in_label"]} | {item["count"]} | '
			f'{item["hex_color"]} |',
		)
	return lines


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _write_text(path: str | Path, content: str) -> None:
	text_path = Path(path)
	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text(content, encoding='utf-8')


def _write_geometry_csv(
	path: str | Path,
	geometries: Sequence[F3SegyGeometry],
) -> None:
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=GEOMETRY_CSV_FIELDNAMES)
		writer.writeheader()
		for geometry in geometries:
			row = geometry.to_dict()
			row['cube_shape'] = 'x'.join(str(axis) for axis in geometry.cube_shape)
			writer.writerow(row)


__all__ = [
	'AXIS_ASSUMPTION',
	'F3SegyFileInspection',
	'F3SegyGeometry',
	'F3SegyInspection',
	'F3SegyInspectionOutputConfig',
	'F3SegyPaths',
	'axis_assumption_metadata',
	'calculate_label_unique_values',
	'calculate_seismic_amplitude_stats',
	'find_f3_segy_paths',
	'inspect_f3_segy_files',
	'label_unique_values_to_dict',
	'read_f3_segy_file',
	'render_f3_segy_summary_markdown',
	'segy_inspection_metadata_to_dict',
	'seismic_amplitude_stats_to_dict',
	'write_f3_segy_inspection_outputs',
]
