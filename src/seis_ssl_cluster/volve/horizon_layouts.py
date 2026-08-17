'''Explicit physical-section layouts and split plans for Volve horizons.'''

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_data import (
	HORIZON_NAMES,
	VolveHorizonData,
	array_sha256,
)

LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZE_PREFIX = {'small': 1, 'medium': 2, 'large': 4}
SELECTION_SEMANTICS = 'explicit_section_prefix_all_available_horizon_points_v1'
TWT_WINDOW_MARGIN_SAMPLES = 16
TWT_WINDOW_GRID_SAMPLES = 8


@dataclass(frozen=True)
class PhysicalSectionLines:
	'''Ordered physical inline and crossline numbers.'''

	inline: tuple[int, ...]
	crossline: tuple[int, ...]


@dataclass(frozen=True)
class IndexedSectionLines:
	'''Ordered canonical inline and crossline array indices.'''

	inline: tuple[int, ...]
	crossline: tuple[int, ...]


@dataclass(frozen=True)
class VolveHorizonLayouts:
	'''Five explicit candidate layouts and one fixed validation pair.'''

	semantics: str
	validation: PhysicalSectionLines
	layouts: Mapping[str, PhysicalSectionLines]
	config_path: Path
	config_sha256: str


@dataclass(frozen=True)
class HorizonTwtWindow:
	'''Fixed sample window containing native bound horizons plus margin.'''

	start_index: int
	stop_index_exclusive: int
	length_samples: int
	margin_samples: int = TWT_WINDOW_MARGIN_SAMPLES
	grid_samples: int = TWT_WINDOW_GRID_SAMPLES


@dataclass(frozen=True)
class HorizonSplitPlan:
	'''One reusable layout/size split and its scientific identity.'''

	layout_id: str
	data_size: str
	selected_physical_lines: PhysicalSectionLines
	selected_indices: IndexedSectionLines
	validation_physical_lines: PhysicalSectionLines
	validation_indices: IndexedSectionLines
	reserved_large_physical_lines: PhysicalSectionLines
	reserved_large_indices: IndexedSectionLines
	train_mask: np.ndarray
	validation_mask: np.ndarray
	test_primary_mask: np.ndarray
	test_per_horizon_mask: np.ndarray
	twt_window: HorizonTwtWindow
	_identity: Mapping[str, object]

	@property
	def scientific_identity_sha256(self) -> str:
		'''Return the canonical JSON hash of this plan identity.'''
		return _json_sha256(self._identity)

	def identity(self) -> dict[str, object]:
		'''Return a JSON-safe identity shared by frozen and end-to-end runs.'''
		return {
			**self._identity,
			'scientific_identity_sha256': self.scientific_identity_sha256,
		}


def load_volve_horizon_layouts(
	path: str | Path, data: VolveHorizonData
) -> VolveHorizonLayouts:
	'''Load five physical layouts and validate them against canonical axes.'''
	config_path = Path(path).resolve()
	raw = load_config(config_path)
	if set(raw) != {'selection', 'validation', 'layouts'}:
		raise ValueError(
			'layout config must contain exactly selection, validation, and layouts'
		)
	selection = _mapping(raw, 'selection', 'layout config')
	if set(selection) != {'semantics'}:
		raise ValueError('selection must contain exactly semantics')
	if selection.get('semantics') != SELECTION_SEMANTICS:
		raise ValueError(f'selection.semantics must be {SELECTION_SEMANTICS!r}')
	validation = _physical_lines(
		_mapping(raw, 'validation', 'layout config'),
		label='validation',
		expected=1,
	)
	raw_layouts = _mapping(raw, 'layouts', 'layout config')
	if set(raw_layouts) != set(LAYOUT_IDS):
		raise ValueError(f'layouts must contain exactly {LAYOUT_IDS!r}')
	layouts = {
		layout_id: _physical_lines(
			_mapping(raw_layouts, layout_id, 'layouts'),
			label=layout_id,
			expected=4,
		)
		for layout_id in LAYOUT_IDS
	}
	_validate_physical_lines(validation, layouts, data)
	return VolveHorizonLayouts(
		semantics=SELECTION_SEMANTICS,
		validation=validation,
		layouts=layouts,
		config_path=config_path,
		config_sha256=_file_sha256(config_path),
	)


def selected_training_lines(
	layouts: VolveHorizonLayouts, layout_id: str, data_size: str
) -> PhysicalSectionLines:
	'''Return the nested physical line prefix for one condition.'''
	if layout_id not in layouts.layouts:
		raise ValueError(f'unknown layout_id: {layout_id!r}')
	try:
		prefix = DATA_SIZE_PREFIX[data_size]
	except KeyError as exc:
		raise ValueError(f'unknown data_size: {data_size!r}') from exc
	lines = layouts.layouts[layout_id]
	return PhysicalSectionLines(
		inline=lines.inline[:prefix],
		crossline=lines.crossline[:prefix],
	)


def build_horizon_split_plan(
	data: VolveHorizonData,
	layouts: VolveHorizonLayouts,
	layout_id: str,
	data_size: str,
) -> HorizonSplitPlan:
	'''Build one all-available-points train/validation/common-test plan.'''
	selected_physical = selected_training_lines(layouts, layout_id, data_size)
	selected_indices = _to_indices(selected_physical, data)
	validation_indices = _to_indices(layouts.validation, data)
	reserved_physical = reserved_large_lines(layouts)
	reserved_indices = _to_indices(reserved_physical, data)

	selected_lateral = _section_mask(data.shape_xy, selected_indices)
	validation_lateral = _section_mask(data.shape_xy, validation_indices)
	reserved_lateral = _section_mask(data.shape_xy, reserved_indices)
	test_lateral = (
		data.valid_trace_mask & ~validation_lateral & ~reserved_lateral
	)
	train_mask = (
		data.bound_valid_mask
		& selected_lateral[np.newaxis, :, :]
		& ~validation_lateral[np.newaxis, :, :]
	)
	validation_mask = (
		data.bound_valid_mask & validation_lateral[np.newaxis, :, :]
	)
	test_primary_mask = data.common_bound_mask & test_lateral
	test_per_horizon_mask = (
		data.bound_valid_mask & test_lateral[np.newaxis, :, :]
	)
	window = compute_horizon_twt_window(data)
	identity = _plan_identity(
		data=data,
		layouts=layouts,
		layout_id=layout_id,
		data_size=data_size,
		selected_physical=selected_physical,
		selected_indices=selected_indices,
		validation_indices=validation_indices,
		reserved_physical=reserved_physical,
		reserved_indices=reserved_indices,
		train_mask=train_mask,
		validation_mask=validation_mask,
		test_primary_mask=test_primary_mask,
		test_per_horizon_mask=test_per_horizon_mask,
		window=window,
	)
	return HorizonSplitPlan(
		layout_id=layout_id,
		data_size=data_size,
		selected_physical_lines=selected_physical,
		selected_indices=selected_indices,
		validation_physical_lines=layouts.validation,
		validation_indices=validation_indices,
		reserved_large_physical_lines=reserved_physical,
		reserved_large_indices=reserved_indices,
		train_mask=train_mask,
		validation_mask=validation_mask,
		test_primary_mask=test_primary_mask,
		test_per_horizon_mask=test_per_horizon_mask,
		twt_window=window,
		_identity=identity,
	)


def build_all_horizon_split_plans(
	data: VolveHorizonData, layouts: VolveHorizonLayouts
) -> tuple[HorizonSplitPlan, ...]:
	'''Build the paired 5-layout by 3-size plan suite.'''
	return tuple(
		build_horizon_split_plan(data, layouts, layout_id, data_size)
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZE_PREFIX
	)


def reserved_large_lines(layouts: VolveHorizonLayouts) -> PhysicalSectionLines:
	'''Return the ordered union of every large candidate section.'''
	return PhysicalSectionLines(
		inline=_ordered_union(
			lines.inline for lines in layouts.layouts.values()
		),
		crossline=_ordered_union(
			lines.crossline for lines in layouts.layouts.values()
		),
	)


def compute_horizon_twt_window(data: VolveHorizonData) -> HorizonTwtWindow:
	'''Compute the fixed outward-aligned native horizon sample window.'''
	bound_indices = data.sample_index[data.bound_valid_mask]
	if bound_indices.size == 0:
		raise ValueError('cannot compute TWT window without bound horizons')
	minimum = int(np.min(bound_indices))
	maximum = int(np.max(bound_indices))
	start = math.floor(
		(minimum - TWT_WINDOW_MARGIN_SAMPLES) / TWT_WINDOW_GRID_SAMPLES
	) * TWT_WINDOW_GRID_SAMPLES
	# A padded inclusive maximum exactly on a grid edge still needs an outer edge.
	padded_stop = maximum + TWT_WINDOW_MARGIN_SAMPLES + 1
	stop = (
		padded_stop // TWT_WINDOW_GRID_SAMPLES + 1
	) * TWT_WINDOW_GRID_SAMPLES
	if start < 0 or stop > data.time_ms.size or stop <= start:
		raise ValueError('aligned TWT window is outside the canonical time axis')
	return HorizonTwtWindow(
		start_index=start,
		stop_index_exclusive=stop,
		length_samples=stop - start,
	)


def plans_metadata(plans: Sequence[HorizonSplitPlan]) -> dict[str, object]:
	'''Return a compact suite artifact containing exactly 15 plan identities.'''
	identities = [plan.identity() for plan in plans]
	if len(identities) != len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX):
		raise ValueError('Volve horizon plan suite must contain exactly 15 plans')
	return {
		'schema_version': 1,
		'artifact_type': 'volve_horizon_split_plans',
		'condition_count': len(identities),
		'plans': identities,
	}


def write_plans_metadata(
	plans: Sequence[HorizonSplitPlan], output_path: str | Path
) -> None:
	'''Write compact split-plan identities without writing dense masks.'''
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(plans_metadata(plans), indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _plan_identity(  # noqa: PLR0913
	*,
	data: VolveHorizonData,
	layouts: VolveHorizonLayouts,
	layout_id: str,
	data_size: str,
	selected_physical: PhysicalSectionLines,
	selected_indices: IndexedSectionLines,
	validation_indices: IndexedSectionLines,
	reserved_physical: PhysicalSectionLines,
	reserved_indices: IndexedSectionLines,
	train_mask: np.ndarray,
	validation_mask: np.ndarray,
	test_primary_mask: np.ndarray,
	test_per_horizon_mask: np.ndarray,
	window: HorizonTwtWindow,
) -> dict[str, object]:
	return {
		'schema_version': 1,
		'selection_semantics': layouts.semantics,
		'layout_config_sha256': layouts.config_sha256,
		'layout_id': layout_id,
		'data_size': data_size,
		'input_identity': data.input_identity(),
		'selected_physical_lines': _lines_dict(selected_physical),
		'selected_indices': _lines_dict(selected_indices),
		'validation_physical_lines': _lines_dict(layouts.validation),
		'validation_indices': _lines_dict(validation_indices),
		'reserved_large_physical_lines': _lines_dict(reserved_physical),
		'reserved_large_indices': _lines_dict(reserved_indices),
		'per_horizon_counts': {
			'train': _horizon_counts(train_mask),
			'validation': _horizon_counts(validation_mask),
			'test_secondary': _horizon_counts(test_per_horizon_mask),
		},
		'test_primary_common_count': int(np.count_nonzero(test_primary_mask)),
		'mask_sha256': {
			'train': array_sha256(train_mask),
			'validation': array_sha256(validation_mask),
			'test_primary_common': array_sha256(test_primary_mask),
			'test_secondary_per_horizon': array_sha256(test_per_horizon_mask),
		},
		'twt_window': {
			'start_index': window.start_index,
			'stop_index_exclusive': window.stop_index_exclusive,
			'length_samples': window.length_samples,
			'margin_samples': window.margin_samples,
			'grid_samples': window.grid_samples,
		},
	}


def _validate_physical_lines(
	validation: PhysicalSectionLines,
	layouts: Mapping[str, PhysicalSectionLines],
	data: VolveHorizonData,
) -> None:
	inline_values = {int(value) for value in data.inline_values}
	crossline_values = {int(value) for value in data.crossline_values}
	for label, lines in [('validation', validation), *layouts.items()]:
		if any(value not in inline_values for value in lines.inline):
			raise ValueError(f'{label} contains an unknown physical inline')
		if any(value not in crossline_values for value in lines.crossline):
			raise ValueError(f'{label} contains an unknown physical crossline')
	for layout_id, lines in layouts.items():
		if set(lines.inline) & set(validation.inline):
			raise ValueError(f'{layout_id} inline overlaps validation inline')
		if set(lines.crossline) & set(validation.crossline):
			raise ValueError(f'{layout_id} crossline overlaps validation crossline')
	for data_size, prefix in DATA_SIZE_PREFIX.items():
		signatures = {
			(lines.inline[:prefix], lines.crossline[:prefix])
			for lines in layouts.values()
		}
		if len(signatures) != len(LAYOUT_IDS):
			raise ValueError(f'{data_size} selections must be unique across layouts')


def _physical_lines(
	value: Mapping[str, object], *, label: str, expected: int
) -> PhysicalSectionLines:
	if set(value) != {'inline', 'crossline'}:
		raise ValueError(f'{label} must contain exactly inline and crossline')
	return PhysicalSectionLines(
		inline=_line_values(value.get('inline'), f'{label}.inline', expected),
		crossline=_line_values(
			value.get('crossline'), f'{label}.crossline', expected
		),
	)


def _line_values(value: object, label: str, expected: int) -> tuple[int, ...]:
	if not isinstance(value, list) or len(value) != expected:
		raise ValueError(f'{label} must contain exactly {expected} values')
	if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
		raise TypeError(f'{label} values must be physical integer line numbers')
	result = tuple(value)
	if len(set(result)) != len(result):
		raise ValueError(f'{label} values must be unique')
	return result


def _to_indices(
	lines: PhysicalSectionLines, data: VolveHorizonData
) -> IndexedSectionLines:
	inline_lookup = {
		int(physical): index for index, physical in enumerate(data.inline_values)
	}
	crossline_lookup = {
		int(physical): index
		for index, physical in enumerate(data.crossline_values)
	}
	return IndexedSectionLines(
		inline=tuple(inline_lookup[value] for value in lines.inline),
		crossline=tuple(crossline_lookup[value] for value in lines.crossline),
	)


def _section_mask(
	shape_xy: tuple[int, int], lines: IndexedSectionLines
) -> np.ndarray:
	mask = np.zeros(shape_xy, dtype=np.bool_)
	mask[np.asarray(lines.inline, dtype=np.intp), :] = True
	mask[:, np.asarray(lines.crossline, dtype=np.intp)] = True
	return mask


def _ordered_union(values: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
	result: list[int] = []
	for group in values:
		for value in group:
			if value not in result:
				result.append(value)
	return tuple(result)


def _horizon_counts(mask: np.ndarray) -> dict[str, int]:
	return {
		name: int(np.count_nonzero(mask[index]))
		for index, name in enumerate(HORIZON_NAMES)
	}


def _lines_dict(
	lines: PhysicalSectionLines | IndexedSectionLines,
) -> dict[str, list[int]]:
	return {
		'inline': list(lines.inline),
		'crossline': list(lines.crossline),
	}


def _mapping(
	value: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{label}.{key} must be a mapping')
	return child


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file_obj:
		for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _json_sha256(value: Mapping[str, object]) -> str:
	payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
	return hashlib.sha256(payload).hexdigest()


__all__ = [
	'DATA_SIZE_PREFIX',
	'LAYOUT_IDS',
	'SELECTION_SEMANTICS',
	'HorizonSplitPlan',
	'HorizonTwtWindow',
	'IndexedSectionLines',
	'PhysicalSectionLines',
	'VolveHorizonLayouts',
	'build_all_horizon_split_plans',
	'build_horizon_split_plan',
	'compute_horizon_twt_window',
	'load_volve_horizon_layouts',
	'plans_metadata',
	'reserved_large_lines',
	'selected_training_lines',
	'write_plans_metadata',
]
