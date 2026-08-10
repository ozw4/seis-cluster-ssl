"""Deterministic calibration for the F3 section-layout voxel benchmark."""
# ruff: noqa: CPY001

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	CONTRACT_ARTIFACT_TYPE,
	CONTRACT_SCHEMA_VERSION,
	DATA_SIZES,
	DECODER_SEED,
	FIXED_DECODER_CONTRACT,
	LAYOUT_IDS,
	LINE_COUNTS,
	NESTING_SEMANTICS,
	PATCH_SIZE,
	STATISTICAL_UNIT,
	VALIDATION_MASK_SEMANTICS,
	f3_lithology_voxel_section_layout_contract_from_mapping,
)
from seis_ssl_cluster.config.io import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_split import (
	TRAIN_VOXEL_SPLIT,
	VALIDATION_VOXEL_SPLIT,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

ARTIFACT_TYPE = CONTRACT_ARTIFACT_TYPE
CANDIDATE_ARTIFACT_TYPE = 'f3_lithology_voxel_section_candidates'
SELECTION_SEMANTICS = 'stable_hash_partial_section_token_footprints_v1'
CLASS_IDS = tuple(range(6))
LEGACY_BUDGETS = ('cap25', 'cap50', 'cap100')
LEGACY_SEEDS = (0, 1, 2, 3, 4)
BUDGET_TO_SIZE = {'cap25': 'small', 'cap50': 'medium', 'cap100': 'large'}
ACTIVE_PREFIX_COUNTS = {
	data_size: {'inline': counts[0], 'crossline': counts[1]}
	for data_size, counts in LINE_COUNTS.items()
}


@dataclass(frozen=True)
class LegacyBudgetCount:
	"""One canonical legacy budget/seed teacher-voxel observation."""

	budget_id: str
	subsample_seed: int
	actual_train_voxel_count: int


@dataclass(frozen=True)
class SectionLine:
	"""One physical F3 line and its zero-based volume axis index."""

	slice_type: str
	slice_index: int
	array_index: int
	is_validation_line: bool

	@property
	def key(self) -> tuple[str, int]:
		"""Return the stable physical line identity."""
		return (self.slice_type, self.slice_index)


@dataclass(frozen=True)
class SectionCandidate:
	"""Class/count inspection statistics for one inventoried line."""

	line: SectionLine
	canonical_train_voxel_count: int
	valid_annotation_voxel_count: int
	per_class_voxel_counts: Mapping[str, int]
	intersecting_token_footprint_count: int

	def to_dict(self) -> dict[str, object]:
		"""Return the canonical CSV/JSON row representation."""
		counts = {
			str(class_id): int(self.per_class_voxel_counts[str(class_id)])
			for class_id in CLASS_IDS
		}
		return {
			'slice_type': self.line.slice_type,
			'slice_index': self.line.slice_index,
			'array_index': self.line.array_index,
			'canonical_train_voxel_count': self.canonical_train_voxel_count,
			'valid_annotation_voxel_count': self.valid_annotation_voxel_count,
			'per_class_voxel_counts': counts,
			'class_3_voxel_count': counts['3'],
			'class_5_voxel_count': counts['5'],
			'intersecting_token_footprint_count': (
				self.intersecting_token_footprint_count
			),
			'is_validation_line': self.line.is_validation_line,
		}


@dataclass(frozen=True)
class LayoutLines:
	"""The user-selected ordered four-inline/four-crossline layout."""

	layout_id: str
	ordered_inlines: tuple[int, ...]
	ordered_crosslines: tuple[int, ...]


@dataclass(frozen=True)
class TokenFootprint:
	"""Teacher voxels in one token block intersected with active line planes."""

	token_xyz: tuple[int, int, int]
	flat_voxel_indices: tuple[int, ...]
	per_line_flat_voxel_indices: Mapping[
		tuple[str, int], tuple[int, ...]
	]

	@property
	def voxel_count(self) -> int:
		"""Return the exact partial-plane teacher voxel count."""
		return len(self.flat_voxel_indices)

	def line_voxel_count(self, line_key: tuple[str, int]) -> int:
		"""Return teacher voxels owned by one line under ordered attribution."""
		return len(self.per_line_flat_voxel_indices.get(line_key, ()))


@dataclass(frozen=True)
class SelectionPreview:
	"""One nested size selection and its finalize-gate evidence."""

	layout_id: str
	data_size: str
	inline_lines: tuple[int, ...]
	crossline_lines: tuple[int, ...]
	target_train_voxel_count: int
	actual_train_voxel_count: int
	count_error: int
	relative_count_error: float
	selected_token_xyz: tuple[tuple[int, int, int], ...]
	selected_flat_voxel_indices: tuple[int, ...]
	per_line_contributions: Mapping[str, int]
	per_class_voxel_counts: Mapping[str, int]


@dataclass(frozen=True)
class F3SectionLayoutCalibrationConfig:
	"""Strict paths and scientific constants for the two-mode calibration CLI."""

	legacy_budget_manifest: Path
	canonical_split_grid: Path
	label_volume: Path
	line_inventory: Path
	segy_geometry_json: Path
	layout_lines: Path
	candidate_statistics_csv: Path
	candidate_statistics_json: Path
	canonical_contract: Path
	selection_semantics: str
	patch_size_xyz: tuple[int, int, int]
	allowed_relative_error: float


def f3_section_layout_calibration_config_from_mapping(
	config: Mapping[str, object],
) -> F3SectionLayoutCalibrationConfig:
	"""Resolve the calibration tool config with fail-closed unknown-key checks."""
	_validate_allowed_keys(
		config, frozenset({'inputs', 'selection', 'outputs'}), prefix='config'
	)
	inputs = _required_mapping(config, 'inputs')
	selection = _required_mapping(config, 'selection')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		inputs,
		frozenset(
			{
				'legacy_budget_manifest',
				'canonical_split_grid',
				'label_volume',
				'line_inventory',
				'segy_geometry_json',
				'layout_lines',
			}
		),
		prefix='inputs',
	)
	_validate_allowed_keys(
		selection,
		frozenset({'semantics', 'patch_size_xyz', 'allowed_relative_error'}),
		prefix='selection',
	)
	_validate_allowed_keys(
		outputs,
		frozenset(
			{
				'candidate_statistics_csv',
				'candidate_statistics_json',
				'canonical_contract',
			}
		),
		prefix='outputs',
	)
	semantics = _required_str(selection, 'semantics', prefix='selection')
	if semantics != SELECTION_SEMANTICS:
		raise ValueError(f'selection.semantics must be exactly {SELECTION_SEMANTICS!r}')
	patch = _positive_triplet(
		selection.get('patch_size_xyz'), 'selection.patch_size_xyz'
	)
	if patch != PATCH_SIZE:
		raise ValueError(
			f'selection.patch_size_xyz must be exactly {list(PATCH_SIZE)!r}'
		)
	tolerance = _relative_error(selection.get('allowed_relative_error'))
	return F3SectionLayoutCalibrationConfig(
		legacy_budget_manifest=_required_absolute_path(
			inputs, 'legacy_budget_manifest', prefix='inputs'
		),
		canonical_split_grid=_required_absolute_path(
			inputs, 'canonical_split_grid', prefix='inputs'
		),
		label_volume=_required_absolute_path(inputs, 'label_volume', prefix='inputs'),
		line_inventory=_required_absolute_path(
			inputs, 'line_inventory', prefix='inputs'
		),
		segy_geometry_json=_required_absolute_path(
			inputs, 'segy_geometry_json', prefix='inputs'
		),
		layout_lines=_required_absolute_path(inputs, 'layout_lines', prefix='inputs'),
		candidate_statistics_csv=_required_absolute_path(
			outputs, 'candidate_statistics_csv', prefix='outputs'
		),
		candidate_statistics_json=_required_absolute_path(
			outputs, 'candidate_statistics_json', prefix='outputs'
		),
		canonical_contract=_required_absolute_path(
			outputs, 'canonical_contract', prefix='outputs'
		),
		selection_semantics=semantics,
		patch_size_xyz=patch,
		allowed_relative_error=tolerance,
	)


def load_legacy_budget_counts(
	source: Path | Mapping[str, object],
) -> tuple[LegacyBudgetCount, ...]:
	"""Load and validate the exact cap25/50/100 by seed0..4 legacy matrix."""
	payload = _load_json_mapping(source, label='legacy budget manifest')
	raw_rows = _manifest_rows(payload)
	if len(raw_rows) != 15:
		raise ValueError('legacy budget manifest must contain exactly 15 rows')
	rows: list[LegacyBudgetCount] = []
	seen: set[tuple[str, int]] = set()
	for index, raw in enumerate(raw_rows):
		if not isinstance(raw, Mapping):
			raise TypeError(f'legacy budget row {index} must be a mapping')
		budget = raw.get('budget_id', raw.get('budget'))
		if budget not in LEGACY_BUDGETS:
			raise ValueError(
				f'legacy budget row {index} has unsupported budget {budget!r}'
			)
		seed = _integer(
			raw.get('subsample_seed'), f'legacy budget row {index}.subsample_seed'
		)
		if seed not in LEGACY_SEEDS:
			raise ValueError(
				f'legacy budget row {index} has unsupported subsample seed {seed!r}'
			)
		count = _legacy_actual_count(raw, row_index=index)
		key = (cast('str', budget), seed)
		if key in seen:
			raise ValueError(f'legacy budget matrix contains duplicate row {key!r}')
		seen.add(key)
		rows.append(LegacyBudgetCount(key[0], key[1], count))
	expected = {(budget, seed) for budget in LEGACY_BUDGETS for seed in LEGACY_SEEDS}
	if seen != expected:
		missing = sorted(expected - seen)
		extra = sorted(seen - expected)
		raise ValueError(
			f'legacy budget matrix mismatch; missing={missing!r}, extra={extra!r}'
		)
	return tuple(
		sorted(
			rows,
			key=lambda item: (
				LEGACY_BUDGETS.index(item.budget_id),
				item.subsample_seed,
			),
		)
	)


def median_target_counts(
	rows: Sequence[LegacyBudgetCount | Mapping[str, object]],
) -> dict[str, int]:
	"""Return integer small/medium/large medians from canonical actual counts."""
	resolved = _coerce_legacy_rows(rows)
	if len(resolved) != 15:
		raise ValueError('median target calculation requires exactly 15 legacy rows')
	result: dict[str, int] = {}
	for budget, data_size in BUDGET_TO_SIZE.items():
		counts = sorted(
			item.actual_train_voxel_count
			for item in resolved
			if item.budget_id == budget
		)
		if len(counts) != 5:
			raise ValueError(f'{budget} must have exactly five actual counts')
		result[data_size] = counts[2]
	return result


def inspect_section_candidates(  # noqa: PLR0913
	canonical_split_grid: NDArray[np.integer],
	label_volume: NDArray[np.integer],
	line_inventory: Sequence[SectionLine | Mapping[str, object]],
	*,
	geometry: Mapping[str, object] | None = None,
	patch_size_xyz: Sequence[int] = PATCH_SIZE,
	class_ids: Sequence[int] = CLASS_IDS,
) -> tuple[SectionCandidate, ...]:
	"""Compute deterministic class/count/token statistics for inventoried lines."""
	grid, labels = _volume_pair(canonical_split_grid, label_volume)
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	classes = _class_ids(class_ids)
	lines = _resolve_section_lines(line_inventory, geometry=geometry, shape=grid.shape)
	result: list[SectionCandidate] = []
	for line in lines:
		plane = _line_plane(line, grid.shape)
		train = plane & (grid == TRAIN_VOXEL_SPLIT)
		valid = plane & np.isin(labels, np.asarray(classes, dtype=labels.dtype))
		teacher = train & valid
		counts = {
			str(class_id): int(np.count_nonzero(teacher & (labels == class_id)))
			for class_id in classes
		}
		tokens = _token_coordinates_for_mask(teacher, patch)
		result.append(
			SectionCandidate(
				line=line,
				canonical_train_voxel_count=int(np.count_nonzero(train)),
				valid_annotation_voxel_count=int(np.count_nonzero(valid)),
				per_class_voxel_counts=counts,
				intersecting_token_footprint_count=len(tokens),
			)
		)
	return tuple(
		sorted(result, key=lambda item: (item.line.slice_type, item.line.slice_index))
	)


def validate_layout_lines(  # noqa: C901
	payload: Mapping[str, object] | Sequence[Mapping[str, object]],
	candidates: Sequence[SectionCandidate | Mapping[str, object]],
) -> tuple[LayoutLines, ...]:
	"""Validate exactly five user-ordered 4+4 train-only layouts."""
	if isinstance(payload, Mapping):
		_validate_allowed_keys(payload, frozenset({'layouts'}), prefix='layout config')
		raw_layouts = payload.get('layouts')
	else:
		raw_layouts = payload
	if not isinstance(raw_layouts, Sequence) or isinstance(raw_layouts, str | bytes):
		raise TypeError('layouts must be a list')
	if len(raw_layouts) != len(LAYOUT_IDS):
		raise ValueError(f'layouts must contain exactly {len(LAYOUT_IDS)} entries')
	candidate_lines = _candidate_line_map(candidates)
	result: list[LayoutLines] = []
	for index, raw in enumerate(raw_layouts):
		if not isinstance(raw, Mapping):
			raise TypeError(f'layouts[{index}] must be a mapping')
		_validate_allowed_keys(
			raw,
			frozenset({'layout_id', 'ordered_inlines', 'ordered_crosslines'}),
			prefix=f'layouts[{index}]',
		)
		layout_id = _required_str(raw, 'layout_id', prefix=f'layouts[{index}]')
		inlines = _line_number_list(
			raw.get('ordered_inlines'), f'layouts[{index}].ordered_inlines', expected=4
		)
		crosslines = _line_number_list(
			raw.get('ordered_crosslines'),
			f'layouts[{index}].ordered_crosslines',
			expected=4,
		)
		for slice_type, values in (('inline', inlines), ('crossline', crosslines)):
			for value in values:
				line = candidate_lines.get((slice_type, value))
				if line is None:
					raise ValueError(
						f'{layout_id} selects unknown {slice_type} line {value}'
					)
				if line.is_validation_line:
					raise ValueError(
						f'{layout_id} selects validation {slice_type} line {value}'
					)
		result.append(LayoutLines(layout_id, inlines, crosslines))
	ids = tuple(item.layout_id for item in result)
	if len(set(ids)) != len(ids):
		raise ValueError('layout IDs must be unique')
	if set(ids) != set(LAYOUT_IDS):
		raise ValueError(f'layout IDs must be exactly {list(LAYOUT_IDS)!r}')
	by_id = {item.layout_id: item for item in result}
	return tuple(by_id[layout_id] for layout_id in LAYOUT_IDS)


def candidate_token_footprints(
	canonical_split_grid: NDArray[np.integer],
	label_volume: NDArray[np.integer],
	active_lines: Sequence[SectionLine | Mapping[str, object]],
	*,
	patch_size_xyz: Sequence[int] = PATCH_SIZE,
	class_ids: Sequence[int] = CLASS_IDS,
) -> tuple[TokenFootprint, ...]:
	"""Return disjoint token units containing only active-plane train voxels."""
	grid, labels = _volume_pair(canonical_split_grid, label_volume)
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	classes = _class_ids(class_ids)
	lines = tuple(_coerce_section_line(item, grid.shape) for item in active_lines)
	if not lines:
		raise ValueError('active_lines must not be empty')
	active_mask = np.zeros(grid.shape, dtype=np.bool_)
	for line in lines:
		if line.is_validation_line:
			raise ValueError(f'active line {line.key!r} is a validation line')
		active_mask |= _line_plane(line, grid.shape)
	teacher = (
		active_mask
		& (grid == TRAIN_VOXEL_SPLIT)
		& np.isin(labels, np.asarray(classes, dtype=labels.dtype))
	)
	coordinates = _token_coordinates_for_mask(teacher, patch)
	result: list[TokenFootprint] = []
	for coordinate in coordinates:
		block = _token_block(coordinate, patch, grid.shape)
		block_mask = teacher[block]
		local = np.argwhere(block_mask)
		start = tuple(part.start or 0 for part in block)
		global_xyz = local + np.asarray(start, dtype=np.int64)
		flat_array = np.ravel_multi_index(global_xyz.T, grid.shape)
		order = np.argsort(flat_array)
		global_xyz = global_xyz[order]
		flat = tuple(int(value) for value in flat_array[order])
		per_line = _ordered_line_voxel_ownership(
			global_xyz,
			flat_voxel_indices=flat,
			lines=lines,
		)
		result.append(TokenFootprint(coordinate, flat, per_line))
	return tuple(result)


def stable_token_order(
	footprints: Sequence[TokenFootprint],
	*,
	layout_id: str,
	semantics_version: str = SELECTION_SEMANTICS,
) -> tuple[TokenFootprint, ...]:
	"""Order token coordinates by an explicit SHA-256 identity, never RNG/hash()."""
	if layout_id not in LAYOUT_IDS:
		raise ValueError(f'unsupported layout_id {layout_id!r}')
	if semantics_version != SELECTION_SEMANTICS:
		raise ValueError(f'unsupported selection semantics {semantics_version!r}')

	def key(item: TokenFootprint) -> tuple[str, tuple[int, int, int]]:
		x, y, z = item.token_xyz
		identity = f'{layout_id}|{x},{y},{z}|{semantics_version}'.encode()
		return (hashlib.sha256(identity).hexdigest(), item.token_xyz)

	return tuple(sorted(footprints, key=key))


def preview_nested_selection(  # noqa: PLR0913
	layout: LayoutLines,
	target_train_voxel_counts: Mapping[str, int],
	canonical_split_grid: NDArray[np.integer],
	label_volume: NDArray[np.integer],
	candidates: Sequence[SectionCandidate | Mapping[str, object]],
	*,
	patch_size_xyz: Sequence[int] = PATCH_SIZE,
	allowed_relative_error: float = 0.05,
) -> tuple[SelectionPreview, ...]:
	"""Build coverage-first nested small/medium/large selection previews."""
	grid, labels = _volume_pair(canonical_split_grid, label_volume)
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	tolerance = _relative_error(allowed_relative_error)
	targets = _target_counts(target_train_voxel_counts)
	line_map = _candidate_line_map(candidates)
	selected_tokens: set[tuple[int, int, int]] = set()
	previews: list[SelectionPreview] = []
	previous_voxels: set[int] = set()
	for data_size in DATA_SIZES:
		inline_count, crossline_count = LINE_COUNTS[data_size]
		inline_lines = layout.ordered_inlines[:inline_count]
		crossline_lines = layout.ordered_crosslines[:crossline_count]
		active = tuple(line_map[('inline', value)] for value in inline_lines) + tuple(
			line_map[('crossline', value)] for value in crossline_lines
		)
		footprints = candidate_token_footprints(
			grid, labels, active, patch_size_xyz=patch
		)
		by_xyz = {item.token_xyz: item for item in footprints}
		if not selected_tokens <= set(by_xyz):
			raise AssertionError('nested active lines lost a previously selected token')
		ordered = stable_token_order(footprints, layout_id=layout.layout_id)
		# Coverage pass is deterministic and precedes target filling.
		covered = {
			key
			for xyz in selected_tokens
			for key in by_xyz[xyz].per_line_flat_voxel_indices
		}
		for line in active:
			if line.key in covered:
				continue
			candidate = next(
				(
					item
					for item in ordered
					if item.token_xyz not in selected_tokens
					and item.line_voxel_count(line.key) > 0
				),
				None,
			)
			if candidate is None:
				raise ValueError(
					f'{layout.layout_id}/{data_size} active line '
					f'{line.key!r} contributes no teacher voxels'
				)
			selected_tokens.add(candidate.token_xyz)
			covered.update(candidate.per_line_flat_voxel_indices)
		selected_tokens = _fill_to_nearest_target(
			selected_tokens, ordered, by_xyz=by_xyz, target=targets[data_size]
		)
		selected_footprints = tuple(by_xyz[xyz] for xyz in sorted(selected_tokens))
		selected_voxels = {
			flat for item in selected_footprints for flat in item.flat_voxel_indices
		}
		if not previous_voxels <= selected_voxels:
			raise AssertionError(
				'nested selection lost previously selected teacher voxels'
			)
		previous_voxels = selected_voxels
		actual = len(selected_voxels)
		error = actual - targets[data_size]
		class_counts = _selected_class_counts(labels, selected_voxels)
		line_counts = _per_line_contributions(selected_footprints, active)
		preview = SelectionPreview(
			layout_id=layout.layout_id,
			data_size=data_size,
			inline_lines=inline_lines,
			crossline_lines=crossline_lines,
			target_train_voxel_count=targets[data_size],
			actual_train_voxel_count=actual,
			count_error=error,
			relative_count_error=abs(error) / targets[data_size],
			selected_token_xyz=tuple(sorted(selected_tokens)),
			selected_flat_voxel_indices=tuple(sorted(selected_voxels)),
			per_line_contributions=line_counts,
			per_class_voxel_counts=class_counts,
		)
		_validate_preview_gate(preview, allowed_relative_error=tolerance)
		previews.append(preview)
	return tuple(previews)


def build_section_layout_contract(  # noqa: PLR0913
	layouts: Sequence[LayoutLines],
	target_train_voxel_counts: Mapping[str, int],
	previews: Sequence[SelectionPreview],
	*,
	allowed_relative_error: float,
	validation_identity: Mapping[str, object],
	source_file_identities: Mapping[str, Mapping[str, str]],
	legacy_budget_source_identity: Mapping[str, str],
) -> dict[str, object]:
	"""Build and self-resolve the canonical downstream section-layout contract."""
	targets = _target_counts(target_train_voxel_counts)
	tolerance = _relative_error(allowed_relative_error)
	if tuple(item.layout_id for item in layouts) != LAYOUT_IDS:
		raise ValueError(f'layouts must be ordered exactly as {list(LAYOUT_IDS)!r}')
	preview_map = {(item.layout_id, item.data_size): item for item in previews}
	if set(preview_map) != {
		(layout_id, size) for layout_id in LAYOUT_IDS for size in DATA_SIZES
	}:
		raise ValueError(
			'previews must contain the exact five-layout by three-size matrix'
		)
	if validation_identity.get('unchanged_by_preview') is not True:
		raise ValueError('validation mask must be unchanged by every selection preview')
	contract_layouts: list[dict[str, object]] = []
	for layout in layouts:
		selected_by_size = {
			size: set(preview_map[(layout.layout_id, size)].selected_token_xyz)
			for size in DATA_SIZES
		}
		if not (
			selected_by_size['small'] <= selected_by_size['medium']
			and selected_by_size['medium'] <= selected_by_size['large']
		):
			raise ValueError(
				f'{layout.layout_id} selected tokens must be nested small <= '
				'medium <= large'
			)
		sizes: dict[str, object] = {}
		for data_size in DATA_SIZES:
			preview = preview_map[(layout.layout_id, data_size)]
			_validate_preview_gate(preview, allowed_relative_error=tolerance)
			sizes[data_size] = {
				'inline_lines': list(preview.inline_lines),
				'crossline_lines': list(preview.crossline_lines),
				'target_train_voxel_count': preview.target_train_voxel_count,
				'preview_actual_train_voxel_count': preview.actual_train_voxel_count,
				'preview_count_error': preview.count_error,
				'preview_relative_count_error': preview.relative_count_error,
				'selected_token_xyz': [list(xyz) for xyz in preview.selected_token_xyz],
				'per_line_contributions': dict(preview.per_line_contributions),
				'per_class_voxel_counts': dict(preview.per_class_voxel_counts),
			}
		contract_layouts.append(
			{
				'layout_id': layout.layout_id,
				'ordered_inlines': list(layout.ordered_inlines),
				'ordered_crosslines': list(layout.ordered_crosslines),
				'sizes': sizes,
			}
		)
	payload: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': CONTRACT_SCHEMA_VERSION,
		'selection_semantics': SELECTION_SEMANTICS,
		'stable_selection_semantics': SELECTION_SEMANTICS,
		'statistical_unit': STATISTICAL_UNIT,
		'nesting_semantics': NESTING_SEMANTICS,
		'validation_mask_semantics': VALIDATION_MASK_SEMANTICS,
		'patch_size': list(PATCH_SIZE),
		'patch_size_xyz': list(PATCH_SIZE),
		'allowed_relative_error': tolerance,
		'target_train_voxel_counts': targets,
		'active_prefix_counts': ACTIVE_PREFIX_COUNTS,
		'decoder_seed': DECODER_SEED,
		'decoder': _jsonable_decoder_contract(),
		'layouts': contract_layouts,
		'validation_identity': dict(validation_identity),
		'source_file_identities': {
			key: dict(value) for key, value in source_file_identities.items()
		},
		'legacy_budget_source_identity': dict(legacy_budget_source_identity),
	}
	f3_lithology_voxel_section_layout_contract_from_mapping(payload)
	return payload


def run_section_layout_calibration(
	config: F3SectionLayoutCalibrationConfig,
	*,
	mode: str,
	dry_run: bool,
) -> Mapping[str, object]:
	"""Execute inspect or finalize without ever reading model artifacts or metrics."""
	if mode not in {'inspect', 'finalize'}:
		raise ValueError("mode must be exactly 'inspect' or 'finalize'")
	legacy_rows = load_legacy_budget_counts(config.legacy_budget_manifest)
	targets = median_target_counts(legacy_rows)
	grid = _load_integer_array(
		config.canonical_split_grid, label='canonical split grid'
	)
	labels = _load_integer_array(config.label_volume, label='label volume')
	inventory = _load_inventory_csv(config.line_inventory)
	geometry = _load_json_mapping(config.segy_geometry_json, label='SEGY geometry')
	candidates = inspect_section_candidates(
		grid, labels, inventory, geometry=geometry, patch_size_xyz=config.patch_size_xyz
	)
	candidate_payload = {
		'artifact_type': CANDIDATE_ARTIFACT_TYPE,
		'selection_semantics': SELECTION_SEMANTICS,
		'patch_size_xyz': list(config.patch_size_xyz),
		'target_train_voxel_counts': targets,
		'rows': [item.to_dict() for item in candidates],
		'source_file_identities': _input_identities(config, include_layout=False),
	}
	if mode == 'inspect':
		if not dry_run:
			_write_candidate_outputs(config, candidate_payload)
		return candidate_payload
	layout_payload = _load_yaml_mapping(config.layout_lines)
	layouts = validate_layout_lines(layout_payload, candidates)
	previews = tuple(
		preview
		for layout in layouts
		for preview in preview_nested_selection(
			layout,
			targets,
			grid,
			labels,
			candidates,
			patch_size_xyz=config.patch_size_xyz,
			allowed_relative_error=config.allowed_relative_error,
		)
	)
	validation_identity = _validation_identity(grid, config.canonical_split_grid)
	contract = build_section_layout_contract(
		layouts,
		targets,
		previews,
		allowed_relative_error=config.allowed_relative_error,
		validation_identity=validation_identity,
		source_file_identities=_input_identities(config, include_layout=True),
		legacy_budget_source_identity=_identity(config.legacy_budget_manifest),
	)
	if not dry_run:
		_write_json_new(config.canonical_contract, contract)
	return contract


def _manifest_rows(payload: Mapping[str, object]) -> Sequence[object]:
	for key in ('rows', 'conditions', 'datasets', 'jobs'):
		value = payload.get(key)
		if isinstance(value, Sequence) and not isinstance(value, str | bytes):
			return value
	raise ValueError(
		'legacy budget manifest must contain rows, conditions, datasets, or jobs'
	)


def _legacy_actual_count(row: Mapping[str, object], *, row_index: int) -> int:
	identity = row.get('identity')
	sources = (row, identity) if isinstance(identity, Mapping) else (row,)
	values: list[int] = []
	for source in sources:
		values.extend(
			_positive_integer(source[key], f'legacy budget row {row_index}.{key}')
			for key in ('actual_train_voxel_count', 'train_voxel_count')
			if key in source
		)
	if not values:
		raise ValueError(
			f'legacy budget row {row_index} has no canonical actual train count'
		)
	if len(set(values)) != 1:
		raise ValueError(
			f'legacy budget row {row_index} has inconsistent actual train counts'
		)
	return values[0]


def _coerce_legacy_rows(
	rows: Sequence[LegacyBudgetCount | Mapping[str, object]],
) -> tuple[LegacyBudgetCount, ...]:
	result: list[LegacyBudgetCount] = []
	for index, item in enumerate(rows):
		if isinstance(item, LegacyBudgetCount):
			result.append(item)
		elif isinstance(item, Mapping):
			budget = item.get('budget_id', item.get('budget'))
			seed = _integer(item.get('subsample_seed'), f'rows[{index}].subsample_seed')
			if not isinstance(budget, str):
				raise TypeError(f'rows[{index}].budget_id must be a string')
			result.append(
				LegacyBudgetCount(
					budget, seed, _legacy_actual_count(item, row_index=index)
				)
			)
		else:
			raise TypeError(f'rows[{index}] must be a LegacyBudgetCount or mapping')
	keys = {(item.budget_id, item.subsample_seed) for item in result}
	expected = {(budget, seed) for budget in LEGACY_BUDGETS for seed in LEGACY_SEEDS}
	if keys != expected or len(result) != len(keys):
		raise ValueError('legacy rows must form the exact unique 15-row matrix')
	return tuple(result)


def _resolve_section_lines(
	rows: Sequence[SectionLine | Mapping[str, object]],
	*,
	geometry: Mapping[str, object] | None,
	shape: tuple[int, ...],
) -> tuple[SectionLine, ...]:
	bounds = _geometry_bounds(geometry, shape) if geometry is not None else None
	result: list[SectionLine] = []
	seen: set[tuple[str, int]] = set()
	for index, row in enumerate(rows):
		if isinstance(row, SectionLine):
			line = _coerce_section_line(row, shape)
		elif isinstance(row, Mapping):
			slice_type = row.get('slice_type')
			if slice_type not in {'inline', 'crossline'}:
				raise ValueError(
					f'line_inventory[{index}].slice_type must be inline or crossline'
				)
			slice_index = _positive_integer(
				row.get('slice_index'), f'line_inventory[{index}].slice_index'
			)
			array_value = row.get('array_index')
			if array_value is None:
				if bounds is None:
					raise ValueError(
						'geometry is required when inventory rows omit array_index'
					)
				array_index = slice_index - bounds[cast('str', slice_type)]
			else:
				array_index = _nonnegative_integer(
					array_value, f'line_inventory[{index}].array_index'
				)
			is_validation = row.get('is_validation_line')
			if is_validation is None:
				is_validation = row.get('split') == 'validation'
			if not isinstance(is_validation, bool):
				raise TypeError(
					f'line_inventory[{index}].is_validation_line must be boolean'
				)
			line = _coerce_section_line(
				SectionLine(
					cast('str', slice_type), slice_index, array_index, is_validation
				),
				shape,
			)
		else:
			raise TypeError(f'line_inventory[{index}] must be a mapping')
		if line.key in seen:
			raise ValueError(f'line inventory contains duplicate line {line.key!r}')
		seen.add(line.key)
		result.append(line)
	return tuple(result)


def _geometry_bounds(
	payload: Mapping[str, object], shape: tuple[int, ...]
) -> Mapping[str, int]:
	source: Mapping[str, object] = payload
	segy_files = payload.get('segy_files')
	if isinstance(segy_files, Mapping) and isinstance(segy_files.get('label'), Mapping):
		source = cast('Mapping[str, object]', segy_files['label'])
	elif isinstance(payload.get('label'), Mapping):
		source = cast('Mapping[str, object]', payload['label'])
	inline_min = _integer(source.get('iline_min'), 'geometry.iline_min')
	crossline_min = _integer(source.get('xline_min'), 'geometry.xline_min')
	cube_shape = source.get('cube_shape')
	if (
		cube_shape is not None
		and tuple(_positive_triplet(cube_shape, 'geometry.cube_shape')) != shape
	):
		raise ValueError('geometry cube_shape does not match calibration volume')
	return {'inline': inline_min, 'crossline': crossline_min}


def _coerce_section_line(
	item: SectionLine | Mapping[str, object], shape: tuple[int, ...]
) -> SectionLine:
	if isinstance(item, Mapping):
		line = SectionLine(
			slice_type=str(item.get('slice_type')),
			slice_index=_positive_integer(
				item.get('slice_index'), 'active line slice_index'
			),
			array_index=_nonnegative_integer(
				item.get('array_index'), 'active line array_index'
			),
			is_validation_line=bool(item.get('is_validation_line', False)),
		)
	else:
		line = item
	if line.slice_type not in {'inline', 'crossline'}:
		raise ValueError(f'unsupported slice_type {line.slice_type!r}')
	axis = 0 if line.slice_type == 'inline' else 1
	if line.array_index >= shape[axis]:
		raise ValueError(f'{line.key!r} array index is outside volume')
	return line


def _line_plane(line: SectionLine, shape: tuple[int, ...]) -> NDArray[np.bool_]:
	mask = np.zeros(shape, dtype=np.bool_)
	if line.slice_type == 'inline':
		mask[line.array_index, :, :] = True
	else:
		mask[:, line.array_index, :] = True
	return mask


def _candidate_line_map(
	candidates: Sequence[SectionCandidate | Mapping[str, object]],
) -> dict[tuple[str, int], SectionLine]:
	result: dict[tuple[str, int], SectionLine] = {}
	for item in candidates:
		if isinstance(item, SectionCandidate):
			line = item.line
		elif isinstance(item, Mapping):
			line = SectionLine(
				slice_type=str(item.get('slice_type')),
				slice_index=_positive_integer(
					item.get('slice_index'), 'candidate.slice_index'
				),
				array_index=_nonnegative_integer(
					item.get('array_index'), 'candidate.array_index'
				),
				is_validation_line=_required_bool(
					item.get('is_validation_line'), 'candidate.is_validation_line'
				),
			)
		else:
			raise TypeError('candidates must contain SectionCandidate or mappings')
		if line.key in result:
			raise ValueError(f'candidate lines contain duplicate {line.key!r}')
		result[line.key] = line
	return result


def _fill_to_nearest_target(
	selected: set[tuple[int, int, int]],
	ordered: Sequence[TokenFootprint],
	*,
	by_xyz: Mapping[tuple[int, int, int], TokenFootprint],
	target: int,
) -> set[tuple[int, int, int]]:
	result = set(selected)
	count = sum(by_xyz[xyz].voxel_count for xyz in result)
	if count >= target:
		return result
	for item in ordered:
		if item.token_xyz in result:
			continue
		before_error = abs(target - count)
		after = count + item.voxel_count
		after_error = abs(target - after)
		if after >= target:
			if after_error < before_error:
				result.add(item.token_xyz)
			return result
		result.add(item.token_xyz)
		count = after
	return result


def _selected_class_counts(
	labels: NDArray[np.integer], selected: set[int]
) -> dict[str, int]:
	flat = labels.reshape(-1)
	indices = np.asarray(sorted(selected), dtype=np.int64)
	values = flat[indices] if indices.size else np.empty((0,), dtype=labels.dtype)
	return {
		str(class_id): int(np.count_nonzero(values == class_id))
		for class_id in CLASS_IDS
	}


def _ordered_line_voxel_ownership(
	voxel_xyz: NDArray[np.integer],
	*,
	flat_voxel_indices: Sequence[int],
	lines: Sequence[SectionLine],
) -> Mapping[tuple[str, int], tuple[int, ...]]:
	"""Assign each teacher voxel once using the active line order."""
	if len(voxel_xyz) != len(flat_voxel_indices):
		raise AssertionError('voxel coordinates and flat indices differ in length')
	owned: dict[tuple[str, int], list[int]] = {}
	for xyz, flat in zip(voxel_xyz, flat_voxel_indices, strict=True):
		x, y = int(xyz[0]), int(xyz[1])
		line = next(
			(
				item
				for item in lines
				if (item.slice_type == 'inline' and x == item.array_index)
				or (item.slice_type == 'crossline' and y == item.array_index)
			),
			None,
		)
		if line is None:
			raise AssertionError('teacher voxel lies outside every active line')
		owned.setdefault(line.key, []).append(int(flat))
	return {key: tuple(values) for key, values in owned.items()}


def _per_line_contributions(
	selected: Sequence[TokenFootprint], lines: Sequence[SectionLine]
) -> dict[str, int]:
	"""Count the exact footprint ownership used by the coverage pass."""
	expected = {line.key for line in lines}
	result = {f'{line.slice_type}:{line.slice_index}': 0 for line in lines}
	selected_flat: set[int] = set()
	owned_flat: set[int] = set()
	for footprint in selected:
		flat = set(footprint.flat_voxel_indices)
		if selected_flat & flat:
			raise AssertionError('selected token footprints overlap')
		selected_flat.update(flat)
		for line_key, indices in footprint.per_line_flat_voxel_indices.items():
			if line_key not in expected:
				raise AssertionError('footprint ownership escaped active lines')
			owned = set(indices)
			if not owned <= flat or owned_flat & owned:
				raise AssertionError('footprint line ownership is not disjoint')
			owned_flat.update(owned)
			result[f'{line_key[0]}:{line_key[1]}'] += len(indices)
	if owned_flat != selected_flat:
		raise AssertionError('footprint ownership does not cover teacher voxels')
	return result


def _validate_preview_gate(
	preview: SelectionPreview, *, allowed_relative_error: float
) -> None:
	if preview.relative_count_error > allowed_relative_error + 1e-15:
		raise ValueError(
			f'{preview.layout_id}/{preview.data_size} target relative error '
			f'{preview.relative_count_error:.6g} exceeds '
			f'{allowed_relative_error:.6g}'
		)
	missing = [
		str(class_id)
		for class_id in CLASS_IDS
		if preview.per_class_voxel_counts.get(str(class_id), 0) <= 0
	]
	if missing:
		raise ValueError(
			f'{preview.layout_id}/{preview.data_size} is missing classes {missing!r}'
		)
	if (
		preview.per_class_voxel_counts['3'] <= 0
		or preview.per_class_voxel_counts['5'] <= 0
	):
		raise ValueError(
			f'{preview.layout_id}/{preview.data_size} classes 3 and 5 must be nonzero'
		)
	zero_lines = [
		key for key, count in preview.per_line_contributions.items() if count <= 0
	]
	expected_lines = {
		*(f'inline:{line}' for line in preview.inline_lines),
		*(f'crossline:{line}' for line in preview.crossline_lines),
	}
	if set(preview.per_line_contributions) != expected_lines:
		raise ValueError(
			f'{preview.layout_id}/{preview.data_size} per-line contributions '
			'do not match the active lines'
		)
	if zero_lines:
		raise ValueError(
			f'{preview.layout_id}/{preview.data_size} active lines contribute '
			f'zero teacher voxels: {zero_lines!r}'
		)


def _token_coordinates_for_mask(
	mask: NDArray[np.bool_], patch: tuple[int, int, int]
) -> tuple[tuple[int, int, int], ...]:
	voxel_xyz = np.argwhere(mask)
	if not voxel_xyz.size:
		return ()
	tokens = np.unique(voxel_xyz // np.asarray(patch, dtype=np.int64), axis=0)
	return tuple(tuple(int(value) for value in row) for row in tokens)


def _token_block(
	coordinate: tuple[int, int, int],
	patch: tuple[int, int, int],
	shape: tuple[int, ...],
) -> tuple[slice, slice, slice]:
	start = tuple(coordinate[axis] * patch[axis] for axis in range(3))
	stop = tuple(min(start[axis] + patch[axis], shape[axis]) for axis in range(3))
	return tuple(slice(start[axis], stop[axis]) for axis in range(3))  # type: ignore[return-value]


def _validation_identity(grid: NDArray[np.integer], path: Path) -> dict[str, object]:
	mask = np.asarray(grid == VALIDATION_VOXEL_SPLIT, dtype=np.bool_)
	return {
		'semantics': VALIDATION_MASK_SEMANTICS,
		'voxel_count': int(np.count_nonzero(mask)),
		'mask_sha256': hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
		'source_path': str(path),
		'source_sha256': file_sha256(path),
		'unchanged_by_preview': True,
	}


def _write_candidate_outputs(
	config: F3SectionLayoutCalibrationConfig, payload: Mapping[str, object]
) -> None:
	rows = cast('Sequence[Mapping[str, object]]', payload['rows'])
	flat_rows: list[dict[str, object]] = []
	for row in rows:
		counts = cast('Mapping[str, int]', row['per_class_voxel_counts'])
		flat = {
			key: value for key, value in row.items() if key != 'per_class_voxel_counts'
		}
		flat.update(
			{
				f'class_{class_id}_voxel_count': counts[str(class_id)]
				for class_id in CLASS_IDS
			}
		)
		flat_rows.append(flat)
	_write_csv_new(config.candidate_statistics_csv, flat_rows)
	try:
		_write_json_new(config.candidate_statistics_json, payload)
	except BaseException:
		config.candidate_statistics_csv.unlink(missing_ok=True)
		raise


def _write_csv_new(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	if path.exists():
		raise FileExistsError(f'refusing to overwrite output: {path}')
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('x', encoding='utf-8', newline='') as handle:
		fieldnames = tuple(rows[0]) if rows else ('slice_type',)
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _write_json_new(path: Path, payload: Mapping[str, object]) -> None:
	if path.exists():
		raise FileExistsError(f'refusing to overwrite output: {path}')
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary = tempfile.mkstemp(
		prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
	)
	try:
		with open(fd, 'w', encoding='utf-8') as handle:  # noqa: PTH123
			json.dump(payload, handle, indent=2, sort_keys=True)
			handle.write('\n')
		Path(temporary).replace(path)
	except BaseException:
		Path(temporary).unlink(missing_ok=True)
		raise


def _load_inventory_csv(path: Path) -> tuple[dict[str, object], ...]:
	with path.open(newline='', encoding='utf-8') as handle:
		rows = tuple(dict(row) for row in csv.DictReader(handle))
	if not rows:
		raise ValueError(f'line inventory is empty: {path}')
	return rows


def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
	return load_config(path)


def _load_json_mapping(
	source: Path | Mapping[str, object], *, label: str
) -> Mapping[str, object]:
	if isinstance(source, Mapping):
		return source
	try:
		payload = json.loads(source.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'{label} is not valid JSON: {source}') from exc
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} root must be a mapping')
	return payload


def _load_integer_array(path: Path, *, label: str) -> NDArray[np.integer]:
	array = np.load(path, mmap_mode='r', allow_pickle=False)
	if not np.issubdtype(array.dtype, np.integer) or array.dtype == np.dtype(np.bool_):
		raise TypeError(f'{label} must use an integer dtype')
	if array.ndim != 3:
		raise ValueError(f'{label} must be a 3D volume')
	return cast('NDArray[np.integer]', array)


def _input_identities(
	config: F3SectionLayoutCalibrationConfig, *, include_layout: bool
) -> dict[str, dict[str, str]]:
	paths = {
		'legacy_budget_manifest': config.legacy_budget_manifest,
		'canonical_split_grid': config.canonical_split_grid,
		'label_volume': config.label_volume,
		'line_inventory': config.line_inventory,
		'segy_geometry_json': config.segy_geometry_json,
	}
	if include_layout:
		paths['layout_lines'] = config.layout_lines
	return {key: _identity(path) for key, path in paths.items()}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _jsonable_decoder_contract() -> dict[str, object]:
	return json.loads(json.dumps(dict(FIXED_DECODER_CONTRACT)))


def _volume_pair(
	grid: NDArray[np.integer], labels: NDArray[np.integer]
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
	grid_array = np.asarray(grid)
	label_array = np.asarray(labels)
	for array, name in (
		(grid_array, 'canonical_split_grid'),
		(label_array, 'label_volume'),
	):
		if (
			array.ndim != 3
			or not np.issubdtype(array.dtype, np.integer)
			or array.dtype == np.dtype(np.bool_)
		):
			raise TypeError(f'{name} must be a 3D integer array')
	if grid_array.shape != label_array.shape:
		raise ValueError('canonical split grid and label volume shapes must match')
	return cast('NDArray[np.integer]', grid_array), cast(
		'NDArray[np.integer]', label_array
	)


def _class_ids(value: Sequence[int]) -> tuple[int, ...]:
	result = tuple(_nonnegative_integer(item, 'class_ids entry') for item in value)
	if result != CLASS_IDS:
		raise ValueError(f'class_ids must be exactly {list(CLASS_IDS)!r}')
	return result


def _target_counts(value: Mapping[str, int]) -> dict[str, int]:
	if set(value) != set(DATA_SIZES):
		raise ValueError(f'target counts must define exactly {list(DATA_SIZES)!r}')
	return {
		size: _positive_integer(value[size], f'target counts.{size}')
		for size in DATA_SIZES
	}


def _line_number_list(value: object, label: str, *, expected: int) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a list')
	result = tuple(_positive_integer(item, f'{label} entry') for item in value)
	if len(result) != expected:
		raise ValueError(f'{label} must contain exactly {expected} lines')
	if len(set(result)) != len(result):
		raise ValueError(f'{label} must not contain duplicate lines')
	return result


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'{label} must be an integer triple')
	items = tuple(_positive_integer(item, f'{label} entry') for item in value)
	return cast('tuple[int, int, int]', items)


def _relative_error(value: object) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError('allowed_relative_error must be a number')
	result = float(value)
	if not math.isfinite(result) or not 0.0 < result <= 0.1:
		raise ValueError('allowed_relative_error must be in (0, 0.1]')
	return result


def _integer(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		try:
			if (
				isinstance(value, str)
				and value.strip()
				and str(int(value)) == value.strip()
			):
				return int(value)
		except ValueError:
			pass
		raise TypeError(f'{label} must be an integer; got {value!r}')
	return value


def _positive_integer(value: object, label: str) -> int:
	result = _integer(value, label)
	if result <= 0:
		raise ValueError(f'{label} must be positive; got {value!r}')
	return result


def _nonnegative_integer(value: object, label: str) -> int:
	result = _integer(value, label)
	if result < 0:
		raise ValueError(f'{label} must be nonnegative; got {value!r}')
	return result


def _required_bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be boolean')
	return value


__all__ = [
	'ACTIVE_PREFIX_COUNTS',
	'ARTIFACT_TYPE',
	'CANDIDATE_ARTIFACT_TYPE',
	'CLASS_IDS',
	'SELECTION_SEMANTICS',
	'F3SectionLayoutCalibrationConfig',
	'LayoutLines',
	'LegacyBudgetCount',
	'SectionCandidate',
	'SectionLine',
	'SelectionPreview',
	'TokenFootprint',
	'build_section_layout_contract',
	'candidate_token_footprints',
	'f3_section_layout_calibration_config_from_mapping',
	'inspect_section_candidates',
	'load_legacy_budget_counts',
	'median_target_counts',
	'preview_nested_selection',
	'run_section_layout_calibration',
	'stable_token_order',
	'validate_layout_lines',
]
