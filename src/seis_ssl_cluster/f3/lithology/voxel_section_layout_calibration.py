"""Deterministic candidate inspection and contract finalization for F3 layouts.

The tool never reads model artifacts or metrics. Teacher-voxel targets are
either calibrated from active layout pools or fixed in the versioned config;
both rules validate reachability against every selected layout.
"""

from __future__ import annotations

import csv
import hashlib
import json
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
	CLASS_BALANCED_SELECTION_SEMANTICS,
	CONTRACT_ARTIFACT_TYPE,
	CONTRACT_SCHEMA_VERSION,
	DATA_SIZES,
	DECODER_SEED,
	FIXED_DECODER_CONTRACT,
	FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE,
	FIXED_TRAIN_VOXEL_COUNTS_RULE,
	LAYOUT_IDS,
	LINE_COUNTS,
	NESTING_SEMANTICS,
	PATCH_SIZE,
	STATISTICAL_UNIT,
	TARGET_CALIBRATION_RULE,
	TOKEN_ROW_VALIDATION_PRECEDENCE,
	TOKEN_ROW_VOXEL_MATERIALIZATION,
	TOKENIZATION_POLICY,
	VALIDATION_MASK_SEMANTICS,
	f3_lithology_voxel_section_layout_contract_from_mapping,
)
from seis_ssl_cluster.config.io import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.tokens import (
	F3LithologyTokenPolicy,
	tokenize_f3_lithology_slice,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_selection import (
	CLASS_IDS,
	SELECTION_SEMANTICS,
	LayoutLines,
	SectionLine,
	SelectionPreview,
	TokenRow,
	candidate_token_footprints,
	preview_nested_selection,
	preview_seeded_nested_class_cap_selection,
)
from seis_ssl_cluster.f3.lithology.voxel_split import (
	TRAIN_VOXEL_SPLIT,
	VALIDATION_VOXEL_SPLIT,
)
from seis_ssl_cluster.f3.splits import (
	F3LineGeometry,
	F3SliceSplitRecord,
	load_f3_slice_split_records,
	read_f3_line_geometry,
	resolve_f3_slice_array_index,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

CANDIDATE_ARTIFACT_TYPE = 'f3_lithology_voxel_section_candidates'
# Producer and contract resolver share one rule constant so a stored contract
# is always replayable by the resolver that consumes it.
TARGET_RULE = TARGET_CALIBRATION_RULE
FIXED_TARGET_RULE = FIXED_TRAIN_VOXEL_COUNTS_RULE
CLASS_CAP_TARGET_RULE = FIXED_PER_CLASS_TOKEN_ROW_CAPS_RULE
MONITORED_CLASS_IDS = (3, 5)
ACTIVE_PREFIX_COUNTS = {
	data_size: {'inline': counts[0], 'crossline': counts[1]}
	for data_size, counts in LINE_COUNTS.items()
}


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
class SectionTokenRowPool:
	"""v1-policy token rows after validation-token precedence."""

	train_rows: tuple[TokenRow, ...]
	validation_token_xyz: tuple[tuple[int, int, int], ...]
	train_row_count_before_validation_precedence: int
	train_rows_removed_by_validation_precedence: int

	def provenance(self) -> dict[str, object]:
		"""Return compact selection-input provenance for the contract."""
		return {
			'train_token_row_count': len(self.train_rows),
			'train_row_count_before_validation_precedence': (
				self.train_row_count_before_validation_precedence
			),
			'train_rows_removed_by_validation_precedence': (
				self.train_rows_removed_by_validation_precedence
			),
			'validation_token_xyz_count': len(self.validation_token_xyz),
			'validation_token_xyz_sha256': _sequence_sha256(
				self.validation_token_xyz
			),
		}


@dataclass(frozen=True)
class F3SectionLayoutCalibrationConfig:
	"""Strict paths and scientific constants for the two-mode calibration CLI."""

	canonical_split_grid: Path
	label_volume: Path
	line_inventory: Path
	segy_geometry_json: Path
	layout_lines: Path
	reference_valid_tokens: Path | None
	candidate_statistics_csv: Path
	candidate_statistics_json: Path
	canonical_contract: Path
	selection_semantics: str
	patch_size_xyz: tuple[int, int, int]
	allowed_relative_error: float
	target_rule: str
	target_train_voxel_counts: Mapping[str, int] | None
	per_class_token_row_caps: Mapping[str, int] | None
	layout_subsample_seeds: Mapping[str, int] | None
	tokenization_policy: Mapping[str, object] | None


def f3_section_layout_calibration_config_from_mapping(  # noqa: C901, PLR0912, PLR0915
	config: Mapping[str, object],
) -> F3SectionLayoutCalibrationConfig:
	"""Resolve the calibration tool config with fail-closed unknown-key checks."""
	_validate_allowed_keys(
		config,
		frozenset({'inputs', 'selection', 'targets', 'outputs'}),
		prefix='config',
	)
	inputs = _required_mapping(config, 'inputs')
	selection = _required_mapping(config, 'selection')
	targets = _required_mapping(config, 'targets')
	outputs = _required_mapping(config, 'outputs')
	semantics = _required_str(selection, 'semantics', prefix='selection')
	base_input_names = (
		'canonical_split_grid',
		'label_volume',
		'line_inventory',
		'segy_geometry_json',
		'layout_lines',
	)
	if semantics == SELECTION_SEMANTICS:
		input_names = base_input_names
		selection_names = {
			'semantics',
			'patch_size_xyz',
			'allowed_relative_error',
		}
		target_names = {'rule', 'train_voxel_counts'}
	elif semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
		input_names = (*base_input_names, 'reference_valid_tokens')
		selection_names = {
			'semantics',
			'patch_size_xyz',
			'allowed_relative_error',
			'layout_subsample_seeds',
			'tokenization_policy',
		}
		target_names = {
			'rule',
			'per_class_token_row_caps',
			'nominal_train_voxel_counts',
		}
	else:
		raise ValueError(
			'selection.semantics must be one of '
			f'{[SELECTION_SEMANTICS, CLASS_BALANCED_SELECTION_SEMANTICS]!r}'
		)
	_validate_allowed_keys(inputs, frozenset(input_names), prefix='inputs')
	if set(inputs) != set(input_names):
		missing = sorted(set(input_names) - set(inputs))
		raise ValueError(f'inputs must define every source path; missing={missing!r}')
	_validate_allowed_keys(
		selection,
		frozenset(selection_names),
		prefix='selection',
	)
	_validate_allowed_keys(targets, frozenset(target_names), prefix='targets')
	output_names = (
		'candidate_statistics_csv',
		'candidate_statistics_json',
		'canonical_contract',
	)
	_validate_allowed_keys(outputs, frozenset(output_names), prefix='outputs')
	if set(outputs) != set(output_names):
		missing = sorted(set(output_names) - set(outputs))
		raise ValueError(f'outputs must define every path; missing={missing!r}')
	patch = _positive_triplet(
		selection.get('patch_size_xyz'), 'selection.patch_size_xyz'
	)
	if patch != PATCH_SIZE:
		raise ValueError(
			f'selection.patch_size_xyz must be exactly {list(PATCH_SIZE)!r}'
		)
	tolerance = _relative_error(selection.get('allowed_relative_error'))
	rule = _required_str(targets, 'rule', prefix='targets')
	configured_targets: Mapping[str, int] | None = None
	class_caps: Mapping[str, int] | None = None
	layout_seeds: Mapping[str, int] | None = None
	tokenization_policy: Mapping[str, object] | None = None
	if semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
		if rule != CLASS_CAP_TARGET_RULE:
			raise ValueError(
				'targets.rule must be exactly '
				f'{CLASS_CAP_TARGET_RULE!r} for class-balanced selection'
			)
		class_caps = _configured_size_counts(
			targets.get('per_class_token_row_caps'),
			label='targets.per_class_token_row_caps',
		)
		configured_targets = _configured_size_counts(
			targets.get('nominal_train_voxel_counts'),
			label='targets.nominal_train_voxel_counts',
		)
		for size in DATA_SIZES:
			expected = len(CLASS_IDS) * class_caps[size] * PATCH_SIZE[1] * PATCH_SIZE[2]
			if configured_targets[size] != expected:
				raise ValueError(
					f'targets.nominal_train_voxel_counts.{size} must equal '
					f'six classes * cap * 8 * 8 ({expected})'
				)
		layout_seeds = _configured_layout_seeds(
			selection.get('layout_subsample_seeds')
		)
		tokenization_policy = _configured_tokenization_policy(
			selection.get('tokenization_policy')
		)
	elif rule == TARGET_RULE:
		if 'train_voxel_counts' in targets:
			raise ValueError(
				'targets.train_voxel_counts must not be specified under '
				f'{TARGET_RULE!r}'
			)
	elif rule == FIXED_TARGET_RULE:
		if 'train_voxel_counts' not in targets:
			raise ValueError(
				'targets.train_voxel_counts is required under '
				f'{FIXED_TARGET_RULE!r}'
			)
		configured_targets = _configured_target_counts(
			targets.get('train_voxel_counts')
		)
	else:
		raise ValueError(
			'targets.rule must be one of '
			f'{[TARGET_RULE, FIXED_TARGET_RULE, CLASS_CAP_TARGET_RULE]!r}'
		)
	resolved_inputs = {
		name: _required_absolute_path(inputs, name, prefix='inputs')
		for name in input_names
	}
	resolved_outputs = {
		name: _required_absolute_path(outputs, name, prefix='outputs')
		for name in output_names
	}
	return F3SectionLayoutCalibrationConfig(
		canonical_split_grid=resolved_inputs['canonical_split_grid'],
		label_volume=resolved_inputs['label_volume'],
		line_inventory=resolved_inputs['line_inventory'],
		segy_geometry_json=resolved_inputs['segy_geometry_json'],
		layout_lines=resolved_inputs['layout_lines'],
		reference_valid_tokens=resolved_inputs.get('reference_valid_tokens'),
		candidate_statistics_csv=resolved_outputs['candidate_statistics_csv'],
		candidate_statistics_json=resolved_outputs['candidate_statistics_json'],
		canonical_contract=resolved_outputs['canonical_contract'],
		selection_semantics=semantics,
		patch_size_xyz=patch,
		allowed_relative_error=tolerance,
		target_rule=rule,
		target_train_voxel_counts=configured_targets,
		per_class_token_row_caps=class_caps,
		layout_subsample_seeds=layout_seeds,
		tokenization_policy=tokenization_policy,
	)


def load_section_lines(
	line_inventory: Path, segy_geometry_json: Path
) -> tuple[SectionLine, ...]:
	"""Resolve every inventoried physical line to its zero-based array index."""
	geometry = read_f3_line_geometry(segy_geometry_json)
	records = load_f3_slice_split_records(line_inventory)
	return tuple(
		SectionLine(
			record.slice_type,
			record.slice_index,
			resolve_f3_slice_array_index(record, geometry),
			record.split == 'validation',
		)
		for record in records
	)


def build_section_token_row_pool(  # noqa: C901, PLR0913
	records: Sequence[F3SliceSplitRecord],
	*,
	label_volume: NDArray[np.integer],
	valid_tokens: NDArray[np.bool_],
	geometry: F3LineGeometry,
	patch_size_xyz: Sequence[int] = PATCH_SIZE,
	policy: F3LithologyTokenPolicy | None = None,
) -> SectionTokenRowPool:
	"""Tokenize all sections with the v1 policy, then give validation precedence."""
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	if tuple(label_volume.shape) != tuple(geometry.shape_xyz):
		raise ValueError('label volume shape does not match F3 geometry')
	expected_token_shape = tuple(
		(size + step - 1) // step
		for size, step in zip(geometry.shape_xyz, patch, strict=True)
	)
	if valid_tokens.dtype != np.dtype(np.bool_):
		raise TypeError('reference valid-token mask must use bool dtype')
	if tuple(valid_tokens.shape) != expected_token_shape:
		raise ValueError('reference valid-token mask shape is inconsistent')
	resolved_policy = policy or F3LithologyTokenPolicy(
		min_labeled_fraction=float(TOKENIZATION_POLICY['min_labeled_fraction']),
		min_majority_fraction=float(TOKENIZATION_POLICY['min_majority_fraction']),
		ignore_z_border_samples=int(TOKENIZATION_POLICY['ignore_z_border_samples']),
	)
	train_rows: list[TokenRow] = []
	validation_tokens: set[tuple[int, int, int]] = set()
	for record in records:
		result = tokenize_f3_lithology_slice(
			record,
			label_volume=label_volume,
			valid_tokens=valid_tokens,
			geometry=geometry,
			patch_size_xyz=patch,
			policy=resolved_policy,
		)
		plane_indices = np.argwhere(result.usable_mask)
		fixed = result.tokenization.plane.fixed_token_index
		if record.slice_type == 'inline':
			token_xyz = np.column_stack((
				np.full(plane_indices.shape[0], fixed, dtype=np.int64),
				plane_indices[:, 0],
				plane_indices[:, 1],
			))
		else:
			token_xyz = np.column_stack((
				plane_indices[:, 0],
				np.full(plane_indices.shape[0], fixed, dtype=np.int64),
				plane_indices[:, 1],
			))
		class_ids = result.tokenization.majority_class_ids[result.usable_mask]
		for coordinate, class_id in zip(token_xyz, class_ids, strict=True):
			xyz = tuple(int(value) for value in coordinate)
			if record.split == 'validation':
				validation_tokens.add(cast('tuple[int, int, int]', xyz))
				continue
			if record.split != 'train':
				raise ValueError(f'unsupported section split {record.split!r}')
			resolved_class = int(class_id)
			if resolved_class not in CLASS_IDS:
				raise ValueError(
					f'token row class {resolved_class} is outside the class order'
				)
			train_rows.append(
				TokenRow(
					record.slice_type,
					record.slice_index,
					cast('tuple[int, int, int]', xyz),
					resolved_class,
				)
			)
	before = len(train_rows)
	filtered = tuple(
		sorted(
			(row for row in train_rows if row.token_xyz not in validation_tokens),
			key=lambda row: row.identity,
		)
	)
	if len({row.identity for row in filtered}) != len(filtered):
		raise ValueError('token-row pool contains duplicate row identities')
	return SectionTokenRowPool(
		train_rows=filtered,
		validation_token_xyz=tuple(sorted(validation_tokens)),
		train_row_count_before_validation_precedence=before,
		train_rows_removed_by_validation_precedence=before - len(filtered),
	)


def preview_class_balanced_layout_selection(  # noqa: PLR0913
	layout: LayoutLines,
	token_row_pool: SectionTokenRowPool,
	canonical_split_grid: NDArray[np.integer],
	label_volume: NDArray[np.integer],
	line_inventory: Sequence[SectionLine],
	*,
	subsample_seed: int,
	per_class_token_row_caps: Mapping[str, int],
	target_train_voxel_counts: Mapping[str, int],
	patch_size_xyz: Sequence[int] = PATCH_SIZE,
	allowed_relative_error: float = 0.05,
) -> tuple[SelectionPreview, ...]:
	"""Assemble active section pools and run the shared v3 selection kernel."""
	grid, labels = _volume_pair(canonical_split_grid, label_volume)
	line_map = {
		line.key: line for line in _unique_lines(line_inventory, shape=grid.shape)
	}
	rows_by_size: dict[str, tuple[TokenRow, ...]] = {}
	footprints_by_size = {}
	for data_size in DATA_SIZES:
		inline_count, crossline_count = LINE_COUNTS[data_size]
		active_key_order = tuple(
			('inline', value) for value in layout.ordered_inlines[:inline_count]
		) + tuple(
			('crossline', value)
			for value in layout.ordered_crosslines[:crossline_count]
		)
		active_keys = set(active_key_order)
		active_lines = tuple(line_map[key] for key in active_key_order)
		rows_by_size[data_size] = tuple(
			row for row in token_row_pool.train_rows if row.line_key in active_keys
		)
		footprints_by_size[data_size] = candidate_token_footprints(
			grid,
			labels,
			active_lines,
			patch_size_xyz=patch_size_xyz,
			class_ids=CLASS_IDS,
		)
	return preview_seeded_nested_class_cap_selection(
		layout,
		subsample_seed=subsample_seed,
		per_class_token_row_caps=per_class_token_row_caps,
		target_train_voxel_counts=target_train_voxel_counts,
		token_rows_by_size=rows_by_size,
		token_footprints_by_size=footprints_by_size,
		allowed_relative_error=allowed_relative_error,
	)


def inspect_section_candidates(
	canonical_split_grid: NDArray[np.integer],
	label_volume: NDArray[np.integer],
	line_inventory: Sequence[SectionLine],
	*,
	patch_size_xyz: Sequence[int] = PATCH_SIZE,
	class_ids: Sequence[int] = CLASS_IDS,
) -> tuple[SectionCandidate, ...]:
	"""Compute deterministic class/count/token statistics for inventoried lines."""
	grid, labels = _volume_pair(canonical_split_grid, label_volume)
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	classes = _class_ids(class_ids)
	lines = _unique_lines(line_inventory, shape=grid.shape)
	known = np.asarray(classes, dtype=labels.dtype)
	result: list[SectionCandidate] = []
	for line in lines:
		plane_grid = np.asarray(_plane(grid, line))
		plane_labels = np.asarray(_plane(labels, line))
		train = plane_grid == TRAIN_VOXEL_SPLIT
		valid = np.isin(plane_labels, known)
		teacher = train & valid
		counts = {
			str(class_id): int(np.count_nonzero(teacher & (plane_labels == class_id)))
			for class_id in classes
		}
		result.append(
			SectionCandidate(
				line=line,
				canonical_train_voxel_count=int(np.count_nonzero(train)),
				valid_annotation_voxel_count=int(np.count_nonzero(valid)),
				per_class_voxel_counts=counts,
				intersecting_token_footprint_count=_plane_token_count(
					teacher, line=line, patch=patch
				),
			)
		)
	return tuple(
		sorted(result, key=lambda item: (item.line.slice_type, item.line.slice_index))
	)


def validate_layout_lines(
	payload: Mapping[str, object] | Sequence[Mapping[str, object]],
	candidates: Sequence[SectionCandidate],
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
	candidate_lines = {item.line.key: item.line for item in candidates}
	result = tuple(
		_resolve_layout_lines(raw, index=index, candidate_lines=candidate_lines)
		for index, raw in enumerate(raw_layouts)
	)
	ids = tuple(item.layout_id for item in result)
	if len(set(ids)) != len(ids):
		raise ValueError('layout IDs must be unique')
	if set(ids) != set(LAYOUT_IDS):
		raise ValueError(f'layout IDs must be exactly {list(LAYOUT_IDS)!r}')
	by_id = {item.layout_id: item for item in result}
	return tuple(by_id[layout_id] for layout_id in LAYOUT_IDS)


def _resolve_layout_lines(
	raw: object,
	*,
	index: int,
	candidate_lines: Mapping[tuple[str, int], SectionLine],
) -> LayoutLines:
	if not isinstance(raw, Mapping):
		raise TypeError(f'layouts[{index}] must be a mapping')
	_validate_allowed_keys(
		raw,
		frozenset({'layout_id', 'ordered_inlines', 'ordered_crosslines'}),
		prefix=f'layouts[{index}]',
	)
	layout_id = _required_str(raw, 'layout_id', prefix=f'layouts[{index}]')
	inlines = _line_number_list(
		raw.get('ordered_inlines'), f'layouts[{index}].ordered_inlines'
	)
	crosslines = _line_number_list(
		raw.get('ordered_crosslines'), f'layouts[{index}].ordered_crosslines'
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
	return LayoutLines(layout_id, inlines, crosslines)


def active_pool_train_voxel_counts(
	layouts: Sequence[LayoutLines],
	canonical_split_grid: NDArray[np.integer],
	label_volume: NDArray[np.integer],
	line_inventory: Sequence[SectionLine],
	*,
	class_ids: Sequence[int] = CLASS_IDS,
) -> dict[str, dict[str, int]]:
	"""Count teacher voxels reachable by each layout's nested active sections.

	The pool is the union of the active planes intersected with canonical train
	voxels of a known class. It equals the total footprint volume the selection
	kernel can ever pick for that layout and size.
	"""
	grid, labels = _volume_pair(canonical_split_grid, label_volume)
	classes = _class_ids(class_ids)
	line_map = {
		line.key: line for line in _unique_lines(line_inventory, shape=grid.shape)
	}
	known = np.asarray(classes, dtype=labels.dtype)
	result: dict[str, dict[str, int]] = {size: {} for size in DATA_SIZES}
	for layout in layouts:
		for data_size in DATA_SIZES:
			inline_count, crossline_count = LINE_COUNTS[data_size]
			active = tuple(
				line_map[('inline', value)]
				for value in layout.ordered_inlines[:inline_count]
			) + tuple(
				line_map[('crossline', value)]
				for value in layout.ordered_crosslines[:crossline_count]
			)
			pool = np.zeros(grid.shape, dtype=np.bool_)
			for line in active:
				if line.is_validation_line:
					raise ValueError(f'active line {line.key!r} is a validation line')
				train = np.asarray(_plane(grid, line)) == TRAIN_VOXEL_SPLIT
				valid = np.isin(np.asarray(_plane(labels, line)), known)
				pool[_plane_index(line)] |= train & valid
			result[data_size][layout.layout_id] = int(np.count_nonzero(pool))
	return result


def calibrate_target_train_voxel_counts(
	active_pools: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
	"""Return, per size, the largest target every layout can reach exactly."""
	if set(active_pools) != set(DATA_SIZES):
		raise ValueError(f'active pools must define exactly {list(DATA_SIZES)!r}')
	targets: dict[str, int] = {}
	for data_size in DATA_SIZES:
		pools = active_pools[data_size]
		if set(pools) != set(LAYOUT_IDS):
			raise ValueError(
				f'{data_size} active pools must define exactly {list(LAYOUT_IDS)!r}'
			)
		targets[data_size] = min(
			_positive_integer(value, f'{data_size} active pool {layout_id}')
			for layout_id, value in pools.items()
		)
	if not targets['small'] < targets['medium'] < targets['large']:
		raise ValueError('targets must strictly increase small < medium < large')
	return targets


def build_section_layout_contract(  # noqa: C901, PLR0913
	layouts: Sequence[LayoutLines],
	target_train_voxel_counts: Mapping[str, int],
	previews: Sequence[SelectionPreview],
	*,
	allowed_relative_error: float,
	validation_identity: Mapping[str, object],
	source_file_identities: Mapping[str, Mapping[str, str]],
	target_calibration: Mapping[str, object],
	selection_semantics: str = SELECTION_SEMANTICS,
	class_balanced_selection: Mapping[str, object] | None = None,
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
		nested = (
			selected_by_size['small'] < selected_by_size['medium']
			and selected_by_size['medium'] < selected_by_size['large']
			if selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS
			else selected_by_size['small'] <= selected_by_size['medium']
			and selected_by_size['medium'] <= selected_by_size['large']
		)
		if not nested:
			raise ValueError(
				f'{layout.layout_id} selected tokens do not satisfy '
				f'{NESTING_SEMANTICS!r}'
			)
		sizes: dict[str, object] = {}
		for data_size in DATA_SIZES:
			preview = preview_map[(layout.layout_id, data_size)]
			if preview.target_train_voxel_count != targets[data_size]:
				raise ValueError(
					f'{layout.layout_id}/{data_size} preview target differs from '
					'the calibrated target'
				)
			_validate_preview_gate(preview, allowed_relative_error=tolerance)
			size_payload: dict[str, object] = {
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
			if selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
				size_payload.update({
					'subsample_seed': preview.subsample_seed,
					'per_class_token_row_cap': preview.per_class_token_row_cap,
					'selected_token_row_count': preview.selected_token_row_count,
					'selected_token_row_identity_sha256': (
						preview.selected_token_row_identity_sha256
					),
					'per_class_selected_token_row_counts': dict(
						preview.per_class_selected_token_row_counts or {}
					),
					'active_pool_per_class_token_row_counts': dict(
						preview.active_pool_per_class_token_row_counts or {}
					),
					'per_line_selected_token_row_counts': dict(
						preview.per_line_selected_token_row_counts or {}
					),
				})
			sizes[data_size] = size_payload
		contract_layouts.append(
			{
				'layout_id': layout.layout_id,
				'ordered_inlines': list(layout.ordered_inlines),
				'ordered_crosslines': list(layout.ordered_crosslines),
				'sizes': sizes,
			}
		)
	payload: dict[str, object] = {
		'artifact_type': CONTRACT_ARTIFACT_TYPE,
		'schema_version': CONTRACT_SCHEMA_VERSION,
		'selection_semantics': selection_semantics,
		'statistical_unit': STATISTICAL_UNIT,
		'nesting_semantics': NESTING_SEMANTICS,
		'validation_mask_semantics': VALIDATION_MASK_SEMANTICS,
		'patch_size': list(PATCH_SIZE),
		'patch_size_xyz': list(PATCH_SIZE),
		'allowed_relative_error': tolerance,
		'target_train_voxel_counts': targets,
		'target_calibration': json.loads(json.dumps(dict(target_calibration))),
		'active_prefix_counts': ACTIVE_PREFIX_COUNTS,
		'decoder_seed': DECODER_SEED,
		'decoder': json.loads(json.dumps(dict(FIXED_DECODER_CONTRACT))),
		'layouts': contract_layouts,
		'validation_identity': dict(validation_identity),
		'source_file_identities': {
			key: dict(value) for key, value in source_file_identities.items()
		},
	}
	if selection_semantics == SELECTION_SEMANTICS:
		payload['stable_selection_semantics'] = SELECTION_SEMANTICS
	elif selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
		if class_balanced_selection is None:
			raise ValueError('class_balanced_selection provenance is required')
		payload['class_balanced_selection'] = json.loads(
			json.dumps(dict(class_balanced_selection))
		)
	else:
		raise ValueError(f'unsupported selection semantics {selection_semantics!r}')
	f3_lithology_voxel_section_layout_contract_from_mapping(payload)
	return payload


def run_section_layout_calibration(  # noqa: C901, PLR0912, PLR0915
	config: F3SectionLayoutCalibrationConfig,
	*,
	mode: str,
	dry_run: bool,
) -> Mapping[str, object]:
	"""Execute inspect or finalize without ever reading model artifacts or metrics."""
	if mode not in {'inspect', 'finalize'}:
		raise ValueError("mode must be exactly 'inspect' or 'finalize'")
	grid = _load_integer_array(
		config.canonical_split_grid, label='canonical split grid'
	)
	labels = _load_integer_array(config.label_volume, label='label volume')
	lines = load_section_lines(config.line_inventory, config.segy_geometry_json)
	token_row_pool: SectionTokenRowPool | None = None
	if config.selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS:
		if config.reference_valid_tokens is None or config.tokenization_policy is None:
			raise AssertionError('class-balanced selection inputs were not resolved')
		valid_tokens = _load_bool_array(
			config.reference_valid_tokens, label='reference valid-token mask'
		)
		records = load_f3_slice_split_records(config.line_inventory)
		geometry = read_f3_line_geometry(config.segy_geometry_json)
		token_row_pool = build_section_token_row_pool(
			records,
			label_volume=labels,
			valid_tokens=valid_tokens,
			geometry=geometry,
			patch_size_xyz=config.patch_size_xyz,
			policy=F3LithologyTokenPolicy(
				min_labeled_fraction=float(
					config.tokenization_policy['min_labeled_fraction']
				),
				min_majority_fraction=float(
					config.tokenization_policy['min_majority_fraction']
				),
				ignore_z_border_samples=int(
					config.tokenization_policy['ignore_z_border_samples']
				),
			),
		)
	candidates = inspect_section_candidates(
		grid, labels, lines, patch_size_xyz=config.patch_size_xyz
	)
	candidate_rows = [item.to_dict() for item in candidates]
	if token_row_pool is not None:
		counts_by_line: dict[tuple[str, int], dict[str, int]] = {}
		for line in lines:
			counts_by_line[line.key] = {
				str(class_id): sum(
					row.line_key == line.key and row.class_id == class_id
					for row in token_row_pool.train_rows
				)
				for class_id in CLASS_IDS
			}
		for candidate, row in zip(candidates, candidate_rows, strict=True):
			counts = counts_by_line[candidate.line.key]
			row['per_class_token_row_counts_after_validation_precedence'] = counts
			row['token_row_count_after_validation_precedence'] = sum(counts.values())
	candidate_payload = {
		'artifact_type': CANDIDATE_ARTIFACT_TYPE,
		'selection_semantics': config.selection_semantics,
		'patch_size_xyz': list(config.patch_size_xyz),
		'target_rule': config.target_rule,
		'rows': candidate_rows,
		'source_file_identities': _input_identities(config, include_layout=False),
	}
	if token_row_pool is not None:
		candidate_payload['token_row_pool_provenance'] = token_row_pool.provenance()
	if mode == 'inspect':
		if not dry_run:
			_write_candidate_outputs(config, candidate_payload)
		return candidate_payload
	layouts = validate_layout_lines(load_config(config.layout_lines), candidates)
	active_pools = active_pool_train_voxel_counts(layouts, grid, labels, lines)
	if config.target_rule == CLASS_CAP_TARGET_RULE:
		if (
			token_row_pool is None
			or config.target_train_voxel_counts is None
			or config.per_class_token_row_caps is None
			or config.layout_subsample_seeds is None
			or config.tokenization_policy is None
		):
			raise AssertionError('class-cap selection configuration was lost')
		targets = dict(config.target_train_voxel_counts)
		previews = tuple(
			preview
			for layout in layouts
			for preview in preview_class_balanced_layout_selection(
				layout,
				token_row_pool,
				grid,
				labels,
				lines,
				subsample_seed=config.layout_subsample_seeds[layout.layout_id],
				per_class_token_row_caps=config.per_class_token_row_caps,
				target_train_voxel_counts=targets,
				patch_size_xyz=config.patch_size_xyz,
				allowed_relative_error=config.allowed_relative_error,
			)
		)
		preview_by_key = {
			(preview.layout_id, preview.data_size): preview for preview in previews
		}
		active_token_pools = {
			size: {
				layout_id: dict(cast(
					'Mapping[str, int]',
					preview_by_key[
						(layout_id, size)
					].active_pool_per_class_token_row_counts,
				))
				for layout_id in LAYOUT_IDS
			}
			for size in DATA_SIZES
		}
		target_calibration = {
			'rule': config.target_rule,
			'nominal_target_train_voxel_counts': targets,
			'active_pool_token_row_counts': active_token_pools,
		}
		class_balanced_selection: Mapping[str, object] | None = {
			'per_class_token_row_caps': dict(config.per_class_token_row_caps),
			'layout_subsample_seeds': dict(config.layout_subsample_seeds),
			'tokenization_policy': dict(config.tokenization_policy),
			'validation_precedence': TOKEN_ROW_VALIDATION_PRECEDENCE,
			'voxel_materialization': TOKEN_ROW_VOXEL_MATERIALIZATION,
			'token_row_pool_provenance': token_row_pool.provenance(),
		}
	elif config.target_rule == TARGET_RULE:
		targets = calibrate_target_train_voxel_counts(active_pools)
		previews = tuple(
			preview
			for layout in layouts
			for preview in preview_nested_selection(
				layout,
				targets,
				grid,
				labels,
				lines,
				patch_size_xyz=config.patch_size_xyz,
				allowed_relative_error=config.allowed_relative_error,
			)
		)
		target_calibration = {
			'rule': config.target_rule,
			'active_pool_train_voxel_counts': active_pools,
		}
		class_balanced_selection = None
	else:
		if config.target_train_voxel_counts is None:
			raise AssertionError('fixed target rule lost its configured targets')
		targets = dict(config.target_train_voxel_counts)
		for data_size in DATA_SIZES:
			minimum_pool = min(active_pools[data_size].values())
			if targets[data_size] > minimum_pool:
				raise ValueError(
					f'{data_size} fixed target {targets[data_size]} exceeds '
					f'minimum active pool {minimum_pool}'
				)
		previews = tuple(
			preview
			for layout in layouts
			for preview in preview_nested_selection(
				layout,
				targets,
				grid,
				labels,
				lines,
				patch_size_xyz=config.patch_size_xyz,
				allowed_relative_error=config.allowed_relative_error,
			)
		)
		target_calibration = {
			'rule': config.target_rule,
			'active_pool_train_voxel_counts': active_pools,
		}
		target_calibration['fixed_target_train_voxel_counts'] = targets
		class_balanced_selection = None
	contract = build_section_layout_contract(
		layouts,
		targets,
		previews,
		allowed_relative_error=config.allowed_relative_error,
		validation_identity=_validation_identity(grid, config.canonical_split_grid),
		source_file_identities=_input_identities(config, include_layout=True),
		target_calibration=target_calibration,
		selection_semantics=config.selection_semantics,
		class_balanced_selection=class_balanced_selection,
	)
	if not dry_run:
		_write_json_new(config.canonical_contract, contract)
	return contract


def _unique_lines(
	lines: Sequence[SectionLine], *, shape: tuple[int, ...]
) -> tuple[SectionLine, ...]:
	seen: set[tuple[str, int]] = set()
	for line in lines:
		if line.slice_type not in {'inline', 'crossline'}:
			raise ValueError(f'unsupported slice_type {line.slice_type!r}')
		axis = 0 if line.slice_type == 'inline' else 1
		if not 0 <= line.array_index < shape[axis]:
			raise ValueError(f'{line.key!r} array index is outside volume')
		if line.key in seen:
			raise ValueError(f'line inventory contains duplicate line {line.key!r}')
		seen.add(line.key)
	return tuple(lines)


def _plane_index(line: SectionLine) -> tuple[object, object, object]:
	if line.slice_type == 'inline':
		return line.array_index, slice(None), slice(None)
	return slice(None), line.array_index, slice(None)


def _plane(volume: NDArray[np.integer], line: SectionLine) -> NDArray[np.integer]:
	return cast('NDArray[np.integer]', volume[_plane_index(line)])


def _plane_token_count(
	teacher_plane: NDArray[np.bool_], *, line: SectionLine, patch: tuple[int, int, int]
) -> int:
	coordinates = np.argwhere(teacher_plane)
	if not coordinates.size:
		return 0
	if line.slice_type == 'inline':
		steps = np.asarray(patch[1:], dtype=np.int64)
	else:
		steps = np.asarray((patch[0], patch[2]), dtype=np.int64)
	return int(np.unique(coordinates // steps, axis=0).shape[0])


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
	if any(
		preview.per_class_voxel_counts[str(class_id)] <= 0
		for class_id in MONITORED_CLASS_IDS
	):
		raise ValueError(
			f'{preview.layout_id}/{preview.data_size} monitored classes '
			f'{list(MONITORED_CLASS_IDS)!r} must be nonzero'
		)
	expected_lines = {
		*(f'inline:{line}' for line in preview.inline_lines),
		*(f'crossline:{line}' for line in preview.crossline_lines),
	}
	if set(preview.per_line_contributions) != expected_lines:
		raise ValueError(
			f'{preview.layout_id}/{preview.data_size} per-line contributions '
			'do not match the active lines'
		)
	zero_lines = [
		key for key, count in preview.per_line_contributions.items() if count <= 0
	]
	if zero_lines:
		raise ValueError(
			f'{preview.layout_id}/{preview.data_size} active lines contribute '
			f'zero teacher voxels: {zero_lines!r}'
		)


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
		token_row_counts = cast(
			'Mapping[str, int] | None',
			row.get('per_class_token_row_counts_after_validation_precedence'),
		)
		flat = {
			key: value
			for key, value in row.items()
			if key
			not in {
				'per_class_voxel_counts',
				'per_class_token_row_counts_after_validation_precedence',
			}
		}
		flat.update(
			{
				f'class_{class_id}_voxel_count': counts[str(class_id)]
				for class_id in CLASS_IDS
			}
		)
		if token_row_counts is not None:
			flat.update({
				f'class_{class_id}_token_row_count': token_row_counts[str(class_id)]
				for class_id in CLASS_IDS
			})
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
			json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
			handle.write('\n')
		Path(temporary).replace(path)
	except BaseException:
		Path(temporary).unlink(missing_ok=True)
		raise


def _load_integer_array(path: Path, *, label: str) -> NDArray[np.integer]:
	array = np.load(path, mmap_mode='r', allow_pickle=False)
	if not np.issubdtype(array.dtype, np.integer) or array.dtype == np.dtype(np.bool_):
		raise TypeError(f'{label} must use an integer dtype')
	if array.ndim != 3:
		raise ValueError(f'{label} must be a 3D volume')
	return cast('NDArray[np.integer]', array)


def _load_bool_array(path: Path, *, label: str) -> NDArray[np.bool_]:
	array = np.load(path, mmap_mode='r', allow_pickle=False)
	if array.dtype != np.dtype(np.bool_):
		raise TypeError(f'{label} must use bool dtype')
	if array.ndim != 3:
		raise ValueError(f'{label} must be a 3D array')
	return cast('NDArray[np.bool_]', array)


def _input_identities(
	config: F3SectionLayoutCalibrationConfig, *, include_layout: bool
) -> dict[str, dict[str, str]]:
	paths = {
		'canonical_split_grid': config.canonical_split_grid,
		'label_volume': config.label_volume,
		'line_inventory': config.line_inventory,
		'segy_geometry_json': config.segy_geometry_json,
	}
	if config.reference_valid_tokens is not None:
		paths['reference_valid_tokens'] = config.reference_valid_tokens
	if include_layout:
		paths['layout_lines'] = config.layout_lines
	return {
		key: {'path': str(path), 'sha256': file_sha256(path)}
		for key, path in paths.items()
	}


def _sequence_sha256(values: Sequence[object]) -> str:
	digest = hashlib.sha256()
	for value in values:
		encoded = json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode()
		digest.update(len(encoded).to_bytes(8, byteorder='big'))
		digest.update(encoded)
	return digest.hexdigest()


def _volume_pair(
	grid: NDArray[np.integer], labels: NDArray[np.integer]
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
	for array, name in ((grid, 'canonical_split_grid'), (labels, 'label_volume')):
		if (
			array.ndim != 3
			or not np.issubdtype(array.dtype, np.integer)
			or array.dtype == np.dtype(np.bool_)
		):
			raise TypeError(f'{name} must be a 3D integer array')
	if grid.shape != labels.shape:
		raise ValueError('canonical split grid and label volume shapes must match')
	return grid, labels


def _class_ids(value: Sequence[int]) -> tuple[int, ...]:
	result = tuple(_integer(item, 'class_ids entry') for item in value)
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


def _configured_target_counts(value: object) -> dict[str, int]:
	return _configured_size_counts(value, label='targets.train_voxel_counts')


def _configured_size_counts(value: object, *, label: str) -> dict[str, int]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	if set(value) != set(DATA_SIZES):
		raise ValueError(f'{label} must define exactly {list(DATA_SIZES)!r}')
	result = {
		size: _positive_integer(value[size], f'{label}.{size}')
		for size in DATA_SIZES
	}
	if not result['small'] < result['medium'] < result['large']:
		raise ValueError(f'{label} must strictly increase small < medium < large')
	return result


def _configured_layout_seeds(value: object) -> dict[str, int]:
	label = 'selection.layout_subsample_seeds'
	if not isinstance(value, Mapping) or set(value) != set(LAYOUT_IDS):
		raise ValueError(f'{label} must define exactly {list(LAYOUT_IDS)!r}')
	result = {
		layout_id: _nonnegative_integer(value[layout_id], f'{label}.{layout_id}')
		for layout_id in LAYOUT_IDS
	}
	if len(set(result.values())) != len(result):
		raise ValueError(f'{label} must contain unique seeds')
	for layout_id, seed in result.items():
		if seed != int(layout_id.removeprefix('layout_')):
			raise ValueError(f'{label}.{layout_id} must equal its layout suffix')
	return result


def _configured_tokenization_policy(value: object) -> dict[str, object]:
	label = 'selection.tokenization_policy'
	if not isinstance(value, Mapping) or set(value) != set(TOKENIZATION_POLICY):
		raise ValueError(
			f'{label} must define exactly {list(TOKENIZATION_POLICY)!r}'
		)
	resolved: dict[str, object] = {
		'min_labeled_fraction': _fraction(
			value['min_labeled_fraction'], f'{label}.min_labeled_fraction'
		),
		'min_majority_fraction': _fraction(
			value['min_majority_fraction'], f'{label}.min_majority_fraction'
		),
		'ignore_z_border_samples': _nonnegative_integer(
			value['ignore_z_border_samples'], f'{label}.ignore_z_border_samples'
		),
	}
	if resolved != dict(TOKENIZATION_POLICY):
		raise ValueError(f'{label} must match the fixed v1 tokenization policy')
	return resolved


def _line_number_list(value: object, label: str) -> tuple[int, ...]:
	expected = LINE_COUNTS['large'][0]
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
	if not np.isfinite(result) or not 0.0 < result <= 0.1:
		raise ValueError('allowed_relative_error must be in (0, 0.1]')
	return result


def _integer(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
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


def _fraction(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{label} must be a number')
	result = float(value)
	if not np.isfinite(result) or not 0.0 <= result <= 1.0:
		raise ValueError(f'{label} must be in [0, 1]')
	return result


__all__ = [
	'ACTIVE_PREFIX_COUNTS',
	'CANDIDATE_ARTIFACT_TYPE',
	'CLASS_CAP_TARGET_RULE',
	'FIXED_TARGET_RULE',
	'MONITORED_CLASS_IDS',
	'TARGET_RULE',
	'F3SectionLayoutCalibrationConfig',
	'SectionCandidate',
	'SectionTokenRowPool',
	'active_pool_train_voxel_counts',
	'build_section_layout_contract',
	'build_section_token_row_pool',
	'calibrate_target_train_voxel_counts',
	'f3_section_layout_calibration_config_from_mapping',
	'inspect_section_candidates',
	'load_section_lines',
	'preview_class_balanced_layout_selection',
	'run_section_layout_calibration',
	'validate_layout_lines',
]
