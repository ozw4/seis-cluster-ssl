"""Pure selection kernel for F3 section-layout voxel supervision."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
	LINE_COUNTS,
	PATCH_SIZE,
)
from seis_ssl_cluster.f3.lithology.voxel_split import TRAIN_VOXEL_SPLIT

if TYPE_CHECKING:
	from numpy.typing import NDArray

CLASS_IDS = tuple(range(6))
SELECTION_SEMANTICS = 'stable_hash_partial_section_token_footprints_v1'
CLASS_BALANCED_SELECTION_SEMANTICS = (
	'seeded_nested_class_balanced_section_token_rows_v1'
)

TokenRowIdentity = tuple[str, int, int, int, int, int]


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
class LayoutLines:
	"""The user-selected ordered four-inline/four-crossline layout."""

	layout_id: str
	ordered_inlines: tuple[int, ...]
	ordered_crosslines: tuple[int, ...]


@dataclass(frozen=True)
class TokenFootprint:
	"""Teacher voxels in one token block, with stable per-line ownership."""

	token_xyz: tuple[int, int, int]
	flat_voxel_indices: tuple[int, ...]
	per_line_flat_voxel_indices: Mapping[tuple[str, int], tuple[int, ...]]
	per_class_voxel_counts: Mapping[str, int] | None = None

	@property
	def voxel_count(self) -> int:
		"""Return the exact partial-plane teacher voxel count."""
		return len(self.flat_voxel_indices)

	def line_voxel_count(self, line_key: tuple[str, int]) -> int:
		"""Return teacher voxels owned by one line under ordered attribution."""
		return len(self.per_line_flat_voxel_indices.get(line_key, ()))


@dataclass(frozen=True)
class TokenRow:
	"""One v1-style labeled section/token row before token deduplication."""

	slice_type: str
	slice_index: int
	token_xyz: tuple[int, int, int]
	class_id: int

	@property
	def line_key(self) -> tuple[str, int]:
		"""Return the physical section identity that produced this row."""
		return (self.slice_type, self.slice_index)

	@property
	def identity(self) -> TokenRowIdentity:
		"""Return the canonical v1-style row identity."""
		x, y, z = self.token_xyz
		return (self.slice_type, self.slice_index, x, y, z, self.class_id)


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
	selection_semantics: str = SELECTION_SEMANTICS
	subsample_seed: int | None = None
	per_class_token_row_cap: int | None = None
	selected_token_row_count: int | None = None
	selected_token_row_identity_sha256: str | None = None
	per_class_selected_token_row_counts: Mapping[str, int] | None = None
	active_pool_per_class_token_row_counts: Mapping[str, int] | None = None
	per_line_selected_token_row_counts: Mapping[str, int] | None = None
	selected_token_row_identities: tuple[TokenRowIdentity, ...] = ()
	selected_token_identity_sha256: str | None = None
	selected_voxel_identity_sha256: str | None = None


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
		local = np.argwhere(teacher[block])
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
		flat_labels = labels.reshape(-1)[np.asarray(flat, dtype=np.int64)]
		per_class = {
			str(class_id): int(np.count_nonzero(flat_labels == class_id))
			for class_id in CLASS_IDS
		}
		result.append(TokenFootprint(coordinate, flat, per_line, per_class))
	return tuple(result)


def stable_token_order(
	footprints: Sequence[TokenFootprint],
	*,
	layout_id: str,
	semantics_version: str = SELECTION_SEMANTICS,
) -> tuple[TokenFootprint, ...]:
	"""Order token coordinates by explicit SHA-256, never RNG or hash()."""
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
	line_inventory: Sequence[SectionLine | Mapping[str, object]],
	*,
	patch_size_xyz: Sequence[int] = PATCH_SIZE,
	allowed_relative_error: float = 0.05,
) -> tuple[SelectionPreview, ...]:
	"""Build coverage-first nested small/medium/large selection previews."""
	grid, labels = _volume_pair(canonical_split_grid, label_volume)
	patch = _positive_triplet(patch_size_xyz, 'patch_size_xyz')
	tolerance = _relative_error(allowed_relative_error)
	targets = _target_counts(target_train_voxel_counts)
	line_map = _line_map(line_inventory, shape=grid.shape)
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
			per_line_contributions=per_line_contributions(
				selected_footprints, active
			),
			per_class_voxel_counts=_selected_class_counts(labels, selected_voxels),
		)
		_validate_preview_gate(preview, allowed_relative_error=tolerance)
		previews.append(preview)
	return tuple(previews)


def preview_seeded_nested_class_cap_selection(  # noqa: C901, PLR0912, PLR0913, PLR0915
	layout: LayoutLines,
	*,
	subsample_seed: int,
	per_class_token_row_caps: Mapping[str, int],
	target_train_voxel_counts: Mapping[str, int],
	token_rows_by_size: Mapping[str, Sequence[TokenRow]],
	token_footprints_by_size: Mapping[str, Sequence[TokenFootprint]],
	allowed_relative_error: float = 0.05,
) -> tuple[SelectionPreview, ...]:
	"""Select exact nested per-class token-row caps for one layout.

	The caller supplies pools after applying validation-token precedence. Rows
	remain distinct by physical section even when several rows share one token
	coordinate. Token coordinates are deduplicated only after row selection.
	Input row order is never scientific identity: each pool is normalized to
	lexicographic ``TokenRow.identity`` order before a single per-layout RNG is
	consumed in ascending class order for small, medium, then large.
	"""
	seed = _layout_seed(layout.layout_id, subsample_seed)
	caps = _class_cap_counts(per_class_token_row_caps)
	targets = _target_counts(target_train_voxel_counts)
	tolerance = _relative_error(allowed_relative_error)
	for data_size in DATA_SIZES:
		nominal = len(CLASS_IDS) * caps[data_size] * PATCH_SIZE[1] * PATCH_SIZE[2]
		if targets[data_size] != nominal:
			raise ValueError(
				f'target_train_voxel_counts.{data_size} must equal six classes * '
				f'cap * 8 * 8 ({nominal})'
			)
	rows_by_size = _token_row_pools(token_rows_by_size)
	footprints_by_size = _token_footprint_pools(token_footprints_by_size)
	_validate_token_row_pool_nesting(rows_by_size)
	rng = np.random.default_rng(seed)
	selected_by_class: dict[int, set[TokenRowIdentity]] = {
		class_id: set() for class_id in CLASS_IDS
	}
	previous_rows: set[TokenRowIdentity] = set()
	previous_tokens: set[tuple[int, int, int]] = set()
	previous_voxels: set[int] = set()
	previews: list[SelectionPreview] = []
	for data_size in DATA_SIZES:
		inline_count, crossline_count = LINE_COUNTS[data_size]
		inline_lines = layout.ordered_inlines[:inline_count]
		crossline_lines = layout.ordered_crosslines[:crossline_count]
		if len(inline_lines) != inline_count or len(crossline_lines) != crossline_count:
			raise ValueError(
				f'{layout.layout_id}/{data_size} layout does not define enough lines'
			)
		line_keys = tuple(('inline', line) for line in inline_lines) + tuple(
			('crossline', line) for line in crossline_lines
		)
		if len(line_keys) != len(set(line_keys)):
			raise ValueError(
				f'{layout.layout_id}/{data_size} active lines contain duplicates'
			)
		pool = rows_by_size[data_size]
		escaped_lines = {row.line_key for row in pool} - set(line_keys)
		if escaped_lines:
			raise ValueError(
				f'{layout.layout_id}/{data_size} token rows escaped active lines: '
				f'{sorted(escaped_lines, key=_line_key_sort_key)!r}'
			)
		row_by_identity = {row.identity: row for row in pool}
		per_class_pool = {
			class_id: tuple(row for row in pool if row.class_id == class_id)
			for class_id in CLASS_IDS
		}
		for class_id in CLASS_IDS:
			class_pool = per_class_pool[class_id]
			cap = caps[data_size]
			if len(class_pool) < cap:
				raise ValueError(
					f'{layout.layout_id}/{data_size} class {class_id} token-row pool '
					f'{len(class_pool)} is below cap {cap} after validation '
					'precedence'
				)
			pool_identities = {row.identity for row in class_pool}
			retained = selected_by_class[class_id]
			if not retained <= pool_identities:
				raise ValueError(
					f'{layout.layout_id}/{data_size} class {class_id} active pool lost '
					'previously selected token rows'
				)
			needed = cap - len(retained)
			if needed < 0:
				raise ValueError(
					f'{layout.layout_id}/{data_size} class {class_id} cap {cap} is '
					'smaller than the retained nested selection'
				)
			available = tuple(
				row for row in class_pool if row.identity not in retained
			)
			if needed == len(available):
				# Match the v1 cap sampler's full-pool boundary: taking every
				# available row is deterministic and must not advance the RNG.
				retained.update(row.identity for row in available)
			elif needed:
				chosen = rng.choice(len(available), size=needed, replace=False)
				retained.update(
					available[int(index)].identity for index in np.atleast_1d(chosen)
				)
			if len(retained) != cap:
				raise AssertionError('per-class token-row selection missed its cap')
		selected_identities = tuple(
			sorted(
				(
					identity
					for identities in selected_by_class.values()
					for identity in identities
				),
				key=_token_row_identity_sort_key,
			)
		)
		selected_rows = tuple(
			row_by_identity[identity] for identity in selected_identities
		)
		selected_tokens = tuple(
			sorted({row.token_xyz for row in selected_rows})
		)
		footprint_by_xyz = footprints_by_size[data_size]
		missing_footprints = set(selected_tokens) - set(footprint_by_xyz)
		if missing_footprints:
			raise ValueError(
				f'{layout.layout_id}/{data_size} selected token rows lack footprints: '
				f'{sorted(missing_footprints)!r}'
			)
		selected_footprints = tuple(
			footprint_by_xyz[xyz] for xyz in selected_tokens
		)
		selected_voxels = tuple(
			sorted({
				flat
				for footprint in selected_footprints
				for flat in footprint.flat_voxel_indices
			})
		)
		current_rows = set(selected_identities)
		current_tokens = set(selected_tokens)
		current_voxels = set(selected_voxels)
		if previews:
			_validate_strict_nested_identity(
				previous_rows,
				current_rows,
				label=f'{layout.layout_id}/{data_size} selected token rows',
			)
			_validate_strict_nested_identity(
				previous_tokens,
				current_tokens,
				label=f'{layout.layout_id}/{data_size} selected token_xyz',
			)
			_validate_strict_nested_identity(
				previous_voxels,
				current_voxels,
				label=f'{layout.layout_id}/{data_size} selected teacher voxels',
			)
		previous_rows = current_rows
		previous_tokens = current_tokens
		previous_voxels = current_voxels
		line_row_counts = {
			_line_key_string(key): sum(row.line_key == key for row in selected_rows)
			for key in line_keys
		}
		line_voxel_counts = per_line_contributions(
			selected_footprints,
			tuple(
				SectionLine(
					slice_type=key[0],
					slice_index=key[1],
					array_index=0,
					is_validation_line=False,
				)
				for key in line_keys
			),
		)
		zero_row_lines = [
			key for key, count in line_row_counts.items() if count <= 0
		]
		if zero_row_lines:
			raise ValueError(
				f'{layout.layout_id}/{data_size} active lines have zero selected '
				f'token rows: {zero_row_lines!r}'
			)
		zero_voxel_lines = [
			key for key, count in line_voxel_counts.items() if count <= 0
		]
		if zero_voxel_lines:
			raise ValueError(
				f'{layout.layout_id}/{data_size} active lines contribute zero '
				f'teacher voxels: {zero_voxel_lines!r}'
			)
		per_class_selected = {
			str(class_id): sum(row.class_id == class_id for row in selected_rows)
			for class_id in CLASS_IDS
		}
		if any(value != caps[data_size] for value in per_class_selected.values()):
			raise AssertionError('selected token-row class counts differ from cap')
		per_class_voxels = _selected_footprint_class_counts(selected_footprints)
		if sum(per_class_voxels.values()) != len(selected_voxels):
			raise AssertionError(
				'selected token footprint class counts do not cover teacher voxels'
			)
		missing_classes = [
			key for key, count in per_class_voxels.items() if count <= 0
		]
		if missing_classes:
			raise ValueError(
				f'{layout.layout_id}/{data_size} is missing dense voxel classes '
				f'{missing_classes!r}'
			)
		actual = len(selected_voxels)
		error = actual - targets[data_size]
		relative_error = abs(error) / targets[data_size]
		if relative_error > tolerance + 1e-15:
			raise ValueError(
				f'{layout.layout_id}/{data_size} target relative error '
				f'{relative_error:.6g} exceeds {tolerance:.6g}'
			)
		previews.append(
			SelectionPreview(
				layout_id=layout.layout_id,
				data_size=data_size,
				inline_lines=inline_lines,
				crossline_lines=crossline_lines,
				target_train_voxel_count=targets[data_size],
				actual_train_voxel_count=actual,
				count_error=error,
				relative_count_error=relative_error,
				selected_token_xyz=selected_tokens,
				selected_flat_voxel_indices=selected_voxels,
				per_line_contributions=line_voxel_counts,
				per_class_voxel_counts=per_class_voxels,
				selection_semantics=CLASS_BALANCED_SELECTION_SEMANTICS,
				subsample_seed=seed,
				per_class_token_row_cap=caps[data_size],
				selected_token_row_count=len(selected_rows),
				selected_token_row_identity_sha256=_identity_sha256(
					selected_identities
				),
				per_class_selected_token_row_counts=per_class_selected,
				active_pool_per_class_token_row_counts={
					str(class_id): len(per_class_pool[class_id])
					for class_id in CLASS_IDS
				},
				per_line_selected_token_row_counts=line_row_counts,
				selected_token_row_identities=selected_identities,
				selected_token_identity_sha256=_integer_array_sha256(
					selected_tokens
				),
				selected_voxel_identity_sha256=_integer_array_sha256(selected_voxels),
			)
		)
	return tuple(previews)


def replay_selected_teacher_mask(  # noqa: PLR0913
	canonical_split_grid: NDArray[np.integer],
	label_volume: NDArray[np.integer],
	active_lines: Sequence[SectionLine | Mapping[str, object]],
	selected_token_xyz: Sequence[Sequence[int]],
	*,
	patch_size_xyz: Sequence[int] = PATCH_SIZE,
	class_ids: Sequence[int] = CLASS_IDS,
) -> NDArray[np.bool_]:
	"""Replay selected partial footprints into one exact teacher-voxel mask."""
	grid, labels = _volume_pair(canonical_split_grid, label_volume)
	selected = tuple(
		_token_triplet(item, 'selected_token_xyz entry')
		for item in selected_token_xyz
	)
	if len(selected) != len(set(selected)):
		raise ValueError('selected_token_xyz must not contain duplicates')
	footprints = candidate_token_footprints(
		grid,
		labels,
		active_lines,
		patch_size_xyz=patch_size_xyz,
		class_ids=class_ids,
	)
	by_xyz = {item.token_xyz: item for item in footprints}
	unknown = set(selected) - set(by_xyz)
	if unknown:
		raise ValueError(
			f'selected tokens are not candidate footprints: {sorted(unknown)!r}'
		)
	mask = np.zeros(grid.shape, dtype=np.bool_)
	flat = mask.reshape(-1)
	for coordinate in selected:
		flat[np.asarray(by_xyz[coordinate].flat_voxel_indices, dtype=np.int64)] = True
	return mask


def per_line_contributions(
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


def _line_map(
	lines: Sequence[SectionLine | Mapping[str, object]], *, shape: tuple[int, ...]
) -> dict[tuple[str, int], SectionLine]:
	result: dict[tuple[str, int], SectionLine] = {}
	for item in lines:
		line = _coerce_section_line(item, shape)
		if line.key in result:
			raise ValueError(f'line inventory contains duplicate {line.key!r}')
		result[line.key] = line
	return result


def _coerce_section_line(
	item: SectionLine | Mapping[str, object], shape: tuple[int, ...]
) -> SectionLine:
	if isinstance(item, Mapping):
		slice_type = item.get('slice_type')
		slice_index = item.get('slice_index')
		array_index = item.get('array_index')
		is_validation = item.get('is_validation_line', False)
		if slice_type not in {'inline', 'crossline'}:
			raise ValueError(f'unsupported slice_type {slice_type!r}')
		if not isinstance(is_validation, bool):
			raise TypeError('active line is_validation_line must be boolean')
		line = SectionLine(
			cast('str', slice_type),
			_positive_integer(slice_index, 'active line slice_index'),
			_nonnegative_integer(array_index, 'active line array_index'),
			is_validation,
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


def _ordered_line_voxel_ownership(
	voxel_xyz: NDArray[np.integer],
	*,
	flat_voxel_indices: Sequence[int],
	lines: Sequence[SectionLine],
) -> Mapping[tuple[str, int], tuple[int, ...]]:
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
		if after >= target:
			if abs(target - after) < before_error:
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


def _selected_footprint_class_counts(
	selected: Sequence[TokenFootprint],
) -> dict[str, int]:
	result = {str(class_id): 0 for class_id in CLASS_IDS}
	for footprint in selected:
		counts = footprint.per_class_voxel_counts
		if counts is None:
			raise AssertionError('class-balanced footprint lacks dense class counts')
		for class_id in CLASS_IDS:
			result[str(class_id)] += counts[str(class_id)]
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


def _layout_seed(layout_id: str, value: object) -> int:
	if layout_id not in LAYOUT_IDS:
		raise ValueError(f'unsupported layout_id {layout_id!r}')
	seed = _nonnegative_integer(value, 'subsample_seed')
	expected = LAYOUT_IDS.index(layout_id)
	if seed != expected:
		raise ValueError(
			f'{layout_id} subsample_seed must be exactly {expected}; got {seed}'
		)
	return seed


def _class_cap_counts(value: Mapping[str, int]) -> dict[str, int]:
	if not isinstance(value, Mapping) or set(value) != set(DATA_SIZES):
		raise ValueError(
			'per_class_token_row_caps must define exactly '
			f'{list(DATA_SIZES)!r}'
		)
	result = {
		size: _positive_integer(
			value[size], f'per_class_token_row_caps.{size}'
		)
		for size in DATA_SIZES
	}
	if not result['small'] < result['medium'] < result['large']:
		raise ValueError(
			'per_class_token_row_caps must strictly increase '
			'small < medium < large'
		)
	return result


def _token_row_pools(
	value: Mapping[str, Sequence[TokenRow]],
) -> dict[str, tuple[TokenRow, ...]]:
	if not isinstance(value, Mapping) or set(value) != set(DATA_SIZES):
		raise ValueError(
			f'token_rows_by_size must define exactly {list(DATA_SIZES)!r}'
		)
	return {
		size: _canonical_token_rows(value[size], label=f'{size} token rows')
		for size in DATA_SIZES
	}


def _canonical_token_rows(
	value: object, *, label: str
) -> tuple[TokenRow, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a sequence of TokenRow values')
	rows: list[TokenRow] = []
	seen: set[TokenRowIdentity] = set()
	for index, item in enumerate(value):
		if not isinstance(item, TokenRow):
			raise TypeError(f'{label}[{index}] must be a TokenRow')
		if item.slice_type not in {'inline', 'crossline'}:
			raise ValueError(
				f'{label}[{index}].slice_type must be inline or crossline'
			)
		row = TokenRow(
			item.slice_type,
			_positive_integer(item.slice_index, f'{label}[{index}].slice_index'),
			_token_triplet(item.token_xyz, f'{label}[{index}].token_xyz'),
			_nonnegative_integer(item.class_id, f'{label}[{index}].class_id'),
		)
		if row.class_id not in CLASS_IDS:
			raise ValueError(
				f'{label}[{index}].class_id must be one of {list(CLASS_IDS)!r}'
			)
		if row.identity in seen:
			raise ValueError(f'{label} contains duplicate row {row.identity!r}')
		seen.add(row.identity)
		rows.append(row)
	return tuple(sorted(rows, key=_token_row_sort_key))


def _token_footprint_pools(
	value: Mapping[str, Sequence[TokenFootprint]],
) -> dict[str, dict[tuple[int, int, int], TokenFootprint]]:
	if not isinstance(value, Mapping) or set(value) != set(DATA_SIZES):
		raise ValueError(
			f'token_footprints_by_size must define exactly {list(DATA_SIZES)!r}'
		)
	result: dict[str, dict[tuple[int, int, int], TokenFootprint]] = {}
	for size in DATA_SIZES:
		items = value[size]
		if not isinstance(items, Sequence) or isinstance(items, str | bytes):
			raise TypeError(f'{size} token footprints must be a sequence')
		by_xyz: dict[tuple[int, int, int], TokenFootprint] = {}
		for index, item in enumerate(items):
			if not isinstance(item, TokenFootprint):
				raise TypeError(
					f'{size} token footprints[{index}] must be a TokenFootprint'
				)
			xyz = _token_triplet(
				item.token_xyz, f'{size} token footprints[{index}].token_xyz'
			)
			if xyz in by_xyz:
				raise ValueError(
					f'{size} token footprints contain duplicate token_xyz {xyz!r}'
				)
			flat = tuple(
				_nonnegative_integer(
					flat_index,
					f'{size} token footprints[{index}].flat_voxel_indices entry',
				)
				for flat_index in item.flat_voxel_indices
			)
			if len(flat) != len(set(flat)):
				raise ValueError(
					f'{size} token footprint {xyz!r} contains duplicate voxels'
				)
			counts = item.per_class_voxel_counts
			if not isinstance(counts, Mapping) or set(counts) != {
				str(class_id) for class_id in CLASS_IDS
			}:
				raise ValueError(
					f'{size} token footprint {xyz!r} per_class_voxel_counts '
					f'must define exactly {list(CLASS_IDS)!r}'
				)
			normalized_counts = {
				str(class_id): _nonnegative_integer(
					counts[str(class_id)],
					f'{size} token footprint {xyz!r} class {class_id} count',
				)
				for class_id in CLASS_IDS
			}
			if sum(normalized_counts.values()) != len(flat):
				raise ValueError(
					f'{size} token footprint {xyz!r} class counts do not '
					'cover flat_voxel_indices'
				)
			by_xyz[xyz] = TokenFootprint(
				xyz,
				flat,
				item.per_line_flat_voxel_indices,
				normalized_counts,
			)
		result[size] = by_xyz
	return result


def _validate_token_row_pool_nesting(
	rows_by_size: Mapping[str, Sequence[TokenRow]],
) -> None:
	identities = {
		size: {row.identity for row in rows_by_size[size]} for size in DATA_SIZES
	}
	if not (
		identities['small'] <= identities['medium']
		and identities['medium'] <= identities['large']
	):
		raise ValueError(
			'token-row pools must be nested small <= medium <= large'
		)


def _validate_strict_nested_identity(
	previous: AbstractSet[object], current: AbstractSet[object], *, label: str
) -> None:
	if not previous < current:
		raise ValueError(f'{label} must be strictly nested')


def _token_row_sort_key(
	row: TokenRow,
) -> TokenRowIdentity:
	return row.identity


def _token_row_identity_sort_key(
	identity: TokenRowIdentity,
) -> TokenRowIdentity:
	return identity


def _line_key_sort_key(value: tuple[str, int]) -> tuple[int, int]:
	return (0 if value[0] == 'inline' else 1, value[1])


def _line_key_string(value: tuple[str, int]) -> str:
	return f'{value[0]}:{value[1]}'


def _identity_sha256(values: Sequence[object]) -> str:
	digest = hashlib.sha256()
	for value in values:
		parts = value if isinstance(value, tuple) else (value,)
		encoded = '\x1f'.join(str(part) for part in parts).encode('utf-8')
		digest.update(len(encoded).to_bytes(8, byteorder='big'))
		digest.update(encoded)
	return digest.hexdigest()


def _integer_array_sha256(values: Sequence[object]) -> str:
	array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
	digest = hashlib.sha256()
	digest.update(array.dtype.str.encode('ascii'))
	digest.update(json.dumps(list(array.shape), separators=(',', ':')).encode('ascii'))
	digest.update(array.view(np.uint8))
	return digest.hexdigest()


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'{label} must be an integer triple')
	items = tuple(_positive_integer(item, f'{label} entry') for item in value)
	return cast('tuple[int, int, int]', items)


def _token_triplet(value: object, label: str) -> tuple[int, int, int]:
	array = np.asarray(value)
	if array.shape != (3,) or not np.issubdtype(array.dtype, np.integer):
		raise TypeError(f'{label} must be an integer triple')
	items = tuple(_nonnegative_integer(int(item), f'{label} entry') for item in array)
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


__all__ = [
	'CLASS_BALANCED_SELECTION_SEMANTICS',
	'CLASS_IDS',
	'SELECTION_SEMANTICS',
	'LayoutLines',
	'SectionLine',
	'SelectionPreview',
	'TokenFootprint',
	'TokenRow',
	'TokenRowIdentity',
	'candidate_token_footprints',
	'per_line_contributions',
	'preview_nested_selection',
	'preview_seeded_nested_class_cap_selection',
	'replay_selected_teacher_mask',
	'stable_token_order',
]
