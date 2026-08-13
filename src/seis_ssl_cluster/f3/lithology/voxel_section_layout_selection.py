"""Pure selection kernel for F3 section-layout voxel supervision."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
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
		result.append(TokenFootprint(coordinate, flat, per_line))
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
	'CLASS_IDS',
	'SELECTION_SEMANTICS',
	'LayoutLines',
	'SectionLine',
	'SelectionPreview',
	'TokenFootprint',
	'candidate_token_footprints',
	'per_line_contributions',
	'preview_nested_selection',
	'replay_selected_teacher_mask',
	'stable_token_order',
]
