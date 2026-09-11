from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import pytest

from seis_ssl_cluster.f3.lithology import voxel_section_layout_selection
from seis_ssl_cluster.f3.lithology.voxel_section_layout_selection import (
	CLASS_BALANCED_SELECTION_SEMANTICS,
	CLASS_IDS,
	LayoutLines,
	TokenFootprint,
	TokenRow,
	preview_seeded_nested_class_cap_selection,
)

CAPS = {'small': 4, 'medium': 8, 'large': 12}
TARGETS = {size: 6 * cap * 8 * 8 for size, cap in CAPS.items()}
SIZE_RANKS = {
	'small': range(8),
	'medium': range(16),
	'large': range(24),
}
EXACT_CAPS = {'small': 1, 'medium': 2, 'large': 3}
EXACT_TARGETS = {size: 6 * cap * 8 * 8 for size, cap in EXACT_CAPS.items()}
LAYOUT_000 = LayoutLines(
	'layout_000',
	(100, 101, 102, 103),
	(200, 201, 202, 203),
)


def test_seeded_selection_is_exact_deterministic_and_strictly_nested() -> None:
	rows = _row_pools()
	footprints = _footprint_pools(rows)
	first = preview_seeded_nested_class_cap_selection(
		LAYOUT_000,
		subsample_seed=0,
		per_class_token_row_caps=CAPS,
		target_train_voxel_counts=TARGETS,
		token_rows_by_size=rows,
		token_footprints_by_size=footprints,
	)
	second = preview_seeded_nested_class_cap_selection(
		LAYOUT_000,
		subsample_seed=0,
		per_class_token_row_caps=dict(reversed(tuple(CAPS.items()))),
		target_train_voxel_counts=dict(reversed(tuple(TARGETS.items()))),
		token_rows_by_size={size: tuple(reversed(rows[size])) for size in rows},
		token_footprints_by_size={
			size: tuple(reversed(footprints[size])) for size in footprints
		},
	)

	assert [item.data_size for item in first] == ['small', 'medium', 'large']
	assert [item.selected_token_row_identity_sha256 for item in first] == [
		item.selected_token_row_identity_sha256 for item in second
	]
	assert [item.selected_token_row_identity_sha256 for item in first] == [
		'9a31cb8cb35efe62fe001c098f8d0cd5c9a2829eb191869e7953df67c9cc5396',
		'1d69d709265ed9f726cc465b38a86c2e545c210018e9c38a02523fa6d6915ef0',
		'acf995767e6ccf67c0827f06e420127f0731812ae4d4384a068ee099c068b430',
	]
	for preview in first:
		assert (
			preview.selection_semantics == CLASS_BALANCED_SELECTION_SEMANTICS
		)
		assert preview.selected_token_row_count == (
			6 * preview.per_class_token_row_cap
		)
		assert set(preview.per_class_selected_token_row_counts) == {
			str(class_id) for class_id in CLASS_IDS
		}
		assert set(preview.per_class_selected_token_row_counts.values()) == {
			preview.per_class_token_row_cap
		}
		assert preview.target_train_voxel_count == (
			6 * preview.per_class_token_row_cap * 8 * 8
		)
		assert preview.actual_train_voxel_count == preview.target_train_voxel_count
		assert preview.count_error == 0
		assert preview.relative_count_error == 0.0
		assert set(preview.per_class_voxel_counts.values()) == {
			preview.per_class_token_row_cap * 8 * 8
		}
		assert preview.selected_token_row_count == len(
			preview.selected_token_row_identities
		)
		assert preview.selected_token_row_identities == tuple(
			sorted(preview.selected_token_row_identities)
		)
	assert _identities(first[0]) < _identities(first[1]) < _identities(first[2])
	assert _tokens(first[0]) < _tokens(first[1]) < _tokens(first[2])
	assert _voxels(first[0]) < _voxels(first[1]) < _voxels(first[2])

	other_seed = preview_seeded_nested_class_cap_selection(
		LayoutLines(
			'layout_001',
			LAYOUT_000.ordered_inlines,
			LAYOUT_000.ordered_crosslines,
		),
		subsample_seed=1,
		per_class_token_row_caps=CAPS,
		target_train_voxel_counts=TARGETS,
		token_rows_by_size=rows,
		token_footprints_by_size=footprints,
	)
	assert any(
		left.selected_token_row_identity_sha256
		!= right.selected_token_row_identity_sha256
		for left, right in zip(first, other_seed, strict=True)
	)


def test_exact_full_pool_supplement_does_not_advance_rng(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	class FailOnChoice:
		def choice(self, *_args: object, **_kwargs: object) -> None:
			raise AssertionError('exact full-pool selection must not consume RNG')

	monkeypatch.setattr(
		voxel_section_layout_selection.np.random,
		'default_rng',
		lambda _seed: FailOnChoice(),
	)
	rows = _exact_cap_row_pools()

	previews = preview_seeded_nested_class_cap_selection(
		LAYOUT_000,
		subsample_seed=0,
		per_class_token_row_caps=EXACT_CAPS,
		target_train_voxel_counts=EXACT_TARGETS,
		token_rows_by_size=rows,
		token_footprints_by_size=_footprint_pools(rows),
	)

	assert [preview.selected_token_row_count for preview in previews] == [6, 12, 18]


def test_row_identity_is_not_collapsed_at_duplicate_token_xyz() -> None:
	rows = _exact_cap_row_pools(duplicate_class_zero_large_token=True)
	previews = preview_seeded_nested_class_cap_selection(
		LAYOUT_000,
		subsample_seed=0,
		per_class_token_row_caps=EXACT_CAPS,
		target_train_voxel_counts=EXACT_TARGETS,
		token_rows_by_size=rows,
		token_footprints_by_size=_footprint_pools(rows),
	)
	large = previews[-1]
	class_zero = [
		identity
		for identity in large.selected_token_row_identities
		if identity[-1] == 0
	]

	assert len(class_zero) == 3
	assert len(set(class_zero)) == 3
	assert len({identity[2:5] for identity in class_zero}) == 2
	assert large.selected_token_row_count == 18
	assert len(large.selected_token_xyz) == 17


def test_pool_shortage_after_validation_precedence_has_full_diagnostic() -> None:
	rows = _exact_cap_row_pools()
	rows['large'] = tuple(
		row
		for row in rows['large']
		if not (row.class_id == 5 and row.token_xyz[1] == 2)
	)

	with pytest.raises(
		ValueError,
		match=(
			r'layout_002/large class 5 token-row pool 2 is below cap 3 '
			r'after validation precedence'
		),
	):
		preview_seeded_nested_class_cap_selection(
			LayoutLines(
				'layout_002',
				LAYOUT_000.ordered_inlines,
				LAYOUT_000.ordered_crosslines,
			),
			subsample_seed=2,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=_footprint_pools(rows),
		)


def test_strict_token_nesting_rejects_duplicate_only_growth() -> None:
	rows = _exact_cap_row_pools(duplicate_medium_tokens=True)

	with pytest.raises(ValueError, match='selected token_xyz must be strictly nested'):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=0,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=_footprint_pools(rows),
		)


def test_strict_voxel_nesting_rejects_empty_new_footprints() -> None:
	rows = _exact_cap_row_pools()
	footprints = _footprint_pools(rows, empty_ranks={1})

	with pytest.raises(
		ValueError, match='selected teacher voxels must be strictly nested'
	):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=0,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=footprints,
		)


def test_layout_seed_and_duplicate_row_identity_fail_closed() -> None:
	rows = _exact_cap_row_pools()
	footprints = _footprint_pools(rows)
	with pytest.raises(ValueError, match='subsample_seed must be exactly 0'):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=1,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=footprints,
		)
	with pytest.raises(TypeError, match='subsample_seed must be an integer'):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=True,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=footprints,
		)

	duplicate = dict(rows)
	duplicate['small'] = (*rows['small'], rows['small'][0])
	with pytest.raises(ValueError, match='contains duplicate row'):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=0,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=duplicate,
			token_footprints_by_size=_footprint_pools(duplicate),
		)


def test_every_active_line_requires_a_selected_token_row() -> None:
	rows = {
		size: tuple(
			TokenRow('inline', 100, row.token_xyz, row.class_id)
			if row.token_xyz[1] == 0
			else row
			for row in pool
		)
		for size, pool in _exact_cap_row_pools().items()
	}

	with pytest.raises(
		ValueError,
		match=(
			r"layout_000/small active lines have zero selected token rows: "
			r"\['crossline:200'\]"
		),
	):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=0,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=_footprint_pools(rows),
		)


def test_every_active_line_requires_a_dense_voxel_contribution() -> None:
	rows = _exact_cap_row_pools()
	footprints = _footprint_pools(rows)
	small = []
	for footprint in footprints['small']:
		if ('crossline', 200) not in footprint.per_line_flat_voxel_indices:
			small.append(footprint)
			continue
		small.append(
			replace(
				footprint,
				per_line_flat_voxel_indices={
					('inline', 100): footprint.flat_voxel_indices
				},
			)
		)
	footprints['small'] = tuple(small)

	with pytest.raises(
		ValueError,
		match=(
			r"layout_000/small active lines contribute zero teacher voxels: "
			r"\['crossline:200'\]"
		),
	):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=0,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=footprints,
		)


def test_dense_class_and_relative_error_gates_fail_closed() -> None:
	rows = _exact_cap_row_pools()
	footprints = _footprint_pools(rows)
	small = []
	for footprint in footprints['small']:
		if footprint.token_xyz[0] != 5:
			small.append(footprint)
			continue
		counts = dict(footprint.per_class_voxel_counts or {})
		counts['0'] += counts['5']
		counts['5'] = 0
		small.append(replace(footprint, per_class_voxel_counts=counts))
	footprints['small'] = tuple(small)

	with pytest.raises(ValueError, match=r"missing dense voxel classes \['5'\]"):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=0,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=footprints,
		)

	with pytest.raises(
		ValueError,
		match=r'layout_000/small target relative error .* exceeds 0.05',
	):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=0,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=EXACT_TARGETS,
			token_rows_by_size=rows,
			token_footprints_by_size=_footprint_pools(rows, voxel_width=50),
		)


def test_nominal_target_must_match_exact_per_class_cap() -> None:
	rows = _exact_cap_row_pools()
	targets = dict(EXACT_TARGETS)
	targets['small'] += 1

	with pytest.raises(
		ValueError,
		match=r'target_train_voxel_counts.small must equal six classes \* cap',
	):
		preview_seeded_nested_class_cap_selection(
			LAYOUT_000,
			subsample_seed=0,
			per_class_token_row_caps=EXACT_CAPS,
			target_train_voxel_counts=targets,
			token_rows_by_size=rows,
			token_footprints_by_size=_footprint_pools(rows),
		)


def _row_pools() -> dict[str, tuple[TokenRow, ...]]:
	return {
		size: tuple(
			_candidate_row(class_id, rank)
			for class_id in CLASS_IDS
			for rank in ranks
		)
		for size, ranks in SIZE_RANKS.items()
	}


def _exact_cap_row_pools(
	*,
	duplicate_class_zero_large_token: bool = False,
	duplicate_medium_tokens: bool = False,
) -> dict[str, tuple[TokenRow, ...]]:
	result: dict[str, tuple[TokenRow, ...]] = {}
	for size, stop in (('small', 1), ('medium', 2), ('large', 3)):
		rows = []
		for class_id in CLASS_IDS:
			for rank in range(stop):
				token_rank = rank
				if duplicate_medium_tokens and rank == 1:
					token_rank = 0
				if duplicate_class_zero_large_token and class_id == 0 and rank == 2:
					token_rank = 1
				rows.append(_exact_row(class_id, rank, token_rank=token_rank))
		result[size] = tuple(rows)
	return result


def _candidate_row(class_id: int, rank: int) -> TokenRow:
	if rank < 8:
		lines = (('inline', 100), ('crossline', 200))
	elif rank < 16:
		lines = (('inline', 101), ('crossline', 201))
	else:
		lines = (
			('inline', 102),
			('inline', 103),
			('crossline', 202),
			('crossline', 203),
		)
	slice_type, slice_index = lines[(class_id + rank) % len(lines)]
	return TokenRow(slice_type, slice_index, (class_id, rank, 0), class_id)


def _exact_row(
	class_id: int, rank: int, *, token_rank: int | None = None
) -> TokenRow:
	lines_by_rank = {
		0: (('inline', 100), ('crossline', 200)),
		1: (('inline', 101), ('crossline', 201)),
		2: (
			('inline', 102),
			('inline', 103),
			('crossline', 202),
			('crossline', 203),
		),
	}
	lines = lines_by_rank[rank]
	slice_type, slice_index = lines[class_id % len(lines)]
	return TokenRow(
		slice_type,
		slice_index,
		(class_id, rank if token_rank is None else token_rank, 0),
		class_id,
	)


def _footprint_pools(
	rows_by_size: dict[str, tuple[TokenRow, ...]],
	*,
	empty_ranks: set[int] | None = None,
	voxel_width: int = 64,
) -> dict[str, tuple[TokenFootprint, ...]]:
	identities = sorted({
		row.identity for rows in rows_by_size.values() for row in rows
	})
	identity_index = {identity: index for index, identity in enumerate(identities)}
	result: dict[str, tuple[TokenFootprint, ...]] = {}
	for size, rows in rows_by_size.items():
		by_xyz: dict[tuple[int, int, int], list[TokenRow]] = defaultdict(list)
		for row in rows:
			by_xyz[row.token_xyz].append(row)
		footprints = []
		for xyz, token_rows in sorted(by_xyz.items()):
			per_class = {str(class_id): 0 for class_id in CLASS_IDS}
			if empty_ranks and xyz[1] in empty_ranks:
				flat = ()
				per_line = {}
			else:
				flat_by_line: dict[tuple[str, int], list[int]] = defaultdict(list)
				for row in token_rows:
					start = identity_index[row.identity] * voxel_width
					values = tuple(range(start, start + voxel_width))
					flat_by_line[row.line_key].extend(values)
					per_class[str(row.class_id)] += voxel_width
				per_line = {
					key: tuple(sorted(values))
					for key, values in flat_by_line.items()
				}
				flat = tuple(sorted(
					value for values in per_line.values() for value in values
				))
			footprints.append(TokenFootprint(xyz, flat, per_line, per_class))
		result[size] = tuple(footprints)
	return result


def _identities(preview: object) -> set[object]:
	return set(preview.selected_token_row_identities)  # type: ignore[attr-defined]


def _tokens(preview: object) -> set[object]:
	return set(preview.selected_token_xyz)  # type: ignore[attr-defined]


def _voxels(preview: object) -> set[object]:
	return set(preview.selected_flat_voxel_indices)  # type: ignore[attr-defined]
