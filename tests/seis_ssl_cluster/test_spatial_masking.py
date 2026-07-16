from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_ssl_cluster.masking import (
	SpatialMaskingPlan,
	build_spatial_masking_plan,
	compute_token_grid_shape,
	generate_spatial_block_mask,
	validate_spatial_masking_plan,
)


class _ChoiceSpyGenerator(np.random.Generator):
	def __init__(self, seed: int) -> None:
		super().__init__(np.random.PCG64(seed))
		self.choice_calls: list[tuple[int, int, bool]] = []

	def choice(self, a: int, **kwargs: object) -> np.ndarray:  # type: ignore[override]
		size = kwargs['size']
		replace = kwargs['replace']
		assert isinstance(size, int)
		assert isinstance(replace, bool)
		self.choice_calls.append((a, size, replace))
		return super().choice(a, **kwargs)


def _build_plan(seed: int = 123) -> SpatialMaskingPlan:
	return build_spatial_masking_plan(
		local_crop_size_xyz=(128, 128, 128),
		patch_size_xyz=(8, 8, 8),
		spatial_mask_ratio=0.75,
		spatial_mask_mode='block',
		block_size_tokens_xyz=(2, 2, 2),
		rng=np.random.default_rng(seed),
	)


def test_compute_token_grid_shape_default_mvp_grid() -> None:
	assert compute_token_grid_shape([128, 128, 128], [8, 8, 8]) == (16, 16, 16)


def test_build_spatial_masking_plan_returns_contract() -> None:
	plan = _build_plan()

	assert plan.spatial_mask.shape == (16, 16, 16)
	assert plan.spatial_mask.dtype == np.bool_
	assert plan.visible_spatial_mask.dtype == np.bool_
	assert 0.74 <= float(plan.spatial_mask.mean()) <= 0.76
	np.testing.assert_array_equal(
		plan.visible_spatial_mask,
		np.logical_not(plan.spatial_mask),
	)


def test_build_spatial_masking_plan_is_deterministic_for_seed() -> None:
	first = _build_plan(seed=999)
	second = _build_plan(seed=999)

	np.testing.assert_array_equal(first.spatial_mask, second.spatial_mask)
	np.testing.assert_array_equal(
		first.visible_spatial_mask,
		second.visible_spatial_mask,
	)


def test_generate_spatial_block_mask_trims_large_overshoots() -> None:
	mask = generate_spatial_block_mask(
		(4, 4, 4),
		0.10,
		(4, 4, 4),
		np.random.default_rng(123),
	)

	assert 0.08 <= float(mask.mean()) <= 0.12
	assert np.any(mask)
	assert np.any(np.logical_not(mask))


def test_generate_spatial_block_mask_keeps_visible_and_masked_tokens() -> None:
	mask = generate_spatial_block_mask(
		(2, 2, 2),
		0.99,
		(2, 2, 2),
		np.random.default_rng(123),
	)

	assert np.any(mask)
	assert np.any(np.logical_not(mask))


@pytest.mark.parametrize(
	('token_grid_shape', 'mask_ratio', 'expected_count'),
	[
		((16, 16, 16), 0.75, 3072),
		((16, 16, 16), 0.00001, 1),
		((16, 16, 16), 0.99999, 4095),
		((2, 3, 4), 0.5, 12),
		((1, 1, 2), 0.5, 1),
	],
)
def test_generate_unit_block_mask_has_exact_target_count(
	token_grid_shape: tuple[int, int, int],
	mask_ratio: float,
	expected_count: int,
) -> None:
	mask = generate_spatial_block_mask(
		token_grid_shape,
		mask_ratio,
		(1, 1, 1),
		np.random.default_rng(123),
	)

	assert mask.shape == token_grid_shape
	assert np.count_nonzero(mask) == expected_count


def test_generate_unit_block_mask_is_deterministic_for_seed() -> None:
	first = generate_spatial_block_mask(
		(5, 4, 3),
		0.75,
		(1, 1, 1),
		np.random.default_rng(999),
	)
	second = generate_spatial_block_mask(
		(5, 4, 3),
		0.75,
		(1, 1, 1),
		np.random.default_rng(999),
	)

	np.testing.assert_array_equal(first, second)


def test_generate_unit_block_mask_samples_once_without_replacement() -> None:
	rng = _ChoiceSpyGenerator(seed=123)

	mask = generate_spatial_block_mask((16, 16, 16), 0.75, (1, 1, 1), rng)

	assert rng.choice_calls == [(4096, 3072, False)]
	assert np.count_nonzero(mask) == 3072


def test_generate_general_block_mask_preserves_golden_result() -> None:
	mask = generate_spatial_block_mask(
		(3, 3, 3),
		0.4,
		(2, 2, 2),
		np.random.default_rng(123),
	)

	assert np.flatnonzero(mask).tolist() == [3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 17]


def test_compute_token_grid_shape_rejects_non_divisible_sizes() -> None:
	with pytest.raises(ValueError, match='exactly divisible'):
		compute_token_grid_shape([128, 128, 127], [8, 8, 8])


def test_build_spatial_masking_plan_rejects_non_block_mode() -> None:
	with pytest.raises(ValueError, match='spatial_mask_mode'):
		build_spatial_masking_plan(
			local_crop_size_xyz=(128, 128, 128),
			patch_size_xyz=(8, 8, 8),
			spatial_mask_ratio=0.75,
			spatial_mask_mode='random',
			block_size_tokens_xyz=(2, 2, 2),
			rng=np.random.default_rng(123),
		)


@pytest.mark.parametrize('mask_ratio', [-0.1, 0.0, 1.0, 1.1])
def test_generate_spatial_block_mask_rejects_invalid_ratios(mask_ratio: float) -> None:
	with pytest.raises(ValueError, match='mask_ratio must be in'):
		generate_spatial_block_mask(
			(16, 16, 16),
			mask_ratio,
			(2, 2, 2),
			np.random.default_rng(123),
		)


@pytest.mark.parametrize('block_size', [(0, 2, 2), (-1, 2, 2), (2, 2)])
def test_generate_spatial_block_mask_rejects_invalid_block_sizes(
	block_size: tuple[int, ...],
) -> None:
	with pytest.raises((TypeError, ValueError), match='block_size_tokens_xyz'):
		generate_spatial_block_mask(
			(16, 16, 16),
			0.75,
			block_size,
			np.random.default_rng(123),
		)


def test_validate_spatial_masking_plan_rejects_invalid_visible_mask() -> None:
	spatial_mask = np.zeros((2, 2, 2), dtype=np.bool_)
	spatial_mask[0, 0, 0] = True
	plan = SpatialMaskingPlan(
		spatial_mask=spatial_mask,
		visible_spatial_mask=np.logical_not(spatial_mask),
	)

	with pytest.raises(ValueError, match='visible_spatial_mask must equal'):
		validate_spatial_masking_plan(
			replace(plan, visible_spatial_mask=np.ones((2, 2, 2), dtype=np.bool_)),
		)
