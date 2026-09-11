"""Regression coverage for inference-only, disjoint-core publication exports."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.parihaka.channel_data import SectionLines
from seis_ssl_cluster.parihaka.channel_tiles import (
	ChannelTileSettings,
	build_channel_tile_targets,
	enumerate_channel_tile_records,
)
from seis_ssl_cluster.parihaka.paper_prediction import (
	CONDITIONS,
	core_slices,
	tile_input,
	validate_prediction,
)


def test_core_coverage_and_orientation() -> None:
	"""Every voxel belongs to exactly one core, including partial edge tiles."""
	shape = (78, 69, 70)
	counts = np.zeros(shape, dtype=np.int8)
	for start in itertools.product(range(0, 10, 8), range(0, 9, 8), range(0, 9, 8)):
		global_crop, local_crop = core_slices(start, shape)
		counts[global_crop] += 1
		assert all(s.start == 8 for s in local_crop)
		assert counts[global_crop].shape == tuple(s.stop - s.start for s in local_crop)
	assert (counts == 1).all()


def test_halo_input() -> None:
	"""Input padding and embedding axis movement match the trained decoder."""
	valid = np.ones((12, 13, 14), dtype=np.bool_)
	emb = np.arange(12 * 13 * 14 * 2, dtype=np.float32).reshape(12, 13, 14, 2)
	values, mask = tile_input(emb, valid, (0, 0, 0))
	np.testing.assert_array_equal(
		values[:, 1:, 1:, 1:], np.moveaxis(emb[:9, :9, :9], -1, 0)
	)
	assert not mask[0].any()
	values, mask = tile_input(emb, valid, (8, 8, 8))
	np.testing.assert_array_equal(
		values[:, :5, :6, :7], np.moveaxis(emb[7:, 7:, 7:], -1, 0)
	)
	assert mask.sum() == 5 * 6 * 7


def test_prediction_validation(tmp_path: Path) -> None:
	"""Class zero is valid and only invalid voxels may carry minus one."""
	shape = (2, 3, 4)
	pred = np.zeros(shape, dtype=np.int8)
	mask = np.ones(shape, dtype=np.bool_)
	mask[0, 0, 0] = False
	pred[0, 0, 0] = -1
	np.save(tmp_path / 'parihaka_voxel_predictions.npy', pred)
	np.save(tmp_path / 'parihaka_valid_voxel_mask.npy', mask)
	assert validate_prediction(tmp_path, shape)['valid_voxels'] == 23
	pred[1, 0, 0] = 2
	np.save(tmp_path / 'parihaka_voxel_predictions.npy', pred)
	with pytest.raises(ValueError, match='class'):
		validate_prediction(tmp_path, shape)


def test_condition_order() -> None:
	"""Exclude per-family weight selection."""
	assert CONDITIONS == ('1', '2b', '3', '4b', '5', '6b')


@pytest.mark.parametrize('start', [(0, 0, 0), (8, 8, 8)])
def test_training_halo_contract(start: tuple[int, int, int]) -> None:
	"""Compare export masks directly with the existing training tile helper."""
	shape = (78, 69, 70)
	grid = (10, 9, 9)
	valid = np.ones(grid, dtype=np.bool_)
	valid[3, 4, 5] = False
	emb = np.zeros((*grid, 2), dtype=np.float16)
	settings = ChannelTileSettings(shape, grid, (8, 8, 8), (8, 8, 8), (1, 1, 1))
	kwargs = {
		'valid_tokens': valid,
		'training_selection_mask': None,
		'labels': np.ones(shape, dtype=np.int8),
		'settings': settings,
		'train': SectionLines((0,), (0,)),
		'validation': SectionLines((1,), (1,)),
		'reserved_training': SectionLines((0,), (0,)),
		'split': 'test',
	}
	records, _ = enumerate_channel_tile_records(**kwargs)
	record = next(r for r in records if r.core_start_token == start)
	targets = build_channel_tile_targets(record=record, **kwargs)
	_, mask = tile_input(emb, valid, start)
	np.testing.assert_array_equal(mask, targets.token_valid_mask)
	_, local = core_slices(start, shape)
	assert targets.core_mask[local].all()
