from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudePretrainDataset,
	AmplitudeVolumeRecord,
	BarlowTwinsPretrainDataset,
	SurveyManifest,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	write_normalization_stats,
)
from seis_ssl_cluster.training import build_barlow_twins_dataloader

if TYPE_CHECKING:
	from pathlib import Path


def _manifest(
	tmp_path: Path,
	volume: np.ndarray,
	*,
	valid_mask: np.ndarray | None = None,
) -> SurveyManifest:
	volume_path = tmp_path / 'survey' / 'amplitude.npy'
	volume_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(volume_path, volume.astype(np.float32, copy=False))
	stats_path = tmp_path / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-10_000.0,
			clip_high=10_000.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	valid_mask_path = None
	if valid_mask is not None:
		valid_mask_path = tmp_path / 'survey' / 'valid.npy'
		np.save(valid_mask_path, valid_mask.astype(bool, copy=False))
	return SurveyManifest(
		survey_id='survey',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
			valid_mask_path=valid_mask_path,
		),
	)


def _base_dataset(
	tmp_path: Path,
	*,
	valid_mask: np.ndarray | None = None,
	samples_per_epoch: int = 8,
) -> AmplitudePretrainDataset:
	volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 1.0
	return AmplitudePretrainDataset(
		[_manifest(tmp_path, volume, valid_mask=valid_mask)],
		local_crop_size_xyz=volume.shape,
		patch_size_xyz=(1, 1, 2),
		seed=19,
		samples_per_epoch=samples_per_epoch,
		zero_mask=ZeroMaskConfig(enabled=False),
	)


def test_two_views_share_one_physical_crop_and_coordinates(tmp_path: Path) -> None:
	base = _base_dataset(tmp_path)
	read_candidate = Mock(wraps=base._read_amplitude_crop_candidate)  # noqa: SLF001
	base._read_amplitude_crop_candidate = read_candidate  # type: ignore[method-assign]  # noqa: SLF001
	dataset = BarlowTwinsPretrainDataset(base)

	sample = dataset[0]

	assert read_candidate.call_count == 1
	assert sample['coords'] == {
		'survey_id': 'survey',
		'local_start_xyz': (0, 0, 0),
		'local_size_xyz': (2, 3, 4),
	}
	assert np.array_equal(
		np.sort(sample['view_a'], axis=None),
		np.sort(sample['view_b'], axis=None),
	)


def test_flips_only_horizontal_axes_and_keeps_mask_aligned(tmp_path: Path) -> None:
	valid_mask = np.ones((2, 3, 4), dtype=bool)
	valid_mask[0, 1, 2] = False
	base = _base_dataset(tmp_path, valid_mask=valid_mask)
	base_sample = base[0]
	dataset = BarlowTwinsPretrainDataset(
		base,
		horizontal_flip_probability=1.0,
	)

	sample = dataset[0]
	expected_view = base_sample['x'][:, ::-1, ::-1, :]
	expected_mask = base_sample['local_valid_mask'][::-1, ::-1, :]

	for view_key, mask_key in (
		('view_a', 'valid_mask_a'),
		('view_b', 'valid_mask_b'),
	):
		np.testing.assert_array_equal(sample[view_key], expected_view)
		np.testing.assert_array_equal(sample[mask_key], expected_mask)
		assert np.all(np.diff(sample[view_key][0, 0, 0]) > 0.0)
		assert np.all(sample[view_key][0][~sample[mask_key]] == 0.0)


def test_seed_epoch_and_index_determine_views(tmp_path: Path) -> None:
	dataset = BarlowTwinsPretrainDataset(_base_dataset(tmp_path))

	first = dataset[3]
	second = dataset[3]

	for key in ('view_a', 'view_b', 'valid_mask_a', 'valid_mask_b'):
		np.testing.assert_array_equal(first[key], second[key])
	assert first['coords'] == second['coords']


def test_changing_epoch_changes_at_least_one_augmentation(tmp_path: Path) -> None:
	dataset = BarlowTwinsPretrainDataset(_base_dataset(tmp_path))
	epoch_zero = [
		(dataset[index]['view_a'], dataset[index]['view_b']) for index in range(8)
	]

	dataset.set_epoch(1)
	epoch_one = [
		(dataset[index]['view_a'], dataset[index]['view_b']) for index in range(8)
	]
	dataset.set_epoch(0)

	assert any(
		not np.array_equal(zero_a, one_a)
		or not np.array_equal(zero_b, one_b)
		for (zero_a, zero_b), (one_a, one_b) in zip(
			epoch_zero,
			epoch_one,
			strict=True,
		)
	)
	for index, (expected_a, expected_b) in enumerate(epoch_zero):
		np.testing.assert_array_equal(dataset[index]['view_a'], expected_a)
		np.testing.assert_array_equal(dataset[index]['view_b'], expected_b)


def test_set_epoch_reaches_wrapped_dataset(tmp_path: Path) -> None:
	base = _base_dataset(tmp_path)
	set_epoch = Mock(wraps=base.set_epoch)
	base.set_epoch = set_epoch  # type: ignore[method-assign]
	dataset = BarlowTwinsPretrainDataset(base)

	dataset.set_epoch(7)

	set_epoch.assert_called_once_with(7)
	assert base.epoch == 7


def test_multi_worker_loading_matches_single_process_loading(tmp_path: Path) -> None:
	dataset = BarlowTwinsPretrainDataset(_base_dataset(tmp_path))
	single_process = build_barlow_twins_dataloader(
		dataset,
		batch_size=2,
		num_workers=0,
		shuffle=False,
	)
	multi_worker = build_barlow_twins_dataloader(
		dataset,
		batch_size=2,
		num_workers=2,
		shuffle=False,
		persistent_workers=False,
	)

	single_batch = next(iter(single_process))
	multi_batch = next(iter(multi_worker))

	for key in ('view_a', 'view_b', 'valid_mask_a', 'valid_mask_b'):
		torch.testing.assert_close(single_batch[key], multi_batch[key])
	assert single_batch['coords'] == multi_batch['coords']


def test_dataloader_returns_two_view_batch_without_mae_mask(tmp_path: Path) -> None:
	dataset = BarlowTwinsPretrainDataset(
		_base_dataset(tmp_path, samples_per_epoch=3),
	)
	dataloader = build_barlow_twins_dataloader(
		dataset,
		batch_size=2,
		num_workers=0,
		shuffle=False,
	)

	batch = next(iter(dataloader))

	assert len(dataloader) == 1
	assert set(batch) == {
		'view_a',
		'view_b',
		'valid_mask_a',
		'valid_mask_b',
		'coords',
	}
	assert batch['view_a'].shape == (2, 1, 2, 3, 4)
	assert batch['view_b'].shape == (2, 1, 2, 3, 4)
	assert batch['valid_mask_a'].shape == (2, 2, 3, 4)
	assert batch['valid_mask_b'].shape == (2, 2, 3, 4)
	assert 'spatial_mask' not in batch


def test_dataloader_rejects_batch_size_one(tmp_path: Path) -> None:
	dataset = BarlowTwinsPretrainDataset(_base_dataset(tmp_path))

	with pytest.raises(ValueError, match='at least 2'):
		build_barlow_twins_dataloader(dataset, batch_size=1)
