from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeAgcConfig,
	AmplitudeVolumeRecord,
	NopimsAmplitudePretrainDataset,
	SurveyManifest,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	apply_trace_rms_agc,
	load_normalization_stats,
	normalize_amplitude,
	write_normalization_stats,
)
from seis_ssl_cluster.training.dataloaders import build_mae_dataloader

if TYPE_CHECKING:
	from pathlib import Path

	import torch


def _write_volume(path: Path, volume: np.ndarray) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, volume.astype(np.float32, copy=False))
	return path


def _write_stats(
	path: Path,
	*,
	survey_id: str,
	source_path: Path,
	median: float = 0.0,
	iqr: float = 1.0,
) -> Path:
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id=survey_id,
			source_path=source_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1000.0,
			clip_high=1000.0,
			median=median,
			iqr=iqr,
		),
		path,
	)
	return path


def _manifest(
	tmp_path: Path,
	survey_id: str,
	volume: np.ndarray,
	*,
	median: float = 0.0,
) -> SurveyManifest:
	volume_path = _write_volume(tmp_path / survey_id / 'base.npy', volume)
	stats_path = _write_stats(
		tmp_path / 'stats' / f'{survey_id}.json',
		survey_id=survey_id,
		source_path=volume_path,
		median=median,
	)
	return SurveyManifest(
		survey_id=survey_id,
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id=survey_id,
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)


def test_amplitude_dataset_returns_one_channel_sample_contract(tmp_path: Path) -> None:
	volume = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
	dataset = NopimsAmplitudePretrainDataset(
		[_manifest(tmp_path, 'survey', volume)],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		seed=7,
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	sample = dataset[0]

	assert set(sample) == {
		'x',
		'target',
		'local_valid_mask',
		'coords',
		'spatial_mask',
		'visible_spatial_mask',
	}
	assert sample['x'].shape == (1, 4, 4, 4)
	assert sample['x'].dtype == np.float32
	np.testing.assert_array_equal(sample['x'], sample['target'])
	assert sample['local_valid_mask'].shape == (4, 4, 4)
	assert sample['local_valid_mask'].dtype == np.bool_
	assert sample['local_valid_mask'].all()
	assert sample['spatial_mask'].shape == (2, 2, 2)
	assert sample['spatial_mask'].dtype == np.bool_
	np.testing.assert_array_equal(
		sample['visible_spatial_mask'],
		np.logical_not(sample['spatial_mask']),
	)
	assert sample['coords']['survey_id'] == 'survey'
	assert sample['coords']['local_size_xyz'] == (4, 4, 4)
	assert not any('attribute' in key for key in sample)


def test_amplitude_dataset_uses_manifest_order_round_robin(tmp_path: Path) -> None:
	manifest_a = _manifest(tmp_path, 'a', np.ones((6, 6, 6), dtype=np.float32))
	manifest_b = _manifest(tmp_path, 'b', np.ones((6, 6, 6), dtype=np.float32))
	dataset = NopimsAmplitudePretrainDataset(
		[manifest_a, manifest_b],
		local_crop_size_xyz=(3, 3, 3),
		patch_size_xyz=(1, 1, 1),
		samples_per_epoch=5,
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	assert [dataset[index]['coords']['survey_id'] for index in range(5)] == [
		'a',
		'b',
		'a',
		'b',
		'a',
	]


def test_amplitude_dataset_zero_mask_uses_raw_amplitude_before_normalization(
	tmp_path: Path,
) -> None:
	volume = np.ones((5, 5, 5), dtype=np.float32) * 10.0
	volume[:, :, 2] = 0.0
	dataset = NopimsAmplitudePretrainDataset(
		[_manifest(tmp_path, 'survey', volume, median=10.0)],
		local_crop_size_xyz=(5, 5, 5),
		patch_size_xyz=(1, 1, 1),
		zero_mask=ZeroMaskConfig(
			z_sample_influence_radius=0,
			xy_trace_influence_radius=0,
		),
	)

	sample = dataset[0]

	assert not sample['local_valid_mask'][:, :, 2].any()
	assert sample['local_valid_mask'][:, :, :2].all()
	assert sample['local_valid_mask'][:, :, 3:].all()
	assert np.all(sample['x'][:, :, :, 2] == 0.0)


def test_amplitude_dataset_applies_agc_to_input_and_target(
	tmp_path: Path,
) -> None:
	volume = np.asarray([1.0, 1.0, 0.0, 10.0, 10.0], dtype=np.float32).reshape(
		1,
		1,
		5,
	)
	manifest = _manifest(tmp_path, 'survey', volume)
	dataset = NopimsAmplitudePretrainDataset(
		[manifest],
		local_crop_size_xyz=(1, 1, 5),
		patch_size_xyz=(1, 1, 1),
		spatial_mask_ratio=0.4,
		block_size_tokens_xyz=(1, 1, 1),
		zero_mask=ZeroMaskConfig(
			enabled=True,
			z_sample_influence_radius=0,
			xy_trace_influence_radius=0,
		),
		amplitude_agc=AmplitudeAgcConfig(
			enabled=True,
			mode='trace_rms_z',
			window_z=3,
			eps=1.0e-6,
			clip_abs=2.0,
		),
	)

	sample = dataset[0]

	valid = np.asarray([True, True, False, True, True]).reshape(1, 1, 5)
	normalized = normalize_amplitude(
		volume,
		load_normalization_stats(manifest.amplitude.normalization_stats_path),
	)
	expected = apply_trace_rms_agc(
		normalized,
		valid,
		window_z=3,
		eps=1.0e-6,
		clip_abs=2.0,
	)
	expected[~valid] = 0.0
	np.testing.assert_allclose(sample['x'][0], expected, rtol=1.0e-6)
	np.testing.assert_array_equal(sample['x'], sample['target'])
	assert sample['x'][0, 0, 0, 2] == 0.0
	assert np.max(np.abs(sample['x'])) <= 2.0


def test_amplitude_dataset_set_epoch_changes_coords_deterministically(
	tmp_path: Path,
) -> None:
	dataset = NopimsAmplitudePretrainDataset(
		[_manifest(tmp_path, 'survey', np.ones((20, 20, 20), dtype=np.float32))],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		seed=17,
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	epoch_zero = dataset[0]['coords']['local_start_xyz']
	dataset.set_epoch(1)
	epoch_one = dataset[0]['coords']['local_start_xyz']
	dataset.set_epoch(0)
	epoch_zero_again = dataset[0]['coords']['local_start_xyz']

	assert epoch_zero == epoch_zero_again
	assert epoch_zero != epoch_one


def test_amplitude_dataset_persistent_worker_reads_shared_epoch(
	tmp_path: Path,
) -> None:
	dataset = NopimsAmplitudePretrainDataset(
		[_manifest(tmp_path, 'survey', np.ones((20, 20, 20), dtype=np.float32))],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		seed=17,
		samples_per_epoch=1,
		zero_mask=ZeroMaskConfig(enabled=False),
	)
	dataloader = build_mae_dataloader(
		dataset,
		batch_size=1,
		num_workers=1,
		shuffle=False,
		seed=17,
		device='cpu',
	)
	assert dataloader.persistent_workers is True

	try:
		dataset.set_epoch(0)
		epoch_zero = _first_local_start_xyz(dataloader)
		dataset.set_epoch(1)
		epoch_one = _first_local_start_xyz(dataloader)
		dataset.set_epoch(0)
		epoch_zero_again = _first_local_start_xyz(dataloader)
	finally:
		_shutdown_persistent_workers(dataloader)

	assert epoch_zero == epoch_zero_again
	assert epoch_zero != epoch_one


def test_amplitude_dataset_rejects_manifest_smaller_than_crop(tmp_path: Path) -> None:
	manifest = _manifest(tmp_path, 'small', np.ones((3, 4, 5), dtype=np.float32))

	with pytest.raises(ValueError, match='does not fit'):
		NopimsAmplitudePretrainDataset(
			[manifest],
			local_crop_size_xyz=(4, 4, 4),
			patch_size_xyz=(2, 2, 2),
		)


def test_amplitude_dataset_respects_min_valid_fraction(tmp_path: Path) -> None:
	volume = np.zeros((4, 4, 4), dtype=np.float32)
	dataset = NopimsAmplitudePretrainDataset(
		[_manifest(tmp_path, 'survey', volume)],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		zero_mask=ZeroMaskConfig(
			z_sample_influence_radius=0,
			xy_trace_influence_radius=0,
		),
		min_valid_fraction=0.5,
		max_resample_attempts=2,
	)

	with pytest.raises(ValueError, match='min_valid_fraction'):
		dataset[0]


def test_amplitude_dataset_masks_are_deterministic_for_seed_epoch_index(
	tmp_path: Path,
) -> None:
	volume = np.ones((12, 12, 12), dtype=np.float32)
	dataset = NopimsAmplitudePretrainDataset(
		[_manifest(tmp_path, 'survey', volume)],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		seed=23,
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	first = dataset[0]['spatial_mask']
	second = dataset[0]['spatial_mask']

	np.testing.assert_array_equal(first, second)


def test_amplitude_dataset_from_config_uses_masking_settings(tmp_path: Path) -> None:
	volume = np.ones((8, 8, 8), dtype=np.float32)
	config = {
		'data': {
			'local_crop_size': [4, 4, 4],
		},
		'model': {
			'patch_size': [2, 2, 2],
		},
		'masking': {
			'spatial_mask_ratio': 0.5,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
		},
		'train': {
			'seed': 11,
			'samples_per_epoch': 3,
		},
		'zero_mask': {
			'enabled': False,
		},
	}

	dataset = NopimsAmplitudePretrainDataset.from_config(
		[_manifest(tmp_path, 'survey', volume)],
		config,
	)
	sample = dataset[0]

	assert dataset.patch_size_xyz == (2, 2, 2)
	assert dataset.block_size_tokens_xyz == (1, 1, 1)
	assert len(dataset) == 3
	assert sample['spatial_mask'].shape == (2, 2, 2)
	assert float(sample['spatial_mask'].mean()) == 0.5


def _first_local_start_xyz(
	dataloader: torch.utils.data.DataLoader,
) -> tuple[int, int, int]:
	batch = next(iter(dataloader))
	coords = batch['coords']
	if isinstance(coords, list):
		return tuple(int(axis) for axis in coords[0]['local_start_xyz'])

	local_start_xyz = coords['local_start_xyz']
	if hasattr(local_start_xyz, 'tolist'):
		return tuple(int(axis) for axis in local_start_xyz[0].tolist())
	return tuple(int(axis[0]) for axis in local_start_xyz)


def _shutdown_persistent_workers(dataloader: torch.utils.data.DataLoader) -> None:
	iterator = getattr(dataloader, '_iterator', None)
	if iterator is not None:
		iterator._shutdown_workers()  # noqa: SLF001
