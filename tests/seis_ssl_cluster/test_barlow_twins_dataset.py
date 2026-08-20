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
	LocalBarlowTwinsPretrainDataset,
	SurveyManifest,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	write_normalization_stats,
)
from seis_ssl_cluster.models.mae.patching import patchify_3d
from seis_ssl_cluster.training import (
	barlow_twins_collate_fn,
	build_barlow_twins_dataloader,
)

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
	min_valid_token_count: int = 0,
) -> AmplitudePretrainDataset:
	volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 1.0
	return AmplitudePretrainDataset(
		[_manifest(tmp_path, volume, valid_mask=valid_mask)],
		local_crop_size_xyz=volume.shape,
		patch_size_xyz=(1, 1, 2),
		emit_spatial_mask=False,
		seed=19,
		samples_per_epoch=samples_per_epoch,
		zero_mask=ZeroMaskConfig(enabled=False),
		min_valid_token_count=min_valid_token_count,
	)


def test_base_dataset_does_not_build_mae_mask(tmp_path: Path) -> None:
	base = _base_dataset(tmp_path)
	build_mask = Mock(side_effect=AssertionError('MAE mask should not be built'))
	base._add_spatial_masks = build_mask  # type: ignore[method-assign]  # noqa: SLF001

	sample = base[0]

	assert 'spatial_mask' not in sample
	build_mask.assert_not_called()


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


@pytest.mark.parametrize('probability', [0.0, 0.5, 1.0])
def test_local_views_always_have_distinct_flip_states(
	tmp_path: Path,
	probability: float,
) -> None:
	base = _base_dataset(tmp_path, min_valid_token_count=4)
	read_candidate = Mock(wraps=base._read_amplitude_crop_candidate)  # noqa: SLF001
	base._read_amplitude_crop_candidate = read_candidate  # type: ignore[method-assign]  # noqa: SLF001
	dataset = LocalBarlowTwinsPretrainDataset(
		base,
		local_pairs_per_crop=4,
		horizontal_flip_probability=probability,
	)

	for index in range(len(dataset)):
		sample = dataset[index]
		assert not np.array_equal(
			sample['horizontal_flip_state_a'],
			sample['horizontal_flip_state_b'],
		)

	assert read_candidate.call_count == len(dataset)


def test_local_dataset_rejects_insufficient_base_token_contract(
	tmp_path: Path,
) -> None:
	base = _base_dataset(tmp_path, min_valid_token_count=3)

	with pytest.raises(ValueError, match=r'base_dataset\.min_valid_token_count'):
		LocalBarlowTwinsPretrainDataset(base, local_pairs_per_crop=4)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_local_dataset_rejects_invalid_pair_count(
	tmp_path: Path,
	value: object,
) -> None:
	base = _base_dataset(tmp_path, min_valid_token_count=4)

	with pytest.raises((TypeError, ValueError), match='local_pairs_per_crop'):
		LocalBarlowTwinsPretrainDataset(
			base,
			local_pairs_per_crop=value,  # type: ignore[arg-type]
		)


def test_local_seed_epoch_and_index_determine_complete_sample(tmp_path: Path) -> None:
	dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(tmp_path, min_valid_token_count=4),
		local_pairs_per_crop=4,
	)

	first = dataset[3]
	second = dataset[3]

	for key in (
		'view_a',
		'view_b',
		'valid_mask_a',
		'valid_mask_b',
		'horizontal_flip_state_a',
		'horizontal_flip_state_b',
		'local_pair_indices_a',
		'local_pair_indices_b',
	):
		np.testing.assert_array_equal(first[key], second[key])
	assert first['coords'] == second['coords']


def test_local_epoch_changes_at_least_one_sample_and_resets(tmp_path: Path) -> None:
	dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(tmp_path, min_valid_token_count=4),
		local_pairs_per_crop=4,
	)
	epoch_zero = [dataset[index] for index in range(len(dataset))]

	dataset.set_epoch(1)
	epoch_one = [dataset[index] for index in range(len(dataset))]
	dataset.set_epoch(0)

	assert any(
		not np.array_equal(
			zero['local_pair_indices_a'],
			one['local_pair_indices_a'],
		)
		or not np.array_equal(
			zero['horizontal_flip_state_a'],
			one['horizontal_flip_state_a'],
		)
		for zero, one in zip(epoch_zero, epoch_one, strict=True)
	)
	for index, expected in enumerate(epoch_zero):
		for key in (
			'horizontal_flip_state_a',
			'horizontal_flip_state_b',
			'local_pair_indices_a',
			'local_pair_indices_b',
		):
			np.testing.assert_array_equal(dataset[index][key], expected[key])


def test_local_pair_indices_select_same_physical_patches_after_patchify(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	token_grid_shape_xyz = (2, 3, 2)
	patch_ids = np.arange(1, 13, dtype=np.float32).reshape(
		token_grid_shape_xyz
	)
	volume = patch_ids.repeat(patch_size_xyz[0], axis=0)
	volume = volume.repeat(patch_size_xyz[1], axis=1)
	volume = volume.repeat(patch_size_xyz[2], axis=2)
	valid_mask = np.ones_like(volume, dtype=bool)
	invalid_token_xyz = (1, 1, 0)
	invalid_slices = tuple(
		slice(token * patch, (token + 1) * patch)
		for token, patch in zip(
			invalid_token_xyz,
			patch_size_xyz,
			strict=True,
		)
	)
	valid_mask[invalid_slices] = False
	invalid_patch_id = int(patch_ids[invalid_token_xyz])
	valid_patch_ids = set(range(1, patch_ids.size + 1)) - {invalid_patch_id}
	base = AmplitudePretrainDataset(
		[_manifest(tmp_path, volume, valid_mask=valid_mask)],
		local_crop_size_xyz=volume.shape,
		patch_size_xyz=patch_size_xyz,
		emit_spatial_mask=False,
		seed=19,
		samples_per_epoch=8,
		zero_mask=ZeroMaskConfig(enabled=False),
		min_valid_token_count=len(valid_patch_ids),
	)
	dataset = LocalBarlowTwinsPretrainDataset(
		base,
		local_pairs_per_crop=len(valid_patch_ids),
	)
	exercised_flip_axes = np.zeros(2, dtype=bool)

	for index in range(len(dataset)):
		sample = dataset[index]
		patches_a = patchify_3d(
			torch.as_tensor(sample['view_a']).unsqueeze(0),
			patch_size_xyz,
		)[0]
		patches_b = patchify_3d(
			torch.as_tensor(sample['view_b']).unsqueeze(0),
			patch_size_xyz,
		)[0]
		selected_a = patches_a.index_select(
			0,
			torch.as_tensor(sample['local_pair_indices_a']),
		)
		selected_b = patches_b.index_select(
			0,
			torch.as_tensor(sample['local_pair_indices_b']),
		)

		torch.testing.assert_close(selected_a, selected_b)
		selected_patch_ids = selected_a[:, 0, 0]
		rounded_patch_ids = selected_patch_ids.round().to(torch.int64)
		assert set(rounded_patch_ids.tolist()) == valid_patch_ids
		torch.testing.assert_close(
			selected_patch_ids,
			rounded_patch_ids.to(selected_patch_ids.dtype),
			rtol=0.0,
			atol=2e-5,
		)
		torch.testing.assert_close(
			selected_a,
			selected_patch_ids[:, None, None].expand_as(selected_a),
		)
		exercised_flip_axes |= np.logical_xor(
			sample['horizontal_flip_state_a'],
			sample['horizontal_flip_state_b'],
		)

	assert exercised_flip_axes.all()


def test_local_multi_worker_batch_matches_single_process(tmp_path: Path) -> None:
	dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(tmp_path, min_valid_token_count=4),
		local_pairs_per_crop=4,
	)
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

	for key in (
		'view_a',
		'view_b',
		'valid_mask_a',
		'valid_mask_b',
		'horizontal_flip_state_a',
		'horizontal_flip_state_b',
		'local_pair_indices_a',
		'local_pair_indices_b',
	):
		torch.testing.assert_close(single_batch[key], multi_batch[key])
	assert single_batch['coords'] == multi_batch['coords']
	assert single_batch['local_pair_indices_a'].dtype is torch.int64
	assert single_batch['horizontal_flip_state_a'].dtype is torch.bool


def test_barlow_collate_rejects_mixed_standard_and_local_samples(
	tmp_path: Path,
) -> None:
	base = _base_dataset(tmp_path, min_valid_token_count=4)
	standard = BarlowTwinsPretrainDataset(base)[0]
	local = LocalBarlowTwinsPretrainDataset(base, local_pairs_per_crop=4)[1]

	with pytest.raises(ValueError, match='all contain every local Barlow Twins key'):
		barlow_twins_collate_fn([standard, local])


def test_dataloader_rejects_batch_size_one(tmp_path: Path) -> None:
	dataset = BarlowTwinsPretrainDataset(_base_dataset(tmp_path))

	with pytest.raises(ValueError, match='at least 2'):
		build_barlow_twins_dataloader(dataset, batch_size=1)
