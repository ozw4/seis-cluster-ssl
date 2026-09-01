from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import seis_ssl_cluster.data.barlow_twins_dataset as barlow_dataset_module
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudePretrainDataset,
	AmplitudeVolumeRecord,
	BarlowTwinsPretrainDataset,
	LocalBarlowTwinsD4TraceDropPretrainDataset,
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


def _square_d4_base_dataset(
	tmp_path: Path,
	*,
	valid_mask: np.ndarray | None = None,
	patch_size_xyz: tuple[int, int, int] = (2, 2, 2),
	samples_per_epoch: int = 8,
	min_valid_token_count: int = 8,
) -> AmplitudePretrainDataset:
	token_shape = tuple(4 // patch for patch in patch_size_xyz)
	patch_ids = np.arange(
		1,
		int(np.prod(token_shape)) + 1,
		dtype=np.float32,
	).reshape(token_shape)
	volume = patch_ids
	for axis, repeat in enumerate(patch_size_xyz):
		volume = volume.repeat(repeat, axis=axis)
	return AmplitudePretrainDataset(
		[_manifest(tmp_path, volume, valid_mask=valid_mask)],
		local_crop_size_xyz=volume.shape,
		patch_size_xyz=patch_size_xyz,
		emit_spatial_mask=False,
		seed=31,
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


@pytest.mark.parametrize('value', [-0.1, np.inf, np.nan, True, '0.1'])
def test_local_dataset_rejects_invalid_gaussian_noise_std(
	tmp_path: Path,
	value: object,
) -> None:
	base = _base_dataset(tmp_path, min_valid_token_count=4)

	with pytest.raises((TypeError, ValueError), match='gaussian_noise_std'):
		LocalBarlowTwinsPretrainDataset(
			base,
			local_pairs_per_crop=4,
			gaussian_noise_std=value,  # type: ignore[arg-type]
		)


@pytest.mark.parametrize('value', [-0.1, 1.1, np.inf, np.nan, True, '0.1'])
def test_local_dataset_rejects_invalid_trace_drop_probability(
	tmp_path: Path,
	value: object,
) -> None:
	base = _base_dataset(tmp_path, min_valid_token_count=4)

	with pytest.raises((TypeError, ValueError), match='trace_drop_probability'):
		LocalBarlowTwinsPretrainDataset(
			base,
			local_pairs_per_crop=4,
			trace_drop_probability=value,  # type: ignore[arg-type]
		)


@pytest.mark.parametrize('value', [-0.1, 0.5, 1.0, float('inf'), 'invalid'])
def test_local_dataset_rejects_invalid_z_filter_side_weight(
	tmp_path: Path,
	value: object,
) -> None:
	with pytest.raises((TypeError, ValueError), match='z_filter_side_weight'):
		LocalBarlowTwinsPretrainDataset(
			_base_dataset(tmp_path, min_valid_token_count=4),
			local_pairs_per_crop=4,
			z_filter_side_weight=value,  # type: ignore[arg-type]
		)


@pytest.mark.parametrize('value', [0, 1, None, 'false'])
def test_local_dataset_rejects_invalid_distinct_view_bool(
	tmp_path: Path,
	value: object,
) -> None:
	base = _base_dataset(tmp_path, min_valid_token_count=4)

	with pytest.raises(
		TypeError,
		match='require_distinct_horizontal_views',
	):
		LocalBarlowTwinsPretrainDataset(
			base,
			local_pairs_per_crop=4,
			require_distinct_horizontal_views=value,  # type: ignore[arg-type]
		)


def test_local_identity_gaussian_views_preserve_canonical_geometry(
	tmp_path: Path,
) -> None:
	valid_mask = np.ones((2, 3, 4), dtype=bool)
	valid_mask[0, 1, 2] = False
	base = _base_dataset(
		tmp_path,
		valid_mask=valid_mask,
		min_valid_token_count=4,
	)
	reference = base[0]
	dataset = LocalBarlowTwinsPretrainDataset(
		base,
		local_pairs_per_crop=4,
		horizontal_flip_probability=0.0,
		gaussian_noise_std=0.25,
		require_distinct_horizontal_views=False,
	)

	sample = dataset[0]
	repeated = dataset[0]

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
		np.testing.assert_array_equal(sample[key], repeated[key])
	for suffix in ('a', 'b'):
		assert not np.asarray(sample[f'horizontal_flip_state_{suffix}']).any()
		np.testing.assert_array_equal(
			sample[f'valid_mask_{suffix}'],
			reference['local_valid_mask'],
		)
	np.testing.assert_array_equal(
		sample['local_pair_indices_a'],
		sample['local_pair_indices_b'],
	)

	mask = np.asarray(reference['local_valid_mask'])
	base_view = np.asarray(reference['x'])
	residual_a = np.asarray(sample['view_a']) - base_view
	residual_b = np.asarray(sample['view_b']) - base_view
	for view in (sample['view_a'], sample['view_b']):
		np.testing.assert_array_equal(
			np.asarray(view)[0][~mask],
			base_view[0][~mask],
		)
	assert np.any(residual_a[0][mask] != 0.0)
	assert np.any(residual_b[0][mask] != 0.0)
	assert not np.array_equal(residual_a[0][mask], residual_b[0][mask])


def test_local_gaussian_noise_is_independent_and_only_changes_valid_voxels(
	tmp_path: Path,
) -> None:
	valid_mask = np.ones((2, 3, 4), dtype=bool)
	valid_mask[0, 1, 2] = False
	without_noise = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'without-noise',
			valid_mask=valid_mask,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		horizontal_flip_probability=0.5,
		gaussian_noise_std=0.0,
	)[0]
	noisy_dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'noisy',
			valid_mask=valid_mask,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		horizontal_flip_probability=0.5,
		gaussian_noise_std=0.25,
	)
	with_noise = noisy_dataset[0]

	for key in (
		'valid_mask_a',
		'valid_mask_b',
		'horizontal_flip_state_a',
		'horizontal_flip_state_b',
		'local_pair_indices_a',
		'local_pair_indices_b',
	):
		np.testing.assert_array_equal(with_noise[key], without_noise[key])
	residuals = []
	for suffix in ('a', 'b'):
		view_key = f'view_{suffix}'
		mask = with_noise[f'valid_mask_{suffix}']
		residual = with_noise[view_key] - without_noise[view_key]
		assert with_noise[view_key].dtype == np.float32
		np.testing.assert_array_equal(
			with_noise[view_key][0][~mask],
			without_noise[view_key][0][~mask],
		)
		assert np.any(residual[0][mask] != 0.0)
		residuals.append(residual)
	assert not np.array_equal(*residuals)

	repeated = noisy_dataset[0]
	for key in ('view_a', 'view_b'):
		np.testing.assert_array_equal(repeated[key], with_noise[key])


def test_local_trace_drop_zero_preserves_sample_bytes_and_skips_rng_calls(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	default_dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'default',
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		gaussian_noise_std=0.25,
	)
	explicit_zero_dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'explicit-zero',
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		gaussian_noise_std=0.25,
		trace_drop_probability=0.0,
	)
	apply_trace_drop = Mock(
		side_effect=AssertionError('p=0 must not sample trace-drop RNG')
	)
	monkeypatch.setattr(
		barlow_dataset_module,
		'_apply_trace_drop',
		apply_trace_drop,
	)

	for index in range(len(default_dataset)):
		default_sample = default_dataset[index]
		explicit_sample = explicit_zero_dataset[index]
		assert default_sample.keys() == explicit_sample.keys()
		for key, expected in default_sample.items():
			actual = explicit_sample[key]
			if isinstance(expected, np.ndarray):
				assert isinstance(actual, np.ndarray)
				assert actual.dtype == expected.dtype
				assert actual.shape == expected.shape
				assert actual.tobytes() == expected.tobytes()
			else:
				assert actual == expected
	apply_trace_drop.assert_not_called()


def test_local_trace_drop_one_runs_after_noise_without_changing_contract(
	tmp_path: Path,
) -> None:
	valid_mask = np.ones((2, 3, 4), dtype=bool)
	valid_mask[0, 1, :] = False
	without_drop = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'without-drop',
			valid_mask=valid_mask,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		gaussian_noise_std=0.25,
		trace_drop_probability=0.0,
	)[0]
	with_drop = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'with-drop',
			valid_mask=valid_mask,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		gaussian_noise_std=0.25,
		trace_drop_probability=1.0,
	)[0]
	expected_keys = {
		'view_a',
		'view_b',
		'valid_mask_a',
		'valid_mask_b',
		'coords',
		'horizontal_flip_state_a',
		'horizontal_flip_state_b',
		'local_pair_indices_a',
		'local_pair_indices_b',
	}

	assert set(with_drop) == expected_keys == set(without_drop)
	for key in (
		'valid_mask_a',
		'valid_mask_b',
		'horizontal_flip_state_a',
		'horizontal_flip_state_b',
		'local_pair_indices_a',
		'local_pair_indices_b',
	):
		np.testing.assert_array_equal(with_drop[key], without_drop[key])
	for suffix in ('a', 'b'):
		mask = np.asarray(with_drop[f'valid_mask_{suffix}'])
		eligible_xy = mask.any(axis=2)
		view = np.asarray(with_drop[f'view_{suffix}'])
		baseline = np.asarray(without_drop[f'view_{suffix}'])
		assert eligible_xy.any()
		assert np.all(view[:, eligible_xy, :] == 0.0)
		np.testing.assert_array_equal(
			view[:, ~eligible_xy, :],
			baseline[:, ~eligible_xy, :],
		)


def test_local_trace_drop_is_independent_and_deterministic(tmp_path: Path) -> None:
	dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			samples_per_epoch=16,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		trace_drop_probability=0.5,
	)
	first = [dataset[index] for index in range(len(dataset))]

	for index, expected in enumerate(first):
		actual = dataset[index]
		for key, value in expected.items():
			if isinstance(value, np.ndarray):
				np.testing.assert_array_equal(actual[key], value)
			else:
				assert actual[key] == value
	assert any(
		not np.array_equal(
			np.all(np.asarray(sample['view_a']) == 0.0, axis=(0, 3)),
			np.all(np.asarray(sample['view_b']) == 0.0, axis=(0, 3)),
		)
		for sample in first
	)


def test_zero_phase_z_filter_is_centered_and_preserves_dc() -> None:
	impulse = np.zeros((1, 1, 1, 5), dtype=np.float32)
	impulse[..., 2] = 1.0
	valid_mask = np.ones((1, 1, 5), dtype=bool)

	barlow_dataset_module._apply_zero_phase_z_filter(  # noqa: SLF001
		impulse,
		valid_mask,
		side_weight=0.125,
	)

	np.testing.assert_allclose(
		impulse,
		np.asarray([[[[0.0, 0.125, 0.75, 0.125, 0.0]]]], dtype=np.float32),
		rtol=0.0,
		atol=1e-7,
	)
	constant = np.ones((1, 2, 3, 5), dtype=np.float32)
	barlow_dataset_module._apply_zero_phase_z_filter(  # noqa: SLF001
		constant,
		np.ones((2, 3, 5), dtype=bool),
		side_weight=0.125,
	)
	np.testing.assert_allclose(constant, 1.0, rtol=0.0, atol=1e-7)


def test_zero_phase_z_filter_does_not_cross_invalid_z_gaps() -> None:
	view = np.asarray([[[[1.0, 2.0, np.nan, 100.0, 5.0]]]], dtype=np.float32)
	valid_mask = np.asarray([[[True, True, False, True, True]]])

	barlow_dataset_module._apply_zero_phase_z_filter(  # noqa: SLF001
		view,
		valid_mask,
		side_weight=0.125,
	)

	np.testing.assert_allclose(
		view[..., [0, 1, 3, 4]],
		np.asarray(
			[[[[8.0 / 7.0, 13.0 / 7.0, 605.0 / 7.0, 130.0 / 7.0]]]],
			dtype=np.float32,
		),
		rtol=0.0,
		atol=1e-6,
	)
	assert np.isnan(view[..., 2]).all()


def test_local_zero_phase_z_filter_assigns_one_view_symmetrically_and_deterministically(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			samples_per_epoch=16,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		z_filter_side_weight=0.125,
	)
	original_filter = barlow_dataset_module._apply_zero_phase_z_filter  # noqa: SLF001
	filtered_views: list[np.ndarray] = []

	def record_filter(
		view: np.ndarray,
		valid_mask: np.ndarray,
		*,
		side_weight: float,
	) -> None:
		filtered_views.append(view)
		original_filter(view, valid_mask, side_weight=side_weight)

	monkeypatch.setattr(
		barlow_dataset_module,
		'_apply_zero_phase_z_filter',
		record_filter,
	)
	first = [dataset[index] for index in range(len(dataset))]

	def assignment_for(samples: list[dict[str, object]]) -> list[str]:
		assignments: list[str] = []
		for sample, filtered_view in zip(samples, filtered_views, strict=True):
			if filtered_view is sample['view_a']:
				assignments.append('a')
			elif filtered_view is sample['view_b']:
				assignments.append('b')
			else:
				raise AssertionError('filter target was not returned as a view')
		return assignments

	first_assignments = assignment_for(first)
	assert set(first_assignments) == {'a', 'b'}
	assert len(filtered_views) == len(first)
	filtered_views.clear()
	second = [dataset[index] for index in range(len(dataset))]
	second_assignments = assignment_for(second)
	assert second_assignments == first_assignments
	for expected, actual in zip(first, second, strict=True):
		for key, value in expected.items():
			if isinstance(value, np.ndarray):
				np.testing.assert_array_equal(actual[key], value)
			else:
				assert actual[key] == value


def test_local_zero_phase_z_filter_zero_preserves_legacy_sample_bytes(
	tmp_path: Path,
) -> None:
	default_dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(tmp_path / 'default', min_valid_token_count=4),
		local_pairs_per_crop=4,
		gaussian_noise_std=0.25,
	)
	explicit_zero_dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(tmp_path / 'explicit-zero', min_valid_token_count=4),
		local_pairs_per_crop=4,
		gaussian_noise_std=0.25,
		z_filter_side_weight=0.0,
	)

	for index in range(len(default_dataset)):
		default_sample = default_dataset[index]
		explicit_zero_sample = explicit_zero_dataset[index]
		assert default_sample.keys() == explicit_zero_sample.keys()
		for key, expected in default_sample.items():
			actual = explicit_zero_sample[key]
			if isinstance(expected, np.ndarray):
				assert isinstance(actual, np.ndarray)
				assert actual.tobytes() == expected.tobytes()
			else:
				assert actual == expected


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


def test_xy_d4_ids_produce_eight_distinct_square_permutations() -> None:
	grid = np.arange(9, dtype=np.int64).reshape(3, 3, 1)
	permutations = {
		tuple(
			barlow_dataset_module._apply_xy_d4(  # noqa: SLF001
				grid,
				transform_id,
				xy_axes=(0, 1),
			).ravel()
		)
		for transform_id in range(8)
	}

	assert len(permutations) == 8


def test_xy_d4_rotation_geometry_matches_contract() -> None:
	grid = np.arange(2 * 2 * 3, dtype=np.int64).reshape(2, 2, 3)

	r180 = barlow_dataset_module._apply_xy_d4(  # noqa: SLF001
		grid,
		2,
		xy_axes=(0, 1),
	)
	r90 = barlow_dataset_module._apply_xy_d4(  # noqa: SLF001
		grid,
		1,
		xy_axes=(0, 1),
	)
	r270 = barlow_dataset_module._apply_xy_d4(  # noqa: SLF001
		grid,
		3,
		xy_axes=(0, 1),
	)

	np.testing.assert_array_equal(r180, grid[::-1, ::-1, :])
	np.testing.assert_array_equal(r90[0, 0], grid[0, 1])
	np.testing.assert_array_equal(r90[1, 0], grid[0, 0])
	np.testing.assert_array_equal(r270[0, 0], grid[1, 0])
	np.testing.assert_array_equal(r270[0, 1], grid[0, 0])


@pytest.mark.parametrize('transform_id', range(8))
def test_xy_d4_uses_same_physical_transform_and_preserves_z(
	transform_id: int,
) -> None:
	physical_ids = np.arange(3 * 3, dtype=np.int64).reshape(3, 3, 1)
	z_offsets = np.arange(4, dtype=np.int64).reshape(1, 1, 4)
	mask_values = physical_ids * 10 + z_offsets
	amplitude = mask_values[None, ...]
	token_grid = physical_ids.copy()

	transformed_amplitude = barlow_dataset_module._apply_xy_d4(  # noqa: SLF001
		amplitude,
		transform_id,
		xy_axes=(1, 2),
	)
	transformed_mask = barlow_dataset_module._apply_xy_d4(  # noqa: SLF001
		mask_values,
		transform_id,
		xy_axes=(0, 1),
	)
	transformed_tokens = barlow_dataset_module._apply_xy_d4(  # noqa: SLF001
		token_grid,
		transform_id,
		xy_axes=(0, 1),
	)

	np.testing.assert_array_equal(transformed_amplitude[0], transformed_mask)
	np.testing.assert_array_equal(
		transformed_mask[..., 0] // 10,
		transformed_tokens[..., 0],
	)
	assert np.all(np.diff(transformed_mask, axis=2) == 1)


def test_d4_dataset_rejects_non_square_raw_or_patch_xy(tmp_path: Path) -> None:
	with pytest.raises(ValueError, match='local crop X/Y sizes must be equal'):
		LocalBarlowTwinsD4TraceDropPretrainDataset(
			_base_dataset(tmp_path, min_valid_token_count=4),
			local_pairs_per_crop=4,
			reflection_probability=0.5,
			trace_drop_probability=0.02,
		)

	base = _square_d4_base_dataset(
		tmp_path,
		patch_size_xyz=(1, 2, 2),
		min_valid_token_count=4,
	)
	with pytest.raises(ValueError, match='patch size X/Y sizes must be equal'):
		LocalBarlowTwinsD4TraceDropPretrainDataset(
			base,
			local_pairs_per_crop=4,
			reflection_probability=0.5,
			trace_drop_probability=0.02,
		)


@pytest.mark.parametrize('name', ['reflection_probability', 'trace_drop_probability'])
@pytest.mark.parametrize('value', [-0.1, 1.1, np.inf, np.nan, True])
def test_d4_dataset_rejects_invalid_probabilities(
	tmp_path: Path,
	name: str,
	value: object,
) -> None:
	kwargs: dict[str, object] = {
		'local_pairs_per_crop': 4,
		'reflection_probability': 0.5,
		'trace_drop_probability': 0.02,
	}
	kwargs[name] = value

	with pytest.raises((TypeError, ValueError), match=name):
		LocalBarlowTwinsD4TraceDropPretrainDataset(
			_square_d4_base_dataset(tmp_path),
			**kwargs,  # type: ignore[arg-type]
		)


def test_d4_pair_mapping_selects_same_physical_patch_for_all_view_pairs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset = LocalBarlowTwinsD4TraceDropPretrainDataset(
		_square_d4_base_dataset(tmp_path),
		local_pairs_per_crop=8,
		reflection_probability=0.5,
		trace_drop_probability=0.0,
	)

	for transform_id_a in range(8):
		for transform_id_b in range(8):
			monkeypatch.setattr(
				barlow_dataset_module,
				'_sample_xy_d4_transform_id',
				Mock(side_effect=(transform_id_a, transform_id_b)),
			)
			sample = dataset[0]
			patches_a = patchify_3d(
				torch.as_tensor(sample['view_a']).unsqueeze(0),
				(2, 2, 2),
			)[0]
			patches_b = patchify_3d(
				torch.as_tensor(sample['view_b']).unsqueeze(0),
				(2, 2, 2),
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
			assert set(selected_a[:, 0, 0].round().tolist()) == set(range(1, 9))


def test_d4_pair_candidates_exclude_not_fully_valid_tokens(tmp_path: Path) -> None:
	valid_mask = np.ones((4, 4, 4), dtype=bool)
	valid_mask[2:4, 0:2, 0:2] = False
	dataset = LocalBarlowTwinsD4TraceDropPretrainDataset(
		_square_d4_base_dataset(
			tmp_path,
			valid_mask=valid_mask,
			min_valid_token_count=7,
		),
		local_pairs_per_crop=7,
		reflection_probability=0.5,
		trace_drop_probability=0.0,
	)

	sample = dataset[0]
	patches = patchify_3d(
		torch.as_tensor(sample['view_a']).unsqueeze(0),
		(2, 2, 2),
	)[0]
	selected = patches.index_select(
		0,
		torch.as_tensor(sample['local_pair_indices_a']),
	)

	assert set(selected[:, 0, 0].round().tolist()) == set(range(1, 9)) - {5}


def test_d4_trace_drop_zero_does_not_change_transformed_amplitude(
	tmp_path: Path,
) -> None:
	base = _square_d4_base_dataset(tmp_path)
	dataset = LocalBarlowTwinsD4TraceDropPretrainDataset(
		base,
		local_pairs_per_crop=4,
		reflection_probability=0.5,
		trace_drop_probability=0.0,
	)
	base_sample = base[0]
	sample = dataset[0]

	for suffix in ('a', 'b'):
		expected = barlow_dataset_module._apply_xy_d4(  # noqa: SLF001
			base_sample['x'],
			int(sample[f'xy_transform_id_{suffix}']),
			xy_axes=(1, 2),
		)
		np.testing.assert_array_equal(sample[f'view_{suffix}'], expected)
		assert int(sample[f'trace_drop_count_{suffix}']) == 0


def test_d4_trace_drop_one_zeros_only_eligible_traces_and_counts_them(
	tmp_path: Path,
) -> None:
	valid_mask = np.ones((4, 4, 4), dtype=bool)
	valid_mask[1, 2, :] = False
	dataset = LocalBarlowTwinsD4TraceDropPretrainDataset(
		_square_d4_base_dataset(
			tmp_path,
			valid_mask=valid_mask,
			min_valid_token_count=6,
		),
		local_pairs_per_crop=4,
		reflection_probability=0.5,
		trace_drop_probability=1.0,
	)

	sample = dataset[0]
	for suffix in ('a', 'b'):
		view = sample[f'view_{suffix}']
		transformed_mask = sample[f'valid_mask_{suffix}']
		eligible_xy = transformed_mask.any(axis=2)
		assert np.all(view[:, eligible_xy, :] == 0.0)
		assert int(sample[f'trace_drop_count_{suffix}']) == 15


def test_trace_drop_keeps_masks_and_pair_indices_and_can_drop_selected_pairs(
	tmp_path: Path,
) -> None:
	base_zero = _square_d4_base_dataset(tmp_path / 'zero')
	base_one = _square_d4_base_dataset(tmp_path / 'one')
	without_drop = LocalBarlowTwinsD4TraceDropPretrainDataset(
		base_zero,
		local_pairs_per_crop=8,
		reflection_probability=0.5,
		trace_drop_probability=0.0,
	)[0]
	with_drop = LocalBarlowTwinsD4TraceDropPretrainDataset(
		base_one,
		local_pairs_per_crop=8,
		reflection_probability=0.5,
		trace_drop_probability=1.0,
	)[0]

	for suffix in ('a', 'b'):
		np.testing.assert_array_equal(
			without_drop[f'valid_mask_{suffix}'],
			with_drop[f'valid_mask_{suffix}'],
		)
		np.testing.assert_array_equal(
			without_drop[f'local_pair_indices_{suffix}'],
			with_drop[f'local_pair_indices_{suffix}'],
		)
		patches = patchify_3d(
			torch.as_tensor(with_drop[f'view_{suffix}']).unsqueeze(0),
			(2, 2, 2),
		)[0]
		selected = patches.index_select(
			0,
			torch.as_tensor(with_drop[f'local_pair_indices_{suffix}']),
		)
		assert torch.count_nonzero(selected) == 0


def test_d4_view_trace_drop_sampling_is_independent(tmp_path: Path) -> None:
	dataset = LocalBarlowTwinsD4TraceDropPretrainDataset(
		_square_d4_base_dataset(tmp_path, samples_per_epoch=16),
		local_pairs_per_crop=4,
		reflection_probability=0.5,
		trace_drop_probability=0.5,
	)

	assert any(
		int(sample['trace_drop_count_a']) != int(sample['trace_drop_count_b'])
		for sample in (dataset[index] for index in range(len(dataset)))
	)


def test_d4_seed_epoch_index_determinism_and_epoch_reset(tmp_path: Path) -> None:
	dataset = LocalBarlowTwinsD4TraceDropPretrainDataset(
		_square_d4_base_dataset(tmp_path),
		local_pairs_per_crop=4,
		reflection_probability=0.5,
		trace_drop_probability=0.25,
	)
	epoch_zero = [dataset[index] for index in range(len(dataset))]

	for index, expected in enumerate(epoch_zero):
		actual = dataset[index]
		for key, value in expected.items():
			if isinstance(value, np.ndarray):
				np.testing.assert_array_equal(actual[key], value)
			else:
				assert actual[key] == value

	dataset.set_epoch(1)
	epoch_one = [dataset[index] for index in range(len(dataset))]
	metadata_keys = (
		'xy_transform_id_a',
		'xy_transform_id_b',
		'trace_drop_count_a',
		'trace_drop_count_b',
		'local_pair_indices_a',
		'local_pair_indices_b',
	)
	assert any(
		any(not np.array_equal(zero[key], one[key]) for key in metadata_keys)
		for zero, one in zip(epoch_zero, epoch_one, strict=True)
	)

	dataset.set_epoch(0)
	for index, expected in enumerate(epoch_zero):
		for key in metadata_keys:
			np.testing.assert_array_equal(dataset[index][key], expected[key])


def test_d4_multi_worker_batch_and_collate_metadata(tmp_path: Path) -> None:
	dataset = LocalBarlowTwinsD4TraceDropPretrainDataset(
		_square_d4_base_dataset(tmp_path),
		local_pairs_per_crop=4,
		reflection_probability=0.5,
		trace_drop_probability=0.25,
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
	expected_keys = {
		'view_a',
		'view_b',
		'valid_mask_a',
		'valid_mask_b',
		'coords',
		'xy_transform_id_a',
		'xy_transform_id_b',
		'trace_drop_count_a',
		'trace_drop_count_b',
		'local_pair_indices_a',
		'local_pair_indices_b',
	}
	assert set(single_batch) == expected_keys
	for key in expected_keys - {'coords'}:
		torch.testing.assert_close(single_batch[key], multi_batch[key])
	assert single_batch['coords'] == multi_batch['coords']
	assert single_batch['xy_transform_id_a'].dtype is torch.int64
	assert single_batch['trace_drop_count_a'].dtype is torch.int64


def test_barlow_collate_rejects_partial_or_mixed_local_contracts(
	tmp_path: Path,
) -> None:
	base = _square_d4_base_dataset(tmp_path)
	d4_sample = LocalBarlowTwinsD4TraceDropPretrainDataset(
		base,
		local_pairs_per_crop=4,
		reflection_probability=0.5,
		trace_drop_probability=0.25,
	)[0]
	partial = dict(d4_sample)
	del partial['trace_drop_count_b']
	legacy_base = _base_dataset(tmp_path / 'legacy', min_valid_token_count=4)
	legacy_sample = LocalBarlowTwinsPretrainDataset(
		legacy_base,
		local_pairs_per_crop=4,
	)[0]

	with pytest.raises(ValueError, match='complete Barlow Twins contract'):
		barlow_twins_collate_fn([partial])
	with pytest.raises(ValueError, match='complete Barlow Twins contract'):
		barlow_twins_collate_fn([legacy_sample, d4_sample])


def test_dataloader_rejects_batch_size_one(tmp_path: Path) -> None:
	dataset = BarlowTwinsPretrainDataset(_base_dataset(tmp_path))

	with pytest.raises(ValueError, match='at least 2'):
		build_barlow_twins_dataloader(dataset, batch_size=1)
