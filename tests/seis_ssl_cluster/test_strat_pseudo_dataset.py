from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	NopimsAmplitudeCropDataset,
	NopimsAmplitudePretrainDataset,
	NopimsStratPseudoTargetDataset,
	StratPseudoTargetProvider,
	SurveyManifest,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	write_normalization_stats,
)
from seis_ssl_cluster.stratigraphy.targets import (
	discover_pseudo_target_inputs,
	pseudo_target_paths,
	write_pseudo_target,
)
from seis_ssl_cluster.training.collate import strat_pseudo_target_collate_fn
from tests.seis_ssl_cluster.helpers_window_preprocessing import (
	PATCH_SIZE_XYZ,
	WINDOW_SIZE_XYZ,
	read_fixture_crop,
	write_window_preprocessing_fixture,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_strat_dataset_slices_pseudo_targets_from_token_start(
	tmp_path: Path,
) -> None:
	volume = np.ones((10, 8, 8), dtype=np.float32)
	labels = np.arange(5 * 4 * 4, dtype=np.int32).reshape(5, 4, 4)
	boundary_weight = np.linspace(
		0.0,
		1.0,
		labels.size,
		dtype=np.float32,
	).reshape(labels.shape)
	dataset = _dataset(
		tmp_path,
		volume=volume,
		labels=labels,
		valid_tokens=np.ones(labels.shape, dtype=np.bool_),
		boundary_weight=boundary_weight,
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		seed=4,
	)

	sample = dataset[0]

	token_start = sample['coords']['token_start_xyz']
	token_size = sample['coords']['token_size_xyz']
	expected_slices = tuple(
		slice(start, start + size)
		for start, size in zip(token_start, token_size, strict=True)
	)
	np.testing.assert_array_equal(sample['strat_labels'], labels[expected_slices])
	np.testing.assert_array_equal(
		sample['strat_boundary_weight'],
		boundary_weight[expected_slices],
	)
	assert sample['coords']['local_start_xyz'] == (4, 4, 4)
	assert token_start == (2, 2, 2)
	assert token_size == (2, 2, 2)
	assert sample['strat_valid_mask'].all()


def test_wrapper_matches_generic_dataset_with_strat_provider(
	tmp_path: Path,
) -> None:
	volume = np.arange(10 * 8 * 8, dtype=np.float32).reshape(10, 8, 8) + 1.0
	labels = np.arange(5 * 4 * 4, dtype=np.int32).reshape(5, 4, 4)
	valid_tokens = np.ones(labels.shape, dtype=np.bool_)
	valid_tokens[3, 2, 2] = False
	labels[~valid_tokens] = -1
	confidence = np.linspace(
		0.5,
		1.0,
		labels.size,
		dtype=np.float32,
	).reshape(labels.shape)
	confidence[~valid_tokens] = 0.0
	manifest = _manifest(tmp_path, 'survey', volume)
	write_pseudo_target(
		tmp_path / 'pseudo-composition',
		k=int(labels.max()) + 1,
		survey_id='survey',
		labels=labels,
		confidence=confidence,
		valid_tokens=valid_tokens,
	)
	pseudo_target_inputs = discover_pseudo_target_inputs(
		tmp_path / 'pseudo-composition',
		k=int(labels.max()) + 1,
	)
	wrapper = NopimsStratPseudoTargetDataset(
		[manifest],
		pseudo_target_inputs,
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		seed=4,
		zero_mask=ZeroMaskConfig(enabled=False),
	)
	composed = NopimsAmplitudeCropDataset(
		[manifest],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		seed=4,
		zero_mask=ZeroMaskConfig(enabled=False),
		target_provider=StratPseudoTargetProvider(pseudo_target_inputs),
	)

	wrapper_sample = wrapper[0]
	composed_sample = composed[0]

	assert set(wrapper_sample) == {
		'x',
		'local_valid_mask',
		'strat_labels',
		'strat_confidence',
		'strat_boundary_weight',
		'strat_valid_mask',
		'coords',
	}
	assert set(composed_sample) == set(wrapper_sample)
	for key in (
		'x',
		'local_valid_mask',
		'strat_labels',
		'strat_confidence',
		'strat_boundary_weight',
		'strat_valid_mask',
	):
		np.testing.assert_array_equal(wrapper_sample[key], composed_sample[key])
	assert wrapper_sample['coords'] == composed_sample['coords']
	for sample in (wrapper_sample, composed_sample):
		assert '_token_valid_mask' not in sample
		batch = strat_pseudo_target_collate_fn([sample])
		assert '_token_valid_mask' not in batch


def test_strat_dataset_excludes_invalid_pseudo_tokens_and_sets_label_minus_one(
	tmp_path: Path,
) -> None:
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	labels[0, 0, 0] = -1
	valid_tokens = np.ones(labels.shape, dtype=np.bool_)
	valid_tokens[0, 0, 0] = False
	dataset = _dataset(
		tmp_path,
		volume=np.ones((4, 4, 4), dtype=np.float32),
		labels=labels,
		valid_tokens=valid_tokens,
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
	)

	sample = dataset[0]

	assert sample['strat_labels'][0, 0, 0] == -1
	assert sample['strat_confidence'][0, 0, 0] == 0.0
	assert sample['strat_boundary_weight'][0, 0, 0] == 0.0
	assert not sample['strat_valid_mask'][0, 0, 0]
	assert np.count_nonzero(sample['strat_valid_mask']) == 7


def test_strat_dataset_excludes_local_invalid_voxel_patches(
	tmp_path: Path,
) -> None:
	volume = np.ones((4, 4, 4), dtype=np.float32)
	volume[:, :, 1] = 0.0
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	dataset = _dataset(
		tmp_path,
		volume=volume,
		labels=labels,
		valid_tokens=np.ones(labels.shape, dtype=np.bool_),
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		zero_mask=ZeroMaskConfig(
			z_sample_influence_radius=0,
			xy_trace_influence_radius=0,
		),
	)

	sample = dataset[0]

	assert not sample['local_valid_mask'][:, :, 1].any()
	assert not sample['strat_valid_mask'][:, :, 0].any()
	assert sample['strat_valid_mask'][:, :, 1].all()
	assert np.all(sample['strat_labels'][:, :, 0] == -1)
	assert np.all(sample['strat_boundary_weight'][:, :, 0] == 0.0)


def test_strat_dataset_preprocessing_matches_shared_contract(
	tmp_path: Path,
) -> None:
	fixture = write_window_preprocessing_fixture(tmp_path)
	expected_dataset = read_fixture_crop(fixture, min_token_valid_fraction=1.0)
	expected_extractor_builder = read_fixture_crop(
		fixture,
		min_token_valid_fraction=0.5,
	)
	labels = np.zeros(expected_dataset.token_valid_mask.shape, dtype=np.int32)
	valid_tokens = np.ones(labels.shape, dtype=np.bool_)
	write_pseudo_target(
		tmp_path / 'pseudo-contract',
		k=1,
		survey_id=fixture.manifest.survey_id,
		labels=labels,
		confidence=np.ones(labels.shape, dtype=np.float32),
		valid_tokens=valid_tokens,
	)
	dataset = NopimsStratPseudoTargetDataset(
		[fixture.manifest],
		discover_pseudo_target_inputs(tmp_path / 'pseudo-contract', k=1),
		local_crop_size_xyz=WINDOW_SIZE_XYZ,
		patch_size_xyz=PATCH_SIZE_XYZ,
		zero_mask=fixture.zero_mask,
		normalized_clip_abs=fixture.normalized_clip_abs,
		amplitude_agc=fixture.amplitude_agc,
	)

	sample = dataset[0]

	assert sample['coords']['local_start_xyz'] == fixture.window.start_xyz
	np.testing.assert_allclose(sample['x'], expected_dataset.x, rtol=1.0e-6)
	np.testing.assert_array_equal(
		sample['local_valid_mask'],
		expected_dataset.local_valid_mask,
	)
	np.testing.assert_array_equal(
		sample['strat_valid_mask'],
		expected_dataset.token_valid_mask,
	)
	assert expected_extractor_builder.token_valid_mask[0, 0, 0]
	assert not sample['strat_valid_mask'][0, 0, 0]
	np.testing.assert_array_equal(
		sample['x'][0][~expected_dataset.local_valid_mask],
		0.0,
	)


def test_strat_dataset_samples_have_at_least_one_valid_supervised_token(
	tmp_path: Path,
) -> None:
	labels = np.full((2, 2, 2), -1, dtype=np.int32)
	labels[1, 1, 1] = 0
	valid_tokens = np.zeros(labels.shape, dtype=np.bool_)
	valid_tokens[1, 1, 1] = True
	dataset = _dataset(
		tmp_path,
		volume=np.ones((4, 4, 4), dtype=np.float32),
		labels=labels,
		valid_tokens=valid_tokens,
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
	)

	sample = dataset[0]

	assert np.any(sample['strat_valid_mask'])
	assert np.count_nonzero(sample['strat_valid_mask']) == 1


def test_strat_dataset_rejects_valid_tokens_below_min_confidence(
	tmp_path: Path,
) -> None:
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	valid_tokens = np.ones(labels.shape, dtype=np.bool_)
	dataset = _dataset(
		tmp_path,
		volume=np.ones((4, 4, 4), dtype=np.float32),
		labels=labels,
		valid_tokens=valid_tokens,
		confidence=np.full(labels.shape, 0.25, dtype=np.float32),
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		min_confidence=0.5,
		max_resample_attempts=2,
	)

	with pytest.raises(ValueError, match=r'min_confidence=0\.500000'):
		dataset[0]


def test_strat_dataset_rejects_crop_with_only_zero_boundary_weight(
	tmp_path: Path,
) -> None:
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	valid_tokens = np.ones(labels.shape, dtype=np.bool_)
	dataset = _dataset(
		tmp_path,
		volume=np.ones((4, 4, 4), dtype=np.float32),
		labels=labels,
		valid_tokens=valid_tokens,
		boundary_weight=np.zeros(labels.shape, dtype=np.float32),
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		max_resample_attempts=2,
	)

	with pytest.raises(ValueError, match='positive boundary/effective weight'):
		dataset[0]


def test_strat_dataset_schema_v1_uses_unity_boundary_weight(
	tmp_path: Path,
) -> None:
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	valid_tokens = np.ones(labels.shape, dtype=np.bool_)
	manifest = _manifest(tmp_path, 'survey', np.ones((4, 4, 4), dtype=np.float32))
	paths = pseudo_target_paths(tmp_path / 'pseudo-v1', k=1, survey_id='survey')
	paths.labels.parent.mkdir(parents=True)
	np.save(paths.labels, labels)
	np.save(paths.confidence, np.ones(labels.shape, dtype=np.float32))
	np.save(paths.valid_tokens, valid_tokens)
	paths.metadata.write_text(
		json.dumps({'schema_version': 1}) + '\n',
		encoding='utf-8',
	)
	dataset = NopimsStratPseudoTargetDataset(
		[manifest],
		discover_pseudo_target_inputs(tmp_path / 'pseudo-v1', k=1),
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	sample = dataset[0]

	np.testing.assert_array_equal(
		sample['strat_boundary_weight'],
		np.ones(labels.shape, dtype=np.float32),
	)


def test_strat_dataset_allows_extra_pseudo_target_surveys(
	tmp_path: Path,
) -> None:
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	valid_tokens = np.ones(labels.shape, dtype=np.bool_)
	write_pseudo_target(
		tmp_path / 'pseudo',
		k=1,
		survey_id='survey',
		labels=labels,
		confidence=np.ones(labels.shape, dtype=np.float32),
		valid_tokens=valid_tokens,
	)
	write_pseudo_target(
		tmp_path / 'pseudo',
		k=1,
		survey_id='heldout',
		labels=labels,
		confidence=np.ones(labels.shape, dtype=np.float32),
		valid_tokens=valid_tokens,
	)
	dataset = NopimsStratPseudoTargetDataset(
		[_manifest(tmp_path, 'survey', np.ones((4, 4, 4), dtype=np.float32))],
		discover_pseudo_target_inputs(tmp_path / 'pseudo', k=1),
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	sample = dataset[0]

	assert sample['coords']['survey_id'] == 'survey'
	assert sample['strat_valid_mask'].all()


def test_strat_dataset_raises_when_no_valid_supervised_crop(
	tmp_path: Path,
) -> None:
	labels = np.full((2, 2, 2), -1, dtype=np.int32)
	dataset = _dataset(
		tmp_path,
		volume=np.ones((4, 4, 4), dtype=np.float32),
		labels=labels,
		valid_tokens=np.zeros(labels.shape, dtype=np.bool_),
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		max_resample_attempts=2,
	)

	with pytest.raises(ValueError, match='at least one valid supervised token'):
		dataset[0]


def test_strat_pseudo_target_collate_stacks_pseudo_target_fields(
	tmp_path: Path,
) -> None:
	labels = np.zeros((2, 2, 2), dtype=np.int32)
	dataset = _dataset(
		tmp_path,
		volume=np.ones((4, 4, 4), dtype=np.float32),
		labels=labels,
		valid_tokens=np.ones(labels.shape, dtype=np.bool_),
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
	)
	samples = [dataset[0], dataset[0]]

	batch = strat_pseudo_target_collate_fn(samples)

	assert set(batch) == {
		'x',
		'local_valid_mask',
		'strat_labels',
		'strat_confidence',
		'strat_boundary_weight',
		'strat_valid_mask',
		'coords',
	}
	assert batch['x'].shape == (2, 1, 4, 4, 4)
	assert batch['local_valid_mask'].shape == (2, 4, 4, 4)
	assert batch['strat_labels'].shape == (2, 2, 2, 2)
	assert batch['strat_confidence'].shape == (2, 2, 2, 2)
	assert batch['strat_boundary_weight'].shape == (2, 2, 2, 2)
	assert batch['strat_valid_mask'].shape == (2, 2, 2, 2)
	assert batch['x'].dtype == torch.float32
	assert batch['local_valid_mask'].dtype == torch.bool
	assert batch['strat_labels'].dtype == torch.long
	assert batch['strat_confidence'].dtype == torch.float32
	assert batch['strat_boundary_weight'].dtype == torch.float32
	assert batch['strat_valid_mask'].dtype == torch.bool
	assert batch['coords'] == [sample['coords'] for sample in samples]


def test_original_amplitude_dataset_allows_non_token_aligned_crop_starts(
	tmp_path: Path,
) -> None:
	manifest = _manifest(tmp_path, 'survey', np.ones((10, 10, 10), dtype=np.float32))
	dataset = NopimsAmplitudePretrainDataset(
		[manifest],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		seed=0,
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	sample = dataset[0]

	assert sample['coords']['local_start_xyz'] == (5, 4, 3)


def _dataset(  # noqa: PLR0913
	tmp_path: Path,
	*,
	volume: np.ndarray,
	labels: np.ndarray,
	valid_tokens: np.ndarray,
	local_crop_size_xyz: tuple[int, int, int],
	patch_size_xyz: tuple[int, int, int],
	confidence: np.ndarray | None = None,
	boundary_weight: np.ndarray | None = None,
	seed: int = 0,
	zero_mask: ZeroMaskConfig | None = None,
	max_resample_attempts: int = 16,
	min_confidence: float = 0.0,
) -> NopimsStratPseudoTargetDataset:
	manifest = _manifest(tmp_path, 'survey', volume)
	if confidence is None:
		confidence_array = np.where(valid_tokens, 1.0, 0.0).astype(np.float32)
	else:
		confidence_array = np.asarray(confidence, dtype=np.float32)
	write_pseudo_target(
		tmp_path / 'pseudo',
		k=max(1, int(np.max(labels[valid_tokens])) + 1)
		if np.any(valid_tokens)
		else 1,
		survey_id='survey',
		labels=labels,
		confidence=confidence_array,
		valid_tokens=valid_tokens,
		boundary_weight=boundary_weight,
	)
	return NopimsStratPseudoTargetDataset(
		[manifest],
		discover_pseudo_target_inputs(
			tmp_path / 'pseudo',
			k=max(1, int(np.max(labels[valid_tokens])) + 1)
			if np.any(valid_tokens)
			else 1,
		),
		local_crop_size_xyz=local_crop_size_xyz,
		patch_size_xyz=patch_size_xyz,
		seed=seed,
		zero_mask=zero_mask or ZeroMaskConfig(enabled=False),
		max_resample_attempts=max_resample_attempts,
		min_confidence=min_confidence,
	)


def _manifest(
	tmp_path: Path,
	survey_id: str,
	volume: np.ndarray,
) -> SurveyManifest:
	volume_path = tmp_path / survey_id / 'base.npy'
	volume_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(volume_path, volume.astype(np.float32, copy=False))
	stats_path = tmp_path / 'stats' / f'{survey_id}.json'
	stats_path.parent.mkdir(parents=True, exist_ok=True)
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id=survey_id,
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1000.0,
			clip_high=1000.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
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
