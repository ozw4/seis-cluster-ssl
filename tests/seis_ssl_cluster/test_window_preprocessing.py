from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudePreprocessSettings,
	AmplitudeVolumeRecord,
	CropRequest,
	NpyMemmapVolumeStore,
	SurveyManifest,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	read_amplitude_crop,
	reduce_valid_mask_to_tokens,
	resolve_manifest_path,
	zero_mask_margin_xyz,
)
from seis_ssl_cluster.data.normalization import AmplitudeAgcConfig


def test_reduce_valid_mask_to_tokens_applies_thresholds() -> None:
	mask = np.zeros((6, 2, 2), dtype=bool)
	mask[0:2, :, :] = True
	mask[2, :, :] = True

	threshold_zero = reduce_valid_mask_to_tokens(
		mask,
		patch_size_xyz=(2, 2, 2),
		min_valid_fraction=0.0,
	)
	threshold_half = reduce_valid_mask_to_tokens(
		mask,
		patch_size_xyz=(2, 2, 2),
		min_valid_fraction=0.5,
	)
	threshold_one = reduce_valid_mask_to_tokens(
		mask,
		patch_size_xyz=(2, 2, 2),
		min_valid_fraction=1.0,
	)

	np.testing.assert_array_equal(threshold_zero[:, 0, 0], [True, True, True])
	np.testing.assert_array_equal(threshold_half[:, 0, 0], [True, True, False])
	np.testing.assert_array_equal(threshold_one[:, 0, 0], [True, False, False])


def test_reduce_valid_mask_to_tokens_rejects_non_divisible_shape() -> None:
	mask = np.ones((3, 4, 4), dtype=bool)

	with pytest.raises(ValueError, match='divisible'):
		reduce_valid_mask_to_tokens(
			mask,
			patch_size_xyz=(2, 2, 2),
			min_valid_fraction=0.5,
		)


def test_zero_mask_margin_xyz_respects_enabled_flag() -> None:
	assert zero_mask_margin_xyz(ZeroMaskConfig(enabled=False)) == (0, 0, 0)
	assert zero_mask_margin_xyz(
		ZeroMaskConfig(
			enabled=True,
			z_sample_influence_radius=2,
			xy_trace_influence_radius=3,
		),
	) == (3, 3, 2)


def test_resolve_manifest_path_handles_absolute_and_relative(tmp_path: Path) -> None:
	manifest = _manifest(tmp_path, np.ones((2, 2, 2), dtype=np.float32))
	absolute = tmp_path / 'absolute.npy'

	assert resolve_manifest_path(manifest, absolute) == absolute
	assert resolve_manifest_path(manifest, Path('relative.npy')) == (
		tmp_path / 'relative.npy'
	)


def test_read_amplitude_crop_returns_arrays_and_zero_fills_invalid_voxels(
	tmp_path: Path,
) -> None:
	volume = np.arange(3 * 3 * 3, dtype=np.float32).reshape((3, 3, 3))
	manifest = _manifest(tmp_path, volume)
	settings = AmplitudePreprocessSettings(
		zero_mask=ZeroMaskConfig(enabled=False),
		normalized_clip_abs=None,
		amplitude_agc=AmplitudeAgcConfig(),
		min_token_valid_fraction=0.5,
	)

	prepared = read_amplitude_crop(
		request=CropRequest(
			survey_id=manifest.survey_id,
			start_xyz=(-1, 0, 0),
			size_xyz=(2, 2, 2),
		),
		amplitude_path=manifest.amplitude.path,
		stats=_stats(manifest.amplitude.path),
		store=NpyMemmapVolumeStore(),
		patch_size_xyz=(1, 1, 1),
		settings=settings,
	)

	assert prepared.request.start_xyz == (-1, 0, 0)
	assert prepared.x.shape == (1, 2, 2, 2)
	assert prepared.x.dtype == np.float32
	assert prepared.local_valid_mask.shape == (2, 2, 2)
	assert prepared.local_valid_mask.dtype == np.bool_
	assert prepared.token_valid_mask.shape == (2, 2, 2)
	assert prepared.token_valid_mask.dtype == np.bool_
	assert not prepared.local_valid_mask[0, :, :].any()
	assert not prepared.token_valid_mask[0, :, :].any()
	np.testing.assert_array_equal(prepared.x[0, 0, :, :], 0.0)
	np.testing.assert_allclose(
		prepared.x[0, 1, :, :],
		volume[0, 0:2, 0:2],
		rtol=1.0e-5,
	)


def test_read_amplitude_crop_rejects_non_finite_source_voxels(
	tmp_path: Path,
) -> None:
	volume = np.ones((2, 2, 2), dtype=np.float32)
	volume[0, 0, 0] = np.nan
	manifest = _manifest(tmp_path, volume)

	with pytest.raises(ValueError, match='non-finite'):
		read_amplitude_crop(
			request=CropRequest(
				survey_id=manifest.survey_id,
				start_xyz=(0, 0, 0),
				size_xyz=(2, 2, 2),
			),
			amplitude_path=manifest.amplitude.path,
			stats=_stats(manifest.amplitude.path),
			store=NpyMemmapVolumeStore(),
			patch_size_xyz=(1, 1, 1),
			settings=AmplitudePreprocessSettings(
				zero_mask=ZeroMaskConfig(enabled=False),
				normalized_clip_abs=None,
				amplitude_agc=AmplitudeAgcConfig(),
				min_token_valid_fraction=0.0,
			),
		)


def _manifest(tmp_path: Path, volume: np.ndarray) -> SurveyManifest:
	volume_path = tmp_path / 'survey.npy'
	np.save(volume_path, volume.astype(np.float32, copy=False))
	return SurveyManifest(
		survey_id='survey',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=tmp_path / 'stats.json',
		),
	)


def _stats(path: Path) -> SurveyNormalizationStats:
	return SurveyNormalizationStats(
		survey_id='survey',
		source_path=path,
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=0.0,
		clip_high_percentile=100.0,
		clip_low=-1000.0,
		clip_high=1000.0,
		median=0.0,
		iqr=1.0,
	)
