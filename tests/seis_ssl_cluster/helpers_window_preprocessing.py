from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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
	write_normalization_stats,
)
from seis_ssl_cluster.data.normalization import AmplitudeAgcConfig
from seis_ssl_cluster.embedding import SlidingWindow

if TYPE_CHECKING:
	from pathlib import Path

PATCH_SIZE_XYZ = (2, 2, 2)
WINDOW_SIZE_XYZ = (4, 4, 4)


@dataclass(frozen=True)
class WindowPreprocessingFixture:
	volume: np.ndarray
	manifest: SurveyManifest
	amplitude_path: Path
	stats: SurveyNormalizationStats
	zero_mask: ZeroMaskConfig
	normalized_clip_abs: float | None
	amplitude_agc: AmplitudeAgcConfig
	window: SlidingWindow
	request: CropRequest


def write_window_preprocessing_fixture(
	tmp_path: Path,
) -> WindowPreprocessingFixture:
	volume = np.arange(1, 4 * 4 * 4 + 1, dtype=np.float32).reshape(WINDOW_SIZE_XYZ)
	volume[0, 0, :] = 0.0
	volume_root = tmp_path / 'survey'
	volume_root.mkdir()
	amplitude_path = volume_root / 'amplitude.npy'
	np.save(amplitude_path, volume)
	stats_path = volume_root / 'stats.json'
	stats = SurveyNormalizationStats(
		survey_id='survey',
		source_path=amplitude_path,
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=0.0,
		clip_high_percentile=100.0,
		clip_low=-1000.0,
		clip_high=1000.0,
		median=10.0,
		iqr=2.0,
	)
	write_normalization_stats(stats, stats_path)
	manifest = SurveyManifest(
		survey_id='survey',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=amplitude_path,
			shape_xyz=WINDOW_SIZE_XYZ,
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	window = SlidingWindow(start_xyz=(0, 0, 0), size_xyz=WINDOW_SIZE_XYZ)
	return WindowPreprocessingFixture(
		volume=volume,
		manifest=manifest,
		amplitude_path=amplitude_path,
		stats=stats,
		zero_mask=ZeroMaskConfig(
			enabled=True,
			zero_atol=0.0,
			z_sample_influence_radius=0,
			xy_trace_influence_radius=0,
		),
		normalized_clip_abs=8.0,
		amplitude_agc=AmplitudeAgcConfig(enabled=False),
		window=window,
		request=CropRequest(
			survey_id=manifest.survey_id,
			start_xyz=window.start_xyz,
			size_xyz=window.size_xyz,
		),
	)


def read_fixture_crop(
	fixture: WindowPreprocessingFixture,
	*,
	min_token_valid_fraction: float,
):
	return read_amplitude_crop(
		request=fixture.request,
		amplitude_path=fixture.amplitude_path,
		stats=fixture.stats,
		store=NpyMemmapVolumeStore(),
		patch_size_xyz=PATCH_SIZE_XYZ,
		settings=AmplitudePreprocessSettings(
			zero_mask=fixture.zero_mask,
			normalized_clip_abs=fixture.normalized_clip_abs,
			amplitude_agc=fixture.amplitude_agc,
			min_token_valid_fraction=min_token_valid_fraction,
		),
	)
