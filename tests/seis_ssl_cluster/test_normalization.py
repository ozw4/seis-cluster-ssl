from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	compute_normalization_stats,
	load_normalization_stats,
	normalize_amplitude,
	write_manifest_json,
	write_normalization_stats,
)
from tests.helpers import run_python_proc


def test_normalize_amplitude_clips_and_robust_scales() -> None:
	stats = SurveyNormalizationStats(
		survey_id='survey-a',
		source_path=Path('base.npy'),
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=0.5,
		clip_high_percentile=99.5,
		clip_low=-2.0,
		clip_high=6.0,
		median=2.0,
		iqr=2.0,
		eps=1.0e-6,
	)
	crop = np.asarray([-10.0, -2.0, 2.0, 6.0, 10.0], dtype=np.float32)

	normalized = normalize_amplitude(crop, stats)

	np.testing.assert_allclose(
		normalized,
		np.asarray([-2.0, -2.0, 0.0, 2.0, 2.0], dtype=np.float32),
		atol=1.0e-5,
		rtol=0.0,
	)


def test_compute_normalization_stats_samples_memmap_deterministically(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'volume.npy'
	volume = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
	np.save(path, volume)

	first = compute_normalization_stats(
		path,
		survey_id='survey-a',
		max_samples=100,
		seed=7,
	)
	second = compute_normalization_stats(
		path,
		survey_id='survey-a',
		max_samples=100,
		seed=7,
	)
	other = compute_normalization_stats(
		path,
		survey_id='survey-a',
		max_samples=100,
		seed=8,
	)

	assert first == second
	assert first != other
	assert first.source_path == path
	assert first.grid_order == GRID_ORDER_XYZ


def test_load_normalization_stats_rejects_legacy_center_scale(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'normalization_stats.json'
	path.write_text(json.dumps({'center': 0.0, 'scale': 1.0}), encoding='utf-8')

	with pytest.raises(TypeError, match='survey_id'):
		load_normalization_stats(path)


def test_prepare_nopims_normalization_stats_cli_writes_and_skips(
	tmp_path: Path,
) -> None:
	manifest_path, config_path, stats_paths, source_paths = _write_prepare_inputs(
		tmp_path,
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
	)

	assert result.returncode == 0, result.stderr
	assert f'normalization_stats.manifest_path: {manifest_path}' in result.stdout
	assert 'normalization_stats.written_files: 2' in result.stdout
	loaded = [load_normalization_stats(path) for path in stats_paths]
	assert [stats.survey_id for stats in loaded] == ['survey-a', 'survey-b']
	assert [stats.source_path for stats in loaded] == source_paths

	skip_result = run_python_proc(
		Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
	)

	assert skip_result.returncode == 0, skip_result.stderr
	assert 'normalization_stats.written_files: 0' in skip_result.stdout
	assert 'normalization_stats.skipped_existing_files: 2' in skip_result.stdout


def test_prepare_nopims_normalization_stats_overwrite_replaces_mismatched_stats(
	tmp_path: Path,
) -> None:
	_, config_path, stats_paths, source_paths = _write_prepare_inputs(tmp_path)
	write_normalization_stats(
		_stats('stale-survey', source_path=source_paths[1]),
		stats_paths[0],
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
		'--overwrite',
	)

	assert result.returncode == 0, result.stderr
	assert 'normalization_stats.written_files: 2' in result.stdout
	stats = load_normalization_stats(stats_paths[0])
	assert stats.survey_id == 'survey-a'
	assert stats.source_path == source_paths[0]


def test_prepare_nopims_normalization_stats_dry_run_reports_counts(
	tmp_path: Path,
) -> None:
	_, config_path, stats_paths, source_paths = _write_prepare_inputs(tmp_path)
	write_normalization_stats(
		_stats('survey-a', source_path=source_paths[0]),
		stats_paths[0],
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'normalization_stats.manifest_entries: 2' in result.stdout
	assert 'normalization_stats.existing_files: 1' in result.stdout
	assert 'normalization_stats.missing_files: 1' in result.stdout
	assert 'normalization_stats.max_samples: 1000' in result.stdout
	assert 'normalization_stats.seed: 42' in result.stdout
	assert 'normalization_stats.compute: skipped' in result.stdout


def test_prepare_nopims_normalization_stats_rejects_stats_outside_artifact_root(
	tmp_path: Path,
) -> None:
	manifest_path, config_path, _, source_paths = _write_prepare_inputs(tmp_path)
	write_manifest_json(
		[
			_manifest(
				'survey-a',
				source_paths[0],
				tmp_path / 'outside' / 'survey-a.normalization_stats.json',
			),
		],
		manifest_path,
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
	)

	assert result.returncode != 0
	assert 'paths.artifact_root' in result.stderr


def _write_prepare_inputs(
	tmp_path: Path,
) -> tuple[Path, Path, list[Path], list[Path]]:
	nopims_root = tmp_path / 'NOPIMS'
	artifact_root = tmp_path / 'artifacts'
	first_path = _write_volume(
		nopims_root / 'survey-a' / 'base.npy',
		np.arange(8, dtype=np.float32).reshape((2, 2, 2)),
	)
	second_path = _write_volume(
		nopims_root / 'survey-b' / 'base.npy',
		np.arange(8, 16, dtype=np.float32).reshape((2, 2, 2)),
	)
	stats_paths = [
		artifact_root / 'registry' / 'normalization_stats' / 'a.json',
		artifact_root / 'registry' / 'normalization_stats' / 'b.json',
	]
	manifest_path = artifact_root / 'registry' / 'manifests' / 'm.json'
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(
		[
			_manifest('survey-a', first_path, stats_paths[0]),
			_manifest('survey-b', second_path, stats_paths[1]),
		],
		manifest_path,
	)
	config = _base_config(nopims_root=nopims_root, artifact_root=artifact_root)
	config['manifests'] = {'train': str(manifest_path)}
	config['normalization'] = {
		'clipping_percentiles': [0.5, 99.5],
		'epsilon': 1.0e-6,
		'max_samples': 1000,
		'seed': 42,
		'smooth_time_depth_trend_correction': False,
		'trace_wise_agc': False,
		'patch_wise_zscore': False,
	}
	config_path = tmp_path / 'prepare.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
	return manifest_path, config_path, stats_paths, [first_path, second_path]


def _write_volume(path: Path, values: np.ndarray) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, values)
	return path


def _manifest(survey_id: str, path: Path, stats_path: Path) -> SurveyManifest:
	return SurveyManifest(
		survey_id=survey_id,
		root=path.parent,
		amplitude=AmplitudeVolumeRecord(
			survey_id=survey_id,
			path=path,
			shape_xyz=(2, 2, 2),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)


def _stats(
	survey_id: str,
	*,
	source_path: Path | None = None,
) -> SurveyNormalizationStats:
	return SurveyNormalizationStats(
		survey_id=survey_id,
		source_path=source_path or Path(f'{survey_id}.npy'),
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=0.5,
		clip_high_percentile=99.5,
		clip_low=-2.0,
		clip_high=6.0,
		median=2.0,
		iqr=2.0,
		eps=1.0e-6,
	)


def _base_config(
	*,
	nopims_root: str | Path = '/unused',
	artifact_root: str | Path = '/unused',
) -> dict[str, object]:
	return {
		'paths': {
			'nopims_root': str(nopims_root),
			'artifact_root': str(artifact_root),
		},
	}
