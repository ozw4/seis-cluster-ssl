from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeAgcConfig,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	apply_trace_rms_agc,
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


def test_normalize_amplitude_clips_normalized_values() -> None:
	stats = SurveyNormalizationStats(
		survey_id='survey',
		source_path=Path('survey.npy'),
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=0.0,
		clip_high_percentile=100.0,
		clip_low=-100.0,
		clip_high=100.0,
		median=0.0,
		iqr=1.0,
		eps=1.0e-6,
	)
	crop = np.array([-20.0, -4.0, 0.0, 4.0, 20.0], dtype=np.float32)

	result = normalize_amplitude(
		crop,
		stats,
		normalized_clip_abs=8.0,
	)

	np.testing.assert_allclose(
		result,
		[-8.0, -4.0, 0.0, 4.0, 8.0],
		atol=1.0e-4,
	)


def test_trace_rms_agc_uses_centered_z_window() -> None:
	amplitude = np.asarray([1.0, 1.0, 1.0, 10.0, 10.0, 10.0], dtype=np.float32)
	amplitude = amplitude.reshape(1, 1, 6)
	valid = np.ones_like(amplitude, dtype=bool)

	result = apply_trace_rms_agc(
		amplitude,
		valid,
		window_z=3,
		eps=1.0e-6,
		clip_abs=100.0,
	)

	expected_mean_power = np.asarray(
		[1.0, 1.0, 34.0, 67.0, 100.0, 100.0],
		dtype=np.float32,
	)
	expected = amplitude.reshape(-1) / np.sqrt(expected_mean_power + 1.0e-6)
	np.testing.assert_allclose(result.reshape(-1), expected, rtol=1.0e-6)
	assert result.dtype == np.float32


def test_trace_rms_agc_excludes_invalid_voxels_and_zeros_invalid_output() -> None:
	amplitude = np.asarray(
		[
			[[2.0, 1.0e20, 2.0]],
			[[1.0e20, 1.0e20, 1.0e20]],
		],
		dtype=np.float32,
	)
	valid = np.asarray(
		[
			[[True, False, True]],
			[[False, False, False]],
		],
		dtype=bool,
	)

	result = apply_trace_rms_agc(
		amplitude,
		valid,
		window_z=3,
		eps=1.0e-6,
		clip_abs=5.0,
	)

	np.testing.assert_allclose(result[0, 0, [0, 2]], [1.0, 1.0], atol=1.0e-6)
	assert result[0, 0, 1] == 0.0
	assert np.all(result[1, 0, :] == 0.0)
	assert np.isfinite(result).all()


def test_amplitude_agc_config_serializes_disabled_and_enabled() -> None:
	assert AmplitudeAgcConfig.from_mapping(None).to_dict() == {'enabled': False}
	assert AmplitudeAgcConfig.from_mapping({'enabled': False}).to_dict() == {
		'enabled': False,
	}

	enabled = AmplitudeAgcConfig.from_mapping(
		{
			'enabled': True,
			'mode': 'trace_rms_z',
			'window_z': 65,
			'eps': 1.0e-3,
			'clip_abs': 5.0,
		},
	)

	assert enabled.to_dict() == {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 65,
		'eps': 1.0e-3,
		'clip_abs': 5.0,
	}


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


def test_normalization_stats_exclude_exact_zero_voxels(
	tmp_path: Path,
) -> None:
	volume = np.array(
		[
			[[0.0, 1.0, 3.0], [0.0, 5.0, 7.0]],
			[[0.0, 9.0, 11.0], [0.0, 13.0, 15.0]],
		],
		dtype=np.float32,
	)
	source = tmp_path / 'volume.npy'
	np.save(source, volume)

	stats = compute_normalization_stats(
		source,
		survey_id='survey-a',
		clip_low_percentile=0.0,
		clip_high_percentile=100.0,
		max_samples=None,
		seed=42,
	)

	# Used values: [1, 3, 5, 7, 9, 11, 13, 15]
	assert stats.clip_low == pytest.approx(1.0)
	assert stats.clip_high == pytest.approx(15.0)
	assert stats.median == pytest.approx(8.0)
	assert stats.iqr == pytest.approx(7.0)


def test_normalization_stats_reject_all_zero_volume(
	tmp_path: Path,
) -> None:
	source = tmp_path / 'zero.npy'
	np.save(source, np.zeros((4, 4, 4), dtype=np.float32))

	with pytest.raises(ValueError, match='no finite non-zero voxels'):
		compute_normalization_stats(
			source,
			survey_id='zero-survey',
			max_samples=None,
			seed=42,
		)


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
