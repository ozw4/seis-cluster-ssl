from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	NormalizationStatsQcThresholds,
	SurveyManifest,
	SurveyNormalizationStats,
	filter_manifests_by_stats_qc,
	make_survey_id_from_path,
	read_manifest_json,
	scan_nopims_amplitude_manifests_from_path_list,
	write_manifest_json,
	write_normalization_stats,
)
from tests.helpers import run_python_proc


def test_synthetic_normalization_qc_integration_writes_clean_outputs(
	tmp_path: Path,
) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	stable = _write_volume(
		nopims_root / 'stable' / 'base.npy',
		np.arange(64, dtype=np.float32).reshape((4, 4, 4)),
	)
	unstable = _write_volume(
		nopims_root / 'unstable' / 'base.npy',
		np.ones((4, 4, 4), dtype=np.float32),
	)
	source_split = tmp_path / 'artifacts' / 'registry' / 'splits' / 'nopims' / (
		'pretrain_v1'
	) / 'train_npy_paths.txt'
	source_split.parent.mkdir(parents=True, exist_ok=True)
	source_split.write_text(
		'stable/base.npy\nunstable/base.npy\n',
		encoding='utf-8',
	)
	stats_dir = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'normalization_stats'
		/ 'nopims'
		/ 'pretrain_v1'
	)
	manifest_path = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'manifests'
		/ 'nopims'
		/ 'pretrain_v1'
		/ 'nopims_amplitude_manifests.json'
	)
	result = scan_nopims_amplitude_manifests_from_path_list(
		nopims_root,
		source_split,
		stats_dir,
	)
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, manifest_path)
	original_manifest_text = manifest_path.read_text(encoding='utf-8')
	original_split_text = source_split.read_text(encoding='utf-8')

	prepare_config = _base_config('prepare_nopims_normalization_stats', nopims_root)
	prepare_config['manifests'] = {'train': str(manifest_path)}
	prepare_config['normalization'] = {
		'clipping_percentiles': [0.5, 99.5],
		'epsilon': 1.0e-6,
		'max_samples': 1000000,
		'seed': 42,
		'smooth_time_depth_trend_correction': False,
		'trace_wise_agc': False,
		'patch_wise_zscore': False,
	}
	prepare_config_path = tmp_path / 'prepare.yaml'
	prepare_config_path.write_text(yaml.safe_dump(prepare_config), encoding='utf-8')

	prepare = run_python_proc(
		Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'),
		'--config',
		prepare_config_path,
	)

	assert prepare.returncode == 0, prepare.stderr
	assert all(
		manifest.amplitude.normalization_stats_path.is_file()
		for manifest in result.manifests
	)
	assert stats_dir in result.manifests[0].amplitude.normalization_stats_path.parents

	clean_manifest = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'manifests'
		/ 'nopims'
		/ 'pretrain_v1_clean'
		/ 'nopims_amplitude_manifests.json'
	)
	clean_split = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'splits'
		/ 'nopims'
		/ 'pretrain_v1_clean'
		/ 'train_npy_paths.txt'
	)
	qc_json = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'qc'
		/ 'nopims'
		/ 'pretrain_v1'
		/ 'normalization_stats_qc.json'
	)
	excluded = qc_json.parent / 'excluded_surveys.txt'
	filter_config = _base_config('filter_manifest_by_normalization_qc', nopims_root)
	filter_config['manifests'] = {
		'input': str(manifest_path),
		'output': str(clean_manifest),
	}
	filter_config['splits'] = {
		'input': str(source_split),
		'output': str(clean_split),
	}
	filter_config['qc'] = {
		'output_json': str(qc_json),
		'excluded_surveys': str(excluded),
		'min_iqr': 1.0e-4,
		'max_normalized_abs': 1.0e6,
	}
	filter_config_path = tmp_path / 'filter.yaml'
	filter_config_path.write_text(yaml.safe_dump(filter_config), encoding='utf-8')

	filtered = run_python_proc(
		Path('proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py'),
		'--config',
		filter_config_path,
	)

	assert filtered.returncode == 0, filtered.stderr
	unstable_id = make_survey_id_from_path(unstable, nopims_root)
	stable_id = make_survey_id_from_path(stable, nopims_root)
	report = json.loads(qc_json.read_text(encoding='utf-8'))
	assert report['source_manifest_path'] == str(manifest_path)
	assert report['source_split_path'] == str(source_split)
	assert report['per_survey_reason_codes'][unstable_id] == ['small_iqr']
	assert excluded.read_text(encoding='utf-8') == f'{unstable_id}\n'
	assert manifest_path.read_text(encoding='utf-8') == original_manifest_text
	assert source_split.read_text(encoding='utf-8') == original_split_text
	assert clean_split.read_text(encoding='utf-8') == 'stable/base.npy\n'
	assert [manifest.survey_id for manifest in read_manifest_json(clean_manifest)] == [
		stable_id,
	]


def test_filter_manifests_by_stats_qc_preserves_clean_split_order_and_duplicates(
	tmp_path: Path,
) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	first = _write_volume(
		nopims_root / 'first' / 'base.npy',
		np.arange(8, dtype=np.float32).reshape((2, 2, 2)),
	)
	second = _write_volume(
		nopims_root / 'second' / 'base.npy',
		np.arange(8, dtype=np.float32).reshape((2, 2, 2)),
	)
	stats_dir = tmp_path / 'stats'
	first_id = make_survey_id_from_path(first, nopims_root)
	second_id = make_survey_id_from_path(second, nopims_root)
	first_manifest = _manifest(nopims_root, first, first_id, stats_dir)
	second_manifest = _manifest(nopims_root, second, second_id, stats_dir)
	_write_stats(first_manifest)
	_write_stats(second_manifest)

	result = filter_manifests_by_stats_qc(
		(second_manifest, first_manifest),
		('first/base.npy', 'first/base.npy', 'second/base.npy'),
		nopims_root=nopims_root,
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert result.clean_path_entries == (
		'first/base.npy',
		'first/base.npy',
		'second/base.npy',
	)
	assert [manifest.survey_id for manifest in result.clean_manifests] == [
		first_id,
		first_id,
		second_id,
	]


def test_filter_manifest_qc_dry_run_prints_config_summary_with_existing_inputs(
	tmp_path: Path,
) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	volume = _write_volume(
		nopims_root / 'survey' / 'base.npy',
		np.arange(8, dtype=np.float32).reshape((2, 2, 2)),
	)
	source_split = tmp_path / 'source_split.txt'
	source_split.write_text('survey/base.npy\n', encoding='utf-8')
	survey_id = make_survey_id_from_path(volume, nopims_root)
	manifest = _manifest(
		nopims_root,
		volume,
		survey_id,
		tmp_path / 'artifacts' / 'registry' / 'normalization_stats',
	)
	_write_stats(manifest)
	manifest_path = tmp_path / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	artifact_root = tmp_path / 'artifacts'
	clean_manifest = (
		artifact_root
		/ 'registry'
		/ 'manifests'
		/ 'nopims'
		/ 'pretrain_v1_clean'
		/ 'nopims_amplitude_manifests.json'
	)
	clean_split = (
		artifact_root
		/ 'registry'
		/ 'splits'
		/ 'nopims'
		/ 'pretrain_v1_clean'
		/ 'train_npy_paths.txt'
	)
	qc_json = (
		artifact_root
		/ 'registry'
		/ 'qc'
		/ 'nopims'
		/ 'pretrain_v1'
		/ 'normalization_stats_qc.json'
	)
	config = _base_config('filter_manifest_by_normalization_qc', nopims_root)
	config['manifests'] = {
		'input': str(manifest_path),
		'output': str(clean_manifest),
	}
	config['splits'] = {
		'input': str(source_split),
		'output': str(clean_split),
	}
	config['qc'] = {
		'output_json': str(qc_json),
		'excluded_surveys': str(qc_json.parent / 'excluded_surveys.txt'),
	}
	config_path = tmp_path / 'filter.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'stage: filter_manifest_by_normalization_qc' in result.stdout
	assert 'data.input_channels: 1' in result.stdout
	assert 'normalization_qc.write: false' in result.stdout
	assert not qc_json.exists()
	assert not clean_manifest.exists()
	assert not clean_split.exists()


def _write_volume(path: Path, values: np.ndarray) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, values)
	return path


def _manifest(
	nopims_root: Path,
	volume_path: Path,
	survey_id: str,
	stats_dir: Path,
) -> SurveyManifest:
	return SurveyManifest(
		survey_id=survey_id,
		root=nopims_root,
		amplitude=AmplitudeVolumeRecord(
			survey_id=survey_id,
			path=volume_path,
			shape_xyz=(2, 2, 2),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=(
				stats_dir / f'{survey_id}.normalization_stats.json'
			),
		),
	)


def _write_stats(manifest: SurveyManifest) -> None:
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id=manifest.survey_id,
			source_path=manifest.amplitude.path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.5,
			clip_high_percentile=99.5,
			clip_low=-2.0,
			clip_high=6.0,
			median=2.0,
			iqr=2.0,
		),
		manifest.amplitude.normalization_stats_path,
	)


def _base_config(stage: str, nopims_root: Path) -> dict[str, object]:
	return {
		'stage': stage,
		'paths': {
			'nopims_root': str(nopims_root),
			'artifact_root': str(nopims_root.parent / 'artifacts'),
		},
		'data': {
			'grid_order': ['x', 'y', 'z'],
			'volume_format': 'npy_memmap',
			'input_channels': 1,
			'target_channels': 1,
			'use_context': False,
			'local_crop_size': [128, 128, 128],
		},
		'model': {
			'name': 'amp_mae3d',
			'in_channels': 1,
			'out_channels': 1,
			'patch_size': [8, 8, 8],
		},
		'masking': {
			'spatial_mask_ratio': 0.75,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [2, 2, 2],
		},
		'train': {
			'batch_size': 4,
			'samples_per_epoch': 10000,
			'epochs': 100,
			'num_workers': 8,
			'amp': False,
		},
	}
