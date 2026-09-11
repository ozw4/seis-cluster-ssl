from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.data import GRID_ORDER_XYZ, SurveyNormalizationStats
from seis_ssl_cluster.data.normalization import write_normalization_stats
from seis_ssl_cluster.f3 import (
	F3_SURVEY_ID,
	F3PrepareDatasetConfig,
	F3PrepareInputPaths,
	F3PrepareNormalizationConfig,
	F3PrepareOutputPaths,
	F3PrepareRootPaths,
	F3PrepareVolumeConfig,
	check_f3_prepared_volume_parity,
	inspect_f3_prepared_volume_identity,
)
from tests.helpers import run_python_proc

PROC = Path('proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py')
CLASSES = [
	{'class_id': 0, 'class_name': 'zero', 'hex_color': '#000000', 'rgb': [0, 0, 0]},
	{'class_id': 1, 'class_name': 'one', 'hex_color': '#010101', 'rgb': [1, 1, 1]},
]


def test_identical_preparations_pass(tmp_path: Path) -> None:
	reference = _prepared_version(tmp_path, 'facies_benchmark_v1')
	candidate = _prepared_version(tmp_path, 'facies_benchmark_v2')

	parity = check_f3_prepared_volume_parity(reference, candidate)

	assert parity.passed
	assert parity.mismatches == ()
	assert parity.reference.dataset_version == 'facies_benchmark_v1'
	assert parity.candidate.dataset_version == 'facies_benchmark_v2'
	assert parity.reference.seismic_sha256 == parity.candidate.seismic_sha256
	assert parity.reference.seismic_npy != parity.candidate.seismic_npy
	assert parity.reference.normalization['max_samples'] == 1_000_000
	assert parity.reference.normalization['seed'] == 42
	assert [entry['class_id'] for entry in parity.reference.class_order] == [0, 1]


def test_label_bytes_and_class_order_differences_are_reported(tmp_path: Path) -> None:
	reference = _prepared_version(tmp_path, 'facies_benchmark_v1')
	candidate = _prepared_version(tmp_path, 'facies_benchmark_v2')
	labels = np.load(candidate.outputs.label_npy)
	labels[0, 0, 0] = 1
	np.save(candidate.outputs.label_npy, labels)
	metadata = json.loads(candidate.outputs.metadata_path.read_text(encoding='utf-8'))
	metadata['facies_classes'] = list(reversed(metadata['facies_classes']))
	candidate.outputs.metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	parity = check_f3_prepared_volume_parity(reference, candidate)

	assert not parity.passed
	assert [item.split(':')[0] for item in parity.mismatches] == [
		'label_sha256',
		'class_order',
	]


def test_normalization_semantics_differences_are_reported(tmp_path: Path) -> None:
	reference = _prepared_version(tmp_path, 'facies_benchmark_v1')
	candidate = _prepared_version(tmp_path, 'facies_benchmark_v2', seed=7)
	stats = _stats(candidate)
	write_normalization_stats(
		replace(stats, median=stats.median + 0.5),
		candidate.outputs.normalization_stats_path,
	)

	parity = check_f3_prepared_volume_parity(reference, candidate)

	assert [item.split(':')[0] for item in parity.mismatches] == [
		'normalization.seed',
		'normalization.median',
	]


def test_stats_disagreeing_with_prepare_config_fail_closed(tmp_path: Path) -> None:
	config = _prepared_version(tmp_path, 'facies_benchmark_v2')
	write_normalization_stats(
		replace(_stats(config), eps=1.0e-3),
		config.outputs.normalization_stats_path,
	)

	with pytest.raises(ValueError, match='normalization stats eps'):
		inspect_f3_prepared_volume_identity(config)


def test_metadata_disagreeing_with_npy_fails_closed(tmp_path: Path) -> None:
	config = _prepared_version(tmp_path, 'facies_benchmark_v2')
	metadata = json.loads(config.outputs.metadata_path.read_text(encoding='utf-8'))
	metadata['volumes']['seismic']['dtype'] = 'float64'
	config.outputs.metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match=r'volumes\.seismic dtype'):
		inspect_f3_prepared_volume_identity(config)


def test_proc_reports_pass_and_fails_on_mismatch(tmp_path: Path) -> None:
	reference = _prepared_version(tmp_path, 'facies_benchmark_v1')
	candidate = _prepared_version(tmp_path, 'facies_benchmark_v2')
	reference_yaml = _write_config_yaml(tmp_path, reference)
	candidate_yaml = _write_config_yaml(tmp_path, candidate)

	dry = run_python_proc(
		PROC,
		'--reference-config',
		reference_yaml,
		'--candidate-config',
		candidate_yaml,
		'--dry-run',
	)
	assert dry.returncode == 0, dry.stderr
	assert 'f3_prepared_parity.execution: dry-run' in dry.stdout
	assert 'sha256' not in dry.stdout

	passed = run_python_proc(
		PROC,
		'--reference-config',
		reference_yaml,
		'--candidate-config',
		candidate_yaml,
	)
	assert passed.returncode == 0, passed.stderr
	assert 'f3_prepared_parity.status: PASS' in passed.stdout
	assert 'candidate.class_order: 0=zero, 1=one' in passed.stdout

	seismic = np.load(candidate.outputs.seismic_npy)
	seismic[0, 0, 0] += 1.0
	np.save(candidate.outputs.seismic_npy, seismic)
	failed = run_python_proc(
		PROC,
		'--reference-config',
		reference_yaml,
		'--candidate-config',
		candidate_yaml,
	)
	assert failed.returncode != 0
	assert 'f3_prepared_parity.mismatch: seismic_sha256' in failed.stdout
	assert 'f3_prepared_parity.status: FAIL' in failed.stdout
	assert 'do not reuse the reference checkpoints' in failed.stderr


def _prepared_version(
	tmp_path: Path,
	version: str,
	*,
	seed: int = 42,
) -> F3PrepareVolumeConfig:
	artifact_root = tmp_path / 'artifacts'
	f3_root = tmp_path / 'f3'
	f3_root.mkdir(exist_ok=True)
	volume_dir = artifact_root / 'registry' / 'volumes' / 'f3' / version
	volume_dir.mkdir(parents=True)
	outputs = F3PrepareOutputPaths(
		volume_dir=volume_dir,
		manifest_path=artifact_root / 'registry' / 'manifests' / version / 'm.json',
		split_path=artifact_root / 'registry' / 'splits' / version / 'paths.txt',
		normalization_stats_path=(
			artifact_root / 'registry' / 'normalization_stats' / version / 'stats.json'
		),
		metadata_path=volume_dir / 'f3_metadata.json',
	)
	seismic = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
	labels = np.zeros((2, 3, 4), dtype=np.int16)
	labels[1] = 1
	np.save(outputs.seismic_npy, seismic)
	np.save(outputs.label_npy, labels)
	outputs.metadata_path.write_text(
		json.dumps(
			{
				'dataset': {'version': version},
				'grid_order': list(GRID_ORDER_XYZ),
				'facies_classes': CLASSES,
				'volumes': {
					'seismic': {
						'shape_xyz': [2, 3, 4],
						'dtype': 'float32',
						'grid_order': list(GRID_ORDER_XYZ),
					},
					'label': {
						'shape_xyz': [2, 3, 4],
						'dtype': 'int16',
						'grid_order': list(GRID_ORDER_XYZ),
					},
				},
			}
		),
		encoding='utf-8',
	)
	config = F3PrepareVolumeConfig(
		paths=F3PrepareRootPaths(f3_root=f3_root, artifact_root=artifact_root),
		inputs=F3PrepareInputPaths(
			seismic_segy=f3_root / 'f3_seismic.sgy',
			label_segy=f3_root / 'f3_labels.sgy',
			class_info=f3_root / 'interpretation' / 'class_info.json',
			inspection_report=artifact_root / 'inspection' / 'report.json',
		),
		outputs=outputs,
		dataset=F3PrepareDatasetConfig(
			name='f3_facies_benchmark',
			version=version,
			survey_id=F3_SURVEY_ID,
		),
		normalization=F3PrepareNormalizationConfig(
			clip_low_percentile=0.5,
			clip_high_percentile=99.5,
			eps=1.0e-6,
			max_samples=1_000_000,
			seed=seed,
		),
	)
	write_normalization_stats(_stats(config), outputs.normalization_stats_path)
	return config


def _stats(config: F3PrepareVolumeConfig) -> SurveyNormalizationStats:
	return SurveyNormalizationStats(
		survey_id=F3_SURVEY_ID,
		source_path=config.outputs.seismic_npy,
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=config.normalization.clip_low_percentile,
		clip_high_percentile=config.normalization.clip_high_percentile,
		clip_low=0.1,
		clip_high=22.9,
		median=11.5,
		iqr=12.0,
		eps=config.normalization.eps,
	)


def _write_config_yaml(tmp_path: Path, config: F3PrepareVolumeConfig) -> Path:
	payload = {
		'paths': {
			'f3_root': str(config.paths.f3_root),
			'artifact_root': str(config.paths.artifact_root),
		},
		'inputs': {
			'seismic_segy': str(config.inputs.seismic_segy),
			'label_segy': str(config.inputs.label_segy),
			'class_info': str(config.inputs.class_info),
			'inspection_report': str(config.inputs.inspection_report),
		},
		'outputs': {
			'volume_dir': str(config.outputs.volume_dir),
			'manifest_path': str(config.outputs.manifest_path),
			'split_path': str(config.outputs.split_path),
			'normalization_stats_path': str(config.outputs.normalization_stats_path),
			'metadata_path': str(config.outputs.metadata_path),
		},
		'dataset': {
			'name': config.dataset.name,
			'version': config.dataset.version,
			'survey_id': config.dataset.survey_id,
		},
		'normalization': {
			'clipping_percentiles': [
				config.normalization.clip_low_percentile,
				config.normalization.clip_high_percentile,
			],
			'epsilon': config.normalization.eps,
			'max_samples': config.normalization.max_samples,
			'seed': config.normalization.seed,
		},
	}
	path = tmp_path / f'{config.dataset.version}.yaml'
	path.write_text(yaml.safe_dump(payload), encoding='utf-8')
	return path
