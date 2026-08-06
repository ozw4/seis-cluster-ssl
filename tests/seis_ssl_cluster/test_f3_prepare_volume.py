from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.config import load_config, resolve_embedding_extraction_config
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	load_normalization_stats,
	read_manifest_json,
)
from seis_ssl_cluster.f3 import (
	F3_SURVEY_ID,
	F3PrepareOutputPaths,
	f3_prepare_volume_config_from_mapping,
	prepare_f3_facies_volume,
)
from tests.helpers import run_python_proc

PREPARE_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/10_prepare/01_prepare_f3_volume.yaml',
)
EMBEDDING_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/20_embedding/'
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16.yaml',
)
PREDICT_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/50_lithology/'
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16/'
	'png_slices_segy_labels_v1/04_predict_volume.yaml',
)


def test_prepare_f3_facies_volume_proc_writes_configured_outputs(
	tmp_path: Path,
) -> None:
	segyio = pytest.importorskip('segyio')
	f3_root = _make_f3_segy_fixture(tmp_path, segyio=segyio)
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	config_path, outputs = _write_prepare_config(
		tmp_path,
		f3_root=f3_root,
		artifact_root=artifact_root,
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/prepare_f3_facies_volume.py'),
		'--config',
		config_path,
	)

	assert result.returncode == 0, result.stderr
	seismic = np.load(outputs.seismic_npy)
	labels = np.load(outputs.label_npy)
	metadata = json.loads(outputs.metadata_path.read_text(encoding='utf-8'))
	manifests = read_manifest_json(outputs.manifest_path)
	stats = load_normalization_stats(outputs.normalization_stats_path)

	assert seismic.dtype == np.float32
	assert seismic.shape == (2, 3, 4)
	assert labels.dtype == np.int16
	assert labels.shape == (2, 3, 4)
	assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
	assert labels[0, 0, 0] == 0

	assert metadata['dataset']['survey_id'] == F3_SURVEY_ID
	assert metadata['volumes']['seismic']['shape_xyz'] == [2, 3, 4]
	assert metadata['volumes']['seismic']['grid_order'] == list(GRID_ORDER_XYZ)
	assert metadata['volumes']['label']['label_zero_is_valid_class'] is True
	assert metadata['volumes']['label']['counts_by_value']['0'] == 8
	assert metadata['facies_classes'][0]['class_id'] == 0

	assert len(manifests) == 1
	manifest = manifests[0]
	assert manifest.survey_id == F3_SURVEY_ID
	assert manifest.root == outputs.volume_dir
	assert manifest.amplitude.path == outputs.seismic_npy
	assert manifest.amplitude.shape_xyz == (2, 3, 4)
	assert manifest.amplitude.grid_order == GRID_ORDER_XYZ
	assert manifest.amplitude.normalization_stats_path == (
		outputs.normalization_stats_path
	)
	assert outputs.normalization_stats_path == (
		tmp_path
		/ 'configured-outputs'
		/ 'stats'
		/ 'custom-normalization.json'
	)

	assert outputs.split_path.read_text(encoding='utf-8') == f'{outputs.seismic_npy}\n'
	assert stats.survey_id == F3_SURVEY_ID
	assert stats.source_path == outputs.seismic_npy
	assert stats.grid_order == GRID_ORDER_XYZ
	assert stats.clip_low == pytest.approx(1.0)
	assert stats.clip_high == pytest.approx(23.0)
	assert 'f3_prepare.shape_xyz: (2, 3, 4)' in result.stdout


def test_f3_prepare_and_embedding_configs_preserve_explicit_paths() -> None:
	prepare_raw = load_config(PREPARE_CONFIG)
	prepare_config = f3_prepare_volume_config_from_mapping(prepare_raw)
	embedding_raw = load_config(EMBEDDING_CONFIG)
	embedding_config = resolve_embedding_extraction_config(embedding_raw)
	predict_raw = load_config(PREDICT_CONFIG)

	assert prepare_config.outputs.volume_dir == Path(
		prepare_raw['outputs']['volume_dir']
	)
	assert prepare_config.outputs.manifest_path == Path(
		prepare_raw['outputs']['manifest_path']
	)
	assert prepare_config.outputs.normalization_stats_path == Path(
		prepare_raw['outputs']['normalization_stats_path']
	)
	assert embedding_config['embeddings']['checkpoint'] == (
		embedding_raw['embeddings']['checkpoint']
	)
	assert embedding_config['embeddings']['output_dir'] == (
		embedding_raw['embeddings']['output_dir']
	)
	assert predict_raw['probe']['probe_joblib'].endswith(
		'/probes/linear_balanced_v1/probe.joblib',
	)
	assert predict_raw['probe']['scaler_joblib'].endswith(
		'/probes/linear_balanced_v1/scaler.joblib',
	)
	assert 'probe.pt' not in json.dumps(predict_raw)


@pytest.mark.parametrize(
	('output_label', 'collision_label'),
	[
		('outputs.seismic_npy', 'inputs.seismic_segy'),
		('outputs.metadata_path', 'inputs.inspection_report'),
		('outputs.manifest_path', 'outputs.split_path'),
		('outputs.normalization_stats_path', 'outputs.metadata_path'),
	],
)
def test_f3_prepare_config_rejects_file_path_collisions(
	tmp_path: Path,
	output_label: str,
	collision_label: str,
) -> None:
	f3_root = tmp_path / 'F3'
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	config_path, _outputs = _write_prepare_config(
		tmp_path,
		f3_root=f3_root,
		artifact_root=artifact_root,
	)
	raw = load_config(config_path)
	if output_label == 'outputs.seismic_npy':
		raw['outputs']['volume_dir'] = str(f3_root)
		raw['inputs']['seismic_segy'] = str(f3_root / 'f3_seismic.npy')
	else:
		output_section, output_key = output_label.split('.')
		collision_section, collision_key = collision_label.split('.')
		raw[output_section][output_key] = raw[collision_section][collision_key]

	with pytest.raises(
		ValueError,
		match=rf'({output_label}|{collision_label}).*differ',
	):
		f3_prepare_volume_config_from_mapping(raw)


def test_prepare_f3_volume_rechecks_collisions_when_overwrite_is_enabled(
	tmp_path: Path,
) -> None:
	f3_root = tmp_path / 'F3'
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	config_path, _outputs = _write_prepare_config(
		tmp_path,
		f3_root=f3_root,
		artifact_root=artifact_root,
	)
	config = f3_prepare_volume_config_from_mapping(load_config(config_path))
	config = replace(
		config,
		outputs=replace(
			config.outputs,
			metadata_path=config.inputs.inspection_report,
		),
	)

	with pytest.raises(ValueError, match=r'outputs\.metadata_path.*differ'):
		prepare_f3_facies_volume(config, overwrite=True)


def test_prepare_f3_facies_volume_missing_segy_has_clear_error(
	tmp_path: Path,
) -> None:
	f3_root = tmp_path / 'F3'
	(f3_root / 'interpretation').mkdir(parents=True)
	(f3_root / 'interpretation' / 'class_info.json').write_text(
		json.dumps({'0': {'name': 'Class zero', 'color': [0, 0, 0]}}),
		encoding='utf-8',
	)
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	config_path, _outputs = _write_prepare_config(
		tmp_path,
		f3_root=f3_root,
		artifact_root=artifact_root,
	)
	config = f3_prepare_volume_config_from_mapping(load_config(config_path))

	with pytest.raises(FileNotFoundError, match='F3 seismic SEGY file'):
		prepare_f3_facies_volume(config)


def _make_f3_segy_fixture(tmp_path: Path, *, segyio: Any) -> Path:
	f3_root = tmp_path / 'F3'
	(f3_root / 'interpretation').mkdir(parents=True)
	seismic = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
	labels = np.array(
		[
			[[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]],
			[[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]],
		],
		dtype=np.float32,
	)
	segyio.tools.from_array3D(str(f3_root / 'f3_seismic.sgy'), seismic)
	segyio.tools.from_array3D(str(f3_root / 'f3_labels.sgy'), labels)
	(f3_root / 'interpretation' / 'class_info.json').write_text(
		json.dumps(
			{
				'0': {'name': 'Class zero', 'color': [0, 0, 0]},
				'1': {'name': 'Class one', 'color': [35, 92, 167]},
				'2': {'name': 'Class two', 'color': [102, 194, 165]},
			},
		),
		encoding='utf-8',
	)
	return f3_root


def _write_prepare_config(
	tmp_path: Path,
	*,
	f3_root: Path,
	artifact_root: Path,
) -> tuple[Path, F3PrepareOutputPaths]:
	output_root = tmp_path / 'configured-outputs'
	outputs = F3PrepareOutputPaths(
		volume_dir=output_root / 'volume',
		manifest_path=output_root / 'manifest' / 'custom-manifest.json',
		split_path=output_root / 'split' / 'custom-inputs.txt',
		normalization_stats_path=(
			output_root / 'stats' / 'custom-normalization.json'
		),
		metadata_path=output_root / 'metadata' / 'custom-metadata.json',
	)
	inspection_report = (
		artifact_root / 'inspection' / 'f3' / 'facies_benchmark_v1' / 'report.json'
	)
	inspection_report.parent.mkdir(parents=True, exist_ok=True)
	inspection_report.write_text(
		json.dumps({'downstream_readiness': {'status': 'caution'}}),
		encoding='utf-8',
	)
	config = {
		'paths': {
			'f3_root': str(f3_root),
			'artifact_root': str(artifact_root),
		},
		'inputs': {
			'seismic_segy': str(f3_root / 'f3_seismic.sgy'),
			'label_segy': str(f3_root / 'f3_labels.sgy'),
			'class_info': str(f3_root / 'interpretation' / 'class_info.json'),
			'inspection_report': str(inspection_report),
		},
		'outputs': {
			'volume_dir': str(outputs.volume_dir),
			'manifest_path': str(outputs.manifest_path),
			'split_path': str(outputs.split_path),
			'normalization_stats_path': str(outputs.normalization_stats_path),
			'metadata_path': str(outputs.metadata_path),
		},
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
			'survey_id': F3_SURVEY_ID,
		},
		'normalization': {
			'clipping_percentiles': [0.0, 100.0],
			'epsilon': 1.0e-6,
			'max_samples': None,
			'seed': 42,
		},
	}
	config_path = tmp_path / 'prepare_f3_volume.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
	return config_path, outputs
