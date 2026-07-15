from __future__ import annotations

import json
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
	default_f3_prepare_outputs,
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


def test_prepare_f3_facies_volume_proc_writes_registry_artifacts(
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
	assert _is_relative_to(outputs.normalization_stats_path, artifact_root)

	assert outputs.split_path.read_text(encoding='utf-8') == f'{outputs.seismic_npy}\n'
	assert stats.survey_id == F3_SURVEY_ID
	assert stats.source_path == outputs.seismic_npy
	assert stats.grid_order == GRID_ORDER_XYZ
	assert stats.clip_low == pytest.approx(1.0)
	assert stats.clip_high == pytest.approx(23.0)
	assert '/runs/' not in result.stdout
	assert 'f3_prepare.shape_xyz: (2, 3, 4)' in result.stdout


def test_f3_prepare_and_embedding_configs_follow_path_contracts() -> None:
	prepare_raw = load_config(PREPARE_CONFIG)
	prepare_config = f3_prepare_volume_config_from_mapping(prepare_raw)
	embedding_raw = load_config(EMBEDDING_CONFIG)
	embedding_config = resolve_embedding_extraction_config(embedding_raw)
	predict_raw = load_config(PREDICT_CONFIG)

	assert prepare_config.outputs.volume_dir == Path(
		'/workspace/artifacts/seis_ssl_cluster/registry/volumes/f3/'
		'facies_benchmark_v1',
	)
	assert prepare_config.outputs.manifest_path == Path(
		'/workspace/artifacts/seis_ssl_cluster/registry/manifests/f3/'
		'facies_benchmark_v1/f3_amplitude_manifest.json',
	)
	assert prepare_config.outputs.normalization_stats_path == Path(
		'/workspace/artifacts/seis_ssl_cluster/registry/normalization_stats/f3/'
		'facies_benchmark_v1/f3_seismic.normalization_stats.json',
	)
	assert embedding_config['embeddings']['checkpoint'] == (
		'/workspace/artifacts/seis_ssl_cluster/pretraining/nopims/pretrain_v1/'
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/'
		'mae_latest.pt'
	)
	assert embedding_config['embeddings']['output_dir'] == (
		'/workspace/artifacts/seis_ssl_cluster/embeddings/f3/'
		'facies_benchmark_v1/'
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16'
	)
	assert '/runs/' not in json.dumps(prepare_raw)
	assert '/runs/' not in json.dumps(embedding_raw)
	assert predict_raw['probe']['probe_joblib'].endswith(
		'/probes/linear_balanced_v1/probe.joblib',
	)
	assert predict_raw['probe']['scaler_joblib'].endswith(
		'/probes/linear_balanced_v1/scaler.joblib',
	)
	assert 'probe.pt' not in json.dumps(predict_raw)
	assert '/runs/' not in json.dumps(predict_raw)


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
) -> tuple[Path, object]:
	outputs = default_f3_prepare_outputs(artifact_root)
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


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True
