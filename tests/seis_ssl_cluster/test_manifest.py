from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	read_manifest_json,
	scan_nopims_amplitude_manifests_from_path_list,
	survey_manifest_from_dict,
	survey_manifest_to_dict,
	write_manifest_json,
)
from tests.helpers import run_python_proc


def _write_volume(path: Path, shape: tuple[int, int, int] = (4, 5, 6)) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, np.zeros(shape, dtype=np.float32))
	return path


def _write_path_list(path: Path, entries: list[str]) -> Path:
	path.write_text('\n'.join(entries), encoding='utf-8')
	return path


def test_manifest_json_round_trip_preserves_amplitude_record(
	tmp_path: Path,
) -> None:
	stats_path = tmp_path / 'stats' / 'survey.normalization_stats.json'
	manifest = SurveyManifest(
		survey_id='survey',
		root=Path('nopims/survey'),
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=Path('nopims/survey/base.npy'),
			shape_xyz=(8, 9, 10),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	path = tmp_path / 'manifests.json'

	write_manifest_json([manifest], path)

	loaded = read_manifest_json(path)
	assert loaded == [manifest]
	assert survey_manifest_from_dict(survey_manifest_to_dict(manifest)) == manifest
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	assert 'attribute_volumes' not in payload[0]
	assert 'base_seismic_kind' not in str(payload[0])


def test_manifest_rejects_relative_normalization_stats_path() -> None:
	payload = {
		'survey_id': 'survey',
		'root': '/source/NOPIMS/survey',
		'amplitude': {
			'survey_id': 'survey',
			'path': '/source/NOPIMS/survey/base.npy',
			'shape_xyz': [8, 9, 10],
			'dtype': 'float32',
			'grid_order': ['x', 'y', 'z'],
			'normalization_stats_path': 'stats/survey.normalization_stats.json',
		},
	}

	with pytest.raises(ValueError, match=r'normalization_stats_path.*absolute'):
		survey_manifest_from_dict(payload)


def test_scan_path_list_preserves_order_and_places_stats_outside_source(
	tmp_path: Path,
) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	first = _write_volume(nopims_root / 'survey_b' / 'seismic' / 'base.npy')
	second = _write_volume(nopims_root / 'survey_a' / 'seismic' / 'base.npy')
	path_list = _write_path_list(
		tmp_path / 'paths.txt',
		[str(first), 'survey_a/seismic/base.npy'],
	)
	stats_dir = tmp_path / 'artifacts' / 'normalization_stats'

	result = scan_nopims_amplitude_manifests_from_path_list(
		nopims_root,
		path_list,
		stats_dir,
	)

	assert [manifest.amplitude.path for manifest in result.manifests] == [
		first,
		second,
	]
	for manifest in result.manifests:
		assert manifest.amplitude.normalization_stats_path.parent == stats_dir
		assert manifest.amplitude.normalization_stats_path.name == (
			f'{manifest.survey_id}.normalization_stats.json'
		)
		assert nopims_root not in manifest.amplitude.normalization_stats_path.parents


def test_scan_path_list_rejects_relative_normalization_stats_dir(
	tmp_path: Path,
) -> None:
	with pytest.raises(ValueError, match=r'normalization_stats_dir.*absolute'):
		scan_nopims_amplitude_manifests_from_path_list(
			tmp_path / 'NOPIMS',
			tmp_path / 'missing_paths.txt',
			Path('relative_stats'),
		)


def test_scan_path_list_rejects_normalization_stats_dir_under_source(
	tmp_path: Path,
) -> None:
	with pytest.raises(ValueError, match=r'normalization_stats_dir.*nopims_root'):
		scan_nopims_amplitude_manifests_from_path_list(
			tmp_path / 'NOPIMS',
			tmp_path / 'missing_paths.txt',
			tmp_path / 'NOPIMS' / 'normalization_stats',
		)


def test_scan_path_list_rejects_non_numeric_and_non_3d_arrays(
	tmp_path: Path,
) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	non_3d = nopims_root / 'survey_a' / 'base.npy'
	non_3d.parent.mkdir(parents=True, exist_ok=True)
	np.save(non_3d, np.zeros((2, 3), dtype=np.float32))
	path_list = _write_path_list(tmp_path / 'paths.txt', [str(non_3d)])

	with pytest.raises(ValueError, match='3D'):
		scan_nopims_amplitude_manifests_from_path_list(
			nopims_root,
			path_list,
			tmp_path / 'stats',
		)

	object_path = nopims_root / 'survey_b' / 'base.npy'
	object_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(object_path, np.empty((2, 2, 2), dtype=object))
	path_list.write_text(str(object_path), encoding='utf-8')
	with pytest.raises(TypeError, match='object dtype'):
		scan_nopims_amplitude_manifests_from_path_list(
			nopims_root,
			path_list,
			tmp_path / 'stats',
		)


def test_build_nopims_manifests_cli_writes_amplitude_manifest(
	tmp_path: Path,
) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	volume = _write_volume(nopims_root / 'survey_a' / 'base.npy')
	path_list = _write_path_list(tmp_path / 'paths.txt', [str(volume)])
	artifact_root = tmp_path / 'artifacts'
	output_dir = artifact_root / 'registry' / 'manifests'
	stats_dir = artifact_root / 'registry' / 'normalization_stats'
	config = {
		'paths': {
			'nopims_root': str(nopims_root),
			'artifact_root': str(artifact_root),
		},
		'manifest': {
			'input_path_list': str(path_list),
			'output_dir': str(output_dir),
			'normalization_stats_dir': str(stats_dir),
		},
	}
	config_path = tmp_path / 'build_nopims_manifests.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/build_nopims_manifests.py'),
		'--config',
		config_path,
	)

	assert result.returncode == 0, result.stderr
	output_path = output_dir / 'nopims_amplitude_manifests.json'
	assert output_path.is_file()
	loaded = read_manifest_json(output_path)
	assert loaded[0].amplitude.path == volume
	assert loaded[0].amplitude.normalization_stats_path.parent == stats_dir
	assert 'manifest.amplitude_volume_count: 1' in result.stdout
