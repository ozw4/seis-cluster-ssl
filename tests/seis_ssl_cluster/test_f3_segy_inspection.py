from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.f3 import (
	F3ClassInfo,
	axis_assumption_metadata,
	calculate_label_unique_values,
	calculate_seismic_amplitude_stats,
	find_f3_segy_paths,
)
from tests.helpers import run_python_proc


def test_seismic_amplitude_stats_use_finite_values() -> None:
	values = np.array([-1.0, 0.0, 1.0, 2.0, np.nan, np.inf])

	stats = calculate_seismic_amplitude_stats(values)

	assert stats['finite_count'] == 4
	assert stats['nonfinite_count'] == 2
	assert stats['zero_count'] == 1
	assert stats['min'] == -1.0
	assert stats['p50'] == pytest.approx(0.5)
	assert stats['max'] == 2.0
	assert stats['mean'] == pytest.approx(0.5)
	assert stats['std'] == pytest.approx(np.std([-1.0, 0.0, 1.0, 2.0]))


def test_label_unique_values_compare_with_class_info() -> None:
	classes = (
		F3ClassInfo(class_id=0, class_name='Background', rgb=(0, 0, 0)),
		F3ClassInfo(class_id=1, class_name='Class one', rgb=(35, 92, 167)),
	)
	labels = np.array([0.0, 1.0, 1.0, 2.0, 0.0])

	stats = calculate_label_unique_values(labels, classes)

	assert stats['integer_like'] is True
	assert stats['unique_values'] == [0, 1, 2]
	assert stats['counts_by_value'] == {'0': 2, '1': 2, '2': 1}
	assert stats['min'] == 0
	assert stats['max'] == 2
	assert stats['unexpected_label_values'] == [2]
	class_rows = stats['class_info']['classes']
	assert class_rows[0]['class_id'] == 0
	assert class_rows[0]['present_in_label'] is True
	assert class_rows[0]['count'] == 2


def test_axis_assumption_metadata_maps_cube_to_xyz() -> None:
	metadata = axis_assumption_metadata()
	axis_map = metadata['cube_to_repo_axes']

	assert axis_map['cube_axis_0']['repo_axis'] == 'x'
	assert axis_map['cube_axis_0']['domain_axis'] == 'inline'
	assert axis_map['cube_axis_1']['repo_axis'] == 'y'
	assert axis_map['cube_axis_1']['domain_axis'] == 'crossline'
	assert axis_map['cube_axis_2']['repo_axis'] == 'z'
	assert axis_map['cube_axis_2']['domain_axis'] == 'sample/time'


def test_missing_f3_segy_file_has_clear_error(tmp_path: Path) -> None:
	f3_root = tmp_path / 'F3'
	f3_root.mkdir()

	with pytest.raises(FileNotFoundError, match='seismic SEGY'):
		find_f3_segy_paths(f3_root)


def test_inspect_f3_segy_geometry_proc_writes_required_outputs(
	tmp_path: Path,
) -> None:
	segyio = pytest.importorskip('segyio')
	f3_root = _make_f3_segy_fixture(tmp_path, segyio=segyio)
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	inspection_dir = artifact_root / 'inspection' / 'f3' / 'facies_benchmark_v1'
	segy_dir = inspection_dir / 'segy'
	config = {
		'paths': {
			'f3_root': str(f3_root),
			'artifact_root': str(artifact_root),
		},
		'outputs': {'inspection_dir': str(inspection_dir)},
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v1',
		},
		'inspection': {
			'inventory_json': str(inspection_dir / 'inventory' / 'file_inventory.json'),
			'segy_dir': str(segy_dir),
			'output_json': str(segy_dir / 'segy_geometry.json'),
			'output_csv': str(segy_dir / 'segy_geometry.csv'),
			'metadata_json': str(segy_dir / 'segy_metadata.json'),
			'summary_markdown': str(segy_dir / 'segy_summary.md'),
			'seismic_amplitude_stats_json': str(
				segy_dir / 'seismic_amplitude_stats.json',
			),
			'label_unique_values_json': str(
				segy_dir / 'label_unique_values.json',
			),
			'candidate_extensions': ['.segy', '.sgy'],
		},
	}
	config_path = tmp_path / 'inspect_f3_segy_geometry.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/inspect_f3_segy_geometry.py'),
		'--config',
		config_path,
	)

	assert result.returncode == 0, result.stderr
	metadata = json.loads((segy_dir / 'segy_metadata.json').read_text())
	legacy_geometry = json.loads((segy_dir / 'segy_geometry.json').read_text())
	amplitude_stats = json.loads(
		(segy_dir / 'seismic_amplitude_stats.json').read_text(),
	)
	label_values = json.loads((segy_dir / 'label_unique_values.json').read_text())
	summary = (segy_dir / 'segy_summary.md').read_text(encoding='utf-8')
	with (
		segy_dir / 'segy_geometry.csv'
	).open(encoding='utf-8', newline='') as file_obj:
		geometry_rows = list(csv.DictReader(file_obj))

	assert metadata['segy_files']['seismic']['cube_shape'] == [2, 3, 4]
	assert metadata['segy_files']['label']['cube_shape'] == [2, 3, 4]
	assert metadata['shape_consistency']['matches'] is True
	assert metadata['axis_assumption']['cube_to_repo_axes']['cube_axis_0'][
		'repo_axis'
	] == 'x'
	assert legacy_geometry == metadata
	assert len(geometry_rows) == 2
	assert amplitude_stats['stats']['finite_count'] == 24
	assert label_values['stats']['unique_values'] == [0, 1, 2]
	assert label_values['stats']['unexpected_label_values'] == []
	assert 'XYZ仮定' in summary
	assert 'label値0は有効class' in summary
	assert 'f3_segy.seismic_shape: (2, 3, 4)' in result.stdout


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
