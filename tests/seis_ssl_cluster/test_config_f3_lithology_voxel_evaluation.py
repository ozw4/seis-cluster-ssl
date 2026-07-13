from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	f3_lithology_voxel_evaluation_config_from_mapping,
)


def _config(tmp_path: Path) -> dict[str, object]:
	artifact_root = tmp_path / 'artifacts'
	f3_root = tmp_path / 'f3'
	return {
		'paths': {'artifact_root': str(artifact_root), 'f3_root': str(f3_root)},
		'dataset': {'name': 'f3', 'version': 'test'},
		'labels': {
			'source_label_volume': str(artifact_root / 'labels.npy'),
			'source_label_segy': str(f3_root / 'labels.segy'),
			'png_label_inventory': str(artifact_root / 'inventory.csv'),
			'segy_geometry_json': str(artifact_root / 'geometry.json'),
			'class_info': str(artifact_root / 'class_info.json'),
		},
		'voxel_predictions': {'input_dir': str(artifact_root / 'predictions')},
		'voxel_dataset': {'input_dir': str(artifact_root / 'supervision')},
		'evaluation': {},
		'outputs': {'output_dir': str(artifact_root / 'evaluation')},
	}


def test_config_resolves_defaults_and_paths(tmp_path: Path) -> None:
	config = f3_lithology_voxel_evaluation_config_from_mapping(_config(tmp_path))

	assert config.monitored_class_ids == (3, 5)
	assert config.boundary_tolerances == (1, 2, 4, 8)
	assert config.boundary_region_radii == (1, 2, 4, 8)
	assert config.chunk_size_x == 8
	assert config.overwrite is False


@pytest.mark.parametrize(
	('section', 'key', 'value', 'error'),
	[
		('evaluation', 'boundary_tolerances', [], ValueError),
		('evaluation', 'boundary_region_radii', [1, 1], ValueError),
		('evaluation', 'monitored_class_ids', [3, '5'], TypeError),
		('evaluation', 'chunk_size_x', 0, ValueError),
		('outputs', 'overwrite', 1, TypeError),
	],
)
def test_config_rejects_invalid_policy(
	tmp_path: Path,
	section: str,
	key: str,
	value: object,
	error: type[Exception],
) -> None:
	raw = _config(tmp_path)
	raw[section][key] = value  # type: ignore[index]
	with pytest.raises(error):
		f3_lithology_voxel_evaluation_config_from_mapping(raw)


def test_config_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
	raw = _config(tmp_path)
	Path(raw['outputs']['output_dir']).mkdir(parents=True)  # type: ignore[index]
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		f3_lithology_voxel_evaluation_config_from_mapping(raw)
