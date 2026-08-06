from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_dataset import (
	f3_lithology_voxel_dataset_config_from_mapping,
)

if TYPE_CHECKING:
	from pathlib import Path


def _config(tmp_path: Path) -> dict[str, object]:
	artifact_root = tmp_path / 'artifacts'
	f3_root = tmp_path / 'f3'
	return {
		'paths': {'artifact_root': str(artifact_root), 'f3_root': str(f3_root)},
		'dataset': {'name': 'f3_facies_benchmark', 'version': 'v1'},
		'labels': {
			'source_label_volume': str(artifact_root / 'labels.npy'),
			'source_label_segy': str(f3_root / 'labels.sgy'),
			'png_label_inventory': str(artifact_root / 'inventory.csv'),
			'class_info': str(artifact_root / 'class_info.json'),
			'segy_geometry_json': str(artifact_root / 'geometry.json'),
		},
		'reference_embedding': {
			'metadata_json': str(artifact_root / 'embedding.json'),
			'valid_tokens': str(artifact_root / 'valid.npy'),
		},
		'voxel_dataset': {
			'output_dir': str(artifact_root / 'voxel'),
			'ignore_z_border_samples': 1,
		},
		'outputs': {'overwrite': False},
	}


def test_voxel_dataset_config_resolves_contract(tmp_path: Path) -> None:
	resolved = f3_lithology_voxel_dataset_config_from_mapping(_config(tmp_path))
	assert resolved.output_dir == tmp_path / 'artifacts' / 'voxel'
	assert resolved.ignore_z_border_samples == 1
	assert resolved.overwrite is False


@pytest.mark.parametrize(
	('section', 'key'),
	[('dataset', 'extra'), ('reference_embedding', 'embeddings'), ('outputs', 'skip')],
)
def test_voxel_dataset_config_rejects_unknown_keys(
	tmp_path: Path,
	section: str,
	key: str,
) -> None:
	config = _config(tmp_path)
	config[section][key] = True  # type: ignore[index]
	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_dataset_config_from_mapping(config)


def test_voxel_dataset_config_preserves_runs_output(tmp_path: Path) -> None:
	config = _config(tmp_path)
	explicit_output = str(  # type: ignore[index]
		tmp_path / 'artifacts' / 'runs' / 'voxel'
	)
	config['voxel_dataset']['output_dir'] = explicit_output

	resolved = f3_lithology_voxel_dataset_config_from_mapping(config)

	assert str(resolved.output_dir) == explicit_output
