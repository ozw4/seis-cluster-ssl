from __future__ import annotations

from pathlib import Path

import seis_ssl_cluster.config.f3_lithology as f3_lithology_config
import seis_ssl_cluster.config.validate as validate_config
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology import (
	f3_prepare_volume_config_from_mapping,
)
from seis_ssl_cluster.config.schema import F3_FACIES_DATASET_VERSION

F3_PREPARE_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/10_prepare/01_prepare_f3_volume.yaml',
)


def test_f3_volume_prepare_config_resolves_from_stage_module() -> None:
	raw = load_config(F3_PREPARE_CONFIG)

	config = f3_prepare_volume_config_from_mapping(raw)

	assert config.dataset.version == F3_FACIES_DATASET_VERSION
	assert 'runs' not in config.outputs.volume_dir.parts


def test_f3_volume_prepare_outputs_match_artifact_paths_contract() -> None:
	raw = load_config(F3_PREPARE_CONFIG)
	config = f3_prepare_volume_config_from_mapping(raw)
	artifact_root = config.paths.artifact_root
	version = F3_FACIES_DATASET_VERSION

	assert config.outputs.volume_dir == (
		artifact_root / 'registry' / 'volumes' / 'f3' / version
	)
	assert config.outputs.manifest_path.parent == (
		artifact_root / 'registry' / 'manifests' / 'f3' / version
	)
	assert config.outputs.split_path.parent == (
		artifact_root / 'registry' / 'splits' / 'f3' / version
	)


def test_f3_lithology_config_entrypoints_reexport_from_validate_module() -> None:
	for name in (
		'f3_prepare_volume_config_from_mapping',
		'f3_lithology_token_dataset_config_from_mapping',
		'f3_lithology_probe_config_from_mapping',
		'f3_lithology_prediction_config_from_mapping',
		'f3_lithology_visualization_config_from_mapping',
		'f3_lithology_report_config_from_mapping',
		'f3_lithology_publish_config_from_mapping',
	):
		assert getattr(validate_config, name) is getattr(f3_lithology_config, name)
