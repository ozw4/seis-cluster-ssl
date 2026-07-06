from __future__ import annotations

from pathlib import Path

import pytest

import seis_ssl_cluster.config.f3_baselines as f3_baselines_config
import seis_ssl_cluster.config.validate as validate_config
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_baselines import (
	f3_lithology_baseline_token_dataset_config_from_mapping,
	random_mae_checkpoint_config_from_mapping,
)

F3_BASELINE_CONFIG_DIR = Path(
	'experiments/f3/facies_benchmark_v1/50_lithology_baselines',
)
BASELINE_TOKEN_CONFIGS = (
	(
		F3_BASELINE_CONFIG_DIR / 'z_only_v1/01_build_baseline_token_dataset.yaml',
		'z_only',
	),
	(
		F3_BASELINE_CONFIG_DIR
		/ 'amplitude_stats_v1/01_build_baseline_token_dataset.yaml',
		'amplitude_stats',
	),
	(
		F3_BASELINE_CONFIG_DIR
		/ 'xyz_coordinates_v1/01_build_baseline_token_dataset.yaml',
		'xyz_coordinates',
	),
)
RANDOM_ENCODER_CONFIG = (
	F3_BASELINE_CONFIG_DIR
	/ 'random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1'
	/ '01_create_random_checkpoint.yaml'
)


@pytest.mark.parametrize(('config_path', 'kind'), BASELINE_TOKEN_CONFIGS)
def test_f3_baseline_token_dataset_configs_resolve_from_stage_module(
	config_path: Path,
	kind: str,
) -> None:
	config = f3_lithology_baseline_token_dataset_config_from_mapping(
		load_config(config_path),
	)

	assert config.features.kind == kind
	assert config.feature_source is not None
	assert config.feature_source['kind'] == kind
	assert 'runs' not in config.outputs.output_dir.parts


def test_random_encoder_config_resolves_from_stage_module() -> None:
	config = random_mae_checkpoint_config_from_mapping(
		load_config(RANDOM_ENCODER_CONFIG),
	)

	assert config.seed == 42
	assert config.output_checkpoint.name == 'mae_random_seed42.pt'
	assert 'runs' not in config.output_checkpoint.parts


def test_f3_baseline_config_entrypoints_reexport_from_validate_module() -> None:
	for name in (
		'f3_lithology_baseline_token_dataset_config_from_mapping',
		'random_mae_checkpoint_config_from_mapping',
		'f3_lithology_comparison_report_config_from_mapping',
		'f3_lithology_comparison_publish_config_from_mapping',
	):
		assert getattr(validate_config, name) is getattr(f3_baselines_config, name)
