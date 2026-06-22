from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, validate_config

CONFIG_DIR = Path('proc/configs/seis_ssl_cluster')
DEFAULT_CONFIGS = (
	CONFIG_DIR / 'build_nopims_manifests.yaml',
	CONFIG_DIR / 'prepare_nopims_normalization_stats.yaml',
	CONFIG_DIR / 'filter_manifest_by_normalization_qc.yaml',
	CONFIG_DIR / 'train_amp_mae.yaml',
	CONFIG_DIR / 'extract_embeddings.yaml',
	CONFIG_DIR / 'cluster_embeddings.yaml',
	CONFIG_DIR / 'visualize_clusters.yaml',
)


@pytest.mark.parametrize('config_path', DEFAULT_CONFIGS)
def test_default_configs_validate(config_path: Path) -> None:
	cfg = load_config(config_path)

	validate_config(cfg)

	assert cfg['paths']['nopims_root'] == '/home/dcuser/data/NOPIMS'
	assert cfg['paths']['artifact_root'] == '/workspace/artifacts/seis_ssl_cluster'
	assert cfg['data']['grid_order'] == ['x', 'y', 'z']
	assert cfg['data']['volume_format'] == 'npy_memmap'
	assert cfg['data']['input_channels'] == 1
	assert cfg['data']['target_channels'] == 1
	assert cfg['data']['use_context'] is False
	assert cfg['model']['in_channels'] == 1
	assert cfg['model']['out_channels'] == 1
	if config_path.name == 'build_nopims_manifests.yaml':
		expected_stats_dir = (
			'/workspace/artifacts/seis_ssl_cluster/registry/normalization_stats'
			'/nopims/pretrain_v1'
		)
		assert (
			cfg['manifest']['normalization_stats_dir']
			== expected_stats_dir
		)


def test_default_manifest_and_qc_configs_use_same_source_path_list() -> None:
	build_cfg = load_config(CONFIG_DIR / 'build_nopims_manifests.yaml')
	filter_cfg = load_config(CONFIG_DIR / 'filter_manifest_by_normalization_qc.yaml')

	assert (
		build_cfg['manifest']['input_path_list']
		== filter_cfg['splits']['input']
	)


def test_load_config_applies_path_defaults(tmp_path: Path) -> None:
	config_path = tmp_path / 'amp.yaml'
	config_path.write_text(
		"""
stage: train_amp_mae
data:
  grid_order: [x, y, z]
  volume_format: npy_memmap
  input_channels: 1
  target_channels: 1
  use_context: false
  local_crop_size: [128, 128, 128]
model:
  name: amp_mae3d
  in_channels: 1
  out_channels: 1
  patch_size: [8, 8, 8]
masking:
  spatial_mask_ratio: 0.75
  spatial_mask_mode: block
  block_size_tokens: [2, 2, 2]
train:
  batch_size: 4
  samples_per_epoch: 10000
  epochs: 100
  num_workers: 8
  amp: false
""",
		encoding='utf-8',
	)

	cfg = validate_config(load_config(config_path))

	assert cfg['paths']['nopims_root'] == '/home/dcuser/data/NOPIMS'
	assert cfg['paths']['artifact_root'] == '/workspace/artifacts/seis_ssl_cluster'


def test_legacy_attributes_names_is_rejected_with_actionable_error() -> None:
	cfg = _valid_config()
	cfg['attributes'] = {'names': ['amplitude_norm']}

	with pytest.raises(ValueError, match=r'attributes\.names.*amplitude-only MVP'):
		validate_config(cfg)


def test_legacy_attribute_dropout_is_rejected() -> None:
	cfg = _valid_config()
	cfg['masking']['attribute_dropout_prob'] = 0.25

	with pytest.raises(ValueError, match='attribute_dropout_prob'):
		validate_config(cfg)


def test_nondivisible_crop_patch_geometry_is_rejected() -> None:
	cfg = _valid_config()
	cfg['model']['patch_size'] = [7, 8, 8]

	with pytest.raises(ValueError, match=r'local_crop_size.*patch_size'):
		validate_config(cfg)


def test_context_enabled_is_rejected_for_mvp0() -> None:
	cfg = _valid_config()
	cfg['data']['use_context'] = True

	with pytest.raises(ValueError, match='data\\.use_context'):
		validate_config(cfg)


def test_paths_outside_repo_are_allowed(tmp_path: Path) -> None:
	cfg = _valid_config()
	cfg['paths']['artifact_root'] = str(tmp_path / 'outside-artifacts')
	cfg['paths']['nopims_root'] = '/opt/external/NOPIMS'

	validate_config(cfg)


def test_build_manifest_stats_dir_under_nopims_root_is_rejected() -> None:
	cfg = load_config(CONFIG_DIR / 'build_nopims_manifests.yaml')
	cfg['manifest']['normalization_stats_dir'] = (
		'/home/dcuser/data/NOPIMS/registry/normalization_stats'
	)

	with pytest.raises(
		ValueError,
		match=r'manifest\.normalization_stats_dir.*paths\.nopims_root',
	):
		validate_config(cfg)


def test_filter_qc_output_outside_artifact_root_is_rejected() -> None:
	cfg = load_config(CONFIG_DIR / 'filter_manifest_by_normalization_qc.yaml')
	cfg['qc']['output_json'] = '/external/qc/normalization_stats_qc.json'

	with pytest.raises(
		ValueError,
		match=r'qc\.output_json.*paths\.artifact_root',
	):
		validate_config(cfg)


def _valid_config() -> dict[str, object]:
	return deepcopy(load_config(CONFIG_DIR / 'train_amp_mae.yaml'))
