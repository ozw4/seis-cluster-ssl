from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_cluster_visualization_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_mae_training_config,
	resolve_manifest_build_config,
	resolve_normalization_qc_config,
	resolve_normalization_stats_config,
)

if TYPE_CHECKING:
	from collections.abc import Callable

CONFIG_DIR = Path('proc/configs/seis_ssl_cluster')
DEFAULT_CONFIGS = (
	(
		CONFIG_DIR / 'build_nopims_manifests.yaml',
		resolve_manifest_build_config,
	),
	(
		CONFIG_DIR / 'prepare_nopims_normalization_stats.yaml',
		resolve_normalization_stats_config,
	),
	(
		CONFIG_DIR / 'filter_manifest_by_normalization_qc.yaml',
		resolve_normalization_qc_config,
	),
	(CONFIG_DIR / 'train_amp_mae.yaml', resolve_mae_training_config),
	(CONFIG_DIR / 'extract_embeddings.yaml', resolve_embedding_extraction_config),
	(CONFIG_DIR / 'cluster_embeddings.yaml', resolve_clustering_config),
	(CONFIG_DIR / 'visualize_clusters.yaml', resolve_cluster_visualization_config),
)


@pytest.mark.parametrize(('config_path', 'resolver'), DEFAULT_CONFIGS)
def test_default_configs_resolve_without_mutating_raw(
	config_path: Path,
	resolver: Callable[[dict[str, object]], dict[str, object]],
) -> None:
	raw = load_config(config_path)
	original = deepcopy(raw)

	resolved = resolver(raw)

	assert raw == original
	assert 'stage' not in raw
	assert resolved['stage']
	assert resolved['paths']['nopims_root'] == '/home/dcuser/data/NOPIMS'
	assert resolved['paths']['artifact_root'] == '/workspace/artifacts/seis_ssl_cluster'


@pytest.mark.parametrize(
	('resolver', 'raw_config'),
	[
		(resolve_manifest_build_config, lambda: _minimal_manifest_build_config()),
		(
			resolve_normalization_stats_config,
			lambda: _minimal_normalization_stats_config(),
		),
		(
			resolve_normalization_qc_config,
			lambda: _minimal_normalization_qc_config(),
		),
		(resolve_mae_training_config, lambda: _minimal_training_config()),
		(resolve_embedding_extraction_config, lambda: _minimal_embedding_config()),
		(resolve_clustering_config, lambda: _minimal_clustering_config()),
		(resolve_cluster_visualization_config, lambda: _minimal_visualization_config()),
	],
)
def test_minimal_stage_configs_resolve_without_stage(
	resolver: Callable[[dict[str, object]], dict[str, object]],
	raw_config: Callable[[], dict[str, object]],
) -> None:
	raw = raw_config()

	resolved = resolver(raw)

	assert 'stage' not in raw
	assert isinstance(resolved['stage'], str)


def test_stale_stage_is_rejected_with_migration_message() -> None:
	cfg = _minimal_training_config()
	cfg['stage'] = 'train_amp_mae'

	with pytest.raises(ValueError, match='stage is selected by the entrypoint'):
		resolve_mae_training_config(cfg)


def test_unrelated_top_level_sections_are_rejected() -> None:
	cfg = _minimal_clustering_config()
	cfg['train'] = {'batch_size': 4}

	with pytest.raises(ValueError, match=r'cluster_embeddings.*train'):
		resolve_clustering_config(cfg)


def test_fixed_contract_keys_are_rejected_from_raw_training_config() -> None:
	cfg = _minimal_training_config()
	cfg['data']['grid_order'] = ['x', 'y', 'z']

	with pytest.raises(ValueError, match=r'data\.grid_order.*fixed'):
		resolve_mae_training_config(cfg)


def test_fixed_contracts_appear_in_resolved_training_config() -> None:
	resolved = resolve_mae_training_config(_minimal_training_config())

	assert resolved['data']['grid_order'] == ['x', 'y', 'z']
	assert resolved['data']['volume_format'] == 'npy_memmap'
	assert resolved['data']['input_channels'] == 1
	assert resolved['data']['target_channels'] == 1
	assert resolved['data']['use_context'] is False
	assert resolved['model']['name'] == 'amp_mae3d'
	assert resolved['model']['in_channels'] == 1
	assert resolved['model']['out_channels'] == 1
	assert resolved['masking']['spatial_mask_mode'] == 'block'
	assert resolved['loss']['reconstruction'] == 'huber'
	assert resolved['loss']['valid_mask_mode'] == 'voxel'
	assert resolved['zero_mask'] == {
		'enabled': True,
		'zero_atol': 0.0,
		'z_sample_influence_radius': 16,
		'xy_trace_influence_radius': 1,
	}


@pytest.mark.parametrize(
	'model_key',
	[
		'encoder_dim',
		'encoder_depth',
		'encoder_heads',
		'decoder_dim',
		'decoder_depth',
		'decoder_heads',
	],
)
def test_training_model_architecture_keys_are_required(model_key: str) -> None:
	cfg = _minimal_training_config()
	del cfg['model'][model_key]

	with pytest.raises(ValueError, match=rf'model\.{model_key}'):
		resolve_mae_training_config(cfg)


def test_raw_explicit_paths_are_preserved_exactly() -> None:
	cfg = _minimal_normalization_qc_config()

	resolved = resolve_normalization_qc_config(cfg)

	assert resolved['paths'] == cfg['paths']
	assert resolved['manifests']['input'] == '/artifacts/manifests/input.json'
	assert resolved['manifests']['output'] == '/artifacts/manifests/output.json'
	assert resolved['splits']['input'] == '/data/NOPIMS/inputs/train.txt'
	assert resolved['splits']['output'] == '/artifacts/splits/train.txt'
	assert resolved['qc']['output_json'] == '/artifacts/qc/report.json'


def test_no_output_paths_are_derived_from_dataset_or_version_names() -> None:
	cfg = _minimal_training_config()

	resolved = resolve_mae_training_config(cfg)

	assert 'dataset' not in resolved
	assert 'version' not in resolved
	assert 'output_root' not in resolved['paths']


def test_nondivisible_crop_patch_geometry_is_rejected() -> None:
	cfg = _minimal_training_config()
	cfg['model']['patch_size'] = [7, 8, 8]

	with pytest.raises(ValueError, match=r'local_crop_size.*patch_size'):
		resolve_mae_training_config(cfg)


def test_legacy_attributes_names_is_rejected_with_actionable_error() -> None:
	cfg = _minimal_training_config()
	cfg['attributes'] = {'names': ['amplitude_norm']}

	with pytest.raises(ValueError, match=r'attributes\.names.*amplitude-only MVP'):
		resolve_mae_training_config(cfg)


def test_legacy_attribute_dropout_is_rejected() -> None:
	cfg = _minimal_training_config()
	cfg['masking']['attribute_dropout_prob'] = 0.25

	with pytest.raises(ValueError, match='attribute_dropout_prob'):
		resolve_mae_training_config(cfg)


def test_build_manifest_stats_dir_under_nopims_root_is_rejected() -> None:
	cfg = _minimal_manifest_build_config()
	cfg['manifest']['normalization_stats_dir'] = (
		'/data/NOPIMS/registry/normalization_stats'
	)

	with pytest.raises(
		ValueError,
		match=r'manifest\.normalization_stats_dir.*paths\.nopims_root',
	):
		resolve_manifest_build_config(cfg)


def test_filter_qc_output_outside_artifact_root_is_rejected() -> None:
	cfg = _minimal_normalization_qc_config()
	cfg['qc']['output_json'] = '/external/qc/normalization_stats_qc.json'

	with pytest.raises(
		ValueError,
		match=r'qc\.output_json.*paths\.artifact_root',
	):
		resolve_normalization_qc_config(cfg)


def _paths() -> dict[str, object]:
	return {
		'nopims_root': '/data/NOPIMS',
		'artifact_root': '/artifacts',
	}


def _minimal_manifest_build_config() -> dict[str, object]:
	return {
		'paths': _paths(),
		'manifest': {
			'input_path_list': '/data/NOPIMS/inputs/train.txt',
			'output_dir': '/artifacts/manifests',
			'normalization_stats_dir': '/artifacts/normalization_stats',
		},
	}


def _minimal_normalization_stats_config() -> dict[str, object]:
	return {
		'paths': _paths(),
		'manifests': {'train': '/artifacts/manifests/train.json'},
		'normalization': {},
	}


def _minimal_normalization_qc_config() -> dict[str, object]:
	return {
		'paths': _paths(),
		'manifests': {
			'input': '/artifacts/manifests/input.json',
			'output': '/artifacts/manifests/output.json',
		},
		'splits': {
			'input': '/data/NOPIMS/inputs/train.txt',
			'output': '/artifacts/splits/train.txt',
		},
		'qc': {
			'output_json': '/artifacts/qc/report.json',
			'excluded_surveys': '/artifacts/qc/excluded_surveys.txt',
		},
	}


def _minimal_training_config() -> dict[str, object]:
	return {
		'paths': _paths(),
		'manifests': {'train': '/artifacts/manifests/train.json'},
		'data': {'local_crop_size': [128, 128, 128]},
		'model': {
			'patch_size': [8, 8, 8],
			'encoder_dim': 384,
			'encoder_depth': 8,
			'encoder_heads': 6,
			'decoder_dim': 256,
			'decoder_depth': 4,
			'decoder_heads': 4,
		},
		'masking': {
			'spatial_mask_ratio': 0.75,
			'block_size_tokens': [2, 2, 2],
		},
		'loss': {},
		'train': {
			'batch_size': 4,
			'samples_per_epoch': 10000,
			'epochs': 100,
			'amp': False,
		},
	}


def _minimal_embedding_config() -> dict[str, object]:
	return {
		'paths': _paths(),
		'manifests': {'input': '/artifacts/manifests/train.json'},
		'embeddings': {
			'checkpoint': '/artifacts/runs/train_amp_mae/mae_latest.pt',
			'output_dir': '/artifacts/embeddings',
		},
	}


def _minimal_clustering_config() -> dict[str, object]:
	return {
		'paths': _paths(),
		'embeddings': {'input_dir': '/artifacts/embeddings'},
		'clustering': {'output_dir': '/artifacts/clusters'},
	}


def _minimal_visualization_config() -> dict[str, object]:
	return {
		'paths': _paths(),
		'clustering': {'input_dir': '/artifacts/clusters'},
		'visualization': {'output_dir': '/artifacts/figures'},
	}
