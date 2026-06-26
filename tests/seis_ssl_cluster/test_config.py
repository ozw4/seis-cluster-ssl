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
DEFAULT_CONFIG_TOP_LEVELS = {
	CONFIG_DIR / 'build_nopims_manifests.yaml': {'paths', 'manifest'},
	CONFIG_DIR / 'prepare_nopims_normalization_stats.yaml': {
		'paths',
		'manifests',
		'normalization',
	},
	CONFIG_DIR / 'filter_manifest_by_normalization_qc.yaml': {
		'paths',
		'manifests',
		'splits',
		'qc',
	},
	CONFIG_DIR / 'train_amp_mae.yaml': {
		'paths',
		'manifests',
		'data',
		'zero_mask',
		'model',
		'masking',
		'loss',
		'train',
		'visualization',
	},
	CONFIG_DIR / 'extract_embeddings.yaml': {
		'paths',
		'manifests',
		'embeddings',
		'embedding',
	},
	CONFIG_DIR / 'cluster_embeddings.yaml': {
		'paths',
		'embeddings',
		'clustering',
	},
	CONFIG_DIR / 'visualize_clusters.yaml': {
		'paths',
		'clustering',
		'visualization',
	},
}
DATA_REGISTRY_CONFIGS = DEFAULT_CONFIGS[:3]
DATA_REGISTRY_TOP_LEVELS = {
	path: DEFAULT_CONFIG_TOP_LEVELS[path] for path, _resolver in DATA_REGISTRY_CONFIGS
}
REDUNDANT_DATA_STAGE_SECTIONS = {'stage', 'data', 'model', 'masking', 'loss', 'train'}
CHECKPOINT_OWNED_EXTRACTION_SECTIONS = (
	'data',
	'model',
	'masking',
	'loss',
	'train',
	'zero_mask',
)
DEFAULT_SPLIT_PATH = (
	'/workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1/'
	'train_npy_paths.txt'
)
DEFAULT_MANIFEST_PATH = (
	'/workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/pretrain_v1/'
	'nopims_amplitude_manifests.json'
)
DEFAULT_CLEAN_MANIFEST_PATH = (
	'/workspace/artifacts/seis_ssl_cluster/registry/manifests/nopims/'
	'pretrain_v1_clean/nopims_amplitude_manifests.json'
)
DEFAULT_CLEAN_SPLIT_PATH = (
	'/workspace/artifacts/seis_ssl_cluster/registry/splits/nopims/pretrain_v1_clean/'
	'train_npy_paths.txt'
)
DEFAULT_EMBEDDING_CHECKPOINT_PATH = (
	'/workspace/artifacts/seis_ssl_cluster/runs/amp_mae_pretrain_v1/'
	'mae_latest.pt'
)
DEFAULT_EMBEDDING_DIR = (
	'/workspace/artifacts/seis_ssl_cluster/embeddings/nopims/pretrain_v1'
)
DEFAULT_CLUSTERING_DIR = (
	'/workspace/artifacts/seis_ssl_cluster/clustering/nopims/pretrain_v1'
)
DEFAULT_CLUSTER_VISUALIZATION_DIR = (
	'/workspace/artifacts/seis_ssl_cluster/visualizations/clusters/nopims/'
	'pretrain_v1'
)
FIXED_DISABLED_NORMALIZATION_KEYS = (
	'smooth_time_depth_trend_correction',
	'trace_wise_agc',
	'patch_wise_zscore',
)


@pytest.mark.parametrize('config_path', DEFAULT_CONFIG_TOP_LEVELS)
def test_default_user_yaml_top_level_sections_are_stage_specific(
	config_path: Path,
) -> None:
	raw = load_config(config_path)

	assert set(raw) == DEFAULT_CONFIG_TOP_LEVELS[config_path]


@pytest.mark.parametrize('config_path', DEFAULT_CONFIG_TOP_LEVELS)
def test_default_user_yamls_do_not_define_stage(config_path: Path) -> None:
	raw = load_config(config_path)

	assert 'stage' not in raw


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
	if config_path == CONFIG_DIR / 'train_amp_mae.yaml':
		assert 'nopims_root' not in raw['paths']
		assert 'nopims_root' not in resolved['paths']
		assert (
			resolved['paths']['output_root']
			== '/workspace/artifacts/seis_ssl_cluster/runs/amp_mae_pretrain_v1'
		)
	elif config_path in {
		CONFIG_DIR / 'extract_embeddings.yaml',
		CONFIG_DIR / 'cluster_embeddings.yaml',
		CONFIG_DIR / 'visualize_clusters.yaml',
	}:
		assert 'nopims_root' not in raw['paths']
		assert 'nopims_root' not in resolved['paths']
	else:
		assert resolved['paths']['nopims_root'] == '/home/dcuser/data/NOPIMS'
	assert resolved['paths']['artifact_root'] == '/workspace/artifacts/seis_ssl_cluster'


def test_default_training_config_is_minimal_raw_user_config() -> None:
	raw = load_config(CONFIG_DIR / 'train_amp_mae.yaml')

	assert set(raw) == {
		'paths',
		'manifests',
		'data',
		'zero_mask',
		'model',
		'masking',
		'loss',
		'train',
		'visualization',
	}
	assert raw['paths'] == {
		'artifact_root': '/workspace/artifacts/seis_ssl_cluster',
		'output_root': (
			'/workspace/artifacts/seis_ssl_cluster/runs/amp_mae_pretrain_v1'
		),
	}
	assert raw['manifests']['train'] == DEFAULT_CLEAN_MANIFEST_PATH
	assert raw['manifests']['train_path_list'] == DEFAULT_CLEAN_SPLIT_PATH
	assert raw['data']['min_valid_fraction'] == 0.1
	assert raw['data']['max_resample_attempts'] == 16
	assert raw['zero_mask'] == {
		'enabled': True,
		'zero_atol': 0.0,
		'z_sample_influence_radius': 16,
		'xy_trace_influence_radius': 1,
	}
	assert not {'grid_order', 'volume_format', 'input_channels'} & set(raw['data'])
	assert not {'name', 'in_channels', 'out_channels'} & set(raw['model'])
	assert 'spatial_mask_mode' not in raw['masking']
	assert raw['loss'] == {
		'reconstruction': 'huber',
		'huber_delta': 1.0,
		'gradient_weight': 0.05,
		'target_normalization': {'mode': 'none'},
	}
	assert 'valid_mask_mode' not in raw['loss']


def test_default_embedding_extraction_config_is_minimal_raw_user_config() -> None:
	raw = load_config(CONFIG_DIR / 'extract_embeddings.yaml')

	assert set(raw) == {'paths', 'manifests', 'embeddings', 'embedding'}
	assert raw['paths'] == {
		'artifact_root': '/workspace/artifacts/seis_ssl_cluster',
	}
	assert raw['manifests']['input'] == DEFAULT_CLEAN_MANIFEST_PATH
	assert raw['embeddings']['checkpoint'] == DEFAULT_EMBEDDING_CHECKPOINT_PATH
	assert raw['embeddings']['output_dir'] == DEFAULT_EMBEDDING_DIR
	assert raw['embedding'] == {
		'window_size': [128, 128, 128],
		'overlap': [64, 64, 64],
		'output_dtype': 'float16',
		'batch_size': 1,
		'min_token_valid_fraction': 0.5,
	}
	assert not REDUNDANT_DATA_STAGE_SECTIONS & set(raw)


def test_default_clustering_config_is_minimal_raw_user_config() -> None:
	raw = load_config(CONFIG_DIR / 'cluster_embeddings.yaml')

	assert set(raw) == {'paths', 'embeddings', 'clustering'}
	assert raw['paths'] == {'artifact_root': '/workspace/artifacts/seis_ssl_cluster'}
	assert raw['embeddings'] == {'input_dir': DEFAULT_EMBEDDING_DIR}
	assert raw['clustering'] == {
		'output_dir': DEFAULT_CLUSTERING_DIR,
		'embedding_normalization': 'l2',
		'residualization': {'enabled': False},
		'pca': {'enabled': True, 'n_components': 64, 'whiten': False},
		'sample_tokens': 1000000,
		'method': 'minibatch_kmeans',
		'k_values': [6, 8, 10, 12],
		'minibatch_size': 8192,
		'seed': 42,
	}
	assert not REDUNDANT_DATA_STAGE_SECTIONS & set(raw)


def test_default_cluster_visualization_config_is_minimal_raw_user_config() -> None:
	raw = load_config(CONFIG_DIR / 'visualize_clusters.yaml')

	assert set(raw) == {'paths', 'clustering', 'visualization'}
	assert raw['paths'] == {'artifact_root': '/workspace/artifacts/seis_ssl_cluster'}
	assert raw['clustering'] == {'input_dir': DEFAULT_CLUSTERING_DIR}
	assert raw['visualization'] == {
		'output_dir': DEFAULT_CLUSTER_VISUALIZATION_DIR,
		'survey_ids': [],
		'modes': ['token'],
		'reconstruct_voxel': False,
		'allow_all_surveys_for_voxel_reconstruction': False,
		'skip_existing_voxel_labels': True,
		'max_voxel_output_gib': 50.0,
		'allow_large_voxel_output': False,
		'slice_coordinate_space': 'voxel',
		'xy_slices': [750],
		'xz_slices': [150],
		'dpi': 160,
		'invalid_color': 'lightgray',
		'amplitude_underlay': {'enabled': False, 'alpha': 0.35},
		'amplitude_comparison': {'enabled': False, 'alpha': 0.35},
		'summaries': {'enabled': True, 'include_amplitude_norm': False},
	}
	assert not REDUNDANT_DATA_STAGE_SECTIONS & set(raw)


def test_default_clustering_input_matches_extraction_output() -> None:
	extraction = load_config(CONFIG_DIR / 'extract_embeddings.yaml')
	clustering = load_config(CONFIG_DIR / 'cluster_embeddings.yaml')

	assert (
		clustering['embeddings']['input_dir']
		== extraction['embeddings']['output_dir']
	)


def test_default_visualization_input_matches_clustering_output() -> None:
	clustering = load_config(CONFIG_DIR / 'cluster_embeddings.yaml')
	visualization = load_config(CONFIG_DIR / 'visualize_clusters.yaml')

	assert (
		visualization['clustering']['input_dir']
		== clustering['clustering']['output_dir']
	)


@pytest.mark.parametrize(('config_path', 'resolver'), DATA_REGISTRY_CONFIGS)
def test_default_data_registry_configs_are_minimal(
	config_path: Path,
	resolver: Callable[[dict[str, object]], dict[str, object]],
) -> None:
	raw = load_config(config_path)

	resolver(raw)

	assert set(raw) == DATA_REGISTRY_TOP_LEVELS[config_path]
	assert not REDUNDANT_DATA_STAGE_SECTIONS & set(raw)
	if 'normalization' in raw:
		normalization = raw['normalization']
		assert isinstance(normalization, dict)
		assert not set(FIXED_DISABLED_NORMALIZATION_KEYS) & set(normalization)


def test_default_data_registry_handoff_paths_are_explicit() -> None:
	build = load_config(CONFIG_DIR / 'build_nopims_manifests.yaml')
	normalization = load_config(
		CONFIG_DIR / 'prepare_nopims_normalization_stats.yaml',
	)
	qc = load_config(CONFIG_DIR / 'filter_manifest_by_normalization_qc.yaml')

	assert build['manifest']['input_path_list'] == DEFAULT_SPLIT_PATH
	assert (
		build['manifest']['output_dir'] + '/' + build['manifest']['output_name']
		== DEFAULT_MANIFEST_PATH
	)
	assert normalization['manifests']['train'] == DEFAULT_MANIFEST_PATH
	assert qc['manifests']['input'] == DEFAULT_MANIFEST_PATH
	assert qc['manifests']['output'] == DEFAULT_CLEAN_MANIFEST_PATH
	assert qc['splits']['input'] == DEFAULT_SPLIT_PATH
	assert qc['splits']['output'] == DEFAULT_CLEAN_SPLIT_PATH


def test_cluster_visualization_config_rejects_invalid_comparison_alpha() -> None:
	cfg = _minimal_visualization_config()
	cfg['visualization']['amplitude_comparison']['alpha'] = float('nan')

	with pytest.raises(ValueError, match=r'amplitude_comparison\.alpha'):
		resolve_cluster_visualization_config(cfg)


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


@pytest.mark.parametrize(
	('resolver', 'raw_config'),
	[
		(resolve_manifest_build_config, lambda: _minimal_manifest_build_config()),
		(
			resolve_normalization_stats_config,
			lambda: _minimal_normalization_stats_config(),
		),
		(resolve_normalization_qc_config, lambda: _minimal_normalization_qc_config()),
		(resolve_mae_training_config, lambda: _minimal_training_config()),
		(resolve_embedding_extraction_config, lambda: _minimal_embedding_config()),
		(resolve_clustering_config, lambda: _minimal_clustering_config()),
		(resolve_cluster_visualization_config, lambda: _minimal_visualization_config()),
	],
)
def test_stage_configs_reject_unknown_path_keys(
	resolver: Callable[[dict[str, object]], dict[str, object]],
	raw_config: Callable[[], dict[str, object]],
) -> None:
	cfg = raw_config()
	cfg['paths']['unexpected_path'] = '/unused'

	with pytest.raises(ValueError, match=r'paths.*unexpected_path'):
		resolver(cfg)


@pytest.mark.parametrize(
	('resolver', 'raw_config'),
	[
		(resolve_mae_training_config, lambda: _minimal_training_config()),
		(resolve_embedding_extraction_config, lambda: _minimal_embedding_config()),
		(resolve_clustering_config, lambda: _minimal_clustering_config()),
		(resolve_cluster_visualization_config, lambda: _minimal_visualization_config()),
	],
)
def test_non_registry_configs_reject_nopims_root_path(
	resolver: Callable[[dict[str, object]], dict[str, object]],
	raw_config: Callable[[], dict[str, object]],
) -> None:
	cfg = raw_config()
	cfg['paths']['nopims_root'] = '/data/NOPIMS'

	with pytest.raises(ValueError, match=r'paths.*nopims_root'):
		resolver(cfg)


@pytest.mark.parametrize(
	('resolver', 'raw_config'),
	[
		(resolve_clustering_config, lambda: _minimal_clustering_config()),
		(resolve_cluster_visualization_config, lambda: _minimal_visualization_config()),
	],
)
@pytest.mark.parametrize('section', ['data', 'model', 'masking', 'loss', 'train'])
def test_downstream_configs_reject_redundant_mae_sections(
	resolver: Callable[[dict[str, object]], dict[str, object]],
	raw_config: Callable[[], dict[str, object]],
	section: str,
) -> None:
	cfg = raw_config()
	cfg[section] = {}

	with pytest.raises(ValueError, match=rf'top-level section.*{section}'):
		resolver(cfg)


def test_clustering_config_defaults_missing_residualization_to_disabled() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering'].pop('residualization', None)

	resolved = resolve_clustering_config(cfg)

	assert resolved['clustering']['residualization'] == {'enabled': False}


def test_clustering_config_accepts_enabled_residualization() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['residualization'] = {
		'enabled': True,
		'mode': 'local_token_position',
		'group_by': 'token_phase',
		'add_global_mean_back': True,
		'min_group_count': 32,
	}

	resolved = resolve_clustering_config(cfg)

	assert resolved['clustering']['residualization']['enabled'] is True


@pytest.mark.parametrize(
	('updates', 'error', 'message'),
	[
		({'enabled': 'false'}, TypeError, 'enabled'),
		(
			{
				'enabled': True,
				'group_by': 'token_phase',
				'add_global_mean_back': True,
				'min_group_count': 32,
			},
			ValueError,
			'mode',
		),
		(
			{
				'enabled': True,
				'mode': 'unknown',
				'group_by': 'token_phase',
				'add_global_mean_back': True,
				'min_group_count': 32,
			},
			ValueError,
			'mode',
		),
		(
			{
				'enabled': True,
				'mode': 'local_token_position',
				'group_by': 'unknown',
				'add_global_mean_back': True,
				'min_group_count': 32,
			},
			ValueError,
			'group_by',
		),
		(
			{
				'enabled': True,
				'mode': 'local_token_position',
				'group_by': 'token_phase',
				'add_global_mean_back': 'true',
				'min_group_count': 32,
			},
			TypeError,
			'add_global_mean_back',
		),
		(
			{
				'enabled': True,
				'mode': 'local_token_position',
				'group_by': 'token_phase',
				'add_global_mean_back': True,
				'min_group_count': 0,
			},
			ValueError,
			'min_group_count',
		),
	],
)
def test_clustering_config_validates_residualization(
	updates: dict[str, object],
	error: type[Exception],
	message: str,
) -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['residualization'] = updates

	with pytest.raises(error, match=message):
		resolve_clustering_config(cfg)


def test_clustering_config_rejects_duplicate_k_values() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['k_values'] = [2, 3, 2]

	with pytest.raises(ValueError, match=r'clustering\.k_values.*duplicates'):
		resolve_clustering_config(cfg)


@pytest.mark.parametrize(
	('field', 'value', 'message'),
	[
		('embedding_normalization', 'zscore', 'embedding_normalization'),
		('method', 'kmeans', 'method'),
		('sample_tokens', 0, 'sample_tokens'),
		('minibatch_size', 0, 'minibatch_size'),
		('seed', 1.5, 'seed'),
	],
)
def test_clustering_config_validates_stage_parameters(
	field: str,
	value: object,
	message: str,
) -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering'][field] = value

	with pytest.raises(ValueError, match=message):
		resolve_clustering_config(cfg)


@pytest.mark.parametrize(
	('field', 'value', 'message'),
	[
		('modes', ['token', 'summary'], 'modes'),
		('modes', ['token', 'token'], 'modes.*duplicates'),
		('xy_slices', [-1], 'xy_slices'),
		('max_voxel_output_gib', float('inf'), 'max_voxel_output_gib'),
		('dpi', 0, 'dpi'),
	],
)
def test_cluster_visualization_config_validates_stage_parameters(
	field: str,
	value: object,
	message: str,
) -> None:
	cfg = _minimal_visualization_config()
	cfg['visualization'][field] = value

	with pytest.raises(ValueError, match=message):
		resolve_cluster_visualization_config(cfg)


def test_cluster_visualization_config_rejects_invalid_alpha() -> None:
	cfg = _minimal_visualization_config()
	cfg['visualization']['amplitude_underlay']['alpha'] = float('nan')

	with pytest.raises(ValueError, match=r'amplitude_underlay\.alpha'):
		resolve_cluster_visualization_config(cfg)


@pytest.mark.parametrize(
	('resolver', 'raw_config'),
	[
		(resolve_manifest_build_config, lambda: _minimal_manifest_build_config()),
		(
			resolve_normalization_stats_config,
			lambda: _minimal_normalization_stats_config(),
		),
		(resolve_normalization_qc_config, lambda: _minimal_normalization_qc_config()),
	],
)
@pytest.mark.parametrize('section', ['model', 'train', 'loss', 'masking'])
def test_data_registry_configs_reject_redundant_mae_sections(
	resolver: Callable[[dict[str, object]], dict[str, object]],
	raw_config: Callable[[], dict[str, object]],
	section: str,
) -> None:
	cfg = raw_config()
	cfg[section] = {}

	with pytest.raises(ValueError, match=rf'top-level section.*{section}'):
		resolver(cfg)


@pytest.mark.parametrize(
	'section',
	CHECKPOINT_OWNED_EXTRACTION_SECTIONS,
)
def test_embedding_extraction_config_rejects_redundant_mae_sections(
	section: str,
) -> None:
	cfg = _minimal_embedding_config()
	cfg[section] = {}

	with pytest.raises(ValueError, match=rf'checkpoint-owned.*{section}'):
		resolve_embedding_extraction_config(cfg)


@pytest.mark.parametrize('key', FIXED_DISABLED_NORMALIZATION_KEYS)
def test_fixed_disabled_normalization_options_are_rejected(key: str) -> None:
	cfg = _minimal_normalization_stats_config()
	cfg['normalization'][key] = False

	with pytest.raises(ValueError, match=rf'normalization\.{key}.*removed'):
		resolve_normalization_stats_config(cfg)


@pytest.mark.parametrize(
	'key',
	['clipping_percentiles', 'epsilon', 'max_samples', 'seed'],
)
def test_normalization_stats_user_parameters_are_required(key: str) -> None:
	cfg = _minimal_normalization_stats_config()
	del cfg['normalization'][key]

	with pytest.raises(ValueError, match=rf'normalization\.{key} is required'):
		resolve_normalization_stats_config(cfg)


@pytest.mark.parametrize('key', ['min_iqr', 'max_normalized_abs'])
def test_normalization_qc_thresholds_are_required(key: str) -> None:
	cfg = _minimal_normalization_qc_config()
	del cfg['qc'][key]

	with pytest.raises(ValueError, match=rf'qc\.{key} is required'):
		resolve_normalization_qc_config(cfg)


def test_fixed_contract_keys_are_rejected_from_raw_training_config() -> None:
	cfg = _minimal_training_config()
	cfg['data']['grid_order'] = ['x', 'y', 'z']

	with pytest.raises(ValueError, match=r'data\.grid_order.*fixed'):
		resolve_mae_training_config(cfg)


def test_loss_valid_mask_mode_is_rejected_from_raw_training_config() -> None:
	cfg = _minimal_training_config()
	cfg['loss']['valid_mask_mode'] = 'voxel'

	with pytest.raises(ValueError, match=r'loss\.valid_mask_mode.*fixed'):
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
	assert resolved['data']['amplitude_agc'] == {'enabled': False}


def test_training_amplitude_agc_validation_accepts_disabled_and_enabled() -> None:
	disabled = _minimal_training_config()
	disabled['data']['amplitude_agc'] = {'enabled': False}
	assert resolve_mae_training_config(disabled)['data']['amplitude_agc'] == {
		'enabled': False,
	}

	enabled = _minimal_training_config()
	enabled['data']['amplitude_agc'] = {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 65,
		'eps': 1.0e-3,
		'clip_abs': 5.0,
	}

	assert resolve_mae_training_config(enabled)['data']['amplitude_agc'] == {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 65,
		'eps': 1.0e-3,
		'clip_abs': 5.0,
	}


@pytest.mark.parametrize(
	('amplitude_agc', 'error'),
	[
		({'enabled': 'true'}, TypeError),
		(
			{
				'enabled': True,
				'mode': 'unknown',
				'window_z': 65,
				'eps': 1.0e-3,
				'clip_abs': 5.0,
			},
			ValueError,
		),
		(
			{
				'enabled': True,
				'mode': 'trace_rms_z',
				'eps': 1.0e-3,
				'clip_abs': 5.0,
			},
			ValueError,
		),
		(
			{
				'enabled': True,
				'mode': 'trace_rms_z',
				'window_z': 0,
				'eps': 1.0e-3,
				'clip_abs': 5.0,
			},
			ValueError,
		),
		(
			{
				'enabled': True,
				'mode': 'trace_rms_z',
				'window_z': 64,
				'eps': 1.0e-3,
				'clip_abs': 5.0,
			},
			ValueError,
		),
		(
			{
				'enabled': True,
				'mode': 'trace_rms_z',
				'window_z': 65,
				'eps': 0.0,
				'clip_abs': 5.0,
			},
			ValueError,
		),
		(
			{
				'enabled': True,
				'mode': 'trace_rms_z',
				'window_z': 65,
				'eps': float('inf'),
				'clip_abs': 5.0,
			},
			ValueError,
		),
		(
			{
				'enabled': True,
				'mode': 'trace_rms_z',
				'window_z': 65,
				'eps': float('nan'),
				'clip_abs': 5.0,
			},
			ValueError,
		),
		(
			{
				'enabled': True,
				'mode': 'trace_rms_z',
				'window_z': 65,
				'eps': 1.0e-3,
				'clip_abs': 0.0,
			},
			ValueError,
		),
	],
)
def test_training_amplitude_agc_validation_rejects_invalid_values(
	amplitude_agc: dict[str, object],
	error: type[Exception],
) -> None:
	cfg = _minimal_training_config()
	cfg['data']['amplitude_agc'] = amplitude_agc

	with pytest.raises(error, match='amplitude_agc'):
		resolve_mae_training_config(cfg)


@pytest.mark.parametrize('reconstruction', ['huber', 'mse', 'l1'])
def test_training_loss_reconstruction_modes_resolve(reconstruction: str) -> None:
	cfg = _minimal_training_config()
	cfg['loss']['reconstruction'] = reconstruction
	if reconstruction == 'huber':
		cfg['loss']['huber_delta'] = 1.0
	else:
		cfg['loss'].pop('huber_delta')

	resolved = resolve_mae_training_config(cfg)

	assert resolved['loss']['reconstruction'] == reconstruction
	assert resolved['loss']['gradient_weight'] == 0.05
	assert resolved['loss']['valid_mask_mode'] == 'voxel'
	if reconstruction == 'huber':
		assert resolved['loss']['huber_delta'] == 1.0
	else:
		assert 'huber_delta' not in resolved['loss']


@pytest.mark.parametrize('key', ['reconstruction', 'gradient_weight'])
def test_training_loss_user_keys_are_required(key: str) -> None:
	cfg = _minimal_training_config()
	del cfg['loss'][key]

	with pytest.raises(ValueError, match=rf'loss\.{key} is required'):
		resolve_mae_training_config(cfg)


def test_training_loss_huber_delta_is_required_for_huber() -> None:
	cfg = _minimal_training_config()
	del cfg['loss']['huber_delta']

	with pytest.raises(ValueError, match=r'loss\.huber_delta is required'):
		resolve_mae_training_config(cfg)


def test_training_loss_huber_delta_is_rejected_for_mse() -> None:
	cfg = _minimal_training_config()
	cfg['loss']['reconstruction'] = 'mse'

	with pytest.raises(ValueError, match=r'loss\.huber_delta.*huber'):
		resolve_mae_training_config(cfg)


def test_training_loss_rejects_unsupported_reconstruction() -> None:
	cfg = _minimal_training_config()
	cfg['loss']['reconstruction'] = 'MSE'

	with pytest.raises(ValueError, match=r'loss\.reconstruction.*MSE'):
		resolve_mae_training_config(cfg)


@pytest.mark.parametrize(
	('key', 'value', 'message'),
	[
		('huber_delta', 0.0, r'loss\.huber_delta'),
		('huber_delta', float('inf'), r'loss\.huber_delta'),
		('gradient_weight', -1.0, r'loss\.gradient_weight'),
		('gradient_weight', float('inf'), r'loss\.gradient_weight'),
	],
)
def test_training_loss_numbers_are_validated(
	key: str,
	value: object,
	message: str,
) -> None:
	cfg = _minimal_training_config()
	cfg['loss'][key] = value

	with pytest.raises(ValueError, match=message):
		resolve_mae_training_config(cfg)


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


def test_embedding_extraction_explicit_output_path_is_preserved() -> None:
	cfg = _minimal_embedding_config()
	cfg['embeddings']['output_dir'] = '/artifacts/explicit/embeddings'

	resolved = resolve_embedding_extraction_config(cfg)

	assert resolved['embeddings']['output_dir'] == '/artifacts/explicit/embeddings'


def test_clustering_explicit_paths_are_preserved() -> None:
	cfg = _minimal_clustering_config()
	cfg['embeddings']['input_dir'] = '/artifacts/explicit/embeddings'
	cfg['clustering']['output_dir'] = '/artifacts/explicit/clustering'

	resolved = resolve_clustering_config(cfg)

	assert resolved['embeddings']['input_dir'] == '/artifacts/explicit/embeddings'
	assert resolved['clustering']['output_dir'] == '/artifacts/explicit/clustering'


def test_cluster_visualization_explicit_paths_are_preserved() -> None:
	cfg = _minimal_visualization_config()
	cfg['clustering']['input_dir'] = '/artifacts/explicit/clustering'
	cfg['visualization']['output_dir'] = '/artifacts/explicit/visualizations'

	resolved = resolve_cluster_visualization_config(cfg)

	assert resolved['clustering']['input_dir'] == '/artifacts/explicit/clustering'
	assert (
		resolved['visualization']['output_dir']
		== '/artifacts/explicit/visualizations'
	)


def test_embedding_extraction_output_dir_must_be_under_artifact_root() -> None:
	cfg = _minimal_embedding_config()
	cfg['embeddings']['output_dir'] = '/external/embeddings'

	with pytest.raises(
		ValueError,
		match=r'embeddings\.output_dir.*paths\.artifact_root',
	):
		resolve_embedding_extraction_config(cfg)


def test_training_output_root_must_be_absolute() -> None:
	cfg = _minimal_training_config()
	cfg['paths']['output_root'] = 'relative/run'

	with pytest.raises(ValueError, match=r'paths\.output_root.*absolute'):
		resolve_mae_training_config(cfg)


def test_training_output_root_must_be_under_artifact_root() -> None:
	cfg = _minimal_training_config()
	cfg['paths']['output_root'] = '/external/runs/train_amp_mae'

	with pytest.raises(
		ValueError,
		match=r'paths\.output_root.*paths\.artifact_root',
	):
		resolve_mae_training_config(cfg)


@pytest.mark.parametrize(
	('resolver', 'raw_config', 'section', 'key'),
	[
		(
			resolve_mae_training_config,
			lambda: _minimal_training_config(),
			'paths',
			'output_root',
		),
		(
			resolve_clustering_config,
			lambda: _minimal_clustering_config(),
			'clustering',
			'output_dir',
		),
		(
			resolve_cluster_visualization_config,
			lambda: _minimal_visualization_config(),
			'visualization',
			'output_dir',
		),
	],
)
def test_artifact_output_paths_reject_traversal_after_resolution(
	resolver: Callable[[dict[str, object]], dict[str, object]],
	raw_config: Callable[[], dict[str, object]],
	section: str,
	key: str,
) -> None:
	cfg = raw_config()
	cfg[section][key] = '/artifacts/../outside'

	with pytest.raises(ValueError, match=rf'{section}\.{key}.*paths\.artifact_root'):
		resolver(cfg)


def test_clustering_output_dir_must_be_under_artifact_root() -> None:
	cfg = _minimal_clustering_config()
	cfg['clustering']['output_dir'] = '/external/clustering'

	with pytest.raises(
		ValueError,
		match=r'clustering\.output_dir.*paths\.artifact_root',
	):
		resolve_clustering_config(cfg)


def test_visualization_output_dir_must_be_under_artifact_root() -> None:
	cfg = _minimal_visualization_config()
	cfg['visualization']['output_dir'] = '/external/visualizations'

	with pytest.raises(
		ValueError,
		match=r'visualization\.output_dir.*paths\.artifact_root',
	):
		resolve_cluster_visualization_config(cfg)


def test_embedding_extraction_overlap_must_be_less_than_window() -> None:
	cfg = _minimal_embedding_config()
	cfg['embedding']['window_size'] = [8, 8, 8]
	cfg['embedding']['overlap'] = [4, 8, 4]

	with pytest.raises(ValueError, match=r'embedding\.overlap.*window_size'):
		resolve_embedding_extraction_config(cfg)


def test_embedding_extraction_output_dtype_is_limited() -> None:
	cfg = _minimal_embedding_config()
	cfg['embedding']['output_dtype'] = 'float64'

	with pytest.raises(ValueError, match=r'embedding\.output_dtype.*float16.*float32'):
		resolve_embedding_extraction_config(cfg)


def test_training_config_requires_explicit_output_root() -> None:
	cfg = _minimal_training_config()
	del cfg['paths']['output_root']

	with pytest.raises(TypeError, match=r'paths\.output_root'):
		resolve_mae_training_config(cfg)


def test_training_config_requires_explicit_train_path_list() -> None:
	cfg = _minimal_training_config()
	del cfg['manifests']['train_path_list']

	with pytest.raises(TypeError, match=r'manifests\.train_path_list'):
		resolve_mae_training_config(cfg)


def test_embedding_extraction_config_requires_explicit_geometry() -> None:
	cfg = _minimal_embedding_config()
	del cfg['embedding']

	with pytest.raises(ValueError, match=r'missing required.*embedding'):
		resolve_embedding_extraction_config(cfg)


def test_no_output_paths_are_derived_from_dataset_or_version_names() -> None:
	cfg = _minimal_training_config()

	resolved = resolve_mae_training_config(cfg)

	assert 'dataset' not in resolved
	assert 'version' not in resolved
	assert resolved['paths']['output_root'] == cfg['paths']['output_root']


def test_mae_debug_visualization_enabled_step_trigger_resolves() -> None:
	cfg = _minimal_training_config()
	cfg['visualization'] = {
		'mae_debug': {
			'enabled': True,
			'every_steps': 10,
			'every_epochs': None,
		},
	}

	resolved = resolve_mae_training_config(cfg)

	assert resolved['visualization']['mae_debug']['every_steps'] == 10


def test_mae_debug_visualization_enabled_epoch_trigger_resolves() -> None:
	cfg = _minimal_training_config()
	cfg['visualization'] = {
		'mae_debug': {
			'enabled': True,
			'every_steps': None,
			'every_epochs': 1,
		},
	}

	resolved = resolve_mae_training_config(cfg)

	assert resolved['visualization']['mae_debug']['every_epochs'] == 1


@pytest.mark.parametrize(
	('mae_debug', 'message'),
	[
		({'enabled': True, 'every_steps': 0}, r'mae_debug\.every_steps'),
		(
			{'enabled': True, 'every_steps': None, 'every_epochs': 0},
			r'mae_debug\.every_epochs',
		),
		(
			{'enabled': True, 'every_steps': None, 'every_epochs': None},
			r'every_steps or every_epochs',
		),
		(
			{'enabled': True, 'every_steps': 1, 'xy_slice_index': -1},
			r'mae_debug\.xy_slice_index',
		),
		(
			{'enabled': True, 'every_steps': 1, 'xz_slice_y_index': -1},
			r'mae_debug\.xz_slice_y_index',
		),
		(
			{'enabled': True, 'every_steps': 1, 'clip_percentiles': [99.0, 1.0]},
			r'mae_debug\.clip_percentiles',
		),
		(
			{'enabled': True, 'every_steps': 1, 'columns': ['target', 'target']},
			r'mae_debug\.columns.*duplicates',
		),
		(
			{'enabled': True, 'every_steps': 1, 'columns': ['target', 'bogus']},
			r'mae_debug\.columns.*bogus',
		),
	],
)
def test_mae_debug_visualization_config_rejects_invalid_fields(
	mae_debug: dict[str, object],
	message: str,
) -> None:
	cfg = _minimal_training_config()
	cfg['visualization'] = {'mae_debug': mae_debug}

	with pytest.raises(ValueError, match=message):
		resolve_mae_training_config(cfg)


def test_mae_debug_visualization_rejects_unknown_visualization_key() -> None:
	cfg = _minimal_training_config()
	cfg['visualization'] = {
		'mae_debug': {'enabled': False},
		'unexpected': {},
	}

	with pytest.raises(ValueError, match=r'visualization\.unexpected'):
		resolve_mae_training_config(cfg)


def test_mae_debug_visualization_rejects_unknown_nested_key() -> None:
	cfg = _minimal_training_config()
	cfg['visualization'] = {
		'mae_debug': {
			'enabled': False,
			'unexpected': True,
		},
	}

	with pytest.raises(ValueError, match=r'visualization\.mae_debug\.unexpected'):
		resolve_mae_training_config(cfg)


def test_mae_debug_explicit_output_dir_must_stay_under_output_root() -> None:
	cfg = _minimal_training_config()
	cfg['visualization'] = {
		'mae_debug': {
			'enabled': True,
			'every_steps': 1,
			'output_dir': '/external/debug',
		},
	}

	with pytest.raises(
		ValueError,
		match=r'visualization\.mae_debug\.output_dir.*paths\.output_root',
	):
		resolve_mae_training_config(cfg)


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


def test_build_manifest_config_requires_output_name() -> None:
	cfg = _minimal_manifest_build_config()
	del cfg['manifest']['output_name']

	with pytest.raises(TypeError, match=r'manifest\.output_name'):
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
			'output_name': 'train.json',
			'normalization_stats_dir': '/artifacts/normalization_stats',
		},
	}


def _minimal_normalization_stats_config() -> dict[str, object]:
	return {
		'paths': _paths(),
		'manifests': {'train': '/artifacts/manifests/train.json'},
		'normalization': {
			'clipping_percentiles': [0.5, 99.5],
			'epsilon': 1.0e-6,
			'max_samples': 1000000,
			'seed': 42,
		},
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
			'min_iqr': 1.0e-4,
			'max_normalized_abs': 1.0e6,
		},
	}


def _minimal_training_config() -> dict[str, object]:
	return {
		'paths': {
			'artifact_root': '/artifacts',
			'output_root': '/artifacts/runs/train_amp_mae',
		},
		'manifests': {
			'train': '/artifacts/manifests/train.json',
			'train_path_list': '/artifacts/splits/train_npy_paths.txt',
		},
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
		'loss': {
			'reconstruction': 'huber',
			'huber_delta': 1.0,
			'gradient_weight': 0.05,
			'target_normalization': {'mode': 'none'},
		},
		'train': {
			'batch_size': 4,
			'samples_per_epoch': 10000,
			'epochs': 100,
			'amp': False,
		},
	}


def _minimal_embedding_config() -> dict[str, object]:
	return {
		'paths': {'artifact_root': '/artifacts'},
		'manifests': {'input': '/artifacts/manifests/train.json'},
		'embeddings': {
			'checkpoint': '/artifacts/runs/train_amp_mae/mae_latest.pt',
			'output_dir': '/artifacts/embeddings',
		},
		'embedding': {
			'window_size': [128, 128, 128],
			'overlap': [64, 64, 64],
			'output_dtype': 'float16',
			'batch_size': 1,
			'min_token_valid_fraction': 0.5,
		},
	}


def _minimal_clustering_config() -> dict[str, object]:
	return {
		'paths': {'artifact_root': '/artifacts'},
		'embeddings': {'input_dir': '/artifacts/embeddings'},
		'clustering': {
			'output_dir': '/artifacts/clustering',
			'embedding_normalization': 'l2',
			'pca': {'enabled': True, 'n_components': 64, 'whiten': False},
			'sample_tokens': 1000000,
			'method': 'minibatch_kmeans',
			'k_values': [6, 8, 10, 12],
			'minibatch_size': 8192,
			'seed': 42,
		},
	}


def _minimal_visualization_config() -> dict[str, object]:
	return {
		'paths': {'artifact_root': '/artifacts'},
		'clustering': {'input_dir': '/artifacts/clustering'},
		'visualization': {
			'output_dir': '/artifacts/visualizations',
			'survey_ids': [],
			'modes': ['token'],
			'reconstruct_voxel': False,
			'allow_all_surveys_for_voxel_reconstruction': False,
			'skip_existing_voxel_labels': True,
			'max_voxel_output_gib': 50.0,
			'allow_large_voxel_output': False,
			'slice_coordinate_space': 'voxel',
			'xy_slices': [750],
			'xz_slices': [150],
			'dpi': 160,
			'invalid_color': 'lightgray',
			'amplitude_underlay': {'enabled': False, 'alpha': 0.35},
			'amplitude_comparison': {'enabled': False, 'alpha': 0.35},
			'summaries': {'enabled': True, 'include_amplitude_norm': False},
		},
	}


def test_training_loss_target_normalization_validation() -> None:
	raw = load_config(CONFIG_DIR / 'train_amp_mae.yaml')
	patchnorm = deepcopy(raw)
	patchnorm['loss'] = {
		'reconstruction': 'mse',
		'gradient_weight': 0.0,
		'target_normalization': {
			'mode': 'patch_zscore',
			'eps': 1.0e-6,
			'min_std': 0.05,
		},
	}
	resolved = resolve_mae_training_config(patchnorm)
	assert resolved['loss']['target_normalization'] == {
		'mode': 'patch_zscore',
		'eps': 1.0e-6,
		'min_std': 0.05,
	}

	bad_cases = [
		({'mode': 'bad'}, 'mode'),
		({'mode': 'patch_zscore', 'min_std': 0.05}, 'eps'),
		({'mode': 'patch_zscore', 'eps': 1.0e-6}, 'min_std'),
		({'mode': 'patch_zscore', 'eps': 0.0, 'min_std': 0.05}, 'eps'),
		({'mode': 'patch_zscore', 'eps': 1.0e-6, 'min_std': 0.0}, 'min_std'),
		({'mode': 'none', 'eps': 1.0e-6}, 'eps'),
		({'mode': 'none', 'min_std': 0.05}, 'min_std'),
	]
	for target_normalization, error in bad_cases:
		cfg = deepcopy(raw)
		cfg['loss']['target_normalization'] = target_normalization
		with pytest.raises(ValueError, match=error):
			resolve_mae_training_config(cfg)

	bad_gradient = deepcopy(patchnorm)
	bad_gradient['loss']['gradient_weight'] = 0.05
	with pytest.raises(ValueError, match=r'gradient_weight must be 0\.0'):
		resolve_mae_training_config(bad_gradient)
