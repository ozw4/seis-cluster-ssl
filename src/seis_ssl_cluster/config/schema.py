"""Amplitude-only MVP configuration constants."""

from __future__ import annotations

from typing import Final

DEFAULT_NOPIMS_ROOT: Final = '/home/dcuser/data/NOPIMS'
DEFAULT_ARTIFACT_ROOT: Final = '/workspace/artifacts/seis_ssl_cluster'

EXPECTED_GRID_ORDER: Final = ['x', 'y', 'z']
EXPECTED_VOLUME_FORMAT: Final = 'npy_memmap'
EXPECTED_INPUT_CHANNELS: Final = 1
EXPECTED_TARGET_CHANNELS: Final = 1
EXPECTED_USE_CONTEXT: Final = False
EXPECTED_MODEL_NAME: Final = 'amp_mae3d'
EXPECTED_SPATIAL_MASK_MODE: Final = 'block'
SUPPORTED_RECONSTRUCTION_LOSSES: Final = frozenset({'huber', 'mse', 'l1'})
SUPPORTED_TARGET_NORMALIZATION_MODES: Final = frozenset({'none', 'patch_zscore'})
EXPECTED_VALID_MASK_MODE: Final = 'voxel'

STAGE_BUILD_MANIFESTS: Final = 'build_nopims_manifests'
STAGE_NORMALIZATION_STATS: Final = 'prepare_nopims_normalization_stats'
STAGE_NORMALIZATION_QC: Final = 'filter_manifest_by_normalization_qc'
STAGE_MAE_TRAINING: Final = 'train_amp_mae'
STAGE_EMBEDDING_EXTRACTION: Final = 'extract_embeddings'
STAGE_CLUSTERING: Final = 'cluster_embeddings'
STAGE_CLUSTER_VISUALIZATION: Final = 'visualize_clusters'

KNOWN_STAGES: Final = {
	STAGE_BUILD_MANIFESTS,
	STAGE_NORMALIZATION_STATS,
	STAGE_NORMALIZATION_QC,
	STAGE_MAE_TRAINING,
	STAGE_EMBEDDING_EXTRACTION,
	STAGE_CLUSTERING,
	STAGE_CLUSTER_VISUALIZATION,
}

STAGE_PATH_KEYS: Final = {
	STAGE_BUILD_MANIFESTS: frozenset({'nopims_root', 'artifact_root'}),
	STAGE_NORMALIZATION_STATS: frozenset({'nopims_root', 'artifact_root'}),
	STAGE_NORMALIZATION_QC: frozenset({'nopims_root', 'artifact_root'}),
	STAGE_MAE_TRAINING: frozenset({'artifact_root', 'output_root'}),
	STAGE_EMBEDDING_EXTRACTION: frozenset({'artifact_root'}),
	STAGE_CLUSTERING: frozenset({'artifact_root'}),
	STAGE_CLUSTER_VISUALIZATION: frozenset({'artifact_root'}),
}

FIXED_DATA_CONTRACT: Final = {
	'grid_order': EXPECTED_GRID_ORDER,
	'volume_format': EXPECTED_VOLUME_FORMAT,
	'input_channels': EXPECTED_INPUT_CHANNELS,
	'target_channels': EXPECTED_TARGET_CHANNELS,
	'use_context': EXPECTED_USE_CONTEXT,
}

FIXED_MODEL_CONTRACT: Final = {
	'name': EXPECTED_MODEL_NAME,
	'in_channels': EXPECTED_INPUT_CHANNELS,
	'out_channels': EXPECTED_TARGET_CHANNELS,
}

FIXED_MASKING_CONTRACT: Final = {
	'spatial_mask_mode': EXPECTED_SPATIAL_MASK_MODE,
}

FIXED_LOSS_CONTRACT: Final = {
	'valid_mask_mode': EXPECTED_VALID_MASK_MODE,
}

DEFAULT_MAE_LOSS_OPTIONS: Final = {
	'visible_reconstruction_weight': 0.0,
}

DEFAULT_ZERO_MASK_CONTRACT: Final = {
	'enabled': True,
	'zero_atol': 0.0,
	'z_sample_influence_radius': 16,
	'xy_trace_influence_radius': 1,
}

DEFAULT_MAE_DATA_OPTIONS: Final = {
	'min_valid_fraction': 0.1,
	'max_resample_attempts': 16,
	'amplitude_agc': {'enabled': False},
}

DEFAULT_MAE_TRAIN_OPTIONS: Final = {
	'num_workers': 8,
	'shuffle': True,
	'lr': 3.0e-5,
	'weight_decay': 0.05,
	'amp': False,
	'device': 'cuda',
	'seed': 42,
	'grad_clip_norm': 1.0,
}

DEFAULT_MAE_DEBUG_VISUALIZATION_COLUMNS: Final = (
	'input',
	'masked_input',
	'target',
	'prediction',
	'abs_error',
	'valid_mask',
)

MAE_DEBUG_VISUALIZATION_COLUMNS: Final = frozenset(
	(
		*DEFAULT_MAE_DEBUG_VISUALIZATION_COLUMNS,
		'prediction_norm',
		'prediction_oracle_denorm',
		'abs_error_oracle_denorm',
	),
)

MAE_DEBUG_VISUALIZATION_KEYS: Final = frozenset(
	{
		'enabled',
		'output_dir',
		'every_steps',
		'every_epochs',
		'max_samples',
		'xy_slice_index',
		'xz_slice_y_index',
		'dpi',
		'clip_percentiles',
		'columns',
		'panel_width',
		'panel_height',
		'invalid_color',
	},
)

DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS: Final = {
	'enabled': False,
	'every_steps': 1000,
	'every_epochs': None,
	'max_samples': 1,
	'xy_slice_index': None,
	'xz_slice_y_index': None,
	'dpi': 160,
	'clip_percentiles': (1.0, 99.0),
	'columns': DEFAULT_MAE_DEBUG_VISUALIZATION_COLUMNS,
	'panel_width': 2.6,
	'panel_height': 2.4,
	'invalid_color': 'lightgray',
}

LEGACY_ATTRIBUTE_KEY_PATHS: Final = {
	'attributes.names',
	'attributes.registry',
	'attribute_ids',
	'attribute_dropout_prob',
	'group_dropout_prob',
	'dropped_attribute_weight',
	'attribute_registry',
	'fixed_attribute_registry',
}

LEGACY_ATTRIBUTE_KEY_NAMES: Final = {
	'attribute_ids',
	'attribute_dropout_prob',
	'group_dropout_prob',
	'dropped_attribute_weight',
	'attribute_registry',
	'fixed_attribute_registry',
}
