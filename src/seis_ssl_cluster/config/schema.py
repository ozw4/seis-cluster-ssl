"""SeisSSLCluster configuration constants."""

from __future__ import annotations

from typing import Final

from seis_ssl_cluster.runtime_checks import (
	SUPPORTED_RUNTIME_CHECK_MODES as _SUPPORTED_RUNTIME_CHECK_MODES,
)

DEFAULT_NOPIMS_ROOT: Final = '/home/dcuser/data/NOPIMS'
DEFAULT_ARTIFACT_ROOT: Final = '/workspace/artifacts/seis_ssl_cluster'
DEFAULT_F3_ROOT: Final = '/home/dcuser/data/public_data/field/F3'
F3_FACIES_DATASET_NAME: Final = 'f3_facies_benchmark'
F3_FACIES_DATASET_VERSION: Final = 'facies_benchmark_v1'
# Dataset versions share one raw survey; the version only separates artifact
# namespaces, so inspection and preparation accept every registered version.
F3_FACIES_DATASET_VERSIONS: Final = frozenset(
	{F3_FACIES_DATASET_VERSION, 'facies_benchmark_v2'},
)
F3_FACIES_INSPECTION_ARTIFACT_SUBDIR: Final = (
	'inspection/f3/facies_benchmark_v1'
)
DEFAULT_F3_FACIES_INSPECTION_DIR: Final = (
	f'{DEFAULT_ARTIFACT_ROOT}/{F3_FACIES_INSPECTION_ARTIFACT_SUBDIR}'
)

EXPECTED_GRID_ORDER: Final = ['x', 'y', 'z']
EXPECTED_VOLUME_FORMAT: Final = 'npy_memmap'
EXPECTED_INPUT_CHANNELS: Final = 1
EXPECTED_TARGET_CHANNELS: Final = 1
EXPECTED_USE_CONTEXT: Final = False
EXPECTED_MODEL_NAME: Final = 'amp_mae3d'
EXPECTED_SPATIAL_MASK_MODE: Final = 'block'
BARLOW_TWINS_PRETRAINING_METHOD: Final = 'barlow_twins_3d'
LOCAL_BARLOW_TWINS_PRETRAINING_METHOD: Final = 'local_barlow_twins_3d'
HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY: Final = (
	'horizontal_flip_gaussian_noise_v1'
)
HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY: Final = (
	'horizontal_flip_trace_drop_v1'
)
HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY: Final = (
	'horizontal_flip_zero_phase_z_filter_v1'
)
IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY: Final = 'identity_gaussian_noise_v1'
XY_D4_TRACE_DROP_AUGMENTATION_POLICY: Final = 'xy_d4_trace_drop_v1'
SUPPORTED_BARLOW_TWINS_PRETRAINING_METHODS: Final = frozenset(
	{
		BARLOW_TWINS_PRETRAINING_METHOD,
		LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	}
)
SUPPORTED_RECONSTRUCTION_LOSSES: Final = frozenset({'huber', 'mse', 'l1'})
SUPPORTED_TARGET_NORMALIZATION_MODES: Final = frozenset({'none', 'patch_zscore'})
SUPPORTED_FINITE_CHECK_MODES: Final = frozenset({'strict', 'output_only', 'off'})
SUPPORTED_RUNTIME_CHECK_MODES: Final = _SUPPORTED_RUNTIME_CHECK_MODES
SUPPORTED_AMP_DTYPES: Final = frozenset({'auto', 'bfloat16', 'float16'})
EXPECTED_VALID_MASK_MODE: Final = 'voxel'

STAGE_BUILD_MANIFESTS: Final = 'build_nopims_manifests'
STAGE_NORMALIZATION_STATS: Final = 'prepare_nopims_normalization_stats'
STAGE_NORMALIZATION_QC: Final = 'filter_manifest_by_normalization_qc'
STAGE_MAE_TRAINING: Final = 'train_amp_mae'
STAGE_BARLOW_TWINS_TRAINING: Final = 'barlow_twins_training'
STAGE_STRAT_HMM_PRETEXT_TRAINING: Final = 'train_strat_hmm_pretext'
STAGE_STRAT_HMM_PSEUDO_TARGETS: Final = 'build_strat_hmm_pseudo_targets'
STAGE_EMBEDDING_EXTRACTION: Final = 'extract_embeddings'
STAGE_CLUSTERING: Final = 'cluster_embeddings'
STAGE_CLUSTER_VISUALIZATION: Final = 'visualize_clusters'
STAGE_F3_INSPECT_FILES: Final = 'inspect_f3_files'
STAGE_F3_SEGY_GEOMETRY: Final = 'inspect_f3_segy_geometry'
STAGE_F3_PNG_LABELS: Final = 'inspect_f3_png_labels'
STAGE_F3_QUICKLOOK: Final = 'visualize_f3_quicklook'
STAGE_F3_LABEL_CONSISTENCY: Final = 'check_f3_label_consistency'
STAGE_F3_TOKENIZATION_PREVIEW: Final = 'preview_f3_tokenization'
F3_FACIES_INSPECTION_STAGES: Final = frozenset(
	{
		STAGE_F3_INSPECT_FILES,
		STAGE_F3_SEGY_GEOMETRY,
		STAGE_F3_PNG_LABELS,
		STAGE_F3_QUICKLOOK,
		STAGE_F3_LABEL_CONSISTENCY,
		STAGE_F3_TOKENIZATION_PREVIEW,
	},
)

KNOWN_STAGES: Final = {
	STAGE_BUILD_MANIFESTS,
	STAGE_NORMALIZATION_STATS,
	STAGE_NORMALIZATION_QC,
	STAGE_MAE_TRAINING,
	STAGE_BARLOW_TWINS_TRAINING,
	STAGE_STRAT_HMM_PRETEXT_TRAINING,
	STAGE_STRAT_HMM_PSEUDO_TARGETS,
	STAGE_EMBEDDING_EXTRACTION,
	STAGE_CLUSTERING,
	STAGE_CLUSTER_VISUALIZATION,
}

STAGE_PATH_KEYS: Final = {
	STAGE_BUILD_MANIFESTS: frozenset({'nopims_root', 'artifact_root'}),
	STAGE_NORMALIZATION_STATS: frozenset({'nopims_root', 'artifact_root'}),
	STAGE_NORMALIZATION_QC: frozenset({'nopims_root', 'artifact_root'}),
	STAGE_MAE_TRAINING: frozenset({'artifact_root', 'output_root'}),
	STAGE_BARLOW_TWINS_TRAINING: frozenset({'artifact_root', 'output_root'}),
	STAGE_STRAT_HMM_PRETEXT_TRAINING: frozenset(
		{'artifact_root', 'output_root'},
	),
	STAGE_STRAT_HMM_PSEUDO_TARGETS: frozenset({'artifact_root'}),
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
	'finite_check_mode': 'strict',
}

DEFAULT_MAE_TRAIN_OPTIONS: Final = {
	'num_workers': 8,
	'prefetch_factor': 2,
	'persistent_workers': True,
	'shuffle': True,
	'lr': 3.0e-5,
	'weight_decay': 0.05,
	'amp': False,
	'amp_dtype': 'auto',
	'device': 'cuda',
	'seed': 42,
	'grad_clip_norm': 1.0,
	'runtime_check_mode': 'once',
	'stage_timing': False,
}

DEFAULT_BARLOW_TWINS_AUGMENTATION_OPTIONS: Final = {
	'horizontal_flip_probability': 0.5,
}

DEFAULT_BARLOW_TWINS_OPTIONS: Final = {
	'projector_dim': 384,
	'redundancy_weight': 0.005,
	'normalization_eps': 1.0e-4,
}

DEFAULT_BARLOW_TWINS_TRAIN_OPTIONS: Final = {
	'num_workers': 8,
	'prefetch_factor': 2,
	'persistent_workers': True,
	'shuffle': True,
	'lr': 3.0e-5,
	'weight_decay': 0.05,
	'amp': False,
	'amp_dtype': 'auto',
	'device': 'cuda',
	'seed': 42,
	'grad_clip_norm': 1.0,
	'max_steps': None,
	'allow_overwrite_output': False,
}

DEFAULT_STRAT_HMM_PRETEXT_DATA_OPTIONS: Final = {
	'min_valid_fraction': 0.1,
	'max_resample_attempts': 32,
	'normalized_clip_abs': None,
	'amplitude_agc': {'enabled': False},
	'finite_check_mode': 'strict',
}

DEFAULT_STRAT_HMM_PRETEXT_PSEUDO_TARGET_OPTIONS: Final = {
	'min_confidence': 0.0,
}

DEFAULT_STRAT_HMM_PRETEXT_STUDENT_OPTIONS: Final = {
	'init_checkpoint': None,
	'unfreeze_top_blocks': 0,
}

DEFAULT_STRAT_HMM_PRETEXT_HEAD_OPTIONS: Final = {
	'projection_dim': None,
	'temperature': 0.1,
	'normalize': True,
}

DEFAULT_STRAT_HMM_PRETEXT_LOSS_OPTIONS: Final = {
	'prototype_weight': 1.0,
	'usage_weight': 0.01,
	'entropy_floor': None,
	'distillation_weight': 0.0,
}

DEFAULT_STRAT_HMM_PRETEXT_TRAIN_OPTIONS: Final = {
	'num_workers': 4,
	'shuffle': True,
	'lr': 3.0e-4,
	'encoder_lr': 1.0e-5,
	'weight_decay': 0.05,
	'amp': False,
	'device': 'auto',
	'seed': 42,
	'grad_clip_norm': 1.0,
	'checkpoint_every_steps': None,
	'max_steps': None,
	'allow_overwrite_output': False,
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
