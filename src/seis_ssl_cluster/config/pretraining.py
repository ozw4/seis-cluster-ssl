"""Validation and resolution for MAE pretraining configs."""


from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import TypeAlias, TypeVar, cast

import numpy as np
import yaml

from seis_ssl_cluster.clustering.residualization import read_residualizer_npz
from seis_ssl_cluster.config.base import _resolve_base
from seis_ssl_cluster.config.common import (
	_is_int,
	_is_number,
	_merge_section_defaults,
	_required_child_mapping,
	_required_mapping,
	_validate_absolute_path,
	_validate_allowed_keys,
	_validate_bool,
	_validate_fraction,
	_validate_non_empty_path,
	_validate_non_empty_str,
	_validate_nonnegative_finite_number,
	_validate_nonnegative_int,
	_validate_nonnegative_number,
	_validate_optional_fraction,
	_validate_optional_nonnegative_int,
	_validate_optional_positive_int,
	_validate_output_path,
	_validate_path,
	_validate_positive_finite_number,
	_validate_positive_int,
	_validate_positive_int_triplet,
	_validate_positive_number,
	_validate_required_key,
	_validate_required_keys,
)
from seis_ssl_cluster.config.schema import (
	BARLOW_TWINS_PRETRAINING_METHOD,
	DEFAULT_BARLOW_TWINS_AUGMENTATION_OPTIONS,
	DEFAULT_BARLOW_TWINS_OPTIONS,
	DEFAULT_BARLOW_TWINS_TRAIN_OPTIONS,
	DEFAULT_MAE_DATA_OPTIONS,
	DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS,
	DEFAULT_MAE_LOSS_OPTIONS,
	DEFAULT_MAE_TRAIN_OPTIONS,
	DEFAULT_STRAT_HMM_PRETEXT_DATA_OPTIONS,
	DEFAULT_STRAT_HMM_PRETEXT_HEAD_OPTIONS,
	DEFAULT_STRAT_HMM_PRETEXT_LOSS_OPTIONS,
	DEFAULT_STRAT_HMM_PRETEXT_PSEUDO_TARGET_OPTIONS,
	DEFAULT_STRAT_HMM_PRETEXT_STUDENT_OPTIONS,
	DEFAULT_STRAT_HMM_PRETEXT_TRAIN_OPTIONS,
	DEFAULT_VICREG_OPTIONS,
	DEFAULT_ZERO_MASK_CONTRACT,
	EXPECTED_VALID_MASK_MODE,
	FIXED_DATA_CONTRACT,
	FIXED_LOSS_CONTRACT,
	FIXED_MASKING_CONTRACT,
	FIXED_MODEL_CONTRACT,
	HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY,
	HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY,
	IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY,
	LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	LOCAL_VICREG_PRETRAINING_METHOD,
	MAE_DEBUG_VISUALIZATION_COLUMNS,
	MAE_DEBUG_VISUALIZATION_KEYS,
	STAGE_BARLOW_TWINS_TRAINING,
	STAGE_MAE_TRAINING,
	STAGE_STRAT_HMM_PRETEXT_TRAINING,
	STAGE_VICREG_TRAINING,
	SUPPORTED_AMP_DTYPES,
	SUPPORTED_BARLOW_TWINS_PRETRAINING_METHODS,
	SUPPORTED_FINITE_CHECK_MODES,
	SUPPORTED_RECONSTRUCTION_LOSSES,
	SUPPORTED_RUNTIME_CHECK_MODES,
	SUPPORTED_TARGET_NORMALIZATION_MODES,
	SUPPORTED_VICREG_PRETRAINING_METHODS,
	XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
)
from seis_ssl_cluster.stratigraphy.prototypes import (
	MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1,
	validate_multi_resolution_head_ks,
)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])

_FIXED_RAW_KEYS: dict[str, frozenset[str]] = {
	'data': frozenset(FIXED_DATA_CONTRACT),
	'model': frozenset(FIXED_MODEL_CONTRACT),
	'masking': frozenset(FIXED_MASKING_CONTRACT),
	'loss': frozenset(FIXED_LOSS_CONTRACT),
}
_AMPLITUDE_AGC_KEYS = frozenset(
	{'enabled', 'mode', 'window_z', 'eps', 'clip_abs'},
)
_AMPLITUDE_AGC_ENABLED_REQUIRED_KEYS = _AMPLITUDE_AGC_KEYS
_MAE_TRAINING_VISUALIZATION_KEYS = frozenset({'mae_debug'})
_CONTINUATION_KEYS = frozenset({'init_checkpoint', 'unfreeze_top_blocks'})
_D4_TRACE_DROP_AUGMENTATION_KEYS = frozenset(
	{'policy', 'reflection_probability', 'trace_drop_probability'}
)
_HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_KEYS = frozenset(
	{'policy', 'horizontal_flip_probability', 'gaussian_noise_std'}
)
_HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_KEYS = frozenset(
	{'policy', 'horizontal_flip_probability', 'trace_drop_probability'}
)
_HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_KEYS = frozenset(
	{'policy', 'horizontal_flip_probability', 'z_filter_side_weight'}
)
_IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_KEYS = frozenset({'policy', 'gaussian_noise_std'})

_BARLOW_TWINS_SECTION_KEYS: dict[str, frozenset[str]] = {
	'manifests': frozenset({'train', 'train_path_list', 'canonical_input_metadata'}),
	'data': frozenset(
		{
			'local_crop_size',
			'min_valid_fraction',
			'max_resample_attempts',
			'normalized_clip_abs',
			'amplitude_agc',
			'finite_check_mode',
		}
	),
	'zero_mask': frozenset(DEFAULT_ZERO_MASK_CONTRACT),
	'model': frozenset(
		{
			'patch_size',
			'encoder_dim',
			'encoder_depth',
			'encoder_heads',
			'decoder_dim',
			'decoder_depth',
			'decoder_heads',
		}
	),
	'augmentations': frozenset(
		{
			*DEFAULT_BARLOW_TWINS_AUGMENTATION_OPTIONS,
			*_D4_TRACE_DROP_AUGMENTATION_KEYS,
			*_HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_KEYS,
			*_HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_KEYS,
			*_HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_KEYS,
			*_IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_KEYS,
		}
	),
	'barlow_twins': frozenset(
		{
			*DEFAULT_BARLOW_TWINS_OPTIONS,
			'local_pairs_per_crop',
			'method',
		}
	),
	'train': frozenset(
		{
			'batch_size',
			'samples_per_epoch',
			'epochs',
			*DEFAULT_BARLOW_TWINS_TRAIN_OPTIONS,
		}
	),
}

_VICREG_SECTION_KEYS: dict[str, frozenset[str]] = {
	'manifests': _BARLOW_TWINS_SECTION_KEYS['manifests'],
	'data': _BARLOW_TWINS_SECTION_KEYS['data'],
	'zero_mask': _BARLOW_TWINS_SECTION_KEYS['zero_mask'],
	'model': _BARLOW_TWINS_SECTION_KEYS['model'],
	'augmentations': _BARLOW_TWINS_SECTION_KEYS['augmentations'],
	'vicreg': frozenset(
		{
			*DEFAULT_VICREG_OPTIONS,
			'local_pairs_per_crop',
			'method',
		}
	),
	'train': _BARLOW_TWINS_SECTION_KEYS['train'],
}

_STRAT_HMM_PRETEXT_SECTION_KEYS: dict[str, frozenset[str]] = {
	'manifests': frozenset({'train', 'train_path_list'}),
	'data': frozenset(
		{
			'local_crop_size',
			'min_valid_fraction',
			'max_resample_attempts',
			'normalized_clip_abs',
			'amplitude_agc',
			'finite_check_mode',
		},
	),
	'model': frozenset(
		{
			'patch_size',
			'encoder_dim',
			'encoder_depth',
			'encoder_heads',
			'decoder_dim',
			'decoder_depth',
			'decoder_heads',
		},
	),
	'pseudo_targets': frozenset(
		{'input_dir', 'k', 'manifest', 'min_confidence', 'target_representation'}
	),
	'teacher': frozenset({'checkpoint'}),
	'student': frozenset({'init_checkpoint', 'unfreeze_top_blocks'}),
	'head': frozenset(
		{
			'num_prototypes',
			'spec',
			'ks',
			'projection_dim',
			'temperature',
			'normalize',
		},
	),
	'loss': frozenset(
		{
			'prototype_weight',
			'usage_weight',
			'entropy_floor',
			'consistency_weight',
			'consistency_beta',
			'distillation_weight',
		},
	),
	'train': frozenset(
		{
			'batch_size',
			'samples_per_epoch',
			'epochs',
			'num_workers',
			'shuffle',
			'lr',
			'encoder_lr',
			'weight_decay',
			'amp',
			'device',
			'seed',
			'grad_clip_norm',
			'checkpoint_every_steps',
			'max_steps',
			'allow_overwrite_output',
		},
	),
	'zero_mask': frozenset(DEFAULT_ZERO_MASK_CONTRACT),
	'spatial_context': frozenset(
		{
			'objective',
			'mask_semantics',
			'column_fraction',
			'selection_policy',
			'replacement',
			'replacement_initialization',
			'rng_policy',
			'masked_prototype_weight',
			'visible_prototype_weight',
			'distillation_scope',
		}
	),
	'pseudo_target_refresh': frozenset(
		{
			'enabled',
			'semantics',
			'generation_root',
			'refresh_after_epochs',
			'hmm_iterations_per_refresh',
			'embedding_source',
			'embedding_mode',
			'center_initialization',
			'center_update',
			'preprocessing_policy',
			'target_replacement',
			'empty_cluster_policy',
			'checkpoint_selection',
			'initial_hmm_artifacts',
		}
	),
}

_STRAT_HMM_MULTI_HEAD_SPEC = MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1
_STRAT_HMM_MULTI_HEAD_CONSISTENCY_POLICY = 'normalized_order_smooth_l1_v1'
_STRAT_HMM_HARD_TARGET_REPRESENTATION = 'hard_viterbi_labels_v1'
_STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION = 'ordered_path_state_posterior_v1'
_STRAT_HMM_LATERAL_TARGET_REPRESENTATION = 'lateral_mean_field_hard_labels_v1'
_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION = (
	'xy_neighbor_consensus_hard_labels_v1'
)
_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION = (
	'xy_neighbor_unanimous_hard_labels_v1'
)
_STRAT_HMM_POSTERIOR_SEMANTICS = 'ordered_path_cost_gibbs_state_marginal_v1'
_STRAT_HMM_LATERAL_SEMANTICS = 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1'
_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_SEMANTICS = (
	'xy_neighbor_consensus_hard_label_smoothing_v1'
)
_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_SEMANTICS = (
	'xy_neighbor_unanimous_outlier_correction_v1'
)

CENTER_TRACE_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
)
CENTER_TRACE_EXPERIMENT_ROLE = 'multi_head_center_trace_masked_hard_pretext'
CENTER_TRACE_VARIANT = 'ctmask010_nocons'
CENTER_TRACE_OBJECTIVE = 'center_trace_masked_hmm_path_reconstruction_v1'
CENTER_TRACE_MASK_SEMANTICS = 'xy_token_column_full_z_v1'
CENTER_TRACE_SELECTION_POLICY = (
	'supervised_valid_xy_columns_round_half_up_leave_one_v1'
)
CENTER_TRACE_REPLACEMENT = 'learned_encoder_mask_token_v1'
CENTER_TRACE_REPLACEMENT_INITIALIZATION = (
	'normal_std_0p02_train_seed_salted_v1'
)
CENTER_TRACE_RNG_POLICY = 'stateless_step_seed_v1'
CENTER_TRACE_DISTILLATION_SCOPE = 'visible_only_v1'
CENTER_TRACE_SUPERVISED_LOSS = 'structured_hmm_center_trace_masked_hard_v1'
CENTER_TRACE_CONSISTENCY_POLICY = 'disabled_for_center_trace_masked_v1'
CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT = {
	'objective': CENTER_TRACE_OBJECTIVE,
	'mask_semantics': CENTER_TRACE_MASK_SEMANTICS,
	'column_fraction': 0.10,
	'selection_policy': CENTER_TRACE_SELECTION_POLICY,
	'replacement': CENTER_TRACE_REPLACEMENT,
	'replacement_initialization': CENTER_TRACE_REPLACEMENT_INITIALIZATION,
	'rng_policy': CENTER_TRACE_RNG_POLICY,
	'masked_prototype_weight': 0.50,
	'visible_prototype_weight': 0.50,
	'distillation_scope': CENTER_TRACE_DISTILLATION_SCOPE,
}

PERIODIC_REFRESH_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_topblock1_distill_v1'
)
PERIODIC_REFRESH_MODEL_ROLE = 'mh_ctmask010_refresh3ep_hmm2_nocons'
PERIODIC_REFRESH_EXPERIMENT_ROLE = (
	'multi_head_center_trace_masked_periodic_hmm_refresh_hard_pretext'
)
PERIODIC_REFRESH_VARIANT = 'ctmask010_refresh3ep_hmm2_nocons'
PERIODIC_REFRESH_SEMANTICS = 'periodic_student_hmm_center_refresh_v1'
PERIODIC_REFRESH_SCHEDULE = (2, 5, 8, 11, 14, 17, 20)
PERIODIC_REFRESH_SCHEDULE_SEMANTICS = 'after_epochs_2_5_8_11_14_17_20_v1'
PERIODIC_REFRESH_CENTER_UPDATE_SEMANTICS = (
	'warm_start_full_mean_two_iterations_final_decode_v1'
)
PERIODIC_REFRESH_EMBEDDING_SEMANTICS = (
	'current_student_unmasked_eval_full_survey_v1'
)
PERIODIC_REFRESH_PREPROCESSING_POLICY = 'freeze_initial_residualizer_pca_v1'
PERIODIC_REFRESH_TARGET_ACTIVATION_POLICY = 'atomic_next_epoch_activation_v1'
PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY = 'final_completed_epoch_v1'
PERIODIC_REFRESH_HEAD_SPEC = MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1
PERIODIC_REFRESH_TARGET_REPRESENTATION = _STRAT_HMM_HARD_TARGET_REPRESENTATION
PERIODIC_REFRESH_INITIAL_ARTIFACT_COMMON_KEYS = frozenset(
	{
		'clustering_config',
		'preprocessor',
		'residualizer',
		'source_embedding_metadata',
	}
)
PERIODIC_REFRESH_INITIAL_ARTIFACT_HEAD_KEYS = frozenset(
	{'model_metadata', 'hmm_model', 'centers'}
)
PERIODIC_REFRESH_CONFIG_KEYS = frozenset(
	{
		'enabled',
		'semantics',
		'generation_root',
		'refresh_after_epochs',
		'hmm_iterations_per_refresh',
		'embedding_source',
		'embedding_mode',
		'center_initialization',
		'center_update',
		'preprocessing_policy',
		'target_replacement',
		'empty_cluster_policy',
		'checkpoint_selection',
		'initial_hmm_artifacts',
	}
)

_CENTER_TRACE_SCIENTIFIC_IDENTITY_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'head_spec',
		'head_ks',
		'head_projection_dim',
		'head_temperature',
		'head_normalize',
		'target_representation',
		'target_manifest_sha256',
		'target_head_hashes',
		'objective_semantics',
		'mask_semantics',
		'column_fraction',
		'selection_policy',
		'replacement',
		'replacement_initialization',
		'rng_policy',
		'masked_prototype_weight',
		'visible_prototype_weight',
		'distillation_scope',
		'supervised_loss',
		'consistency_policy',
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
		'teacher_checkpoint',
		'student_init_checkpoint',
		'student_unfreeze_top_blocks',
		'model',
		'data',
		'zero_mask',
		'train',
	}
)

_PERIODIC_REFRESH_SCIENTIFIC_IDENTITY_FIELDS = (
	_CENTER_TRACE_SCIENTIFIC_IDENTITY_FIELDS
	| frozenset(
		{
			'model_role',
			'target_refresh_semantics',
			'refresh_schedule_semantics',
			'refresh_after_epochs',
			'hmm_iterations_per_refresh',
			'embedding_source',
			'embedding_mode',
			'refresh_embedding_semantics',
			'center_initialization',
			'center_update',
			'center_update_semantics',
			'preprocessing_policy',
			'target_activation_policy',
			'empty_state_policy',
			'checkpoint_selection_policy',
			'initial_hard_target_manifest_sha256',
			'initial_hmm_artifacts',
			'fixed_preprocessor_sha256',
			'fixed_residualizer_sha256',
			'fixed_clustering_config_sha256',
			'source_embedding_metadata_sha256',
			'source_valid_token_hashes',
			'feature_dimension',
			'generation_root',
		}
	)
)

# These fields are deliberately centralized so checkpoint identity construction can
# distinguish scientific settings from machine-specific execution settings.
MULTI_HEAD_SCIENTIFIC_IDENTITY_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'head_spec',
		'head_ks',
		'head_projection_dim',
		'head_temperature',
		'head_normalize',
		'target_representation',
		'target_manifest_sha256',
		'target_head_hashes',
		'posterior_manifest_sha256',
		'posterior_semantics',
		'posterior_cost_temperature',
		'posterior_head_hashes',
		'lateral_target_manifest_sha256',
		'lateral_target_head_hashes',
		'target_semantics',
		'source_hard_manifest_sha256',
		'source_posterior_manifest_sha256',
		'lateral_smoothing',
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_consensus_target_head_hashes',
		'xy_neighbor_consensus_smoothing',
		'xy_neighbor_unanimous_target_manifest_sha256',
		'xy_neighbor_unanimous_target_head_hashes',
		'xy_neighbor_unanimous_smoothing',
		'supervised_loss',
		'consistency_policy',
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
		'teacher_checkpoint',
		'student_init_checkpoint',
		'student_unfreeze_top_blocks',
		'model',
		'data',
		'zero_mask',
		'train',
	}
)

_XY_NEIGHBOR_CONSENSUS_SCIENTIFIC_IDENTITY_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'head_spec',
		'head_ks',
		'head_projection_dim',
		'head_temperature',
		'head_normalize',
		'target_representation',
		'target_semantics',
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_consensus_target_head_hashes',
		'source_hard_manifest_sha256',
		'xy_neighbor_consensus_smoothing',
		'supervised_loss',
		'consistency_policy',
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
		'teacher_checkpoint',
		'student_init_checkpoint',
		'student_unfreeze_top_blocks',
		'model',
		'data',
		'zero_mask',
		'train',
	}
)

_XY_NEIGHBOR_UNANIMOUS_SCIENTIFIC_IDENTITY_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'head_spec',
		'head_ks',
		'head_projection_dim',
		'head_temperature',
		'head_normalize',
		'target_representation',
		'target_semantics',
		'xy_neighbor_unanimous_target_manifest_sha256',
		'xy_neighbor_unanimous_target_head_hashes',
		'source_hard_manifest_sha256',
		'xy_neighbor_unanimous_smoothing',
		'supervised_loss',
		'consistency_policy',
		'prototype_weight',
		'usage_weight',
		'consistency_weight',
		'consistency_beta',
		'distillation_weight',
		'teacher_checkpoint',
		'student_init_checkpoint',
		'student_unfreeze_top_blocks',
		'model',
		'data',
		'zero_mask',
		'train',
	}
)

_MULTI_HEAD_REPRESENTATION_SPECIFIC_IDENTITY_FIELDS = frozenset(
	{
		'target_representation',
		'target_manifest_sha256',
		'target_head_hashes',
		'posterior_manifest_sha256',
		'posterior_semantics',
		'posterior_cost_temperature',
		'posterior_head_hashes',
		'lateral_target_manifest_sha256',
		'lateral_target_head_hashes',
		'target_semantics',
		'source_hard_manifest_sha256',
		'source_posterior_manifest_sha256',
		'lateral_smoothing',
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_consensus_target_head_hashes',
		'xy_neighbor_consensus_smoothing',
		'xy_neighbor_unanimous_target_manifest_sha256',
		'xy_neighbor_unanimous_target_head_hashes',
		'xy_neighbor_unanimous_smoothing',
	}
)
_MULTI_HEAD_COMMON_SCIENTIFIC_IDENTITY_FIELDS = (
	MULTI_HEAD_SCIENTIFIC_IDENTITY_FIELDS
	- _MULTI_HEAD_REPRESENTATION_SPECIFIC_IDENTITY_FIELDS
)
_MULTI_HEAD_HARD_SCIENTIFIC_IDENTITY_FIELDS = (
	_MULTI_HEAD_COMMON_SCIENTIFIC_IDENTITY_FIELDS
	| frozenset(
		{
			'target_representation',
			'target_manifest_sha256',
			'target_head_hashes',
		}
	)
)
_MULTI_HEAD_POSTERIOR_SCIENTIFIC_IDENTITY_FIELDS = (
	_MULTI_HEAD_COMMON_SCIENTIFIC_IDENTITY_FIELDS
	| frozenset(
		{
			'target_representation',
			'posterior_manifest_sha256',
			'posterior_semantics',
			'posterior_cost_temperature',
			'posterior_head_hashes',
		}
	)
)
_MULTI_HEAD_LATERAL_SCIENTIFIC_IDENTITY_FIELDS = (
	_MULTI_HEAD_COMMON_SCIENTIFIC_IDENTITY_FIELDS
	| frozenset(
		{
			'target_representation',
			'target_semantics',
			'lateral_target_manifest_sha256',
			'lateral_target_head_hashes',
			'source_hard_manifest_sha256',
			'source_posterior_manifest_sha256',
			'lateral_smoothing',
		}
	)
)


def _multi_head_scientific_identity_fields(
	target_representation: str | None,
) -> frozenset[str]:
	"""Return the closed identity field set for one target representation."""
	if target_representation == _STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION:
		return _MULTI_HEAD_POSTERIOR_SCIENTIFIC_IDENTITY_FIELDS
	if target_representation == _STRAT_HMM_LATERAL_TARGET_REPRESENTATION:
		return _MULTI_HEAD_LATERAL_SCIENTIFIC_IDENTITY_FIELDS
	if target_representation == _STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION:
		return _XY_NEIGHBOR_CONSENSUS_SCIENTIFIC_IDENTITY_FIELDS
	if target_representation == _STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION:
		return _XY_NEIGHBOR_UNANIMOUS_SCIENTIFIC_IDENTITY_FIELDS
	return _MULTI_HEAD_HARD_SCIENTIFIC_IDENTITY_FIELDS


MULTI_HEAD_RUNTIME_IDENTITY_FIELDS = frozenset(
	{'device', 'workers', 'stage_timing', 'cache_directory', 'resume_path'}
)
MULTI_HEAD_SCIENTIFIC_TRAIN_FIELDS = frozenset(
	{
		'batch_size',
		'samples_per_epoch',
		'epochs',
		'shuffle',
		'lr',
		'encoder_lr',
		'weight_decay',
		'amp',
		'seed',
		'grad_clip_norm',
		'max_steps',
	}
)


def resolve_barlow_twins_training_config(config: _T) -> Config:
	"""Validate and resolve raw config for 3D Barlow Twins training."""
	resolved, paths = _resolve_base(
		config,
		STAGE_BARLOW_TWINS_TRAINING,
		require_nopims_root=False,
	)
	paths_config = _required_mapping(resolved, 'paths')
	output_root = _validate_path(paths_config, 'output_root', prefix='paths')
	_reject_fixed_contract_keys(resolved)
	_merge_section_defaults(resolved, 'data', DEFAULT_MAE_DATA_OPTIONS)
	raw_augmentations = resolved.get('augmentations')
	if not (
		isinstance(raw_augmentations, Mapping)
		and 'policy' in raw_augmentations
	):
		_merge_section_defaults(
			resolved,
			'augmentations',
			DEFAULT_BARLOW_TWINS_AUGMENTATION_OPTIONS,
		)
	_merge_section_defaults(resolved, 'barlow_twins', DEFAULT_BARLOW_TWINS_OPTIONS)
	_merge_section_defaults(resolved, 'train', DEFAULT_BARLOW_TWINS_TRAIN_OPTIONS)
	_merge_section_defaults(resolved, 'zero_mask', DEFAULT_ZERO_MASK_CONTRACT)

	for section, allowed_keys in _BARLOW_TWINS_SECTION_KEYS.items():
		_validate_allowed_keys(
			_required_mapping(resolved, section),
			allowed_keys,
			prefix=section,
		)

	manifests = _required_mapping(resolved, 'manifests')
	_validate_manifests(manifests)
	if 'canonical_input_metadata' in manifests:
		_validate_non_empty_path(
			manifests,
			'canonical_input_metadata',
			prefix='manifests',
		)

	data = _required_mapping(resolved, 'data')
	local_crop_size = _validate_positive_int_triplet(
		data,
		'local_crop_size',
		prefix='data',
	)
	_validate_optional_fraction(data, 'min_valid_fraction', prefix='data')
	_validate_positive_int(data, 'max_resample_attempts', prefix='data')
	if data.get('normalized_clip_abs') is not None:
		_validate_positive_finite_number(
			data,
			'normalized_clip_abs',
			prefix='data',
		)
	_validate_amplitude_agc(data)
	_validate_finite_check_mode(data)

	model = _required_mapping(resolved, 'model')
	patch_size = _validate_positive_int_triplet(
		model,
		'patch_size',
		prefix='model',
	)
	_validate_model(model)
	if 'continuation' in resolved:
		_validate_continuation(
			_required_mapping(resolved, 'continuation'),
			encoder_depth=int(model['encoder_depth']),
		)
	_validate_divisible_crop_patch(local_crop_size, patch_size)

	barlow_twins = _required_mapping(resolved, 'barlow_twins')
	_validate_positive_int(barlow_twins, 'projector_dim', prefix='barlow_twins')
	_validate_nonnegative_finite_number(
		barlow_twins,
		'redundancy_weight',
		prefix='barlow_twins',
	)
	_validate_positive_finite_number(
		barlow_twins,
		'normalization_eps',
		prefix='barlow_twins',
	)
	_validate_barlow_twins_method(
		barlow_twins,
		local_crop_size=local_crop_size,
		patch_size=patch_size,
	)
	_validate_barlow_twins_augmentations(
		_required_mapping(resolved, 'augmentations'),
		method=cast('str', barlow_twins.get('method', BARLOW_TWINS_PRETRAINING_METHOD)),
		local_crop_size=local_crop_size,
		patch_size=patch_size,
	)

	train = _required_mapping(resolved, 'train')
	_validate_barlow_twins_train(train)
	_validate_zero_mask(_required_mapping(resolved, 'zero_mask'))
	_validate_output_path(
		output_root,
		'paths.output_root',
		input_root=paths.nopims_root,
		input_root_label='paths.nopims_root',
	)

	_merge_section_defaults(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_section_defaults(resolved, 'model', FIXED_MODEL_CONTRACT)
	return resolved


def resolve_vicreg_training_config(config: _T) -> Config:
	"""Validate and resolve raw config for local 3D VICReg training."""
	resolved, paths = _resolve_base(
		config,
		STAGE_VICREG_TRAINING,
		require_nopims_root=False,
	)
	paths_config = _required_mapping(resolved, 'paths')
	output_root = _validate_path(paths_config, 'output_root', prefix='paths')
	_reject_fixed_contract_keys(resolved)
	_merge_section_defaults(resolved, 'data', DEFAULT_MAE_DATA_OPTIONS)
	raw_augmentations = resolved.get('augmentations')
	if not (
		isinstance(raw_augmentations, Mapping)
		and 'policy' in raw_augmentations
	):
		_merge_section_defaults(
			resolved,
			'augmentations',
			DEFAULT_BARLOW_TWINS_AUGMENTATION_OPTIONS,
		)
	_merge_section_defaults(resolved, 'vicreg', DEFAULT_VICREG_OPTIONS)
	_merge_section_defaults(resolved, 'train', DEFAULT_BARLOW_TWINS_TRAIN_OPTIONS)
	_merge_section_defaults(resolved, 'zero_mask', DEFAULT_ZERO_MASK_CONTRACT)

	for section, allowed_keys in _VICREG_SECTION_KEYS.items():
		_validate_allowed_keys(
			_required_mapping(resolved, section),
			allowed_keys,
			prefix=section,
		)

	manifests = _required_mapping(resolved, 'manifests')
	_validate_manifests(manifests)
	if 'canonical_input_metadata' in manifests:
		_validate_non_empty_path(
			manifests,
			'canonical_input_metadata',
			prefix='manifests',
		)

	data = _required_mapping(resolved, 'data')
	local_crop_size = _validate_positive_int_triplet(
		data,
		'local_crop_size',
		prefix='data',
	)
	_validate_optional_fraction(data, 'min_valid_fraction', prefix='data')
	_validate_positive_int(data, 'max_resample_attempts', prefix='data')
	if data.get('normalized_clip_abs') is not None:
		_validate_positive_finite_number(
			data,
			'normalized_clip_abs',
			prefix='data',
		)
	_validate_amplitude_agc(data)
	_validate_finite_check_mode(data)

	model = _required_mapping(resolved, 'model')
	patch_size = _validate_positive_int_triplet(
		model,
		'patch_size',
		prefix='model',
	)
	_validate_model(model)
	if 'continuation' in resolved:
		_validate_vicreg_continuation(
			_required_mapping(resolved, 'continuation'),
			encoder_depth=int(model['encoder_depth']),
		)
	_validate_divisible_crop_patch(local_crop_size, patch_size)

	vicreg = _required_mapping(resolved, 'vicreg')
	_validate_positive_int(vicreg, 'projector_dim', prefix='vicreg')
	for key in (
		'invariance_weight',
		'variance_weight',
		'covariance_weight',
	):
		_validate_nonnegative_finite_number(vicreg, key, prefix='vicreg')
	for key in ('variance_target_std', 'variance_eps'):
		_validate_positive_finite_number(vicreg, key, prefix='vicreg')
	_validate_vicreg_method(
		vicreg,
		local_crop_size=local_crop_size,
		patch_size=patch_size,
	)
	_validate_barlow_twins_augmentations(
		_required_mapping(resolved, 'augmentations'),
		method=cast('str', vicreg['method']),
		local_crop_size=local_crop_size,
		patch_size=patch_size,
		local_method=LOCAL_VICREG_PRETRAINING_METHOD,
		method_section='vicreg',
	)

	_validate_barlow_twins_train(_required_mapping(resolved, 'train'))
	_validate_zero_mask(_required_mapping(resolved, 'zero_mask'))
	_validate_output_path(
		output_root,
		'paths.output_root',
		input_root=paths.nopims_root,
		input_root_label='paths.nopims_root',
	)

	_merge_section_defaults(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_section_defaults(resolved, 'model', FIXED_MODEL_CONTRACT)
	return resolved


def resolve_mae_training_config(config: _T) -> Config:
	"""Validate and resolve raw config for MAE training."""
	resolved, paths = _resolve_base(
		config,
		STAGE_MAE_TRAINING,
		require_nopims_root=False,
	)
	paths_config = _required_mapping(resolved, 'paths')
	output_root = _validate_path(
		paths_config,
		'output_root',
		prefix='paths',
	)
	_reject_fixed_contract_keys(resolved)
	_merge_section_defaults(resolved, 'data', DEFAULT_MAE_DATA_OPTIONS)
	_merge_section_defaults(resolved, 'train', DEFAULT_MAE_TRAIN_OPTIONS)
	_merge_section_defaults(resolved, 'loss', DEFAULT_MAE_LOSS_OPTIONS)
	_merge_section_defaults(resolved, 'zero_mask', DEFAULT_ZERO_MASK_CONTRACT)

	manifests = _required_mapping(resolved, 'manifests')
	_validate_non_empty_path(manifests, 'train', prefix='manifests')
	_validate_non_empty_path(manifests, 'train_path_list', prefix='manifests')
	if 'canonical_input_metadata' in manifests:
		_validate_non_empty_path(
			manifests,
			'canonical_input_metadata',
			prefix='manifests',
		)

	data = _required_mapping(resolved, 'data')
	model = _required_mapping(resolved, 'model')
	masking = _required_mapping(resolved, 'masking')
	loss = _required_mapping(resolved, 'loss')
	train = _required_mapping(resolved, 'train')

	local_crop_size = _validate_positive_int_triplet(
		data,
		'local_crop_size',
		prefix='data',
	)
	_validate_optional_fraction(data, 'min_valid_fraction', prefix='data')
	if 'max_resample_attempts' in data:
		_validate_positive_int(data, 'max_resample_attempts', prefix='data')
	if 'normalized_clip_abs' in data:
		_validate_positive_finite_number(
			data,
			'normalized_clip_abs',
			prefix='data',
		)
	_validate_amplitude_agc(data)
	_validate_finite_check_mode(data)

	patch_size = _validate_positive_int_triplet(
		model,
		'patch_size',
		prefix='model',
	)
	_validate_model(model)
	if 'continuation' in resolved:
		_validate_continuation(
			_required_mapping(resolved, 'continuation'),
			encoder_depth=int(model['encoder_depth']),
		)
	_validate_divisible_crop_patch(local_crop_size, patch_size)
	_validate_output_path(
		output_root,
		'paths.output_root',
		input_root=paths.nopims_root,
		input_root_label='paths.nopims_root',
	)
	_validate_masking(masking)
	_validate_loss(loss)
	_validate_train(train)
	_validate_zero_mask(_required_mapping(resolved, 'zero_mask'))
	if 'visualization' in resolved:
		_validate_mae_training_visualization(
			_required_mapping(resolved, 'visualization'),
		)

	_merge_section_defaults(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_section_defaults(resolved, 'model', FIXED_MODEL_CONTRACT)
	_merge_section_defaults(resolved, 'masking', FIXED_MASKING_CONTRACT)
	_merge_section_defaults(resolved, 'loss', FIXED_LOSS_CONTRACT)
	return resolved


def resolve_strat_hmm_pretext_config(  # noqa: C901, PLR0915
	config: _T,
) -> Config:
	"""Validate and resolve raw config for strat HMM pretext training."""
	resolved, paths = _resolve_base(
		config,
		STAGE_STRAT_HMM_PRETEXT_TRAINING,
		require_nopims_root=False,
	)
	paths_config = _required_mapping(resolved, 'paths')
	output_root = _validate_path(
		paths_config,
		'output_root',
		prefix='paths',
	)
	_reject_fixed_contract_keys(resolved)
	multi_head = _is_strat_hmm_multi_head_config(resolved)
	_merge_strat_hmm_pretext_defaults(resolved)
	_validate_strat_hmm_pretext_sections(resolved)

	manifests = _required_mapping(resolved, 'manifests')
	data = _required_mapping(resolved, 'data')
	model = _required_mapping(resolved, 'model')
	pseudo_targets = _required_mapping(resolved, 'pseudo_targets')
	teacher = _required_mapping(resolved, 'teacher')
	student = _required_mapping(resolved, 'student')
	head = _required_mapping(resolved, 'head')
	loss = _required_mapping(resolved, 'loss')
	train = _required_mapping(resolved, 'train')
	periodic_refresh_identity: Mapping[str, object] | None = None

	_validate_manifests(manifests)
	local_crop_size = _validate_strat_hmm_pretext_data(data)
	patch_size = _validate_positive_int_triplet(
		model,
		'patch_size',
		prefix='model',
	)
	_validate_model(model)
	_validate_divisible_crop_patch(local_crop_size, patch_size)
	_validate_strat_hmm_pretext_pseudo_targets(pseudo_targets, multi_head=multi_head)
	if 'pseudo_target_refresh' in resolved:
		if _is_periodic_refresh_config(resolved) and not isinstance(
			resolved.get('identity'), Mapping
		):
			raise ValueError(
				'periodic refresh requires the top-level scientific identity section'
			)
		periodic_refresh_identity = _validate_periodic_refresh_config(
			_required_mapping(resolved, 'pseudo_target_refresh'),
			output_root=output_root,
			train=train,
			pseudo_targets=pseudo_targets,
			head=head,
			multi_head=multi_head,
		)
	if _is_center_trace_masked_config(resolved):
		if not multi_head:
			raise ValueError(
				'spatial_context is only supported for multi-head strat HMM pretext'
			)
		_validate_center_trace_spatial_context(
			_required_mapping(resolved, 'spatial_context')
		)
	_validate_strat_hmm_pretext_teacher(teacher)
	_validate_strat_hmm_pretext_student(
		student,
		encoder_depth=int(model['encoder_depth']),
	)
	_validate_strat_hmm_pretext_head(head, multi_head=multi_head)
	_validate_strat_hmm_pretext_loss(
		loss,
		unfreeze_top_blocks=int(student['unfreeze_top_blocks']),
		multi_head=multi_head,
	)
	if multi_head:
		target_representation = _strat_hmm_multi_head_target_representation(
			pseudo_targets
		)
		if periodic_refresh_identity is not None and not _is_center_trace_masked_config(
			resolved
		):
			raise ValueError(
				'periodic refresh requires the center-trace spatial_context route'
			)
		if periodic_refresh_identity is not None and target_representation != (
			_STRAT_HMM_HARD_TARGET_REPRESENTATION
		):
			raise ValueError(
				'periodic refresh requires pseudo_targets.target_representation to be '
				f'{_STRAT_HMM_HARD_TARGET_REPRESENTATION!r}'
			)
		if _is_center_trace_masked_config(resolved) and target_representation != (
			_STRAT_HMM_HARD_TARGET_REPRESENTATION
		):
			raise ValueError(
				'spatial_context requires pseudo_targets.target_representation to be '
				f'{_STRAT_HMM_HARD_TARGET_REPRESENTATION!r}'
			)
		if (
			target_representation
			in {
				_STRAT_HMM_LATERAL_TARGET_REPRESENTATION,
				_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION,
				_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION,
			}
			and pseudo_targets['min_confidence'] != 0.0
		):
			raise ValueError(
				'pseudo_targets.min_confidence must be 0.0 for immutable hard targets'
			)
		manifest = _validate_strat_hmm_multi_head_manifest(
			pseudo_targets,
			head,
			target_representation=target_representation,
		)
		_validate_strat_hmm_pretext_identity(
			resolved,
			multi_head=True,
			manifest_sha256=_file_sha256(str(pseudo_targets['manifest'])),
			manifest=manifest,
			target_representation=target_representation,
			periodic_refresh_identity=periodic_refresh_identity,
		)
	else:
		_validate_strat_hmm_pretext_cross_section_values(pseudo_targets, head)
		_validate_strat_hmm_pretext_identity(resolved, multi_head=False)
	_validate_strat_hmm_pretext_train(train)
	if _is_center_trace_masked_config(resolved) and int(train['seed']) < 0:
		raise ValueError('train.seed must be nonnegative for center-trace masking')
	_validate_zero_mask(_required_mapping(resolved, 'zero_mask'))
	_validate_output_path(
		output_root,
		'paths.output_root',
		input_root=paths.nopims_root,
		input_root_label='paths.nopims_root',
	)

	_merge_section_defaults(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_section_defaults(resolved, 'model', FIXED_MODEL_CONTRACT)
	_merge_section_defaults(resolved, 'loss', FIXED_LOSS_CONTRACT)
	return resolved


def _merge_strat_hmm_pretext_defaults(config: Config) -> None:
	_merge_section_defaults(
		config,
		'data',
		DEFAULT_STRAT_HMM_PRETEXT_DATA_OPTIONS,
	)
	_merge_section_defaults(
		config,
		'pseudo_targets',
		DEFAULT_STRAT_HMM_PRETEXT_PSEUDO_TARGET_OPTIONS,
	)
	_merge_section_defaults(
		config,
		'student',
		DEFAULT_STRAT_HMM_PRETEXT_STUDENT_OPTIONS,
	)
	_merge_section_defaults(config, 'head', DEFAULT_STRAT_HMM_PRETEXT_HEAD_OPTIONS)
	_merge_section_defaults(config, 'loss', DEFAULT_STRAT_HMM_PRETEXT_LOSS_OPTIONS)
	_merge_section_defaults(config, 'train', DEFAULT_STRAT_HMM_PRETEXT_TRAIN_OPTIONS)
	_merge_section_defaults(config, 'zero_mask', DEFAULT_ZERO_MASK_CONTRACT)


def _is_strat_hmm_multi_head_config(config: Mapping[str, object]) -> bool:
	head = config.get('head')
	return isinstance(head, Mapping) and 'spec' in head


def _is_center_trace_masked_config(config: Mapping[str, object]) -> bool:
	"""Return whether the explicit center-trace route is configured."""
	return isinstance(config.get('spatial_context'), Mapping)


def _is_periodic_refresh_config(config: Mapping[str, object]) -> bool:
	"""Return whether the explicit periodic-refresh route is enabled."""
	value = config.get('pseudo_target_refresh')
	return isinstance(value, Mapping) and value.get('enabled') is True


def _validate_center_trace_spatial_context(
	spatial_context: Mapping[str, object],
) -> None:
	"""Validate the single closed center-trace masking contract."""
	_validate_allowed_keys(
		spatial_context,
		_STRAT_HMM_PRETEXT_SECTION_KEYS['spatial_context'],
		prefix='spatial_context',
	)
	missing = sorted(
		set(_STRAT_HMM_PRETEXT_SECTION_KEYS['spatial_context']) - set(spatial_context)
	)
	if missing:
		raise ValueError(f'spatial_context is missing required key(s): {missing!r}')
	for key, expected in CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT.items():
		actual = spatial_context.get(key)
		if actual != expected:
			raise ValueError(
				f'spatial_context.{key} must be {expected!r}; got {actual!r}'
			)


def _validate_periodic_refresh_config(  # noqa: C901, PLR0912, PLR0913, PLR0915
	refresh: Mapping[str, object],
	*,
	output_root: Path,
	train: Mapping[str, object],
	pseudo_targets: Mapping[str, object],
	head: Mapping[str, object],
	multi_head: bool,
) -> Mapping[str, object] | None:
	"""Validate and fingerprint the closed periodic-refresh input contract."""
	_validate_allowed_keys(
		refresh,
		PERIODIC_REFRESH_CONFIG_KEYS,
		prefix='pseudo_target_refresh',
	)
	_validate_bool(refresh, 'enabled', prefix='pseudo_target_refresh')
	if refresh['enabled'] is False:
		raised = sorted(set(refresh) - {'enabled'})
		if raised:
			raise ValueError(
				'pseudo_target_refresh policy fields are not allowed '
				'when disabled: '
				f'{raised!r}'
			)
		return None
	_validate_required_keys(
		refresh,
		PERIODIC_REFRESH_CONFIG_KEYS,
		prefix='pseudo_target_refresh',
	)
	if not multi_head:
		raise ValueError('periodic refresh requires a multi-head strat HMM route')
	if tuple(head.get('ks', ())) != (6, 8, 10):
		raise ValueError('periodic refresh requires head.ks exactly [6, 8, 10]')
	if head.get('spec') != PERIODIC_REFRESH_HEAD_SPEC:
		raise ValueError('periodic refresh requires the ordered multi-head head spec')
	if train.get('epochs') != 25 or isinstance(train.get('epochs'), bool):
		raise ValueError('periodic refresh requires train.epochs == 25')
	_exact_string_policy(
		refresh,
		'semantics',
		PERIODIC_REFRESH_SEMANTICS,
		prefix='pseudo_target_refresh',
	)
	_exact_string_policy(
		refresh,
		'embedding_source',
		'current_student',
		prefix='pseudo_target_refresh',
	)
	_exact_string_policy(
		refresh,
		'embedding_mode',
		'unmasked_eval_full_survey',
		prefix='pseudo_target_refresh',
	)
	_exact_string_policy(
		refresh,
		'center_initialization',
		'previous_generation',
		prefix='pseudo_target_refresh',
	)
	_exact_string_policy(
		refresh,
		'center_update',
		'full_mean',
		prefix='pseudo_target_refresh',
	)
	_exact_string_policy(
		refresh,
		'preprocessing_policy',
		'freeze_initial',
		prefix='pseudo_target_refresh',
	)
	_exact_string_policy(
		refresh,
		'target_replacement',
		'atomic_next_epoch',
		prefix='pseudo_target_refresh',
	)
	_exact_string_policy(
		refresh,
		'empty_cluster_policy',
		'error',
		prefix='pseudo_target_refresh',
	)
	_exact_string_policy(
		refresh,
		'checkpoint_selection',
		'final_completed_epoch',
		prefix='pseudo_target_refresh',
	)
	schedule = refresh['refresh_after_epochs']
	if (
		not isinstance(schedule, Sequence)
		or isinstance(schedule, str | bytes)
		or any(
			isinstance(epoch, bool) or not isinstance(epoch, int)
			for epoch in schedule
		)
		or tuple(schedule) != PERIODIC_REFRESH_SCHEDULE
	):
		raise ValueError(
			'pseudo_target_refresh.refresh_after_epochs must be exactly '
			f'{list(PERIODIC_REFRESH_SCHEDULE)!r}'
		)
	if any(epoch < 1 or epoch >= int(train['epochs']) for epoch in schedule):
		raise ValueError(
			'pseudo_target_refresh.refresh_after_epochs must be within the '
			'training epochs and strictly increasing'
		)
	if refresh['hmm_iterations_per_refresh'] != 2 or isinstance(
		refresh['hmm_iterations_per_refresh'], bool
	):
		raise ValueError(
			'pseudo_target_refresh.hmm_iterations_per_refresh must be exactly 2'
		)

	generation_root = _validate_absolute_path(
		refresh,
		'generation_root',
		prefix='pseudo_target_refresh',
	)
	_validate_periodic_generation_root_ownership(generation_root, output_root)

	artifacts = _required_child_mapping(
		refresh,
		'initial_hmm_artifacts',
		prefix='pseudo_target_refresh',
	)
	_validate_allowed_keys(
		artifacts,
		frozenset({'common', 'heads'}),
		prefix='pseudo_target_refresh.initial_hmm_artifacts',
	)
	_validate_required_keys(
		artifacts,
		frozenset({'common', 'heads'}),
		prefix='pseudo_target_refresh.initial_hmm_artifacts',
	)
	common = _required_child_mapping(
		artifacts,
		'common',
		prefix='pseudo_target_refresh.initial_hmm_artifacts',
	)
	_validate_allowed_keys(
		common,
		PERIODIC_REFRESH_INITIAL_ARTIFACT_COMMON_KEYS,
		prefix='pseudo_target_refresh.initial_hmm_artifacts.common',
	)
	_validate_required_keys(
		common,
		PERIODIC_REFRESH_INITIAL_ARTIFACT_COMMON_KEYS,
		prefix='pseudo_target_refresh.initial_hmm_artifacts.common',
	)
	common_refs: dict[str, object] = {}
	for key in ('clustering_config', 'preprocessor', 'source_embedding_metadata'):
		path = _validate_absolute_path(
			common,
			key,
			prefix='pseudo_target_refresh.initial_hmm_artifacts.common',
		)
		if not path.is_file():
			raise FileNotFoundError(
				f'pseudo_target_refresh initial artifact is missing: {path}'
			)
		_validate_initial_artifact_not_in_generation_root(path, generation_root, key)
		common_refs[key] = _artifact_reference(path)
	residualizer_value = common.get('residualizer')
	if residualizer_value is None:
		common_refs['residualizer'] = None
	else:
		residualizer = _validate_absolute_path(
			common,
			'residualizer',
			prefix='pseudo_target_refresh.initial_hmm_artifacts.common',
		)
		if not residualizer.is_file():
			raise FileNotFoundError(
				f'pseudo_target_refresh initial artifact is missing: {residualizer}'
			)
		_validate_initial_artifact_not_in_generation_root(
			residualizer, generation_root, 'residualizer'
		)
		common_refs['residualizer'] = _artifact_reference(residualizer)

	heads = _required_child_mapping(
		artifacts,
		'heads',
		prefix='pseudo_target_refresh.initial_hmm_artifacts',
	)
	if set(heads) != {'6', '8', '10'}:
		raise ValueError(
			'pseudo_target_refresh.initial_hmm_artifacts.heads must contain '
			'exactly string keys 6, 8, and 10'
		)
	initial_heads: dict[str, object] = {}
	feature_dimensions: set[int] = set()
	metadata_identities: list[dict[str, object]] = []
	for key in ('6', '8', '10'):
		head_entry = _required_child_mapping(
			heads,
			key,
			prefix='pseudo_target_refresh.initial_hmm_artifacts.heads',
		)
		_validate_allowed_keys(
			head_entry,
			PERIODIC_REFRESH_INITIAL_ARTIFACT_HEAD_KEYS,
			prefix=f'pseudo_target_refresh.initial_hmm_artifacts.heads.{key}',
		)
		_validate_required_keys(
			head_entry,
			PERIODIC_REFRESH_INITIAL_ARTIFACT_HEAD_KEYS,
			prefix=f'pseudo_target_refresh.initial_hmm_artifacts.heads.{key}',
		)
		metadata_path = _validate_absolute_path(
			head_entry,
			'model_metadata',
			prefix=f'pseudo_target_refresh.initial_hmm_artifacts.heads.{key}',
		)
		hmm_model_path = _validate_absolute_path(
			head_entry,
			'hmm_model',
			prefix=f'pseudo_target_refresh.initial_hmm_artifacts.heads.{key}',
		)
		centers_path = _validate_absolute_path(
			head_entry,
			'centers',
			prefix=f'pseudo_target_refresh.initial_hmm_artifacts.heads.{key}',
		)
		for label, path in (
			('model_metadata', metadata_path),
			('hmm_model', hmm_model_path),
			('centers', centers_path),
		):
			if not path.is_file():
				raise FileNotFoundError(
					f'pseudo_target_refresh initial artifact is missing: {path}'
				)
			_validate_initial_artifact_not_in_generation_root(
				path, generation_root, f'heads.{key}.{label}'
			)
		model_metadata = _load_json_object(
			metadata_path,
			f'pseudo_target_refresh heads.{key}.model_metadata',
		)
		k = int(key)
		if model_metadata.get('k') != k:
			raise ValueError(
				f'pseudo_target_refresh model_metadata K mismatch for head {key}'
			)
		centers = np.load(centers_path, mmap_mode='r', allow_pickle=False)
		try:
			if (
				centers.dtype != np.dtype('float32')
				or centers.ndim != 2
				or centers.shape[0] != k
				or centers.shape[1] <= 0
				or not np.isfinite(centers).all()
			):
				raise ValueError(
					f'pseudo_target_refresh centers shape/dtype is invalid for K={k}'
				)
			center_feature_dimension = int(centers.shape[1])
			feature_dimensions.add(center_feature_dimension)
		finally:
			del centers
		metadata_identity = _periodic_metadata_identity(model_metadata)
		if center_feature_dimension != metadata_identity['feature_dimension']:
			raise ValueError(
				'pseudo_target_refresh centers feature dimension does not match '
				f'model metadata for K={k}'
			)
		metadata_identities.append(metadata_identity)
		initial_heads[key] = {
			'model_metadata': _artifact_reference(metadata_path),
			'hmm_model': _artifact_reference(hmm_model_path),
			'centers': _artifact_reference(centers_path),
		}
	if len(feature_dimensions) != 1:
		raise ValueError(
			'pseudo_target_refresh initial centers must share one feature dimension'
		)
	if len({
		json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
		for identity in metadata_identities
	}) != 1:
		raise ValueError(
		'pseudo_target_refresh initial HMM metadata/preprocessing identity differs '
		'across K values'
	)

	target_manifest_path = _validate_absolute_path(
		pseudo_targets,
		'manifest',
		prefix='pseudo_targets',
	)
	_validate_initial_artifact_not_in_generation_root(
		target_manifest_path, generation_root, 'pseudo_targets.manifest'
	)
	target_manifest = _load_json_object(
		target_manifest_path, 'pseudo_target_refresh initial target manifest'
	)
	if target_manifest.get('head_ks') != [6, 8, 10]:
		raise ValueError(
		'pseudo_target_refresh initial target manifest must have head_ks [6, 8, 10]'
	)
	_source_target_roots_must_be_immutable(
		target_manifest,
		generation_root,
		'pseudo_target_refresh initial target manifest',
	)
	initial_artifact_paths = [
		target_manifest_path,
		Path(str(common_refs['clustering_config']['path'])),
		Path(str(common_refs['preprocessor']['path'])),
		Path(str(common_refs['source_embedding_metadata']['path'])),
	]
	if common_refs['residualizer'] is not None:
		initial_artifact_paths.append(
			Path(str(common_refs['residualizer']['path']))
		)
	initial_artifact_paths.extend(
		Path(str(reference['path']))
		for head in initial_heads.values()
		for reference in head.values()
	)
	if len({path.resolve() for path in initial_artifact_paths}) != len(
		initial_artifact_paths
	):
		raise ValueError(
			'pseudo_target_refresh initial artifacts must be distinct immutable files'
		)
	source_metadata_path = Path(str(common_refs['source_embedding_metadata']['path']))
	source_metadata = _load_json_object(
		source_metadata_path,
		'pseudo_target_refresh initial source embedding metadata',
	)
	common_target = target_manifest.get('common')
	if not isinstance(common_target, Mapping):
		raise TypeError(
			'pseudo_target_refresh initial target manifest common must be a mapping'
		)
	target_valid_token_hashes = common_target.get('valid_tokens_sha256')
	if not isinstance(target_valid_token_hashes, Mapping):
		raise TypeError(
			'pseudo_target_refresh initial target manifest must record '
			'valid-token hashes'
		)
	source_valid_token_hashes = _periodic_valid_token_hashes(
		target_valid_token_hashes,
		'pseudo_target_refresh initial target manifest valid-token hashes',
	)
	_validate_periodic_initial_artifact_identity(
		common_refs=common_refs,
		metadata_identity=metadata_identities[0],
		source_metadata_path=source_metadata_path,
		source_metadata=source_metadata,
		target_manifest=target_manifest,
	)
	return {
		'generation_root': str(generation_root),
		'refresh_after_epochs': list(PERIODIC_REFRESH_SCHEDULE),
		'hmm_iterations_per_refresh': 2,
		'initial_hard_target_manifest': _artifact_reference(target_manifest_path),
		'initial_hmm_artifacts': {
			'common': common_refs,
			'heads': initial_heads,
		},
		'fixed_preprocessor_sha256': common_refs['preprocessor']['sha256'],
		'fixed_residualizer_sha256': (
			None
			if common_refs['residualizer'] is None
			else common_refs['residualizer']['sha256']
		),
		'fixed_clustering_config_sha256': common_refs['clustering_config']['sha256'],
		'source_embedding_metadata_sha256': common_refs[
			'source_embedding_metadata'
		]['sha256'],
		'source_valid_token_hashes': source_valid_token_hashes,
		'feature_dimension': next(iter(feature_dimensions)),
	}


def _exact_string_policy(
	parent: Mapping[str, object],
	key: str,
	expected: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if value != expected:
		raise ValueError(f'{prefix}.{key} must be {expected!r}; got {value!r}')


def _validate_initial_artifact_not_in_generation_root(
	path: Path,
	generation_root: Path,
	label: str,
) -> None:
	try:
		path.resolve(strict=False).relative_to(
			generation_root.resolve(strict=False)
		)
	except ValueError:
		return
	raise ValueError(
		f'pseudo_target_refresh initial artifact {label} must be outside '
		'generation_root'
	)


def _validate_periodic_generation_root_ownership(
	generation_root: Path,
	output_root: Path,
) -> None:
	resolved_generation_root = generation_root.resolve(strict=False)
	resolved_output_root = output_root.resolve(strict=False)
	try:
		relative = resolved_generation_root.relative_to(resolved_output_root)
	except ValueError as exc:
		raise ValueError(
			'pseudo_target_refresh.generation_root must be a strict child of '
			'paths.output_root'
		) from exc
	if relative == Path():
		raise ValueError(
			'pseudo_target_refresh.generation_root must be a strict child of '
			'paths.output_root'
		)


def _artifact_reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': _file_sha256(str(path))}


def _load_json_object(path: Path, label: str) -> dict[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as exc:
		raise ValueError(f'{label} must be valid JSON: {path}') from exc
	if not isinstance(value, dict):
		raise TypeError(f'{label} must be a JSON object')
	return value


def _periodic_metadata_identity(  # noqa: C901, PLR0912, PLR0915
	value: Mapping[str, object]
) -> dict[str, object]:
	"""Return the cross-K identity that fixes the initial HMM input contract."""
	embedding_inputs = value.get('embedding_inputs')
	if (
		not isinstance(embedding_inputs, Sequence)
		or isinstance(embedding_inputs, str | bytes)
		or not embedding_inputs
	):
		raise ValueError(
			'periodic refresh model metadata must record ordered embedding_inputs'
		)
	ordered_inputs: list[dict[str, str]] = []
	for entry in embedding_inputs:
		if not isinstance(entry, Mapping):
			raise TypeError(
				'periodic refresh embedding input metadata must be a mapping'
			)
		survey_id = entry.get('survey_id')
		if not isinstance(survey_id, str) or not survey_id:
			raise ValueError('periodic refresh embedding input survey_id is invalid')
		if any(item['survey_id'] == survey_id for item in ordered_inputs):
			raise ValueError(
				'periodic refresh embedding input ordering contains duplicate survey_id'
			)
		for key in (
			'embeddings_path',
			'valid_tokens_path',
			'metadata_path',
			'metadata_sha256',
		):
			if not isinstance(entry.get(key), str) or not entry[key]:
				raise ValueError(
					f'periodic refresh embedding input {key} is missing for {survey_id}'
				)
		ordered_inputs.append(
				{
					'survey_id': survey_id,
					'embeddings_path': str(entry['embeddings_path']),
					'valid_tokens_path': str(entry['valid_tokens_path']),
					'metadata_path': str(entry['metadata_path']),
					'metadata_sha256': str(entry['metadata_sha256']),
				}
			)

	compatibility = value.get('embedding_compatibility_signature')
	if not isinstance(compatibility, Mapping):
		raise TypeError(
			'periodic refresh model metadata is missing '
			'embedding_compatibility_signature'
		)
	raw_feature_dimension = _periodic_positive_int(
		compatibility.get('embedding_dim'),
		'periodic embedding compatibility embedding_dim',
	)
	normalization = value.get('normalization')
	if normalization not in {'l2', 'none'}:
		raise ValueError(
			'periodic refresh model metadata normalization is invalid: '
			f'{normalization!r}'
		)
	residualization = _periodic_preprocessing_identity(
		value.get('residualization'), 'residualization'
	)
	pca = _periodic_pca_identity(value.get('pca'))

	strat = value.get('stratigraphic_hmm')
	if not isinstance(strat, Mapping):
		raise TypeError('periodic refresh model metadata is missing stratigraphic_hmm')
	if strat.get('emission_source') != 'embedding':
		raise ValueError(
			'periodic refresh initial HMM ordering requires embedding emissions'
	)
	if strat.get('z_axis') != 2 or strat.get('z_direction') != (
		'increasing_downward'
	):
		raise ValueError(
			'periodic refresh initial HMM ordering identity is invalid '
			'(z_axis/z_direction)'
	)
	initialization = strat.get('init')
	if not isinstance(initialization, Mapping) or initialization.get(
		'order_by'
	) != 'mean_z':
		raise ValueError(
			'periodic refresh initial HMM ordering requires init.order_by == '
			'mean_z'
	)
	edge_margin = _periodic_nonnegative_int_triplet(
		strat.get('edge_margin_tokens'),
		'periodic initial HMM edge_margin_tokens',
	)
	update = strat.get('update')
	if not isinstance(update, Mapping):
		raise TypeError(
			'periodic refresh initial HMM update policy must be a mapping'
		)
	if not isinstance(update.get('empty_cluster_policy'), str):
		raise TypeError(
			'periodic refresh initial HMM ordering metadata is missing update policy'
		)
	prepared = strat.get('prepared_feature_cache')
	if not isinstance(prepared, Mapping):
		raise TypeError(
		'periodic refresh model metadata is missing prepared_feature_cache identity'
	)
	if prepared.get('feature_mode') != 'embedding':
		raise ValueError(
		'periodic refresh prepared-feature identity must use embedding features'
	)
	prepared_surveys = prepared.get('surveys')
	if (
		not isinstance(prepared_surveys, Sequence)
		or isinstance(prepared_surveys, str | bytes)
		or not prepared_surveys
	):
		raise ValueError(
		'periodic refresh prepared-feature identity must record surveys'
	)
	prepared_identity: list[dict[str, object]] = []
	feature_dimensions: set[int] = set()
	for entry in prepared_surveys:
		if not isinstance(entry, Mapping):
			raise TypeError('periodic prepared survey identity must be a mapping')
		survey_id = entry.get('survey_id')
		if not isinstance(survey_id, str) or not survey_id:
			raise ValueError('periodic prepared survey identity has invalid survey_id')
		feature_dimension = _periodic_positive_int(
			entry.get('feature_dim'),
			f'periodic prepared feature dimension for {survey_id}',
		)
		feature_dimensions.add(feature_dimension)
		prepared_identity.append(
			{'survey_id': survey_id, 'feature_dim': feature_dimension}
		)
	if len(feature_dimensions) != 1:
		raise ValueError(
			'periodic refresh prepared-feature surveys have inconsistent feature '
			'dimensions'
		)
	feature_dimension = next(iter(feature_dimensions))
	pca_enabled = pca['enabled']
	if pca_enabled:
		if pca['effective_n_components'] != feature_dimension:
			raise ValueError(
				'periodic refresh PCA output dimension does not match centers'
			)
	elif feature_dimension != raw_feature_dimension:
		raise ValueError(
			'periodic refresh unprojected feature dimension does not match '
			'embedding dimension'
		)

	return {
		'embedding_inputs': ordered_inputs,
		'embedding_compatibility_signature': _json_normalize(compatibility),
		'raw_feature_dimension': raw_feature_dimension,
		'normalization': normalization,
		'residualization': residualization,
		'pca': pca,
		'ordering': {
			'emission_source': 'embedding',
			'z_axis': 2,
			'z_direction': 'increasing_downward',
			'init': {'order_by': 'mean_z'},
			'edge_margin_tokens': list(edge_margin),
			'update': {'empty_cluster_policy': update['empty_cluster_policy']},
		},
		'prepared_feature_identity': {
			'feature_mode': 'embedding',
			'dtype': prepared.get('dtype'),
			'schema_version': prepared.get('schema_version'),
			'edge_margin_tokens': list(edge_margin),
			'surveys': prepared_identity,
		},
		'feature_dimension': feature_dimension,
	}


def _periodic_preprocessing_identity(
	value: object, label: str
) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'periodic refresh metadata {label} must be a mapping')
	enabled = value.get('enabled')
	if not isinstance(enabled, bool):
		raise TypeError(f'periodic refresh metadata {label}.enabled must be a boolean')
	identity: dict[str, object] = {'enabled': enabled}
	if enabled:
		for key in ('mode', 'group_by', 'add_global_mean_back', 'min_group_count'):
			if key not in value:
				raise ValueError(
					f'periodic refresh metadata {label} is missing {key}'
				)
		identity.update(
			{
				key: _json_normalize(value[key])
				for key in (
					'mode',
					'group_by',
					'add_global_mean_back',
					'min_group_count',
				)
			}
		)
	return identity


def _periodic_pca_identity(value: object) -> dict[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('periodic refresh model metadata pca must be a mapping')
	enabled = value.get('enabled')
	if not isinstance(enabled, bool):
		raise TypeError('periodic refresh model metadata pca.enabled must be a boolean')
	n_components = _periodic_positive_int(
		value.get('n_components'), 'periodic PCA n_components'
	)
	whiten = value.get('whiten')
	if not isinstance(whiten, bool):
		raise TypeError('periodic refresh model metadata pca.whiten must be a boolean')
	effective = value.get('effective_n_components')
	if effective is not None:
		effective = _periodic_positive_int(
			effective, 'periodic PCA effective_n_components'
	)
	if enabled and effective is None:
		raise ValueError(
			'periodic refresh model metadata must record enabled PCA output dimension'
	)
	if not enabled and effective is not None:
		raise ValueError(
			'periodic refresh model metadata must not record disabled PCA output '
			'dimension'
	)
	return {
		'enabled': enabled,
		'n_components': n_components,
		'effective_n_components': effective,
		'whiten': whiten,
	}


def _periodic_positive_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
		raise ValueError(f'{label} must be a positive integer')
	return value


def _periodic_nonnegative_int_triplet(
	value: object, label: str
) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
		or any(
			isinstance(item, bool) or not isinstance(item, int) or item < 0
			for item in value
		)
	):
		raise ValueError(
			f'{label} must be a length-three nonnegative integer sequence'
		)
	return (int(value[0]), int(value[1]), int(value[2]))


def _periodic_edge_margin_mask_for_shape(
	shape: tuple[int, int, int], edge_margin_tokens: tuple[int, int, int]
) -> np.ndarray:
	for size, margin in zip(shape, edge_margin_tokens, strict=True):
		if margin * 2 > size:
			raise ValueError('periodic edge margins do not fit token shape')
	mask = np.ones(shape, dtype=np.bool_)
	for axis, margin in enumerate(edge_margin_tokens):
		if margin == 0:
			continue
		leading = [slice(None)] * 3
		trailing = [slice(None)] * 3
		leading[axis] = slice(0, margin)
		trailing[axis] = slice(shape[axis] - margin, None)
		mask[tuple(leading)] = False
		mask[tuple(trailing)] = False
	return mask


def _validate_periodic_initial_artifact_identity(  # noqa: C901, PLR0912, PLR0915
	*,
	common_refs: Mapping[str, object],
	metadata_identity: Mapping[str, object],
	source_metadata_path: Path,
	source_metadata: Mapping[str, object],
	target_manifest: Mapping[str, object],
) -> None:
	"""Bind every initial-HMM artifact to one prepared feature identity."""
	input_identity = metadata_identity['embedding_inputs']
	if not isinstance(input_identity, list):
		raise TypeError('periodic refresh embedding input identity must be a list')
	source_embedding = target_manifest.get('source_embedding')
	if not isinstance(source_embedding, Mapping):
		raise TypeError(
			'pseudo_target_refresh target manifest source_embedding must be a mapping'
		)
	source_surveys = source_embedding.get('surveys')
	if not isinstance(source_surveys, Mapping):
		raise TypeError(
			'pseudo_target_refresh target manifest source embedding surveys must be '
			'a mapping'
		)
	input_survey_ids = {entry['survey_id'] for entry in input_identity}
	if set(source_surveys) != input_survey_ids:
		raise ValueError(
			'pseudo_target_refresh target manifest source embedding survey set does '
			'not match model inputs'
		)
	common_target = target_manifest.get('common')
	if not isinstance(common_target, Mapping):
		raise TypeError(
			'pseudo_target_refresh target manifest common must be a mapping'
		)
	common_valid_hashes = _periodic_valid_token_hashes(
		common_target.get('valid_tokens_sha256'),
		'pseudo_target_refresh target manifest common valid-token hashes',
	)
	if set(common_valid_hashes) != input_survey_ids:
		raise ValueError(
			'pseudo_target_refresh target manifest valid-mask survey set does not '
			'match model inputs'
		)
	metadata_compatibility = _required_mapping(
		metadata_identity, 'embedding_compatibility_signature'
	)
	raw_feature_dimension = _periodic_positive_int(
		metadata_compatibility.get('embedding_dim'),
		'periodic embedding compatibility embedding_dim',
	)
	source_survey_id = source_metadata.get('survey_id')
	if not isinstance(source_survey_id, str):
		raise TypeError(
			'pseudo_target_refresh source metadata must record survey_id'
		)
	matching_inputs = [
		entry for entry in input_identity if entry['survey_id'] == source_survey_id
	]
	if len(matching_inputs) != 1:
		raise ValueError(
			'pseudo_target_refresh source metadata survey is not in ordered '
			'model inputs'
		)
	if Path(matching_inputs[0]['metadata_path']).resolve() != (
		source_metadata_path.resolve()
	):
		raise ValueError(
			'pseudo_target_refresh source metadata path does not match model inputs'
		)
	if matching_inputs[0]['metadata_sha256'] != _file_sha256(
		str(source_metadata_path)
	):
		raise ValueError(
			'pseudo_target_refresh source metadata hash does not match model inputs'
		)
	for key in (
		'model_geometry',
		'patch_size',
		'window_size',
		'overlap',
		'min_token_valid_fraction',
		'zero_mask',
	):
		if key not in source_metadata:
			raise ValueError(
				f'pseudo_target_refresh source metadata is missing {key}'
			)
		if (
			key not in metadata_compatibility
			or _json_normalize(source_metadata[key]) != metadata_compatibility[key]
		):
			raise ValueError(
				'pseudo_target_refresh source embedding compatibility identity mismatch'
			)
	target_heads = target_manifest.get('heads')
	if not isinstance(target_heads, Mapping):
		raise TypeError(
			'pseudo_target_refresh target manifest heads must be a mapping'
		)
	common_head = target_heads.get('6')
	if not isinstance(common_head, Mapping):
		raise TypeError(
			'pseudo_target_refresh target manifest must contain a K=6 head'
		)
	common_head_surveys = common_head.get('surveys')
	if not isinstance(common_head_surveys, Mapping):
		raise TypeError(
			'pseudo_target_refresh target manifest K=6 surveys must be a mapping'
		)
	edge_margin = _periodic_nonnegative_int_triplet(
		metadata_identity['ordering']['edge_margin_tokens'],
		'periodic edge margin tokens',
	)
	for entry in input_identity:
		survey_id = entry['survey_id']
		target_entry = source_surveys.get(survey_id)
		if not isinstance(target_entry, Mapping):
			raise TypeError(
				'pseudo_target_refresh target manifest source embedding survey entry '
				'must be a mapping'
			)
		for input_key, target_key, hash_key in (
			('embeddings_path', 'embedding_path', 'embedding_sha256'),
			('valid_tokens_path', 'valid_tokens_path', 'valid_tokens_sha256'),
			('metadata_path', 'metadata_path', 'metadata_sha256'),
		):
			input_path = entry[input_key]
			target_path = target_entry.get(target_key)
			if not isinstance(target_path, str) or not target_path:
				raise ValueError(
					f'pseudo_target_refresh target manifest is missing {target_key} '
					f'for {survey_id}'
				)
			if Path(str(input_path)).resolve() != Path(target_path).resolve():
				raise ValueError(
					f'pseudo_target_refresh {input_key} identity mismatch for '
					f'{survey_id}'
				)
			digest = target_entry.get(hash_key)
			if not isinstance(digest, str) or _file_sha256(target_path) != digest:
				raise ValueError(
					f'pseudo_target_refresh target manifest {hash_key} mismatch '
					f'for {survey_id}'
				)
		target_valid_hash = target_entry.get('valid_tokens_sha256')
		if target_valid_hash != _file_sha256(str(target_entry['valid_tokens_path'])):
			raise ValueError(
				f'pseudo_target_refresh source embedding valid-mask hash mismatch '
				f'{survey_id}'
			)
		common_entry = common_head_surveys.get(survey_id)
		if not isinstance(common_entry, Mapping):
			raise TypeError(
				f'pseudo_target_refresh target manifest is missing K=6 survey '
				f'{survey_id}'
			)
		common_valid_reference = common_entry.get('valid_tokens')
		if not isinstance(common_valid_reference, Mapping):
			raise TypeError(
				f'pseudo_target_refresh K=6 valid-token reference is invalid for '
				f'{survey_id}'
			)
		common_valid_path = common_valid_reference.get('path')
		if not isinstance(common_valid_path, str) or not common_valid_path:
			raise ValueError(
				f'pseudo_target_refresh K=6 valid-token path is missing for '
				f'{survey_id}'
			)
		if _file_sha256(common_valid_path) != common_valid_hashes[survey_id]:
			raise ValueError(
				f'pseudo_target_refresh common valid-token hash mismatch for '
				f'{survey_id}'
			)
		source_valid_tokens = np.asarray(
			np.load(entry['valid_tokens_path'], mmap_mode='r', allow_pickle=False),
			dtype=np.bool_,
		)
		common_valid_tokens = np.asarray(
			np.load(common_valid_path, mmap_mode='r', allow_pickle=False),
			dtype=np.bool_,
		)
		if source_valid_tokens.shape != common_valid_tokens.shape:
			raise ValueError(
				f'pseudo_target_refresh source/common valid-mask shape mismatch '
				f'for {survey_id}'
			)
		effective_valid_tokens = np.logical_and(
			source_valid_tokens,
			_periodic_edge_margin_mask_for_shape(
				source_valid_tokens.shape, edge_margin
			),
		)
		if not np.array_equal(effective_valid_tokens, common_valid_tokens):
			raise ValueError(
				f'pseudo_target_refresh source valid-mask/edge-margin identity '
				f'mismatch for {survey_id}'
			)
		embeddings = np.load(
			entry['embeddings_path'], mmap_mode='r', allow_pickle=False
		)
		valid_tokens = np.load(
			entry['valid_tokens_path'], mmap_mode='r', allow_pickle=False
		)
		if embeddings.ndim != 4 or embeddings.shape[-1] != raw_feature_dimension:
			raise ValueError(
				f'pseudo_target_refresh source embedding feature dimension mismatch '
				f'for {survey_id}'
			)
		if valid_tokens.shape != embeddings.shape[:3]:
			raise ValueError(
				'pseudo_target_refresh source valid-mask shape mismatch for '
				f'{survey_id}'
			)
		del embeddings, valid_tokens

	_validate_periodic_common_config_binding(
		Path(str(_required_mapping(common_refs, 'clustering_config')['path'])),
		metadata_identity,
	)
	_validate_periodic_preprocessing_artifacts(common_refs, metadata_identity)


def _validate_periodic_common_config_binding(  # noqa: C901, PLR0912
	path: Path, metadata_identity: Mapping[str, object]
) -> None:
	try:
		loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
	except (OSError, yaml.YAMLError) as exc:
		raise ValueError(
			f'pseudo_target_refresh clustering_config must be valid YAML: {path}'
		) from exc
	if not isinstance(loaded, Mapping):
		raise TypeError('pseudo_target_refresh clustering_config must be a mapping')
	clustering = loaded.get('clustering')
	if not isinstance(clustering, Mapping):
		raise TypeError(
			'pseudo_target_refresh clustering_config.clustering must be a mapping'
		)
	if clustering.get('method') != 'stratigraphic_hmm_kmeans':
		raise ValueError(
			'pseudo_target_refresh clustering_config must select '
			'stratigraphic_hmm_kmeans'
	)
	if tuple(clustering.get('k_values', ())) != (6, 8, 10):
		raise ValueError(
		'pseudo_target_refresh clustering_config k_values must be [6, 8, 10]'
	)
	if clustering.get('embedding_normalization') != metadata_identity['normalization']:
		raise ValueError(
		'pseudo_target_refresh normalization differs between clustering_config and '
		'model metadata'
	)
	config_residualization = clustering.get('residualization')
	if not isinstance(config_residualization, Mapping):
		raise TypeError(
			'pseudo_target_refresh clustering_config residualization must be a mapping'
		)
	metadata_residualization = _required_mapping(
		metadata_identity, 'residualization'
	)
	if config_residualization.get('enabled') != metadata_residualization.get('enabled'):
		raise ValueError(
		'pseudo_target_refresh residualization identity differs between common '
		'artifacts'
	)
	if config_residualization.get('enabled') is True:
		for key in ('mode', 'group_by', 'add_global_mean_back', 'min_group_count'):
			if config_residualization.get(key) != metadata_residualization.get(key):
				raise ValueError(
					f'pseudo_target_refresh residualization.{key} identity mismatch'
				)
	config_pca = clustering.get('pca')
	if not isinstance(config_pca, Mapping):
		raise TypeError('pseudo_target_refresh clustering_config pca must be a mapping')
	metadata_pca = _required_mapping(metadata_identity, 'pca')
	for key in ('enabled', 'n_components', 'whiten'):
		if config_pca.get(key) != metadata_pca.get(key):
			raise ValueError(
				f'pseudo_target_refresh pca.{key} identity mismatch across common '
				'artifacts'
			)
	hmm = clustering.get('stratigraphic_hmm')
	if not isinstance(hmm, Mapping):
		raise TypeError(
			'pseudo_target_refresh clustering_config stratigraphic_hmm must be '
			'a mapping'
		)
	ordering = _required_mapping(metadata_identity, 'ordering')
	if hmm.get('emission_source') != ordering['emission_source']:
		raise ValueError('pseudo_target_refresh emission ordering identity mismatch')
	for key in ('z_axis', 'z_direction'):
		if hmm.get(key) != ordering[key]:
			raise ValueError(
				f'pseudo_target_refresh ordering {key} identity mismatch'
			)
	for section, key in (('init', 'order_by'), ('update', 'empty_cluster_policy')):
		config_section = hmm.get(section)
		if not isinstance(config_section, Mapping) or config_section.get(key) != (
			_required_mapping(ordering, section)[key]
		):
			raise ValueError(
				f'pseudo_target_refresh ordering {section}.{key} identity mismatch'
			)
	config_edge_margin = hmm.get('edge_margin_tokens', [0, 0, 0])
	if _periodic_nonnegative_int_triplet(
		config_edge_margin, 'clustering_config edge_margin_tokens'
	) != tuple(ordering['edge_margin_tokens']):
		raise ValueError(
			'pseudo_target_refresh ordering edge_margin_tokens identity mismatch'
	)


def _validate_periodic_preprocessing_artifacts(  # noqa: C901, PLR0912, PLR0915
	common_refs: Mapping[str, object], metadata_identity: Mapping[str, object]
) -> None:
	"""Verify the shared fitted preprocessing artifacts produce this identity."""
	raw_dimension = _periodic_positive_int(
		metadata_identity.get('raw_feature_dimension'),
		'periodic raw feature dimension',
	)
	feature_dimension = _periodic_positive_int(
		metadata_identity.get('feature_dimension'),
		'periodic prepared feature dimension',
	)
	preprocessor_ref = _required_mapping(common_refs, 'preprocessor')
	preprocessor_path = Path(str(preprocessor_ref['path']))
	try:
		joblib_module = importlib.import_module('joblib')
		preprocessor = joblib_module.load(preprocessor_path)
	except (OSError, EOFError, ValueError, TypeError, ImportError) as exc:
		raise ValueError(
			f'pseudo_target_refresh preprocessor cannot be loaded: {preprocessor_path}'
		) from exc
	steps = getattr(preprocessor, 'named_steps', None)
	if not isinstance(steps, Mapping):
		raise TypeError(
			'pseudo_target_refresh preprocessor must expose named_steps identity'
		)
	normalization = metadata_identity['normalization']
	if normalization == 'l2':
		if 'normalizer' not in steps or 'identity' in steps:
			raise ValueError(
				'pseudo_target_refresh preprocessor normalization identity mismatch'
			)
	elif 'identity' not in steps or 'normalizer' in steps:
		raise ValueError(
			'pseudo_target_refresh preprocessor normalization identity mismatch'
		)
	metadata_pca = _required_mapping(metadata_identity, 'pca')
	pca_step = steps.get('pca')
	if metadata_pca['enabled']:
		if pca_step is None:
			raise ValueError(
				'pseudo_target_refresh preprocessor PCA identity is missing'
			)
		if getattr(pca_step, 'n_components', None) != metadata_pca['n_components']:
			raise ValueError(
				'pseudo_target_refresh preprocessor PCA component identity mismatch'
			)
		if bool(getattr(pca_step, 'whiten', False)) != metadata_pca['whiten']:
			raise ValueError(
				'pseudo_target_refresh preprocessor PCA whiten identity mismatch'
			)
	elif pca_step is not None:
		raise ValueError(
			'pseudo_target_refresh disabled PCA is present in preprocessor'
		)
	try:
		probe = np.zeros((1, raw_dimension), dtype=np.float32)
		prepared = np.asarray(preprocessor.transform(probe))
	except (AttributeError, TypeError, ValueError) as exc:
		raise ValueError(
			'pseudo_target_refresh preprocessor cannot transform the source feature '
			'dimension'
		) from exc
	if prepared.shape != (1, feature_dimension) or not np.isfinite(prepared).all():
		raise ValueError(
			'pseudo_target_refresh preprocessor output dimension does not match '
			'centers'
		)

	metadata_residualization = _required_mapping(
		metadata_identity, 'residualization'
	)
	residualizer_ref = common_refs.get('residualizer')
	if (residualizer_ref is None) != (metadata_residualization['enabled'] is False):
		raise ValueError(
			'pseudo_target_refresh residualizer presence does not match preprocessing '
			'identity'
	)
	if residualizer_ref is None:
		return
	if not isinstance(residualizer_ref, Mapping):
		raise TypeError(
			'pseudo_target_refresh residualizer reference must be a mapping'
		)
	residualizer_path_value = residualizer_ref.get('path')
	if not isinstance(residualizer_path_value, str) or not residualizer_path_value:
		raise ValueError(
			'pseudo_target_refresh residualizer reference path is missing'
		)
	residualizer_path = Path(residualizer_path_value)
	try:
		residualizer = read_residualizer_npz(residualizer_path)
	except (OSError, KeyError, TypeError, ValueError) as exc:
		raise ValueError(
			f'pseudo_target_refresh residualizer cannot be loaded: {residualizer_path}'
		) from exc
	means = np.asarray(residualizer.means)
	if means.ndim != 2 or means.shape[1] != raw_dimension:
		raise ValueError(
			'pseudo_target_refresh residualizer feature dimension does not match '
			'source embeddings'
	)
	for key in ('mode', 'group_by', 'add_global_mean_back', 'min_group_count'):
		if getattr(residualizer, key) != metadata_residualization[key]:
			raise ValueError(
				f'pseudo_target_refresh residualizer {key} identity mismatch'
			)


def _periodic_valid_token_hashes(
	value: object, label: str
) -> dict[str, str]:
	if not isinstance(value, Mapping) or not value:
		raise ValueError(f'{label} must be a non-empty mapping')
	result: dict[str, str] = {}
	for survey_id, digest in value.items():
		if not isinstance(survey_id, str) or not survey_id:
			raise ValueError(f'{label} has an invalid survey_id')
		if (
			not isinstance(digest, str)
			or len(digest) != 64
			or any(character not in '0123456789abcdef' for character in digest)
		):
			raise ValueError(f'{label} is invalid for {survey_id!r}')
		result[survey_id] = digest
	return result


def _source_target_roots_must_be_immutable(
	manifest: Mapping[str, object], generation_root: Path, label: str
) -> None:
	heads = manifest.get('heads')
	if not isinstance(heads, Mapping):
		return
	for key, entry in heads.items():
		if not isinstance(entry, Mapping):
			continue
		root = entry.get('pseudo_target_root')
		if not isinstance(root, str) or not root:
			continue
		_validate_initial_artifact_not_in_generation_root(
			Path(root), generation_root, f'{label}.heads.{key}.pseudo_target_root'
		)


def _json_normalize(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _json_normalize(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_json_normalize(child) for child in value]
	if isinstance(value, np.generic):
		return _json_normalize(value.item())
	if value is None or isinstance(value, str | int | float | bool):
		return value
	return str(value)


def _validate_strat_hmm_pretext_sections(config: Mapping[str, object]) -> None:
	for section, allowed in _STRAT_HMM_PRETEXT_SECTION_KEYS.items():
		value = config.get(section)
		if not isinstance(value, Mapping):
			if section == 'spatial_context' and section in config:
				raise TypeError('spatial_context must be a mapping')
			continue
		_validate_allowed_keys(value, allowed, prefix=section)


def _validate_strat_hmm_pretext_identity(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
	config: Mapping[str, object],
	*,
	multi_head: bool,
	manifest_sha256: str | None = None,
	manifest: Mapping[str, object] | None = None,
	target_representation: str | None = None,
	periodic_refresh_identity: Mapping[str, object] | None = None,
) -> None:
	"""Validate optional provenance that is stored beside model weights."""
	value = config.get('identity')
	if value is None:
		if multi_head:
			raise ValueError('identity is required for multi-head strat HMM pretext')
		return
	if not isinstance(value, Mapping):
		raise TypeError('identity must be a mapping')
	_validate_allowed_keys(
		value,
		frozenset({'model_tag', 'scientific_identity', 'runtime_identity'}),
		prefix='identity',
	)
	model_tag = value.get('model_tag')
	if not isinstance(model_tag, str) or not model_tag:
		raise TypeError('identity.model_tag must be a non-empty string')
	for key in ('scientific_identity', 'runtime_identity'):
		child = value.get(key)
		if child is not None and not isinstance(child, Mapping):
			raise TypeError(f'identity.{key} must be a mapping when provided')
	if not multi_head:
		return
	scientific = _required_child_mapping(
		value,
		'scientific_identity',
		prefix='identity',
	)
	if manifest is None or manifest_sha256 is None:
		raise AssertionError('multi-head identity validation requires a manifest')
	if not isinstance(scientific, dict):
		raise TypeError('identity.scientific_identity must be a mutable mapping')
	if periodic_refresh_identity is not None:
		_validate_periodic_refresh_scientific_identity(
			scientific,
			config=config,
			manifest=manifest,
			manifest_sha256=manifest_sha256,
			periodic_refresh_identity=periodic_refresh_identity,
		)
		runtime = value.get('runtime_identity')
		if runtime is not None:
			_validate_allowed_keys(
				runtime,
				MULTI_HEAD_RUNTIME_IDENTITY_FIELDS,
				prefix='identity.runtime_identity',
			)
		return
	if _is_center_trace_masked_config(config):
		for key in (
			'experiment_role',
			'variant',
			'head_spec',
			'head_ks',
			'target_representation',
			'target_manifest_sha256',
			'target_head_hashes',
			'objective_semantics',
			'mask_semantics',
			'column_fraction',
			'selection_policy',
			'replacement',
			'replacement_initialization',
			'rng_policy',
			'masked_prototype_weight',
			'visible_prototype_weight',
			'distillation_scope',
			'supervised_loss',
			'consistency_policy',
		):
			_validate_required_key(
				scientific, key, prefix='identity.scientific_identity'
			)
		_expected_or_record_multi_head_scientific_identity(
			scientific,
			config=config,
			manifest=manifest,
			target_representation=target_representation,
		)
		_validate_allowed_keys(
			scientific,
			_CENTER_TRACE_SCIENTIFIC_IDENTITY_FIELDS,
			prefix='identity.scientific_identity',
		)
		_validate_center_trace_scientific_identity(
			scientific,
			model_tag=model_tag,
			manifest_sha256=manifest_sha256,
			manifest=manifest,
			loss=_required_mapping(config, 'loss'),
		)
		runtime = value.get('runtime_identity')
		if runtime is not None:
			_validate_allowed_keys(
				runtime,
				MULTI_HEAD_RUNTIME_IDENTITY_FIELDS,
				prefix='identity.runtime_identity',
			)
		return
	if target_representation == _STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION:
		for key in (
			'experiment_role',
			'variant',
			'target_representation',
			'posterior_manifest_sha256',
			'posterior_semantics',
			'posterior_cost_temperature',
			'posterior_head_hashes',
			'supervised_loss',
			'head_spec',
			'head_ks',
			'consistency_policy',
			'consistency_weight',
		):
			_validate_required_key(
				scientific, key, prefix='identity.scientific_identity'
			)
	_expected_or_record_multi_head_scientific_identity(
		scientific,
		config=config,
		manifest=manifest,
		target_representation=target_representation,
	)
	_validate_allowed_keys(
		scientific,
		_multi_head_scientific_identity_fields(target_representation),
		prefix='identity.scientific_identity',
	)
	if target_representation == _STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION:
		_validate_m5_u_scientific_identity(
			scientific,
			model_tag=model_tag,
			manifest_sha256=manifest_sha256,
			manifest=manifest,
			loss=_required_mapping(config, 'loss'),
		)
		runtime = value.get('runtime_identity')
		if runtime is not None:
			_validate_allowed_keys(
				runtime,
				MULTI_HEAD_RUNTIME_IDENTITY_FIELDS,
				prefix='identity.runtime_identity',
			)
		return
	if target_representation == _STRAT_HMM_LATERAL_TARGET_REPRESENTATION:
		for key in (
			'experiment_role',
			'variant',
			'target_representation',
			'target_semantics',
			'lateral_target_manifest_sha256',
			'lateral_target_head_hashes',
			'source_hard_manifest_sha256',
			'source_posterior_manifest_sha256',
			'lateral_smoothing',
			'supervised_loss',
			'head_spec',
			'head_ks',
			'consistency_policy',
			'consistency_weight',
		):
			_validate_required_key(
				scientific, key, prefix='identity.scientific_identity'
			)
		_validate_m5_ls_scientific_identity(
			scientific,
			model_tag=model_tag,
			manifest_sha256=manifest_sha256,
			manifest=manifest,
			loss=_required_mapping(config, 'loss'),
		)
		runtime = value.get('runtime_identity')
		if runtime is not None:
			_validate_allowed_keys(
				runtime,
				MULTI_HEAD_RUNTIME_IDENTITY_FIELDS,
				prefix='identity.runtime_identity',
			)
		return
	if target_representation == _STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION:
		for key in (
			'experiment_role',
			'variant',
			'target_representation',
			'target_semantics',
			'xy_neighbor_consensus_target_manifest_sha256',
			'xy_neighbor_consensus_target_head_hashes',
			'source_hard_manifest_sha256',
			'xy_neighbor_consensus_smoothing',
			'supervised_loss',
			'head_spec',
			'head_ks',
			'consistency_policy',
			'consistency_weight',
		):
			_validate_required_key(
				scientific, key, prefix='identity.scientific_identity'
			)
		_validate_allowed_keys(
			scientific,
			_XY_NEIGHBOR_CONSENSUS_SCIENTIFIC_IDENTITY_FIELDS,
			prefix='identity.scientific_identity',
		)
		_validate_xy_neighbor_consensus_scientific_identity(
			scientific,
			model_tag=model_tag,
			manifest_sha256=manifest_sha256,
			manifest=manifest,
			loss=_required_mapping(config, 'loss'),
		)
		runtime = value.get('runtime_identity')
		if runtime is not None:
			_validate_allowed_keys(
				runtime,
				MULTI_HEAD_RUNTIME_IDENTITY_FIELDS,
				prefix='identity.runtime_identity',
			)
		return
	if target_representation == _STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION:
		for key in (
			'experiment_role',
			'variant',
			'target_representation',
			'target_semantics',
			'xy_neighbor_unanimous_target_manifest_sha256',
			'xy_neighbor_unanimous_target_head_hashes',
			'source_hard_manifest_sha256',
			'xy_neighbor_unanimous_smoothing',
			'supervised_loss',
			'head_spec',
			'head_ks',
			'consistency_policy',
			'consistency_weight',
		):
			_validate_required_key(
				scientific, key, prefix='identity.scientific_identity'
			)
		_validate_allowed_keys(
			scientific,
			_XY_NEIGHBOR_UNANIMOUS_SCIENTIFIC_IDENTITY_FIELDS,
			prefix='identity.scientific_identity',
		)
		_validate_xy_neighbor_unanimous_scientific_identity(
			scientific,
			model_tag=model_tag,
			manifest_sha256=manifest_sha256,
			manifest=manifest,
			loss=_required_mapping(config, 'loss'),
		)
		runtime = value.get('runtime_identity')
		if runtime is not None:
			_validate_allowed_keys(
				runtime,
				MULTI_HEAD_RUNTIME_IDENTITY_FIELDS,
				prefix='identity.runtime_identity',
			)
		return
	for key in (
		'experiment_role',
		'variant',
		'head_spec',
		'head_ks',
		'target_manifest_sha256',
		'consistency_policy',
	):
		_validate_required_key(scientific, key, prefix='identity.scientific_identity')
	if scientific['experiment_role'] != 'multi_head_ordered_pretext':
		raise ValueError(
			'identity.scientific_identity.experiment_role must be '
			"'multi_head_ordered_pretext'"
		)
	if scientific['variant'] not in {'nocons', 'cons010'}:
		raise ValueError(
			'identity.scientific_identity.variant must be "nocons" or "cons010"'
		)
	expected_consistency_weight = {
		'nocons': 0.0,
		'cons010': 0.1,
	}[scientific['variant']]
	loss = _required_mapping(config, 'loss')
	if loss['consistency_weight'] != expected_consistency_weight:
		raise ValueError(
			'loss.consistency_weight does not match '
			'identity.scientific_identity.variant'
		)
	if scientific['consistency_weight'] != expected_consistency_weight:
		raise ValueError(
			'identity.scientific_identity.consistency_weight does not match '
			'identity.scientific_identity.variant'
		)
	expected_tag = {
		'nocons': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		'cons010': 'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
	}[scientific['variant']]
	if model_tag != expected_tag:
		raise ValueError(
			'identity.model_tag does not match identity.scientific_identity.variant'
		)
	if scientific['head_spec'] != _STRAT_HMM_MULTI_HEAD_SPEC:
		raise ValueError(
			'identity.scientific_identity.head_spec does not match head.spec'
		)
	_validate_head_ks(scientific['head_ks'], prefix='identity.scientific_identity')
	if tuple(scientific['head_ks']) != tuple(manifest['head_ks']):
		raise ValueError('identity.scientific_identity.head_ks does not match manifest')
	if scientific['target_manifest_sha256'] != manifest_sha256:
		raise ValueError(
			'identity.scientific_identity.target_manifest_sha256 does not match '
			'the manifest file'
		)
	if scientific['consistency_policy'] != _STRAT_HMM_MULTI_HEAD_CONSISTENCY_POLICY:
		raise ValueError(
			'identity.scientific_identity.consistency_policy must be '
			f'{_STRAT_HMM_MULTI_HEAD_CONSISTENCY_POLICY!r}'
		)
	if (
		'target_representation' in scientific
		and scientific['target_representation'] != _STRAT_HMM_HARD_TARGET_REPRESENTATION
	):
		raise ValueError(
			'identity.scientific_identity.target_representation must be '
			f'{_STRAT_HMM_HARD_TARGET_REPRESENTATION!r} when recorded'
		)
	runtime = value.get('runtime_identity')
	if runtime is not None:
		_validate_allowed_keys(
			runtime,
			MULTI_HEAD_RUNTIME_IDENTITY_FIELDS,
			prefix='identity.runtime_identity',
		)


def _expected_or_record_multi_head_scientific_identity(  # noqa: C901
	scientific: dict[str, object],
	*,
	config: Mapping[str, object],
	manifest: Mapping[str, object],
	target_representation: str | None,
	periodic_refresh_identity: Mapping[str, object] | None = None,
) -> None:
	"""Bind resolved scientific settings into a multi-head identity."""
	head = _required_mapping(config, 'head')
	loss = _required_mapping(config, 'loss')
	teacher = _required_mapping(config, 'teacher')
	student = _required_mapping(config, 'student')
	model = _required_mapping(config, 'model')
	data = _required_mapping(config, 'data')
	zero_mask = _required_mapping(config, 'zero_mask')
	train = _required_mapping(config, 'train')
	expected = {
		'head_projection_dim': head['projection_dim'],
		'head_temperature': head['temperature'],
		'head_normalize': head['normalize'],
		'prototype_weight': loss['prototype_weight'],
		'usage_weight': loss['usage_weight'],
		'consistency_weight': loss['consistency_weight'],
		'consistency_beta': loss['consistency_beta'],
		'distillation_weight': loss['distillation_weight'],
		'teacher_checkpoint': teacher['checkpoint'],
		'student_init_checkpoint': student['init_checkpoint'],
		'student_unfreeze_top_blocks': student['unfreeze_top_blocks'],
		'model': {**FIXED_MODEL_CONTRACT, **model},
		'data': {**FIXED_DATA_CONTRACT, **data},
		'zero_mask': zero_mask,
		'train': {key: train[key] for key in MULTI_HEAD_SCIENTIFIC_TRAIN_FIELDS},
	}
	if _is_periodic_refresh_config(config):
		spatial_context = _required_mapping(config, 'spatial_context')
		periodic = periodic_refresh_identity
		if periodic is None:
			raise AssertionError(
				'periodic identity requires validated refresh artifacts'
			)
		expected.update(
			{
				'experiment_role': PERIODIC_REFRESH_EXPERIMENT_ROLE,
				'variant': PERIODIC_REFRESH_VARIANT,
				'head_spec': head['spec'],
				'head_ks': list(head['ks']),
				'target_representation': _STRAT_HMM_HARD_TARGET_REPRESENTATION,
				'target_manifest_sha256': _file_sha256(
					str(_required_mapping(config, 'pseudo_targets')['manifest'])
				),
				'objective_semantics': spatial_context['objective'],
				'mask_semantics': spatial_context['mask_semantics'],
				'column_fraction': spatial_context['column_fraction'],
				'selection_policy': spatial_context['selection_policy'],
				'replacement': spatial_context['replacement'],
				'replacement_initialization': spatial_context[
					'replacement_initialization'
				],
				'rng_policy': spatial_context['rng_policy'],
				'masked_prototype_weight': spatial_context[
					'masked_prototype_weight'
				],
				'visible_prototype_weight': spatial_context[
					'visible_prototype_weight'
				],
				'distillation_scope': spatial_context['distillation_scope'],
				'supervised_loss': CENTER_TRACE_SUPERVISED_LOSS,
				'consistency_policy': CENTER_TRACE_CONSISTENCY_POLICY,
				'target_head_hashes': _multi_head_target_hashes(manifest),
				'model_role': PERIODIC_REFRESH_MODEL_ROLE,
				'target_refresh_semantics': PERIODIC_REFRESH_SEMANTICS,
				'refresh_schedule_semantics': PERIODIC_REFRESH_SCHEDULE_SEMANTICS,
				'refresh_after_epochs': list(PERIODIC_REFRESH_SCHEDULE),
				'hmm_iterations_per_refresh': 2,
				'embedding_source': 'current_student',
				'embedding_mode': 'unmasked_eval_full_survey',
				'refresh_embedding_semantics': PERIODIC_REFRESH_EMBEDDING_SEMANTICS,
				'center_initialization': 'previous_generation',
				'center_update': 'full_mean',
				'center_update_semantics': PERIODIC_REFRESH_CENTER_UPDATE_SEMANTICS,
				'preprocessing_policy': PERIODIC_REFRESH_PREPROCESSING_POLICY,
				'target_activation_policy': PERIODIC_REFRESH_TARGET_ACTIVATION_POLICY,
				'empty_state_policy': 'error',
				'checkpoint_selection_policy': (
					PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY
				),
				'initial_hard_target_manifest_sha256': periodic[
					'initial_hard_target_manifest'
				]['sha256'],
				'initial_hmm_artifacts': periodic['initial_hmm_artifacts'],
				'fixed_preprocessor_sha256': periodic['fixed_preprocessor_sha256'],
				'fixed_residualizer_sha256': periodic['fixed_residualizer_sha256'],
				'fixed_clustering_config_sha256': periodic[
					'fixed_clustering_config_sha256'
				],
				'source_embedding_metadata_sha256': periodic[
					'source_embedding_metadata_sha256'
				],
				'source_valid_token_hashes': periodic['source_valid_token_hashes'],
				'feature_dimension': periodic['feature_dimension'],
				'generation_root': periodic['generation_root'],
			}
		)
	elif _is_center_trace_masked_config(config):
		spatial_context = _required_mapping(config, 'spatial_context')
		expected.update(
			{
				'head_spec': head['spec'],
				'head_ks': list(head['ks']),
				'target_representation': _STRAT_HMM_HARD_TARGET_REPRESENTATION,
				'target_manifest_sha256': _file_sha256(
					str(_required_mapping(config, 'pseudo_targets')['manifest'])
				),
				'objective_semantics': spatial_context['objective'],
				'mask_semantics': spatial_context['mask_semantics'],
				'column_fraction': spatial_context['column_fraction'],
				'selection_policy': spatial_context['selection_policy'],
				'replacement': spatial_context['replacement'],
				'replacement_initialization': spatial_context[
					'replacement_initialization'
				],
				'rng_policy': spatial_context['rng_policy'],
				'masked_prototype_weight': spatial_context[
					'masked_prototype_weight'
				],
				'visible_prototype_weight': spatial_context[
					'visible_prototype_weight'
				],
				'distillation_scope': spatial_context['distillation_scope'],
				'supervised_loss': CENTER_TRACE_SUPERVISED_LOSS,
			}
		)
		expected['target_head_hashes'] = _multi_head_target_hashes(manifest)
	if target_representation == _STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION:
		expected.update(
			{
				'target_representation': target_representation,
				'posterior_manifest_sha256': _file_sha256(
					str(_required_mapping(config, 'pseudo_targets')['manifest'])
				),
				'posterior_semantics': manifest['posterior_semantics'],
				'posterior_cost_temperature': manifest['cost_temperature'],
				'posterior_head_hashes': _multi_head_posterior_hashes(manifest),
				'supervised_loss': 'soft_categorical_cross_entropy_v1',
			}
		)
	elif target_representation not in {
		_STRAT_HMM_LATERAL_TARGET_REPRESENTATION,
		_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION,
		_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION,
	}:
		expected['target_head_hashes'] = _multi_head_target_hashes(manifest)
	if target_representation == _STRAT_HMM_LATERAL_TARGET_REPRESENTATION:
		pseudo_targets = _required_mapping(config, 'pseudo_targets')
		expected.update(
			{
				'target_representation': target_representation,
				'target_semantics': manifest['target_semantics'],
				'lateral_target_manifest_sha256': _file_sha256(
					str(pseudo_targets['manifest'])
				),
				'lateral_target_head_hashes': _multi_head_target_hashes(manifest),
				'source_hard_manifest_sha256': _manifest_reference_sha256(
					manifest, 'source_hard_manifest'
				),
				'source_posterior_manifest_sha256': _manifest_reference_sha256(
					manifest, 'source_posterior_manifest'
				),
				'lateral_smoothing': _lateral_smoothing_identity(manifest),
				'supervised_loss': 'structured_hmm_hard_categorical_v1',
			}
		)
	if target_representation == _STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION:
		pseudo_targets = _required_mapping(config, 'pseudo_targets')
		expected.update(
			{
				'target_representation': target_representation,
				'target_semantics': manifest['target_semantics'],
				'xy_neighbor_consensus_target_manifest_sha256': _file_sha256(
					str(pseudo_targets['manifest'])
				),
				'xy_neighbor_consensus_target_head_hashes': (
					_multi_head_target_hashes(manifest)
				),
				'source_hard_manifest_sha256': _manifest_reference_sha256(
					manifest, 'source_hard_manifest'
				),
				'xy_neighbor_consensus_smoothing': (
					_xy_neighbor_consensus_smoothing_identity(manifest)
				),
				'supervised_loss': 'structured_hmm_hard_categorical_v1',
			}
		)
	if target_representation == _STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION:
		pseudo_targets = _required_mapping(config, 'pseudo_targets')
		expected.update(
			{
				'target_representation': target_representation,
				'target_semantics': manifest['target_semantics'],
				'xy_neighbor_unanimous_target_manifest_sha256': _file_sha256(
					str(pseudo_targets['manifest'])
				),
				'xy_neighbor_unanimous_target_head_hashes': (
					_multi_head_target_hashes(manifest)
				),
				'source_hard_manifest_sha256': _manifest_reference_sha256(
					manifest, 'source_hard_manifest'
				),
				'xy_neighbor_unanimous_smoothing': (
					_xy_neighbor_unanimous_smoothing_identity(manifest)
				),
				'supervised_loss': 'structured_hmm_hard_categorical_v1',
			}
		)
	for key, expected_value in expected.items():
		if key in scientific:
			if scientific[key] != expected_value:
				raise ValueError(
					f'identity.scientific_identity.{key} does not match the '
					'resolved scientific setting'
				)
		else:
			scientific[key] = deepcopy(expected_value)


def _validate_periodic_refresh_scientific_identity(  # noqa: C901
	scientific: dict[str, object],
	*,
	config: Mapping[str, object],
	manifest: Mapping[str, object] | None,
	manifest_sha256: str | None,
	periodic_refresh_identity: Mapping[str, object],
) -> None:
	if manifest is None or manifest_sha256 is None:
		raise AssertionError('periodic identity validation requires a target manifest')
	_expected_or_record_multi_head_scientific_identity(
		scientific,
		config=config,
		manifest=manifest,
		target_representation=_STRAT_HMM_HARD_TARGET_REPRESENTATION,
		periodic_refresh_identity=periodic_refresh_identity,
	)
	for key in (
		'experiment_role',
		'variant',
		'model_role',
		'head_spec',
		'head_ks',
		'target_representation',
		'target_manifest_sha256',
		'target_head_hashes',
		'objective_semantics',
		'mask_semantics',
		'column_fraction',
		'selection_policy',
		'replacement',
		'replacement_initialization',
		'rng_policy',
		'masked_prototype_weight',
		'visible_prototype_weight',
		'distillation_scope',
		'supervised_loss',
		'consistency_policy',
		'target_refresh_semantics',
		'refresh_schedule_semantics',
		'refresh_after_epochs',
		'hmm_iterations_per_refresh',
		'embedding_source',
		'embedding_mode',
		'refresh_embedding_semantics',
		'center_initialization',
		'center_update',
		'center_update_semantics',
		'preprocessing_policy',
		'target_activation_policy',
		'empty_state_policy',
		'checkpoint_selection_policy',
		'initial_hard_target_manifest_sha256',
		'initial_hmm_artifacts',
		'fixed_preprocessor_sha256',
		'fixed_residualizer_sha256',
		'fixed_clustering_config_sha256',
		'source_embedding_metadata_sha256',
		'source_valid_token_hashes',
		'feature_dimension',
		'generation_root',
	):
		_validate_required_key(scientific, key, prefix='identity.scientific_identity')
	_validate_allowed_keys(
		scientific,
		_PERIODIC_REFRESH_SCIENTIFIC_IDENTITY_FIELDS,
		prefix='identity.scientific_identity',
	)
	identity = _required_mapping(config, 'identity')
	if identity.get('model_tag') != PERIODIC_REFRESH_MODEL_TAG:
		raise ValueError(
			'identity.model_tag does not match the periodic center-trace '
			'refresh contract'
		)
	checks = {
		'experiment_role': PERIODIC_REFRESH_EXPERIMENT_ROLE,
		'variant': PERIODIC_REFRESH_VARIANT,
		'head_spec': PERIODIC_REFRESH_HEAD_SPEC,
		'head_ks': [6, 8, 10],
		'target_representation': PERIODIC_REFRESH_TARGET_REPRESENTATION,
		'target_manifest_sha256': manifest_sha256,
		'target_head_hashes': _multi_head_target_hashes(manifest),
		'model_role': PERIODIC_REFRESH_MODEL_ROLE,
		'target_refresh_semantics': PERIODIC_REFRESH_SEMANTICS,
		'refresh_schedule_semantics': PERIODIC_REFRESH_SCHEDULE_SEMANTICS,
		'refresh_after_epochs': list(PERIODIC_REFRESH_SCHEDULE),
		'hmm_iterations_per_refresh': 2,
		'embedding_source': 'current_student',
		'embedding_mode': 'unmasked_eval_full_survey',
		'refresh_embedding_semantics': PERIODIC_REFRESH_EMBEDDING_SEMANTICS,
		'center_initialization': 'previous_generation',
		'center_update': 'full_mean',
		'center_update_semantics': PERIODIC_REFRESH_CENTER_UPDATE_SEMANTICS,
		'preprocessing_policy': PERIODIC_REFRESH_PREPROCESSING_POLICY,
		'target_activation_policy': PERIODIC_REFRESH_TARGET_ACTIVATION_POLICY,
		'empty_state_policy': 'error',
		'checkpoint_selection_policy': PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY,
		'initial_hard_target_manifest_sha256': periodic_refresh_identity[
			'initial_hard_target_manifest'
		]['sha256'],
		'initial_hmm_artifacts': periodic_refresh_identity['initial_hmm_artifacts'],
		'fixed_preprocessor_sha256': periodic_refresh_identity[
			'fixed_preprocessor_sha256'
		],
		'fixed_residualizer_sha256': periodic_refresh_identity[
			'fixed_residualizer_sha256'
		],
		'fixed_clustering_config_sha256': periodic_refresh_identity[
			'fixed_clustering_config_sha256'
		],
		'source_embedding_metadata_sha256': periodic_refresh_identity[
			'source_embedding_metadata_sha256'
		],
		'source_valid_token_hashes': periodic_refresh_identity[
			'source_valid_token_hashes'
		],
		'feature_dimension': periodic_refresh_identity['feature_dimension'],
		'generation_root': periodic_refresh_identity['generation_root'],
	}
	for key, expected in checks.items():
		if scientific.get(key) != expected:
			raise ValueError(
				f'identity.scientific_identity.{key} does not match the periodic '
				'refresh contract'
			)
	spatial = _required_mapping(config, 'spatial_context')
	for key, expected in CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT.items():
		if spatial.get(key) != expected:
			raise ValueError(
				f'periodic refresh spatial_context.{key} contract mismatch'
			)
	loss = _required_mapping(config, 'loss')
	for key, expected in (
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('distillation_weight', 0.2),
	):
		if loss.get(key) != expected or scientific.get(key) != expected:
			raise ValueError(f'periodic refresh {key} identity mismatch')
	student = _required_mapping(config, 'student')
	if student.get('unfreeze_top_blocks') != 1:
		raise ValueError('periodic refresh requires student.unfreeze_top_blocks == 1')
	if scientific.get('student_unfreeze_top_blocks') != 1:
		raise ValueError(
		'periodic refresh scientific identity requires student_unfreeze_top_blocks == 1'
	)
	if _required_mapping(config, 'train').get('epochs') != 25:
		raise ValueError('periodic refresh requires train.epochs == 25')


def _validate_manifests(manifests: Mapping[str, object]) -> None:
	_validate_non_empty_path(manifests, 'train', prefix='manifests')
	_validate_non_empty_path(manifests, 'train_path_list', prefix='manifests')


def _validate_strat_hmm_pretext_data(
	data: Mapping[str, object],
) -> tuple[int, int, int]:
	local_crop_size = _validate_positive_int_triplet(
		data,
		'local_crop_size',
		prefix='data',
	)
	_validate_optional_fraction(data, 'min_valid_fraction', prefix='data')
	_validate_positive_int(data, 'max_resample_attempts', prefix='data')
	if data.get('normalized_clip_abs') is not None:
		_validate_positive_finite_number(
			data,
			'normalized_clip_abs',
			prefix='data',
		)
	_validate_amplitude_agc(data)
	_validate_finite_check_mode(data)
	return local_crop_size


def _validate_finite_check_mode(data: Mapping[str, object]) -> None:
	mode = data.get('finite_check_mode')
	if mode not in SUPPORTED_FINITE_CHECK_MODES:
		msg = (
			'data.finite_check_mode must be one of '
			f'{sorted(SUPPORTED_FINITE_CHECK_MODES)!r}; got {mode!r}'
		)
		raise ValueError(msg)


def _validate_strat_hmm_pretext_pseudo_targets(
	pseudo_targets: Mapping[str, object],
	*,
	multi_head: bool,
) -> None:
	if multi_head:
		if 'input_dir' in pseudo_targets or 'k' in pseudo_targets:
			raise ValueError(
				'multi-head pseudo_targets must use manifest, not input_dir or k'
			)
		manifest = _validate_non_empty_path(
			pseudo_targets,
			'manifest',
			prefix='pseudo_targets',
		)
		if not manifest.is_file():
			raise FileNotFoundError(
				f'pseudo_targets.manifest must exist and be a file: {manifest}'
			)
		_validate_optional_fraction(
			pseudo_targets,
			'min_confidence',
			prefix='pseudo_targets',
		)
		_strat_hmm_multi_head_target_representation(pseudo_targets)
		return
	if 'manifest' in pseudo_targets:
		raise ValueError(
			'single-head pseudo_targets must use input_dir and k, not manifest'
		)
	input_dir = _validate_non_empty_path(
		pseudo_targets,
		'input_dir',
		prefix='pseudo_targets',
	)
	if not input_dir.is_dir():
		msg = f'pseudo_targets.input_dir must exist and be a directory: {input_dir}'
		raise FileNotFoundError(msg)
	_validate_positive_int(pseudo_targets, 'k', prefix='pseudo_targets')
	_validate_optional_fraction(
		pseudo_targets,
		'min_confidence',
		prefix='pseudo_targets',
	)


def _validate_strat_hmm_pretext_teacher(teacher: Mapping[str, object]) -> None:
	checkpoint = _validate_non_empty_path(teacher, 'checkpoint', prefix='teacher')
	if not checkpoint.is_file():
		msg = f'teacher.checkpoint must exist and be a file: {checkpoint}'
		raise FileNotFoundError(msg)


def _validate_strat_hmm_pretext_student(
	student: Mapping[str, object],
	*,
	encoder_depth: int,
) -> None:
	_validate_nonnegative_int(
		student,
		'unfreeze_top_blocks',
		prefix='student',
	)
	unfreeze_top_blocks = int(student['unfreeze_top_blocks'])
	if unfreeze_top_blocks > encoder_depth:
		msg = (
			'student.unfreeze_top_blocks must be less than or equal to '
			f'model.encoder_depth ({encoder_depth}); got {unfreeze_top_blocks}'
		)
		raise ValueError(msg)
	init_checkpoint_value = student.get('init_checkpoint')
	if init_checkpoint_value is None:
		return
	init_checkpoint = _validate_non_empty_path(
		student,
		'init_checkpoint',
		prefix='student',
	)
	if not init_checkpoint.is_file():
		msg = f'student.init_checkpoint must exist and be a file: {init_checkpoint}'
		raise FileNotFoundError(msg)


def _validate_strat_hmm_pretext_head(
	head: Mapping[str, object],
	*,
	multi_head: bool,
) -> None:
	if multi_head:
		if 'num_prototypes' in head:
			raise ValueError('multi-head head must use ks, not num_prototypes')
		if head.get('spec') != _STRAT_HMM_MULTI_HEAD_SPEC:
			raise ValueError(
				f'head.spec must be {_STRAT_HMM_MULTI_HEAD_SPEC!r}; '
				f'got {head.get("spec")!r}'
			)
		_validate_required_key(head, 'ks', prefix='head')
		_validate_head_ks(head['ks'], prefix='head')
		_validate_required_key(head, 'projection_dim', prefix='head')
		_validate_positive_int(head, 'projection_dim', prefix='head')
	else:
		if 'ks' in head:
			raise ValueError('single-head head must use num_prototypes, not ks')
		_validate_positive_int(head, 'num_prototypes', prefix='head')
	if head.get('projection_dim') is not None:
		_validate_positive_int(head, 'projection_dim', prefix='head')
	_validate_positive_finite_number(head, 'temperature', prefix='head')
	_validate_bool(head, 'normalize', prefix='head')


def _validate_strat_hmm_pretext_loss(
	loss: Mapping[str, object],
	*,
	unfreeze_top_blocks: int,
	multi_head: bool,
) -> None:
	if not multi_head and ('consistency_weight' in loss or 'consistency_beta' in loss):
		raise ValueError(
			'single-head loss must not define multi-head consistency fields'
		)
	weight_keys = ['prototype_weight', 'usage_weight', 'distillation_weight']
	if multi_head:
		_validate_required_key(loss, 'consistency_weight', prefix='loss')
		_validate_required_key(loss, 'consistency_beta', prefix='loss')
		weight_keys.append('consistency_weight')
		_validate_positive_finite_number(loss, 'consistency_beta', prefix='loss')
	for key in weight_keys:
		_validate_nonnegative_finite_number(loss, key, prefix='loss')
	if not any(float(loss[key]) > 0.0 for key in weight_keys):
		msg = 'at least one strat HMM pretext loss weight must be positive'
		raise ValueError(msg)
	if loss.get('entropy_floor') is not None:
		_validate_nonnegative_finite_number(loss, 'entropy_floor', prefix='loss')
	if unfreeze_top_blocks > 0 and float(loss['distillation_weight']) <= 0.0:
		msg = (
			'loss.distillation_weight must be positive when '
			'student.unfreeze_top_blocks is greater than 0'
		)
		raise ValueError(msg)


def _validate_head_ks(value: object, *, prefix: str) -> None:
	validate_multi_resolution_head_ks(value, prefix=prefix)


def _strat_hmm_multi_head_target_representation(
	pseudo_targets: Mapping[str, object],
) -> str:
	"""Return the explicit representation, retaining the legacy hard default."""
	value = pseudo_targets.get('target_representation')
	if value is None:
		return _STRAT_HMM_HARD_TARGET_REPRESENTATION
	if not isinstance(value, str):
		raise TypeError('pseudo_targets.target_representation must be a string')
	if value not in {
		_STRAT_HMM_HARD_TARGET_REPRESENTATION,
		_STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION,
		_STRAT_HMM_LATERAL_TARGET_REPRESENTATION,
		_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION,
		_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION,
	}:
		supported = [
			_STRAT_HMM_HARD_TARGET_REPRESENTATION,
			_STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION,
			_STRAT_HMM_LATERAL_TARGET_REPRESENTATION,
			_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION,
			_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION,
		]
		raise ValueError(
			f'pseudo_targets.target_representation must be one of {supported!r}'
		)
	return str(value)


def _validate_strat_hmm_multi_head_manifest(
	pseudo_targets: Mapping[str, object],
	head: Mapping[str, object],
	*,
	target_representation: str,
) -> Mapping[str, object]:
	"""Validate manifest references without loading pseudo-target arrays."""
	if target_representation == _STRAT_HMM_HARD_TARGET_REPRESENTATION:
		multi_head = importlib.import_module('seis_ssl_cluster.stratigraphy.multi_head')
		manifest = multi_head.load_multi_head_target_manifest(
			str(pseudo_targets['manifest']),
			validate_array_semantics=False,
		)
	elif target_representation == _STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION:
		state_posterior = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.state_posterior'
		)
		manifest = state_posterior.load_multi_head_state_posterior_manifest(
			str(pseudo_targets['manifest']),
			validate_array_semantics=False,
		)
	elif target_representation == _STRAT_HMM_LATERAL_TARGET_REPRESENTATION:
		lateral_targets = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.lateral_targets'
		)
		manifest = lateral_targets.load_multi_head_lateral_target_manifest(
			str(pseudo_targets['manifest']), validate_array_semantics=False
		)
	elif (
		target_representation
		== _STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION
	):
		xy_neighbor_consensus_targets = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets'
		)
		loader_name = 'load_multi_head_xy_neighbor_consensus_target_manifest'
		load_manifest = getattr(xy_neighbor_consensus_targets, loader_name)
		manifest = load_manifest(
			str(pseudo_targets['manifest']), validate_array_semantics=False
		)
	elif (
		target_representation
		== _STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION
	):
		xy_neighbor_unanimous_targets = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets'
		)
		loader_name = 'load_multi_head_xy_neighbor_unanimous_target_manifest'
		load_manifest = getattr(xy_neighbor_unanimous_targets, loader_name)
		manifest = load_manifest(
			str(pseudo_targets['manifest']), validate_array_semantics=False
		)
	else:  # pragma: no cover - representation validation runs first
		raise AssertionError('unsupported validated multi-head target representation')
	if tuple(manifest['head_ks']) != tuple(head['ks']):
		raise ValueError('manifest.head_ks must equal head.ks')
	return manifest


def _file_sha256(path: str) -> str:
	"""Return a file digest without importing optional clustering dependencies."""
	digest = sha256()
	with Path(path).open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _multi_head_target_hashes(
	manifest: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	"""Extract per-head artifact hashes for the resolved scientific identity."""
	heads = manifest['heads']
	if not isinstance(heads, Mapping):
		raise TypeError('validated multi-head manifest has mapping heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in manifest['head_ks']:
		head = heads[str(k)]
		if not isinstance(head, Mapping):
			raise TypeError('validated multi-head manifest has mapping head entries')
		surveys = head['surveys']
		if not isinstance(surveys, Mapping):
			raise TypeError('validated multi-head manifest has mapping surveys')
		result[str(k)] = {
			str(survey_id): {
				name: str(entry[name]['sha256'])
				for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
			}
			for survey_id, entry in surveys.items()
			if isinstance(entry, Mapping)
		}
	return result


def _multi_head_posterior_hashes(
	manifest: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	"""Extract the posterior, valid-token, and metadata hashes by head/survey."""
	heads = manifest['heads']
	if not isinstance(heads, Mapping):
		raise TypeError('validated posterior manifest has mapping heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in manifest['head_ks']:
		head = heads[str(k)]
		if not isinstance(head, Mapping) or not isinstance(
			head.get('surveys'), Mapping
		):
			raise TypeError('validated posterior manifest has mapping surveys')
		result[str(k)] = {
			str(survey_id): {
				name: str(entry[name]['sha256'])
				for name in ('posterior', 'valid_tokens', 'metadata')
			}
			for survey_id, entry in head['surveys'].items()
			if isinstance(entry, Mapping)
		}
	return result


def _manifest_reference_sha256(manifest: Mapping[str, object], key: str) -> str:
	reference = manifest.get(key)
	if not isinstance(reference, Mapping):
		raise TypeError(f'validated target manifest {key} must be a mapping')
	value = reference.get('sha256')
	if not isinstance(value, str):
		raise TypeError(f'validated target manifest {key}.sha256 must be a string')
	return value


def _lateral_smoothing_identity(manifest: Mapping[str, object]) -> dict[str, object]:
	smoothing = manifest.get('smoothing')
	if not isinstance(smoothing, Mapping):
		raise TypeError('validated lateral manifest smoothing must be a mapping')
	heads = manifest.get('heads')
	if not isinstance(heads, Mapping):
		raise TypeError('validated lateral manifest heads must be a mapping')
	resolved_scales: dict[str, object] = {}
	for k in manifest['head_ks']:
		head = heads.get(str(k))
		if not isinstance(head, Mapping) or not isinstance(
			head.get('diagnostics'), Mapping
		):
			raise TypeError('validated lateral manifest head diagnostics are invalid')
		resolved = head['diagnostics'].get('resolved_scales')
		if not isinstance(resolved, Mapping):
			raise TypeError('validated lateral manifest resolved scales are invalid')
		resolved_scales[str(k)] = deepcopy(resolved)
	return {**deepcopy(dict(smoothing)), 'resolved_scales': resolved_scales}


def _xy_neighbor_consensus_smoothing_identity(
	manifest: Mapping[str, object],
) -> dict[str, object]:
	"""Return the fixed XY consensus policy recorded by an immutable export."""
	smoothing = manifest.get('smoothing')
	if not isinstance(smoothing, Mapping):
		raise TypeError(
			'validated XY neighbor consensus manifest smoothing must be a mapping'
		)
	return deepcopy(dict(smoothing))


def _xy_neighbor_unanimous_smoothing_identity(
	manifest: Mapping[str, object],
) -> dict[str, object]:
	"""Return the fixed XY unanimous policy recorded by an immutable export."""
	smoothing = manifest.get('smoothing')
	if not isinstance(smoothing, Mapping):
		raise TypeError(
			'validated XY neighbor unanimous manifest smoothing must be a mapping'
		)
	return deepcopy(dict(smoothing))


def _validate_m5_u_scientific_identity(
	scientific: Mapping[str, object],
	*,
	model_tag: object,
	manifest_sha256: str,
	manifest: Mapping[str, object],
	loss: Mapping[str, object],
) -> None:
	"""Keep M5-U's soft-posterior identity separate from the M4 contract."""
	expected = {
		'experiment_role': 'multi_head_ordered_soft_posterior_pretext',
		'variant': 'soft_nocons',
		'target_representation': _STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION,
		'posterior_semantics': _STRAT_HMM_POSTERIOR_SEMANTICS,
		'posterior_cost_temperature': 1.0,
		'supervised_loss': 'soft_categorical_cross_entropy_v1',
		'head_spec': _STRAT_HMM_MULTI_HEAD_SPEC,
		'head_ks': [6, 8, 10],
		'consistency_policy': 'disabled_for_m5_u_v1',
		'consistency_weight': 0.0,
		'posterior_manifest_sha256': manifest_sha256,
		'posterior_head_hashes': _multi_head_posterior_hashes(manifest),
	}
	if model_tag != 'strat_hmm_pretext_mh_k6810_soft_nocons_topblock1_distill_v1':
		raise ValueError('identity.model_tag does not match M5-U soft_nocons')
	for key, value in expected.items():
		if scientific.get(key) != value:
			raise ValueError(
				f'identity.scientific_identity.{key} does not match M5-U '
				'soft posterior contract'
			)
	if loss['consistency_weight'] != 0.0:
		raise ValueError(
			'loss.consistency_weight must be 0.0 for soft posterior training'
		)
	if loss['prototype_weight'] != 1.0:
		raise ValueError(
			'loss.prototype_weight must be 1.0 for soft posterior training'
		)
	if loss['usage_weight'] != 0.005:
		raise ValueError('loss.usage_weight must be 0.005 for soft posterior training')
	if loss['distillation_weight'] != 0.2:
		raise ValueError(
			'loss.distillation_weight must be 0.2 for soft posterior training'
		)


def _validate_center_trace_scientific_identity(
	scientific: Mapping[str, object],
	*,
	model_tag: object,
	manifest_sha256: str,
	manifest: Mapping[str, object],
	loss: Mapping[str, object],
) -> None:
	"""Validate the fixed schema-7 center-trace scientific identity."""
	expected = {
		'experiment_role': CENTER_TRACE_EXPERIMENT_ROLE,
		'variant': CENTER_TRACE_VARIANT,
		'head_spec': _STRAT_HMM_MULTI_HEAD_SPEC,
		'head_ks': [6, 8, 10],
		'target_representation': _STRAT_HMM_HARD_TARGET_REPRESENTATION,
		'target_manifest_sha256': manifest_sha256,
		'target_head_hashes': _multi_head_target_hashes(manifest),
		'objective_semantics': CENTER_TRACE_OBJECTIVE,
		'mask_semantics': CENTER_TRACE_MASK_SEMANTICS,
		'column_fraction': 0.10,
		'selection_policy': CENTER_TRACE_SELECTION_POLICY,
		'replacement': CENTER_TRACE_REPLACEMENT,
		'replacement_initialization': CENTER_TRACE_REPLACEMENT_INITIALIZATION,
		'rng_policy': CENTER_TRACE_RNG_POLICY,
		'masked_prototype_weight': 0.50,
		'visible_prototype_weight': 0.50,
		'distillation_scope': CENTER_TRACE_DISTILLATION_SCOPE,
		'supervised_loss': CENTER_TRACE_SUPERVISED_LOSS,
		'consistency_policy': CENTER_TRACE_CONSISTENCY_POLICY,
		'consistency_weight': 0.0,
	}
	if model_tag != CENTER_TRACE_MODEL_TAG:
		raise ValueError(
			'identity.model_tag does not match the fixed center-trace masked contract'
		)
	for key, value in expected.items():
		if scientific.get(key) != value:
			raise ValueError(
				f'identity.scientific_identity.{key} does not match the fixed '
				'center-trace masked contract'
			)
	if scientific.get('head_ks') != [6, 8, 10]:
		raise ValueError('center-trace scientific identity requires head K=[6, 8, 10]')
	for key, value in (
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('consistency_weight', 0.0),
		('distillation_weight', 0.2),
	):
		if loss.get(key) != value:
			raise ValueError(
				f'loss.{key} must be {value} for center-trace masked training'
			)
	if loss.get('consistency_beta') != 0.1:
		raise ValueError(
			'loss.consistency_beta must be 0.1 for center-trace masked training'
		)
	if scientific.get('student_unfreeze_top_blocks') != 1:
		raise ValueError(
			'identity.scientific_identity.student_unfreeze_top_blocks must be 1 '
			'for center-trace masked training'
		)


def _validate_m5_ls_scientific_identity(
	scientific: Mapping[str, object],
	*,
	model_tag: object,
	manifest_sha256: str,
	manifest: Mapping[str, object],
	loss: Mapping[str, object],
) -> None:
	"""Keep the lateral hard-label experiment separate from M4 and M5-U."""
	expected = {
		'experiment_role': 'multi_head_ordered_lateral_hard_pretext',
		'variant': 'latmf1_nocons',
		'target_representation': _STRAT_HMM_LATERAL_TARGET_REPRESENTATION,
		'target_semantics': _STRAT_HMM_LATERAL_SEMANTICS,
		'lateral_target_manifest_sha256': manifest_sha256,
		'lateral_target_head_hashes': _multi_head_target_hashes(manifest),
		'source_hard_manifest_sha256': _manifest_reference_sha256(
			manifest, 'source_hard_manifest'
		),
		'source_posterior_manifest_sha256': _manifest_reference_sha256(
			manifest, 'source_posterior_manifest'
		),
		'lateral_smoothing': _lateral_smoothing_identity(manifest),
		'supervised_loss': 'structured_hmm_hard_categorical_v1',
		'head_spec': _STRAT_HMM_MULTI_HEAD_SPEC,
		'head_ks': [6, 8, 10],
		'consistency_policy': 'disabled_for_m5_ls_v1',
		'consistency_weight': 0.0,
	}
	if model_tag != 'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1':
		raise ValueError('identity.model_tag does not match M5-LS latmf1_nocons')
	for key, value in expected.items():
		if scientific.get(key) != value:
			raise ValueError(
				f'identity.scientific_identity.{key} does not match M5-LS '
				'lateral contract'
			)
	for key, value in (
		('consistency_weight', 0.0),
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('distillation_weight', 0.2),
	):
		if loss[key] != value:
			raise ValueError(f'loss.{key} must be {value} for lateral hard training')


def _validate_xy_neighbor_consensus_scientific_identity(
	scientific: Mapping[str, object],
	*,
	model_tag: object,
	manifest_sha256: str,
	manifest: Mapping[str, object],
	loss: Mapping[str, object],
) -> None:
	"""Keep XY consensus hard labels independent of M4, M5-U, and M5-LS."""
	expected = {
		'experiment_role': 'multi_head_ordered_xy_neighbor_consensus_hard_pretext',
		'variant': 'xycons1_nocons',
		'target_representation': (
			_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION
		),
		'target_semantics': _STRAT_HMM_XY_NEIGHBOR_CONSENSUS_SEMANTICS,
		'xy_neighbor_consensus_target_manifest_sha256': manifest_sha256,
		'xy_neighbor_consensus_target_head_hashes': _multi_head_target_hashes(manifest),
		'source_hard_manifest_sha256': _manifest_reference_sha256(
			manifest, 'source_hard_manifest'
		),
		'xy_neighbor_consensus_smoothing': (
			_xy_neighbor_consensus_smoothing_identity(manifest)
		),
		'supervised_loss': 'structured_hmm_hard_categorical_v1',
		'head_spec': _STRAT_HMM_MULTI_HEAD_SPEC,
		'head_ks': [6, 8, 10],
		'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
		'consistency_weight': 0.0,
	}
	if model_tag != 'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1':
		raise ValueError(
			'identity.model_tag does not match XY consensus xycons1_nocons'
		)
	for key, value in expected.items():
		if scientific.get(key) != value:
			raise ValueError(
				f'identity.scientific_identity.{key} does not match XY neighbor '
				'consensus hard-label contract'
			)
	for key, value in (
		('consistency_weight', 0.0),
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('distillation_weight', 0.2),
	):
		if loss[key] != value:
			raise ValueError(
				f'loss.{key} must be {value} for XY consensus hard-label training'
			)


def _validate_xy_neighbor_unanimous_scientific_identity(
	scientific: Mapping[str, object],
	*,
	model_tag: object,
	manifest_sha256: str,
	manifest: Mapping[str, object],
	loss: Mapping[str, object],
) -> None:
	"""Keep unanimous hard labels independent of all older pretext routes."""
	expected = {
		'experiment_role': (
			'multi_head_ordered_xy_neighbor_unanimous_hard_pretext'
		),
		'variant': 'xyunanim1_nocons',
		'target_representation': (
			_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_TARGET_REPRESENTATION
		),
		'target_semantics': _STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_SEMANTICS,
		'xy_neighbor_unanimous_target_manifest_sha256': manifest_sha256,
		'xy_neighbor_unanimous_target_head_hashes': _multi_head_target_hashes(
			manifest
		),
		'source_hard_manifest_sha256': _manifest_reference_sha256(
			manifest, 'source_hard_manifest'
		),
		'xy_neighbor_unanimous_smoothing': (
			_xy_neighbor_unanimous_smoothing_identity(manifest)
		),
		'supervised_loss': 'structured_hmm_hard_categorical_v1',
		'head_spec': _STRAT_HMM_MULTI_HEAD_SPEC,
		'head_ks': [6, 8, 10],
		'consistency_policy': 'disabled_for_xy_neighbor_unanimous_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'student_unfreeze_top_blocks': 1,
	}
	if model_tag != 'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1':
		raise ValueError(
			'identity.model_tag does not match XY unanimous xyunanim1_nocons'
		)
	for key, value in expected.items():
		if scientific.get(key) != value:
			raise ValueError(
				f'identity.scientific_identity.{key} does not match XY neighbor '
				'unanimous hard-label contract'
			)
	for key, value in (
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('prototype_weight', 1.0),
		('usage_weight', 0.005),
		('distillation_weight', 0.2),
	):
		if loss[key] != value:
			raise ValueError(
				f'loss.{key} must be {value} for XY unanimous hard-label training'
			)


def _validate_strat_hmm_pretext_cross_section_values(
	pseudo_targets: Mapping[str, object],
	head: Mapping[str, object],
) -> None:
	if int(pseudo_targets['k']) != int(head['num_prototypes']):
		msg = (
			'pseudo_targets.k must equal head.num_prototypes; '
			f'got {pseudo_targets["k"]!r} and {head["num_prototypes"]!r}'
		)
		raise ValueError(msg)


def _validate_strat_hmm_pretext_train(train: Mapping[str, object]) -> None:
	for key in ('batch_size', 'samples_per_epoch', 'epochs'):
		_validate_positive_int(train, key, prefix='train')
	for key in ('num_workers', 'max_steps'):
		_validate_optional_nonnegative_int(train, key, prefix='train')
	_validate_optional_positive_int(train, 'checkpoint_every_steps', prefix='train')
	for key in ('lr', 'encoder_lr', 'grad_clip_norm'):
		_validate_positive_number(train, key, prefix='train')
	_validate_nonnegative_number(train, 'weight_decay', prefix='train')
	for key in ('amp', 'shuffle', 'allow_overwrite_output'):
		_validate_bool(train, key, prefix='train')
	_validate_optional_train_seed(train)
	_validate_optional_train_device(train)


def _reject_fixed_contract_keys(config: Mapping[str, object]) -> None:
	for section, fixed_keys in _FIXED_RAW_KEYS.items():
		value = config.get(section)
		if not isinstance(value, Mapping):
			continue
		stale = sorted(set(value) & set(fixed_keys))
		if stale:
			labels = [f'{section}.{key}' for key in stale]
			msg = (
				f'{labels[0]} is fixed by the amplitude-only MVP config '
				'resolver and must be removed from raw YAML.'
			)
			raise ValueError(msg)


def _validate_model(model: Mapping[str, object]) -> None:
	for key in (
		'encoder_dim',
		'encoder_depth',
		'encoder_heads',
		'decoder_dim',
		'decoder_depth',
		'decoder_heads',
	):
		_validate_positive_int(model, key, prefix='model')


def _validate_continuation(
	continuation: Mapping[str, object],
	*,
	encoder_depth: int,
) -> None:
	_validate_allowed_keys(
		continuation,
		_CONTINUATION_KEYS,
		prefix='continuation',
	)
	_validate_required_keys(
		continuation,
		_CONTINUATION_KEYS,
		prefix='continuation',
	)
	_validate_absolute_path(
		continuation,
		'init_checkpoint',
		prefix='continuation',
	)
	_validate_positive_int(
		continuation,
		'unfreeze_top_blocks',
		prefix='continuation',
	)
	unfreeze_top_blocks = int(continuation['unfreeze_top_blocks'])
	if unfreeze_top_blocks > encoder_depth:
		msg = (
			'continuation.unfreeze_top_blocks must be less than or equal to '
			f'model.encoder_depth ({encoder_depth}); got {unfreeze_top_blocks}'
		)
		raise ValueError(msg)


def _validate_vicreg_continuation(
	continuation: Mapping[str, object],
	*,
	encoder_depth: int,
) -> None:
	_validate_allowed_keys(
		continuation,
		_CONTINUATION_KEYS,
		prefix='continuation',
	)
	_validate_required_keys(
		continuation,
		_CONTINUATION_KEYS,
		prefix='continuation',
	)
	_validate_absolute_path(
		continuation,
		'init_checkpoint',
		prefix='continuation',
	)
	_validate_nonnegative_int(
		continuation,
		'unfreeze_top_blocks',
		prefix='continuation',
	)
	unfreeze_top_blocks = int(continuation['unfreeze_top_blocks'])
	if unfreeze_top_blocks > encoder_depth:
		msg = (
			'continuation.unfreeze_top_blocks must be less than or equal to '
			f'model.encoder_depth ({encoder_depth}); got {unfreeze_top_blocks}'
		)
		raise ValueError(msg)


def _validate_masking(masking: Mapping[str, object]) -> None:
	ratio = masking.get('spatial_mask_ratio')
	if (
		not isinstance(ratio, Real)
		or isinstance(ratio, bool)
		or ratio <= 0.0
		or ratio >= 1.0
	):
		msg = 'masking.spatial_mask_ratio must be greater than 0 and less than 1'
		raise ValueError(msg)

	_validate_positive_int_triplet(
		masking,
		'block_size_tokens',
		prefix='masking',
	)


def _validate_train(train: Mapping[str, object]) -> None:
	for key in ('batch_size', 'samples_per_epoch', 'epochs'):
		_validate_positive_int(train, key, prefix='train')
	_validate_optional_train_numbers(train)
	_validate_bool(train, 'amp', prefix='train')
	amp_dtype = train.get('amp_dtype')
	if amp_dtype not in SUPPORTED_AMP_DTYPES:
		msg = (
			'train.amp_dtype must be one of '
			f'{sorted(SUPPORTED_AMP_DTYPES)!r}; got {amp_dtype!r}'
		)
		raise ValueError(msg)
	for key in (
		'shuffle',
		'allow_overwrite_output',
		'persistent_workers',
		'stage_timing',
	):
		if key in train:
			_validate_bool(train, key, prefix='train')
	_validate_optional_train_seed(train)
	_validate_optional_train_device(train)
	_validate_runtime_check_mode(train)


def _validate_barlow_twins_train(train: Mapping[str, object]) -> None:
	_validate_positive_int(train, 'batch_size', prefix='train')
	if int(train['batch_size']) < 2:
		raise ValueError('train.batch_size must be at least 2')
	for key in ('samples_per_epoch', 'epochs'):
		_validate_positive_int(train, key, prefix='train')
	for key in ('num_workers', 'max_steps'):
		_validate_optional_nonnegative_int(train, key, prefix='train')
	if train.get('prefetch_factor') is not None:
		_validate_positive_int(train, 'prefetch_factor', prefix='train')
	for key in ('lr', 'grad_clip_norm'):
		_validate_positive_finite_number(train, key, prefix='train')
	_validate_nonnegative_finite_number(train, 'weight_decay', prefix='train')
	for key in (
		'amp',
		'shuffle',
		'allow_overwrite_output',
		'persistent_workers',
	):
		_validate_bool(train, key, prefix='train')
	amp_dtype = train.get('amp_dtype')
	if amp_dtype not in SUPPORTED_AMP_DTYPES:
		msg = (
			'train.amp_dtype must be one of '
			f'{sorted(SUPPORTED_AMP_DTYPES)!r}; got {amp_dtype!r}'
		)
		raise ValueError(msg)
	_validate_optional_train_seed(train)
	_validate_optional_train_device(train)


def _validate_barlow_twins_method(
	barlow_twins: Mapping[str, object],
	*,
	local_crop_size: Sequence[int],
	patch_size: Sequence[int],
) -> None:
	method = barlow_twins.get('method', BARLOW_TWINS_PRETRAINING_METHOD)
	if not isinstance(method, str):
		msg = f'barlow_twins.method must be a string; got {method!r}'
		raise TypeError(msg)
	if method not in SUPPORTED_BARLOW_TWINS_PRETRAINING_METHODS:
		msg = (
			'barlow_twins.method must be one of '
			f'{sorted(SUPPORTED_BARLOW_TWINS_PRETRAINING_METHODS)!r}; '
			f'got {method!r}'
		)
		raise ValueError(msg)

	if method != LOCAL_BARLOW_TWINS_PRETRAINING_METHOD:
		if 'local_pairs_per_crop' in barlow_twins:
			msg = (
				'barlow_twins.local_pairs_per_crop is only allowed when '
				'barlow_twins.method is '
				f'{LOCAL_BARLOW_TWINS_PRETRAINING_METHOD!r}'
			)
			raise ValueError(msg)
		return

	_validate_required_key(
		barlow_twins,
		'local_pairs_per_crop',
		prefix='barlow_twins',
	)
	_validate_positive_int(
		barlow_twins,
		'local_pairs_per_crop',
		prefix='barlow_twins',
	)
	local_pairs_per_crop = int(barlow_twins['local_pairs_per_crop'])
	token_count = math.prod(
		crop_axis // patch_axis
		for crop_axis, patch_axis in zip(
			local_crop_size,
			patch_size,
			strict=True,
		)
	)
	if local_pairs_per_crop > token_count:
		msg = (
			'barlow_twins.local_pairs_per_crop must be less than or equal to '
			f'the crop token count ({token_count}); got {local_pairs_per_crop}'
		)
		raise ValueError(msg)


def _validate_vicreg_method(
	vicreg: Mapping[str, object],
	*,
	local_crop_size: Sequence[int],
	patch_size: Sequence[int],
) -> None:
	_validate_required_keys(
		vicreg,
		frozenset({'method', 'local_pairs_per_crop'}),
		prefix='vicreg',
	)
	method = vicreg['method']
	if not isinstance(method, str):
		msg = f'vicreg.method must be a string; got {method!r}'
		raise TypeError(msg)
	if method not in SUPPORTED_VICREG_PRETRAINING_METHODS:
		msg = (
			'vicreg.method must be one of '
			f'{sorted(SUPPORTED_VICREG_PRETRAINING_METHODS)!r}; got {method!r}'
		)
		raise ValueError(msg)
	_validate_positive_int(
		vicreg,
		'local_pairs_per_crop',
		prefix='vicreg',
	)
	local_pairs_per_crop = int(vicreg['local_pairs_per_crop'])
	token_count = math.prod(
		crop_axis // patch_axis
		for crop_axis, patch_axis in zip(
			local_crop_size,
			patch_size,
			strict=True,
		)
	)
	if local_pairs_per_crop > token_count:
		msg = (
			'vicreg.local_pairs_per_crop must be less than or equal to '
			f'the crop token count ({token_count}); got {local_pairs_per_crop}'
		)
		raise ValueError(msg)


def _validate_barlow_twins_augmentations(  # noqa: C901, PLR0912, PLR0913, PLR0915
	augmentations: Mapping[str, object],
	*,
	method: str,
	local_crop_size: Sequence[int],
	patch_size: Sequence[int],
	local_method: str = LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	method_section: str = 'barlow_twins',
) -> None:
	if 'policy' not in augmentations:
		legacy_keys = frozenset(DEFAULT_BARLOW_TWINS_AUGMENTATION_OPTIONS)
		_validate_allowed_keys(
			augmentations,
			legacy_keys,
			prefix='augmentations',
		)
		_validate_optional_fraction(
			augmentations,
			'horizontal_flip_probability',
			prefix='augmentations',
		)
		return

	policy = augmentations.get('policy')
	if not isinstance(policy, str):
		msg = f'augmentations.policy must be a string; got {policy!r}'
		raise TypeError(msg)
	if policy == IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY:
		_validate_allowed_keys(
			augmentations,
			_IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_KEYS,
			prefix='augmentations',
		)
		_validate_required_keys(
			augmentations,
			_IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_KEYS,
			prefix='augmentations',
		)
		if method != local_method:
			msg = (
				'augmentations.policy '
				f'{IDENTITY_GAUSSIAN_NOISE_AUGMENTATION_POLICY!r} '
				f'requires {method_section}.method '
				f'{local_method!r}'
			)
			raise ValueError(msg)
		_validate_positive_finite_number(
			augmentations,
			'gaussian_noise_std',
			prefix='augmentations',
		)
		return

	if policy == HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY:
		_validate_allowed_keys(
			augmentations,
			_HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_KEYS,
			prefix='augmentations',
		)
		_validate_required_keys(
			augmentations,
			_HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_KEYS,
			prefix='augmentations',
		)
		if method != local_method:
			msg = (
				'augmentations.policy '
				f'{HORIZONTAL_FLIP_GAUSSIAN_NOISE_AUGMENTATION_POLICY!r} '
				f'requires {method_section}.method '
				f'{local_method!r}'
			)
			raise ValueError(msg)
		_validate_fraction(
			augmentations,
			'horizontal_flip_probability',
			prefix='augmentations',
		)
		_validate_nonnegative_finite_number(
			augmentations,
			'gaussian_noise_std',
			prefix='augmentations',
		)
		return

	if policy == HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY:
		_validate_allowed_keys(
			augmentations,
			_HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_KEYS,
			prefix='augmentations',
		)
		_validate_required_keys(
			augmentations,
			_HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_KEYS,
			prefix='augmentations',
		)
		if method != local_method:
			msg = (
				'augmentations.policy '
				f'{HORIZONTAL_FLIP_TRACE_DROP_AUGMENTATION_POLICY!r} '
				f'requires {method_section}.method '
				f'{local_method!r}'
			)
			raise ValueError(msg)
		for key in ('horizontal_flip_probability', 'trace_drop_probability'):
			_validate_fraction(augmentations, key, prefix='augmentations')
		return

	if policy == HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY:
		_validate_allowed_keys(
			augmentations,
			_HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_KEYS,
			prefix='augmentations',
		)
		_validate_required_keys(
			augmentations,
			_HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_KEYS,
			prefix='augmentations',
		)
		if method != local_method:
			msg = (
				'augmentations.policy '
				f'{HORIZONTAL_FLIP_ZERO_PHASE_Z_FILTER_AUGMENTATION_POLICY!r} '
				f'requires {method_section}.method '
				f'{local_method!r}'
			)
			raise ValueError(msg)
		_validate_fraction(
			augmentations,
			'horizontal_flip_probability',
			prefix='augmentations',
		)
		_validate_zero_phase_z_filter_side_weight(
			augmentations,
			'z_filter_side_weight',
			prefix='augmentations',
		)
		return

	_validate_allowed_keys(
		augmentations,
		_D4_TRACE_DROP_AUGMENTATION_KEYS,
		prefix='augmentations',
	)
	_validate_required_keys(
		augmentations,
		_D4_TRACE_DROP_AUGMENTATION_KEYS,
		prefix='augmentations',
	)
	if policy != XY_D4_TRACE_DROP_AUGMENTATION_POLICY:
		msg = (
			'augmentations.policy must be '
			f'{XY_D4_TRACE_DROP_AUGMENTATION_POLICY!r}; got {policy!r}'
		)
		raise ValueError(msg)
	if method != local_method:
		msg = (
			f'augmentations.policy {XY_D4_TRACE_DROP_AUGMENTATION_POLICY!r} '
			f'requires {method_section}.method '
			f'{local_method!r}'
		)
		raise ValueError(msg)
	for key in ('reflection_probability', 'trace_drop_probability'):
		_validate_fraction(augmentations, key, prefix='augmentations')
	if local_crop_size[0] != local_crop_size[1]:
		raise ValueError(
			'data.local_crop_size X/Y dimensions must be equal for '
			f'augmentations.policy {XY_D4_TRACE_DROP_AUGMENTATION_POLICY!r}'
		)
	if patch_size[0] != patch_size[1]:
		raise ValueError(
			'model.patch_size X/Y dimensions must be equal for '
			f'augmentations.policy {XY_D4_TRACE_DROP_AUGMENTATION_POLICY!r}'
		)


def _validate_zero_phase_z_filter_side_weight(
	values: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	"""Require a non-degenerate convex centered Z-filter coefficient."""
	_validate_nonnegative_finite_number(values, key, prefix=prefix)
	side_weight = float(values[key])
	if not 0.0 < side_weight < 0.5:
		msg = f'{prefix}.{key} must be in (0, 0.5); got {side_weight!r}'
		raise ValueError(msg)


def _validate_runtime_check_mode(train: Mapping[str, object]) -> None:
	mode = train.get('runtime_check_mode')
	if mode not in SUPPORTED_RUNTIME_CHECK_MODES:
		msg = (
			'train.runtime_check_mode must be one of '
			f'{sorted(SUPPORTED_RUNTIME_CHECK_MODES)!r}; got {mode!r}'
		)
		raise ValueError(msg)


def _validate_optional_train_numbers(train: Mapping[str, object]) -> None:
	for key in ('num_workers', 'max_steps', 'checkpoint_every_steps'):
		if key in train:
			_validate_nonnegative_int(train, key, prefix='train')
	if train.get('prefetch_factor') is not None:
		_validate_positive_int(train, 'prefetch_factor', prefix='train')
	for key in ('lr', 'grad_clip_norm'):
		if key in train:
			_validate_positive_number(train, key, prefix='train')
	if 'weight_decay' in train:
		_validate_nonnegative_number(train, 'weight_decay', prefix='train')


def _validate_optional_train_seed(train: Mapping[str, object]) -> None:
	if 'seed' in train and not _is_int(train.get('seed')):
		msg = f'train.seed must be an integer; got {train.get("seed")!r}'
		raise ValueError(msg)


def _validate_optional_train_device(train: Mapping[str, object]) -> None:
	if 'device' in train:
		value = train.get('device')
		if value not in {'auto', 'cpu', 'cuda'}:
			msg = 'train.device must be "auto", "cpu", or "cuda"'
			raise ValueError(msg)


def _validate_loss(loss: Mapping[str, object]) -> None:
	_validate_required_key(loss, 'reconstruction', prefix='loss')
	reconstruction = loss.get('reconstruction')
	if reconstruction not in SUPPORTED_RECONSTRUCTION_LOSSES:
		msg = (
			'loss.reconstruction must be one of '
			f'{sorted(SUPPORTED_RECONSTRUCTION_LOSSES)!r}; '
			f'got {reconstruction!r}'
		)
		raise ValueError(msg)

	if reconstruction == 'huber':
		_validate_required_key(loss, 'huber_delta', prefix='loss')
		_validate_positive_finite_number(loss, 'huber_delta', prefix='loss')
	elif 'huber_delta' in loss:
		msg = 'loss.huber_delta must be omitted unless loss.reconstruction is huber'
		raise ValueError(msg)

	_validate_required_key(loss, 'gradient_weight', prefix='loss')
	_validate_nonnegative_finite_number(loss, 'gradient_weight', prefix='loss')
	_validate_required_key(loss, 'visible_reconstruction_weight', prefix='loss')
	_validate_nonnegative_finite_number(
		loss,
		'visible_reconstruction_weight',
		prefix='loss',
	)
	_validate_loss_target_normalization(loss)
	if (
		'valid_mask_mode' in loss
		and loss.get('valid_mask_mode') != EXPECTED_VALID_MASK_MODE
	):
		msg = "loss.valid_mask_mode must be resolved internally as 'voxel'"
		raise ValueError(msg)


def _validate_loss_target_normalization(loss: Mapping[str, object]) -> None:
	target_normalization = _required_child_mapping(
		loss,
		'target_normalization',
		prefix='loss',
	)
	_validate_allowed_keys(
		target_normalization,
		frozenset({'mode', 'eps', 'min_std'}),
		prefix='loss.target_normalization',
	)
	_validate_required_key(
		target_normalization,
		'mode',
		prefix='loss.target_normalization',
	)
	mode = target_normalization.get('mode')
	if mode not in SUPPORTED_TARGET_NORMALIZATION_MODES:
		msg = (
			'loss.target_normalization.mode must be one of '
			f'{sorted(SUPPORTED_TARGET_NORMALIZATION_MODES)!r}; got {mode!r}'
		)
		raise ValueError(msg)
	if mode == 'none':
		for key in ('eps', 'min_std'):
			if key in target_normalization:
				msg = (
					f'loss.target_normalization.{key} must be omitted '
					"when mode is 'none'"
				)
				raise ValueError(msg)
		return
	_validate_required_key(
		target_normalization,
		'eps',
		prefix='loss.target_normalization',
	)
	_validate_required_key(
		target_normalization,
		'min_std',
		prefix='loss.target_normalization',
	)
	_validate_positive_finite_number(
		target_normalization,
		'eps',
		prefix='loss.target_normalization',
	)
	_validate_positive_finite_number(
		target_normalization,
		'min_std',
		prefix='loss.target_normalization',
	)
	if float(loss.get('gradient_weight', 0.0)) != 0.0:
		msg = (
			'loss.gradient_weight must be 0.0 when '
			"loss.target_normalization.mode is 'patch_zscore'; "
			'the current gradient loss operates in survey-normalized amplitude space'
		)
		raise ValueError(msg)


def _validate_zero_mask(zero_mask: Mapping[str, object]) -> None:
	if 'enabled' in zero_mask:
		_validate_bool(zero_mask, 'enabled', prefix='zero_mask')
	if 'zero_atol' in zero_mask:
		_validate_nonnegative_number(zero_mask, 'zero_atol', prefix='zero_mask')
	for key in ('z_sample_influence_radius', 'xy_trace_influence_radius'):
		if key in zero_mask:
			_validate_nonnegative_int(zero_mask, key, prefix='zero_mask')


def _validate_amplitude_agc(data: Mapping[str, object]) -> None:
	amplitude_agc = _required_child_mapping(
		data,
		'amplitude_agc',
		prefix='data',
	)
	_validate_allowed_keys(
		amplitude_agc,
		_AMPLITUDE_AGC_KEYS,
		prefix='data.amplitude_agc',
	)
	_validate_required_key(amplitude_agc, 'enabled', prefix='data.amplitude_agc')
	_validate_bool(amplitude_agc, 'enabled', prefix='data.amplitude_agc')
	if not amplitude_agc['enabled']:
		extra = sorted(set(amplitude_agc) - {'enabled'})
		if extra:
			msg = (
				'data.amplitude_agc fields must be omitted when disabled; '
				f'got {extra!r}'
			)
			raise ValueError(msg)
		return
	_validate_required_keys(
		amplitude_agc,
		_AMPLITUDE_AGC_ENABLED_REQUIRED_KEYS,
		prefix='data.amplitude_agc',
	)
	if amplitude_agc.get('mode') != 'trace_rms_z':
		msg = (
			"data.amplitude_agc.mode must be 'trace_rms_z'; "
			f'got {amplitude_agc.get("mode")!r}'
		)
		raise ValueError(msg)
	_validate_positive_int(amplitude_agc, 'window_z', prefix='data.amplitude_agc')
	if int(amplitude_agc['window_z']) % 2 == 0:
		msg = (
			'data.amplitude_agc.window_z must be odd; '
			f'got {amplitude_agc["window_z"]!r}'
		)
		raise ValueError(msg)
	_validate_positive_finite_number(amplitude_agc, 'eps', prefix='data.amplitude_agc')
	_validate_positive_finite_number(
		amplitude_agc,
		'clip_abs',
		prefix='data.amplitude_agc',
	)


def _validate_mae_training_visualization(
	visualization: Mapping[str, object],
) -> None:
	_validate_allowed_keys(
		visualization,
		_MAE_TRAINING_VISUALIZATION_KEYS,
		prefix='visualization',
	)
	if 'mae_debug' not in visualization:
		return
	mae_debug = _required_child_mapping(
		visualization,
		'mae_debug',
		prefix='visualization',
	)
	_validate_allowed_keys(
		mae_debug,
		MAE_DEBUG_VISUALIZATION_KEYS,
		prefix='visualization.mae_debug',
	)
	_validate_mae_debug_general_fields(mae_debug)
	_validate_mae_debug_triggers(mae_debug)
	_validate_mae_debug_rendering_fields(mae_debug)


def _validate_mae_debug_general_fields(
	mae_debug: Mapping[str, object],
) -> None:
	if 'enabled' in mae_debug:
		_validate_bool(mae_debug, 'enabled', prefix='visualization.mae_debug')
	if 'output_dir' in mae_debug:
		value = mae_debug.get('output_dir')
		if value is not None:
			if not isinstance(value, str) or not value:
				msg = (
					'visualization.mae_debug.output_dir must be a non-empty '
					f'string or null; got {value!r}'
				)
				raise TypeError(msg)
			_validate_output_path(
				Path(value),
				'visualization.mae_debug.output_dir',
			)


def _validate_mae_debug_triggers(mae_debug: Mapping[str, object]) -> None:
	for key in ('every_steps', 'every_epochs'):
		_validate_optional_positive_int(
			mae_debug,
			key,
			prefix='visualization.mae_debug',
		)
	if _mae_debug_enabled(mae_debug) and not _mae_debug_has_trigger(mae_debug):
		msg = (
			'visualization.mae_debug requires every_steps or every_epochs '
			'when enabled is true'
		)
		raise ValueError(msg)


def _validate_mae_debug_rendering_fields(mae_debug: Mapping[str, object]) -> None:
	if 'max_samples' in mae_debug:
		_validate_positive_int(
			mae_debug,
			'max_samples',
			prefix='visualization.mae_debug',
		)
	for key in ('xy_slice_index', 'xz_slice_y_index'):
		_validate_optional_nonnegative_int(
			mae_debug,
			key,
			prefix='visualization.mae_debug',
		)
	if 'dpi' in mae_debug:
		_validate_positive_int(mae_debug, 'dpi', prefix='visualization.mae_debug')
	if 'clip_percentiles' in mae_debug:
		_validate_mae_debug_clip_percentiles(mae_debug)
	if 'columns' in mae_debug:
		_validate_mae_debug_columns(mae_debug)
	for key in ('panel_width', 'panel_height'):
		if key in mae_debug:
			_validate_positive_finite_number(
				mae_debug,
				key,
				prefix='visualization.mae_debug',
			)
	if 'invalid_color' in mae_debug:
		_validate_non_empty_str(
			mae_debug,
			'invalid_color',
			prefix='visualization.mae_debug',
		)


def _mae_debug_enabled(mae_debug: Mapping[str, object]) -> bool:
	value = mae_debug.get(
		'enabled',
		DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS['enabled'],
	)
	return bool(value)


def _mae_debug_has_trigger(mae_debug: Mapping[str, object]) -> bool:
	every_steps = mae_debug.get(
		'every_steps',
		DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS['every_steps'],
	)
	every_epochs = mae_debug.get(
		'every_epochs',
		DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS['every_epochs'],
	)
	return every_steps is not None or every_epochs is not None


def _validate_divisible_crop_patch(
	crop_size: Sequence[int],
	patch_size: Sequence[int],
) -> None:
	if any(
		crop % patch != 0 for crop, patch in zip(crop_size, patch_size, strict=True)
	):
		msg = (
			'data.local_crop_size dimensions must be divisible by '
			'model.patch_size dimensions'
		)
		raise ValueError(msg)


def _validate_mae_debug_clip_percentiles(
	mae_debug: Mapping[str, object],
) -> None:
	value = mae_debug.get('clip_percentiles')
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 2
	):
		msg = (
			'visualization.mae_debug.clip_percentiles must contain two '
			f'finite values; got {value!r}'
		)
		raise ValueError(msg)
	low, high = value
	if not _is_number(low) or not _is_number(high):
		msg = (
			'visualization.mae_debug.clip_percentiles must contain numeric '
			f'values; got {value!r}'
		)
		raise ValueError(msg)
	low_float = float(low)
	high_float = float(high)
	if (
		not math.isfinite(low_float)
		or not math.isfinite(high_float)
		or not 0.0 <= low_float < high_float <= 100.0
	):
		msg = (
			'visualization.mae_debug.clip_percentiles must satisfy '
			f'0 <= low < high <= 100; got {value!r}'
		)
		raise ValueError(msg)


def _validate_mae_debug_columns(mae_debug: Mapping[str, object]) -> None:
	value = mae_debug.get('columns')
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or not value
		or any(not isinstance(item, str) or not item for item in value)
	):
		msg = (
			'visualization.mae_debug.columns must be a non-empty sequence '
			f'of strings; got {value!r}'
		)
		raise ValueError(msg)
	if len(set(value)) != len(value):
		msg = (
			'visualization.mae_debug.columns must not contain duplicates; '
			f'got {list(value)!r}'
		)
		raise ValueError(msg)
	unknown = sorted(set(value) - MAE_DEBUG_VISUALIZATION_COLUMNS)
	if unknown:
		msg = (
			'visualization.mae_debug.columns contains unsupported column(s): '
			f'{unknown!r}; allowed columns are '
			f'{sorted(MAE_DEBUG_VISUALIZATION_COLUMNS)!r}'
		)
		raise ValueError(msg)


__all__ = [
	'CENTER_TRACE_CONSISTENCY_POLICY',
	'CENTER_TRACE_DISTILLATION_SCOPE',
	'CENTER_TRACE_EXPERIMENT_ROLE',
	'CENTER_TRACE_MASK_SEMANTICS',
	'CENTER_TRACE_MODEL_TAG',
	'CENTER_TRACE_OBJECTIVE',
	'CENTER_TRACE_REPLACEMENT',
	'CENTER_TRACE_REPLACEMENT_INITIALIZATION',
	'CENTER_TRACE_RNG_POLICY',
	'CENTER_TRACE_SELECTION_POLICY',
	'CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT',
	'CENTER_TRACE_SUPERVISED_LOSS',
	'CENTER_TRACE_VARIANT',
	'PERIODIC_REFRESH_CENTER_UPDATE_SEMANTICS',
	'PERIODIC_REFRESH_CHECKPOINT_SELECTION_POLICY',
	'PERIODIC_REFRESH_EMBEDDING_SEMANTICS',
	'PERIODIC_REFRESH_EXPERIMENT_ROLE',
	'PERIODIC_REFRESH_MODEL_ROLE',
	'PERIODIC_REFRESH_MODEL_TAG',
	'PERIODIC_REFRESH_PREPROCESSING_POLICY',
	'PERIODIC_REFRESH_SCHEDULE',
	'PERIODIC_REFRESH_SCHEDULE_SEMANTICS',
	'PERIODIC_REFRESH_SEMANTICS',
	'PERIODIC_REFRESH_TARGET_ACTIVATION_POLICY',
	'PERIODIC_REFRESH_VARIANT',
	'_is_center_trace_masked_config',
	'_is_periodic_refresh_config',
	'resolve_barlow_twins_training_config',
	'resolve_mae_training_config',
	'resolve_strat_hmm_pretext_config',
	'resolve_vicreg_training_config',
]
