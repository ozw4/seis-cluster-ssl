"""Validation and resolution for MAE pretraining configs."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.artifact_paths import (
	_validate_artifact_output_path,
	_validate_nopims_pretraining_path,
)
from seis_ssl_cluster.config.base import _resolve_base
from seis_ssl_cluster.config.common import (
	_is_int,
	_is_number,
	_merge_section_defaults,
	_required_child_mapping,
	_required_mapping,
	_validate_allowed_keys,
	_validate_bool,
	_validate_non_empty_path,
	_validate_non_empty_str,
	_validate_nonnegative_finite_number,
	_validate_nonnegative_int,
	_validate_nonnegative_number,
	_validate_optional_fraction,
	_validate_optional_nonnegative_int,
	_validate_optional_output_path_under_root,
	_validate_optional_positive_int,
	_validate_path,
	_validate_positive_finite_number,
	_validate_positive_int,
	_validate_positive_int_triplet,
	_validate_positive_number,
	_validate_required_key,
	_validate_required_keys,
)
from seis_ssl_cluster.config.schema import (
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
	DEFAULT_ZERO_MASK_CONTRACT,
	EXPECTED_VALID_MASK_MODE,
	FIXED_DATA_CONTRACT,
	FIXED_LOSS_CONTRACT,
	FIXED_MASKING_CONTRACT,
	FIXED_MODEL_CONTRACT,
	MAE_DEBUG_VISUALIZATION_COLUMNS,
	MAE_DEBUG_VISUALIZATION_KEYS,
	STAGE_MAE_TRAINING,
	STAGE_STRAT_HMM_PRETEXT_TRAINING,
	SUPPORTED_AMP_DTYPES,
	SUPPORTED_FINITE_CHECK_MODES,
	SUPPORTED_RECONSTRUCTION_LOSSES,
	SUPPORTED_RUNTIME_CHECK_MODES,
	SUPPORTED_TARGET_NORMALIZATION_MODES,
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
}

_STRAT_HMM_MULTI_HEAD_SPEC = MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1
_STRAT_HMM_MULTI_HEAD_CONSISTENCY_POLICY = 'normalized_order_smooth_l1_v1'
_STRAT_HMM_HARD_TARGET_REPRESENTATION = 'hard_viterbi_labels_v1'
_STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION = 'ordered_path_state_posterior_v1'
_STRAT_HMM_LATERAL_TARGET_REPRESENTATION = 'lateral_mean_field_hard_labels_v1'
_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION = (
	'xy_neighbor_consensus_hard_labels_v1'
)
_STRAT_HMM_POSTERIOR_SEMANTICS = 'ordered_path_cost_gibbs_state_marginal_v1'
_STRAT_HMM_LATERAL_SEMANTICS = 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1'
_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_SEMANTICS = (
	'xy_neighbor_consensus_hard_label_smoothing_v1'
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
	_validate_divisible_crop_patch(local_crop_size, patch_size)
	_validate_artifact_output_path(
		output_root,
		'paths.output_root',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_pretraining_path(
		output_root,
		'paths.output_root',
		artifact_root=paths.artifact_root,
	)
	_validate_masking(masking)
	_validate_loss(loss)
	_validate_train(train)
	_validate_zero_mask(_required_mapping(resolved, 'zero_mask'))
	if 'visualization' in resolved:
		_validate_mae_training_visualization(
			_required_mapping(resolved, 'visualization'),
			output_root=output_root,
		)

	_merge_section_defaults(resolved, 'data', FIXED_DATA_CONTRACT)
	_merge_section_defaults(resolved, 'model', FIXED_MODEL_CONTRACT)
	_merge_section_defaults(resolved, 'masking', FIXED_MASKING_CONTRACT)
	_merge_section_defaults(resolved, 'loss', FIXED_LOSS_CONTRACT)
	return resolved


def resolve_strat_hmm_pretext_config(config: _T) -> Config:
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
		if (
			target_representation
			in {
				_STRAT_HMM_LATERAL_TARGET_REPRESENTATION,
				_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION,
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
		)
	else:
		_validate_strat_hmm_pretext_cross_section_values(pseudo_targets, head)
		_validate_strat_hmm_pretext_identity(resolved, multi_head=False)
	_validate_strat_hmm_pretext_train(train)
	_validate_zero_mask(_required_mapping(resolved, 'zero_mask'))
	_validate_artifact_output_path(
		output_root,
		'paths.output_root',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_pretraining_path(
		output_root,
		'paths.output_root',
		artifact_root=paths.artifact_root,
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


def _validate_strat_hmm_pretext_sections(config: Mapping[str, object]) -> None:
	for section, allowed in _STRAT_HMM_PRETEXT_SECTION_KEYS.items():
		value = config.get(section)
		if not isinstance(value, Mapping):
			continue
		_validate_allowed_keys(value, allowed, prefix=section)


def _validate_strat_hmm_pretext_identity(  # noqa: C901, PLR0912, PLR0915
	config: Mapping[str, object],
	*,
	multi_head: bool,
	manifest_sha256: str | None = None,
	manifest: Mapping[str, object] | None = None,
	target_representation: str | None = None,
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


def _expected_or_record_multi_head_scientific_identity(
	scientific: dict[str, object],
	*,
	config: Mapping[str, object],
	manifest: Mapping[str, object],
	target_representation: str | None,
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
	for key, expected_value in expected.items():
		if key in scientific:
			if scientific[key] != expected_value:
				raise ValueError(
					f'identity.scientific_identity.{key} does not match the '
					'resolved scientific setting'
				)
		else:
			scientific[key] = deepcopy(expected_value)


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
	}:
		supported = [
			_STRAT_HMM_HARD_TARGET_REPRESENTATION,
			_STRAT_HMM_POSTERIOR_TARGET_REPRESENTATION,
			_STRAT_HMM_LATERAL_TARGET_REPRESENTATION,
			_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_TARGET_REPRESENTATION,
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
	else:
		xy_neighbor_consensus_targets = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets'
		)
		loader_name = 'load_multi_head_xy_neighbor_consensus_target_manifest'
		load_manifest = getattr(xy_neighbor_consensus_targets, loader_name)
		manifest = load_manifest(
			str(pseudo_targets['manifest']), validate_array_semantics=False
		)
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
	*,
	output_root: Path,
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
	_validate_mae_debug_general_fields(mae_debug, output_root=output_root)
	_validate_mae_debug_triggers(mae_debug)
	_validate_mae_debug_rendering_fields(mae_debug)


def _validate_mae_debug_general_fields(
	mae_debug: Mapping[str, object],
	*,
	output_root: Path,
) -> None:
	if 'enabled' in mae_debug:
		_validate_bool(mae_debug, 'enabled', prefix='visualization.mae_debug')
	if 'output_dir' in mae_debug:
		_validate_optional_output_path_under_root(
			mae_debug,
			'output_dir',
			prefix='visualization.mae_debug',
			root=output_root,
			root_label='paths.output_root',
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


__all__ = ['resolve_mae_training_config', 'resolve_strat_hmm_pretext_config']
