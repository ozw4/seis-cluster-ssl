"""Validation and resolution for MAE pretraining configs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, TypeAlias, TypeVar

if TYPE_CHECKING:
	from pathlib import Path

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
	SUPPORTED_FINITE_CHECK_MODES,
	SUPPORTED_RECONSTRUCTION_LOSSES,
	SUPPORTED_RUNTIME_CHECK_MODES,
	SUPPORTED_TARGET_NORMALIZATION_MODES,
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
	'pseudo_targets': frozenset({'input_dir', 'k', 'min_confidence'}),
	'teacher': frozenset({'checkpoint'}),
	'student': frozenset({'init_checkpoint', 'unfreeze_top_blocks'}),
	'head': frozenset(
		{'num_prototypes', 'projection_dim', 'temperature', 'normalize'},
	),
	'loss': frozenset(
		{
			'prototype_weight',
			'usage_weight',
			'entropy_floor',
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
	_validate_strat_hmm_pretext_pseudo_targets(pseudo_targets)
	_validate_strat_hmm_pretext_teacher(teacher)
	_validate_strat_hmm_pretext_student(
		student,
		encoder_depth=int(model['encoder_depth']),
	)
	_validate_strat_hmm_pretext_head(head)
	_validate_strat_hmm_pretext_loss(
		loss,
		unfreeze_top_blocks=int(student['unfreeze_top_blocks']),
	)
	_validate_strat_hmm_pretext_cross_section_values(pseudo_targets, head)
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


def _validate_strat_hmm_pretext_sections(config: Mapping[str, object]) -> None:
	for section, allowed in _STRAT_HMM_PRETEXT_SECTION_KEYS.items():
		value = config.get(section)
		if not isinstance(value, Mapping):
			continue
		_validate_allowed_keys(value, allowed, prefix=section)


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
) -> None:
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


def _validate_strat_hmm_pretext_head(head: Mapping[str, object]) -> None:
	_validate_positive_int(head, 'num_prototypes', prefix='head')
	if head.get('projection_dim') is not None:
		_validate_positive_int(head, 'projection_dim', prefix='head')
	_validate_positive_finite_number(head, 'temperature', prefix='head')
	_validate_bool(head, 'normalize', prefix='head')


def _validate_strat_hmm_pretext_loss(
	loss: Mapping[str, object],
	*,
	unfreeze_top_blocks: int,
) -> None:
	for key in ('prototype_weight', 'usage_weight', 'distillation_weight'):
		_validate_nonnegative_finite_number(loss, key, prefix='loss')
	if not any(
		float(loss[key]) > 0.0
		for key in ('prototype_weight', 'usage_weight', 'distillation_weight')
	):
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


def _validate_strat_hmm_pretext_cross_section_values(
	pseudo_targets: Mapping[str, object],
	head: Mapping[str, object],
) -> None:
	if int(pseudo_targets['k']) != int(head['num_prototypes']):
		msg = (
			'pseudo_targets.k must equal head.num_prototypes; '
			f"got {pseudo_targets['k']!r} and {head['num_prototypes']!r}"
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
	for key in ('shuffle', 'allow_overwrite_output'):
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
			f"got {amplitude_agc.get('mode')!r}"
		)
		raise ValueError(msg)
	_validate_positive_int(amplitude_agc, 'window_z', prefix='data.amplitude_agc')
	if int(amplitude_agc['window_z']) % 2 == 0:
		msg = (
			'data.amplitude_agc.window_z must be odd; '
			f"got {amplitude_agc['window_z']!r}"
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
		crop % patch != 0
		for crop, patch in zip(crop_size, patch_size, strict=True)
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
