"""Strict source audit for the F3 LocalBT/HMM K6 comparison."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.stratigraphy import (
	StratPseudoTargetInput,
	discover_pseudo_target_inputs,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint

F3LocalBTSourceRole = Literal['candidate', 'control']

CONTROL_EPOCH = 3
CONTROL_GLOBAL_STEP = 1_875
CANDIDATE_EPOCH = 25
CANDIDATE_GLOBAL_STEP = 15_625
LOCAL_BT_METHOD = 'local_barlow_twins_3d'
HMM_MODEL_TAG = 'f3_local_bt_nr_rot90_asym_g060_3ep_hmm_k6_full_25ep'

_SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')
_BASE_AUGMENTATION_CONTRACT: Mapping[str, object] = {
	'policy': 'xy_rot90_asymmetric_noise_v1',
	'gaussian_noise_std': 0.6,
}
_BASE_OBJECTIVE_CONTRACT: Mapping[str, object] = {
	'method': LOCAL_BT_METHOD,
	'local_pairs_per_crop': 128,
	'projector_dim': 384,
	'redundancy_weight': 0.005,
	'normalization_eps': 1.0e-4,
	'positive_window_tokens': [2, 2, 1],
}
_BASE_TRAIN_CONTRACT: Mapping[str, object] = {
	'batch_size': 16,
	'samples_per_epoch': 10_000,
	'epochs': CONTROL_EPOCH,
	'lr': 1.0e-4,
	'weight_decay': 0.05,
	'amp': False,
	'seed': 42,
	'grad_clip_norm': 1.0,
	'max_steps': None,
}
_HMM_HEAD_CONTRACT: Mapping[str, object] = {
	'num_prototypes': 6,
	'projection_dim': 128,
	'temperature': 0.1,
	'normalize': True,
}
_HMM_LOSS_CONTRACT: Mapping[str, object] = {
	'valid_mask_mode': 'voxel',
	'prototype_weight': 1.0,
	'usage_weight': 0.005,
	'entropy_floor': None,
	'distillation_weight': 0.2,
}
_HMM_TRAIN_CONTRACT: Mapping[str, object] = {
	'batch_size': 16,
	'samples_per_epoch': 10_000,
	'epochs': CANDIDATE_EPOCH,
	'lr': 1.0e-5,
	'encoder_lr': 1.0e-5,
	'weight_decay': 0.05,
	'amp': False,
	'seed': 42,
	'grad_clip_norm': 1.0,
	'max_steps': None,
}
_PSEUDO_BOUNDARY_CONTRACT: Mapping[str, object] = {
	'method': 'transition_distance_exponential',
	'alpha': 0.0,
	'tau': 1.0,
}
_CLUSTER_RECIPE_CONTRACT: Mapping[str, object] = {
	'method': 'stratigraphic_hmm_kmeans',
	'k': 6,
	'k_values': [6],
	'normalization': 'l2',
	'random_seed': 42,
	'emission_source': 'embedding',
}
_RESIDUALIZATION_CONTRACT: Mapping[str, object] = {
	'enabled': True,
	'mode': 'local_token_position',
	'group_by': 'token_phase',
	'add_global_mean_back': True,
	'min_group_count': 32,
}
_PCA_CONTRACT: Mapping[str, object] = {
	'enabled': True,
	'n_components': 64,
	'effective_n_components': 64,
	'whiten': False,
}
_HMM_TRANSITION_CONTRACT: Mapping[str, object] = {
	'same_cost': 0.03,
	'advance_cost': 0.0,
	'jump_cost': 1.0,
	'reverse_cost': 1.0e6,
	'forbid_reverse': True,
	'max_jump': 1,
}
_ENCODER_BLOCK_PARAMETER_SUFFIXES = frozenset(
	{
		'norm1.weight',
		'norm1.bias',
		'attention.in_proj_weight',
		'attention.in_proj_bias',
		'attention.out_proj.weight',
		'attention.out_proj.bias',
		'norm2.weight',
		'norm2.bias',
		'mlp.0.weight',
		'mlp.0.bias',
		'mlp.3.weight',
		'mlp.3.bias',
	}
)
_PROTOTYPE_PARAMETER_NAMES = frozenset(
	{'prototypes', 'projection.weight', 'projection.bias'}
)


def validate_f3_local_bt_candidate_source(  # noqa: PLR0913
	checkpoint: str | Path,
	embedding_metadata: str | Path,
	*,
	role: F3LocalBTSourceRole,
	base_training_config: str | Path,
	stratigraphy_training_config: str | Path | None,
	epoch: int,
	global_step: int,
) -> dict[str, object]:
	"""Validate one completed source for the fixed 3ep-control/25ep-HMM pair."""
	checkpoint_path = _absolute_file(checkpoint, 'checkpoint')
	metadata_path = _absolute_file(embedding_metadata, 'embedding metadata')
	base_config_path = _absolute_file(base_training_config, 'base training config')
	_validate_declared_expectation(
		role=role,
		stratigraphy_training_config=stratigraphy_training_config,
		epoch=epoch,
		global_step=global_step,
	)

	base_config = resolve_barlow_twins_training_config(load_config(base_config_path))
	_validate_base_recipe(base_config)
	stratigraphy_config: Mapping[str, object] | None = None
	stratigraphy_config_path: Path | None = None
	if role == 'candidate':
		stratigraphy_config_path = _absolute_file(
			stratigraphy_training_config,
			'stratigraphy training config',
		)
		stratigraphy_config = resolve_strat_hmm_pretext_config(
			load_config(stratigraphy_config_path)
		)
		_validate_hmm_recipe(stratigraphy_config, base_config=base_config)
	_validate_checkpoint_location(
		checkpoint_path,
		role=role,
		base_config=base_config,
		stratigraphy_config=stratigraphy_config,
	)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'{checkpoint_path} checkpoint payload must be a mapping')
	_validate_checkpoint_payload(
		payload,
		path=checkpoint_path,
		role=role,
		base_config=base_config,
		stratigraphy_config=stratigraphy_config,
		epoch=epoch,
		global_step=global_step,
	)
	control_identity = (
		_validate_hmm_control_identity(
			payload,
			path=checkpoint_path,
			base_config=base_config,
			stratigraphy_config=_mapping(
				stratigraphy_config, 'resolved stratigraphy training config'
			),
		)
		if role == 'candidate'
		else None
	)
	pseudo_target_science = (
		_validate_pseudo_target_science(
			base_config=base_config,
			stratigraphy_config=_mapping(
				stratigraphy_config, 'resolved stratigraphy training config'
			),
		)
		if role == 'candidate'
		else None
	)
	metadata = _read_json(metadata_path, label='embedding metadata')
	_validate_embedding_metadata(
		metadata,
		path=metadata_path,
		checkpoint=checkpoint_path,
		checkpoint_payload=payload,
		role=role,
		base_config=base_config,
		stratigraphy_config=stratigraphy_config,
		control_identity=control_identity,
	)

	lineage: dict[str, object] = {
		'schema_version': 1,
		'role': role,
		'checkpoint': {
			'path': str(checkpoint_path),
			'sha256': file_sha256(checkpoint_path),
			'epoch': epoch,
			'global_step': global_step,
			'completed_epoch': True,
		},
		'embedding_metadata': {
			'path': str(metadata_path),
			'sha256': file_sha256(metadata_path),
		},
		'base_training_config': {
			'path': str(base_config_path),
			'sha256': file_sha256(base_config_path),
			'resolved_sha256': _canonical_sha256(base_config),
		},
	}
	if stratigraphy_config_path is not None and stratigraphy_config is not None:
		lineage['stratigraphy_training_config'] = {
			'path': str(stratigraphy_config_path),
			'sha256': file_sha256(stratigraphy_config_path),
			'resolved_sha256': _canonical_sha256(stratigraphy_config),
		}
		lineage['model_tag'] = HMM_MODEL_TAG
		lineage['pseudo_target_science'] = pseudo_target_science
	return lineage


def validate_f3_local_bt_pseudo_target_science(
	*,
	base_training_config: str | Path,
	stratigraphy_training_config: str | Path,
) -> list[dict[str, object]]:
	"""Validate the fixed pseudo-target chain without requiring the final model."""
	base_config_path = _absolute_file(base_training_config, 'base training config')
	stratigraphy_config_path = _absolute_file(
		stratigraphy_training_config,
		'stratigraphy training config',
	)
	base_config = resolve_barlow_twins_training_config(load_config(base_config_path))
	stratigraphy_config = resolve_strat_hmm_pretext_config(
		load_config(stratigraphy_config_path)
	)
	_validate_base_recipe(base_config)
	_validate_hmm_recipe(stratigraphy_config, base_config=base_config)
	return _validate_pseudo_target_science(
		base_config=base_config,
		stratigraphy_config=stratigraphy_config,
	)


def _validate_declared_expectation(
	*,
	role: object,
	stratigraphy_training_config: object,
	epoch: object,
	global_step: object,
) -> None:
	if role not in {'candidate', 'control'}:
		raise ValueError("role must be 'candidate' or 'control'")
	if role == 'control':
		if stratigraphy_training_config is not None:
			raise ValueError('control must not declare a stratigraphy training config')
		expected = (CONTROL_EPOCH, CONTROL_GLOBAL_STEP)
	else:
		if stratigraphy_training_config is None:
			raise ValueError('candidate requires a stratigraphy training config')
		expected = (CANDIDATE_EPOCH, CANDIDATE_GLOBAL_STEP)
	if (epoch, global_step) != expected:
		raise ValueError(
			f'{role} expectation must be epoch/global_step='
			f'{expected[0]}/{expected[1]}; '
			f'got {epoch!r}/{global_step!r}'
		)


def _validate_base_recipe(config: Mapping[str, object]) -> None:
	if config.get('stage') != 'barlow_twins_training':
		raise ValueError('base training config must resolve as barlow_twins_training')
	if 'continuation' in config:
		raise ValueError('3ep LocalBT control must not be a continuation')
	_expect_equal(
		config.get('augmentations'),
		_BASE_AUGMENTATION_CONTRACT,
		'base training config augmentations',
	)
	_expect_equal(
		config.get('barlow_twins'),
		_BASE_OBJECTIVE_CONTRACT,
		'base training config barlow_twins',
	)
	_expect_fields(
		config.get('train'),
		_BASE_TRAIN_CONTRACT,
		'base training config train',
	)


def _validate_hmm_recipe(
	config: Mapping[str, object], *, base_config: Mapping[str, object]
) -> None:
	if config.get('stage') != 'train_strat_hmm_pretext':
		raise ValueError(
			'stratigraphy training config must resolve as train_strat_hmm_pretext'
		)
	for section in ('data', 'model', 'zero_mask'):
		if config.get(section) != base_config.get(section):
			raise ValueError(
				f'stratigraphy training config {section} must match the base recipe'
			)
	identity = _required_mapping(config, 'identity', 'stratigraphy training config')
	if identity != {'model_tag': HMM_MODEL_TAG}:
		raise ValueError(f'stratigraphy identity must be exactly {HMM_MODEL_TAG!r}')
	pseudo_targets = _required_mapping(
		config, 'pseudo_targets', 'stratigraphy training config'
	)
	_expect_fields(
		pseudo_targets,
		{'k': 6, 'min_confidence': 0.0},
		'stratigraphy training config pseudo_targets',
	)
	student = _required_mapping(config, 'student', 'stratigraphy training config')
	_expect_fields(
		student,
		{'unfreeze_top_blocks': 1},
		'stratigraphy training config student',
	)
	_expect_equal(
		config.get('head'),
		_HMM_HEAD_CONTRACT,
		'stratigraphy training config head',
	)
	_expect_equal(
		config.get('loss'),
		_HMM_LOSS_CONTRACT,
		'stratigraphy training config loss',
	)
	_expect_fields(
		config.get('train'),
		_HMM_TRAIN_CONTRACT,
		'stratigraphy training config train',
	)
	base_checkpoint = _expected_latest_checkpoint(base_config)
	teacher = _required_mapping(config, 'teacher', 'stratigraphy training config')
	for value, label in (
		(teacher.get('checkpoint'), 'teacher.checkpoint'),
		(student.get('init_checkpoint'), 'student.init_checkpoint'),
	):
		if not _paths_equal(value, base_checkpoint):
			raise ValueError(
				f'stratigraphy training config {label} must select the 3ep control'
			)


def _validate_checkpoint_location(
	checkpoint: Path,
	*,
	role: F3LocalBTSourceRole,
	base_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object] | None,
) -> None:
	config = (
		base_config
		if role == 'control'
		else _mapping(stratigraphy_config, 'resolved stratigraphy training config')
	)
	expected = _expected_latest_checkpoint(config)
	if checkpoint.resolve(strict=False) != expected.resolve(strict=False):
		raise ValueError(
			f'{role} checkpoint must be the configured completed latest.pt: {expected}'
		)


def _validate_checkpoint_payload(  # noqa: C901, PLR0913
	payload: Mapping[str, object],
	*,
	path: Path,
	role: F3LocalBTSourceRole,
	base_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object] | None,
	epoch: int,
	global_step: int,
) -> None:
	if payload.get('epoch') != epoch or payload.get('global_step') != global_step:
		raise ValueError(
			f'{path} must record epoch/global_step={epoch}/{global_step}; got '
			f'{payload.get("epoch")!r}/{payload.get("global_step")!r}'
		)
	expected_checkpoint_config = _expected_checkpoint_base_config(
		role=role,
		base_config=base_config,
		stratigraphy_config=stratigraphy_config,
	)
	if payload.get('config') != expected_checkpoint_config:
		raise ValueError(
			f'{path} config does not exactly match the resolved base training '
			'config with the checkpoint output root'
		)
	model_state = payload.get('model_state_dict')
	if not isinstance(model_state, Mapping) or not model_state:
		raise TypeError(f'{path} model_state_dict must be a non-empty mapping')
	training_state = _required_mapping(payload, 'training_state', str(path))
	if role == 'control':
		if 'stratigraphy_config' in payload or 'control_identity' in payload:
			raise ValueError(
				f'{path} 3ep control must not contain stratigraphy lineage'
			)
		if payload.get('pretraining_method') != LOCAL_BT_METHOD:
			raise ValueError(f'{path} pretraining_method must be {LOCAL_BT_METHOD!r}')
		if payload.get('checkpoint_kind') != 'barlow_twins_pretraining':
			raise ValueError(f'{path} is not a LocalBT pretraining checkpoint')
		if not isinstance(payload.get('projector_state_dict'), Mapping):
			raise TypeError(f'{path} projector_state_dict must be a mapping')
		if (
			training_state.get('stage') != 'barlow_twins_training'
			or training_state.get('resume_boundary') != 'epoch'
			or training_state.get('completed_epoch') is not True
			or training_state.get('dataset_epoch') != epoch - 1
		):
			raise ValueError(f'{path} does not record a completed LocalBT epoch')
		return
	if payload.get('stratigraphy_config') != stratigraphy_config:
		raise ValueError(
			f'{path} stratigraphy_config does not exactly match the resolved HMM config'
		)
	if not isinstance(payload.get('stratigraphy_state_dict'), Mapping):
		raise TypeError(f'{path} stratigraphy_state_dict must be a mapping')
	if (
		training_state.get('stage') != 'train_strat_hmm_pretext'
		or training_state.get('checkpoint_kind') != 'epoch'
		or training_state.get('batch_index') is not None
	):
		raise ValueError(f'{path} does not record a completed HMM epoch')


def _expected_checkpoint_base_config(
	*,
	role: F3LocalBTSourceRole,
	base_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object] | None,
) -> Mapping[str, object]:
	if role == 'control':
		return base_config
	hmm_config = _mapping(stratigraphy_config, 'resolved stratigraphy training config')
	hmm_paths = _required_mapping(hmm_config, 'paths', 'stratigraphy training config')
	hmm_output_root = _required_absolute_path(
		hmm_paths.get('output_root'), 'stratigraphy training config paths.output_root'
	)
	base_paths = _required_mapping(base_config, 'paths', 'base training config')
	expected_paths = dict(base_paths)
	expected_paths['output_root'] = str(hmm_output_root)
	expected = dict(base_config)
	expected['paths'] = expected_paths
	return expected


def _validate_hmm_control_identity(
	payload: Mapping[str, object],
	*,
	path: Path,
	base_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
) -> Mapping[str, object]:
	control = _required_mapping(payload, 'control_identity', str(path))
	if control.get('schema_version') != 1:
		raise ValueError(f'{path} control_identity.schema_version must be 1')
	if control.get('model_tag') != HMM_MODEL_TAG:
		raise ValueError(f'{path} control_identity.model_tag is incorrect')
	identity = _required_mapping(
		stratigraphy_config, 'identity', 'resolved stratigraphy config'
	)
	if control.get('scientific_identity') != identity.get('scientific_identity', {}):
		raise ValueError(f'{path} control_identity.scientific_identity is incorrect')
	runtime = _required_mapping(control, 'runtime_identity', f'{path} control_identity')
	data = _required_mapping(
		stratigraphy_config, 'data', 'resolved stratigraphy config'
	)
	if runtime.get('finite_check_mode') != data.get('finite_check_mode'):
		raise ValueError(
			f'{path} control_identity runtime finite_check_mode is incorrect'
		)
	_required_sha256(runtime.get('git_diff_sha256'), f'{path} runtime git diff')
	if control.get('resolved_training_config_sha256') != _canonical_sha256(
		stratigraphy_config
	):
		raise ValueError(
			f'{path} control_identity resolved training config SHA is incorrect'
		)
	inputs = _required_mapping(control, 'input_identities', f'{path} control_identity')
	base_checkpoint = _expected_latest_checkpoint(base_config)
	_validate_file_identity(
		inputs.get('teacher_checkpoint'),
		expected_path=base_checkpoint,
		label=f'{path} control_identity teacher checkpoint',
	)
	_validate_file_identity(
		inputs.get('student_init_checkpoint'),
		expected_path=base_checkpoint,
		label=f'{path} control_identity student init checkpoint',
	)
	_validate_pseudo_target_identities(
		inputs.get('pseudo_targets'),
		path=path,
		stratigraphy_config=stratigraphy_config,
	)
	_validate_initial_parameter_identities(
		control,
		path=path,
		base_checkpoint=base_checkpoint,
		stratigraphy_config=stratigraphy_config,
		payload=payload,
	)
	return control


def _validate_pseudo_target_identities(  # noqa: C901
	value: object,
	*,
	path: Path,
	stratigraphy_config: Mapping[str, object],
) -> None:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{path} control_identity pseudo_targets must be a list')
	recorded: dict[str, Mapping[str, object]] = {}
	for index, item in enumerate(value):
		if not isinstance(item, Mapping):
			raise TypeError(
				f'{path} control_identity pseudo_targets[{index}] must be a mapping'
			)
		survey_id = item.get('survey_id')
		if not isinstance(survey_id, str) or not survey_id:
			raise ValueError(
				f'{path} control_identity pseudo_targets[{index}] survey_id is invalid'
			)
		if survey_id in recorded:
			raise ValueError(f'{path} has duplicate pseudo-target survey {survey_id!r}')
		recorded[survey_id] = item
	pseudo = _required_mapping(
		stratigraphy_config, 'pseudo_targets', 'resolved stratigraphy config'
	)
	current = discover_pseudo_target_inputs(
		_required_absolute_path(
			pseudo.get('input_dir'), 'stratigraphy pseudo_targets.input_dir'
		),
		k=_required_positive_int(pseudo.get('k'), 'stratigraphy pseudo_targets.k'),
	)
	current_by_survey = {item.survey_id: item for item in current}
	if set(recorded) != set(current_by_survey):
		raise ValueError(
			f'{path} control_identity pseudo-target survey set is incorrect'
		)
	for survey_id, target in current_by_survey.items():
		item = recorded[survey_id]
		for field, current_path in (
			('labels', target.labels_path),
			('confidence', target.confidence_path),
			('valid_tokens', target.valid_tokens_path),
			('metadata', target.metadata_path),
		):
			_validate_file_identity(
				item.get(field),
				expected_path=current_path,
				label=f'{path} pseudo-target {survey_id} {field}',
			)
		boundary_present = target.boundary_weight_path is not None
		if item.get('boundary_weight_present') is not boundary_present:
			raise ValueError(
				f'{path} pseudo-target {survey_id} boundary presence is incorrect'
			)
		if boundary_present:
			_validate_file_identity(
				item.get('boundary_weight'),
				expected_path=target.boundary_weight_path,
				label=f'{path} pseudo-target {survey_id} boundary_weight',
			)
		elif 'boundary_weight' in item:
			raise ValueError(
				f'{path} pseudo-target {survey_id} has unexpected boundary identity'
			)


def _validate_pseudo_target_science(
	*,
	base_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
) -> list[dict[str, object]]:
	pseudo = _required_mapping(
		stratigraphy_config, 'pseudo_targets', 'resolved stratigraphy config'
	)
	k = _required_positive_int(pseudo.get('k'), 'stratigraphy pseudo_targets.k')
	targets = discover_pseudo_target_inputs(
		_required_absolute_path(
			pseudo.get('input_dir'), 'stratigraphy pseudo_targets.input_dir'
		),
		k=k,
	)
	base_checkpoint = _absolute_file(
		_expected_latest_checkpoint(base_config), 'current 3ep base checkpoint'
	)
	base_payload = load_checkpoint(base_checkpoint, map_location='cpu')
	if not isinstance(base_payload, Mapping):
		raise TypeError(f'{base_checkpoint} checkpoint payload must be a mapping')
	_validate_checkpoint_payload(
		base_payload,
		path=base_checkpoint,
		role='control',
		base_config=base_config,
		stratigraphy_config=None,
		epoch=CONTROL_EPOCH,
		global_step=CONTROL_GLOBAL_STEP,
	)
	return [
		_validate_one_pseudo_target_science(
			target,
			k=k,
			base_config=base_config,
			base_checkpoint=base_checkpoint,
		)
		for target in targets
	]


def _validate_one_pseudo_target_science(
	target: StratPseudoTargetInput,
	*,
	k: int,
	base_config: Mapping[str, object],
	base_checkpoint: Path,
) -> dict[str, object]:
	label = f'pseudo-target {target.survey_id}'
	metadata_path = _absolute_file(target.metadata_path, f'{label} metadata')
	metadata = _read_json(metadata_path, label=f'{label} metadata')
	_expect_science_fields(
		metadata,
		{
			'artifact_type': 'strat_hmm_pseudo_target',
			'schema_version': 2,
			'k': k,
			'survey_id': target.survey_id,
			'boundary_weight_source': 'explicit',
		},
		f'{label} metadata',
	)
	source = _required_mapping(metadata, 'source', f'{label} metadata')
	_expect_science_fields(
		source,
		{
			'source_method': 'stratigraphic_hmm_kmeans',
			'export_confidence': 1.0,
		},
		f'{label} source',
	)
	boundary = _required_mapping(source, 'boundary_weighting', f'{label} source')
	_expect_science_fields(
		boundary,
		_PSEUDO_BOUNDARY_CONTRACT,
		f'{label} boundary weighting',
	)
	clustering_root = _required_absolute_path(
		source.get('source_clustering_output_dir'),
		f'{label} source clustering output directory',
	)
	if not clustering_root.is_dir():
		raise FileNotFoundError(
			f'{label} source clustering output directory does not exist: '
			f'{clustering_root}'
		)
	expected_label_path = (
		clustering_root
		/ 'labels'
		/ f'k{k}'
		/ f'{target.survey_id}.cluster_labels_token.npy'
	)
	source_label_path = _absolute_file(
		source.get('source_label_path'), f'{label} source labels'
	)
	if source_label_path.resolve() != expected_label_path.resolve(strict=False):
		raise ValueError(f'{label} source label path is not derived from its root')
	source_label_sha = _required_sha256(
		source.get('source_label_sha256'), f'{label} source label SHA'
	)
	if source_label_sha != file_sha256(source_label_path):
		raise ValueError(f'{label} source label SHA does not match the current file')
	if source_label_sha != file_sha256(target.labels_path):
		raise ValueError(f'{label} exported labels differ from the source labels')
	source_labels = np.load(source_label_path, mmap_mode='r', allow_pickle=False)
	pseudo_valid_tokens = np.load(
		target.valid_tokens_path, mmap_mode='r', allow_pickle=False
	)
	if source_labels.shape != pseudo_valid_tokens.shape or not np.array_equal(
		pseudo_valid_tokens, source_labels >= 0
	):
		raise ValueError(f'{label} valid_tokens must equal source cluster labels >= 0')
	expected_label_metadata_path = source_label_path.with_name(
		f'{target.survey_id}.cluster_label_metadata.json'
	)
	label_metadata_path = _absolute_file(
		source.get('source_metadata_path'), f'{label} cluster label metadata'
	)
	if label_metadata_path.resolve() != expected_label_metadata_path.resolve(
		strict=False
	):
		raise ValueError(
			f'{label} cluster label metadata path is not derived from its labels'
		)
	label_metadata_sha = _required_sha256(
		source.get('source_metadata_sha256'), f'{label} cluster label metadata SHA'
	)
	if label_metadata_sha != file_sha256(label_metadata_path):
		raise ValueError(
			f'{label} cluster label metadata SHA does not match the current file'
		)
	label_metadata = _read_json(
		label_metadata_path, label=f'{label} cluster label metadata'
	)
	model_metadata_path = _absolute_file(
		clustering_root / 'models' / f'k{k}' / 'clustering_metadata.json',
		f'{label} clustering model metadata',
	)
	model_metadata = _read_json(
		model_metadata_path, label=f'{label} clustering model metadata'
	)
	embedding_metadata_path = _validate_cluster_label_metadata(
		label_metadata,
		path=label_metadata_path,
		survey_id=target.survey_id,
		label_path=source_label_path,
		base_checkpoint=base_checkpoint,
	)
	_validate_clustering_model_metadata(
		model_metadata,
		path=model_metadata_path,
		survey_id=target.survey_id,
		label_path=source_label_path,
		label_metadata_path=label_metadata_path,
		embedding_metadata_path=embedding_metadata_path,
		base_checkpoint=base_checkpoint,
	)
	_validate_target_embedding_metadata(
		_read_json(
			embedding_metadata_path,
			label=f'{label} target embedding metadata',
		),
		path=embedding_metadata_path,
		base_config=base_config,
		base_checkpoint=base_checkpoint,
	)
	return {
		'survey_id': target.survey_id,
		'pseudo_target_metadata': _path_identity(metadata_path),
		'source_labels': _path_identity(source_label_path),
		'cluster_label_metadata': _path_identity(label_metadata_path),
		'clustering_metadata': _path_identity(model_metadata_path),
		'target_embedding_metadata': _path_identity(embedding_metadata_path),
		'base_checkpoint': _path_identity(base_checkpoint),
	}


def _validate_cluster_label_metadata(
	metadata: Mapping[str, object],
	*,
	path: Path,
	survey_id: str,
	label_path: Path,
	base_checkpoint: Path,
) -> Path:
	_validate_cluster_recipe(metadata, label=str(path), base_checkpoint=base_checkpoint)
	_expect_science_fields(
		metadata,
		{'survey_id': survey_id},
		str(path),
	)
	if not _paths_equal(metadata.get('label_path'), label_path):
		raise ValueError(f'{path} label_path does not match the source labels')
	embedding_input = _required_mapping(metadata, 'embedding_input', str(path))
	embedding_inputs = _required_mapping_sequence(
		metadata.get('embedding_inputs'), f'{path}.embedding_inputs'
	)
	if len(embedding_inputs) != 1 or embedding_inputs[0] != embedding_input:
		raise ValueError(f'{path} must record one matching embedding input')
	return _validate_embedding_input_identity(
		embedding_input,
		label=f'{path} embedding input',
		survey_id=survey_id,
		base_checkpoint=base_checkpoint,
	)


def _validate_clustering_model_metadata(  # noqa: PLR0913
	metadata: Mapping[str, object],
	*,
	path: Path,
	survey_id: str,
	label_path: Path,
	label_metadata_path: Path,
	embedding_metadata_path: Path,
	base_checkpoint: Path,
) -> None:
	_validate_cluster_recipe(metadata, label=str(path), base_checkpoint=base_checkpoint)
	embedding_inputs = _required_mapping_sequence(
		metadata.get('embedding_inputs'), f'{path}.embedding_inputs'
	)
	if len(embedding_inputs) != 1:
		raise ValueError(f'{path} must record exactly one embedding input')
	model_embedding_metadata = _validate_embedding_input_identity(
		embedding_inputs[0],
		label=f'{path} embedding input',
		survey_id=survey_id,
		base_checkpoint=base_checkpoint,
	)
	if model_embedding_metadata.resolve() != embedding_metadata_path.resolve():
		raise ValueError(f'{path} embedding input differs from label metadata')
	surveys = _required_mapping_sequence(metadata.get('surveys'), f'{path}.surveys')
	if len(surveys) != 1:
		raise ValueError(f'{path} must record exactly one survey')
	survey = surveys[0]
	_expect_science_fields(survey, {'survey_id': survey_id}, f'{path} survey')
	for value, expected, field in (
		(survey.get('label_path'), label_path, 'label_path'),
		(survey.get('label_metadata_path'), label_metadata_path, 'label_metadata_path'),
	):
		if not _paths_equal(value, expected):
			raise ValueError(f'{path} survey {field} is incorrect')
	hmm = _required_mapping(metadata, 'stratigraphic_hmm', str(path))
	_expect_science_fields(
		hmm,
		{
			'iterations': 10,
			'z_axis': 2,
			'z_direction': 'increasing_downward',
			'edge_margin_tokens': [8, 8, 0],
			'emission_source': 'embedding',
		},
		f'{path} stratigraphic_hmm',
	)
	_expect_science_fields(
		_required_mapping(hmm, 'transition', f'{path} stratigraphic_hmm'),
		_HMM_TRANSITION_CONTRACT,
		f'{path} stratigraphic_hmm.transition',
	)
	path_prior = _required_mapping(hmm, 'path_prior', f'{path} stratigraphic_hmm')
	_expect_science_fields(
		path_prior,
		{'enabled': True},
		f'{path} stratigraphic_hmm.path_prior',
	)
	_expect_science_fields(
		_required_mapping(
			path_prior, 'initial_state', f'{path} stratigraphic_hmm.path_prior'
		),
		{'mode': 'shallow_anchor', 'weight': 0.25},
		f'{path} stratigraphic_hmm.path_prior.initial_state',
	)
	_expect_science_fields(
		_required_mapping(
			path_prior, 'terminal_state', f'{path} stratigraphic_hmm.path_prior'
		),
		{'mode': 'deep_anchor', 'weight': 0.25},
		f'{path} stratigraphic_hmm.path_prior.terminal_state',
	)
	_expect_science_fields(
		_required_mapping(
			path_prior,
			'expected_boundaries',
			f'{path} stratigraphic_hmm.path_prior',
		),
		{'enabled': False},
		f'{path} stratigraphic_hmm.path_prior.expected_boundaries',
	)
	_expect_science_fields(
		_required_mapping(hmm, 'init', f'{path} stratigraphic_hmm'),
		{'order_by': 'mean_z'},
		f'{path} stratigraphic_hmm.init',
	)
	_expect_science_fields(
		_required_mapping(hmm, 'update', f'{path} stratigraphic_hmm'),
		{'empty_cluster_policy': 'keep_previous'},
		f'{path} stratigraphic_hmm.update',
	)


def _validate_cluster_recipe(
	metadata: Mapping[str, object], *, label: str, base_checkpoint: Path
) -> None:
	_expect_science_fields(metadata, _CLUSTER_RECIPE_CONTRACT, label)
	compatibility = _required_mapping(
		metadata, 'embedding_compatibility_signature', label
	)
	if compatibility.get('checkpoint_sha256') != file_sha256(base_checkpoint):
		raise ValueError(f'{label} base checkpoint compatibility SHA is incorrect')
	_expect_science_fields(
		_required_mapping(metadata, 'residualization', label),
		_RESIDUALIZATION_CONTRACT,
		f'{label}.residualization',
	)
	_expect_science_fields(
		_required_mapping(metadata, 'pca', label),
		_PCA_CONTRACT,
		f'{label}.pca',
	)
	_expect_science_fields(
		_required_mapping(metadata, 'emission_features', label),
		{
			'source': 'embedding',
			'embedding_features_used_for_emissions': True,
		},
		f'{label}.emission_features',
	)


def _validate_embedding_input_identity(
	value: Mapping[str, object],
	*,
	label: str,
	survey_id: str,
	base_checkpoint: Path,
) -> Path:
	_expect_science_fields(value, {'survey_id': survey_id}, label)
	metadata_path = _absolute_file(value.get('metadata_path'), f'{label} metadata')
	expected_metadata_path = (
		metadata_path.parent / f'{survey_id}.embedding_metadata.json'
	)
	if metadata_path.resolve() != expected_metadata_path.resolve(strict=False):
		raise ValueError(
			f'{label} metadata path does not match its survey and embedding directory'
		)
	for field, suffix in (
		('embeddings_path', 'embeddings.npy'),
		('valid_tokens_path', 'valid_tokens.npy'),
	):
		artifact_path = _absolute_file(value.get(field), f'{label} {field}')
		expected_path = metadata_path.parent / f'{survey_id}.{suffix}'
		if artifact_path.resolve() != expected_path.resolve(strict=False):
			raise ValueError(
				f'{label} {field} does not match its metadata directory and survey'
			)
	recorded_sha = _required_sha256(
		value.get('metadata_sha256'), f'{label} metadata SHA'
	)
	if recorded_sha != file_sha256(metadata_path):
		raise ValueError(f'{label} metadata SHA does not match the current file')
	compatibility = _read_json(metadata_path, label=f'{label} metadata')
	if not _paths_equal(compatibility.get('checkpoint_path'), base_checkpoint):
		raise ValueError(f'{label} does not select the current 3ep base checkpoint')
	if compatibility.get('checkpoint_sha256') != file_sha256(base_checkpoint):
		raise ValueError(f'{label} base checkpoint SHA is incorrect')
	return metadata_path


def _validate_target_embedding_metadata(
	metadata: Mapping[str, object],
	*,
	path: Path,
	base_config: Mapping[str, object],
	base_checkpoint: Path,
) -> None:
	if not _paths_equal(metadata.get('checkpoint_path'), base_checkpoint):
		raise ValueError(f'{path} checkpoint_path is not the current 3ep base')
	if metadata.get('checkpoint_sha256') != file_sha256(base_checkpoint):
		raise ValueError(f'{path} checkpoint_sha256 is not the current 3ep base SHA')
	if metadata.get('pretraining_method') != LOCAL_BT_METHOD:
		raise ValueError(f'{path} pretraining_method must be {LOCAL_BT_METHOD!r}')
	_expect_equal(
		metadata.get('pretraining_objective'),
		{
			**_BASE_OBJECTIVE_CONTRACT,
			'augmentations': dict(_BASE_AUGMENTATION_CONTRACT),
		},
		f'{path} pretraining_objective',
	)
	model = _required_mapping(base_config, 'model', 'base training config')
	_expect_equal(
		metadata.get('model_geometry'),
		{'name': 'amp_mae3d', 'in_channels': 1, 'out_channels': 1, **model},
		f'{path} model_geometry',
	)
	if metadata.get('patch_size') != model.get('patch_size'):
		raise ValueError(f'{path} patch_size does not match the base model')
	data = _required_mapping(base_config, 'data', 'base training config')
	for key in ('normalized_clip_abs', 'finite_check_mode', 'amplitude_agc'):
		if metadata.get(key) != data.get(key):
			raise ValueError(f'{path} {key} does not match the base recipe')
	if metadata.get('zero_mask') != base_config.get('zero_mask'):
		raise ValueError(f'{path} zero_mask does not match the base recipe')


def _path_identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _validate_initial_parameter_identities(  # noqa: C901
	control: Mapping[str, object],
	*,
	path: Path,
	base_checkpoint: Path,
	stratigraphy_config: Mapping[str, object],
	payload: Mapping[str, object],
) -> None:
	parameter_hashes = _required_mapping(
		control, 'initial_parameter_sha256', f'{path} control_identity'
	)
	student_hashes = _required_mapping(
		parameter_hashes,
		'student_trainable',
		f'{path} initial_parameter_sha256',
	)
	head_hashes = _required_mapping(
		parameter_hashes,
		'prototype_head',
		f'{path} initial_parameter_sha256',
	)
	model = _required_mapping(stratigraphy_config, 'model', 'stratigraphy config')
	student = _required_mapping(stratigraphy_config, 'student', 'stratigraphy config')
	depth = _required_positive_int(model.get('encoder_depth'), 'model.encoder_depth')
	unfreeze = _required_positive_int(
		student.get('unfreeze_top_blocks'), 'student.unfreeze_top_blocks'
	)
	expected_student_names = {
		f'encoder.layers.{block}.{suffix}'
		for block in range(depth - unfreeze, depth)
		for suffix in _ENCODER_BLOCK_PARAMETER_SUFFIXES
	}
	if set(student_hashes) != expected_student_names:
		raise ValueError(
			f'{path} initial student trainable parameter hash set is incorrect'
		)
	if set(head_hashes) != _PROTOTYPE_PARAMETER_NAMES:
		raise ValueError(
			f'{path} initial prototype head parameter hash set is incorrect'
		)
	base_payload = load_checkpoint(base_checkpoint, map_location='cpu')
	if not isinstance(base_payload, Mapping):
		raise TypeError(f'{base_checkpoint} payload must be a mapping')
	base_state = _required_mapping(
		base_payload, 'model_state_dict', str(base_checkpoint)
	)
	for name, digest in student_hashes.items():
		_required_sha256(digest, f'{path} initial student parameter {name}')
		tensor = base_state.get(name)
		if not isinstance(tensor, torch.Tensor):
			raise TypeError(f'{base_checkpoint} is missing tensor {name!r}')
		if digest != _parameter_sha256(name, tensor):
			raise ValueError(
				f'{path} initial student parameter hash for {name!r} is incorrect'
			)
	for name, digest in head_hashes.items():
		_required_sha256(digest, f'{path} initial prototype parameter {name}')
	initial_states = _required_mapping(
		control, 'initial_state_sha256', f'{path} control_identity'
	)
	if set(initial_states) != {'student', 'head'}:
		raise ValueError(f'{path} initial state hash set is incorrect')
	student_state_sha256 = _required_sha256(
		initial_states.get('student'), f'{path} initial student state'
	)
	_required_sha256(initial_states.get('head'), f'{path} initial head state')
	if student_state_sha256 != _state_dict_sha256(base_state):
		raise ValueError(f'{path} initial student state hash is incorrect')
	trainability = _required_mapping(payload, 'trainability_summary', str(path))
	trainable_names = trainability.get('trainable_names')
	if not isinstance(trainable_names, Sequence) or isinstance(
		trainable_names, str | bytes
	):
		raise TypeError(f'{path} trainability_summary.trainable_names must be a list')
	if set(trainable_names) != expected_student_names:
		raise ValueError(f'{path} trainability summary does not match initial hashes')


def _validate_embedding_metadata(  # noqa: C901, PLR0912, PLR0913
	metadata: Mapping[str, object],
	*,
	path: Path,
	checkpoint: Path,
	checkpoint_payload: Mapping[str, object],
	role: F3LocalBTSourceRole,
	base_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object] | None,
	control_identity: Mapping[str, object] | None,
) -> None:
	if not _paths_equal(metadata.get('checkpoint_path'), checkpoint):
		raise ValueError(
			f'{path} checkpoint_path does not match the audited checkpoint'
		)
	if metadata.get('checkpoint_sha256') != file_sha256(checkpoint):
		raise ValueError(f'{path} checkpoint_sha256 does not match the checkpoint')
	if metadata.get('pretraining_method') != LOCAL_BT_METHOD:
		raise ValueError(f'{path} pretraining_method must be {LOCAL_BT_METHOD!r}')
	model = _required_mapping(base_config, 'model', 'base training config')
	expected_geometry = {
		'name': 'amp_mae3d',
		'in_channels': 1,
		'out_channels': 1,
		**model,
	}
	_expect_equal(
		metadata.get('model_geometry'), expected_geometry, f'{path} model_geometry'
	)
	if metadata.get('patch_size') != model.get('patch_size'):
		raise ValueError(f'{path} patch_size does not match the base model')
	expected_objective = {
		**_BASE_OBJECTIVE_CONTRACT,
		'augmentations': dict(_BASE_AUGMENTATION_CONTRACT),
	}
	objective = metadata.get('pretraining_objective')
	if role == 'control' and isinstance(objective, Mapping):
		# The existing control embedding predates positive-window metadata export.
		expected_without_window = dict(expected_objective)
		expected_without_window.pop('positive_window_tokens')
		if objective not in (expected_objective, expected_without_window):
			raise ValueError(f'{path} pretraining_objective is incorrect')
	elif objective != expected_objective:
		raise ValueError(
			f'{path} pretraining_objective must include the positive-window contract'
		)
	data = _required_mapping(base_config, 'data', 'base training config')
	zero_mask = _required_mapping(base_config, 'zero_mask', 'base training config')
	for key in ('normalized_clip_abs', 'finite_check_mode', 'amplitude_agc'):
		if metadata.get(key) != data.get(key):
			raise ValueError(f'{path} {key} does not match the base training config')
	if metadata.get('zero_mask') != zero_mask:
		raise ValueError(f'{path} zero_mask does not match the base training config')
	preprocessing = _required_mapping(metadata, 'preprocessing', str(path))
	for key in ('normalized_clip_abs', 'finite_check_mode', 'amplitude_agc'):
		if preprocessing.get(key) != data.get(key):
			raise ValueError(f'{path} preprocessing.{key} is incorrect')
	pretext = metadata.get('stratigraphy_pretext')
	if role == 'control':
		if pretext is not None:
			raise ValueError(f'{path} control must not declare stratigraphy_pretext')
		return
	if not isinstance(pretext, Mapping):
		raise TypeError(f'{path} candidate stratigraphy_pretext must be a mapping')
	hmm = _mapping(stratigraphy_config, 'resolved stratigraphy training config')
	student = _required_mapping(hmm, 'student', 'stratigraphy training config')
	head = _required_mapping(hmm, 'head', 'stratigraphy training config')
	loss = _required_mapping(hmm, 'loss', 'stratigraphy training config')
	pseudo = _required_mapping(hmm, 'pseudo_targets', 'stratigraphy training config')
	expected_pretext = {
		'method': 'strat_hmm_pretext',
		'base_objective': LOCAL_BT_METHOD,
		'head_num_prototypes': head['num_prototypes'],
		'unfreeze_top_blocks': student['unfreeze_top_blocks'],
		'distillation_weight': loss['distillation_weight'],
		'pseudo_target_input_dir': pseudo['input_dir'],
		'model_tag': HMM_MODEL_TAG,
		'control_identity_sha256': _canonical_sha256(
			_mapping(control_identity, f'{checkpoint} control_identity')
		),
	}
	if pretext != expected_pretext:
		raise ValueError(f'{path} stratigraphy_pretext identity is incorrect')
	if checkpoint_payload.get('control_identity') != control_identity:
		raise ValueError(f'{path} control identity changed during metadata audit')


def _validate_file_identity(
	value: object, *, expected_path: Path | None, label: str
) -> None:
	if expected_path is None:
		raise ValueError(f'{label} expected path is missing')
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	if not _paths_equal(value.get('path'), expected_path):
		raise ValueError(f'{label} path does not match the current input')
	if value.get('sha256') != file_sha256(expected_path):
		raise ValueError(f'{label} SHA does not match the current input')


def _expected_latest_checkpoint(config: Mapping[str, object]) -> Path:
	paths = _required_mapping(config, 'paths', 'resolved training config')
	output_root = _required_absolute_path(
		paths.get('output_root'), 'training config paths.output_root'
	)
	return output_root / 'latest.pt'


def _parameter_sha256(name: str, parameter: torch.Tensor) -> str:
	value = parameter.detach().cpu().contiguous()
	digest = hashlib.sha256()
	digest.update(name.encode('utf-8'))
	digest.update(str(value.dtype).encode('utf-8'))
	digest.update(str(tuple(value.shape)).encode('utf-8'))
	digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, object]) -> str:
	digest = hashlib.sha256()
	for name in sorted(state_dict):
		value = state_dict[name]
		if not isinstance(value, torch.Tensor):
			raise TypeError(f'model state {name!r} must be a tensor')
		value = value.detach().cpu().contiguous()
		digest.update(name.encode('utf-8'))
		digest.update(str(value.dtype).encode('utf-8'))
		digest.update(str(tuple(value.shape)).encode('utf-8'))
		digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
	payload = json.dumps(
		value,
		sort_keys=True,
		separators=(',', ':'),
		allow_nan=False,
	).encode('utf-8')
	return hashlib.sha256(payload).hexdigest()


def _expect_equal(value: object, expected: object, label: str) -> None:
	if value != expected:
		raise ValueError(f'{label} must equal {expected!r}; got {value!r}')


def _expect_fields(value: object, expected: Mapping[str, object], label: str) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	for key, expected_value in expected.items():
		if value.get(key) != expected_value:
			raise ValueError(
				f'{label}.{key} must equal {expected_value!r}; got {value.get(key)!r}'
			)


def _expect_science_fields(
	value: object, expected: Mapping[str, object], label: str
) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	for key, expected_value in expected.items():
		actual = value.get(key)
		if not _science_values_equal(actual, expected_value):
			raise ValueError(
				f'{label}.{key} must equal {expected_value!r}; got {actual!r}'
			)


def _science_values_equal(value: object, expected: object) -> bool:
	if isinstance(expected, bool):
		return value is expected
	if isinstance(expected, int):
		return (
			isinstance(value, int) and not isinstance(value, bool) and value == expected
		)
	if isinstance(expected, float):
		return (
			isinstance(value, int | float)
			and not isinstance(value, bool)
			and float(value) == expected
		)
	if isinstance(expected, list):
		return (
			isinstance(value, list)
			and len(value) == len(expected)
			and all(
				_science_values_equal(item, expected_item)
				for item, expected_item in zip(value, expected, strict=True)
			)
		)
	return type(value) is type(expected) and value == expected


def _absolute_file(value: object, label: str) -> Path:
	path = _required_absolute_path(value, label)
	if not path.is_file():
		raise FileNotFoundError(f'{label} does not exist: {path}')
	return path


def _required_absolute_path(value: object, label: str) -> Path:
	if not isinstance(value, str | Path) or not str(value):
		raise TypeError(f'{label} must be a non-empty path')
	path = Path(value)
	if not path.is_absolute():
		raise ValueError(f'{label} must be absolute: {path}')
	return path


def _required_mapping(
	value: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{label}.{key} must be a mapping')
	return child


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _required_mapping_sequence(value: object, label: str) -> list[Mapping[str, object]]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a list of mappings')
	result: list[Mapping[str, object]] = []
	for index, item in enumerate(value):
		if not isinstance(item, Mapping):
			raise TypeError(f'{label}[{index}] must be a mapping')
		result.append(item)
	return result


def _required_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise TypeError(f'{label} must be a positive integer')
	return value


def _required_sha256(value: object, label: str) -> str:
	if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _paths_equal(value: object, expected: Path) -> bool:
	if not isinstance(value, str | Path) or not str(value):
		return False
	path = Path(value)
	return path.is_absolute() and path.resolve(strict=False) == expected.resolve(
		strict=False
	)


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'{label} is not valid JSON: {path}') from exc
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must contain a mapping: {path}')
	return value


__all__ = [
	'CANDIDATE_EPOCH',
	'CANDIDATE_GLOBAL_STEP',
	'CONTROL_EPOCH',
	'CONTROL_GLOBAL_STEP',
	'HMM_MODEL_TAG',
	'LOCAL_BT_METHOD',
	'F3LocalBTSourceRole',
	'validate_f3_local_bt_candidate_source',
	'validate_f3_local_bt_pseudo_target_science',
]
