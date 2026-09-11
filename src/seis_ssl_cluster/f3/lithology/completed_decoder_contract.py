"""Strict decoder-contract evidence for completed F3 five-way jobs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	f3_lithology_voxel_decoder_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	FIXED_DECODER_CONTRACT,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	BEST_CHECKPOINT_NAME,
	DECODER_DIR_NAME,
	FIVE_WAY_TILE_SETTINGS,
	LATEST_CHECKPOINT_NAME,
	PREDICTION_DIR_NAME,
	PREDICTION_METADATA_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_dataset import GRID_NAME, METADATA_NAME
from seis_ssl_cluster.f3.lithology.voxel_tiles import read_voxel_tile_manifest
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)

_DECODER_CONTRACT_KEYS = (
	'spec',
	'embedding_dim',
	'class_count',
	'hidden_channels',
	'upsample_factors',
	'upsample_mode',
	'normalization',
)
_TRAIN_CONTRACT_KEYS = (
	'epochs',
	'batch_size',
	'learning_rate',
	'weight_decay',
	'class_weight',
	'seed',
	'amp',
	'gradient_clip_norm',
	'sampling_mode',
	'steps_per_epoch',
)
_SOURCE_IDENTITY_KEYS = (
	'embeddings',
	'embedding_metadata',
	'valid_tokens',
	'voxel_dataset_metadata',
	'voxel_split_grid',
	'label_volume',
)
_RUN_METADATA_KEYS = frozenset(
	{
		'created_at_utc',
		'git_commit',
		'package_version',
		'source_embedding_metadata',
		'source_valid_tokens',
		'voxel_dataset_metadata',
		'initial_model_state_sha256',
		'sampling_mode',
		'steps_per_epoch',
		'train_seed',
		'train_tile_manifest_sha256',
		'validation_tile_manifest_sha256',
	}
)
_RESOLVED_CONFIG_NAME = 'resolved_config.json'
_RUN_METADATA_NAME = 'run_metadata.json'
_TRAIN_TILE_MANIFEST_NAME = 'train_tile_manifest.json'
_VALIDATION_TILE_MANIFEST_NAME = 'validation_tile_manifest.json'
_COMPLETED_EPOCH = 49
_COMPLETED_GLOBAL_STEP = 22_000


class F3CompletedDecoderContractEvidence(TypedDict):
	"""Validated decoder identity for one F3 five-way benchmark cell."""

	model_id: str
	layout_id: str
	data_size: str
	freeze_encoder: bool
	decoder_initial_state_sha256: str
	decoder_contract_identity_sha256: str
	decoder_resolved_config_path: str
	decoder_resolved_config_sha256: str
	decoder_run_metadata_path: str
	decoder_run_metadata_sha256: str
	decoder_latest_checkpoint_path: str
	decoder_latest_checkpoint_sha256: str
	decoder_best_checkpoint_path: str
	decoder_best_checkpoint_sha256: str
	decoder_completed_epoch: int
	decoder_completed_global_step: int
	decoder_best_epoch: int
	decoder_best_global_step: int
	decoder_train_tile_manifest_path: str
	decoder_train_tile_manifest_sha256: str
	decoder_train_tile_manifest_identity_sha256: str
	decoder_validation_tile_manifest_path: str
	decoder_validation_tile_manifest_sha256: str
	decoder_validation_tile_manifest_identity_sha256: str
	prediction_metadata_path: str
	prediction_metadata_sha256: str


def read_f3_completed_decoder_contract(
	job_dir: Path,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
	freeze_encoder: bool = True,
) -> F3CompletedDecoderContractEvidence:
	"""Validate and return strict decoder evidence for one completed job.

	The returned contract identity deliberately excludes ``model_id`` so two
	models evaluated in the same cell can prove that their decoder settings are
	identical. The model tag and output paths are still validated separately.
	"""
	_validate_expected_identity(
		job_dir,
		model_id=model_id,
		layout_id=layout_id,
		data_size=data_size,
		freeze_encoder=freeze_encoder,
	)
	decoder_dir = job_dir / DECODER_DIR_NAME
	resolved_path = decoder_dir / _RESOLVED_CONFIG_NAME
	run_metadata_path = decoder_dir / _RUN_METADATA_NAME
	latest_path = decoder_dir / LATEST_CHECKPOINT_NAME
	best_path = decoder_dir / BEST_CHECKPOINT_NAME
	train_manifest_path = decoder_dir / _TRAIN_TILE_MANIFEST_NAME
	validation_manifest_path = decoder_dir / _VALIDATION_TILE_MANIFEST_NAME
	prediction_metadata_path = job_dir / PREDICTION_DIR_NAME / PREDICTION_METADATA_NAME

	resolved = _read_json(resolved_path, label='decoder resolved config')
	_validate_resolved_config(
		resolved,
		decoder_dir=decoder_dir,
		model_id=model_id,
		layout_id=layout_id,
		data_size=data_size,
		freeze_encoder=freeze_encoder,
	)
	decoder_expected = _contract_subset(_DECODER_CONTRACT_KEYS)
	train_expected = {
		**_contract_subset(_TRAIN_CONTRACT_KEYS),
		'num_workers': 0,
	}
	tile_expected = _plain_json_value(dict(FIVE_WAY_TILE_SETTINGS))

	manifests = _validate_tile_manifests(
		train_manifest_path,
		validation_manifest_path,
		tile_expected=tile_expected,
	)
	manifest_identities = {
		'train': manifests['train'],
		'validation': manifests['validation'],
	}

	latest = _load_checkpoint(latest_path, label='decoder latest checkpoint')
	_validate_completed_checkpoint(
		latest,
		path=latest_path,
		resolved=resolved,
		manifest_identities=manifest_identities,
	)
	artifact_identities = _validate_artifact_identities(
		latest.get('artifact_identities'),
		resolved=resolved,
		label='decoder latest checkpoint artifact_identities',
	)

	best_sha256 = _validate_best_checkpoint(
		best_path,
		latest=latest,
		resolved=resolved,
		manifest_identities=manifest_identities,
		artifact_identities=artifact_identities,
	)
	best_state = _required_mapping(
		latest.get('best_selection_state'),
		label='decoder latest checkpoint best_selection_state',
	)
	best_epoch = _required_int(
		best_state.get('epoch'),
		label='decoder latest checkpoint best_selection_state.epoch',
	)
	best_global_step = (best_epoch + 1) * _required_int(
		train_expected['steps_per_epoch'], label='fixed train.steps_per_epoch'
	)

	run_metadata = _read_json(run_metadata_path, label='decoder run metadata')
	initial_state = _validate_run_metadata(
		run_metadata,
		train_expected=train_expected,
		manifest_identities=manifest_identities,
		artifact_identities=artifact_identities,
	)
	_validate_prediction_metadata(
		prediction_metadata_path,
		model_id=model_id,
		resolved=resolved,
		resolved_path=resolved_path,
		best_path=best_path,
		best_sha256=best_sha256,
		train_manifest_path=train_manifest_path,
		validation_manifest_path=validation_manifest_path,
		manifest_identities=manifest_identities,
		artifact_identities=artifact_identities,
		train_expected=train_expected,
	)

	contract_identity = {
		'schema_version': 1,
		'layout_id': layout_id,
		'data_size': data_size,
		'freeze_encoder': freeze_encoder,
		'decoder': decoder_expected,
		'tiles': tile_expected,
		'train': train_expected,
	}
	return {
		'model_id': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'freeze_encoder': freeze_encoder,
		'decoder_initial_state_sha256': initial_state,
		'decoder_contract_identity_sha256': _canonical_sha256(contract_identity),
		'decoder_resolved_config_path': str(resolved_path),
		'decoder_resolved_config_sha256': file_sha256(resolved_path),
		'decoder_run_metadata_path': str(run_metadata_path),
		'decoder_run_metadata_sha256': file_sha256(run_metadata_path),
		'decoder_latest_checkpoint_path': str(latest_path),
		'decoder_latest_checkpoint_sha256': file_sha256(latest_path),
		'decoder_best_checkpoint_path': str(best_path),
		'decoder_best_checkpoint_sha256': best_sha256,
		'decoder_completed_epoch': _COMPLETED_EPOCH,
		'decoder_completed_global_step': _COMPLETED_GLOBAL_STEP,
		'decoder_best_epoch': best_epoch,
		'decoder_best_global_step': best_global_step,
		'decoder_train_tile_manifest_path': str(train_manifest_path),
		'decoder_train_tile_manifest_sha256': file_sha256(train_manifest_path),
		'decoder_train_tile_manifest_identity_sha256': manifests['train'],
		'decoder_validation_tile_manifest_path': str(validation_manifest_path),
		'decoder_validation_tile_manifest_sha256': file_sha256(
			validation_manifest_path
		),
		'decoder_validation_tile_manifest_identity_sha256': manifests['validation'],
		'prediction_metadata_path': str(prediction_metadata_path),
		'prediction_metadata_sha256': file_sha256(prediction_metadata_path),
	}


def _validate_resolved_config(  # noqa: PLR0913
	resolved: Mapping[str, object],
	*,
	decoder_dir: Path,
	model_id: str,
	layout_id: str,
	data_size: str,
	freeze_encoder: bool,
) -> None:
	parsed = f3_lithology_voxel_decoder_config_from_mapping(resolved)
	_validate_exact_value(
		resolved,
		parsed.to_dict(),
		label='decoder resolved config',
	)
	_validate_exact_mapping(
		resolved.get('decoder'),
		expected=_contract_subset(_DECODER_CONTRACT_KEYS),
		label='decoder resolved decoder',
	)
	_validate_exact_mapping(
		resolved.get('tiles'),
		expected=_plain_json_value(dict(FIVE_WAY_TILE_SETTINGS)),
		label='decoder resolved tiles',
	)
	_validate_exact_mapping(
		resolved.get('train'),
		expected={
			**_contract_subset(_TRAIN_CONTRACT_KEYS),
			'num_workers': 0,
		},
		label='decoder resolved train',
	)
	_validate_exact_mapping(
		resolved.get('model'),
		expected={'tag': model_id, 'freeze_encoder': freeze_encoder},
		label='decoder resolved model',
	)

	outputs = _required_mapping(
		resolved.get('outputs'), label='decoder resolved outputs'
	)
	output_dir = _absolute_path(
		outputs.get('output_dir'), label='decoder resolved outputs.output_dir'
	)
	if not _same_path(output_dir, decoder_dir):
		raise ValueError(
			f'decoder resolved outputs.output_dir does not match this job: {output_dir}'
		)
	voxel_dataset = _required_mapping(
		resolved.get('voxel_dataset'), label='decoder resolved voxel_dataset'
	)
	input_dir = _absolute_path(
		voxel_dataset.get('input_dir'),
		label='decoder resolved voxel_dataset.input_dir',
	)
	expected_suffix = (
		'datasets',
		f'layout={layout_id}',
		f'size={data_size}',
		'voxel_supervision',
	)
	if input_dir.parts[-4:] != expected_suffix:
		raise ValueError(
			'decoder resolved voxel_dataset.input_dir does not match the expected '
			f'layout/size: {input_dir}'
		)


def _validate_tile_manifests(
	train_path: Path,
	validation_path: Path,
	*,
	tile_expected: Mapping[str, object],
) -> dict[str, str]:
	identities: dict[str, str] = {}
	geometries: dict[str, tuple[object, ...]] = {}
	for split, path in (
		('train', train_path),
		('validation', validation_path),
	):
		if not path.is_file():
			raise FileNotFoundError(f'missing decoder {split} tile manifest: {path}')
		manifest = read_voxel_tile_manifest(path)
		if manifest.split != split:
			raise ValueError(
				f'decoder {split} tile manifest declares split {manifest.split!r}'
			)
		_validate_exact_value(
			list(manifest.core_size_tokens),
			tile_expected['core_size_tokens'],
			label=f'decoder {split} tile manifest core_size_tokens',
		)
		_validate_exact_value(
			list(manifest.context_halo_tokens),
			tile_expected['context_halo_tokens'],
			label=f'decoder {split} tile manifest context_halo_tokens',
		)
		identities[split] = manifest.identity_sha256
		geometries[split] = (
			manifest.volume_shape_xyz,
			manifest.token_grid_shape_xyz,
			manifest.patch_size_xyz,
			manifest.class_ids,
		)
	if geometries['train'] != geometries['validation']:
		raise ValueError('decoder train and validation tile manifest geometry differs')
	return identities


def _validate_completed_checkpoint(
	payload: Mapping[str, object],
	*,
	path: Path,
	resolved: Mapping[str, object],
	manifest_identities: Mapping[str, str],
) -> None:
	_validate_exact_value(
		payload.get('checkpoint_kind'),
		'completed',
		label=f'{path} checkpoint_kind',
	)
	_validate_exact_value(payload.get('batch_index'), None, label=f'{path} batch_index')
	_validate_counter(payload, 'epoch', _COMPLETED_EPOCH, path=path)
	_validate_counter(payload, 'global_step', _COMPLETED_GLOBAL_STEP, path=path)
	_validate_exact_mapping(
		payload.get('resolved_config'),
		expected=resolved,
		label=f'{path} resolved_config',
	)
	_validate_manifest_hashes(
		payload.get('tile_manifest_hashes'),
		expected=manifest_identities,
		label=f'{path} tile_manifest_hashes',
	)
	_validate_history(
		payload.get('training_history'),
		epoch_count=_COMPLETED_EPOCH + 1,
		steps_per_epoch=FIXED_DECODER_CONTRACT['steps_per_epoch'],
		label=f'{path} training_history',
	)
	_validate_exact_value(
		payload.get('history'),
		payload.get('training_history'),
		label=f'{path} history alias',
	)
	best_state = _required_mapping(
		payload.get('best_selection_state'),
		label=f'{path} best_selection_state',
	)
	_require_exact_keys(
		best_state,
		{'epoch', 'validation_metrics', 'rule', 'epsilon'},
		label=f'{path} best_selection_state',
	)
	best_epoch = _required_int(
		best_state.get('epoch'), label=f'{path} best_selection_state.epoch'
	)
	if not 0 <= best_epoch <= _COMPLETED_EPOCH:
		raise ValueError(f'{path} best_selection_state.epoch is out of range')
	_required_mapping(
		best_state.get('validation_metrics'),
		label=f'{path} best_selection_state.validation_metrics',
	)
	_sha256(
		payload.get('best_checkpoint_sha256'),
		label=f'{path} best_checkpoint_sha256',
	)


def _validate_best_checkpoint(
	path: Path,
	*,
	latest: Mapping[str, object],
	resolved: Mapping[str, object],
	manifest_identities: Mapping[str, str],
	artifact_identities: Mapping[str, object],
) -> str:
	recorded_sha256 = _sha256(
		latest.get('best_checkpoint_sha256'),
		label='decoder latest checkpoint best_checkpoint_sha256',
	)
	if not path.is_file():
		raise FileNotFoundError(f'missing decoder best checkpoint: {path}')
	actual_sha256 = file_sha256(path)
	if recorded_sha256 != actual_sha256:
		raise ValueError('decoder latest checkpoint best.pt SHA-256 linkage mismatch')
	best = _load_checkpoint(path, label='decoder best checkpoint')
	kind = best.get('checkpoint_kind')
	if type(kind) is not str or kind not in {'epoch', 'completed'}:
		raise ValueError('decoder best checkpoint must be an epoch checkpoint')
	_validate_exact_value(best.get('batch_index'), None, label=f'{path} batch_index')
	_validate_exact_mapping(
		best.get('resolved_config'),
		expected=resolved,
		label=f'{path} resolved_config',
	)
	_validate_manifest_hashes(
		best.get('tile_manifest_hashes'),
		expected=manifest_identities,
		label=f'{path} tile_manifest_hashes',
	)
	_validate_exact_mapping(
		best.get('artifact_identities'),
		expected=artifact_identities,
		label=f'{path} artifact_identities',
	)
	_validate_exact_value(
		best.get('best_checkpoint_sha256'),
		None,
		label=f'{path} best_checkpoint_sha256',
	)
	latest_state = _required_mapping(
		latest.get('best_selection_state'),
		label='decoder latest checkpoint best_selection_state',
	)
	_validate_exact_mapping(
		best.get('best_selection_state'),
		expected=latest_state,
		label=f'{path} best_selection_state',
	)
	best_epoch = _required_int(
		latest_state.get('epoch'), label=f'{path} selected epoch'
	)
	expected_global_step = (best_epoch + 1) * _required_int(
		FIXED_DECODER_CONTRACT['steps_per_epoch'],
		label='fixed train.steps_per_epoch',
	)
	_validate_counter(best, 'epoch', best_epoch, path=path)
	_validate_counter(best, 'global_step', expected_global_step, path=path)
	_validate_history(
		best.get('training_history'),
		epoch_count=best_epoch + 1,
		steps_per_epoch=FIXED_DECODER_CONTRACT['steps_per_epoch'],
		label=f'{path} training_history',
	)
	_validate_exact_value(
		best.get('history'),
		best.get('training_history'),
		label=f'{path} history alias',
	)
	current_metrics = _required_mapping(
		best.get('current_metrics'), label=f'{path} current_metrics'
	)
	_validate_exact_value(
		current_metrics.get('validation'),
		latest_state.get('validation_metrics'),
		label=f'{path} selected validation metrics',
	)
	return actual_sha256


def _validate_artifact_identities(
	value: object,
	*,
	resolved: Mapping[str, object],
	label: str,
) -> Mapping[str, object]:
	identities = _required_mapping(value, label=label)
	_require_exact_keys(
		identities,
		{'name', *_SOURCE_IDENTITY_KEYS},
		label=label,
	)
	_validate_exact_value(
		identities.get('name'), 'f3_voxel_decoder_sources', label=f'{label}.name'
	)
	for key in _SOURCE_IDENTITY_KEYS:
		_validate_identity_entry(identities.get(key), label=f'{label}.{key}')

	dataset = _required_mapping(resolved.get('dataset'), label='resolved dataset')
	embeddings = _required_mapping(
		resolved.get('embeddings'), label='resolved embeddings'
	)
	voxel_dataset = _required_mapping(
		resolved.get('voxel_dataset'), label='resolved voxel_dataset'
	)
	input_dir = _absolute_path(
		embeddings.get('input_dir'), label='resolved embeddings.input_dir'
	)
	survey_id = _required_str(dataset.get('name'), label='resolved dataset.name')
	files = output_paths(input_dir, survey_id)
	voxel_dir = _absolute_path(
		voxel_dataset.get('input_dir'), label='resolved voxel_dataset.input_dir'
	)
	for key, expected_path in (
		('embeddings', files.embeddings),
		('embedding_metadata', files.metadata),
		('valid_tokens', files.valid_tokens),
		('voxel_dataset_metadata', voxel_dir / METADATA_NAME),
		('voxel_split_grid', voxel_dir / GRID_NAME),
	):
		path, _ = _validate_identity_entry(identities[key], label=f'{label}.{key}')
		if not _same_path(path, expected_path):
			raise ValueError(f'{label}.{key}.path does not match resolved config')
	return identities


def _validate_run_metadata(
	metadata: Mapping[str, object],
	*,
	train_expected: Mapping[str, object],
	manifest_identities: Mapping[str, str],
	artifact_identities: Mapping[str, object],
) -> str:
	_require_exact_keys(metadata, _RUN_METADATA_KEYS, label='decoder run metadata')
	initial_state = _sha256(
		metadata.get('initial_model_state_sha256'),
		label='decoder run metadata initial_model_state_sha256',
	)
	for key, expected in (
		('sampling_mode', train_expected['sampling_mode']),
		('steps_per_epoch', train_expected['steps_per_epoch']),
		('train_seed', train_expected['seed']),
		('train_tile_manifest_sha256', manifest_identities['train']),
		('validation_tile_manifest_sha256', manifest_identities['validation']),
	):
		_validate_exact_value(
			metadata.get(key), expected, label=f'decoder run metadata {key}'
		)
	for key, identity_key in (
		('source_embedding_metadata', 'embedding_metadata'),
		('source_valid_tokens', 'valid_tokens'),
		('voxel_dataset_metadata', 'voxel_dataset_metadata'),
	):
		expected_path, _ = _validate_identity_entry(
			artifact_identities[identity_key],
			label=f'decoder checkpoint artifact_identities.{identity_key}',
		)
		actual_path = _absolute_path(
			metadata.get(key), label=f'decoder run metadata {key}'
		)
		if not _same_path(actual_path, expected_path):
			raise ValueError(
				f'decoder run metadata {key} does not match checkpoint source identity'
			)
	_required_str(
		metadata.get('created_at_utc'),
		label='decoder run metadata created_at_utc',
	)
	for key in ('git_commit', 'package_version'):
		value = metadata.get(key)
		if value is not None and (type(value) is not str or not value):
			raise TypeError(f'decoder run metadata {key} must be null or a string')
	return initial_state


def _validate_prediction_metadata(  # noqa: PLR0913
	path: Path,
	*,
	model_id: str,
	resolved: Mapping[str, object],
	resolved_path: Path,
	best_path: Path,
	best_sha256: str,
	train_manifest_path: Path,
	validation_manifest_path: Path,
	manifest_identities: Mapping[str, str],
	artifact_identities: Mapping[str, object],
	train_expected: Mapping[str, object],
) -> None:
	metadata = _read_json(path, label='prediction metadata')
	_validate_exact_value(
		metadata.get('prediction_kind'),
		'frozen_embedding_decoder',
		label='prediction metadata prediction_kind',
	)
	_validate_exact_value(
		metadata.get('model_tag'), model_id, label='prediction metadata model_tag'
	)
	_validate_exact_mapping(
		metadata.get('decoder_architecture'),
		expected=_required_mapping(
			resolved.get('decoder'), label='resolved decoder architecture'
		),
		label='prediction metadata decoder_architecture',
	)
	_validate_exact_mapping(
		metadata.get('training_sampling'),
		expected={
			'sampling_mode': train_expected['sampling_mode'],
			'steps_per_epoch': train_expected['steps_per_epoch'],
			'train_seed': train_expected['seed'],
			'train_tile_manifest_sha256': manifest_identities['train'],
			'validation_tile_manifest_sha256': manifest_identities['validation'],
		},
		label='prediction metadata training_sampling',
	)

	source = _required_mapping(
		metadata.get('source_identity'), label='prediction metadata source_identity'
	)
	_require_exact_keys(
		source,
		{
			'decoder_checkpoint',
			'resolved_decoder_config',
			'class_info',
			'artifact_identities',
			'tile_manifests',
		},
		label='prediction metadata source_identity',
	)
	_validate_recorded_file_identity(
		source.get('decoder_checkpoint'),
		expected_path=best_path,
		expected_sha256=best_sha256,
		label='prediction metadata source_identity.decoder_checkpoint',
	)
	_validate_recorded_file_identity(
		source.get('resolved_decoder_config'),
		expected_path=resolved_path,
		expected_sha256=file_sha256(resolved_path),
		label='prediction metadata source_identity.resolved_decoder_config',
	)
	_validate_exact_mapping(
		source.get('artifact_identities'),
		expected=artifact_identities,
		label='prediction metadata source_identity.artifact_identities',
	)
	class_info_path, class_info_sha256 = _validate_identity_entry(
		source.get('class_info'),
		label='prediction metadata source_identity.class_info',
	)
	if file_sha256(class_info_path) != class_info_sha256:
		raise ValueError('prediction metadata class_info SHA-256 does not match file')
	tile_sources = _required_mapping(
		source.get('tile_manifests'),
		label='prediction metadata source_identity.tile_manifests',
	)
	_require_exact_keys(
		tile_sources, {'train', 'validation'}, label='prediction tile manifests'
	)
	for split, manifest_path in (
		('train', train_manifest_path),
		('validation', validation_manifest_path),
	):
		_validate_recorded_file_identity(
			tile_sources.get(split),
			expected_path=manifest_path,
			expected_sha256=file_sha256(manifest_path),
			label=f'prediction metadata {split} tile manifest',
		)

	inputs = _required_mapping(
		metadata.get('inputs'), label='prediction metadata inputs'
	)
	_require_exact_keys(
		inputs,
		{
			'embeddings',
			'embedding_metadata',
			'valid_tokens',
			'class_info',
			'decoder_checkpoint',
		},
		label='prediction metadata inputs',
	)
	expected_inputs: dict[str, Path] = {
		'class_info': class_info_path,
		'decoder_checkpoint': best_path,
	}
	for key in ('embeddings', 'embedding_metadata', 'valid_tokens'):
		expected_inputs[key] = _validate_identity_entry(
			artifact_identities[key], label=f'decoder artifact_identities.{key}'
		)[0]
	for key, expected_path in expected_inputs.items():
		actual_path = _absolute_path(
			inputs.get(key), label=f'prediction metadata inputs.{key}'
		)
		if not _same_path(actual_path, expected_path):
			raise ValueError(
				f'prediction metadata inputs.{key} does not match source identity'
			)


def _validate_manifest_hashes(
	value: object,
	*,
	expected: Mapping[str, str],
	label: str,
) -> None:
	mapping = _required_mapping(value, label=label)
	_require_exact_keys(mapping, {'train', 'validation'}, label=label)
	for split in ('train', 'validation'):
		actual = _sha256(mapping.get(split), label=f'{label}.{split}')
		if actual != expected[split]:
			raise ValueError(f'{label}.{split} does not match tile manifest identity')


def _validate_history(
	value: object,
	*,
	epoch_count: int,
	steps_per_epoch: object,
	label: str,
) -> None:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a sequence')
	if len(value) != epoch_count:
		raise ValueError(f'{label} must contain {epoch_count} completed epochs')
	steps = _required_int(steps_per_epoch, label='fixed train.steps_per_epoch')
	for expected_epoch, row in enumerate(value):
		entry = _required_mapping(row, label=f'{label}[{expected_epoch}]')
		_validate_counter(entry, 'epoch', expected_epoch, path=Path(label))
		_validate_counter(
			entry,
			'global_step',
			(expected_epoch + 1) * steps,
			path=Path(label),
		)


def _validate_counter(
	payload: Mapping[str, object], key: str, expected: int, *, path: Path
) -> None:
	value = _required_int(payload.get(key), label=f'{path} {key}')
	if value != expected:
		raise ValueError(f'{path} {key} must equal {expected}; got {value}')


def _load_checkpoint(path: Path, *, label: str) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	return load_voxel_decoder_checkpoint(path, map_location='cpu')


def _validate_expected_identity(
	job_dir: Path,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
	freeze_encoder: bool,
) -> None:
	if not model_id or Path(model_id).name != model_id or model_id in {'.', '..'}:
		raise ValueError('model_id must be one non-empty path segment')
	if layout_id not in LAYOUT_IDS:
		raise ValueError(
			f'layout_id must be one of {list(LAYOUT_IDS)!r}; got {layout_id!r}'
		)
	if data_size not in DATA_SIZES:
		raise ValueError(
			f'data_size must be one of {list(DATA_SIZES)!r}; got {data_size!r}'
		)
	if not isinstance(freeze_encoder, bool):
		raise TypeError('freeze_encoder must be a bool')
	expected_suffix = (
		f'model={model_id}',
		f'layout={layout_id}',
		f'size={data_size}',
	)
	if job_dir.parts[-3:] != expected_suffix:
		raise ValueError(
			'decoder job directory does not match the expected model/layout/size: '
			f'{job_dir}'
		)


def _contract_subset(keys: tuple[str, ...]) -> dict[str, object]:
	return {key: _plain_json_value(FIXED_DECODER_CONTRACT[key]) for key in keys}


def _validate_exact_mapping(
	value: object,
	*,
	expected: Mapping[str, object],
	label: str,
) -> None:
	actual = _required_mapping(value, label=label)
	_require_exact_keys(actual, set(expected), label=label)
	for key, expected_value in expected.items():
		_validate_exact_value(actual[key], expected_value, label=f'{label}.{key}')


def _validate_exact_value(value: object, expected: object, *, label: str) -> None:
	if isinstance(expected, Mapping):
		_validate_exact_mapping(value, expected=expected, label=label)
		return
	if isinstance(expected, list):
		if type(value) is not list:
			raise TypeError(f'{label} must be a list; got {type(value).__name__}')
		if len(value) != len(expected):
			raise ValueError(
				f'{label} must have length {len(expected)}; got {len(value)}'
			)
		for index, (actual_item, expected_item) in enumerate(
			zip(value, expected, strict=True)
		):
			_validate_exact_value(actual_item, expected_item, label=f'{label}[{index}]')
		return
	if type(value) is not type(expected):
		raise TypeError(
			f'{label} must have type {type(expected).__name__}; '
			f'got {type(value).__name__}'
		)
	if value != expected:
		raise ValueError(f'{label} must equal {expected!r}; got {value!r}')


def _require_exact_keys(
	value: Mapping[str, object], expected: set[str] | frozenset[str], *, label: str
) -> None:
	actual = set(value)
	if actual != set(expected):
		missing = sorted(set(expected) - actual)
		unexpected = sorted(actual - set(expected))
		raise ValueError(
			f'{label} keys differ; missing={missing!r}, unexpected={unexpected!r}'
		)


def _validate_identity_entry(value: object, *, label: str) -> tuple[Path, str]:
	entry = _required_mapping(value, label=label)
	_require_exact_keys(entry, {'path', 'sha256'}, label=label)
	path = _absolute_path(entry.get('path'), label=f'{label}.path')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label} file: {path}')
	return path, _sha256(entry.get('sha256'), label=f'{label}.sha256')


def _validate_recorded_file_identity(
	value: object,
	*,
	expected_path: Path,
	expected_sha256: str,
	label: str,
) -> None:
	path, sha256 = _validate_identity_entry(value, label=label)
	if not _same_path(path, expected_path):
		raise ValueError(f'{label}.path does not match the expected artifact')
	if sha256 != expected_sha256:
		raise ValueError(f'{label}.sha256 does not match the expected artifact')
	if file_sha256(path) != expected_sha256:
		raise ValueError(f'{label}.sha256 does not match the current file')


def _plain_json_value(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _plain_json_value(item) for key, item in value.items()}
	if isinstance(value, tuple | list):
		return [_plain_json_value(item) for item in value]
	return value


def _canonical_sha256(value: object) -> str:
	payload = json.dumps(
		value,
		allow_nan=False,
		separators=(',', ':'),
		sort_keys=True,
	).encode('utf-8')
	return hashlib.sha256(payload).hexdigest()


def _absolute_path(value: object, *, label: str) -> Path:
	if type(value) is not str or not value:
		raise ValueError(f'{label} must be a non-empty path')
	path = Path(value)
	if not path.is_absolute():
		raise ValueError(f'{label} must be absolute')
	return path


def _required_str(value: object, *, label: str) -> str:
	if type(value) is not str or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _required_int(value: object, *, label: str) -> int:
	if type(value) is not int:
		raise TypeError(f'{label} must be an integer')
	return value


def _sha256(value: object, *, label: str) -> str:
	if (
		type(value) is not str
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256')
	return value


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	if any(type(key) is not str for key in value):
		raise TypeError(f'{label} keys must be strings')
	return value


def _same_path(first: Path, second: Path) -> bool:
	return first.resolve(strict=False) == second.resolve(strict=False)


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	payload = json.loads(path.read_text(encoding='utf-8'))
	if type(payload) is not dict:
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return payload


__all__ = [
	'F3CompletedDecoderContractEvidence',
	'read_f3_completed_decoder_contract',
]
