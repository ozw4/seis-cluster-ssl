# ruff: noqa: CPY001, PT018, TC003

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.parihaka.channel_data import SectionLines, split_mask_for_crop
from seis_ssl_cluster.parihaka.channel_decoder import (
	CHANNEL_PRETRAINED_MODEL_TAG,
	ChannelDecoderConfig,
	DecoderArchitecture,
	DecoderTiles,
	DecoderTrain,
	channel_decoder_config_from_mapping,
	channel_metrics,
	decoder_initial_state_sha256,
	deterministic_tile_order,
	inspect_channel_decoder_job,
	inspect_embedding_pair,
	run_channel_decoder_job,
)


def _config(tmp_path: Path) -> ChannelDecoderConfig:
	return ChannelDecoderConfig(
		survey_id='parihaka',
		labels=tmp_path / 'labels.npy',
		labels_metadata=tmp_path / 'parihaka_labels_metadata.json',
		pretrained_embeddings=tmp_path / 'pretrained',
		random_embeddings=tmp_path / 'random',
		runs_root=tmp_path / 'runs',
		decoder=DecoderArchitecture(
			embedding_dim=384,
			class_count=2,
			hidden_channels=(128, 64, 32),
			upsample_factors=((2, 2, 2),) * 3,
			upsample_mode='nearest',
			normalization='voxelwise_layer_norm',
		),
		train=DecoderTrain(
			epochs=50,
			batch_size=1,
			learning_rate=0.001,
			weight_decay=0.0001,
			class_weight='balanced',
			sampling_mode='all_tiles_once',
			seed=42000,
			amp=True,
			gradient_clip_norm=1.0,
		),
		tiles=DecoderTiles((2, 2, 2), (1, 1, 1)),
	)


def _write_labels(config: ChannelDecoderConfig, labels: np.ndarray) -> None:
	np.save(config.labels, labels, allow_pickle=False)
	config.labels_metadata.write_text(
		json.dumps(
			{
				'schema_version': 2,
				'artifact_type': 'parihaka_channel_labels',
				'output_labels': str(config.labels),
				'prepared_label_identity': {
					'labels_sha256': file_sha256(config.labels),
					'source_npz_path': '/data/parihaka_labels.npz',
					'source_key': 'labels',
					'shape': list(labels.shape),
					'dtype': labels.dtype.name,
					'class_definition': {
						'positive_class_id': 5,
						'negative_class_ids': [1, 2, 3, 4, 6],
					},
				},
			}
		),
		encoding='utf-8',
	)


def _write_model_checkpoints(config: ChannelDecoderConfig) -> tuple[Path, Path]:
	checkpoint_root = config.pretrained_embeddings.parent / 'pretraining'
	pretrained_checkpoint = (
		checkpoint_root
		/ 'parihaka'
		/ 'facies_benchmark_v1'
		/ CHANNEL_PRETRAINED_MODEL_TAG
		/ 'full_100ep'
		/ 'latest.pt'
	)
	random_checkpoint = checkpoint_root / 'random' / 'mae_random_seed42.pt'
	pretrained_checkpoint.parent.mkdir(parents=True)
	random_checkpoint.parent.mkdir(parents=True)
	torch.save(
		{'model_state_dict': {'weight': torch.ones(1)}}, pretrained_checkpoint
	)
	torch.save(
		{
			'model_state_dict': {'weight': torch.zeros(1)},
			'metadata': {
				'random_encoder_baseline': True,
				'pretrained_weights_loaded': False,
				'seed': 42,
				'reference_checkpoint': str(pretrained_checkpoint),
				'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
			},
			'training_state': {'checkpoint_kind': 'random_init'},
		},
		random_checkpoint,
	)
	return pretrained_checkpoint, random_checkpoint


def _write_pair(config: ChannelDecoderConfig) -> tuple[Path, Path]:
	pretrained_checkpoint, random_checkpoint = _write_model_checkpoints(config)
	metadata = {
		'survey_id': config.survey_id,
		'source_amplitude_path': '/data/parihaka_amplitude.npy',
		'patch_size': [8, 8, 8],
		'token_grid_shape': [2, 2, 2],
		'volume_shape_xyz': [16, 16, 16],
		'embedding_dim': 384,
		'model_geometry': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
		'window_size': [16, 16, 16],
		'overlap': [8, 8, 8],
		'output_dtype': 'float16',
		'min_token_valid_fraction': 0.5,
		'normalization_stats_path': '/data/parihaka_stats.json',
		'preprocessing': {'mode': 'same'},
		'zero_mask': {'enabled': True},
		'precision': {'device_type': 'cpu', 'autocast': False},
		'pretraining_objective': {'name': 'masked_autoencoding'},
	}
	for root, checkpoint_path in (
		(config.pretrained_embeddings, pretrained_checkpoint),
		(config.random_embeddings, random_checkpoint),
	):
		paths = output_paths(root, config.survey_id)
		root.mkdir(parents=True)
		np.save(paths.embeddings, np.zeros((2, 2, 2, 384), dtype=np.float16))
		np.save(paths.valid_tokens, np.ones((2, 2, 2), dtype=np.bool_))
		paths.metadata.write_text(
			json.dumps(
				{
					**metadata,
					'checkpoint_path': str(checkpoint_path),
					'checkpoint_sha256': file_sha256(checkpoint_path),
				}
			),
			encoding='utf-8',
		)
	return pretrained_checkpoint, random_checkpoint


def _config_mapping(tmp_path: Path) -> dict[str, object]:
	config = _config(tmp_path)
	return {
		'dataset': {'survey_id': config.survey_id},
		'inputs': {
			'labels_npy': str(config.labels),
			'labels_metadata_json': str(config.labels_metadata),
		},
		'embeddings': {
			'pretrained_dir': str(config.pretrained_embeddings),
			'random_dir': str(config.random_embeddings),
		},
		'outputs': {'runs_root': str(config.runs_root)},
		'decoder': {
			'spec': config.decoder.spec,
			'embedding_dim': config.decoder.embedding_dim,
			'class_count': config.decoder.class_count,
			'hidden_channels': list(config.decoder.hidden_channels),
			'upsample_factors': [
				list(item) for item in config.decoder.upsample_factors
			],
			'upsample_mode': config.decoder.upsample_mode,
			'normalization': config.decoder.normalization,
		},
		'train': {
			'epochs': config.train.epochs,
			'batch_size': config.train.batch_size,
			'learning_rate': config.train.learning_rate,
			'weight_decay': config.train.weight_decay,
			'class_weight': config.train.class_weight,
			'sampling_mode': config.train.sampling_mode,
			'seed': config.train.seed,
			'amp': config.train.amp,
			'gradient_clip_norm': config.train.gradient_clip_norm,
		},
		'tiles': {
			'core_size_tokens': [8, 8, 8],
			'context_halo_tokens': [1, 1, 1],
		},
	}


def _write_layout(tmp_path: Path) -> Path:
	layout_path = tmp_path / 'layouts.yaml'
	layout_path.write_text(
		yaml.safe_dump(
			{
				'validation': {'inline': [12], 'crossline': [12]},
				'test': {'inline': [13], 'crossline': [13]},
				'layouts': {
					f'layout_{index:03d}': {
						'inline': [index, index + 1, index + 2, index + 3],
						'crossline': [index, index + 1, index + 2, index + 3],
					}
					for index in range(5)
				},
			}
		),
		encoding='utf-8',
	)
	return layout_path


def test_embedding_geometry_mismatch_is_rejected(tmp_path: Path) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	np.save(random_paths.embeddings, np.zeros((2, 2, 3, 384), dtype=np.float16))
	with pytest.raises(ValueError, match='embedding shape mismatch'):
		inspect_embedding_pair(config)


def test_valid_token_mask_mismatch_is_rejected(tmp_path: Path) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	mask = np.ones((2, 2, 2), dtype=np.bool_)
	mask[0, 0, 0] = False
	np.save(random_paths.valid_tokens, mask)
	with pytest.raises(ValueError, match='valid-token mask mismatch'):
		inspect_embedding_pair(config)


@pytest.mark.parametrize(
	('key', 'different_value'),
	[
		('survey_id', 'another-survey'),
		('source_amplitude_path', '/data/another.npy'),
		('volume_shape_xyz', [24, 16, 16]),
		('model_geometry', {'embed_dim': 384, 'depth': 24, 'num_heads': 6}),
		('patch_size', [4, 8, 8]),
		('token_grid_shape', [3, 2, 2]),
		('window_size', [8, 16, 16]),
		('overlap', [4, 8, 8]),
		('min_token_valid_fraction', 0.75),
		('normalization_stats_path', '/data/another_stats.json'),
		('preprocessing', {'mode': 'different'}),
		('zero_mask', {'enabled': False}),
		('precision', {'device_type': 'cuda', 'autocast': True}),
		('pretraining_objective', {'name': 'different'}),
	],
)
def test_embedding_pair_metadata_mismatch_is_rejected(
	tmp_path: Path,
	key: str,
	different_value: object,
) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	metadata = json.loads(random_paths.metadata.read_text(encoding='utf-8'))
	metadata[key] = different_value
	random_paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match=rf'metadata {key} mismatch'):
		inspect_embedding_pair(config)


@pytest.mark.parametrize('model', ['pretrained', 'random'])
def test_embedding_array_dtype_must_match_metadata(
	tmp_path: Path,
	model: str,
) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	root = (
		config.pretrained_embeddings
		if model == 'pretrained'
		else config.random_embeddings
	)
	paths = output_paths(root, config.survey_id)
	np.save(paths.embeddings, np.zeros((2, 2, 2, 384), dtype=np.float32))
	with pytest.raises(TypeError, match=rf'{model} embedding array dtype'):
		inspect_embedding_pair(config)


def test_embedding_arrays_must_have_same_actual_dtype(tmp_path: Path) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	np.save(
		random_paths.embeddings,
		np.zeros((2, 2, 2, 384), dtype=np.float32),
	)
	for root, output_dtype in (
		(config.pretrained_embeddings, 'float16'),
		(config.random_embeddings, 'float32'),
	):
		paths = output_paths(root, config.survey_id)
		metadata = json.loads(paths.metadata.read_text(encoding='utf-8'))
		metadata['output_dtype'] = output_dtype
		paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(TypeError, match='embedding array dtype mismatch'):
		inspect_embedding_pair(config)


def test_random_embedding_array_must_be_floating(tmp_path: Path) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	np.save(random_paths.embeddings, np.zeros((2, 2, 2, 384), dtype=np.int16))
	with pytest.raises(TypeError, match='embeddings must be floating'):
		inspect_embedding_pair(config)


def test_embedding_pair_must_use_distinct_checkpoint_sha256(tmp_path: Path) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	pretrained_paths = output_paths(config.pretrained_embeddings, config.survey_id)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	pretrained_metadata = json.loads(
		pretrained_paths.metadata.read_text(encoding='utf-8')
	)
	metadata = json.loads(random_paths.metadata.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = pretrained_metadata['checkpoint_sha256']
	random_paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match='checkpoint_sha256 must differ'):
		inspect_embedding_pair(config)


def test_embedding_pair_records_validated_model_sources(tmp_path: Path) -> None:
	config = _config(tmp_path)
	pretrained_checkpoint, random_checkpoint = _write_pair(config)
	geometry = inspect_embedding_pair(config)

	assert geometry.pretrained_model_source == {
		'role': 'pretrained',
		'checkpoint_path': str(pretrained_checkpoint),
		'checkpoint_sha256': file_sha256(pretrained_checkpoint),
		'model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
	}
	assert geometry.random_model_source == {
		'role': 'random',
		'checkpoint_path': str(random_checkpoint),
		'checkpoint_sha256': file_sha256(random_checkpoint),
		'random_encoder_baseline': True,
		'pretrained_weights_loaded': False,
		'seed': 42,
		'checkpoint_kind': 'random_init',
		'reference_checkpoint': str(pretrained_checkpoint),
		'reference_checkpoint_sha256': file_sha256(pretrained_checkpoint),
		'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
	}


@pytest.mark.parametrize(
	('section', 'key', 'value', 'message'),
	[
		('metadata', 'random_encoder_baseline', False, 'random_encoder_baseline'),
		('metadata', 'pretrained_weights_loaded', True, 'pretrained_weights_loaded'),
		('metadata', 'seed', 43, 'metadata.seed'),
		(
			'training_state',
			'checkpoint_kind',
			'epoch',
			'training_state.checkpoint_kind',
		),
		(
			'metadata',
			'reference_checkpoint',
			'/wrong/latest.pt',
			'reference_checkpoint',
		),
		('metadata', 'reference_model_tag', 'wrong-model', 'reference_model_tag'),
	],
)
def test_random_checkpoint_source_metadata_is_required(
	tmp_path: Path,
	section: str,
	key: str,
	value: object,
	message: str,
) -> None:
	config = _config(tmp_path)
	_, random_checkpoint = _write_pair(config)
	payload = torch.load(random_checkpoint, map_location='cpu', weights_only=False)
	payload[section][key] = value
	torch.save(payload, random_checkpoint)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	metadata = json.loads(random_paths.metadata.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = file_sha256(random_checkpoint)
	random_paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match=message):
		inspect_embedding_pair(config)


def test_random_embedding_rejects_a_regular_trained_checkpoint(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	_, random_checkpoint = _write_pair(config)
	torch.save(
		{
			'model_state_dict': {'weight': torch.ones(1)},
			'training_state': {'checkpoint_kind': 'epoch'},
		},
		random_checkpoint,
	)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	metadata = json.loads(random_paths.metadata.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = file_sha256(random_checkpoint)
	random_paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(TypeError, match='checkpoint metadata must be a mapping'):
		inspect_embedding_pair(config)


def test_pretrained_and_random_embedding_directories_cannot_be_reversed(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	reversed_config = replace(
		config,
		pretrained_embeddings=config.random_embeddings,
		random_embeddings=config.pretrained_embeddings,
	)
	with pytest.raises(ValueError, match=r'expected Parihaka full_100ep/latest\.pt'):
		inspect_embedding_pair(reversed_config)


def test_pretrained_embedding_requires_expected_parihaka_checkpoint(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	unexpected = tmp_path / 'other_model' / 'latest.pt'
	unexpected.parent.mkdir(parents=True)
	torch.save({'model_state_dict': {'weight': torch.ones(1)}}, unexpected)
	pretrained_paths = output_paths(config.pretrained_embeddings, config.survey_id)
	metadata = json.loads(pretrained_paths.metadata.read_text(encoding='utf-8'))
	metadata['checkpoint_path'] = str(unexpected)
	metadata['checkpoint_sha256'] = file_sha256(unexpected)
	pretrained_paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match=r'expected Parihaka full_100ep/latest\.pt'):
		inspect_embedding_pair(config)


def test_embedding_checkpoint_sha_must_match_actual_file(tmp_path: Path) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	metadata = json.loads(random_paths.metadata.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = 'c' * 64
	random_paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	with pytest.raises(ValueError, match='does not match checkpoint file'):
		inspect_embedding_pair(config)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('core_size_tokens', [4, 8, 8]),
		('context_halo_tokens', [0, 1, 1]),
	],
)
def test_channel_benchmark_tile_settings_are_fixed(
	tmp_path: Path,
	key: str,
	value: list[int],
) -> None:
	raw = _config_mapping(tmp_path)
	configured = channel_decoder_config_from_mapping(raw)
	assert configured.tiles == DecoderTiles((8, 8, 8), (1, 1, 1))
	assert isinstance(raw['tiles'], dict)
	raw['tiles'][key] = value
	with pytest.raises(ValueError, match='tile settings differ'):
		channel_decoder_config_from_mapping(raw)


@pytest.mark.parametrize(
	('split', 'missing_class'),
	[
		('train', 'Channel'),
		('train', 'non-Channel'),
		('validation', 'Channel'),
		('validation', 'non-Channel'),
		('test', 'Channel'),
		('test', 'non-Channel'),
	],
)
def test_every_split_must_contain_both_channel_classes(
	tmp_path: Path,
	split: str,
	missing_class: str,
) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	layout_path = _write_layout(tmp_path)
	labels = np.ones((16, 16, 16), dtype=np.int8)
	labels[:, :, ::2] = 5
	mask = split_mask_for_crop(
		shape=labels.shape,
		start_xyz=(0, 0, 0),
		train=SectionLines((0,), (0,)),
		validation=SectionLines((12,), (12,)),
		test=SectionLines((13,), (13,)),
		split=split,
	)
	labels[mask] = 1 if missing_class == 'Channel' else 5
	_write_labels(config, labels)
	with pytest.raises(
		ValueError,
		match=rf'{split} sections must contain both Channel and non-Channel voxels',
	):
		inspect_channel_decoder_job(
			config,
			model='pretrained',
			layout_id='layout_000',
			data_size='small',
			layout_config=layout_path,
		)


def test_decoder_preflight_rejects_replaced_label_content(tmp_path: Path) -> None:
	config = _config(tmp_path)
	_write_pair(config)
	labels = np.ones((16, 16, 16), dtype=np.int8)
	labels[:, :, ::2] = 5
	_write_labels(config, labels)
	np.save(config.labels, np.roll(labels, 1, axis=2), allow_pickle=False)
	with pytest.raises(ValueError, match='SHA-256 does not match'):
		inspect_channel_decoder_job(
			config,
			model='pretrained',
			layout_id='layout_000',
			data_size='small',
			layout_config=_write_layout(tmp_path),
		)


def test_decoder_initialization_and_tile_order_are_paired(tmp_path: Path) -> None:
	config = _config(tmp_path)
	first = decoder_initial_state_sha256(config.decoder, (8, 8, 8), 42000)
	second = decoder_initial_state_sha256(config.decoder, (8, 8, 8), 42000)
	assert first == second
	assert deterministic_tile_order(17, 42000, 3) == deterministic_tile_order(
		17, 42000, 3
	)
	assert len(set(deterministic_tile_order(17, 42000, 3))) == 17


def test_channel_metrics() -> None:
	metrics = channel_metrics(np.asarray([[8, 2], [3, 7]], dtype=np.int64))
	assert metrics['channel_iou'] == pytest.approx(7 / 12)
	assert metrics['channel_f1'] == pytest.approx(14 / 19)
	assert metrics['channel_precision'] == pytest.approx(7 / 9)
	assert metrics['channel_recall'] == pytest.approx(7 / 10)
	assert metrics['balanced_accuracy'] == pytest.approx((0.8 + 0.7) / 2)


def test_one_job_max_steps_resume_and_evaluate(  # noqa: PLR0915
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	config = replace(
		config,
		train=replace(config.train, epochs=1),
		tiles=DecoderTiles((1, 1, 1), (1, 1, 1)),
	)
	metadata = {
		'survey_id': config.survey_id,
		'source_amplitude_path': '/data/parihaka_amplitude.npy',
		'patch_size': [8, 8, 8],
		'token_grid_shape': [1, 1, 1],
		'volume_shape_xyz': [8, 8, 8],
		'embedding_dim': 384,
		'model_geometry': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
		'window_size': [8, 8, 8],
		'overlap': [0, 0, 0],
		'output_dtype': 'float16',
		'min_token_valid_fraction': 0.5,
		'normalization_stats_path': '/data/parihaka_stats.json',
		'preprocessing': {'mode': 'same'},
		'zero_mask': {'enabled': True},
		'precision': {'device_type': 'cpu', 'autocast': False},
		'pretraining_objective': {'name': 'masked_autoencoding'},
	}
	pretrained_checkpoint, random_checkpoint = _write_model_checkpoints(config)
	pretrained_checkpoint_sha256 = file_sha256(pretrained_checkpoint)
	random_checkpoint_sha256 = file_sha256(random_checkpoint)
	for root, checkpoint_path, checkpoint_sha256 in (
		(
			config.pretrained_embeddings,
			pretrained_checkpoint,
			pretrained_checkpoint_sha256,
		),
		(config.random_embeddings, random_checkpoint, random_checkpoint_sha256),
	):
		paths = output_paths(root, config.survey_id)
		root.mkdir(parents=True)
		np.save(paths.embeddings, np.zeros((1, 1, 1, 384), dtype=np.float16))
		np.save(paths.valid_tokens, np.ones((1, 1, 1), dtype=np.bool_))
		paths.metadata.write_text(
			json.dumps(
				{
					**metadata,
					'checkpoint_path': str(checkpoint_path),
					'checkpoint_sha256': checkpoint_sha256,
				}
			),
			encoding='utf-8',
		)
	labels = np.ones((8, 8, 8), dtype=np.int8)
	labels[:, :, ::2] = 5
	_write_labels(config, labels)
	training_lines = (0, 1, 2, 3, 6, 7)
	layout_path = tmp_path / 'layouts.yaml'
	layout_path.write_text(
		yaml.safe_dump(
			{
				'validation': {'inline': [4], 'crossline': [4]},
				'test': {'inline': [5], 'crossline': [5]},
				'layouts': {
					f'layout_{index:03d}': {
						'inline': [
							training_lines[(index + offset) % len(training_lines)]
							for offset in range(4)
						],
						'crossline': [
							training_lines[(index + offset) % len(training_lines)]
							for offset in range(4)
						],
					}
					for index in range(5)
				},
			}
		),
		encoding='utf-8',
	)
	plan = inspect_channel_decoder_job(
		config,
		model='pretrained',
		layout_id='layout_000',
		data_size='small',
		layout_config=layout_path,
	)
	assert run_channel_decoder_job(plan, device='cpu', max_steps=1) is None
	latest = plan.output_dir / 'latest.pt'
	assert latest.is_file()
	payload = torch.load(latest, map_location='cpu', weights_only=False)
	identity = payload['run_identity']
	assert identity['embedding'] == {
		'checkpoint_path': str(pretrained_checkpoint),
		'checkpoint_sha256': pretrained_checkpoint_sha256,
		'model_source': {
			'role': 'pretrained',
			'checkpoint_path': str(pretrained_checkpoint),
			'checkpoint_sha256': pretrained_checkpoint_sha256,
			'model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		},
		'common_metadata': {
			key: metadata[key]
			for key in (
				'survey_id',
				'source_amplitude_path',
				'volume_shape_xyz',
				'model_geometry',
				'patch_size',
				'token_grid_shape',
				'window_size',
				'overlap',
				'output_dtype',
				'min_token_valid_fraction',
				'normalization_stats_path',
				'preprocessing',
				'zero_mask',
				'precision',
				'pretraining_objective',
			)
		},
	}
	assert identity['label_path'] == str(config.labels)
	assert identity['label_metadata_path'] == str(config.labels_metadata)
	assert identity['prepared_label_identity'] == {
		'labels_sha256': file_sha256(config.labels),
		'source_npz_path': '/data/parihaka_labels.npz',
		'source_key': 'labels',
		'shape': [8, 8, 8],
		'dtype': 'int8',
		'class_definition': {
			'positive_class_id': 5,
			'negative_class_ids': [1, 2, 3, 4, 6],
		},
	}
	assert identity['decoder']['spec'] == config.decoder.spec
	assert identity['decoder_initial_state_sha256'] == decoder_initial_state_sha256(
		config.decoder, plan.geometry.patch_size_xyz, config.train.seed
	)
	assert identity['geometry']['token_grid_shape_xyz'] == [1, 1, 1]
	assert identity['tiles'] == {
		'core_size_tokens': [1, 1, 1],
		'context_halo_tokens': [1, 1, 1],
	}
	assert identity['split_class_counts'] == {
		split: list(plan.split_counts[split])
		for split in ('train', 'validation', 'test')
	}
	assert identity['tile_counts'] == dict(plan.tile_counts)
	pretrained_metadata_path = plan.geometry.pretrained.metadata
	changed_metadata = json.loads(
		pretrained_metadata_path.read_text(encoding='utf-8')
	)
	original_checkpoint = pretrained_checkpoint.read_bytes()
	torch.save(
		{'model_state_dict': {'weight': torch.full((1,), 2.0)}},
		pretrained_checkpoint,
	)
	changed_metadata['checkpoint_sha256'] = file_sha256(pretrained_checkpoint)
	pretrained_metadata_path.write_text(
		json.dumps(changed_metadata), encoding='utf-8'
	)
	changed_plan = inspect_channel_decoder_job(
		config,
		model='pretrained',
		layout_id='layout_000',
		data_size='small',
		layout_config=layout_path,
	)
	with pytest.raises(ValueError, match='resume checkpoint does not match'):
		run_channel_decoder_job(changed_plan, device='cpu', resume=latest)
	pretrained_checkpoint.write_bytes(original_checkpoint)
	changed_metadata['checkpoint_sha256'] = pretrained_checkpoint_sha256
	pretrained_metadata_path.write_text(
		json.dumps(changed_metadata), encoding='utf-8'
	)
	metrics_path = run_channel_decoder_job(plan, device='cpu', resume=latest)
	assert metrics_path is not None and metrics_path.is_file()
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	assert metrics['best_epoch'] == 0
	assert metrics['benchmark_identity'] == identity
	assert metrics['supervision'] == {
		'axis_mapping': {'inline': 'x', 'crossline': 'y'},
		'train_inline': [0],
		'train_crossline': [0],
		'validation_inline': [4],
		'validation_crossline': [4],
		'test_inline': [5],
		'test_crossline': [5],
		'split_class_counts': {
			split: list(plan.split_counts[split])
			for split in ('train', 'validation', 'test')
		},
	}
	assert set(metrics['test']) == {
		'channel_iou',
		'channel_f1',
		'channel_precision',
		'channel_recall',
		'balanced_accuracy',
	}
	assert not list(plan.output_dir.glob('*probability*'))
