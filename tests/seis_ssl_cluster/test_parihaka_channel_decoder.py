# ruff: noqa: CPY001, PT018, TC003

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.embedding.writer import output_paths
from seis_ssl_cluster.parihaka.channel_decoder import (
	ChannelDecoderConfig,
	DecoderArchitecture,
	DecoderTiles,
	DecoderTrain,
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


def _write_pair(config: ChannelDecoderConfig) -> None:
	metadata = {
		'patch_size': [8, 8, 8],
		'token_grid_shape': [2, 2, 2],
		'volume_shape_xyz': [16, 16, 16],
		'embedding_dim': 384,
		'preprocessing': {'mode': 'same'},
		'zero_mask': {'enabled': True},
	}
	for root in (config.pretrained_embeddings, config.random_embeddings):
		paths = output_paths(root, config.survey_id)
		root.mkdir(parents=True)
		np.save(paths.embeddings, np.zeros((2, 2, 2, 384), dtype=np.float16))
		np.save(paths.valid_tokens, np.ones((2, 2, 2), dtype=np.bool_))
		paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')


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


def test_one_job_max_steps_resume_and_evaluate(tmp_path: Path) -> None:
	config = _config(tmp_path)
	config = replace(
		config,
		train=replace(config.train, epochs=1),
		tiles=DecoderTiles((1, 1, 1), (1, 1, 1)),
	)
	metadata = {
		'patch_size': [8, 8, 8],
		'token_grid_shape': [1, 1, 1],
		'volume_shape_xyz': [8, 8, 8],
		'embedding_dim': 384,
		'preprocessing': {'mode': 'same'},
		'zero_mask': {'enabled': True},
	}
	for root in (config.pretrained_embeddings, config.random_embeddings):
		paths = output_paths(root, config.survey_id)
		root.mkdir(parents=True)
		np.save(paths.embeddings, np.zeros((1, 1, 1, 384), dtype=np.float16))
		np.save(paths.valid_tokens, np.ones((1, 1, 1), dtype=np.bool_))
		paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
	labels = np.ones((8, 8, 8), dtype=np.int8)
	labels[:, :, ::2] = 5
	np.save(config.labels, labels)
	layout_path = tmp_path / 'layouts.yaml'
	layout_path.write_text(
		yaml.safe_dump(
			{
				'axis_mapping': {'inline': 'x', 'crossline': 'y'},
				'validation': {'inline': [4], 'crossline': [4]},
				'test': {'inline': [5], 'crossline': [5]},
				'layouts': {
					f'layout_{index:03d}': {
						'inline': [0, 1, 2, 3],
						'crossline': [0, 1, 2, 3],
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
	metrics_path = run_channel_decoder_job(plan, device='cpu', resume=latest)
	assert metrics_path is not None and metrics_path.is_file()
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	assert metrics['best_epoch'] == 0
	assert set(metrics['test']) == {
		'channel_iou',
		'channel_f1',
		'channel_precision',
		'channel_recall',
		'balanced_accuracy',
	}
	assert not list(plan.output_dir.glob('*probability*'))
