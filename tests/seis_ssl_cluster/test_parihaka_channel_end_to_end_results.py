
from __future__ import annotations

import csv
import json
import statistics
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
	from pathlib import Path

from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_decoder import (
	CHANNEL_PRETRAINED_MODEL_TAG,
)
from seis_ssl_cluster.parihaka.channel_end_to_end_results import (
	ChannelEndToEndSummaryConfig,
	inspect_channel_end_to_end_results,
	summarize_channel_end_to_end,
	summarize_channel_four_way,
)
from seis_ssl_cluster.parihaka.channel_results import ChannelSummaryConfig

_PRETRAINED_CHECKPOINT = (
	'/artifacts/pretraining/parihaka/facies_benchmark_v1/'
	f'{CHANNEL_PRETRAINED_MODEL_TAG}/full_100ep/latest.pt'
)
_RANDOM_CHECKPOINT = '/artifacts/pretraining/random/mae_random_seed42.pt'
_DELTA_BY_LAYOUT = (0.1, 0.2, 0.0, -0.1, 0.3)


def _lines(layout_index: int, data_size: str) -> dict[str, object]:
	prefix = DATA_SIZE_PREFIX[data_size]
	inline = [10 + layout_index * 10 + index for index in range(4)]
	crossline = [20 + layout_index * 10 + index for index in range(4)]
	return {
		'train_inline': inline[:prefix],
		'train_crossline': crossline[:prefix],
		'validation_inline': [100, 101],
		'validation_crossline': [200, 201],
		'test_inline': [102, 103],
		'test_crossline': [202, 203],
		'split_class_counts': {
			'train': [1000 * prefix, 100 * prefix],
			'validation': [2000, 200],
			'test': [3000, 300],
		},
		'tile_counts': {'train': 10 * prefix, 'validation': 4, 'test': 4},
	}


def _prepared_label_identity() -> dict[str, object]:
	return {
		'labels_sha256': 'd' * 64,
		'source_npz_path': '/data/parihaka_labels.npz',
		'source_key': 'labels',
		'shape': [100, 200, 300],
		'dtype': 'int8',
		'class_definition': {
			'positive_class_id': 5,
			'negative_class_ids': [1, 2, 3, 4, 6],
		},
	}


def _model_geometry() -> dict[str, object]:
	return {
		'in_channels': 1,
		'out_channels': 1,
		'patch_size': [8, 8, 8],
		'encoder_dim': 384,
		'encoder_depth': 8,
		'encoder_heads': 6,
		'decoder_dim': 256,
		'decoder_depth': 4,
		'decoder_heads': 4,
	}


def _encoder_source(encoder_init: str) -> dict[str, object]:
	common = {
		'role': encoder_init,
		'checkpoint_path': (
			_PRETRAINED_CHECKPOINT
			if encoder_init == 'pretrained'
			else _RANDOM_CHECKPOINT
		),
		'checkpoint_sha256': ('a' if encoder_init == 'pretrained' else 'b') * 64,
		'model_geometry': _model_geometry(),
		'parameter_dtype': 'float32',
		'trainable_modules': ['patch_projection', 'encoder'],
		'initial_state_sha256': (
			'e' if encoder_init == 'pretrained' else 'f'
		)
		* 64,
	}
	if encoder_init == 'pretrained':
		return {**common, 'model_tag': CHANNEL_PRETRAINED_MODEL_TAG}
	return {
		**common,
		'random_encoder_baseline': True,
		'pretrained_weights_loaded': False,
		'seed': 42,
		'checkpoint_kind': 'random_init',
		'reference_checkpoint': _PRETRAINED_CHECKPOINT,
		'reference_checkpoint_sha256': 'a' * 64,
		'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
	}


def _end_identity(
	encoder_init: str, layout_index: int, layout_id: str, data_size: str
) -> dict[str, object]:
	lines = _lines(layout_index, data_size)
	return {
		'encoder_init': encoder_init,
		'layout_id': layout_id,
		'data_size': data_size,
		'encoder_source': _encoder_source(encoder_init),
		'encoder_initial_states': {
			'pretrained_sha256': 'e' * 64,
			'random_sha256': 'f' * 64,
		},
		'reference_input': {
			'amplitude_path': '/data/parihaka.npy',
			'normalization_stats_path': '/data/stats.json',
			'reference_metadata_path': '/embeddings/parihaka.metadata.json',
			'reference_metadata_sha256': '1' * 64,
			'reference_valid_tokens_path': '/embeddings/parihaka.valid_tokens.npy',
			'reference_valid_tokens_sha256': '2' * 64,
			'preprocessing': {'normalized_clip_abs': 8.0},
			'zero_mask': {'enabled': True},
			'min_token_valid_fraction': 0.5,
			'patch_size': [8, 8, 8],
			'volume_shape': [100, 200, 300],
			'token_grid_shape': [13, 25, 38],
		},
		'labels': {
			'path': '/data/parihaka_labels.npy',
			'metadata_path': '/data/parihaka_labels_metadata.json',
			'prepared_label_identity': _prepared_label_identity(),
		},
		'supervision': {
			'train_lines': {
				'inline': lines['train_inline'],
				'crossline': lines['train_crossline'],
			},
			'validation_lines': {
				'inline': lines['validation_inline'],
				'crossline': lines['validation_crossline'],
			},
			'test_lines': {
				'inline': lines['test_inline'],
				'crossline': lines['test_crossline'],
			},
			'split_class_counts': lines['split_class_counts'],
			'tile_counts': lines['tile_counts'],
			'class_weights': [0.55, 5.5],
		},
		'decoder': {
			'architecture': {
				'spec': 'frozen_embedding_decoder_v1',
				'embedding_dim': 384,
				'class_count': 2,
				'hidden_channels': [128, 64, 32],
				'upsample_factors': [[2, 2, 2]] * 3,
				'upsample_mode': 'nearest',
				'normalization': 'voxelwise_layer_norm',
			},
			'initial_state_sha256': 'c' * 64,
		},
		'optimizer': {
			'encoder_learning_rate': 0.0001,
			'decoder_learning_rate': 0.001,
			'weight_decay': 0.0001,
			'parameter_group_names': ['encoder', 'decoder'],
		},
		'training': {
			'epochs': 50,
			'batch_size': 1,
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'gradient_clip_norm': 1.0,
		},
		'tiles': {
			'core_size_tokens': [8, 8, 8],
			'context_halo_tokens': [1, 1, 1],
		},
		'runtime': {
			'resolved_device_type': 'cuda',
			'amp_enabled': True,
			'autocast_dtype': 'float16',
			'grad_scaler_enabled': True,
		},
	}


def _end_metrics(
	encoder_init: str, layout_index: int, layout_id: str, data_size: str
) -> dict[str, object]:
	lines = _lines(layout_index, data_size)
	scratch = 0.5 + layout_index / 100
	channel_iou = (
		scratch + _DELTA_BY_LAYOUT[layout_index]
		if encoder_init == 'pretrained'
		else scratch
	)
	return {
		'encoder_init': encoder_init,
		'condition_name': (
			'finetune_pretrained'
			if encoder_init == 'pretrained'
			else 'train_from_scratch'
		),
		'layout_id': layout_id,
		'data_size': data_size,
		'supervision': {
			'axis_mapping': {'inline': 'x', 'crossline': 'y'},
			'train_inline': lines['train_inline'],
			'train_crossline': lines['train_crossline'],
			'validation_inline': lines['validation_inline'],
			'validation_crossline': lines['validation_crossline'],
			'test_inline': lines['test_inline'],
			'test_crossline': lines['test_crossline'],
			'split_class_counts': lines['split_class_counts'],
			'tile_counts': lines['tile_counts'],
		},
		'class_weights': [0.55, 5.5],
		'benchmark_identity': _end_identity(
			encoder_init, layout_index, layout_id, data_size
		),
		'test': {'channel_iou': channel_iou},
	}


def _frozen_identity(
	model: str, layout_index: int, layout_id: str, data_size: str
) -> dict[str, object]:
	lines = _lines(layout_index, data_size)
	checkpoint = _PRETRAINED_CHECKPOINT if model == 'pretrained' else _RANDOM_CHECKPOINT
	checkpoint_sha = ('a' if model == 'pretrained' else 'b') * 64
	model_source = (
		{
			'role': 'pretrained',
			'checkpoint_path': checkpoint,
			'checkpoint_sha256': checkpoint_sha,
			'model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		}
		if model == 'pretrained'
		else {
			'role': 'random',
			'checkpoint_path': checkpoint,
			'checkpoint_sha256': checkpoint_sha,
			'random_encoder_baseline': True,
			'pretrained_weights_loaded': False,
			'seed': 42,
			'checkpoint_kind': 'random_init',
			'reference_checkpoint': _PRETRAINED_CHECKPOINT,
			'reference_checkpoint_sha256': 'a' * 64,
			'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		}
	)
	return {
		'model': model,
		'layout_id': layout_id,
		'data_size': data_size,
		'embedding': {
			'checkpoint_path': checkpoint,
			'checkpoint_sha256': checkpoint_sha,
			'model_source': model_source,
			'common_metadata': {
				'survey_id': 'parihaka_full',
				'source_amplitude_path': '/data/parihaka.npy',
				'volume_shape_xyz': [100, 200, 300],
				'model_geometry': _model_geometry(),
				'patch_size': [8, 8, 8],
				'token_grid_shape': [13, 25, 38],
				'window_size': [16, 16, 16],
				'overlap': [8, 8, 8],
				'output_dtype': 'float16',
				'min_token_valid_fraction': 0.5,
				'normalization_stats_path': '/data/stats.json',
				'preprocessing': {'normalized_clip_abs': 8.0},
				'zero_mask': {'enabled': True},
				'precision': {'autocast': True},
				'pretraining_objective': {'name': 'mae'},
			},
		},
		'decoder_initial_state_sha256': 'c' * 64,
		'label_path': '/data/parihaka_labels.npy',
		'label_metadata_path': '/data/parihaka_labels_metadata.json',
		'prepared_label_identity': _prepared_label_identity(),
		'train_lines': {
			'inline': lines['train_inline'],
			'crossline': lines['train_crossline'],
		},
		'validation': {
			'inline': lines['validation_inline'],
			'crossline': lines['validation_crossline'],
		},
		'test': {
			'inline': lines['test_inline'],
			'crossline': lines['test_crossline'],
		},
		'geometry': {
			'embedding_shape': [13, 25, 38, 384],
			'volume_shape_xyz': [100, 200, 300],
			'token_grid_shape_xyz': [13, 25, 38],
			'patch_size_xyz': [8, 8, 8],
		},
		'class_weights': [0.55, 5.5],
		'decoder': {
			'spec': 'frozen_embedding_decoder_v1',
			'embedding_dim': 384,
			'class_count': 2,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2]] * 3,
			'upsample_mode': 'trilinear',
			'normalization': 'group_norm',
		},
		'training': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 0.001,
			'weight_decay': 0.0001,
			'class_weight': 'balanced_train_voxels',
			'sampling_mode': 'all_tiles_once_per_epoch',
			'seed': 42000,
			'amp': True,
			'gradient_clip_norm': 1.0,
		},
		'tiles': {
			'core_size_tokens': [8, 8, 8],
			'context_halo_tokens': [1, 1, 1],
		},
		'split_class_counts': lines['split_class_counts'],
		'tile_counts': lines['tile_counts'],
	}


def _frozen_metrics(
	model: str, layout_index: int, layout_id: str, data_size: str
) -> dict[str, object]:
	lines = _lines(layout_index, data_size)
	return {
		'model': model,
		'layout_id': layout_id,
		'data_size': data_size,
		'benchmark_identity': _frozen_identity(
			model, layout_index, layout_id, data_size
		),
		'supervision': {
			'axis_mapping': {'inline': 'x', 'crossline': 'y'},
			'train_inline': lines['train_inline'],
			'train_crossline': lines['train_crossline'],
			'validation_inline': lines['validation_inline'],
			'validation_crossline': lines['validation_crossline'],
			'test_inline': lines['test_inline'],
			'test_crossline': lines['test_crossline'],
			'split_class_counts': lines['split_class_counts'],
		},
		'class_weights': [0.55, 5.5],
		'test': {'channel_iou': 0.65 if model == 'pretrained' else 0.6},
	}


def _write_complete(
	end_config: ChannelEndToEndSummaryConfig,
	frozen_config: ChannelSummaryConfig | None = None,
) -> None:
	for encoder_init in ('pretrained', 'random'):
		for layout_index, layout_id in enumerate(LAYOUT_IDS):
			for data_size in DATA_SIZE_PREFIX:
				path = (
					end_config.runs_root
					/ f'encoder_init={encoder_init}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'metrics.json'
				)
				path.parent.mkdir(parents=True, exist_ok=True)
				path.write_text(
					json.dumps(
						_end_metrics(
							encoder_init, layout_index, layout_id, data_size
						)
					),
					encoding='utf-8',
				)
	if frozen_config is None:
		return
	for model in ('pretrained', 'random'):
		for layout_index, layout_id in enumerate(LAYOUT_IDS):
			for data_size in DATA_SIZE_PREFIX:
				path = (
					frozen_config.runs_root
					/ f'model={model}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'metrics.json'
				)
				path.parent.mkdir(parents=True, exist_ok=True)
				path.write_text(
					json.dumps(
						_frozen_metrics(model, layout_index, layout_id, data_size)
					),
					encoding='utf-8',
				)


def _end_config(tmp_path: Path) -> ChannelEndToEndSummaryConfig:
	return ChannelEndToEndSummaryConfig(
		tmp_path / 'end_runs', tmp_path / 'summary', tmp_path / 'four_way'
	)


def _mutate_end(
	config: ChannelEndToEndSummaryConfig,
	encoder_init: str,
	mutator: object,
	*,
	layout_id: str = 'layout_000',
	data_size: str = 'small',
) -> None:
	path = (
		config.runs_root
		/ f'encoder_init={encoder_init}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ 'metrics.json'
	)
	payload = json.loads(path.read_text(encoding='utf-8'))
	assert callable(mutator)
	mutator(payload)
	path.write_text(json.dumps(payload), encoding='utf-8')


def test_end_to_end_summary_requires_all_30_jobs(tmp_path: Path) -> None:
	config = _end_config(tmp_path)
	with pytest.raises(FileNotFoundError, match='all 30 jobs'):
		inspect_channel_end_to_end_results(config)


def test_complete_end_to_end_summary_and_paired_statistics(tmp_path: Path) -> None:
	config = _end_config(tmp_path)
	_write_complete(config)
	paths = summarize_channel_end_to_end(config)
	assert {path.name for path in paths} == {
		'comparison.csv',
		'summary.json',
		'summary.md',
	}
	payload = json.loads((config.output_dir / 'summary.json').read_text())
	small = payload['by_size']['small']
	assert small['paired_mean'] == pytest.approx(statistics.fmean(_DELTA_BY_LAYOUT))
	assert small['paired_median'] == pytest.approx(statistics.median(_DELTA_BY_LAYOUT))
	assert small['sample_standard_deviation'] == pytest.approx(
		statistics.stdev(_DELTA_BY_LAYOUT)
	)
	assert (small['pretrained_wins'], small['ties'], small['pretrained_losses']) == (
		3,
		1,
		1,
	)
	assert list(small['layout_deltas']) == list(LAYOUT_IDS)
	with (config.output_dir / 'comparison.csv').open(newline='') as file_obj:
		rows = list(csv.DictReader(file_obj))
	assert len(rows) == 15
	assert float(rows[0]['end_to_end_pretraining_delta']) == pytest.approx(0.1)
	with pytest.raises(FileExistsError, match='already exist'):
		summarize_channel_end_to_end(config)


@pytest.mark.parametrize(
	('mutator', 'message'),
	[
		(
			lambda value: value.__setitem__('condition_name', 'train_from_scratch'),
			'incorrect condition_name',
		),
		(
			lambda value: value['benchmark_identity']['encoder_source'].__setitem__(
				'role', 'random'
			),
			'encoder role',
		),
		(
			lambda value: value['benchmark_identity']['encoder_source'].__setitem__(
				'model_geometry', {'encoder_depth': 9}
			),
			'encoder source',
		),
		(
			lambda value: value['benchmark_identity']['optimizer'].__setitem__(
				'weight_decay', 0.2
			),
			'optimizer drift',
		),
		(
			lambda value: value['benchmark_identity']['runtime'].__setitem__(
				'amp_enabled', bool(0)
			),
			'runtime drift',
		),
		(
			lambda value: value['benchmark_identity']['decoder'].__setitem__(
				'initial_state_sha256', '9' * 64
			),
			'decoder drift',
		),
	],
)
def test_end_to_end_summary_rejects_identity_drift(
	tmp_path: Path, mutator: object, message: str
) -> None:
	config = _end_config(tmp_path)
	_write_complete(config)
	_mutate_end(config, 'pretrained', mutator)
	with pytest.raises(ValueError, match=message):
		inspect_channel_end_to_end_results(config)


def test_end_to_end_summary_rejects_same_checkpoint_sha(tmp_path: Path) -> None:
	config = _end_config(tmp_path)
	_write_complete(config)
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZE_PREFIX:
			_mutate_end(
				config,
				'random',
				lambda value: value['benchmark_identity']['encoder_source'].__setitem__(
					'checkpoint_sha256', 'a' * 64
				),
				layout_id=layout_id,
				data_size=data_size,
			)
	with pytest.raises(ValueError, match='checkpoint SHA-256 must differ'):
		inspect_channel_end_to_end_results(config)


@pytest.mark.parametrize(
	('field', 'value', 'message'),
	[
		('class_weights', [1.0, 2.0], 'supervision mismatch'),
		('split_class_counts', {'train': [1, 1]}, 'split/tile counts'),
		('tile_counts', {'train': 99, 'validation': 4, 'test': 4}, 'supervision'),
	],
)
def test_end_to_end_summary_rejects_paired_supervision_drift(
	tmp_path: Path, field: str, value: object, message: str
) -> None:
	config = _end_config(tmp_path)
	_write_complete(config)

	def mutate(payload: dict[str, object]) -> None:
		identity = payload['benchmark_identity']
		assert isinstance(identity, dict)
		supervision = identity['supervision']
		assert isinstance(supervision, dict)
		supervision[field] = value
		if field == 'class_weights':
			payload['class_weights'] = value
		else:
			metrics_supervision = payload['supervision']
			assert isinstance(metrics_supervision, dict)
			metrics_supervision[field] = value

	_mutate_end(config, 'random', mutate)
	with pytest.raises((TypeError, ValueError), match=message):
		inspect_channel_end_to_end_results(config)


def test_end_to_end_summary_preserves_layout_nesting_and_uniqueness(
	tmp_path: Path,
) -> None:
	config = _end_config(tmp_path)
	_write_complete(config)

	def mutate(payload: dict[str, object]) -> None:
		identity = payload['benchmark_identity']
		assert isinstance(identity, dict)
		supervision = identity['supervision']
		assert isinstance(supervision, dict)
		lines = supervision['train_lines']
		assert isinstance(lines, dict)
		lines['inline'] = [11, 10]
		metrics_supervision = payload['supervision']
		assert isinstance(metrics_supervision, dict)
		metrics_supervision['train_inline'] = [11, 10]

	for encoder_init in ('pretrained', 'random'):
		_mutate_end(
			config,
			encoder_init,
			mutate,
			layout_id='layout_000',
			data_size='medium',
		)
	with pytest.raises(ValueError, match='not nested'):
		inspect_channel_end_to_end_results(config)


def test_four_way_summary_validates_pairing_and_has_no_cross_regime_delta(
	tmp_path: Path,
) -> None:
	end_config = _end_config(tmp_path)
	frozen_config = ChannelSummaryConfig(
		tmp_path / 'frozen_runs', tmp_path / 'frozen_summary'
	)
	_write_complete(end_config, frozen_config)
	paths = summarize_channel_four_way(end_config, frozen_config)
	assert {path.name for path in paths} == {
		'four_way_comparison.csv',
		'four_way_summary.json',
		'four_way_summary.md',
	}
	with paths[0].open(newline='') as file_obj:
		rows = list(csv.DictReader(file_obj))
	assert len(rows) == 15
	assert float(rows[0]['frozen_representation_delta']) == pytest.approx(0.05)
	assert float(rows[0]['end_to_end_pretraining_delta']) == pytest.approx(0.1)
	assert not any('cross_regime' in key or 'fine_tuning' in key for key in rows[0])
	markdown = paths[2].read_text(encoding='utf-8')
	assert 'different scientific questions' in markdown
	assert 'input context differs' in markdown
	with pytest.raises(FileExistsError, match='already exist'):
		summarize_channel_four_way(end_config, frozen_config)


@pytest.mark.parametrize(
	('mutator', 'message'),
	[
		(
			lambda value: value['benchmark_identity'][
				'prepared_label_identity'
			].__setitem__('labels_sha256', '9' * 64),
			'label identity mismatch',
		),
		(
			lambda value: value['benchmark_identity']['decoder'].__setitem__(
				'class_count', 3
			),
			'class count mismatch',
		),
	],
)
def test_four_way_summary_rejects_cross_regime_identity_drift(
	tmp_path: Path, mutator: object, message: str
) -> None:
	end_config = _end_config(tmp_path)
	frozen_config = ChannelSummaryConfig(
		tmp_path / 'frozen_runs', tmp_path / 'frozen_summary'
	)
	_write_complete(end_config, frozen_config)
	for model in ('pretrained', 'random'):
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				path = (
					frozen_config.runs_root
					/ f'model={model}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'metrics.json'
				)
				payload = json.loads(path.read_text(encoding='utf-8'))
				assert callable(mutator)
				mutator(payload)
				path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match=message):
		summarize_channel_four_way(end_config, frozen_config)


@pytest.mark.parametrize(
	('drift', 'message'),
	[
		('supervision', 'four-way supervision mismatch'),
		('class_weights', 'class weight mismatch'),
		('split_counts', 'four-way supervision mismatch'),
	],
)
def test_four_way_summary_rejects_cross_regime_supervision_drift(
	tmp_path: Path, drift: str, message: str
) -> None:
	end_config = _end_config(tmp_path)
	frozen_config = ChannelSummaryConfig(
		tmp_path / 'frozen_runs', tmp_path / 'frozen_summary'
	)
	_write_complete(end_config, frozen_config)
	for model in ('pretrained', 'random'):
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				path = (
					frozen_config.runs_root
					/ f'model={model}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'metrics.json'
				)
				payload = json.loads(path.read_text(encoding='utf-8'))
				identity = payload['benchmark_identity']
				assert isinstance(identity, dict)
				supervision = payload['supervision']
				assert isinstance(supervision, dict)
				if drift == 'supervision':
					train_inline = supervision['train_inline']
					assert isinstance(train_inline, list)
					changed = [int(value) + 1000 for value in train_inline]
					supervision['train_inline'] = changed
					train_lines = identity['train_lines']
					assert isinstance(train_lines, dict)
					train_lines['inline'] = changed
				elif drift == 'class_weights':
					payload['class_weights'] = [0.6, 5.0]
					identity['class_weights'] = [0.6, 5.0]
				else:
					counts = supervision['split_class_counts']
					identity_counts = identity['split_class_counts']
					assert isinstance(counts, dict)
					assert isinstance(identity_counts, dict)
					counts['test'] = [3001, 300]
					identity_counts['test'] = [3001, 300]
				path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match=message):
		summarize_channel_four_way(end_config, frozen_config)


def test_four_way_summary_fails_when_either_job_set_is_missing(
	tmp_path: Path,
) -> None:
	end_config = _end_config(tmp_path)
	frozen_config = ChannelSummaryConfig(
		tmp_path / 'frozen_runs', tmp_path / 'frozen_summary'
	)
	_write_complete(end_config)
	with pytest.raises(FileNotFoundError, match='all 30 jobs'):
		summarize_channel_four_way(end_config, frozen_config)
