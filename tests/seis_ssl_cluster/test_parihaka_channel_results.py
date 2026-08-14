# ruff: noqa: TC003

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.parihaka.channel_data import (
	CHANNEL_TEST_MODE,
	DATA_SIZE_PREFIX,
	LAYOUT_IDS,
	selected_token_xyz_sha256,
)
from seis_ssl_cluster.parihaka.channel_decoder import (
	CHANNEL_PRETRAINED_MODEL_TAG,
)
from seis_ssl_cluster.parihaka.channel_results import (
	ChannelSummaryConfig,
	inspect_channel_benchmark_results,
	summarize_channel_benchmark,
)

_EXPECTED_PRETRAINED_CHECKPOINT = (
	'/artifacts/pretraining/parihaka/facies_benchmark_v1/'
	f'{CHANNEL_PRETRAINED_MODEL_TAG}/full_100ep/latest.pt'
)
_TEST_DEFINITION = {
	'mode': CHANNEL_TEST_MODE,
	'reserved_large_inline': list(range(10, 54)),
	'reserved_large_crossline': list(range(20, 64)),
}


def _supervision(layout_index: int, data_size: str) -> dict[str, object]:
	prefix = DATA_SIZE_PREFIX[data_size]
	inline = [10 + layout_index * 10 + index for index in range(4)]
	crossline = [20 + layout_index * 10 + index for index in range(4)]
	return {
		'axis_mapping': {'inline': 'x', 'crossline': 'y'},
		'train_inline': inline[:prefix],
		'train_crossline': crossline[:prefix],
		'validation_inline': [100, 101],
		'validation_crossline': [200, 201],
		'test_definition': _TEST_DEFINITION,
		'split_class_counts': {
			'train': [1000 * prefix, 100 * prefix],
			'validation': [2000, 200],
			'test': [3000, 300],
		},
	}


def _benchmark_identity(
	model: str, layout_index: int, layout_id: str, data_size: str
) -> dict[str, object]:
	supervision = _supervision(layout_index, data_size)
	prefix = DATA_SIZE_PREFIX[data_size]
	tokens = tuple((layout_index, index, 0) for index in range(prefix))
	actual = 1100 * prefix
	selection = {
		'semantics': 'stable_hash_partial_section_token_footprints_v1',
		'target_train_voxel_count': actual,
		'actual_train_voxel_count': actual,
		'count_error': 0,
		'relative_count_error': 0.0,
		'selected_token_xyz': [list(item) for item in tokens],
		'selected_token_xyz_sha256': selected_token_xyz_sha256(tokens),
		'per_line_contributions': {
			**{
				f'inline:{line}': 550
				for line in supervision['train_inline']
			},
			**{
				f'crossline:{line}': 550
				for line in supervision['train_crossline']
			},
		},
	}
	pretrained_checkpoint = _EXPECTED_PRETRAINED_CHECKPOINT
	checkpoint_path = (
		pretrained_checkpoint
		if model == 'pretrained'
		else '/artifacts/pretraining/random/mae_random_seed42.pt'
	)
	checkpoint_sha256 = ('a' if model == 'pretrained' else 'b') * 64
	model_source = (
		{
			'role': 'pretrained',
			'checkpoint_path': checkpoint_path,
			'checkpoint_sha256': checkpoint_sha256,
			'model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		}
		if model == 'pretrained'
		else {
			'role': 'random',
			'checkpoint_path': checkpoint_path,
			'checkpoint_sha256': checkpoint_sha256,
			'random_encoder_baseline': True,
			'pretrained_weights_loaded': False,
			'seed': 42,
			'checkpoint_kind': 'random_init',
			'reference_checkpoint': pretrained_checkpoint,
			'reference_checkpoint_sha256': 'a' * 64,
			'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		}
	)
	return {
		'model': model,
		'layout_id': layout_id,
		'data_size': data_size,
		'embedding': {
			'checkpoint_path': checkpoint_path,
			'checkpoint_sha256': checkpoint_sha256,
			'model_source': model_source,
			'common_metadata': {
				'survey_id': 'parihaka_full',
				'source_amplitude_path': '/data/parihaka.npy',
				'volume_shape_xyz': [100, 200, 300],
				'model_geometry': {'embedding_dim': 384, 'depth': 12},
				'patch_size': [8, 8, 8],
				'token_grid_shape': [13, 25, 38],
				'window_size': [16, 16, 16],
				'overlap': [8, 8, 8],
				'output_dtype': 'float16',
				'min_token_valid_fraction': 0.5,
				'normalization_stats_path': '/data/stats.json',
				'preprocessing': {'normalization': 'zscore'},
				'zero_mask': {'enabled': True},
				'precision': {'autocast': True},
				'pretraining_objective': {'name': 'mae'},
			},
		},
		'decoder_initial_state_sha256': 'c' * 64,
		'label_path': '/data/parihaka_labels.npy',
		'label_metadata_path': '/data/parihaka_labels_metadata.json',
		'prepared_label_identity': {
			'labels_sha256': 'd' * 64,
			'source_npz_path': '/data/parihaka_labels.npz',
			'source_key': 'labels',
			'shape': [100, 200, 300],
			'dtype': 'int8',
			'class_definition': {
				'positive_class_id': 5,
				'negative_class_ids': [1, 2, 3, 4, 6],
			},
		},
		'train_lines': {
			'inline': supervision['train_inline'],
			'crossline': supervision['train_crossline'],
		},
		'selection': selection,
		'validation': {
			'inline': supervision['validation_inline'],
			'crossline': supervision['validation_crossline'],
		},
		'test_definition': supervision['test_definition'],
		'geometry': {
			'embedding_shape': [13, 25, 38, 384],
			'volume_shape_xyz': [100, 200, 300],
			'token_grid_shape_xyz': [13, 25, 38],
			'patch_size_xyz': [8, 8, 8],
			'valid_tokens_sha256': 'e' * 64,
		},
		'class_weights': [0.55, 5.5],
		'decoder': {
			'spec': 'frozen_embedding_decoder_v1',
			'embedding_dim': 384,
			'class_count': 2,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
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
		'split_class_counts': supervision['split_class_counts'],
		'tile_counts': {'train': 10, 'validation': 4, 'test': 4},
	}


def _write_complete_results(config: ChannelSummaryConfig) -> None:
	for model in ('pretrained', 'random'):
		for layout_index, layout_id in enumerate(LAYOUT_IDS):
			for size in DATA_SIZE_PREFIX:
				root = _job_root(config, model, layout_id, size)
				root.mkdir(parents=True)
				value = 0.5 + layout_index / 100
				if model == 'pretrained':
					value += 0.1
				(root / 'metrics.json').write_text(
					json.dumps(
						{
							'model': model,
							'layout_id': layout_id,
							'data_size': size,
							'benchmark_identity': _benchmark_identity(
								model, layout_index, layout_id, size
							),
							'supervision': _supervision(layout_index, size),
							'class_weights': [0.55, 5.5],
							'train_channel_voxels': 100 * DATA_SIZE_PREFIX[size],
							'train_non_channel_voxels': (
								1000 * DATA_SIZE_PREFIX[size]
							),
							'test': {'channel_iou': value},
						}
					),
					encoding='utf-8',
				)


def _job_root(
	config: ChannelSummaryConfig, model: str, layout_id: str, size: str
) -> Path:
	return (
		config.runs_root
		/ f'model={model}'
		/ f'layout={layout_id}'
		/ f'size={size}'
	)


def _mutate_metrics(
	config: ChannelSummaryConfig,
	model: str,
	layout_id: str,
	size: str,
	mutator: object,
) -> None:
	path = _job_root(config, model, layout_id, size) / 'metrics.json'
	payload = json.loads(path.read_text(encoding='utf-8'))
	assert callable(mutator)
	mutator(payload)
	path.write_text(json.dumps(payload), encoding='utf-8')


def _set_nested(
	payload: dict[str, object], keys: tuple[str, ...], value: object
) -> None:
	target = payload
	for key in keys[:-1]:
		child = target[key]
		assert isinstance(child, dict)
		target = child
	target[keys[-1]] = value


def _set_embedding_checkpoint(
	payload: dict[str, object], field: str, value: str
) -> None:
	identity = payload['benchmark_identity']
	assert isinstance(identity, dict)
	embedding = identity['embedding']
	assert isinstance(embedding, dict)
	embedding[field] = value
	model_source = embedding['model_source']
	assert isinstance(model_source, dict)
	model_source[field] = value


def test_summary_fails_when_any_of_30_jobs_is_missing(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	with pytest.raises(FileNotFoundError, match='all 30 jobs'):
		inspect_channel_benchmark_results(config)


def test_complete_summary_writes_only_three_outputs(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	paths = summarize_channel_benchmark(config)
	assert {path.name for path in paths} == {
		'comparison.csv',
		'summary.json',
		'summary.md',
	}
	assert {path.name for path in config.output_dir.iterdir()} == {
		'comparison.csv',
		'summary.json',
		'summary.md',
	}
	payload = json.loads((config.output_dir / 'summary.json').read_text())
	assert payload['job_count'] == 30
	assert payload['by_size']['small']['paired_mean'] == pytest.approx(0.1)


def test_summary_requires_benchmark_identity(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	_mutate_metrics(
		config,
		'pretrained',
		'layout_000',
		'small',
		lambda payload: payload.pop('benchmark_identity'),
	)
	with pytest.raises(TypeError, match='benchmark_identity must be a mapping'):
		inspect_channel_benchmark_results(config)


@pytest.mark.parametrize(
	('keys', 'value', 'message'),
	[
		(('label_path',), '/data/different_labels.npy', 'label_path'),
		(
			('label_metadata_path',),
			'/data/different_labels_metadata.json',
			'label_metadata_path',
		),
		(
			('prepared_label_identity', 'labels_sha256'),
			'e' * 64,
			'prepared_label_identity',
		),
		(
			('embedding', 'common_metadata', 'source_amplitude_path'),
			'/data/different_amplitude.npy',
			'embedding common metadata',
		),
		(('geometry', 'token_grid_shape_xyz'), [99, 25, 38], 'geometry'),
		(('decoder', 'hidden_channels'), [64, 32], 'decoder'),
		(('training', 'epochs'), 51, 'training'),
		(('tiles', 'core_size_tokens'), [4, 4, 4], 'tiles'),
		(('decoder_initial_state_sha256',), 'd' * 64, 'decoder_initial_state'),
	],
)
def test_summary_rejects_global_benchmark_identity_drift(
	tmp_path: Path,
	keys: tuple[str, ...],
	value: object,
	message: str,
) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	for model in ('pretrained', 'random'):
		_mutate_metrics(
			config,
			model,
			'layout_001',
			'medium',
			lambda payload, keys=keys, value=value: _set_nested(
				payload['benchmark_identity'], keys, value
			),
		)
	with pytest.raises(ValueError, match=message):
		inspect_channel_benchmark_results(config)


@pytest.mark.parametrize(
	('field', 'value'),
	[
		(
			'checkpoint_path',
			'/other' + _EXPECTED_PRETRAINED_CHECKPOINT,
		),
		('checkpoint_sha256', 'd' * 64),
	],
)
def test_summary_rejects_checkpoint_drift_within_model_jobs(
	tmp_path: Path, field: str, value: str
) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	_mutate_metrics(
		config,
		'pretrained',
		'layout_004',
		'large',
		lambda payload: _set_embedding_checkpoint(payload, field, value),
	)
	with pytest.raises(ValueError, match='checkpoint does not match its other 15 jobs'):
		inspect_channel_benchmark_results(config)


def test_summary_rejects_same_checkpoint_sha_for_both_models(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	for layout_id in LAYOUT_IDS:
		for size in DATA_SIZE_PREFIX:
			_mutate_metrics(
				config,
				'random',
				layout_id,
				size,
				lambda payload: _set_embedding_checkpoint(
					payload, 'checkpoint_sha256', 'a' * 64
				),
			)
	with pytest.raises(ValueError, match='checkpoint SHA-256 must differ'):
		inspect_channel_benchmark_results(config)


@pytest.mark.parametrize(
	('model', 'field', 'value', 'message'),
	[
		('pretrained', 'role', 'random', 'model-source role'),
		('pretrained', 'model_tag', 'wrong-model', 'model_tag mismatch'),
		('random', 'role', 'pretrained', 'model-source role'),
		('random', 'random_encoder_baseline', False, 'random_encoder_baseline'),
		('random', 'pretrained_weights_loaded', True, 'pretrained_weights_loaded'),
		('random', 'seed', 43, 'seed mismatch'),
		('random', 'checkpoint_kind', 'epoch', 'checkpoint_kind mismatch'),
		('random', 'reference_model_tag', 'wrong-model', 'reference_model_tag'),
	],
)
def test_summary_rejects_invalid_model_source_role_identity(
	tmp_path: Path,
	model: str,
	field: str,
	value: object,
	message: str,
) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	_mutate_metrics(
		config,
		model,
		'layout_000',
		'small',
		lambda payload: payload['benchmark_identity']['embedding'][
			'model_source'
		].__setitem__(field, value),
	)
	with pytest.raises(ValueError, match=message):
		inspect_channel_benchmark_results(config)


def test_summary_rejects_random_reference_to_another_pretrained_source(
	tmp_path: Path,
) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	for layout_id in LAYOUT_IDS:
		for size in DATA_SIZE_PREFIX:
			_mutate_metrics(
				config,
				'random',
				layout_id,
				size,
				lambda payload: payload['benchmark_identity']['embedding'][
					'model_source'
				].__setitem__('reference_checkpoint', '/other/latest.pt'),
			)
	with pytest.raises(ValueError, match='does not match pretrained source'):
		inspect_channel_benchmark_results(config)


def test_summary_rejects_paired_tile_count_mismatch(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	_mutate_metrics(
		config,
		'random',
		'layout_002',
		'medium',
		lambda payload: payload['benchmark_identity']['tile_counts'].__setitem__(
			'train', 11
		),
	)
	with pytest.raises(ValueError, match='mismatch outside model-specific checkpoint'):
		inspect_channel_benchmark_results(config)


def test_summary_requires_complete_supervision(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	_mutate_metrics(
		config,
		'pretrained',
		'layout_000',
		'small',
		lambda payload: payload.pop('supervision'),
	)
	with pytest.raises(TypeError, match='supervision must be a mapping'):
		inspect_channel_benchmark_results(config)


def test_summary_rejects_explicit_test_line_schema(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)

	def mutate(payload: dict[str, object]) -> None:
		supervision = payload['supervision']
		assert isinstance(supervision, dict)
		supervision.pop('test_definition')
		supervision['test_inline'] = [102]
		supervision['test_crossline'] = [202]

	_mutate_metrics(config, 'pretrained', 'layout_000', 'small', mutate)
	with pytest.raises(ValueError, match='supervision must contain exactly'):
		inspect_channel_benchmark_results(config)


@pytest.mark.parametrize(
	'drift', ['definition', 'reserved', 'class_count', 'tile_count']
)
def test_summary_rejects_common_test_drift(tmp_path: Path, drift: str) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)

	def mutate(payload: dict[str, object]) -> None:
		supervision = payload['supervision']
		identity = payload['benchmark_identity']
		assert isinstance(supervision, dict)
		assert isinstance(identity, dict)
		if drift in {'definition', 'reserved'}:
			definition = dict(supervision['test_definition'])
			if drift == 'definition':
				definition['mode'] = 'wrong'
			else:
				definition['reserved_large_inline'] = [*range(10, 54), 99]
			supervision['test_definition'] = definition
			identity['test_definition'] = definition
		elif drift == 'class_count':
			counts = dict(supervision['split_class_counts'])
			counts['test'] = [3001, 300]
			supervision['split_class_counts'] = counts
			identity['split_class_counts'] = counts
		else:
			tiles = dict(identity['tile_counts'])
			tiles['test'] = 5
			identity['tile_counts'] = tiles

	for model in ('pretrained', 'random'):
		_mutate_metrics(config, model, 'layout_001', 'medium', mutate)
	with pytest.raises(ValueError, match=r'mode|does not match|tile counts'):
		inspect_channel_benchmark_results(config)


def test_summary_rejects_paired_supervision_mismatch(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	_mutate_metrics(
		config,
		'random',
		'layout_000',
		'small',
		lambda payload: payload['supervision'].__setitem__('train_inline', [999]),
	)
	with pytest.raises(ValueError, match='pretrained/random supervision mismatch'):
		inspect_channel_benchmark_results(config)


def test_summary_rejects_held_out_definition_drift(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	for model in ('pretrained', 'random'):
		_mutate_metrics(
			config,
			model,
			'layout_001',
			'medium',
			lambda payload: payload['supervision'].__setitem__(
				'validation_inline', [104, 105]
			),
		)
	with pytest.raises(ValueError, match='does not match all 30 jobs'):
		inspect_channel_benchmark_results(config)


def test_summary_rejects_non_nested_training_sections(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	for model in ('pretrained', 'random'):
		_mutate_metrics(
			config,
			model,
			'layout_002',
			'medium',
			lambda payload: payload['supervision'].__setitem__(
				'train_inline', [31, 30]
			),
		)
	with pytest.raises(ValueError, match='not nested'):
		inspect_channel_benchmark_results(config)


@pytest.mark.parametrize(
	('data_size', 'inline', 'crossline'),
	[
		('small', [10, 51, 52, 53], [20, 61, 62, 63]),
		('medium', [11, 10, 52, 53], [21, 20, 62, 63]),
		('large', [13, 12, 11, 10], [23, 22, 21, 20]),
	],
)
def test_summary_rejects_duplicate_training_sets_for_every_size(
	tmp_path: Path,
	data_size: str,
	inline: list[int],
	crossline: list[int],
) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	for model in ('pretrained', 'random'):
		for size, prefix in DATA_SIZE_PREFIX.items():
			_mutate_metrics(
				config,
				model,
				'layout_004',
				size,
				lambda payload, prefix=prefix: payload['supervision'].update(
					{
						'train_inline': inline[:prefix],
						'train_crossline': crossline[:prefix],
					}
				),
			)
	with pytest.raises(ValueError, match=rf'{data_size} training section sets'):
		inspect_channel_benchmark_results(config)


def test_summary_rejects_paired_class_weight_mismatch(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	_write_complete_results(config)
	_mutate_metrics(
		config,
		'random',
		'layout_003',
		'large',
		lambda payload: payload.__setitem__('class_weights', [1.0, 2.0]),
	)
	with pytest.raises(ValueError, match='pretrained/random class_weights mismatch'):
		inspect_channel_benchmark_results(config)
