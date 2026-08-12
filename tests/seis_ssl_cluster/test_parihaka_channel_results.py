# ruff: noqa: CPY001, TC003

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_results import (
	ChannelSummaryConfig,
	inspect_channel_benchmark_results,
	summarize_channel_benchmark,
)


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
		'test_inline': [102, 103],
		'test_crossline': [202, 203],
		'split_class_counts': {
			'train': [1000 * prefix, 100 * prefix],
			'validation': [2000, 200],
			'test': [3000, 300],
		},
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
							'supervision': _supervision(layout_index, size),
							'class_weights': [0.55, 5.5],
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
