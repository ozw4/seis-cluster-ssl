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


def test_summary_fails_when_any_of_30_jobs_is_missing(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	with pytest.raises(FileNotFoundError, match='all 30 jobs'):
		inspect_channel_benchmark_results(config)


def test_complete_summary_writes_only_three_outputs(tmp_path: Path) -> None:
	config = ChannelSummaryConfig(tmp_path / 'runs', tmp_path / 'summary')
	for model in ('pretrained', 'random'):
		for layout_index, layout_id in enumerate(LAYOUT_IDS):
			for size in DATA_SIZE_PREFIX:
				root = (
					config.runs_root
					/ f'model={model}'
					/ f'layout={layout_id}'
					/ f'size={size}'
				)
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
							'test': {'channel_iou': value},
						}
					),
					encoding='utf-8',
				)
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
