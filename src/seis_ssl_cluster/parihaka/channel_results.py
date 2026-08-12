# ruff: noqa: CPY001
"""Paired results for the Parihaka Channel benchmark."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS

MODELS = ('pretrained', 'random')
OUTPUT_NAMES = ('comparison.csv', 'summary.json', 'summary.md')


@dataclass(frozen=True)
class ChannelSummaryConfig:
	"""Resolved benchmark result paths."""

	runs_root: Path
	output_dir: Path


def channel_summary_config_from_mapping(
	config: Mapping[str, object],
) -> ChannelSummaryConfig:
	"""Resolve the direct summary configuration."""
	inputs = _mapping(config, 'inputs')
	outputs = _mapping(config, 'outputs')
	return ChannelSummaryConfig(
		runs_root=_absolute_path(inputs, 'runs_root', 'inputs'),
		output_dir=_absolute_path(outputs, 'output_dir', 'outputs'),
	)


def inspect_channel_benchmark_results(
	config: ChannelSummaryConfig,
) -> dict[tuple[str, str, str], Mapping[str, object]]:
	"""Require and validate all 30 metrics files."""
	rows: dict[tuple[str, str, str], Mapping[str, object]] = {}
	missing: list[Path] = []
	for model in MODELS:
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				path = _metrics_path(config.runs_root, model, layout_id, data_size)
				if not path.is_file():
					missing.append(path)
					continue
				payload = _read_json(path)
				_validate_identity(payload, model, layout_id, data_size, path)
				_metric(payload, 'test', 'channel_iou', path)
				rows[(model, layout_id, data_size)] = payload
	if missing:
		raise FileNotFoundError(
			f'Parihaka Channel summary requires all 30 jobs; missing {len(missing)}: '
			+ ', '.join(str(path) for path in missing)
		)
	if len(rows) != 30:
		raise ValueError(f'expected 30 unique benchmark conditions; got {len(rows)}')
	return rows


def summarize_channel_benchmark(
	config: ChannelSummaryConfig,
) -> tuple[Path, Path, Path]:
	"""Write the paired Channel-IoU comparison and size aggregates."""
	jobs = inspect_channel_benchmark_results(config)
	if config.output_dir.exists() and any(
		(config.output_dir / name).exists() for name in OUTPUT_NAMES
	):
		raise FileExistsError(
			f'channel summary outputs already exist: {config.output_dir}'
		)
	comparison: list[dict[str, object]] = []
	for data_size in DATA_SIZE_PREFIX:
		for layout_id in LAYOUT_IDS:
			pretrained = _metric(
				jobs[('pretrained', layout_id, data_size)],
				'test',
				'channel_iou',
				_metrics_path(config.runs_root, 'pretrained', layout_id, data_size),
			)
			random = _metric(
				jobs[('random', layout_id, data_size)],
				'test',
				'channel_iou',
				_metrics_path(config.runs_root, 'random', layout_id, data_size),
			)
			comparison.append(
				{
					'data_size': data_size,
					'layout_id': layout_id,
					'pretrained_channel_iou': pretrained,
					'random_channel_iou': random,
					'delta_channel_iou': pretrained - random,
				}
			)
	aggregates: dict[str, object] = {}
	for data_size in DATA_SIZE_PREFIX:
		selected = [row for row in comparison if row['data_size'] == data_size]
		deltas = [float(row['delta_channel_iou']) for row in selected]
		aggregates[data_size] = {
			'paired_mean': statistics.fmean(deltas),
			'paired_median': statistics.median(deltas),
			'sample_standard_deviation': statistics.stdev(deltas),
			'pretrained_wins': sum(delta > 0 for delta in deltas),
			'ties': sum(delta == 0 for delta in deltas),
			'pretrained_losses': sum(delta < 0 for delta in deltas),
			'layout_deltas': {
				str(row['layout_id']): row['delta_channel_iou'] for row in selected
			},
		}
	payload = {
		'schema_version': 1,
		'primary_metric': 'test.channel_iou',
		'job_count': len(jobs),
		'comparison': comparison,
		'by_size': aggregates,
	}
	config.output_dir.mkdir(parents=True, exist_ok=True)
	comparison_path = config.output_dir / OUTPUT_NAMES[0]
	with comparison_path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=tuple(comparison[0]))
		writer.writeheader()
		writer.writerows(comparison)
	json_path = config.output_dir / OUTPUT_NAMES[1]
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
	markdown_path = config.output_dir / OUTPUT_NAMES[2]
	markdown_path.write_text(_markdown(aggregates, comparison), encoding='utf-8')
	return comparison_path, json_path, markdown_path


def _markdown(
	aggregates: Mapping[str, object], comparison: list[dict[str, object]]
) -> str:
	lines = [
		'# Parihaka Channel benchmark',
		'',
		'Primary metric: test Channel IoU. Deltas are pretrained minus random.',
		'',
		'| size | paired mean | paired median | sample std | wins/ties/losses |',
		'|---|---:|---:|---:|---:|',
	]
	for data_size in DATA_SIZE_PREFIX:
		row = _mapping(aggregates, data_size)
		lines.append(
			f'| {data_size} | {float(row["paired_mean"]):.6f} | '
			f'{float(row["paired_median"]):.6f} | '
			f'{float(row["sample_standard_deviation"]):.6f} | '
			f'{row["pretrained_wins"]}/{row["ties"]}/{row["pretrained_losses"]} |'
		)
	lines.extend(
		[
			'',
			'| size | layout | pretrained | random | delta |',
			'|---|---|---:|---:|---:|',
		]
	)
	lines.extend(
		f'| {row["data_size"]} | {row["layout_id"]} | '
		f'{float(row["pretrained_channel_iou"]):.6f} | '
		f'{float(row["random_channel_iou"]):.6f} | '
		f'{float(row["delta_channel_iou"]):.6f} |'
		for row in comparison
	)
	return '\n'.join(lines) + '\n'


def _metrics_path(root: Path, model: str, layout: str, size: str) -> Path:
	return (
		root / f'model={model}' / f'layout={layout}' / f'size={size}' / 'metrics.json'
	)


def _validate_identity(
	payload: Mapping[str, object], model: str, layout: str, size: str, path: Path
) -> None:
	expected = {'model': model, 'layout_id': layout, 'data_size': size}
	for key, value in expected.items():
		if payload.get(key) != value:
			raise ValueError(f'{path} has incorrect {key}: {payload.get(key)!r}')


def _metric(payload: Mapping[str, object], split: str, key: str, path: Path) -> float:
	value = payload.get(split)
	if not isinstance(value, Mapping):
		raise TypeError(f'{path} {split} must be a mapping')
	metric = value.get(key)
	if not isinstance(metric, int | float) or isinstance(metric, bool):
		raise TypeError(f'{path} {split}.{key} must be numeric')
	return float(metric)


def _read_json(path: Path) -> Mapping[str, object]:
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, Mapping):
		raise TypeError(f'metrics must contain an object: {path}')
	return value


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _absolute_path(value: Mapping[str, object], key: str, prefix: str) -> Path:
	item = value.get(key)
	if not isinstance(item, str) or not item:
		raise ValueError(f'{prefix}.{key} must be a non-empty path')
	path = Path(item)
	if not path.is_absolute():
		raise ValueError(f'{prefix}.{key} must be absolute')
	return path


__all__ = [
	'ChannelSummaryConfig',
	'channel_summary_config_from_mapping',
	'inspect_channel_benchmark_results',
	'summarize_channel_benchmark',
]
