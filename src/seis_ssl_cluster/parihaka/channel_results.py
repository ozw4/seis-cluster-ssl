# ruff: noqa: CPY001
"""Paired results for the Parihaka Channel benchmark."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
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
				_validate_supervision(payload, path)
				_class_weights(payload, path)
				rows[(model, layout_id, data_size)] = payload
	if missing:
		raise FileNotFoundError(
			f'Parihaka Channel summary requires all 30 jobs; missing {len(missing)}: '
			+ ', '.join(str(path) for path in missing)
		)
	if len(rows) != 30:
		raise ValueError(f'expected 30 unique benchmark conditions; got {len(rows)}')
	_validate_supervision_parity(rows, config.runs_root)
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


def _validate_supervision(payload: Mapping[str, object], path: Path) -> None:
	supervision = payload.get('supervision')
	if not isinstance(supervision, Mapping):
		raise TypeError(f'{path} supervision must be a mapping')
	expected_keys = {
		'axis_mapping',
		'train_inline',
		'train_crossline',
		'validation_inline',
		'validation_crossline',
		'test_inline',
		'test_crossline',
		'split_class_counts',
	}
	if set(supervision) != expected_keys:
		raise ValueError(
			f'{path} supervision must contain exactly {sorted(expected_keys)!r}'
		)
	if supervision.get('axis_mapping') != {'inline': 'x', 'crossline': 'y'}:
		raise ValueError(
			f"{path} supervision.axis_mapping must be "
			"{'inline': 'x', 'crossline': 'y'}"
		)
	for key in (
		'train_inline',
		'train_crossline',
		'validation_inline',
		'validation_crossline',
		'test_inline',
		'test_crossline',
	):
		_indices(supervision.get(key), f'{path} supervision.{key}')
	counts = supervision.get('split_class_counts')
	if not isinstance(counts, Mapping):
		raise TypeError(f'{path} supervision.split_class_counts must be a mapping')
	if set(counts) != {'train', 'validation', 'test'}:
		raise ValueError(
			f'{path} supervision.split_class_counts must contain exactly '
			"'train', 'validation', and 'test'"
		)
	for split in ('train', 'validation', 'test'):
		_class_counts(
			counts.get(split), f'{path} supervision.split_class_counts.{split}'
		)


def _validate_supervision_parity(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	_validate_common_held_out(rows, runs_root)
	_validate_pairs(rows, runs_root)
	_validate_nested_training(rows)
	_validate_unique_layout_training(rows)


def _validate_common_held_out(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	first_key = (MODELS[0], LAYOUT_IDS[0], next(iter(DATA_SIZE_PREFIX)))
	common_supervision = _supervision(rows[first_key])
	common_held_out = _held_out_identity(common_supervision)
	for key, payload in rows.items():
		if _held_out_identity(_supervision(payload)) != common_held_out:
			raise ValueError(
				f'{_metrics_path(runs_root, *key)} validation/test supervision '
				'does not match all 30 jobs'
			)


def _validate_pairs(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]], runs_root: Path
) -> None:
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZE_PREFIX:
			pretrained = rows[('pretrained', layout_id, data_size)]
			random = rows[('random', layout_id, data_size)]
			if _supervision(pretrained) != _supervision(random):
				raise ValueError(
					f'{layout_id}/{data_size} pretrained/random supervision mismatch'
				)
			pretrained_weights = _class_weights(
				pretrained,
				_metrics_path(runs_root, 'pretrained', layout_id, data_size),
			)
			random_weights = _class_weights(
				random,
				_metrics_path(runs_root, 'random', layout_id, data_size),
			)
			if pretrained_weights != random_weights:
				raise ValueError(
					f'{layout_id}/{data_size} pretrained/random class_weights mismatch'
				)


def _validate_nested_training(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
	for model in MODELS:
		for layout_id in LAYOUT_IDS:
			by_size = {
				data_size: _supervision(rows[(model, layout_id, data_size)])
				for data_size in DATA_SIZE_PREFIX
			}
			for orientation in ('inline', 'crossline'):
				key = f'train_{orientation}'
				large: tuple[int, ...] | None = None
				for data_size, prefix in DATA_SIZE_PREFIX.items():
					current = _indices(
						by_size[data_size].get(key),
						f'{model}/{layout_id}/{data_size} supervision.{key}',
					)
					if len(current) != prefix:
						raise ValueError(
							f'{model}/{layout_id}/{data_size} {key} must contain '
							f'exactly {prefix} indices'
						)
					if data_size == 'large':
						large = current
				if large is None:
					raise RuntimeError('large Channel supervision is unavailable')
				for data_size, prefix in DATA_SIZE_PREFIX.items():
					current = _indices(
						by_size[data_size].get(key),
						f'{model}/{layout_id}/{data_size} supervision.{key}',
					)
					if current != large[:prefix]:
						raise ValueError(
							f'{model}/{layout_id} {key} is not nested in '
							'small/medium/large prefix order'
						)


def _validate_unique_layout_training(
	rows: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
	for model in MODELS:
		for data_size in DATA_SIZE_PREFIX:
			seen: dict[tuple[frozenset[int], frozenset[int]], str] = {}
			for layout_id in LAYOUT_IDS:
				supervision = _supervision(rows[(model, layout_id, data_size)])
				identity = (
					frozenset(
						_indices(
							supervision.get('train_inline'),
							f'{model}/{layout_id}/{data_size} '
							'supervision.train_inline',
						)
					),
					frozenset(
						_indices(
							supervision.get('train_crossline'),
							f'{model}/{layout_id}/{data_size} '
							'supervision.train_crossline',
						)
					),
				)
				if duplicate := seen.get(identity):
					raise ValueError(
						f'{model}/{data_size} training section sets must be unique '
						f'across layouts; {duplicate} and {layout_id} select the '
						'same sections'
					)
				seen[identity] = layout_id


def _supervision(payload: Mapping[str, object]) -> Mapping[str, object]:
	value = payload.get('supervision')
	if not isinstance(value, Mapping):
		raise TypeError('supervision must be a mapping')
	return value


def _held_out_identity(supervision: Mapping[str, object]) -> tuple[object, ...]:
	counts = supervision.get('split_class_counts')
	if not isinstance(counts, Mapping):
		raise TypeError('supervision.split_class_counts must be a mapping')
	return (
		supervision.get('axis_mapping'),
		supervision.get('validation_inline'),
		supervision.get('validation_crossline'),
		supervision.get('test_inline'),
		supervision.get('test_crossline'),
		counts.get('validation'),
		counts.get('test'),
	)


def _indices(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, list) or not value:
		raise TypeError(f'{label} must be a non-empty list')
	if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
		raise TypeError(f'{label} must contain integers')
	items = tuple(value)
	if len(set(items)) != len(items):
		raise ValueError(f'{label} must not contain duplicates')
	return items


def _class_counts(value: object, label: str) -> tuple[int, int]:
	if (
		not isinstance(value, list)
		or len(value) != 2
		or any(
			not isinstance(item, int) or isinstance(item, bool) or item < 0
			for item in value
		)
	):
		raise TypeError(f'{label} must contain two non-negative integers')
	return value[0], value[1]


def _class_weights(payload: Mapping[str, object], path: Path) -> tuple[float, float]:
	value = payload.get('class_weights')
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 2
		or any(
			not isinstance(item, int | float)
			or isinstance(item, bool)
			or not math.isfinite(float(item))
			for item in value
		)
	):
		raise TypeError(f'{path} class_weights must contain two finite numbers')
	return float(value[0]), float(value[1])


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
