"""Summarize the five-layout final Channel test for one frozen model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZE = 'medium'
EVALUATION_MODE = 'validation_and_test'


def _mapping(value: object, label: str) -> Mapping[str, Any]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _metrics_path(runs_root: Path, model: str, layout: str) -> Path:
	return (
		runs_root
		/ f'model={model}'
		/ f'layout={layout}'
		/ f'size={DATA_SIZE}'
		/ 'metrics.json'
	)


def _validate_metrics_set(runs_root: Path, model: str) -> None:
	expected = {_metrics_path(runs_root, model, layout) for layout in LAYOUTS}
	actual = set(runs_root.glob('model=*/layout=*/size=*/metrics.json'))
	if actual != expected:
		missing = sorted(str(path.relative_to(runs_root)) for path in expected - actual)
		unexpected = sorted(
			str(path.relative_to(runs_root)) for path in actual - expected
		)
		raise ValueError(
			'final runs must contain exactly the five predefined jobs; '
			f'missing={missing}; unexpected={unexpected}'
		)


def read_metrics(runs_root: Path, model: str, layout: str) -> float:
	"""Read one final metrics file and return its test Channel IoU."""
	path = _metrics_path(runs_root, model, layout)
	payload = _mapping(
		json.loads(path.read_text(encoding='utf-8')),
		f'{path} metrics payload',
	)
	if payload.get('model') != model:
		raise ValueError(f'{path}: model identity mismatch')
	if payload.get('layout_id') != layout:
		raise ValueError(f'{path}: layout identity mismatch')
	if payload.get('data_size') != DATA_SIZE:
		raise ValueError(f'{path}: data_size must be {DATA_SIZE}')
	if payload.get('evaluation_mode') != EVALUATION_MODE:
		raise ValueError(f'{path}: evaluation_mode must be {EVALUATION_MODE}')
	test = _mapping(payload.get('test'), f'{path} test metrics')
	channel_iou = test.get('channel_iou')
	if (
		not isinstance(channel_iou, int | float)
		or isinstance(channel_iou, bool)
		or not math.isfinite(float(channel_iou))
	):
		raise ValueError(f'{path}: test Channel IoU must be finite')
	return float(channel_iou)


def write_report(result: Mapping[str, Any], report_root: Path) -> None:
	"""Write per-layout CSV plus machine- and human-readable summaries."""
	report_root.mkdir(parents=True, exist_ok=True)
	layout_values = _mapping(result['layouts'], 'layout results')
	with (report_root / 'final_test_layouts.csv').open(
		'w',
		encoding='utf-8',
		newline='',
	) as stream:
		writer = csv.DictWriter(
			stream,
			fieldnames=('model', 'layout_id', 'data_size', 'test_channel_iou'),
		)
		writer.writeheader()
		for layout in LAYOUTS:
			writer.writerow(
				{
					'model': result['model'],
					'layout_id': layout,
					'data_size': result['data_size'],
					'test_channel_iou': layout_values[layout],
				}
			)
	(report_root / 'final_test_summary.json').write_text(
		json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	lines = [
		'# Final Channel test across predefined layouts',
		'',
		f'- Frozen model: {result["model"]}',
		f'- Metric: {result["metric"]}',
		f'- Data size: {result["data_size"]}',
		f'- Layout count: {result["layout_count"]}',
		'',
		'| layout | test Channel IoU |',
		'|---|---:|',
	]
	lines.extend(
		f'| {layout} | {float(layout_values[layout]):.6f} |'
		for layout in LAYOUTS
	)
	lines.extend(
		[
			'',
			f'- Mean: {float(result["mean"]):.6f}',
			f'- Median: {float(result["median"]):.6f}',
			(
				'- Sample standard deviation: '
				f'{float(result["sample_standard_deviation"]):.6f}'
			),
			'',
			(
				'All five layouts are reported as repeated evaluations. '
				'They are not ranked.'
			),
		]
	)
	(report_root / 'final_test_summary.md').write_text(
		'\n'.join(lines) + '\n',
		encoding='utf-8',
	)


def summarize_final_test(
	runs_root: Path,
	model: str,
	report_root: Path,
) -> dict[str, Any]:
	"""Validate and summarize exactly five final test jobs for one model."""
	if not model:
		raise ValueError('model must be non-empty')
	_validate_metrics_set(runs_root, model)
	layout_values = {
		layout: read_metrics(runs_root, model, layout) for layout in LAYOUTS
	}
	values = list(layout_values.values())
	result = {
		'model': model,
		'metric': 'test.channel_iou',
		'data_size': DATA_SIZE,
		'layout_count': len(layout_values),
		'layouts': layout_values,
		'mean': statistics.mean(values),
		'median': statistics.median(values),
		'sample_standard_deviation': statistics.stdev(values),
	}
	write_report(result, report_root)
	return result


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--runs-root', type=Path, required=True)
	parser.add_argument('--model', required=True)
	parser.add_argument('--report-root', type=Path, required=True)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Run the fixed five-layout final test summary."""
	args = _parser().parse_args(argv)
	result = summarize_final_test(args.runs_root, args.model, args.report_root)
	print(f'metrics read: {result["layout_count"]}')
	print(f'mean test Channel IoU: {result["mean"]:.6f}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
