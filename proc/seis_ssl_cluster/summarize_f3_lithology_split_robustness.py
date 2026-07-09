"""Summarize paired F3 lithology split/index robustness probe metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

STAGE = 'summarize_f3_lithology_split_robustness'
SPLIT_INVENTORY_ARTIFACT_TYPE = 'f3_lithology_split_inventory_manifest'
SPLIT_DATASET_ARTIFACT_TYPE = 'f3_lithology_split_sweep_token_dataset_manifest'
SPLIT_PROBE_ARTIFACT_TYPE = 'f3_lithology_split_probe_run_manifest'
OVERALL_METRICS = (
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'mean_iou',
)
SUMMARY_FIELDNAMES = (
	'n_splits',
	'mean_delta_macro_f1',
	'median_delta_macro_f1',
	'std_delta_macro_f1',
	'min_delta_macro_f1',
	'max_delta_macro_f1',
	'win_rate_macro_f1',
	'mean_delta_mean_iou',
	'win_rate_mean_iou',
	'mean_delta_balanced_accuracy',
	'win_rate_balanced_accuracy',
)
SMALL_CLASS_SUPPORT_THRESHOLD = 5


@dataclass(frozen=True)
class SplitRobustnessSummaryResult:
	"""Paths written by the split/index robustness summarizer."""

	paired_metrics_csv: Path
	paired_deltas_csv: Path
	summary_csv: Path
	summary_markdown: Path
	paired_metric_count: int
	pair_count: int


@dataclass(frozen=True)
class _JoinedRun:
	split_id: str
	model_role: str
	model_tag: str
	train_token_count: int
	validation_token_count: int
	paired_identity_hash: str
	metrics_json: Path
	metrics: Mapping[str, object]
	validation_class_counts: Mapping[str, int]
	validation_slices: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class _SummaryPayload:
	model_tags: Mapping[str, str]
	paired_metric_fieldnames: tuple[str, ...]
	paired_delta_fieldnames: tuple[str, ...]
	paired_metric_rows: tuple[Mapping[str, object], ...]
	paired_delta_rows: tuple[Mapping[str, object], ...]
	summary_rows: tuple[Mapping[str, object], ...]


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for paired split/index robustness summaries."""
	parser = argparse.ArgumentParser(
		description='Summarize paired F3 lithology split/index robustness metrics.',
	)
	parser.add_argument(
		'--suite-root',
		type=Path,
		required=True,
		help=(
			'Directory containing split_dataset_manifest.json and '
			'split_probe_run_manifest.json; the dataset manifest must point to '
			'the split inventory manifest.'
		),
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate inputs and print planned report paths without writing files.',
	)
	return parser


def main() -> None:
	"""Summarize paired split/index probe metrics."""
	parser = build_parser()
	args = parser.parse_args()
	if args.dry_run:
		_print_dry_run(args.suite_root)
		print('execution: dry-run; split robustness summary skipped')
		return
	result = summarize_split_robustness(args.suite_root)
	print(f'f3_lithology_split_summary.paired_rows: {result.paired_metric_count}')
	print(f'f3_lithology_split_summary.pairs: {result.pair_count}')
	print(f'f3_lithology_split_summary.paired_metrics_csv: {result.paired_metrics_csv}')
	print(f'f3_lithology_split_summary.paired_deltas_csv: {result.paired_deltas_csv}')
	print(f'f3_lithology_split_summary.summary_csv: {result.summary_csv}')
	print(f'f3_lithology_split_summary.summary_markdown: {result.summary_markdown}')


def summarize_split_robustness(suite_root: Path) -> SplitRobustnessSummaryResult:
	"""Read split manifests and metrics, then write paired delta summary reports."""
	suite_root = suite_root.resolve()
	payload = _load_summary_payload(suite_root)
	reports_dir = suite_root / 'reports'
	reports_dir.mkdir(parents=True, exist_ok=True)

	_write_csv(
		reports_dir / 'split_paired_metrics.csv',
		payload.paired_metric_fieldnames,
		payload.paired_metric_rows,
	)
	_write_csv(
		reports_dir / 'split_paired_deltas.csv',
		payload.paired_delta_fieldnames,
		payload.paired_delta_rows,
	)
	_write_csv(
		reports_dir / 'split_summary.csv',
		SUMMARY_FIELDNAMES,
		payload.summary_rows,
	)
	(reports_dir / 'summary.md').write_text(
		_render_summary_markdown(payload),
		encoding='utf-8',
	)
	return SplitRobustnessSummaryResult(
		paired_metrics_csv=reports_dir / 'split_paired_metrics.csv',
		paired_deltas_csv=reports_dir / 'split_paired_deltas.csv',
		summary_csv=reports_dir / 'split_summary.csv',
		summary_markdown=reports_dir / 'summary.md',
		paired_metric_count=len(payload.paired_metric_rows),
		pair_count=len(payload.paired_delta_rows),
	)


def _print_dry_run(suite_root: Path) -> None:
	suite_root = suite_root.resolve()
	payload = _load_summary_payload(suite_root)
	dataset_manifest = _read_json(suite_root / 'split_dataset_manifest.json')
	reports_dir = suite_root / 'reports'
	print(f'stage: {STAGE}')
	print(f'suite_root: {suite_root}')
	print(
		'split_inventory_manifest: '
		f'{_split_inventory_manifest_path(dataset_manifest)}',
	)
	print(f'split_dataset_manifest: {suite_root / "split_dataset_manifest.json"}')
	print(f'split_probe_run_manifest: {suite_root / "split_probe_run_manifest.json"}')
	print(f'paired_metric_rows: {len(payload.paired_metric_rows)}')
	print(f'paired_delta_rows: {len(payload.paired_delta_rows)}')
	print(f'summary_rows: {len(payload.summary_rows)}')
	print(f'split_paired_metrics_csv: {reports_dir / "split_paired_metrics.csv"}')
	print(f'split_paired_deltas_csv: {reports_dir / "split_paired_deltas.csv"}')
	print(f'split_summary_csv: {reports_dir / "split_summary.csv"}')
	print(f'summary_markdown: {reports_dir / "summary.md"}')


def _load_summary_payload(suite_root: Path) -> _SummaryPayload:
	dataset_manifest = _read_json(suite_root / 'split_dataset_manifest.json')
	inventory_manifest = _read_json(_split_inventory_manifest_path(dataset_manifest))
	probe_manifest = _read_json(suite_root / 'split_probe_run_manifest.json')
	_validate_artifact_types(inventory_manifest, dataset_manifest, probe_manifest)
	inventory_rows = _rows(inventory_manifest, label='split_inventory_manifest')
	dataset_rows = _rows(dataset_manifest, label='split_dataset_manifest')
	probe_rows = _rows(probe_manifest, label='split_probe_run_manifest')

	joined = _joined_runs(inventory_rows, dataset_rows, probe_rows)
	_validate_complete_pairs(joined)
	class_columns = _class_columns(run.metrics for run in joined)
	pairs = _paired_runs(joined)
	paired_metric_rows = tuple(
		_paired_metric_row(run, class_columns=class_columns) for run in joined
	)
	paired_delta_rows = tuple(
		_delta_row(baseline, candidate, class_columns=class_columns)
		for baseline, candidate in pairs
	)
	return _SummaryPayload(
		model_tags=_model_tags(joined),
		paired_metric_fieldnames=(
			'split_id',
			'model_role',
			'model_tag',
			'train_token_count',
			'validation_token_count',
			'validation_class_counts',
			*OVERALL_METRICS,
			*class_columns,
			'metrics_json',
		),
		paired_delta_fieldnames=(
			'split_id',
			*(f'delta_{metric}' for metric in OVERALL_METRICS),
			*(f'delta_{column}' for column in class_columns),
			'baseline_metrics_json',
			'candidate_metrics_json',
			'validation_class_counts',
			'validation_slices',
		),
		paired_metric_rows=paired_metric_rows,
		paired_delta_rows=paired_delta_rows,
		summary_rows=_summary_rows(paired_delta_rows),
	)


def _split_inventory_manifest_path(dataset_manifest: Mapping[str, object]) -> Path:
	suite = dataset_manifest.get('suite')
	if not isinstance(suite, Mapping):
		msg = 'split_dataset_manifest suite must be a mapping'
		raise TypeError(msg)
	path = Path(_required_str(suite, 'split_inventory_manifest'))
	if not path.is_file():
		msg = f'required file does not exist: {path}'
		raise FileNotFoundError(msg)
	return path


def _validate_artifact_types(
	inventory_manifest: Mapping[str, object],
	dataset_manifest: Mapping[str, object],
	probe_manifest: Mapping[str, object],
) -> None:
	if inventory_manifest.get('artifact_type') != SPLIT_INVENTORY_ARTIFACT_TYPE:
		msg = 'split_inventory_manifest.json is not an F3 split inventory manifest'
		raise ValueError(msg)
	if dataset_manifest.get('artifact_type') != SPLIT_DATASET_ARTIFACT_TYPE:
		msg = 'split_dataset_manifest.json is not an F3 split dataset manifest'
		raise ValueError(msg)
	if probe_manifest.get('artifact_type') != SPLIT_PROBE_ARTIFACT_TYPE:
		msg = 'split_probe_run_manifest.json is not an F3 split probe manifest'
		raise ValueError(msg)


def _joined_runs(
	inventory_rows: Sequence[Mapping[str, object]],
	dataset_rows: Sequence[Mapping[str, object]],
	probe_rows: Sequence[Mapping[str, object]],
) -> tuple[_JoinedRun, ...]:
	split_metadata = _split_metadata_by_id(inventory_rows)
	dataset_by_key = _index_rows(dataset_rows, label='split_dataset_manifest')
	probe_by_key = _index_rows(probe_rows, label='split_probe_run_manifest')
	if set(dataset_by_key) != set(probe_by_key):
		missing_probe = sorted(set(dataset_by_key) - set(probe_by_key))
		missing_dataset = sorted(set(probe_by_key) - set(dataset_by_key))
		msg = (
			'split dataset/probe manifest row mismatch; '
			f'missing_probe={missing_probe!r}, missing_dataset={missing_dataset!r}'
		)
		raise ValueError(msg)
	_validate_inventory_split_coverage(
		inventory_split_ids=split_metadata,
		manifest_split_ids={key[0] for key in dataset_by_key},
	)

	joined = []
	for key in sorted(dataset_by_key, key=_run_key_sort_key):
		split_id = key[0]
		if split_id not in split_metadata:
			msg = f'split metadata missing for split_id={split_id!r}'
			raise ValueError(msg)
		dataset_row = dataset_by_key[key]
		probe_row = probe_by_key[key]
		_validate_manifest_row_agreement(dataset_row, probe_row)
		metrics_path = Path(_required_str(probe_row, 'metrics_json'))
		metrics = _read_json(metrics_path)
		_validate_metrics(metrics, metrics_path=metrics_path)
		joined.append(
			_JoinedRun(
				split_id=split_id,
				model_role=str(key[1]),
				model_tag=_required_str(probe_row, 'model_tag'),
				train_token_count=_required_int(probe_row, 'train_token_count'),
				validation_token_count=_required_int(
					probe_row,
					'validation_token_count',
				),
				paired_identity_hash=_required_str(
					dataset_row,
					'paired_identity_hash',
				),
				metrics_json=metrics_path,
				metrics=metrics,
				validation_class_counts=_validation_class_counts(dataset_row),
				validation_slices=split_metadata[split_id],
			),
		)
	return tuple(joined)


def _split_metadata_by_id(
	inventory_rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Mapping[str, object], ...]]:
	result = {}
	for index, row in enumerate(inventory_rows):
		split_id = _required_str(row, 'split_id')
		if split_id in result:
			msg = f'duplicate split inventory row for split_id={split_id!r}'
			raise ValueError(msg)
		path = Path(_required_str(row, 'split_metadata'))
		if not path.is_file():
			msg = f'split metadata missing for split_id={split_id!r}: {path}'
			raise FileNotFoundError(msg)
		metadata = _read_json(path)
		if metadata.get('split_id') not in (None, split_id):
			msg = f'split metadata split_id mismatch for row {index}: {path}'
			raise ValueError(msg)
		slices = metadata.get('validation_slices')
		if not isinstance(slices, Sequence) or isinstance(slices, str | bytes):
			msg = f'split metadata validation_slices must be a list: {path}'
			raise TypeError(msg)
		result[split_id] = tuple(
			_mapping_item(item, 'validation_slices') for item in slices
		)
	return result


def _validation_class_counts(row: Mapping[str, object]) -> Mapping[str, int]:
	path = Path(_required_str(row, 'class_counts_csv'))
	if not path.is_file():
		msg = f'class_counts_csv does not exist: {path}'
		raise FileNotFoundError(msg)
	counts: dict[str, int] = {}
	with path.open(newline='', encoding='utf-8') as handle:
		for csv_row in csv.DictReader(handle):
			if csv_row.get('split') != 'validation':
				continue
			class_id = csv_row.get('class_id')
			count = csv_row.get('count')
			if class_id is None or count is None:
				msg = f'class_counts_csv missing class_id/count columns: {path}'
				raise ValueError(msg)
			counts[str(class_id)] = _nonnegative_int_text(count, f'{path}:count')
	if not counts:
		msg = f'class_counts_csv contains no validation counts: {path}'
		raise ValueError(msg)
	return dict(sorted(counts.items(), key=lambda item: _class_id_sort_key(item[0])))


def _validate_inventory_split_coverage(
	*,
	inventory_split_ids: Iterable[str],
	manifest_split_ids: Iterable[str],
) -> None:
	inventory = set(inventory_split_ids)
	manifest = set(manifest_split_ids)
	if inventory == manifest:
		return
	missing = sorted(inventory - manifest)
	unexpected = sorted(manifest - inventory)
	msg = (
		'split inventory/manifest split mismatch; '
		f'missing_manifest_splits={missing!r}, '
		f'unexpected_manifest_splits={unexpected!r}'
	)
	raise ValueError(msg)


def _validate_complete_pairs(runs: Sequence[_JoinedRun]) -> None:
	by_split: dict[str, list[_JoinedRun]] = defaultdict(list)
	for run in runs:
		by_split[run.split_id].append(run)
	for split_id, split_runs in by_split.items():
		roles = sorted(run.model_role for run in split_runs)
		if roles != ['baseline', 'candidate']:
			msg = (
				'split summary requires exactly one baseline and one candidate '
				f'for every split; split_id={split_id!r}, roles={roles!r}'
			)
			raise ValueError(msg)
		hashes = {run.model_role: run.paired_identity_hash for run in split_runs}
		if hashes['baseline'] != hashes['candidate']:
			msg = (
				'paired_identity_hash mismatch; '
				f'split_id={split_id!r}, baseline={hashes["baseline"]}, '
				f'candidate={hashes["candidate"]}'
			)
			raise ValueError(msg)
		counts = {
			json.dumps(dict(run.validation_class_counts), sort_keys=True)
			for run in split_runs
		}
		if len(counts) != 1:
			msg = f'validation_class_counts mismatch for split_id={split_id!r}'
			raise ValueError(msg)


def _paired_runs(
	runs: Sequence[_JoinedRun],
) -> tuple[tuple[_JoinedRun, _JoinedRun], ...]:
	by_split: dict[str, dict[str, _JoinedRun]] = defaultdict(dict)
	for run in runs:
		by_split[run.split_id][run.model_role] = run
	pairs = []
	for split_id in sorted(by_split):
		split_runs = by_split[split_id]
		baseline = split_runs['baseline']
		candidate = split_runs['candidate']
		_validate_per_class_pair(baseline, candidate)
		pairs.append((baseline, candidate))
	return tuple(pairs)


def _paired_metric_row(
	run: _JoinedRun,
	*,
	class_columns: Sequence[str],
) -> dict[str, object]:
	row: dict[str, object] = {
		'split_id': run.split_id,
		'model_role': run.model_role,
		'model_tag': run.model_tag,
		'train_token_count': run.train_token_count,
		'validation_token_count': run.validation_token_count,
		'validation_class_counts': _json_cell(run.validation_class_counts),
		'metrics_json': str(run.metrics_json),
	}
	for metric in OVERALL_METRICS:
		row[metric] = _finite_float(run.metrics[metric], f'{run.metrics_json}:{metric}')
	per_class_f1 = _per_class_f1(run.metrics)
	for column in class_columns:
		class_id = column.removeprefix('class_').removesuffix('_f1')
		row[column] = (
			''
			if class_id not in per_class_f1
			else _finite_float(
				per_class_f1[class_id],
				f'{run.metrics_json}:per_class_f1.{class_id}',
			)
		)
	return row


def _delta_row(
	baseline: _JoinedRun,
	candidate: _JoinedRun,
	*,
	class_columns: Sequence[str],
) -> dict[str, object]:
	row: dict[str, object] = {
		'split_id': baseline.split_id,
		'baseline_metrics_json': str(baseline.metrics_json),
		'candidate_metrics_json': str(candidate.metrics_json),
		'validation_class_counts': _json_cell(baseline.validation_class_counts),
		'validation_slices': _json_cell(baseline.validation_slices),
	}
	for metric in OVERALL_METRICS:
		row[f'delta_{metric}'] = _metric_delta(baseline, candidate, metric)
	baseline_f1 = _per_class_f1(baseline.metrics)
	candidate_f1 = _per_class_f1(candidate.metrics)
	for column in class_columns:
		class_id = column.removeprefix('class_').removesuffix('_f1')
		row[f'delta_{column}'] = (
			''
			if class_id not in baseline_f1
			else _finite_float(
				candidate_f1[class_id],
				f'{candidate.metrics_json}:per_class_f1.{class_id}',
			)
			- _finite_float(
				baseline_f1[class_id],
				f'{baseline.metrics_json}:per_class_f1.{class_id}',
			)
		)
	return row


def _summary_rows(
	paired_delta_rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
	macro = _metric_values(paired_delta_rows, 'delta_macro_f1')
	mean_iou = _metric_values(paired_delta_rows, 'delta_mean_iou')
	balanced_accuracy = _metric_values(paired_delta_rows, 'delta_balanced_accuracy')
	return (
		{
			'n_splits': len(paired_delta_rows),
			'mean_delta_macro_f1': mean(macro),
			'median_delta_macro_f1': median(macro),
			'std_delta_macro_f1': stdev(macro) if len(macro) > 1 else 0.0,
			'min_delta_macro_f1': min(macro),
			'max_delta_macro_f1': max(macro),
			'win_rate_macro_f1': _win_rate(macro),
			'mean_delta_mean_iou': mean(mean_iou),
			'win_rate_mean_iou': _win_rate(mean_iou),
			'mean_delta_balanced_accuracy': mean(balanced_accuracy),
			'win_rate_balanced_accuracy': _win_rate(balanced_accuracy),
		},
	)


def _render_summary_markdown(payload: _SummaryPayload) -> str:
	summary = payload.summary_rows[0]
	lines = [
		'# F3 Lithology Split Robustness Summary',
		'',
		f'- baseline model: {payload.model_tags.get("baseline", "")}',
		f'- candidate model: {payload.model_tags.get("candidate", "")}',
		'- analysis: paired split/index robustness, not probe seed sweep',
		'',
		'## Split Deltas',
		'',
		(
			'| split_id | delta_macro_f1 | delta_mean_iou | '
			'delta_balanced_accuracy |'
		),
		'| --- | ---: | ---: | ---: |',
	]
	lines.extend(
		(
			'| '
			f'{row["split_id"]} | {_format_float(row["delta_macro_f1"])} | '
			f'{_format_float(row["delta_mean_iou"])} | '
			f'{_format_float(row["delta_balanced_accuracy"])} |'
		)
		for row in payload.paired_delta_rows
	)
	lines.extend(
		(
			'',
			'## Overall Win Rates',
			'',
			f'- macro_f1: {_format_float(summary["win_rate_macro_f1"])}',
			f'- mean_iou: {_format_float(summary["win_rate_mean_iou"])}',
			(
				'- balanced_accuracy: '
				f'{_format_float(summary["win_rate_balanced_accuracy"])}'
			),
			'',
			'## Validation Slices',
			'',
		),
	)
	lines.extend(
		f'- {row["split_id"]}: {row["validation_slices"]}'
		for row in payload.paired_delta_rows
	)
	lines.extend(('', '## Validation Class Counts', ''))
	lines.extend(
		f'- {row["split_id"]}: {row["validation_class_counts"]}'
		for row in payload.paired_delta_rows
	)
	lines.extend(
		(
			'',
			'## Warnings',
			'',
			*_warnings(payload.paired_delta_rows),
			'',
			'## Go/Hold/Stop Guidance',
			'',
			(
				'Go if strat-HMM wins macro_f1 and mean_iou on most splits and '
				'mean deltas are positive.'
			),
			'Hold if wins are split-dependent or balanced_accuracy degrades.',
			'Stop if improvements vanish outside the original split.',
			'',
			f'Current guidance: {_guidance(summary)}.',
			'',
		),
	)
	return '\n'.join(lines)


def _warnings(
	paired_delta_rows: Sequence[Mapping[str, object]],
) -> list[str]:
	warnings: list[str] = []
	for class_id in ('3', '5'):
		small = []
		for row in paired_delta_rows:
			counts = _json_mapping_cell(row['validation_class_counts'])
			count = int(counts.get(class_id, 0))
			if count < SMALL_CLASS_SUPPORT_THRESHOLD:
				small.append(f'{row["split_id"]}={count}')
		if small:
			warnings.append(
				(
					f'- class {class_id} validation support is small '
					f'(<{SMALL_CLASS_SUPPORT_THRESHOLD}): {", ".join(small)}'
				),
			)
		column = f'delta_class_{class_id}_f1'
		values = _optional_metric_values(paired_delta_rows, column)
		if values and all(value < 0 for value in values):
			warnings.append(
				f'- class {class_id} F1 deltas are consistently negative',
			)
	for column in ('delta_macro_f1', 'delta_mean_iou', 'delta_balanced_accuracy'):
		values = _metric_values(paired_delta_rows, column)
		if all(value < 0 for value in values):
			warnings.append(f'- {column} is consistently negative')
	return warnings or ['- none']


def _guidance(summary: Mapping[str, object]) -> str:
	macro_win = float(summary['win_rate_macro_f1'])
	mean_iou_win = float(summary['win_rate_mean_iou'])
	mean_macro = float(summary['mean_delta_macro_f1'])
	mean_iou = float(summary['mean_delta_mean_iou'])
	mean_balanced = float(summary['mean_delta_balanced_accuracy'])
	if mean_macro <= 0 and mean_iou <= 0:
		return 'Stop'
	if macro_win > 0.5 and mean_iou_win > 0.5 and mean_macro > 0 and mean_iou > 0:
		return 'Go'
	if mean_balanced < 0:
		return 'Hold'
	return 'Hold'


def _validate_manifest_row_agreement(
	dataset_row: Mapping[str, object],
	probe_row: Mapping[str, object],
) -> None:
	for key in ('split_id', 'model_role', 'model_tag', 'token_dataset_root'):
		if dataset_row.get(key) != probe_row.get(key):
			msg = (
				'split dataset/probe manifest value mismatch for '
				f'{key}: {dataset_row!r}'
			)
			raise ValueError(msg)
	if dataset_row.get('paired_identity_hash') != probe_row.get('paired_identity_hash'):
		msg = (
			'split dataset/probe paired_identity_hash mismatch; '
			f'row={dataset_row!r}'
		)
		raise ValueError(msg)


def _index_rows(
	rows: Sequence[Mapping[str, object]],
	*,
	label: str,
) -> dict[tuple[str, str], Mapping[str, object]]:
	by_key: dict[tuple[str, str], Mapping[str, object]] = {}
	for row in rows:
		key = _run_key(row)
		if key in by_key:
			msg = f'duplicate {label} row for key {key!r}'
			raise ValueError(msg)
		by_key[key] = row
	return by_key


def _run_key(row: Mapping[str, object]) -> tuple[str, str]:
	return (_required_str(row, 'split_id'), _required_str(row, 'model_role'))


def _run_key_sort_key(key: tuple[str, str]) -> tuple[str, str]:
	return (key[0], key[1])


def _metric_delta(
	baseline: _JoinedRun,
	candidate: _JoinedRun,
	metric: str,
) -> float:
	return _finite_float(
		candidate.metrics[metric],
		f'{candidate.metrics_json}:{metric}',
	) - _finite_float(baseline.metrics[metric], f'{baseline.metrics_json}:{metric}')


def _validate_metrics(metrics: Mapping[str, object], *, metrics_path: Path) -> None:
	_validate_finite_json_numbers(metrics, label=str(metrics_path))
	for key in OVERALL_METRICS:
		if key not in metrics:
			msg = f'metrics missing required key {key!r}: {metrics_path}'
			raise ValueError(msg)
		_finite_float(metrics[key], f'{metrics_path}:{key}')
	for class_id, value in _per_class_f1(metrics).items():
		_finite_float(value, f'{metrics_path}:per_class_f1.{class_id}')


def _validate_finite_json_numbers(value: object, *, label: str) -> None:
	if isinstance(value, bool):
		return
	if isinstance(value, int | float):
		_finite_float(value, label)
		return
	if isinstance(value, Mapping):
		for key, item in value.items():
			_validate_finite_json_numbers(item, label=f'{label}.{key}')
		return
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		for index, item in enumerate(value):
			_validate_finite_json_numbers(item, label=f'{label}[{index}]')


def _validate_per_class_pair(baseline: _JoinedRun, candidate: _JoinedRun) -> None:
	baseline_classes = set(_per_class_f1(baseline.metrics))
	candidate_classes = set(_per_class_f1(candidate.metrics))
	if baseline_classes != candidate_classes:
		msg = (
			'per_class_f1 class mismatch for paired split; '
			f'split_id={baseline.split_id!r}, '
			f'baseline={sorted(baseline_classes)!r}, '
			f'candidate={sorted(candidate_classes)!r}'
		)
		raise ValueError(msg)


def _class_columns(metrics_items: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
	columns = {
		f'class_{class_id}_f1'
		for metrics in metrics_items
		for class_id in _per_class_f1(metrics)
	}
	return tuple(sorted(columns, key=_class_metric_sort_key))


def _class_metric_sort_key(value: str) -> tuple[int, int | str]:
	class_id = value.removeprefix('class_').removesuffix('_f1')
	return _class_id_sort_key(class_id)


def _class_id_sort_key(value: str) -> tuple[int, int | str]:
	return (0, int(value)) if str(value).isdigit() else (1, str(value))


def _per_class_f1(metrics: Mapping[str, object]) -> Mapping[str, object]:
	value = metrics.get('per_class_f1')
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = 'metrics per_class_f1 must be a mapping'
		raise TypeError(msg)
	return {str(key): item for key, item in value.items()}


def _metric_values(
	rows: Sequence[Mapping[str, object]],
	key: str,
) -> tuple[float, ...]:
	return tuple(_finite_float(row[key], key) for row in rows)


def _optional_metric_values(
	rows: Sequence[Mapping[str, object]],
	key: str,
) -> tuple[float, ...]:
	return tuple(
		_finite_float(row[key], key)
		for row in rows
		if key in row and row[key] != ''
	)


def _win_rate(values: Sequence[float]) -> float:
	return sum(value > 0 for value in values) / len(values)


def _model_tags(runs: Sequence[_JoinedRun]) -> Mapping[str, str]:
	tags = {}
	for run in runs:
		tags.setdefault(run.model_role, run.model_tag)
	return tags


def _rows(
	payload: Mapping[str, object],
	*,
	label: str,
) -> tuple[Mapping[str, object], ...]:
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		msg = f'{label} rows must be a list'
		raise TypeError(msg)
	result = []
	for index, row in enumerate(rows):
		if not isinstance(row, Mapping):
			msg = f'{label} row {index} must be a mapping'
			raise TypeError(msg)
		result.append(row)
	if not result:
		msg = f'{label} contains no rows'
		raise ValueError(msg)
	return tuple(result)


def _read_json(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		msg = f'required file does not exist: {path}'
		raise FileNotFoundError(msg)
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		msg = f'JSON file must contain an object: {path}'
		raise TypeError(msg)
	return payload


def _write_csv(
	path: Path,
	fieldnames: Sequence[str],
	rows: Sequence[Mapping[str, object]],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _required_str(row: Mapping[str, object], key: str) -> str:
	value = row.get(key)
	if not isinstance(value, str) or not value:
		msg = f'required string missing or empty: {key}'
		raise ValueError(msg)
	return value


def _required_int(row: Mapping[str, object], key: str) -> int:
	value = row.get(key)
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'required integer missing: {key}'
		raise TypeError(msg)
	return value


def _nonnegative_int_text(value: str, label: str) -> int:
	try:
		result = int(value)
	except ValueError as exc:
		msg = f'{label} must be an integer; got {value!r}'
		raise ValueError(msg) from exc
	if result < 0:
		msg = f'{label} must be nonnegative; got {value!r}'
		raise ValueError(msg)
	return result


def _finite_float(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'metric must be numeric: {label}'
		raise TypeError(msg)
	result = float(value)
	if not math.isfinite(result):
		msg = f'metric must be finite: {label}'
		raise ValueError(msg)
	return result


def _mapping_item(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		msg = f'{label} item must be a mapping'
		raise TypeError(msg)
	return value


def _json_cell(value: object) -> str:
	return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _json_mapping_cell(value: object) -> Mapping[str, object]:
	payload = json.loads(str(value))
	if not isinstance(payload, Mapping):
		msg = f'expected JSON object cell; got {value!r}'
		raise TypeError(msg)
	return payload


def _format_float(value: object) -> str:
	return f'{float(value):.6f}'


if __name__ == '__main__':
	main()
