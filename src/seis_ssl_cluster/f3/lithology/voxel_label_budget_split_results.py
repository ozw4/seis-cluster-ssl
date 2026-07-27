"""Split-level paired deltas and the fixed #282 confirmatory decision."""
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.results import (
	PublishItem,
	PublishManifest,
	publish_selected_results,
)

SPLITS = tuple(f'split_{index:03d}' for index in range(6))
BUDGETS = ('cap25', 'cap50')
MODELS = ('mae', 'm1_current_k6', 'mh_nocons')
PRIMARY = ('macro_f1', 'mean_iou')
MONITORED = (
	'class_3_f1', 'class_3_iou', 'class_3_boundary_recall_tolerance_2',
	'class_3_boundary_recall_tolerance_4', 'class_5_f1', 'class_5_iou',
	'class_5_boundary_recall_tolerance_2', 'class_5_boundary_recall_tolerance_4',
)
REQUIRED_METRICS = (*PRIMARY, *(
	'balanced_accuracy', 'accuracy', 'weighted_f1',
	'boundary_region_macro_f1_r2', 'boundary_region_mean_iou_r2',
	'boundary_region_macro_f1_r4', 'boundary_region_mean_iou_r4',
	'boundary_f1_t2', 'boundary_f1_t4', 'boundary_position_mae',
), *MONITORED)
LOWER_IS_BETTER = frozenset({'boundary_position_mae'})
COMPARISONS = (('mh_nocons', 'm1_current_k6'), ('mh_nocons', 'mae'), ('m1_current_k6', 'mae'))
OUTPUT_NAMES = (
	'low_label_split_paired_metrics.csv',
	'low_label_split_paired_deltas.csv',
	'low_label_split_aggregates.csv',
	'low_label_split_monitored_class_summary.csv',
	'low_label_split_decisions.json',
	'low_label_split_results_summary.json',
	'low_label_split_results_summary.md',
	'low_label_split_handoff.md',
)
PUBLISH_MANIFEST_NAME = 'publish_manifest.json'


def aggregate_low_label_split_results(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
	"""Create complete paired deltas, aggregates, or a publishable BLOCKED result."""
	try:
		by_key = _validate_rows(rows)
	except (TypeError, ValueError) as error:
		return [], [], _blocked(str(error))
	deltas = []
	for split_id in SPLITS:
		for budget_id in BUDGETS:
			for left, right in COMPARISONS:
				for metric in REQUIRED_METRICS:
					left_value, right_value = by_key[(split_id, budget_id, left)][metric], by_key[(split_id, budget_id, right)][metric]
					delta = left_value - right_value
					if metric in LOWER_IS_BETTER:
						delta = -delta
					deltas.append({'split_id': split_id, 'budget_id': budget_id, 'comparison': f'{left}_minus_{right}', 'metric': metric, 'delta': delta})
	aggregates = [_aggregate(group) for group in _groups(deltas)]
	return deltas, aggregates, _decision(aggregates)


def write_low_label_split_summary(rows: Sequence[Mapping[str, object]], output_root: Path) -> Mapping[str, Path]:
	"""Write lightweight summary artifacts, including a blocked decision when invalid."""
	deltas, aggregates, decision = aggregate_low_label_split_results(rows)
	output_root.mkdir(parents=True, exist_ok=True)
	paths = {
		'paired_metrics': output_root / 'low_label_split_paired_metrics.csv',
		'paired_deltas': output_root / 'low_label_split_paired_deltas.csv',
		'aggregates': output_root / 'low_label_split_aggregates.csv',
		'monitored_classes': output_root / 'low_label_split_monitored_class_summary.csv',
		'decisions': output_root / 'low_label_split_decisions.json',
		'summary': output_root / 'low_label_split_results_summary.json',
		'markdown': output_root / 'low_label_split_results_summary.md',
		'handoff': output_root / 'low_label_split_handoff.md',
	}
	_write_csv(paths['paired_metrics'], rows, tuple(rows[0]) if rows else ('split_id', 'budget_id', 'model_role'))
	_write_csv(paths['paired_deltas'], deltas, ('split_id', 'budget_id', 'comparison', 'metric', 'delta'))
	_write_csv(paths['aggregates'], aggregates, ('budget_id', 'comparison', 'metric', 'mean', 'median', 'sample_sd', 'min', 'max', 'wins', 'losses', 'ties', 'worst_split', 'worst_delta'))
	_write_csv(paths['monitored_classes'], [row for row in aggregates if str(row['metric']) in MONITORED], ('budget_id', 'comparison', 'metric', 'mean', 'median', 'sample_sd', 'min', 'max', 'wins', 'losses', 'ties', 'worst_split', 'worst_delta'))
	payload = {'status': decision['status'], 'decision': decision, 'job_count': len(rows), 'aggregate_count': len(aggregates)}
	for key in ('decisions', 'summary'):
		paths[key].write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
	paths['markdown'].write_text(f"# M4 six-split low-label summary\n\nStatus: `{decision['status']}`.\n", encoding='utf-8')
	paths['handoff'].write_text(f"# M4 six-split handoff\n\nDecision status: `{decision['status']}`.\n", encoding='utf-8')
	return paths


def publish_low_label_split_summary(
	config: object, paths: Mapping[str, Path]
) -> PublishManifest:
	"""Copy the exact lightweight six-split summary into the results tree."""
	sources = {path.name: path for path in paths.values()}
	if set(sources) != set(OUTPUT_NAMES):
		raise ValueError('six-split summary publish sources are incomplete or unexpected')
	publish_dir = _publish_dir(config)
	_validate_existing_publish_tree(publish_dir)
	manifest = publish_selected_results(
		items=[PublishItem(sources[name], Path(name)) for name in OUTPUT_NAMES],
		output_dir=publish_dir,
		max_file_size_bytes=10 * 1024 * 1024,
		overwrite=True,
	)
	_validate_published_tree(publish_dir, manifest)
	return manifest


def _publish_dir(config: object) -> Path:
	return config.results_root / 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_six_split_v1'


def _publish_target_names() -> set[str]:
	return {*OUTPUT_NAMES, PUBLISH_MANIFEST_NAME}


def _validate_existing_publish_tree(publish_dir: Path) -> None:
	if not publish_dir.exists():
		return
	actual = _published_relative_names(publish_dir)
	if actual and actual != _publish_target_names():
		expected = _publish_target_names()
		raise FileExistsError(
		'six-split publish root has an unexpected file set; '
		f'missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}'
	)


def _validate_published_tree(  # noqa: C901
	publish_dir: Path, manifest: PublishManifest
) -> None:
	expected = _publish_target_names()
	actual = _published_relative_names(publish_dir)
	if actual != expected:
		raise ValueError(
		'six-split published file inventory mismatch; '
		f'missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}'
	)
	if manifest.manifest_path != publish_dir.resolve() / PUBLISH_MANIFEST_NAME:
		raise ValueError('six-split publish manifest path mismatch')
	payload = _mapping(json.loads(manifest.manifest_path.read_text(encoding='utf-8')))
	recorded = payload.get('items')
	if not isinstance(recorded, list) or len(recorded) != len(OUTPUT_NAMES):
		raise ValueError('six-split publish manifest item count mismatch')
	recorded_by_target: dict[str, Mapping[str, object]] = {}
	for value in recorded:
		item = _mapping(value)
		target = item.get('target')
		if not isinstance(target, str) or target in recorded_by_target:
			raise ValueError('six-split publish manifest targets are invalid')
		recorded_by_target[target] = item
	if set(recorded_by_target) != set(OUTPUT_NAMES):
		raise ValueError('six-split publish manifest target set mismatch')
	for item in manifest.items:
		target = item.target.resolve()
		if not target.is_file() or not item.source.is_file():
			raise FileNotFoundError(f'six-split publish source or target is missing: {target}')
		if item.size_bytes != target.stat().st_size:
			raise ValueError(f'six-split published target size mismatch: {target}')
		if item.sha256 != file_sha256(item.source) or item.sha256 != file_sha256(target):
			raise ValueError(f'six-split publish source/target SHA-256 mismatch: {target}')
		target_name = target.relative_to(publish_dir.resolve()).as_posix()
		record = recorded_by_target.get(target_name)
		if record is None or record.get('source') != str(item.source) or record.get('size_bytes') != item.size_bytes or record.get('sha256') != item.sha256:
			raise ValueError(f'six-split publish manifest SHA-256 mismatch: {target}')


def _published_relative_names(publish_dir: Path) -> set[str]:
	if not publish_dir.is_dir():
		raise NotADirectoryError(f'six-split publish root is not a directory: {publish_dir}')
	names = set()
	for path in publish_dir.rglob('*'):
		if path.is_symlink() or not path.is_file():
			raise ValueError(f'six-split publish root has a non-file entry: {path}')
		names.add(path.relative_to(publish_dir).as_posix())
	return names


def _validate_rows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, str], dict[str, float]]:
	if len(rows) != len(SPLITS) * len(BUDGETS) * len(MODELS):
		raise ValueError('coverage failure: expected exactly 36 completed job metrics')
	indexed: dict[tuple[str, str, str], dict[str, float]] = {}
	for row in rows:
		key = tuple(str(row.get(name, '')) for name in ('split_id', 'budget_id', 'model_role'))
		if key in indexed:
			raise ValueError(f'coverage failure: duplicate job metric row {key!r}')
		if key[0] not in SPLITS or key[1] not in BUDGETS or key[2] not in MODELS:
			raise ValueError(f'coverage failure: unknown job metric row {key!r}')
		values: dict[str, float] = {}
		for metric in REQUIRED_METRICS:
			try:
				value = float(row[metric])
			except (KeyError, TypeError, ValueError) as error:
				raise ValueError(f'metric contract failure: {key!r} lacks finite {metric}') from error
			if not math.isfinite(value):
				raise ValueError(f'metric contract failure: {key!r} has nonfinite {metric}')
			values[metric] = value
		indexed[key] = values
	expected = {(split_id, budget_id, model) for split_id in SPLITS for budget_id in BUDGETS for model in MODELS}
	if set(indexed) != expected:
		raise ValueError('coverage failure: split/budget/model matrix is incomplete')
	return indexed


def _groups(rows: Sequence[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
	grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
	for row in rows:
		grouped.setdefault((str(row['budget_id']), str(row['comparison']), str(row['metric'])), []).append(row)
	return list(grouped.values())


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
	values = [float(row['delta']) for row in rows]
	return {'budget_id': rows[0]['budget_id'], 'comparison': rows[0]['comparison'], 'metric': rows[0]['metric'], 'mean': statistics.mean(values), 'median': statistics.median(values), 'sample_sd': statistics.stdev(values), 'min': min(values), 'max': max(values), 'wins': sum(value > 0 for value in values), 'losses': sum(value < 0 for value in values), 'ties': sum(value == 0 for value in values), 'worst_split': rows[values.index(min(values))]['split_id'], 'worst_delta': min(values)}


def _decision(aggregates: Sequence[Mapping[str, object]]) -> dict[str, object]:
	lookup = {(str(row['budget_id']), str(row['comparison']), str(row['metric'])): row for row in aggregates}
	def row(budget: str, comparison: str, metric: str) -> Mapping[str, object]:
		return lookup[(budget, comparison, metric)]
	def positive(comparison: str) -> bool:
		return all(float(row(budget, comparison, metric)['mean']) > 0 and float(row(budget, comparison, metric)['median']) > 0 and int(row(budget, comparison, metric)['wins']) >= 4 for budget in BUDGETS for metric in PRIMARY)
	negative = all(float(row(budget, 'mh_nocons_minus_m1_current_k6', metric)['mean']) < 0 and int(row(budget, 'mh_nocons_minus_m1_current_k6', metric)['wins']) <= 2 for budget in BUDGETS for metric in PRIMARY)
	degraded = any(all(float(row(budget, 'mh_nocons_minus_m1_current_k6', metric)['mean']) <= -0.05 for budget in BUDGETS) for metric in MONITORED)
	status = 'M4_MH_SPLIT_CONFIRMED' if positive('mh_nocons_minus_m1_current_k6') and positive('mh_nocons_minus_mae') and not degraded else 'M4_MH_SPLIT_NEGATIVE' if negative else 'M4_MH_SPLIT_HOLD'
	return {'status': status, 'systematic_major_degradation': degraded}


def _blocked(reason: str) -> dict[str, object]:
	return {'status': 'M4_MH_SPLIT_BLOCKED', 'blocked_reason': reason, 'systematic_major_degradation': None}


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('expected a JSON object')
	return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
	with path.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


__all__ = [
	'OUTPUT_NAMES',
	'PUBLISH_MANIFEST_NAME',
	'aggregate_low_label_split_results',
	'publish_low_label_split_summary',
	'write_low_label_split_summary',
]
