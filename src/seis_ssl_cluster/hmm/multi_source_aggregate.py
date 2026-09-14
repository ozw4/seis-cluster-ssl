"""Combine complete survey summaries without averaging incompatible metrics."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.hmm.multi_source_receipts import (
	CANDIDATE_IDS,
	DATA_SIZES,
	METRICS,
	SOURCE_FAMILIES,
	SURVEYS,
	canonical_cells_sha256,
	control_receipt_for,
	file_sha256,
	load_control_receipts,
	load_matrix,
	load_reuse_receipt,
	object_sha256,
	read_json,
)

COMPARISON_FIELDS = (
	'survey',
	'task',
	'evaluation_split',
	'source_family',
	'candidate_id',
	'data_size',
	'reused_existing',
	'primary_metric',
	'primary_direction',
	'k6810_mean',
	'k6_mean',
	'primary_improvement',
	'wins',
	'ties',
	'losses',
	'secondary_metric',
	'k6810_secondary_mean',
	'k6_secondary_mean',
	'secondary_delta',
	'cell_count',
)


def comparison_row(
	survey: str, family: str, size: str, cells: list[dict[str, Any]]
) -> dict[str, Any]:
	"""Summarize five paired layouts; positive primary improvement is always better."""
	selected = [c for c in cells if c['data_size'] == size]
	if len(selected) != 5 or {c['layout_id'] for c in selected} != {
		f'layout_{i:03d}' for i in range(5)
	}:
		raise ValueError('size summary needs five distinct layouts')
	sign = -1 if survey == 'volve' else 1
	deltas = [sign * (c['primary'] - c['control_primary']) for c in selected]
	return {
		'survey': survey,
		**METRICS[survey],
		'source_family': family,
		'candidate_id': CANDIDATE_IDS[family],
		'data_size': size,
		'reused_existing': (survey, family) == ('f3', 'mae'),
		'k6810_mean': statistics.fmean(c['primary'] for c in selected),
		'k6_mean': statistics.fmean(c['control_primary'] for c in selected),
		'primary_improvement': statistics.fmean(deltas),
		'wins': sum(d > 0 for d in deltas),
		'ties': sum(d == 0 for d in deltas),
		'losses': sum(d < 0 for d in deltas),
		'k6810_secondary_mean': statistics.fmean(c['secondary'] for c in selected),
		'k6_secondary_mean': statistics.fmean(c['control_secondary'] for c in selected),
		'secondary_delta': statistics.fmean(
			c['secondary'] - c['control_secondary'] for c in selected
		),
		'cell_count': 5,
		'primary_improvement_sample_sd': statistics.stdev(deltas),
		'k6810_sample_sd': statistics.stdev(c['primary'] for c in selected),
		'k6_sample_sd': statistics.stdev(c['control_primary'] for c in selected),
	}


def arm_statistics(survey: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
	"""Keep all-15 descriptives separate from five independent layout clusters."""
	sign = -1 if survey == 'volve' else 1
	deltas = [sign * (c['primary'] - c['control_primary']) for c in cells]
	layout_means = [
		statistics.fmean(
			sign * (c['primary'] - c['control_primary'])
			for c in cells
			if c['layout_id'] == f'layout_{i:03d}'
		)
		for i in range(5)
	]
	return {
		'all_15': {
			'cell_count': 15,
			'k6810_mean': statistics.fmean(c['primary'] for c in cells),
			'k6_mean': statistics.fmean(c['control_primary'] for c in cells),
			'primary_improvement': statistics.fmean(deltas),
			'primary_improvement_sample_sd': statistics.stdev(deltas),
		},
		'layout_clustered': {
			'unit': 'layout_id',
			'cluster_count': 5,
			'primary_improvement': statistics.fmean(layout_means),
			'primary_improvement_sample_sd': statistics.stdev(layout_means),
			'layout_primary_improvements': layout_means,
		},
	}


def publish_summary(
	output_root: Path,
	payload: dict[str, Any],
	rows: list[dict[str, Any]],
	markdown: str,
) -> tuple[Path, Path, Path]:
	"""Publish exactly three files with an atomic rename and no replacement."""
	if output_root.exists() or output_root.is_symlink():
		raise FileExistsError(f'summary output already exists: {output_root}')
	stream = io.StringIO(newline='')
	writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS, extrasaction='ignore')
	writer.writeheader()
	writer.writerows(rows)
	contents = {
		'comparison.csv': stream.getvalue(),
		'summary.json': json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
		+ '\n',
		'summary.md': markdown,
	}
	output_root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(prefix=f'.{output_root.name}-', dir=output_root.parent)
	)
	try:
		for name, content in contents.items():
			(staging / name).write_text(content, encoding='utf-8')
		# Reserve the destination so racing publishers cannot replace one another.
		output_root.mkdir()
		try:
			staging.replace(output_root)
		except BaseException:
			output_root.rmdir()
			raise
	finally:
		if staging.exists():
			shutil.rmtree(staging)
	return tuple(output_root / name for name in contents)


def _same_number(actual: object, expected: float) -> bool:
	return (
		type(actual) in (int, float)
		and math.isfinite(actual)
		and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14)
	)


def _read_bound_summary(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
	path = Path(entry['path'])
	if 'reports' in path.resolve().parts:
		raise ValueError('reports cannot be aggregate inputs')
	before = file_sha256(path)
	if entry.get('sha256', before) != before:
		raise ValueError('source summary hash drift')
	payload = read_json(path)
	if payload.get('summary_sha256') != object_sha256(
		{k: v for k, v in payload.items() if k != 'summary_sha256'}
	):
		raise ValueError('source summary content digest drift')
	if file_sha256(path) != before:
		raise ValueError('source summary changed during read')
	return payload, {'path': str(path), 'sha256': before}


def _validate_summary_identity(
	payload: dict[str, Any], survey: str, bindings: dict[str, Any]
) -> None:
	if (
		payload.get('schema_version') != 1
		or payload.get('status') != 'complete'
		or payload.get('survey') != survey
	):
		raise ValueError('survey summary identity or completion drift')
	for key, expected in METRICS[survey].items():
		if payload.get(key) != expected:
			raise ValueError(f'survey metric contract drift: {key}')
	for name, expected in bindings.items():
		actual = payload[name]
		if (
			actual['sha256'] != expected['sha256']
			or Path(actual['path']).resolve() != Path(expected['path']).resolve()
		):
			raise ValueError('survey summary receipt binding drift')


def _validated_arm_cells(
	survey: str,
	family: str,
	cells: list[dict[str, Any]],
	controls: dict[str, Any],
	reuse: dict[str, Any],
) -> list[dict[str, Any]]:
	selected = [c for c in cells if c['source_family'] == family]
	canonical_cells_sha256(survey, selected)
	for cell in selected:
		if cell['candidate_id'] != CANDIDATE_IDS[family] or cell[
			'reused_existing'
		] is not ((survey, family) == ('f3', 'mae')):
			raise ValueError('cell candidate or reuse identity drift')
	control = control_receipt_for(controls, survey, family)
	control_cells = [
		{
			**c,
			'primary': c['control_primary'],
			'secondary': c['control_secondary'],
		}
		for c in selected
	]
	if (
		canonical_cells_sha256(survey, control_cells)
		!= control['canonical_cells_sha256']
	):
		raise ValueError('aggregate matching control cell drift')
	if (survey, family) == ('f3', 'mae') and canonical_cells_sha256(
		survey, selected
	) != reuse['canonical_cells_sha256']:
		raise ValueError('aggregate F3 reuse cell drift')
	return selected


def _validate_comparison_row(given: dict[str, Any], row: dict[str, Any]) -> None:
	for key, value in row.items():
		if (type(value) is float and not _same_number(given.get(key), value)) or (
			type(value) is not float and given.get(key) != value
		):
			raise ValueError(f'summary row disagrees with audited cells: {key}')


def inspect_aggregate(config: dict[str, Any]) -> dict[str, Any]:
	"""Validate all 135 cells and recompute every one of the 27 comparison rows."""
	matrix_path = Path(config['matrix'])
	control_path = Path(config['control_receipt'])
	reuse_path = Path(config['selection_receipt'])
	for path in (matrix_path, control_path, reuse_path):
		if 'reports' in path.resolve().parts:
			raise ValueError('reports cannot be pipeline inputs')
	load_matrix(matrix_path)
	controls = load_control_receipts(control_path)
	reuse = load_reuse_receipt(reuse_path)
	bindings = {
		'control_receipt': {
			'path': str(control_path),
			'sha256': file_sha256(control_path),
		},
		'selection_receipt': {
			'path': str(reuse_path),
			'sha256': file_sha256(reuse_path),
		},
	}
	if set(config['summaries']) != set(SURVEYS):
		raise ValueError('aggregate requires every survey summary')
	rows, all_cells, arms, evidence = [], [], [], {}
	for survey in SURVEYS:
		payload, source_identity = _read_bound_summary(config['summaries'][survey])
		_validate_summary_identity(payload, survey, bindings)
		cells = payload['cells']
		if len(cells) != 45 or len(payload['rows']) != 9:
			raise ValueError('survey must contain 45 cells and nine rows')
		row_ids = [(r['source_family'], r['data_size']) for r in payload['rows']]
		if len(set(row_ids)) != 9 or set(row_ids) != {
			(f, size) for f in SOURCE_FAMILIES for size in DATA_SIZES
		}:
			raise ValueError('duplicate or missing survey/source/size row')
		for family in SOURCE_FAMILIES:
			selected = _validated_arm_cells(survey, family, cells, controls, reuse)
			for size in DATA_SIZES:
				row = comparison_row(survey, family, size, selected)
				given = next(
					r
					for r in payload['rows']
					if (r['source_family'], r['data_size']) == (family, size)
				)
				_validate_comparison_row(given, row)
				rows.append(row)
			arms.append(
				{
					'survey': survey,
					'source_family': family,
					'candidate_id': CANDIDATE_IDS[family],
					'reused_existing': (survey, family) == ('f3', 'mae'),
					**arm_statistics(survey, selected),
				}
			)
		all_cells.extend({**c, 'survey': survey} for c in cells)
		evidence[survey] = source_identity
	return {
		'schema_version': 1,
		'status': 'complete',
		'completeness': {
			'arms': 9,
			'new_arms': 8,
			'total_cells': 135,
			'new_cells': 120,
			'reused_cells': 15,
			'comparison_rows': 27,
		},
		'matrix': {'path': str(matrix_path), 'sha256': file_sha256(matrix_path)},
		**bindings,
		'source_summaries': evidence,
		'survey_metrics': METRICS,
		'rows': rows,
		'cells': all_cells,
		'arms': arms,
	}


def summarize_aggregate(config: dict[str, Any]) -> tuple[Path, Path, Path]:
	"""Write a complete cross-survey summary without a scalar score or promotion."""
	output = Path(config['summary_root'])
	if output.exists() or output.is_symlink():
		raise FileExistsError('aggregate output already exists')
	payload = inspect_aggregate(config)
	lines = [
		'# K6810 multi-source evaluation',
		'',
		(
			'Each survey retains its own metric and evaluation split. '
			'Positive primary improvement favors K6810. The 15 cells are descriptive; '
			'overall variability uses five layout clusters.'
		),
		'',
	]
	for survey in SURVEYS:
		lines.extend(
			[
				f'## {survey}: {METRICS[survey]["evaluation_split"]}',
				'',
				(
					f'Primary: {METRICS[survey]["primary_metric"]}; '
					f'secondary: {METRICS[survey]["secondary_metric"]}.'
				),
				'',
				'| Source | Size | K6810 | K6 | Improvement | Wins/ties/losses |',
				'| --- | --- | ---: | ---: | ---: | --- |',
			]
		)
		lines.extend(
			f'| {row["source_family"]} | {row["data_size"]} | '
			f'{row["k6810_mean"]:.6f} | {row["k6_mean"]:.6f} | '
			f'{row["primary_improvement"]:.6f} | '
			f'{row["wins"]}/{row["ties"]}/{row["losses"]} |'
			for row in payload['rows']
			if row['survey'] == survey
		)
		lines.append('')
	return publish_summary(output, payload, payload['rows'], '\n'.join(lines))


def main() -> None:
	"""Validate or publish an aggregate of three complete artifact summaries."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()
	config = load_config(args.config)
	if args.dry_run:
		inspect_aggregate(config)
	else:
		summarize_aggregate(config)


if __name__ == '__main__':
	main()
