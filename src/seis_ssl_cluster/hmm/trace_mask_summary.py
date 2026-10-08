"""Audit and publish v3 versus the exact canonical v02 decoder cells."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.hmm.multi_source_receipts import (
	CELL_IDENTITIES,
	DATA_SIZES,
	METRICS,
	canonical_cells_sha256,
	file_sha256,
	object_sha256,
	read_json,
)
from seis_ssl_cluster.hmm.multi_source_survey import (
	_audit_cell,
	_check_directories,
	_metrics,
	_paired_identity,
	_paths,
)
from seis_ssl_cluster.hmm.trace_mask_definition import (
	definition_path,
	read_yaml,
	validate_trace_mask_definition,
)
from seis_ssl_cluster.hmm.trace_mask_driver import validate_canonical_evidence
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source

PUBLISHED_FILES = ('comparison.csv', 'summary.json', 'summary.md')
COMPARISON_FIELDS = (
	'survey',
	'task',
	'evaluation_split',
	'source_family',
	'candidate_id',
	'baseline_id',
	'data_size',
	'primary_metric',
	'primary_direction',
	'v3_mean',
	'v02_mean',
	'primary_improvement',
	'primary_improvement_sample_sd',
	'wins',
	'ties',
	'losses',
	'secondary_metric',
	'v3_secondary_mean',
	'v02_secondary_mean',
	'secondary_delta',
	'cell_count',
)


def comparison_row(
	survey: str, family: str, size: str, cells: list[dict[str, Any]]
) -> dict[str, Any]:
	"""Summarize five paired layouts within one survey and one label budget."""
	selected = [cell for cell in cells if cell['data_size'] == size]
	if len(selected) != 5 or {cell['layout_id'] for cell in selected} != {
		f'layout_{i:03d}' for i in range(5)
	}:
		raise ValueError('v3 size summary requires five distinct layouts')
	sign = -1 if survey == 'volve' else 1
	deltas = [sign * (cell['primary'] - cell['control_primary']) for cell in selected]
	return {
		'survey': survey,
		**METRICS[survey],
		'source_family': family,
		'candidate_id': selected[0]['candidate_id'],
		'baseline_id': selected[0]['baseline_id'],
		'data_size': size,
		'v3_mean': statistics.fmean(cell['primary'] for cell in selected),
		'v02_mean': statistics.fmean(cell['control_primary'] for cell in selected),
		'primary_improvement': statistics.fmean(deltas),
		'primary_improvement_sample_sd': statistics.stdev(deltas),
		'wins': sum(delta > 0 for delta in deltas),
		'ties': sum(delta == 0 for delta in deltas),
		'losses': sum(delta < 0 for delta in deltas),
		'v3_secondary_mean': statistics.fmean(cell['secondary'] for cell in selected),
		'v02_secondary_mean': statistics.fmean(
			cell['control_secondary'] for cell in selected
		),
		'secondary_delta': statistics.fmean(
			cell['secondary'] - cell['control_secondary'] for cell in selected
		),
		'cell_count': 5,
	}


def _lineage(
	experiment: Path, cell: dict[str, Any], *, baseline: bool
) -> dict[str, Any]:
	field = 'baseline_train_config' if baseline else 'train_full_config'
	lineage = audit_multi_head_source(definition_path(cell[field], experiment))
	if (
		baseline
		and lineage['checkpoint']['sha256'] != cell['baseline_checkpoint_sha256']
	):
		raise ValueError('canonical v02 baseline checkpoint hash drift')
	if lineage['target_manifest']['sha256'] != cell['target_manifest_sha256']:
		raise ValueError('v3/v02 target manifest binding drift')
	return lineage


def _audited_arm(
	experiment: Path,
	survey: str,
	cell: dict[str, Any],
	*,
	frozen_control_cells: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	candidate = load_config(definition_path(cell['downstream_config'], experiment))
	baseline = load_config(
		definition_path(cell['baseline_downstream_config'], experiment)
	)
	candidate_id = cell['candidate_id']
	baseline_training = read_yaml(
		definition_path(cell['baseline_train_config'], experiment)
	)
	baseline_id = baseline_training['identity']['model_tag']
	candidate_root, candidate_checkpoint = _paths(candidate, survey, candidate_id)
	baseline_root, baseline_checkpoint = _paths(baseline, survey, baseline_id)
	candidate_lineage = _lineage(experiment, cell, baseline=False)
	baseline_lineage = _lineage(experiment, cell, baseline=True)
	for lineage, checkpoint, model in (
		(candidate_lineage, candidate_checkpoint, candidate_id),
		(baseline_lineage, baseline_checkpoint, baseline_id),
	):
		if (
			lineage['model_tag'] != model
			or lineage['head_ks'] != [6, 8, 10]
			or lineage['epoch'] != 25
			or lineage['checkpoint']
			!= {'path': str(checkpoint), 'sha256': file_sha256(checkpoint)}
		):
			raise ValueError('paired v3/v02 completed source lineage drift')
	for root, model in ((candidate_root, candidate_id), (baseline_root, baseline_id)):
		_check_directories(root, model)
	cells, baseline_cells, audits = [], [], []
	for layout, size in sorted(CELL_IDENTITIES):
		name = 'evaluation/metrics.json' if survey == 'f3' else 'metrics.json'
		candidate_metrics = (
			candidate_root
			/ f'model={candidate_id}'
			/ f'layout={layout}'
			/ f'size={size}'
			/ name
		)
		baseline_metrics = (
			baseline_root
			/ f'model={baseline_id}'
			/ f'layout={layout}'
			/ f'size={size}'
			/ name
		)
		before = file_sha256(candidate_metrics), file_sha256(baseline_metrics)
		candidate_audit = _audit_cell(survey, candidate, candidate_id, layout, size)
		baseline_audit = _audit_cell(survey, baseline, baseline_id, layout, size)
		_paired_identity(survey, candidate_audit, baseline_audit)
		primary, secondary, digest = _metrics(candidate_metrics, survey)
		control_primary, control_secondary, control_digest = _metrics(
			baseline_metrics, survey
		)
		if before != (digest, control_digest):
			raise ValueError('paired metrics changed during completion audit')
		cells.append(
			{
				'source_family': cell['source'],
				'candidate_id': candidate_id,
				'baseline_id': baseline_id,
				'layout_id': layout,
				'data_size': size,
				'primary': primary,
				'secondary': secondary,
				'control_primary': control_primary,
				'control_secondary': control_secondary,
				'metrics_path': str(candidate_metrics),
				'metrics_sha256': digest,
				'control_metrics_path': str(baseline_metrics),
				'control_metrics_sha256': control_digest,
			}
		)
		baseline_cells.append(
			{
				'layout_id': layout,
				'data_size': size,
				'primary': control_primary,
				'secondary': control_secondary,
			}
		)
		audits.append(
			{
				'layout_id': layout,
				'data_size': size,
				'candidate': candidate_audit,
				'baseline': baseline_audit,
			}
		)
	canonical_cells_sha256(survey, cells)
	baseline_digest = canonical_cells_sha256(survey, baseline_cells)
	if frozen_control_cells is None:
		selection = read_yaml(
			definition_path(
				read_yaml(experiment / 'execution.yaml')['canonical_selection']['path'],
				experiment,
			)
		)
		canonical = read_json(
			Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
			/ selection['evidence']['artifact_summary']
		)
		frozen_control_cells = canonical['cells']
	frozen = [
		item
		for item in frozen_control_cells
		if item.get('survey', survey) == survey
		and item['source_family'] == cell['source']
	]
	if (
		len(frozen) != 15
		or any(item['candidate_id'] != baseline_id for item in frozen)
		or canonical_cells_sha256(survey, frozen) != baseline_digest
	):
		raise ValueError('live v02 cells differ from selected canonical evidence')
	if (
		'baseline_cells_sha256' in cell
		and baseline_digest != cell['baseline_cells_sha256']
	):
		raise ValueError('frozen canonical v02 decoder cell digest drift')
	return cells, {
		'source': candidate_lineage,
		'baseline_source': baseline_lineage,
		'baseline_cells_sha256': baseline_digest,
		'paired_cells': audits,
	}


def inspect_summary(experiment: Path) -> dict[str, Any]:
	"""Require all 45 paired cells with source and shared decoder provenance."""
	definition = validate_trace_mask_definition(experiment)
	validate_canonical_evidence(experiment, definition)
	survey = definition['survey']
	cells, rows, arms, evidence = [], [], [], {}
	for cell in definition['cells']:
		selected, audit = _audited_arm(experiment, survey, cell)
		cells.extend(selected)
		rows.extend(
			comparison_row(survey, cell['source'], size, selected)
			for size in DATA_SIZES
		)
		sign = -1 if survey == 'volve' else 1
		layout_means = [
			statistics.fmean(
				sign * (item['primary'] - item['control_primary'])
				for item in selected
				if item['layout_id'] == f'layout_{i:03d}'
			)
			for i in range(5)
		]
		arms.append(
			{
				'source_family': cell['source'],
				'candidate_id': cell['candidate_id'],
				'all_15': {
					'cell_count': 15,
					'v3_mean': statistics.fmean(item['primary'] for item in selected),
					'v02_mean': statistics.fmean(
						item['control_primary'] for item in selected
					),
				},
				'layout_clustered': {
					'unit': 'layout_id',
					'cluster_count': 5,
					'primary_improvement': statistics.fmean(layout_means),
					'primary_improvement_sample_sd': statistics.stdev(layout_means),
					'layout_primary_improvements': layout_means,
				},
			}
		)
		evidence[cell['source']] = audit
	if len(cells) != 45:
		raise ValueError('v3 summary requires exactly 45 candidate cells per survey')
	payload = {
		'schema_version': 1,
		'status': 'complete',
		'survey': survey,
		**METRICS[survey],
		'comparison': 'hmm_v3_trace_mask_vs_canonical_v02_cons010',
		'head_ks': [6, 8, 10],
		'distillation_weight': 0.2,
		'consistency_weight': 0.1,
		'column_fraction': 0.1,
		'execution': {
			'path': str(experiment / 'execution.yaml'),
			'sha256': file_sha256(experiment / 'execution.yaml'),
		},
		'canonical_selection': definition['canonical_selection'],
		'cells': cells,
		'rows': rows,
		'arms': arms,
		'evidence': evidence,
	}
	payload['summary_sha256'] = object_sha256(payload)
	return payload


def _publication_contents(payload: dict[str, Any]) -> dict[str, str]:
	stream = io.StringIO(newline='')
	writer = csv.DictWriter(
		stream, fieldnames=COMPARISON_FIELDS, extrasaction='ignore', lineterminator='\n'
	)
	writer.writeheader()
	writer.writerows(payload['rows'])
	lines = [
		f'# {payload["survey"]}: trace-mask v3 versus canonical v02',
		'',
		(
			f'Evaluation: {payload["evaluation_split"]}; '
			f'primary: {payload["primary_metric"]}.'
		),
		'',
		'K=[6,8,10], distillation=0.2, consistency=0.1, trace-column mask=10%.',
		'',
		(
			'Positive primary improvement favors v3. '
			'Overall uncertainty uses five layout clusters. '
			'Results remain separate by survey.'
		),
		'',
		'| Source | Size | v3 | v02 | Improvement | SD | Wins/ties/losses |',
		'| --- | --- | ---: | ---: | ---: | ---: | --- |',
	]
	lines.extend(
		f'| {row["source_family"]} | {row["data_size"]} | '
		f'{row["v3_mean"]:.6f} | {row["v02_mean"]:.6f} | '
		f'{row["primary_improvement"]:.6f} | '
		f'{row["primary_improvement_sample_sd"]:.6f} | '
		f'{row["wins"]}/{row["ties"]}/{row["losses"]} |'
		for row in payload['rows']
	)
	return {
		'comparison.csv': stream.getvalue(),
		'summary.json': json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
		+ '\n',
		'summary.md': '\n'.join(lines) + '\n',
	}


def _publish(output: Path, payload: dict[str, Any]) -> tuple[Path, ...]:
	if output.exists() or output.is_symlink():
		raise FileExistsError(f'v3 summary output already exists: {output}')
	contents = _publication_contents(payload)
	output.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
	try:
		for name, content in contents.items():
			(staging / name).write_text(content, encoding='utf-8')
		output.mkdir()
		try:
			staging.replace(output)
		except BaseException:
			output.rmdir()
			raise
	finally:
		if staging.exists():
			shutil.rmtree(staging)
	return tuple(output / name for name in PUBLISHED_FILES)


def summarize(experiment: Path) -> tuple[Path, ...]:
	"""Publish exactly the three review files after complete paired auditing."""
	definition = validate_trace_mask_definition(experiment)
	output = Path(os.path.expandvars(definition['summary_root']))
	if 'reports' in output.resolve().parts:
		raise ValueError('v3 summary output must be an artifact')
	payload = inspect_summary(experiment)
	if output.exists():
		if {path.name for path in output.iterdir()} != set(PUBLISHED_FILES) or any(
			(output / name).read_text(encoding='utf-8') != content
			for name, content in _publication_contents(payload).items()
		):
			raise ValueError('existing v3 summary publication or provenance drift')
		print('execution: reused audited complete v3 paired summary')
		return tuple(output / name for name in PUBLISHED_FILES)
	return _publish(output, payload)


def main() -> None:
	"""Inspect or publish one complete survey without combining its metrics."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--experiment', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()
	if args.dry_run:
		print(json.dumps({'status': inspect_summary(args.experiment)['status']}))
	else:
		summarize(args.experiment)


if __name__ == '__main__':
	main()
