"""Controlled F3 mask-token learning-rate sweep against completed masked v3."""

from __future__ import annotations

# ruff: noqa: PLC0415
import argparse
import copy
import csv
import fcntl
import io
import json
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config.pretraining import CENTER_TRACE_MASK_LR_TAGS
from seis_ssl_cluster.hmm.multi_source_receipts import (
	DATA_SIZES,
	file_sha256,
	object_sha256,
	read_json,
)
from seis_ssl_cluster.hmm.trace_mask_definition import (
	CANDIDATE_IDS,
	definition_path,
	read_yaml,
	replace_values,
	validate_trace_mask_definition,
)
from seis_ssl_cluster.hmm.trace_mask_driver import (
	LAYOUTS,
	STAGES,
	run_command,
	validate_canonical_evidence,
)
from seis_ssl_cluster.hmm.trace_mask_summary import _audited_arm, comparison_row

NAMESPACE = 'hmm_v3_mask_token_lr_sweep_v1'
EXPERIMENT = '133_hmm_v3_mask_token_lr_sweep_v1'
CONTROL = '132_hmm_v3_k6810_trace_mask_multi_source_v1'
RATES = {'3em5': 3.0e-5, '1em4': 1.0e-4}
PUBLISHED_FILES = ('comparison.csv', 'summary.json', 'summary.md')


def _owned(value: str, *, candidate: str | None = None) -> None:
	parts = Path(value).parts
	if 'reports' in parts or NAMESPACE not in parts or 'f3' not in parts:
		raise ValueError('mask-LR outputs must use the owned F3 artifact namespace')
	if candidate is not None and candidate not in parts:
		raise ValueError('mask-LR output candidate drift')


def validate_definition(experiment: Path) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
	"""Require six independently initialized arms changing only mask-token LR."""
	definition = read_yaml(experiment / 'execution.yaml')
	if experiment.name != EXPERIMENT or definition.get('experiment_id') != EXPERIMENT:
		raise ValueError('mask-LR experiment identity drift')
	if (
		definition_path(definition['experiment_root'], experiment).resolve()
		!= experiment.resolve()
	):
		raise ValueError('mask-LR experiment root drift')
	baseline_root = experiment.parent / CONTROL
	control = validate_trace_mask_definition(baseline_root)
	if (
		file_sha256(baseline_root / 'execution.yaml')
		!= definition['control_definition_sha256']
	):
		raise ValueError('mask-LR control definition hash drift')
	expected_ids = set(CENTER_TRACE_MASK_LR_TAGS)
	cells = definition['cells']
	if len(cells) != 6 or {cell['candidate_id'] for cell in cells} != expected_ids:
		raise ValueError('mask-LR matrix requires three sources times two rates')
	_owned(definition['output_root'])
	_owned(definition['summary_root'])
	for cell in cells:
		family, candidate = cell['source'], cell['candidate_id']
		baseline_id = CANDIDATE_IDS[family]
		if candidate not in {f'{baseline_id}_masklr{suffix}' for suffix in RATES}:
			raise ValueError('mask-LR source identity drift')
		if cell['mask_token_lr'] != CENTER_TRACE_MASK_LR_TAGS[candidate]:
			raise ValueError('mask-LR candidate rate drift')
		base_cell = next(item for item in control['cells'] if item['source'] == family)
		if cell['target_manifest_sha256'] != base_cell['target_manifest_sha256']:
			raise ValueError('mask-LR immutable target binding drift')
		baseline = {}
		for stage, field in (
			('train', 'train_full_config'),
			('embedding', 'embedding_config'),
			('downstream', 'downstream_config'),
		):
			path = definition_path(cell[f'baseline_{stage}_config'], experiment)
			if (
				path.resolve()
				!= definition_path(base_cell[field], baseline_root).resolve()
			):
				raise ValueError(
					'mask-LR baseline must be its matched masked-v3 source'
				)
			if file_sha256(path) != cell['baseline_config_sha256'][stage]:
				raise ValueError('mask-LR baseline configuration hash drift')
			baseline[stage] = read_yaml(path)
		full = read_yaml(definition_path(cell['train_full_config'], experiment))
		expected = copy.deepcopy(baseline['train'])
		expected['identity']['model_tag'] = candidate
		expected['train']['mask_token_lr'] = cell['mask_token_lr']
		if 'train' in expected['identity']['scientific_identity']:
			expected['identity']['scientific_identity']['train']['mask_token_lr'] = (
				cell['mask_token_lr']
			)
		_owned(full['paths']['output_root'], candidate=candidate)
		expected['paths']['output_root'] = full['paths']['output_root']
		if full != expected:
			raise ValueError(
				'mask-LR training changes settings other than mask-token LR'
			)
		smoke = read_yaml(definition_path(cell['train_smoke_config'], experiment))
		expected = copy.deepcopy(full)
		expected['paths']['output_root'] = (
			full['paths']['output_root'].rsplit('/', 1)[0] + '/gpu_feasibility_1step'
		)
		expected['train'].update(epochs=1, max_steps=1)
		if 'train' in expected['identity']['scientific_identity']:
			expected['identity']['scientific_identity']['train'].update(
				epochs=1, max_steps=1
			)
		if smoke != expected:
			raise ValueError('mask-LR smoke inheritance drift')
		if read_yaml(definition_path(cell['audit_config'], experiment)) != {
			'training_configs': [cell['train_full_config']]
		}:
			raise ValueError('mask-LR source audit drift')
		embedding = read_yaml(definition_path(cell['embedding_config'], experiment))
		expected = copy.deepcopy(baseline['embedding'])
		_owned(embedding['embeddings']['output_dir'], candidate=candidate)
		expected['embeddings'].update(
			checkpoint=full['paths']['output_root'] + '/latest.pt',
			output_dir=embedding['embeddings']['output_dir'],
		)
		if embedding != expected:
			raise ValueError('mask-LR embedding inheritance drift')
		downstream = read_yaml(definition_path(cell['downstream_config'], experiment))
		replacements = {
			baseline_id: candidate,
			baseline['train']['paths']['output_root'] + '/latest.pt': full['paths'][
				'output_root'
			]
			+ '/latest.pt',
			baseline['embedding']['embeddings']['output_dir']: embedding['embeddings'][
				'output_dir'
			],
			cell['baseline_train_config']: cell['train_full_config'],
		}
		for key, value in downstream['outputs'].items():
			_owned(value)
			replacements[baseline['downstream']['outputs'][key]] = value
		if downstream != replace_values(baseline['downstream'], replacements):
			raise ValueError('mask-LR downstream decoder inheritance drift')
	return definition


def validate_live(experiment: Path, definition: dict[str, Any]) -> dict[str, Any]:
	"""Bind all controls to the completed F3 masked-v3 summary and sources."""
	baseline_root = experiment.parent / CONTROL
	control = validate_trace_mask_definition(baseline_root)
	validate_canonical_evidence(baseline_root, control)
	path = Path(os.path.expandvars(control['summary_root'])) / 'summary.json'
	if file_sha256(path) != definition['control_summary_sha256']:
		raise ValueError('mask-LR completed v3 control summary hash drift')
	completed = read_json(path)
	if completed.get('status') != 'complete' or len(completed['cells']) != 45:
		raise ValueError('mask-LR control requires all 45 completed masked-v3 cells')
	for cell in definition['cells']:
		baseline = read_yaml(definition_path(cell['baseline_train_config'], experiment))
		checkpoint = (
			Path(os.path.expandvars(baseline['paths']['output_root'])) / 'latest.pt'
		)
		if file_sha256(checkpoint) != cell['baseline_checkpoint_sha256']:
			raise ValueError('mask-LR completed control checkpoint hash drift')
		manifest = Path(os.path.expandvars(baseline['pseudo_targets']['manifest']))
		if file_sha256(manifest) != cell['target_manifest_sha256']:
			raise ValueError('mask-LR target manifest hash drift')
	return completed


def command_plan(  # noqa: C901, PLR0912
	experiment: Path, args: argparse.Namespace
) -> list[tuple[str, list[str]]]:
	"""Run the established stages, narrowing only to an owned candidate."""
	definition = validate_definition(experiment)
	first, last = STAGES.index(args.first), STAGES.index(args.last)
	if first > last:
		raise ValueError('mask-LR stage range is reversed')
	cells = definition['cells']
	if args.candidate:
		cells = [cell for cell in cells if cell['candidate_id'] == args.candidate]
		if not cells:
			raise ValueError('unknown mask-LR candidate')
		if 'summary' in STAGES[first : last + 1]:
			raise ValueError('mask-LR summary requires all six candidates')
	commands = []
	audited = False
	for stage in STAGES[first : last + 1]:
		if stage in ('audit', 'embeddings', 'downstream', 'summary') and not audited:
			commands.extend(
				(
					'audit',
					[
						sys.executable,
						'proc/seis_ssl_cluster/audit_strat_hmm_multi_head_sources.py',
						'--config',
						str(definition_path(cell['audit_config'], experiment)),
					],
				)
				for cell in cells
			)
			audited = True
		if stage == 'audit':
			continue
		if stage == 'summary':
			commands.append(
				(
					'summary',
					[
						sys.executable,
						'-m',
						'seis_ssl_cluster.hmm.mask_token_lr',
						'--experiment',
						str(experiment),
						'--publish-summary',
					],
				)
			)
			continue
		for cell in cells:
			if stage in ('smoke', 'full'):
				field = (
					'train_smoke_config' if stage == 'smoke' else 'train_full_config'
				)
				commands.append(
					(
						stage,
						[
							sys.executable,
							'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
							'--config',
							str(definition_path(cell[field], experiment)),
						],
					)
				)
			elif stage == 'embeddings':
				commands.append(
					(
						stage,
						[
							sys.executable,
							'proc/seis_ssl_cluster/extract_embeddings.py',
							'--config',
							str(definition_path(cell['embedding_config'], experiment)),
							'--device',
							'cuda',
						],
					)
				)
			else:
				for layout in LAYOUTS:
					for size in DATA_SIZES:
						commands.append(  # noqa: PERF401
							(
								stage,
								[
									sys.executable,
									'-m',
									'seis_ssl_cluster.hmm.multi_source_cell',
									'--survey',
									'f3',
									'--config',
									str(
										definition_path(
											cell['downstream_config'], experiment
										)
									),
									'--candidate',
									cell['candidate_id'],
									'--layout',
									layout,
									'--size',
									size,
								],
							)
						)
	return commands


def inspect_summary(experiment: Path) -> dict[str, Any]:
	"""Audit 90 matched candidate cells against the frozen masked-v3 control."""
	definition = validate_definition(experiment)
	control = validate_live(experiment, definition)
	cells, rows, arms, evidence = [], [], [], {}
	for cell in definition['cells']:
		selected, audit = _audited_arm(
			experiment, 'f3', cell, frozen_control_cells=control['cells']
		)
		cells.extend(
			{**item, 'mask_token_lr': cell['mask_token_lr']} for item in selected
		)
		for size in DATA_SIZES:
			row = comparison_row('f3', cell['source'], size, selected)
			row['candidate_mean'] = row.pop('v3_mean')
			row['control_mean'] = row.pop('v02_mean')
			row['candidate_secondary_mean'] = row.pop('v3_secondary_mean')
			row['control_secondary_mean'] = row.pop('v02_secondary_mean')
			rows.append({**row, 'mask_token_lr': cell['mask_token_lr']})
		layout_deltas = [
			statistics.fmean(
				item['primary'] - item['control_primary']
				for item in selected
				if item['layout_id'] == layout
			)
			for layout in LAYOUTS
		]
		arms.append(
			{
				'candidate_id': cell['candidate_id'],
				'source_family': cell['source'],
				'mask_token_lr': cell['mask_token_lr'],
				'cell_count': 15,
				'layout_clustered': {
					'cluster_count': 5,
					'primary_improvement': statistics.fmean(layout_deltas),
					'sample_sd': statistics.stdev(layout_deltas),
				},
			}
		)
		evidence[cell['candidate_id']] = audit
	if len(cells) != 90:
		raise ValueError('mask-LR summary requires exactly 90 candidate cells')
	payload = {
		'schema_version': 1,
		'status': 'complete',
		'survey': 'f3',
		'comparison': 'mask_token_lr_vs_masked_v3_1em5',
		'control_mask_token_lr': 1.0e-5,
		'execution_sha256': file_sha256(experiment / 'execution.yaml'),
		'control_summary_sha256': definition['control_summary_sha256'],
		'cells': cells,
		'rows': rows,
		'arms': arms,
		'evidence': evidence,
	}
	payload['summary_sha256'] = object_sha256(payload)
	return payload


def publication_contents(payload: dict[str, Any]) -> dict[str, str]:
	"""Create the exact three review files with explicit mask-only controls."""
	stream = io.StringIO(newline='')
	fields = (
		'source_family',
		'candidate_id',
		'baseline_id',
		'mask_token_lr',
		'data_size',
		'candidate_mean',
		'control_mean',
		'primary_improvement',
		'primary_improvement_sample_sd',
		'wins',
		'ties',
		'losses',
		'candidate_secondary_mean',
		'control_secondary_mean',
		'secondary_delta',
		'cell_count',
	)
	writer = csv.DictWriter(
		stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n'
	)
	writer.writeheader()
	writer.writerows(payload['rows'])
	lines = [
		'# F3 mask-token LR sweep versus masked v3',
		'',
		'Control mask-token LR=1e-5; encoder/head LR=1e-5; consistency=0.1.',
		(
			'Five paired layouts per size. '
			'Positive Mean IoU improvement favors the candidate.'
		),
		'',
		'| Source | Mask LR | Size | Candidate IoU | Control IoU | Delta |',
		'| --- | ---: | --- | ---: | ---: | ---: |',
	]
	lines.extend(
		f'| {row["source_family"]} | {row["mask_token_lr"]:g} | {row["data_size"]} '
		f'| {row["candidate_mean"]:.6f} | {row["control_mean"]:.6f} '
		f'| {row["primary_improvement"]:.6f} |'
		for row in payload['rows']
	)
	return {
		'comparison.csv': stream.getvalue(),
		'summary.json': json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
		+ '\n',
		'summary.md': '\n'.join(lines) + '\n',
	}


def summarize(experiment: Path) -> tuple[Path, ...]:
	"""Publish only complete audited results and reject conflicting publication."""
	definition = validate_definition(experiment)
	output = Path(os.path.expandvars(definition['summary_root']))
	contents = publication_contents(inspect_summary(experiment))
	if output.exists():
		if {p.name for p in output.iterdir()} != set(PUBLISHED_FILES) or any(
			(output / name).read_text() != content for name, content in contents.items()
		):
			raise ValueError('existing mask-LR publication drift')
	else:
		output.parent.mkdir(parents=True, exist_ok=True)
		staging = Path(tempfile.mkdtemp(prefix='.mask-lr-summary-', dir=output.parent))
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


def build_parser() -> argparse.ArgumentParser:
	"""Build a planning-first parser for the fixed six-arm experiment."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--experiment', type=Path, required=True)
	modes = parser.add_mutually_exclusive_group()
	for mode in ('plan', 'dry-run', 'execute', 'publish-summary'):
		modes.add_argument('--' + mode, action='store_true')
	parser.add_argument('--from', dest='first', choices=STAGES, default='smoke')
	parser.add_argument('--to', dest='last', choices=STAGES, default='summary')
	parser.add_argument('--candidate')
	return parser


def main() -> None:
	"""Run only explicitly requested stages under a candidate/summary lock."""
	args = build_parser().parse_args()
	definition = validate_definition(args.experiment)
	if args.publish_summary:
		if args.candidate:
			raise ValueError('mask-LR summary requires all six candidates')
		root = Path(os.path.expandvars(definition['output_root']))
		root.mkdir(parents=True, exist_ok=True)
		with (root / 'summary.lock').open('a') as lock:
			fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
			summarize(args.experiment)
		return
	commands = command_plan(args.experiment, args)
	for stage, command in commands:
		print(f'[{stage}] {command}')
	if not (args.execute or args.dry_run):
		return
	validate_live(args.experiment, definition)
	if args.dry_run:
		for stage, command in commands:
			run_command(stage, command, dry_run=True, log=None)
		return
	root = Path(os.path.expandvars(definition['output_root']))
	root.mkdir(parents=True, exist_ok=True)
	# An unfiltered executor owns all six candidate locks; a worker owns one.
	from contextlib import ExitStack

	with ExitStack() as stack:
		for candidate in sorted(
			[args.candidate] if args.candidate else CENTER_TRACE_MASK_LR_TAGS
		):
			lock = stack.enter_context((root / f'{candidate}.lock').open('a'))
			fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		logs = Path(tempfile.mkdtemp(prefix='invocation-', dir=root))
		print(f'logs: {logs}', flush=True)
		for index, (stage, command) in enumerate(commands):
			run_command(
				stage, command, dry_run=False, log=logs / f'{index:03d}-{stage}.log'
			)


if __name__ == '__main__':
	main()
