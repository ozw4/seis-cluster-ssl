"""Frozen 3e-5, 50:50 trace-mask follow-up for Parihaka and Volve."""

from __future__ import annotations

# ruff: noqa: PLC0415
import argparse
import copy
import fcntl
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from seis_ssl_cluster.hmm.multi_source_receipts import (
	DATA_SIZES,
	METRICS,
	file_sha256,
	object_sha256,
)
from seis_ssl_cluster.hmm.trace_mask_definition import (
	CANDIDATE_IDS as PARENT_IDS,
)
from seis_ssl_cluster.hmm.trace_mask_definition import (
	definition_path,
	read_yaml,
	replace_values,
	validate_trace_mask_definition,
)
from seis_ssl_cluster.hmm.trace_mask_driver import (
	STAGES,
	run_command,
	validate_canonical_evidence,
)
from seis_ssl_cluster.hmm.trace_mask_driver import (
	command_plan as parent_command_plan,
)
from seis_ssl_cluster.hmm.trace_mask_summary import (
	PUBLISHED_FILES,
	_audited_arm,
	_publication_contents,
	comparison_row,
)

NAMESPACE = 'hmm_v3_k6810_trace_mask_masklr3em5_v1'
EXPERIMENTS = {'parihaka': '46_' + NAMESPACE, 'volve': '39_' + NAMESPACE}
PARENTS = {
	'parihaka': '45_hmm_v3_k6810_trace_mask_multi_source_v1',
	'volve': '38_hmm_v3_k6810_trace_mask_multi_source_v1',
}
CANDIDATE_IDS = {
	family: candidate + '_masklr3em5' for family, candidate in PARENT_IDS.items()
}


def _owned(value: str, survey: str, candidate: str | None = None) -> None:
	parts = Path(value).parts
	if 'reports' in parts or NAMESPACE not in parts or survey not in parts:
		raise ValueError('LR3 outputs require the owned survey artifact namespace')
	if candidate is not None and candidate not in parts:
		raise ValueError('LR3 output candidate identity drift')


def validate_definition(experiment: Path) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
	"""Require exact parent inheritance, changing only the mask optimizer rate."""
	definition = read_yaml(experiment / 'execution.yaml')
	survey = definition.get('survey')
	if survey not in EXPERIMENTS or experiment.name != EXPERIMENTS[survey]:
		raise ValueError('LR3 experiment survey or root identity drift')
	parent_root = experiment.parent / PARENTS[survey]
	parent = validate_trace_mask_definition(parent_root)
	if (
		definition.get('schema_version') != 1
		or definition['experiment_id'] != experiment.name
		or definition_path(definition['experiment_root'], experiment).resolve()
		!= experiment.resolve()
		or definition.get('head_ks') != [6, 8, 10]
		or definition.get('mask_token_lr') != 3.0e-5
		or definition.get('parent_experiment') != parent_root.name
		or definition.get('parent_definition_sha256')
		!= file_sha256(parent_root / 'execution.yaml')
		or definition.get('canonical_selection') != parent['canonical_selection']
	):
		raise ValueError('LR3 frozen parent or canonical-selection identity drift')
	_owned(definition['output_root'], survey)
	_owned(definition['summary_root'], survey)
	cells = definition['cells']
	if len(cells) != 3 or {cell['candidate_id'] for cell in cells} != set(
		CANDIDATE_IDS.values()
	):
		raise ValueError('LR3 definition requires all three versioned source families')
	for cell in cells:
		family, candidate = cell['source'], cell['candidate_id']
		if (
			candidate != CANDIDATE_IDS[family]
			or cell['survey'] != survey
			or cell.get('mask_token_lr') != 3.0e-5
		):
			raise ValueError('LR3 candidate, survey or rate drift')
		base = next(item for item in parent['cells'] if item['source'] == family)
		for field in (
			'baseline_train_config',
			'baseline_embedding_config',
			'baseline_downstream_config',
			'baseline_config_sha256',
			'baseline_checkpoint_sha256',
			'target_manifest_sha256',
		):
			if cell[field] != base[field]:
				raise ValueError('LR3 canonical v02 baseline binding drift')
		full = read_yaml(definition_path(cell['train_full_config'], experiment))
		original = read_yaml(definition_path(base['train_full_config'], parent_root))
		expected = copy.deepcopy(original)
		expected['identity']['model_tag'] = candidate
		expected['train']['mask_token_lr'] = 3.0e-5
		if 'train' in expected['identity']['scientific_identity']:
			expected['identity']['scientific_identity']['train']['mask_token_lr'] = (
				3.0e-5
			)
		_owned(full['paths']['output_root'], survey, candidate)
		expected['paths']['output_root'] = full['paths']['output_root']
		if full != expected:
			raise ValueError('LR3 training changes settings other than mask-token LR')
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
			raise ValueError('LR3 smoke inheritance drift')
		if read_yaml(definition_path(cell['audit_config'], experiment)) != {
			'training_configs': [cell['train_full_config']]
		}:
			raise ValueError('LR3 source audit binding drift')
		embedding = read_yaml(definition_path(cell['embedding_config'], experiment))
		original_embedding = read_yaml(
			definition_path(base['embedding_config'], parent_root)
		)
		expected = copy.deepcopy(original_embedding)
		_owned(embedding['embeddings']['output_dir'], survey, candidate)
		expected['embeddings'].update(
			checkpoint=full['paths']['output_root'] + '/latest.pt',
			output_dir=embedding['embeddings']['output_dir'],
		)
		if embedding != expected:
			raise ValueError('LR3 embedding inheritance drift')
		downstream = read_yaml(definition_path(cell['downstream_config'], experiment))
		original_downstream = read_yaml(
			definition_path(base['downstream_config'], parent_root)
		)
		replacements = {
			base['candidate_id']: candidate,
			original['paths']['output_root'] + '/latest.pt': full['paths'][
				'output_root'
			]
			+ '/latest.pt',
			original_embedding['embeddings']['output_dir']: embedding['embeddings'][
				'output_dir'
			],
			base['train_full_config']: cell['train_full_config'],
		}
		for key, value in downstream['outputs'].items():
			_owned(value, survey)
			replacements[original_downstream['outputs'][key]] = value
		if downstream != replace_values(original_downstream, replacements):
			raise ValueError('LR3 decoder inheritance drift')
	return definition


def command_plan(
	experiment: Path, args: argparse.Namespace
) -> list[tuple[str, list[str]]]:
	"""Map established survey stages to isolated LR3 outputs without rerunning v2."""
	definition = validate_definition(experiment)
	if STAGES.index(args.first) > STAGES.index(args.last):
		raise ValueError('LR3 stage range is reversed')
	base_args = copy.copy(args)
	base_args.layout = base_args.size = None
	if args.candidate:
		if args.candidate not in CANDIDATE_IDS.values():
			raise ValueError('unknown LR3 candidate')
		base_args.candidate = args.candidate.removesuffix('_masklr3em5')
	if args.last == 'summary':
		base_args.last = 'downstream'
		if args.first == 'summary':
			base_args.first = base_args.last = 'audit'
	parent_root = experiment.parent / PARENTS[definition['survey']]
	parent = validate_trace_mask_definition(parent_root)
	replacements = {}
	for cell in definition['cells']:
		base = next(
			item for item in parent['cells'] if item['source'] == cell['source']
		)
		replacements[base['candidate_id']] = cell['candidate_id']
		for field in (
			'train_smoke_config',
			'train_full_config',
			'audit_config',
			'embedding_config',
			'downstream_config',
		):
			replacements[str(definition_path(base[field], parent_root))] = str(
				definition_path(cell[field], experiment)
			)
	commands = [
		(stage, replace_values(command, replacements))
		for stage, command in parent_command_plan(parent_root, base_args)
	]
	if args.last == 'summary':
		command = [
			sys.executable,
			'-m',
			'seis_ssl_cluster.hmm.trace_mask_lr3',
			'--experiment',
			str(experiment),
			'--publish-summary',
		]
		if args.candidate:
			command += ['--candidate', args.candidate]
		commands.append(('summary', command))
	return commands


def inspect_summary(experiment: Path, candidate: str | None = None) -> dict[str, Any]:
	"""Audit a complete 15-cell candidate or all 45 cells against canonical v02."""
	definition = validate_definition(experiment)
	validate_canonical_evidence(experiment, definition)
	chosen = [
		cell
		for cell in definition['cells']
		if candidate is None or cell['candidate_id'] == candidate
	]
	if not chosen:
		raise ValueError('unknown LR3 summary candidate')
	survey = definition['survey']
	cells, rows, arms, evidence = [], [], [], {}
	for cell in chosen:
		selected, audit = _audited_arm(experiment, survey, cell)
		if len(selected) != 15:
			raise ValueError('LR3 summary requires 15 complete paired cells per source')
		cells.extend(selected)
		rows.extend(
			comparison_row(survey, cell['source'], size, selected)
			for size in DATA_SIZES
		)
		sign = -1 if survey == 'volve' else 1
		deltas = [
			statistics.fmean(
				sign * (item['primary'] - item['control_primary'])
				for item in selected
				if item['layout_id'] == f'layout_{index:03d}'
			)
			for index in range(5)
		]
		arms.append(
			{
				'candidate_id': cell['candidate_id'],
				'source_family': cell['source'],
				'layout_clustered': {
					'cluster_count': 5,
					'primary_improvement': statistics.fmean(deltas),
					'sample_sd': statistics.stdev(deltas),
				},
			}
		)
		evidence[cell['candidate_id']] = audit
	payload = {
		'schema_version': 1,
		'status': 'complete_candidate' if candidate else 'complete',
		'survey': survey,
		**METRICS[survey],
		'candidate_scope': candidate,
		'comparison': 'hmm_v3_masklr3em5_50_50_vs_canonical_v02_cons010',
		'mask_token_lr': 3.0e-5,
		'encoder_lr': 1.0e-5,
		'head_lr': 1.0e-5,
		'masked_prototype_weight': 0.5,
		'visible_prototype_weight': 0.5,
		'head_ks': [6, 8, 10],
		'consistency_weight': 0.1,
		'execution_sha256': file_sha256(experiment / 'execution.yaml'),
		'canonical_selection': definition['canonical_selection'],
		'cells': cells,
		'rows': rows,
		'arms': arms,
		'evidence': evidence,
	}
	payload['summary_sha256'] = object_sha256(payload)
	return payload


def publication_contents(payload: dict[str, Any]) -> dict[str, str]:
	"""Retain the exact published set and make LR/scope explicit in the review."""
	contents = _publication_contents(payload)
	contents['summary.md'] += (
		'\nMask-token LR=3e-5; encoder/head LR=1e-5; masked:visible=50:50.\n'
		f'Scope: {payload["candidate_scope"] or "all three source families"}. '
		'Five paired layouts per size; one pretraining seed.\n'
		'Fixed external test comparison, not independent held-out tuning.\n'
	)
	return contents


def summarize(experiment: Path, candidate: str | None = None) -> tuple[Path, ...]:
	"""Publish only complete audited scopes; never relabel a partial matrix."""
	definition = validate_definition(experiment)
	output = Path(os.path.expandvars(definition['summary_root']))
	if candidate is not None:
		output = output.parent / 'candidates' / candidate
	contents = publication_contents(inspect_summary(experiment, candidate))
	if output.exists():
		if {path.name for path in output.iterdir()} != set(PUBLISHED_FILES) or any(
			(output / name).read_text() != content for name, content in contents.items()
		):
			raise ValueError('existing LR3 publication drift')
	else:
		output.parent.mkdir(parents=True, exist_ok=True)
		staging = Path(tempfile.mkdtemp(prefix='.lr3-summary-', dir=output.parent))
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
	"""Build planning-first, explicitly scoped follow-up arguments."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--experiment', type=Path, required=True)
	modes = parser.add_mutually_exclusive_group()
	for mode in ('plan', 'dry-run', 'execute', 'publish-summary'):
		modes.add_argument('--' + mode, action='store_true')
	parser.add_argument('--from', dest='first', choices=STAGES, default='smoke')
	parser.add_argument('--to', dest='last', choices=STAGES, default='summary')
	parser.add_argument('--candidate', choices=tuple(CANDIDATE_IDS.values()))
	return parser


def main() -> None:
	"""Execute only the selected new candidates with matching-checkpoint recovery."""
	args = build_parser().parse_args()
	definition = validate_definition(args.experiment)
	if args.publish_summary:
		root = Path(os.path.expandvars(definition['output_root']))
		root.mkdir(parents=True, exist_ok=True)
		with (root / f'{args.candidate or "all"}-summary.lock').open('a') as lock:
			fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
			summarize(args.experiment, args.candidate)
		return
	commands = command_plan(args.experiment, args)
	for stage, command in commands:
		print(f'[{stage}] {command}', flush=True)
	if not (args.execute or args.dry_run):
		return
	validate_canonical_evidence(args.experiment, definition)
	if args.dry_run:
		for stage, command in commands:
			run_command(stage, command, dry_run=True, log=None)
		return
	root = Path(os.path.expandvars(definition['output_root']))
	root.mkdir(parents=True, exist_ok=True)
	from contextlib import ExitStack

	with ExitStack() as stack:
		for candidate in sorted(
			[args.candidate] if args.candidate else CANDIDATE_IDS.values()
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
