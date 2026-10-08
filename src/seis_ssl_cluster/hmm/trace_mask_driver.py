"""Plan and execute the source-bound K6810 v3 trace-mask experiment."""

from __future__ import annotations

# ruff: noqa: PLC0415, S603, PERF401
import argparse
import fcntl
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.hmm.multi_source_driver import audit_existing_embeddings
from seis_ssl_cluster.hmm.multi_source_receipts import (
	DATA_SIZES,
	file_sha256,
	read_json,
)
from seis_ssl_cluster.hmm.trace_mask_definition import (
	definition_path,
	read_yaml,
	validate_trace_mask_definition,
)

STAGES = ('smoke', 'full', 'audit', 'embeddings', 'downstream', 'summary')
LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))


def validate_canonical_evidence(experiment: Path, definition: dict[str, Any]) -> None:
	"""Bind execution to the completed artifact selected as canonical v02."""
	selection = read_yaml(
		definition_path(definition['canonical_selection']['path'], experiment)
	)
	evidence = selection['evidence']
	root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	path = root / evidence['artifact_summary']
	if 'reports' in path.resolve().parts or not path.resolve().is_relative_to(
		root.resolve()
	):
		raise ValueError('canonical v02 evidence must be an artifact input')
	if file_sha256(path) != evidence['artifact_summary_sha256']:
		raise ValueError('canonical v02 artifact summary hash drift')
	if read_json(path).get('status') != 'complete':
		raise ValueError('canonical v02 evidence is incomplete')
	for cell in definition['cells']:
		training = read_yaml(definition_path(cell['train_full_config'], experiment))
		manifest = Path(os.path.expandvars(training['pseudo_targets']['manifest']))
		if file_sha256(manifest) != cell['target_manifest_sha256']:
			raise ValueError('v3 fixed target manifest hash drift')
		baseline = read_yaml(definition_path(cell['baseline_train_config'], experiment))
		checkpoint = (
			Path(os.path.expandvars(baseline['paths']['output_root'])) / 'latest.pt'
		)
		if file_sha256(checkpoint) != cell['baseline_checkpoint_sha256']:
			raise ValueError('canonical v02 baseline checkpoint hash drift')


def command_plan(  # noqa: C901, PLR0912
	experiment: Path, args: argparse.Namespace
) -> list[tuple[str, list[str]]]:
	"""Enumerate only the three new v3 arms and their fifteen decoder cells."""
	definition = validate_trace_mask_definition(experiment)
	cells = definition['cells']
	if args.candidate:
		cells = [cell for cell in cells if cell['candidate_id'] == args.candidate]
		if not cells:
			raise ValueError('unknown v3 candidate')
	first, last = STAGES.index(args.first), STAGES.index(args.last)
	if first > last:
		raise ValueError('stage range is reversed')
	if (args.layout or args.size) and (args.first, args.last) != (
		'downstream',
		'downstream',
	):
		raise ValueError('cell narrowing requires downstream only')
	if args.candidate and 'summary' in STAGES[first : last + 1]:
		raise ValueError('summary requires all three source families')
	commands = []
	audited = False
	for stage in STAGES[first : last + 1]:
		if stage in ('audit', 'embeddings', 'downstream', 'summary') and not audited:
			for cell in cells:
				commands.append(
					(
						'audit',
						[
							sys.executable,
							'proc/seis_ssl_cluster/audit_strat_hmm_multi_head_sources.py',
							'--config',
							str(definition_path(cell['audit_config'], experiment)),
						],
					)
				)
			audited = True
		if stage == 'audit':
			continue
		if stage == 'summary':
			commands.append(
				(
					stage,
					[
						sys.executable,
						'-m',
						'seis_ssl_cluster.hmm.trace_mask_summary',
						'--experiment',
						str(experiment),
					],
				)
			)
			continue
		for cell in cells:
			candidate = cell['candidate_id']
			if stage in ('smoke', 'full'):
				field = (
					'train_smoke_config' if stage == 'smoke' else 'train_full_config'
				)
				command = [
					sys.executable,
					'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
					'--config',
					str(definition_path(cell[field], experiment)),
				]
				commands.append((stage, command))
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
				for layout in [args.layout] if args.layout else LAYOUTS:
					for size in [args.size] if args.size else DATA_SIZES:
						commands.append(
							(
								stage,
								[
									sys.executable,
									'-m',
									'seis_ssl_cluster.hmm.multi_source_cell',
									'--survey',
									definition['survey'],
									'--config',
									str(
										definition_path(
											cell['downstream_config'], experiment
										)
									),
									'--candidate',
									candidate,
									'--layout',
									layout,
									'--size',
									size,
								],
							)
						)
	return commands


def training_restart(config_path: Path) -> tuple[bool, Path | None]:
	"""Reuse a validated final checkpoint or resume only this exact v3 identity."""
	from seis_ssl_cluster.training import load_checkpoint
	from seis_ssl_cluster.training.strat_hmm_checkpoint import (
		inspect_stratigraphy_checkpoint,
		validate_stratigraphy_checkpoint_payload,
	)
	from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source

	config = resolve_strat_hmm_pretext_config(load_config(config_path))
	latest = Path(config['paths']['output_root']) / 'latest.pt'
	if not latest.exists():
		return False, None
	payload = load_checkpoint(latest, map_location='cpu')
	validate_stratigraphy_checkpoint_payload(payload, expected_config=config)
	if payload.get('stratigraphy_config') != config:
		raise ValueError('v3 restart checkpoint configuration or output identity drift')
	compatibility = inspect_stratigraphy_checkpoint(payload, expected_config=config)[
		'resume_compatibility'
	]
	if not compatibility['compatible']:
		raise ValueError(f'v3 checkpoint cannot resume: {compatibility["reason"]}')
	train = config['train']
	if train['max_steps'] is not None:
		return payload['global_step'] >= train['max_steps'], latest
	steps = train['epochs'] * math.ceil(
		train['samples_per_epoch'] / train['batch_size']
	)
	if payload['epoch'] == train['epochs'] and payload['global_step'] == steps:
		audit_multi_head_source(config_path)
		return True, latest
	if payload['epoch'] > train['epochs'] or payload['global_step'] > steps:
		raise ValueError('v3 checkpoint counters exceed the owned training recipe')
	return False, latest


def run_command(  # noqa: C901
	stage: str, command: list[str], *, dry_run: bool, log: Path | None
) -> None:
	"""Dispatch existing stages with automatic identity-bound restart."""
	environment = os.environ.copy()
	command = list(command)
	if '--config' in command:
		path = Path(command[command.index('--config') + 1])
		raw = read_yaml(path)
	if stage in ('smoke', 'full'):
		manifest = Path(os.path.expandvars(raw['pseudo_targets']['manifest']))
		environment['SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256'] = file_sha256(
			manifest
		)
		if not dry_run:
			complete, latest = training_restart(path)
			if complete:
				print('execution: reused audited complete v3 pretraining')
				return
			if latest is not None:
				command += ['--resume', str(latest)]
	elif stage == 'embeddings' and not dry_run:
		output = Path(os.path.expandvars(raw['embeddings']['output_dir']))
		if output.exists():
			audit_existing_embeddings(path)
			print('execution: reused audited complete v3 embeddings')
			return
	elif stage == 'downstream' and not dry_run:
		root = Path(os.path.expandvars(raw['outputs']['runs_root']))
		candidate = command[command.index('--candidate') + 1]
		layout = command[command.index('--layout') + 1]
		size = command[command.index('--size') + 1]
		survey = command[command.index('--survey') + 1]
		cell = root / f'model={candidate}' / f'layout={layout}' / f'size={size}'
		latest = cell / ('decoder/latest.pt' if survey == 'f3' else 'latest.pt')
		if latest.exists():
			command += ['--resume', str(latest)]
	if dry_run:
		command += ['--dry-run']
	if log is None:
		subprocess.run(command, check=True, env=environment)
	else:
		with log.open('w', encoding='utf-8') as stream:
			stream.write(shlex.join(command) + '\n')
			stream.flush()
			subprocess.run(
				command,
				check=True,
				env=environment,
				stdout=stream,
				stderr=subprocess.STDOUT,
			)


def build_parser() -> argparse.ArgumentParser:
	"""Build the parser for source-bound trace-mask experiment execution."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--experiment', type=Path, required=True)
	modes = parser.add_mutually_exclusive_group()
	for mode in ('plan', 'dry-run', 'execute'):
		modes.add_argument('--' + mode, action='store_true')
	parser.add_argument('--from', dest='first', choices=STAGES, default='smoke')
	parser.add_argument('--to', dest='last', choices=STAGES, default='summary')
	parser.add_argument('--candidate')
	parser.add_argument('--layout', choices=LAYOUTS)
	parser.add_argument('--size', choices=DATA_SIZES)
	return parser


def main() -> None:
	"""Plan by default; executing takes an exclusive v3 survey lock."""
	parser = build_parser()
	args = parser.parse_args()
	commands = command_plan(args.experiment, args)
	for stage, command in commands:
		print(f'[{stage}] {shlex.join(command)}')
	if not (args.execute or args.dry_run):
		return
	definition = validate_trace_mask_definition(args.experiment)
	validate_canonical_evidence(args.experiment, definition)
	if args.dry_run:
		for stage, command in commands:
			run_command(stage, command, dry_run=True, log=None)
		return
	logs = Path(os.path.expandvars(definition['output_root'])) / 'logs'
	if not logs.is_absolute() or 'reports' in logs.parts:
		raise ValueError('v3 execution logs must use an absolute artifact path')
	logs.mkdir(parents=True, exist_ok=True)
	with (logs / 'driver.lock').open('a', encoding='utf-8') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		invocation = Path(tempfile.mkdtemp(prefix='invocation-', dir=logs))
		print(f'logs: {invocation}')
		for index, (stage, command) in enumerate(commands):
			run_command(
				stage,
				command,
				dry_run=False,
				log=invocation / f'{index:03d}-{stage}.log',
			)


if __name__ == '__main__':
	main()
