"""Sequential, resumable command driver for the fixed K6810 experiment."""

from __future__ import annotations

# ruff: noqa: PLC0415, S603
import argparse
import fcntl
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from seis_ssl_cluster.hmm.multi_source_definition import (
	validate_multi_source_experiment_definition,
)
from seis_ssl_cluster.hmm.multi_source_receipts import DATA_SIZES

STAGES = (
	'targets',
	'replay',
	'export',
	'manifests',
	'smoke',
	'full',
	'audit',
	'embeddings',
	'downstream',
	'summary',
)
LAYOUTS = tuple(f'layout_{i:03d}' for i in range(5))


def _read(path: Path) -> dict:
	return yaml.safe_load(path.read_text())


def _expand(value: str) -> str:
	return os.path.expandvars(value)


def command_plan(  # noqa: C901, PLR0912, PLR0915
	experiment: Path, args: argparse.Namespace
) -> list[tuple[str, list[str]]]:
	"""Enumerate only the eight new arms; the reused arm has no write route."""
	validate_multi_source_experiment_definition(experiment)
	definition = _read(experiment / 'execution.yaml')
	survey = definition['survey']
	arms = definition['arms']
	if args.candidate:
		arms = {f: a for f, a in arms.items() if a['candidate_id'] == args.candidate}
		if not arms:
			raise ValueError('candidate is unknown or read-only reuse')
	start, stop = STAGES.index(args.first), STAGES.index(args.last)
	if start > stop:
		raise ValueError('stage range is reversed')
	if (args.layout or args.size) and (args.first, args.last) != (
		'downstream',
		'downstream',
	):
		raise ValueError('cell narrowing requires downstream only')
	if args.candidate and args.first != args.last:
		raise ValueError('candidate narrowing requires one stage')
	if args.resume and (
		args.first != args.last
		or args.first not in ('smoke', 'full', 'downstream')
		or not args.candidate
		or (args.first == 'downstream' and not (args.layout and args.size))
	):
		raise ValueError(
			'resume requires one candidate stage and an exact downstream cell'
		)
	commands = []
	python = sys.executable

	def add(stage: str, command: list[str]) -> None:
		commands.append((stage, command))

	def audit() -> None:
		for arm in arms.values():
			add(
				'audit',
				[
					python,
					'proc/seis_ssl_cluster/audit_strat_hmm_multi_head_sources.py',
					'--config',
					str(
						experiment
						/ '30_pretraining'
						/ arm['candidate_id']
						/ '03_audit.yaml'
					),
				],
			)

	audited = False
	for stage in STAGES[start : stop + 1]:
		if stage in ('audit', 'embeddings', 'downstream', 'summary') and not audited:
			audit()
			audited = True
		if stage == 'audit':
			continue
		if stage == 'summary':
			add(
				stage,
				[
					python,
					'-m',
					'seis_ssl_cluster.hmm.multi_source_survey',
					'--config',
					str(experiment / '60_summary/01_paired_comparison.yaml'),
				],
			)
			continue
		for arm in arms.values():
			cid = arm['candidate_id']
			family = arm['target_family']
			if stage in ('targets', 'replay'):
				name = (
					'01_cluster_hmm_k8k10.yaml'
					if stage == 'targets'
					else '02_replay_hmm_k6.yaml'
				)
				add(
					stage,
					[
						python,
						'proc/seis_ssl_cluster/cluster_embeddings.py',
						'--config',
						str(experiment / '10_targets' / family / name),
					],
				)
			elif stage == 'export':
				add(
					stage,
					[
						'bash',
						str(
							experiment
							/ '10_targets'
							/ family
							/ '03_export_pseudo_targets.sh'
						),
					],
				)
			elif stage == 'manifests':
				c = _read(experiment / '20_manifests' / f'{cid}.yaml')
				cmd = [
					python,
					'proc/seis_ssl_cluster/build_strat_hmm_multi_head_targets.py',
					'--source-embedding-dir',
					_expand(c['source_embedding_dir']),
					'--manifest',
					_expand(c['manifest']),
					'--replay-k6-root',
					_expand(c['replay_k6_root']),
					'--only-missing',
				]
				for k, root in c['head_roots'].items():
					cmd += ['--head-root', f'{k}={_expand(root)}']
				add(stage, cmd)
			elif stage in ('smoke', 'full'):
				name = (
					'01_gpu_feasibility_1step.yaml'
					if stage == 'smoke'
					else '02_full_25ep.yaml'
				)
				path = experiment / '30_pretraining' / cid / name
				cmd = [
					python,
					'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
					'--config',
					str(path),
				]
				if args.resume:
					cmd += [
						'--resume',
						_expand(_read(path)['paths']['output_root']) + '/latest.pt',
					]
				add(stage, cmd)
			elif stage == 'embeddings':
				add(
					stage,
					[
						python,
						'proc/seis_ssl_cluster/extract_embeddings.py',
						'--config',
						str(experiment / '40_embeddings' / f'{cid}.yaml'),
						'--device',
						'cuda',
					],
				)
			elif stage == 'downstream':
				path = experiment / '50_downstream' / f'{cid}.yaml'
				for layout in [args.layout] if args.layout else LAYOUTS:
					for size in [args.size] if args.size else DATA_SIZES:
						cmd = [
							python,
							'-m',
							'seis_ssl_cluster.hmm.multi_source_cell',
							'--survey',
							survey,
							'--config',
							str(path),
							'--candidate',
							cid,
							'--layout',
							layout,
							'--size',
							size,
						]
						if args.resume:
							root = _expand(_read(path)['outputs']['runs_root'])
							suffix = (
								'decoder/latest.pt' if survey == 'f3' else 'latest.pt'
							)
							cmd += [
								'--resume',
								f'{root}/model={cid}/layout={layout}/size={size}/{suffix}',
							]
						add(stage, cmd)
	return commands


def audit_existing_embeddings(path: Path) -> None:
	"""Require complete, source-bound embeddings before read-only reuse."""
	from seis_ssl_cluster.config import load_config

	experiment = path.parent.parent
	survey = _read(experiment / 'execution.yaml')['survey']
	raw = load_config(experiment / '50_downstream' / path.name)
	if survey == 'f3':
		from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
			audit_f3_lithology_candidate_source,
			f3_lithology_candidate_config_from_mapping,
			load_f3_lithology_candidate_canonical_config,
		)

		config = f3_lithology_candidate_config_from_mapping(raw)
		audit_f3_lithology_candidate_source(
			config, load_f3_lithology_candidate_canonical_config(config)
		)
	elif survey == 'parihaka':
		from seis_ssl_cluster.parihaka.channel_decoder import (
			channel_decoder_config_from_mapping,
			inspect_embedding_sources,
		)

		inspect_embedding_sources(channel_decoder_config_from_mapping(raw))
	else:
		from seis_ssl_cluster.volve.horizon_recipe_arm import (
			inspect_volve_horizon_recipe_arm_embedding_suite,
			volve_horizon_recipe_arm_config_from_mapping,
		)

		config = volve_horizon_recipe_arm_config_from_mapping(raw)
		inspect_volve_horizon_recipe_arm_embedding_suite(
			config, model_ids=(config.arm_id,)
		)


def run_command(
	stage: str, command: list[str], *, dry_run: bool, log: Path | None
) -> None:
	"""Bind the live manifest and refuse to replace any clustering output."""
	environment = os.environ.copy()
	if stage in ('smoke', 'full'):
		from seis_ssl_cluster.clustering.features import file_sha256
		from seis_ssl_cluster.training.strat_hmm_source_audit import (
			audit_multi_head_source,
		)

		path = Path(command[command.index('--config') + 1])
		raw = _read(path)
		environment['SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256'] = file_sha256(
			_expand(raw['pseudo_targets']['manifest'])
		)
		output = Path(_expand(raw['paths']['output_root']))
		if (
			not dry_run
			and stage == 'full'
			and (output / 'latest.pt').exists()
			and '--resume' not in command
		):
			audit_multi_head_source(path)
			print('execution: reused audited complete pretraining')
			return
	if stage == 'embeddings' and not dry_run:
		path = Path(command[command.index('--config') + 1])
		output = Path(_expand(_read(path)['embeddings']['output_dir']))
		if output.exists():
			audit_existing_embeddings(path)
			print('execution: reused audited complete embeddings')
			return
	if stage in ('targets', 'replay') and not dry_run:
		path = Path(command[command.index('--config') + 1])
		output = Path(_expand(_read(path)['clustering']['output_dir']))
		if output.exists():
			raise FileExistsError(f'clustering output exists: {output}')
	if dry_run:
		command = [*command, '--dry-run']
	if log is None:
		subprocess.run(command, check=True, env=environment)
	else:
		with log.open('w') as stream:
			stream.write(shlex.join(command) + '\n')
			stream.flush()
			subprocess.run(
				command,
				check=True,
				env=environment,
				stdout=stream,
				stderr=subprocess.STDOUT,
			)


def main() -> None:
	"""Plan by default; live execution takes an exclusive survey lock."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--experiment', type=Path, required=True)
	modes = parser.add_mutually_exclusive_group()
	for mode in ('plan', 'dry-run', 'execute'):
		modes.add_argument('--' + mode, action='store_true')
	parser.add_argument('--from', dest='first', choices=STAGES, default=STAGES[0])
	parser.add_argument('--to', dest='last', choices=STAGES, default=STAGES[-1])
	parser.add_argument('--candidate')
	parser.add_argument('--layout', choices=LAYOUTS)
	parser.add_argument('--size', choices=DATA_SIZES)
	parser.add_argument('--resume', action='store_true')
	args = parser.parse_args()
	commands = command_plan(args.experiment, args)
	for stage, command in commands:
		print(f'[{stage}] {shlex.join(command)}')
	if not (args.execute or args.dry_run):
		return
	if args.dry_run:
		for stage, command in commands:
			run_command(stage, command, dry_run=True, log=None)
		return
	artifact = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	if not artifact.is_absolute():
		raise ValueError('artifact root must be absolute')
	survey = _read(args.experiment / 'execution.yaml')['survey']
	logs = artifact / 'hmm_v2/k6810_multi_source_evaluation_v1' / survey / 'logs'
	logs.mkdir(parents=True, exist_ok=True)
	with (logs / 'driver.lock').open('a') as lock:
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
