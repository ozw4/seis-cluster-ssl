"""Execute one source-audited cell with strict completed-output reuse."""

from __future__ import annotations

# ruff: noqa: PLC0415, S603
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from seis_ssl_cluster.config import load_config


def audit_completed_cell(
	survey: str, raw: dict, candidate: str, layout: str, size: str
) -> dict[str, object]:
	"""Reuse the survey's existing completed decoder and evaluation auditors."""
	if survey == 'f3':
		from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
			audit_f3_lithology_candidate_source,
			f3_lithology_candidate_config_from_mapping,
			load_f3_lithology_candidate_canonical_config,
		)
		from seis_ssl_cluster.f3.lithology.paired_candidate_results import (
			F3PairedCandidateModel,
			F3PairedCandidateSummaryConfig,
			read_f3_paired_candidate_job,
		)

		config = f3_lithology_candidate_config_from_mapping(raw)
		canonical = load_f3_lithology_candidate_canonical_config(config)
		model = F3PairedCandidateModel(
			candidate, config.checkpoint, config.embeddings_dir
		)
		pair = F3PairedCandidateSummaryConfig(
			config.canonical_config, config.runs_root, config.summary_root, model, model
		)
		return read_f3_paired_candidate_job(
			pair,
			canonical,
			model,
			audit_f3_lithology_candidate_source(config, canonical),
			layout_id=layout,
			data_size=size,
		)
	if survey == 'volve':
		from seis_ssl_cluster.volve.horizon_five_way_results import (
			inspect_volve_horizon_recipe_arm_cell,
		)
		from seis_ssl_cluster.volve.horizon_recipe_arm import (
			volve_horizon_recipe_arm_config_from_mapping,
		)

		return inspect_volve_horizon_recipe_arm_cell(
			volve_horizon_recipe_arm_config_from_mapping(raw),
			layout_id=layout,
			data_size=size,
		)
	from seis_ssl_cluster.parihaka.channel_completion import (
		inspect_completed_channel_job,
	)
	from seis_ssl_cluster.parihaka.channel_decoder import (
		channel_decoder_config_from_mapping,
		channel_decoder_run_identity,
		inspect_channel_decoder_job,
	)

	config = channel_decoder_config_from_mapping(raw)
	plan = inspect_channel_decoder_job(
		config,
		model=candidate,
		layout_id=layout,
		data_size=size,
		layout_config=Path(
			'experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml'
		),
	)
	metrics = json.loads((plan.output_dir / 'metrics.json').read_text())
	if metrics['benchmark_identity'] != channel_decoder_run_identity(plan):
		raise ValueError('completed Channel cell differs from current plan')
	return {
		'benchmark_identity': metrics['benchmark_identity'],
		'completion': inspect_completed_channel_job(
			plan.output_dir, metrics=metrics
		),
		'metrics_path': str(plan.output_dir / 'metrics.json'),
	}


def main() -> None:
	"""Validate, audit or execute exactly one fixed benchmark cell."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--survey', choices=('f3', 'parihaka', 'volve'), required=True)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--candidate', required=True)
	parser.add_argument('--layout', required=True)
	parser.add_argument('--size', choices=('small', 'medium', 'large'), required=True)
	parser.add_argument('--resume', type=Path)
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()
	raw = load_config(args.config)
	root = Path(raw['outputs']['runs_root'])
	cell = (
		root / f'model={args.candidate}' / f'layout={args.layout}' / f'size={args.size}'
	)
	latest = cell / ('decoder/latest.pt' if args.survey == 'f3' else 'latest.pt')
	if args.resume is not None and args.resume.resolve() != latest.resolve():
		raise ValueError('resume must identify this exact cell latest.pt')
	scripts = {
		'f3': 'run_f3_lithology_candidate.py',
		'parihaka': 'run_parihaka_channel_decoder.py',
		'volve': 'run_volve_horizon_recipe_arm.py',
	}
	command = [
		sys.executable,
		'proc/seis_ssl_cluster/' + scripts[args.survey],
		'--config',
		str(args.config),
		'--layout',
		args.layout,
		'--size',
		args.size,
	]
	if args.survey != 'f3':
		command += ['--model', args.candidate]
	if args.survey == 'parihaka':
		command += [
			'--layout-config',
			'experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml',
		]
	if args.resume is not None:
		command += ['--resume', str(args.resume)]
	if args.dry_run:
		subprocess.run([*command, '--dry-run'], check=True)
		return
	lock_root = root.parent / 'cell_locks'
	lock_root.mkdir(parents=True, exist_ok=True)
	with (lock_root / f'{args.candidate}.{args.layout}.{args.size}.lock').open(
		'a'
	) as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		metrics = cell / (
			'evaluation/metrics.json' if args.survey == 'f3' else 'metrics.json'
		)
		if metrics.exists():
			audit_completed_cell(
				args.survey, raw, args.candidate, args.layout, args.size
			)
			print('execution: reused audited complete cell')
		else:
			subprocess.run(command, check=True)


if __name__ == '__main__':
	main()
