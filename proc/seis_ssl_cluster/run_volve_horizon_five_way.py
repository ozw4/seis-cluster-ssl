'''Run one cell of the Volve horizon five-way comparison.'''

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_five_way_config import (
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_runner import (
	inspect_volve_horizon_five_way_job,
	resolve_volve_horizon_five_way_job,
	run_volve_horizon_five_way_job,
)
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments/volve/horizon_benchmark_v1'
	/ '31_mae_local_bt_hmm_five_way_v1/50_five_way.yaml'
)
DEFAULT_LAYOUT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments/volve/horizon_benchmark_v1/20_horizon_supervision'
	/ '01_layouts.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	'''Build the one-cell five-way CLI parser.'''
	parser = argparse.ArgumentParser(
		description='Run one frozen Volve horizon five-way decoder job.'
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument('--model', required=True)
	parser.add_argument('--layout', required=True, choices=LAYOUT_IDS)
	parser.add_argument('--size', required=True, choices=tuple(DATA_SIZE_PREFIX))
	parser.add_argument(
		'--layout-config', type=Path, default=DEFAULT_LAYOUT_CONFIG
	)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
	parser.add_argument('--max-steps', type=int)
	parser.add_argument('--resume', type=Path)
	return parser


def main() -> None:
	'''Inspect or execute exactly one five-way condition.'''
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = volve_horizon_five_way_config_from_mapping(raw)
	job = resolve_volve_horizon_five_way_job(
		config,
		model=args.model,
		layout=args.layout,
		size=args.size,
	)
	plan = inspect_volve_horizon_five_way_job(
		job,
		layout_config=args.layout_config,
	)
	if args.dry_run:
		identity = plan.run_identity
		print(f'model: {plan.model}')
		print(f'checkpoint: {job.model.checkpoint}')
		print(f'embeddings_dir: {job.model.embeddings_dir}')
		print(f'layout_id: {plan.layout_id}')
		print(f'data_size: {plan.data_size}')
		print(
			'selected_physical_lines: '
			f'{plan.split_plan.identity()["selected_physical_lines"]}'
		)
		print(f'split_plan_sha256: {plan.split_plan.scientific_identity_sha256}')
		print(
			'decoder_initial_state_sha256: '
			f'{identity["decoder"]["initial_state_sha256"]}'
		)
		print(
			'effective_model_valid_observation_counts: '
			f'{dict(plan.effective_per_horizon_counts)}'
		)
		print(
			'tile_counts: '
			f'{ {key: len(value) for key, value in plan.tile_records.items()} }'
		)
		print(f'output_dir: {plan.output_dir}')
		print('execution: dry-run; no files written')
		return
	result = run_volve_horizon_five_way_job(
		plan,
		device=args.device,
		max_steps=args.max_steps,
		resume=args.resume,
	)
	if result is None:
		print(f'latest: {plan.output_dir / "latest.pt"}')
		print('execution: stopped at max-steps; resume from latest.pt')
	else:
		print(f'metrics: {result}')


if __name__ == '__main__':
	main()
