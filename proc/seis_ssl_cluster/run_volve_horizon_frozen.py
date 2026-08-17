'''Run one frozen Volve five-horizon decoder condition.'''

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_frozen import (
	frozen_horizon_config_from_mapping,
	inspect_frozen_horizon_job,
	run_frozen_horizon_job,
)
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments/volve/horizon_benchmark_v1/30_mae_vs_random_frozen_v1'
	/ '03_horizon_frozen.yaml'
)
DEFAULT_LAYOUT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments/volve/horizon_benchmark_v1/20_horizon_supervision'
	/ '01_layouts.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	'''Build the one-job frozen benchmark parser.'''
	parser = argparse.ArgumentParser(
		description='Run one frozen Volve horizon decoder job.'
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument('--model', required=True, choices=('pretrained', 'random'))
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
	'''Inspect or execute exactly one paired benchmark condition.'''
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = frozen_horizon_config_from_mapping(raw)
	plan = inspect_frozen_horizon_job(
		config,
		model=args.model,
		layout_id=args.layout,
		data_size=args.size,
		layout_config=args.layout_config,
	)
	if args.dry_run:
		identity = plan.run_identity
		print(f'model: {plan.model}')
		print(f'layout_id: {plan.layout_id}')
		print(f'data_size: {plan.data_size}')
		print(
			'selected_physical_lines: '
			f'{plan.split_plan.identity()["selected_physical_lines"]}'
		)
		print(
			'selected_indices: '
			f'{plan.split_plan.identity()["selected_indices"]}'
		)
		print(
			f'split_plan_sha256: '
			f'{plan.split_plan.scientific_identity_sha256}'
		)
		print(
			f'decoder_initial_state_sha256: '
			f'{identity["decoder"]["initial_state_sha256"]}'
		)
		print(f'per_horizon_counts: {dict(plan.per_horizon_counts)}')
		print(
			'tile_counts: '
			f'{ {key: len(value) for key, value in plan.tile_records.items()} }'
		)
		print(f'output_dir: {plan.output_dir}')
		print('execution: dry-run; no files written')
		return
	result = run_frozen_horizon_job(
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
