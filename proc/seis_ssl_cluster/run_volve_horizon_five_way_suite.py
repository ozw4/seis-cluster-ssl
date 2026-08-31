'''Run the complete Volve horizon five-way comparison in one process.'''

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_five_way_config import (
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_runner import (
	VolveHorizonFiveWayJob,
	VolveHorizonFiveWaySuiteCellResult,
	plan_volve_horizon_five_way_jobs,
	resolve_volve_horizon_five_way_job,
	run_volve_horizon_five_way_suite,
)

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
	'''Build the sequential five-way suite CLI parser.'''
	parser = argparse.ArgumentParser(
		description='Run all 75 frozen Volve horizon five-way decoder jobs.'
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument(
		'--layout-config', type=Path, default=DEFAULT_LAYOUT_CONFIG
	)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
	parser.add_argument('--max-steps', type=int)
	parser.add_argument(
		'--continue',
		dest='continue_existing',
		action='store_true',
		help=(
			'Skip cells with metrics.json and resume incomplete cells from '
			'their own latest.pt.'
		),
	)
	return parser


def main() -> None:
	'''Plan or execute the suite with one shared artifact preflight.'''
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = volve_horizon_five_way_config_from_mapping(raw)
	if args.dry_run:
		for model, layout, size in plan_volve_horizon_five_way_jobs(config):
			job = resolve_volve_horizon_five_way_job(
				config,
				model=model,
				layout=layout,
				size=size,
			)
			action = _planned_action(
				job,
				continue_existing=args.continue_existing,
			)
			print(
				f'model={model} layout={layout} size={size} '
				f'action={action} '
				f'output_dir={job.output_dir}'
			)
		print('execution: dry-run; no artifact preflight or files written')
		return
	run_volve_horizon_five_way_suite(
		config,
		layout_config=args.layout_config,
		device=args.device,
		max_steps=args.max_steps,
		continue_existing=args.continue_existing,
		progress=_print_result,
	)


def _planned_action(
	job: VolveHorizonFiveWayJob,
	*,
	continue_existing: bool,
) -> str:
	if not continue_existing:
		return 'fresh'
	if job.metrics_path.is_file():
		return 'skip'
	if job.latest_path.is_file():
		return 'resume'
	return 'fresh'


def _print_result(cell: VolveHorizonFiveWaySuiteCellResult) -> None:
	job = cell.job
	if cell.action == 'skip':
		status = f'skipped metrics={job.metrics_path}'
	elif cell.result is None:
		status = f'stopped latest={job.latest_path}'
	else:
		status = f'complete metrics={cell.result}'
	print(
		f'model={job.model.model_id} layout={job.layout_id} '
		f'size={job.data_size} action={cell.action} status={status}',
		flush=True,
	)


if __name__ == '__main__':
	main()
