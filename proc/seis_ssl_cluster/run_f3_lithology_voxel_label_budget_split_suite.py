"""Plan or run the M4 selected multi-head six-split decoder matrix."""
# ruff: noqa: D103, E501

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	add_device_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_split import (
	f3_lithology_voxel_label_budget_split_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split_runner import (
	inspect_f3_lithology_voxel_label_budget_split_suite,
	run_f3_lithology_voxel_label_budget_split_suite,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the strict config and six-split runner parser."""
	parser = build_config_parser(
		'Run M4 selected multi-head six-split low-label decoders.',
		config_required=True,
		dry_run_help='Validate provenance and print the exact 36-job matrix.',
	)
	add_device_argument(parser, help_text='Training device override.')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--smoke-only', action='store_true')
	parser.add_argument('--split-id')
	parser.add_argument('--budget')
	parser.add_argument('--model-role', choices=('mae', 'm1_current_k6', 'mh_nocons'))
	return parser


def main() -> None:
	args = build_parser().parse_args()
	if args.only_missing and args.resume:
		raise ValueError('--only-missing and --resume are mutually exclusive')
	path = parse_config_path(args)
	config = resolve_config_for_cli(load_config_for_cli(path, loader=load_config), resolver=f3_lithology_voxel_label_budget_split_config_from_mapping, config_path=path)
	jobs = inspect_f3_lithology_voxel_label_budget_split_suite(config)
	jobs = tuple(job for job in jobs if (args.split_id is None or job.split_id == args.split_id) and (args.budget is None or job.budget_id == args.budget) and (args.model_role is None or job.model_role == args.model_role))
	if not jobs:
		raise ValueError('filters selected no jobs')
	if args.dry_run:
		print(f'job_count: {len(jobs)}')
		for job in jobs:
			print(f'{job.split_id}/{job.budget_id}/{job.model_role}: PLANNED')
		return
	if args.smoke_only and (args.split_id not in {None, 'split_000'} or args.budget not in {None, 'cap25'} or args.model_role is not None):
		raise ValueError('--smoke-only is fixed to split_000/cap25 and all three models')
	rows = run_f3_lithology_voxel_label_budget_split_suite(
		config,
		only_missing=args.only_missing,
		resume=args.resume,
		device=args.device or 'auto',
		split_id='split_000' if args.smoke_only else args.split_id,
		budget='cap25' if args.smoke_only else args.budget,
		model_role=args.model_role,
		smoke_only=args.smoke_only,
	)
	print(f'low_label_split_jobs.complete: {len(rows)}/{len(jobs)}')


if __name__ == '__main__':
	main()
