"""Run the paired two-candidate multi-head low-label voxel matrix."""

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
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_multi_head import (
	inspect_f3_lithology_voxel_label_budget_multi_head,
	run_f3_lithology_voxel_label_budget_multi_head,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the multi-candidate runner CLI."""
	parser = build_config_parser(
		'Run the multi-head low-label voxel decoder matrix.',
		config_required=True,
		dry_run_help='Validate references and print the 30-job action plan.',
	)
	add_device_argument(parser, help_text='Training and inference device override.')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--candidate', help='Restrict execution to one candidate ID.')
	parser.add_argument('--budget', help='Restrict execution to one budget ID.')
	parser.add_argument(
		'--subsample-seed', type=int, help='Restrict execution to one seed.'
	)
	return parser


def main() -> None:
	"""Inspect or run selected multi-head candidate jobs."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	config = resolve_config_for_cli(
		load_config_for_cli(config_path, loader=load_config),
		resolver=f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run and args.resume:
		raise ValueError('--dry-run and --resume are mutually exclusive')
	inspection = inspect_f3_lithology_voxel_label_budget_multi_head(
		config,
		candidate=args.candidate,
		budget=args.budget,
		subsample_seed=args.subsample_seed,
	)
	if args.dry_run:
		print('stage: run_f3_lithology_multi_head_voxel_label_budget')
		print(f'job_count: {len(inspection.jobs)}')
		print(f'estimated_new_bytes: {inspection.estimated_new_bytes}')
		print(f'disk_free_bytes: {inspection.disk_free_bytes}')
		for plan in inspection.plans:
			job = plan.job
			print(f'{job.model_role}/{job.budget_id}/seed={job.subsample_seed}:')
			print(f'  action={plan.state}; estimated_bytes={plan.estimated_bytes}')
			print(f'  output={job.output_root}')
		print('execution: dry-run')
		return
	result = run_f3_lithology_voxel_label_budget_multi_head(
		config,
		only_missing=args.only_missing,
		resume=args.resume,
		device=args.device or 'auto',
		candidate=args.candidate,
		budget=args.budget,
		subsample_seed=args.subsample_seed,
	)
	print(f'multi_head_job_manifest: {result.manifest_json}')
	complete = sum(row.get('status') == 'complete' for row in result.rows)
	print(f'multi_head_jobs.complete: {complete}/{len(result.rows)}')


if __name__ == '__main__':
	main()
