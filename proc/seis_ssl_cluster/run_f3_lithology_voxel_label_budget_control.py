"""Run only the current-code K=6 member of the paired voxel budget matrix."""

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
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_control import (
	f3_lithology_voxel_label_budget_control_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_control import (
	inspect_f3_lithology_voxel_label_budget_control,
	run_f3_lithology_voxel_label_budget_control,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the candidate-only control runner CLI."""
	parser = build_config_parser(
		'Run the current-code K=6 low-label voxel control matrix.',
		config_required=True,
		dry_run_help=(
			'Validate immutable MAE/M1 references and print the 15-job action plan.'
		),
	)
	add_device_argument(parser, help_text='Training and inference device override.')
	parser.add_argument(
		'--only-missing',
		action='store_true',
		help=(
			'Reuse complete jobs, resume valid latest.pt jobs, and quarantine '
			'mismatches.'
		),
	)
	parser.add_argument(
		'--resume',
		action='store_true',
		help='Resume only selected jobs with a valid incomplete latest.pt checkpoint.',
	)
	parser.add_argument('--budget', help='Restrict execution to one budget ID.')
	parser.add_argument('--subsample-seed', type=int, help='Restrict to one seed.')
	return parser


def main() -> None:
	"""Inspect or run the current-code K=6 candidate jobs."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	config = resolve_config_for_cli(
		load_config_for_cli(config_path, loader=load_config),
		resolver=f3_lithology_voxel_label_budget_control_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run and args.resume:
		raise ValueError('--dry-run and --resume are mutually exclusive')
	inspection = inspect_f3_lithology_voxel_label_budget_control(
		config,
		budget=args.budget,
		subsample_seed=args.subsample_seed,
	)
	if args.dry_run:
		print('stage: run_f3_lithology_voxel_label_budget_control')
		print(f'job_count: {len(inspection.jobs)}')
		print(f'estimated_new_bytes: {inspection.estimated_new_bytes}')
		print(f'disk_free_bytes: {inspection.disk_free_bytes}')
		print('historical_reference: MAE/M1 45-job manifest validated')
		print('candidate_valid_tokens: exact to historical MAE/M1')
		for plan in inspection.plans:
			job = plan.job
			print(
				f'{job.budget_id}/seed={job.subsample_seed}/'
				f'model={job.model_role}: {plan.state}'
				+ ('' if plan.reason is None else f' ({plan.reason})')
			)
		print('execution: dry-run')
		return
	result = run_f3_lithology_voxel_label_budget_control(
		config,
		only_missing=args.only_missing,
		resume=args.resume,
		device=args.device or 'auto',
		budget=args.budget,
		subsample_seed=args.subsample_seed,
	)
	complete = sum(row.get('status') == 'complete' for row in result.rows)
	print(f'control_job_manifest: {result.manifest_json}')
	print(f'current_k6_control_jobs.complete: {complete}/{len(result.rows)}')


if __name__ == '__main__':
	main()
