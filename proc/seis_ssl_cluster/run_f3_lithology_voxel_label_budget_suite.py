"""Run the original-split F3 low-label voxel decoder suite."""

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
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_suite import (
	f3_lithology_voxel_label_budget_suite_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	inspect_f3_lithology_voxel_label_budget_suite,
	run_f3_lithology_voxel_label_budget_smoke,
	run_f3_lithology_voxel_label_budget_suite,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the suite runner CLI."""
	parser = build_config_parser(
		'Run the F3 original-split low-label voxel decoder suite.',
		config_required=True,
		dry_run_help='Validate all identities and print the 45-job action plan.',
	)
	add_device_argument(parser, help_text='Training and inference device override.')
	parser.add_argument(
		'--only-missing',
		action='store_true',
		help='Strictly reuse completed jobs and resume valid latest.pt checkpoints.',
	)
	parser.add_argument(
		'--smoke-only',
		action='store_true',
		help='Run the non-scientific two-step three-model smoke gate.',
	)
	parser.add_argument('--budget', help='Restrict execution to one budget ID.')
	parser.add_argument('--subsample-seed', type=int, help='Restrict to one seed.')
	parser.add_argument(
		'--model', choices=('mae', 'm1', 'm2a'), help='Restrict to one model role.'
	)
	return parser


def main() -> None:
	"""Inspect, smoke, or run the configured suite."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	config = resolve_config_for_cli(
		load_config_for_cli(config_path, loader=load_config),
		resolver=f3_lithology_voxel_label_budget_suite_config_from_mapping,
		config_path=config_path,
	)
	device = args.device or 'auto'
	if args.dry_run and args.smoke_only:
		raise ValueError('--dry-run and --smoke-only are mutually exclusive')
	if args.smoke_only:
		if args.model is not None:
			raise ValueError(
				'--smoke-only always runs the complete three-model triplet'
			)
		rows = run_f3_lithology_voxel_label_budget_smoke(
			config,
			budget=args.budget or 'cap25',
			subsample_seed=(0 if args.subsample_seed is None else args.subsample_seed),
			device=device,
		)
		print(f'voxel_label_budget_smoke.complete: {len(rows)}/3')
		return
	inspection = inspect_f3_lithology_voxel_label_budget_suite(
		config,
		budget=args.budget,
		subsample_seed=args.subsample_seed,
		model=args.model,
	)
	if args.dry_run:
		print('stage: run_f3_lithology_voxel_label_budget_suite')
		print(f'canonical_steps_per_epoch: {inspection.canonical_steps_per_epoch}')
		print(f'job_count: {len(inspection.jobs)}')
		print(f'estimated_new_bytes: {inspection.estimated_new_bytes}')
		print(f'disk_free_bytes: {inspection.disk_free_bytes}')
		for plan in inspection.plans:
			job = plan.job
			print(
				f'{job.budget_id}/seed={job.subsample_seed}/model={job.model_role}: '
				f'{plan.state}' + ('' if plan.reason is None else f' ({plan.reason})')
			)
		print('execution: dry-run')
		return
	result = run_f3_lithology_voxel_label_budget_suite(
		config,
		only_missing=args.only_missing,
		device=device,
		budget=args.budget,
		subsample_seed=args.subsample_seed,
		model=args.model,
	)
	complete = sum(row.get('status') == 'complete' for row in result.rows)
	print(f'voxel_label_budget_run_manifest: {result.manifest_json}')
	print(f'voxel_label_budget_jobs.complete: {complete}/{len(result.rows)}')


if __name__ == '__main__':
	main()
