"""Run the closed periodic-refresh original-split decoder screen."""
# ruff: noqa: CPY001, E501

from __future__ import annotations

from seis_ssl_cluster.cli import (
	add_device_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh as periodic_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked_periodic_refresh as periodic_runner,
)


def build_parser() -> object:
	"""Build the isolated periodic-refresh runner parser."""
	parser = build_config_parser(
		'Run periodic-refresh original-split low-label voxel decoders.',
		config_required=True,
		dry_run_help=(
			'Live-validate and print the 15-job periodic-refresh plan without writing.'
		),
	)
	add_device_argument(parser, help_text='Training and inference device override.')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--resume', action='store_true')
	return parser


def main() -> None:
	"""Inspect or execute only the 15 owned periodic-refresh jobs."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	config = resolve_config_for_cli(
		load_config_for_cli(config_path, loader=load_config),
		resolver=periodic_config.config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run and args.resume:
		raise ValueError('--dry-run and --resume are mutually exclusive')
	if args.dry_run:
		inspection = periodic_runner.inspect_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh(
			config
		)
		print(
			'stage: '
			'run_f3_lithology_center_trace_masked_periodic_refresh_voxel_label_budget'
		)
		print(f'job_count: {len(inspection.jobs)}')
		for plan in inspection.plans:
			label = f'{plan.job.budget_id}/seed={plan.job.subsample_seed}'
			print(f'{label}: action={plan.state}')
		print(f'estimated_additional_storage_bytes: {inspection.estimated_new_bytes}')
		print(f'disk_free_bytes: {inspection.disk_free_bytes}')
		print('execution: dry-run')
		return
	result = periodic_runner.run_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh(
		config,
		only_missing=args.only_missing,
		resume=args.resume,
		device=args.device or 'auto',
	)
	print(f'periodic_refresh_original_job_manifest: {result.manifest_json}')


if __name__ == '__main__':
	main()
