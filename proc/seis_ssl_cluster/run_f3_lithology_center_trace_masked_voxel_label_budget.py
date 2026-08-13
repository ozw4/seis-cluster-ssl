"""Run the closed F3 center-trace masked original-split screen."""

from __future__ import annotations

from seis_ssl_cluster.cli import (
	add_device_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_center_trace_masked as center_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.voxel_label_budget_center_trace_masked import (
	inspect_f3_lithology_voxel_label_budget_center_trace_masked,
	run_f3_lithology_voxel_label_budget_center_trace_masked,
)


def build_parser() -> object:
	"""Build the closed center-trace masked runner parser."""
	parser = build_config_parser(
		'Run center-trace masked original-split low-label voxel decoders.',
		config_required=True,
		dry_run_help=(
			'Live-validate and print the 15-job center-trace plan without writing.'
		),
	)
	add_device_argument(parser, help_text='Training and inference device override.')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--resume', action='store_true')
	return parser


def main() -> None:
	"""Inspect or execute only the frozen 15-job candidate matrix."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	config = resolve_config_for_cli(
		load_config_for_cli(config_path, loader=load_config),
		resolver=(center_config.config_from_mapping),
		config_path=config_path,
	)
	if args.dry_run and args.resume:
		raise ValueError('--dry-run and --resume are mutually exclusive')
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_label_budget_center_trace_masked(config)
		print('stage: run_f3_lithology_center_trace_masked_voxel_label_budget')
		print(f'job_count: {len(inspection.jobs)}')
		for plan in inspection.plans:
			label = f'{plan.job.budget_id}/seed={plan.job.subsample_seed}'
			print(f'{label}: action={plan.state}')
		print(f'estimated_additional_storage_bytes: {inspection.estimated_new_bytes}')
		print(f'disk_free_bytes: {inspection.disk_free_bytes}')
		print('execution: dry-run')
		return
	result = run_f3_lithology_voxel_label_budget_center_trace_masked(
		config,
		only_missing=args.only_missing,
		resume=args.resume,
		device=args.device or 'auto',
	)
	print(f'center_trace_masked_job_manifest: {result.manifest_json}')


if __name__ == '__main__':
	main()
