"""Run the closed F3 XY-neighbour-unanimous original-split screen."""

from __future__ import annotations

from seis_ssl_cluster.cli import (
	add_device_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_xy_neighbor_unanimous as unanimous_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.voxel_label_budget_xy_neighbor_unanimous import (
	inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous,
	run_f3_lithology_voxel_label_budget_xy_neighbor_unanimous,
)


def build_parser() -> object:
	"""Build the closed unanimous-candidate runner parser."""
	parser = build_config_parser(
		'Run XY-neighbour-unanimous original-split low-label voxel decoders.',
		config_required=True,
		dry_run_help=(
			'Live-validate and print the 15-job unanimous plan without writing.'
		),
	)
	add_device_argument(parser, help_text='Training and inference device override.')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--resume', action='store_true')
	return parser


def main() -> None:
	"""Inspect or execute only the frozen schema-v6 candidate matrix."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	config = resolve_config_for_cli(
		load_config_for_cli(config_path, loader=load_config),
		resolver=(
			unanimous_config.f3_lithology_voxel_label_budget_xy_neighbor_unanimous_config_from_mapping
		),
		config_path=config_path,
	)
	inspection = inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous(
		config
	)
	if args.dry_run:
		print('stage: run_f3_lithology_xy_neighbor_unanimous_voxel_label_budget')
		print(f'job_count: {len(inspection.jobs)}')
		for plan in inspection.plans:
			label = f'{plan.job.budget_id}/seed={plan.job.subsample_seed}'
			print(f'{label}: action={plan.state}')
		print(
			'estimated_additional_storage_bytes: '
			f'{inspection.estimated_new_bytes}'
		)
		print(f'disk_free_bytes: {inspection.disk_free_bytes}')
		print('execution: dry-run')
		return
	result = run_f3_lithology_voxel_label_budget_xy_neighbor_unanimous(
		config,
		only_missing=args.only_missing,
		resume=args.resume,
		device=args.device or 'auto',
	)
	print(f'xy_neighbor_unanimous_job_manifest: {result.manifest_json}')


if __name__ == '__main__':
	main()
