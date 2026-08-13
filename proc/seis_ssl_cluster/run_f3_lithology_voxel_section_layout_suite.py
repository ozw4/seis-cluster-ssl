"""Run one roster model through the F3 section-layout benchmark."""

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
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_benchmark import (
	f3_lithology_voxel_section_layout_benchmark_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_runner import (
	inspect_f3_lithology_voxel_section_layout_suite,
	run_f3_lithology_voxel_section_layout_suite,
)

if TYPE_CHECKING:
	import argparse


DEFAULT_CONFIG = (
	'experiments/f3/facies_benchmark_v1/109_f3_voxel_section_layout_v1/'
	'04_run_section_layout_benchmark.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the explicit one-model suite CLI."""
	parser = build_config_parser(
		'Run one F3 section-layout roster model.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate inputs and print the selected job plan without writes.',
	)
	add_device_argument(parser, help_text='Training and inference device override.')
	parser.add_argument(
		'--model-id', required=True, help='One exact ID from the closed roster.'
	)
	parser.add_argument('--layout-id', choices=LAYOUT_IDS)
	parser.add_argument('--data-size', choices=DATA_SIZES)
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	parser.add_argument('--smoke-only', action='store_true')
	return parser


def main() -> None:
	"""Inspect or execute the selected single-model condition matrix."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	config = resolve_config_for_cli(
		load_config_for_cli(config_path, loader=load_config),
		resolver=f3_lithology_voxel_section_layout_benchmark_config_from_mapping,
		config_path=config_path,
	)
	if args.dry_run:
		inspection = inspect_f3_lithology_voxel_section_layout_suite(
			config,
			model_id=args.model_id,
			layout_id=args.layout_id,
			data_size=args.data_size,
			smoke_only=args.smoke_only,
		)
		print('stage: run_f3_lithology_voxel_section_layout_suite')
		print(f'model_id: {inspection.model.model_id}')
		print(f'job_count: {len(inspection.jobs)}')
		for plan in inspection.plans:
			print(
				f'{plan.job.layout_id}/size={plan.job.data_size}: {plan.state}'
				+ ('' if plan.reason is None else f' ({plan.reason})')
			)
		print(
			'execution: dry-run '
			'(zero writes, training, inference, evaluation, quarantine)'
		)
		return
	result = run_f3_lithology_voxel_section_layout_suite(
		config,
		model_id=args.model_id,
		layout_id=args.layout_id,
		data_size=args.data_size,
		only_missing=args.only_missing,
		resume=args.resume,
		quarantine_invalid=args.quarantine_invalid,
		smoke_only=args.smoke_only,
		device=args.device or 'auto',
	)
	print(f'section_layout_run_manifest: {result.manifest_json}')
	complete = sum(row.get('status') == 'complete' for row in result.rows)
	print(f'jobs.complete: {complete}/{len(result.rows)}')


if __name__ == '__main__':
	main()
