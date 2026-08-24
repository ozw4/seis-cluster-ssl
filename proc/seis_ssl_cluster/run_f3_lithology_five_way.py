"""Run one (model, layout, size) job of the F3 lithology five-way benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_MODEL_IDS,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	inspect_f3_lithology_five_way_job,
	resolve_f3_lithology_five_way_job,
	run_f3_lithology_five_way_job,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for one five-way benchmark job."""
	parser = build_config_parser(
		'Run one F3 lithology five-way decoder job.',
		config_help='Path to the canonical five-way comparison YAML.',
		dry_run_help=(
			'Print the resolved job identity and outputs without writing.'
		),
	)
	parser.add_argument(
		'--model',
		required=True,
		choices=FIVE_WAY_MODEL_IDS,
		help='Model ID of the frozen encoder to evaluate.',
	)
	parser.add_argument(
		'--layout',
		required=True,
		choices=LAYOUT_IDS,
		help='Section-layout statistical unit.',
	)
	parser.add_argument(
		'--size',
		required=True,
		choices=DATA_SIZES,
		help='Nested teacher-line data size.',
	)
	parser.add_argument(
		'--resume',
		type=Path,
		default=None,
		help='Resume decoder training from this latest.pt checkpoint.',
	)
	return parser


def main() -> None:
	"""Run one job or print its dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_five_way_config_from_mapping,
		config_path=config_path,
	)
	job = resolve_f3_lithology_five_way_job(
		config,
		model=args.model,
		layout=args.layout,
		size=args.size,
	)
	if args.resume is not None and not args.resume.is_file():
		raise FileNotFoundError(f'resume checkpoint does not exist: {args.resume}')
	if args.dry_run:
		summary = inspect_f3_lithology_five_way_job(job)
		for key, value in summary.items():
			print(f'{key}: {value}')
		print('execution: dry-run; no files written')
		return

	result = run_f3_lithology_five_way_job(job, resume=args.resume)
	for key, value in result.items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
