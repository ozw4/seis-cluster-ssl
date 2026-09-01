"""Run one F3 lithology candidate decoder job."""

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
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	inspect_f3_lithology_candidate_job,
	load_f3_lithology_candidate_canonical_config,
	resolve_f3_lithology_candidate_job,
	run_f3_lithology_candidate_job,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for one candidate cell."""
	parser = build_config_parser(
		'Run one F3 lithology candidate decoder job.',
		config_help='Path to the minimal candidate YAML.',
		dry_run_help=(
			'Validate the candidate source and print the resolved job without writing.'
		),
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
	"""Run one candidate job or print its source-audited dry-run plan."""
	parser = build_parser()
	args = parser.parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_candidate_config_from_mapping,
		config_path=config_path,
	)
	canonical_config = load_f3_lithology_candidate_canonical_config(config)
	job = resolve_f3_lithology_candidate_job(
		config, canonical_config, layout=args.layout, size=args.size
	)
	if args.resume is not None and not args.resume.is_file():
		raise FileNotFoundError(f'resume checkpoint does not exist: {args.resume}')
	if args.dry_run:
		summary = inspect_f3_lithology_candidate_job(config, canonical_config, job)
		for key, value in summary.items():
			print(f'{key}: {value}')
		print('execution: dry-run; no files written')
		return
	result = run_f3_lithology_candidate_job(
		config, canonical_config, job, resume=args.resume
	)
	for key, value in result.items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
