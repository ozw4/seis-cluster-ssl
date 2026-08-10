"""Inspect F3 section candidates or finalize a canonical layout contract."""
# ruff: noqa: CPY001

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
from seis_ssl_cluster.f3.lithology.voxel_section_layout_calibration import (
	F3SectionLayoutCalibrationConfig,
	f3_section_layout_calibration_config_from_mapping,
	run_section_layout_calibration,
)

if TYPE_CHECKING:
	import argparse

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '109_f3_voxel_section_layout_v1'
	/ '01_prepare_section_layout_contract.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the closed two-mode calibration parser."""
	parser = build_config_parser(
		'Inspect F3 section candidates or finalize a section-layout contract.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate and preview the selected mode without writing outputs.',
	)
	parser.add_argument(
		'--mode',
		choices=('inspect', 'finalize'),
		required=True,
		help=(
			'Write candidate statistics (inspect) or validate and write the '
			'contract (finalize).'
		),
	)
	return parser


def main() -> None:
	"""Run deterministic calibration in exactly one of the two supported modes."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_section_layout_calibration_config_from_mapping,
		config_path=config_path,
	)
	run_section_layout_calibration(
		config,
		mode=str(args.mode),
		dry_run=bool(args.dry_run),
	)
	_print_summary(config, mode=str(args.mode), dry_run=bool(args.dry_run))


def _print_summary(
	config: F3SectionLayoutCalibrationConfig,
	*,
	mode: str,
	dry_run: bool,
) -> None:
	print(f'section_layout.mode: {mode}')
	print(f'section_layout.selection_semantics: {config.selection_semantics}')
	print(f'section_layout.dry_run: {str(dry_run).lower()}')
	if mode == 'inspect':
		print(f'section_layout.candidate_csv: {config.candidate_statistics_csv}')
		print(f'section_layout.candidate_json: {config.candidate_statistics_json}')
	else:
		print(f'section_layout.contract: {config.canonical_contract}')
	if dry_run:
		print('section_layout.execution: dry-run; no artifacts written')


if __name__ == '__main__':
	main()
