"""Audit or summarize Local VICReg F3 benchmark suites."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.vicreg_benchmark import (
	audit_f3_vicreg_screening_source,
	audit_f3_vicreg_sources,
	f3_vicreg_extension_config_from_mapping,
	inspect_f3_vicreg_combined_results,
	inspect_f3_vicreg_extension_results,
	inspect_f3_vicreg_screening_results,
	load_f3_vicreg_canonical_config,
	plan_f3_vicreg_extension_jobs,
	plan_f3_vicreg_screening_jobs,
	summarize_f3_vicreg_combined,
	summarize_f3_vicreg_extension,
	summarize_f3_vicreg_screening,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the source/screen/extension/combined summary parser."""
	parser = build_config_parser(
		'Audit or summarize the F3 Local VICReg benchmark extension.',
		config_help='Path to the VICReg extension YAML.',
		dry_run_help='Audit the selected mode without writing summaries.',
	)
	parser.add_argument(
		'--mode',
		required=True,
		choices=(
			'screening-source',
			'sources',
			'screening',
			'extension',
			'combined',
		),
		help='Select source audit or one summary scope.',
	)
	return parser


def main() -> None:
	"""Audit sources/jobs or atomically publish the selected summary."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_vicreg_extension_config_from_mapping,
		config_path=config_path,
	)
	canonical = load_f3_vicreg_canonical_config(config)
	if args.mode in {'screening-source', 'sources'}:
		if args.dry_run:
			print(f'screening_jobs: {len(plan_f3_vicreg_screening_jobs())}')
			if args.mode == 'sources':
				print(f'extension_jobs: {len(plan_f3_vicreg_extension_jobs())}')
			print('execution: dry-run; live source audit skipped')
			return
		result = (
			audit_f3_vicreg_screening_source(config, canonical)
			if args.mode == 'screening-source'
			else audit_f3_vicreg_sources(config, canonical)
		)
		print(f'model_order: {", ".join(result["model_order"])}')
		print('VICReg source audit passed')
		return
	inspectors = {
		'screening': inspect_f3_vicreg_screening_results,
		'extension': inspect_f3_vicreg_extension_results,
		'combined': inspect_f3_vicreg_combined_results,
	}
	summarizers = {
		'screening': summarize_f3_vicreg_screening,
		'extension': summarize_f3_vicreg_extension,
		'combined': summarize_f3_vicreg_combined,
	}
	function = inspectors[args.mode] if args.dry_run else summarizers[args.mode]
	result = function(config, canonical)
	for key, value in result.items():
		print(f'{key}: {value}')
	if args.dry_run:
		print('execution: dry-run; summary files skipped')


if __name__ == '__main__':
	main()
