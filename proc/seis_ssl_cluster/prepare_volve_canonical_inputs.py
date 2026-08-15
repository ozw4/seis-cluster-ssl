'''Thin entrypoint for registering read-only Volve canonical inputs.'''

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import argparse

from seis_ssl_cluster.cli import (
	add_store_true_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve import (
	VolveCanonicalInputConfig,
	prepare_volve_canonical_inputs,
	resolve_volve_canonical_input_config,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'prepare_volve_canonical_inputs.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	'''Build the Volve canonical input registration parser.'''
	parser = build_config_parser(
		'Validate and register the read-only Volve canonical amplitude volume.',
		default_config=DEFAULT_CONFIG,
		dry_run_help='Validate inputs and print the plan without writing outputs.',
	)
	add_store_true_argument(
		parser,
		'--only-missing',
		help_text='Reuse a complete output set only when its identity matches.',
	)
	return parser


def main() -> None:
	'''Validate and register Volve canonical inputs.'''
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=resolve_volve_canonical_input_config,
		config_path=config_path,
	)
	result = prepare_volve_canonical_inputs(
		config,
		dry_run=args.dry_run,
		only_missing=args.only_missing,
	)
	_print_summary(config, result.action, result.scientific_identity_sha256)


def _print_summary(
	config: VolveCanonicalInputConfig,
	action: str,
	identity_sha256: str,
) -> None:
	paths = config.paths
	print(f'volve_inputs.canonical_root: {paths.canonical_root}')
	print(f'volve_inputs.output_dir: {paths.output_dir}')
	print(f'volve_inputs.manifest: {paths.manifest_path}')
	print(f'volve_inputs.npy_paths: {paths.path_list_path}')
	print(f'volve_inputs.normalization_stats: {paths.normalization_stats_path}')
	print(f'volve_inputs.metadata: {paths.metadata_path}')
	print(f'volve_inputs.scientific_identity_sha256: {identity_sha256}')
	normalized_action = action.lower().replace('_', '-')
	print(f'volve_inputs.execution: {normalized_action}')


if __name__ == '__main__':
	main()
