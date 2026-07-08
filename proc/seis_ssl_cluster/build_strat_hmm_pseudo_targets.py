"""Thin entrypoint for strat HMM pseudo-target refresh."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	add_device_argument,
	add_overwrite_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import (
	load_config,
	resolve_strat_hmm_pseudo_target_config,
)
from seis_ssl_cluster.stratigraphy.pseudo_target_builder import (
	build_strat_hmm_pseudo_targets,
)
from seis_ssl_cluster.utils.cli import print_config_summary

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for strat HMM pseudo-target refresh."""
	parser = build_config_parser(
		'Refresh strat HMM pseudo-targets from trained prototype logits.',
		config_required=True,
	)
	add_device_argument(parser, help_text='Inference device override.')
	add_overwrite_argument(parser, help_text='Replace existing pseudo-target output.')
	return parser


def main() -> None:
	"""Validate config, dry-run, or run pseudo-target refresh."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = dict(load_config_for_cli(config_path, loader=load_config))
	_apply_cli_overrides(
		raw_config,
		device=args.device,
		overwrite=True if args.overwrite else None,
	)
	config = resolve_config_for_cli(
		raw_config,
		resolver=resolve_strat_hmm_pseudo_target_config,
		config_path=config_path,
	)
	if args.dry_run:
		print_config_summary(config)
		print('execution: dry-run; pseudo-target refresh skipped')
		return

	outputs = build_strat_hmm_pseudo_targets(
		config,
		device=args.device,
		overwrite=True if args.overwrite else None,
	)
	for output in outputs:
		print(f'pseudo_target: {output}')


def _apply_cli_overrides(
	config: dict[str, object],
	*,
	device: str | None,
	overwrite: bool | None,
) -> None:
	if device is not None:
		_section(config, 'inference')['device'] = device
	if overwrite is not None:
		_section(config, 'outputs')['overwrite'] = overwrite


def _section(config: dict[str, object], key: str) -> dict[str, object]:
	value = config[key]
	if not isinstance(value, dict):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


if __name__ == '__main__':
	main()
