"""Thin entrypoint for amplitude-only local VICReg pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	add_device_argument,
	add_path_argument,
	build_config_parser,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config, resolve_vicreg_training_config
from seis_ssl_cluster.training.vicreg import run_vicreg_pretraining
from seis_ssl_cluster.utils.cli import print_config_summary

if TYPE_CHECKING:
	import argparse

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'train_amp_vicreg.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for local VICReg pretraining."""
	parser = build_config_parser(
		'Train an amplitude-only local 3D VICReg model.',
		default_config=DEFAULT_CONFIG,
	)
	add_device_argument(parser, help_text='Training device override.')
	parser.add_argument(
		'--max-steps',
		type=int,
		help='Stop after N optimizer steps for smoke runs.',
	)
	add_path_argument(
		parser,
		'--output-root',
		help_text='Override the absolute checkpoint and run output path.',
	)
	add_path_argument(
		parser,
		'--resume',
		help_text='Resume from a completed-epoch VICReg checkpoint.',
	)
	return parser


def main() -> None:
	"""Run VICReg pretraining or print a dry-run summary."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	_apply_cli_overrides(
		raw_config,
		device=args.device,
		max_steps=args.max_steps,
		output_root=args.output_root,
	)
	config = resolve_config_for_cli(
		raw_config,
		resolver=resolve_vicreg_training_config,
		config_path=config_path,
	)
	if args.resume is not None and not args.resume.is_file():
		raise FileNotFoundError(f'resume checkpoint does not exist: {args.resume}')
	if args.dry_run:
		print_config_summary(config)
		if args.resume is not None:
			print(f'resume: {args.resume}')
		print('execution: dry-run; training skipped')
		return
	checkpoint_path = run_vicreg_pretraining(config, resume=args.resume)
	print(f'checkpoint: {checkpoint_path}')


def _apply_cli_overrides(
	config: dict[str, object],
	*,
	device: str | None,
	max_steps: int | None,
	output_root: Path | None,
) -> None:
	if device is not None or max_steps is not None:
		train = _section(config, 'train')
	if output_root is not None:
		paths = _section(config, 'paths')
	if device is not None:
		train['device'] = device
	if max_steps is not None:
		train['max_steps'] = max_steps
	if output_root is not None:
		paths['output_root'] = str(output_root)


def _section(config: dict[str, object], key: str) -> dict[str, object]:
	value = config[key]
	if not isinstance(value, dict):
		raise TypeError(f'{key} must be a mapping')
	return value


if __name__ == '__main__':
	main()
