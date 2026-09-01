"""Run one Local VICReg screening or extension decoder job."""

from __future__ import annotations

from pathlib import Path
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
from seis_ssl_cluster.f3.lithology.vicreg_benchmark import (
	EXTENSION_MODEL_IDS,
	SCREENING_MODEL_IDS,
	f3_vicreg_extension_config_from_mapping,
	inspect_f3_vicreg_job,
	load_f3_vicreg_canonical_config,
	resolve_f3_vicreg_extension_job,
	resolve_f3_vicreg_screening_job,
	run_f3_vicreg_job,
)

if TYPE_CHECKING:
	import argparse


def build_parser() -> argparse.ArgumentParser:
	"""Build the one-cell VICReg benchmark parser."""
	parser = build_config_parser(
		'Run one F3 Local VICReg frozen-encoder benchmark cell.',
		config_help='Path to the VICReg extension YAML.',
		dry_run_help='Audit sources and print one no-write job plan.',
	)
	parser.add_argument(
		'--suite',
		required=True,
		choices=('screening', 'extension'),
		help='Select the medium screen or all-size two-arm extension.',
	)
	parser.add_argument(
		'--model',
		required=True,
		choices=(*SCREENING_MODEL_IDS, *EXTENSION_MODEL_IDS),
		help='Model ID within the selected suite.',
	)
	parser.add_argument('--layout', required=True, choices=LAYOUT_IDS)
	parser.add_argument('--size', required=True, choices=DATA_SIZES)
	add_device_argument(parser, help_text='Decoder device override.')
	parser.add_argument(
		'--max-steps', type=int, help='Stop decoder training after N smoke steps.'
	)
	parser.add_argument(
		'--resume',
		type=Path,
		help='Resume this job from its decoder/latest.pt checkpoint.',
	)
	return parser


def main() -> None:
	"""Resolve, audit, and run or inspect one benchmark cell."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_vicreg_extension_config_from_mapping,
		config_path=config_path,
	)
	canonical = load_f3_vicreg_canonical_config(config)
	resolver = (
		resolve_f3_vicreg_screening_job
		if args.suite == 'screening'
		else resolve_f3_vicreg_extension_job
	)
	job = resolver(
		config,
		canonical,
		model=args.model,
		layout=args.layout,
		size=args.size,
	)
	if args.resume is not None and not args.resume.is_file():
		raise FileNotFoundError(f'resume checkpoint does not exist: {args.resume}')
	if args.dry_run:
		result = inspect_f3_vicreg_job(
			config, canonical, job, suite=args.suite
		)
		for key, value in result.items():
			print(f'{key}: {value}')
		print('execution: dry-run; no files written')
		return
	result = run_f3_vicreg_job(
		config,
		canonical,
		job,
		suite=args.suite,
		device='auto' if args.device is None else args.device,
		max_steps=args.max_steps,
		resume=args.resume,
	)
	for key, value in result.items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
