"""Run one medium F3 overlap-subcrop PoC decoder job without source audit."""

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
from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_MODEL_IDS,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import LAYOUT_IDS
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	inspect_f3_lithology_five_way_job,
	resolve_f3_lithology_five_way_job,
	run_f3_lithology_five_way_job,
)

if TYPE_CHECKING:
	import argparse

	from seis_ssl_cluster.config.f3_lithology_five_way import F3FiveWayConfig

POC_DATA_SIZE = 'medium'
RANDOM_MODEL_ID = 'random'
CANDIDATE_MODEL_ID = 'local_barlow_twins'
RANDOM_CONFIG_STEM = 'random_medium'
MEDIUM_CONFIG_SUFFIX = '_medium'


def build_parser() -> argparse.ArgumentParser:
	"""Build the deliberately small one-cell PoC parser."""
	parser = build_config_parser(
		'Run one F3 overlap-subcrop Local BT PoC decoder job.',
		config_help=(
			'Path to random_medium.yaml or <candidate_id>_medium.yaml.'
		),
		dry_run_help='Print the resolved medium job without writing artifacts.',
	)
	parser.add_argument('--layout', required=True, choices=LAYOUT_IDS)
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


def poc_model_and_namespace(config_path: Path) -> tuple[str, str]:
	"""Map the documented config filename to one of the two PoC model slots."""
	stem = config_path.stem
	if stem == RANDOM_CONFIG_STEM:
		return RANDOM_MODEL_ID, RANDOM_MODEL_ID
	if not stem.endswith(MEDIUM_CONFIG_SUFFIX):
		raise ValueError(
			'PoC config filename must be random_medium.yaml or '
			'<candidate_id>_medium.yaml'
		)
	candidate_id = stem[: -len(MEDIUM_CONFIG_SUFFIX)]
	if not candidate_id or candidate_id in FIVE_WAY_MODEL_IDS:
		raise ValueError(
			'candidate config filename must use a candidate ID distinct from the '
			'canonical five-way model IDs'
		)
	return CANDIDATE_MODEL_ID, candidate_id


def _validate_output_namespace(
	config: F3FiveWayConfig, *, namespace: str
) -> None:
	runs_root = config.runs_root
	summary_root = config.summary_root
	if runs_root.name != 'runs' or runs_root.parent.name != namespace:
		raise ValueError(
			'outputs.runs_root must end with the config filename namespace: '
			f'{namespace}/runs'
		)
	if summary_root.name != 'summary' or summary_root.parent.name != namespace:
		raise ValueError(
			'outputs.summary_root must end with the config filename namespace: '
			f'{namespace}/summary'
		)


def main() -> None:
	"""Resolve and run the random or candidate medium cell without source audit."""
	args = build_parser().parse_args()
	config_path = parse_config_path(args)
	model, namespace = poc_model_and_namespace(config_path)
	raw = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_five_way_config_from_mapping,
		config_path=config_path,
	)
	_validate_output_namespace(config, namespace=namespace)
	job = resolve_f3_lithology_five_way_job(
		config,
		model=model,
		layout=args.layout,
		size=POC_DATA_SIZE,
	)
	if args.resume is not None and not args.resume.is_file():
		raise FileNotFoundError(f'resume checkpoint does not exist: {args.resume}')
	if args.dry_run:
		for key, value in inspect_f3_lithology_five_way_job(job).items():
			print(f'{key}: {value}')
		print('execution: dry-run; no files written')
		return
	result = run_f3_lithology_five_way_job(
		job,
		device='auto' if args.device is None else args.device,
		max_steps=args.max_steps,
		resume=args.resume,
		audit_sources=False,
	)
	for key, value in result.items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
