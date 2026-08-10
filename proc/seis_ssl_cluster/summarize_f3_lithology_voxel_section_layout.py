"""Summarize generic F3 section-layout benchmark manifests."""
# ruff: noqa: CPY001

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli, resolve_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_roster import (
	EXPECTED_MODEL_IDS,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_results import (
	f3_lithology_voxel_section_layout_results_config_from_mapping,
	summarize_f3_lithology_voxel_section_layout_results,
)

DEFAULT_CONFIG = (
	'experiments/f3/facies_benchmark_v1/109_f3_voxel_section_layout_v1/'
	'05_summarize_section_layout_benchmark.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the two-mode summarizer CLI."""
	parser = argparse.ArgumentParser(
		description='Summarize paired-layout F3 voxel benchmark metrics.'
	)
	parser.add_argument('--config', type=Path, default=Path(DEFAULT_CONFIG))
	parser.add_argument(
		'--model-id', choices=EXPECTED_MODEL_IDS, help='Summarize one roster model.'
	)
	parser.add_argument(
		'--no-publish',
		action='store_true',
		help='Required in model mode; write only below the artifact benchmark root.',
	)
	return parser


def main() -> None:
	"""Run model/no-publish mode or complete-roster final mode."""
	args = build_parser().parse_args()
	if args.model_id is not None and not args.no_publish:
		raise ValueError('--model-id requires --no-publish')
	if args.model_id is None and args.no_publish:
		raise ValueError('--no-publish requires --model-id')
	raw = load_config_for_cli(args.config, loader=load_config)
	config = resolve_config_for_cli(
		raw,
		resolver=f3_lithology_voxel_section_layout_results_config_from_mapping,
		config_path=args.config,
	)
	result = summarize_f3_lithology_voxel_section_layout_results(
		config, model_id=args.model_id, no_publish=args.no_publish
	)
	print(f'section_layout_results.mode: {result.inspection.mode}')
	print(f'section_layout_results.output_dir: {result.output_dir}')
	print(f'section_layout_results.files: {len(result.files)}')


if __name__ == '__main__':
	main()
