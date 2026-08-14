"""Run one Parihaka Channel decoder condition."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_data import CHANNEL_TEST_MODE
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
	inspect_channel_decoder_job,
	run_channel_decoder_job,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/30_channel_benchmark_v1'
	/ '06_channel_benchmark.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the one-job parser."""
	parser = argparse.ArgumentParser(
		description='Run one Parihaka Channel decoder job.'
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument('--model', required=True, choices=('pretrained', 'random'))
	parser.add_argument(
		'--layout',
		required=True,
		choices=tuple(f'layout_{index:03d}' for index in range(5)),
	)
	parser.add_argument('--size', required=True, choices=('small', 'medium', 'large'))
	parser.add_argument('--layout-config', required=True, type=Path)
	parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--max-steps', type=int)
	parser.add_argument('--resume', type=Path)
	return parser


def main() -> None:
	"""Inspect or execute exactly one condition."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = channel_decoder_config_from_mapping(raw)
	plan = inspect_channel_decoder_job(
		config,
		model=args.model,
		layout_id=args.layout,
		data_size=args.size,
		layout_config=args.layout_config,
	)
	if args.dry_run:
		print(f'model: {plan.model}')
		print(f'layout_id: {plan.layout_id}')
		print(f'data_size: {plan.data_size}')
		print(f'selected_inline_indices: {plan.train_lines.inline}')
		print(f'selected_crossline_indices: {plan.train_lines.crossline}')
		print(f'validation_inline_indices: {plan.layouts.validation.inline}')
		print(f'validation_crossline_indices: {plan.layouts.validation.crossline}')
		print(f'test_mode: {CHANNEL_TEST_MODE}')
		print(
			f'reserved_large_inline_indices: {plan.reserved_training_lines.inline}'
		)
		print(
			'reserved_large_crossline_indices: '
			f'{plan.reserved_training_lines.crossline}'
		)
		print(f'split_class_counts: {dict(plan.split_counts)}')
		print(f'class_weights: {plan.class_weights}')
		print(f'tile_counts: {dict(plan.tile_counts)}')
		print(f'output_dir: {plan.output_dir}')
		print('execution: dry-run; no files written')
		return
	metrics = run_channel_decoder_job(
		plan, device=args.device, max_steps=args.max_steps, resume=args.resume
	)
	if metrics is None:
		print(f'latest: {plan.output_dir / "latest.pt"}')
		print('execution: stopped at max-steps; resume from latest.pt')
	else:
		print(f'metrics: {metrics}')


if __name__ == '__main__':
	main()
