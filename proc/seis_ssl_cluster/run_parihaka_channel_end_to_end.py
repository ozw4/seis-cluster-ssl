"""Inspect one Parihaka Channel end-to-end condition without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seis_ssl_cluster.cli import load_config_for_cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka.channel_end_to_end import (
	ChannelEndToEndPlan,
	channel_end_to_end_config_from_mapping,
	inspect_channel_end_to_end_job,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'parihaka/facies_benchmark_v1/31_channel_end_to_end_v1'
	/ '01_channel_end_to_end.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the read-only one-job parser."""
	parser = argparse.ArgumentParser(
		description='Inspect one Parihaka Channel end-to-end job.'
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument(
		'--encoder-init',
		required=True,
		choices=('pretrained', 'random'),
	)
	parser.add_argument(
		'--layout',
		required=True,
		choices=tuple(f'layout_{index:03d}' for index in range(5)),
	)
	parser.add_argument('--size', required=True, choices=('small', 'medium', 'large'))
	parser.add_argument('--layout-config', required=True, type=Path)
	parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
	parser.add_argument('--dry-run', action='store_true')
	return parser


def _print_dry_run(plan: ChannelEndToEndPlan) -> None:
	selected_source = (
		plan.pretrained_model_source
		if plan.encoder_init == 'pretrained'
		else plan.random_model_source
	)
	selected_encoder_sha = (
		plan.pretrained_encoder_initial_state_sha256
		if plan.encoder_init == 'pretrained'
		else plan.random_encoder_initial_state_sha256
	)
	print(f'encoder_init: {plan.encoder_init}')
	print(f'layout_id: {plan.layout_id}')
	print(f'data_size: {plan.data_size}')
	print(f'selected_inline_indices: {plan.train_lines.inline}')
	print(f'selected_crossline_indices: {plan.train_lines.crossline}')
	print(f'split_class_counts: {dict(plan.split_counts)}')
	print(f'class_weights: {plan.class_weights}')
	print(f'tile_counts: {dict(plan.tile_counts)}')
	print(f'encoder_checkpoint_path: {selected_source["checkpoint_path"]}')
	print(f'encoder_checkpoint_sha256: {selected_source["checkpoint_sha256"]}')
	print(f'encoder_initial_state_sha256: {selected_encoder_sha}')
	print(f'decoder_initial_state_sha256: {plan.decoder_initial_state_sha256}')
	print(f'amplitude_path: {plan.reference.source_amplitude_path}')
	print(f'reference_valid_tokens_path: {plan.reference.valid_tokens_path}')
	print(
		'preprocessing: '
		+ json.dumps(
			{
				'preprocessing': dict(plan.reference.preprocessing),
				'zero_mask': dict(plan.reference.zero_mask),
				'min_token_valid_fraction': (
					plan.reference.min_token_valid_fraction
				),
			},
			sort_keys=True,
		)
	)
	print(
		'runtime: '
		f'device={plan.runtime.device_type}, '
		f'amp={plan.runtime.amp_enabled}, '
		f'autocast_dtype={plan.runtime.autocast_dtype}, '
		f'grad_scaler={plan.runtime.grad_scaler_enabled}'
	)
	print(f'output_dir: {plan.output_dir}')
	print('execution: dry-run; no files written')


def main() -> None:
	"""Run preflight, print dry-run, and reject not-yet-implemented training."""
	args = build_parser().parse_args()
	raw = load_config_for_cli(args.config, loader=load_config)
	config = channel_end_to_end_config_from_mapping(raw)
	plan = inspect_channel_end_to_end_job(
		config,
		encoder_init=args.encoder_init,
		layout_id=args.layout,
		data_size=args.size,
		layout_config=args.layout_config,
		device=args.device,
	)
	if args.dry_run:
		_print_dry_run(plan)
		return
	raise NotImplementedError(
		'Parihaka Channel end-to-end training is not implemented; use --dry-run'
	)


if __name__ == '__main__':
	main()
