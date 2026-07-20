"""Validate and record current-code F3 K=6 control contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.current_k6_control import (
	validate_current_k6_checkpoint,
	validate_current_k6_embeddings,
	write_control_preflight,
	write_token_probe_comparison,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the current K=6 control contract validator parser."""
	parser = argparse.ArgumentParser(
		description='Validate current-code F3 single-head K=6 control artifacts.'
	)
	subparsers = parser.add_subparsers(dest='stage', required=True)
	preflight = subparsers.add_parser('preflight')
	preflight.add_argument('--config', type=Path, required=True)
	checkpoint = subparsers.add_parser('checkpoint')
	checkpoint.add_argument('--config', type=Path, required=True)
	checkpoint.add_argument('--reports-dir', type=Path, required=True)
	embeddings = subparsers.add_parser('embeddings')
	embeddings.add_argument('--embeddings-dir', type=Path, required=True)
	embeddings.add_argument('--checkpoint', type=Path, required=True)
	embeddings.add_argument('--reports-dir', type=Path, required=True)
	token = subparsers.add_parser('token-probe')
	token.add_argument('--current-metrics', type=Path, required=True)
	token.add_argument('--output', type=Path, required=True)
	return parser


def main() -> None:
	"""Execute one current K=6 control validation stage."""
	args = build_parser().parse_args()
	if args.stage == 'preflight':
		paths = write_control_preflight(args.config)
		print(f'control_input_manifest: {paths[0]}')
		print(f'control_input_manifest_markdown: {paths[1]}')
	elif args.stage == 'checkpoint':
		path = validate_current_k6_checkpoint(
			config_path=args.config,
			reports_dir=args.reports_dir,
		)
		print(f'checkpoint_validation: {path}')
	elif args.stage == 'embeddings':
		path = validate_current_k6_embeddings(
			embeddings_dir=args.embeddings_dir,
			checkpoint_path=args.checkpoint,
			reports_dir=args.reports_dir,
		)
		print(f'embedding_validation: {path}')
	else:
		path = write_token_probe_comparison(
			current_metrics_path=args.current_metrics,
			output_path=args.output,
		)
		print(f'token_probe_comparison: {path}')


if __name__ == '__main__':
	main()
