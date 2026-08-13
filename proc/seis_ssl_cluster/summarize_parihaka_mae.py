"""Thin CLI for completed Parihaka MAE review results."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.parihaka.mae_results import summarize_parihaka_mae


def build_parser() -> argparse.ArgumentParser:
	"""Build the direct Parihaka results parser."""
	parser = argparse.ArgumentParser(
		description='Validate and summarize a completed Parihaka MAE full run.',
	)
	parser.add_argument('--prepare-config', type=Path, required=True)
	parser.add_argument('--full-config', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--overwrite', action='store_true')
	return parser


def main() -> None:
	"""Generate or reuse the three producer-owned review files."""
	args = build_parser().parse_args()
	result = summarize_parihaka_mae(
		prepare_config_path=args.prepare_config,
		full_config_path=args.full_config,
		output_dir=args.output_dir,
		overwrite=args.overwrite,
	)
	print(f'output_dir: {result.output_dir}')
	print(f'reused: {str(result.reused).lower()}')
	for path in result.paths:
		print(f'result: {path}')


if __name__ == '__main__':
	main()
