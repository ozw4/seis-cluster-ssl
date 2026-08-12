# ruff: noqa: CPY001
"""Thin CLI for validating Parihaka MAE inputs and training artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.parihaka import (
	ParihakaMaeValidationResult,
	validate_parihaka_mae,
	write_parihaka_mae_validation_report,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the closed Parihaka MAE validation parser."""
	parser = argparse.ArgumentParser(
		description='Validate Parihaka inputs, CPU smoke, or completed full run.',
	)
	parser.add_argument('--prepare-config', type=Path, required=True)
	parser.add_argument('--smoke-config', type=Path, required=True)
	parser.add_argument('--full-config', type=Path, required=True)
	parser.add_argument('--check', choices=('inputs', 'smoke', 'full'), required=True)
	parser.add_argument(
		'--json-output',
		type=Path,
		help='Optional explicit path for a small validation JSON report.',
	)
	return parser


def main() -> None:
	"""Run one stateless validation and print its key/value result."""
	args = build_parser().parse_args()
	result = validate_parihaka_mae(
		prepare_config_path=args.prepare_config,
		smoke_config_path=args.smoke_config,
		full_config_path=args.full_config,
		check=args.check,
	)
	_print_result(result)
	if args.json_output is not None:
		path = write_parihaka_mae_validation_report(result, args.json_output)
		print(f'json_output: {path}')


def _print_result(result: ParihakaMaeValidationResult) -> None:
	for key, value in result.to_dict().items():
		if isinstance(value, dict):
			for child_key, child_value in value.items():
				print(f'{key}.{child_key}: {child_value}')
		else:
			print(f'{key}: {value}')


if __name__ == '__main__':
	main()
