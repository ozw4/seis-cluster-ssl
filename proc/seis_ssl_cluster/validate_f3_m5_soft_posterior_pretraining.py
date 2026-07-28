"""Validate and publish the strict F3 M5-U soft-posterior handoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.soft_posterior_pretraining_validation import (
	load_f3_m5_soft_posterior_pretraining_validation_config,
	validate_f3_m5_soft_posterior_pretraining,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the standalone M5-U validator CLI."""
	parser = argparse.ArgumentParser(
		description='Validate F3 M5-U soft-posterior pretraining.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument(
		'--phase',
		choices=('targets', 'smoke', 'checkpoints', 'complete'),
		required=True,
	)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	return parser


def main() -> int:
	"""Run the requested strict validation phase."""
	args = build_parser().parse_args()
	result = validate_f3_m5_soft_posterior_pretraining(
		load_f3_m5_soft_posterior_pretraining_validation_config(args.config),
		phase=args.phase,
		dry_run=args.dry_run,
		only_missing=args.only_missing,
		quarantine_invalid=args.quarantine_invalid,
	)
	print(f'phase: {result.phase}')
	print(f'status: {result.evidence["status"]}')
	if result.evidence['status'] != 'PASS':
		print(f'error: {result.evidence["error"]}')
		return 1
	if result.published_handoff is not None:
		print(f'handoff: {result.published_handoff}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
