"""Validate and publish the experiment-107 periodic refresh handoff."""
# ruff: noqa: CPY001

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.center_trace_masked_periodic_refresh_validation import (
	load_f3_center_trace_masked_periodic_refresh_validation_config,
	validate_f3_center_trace_masked_periodic_refresh,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the strict experiment-107 validator CLI parser."""
	parser = argparse.ArgumentParser(
		description='Validate F3 center-trace masked periodic HMM refresh.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument(
		'--phase',
		choices=('inputs', 'smoke', 'checkpoints', 'complete'),
		required=True,
	)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	return parser


def main() -> int:
	"""Run one immutable validation phase."""
	args = build_parser().parse_args()
	result = validate_f3_center_trace_masked_periodic_refresh(
		load_f3_center_trace_masked_periodic_refresh_validation_config(args.config),
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
	if result.evidence.get('phase_evidence_path') is not None:
		print(f'evidence: {result.evidence["phase_evidence_path"]}')
	if result.published_handoff is not None:
		print(f'handoff: {result.published_handoff}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
