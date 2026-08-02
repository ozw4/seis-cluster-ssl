"""Audit the immutable 3-of-4 and unanimous XY-neighbour target lineage."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.xy_neighbor_unanimous_target_audit import (
	audit_f3_xy_neighbor_unanimous_targets,
	load_f3_xy_neighbor_unanimous_target_audit_config,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the standalone target-only unanimous audit parser."""
	parser = argparse.ArgumentParser(
		description='Audit F3 unanimous XY-neighbour hard-target publication.'
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	return parser


def main() -> int:
	"""Write or immutably reuse target-only GO/HOLD evidence."""
	args = build_parser().parse_args()
	result = audit_f3_xy_neighbor_unanimous_targets(
		load_f3_xy_neighbor_unanimous_target_audit_config(args.config),
		dry_run=args.dry_run,
		only_missing=args.only_missing,
		quarantine_invalid=args.quarantine_invalid,
	)
	print(f'action: {result.action}')
	print(f'status: {result.payload["status"]}')
	print(f'audit: {result.output_path}')
	if result.quarantine_path is not None:
		print(f'quarantine: {result.quarantine_path}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
