"""Validate and publish F3 multi-head pretraining handoffs."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.multi_head_pretraining_validation import (
	load_f3_multi_head_pretraining_validation_config,
	validate_f3_multi_head_pretraining,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the standalone upstream validation CLI."""
	parser = argparse.ArgumentParser(
		description='Validate F3 K=6/8/10 pretraining and publish PASS handoffs.',
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--phase', choices=('checkpoints', 'complete'), required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	return parser


def main() -> int:
	"""Run one explicit validation phase."""
	args = build_parser().parse_args()
	config = load_f3_multi_head_pretraining_validation_config(args.config)
	result = validate_f3_multi_head_pretraining(
		config,
		phase=args.phase,
		dry_run=args.dry_run,
		only_missing=args.only_missing,
		quarantine_invalid=args.quarantine_invalid,
	)
	print(f'phase: {result.phase}')
	for variant, evidence in result.candidates.items():
		if args.dry_run:
			print(f'{variant}: plan={evidence["planned_action"]}')
		if evidence['status'] == 'PASS':
			print(f'{variant}: PASS; best={evidence["best_path"]}')
		else:
			print(f'{variant}: FAIL; error={evidence["error"]}')
	for path in result.published_handoffs:
		print(f'handoff: {path}')
	if args.dry_run:
		print('execution: dry-run; publication skipped')
	failed = any(
		evidence['status'] != 'PASS' for evidence in result.candidates.values()
	)
	return int(failed)


if __name__ == '__main__':
	raise SystemExit(main())
