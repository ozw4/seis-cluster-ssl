"""Calibrate preregistered F3 M5-LS lateral hard targets target-only."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.lateral_smoothing_target_calibration import (
	calibrate_f3_m5_lateral_targets,
	load_f3_m5_lateral_target_calibration_config,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the deliberately small target-only calibration CLI."""
	parser = argparse.ArgumentParser(
		description=(
			'Calibrate the fixed F3 M5-LS beta010/beta025/beta050 target set '
			'without downstream labels or metrics.'
		)
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	return parser


def main() -> int:
	"""Run the read-only decision or publish immutable calibration evidence."""
	args = build_parser().parse_args()
	result = calibrate_f3_m5_lateral_targets(
		load_f3_m5_lateral_target_calibration_config(args.config),
		dry_run=args.dry_run,
		only_missing=args.only_missing,
		quarantine_invalid=args.quarantine_invalid,
	)
	print(f'status: {result.status}')
	if result.status == 'FAIL':
		print(f'error: {result.evidence["error"]}')
		return 1
	print(
		'selected_beta: '
		+ ('HOLD' if result.selected_beta is None else f'{result.selected_beta:.2f}')
	)
	for name, candidate in result.evidence['candidates'].items():
		eligibility = candidate['eligibility']
		print(f'{name}: eligible={eligibility["eligible"]}')
		for reason in eligibility['reasons']:
			print(f'  reason: {reason}')
	if result.published_selected_manifest is not None:
		print(f'selected_manifest: {result.published_selected_manifest}')
	if result.published_handoff is not None:
		print(f'calibration_handoff: {result.published_handoff}')
	if result.published_report is not None:
		print(f'calibration_report: {result.published_report}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
