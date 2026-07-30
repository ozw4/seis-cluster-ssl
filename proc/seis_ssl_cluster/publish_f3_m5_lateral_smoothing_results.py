"""Publish lightweight, target-only F3 M5-LS calibration review artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.lateral_smoothing_results import (
	f3_m5_lateral_smoothing_review_config_from_mapping,
	publish_f3_m5_lateral_smoothing_review,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the explicit-path M5-LS review-publisher CLI."""
	parser = argparse.ArgumentParser(
		description=(
			'Publish lightweight F3 M5-LS target-calibration evidence without '
			'facies/lithology labels or downstream metrics.'
		),
	)
	parser.add_argument('--artifact-root', type=Path, required=True)
	parser.add_argument('--workspace-root', type=Path, default=Path.cwd())
	parser.add_argument('--calibration-handoff', type=Path, required=True)
	parser.add_argument('--calibration-report', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--smoke-evidence', type=Path)
	parser.add_argument('--dry-run', action='store_true')
	return parser


def main() -> int:
	"""Publish or dry-run the M5-LS target-only review artifact set."""
	args = build_parser().parse_args()
	config = f3_m5_lateral_smoothing_review_config_from_mapping(
		{
			'artifact_root': str(args.artifact_root),
			'workspace_root': str(args.workspace_root),
			'calibration_handoff': str(args.calibration_handoff),
			'calibration_report': str(args.calibration_report),
			'output_dir': str(args.output_dir),
			'smoke_evidence': None
			if args.smoke_evidence is None
			else str(args.smoke_evidence),
		},
	)
	result = publish_f3_m5_lateral_smoothing_review(config, dry_run=args.dry_run)
	print(f'target_calibration: {result.calibration_status}')
	print(
		'selected_beta: '
		+ ('HOLD' if result.selected_beta is None else f'{result.selected_beta:.2f}')
	)
	print(f'smoke: {result.smoke_status}')
	for path in (
		result.candidate_csv,
		result.summary_json,
		result.summary_markdown,
		result.calibration_handoff,
		result.smoke_summary,
	):
		if path is not None:
			print(f'output: {path}')
	if result.publish_manifest is not None:
		print(f'publish_manifest: {result.publish_manifest.manifest_path}')
	if args.dry_run:
		print('execution: dry-run; review artifacts were not written')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
