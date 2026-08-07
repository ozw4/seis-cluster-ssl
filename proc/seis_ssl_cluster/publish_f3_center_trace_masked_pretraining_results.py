"""Publish lightweight review artifacts for the experiment-104 handoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.f3.center_trace_masked_pretraining_results import (
	f3_center_trace_masked_pretraining_review_config_from_mapping,
	publish_f3_center_trace_masked_pretraining_review,
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the explicit-path center-trace review publisher parser."""
	parser = argparse.ArgumentParser(
		description='Publish F3 center-trace masked pretraining review artifacts.'
	)
	parser.add_argument('--artifact-root', type=Path, required=True)
	parser.add_argument('--workspace-root', type=Path, default=Path.cwd())
	parser.add_argument('--pretraining-handoff', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	return parser


def main() -> int:
	"""Validate the live PASS handoff and publish the lightweight review tree."""
	args = build_parser().parse_args()
	config = f3_center_trace_masked_pretraining_review_config_from_mapping(
		{
			'artifact_root': str(args.artifact_root),
			'workspace_root': str(args.workspace_root),
			'pretraining_handoff': str(args.pretraining_handoff),
			'output_dir': str(args.output_dir),
		}
	)
	result = publish_f3_center_trace_masked_pretraining_review(
		config, dry_run=args.dry_run
	)
	for path in (
		result.summary_json,
		result.summary_markdown,
		result.training_diagnostics,
		result.checkpoint_selection_summary,
		result.pretraining_handoff,
	):
		print(f'output: {path}')
	if args.dry_run:
		print('execution: dry-run; review artifacts were not written')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
