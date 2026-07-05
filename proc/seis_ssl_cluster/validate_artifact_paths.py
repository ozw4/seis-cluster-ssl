"""Validate repository artifact path strings."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from seis_ssl_cluster.paths import DEFAULT_ARTIFACT_ROOT
from seis_ssl_cluster.validation import (
	ArtifactPathFinding,
	ArtifactPathValidationReport,
	validate_artifact_paths,
)
from seis_ssl_cluster.validation.artifact_paths import DEFAULT_SCAN_PATHS


def main() -> int:
	"""Validate artifact path contracts and return a process exit code."""
	parser = ArgumentParser(
		description='Validate artifact path strings in configs, docs, and results.',
	)
	parser.add_argument(
		'--root',
		type=Path,
		default=DEFAULT_ARTIFACT_ROOT,
		help='Artifact root that path strings must follow.',
	)
	parser.add_argument(
		'--scan',
		type=Path,
		nargs='+',
		default=list(DEFAULT_SCAN_PATHS),
		help='Files or directories to scan.',
	)
	parser.add_argument(
		'--fail-on-runs',
		action='store_true',
		help='Treat runs/ artifact paths as errors.',
	)
	parser.add_argument(
		'--allow-pattern',
		action='append',
		default=[],
		help='fnmatch pattern for files to skip. May be passed multiple times.',
	)
	parser.add_argument(
		'--allow-test-fixtures',
		action='store_true',
		help='Skip test files that intentionally contain rejected path examples.',
	)
	args = parser.parse_args()

	report = validate_artifact_paths(
		root=args.root,
		scan_paths=tuple(args.scan),
		fail_on_runs=args.fail_on_runs,
		allow_patterns=tuple(args.allow_pattern),
		allow_test_fixtures=args.allow_test_fixtures,
	)
	_print_report(report)
	return 0 if report.ok else 1


def _print_report(report: ArtifactPathValidationReport) -> None:
	status = 'ok' if report.ok else 'failed'
	print(f'artifact path validation: {status}')
	print(f'root: {report.root}')
	print(f'scanned_file_count: {report.scanned_file_count}')
	print(f'error_count: {len(report.errors)}')
	print(f'warning_count: {len(report.warnings)}')
	for finding in report.errors:
		print(_format_finding(finding))
	for finding in report.warnings:
		print(_format_finding(finding))


def _format_finding(finding: ArtifactPathFinding) -> str:
	location = (
		f'{finding.path}:{finding.line_number}'
		if finding.line_number
		else finding.path
	)
	return f'{finding.severity}: {location}: {finding.message}: {finding.value}'


if __name__ == '__main__':
	raise SystemExit(main())
