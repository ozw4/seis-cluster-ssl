"""Validate lightweight artifacts stored under results/."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from seis_ssl_cluster.results import (
	DEFAULT_MAX_FILE_SIZE_BYTES,
	LOCAL_PATH_POLICY_ERROR,
	LOCAL_PATH_POLICY_WARNING,
	ResultsValidationFinding,
	ResultsValidationReport,
	validate_results_artifacts,
)

BYTES_PER_MIB = 1024 * 1024


def main() -> int:
	"""Validate results artifacts and return a process exit code."""
	parser = ArgumentParser(
		description='Validate lightweight artifacts stored under results/.',
	)
	parser.add_argument(
		'--root',
		type=Path,
		default=Path('results'),
		help='Results directory to validate.',
	)
	parser.add_argument(
		'--max-file-size-mb',
		type=float,
		default=DEFAULT_MAX_FILE_SIZE_BYTES / BYTES_PER_MIB,
		help='Maximum allowed file size in MiB.',
	)
	parser.add_argument(
		'--required-file',
		type=Path,
		action='append',
		default=[],
		help='Required file path under --root. May be passed multiple times.',
	)
	parser.add_argument(
		'--local-path-policy',
		choices=(LOCAL_PATH_POLICY_WARNING, LOCAL_PATH_POLICY_ERROR),
		default=LOCAL_PATH_POLICY_WARNING,
		help='Treat local absolute paths as warnings or errors.',
	)
	args = parser.parse_args()

	report = validate_results_artifacts(
		args.root,
		max_file_size_bytes=_max_file_size_bytes(args.max_file_size_mb),
		required_files=tuple(args.required_file),
		local_path_policy=args.local_path_policy,
	)
	_print_report(report)
	return 0 if report.ok else 1


def _max_file_size_bytes(max_file_size_mb: float) -> int:
	if max_file_size_mb <= 0:
		msg = f'--max-file-size-mb must be positive; got {max_file_size_mb!r}'
		raise ValueError(msg)
	return int(max_file_size_mb * BYTES_PER_MIB)


def _print_report(report: ResultsValidationReport) -> None:
	status = 'ok' if report.ok else 'failed'
	print(f'results validation: {status}')
	print(f'root: {report.root}')
	print(f'file_count: {report.file_count}')
	print(f'error_count: {len(report.errors)}')
	print(f'warning_count: {len(report.warnings)}')
	for finding in report.errors:
		print(_format_finding(finding))
	for finding in report.warnings:
		print(_format_finding(finding))


def _format_finding(finding: ResultsValidationFinding) -> str:
	return f'{finding.severity}: {finding.path}: {finding.message}'


if __name__ == '__main__':
	raise SystemExit(main())
