"""Validate repository-managed lightweight result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from seis_ssl_cluster.results.publish import (
	DEFAULT_MAX_FILE_SIZE_BYTES,
	FORBIDDEN_SUFFIXES,
)

DEFAULT_LOCAL_PATH_MARKERS = (
	'/home/dcuser/',
	'/workspace/artifacts/',
)
LOCAL_PATH_POLICY_WARNING = 'warn'
LOCAL_PATH_POLICY_ERROR = 'error'
LocalPathPolicy = Literal['warn', 'error']


@dataclass(frozen=True)
class ResultsValidationFinding:
	"""One validation warning or error for a path under results."""

	severity: Literal['error', 'warning']
	path: Path
	message: str


@dataclass(frozen=True)
class ResultsValidationReport:
	"""Summary of a results artifact validation run."""

	root: Path
	file_count: int
	errors: tuple[ResultsValidationFinding, ...]
	warnings: tuple[ResultsValidationFinding, ...]

	@property
	def ok(self) -> bool:
		"""Return whether validation found no errors."""
		return not self.errors


@dataclass(frozen=True)
class _FileValidationRules:
	max_file_size_bytes: int
	local_path_policy: LocalPathPolicy
	local_path_markers: tuple[str, ...]


def validate_results_artifacts(
	root: Path,
	*,
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
	required_files: tuple[Path, ...] = (),
	local_path_policy: LocalPathPolicy = LOCAL_PATH_POLICY_WARNING,
	local_path_markers: tuple[str, ...] = DEFAULT_LOCAL_PATH_MARKERS,
) -> ResultsValidationReport:
	"""Validate lightweight files stored below ``root``."""
	_validate_max_file_size_bytes(max_file_size_bytes)
	_validate_local_path_policy(local_path_policy)

	root = Path(root)
	errors: list[ResultsValidationFinding] = []
	warnings: list[ResultsValidationFinding] = []

	if not root.exists():
		errors.append(_finding('error', root, f'results root does not exist: {root}'))
		return ResultsValidationReport(
			root=root,
			file_count=0,
			errors=tuple(errors),
			warnings=tuple(warnings),
		)
	if not root.is_dir():
		errors.append(
			_finding('error', root, f'results root is not a directory: {root}')
		)
		return ResultsValidationReport(
			root=root,
			file_count=0,
			errors=tuple(errors),
			warnings=tuple(warnings),
		)

	root_resolved = root.resolve(strict=False)
	errors.extend(
		_required_file_findings(root=root_resolved, required_files=required_files),
	)
	rules = _FileValidationRules(
		max_file_size_bytes=max_file_size_bytes,
		local_path_policy=local_path_policy,
		local_path_markers=local_path_markers,
	)

	file_count = 0
	for path in sorted(root.rglob('*')):
		relative_path = path.relative_to(root)
		if 'artifacts' in relative_path.parts:
			errors.append(
				_finding(
					'error',
					path,
					'artifacts/ must not be stored inside results/',
				),
			)
		if not path.is_file():
			continue

		file_count += 1
		file_errors, file_warnings = _validate_file(path=path, rules=rules)
		errors.extend(file_errors)
		warnings.extend(file_warnings)

	return ResultsValidationReport(
		root=root,
		file_count=file_count,
		errors=tuple(errors),
		warnings=tuple(warnings),
	)


def _validate_file(
	*,
	path: Path,
	rules: _FileValidationRules,
) -> tuple[list[ResultsValidationFinding], list[ResultsValidationFinding]]:
	errors: list[ResultsValidationFinding] = []
	warnings: list[ResultsValidationFinding] = []
	suffix = path.suffix.lower()
	if suffix in FORBIDDEN_SUFFIXES:
		errors.append(
			_finding('error', path, f'forbidden heavy artifact suffix {suffix!r}'),
		)

	size_bytes = path.stat().st_size
	size_ok = size_bytes <= rules.max_file_size_bytes
	if not size_ok:
		errors.append(
			_finding(
				'error',
				path,
				(
					'file exceeds max_file_size_bytes: '
					f'{size_bytes} bytes > {rules.max_file_size_bytes} bytes'
				),
			),
		)

	if not size_ok:
		return errors, warnings
	for marker in _markers_found(path, rules.local_path_markers):
		severity: Literal['error', 'warning'] = (
			'error' if rules.local_path_policy == LOCAL_PATH_POLICY_ERROR else 'warning'
		)
		finding = _finding(
			severity,
			path,
			f'local absolute path marker found: {marker}',
		)
		if severity == 'error':
			errors.append(finding)
		else:
			warnings.append(finding)
	return errors, warnings


def _required_file_findings(
	*,
	root: Path,
	required_files: tuple[Path, ...],
) -> list[ResultsValidationFinding]:
	findings: list[ResultsValidationFinding] = []
	for required_file in required_files:
		required = Path(required_file)
		path = required if required.is_absolute() else root / required
		resolved = path.resolve(strict=False)
		try:
			resolved.relative_to(root)
		except ValueError:
			findings.append(
				_finding(
					'error',
					path,
					f'required file must be under results root: {required_file}',
				),
			)
			continue
		if not resolved.exists():
			findings.append(
				_finding('error', path, f'required file is missing: {required_file}'),
			)
		elif not resolved.is_file():
			findings.append(
				_finding(
					'error', path, f'required path is not a file: {required_file}'
				),
			)
	return findings


def _markers_found(path: Path, markers: tuple[str, ...]) -> tuple[str, ...]:
	marker_bytes = tuple(marker.encode('utf-8') for marker in markers)
	if not marker_bytes:
		return ()
	found: list[str] = []
	with path.open('rb') as file_obj:
		previous = b''
		while chunk := file_obj.read(8192):
			window = previous + chunk
			for marker, encoded in zip(markers, marker_bytes, strict=True):
				if marker not in found and encoded in window:
					found.append(marker)
			previous = window[-max(len(item) for item in marker_bytes) :]
	return tuple(found)


def _validate_max_file_size_bytes(max_file_size_bytes: int) -> None:
	if max_file_size_bytes <= 0:
		msg = f'max_file_size_bytes must be positive; got {max_file_size_bytes!r}'
		raise ValueError(msg)


def _validate_local_path_policy(local_path_policy: str) -> None:
	if local_path_policy not in {
		LOCAL_PATH_POLICY_WARNING,
		LOCAL_PATH_POLICY_ERROR,
	}:
		msg = f'local_path_policy must be "warn" or "error"; got {local_path_policy!r}'
		raise ValueError(msg)


def _finding(
	severity: Literal['error', 'warning'],
	path: Path,
	message: str,
) -> ResultsValidationFinding:
	return ResultsValidationFinding(severity=severity, path=path, message=message)
