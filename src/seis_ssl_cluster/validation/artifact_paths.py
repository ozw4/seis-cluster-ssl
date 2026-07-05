"""Validate artifact path strings embedded in repository files."""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
	from collections.abc import Iterable

from seis_ssl_cluster.paths import DEFAULT_ARTIFACT_ROOT

Severity = Literal['error', 'warning']

DEFAULT_SCAN_PATHS = (
	Path('experiments'),
	Path('proc'),
	Path('docs'),
	Path('README.md'),
	Path('results'),
)
DEFAULT_TEST_FIXTURE_ALLOW_PATTERNS = (
	'tests/seis_ssl_cluster/test_config.py',
	'tests/seis_ssl_cluster/test_artifact_paths.py',
)

_TEXT_SUFFIXES = frozenset(
	{
		'.csv',
		'.json',
		'.md',
		'.py',
		'.toml',
		'.txt',
		'.yaml',
		'.yml',
	},
)
_CHECKPOINT_SUFFIXES = frozenset({'.ckpt', '.pt', '.pth'})
_HEAVY_RESULT_SUFFIXES = frozenset(
	{
		'.ckpt',
		'.joblib',
		'.npy',
		'.npz',
		'.pkl',
		'.pt',
		'.pth',
	},
)
_PATH_BOUNDARY_CHARS = ' \t\r\n"\'`),]}'
_TRAILING_PATH_CHARS = '.,;:"\'`)]}'
_RUNS_DOC_RE = re.compile(r'(?<![A-Za-z0-9_.-])runs/(?![A-Za-z0-9_.-])')


@dataclass(frozen=True)
class ArtifactPathFinding:
	"""One artifact path validation finding."""

	severity: Severity
	path: Path
	line_number: int
	value: str
	message: str


@dataclass(frozen=True)
class ArtifactPathValidationReport:
	"""Summary of an artifact path validation run."""

	root: Path
	scanned_file_count: int
	errors: tuple[ArtifactPathFinding, ...]
	warnings: tuple[ArtifactPathFinding, ...]

	@property
	def ok(self) -> bool:
		"""Return whether validation found no errors."""
		return not self.errors


@dataclass(frozen=True)
class _PathOccurrence:
	path: Path
	line_number: int
	value: str
	line: str


def validate_artifact_paths(
	*,
	root: Path = DEFAULT_ARTIFACT_ROOT,
	scan_paths: Iterable[Path] = DEFAULT_SCAN_PATHS,
	fail_on_runs: bool = False,
	allow_patterns: Iterable[str] = (),
	allow_test_fixtures: bool = False,
) -> ArtifactPathValidationReport:
	"""Validate artifact path strings found under ``scan_paths``."""
	root = Path(root)
	errors: list[ArtifactPathFinding] = []
	warnings: list[ArtifactPathFinding] = []
	allow_patterns = tuple(allow_patterns)
	if allow_test_fixtures:
		allow_patterns = (*allow_patterns, *DEFAULT_TEST_FIXTURE_ALLOW_PATTERNS)

	scanned_file_count = 0
	for scan_path in scan_paths:
		path = Path(scan_path)
		if _path_is_allowed(path, allow_patterns):
			continue
		if not path.exists():
			errors.append(
				_finding(
					'error',
					path,
					0,
					str(path),
					f'scan path does not exist: {path}',
				),
			)
			continue
		if path.is_file():
			if _should_scan_file(path):
				scanned_file_count += 1
				file_errors, file_warnings = _scan_file(
					path,
					root=root,
					fail_on_runs=fail_on_runs,
				)
				errors.extend(file_errors)
				warnings.extend(file_warnings)
			continue
		for file_path in _iter_scan_files(path, allow_patterns):
			scanned_file_count += 1
			file_errors, file_warnings = _scan_file(
				file_path,
				root=root,
				fail_on_runs=fail_on_runs,
			)
			errors.extend(file_errors)
			warnings.extend(file_warnings)

	return ArtifactPathValidationReport(
		root=root,
		scanned_file_count=scanned_file_count,
		errors=tuple(errors),
		warnings=tuple(warnings),
	)


def _iter_scan_files(root: Path, allow_patterns: tuple[str, ...]) -> Iterable[Path]:
	for path in sorted(root.rglob('*')):
		if not path.is_file():
			continue
		if _path_is_allowed(path, allow_patterns):
			continue
		if not _should_scan_file(path):
			continue
		yield path


def _should_scan_file(path: Path) -> bool:
	return path.suffix.lower() in _TEXT_SUFFIXES


def _scan_file(
	path: Path,
	*,
	root: Path,
	fail_on_runs: bool,
) -> tuple[list[ArtifactPathFinding], list[ArtifactPathFinding]]:
	errors: list[ArtifactPathFinding] = []
	warnings: list[ArtifactPathFinding] = []
	try:
		text = path.read_text(encoding='utf-8')
	except UnicodeDecodeError:
		return errors, warnings

	for occurrence in _path_occurrences(path=path, text=text, root=root):
		file_errors, file_warnings = _validate_occurrence(
			occurrence,
			root=root,
			fail_on_runs=fail_on_runs,
		)
		errors.extend(file_errors)
		warnings.extend(file_warnings)
	warnings.extend(
		[
			_finding(
				'warning',
				occurrence.path,
				occurrence.line_number,
				occurrence.value,
				'legacy docs mention runs/; active artifact paths must not use it',
			)
			for occurrence in _runs_doc_occurrences(path=path, text=text)
		]
	)

	if path.name == 'publish_manifest.json':
		file_errors, file_warnings = _validate_publish_manifest(path, root=root)
		errors.extend(file_errors)
		warnings.extend(file_warnings)

	return errors, warnings


def _path_occurrences(
	*,
	path: Path,
	text: str,
	root: Path,
) -> Iterable[_PathOccurrence]:
	pattern = _path_pattern(root)
	for line_number, line in enumerate(text.splitlines(), start=1):
		for match in pattern.finditer(line):
			value = _clean_path_token(match.group(0))
			if not value:
				continue
			yield _PathOccurrence(
				path=path,
				line_number=line_number,
				value=value,
				line=line,
			)


def _runs_doc_occurrences(*, path: Path, text: str) -> Iterable[_PathOccurrence]:
	if path.suffix.lower() != '.md':
		return ()
	occurrences: list[_PathOccurrence] = []
	for line_number, line in enumerate(text.splitlines(), start=1):
		if 'runs/' not in line:
			continue
		if _RUNS_DOC_RE.search(line) is None:
			continue
		occurrences.append(
			_PathOccurrence(
				path=path,
				line_number=line_number,
				value='runs/',
				line=line,
			),
		)
	return tuple(occurrences)


def _path_pattern(root: Path) -> re.Pattern[str]:
	root_text = re.escape(root.as_posix().rstrip('/'))
	prefixes = (root_text, 'artifacts', 'results', 'runs')
	prefix_pattern = '|'.join(prefixes)
	return re.compile(
		rf'(?<![A-Za-z0-9_.-])(?:{prefix_pattern})(?:/[^{re.escape(_PATH_BOUNDARY_CHARS)}]+)+',
	)


def _clean_path_token(value: str) -> str:
	value = value.rstrip(_TRAILING_PATH_CHARS)
	while value.endswith('/'):
		value = value[:-1]
	return value


def _validate_occurrence(
	occurrence: _PathOccurrence,
	*,
	root: Path,
	fail_on_runs: bool,
) -> tuple[list[ArtifactPathFinding], list[ArtifactPathFinding]]:
	errors: list[ArtifactPathFinding] = []
	warnings: list[ArtifactPathFinding] = []
	value = occurrence.value

	if _is_runs_path(value, root=root):
		severity: Severity = 'error' if fail_on_runs else 'warning'
		message = 'runs/ artifact path is not allowed'
		finding = _finding(
			severity,
			occurrence.path,
			occurrence.line_number,
			value,
			message,
		)
		if severity == 'error':
			errors.append(finding)
		else:
			warnings.append(finding)
		return errors, warnings

	artifact_parts = _artifact_relative_parts(value, root=root)
	if artifact_parts is not None:
		errors.extend(_artifact_path_errors(occurrence, artifact_parts))
		warnings.extend(_artifact_path_warnings(occurrence, artifact_parts))

	return errors, warnings


def _artifact_relative_parts(value: str, *, root: Path) -> tuple[str, ...] | None:
	root_text = root.as_posix().rstrip('/')
	if value == root_text:
		return ()
	if value.startswith(f'{root_text}/'):
		return tuple(part for part in value[len(root_text) + 1 :].split('/') if part)
	if value.startswith('artifacts/'):
		return tuple(
			part for part in value.removeprefix('artifacts/').split('/') if part
		)
	return None


def _artifact_path_errors(
	occurrence: _PathOccurrence,
	parts: tuple[str, ...],
) -> list[ArtifactPathFinding]:
	if not parts:
		return []
	stage = parts[0]
	errors: list[ArtifactPathFinding] = []
	if stage == 'runs':
		errors.append(
			_finding(
				'error',
				occurrence.path,
				occurrence.line_number,
				occurrence.value,
				'runs/ artifact path is not allowed',
			),
		)
	if _is_checkpoint_like(parts) and stage != 'pretraining':
		errors.append(
			_finding(
				'error',
				occurrence.path,
				occurrence.line_number,
				occurrence.value,
				'checkpoint artifacts must be written under pretraining/',
			),
		)
	if stage == 'embeddings' and len(parts) >= 7:
		errors.append(
			_finding(
				'error',
				occurrence.path,
				occurrence.line_number,
				occurrence.value,
				(
					'embeddings/ paths must not include cluster_spec '
					'or viz_spec components'
				),
			),
		)
	if stage == 'clustering' and len(parts) >= 8:
		errors.append(
			_finding(
				'error',
				occurrence.path,
				occurrence.line_number,
				occurrence.value,
				'clustering/ paths must not include viz_spec components',
			),
		)
	return errors


def _artifact_path_warnings(
	occurrence: _PathOccurrence,
	parts: tuple[str, ...],
) -> list[ArtifactPathFinding]:
	if not parts:
		return []
	stage = parts[0]
	if stage != 'embeddings':
		return []
	if _is_file_path(parts):
		return []
	if len(parts) in {4, 5}:
		return [
			_finding(
				'warning',
				occurrence.path,
				occurrence.line_number,
				occurrence.value,
				'legacy embeddings path omits either subset or embed_spec component',
			),
		]
	return []


def _validate_publish_manifest(
	path: Path,
	*,
	root: Path,
) -> tuple[list[ArtifactPathFinding], list[ArtifactPathFinding]]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		return [
			_finding(
				'error',
				path,
				0,
				str(path),
				f'publish_manifest.json is not valid JSON: {exc}',
			),
		], []
	if not isinstance(payload, dict):
		return [
			_finding(
				'error',
				path,
				0,
				str(path),
				'publish_manifest.json must be a JSON object',
			),
		], []

	errors: list[ArtifactPathFinding] = []
	warnings: list[ArtifactPathFinding] = []
	items = payload.get('items')
	if isinstance(items, list):
		for index, item in enumerate(items):
			if not isinstance(item, dict):
				continue
			errors.extend(_publish_manifest_item_errors(path, index, item))
			warnings.extend(
				_publish_manifest_source_warnings(
					path,
					root=root,
					index=index,
					item=item,
				),
			)
	source_artifact_root = payload.get('source_artifact_root')
	if isinstance(source_artifact_root, str) and _starts_with_root(
		source_artifact_root,
		root=root,
	):
		warnings.append(
			_finding(
				'warning',
				path,
				0,
				source_artifact_root,
				'publish_manifest source_artifact_root records a local artifact path',
			),
		)
	return errors, warnings


def _publish_manifest_item_errors(
	path: Path,
	index: int,
	item: dict[object, object],
) -> list[ArtifactPathFinding]:
	errors: list[ArtifactPathFinding] = []
	for key in ('source', 'target'):
		value = item.get(key)
		if not isinstance(value, str):
			continue
		suffix = Path(value).suffix.lower()
		if suffix not in _HEAVY_RESULT_SUFFIXES:
			continue
		errors.append(
			_finding(
				'error',
				path,
				0,
				value,
				(
					f'publish_manifest items[{index}].{key} '
					f'references heavy artifact {suffix!r}'
				),
			),
		)
	return errors


def _publish_manifest_source_warnings(
	path: Path,
	*,
	root: Path,
	index: int,
	item: dict[object, object],
) -> list[ArtifactPathFinding]:
	source = item.get('source')
	if not isinstance(source, str):
		return []
	if not _starts_with_root(source, root=root):
		return []
	return [
		_finding(
			'warning',
			path,
			0,
			source,
			f'publish_manifest items[{index}].source records a local artifact path',
		),
	]


def _starts_with_root(value: str, *, root: Path) -> bool:
	root_text = root.as_posix().rstrip('/')
	return value == root_text or value.startswith(f'{root_text}/')


def _is_runs_path(value: str, *, root: Path) -> bool:
	root_text = root.as_posix().rstrip('/')
	return value.startswith(('runs/', f'{root_text}/runs/'))


def _is_checkpoint_like(parts: tuple[str, ...]) -> bool:
	name = parts[-1].lower()
	return Path(name).suffix in _CHECKPOINT_SUFFIXES or 'checkpoint' in name


def _is_file_path(parts: tuple[str, ...]) -> bool:
	return bool(Path(parts[-1]).suffix)


def _path_is_allowed(path: Path, allow_patterns: tuple[str, ...]) -> bool:
	if not allow_patterns:
		return False
	path_text = path.as_posix()
	return any(fnmatch.fnmatch(path_text, pattern) for pattern in allow_patterns)


def _finding(
	severity: Severity,
	path: Path,
	line_number: int,
	value: str,
	message: str,
) -> ArtifactPathFinding:
	return ArtifactPathFinding(
		severity=severity,
		path=path,
		line_number=line_number,
		value=value,
		message=message,
	)


__all__ = [
	'DEFAULT_SCAN_PATHS',
	'DEFAULT_TEST_FIXTURE_ALLOW_PATTERNS',
	'ArtifactPathFinding',
	'ArtifactPathValidationReport',
	'validate_artifact_paths',
]
