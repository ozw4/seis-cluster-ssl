"""Path-list utilities for explicit NOPIMS `.npy` training manifests."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SURVEY_ID_ALLOWED = re.compile(r'[^A-Za-z0-9_.-]')


def load_npy_path_list(path_list: str | Path) -> list[str]:
	"""Load non-empty, non-comment entries from a plain text path-list file."""
	source = Path(path_list)
	if not source.is_file():
		msg = f'path-list file does not exist: {source}'
		raise FileNotFoundError(msg)

	entries: list[str] = []
	for line_number, line in enumerate(
		source.read_text(encoding='utf-8').splitlines(),
		start=1,
	):
		entry = line.strip()
		if not entry or entry.startswith('#'):
			continue
		if '#' in entry:
			msg = (
				f'path-list entry on line {line_number} must contain exactly one '
				f'.npy path and no inline comment: {source}'
			)
			raise ValueError(msg)
		entries.append(entry)
	return entries


def resolve_npy_path_list(
	path_list: str | Path,
	nopims_root: str | Path,
) -> list[Path]:
	"""Resolve and validate `.npy` paths from a path-list file."""
	root = Path(nopims_root)
	paths: list[Path] = []
	seen: dict[Path, Path] = {}
	for entry in load_npy_path_list(path_list):
		path = Path(entry)
		if not path.is_absolute():
			path = root / path
		if path.suffix != '.npy':
			msg = f'path-list entry must have .npy suffix: {path}'
			raise ValueError(msg)
		if not path.is_file():
			msg = f'path-list entry does not exist: {path}'
			raise FileNotFoundError(msg)
		key = path.resolve(strict=True)
		if key in seen:
			msg = f'duplicate path-list entry: {seen[key]} and {path}'
			raise ValueError(msg)
		seen[key] = path
		paths.append(path)
	return paths


def make_survey_id_from_path(path: str | Path, nopims_root: str | Path) -> str:
	"""Create a deterministic, collision-safe survey id from an `.npy` path."""
	volume_path = Path(path)
	root = Path(nopims_root)
	try:
		source = volume_path.resolve(strict=False).relative_to(
			root.resolve(strict=False),
		)
	except ValueError:
		source = volume_path.resolve(strict=False)

	source_without_suffix = source.with_suffix('')
	raw_id = '__'.join(source_without_suffix.parts)
	base_id = _SURVEY_ID_ALLOWED.sub('_', raw_id).strip('._-')
	if not base_id:
		base_id = 'volume'
	digest = hashlib.sha256(
		volume_path.resolve(strict=False).as_posix().encode('utf-8'),
	).hexdigest()[:12]
	return f'{base_id}__{digest}'


__all__ = [
	'load_npy_path_list',
	'make_survey_id_from_path',
	'resolve_npy_path_list',
]
