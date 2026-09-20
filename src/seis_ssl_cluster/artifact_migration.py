"""Move local artifacts without rewriting their contents or historical paths."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from collections.abc import Iterator, Sequence


@dataclass(frozen=True)
class ArtifactMove:
	"""One same-filesystem move, bound to the original object's identity."""

	source: str
	destination: str
	device: int
	inode: int


def _relative(value: str) -> PurePosixPath:
	path = PurePosixPath(value)
	if (
		not value
		or path.is_absolute()
		or '..' in path.parts
		or path == PurePosixPath('.')
		or path.as_posix() != value
	):
		raise ValueError(f'expected a normalized root-relative path: {value!r}')
	return path


def validate_moves(moves: Sequence[ArtifactMove]) -> None:
	"""Reject overlapping sources, destinations, and invalid relative paths."""
	paths = [
		_relative(value) for move in moves for value in (move.source, move.destination)
	]
	for index, path in enumerate(paths):
		for other in paths[index + 1 :]:
			if path.is_relative_to(other) or other.is_relative_to(path):
				raise ValueError(f'overlapping migration paths: {path}, {other}')


def _exists(path: Path) -> bool:
	return path.exists() or path.is_symlink()


def _identity(path: Path, move: ArtifactMove) -> None:
	if path.is_symlink():
		raise ValueError(f'payload root unexpectedly became a symlink: {path}')
	info = path.stat()
	if (info.st_dev, info.st_ino) != (move.device, move.inode):
		raise ValueError(f'payload identity changed: {path}')


def _inside(root: Path, path: Path) -> None:
	if not path.parent.resolve().is_relative_to(root):
		raise ValueError(f'migration path escapes artifact root: {path}')


def _same_filesystem(destination: Path, move: ArtifactMove) -> None:
	parent = destination.parent
	while not parent.exists():
		parent = parent.parent
	if parent.stat().st_dev != move.device:
		raise ValueError(f'migration must remain on the same filesystem: {destination}')


def _placement(root: Path, move: ArtifactMove, *, recorded: bool) -> str:
	source, destination = root / move.source, root / move.destination
	_inside(root, source)
	_inside(root, destination)
	_same_filesystem(destination, move)
	if source.is_symlink():
		if not recorded or source.resolve() != destination.resolve():
			raise ValueError(f'unexpected source symlink: {source}')
		_identity(destination, move)
		return 'linked'
	if _exists(source):
		_identity(source, move)
		if _exists(destination):
			raise FileExistsError(
				f'migration destination already exists: {destination}'
			)
		return 'original'
	if recorded and _exists(destination):
		_identity(destination, move)
		return 'moved'
	raise FileNotFoundError(f'migration source is missing: {source}')


def inspect_moves(root: Path, moves: Sequence[ArtifactMove]) -> dict[str, Any]:
	"""Validate a fresh migration without creating directories, links, or a journal."""
	root = root.resolve(strict=True)
	validate_moves(moves)
	for move in moves:
		_placement(root, move, recorded=False)
	return {'status': 'ready', 'moves': len(moves), 'mode': 'dry-run'}


def _digest(moves: Sequence[ArtifactMove]) -> str:
	payload = json.dumps([asdict(move) for move in moves], sort_keys=True).encode()
	return hashlib.sha256(payload).hexdigest()


def _save(path: Path, state: dict[str, Any]) -> None:
	temporary = path.with_name(path.name + '.tmp')
	with temporary.open('w', encoding='utf-8') as stream:
		json.dump(state, stream, indent=2, sort_keys=True)
		stream.write('\n')
		stream.flush()
		os.fsync(stream.fileno())
	temporary.replace(path)


@contextmanager
def _lock(root: Path) -> Iterator[None]:
	lock_path = root / '.artifact_migration.lock'
	with lock_path.open('a', encoding='utf-8') as stream:
		try:
			fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
		except BlockingIOError as error:
			raise RuntimeError(
				'another artifact migration holds the root lock'
			) from error
		try:
			yield
		finally:
			fcntl.flock(stream, fcntl.LOCK_UN)


def _state(
	root: Path,
	moves: Sequence[ArtifactMove],
	journal: Path,
) -> dict[str, Any]:
	if journal.exists():
		state = json.loads(journal.read_text(encoding='utf-8'))
		if (
			state.get('schema_version') != 1
			or state.get('plan_sha256') != _digest(moves)
			or state.get('artifact_root') != str(root)
			or len(state.get('entries', [])) != len(moves)
		):
			raise ValueError('migration journal does not match this root and plan')
		return state
	return {
		'schema_version': 1,
		'artifact_root': str(root),
		'plan_sha256': _digest(moves),
		'entries': [{**asdict(move), 'phase': 'pending'} for move in moves],
	}


def _journal_outside_moves(
	root: Path,
	moves: Sequence[ArtifactMove],
	journal: Path,
) -> None:
	for path in (journal, journal.with_name(journal.name + '.tmp')):
		for move in moves:
			for relative in (move.source, move.destination):
				if path.resolve().is_relative_to((root / relative).resolve()):
					raise ValueError('journal must be outside the payloads being moved')


def apply_moves(
	root: Path,
	moves: Sequence[ArtifactMove],
	journal: Path,
) -> dict[str, Any]:
	"""Rename payloads and leave relative compatibility links; resume from a journal.

	The caller must arrange a maintenance window without artifact writers.
	The lock coordinates this tool, not arbitrary training or shell processes.
	Every destination must be absent; there is no merge or overwrite mode.
	"""
	root = root.resolve(strict=True)
	journal = journal.resolve()
	validate_moves(moves)
	_journal_outside_moves(root, moves, journal)
	with _lock(root):
		state = _state(root, moves, journal)
		if any(entry['phase'].startswith('rollback') for entry in state['entries']):
			raise ValueError('finish rollback before creating another migration plan')
		for move, entry in zip(moves, state['entries'], strict=True):
			_placement(root, move, recorded=entry['phase'] != 'pending')
		journal.parent.mkdir(parents=True, exist_ok=True)
		_save(journal, state)
		for move, entry in zip(moves, state['entries'], strict=True):
			source, destination = root / move.source, root / move.destination
			placement = _placement(root, move, recorded=entry['phase'] != 'pending')
			if placement == 'linked':
				entry['phase'] = 'linked'
				_save(journal, state)
				continue
			entry['phase'] = 'move_intent'
			_save(journal, state)
			if placement == 'original':
				destination.parent.mkdir(parents=True, exist_ok=True)
				if _exists(destination):
					raise FileExistsError(
						f'destination appeared during migration: {destination}'
					)
				source.rename(destination)
			# Resolve the parent: it may itself be reached through a legacy alias.
			target = os.path.relpath(destination, source.parent.resolve())
			source.symlink_to(target, target_is_directory=destination.is_dir())
			entry['phase'] = 'linked'
			_save(journal, state)
	return {'status': 'complete', 'moves': len(moves), 'journal': str(journal)}


def rollback_moves(
	root: Path,
	moves: Sequence[ArtifactMove],
	journal: Path,
) -> dict[str, Any]:
	"""Restore original locations, removing only links created by this migration."""
	root = root.resolve(strict=True)
	journal = journal.resolve()
	validate_moves(moves)
	_journal_outside_moves(root, moves, journal)
	if not journal.is_file():
		raise FileNotFoundError('rollback requires the original migration journal')
	with _lock(root):
		state = _state(root, moves, journal)
		for move, entry in zip(moves, state['entries'], strict=True):
			_placement(root, move, recorded=entry['phase'] != 'pending')
		for move, entry in reversed(list(zip(moves, state['entries'], strict=True))):
			source, destination = root / move.source, root / move.destination
			placement = _placement(root, move, recorded=entry['phase'] != 'pending')
			entry['phase'] = 'rollback_intent'
			_save(journal, state)
			if placement == 'linked':
				source.unlink()
			if placement != 'original':
				if _exists(source):
					raise FileExistsError(f'original location is occupied: {source}')
				destination.rename(source)
			entry['phase'] = 'rollback_complete'
			_save(journal, state)
	return {'status': 'rolled_back', 'moves': len(moves), 'journal': str(journal)}


def main() -> None:
	"""Inspect, apply, or roll back an explicit local artifact move plan."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--artifact-root', type=Path, required=True)
	parser.add_argument('--plan', type=Path, required=True)
	parser.add_argument('--journal', type=Path)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--dry-run', action='store_true')
	mode.add_argument('--apply', action='store_true')
	mode.add_argument('--rollback', action='store_true')
	args = parser.parse_args()
	moves = [
		ArtifactMove(**entry)
		for entry in json.loads(args.plan.read_text(encoding='utf-8'))
	]
	if args.apply or args.rollback:
		if args.journal is None:
			parser.error('--journal is required for --apply or --rollback')
		action = rollback_moves if args.rollback else apply_moves
		result = action(args.artifact_root, moves, args.journal)
	else:
		result = inspect_moves(args.artifact_root, moves)
	print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
