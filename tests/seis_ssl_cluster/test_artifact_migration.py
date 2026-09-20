"""Filesystem contracts for reversible, content-preserving artifact relocation."""

from __future__ import annotations

import fcntl
import json
from dataclasses import replace
from pathlib import Path

import pytest

from seis_ssl_cluster.artifact_migration import (
	ArtifactMove,
	apply_moves,
	inspect_moves,
	rollback_moves,
	validate_moves,
)


def _move(root, source='old/run', destination='surveys/f3/run'):
	path = root / source
	path.mkdir(parents=True)
	(path / 'checkpoint.pt').write_bytes(b'opaque checkpoint: keep bytes and inode')
	info = path.stat()
	return ArtifactMove(source, destination, info.st_dev, info.st_ino)


def test_move_keeps_historical_reads_and_can_rollback_new_files(tmp_path):
	move = _move(tmp_path)
	journal = tmp_path / 'operations/state.json'
	checkpoint = tmp_path / move.source / 'checkpoint.pt'
	original_inode = checkpoint.stat().st_ino
	assert inspect_moves(tmp_path, [move])['status'] == 'ready'
	assert not journal.parent.exists()
	assert not (tmp_path / '.artifact_migration.lock').exists()

	apply_moves(tmp_path, [move], journal)
	assert (tmp_path / move.source).is_symlink()
	assert not Path((tmp_path / move.source).readlink()).is_absolute()
	assert checkpoint.stat().st_ino == original_inode
	assert checkpoint.read_bytes() == b'opaque checkpoint: keep bytes and inode'
	(checkpoint.parent / 'later.json').write_text('{}')
	apply_moves(tmp_path, [move], journal)
	rollback_moves(tmp_path, [move], journal)
	assert not (tmp_path / move.source).is_symlink()
	assert checkpoint.stat().st_ino == original_inode
	assert (checkpoint.parent / 'later.json').read_text() == '{}'
	assert not (tmp_path / move.destination).exists()
	rollback_moves(tmp_path, [move], journal)


@pytest.mark.parametrize('destination', ['/outside', '../outside', 'a/../b', '.', ''])
def test_rejects_invalid_paths_before_any_move(tmp_path, destination):
	move = replace(_move(tmp_path), destination=destination)
	with pytest.raises(ValueError, match='root-relative'):
		apply_moves(tmp_path, [move], tmp_path / 'state.json')
	assert (tmp_path / move.source / 'checkpoint.pt').exists()
	assert not (tmp_path / 'state.json').exists()


@pytest.mark.parametrize('collision', ['file', 'directory', 'dangling_symlink'])
def test_rejects_destination_collisions_before_moving_any_source(tmp_path, collision):
	first = _move(tmp_path)
	second = _move(tmp_path, 'old/second', 'surveys/f3/second')
	destination = tmp_path / second.destination
	destination.parent.mkdir(parents=True)
	if collision == 'directory':
		destination.mkdir()
	elif collision == 'file':
		destination.write_text('must survive')
	else:
		destination.symlink_to('missing')
	with pytest.raises(FileExistsError):
		apply_moves(tmp_path, [first, second], tmp_path / 'state.json')
	assert not (tmp_path / first.source).is_symlink()
	assert not (tmp_path / first.destination).exists()
	assert destination.exists() or destination.is_symlink()


def test_rejects_overlaps_and_identity_drift(tmp_path):
	move = _move(tmp_path)
	for other in [
		replace(move, source='old/run/child', destination='surveys/other'),
		replace(move, source='other', destination='surveys/f3/run/child'),
		replace(move, source='surveys/f3/run', destination='other'),
	]:
		with pytest.raises(ValueError, match='overlapping'):
			validate_moves([move, other])
	with pytest.raises(ValueError, match='identity changed'):
		inspect_moves(tmp_path, [replace(move, inode=move.inode + 1)])


def test_rejects_symlink_escape_and_journal_inside_payload(tmp_path):
	root = tmp_path / 'root'
	root.mkdir()
	outside = tmp_path / 'outside'
	outside.mkdir()
	move = _move(root)
	(root / 'escape').symlink_to(outside, target_is_directory=True)
	with pytest.raises(ValueError, match='escapes'):
		inspect_moves(root, [replace(move, destination='escape/run')])
	with pytest.raises(ValueError, match='journal must be outside'):
		apply_moves(root, [move], root / move.source / 'journal.json')


def test_recovers_interruption_between_rename_and_alias(tmp_path, monkeypatch):
	move = _move(tmp_path)
	journal = tmp_path / 'operations/state.json'
	original_symlink = Path.symlink_to

	def interrupted(*_args, **_kwargs):
		raise OSError('simulated interruption')

	monkeypatch.setattr(Path, 'symlink_to', interrupted)
	with pytest.raises(OSError, match='simulated'):
		apply_moves(tmp_path, [move], journal)
	assert not (tmp_path / move.source).exists()
	assert (tmp_path / move.destination).exists()
	assert json.loads(journal.read_text())['entries'][0]['phase'] == 'move_intent'
	monkeypatch.setattr(Path, 'symlink_to', original_symlink)
	apply_moves(tmp_path, [move], journal)
	assert (tmp_path / move.source / 'checkpoint.pt').exists()
	rollback_moves(tmp_path, [move], journal)
	assert not (tmp_path / move.source).is_symlink()


def test_file_move_and_relative_alias_under_legacy_parent(tmp_path):
	move = _move(tmp_path)
	top = tmp_path / 'old'
	top_info = top.stat()
	top_move = ArtifactMove('old', '.legacy/old', top_info.st_dev, top_info.st_ino)
	apply_moves(tmp_path, [top_move], tmp_path / 'operations/view.json')
	apply_moves(tmp_path, [move], tmp_path / 'operations/units.json')
	assert (tmp_path / 'old/run/checkpoint.pt').read_bytes().startswith(b'opaque')
	checkpoint = tmp_path / move.destination / 'checkpoint.pt'
	info = checkpoint.stat()
	file_move = ArtifactMove(
		'surveys/f3/run/checkpoint.pt',
		'archive/checkpoint.pt',
		info.st_dev,
		info.st_ino,
	)
	apply_moves(tmp_path, [file_move], tmp_path / 'operations/file.json')
	assert (tmp_path / 'old/run/checkpoint.pt').read_bytes().startswith(b'opaque')
	rollback_moves(tmp_path, [file_move], tmp_path / 'operations/file.json')
	rollback_moves(tmp_path, [move], tmp_path / 'operations/units.json')
	rollback_moves(tmp_path, [top_move], tmp_path / 'operations/view.json')
	assert not top.is_symlink()
	assert (top / 'run/checkpoint.pt').read_bytes().startswith(b'opaque')


def test_rejects_plan_drift_unrecorded_symlink_and_concurrent_migrator(tmp_path):
	move = _move(tmp_path)
	journal = tmp_path / 'operations/state.json'
	with (tmp_path / '.artifact_migration.lock').open('w') as stream:
		fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
		with pytest.raises(RuntimeError, match='holds the root lock'):
			apply_moves(tmp_path, [move], journal)
	apply_moves(tmp_path, [move], journal)
	with pytest.raises(ValueError, match='journal does not match'):
		apply_moves(tmp_path, [replace(move, destination='different')], journal)
	with pytest.raises(ValueError, match='unexpected source symlink'):
		apply_moves(tmp_path, [move], tmp_path / 'other.json')
	rollback_moves(tmp_path, [move], journal)
	with pytest.raises(ValueError, match='finish rollback'):
		apply_moves(tmp_path, [move], journal)
