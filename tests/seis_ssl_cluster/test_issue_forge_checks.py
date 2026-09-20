from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

HOOK = Path(__file__).resolve().parents[2] / '.issue_forge/checks/run_changed.sh'
PORTABLE_ARGS = ['-q', '-m', 'not slow and not requires_segy and not requires_cuda']


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(  # noqa: S603 - fixed commands in a disposable Git fixture
		args,
		cwd=repo,
		capture_output=True,
		text=True,
		check=False,
	)


def _git(repo: Path, *args: str) -> str:
	result = _run(repo, 'git', *args)
	assert result.returncode == 0, result.stderr
	return result.stdout


@pytest.fixture
def checks_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	repo = tmp_path / 'repo'
	repo.mkdir()
	_git(repo, 'init', '-q')
	_git(repo, 'config', 'user.name', 'Checks Test')
	_git(repo, 'config', 'user.email', 'checks@example.invalid')
	_git(repo, 'config', 'commit.gpgsign', 'false')
	(repo / 'README.md').write_text('Baseline\n')
	old_test = repo / 'tests/test_existing.py'
	old_test.parent.mkdir()
	old_test.write_text('def test_existing():\n    pass\n')
	_git(repo, 'add', '.')
	_git(repo, 'commit', '-qm', 'Baseline')
	bin_dir = tmp_path / 'bin'
	bin_dir.mkdir()
	for command in ('pytest', 'shellcheck'):
		stub = bin_dir / command
		stub.write_text(
			'#!/usr/bin/env bash\n'
			f'printf "%s\\n" {command} "$@" >> "$CHECKS_CALL_LOG"\n'
			f'exit "${{{command.upper()}_EXIT_CODE:-0}}"\n',
		)
		stub.chmod(0o755)
	monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
	monkeypatch.setenv('CHECKS_CALL_LOG', str(tmp_path / 'calls.log'))
	monkeypatch.setenv('PYTEST_EXIT_CODE', '0')
	monkeypatch.setenv('SHELLCHECK_EXIT_CODE', '0')
	return repo


def _calls(repo: Path) -> list[str]:
	log = repo.parent / 'calls.log'
	return log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize(
	('paths', 'expected'),
	[
		([], []),
		(['docs/note.md'], []),
		(['.work/ignored.py', 'vendor/issue_forge/ignored.py'], []),
		(['tests/test_new.py'], ['pytest', '-q', 'tests/test_new.py']),
		(
			['tests/seis_ssl_cluster/test_new.py', 'docs/note.md'],
			['pytest', '-q', 'tests/seis_ssl_cluster/test_new.py'],
		),
		(['src/seis_ssl_cluster/example.py'], ['pytest', *PORTABLE_ARGS]),
		(['tests/conftest.py'], ['pytest', *PORTABLE_ARGS]),
		(['tests/fixtures/input.json'], ['pytest', *PORTABLE_ARGS]),
		(['pyproject.toml'], ['pytest', '-q']),
		(['setup.py'], ['pytest', '-q']),
		(['pytest.ini', 'tests/test_new.py'], ['pytest', '-q']),
		(['requirements-dev.txt', 'src/example.py'], ['pytest', '-q']),
		(['scripts/example.sh'], ['shellcheck', '-x', 'scripts/example.sh']),
		(
			[
				'.issue_forge/checks/run_changed.sh',
				'tests/seis_ssl_cluster/test_issue_forge_checks.py',
			],
			[
				'shellcheck',
				'-x',
				'.issue_forge/checks/run_changed.sh',
				'pytest',
				'-q',
				'tests/seis_ssl_cluster/test_issue_forge_checks.py',
			],
		),
	],
)
def test_checks_select_validation_without_changing_worktree(
	checks_repo: Path,
	paths: list[str],
	expected: list[str],
) -> None:
	for path in paths:
		file = checks_repo / path
		file.parent.mkdir(parents=True, exist_ok=True)
		file.write_text('Changed\n')
	before = _git(checks_repo, 'status', '--porcelain')
	result = _run(checks_repo, 'bash', str(HOOK), 'HEAD')
	assert result.returncode == 0, result.stderr
	assert _calls(checks_repo) == expected
	assert _git(checks_repo, 'status', '--porcelain') == before


@pytest.mark.parametrize('state', ['committed', 'staged', 'unstaged', 'deleted'])
def test_checks_include_changes_since_fixed_base(checks_repo: Path, state: str) -> None:
	base = _git(checks_repo, 'rev-parse', 'HEAD').strip()
	test_file = checks_repo / 'tests/test_existing.py'
	if state == 'deleted':
		test_file.unlink()
	else:
		test_file.write_text('def test_existing():\n    assert True\n')
	if state in {'staged', 'committed'}:
		_git(checks_repo, 'add', '.')
	if state == 'committed':
		_git(checks_repo, 'commit', '-qm', 'Change test')
	result = _run(checks_repo, 'bash', str(HOOK), base)
	assert result.returncode == 0, result.stderr
	expected = (
		['pytest', *PORTABLE_ARGS]
		if state == 'deleted'
		else ['pytest', '-q', 'tests/test_existing.py']
	)
	assert _calls(checks_repo) == expected


@pytest.mark.parametrize('command', ['pytest', 'shellcheck'])
def test_checks_propagate_failure(
	checks_repo: Path,
	monkeypatch: pytest.MonkeyPatch,
	command: str,
) -> None:
	(checks_repo / 'example.py').write_text('pass\n')
	(checks_repo / 'example.sh').write_text('#!/usr/bin/env bash\ntrue\n')
	monkeypatch.setenv(f'{command.upper()}_EXIT_CODE', '7')
	result = _run(checks_repo, 'bash', str(HOOK), 'HEAD')
	assert result.returncode == 7
	if command == 'shellcheck':
		assert 'pytest' not in _calls(checks_repo)


@pytest.mark.parametrize('args', [[], ['HEAD', 'extra'], ['missing-base']])
def test_checks_reject_invalid_invocation(checks_repo: Path, args: list[str]) -> None:
	result = _run(checks_repo, 'bash', str(HOOK), *args)
	assert result.returncode != 0
	assert _calls(checks_repo) == []
	assert 'Usage:' in result.stderr or 'Missing base ref' in result.stderr
