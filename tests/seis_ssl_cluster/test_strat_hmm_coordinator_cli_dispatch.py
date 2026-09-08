"""Keep exact-survey HMM coordination fail-closed at the thin CLI boundary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
	from types import ModuleType

SCRIPT = (
	Path(__file__).resolve().parents[2]
	/ 'proc/seis_ssl_cluster/train_strat_hmm_pretext.py'
)


def _prepare_cli(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	arguments: list[str],
) -> tuple[ModuleType, Path, dict[str, object]]:
	spec = importlib.util.spec_from_file_location('hmm_dispatch_cli_test', SCRIPT)
	assert spec is not None
	assert spec.loader is not None
	cli = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(cli)
	config_path = tmp_path / 'config.yaml'
	config_path.write_text('{}\n', encoding='utf-8')
	resolved: dict[str, object] = {'train': {'max_steps': None}}
	monkeypatch.setattr(cli, 'load_config_for_cli', lambda *_args, **_kwargs: {})
	monkeypatch.setattr(
		cli, 'resolve_config_for_cli', lambda *_args, **_kwargs: resolved
	)
	monkeypatch.setattr(cli, 'print_config_summary', lambda *_args, **_kwargs: None)
	monkeypatch.setattr(
		sys, 'argv', [str(SCRIPT), '--config', str(config_path), *arguments]
	)
	return cli, config_path, resolved


@pytest.mark.parametrize(
	('handled_by', 'expected_calls'),
	[
		('parihaka', ['parihaka']),
		('volve', ['parihaka', 'volve']),
		('generic', ['parihaka', 'volve', 'generic']),
		('error', ['parihaka', 'volve']),
	],
)
def test_coordinator_dispatch_and_argument_forwarding(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
	handled_by: str,
	expected_calls: list[str],
) -> None:
	resume = tmp_path / 'latest.pt'
	resume.write_bytes(b'Only existence is needed by this mocked CLI contract.')
	cli, config_path, resolved = _prepare_cli(
		tmp_path,
		monkeypatch,
		['--resume', str(resume), '--quarantine-invalid'],
	)
	calls: list[tuple[str, dict[str, object]]] = []
	result = tmp_path / 'result.pt'

	def parihaka(config: dict[str, object], **kwargs: object) -> Path | None:
		assert config is resolved
		calls.append(('parihaka', kwargs))
		return result if handled_by == 'parihaka' else None

	def volve(config: dict[str, object], **kwargs: object) -> Path | None:
		assert config is resolved
		calls.append(('volve', kwargs))
		if handled_by == 'error':
			raise ValueError('foreign Volve checkpoint must not fall back')
		return result if handled_by == 'volve' else None

	def generic(config: dict[str, object], **kwargs: object) -> Path:
		assert config is resolved
		calls.append(('generic', kwargs))
		return result

	monkeypatch.setattr(cli, 'run_coordinated_hmm_if_scoped', parihaka)
	monkeypatch.setattr(cli, 'run_coordinated_volve_hmm_if_scoped', volve)
	monkeypatch.setattr(cli, 'run_strat_hmm_pretext_training', generic)
	if handled_by == 'error':
		with pytest.raises(ValueError, match='must not fall back'):
			cli.main()
		assert 'checkpoint:' not in capsys.readouterr().out
	else:
		cli.main()
		assert f'checkpoint: {result}' in capsys.readouterr().out
	assert [name for name, _ in calls] == expected_calls
	for name, kwargs in calls:
		expected: dict[str, object] = {
			'resume': resume,
			'quarantine_invalid': True,
		}
		if name != 'generic':
			expected['config_path'] = config_path
		assert kwargs == expected


def test_dry_run_does_not_call_any_coordinator_or_trainer(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	cli, _, _ = _prepare_cli(tmp_path, monkeypatch, ['--dry-run'])

	def forbidden(*_args: object, **_kwargs: object) -> None:
		pytest.fail('dry-run must not enter a coordinator or trainer')

	for name in (
		'run_coordinated_hmm_if_scoped',
		'run_coordinated_volve_hmm_if_scoped',
		'run_strat_hmm_pretext_training',
	):
		monkeypatch.setattr(cli, name, forbidden)
	cli.main()
	assert 'execution: dry-run; training skipped' in capsys.readouterr().out
	assert {path.name for path in tmp_path.iterdir()} == {'config.yaml'}
