"""Thin CLI cooperation hook ordering and unrelated invocation compatibility."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from seis_ssl_cluster.parihaka import coordinated_channel as coordination

CLI = (
	Path(__file__).resolve().parents[2]
	/ 'proc/seis_ssl_cluster/run_parihaka_channel_decoder.py'
)


def test_public_cli_import_identity_supports_aux_launch_guard():
	namespace = runpy.run_path(str(CLI), run_name='coordinated_channel_cli_contract')
	assert (
		namespace['run_coordinated_channel_if_scoped']
		is coordination.run_coordinated_channel_if_scoped
	)


def _namespace(monkeypatch):
	namespace = runpy.run_path(str(CLI), run_name='channel_cli_test')
	monkeypatch.setattr(
		sys,
		'argv',
		[
			str(CLI),
			'--config',
			'/unused/config.yaml',
			'--model',
			'test',
			'--layout',
			'layout_000',
			'--size',
			'small',
			'--layout-config',
			'/unused/layouts.yaml',
			'--device',
			'cpu',
		],
	)
	functions = namespace['main'].__globals__
	monkeypatch.setitem(
		functions, 'load_config_for_cli', lambda *_args, **_kwargs: {'raw': True}
	)
	return namespace['main'], functions


def test_cli_coordinator_precedes_plan_and_scientific_runner(monkeypatch, capsys):
	main, functions = _namespace(monkeypatch)
	calls = []

	def hook(raw, **kwargs):
		assert raw == {'raw': True}
		assert kwargs['device'] == 'cpu'
		assert kwargs['layout_id'] == 'layout_000'
		assert kwargs['data_size'] == 'small'
		calls.append('hook')
		return Path('/owned/metrics.json')

	monkeypatch.setitem(functions, 'run_coordinated_channel_if_scoped', hook)
	monkeypatch.setitem(
		functions,
		'channel_decoder_config_from_mapping',
		lambda *_: pytest.fail('old plan reached'),
	)
	monkeypatch.setitem(
		functions,
		'run_channel_decoder_job',
		lambda *_args, **_kwargs: pytest.fail('old runner reached'),
	)
	main()
	assert calls == ['hook']
	assert 'metrics: /owned/metrics.json' in capsys.readouterr().out


def test_unrelated_none_return_reaches_existing_runner(monkeypatch):
	main, functions = _namespace(monkeypatch)
	calls = []

	def hook(*_args, **_kwargs):
		calls.append('hook')

	def inspect(*_args, **_kwargs):
		calls.append('plan')
		return SimpleNamespace(output_dir=Path('/unrelated'))

	def run(*_args, **_kwargs):
		calls.append('runner')
		return Path('/unrelated/metrics.json')

	monkeypatch.setitem(functions, 'run_coordinated_channel_if_scoped', hook)
	monkeypatch.setitem(
		functions, 'channel_decoder_config_from_mapping', lambda raw: raw
	)
	monkeypatch.setitem(functions, 'inspect_channel_decoder_job', inspect)
	monkeypatch.setitem(functions, 'run_channel_decoder_job', run)
	main()
	assert calls == ['hook', 'plan', 'runner']
