from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from seis_ssl_cluster.cli import (
	add_config_argument,
	add_device_argument,
	add_overwrite_argument,
	add_skip_existing_argument,
	load_config_for_cli,
	parse_config_path,
	print_cli_summary,
	resolve_config_for_cli,
)


def test_config_argument_parses_path() -> None:
	parser = argparse.ArgumentParser()
	add_config_argument(parser)

	args = parser.parse_args(['--config', 'config.yaml'])

	assert parse_config_path(args) == Path('config.yaml')


def test_load_config_for_cli_reports_missing_config(tmp_path: Path) -> None:
	config_path = tmp_path / 'missing.yaml'

	with pytest.raises(
		FileNotFoundError,
		match=r'config file does not exist: .*missing.yaml',
	):
		load_config_for_cli(config_path, loader=lambda path: {'path': path})


def test_load_config_for_cli_returns_loader_mapping(tmp_path: Path) -> None:
	config_path = tmp_path / 'config.yaml'
	config_path.write_text('paths: {}\n', encoding='utf-8')

	def loader(path: Path) -> dict[str, object]:
		return {'config': path}

	assert load_config_for_cli(config_path, loader=loader) == {'config': config_path}


def test_print_cli_summary_outputs_stable_lines(
	capsys: pytest.CaptureFixture[str],
) -> None:
	print_cli_summary(
		'F3 lithology probe',
		{
			'config': Path('experiment.yaml'),
			'output_dir': Path('artifacts/probe'),
			'dry_run': True,
			'tags': ['f3', 'probe'],
			'checkpoint': None,
		},
	)

	assert capsys.readouterr().out == (
		'F3 lithology probe\n'
		'config: experiment.yaml\n'
		'output_dir: artifacts/probe\n'
		'dry_run: true\n'
		'tags: f3, probe\n'
		'checkpoint: null\n'
	)


def test_common_flag_helpers_add_existing_flag_names() -> None:
	parser = argparse.ArgumentParser()
	add_device_argument(parser)
	add_skip_existing_argument(parser)
	add_overwrite_argument(parser)

	args = parser.parse_args(['--device', 'cpu', '--skip-existing', '--overwrite'])

	assert args.device == 'cpu'
	assert args.skip_existing is True
	assert args.overwrite is True


def test_resolve_config_for_cli_returns_resolved_value() -> None:
	def resolver(raw_config: dict[str, object]) -> tuple[str, object]:
		return ('resolved', raw_config['value'])

	assert resolve_config_for_cli(
		{'value': 3},
		resolver=resolver,
		config_path=Path('config.yaml'),
	) == ('resolved', 3)


def test_resolve_config_for_cli_includes_config_path_on_failure() -> None:
	def resolver(_raw_config: dict[str, object]) -> object:
		raise ValueError('invalid probe config')

	with pytest.raises(ValueError, match=r'invalid probe config .*config: config.yaml'):
		resolve_config_for_cli({}, resolver=resolver, config_path=Path('config.yaml'))
