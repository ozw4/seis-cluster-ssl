from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from seis_ssl_cluster.cli import (
	add_append_path_argument,
	add_config_argument,
	add_device_argument,
	add_overwrite_argument,
	add_path_argument,
	add_skip_existing_argument,
	add_store_true_argument,
	build_config_parser,
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


def test_build_config_parser_preserves_default_config_and_dry_run() -> None:
	parser = build_config_parser(
		'Example command.',
		default_config=Path('default.yaml'),
		dry_run_help='Validate without writing outputs.',
	)

	args = parser.parse_args(['--dry-run'])

	assert args.config == Path('default.yaml')
	assert args.dry_run is True


def test_build_config_parser_can_require_config() -> None:
	parser = build_config_parser(
		'Example command.',
		config_help='Path to a required YAML configuration file.',
	)

	with pytest.raises(SystemExit):
		parser.parse_args([])

	args = parser.parse_args(['--config', 'required.yaml'])
	assert args.config == Path('required.yaml')


def test_build_config_parser_can_leave_config_optional() -> None:
	parser = build_config_parser(
		'Example command.',
		default_config=None,
		config_required=False,
	)

	args = parser.parse_args([])

	assert args.config is None


def test_generic_argument_helpers_preserve_option_shapes() -> None:
	parser = argparse.ArgumentParser()
	add_path_argument(parser, '--root', default=Path('reports'), help_text='Root.')
	add_append_path_argument(
		parser,
		'--required-file',
		help_text='Required file.',
	)
	add_store_true_argument(parser, '--fail-on-runs', help_text='Fail on runs.')

	args = parser.parse_args(
		[
			'--root',
			'other-results',
			'--required-file',
			'a.json',
			'--required-file',
			'b.md',
			'--fail-on-runs',
		],
	)

	assert args.root == Path('other-results')
	assert args.required_file == [Path('a.json'), Path('b.md')]
	assert args.fail_on_runs is True


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
