from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_clustering_config

BASELINE = Path(
	'experiments/f3/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/20_hmm_targets/mae100/k6'
)
TARGETS = Path(
	'experiments/f3/facies_benchmark_v2/127_hmm_v2_multi_head_screening_v1/10_targets'
)
BASH = shutil.which('bash') or 'bash'
NAMESPACE = 'f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1/mae100'
RECIPES = (
	('01_cluster_hmm_k4810.yaml', [4, 8, 10], 'k4810'),
	('02_replay_hmm_k6.yaml', [6], 'k6_replay'),
)


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifact store'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


@pytest.mark.parametrize(('filename', 'ks', 'suffix'), RECIPES)
def test_target_configs_preserve_baseline(
	artifact_root: Path, filename: str, ks: list[int], suffix: str
) -> None:
	baseline = load_config(BASELINE / '02_cluster_hmm_k6.yaml')
	actual = load_config(TARGETS / filename)
	expected_root = artifact_root / 'clustering' / NAMESPACE / suffix
	assert actual['clustering']['k_values'] == ks
	assert Path(actual['clustering']['output_dir']) == expected_root
	immutable = Path(baseline['clustering']['output_dir'])
	assert not expected_root.is_relative_to(immutable)
	assert not immutable.is_relative_to(expected_root)
	baseline['clustering']['k_values'] = ks
	baseline['clustering']['output_dir'] = str(expected_root)
	assert actual == baseline
	assert resolve_clustering_config(actual) == resolve_clustering_config(baseline)
	assert {p.name for p in TARGETS.glob('*.yaml')} == {r[0] for r in RECIPES}


@pytest.fixture
def command_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	log = tmp_path / 'commands.jsonl'
	fake_python = tmp_path / 'python'
	fake_python.write_text(
		f'#!{sys.executable}\n'
		'import json, os, sys\n'
		'with open(os.environ["COMMAND_LOG"], "a") as stream:\n'
		'    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
		'sys.exit(int(os.environ.get("EXPORT_EXIT", "0")))\n',
		encoding='utf-8',
	)
	fake_python.chmod(0o755)
	monkeypatch.setenv('PATH', f'{tmp_path}{os.pathsep}{os.environ["PATH"]}')
	monkeypatch.setenv('COMMAND_LOG', str(log))
	return log


@pytest.mark.parametrize('flags', [[], ['--dry-run']])
def test_export_wrapper_uses_only_baseline_hard_export_contract(
	artifact_root: Path, command_capture: Path, flags: list[str]
) -> None:
	subprocess.run(  # noqa: S603 - repository wrapper and fixed test arguments
		[BASH, str(TARGETS / '03_export_pseudo_targets.sh'), *flags], check=True
	)
	commands = [json.loads(line) for line in command_capture.read_text().splitlines()]
	baseline_tokens = shlex.split(
		os.path.expandvars(
			(BASELINE / '03_export_pseudo_targets.sh').read_text()
		).replace('\\\n', '')
	)
	start = baseline_tokens.index('--clustering-output-dir')
	baseline_args = dict(
		zip(baseline_tokens[start::2], baseline_tokens[start + 1 :: 2], strict=True)
	)
	assert len(commands) == 4
	for k, command in zip((4, 8, 10, 6), commands, strict=True):
		source_suffix = 'k6_replay' if k == 6 else 'k4810'
		target_suffix = 'k6_replay' if k == 6 else 'shared'
		export_root = artifact_root / 'pseudo_targets' / NAMESPACE / target_suffix
		immutable = Path(baseline_args['--pseudo-target-root'])
		assert not export_root.is_relative_to(immutable)
		assert not immutable.is_relative_to(export_root)
		assert command == [
			'proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py',
			'--clustering-output-dir',
			str(artifact_root / 'clustering' / NAMESPACE / source_suffix),
			'--pseudo-target-root',
			str(export_root),
			'--k',
			str(k),
			'--confidence',
			baseline_args['--confidence'],
			'--boundary-alpha',
			baseline_args['--boundary-alpha'],
			'--boundary-tau',
			baseline_args['--boundary-tau'],
			'--schema-version',
			baseline_args['--schema-version'],
			*flags,
		]
	assert not artifact_root.exists()


@pytest.mark.parametrize(
	'flags', [['--overwrite'], ['--k', '6'], ['--dry-run', '--overwrite']]
)
def test_export_wrapper_rejects_overrides(
	artifact_root: Path, command_capture: Path, flags: list[str]
) -> None:
	del artifact_root
	result = subprocess.run(  # noqa: S603 - repository wrapper and fixed test arguments
		[BASH, str(TARGETS / '03_export_pseudo_targets.sh'), *flags],
		check=False,
		capture_output=True,
		text=True,
	)
	assert result.returncode == 2
	assert 'Usage:' in result.stderr
	assert not command_capture.exists()


def test_export_wrapper_stops_on_failure(
	artifact_root: Path, command_capture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	del artifact_root
	monkeypatch.setenv('EXPORT_EXIT', '7')
	result = subprocess.run(  # noqa: S603 - repository wrapper and fixed test arguments
		[BASH, str(TARGETS / '03_export_pseudo_targets.sh')], check=False
	)
	assert result.returncode == 7
	assert len(command_capture.read_text().splitlines()) == 1
