"""Portable shell orchestration contracts; no experiment processes are launched."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.seis_ssl_cluster.test_f3_hmm_v2_manifest_configs import (
	EXPERIMENT,
	configs,  # noqa: F401 - fixture dependency
	live_configs,  # noqa: F401 - fixture dependency
)

DRIVER = (EXPERIMENT / 'run_all.sh').resolve()
CANDIDATE = 'mae100_hmm_v2_mh_k46_distill020'


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, str]:
	binary = tmp_path / 'bin'
	binary.mkdir()
	stub = binary / 'python'
	stub.write_text(
		f'#!{sys.executable}\n'
		'import hashlib, json, os, sys\n'
		'with open(os.environ["CALLS"], "a") as f:\n'
		' value=os.getenv("SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256")\n'
		' f.write(json.dumps({"args":sys.argv[1:], "hash":value})+"\\n")\n'
		'if os.getenv("FAIL_AT") and os.environ["FAIL_AT"] in " ".join(sys.argv):\n'
		' sys.exit(17)\n'
		'if "file_sha256" in " ".join(sys.argv):\n'
		' print(hashlib.sha256(sys.argv[-1].encode()).hexdigest())\n'
	)
	stub.chmod(0o755)
	return {
		**os.environ,
		'PATH': f'{binary}:{os.environ["PATH"]}',
		'CALLS': str(tmp_path / 'calls.jsonl'),
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(tmp_path / 'artifacts with spaces'),
		'SEIS_SSL_CLUSTER_WORKSPACE': str(Path.cwd()),
		'F3_ROOT': str(tmp_path / 'f3'),
		'CUDA_VISIBLE_DEVICES': '0',
	}


def _run(environment: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(  # noqa: S603 - fixed driver with test-owned arguments
		['/bin/bash', str(DRIVER), *args],
		env=environment,
		cwd=Path(environment['CALLS']).parent,
		capture_output=True,
		text=True,
		check=False,
	)


def _calls(environment: dict[str, str]) -> list[dict[str, object]]:
	return [
		json.loads(line) for line in Path(environment['CALLS']).read_text().splitlines()
	]


def test_default_plan_needs_no_artifacts_or_python(environment: dict[str, str]) -> None:
	for name in (
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'SEIS_SSL_CLUSTER_WORKSPACE',
		'F3_ROOT',
		'CUDA_VISIBLE_DEVICES',
	):
		environment.pop(name, None)
	result = _run(environment)
	assert result.returncode == 0, result.stderr
	assert (
		result.stdout.count('proc/seis_ssl_cluster/run_f3_lithology_candidate.py') == 60
	)
	assert result.stdout.count('proc/seis_ssl_cluster/train_strat_hmm_pretext.py') == 8
	assert not Path(environment['CALLS']).exists()


def test_all_dry_runs_have_flags_distinct_hashes_and_no_logs(
	environment: dict[str, str],
) -> None:
	result = _run(environment, '--dry-run')
	assert result.returncode == 0, result.stderr
	calls = _calls(environment)
	commands = [call for call in calls if 'file_sha256' not in ' '.join(call['args'])]
	assert len(commands) == 84
	assert all('--dry-run' in call['args'] for call in commands)
	training = [
		call for call in commands if 'train_strat_hmm_pretext.py' in call['args'][0]
	]
	assert len({call['hash'] for call in training}) == 4
	assert all('--resume' not in call['args'] for call in training)
	assert not Path(environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT']).exists()


@pytest.mark.parametrize(
	('stage', 'subpath'),
	[('smoke', 'smoke_1step'), ('full', 'full_25ep'), ('downstream', 'decoder')],
)
def test_resume_uses_only_selected_runs_latest(
	environment: dict[str, str], stage: str, subpath: str
) -> None:
	args = [
		'--dry-run',
		'--from',
		stage,
		'--to',
		stage,
		'--candidate',
		CANDIDATE,
		'--resume',
	]
	if stage == 'downstream':
		args += ['--layout', 'layout_002', '--size', 'medium']
	result = _run(environment, *args)
	assert result.returncode == 0, result.stderr
	calls = [call['args'] for call in _calls(environment) if '--resume' in call['args']]
	assert len(calls) == 1
	checkpoint = calls[0][calls[0].index('--resume') + 1]
	assert checkpoint.endswith(f'{subpath}/latest.pt')
	assert CANDIDATE in checkpoint
	if stage == 'downstream':
		assert 'layout=layout_002/size=medium' in checkpoint


@pytest.mark.parametrize(
	'args',
	[
		['--from', 'unknown'],
		['--from', 'summary', '--to', 'full'],
		['--execute', '--dry-run'],
		['--candidate', 'random'],
		['--resume'],
		[
			'--from',
			'downstream',
			'--to',
			'downstream',
			'--candidate',
			CANDIDATE,
			'--resume',
		],
		['--size', 'small'],
		['--from'],
	],
)
def test_invalid_selections_fail_before_commands(
	environment: dict[str, str], args: list[str]
) -> None:
	assert _run(environment, *args).returncode == 2
	assert not Path(environment['CALLS']).exists()


def test_execution_stops_on_failure_and_preserves_separate_logs(
	environment: dict[str, str],
) -> None:
	environment['FAIL_AT'] = '02_replay_hmm_k6'
	for _ in range(2):
		result = _run(environment, '--execute', '--to', 'cluster-replay')
		assert result.returncode == 17
	logs = (
		Path(environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/logs'
	)
	events = list(logs.glob('*/events.log'))
	assert len(events) == 2
	assert all('exit=17 step=cluster-replay' in path.read_text() for path in events)
	assert len(_calls(environment)) == 4
	assert all('export' not in str(call) for call in _calls(environment))


def test_existing_cluster_output_is_never_rewritten(
	environment: dict[str, str],
) -> None:
	root = (
		Path(environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'clustering/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1'
		/ 'mae100/k4810'
	)
	root.mkdir(parents=True)
	assert _run(environment, '--execute', '--to', 'cluster-targets').returncode == 2
	assert not Path(environment['CALLS']).exists()


def test_downstream_selection_always_audits_sources_first(
	environment: dict[str, str],
) -> None:
	result = _run(
		environment,
		'--dry-run',
		'--from',
		'downstream',
		'--to',
		'downstream',
		'--candidate',
		CANDIDATE,
		'--layout',
		'layout_000',
		'--size',
		'small',
	)
	assert result.returncode == 0, result.stderr
	calls = _calls(environment)
	assert len(calls) == 2
	assert calls[0]['args'][0].endswith('audit_strat_hmm_multi_head_sources.py')
	assert calls[1]['args'][0].endswith('run_f3_lithology_candidate.py')


def test_lock_rejects_second_driver(environment: dict[str, str]) -> None:
	logs = (
		Path(environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/logs'
	)
	logs.mkdir(parents=True)
	with (logs / 'driver.lock').open('w') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		result = _run(environment, '--execute', '--to', 'cluster-targets')
	assert result.returncode == 2
	assert 'another screening driver' in result.stderr
	assert not Path(environment['CALLS']).exists()
	assert not list(logs.glob('*/events.log'))


def test_real_manifest_bridge_dry_run(
	live_configs: dict[str, object],  # noqa: F811 - imported fixture
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	env = {**os.environ, 'CALLS': str(tmp_path / 'unused-calls')}
	before = {
		path: (path.read_bytes(), path.stat().st_mtime_ns)
		for path in tmp_path.rglob('*')
		if path.is_file()
	}
	result = _run(env, '--dry-run', '--from', 'manifests', '--to', 'manifests')
	assert result.returncode == 0, result.stderr
	assert result.stdout.count('execution: dry-run; validated heads') == 4
	assert all(
		not Path(config['manifest']).exists() for config in live_configs.values()
	)
	assert before == {
		path: (path.read_bytes(), path.stat().st_mtime_ns)
		for path in tmp_path.rglob('*')
		if path.is_file()
	}
