"""CPU-only command and output-ownership contracts for the auxiliary HMM lane."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = (
	Path(__file__).resolve().parents[2] / 'experiments/parihaka/facies_benchmark_v1/'
	'40_pretraining_comparison_completion_v1/run_hmm_lane.sh'
)


@pytest.fixture
def lane_environment(tmp_path: Path) -> dict[str, str]:
	"""Record Python invocations without loading CUDA or executing any training."""
	stub = tmp_path / 'bin' / 'python'
	stub.parent.mkdir()
	stub.write_text(
		f'#!{sys.executable}\n'
		'import json, os, sys\n'
		'if len(sys.argv) > 2 and "verify_open_lock" in sys.argv[2]:\n'
		'    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n'
		'with open(os.environ["LANE_TEST_CALLS"], "a") as stream:\n'
		'    stream.write(json.dumps({"args": sys.argv[1:], '
		'"gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), '
		'"threads": os.environ.get("OMP_NUM_THREADS")}) + "\\n")\n'
		'if "-c" in sys.argv and os.environ.get("LANE_TEST_NO_HELPER"):\n'
		'    sys.exit(1)\n',
	)
	stub.chmod(0o755)
	return {
		**os.environ,
		'PATH': f'{stub.parent}{os.pathsep}{os.environ["PATH"]}',
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(tmp_path / 'artifacts'),
		'LANE_TEST_CALLS': str(tmp_path / 'calls.jsonl'),
	}


def _run(environment: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(  # noqa: S603
		['bash', str(DRIVER), *args],  # noqa: S607
		env=environment,
		capture_output=True,
		text=True,
		check=False,
		timeout=20,
	)


def _calls(environment: dict[str, str]) -> list[dict[str, object]]:
	return [
		json.loads(line)
		for line in Path(environment['LANE_TEST_CALLS']).read_text().splitlines()
	]


def test_lane_dry_run_is_read_only(lane_environment: dict[str, str]) -> None:
	result = _run(lane_environment, '--dry-run')
	assert result.returncode == 0, result.stderr
	calls = _calls(lane_environment)
	assert len(calls) == 7  # Import guard, then full and smoke dry-run for each arm.
	assert all('--dry-run' in call['args'] for call in calls[1:])
	assert all(call['gpu'] == '0' and call['threads'] == '4' for call in calls)
	assert not Path(lane_environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT']).exists()


@pytest.mark.parametrize('arguments', [('--invalid',), ('--dry-run', '--invalid')])
def test_lane_rejects_unknown_arguments(
	lane_environment: dict[str, str],
	arguments: tuple[str, ...],
) -> None:
	assert _run(lane_environment, *arguments).returncode == 2
	assert not Path(lane_environment['LANE_TEST_CALLS']).exists()


def test_lane_requires_cooperative_helper(lane_environment: dict[str, str]) -> None:
	lane_environment['LANE_TEST_NO_HELPER'] = '1'
	assert _run(lane_environment).returncode != 0
	assert len(_calls(lane_environment)) == 1
	assert not Path(lane_environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT']).exists()


def test_lane_rejects_hardlinked_driver_lock(
	lane_environment: dict[str, str],
	tmp_path: Path,
) -> None:
	run_root = (
		Path(lane_environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'channel_benchmark/pretraining_comparison_completion_v1'
	)
	run_root.mkdir(parents=True)
	unrelated = tmp_path / 'unrelated.txt'
	unrelated.write_text('preserve this content')
	os.link(unrelated, run_root / 'hmm_lane_driver.lock')
	assert _run(lane_environment).returncode != 0
	assert unrelated.read_text() == 'preserve this content'
	assert len(_calls(lane_environment)) == 1


def test_lane_rejects_fifo_driver_lock(lane_environment: dict[str, str]) -> None:
	run_root = (
		Path(lane_environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'channel_benchmark/pretraining_comparison_completion_v1'
	)
	run_root.mkdir(parents=True)
	os.mkfifo(run_root / 'hmm_lane_driver.lock')
	assert _run(lane_environment).returncode != 0
	assert len(_calls(lane_environment)) == 1


def test_lane_orders_three_sources_and_preserves_logs(
	lane_environment: dict[str, str],
) -> None:
	for _ in range(2):
		result = _run(lane_environment)
		assert result.returncode == 0, result.stderr
	calls = _calls(lane_environment)
	assert len(calls) == 26  # Two invocations, each import plus 3 x 4 commands.
	for offset, arm in enumerate(
		('local_bt3/distill020', 'random/distill010', 'random/distill020')
	):
		start = 1 + 4 * offset
		dry, smoke, full, audit = [call['args'] for call in calls[start : start + 4]]
		assert arm in dry[dry.index('--config') + 1]
		assert '--dry-run' in dry
		assert smoke[smoke.index('--max-steps') + 1] == '1'
		assert smoke[smoke.index('--output-root') + 1].endswith('/smoke_1step')
		assert '--max-steps' not in full
		assert '--output-root' not in full
		assert '--checkpoint-complete' in audit
	run_root = (
		Path(lane_environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'channel_benchmark/pretraining_comparison_completion_v1'
	)
	log_directories = list((run_root / 'logs').glob('hmm_lane_*'))
	assert len(log_directories) == 2
	assert all(len(list(directory.iterdir())) == 12 for directory in log_directories)
	assert (run_root / 'hmm_lane_driver.lock').is_file()


def test_lane_driver_lock_prevents_duplicate_auxiliary(
	lane_environment: dict[str, str],
) -> None:
	run_root = (
		Path(lane_environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'channel_benchmark/pretraining_comparison_completion_v1'
	)
	run_root.mkdir(parents=True)
	with (run_root / 'hmm_lane_driver.lock').open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		assert _run(lane_environment).returncode != 0
	assert len(_calls(lane_environment)) == 1
	assert not list((run_root / 'logs').iterdir())


def test_lane_rejects_symlinked_driver_lock(
	lane_environment: dict[str, str],
	tmp_path: Path,
) -> None:
	run_root = (
		Path(lane_environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ 'channel_benchmark/pretraining_comparison_completion_v1'
	)
	run_root.mkdir(parents=True)
	unrelated = tmp_path / 'unrelated.txt'
	unrelated.write_text('preserve this content')
	(run_root / 'hmm_lane_driver.lock').symlink_to(unrelated)
	assert _run(lane_environment).returncode != 0
	assert unrelated.read_text() == 'preserve this content'
	assert len(_calls(lane_environment)) == 1
