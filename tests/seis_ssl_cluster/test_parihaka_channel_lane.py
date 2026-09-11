"""CPU-only safety, command ownership, and fixed-concurrency lane contracts."""

# ruff: noqa: SLF001 - Exercise the scheduler's private process and lock gates.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Barrier, Lock

import pytest

from seis_ssl_cluster.parihaka import channel_lane as lane

MODEL = 'local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill010'


@pytest.fixture
def run_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	# The real driver adjusts child environment variables. Register their old
	# values so mocked driver tests cannot affect unrelated tests afterwards.
	for name in (
		'CUDA_VISIBLE_DEVICES',
		'OMP_NUM_THREADS',
		'OPENBLAS_NUM_THREADS',
		'MKL_NUM_THREADS',
		'NUMEXPR_NUM_THREADS',
		'PYTHONUNBUFFERED',
	):
		monkeypatch.setenv(name, os.environ.get(name, ''))
	return root / lane.coordinator.RUNS_STEM.parent


@pytest.mark.parametrize('model', tuple(lane.coordinator.MODELS))
def test_commands_are_exact_full_budget_cells(run_root: Path, model: str) -> None:
	commands = lane.cell_commands(model)
	assert len(commands) == 15
	assert {lane._argument(command, '--model') for command in commands} == {model}
	assert [
		(lane._argument(command, '--layout'), lane._argument(command, '--size'))
		for command in commands
	] == [
		(f'layout_{index:03d}', size)
		for index in reversed(range(5))
		for size in ('large', 'medium', 'small')
	]
	for command in commands:
		assert Path(command[1]).is_file()
		assert lane._argument(command, '--device') == 'cuda'
		assert not {
			'--max-steps',
			'--resume',
			'--validation-only',
			'--output-root',
		} & set(command)
	assert not run_root.exists()


def test_lane_dry_run_has_no_writes_or_gpu_calls(
	run_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	def forbidden(*_args: object, **_kwargs: object) -> None:
		raise AssertionError('dry run attempted execution')

	monkeypatch.setattr(lane, '_driver_lock', forbidden)
	monkeypatch.setattr(lane.subprocess, 'run', forbidden)
	monkeypatch.setattr(lane.coordinator.torch.cuda, 'is_available', forbidden)
	lane.run_lane(MODEL, dry_run=True)
	payload = json.loads(capsys.readouterr().out)
	assert payload['dry_run'] is True
	assert payload['workers'] == 3
	assert len(payload['commands']) == 15
	assert not run_root.exists()


@pytest.mark.parametrize(
	'model', ['mae_hmm_k6_distill010', 'random_init', '../' + MODEL]
)
def test_lane_refuses_other_models(run_root: Path, model: str) -> None:
	with pytest.raises(ValueError, match='four new HMM arms'):
		lane.run_lane(model, dry_run=True)
	assert not run_root.exists()


def _published_fixture(run_root: Path) -> list[Path]:
	paths = []
	for index in range(5):
		for size in ('small', 'large'):
			cell = f'model={lane.LEGACY_MODEL}/layout=layout_{index:03d}/size={size}'
			for category in ('gpu1_staging', 'runs'):
				path = run_root / category / cell
				path.mkdir(parents=True)
				for name in lane.CELL_FILES:
					(path / name).write_text(f'{index}/{size}/{name}')
				identity = {
					'model': lane.LEGACY_MODEL,
					'layout_id': f'layout_{index:03d}',
					'data_size': size,
				}
				(path / 'metrics.json').write_text(
					json.dumps({**identity, 'benchmark_identity': identity})
				)
				paths.append(path)
	return paths


def test_legacy_gate_audits_all_twenty_cells(
	run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	paths = _published_fixture(run_root)
	audited = []
	monkeypatch.setattr(lane, 'inspect_completed_channel_job', audited.append)
	lane.inspect_legacy_publication(run_root)
	assert audited == paths


@pytest.mark.parametrize(
	'failure', ['different', 'missing', 'extra', 'mutating', 'partial']
)
def test_legacy_gate_fails_closed(
	run_root: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
	paths = _published_fixture(run_root)

	def inspect(path: Path) -> None:
		if failure == 'partial':
			raise ValueError('incomplete checkpoint')
		if failure == 'mutating':
			(path / 'metrics.json').write_text('changed while checking')

	monkeypatch.setattr(lane, 'inspect_completed_channel_job', inspect)
	if failure == 'different':
		(paths[-1] / 'history.csv').write_text('foreign bytes')
	elif failure == 'missing':
		(paths[-1] / 'metrics.json').unlink()
	elif failure == 'extra':
		(paths[-1] / 'foreign.json').write_text('{}')
	with pytest.raises(ValueError, match=r'file set|publication|incomplete'):
		lane.inspect_legacy_publication(run_root)


def test_legacy_publication_rejects_misplaced_complete_cells(
	run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	paths = _published_fixture(run_root)
	monkeypatch.setattr(lane, 'inspect_completed_channel_job', lambda *_args: None)
	for path in paths[-2:]:
		(path / 'metrics.json').write_bytes((paths[0] / 'metrics.json').read_bytes())
	with pytest.raises(ValueError, match='identity differs'):
		lane.inspect_legacy_publication(run_root)


def _process(tmp_path: Path, arguments: list[str]) -> tuple[Path, Path]:
	proc = tmp_path / 'proc'
	process = proc / '1234'
	process.mkdir(parents=True)
	(process / 'cmdline').write_bytes('\0'.join(arguments).encode())
	(process / 'fd').mkdir()
	(process / 'fdinfo').mkdir()
	return proc, process


def _decoder_arguments() -> list[str]:
	return [
		'python',
		'proc/seis_ssl_cluster/run_parihaka_channel_decoder.py',
		'--model',
		MODEL,
		'--layout',
		'layout_000',
		'--size',
		'medium',
	]


def test_noncooperative_live_decoder_rejected(run_root: Path, tmp_path: Path) -> None:
	proc, _ = _process(tmp_path, _decoder_arguments())
	with pytest.raises(ValueError, match='noncooperative'):
		lane.inspect_live_producers(run_root, MODEL, proc_root=proc)


@pytest.mark.parametrize('kind', ['decoder', 'embedding'])
def test_equals_cli_syntax_cannot_bypass_live_gate(
	run_root: Path, tmp_path: Path, kind: str
) -> None:
	arguments = (
		['python', 'extract_embeddings.py', f'--config=01_extract_{MODEL}.yaml']
		if kind == 'embedding'
		else [
			'python',
			'run_parihaka_channel_decoder.py',
			f'--model={MODEL}',
			'--layout=layout_000',
			'--size=medium',
		]
	)
	proc, _ = _process(tmp_path, arguments)
	with pytest.raises(ValueError, match='still live'):
		lane.inspect_live_producers(run_root, MODEL, proc_root=proc)


def test_argument_parser_uses_last_value() -> None:
	assert lane._argument(['--model', 'first', '--model=second'], '--model') == 'second'
	assert lane._argument(['--model=first', '--model', 'second'], '--model') == 'second'


def test_lane_refuses_cli_without_coordination(
	run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(lane.runpy, 'run_path', lambda *_args, **_kwargs: {})
	with pytest.raises(ValueError, match='cooperative decoder CLI'):
		lane.run_lane(MODEL)
	assert not run_root.exists()


@pytest.mark.parametrize('locked', [True, False])
def test_live_decoder_requires_exact_owned_flock(
	run_root: Path, tmp_path: Path, *, locked: bool
) -> None:
	proc, process = _process(tmp_path, _decoder_arguments())
	lock_path = run_root / '.channel_cell_locks' / f'{MODEL}__layout_000__medium.lock'
	lock_path.parent.mkdir(parents=True)
	lock_path.touch()
	(process / 'fd/8').symlink_to(lock_path)
	(process / 'fdinfo/8').write_text(
		'pos:\t0\nlock:\t1: FLOCK  ADVISORY  WRITE 1234 00:02:123 0 EOF\n'
		if locked
		else 'pos:\t0\n'
	)
	if locked:
		lane.inspect_live_producers(run_root, MODEL, proc_root=proc)
	else:
		with pytest.raises(ValueError, match='noncooperative'):
			lane.inspect_live_producers(run_root, MODEL, proc_root=proc)


@pytest.mark.parametrize('kind', ['staging', 'embedding'])
def test_live_legacy_or_embedding_producer_rejected(
	run_root: Path, tmp_path: Path, kind: str
) -> None:
	arguments = (
		['python', 'extract_embeddings.py', '--config', f'01_extract_{MODEL}.yaml']
		if kind == 'embedding'
		else [
			'python',
			'run_parihaka_channel_decoder.py',
			'--config',
			'03_mae_hmm_k6_distill010_gpu1_staging.yaml',
		]
	)
	proc, _ = _process(tmp_path, arguments)
	with pytest.raises(ValueError, match='still live'):
		lane.inspect_live_producers(run_root, MODEL, proc_root=proc)


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'fifo'])
def test_driver_lock_rejects_aliases_without_truncation(
	tmp_path: Path, kind: str
) -> None:
	path = tmp_path / 'driver.lock'
	original = tmp_path / 'unrelated.txt'
	original.write_text('preserve me')
	if kind == 'symlink':
		path.symlink_to(original)
	elif kind == 'hardlink':
		os.link(original, path)
	else:
		os.mkfifo(path)
	with (
		pytest.raises(ValueError, match=r'aliases|regular-file'),
		lane._driver_lock(path),
	):
		pytest.fail('unsafe lock accepted')
	assert original.read_text() == 'preserve me'


def test_driver_lock_refuses_second_owner(tmp_path: Path) -> None:
	path = tmp_path / 'driver.lock'
	with (
		lane._driver_lock(path),
		pytest.raises(BlockingIOError),
		lane._driver_lock(path),
	):
		pytest.fail('second owner acquired the singleton')


def test_surviving_child_retains_driver_lock(tmp_path: Path) -> None:
	path = tmp_path / 'driver.lock'
	with lane._driver_lock(path) as descriptor:
		child = subprocess.Popen(
			[
				sys.executable,
				'-c',
				'import sys; print("ready", flush=True); sys.stdin.readline()',
			],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			text=True,
			pass_fds=(descriptor,),
		)
		assert child.stdout is not None
		assert child.stdout.readline() == 'ready\n'
	try:
		with pytest.raises(BlockingIOError), lane._driver_lock(path):
			pytest.fail('scheduler death released a live child lock')
	finally:
		child.communicate('\n', timeout=10)
	assert child.returncode == 0
	with lane._driver_lock(path):
		pass


def test_lane_runs_three_workers_and_preserves_logs(
	run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(lane, 'inspect_live_producers', lambda *_args: None)
	monkeypatch.setattr(lane, 'inspect_legacy_publication', lambda *_args: None)
	barrier, guard = Barrier(3), Lock()
	calls, active, maximum = [], 0, 0

	def run(arguments: list[str], **kwargs: object) -> None:
		nonlocal active, maximum
		with guard:
			active += 1
			maximum = max(maximum, active)
			calls.append(arguments)
		assert len(kwargs['pass_fds']) == 2
		assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
		barrier.wait(timeout=10)
		with guard:
			active -= 1

	monkeypatch.setattr(lane.subprocess, 'run', run)
	for _ in range(2):
		lane.run_lane(MODEL)
	assert len(calls) == 60
	assert maximum == 3
	logs = list((run_root / 'logs').glob('channel_lane_*'))
	assert len(logs) == 2
	assert all(len(list(path.iterdir())) == 30 for path in logs)
	assert sum('--dry-run' in command for command in calls) == 30


def test_failed_gate_launches_nothing(
	run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	def fail(*_args: object) -> None:
		raise ValueError('legacy not complete')

	monkeypatch.setattr(lane, 'inspect_live_producers', lambda *_args: None)
	monkeypatch.setattr(lane, 'inspect_legacy_publication', fail)
	with pytest.raises(ValueError, match='legacy not complete'):
		lane.run_lane(MODEL)
	assert not (run_root / 'logs').exists()
	with lane._driver_lock(run_root / 'gpu1_staging/driver.lock'):
		pass  # A failed gate must not block legitimate staging recovery.


def test_child_failure_stops_queued_work(
	run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(lane, 'inspect_live_producers', lambda *_args: None)
	monkeypatch.setattr(lane, 'inspect_legacy_publication', lambda *_args: None)
	calls = []

	def fail(arguments: list[str], **_kwargs: object) -> None:
		calls.append(arguments)
		raise subprocess.CalledProcessError(1, arguments)

	monkeypatch.setattr(lane.subprocess, 'run', fail)
	with pytest.raises(subprocess.CalledProcessError):
		lane.run_lane(MODEL)
	assert 1 <= len(calls) <= 3
	assert all('--dry-run' in command for command in calls)
	assert len(list((run_root / 'logs').glob('channel_lane_*'))) == 1


def test_shell_wrapper_is_thin_and_syntax_valid() -> None:
	path = lane.coordinator.EXPERIMENT / 'run_channel_lane.sh'
	result = subprocess.run(['bash', '-n', str(path)], capture_output=True, check=False)  # noqa: S603, S607
	assert result.returncode == 0
	assert (
		'exec python -m seis_ssl_cluster.parihaka.channel_lane "$@"' in path.read_text()
	)
