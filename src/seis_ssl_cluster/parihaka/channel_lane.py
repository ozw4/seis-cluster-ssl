"""Run three cooperative GPU-1 decoders for one ready experiment-40 HMM arm.

The existing main driver still owns source training, embedding publication, and
the final nine-arm summary. This optional lane never stops or migrates a job.
"""

# ruff: noqa: SLF001 - Reuse the coordinator's exact scope and path checks.

from __future__ import annotations

import argparse
import fcntl
import json
import os
import runpy
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka import coordinated_channel as coordinator
from seis_ssl_cluster.parihaka.channel_completion import inspect_completed_channel_job
from seis_ssl_cluster.parihaka.coordinated_hmm import verify_open_lock

if TYPE_CHECKING:
	from collections.abc import Iterator, Sequence

CELL_FILES = frozenset({'best.pt', 'latest.pt', 'history.csv', 'metrics.json'})
LEGACY_MODEL = 'mae_hmm_k6_distill010'


def cell_commands(model: str) -> list[list[str]]:
	"""Return all fifteen full-budget commands, without inspecting GPU inputs."""
	if model not in coordinator.MODELS:
		raise ValueError(
			'the auxiliary Channel lane accepts only the four new HMM arms'
		)
	config = coordinator.EXPERIMENT / f'40_downstream/01_{model}.yaml'
	cli = (
		coordinator.EXPERIMENT.parents[3]
		/ 'proc/seis_ssl_cluster/run_parihaka_channel_decoder.py'
	)
	commands = []
	for index in reversed(range(5)):
		for size in ('large', 'medium', 'small'):
			layout = f'layout_{index:03d}'
			scope = coordinator.classify_coordinated_channel_run(
				load_config(config),
				config_path=config,
				model=model,
				layout_id=layout,
				data_size=size,
				layout_config=coordinator.LAYOUT_CONFIG,
			)
			if scope is None:
				raise ValueError(
					'auxiliary Channel command escaped the coordinated scope'
				)
			commands.append(
				[
					sys.executable,
					str(cli),
					'--config',
					str(config),
					'--model',
					model,
					'--layout',
					layout,
					'--size',
					size,
					'--layout-config',
					str(coordinator.LAYOUT_CONFIG),
					'--device',
					'cuda',
				]
			)
	return commands


@contextmanager
def _driver_lock(path: Path) -> Iterator[int]:
	coordinator._safe_path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	coordinator._safe_path(path.parent)
	descriptor = os.open(
		path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
	)
	try:
		verify_open_lock(path, descriptor)
		fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
		verify_open_lock(path, descriptor)
		yield descriptor
	finally:
		# Do not explicitly unlock: a surviving child must retain this open-file
		# description until it exits if its scheduler has died unexpectedly.
		os.close(descriptor)


def _cell_hashes(path: Path) -> dict[str, str]:
	coordinator._safe_path(path)
	if {item.name for item in path.iterdir()} != CELL_FILES:
		raise ValueError(f'legacy Channel cell has an unexpected file set: {path}')
	return coordinator._hashes([path / name for name in sorted(CELL_FILES)])


def inspect_legacy_publication(run_root: Path) -> None:
	"""Require all ten staging cells completed and byte-identically published."""
	for index in range(5):
		for size in ('small', 'large'):
			cell = Path(f'model={LEGACY_MODEL}/layout=layout_{index:03d}/size={size}')
			staging = run_root / 'gpu1_staging' / cell
			canonical = run_root / 'runs' / cell
			before = _cell_hashes(staging)
			published = _cell_hashes(canonical)
			expected = {
				'model': LEGACY_MODEL,
				'layout_id': f'layout_{index:03d}',
				'data_size': size,
			}
			for path in (staging, canonical):
				metrics = json.loads((path / 'metrics.json').read_text())
				identity = metrics.get('benchmark_identity', {})
				if any(
					metrics.get(key) != value or identity.get(key) != value
					for key, value in expected.items()
				):
					raise ValueError(
						'legacy publication identity differs from its expected cell'
					)
			inspect_completed_channel_job(staging)
			inspect_completed_channel_job(canonical)
			if (
				{Path(path).name: digest for path, digest in before.items()}
				!= {Path(path).name: digest for path, digest in published.items()}
				or _cell_hashes(staging) != before
				or _cell_hashes(canonical) != published
			):
				raise ValueError(
					'legacy staging publication is missing or changed during audit'
				)


def _argument(arguments: Sequence[str], name: str) -> str | None:
	value = None
	for index, argument in enumerate(arguments):
		if argument.startswith(f'{name}='):
			value = argument.partition('=')[2]
		elif argument == name and index + 1 < len(arguments):
			value = arguments[index + 1]
	return value


def _inspect_cli_contract(cli: Path) -> None:
	coordinator._safe_path(cli, regular=True)
	namespace = runpy.run_path(str(cli), run_name='coordinated_channel_cli_contract')
	if (
		namespace.get('run_coordinated_channel_if_scoped')
		is not coordinator.run_coordinated_channel_if_scoped
	):
		raise ValueError('auxiliary Channel lane requires the cooperative decoder CLI')


def _holds_cell_lock(process: Path, lock_path: Path) -> bool:
	"""Prove the live decoder itself holds the expected advisory write lock."""
	if not lock_path.exists():
		return False
	coordinator._safe_path(lock_path, regular=True)
	expected = lock_path.stat()
	for descriptor in (process / 'fd').iterdir():
		try:
			actual = descriptor.stat()
			if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
				continue
			info = (process / 'fdinfo' / descriptor.name).read_text()
			if any(
				line.startswith('lock:')
				and f' FLOCK  ADVISORY  WRITE {process.name} ' in line
				for line in info.splitlines()
			):
				return True
		except FileNotFoundError:
			continue
	return False


def inspect_live_producers(  # noqa: C901, PLR0912 - Keep the process evidence gate together.
	run_root: Path, model: str, *, proc_root: Path = Path('/proc')
) -> None:
	"""Reject live staging workers, old noncooperative decoders, or this extractor.

	The command line and open lock descriptors are read, never process environment
	variables. Inaccessible evidence fails closed; naturally exited PIDs are safe.
	"""
	for process in proc_root.iterdir():
		if not process.name.isdecimal():
			continue
		try:
			if process.stat().st_uid != os.getuid():
				continue
			arguments = (process / 'cmdline').read_bytes().decode().split('\0')
			basenames = {Path(argument).name for argument in arguments if argument}
			if 'extract_embeddings.py' in basenames:
				config = _argument(arguments, '--config')
				if config is None:
					raise ValueError(
						f'cannot classify live embedding producer: PID {process.name}'
					)
				if config and Path(config).name == f'01_extract_{model}.yaml':
					raise ValueError(
						f'Channel embedding producer is still live: PID {process.name}'
					)
			if 'run_parihaka_channel_decoder.py' not in basenames:
				continue
			config = _argument(arguments, '--config')
			if (
				config
				and Path(config).name == '03_mae_hmm_k6_distill010_gpu1_staging.yaml'
			):
				raise ValueError(
					f'legacy GPU-1 staging decoder is still live: PID {process.name}'
				)
			live_model = _argument(arguments, '--model')
			if live_model is None:
				raise ValueError(
					f'cannot classify live Channel decoder: PID {process.name}'
				)
			if live_model not in coordinator.MODELS:
				if config and Path(config).name in {
					f'01_{name}.yaml' for name in coordinator.MODELS
				}:
					raise ValueError(
						f'noncanonical live Channel model: PID {process.name}'
					)
				continue
			layout, size = (
				_argument(arguments, '--layout'),
				_argument(arguments, '--size'),
			)
			lock_path = (
				run_root
				/ '.channel_cell_locks'
				/ f'{live_model}__{layout}__{size}.lock'
			)
			if not _holds_cell_lock(process, lock_path):
				raise ValueError(
					f'noncooperative Channel decoder is still live: PID {process.name}'
				)
		except (FileNotFoundError, ProcessLookupError):
			if process.exists():
				raise


def _run_cell(
	command: list[str], logs: Path, descriptors: tuple[int, int], stopped: Event
) -> None:
	layout, size = _argument(command, '--layout'), _argument(command, '--size')
	for stage, arguments in (('dry_run', [*command, '--dry-run']), ('train', command)):
		if stopped.is_set():
			return
		print(f'START {layout}/{size} {stage}', flush=True)
		try:
			with (logs / f'{layout}_{size}_{stage}.log').open('x') as stream:
				subprocess.run(  # noqa: S603
					arguments,
					stdout=stream,
					stderr=subprocess.STDOUT,
					check=True,
					pass_fds=descriptors,
				)
		except BaseException:
			stopped.set()
			raise
		print(f'DONE {layout}/{size} {stage}', flush=True)


def run_lane(model: str, *, dry_run: bool = False) -> None:
	"""Drain no jobs; require safe existing state, then launch at most three cells."""
	commands = cell_commands(model)
	_inspect_cli_contract(Path(commands[0][1]))
	if dry_run:
		print(
			json.dumps({'dry_run': True, 'gpu': 1, 'workers': 3, 'commands': commands})
		)
		return
	os.environ.update(
		{
			'CUDA_VISIBLE_DEVICES': '1',
			'OMP_NUM_THREADS': '4',
			'OPENBLAS_NUM_THREADS': '4',
			'MKL_NUM_THREADS': '4',
			'NUMEXPR_NUM_THREADS': '4',
			'PYTHONUNBUFFERED': '1',
		}
	)
	run_root = (
		Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ coordinator.RUNS_STEM.parent
	)
	with (
		_driver_lock(run_root / 'channel_lane_driver.lock') as own_fd,
		_driver_lock(run_root / 'gpu1_staging/driver.lock') as legacy_fd,
	):
		inspect_live_producers(run_root, model)
		inspect_legacy_publication(run_root)
		# Each child rechecks the complete source, all embedding bytes, and its
		# fresh identity while holding its cell lock before initializing CUDA.
		inspect_live_producers(run_root, model)
		logs_root = run_root / 'logs'
		coordinator._safe_path(logs_root)
		logs_root.mkdir(parents=True, exist_ok=True)
		logs = Path(tempfile.mkdtemp(prefix='channel_lane_', dir=logs_root))
		print(f'LOG DIRECTORY {logs}', flush=True)
		stopped = Event()
		with ThreadPoolExecutor(max_workers=3) as executor:
			futures = [
				executor.submit(_run_cell, command, logs, (own_fd, legacy_fd), stopped)
				for command in commands
			]
			try:
				for future in as_completed(futures):
					future.result()
			except BaseException:
				# Leave already-running children to finish naturally; do not start
				# additional queued work after a detected failure.
				stopped.set()
				for future in futures:
					future.cancel()
				raise
		print(
			f'COMPLETE {model}: fifteen audited canonical cells; main owns summary',
			flush=True,
		)


def main() -> None:
	"""Parse the deliberately narrow, single-ready-arm auxiliary interface."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--model', required=True, choices=tuple(coordinator.MODELS))
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()
	run_lane(args.model, dry_run=args.dry_run)


if __name__ == '__main__':
	main()
