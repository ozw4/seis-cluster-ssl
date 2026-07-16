from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import tools.benchmark_seis_ssl_cluster as benchmark_module

if TYPE_CHECKING:
	from pathlib import Path
	from types import TracebackType

	import pytest


def test_synthetic_benchmark_cli_writes_reproducible_case_contract(
	tmp_path: Path,
) -> None:
	first_path = tmp_path / 'first.json'
	second_path = tmp_path / 'nested' / 'second.json'
	command = [
		sys.executable,
		'tools/benchmark_seis_ssl_cluster.py',
		'--seed',
		'123',
		'--warm-up',
		'0',
		'--repeat',
		'1',
	]
	for output_path in (first_path, second_path):
		subprocess.run(  # noqa: S603
			[*command, '--output-json', str(output_path)],
			check=True,
			capture_output=True,
			text=True,
		)

	first = json.loads(first_path.read_text(encoding='utf-8'))
	second = json.loads(second_path.read_text(encoding='utf-8'))

	assert first['seed'] == second['seed'] == 123
	assert first['warm_up'] == second['warm_up'] == 0
	assert first['repeat'] == second['repeat'] == 1
	assert [(case['name'], case['shape']) for case in first['cases']] == [
		(case['name'], case['shape']) for case in second['cases']
	]
	assert [case['name'] for case in first['cases']] == [
		'memmap_repeated_open_crop',
		'spatial_mask_16_cubed_m075_block1',
		'amplitude_preprocessing',
		'position_embedding_visible_selection',
		'embedding_merge_token_to_voxel',
		'token_phase_residualization',
	]
	for case in first['cases']:
		assert case['median_seconds'] >= 0.0
		assert case['p25_seconds'] >= 0.0
		assert case['p75_seconds'] >= 0.0
	assert first['environment']['device'] == 'cpu'


def test_benchmark_resources_close_before_temporary_directory(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	closed: list[str] = []

	class TrackedTemporaryDirectory:
		def __init__(self, *, prefix: str) -> None:
			assert prefix

		def __enter__(self) -> str:
			return str(tmp_path)

		def __exit__(
			self,
			exc_type: type[BaseException] | None,
			exc_value: BaseException | None,
			traceback: TracebackType | None,
		) -> None:
			assert closed == ['amplitude_store', 'memmap_store']

	def build_cases(
		seed: int,
		temp_dir: Path,
		resources: benchmark_module.ExitStack,
	) -> tuple[benchmark_module.BenchmarkCase, ...]:
		assert seed == 7
		assert temp_dir == tmp_path
		resources.callback(closed.append, 'memmap_store')
		resources.callback(closed.append, 'amplitude_store')
		return (
			benchmark_module.BenchmarkCase(name='noop', shape={}, run=lambda: None),
		)

	monkeypatch.setattr(
		benchmark_module.tempfile,
		'TemporaryDirectory',
		TrackedTemporaryDirectory,
	)
	monkeypatch.setattr(benchmark_module, '_build_cases', build_cases)
	monkeypatch.setattr(benchmark_module, '_git_commit', lambda: None)

	report = benchmark_module.run_benchmarks(seed=7, warm_up=0, repeat=1)

	assert [case['name'] for case in report['cases']] == ['noop']
