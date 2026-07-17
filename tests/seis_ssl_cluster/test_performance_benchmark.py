from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np

import tools.benchmark_seis_ssl_cluster_performance as benchmark_module

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
		'tools/benchmark_seis_ssl_cluster_performance.py',
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
	assert [case['input_fingerprint'] for case in first['cases']] == [
		case['input_fingerprint'] for case in second['cases']
	]
	assert [case['name'] for case in first['cases']] == [
		'memmap_repeated_open_crop',
		'spatial_mask_16_cubed_m075_block1',
		'amplitude_preprocessing',
		'position_embedding_visible_selection',
		'embedding_merge_token_to_voxel',
		'token_phase_residualization',
		'hmm_squared_euclidean_emission',
	]
	for case in first['cases']:
		expected_version = (
			2 if case['name'] == 'hmm_squared_euclidean_emission' else 1
		)
		assert case['version'] == expected_version
		assert len(case['input_fingerprint']) == 16
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
			benchmark_module.BenchmarkCase(
				name='noop',
				version=1,
				shape={},
				input_fingerprint='abc',
				run=lambda: None,
			),
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


def test_baseline_comparison_requires_matching_case_contract() -> None:
	current = _report_case(version=2, fingerprint='same', median=1.0)
	baseline = _report_case(version=2, fingerprint='same', median=4.0)

	comparison = benchmark_module.compare_reports(current, baseline)

	assert comparison['cases'][0]['comparable'] is True
	assert comparison['cases'][0]['speedup_multiplier'] == 4.0
	for key, value in (
		('name', 'renamed'),
		('version', 1),
		('input_fingerprint', 'different'),
	):
		incompatible = _report_case(version=2, fingerprint='same', median=4.0)
		incompatible['cases'][0][key] = value
		case = benchmark_module.compare_reports(current, incompatible)['cases'][0]
		assert case['comparable'] is False
		assert 'speedup_multiplier' not in case
		expected_note = (
			'case is absent from baseline report'
			if key == 'name'
			else f'{key} mismatch'
		)
		assert case['note'] == expected_note


def test_case_fingerprint_covers_array_contents_and_complete_settings() -> None:
	base = np.zeros((2, 3), dtype=np.float32)
	changed = base.copy()
	changed[1, 2] = 1.0

	def fingerprint(array: np.ndarray, *, setting: int) -> str:
		case = benchmark_module._case(  # noqa: SLF001
			name='case',
			shape={'array': [2, 3]},
			inputs={
				'array': benchmark_module._array_descriptor(array),  # noqa: SLF001
				'setting': setting,
			},
			run=lambda: None,
		)
		return case.input_fingerprint

	assert len({
		fingerprint(base, setting=1),
		fingerprint(changed, setting=1),
		fingerprint(base, setting=2),
	}) == 3


def test_hmm_benchmark_runs_production_emission_kernel_with_precomputed_norms(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	calls: list[tuple[np.dtype, np.dtype, np.dtype]] = []

	def record_kernel(
		features: np.ndarray,
		centers: np.ndarray,
		center_squared_norms: np.ndarray,
	) -> np.ndarray:
		calls.append((features.dtype, centers.dtype, center_squared_norms.dtype))
		np.testing.assert_allclose(
			center_squared_norms,
			np.einsum('kd,kd->k', centers, centers, optimize=True),
		)
		return np.empty((features.shape[0], centers.shape[0]), dtype=features.dtype)

	monkeypatch.setattr(
		benchmark_module,
		'_squared_euclidean_emission_costs_with_center_norms',
		record_kernel,
	)
	case = benchmark_module._hmm_emission_case(13)  # noqa: SLF001

	case.run()

	assert case.version == 2
	assert calls == [(np.dtype(np.float32),) * 3]


def test_zero_current_median_omits_speedup_multiplier() -> None:
	current = _report_case(version=2, fingerprint='same', median=0.0)
	baseline = _report_case(version=2, fingerprint='same', median=4.0)

	case = benchmark_module.compare_reports(current, baseline)['cases'][0]

	assert case['comparable'] is True
	assert 'speedup_multiplier' not in case
	assert case['note'] == 'current median is zero; multiplier is undefined'


def test_smoke_cli_writes_json_and_markdown(tmp_path: Path) -> None:
	json_path = tmp_path / 'report.json'
	markdown_path = tmp_path / 'report.md'
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'tools/benchmark_seis_ssl_cluster_performance.py',
			'--smoke',
			'--output-json',
			str(json_path),
			'--output-markdown',
			str(markdown_path),
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert 'benchmark JSON:' in result.stdout
	assert json.loads(json_path.read_text(encoding='utf-8'))['repeat'] == 1
	markdown = markdown_path.read_text(encoding='utf-8')
	assert 'Input conditions' in markdown
	assert '| Case | Shape and settings |' in markdown
	assert (
		'`{"dtype":"float32","feature_dim":128,"states":12,"tokens":4096}`'
		in markdown
	)
	assert 'Median (s)' in markdown
	assert 'Comparable' in markdown
	assert 'Speedup' in markdown
	assert 'Cautions' in markdown


def _report_case(
	*,
	version: int,
	fingerprint: str,
	median: float,
) -> dict[str, object]:
	return {
		'schema_version': 2,
		'cases': [
			{
				'name': 'case',
				'version': version,
				'input_fingerprint': fingerprint,
				'median_seconds': median,
				'p25_seconds': median,
				'p75_seconds': median,
			},
		],
	}
