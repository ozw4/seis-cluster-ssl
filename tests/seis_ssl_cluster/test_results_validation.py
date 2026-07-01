from __future__ import annotations

from pathlib import Path

from seis_ssl_cluster.results import validate_results_artifacts
from tests.helpers import run_python_proc


def test_results_validation_detects_forbidden_suffix(tmp_path: Path) -> None:
	root = tmp_path / 'results'
	_write_file(root / 'f3' / 'model.pt', b'checkpoint')

	report = validate_results_artifacts(root)

	assert not report.ok
	assert any(
		'forbidden heavy artifact suffix' in item.message for item in report.errors
	)


def test_results_validation_detects_max_file_size_exceeded(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'results'
	_write_file(root / 'summary.txt', b'12345')

	report = validate_results_artifacts(root, max_file_size_bytes=4)

	assert not report.ok
	assert any(
		'file exceeds max_file_size_bytes' in item.message for item in report.errors
	)


def test_results_validation_detects_local_absolute_path(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'results'
	_write_file(
		root / 'report.md',
		b'local output: /workspace/artifacts/seis_ssl_cluster/inspection\n',
	)

	report = validate_results_artifacts(root)
	strict_report = validate_results_artifacts(root, local_path_policy='error')

	assert report.ok
	assert any(
		'local absolute path marker found' in item.message for item in report.warnings
	)
	assert not strict_report.ok
	assert any(
		'local absolute path marker found' in item.message
		for item in strict_report.errors
	)


def test_results_validation_detects_missing_required_file(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'results'
	root.mkdir()

	report = validate_results_artifacts(
		root,
		required_files=(Path('f3/facies_benchmark_v1/inspection/report.md'),),
	)

	assert not report.ok
	assert any('required file is missing' in item.message for item in report.errors)


def test_results_validation_passes_valid_small_results_tree(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'results'
	required_files = (
		Path('f3/facies_benchmark_v1/inspection/report.md'),
		Path('f3/facies_benchmark_v1/baseline_comparison/comparison_report.md'),
	)
	_write_file(root / required_files[0], b'# inspection\n')
	_write_file(root / required_files[1], b'# comparison\n')
	_write_file(
		root / 'f3/facies_benchmark_v1/baseline_comparison/comparison_table.csv',
		b'feature_kind,macro_f1\npretrained_encoder,0.7\n',
	)
	_write_file(
		root / 'f3/facies_benchmark_v1/inspection/figures/example.png',
		b'\x89PNG\r\n',
	)

	report = validate_results_artifacts(root, required_files=required_files)

	assert report.ok
	assert report.errors == ()
	assert report.warnings == ()
	assert report.file_count == 4


def test_results_validation_rejects_artifacts_directory_inside_results(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'results'
	_write_file(root / 'f3' / 'artifacts' / 'summary.md', b'# wrong place\n')

	report = validate_results_artifacts(root)

	assert not report.ok
	assert any(
		'artifacts/ must not be stored inside results/' in item.message
		for item in report.errors
	)


def test_validate_results_artifacts_proc_passes_valid_tree(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'results'
	_write_file(root / 'report.md', b'# report\n')

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/validate_results_artifacts.py'),
		'--root',
		root,
		'--required-file',
		'report.md',
	)

	assert result.returncode == 0, result.stderr
	assert 'results validation: ok' in result.stdout
	assert 'file_count: 1' in result.stdout


def _write_file(path: Path, content: bytes) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(content)
	return path
