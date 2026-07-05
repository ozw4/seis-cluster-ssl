from __future__ import annotations

import json
from pathlib import Path

from seis_ssl_cluster.results import validate_results_artifacts
from seis_ssl_cluster.validation import validate_artifact_paths
from tests.helpers import run_python_proc


def test_artifact_path_validation_detects_runs_path(tmp_path: Path) -> None:
	config = _write_text(
		tmp_path / 'config.yaml',
		'output_dir: runs/nopims/pretrain_v1\n',
	)

	report = validate_artifact_paths(
		root=tmp_path / 'artifacts',
		scan_paths=(config,),
		fail_on_runs=True,
	)

	assert not report.ok
	assert _has_message(report.errors, 'runs/ artifact path is not allowed')


def test_artifact_path_validation_allows_pretraining_checkpoint(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'artifacts'
	config = _write_text(
		tmp_path / 'config.yaml',
		(
			f'checkpoint: {root}/pretraining/nopims/pretrain_v1'
			'/model/full_100ep/mae_latest.pt\n'
		),
	)

	report = validate_artifact_paths(root=root, scan_paths=(config,), fail_on_runs=True)

	assert report.ok


def test_artifact_path_validation_rejects_checkpoint_outside_pretraining(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'artifacts'
	config = _write_text(
		tmp_path / 'config.yaml',
		(
			f'output_checkpoint: {root}/clustering/nopims/pretrain_v1'
			'/model/full/overlap_x64/mae_latest.pt\n'
		),
	)

	report = validate_artifact_paths(root=root, scan_paths=(config,))

	assert not report.ok
	assert _has_message(
		report.errors,
		'checkpoint artifacts must be written under pretraining/',
	)


def test_artifact_path_validation_rejects_stage_spec_mixing(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'artifacts'
	config = _write_text(
		tmp_path / 'config.yaml',
		'\n'.join(
			(
				(
					f'embeddings_output: {root}/embeddings/nopims/pretrain_v1'
					'/model/full/overlap_x64/k6_8'
				),
				(
					f'clustering_output: {root}/clustering/nopims/pretrain_v1'
					'/model/full/overlap_x64/k6_8/token_xy750'
				),
				'',
			)
		),
	)

	report = validate_artifact_paths(root=root, scan_paths=(config,))

	assert not report.ok
	assert _has_message(report.errors, 'embeddings/ paths must not include')
	assert _has_message(report.errors, 'clustering/ paths must not include')


def test_artifact_path_validation_warns_for_old_embedding_path(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'artifacts'
	config = _write_text(
		tmp_path / 'config.yaml',
		f'output_dir: {root}/embeddings/nopims/pretrain_v1/model\n',
	)

	report = validate_artifact_paths(root=root, scan_paths=(config,))

	assert report.ok
	assert _has_message(report.warnings, 'legacy embeddings path omits')


def test_artifact_path_validation_allow_pattern_skips_fixture(
	tmp_path: Path,
) -> None:
	config = _write_text(
		tmp_path / 'test_config.py',
		"BAD = 'runs/nopims/pretrain_v1'\n",
	)

	report = validate_artifact_paths(
		root=tmp_path / 'artifacts',
		scan_paths=(config,),
		fail_on_runs=True,
		allow_patterns=(config.as_posix(),),
	)

	assert report.ok
	assert report.scanned_file_count == 0


def test_artifact_path_validation_checks_results_publish_paths(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'artifacts'
	manifest = _write_text(
		tmp_path / 'results' / 'publish_manifest.json',
		json.dumps(
			{
				'items': [
					{
						'source': (
							f'{root}/pretraining/nopims/pretrain_v1'
							'/model/full_100ep/mae_latest.pt'
						),
						'target': 'checkpoints/mae_latest.pt',
					}
				],
				'source_artifact_root': str(root),
			}
		),
	)

	report = validate_artifact_paths(root=root, scan_paths=(manifest,))

	assert not report.ok
	assert _has_message(report.errors, 'references heavy artifact')
	assert _has_message(report.warnings, 'records a local artifact path')


def test_artifact_path_validation_is_separate_from_results_validation(
	tmp_path: Path,
) -> None:
	results_root = tmp_path / 'results'
	_write_text(results_root / 'model.pt', 'checkpoint\n')
	report_md = _write_text(tmp_path / 'report.md', 'see results/model.pt\n')

	artifact_report = validate_artifact_paths(
		root=tmp_path / 'artifacts',
		scan_paths=(report_md,),
	)
	results_report = validate_results_artifacts(results_root)

	assert artifact_report.ok
	assert not results_report.ok
	assert _has_message(results_report.errors, 'forbidden heavy artifact suffix')


def test_validate_artifact_paths_proc_returns_nonzero_for_error(
	tmp_path: Path,
) -> None:
	config = _write_text(
		tmp_path / 'config.yaml',
		'output_dir: runs/nopims/pretrain_v1\n',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/validate_artifact_paths.py'),
		'--root',
		tmp_path / 'artifacts',
		'--scan',
		config,
		'--fail-on-runs',
	)

	assert result.returncode == 1
	assert 'artifact path validation: failed' in result.stdout
	assert 'runs/ artifact path is not allowed' in result.stdout


def _write_text(path: Path, content: str) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding='utf-8')
	return path


def _has_message(findings: object, text: str) -> bool:
	return any(text in item.message for item in findings)
