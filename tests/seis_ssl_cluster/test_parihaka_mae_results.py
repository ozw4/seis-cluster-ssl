# ruff: noqa: CPY001

from __future__ import annotations

import json
from pathlib import Path

import pytest

import seis_ssl_cluster.parihaka.mae_results as results_module
import seis_ssl_cluster.parihaka.mae_validation as validation_module
import seis_ssl_cluster.training.mae_checkpoint as mae_checkpoint_module
from seis_ssl_cluster.parihaka.mae_results import (
	PARIHAKA_MAE_RESULT_FILES,
	summarize_parihaka_mae,
)
from tests.seis_ssl_cluster.test_parihaka_mae_validation import _full_fixture


def test_results_write_exact_portable_three_file_allowlist(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	kwargs, output_dir = _results_fixture(tmp_path, monkeypatch)
	unknown = output_dir / 'reviewer-note.txt'
	output_dir.mkdir(parents=True)
	unknown.write_text('keep\n', encoding='utf-8')

	result = summarize_parihaka_mae(**kwargs)

	assert result.reused is False
	assert {path.name for path in result.paths} == set(PARIHAKA_MAE_RESULT_FILES)
	assert {path.name for path in output_dir.iterdir()} == {
		*PARIHAKA_MAE_RESULT_FILES,
		unknown.name,
	}
	assert unknown.read_text(encoding='utf-8') == 'keep\n'
	assert all(path.stat().st_size < 10 * 1024 * 1024 for path in result.paths)
	assert not list(output_dir.glob('*.pt'))
	assert not list(output_dir.glob('*.npy'))
	assert not list(output_dir.glob('*.npz'))
	assert not list(output_dir.glob('*.log'))

	summary = json.loads((output_dir / PARIHAKA_MAE_RESULT_FILES[0]).read_text())
	checkpoint = json.loads(
		(output_dir / 'parihaka_mae_checkpoint_summary.json').read_text()
	)
	serialized = json.dumps(summary, sort_keys=True)
	assert '/workspace/' not in serialized
	assert '/home/' not in serialized
	assert '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}' in serialized
	assert '${PARIHAKA_DATA_ROOT}' in serialized
	assert summary['input_boundary']['labels_used'] is False
	assert summary['scientific_scope']['kind'].startswith('survey-specific')
	assert summary['training_completion'] == {'epoch': 100, 'global_step': 250_000}
	assert checkpoint['latest']['sha256']
	assert checkpoint['latest']['epoch'] == 100
	assert checkpoint['latest']['global_step'] == 250_000
	assert checkpoint['prepared_amplitude']['shape_xyz'] == [3, 4, 5]
	assert checkpoint['precision']['resolved'] == 'bfloat16'


def test_results_second_identical_run_preserves_bytes_and_mtime(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	kwargs, _output_dir = _results_fixture(tmp_path, monkeypatch)
	first = summarize_parihaka_mae(**kwargs)
	before = {
		path: (path.read_bytes(), path.stat().st_mtime_ns) for path in first.paths
	}

	second = summarize_parihaka_mae(**kwargs)

	assert second.reused is True
	assert {
		path: (path.read_bytes(), path.stat().st_mtime_ns) for path in second.paths
	} == before


def test_results_use_validation_evidence_without_loading_checkpoints(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	kwargs, _output_dir = _results_fixture(tmp_path, monkeypatch)
	monkeypatch.setattr(
		mae_checkpoint_module,
		'load_checkpoint',
		lambda *_args, **_kwargs: pytest.fail(
			'results producer loaded a checkpoint after validation'
		),
	)

	summarize_parihaka_mae(**kwargs)


def test_results_reject_conflict_without_touching_other_targets(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	kwargs, output_dir = _results_fixture(tmp_path, monkeypatch)
	first = summarize_parihaka_mae(**kwargs)
	conflict = output_dir / PARIHAKA_MAE_RESULT_FILES[0]
	conflict.write_text('foreign\n', encoding='utf-8')
	before = {path: path.read_bytes() for path in first.paths}
	before[conflict] = b'foreign\n'

	with pytest.raises(FileExistsError, match='--overwrite'):
		summarize_parihaka_mae(**kwargs)

	assert {path: path.read_bytes() for path in first.paths} == before


def test_results_validation_failure_writes_nothing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	output_dir = tmp_path / 'results'
	monkeypatch.setattr(
		results_module,
		'validate_parihaka_mae',
		lambda **_kwargs: (_ for _ in ()).throw(ValueError('checkpoint drift')),
	)

	with pytest.raises(ValueError, match='checkpoint drift'):
		summarize_parihaka_mae(
			prepare_config_path=tmp_path / 'prepare.yaml',
			full_config_path=tmp_path / '02_full_100ep.yaml',
			output_dir=output_dir,
		)

	assert not output_dir.exists()


def _results_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], Path]:
	inputs, base = _full_fixture(tmp_path / 'live', monkeypatch)
	validation = validation_module._validate_full(inputs, base=base)  # noqa: SLF001
	prepare_path = tmp_path / 'prepare.yaml'
	full_path = tmp_path / 'pretrain' / '02_full_100ep.yaml'
	monkeypatch.setattr(
		results_module,
		'validate_parihaka_mae',
		lambda **_kwargs: validation,
	)
	monkeypatch.setattr(
		results_module,
		'load_config',
		lambda path: {'prepare': True} if Path(path) == prepare_path else inputs.full,
	)
	monkeypatch.setattr(
		results_module,
		'parihaka_prepare_volume_config_from_mapping',
		lambda _mapping: inputs.prepare,
	)
	monkeypatch.setattr(
		results_module,
		'resolve_mae_training_config',
		lambda config: config,
	)
	monkeypatch.setattr(results_module, '_git_sha', lambda: 'a' * 40)
	monkeypatch.setattr(results_module, '_git_dirty', lambda: True)
	output_dir = tmp_path / 'results'
	return (
		{
			'prepare_config_path': prepare_path,
			'full_config_path': full_path,
			'output_dir': output_dir,
			'execution_classification': 'fresh',
		},
		output_dir,
	)
