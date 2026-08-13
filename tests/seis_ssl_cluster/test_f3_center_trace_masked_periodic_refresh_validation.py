# ruff: noqa: TC003

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.f3 import (
	center_trace_masked_periodic_refresh_validation as validation,
)


def test_checkpoint_report_reuses_exact_and_quarantines_stale(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'full'
	report = validation._write_checkpoint_report(  # noqa: SLF001
		root,
		checkpoint_sha256='a' * 64,
		only_missing=False,
		quarantine_invalid=False,
	)
	report_bytes = report.read_bytes()
	os.utime(report, ns=(1_700_000_000_000_000_000,) * 2)
	report_mtime = report.stat().st_mtime_ns

	assert validation._write_checkpoint_report(  # noqa: SLF001
		root,
		checkpoint_sha256='a' * 64,
		only_missing=True,
		quarantine_invalid=False,
	) == report
	assert report.read_bytes() == report_bytes
	assert report.stat().st_mtime_ns == report_mtime

	report.write_text(json.dumps({'status': 'STALE'}), encoding='utf-8')
	with pytest.raises(ValueError, match='quarantine-invalid'):
		validation._write_checkpoint_report(  # noqa: SLF001
			root,
			checkpoint_sha256='a' * 64,
			only_missing=False,
			quarantine_invalid=False,
		)

	validation._write_checkpoint_report(  # noqa: SLF001
		root,
		checkpoint_sha256='a' * 64,
		only_missing=False,
		quarantine_invalid=True,
	)
	assert json.loads(report.read_text(encoding='utf-8'))['status'] == 'PASS'
	quarantined = list(report.parent.glob(f'{report.name}.quarantine.*'))
	assert len(quarantined) == 1
	assert quarantined[0].read_text(encoding='utf-8') == '{"status": "STALE"}'


def test_execution_marker_rejects_corrupt_and_quarantines_before_recovery(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = SimpleNamespace(experiment_root=tmp_path)
	path = validation._execution_evidence_path(config)  # noqa: SLF001
	path.write_text('{not-json', encoding='utf-8')
	monkeypatch.setattr(
		validation,
		'_execution_binding',
		lambda _config: {'binding': 'current'},
	)
	monkeypatch.setattr(
		validation,
		'_execution_identity',
		lambda: {'git_commit': 'a' * 40, 'git_status_short': []},
	)

	with pytest.raises(ValueError, match='quarantine-invalid'):
		validation._start_execution_evidence(  # noqa: SLF001
			config, dry_run=False, quarantine_invalid=False
		)

	validation._start_execution_evidence(  # noqa: SLF001
		config, dry_run=False, quarantine_invalid=True
	)
	quarantined = list(tmp_path.glob(f'{path.name}.quarantine.*'))
	assert len(quarantined) == 1
	assert quarantined[0].read_text(encoding='utf-8') == '{not-json'
	assert json.loads(path.read_text(encoding='utf-8'))['phase'] == 'inputs'


@pytest.mark.parametrize('phase', ['smoke', 'complete'])
@pytest.mark.parametrize('marker_binding', ['current', 'foreign'])
def test_execution_marker_never_quarantines_completed_phase(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	phase: str,
	marker_binding: str,
) -> None:
	config = SimpleNamespace(experiment_root=tmp_path)
	path = validation._execution_evidence_path(config)  # noqa: SLF001
	state = {
		'git_commit': 'a' * 40,
		'git_status_short': [],
		'git_diff_sha256': 'b' * 64,
	}
	monkeypatch.setattr(
		validation,
		'_execution_binding',
		lambda _config: {'binding': 'current'},
	)
	path.write_text(
		json.dumps(
			{
				'artifact_type': validation._EXECUTION_ARTIFACT_TYPE,  # noqa: SLF001
				'schema_version': 1,
				'phase': phase,
				'binding': {'binding': marker_binding},
				'execution': {'before': state, 'after': state},
			}
		),
		encoding='utf-8',
	)
	original = path.read_bytes()

	with pytest.raises(ValueError, match=f'existing periodic {phase}'):
		validation._start_execution_evidence(  # noqa: SLF001
			config, dry_run=False, quarantine_invalid=True
		)

	assert path.read_bytes() == original
	assert list(tmp_path.glob(f'{path.name}.quarantine.*')) == []


def test_complete_dry_run_validates_smoke_phase_evidence(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = SimpleNamespace(periodic_refresh_smoke_config=tmp_path / 'smoke.yaml')
	calls: list[tuple[object, object]] = []
	monkeypatch.setattr(validation, '_training_config', lambda _path: {})
	monkeypatch.setattr(
		validation,
		'_inputs_evidence',
		lambda _config: {'target_manifest': {'path': 'target', 'sha256': 'a' * 64}},
	)
	monkeypatch.setattr(
		validation,
		'_checkpoint_evidence',
		lambda *_args, **_kwargs: {'checkpoint': {'root': str(tmp_path)}},
	)
	monkeypatch.setattr(
		validation,
		'_embedding_evidence',
		lambda *_args, **_kwargs: {'embeddings': 'validated'},
	)
	monkeypatch.setattr(
		validation,
		'_validate_smoke_phase_evidence',
		lambda _config, *, inputs: calls.append((_config, inputs)),
	)
	monkeypatch.setattr(
		validation,
		'_update_execution_evidence',
		lambda *_args, **_kwargs: {'before': {}, 'after': {}},
	)
	monkeypatch.setattr(validation, '_handoff', lambda _evidence: {})

	result = validation.validate_f3_center_trace_masked_periodic_refresh(
		config,
		phase='complete',
		dry_run=True,
	)

	assert result.evidence['status'] == 'PASS'
	assert calls[0][0] is config
	assert calls[0][1]['target_manifest'] == result.evidence['target_manifest']


def test_handoff_initial_target_manifest_is_a_loadable_reference(
	tmp_path: Path,
) -> None:
	target = tmp_path / 'target.json'
	target.write_text('{}\n', encoding='utf-8')
	checkpoint = tmp_path / 'checkpoint.pt'
	checkpoint.write_bytes(b'checkpoint')
	digest = file_sha256(target)
	evidence = {
		'target_manifest': {
			'path': str(target),
			'sha256': digest,
			'per_head_target_hashes': {'6': {}, '8': {}, '10': {}},
			'common_valid_token_hashes': {'survey': 'a' * 64},
		},
		'checkpoint': {
			'path': str(checkpoint),
			'sha256': 'b' * 64,
			'latest_path': str(checkpoint),
			'latest_sha256': 'c' * 64,
			'epoch': 25,
			'global_step': 1,
			'schema_version': 8,
			'scientific_identity_sha256': 'd' * 64,
			'target_refresh_state_sha256': 'e' * 64,
			'optimizer_group_identity': [],
			'initial_student_state_sha256': 'f' * 64,
			'initial_head_state_sha256': '0' * 64,
			'initial_spatial_context_state_sha256': '1' * 64,
		},
		'refresh': {
			'generations': [
				{
					'generation_index': 7,
					'generation_id': 'refresh_0007_epoch020',
					'manifest_path': str(target),
					'manifest_sha256': digest,
					'generation_content_sha256': '2' * 64,
				}
			],
			'final_target_manifest': {'path': str(target), 'sha256': digest},
			'chain_path': str(target),
			'chain_sha256': digest,
		},
		'embedding': {},
		'fixed_preprocessing': {},
		'execution': {},
	}

	handoff = validation._handoff(evidence)  # noqa: SLF001
	initial_target = handoff['targets']['initial_hard_target_manifest']  # type: ignore[index]
	assert initial_target == {'path': str(target), 'sha256': digest}
	validation._validate_reference(  # noqa: SLF001
		initial_target, 'handoff initial target manifest'
	)


def test_initialization_checkpoint_hashes_reject_source_drift(
	tmp_path: Path,
) -> None:
	teacher = tmp_path / 'teacher.pt'
	student = tmp_path / 'student.pt'
	baseline = tmp_path / 'baseline.pt'
	teacher.write_bytes(b'teacher')
	student.write_bytes(b'student')
	torch.save(
		{
			'stratigraphy_checkpoint': {
				'teacher_checkpoint_sha256': file_sha256(teacher),
				'student_init_checkpoint_sha256': file_sha256(student),
			}
		},
		baseline,
	)
	training = {
		'teacher': {'checkpoint': str(teacher)},
		'student': {'init_checkpoint': str(student)},
	}

	evidence = validation._validate_initialization_checkpoint_hashes(  # noqa: SLF001
		baseline_handoff={
			'checkpoint': {
				'path': str(baseline),
				'sha256': file_sha256(baseline),
			}
		},
		trainings={'full': training},
	)
	assert evidence['configs']['full']['student_init_checkpoint_sha256'] == {
		'path': str(student.resolve()),
		'sha256': file_sha256(student),
	}

	student.write_bytes(b'drifted')
	with pytest.raises(ValueError, match='SHA-256 drift'):
		validation._validate_initialization_checkpoint_hashes(  # noqa: SLF001
			baseline_handoff={
				'checkpoint': {
					'path': str(baseline),
					'sha256': file_sha256(baseline),
				}
			},
			trainings={'full': training},
		)
