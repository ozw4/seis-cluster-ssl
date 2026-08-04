# ruff: noqa: CPY001, TC003

from __future__ import annotations

import json
import os
from pathlib import Path

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
