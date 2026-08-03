"""Focused contracts for center-trace screening audit and recovery."""
# ruff: noqa: CPY001, I001, SLF001, TC002, TC003

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seis_ssl_cluster.embedding import extractor
from seis_ssl_cluster.f3 import center_trace_masked_screening_audit as audit
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked as runner,
	voxel_label_budget_multi_head as shared,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJobPlan,
)


def test_screening_audit_dry_run_and_only_missing_reuse(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _audit_config(tmp_path)
	payload = _audit_payload()
	monkeypatch.setattr(audit, '_audit_payload', lambda *_args, **_kwargs: payload)
	monkeypatch.setattr(audit, '_clean_git_identity', lambda _root: {})

	dry_run = audit.audit_f3_center_trace_masked_screening(config, dry_run=True)
	assert dry_run.action == 'DRY_RUN'
	assert not config.output_path.exists()

	first = audit.audit_f3_center_trace_masked_screening(config)
	stamp = config.output_path.stat().st_mtime_ns
	assert first.action == 'WRITTEN'
	second = audit.audit_f3_center_trace_masked_screening(config, only_missing=True)
	assert second.action == 'REUSE_COMPLETED'
	assert config.output_path.stat().st_mtime_ns == stamp


def test_screening_audit_quarantines_only_invalid_owned_output(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _audit_config(tmp_path)
	config.output_path.parent.mkdir(parents=True, exist_ok=True)
	config.output_path.write_text('{"partial": true}\n', encoding='utf-8')
	monkeypatch.setattr(
		audit, '_audit_payload', lambda *_args, **_kwargs: _audit_payload()
	)
	monkeypatch.setattr(audit, '_clean_git_identity', lambda _root: {})

	result = audit.audit_f3_center_trace_masked_screening(
		config, quarantine_invalid=True
	)

	assert result.action == 'WRITTEN'
	assert result.quarantine_path is not None
	assert result.quarantine_path.is_file()
	assert config.output_path.is_file()


def test_persisted_screening_audit_load_revalidates_by_default(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	payload = _audit_payload()
	path = tmp_path / 'audit.json'
	path.write_text(json.dumps(payload), encoding='utf-8')
	seen: list[object] = []

	def record(value: object) -> None:
		seen.append(value)

	monkeypatch.setattr(
		audit,
		'_revalidate_persisted_audit',
		record,
	)

	assert audit.load_f3_center_trace_masked_screening_audit(path) == payload
	assert seen == [payload]


def test_embedding_execution_summary_declares_unmasked_encoder_mode(
	tmp_path: Path,
) -> None:
	extractor._write_embedding_execution_summary(
		tmp_path, [SimpleNamespace(skipped=True)]
	)
	payload = json.loads(
		(tmp_path / 'embedding_extraction_execution.json').read_text(
			encoding='utf-8'
		)
	)

	assert payload['encoder_input_mode'] == extractor.UNMASKED_ENCODER_INPUT_MODE


def test_center_trace_job_matrix_is_exactly_three_by_five(tmp_path: Path) -> None:
	candidate = SimpleNamespace(
		model_id='mh_ctmask010_nocons',
		model_tag='strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1',
		embeddings_dir=tmp_path / 'embeddings',
		pretraining_handoff=tmp_path / 'handoff.json',
	)
	config = SimpleNamespace(
		candidates=(candidate,),
		budgets=('cap25', 'cap50', 'cap100'),
		subsample_seeds=(0, 1, 2, 3, 4),
		output_root=tmp_path / 'outputs',
		decoder_seed=lambda seed: 42000 + seed,
	)
	dataset_rows = {
		(budget, seed): {
			'per_class_cap': int(budget.removeprefix('cap')),
			'voxel_dataset_root': str(tmp_path / budget / str(seed)),
		}
		for budget in config.budgets
		for seed in config.subsample_seeds
	}

	jobs = shared._jobs(config, dataset_rows)

	assert len(jobs) == 15
	assert {job.model_role for job in jobs} == {'mh_ctmask010_nocons'}
	assert {job.decoder_seed for job in jobs} == {42000, 42001, 42002, 42003, 42004}


def test_center_trace_resume_state_is_not_generic_latest_state(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	job = SimpleNamespace(model_role='mh_ctmask010_nocons')
	monkeypatch.setattr(
		runner.shared, '_stage_config', lambda *_args: SimpleNamespace()
	)
	monkeypatch.setattr(
		runner,
		'classify_voxel_label_budget_job',
		lambda *_args, **_kwargs: VoxelLabelBudgetJobPlan(
			job, 'RESUME_LATEST', 'same identity', 123
		),
	)

	plan = runner._classify_center_job(SimpleNamespace(), job, estimated_bytes=123)

	assert plan.state == 'RESUME_SAME_IDENTITY'


def test_center_trace_runner_revalidates_cached_screening_audit(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = SimpleNamespace(screening_audit_payload={'cached': True})
	seen: list[object] = []
	monkeypatch.setattr(
		runner.center_config,
		'validate_f3_center_trace_masked_screening_audit',
		seen.append,
	)

	runner._validate_screening_audit(config)

	assert seen == [config]


def _audit_config(tmp_path: Path) -> audit.F3CenterTraceMaskedScreeningAuditConfig:
	artifact_root = tmp_path / 'artifacts'
	workspace_root = tmp_path / 'workspace'
	artifact_root.mkdir()
	workspace_root.mkdir()
	return audit.F3CenterTraceMaskedScreeningAuditConfig(
		artifact_root=artifact_root,
		workspace_root=workspace_root,
		source_hard_manifest=artifact_root / 'target.json',
		hard_full_config=workspace_root / 'hard.yaml',
		hard_pretraining_handoff=artifact_root / 'hard_handoff.json',
		candidate_full_config=workspace_root / 'candidate.yaml',
		candidate_pretraining_handoff=artifact_root / 'candidate_handoff.json',
		candidate_embeddings_dir=artifact_root / 'embeddings',
		output_path=artifact_root / 'audit.json',
	)


def _audit_payload() -> dict[str, object]:
	return {
		'artifact_type': audit.ARTIFACT_TYPE,
		'schema_version': audit.SCHEMA_VERSION,
		'status': 'PASS',
		**{
			key: {}
			for key in (
				'candidate',
				'handoff_contract',
				'checkpoint',
				'embedding',
				'valid_mask_parity',
				'hard_baseline_parity',
				'reference_run_manifests',
				'dataset_job_pairing',
				'git',
			)
		},
	}
