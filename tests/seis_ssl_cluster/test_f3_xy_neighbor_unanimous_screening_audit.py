"""Focused contracts for the immutable unanimous screening preflight."""
# ruff: noqa: SLF001, TC003

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from seis_ssl_cluster.f3.lithology import (
	xy_neighbor_unanimous_screening_audit as audit,
)


def test_parity_rejects_config_and_optimizer_drift() -> None:
	hard = _training_config()
	candidate = _training_config()
	candidate['identity']['model_tag'] = 'candidate'
	candidate['identity']['scientific_identity'] = {
		'experiment_role': 'candidate',
		'target_representation': 'xy_neighbor_unanimous_hard_labels_v1',
	}
	candidate['pseudo_targets']['manifest'] = 'candidate.json'
	candidate['pseudo_targets']['target_representation'] = (
		'xy_neighbor_unanimous_hard_labels_v1'
	)
	audit._validate_allowed_config_delta(hard, candidate)
	candidate['train']['lr'] = 0.2
	with pytest.raises(ValueError, match='config drift'):
		audit._validate_allowed_config_delta(hard, candidate)

	runtime = {
		'initial_student_state_sha256': 'a' * 64,
		'initial_head_state_sha256': 'b' * 64,
		'trainability_summary': {'trainable_parameter_count': 1},
		'optimizer_group_identity': [{'name': 'head'}],
	}
	hard_handoff = {
		'stratigraphy_pretext': {
			'initial_student_state_sha256': 'a' * 64,
			'initial_head_state_sha256': 'b' * 64,
		}
	}
	unanimous_handoff = {
		'targets': {
			'initial_student_state_sha256': 'a' * 64,
			'initial_head_state_sha256': 'b' * 64,
		}
	}
	audit._validate_runtime_parity(
		runtime,
		dict(runtime),
		hard_handoff=hard_handoff,
		candidate_handoff=unanimous_handoff,
	)
	with pytest.raises(ValueError, match='optimizer group mismatch'):
		audit._validate_runtime_parity(
			runtime,
			dict(runtime, optimizer_group_identity=[]),
			hard_handoff=hard_handoff,
			candidate_handoff=unanimous_handoff,
		)


def test_only_missing_preserves_identical_audit_bytes_and_mtime(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	output = tmp_path / 'audit.json'
	config = _config(tmp_path, output)
	payload = _payload()
	monkeypatch.setattr(audit, '_clean_git_identity', lambda _path: payload['git'])
	monkeypatch.setattr(audit, '_audit_payload', lambda *_args, **_kwargs: payload)

	first = audit.audit_f3_xy_neighbor_unanimous_screening(config)
	before, mtime = output.read_bytes(), output.stat().st_mtime_ns
	second = audit.audit_f3_xy_neighbor_unanimous_screening(
		config, only_missing=True
	)

	assert first.action == 'WRITTEN'
	assert second.action == 'REUSE_COMPLETED'
	assert output.read_bytes() == before
	assert output.stat().st_mtime_ns == mtime


def test_incompatible_audit_requires_explicit_quarantine(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	output = tmp_path / 'audit.json'
	config = _config(tmp_path, output)
	payload = _payload()
	output.write_text(json.dumps(dict(payload, status='HOLD')), encoding='utf-8')
	monkeypatch.setattr(audit, '_clean_git_identity', lambda _path: payload['git'])
	monkeypatch.setattr(audit, '_audit_payload', lambda *_args, **_kwargs: payload)

	with pytest.raises(ValueError, match='incompatible existing'):
		audit.audit_f3_xy_neighbor_unanimous_screening(config, only_missing=True)
	result = audit.audit_f3_xy_neighbor_unanimous_screening(
		config, only_missing=True, quarantine_invalid=True
	)
	assert result.quarantine_path is not None
	assert result.quarantine_path.is_file()
	assert (
		json.loads(result.quarantine_path.read_text(encoding='utf-8'))['status']
		== 'HOLD'
	)
	assert json.loads(output.read_text(encoding='utf-8')) == payload
	assert result.action == 'WRITTEN'


def test_target_audit_requires_go_and_exact_three_artifact_identities(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	paths = {
		'source_hard_manifest': tmp_path / 'source.json',
		'xy_neighbor_consensus_target_manifest': tmp_path / 'consensus.json',
		'xy_neighbor_unanimous_target_manifest': tmp_path / 'unanimous.json',
	}
	for path in paths.values():
		path.write_text('{}', encoding='utf-8')
	config = _config(tmp_path, tmp_path / 'output.json', **paths)
	payload = {
		'artifact_type': 'f3_xy_neighbor_unanimous_target_audit',
		'schema_version': 1,
		'status': 'XYUNANIM_TARGET_GO',
		**{
			key: audit._target_audit_identity(path)
			for key, path in paths.items()
		},
		'go_conditions': {'6': {'all': True}, '8': {'all': True}, '10': {'all': True}},
	}
	calls: list[tuple[Path, Path]] = []

	def replay(path: Path, *, artifact_root: Path) -> dict[str, object]:
		calls.append((path, artifact_root))
		return payload

	monkeypatch.setattr(
		audit, 'replay_f3_xy_neighbor_unanimous_target_audit', replay
	)
	result = audit._validate_target_audit(
		config,
		source={'head_ks': [6, 8, 10]},
		consensus={'head_ks': [6, 8, 10]},
		unanimous={'head_ks': [6, 8, 10]},
	)
	assert result['status'] == 'XYUNANIM_TARGET_GO'
	assert calls == [(config.target_audit, config.artifact_root)]
	payload['status'] = 'XYUNANIM_TARGET_HOLD'
	with pytest.raises(ValueError, match='must be XYUNANIM_TARGET_GO'):
		audit._validate_target_audit(
			config,
			source={'head_ks': [6, 8, 10]},
			consensus={'head_ks': [6, 8, 10]},
			unanimous={'head_ks': [6, 8, 10]},
		)


def test_hard_config_source_manifest_hash_fallback_is_scoped(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	name = 'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256'
	source = tmp_path / 'source.json'
	source.write_text('{"frozen": true}\n', encoding='utf-8')
	config = _config(tmp_path, tmp_path / 'output.json', source_hard_manifest=source)
	seen: list[tuple[Path, str | None]] = []

	def load(path: Path) -> dict[str, object]:
		seen.append((path, os.environ.get(name)))
		return {'resolved': True}

	monkeypatch.setattr(audit, 'load_config', load)
	monkeypatch.setattr(
		audit, 'resolve_strat_hmm_pretext_config', lambda value: value
	)
	monkeypatch.delenv(name, raising=False)
	assert audit._load_hard_full_config(config) == {'resolved': True}
	assert seen == [(config.hard_full_config, audit.file_sha256(source))]
	assert name not in os.environ

	monkeypatch.setenv(name, 'pre-existing-value')
	assert audit._load_hard_full_config(config) == {'resolved': True}
	assert os.environ[name] == 'pre-existing-value'


def test_screening_binding_replays_target_audit_and_handoff_lineage(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	target_audit_path = tmp_path / 'target-audit.json'
	source_path = tmp_path / 'source.json'
	target_path = tmp_path / 'unanimous.json'
	handoff_path = tmp_path / 'handoff.json'
	embeddings_root = tmp_path / 'embeddings'
	embeddings_root.mkdir()
	for path in (target_audit_path, source_path, target_path, handoff_path):
		path.write_text('{}', encoding='utf-8')
	files = {}
	for name in ('embeddings', 'valid_tokens', 'metadata'):
		path = embeddings_root / f'{name}.bin'
		path.write_text(name, encoding='utf-8')
		files[name] = audit._identity(path)
	target_audit = {
		**audit._identity(target_audit_path),
		'status': 'XYUNANIM_TARGET_GO',
	}
	payload = {
		'artifact_type': audit.ARTIFACT_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'artifact_root': str(tmp_path),
		'candidate': {
			'model_id': audit.XY_UNANIM_MODEL_ID,
			'model_tag': audit.XY_UNANIM_MODEL_TAG,
			'pretraining_handoff': audit._identity(handoff_path),
			'embeddings': {'root': str(embeddings_root), **files},
		},
		'hard_baseline_parity': {'status': 'PASS'},
		'target_audit': target_audit,
		'source_hard_manifest': audit._identity(source_path),
		'xy_neighbor_unanimous_target_manifest': audit._identity(target_path),
	}
	calls: list[tuple[Path, Path]] = []

	def replay(path: Path, *, artifact_root: Path) -> dict[str, object]:
		calls.append((path, artifact_root))
		return {'status': 'XYUNANIM_TARGET_GO'}

	monkeypatch.setattr(
		audit, 'replay_f3_xy_neighbor_unanimous_target_audit', replay
	)
	monkeypatch.setattr(
		audit,
		'load_f3_xy_neighbor_unanimous_pretraining_handoff',
		lambda _path: {
			'targets': {
				'target_audit': _reference(target_audit_path),
				'source_hard_manifest': _reference(source_path),
				'target_manifest': _reference(target_path),
			}
		},
	)

	audit.validate_f3_xy_neighbor_unanimous_screening_audit_binding(
		payload,
		model_id=audit.XY_UNANIM_MODEL_ID,
		model_tag=audit.XY_UNANIM_MODEL_TAG,
		pretraining_handoff=handoff_path,
		embeddings_dir=embeddings_root,
	)
	assert calls == [(target_audit_path, tmp_path)]


def test_reference_manifests_require_completed_read_only_role_matrices(
	tmp_path: Path,
) -> None:
	paths = {
		'xy_neighbor_consensus_run_manifest': tmp_path / 'xycons.json',
		'hard_reference_run_manifest': tmp_path / 'hard.json',
		'current_k6_run_manifest': tmp_path / 'current.json',
		'mae_reference_run_manifest': tmp_path / 'mae.json',
	}
	specs = (
		(
			'xy_neighbor_consensus_run_manifest',
			'f3_lithology_voxel_label_budget_xy_neighbor_consensus',
			'mh_xycons1_nocons',
		),
		(
			'hard_reference_run_manifest',
			'f3_lithology_voxel_label_budget_multi_head',
			'mh_nocons',
		),
		(
			'current_k6_run_manifest',
			'f3_lithology_voxel_label_budget_current_k6_control',
			'm1_current_k6',
		),
		(
			'mae_reference_run_manifest',
			'f3_lithology_voxel_label_budget_run_manifest',
			'mae',
		),
	)
	for key, artifact_type, role in specs:
		paths[key].write_text(
			json.dumps(
				{
					'artifact_type': artifact_type,
					'schema_version': 1,
					'rows': [
						{'model_role': role, 'status': 'complete'} for _ in range(15)
					],
				}
			),
			encoding='utf-8',
		)
	config = _config(tmp_path, tmp_path / 'output.json')
	config = audit.F3XYNeighborUnanimousScreeningAuditConfig(
		**{**config.__dict__, **paths}
	)
	assert set(audit._reference_run_manifests(config)) == {
		'xy_neighbor_consensus',
		'hard_multi_head',
		'current_k6',
		'mae',
	}
	paths['mae_reference_run_manifest'].write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_voxel_label_budget_run_manifest',
				'schema_version': 1,
				'rows': [{'model_role': 'mae', 'status': 'complete'}],
			}
		),
		encoding='utf-8',
	)
	with pytest.raises(ValueError, match='matrix is incomplete'):
		audit._reference_run_manifests(config)


def _training_config() -> dict[str, object]:
	return {
		'paths': {'output_root': 'hard-output'},
		'identity': {
			'model_tag': 'hard',
			'scientific_identity': {'experiment_role': 'hard'},
		},
		'pseudo_targets': {'manifest': 'hard.json'},
		'train': {'lr': 0.1},
	}


def _config(
	tmp_path: Path,
	output: Path,
	**paths: Path,
) -> audit.F3XYNeighborUnanimousScreeningAuditConfig:
	return audit.F3XYNeighborUnanimousScreeningAuditConfig(
		artifact_root=tmp_path,
		workspace_root=tmp_path,
		source_hard_manifest=paths.get(
			'source_hard_manifest', tmp_path / 'source.json'
		),
		xy_neighbor_consensus_target_manifest=paths.get(
			'xy_neighbor_consensus_target_manifest', tmp_path / 'consensus.json'
		),
		xy_neighbor_unanimous_target_manifest=paths.get(
			'xy_neighbor_unanimous_target_manifest', tmp_path / 'unanimous.json'
		),
		target_audit=tmp_path / 'target-audit.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_pretraining_handoff=tmp_path / 'hard.json',
		candidate_full_config=tmp_path / 'candidate.yaml',
		candidate_pretraining_handoff=tmp_path / 'candidate.json',
		candidate_embeddings_dir=tmp_path,
		xy_neighbor_consensus_run_manifest=tmp_path / 'xycons-runs.json',
		hard_reference_run_manifest=tmp_path / 'hard-runs.json',
		current_k6_run_manifest=tmp_path / 'current-runs.json',
		mae_reference_run_manifest=tmp_path / 'mae-runs.json',
		output_path=output,
	)


def _payload() -> dict[str, object]:
	return {
		'artifact_type': audit.ARTIFACT_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'git': {'git_sha': 'a' * 40, 'dirty': False},
		'candidate': {},
		'hard_baseline_parity': {'status': 'PASS'},
		'xy_neighbor_unanimous_spatial_smoothness': {'per_k': {}},
		'target_audit': {'status': 'XYUNANIM_TARGET_GO'},
		'reference_run_manifests': {},
	}


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': audit.file_sha256(path)}
