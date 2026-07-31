"""Focused contracts for the immutable XY screening preflight audit."""
# ruff: noqa: SLF001, TC003

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology import (
	xy_neighbor_consensus_screening_audit as audit,
)


def test_same_z_xy_edges_are_undirected_once_and_exclude_invalid_endpoints() -> None:
	"""Count x/y neighbours once, never across invalid endpoints or z."""
	source = np.array(
		[
			[[0, 0], [1, 1]],
			[[1, 0], [1, 2]],
		],
		dtype=np.int32,
	)
	output = source.copy()
	output[1, 1, 1] = 1
	valid = np.ones((2, 2, 2), dtype=np.bool_)
	valid[0, 1, 1] = False

	evidence = audit._spatial_one(source, output, valid, k=3)

	assert evidence['x_edges']['valid_edge_count'] == 3
	assert evidence['y_edges']['valid_edge_count'] == 3
	assert evidence['x_edges']['source_disagreement_count'] == 1
	assert evidence['y_edges']['source_disagreement_count'] == 2
	assert evidence['x_edges']['output_disagreement_count'] == 1
	assert evidence['y_edges']['output_disagreement_count'] == 2
	assert evidence['valid_token_count'] == 7
	assert evidence['changed_token_count'] == 1
	assert evidence['source_state_occupancy'] == [3, 3, 1]
	assert evidence['output_state_occupancy'] == [3, 4, 0]
	assert evidence['source_temporal_transition_count'] == 2
	assert evidence['output_temporal_transition_count'] == 1
	assert evidence['ordered_path_violations'] == {'source': 1, 'output': 1}


def test_spatial_merge_keeps_unfavourable_metrics_descriptive() -> None:
	"""An increased disagreement/transition count must not become an audit gate."""
	item = {
		'x_edges': _edge(10, 2, 4),
		'y_edges': _edge(10, 2, 5),
		'valid_token_count': 10,
		'changed_token_count': 4,
		'source_state_occupancy': [5, 5],
		'output_state_occupancy': [10, 0],
		'source_temporal_transition_count': 1,
		'output_temporal_transition_count': 9,
		'ordered_path_violations': {'source': 0, 'output': 0},
	}

	merged = audit._merge_spatial((item,), k=2)

	assert merged['combined']['output_disagreement_count'] == 9
	assert merged['output_temporal_transition_count'] == 9
	assert merged['empty_output_state_count'] == 1


def test_parity_rejects_config_runtime_and_optimizer_drift() -> None:
	"""Only representation fields may differ and CPU identity must match."""
	hard = _training_config()
	candidate = _training_config()
	candidate['identity']['model_tag'] = 'candidate'
	candidate['identity']['scientific_identity'] = {
		'experiment_role': 'candidate',
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
	}
	candidate['pseudo_targets']['manifest'] = 'candidate.json'
	candidate['pseudo_targets']['target_representation'] = (
		'xy_neighbor_consensus_hard_labels_v1'
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
	xy_handoff = {
		'targets': {
			'initial_student_state_sha256': 'a' * 64,
			'initial_head_state_sha256': 'b' * 64,
		}
	}
	audit._validate_runtime_parity(
		runtime,
		dict(runtime),
		hard_handoff=hard_handoff,
		candidate_handoff=xy_handoff,
	)
	broken = dict(runtime, optimizer_group_identity=[])
	with pytest.raises(ValueError, match='optimizer group mismatch'):
		audit._validate_runtime_parity(
			runtime,
			broken,
			hard_handoff=hard_handoff,
			candidate_handoff=xy_handoff,
		)


def test_only_missing_preserves_identical_audit_bytes_and_mtime(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A complete immutable audit is revalidated but never rewritten."""
	output = tmp_path / 'audit.json'
	config = _config(tmp_path, output)
	payload = _payload()
	monkeypatch.setattr(audit, '_clean_git_identity', lambda _path: payload['git'])
	monkeypatch.setattr(audit, '_audit_payload', lambda *_args, **_kwargs: payload)

	first = audit.audit_f3_xy_neighbor_consensus_screening(config)
	before, mtime = output.read_bytes(), output.stat().st_mtime_ns
	second = audit.audit_f3_xy_neighbor_consensus_screening(config, only_missing=True)

	assert first.action == 'WRITTEN'
	assert second.action == 'REUSE_COMPLETED'
	assert output.read_bytes() == before
	assert output.stat().st_mtime_ns == mtime


def test_incompatible_audit_requires_explicit_quarantine(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Never silently overwrite incompatible preflight evidence."""
	output = tmp_path / 'audit.json'
	config = _config(tmp_path, output)
	payload = _payload()
	output.write_text(json.dumps(dict(payload, status='HOLD')), encoding='utf-8')
	monkeypatch.setattr(audit, '_clean_git_identity', lambda _path: payload['git'])
	monkeypatch.setattr(audit, '_audit_payload', lambda *_args, **_kwargs: payload)

	with pytest.raises(ValueError, match='incompatible existing'):
		audit.audit_f3_xy_neighbor_consensus_screening(config, only_missing=True)
	result = audit.audit_f3_xy_neighbor_consensus_screening(
		config, only_missing=True, quarantine_invalid=True
	)

	assert result.quarantine_path is not None
	assert result.quarantine_path.is_file()
	assert result.action == 'WRITTEN'


def _edge(count: int, source: int, output: int) -> dict[str, object]:
	source_fraction = source / count
	output_fraction = output / count
	return {
		'valid_edge_count': count,
		'source_disagreement_count': source,
		'source_disagreement_fraction': source_fraction,
		'output_disagreement_count': output,
		'output_disagreement_fraction': output_fraction,
		'absolute_disagreement_reduction': source_fraction - output_fraction,
		'relative_disagreement_reduction': (
			(source_fraction - output_fraction) / source_fraction
		),
	}


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
	tmp_path: Path, output: Path
) -> audit.F3XYNeighborConsensusScreeningAuditConfig:
	return audit.F3XYNeighborConsensusScreeningAuditConfig(
		artifact_root=tmp_path,
		workspace_root=tmp_path,
		source_hard_manifest=tmp_path / 'source.json',
		xy_target_manifest=tmp_path / 'target.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_pretraining_handoff=tmp_path / 'hard.json',
		candidate_full_config=tmp_path / 'candidate.yaml',
		candidate_pretraining_handoff=tmp_path / 'candidate.json',
		candidate_embeddings_dir=tmp_path,
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
		'xy_spatial_smoothness': {'per_k': {}},
	}
