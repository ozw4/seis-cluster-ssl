"""Focused validation and lightweight-publication contracts for unanimity."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

import seis_ssl_cluster.f3.xy_neighbor_unanimous_pretraining_validation as validation
import seis_ssl_cluster.f3.xy_neighbor_unanimous_results as results


def test_validator_config_is_closed_and_requires_target_audit(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	experiment_root = artifact_root / 'pretraining/f3/facies_benchmark_v1'
	experiment_root.mkdir(parents=True)
	paths = {
		'target_manifest': artifact_root / 'target.json',
		'target_audit': artifact_root / 'audit.json',
		'hard_full_config': artifact_root / 'hard.yaml',
		'xy_neighbor_unanimous_smoke_config': artifact_root / 'smoke.yaml',
		'xy_neighbor_unanimous_full_config': artifact_root / 'full.yaml',
	}
	for path in paths.values():
		path.write_text('{}\n', encoding='utf-8')
	mapping = {
		'artifact_root': str(artifact_root),
		'experiment_root': str(experiment_root),
		**{key: str(value) for key, value in paths.items()},
	}

	resolved = (
		validation.f3_xy_neighbor_unanimous_pretraining_validation_config_from_mapping(
			mapping
		)
	)

	assert resolved.target_audit == paths['target_audit']
	with pytest.raises(ValueError, match='unknown unanimous'):
		validation.f3_xy_neighbor_unanimous_pretraining_validation_config_from_mapping(
			{**mapping, 'posterior_calibration': 'forbidden'}
		)
	with pytest.raises(ValueError, match='target_audit'):
		validation.f3_xy_neighbor_unanimous_pretraining_validation_config_from_mapping(
			{key: value for key, value in mapping.items() if key != 'target_audit'}
		)


@pytest.mark.parametrize(
	'phase',
	['targets', 'smoke', 'checkpoints', 'complete'],
)
def test_validator_phase_dispatches_without_publishing_partial_handoff(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	phase: str,
) -> None:
	config = _validation_config(tmp_path)
	full = {'paths': {'output_root': str(tmp_path / 'full')}}
	hard = {'paths': {'output_root': str(tmp_path / 'hard')}}
	smoke = {'paths': {'output_root': str(tmp_path / 'smoke')}}
	training_by_path = {
		config.xy_neighbor_unanimous_full_config: full,
		config.hard_full_config: hard,
		config.xy_neighbor_unanimous_smoke_config: smoke,
	}
	parity = {
		'hard_runtime': {},
		'candidate_runtime': {},
	}
	monkeypatch.setattr(
		validation,
		'load_multi_head_xy_neighbor_unanimous_target_manifest',
		lambda *_args: {},
	)
	monkeypatch.setattr(
		validation,
		'load_f3_xy_neighbor_unanimous_target_audit',
		lambda *_args: {},
	)
	monkeypatch.setattr(
		validation,
		'replay_f3_xy_neighbor_unanimous_target_audit',
		lambda *_args, **_kwargs: {},
	)
	monkeypatch.setattr(
		validation,
		'_training_config',
		lambda path, **_kwargs: training_by_path[Path(path)],
	)
	monkeypatch.setattr(
		validation,
		'_target_evidence',
		lambda *_args, **_kwargs: {'hard_baseline_config_parity': parity},
	)
	monkeypatch.setattr(
		validation, '_smoke_config_contract', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		validation,
		'_checkpoint_evidence',
		lambda *_args, **_kwargs: {
			'root': str(tmp_path / 'full'),
			'identity': {},
			'selected_path': str(tmp_path / 'full' / 'best.pt'),
			'selected_sha256': 'a' * 64,
			'selected_checkpoint_kind': 'epoch',
			'selected_epoch': 25,
			'selected_global_step': 25600,
			'selected_loss': 1.0,
			'initial_student_state_sha256': 'a' * 64,
			'initial_head_state_sha256': 'b' * 64,
		},
	)
	monkeypatch.setattr(
		validation, '_validate_initial_state_parity', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		validation,
		'_embedding_evidence',
		lambda *_args, **_kwargs: {'root': str(tmp_path / 'embeddings')},
	)
	monkeypatch.setattr(
		validation,
		'_handoff',
		lambda _evidence: {'artifact_type': 'test'},
	)

	result = validation.validate_f3_xy_neighbor_unanimous_pretraining(
		config,
		phase=phase,
		dry_run=True,
	)

	assert result.evidence['status'] == 'PASS'
	assert result.published_handoff is None
	if phase == 'complete':
		assert 'embedding' in result.evidence


def test_hard_label_metrics_require_zero_consistency_and_no_posterior() -> None:
	payload = {
		'metrics': {
			'loss': 1.0,
			'loss_consistency': 0.081,
			'loss_consistency_contribution': 0.0,
		},
		'stratigraphy_checkpoint': {'consistency_weight': 0.0},
	}
	validation._validate_hard_label_metrics(payload)  # noqa: SLF001
	payload['metrics']['loss_consistency_contribution'] = 0.001  # type: ignore[index]
	with pytest.raises(ValueError, match='consistency contribution'):
		validation._validate_hard_label_metrics(payload)  # noqa: SLF001
	payload['metrics']['loss_consistency_contribution'] = 0.0  # type: ignore[index]
	payload['metrics']['posterior_loss'] = 1.0  # type: ignore[index]
	with pytest.raises(ValueError, match='hard-label route'):
		validation._validate_hard_label_metrics(payload)  # noqa: SLF001


def test_hard_config_parity_allows_representation_scientific_identity_only(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Required fixed hard-route settings remain comparable across identities."""
	hard = {
		'paths': {'output_root': str(tmp_path / 'hard')},
		'identity': {
			'model_tag': 'hard',
			'scientific_identity': {
				'experiment_role': 'hard',
				'variant': 'nocons',
				'head_spec': 'multi_resolution_ordered_prototypes_v1',
				'head_ks': [6, 8, 10],
				'target_manifest_sha256': 'a' * 64,
				'consistency_policy': 'normalized_order_smooth_l1_v1',
			},
		},
		'pseudo_targets': {'manifest': str(tmp_path / 'hard-target.json')},
		'loss': {'prototype_weight': 1.0},
	}
	candidate = json.loads(json.dumps(hard))
	candidate['paths']['output_root'] = str(tmp_path / 'candidate')
	candidate['identity']['model_tag'] = 'candidate'
	candidate['identity']['scientific_identity'].update(
		{
			'experiment_role': (
				'multi_head_ordered_xy_neighbor_unanimous_hard_pretext'
			),
			'variant': 'xyunanim1_nocons',
			'target_representation': 'xy_neighbor_unanimous_hard_labels_v1',
			'target_semantics': 'xy_neighbor_unanimous_outlier_correction_v1',
			'supervised_loss': 'structured_hmm_hard_categorical_v1',
			'consistency_policy': 'disabled_for_xy_neighbor_unanimous_v1',
			'consistency_weight': 0.0,
		}
	)
	candidate['pseudo_targets'] = {
		'manifest': str(tmp_path / 'candidate-target.json'),
		'target_representation': 'xy_neighbor_unanimous_hard_labels_v1',
	}
	runtime = {
		'initial_student_state_sha256': 'b' * 64,
		'initial_head_state_sha256': 'c' * 64,
		'trainability_summary': {},
		'optimizer_group_identity': {},
	}
	monkeypatch.setattr(validation, '_runtime_contract', lambda _value: runtime)

	assert validation._hard_config_parity(candidate, hard)['status'] == 'PASS'  # noqa: SLF001


def test_hard_baseline_resolution_derives_only_its_frozen_manifest_digest(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	manifest = (
		artifact_root
		/ 'pseudo_targets/f3/facies_benchmark_v1'
		/ 'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1'
		/ 'multi_head_target_manifest.json'
	)
	manifest.parent.mkdir(parents=True)
	manifest.write_bytes(b'frozen-hard-target')
	variable = 'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256'
	monkeypatch.delenv(variable, raising=False)
	calls = 0

	def load(_path: Path) -> dict[str, object]:
		nonlocal calls
		calls += 1
		if calls == 1:
			raise ValueError(f'config environment variable is required: {variable}')
		assert os.environ[variable] == sha256(manifest.read_bytes()).hexdigest()
		return {'resolved': True}

	monkeypatch.setattr(validation, 'load_config', load)
	monkeypatch.setattr(
		validation, 'resolve_strat_hmm_pretext_config', lambda value: value
	)

	assert validation._training_config(  # noqa: SLF001
		tmp_path / 'hard.yaml', artifact_root=artifact_root
	) == {'resolved': True}
	assert variable not in os.environ


def test_embedding_identity_binds_all_hard_route_weights() -> None:
	identity = _checkpoint_identity()
	training = _training_identity_config()
	metadata = {'stratigraphy_pretext': _embedding_identity(identity)}

	validation._validate_embedding_identity(  # noqa: SLF001
		metadata,
		identity,
		training=training,
	)
	metadata['stratigraphy_pretext']['prototype_weight'] = 0.5  # type: ignore[index]
	with pytest.raises(ValueError, match='prototype_weight'):
		validation._validate_embedding_identity(  # noqa: SLF001
			metadata,
			identity,
			training=training,
		)


def test_review_publishes_portable_evidence_and_manifest(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	workspace = tmp_path / 'workspace'
	artifact_root = workspace / 'artifacts' / 'seis_ssl_cluster'
	artifact_root.mkdir(parents=True)
	target_path = artifact_root / 'target.json'
	audit_path = artifact_root / 'audit.json'
	handoff_path = artifact_root / 'handoff.json'
	for path in (target_path, audit_path, handoff_path):
		path.write_text('{}\n', encoding='utf-8')
	config = results.f3_xy_neighbor_unanimous_review_config_from_mapping(
		{
			'artifact_root': str(artifact_root),
			'workspace_root': str(workspace),
			'target_manifest': str(target_path),
			'target_audit': str(audit_path),
			'pretraining_handoff': str(handoff_path),
			'output_dir': str(workspace / 'results/f3/unanimous'),
		}
	)
	target = _target_manifest(artifact_root / 'source.json')
	handoff = {
		'targets': {
			'temporal_transition_counts': {
				str(k): {'source': k, 'output': k + 1} for k in (6, 8, 10)
			}
		}
	}
	monkeypatch.setattr(
		results,
		'load_multi_head_xy_neighbor_unanimous_target_manifest',
		lambda *_args, **_kwargs: target,
	)
	monkeypatch.setattr(
		results,
		'load_f3_xy_neighbor_unanimous_target_audit',
		lambda *_args: {'status': 'XYUNANIM_TARGET_GO'},
	)
	monkeypatch.setattr(
		results,
		'replay_f3_xy_neighbor_unanimous_target_audit',
		lambda *_args, **_kwargs: {'status': 'XYUNANIM_TARGET_GO'},
	)
	monkeypatch.setattr(
		results,
		'load_f3_xy_neighbor_unanimous_pretraining_handoff',
		lambda *_args: handoff,
	)
	monkeypatch.setattr(results, '_validate_lineage', lambda *_args, **_kwargs: None)

	results.publish_f3_xy_neighbor_unanimous_review(config)

	assert {path.name for path in config.output_dir.iterdir()} == {
		results.SUMMARY_JSON,
		results.SUMMARY_MARKDOWN,
	}
	assert not (config.output_dir / 'publish_manifest.json').exists()
	for path in config.output_dir.iterdir():
		if path.suffix in {'.json', '.md'}:
			text = path.read_text(encoding='utf-8')
			assert str(artifact_root) not in text
			assert str(workspace) not in text
def _validation_config(
	tmp_path: Path,
) -> validation.F3XYNeighborUnanimousPretrainingValidationConfig:
	artifact_root = tmp_path / 'artifacts'
	experiment_root = artifact_root / 'pretraining'
	experiment_root.mkdir(parents=True)
	paths = [
		artifact_root / name
		for name in (
			'target.json',
			'audit.json',
			'hard.yaml',
			'smoke.yaml',
			'full.yaml',
		)
	]
	for path in paths:
		path.write_text('{}\n', encoding='utf-8')
	return validation.F3XYNeighborUnanimousPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=experiment_root,
		target_manifest=paths[0],
		target_audit=paths[1],
		hard_full_config=paths[2],
		xy_neighbor_unanimous_smoke_config=paths[3],
		xy_neighbor_unanimous_full_config=paths[4],
	)


def _checkpoint_identity() -> dict[str, object]:
	return {
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'model_tag': 'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1',
		'target_representation': 'xy_neighbor_unanimous_hard_labels_v1',
		'target_semantics': 'xy_neighbor_unanimous_outlier_correction_v1',
		'xy_neighbor_unanimous_target_manifest_sha256': 'a' * 64,
		'xy_neighbor_unanimous_target_manifest': {
			'path': 'target.json',
			'sha256': 'a' * 64,
		},
		'per_head_xy_neighbor_unanimous_targets': {'6': {}, '8': {}, '10': {}},
		'source_hard_manifest_sha256': 'b' * 64,
		'xy_neighbor_unanimous_smoothing': {'application': 'single_pass'},
		'consistency_policy': 'disabled_for_xy_neighbor_unanimous_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'scientific_identity_sha256': 'c' * 64,
		'stratigraphy_state_sha256': 'd' * 64,
	}


def _training_identity_config() -> dict[str, object]:
	return {
		'head': {'spec': 'multi_resolution_ordered_prototypes_v1'},
		'student': {'unfreeze_top_blocks': 1},
		'loss': {
			'distillation_weight': 0.2,
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
		},
	}


def _embedding_identity(identity: dict[str, object]) -> dict[str, object]:
	return {
		'method': 'strat_hmm_multi_head_pretext',
		'base_objective': 'amp_mae3d',
		'head_spec': identity['head_spec'],
		'head_ks': identity['head_ks'],
		'head_count': 3,
		'unfreeze_top_blocks': 1,
		'distillation_weight': 0.2,
		'prototype_weight': 1.0,
		'prototype_weight_semantics': 'mean_across_heads',
		'usage_weight': 0.005,
		'usage_weight_semantics': 'mean_across_heads',
		'consistency_policy': identity['consistency_policy'],
		'consistency_weight': identity['consistency_weight'],
		'consistency_beta': identity['consistency_beta'],
		'model_tag': identity['model_tag'],
		'scientific_identity_sha256': identity['scientific_identity_sha256'],
		'checkpoint_stratigraphy_state_sha256': identity['stratigraphy_state_sha256'],
		'target_representation': identity['target_representation'],
		'target_semantics': identity['target_semantics'],
		'xy_neighbor_unanimous_target_manifest_path': 'target.json',
		'xy_neighbor_unanimous_target_manifest_sha256': identity[
			'xy_neighbor_unanimous_target_manifest_sha256'
		],
		'per_head_xy_neighbor_unanimous_target_sha256': identity[
			'per_head_xy_neighbor_unanimous_targets'
		],
		'source_hard_manifest_sha256': identity['source_hard_manifest_sha256'],
		'xy_neighbor_unanimous_smoothing': identity['xy_neighbor_unanimous_smoothing'],
	}


def _target_manifest(source: Path) -> dict[str, object]:
	return {
		'target_representation': 'xy_neighbor_unanimous_hard_labels_v1',
		'target_semantics': 'xy_neighbor_unanimous_outlier_correction_v1',
		'source_hard_manifest': {
			'path': str(source),
			'sha256': sha256(b'source').hexdigest(),
		},
		'heads': {
			str(k): {
				'diagnostics': {
					'aggregate': {
						'valid_token_count': 10 * k,
						'changed_token_count': 2,
						'changed_fraction': 2 / (10 * k),
					}
				}
			}
			for k in (6, 8, 10)
		},
	}
