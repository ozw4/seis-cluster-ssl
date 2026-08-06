"""Focused contracts for the strict F3 M5-LS pretraining validator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

import seis_ssl_cluster.f3.lateral_smoothing_pretraining_validation as validation
from seis_ssl_cluster.f3.lateral_smoothing_pretraining_validation import (
	F3M5LateralSmoothingPretrainingValidationConfig,
	f3_m5_lateral_smoothing_pretraining_validation_config_from_mapping,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import scientific_identity_sha256

_HARD_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
_LATERAL_TAG = 'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1'
_DIGEST = 'a' * 64


def test_validator_config_is_closed_and_requires_all_inputs(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	for name in ('calibration.json', 'selected.json', 'hard_handoff.json'):
		(artifact_root / name).write_text('{}', encoding='utf-8')
	configs = {}
	for name in ('hard', 'smoke', 'lateral'):
		path = tmp_path / f'{name}.yaml'
		path.write_text('{}', encoding='utf-8')
		configs[name] = path
	base = {
		'artifact_root': str(artifact_root),
		'experiment_root': str(artifact_root / 'pretraining'),
		'calibration_handoff': str(artifact_root / 'calibration.json'),
		'selected_manifest': str(artifact_root / 'selected.json'),
		'hard_full_config': str(configs['hard']),
		'hard_handoff': str(artifact_root / 'hard_handoff.json'),
		'lateral_smoke_config': str(configs['smoke']),
		'lateral_full_config': str(configs['lateral']),
	}

	resolved = f3_m5_lateral_smoothing_pretraining_validation_config_from_mapping(
		base
	)

	assert resolved.selected_manifest == artifact_root / 'selected.json'
	with pytest.raises(ValueError, match='unknown M5-LS validation config keys'):
		f3_m5_lateral_smoothing_pretraining_validation_config_from_mapping(
			{**base, 'lithology_labels': 'forbidden.npy'}
		)
	with pytest.raises(ValueError, match='lateral_smoke_config'):
		f3_m5_lateral_smoothing_pretraining_validation_config_from_mapping(
			{key: value for key, value in base.items() if key != 'lateral_smoke_config'}
		)


def test_selected_calibration_requires_selected_status_and_byte_exact_copy(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	candidate = artifact_root / 'candidate.json'
	selected_path = artifact_root / 'selected.json'
	calibration_path = artifact_root / 'calibration.json'
	for path in (candidate, selected_path):
		path.write_bytes(b'{"immutable":true}\n')
	calibration_path.write_text('{}', encoding='utf-8')
	config = _validator_config(
		artifact_root,
		calibration_handoff=calibration_path,
		selected_manifest=selected_path,
	)
	head_hashes = _head_hashes()
	monkeypatch.setattr(validation, '_multi_head_target_hashes', lambda _: head_hashes)
	calibration = _calibration_fixture(
		candidate=candidate,
		selected=selected_path,
		head_hashes=head_hashes,
	)

	evidence = validation._validate_calibration_selection(  # noqa: SLF001
		config, calibration, {}
	)

	assert evidence['selected_beta'] == 0.10
	assert evidence['selected_manifest']['sha256'] == validation.file_sha256(
		selected_path
	)
	calibration['status'] = 'M5_LS_TARGET_HOLD'
	with pytest.raises(ValueError, match='did not select a target'):
		validation._validate_calibration_selection(config, calibration, {})  # noqa: SLF001
	calibration['status'] = 'M5_LS_TARGET_SELECTED'
	selected_path.write_bytes(b'{"different":true}\n')
	with pytest.raises(ValueError, match='selected manifest SHA-256 mismatch'):
		validation._validate_calibration_selection(config, calibration, {})  # noqa: SLF001


def test_hard_lateral_config_delta_rejects_every_non_target_change() -> None:
	hard, lateral = _paired_training_configs()
	validation._validate_allowed_config_delta(hard, lateral)  # noqa: SLF001
	mutations = (
		('model', 'encoder_dim', 385),
		('head', 'projection_dim', 129),
		('loss', 'prototype_weight', 0.9),
		('data', 'min_valid_fraction', 0.2),
		('train', 'seed', 43),
		('teacher', 'checkpoint', 'other_teacher.pt'),
		('student', 'init_checkpoint', 'other_student.pt'),
		('student', 'unfreeze_top_blocks', 2),
	)
	for section, key, value in mutations:
		altered = deepcopy(lateral)
		altered[section][key] = value
		with pytest.raises(ValueError, match='scientific config drift'):
			validation._validate_allowed_config_delta(hard, altered)  # noqa: SLF001


def test_target_contract_requires_initial_trainability_and_optimizer_parity(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	hard_manifest = artifact_root / 'hard_manifest.json'
	selected_manifest = artifact_root / 'selected_manifest.json'
	posterior_manifest = artifact_root / 'posterior_manifest.json'
	for path in (hard_manifest, selected_manifest, posterior_manifest):
		path.write_text('{}', encoding='utf-8')
	config = _validator_config(
		artifact_root,
		selected_manifest=selected_manifest,
	)
	head_hashes = _head_hashes()
	smoothing = {'pairwise_strength_ratio': 0.10, 'resolved_scales': {}}
	source_hard = {
		'path': str(hard_manifest),
		'sha256': validation.file_sha256(hard_manifest),
	}
	source_posterior = {
		'path': str(posterior_manifest),
		'sha256': validation.file_sha256(posterior_manifest),
	}
	hard = {
		'paths': {'output_root': str(artifact_root / 'pretraining' / _HARD_TAG)},
		'identity': {'model_tag': _HARD_TAG, 'scientific_identity': {}},
		'pseudo_targets': {'manifest': str(hard_manifest)},
	}
	lateral = {
		'paths': {
			'output_root': str(artifact_root / 'pretraining' / _LATERAL_TAG)
		},
		'identity': {
			'model_tag': _LATERAL_TAG,
			'scientific_identity': {
				'target_representation': 'lateral_mean_field_hard_labels_v1',
				'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
				'supervised_loss': 'structured_hmm_hard_categorical_v1',
				'consistency_policy': 'disabled_for_m5_ls_v1',
				'consistency_weight': 0.0,
				'lateral_target_manifest_sha256': validation.file_sha256(
					selected_manifest
				),
				'lateral_target_head_hashes': head_hashes,
				'lateral_smoothing': smoothing,
				'source_hard_manifest_sha256': source_hard['sha256'],
				'source_posterior_manifest_sha256': source_posterior['sha256'],
			},
		},
		'pseudo_targets': {'manifest': str(selected_manifest)},
		'loss': {'consistency_weight': 0.0},
	}
	selected = {
		'head_ks': [6, 8, 10],
		'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
		'smoothing': {'pairwise_strength_ratio': 0.10},
		'source_hard_manifest': source_hard,
		'source_posterior_manifest': source_posterior,
	}
	hard_runtime = _runtime_contract_fixture()
	lateral_runtime = deepcopy(hard_runtime)
	lateral_runtime['initial_head_state_sha256'] = 'd' * 64
	monkeypatch.setattr(
		validation,
		'_validate_calibration_selection',
		lambda *_: {
			'selected_beta': 0.10,
			'calibration_handoff': {'path': 'calibration.json', 'sha256': _DIGEST},
			'selected_manifest': {
				'path': str(selected_manifest),
				'sha256': validation.file_sha256(selected_manifest),
			},
			'source_hard_manifest': source_hard,
			'source_posterior_manifest': source_posterior,
		},
	)
	monkeypatch.setattr(validation, '_validate_allowed_config_delta', lambda *_: None)
	monkeypatch.setattr(validation, 'load_multi_head_target_manifest', lambda _: {})
	monkeypatch.setattr(validation, '_validate_canonical_valid_masks', lambda *_: None)
	monkeypatch.setattr(validation, '_multi_head_target_hashes', lambda _: head_hashes)
	monkeypatch.setattr(validation, '_lateral_smoothing_identity', lambda _: smoothing)
	monkeypatch.setattr(
		validation.hard_validation,
		'load_f3_multi_head_pretraining_handoff',
		lambda _: {
			'model_tag': _HARD_TAG,
			'stratigraphy_pretext': {
				'initial_student_state_sha256': 'b' * 64,
				'initial_head_state_sha256': 'c' * 64,
			},
		},
	)
	monkeypatch.setattr(
		validation,
		'_hard_baseline_checkpoint_evidence',
		lambda *_: {
			'hard_baseline_checkpoint': 'hard_best.pt',
			'hard_baseline_checkpoint_sha256': _DIGEST,
			'hard_baseline_trainability_summary': hard_runtime[
				'trainability_summary'
			],
			'hard_baseline_optimizer_group_identity': hard_runtime[
				'optimizer_group_identity'
			],
			'hard_checkpoint_identity': {
				'initial_student_state_sha256': 'b' * 64,
				'initial_head_state_sha256': 'c' * 64,
			},
		},
	)
	monkeypatch.setattr(
		validation,
		'_runtime_contract',
		lambda training: hard_runtime if training is hard else lateral_runtime,
	)

	with pytest.raises(ValueError, match='initial student/head hashes differ'):
		validation._validate_target_contract(  # noqa: SLF001
			config, {}, selected, hard, lateral
		)
	lateral_runtime['initial_head_state_sha256'] = 'c' * 64
	lateral_runtime['trainability_summary'] = {
		**hard_runtime['trainability_summary'],
		'trainable_parameter_count': 2,
	}
	with pytest.raises(ValueError, match='trainability differs'):
		validation._validate_target_contract(  # noqa: SLF001
			config, {}, selected, hard, lateral
		)
	lateral_runtime['trainability_summary'] = hard_runtime['trainability_summary']
	lateral_runtime['optimizer_group_identity'] = [
		{'name': 'encoder', 'parameter_names': ['student.fixture'], 'lr': 1.0e-5}
	]
	with pytest.raises(ValueError, match='optimizer groups differ'):
		validation._validate_target_contract(  # noqa: SLF001
			config, {}, selected, hard, lateral
		)


def test_smoke_config_isolated_cpu_two_step_and_exactly_paired(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	_, full = _paired_training_configs(artifact_root=artifact_root)
	smoke = deepcopy(full)
	smoke['paths']['output_root'] = str(artifact_root / 'lateral_smoke')
	smoke['identity']['runtime_identity']['device'] = 'cpu'
	smoke['train'].update(device='cpu', max_steps=2)
	smoke['identity']['scientific_identity']['train']['max_steps'] = 2

	validation._validate_smoke_config(full=full, smoke=smoke)  # noqa: SLF001
	altered = deepcopy(smoke)
	altered['train']['seed'] = 99
	with pytest.raises(ValueError, match='config drift'):
		validation._validate_smoke_config(full=full, smoke=altered)  # noqa: SLF001
	Path(full['paths']['output_root']).mkdir()
	with pytest.raises(ValueError, match='must remain unmodified'):
		validation._validate_smoke_config(full=full, smoke=smoke)  # noqa: SLF001


def test_checkpoint_payload_requires_schema_v4_hard_loss_and_zero_consistency(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	training = _checkpoint_training()
	summary = _summary()
	groups = _optimizer_groups()
	payload = _checkpoint_payload(training, summary=summary, groups=groups)
	monkeypatch.setattr(
		validation,
		'validate_stratigraphy_checkpoint_payload',
		lambda *_args, **_kwargs: None,
	)

	validation._validate_lateral_checkpoint_payload(  # noqa: SLF001
		payload,
		training=training,
		hard_trainability_summary=summary,
		hard_optimizer_group_identity=groups,
	)
	payload['stratigraphy_checkpoint']['target_representation'] = (
		'ordered_path_state_posterior_v1'
	)
	with pytest.raises(ValueError, match='target representation'):
		validation._validate_lateral_checkpoint_payload(  # noqa: SLF001
			payload,
			training=training,
			hard_trainability_summary=summary,
			hard_optimizer_group_identity=groups,
		)
	payload['stratigraphy_checkpoint']['target_representation'] = (
		'lateral_mean_field_hard_labels_v1'
	)
	payload['metrics'].pop('loss_consistency')
	with pytest.raises(ValueError, match='hard multi-head loss path'):
		validation._validate_lateral_checkpoint_payload(  # noqa: SLF001
			payload,
			training=training,
			hard_trainability_summary=summary,
			hard_optimizer_group_identity=groups,
		)
	payload['metrics']['loss_consistency'] = 0.25
	payload['metrics']['loss'] = float('nan')
	with pytest.raises(ValueError, match='metrics must all be finite'):
		validation._validate_lateral_checkpoint_payload(  # noqa: SLF001
			payload,
			training=training,
			hard_trainability_summary=summary,
			hard_optimizer_group_identity=groups,
		)
	payload['metrics']['loss'] = 1.0
	payload['stratigraphy_checkpoint']['consistency_weight'] = 0.1
	with pytest.raises(ValueError, match='consistency weight must be zero'):
		validation._validate_lateral_checkpoint_payload(  # noqa: SLF001
			payload,
			training=training,
			hard_trainability_summary=summary,
			hard_optimizer_group_identity=groups,
		)


def test_smoke_evidence_requires_two_steps_and_never_publishes_final_handoff(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	_, full = _paired_training_configs(artifact_root=artifact_root)
	smoke = deepcopy(full)
	smoke['paths']['output_root'] = str(artifact_root / 'lateral_smoke')
	smoke['identity']['runtime_identity']['device'] = 'cpu'
	smoke['train'].update(device='cpu', max_steps=2)
	smoke['identity']['scientific_identity']['train']['max_steps'] = 2
	summary = _summary()
	groups = _optimizer_groups()
	checkpoint = {
		'latest': {
			'epoch': 1,
			'global_step': 2,
			'training_state': {'checkpoint_kind': 'step'},
		},
		'identity': {
			'schema_version': 4,
			'target_representation': 'lateral_mean_field_hard_labels_v1',
			'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
			'consistency_weight': 0.0,
		},
	}
	calls: list[int] = []
	monkeypatch.setattr(validation, '_initial_hashes', lambda _: ('b' * 64, 'c' * 64))
	monkeypatch.setattr(
		validation,
		'_checkpoint_evidence',
		lambda *_args, **kwargs: calls.append(
			kwargs['expected_global_step']
		)
		or checkpoint,
	)

	evidence = validation._smoke_evidence(  # noqa: SLF001
		full=full,
		smoke=smoke,
		target_evidence={
			'initial_student_state_sha256': 'b' * 64,
			'initial_head_state_sha256': 'c' * 64,
			'hard_baseline_trainability_summary': summary,
			'hard_baseline_optimizer_group_identity': groups,
		},
	)

	assert calls == [2]
	assert evidence['hard_multi_head_loss_path_used'] is True
	assert evidence['posterior_loss_path_used'] is False
	assert evidence['consistency_contribution'] == 0.0
	with pytest.raises(ValueError, match='finish at global step 2'):
		validation._validate_checkpoint_progress(  # noqa: SLF001
			{'global_step': 1},
			root=artifact_root,
			expected_global_step=2,
			require_full_epoch_history=False,
		)


def test_targets_and_smoke_phases_do_not_publish_final_pass_handoff(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	config = _validator_config(artifact_root)
	target_evidence = {
		'initial_student_state_sha256': 'b' * 64,
		'initial_head_state_sha256': 'c' * 64,
		'hard_baseline_trainability_summary': _summary(),
		'hard_baseline_optimizer_group_identity': _optimizer_groups(),
	}
	monkeypatch.setattr(validation, '_load_calibration_handoff', lambda _: {})
	monkeypatch.setattr(
		validation, 'load_multi_head_lateral_target_manifest', lambda _: {}
	)
	monkeypatch.setattr(
		validation,
		'_training_config',
		lambda _: _checkpoint_training(),
	)
	monkeypatch.setattr(
		validation, '_validate_target_contract', lambda *_: target_evidence
	)
	monkeypatch.setattr(validation, '_smoke_evidence', lambda *_args, **_kwargs: {})
	monkeypatch.setattr(validation, '_publish_handoff', _unexpected_publication)

	targets = validation.validate_f3_m5_lateral_smoothing_pretraining(
		config, phase='targets'
	)
	smoke = validation.validate_f3_m5_lateral_smoothing_pretraining(
		config, phase='smoke'
	)

	assert targets.published_handoff is None
	assert smoke.published_handoff is None


def _validator_config(
	artifact_root: Path,
	*,
	calibration_handoff: Path | None = None,
	selected_manifest: Path | None = None,
) -> F3M5LateralSmoothingPretrainingValidationConfig:
	return F3M5LateralSmoothingPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root / 'pretraining',
		calibration_handoff=calibration_handoff or artifact_root / 'calibration.json',
		selected_manifest=selected_manifest or artifact_root / 'selected.json',
		hard_full_config=artifact_root / 'hard.yaml',
		hard_handoff=artifact_root / 'hard_handoff.json',
		lateral_smoke_config=artifact_root / 'smoke.yaml',
		lateral_full_config=artifact_root / 'lateral.yaml',
	)


def _paired_training_configs(
	*, artifact_root: Path | None = None
) -> tuple[dict[str, object], dict[str, object]]:
	root = artifact_root or Path('/artifacts')
	base = {
		'paths': {'artifact_root': str(root), 'output_root': str(root / 'hard')},
		'identity': {
			'model_tag': _HARD_TAG,
			'runtime_identity': {'device': 'auto', 'workers': 4},
			'scientific_identity': {
				'experiment_role': 'multi_head_ordered_pretext',
				'variant': 'nocons',
				'target_representation': 'hard_viterbi_labels_v1',
				'target_manifest_sha256': _DIGEST,
				'target_head_hashes': _head_hashes(),
				'supervised_loss': 'structured_hmm_hard_categorical_v1',
				'consistency_policy': 'normalized_order_smooth_l1_v1',
				'consistency_weight': 0.0,
				'train': {'max_steps': None},
			},
		},
		'pseudo_targets': {
			'manifest': str(root / 'hard_manifest.json'),
			'target_representation': 'hard_viterbi_labels_v1',
			'min_confidence': 0.0,
		},
		'model': {'encoder_dim': 384},
		'head': {'projection_dim': 128},
		'loss': {'prototype_weight': 1.0, 'consistency_weight': 0.0},
		'data': {'min_valid_fraction': 0.1},
		'teacher': {'checkpoint': 'teacher.pt'},
		'student': {'init_checkpoint': 'student.pt', 'unfreeze_top_blocks': 1},
		'train': {'seed': 42, 'device': 'auto', 'max_steps': None},
	}
	lateral = deepcopy(base)
	lateral['paths']['output_root'] = str(root / 'lateral')
	lateral['identity']['model_tag'] = _LATERAL_TAG
	scientific = lateral['identity']['scientific_identity']
	for key in ('target_manifest_sha256', 'target_head_hashes'):
		scientific.pop(key)
	scientific.update(
		{
			'experiment_role': 'multi_head_ordered_lateral_hard_pretext',
			'variant': 'latmf1_nocons',
			'target_representation': 'lateral_mean_field_hard_labels_v1',
			'lateral_target_manifest_sha256': _DIGEST,
			'lateral_target_head_hashes': _head_hashes(),
			'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
			'source_hard_manifest_sha256': _DIGEST,
			'source_posterior_manifest_sha256': _DIGEST,
			'lateral_smoothing': {'pairwise_strength_ratio': 0.1},
			'consistency_policy': 'disabled_for_m5_ls_v1',
		}
	)
	lateral['pseudo_targets'].update(
		manifest=str(root / 'selected.json'),
		target_representation='lateral_mean_field_hard_labels_v1',
	)
	return base, lateral


def _head_hashes() -> dict[str, dict[str, dict[str, str]]]:
	return {
		str(k): {
			'f3': dict.fromkeys(
				('labels', 'confidence', 'valid_tokens', 'metadata'), _DIGEST
			)
		}
		for k in (6, 8, 10)
	}


def _calibration_fixture(
	*, candidate: Path, selected: Path, head_hashes: dict[str, object]
) -> dict[str, object]:
	def reference(path: Path) -> dict[str, str]:
		return {'path': str(path), 'sha256': validation.file_sha256(path)}

	return {
		'artifact_type': 'f3_m5_lateral_target_calibration',
		'schema_version': 1,
		'status': 'M5_LS_TARGET_SELECTED',
		'selection_policy': 'target_only_smallest_eligible_beta_v1',
		'candidate_betas': [0.10, 0.25, 0.50],
		'beta_zero_parity': {
			'status': 'PASS',
			'heads': {str(k): {} for k in (6, 8, 10)},
		},
		'source_hard_manifest': {'path': '/hard.json', 'sha256': _DIGEST},
		'source_posterior_manifest': {'path': '/posterior.json', 'sha256': _DIGEST},
		'candidates': {
			'beta010': {
				'beta': 0.10,
				'manifest': reference(candidate),
				'head_hashes': head_hashes,
				'eligibility': {'eligible': True, 'checks': {}, 'reasons': []},
			},
			'beta025': {'beta': 0.25},
			'beta050': {'beta': 0.50},
		},
		'selected_beta': 0.10,
		'selected_candidate_manifest': reference(candidate),
		'selected_manifest': reference(selected),
	}


def _summary() -> dict[str, object]:
	return {
		'trainable_parameter_count': 1,
		'frozen_parameter_count': 2,
		'trainable_names': ['encoder.layers.7.weight'],
	}


def _runtime_contract_fixture() -> dict[str, object]:
	return {
		'initial_student_state_sha256': 'b' * 64,
		'initial_head_state_sha256': 'c' * 64,
		'trainability_summary': _summary(),
		'optimizer_group_identity': _optimizer_groups(),
	}


def _optimizer_groups() -> list[dict[str, object]]:
	return [{'name': 'head', 'parameter_names': ['head.fixture'], 'lr': 3.0e-4}]


def _checkpoint_training() -> dict[str, object]:
	scientific = {'target_representation': 'lateral_mean_field_hard_labels_v1'}
	return {
		'paths': {'output_root': str(Path.cwd() / 'lateral')},
		'identity': {'model_tag': _LATERAL_TAG, 'scientific_identity': scientific},
	}


def _checkpoint_payload(
	training: dict[str, object],
	*,
	summary: dict[str, object],
	groups: list[dict[str, object]],
) -> dict[str, object]:
	return {
		'stratigraphy_config': training,
		'metrics': {'loss': 1.0, 'loss_consistency': 0.25},
		'model_state_dict': {'weight': torch.ones(1)},
		'stratigraphy_state_dict': {'weight': torch.ones(1)},
		'optimizer_state_dict': {'state': {0: {'exp_avg': torch.ones(1)}}},
		'trainability_summary': summary,
		'stratigraphy_checkpoint': {
			'schema_version': 4,
			'model_tag': _LATERAL_TAG,
			'target_representation': 'lateral_mean_field_hard_labels_v1',
			'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
			'consistency_weight': 0.0,
			'scientific_identity_sha256': scientific_identity_sha256(
				training['identity']['scientific_identity']
			),
			'optimizer_group_identity': groups,
		},
	}


def _unexpected_publication(*_args: object, **_kwargs: object) -> bool:
	raise AssertionError('partial target/smoke validation published a final handoff')
