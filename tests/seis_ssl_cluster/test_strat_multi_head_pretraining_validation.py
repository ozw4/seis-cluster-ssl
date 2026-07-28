from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import seis_ssl_cluster.f3.multi_head_pretraining_validation as pretraining_validation
import seis_ssl_cluster.f3.soft_posterior_pretraining_validation as soft_validation
from seis_ssl_cluster.f3.multi_head_pretraining_validation import (
	F3MultiHeadPretrainingValidationConfig,
	_identity_contract,
	_manifest_per_head_target_hashes,
	_publish_handoff,
	_state_sha256,
	_tensor_sha256,
	_validate_best_selection,
	_validate_control_config,
	_validate_freeze_contract,
	_validate_initial_state_hashes,
	_validate_pair,
	_validate_pair_config_contract,
	f3_multi_head_pretraining_validation_config_from_mapping,
	load_f3_multi_head_pretraining_handoff,
	validate_f3_multi_head_pretraining,
)
from seis_ssl_cluster.f3.soft_posterior_pretraining_validation import (
	F3M5SoftPosteriorPretrainingValidationConfig,
	_canonical_valid_token_identities,
	_validate_allowed_config_delta,
	_validate_target_contract,
)
from seis_ssl_cluster.f3.soft_posterior_pretraining_validation import (
	_handoff as _soft_handoff,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import scientific_identity_sha256


def test_validation_config_rejects_unknown_keys_and_paths_outside_artifacts(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	target = artifact_root / 'targets.json'
	target.write_text('{}', encoding='utf-8')
	configs = []
	for name in ('control.yaml', 'nocons.yaml', 'cons010.yaml'):
		path = tmp_path / name
		path.write_text('{}', encoding='utf-8')
		configs.append(path)
	base = {
		'artifact_root': str(artifact_root),
		'experiment_root': str(artifact_root / 'pretraining'),
		'target_manifest': str(target),
		'control_full_config': str(configs[0]),
		'nocons_full_config': str(configs[1]),
		'cons010_full_config': str(configs[2]),
	}
	resolved = f3_multi_head_pretraining_validation_config_from_mapping(base)
	assert resolved.target_manifest == target
	with pytest.raises(ValueError, match='unknown validation config keys'):
		f3_multi_head_pretraining_validation_config_from_mapping(
			{**base, 'unknown': 'value'}
		)


def test_soft_validation_config_requires_a_smoke_config(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	posterior_manifest = artifact_root / 'posterior_manifest.json'
	hard_handoff = artifact_root / 'hard_handoff.json'
	for path in (posterior_manifest, hard_handoff):
		path.write_text('{}', encoding='utf-8')
	config_paths = {
		name: tmp_path / f'{name}.yaml'
		for name in ('hard', 'smoke', 'soft')
	}
	for path in config_paths.values():
		path.write_text('{}', encoding='utf-8')
	base = {
		'artifact_root': str(artifact_root),
		'experiment_root': str(artifact_root / 'pretraining'),
		'posterior_manifest': str(posterior_manifest),
		'hard_full_config': str(config_paths['hard']),
		'hard_handoff': str(hard_handoff),
		'soft_smoke_config': str(config_paths['smoke']),
		'soft_full_config': str(config_paths['soft']),
	}

	resolved = (
		soft_validation.f3_m5_soft_posterior_pretraining_validation_config_from_mapping(
			base
		)
	)

	assert resolved.soft_smoke_config == config_paths['smoke']
	with pytest.raises(ValueError, match='soft_smoke_config'):
		soft_validation.f3_m5_soft_posterior_pretraining_validation_config_from_mapping(
			{key: value for key, value in base.items() if key != 'soft_smoke_config'}
		)


def test_soft_target_evidence_derives_per_head_hashes_from_manifest(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	experiment_root = artifact_root / 'pretraining'
	posterior_manifest = artifact_root / 'posterior_manifest.json'
	posterior_manifest.parent.mkdir(parents=True)
	posterior_manifest.write_text('{}', encoding='utf-8')
	hard_manifest = artifact_root / 'hard_target_manifest.json'
	hard_manifest.write_text('{}', encoding='utf-8')
	soft_model_tag = 'strat_hmm_pretext_mh_k6810_soft_nocons_topblock1_distill_v1'
	hard_model_tag = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
	config = F3M5SoftPosteriorPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=experiment_root,
		posterior_manifest=posterior_manifest,
		hard_full_config=tmp_path / 'hard.yaml',
		hard_handoff=artifact_root / 'hard_handoff.json',
		soft_smoke_config=tmp_path / 'smoke.yaml',
		soft_full_config=tmp_path / 'soft.yaml',
	)
	posterior_hashes = {
		str(head_k): {
			'f3': {
				name: f'{head_k + index:064x}'
				for index, name in enumerate(('posterior', 'valid_tokens', 'metadata'))
			}
		}
		for head_k in (6, 8, 10)
	}
	posterior = {
		'head_ks': [6, 8, 10],
		'posterior_semantics': 'ordered_path_cost_gibbs_state_marginal_v1',
		'cost_temperature': 1.0,
		'heads': {
			key: {
				'surveys': {
					survey_id: {
						name: {'sha256': digest}
						for name, digest in artifacts.items()
					}
					for survey_id, artifacts in surveys.items()
				},
				'diagnostics': {},
			}
			for key, surveys in posterior_hashes.items()
		},
		'source_hard_manifest': {
			'path': str(hard_manifest),
			'sha256': soft_validation.file_sha256(hard_manifest),
		},
		'source_embedding': {'sha256': 'b' * 64},
	}
	soft = {
		'paths': {
			'output_root': str(experiment_root / soft_model_tag),
		},
		'identity': {
			'model_tag': soft_model_tag,
			'scientific_identity': {
				'target_representation': 'ordered_path_state_posterior_v1',
				'posterior_manifest_sha256': soft_validation.file_sha256(
					posterior_manifest
				),
			},
		},
		'pseudo_targets': {'manifest': str(posterior_manifest)},
	}
	hard = {
		'paths': {
			'output_root': str(experiment_root / hard_model_tag),
		},
		'identity': {'model_tag': hard_model_tag},
		'pseudo_targets': {'manifest': str(hard_manifest)},
	}
	monkeypatch.setattr(
		soft_validation, '_validate_allowed_config_delta', lambda *_: None
	)
	monkeypatch.setattr(
		soft_validation.hard_validation,
		'load_f3_multi_head_pretraining_handoff',
		lambda _: {
			'model_tag': hard_model_tag,
			'stratigraphy_pretext': {
				'initial_student_state_sha256': 'c' * 64,
				'initial_head_state_sha256': 'd' * 64,
			}
		},
	)
	monkeypatch.setattr(
		soft_validation, '_initial_hashes', lambda _: ('c' * 64, 'd' * 64)
	)
	monkeypatch.setattr(
		soft_validation,
		'_hard_baseline_checkpoint_evidence',
		lambda *_: {
			'hard_baseline_checkpoint': 'hard_best.pt',
			'hard_baseline_checkpoint_sha256': 'e' * 64,
			'hard_baseline_trainability_summary': {
				'trainable_names': ['encoder.layers.7.weight']
			},
			'hard_baseline_optimizer_group_identity': [
				{'name': 'head', 'parameter_names': ['head.fixture']}
			],
		},
	)

	evidence = _validate_target_contract(config, posterior, hard, soft)

	assert evidence['posterior_head_hashes'] == posterior_hashes
	other_hard_manifest = artifact_root / 'other_hard_target_manifest.json'
	other_hard_manifest.write_text('{}', encoding='utf-8')
	posterior['source_hard_manifest'] = {
		'path': str(other_hard_manifest),
		'sha256': soft_validation.file_sha256(other_hard_manifest),
	}
	with pytest.raises(
		ValueError, match='does not match hard baseline target manifest'
	):
		_validate_target_contract(config, posterior, hard, soft)


def test_soft_config_delta_rejects_pseudo_target_drift() -> None:
	hard = {
		'paths': {'output_root': 'hard'},
		'identity': {'model_tag': 'hard', 'scientific_identity': {}},
		'pseudo_targets': {'manifest': 'hard_manifest', 'min_confidence': 0.0},
	}
	soft = {
		'paths': {'output_root': 'soft'},
		'identity': {'model_tag': 'soft', 'scientific_identity': {}},
		'pseudo_targets': {
			'manifest': 'soft_manifest',
			'min_confidence': 0.0,
			'target_representation': 'ordered_path_state_posterior_v1',
		},
	}

	_validate_allowed_config_delta(hard, soft)
	soft['pseudo_targets']['min_confidence'] = 0.1
	with pytest.raises(ValueError, match='scientific config drift'):
		_validate_allowed_config_delta(hard, soft)


def test_soft_validator_binds_hard_config_to_handoff_checkpoint(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	hard_checkpoint = tmp_path / 'hard_best.pt'
	hard_checkpoint.write_bytes(b'canonical hard checkpoint')
	hard = {'data': {'min_valid_fraction': 0.1}, 'train': {'seed': 42}}
	identity = {
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'consistency_policy': 'normalized_order_smooth_l1_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'scientific_identity_sha256': 'a' * 64,
		'initial_student_state_sha256': 'b' * 64,
		'initial_head_state_sha256': 'c' * 64,
		'target_manifest': {'sha256': 'd' * 64},
		'per_head_targets': {},
		'optimizer_group_identity': [
			{'name': 'head', 'parameter_names': ['head.fixture']}
		],
	}
	handoff = {
		'checkpoint': {
			'path': str(hard_checkpoint),
			'sha256': soft_validation.file_sha256(hard_checkpoint),
		},
		'stratigraphy_pretext': {
			**{
				key: value
				for key, value in identity.items()
				if key not in {'target_manifest', 'per_head_targets'}
			},
			'target_manifest_sha256': 'd' * 64,
			'per_head_target_sha256': {},
		},
	}
	monkeypatch.setattr(
		soft_validation,
		'_torch_mapping',
		lambda _: {
			'stratigraphy_config': {
				'data': {'min_valid_fraction': 0.2},
				'train': {'seed': 42},
			},
			'stratigraphy_checkpoint': identity,
			'trainability_summary': {'trainable_names': ['encoder.layers.7.weight']},
		},
	)
	monkeypatch.setattr(
		soft_validation,
		'validate_stratigraphy_checkpoint_payload',
		lambda *_args, **_kwargs: None,
	)

	with pytest.raises(ValueError, match='does not match canonical hard handoff'):
		soft_validation._hard_baseline_checkpoint_evidence(  # noqa: SLF001
			hard, handoff
		)


@pytest.mark.parametrize(
	('hard_trainability', 'hard_optimizer_groups', 'match'),
	[
		(
			{'trainable_names': ['encoder.layers.6.weight']},
			[{'name': 'head', 'parameter_names': ['head.fixture']}],
			'trainability differs',
		),
		(
			{'trainable_names': ['encoder.layers.7.weight']},
			[{'name': 'encoder', 'parameter_names': ['student.fixture']}],
			'optimizer groups differ',
		),
	],
)
def test_soft_checkpoint_evidence_requires_hard_trainability_and_optimizer_parity(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	hard_trainability: dict[str, object],
	hard_optimizer_groups: list[dict[str, object]],
	match: str,
) -> None:
	root = tmp_path / 'soft'
	root.mkdir()
	for name in ('latest.pt', 'best.pt'):
		(root / name).write_bytes(name.encode())
	scientific = {'target_representation': 'ordered_path_state_posterior_v1'}
	optimizer_groups = [{'name': 'head', 'parameter_names': ['head.fixture']}]
	payload = {
		'epoch': 25,
		'global_step': 25600,
		'trainability_summary': {'trainable_names': ['encoder.layers.7.weight']},
		'stratigraphy_checkpoint': {
			'schema_version': 3,
			'model_tag': 'strat_hmm_pretext_mh_k6810_soft_nocons_topblock1_distill_v1',
			'scientific_identity_sha256': scientific_identity_sha256(scientific),
			'initial_student_state_sha256': 'a' * 64,
			'initial_head_state_sha256': 'b' * 64,
			'optimizer_group_identity': optimizer_groups,
		},
	}
	training = {
		'paths': {'output_root': str(root)},
		'identity': {'scientific_identity': scientific},
	}
	monkeypatch.setattr(soft_validation, '_torch_mapping', lambda _: payload)
	monkeypatch.setattr(
		soft_validation,
		'validate_stratigraphy_checkpoint_payload',
		lambda *_args, **_kwargs: None,
	)
	monkeypatch.setattr(
		soft_validation.hard_validation, '_metrics_finite', lambda _: None
	)

	with pytest.raises(ValueError, match=match):
		soft_validation._checkpoint_evidence(  # noqa: SLF001
			training,
			hard_trainability_summary=hard_trainability,
			hard_optimizer_group_identity=hard_optimizer_groups,
		)


def test_soft_smoke_checkpoint_evidence_requires_two_finite_steps(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	root = tmp_path / 'soft_smoke'
	root.mkdir()
	for name in ('latest.pt', 'best.pt'):
		(root / name).write_bytes(name.encode())
	scientific = {'target_representation': 'ordered_path_state_posterior_v1'}
	event = {
		'sequence': 0,
		'epoch': 1,
		'global_step': 2,
		'checkpoint_kind': 'step',
		'batch_index': 1,
		'loss': 1.0,
	}
	selection = {
		'schema_version': 1,
		'criterion': 'metrics.loss',
		'improvement_policy': 'strictly_lower_loss_v1',
		'events': [
			{
				**event,
				'previous_best_score': None,
				'best_updated': True,
				'best_score_after': 1.0,
			}
		],
		'selected': event,
	}
	optimizer_groups = [{'name': 'head', 'parameter_names': ['head.fixture']}]
	payload = {
		'epoch': 1,
		'global_step': 2,
		'metrics': {'loss': 1.0, 'loss_prototype': 0.8},
		'training_state': {'checkpoint_kind': 'step', 'batch_index': 1},
		'checkpoint_selection': selection,
		'trainability_summary': {'trainable_names': ['encoder.layers.7.weight']},
		'stratigraphy_checkpoint': {
			'schema_version': 3,
			'model_tag': 'strat_hmm_pretext_mh_k6810_soft_nocons_topblock1_distill_v1',
			'scientific_identity_sha256': scientific_identity_sha256(scientific),
			'initial_student_state_sha256': 'a' * 64,
			'initial_head_state_sha256': 'b' * 64,
			'optimizer_group_identity': optimizer_groups,
		},
	}
	training = {
		'paths': {'output_root': str(root)},
		'identity': {'scientific_identity': scientific},
	}
	monkeypatch.setattr(soft_validation, '_torch_mapping', lambda _: payload)
	monkeypatch.setattr(
		soft_validation,
		'validate_stratigraphy_checkpoint_payload',
		lambda *_args, **_kwargs: None,
	)
	monkeypatch.setattr(
		soft_validation, '_initial_hashes', lambda _: ('a' * 64, 'b' * 64)
	)
	monkeypatch.setattr(
		soft_validation.hard_validation, '_validate_freeze_contract', lambda *_: None
	)

	evidence = soft_validation._checkpoint_evidence(  # noqa: SLF001
		training,
		hard_trainability_summary=payload['trainability_summary'],
		hard_optimizer_group_identity=optimizer_groups,
		expected_global_step=2,
		require_full_epoch_history=False,
	)

	assert evidence['latest']['global_step'] == 2
	payload['global_step'] = 1
	with pytest.raises(ValueError, match='finish at global step 2'):
		soft_validation._checkpoint_evidence(  # noqa: SLF001
			training,
			hard_trainability_summary=payload['trainability_summary'],
			hard_optimizer_group_identity=optimizer_groups,
			expected_global_step=2,
			require_full_epoch_history=False,
		)


def test_soft_smoke_evidence_requires_isolation_and_initial_parity(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	full_root = artifact_root / 'full'
	smoke_root = artifact_root / 'smoke'
	config = F3M5SoftPosteriorPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root / 'pretraining',
		posterior_manifest=artifact_root / 'posterior.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_handoff=artifact_root / 'hard_handoff.json',
		soft_smoke_config=tmp_path / 'smoke.yaml',
		soft_full_config=tmp_path / 'soft.yaml',
	)
	full = {
		'paths': {'output_root': str(full_root)},
		'identity': {
			'runtime_identity': {'device': 'auto', 'workers': 4},
			'scientific_identity': {'train': {'max_steps': None}},
		},
		'train': {'device': 'auto', 'max_steps': None},
	}
	smoke = deepcopy(full)
	smoke['paths']['output_root'] = str(smoke_root)
	smoke['identity']['runtime_identity']['device'] = 'cpu'
	smoke['identity']['scientific_identity']['train']['max_steps'] = 2
	smoke['train'].update(device='cpu', max_steps=2)
	checkpoint_evidence = {
		'latest': {
			'epoch': 1,
			'training_state': {'checkpoint_kind': 'step'},
		},
		'identity': {
			'target_representation': 'ordered_path_state_posterior_v1',
			'consistency_weight': 0.0,
		},
	}
	monkeypatch.setattr(
		soft_validation,
		'_checkpoint_evidence',
		lambda *_args, **_kwargs: checkpoint_evidence,
	)
	monkeypatch.setattr(
		soft_validation, '_initial_hashes', lambda _: ('a' * 64, 'b' * 64)
	)

	evidence = soft_validation._smoke_evidence(  # noqa: SLF001
		config,
		full=full,
		smoke=smoke,
		hard_trainability_summary={},
		hard_optimizer_group_identity=[],
	)

	assert evidence is checkpoint_evidence
	monkeypatch.setattr(
		soft_validation,
		'_initial_hashes',
		lambda training: ('a' * 64, 'b' * 64)
		if training is full
		else ('c' * 64, 'd' * 64),
	)
	with pytest.raises(ValueError, match='initial state hashes differ'):
		soft_validation._smoke_evidence(  # noqa: SLF001
			config,
			full=full,
			smoke=smoke,
			hard_trainability_summary={},
			hard_optimizer_group_identity=[],
		)
	monkeypatch.setattr(
		soft_validation, '_initial_hashes', lambda _: ('a' * 64, 'b' * 64)
	)
	full_root.mkdir()
	with pytest.raises(ValueError, match='must remain unmodified'):
		soft_validation._smoke_evidence(  # noqa: SLF001
			config,
			full=full,
			smoke=smoke,
			hard_trainability_summary={},
			hard_optimizer_group_identity=[],
		)


def test_soft_complete_requires_all_canonical_valid_token_identities(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	config = F3M5SoftPosteriorPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root / 'pretraining',
		posterior_manifest=artifact_root / 'posterior.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_handoff=artifact_root / 'hard_handoff.json',
		soft_smoke_config=tmp_path / 'smoke.yaml',
		soft_full_config=tmp_path / 'soft.yaml',
	)
	for model_tag in (
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1',
		'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
	):
		path = (
			artifact_root
			/ 'embeddings/f3/facies_benchmark_v1'
			/ model_tag
			/ 'overlap_x16/f3_facies_benchmark.valid_tokens.npy'
		)
		path.parent.mkdir(parents=True, exist_ok=True)
		np.save(path, np.ones((1, 1, 1), dtype=np.bool_))

	identities = _canonical_valid_token_identities(config)

	assert set(identities) == {'mae', 'current_k6', 'mh_nocons'}
	current_path = Path(identities['current_k6']['path'])
	np.save(current_path, np.zeros((1, 1, 1), dtype=np.bool_))
	with pytest.raises(
		ValueError, match='canonical valid-token identities do not match'
	):
		_canonical_valid_token_identities(config)


def test_soft_handoff_binds_trainability_and_canonical_valid_token_identities(
	tmp_path: Path,
) -> None:
	best_path = tmp_path / 'best.pt'
	best_path.write_bytes(b'best checkpoint')
	digest = 'a' * 64
	targets = {
		'target_representation': 'ordered_path_state_posterior_v1',
		'posterior_manifest_path': 'posterior.json',
		'posterior_manifest_sha256': digest,
		'posterior_semantics': 'ordered_path_cost_gibbs_state_marginal_v1',
		'posterior_cost_temperature': 1.0,
		'posterior_head_hashes': {
			str(head_k): {
				'f3': dict.fromkeys(('posterior', 'valid_tokens', 'metadata'), digest)
			}
			for head_k in (6, 8, 10)
		},
		'posterior_source_hard_manifest': {},
		'posterior_source_embedding': {},
		'posterior_head_diagnostics': {},
		'initial_student_state_sha256': digest,
		'initial_head_state_sha256': digest,
		'hard_baseline_config': 'hard.yaml',
		'hard_baseline_handoff': 'hard_handoff.json',
	}
	trainability = {
		'trainable_parameter_count': 1,
		'frozen_parameter_count': 2,
		'trainable_names': ['encoder.layers.7.weight'],
	}
	evidence = {
		**targets,
		'best_path': best_path,
		'best': {'trainability_summary': trainability},
		'selection': {
			'sha256': digest,
			'selected': {
				'epoch': 25,
				'global_step': 25600,
				'checkpoint_kind': 'epoch',
				'loss': 1.0,
			},
		},
		'identity': {
			'optimizer_group_identity': [
				{'name': 'head', 'parameter_names': ['head.fixture']}
			]
		},
		'embedding': {
			'root': 'embeddings',
			'metadata_path': 'metadata.json',
			'metadata_sha256': digest,
			'embeddings_sha256': digest,
			'valid_tokens_sha256': digest,
			'embeddings_shape': [76, 113, 32, 384],
			'embeddings_dtype': 'float16',
			'valid_tokens_shape': [76, 113, 32],
			'valid_tokens_dtype': 'bool',
			'finite_valid_count': 1,
			'canonical_valid_token_identities': {
				role: {'path': f'{role}.npy', 'sha256': digest}
				for role in ('mae', 'current_k6', 'mh_nocons')
			},
		},
	}
	handoff = _soft_handoff(evidence)
	path = tmp_path / 'soft_handoff.json'
	path.write_text(json.dumps(handoff), encoding='utf-8')

	loaded = soft_validation.load_f3_m5_soft_posterior_pretraining_handoff(path)

	assert loaded['checkpoint']['trainability_summary'] == trainability
	expected_identities = evidence['embedding']['canonical_valid_token_identities']
	loaded_identities = loaded['embedding']['canonical_valid_token_identities']
	assert loaded_identities == expected_identities
	handoff['checkpoint'].pop('trainability_summary')
	path.write_text(json.dumps(handoff), encoding='utf-8')
	with pytest.raises(TypeError, match='trainability summary'):
		soft_validation.load_f3_m5_soft_posterior_pretraining_handoff(path)


def test_control_config_must_identify_the_canonical_current_k6_run(
	tmp_path: Path,
) -> None:
	config = F3MultiHeadPretrainingValidationConfig(
		artifact_root=tmp_path,
		experiment_root=tmp_path / 'pretraining',
		target_manifest=tmp_path / 'targets.json',
		control_full_config=tmp_path / 'control.yaml',
		nocons_full_config=tmp_path / 'nocons.yaml',
		cons010_full_config=tmp_path / 'cons010.yaml',
	)
	control = {
		'paths': {
			'output_root': str(
				config.experiment_root
				/ 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
			)
		},
		'identity': {
			'model_tag': 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
		},
		'pseudo_targets': {'k': 6},
		'head': {'num_prototypes': 6},
	}
	_validate_control_config(config, control)
	control['identity']['model_tag'] = 'unrelated_control'
	with pytest.raises(ValueError, match='control model tag mismatch'):
		_validate_control_config(config, control)


def test_dry_run_reports_a_failure_and_plan_for_each_candidate(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = F3MultiHeadPretrainingValidationConfig(
		artifact_root=tmp_path,
		experiment_root=tmp_path / 'pretraining',
		target_manifest=tmp_path / 'targets.json',
		control_full_config=tmp_path / 'control.yaml',
		nocons_full_config=tmp_path / 'nocons.yaml',
		cons010_full_config=tmp_path / 'cons010.yaml',
	)
	monkeypatch.setattr(
		pretraining_validation,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': [6, 8, 10]},
	)
	monkeypatch.setattr(
		pretraining_validation, '_manifest_per_head_target_hashes', lambda _target: {}
	)
	monkeypatch.setattr(pretraining_validation, '_training_config', lambda _path: {})
	monkeypatch.setattr(
		pretraining_validation, '_validate_control_config', lambda *_: None
	)
	monkeypatch.setattr(
		pretraining_validation,
		'_validate_candidate_config_contract',
		lambda *_args, **_kwargs: None,
	)

	def fail_candidate(*_args: object, variant: str, **_kwargs: object) -> None:
		raise ValueError(f'{variant} checkpoint is unavailable')

	monkeypatch.setattr(pretraining_validation, '_checkpoint_evidence', fail_candidate)

	result = validate_f3_multi_head_pretraining(
		config, phase='checkpoints', dry_run=True
	)
	assert result.published_handoffs == ()
	for variant in ('nocons', 'cons010'):
		evidence = result.candidates[variant]
		assert evidence['status'] == 'FAIL'
		assert variant in evidence['error']
		assert evidence['planned_action'].startswith('strat_hmm_pretext_mh_k6810_')


def test_public_handoff_loader_requires_complete_pass_schema(tmp_path: Path) -> None:
	path = tmp_path / 'multi_head_handoff.json'
	payload = _handoff_payload()
	path.write_text(json.dumps(payload), encoding='utf-8')
	assert load_f3_multi_head_pretraining_handoff(path)['status'] == 'PASS'
	payload['embedding'] = {}
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(TypeError, match=r'handoff embedding\.root is missing'):
		load_f3_multi_head_pretraining_handoff(path)
	payload = _handoff_payload()
	payload['stratigraphy_pretext'].pop('consistency_beta')
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(TypeError, match=r'consistency_beta is missing'):
		load_f3_multi_head_pretraining_handoff(path)
	for field, value, message in (
		('best_epoch', 24, 'best identity does not match selected checkpoint'),
		('best_global_step', 25500, 'best identity does not match selected checkpoint'),
		('selection_history_event_count', 0, 'selection history must not be empty'),
		(
			'selection_history_schema_version',
			2,
			'selection history schema version mismatch',
		),
	):
		payload = _handoff_payload()
		payload['checkpoint'][field] = value
		path.write_text(json.dumps(payload), encoding='utf-8')
		with pytest.raises(ValueError, match=message):
			load_f3_multi_head_pretraining_handoff(path)


def _handoff_payload() -> dict[str, object]:
	return {
		'artifact_type': 'f3_multi_head_pretraining_handoff',
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		'variant': 'nocons',
		'checkpoint': {
			'path': '/artifact/best.pt',
			'sha256': 'a' * 64,
			'latest_path': '/artifact/latest.pt',
			'latest_sha256': 'b' * 64,
			'best_epoch': 25,
			'best_global_step': 25600,
			'selection_metric': 'metrics.loss',
			'selection_history_schema_version': 1,
			'selection_history_event_count': 1,
			'selected_checkpoint_kind': 'epoch',
			'selected_epoch': 25,
			'selected_global_step': 25600,
			'selected_loss': 0.1,
			'selection_history_sha256': '1' * 64,
		},
		'embedding': {
			'root': '/artifact/overlap_x16',
			'metadata_path': '/artifact/metadata.json',
			'metadata_sha256': 'c' * 64,
			'embeddings_sha256': 'd' * 64,
			'valid_tokens_sha256': 'e' * 64,
		},
		'embedding_metadata_sha256': 'c' * 64,
		'stratigraphy_pretext': {
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_manifest_path': '/artifact/targets.json',
			'target_manifest_sha256': 'f' * 64,
			'per_head_target_sha256': {
				str(head_k): {
					'f3': dict.fromkeys(
						('labels', 'confidence', 'valid_tokens', 'metadata'),
						f'{head_k:x}' * 64,
					)
				}
				for head_k in (6, 8, 10)
			},
			'consistency_policy': 'normalized_order_smooth_l1_v1',
			'consistency_weight': 0.0,
			'consistency_beta': 0.1,
			'scientific_identity_sha256': '2' * 64,
			'initial_student_state_sha256': '3' * 64,
			'initial_head_state_sha256': '4' * 64,
		},
	}


def test_public_handoff_loader_rejects_empty_per_head_target_hashes(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'multi_head_handoff.json'
	payload = _handoff_payload()
	payload['stratigraphy_pretext']['per_head_target_sha256'] = {}
	path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='per_head_target_sha256 K keys mismatch'):
		load_f3_multi_head_pretraining_handoff(path)


def test_complete_publish_requires_quarantine_flag_for_stale_handoff(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'multi_head_handoff.json'
	path.write_text('{"status": "partial"}', encoding='utf-8')

	with pytest.raises(ValueError, match='pass --quarantine-invalid'):
		_publish_handoff(
			path,
			_handoff_payload(),
			only_missing=False,
			_quarantine_invalid=False,
		)

	assert path.read_text(encoding='utf-8') == '{"status": "partial"}'
	assert not list(tmp_path.glob('multi_head_handoff.json.quarantine.*'))


def test_complete_publish_quarantines_stale_handoff_when_requested(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'multi_head_handoff.json'
	path.write_text('{"status": "partial"}', encoding='utf-8')
	handoff = _handoff_payload()

	published = _publish_handoff(
		path,
		handoff,
		only_missing=False,
		_quarantine_invalid=True,
	)

	assert published
	assert load_f3_multi_head_pretraining_handoff(path) == handoff
	quarantined = list(tmp_path.glob('multi_head_handoff.json.quarantine.*'))
	assert len(quarantined) == 1
	assert quarantined[0].read_text(encoding='utf-8') == '{"status": "partial"}'


def test_publish_preserves_canonical_handoff_when_atomic_write_fails(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	path = tmp_path / 'multi_head_handoff.json'
	previous_handoff = _handoff_payload()
	previous_handoff['checkpoint']['sha256'] = '0' * 64
	previous = json.dumps(previous_handoff)
	path.write_text(previous, encoding='utf-8')

	def fail_atomic_write(*_args: object, **_kwargs: object) -> None:
		raise OSError('simulated write failure')

	monkeypatch.setattr(pretraining_validation, '_atomic_json', fail_atomic_write)
	with pytest.raises(OSError, match='simulated write failure'):
		_publish_handoff(
			path,
			_handoff_payload(),
			only_missing=False,
			_quarantine_invalid=True,
		)

	assert path.read_text(encoding='utf-8') == previous
	quarantined = list(tmp_path.glob('multi_head_handoff.json.quarantine.*'))
	assert len(quarantined) == 1
	assert quarantined[0].read_text(encoding='utf-8') == previous


def test_only_missing_reuses_an_exact_live_handoff(tmp_path: Path) -> None:
	path = tmp_path / 'multi_head_handoff.json'
	handoff = _handoff_payload()
	path.write_text(json.dumps(handoff), encoding='utf-8')

	published = _publish_handoff(
		path,
		handoff,
		only_missing=True,
		_quarantine_invalid=False,
	)

	assert not published
	assert not list(tmp_path.glob('multi_head_handoff.json.quarantine.*'))


def test_complete_phase_publishes_two_handoffs_from_synthetic_artifacts(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Exercise the public complete phase through both atomic handoff publishes."""
	artifact_root = tmp_path / 'artifacts'
	experiment_root = artifact_root / 'pretraining'
	target_manifest = artifact_root / 'targets.json'
	target_manifest.parent.mkdir(parents=True)
	target_manifest.write_text('{}', encoding='utf-8')
	config_paths = {
		name: artifact_root / f'{name}.yaml'
		for name in ('control', 'nocons', 'cons010')
	}
	for path in config_paths.values():
		path.write_text('{}', encoding='utf-8')
	config = F3MultiHeadPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=experiment_root,
		target_manifest=target_manifest,
		control_full_config=config_paths['control'],
		nocons_full_config=config_paths['nocons'],
		cons010_full_config=config_paths['cons010'],
	)
	expected_targets = {
		str(head_k): {
			'f3': dict.fromkeys(
				('labels', 'confidence', 'valid_tokens', 'metadata'),
				f'{head_k:x}' * 64,
			)
		}
		for head_k in (6, 8, 10)
	}
	control = {
		'paths': {
			'output_root': str(
				experiment_root / 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
			)
		},
		'identity': {
			'model_tag': 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
		},
		'pseudo_targets': {'k': 6},
		'head': {'num_prototypes': 6},
	}
	training: dict[Path, dict[str, object]] = {config_paths['control']: control}
	for variant, model_tag, weight in (
		(
			'nocons',
			'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
			0.0,
		),
		(
			'cons010',
			'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
			0.1,
		),
	):
		training[config_paths[variant]] = {
			'paths': {'output_root': str(experiment_root / model_tag)},
			'identity': {
				'model_tag': model_tag,
				'scientific_identity': {
					'variant': variant,
					'consistency_weight': weight,
				},
			},
			'loss': {'consistency_weight': weight},
			'pseudo_targets': {'manifest': str(target_manifest)},
		}

	monkeypatch.setattr(
		pretraining_validation,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': [6, 8, 10]},
	)
	monkeypatch.setattr(
		pretraining_validation,
		'_manifest_per_head_target_hashes',
		lambda _target: expected_targets,
	)
	monkeypatch.setattr(
		pretraining_validation,
		'_training_config',
		lambda path: training[path],
	)
	monkeypatch.setattr(
		pretraining_validation,
		'_canonical_k6_valid_tokens_sha256',
		lambda *_args: 'a' * 64,
	)

	def checkpoint_evidence(
		_validation_config: F3MultiHeadPretrainingValidationConfig,
		candidate: dict[str, object],
		*,
		variant: str,
		model_tag: str,
		weight: float,
		expected_per_head_targets: dict[str, object],
	) -> dict[str, object]:
		root = Path(candidate['paths']['output_root'])  # type: ignore[index]
		root.mkdir(parents=True)
		best, latest = root / 'best.pt', root / 'latest.pt'
		torch.save({'synthetic': variant, 'kind': 'best'}, best)
		torch.save({'synthetic': variant, 'kind': 'latest'}, latest)
		scientific = candidate['identity']['scientific_identity']  # type: ignore[index]
		return {
			'root': root,
			'best_path': best,
			'latest_path': latest,
			'best': {'epoch': 25, 'global_step': 25600},
			'latest': {'epoch': 25, 'global_step': 25600},
			'identity': {
				'model_tag': model_tag,
				'head_spec': 'multi_resolution_ordered_prototypes_v1',
				'head_ks': [6, 8, 10],
				'target_manifest': {
					'path': str(target_manifest),
					'sha256': pretraining_validation.file_sha256(target_manifest),
				},
				'per_head_targets': expected_per_head_targets,
				'consistency_policy': 'normalized_order_smooth_l1_v1',
				'consistency_weight': weight,
				'consistency_beta': 0.1,
				'scientific_identity_sha256': scientific_identity_sha256(scientific),
				'initial_student_state_sha256': 'b' * 64,
				'initial_head_state_sha256': 'c' * 64,
				'teacher_checkpoint_sha256': 'd' * 64,
				'student_init_checkpoint_sha256': 'e' * 64,
				'optimizer_group_identity': [
					{'name': 'head', 'parameter_names': ['head.fixture']},
					{'name': 'encoder', 'parameter_names': ['student.fixture']},
				],
			},
			'checkpoint_selection': {
				'schema_version': 1,
				'event_count': 1,
				'selected': {
					'checkpoint_kind': 'epoch',
					'epoch': 25,
					'global_step': 25600,
					'loss': 1.0,
				},
				'sha256': '1' * 64,
			},
		}

	def embedding_evidence(
		_validation_config: F3MultiHeadPretrainingValidationConfig,
		_checkpoint: dict[str, object],
		model_tag: str,
		*,
		canonical_valid_tokens_sha256: str,
	) -> dict[str, object]:
		root = artifact_root / 'synthetic-embeddings' / model_tag
		root.mkdir(parents=True)
		embeddings, valid, metadata = (
			root / 'embeddings.npy',
			root / 'valid_tokens.npy',
			root / 'embedding_metadata.json',
		)
		np.save(embeddings, np.zeros((1, 1, 1, 1), dtype=np.float16))
		np.save(valid, np.ones((1, 1, 1), dtype=np.bool_))
		metadata.write_text(json.dumps({'synthetic': model_tag}), encoding='utf-8')
		return {
			'root': root,
			'metadata_path': metadata,
			'metadata_sha256': pretraining_validation.file_sha256(metadata),
			'embeddings_sha256': pretraining_validation.file_sha256(embeddings),
			'valid_tokens_sha256': canonical_valid_tokens_sha256,
		}

	monkeypatch.setattr(
		pretraining_validation, '_checkpoint_evidence', checkpoint_evidence
	)
	monkeypatch.setattr(
		pretraining_validation, '_embedding_evidence', embedding_evidence
	)

	result = validate_f3_multi_head_pretraining(config, phase='complete')

	assert len(result.published_handoffs) == 2
	for handoff in result.published_handoffs:
		assert load_f3_multi_head_pretraining_handoff(handoff)['status'] == 'PASS'
		assert (handoff.parent / 'checkpoint_validation.json').is_file()
		assert (handoff.parent / 'embedding_validation.json').is_file()


def test_checkpoint_per_head_targets_must_match_the_canonical_manifest() -> None:
	manifest = {
		'heads': {
			str(head_k): {
				'surveys': {
					'f3': {
						name: {'sha256': f'{head_k:x}' * 64}
						for name in (
							'labels',
							'confidence',
							'valid_tokens',
							'metadata',
						)
					}
				}
			}
			for head_k in (6, 8, 10)
		}
	}
	expected = _manifest_per_head_target_hashes(manifest)
	assert expected['6']['f3']['labels'] == '6' * 64
	manifest['heads']['8']['surveys']['f3']['labels']['sha256'] = 'a' * 64
	assert _manifest_per_head_target_hashes(manifest) != expected


def test_checkpoint_scientific_identity_must_match_the_resolved_training_config(
	tmp_path: Path,
) -> None:
	scientific_identity = {
		'experiment_role': 'multi_head_ordered_pretext',
		'variant': 'nocons',
		'consistency_weight': 0.0,
		'target_head_hashes': {},
	}
	target = tmp_path / 'targets.json'
	target.write_text('{}', encoding='utf-8')
	model_tag = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
	root = tmp_path / model_tag
	config = F3MultiHeadPretrainingValidationConfig(
		artifact_root=tmp_path,
		experiment_root=tmp_path,
		target_manifest=target,
		control_full_config=tmp_path / 'control.yaml',
		nocons_full_config=tmp_path / 'nocons.yaml',
		cons010_full_config=tmp_path / 'cons010.yaml',
	)
	training = {
		'paths': {'output_root': str(root)},
		'identity': {'scientific_identity': scientific_identity},
	}
	teacher, student = tmp_path / 'teacher.pt', tmp_path / 'student-init.pt'
	teacher.write_bytes(b'teacher checkpoint')
	student.write_bytes(b'student checkpoint')
	training['teacher'] = {'checkpoint': str(teacher)}
	training['student'] = {'init_checkpoint': str(student)}
	identity = {
		'scientific_identity_sha256': scientific_identity_sha256(scientific_identity),
		'teacher_checkpoint_sha256': pretraining_validation.file_sha256(teacher),
		'student_init_checkpoint_sha256': pretraining_validation.file_sha256(student),
	}
	payload = {
		'stratigraphy_checkpoint': {
			**identity,
			'model_tag': model_tag,
			'output_root': str(root),
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'consistency_weight': 0.0,
			'consistency_policy': 'normalized_order_smooth_l1_v1',
			'consistency_beta': 0.1,
			'target_manifest': {
				'path': str(target),
				'sha256': pretraining_validation.file_sha256(target),
			},
			'per_head_targets': {},
		}
	}
	_identity_contract(config, training, payload, model_tag, 0.0, {})
	teacher.write_bytes(b'replaced teacher checkpoint')
	with pytest.raises(
		ValueError,
		match='teacher_checkpoint_sha256 does not match configured source',
	):
		_identity_contract(config, training, payload, model_tag, 0.0, {})
	teacher.write_bytes(b'teacher checkpoint')
	student.write_bytes(b'replaced student checkpoint')
	with pytest.raises(
		ValueError,
		match='student_init_checkpoint_sha256 does not match configured source',
	):
		_identity_contract(config, training, payload, model_tag, 0.0, {})
	student.write_bytes(b'student checkpoint')
	scientific_identity['target_head_hashes'] = {'6': {}}
	with pytest.raises(
		ValueError,
		match='scientific identity per-head target hashes do not match target manifest',
	):
		_identity_contract(config, training, payload, model_tag, 0.0, {})
	scientific_identity['target_head_hashes'] = {}
	payload['stratigraphy_checkpoint']['scientific_identity_sha256'] = '0' * 64
	with pytest.raises(
		ValueError, match='scientific identity does not match training config'
	):
		_identity_contract(config, training, payload, model_tag, 0.0, {})

	identity = {
		'scientific_identity_sha256': scientific_identity_sha256(scientific_identity)
	}
	configs = {
		'nocons': {
			'paths': {'output_root': '/artifact/nocons'},
			'identity': {
				'model_tag': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
				'scientific_identity': scientific_identity,
			},
			'loss': {'consistency_weight': 0.0},
		},
		'cons010': {
			'paths': {'output_root': '/artifact/cons010'},
			'identity': {
				'model_tag': 'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
				'scientific_identity': {
					**scientific_identity,
					'variant': 'cons010',
					'consistency_weight': 0.1,
				},
			},
			'loss': {'consistency_weight': 0.1},
		},
	}
	_validate_pair_config_contract(configs)
	left = {'identity': identity}
	right = {
		'identity': {
			'scientific_identity_sha256': scientific_identity_sha256(
				configs['cons010']['identity']['scientific_identity']
			)
		}
	}
	_validate_pair(left, right, configs)
	identity['scientific_identity_sha256'] = '0' * 64
	with pytest.raises(
		ValueError, match='paired pretraining scientific identity mismatch: nocons'
	):
		_validate_pair(left, right, configs)


def test_embedding_validation_rejects_metadata_without_checkpoint_binding(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	model_tag = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
	config = F3MultiHeadPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root / 'pretraining',
		target_manifest=artifact_root / 'targets.json',
		control_full_config=tmp_path / 'control.yaml',
		nocons_full_config=tmp_path / 'nocons.yaml',
		cons010_full_config=tmp_path / 'cons010.yaml',
	)
	best = tmp_path / 'best.pt'
	best.write_bytes(b'best checkpoint')
	root = (
		artifact_root / 'embeddings/f3/facies_benchmark_v1' / model_tag / 'overlap_x16'
	)
	files = pretraining_validation.output_paths(root, 'f3_facies_benchmark')
	root.mkdir(parents=True)
	files.embeddings.touch()
	files.valid_tokens.touch()
	files.metadata.write_text(
		json.dumps(
			{
				'checkpoint_path': str(best),
				'checkpoint_sha256': '0' * 64,
			}
		),
		encoding='utf-8',
	)

	with pytest.raises(
		ValueError, match=r'embedding metadata does not bind selected best\.pt'
	):
		pretraining_validation._embedding_evidence(  # noqa: SLF001
			config,
			{'best_path': best},
			model_tag,
			canonical_valid_tokens_sha256='a' * 64,
		)


def test_pair_config_contract_rejects_unbound_scientific_consistency_weight() -> None:
	configs = {
		'nocons': {
			'paths': {'output_root': '/artifact/nocons'},
			'identity': {
				'model_tag': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
				'scientific_identity': {'variant': 'nocons', 'consistency_weight': 0.0},
			},
			'loss': {'consistency_weight': 0.0},
		},
		'cons010': {
			'paths': {'output_root': '/artifact/cons010'},
			'identity': {
				'model_tag': 'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
				'scientific_identity': {
					'variant': 'cons010',
					'consistency_weight': 0.0,
				},
			},
			'loss': {'consistency_weight': 0.1},
		},
	}

	with pytest.raises(
		ValueError,
		match='scientific consistency weight must match loss consistency weight',
	):
		_validate_pair_config_contract(configs)


def test_initial_state_hashes_must_bind_the_actual_initial_states() -> None:
	student = {'encoder.layers.7.weight': torch.tensor([1.0])}
	head = {'heads.k6.prototypes': torch.tensor([6.0])}
	identity = {
		'initial_student_state_sha256': _state_sha256(student),
		'initial_head_state_sha256': _state_sha256(head),
	}
	_validate_initial_state_hashes(identity, student_state=student, head_state=head)
	identity['initial_head_state_sha256'] = '0' * 64
	with pytest.raises(ValueError, match='initial head state SHA-256 mismatch'):
		_validate_initial_state_hashes(identity, student_state=student, head_state=head)


def test_initial_state_validation_reconstructs_and_binds_initial_states(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	student = torch.nn.Linear(1, 1)
	heads = torch.nn.Module()
	heads.register_parameter('prototypes', torch.nn.Parameter(torch.tensor([6.0])))
	identity = {
		'initial_student_state_sha256': _state_sha256(student.state_dict()),
		'initial_head_state_sha256': _state_sha256(heads.state_dict()),
	}
	monkeypatch.setattr(
		pretraining_validation,
		'build_strat_hmm_components',
		lambda _training, *, device: SimpleNamespace(
			student=student, heads=heads, device=device
		),
	)
	payload = {'stratigraphy_checkpoint': identity}
	pretraining_validation._validate_initial_states(  # noqa: SLF001
		{'train': {'seed': 42}}, payload
	)
	identity['initial_student_state_sha256'] = '0' * 64
	with pytest.raises(ValueError, match='initial student state SHA-256 mismatch'):
		pretraining_validation._validate_initial_states(  # noqa: SLF001
			{'train': {'seed': 42}}, payload
		)


def test_best_selection_accepts_a_step_selected_before_the_final_epoch() -> None:
	step = {
		'sequence': 0,
		'epoch': 25,
		'global_step': 25500,
		'checkpoint_kind': 'step',
		'batch_index': 499,
		'loss': 0.262,
	}
	epoch = {
		'sequence': 1,
		'epoch': 25,
		'global_step': 25600,
		'checkpoint_kind': 'epoch',
		'batch_index': None,
		'loss': 0.310,
	}
	selection = {
		'schema_version': 1,
		'criterion': 'metrics.loss',
		'improvement_policy': 'strictly_lower_loss_v1',
		'events': [
			{
				**step,
				'previous_best_score': None,
				'best_updated': True,
				'best_score_after': 0.262,
			},
			{
				**epoch,
				'previous_best_score': 0.262,
				'best_updated': False,
				'best_score_after': 0.262,
			},
		],
		'selected': step,
	}
	best_selection = {
		**selection,
		'events': [selection['events'][0]],
	}
	best = {
		'epoch': 25,
		'global_step': 25500,
		'metrics': {'loss': 0.262},
		'training_state': {'checkpoint_kind': 'step', 'batch_index': 499},
		'checkpoint_selection': best_selection,
	}
	latest = {
		'epoch': 25,
		'global_step': 25600,
		'metrics': {'loss': 0.310},
		'training_state': {'checkpoint_kind': 'epoch', 'batch_index': None},
		'checkpoint_selection': selection,
	}

	assert _validate_best_selection(best, latest, variant='nocons')['selected'] == step


def test_checkpoint_phase_accepts_a_step_best_before_the_final_epoch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	experiment_root = artifact_root / 'pretraining'
	config = F3MultiHeadPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=experiment_root,
		target_manifest=artifact_root / 'targets.json',
		control_full_config=artifact_root / 'control.yaml',
		nocons_full_config=artifact_root / 'nocons.yaml',
		cons010_full_config=artifact_root / 'cons010.yaml',
	)
	step = {
		'sequence': 0,
		'epoch': 25,
		'global_step': 25500,
		'checkpoint_kind': 'step',
		'batch_index': 499,
		'loss': 0.262,
	}
	epoch = {
		'sequence': 1,
		'epoch': 25,
		'global_step': 25600,
		'checkpoint_kind': 'epoch',
		'batch_index': None,
		'loss': 0.310,
	}
	selection = {
		'schema_version': 1,
		'criterion': 'metrics.loss',
		'improvement_policy': 'strictly_lower_loss_v1',
		'events': [
			{
				**step,
				'previous_best_score': None,
				'best_updated': True,
				'best_score_after': 0.262,
			},
			{
				**epoch,
				'previous_best_score': 0.262,
				'best_updated': False,
				'best_score_after': 0.262,
			},
		],
		'selected': step,
	}
	best_selection = {**selection, 'events': [selection['events'][0]]}
	training: dict[Path, dict[str, object]] = {
		config.control_full_config: {},
	}
	for variant, model_tag, _weight in pretraining_validation._CANDIDATES:  # noqa: SLF001
		root = experiment_root / model_tag
		root.mkdir(parents=True)
		best = {
			'epoch': 25,
			'global_step': 25500,
			'metrics': {'loss': 0.262},
			'training_state': {'checkpoint_kind': 'step', 'batch_index': 499},
			'checkpoint_selection': best_selection,
			'stratigraphy_checkpoint': {},
		}
		latest = {
			'epoch': 25,
			'global_step': 25600,
			'metrics': {'loss': 0.310},
			'training_state': {'checkpoint_kind': 'epoch', 'batch_index': None},
			'checkpoint_selection': selection,
			'stratigraphy_checkpoint': {},
		}
		torch.save(best, root / 'best.pt')
		torch.save(latest, root / 'latest.pt')
		(root / 'multi_head_epoch_metrics.csv').write_text(
			'epoch,global_step,loss\n'
			+ ''.join(
				f'{number},{number * 1024},0.310\n'
				for number in range(1, 26)
			),
			encoding='utf-8',
		)
		training[getattr(config, f'{variant}_full_config')] = {
			'paths': {'output_root': str(root)},
		}
	monkeypatch.setattr(
		pretraining_validation,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': [6, 8, 10]},
	)
	monkeypatch.setattr(
		pretraining_validation, '_manifest_per_head_target_hashes', lambda _target: {}
	)
	monkeypatch.setattr(
		pretraining_validation, '_training_config', lambda path: training[path]
	)
	monkeypatch.setattr(
		pretraining_validation, '_validate_control_config', lambda *_: None
	)
	monkeypatch.setattr(
		pretraining_validation,
		'_validate_candidate_config_contract',
		lambda *_args, **_kwargs: None,
	)
	monkeypatch.setattr(
		pretraining_validation,
		'validate_stratigraphy_checkpoint_payload',
		lambda *_args, **_kwargs: None,
	)
	monkeypatch.setattr(
		pretraining_validation, '_identity_contract', lambda *_: None
	)
	monkeypatch.setattr(
		pretraining_validation, '_validate_initial_states', lambda *_: None
	)
	monkeypatch.setattr(
		pretraining_validation, '_validate_freeze_contract', lambda *_: None
	)
	monkeypatch.setattr(
		pretraining_validation, '_validate_pair_config_contract', lambda *_: None
	)
	monkeypatch.setattr(pretraining_validation, '_validate_pair', lambda *_: None)

	result = validate_f3_multi_head_pretraining(config, phase='checkpoints')

	for variant in ('nocons', 'cons010'):
		evidence = result.candidates[variant]
		assert evidence['status'] == 'PASS'
		selection_evidence = evidence['checkpoint_selection']
		assert isinstance(selection_evidence, dict)
		assert selection_evidence['selected']['global_step'] == 25500
		assert (
			experiment_root
			/ evidence['planned_action'].split(':', maxsplit=1)[0]
			/ 'preflight'
			/ 'checkpoint_validation.json'
		).is_file()


def test_freeze_contract_rejects_a_multi_head_run_with_only_one_updated_head(
	tmp_path: Path,
) -> None:
	payload, training = _freeze_contract_inputs(tmp_path, changed_heads={6})

	with pytest.raises(ValueError, match='K=8 head to update'):
		_validate_freeze_contract(payload, training)


@pytest.mark.parametrize('group_index', [0, 1])
def test_freeze_contract_rejects_duplicate_optimizer_parameters(
	tmp_path: Path, group_index: int
) -> None:
	payload, training = _freeze_contract_inputs(tmp_path, changed_heads={6, 8, 10})
	groups = payload['stratigraphy_checkpoint']['optimizer_group_identity']
	groups[group_index]['parameter_names'].append(
		groups[group_index]['parameter_names'][0]
	)

	with pytest.raises(
		ValueError, match='optimizer parameters must appear exactly once'
	):
		_validate_freeze_contract(payload, training)


def test_freeze_contract_rejects_duplicate_optimizer_group(tmp_path: Path) -> None:
	payload, training = _freeze_contract_inputs(tmp_path, changed_heads={6, 8, 10})
	groups = payload['stratigraphy_checkpoint']['optimizer_group_identity']
	groups.append(dict(groups[0]))

	with pytest.raises(ValueError, match='optimizer groups are invalid'):
		_validate_freeze_contract(payload, training)


def _freeze_contract_inputs(
	tmp_path: Path, *, changed_heads: set[int]
) -> tuple[dict[str, object], dict[str, object]]:
	initial_student = {
		'encoder.layers.6.weight': torch.tensor([1.0]),
		'encoder.layers.7.weight': torch.tensor([2.0]),
	}
	init_path = tmp_path / 'student-init.pt'
	torch.save({'model_state_dict': initial_student}, init_path)
	initial_head = {
		f'heads.k{head_k}.prototypes': torch.tensor([float(head_k)])
		for head_k in (6, 8, 10)
	}
	current_head = {
		name: value + int(head_k in changed_heads)
		for name, value in initial_head.items()
		for head_k in (int(name.split('.')[1][1:]),)
	}
	initial_head_hashes = {
		name: _tensor_sha256(name, value) for name, value in initial_head.items()
	}
	head_names = [f'head.{name}' for name in initial_head]
	payload: dict[str, object] = {
		'trainability_summary': {'trainable_names': ['encoder.layers.7.weight']},
		'control_identity': {
			'initial_parameter_sha256': {'prototype_head': initial_head_hashes},
		},
		'stratigraphy_checkpoint': {
			'initial_head_state_sha256': _state_sha256(initial_head),
			'optimizer_group_identity': [
				{'name': 'head', 'parameter_names': head_names},
				{
					'name': 'encoder',
					'parameter_names': ['student.encoder.layers.7.weight'],
				},
			],
		},
		'stratigraphy_state_dict': current_head,
		'model_state_dict': {
			'encoder.layers.6.weight': initial_student['encoder.layers.6.weight'],
			'encoder.layers.7.weight': initial_student['encoder.layers.7.weight'] + 1,
		},
	}
	return payload, {
		'student': {'unfreeze_top_blocks': 1, 'init_checkpoint': str(init_path)}
	}
