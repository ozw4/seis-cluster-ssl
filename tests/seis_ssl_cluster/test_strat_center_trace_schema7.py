from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from seis_ssl_cluster.config.pretraining import (
	CENTER_TRACE_MODEL_TAG,
	CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT,
	_multi_head_target_hashes,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.embedding.extractor import _stratigraphy_pretext_metadata
from seis_ssl_cluster.models.mae import LearnedEncoderReplacementToken
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.prototypes import (
	MultiResolutionOrderedPrototypeHeads,
)
from seis_ssl_cluster.training.checkpoint import capture_rng_state, load_checkpoint
from seis_ssl_cluster.training.strat_hmm.resume import (
	restore_strat_hmm_training_checkpoint,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	_state_sha256,
	inspect_stratigraphy_checkpoint,
	save_strat_hmm_checkpoint,
	validate_stratigraphy_checkpoint_payload,
)
from tests.seis_ssl_cluster.test_config_strat_hmm_multi_head import (
	_multi_head_config,
)
from tests.seis_ssl_cluster.test_strat_checkpoint_extraction import (
	_multi_head_resume_config,
	_new_multi_head_components,
	_per_head_target_hashes,
	_sha256,
)


def test_center_trace_config_is_closed_and_records_fixed_identity(
	tmp_path: Path,
) -> None:
	config = _multi_head_config(tmp_path)
	manifest_path = Path(config['pseudo_targets']['manifest'])  # type: ignore[index]
	manifest = load_multi_head_target_manifest(
		manifest_path,
		validate_array_semantics=False,
	)
	config['spatial_context'] = deepcopy(CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT)
	config['student']['unfreeze_top_blocks'] = 1  # type: ignore[index]
	config['identity']['model_tag'] = CENTER_TRACE_MODEL_TAG  # type: ignore[index]
	scientific = config['identity']['scientific_identity']  # type: ignore[index]
	scientific.update(
		{
			'experiment_role': 'multi_head_center_trace_masked_hard_pretext',
			'variant': 'ctmask010_nocons',
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_representation': 'hard_viterbi_labels_v1',
			'target_manifest_sha256': config['identity']['scientific_identity'][
				'target_manifest_sha256'
			],
			'target_head_hashes': _multi_head_target_hashes(manifest),
			'objective_semantics': 'center_trace_masked_hmm_path_reconstruction_v1',
			'mask_semantics': 'xy_token_column_full_z_v1',
			'column_fraction': 0.10,
			'selection_policy': (
				'supervised_valid_xy_columns_round_half_up_leave_one_v1'
			),
			'replacement': 'learned_encoder_mask_token_v1',
			'replacement_initialization': 'normal_std_0p02_train_seed_salted_v1',
			'rng_policy': 'stateless_step_seed_v1',
			'masked_prototype_weight': 0.50,
			'visible_prototype_weight': 0.50,
			'distillation_scope': 'visible_only_v1',
			'supervised_loss': 'structured_hmm_center_trace_masked_hard_v1',
			'consistency_policy': 'disabled_for_center_trace_masked_v1',
		}
	)

	resolved = resolve_strat_hmm_pretext_config(config)

	assert resolved['spatial_context'] == CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT
	assert resolved['identity']['model_tag'] == CENTER_TRACE_MODEL_TAG  # type: ignore[index]
	assert resolved['identity']['scientific_identity']['head_ks'] == [  # type: ignore[index]
		6,
		8,
		10,
	]

	bad = deepcopy(config)
	bad['spatial_context']['column_fraction'] = 0.20  # type: ignore[index]
	with pytest.raises(ValueError, match='column_fraction'):
		resolve_strat_hmm_pretext_config(bad)


def _center_checkpoint_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> tuple[
	dict[str, object],
	dict[str, object],
	torch.nn.Module,
	torch.nn.Module,
	torch.nn.Module,
	dict[str, torch.Tensor],
	dict[str, torch.Tensor],
	dict[str, torch.Tensor],
]:
	config = _multi_head_resume_config(tmp_path, monkeypatch, variant='nocons')
	manifest_path = Path(config['pseudo_targets']['manifest'])  # type: ignore[index]
	student, heads, _ = _new_multi_head_components()
	replacement = LearnedEncoderReplacementToken(3, seed=17)
	config['model'] = {'encoder_dim': 3}
	config['spatial_context'] = deepcopy(CENTER_TRACE_SPATIAL_CONTEXT_CONTRACT)
	config['student']['unfreeze_top_blocks'] = 1  # type: ignore[index]
	config['train']['encoder_lr'] = 1.0e-4  # type: ignore[index]
	config['identity']['model_tag'] = CENTER_TRACE_MODEL_TAG  # type: ignore[index]
	hashes = _per_head_target_hashes()
	scientific = config['identity']['scientific_identity']  # type: ignore[index]
	scientific.clear()
	scientific.update(
		{
			'experiment_role': 'multi_head_center_trace_masked_hard_pretext',
			'variant': 'ctmask010_nocons',
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'head_projection_dim': 2,
			'head_temperature': 0.1,
			'head_normalize': True,
			'target_representation': 'hard_viterbi_labels_v1',
			'target_manifest_sha256': _sha256(manifest_path),
			'target_head_hashes': hashes,
			'objective_semantics': 'center_trace_masked_hmm_path_reconstruction_v1',
			'mask_semantics': 'xy_token_column_full_z_v1',
			'column_fraction': 0.10,
			'selection_policy': (
				'supervised_valid_xy_columns_round_half_up_leave_one_v1'
			),
			'replacement': 'learned_encoder_mask_token_v1',
			'replacement_initialization': 'normal_std_0p02_train_seed_salted_v1',
			'rng_policy': 'stateless_step_seed_v1',
			'masked_prototype_weight': 0.50,
			'visible_prototype_weight': 0.50,
			'distillation_scope': 'visible_only_v1',
			'supervised_loss': 'structured_hmm_center_trace_masked_hard_v1',
			'consistency_policy': 'disabled_for_center_trace_masked_v1',
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'consistency_weight': 0.0,
			'consistency_beta': 0.1,
			'distillation_weight': 0.2,
			'student_unfreeze_top_blocks': 1,
		}
	)
	optimizer = torch.optim.AdamW(
		[
			{'params': heads.parameters(), 'name': 'head', 'lr': 1.0e-3},
			{'params': student.parameters(), 'name': 'encoder', 'lr': 1.0e-4},
			{
				'params': replacement.parameters(),
				'name': 'spatial_context',
				'lr': 1.0e-3,
			},
		],
		weight_decay=0.05,
	)
	initial_student = deepcopy(student.state_dict())
	initial_heads = deepcopy(heads.state_dict())
	initial_replacement = deepcopy(replacement.state_dict())
	control = {
		'input_identities': {
			'teacher_checkpoint': {
				'sha256': _sha256(Path(config['teacher']['checkpoint'])),  # type: ignore[index]
			},
			'student_init_checkpoint': {
				'sha256': _sha256(Path(config['student']['init_checkpoint'])),  # type: ignore[index]
			},
		},
		'initial_state_sha256': {
			'student': _state_sha256(student.state_dict()),
			'head': _state_sha256(heads.state_dict()),
			'spatial_context': _state_sha256(replacement.state_dict()),
		},
	}
	with torch.no_grad():
		for parameter in (*student.parameters(), *heads.parameters()):
			parameter.add_(0.01)
		replacement.replacement_token.add_(0.01)
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = torch.Generator().manual_seed(3).get_state()
	path = save_strat_hmm_checkpoint(
		tmp_path / 'center.pt',
		student=student,
		head=heads,
		spatial_context=replacement,
		optimizer=optimizer,
		epoch=1,
		mae_config={'stage': 'train_amp_mae'},
		stratigraphy_config=config,
		metrics={'loss': 1.0},
		global_step=2,
		checkpoint_kind='step',
		batch_index=1,
		rng_state=rng_state,
		control_identity=control,
	)
	return (
		config,
		load_checkpoint(path, map_location='cpu'),
		student,
		heads,
		replacement,
		initial_student,
		initial_heads,
		initial_replacement,
	)


def test_schema7_roundtrip_restores_auxiliary_state_and_metadata(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	(
		config,
		payload,
		_student,
		_heads,
		replacement,
		initial_student,
		initial_heads,
		initial_replacement,
	) = _center_checkpoint_fixture(tmp_path, monkeypatch)
	identity = payload['stratigraphy_checkpoint']
	assert identity['schema_version'] == 7  # type: ignore[index]
	assert 'spatial_context_state_dict' in payload
	assert 'replacement_token' in payload['spatial_context_state_dict']  # type: ignore[index]
	validate_stratigraphy_checkpoint_payload(payload)
	inspection = inspect_stratigraphy_checkpoint(payload)
	assert inspection['objective'] == (
		'center_trace_masked_hmm_path_reconstruction_v1'
	)
	metadata = _stratigraphy_pretext_metadata(payload)
	assert metadata is not None
	assert metadata['target_representation'] == 'hard_viterbi_labels_v1'
	assert metadata['replacement'] == 'learned_encoder_mask_token_v1'
	assert metadata['distillation_scope'] == 'visible_only_v1'

	restored_student = torch.nn.Linear(2, 3)
	restored_student.load_state_dict(initial_student)
	restored_heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	restored_heads.load_state_dict(initial_heads)
	restored_replacement = LearnedEncoderReplacementToken(3, seed=17)
	restored_replacement.load_state_dict(initial_replacement)
	restored_optimizer = torch.optim.AdamW(
		[
			{'params': restored_heads.parameters(), 'name': 'head', 'lr': 1.0e-3},
			{'params': restored_student.parameters(), 'name': 'encoder', 'lr': 1.0e-4},
			{
				'params': restored_replacement.parameters(),
				'name': 'spatial_context',
				'lr': 1.0e-3,
			},
		],
		weight_decay=0.05,
	)
	resume = restore_strat_hmm_training_checkpoint(
		payload=payload,
		student=restored_student,
		head=restored_heads,
		spatial_context=restored_replacement,
		optimizer=restored_optimizer,
		scaler=None,
		amp_enabled=False,
		config=config,
	)
	assert resume.skip_batches == 2
	torch.testing.assert_close(
		restored_replacement.replacement_token,
		replacement.replacement_token.detach(),
	)


def test_schema7_resume_requires_the_replacement_token_module(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	(
		config,
		payload,
		_student,
		_heads,
		_replacement,
		initial_student,
		initial_heads,
		_initial_replacement,
	) = _center_checkpoint_fixture(tmp_path, monkeypatch)
	restored_student = torch.nn.Linear(2, 3)
	restored_student.load_state_dict(initial_student)
	restored_heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	restored_heads.load_state_dict(initial_heads)
	restored_replacement = LearnedEncoderReplacementToken(3, seed=17)
	restored_optimizer = torch.optim.AdamW(
		[
			{'params': restored_heads.parameters(), 'name': 'head', 'lr': 1.0e-3},
			{'params': restored_student.parameters(), 'name': 'encoder', 'lr': 1.0e-4},
			{
				'params': restored_replacement.parameters(),
				'name': 'spatial_context',
				'lr': 1.0e-3,
			},
		],
		weight_decay=0.05,
	)
	with pytest.raises(
		ValueError,
		match='schema-7 resume requires the spatial_context replacement-token module',
	):
		restore_strat_hmm_training_checkpoint(
			payload=payload,
			student=restored_student,
			head=restored_heads,
			spatial_context=None,
			optimizer=restored_optimizer,
			scaler=None,
			amp_enabled=False,
			config=config,
		)
