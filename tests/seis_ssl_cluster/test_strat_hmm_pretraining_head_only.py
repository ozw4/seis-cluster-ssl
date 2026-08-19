from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import seis_ssl_cluster.training.strat_hmm.runner as strat_hmm_runner
from seis_ssl_cluster.config.pretraining import (
	resolve_barlow_twins_training_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	NopimsStratPseudoTargetDataset,
	SurveyManifest,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	read_manifest_json,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.stratigraphy import (
	OrderedPrototypeHead,
	discover_pseudo_target_inputs,
	write_pseudo_target,
)
from seis_ssl_cluster.training import load_checkpoint, save_checkpoint
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	save_barlow_twins_checkpoint,
)
from seis_ssl_cluster.training.dataloaders import build_strat_pseudo_target_dataloader
from seis_ssl_cluster.training.strat_hmm import compute_strat_hmm_pretext_losses
from seis_ssl_cluster.training.strat_hmm import (
	run_strat_hmm_pretext_training as run_strat_hmm_pretext_training_new,
)
from seis_ssl_cluster.training.strat_hmm_pretraining import (
	build_strat_hmm_head_only_components,
	configure_student_trainability,
	run_strat_hmm_pretext_training,
	train_strat_hmm_head_only_one_epoch,
)


def test_strat_hmm_pretraining_legacy_import_path_is_supported() -> None:
	assert run_strat_hmm_pretext_training is run_strat_hmm_pretext_training_new


def test_head_only_training_runs_cpu_writes_checkpoints_and_payloads(
	tmp_path: Path,
) -> None:
	config = _resolved_config(tmp_path, max_steps=2)

	checkpoint_path = run_strat_hmm_pretext_training(config)

	assert checkpoint_path == Path(config['paths']['output_root']) / 'latest.pt'
	assert checkpoint_path.is_file()
	assert (Path(config['paths']['output_root']) / 'best.pt').is_file()
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert set(payload) == {
		'amp_enabled',
		'config',
		'epoch',
		'global_step',
		'metrics',
		'model_state_dict',
		'optimizer_state_dict',
		'package_version',
		'rng_state',
		'scaler_state_dict',
		'stratigraphy_config',
		'stratigraphy_state_dict',
		'trainability_summary',
		'training_state',
	}
	assert payload['global_step'] == 2
	assert payload['training_state']['stage'] == 'train_strat_hmm_pretext'
	assert payload['config']['stage'] == 'train_amp_mae'
	assert payload['stratigraphy_config']['stage'] == 'train_strat_hmm_pretext'
	assert math.isfinite(payload['metrics']['loss'])
	assert math.isfinite(payload['metrics']['loss_prototype'])
	assert math.isfinite(payload['metrics']['loss_usage'])
	assert payload['metrics']['trainable_parameter_count'] == pytest.approx(0.0)
	assert payload['trainability_summary']['trainable_names'] == []
	assert set(payload['metrics']) == {
		'amp_enabled',
		'frozen_parameter_count',
		'loss',
		'loss_distillation',
		'loss_prototype',
		'loss_usage',
		'mean_boundary_weight_valid',
		'mean_effective_prototype_weight',
		'positive_effective_weight_fraction',
		'prototype_usage_entropy',
		'target_usage_entropy',
		'trainable_parameter_count',
		'valid_distillation_token_fraction',
		'valid_supervised_token_fraction',
	}
	model_keys = set(payload['model_state_dict'])
	head_keys = set(payload['stratigraphy_state_dict'])
	assert 'patch_projection.weight' in model_keys
	assert 'prediction_head.weight' in model_keys
	assert not (model_keys & {'prototypes', 'projection.weight', 'projection.bias'})
	assert {'prototypes', 'projection.weight', 'projection.bias'} <= head_keys


def test_explicit_control_identity_is_persisted_with_initial_parameter_hashes(
	tmp_path: Path,
) -> None:
	raw = _raw_config(
		tmp_path,
		max_steps=1,
		encoder_depth=1,
		unfreeze_top_blocks=1,
		distillation_weight=0.2,
	)
	raw['identity'] = {
		'model_tag': 'current-k6-control-fixture',
		'scientific_identity': {'pretext': 'single-head-k6'},
		'runtime_identity': {'finite_check_mode_reason': 'fixture'},
	}
	config = resolve_strat_hmm_pretext_config(raw)

	checkpoint_path = run_strat_hmm_pretext_training(config)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	identity = payload['control_identity']
	assert identity['model_tag'] == 'current-k6-control-fixture'
	assert identity['scientific_identity'] == {'pretext': 'single-head-k6'}
	assert identity['runtime_identity']['finite_check_mode'] == 'strict'
	assert identity['input_identities']['teacher_checkpoint']['sha256']
	assert identity['input_identities']['student_init_checkpoint']['sha256']
	assert len(identity['input_identities']['pseudo_targets']) == 1
	assert identity['initial_parameter_sha256']['student_trainable']
	assert identity['initial_parameter_sha256']['prototype_head']
	groups = payload['optimizer_state_dict']['param_groups']
	assert [(group['name'], group['lr']) for group in groups] == [
		('head', pytest.approx(1.0e-2)),
		('encoder', pytest.approx(1.0e-3)),
	]


@pytest.mark.parametrize(
	'identity',
	[
		{},
		{'model_tag': ''},
		{'model_tag': 'fixture', 'unexpected': True},
		{'model_tag': 'fixture', 'scientific_identity': 'not-a-mapping'},
	],
)
def test_control_identity_schema_rejects_invalid_values(
	tmp_path: Path, identity: dict[str, object]
) -> None:
	raw = _raw_config(tmp_path)
	raw['identity'] = identity

	with pytest.raises((TypeError, ValueError)):
		resolve_strat_hmm_pretext_config(raw)


def test_training_runner_keeps_dataset_import_and_batch_schema(tmp_path: Path) -> None:
	assert (
		strat_hmm_runner.NopimsStratPseudoTargetDataset
		is NopimsStratPseudoTargetDataset
	)

	batch = next(iter(_single_batch_dataloader(_resolved_config(tmp_path))))

	assert '_token_valid_mask' not in batch


def test_head_only_components_freeze_student_and_train_head(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path)

	components = build_strat_hmm_head_only_components(
		config,
		device=torch.device('cpu'),
	)

	assert all(
		not parameter.requires_grad for parameter in components.student.parameters()
	)
	assert all(parameter.requires_grad for parameter in components.head.parameters())
	head_parameter_ids = {
		id(parameter) for parameter in components.head.parameters()
	}
	optimizer_parameter_ids = {
		id(parameter)
		for group in components.optimizer.param_groups
		for parameter in group['params']
	}
	assert optimizer_parameter_ids == head_parameter_ids


def test_components_reject_unfreeze_without_distillation(tmp_path: Path) -> None:
	config = _raw_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.0,
	)

	with pytest.raises(ValueError, match=r'loss\.distillation_weight'):
		build_strat_hmm_head_only_components(config, device=torch.device('cpu'))


def test_configure_student_trainability_unfreezes_only_top_blocks() -> None:
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=3,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)

	summary = configure_student_trainability(model, unfreeze_top_blocks=0)

	assert summary.trainable_parameter_count == 0
	assert summary.trainable_names == ()
	assert all(not parameter.requires_grad for parameter in model.parameters())

	summary = configure_student_trainability(model, unfreeze_top_blocks=2)

	assert summary.trainable_parameter_count > 0
	assert summary.trainable_names
	assert all(
		name.startswith(('encoder.layers.1.', 'encoder.layers.2.'))
		for name in summary.trainable_names
	)
	assert all(
		not parameter.requires_grad
		for name, parameter in model.named_parameters()
		if not name.startswith(('encoder.layers.1.', 'encoder.layers.2.'))
	)

	with pytest.raises(ValueError, match='unfreeze_top_blocks'):
		configure_student_trainability(model, unfreeze_top_blocks=4)


def test_unfreeze_top_block_optimizer_lrs_and_gradients(tmp_path: Path) -> None:
	config = _resolved_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.25,
	)
	components = build_strat_hmm_head_only_components(
		config,
		device=torch.device('cpu'),
	)
	dataloader = _single_batch_dataloader(config)

	state = train_strat_hmm_head_only_one_epoch(
		student=components.student,
		teacher=components.teacher,
		head=components.head,
		dataloader=dataloader,
		optimizer=components.optimizer,
		device=torch.device('cpu'),
		epoch=1,
		loss_config=config['loss'],
		pseudo_target_config=config['pseudo_targets'],
		max_steps=1,
	)

	assert state.global_step == 1
	assert state.metrics['loss_distillation'] >= -1.0e-6
	assert [group['lr'] for group in components.optimizer.param_groups] == [
		pytest.approx(config['train']['lr']),
		pytest.approx(config['train']['encoder_lr']),
	]
	assert all(
		name.startswith('encoder.layers.1.')
		for name in components.trainability_summary.trainable_names
	)
	trainable_parameter_ids = {
		id(parameter)
		for module in (components.student, components.head)
		for parameter in module.parameters()
		if parameter.requires_grad
	}
	optimizer_parameter_ids = {
		id(parameter)
		for group in components.optimizer.param_groups
		for parameter in group['params']
	}
	assert optimizer_parameter_ids == trainable_parameter_ids
	bottom_grads = [
		parameter.grad
		for parameter in components.student.encoder.layers[0].parameters()
	]
	top_grads = [
		parameter.grad
		for parameter in components.student.encoder.layers[-1].parameters()
	]
	assert all(grad is None for grad in bottom_grads)
	assert any(
		grad is not None and bool(grad.abs().sum().gt(0).item())
		for grad in top_grads
	)
	assert components.teacher is not None
	assert all(parameter.grad is None for parameter in components.teacher.parameters())


@pytest.mark.parametrize('base_method', ['mae', 'barlow_twins'])
def test_k6_components_share_mae_and_barlow_backbone_contract(  # noqa: PLR0915
	tmp_path: Path,
	base_method: str,
) -> None:
	config, source_state, source_projector = _k6_component_fixture(
		tmp_path,
		base_method=base_method,
	)
	assert config['pseudo_targets'] == {
		'input_dir': str(tmp_path / 'pseudo_targets'),
		'k': 6,
		'min_confidence': 0.0,
	}
	assert 'spec' not in config['head']
	assert 'ks' not in config['head']

	components = build_strat_hmm_head_only_components(
		config,
		device=torch.device('cpu'),
	)
	expected_stage = (
		'train_amp_mae' if base_method == 'mae' else 'barlow_twins_training'
	)
	assert components.mae_checkpoint_config['stage'] == expected_stage
	assert components.teacher is not None
	_assert_tensor_state_equal(components.student.state_dict(), source_state)
	_assert_tensor_state_equal(components.teacher.state_dict(), source_state)
	assert components.teacher.training is False
	assert all(
		not parameter.requires_grad for parameter in components.teacher.parameters()
	)
	assert all(
		not parameter.requires_grad
		for parameter in components.student.patch_projection.parameters()
	)
	assert all(
		not parameter.requires_grad
		for parameter in components.student.encoder.layers[0].parameters()
	)
	assert all(
		parameter.requires_grad
		for parameter in components.student.encoder.layers[-1].parameters()
	)
	decoder_parameters = [
		parameter
		for name, parameter in components.student.named_parameters()
		if not name.startswith(('patch_projection.', 'encoder.'))
	]
	assert decoder_parameters
	assert all(not parameter.requires_grad for parameter in decoder_parameters)

	trainable_names = tuple(
		name
		for name, parameter in components.student.named_parameters()
		if parameter.requires_grad
	)
	assert trainable_names == components.trainability_summary.trainable_names
	assert all(name.startswith('encoder.layers.1.') for name in trainable_names)
	assert components.trainability_summary.trainable_parameter_count == sum(
		parameter.numel()
		for parameter in components.student.parameters()
		if parameter.requires_grad
	)
	assert components.trainability_summary.frozen_parameter_count == sum(
		parameter.numel()
		for parameter in components.student.parameters()
		if not parameter.requires_grad
	)

	assert isinstance(components.head, OrderedPrototypeHead)
	assert components.head.num_prototypes == 6
	assert components.head.projection_dim == 128
	assert components.head.temperature == pytest.approx(0.1)
	assert components.head.normalize is True
	assert all(parameter.requires_grad for parameter in components.head.parameters())
	assert isinstance(components.optimizer, torch.optim.AdamW)
	assert components.optimizer.state == {}
	assert components.optimizer.defaults['weight_decay'] == pytest.approx(0.05)
	assert [group['name'] for group in components.optimizer.param_groups] == [
		'head',
		'encoder',
	]
	assert [group['lr'] for group in components.optimizer.param_groups] == [
		pytest.approx(1.0e-5),
		pytest.approx(1.0e-5),
	]
	optimizer_parameters = [
		parameter
		for group in components.optimizer.param_groups
		for parameter in group['params']
	]
	expected_parameters = [
		*components.head.parameters(),
		*components.student.encoder.layers[-1].parameters(),
	]
	assert len({id(parameter) for parameter in optimizer_parameters}) == len(
		optimizer_parameters
	)
	assert {id(parameter) for parameter in optimizer_parameters} == {
		id(parameter) for parameter in expected_parameters
	}
	if source_projector is not None:
		projector_parameter_ids = {
			id(parameter) for parameter in source_projector.parameters()
		}
		component_parameter_ids = {
			id(parameter)
			for module in (components.student, components.teacher, components.head)
			for parameter in module.parameters()
		}
		assert projector_parameter_ids.isdisjoint(component_parameter_ids)
		assert projector_parameter_ids.isdisjoint(
			{id(parameter) for parameter in optimizer_parameters}
		)
		assert not any(
			isinstance(module, BarlowTwins3D)
			for root in (components.student, components.teacher, components.head)
			for module in root.modules()
		)

	student_before = _clone_tensor_state(components.student.state_dict())
	teacher_before = _clone_tensor_state(components.teacher.state_dict())
	head_before = _clone_tensor_state(components.head.state_dict())
	state = train_strat_hmm_head_only_one_epoch(
		student=components.student,
		teacher=components.teacher,
		head=components.head,
		dataloader=_single_batch_dataloader(config),
		optimizer=components.optimizer,
		device=torch.device('cpu'),
		epoch=1,
		loss_config=config['loss'],
		pseudo_target_config=config['pseudo_targets'],
		max_steps=1,
		grad_clip_norm=1.0,
	)

	assert state.global_step == 1
	for metric in ('loss', 'loss_prototype', 'loss_usage', 'loss_distillation'):
		assert math.isfinite(state.metrics[metric])
	gradients = [
		parameter.grad.detach()
		for parameter in expected_parameters
		if parameter.grad is not None
	]
	assert gradients
	gradient_norm = torch.linalg.vector_norm(
		torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients])
	)
	assert bool(torch.isfinite(gradient_norm).item())
	_assert_state_prefix_unchanged(
		student_before,
		components.student.state_dict(),
		'patch_projection.',
	)
	_assert_state_prefix_unchanged(
		student_before,
		components.student.state_dict(),
		'encoder.layers.0.',
	)
	_assert_decoder_state_unchanged(student_before, components.student.state_dict())
	_assert_tensor_state_equal(components.teacher.state_dict(), teacher_before)
	assert _state_prefix_changed(
		student_before,
		components.student.state_dict(),
		'encoder.layers.1.',
	)
	assert any(
		not torch.equal(head_before[name], value)
		for name, value in components.head.state_dict().items()
	)


def test_distillation_only_skips_prototype_head_and_invalid_labels() -> None:
	student_tokens = torch.randn(1, 3, 4, requires_grad=True)
	teacher_tokens = torch.randn(1, 3, 4)
	head = OrderedPrototypeHead(feature_dim=4, num_prototypes=3)
	losses = compute_strat_hmm_pretext_losses(
		head=head,
		encoded={
			'tokens': student_tokens,
			'token_valid_mask': torch.tensor([[True, True, False]]),
		},
		teacher_encoded={
			'tokens': teacher_tokens,
			'token_valid_mask': torch.tensor([[True, False, True]]),
		},
		batch={
			'strat_labels': torch.full((1, 3), 99),
			'strat_confidence': torch.ones(1, 3),
			'strat_boundary_weight': torch.zeros(1, 3),
			'strat_valid_mask': torch.ones(1, 3, dtype=torch.bool),
		},
		loss_config={
			'prototype_weight': 0.0,
			'usage_weight': 0.0,
			'distillation_weight': 0.2,
		},
		pseudo_target_config={'min_confidence': 0.0},
	)

	assert losses['loss_prototype'].item() == 0.0
	assert losses['loss_usage'].item() == 0.0
	assert losses['valid_distillation_token_fraction'].item() == pytest.approx(
		1.0 / 3.0,
	)
	assert losses['loss'].item() == pytest.approx(
		0.2 * losses['loss_distillation'].item(),
	)
	losses['loss'].backward()
	assert student_tokens.grad is not None
	assert all(parameter.grad is None for parameter in head.parameters())


def test_min_confidence_precedes_boundary_weight_and_reports_metrics() -> None:
	torch.manual_seed(7)
	tokens = torch.randn(1, 3, 4)
	head = OrderedPrototypeHead(feature_dim=4, num_prototypes=3)
	confidence = torch.tensor([[0.9, 0.4, 0.8]])
	boundary_weight = torch.tensor([[0.0, 1.0, 0.5]])
	losses = compute_strat_hmm_pretext_losses(
		head=head,
		encoded={'tokens': tokens},
		teacher_encoded=None,
		batch={
			'strat_labels': torch.tensor([[0, 1, 2]]),
			'strat_confidence': confidence,
			'strat_boundary_weight': boundary_weight,
			'strat_valid_mask': torch.ones(1, 3, dtype=torch.bool),
		},
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.0,
			'distillation_weight': 0.0,
		},
		pseudo_target_config={'min_confidence': 0.5},
	)
	expected = torch.nn.functional.cross_entropy(
		head(tokens).logits[:, 2],
		torch.tensor([2]),
	)

	assert torch.allclose(losses['loss_prototype'], expected)
	assert losses['valid_supervised_token_fraction'].item() == pytest.approx(2 / 3)
	assert losses['mean_boundary_weight_valid'].item() == pytest.approx(0.25)
	assert losses['mean_effective_prototype_weight'].item() == pytest.approx(0.2)
	assert losses['positive_effective_weight_fraction'].item() == pytest.approx(0.5)


def test_usage_and_distillation_do_not_depend_on_boundary_weight() -> None:
	torch.manual_seed(13)
	student_tokens = torch.randn(1, 3, 4)
	teacher_tokens = torch.randn(1, 3, 4)
	head = OrderedPrototypeHead(feature_dim=4, num_prototypes=3)
	common = {
		'head': head,
		'encoded': {'tokens': student_tokens},
		'teacher_encoded': {'tokens': teacher_tokens},
		'loss_config': {
			'prototype_weight': 0.0,
			'usage_weight': 0.2,
			'distillation_weight': 0.3,
			'entropy_floor': 0.5,
		},
		'pseudo_target_config': {'min_confidence': 0.0},
	}
	batch = {
		'strat_labels': torch.tensor([[0, 1, 2]]),
		'strat_confidence': torch.ones(1, 3),
		'strat_valid_mask': torch.ones(1, 3, dtype=torch.bool),
	}
	zero_boundary = compute_strat_hmm_pretext_losses(
		**common,
		batch={**batch, 'strat_boundary_weight': torch.zeros(1, 3)},
	)
	varied_boundary = compute_strat_hmm_pretext_losses(
		**common,
		batch={
			**batch,
			'strat_boundary_weight': torch.tensor([[0.1, 0.5, 1.0]]),
		},
	)

	assert zero_boundary['loss_prototype'].item() == 0.0
	assert torch.equal(zero_boundary['loss_usage'], varied_boundary['loss_usage'])
	assert torch.equal(
		zero_boundary['loss_distillation'],
		varied_boundary['loss_distillation'],
	)


def test_distillation_only_trains_top_block_and_omits_head_optimizer_group(
	tmp_path: Path,
) -> None:
	config = _resolved_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		prototype_weight=0.0,
		usage_weight=0.0,
		distillation_weight=0.2,
	)
	components = build_strat_hmm_head_only_components(
		config,
		device=torch.device('cpu'),
	)

	state = train_strat_hmm_head_only_one_epoch(
		student=components.student,
		teacher=components.teacher,
		head=components.head,
		dataloader=_single_batch_dataloader(config),
		optimizer=components.optimizer,
		device=torch.device('cpu'),
		epoch=1,
		loss_config=config['loss'],
		pseudo_target_config=config['pseudo_targets'],
		max_steps=1,
	)

	assert [group['name'] for group in components.optimizer.param_groups] == [
		'encoder',
	]
	assert state.metrics['loss'] == pytest.approx(
		config['loss']['distillation_weight']
		* state.metrics['loss_distillation'],
	)
	assert any(
		parameter.grad is not None
		and bool(parameter.grad.abs().sum().gt(0).item())
		for parameter in components.student.encoder.layers[-1].parameters()
	)
	assert all(parameter.grad is None for parameter in components.head.parameters())


def test_unfreeze_distillation_smoke_writes_checkpoint(tmp_path: Path) -> None:
	config = _resolved_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		prototype_weight=0.0,
		usage_weight=0.0,
		distillation_weight=0.2,
		max_steps=1,
	)

	checkpoint_path = run_strat_hmm_pretext_training(config)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['global_step'] == 1
	assert math.isfinite(payload['metrics']['loss_distillation'])
	assert payload['metrics']['loss_prototype'] == 0.0
	assert payload['metrics']['loss_usage'] == 0.0
	assert payload['metrics']['loss'] == pytest.approx(
		0.2 * payload['metrics']['loss_distillation'],
	)
	assert payload['metrics']['trainable_parameter_count'] > 0.0
	assert payload['metrics']['frozen_parameter_count'] > 0.0
	assert payload['trainability_summary']['trainable_names']


def test_invalid_pseudo_target_tokens_are_ignored(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path, invalid_first_token=True, max_steps=1)

	checkpoint_path = run_strat_hmm_pretext_training(config)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['metrics']['valid_supervised_token_fraction'] == pytest.approx(
		7.0 / 8.0,
	)
	assert math.isfinite(payload['metrics']['loss'])


def test_checkpoint_config_uses_strat_preprocessing_contract(
	tmp_path: Path,
) -> None:
	config = _resolved_config(tmp_path, max_steps=1)
	config['data']['normalized_clip_abs'] = 0.5
	config['data']['amplitude_agc'] = {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 3,
		'eps': 1.0e-6,
		'clip_abs': 2.0,
	}
	config['zero_mask'] = {
		'enabled': True,
		'zero_atol': 0.125,
		'z_sample_influence_radius': 0,
		'xy_trace_influence_radius': 0,
	}

	checkpoint_path = run_strat_hmm_pretext_training(config)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['config']['data']['normalized_clip_abs'] == 0.5
	assert payload['config']['data']['amplitude_agc'] == {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 3,
		'eps': 1.0e-6,
		'clip_abs': 2.0,
	}
	assert payload['config']['zero_mask'] == {
		'enabled': True,
		'zero_atol': 0.125,
		'z_sample_influence_radius': 0,
		'xy_trace_influence_radius': 0,
	}


def test_resume_advances_and_rejects_incompatible_head_config(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path, max_steps=1)

	first_checkpoint = run_strat_hmm_pretext_training(config)
	config['train']['max_steps'] = 2
	resumed_checkpoint = run_strat_hmm_pretext_training(
		config,
		resume=first_checkpoint,
	)

	payload = load_checkpoint(resumed_checkpoint, map_location='cpu')
	assert payload['global_step'] == 2

	incompatible = deepcopy(config)
	incompatible['head'] = dict(config['head'])
	incompatible['head']['temperature'] = 0.25
	with pytest.raises(ValueError, match=r'head\.temperature'):
		run_strat_hmm_pretext_training(incompatible, resume=resumed_checkpoint)


def test_resume_preserves_existing_best_score(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path, max_steps=1)

	first_checkpoint = run_strat_hmm_pretext_training(config)
	best_path = Path(config['paths']['output_root']) / 'best.pt'
	best_payload = load_checkpoint(best_path, map_location='cpu')
	best_payload['metrics']['loss'] = -1.0
	torch.save(best_payload, best_path)
	config['train']['max_steps'] = 2

	resumed_checkpoint = run_strat_hmm_pretext_training(
		config,
		resume=first_checkpoint,
	)

	latest_payload = load_checkpoint(resumed_checkpoint, map_location='cpu')
	resumed_best_payload = load_checkpoint(best_path, map_location='cpu')
	assert latest_payload['global_step'] == 2
	assert resumed_best_payload['metrics']['loss'] == -1.0


def test_cli_non_dry_run_executes_one_step(tmp_path: Path) -> None:
	raw_config = _raw_config(tmp_path, max_steps=None)
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(yaml.safe_dump(raw_config), encoding='utf-8')
	repo_root = Path(__file__).resolve().parents[2]
	script = repo_root / 'proc' / 'seis_ssl_cluster' / 'train_strat_hmm_pretext.py'
	env = os.environ.copy()
	src_path = str(repo_root / 'src')
	existing_pythonpath = env.get('PYTHONPATH')
	env['PYTHONPATH'] = (
		src_path
		if not existing_pythonpath
		else f'{src_path}{os.pathsep}{existing_pythonpath}'
	)

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			str(script),
			'--config',
			str(config_path),
			'--device',
			'cpu',
			'--max-steps',
			'1',
		],
		cwd=repo_root,
		env=env,
		check=True,
		capture_output=True,
		text=True,
	)

	checkpoint_path = Path(raw_config['paths']['output_root']) / 'latest.pt'
	assert f'checkpoint: {checkpoint_path}' in result.stdout
	assert checkpoint_path.is_file()
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['global_step'] == 1


def _resolved_config(  # noqa: PLR0913
	tmp_path: Path,
	*,
	invalid_first_token: bool = False,
	max_steps: int | None = None,
	encoder_depth: int = 1,
	unfreeze_top_blocks: int = 0,
	prototype_weight: float = 1.0,
	usage_weight: float = 0.01,
	distillation_weight: float = 0.0,
) -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(
		_raw_config(
			tmp_path,
			invalid_first_token=invalid_first_token,
			max_steps=max_steps,
			encoder_depth=encoder_depth,
			unfreeze_top_blocks=unfreeze_top_blocks,
			prototype_weight=prototype_weight,
			usage_weight=usage_weight,
			distillation_weight=distillation_weight,
		),
	)


def _raw_config(  # noqa: PLR0913
	tmp_path: Path,
	*,
	invalid_first_token: bool = False,
	max_steps: int | None = None,
	encoder_depth: int = 1,
	unfreeze_top_blocks: int = 0,
	prototype_weight: float = 1.0,
	usage_weight: float = 0.01,
	distillation_weight: float = 0.0,
) -> dict[str, object]:
	paths = _write_fixture_files(
		tmp_path,
		invalid_first_token=invalid_first_token,
		encoder_depth=encoder_depth,
	)
	artifact_root = tmp_path / 'artifacts'
	output_root = artifact_root / 'pretraining' / 'strat_hmm_head_only'
	config: dict[str, object] = {
		'paths': {
			'artifact_root': str(artifact_root),
			'output_root': str(output_root),
		},
		'manifests': {
			'train': str(paths['manifest']),
			'train_path_list': str(paths['path_list']),
		},
		'data': {
			'local_crop_size': [4, 4, 4],
			'min_valid_fraction': 0.0,
			'max_resample_attempts': 2,
		},
		'model': {
			'patch_size': [2, 2, 2],
			'encoder_dim': 12,
			'encoder_depth': encoder_depth,
			'encoder_heads': 3,
			'decoder_dim': 12,
			'decoder_depth': 1,
			'decoder_heads': 3,
		},
		'pseudo_targets': {
			'input_dir': str(paths['pseudo_root']),
			'k': 3,
			'min_confidence': 0.0,
		},
		'teacher': {'checkpoint': str(paths['teacher_checkpoint'])},
		'student': {'unfreeze_top_blocks': unfreeze_top_blocks},
		'head': {
			'num_prototypes': 3,
			'projection_dim': 6,
			'temperature': 0.5,
		},
		'loss': {
			'prototype_weight': prototype_weight,
			'usage_weight': usage_weight,
			'distillation_weight': distillation_weight,
			'entropy_floor': None,
		},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 2,
			'epochs': 1,
			'num_workers': 0,
			'shuffle': False,
			'lr': 1.0e-2,
			'encoder_lr': 1.0e-3,
			'weight_decay': 0.0,
			'amp': False,
			'device': 'cpu',
			'seed': 11,
			'grad_clip_norm': 1.0,
			'allow_overwrite_output': False,
		},
		'zero_mask': {
			'enabled': False,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 0,
			'xy_trace_influence_radius': 0,
		},
	}
	if max_steps is not None:
		config['train']['max_steps'] = max_steps
	return config


def _single_batch_dataloader(
	config: Mapping[str, object],
) -> torch.utils.data.DataLoader:
	manifests = read_manifest_json(Path(config['manifests']['train']))
	pseudo_inputs = discover_pseudo_target_inputs(
		Path(config['pseudo_targets']['input_dir']),
		k=int(config['pseudo_targets']['k']),
	)
	dataset = NopimsStratPseudoTargetDataset(
		manifests,
		pseudo_inputs,
		local_crop_size_xyz=tuple(config['data']['local_crop_size']),
		patch_size_xyz=tuple(config['model']['patch_size']),
		seed=int(config['train']['seed']),
		samples_per_epoch=1,
		zero_mask=ZeroMaskConfig(
			enabled=False,
			zero_atol=0.0,
			z_sample_influence_radius=0,
			xy_trace_influence_radius=0,
		),
		min_valid_fraction=0.0,
		max_resample_attempts=2,
	)
	return build_strat_pseudo_target_dataloader(
		dataset,
		batch_size=1,
		num_workers=0,
		shuffle=False,
		seed=int(config['train']['seed']),
		device='cpu',
	)


def _write_fixture_files(
	tmp_path: Path,
	*,
	invalid_first_token: bool,
	encoder_depth: int = 1,
) -> Mapping[str, Path]:
	volume = np.linspace(-1.0, 1.0, num=4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
	volume_path = tmp_path / 'survey' / 'amplitude.npy'
	volume_path.parent.mkdir(parents=True)
	np.save(volume_path, volume)
	stats_path = tmp_path / 'survey' / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1.0,
			clip_high=1.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	manifest = SurveyManifest(
		survey_id='survey',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=volume_path,
			shape_xyz=(4, 4, 4),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	manifest_path = tmp_path / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	path_list = tmp_path / 'train_path_list.txt'
	path_list.write_text(f'{volume_path}\n', encoding='utf-8')

	pseudo_root = tmp_path / 'pseudo_targets'
	labels = (np.arange(8, dtype=np.int32) % 3).reshape(2, 2, 2)
	valid_tokens = np.ones((2, 2, 2), dtype=np.bool_)
	confidence = np.ones((2, 2, 2), dtype=np.float32)
	if invalid_first_token:
		labels[0, 0, 0] = -1
		valid_tokens[0, 0, 0] = False
		confidence[0, 0, 0] = 0.0
	write_pseudo_target(
		pseudo_root,
		k=3,
		survey_id='survey',
		labels=labels,
		confidence=confidence,
		valid_tokens=valid_tokens,
	)
	teacher_checkpoint = tmp_path / 'teacher.pt'
	_write_teacher_checkpoint(
		teacher_checkpoint,
		manifest_path=manifest_path,
		path_list=path_list,
		encoder_depth=encoder_depth,
	)
	return {
		'manifest': manifest_path,
		'path_list': path_list,
		'pseudo_root': pseudo_root,
		'teacher_checkpoint': teacher_checkpoint,
	}


def _write_teacher_checkpoint(
	path: Path,
	*,
	manifest_path: Path,
	path_list: Path,
	encoder_depth: int = 1,
) -> None:
	torch.manual_seed(5)
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=encoder_depth,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)
	checkpoint_config = _teacher_checkpoint_config(
		manifest_path=manifest_path,
		path_list=path_list,
		encoder_depth=encoder_depth,
	)
	torch.save(
		{
			'model_state_dict': model.state_dict(),
			'config': checkpoint_config,
		},
		path,
	)


def _teacher_checkpoint_config(
	*,
	manifest_path: Path,
	path_list: Path,
	encoder_depth: int = 1,
) -> dict[str, object]:
	return deepcopy(
		{
			'stage': 'train_amp_mae',
			'paths': {'output_root': str(manifest_path.parent / 'teacher_run')},
			'manifests': {
				'train': str(manifest_path),
				'train_path_list': str(path_list),
			},
			'data': {
				'grid_order': list(GRID_ORDER_XYZ),
				'volume_format': 'npy_memmap',
				'input_channels': 1,
				'target_channels': 1,
				'use_context': False,
				'local_crop_size': [4, 4, 4],
				'min_valid_fraction': 0.0,
				'max_resample_attempts': 2,
				'amplitude_agc': {'enabled': False},
				'finite_check_mode': 'strict',
			},
			'model': {
				'name': 'amp_mae3d',
				'in_channels': 1,
				'out_channels': 1,
				'patch_size': [2, 2, 2],
				'encoder_dim': 12,
				'encoder_depth': encoder_depth,
				'encoder_heads': 3,
				'decoder_dim': 12,
				'decoder_depth': 1,
				'decoder_heads': 3,
			},
			'masking': {
				'spatial_mask_ratio': 0.5,
				'spatial_mask_mode': 'block',
				'block_size_tokens': [1, 1, 1],
			},
			'loss': {
				'reconstruction': 'huber',
				'huber_delta': 1.0,
				'gradient_weight': 0.0,
				'visible_reconstruction_weight': 0.0,
				'target_normalization': {'mode': 'none'},
				'valid_mask_mode': 'voxel',
			},
			'train': {
				'batch_size': 1,
				'samples_per_epoch': 1,
				'epochs': 1,
				'num_workers': 0,
				'shuffle': False,
				'lr': 1.0e-4,
				'weight_decay': 0.0,
				'amp': False,
				'device': 'cpu',
				'seed': 5,
				'grad_clip_norm': 1.0,
			},
			'zero_mask': {
				'enabled': False,
				'zero_atol': 0.0,
				'z_sample_influence_radius': 0,
				'xy_trace_influence_radius': 0,
			},
		},
	)


def _k6_component_fixture(
	tmp_path: Path,
	*,
	base_method: str,
) -> tuple[dict[str, object], dict[str, torch.Tensor], torch.nn.Module | None]:
	raw = _raw_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.2,
	)
	pseudo_root = tmp_path / 'pseudo_targets'
	write_pseudo_target(
		pseudo_root,
		k=6,
		survey_id='survey',
		labels=(np.arange(8, dtype=np.int32) % 6).reshape(2, 2, 2),
		confidence=np.ones((2, 2, 2), dtype=np.float32),
		valid_tokens=np.ones((2, 2, 2), dtype=np.bool_),
	)
	raw['pseudo_targets'] = {
		'input_dir': str(pseudo_root.resolve()),
		'k': 6,
		'min_confidence': 0.0,
	}
	raw['head'] = {
		'num_prototypes': 6,
		'projection_dim': 128,
		'temperature': 0.1,
		'normalize': True,
	}
	raw['loss'] = {
		'prototype_weight': 1.0,
		'usage_weight': 0.005,
		'entropy_floor': None,
		'distillation_weight': 0.2,
	}
	train = raw['train']
	assert isinstance(train, dict)
	train['lr'] = 1.0e-5
	train['encoder_lr'] = 1.0e-5
	train['weight_decay'] = 0.05
	train['amp'] = False
	train['grad_clip_norm'] = 1.0

	teacher = raw['teacher']
	assert isinstance(teacher, dict)
	checkpoint_path = Path(teacher['checkpoint'])
	initial_payload = load_checkpoint(checkpoint_path, map_location='cpu')
	checkpoint_config = initial_payload['config']
	model_state = initial_payload['model_state_dict']
	assert isinstance(checkpoint_config, dict)
	assert isinstance(model_state, dict)
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=2,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)
	model.load_state_dict(model_state, strict=True)
	projector: torch.nn.Module | None = None
	if base_method == 'mae':
		optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
		save_checkpoint(
			checkpoint_path,
			model=model,
			optimizer=optimizer,
			epoch=100,
			global_step=1,
			config=checkpoint_config,
			metrics={'loss': 0.5},
			amp_enabled=False,
			scaler=None,
			scaler_required=False,
			training_state={
				'schema_version': 2,
				'stage': 'train_amp_mae',
				'checkpoint_kind': 'epoch',
				'batch_index': None,
				'resolved_precision': 'float32',
			},
		)
	elif base_method == 'barlow_twins':
		barlow_config = _barlow_source_config(checkpoint_config, checkpoint_path)
		wrapper = BarlowTwins3D(model, projector_dim=8)
		optimizer = torch.optim.AdamW(wrapper.pretraining_parameters(), lr=1.0e-4)
		save_barlow_twins_checkpoint(
			checkpoint_path,
			backbone=wrapper.backbone,
			projector=wrapper.projector,
			optimizer=optimizer,
			epoch=100,
			global_step=1,
			config=barlow_config,
			metrics={'train_loss': 0.5},
			amp_enabled=False,
			scaler=None,
			scaler_required=False,
			dataset_epoch=100,
			completed_epoch=True,
		)
		projector = wrapper.projector
	else:
		raise ValueError(f'unsupported fixture base_method: {base_method!r}')

	raw['student'] = {
		'init_checkpoint': str(checkpoint_path),
		'unfreeze_top_blocks': 1,
	}
	resolved = resolve_strat_hmm_pretext_config(raw)
	source_payload = load_checkpoint(checkpoint_path, map_location='cpu')
	if base_method == 'barlow_twins':
		assert source_payload['config']['stage'] == 'barlow_twins_training'
		assert source_payload['pretraining_method'] == 'barlow_twins_3d'
		assert source_payload['checkpoint_kind'] == 'barlow_twins_pretraining'
		assert source_payload['trained_parameter_prefixes'] == [
			'patch_projection.',
			'encoder.',
		]
		assert isinstance(source_payload['projector_state_dict'], dict)
	return resolved, _clone_tensor_state(source_payload['model_state_dict']), projector


def _barlow_source_config(
	mae_config: Mapping[str, object],
	checkpoint_path: Path,
) -> dict[str, object]:
	manifests = mae_config['manifests']
	model = mae_config['model']
	assert isinstance(manifests, Mapping)
	assert isinstance(model, Mapping)
	return resolve_barlow_twins_training_config(
		{
			'paths': {
				'artifact_root': str(checkpoint_path.parent / 'barlow_artifacts'),
				'output_root': str(checkpoint_path.parent / 'barlow_artifacts' / 'run'),
			},
			'manifests': dict(manifests),
			'data': {'local_crop_size': [4, 4, 4]},
			'zero_mask': {'enabled': False},
			'model': {
				key: model[key]
				for key in (
					'patch_size',
					'encoder_dim',
					'encoder_depth',
					'encoder_heads',
					'decoder_dim',
					'decoder_depth',
					'decoder_heads',
				)
			},
			'barlow_twins': {'projector_dim': 8},
			'train': {
				'batch_size': 2,
				'samples_per_epoch': 2,
				'epochs': 100,
				'num_workers': 0,
				'shuffle': False,
				'lr': 1.0e-4,
				'weight_decay': 0.05,
				'amp': False,
				'device': 'cpu',
				'seed': 42,
				'grad_clip_norm': 1.0,
			},
		},
	)


def _clone_tensor_state(
	state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
	return {name: value.detach().clone() for name, value in state.items()}


def _assert_tensor_state_equal(
	actual: Mapping[str, torch.Tensor],
	expected: Mapping[str, torch.Tensor],
) -> None:
	assert set(actual) == set(expected)
	assert all(torch.equal(actual[name], expected[name]) for name in expected)


def _assert_state_prefix_unchanged(
	before: Mapping[str, torch.Tensor],
	after: Mapping[str, torch.Tensor],
	prefix: str,
) -> None:
	names = [name for name in before if name.startswith(prefix)]
	assert names
	assert all(torch.equal(before[name], after[name]) for name in names)


def _state_prefix_changed(
	before: Mapping[str, torch.Tensor],
	after: Mapping[str, torch.Tensor],
	prefix: str,
) -> bool:
	names = [name for name in before if name.startswith(prefix)]
	assert names
	return any(not torch.equal(before[name], after[name]) for name in names)


def _assert_decoder_state_unchanged(
	before: Mapping[str, torch.Tensor],
	after: Mapping[str, torch.Tensor],
) -> None:
	names = [
		name
		for name in before
		if not name.startswith(('patch_projection.', 'encoder.'))
	]
	assert names
	assert all(torch.equal(before[name], after[name]) for name in names)
