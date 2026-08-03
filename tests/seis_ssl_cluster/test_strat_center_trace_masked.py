"""Focused contracts for center-trace masked strat-HMM pretraining."""

from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.models.mae import LearnedEncoderReplacementToken
from seis_ssl_cluster.stratigraphy import (
	MultiResolutionOrderedPrototypeHeads,
	MultiResolutionOrderedPrototypeOutput,
	OrderedPrototypeOutput,
)
from seis_ssl_cluster.training.strat_hmm import (
	build_strat_hmm_center_trace_masked_components,
	build_strat_hmm_multi_head_components,
	compute_strat_hmm_center_trace_masked_losses,
	train_strat_hmm_center_trace_masked_one_epoch,
)
from seis_ssl_cluster.training.strat_hmm import components as components_module
from seis_ssl_cluster.training.strat_hmm.state import TrainabilitySummary


def test_center_trace_loss_separates_weighted_masked_and_visible_branches() -> None:
	logits = [torch.zeros((1, 4, k), requires_grad=True) for k in (6, 8, 10)]
	heads = _StaticHeads(logits)
	tokens = torch.tensor(
		[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
		requires_grad=True,
	)
	teacher = torch.tensor(
		[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
		requires_grad=True,
	)
	confidence = torch.tensor([[0.5, 1.0, 0.25, 0.75]])
	boundary_weight = torch.tensor([[0.4, 0.8, 0.6, 0.2]])
	batch = _target_batch(
		labels=torch.zeros((1, 4), dtype=torch.long),
		confidence=confidence,
		boundary_weight=boundary_weight,
	)
	replacement_mask = torch.tensor(
		[[[[True], [False]], [[False], [False]]]],
		dtype=torch.bool,
	)

	losses = compute_strat_hmm_center_trace_masked_losses(
		heads=heads,  # type: ignore[arg-type]
		encoded={'tokens': tokens},
		teacher_encoded={'tokens': teacher},
		batch=batch,
		replacement_mask=replacement_mask,
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.0,
			'distillation_weight': 0.2,
			'consistency_weight': 0.0,
		},
		pseudo_target_config={},
	)

	for k, head_logits in zip((6, 8, 10), logits, strict=True):
		per_token = torch.nn.functional.cross_entropy(
			head_logits.detach().reshape(4, k),
			torch.zeros(4, dtype=torch.long),
			reduction='none',
		)
		weights = confidence * boundary_weight
		expected_masked = per_token[0]
		expected_visible = (per_token[1:] * weights[0, 1:]).sum() / weights[0, 1:].sum()
		assert losses[f'loss_prototype_masked_k{k}'].item() == pytest.approx(
			expected_masked.item()
		)
		assert losses[f'loss_prototype_visible_k{k}'].item() == pytest.approx(
			expected_visible.item()
		)

	assert losses['loss_prototype_masked'].item() == pytest.approx(
		torch.stack([losses[f'loss_prototype_masked_k{k}'] for k in (6, 8, 10)])
		.mean()
		.item()
	)
	assert losses['loss_prototype_visible'].item() == pytest.approx(
		torch.stack([losses[f'loss_prototype_visible_k{k}'] for k in (6, 8, 10)])
		.mean()
		.item()
	)
	assert losses['loss_consistency_contribution'].item() == 0.0
	assert losses['loss_consistency_contribution'].requires_grad
	assert losses['masked_supervised_token_fraction'].item() == pytest.approx(0.25)
	assert losses['visible_supervised_token_fraction'].item() == pytest.approx(0.75)
	assert losses['valid_distillation_token_fraction'].item() == pytest.approx(0.75)

	losses['loss'].backward()
	assert teacher.grad is None


def test_center_trace_distillation_respects_min_confidence() -> None:
	logits = [torch.zeros((1, 4, k), requires_grad=True) for k in (6, 8, 10)]
	heads = _StaticHeads(logits)
	tokens = torch.tensor(
		[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
		requires_grad=True,
	)
	teacher = torch.tensor(
		[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
		requires_grad=True,
	)
	batch = _target_batch(
		labels=torch.zeros((1, 4), dtype=torch.long),
		confidence=torch.tensor([[0.5, 1.0, 0.25, 0.75]]),
		boundary_weight=torch.ones((1, 4)),
	)
	replacement_mask = torch.tensor(
		[[[[True], [False]], [[False], [False]]]],
		dtype=torch.bool,
	)

	losses = compute_strat_hmm_center_trace_masked_losses(
		heads=heads,  # type: ignore[arg-type]
		encoded={'tokens': tokens},
		teacher_encoded={'tokens': teacher},
		batch=batch,
		replacement_mask=replacement_mask,
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.0,
			'distillation_weight': 0.2,
			'consistency_weight': 0.0,
		},
		pseudo_target_config={'min_confidence': 0.5},
	)

	assert losses['visible_supervised_token_fraction'].item() == pytest.approx(0.5)
	assert losses['valid_distillation_token_fraction'].item() == pytest.approx(0.5)
	assert losses['loss_distillation'].item() == pytest.approx(0.5)


def test_center_trace_loss_rejects_consistency_and_empty_branch() -> None:
	heads = _StaticHeads(
		[torch.zeros((1, 4, k), requires_grad=True) for k in (6, 8, 10)]
	)
	kwargs = {
		'heads': heads,
		'encoded': {'tokens': torch.zeros((1, 4, 2), requires_grad=True)},
		'teacher_encoded': None,
		'batch': _target_batch(
			labels=torch.zeros((1, 4), dtype=torch.long),
			confidence=torch.ones((1, 4)),
			boundary_weight=torch.ones((1, 4)),
		),
		'replacement_mask': torch.tensor(
			[[[[True], [False]], [[False], [False]]]],
			dtype=torch.bool,
		),
		'pseudo_target_config': {},
	}
	with pytest.raises(ValueError, match='consistency_weight'):
		compute_strat_hmm_center_trace_masked_losses(
			**kwargs,
			loss_config={'consistency_weight': 0.1},
		)

	empty_branch_kwargs = dict(kwargs)
	empty_branch_kwargs['replacement_mask'] = torch.zeros(
		(1, 2, 2, 1), dtype=torch.bool
	)
	with pytest.raises(ValueError, match='masked branch'):
		compute_strat_hmm_center_trace_masked_losses(
			**empty_branch_kwargs,
			loss_config={'consistency_weight': 0.0},
		)


def test_center_trace_builder_preserves_hard_multi_head_initialization_parity(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path,
) -> None:
	def build_student_teacher(*_args, **_kwargs):
		student = _BuilderStudent()
		return (
			student,
			None,
			{'paths': {'output_root': 'unused'}, 'data': {}, 'zero_mask': {}},
			TrainabilitySummary(
				trainable_parameter_count=sum(
					parameter.numel() for parameter in student.parameters()
				),
				frozen_parameter_count=0,
				trainable_names=tuple(name for name, _ in student.named_parameters()),
			),
		)

	monkeypatch.setattr(
		components_module,
		'_build_student_teacher',
		build_student_teacher,
	)
	config = {
		'paths': {'output_root': str(tmp_path)},
		'data': {},
		'zero_mask': {},
		'head': {
			'spec': 'multi_resolution_ordered_prototypes_v1',
			'ks': [6, 8, 10],
			'projection_dim': 3,
			'temperature': 0.1,
			'normalize': True,
		},
		'loss': {'distillation_weight': 0.2, 'consistency_weight': 0.0},
		'train': {
			'seed': 273,
			'lr': 3.0e-4,
			'encoder_lr': 1.0e-5,
			'weight_decay': 0.05,
		},
	}
	torch.manual_seed(273)
	baseline = build_strat_hmm_multi_head_components(config, device='cpu')
	baseline_rng_state = torch.get_rng_state()
	torch.manual_seed(273)
	center = build_strat_hmm_center_trace_masked_components(
		config,
		device='cpu',
		replacement_token_seed=5,
	)

	assert all(
		torch.equal(value, center.student.state_dict()[key])
		for key, value in baseline.student.state_dict().items()
	)
	assert all(
		torch.equal(value, center.heads.state_dict()[key])
		for key, value in baseline.heads.state_dict().items()
	)
	assert torch.equal(torch.get_rng_state(), baseline_rng_state)
	assert center.trainability_summary == baseline.trainability_summary
	assert [group['name'] for group in center.optimizer.param_groups] == [
		'head',
		'encoder',
		'spatial_context',
	]
	assert [group['lr'] for group in center.optimizer.param_groups] == [
		3.0e-4,
		1.0e-5,
		3.0e-4,
	]
	group_ids = [
		id(parameter)
		for group in center.optimizer.param_groups
		for parameter in group['params']
	]
	assert len(group_ids) == len(set(group_ids))
	assert {
		id(parameter)
		for module in (center.student, center.heads, center.replacement_token)
		for parameter in module.parameters()
		if parameter.requires_grad
	} == set(group_ids)


def test_center_trace_cpu_one_step_masks_student_and_trains_replacement_token() -> None:
	student = _RuntimeStudent()
	teacher = _RuntimeStudent()
	teacher.requires_grad_(requires_grad=False)
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=4,
		ks=(6, 8, 10),
		projection_dim=3,
		temperature=0.5,
		normalize=True,
	)
	replacement_token = LearnedEncoderReplacementToken(4, seed=5)
	optimizer = torch.optim.AdamW(
		[
			{'params': heads.parameters(), 'lr': 3.0e-4, 'name': 'head'},
			{'params': student.parameters(), 'lr': 1.0e-5, 'name': 'encoder'},
			{
				'params': replacement_token.parameters(),
				'lr': 3.0e-4,
				'name': 'spatial_context',
			},
		],
	)
	batch = _runtime_batch()
	teacher_before = {
		name: parameter.detach().clone()
		for name, parameter in teacher.named_parameters()
	}

	state = train_strat_hmm_center_trace_masked_one_epoch(
		student=student,
		teacher=teacher,
		heads=heads,
		replacement_token=replacement_token,
		dataloader=[batch],
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		training_seed=273,
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'distillation_weight': 0.2,
			'consistency_weight': 0.0,
		},
		pseudo_target_config={},
		grad_clip_norm=0.1,
	)

	assert state.global_step == 1
	assert state.metrics['eligible_xy_column_count'] == pytest.approx(4.0)
	assert state.metrics['selected_xy_column_count'] == pytest.approx(1.0)
	assert torch.isfinite(torch.tensor(tuple(state.metrics.values()))).all()
	assert student.masked_calls == 1
	assert teacher.unmasked_calls == 1
	assert replacement_token.replacement_token.grad is not None
	assert torch.isfinite(replacement_token.replacement_token.grad).all()
	assert all(
		parameter.grad is not None and torch.isfinite(parameter.grad).all()
		for parameter in heads.parameters()
	)
	assert all(
		torch.equal(teacher_before[name], parameter)
		for name, parameter in teacher.named_parameters()
	)


class _StaticHeads(torch.nn.Module):
	def __init__(self, logits: list[torch.Tensor]) -> None:
		super().__init__()
		self.head_ks = (6, 8, 10)
		self._logits = torch.nn.ParameterList(
			[torch.nn.Parameter(value) for value in logits]
		)

	def forward(self, _tokens: torch.Tensor) -> MultiResolutionOrderedPrototypeOutput:
		return MultiResolutionOrderedPrototypeOutput(
			outputs={
				f'k{k}': OrderedPrototypeOutput(
					logits=logits,
					projected_features=logits[..., :1],
				)
				for k, logits in zip(self.head_ks, self._logits, strict=True)
			},
			head_ks=self.head_ks,
		)


class _BuilderStudent(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder_dim = 4
		self.encoder = torch.nn.Module()
		self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])


class _RuntimeStudent(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder = torch.nn.Module()
		self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(1, 4)])
		self.patch_size_xyz = (1, 1, 1)
		self.masked_calls = 0
		self.unmasked_calls = 0

	def encode_tokens(
		self,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor,
		replacement_mask: torch.Tensor | None = None,
		replacement_token: torch.Tensor | None = None,
	) -> dict[str, torch.Tensor | tuple[int, int, int]]:
		if replacement_mask is None:
			self.unmasked_calls += 1
		else:
			self.masked_calls += 1
			assert replacement_token is not None
		features = self.encoder.layers[0](x.reshape(x.shape[0], -1, 1))
		if replacement_mask is not None:
			features = torch.where(
				replacement_mask.reshape(x.shape[0], -1).unsqueeze(-1),
				replacement_token.reshape(1, 1, -1),
				features,
			)
		return {
			'tokens': features,
			'token_grid_shape': (2, 2, 1),
			'token_valid_mask': valid_mask.reshape(x.shape[0], -1),
		}


def _target_batch(
	*,
	labels: torch.Tensor,
	confidence: torch.Tensor,
	boundary_weight: torch.Tensor,
) -> dict[str, object]:
	return {
		'strat_multi_targets': {
			f'k{k}': {
				'labels': labels.clone(),
				'confidence': confidence.clone(),
				'boundary_weight': boundary_weight.clone(),
				'valid_mask': torch.ones_like(labels, dtype=torch.bool),
			}
			for k in (6, 8, 10)
		}
	}


def _runtime_batch() -> dict[str, object]:
	labels = torch.zeros((1, 2, 2, 1), dtype=torch.long)
	targets = _target_batch(
		labels=labels,
		confidence=torch.ones_like(labels, dtype=torch.float32),
		boundary_weight=torch.ones_like(labels, dtype=torch.float32),
	)
	return {
		'x': torch.randn((1, 1, 2, 2, 1)),
		'local_valid_mask': torch.ones((1, 2, 2, 1), dtype=torch.bool),
		**targets,
	}
