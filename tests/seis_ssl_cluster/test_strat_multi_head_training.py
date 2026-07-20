"""Shared-forward training coverage for multi-head strat HMM training."""

from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.stratigraphy import MultiResolutionOrderedPrototypeHeads
from seis_ssl_cluster.training.strat_hmm import (
	runner,
	train_strat_hmm_multi_head_one_epoch,
)
from seis_ssl_cluster.training.strat_hmm.state import (
	StratHmmMultiHeadComponents,
	StratHmmTrainingState,
	TrainabilitySummary,
)


class _Student(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder = torch.nn.Module()
		self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(1, 3)])
		self.calls = 0

	def encode_tokens(
		self,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor,
	) -> dict[str, torch.Tensor]:
		del valid_mask
		self.calls += 1
		features = self.encoder.layers[0](x.mean(dim=(2, 3, 4)))
		return {'tokens': features.unsqueeze(1).expand(-1, 4, -1)}


def test_multi_head_epoch_uses_one_student_and_teacher_forward_per_batch() -> None:
	torch.manual_seed(273)
	student = _Student()
	teacher = _Student()
	teacher.requires_grad_(requires_grad=False)
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	optimizer = torch.optim.AdamW(
		[
			{'params': heads.parameters(), 'lr': 3.0e-4, 'name': 'head'},
			{'params': student.parameters(), 'lr': 1.0e-5, 'name': 'encoder'},
		],
		weight_decay=0.05,
	)
	frozen_teacher = {
		name: parameter.detach().clone()
		for name, parameter in teacher.named_parameters()
	}
	before_student = {
		name: parameter.detach().clone()
		for name, parameter in student.named_parameters()
	}

	state = train_strat_hmm_multi_head_one_epoch(
		student=student,
		teacher=teacher,
		heads=heads,
		dataloader=[_batch()],
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'distillation_weight': 0.2,
			'consistency_weight': 0.1,
			'consistency_beta': 0.1,
		},
		pseudo_target_config={'min_confidence': 0.0},
	)

	assert student.calls == 1
	assert teacher.calls == 1
	assert state.global_step == 1
	assert torch.isfinite(torch.tensor(tuple(state.metrics.values()))).all()
	assert all(
		parameter.grad is not None and torch.isfinite(parameter.grad).all()
		for parameter in heads.parameters()
	)
	assert any(
		not torch.equal(before_student[name], parameter)
		for name, parameter in student.named_parameters()
	)
	assert all(
		torch.equal(frozen_teacher[name], parameter)
		for name, parameter in teacher.named_parameters()
	)


@pytest.mark.parametrize('consistency_weight', [0.0, 0.1])
def test_multi_head_cpu_two_step_smoke_is_finite(
	consistency_weight: float,
) -> None:
	student = _Student()
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	optimizer = torch.optim.AdamW(
		[*heads.parameters(), *student.parameters()],
		lr=3.0e-4,
	)

	state = train_strat_hmm_multi_head_one_epoch(
		student=student,
		heads=heads,
		dataloader=[_batch(), _batch()],
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'consistency_weight': consistency_weight,
			'consistency_beta': 0.1,
		},
		pseudo_target_config={'min_confidence': 0.0},
	)

	assert student.calls == 2
	assert state.global_step == 2
	assert torch.isfinite(torch.tensor(tuple(state.metrics.values()))).all()


def test_multi_head_epoch_rejects_nonfinite_student_tokens() -> None:
	student = _Student()
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	batch = _batch()
	batch['x'] = torch.full((1, 1, 2, 2, 2), float('nan'))

	with pytest.raises(FloatingPointError, match='non-finite student encoded tokens'):
		train_strat_hmm_multi_head_one_epoch(
			student=student,
			heads=heads,
			dataloader=[batch],
			optimizer=torch.optim.AdamW(
				[*heads.parameters(), *student.parameters()],
				lr=3.0e-4,
			),
			device=torch.device('cpu'),
			epoch=1,
			loss_config={'prototype_weight': 1.0},
			pseudo_target_config={'min_confidence': 0.0},
		)


def test_multi_head_epoch_rejects_nonfinite_post_clip_gradient_norm(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	student = _Student()
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)

	def corrupt_gradients(
		parameters: object,
		max_norm: float,
	) -> torch.Tensor:
		del max_norm
		for parameter in parameters:  # type: ignore[union-attr]
			if parameter.grad is not None:
				parameter.grad.fill_(float('nan'))
		return torch.tensor(1.0)

	monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', corrupt_gradients)

	with pytest.raises(
		FloatingPointError,
		match='non-finite strat HMM pretext gradient norm',
	):
		train_strat_hmm_multi_head_one_epoch(
			student=student,
			heads=heads,
			dataloader=[_batch()],
			optimizer=torch.optim.AdamW(
				[*heads.parameters(), *student.parameters()],
				lr=3.0e-4,
			),
			device=torch.device('cpu'),
			epoch=1,
			loss_config={'prototype_weight': 1.0},
			pseudo_target_config={'min_confidence': 0.0},
			grad_clip_norm=1.0,
		)


def test_multi_head_runner_never_calls_rolling_checkpoint_paths(
	monkeypatch,
	tmp_path,
) -> None:
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	components = StratHmmMultiHeadComponents(
		student=torch.nn.Linear(1, 1),
		teacher=None,
		heads=heads,
		optimizer=torch.optim.AdamW(heads.parameters()),
		mae_checkpoint_config={},
		trainability_summary=TrainabilitySummary(0, 0, ()),
		head_spec='multi_resolution_ordered_prototypes_v1',
		head_ks=(6, 8, 10),
	)
	checkpoint_calls: list[str] = []
	monkeypatch.setattr(runner, 'read_manifest_json', lambda _path: [])
	monkeypatch.setattr(
		runner,
		'NopimsStratMultiHeadTargetDataset',
		lambda *_args, **_kwargs: [],
	)
	monkeypatch.setattr(
		runner,
		'build_strat_multi_head_target_dataloader',
		lambda *_args, **_kwargs: [object()],
	)
	monkeypatch.setattr(
		runner,
		'build_strat_hmm_components',
		lambda *_args, **_kwargs: components,
	)
	monkeypatch.setattr(runner, '_strat_hmm_control_identity', lambda _config: None)
	monkeypatch.setattr(runner, '_snapshot_run_inputs', lambda **_kwargs: None)
	monkeypatch.setattr(runner, '_write_run_metadata', lambda **_kwargs: None)
	monkeypatch.setattr(runner, 'prepare_run_directory', lambda **_kwargs: None)
	monkeypatch.setattr(
		runner,
		'train_strat_hmm_multi_head_one_epoch',
		lambda **_kwargs: StratHmmTrainingState(
			epoch=1,
			global_step=1,
			metrics={'loss': 1.0},
			last_batch_index=0,
			completed_epoch=True,
		),
	)
	monkeypatch.setattr(
		runner,
		'save_strat_hmm_rolling_checkpoint',
		lambda *_args, **_kwargs: checkpoint_calls.append('save'),
	)
	monkeypatch.setattr(
		runner,
		'restore_strat_hmm_training_checkpoint',
		lambda **_kwargs: checkpoint_calls.append('restore'),
	)

	result = runner.run_strat_hmm_pretext_training(_runner_config(tmp_path))

	assert result == tmp_path
	assert checkpoint_calls == []


def _batch() -> dict[str, object]:
	return {
		'x': torch.ones((1, 1, 2, 2, 2)),
		'local_valid_mask': torch.ones((1, 2, 2, 2), dtype=torch.bool),
		'strat_multi_targets': {
			f'k{k}': {
				'labels': torch.tensor([[0, 1, 2, 3]]),
				'confidence': torch.ones((1, 4)),
				'boundary_weight': torch.ones((1, 4)),
				'valid_mask': torch.ones((1, 4), dtype=torch.bool),
			}
			for k in (6, 8, 10)
		},
	}


def _runner_config(tmp_path) -> dict[str, object]:
	return {
		'paths': {'output_root': str(tmp_path)},
		'manifests': {'train': str(tmp_path / 'train.json')},
		'head': {'spec': 'multi_resolution_ordered_prototypes_v1'},
		'pseudo_targets': {'manifest': str(tmp_path / 'targets.json')},
		'data': {'local_crop_size': [2, 2, 2]},
		'model': {'patch_size': [1, 1, 1]},
		'zero_mask': {},
		'loss': {},
		'train': {
			'device': 'cpu',
			'seed': 273,
			'epochs': 1,
			'batch_size': 1,
		},
	}
