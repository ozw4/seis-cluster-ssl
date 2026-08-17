'''Tests for the regime-neutral Volve horizon training lifecycle.'''

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from seis_ssl_cluster.volve.horizon_runner import (
	HorizonRunnerSettings,
	HorizonRuntimeContext,
	backward_and_step_horizon_optimizer,
	run_horizon_training_job,
	validate_horizon_resume_runtime,
)

if TYPE_CHECKING:
	from pathlib import Path


class _OneItemHorizonDataset(Dataset[dict[str, Any]]):
	def __init__(self) -> None:
		target = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1) + 600.0
		self.item = {
			'target_sample_float': target,
			'supervision_mask': torch.ones(5, 1, 1, dtype=torch.bool),
			'primary_evaluation_mask': torch.ones(5, 1, 1, dtype=torch.bool),
			'output_valid_mask': torch.ones(1, 1, dtype=torch.bool),
		}

	def __len__(self) -> int:
		return 1

	def __getitem__(self, _index: int) -> dict[str, Any]:
		return self.item


class _ScalarHorizonModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.value = nn.Parameter(torch.tensor(1.0))


class _FiniteValueNonfiniteGradient(torch.autograd.Function):
	'''Return a finite value while injecting a non-finite backward gradient.'''

	@staticmethod
	def forward(_ctx: object, value: torch.Tensor) -> torch.Tensor:
		return value.detach().new_zeros(())

	@staticmethod
	def backward(_ctx: object, gradient: torch.Tensor) -> torch.Tensor:
		return torch.full_like(gradient, float('nan'))


def test_runner_lifecycle_uses_only_regime_callables(tmp_path: Path) -> None:
	callback_counts = {'build': 0, 'train': 0, 'predict': 0}

	def build_model_and_optimizer(
		device: torch.device,
	) -> tuple[nn.Module, torch.optim.Optimizer]:
		callback_counts['build'] += 1
		model = _ScalarHorizonModel().to(device)
		return model, torch.optim.SGD(model.parameters(), lr=0.1)

	def train_one_item(
		model: nn.Module,
		_item: dict[str, object],
		optimizer: torch.optim.Optimizer,
		runtime: HorizonRuntimeContext,
	) -> float:
		callback_counts['train'] += 1
		optimizer.zero_grad(set_to_none=True)
		parameter = next(model.parameters())
		loss = parameter.square()
		backward_and_step_horizon_optimizer(
			loss=loss,
			model=model,
			optimizer=optimizer,
			scaler=runtime.scaler,
			gradient_clip_norm=runtime.gradient_clip_norm,
		)
		return float(loss.detach())

	def predict_one_item(
		_model: nn.Module,
		item: dict[str, object],
		_runtime: HorizonRuntimeContext,
	) -> torch.Tensor:
		callback_counts['predict'] += 1
		target = item['target_sample_float']
		assert isinstance(target, torch.Tensor)
		return target.unsqueeze(0)

	dataset = _OneItemHorizonDataset()
	metrics_path = run_horizon_training_job(
		output_dir=tmp_path / 'run',
		run_identity={'benchmark': 'synthetic_runner_contract_v1'},
		settings=HorizonRunnerSettings(
			epochs=1,
			seed=42000,
			amp_on_cuda=True,
			gradient_clip_norm=1.0,
		),
		datasets={
			'train': dataset,
			'validation': dataset,
			'test': dataset,
		},
		expected_counts={
			'train': (1, 1, 1, 1, 1),
			'validation': (1, 1, 1, 1, 1),
			'test': (1, 1, 1, 1, 1),
			'test_primary': (1, 1, 1, 1, 1),
		},
		metrics_metadata={
			'schema_version': 1,
			'artifact_type': 'synthetic_horizon_runner_metrics',
		},
		build_model_and_optimizer=build_model_and_optimizer,
		train_one_item=train_one_item,
		predict_one_item=predict_one_item,
		device='cpu',
	)

	assert metrics_path == tmp_path / 'run/metrics.json'
	assert callback_counts == {'build': 1, 'train': 1, 'predict': 2}
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	assert metrics['test']['evaluation_pass_count'] == 1
	assert metrics['test']['primary_common']['macro_mae_samples'] == 0.0
	latest = torch.load(
		tmp_path / 'run/latest.pt', map_location='cpu', weights_only=False
	)
	assert latest['completed'] is True


def test_amp_resume_requires_grad_scaler_state() -> None:
	runtime_precision = {
		'device_type': 'cuda',
		'amp_enabled': True,
		'autocast_dtype': 'float16',
		'scaler_required': True,
	}
	with pytest.raises(ValueError, match='runtime precision'):
		validate_horizon_resume_runtime(
			{
				'runtime_precision': {
					'device_type': 'cpu',
					'amp_enabled': False,
					'autocast_dtype': None,
					'scaler_required': False,
				},
				'scaler_state_dict': None,
			},
			expected=runtime_precision,
			scaler=object(),  # type: ignore[arg-type]
		)
	with pytest.raises(ValueError, match='missing required GradScaler state'):
		validate_horizon_resume_runtime(
			{
				'runtime_precision': runtime_precision,
				'scaler_state_dict': None,
			},
			expected=runtime_precision,
			scaler=object(),  # type: ignore[arg-type]
		)


def test_nonfinite_gradient_fails_before_parameter_update(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = nn.Linear(1, 1, bias=False)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
	parameter_before = model.weight.detach().clone()
	optimizer_steps = 0
	original_step = optimizer.step

	def counted_step(*args: object, **kwargs: object) -> object:
		nonlocal optimizer_steps
		optimizer_steps += 1
		return original_step(*args, **kwargs)

	monkeypatch.setattr(optimizer, 'step', counted_step)
	loss = _FiniteValueNonfiniteGradient.apply(model.weight.sum())
	assert torch.isfinite(loss)
	with pytest.raises(
		FloatingPointError, match='non-finite Volve horizon gradient norm'
	):
		backward_and_step_horizon_optimizer(
			loss=loss,
			model=model,
			optimizer=optimizer,
			scaler=None,
			gradient_clip_norm=1.0,
		)

	assert optimizer_steps == 0
	assert torch.equal(model.weight.detach(), parameter_before)
