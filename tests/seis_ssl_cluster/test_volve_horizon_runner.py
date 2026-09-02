'''Tests for the regime-neutral Volve horizon training lifecycle.'''

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import seis_ssl_cluster.volve.horizon_runner as horizon_runner_module
from seis_ssl_cluster.volve.horizon_runner import (
	CHECKPOINT_SELECTION_VALIDATION_MAE,
	CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
	HorizonRunnerSettings,
	HorizonRuntimeContext,
	backward_and_step_horizon_optimizer,
	initial_best_validation_score,
	run_horizon_training_job,
	validate_horizon_resume_runtime,
	validation_checkpoint_score,
	validation_score_improved,
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


@pytest.mark.parametrize(
	('selection', 'candidate', 'best', 'expected'),
	[
		(CHECKPOINT_SELECTION_VALIDATION_MAE, 1.0, 2.0, True),
		(CHECKPOINT_SELECTION_VALIDATION_MAE, 1.0, 1.0, False),
		(CHECKPOINT_SELECTION_VALIDATION_MAE, 2.0, 1.0, False),
		(CHECKPOINT_SELECTION_VALIDATION_WITHIN_2, 0.4, 0.3, True),
		(CHECKPOINT_SELECTION_VALIDATION_WITHIN_2, 0.3, 0.3, False),
		(CHECKPOINT_SELECTION_VALIDATION_WITHIN_2, 0.2, 0.3, False),
	],
)
def test_validation_checkpoint_score_improvement_is_strict(
	selection: str,
	candidate: float,
	best: float,
	expected: object,
) -> None:
	assert validation_score_improved(candidate, best, selection) is expected


@pytest.mark.parametrize('candidate', [float('nan'), float('inf'), -float('inf')])
def test_validation_checkpoint_score_rejects_nonfinite_candidate(
	candidate: float,
) -> None:
	with pytest.raises(ValueError, match='candidate must be finite'):
		validation_score_improved(
			candidate,
			initial_best_validation_score(CHECKPOINT_SELECTION_VALIDATION_MAE),
			CHECKPOINT_SELECTION_VALIDATION_MAE,
		)


def test_validation_checkpoint_score_rejects_unknown_selection() -> None:
	with pytest.raises(ValueError, match='unknown horizon checkpoint selection'):
		initial_best_validation_score('unknown')
	with pytest.raises(ValueError, match='unknown horizon checkpoint selection'):
		validation_checkpoint_score({'macro_mae_samples': 1.0}, 'unknown')
	with pytest.raises(ValueError, match='unknown horizon checkpoint selection'):
		validation_score_improved(1.0, 2.0, 'unknown')


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


@pytest.mark.parametrize(
	('selection', 'expected_epoch', 'expected_weight'),
	[
		(CHECKPOINT_SELECTION_VALIDATION_MAE, 1, 2.0),
		(CHECKPOINT_SELECTION_VALIDATION_WITHIN_2, 2, 3.0),
	],
)
def test_runner_selects_policy_epoch_and_resume_matches_uninterrupted(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	selection: str,
	expected_epoch: int,
	expected_weight: float,
) -> None:
	validation_by_weight = {
		1: (5.0, 0.20),
		2: (4.0, 0.25),
		3: (4.5, 0.40),
	}

	def fake_evaluate(
		model: nn.Module,
		*_args: object,
		**_kwargs: object,
	) -> dict[str, object]:
		weight = round(float(next(model.parameters()).detach()))
		mae, within_2 = validation_by_weight[weight]
		metrics = {
			'macro_mae_samples': mae,
			'macro_within_2_samples': within_2,
		}
		return {'primary': dict(metrics), 'secondary': dict(metrics)}

	monkeypatch.setattr(
		horizon_runner_module,
		'evaluate_horizon_dataset',
		fake_evaluate,
	)

	def run(
		output_dir: Path,
		*,
		max_steps: int | None = None,
		resume: Path | None = None,
	) -> Path | None:
		def build(
			device: torch.device,
		) -> tuple[nn.Module, torch.optim.Optimizer]:
			model = _ScalarHorizonModel().to(device)
			with torch.no_grad():
				model.value.zero_()
			return model, torch.optim.SGD(model.parameters(), lr=0.1)

		def train(
			model: nn.Module,
			_item: dict[str, object],
			_optimizer: torch.optim.Optimizer,
			_runtime: HorizonRuntimeContext,
		) -> float:
			with torch.no_grad():
				next(model.parameters()).add_(1.0)
			return 1.0

		def predict(
			_model: nn.Module,
			item: dict[str, object],
			_runtime: HorizonRuntimeContext,
		) -> torch.Tensor:
			target = item['target_sample_float']
			assert isinstance(target, torch.Tensor)
			return target.unsqueeze(0)

		dataset = _OneItemHorizonDataset()
		return run_horizon_training_job(
			output_dir=output_dir,
			run_identity={
				'benchmark': 'synthetic_selection_v1',
				'objective': {'checkpoint_selection': selection},
			},
			settings=HorizonRunnerSettings(
				epochs=3,
				seed=7,
				amp_on_cuda=False,
				gradient_clip_norm=1.0,
				checkpoint_selection=selection,
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
			metrics_metadata={'schema_version': 1},
			build_model_and_optimizer=build,
			train_one_item=train,
			predict_one_item=predict,
			device='cpu',
			max_steps=max_steps,
			resume=resume,
		)

	uninterrupted_dir = tmp_path / f'{selection}_uninterrupted'
	resumed_dir = tmp_path / f'{selection}_resumed'
	assert run(uninterrupted_dir) == uninterrupted_dir / 'metrics.json'
	assert run(resumed_dir, max_steps=2) is None
	assert run(resumed_dir, resume=resumed_dir / 'latest.pt') == (
		resumed_dir / 'metrics.json'
	)

	for output_dir in (uninterrupted_dir, resumed_dir):
		metrics = json.loads(
			(output_dir / 'metrics.json').read_text(encoding='utf-8')
		)
		best = torch.load(
			output_dir / 'best.pt', map_location='cpu', weights_only=False
		)
		latest = torch.load(
			output_dir / 'latest.pt', map_location='cpu', weights_only=False
		)
		assert metrics['best_epoch'] == best['epoch'] == expected_epoch
		assert metrics['validation'] == best['validation']
		assert best['checkpoint_selection'] == selection
		assert latest['checkpoint_selection'] == selection
		assert latest['best_validation_score'] == best['best_validation_score']
		assert float(best['model_state_dict']['value']) == expected_weight
		assert float(latest['model_state_dict']['value']) == expected_weight
	assert json.loads(
		(uninterrupted_dir / 'metrics.json').read_text(encoding='utf-8')
	)['validation'] == json.loads(
		(resumed_dir / 'metrics.json').read_text(encoding='utf-8')
	)['validation']


def test_resume_legacy_mae_score_fallback_and_within2_rejection(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset = _OneItemHorizonDataset()

	def fake_evaluate(*_args: object, **_kwargs: object) -> dict[str, object]:
		metrics = {'macro_mae_samples': 4.0, 'macro_within_2_samples': 0.25}
		return {'primary': dict(metrics), 'secondary': dict(metrics)}

	monkeypatch.setattr(
		horizon_runner_module,
		'evaluate_horizon_dataset',
		fake_evaluate,
	)

	def run(
		output_dir: Path,
		selection: str,
		*,
		resume: Path | None = None,
	) -> Path | None:
		def build(
			device: torch.device,
		) -> tuple[nn.Module, torch.optim.Optimizer]:
			model = _ScalarHorizonModel().to(device)
			return model, torch.optim.SGD(model.parameters(), lr=0.1)

		def train(
			_model: nn.Module,
			_item: dict[str, object],
			_optimizer: torch.optim.Optimizer,
			_runtime: HorizonRuntimeContext,
		) -> float:
			return 1.0

		def predict(
			_model: nn.Module,
			item: dict[str, object],
			_runtime: HorizonRuntimeContext,
		) -> torch.Tensor:
			return item['target_sample_float'].unsqueeze(0)  # type: ignore[union-attr]

		return run_horizon_training_job(
			output_dir=output_dir,
			run_identity={'checkpoint_selection': selection},
			settings=HorizonRunnerSettings(
				epochs=2,
				seed=1,
				amp_on_cuda=False,
				gradient_clip_norm=1.0,
				checkpoint_selection=selection,
			),
			datasets=dict.fromkeys(
				('train', 'validation', 'test'), dataset
			),
			expected_counts=dict.fromkeys(
				('train', 'validation', 'test', 'test_primary'),
				(1, 1, 1, 1, 1),
			),
			metrics_metadata={},
			build_model_and_optimizer=build,
			train_one_item=train,
			predict_one_item=predict,
			device='cpu',
			max_steps=1 if resume is None else None,
			resume=resume,
		)

	mae_dir = tmp_path / 'legacy_mae'
	assert run(mae_dir, CHECKPOINT_SELECTION_VALIDATION_MAE) is None
	mae_latest = mae_dir / 'latest.pt'
	with pytest.raises(ValueError, match='does not match this horizon job'):
		run(
			mae_dir,
			CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
			resume=mae_latest,
		)
	payload = torch.load(mae_latest, map_location='cpu', weights_only=False)
	payload['best_validation_macro_mae_samples'] = payload.pop(
		'best_validation_score'
	)
	payload.pop('checkpoint_selection')
	torch.save(payload, mae_latest)
	assert run(
		mae_dir,
		CHECKPOINT_SELECTION_VALIDATION_MAE,
		resume=mae_latest,
	) == mae_dir / 'metrics.json'

	within_dir = tmp_path / 'legacy_within2'
	assert run(within_dir, CHECKPOINT_SELECTION_VALIDATION_WITHIN_2) is None
	within_latest = within_dir / 'latest.pt'
	payload = torch.load(within_latest, map_location='cpu', weights_only=False)
	payload['best_validation_macro_mae_samples'] = 4.0
	payload.pop('best_validation_score')
	payload.pop('checkpoint_selection')
	torch.save(payload, within_latest)
	with pytest.raises(ValueError, match='legacy MAE-only state'):
		run(
			within_dir,
			CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
			resume=within_latest,
		)


def test_amp_resume_requires_matching_precision_and_grad_scaler_state(
	tmp_path: Path,
) -> None:
	runtime_precision = {
		'device_type': 'cuda',
		'amp_enabled': True,
		'autocast_dtype': 'float16',
		'scaler_required': True,
	}
	checkpoint = tmp_path / 'synthetic_cuda_amp.pt'
	torch.save(
		{
			'runtime_precision': runtime_precision,
			'scaler_state_dict': None,
		},
		checkpoint,
	)
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	with pytest.raises(ValueError, match='runtime precision'):
		validate_horizon_resume_runtime(
			payload,
			expected={
				'device_type': 'cpu',
				'amp_enabled': False,
				'autocast_dtype': None,
				'scaler_required': False,
			},
			scaler=None,
		)
	with pytest.raises(ValueError, match='missing required GradScaler state'):
		validate_horizon_resume_runtime(
			payload,
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
