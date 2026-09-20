"""Numerical and failure contracts for grouped strat HMM host reads."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from seis_ssl_cluster.models.mae import LearnedEncoderReplacementToken
from seis_ssl_cluster.stratigraphy import (
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
)
from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.strat_hmm import epoch, losses
from seis_ssl_cluster.training.strat_hmm.tensor_ops import (
	all_tensors_finite,
	materialize_metrics,
)
from tests.seis_ssl_cluster.test_strat_center_trace_masked import (
	_runtime_batch,
	_RuntimeStudent,
)

_CUDA = pytest.param(
	'cuda',
	marks=[
		pytest.mark.requires_cuda,
		pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable'),
	],
)


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('bad_value', [None, float('nan'), float('inf'), -float('inf')])
def test_finite_checks_cover_every_tensor_and_allow_empty_inputs(device, bad_value):
	values = [
		torch.empty(0, device=device),
		torch.ones(3, device=device, dtype=torch.float16),
		torch.ones(2, device=device, dtype=torch.float64),
	]
	if bad_value is not None:
		values[-1][-1] = bad_value
	assert all_tensors_finite(iter(values)) is (bad_value is None)
	assert all_tensors_finite([])
	assert all_tensors_finite([torch.empty(0, device=device)])


@pytest.mark.parametrize('device', ['cpu', _CUDA])
def test_metric_materialization_preserves_scalar_precision_order_and_autograd(device):
	metrics = {
		'half': torch.tensor(0.1, dtype=torch.float16, device=device),
		'count': torch.tensor(2**60 + 129, dtype=torch.int64, device=device),
		'loss': torch.tensor(0.3, device=device, requires_grad=True),
		'double': torch.tensor(1.0 + 2**-40, dtype=torch.float64, device=device),
		'bfloat': torch.tensor(0.3, dtype=torch.bfloat16, device=device),
		'fraction': torch.tensor([0.2], device=device),
		'cpu_metric': torch.tensor(0.7),
	}
	expected = {
		key: float(value.detach().cpu().item()) for key, value in metrics.items()
	}
	assert materialize_metrics(metrics) == expected
	assert list(materialize_metrics(metrics)) == list(metrics)
	assert materialize_metrics({}) == {}
	metrics['loss'].backward()
	assert metrics['loss'].grad.item() == 1.0
	with pytest.raises(RuntimeError):
		materialize_metrics({'not_scalar': torch.ones(2, device=device)})


@pytest.mark.parametrize('kind', ['loss', 'gradient', 'gradient norm'])
@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -float('inf')])
def test_guards_keep_nonfinite_diagnostics(kind, bad_value):
	context = {'epoch': 2, 'global_step': 3, 'batch_index': 4}
	module = torch.nn.Linear(2, 1)
	for parameter in module.parameters():
		parameter.grad = torch.ones_like(parameter)
	module.bias.grad.fill_(bad_value)
	guard, args = {
		'loss': (
			epoch._ensure_finite_losses,  # noqa: SLF001
			({'loss': torch.tensor(1.0), 'last_metric': torch.tensor(bad_value)},),
		),
		'gradient': (epoch._ensure_finite_gradients, ((module,),)),  # noqa: SLF001
		'gradient norm': (epoch._clip_gradients_for_modules, ((module,), 0.1)),  # noqa: SLF001
	}[kind]
	with pytest.raises(
		FloatingPointError,
		match=f'non-finite strat HMM pretext {kind} at epoch 2, step 3, batch 4',
	):
		guard(*args, **context)


def test_gradient_guard_allows_absent_gradients_and_ignores_frozen_parameters():
	module = torch.nn.Linear(2, 1)
	module.bias.requires_grad_(requires_grad=False)
	module.bias.grad = torch.full_like(module.bias, float('nan'))
	context = {'epoch': 1, 'global_step': 0, 'batch_index': 0}
	epoch._ensure_finite_gradients((module,), **context)  # noqa: SLF001
	epoch._ensure_finite_gradients((), **context)  # noqa: SLF001


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('kind', ['loss', 'gradient'])
def test_epoch_rejects_nonfinite_values_before_optimizer_step(
	monkeypatch, device, kind
):
	original_losses = epoch.compute_strat_hmm_multi_head_losses
	steps = []

	def corrupt_losses(**kwargs):
		result = original_losses(**kwargs)
		if kind == 'loss':
			result['last_diagnostic'] = result['loss'].new_tensor(float('nan'))
		else:
			result['loss'].register_hook(
				lambda gradient: torch.full_like(gradient, float('inf'))
			)
		return result

	monkeypatch.setattr(epoch, 'compute_strat_hmm_multi_head_losses', corrupt_losses)
	monkeypatch.setattr(
		torch.optim.AdamW, 'step', lambda *_args, **_kwargs: steps.append(1)
	)
	with pytest.raises(
		FloatingPointError, match=f'non-finite strat HMM pretext {kind}'
	):
		_run_epoch(
			'multi_head', device, grad_clip_norm=0.1, amp_enabled=device == 'cuda'
		)
	assert not steps


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize(
	'route', ['head_only', 'multi_head', 'posterior', 'center_trace']
)
@pytest.mark.parametrize('grad_clip_norm', [None, 0.1])
def test_epoch_matches_scalar_reads_exactly(monkeypatch, device, route, grad_clip_norm):
	def transfer(batch, device, *, non_blocking=False):
		assert non_blocking
		return move_batch_to_device(batch, device, non_blocking=non_blocking)

	monkeypatch.setattr(epoch, 'move_batch_to_device', transfer)
	actual = _run_epoch(route, device, grad_clip_norm=grad_clip_norm)
	_with_scalar_reads(monkeypatch)
	expected = _run_epoch(route, device, grad_clip_norm=grad_clip_norm)
	_assert_same_run(actual, expected)


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('route', ['head_only', 'multi_head', 'center_trace'])
def test_amp_epoch_matches_scalar_reads_exactly(monkeypatch, route):
	actual = _run_epoch(route, 'cuda', grad_clip_norm=0.1, amp_enabled=True)
	_with_scalar_reads(monkeypatch)
	expected = _run_epoch(route, 'cuda', grad_clip_norm=0.1, amp_enabled=True)
	_assert_same_run(actual, expected)


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_host_reads_are_grouped(monkeypatch):
	values = [torch.ones(3, device='cuda') for _ in range(12)]
	with torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU]
	) as profile:
		assert all_tensors_finite(values)
	reads = sum(
		event.count
		for event in profile.key_averages()
		if event.key == 'aten::_local_scalar_dense'
	)
	assert reads == 1
	metrics = {f'metric_{index}': value.sum() for index, value in enumerate(values)}
	transfers = []
	original_cpu = torch.Tensor.cpu

	def record_cpu(value, *args, **kwargs):
		transfers.append((value.numel(), value.requires_grad))
		return original_cpu(value, *args, **kwargs)

	monkeypatch.setattr(torch.Tensor, 'cpu', record_cpu)
	assert materialize_metrics(metrics) == dict.fromkeys(metrics, 3.0)
	assert transfers == [(12, False)]


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_nested_pinned_inputs_match_blocking_transfer_and_preserve_metadata():
	batch = _pin_batch(_runtime_batch())
	batch['pageable'] = torch.arange(3)
	batch['coords'] = [{'survey_id': 'synthetic', 'start_xyz': (0, 0, 0)}]
	actual = move_batch_to_device(batch, torch.device('cuda'), non_blocking=True)
	expected = move_batch_to_device(batch, torch.device('cuda'))
	torch.testing.assert_close(
		{key: value for key, value in actual.items() if key != 'coords'},
		{key: value for key, value in expected.items() if key != 'coords'},
		rtol=0,
		atol=0,
	)
	assert actual['coords'] is batch['coords']
	assert all_tensors_finite([actual['x'], batch['x']])
	actual['strat_multi_targets']['k10']['confidence'].fill_(float('inf'))
	assert not all_tensors_finite(
		[batch['x'], actual['strat_multi_targets']['k10']['confidence']]
	)


def _with_scalar_reads(monkeypatch):
	def scalar_finite(values):
		return all(bool(torch.isfinite(value).all()) for value in values)

	def scalar_metrics(metrics):
		return {
			key: float(value.detach().cpu().item()) for key, value in metrics.items()
		}

	monkeypatch.setattr(epoch, 'all_tensors_finite', scalar_finite)
	monkeypatch.setattr(losses, 'all_tensors_finite', scalar_finite)
	monkeypatch.setattr(epoch, 'materialize_metrics', scalar_metrics)
	monkeypatch.setattr(
		epoch,
		'move_batch_to_device',
		lambda batch, device, **_kwargs: move_batch_to_device(batch, device),
	)


def _pin_batch(value):
	if isinstance(value, torch.Tensor):
		return value.pin_memory()
	return {key: _pin_batch(item) for key, item in value.items()}


def _run_epoch(route, device, *, grad_clip_norm, amp_enabled=False):
	torch.manual_seed(321)
	student = _RuntimeStudent().to(device)
	teacher = _RuntimeStudent().to(device).requires_grad_(requires_grad=False)
	replacement = LearnedEncoderReplacementToken(4, seed=5).to(device)
	head_type = (
		OrderedPrototypeHead
		if route == 'head_only'
		else MultiResolutionOrderedPrototypeHeads
	)
	head_options = {'num_prototypes': 6} if route == 'head_only' else {'ks': (6, 8, 10)}
	head = head_type(
		feature_dim=4, projection_dim=3, temperature=0.5, normalize=True, **head_options
	).to(device)
	modules = [student, head]
	if route == 'center_trace':
		modules.append(replacement)
	optimizer = torch.optim.AdamW(
		[parameter for module in modules for parameter in module.parameters()], lr=3e-4
	)
	batches = [_runtime_batch(), _runtime_batch(), _runtime_batch()]
	for batch in batches:
		if route == 'head_only':
			target = batch.pop('strat_multi_targets')['k6']
			batch.update({f'strat_{key}': value for key, value in target.items()})
		elif route == 'posterior':
			batch['strat_multi_posteriors'] = {
				key: {
					'posterior': torch.nn.functional.one_hot(
						target['labels'], int(key[1:])
					).float(),
					'valid_mask': target['valid_mask'],
				}
				for key, target in batch.pop('strat_multi_targets').items()
			}
	if device == 'cuda':
		batches = [_pin_batch(batch) for batch in batches]
	callbacks = []
	kwargs = {'head': head} if route == 'head_only' else {'heads': head}
	if route == 'center_trace':
		kwargs['replacement_token'] = replacement
		train = epoch.train_strat_hmm_center_trace_masked_one_epoch
	elif route == 'head_only':
		train = epoch.train_strat_hmm_head_only_one_epoch
	else:
		train = epoch.train_strat_hmm_multi_head_one_epoch
		if route == 'posterior':
			kwargs['target_representation'] = 'ordered_path_state_posterior_v1'
	scaler = torch.amp.GradScaler('cuda', init_scale=16.0) if amp_enabled else None
	state = train(
		student=student,
		teacher=teacher,
		dataloader=batches,
		optimizer=optimizer,
		device=torch.device(device),
		epoch=2,
		global_step=5,
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'distillation_weight': 0.2,
			'consistency_weight': 0.1 if route == 'multi_head' else 0.0,
		},
		pseudo_target_config={},
		grad_clip_norm=grad_clip_norm,
		amp_enabled=amp_enabled,
		scaler=scaler,
		step_callback=callbacks.append,
		**kwargs,
	)
	return (
		(state, callbacks),
		deepcopy([module.state_dict() for module in [*modules, teacher]]),
		deepcopy(optimizer.state_dict()),
		None if scaler is None else scaler.state_dict(),
		torch.get_rng_state(),
	)


def _assert_same_run(actual, expected):
	assert actual[0] == expected[0]
	torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)
