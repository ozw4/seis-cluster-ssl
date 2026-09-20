"""Contracts for batched hard-target weight validation."""

from __future__ import annotations

import re

import pytest
import torch

from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.strat_hmm import losses
from tests.seis_ssl_cluster.test_strat_hmm_synchronization import (
	_CUDA,
	_assert_same_run,
	_run_epoch,
)
from tests.seis_ssl_cluster.test_strat_multi_head_losses import _batch, _StaticHeads


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('route', ['multi_head', 'center_trace'])
@pytest.mark.parametrize('field', ['confidence', 'boundary_weight'])
@pytest.mark.parametrize(
	'bad_value', [float('nan'), float('inf'), -float('inf'), -0.25]
)
def test_weight_validation_rejects_invalid_values_even_outside_mask(
	device, route, field, bad_value
):
	batch = _batch()
	for target in batch['strat_multi_targets'].values():
		target['valid_mask'][0, -1] = False
	batch['strat_multi_targets']['k10'][field][0, -1] = bad_value
	expected = 'nonnegative' if bad_value == -0.25 else 'finite'
	with pytest.raises(
		ValueError,
		match=f"multi-head 'k10' {field.replace('_', ' ')} must be {expected}",
	):
		_call_losses(batch, route, device)


@pytest.mark.parametrize('route', ['multi_head', 'center_trace'])
@pytest.mark.parametrize(
	('corruptions', 'expected'),
	[
		(
			[('k6', 'confidence', -0.5), ('k6', 'boundary_weight', float('nan'))],
			"multi-head 'k6' confidence must be nonnegative",
		),
		(
			[('k6', 'confidence', float('nan')), ('k6', 'labels', 6)],
			"multi-head 'k6' confidence must be finite",
		),
		(
			[('k6', 'boundary_weight', -0.5), ('k8', 'confidence', float('nan'))],
			"multi-head 'k6' boundary weight must be nonnegative",
		),
		(
			[('k6', 'confidence', -0.5), ('k8', 'confidence', None)],
			"multi-head 'k6' confidence must be nonnegative",
		),
		(
			[('k6', 'confidence', -0.5), ('k8', 'labels', torch.zeros(1, 5))],
			"multi-head 'k6' confidence must be nonnegative",
		),
		(
			[
				('k6', 'confidence', -0.5),
				('k8', 'boundary_weight', torch.ones(1, 4, dtype=torch.float64)),
			],
			"multi-head 'k6' confidence must be nonnegative",
		),
		(
			[('k6', 'labels', 6), ('k8', 'confidence', float('nan'))],
			"multi-head 'k6' valid labels must be in prototype range [0, 6)",
		),
		(
			[('k8', 'confidence', -0.5), ('k8', 'valid_mask', False)],
			"multi-head 'k8' confidence must be nonnegative",
		),
	],
)
def test_deferred_checks_preserve_first_error_across_fields_and_heads(
	monkeypatch, route, corruptions, expected
):
	batch = _batch()
	for head, field, value in corruptions:
		if isinstance(value, (float, int)):
			batch['strat_multi_targets'][head][field].fill_(value)
		else:
			batch['strat_multi_targets'][head][field] = value
	with pytest.raises(ValueError, match=re.escape(expected)) as actual_error:
		_call_losses(batch, route, 'cpu')
	assert str(actual_error.value) == expected
	_use_immediate_scalar_checks(monkeypatch)
	with pytest.raises(ValueError, match=re.escape(expected)) as original_error:
		_call_losses(batch, route, 'cpu')
	assert str(original_error.value) == expected


@pytest.mark.parametrize('route', ['multi_head', 'center_trace'])
def test_structural_error_precedes_numeric_error_in_same_head(route):
	batch = _batch()
	batch['strat_multi_targets']['k6']['confidence'].fill_(float('nan'))
	batch['strat_multi_targets']['k6']['boundary_weight'] = torch.ones(
		1, 4, dtype=torch.float64
	)
	with pytest.raises(TypeError, match='dtype must match strat_confidence dtype'):
		_call_losses(batch, route, 'cpu')


@pytest.mark.parametrize('route', ['multi_head', 'center_trace'])
def test_later_structural_error_is_preserved_when_queued_weights_are_valid(route):
	batch = _batch()
	batch['strat_multi_targets']['k8']['confidence'] = None
	with pytest.raises(TypeError, match='confidence'):
		_call_losses(batch, route, 'cpu')


@pytest.mark.parametrize('route', ['multi_head', 'center_trace'])
def test_logits_error_priority_remains_route_specific(monkeypatch, route):
	batch = _batch()
	batch['strat_multi_targets']['k6']['confidence'].fill_(float('nan'))
	first_error = (
		(ValueError, "multi-head 'k6' confidence must be finite")
		if route == 'multi_head'
		else (FloatingPointError, 'non-finite multi-head logits for k8')
	)
	with pytest.raises(first_error[0]) as actual_error:
		_call_losses(batch, route, 'cpu', bad_logits_head='k8')
	assert str(actual_error.value) == first_error[1]
	_use_immediate_scalar_checks(monkeypatch)
	with pytest.raises(first_error[0]) as original_error:
		_call_losses(batch, route, 'cpu', bad_logits_head='k8')
	assert str(original_error.value) == first_error[1]


def test_finite_error_precedes_negative_error_within_one_tensor():
	batch = _batch()
	batch['strat_multi_targets']['k6']['confidence'][0, :2] = torch.tensor(
		[-0.5, float('nan')]
	)
	with pytest.raises(ValueError, match="multi-head 'k6' confidence must be finite"):
		_call_losses(batch, 'multi_head', 'cpu')


def test_unity_boundary_error_keeps_its_priority():
	batch = _batch()
	batch['strat_multi_targets']['k6']['boundary_weight'].fill_(0.5)
	batch['strat_multi_targets']['k8']['confidence'].fill_(float('nan'))
	with pytest.raises(ValueError, match='boundary weight must be one'):
		_call_losses(batch, 'multi_head', 'cpu')
	batch['strat_multi_targets']['k6']['confidence'].fill_(-0.5)
	with pytest.raises(
		ValueError, match="multi-head 'k6' confidence must be nonnegative"
	):
		_call_losses(batch, 'multi_head', 'cpu')


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('route', ['multi_head', 'center_trace'])
def test_training_matches_immediate_scalar_validation_exactly(
	monkeypatch, route, device
):
	actual = _run_epoch(route, device, grad_clip_norm=0.1)
	_use_immediate_scalar_checks(monkeypatch)
	expected = _run_epoch(route, device, grad_clip_norm=0.1)
	_assert_same_run(actual, expected)


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('route', ['multi_head', 'center_trace'])
def test_amp_training_matches_immediate_scalar_validation_exactly(monkeypatch, route):
	actual = _run_epoch(route, 'cuda', grad_clip_norm=0.1, amp_enabled=True)
	_use_immediate_scalar_checks(monkeypatch)
	expected = _run_epoch(route, 'cuda', grad_clip_norm=0.1, amp_enabled=True)
	_assert_same_run(actual, expected)


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('token_count', [0, 4])
def test_target_preparation_preserves_casts_and_gradient_contract(device, token_count):
	confidence = torch.ones(
		1, token_count, dtype=torch.float64, device=device, requires_grad=True
	)
	boundary = torch.ones_like(confidence, requires_grad=True)
	values = losses._multi_head_target_values(  # noqa: SLF001
		{
			'labels': torch.zeros(1, token_count, dtype=torch.int32, device=device),
			'confidence': confidence,
			'boundary_weight': boundary,
			'valid_mask': torch.zeros(1, token_count, dtype=torch.bool, device=device),
		},
		reference=torch.zeros(1, token_count, 6, device=device),
		head_key='k6',
	)
	assert values.labels.dtype == torch.long
	assert values.confidence.dtype == torch.float32
	assert values.boundary_weight.dtype == torch.float32
	assert not values.confidence.requires_grad
	assert values.boundary_weight.requires_grad
	values.boundary_weight.sum().backward()
	assert torch.equal(boundary.grad, torch.ones_like(boundary))
	assert confidence.grad is None


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_three_head_weight_checks_use_one_host_transfer(monkeypatch):
	batch = move_batch_to_device(_batch(), torch.device('cuda'))
	transfers = []
	original_cpu = torch.Tensor.cpu

	def record_cpu(value, *args, **kwargs):
		transfers.append((value.device.type, value.dtype, value.numel()))
		return original_cpu(value, *args, **kwargs)

	monkeypatch.setattr(torch.Tensor, 'cpu', record_cpu)
	with (
		torch.profiler.profile(
			activities=[torch.profiler.ProfilerActivity.CPU]
		) as profile,
		losses._batch_target_weight_validation() as checks,  # noqa: SLF001
	):
		for head_key, target in batch['strat_multi_targets'].items():
			losses._multi_head_target_values(  # noqa: SLF001
				target,
				reference=torch.zeros(1, 4, int(head_key[1:]), device='cuda'),
				head_key=head_key,
				require_unity_boundary_weight=False,
				weight_checks=checks,
			)
	assert transfers == [('cuda', torch.bool, 12)]
	assert not any(
		event.key == 'aten::_local_scalar_dense' for event in profile.key_averages()
	)


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_mixed_device_flags_keep_original_error_order():
	checks = [
		(torch.ones((), dtype=torch.bool), 'first'),
		(torch.zeros((), dtype=torch.bool, device='cuda'), 'second'),
		(torch.zeros((), dtype=torch.bool), 'third'),
	]
	with pytest.raises(ValueError, match=r'^second$'):
		losses._raise_for_target_weight_checks(checks)  # noqa: SLF001


def _use_immediate_scalar_checks(monkeypatch):
	original = losses._multi_head_target_values  # noqa: SLF001

	def immediate_values(target, **kwargs):
		kwargs.pop('weight_checks', None)
		return original(target, **kwargs)

	def scalar_checks(checks):
		for valid, message in checks:
			if not bool(valid.item()):
				raise ValueError(message)

	monkeypatch.setattr(losses, '_multi_head_target_values', immediate_values)
	monkeypatch.setattr(losses, '_raise_for_target_weight_checks', scalar_checks)


def _call_losses(batch, route, device, *, bad_logits_head=None):
	batch = move_batch_to_device(batch, torch.device(device))
	logits = {
		f'k{k}': torch.zeros(1, 4, k, device=device, requires_grad=True)
		for k in (6, 8, 10)
	}
	if bad_logits_head is not None:
		logits[bad_logits_head] = torch.full_like(logits[bad_logits_head], float('nan'))
	kwargs = {}
	if route == 'center_trace':
		compute = losses.compute_strat_hmm_center_trace_masked_losses
		kwargs['replacement_mask'] = torch.tensor(
			[True, False, False, False], device=device
		).reshape(1, 2, 2, 1)
	else:
		compute = losses.compute_strat_hmm_multi_head_losses
	return compute(
		heads=_StaticHeads(logits),
		encoded={'tokens': torch.zeros(1, 4, 3, device=device)},
		teacher_encoded=None,
		batch=batch,
		loss_config={
			'prototype_weight': 0.0,
			'usage_weight': 0.0,
			'consistency_weight': 0.0,
		},
		pseudo_target_config={},
		**kwargs,
	)
