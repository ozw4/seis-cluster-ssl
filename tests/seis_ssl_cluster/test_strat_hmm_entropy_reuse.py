"""Exact parity and reuse contracts for prototype usage entropy."""

from __future__ import annotations

import math

import pytest
import torch

from seis_ssl_cluster.stratigraphy import (
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
	usage_entropy_floor_loss,
	usage_entropy_floor_loss_with_entropy,
)
from seis_ssl_cluster.training.strat_hmm import losses
from tests.seis_ssl_cluster.test_strat_hmm_synchronization import (
	_CUDA,
	_assert_same_run,
	_run_epoch,
)

_ROUTES = ['head_only', 'multi_head', 'posterior', 'center_trace']


def _reference_entropy(probs, valid_mask, eps=1.0e-8):
	selected = probs.reshape(-1, probs.shape[-1])[valid_mask.reshape(-1)]
	if selected.numel() == 0:
		return probs.new_zeros(())
	q_bar = selected.mean(dim=0)
	return -(q_bar * (q_bar + eps).log()).sum()


def _separate_loss_and_entropy(probs, *, valid_mask, entropy_floor, eps=1.0e-8):
	# Independent reductions reproduce the original loss and metric graphs.
	entropy = _reference_entropy(probs, valid_mask, eps)
	loss = (probs.new_tensor(float(entropy_floor)) - entropy).clamp_min(0.0).square()
	return loss, _reference_entropy(probs, valid_mask, eps)


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize(
	'dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize('entropy_floor', [0.0, math.log(3)])
@pytest.mark.parametrize('eps', [1.0e-8, 1.0e-3])
def test_shared_entropy_matches_separate_values_and_gradients(
	device, dtype, entropy_floor, eps
):
	# Binary fractions sum exactly to one in every dtype; inputs are strided.
	probs = (
		torch.tensor(
			[[[0.75, 0.125, 0.125], [0.5, 0.25, 0.25]]] * 3,
			device=device,
			dtype=dtype,
		)
		.transpose(0, 1)
		.requires_grad_()
	)
	valid_mask = torch.tensor(
		[[True, False], [False, True], [True, True]], device=device
	).T
	kwargs = {'valid_mask': valid_mask, 'entropy_floor': entropy_floor, 'eps': eps}
	actual = usage_entropy_floor_loss_with_entropy(probs, **kwargs)
	expected = _separate_loss_and_entropy(probs, **kwargs)
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	torch.testing.assert_close(
		usage_entropy_floor_loss(probs, **kwargs), expected[0], rtol=0, atol=0
	)
	for actual_value, expected_value in zip(actual, expected, strict=True):
		assert actual_value.dtype == dtype
		actual_grad = torch.autograd.grad(actual_value, probs, retain_graph=True)[0]
		expected_grad = torch.autograd.grad(expected_value, probs, retain_graph=True)[0]
		torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
		assert torch.equal(
			actual_grad[~valid_mask], torch.zeros_like(actual_grad[~valid_mask])
		)


@pytest.mark.parametrize('device', ['cpu', _CUDA])
def test_single_prototype_and_scalar_mask_match_original_formula(device):
	probs = torch.ones(1, device=device, requires_grad=True)
	kwargs = {
		'valid_mask': torch.ones((), dtype=torch.bool, device=device),
		'entropy_floor': 0.0,
	}
	actual = usage_entropy_floor_loss_with_entropy(probs, **kwargs)
	expected = _separate_loss_and_entropy(probs, **kwargs)
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
	'function', [usage_entropy_floor_loss, usage_entropy_floor_loss_with_entropy]
)
@pytest.mark.parametrize(
	('case', 'error_type', 'message'),
	[
		('empty', ValueError, 'requires at least one valid token'),
		('nan', ValueError, 'probs must be finite'),
		('negative', ValueError, 'probs must be nonnegative'),
		('unnormalized', ValueError, 'probs must sum to 1'),
		('mask_shape', ValueError, 'valid_mask shape must match'),
		('mask_dtype', TypeError, 'valid_mask must have dtype torch.bool'),
		('floor', ValueError, 'entropy_floor must be finite'),
		('floor_bool', TypeError, 'entropy_floor must be a float'),
		('eps', ValueError, 'eps must be positive and finite'),
		('eps_bool', TypeError, 'eps must be a float'),
		('priority', ValueError, 'probs must be finite'),
	],
)
def test_usage_api_preserves_validation(function, case, error_type, message):
	probs = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
	kwargs = {'valid_mask': torch.tensor([True, False]), 'entropy_floor': 0.5}
	if case == 'empty':
		kwargs['valid_mask'].zero_()
	elif case in {'nan', 'negative', 'unnormalized', 'priority'}:
		# Probability validation also covers tokens outside the selected mask.
		probs[-1, 0] = {
			'nan': float('nan'),
			'negative': -0.5,
			'unnormalized': 2.0,
			'priority': float('nan'),
		}[case]
		if case == 'priority':
			kwargs.update(valid_mask=torch.zeros(1), entropy_floor=-1.0)
	elif case == 'mask_shape':
		kwargs['valid_mask'] = torch.ones(3, dtype=torch.bool)
	elif case == 'mask_dtype':
		kwargs['valid_mask'] = torch.ones(2)
	elif case in {'floor', 'floor_bool'}:
		kwargs['entropy_floor'] = -1.0 if case == 'floor' else True
	else:
		kwargs['eps'] = 0.0 if case == 'eps' else True
	with pytest.raises(error_type, match=message):
		function(probs, **kwargs)


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('route', _ROUTES)
@pytest.mark.parametrize('mode', ['full', 'filtered'])
@pytest.mark.parametrize('usage_weight', [0.0, 0.2])
def test_training_reuses_entropy_within_each_head_and_batch(
	monkeypatch, device, route, mode, usage_weight
):
	shared_values = []
	fallback_values = []
	selected_masks = []
	original = losses.usage_entropy_floor_loss_with_entropy

	def record_shared(probs, **kwargs):
		pair = original(probs, **kwargs)
		shared_values.append(pair[1])
		selected_masks.append(kwargs['valid_mask'].tolist())
		return pair

	def record_fallback(probs, valid_mask):
		value = _reference_entropy(probs, valid_mask)
		fallback_values.append(value)
		selected_masks.append(valid_mask.tolist())
		return value

	monkeypatch.setattr(losses, 'usage_entropy_floor_loss_with_entropy', record_shared)
	monkeypatch.setattr(losses, '_prototype_usage_entropy', record_fallback)
	actual, actual_grad = _loss_case(route, device, mode, usage_weight)
	metrics = [
		value
		for key, value in actual.items()
		if key.startswith('prototype_usage_entropy')
	]
	assert len(metrics) == (1 if route == 'head_only' else 3)
	assert len(shared_values) == (len(metrics) if usage_weight else 0)
	assert len(fallback_values) == (0 if usage_weight else len(metrics))
	assert all(
		a is b for a, b in zip(metrics, shared_values or fallback_values, strict=True)
	)
	if mode == 'full':
		expected_masks = [[[True, True, True, True]]] * len(metrics)
	elif route == 'posterior':
		expected_masks = [[[True, True, True, False]]] * 3
	else:
		expected_masks = [
			[[True, True, False, False]],
			[[False, True, True, False]],
			[[True, True, True, False]],
		][: len(metrics)]
	assert selected_masks == expected_masks
	monkeypatch.setattr(
		losses, 'usage_entropy_floor_loss_with_entropy', _separate_loss_and_entropy
	)
	expected, expected_grad = _loss_case(route, device, mode, usage_weight)
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
	if usage_weight:
		assert actual['loss_usage'].item() > 0.0


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('route', ['multi_head', 'posterior'])
@pytest.mark.parametrize('usage_weight', [0.0, 0.2])
def test_empty_supervision_preserves_graph_zero_and_entropy_zero(
	monkeypatch, device, route, usage_weight
):
	def reject_unused_loss(*_args, **_kwargs):
		raise AssertionError('empty supervision must bypass usage loss validation')

	monkeypatch.setattr(
		losses, 'usage_entropy_floor_loss_with_entropy', reject_unused_loss
	)
	result, gradients = _loss_case(route, device, 'empty', usage_weight)
	assert result['loss'].requires_grad
	assert result['loss'].item() == 0.0
	for k in (6, 8, 10):
		entropy = result[f'prototype_usage_entropy_k{k}']
		assert entropy.item() == 0.0
		assert not entropy.requires_grad
	assert all(torch.equal(grad, torch.zeros_like(grad)) for grad in gradients)


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('route', _ROUTES)
def test_epoch_matches_separate_entropy_computation_exactly(monkeypatch, device, route):
	actual = _run_epoch(route, device, grad_clip_norm=0.1)
	monkeypatch.setattr(
		losses, 'usage_entropy_floor_loss_with_entropy', _separate_loss_and_entropy
	)
	expected = _run_epoch(route, device, grad_clip_norm=0.1)
	_assert_same_run(actual, expected)


def test_distillation_only_skips_logits_and_keeps_zero_entropy(monkeypatch):
	def reject_unused_computation(*_args, **_kwargs):
		raise AssertionError('distillation alone must not compute prototype entropy')

	head = OrderedPrototypeHead(feature_dim=3, num_prototypes=6)
	monkeypatch.setattr(head, 'forward', reject_unused_computation)
	monkeypatch.setattr(losses, '_prototype_usage_entropy', reject_unused_computation)
	monkeypatch.setattr(
		losses, 'usage_entropy_floor_loss_with_entropy', reject_unused_computation
	)
	tokens = torch.ones(1, 2, 3, requires_grad=True)
	result = losses.compute_strat_hmm_pretext_losses(
		head=head,
		encoded={'tokens': tokens},
		teacher_encoded={'tokens': -tokens.detach()},
		batch={
			'strat_labels': torch.zeros(1, 2, dtype=torch.long),
			'strat_confidence': torch.ones(1, 2),
			'strat_boundary_weight': torch.ones(1, 2),
			'strat_valid_mask': torch.ones(1, 2, dtype=torch.bool),
		},
		loss_config={
			'prototype_weight': 0.0,
			'usage_weight': 0.0,
			'distillation_weight': 0.2,
		},
		pseudo_target_config={},
	)
	assert result['prototype_usage_entropy'].item() == 0.0
	assert not result['prototype_usage_entropy'].requires_grad
	result['loss'].backward()
	assert tokens.grad is not None


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('route', ['head_only', 'multi_head', 'center_trace'])
def test_amp_epoch_matches_separate_entropy_computation_exactly(monkeypatch, route):
	actual = _run_epoch(route, 'cuda', grad_clip_norm=0.1, amp_enabled=True)
	monkeypatch.setattr(
		losses, 'usage_entropy_floor_loss_with_entropy', _separate_loss_and_entropy
	)
	expected = _run_epoch(route, 'cuda', grad_clip_norm=0.1, amp_enabled=True)
	_assert_same_run(actual, expected)


def _loss_case(route, device, mode, usage_weight):
	torch.manual_seed(821)
	head = (
		OrderedPrototypeHead(feature_dim=3, num_prototypes=6, temperature=0.05)
		if route == 'head_only'
		else MultiResolutionOrderedPrototypeHeads(
			feature_dim=3,
			ks=(6, 8, 10),
			projection_dim=None,
			normalize=True,
			temperature=0.05,
		)
	).to(device)
	tokens = (
		torch.tensor(
			[
				[[1.0, 0.5, -0.25]],
				[[1.1, 0.45, -0.2]],
				[[0.9, 0.55, -0.3]],
				[[1.2, 0.4, -0.15]],
			],
			device=device,
		)
		.transpose(0, 1)
		.requires_grad_()
	)
	student_mask = torch.ones(1, 4, dtype=torch.bool, device=device)
	if mode == 'filtered':
		student_mask[0, 3] = False
	elif mode == 'empty':
		student_mask.zero_()
	targets = {
		f'k{k}': {
			'labels': torch.tensor([[0, 1, 2, 3]], device=device),
			'confidence': torch.ones(1, 4, device=device),
			'boundary_weight': torch.ones(1, 4, device=device),
			'valid_mask': torch.ones(1, 4, dtype=torch.bool, device=device),
		}
		for k in (6, 8, 10)
	}
	if mode == 'filtered':
		targets['k6']['confidence'][0, 2] = 0.0
		targets['k8']['confidence'][0, 0] = 0.0
	kwargs = {
		'encoded': {'tokens': tokens, 'token_valid_mask': student_mask},
		'teacher_encoded': None,
		'loss_config': {
			'prototype_weight': 1.0,
			'usage_weight': usage_weight,
			'entropy_floor': 1.7,
		},
		'pseudo_target_config': {'min_confidence': 0.5 if mode == 'filtered' else 0.0},
		'batch': {'strat_multi_targets': targets},
	}
	if route == 'head_only':
		kwargs.update(
			head=head,
			batch={f'strat_{key}': value for key, value in targets['k6'].items()},
		)
		function = losses.compute_strat_hmm_pretext_losses
	else:
		kwargs['heads'] = head
		function = losses.compute_strat_hmm_multi_head_losses
		if route == 'center_trace':
			kwargs['replacement_mask'] = torch.tensor(
				[[True, False, True, False]], device=device
			).reshape(1, 2, 2, 1)
			function = losses.compute_strat_hmm_center_trace_masked_losses
		elif route == 'posterior':
			kwargs.pop('pseudo_target_config')
			kwargs['batch'] = {
				'strat_multi_posteriors': {
					key: {
						'posterior': torch.nn.functional.one_hot(
							target['labels'], int(key[1:])
						).float(),
						'valid_mask': target['valid_mask'],
					}
					for key, target in targets.items()
				}
			}
			function = losses.compute_strat_hmm_multi_head_posterior_losses
	result = function(**kwargs)
	gradients = torch.autograd.grad(result['loss'], (tokens, *head.parameters()))
	return result, gradients
