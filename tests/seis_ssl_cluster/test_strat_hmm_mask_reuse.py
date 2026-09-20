"""Regression coverage for reusing hard-target mask emptiness checks."""

from __future__ import annotations

import inspect

import pytest
import torch

from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.strat_hmm import losses
from tests.seis_ssl_cluster.test_strat_hmm_synchronization import _CUDA
from tests.seis_ssl_cluster.test_strat_multi_head_losses import _batch, _heads


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize(
	'dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize('case', ['empty', 'zero', 'mixed', 'large'])
def test_cached_masked_mean_preserves_values_and_gradients(
	monkeypatch, device, dtype, case
):
	mask = torch.tensor([True, False, True, False], device=device)
	if case == 'empty':
		mask.zero_()
	values = torch.tensor([-3.0, 2.0, 4.0, -1.0], device=device, dtype=dtype)
	if case == 'zero':
		values.zero_()
	elif case == 'large':
		values.fill_(torch.finfo(dtype).max / 2)
	values.requires_grad_()
	has_valid_tokens = bool(mask.any().item())
	expected = losses._masked_mean(values, mask)  # noqa: SLF001
	expected_gradient = (
		torch.autograd.grad(expected, values)[0] if expected.requires_grad else None
	)

	def reject_repeated_any(*_args, **_kwargs):
		raise AssertionError('cached emptiness must not read the mask again')

	with monkeypatch.context() as scoped:
		scoped.setattr(torch.Tensor, 'any', reject_repeated_any)
		actual = losses._masked_mean(  # noqa: SLF001
			values, mask, has_valid_tokens=has_valid_tokens
		)
	assert actual.requires_grad == expected.requires_grad
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	if actual.requires_grad:
		actual_gradient = torch.autograd.grad(actual, values)[0]
		torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
	else:
		assert actual.item() == 0.0


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('mode', ['full', 'filtered', 'empty'])
@pytest.mark.parametrize(
	('prototype_weight', 'usage_weight'),
	[(1.0, 0.2), (0.0, 0.2), (1.0, 0.0), (0.0, 0.0)],
)
def test_each_filtered_head_mask_is_read_once_and_reused(
	monkeypatch, device, mode, prototype_weight, usage_weight
):
	torch.manual_seed(248)
	batch, student_mask, min_confidence, expected_flags = _mask_fixture(mode)
	batch = move_batch_to_device(batch, torch.device(device))
	tokens = torch.randn(1, 4, 3, device=device, requires_grad=True)
	heads = _heads().to(device)
	head_mask_reads = []
	metric_flags = []
	original_any = torch.Tensor.any
	original_mean = losses._masked_mean  # noqa: SLF001

	def record_any(value, *args, **kwargs):
		frame = inspect.currentframe().f_back
		if (
			frame.f_code.co_name == 'compute_strat_hmm_multi_head_losses'
			and value.ndim == 2
			and value.dtype == torch.bool
		):
			head_mask_reads.append(value)
		return original_any(value, *args, **kwargs)

	def record_mean(values, mask, *, has_valid_tokens=None):
		actual = original_mean(values, mask, has_valid_tokens=has_valid_tokens)
		if has_valid_tokens is not None:
			metric_flags.append(has_valid_tokens)
			expected = original_mean(values, mask)
			torch.testing.assert_close(actual, expected, rtol=0, atol=0)
		return actual

	monkeypatch.setattr(torch.Tensor, 'any', record_any)
	monkeypatch.setattr(losses, '_masked_mean', record_mean)
	result = losses.compute_strat_hmm_multi_head_losses(
		heads=heads,
		encoded={'tokens': tokens, 'token_valid_mask': student_mask.to(device)},
		teacher_encoded=None,
		batch=batch,
		loss_config={
			'prototype_weight': prototype_weight,
			'usage_weight': usage_weight,
			'consistency_weight': 0.1,
		},
		pseudo_target_config={'min_confidence': min_confidence},
	)
	assert len(head_mask_reads) == 3
	assert metric_flags == expected_flags
	assert [
		bool(original_any(mask).item()) for mask in head_mask_reads
	] == expected_flags
	result['loss'].backward()
	assert tokens.grad is not None
	assert torch.isfinite(tokens.grad).all()
	if mode == 'empty':
		assert result['loss'].item() == 0.0
		assert torch.equal(tokens.grad, torch.zeros_like(tokens))
	for k, has_tokens in zip((6, 8, 10), expected_flags, strict=True):
		if not has_tokens or prototype_weight == 0.0:
			assert result[f'loss_prototype_k{k}'].item() == 0.0
		if not has_tokens or usage_weight == 0.0:
			assert result[f'loss_usage_k{k}'].item() == 0.0
		if not has_tokens:
			assert result[f'mean_confidence_valid_k{k}'].item() == 0.0


def _mask_fixture(mode):
	batch = _batch()
	student_mask = torch.ones(1, 4, dtype=torch.bool)
	min_confidence = 0.0
	expected_flags = [True, True, True]
	if mode == 'filtered':
		min_confidence = 0.5
		student_mask[0, 2] = False
		batch['strat_multi_targets']['k8']['confidence'].zero_()
		batch['strat_multi_targets']['k10']['confidence'] = torch.tensor(
			[[0.0, 1.0, 0.0, 0.0]]
		)
		expected_flags = [True, False, True]
	elif mode == 'empty':
		student_mask.zero_()
		expected_flags = [False, False, False]
	return batch, student_mask, min_confidence, expected_flags
