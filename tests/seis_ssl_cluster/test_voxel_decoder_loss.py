"""Tests for masked class-balanced voxel cross entropy."""

from __future__ import annotations

import pytest
import torch
from torch.nn.functional import cross_entropy

from seis_ssl_cluster.training.voxel_decoder.losses import (
	balanced_class_weights_from_counts,
	masked_weighted_voxel_cross_entropy,
)


def test_weighted_cross_entropy_uses_weight_sum_denominator() -> None:
	logits = torch.tensor(
		[[[[[2.0]], [[-1.0]], [[0.5]]], [[[0.0]], [[1.0]], [[-0.5]]]]]
	)
	labels = torch.tensor([[[[0]], [[1]], [[0]]]])
	mask = torch.ones_like(labels, dtype=torch.bool)
	weights = torch.tensor([1.0, 3.0])
	per_voxel = cross_entropy(
		logits.movedim(1, -1).reshape(-1, 2), labels.reshape(-1), reduction='none'
	)
	expected = (per_voxel * torch.tensor([1.0, 3.0, 1.0])).sum() / 5.0

	loss, summary = masked_weighted_voxel_cross_entropy(
		logits, labels, mask, weights
	)

	torch.testing.assert_close(loss, expected)
	assert summary['class_weight_sum'] == pytest.approx(5.0)
	assert summary['supervised_voxel_count'] == 3
	assert summary['class_0_count'] == 2
	assert summary['class_1_count'] == 1


def test_masked_values_do_not_affect_loss() -> None:
	logits = torch.tensor([[[[[1.0]], [[2.0]]], [[[-1.0]], [[0.0]]]]])
	labels = torch.tensor([[[[0]], [[-1]]]])
	mask = torch.tensor([[[[True]], [[False]]]])
	weights = torch.ones(2)
	expected, _ = masked_weighted_voxel_cross_entropy(logits, labels, mask, weights)
	changed = logits.clone()
	changed[:, :, 1] = torch.tensor([torch.nan, torch.inf]).reshape(1, 2, 1, 1)
	changed_labels = labels.clone()
	changed_labels[:, 1] = 999

	actual, _ = masked_weighted_voxel_cross_entropy(
		changed, changed_labels, mask, weights
	)

	torch.testing.assert_close(actual, expected)


def test_balanced_weights_reject_zero_train_class() -> None:
	weights = balanced_class_weights_from_counts([2, 6])
	torch.testing.assert_close(weights, torch.tensor([2.0, 2.0 / 3.0]))
	with pytest.raises(ValueError, match='zero-count'):
		balanced_class_weights_from_counts([2, 0])


@pytest.mark.parametrize('failure', ['empty', 'unknown', 'shape', 'nan'])
def test_invalid_loss_inputs_fail_fast(failure: str) -> None:
	logits = torch.zeros(1, 2, 2, 1, 1)
	labels = torch.zeros(1, 2, 1, 1, dtype=torch.long)
	mask = torch.ones_like(labels, dtype=torch.bool)
	if failure == 'empty':
		mask[:] = False
	elif failure == 'unknown':
		labels[0, 0, 0, 0] = 2
	elif failure == 'shape':
		labels = labels[:, :1]
	else:
		logits[0, 0, 0, 0, 0] = torch.nan

	with pytest.raises((ValueError, FloatingPointError)):
		masked_weighted_voxel_cross_entropy(logits, labels, mask, torch.ones(2))
