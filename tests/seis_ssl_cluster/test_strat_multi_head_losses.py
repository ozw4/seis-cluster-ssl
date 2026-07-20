"""Regression coverage for multi-resolution stratigraphic loss aggregation."""

from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.stratigraphy import (
	MultiResolutionOrderedPrototypeHeads,
	MultiResolutionOrderedPrototypeOutput,
	OrderedPrototypeOutput,
)
from seis_ssl_cluster.training.strat_hmm import (
	compute_strat_hmm_multi_head_losses,
)


def test_multi_head_losses_average_heads_and_apply_total_weights() -> None:
	torch.manual_seed(24)
	heads = _heads()
	tokens = torch.randn(1, 4, 3, requires_grad=True)
	teacher_tokens = torch.randn(1, 4, 3, requires_grad=True)
	batch = _batch()
	losses = compute_strat_hmm_multi_head_losses(
		heads=heads,
		encoded={'tokens': tokens},
		teacher_encoded={'tokens': teacher_tokens},
		batch=batch,
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.2,
			'consistency_weight': 0.1,
			'consistency_beta': 0.1,
			'distillation_weight': 0.3,
		},
		pseudo_target_config={'min_confidence': 0.0},
	)

	assert torch.allclose(
		losses['loss_prototype'],
		torch.stack([losses[f'loss_prototype_k{k}'] for k in (6, 8, 10)]).mean(),
	)
	assert torch.allclose(
		losses['loss_usage'],
		torch.stack([losses[f'loss_usage_k{k}'] for k in (6, 8, 10)]).mean(),
	)
	expected = (
		losses['loss_prototype']
		+ 0.2 * losses['loss_usage']
		+ 0.1 * losses['loss_consistency']
		+ 0.3 * losses['loss_distillation']
	)
	assert torch.allclose(losses['loss'], expected)
	assert losses['consistency_eligible_pair_count'].item() == 3

	losses['loss'].backward()
	assert all(
		parameter.grad is not None and torch.isfinite(parameter.grad).all()
		for parameter in heads.parameters()
	)
	assert teacher_tokens.grad is None


def test_consistency_uses_normalized_order_not_direct_state_ids() -> None:
	logits_k6 = torch.full((1, 1, 6), -100.0, requires_grad=True)
	logits_k11 = torch.full((1, 1, 11), -100.0, requires_grad=True)
	with torch.no_grad():
		logits_k6[..., 3] = 100.0
		logits_k11[..., 6] = 100.0
	heads = _StaticHeads({'k6': logits_k6, 'k11': logits_k11})
	batch = {
		'strat_multi_targets': {
			f'k{k}': {
				'labels': torch.zeros((1, 1), dtype=torch.long),
				'confidence': torch.ones((1, 1)),
				'boundary_weight': torch.ones((1, 1)),
				'valid_mask': torch.ones((1, 1), dtype=torch.bool),
			}
			for k in (6, 11)
		},
	}
	losses = compute_strat_hmm_multi_head_losses(
		heads=heads,  # type: ignore[arg-type]
		encoded={'tokens': torch.zeros((1, 1, 1))},
		teacher_encoded=None,
		batch=batch,
		loss_config={
			'prototype_weight': 0.0,
			'usage_weight': 0.0,
			'consistency_weight': 1.0,
			'consistency_beta': 0.1,
			'distillation_weight': 0.0,
		},
		pseudo_target_config={'min_confidence': 0.0},
	)

	assert losses['loss_consistency'].item() == pytest.approx(0.0, abs=1.0e-6)


def test_consistency_pair_metric_keys_are_ordered_by_k() -> None:
	logits_k10 = torch.zeros((1, 1, 10))
	logits_k6 = torch.zeros((1, 1, 6))
	heads = _StaticHeads({'k10': logits_k10, 'k6': logits_k6})
	batch = {
		'strat_multi_targets': {
			f'k{k}': {
				'labels': torch.zeros((1, 1), dtype=torch.long),
				'confidence': torch.ones((1, 1)),
				'boundary_weight': torch.ones((1, 1)),
				'valid_mask': torch.ones((1, 1), dtype=torch.bool),
			}
			for k in (10, 6)
		},
	}

	losses = compute_strat_hmm_multi_head_losses(
		heads=heads,  # type: ignore[arg-type]
		encoded={'tokens': torch.zeros((1, 1, 1))},
		teacher_encoded=None,
		batch=batch,
		loss_config={'consistency_weight': 1.0},
		pseudo_target_config={'min_confidence': 0.0},
	)

	assert 'loss_consistency_k6_k10' in losses
	assert 'loss_consistency_k10_k6' not in losses


def test_multi_head_rejects_invalid_valid_label_when_prototype_loss_is_disabled(
) -> None:
	heads = _heads()
	batch = _batch()
	batch['strat_multi_targets']['k6']['labels'][0, 0] = 6  # type: ignore[index]

	with pytest.raises(ValueError, match=r"multi-head 'k6' valid labels"):
		compute_strat_hmm_multi_head_losses(
			heads=heads,
			encoded={'tokens': torch.zeros((1, 4, 3))},
			teacher_encoded=None,
			batch=batch,
			loss_config={
				'prototype_weight': 0.0,
				'usage_weight': 0.0,
				'consistency_weight': 0.0,
			},
			pseudo_target_config={'min_confidence': 0.0},
		)


def test_multi_head_confidence_is_stop_gradient_input() -> None:
	logits_k6 = torch.randn((1, 2, 6), requires_grad=True)
	logits_k8 = torch.randn((1, 2, 8), requires_grad=True)
	confidences = {
		'k6': torch.tensor([[1.0, 0.5]], requires_grad=True),
		'k8': torch.tensor([[0.8, 0.4]], requires_grad=True),
	}
	batch = {
		'strat_multi_targets': {
			key: {
				'labels': torch.zeros((1, 2), dtype=torch.long),
				'confidence': confidence,
				'boundary_weight': torch.ones((1, 2)),
				'valid_mask': torch.ones((1, 2), dtype=torch.bool),
			}
			for key, confidence in confidences.items()
		},
	}
	losses = compute_strat_hmm_multi_head_losses(
		heads=_StaticHeads({'k6': logits_k6, 'k8': logits_k8}),  # type: ignore[arg-type]
		encoded={'tokens': torch.zeros((1, 2, 1))},
		teacher_encoded=None,
		batch=batch,
		loss_config={
			'prototype_weight': 1.0,
			'consistency_weight': 1.0,
		},
		pseudo_target_config={'min_confidence': 0.0},
	)

	losses['loss'].backward()

	assert logits_k6.grad is not None
	assert logits_k8.grad is not None
	assert all(confidence.grad is None for confidence in confidences.values())


def _heads() -> MultiResolutionOrderedPrototypeHeads:
	return MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.5,
		normalize=True,
	)


def _batch() -> dict[str, object]:
	return {
		'strat_multi_targets': {
			f'k{k}': {
				'labels': torch.tensor([[0, 1, 2, 3]]),
				'confidence': torch.tensor([[1.0, 0.8, 0.6, 0.4]]),
				'boundary_weight': torch.ones((1, 4)),
				'valid_mask': torch.ones((1, 4), dtype=torch.bool),
			}
			for k in (6, 8, 10)
		},
	}


class _StaticHeads:
	def __init__(self, logits_by_key: dict[str, torch.Tensor]) -> None:
		self.head_ks = tuple(int(key[1:]) for key in logits_by_key)
		self._logits_by_key = logits_by_key

	def __call__(self, _tokens: torch.Tensor) -> MultiResolutionOrderedPrototypeOutput:
		return MultiResolutionOrderedPrototypeOutput(
			outputs={
				key: OrderedPrototypeOutput(
					logits=logits,
					projected_features=logits[..., :1],
				)
				for key, logits in self._logits_by_key.items()
			},
			head_ks=self.head_ks,
		)
