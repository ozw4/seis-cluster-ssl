"""Coverage for soft multi-head posterior supervision."""

from __future__ import annotations

import numpy as np
import torch

from seis_ssl_cluster.stratigraphy import MultiResolutionOrderedPrototypeHeads
from seis_ssl_cluster.stratigraphy.losses import soft_categorical_cross_entropy
from seis_ssl_cluster.training.collate import (
	move_batch_to_device,
	strat_multi_head_posterior_collate_fn,
)
from seis_ssl_cluster.training.strat_hmm.losses import (
	compute_strat_hmm_multi_head_posterior_losses,
)


def test_soft_categorical_cross_entropy_is_detached_and_graph_safe() -> None:
	logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
	target = torch.tensor(
		[[[1.0, 0.0], [0.0, 0.0]]], requires_grad=True
	)
	valid_mask = torch.tensor([[True, False]])

	loss = soft_categorical_cross_entropy(logits, target, valid_mask=valid_mask)
	loss.backward()

	expected = -torch.nn.functional.log_softmax(logits, -1)[0, 0, 0]
	assert torch.allclose(loss.detach(), expected)
	assert target.grad is None
	assert logits.grad is not None
	empty = soft_categorical_cross_entropy(
		logits,
		torch.zeros_like(target),
		valid_mask=torch.zeros_like(valid_mask),
	)
	empty.backward()


def test_posterior_collate_stacks_nested_targets_and_moves_them() -> None:
	batch = strat_multi_head_posterior_collate_fn([_sample(), _sample()])
	moved = move_batch_to_device(batch, torch.device('cpu'))

	assert list(batch['strat_multi_posteriors']) == ['k6', 'k8', 'k10']
	assert batch['strat_multi_posteriors']['k8']['posterior'].shape == (2, 1, 2, 2, 8)
	assert moved['strat_multi_posteriors']['k10']['valid_mask'].device.type == 'cpu'


def test_posterior_losses_average_heads_without_consistency() -> None:
	torch.manual_seed(289)
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	tokens = torch.randn(1, 4, 3, requires_grad=True)
	batch = {
		'strat_multi_posteriors': {
			f'k{k}': {
				'posterior': torch.nn.functional.one_hot(
					torch.zeros((1, 4), dtype=torch.long), k
				).float(),
				'valid_mask': torch.ones((1, 4), dtype=torch.bool),
			}
			for k in (6, 8, 10)
		},
	}
	result = compute_strat_hmm_multi_head_posterior_losses(
		heads=heads,
		encoded={
			'tokens': tokens,
			'token_valid_mask': torch.tensor([[True, False, True, False]]),
		},
		teacher_encoded=None,
		batch=batch,
		loss_config={'usage_weight': 0.1},
	)

	assert torch.allclose(
		result['loss_prototype'],
		torch.stack([result[f'loss_prototype_k{k}'] for k in (6, 8, 10)]).mean(),
	)
	assert {'target_entropy_k6', 'prototype_kl_k8', 'loss_distillation'} <= set(result)
	result['loss'].backward()
	assert tokens.grad is not None


def _sample() -> dict[str, object]:
	valid_mask = np.ones((1, 2, 2), dtype=np.bool_)
	return {
		'x': np.ones((1, 2, 2, 2), dtype=np.float32),
		'local_valid_mask': np.ones((2, 2, 2), dtype=np.bool_),
		'strat_multi_posteriors': {
			f'k{k}': {
				'posterior': np.full((1, 2, 2, k), 1.0 / k, dtype=np.float32),
				'valid_mask': valid_mask,
			}
			for k in (6, 8, 10)
		},
		'coords': {'survey_id': 'survey-a'},
	}
