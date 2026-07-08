from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional

from seis_ssl_cluster.stratigraphy import (
	OrderedPrototypeHead,
	feature_distillation_loss,
	ordered_soft_coordinate,
	structured_hmm_prototype_loss,
	usage_entropy_floor_loss,
)


def test_ordered_prototype_head_accepts_token_and_grid_features() -> None:
	head = OrderedPrototypeHead(feature_dim=4, num_prototypes=3, temperature=0.5)

	token_output = head(torch.randn(2, 5, 4))
	grid_output = head(torch.randn(2, 3, 2, 1, 4))

	assert token_output.logits.shape == (2, 5, 3)
	assert token_output.projected_features.shape == (2, 5, 4)
	assert grid_output.logits.shape == (2, 3, 2, 1, 3)
	assert grid_output.projected_features.shape == (2, 3, 2, 1, 4)


def test_ordered_prototype_head_projection_and_identity_modes() -> None:
	projected = OrderedPrototypeHead(
		feature_dim=4,
		num_prototypes=5,
		projection_dim=2,
	)
	identity = OrderedPrototypeHead(feature_dim=4, num_prototypes=5)
	features = torch.randn(2, 3, 4)

	projected_output = projected(features)
	identity_output = identity(features)

	assert projected_output.projected_features.shape == (2, 3, 2)
	assert projected_output.logits.shape == (2, 3, 5)
	assert identity_output.projected_features.shape == (2, 3, 4)
	assert identity_output.logits.shape == (2, 3, 5)


@pytest.mark.parametrize(
	'kwargs',
	[
		{'feature_dim': 0, 'num_prototypes': 3},
		{'feature_dim': 4, 'num_prototypes': 0},
		{'feature_dim': 4, 'num_prototypes': 3, 'projection_dim': 0},
		{'feature_dim': 4, 'num_prototypes': 3, 'temperature': 0.0},
		{'feature_dim': 4, 'num_prototypes': 3, 'temperature': math.inf},
	],
)
def test_ordered_prototype_head_rejects_invalid_constructor_args(
	kwargs: dict[str, object],
) -> None:
	with pytest.raises((TypeError, ValueError)):
		OrderedPrototypeHead(**kwargs)


def test_structured_hmm_prototype_loss_ignores_invalid_and_zero_confidence() -> None:
	logits = torch.tensor(
		[
			[[3.0, 0.0], [0.0, 3.0], [4.0, 0.0]],
			[[0.0, 4.0], [2.0, 0.0], [0.0, 2.0]],
		],
	)
	labels = torch.tensor([[0, -1, 1], [1, 0, 1]])
	valid_mask = torch.tensor([[True, True, False], [True, True, True]])
	confidence = torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.0, 1.0]])

	loss = structured_hmm_prototype_loss(
		logits,
		labels,
		valid_mask=valid_mask,
		confidence=confidence,
	)

	token_loss = torch.nn.functional.cross_entropy(
		torch.stack([logits[0, 0], logits[1, 0], logits[1, 2]]),
		torch.tensor([0, 1, 1]),
		reduction='none',
	)
	expected = (token_loss * torch.tensor([1.0, 0.5, 1.0])).sum() / 2.5
	assert torch.allclose(loss, expected)


@pytest.mark.parametrize(
	('valid_mask', 'confidence'),
	[
		(torch.zeros(2, 3, dtype=torch.bool), None),
		(torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3)),
	],
)
def test_structured_hmm_prototype_loss_raises_without_positive_weight_tokens(
	valid_mask: torch.Tensor,
	confidence: torch.Tensor | None,
) -> None:
	logits = torch.randn(2, 3, 2)
	labels = torch.zeros(2, 3, dtype=torch.long)

	with pytest.raises(ValueError, match='target tokens'):
		structured_hmm_prototype_loss(
			logits,
			labels,
			valid_mask=valid_mask,
			confidence=confidence,
		)


def test_usage_entropy_floor_loss_penalizes_only_low_entropy_usage() -> None:
	balanced_probs = torch.tensor(
		[
			[[0.5, 0.5], [0.5, 0.5]],
			[[0.5, 0.5], [0.5, 0.5]],
		],
	)
	collapsed_probs = torch.tensor(
		[
			[[1.0, 0.0], [1.0, 0.0]],
			[[1.0, 0.0], [1.0, 0.0]],
		],
	)
	valid_mask = torch.ones(2, 2, dtype=torch.bool)

	balanced_loss = usage_entropy_floor_loss(
		balanced_probs,
		valid_mask=valid_mask,
		entropy_floor=0.25,
	)
	collapsed_loss = usage_entropy_floor_loss(
		collapsed_probs,
		valid_mask=valid_mask,
		entropy_floor=0.25,
	)

	assert balanced_loss.item() == pytest.approx(0.0)
	assert collapsed_loss.item() > 0.0


def test_feature_distillation_loss_detaches_teacher_features() -> None:
	student = torch.randn(2, 3, 4, requires_grad=True)
	teacher = student.detach().clone().requires_grad_()
	valid_mask = torch.tensor([[True, True, False], [True, False, True]])

	loss = feature_distillation_loss(student, teacher, valid_mask=valid_mask)
	loss.backward()

	assert loss.item() == pytest.approx(0.0, abs=1.0e-6)
	assert student.grad is not None
	assert teacher.grad is None


def test_ordered_soft_coordinate_preserves_prefix_and_range() -> None:
	probs = torch.tensor(
		[
			[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
			[[0.0, 0.0, 1.0], [0.25, 0.5, 0.25]],
		],
	)

	coordinates = ordered_soft_coordinate(probs)
	single = ordered_soft_coordinate(torch.ones(2, 3, 1))

	assert coordinates.shape == (2, 2)
	assert bool(coordinates.ge(0.0).all().item())
	assert bool(coordinates.le(1.0).all().item())
	assert torch.allclose(coordinates, torch.tensor([[0.0, 0.5], [1.0, 0.5]]))
	assert single.shape == (2, 3)
	assert bool(single.eq(0.0).all().item())
