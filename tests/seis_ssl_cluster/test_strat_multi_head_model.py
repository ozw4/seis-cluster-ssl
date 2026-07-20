from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.stratigraphy import (
	MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1,
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
	expected_normalized_order_coordinate,
)


def test_multi_resolution_heads_forward_and_state_dict_contract() -> None:
	model = MultiResolutionOrderedPrototypeHeads(
		feature_dim=4,
		ks=(6, 8, 10),
		projection_dim=3,
		temperature=0.5,
		normalize=True,
	)
	features = torch.randn(2, 3, 4)

	output = model(features)

	assert MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1 == (
		'multi_resolution_ordered_prototypes_v1'
	)
	assert output.head_ks == (6, 8, 10)
	assert tuple(output.outputs) == ('k6', 'k8', 'k10')
	assert output.outputs['k6'].logits.shape == (2, 3, 6)
	assert output.outputs['k8'].logits.shape == (2, 3, 8)
	assert output.outputs['k10'].logits.shape == (2, 3, 10)
	assert all(
		item.projected_features.shape == (2, 3, 3)
		for item in output.outputs.values()
	)
	assert all(item.logits.dtype == features.dtype for item in output.outputs.values())
	assert all(
		item.logits.device == features.device for item in output.outputs.values()
	)
	assert set(model.state_dict()) == {
		'heads.k6.projection.weight',
		'heads.k6.projection.bias',
		'heads.k6.prototypes',
		'heads.k8.projection.weight',
		'heads.k8.projection.bias',
		'heads.k8.prototypes',
		'heads.k10.projection.weight',
		'heads.k10.projection.bias',
		'heads.k10.prototypes',
	}


def test_multi_resolution_heads_are_independent_and_match_standalone_heads() -> None:
	model = MultiResolutionOrderedPrototypeHeads(
		feature_dim=4,
		ks=(6, 8, 10),
		projection_dim=3,
		temperature=0.5,
		normalize=True,
	)
	features = torch.randn(2, 4)
	output = model(features)

	assert model.heads['k6'].prototypes is not model.heads['k8'].prototypes
	assert (
		model.heads['k8'].projection.weight
		is not model.heads['k10'].projection.weight
	)
	assert sum(parameter.numel() for parameter in model.parameters()) == sum(
		sum(parameter.numel() for parameter in head.parameters())
		for head in model.heads.values()
	)
	for k in output.head_ks:
		key = f'k{k}'
		standalone = OrderedPrototypeHead(
			feature_dim=4,
			num_prototypes=k,
			projection_dim=3,
			temperature=0.5,
			normalize=True,
		)
		standalone.load_state_dict(model.heads[key].state_dict())
		standalone_output = standalone(features)
		assert torch.equal(output.outputs[key].logits, standalone_output.logits)
		assert torch.equal(
			output.outputs[key].projected_features,
			standalone_output.projected_features,
		)


@pytest.mark.parametrize('ks', [(8, 6, 10), (6, 6, 10), (True, 8), (1, 8), (6,)])
def test_multi_resolution_heads_reject_invalid_ks(ks: tuple[object, ...]) -> None:
	with pytest.raises((TypeError, ValueError), match='ks'):
		MultiResolutionOrderedPrototypeHeads(
			feature_dim=4,
			ks=ks,  # type: ignore[arg-type]
			projection_dim=None,
			temperature=0.1,
			normalize=True,
		)


def test_multi_resolution_heads_have_deterministic_initial_state_and_strict_load(
) -> None:
	torch.manual_seed(10)
	first = MultiResolutionOrderedPrototypeHeads(
		feature_dim=4,
		ks=(6, 8, 10),
		projection_dim=3,
		temperature=0.1,
		normalize=True,
	)
	torch.manual_seed(10)
	second = MultiResolutionOrderedPrototypeHeads(
		feature_dim=4,
		ks=(6, 8, 10),
		projection_dim=3,
		temperature=0.1,
		normalize=True,
	)

	assert all(
		torch.equal(first_state, second.state_dict()[key])
		for key, first_state in first.state_dict().items()
	)
	state = first.state_dict()
	state.pop('heads.k10.prototypes')
	with pytest.raises(RuntimeError):
		second.load_state_dict(state, strict=True)
	with pytest.raises(RuntimeError):
		first.load_state_dict(
			OrderedPrototypeHead(feature_dim=4, num_prototypes=6).state_dict(),
		)


def test_expected_normalized_order_coordinate_contract_and_gradient() -> None:
	logits = torch.tensor(
		[
			[[100.0, -100.0, -100.0], [-100.0, -100.0, 100.0]],
			[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
		],
		requires_grad=True,
	)

	coordinate = expected_normalized_order_coordinate(logits)

	assert coordinate.shape == (2, 2)
	assert torch.allclose(coordinate[0], torch.tensor([0.0, 1.0]))
	assert coordinate[1, 0].item() == pytest.approx(0.5)
	assert bool(coordinate.ge(0.0).all())
	assert bool(coordinate.le(1.0).all())
	coordinate.sum().backward()
	assert logits.grad is not None
	assert bool(logits.grad.abs().sum().gt(0).item())


@pytest.mark.parametrize(
	'logits',
	[
		torch.tensor(1.0),
		torch.ones(2, 1),
		torch.ones(2, 3, dtype=torch.int64),
		torch.tensor([[0.0, float('inf')]]),
	],
)
def test_expected_normalized_order_coordinate_rejects_invalid_logits(
	logits: torch.Tensor,
) -> None:
	with pytest.raises((TypeError, ValueError)):
		expected_normalized_order_coordinate(logits)
