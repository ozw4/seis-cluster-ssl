from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.models.vicreg import VICRegLoss, vicreg_loss

if TYPE_CHECKING:
	from collections.abc import Callable


def test_identical_inputs_have_zero_invariance_loss() -> None:
	projection = torch.randn((16, 8), generator=torch.Generator().manual_seed(7))

	result = vicreg_loss(projection, projection)

	assert result['invariance_loss'].item() == pytest.approx(0.0)


def test_constant_inputs_have_positive_variance_penalty() -> None:
	projection = torch.ones((8, 4))

	result = vicreg_loss(projection, projection)

	assert result['variance_loss'].item() > 0.0
	assert result['variance_loss_a'].item() == pytest.approx(
		result['variance_loss_b'].item()
	)


def test_distributed_features_satisfy_variance_target() -> None:
	projection = 2.0 * torch.randn(
		(4096, 8),
		generator=torch.Generator().manual_seed(11),
	)

	result = vicreg_loss(projection, projection)

	assert result['variance_loss'].item() == pytest.approx(0.0, abs=1.0e-7)


def test_correlated_features_have_positive_covariance_penalty() -> None:
	base = torch.linspace(-2.0, 2.0, steps=32).unsqueeze(1)
	projection = base.repeat(1, 4)

	result = vicreg_loss(projection, projection)

	assert result['covariance_loss'].item() > 0.0
	assert result['covariance_offdiag_rms'].item() > 0.0


def test_total_matches_weighted_objective_terms() -> None:
	generator = torch.Generator().manual_seed(13)
	z_a = torch.randn((32, 6), generator=generator)
	z_b = torch.randn((32, 6), generator=generator)

	result = vicreg_loss(
		z_a,
		z_b,
		invariance_weight=2.0,
		variance_weight=3.0,
		covariance_weight=4.0,
	)

	expected = (
		2.0 * result['invariance_loss']
		+ 3.0 * result['variance_loss']
		+ 4.0 * result['covariance_loss']
	)
	assert torch.allclose(result['loss'], expected)
	assert torch.allclose(
		result['loss'],
		result['weighted_invariance']
		+ result['weighted_variance']
		+ result['weighted_covariance'],
	)


def test_module_backward_produces_finite_gradients() -> None:
	generator = torch.Generator().manual_seed(17)
	z_a = torch.randn((16, 5), generator=generator, requires_grad=True)
	z_b = torch.randn((16, 5), generator=generator, requires_grad=True)

	result = VICRegLoss()(z_a, z_b)
	result['loss'].backward()

	assert z_a.grad is not None
	assert z_b.grad is not None
	assert bool(torch.isfinite(z_a.grad).all())
	assert bool(torch.isfinite(z_b.grad).all())


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_inputs_are_computed_in_float32(dtype: torch.dtype) -> None:
	generator = torch.Generator().manual_seed(19)
	z_a = torch.randn((16, 4), generator=generator).to(dtype).requires_grad_()
	z_b = torch.randn((16, 4), generator=generator).to(dtype).requires_grad_()

	with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
		result = vicreg_loss(z_a, z_b)
	result['loss'].backward()

	assert all(value.dtype == torch.float32 for value in result.values())
	assert z_a.grad is not None
	assert z_b.grad is not None
	assert bool(torch.isfinite(z_a.grad).all())
	assert bool(torch.isfinite(z_b.grad).all())


@pytest.mark.parametrize(
	'build_inputs',
	[
		lambda: (torch.randn((3, 2)), torch.randn((3, 3))),
		lambda: (torch.randn((2, 2, 2)), torch.randn((2, 2, 2))),
		lambda: (torch.randn((1, 2)), torch.randn((1, 2))),
		lambda: (torch.empty((2, 0)), torch.empty((2, 0))),
		lambda: (torch.ones((2, 2), dtype=torch.int64), torch.randn((2, 2))),
	],
)
def test_rejects_invalid_inputs(
	build_inputs: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> None:
	z_a, z_b = build_inputs()

	with pytest.raises((TypeError, ValueError)):
		vicreg_loss(z_a, z_b)


def test_rejects_inputs_on_different_devices() -> None:
	z_a = torch.randn((2, 2))
	z_b = torch.empty((2, 2), device='meta')

	with pytest.raises(ValueError, match='same device'):
		vicreg_loss(z_a, z_b)


@pytest.mark.parametrize(
	('name', 'value'),
	[
		('invariance_weight', -1.0),
		('invariance_weight', float('inf')),
		('variance_weight', True),
		('covariance_weight', float('nan')),
		('variance_target_std', 0.0),
		('variance_target_std', False),
		('variance_eps', 0.0),
		('variance_eps', float('inf')),
	],
)
def test_rejects_invalid_parameters(name: str, value: object) -> None:
	projection = torch.randn((4, 3))
	kwargs = {name: value}

	with pytest.raises((TypeError, ValueError), match=name):
		vicreg_loss(projection, projection, **kwargs)


def test_does_not_modify_inputs_in_place() -> None:
	generator = torch.Generator().manual_seed(23)
	z_a = torch.randn((8, 3), generator=generator, requires_grad=True)
	z_b = torch.randn((8, 3), generator=generator, requires_grad=True)
	expected_a = z_a.detach().clone()
	expected_b = z_b.detach().clone()

	vicreg_loss(z_a, z_b)['loss'].backward()

	assert torch.equal(z_a.detach(), expected_a)
	assert torch.equal(z_b.detach(), expected_b)


def test_single_feature_has_zero_covariance_penalty() -> None:
	projection = torch.arange(4, dtype=torch.float32).unsqueeze(1)

	result = vicreg_loss(projection, projection)

	assert result['covariance_loss'].item() == pytest.approx(0.0)
	assert result['covariance_offdiag_rms'].item() == pytest.approx(0.0)
