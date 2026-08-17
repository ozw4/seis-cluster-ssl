"""Focused tests for the 3D Barlow Twins model and loss core."""

from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.models.barlow_twins import (
	BarlowTwins3D,
	barlow_twins_loss,
	mean_pool_encoded_tokens,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D


def _make_model(*, projector_dim: int = 6) -> BarlowTwins3D:
	backbone = AmplitudeMAE3D(
		patch_size_xyz=(2, 2, 2),
		encoder_dim=8,
		encoder_depth=1,
		encoder_heads=2,
		decoder_dim=4,
		decoder_depth=1,
		decoder_heads=1,
		runtime_check_mode='strict',
	)
	return BarlowTwins3D(backbone, projector_dim=projector_dim)


def test_two_views_return_projection_shapes() -> None:
	model = _make_model()
	view_a = torch.randn((2, 1, 4, 4, 4))
	view_b = torch.randn((2, 1, 4, 4, 4))

	output = model(view_a, view_b)

	assert output['z_a'].shape == (2, 6)
	assert output['z_b'].shape == (2, 6)


def test_mean_pool_excludes_invalid_tokens() -> None:
	tokens = torch.tensor(
		[
			[[1.0, 3.0], [1000.0, 1000.0], [5.0, 7.0]],
			[[2.0, 4.0], [6.0, 8.0], [-1000.0, -1000.0]],
		],
	)
	valid_mask = torch.tensor(
		[[True, False, True], [True, True, False]],
	)

	pooled = mean_pool_encoded_tokens(tokens, valid_mask)

	torch.testing.assert_close(pooled, torch.tensor([[3.0, 5.0], [4.0, 6.0]]))


def test_mean_pool_rejects_sample_without_valid_tokens() -> None:
	tokens = torch.randn((2, 3, 4))
	valid_mask = torch.tensor(
		[[True, False, False], [False, False, False]],
	)

	with pytest.raises(ValueError, match=r'each sample.*valid token'):
		mean_pool_encoded_tokens(tokens, valid_mask)


def test_loss_is_finite_and_has_finite_gradients() -> None:
	z_a = torch.randn((4, 5), requires_grad=True)
	z_b = torch.randn((4, 5), requires_grad=True)

	result = barlow_twins_loss(z_a, z_b)
	result['loss'].backward()

	assert result['loss'].ndim == 0
	assert torch.isfinite(result['loss'])
	assert z_a.grad is not None
	assert z_b.grad is not None
	assert torch.isfinite(z_a.grad).all()
	assert torch.isfinite(z_b.grad).all()


def test_loss_matches_reported_decomposition() -> None:
	weight = 0.03
	result = barlow_twins_loss(
		torch.randn((4, 3)),
		torch.randn((4, 3)),
		redundancy_weight=weight,
	)

	torch.testing.assert_close(
		result['loss'],
		result['on_diag'] + weight * result['off_diag'],
	)


def test_identity_like_cross_correlation_beats_redundant_features() -> None:
	identity_like = torch.tensor(
		[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
	)
	redundant = identity_like[:, :1].expand(-1, 2).clone()

	identity_loss = barlow_twins_loss(identity_like, identity_like)['loss']
	redundant_loss = barlow_twins_loss(redundant, redundant)['loss']

	assert identity_loss < redundant_loss


def test_only_encoder_path_and_projector_are_trainable_and_receive_gradients() -> None:
	model = _make_model(projector_dim=4)
	view_a = torch.randn((3, 1, 4, 4, 4))
	view_b = torch.randn((3, 1, 4, 4, 4))
	output = model(view_a, view_b)

	barlow_twins_loss(output['z_a'], output['z_b'])['loss'].backward()

	trainable_names = {
		name for name, parameter in model.named_parameters() if parameter.requires_grad
	}
	assert any(
		name.startswith('backbone.patch_projection.') for name in trainable_names
	)
	assert any(name.startswith('backbone.encoder.') for name in trainable_names)
	assert any(name.startswith('projector.') for name in trainable_names)
	assert not any(
		name.startswith('backbone.encoder_to_decoder.')
		for name in trainable_names
	)
	assert 'backbone.mask_token' not in trainable_names
	assert not any(name.startswith('backbone.decoder.') for name in trainable_names)
	assert not any(
		name.startswith('backbone.prediction_head.') for name in trainable_names
	)
	assert model.backbone.patch_projection.weight.grad is not None
	assert any(
		parameter.grad is not None for parameter in model.backbone.encoder.parameters()
	)
	assert all(
		parameter.grad is None for parameter in model.backbone.decoder.parameters()
	)
	assert set(model.pretraining_parameters()) == {
		parameter for parameter in model.parameters() if parameter.requires_grad
	}


def test_loss_rejects_batch_size_one() -> None:
	with pytest.raises(ValueError, match='batch size at least 2'):
		barlow_twins_loss(torch.randn((1, 4)), torch.randn((1, 4)))
