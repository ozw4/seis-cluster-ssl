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


def _local_inputs() -> tuple[
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
]:
	view_a = torch.randn((2, 1, 4, 4, 4))
	view_b = torch.randn((2, 1, 4, 4, 4))
	valid_mask_a = torch.ones((2, 4, 4, 4), dtype=torch.bool)
	valid_mask_b = torch.ones((2, 4, 4, 4), dtype=torch.bool)
	indices_a = torch.tensor([[0, 3, 7], [1, 4, 6]], dtype=torch.int64)
	indices_b = torch.tensor([[6, 2, 0], [7, 3, 1]], dtype=torch.int64)
	return (
		view_a,
		view_b,
		valid_mask_a,
		valid_mask_b,
		indices_a,
		indices_b,
	)


def _call_forward_local(  # noqa: PLR0913
	model: BarlowTwins3D,
	*,
	view_a: torch.Tensor,
	view_b: torch.Tensor,
	valid_mask_a: torch.Tensor,
	valid_mask_b: torch.Tensor,
	indices_a: torch.Tensor,
	indices_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
	return model.forward_local(
		view_a,
		view_b,
		valid_mask_a=valid_mask_a,
		valid_mask_b=valid_mask_b,
		local_pair_indices_a=indices_a,
		local_pair_indices_b=indices_b,
	)


def test_two_views_return_projection_shapes() -> None:
	model = _make_model()
	view_a = torch.randn((2, 1, 4, 4, 4))
	view_b = torch.randn((2, 1, 4, 4, 4))

	output = model(view_a, view_b)

	assert output['z_a'].shape == (2, 6)
	assert output['z_b'].shape == (2, 6)


def test_local_views_project_batch_times_pair_rows() -> None:
	model = _make_model()
	view_a, view_b, mask_a, mask_b, indices_a, indices_b = _local_inputs()

	output = _call_forward_local(
		model,
		view_a=view_a,
		view_b=view_b,
		valid_mask_a=mask_a,
		valid_mask_b=mask_b,
		indices_a=indices_a,
		indices_b=indices_b,
	)

	assert set(output) == {'z_a', 'z_b'}
	assert output['z_a'].shape == (6, 6)
	assert output['z_b'].shape == (6, 6)


def test_local_view_specific_indices_gather_corresponding_tokens(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model().eval()
	projector = model.projector
	view_a, view_b, mask_a, mask_b, indices_a, indices_b = _local_inputs()
	tokens_a = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8)
	tokens_b = torch.full_like(tokens_a, -1.0)
	for batch_index in range(2):
		tokens_b[batch_index, indices_b[batch_index]] = tokens_a[
			batch_index,
			indices_a[batch_index],
		]
	encoded_outputs = iter(
		(
			{'tokens': tokens_a},
			{'tokens': tokens_b},
		)
	)

	def fake_encode_tokens(
		_view: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> dict[str, object]:
		assert valid_mask is not None
		return next(encoded_outputs)

	projector_inputs: list[torch.Tensor] = []

	def capture_projector_input(
		_module: torch.nn.Module,
		inputs: tuple[torch.Tensor, ...],
	) -> None:
		projector_inputs.append(inputs[0].detach().clone())

	monkeypatch.setattr(model.backbone, 'encode_tokens', fake_encode_tokens)
	handle = projector.register_forward_pre_hook(capture_projector_input)
	output = _call_forward_local(
		model,
		view_a=view_a,
		view_b=view_b,
		valid_mask_a=mask_a,
		valid_mask_b=mask_b,
		indices_a=indices_a,
		indices_b=indices_b,
	)
	handle.remove()

	expected = torch.gather(
		tokens_a,
		1,
		indices_a.unsqueeze(-1).expand(-1, -1, tokens_a.shape[2]),
	).reshape(6, 8)
	assert model.projector is projector
	assert not any('local_projector' in name for name, _module in model.named_modules())
	assert len(projector_inputs) == 2
	torch.testing.assert_close(projector_inputs[0], expected)
	torch.testing.assert_close(projector_inputs[1], expected)
	torch.testing.assert_close(output['z_a'], output['z_b'])


@pytest.mark.parametrize(
	('indices_a', 'indices_b', 'error_type', 'match'),
	[
		(
			torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
			torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64),
			TypeError,
			'dtype must be torch.int64',
		),
		(
			torch.zeros((2, 3, 1), dtype=torch.int64),
			torch.zeros((2, 3), dtype=torch.int64),
			ValueError,
			'shape \\[B, K\\]',
		),
		(
			torch.zeros((1, 3), dtype=torch.int64),
			torch.zeros((2, 3), dtype=torch.int64),
			ValueError,
			'matching \\[B, K\\] shapes',
		),
		(
			torch.zeros((1, 3), dtype=torch.int64),
			torch.zeros((1, 3), dtype=torch.int64),
			ValueError,
			'batch dimension must match encoded token batch',
		),
		(
			torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64),
			torch.tensor([[0, 1], [3, 4]], dtype=torch.int64),
			ValueError,
			'matching \\[B, K\\] shapes',
		),
	],
)
def test_local_forward_rejects_invalid_pair_indices(
	indices_a: torch.Tensor,
	indices_b: torch.Tensor,
	error_type: type[Exception],
	match: str,
) -> None:
	model = _make_model()
	view_a, view_b, mask_a, mask_b, _indices_a, _indices_b = _local_inputs()

	with pytest.raises(error_type, match=match):
		_call_forward_local(
			model,
			view_a=view_a,
			view_b=view_b,
			valid_mask_a=mask_a,
			valid_mask_b=mask_b,
			indices_a=indices_a,
			indices_b=indices_b,
		)


def test_local_forward_requires_valid_masks() -> None:
	model = _make_model()
	view_a, view_b, _mask_a, mask_b, indices_a, indices_b = _local_inputs()

	with pytest.raises(TypeError, match='valid_mask_a must be a tensor'):
		model.forward_local(
			view_a,
			view_b,
			valid_mask_a=None,  # type: ignore[arg-type]
			valid_mask_b=mask_b,
			local_pair_indices_a=indices_a,
			local_pair_indices_b=indices_b,
		)


def test_local_forward_does_not_extract_tensor_scalars(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model().eval()
	view_a, view_b, mask_a, mask_b, indices_a, indices_b = _local_inputs()
	encoded_outputs = iter(
		(
			{'tokens': torch.randn((2, 8, 8))},
			{'tokens': torch.randn((2, 8, 8))},
		)
	)

	def fake_encode_tokens(
		_view: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> dict[str, object]:
		assert valid_mask is not None
		return next(encoded_outputs)

	def unexpected_item(_tensor: torch.Tensor) -> object:
		raise AssertionError('forward_local() must not extract device scalars')

	monkeypatch.setattr(model.backbone, 'encode_tokens', fake_encode_tokens)
	monkeypatch.setattr(torch.Tensor, 'item', unexpected_item)

	output = _call_forward_local(
		model,
		view_a=view_a,
		view_b=view_b,
		valid_mask_a=mask_a,
		valid_mask_b=mask_b,
		indices_a=indices_a,
		indices_b=indices_b,
	)

	assert output['z_a'].shape == (6, 6)
	assert output['z_b'].shape == (6, 6)


def test_local_loss_has_encoder_projector_gradients_but_not_decoder_gradients() -> None:
	model = _make_model(projector_dim=4)
	view_a, view_b, mask_a, mask_b, indices_a, indices_b = _local_inputs()
	output = _call_forward_local(
		model,
		view_a=view_a,
		view_b=view_b,
		valid_mask_a=mask_a,
		valid_mask_b=mask_b,
		indices_a=indices_a,
		indices_b=indices_b,
	)

	loss = barlow_twins_loss(output['z_a'], output['z_b'])['loss']
	loss.backward()

	assert torch.isfinite(loss)
	assert model.backbone.patch_projection.weight.grad is not None
	encoder_gradients = [
		parameter.grad
		for parameter in model.backbone.encoder.parameters()
		if parameter.grad is not None
	]
	projector_gradients = [
		parameter.grad
		for parameter in model.projector.parameters()
		if parameter.grad is not None
	]
	assert encoder_gradients
	assert projector_gradients
	assert all(torch.isfinite(gradient).all() for gradient in encoder_gradients)
	assert all(torch.isfinite(gradient).all() for gradient in projector_gradients)
	assert model.backbone.mask_token.grad is None
	assert all(
		parameter.grad is None
		for module in (
			model.backbone.encoder_to_decoder,
			model.backbone.decoder,
			model.backbone.prediction_head,
		)
		for parameter in module.parameters()
	)


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


def test_loss_reports_dimension_normalized_representation_diagnostics() -> None:
	z_a = torch.tensor(
		[[1.0, 2.0, 4.0], [3.0, 6.0, 8.0], [5.0, 10.0, 12.0]]
	)
	z_b = torch.tensor(
		[[2.0, 1.0, 3.0], [4.0, 5.0, 7.0], [8.0, 9.0, 11.0]]
	)
	weight = 0.03

	result = barlow_twins_loss(z_a, z_b, redundancy_weight=weight)
	stds = torch.cat(
		(z_a.std(dim=0, unbiased=False), z_b.std(dim=0, unbiased=False))
	)
	norm_mean = torch.cat((z_a.norm(dim=1), z_b.norm(dim=1))).mean()
	normalized_a = (z_a - z_a.mean(dim=0)) / torch.sqrt(
		(z_a - z_a.mean(dim=0)).square().mean(dim=0) + 1.0e-12
	)
	normalized_b = (z_b - z_b.mean(dim=0)) / torch.sqrt(
		(z_b - z_b.mean(dim=0)).square().mean(dim=0) + 1.0e-12
	)
	cross_correlation = normalized_a.T @ normalized_b / z_a.shape[0]
	off_diagonal = cross_correlation[~torch.eye(3, dtype=torch.bool)]

	torch.testing.assert_close(result['projection_std_mean'], stds.mean())
	torch.testing.assert_close(result['projection_std_min'], stds.min())
	torch.testing.assert_close(result['projection_norm_mean'], norm_mean)
	torch.testing.assert_close(
		result['cross_correlation_diag_mean'],
		cross_correlation.diagonal().mean(),
	)
	torch.testing.assert_close(
		result['cross_correlation_offdiag_rms'],
		off_diagonal.square().mean().sqrt(),
	)
	torch.testing.assert_close(result['weighted_off_diag'], weight * result['off_diag'])


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
