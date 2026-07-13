"""Tests for the frozen-embedding voxel decoder."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch

from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D


def _small_model() -> VoxelDecoder3D:
	return VoxelDecoder3D(
		embedding_dim=8,
		class_count=3,
		hidden_channels=(8, 4),
		upsample_factors=((2, 1, 1), (1, 2, 2)),
		patch_size_xyz=(2, 2, 2),
	)


def test_default_decoder_upsamples_each_axis_by_eight() -> None:
	model = VoxelDecoder3D()
	embeddings = torch.randn(1, 384, 2, 1, 3)

	logits = model(embeddings)

	assert logits.shape == (1, 6, 16, 8, 24)
	assert model.group_norm_groups == (8, 8, 8)


@pytest.mark.parametrize(
	('batch_size', 'token_shape'),
	[(1, (1, 2, 3)), (2, (3, 1, 2))],
)
def test_decoder_supports_non_cubic_factors_and_variable_token_crops(
	batch_size: int,
	token_shape: tuple[int, int, int],
) -> None:
	model = VoxelDecoder3D(
		embedding_dim=5,
		class_count=2,
		hidden_channels=(7, 5),
		upsample_factors=((2, 1, 3), (1, 4, 1)),
		patch_size_xyz=(2, 4, 3),
	)
	embeddings = torch.randn(batch_size, 5, *token_shape)

	logits = model(embeddings)

	assert logits.shape == (
		batch_size,
		2,
		token_shape[0] * 2,
		token_shape[1] * 4,
		token_shape[2] * 3,
	)
	assert torch.isfinite(logits).all()


def test_decoder_supports_singleton_crop_with_one_value_per_group() -> None:
	model = VoxelDecoder3D(
		embedding_dim=4,
		class_count=2,
		hidden_channels=(8,),
		upsample_factors=((1, 1, 1),),
		patch_size_xyz=(1, 1, 1),
	)
	embeddings = torch.randn(1, 4, 1, 1, 1)

	logits = model(embeddings)

	assert model.group_norm_groups == (8,)
	assert logits.shape == (1, 2, 1, 1, 1)
	assert torch.isfinite(logits).all()


def test_token_mask_is_validated_and_masked_forward_is_finite() -> None:
	model = _small_model()
	embeddings = torch.randn(2, 8, 2, 1, 3)
	mask = torch.ones(2, 2, 1, 3, dtype=torch.bool)
	mask[:, 0, 0, 0] = False
	embeddings[0, :, 0, 0, 0] = torch.nan
	embeddings[1, :, 0, 0, 0] = torch.inf
	sanitized_embeddings = embeddings.masked_fill(~mask.unsqueeze(1), 0.0)

	logits = model(embeddings, mask)

	assert torch.isfinite(logits).all()
	torch.testing.assert_close(logits, model(sanitized_embeddings, mask))
	with pytest.raises(TypeError, match='dtype bool'):
		model(embeddings, mask.float())
	with pytest.raises(ValueError, match='must have shape'):
		model(embeddings, mask[:, :-1])
	with pytest.raises(ValueError, match='same device'):
		model(embeddings, torch.empty(mask.shape, dtype=torch.bool, device='meta'))


def test_backward_populates_decoder_gradients_without_mutating_input() -> None:
	model = _small_model()
	embeddings = torch.randn(2, 8, 2, 2, 1, requires_grad=True)
	before = embeddings.detach().clone()

	model(embeddings).square().mean().backward()

	assert torch.equal(embeddings.detach(), before)
	assert embeddings.grad is not None
	assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
	'kwargs',
	[
		{'hidden_channels': ()},
		{'hidden_channels': (8, 0, 2)},
		{'hidden_channels': (8, 4)},
		{'upsample_factors': ((2, 2), (2, 2, 2), (2, 2, 2))},
		{'upsample_factors': ((2, 2, 2), (2, 0, 2), (2, 2, 2))},
		{'patch_size_xyz': (8, 8, 4)},
	],
)
def test_malformed_architecture_settings_raise(kwargs: dict[str, object]) -> None:
	with pytest.raises((TypeError, ValueError)):
		VoxelDecoder3D(**kwargs)  # type: ignore[arg-type]


def test_group_counts_are_deterministic_divisors() -> None:
	model = VoxelDecoder3D(
		hidden_channels=(10, 7, 12),
		max_group_count=8,
	)

	assert model.group_norm_groups == (5, 7, 6)
	assert all(
		channels % groups == 0
		for channels, groups in zip(
			model.hidden_channels,
			model.group_norm_groups,
			strict=True,
		)
	)


def test_state_dict_round_trip_preserves_output() -> None:
	torch.manual_seed(13)
	model = _small_model().eval()
	embeddings = torch.randn(1, 8, 2, 2, 2)
	expected = model(embeddings)
	buffer = BytesIO()
	torch.save(model.state_dict(), buffer)
	buffer.seek(0)

	restored = _small_model().eval()
	restored.load_state_dict(torch.load(buffer, weights_only=True))

	torch.testing.assert_close(restored(embeddings), expected)


def test_float32_and_cpu_autocast_outputs_are_finite() -> None:
	model = _small_model()
	embeddings = torch.randn(1, 8, 2, 2, 2)

	assert torch.isfinite(model(embeddings)).all()
	with torch.autocast('cpu', dtype=torch.bfloat16):
		amp_logits = model(embeddings)
	assert torch.isfinite(amp_logits).all()
