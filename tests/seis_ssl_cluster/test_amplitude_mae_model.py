from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

import seis_ssl_cluster.models.mae.model as mae_model_module
from seis_ssl_cluster.losses import mae_pretraining_loss
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.models.mae.patching import compute_num_patches, patchify_3d

if TYPE_CHECKING:
	import pytest


def _make_model() -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		patch_size_xyz=(4, 4, 4),
		encoder_dim=32,
		encoder_depth=1,
		encoder_heads=4,
		decoder_dim=16,
		decoder_depth=1,
		decoder_heads=4,
	)


def _make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
	x = torch.randn((batch_size, 1, 16, 16, 16))
	spatial_mask = torch.zeros((batch_size, 4, 4, 4), dtype=torch.bool)
	spatial_mask[:, 0, 0, 0] = True
	spatial_mask[:, 1, 1, 1] = True
	return {
		'x': x,
		'spatial_mask': spatial_mask,
	}


def test_forward_pass_returns_single_channel_patch_predictions() -> None:
	model = _make_model()
	batch = _make_batch()

	out = model(batch)

	assert out['pred_patches'].shape == (2, 64, 1, 64)
	assert out['target_patches'].shape == (2, 64, 1, 64)
	assert not out['target_patches'].requires_grad
	torch.testing.assert_close(
		out['target_patches'],
		patchify_3d(batch['x'], model.patch_size_xyz),
	)
	assert out['encoded_visible_tokens'].shape == (2, 62, 32)
	assert out['spatial_mask'].shape == (2, 4, 4, 4)
	assert out['token_grid_shape'] == (4, 4, 4)


def test_default_patch_geometry_yields_4096_tokens_for_standard_crop() -> None:
	model = AmplitudeMAE3D()

	assert compute_num_patches((128, 128, 128), model.patch_size_xyz)[-1] == 4096


def test_encoder_receives_only_visible_tokens(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model()
	batch = _make_batch(batch_size=1)
	captured: dict[str, torch.Tensor] = {}

	def capture_encoder(
		tokens: torch.Tensor,
		key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		captured['tokens'] = tokens
		assert key_padding_mask is None
		return tokens

	monkeypatch.setattr(model.encoder, 'forward', capture_encoder)
	model(batch)

	assert captured['tokens'].shape == (1, 62, 32)


def test_forward_routes_unequal_visible_counts_through_padded_fallback(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model().eval()
	batch = _make_batch()
	batch['spatial_mask'][1, 2, 2, 2] = True
	captured_padding_masks: list[torch.Tensor | None] = []
	original_encoder_forward = model.encoder.forward

	def capture_encoder(
		tokens: torch.Tensor,
		key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		captured_padding_masks.append(key_padding_mask)
		return original_encoder_forward(tokens, key_padding_mask)

	monkeypatch.setattr(model.encoder, 'forward', capture_encoder)
	batched_output = model(batch)

	assert len(captured_padding_masks) == 1
	padding_mask = captured_padding_masks[0]
	assert padding_mask is not None
	assert torch.equal(padding_mask.sum(dim=1), torch.tensor([0, 1]))
	for batch_index in range(2):
		sample_output = model(
			{
				'x': batch['x'][batch_index : batch_index + 1],
				'spatial_mask': batch['spatial_mask'][
					batch_index : batch_index + 1
				],
			},
		)
		torch.testing.assert_close(
			batched_output['pred_patches'][batch_index],
			sample_output['pred_patches'][0],
		)


def test_position_embeddings_are_cached_across_forward_modes(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model()
	build_position_embedding = mae_model_module.build_3d_sincos_position_embedding
	call_count = 0

	def counted_build_position_embedding(
		grid_shape_xyz: tuple[int, int, int],
		embed_dim: int,
		*,
		device: torch.device | str | None = None,
		dtype: torch.dtype = torch.float32,
	) -> torch.Tensor:
		nonlocal call_count
		call_count += 1
		return build_position_embedding(
			grid_shape_xyz,
			embed_dim,
			device=device,
			dtype=dtype,
		)

	monkeypatch.setattr(
		mae_model_module,
		'build_3d_sincos_position_embedding',
		counted_build_position_embedding,
	)
	x = torch.randn((1, 1, 16, 16, 16))
	model.encode_tokens(x)
	model.encode_tokens(x)
	model(_make_batch(batch_size=1))

	assert call_count == 2
	assert len(model._position_embedding_cache) == 2  # noqa: SLF001
	assert not (set(model.state_dict()) & {'_position_embedding_cache'})


def test_position_embedding_cache_misses_and_is_bounded(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model()
	build_position_embedding = mae_model_module.build_3d_sincos_position_embedding
	call_count = 0

	def counted_build_position_embedding(
		grid_shape_xyz: tuple[int, int, int],
		embed_dim: int,
		*,
		device: torch.device | str | None = None,
		dtype: torch.dtype = torch.float32,
	) -> torch.Tensor:
		nonlocal call_count
		call_count += 1
		return build_position_embedding(
			grid_shape_xyz,
			embed_dim,
			device=device,
			dtype=dtype,
		)

	monkeypatch.setattr(
		mae_model_module,
		'build_3d_sincos_position_embedding',
		counted_build_position_embedding,
	)
	cpu_float = torch.empty((), dtype=torch.float32)
	cache_inputs = [
		((1, 1, 1), 4, cpu_float),
		((1, 1, 1), 4, cpu_float),
		((2, 1, 1), 4, cpu_float),
		((1, 1, 1), 6, cpu_float),
		((1, 1, 1), 4, torch.empty((), dtype=torch.float64)),
		((1, 1, 1), 4, torch.empty((), device='meta')),
		*[
			((size, 2, 1), 4, cpu_float)
			for size in range(3, 8)
		],
	]
	for grid_shape, embed_dim, reference in cache_inputs:
		model._position_embedding(  # noqa: SLF001
			grid_shape,
			embed_dim,
			reference,
		)

	assert call_count == 10
	assert len(model._position_embedding_cache) == 8  # noqa: SLF001
	cache_keys = set(model._position_embedding_cache)  # noqa: SLF001
	assert any(key[2].type == 'meta' for key in cache_keys)
	assert any(key[3] == torch.float64 for key in cache_keys)


def test_encode_tokens_returns_all_token_embeddings() -> None:
	model = _make_model()
	x = torch.randn((2, 1, 16, 16, 16))

	out = model.encode_tokens(x)

	assert out['tokens'].shape == (2, 64, 32)
	assert out['token_grid_shape'] == (4, 4, 4)
	assert out['token_valid_mask'] is None


def test_encode_tokens_returns_flat_token_valid_mask_from_voxel_mask() -> None:
	model = _make_model()
	x = torch.randn((1, 1, 16, 16, 16))
	valid_mask = torch.ones((1, 16, 16, 16), dtype=torch.bool)
	valid_mask[:, :4, :4, :4] = False

	out = model.encode_tokens(x, valid_mask=valid_mask)

	assert out['tokens'].shape == (1, 64, 32)
	assert isinstance(out['token_valid_mask'], torch.Tensor)
	assert out['token_valid_mask'].shape == (1, 64)
	assert not out['token_valid_mask'][0, 0]
	assert out['token_valid_mask'][0, 1:].all()


def test_encode_tokens_accepts_token_grid_valid_mask() -> None:
	model = _make_model()
	x = torch.randn((1, 1, 16, 16, 16))
	valid_mask = torch.ones((1, 4, 4, 4), dtype=torch.bool)
	valid_mask[:, 0, 0, 0] = False

	out = model.encode_tokens(x, valid_mask=valid_mask)

	assert isinstance(out['token_valid_mask'], torch.Tensor)
	assert out['token_valid_mask'].shape == (1, 64)
	assert not out['token_valid_mask'][0, 0]


def test_gradients_flow_from_pred_patches_sum() -> None:
	model = _make_model()
	out = model(_make_batch())

	out['pred_patches'].sum().backward()

	assert model.patch_projection.weight.grad is not None


def test_training_path_patchifies_input_once(monkeypatch: pytest.MonkeyPatch) -> None:
	model = _make_model()
	batch = _make_batch(batch_size=1)
	patchify = mae_model_module.patchify_3d
	call_count = 0

	def counted_patchify(
		x: torch.Tensor,
		patch_size_xyz: tuple[int, int, int],
	) -> torch.Tensor:
		nonlocal call_count
		call_count += 1
		return patchify(x, patch_size_xyz)

	monkeypatch.setattr(mae_model_module, 'patchify_3d', counted_patchify)
	output = model(batch)
	mae_pretraining_loss(
		pred_patches=output['pred_patches'],
		target_patches=output['target_patches'],
		x=batch['x'],
		spatial_mask=batch['spatial_mask'],
		local_valid_mask=torch.ones((1, 16, 16, 16), dtype=torch.bool),
		patch_size_xyz=model.patch_size_xyz,
		gradient_weight=0.0,
	)

	assert call_count == 1


def test_state_dict_keys_and_strict_checkpoint_load_are_unchanged() -> None:
	expected_keys = {
		'mask_token',
		'patch_projection.weight',
		'patch_projection.bias',
		'encoder.layers.0.norm1.weight',
		'encoder.layers.0.norm1.bias',
		'encoder.layers.0.attention.in_proj_weight',
		'encoder.layers.0.attention.in_proj_bias',
		'encoder.layers.0.attention.out_proj.weight',
		'encoder.layers.0.attention.out_proj.bias',
		'encoder.layers.0.norm2.weight',
		'encoder.layers.0.norm2.bias',
		'encoder.layers.0.mlp.0.weight',
		'encoder.layers.0.mlp.0.bias',
		'encoder.layers.0.mlp.3.weight',
		'encoder.layers.0.mlp.3.bias',
		'encoder_to_decoder.weight',
		'encoder_to_decoder.bias',
		'decoder.layers.0.norm1.weight',
		'decoder.layers.0.norm1.bias',
		'decoder.layers.0.attention.in_proj_weight',
		'decoder.layers.0.attention.in_proj_bias',
		'decoder.layers.0.attention.out_proj.weight',
		'decoder.layers.0.attention.out_proj.bias',
		'decoder.layers.0.norm2.weight',
		'decoder.layers.0.norm2.bias',
		'decoder.layers.0.mlp.0.weight',
		'decoder.layers.0.mlp.0.bias',
		'decoder.layers.0.mlp.3.weight',
		'decoder.layers.0.mlp.3.bias',
		'prediction_head.weight',
		'prediction_head.bias',
	}
	checkpoint_state = _make_model().state_dict()

	assert set(checkpoint_state) == expected_keys
	_make_model().load_state_dict(checkpoint_state, strict=True)


def test_constructor_and_batch_contract_do_not_use_excluded_names() -> None:
	parameter_names = set(inspect.signature(AmplitudeMAE3D).parameters)
	forbidden = {'attribute_ids', 'num_attributes', 'context', 'num_context_tokens'}

	assert parameter_names.isdisjoint(forbidden)
	assert set(_make_batch()) == {'x', 'spatial_mask'}
