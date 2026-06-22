from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.models.mae.patching import compute_num_patches

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
		'visible_spatial_mask': ~spatial_mask,
	}


def test_forward_pass_returns_single_channel_patch_predictions() -> None:
	model = _make_model()

	out = model(_make_batch())

	assert out['pred_patches'].shape == (2, 64, 1, 64)
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
		assert key_padding_mask is not None
		captured['key_padding_mask'] = key_padding_mask
		return tokens

	monkeypatch.setattr(model.encoder, 'forward', capture_encoder)
	model(batch)

	assert captured['tokens'].shape == (1, 62, 32)
	assert captured['key_padding_mask'].shape == (1, 62)
	assert not captured['key_padding_mask'].any()


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


def test_constructor_and_batch_contract_do_not_use_excluded_names() -> None:
	parameter_names = set(inspect.signature(AmplitudeMAE3D).parameters)
	forbidden = {'attribute_ids', 'num_attributes', 'context', 'num_context_tokens'}

	assert parameter_names.isdisjoint(forbidden)
	assert set(_make_batch()) == {'x', 'spatial_mask', 'visible_spatial_mask'}
