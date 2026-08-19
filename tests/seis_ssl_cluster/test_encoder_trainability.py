from __future__ import annotations

import pytest
from torch import nn

from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.encoder_trainability import (
	freeze_all_and_unfreeze_top_encoder_blocks,
)


def test_freezes_all_parameters_when_no_blocks_are_unfrozen() -> None:
	model = _model()
	assert all(parameter.requires_grad for parameter in model.parameters())

	trainable_names = freeze_all_and_unfreeze_top_encoder_blocks(
		model,
		unfreeze_top_blocks=0,
	)

	assert trainable_names == ()
	assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
	('unfreeze_top_blocks', 'trainable_block_indices'),
	[(1, (2,)), (3, (0, 1, 2))],
)
def test_unfreezes_only_requested_top_encoder_blocks(
	unfreeze_top_blocks: int,
	trainable_block_indices: tuple[int, ...],
) -> None:
	model = _model()

	trainable_names = freeze_all_and_unfreeze_top_encoder_blocks(
		model,
		unfreeze_top_blocks=unfreeze_top_blocks,
	)

	assert trainable_names
	assert all(
		name.startswith(
			tuple(f'encoder.layers.{index}.' for index in trainable_block_indices)
		)
		for name in trainable_names
	)
	for index, block in enumerate(model.encoder.layers):
		assert all(
			parameter.requires_grad == (index in trainable_block_indices)
			for parameter in block.parameters()
		)


@pytest.mark.parametrize('unfreeze_top_blocks', [0, 1, 3])
def test_non_encoder_components_remain_frozen(unfreeze_top_blocks: int) -> None:
	model = _model()

	freeze_all_and_unfreeze_top_encoder_blocks(
		model,
		unfreeze_top_blocks=unfreeze_top_blocks,
	)

	assert all(
		not parameter.requires_grad
		for parameter in model.patch_projection.parameters()
	)
	assert all(
		not parameter.requires_grad
		for parameter in model.encoder_to_decoder.parameters()
	)
	assert not model.mask_token.requires_grad
	assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
	assert all(
		not parameter.requires_grad
		for parameter in model.prediction_head.parameters()
	)


def test_returns_exact_trainable_parameter_names() -> None:
	model = _model()

	trainable_names = freeze_all_and_unfreeze_top_encoder_blocks(
		model,
		unfreeze_top_blocks=1,
	)

	assert set(trainable_names) == {
		name for name, parameter in model.named_parameters() if parameter.requires_grad
	}


@pytest.mark.parametrize(
	('unfreeze_top_blocks', 'error_type'),
	[(True, TypeError), (-1, ValueError), (4, ValueError)],
)
def test_rejects_invalid_unfreeze_top_blocks(
	unfreeze_top_blocks: int,
	error_type: type[Exception],
) -> None:
	with pytest.raises(error_type, match='unfreeze_top_blocks'):
		freeze_all_and_unfreeze_top_encoder_blocks(
			_model(),
			unfreeze_top_blocks=unfreeze_top_blocks,
		)


def test_rejects_non_amplitude_mae_model() -> None:
	with pytest.raises(TypeError, match='model must be an AmplitudeMAE3D'):
		freeze_all_and_unfreeze_top_encoder_blocks(
			nn.Linear(2, 2),  # type: ignore[arg-type]
			unfreeze_top_blocks=0,
		)


def _model() -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=3,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)
