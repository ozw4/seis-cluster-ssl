from __future__ import annotations

import pytest
import torch

from seis_ssl_cluster.models.mae.positional_encoding import (
	build_3d_sincos_position_embedding,
	restore_decoder_sequence,
	select_visible_tokens,
)


def _reference_select(
	tokens: torch.Tensor,
	pos: torch.Tensor,
	mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	flat_mask = mask.reshape(mask.shape[0], -1)
	return (
		torch.stack([tokens[index, row] for index, row in enumerate(flat_mask)]),
		torch.stack([pos[row] for row in flat_mask]),
	)


def _reference_restore(
	visible_tokens: torch.Tensor,
	pos: torch.Tensor,
	mask: torch.Tensor,
	mask_token: torch.Tensor,
) -> torch.Tensor:
	flat_mask = mask.reshape(mask.shape[0], -1)
	rows = []
	for batch_index, row_mask in enumerate(flat_mask):
		row = mask_token.expand(pos.shape[0], -1).clone()
		row[row_mask] = visible_tokens[batch_index]
		rows.append(row + pos)
	return torch.stack(rows)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_position_embedding_generates_in_float32_before_cast(
	dtype: torch.dtype,
) -> None:
	reference = build_3d_sincos_position_embedding((16, 2, 2), 24)
	actual = build_3d_sincos_position_embedding(
		(16, 2, 2),
		24,
		dtype=dtype,
	)

	assert torch.equal(actual, reference.to(dtype=dtype))


def test_equal_count_vectorization_matches_reference_and_gradients() -> None:
	mask = torch.tensor(
		[
			[[[True, False], [True, False]], [[False, True], [False, True]]],
			[[[False, True], [True, False]], [[True, False], [False, True]]],
		],
	)
	tokens = torch.randn(2, 5, 8).transpose(1, 2).requires_grad_()
	pos = torch.randn(5, 8).transpose(0, 1).requires_grad_()
	mask_token = torch.randn(5, requires_grad=True)

	visible_tokens, visible_pos, valid_mask = select_visible_tokens(
		tokens,
		pos,
		mask,
		equal_visible_count=True,
	)
	decoder_tokens, masked_token_mask = restore_decoder_sequence(
		visible_tokens,
		pos,
		mask,
		mask_token,
		equal_visible_count=True,
	)
	reference_tokens, reference_pos = _reference_select(tokens, pos, mask)
	reference_decoder = _reference_restore(
		reference_tokens,
		pos,
		mask,
		mask_token,
	)

	assert valid_mask is None
	torch.testing.assert_close(visible_tokens, reference_tokens)
	torch.testing.assert_close(visible_pos, reference_pos)
	torch.testing.assert_close(decoder_tokens, reference_decoder)
	torch.testing.assert_close(masked_token_mask, ~mask.reshape(2, -1))

	actual_loss = visible_pos.square().sum() + decoder_tokens.square().sum()
	actual_gradients = torch.autograd.grad(actual_loss, (tokens, pos, mask_token))
	reference_loss = reference_pos.square().sum() + reference_decoder.square().sum()
	reference_gradients = torch.autograd.grad(reference_loss, (tokens, pos, mask_token))
	for actual, reference in zip(actual_gradients, reference_gradients, strict=True):
		torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('visible_count', [0, 8])
def test_equal_count_vectorization_handles_zero_or_all_visible(
	batch_size: int,
	visible_count: int,
) -> None:
	tokens = torch.randn(batch_size, 8, 4)
	pos = torch.randn(8, 4)
	mask = torch.full((batch_size, 2, 2, 2), visible_count == 8)
	mask_token = torch.randn(4)

	visible_tokens, visible_pos, valid_mask = select_visible_tokens(
		tokens,
		pos,
		mask,
		equal_visible_count=True,
	)
	decoder_tokens, masked_token_mask = restore_decoder_sequence(
		visible_tokens,
		pos,
		mask,
		mask_token,
		equal_visible_count=True,
	)

	assert visible_tokens.shape == (batch_size, visible_count, 4)
	assert visible_pos.shape == (batch_size, visible_count, 4)
	assert valid_mask is None
	assert torch.equal(masked_token_mask, ~mask.reshape(batch_size, -1))
	if visible_count:
		torch.testing.assert_close(decoder_tokens, tokens + pos.unsqueeze(0))
	else:
		torch.testing.assert_close(
			decoder_tokens,
			(mask_token.view(1, 1, 4) + pos.unsqueeze(0)).expand(
				batch_size,
				-1,
				-1,
			),
		)


def test_variable_count_fallback_pads_and_restores_tokens() -> None:
	tokens = torch.arange(3 * 8 * 4, dtype=torch.float32).reshape(3, 8, 4)
	pos = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
	mask = torch.zeros((3, 2, 2, 2), dtype=torch.bool)
	mask[0].reshape(-1)[:4] = True
	mask[1].reshape(-1)[:2] = True
	mask_token = torch.full((4,), -1.0)

	visible_tokens, visible_pos, valid_mask = select_visible_tokens(tokens, pos, mask)
	decoder_tokens, _masked_token_mask = restore_decoder_sequence(
		visible_tokens,
		pos,
		mask,
		mask_token,
	)

	assert valid_mask is not None
	assert torch.equal(valid_mask.sum(dim=1), torch.tensor([4, 2, 0]))
	for batch_index, count in enumerate((4, 2, 0)):
		flat_mask = mask[batch_index].reshape(-1)
		torch.testing.assert_close(
			visible_tokens[batch_index, :count],
			tokens[batch_index, flat_mask],
		)
		torch.testing.assert_close(
			visible_pos[batch_index, :count],
			pos[flat_mask],
		)
		expected = mask_token.expand(8, -1).clone()
		expected[flat_mask] = tokens[batch_index, flat_mask]
		torch.testing.assert_close(decoder_tokens[batch_index], expected + pos)
