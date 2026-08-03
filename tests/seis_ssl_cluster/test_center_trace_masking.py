"""Focused contracts for center-trace XY masking and encoder replacement."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from seis_ssl_cluster.models.mae import (
	AmplitudeMAE3D,
	LearnedEncoderReplacementToken,
)
from seis_ssl_cluster.models.mae.patching import patchify_3d
from seis_ssl_cluster.training.strat_hmm.masking import (
	plan_xy_token_column_mask,
	validate_common_hard_target_valid_masks,
)


def _make_model() -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		patch_size_xyz=(2, 2, 2),
		encoder_dim=8,
		encoder_depth=1,
		encoder_heads=2,
		decoder_dim=4,
		decoder_depth=1,
		decoder_heads=2,
	)


def _plan(
	valid: torch.Tensor,
	student: torch.Tensor | None = None,
	**kwargs: object,
):
	arguments: dict[str, object] = {
		'training_seed': 19,
		'epoch': 2,
		'global_step': 7,
		'batch_index': 3,
	}
	arguments.update(kwargs)
	return plan_xy_token_column_mask(
		valid,
		valid if student is None else student,
		**arguments,
	)


@pytest.mark.parametrize(
	('eligible_count', 'expected_selected'),
	[(2, 1), (9, 1), (10, 1), (15, 2), (16, 2), (100, 10)],
)
def test_xy_column_count_uses_fixed_rounding_contract(
	eligible_count: int,
	expected_selected: int,
) -> None:
	valid = torch.ones((1, 1, eligible_count, 1), dtype=torch.bool)

	plan = _plan(valid)

	assert plan.eligible_counts.tolist() == [eligible_count]
	assert plan.selected_counts.tolist() == [expected_selected]
	assert int(plan.mask.sum()) == expected_selected


def test_xy_mask_is_full_z_and_uses_target_student_intersection() -> None:
	common = torch.zeros((1, 2, 4, 3), dtype=torch.bool)
	common[:, 0, :3, 0] = True
	common[:, 1, :3, 1] = True
	student = torch.ones_like(common)
	student[:, 0, 1, :] = False
	student[:, 1, 2, :] = False

	plan = _plan(common, student)

	assert plan.eligible_counts.tolist() == [4]
	assert plan.selected_counts.tolist() == [1]
	selected = {
		tuple(coordinates)
		for coordinates in plan.selected_xy_coordinates[0].tolist()
		if coordinates != [-1, -1]
	}
	assert len(selected) == 1
	for tx, ty in selected:
		assert plan.mask[0, tx, ty].all()
	for tx in range(common.shape[1]):
		for ty in range(common.shape[2]):
			if (tx, ty) not in selected:
				assert not plan.mask[0, tx, ty].any()


def test_xy_mask_accepts_flat_student_mask_and_rejects_k_mismatch() -> None:
	common = torch.ones((1, 2, 2, 2), dtype=torch.bool)
	flat_student = common.reshape(1, -1)
	plan = _plan(common, flat_student)
	assert plan.mask.shape == common.shape

	other = common.clone()
	other[0, 0, 0, 0] = False
	with pytest.raises(ValueError, match='K=6/8/10'):
		validate_common_hard_target_valid_masks([common, other, common])


def test_xy_mask_is_stateless_and_identity_sensitive() -> None:
	valid = torch.ones((2, 1, 20, 1), dtype=torch.bool)
	python_state = random.getstate()
	numpy_state = np.random.get_state()  # noqa: NPY002
	torch_state = torch.get_rng_state()

	first = _plan(valid)
	repeat = _plan(valid)
	changed_epoch = _plan(valid, epoch=3)
	changed_step = _plan(valid, global_step=8)
	changed_sample = _plan(valid, sample_indices=[10, 11])

	assert torch.equal(first.mask, repeat.mask)
	assert not torch.equal(first.mask, changed_epoch.mask)
	assert not torch.equal(first.mask, changed_step.mask)
	assert not torch.equal(first.mask, changed_sample.mask)
	assert random.getstate() == python_state
	after_numpy = np.random.get_state()  # noqa: NPY002
	assert after_numpy[0] == numpy_state[0]
	assert np.array_equal(after_numpy[1], numpy_state[1])
	assert after_numpy[2:] == numpy_state[2:]
	assert torch.equal(torch.get_rng_state(), torch_state)

	for sample_coordinates in first.selected_xy_coordinates.tolist():
		selected = [tuple(item) for item in sample_coordinates if item != [-1, -1]]
		assert len(selected) == len(set(selected))


def test_xy_mask_rejects_invalid_contracts() -> None:
	valid = torch.ones((1, 1, 2, 1), dtype=torch.bool)
	with pytest.raises(ValueError, match='at least two'):
		_plan(valid[:, :, :1])
	with pytest.raises(TypeError, match='dtype'):
		_plan(valid.to(dtype=torch.float32))
	with pytest.raises(ValueError, match='column_fraction'):
		_plan(valid, column_fraction=0.2)
	with pytest.raises(ValueError, match='shape'):
		_plan(valid, torch.ones((1, 3), dtype=torch.bool))
	with pytest.raises(ValueError, match='same device'):
		_plan(valid, torch.ones_like(valid, device='meta'))


def test_replacement_token_initialization_does_not_consume_global_rng() -> None:
	torch_state = torch.get_rng_state()
	first = LearnedEncoderReplacementToken(8, seed=31)
	second = LearnedEncoderReplacementToken(8, seed=31)

	assert torch.equal(torch.get_rng_state(), torch_state)
	torch.testing.assert_close(first(), second())
	assert first().shape == (8,)
	assert first().dtype.is_floating_point
	assert torch.isfinite(first()).all()


def test_encode_tokens_replaces_before_position_embedding(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model()
	x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
	replacement_mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	replacement_mask[0, 1, 0, 1] = True
	replacement_token = torch.full((8,), 3.0)
	captured: dict[str, torch.Tensor] = {}

	def capture_encoder(
		tokens: torch.Tensor,
		key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		assert key_padding_mask is None
		captured['tokens'] = tokens
		return tokens

	monkeypatch.setattr(model.encoder, 'forward', capture_encoder)
	output = model.encode_tokens(
		x,
		replacement_mask=replacement_mask,
		replacement_token=replacement_token,
	)
	_patches, projected, grid = model._project_patches(x)  # noqa: SLF001
	position = model._position_embedding(  # noqa: SLF001
		grid,
		model.encoder_dim,
		projected,
	)
	expected = projected.clone()
	expected[replacement_mask.reshape(1, -1)] = replacement_token
	expected = expected + position.unsqueeze(0)

	torch.testing.assert_close(captured['tokens'], expected)
	torch.testing.assert_close(output['replacement_mask'], replacement_mask)
	assert torch.equal(output['replacement_mask'], replacement_mask)


def test_replacement_hides_raw_projected_value_and_preserves_input_state() -> None:
	model = _make_model()
	x = torch.randn((1, 1, 4, 4, 4))
	x_before = x.clone()
	valid_mask = torch.ones((1, 2, 2, 2), dtype=torch.bool)
	replacement_mask = torch.zeros_like(valid_mask)
	replacement_mask[0, 0, 1, 1] = True
	replacement_before = replacement_mask.clone()
	token = torch.full((8,), 0.25)

	first = model.encode_tokens(
		x,
		valid_mask=valid_mask,
		replacement_mask=replacement_mask,
		replacement_token=token,
	)
	x_changed = x.clone()
	x_changed[:, :, :2, 2:4, 2:4] += 1000.0
	second = model.encode_tokens(
		x_changed,
		valid_mask=valid_mask,
		replacement_mask=replacement_mask,
		replacement_token=token,
	)

	torch.testing.assert_close(first['tokens'], second['tokens'])
	torch.testing.assert_close(x, x_before)
	torch.testing.assert_close(replacement_mask, replacement_before)


def test_replacement_token_and_encoder_receive_finite_gradients() -> None:
	model = _make_model()
	replacement = LearnedEncoderReplacementToken(8, seed=5)
	x = torch.randn((1, 1, 4, 4, 4))
	mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	mask[0, 0, 0, 0] = True

	output = model.encode_tokens(
		x,
		replacement_mask=mask,
		replacement_token=replacement(),
	)
	output['tokens'].sum().backward()

	assert replacement.replacement_token.grad is not None
	assert model.encoder.layers[0].norm1.weight.grad is not None
	assert torch.isfinite(replacement.replacement_token.grad).all()
	assert torch.isfinite(model.encoder.layers[0].norm1.weight.grad).all()


def test_encode_tokens_replacement_validation_is_fail_closed() -> None:
	model = _make_model()
	x = torch.randn((1, 1, 4, 4, 4))
	mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	with pytest.raises(ValueError, match='provided together'):
		model.encode_tokens(x, replacement_mask=mask)
	with pytest.raises(ValueError, match='shape'):
		model.encode_tokens(
			x,
			replacement_mask=mask.reshape(1, -1),
			replacement_token=torch.zeros(8),
		)
	with pytest.raises(TypeError, match='dtype'):
		model.encode_tokens(
			x,
			replacement_mask=mask,
			replacement_token=torch.zeros(8, dtype=torch.int64),
		)
	bad_token = torch.zeros(8)
	bad_token[0] = float('nan')
	with pytest.raises(ValueError, match='finite'):
		model.encode_tokens(
			x,
			replacement_mask=mask,
			replacement_token=bad_token,
		)


def test_unmasked_encode_output_keys_remain_unchanged() -> None:
	model = _make_model()
	x = torch.randn((1, 1, 4, 4, 4))
	out = model.encode_tokens(x)

	assert set(out) == {'tokens', 'token_grid_shape', 'token_valid_mask'}
	assert torch.equal(out['tokens'], model.encode_tokens(x)['tokens'])
	assert torch.equal(
		patchify_3d(x, model.patch_size_xyz),
		model._project_patches(x)[0],  # noqa: SLF001
	)
