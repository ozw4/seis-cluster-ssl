'''Tests for the shared Volve horizon model, tiles, loss, and metrics.'''

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from seis_ssl_cluster.data.normalization import AmplitudeAgcConfig
from seis_ssl_cluster.data.schema import CropRequest
from seis_ssl_cluster.data.window_preprocessing import (
	AmplitudePreprocessSettings,
	PreparedAmplitudeCrop,
	read_prepared_survey_amplitude_crop,
)
from seis_ssl_cluster.data.zero_mask import ZeroMaskConfig
from seis_ssl_cluster.volve.horizon_loss import (
	fractional_horizon_cross_entropy,
	fractional_target_weights,
	training_horizon_observation_counts,
	validate_training_horizon_coverage,
)
from seis_ssl_cluster.volve.horizon_metrics import (
	compute_horizon_metrics,
	predicted_adjacent_order_violation,
	soft_argmax_global_sample,
)
from seis_ssl_cluster.volve.horizon_model import (
	HORIZON_DECODER_SEED,
	VolveHorizonDecoder,
	create_volve_horizon_decoder,
)
from seis_ssl_cluster.volve.horizon_tiles import (
	HORIZON_WINDOW_LENGTH,
	HorizonTileRecord,
	HorizonTileSettings,
	build_frozen_horizon_tile,
	build_horizon_tile_targets,
	build_raw_horizon_tile,
	enumerate_horizon_tile_records,
	horizon_supervision_mask,
)


def _small_decoder() -> VolveHorizonDecoder:
	return VolveHorizonDecoder(
		embedding_dim=8,
		hidden_channels=(4, 4, 4),
	)


def _tile_arrays(
	shape: tuple[int, int] = (65, 65),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	samples = np.full((5, *shape), 600.25, dtype=np.float32)
	native = np.ones(samples.shape, dtype=np.bool_)
	split = np.zeros(samples.shape, dtype=np.bool_)
	traces = np.ones(shape, dtype=np.bool_)
	return samples, native, split, traces


def test_decoder_output_has_five_horizons_and_full_216_sample_window() -> None:
	model = _small_decoder()
	embeddings = torch.randn(1, 8, 10, 10, 27)
	valid = torch.ones(1, 10, 10, 27, dtype=torch.bool)

	logits = model(embeddings, valid)

	assert logits.shape == (1, 5, 64, 64, HORIZON_WINDOW_LENGTH)
	assert torch.isfinite(logits).all()


class _CoordinateContextDecoder(nn.Module):
	embedding_dim = 8

	def forward(
		self, embeddings: torch.Tensor, _token_valid_mask: torch.Tensor | None
	) -> torch.Tensor:
		x_size = embeddings.shape[2] * 8
		y_size = embeddings.shape[3] * 8
		z_size = embeddings.shape[4] * 8
		x = torch.arange(x_size, dtype=embeddings.dtype).reshape(1, 1, -1, 1, 1)
		return x.expand(embeddings.shape[0], 5, x_size, y_size, z_size)


def test_decoder_crops_only_the_central_lateral_core() -> None:
	model = _small_decoder()
	model.voxel_decoder = _CoordinateContextDecoder()
	embeddings = torch.zeros(1, 8, 10, 10, 27)

	logits = model(embeddings)

	assert logits.shape == (1, 5, 64, 64, 216)
	assert torch.equal(logits[0, 0, :, 0, 0], torch.arange(8, 72).float())


def test_frozen_embeddings_and_encoder_mapping_share_decoder_wrapper() -> None:
	model = _small_decoder().eval()
	embeddings = torch.randn(1, 8, 10, 10, 27)
	valid = torch.ones(1, 10, 10, 27, dtype=torch.bool)
	tokens = embeddings.movedim(1, -1).reshape(1, -1, 8)
	encoded = {
		'tokens': tokens,
		'token_grid_shape': (10, 10, 27),
		'token_valid_mask': valid.reshape(1, -1),
	}

	with torch.no_grad():
		frozen = model(embeddings, valid)
		end_to_end = model.forward_encoder_output(encoded)

	torch.testing.assert_close(frozen, end_to_end)


def test_decoder_initialization_is_deterministic_at_seed_42000() -> None:
	kwargs = {
		'embedding_dim': 8,
		'hidden_channels': (4, 4, 4),
	}
	first = create_volve_horizon_decoder(seed=HORIZON_DECODER_SEED, **kwargs)
	second = create_volve_horizon_decoder(seed=HORIZON_DECODER_SEED, **kwargs)
	different = create_volve_horizon_decoder(seed=HORIZON_DECODER_SEED + 1, **kwargs)

	assert all(
		torch.equal(first.state_dict()[name], second.state_dict()[name])
		for name in first.state_dict()
	)
	assert any(
		not torch.equal(first.state_dict()[name], different.state_dict()[name])
		for name in first.state_dict()
	)


def test_default_decoder_architecture_is_the_fixed_scientific_contract() -> None:
	model = create_volve_horizon_decoder()

	assert model.architecture['embedding_dim'] == 384
	assert model.architecture['class_count'] == 5
	assert model.architecture['hidden_channels'] == [128, 64, 32]
	assert model.architecture['upsample_factors'] == [[2, 2, 2]] * 3
	assert model.architecture['upsample_mode'] == 'nearest'
	assert model.architecture['normalization'] == 'voxelwise_layer_norm'
	assert model.architecture['patch_size_xyz'] == [8, 8, 8]
	assert model.architecture['core_size_tokens'] == [8, 8, 27]
	assert model.architecture['context_halo_tokens'] == [1, 1, 0]


def test_fractional_target_weights_split_floor_and_ceil() -> None:
	values = torch.tensor([615.25, 616.0, 551.9, 768.0])

	valid, floor, ceil, floor_weight, ceil_weight = fractional_target_weights(values)

	assert valid.tolist() == [True, True, False, False]
	assert floor[:2].tolist() == [63, 64]
	assert ceil[:2].tolist() == [64, 65]
	torch.testing.assert_close(floor_weight[:2], torch.tensor([0.75, 1.0]))
	torch.testing.assert_close(ceil_weight[:2], torch.tensor([0.25, 0.0]))


def test_fractional_loss_macro_averages_only_active_horizons() -> None:
	logits = torch.zeros(1, 5, 1, 1, 216, requires_grad=True)
	targets = torch.tensor([[[[600.25]], [[610.0]], [[620.5]], [[630.0]], [[640.0]]]])
	mask = torch.ones(1, 5, 1, 1, dtype=torch.bool)
	mask[:, 4] = False

	loss, summary = fractional_horizon_cross_entropy(logits, targets, mask)

	assert loss.item() == pytest.approx(math.log(216.0))
	assert summary['active_horizon_count'] == 4
	assert summary['supervised_observation_count'] == 4
	assert summary['per_horizon']['hugin_base']['cross_entropy'] is None
	loss.backward()
	assert torch.isfinite(logits.grad).all()


def test_supervision_combines_native_split_trace_finite_and_window_masks() -> None:
	samples, native, split, traces = _tile_arrays((3, 3))
	split[:] = True
	native[0, 0, 0] = False
	split[1, 0, 1] = False
	traces[0, 2] = False
	samples[3, 1, 0] = np.nan
	samples[4, 1, 1] = 800.0

	mask = horizon_supervision_mask(
		sample_float=samples,
		native_valid_mask=native,
		split_mask=split,
		trace_valid_mask=traces,
	)

	assert not mask[0, 0, 0]
	assert not mask[1, 0, 1]
	assert not mask[:, 0, 2].any()
	assert not mask[3, 1, 0]
	assert not mask[4, 1, 1]
	assert mask[2, 2, 2]


def test_tile_enumeration_and_targets_keep_all_available_points() -> None:
	settings = HorizonTileSettings(
		lateral_shape_xy=(65, 65), min_token_valid_fraction=0.1
	)
	samples, native, split, traces = _tile_arrays()
	split[:, 0, :] = True
	split[:, :, 64] = True
	native[2, 0, 1] = False
	records = enumerate_horizon_tile_records(
		sample_float=samples,
		native_valid_mask=native,
		split_mask=split,
		trace_valid_mask=traces,
		settings=settings,
	)

	assert [record.tile_id for record in records] == [0, 1, 3]
	assert sum(record.supervised_observation_count for record in records) == int(
		np.count_nonzero(
			horizon_supervision_mask(
				sample_float=samples,
				native_valid_mask=native,
				split_mask=split,
				trace_valid_mask=traces,
			)
		)
	)
	edge = build_horizon_tile_targets(
		record=records[-1],
		sample_float=samples,
		native_valid_mask=native,
		split_mask=split,
		trace_valid_mask=traces,
		settings=settings,
	)
	assert edge.sample_float.shape == (5, 64, 64)
	assert edge.input_start_token == (7, 7, 69)
	assert edge.supervision_mask[:, 0, 0].all()
	assert not edge.trace_valid_mask[1:, :].any()


def test_frozen_and_raw_edge_tiles_pad_zero_and_preserve_preprocessing() -> None:
	settings = HorizonTileSettings(
		lateral_shape_xy=(65, 65), min_token_valid_fraction=1.0
	)
	samples, native, split, traces = _tile_arrays()
	split[:, 0, 0] = True
	record = enumerate_horizon_tile_records(
		sample_float=samples,
		native_valid_mask=native,
		split_mask=split,
		trace_valid_mask=traces,
		settings=settings,
	)[0]
	token_shape = settings.token_grid_shape
	embeddings = np.ones((*token_shape, 3), dtype=np.float32)
	valid_tokens = np.ones(token_shape, dtype=np.bool_)
	frozen = build_frozen_horizon_tile(
		record=record,
		embeddings=embeddings,
		valid_tokens=valid_tokens,
		settings=settings,
	)
	assert frozen.embeddings.shape == (3, 10, 10, 27)
	assert not frozen.token_valid_mask[0].any()
	assert not frozen.token_valid_mask[:, 0].any()
	assert np.count_nonzero(frozen.embeddings[:, 0]) == 0

	normalized_amplitude = np.ones((65, 65, 768), dtype=np.float32)
	zero_like_mask = np.zeros(normalized_amplitude.shape, dtype=np.bool_)
	zero_like_mask[0, 0, :] = True
	zero_like_mask[:, :, 600] = True
	request = CropRequest(
		survey_id='volve',
		start_xyz=(-8, -8, 552),
		size_xyz=settings.input_size_voxels,
	)
	prepared = read_prepared_survey_amplitude_crop(
		request=request,
		normalized_amplitude=normalized_amplitude,
		zero_like_mask=zero_like_mask,
		patch_size_xyz=settings.patch_size_xyz,
		settings=AmplitudePreprocessSettings(
			zero_mask=ZeroMaskConfig(
				enabled=True,
				zero_atol=0.0,
				z_sample_influence_radius=16,
				xy_trace_influence_radius=1,
			),
			normalized_clip_abs=8.0,
			amplitude_agc=AmplitudeAgcConfig(
				enabled=True,
				mode='trace_rms_z',
				window_z=65,
				eps=1.0e-3,
				clip_abs=5.0,
			),
			min_token_valid_fraction=1.0,
			finite_check_mode='strict',
		),
	)
	raw = build_raw_horizon_tile(
		record=record,
		prepared_crop=prepared,
		settings=settings,
	)
	assert raw.amplitude.shape == (1, 80, 80, 216)
	assert not raw.local_valid_mask[:8].any()
	assert not raw.local_valid_mask[:, :8].any()
	assert not raw.local_valid_mask[8, 8].any()
	assert not raw.local_valid_mask[:, :, 32:65].any()
	assert np.count_nonzero(raw.amplitude[0, 8, 8]) == 0
	np.testing.assert_array_equal(raw.local_valid_mask, prepared.local_valid_mask)
	np.testing.assert_array_equal(raw.token_valid_mask, prepared.token_valid_mask)
	assert np.isfinite(raw.amplitude).all()


def test_raw_tile_rejects_token_validity_from_a_different_threshold() -> None:
	settings = HorizonTileSettings(
		lateral_shape_xy=(64, 64), min_token_valid_fraction=1.0
	)
	record = HorizonTileRecord(
		tile_id=0,
		core_start_token_xy=(0, 0),
		core_stop_token_xy=(8, 8),
		per_horizon_observation_counts=(1, 1, 1, 1, 1),
	)
	local_valid = np.ones(settings.input_size_voxels, dtype=np.bool_)
	local_valid[0, 0, 0] = False
	prepared = PreparedAmplitudeCrop(
		request=CropRequest(
			survey_id='volve',
			start_xyz=(-8, -8, 552),
			size_xyz=settings.input_size_voxels,
		),
		x=np.where(local_valid[np.newaxis], 1.0, 0.0).astype(np.float32),
		local_valid_mask=local_valid,
		token_valid_mask=np.ones(settings.input_size_tokens, dtype=np.bool_),
	)

	with pytest.raises(ValueError, match='min_token_valid_fraction'):
		build_raw_horizon_tile(
			record=record,
			prepared_crop=prepared,
			settings=settings,
		)


def test_training_preflight_rejects_any_zero_coverage_horizon() -> None:
	mask = torch.ones(2, 5, 3, 4, dtype=torch.bool)
	mask[:, 3] = False

	assert training_horizon_observation_counts(mask) == (24, 24, 24, 0, 24)
	with pytest.raises(ValueError, match='hugin_top'):
		validate_training_horizon_coverage(mask)
	assert validate_training_horizon_coverage((1, 2, 3, 4, 5)) == (1, 2, 3, 4, 5)


def test_soft_argmax_returns_exact_fractional_global_sample() -> None:
	logits = torch.full((1, 5, 1, 1, 216), -torch.inf)
	logits[..., 63] = math.log(3.0)
	logits[..., 64] = 0.0

	predicted = soft_argmax_global_sample(logits)

	torch.testing.assert_close(predicted, torch.full((1, 5, 1, 1), 615.25))


def test_metrics_are_horizon_macro_and_convert_samples_to_ms() -> None:
	target = np.array([600, 610, 620, 630, 640], dtype=np.float32).reshape(1, 5, 1, 1)
	predicted = target + np.arange(5, dtype=np.float32).reshape(1, 5, 1, 1)
	mask = np.ones(target.shape, dtype=np.bool_)

	metrics = compute_horizon_metrics(predicted, target, mask)

	assert metrics['macro_mae_samples'] == pytest.approx(2.0)
	assert metrics['macro_within_2_samples'] == pytest.approx(0.6)
	assert metrics['macro']['within_1'] == pytest.approx(0.4)
	assert metrics['macro']['within_4'] == pytest.approx(1.0)
	assert metrics['per_horizon']['hugin_base']['mae_samples'] == pytest.approx(4.0)
	assert metrics['per_horizon']['hugin_base']['mae_ms'] == pytest.approx(16.0)
	assert metrics['coverage']['fraction'] == pytest.approx(1.0)


def test_metrics_report_missing_predictions_and_order_violations() -> None:
	predicted = np.array(
		[600, 630, 620, 640, 640], dtype=np.float32
	).reshape(1, 5, 1, 1)
	target = np.array([600, 610, 620, 630, 640], dtype=np.float32).reshape(1, 5, 1, 1)
	mask = np.ones(target.shape, dtype=np.bool_)
	order = predicted_adjacent_order_violation(predicted, mask)
	assert order == {'pair_count': 4, 'violation_count': 2, 'rate': 0.5}

	predicted[:, 4] = np.nan
	metrics = compute_horizon_metrics(predicted, target, mask)
	assert metrics['missing_prediction_count'] == 1
	assert metrics['coverage']['fraction'] == pytest.approx(0.8)
	assert metrics['per_horizon']['hugin_base']['count'] == 1
	assert metrics['per_horizon']['hugin_base']['predicted_count'] == 0


def test_cpu_forward_fractional_loss_backward_and_optimizer_step_are_finite() -> None:
	torch.manual_seed(7)
	model = _small_decoder()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	embeddings = torch.randn(1, 8, 10, 10, 27)
	valid = torch.ones(1, 10, 10, 27, dtype=torch.bool)
	target = torch.full((1, 5, 64, 64), 600.25)
	mask = torch.zeros(1, 5, 64, 64, dtype=torch.bool)
	mask[:, :, 3, 4] = True
	before = next(model.parameters()).detach().clone()

	logits = model(embeddings, valid)
	loss, _ = fractional_horizon_cross_entropy(logits, target, mask)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	assert torch.isfinite(loss)
	assert all(
		parameter.grad is None or torch.isfinite(parameter.grad).all()
		for parameter in model.parameters()
	)
	optimizer.step()
	assert not torch.equal(before, next(model.parameters()).detach())
