from __future__ import annotations

import numpy as np

from seis_ssl_cluster.data import (
	ZeroMaskConfig,
	compute_zero_amplitude_invalid_mask,
	detect_all_zero_traces,
	detect_all_zero_z_samples,
)


def test_zero_mask_detects_all_zero_z_samples_with_z_radius() -> None:
	amplitude = np.ones((4, 5, 6), dtype=np.float32)
	amplitude[:, :, 2] = 0.0

	mask = compute_zero_amplitude_invalid_mask(
		amplitude,
		config=ZeroMaskConfig(
			z_sample_influence_radius=1,
			xy_trace_influence_radius=0,
		),
	)

	assert mask[:, :, 1:4].all()
	assert not mask[:, :, :1].any()
	assert not mask[:, :, 4:].any()


def test_zero_mask_detects_all_zero_traces_with_xy_radius() -> None:
	amplitude = np.ones((5, 5, 4), dtype=np.float32)
	amplitude[2, 3, :] = 0.0

	mask = compute_zero_amplitude_invalid_mask(
		amplitude,
		config=ZeroMaskConfig(
			z_sample_influence_radius=0,
			xy_trace_influence_radius=1,
		),
	)

	assert mask[1:4, 2:5, :].all()
	assert not mask[:1].any()
	assert not mask[4:].any()
	assert not mask[:, :2].any()


def test_zero_detection_uses_only_valid_voxels() -> None:
	amplitude = np.zeros((2, 2, 3), dtype=np.float32)
	amplitude[0, 0, 1] = 5.0
	valid = np.ones_like(amplitude, dtype=bool)
	valid[0, 0, 1] = False

	assert detect_all_zero_z_samples(
		amplitude,
		valid_mask=valid,
		zero_atol=0.0,
	).tolist() == [True, True, True]


def test_disabled_zero_mask_returns_all_valid_invalid_mask() -> None:
	amplitude = np.zeros((3, 3, 3), dtype=np.float32)

	mask = compute_zero_amplitude_invalid_mask(
		amplitude,
		config=ZeroMaskConfig(enabled=False),
	)

	assert mask.shape == amplitude.shape
	assert not mask.any()


def test_all_zero_trace_detection_respects_zero_atol() -> None:
	amplitude = np.ones((2, 2, 3), dtype=np.float32)
	amplitude[1, 1, :] = 0.05

	assert not detect_all_zero_traces(amplitude, zero_atol=0.0)[1, 1]
	assert detect_all_zero_traces(amplitude, zero_atol=0.1)[1, 1]
