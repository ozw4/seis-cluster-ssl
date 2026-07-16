from __future__ import annotations

import numpy as np
import pytest

import seis_ssl_cluster.data.zero_mask as zero_mask_module
from seis_ssl_cluster.data import (
	ZeroMaskConfig,
	compute_zero_amplitude_invalid_mask,
	detect_all_zero_traces,
	detect_all_zero_z_samples,
	dilate_zero_sample_mask,
	dilate_zero_trace_mask,
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


@pytest.mark.parametrize('radius', [0, 1, 2, 3, 4])
def test_vectorized_dilation_matches_reference(radius: int) -> None:
	zero_z = np.asarray([True, False, False, True, False], dtype=bool)
	zero_traces = np.zeros((4, 5), dtype=bool)
	zero_traces[0, 0] = True
	zero_traces[2, 3] = True
	shape = (4, 5, 5)
	expected_z = np.zeros(shape, dtype=bool)
	for z_index in np.flatnonzero(zero_z):
		expected_z[:, :, max(0, z_index - radius) : z_index + radius + 1] = True
	expected_traces = np.zeros(shape, dtype=bool)
	for x_index, y_index in zip(*np.nonzero(zero_traces), strict=True):
		expected_traces[
			max(0, x_index - radius) : x_index + radius + 1,
			max(0, y_index - radius) : y_index + radius + 1,
			:,
		] = True

	np.testing.assert_array_equal(
		dilate_zero_sample_mask(zero_z, shape, radius_z=radius),
		expected_z,
	)
	np.testing.assert_array_equal(
		dilate_zero_trace_mask(zero_traces, shape, radius_xy=radius),
		expected_traces,
	)


def test_combined_zero_mask_prepares_amplitude_once(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	calls = 0
	original = zero_mask_module._as_amplitude_and_valid_mask  # noqa: SLF001

	def counted_prepare(
		amplitude: np.ndarray,
		valid_mask: np.ndarray | None,
	) -> tuple[np.ndarray, np.ndarray | None]:
		nonlocal calls
		calls += 1
		return original(amplitude, valid_mask)

	monkeypatch.setattr(
		zero_mask_module,
		'_as_amplitude_and_valid_mask',
		counted_prepare,
	)
	compute_zero_amplitude_invalid_mask(np.ones((3, 4, 5), dtype=np.float32))

	assert calls == 1
