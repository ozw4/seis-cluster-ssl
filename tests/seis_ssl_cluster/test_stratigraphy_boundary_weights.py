from __future__ import annotations

import numpy as np
import pytest

from seis_ssl_cluster.stratigraphy import (
	boundary_distance_tokens,
	boundary_weight_tokens,
)


def _trace(values: list[object], *, dtype: np.dtype | type = np.int32) -> np.ndarray:
	return np.asarray(values, dtype=dtype)[None, None, :]


def test_single_boundary_has_zero_distance_on_both_sides() -> None:
	labels = _trace([0, 0, 1, 1, 1])
	valid = np.ones(labels.shape, dtype=np.bool_)

	distances = boundary_distance_tokens(labels, valid)
	weights = boundary_weight_tokens(labels, valid, alpha=0.5, tau=2.0)

	np.testing.assert_array_equal(distances, _trace([1, 0, 0, 1, 2]))
	expected = 1.0 - 0.5 * np.exp(-distances / 2.0)
	np.testing.assert_allclose(weights, expected.astype(np.float32))
	assert distances.dtype == np.int32
	assert weights.dtype == np.float32


def test_multiple_boundaries_use_nearest_distance() -> None:
	labels = _trace([0, 0, 1, 1, 1, 2, 2, 2])
	valid = np.ones(labels.shape, dtype=np.bool_)

	distances = boundary_distance_tokens(labels, valid)

	np.testing.assert_array_equal(distances, _trace([1, 0, 0, 1, 0, 0, 1, 2]))


def test_invalid_gap_stops_boundary_distance_propagation() -> None:
	labels = _trace([0, 1, -1, 2, 2, 2])
	valid = _trace([True, True, False, True, True, True], dtype=np.bool_)

	distances = boundary_distance_tokens(labels, valid)
	weights = boundary_weight_tokens(labels, valid, alpha=0.5, tau=1.0)

	np.testing.assert_array_equal(distances, _trace([0, 0, -1, -1, -1, -1]))
	np.testing.assert_allclose(
		weights,
		_trace([0.5, 0.5, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
	)


def test_run_without_boundary_has_negative_distance_and_unity_weight() -> None:
	labels = _trace([3, 3, 3])
	valid = np.ones(labels.shape, dtype=np.bool_)

	distances = boundary_distance_tokens(labels, valid)
	weights = boundary_weight_tokens(labels, valid, alpha=1.0, tau=0.25)

	np.testing.assert_array_equal(distances, np.full(labels.shape, -1, dtype=np.int32))
	np.testing.assert_array_equal(weights, np.ones(labels.shape, dtype=np.float32))


def test_zero_alpha_matches_valid_mask_exactly() -> None:
	labels = _trace([0, 1, -1, 2])
	valid = _trace([True, True, False, True], dtype=np.bool_)

	weights = boundary_weight_tokens(labels, valid, alpha=0.0, tau=1.0)

	np.testing.assert_array_equal(weights, valid.astype(np.float32))


def test_small_tau_produces_finite_bounded_weights() -> None:
	labels = _trace([0, 0, 1, 1])
	valid = np.ones(labels.shape, dtype=np.bool_)

	weights = boundary_weight_tokens(labels, valid, alpha=1.0, tau=1e-12)

	assert np.all(np.isfinite(weights))
	assert np.all((weights >= 0.0) & (weights <= 1.0))


@pytest.mark.parametrize(
	('labels', 'valid', 'error'),
	[
		(np.zeros((1, 2), dtype=np.int32), np.ones((1, 2), dtype=np.bool_), '3D'),
		(
			np.zeros((1, 1, 2), dtype=np.int32),
			np.ones((1, 1, 1), dtype=np.bool_),
			'shapes must match',
		),
		(_trace([0.0], dtype=np.float32), _trace([True], dtype=np.bool_), 'integer'),
		(_trace([0]), _trace([1], dtype=np.int32), 'bool'),
		(_trace([-1]), _trace([True], dtype=np.bool_), 'nonnegative'),
		(_trace([0]), _trace([False], dtype=np.bool_), 'must be -1'),
	],
)
def test_invalid_array_contracts_are_rejected(
	labels: np.ndarray,
	valid: np.ndarray,
	error: str,
) -> None:
	with pytest.raises((TypeError, ValueError), match=error):
		boundary_distance_tokens(labels, valid)


@pytest.mark.parametrize('alpha', [-0.1, 1.1, np.nan, np.inf])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
	labels = _trace([0])
	valid = np.ones(labels.shape, dtype=np.bool_)

	with pytest.raises(ValueError, match='alpha'):
		boundary_weight_tokens(labels, valid, alpha=alpha, tau=1.0)


@pytest.mark.parametrize('tau', [0.0, -1.0, np.nan, np.inf])
def test_invalid_tau_is_rejected(tau: float) -> None:
	labels = _trace([0])
	valid = np.ones(labels.shape, dtype=np.bool_)

	with pytest.raises(ValueError, match='tau'):
		boundary_weight_tokens(labels, valid, alpha=0.5, tau=tau)


def test_inputs_are_not_modified() -> None:
	labels = _trace([0, 1, -1, 2])
	valid = _trace([True, True, False, True], dtype=np.bool_)
	labels_before = labels.copy()
	valid_before = valid.copy()

	boundary_distance_tokens(labels, valid)
	boundary_weight_tokens(labels, valid, alpha=0.5, tau=1.0)

	np.testing.assert_array_equal(labels, labels_before)
	np.testing.assert_array_equal(valid, valid_before)
