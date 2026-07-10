"""Boundary-aware token distances and weights for stratigraphic labels."""

from __future__ import annotations

import numpy as np


def boundary_distance_tokens(
	labels: np.ndarray,
	valid_tokens: np.ndarray,
) -> np.ndarray:
	"""Return int32 distances to the nearest boundary within each valid run."""
	label_array, valid_array = _validate_inputs(labels, valid_tokens)
	return _boundary_distance_tokens(label_array, valid_array)


def boundary_weight_tokens(
	labels: np.ndarray,
	valid_tokens: np.ndarray,
	*,
	alpha: float,
	tau: float,
) -> np.ndarray:
	"""Return float32 boundary weights, with zero weight for invalid tokens."""
	label_array, valid_array = _validate_inputs(labels, valid_tokens)
	alpha_value = _validate_alpha(alpha)
	tau_value = _validate_tau(tau)

	distances = _boundary_distance_tokens(label_array, valid_array)
	weights = np.zeros(label_array.shape, dtype=np.float32)
	without_boundary = valid_array & (distances < 0)
	weights[without_boundary] = np.float32(1.0)
	with_boundary = distances >= 0
	weights[with_boundary] = (
		1.0 - alpha_value * np.exp(-distances[with_boundary] / tau_value)
	).astype(np.float32)
	return weights


def _boundary_distance_tokens(
	labels: np.ndarray,
	valid_tokens: np.ndarray,
) -> np.ndarray:
	distances = np.full(labels.shape, -1, dtype=np.int32)
	if labels.shape[-1] == 0:
		return distances

	boundaries = np.zeros(labels.shape, dtype=np.bool_)
	transitions = (
		valid_tokens[..., :-1]
		& valid_tokens[..., 1:]
		& (labels[..., :-1] != labels[..., 1:])
	)
	boundaries[..., :-1] |= transitions
	boundaries[..., 1:] |= transitions

	nearest = np.full(labels.shape[:-1], -1, dtype=np.int32)
	for z in range(labels.shape[-1]):
		nearest = _advance_distance(
			nearest,
			valid=valid_tokens[..., z],
			boundary=boundaries[..., z],
		)
		distances[..., z] = nearest

	nearest.fill(-1)
	for z in range(labels.shape[-1] - 1, -1, -1):
		nearest = _advance_distance(
			nearest,
			valid=valid_tokens[..., z],
			boundary=boundaries[..., z],
		)
		known = nearest >= 0
		use_nearest = known & (
			(distances[..., z] < 0) | (nearest < distances[..., z])
		)
		distances[..., z][use_nearest] = nearest[use_nearest]
	return distances


def _advance_distance(
	previous: np.ndarray,
	*,
	valid: np.ndarray,
	boundary: np.ndarray,
) -> np.ndarray:
	advanced = np.where(previous >= 0, previous + 1, -1)
	return np.where(valid, np.where(boundary, 0, advanced), -1).astype(
		np.int32,
		copy=False,
	)


def _validate_inputs(
	labels: np.ndarray,
	valid_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	label_array = np.asarray(labels)
	valid_array = np.asarray(valid_tokens)
	if label_array.ndim != 3:
		msg = f'labels must be 3D [TX, TY, TZ]; got shape {label_array.shape!r}'
		raise ValueError(msg)
	if valid_array.ndim != 3:
		msg = (
			'valid_tokens must be 3D [TX, TY, TZ]; '
			f'got shape {valid_array.shape!r}'
		)
		raise ValueError(msg)
	if label_array.shape != valid_array.shape:
		msg = (
			'labels and valid_tokens shapes must match; '
			f'got {label_array.shape!r} and {valid_array.shape!r}'
		)
		raise ValueError(msg)
	if label_array.dtype.kind not in {'i', 'u'}:
		msg = f'labels dtype must be integer; got {label_array.dtype}'
		raise TypeError(msg)
	if valid_array.dtype != np.bool_:
		msg = f'valid_tokens dtype must be bool; got {valid_array.dtype}'
		raise TypeError(msg)
	if np.any(label_array[valid_array] < 0):
		msg = 'labels where valid_tokens is true must be nonnegative'
		raise ValueError(msg)
	if np.any(label_array[~valid_array] != -1):
		msg = 'labels must be -1 where valid_tokens is false'
		raise ValueError(msg)
	return label_array, valid_array


def _validate_alpha(alpha: float) -> float:
	try:
		value = float(alpha)
	except (TypeError, ValueError) as exc:
		msg = f'alpha must be finite and in [0, 1]; got {alpha!r}'
		raise TypeError(msg) from exc
	if not np.isfinite(value) or not 0.0 <= value <= 1.0:
		msg = f'alpha must be finite and in [0, 1]; got {alpha!r}'
		raise ValueError(msg)
	return value


def _validate_tau(tau: float) -> float:
	try:
		value = float(tau)
	except (TypeError, ValueError) as exc:
		msg = f'tau must be finite and positive; got {tau!r}'
		raise TypeError(msg) from exc
	if not np.isfinite(value) or value <= 0.0:
		msg = f'tau must be finite and positive; got {tau!r}'
		raise ValueError(msg)
	return value


__all__ = ['boundary_distance_tokens', 'boundary_weight_tokens']
