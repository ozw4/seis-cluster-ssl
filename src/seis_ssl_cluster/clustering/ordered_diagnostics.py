"""Diagnostics for ordered stratigraphic token labels."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def ordered_label_diagnostics(
	labels: np.ndarray,
	*,
	k: int,
	z_axis: int = 2,
) -> dict[str, object]:
	"""Summarize whether a 3D label grid is ordered along the z axis."""
	grid = _validate_label_grid(labels, k=k, z_axis=z_axis)
	valid = (grid >= 0) & (grid < k)
	cluster_counts = {
		str(label): int(np.count_nonzero(grid == label))
		for label in range(k)
	}

	trace_valid = np.any(valid, axis=2)
	valid_trace_count = int(np.count_nonzero(trace_valid))
	trace_count = int(grid.shape[0] * grid.shape[1])
	pair_count = 0
	same_count = 0
	forward_count = 0
	reverse_count = 0
	jump_count = 0
	changed_pair_counts = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.int64)
	for x_index in range(grid.shape[0]):
		for y_index in range(grid.shape[1]):
			trace = grid[x_index, y_index, :]
			valid_trace = trace[trace >= 0]
			if valid_trace.size < 2:
				continue
			delta = np.diff(valid_trace)
			pair_count += int(delta.size)
			same_count += int(np.count_nonzero(delta == 0))
			forward_count += int(np.count_nonzero(delta > 0))
			reverse_count += int(np.count_nonzero(delta < 0))
			jump_count += int(np.count_nonzero(np.abs(delta) > 1))
			changed_pair_counts[x_index, y_index] = int(np.count_nonzero(delta != 0))
	valid_boundary_counts = changed_pair_counts[trace_valid]
	if valid_boundary_counts.size == 0:
		mean_boundaries = 0.0
		max_boundaries = 0
	else:
		mean_boundaries = float(np.mean(valid_boundary_counts, dtype=np.float64))
		max_boundaries = int(np.max(valid_boundary_counts))

	return {
		'token_grid_shape': [int(value) for value in grid.shape],
		'valid_token_count': int(np.count_nonzero(valid)),
		'invalid_token_count': int(np.count_nonzero(grid == -1)),
		'cluster_counts': cluster_counts,
		'vertical_adjacent_pair_count': pair_count,
		'same_transition_count': same_count,
		'forward_transition_count': forward_count,
		'reverse_transition_count': reverse_count,
		'jump_transition_count': jump_count,
		'reverse_transition_rate': _rate(reverse_count, pair_count),
		'jump_transition_rate': _rate(jump_count, pair_count),
		'trace_count': trace_count,
		'valid_trace_count': valid_trace_count,
		'mean_boundaries_per_valid_trace': mean_boundaries,
		'max_boundaries_per_valid_trace': max_boundaries,
	}


def ordered_boundary_summary(
	labels: np.ndarray,
	*,
	k: int,
	z_axis: int = 2,
) -> dict[str, object]:
	"""Summarize observed z positions for each adjacent ordered boundary."""
	grid = _validate_label_grid(labels, k=k, z_axis=z_axis)
	summary: dict[str, object] = {}
	for lower_label in range(k - 1):
		boundary = f'{lower_label}_to_{lower_label + 1}'
		threshold = lower_label + 1
		z_indices: list[int] = []
		for x_index in range(grid.shape[0]):
			for y_index in range(grid.shape[1]):
				trace = grid[x_index, y_index, :]
				valid_boundary_positions = np.flatnonzero(trace >= threshold)
				if valid_boundary_positions.size:
					z_indices.append(int(valid_boundary_positions[0]))
		if not z_indices:
			summary[boundary] = {
				'boundary': boundary,
				'observed_trace_count': 0,
				'mean_z': None,
				'min_z': None,
				'max_z': None,
				'std_z': None,
			}
			continue

		values = np.asarray(z_indices, dtype=np.float64)
		summary[boundary] = {
			'boundary': boundary,
			'observed_trace_count': int(values.size),
			'mean_z': float(np.mean(values)),
			'min_z': int(np.min(values)),
			'max_z': int(np.max(values)),
			'std_z': float(np.std(values)),
		}
	return summary


def aggregate_ordered_label_diagnostics(
	per_survey: dict[str, dict[str, object]],
	*,
	k: int,
) -> dict[str, object]:
	"""Aggregate ordered-label diagnostics across surveys."""
	_validate_positive_int(k, 'k')
	cluster_counts = {str(label): 0 for label in range(k)}
	aggregate: dict[str, Any] = {
		'survey_count': len(per_survey),
		'token_grid_shape': None,
		'valid_token_count': 0,
		'invalid_token_count': 0,
		'cluster_counts': cluster_counts,
		'vertical_adjacent_pair_count': 0,
		'same_transition_count': 0,
		'forward_transition_count': 0,
		'reverse_transition_count': 0,
		'jump_transition_count': 0,
		'trace_count': 0,
		'valid_trace_count': 0,
		'mean_boundaries_per_valid_trace': 0.0,
		'max_boundaries_per_valid_trace': 0,
	}
	weighted_boundary_total = 0.0
	for diagnostics in per_survey.values():
		for key in (
			'valid_token_count',
			'invalid_token_count',
			'vertical_adjacent_pair_count',
			'same_transition_count',
			'forward_transition_count',
			'reverse_transition_count',
			'jump_transition_count',
			'trace_count',
			'valid_trace_count',
		):
			aggregate[key] += int(diagnostics[key])
		for label in range(k):
			cluster_counts[str(label)] += int(
				dict(diagnostics['cluster_counts'])[str(label)],
			)
		valid_trace_count = int(diagnostics['valid_trace_count'])
		weighted_boundary_total += (
			float(diagnostics['mean_boundaries_per_valid_trace'])
			* valid_trace_count
		)
		aggregate['max_boundaries_per_valid_trace'] = max(
			int(aggregate['max_boundaries_per_valid_trace']),
			int(diagnostics['max_boundaries_per_valid_trace']),
		)

	pair_count = int(aggregate['vertical_adjacent_pair_count'])
	valid_trace_total = int(aggregate['valid_trace_count'])
	aggregate['reverse_transition_rate'] = _rate(
		int(aggregate['reverse_transition_count']),
		pair_count,
	)
	aggregate['jump_transition_rate'] = _rate(
		int(aggregate['jump_transition_count']),
		pair_count,
	)
	aggregate['mean_boundaries_per_valid_trace'] = (
		0.0
		if valid_trace_total == 0
		else float(weighted_boundary_total / valid_trace_total)
	)
	return aggregate


def _validate_label_grid(
	labels: np.ndarray,
	*,
	k: int,
	z_axis: int,
) -> np.ndarray:
	_validate_positive_int(k, 'k')
	if z_axis != 2:
		raise ValueError(f'z_axis must currently be 2; got {z_axis!r}')
	grid = np.asarray(labels)
	if grid.ndim != 3:
		raise ValueError(f'labels must be a 3D array; got shape {grid.shape!r}')
	if grid.dtype == np.bool_ or not np.issubdtype(grid.dtype, np.integer):
		raise TypeError(f'labels must be an integer array; got dtype {grid.dtype}')
	invalid_values = np.unique(grid[(grid < -1) | (grid >= k)])
	if invalid_values.size:
		values = [int(value) for value in invalid_values[:10]]
		raise ValueError(
			'labels contains values outside valid range '
			f'[-1, 0..{k - 1}]: {values!r}',
		)
	return grid


def _validate_positive_int(value: object, name: str) -> None:
	if isinstance(value, bool) or not isinstance(value, Integral):
		raise TypeError(f'{name} must be a positive integer')
	if int(value) <= 0:
		raise ValueError(f'{name} must be positive; got {value!r}')


def _rate(numerator: int, denominator: int) -> float:
	return 0.0 if denominator == 0 else float(numerator / denominator)


__all__ = [
	'aggregate_ordered_label_diagnostics',
	'ordered_boundary_summary',
	'ordered_label_diagnostics',
]
