"""Artifact-independent warm-start refresh for ordered HMM centers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real

import numpy as np

from seis_ssl_cluster.clustering import stratigraphic_hmm as hmm
from seis_ssl_cluster.clustering.ordered_diagnostics import (
	aggregate_ordered_label_diagnostics,
	ordered_boundary_summary,
	ordered_label_diagnostics,
)
from seis_ssl_cluster.clustering.prepared_features import (
	PreparedFeatureStore,
	PreparedSurveyFeatures,
)
from seis_ssl_cluster.utils import StageTimer  # noqa: TC001

DEFAULT_REFRESH_PREDICTION_BATCH_SIZE = 65_536


@dataclass(frozen=True)
class WarmStartOrderedHMMRefreshIterationDiagnostics:
	"""Diagnostics produced by one warm-start center replacement."""

	iteration: int
	cluster_counts: dict[int, int]
	empty_states: list[int]
	center_shift_l2: list[float]
	total_center_shift_l2: float


@dataclass(frozen=True)
class WarmStartOrderedHMMRefreshResult:
	"""Final centers, labels, and diagnostics for one pure HMM refresh."""

	centers: np.ndarray
	labels_by_survey: dict[str, np.ndarray]
	iteration_diagnostics: tuple[WarmStartOrderedHMMRefreshIterationDiagnostics, ...]
	final_state_counts: dict[int, int]
	final_boundary_counts: dict[str, int]
	final_transition_counts: dict[str, int]
	final_state_mean_z: dict[int, float | None]
	final_ordered_diagnostics: dict[str, object]
	final_boundary_summary: dict[str, dict[str, object]]

	@property
	def final_centers(self) -> np.ndarray:
		"""Return the final centers under an explicit result-oriented name."""
		return self.centers

	@property
	def final_labels_by_survey(self) -> dict[str, np.ndarray]:
		"""Return final labels under an explicit result-oriented name."""
		return self.labels_by_survey


def run_warm_start_ordered_hmm_refresh(  # noqa: PLR0913
	prepared_features: PreparedFeatureStore,
	previous_centers: np.ndarray,
	*,
	transition_costs: np.ndarray,
	initial_state_costs: np.ndarray | None,
	terminal_state_costs: np.ndarray | None,
	expected_boundaries: hmm.HMMExpectedBoundariesSettings | None,
	iterations: int,
	empty_cluster_policy: str = 'error',
	prediction_batch_size: int = DEFAULT_REFRESH_PREDICTION_BATCH_SIZE,
	timer: StageTimer | None = None,
) -> WarmStartOrderedHMMRefreshResult:
	"""Refresh ordered HMM centers from a previous generation.

	Each requested iteration decodes every prepared survey and replaces every
	non-empty center with the exact mean of its assigned prepared feature rows.
	The input centers and prepared feature store are never modified.  State row
	identity is preserved throughout; no initialization, permutation, or
	depth-based reordering is performed.
	"""
	hmm._validate_positive_int(iterations, 'iterations')  # noqa: SLF001
	hmm._validate_positive_int(  # noqa: SLF001
		prediction_batch_size,
		'prediction_batch_size',
	)
	if empty_cluster_policy != 'error':
		raise ValueError("empty_cluster_policy must be 'error'")

	centers = _copy_and_validate_centers(previous_centers)
	k = centers.shape[0]
	feature_dim = _validate_prepared_feature_store(
		prepared_features,
		prediction_batch_size=prediction_batch_size,
	)
	if centers.shape[1] != feature_dim:
		raise ValueError(
			'prepared feature dimension must match centers; got '
			f'{feature_dim} and {centers.shape[1]}'
		)
	transition_matrix, initial_costs, terminal_costs = _validate_hmm_costs(
		k,
		transition_costs,
		initial_state_costs,
		terminal_state_costs,
	)
	_validate_expected_boundaries(
		expected_boundaries,
		k=k,
		max_trace_length=max(
			int(survey.token_shape_xyz[2])
			for survey in prepared_features.surveys
		),
	)

	iteration_diagnostics: list[WarmStartOrderedHMMRefreshIterationDiagnostics] = []
	for iteration in range(1, int(iterations) + 1):
		labels_by_survey = _decode_and_validate_labels(
			prepared_features,
			centers=centers,
			transition_costs=transition_matrix,
			initial_state_costs=initial_costs,
			terminal_state_costs=terminal_costs,
			expected_boundaries=expected_boundaries,
			k=k,
			prediction_batch_size=prediction_batch_size,
			timer=timer,
		)
		updated_centers, summary = hmm.update_centers_from_prepared_labels(
			prepared_features,
			labels_by_survey,
			centers=centers,
			prediction_batch_size=prediction_batch_size,
			empty_cluster_policy='keep_previous',
			timer=timer,
		)
		updated_centers = _copy_and_validate_updated_centers(
			updated_centers,
			previous_shape=centers.shape,
		)
		cluster_counts = _validated_cluster_counts(summary, k=k)
		empty_states = [state for state in range(k) if cluster_counts[state] == 0]
		if empty_states:
			raise ValueError(
				'empty HMM state(s) after center update: '
				f'{empty_states!r}'
			)
		center_shift_l2 = _validated_center_shifts(summary, k=k)
		total_center_shift_l2 = _validated_total_center_shift(summary)
		iteration_diagnostics.append(
			WarmStartOrderedHMMRefreshIterationDiagnostics(
				iteration=iteration,
				cluster_counts=cluster_counts,
				empty_states=empty_states,
				center_shift_l2=center_shift_l2,
				total_center_shift_l2=total_center_shift_l2,
			)
		)
		centers = updated_centers

	final_labels_by_survey = _decode_and_validate_labels(
		prepared_features,
		centers=centers,
		transition_costs=transition_matrix,
		initial_state_costs=initial_costs,
		terminal_state_costs=terminal_costs,
		expected_boundaries=expected_boundaries,
		k=k,
		prediction_batch_size=prediction_batch_size,
		timer=timer,
	)
	final_state_counts, final_state_mean_z = _final_state_statistics(
		prepared_features,
		final_labels_by_survey,
		k=k,
		prediction_batch_size=prediction_batch_size,
	)
	final_empty_states = [
		state for state in range(k) if final_state_counts[state] == 0
	]
	if final_empty_states:
		raise ValueError(
			'empty HMM state(s) in final decode: '
			f'{final_empty_states!r}'
		)
	final_ordered_diagnostics, final_boundary_summary = _final_ordered_diagnostics(
		final_labels_by_survey,
		k=k,
	)
	final_transition_counts = {
		'same': int(final_ordered_diagnostics['same_transition_count']),
		'forward': int(final_ordered_diagnostics['forward_transition_count']),
		'reverse': int(final_ordered_diagnostics['reverse_transition_count']),
		'jump': int(final_ordered_diagnostics['jump_transition_count']),
	}
	final_boundary_counts = _final_boundary_counts(final_labels_by_survey, k=k)

	return WarmStartOrderedHMMRefreshResult(
		centers=centers.copy(),
		labels_by_survey={
			survey_id: np.asarray(labels, dtype=np.int32).copy()
			for survey_id, labels in final_labels_by_survey.items()
		},
		iteration_diagnostics=tuple(iteration_diagnostics),
		final_state_counts=final_state_counts,
		final_boundary_counts=final_boundary_counts,
		final_transition_counts=final_transition_counts,
		final_state_mean_z=final_state_mean_z,
		final_ordered_diagnostics=final_ordered_diagnostics,
		final_boundary_summary=final_boundary_summary,
	)


def _copy_and_validate_centers(value: np.ndarray) -> np.ndarray:
	try:
		centers = np.array(value, dtype=np.float32, copy=True)
	except (TypeError, ValueError) as exc:
		raise TypeError('centers must be a numeric array') from exc
	if centers.ndim != 2 or centers.shape[0] == 0:
		raise ValueError(
			f'centers must be a non-empty 2D matrix; got {centers.shape!r}'
		)
	if centers.shape[1] == 0:
		raise ValueError('centers must have a positive feature dimension')
	if not np.all(np.isfinite(centers)):
		raise ValueError('centers must contain only finite values')
	return centers


def _copy_and_validate_updated_centers(
	value: np.ndarray,
	*,
	previous_shape: tuple[int, int],
) -> np.ndarray:
	try:
		centers = np.array(value, dtype=np.float32, copy=True)
	except (TypeError, ValueError) as exc:
		raise ValueError('updated centers must be a numeric matrix') from exc
	if centers.shape != previous_shape:
		raise ValueError(
			'updated centers must preserve center shape; got '
			f'{centers.shape!r}, expected {previous_shape!r}'
		)
	if not np.all(np.isfinite(centers)):
		raise ValueError('updated centers must contain only finite values')
	return centers


def _validate_prepared_feature_store(  # noqa: C901, PLR0912
	prepared_features: PreparedFeatureStore,
	*,
	prediction_batch_size: int,
) -> int:
	if not isinstance(prepared_features, PreparedFeatureStore):
		raise TypeError('prepared_features must be a PreparedFeatureStore')
	surveys = tuple(prepared_features.surveys)
	if not surveys:
		raise ValueError('prepared_features must contain at least one survey')
	survey_ids = [survey.embedding_input.survey_id for survey in surveys]
	if len(set(survey_ids)) != len(survey_ids):
		raise ValueError('prepared_features must contain unique survey ids')
	feature_dims: set[int] = set()
	for survey in surveys:
		shape = tuple(survey.token_shape_xyz)
		if len(shape) != 3 or any(int(value) <= 0 for value in shape):
			raise ValueError(
				f'invalid token grid shape for {survey.embedding_input.survey_id!r}: '
				f'{shape!r}'
			)
		if isinstance(survey.feature_dim, bool) or not isinstance(
			survey.feature_dim,
			Integral,
		) or int(survey.feature_dim) <= 0:
			raise ValueError(
				f'invalid prepared feature dimension for '
				f'{survey.embedding_input.survey_id!r}'
			)
		if isinstance(survey.valid_token_count, bool) or not isinstance(
			survey.valid_token_count,
			Integral,
		) or int(survey.valid_token_count) < 0:
			raise ValueError(
				f'invalid prepared feature count for '
				f'{survey.embedding_input.survey_id!r}'
			)
		feature_dims.add(int(survey.feature_dim))
		seen = 0
		with survey.open() as opened:
			for indices, features in opened.iter_feature_chunks(prediction_batch_size):
				index_array = np.asarray(indices)
				feature_array = np.asarray(features)
				if index_array.ndim != 1:
					raise ValueError(
						f'prepared feature indices must be 1D for '
						f'{survey.embedding_input.survey_id}'
					)
				if not np.issubdtype(index_array.dtype, np.integer):
					raise TypeError(
						f'prepared feature indices must be integer for '
						f'{survey.embedding_input.survey_id}'
					)
				if np.any(index_array < 0) or np.any(
					index_array >= np.prod(shape, dtype=np.int64)
				):
					raise ValueError(
						f'prepared feature indices are outside the token grid for '
						f'{survey.embedding_input.survey_id}'
					)
				if feature_array.ndim != 2 or feature_array.shape != (
					index_array.size,
					int(survey.feature_dim),
				):
					raise ValueError(
						f'prepared feature chunk shape is invalid for '
						f'{survey.embedding_input.survey_id}'
					)
				if not np.all(np.isfinite(feature_array)):
					raise ValueError(
						f'prepared features must contain only finite values for '
						f'{survey.embedding_input.survey_id}'
					)
				seen += int(index_array.size)
		if seen != int(survey.valid_token_count):
			raise ValueError(
				'prepared feature count mismatch for '
				f'{survey.embedding_input.survey_id}: '
				f'{seen} rows, expected {survey.valid_token_count}'
			)
	if len(feature_dims) != 1:
		raise ValueError('all prepared surveys must have the same feature dimension')
	return next(iter(feature_dims))


def _validate_hmm_costs(
	k: int,
	transition_costs: np.ndarray,
	initial_state_costs: np.ndarray | None,
	terminal_state_costs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	transition_matrix = hmm._as_float_matrix(  # noqa: SLF001
		transition_costs,
		'transition_costs',
	)
	if transition_matrix.shape != (k, k):
		raise ValueError(
			f'transition_costs must have shape ({k}, {k}); '
			f'got {transition_matrix.shape}'
		)
	hmm._validate_cost_matrix(transition_matrix, 'transition_costs')  # noqa: SLF001
	initial = hmm._optional_state_costs(  # noqa: SLF001
		initial_state_costs,
		k,
		'initial_state_costs',
	)
	terminal = hmm._optional_state_costs(  # noqa: SLF001
		terminal_state_costs,
		k,
		'terminal_state_costs',
	)
	return (
		transition_matrix.copy(),
		initial.copy(),
		terminal.copy(),
	)


def _validate_expected_boundaries(
	settings: hmm.HMMExpectedBoundariesSettings | None,
	*,
	k: int,
	max_trace_length: int,
) -> None:
	if settings is None:
		return
	if not isinstance(settings, hmm.HMMExpectedBoundariesSettings):
		raise TypeError(
			'expected_boundaries must be HMMExpectedBoundariesSettings or None'
		)
	if not isinstance(settings.enabled, bool):
		raise TypeError('expected_boundaries.enabled must be a boolean')
	if (
		isinstance(settings.weight, bool)
		or not isinstance(settings.weight, Real)
		or not np.isfinite(float(settings.weight))
		or float(settings.weight) < 0.0
	):
		raise ValueError('expected_boundaries.weight must be finite and non-negative')
	if not settings.enabled:
		return
	hmm._resolve_expected_boundary_count(  # noqa: SLF001
		settings,
		k=k,
		valid_trace_length=max_trace_length,
	)


def _decode_and_validate_labels(  # noqa: PLR0913
	prepared_features: PreparedFeatureStore,
	*,
	centers: np.ndarray,
	transition_costs: np.ndarray,
	initial_state_costs: np.ndarray,
	terminal_state_costs: np.ndarray,
	expected_boundaries: hmm.HMMExpectedBoundariesSettings | None,
	k: int,
	prediction_batch_size: int,
	timer: StageTimer | None,
) -> dict[str, np.ndarray]:
	labels_by_survey = hmm._decode_all_surveys(  # noqa: SLF001
		prepared_features,
		centers=centers,
		transition_costs=transition_costs,
		initial_state_costs=initial_state_costs,
		terminal_state_costs=terminal_state_costs,
		expected_boundaries=expected_boundaries,
		timer=timer,
	)
	_validate_label_grids(
		prepared_features,
		labels_by_survey,
		k=k,
		prediction_batch_size=prediction_batch_size,
	)
	return {
		survey_id: np.asarray(labels, dtype=np.int32).copy()
		for survey_id, labels in labels_by_survey.items()
	}


def _validate_label_grids(
	prepared_features: PreparedFeatureStore,
	labels_by_survey: Mapping[str, np.ndarray],
	*,
	k: int,
	prediction_batch_size: int,
) -> None:
	expected = {
		survey.embedding_input.survey_id: survey for survey in prepared_features.surveys
	}
	unknown = sorted(set(labels_by_survey) - set(expected))
	if unknown:
		raise ValueError(f'labels_by_survey contains unknown survey ids: {unknown!r}')
	missing = sorted(set(expected) - set(labels_by_survey))
	if missing:
		raise ValueError(f'labels_by_survey is missing survey ids: {missing!r}')
	for survey_id, survey in expected.items():
		grid = np.asarray(labels_by_survey[survey_id])
		if grid.shape != survey.token_shape_xyz:
			raise ValueError(f'label grid shape is invalid for {survey_id}')
		if grid.dtype == np.bool_ or not np.issubdtype(grid.dtype, np.integer):
			raise TypeError(f'labels must be an integer array for {survey_id}')
		invalid_values = np.unique(grid[(grid < -1) | (grid >= k)])
		if invalid_values.size:
			values = [int(value) for value in invalid_values[:10]]
			raise ValueError(
				f'label id out of range for {survey_id}: {values!r}'
			)
		_validate_prepared_label_alignment(
			survey,
			grid,
			prediction_batch_size=prediction_batch_size,
		)


def _validate_prepared_label_alignment(
	survey: PreparedSurveyFeatures,
	labels: np.ndarray,
	*,
	prediction_batch_size: int,
) -> None:
	flat_labels = labels.reshape(-1)
	prepared_count = 0
	# Avoid materializing an additional full valid-token mask.
	with survey.open() as opened:
		for indices, _features in opened.iter_feature_chunks(prediction_batch_size):
			index_array = np.asarray(indices, dtype=np.int64)
			if np.any(flat_labels[index_array] < 0):
				raise ValueError(
					f'label grid does not cover all prepared features for '
					f'{survey.embedding_input.survey_id}'
				)
			prepared_count += int(index_array.size)
	grid_count = int(np.count_nonzero(flat_labels >= 0))
	if grid_count != prepared_count:
		raise ValueError(
			f'label grid contains tokens outside prepared features for '
			f'{survey.embedding_input.survey_id}'
		)


def _validated_cluster_counts(
	summary: Mapping[str, object],
	*,
	k: int,
) -> dict[int, int]:
	value = summary.get('cluster_counts')
	if not isinstance(value, Mapping):
		raise TypeError('center update did not return cluster counts')
	counts: dict[int, int] = {}
	for state in range(k):
		if state in value:
			count_value = value[state]
		elif str(state) in value:
			count_value = value[str(state)]
		else:
			raise ValueError(f'center update omitted cluster count for state {state}')
		if isinstance(count_value, bool) or not isinstance(count_value, Integral):
			raise TypeError(f'invalid cluster count for state {state}')
		if int(count_value) < 0:
			raise ValueError(f'negative cluster count for state {state}')
		counts[state] = int(count_value)
	return counts


def _validated_center_shifts(summary: Mapping[str, object], *, k: int) -> list[float]:
	value = summary.get('center_shift_l2')
	if value is None:
		raise ValueError('center update did not return per-state center shifts')
	shifts = np.asarray(value, dtype=np.float64)
	if shifts.shape != (k,) or not np.all(np.isfinite(shifts)) or np.any(shifts < 0.0):
		raise ValueError('center update returned invalid per-state center shifts')
	return [float(item) for item in shifts]


def _validated_total_center_shift(summary: Mapping[str, object]) -> float:
	value = summary.get('total_center_shift_l2')
	if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
		raise TypeError('center update did not return a total center shift')
	total = float(value)
	if not np.isfinite(total) or total < 0.0:
		raise ValueError('center update returned an invalid total center shift')
	return total


def _final_state_statistics(
	prepared_features: PreparedFeatureStore,
	labels_by_survey: Mapping[str, np.ndarray],
	*,
	k: int,
	prediction_batch_size: int,
) -> tuple[dict[int, int], dict[int, float | None]]:
	counts = np.zeros(k, dtype=np.int64)
	z_sums = np.zeros(k, dtype=np.float64)
	for survey in prepared_features.surveys:
		survey_id = survey.embedding_input.survey_id
		flat_labels = np.asarray(labels_by_survey[survey_id]).reshape(-1)
		z_count = int(survey.token_shape_xyz[2])
		with survey.open() as opened:
			for indices, _features in opened.iter_feature_chunks(prediction_batch_size):
				index_array = np.asarray(indices, dtype=np.int64)
				batch_labels = np.asarray(flat_labels[index_array], dtype=np.int64)
				counts += np.bincount(batch_labels, minlength=k)
				z_values = (index_array % z_count).astype(np.float64, copy=False)
				z_sums += np.bincount(
					batch_labels,
					weights=z_values,
					minlength=k,
				)
	state_counts = {state: int(counts[state]) for state in range(k)}
	state_mean_z = {
		state: (
				None
				if counts[state] == 0
				else float(z_sums[state] / float(counts[state]))
		)
		for state in range(k)
	}
	return state_counts, state_mean_z


def _final_ordered_diagnostics(
	labels_by_survey: Mapping[str, np.ndarray],
	*,
	k: int,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
	per_survey = {
		survey_id: ordered_label_diagnostics(labels, k=k)
		for survey_id, labels in labels_by_survey.items()
	}
	boundary_summary = {
		survey_id: ordered_boundary_summary(labels, k=k)
		for survey_id, labels in labels_by_survey.items()
	}
	return aggregate_ordered_label_diagnostics(per_survey, k=k), boundary_summary


def _final_boundary_counts(
	labels_by_survey: Mapping[str, np.ndarray],
	*,
	k: int,
) -> dict[str, int]:
	counts = {f'{state}_to_{state + 1}': 0 for state in range(k - 1)}
	for labels in labels_by_survey.values():
		grid = np.asarray(labels)
		for x_index in range(grid.shape[0]):
			for y_index in range(grid.shape[1]):
				trace = grid[x_index, y_index, :]
				valid_trace = trace[trace >= 0]
				for previous, current in pairwise(valid_trace):
					if current == previous:
						continue
					lower = min(int(previous), int(current))
					upper = max(int(previous), int(current))
					for boundary in range(lower, upper):
						counts[f'{boundary}_to_{boundary + 1}'] += 1
	return counts


__all__ = [
	'DEFAULT_REFRESH_PREDICTION_BATCH_SIZE',
	'WarmStartOrderedHMMRefreshIterationDiagnostics',
	'WarmStartOrderedHMMRefreshResult',
	'run_warm_start_ordered_hmm_refresh',
]
