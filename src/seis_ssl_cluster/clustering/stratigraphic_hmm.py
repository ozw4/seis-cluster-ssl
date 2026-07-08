"""Stratigraphic HMM clustering backend scaffold."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import joblib
import numpy as np

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	discover_embedding_inputs,
	embedding_input_metadata,
	extract_token_features,
	load_valid_tokens,
	open_embedding_array,
	validate_compatible_embedding_inputs,
)
from seis_ssl_cluster.clustering.kmeans import (
	ClusteringRunResult,
	KClusteringResult,
	PCASettings,
	ResidualizationSettings,
	_aggregate_counts,
	_common_metadata,
	apply_residualizer_to_sample,
	clustering_settings_from_config,
	fit_minibatch_kmeans,
	fit_preprocessor,
	fit_residualizer,
)
from seis_ssl_cluster.clustering.ordered_diagnostics import (
	aggregate_ordered_label_diagnostics,
	ordered_boundary_summary,
	ordered_label_diagnostics,
)
from seis_ssl_cluster.clustering.residualization import (
	LocalTokenPositionResidualizer,
	residualization_keys_for_flat_indices,
	write_residualizer_npz,
)
from seis_ssl_cluster.clustering.sampling import (
	SampledTokens,
)
from seis_ssl_cluster.clustering.writer import SurveyLabelResult, write_json


@dataclass(frozen=True)
class HMMTransitionSettings:
	"""Transition penalties for ordered stratigraphic HMM decoding."""

	same_cost: float
	advance_cost: float
	jump_cost: float
	reverse_cost: float
	forbid_reverse: bool
	max_jump: int | None


@dataclass(frozen=True)
class HMMAnchorPriorSettings:
	"""Path-prior anchor settings for one endpoint."""

	mode: str
	weight: float


@dataclass(frozen=True)
class HMMExpectedBoundariesSettings:
	"""Path-prior expected boundary count settings."""

	enabled: bool
	target: str | int
	weight: float


@dataclass(frozen=True)
class HMMPathPriorSettings:
	"""Optional path-prior settings for future HMM decoding."""

	enabled: bool
	initial_state: HMMAnchorPriorSettings
	terminal_state: HMMAnchorPriorSettings
	expected_boundaries: HMMExpectedBoundariesSettings


@dataclass(frozen=True)
class StratigraphicHMMSettings:
	"""Validated stratigraphic HMM backend settings."""

	emission_source: str
	iterations: int
	z_axis: int
	z_direction: str
	transition: HMMTransitionSettings
	init_order_by: str
	empty_cluster_policy: str
	edge_margin_tokens: tuple[int, int, int]
	path_prior: HMMPathPriorSettings


def stratigraphic_hmm_settings_from_config(
	config: Mapping[str, object],
) -> StratigraphicHMMSettings:
	"""Build stratigraphic HMM settings from a resolved config mapping."""
	clustering = _required_mapping(config, 'clustering')
	hmm = _required_mapping(clustering, 'stratigraphic_hmm')
	transition = _required_mapping(hmm, 'transition')
	init = _required_mapping(hmm, 'init')
	update = _required_mapping(hmm, 'update')
	path_prior = _path_prior_settings_from_hmm_config(hmm)
	return StratigraphicHMMSettings(
		emission_source=str(hmm.get('emission_source', 'embedding')),
		iterations=int(hmm['iterations']),
		z_axis=int(hmm['z_axis']),
		z_direction=str(hmm['z_direction']),
		transition=HMMTransitionSettings(
			same_cost=float(transition['same_cost']),
			advance_cost=float(transition['advance_cost']),
			jump_cost=float(transition['jump_cost']),
			reverse_cost=float(transition['reverse_cost']),
			forbid_reverse=bool(transition['forbid_reverse']),
			max_jump=(
				None if transition['max_jump'] is None else int(transition['max_jump'])
			),
		),
		init_order_by=str(init['order_by']),
		empty_cluster_policy=str(update['empty_cluster_policy']),
		edge_margin_tokens=_edge_margin_tokens_from_hmm_config(hmm),
		path_prior=path_prior,
	)


def _edge_margin_tokens_from_hmm_config(
	hmm: Mapping[str, object],
) -> tuple[int, int, int]:
	value = hmm.get('edge_margin_tokens', (0, 0, 0))
	return (int(value[0]), int(value[1]), int(value[2]))  # type: ignore[index]


def edge_margin_mask_for_shape(
	shape: tuple[int, int, int],
	edge_margin_tokens: tuple[int, int, int],
) -> np.ndarray:
	"""Return a boolean mask for tokens inside the configured edge margins."""
	_validate_edge_margin_shape(shape, edge_margin_tokens, survey_id=None)
	return _edge_margin_mask_for_validated_shape(shape, edge_margin_tokens)


def _edge_margin_mask_for_validated_shape(
	shape: tuple[int, int, int],
	edge_margin_tokens: tuple[int, int, int],
) -> np.ndarray:
	x_count, y_count, z_count = shape
	mx, my, mz = edge_margin_tokens
	mask = np.zeros(shape, dtype=np.bool_)
	mask[
		mx : x_count - mx,
		my : y_count - my,
		mz : z_count - mz,
	] = True
	return mask


def hmm_valid_token_mask(
	embedding_input: EmbeddingInput,
	edge_margin_tokens: tuple[int, int, int],
) -> np.ndarray:
	"""Return original valid tokens after excluding configured HMM edge margins."""
	valid = np.asarray(load_valid_tokens(embedding_input), dtype=np.bool_)
	shape = valid.shape
	_validate_edge_margin_shape(
		shape,
		edge_margin_tokens,
		survey_id=embedding_input.survey_id,
	)
	return valid & edge_margin_mask_for_shape(shape, edge_margin_tokens)


def hmm_valid_flat_indices(
	embedding_input: EmbeddingInput,
	edge_margin_tokens: tuple[int, int, int],
) -> np.ndarray:
	"""Return flattened HMM-valid token indices for one survey."""
	return np.flatnonzero(
		hmm_valid_token_mask(embedding_input, edge_margin_tokens).reshape(-1),
	)


def _path_prior_settings_from_hmm_config(
	hmm: Mapping[str, object],
) -> HMMPathPriorSettings:
	if 'path_prior' not in hmm:
		return _disabled_path_prior_settings()
	path_prior = _required_mapping(hmm, 'path_prior')
	if not bool(path_prior['enabled']):
		return HMMPathPriorSettings(
			enabled=False,
			initial_state=_anchor_prior_settings_from_config(
				path_prior,
				'initial_state',
				default=HMMAnchorPriorSettings(mode='none', weight=0.0),
			),
			terminal_state=_anchor_prior_settings_from_config(
				path_prior,
				'terminal_state',
				default=HMMAnchorPriorSettings(mode='none', weight=0.0),
			),
			expected_boundaries=_expected_boundaries_settings_from_config(
				path_prior,
				default=HMMExpectedBoundariesSettings(
					enabled=False,
					target='auto_k_minus_1',
					weight=0.0,
				),
			),
		)
	return HMMPathPriorSettings(
		enabled=True,
		initial_state=_anchor_prior_settings_from_config(
			path_prior,
			'initial_state',
			default=HMMAnchorPriorSettings(mode='none', weight=0.0),
		),
		terminal_state=_anchor_prior_settings_from_config(
			path_prior,
			'terminal_state',
			default=HMMAnchorPriorSettings(mode='none', weight=0.0),
		),
		expected_boundaries=_expected_boundaries_settings_from_config(
			path_prior,
			default=HMMExpectedBoundariesSettings(
				enabled=False,
				target='auto_k_minus_1',
				weight=0.0,
			),
		),
	)


def _disabled_path_prior_settings() -> HMMPathPriorSettings:
	return HMMPathPriorSettings(
		enabled=False,
		initial_state=HMMAnchorPriorSettings(mode='none', weight=0.0),
		terminal_state=HMMAnchorPriorSettings(mode='none', weight=0.0),
		expected_boundaries=HMMExpectedBoundariesSettings(
			enabled=False,
			target='auto_k_minus_1',
			weight=0.0,
		),
	)


def _anchor_prior_settings_from_config(
	path_prior: Mapping[str, object],
	key: str,
	*,
	default: HMMAnchorPriorSettings,
) -> HMMAnchorPriorSettings:
	if key not in path_prior:
		return default
	anchor = _required_mapping(path_prior, key)
	return HMMAnchorPriorSettings(
		mode=str(anchor['mode']),
		weight=float(anchor['weight']),
	)


def _expected_boundaries_settings_from_config(
	path_prior: Mapping[str, object],
	*,
	default: HMMExpectedBoundariesSettings,
) -> HMMExpectedBoundariesSettings:
	if 'expected_boundaries' not in path_prior:
		return default
	boundaries = _required_mapping(path_prior, 'expected_boundaries')
	return HMMExpectedBoundariesSettings(
		enabled=bool(boundaries['enabled']),
		target=boundaries.get('target', default.target),
		weight=float(boundaries.get('weight', default.weight)),
	)


def build_ordered_transition_costs(
	k: int,
	settings: HMMTransitionSettings,
) -> np.ndarray:
	"""Build ordered-state transition costs with rows as previous states."""
	_validate_positive_int(k, 'k')
	_validate_transition_settings(settings)

	previous = np.arange(k)[:, np.newaxis]
	next_state = np.arange(k)[np.newaxis, :]
	delta = next_state - previous

	costs = np.full((k, k), np.inf, dtype=np.float32)
	same = delta == 0
	forward = delta > 0
	reverse = delta < 0

	costs[same] = np.float32(settings.same_cost)
	costs[forward] = np.float32(
		settings.advance_cost + settings.jump_cost * (delta[forward] - 1),
	)
	if not settings.forbid_reverse:
		costs[reverse] = np.float32(
			settings.reverse_cost + settings.jump_cost * (-delta[reverse] - 1),
		)

	if settings.max_jump is not None:
		too_far = np.abs(delta) > settings.max_jump
		costs[too_far & ~same] = np.inf

	return costs


def build_initial_state_costs(k: int, settings: HMMPathPriorSettings) -> np.ndarray:
	"""Build soft initial-state costs for ordered HMM decoding."""
	_validate_positive_int(k, 'k')
	if not settings.enabled or settings.initial_state.mode == 'none':
		return np.zeros(k, dtype=np.float32)
	if settings.initial_state.mode != 'shallow_anchor':
		msg = (
			'unsupported initial_state path prior mode: '
			f'{settings.initial_state.mode!r}'
		)
		raise ValueError(msg)
	_validate_nonnegative_finite_cost(
		settings.initial_state.weight,
		'initial_state.weight',
	)
	denominator = float(max(k - 1, 1))
	costs = settings.initial_state.weight * np.arange(k, dtype=np.float32) / denominator
	return _validate_state_costs(costs, k, 'initial_state_costs').astype(np.float32)


def build_terminal_state_costs(k: int, settings: HMMPathPriorSettings) -> np.ndarray:
	"""Build soft terminal-state costs for ordered HMM decoding."""
	_validate_positive_int(k, 'k')
	if not settings.enabled or settings.terminal_state.mode == 'none':
		return np.zeros(k, dtype=np.float32)
	if settings.terminal_state.mode != 'deep_anchor':
		msg = (
			'unsupported terminal_state path prior mode: '
			f'{settings.terminal_state.mode!r}'
		)
		raise ValueError(msg)
	_validate_nonnegative_finite_cost(
		settings.terminal_state.weight,
		'terminal_state.weight',
	)
	denominator = float(max(k - 1, 1))
	states = np.arange(k, dtype=np.float32)
	costs = settings.terminal_state.weight * (float(k - 1) - states) / denominator
	return _validate_state_costs(costs, k, 'terminal_state_costs').astype(np.float32)


def sample_token_z_coordinates(
	embedding_inputs: tuple[EmbeddingInput, ...],
	per_survey_token_indices: Mapping[str, np.ndarray],
) -> np.ndarray:
	"""Return sampled token z coordinates ordered like sampled feature rows."""
	inputs_by_survey = {item.survey_id: item for item in embedding_inputs}
	unknown = sorted(set(per_survey_token_indices) - set(inputs_by_survey))
	if unknown:
		msg = f'per_survey_token_indices contains unknown survey ids: {unknown!r}'
		raise ValueError(msg)

	blocks: list[np.ndarray] = []
	for item in embedding_inputs:
		indices = np.asarray(
			per_survey_token_indices.get(item.survey_id, np.empty(0, dtype=np.int64)),
			dtype=np.int64,
		)
		if indices.ndim != 1:
			msg = (
				'per_survey_token_indices values must be 1D; '
				f'{item.survey_id} has shape {indices.shape!r}'
			)
			raise ValueError(msg)
		if indices.size == 0:
			continue
		token_shape_xyz = open_embedding_array(item).shape[:3]
		coords = np.unravel_index(indices, token_shape_xyz)
		blocks.append(np.asarray(coords[2], dtype=np.int32))
	if not blocks:
		return np.empty(0, dtype=np.int32)
	return np.concatenate(blocks).astype(np.int32, copy=False)


def normalized_z_features_for_indices(
	embedding_input: EmbeddingInput,
	flat_indices: np.ndarray,
) -> np.ndarray:
	"""Return normalized z-coordinate features for flattened token indices."""
	indices = np.asarray(flat_indices, dtype=np.int64)
	if indices.ndim != 1:
		msg = f'flat_indices must be 1D; got shape {indices.shape!r}'
		raise ValueError(msg)
	if indices.size == 0:
		return np.empty((0, 1), dtype=np.float32)
	token_shape_xyz = open_embedding_array(embedding_input).shape[:3]
	z_size = int(token_shape_xyz[2])
	coords = np.unravel_index(indices, token_shape_xyz)
	z_values = np.asarray(coords[2], dtype=np.float32)
	denominator = np.float32(max(z_size - 1, 1))
	return (z_values / denominator).reshape(-1, 1).astype(np.float32, copy=False)


def initialize_ordered_centers(
	training_features: np.ndarray,
	sample_z: np.ndarray,
	*,
	k: int,
	batch_size: int,
	seed: int,
) -> np.ndarray:
	"""Initialize cluster centers ordered from shallow to deep mean sample z."""
	matrix = np.asarray(training_features, dtype=np.float32)
	z_coordinates = np.asarray(sample_z, dtype=np.int32)
	if matrix.ndim != 2:
		msg = f'training_features must be 2D; got shape {matrix.shape!r}'
		raise ValueError(msg)
	if z_coordinates.shape != (matrix.shape[0],):
		msg = (
			'sample_z must have one coordinate per training row; '
			f'got {z_coordinates.shape!r} for {matrix.shape[0]} rows'
		)
		raise ValueError(msg)

	kmeans = fit_minibatch_kmeans(
		matrix,
		k=k,
		batch_size=batch_size,
		seed=seed,
	)
	labels = kmeans.predict(matrix)
	raw_centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
	mean_z = np.full(k, np.inf, dtype=np.float64)
	is_empty = np.ones(k, dtype=np.bool_)
	for label in range(k):
		mask = labels == label
		if np.any(mask):
			mean_z[label] = float(np.mean(z_coordinates[mask], dtype=np.float64))
			is_empty[label] = False
	order = sorted(
		range(k),
		key=lambda label: (bool(is_empty[label]), mean_z[label], label),
	)
	return raw_centers[np.asarray(order, dtype=np.int64)].astype(np.float32, copy=False)


def prepare_feature_batch_for_indices(
	embedding_input: EmbeddingInput,
	flat_indices: np.ndarray,
	*,
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	emission_source: str = 'embedding',
) -> np.ndarray:
	"""Load, residualize, and preprocess one arbitrary batch of token features."""
	indices = np.asarray(flat_indices, dtype=np.int64)
	if indices.ndim != 1:
		msg = f'flat_indices must be 1D; got shape {indices.shape!r}'
		raise ValueError(msg)
	if indices.size == 0:
		return np.empty(
			(
				0,
				_transformed_feature_dim(
					embedding_input, preprocessor, emission_source
				),
			),
			dtype=np.float32,
		)

	if emission_source == 'embedding':
		features = extract_token_features(embedding_input, indices)
	elif emission_source == 'z_coordinate':
		features = normalized_z_features_for_indices(embedding_input, indices)
	else:
		msg = (
			"emission_source must be 'embedding' or 'z_coordinate'; "
			f'got {emission_source!r}'
		)
		raise ValueError(msg)
	if residualizer is not None and emission_source == 'embedding':
		group_keys = residualization_keys_for_flat_indices(
			embedding_input,
			indices,
			group_by=residualizer.group_by,
		)
		features = residualizer.transform(features, group_keys)
	transformed = np.asarray(preprocessor.transform(features), dtype=np.float32)
	if transformed.ndim != 2:
		msg = f'preprocessor output must be 2D; got shape {transformed.shape!r}'
		raise ValueError(msg)
	if not np.all(np.isfinite(transformed)):
		msg = (
			'preprocessed features must contain only finite values for '
			f'{embedding_input.survey_id}'
		)
		raise ValueError(msg)
	return transformed


def viterbi_decode_costs(
	emission_costs: np.ndarray,
	transition_costs: np.ndarray,
	*,
	initial_state_costs: np.ndarray | None = None,
	terminal_state_costs: np.ndarray | None = None,
) -> np.ndarray:
	"""Decode the minimum-cost state path with deterministic tie-breaking."""
	emissions = _as_float_matrix(emission_costs, 'emission_costs')
	transitions = _as_float_matrix(transition_costs, 'transition_costs')
	if emissions.shape[0] == 0 or emissions.shape[1] == 0:
		raise ValueError('emission_costs must be non-empty in both dimensions')
	if not np.all(np.isfinite(emissions)):
		raise ValueError('emission_costs must contain only finite values')

	k = emissions.shape[1]
	if transitions.shape != (k, k):
		raise ValueError(
			f'transition_costs must have shape ({k}, {k}); got {transitions.shape}'
		)
	if np.isnan(transitions).any():
		raise ValueError('transition_costs must not contain NaN values')
	initial_costs = _optional_state_costs(
		initial_state_costs,
		k,
		'initial_state_costs',
	)
	terminal_costs = _optional_state_costs(
		terminal_state_costs,
		k,
		'terminal_state_costs',
	)

	t_count = emissions.shape[0]
	dp = np.empty((t_count, k), dtype=np.float64)
	backpointers = np.zeros((t_count, k), dtype=np.int32)
	dp[0] = emissions[0] + initial_costs

	for t_index in range(1, t_count):
		candidates = dp[t_index - 1, :, np.newaxis] + transitions
		previous = np.argmin(candidates, axis=0)
		best_transition_costs = candidates[previous, np.arange(k)]
		dp[t_index] = best_transition_costs + emissions[t_index]
		backpointers[t_index] = previous.astype(np.int32)

	final_costs = dp[-1] + terminal_costs
	final_state = int(np.argmin(final_costs))
	if not np.isfinite(final_costs[final_state]):
		raise ValueError(
			'no finite path exists for emission_costs and transition_costs',
		)

	path = np.empty(t_count, dtype=np.int32)
	path[-1] = final_state
	for t_index in range(t_count - 1, 0, -1):
		path[t_index - 1] = backpointers[t_index, path[t_index]]
	return path


def decode_trace_segments(
	emission_costs: np.ndarray,
	valid_mask: np.ndarray,
	transition_costs: np.ndarray,
	*,
	initial_state_costs: np.ndarray | None = None,
	terminal_state_costs: np.ndarray | None = None,
) -> np.ndarray:
	"""Decode valid vertical trace tokens as one sequence, preserving invalid gaps."""
	emissions = _as_float_matrix(emission_costs, 'emission_costs')
	if emissions.shape[0] == 0 or emissions.shape[1] == 0:
		raise ValueError('emission_costs must be non-empty in both dimensions')
	if not np.all(np.isfinite(emissions)):
		raise ValueError('emission_costs must contain only finite values')

	mask = np.asarray(valid_mask)
	if mask.ndim != 1:
		raise ValueError(f'valid_mask must be 1D; got shape {mask.shape}')
	if mask.dtype != np.bool_:
		raise TypeError('valid_mask must have boolean dtype')
	if mask.shape != (emissions.shape[0],):
		raise ValueError(
			f'valid_mask must have shape ({emissions.shape[0]},); got {mask.shape}'
		)

	transitions = _as_float_matrix(transition_costs, 'transition_costs')
	k = emissions.shape[1]
	if transitions.shape != (k, k):
		raise ValueError(
			f'transition_costs must have shape ({k}, {k}); got {transitions.shape}'
		)
	if np.isnan(transitions).any():
		raise ValueError('transition_costs must not contain NaN values')
	initial_costs = _optional_state_costs(
		initial_state_costs,
		k,
		'initial_state_costs',
	)
	terminal_costs = _optional_state_costs(
		terminal_state_costs,
		k,
		'terminal_state_costs',
	)

	labels = np.full(emissions.shape[0], -1, dtype=np.int32)
	z_indices = np.flatnonzero(mask)
	if z_indices.size:
		labels[z_indices] = viterbi_decode_costs(
			emissions[z_indices],
			transitions,
			initial_state_costs=initial_costs,
			terminal_state_costs=terminal_costs,
		)
	return labels


def decode_survey_ordered_labels(  # noqa: PLR0913
	embedding_input: EmbeddingInput,
	*,
	centers: np.ndarray,
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	transition_costs: np.ndarray,
	initial_state_costs: np.ndarray | None = None,
	terminal_state_costs: np.ndarray | None = None,
	emission_source: str = 'embedding',
	edge_margin_tokens: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
	"""Decode HMM labels for one survey without flattening all features at once."""
	center_matrix = np.asarray(centers, dtype=np.float32)
	if center_matrix.ndim != 2 or center_matrix.shape[0] == 0:
		msg = f'centers must be a non-empty 2D matrix; got {center_matrix.shape!r}'
		raise ValueError(msg)
	embeddings = open_embedding_array(embedding_input)
	valid = hmm_valid_token_mask(embedding_input, edge_margin_tokens)
	shape = embeddings.shape[:3]
	labels = np.full(shape, -1, dtype=np.int32)
	x_count, y_count, z_count = shape
	k = center_matrix.shape[0]
	for x_index in range(x_count):
		for y_index in range(y_count):
			trace_valid = np.asarray(valid[x_index, y_index, :], dtype=np.bool_)
			if not np.any(trace_valid):
				continue
			z_indices = np.flatnonzero(trace_valid)
			flat_indices = np.ravel_multi_index(
				(
					np.full(z_indices.shape, x_index, dtype=np.int64),
					np.full(z_indices.shape, y_index, dtype=np.int64),
					z_indices,
				),
				shape,
			).astype(np.int64, copy=False)
			prepared = prepare_feature_batch_for_indices(
				embedding_input,
				flat_indices,
				residualizer=residualizer,
				preprocessor=preprocessor,
				emission_source=emission_source,
			)
			if prepared.shape[1] != center_matrix.shape[1]:
				msg = (
					'prepared feature dimension must match centers; got '
					f'{prepared.shape[1]} and {center_matrix.shape[1]}'
				)
				raise ValueError(msg)
			emission_costs = np.zeros((z_count, k), dtype=np.float32)
			deltas = prepared[:, np.newaxis, :] - center_matrix[np.newaxis, :, :]
			emission_costs[z_indices] = np.sum(deltas * deltas, axis=2)
			labels[x_index, y_index, :] = decode_trace_segments(
				emission_costs,
				trace_valid,
				transition_costs,
				initial_state_costs=initial_state_costs,
				terminal_state_costs=terminal_state_costs,
			)
	return labels


def _decode_all_surveys(  # noqa: PLR0913
	embedding_inputs: tuple[EmbeddingInput, ...],
	*,
	centers: np.ndarray,
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	transition_costs: np.ndarray,
	initial_state_costs: np.ndarray | None,
	terminal_state_costs: np.ndarray | None,
	emission_source: str,
	edge_margin_tokens: tuple[int, int, int],
) -> dict[str, np.ndarray]:
	return {
		item.survey_id: decode_survey_ordered_labels(
			item,
			centers=centers,
			residualizer=residualizer,
			preprocessor=preprocessor,
			transition_costs=transition_costs,
			initial_state_costs=initial_state_costs,
			terminal_state_costs=terminal_state_costs,
			emission_source=emission_source,
			edge_margin_tokens=edge_margin_tokens,
		)
		for item in embedding_inputs
	}


def update_centers_from_labels(  # noqa: C901, PLR0913
	embedding_inputs: tuple[EmbeddingInput, ...],
	labels_by_survey: Mapping[str, np.ndarray],
	*,
	centers: np.ndarray,
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	prediction_batch_size: int,
	empty_cluster_policy: str,
	emission_source: str = 'embedding',
) -> tuple[np.ndarray, dict[str, object]]:
	"""Update centers as count-weighted means in transformed feature space."""
	if prediction_batch_size <= 0:
		msg = f'prediction_batch_size must be positive; got {prediction_batch_size!r}'
		raise ValueError(msg)
	if empty_cluster_policy != 'keep_previous':
		msg = (
			"empty_cluster_policy must be 'keep_previous'; "
			f'got {empty_cluster_policy!r}'
		)
		raise ValueError(msg)
	center_matrix = np.asarray(centers, dtype=np.float32)
	if center_matrix.ndim != 2 or center_matrix.shape[0] == 0:
		msg = f'centers must be a non-empty 2D matrix; got {center_matrix.shape!r}'
		raise ValueError(msg)
	k, feature_dim = center_matrix.shape
	sums = np.zeros((k, feature_dim), dtype=np.float64)
	counts = np.zeros(k, dtype=np.int64)
	inputs_by_survey = {item.survey_id: item for item in embedding_inputs}
	unknown = sorted(set(labels_by_survey) - set(inputs_by_survey))
	if unknown:
		msg = f'labels_by_survey contains unknown survey ids: {unknown!r}'
		raise ValueError(msg)

	for item in embedding_inputs:
		if item.survey_id not in labels_by_survey:
			msg = f'labels_by_survey missing survey id: {item.survey_id!r}'
			raise ValueError(msg)
		embeddings = open_embedding_array(item)
		labels = np.asarray(labels_by_survey[item.survey_id])
		if labels.shape != embeddings.shape[:3]:
			msg = (
				'label grid shape for '
				f'{item.survey_id} must be {embeddings.shape[:3]!r}; '
				f'got {labels.shape!r}'
			)
			raise ValueError(msg)
		flat_labels = labels.reshape(-1)
		labeled_indices = np.flatnonzero(flat_labels >= 0).astype(np.int64, copy=False)
		for start in range(0, labeled_indices.size, prediction_batch_size):
			batch_indices = labeled_indices[start : start + prediction_batch_size]
			batch_labels = np.asarray(flat_labels[batch_indices], dtype=np.int64)
			if np.any(batch_labels >= k):
				msg = f'label id out of range for {item.survey_id}'
				raise ValueError(msg)
			prepared = prepare_feature_batch_for_indices(
				item,
				batch_indices,
				residualizer=residualizer,
				preprocessor=preprocessor,
				emission_source=emission_source,
			)
			if prepared.shape[1] != feature_dim:
				msg = (
					'prepared feature dimension must match centers; got '
					f'{prepared.shape[1]} and {feature_dim}'
				)
				raise ValueError(msg)
			np.add.at(sums, batch_labels, prepared.astype(np.float64, copy=False))
			counts += np.bincount(batch_labels, minlength=k)

	new_centers = center_matrix.copy()
	non_empty = counts > 0
	new_centers[non_empty] = (sums[non_empty] / counts[non_empty, np.newaxis]).astype(
		np.float32
	)
	shifts = np.linalg.norm(new_centers - center_matrix, axis=1)
	empty_clusters = [int(label) for label in np.flatnonzero(~non_empty)]
	summary = {
		'cluster_counts': {
			int(label): int(count) for label, count in enumerate(counts)
		},
		'empty_clusters': empty_clusters,
		'center_shift_l2': [float(value) for value in shifts],
		'total_center_shift_l2': float(np.linalg.norm(new_centers - center_matrix)),
	}
	return new_centers.astype(np.float32, copy=False), summary


def run_stratigraphic_hmm_clustering(
	config: Mapping[str, object],
) -> ClusteringRunResult:
	"""Run stratigraphic HMM clustering from a validated config mapping."""
	settings = clustering_settings_from_config(config)
	hmm_settings = stratigraphic_hmm_settings_from_config(config)
	embedding_inputs = tuple(discover_embedding_inputs(settings.input_dir))
	compatibility_signature = validate_compatible_embedding_inputs(embedding_inputs)
	edge_margin_excluded_valid_token_count = (
		_edge_margin_excluded_valid_token_count(
			embedding_inputs,
			hmm_settings.edge_margin_tokens,
		)
	)
	if hmm_settings.emission_source == 'embedding':
		sample = _sample_valid_hmm_embedding_tokens(
			embedding_inputs,
			sample_tokens=settings.sample_tokens,
			seed=settings.seed,
			edge_margin_tokens=hmm_settings.edge_margin_tokens,
		)
		residualizer = fit_residualizer(
			sample.features,
			embedding_inputs=embedding_inputs,
			per_survey_token_indices=sample.per_survey_token_indices,
			settings=settings.residualization,
		)
		training_input_features = apply_residualizer_to_sample(
			sample.features,
			embedding_inputs=embedding_inputs,
			per_survey_token_indices=sample.per_survey_token_indices,
			residualizer=residualizer,
		)
		metadata_settings = settings
	elif hmm_settings.emission_source == 'z_coordinate':
		sample = _sample_valid_z_tokens(
			embedding_inputs,
			sample_tokens=settings.sample_tokens,
			seed=settings.seed,
			edge_margin_tokens=hmm_settings.edge_margin_tokens,
		)
		residualizer = None
		training_input_features = np.asarray(sample.features, dtype=np.float32)
		metadata_settings = replace(
			settings,
			embedding_normalization='none',
			residualization=ResidualizationSettings(
				enabled=False,
				mode='local_token_position',
				group_by='token_phase',
				add_global_mean_back=True,
				min_group_count=32,
			),
			pca=PCASettings(enabled=False, n_components=1, whiten=False),
		)
	else:
		msg = (
			"clustering.stratigraphic_hmm.emission_source must be 'embedding' "
			f"or 'z_coordinate'; got {hmm_settings.emission_source!r}"
		)
		raise ValueError(msg)
	preprocessor = fit_preprocessor(
		training_input_features,
		normalization=metadata_settings.embedding_normalization,
		pca=metadata_settings.pca,
		seed=settings.seed,
	)
	training_features = np.asarray(
		preprocessor.transform(training_input_features),
		dtype=np.float32,
	)
	sample_z = sample_token_z_coordinates(
		embedding_inputs,
		sample.per_survey_token_indices,
	)
	residualizer_path: Path | None = None
	if residualizer is not None:
		residualizer_path = settings.output_dir / 'models' / 'residualizer.npz'
		write_residualizer_npz(residualizer_path, residualizer)
	common_metadata = _common_metadata(
		settings=metadata_settings,
		embedding_inputs=embedding_inputs,
		compatibility_signature=compatibility_signature,
		sample=sample,
		preprocessor=preprocessor,
		residualizer=residualizer,
		residualizer_path=residualizer_path,
	)
	common_metadata = {
		**common_metadata,
		'emission_source': hmm_settings.emission_source,
		'emission_features': _emission_feature_metadata(hmm_settings.emission_source),
	}

	results: list[KClusteringResult] = []
	for k in settings.k_values:
		transition_costs = build_ordered_transition_costs(k, hmm_settings.transition)
		initial_state_costs = build_initial_state_costs(k, hmm_settings.path_prior)
		terminal_state_costs = build_terminal_state_costs(k, hmm_settings.path_prior)
		centers = initialize_ordered_centers(
			training_features,
			sample_z,
			k=k,
			batch_size=settings.minibatch_size,
			seed=settings.seed,
		)
		iteration_summaries: list[dict[str, object]] = []
		labels_by_survey: dict[str, np.ndarray] = {}
		for iteration in range(1, hmm_settings.iterations + 1):
			labels_by_survey = _decode_all_surveys(
				embedding_inputs,
				centers=centers,
				residualizer=residualizer,
				preprocessor=preprocessor,
				transition_costs=transition_costs,
				initial_state_costs=initial_state_costs,
				terminal_state_costs=terminal_state_costs,
				emission_source=hmm_settings.emission_source,
				edge_margin_tokens=hmm_settings.edge_margin_tokens,
			)
			centers, summary = update_centers_from_labels(
				embedding_inputs,
				labels_by_survey,
				centers=centers,
				residualizer=residualizer,
				preprocessor=preprocessor,
				prediction_batch_size=settings.prediction_batch_size,
				empty_cluster_policy=hmm_settings.empty_cluster_policy,
				emission_source=hmm_settings.emission_source,
			)
			iteration_summaries.append({'iteration': iteration, **summary})

		labels_by_survey = _decode_all_surveys(
			embedding_inputs,
			centers=centers,
			residualizer=residualizer,
			preprocessor=preprocessor,
			transition_costs=transition_costs,
			initial_state_costs=initial_state_costs,
			terminal_state_costs=terminal_state_costs,
			emission_source=hmm_settings.emission_source,
			edge_margin_tokens=hmm_settings.edge_margin_tokens,
		)

		label_results = _write_hmm_labels_for_k(
			output_dir=settings.output_dir,
			k=k,
			embedding_inputs=embedding_inputs,
			labels_by_survey=labels_by_survey,
			label_metadata=common_metadata,
		)
		cluster_counts = _aggregate_counts(label_results, k)
		invalid_token_count = int(
			sum(result.invalid_token_count for result in label_results),
		)
		per_survey_ordered_diagnostics = {
			item.survey_id: ordered_label_diagnostics(
				np.asarray(labels_by_survey[item.survey_id]),
				k=k,
				z_axis=hmm_settings.z_axis,
			)
			for item in embedding_inputs
		}
		hmm_metadata = _hmm_metadata(
			hmm_settings=hmm_settings,
			transition_costs=transition_costs,
			initial_state_costs=initial_state_costs,
			terminal_state_costs=terminal_state_costs,
			iteration_summaries=iteration_summaries,
			edge_margin_excluded_valid_token_count=(
				edge_margin_excluded_valid_token_count
			),
		)
		metadata = {
			**common_metadata,
			'k': int(k),
			'cluster_counts': cluster_counts,
			'invalid_token_count': invalid_token_count,
			'stratigraphic_hmm': hmm_metadata,
			'ordered_diagnostics': {
				'per_survey': per_survey_ordered_diagnostics,
				'aggregate': aggregate_ordered_label_diagnostics(
					per_survey_ordered_diagnostics,
					k=k,
				),
			},
			'surveys': [
				{
					'survey_id': result.survey_id,
					'label_path': str(result.labels_path),
					'label_metadata_path': str(result.metadata_path),
					'valid_token_count': result.valid_token_count,
					'invalid_token_count': result.invalid_token_count,
					'cluster_counts': result.cluster_counts,
				}
				for result in label_results
			],
		}
		_write_hmm_model_artifacts(
			output_dir=settings.output_dir,
			k=k,
			preprocessor=preprocessor,
			centers=centers,
			hmm_model={
				'method': 'stratigraphic_hmm_kmeans',
				'emission_source': hmm_settings.emission_source,
				'centers': centers,
				'transition_settings': asdict(hmm_settings.transition),
				'edge_margin_tokens': hmm_settings.edge_margin_tokens,
				'path_prior': asdict(hmm_settings.path_prior),
				'transition_costs': transition_costs,
				'initial_state_costs': initial_state_costs,
				'terminal_state_costs': terminal_state_costs,
				'iteration_count': hmm_settings.iterations,
				'iteration_summaries': iteration_summaries,
			},
			metadata=metadata,
		)
		results.append(
			KClusteringResult(
				k=k,
				model_dir=settings.output_dir / 'models' / f'k{k}',
				label_results=tuple(label_results),
				cluster_counts=cluster_counts,
				invalid_token_count=invalid_token_count,
			),
		)

	return ClusteringRunResult(
		embedding_inputs=embedding_inputs,
		sample=sample,
		results=tuple(results),
	)


def _sample_valid_hmm_embedding_tokens(
	embedding_inputs: tuple[EmbeddingInput, ...],
	*,
	sample_tokens: int,
	seed: int,
	edge_margin_tokens: tuple[int, int, int],
) -> SampledTokens:
	"""Sample HMM-valid token embeddings after edge-margin exclusion."""
	return _sample_valid_hmm_tokens(
		embedding_inputs,
		sample_tokens=sample_tokens,
		seed=seed,
		edge_margin_tokens=edge_margin_tokens,
		feature_loader=extract_token_features,
	)


def _sample_valid_z_tokens(
	embedding_inputs: tuple[EmbeddingInput, ...],
	*,
	sample_tokens: int,
	seed: int,
	edge_margin_tokens: tuple[int, int, int],
) -> SampledTokens:
	"""Sample valid token indices and use normalized z as training features."""
	return _sample_valid_hmm_tokens(
		embedding_inputs,
		sample_tokens=sample_tokens,
		seed=seed,
		edge_margin_tokens=edge_margin_tokens,
		feature_loader=normalized_z_features_for_indices,
	)


def _sample_valid_hmm_tokens(
	embedding_inputs: tuple[EmbeddingInput, ...],
	*,
	sample_tokens: int,
	seed: int,
	edge_margin_tokens: tuple[int, int, int],
	feature_loader: Callable[[EmbeddingInput, np.ndarray], np.ndarray],
) -> SampledTokens:
	"""Sample HMM-valid token indices and load matching training features."""
	if sample_tokens <= 0:
		msg = f'sample_tokens must be positive; got {sample_tokens!r}'
		raise ValueError(msg)
	if not embedding_inputs:
		msg = 'at least one embedding input is required'
		raise ValueError(msg)

	valid_indices_by_survey = {
		item.survey_id: hmm_valid_flat_indices(item, edge_margin_tokens)
		for item in embedding_inputs
	}
	valid_counts = [
		int(valid_indices_by_survey[item.survey_id].size)
		for item in embedding_inputs
	]
	total_valid = int(sum(valid_counts))
	if total_valid == 0:
		msg = 'cannot cluster embeddings because no valid tokens were found'
		raise ValueError(msg)

	sample_count = min(int(sample_tokens), total_valid)
	rng = np.random.default_rng(seed)
	selected_global = np.sort(
		rng.choice(total_valid, size=sample_count, replace=False),
	)
	per_survey_indices: dict[str, np.ndarray] = {}
	feature_blocks: list[np.ndarray] = []
	offset = 0
	for item, count in zip(embedding_inputs, valid_counts, strict=True):
		stop = offset + count
		mask = (selected_global >= offset) & (selected_global < stop)
		local_valid_ordinals = selected_global[mask] - offset
		if local_valid_ordinals.size:
			all_valid_indices = valid_indices_by_survey[item.survey_id]
			token_indices = all_valid_indices[local_valid_ordinals]
			per_survey_indices[item.survey_id] = token_indices
			feature_blocks.append(feature_loader(item, token_indices))
		else:
			per_survey_indices[item.survey_id] = np.empty(0, dtype=np.int64)
		offset = stop

	features = np.concatenate(feature_blocks, axis=0)
	return SampledTokens(
		features=np.asarray(features, dtype=np.float32),
		per_survey_token_indices=per_survey_indices,
		requested_count=int(sample_tokens),
		total_valid_count=total_valid,
		sample_count=sample_count,
	)


def _write_hmm_labels_for_k(
	*,
	output_dir: str | Path,
	k: int,
	embedding_inputs: tuple[EmbeddingInput, ...],
	labels_by_survey: Mapping[str, np.ndarray],
	label_metadata: Mapping[str, object],
) -> list[SurveyLabelResult]:
	return [
		_write_hmm_survey_labels(
			output_dir=output_dir,
			k=k,
			embedding_input=item,
			labels=np.asarray(labels_by_survey[item.survey_id]),
			label_metadata=label_metadata,
		)
		for item in embedding_inputs
	]


def _write_hmm_survey_labels(
	*,
	output_dir: str | Path,
	k: int,
	embedding_input: EmbeddingInput,
	labels: np.ndarray,
	label_metadata: Mapping[str, object],
) -> SurveyLabelResult:
	embeddings = open_embedding_array(embedding_input)
	if labels.shape != embeddings.shape[:3]:
		msg = (
			'label grid shape for '
			f'{embedding_input.survey_id} must be {embeddings.shape[:3]!r}; '
			f'got {labels.shape!r}'
		)
		raise ValueError(msg)
	labels_dir = Path(output_dir) / 'labels' / f'k{k}'
	labels_dir.mkdir(parents=True, exist_ok=True)
	labels_path = labels_dir / f'{embedding_input.survey_id}.cluster_labels_token.npy'
	np.save(labels_path, np.asarray(labels, dtype=np.int32))
	counts = np.bincount(labels[labels >= 0].astype(np.int64), minlength=k)
	cluster_counts = {int(label): int(count) for label, count in enumerate(counts[:k])}
	valid = int(np.count_nonzero(labels >= 0))
	invalid = int(labels.size - valid)
	ordered_diagnostics = ordered_label_diagnostics(labels, k=k)
	metadata_path = (
		labels_dir / f'{embedding_input.survey_id}.cluster_label_metadata.json'
	)
	metadata = {
		**dict(label_metadata),
		'method': 'stratigraphic_hmm_kmeans',
		'k': int(k),
		'survey_id': embedding_input.survey_id,
		'embedding_input': embedding_input_metadata(embedding_input),
		'label_path': str(labels_path),
		'token_grid_shape': list(embeddings.shape[:3]),
		'embedding_dim': int(embeddings.shape[-1]),
		'valid_token_count': valid,
		'invalid_token_count': invalid,
		'cluster_counts': cluster_counts,
		'ordered_diagnostics': ordered_diagnostics,
		'ordered_boundary_summary': ordered_boundary_summary(labels, k=k),
	}
	write_json(metadata_path, metadata)
	return SurveyLabelResult(
		survey_id=embedding_input.survey_id,
		labels_path=labels_path,
		metadata_path=metadata_path,
		cluster_counts=cluster_counts,
		invalid_token_count=invalid,
		valid_token_count=valid,
	)


def _write_hmm_model_artifacts(  # noqa: PLR0913
	*,
	output_dir: str | Path,
	k: int,
	preprocessor: object,
	centers: np.ndarray,
	hmm_model: Mapping[str, object],
	metadata: Mapping[str, object],
) -> None:
	model_dir = Path(output_dir) / 'models' / f'k{k}'
	model_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(preprocessor, model_dir / 'preprocessor.joblib')
	joblib.dump(dict(hmm_model), model_dir / 'hmm_model.joblib')
	np.save(model_dir / 'cluster_centers.npy', np.asarray(centers, dtype=np.float32))
	write_json(model_dir / 'clustering_metadata.json', metadata)


def _hmm_metadata(  # noqa: PLR0913
	*,
	hmm_settings: StratigraphicHMMSettings,
	transition_costs: np.ndarray,
	initial_state_costs: np.ndarray,
	terminal_state_costs: np.ndarray,
	iteration_summaries: list[dict[str, object]],
	edge_margin_excluded_valid_token_count: int,
) -> dict[str, object]:
	path_prior = {
		**asdict(hmm_settings.path_prior),
		'initial_state_costs': _json_safe_state_costs(
			initial_state_costs,
			'initial_state_costs',
		),
		'terminal_state_costs': _json_safe_state_costs(
			terminal_state_costs,
			'terminal_state_costs',
		),
	}
	return {
		'emission_source': hmm_settings.emission_source,
		'iterations': hmm_settings.iterations,
		'z_axis': hmm_settings.z_axis,
		'z_direction': hmm_settings.z_direction,
		'transition': asdict(hmm_settings.transition),
		'transition_costs': _json_safe_transition_costs(transition_costs),
		'init': {'order_by': hmm_settings.init_order_by},
		'update': {'empty_cluster_policy': hmm_settings.empty_cluster_policy},
		'edge_margin_tokens': list(hmm_settings.edge_margin_tokens),
		'edge_margin_excluded_valid_token_count': int(
			edge_margin_excluded_valid_token_count
		),
		'path_prior': path_prior,
		'iteration_summaries': iteration_summaries,
	}


def _json_safe_transition_costs(costs: np.ndarray) -> list[list[float | None]]:
	matrix = np.asarray(costs, dtype=np.float32)
	if matrix.ndim != 2:
		msg = f'transition_costs must be 2D; got shape {matrix.shape!r}'
		raise ValueError(msg)
	return [
		[None if not np.isfinite(value) else float(value) for value in row]
		for row in matrix
	]


def _json_safe_state_costs(costs: np.ndarray, name: str) -> list[float]:
	vector = np.asarray(costs, dtype=np.float64)
	_validate_state_costs(vector, vector.size, name)
	return [float(value) for value in vector]


def _emission_feature_metadata(emission_source: str) -> dict[str, object]:
	if emission_source == 'z_coordinate':
		return {
			'source': 'z_coordinate',
			'feature_dim': 1,
			'normalization': 'z / max(z_size - 1, 1)',
			'embedding_features_used_for_emissions': False,
			'embedding_artifacts_used_for': ['token_grid_shape', 'validity_masks'],
		}
	return {
		'source': 'embedding',
		'embedding_features_used_for_emissions': True,
		'embedding_artifacts_used_for': [
			'token_grid_shape',
			'validity_masks',
			'embedding_features',
		],
	}


def _edge_margin_excluded_valid_token_count(
	embedding_inputs: tuple[EmbeddingInput, ...],
	edge_margin_tokens: tuple[int, int, int],
) -> int:
	count = 0
	for item in embedding_inputs:
		valid = np.asarray(load_valid_tokens(item), dtype=np.bool_)
		_validate_edge_margin_shape(
			valid.shape,
			edge_margin_tokens,
			survey_id=item.survey_id,
		)
		margin_mask = _edge_margin_mask_for_validated_shape(
			valid.shape,
			edge_margin_tokens,
		)
		count += int(np.count_nonzero(valid & ~margin_mask))
	return count


def _validate_edge_margin_shape(
	shape: tuple[int, int, int],
	edge_margin_tokens: tuple[int, int, int],
	*,
	survey_id: str | None,
) -> None:
	if len(shape) != 3:
		msg = f'token grid shape must have three axes; got {shape!r}'
		raise ValueError(msg)
	if len(edge_margin_tokens) != 3:
		msg = (
			'edge_margin_tokens must contain exactly three integers; '
			f'got {edge_margin_tokens!r}'
		)
		raise ValueError(msg)
	for value in edge_margin_tokens:
		if value < 0:
			msg = f'edge_margin_tokens must be non-negative; got {edge_margin_tokens!r}'
			raise ValueError(msg)
	if any(
		(2 * margin) >= size
		for size, margin in zip(shape, edge_margin_tokens, strict=True)
	):
		margin_text = '[' + ','.join(str(value) for value in edge_margin_tokens) + ']'
		if survey_id is None:
			msg = (
				f'edge_margin_tokens {margin_text} leave no interior tokens '
				f'for token grid shape {shape}'
			)
		else:
			msg = (
				f'edge_margin_tokens {margin_text} leave no interior tokens '
				f'for survey {survey_id} with token grid shape {shape}'
			)
		raise ValueError(msg)


def _validate_transition_settings(settings: HMMTransitionSettings) -> None:
	for field in ('same_cost', 'advance_cost', 'jump_cost', 'reverse_cost'):
		_validate_nonnegative_finite_cost(getattr(settings, field), field)
	if not isinstance(settings.forbid_reverse, bool):
		raise TypeError('forbid_reverse must be bool')
	if settings.max_jump is not None:
		_validate_positive_int(settings.max_jump, 'max_jump')


def _validate_nonnegative_finite_cost(value: object, name: str) -> None:
	if isinstance(value, bool) or not isinstance(value, Real):
		raise TypeError(f'{name} must be a finite non-negative number')
	if not np.isfinite(float(value)) or float(value) < 0.0:
		raise ValueError(f'{name} must be a finite non-negative number')


def _validate_positive_int(value: object, name: str) -> None:
	if isinstance(value, bool) or not isinstance(value, Integral):
		raise TypeError(f'{name} must be a positive integer')
	if int(value) <= 0:
		raise ValueError(f'{name} must be a positive integer')


def _as_float_matrix(value: np.ndarray, name: str) -> np.ndarray:
	array = np.asarray(value, dtype=np.float64)
	if array.ndim != 2:
		raise ValueError(f'{name} must be 2D; got shape {array.shape}')
	return array


def _optional_state_costs(
	value: np.ndarray | None,
	k: int,
	name: str,
) -> np.ndarray:
	if value is None:
		return np.zeros(k, dtype=np.float64)
	return _validate_state_costs(np.asarray(value, dtype=np.float64), k, name)


def _validate_state_costs(value: np.ndarray, k: int, name: str) -> np.ndarray:
	if value.ndim != 1:
		raise ValueError(f'{name} must be 1D; got shape {value.shape}')
	if value.shape != (k,):
		raise ValueError(f'{name} must have shape ({k},); got {value.shape}')
	if not np.all(np.isfinite(value)):
		raise ValueError(f'{name} must contain only finite values')
	if np.any(value < 0.0):
		raise ValueError(f'{name} must contain only non-negative values')
	return value


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _transformed_feature_dim(
	embedding_input: EmbeddingInput,
	preprocessor: object,
	emission_source: str,
) -> int:
	if emission_source == 'z_coordinate':
		return 1
	if hasattr(preprocessor, 'named_steps'):
		pipeline = preprocessor
		named_steps = pipeline.named_steps
		pca = named_steps.get('pca')
		if pca is not None:
			return int(getattr(pca, 'n_components_', pca.n_components))
	return int(open_embedding_array(embedding_input).shape[-1])


__all__ = [
	'HMMAnchorPriorSettings',
	'HMMExpectedBoundariesSettings',
	'HMMPathPriorSettings',
	'HMMTransitionSettings',
	'StratigraphicHMMSettings',
	'build_initial_state_costs',
	'build_ordered_transition_costs',
	'build_terminal_state_costs',
	'decode_survey_ordered_labels',
	'decode_trace_segments',
	'initialize_ordered_centers',
	'normalized_z_features_for_indices',
	'prepare_feature_batch_for_indices',
	'run_stratigraphic_hmm_clustering',
	'sample_token_z_coordinates',
	'stratigraphic_hmm_settings_from_config',
	'update_centers_from_labels',
	'viterbi_decode_costs',
]
