"""Stratigraphic HMM clustering backend scaffold."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import NoReturn

import numpy as np

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	extract_token_features,
	open_embedding_array,
)
from seis_ssl_cluster.clustering.kmeans import fit_minibatch_kmeans
from seis_ssl_cluster.clustering.residualization import (
	LocalTokenPositionResidualizer,
	residualization_keys_for_flat_indices,
)


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
class StratigraphicHMMSettings:
	"""Validated stratigraphic HMM backend settings."""

	iterations: int
	z_axis: int
	z_direction: str
	transition: HMMTransitionSettings
	init_order_by: str
	empty_cluster_policy: str


def stratigraphic_hmm_settings_from_config(
	config: Mapping[str, object],
) -> StratigraphicHMMSettings:
	"""Build stratigraphic HMM settings from a resolved config mapping."""
	clustering = _required_mapping(config, 'clustering')
	hmm = _required_mapping(clustering, 'stratigraphic_hmm')
	transition = _required_mapping(hmm, 'transition')
	init = _required_mapping(hmm, 'init')
	update = _required_mapping(hmm, 'update')
	return StratigraphicHMMSettings(
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
				None
				if transition['max_jump'] is None
				else int(transition['max_jump'])
			),
		),
		init_order_by=str(init['order_by']),
		empty_cluster_policy=str(update['empty_cluster_policy']),
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
) -> np.ndarray:
	"""Load, residualize, and preprocess one arbitrary batch of token features."""
	indices = np.asarray(flat_indices, dtype=np.int64)
	if indices.ndim != 1:
		msg = f'flat_indices must be 1D; got shape {indices.shape!r}'
		raise ValueError(msg)
	if indices.size == 0:
		return np.empty(
			(0, _transformed_feature_dim(embedding_input, preprocessor)),
			dtype=np.float32,
		)

	features = extract_token_features(embedding_input, indices)
	if residualizer is not None:
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
			'transition_costs must have shape '
			f'({k}, {k}); got {transitions.shape}'
		)
	if np.isnan(transitions).any():
		raise ValueError('transition_costs must not contain NaN values')

	t_count = emissions.shape[0]
	dp = np.empty((t_count, k), dtype=np.float64)
	backpointers = np.zeros((t_count, k), dtype=np.int32)
	dp[0] = emissions[0]

	for t_index in range(1, t_count):
		candidates = dp[t_index - 1, :, np.newaxis] + transitions
		previous = np.argmin(candidates, axis=0)
		best_transition_costs = candidates[previous, np.arange(k)]
		dp[t_index] = best_transition_costs + emissions[t_index]
		backpointers[t_index] = previous.astype(np.int32)

	final_state = int(np.argmin(dp[-1]))
	if not np.isfinite(dp[-1, final_state]):
		raise ValueError(
			'no finite path exists for emission_costs and transition_costs',
		)

	path = np.empty(t_count, dtype=np.int32)
	path[-1] = final_state
	for t_index in range(t_count - 1, 0, -1):
		path[t_index - 1] = backpointers[t_index, path[t_index]]
	return path


def contiguous_true_segments(mask: np.ndarray) -> tuple[slice, ...]:
	"""Return contiguous true spans from a one-dimensional boolean mask."""
	mask_array = np.asarray(mask)
	if mask_array.ndim != 1:
		raise ValueError(f'mask must be 1D; got shape {mask_array.shape}')
	if mask_array.dtype != np.bool_:
		raise TypeError('mask must have boolean dtype')

	segments: list[slice] = []
	start: int | None = None
	for index, is_valid in enumerate(mask_array):
		if bool(is_valid) and start is None:
			start = index
		elif not bool(is_valid) and start is not None:
			segments.append(slice(start, index))
			start = None
	if start is not None:
		segments.append(slice(start, mask_array.size))
	return tuple(segments)


def decode_trace_segments(
	emission_costs: np.ndarray,
	valid_mask: np.ndarray,
	transition_costs: np.ndarray,
) -> np.ndarray:
	"""Decode valid vertical trace segments independently."""
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
			'valid_mask must have shape '
			f'({emissions.shape[0]},); got {mask.shape}'
		)

	transitions = _as_float_matrix(transition_costs, 'transition_costs')
	k = emissions.shape[1]
	if transitions.shape != (k, k):
		raise ValueError(
			'transition_costs must have shape '
			f'({k}, {k}); got {transitions.shape}'
		)
	if np.isnan(transitions).any():
		raise ValueError('transition_costs must not contain NaN values')

	labels = np.full(emissions.shape[0], -1, dtype=np.int32)
	for segment in contiguous_true_segments(mask):
		labels[segment] = viterbi_decode_costs(emissions[segment], transitions)
	return labels


def run_stratigraphic_hmm_clustering(config: Mapping[str, object]) -> NoReturn:
	"""Run stratigraphic HMM clustering from a validated config mapping."""
	raise NotImplementedError('stratigraphic_hmm_kmeans is not implemented yet')


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
) -> int:
	if hasattr(preprocessor, 'named_steps'):
		pipeline = preprocessor
		named_steps = pipeline.named_steps
		pca = named_steps.get('pca')
		if pca is not None:
			return int(getattr(pca, 'n_components_', pca.n_components))
	return int(open_embedding_array(embedding_input).shape[-1])


__all__ = [
	'HMMTransitionSettings',
	'StratigraphicHMMSettings',
	'build_ordered_transition_costs',
	'contiguous_true_segments',
	'decode_trace_segments',
	'initialize_ordered_centers',
	'prepare_feature_batch_for_indices',
	'run_stratigraphic_hmm_clustering',
	'sample_token_z_coordinates',
	'stratigraphic_hmm_settings_from_config',
	'viterbi_decode_costs',
]
