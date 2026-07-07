"""Stratigraphic HMM clustering backend scaffold."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
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
from seis_ssl_cluster.clustering.sampling import sample_valid_embedding_tokens
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


def decode_survey_ordered_labels(
	embedding_input: EmbeddingInput,
	*,
	centers: np.ndarray,
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	transition_costs: np.ndarray,
) -> np.ndarray:
	"""Decode HMM labels for one survey without flattening all features at once."""
	center_matrix = np.asarray(centers, dtype=np.float32)
	if center_matrix.ndim != 2 or center_matrix.shape[0] == 0:
		msg = f'centers must be a non-empty 2D matrix; got {center_matrix.shape!r}'
		raise ValueError(msg)
	embeddings = open_embedding_array(embedding_input)
	valid = load_valid_tokens(embedding_input)
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
			)
	return labels


def update_centers_from_labels(  # noqa: C901, PLR0913
	embedding_inputs: tuple[EmbeddingInput, ...],
	labels_by_survey: Mapping[str, np.ndarray],
	*,
	centers: np.ndarray,
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	prediction_batch_size: int,
	empty_cluster_policy: str,
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
	new_centers[non_empty] = (
		sums[non_empty] / counts[non_empty, np.newaxis]
	).astype(np.float32)
	shifts = np.linalg.norm(new_centers - center_matrix, axis=1)
	empty_clusters = [int(label) for label in np.flatnonzero(~non_empty)]
	summary = {
		'cluster_counts': {
			int(label): int(count)
			for label, count in enumerate(counts)
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
	sample = sample_valid_embedding_tokens(
		embedding_inputs,
		sample_tokens=settings.sample_tokens,
		seed=settings.seed,
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
	preprocessor = fit_preprocessor(
		training_input_features,
		normalization=settings.embedding_normalization,
		pca=settings.pca,
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
		settings=settings,
		embedding_inputs=embedding_inputs,
		compatibility_signature=compatibility_signature,
		sample=sample,
		preprocessor=preprocessor,
		residualizer=residualizer,
		residualizer_path=residualizer_path,
	)

	results: list[KClusteringResult] = []
	for k in settings.k_values:
		transition_costs = build_ordered_transition_costs(k, hmm_settings.transition)
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
			labels_by_survey = {
				item.survey_id: decode_survey_ordered_labels(
					item,
					centers=centers,
					residualizer=residualizer,
					preprocessor=preprocessor,
					transition_costs=transition_costs,
				)
				for item in embedding_inputs
			}
			centers, summary = update_centers_from_labels(
				embedding_inputs,
				labels_by_survey,
				centers=centers,
				residualizer=residualizer,
				preprocessor=preprocessor,
				prediction_batch_size=settings.prediction_batch_size,
				empty_cluster_policy=hmm_settings.empty_cluster_policy,
			)
			iteration_summaries.append({'iteration': iteration, **summary})

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
			iteration_summaries=iteration_summaries,
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
				'centers': centers,
				'transition_settings': asdict(hmm_settings.transition),
				'transition_costs': transition_costs,
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
	cluster_counts = {
		int(label): int(count)
		for label, count in enumerate(counts[:k])
	}
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


def _hmm_metadata(
	*,
	hmm_settings: StratigraphicHMMSettings,
	transition_costs: np.ndarray,
	iteration_summaries: list[dict[str, object]],
) -> dict[str, object]:
	return {
		'iterations': hmm_settings.iterations,
		'z_axis': hmm_settings.z_axis,
		'z_direction': hmm_settings.z_direction,
		'transition': asdict(hmm_settings.transition),
		'transition_costs': np.asarray(transition_costs, dtype=np.float32).tolist(),
		'init': {'order_by': hmm_settings.init_order_by},
		'update': {'empty_cluster_policy': hmm_settings.empty_cluster_policy},
		'iteration_summaries': iteration_summaries,
	}


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
	'decode_survey_ordered_labels',
	'decode_trace_segments',
	'initialize_ordered_centers',
	'prepare_feature_batch_for_indices',
	'run_stratigraphic_hmm_clustering',
	'sample_token_z_coordinates',
	'stratigraphic_hmm_settings_from_config',
	'update_centers_from_labels',
	'viterbi_decode_costs',
]
