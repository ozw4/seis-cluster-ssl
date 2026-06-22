"""Embedding-only MiniBatchKMeans clustering pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	discover_embedding_inputs,
	embedding_input_metadata,
	validate_compatible_embedding_inputs,
)
from seis_ssl_cluster.clustering.sampling import (
	SampledTokens,
	sample_valid_embedding_tokens,
)
from seis_ssl_cluster.clustering.writer import (
	SurveyLabelResult,
	write_labels_for_k,
	write_model_artifacts,
)


@dataclass(frozen=True)
class PCASettings:
	"""PCA feature projection settings."""

	enabled: bool
	n_components: int
	whiten: bool


@dataclass(frozen=True)
class ClusteringSettings:
	"""Validated embedding clustering settings."""

	input_dir: Path
	output_dir: Path
	embedding_normalization: str
	pca: PCASettings
	sample_tokens: int
	method: str
	k_values: tuple[int, ...]
	minibatch_size: int
	seed: int
	prediction_batch_size: int


@dataclass(frozen=True)
class KClusteringResult:
	"""Outputs for one fitted k value."""

	k: int
	model_dir: Path
	label_results: tuple[SurveyLabelResult, ...]
	cluster_counts: dict[int, int]
	invalid_token_count: int


@dataclass(frozen=True)
class ClusteringRunResult:
	"""Result for a full clustering run."""

	embedding_inputs: tuple[EmbeddingInput, ...]
	sample: SampledTokens
	results: tuple[KClusteringResult, ...]


def run_embedding_clustering(config: Mapping[str, object]) -> ClusteringRunResult:
	"""Run embedding-only clustering from a validated config mapping."""
	settings = clustering_settings_from_config(config)
	embedding_inputs = discover_embedding_inputs(settings.input_dir)
	compatibility_signature = validate_compatible_embedding_inputs(embedding_inputs)
	sample = sample_valid_embedding_tokens(
		embedding_inputs,
		sample_tokens=settings.sample_tokens,
		seed=settings.seed,
	)
	preprocessor = fit_preprocessor(
		sample.features,
		normalization=settings.embedding_normalization,
		pca=settings.pca,
		seed=settings.seed,
	)
	training_features = np.asarray(
		preprocessor.transform(sample.features),
		dtype=np.float32,
	)
	common_metadata = _common_metadata(
		settings=settings,
		embedding_inputs=embedding_inputs,
		compatibility_signature=compatibility_signature,
		sample=sample,
		preprocessor=preprocessor,
	)
	results: list[KClusteringResult] = []
	for k in settings.k_values:
		kmeans = fit_minibatch_kmeans(
			training_features,
			k=k,
			batch_size=settings.minibatch_size,
			seed=settings.seed,
		)
		label_results = write_labels_for_k(
			output_dir=settings.output_dir,
			k=k,
			embedding_inputs=embedding_inputs,
			preprocessor=preprocessor,
			kmeans=kmeans,
			prediction_batch_size=settings.prediction_batch_size,
			label_metadata=common_metadata,
		)
		cluster_counts = _aggregate_counts(label_results, k)
		invalid_token_count = int(
			sum(result.invalid_token_count for result in label_results),
		)
		metadata = {
			**common_metadata,
			'k': int(k),
			'kmeans': _kmeans_metadata(kmeans, settings=settings, k=k),
			'cluster_counts': cluster_counts,
			'invalid_token_count': invalid_token_count,
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
		write_model_artifacts(
			output_dir=settings.output_dir,
			k=k,
			preprocessor=preprocessor,
			kmeans=kmeans,
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
		embedding_inputs=tuple(embedding_inputs),
		sample=sample,
		results=tuple(results),
	)


def clustering_settings_from_config(
	config: Mapping[str, object],
) -> ClusteringSettings:
	"""Build clustering settings from config sections."""
	embeddings = _required_mapping(config, 'embeddings')
	clustering = _required_mapping(config, 'clustering')
	pca_cfg = _optional_mapping(clustering, 'pca')
	minibatch_size = _positive_int(
		clustering.get('minibatch_size', 8192),
		'clustering.minibatch_size',
	)
	return ClusteringSettings(
		input_dir=_required_path(embeddings, 'input_dir', 'embeddings'),
		output_dir=_required_path(clustering, 'output_dir', 'clustering'),
		embedding_normalization=_normalization_method(
			clustering.get('embedding_normalization', 'l2'),
		),
		pca=PCASettings(
			enabled=_bool(pca_cfg.get('enabled', False), 'clustering.pca.enabled'),
			n_components=_positive_int(
				pca_cfg.get('n_components', 64),
				'clustering.pca.n_components',
			),
			whiten=_bool(pca_cfg.get('whiten', False), 'clustering.pca.whiten'),
		),
		sample_tokens=_positive_int(
			clustering.get('sample_tokens', 1_000_000),
			'clustering.sample_tokens',
		),
		method=_method(clustering.get('method', 'minibatch_kmeans')),
		k_values=_k_values(clustering.get('k_values')),
		minibatch_size=minibatch_size,
		seed=_int(clustering.get('seed', 42), 'clustering.seed'),
		prediction_batch_size=_positive_int(
			clustering.get('prediction_batch_size', minibatch_size),
			'clustering.prediction_batch_size',
		),
	)


def fit_preprocessor(
	features: np.ndarray,
	*,
	normalization: str,
	pca: PCASettings,
	seed: int,
) -> Pipeline:
	"""Fit normalization and optional PCA on sampled training features."""
	matrix = np.asarray(features, dtype=np.float32)
	if matrix.ndim != 2 or matrix.shape[0] == 0:
		msg = f'features must be a non-empty 2D matrix; got {matrix.shape!r}'
		raise ValueError(msg)
	steps: list[tuple[str, object]] = []
	if normalization == 'l2':
		steps.append(('normalizer', Normalizer(norm='l2')))
	elif normalization == 'none':
		steps.append(
			(
				'identity',
				FunctionTransformer(validate=False, feature_names_out='one-to-one'),
			),
		)
	else:
		msg = f'unsupported embedding_normalization: {normalization!r}'
		raise ValueError(msg)
	if pca.enabled:
		_validate_pca_components(pca.n_components, matrix.shape)
		steps.append(
			(
				'pca',
				PCA(
					n_components=pca.n_components,
					whiten=pca.whiten,
					random_state=seed,
				),
			),
		)
	pipeline = Pipeline(steps)
	pipeline.fit(matrix)
	return pipeline


def fit_minibatch_kmeans(
	features: np.ndarray,
	*,
	k: int,
	batch_size: int,
	seed: int,
) -> MiniBatchKMeans:
	"""Fit one deterministic MiniBatchKMeans model."""
	matrix = np.asarray(features, dtype=np.float32)
	if matrix.ndim != 2 or matrix.shape[0] == 0:
		msg = f'features must be a non-empty 2D matrix; got {matrix.shape!r}'
		raise ValueError(msg)
	if k > matrix.shape[0]:
		msg = f'k must be <= sample count; got k={k}, sample_count={matrix.shape[0]}'
		raise ValueError(msg)
	model = MiniBatchKMeans(
		n_clusters=k,
		random_state=seed,
		batch_size=batch_size,
		n_init='auto',
		reassignment_ratio=0.0,
	)
	model.fit(matrix)
	return model


def _common_metadata(
	*,
	settings: ClusteringSettings,
	embedding_inputs: Sequence[EmbeddingInput],
	compatibility_signature: Mapping[str, object],
	sample: SampledTokens,
	preprocessor: Pipeline,
) -> dict[str, object]:
	return {
		'embedding_inputs': [
			embedding_input_metadata(item)
			for item in embedding_inputs
		],
		'embedding_compatibility_signature': dict(compatibility_signature),
		'normalization': settings.embedding_normalization,
		'pca': {
			'enabled': settings.pca.enabled,
			'n_components': settings.pca.n_components,
			'effective_n_components': _effective_pca_components(preprocessor),
			'whiten': settings.pca.whiten,
		},
		'sample': {
			'requested_count': sample.requested_count,
			'count': sample.sample_count,
			'total_valid_count': sample.total_valid_count,
			'per_survey_count': {
				survey_id: int(indices.size)
				for survey_id, indices in sample.per_survey_token_indices.items()
			},
		},
		'method': settings.method,
		'k_values': list(settings.k_values),
		'minibatch_size': settings.minibatch_size,
		'prediction_batch_size': settings.prediction_batch_size,
		'random_seed': settings.seed,
	}


def _kmeans_metadata(
	kmeans: MiniBatchKMeans,
	*,
	settings: ClusteringSettings,
	k: int,
) -> dict[str, object]:
	return {
		'method': settings.method,
		'n_clusters': int(k),
		'batch_size': settings.minibatch_size,
		'random_state': settings.seed,
		'n_iter': int(kmeans.n_iter_),
		'inertia': float(kmeans.inertia_),
	}


def _aggregate_counts(
	label_results: Sequence[SurveyLabelResult],
	k: int,
) -> dict[int, int]:
	counts = np.zeros(k, dtype=np.int64)
	for result in label_results:
		for label, count in result.cluster_counts.items():
			counts[int(label)] += int(count)
	return {
		int(label): int(count)
		for label, count in enumerate(counts)
	}


def _validate_pca_components(
	n_components: int,
	shape: tuple[int, int],
) -> None:
	limit = min(shape)
	if n_components > limit:
		msg = (
			'clustering.pca.n_components must be <= min(sample_count, '
			f'embedding_dim); got {n_components}, limit={limit}'
		)
		raise ValueError(msg)


def _effective_pca_components(preprocessor: Pipeline) -> int | None:
	pca = preprocessor.named_steps.get('pca')
	if pca is None:
		return None
	return int(cast('PCA', pca).n_components_)


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _required_path(
	parent: Mapping[str, object],
	key: str,
	prefix: str,
) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _positive_int(value: object, name: str) -> int:
	if not isinstance(value, Integral) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	result = int(value)
	if result <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return result


def _int(value: object, name: str) -> int:
	if not isinstance(value, Integral) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	return int(value)


def _bool(value: object, name: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{name} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _normalization_method(value: object) -> str:
	if value not in {'l2', 'none'}:
		msg = "clustering.embedding_normalization must be 'l2' or 'none'"
		raise ValueError(msg)
	return str(value)


def _method(value: object) -> str:
	if value != 'minibatch_kmeans':
		msg = "clustering.method must be 'minibatch_kmeans'"
		raise ValueError(msg)
	return str(value)


def _k_values(value: object) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'clustering.k_values must be a non-empty integer list; got {value!r}'
		raise TypeError(msg)
	values = tuple(_positive_int(item, 'clustering.k_values') for item in value)
	if not values:
		msg = 'clustering.k_values must not be empty'
		raise ValueError(msg)
	if len(set(values)) != len(values):
		msg = f'clustering.k_values must not contain duplicates; got {values!r}'
		raise ValueError(msg)
	return values


__all__ = [
	'ClusteringRunResult',
	'ClusteringSettings',
	'KClusteringResult',
	'PCASettings',
	'clustering_settings_from_config',
	'fit_minibatch_kmeans',
	'fit_preprocessor',
	'run_embedding_clustering',
]
