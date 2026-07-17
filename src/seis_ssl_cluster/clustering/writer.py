"""Writers for embedding clustering models and token labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import joblib
import numpy as np

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	embedding_input_metadata,
	open_embedding_array,
	valid_flat_indices,
	validate_finite_feature_batch,
)
from seis_ssl_cluster.clustering.residualization import (
	LocalTokenPositionResidualizer,
	residualization_groups_for_flat_indices,
)
from seis_ssl_cluster.utils import StageTimer

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class SurveyLabelResult:
	"""Written labels and counts for one survey."""

	survey_id: str
	labels_path: Path
	metadata_path: Path
	cluster_counts: dict[int, int]
	invalid_token_count: int
	valid_token_count: int


def write_model_artifacts(
	*,
	output_dir: str | Path,
	k: int,
	preprocessor: object,
	kmeans: object,
	metadata: Mapping[str, object],
) -> None:
	"""Write model artifacts for one k value."""
	model_dir = Path(output_dir) / 'models' / f'k{k}'
	model_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(preprocessor, model_dir / 'preprocessor.joblib')
	joblib.dump(kmeans, model_dir / 'kmeans.joblib')
	centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
	np.save(model_dir / 'cluster_centers.npy', centers)
	write_json(model_dir / 'clustering_metadata.json', metadata)


def write_labels_for_k(  # noqa: PLR0913
	*,
	output_dir: str | Path,
	k: int,
	embedding_inputs: Sequence[EmbeddingInput],
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	kmeans: object,
	prediction_batch_size: int,
	label_metadata: Mapping[str, object],
	timer: StageTimer | None = None,
) -> list[SurveyLabelResult]:
	"""Predict and write token labels for every survey for one k value."""
	return write_labels_for_models(
		output_dir=output_dir,
		kmeans_by_k={k: kmeans},
		embedding_inputs=embedding_inputs,
		residualizer=residualizer,
		preprocessor=preprocessor,
		prediction_batch_size=prediction_batch_size,
		label_metadata=label_metadata,
		timer=timer,
	)[k]


def write_labels_for_models(  # noqa: PLR0913
	*,
	output_dir: str | Path,
	kmeans_by_k: Mapping[int, object],
	embedding_inputs: Sequence[EmbeddingInput],
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	prediction_batch_size: int,
	label_metadata: Mapping[str, object],
	timer: StageTimer | None = None,
) -> dict[int, list[SurveyLabelResult]]:
	"""Write labels for many models while preparing each feature batch once."""
	if prediction_batch_size <= 0:
		msg = (
			'prediction_batch_size must be positive; '
			f'got {prediction_batch_size!r}'
		)
		raise ValueError(msg)
	models = _validated_models(kmeans_by_k)
	stage_timer = timer or StageTimer()
	results = {k: [] for k in models}
	partial_paths: list[Path] = []
	pending: list[tuple[Path, Path]] = []
	try:
		for item in embedding_inputs:
			survey_results, survey_pending = _write_survey_labels_for_models(
				output_dir=output_dir,
				models=models,
				embedding_input=item,
				residualizer=residualizer,
				preprocessor=preprocessor,
				prediction_batch_size=prediction_batch_size,
				label_metadata=label_metadata,
				timer=stage_timer,
				cleanup_paths=partial_paths,
			)
			pending.extend(survey_pending)
			for k, result in survey_results.items():
				results[k].append(result)
		for partial, final in pending:
			partial.replace(final)
			partial_paths.remove(partial)
	except BaseException:
		for path in partial_paths:
			path.unlink(missing_ok=True)
		raise
	return results


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	"""Write stable JSON with a trailing newline."""
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _write_survey_labels_for_models(  # noqa: PLR0913
	*,
	output_dir: str | Path,
	models: Mapping[int, object],
	embedding_input: EmbeddingInput,
	residualizer: LocalTokenPositionResidualizer | None,
	preprocessor: object,
	prediction_batch_size: int,
	label_metadata: Mapping[str, object],
	timer: StageTimer,
	cleanup_paths: list[Path],
) -> tuple[dict[int, SurveyLabelResult], list[tuple[Path, Path]]]:
	embeddings = open_embedding_array(embedding_input)
	indices = valid_flat_indices(embedding_input)
	flat_embeddings = embeddings.reshape((-1, embeddings.shape[-1]))
	paths: dict[int, tuple[Path, Path, Path, Path]] = {}
	counts = {k: np.zeros(k, dtype=np.int64) for k in models}
	for k in models:
		labels_dir = Path(output_dir) / 'labels' / f'k{k}'
		labels_dir.mkdir(parents=True, exist_ok=True)
		labels_path = (
			labels_dir / f'{embedding_input.survey_id}.cluster_labels_token.npy'
		)
		metadata_path = (
			labels_dir
			/ f'{embedding_input.survey_id}.cluster_label_metadata.json'
		)
		partial_label = _partial_path(labels_path)
		partial_metadata = _partial_path(metadata_path)
		paths[k] = (labels_path, metadata_path, partial_label, partial_metadata)
		cleanup_paths.extend((partial_label, partial_metadata))
		with timer.stage(
			'write',
			sample_count=int(np.prod(embeddings.shape[:3])),
		):
			labels = np.lib.format.open_memmap(
				partial_label,
				mode='w+',
				dtype=np.int32,
				shape=embeddings.shape[:3],
			)
			labels[...] = -1
			labels.flush()
			del labels
	for start in range(0, indices.size, prediction_batch_size):
		batch_indices = indices[start : start + prediction_batch_size]
		with timer.stage('feature_read', sample_count=int(batch_indices.size)):
			features = np.asarray(flat_embeddings[batch_indices], dtype=np.float32)
			validate_finite_feature_batch(features, embedding_input.survey_id)
		with timer.stage('preprocess', sample_count=int(batch_indices.size)):
			if residualizer is not None:
				groups = residualization_groups_for_flat_indices(
					embedding_input,
					batch_indices,
					residualizer=residualizer,
				)
				features = residualizer.transform(features, groups)
			prepared = preprocessor.transform(features)
		for k, kmeans in models.items():
			with timer.stage(f'predict_k{k}', sample_count=int(batch_indices.size)):
				predicted = np.asarray(kmeans.predict(prepared), dtype=np.int32)
			counts[k] += np.bincount(predicted, minlength=k)
			with timer.stage('write', sample_count=int(batch_indices.size)):
				labels = np.lib.format.open_memmap(paths[k][2], mode='r+')
				labels.reshape(-1)[batch_indices] = predicted
				labels.flush()
				del labels

	invalid = int(np.prod(embeddings.shape[:3]) - indices.size)
	results: dict[int, SurveyLabelResult] = {}
	pending: list[tuple[Path, Path]] = []
	for k in models:
		labels_path, metadata_path, partial_label, partial_metadata = paths[k]
		cluster_counts = {
			int(label): int(count)
			for label, count in enumerate(counts[k])
		}
		metadata = {
			**dict(label_metadata),
			'k': int(k),
			'survey_id': embedding_input.survey_id,
			'embedding_input': embedding_input_metadata(embedding_input),
			'label_path': str(labels_path),
			'token_grid_shape': list(embeddings.shape[:3]),
			'embedding_dim': int(embeddings.shape[-1]),
			'valid_token_count': int(indices.size),
			'invalid_token_count': invalid,
			'cluster_counts': cluster_counts,
		}
		with timer.stage('write'):
			write_json(partial_metadata, metadata)
		results[k] = SurveyLabelResult(
			survey_id=embedding_input.survey_id,
			labels_path=labels_path,
			metadata_path=metadata_path,
			cluster_counts=cluster_counts,
			invalid_token_count=invalid,
			valid_token_count=int(indices.size),
		)
		pending.extend(
			((partial_label, labels_path), (partial_metadata, metadata_path)),
		)
	return results, pending


def _validated_models(kmeans_by_k: Mapping[int, object]) -> dict[int, object]:
	if not kmeans_by_k:
		msg = 'kmeans_by_k must not be empty'
		raise ValueError(msg)
	models: dict[int, object] = {}
	for k, model in kmeans_by_k.items():
		if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
			msg = f'kmeans_by_k keys must be positive integers; got {k!r}'
			raise ValueError(msg)
		models[k] = model
	return models


def _partial_path(path: Path) -> Path:
	return path.with_name(f'.{path.name}.{uuid4().hex}.partial')


__all__ = [
	'SurveyLabelResult',
	'write_json',
	'write_labels_for_k',
	'write_labels_for_models',
	'write_model_artifacts',
]
