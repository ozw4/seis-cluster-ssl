"""Writers for embedding clustering models and token labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	embedding_input_metadata,
	open_embedding_array,
	valid_flat_indices,
	validate_finite_feature_batch,
)

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
	preprocessor: object,
	kmeans: object,
	prediction_batch_size: int,
	label_metadata: Mapping[str, object],
) -> list[SurveyLabelResult]:
	"""Predict and write token labels for every survey for one k value."""
	if prediction_batch_size <= 0:
		msg = (
			'prediction_batch_size must be positive; '
			f'got {prediction_batch_size!r}'
		)
		raise ValueError(msg)
	return [
		_write_survey_labels(
			output_dir=output_dir,
			k=k,
			embedding_input=item,
			preprocessor=preprocessor,
			kmeans=kmeans,
			prediction_batch_size=prediction_batch_size,
			label_metadata=label_metadata,
		)
		for item in embedding_inputs
	]


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	"""Write stable JSON with a trailing newline."""
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _write_survey_labels(  # noqa: PLR0913
	*,
	output_dir: str | Path,
	k: int,
	embedding_input: EmbeddingInput,
	preprocessor: object,
	kmeans: object,
	prediction_batch_size: int,
	label_metadata: Mapping[str, object],
) -> SurveyLabelResult:
	embeddings = open_embedding_array(embedding_input)
	indices = valid_flat_indices(embedding_input)
	labels_dir = Path(output_dir) / 'labels' / f'k{k}'
	labels_dir.mkdir(parents=True, exist_ok=True)
	labels_path = labels_dir / f'{embedding_input.survey_id}.cluster_labels_token.npy'
	labels = np.lib.format.open_memmap(
		labels_path,
		mode='w+',
		dtype=np.int32,
		shape=embeddings.shape[:3],
	)
	labels[...] = -1
	flat_labels = labels.reshape(-1)
	flat_embeddings = embeddings.reshape((-1, embeddings.shape[-1]))
	counts = np.zeros(k, dtype=np.int64)
	for start in range(0, indices.size, prediction_batch_size):
		batch_indices = indices[start : start + prediction_batch_size]
		features = np.asarray(flat_embeddings[batch_indices], dtype=np.float32)
		validate_finite_feature_batch(features, embedding_input.survey_id)
		prepared = preprocessor.transform(features)
		predicted = np.asarray(kmeans.predict(prepared), dtype=np.int32)
		flat_labels[batch_indices] = predicted
		counts += np.bincount(predicted, minlength=k)
	labels.flush()

	cluster_counts = {
		int(label): int(count)
		for label, count in enumerate(counts)
	}
	invalid = int(labels.size - indices.size)
	metadata_path = (
		labels_dir / f'{embedding_input.survey_id}.cluster_label_metadata.json'
	)
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
	write_json(metadata_path, metadata)
	return SurveyLabelResult(
		survey_id=embedding_input.survey_id,
		labels_path=labels_path,
		metadata_path=metadata_path,
		cluster_counts=cluster_counts,
		invalid_token_count=invalid,
		valid_token_count=int(indices.size),
	)


__all__ = [
	'SurveyLabelResult',
	'write_json',
	'write_labels_for_k',
	'write_model_artifacts',
]
