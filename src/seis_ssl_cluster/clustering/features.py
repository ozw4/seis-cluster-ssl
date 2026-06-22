"""Feature loading helpers for embedding-only clustering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Iterator, Sequence


@dataclass(frozen=True)
class EmbeddingInput:
	"""Input artifact paths for one survey's embedding grid."""

	survey_id: str
	embeddings_path: Path
	valid_tokens_path: Path
	metadata_path: Path


@dataclass(frozen=True)
class FeatureBatch:
	"""A batch of valid token features from one survey."""

	survey_id: str
	token_indices: np.ndarray
	features: np.ndarray


COMPATIBILITY_METADATA_FIELDS = (
	'checkpoint_sha256',
	'model_geometry',
	'patch_size',
	'window_size',
	'overlap',
	'min_token_valid_fraction',
	'zero_mask',
)


def discover_embedding_inputs(input_dir: str | Path) -> list[EmbeddingInput]:
	"""Discover per-survey embedding artifacts in deterministic survey order."""
	root = Path(input_dir)
	if not root.is_dir():
		msg = f'embeddings.input_dir must be an existing directory: {root}'
		raise FileNotFoundError(msg)

	inputs: list[EmbeddingInput] = []
	for embeddings_path in sorted(root.glob('*.embeddings.npy')):
		survey_id = embeddings_path.name.removesuffix('.embeddings.npy')
		item = EmbeddingInput(
			survey_id=survey_id,
			embeddings_path=embeddings_path,
			valid_tokens_path=root / f'{survey_id}.valid_tokens.npy',
			metadata_path=root / f'{survey_id}.embedding_metadata.json',
		)
		_validate_embedding_input(item)
		inputs.append(item)
	if not inputs:
		msg = f'no embedding inputs found in {root}'
		raise ValueError(msg)
	return inputs


def count_valid_tokens(embedding_input: EmbeddingInput) -> int:
	"""Return the number of valid tokens in one survey."""
	valid = load_valid_tokens(embedding_input)
	return int(np.count_nonzero(valid))


def embedding_dim(embedding_input: EmbeddingInput) -> int:
	"""Return the embedding channel dimension for one survey."""
	embeddings = open_embedding_array(embedding_input)
	return int(embeddings.shape[-1])


def load_valid_tokens(embedding_input: EmbeddingInput) -> np.ndarray:
	"""Load a survey valid-token mask as a memory-mapped array."""
	valid = np.load(embedding_input.valid_tokens_path, mmap_mode='r')
	if valid.dtype != np.bool_:
		msg = (
			f'valid_tokens dtype must be bool for {embedding_input.survey_id}; '
			f'got {valid.dtype}'
		)
		raise TypeError(msg)
	if valid.ndim != 3:
		msg = (
			f'valid_tokens must be 3D for {embedding_input.survey_id}; '
			f'got shape={valid.shape!r}'
		)
		raise ValueError(msg)
	return valid


def open_embedding_array(embedding_input: EmbeddingInput) -> np.ndarray:
	"""Open a survey embedding grid as a memory-mapped array."""
	embeddings = np.load(embedding_input.embeddings_path, mmap_mode='r')
	if embeddings.ndim != 4:
		msg = (
			f'embeddings must be 4D for {embedding_input.survey_id}; '
			f'got shape={embeddings.shape!r}'
		)
		raise ValueError(msg)
	if embeddings.dtype.kind not in {'f', 'i', 'u'}:
		msg = (
			f'embeddings dtype must be numeric for {embedding_input.survey_id}; '
			f'got {embeddings.dtype}'
		)
		raise TypeError(msg)
	valid = load_valid_tokens(embedding_input)
	if embeddings.shape[:3] != valid.shape:
		msg = (
			f'embeddings token grid must match valid_tokens for '
			f'{embedding_input.survey_id}; got {embeddings.shape[:3]!r} and '
			f'{valid.shape!r}'
		)
		raise ValueError(msg)
	return embeddings


def valid_flat_indices(embedding_input: EmbeddingInput) -> np.ndarray:
	"""Return flattened token indices whose embedding token is valid."""
	valid = load_valid_tokens(embedding_input)
	return np.flatnonzero(valid.reshape(-1))


def extract_token_features(
	embedding_input: EmbeddingInput,
	token_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
	"""Read selected flattened token embeddings as a float32 feature matrix."""
	embeddings = open_embedding_array(embedding_input)
	flat = embeddings.reshape((-1, embeddings.shape[-1]))
	indices = np.asarray(token_indices, dtype=np.int64)
	if indices.ndim != 1:
		msg = f'token_indices must be 1D; got shape={indices.shape!r}'
		raise ValueError(msg)
	if indices.size == 0:
		return np.empty((0, embeddings.shape[-1]), dtype=np.float32)
	features = np.asarray(flat[indices], dtype=np.float32)
	_validate_finite_features(features, embedding_input.survey_id)
	return features


def iter_valid_feature_batches(
	embedding_input: EmbeddingInput,
	*,
	batch_size: int,
) -> Iterator[FeatureBatch]:
	"""Yield valid embedding features for one survey in flattened token order."""
	if batch_size <= 0:
		msg = f'batch_size must be positive; got {batch_size!r}'
		raise ValueError(msg)
	indices = valid_flat_indices(embedding_input)
	for start in range(0, indices.size, batch_size):
		batch_indices = indices[start : start + batch_size]
		yield FeatureBatch(
			survey_id=embedding_input.survey_id,
			token_indices=batch_indices,
			features=extract_token_features(embedding_input, batch_indices),
		)


def file_sha256(path: str | Path) -> str:
	"""Return the SHA-256 hex digest for a file."""
	digest = hashlib.sha256()
	with Path(path).open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def embedding_input_metadata(embedding_input: EmbeddingInput) -> dict[str, object]:
	"""Return deterministic metadata describing one embedding input."""
	return {
		'survey_id': embedding_input.survey_id,
		'embeddings_path': str(embedding_input.embeddings_path),
		'valid_tokens_path': str(embedding_input.valid_tokens_path),
		'metadata_path': str(embedding_input.metadata_path),
		'metadata_sha256': file_sha256(embedding_input.metadata_path),
	}


def load_embedding_metadata(embedding_input: EmbeddingInput) -> dict[str, object]:
	"""Load one survey's extraction metadata JSON."""
	try:
		payload = json.loads(embedding_input.metadata_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		msg = (
			f'embedding metadata must be valid JSON for '
			f'{embedding_input.survey_id}: {embedding_input.metadata_path}'
		)
		raise ValueError(msg) from exc
	if not isinstance(payload, dict):
		msg = (
			f'embedding metadata must be a JSON object for '
			f'{embedding_input.survey_id}: {embedding_input.metadata_path}'
		)
		raise TypeError(msg)
	return payload


def embedding_compatibility_signature(
	embedding_input: EmbeddingInput,
) -> dict[str, object]:
	"""Return representation-defining metadata for clustering compatibility."""
	metadata = load_embedding_metadata(embedding_input)
	missing = [
		field
		for field in COMPATIBILITY_METADATA_FIELDS
		if field not in metadata
	]
	if missing:
		msg = (
			f'embedding metadata missing compatibility fields for '
			f'{embedding_input.survey_id}: {missing!r}'
		)
		raise ValueError(msg)
	return {
		**{
			field: metadata[field]
			for field in COMPATIBILITY_METADATA_FIELDS
		},
		'embedding_dim': embedding_dim(embedding_input),
	}


def validate_compatible_embedding_inputs(
	embedding_inputs: Sequence[EmbeddingInput],
) -> dict[str, object]:
	"""Require all survey embeddings to share one representation signature."""
	if not embedding_inputs:
		msg = 'at least one embedding input is required'
		raise ValueError(msg)
	signatures = [
		(item, embedding_compatibility_signature(item))
		for item in embedding_inputs
	]
	baseline_input, baseline = signatures[0]
	for item, signature in signatures[1:]:
		if signature != baseline:
			differing = [
				field
				for field, value in baseline.items()
				if signature.get(field) != value
			]
			msg = (
				'incompatible embedding artifacts for surveys '
				f'{baseline_input.survey_id!r} and {item.survey_id!r}; '
				f'differing fields: {", ".join(differing)}'
			)
			raise ValueError(msg)
	return baseline


def validate_finite_feature_batch(features: np.ndarray, survey_id: str) -> None:
	"""Raise when a feature batch contains NaN or infinity values."""
	_validate_finite_features(features, survey_id)


def _validate_embedding_input(embedding_input: EmbeddingInput) -> None:
	missing = [
		path
		for path in (
			embedding_input.embeddings_path,
			embedding_input.valid_tokens_path,
			embedding_input.metadata_path,
		)
		if not path.is_file()
	]
	if missing:
		msg = (
			f'missing embedding artifacts for {embedding_input.survey_id}: '
			f'{[str(path) for path in missing]!r}'
		)
		raise FileNotFoundError(msg)


def _validate_finite_features(features: np.ndarray, survey_id: str) -> None:
	if not np.isfinite(features).all():
		msg = f'non-finite embedding features found for survey {survey_id}'
		raise ValueError(msg)


__all__ = [
	'COMPATIBILITY_METADATA_FIELDS',
	'EmbeddingInput',
	'FeatureBatch',
	'count_valid_tokens',
	'discover_embedding_inputs',
	'embedding_compatibility_signature',
	'embedding_dim',
	'embedding_input_metadata',
	'extract_token_features',
	'file_sha256',
	'iter_valid_feature_batches',
	'load_embedding_metadata',
	'load_valid_tokens',
	'open_embedding_array',
	'valid_flat_indices',
	'validate_compatible_embedding_inputs',
	'validate_finite_feature_batch',
]
