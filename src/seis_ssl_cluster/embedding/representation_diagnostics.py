"""Deterministic diagnostics for full-volume encoder token embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from seis_ssl_cluster.config.barlow_twins import (
	resolve_barlow_twins_pretraining_method,
)
from seis_ssl_cluster.config.schema import (
	LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	OVERLAPPING_SUBCROP_XY_AUGMENTATION_POLICY,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.models.amplitude_encoder_factory import (
	checkpoint_config_from_payload,
	is_random_encoder_checkpoint,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint

REPRESENTATION_DIAGNOSTIC_SCHEMA_VERSION = 1
REPRESENTATION_DIAGNOSTIC_ARTIFACT_TYPE = 'embedding_representation_diagnostics'
DEFAULT_REPRESENTATION_SAMPLE_SIZE = 8_192
DEFAULT_REPRESENTATION_LAYER_NORM_EPS = 1.0e-5
REPRESENTATION_METRIC_KEYS = (
	'raw_feature_norm',
	'token_wise_feature_std',
	'raw_feature_effective_rank',
	'layer_norm_feature_std',
	'layer_norm_effective_rank',
)
SYSTEMATIC_SAMPLING_METHOD = 'valid_token_midpoint_systematic_c_order_v1'
EFFECTIVE_RANK_METHOD = 'centered_covariance_eigenvalue_entropy_v1'


@dataclass(frozen=True)
class EmbeddingRepresentationSource:
	"""Paths and scientific identity for one representation source."""

	source_id: str
	survey_id: str
	checkpoint_path: Path
	embeddings_path: Path
	valid_tokens_path: Path
	metadata_path: Path
	random_baseline: bool


def systematic_valid_token_indices(
	valid_tokens: np.ndarray,
	*,
	sample_size: int = DEFAULT_REPRESENTATION_SAMPLE_SIZE,
) -> np.ndarray:
	"""Select fixed midpoint-systematic flat indices from C-ordered valid tokens."""
	if not isinstance(sample_size, int) or isinstance(sample_size, bool):
		raise TypeError(f'sample_size must be an integer; got {sample_size!r}')
	if sample_size <= 0:
		raise ValueError(f'sample_size must be positive; got {sample_size!r}')
	mask = np.asarray(valid_tokens)
	if mask.dtype != np.dtype(bool):
		raise TypeError(f'valid_tokens dtype must be bool; got {mask.dtype!s}')
	if mask.ndim < 1:
		raise ValueError('valid_tokens must have at least one dimension')
	eligible = np.flatnonzero(mask.reshape(-1, order='C'))
	eligible_count = int(eligible.size)
	if eligible_count < sample_size:
		raise ValueError(
			'valid token count is smaller than the fixed representation sample: '
			f'valid={eligible_count}, required={sample_size}'
		)
	positions = ((2 * np.arange(sample_size, dtype=np.int64) + 1) * eligible_count) // (
		2 * sample_size
	)
	indices = np.asarray(eligible[positions], dtype=np.int64)
	if indices.size != sample_size or np.unique(indices).size != sample_size:
		raise RuntimeError('midpoint-systematic sampling produced duplicate indices')
	return indices


def representation_metrics(
	features: np.ndarray,
	*,
	layer_norm_eps: float = DEFAULT_REPRESENTATION_LAYER_NORM_EPS,
) -> dict[str, float]:
	"""Calculate the fixed five raw and post-LayerNorm feature diagnostics."""
	values = np.asarray(features, dtype=np.float64)
	if values.ndim != 2:
		raise ValueError(
			f'features must have shape [token, feature]; got {tuple(values.shape)!r}'
		)
	if values.shape[0] < 2 or values.shape[1] < 1:
		raise ValueError('features require at least two tokens and one dimension')
	if not math.isfinite(layer_norm_eps) or layer_norm_eps <= 0.0:
		raise ValueError(
			f'layer_norm_eps must be positive and finite; got {layer_norm_eps!r}'
		)
	if not bool(np.isfinite(values).all()):
		raise FloatingPointError('sampled representation features are non-finite')

	raw_norm = float(np.linalg.norm(values, axis=1).mean())
	raw_std = _mean_feature_std(values)
	raw_effective_rank = _effective_rank(values)

	per_token_centered = values - values.mean(axis=1, keepdims=True)
	per_token_variance = np.mean(
		per_token_centered * per_token_centered,
		axis=1,
		keepdims=True,
	)
	layer_normalized = per_token_centered / np.sqrt(per_token_variance + layer_norm_eps)
	metrics = {
		'raw_feature_norm': _finite_metric(raw_norm, 'raw feature norm'),
		'token_wise_feature_std': _finite_metric(raw_std, 'raw feature std'),
		'raw_feature_effective_rank': _finite_metric(
			raw_effective_rank,
			'raw feature effective rank',
		),
		'layer_norm_feature_std': _finite_metric(
			_mean_feature_std(layer_normalized),
			'LayerNorm feature std',
		),
		'layer_norm_effective_rank': _finite_metric(
			_effective_rank(layer_normalized),
			'LayerNorm effective rank',
		),
	}
	if tuple(metrics) != REPRESENTATION_METRIC_KEYS:
		raise RuntimeError('representation metric key order differs from the contract')
	return metrics


def build_embedding_representation_diagnostics(  # noqa: PLR0913
	source: EmbeddingRepresentationSource,
	*,
	expected_token_grid_shape: Sequence[int],
	expected_embedding_dim: int,
	expected_valid_mask_sha256: str,
	sample_size: int = DEFAULT_REPRESENTATION_SAMPLE_SIZE,
	layer_norm_eps: float = DEFAULT_REPRESENTATION_LAYER_NORM_EPS,
	expected_candidate_epoch: int = 10,
	expected_candidate_global_step: int = 6_250,
	expected_random_seed: int = 42,
) -> dict[str, object]:
	"""Build one provenance-bound diagnostic payload without writing it."""
	_validate_source(source)
	grid_shape = _positive_shape(expected_token_grid_shape, 'token grid shape')
	if not isinstance(expected_embedding_dim, int) or isinstance(
		expected_embedding_dim,
		bool,
	):
		raise TypeError('expected_embedding_dim must be an integer')
	if expected_embedding_dim <= 0:
		raise ValueError('expected_embedding_dim must be positive')
	_validate_sha256(expected_valid_mask_sha256, 'expected valid-mask SHA-256')

	for label, path in _source_paths(source).items():
		if not path.is_file():
			raise FileNotFoundError(f'{label} does not exist: {path}')

	metadata = _read_json_mapping(source.metadata_path, 'embedding metadata')
	checkpoint_sha256 = file_sha256(source.checkpoint_path)
	_validate_embedding_metadata(
		metadata,
		source=source,
		checkpoint_sha256=checkpoint_sha256,
		expected_token_grid_shape=grid_shape,
		expected_embedding_dim=expected_embedding_dim,
	)
	payload = load_checkpoint(source.checkpoint_path, map_location='cpu')
	checkpoint_config = checkpoint_config_from_payload(payload)
	checkpoint_identity = _validate_checkpoint_identity(
		payload,
		checkpoint_config=checkpoint_config,
		random_baseline=source.random_baseline,
		expected_candidate_epoch=expected_candidate_epoch,
		expected_candidate_global_step=expected_candidate_global_step,
		expected_random_seed=expected_random_seed,
	)

	embeddings = np.load(
		source.embeddings_path,
		mmap_mode='r',
		allow_pickle=False,
	)
	valid_tokens = np.load(
		source.valid_tokens_path,
		mmap_mode='r',
		allow_pickle=False,
	)
	try:
		_validate_arrays(
			embeddings,
			valid_tokens,
			expected_token_grid_shape=grid_shape,
			expected_embedding_dim=expected_embedding_dim,
		)
		valid_mask_sha256 = file_sha256(source.valid_tokens_path)
		if valid_mask_sha256 != expected_valid_mask_sha256:
			raise ValueError(
				'valid-token mask SHA-256 differs from the fixed Random baseline '
				f'contract: expected={expected_valid_mask_sha256}, '
				f'actual={valid_mask_sha256}'
			)
		indices = systematic_valid_token_indices(
			valid_tokens,
			sample_size=sample_size,
		)
		flat_embeddings = embeddings.reshape(-1, expected_embedding_dim)
		features = np.asarray(flat_embeddings[indices], dtype=np.float64)
		metrics = representation_metrics(
			features,
			layer_norm_eps=layer_norm_eps,
		)
		valid_count = int(np.count_nonzero(valid_tokens))
	finally:
		del embeddings
		del valid_tokens

	return {
		'artifact_type': REPRESENTATION_DIAGNOSTIC_ARTIFACT_TYPE,
		'schema_version': REPRESENTATION_DIAGNOSTIC_SCHEMA_VERSION,
		'source_id': source.source_id,
		'source_kind': 'random_baseline' if source.random_baseline else 'candidate',
		'survey_id': source.survey_id,
		'metrics': metrics,
		'sampling': {
			'method': SYSTEMATIC_SAMPLING_METHOD,
			'eligible_valid_token_count': valid_count,
			'sample_size': sample_size,
			'sample_flat_indices_sha256': _indices_sha256(indices),
			'sample_flat_indices_hash_dtype': 'little_endian_int64',
			'token_grid_shape': list(grid_shape),
			'embedding_dim': expected_embedding_dim,
		},
		'calculation': {
			'input': 'merged_full_volume_bare_encoder_embedding_v1',
			'input_dtype': 'float16',
			'calculation_dtype': 'float64',
			'feature_norm': 'mean_token_l2_v1',
			'feature_std': 'mean_dimension_population_std_across_tokens_v1',
			'effective_rank': EFFECTIVE_RANK_METHOD,
			'layer_norm': {
				'axis': 'per_token_feature_dimension',
				'affine': False,
				'eps': layer_norm_eps,
				'variance': 'population',
			},
		},
		'provenance': {
			'checkpoint': {
				'path': str(source.checkpoint_path),
				'sha256': checkpoint_sha256,
				**checkpoint_identity,
			},
			'embeddings': {
				'path': str(source.embeddings_path),
				'sha256': file_sha256(source.embeddings_path),
				'shape': [*grid_shape, expected_embedding_dim],
				'dtype': 'float16',
			},
			'valid_tokens': {
				'path': str(source.valid_tokens_path),
				'sha256': expected_valid_mask_sha256,
				'shape': list(grid_shape),
				'dtype': 'bool',
			},
			'embedding_metadata': {
				'path': str(source.metadata_path),
				'sha256': file_sha256(source.metadata_path),
			},
		},
	}


def write_embedding_representation_diagnostics(
	path: str | Path,
	payload: Mapping[str, object],
) -> Path:
	"""Atomically write one diagnostic JSON artifact."""
	metrics = payload.get('metrics')
	if not isinstance(metrics, Mapping) or set(metrics) != set(
		REPRESENTATION_METRIC_KEYS
	):
		raise ValueError('diagnostic payload must contain the exact five metric keys')
	target = Path(path)
	target.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary_name = tempfile.mkstemp(
		prefix=f'.{target.name}.',
		suffix='.tmp',
		dir=target.parent,
	)
	temporary = Path(temporary_name)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as file_obj:
			json.dump(payload, file_obj, indent=2, sort_keys=True)
			file_obj.write('\n')
			file_obj.flush()
			os.fsync(file_obj.fileno())
		temporary.replace(target)
	finally:
		if temporary.exists():
			temporary.unlink()
	return target


def _mean_feature_std(features: np.ndarray) -> float:
	return float(features.std(axis=0, ddof=0).mean())


def _effective_rank(features: np.ndarray) -> float:
	centered = features - features.mean(axis=0, keepdims=True)
	covariance = centered.T @ centered / features.shape[0]
	eigenvalues = np.linalg.eigvalsh(covariance)
	maximum = float(np.max(np.abs(eigenvalues), initial=0.0))
	tolerance = max(1.0, maximum) * np.finfo(np.float64).eps * 1_000.0
	if float(eigenvalues.min(initial=0.0)) < -tolerance:
		raise FloatingPointError(
			'feature covariance has a materially negative eigenvalue'
		)
	eigenvalues = np.maximum(eigenvalues, 0.0)
	total = float(eigenvalues.sum())
	if total == 0.0:
		return 0.0
	probabilities = eigenvalues / total
	positive = probabilities[probabilities > 0.0]
	return float(np.exp(-np.sum(positive * np.log(positive))))


def _finite_metric(value: float, label: str) -> float:
	if not math.isfinite(value):
		raise FloatingPointError(f'{label} is non-finite')
	return value


def _validate_source(source: EmbeddingRepresentationSource) -> None:
	for label, value in (
		('source_id', source.source_id),
		('survey_id', source.survey_id),
	):
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f'{label} must be a non-empty string')
	if not isinstance(source.random_baseline, bool):
		raise TypeError('random_baseline must be a boolean')


def _source_paths(source: EmbeddingRepresentationSource) -> dict[str, Path]:
	return {
		'checkpoint': source.checkpoint_path,
		'embeddings': source.embeddings_path,
		'valid tokens': source.valid_tokens_path,
		'embedding metadata': source.metadata_path,
	}


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as exc:
		raise ValueError(f'{label} is not valid JSON: {path}') from exc
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a JSON object: {path}')
	return cast('Mapping[str, object]', value)


def _validate_embedding_metadata(
	metadata: Mapping[str, object],
	*,
	source: EmbeddingRepresentationSource,
	checkpoint_sha256: str,
	expected_token_grid_shape: tuple[int, ...],
	expected_embedding_dim: int,
) -> None:
	if metadata.get('survey_id') != source.survey_id:
		raise ValueError('embedding metadata survey_id differs from the source')
	if metadata.get('checkpoint_sha256') != checkpoint_sha256:
		raise ValueError(
			'embedding metadata checkpoint SHA-256 differs from checkpoint'
		)
	checkpoint_path = metadata.get('checkpoint_path')
	if not isinstance(checkpoint_path, str) or (
		Path(checkpoint_path).resolve() != source.checkpoint_path.resolve()
	):
		raise ValueError('embedding metadata checkpoint path differs from the source')
	if metadata.get('token_grid_shape') != list(expected_token_grid_shape):
		raise ValueError(
			'embedding metadata token grid differs from the fixed contract'
		)
	if metadata.get('output_dtype') != 'float16':
		raise ValueError('embedding metadata output_dtype must be float16')
	geometry = metadata.get('model_geometry')
	if not isinstance(geometry, Mapping):
		raise TypeError('embedding metadata model_geometry must be a mapping')
	if geometry.get('encoder_dim') != expected_embedding_dim:
		raise ValueError('embedding metadata encoder dimension differs from contract')


def _validate_checkpoint_identity(  # noqa: C901, PLR0913
	payload: Mapping[str, Any],
	*,
	checkpoint_config: Mapping[str, object],
	random_baseline: bool,
	expected_candidate_epoch: int,
	expected_candidate_global_step: int,
	expected_random_seed: int,
) -> dict[str, object]:
	epoch = _checkpoint_integer(payload, 'epoch')
	global_step = _checkpoint_integer(payload, 'global_step')
	if random_baseline:
		if not is_random_encoder_checkpoint(payload):
			raise ValueError(
				'Random diagnostic source is not a random encoder checkpoint'
			)
		metadata = payload.get('metadata')
		if not isinstance(metadata, Mapping):
			raise TypeError('Random checkpoint metadata must be a mapping')
		if metadata.get('seed') != expected_random_seed:
			raise ValueError('Random checkpoint seed differs from the fixed baseline')
		if epoch != 0 or global_step != 0:
			raise ValueError('Random checkpoint must be the untrained epoch-0 endpoint')
		return {
			'epoch': epoch,
			'global_step': global_step,
			'random_encoder_baseline': True,
			'random_seed': expected_random_seed,
		}

	if is_random_encoder_checkpoint(payload):
		raise ValueError('candidate diagnostic source must not be a random checkpoint')
	if (
		epoch != expected_candidate_epoch
		or global_step != expected_candidate_global_step
	):
		raise ValueError(
			'candidate checkpoint is not the complete 10-epoch/6250-step endpoint: '
			f'epoch={epoch}, global_step={global_step}'
		)
	training_state = payload.get('training_state')
	if not isinstance(training_state, Mapping) or (
		training_state.get('completed_epoch') is not True
	):
		raise ValueError('candidate checkpoint is not a completed epoch endpoint')
	method = resolve_barlow_twins_pretraining_method(checkpoint_config)
	if method != LOCAL_BARLOW_TWINS_PRETRAINING_METHOD:
		raise ValueError('candidate checkpoint must use Local Barlow Twins')
	augmentations = checkpoint_config.get('augmentations')
	if not isinstance(augmentations, Mapping) or (
		augmentations.get('policy') != OVERLAPPING_SUBCROP_XY_AUGMENTATION_POLICY
	):
		raise ValueError('candidate checkpoint must use overlapping-subcrop views')
	return {
		'epoch': epoch,
		'global_step': global_step,
		'pretraining_method': method,
		'augmentation_policy': OVERLAPPING_SUBCROP_XY_AUGMENTATION_POLICY,
		'bare_encoder_state': True,
	}


def _checkpoint_integer(payload: Mapping[str, Any], key: str) -> int:
	value = payload.get(key)
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'checkpoint {key} must be an integer')
	return value


def _validate_arrays(
	embeddings: np.ndarray,
	valid_tokens: np.ndarray,
	*,
	expected_token_grid_shape: tuple[int, ...],
	expected_embedding_dim: int,
) -> None:
	expected_embedding_shape = (*expected_token_grid_shape, expected_embedding_dim)
	if tuple(embeddings.shape) != expected_embedding_shape:
		raise ValueError(
			'embedding array shape differs from the fixed contract: '
			f'expected={expected_embedding_shape!r}, got={tuple(embeddings.shape)!r}'
		)
	if embeddings.dtype != np.dtype(np.float16):
		raise TypeError(
			f'embedding array dtype must be float16; got {embeddings.dtype!s}'
		)
	if tuple(valid_tokens.shape) != expected_token_grid_shape:
		raise ValueError('valid-token array shape differs from the fixed contract')
	if valid_tokens.dtype != np.dtype(bool):
		raise TypeError(
			f'valid-token array dtype must be bool; got {valid_tokens.dtype!s}'
		)


def _indices_sha256(indices: np.ndarray) -> str:
	canonical = np.asarray(indices, dtype='<i8')
	return hashlib.sha256(canonical.tobytes(order='C')).hexdigest()


def _positive_shape(value: Sequence[int], label: str) -> tuple[int, ...]:
	if isinstance(value, str | bytes) or not isinstance(value, Sequence):
		raise TypeError(f'{label} must be a sequence of integers')
	shape = tuple(value)
	if not shape or any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in shape
	):
		raise ValueError(f'{label} must contain positive integers')
	return shape


def _validate_sha256(value: str, label: str) -> None:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 hex digest')


__all__ = [
	'DEFAULT_REPRESENTATION_LAYER_NORM_EPS',
	'DEFAULT_REPRESENTATION_SAMPLE_SIZE',
	'EFFECTIVE_RANK_METHOD',
	'REPRESENTATION_DIAGNOSTIC_ARTIFACT_TYPE',
	'REPRESENTATION_DIAGNOSTIC_SCHEMA_VERSION',
	'REPRESENTATION_METRIC_KEYS',
	'SYSTEMATIC_SAMPLING_METHOD',
	'EmbeddingRepresentationSource',
	'build_embedding_representation_diagnostics',
	'representation_metrics',
	'systematic_valid_token_indices',
	'write_embedding_representation_diagnostics',
]
