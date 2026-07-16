"""Validation and resolution for embedding extraction configs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypeVar

from seis_ssl_cluster.config.artifact_paths import (
	_validate_artifact_output_path,
	_validate_nopims_checkpoint_path,
	_validate_nopims_embedding_path,
)
from seis_ssl_cluster.config.base import _resolve_base
from seis_ssl_cluster.config.common import (
	_required_mapping,
	_validate_bool,
	_validate_fraction,
	_validate_non_empty_path,
	_validate_nonnegative_int,
	_validate_nonnegative_int_triplet,
	_validate_path,
	_validate_positive_int,
	_validate_positive_int_triplet,
)
from seis_ssl_cluster.config.schema import STAGE_EMBEDDING_EXTRACTION

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])

_CHECKPOINT_OWNED_EXTRACTION_SECTIONS = frozenset(
	{'data', 'model', 'masking', 'loss', 'train', 'zero_mask'},
)


def resolve_embedding_extraction_config(config: _T) -> Config:
	"""Validate and resolve raw config for embedding extraction."""
	_reject_checkpoint_owned_extraction_sections(config)
	resolved, paths = _resolve_base(
		config,
		STAGE_EMBEDDING_EXTRACTION,
		require_nopims_root=False,
	)
	manifests = _required_mapping(resolved, 'manifests')
	embeddings = _required_mapping(resolved, 'embeddings')
	_validate_non_empty_path(manifests, 'input', prefix='manifests')
	checkpoint = _validate_non_empty_path(
		embeddings,
		'checkpoint',
		prefix='embeddings',
	)
	_validate_nopims_checkpoint_path(
		checkpoint,
		'embeddings.checkpoint',
		artifact_root=paths.artifact_root,
	)
	output_dir = _validate_path(embeddings, 'output_dir', prefix='embeddings')
	_validate_artifact_output_path(
		output_dir,
		'embeddings.output_dir',
		artifact_root=paths.artifact_root,
		nopims_root=paths.nopims_root,
	)
	_validate_nopims_embedding_path(
		output_dir,
		'embeddings.output_dir',
		artifact_root=paths.artifact_root,
	)

	embedding = _required_mapping(resolved, 'embedding')
	window_size = _validate_positive_int_triplet(
		embedding,
		'window_size',
		prefix='embedding',
	)
	overlap = _validate_nonnegative_int_triplet(
		embedding,
		'overlap',
		prefix='embedding',
	)
	_validate_overlap_less_than_window(overlap, window_size)
	_validate_embedding_output_dtype(embedding)
	_validate_positive_int(embedding, 'batch_size', prefix='embedding')
	if 'prefetch_queue_depth' in embedding:
		_validate_nonnegative_int(
			embedding,
			'prefetch_queue_depth',
			prefix='embedding',
		)
	for key in ('amp', 'stage_timing'):
		if key in embedding:
			_validate_bool(embedding, key, prefix='embedding')
	amp_dtype = embedding.get('amp_dtype', 'auto')
	if amp_dtype not in {'auto', 'bfloat16', 'float16'}:
		msg = (
			'embedding.amp_dtype must be one of '
			f"['auto', 'bfloat16', 'float16']; got {amp_dtype!r}"
		)
		raise ValueError(msg)
	_validate_fraction(
		embedding,
		'min_token_valid_fraction',
		prefix='embedding',
	)
	_validate_preprocessing_cache(embedding)
	return resolved


def _validate_preprocessing_cache(embedding: Mapping[str, object]) -> None:
	value = embedding.get('preprocessing_cache')
	if value is None:
		return
	if not isinstance(value, Mapping):
		raise TypeError('embedding.preprocessing_cache must be a mapping')
	unexpected = sorted(
		set(value) - {'mode', 'chunk_size_x', 'reuse', 'cleanup', 'directory'},
	)
	if unexpected:
		msg = f'embedding.preprocessing_cache key(s) not allowed: {unexpected!r}'
		raise ValueError(msg)
	mode = value.get('mode', 'off')
	if mode not in {'off', 'memory', 'memmap'}:
		msg = (
			'embedding.preprocessing_cache.mode must be "off", "memory", or '
			f'"memmap"; got {mode!r}'
		)
		raise ValueError(msg)
	if 'chunk_size_x' in value:
		_validate_positive_int(
			value,
			'chunk_size_x',
			prefix='embedding.preprocessing_cache',
		)
	for key in ('reuse', 'cleanup'):
		if key in value:
			_validate_bool(value, key, prefix='embedding.preprocessing_cache')
	if 'directory' in value:
		_validate_non_empty_path(
			value,
			'directory',
			prefix='embedding.preprocessing_cache',
		)


def _reject_checkpoint_owned_extraction_sections(
	config: Mapping[str, object],
) -> None:
	stale = sorted(set(config) & _CHECKPOINT_OWNED_EXTRACTION_SECTIONS)
	if stale:
		msg = (
			'embedding extraction config must not include checkpoint-owned '
			f'section(s): {stale!r}'
		)
		raise ValueError(msg)


def _validate_overlap_less_than_window(
	overlap: Sequence[int],
	window_size: Sequence[int],
) -> None:
	if any(
		overlap_axis >= window_axis
		for overlap_axis, window_axis in zip(overlap, window_size, strict=True)
	):
		msg = (
			'embedding.overlap values must be less than embedding.window_size '
			f'values; got overlap={list(overlap)!r}, '
			f'window_size={list(window_size)!r}'
		)
		raise ValueError(msg)


def _validate_embedding_output_dtype(embedding: Mapping[str, object]) -> None:
	value = embedding.get('output_dtype')
	if not isinstance(value, str) or not value:
		msg = f'embedding.output_dtype must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	if value not in {'float16', 'float32'}:
		msg = (
			'embedding.output_dtype must be "float16" or "float32"; '
			f'got {value!r}'
		)
		raise ValueError(msg)


__all__ = ['resolve_embedding_extraction_config']
