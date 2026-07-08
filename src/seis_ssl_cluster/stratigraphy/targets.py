"""HMM pseudo-target artifact paths, validation, and I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = 'strat_hmm_pseudo_target'


@dataclass(frozen=True)
class StratPseudoTargetPaths:
	"""Output files for one survey's HMM pseudo-target artifact."""

	labels: Path
	confidence: Path
	valid_tokens: Path
	metadata: Path


@dataclass(frozen=True)
class StratPseudoTargetInput:
	"""Input artifact paths for one survey's HMM pseudo-target grid."""

	survey_id: str
	k: int
	labels_path: Path
	confidence_path: Path
	valid_tokens_path: Path
	metadata_path: Path


@dataclass(frozen=True)
class StratPseudoTargetArrays:
	"""Loaded arrays for one survey's HMM pseudo-target grid."""

	labels: np.ndarray
	confidence: np.ndarray
	valid_tokens: np.ndarray


def pseudo_target_paths(
	root: str | Path,
	*,
	k: int,
	survey_id: str,
) -> StratPseudoTargetPaths:
	"""Return deterministic pseudo-target paths for one survey and k value."""
	_validate_k(k)
	if not survey_id:
		msg = 'survey_id must be non-empty'
		raise ValueError(msg)
	output_dir = Path(root) / f'k{k}'
	return StratPseudoTargetPaths(
		labels=output_dir / f'{survey_id}.hmm_labels_token.npy',
		confidence=output_dir / f'{survey_id}.hmm_confidence_token.npy',
		valid_tokens=output_dir / f'{survey_id}.valid_tokens.npy',
		metadata=output_dir / f'{survey_id}.pseudo_target_metadata.json',
	)


def write_pseudo_target(  # noqa: PLR0913
	root: str | Path,
	*,
	k: int,
	survey_id: str,
	labels: np.ndarray,
	confidence: np.ndarray,
	valid_tokens: np.ndarray,
	metadata: Mapping[str, object] | None = None,
) -> StratPseudoTargetPaths:
	"""Validate and write one survey's HMM pseudo-target artifact."""
	validate_pseudo_target_arrays(
		labels,
		confidence,
		valid_tokens,
		k=k,
		survey_id=survey_id,
	)
	paths = pseudo_target_paths(root, k=k, survey_id=survey_id)
	paths.labels.parent.mkdir(parents=True, exist_ok=True)
	np.save(paths.labels, np.asarray(labels, dtype=np.int32))
	np.save(paths.confidence, np.asarray(confidence, dtype=np.float32))
	np.save(paths.valid_tokens, np.asarray(valid_tokens, dtype=np.bool_))
	_write_metadata(
		paths.metadata,
		_pseudo_target_metadata(
			labels=np.asarray(labels),
			valid_tokens=np.asarray(valid_tokens),
			k=k,
			survey_id=survey_id,
			source_metadata=metadata,
		),
	)
	return paths


def load_pseudo_target_arrays(
	input_or_paths: StratPseudoTargetInput | StratPseudoTargetPaths,
	*,
	mmap_mode: str | None = None,
) -> StratPseudoTargetArrays:
	"""Load labels, confidence, and valid-token arrays."""
	paths = _paths_from_input(input_or_paths)
	arrays = StratPseudoTargetArrays(
		labels=np.load(paths.labels, mmap_mode=mmap_mode),
		confidence=np.load(paths.confidence, mmap_mode=mmap_mode),
		valid_tokens=np.load(paths.valid_tokens, mmap_mode=mmap_mode),
	)
	if isinstance(input_or_paths, StratPseudoTargetInput):
		validate_pseudo_target_arrays(
			arrays.labels,
			arrays.confidence,
			arrays.valid_tokens,
			k=input_or_paths.k,
			survey_id=input_or_paths.survey_id,
		)
	return arrays


def discover_pseudo_target_inputs(
	root: str | Path,
	*,
	k: int,
) -> list[StratPseudoTargetInput]:
	"""Discover pseudo-target artifacts in deterministic survey order."""
	_validate_k(k)
	input_dir = Path(root) / f'k{k}'
	if not input_dir.is_dir():
		msg = f'pseudo-target input directory must exist for k={k}: {input_dir}'
		raise FileNotFoundError(msg)

	inputs: list[StratPseudoTargetInput] = []
	for labels_path in sorted(input_dir.glob('*.hmm_labels_token.npy')):
		survey_id = labels_path.name.removesuffix('.hmm_labels_token.npy')
		paths = pseudo_target_paths(root, k=k, survey_id=survey_id)
		item = StratPseudoTargetInput(
			survey_id=survey_id,
			k=int(k),
			labels_path=paths.labels,
			confidence_path=paths.confidence,
			valid_tokens_path=paths.valid_tokens,
			metadata_path=paths.metadata,
		)
		_validate_pseudo_target_input(item)
		inputs.append(item)
	if not inputs:
		msg = f'no pseudo-target inputs found for k={k}: {input_dir}'
		raise ValueError(msg)
	return inputs


def load_pseudo_target_metadata(
	input_or_paths: StratPseudoTargetInput | StratPseudoTargetPaths,
) -> dict[str, object]:
	"""Load one pseudo-target metadata JSON object."""
	metadata_path = _metadata_path_from_input(input_or_paths)
	survey_id = (
		input_or_paths.survey_id
		if isinstance(input_or_paths, StratPseudoTargetInput)
		else None
	)
	try:
		payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		msg = _message(
			'pseudo-target metadata must be valid JSON',
			survey_id=survey_id,
			detail=str(metadata_path),
		)
		raise ValueError(msg) from exc
	if not isinstance(payload, dict):
		msg = _message(
			'pseudo-target metadata must be a JSON object',
			survey_id=survey_id,
			detail=str(metadata_path),
		)
		raise TypeError(msg)
	return payload


def validate_pseudo_target_arrays(
	labels: np.ndarray,
	confidence: np.ndarray,
	valid_tokens: np.ndarray,
	*,
	k: int,
	survey_id: str | None = None,
) -> None:
	"""Validate pseudo-target arrays against the milestone-1 contract."""
	_validate_k(k, survey_id=survey_id)
	labels_array = np.asarray(labels)
	confidence_array = np.asarray(confidence)
	valid_array = np.asarray(valid_tokens)

	_validate_array_ndim(labels_array, name='labels', survey_id=survey_id)
	_validate_array_ndim(confidence_array, name='confidence', survey_id=survey_id)
	_validate_array_ndim(valid_array, name='valid_tokens', survey_id=survey_id)
	_validate_matching_shapes(labels_array, confidence_array, valid_array, survey_id)
	_validate_array_dtypes(labels_array, confidence_array, valid_array, survey_id)
	_validate_confidence_values(confidence_array, survey_id)
	_validate_label_mask_invariants(
		labels_array,
		confidence_array,
		valid_array,
		k,
		survey_id,
	)


def _pseudo_target_metadata(
	*,
	labels: np.ndarray,
	valid_tokens: np.ndarray,
	k: int,
	survey_id: str,
	source_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
	valid_count = int(np.count_nonzero(valid_tokens))
	counts = np.bincount(labels[valid_tokens].astype(np.int64), minlength=k)
	payload: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'invalid_token_count': int(labels.size - valid_count),
		'k': int(k),
		'label_counts': {
			str(label): int(count)
			for label, count in enumerate(counts[:k])
		},
		'schema_version': SCHEMA_VERSION,
		'survey_id': survey_id,
		'token_grid_shape': [int(size) for size in labels.shape],
		'valid_token_count': valid_count,
	}
	if source_metadata is not None:
		payload['source'] = dict(source_metadata)
	return payload


def _write_metadata(path: Path, payload: Mapping[str, object]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _paths_from_input(
	input_or_paths: StratPseudoTargetInput | StratPseudoTargetPaths,
) -> StratPseudoTargetPaths:
	if isinstance(input_or_paths, StratPseudoTargetPaths):
		return input_or_paths
	return StratPseudoTargetPaths(
		labels=input_or_paths.labels_path,
		confidence=input_or_paths.confidence_path,
		valid_tokens=input_or_paths.valid_tokens_path,
		metadata=input_or_paths.metadata_path,
	)


def _metadata_path_from_input(
	input_or_paths: StratPseudoTargetInput | StratPseudoTargetPaths,
) -> Path:
	if isinstance(input_or_paths, StratPseudoTargetPaths):
		return input_or_paths.metadata
	return input_or_paths.metadata_path


def _validate_pseudo_target_input(item: StratPseudoTargetInput) -> None:
	missing = [
		path
		for path in (
			item.labels_path,
			item.confidence_path,
			item.valid_tokens_path,
			item.metadata_path,
		)
		if not path.is_file()
	]
	if missing:
		msg = _message(
			'missing pseudo-target artifacts',
			survey_id=item.survey_id,
			detail=', '.join(str(path) for path in missing),
		)
		raise FileNotFoundError(msg)


def _validate_k(k: int, *, survey_id: str | None = None) -> None:
	if k <= 0:
		msg = _message('k must be positive', survey_id=survey_id, detail=f'got {k!r}')
		raise ValueError(msg)


def _validate_array_ndim(
	array: np.ndarray,
	*,
	name: str,
	survey_id: str | None,
) -> None:
	if array.ndim != 3:
		msg = _message(
			f'{name} must be 3D',
			survey_id=survey_id,
			detail=f'got shape={array.shape!r}',
		)
		raise ValueError(msg)


def _validate_matching_shapes(
	labels: np.ndarray,
	confidence: np.ndarray,
	valid_tokens: np.ndarray,
	survey_id: str | None,
) -> None:
	if labels.shape != confidence.shape:
		msg = _message(
			'labels and confidence shapes must match',
			survey_id=survey_id,
			detail=f'got {labels.shape!r} and {confidence.shape!r}',
		)
		raise ValueError(msg)
	if labels.shape != valid_tokens.shape:
		msg = _message(
			'labels and valid_tokens shapes must match',
			survey_id=survey_id,
			detail=f'got {labels.shape!r} and {valid_tokens.shape!r}',
		)
		raise ValueError(msg)


def _validate_array_dtypes(
	labels: np.ndarray,
	confidence: np.ndarray,
	valid_tokens: np.ndarray,
	survey_id: str | None,
) -> None:
	if labels.dtype.kind not in {'i', 'u'}:
		msg = _message(
			'labels dtype must be integer',
			survey_id=survey_id,
			detail=f'got {labels.dtype}',
		)
		raise TypeError(msg)
	if confidence.dtype.kind not in {'f', 'i', 'u'}:
		msg = _message(
			'confidence dtype must be numeric',
			survey_id=survey_id,
			detail=f'got {confidence.dtype}',
		)
		raise TypeError(msg)
	if valid_tokens.dtype != np.bool_:
		msg = _message(
			'valid_tokens dtype must be bool',
			survey_id=survey_id,
			detail=f'got {valid_tokens.dtype}',
		)
		raise TypeError(msg)


def _validate_confidence_values(
	confidence: np.ndarray,
	survey_id: str | None,
) -> None:
	if not np.all(np.isfinite(confidence)):
		msg = _message('confidence must be finite', survey_id=survey_id)
		raise ValueError(msg)
	if np.any((confidence < 0.0) | (confidence > 1.0)):
		msg = _message(
			'confidence values must be in [0, 1]',
			survey_id=survey_id,
		)
		raise ValueError(msg)


def _validate_label_mask_invariants(
	labels: np.ndarray,
	confidence: np.ndarray,
	valid_tokens: np.ndarray,
	k: int,
	survey_id: str | None,
) -> None:
	invalid = ~valid_tokens
	if np.any(labels[invalid] != -1):
		msg = _message(
			'labels must be -1 where valid_tokens is false',
			survey_id=survey_id,
		)
		raise ValueError(msg)
	if np.any(labels[valid_tokens] < 0) or np.any(labels[valid_tokens] >= k):
		msg = _message(
			f'labels where valid_tokens is true must be in [0, {k})',
			survey_id=survey_id,
		)
		raise ValueError(msg)
	if np.any(confidence[invalid] != 0.0):
		msg = _message(
			'confidence must be 0.0 where valid_tokens is false',
			survey_id=survey_id,
		)
		raise ValueError(msg)


def _message(
	text: str,
	*,
	survey_id: str | None = None,
	detail: str | None = None,
) -> str:
	parts = [text]
	if survey_id is not None:
		parts.append(f'for {survey_id}')
	if detail is not None:
		parts.append(detail)
	return '; '.join(parts)


__all__ = [
	'StratPseudoTargetArrays',
	'StratPseudoTargetInput',
	'StratPseudoTargetPaths',
	'discover_pseudo_target_inputs',
	'load_pseudo_target_arrays',
	'load_pseudo_target_metadata',
	'pseudo_target_paths',
	'validate_pseudo_target_arrays',
	'write_pseudo_target',
]
