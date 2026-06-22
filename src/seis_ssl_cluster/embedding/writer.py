"""Output path, metadata, and memmap helpers for embedding extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingOutputPaths:
	"""Output files for one survey embedding extraction."""

	embeddings: Path
	valid_tokens: Path
	metadata: Path
	embeddings_tmp: Path
	valid_tokens_tmp: Path
	sum_tmp: Path
	count_tmp: Path


def output_paths(output_dir: str | Path, survey_id: str) -> EmbeddingOutputPaths:
	"""Return deterministic output paths for one survey."""
	root = Path(output_dir)
	return EmbeddingOutputPaths(
		embeddings=root / f'{survey_id}.embeddings.npy',
		valid_tokens=root / f'{survey_id}.valid_tokens.npy',
		metadata=root / f'{survey_id}.embedding_metadata.json',
		embeddings_tmp=root / f'.{survey_id}.embeddings.tmp.npy',
		valid_tokens_tmp=root / f'.{survey_id}.valid_tokens.tmp.npy',
		sum_tmp=root / f'.{survey_id}.embedding_sum.float32.npy',
		count_tmp=root / f'.{survey_id}.embedding_count.uint32.npy',
	)


def prepare_outputs(
	paths: EmbeddingOutputPaths,
	metadata: dict[str, object],
	*,
	skip_existing: bool,
) -> bool:
	"""Return true when matching existing outputs should be skipped."""
	paths.embeddings.parent.mkdir(parents=True, exist_ok=True)
	cleanup_temp_outputs(paths)
	complete_outputs = paths.embeddings.exists() and paths.valid_tokens.exists()
	existing_metadata = _read_metadata(paths.metadata)
	if existing_metadata == metadata and complete_outputs and skip_existing:
		return True
	if (
		complete_outputs
		and existing_metadata is not _METADATA_UNAVAILABLE
		and existing_metadata != metadata
	):
		msg = (
			'existing embedding output metadata does not match current settings: '
			f'{paths.metadata}'
		)
		raise ValueError(msg)
	_remove_final_outputs(paths)
	return False


def create_merge_memmaps(
	paths: EmbeddingOutputPaths,
	*,
	token_grid_shape_xyz: tuple[int, int, int],
	embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Create restartable temporary memmaps for embedding sums and counts."""
	paths.sum_tmp.parent.mkdir(parents=True, exist_ok=True)
	sum_array = np.lib.format.open_memmap(
		paths.sum_tmp,
		mode='w+',
		dtype=np.float32,
		shape=(*token_grid_shape_xyz, embedding_dim),
	)
	count_array = np.lib.format.open_memmap(
		paths.count_tmp,
		mode='w+',
		dtype=np.uint32,
		shape=token_grid_shape_xyz,
	)
	sum_array[...] = 0.0
	count_array[...] = 0
	return sum_array, count_array


def write_metadata(path: str | Path, metadata: dict[str, object]) -> None:
	"""Write deterministic extraction metadata."""
	metadata_path = Path(path)
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = _metadata_tmp_path(metadata_path)
	tmp_path.write_text(
		json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	tmp_path.replace(metadata_path)


def commit_staged_outputs(
	paths: EmbeddingOutputPaths,
	metadata: dict[str, object],
) -> None:
	"""Publish staged embedding arrays and matching metadata."""
	for path in (paths.embeddings_tmp, paths.valid_tokens_tmp):
		if not path.is_file():
			msg = f'missing staged embedding output: {path}'
			raise FileNotFoundError(msg)
	write_metadata(paths.metadata, metadata)
	paths.embeddings_tmp.replace(paths.embeddings)
	paths.valid_tokens_tmp.replace(paths.valid_tokens)


def metadata_matches(path: str | Path, metadata: dict[str, object]) -> bool:
	"""Return true when an existing metadata JSON matches exactly."""
	return _read_metadata(Path(path)) == metadata


def cleanup_temp_outputs(paths: EmbeddingOutputPaths) -> None:
	"""Remove temporary merge arrays left after a successful run."""
	for path in (
		paths.embeddings_tmp,
		paths.valid_tokens_tmp,
		_metadata_tmp_path(paths.metadata),
		paths.sum_tmp,
		paths.count_tmp,
	):
		if path.exists():
			path.unlink()


_METADATA_UNAVAILABLE = object()


def _read_metadata(path: Path) -> object:
	if not path.is_file():
		return _METADATA_UNAVAILABLE
	try:
		return json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError:
		return _METADATA_UNAVAILABLE


def _remove_final_outputs(paths: EmbeddingOutputPaths) -> None:
	for path in (paths.embeddings, paths.valid_tokens, paths.metadata):
		if path.exists():
			path.unlink()


def _metadata_tmp_path(path: Path) -> Path:
	return path.with_name(f'.{path.name}.tmp')


def file_sha256(path: str | Path) -> str:
	"""Return the SHA-256 hex digest for a file."""
	digest = hashlib.sha256()
	with Path(path).open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


__all__ = [
	'EmbeddingOutputPaths',
	'cleanup_temp_outputs',
	'commit_staged_outputs',
	'create_merge_memmaps',
	'file_sha256',
	'metadata_matches',
	'output_paths',
	'prepare_outputs',
	'write_metadata',
]
