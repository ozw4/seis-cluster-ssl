"""Summary artifacts for seismic cluster labels."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_ssl_cluster.clustering.reconstruct import resolve_volume_shape_xyz
from seis_ssl_cluster.clustering.writer import write_json
from seis_ssl_cluster.data.normalization import (
	load_normalization_stats,
	normalize_amplitude,
)

_TOKEN_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class ClusterSummaryInput:
	"""Inputs for summarizing one survey's cluster labels."""

	survey_id: str
	labels_path: Path
	metadata_path: Path | None = None
	embeddings_path: Path | None = None


@dataclass(frozen=True)
class ClusterSummaryArtifacts:
	"""Summary artifact paths."""

	csv_path: Path
	png_path: Path
	json_path: Path


def write_cluster_summaries(
	inputs: Sequence[ClusterSummaryInput],
	*,
	k: int,
	output_dir: str | Path,
	include_amplitude_norm: bool = False,
	selected_survey_ids: Sequence[str] | None = None,
) -> ClusterSummaryArtifacts:
	"""Write CSV, PNG, and JSON summaries for a k-cluster label set."""
	if k <= 0:
		msg = f'k must be positive; got {k!r}'
		raise ValueError(msg)
	if not inputs:
		msg = 'at least one summary input is required'
		raise ValueError(msg)
	root = Path(output_dir)
	root.mkdir(parents=True, exist_ok=True)
	accumulator = _new_accumulator(k)
	survey_hits = np.zeros((k, len(inputs)), dtype=bool)
	for survey_index, item in enumerate(inputs):
		_summarize_one(
			item,
			k=k,
			accumulator=accumulator,
			survey_hits=survey_hits[:, survey_index],
			include_amplitude_norm=include_amplitude_norm,
		)

	total_valid = int(accumulator['counts'].sum())
	rows = _summary_rows(
		accumulator,
		survey_hits=survey_hits,
		total_valid=total_valid,
		survey_count=len(inputs),
	)
	csv_path = root / 'cluster_size_histogram.csv'
	png_path = root / 'cluster_size_histogram.png'
	json_path = root / 'cluster_summary.json'
	_write_histogram_csv(csv_path, rows)
	_write_histogram_png(png_path, accumulator['counts'])
	write_json(
		json_path,
		{
			'k': int(k),
			'total_valid_token_count': total_valid,
			'total_invalid_token_count': int(accumulator['invalid_count']),
			'survey_count': len(inputs),
			'survey_ids': [item.survey_id for item in inputs],
			'selected_survey_ids': (
				list(selected_survey_ids)
				if selected_survey_ids is not None
				else None
			),
			'clusters': rows,
		},
	)
	return ClusterSummaryArtifacts(
		csv_path=csv_path,
		png_path=png_path,
		json_path=json_path,
	)


def _new_accumulator(k: int) -> dict[str, np.ndarray | int]:
	return {
		'counts': np.zeros(k, dtype=np.int64),
		'invalid_count': 0,
		'embedding_norm_sum': np.zeros(k, dtype=np.float64),
		'embedding_norm_count': np.zeros(k, dtype=np.int64),
		'amplitude_sum': np.zeros(k, dtype=np.float64),
		'amplitude_sumsq': np.zeros(k, dtype=np.float64),
		'amplitude_count': np.zeros(k, dtype=np.int64),
	}


def _summarize_one(
	item: ClusterSummaryInput,
	*,
	k: int,
	accumulator: dict[str, np.ndarray | int],
	survey_hits: np.ndarray,
	include_amplitude_norm: bool,
) -> None:
	labels = np.load(item.labels_path, mmap_mode='r')
	if labels.ndim != 3:
		msg = f'labels must be 3D: {item.labels_path}'
		raise ValueError(msg)
	embeddings = _load_embeddings(item.embeddings_path, labels.shape)
	counts = _add_token_counts_and_embeddings(
		labels,
		embeddings=embeddings,
		accumulator=accumulator,
		k=k,
	)
	accumulator['counts'] += counts
	accumulator['invalid_count'] = int(accumulator['invalid_count']) + int(
		labels.size - int(counts.sum()),
	)
	survey_hits[...] = counts > 0
	if include_amplitude_norm:
		_add_amplitude_norms(item, labels, accumulator)


def _load_embeddings(
	embeddings_path: Path | None,
	labels_shape: tuple[int, ...],
) -> np.ndarray | None:
	if embeddings_path is None or not embeddings_path.is_file():
		return None
	embeddings = np.load(embeddings_path, mmap_mode='r')
	if embeddings.ndim != 4 or embeddings.shape[:3] != labels_shape:
		msg = (
			'embeddings token grid must match labels; '
			f'got {embeddings.shape[:3]!r} and {labels_shape!r}'
		)
		raise ValueError(msg)
	return embeddings


def _add_token_counts_and_embeddings(
	labels: np.ndarray,
	*,
	embeddings: np.ndarray | None,
	accumulator: dict[str, np.ndarray | int],
	k: int,
) -> np.ndarray:
	counts = np.zeros(k, dtype=np.int64)
	embedding_flat = (
		embeddings.reshape((-1, embeddings.shape[-1]))
		if embeddings is not None
		else None
	)
	for start, labels_chunk in _iter_label_chunks(labels):
		valid_mask = labels_chunk >= 0
		if not bool(np.any(valid_mask)):
			continue
		valid_labels = np.asarray(labels_chunk[valid_mask], dtype=np.int64)
		if np.any(valid_labels >= k):
			msg = f'label value exceeds k={k}'
			raise ValueError(msg)
		counts += np.bincount(valid_labels, minlength=k)
		if embedding_flat is None:
			continue
		embedding_chunk = np.asarray(embedding_flat[start : start + labels_chunk.size])
		norms = np.linalg.norm(np.asarray(embedding_chunk[valid_mask]), axis=1)
		accumulator['embedding_norm_sum'] += np.bincount(
			valid_labels,
			weights=norms,
			minlength=k,
		)
		accumulator['embedding_norm_count'] += np.bincount(
			valid_labels,
			minlength=k,
		)
	return counts


def _add_amplitude_norms(
	item: ClusterSummaryInput,
	labels: np.ndarray,
	accumulator: dict[str, np.ndarray | int],
) -> None:
	metadata = _load_metadata(item.metadata_path)
	source_path = metadata.get('source_amplitude_path')
	stats_path = metadata.get('normalization_stats_path')
	if not isinstance(source_path, str) or not isinstance(stats_path, str):
		return
	if not Path(source_path).is_file() or not Path(stats_path).is_file():
		return
	patch = _xyz(metadata.get('patch_size', (1, 1, 1)))
	volume = np.load(source_path, mmap_mode='r')
	shape = resolve_volume_shape_xyz(metadata, labels.shape, patch)
	stats = load_normalization_stats(stats_path)
	for chunk_start, labels_chunk in _iter_label_chunks(labels):
		valid_offsets = np.flatnonzero(labels_chunk >= 0)
		if valid_offsets.size == 0:
			continue
		for offset in valid_offsets:
			index = int(chunk_start + int(offset))
			label = int(labels_chunk[offset])
			token_xyz = np.unravel_index(index, labels.shape)
			start = tuple(
				axis * patch_axis
				for axis, patch_axis in zip(token_xyz, patch, strict=True)
			)
			stop = tuple(
				min(axis_start + patch_axis, shape_axis)
				for axis_start, patch_axis, shape_axis in zip(
					start,
					patch,
					shape,
					strict=True,
				)
			)
			patch_values = np.asarray(
				volume[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]],
			)
			if patch_values.size == 0:
				continue
			normalized = normalize_amplitude(patch_values, stats).astype(
				np.float64,
				copy=False,
			)
			values = normalized[np.isfinite(normalized)]
			if values.size == 0:
				continue
			accumulator['amplitude_sum'][label] += float(values.sum())
			accumulator['amplitude_sumsq'][label] += float(np.dot(values, values))
			accumulator['amplitude_count'][label] += int(values.size)


def _iter_label_chunks(labels: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
	flat_labels = labels.reshape(-1)
	for start in range(0, flat_labels.size, _TOKEN_CHUNK_SIZE):
		stop = min(start + _TOKEN_CHUNK_SIZE, flat_labels.size)
		yield start, np.asarray(flat_labels[start:stop])


def _summary_rows(
	accumulator: dict[str, np.ndarray | int],
	*,
	survey_hits: np.ndarray,
	total_valid: int,
	survey_count: int,
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	counts = accumulator['counts']
	for label, count in enumerate(counts):
		embedding_count = int(accumulator['embedding_norm_count'][label])
		amplitude_count = int(accumulator['amplitude_count'][label])
		coverage_count = int(np.count_nonzero(survey_hits[label]))
		rows.append(
			{
				'cluster': int(label),
				'token_count': int(count),
				'valid_fraction': (
					float(count / total_valid) if total_valid else 0.0
				),
				'mean_amplitude_norm': _mean_or_none(
					accumulator['amplitude_sum'][label],
					amplitude_count,
				),
				'std_amplitude_norm': _std_or_none(
					accumulator['amplitude_sum'][label],
					accumulator['amplitude_sumsq'][label],
					amplitude_count,
				),
				'mean_embedding_norm': _mean_or_none(
					accumulator['embedding_norm_sum'][label],
					embedding_count,
				),
				'survey_coverage': {
					'count': coverage_count,
					'fraction': float(coverage_count / survey_count),
				},
			},
		)
	return rows


def _write_histogram_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(
			file_obj,
			fieldnames=(
				'cluster',
				'token_count',
				'valid_fraction',
				'mean_amplitude_norm',
				'std_amplitude_norm',
				'mean_embedding_norm',
				'survey_coverage_count',
				'survey_coverage_fraction',
			),
		)
		writer.writeheader()
		for row in rows:
			survey_coverage = row['survey_coverage']
			if not isinstance(survey_coverage, Mapping):
				msg = f'expected survey_coverage mapping; got {survey_coverage!r}'
				raise TypeError(msg)
			writer.writerow(
				{
					'cluster': row['cluster'],
					'token_count': row['token_count'],
					'valid_fraction': row['valid_fraction'],
					'mean_amplitude_norm': row['mean_amplitude_norm'],
					'std_amplitude_norm': row['std_amplitude_norm'],
					'mean_embedding_norm': row['mean_embedding_norm'],
					'survey_coverage_count': survey_coverage['count'],
					'survey_coverage_fraction': survey_coverage['fraction'],
				},
			)


def _write_histogram_png(path: Path, counts: np.ndarray) -> None:
	plt = __import__('matplotlib.pyplot', fromlist=['pyplot'])
	path.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=160)
	ax.bar(np.arange(counts.size), counts.astype(np.int64), color='#4c78a8')
	ax.set_xlabel('cluster')
	ax.set_ylabel('valid token count')
	ax.set_title('Cluster size histogram')
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)


def _load_metadata(path: Path | None) -> dict[str, object]:
	if path is None or not path.is_file():
		return {}
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		return {}
	embedding_input = payload.get('embedding_input')
	if isinstance(embedding_input, dict):
		nested_path = embedding_input.get('metadata_path')
		if isinstance(nested_path, str) and Path(nested_path).is_file():
			nested = json.loads(Path(nested_path).read_text(encoding='utf-8'))
			if isinstance(nested, dict):
				return {**nested, **payload}
	return payload


def _xyz(value: object) -> tuple[int, int, int]:
	if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
		msg = f'expected XYZ sequence; got {value!r}'
		raise TypeError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _mean_or_none(total: float, count: int) -> float | None:
	if count == 0:
		return None
	return float(total / count)


def _std_or_none(total: float, total_squares: float, count: int) -> float | None:
	if count == 0:
		return None
	variance = max(float(total_squares / count) - float(total / count) ** 2, 0.0)
	return float(np.sqrt(variance))


__all__ = [
	'ClusterSummaryArtifacts',
	'ClusterSummaryInput',
	'write_cluster_summaries',
]
