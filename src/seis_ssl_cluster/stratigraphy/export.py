"""Export existing stratigraphic HMM labels as pseudo-target artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetPaths,
	pseudo_target_paths,
	validate_pseudo_target_arrays,
	write_pseudo_target,
)

if TYPE_CHECKING:
	from collections.abc import Mapping


LABEL_SUFFIX = '.cluster_labels_token.npy'
METADATA_SUFFIX = '.cluster_label_metadata.json'


@dataclass(frozen=True)
class ExportedPseudoTargetResult:
	"""Paths and counts for one exported pseudo-target artifact."""

	survey_id: str
	labels_path: Path
	confidence_path: Path
	valid_tokens_path: Path
	metadata_path: Path
	valid_token_count: int


@dataclass(frozen=True)
class _PreparedPseudoTargetExport:
	survey_id: str
	labels: np.ndarray
	confidence: np.ndarray
	valid_tokens: np.ndarray
	output_paths: StratPseudoTargetPaths
	source_metadata: Mapping[str, object]
	valid_token_count: int


def export_hmm_cluster_labels_as_pseudo_targets(
	*,
	clustering_output_dir: str | Path,
	pseudo_target_root: str | Path,
	k: int,
	confidence: float = 1.0,
	overwrite: bool = False,
) -> list[ExportedPseudoTargetResult]:
	"""Export existing stratigraphic HMM cluster labels as pseudo-targets."""
	prepared = _prepare_exports(
		clustering_output_dir=clustering_output_dir,
		pseudo_target_root=pseudo_target_root,
		k=k,
		confidence=confidence,
		overwrite=overwrite,
	)
	results: list[ExportedPseudoTargetResult] = []
	for item in prepared:
		paths = write_pseudo_target(
			pseudo_target_root,
			k=k,
			survey_id=item.survey_id,
			labels=item.labels,
			confidence=item.confidence,
			valid_tokens=item.valid_tokens,
			metadata=item.source_metadata,
		)
		results.append(
			ExportedPseudoTargetResult(
				survey_id=item.survey_id,
				labels_path=paths.labels,
				confidence_path=paths.confidence,
				valid_tokens_path=paths.valid_tokens,
				metadata_path=paths.metadata,
				valid_token_count=item.valid_token_count,
			),
		)
	return results


def prepare_hmm_cluster_label_pseudo_target_exports(
	*,
	clustering_output_dir: str | Path,
	pseudo_target_root: str | Path,
	k: int,
	confidence: float = 1.0,
	overwrite: bool = False,
) -> list[ExportedPseudoTargetResult]:
	"""Validate an export run and return output paths without writing files."""
	prepared = _prepare_exports(
		clustering_output_dir=clustering_output_dir,
		pseudo_target_root=pseudo_target_root,
		k=k,
		confidence=confidence,
		overwrite=overwrite,
	)
	return [
		ExportedPseudoTargetResult(
			survey_id=item.survey_id,
			labels_path=item.output_paths.labels,
			confidence_path=item.output_paths.confidence,
			valid_tokens_path=item.output_paths.valid_tokens,
			metadata_path=item.output_paths.metadata,
			valid_token_count=item.valid_token_count,
		)
		for item in prepared
	]


def _prepare_exports(
	*,
	clustering_output_dir: str | Path,
	pseudo_target_root: str | Path,
	k: int,
	confidence: float,
	overwrite: bool,
) -> list[_PreparedPseudoTargetExport]:
	confidence_value = _validate_confidence(confidence)
	label_paths = _discover_label_paths(clustering_output_dir, k=k)
	prepared = [
		_prepare_export(
			label_path=label_path,
			clustering_output_dir=clustering_output_dir,
			pseudo_target_root=pseudo_target_root,
			k=k,
			confidence=confidence_value,
		)
		for label_path in label_paths
	]
	_validate_output_paths_available(prepared, overwrite=overwrite)
	return prepared


def _discover_label_paths(
	clustering_output_dir: str | Path,
	*,
	k: int,
) -> list[Path]:
	label_dir = Path(clustering_output_dir) / 'labels' / f'k{k}'
	if not label_dir.is_dir():
		msg = f'HMM label directory must exist for k={k}: {label_dir}'
		raise FileNotFoundError(msg)
	label_paths = sorted(label_dir.glob(f'*{LABEL_SUFFIX}'))
	if not label_paths:
		msg = f'no HMM cluster label files found for k={k}: {label_dir}'
		raise ValueError(msg)
	return label_paths


def _prepare_export(
	*,
	label_path: Path,
	clustering_output_dir: str | Path,
	pseudo_target_root: str | Path,
	k: int,
	confidence: float,
) -> _PreparedPseudoTargetExport:
	survey_id = label_path.name.removesuffix(LABEL_SUFFIX)
	labels = np.asarray(np.load(label_path))
	valid_tokens = labels >= 0
	confidence_array = np.zeros(labels.shape, dtype=np.float32)
	confidence_array[valid_tokens] = np.float32(confidence)
	validate_pseudo_target_arrays(
		labels,
		confidence_array,
		valid_tokens,
		k=k,
		survey_id=survey_id,
	)
	return _PreparedPseudoTargetExport(
		survey_id=survey_id,
		labels=labels,
		confidence=confidence_array,
		valid_tokens=valid_tokens,
		output_paths=pseudo_target_paths(pseudo_target_root, k=k, survey_id=survey_id),
		source_metadata=_source_metadata(
			clustering_output_dir=clustering_output_dir,
			label_path=label_path,
			survey_id=survey_id,
			confidence=confidence,
		),
		valid_token_count=int(np.count_nonzero(valid_tokens)),
	)


def _source_metadata(
	*,
	clustering_output_dir: str | Path,
	label_path: Path,
	survey_id: str,
	confidence: float,
) -> dict[str, object]:
	metadata_path = label_path.with_name(f'{survey_id}{METADATA_SUFFIX}')
	payload: dict[str, object] = {
		'export_confidence': confidence,
		'source_clustering_output_dir': str(Path(clustering_output_dir)),
		'source_label_path': str(label_path),
	}
	if metadata_path.is_file():
		source_metadata = _load_source_metadata(metadata_path)
		payload.update(
			{
				'source_metadata_path': str(metadata_path),
				'source_metadata_sha256': _sha256_file(metadata_path),
			},
		)
		if 'method' in source_metadata:
			payload['source_method'] = source_metadata['method']
	return payload


def _load_source_metadata(path: Path) -> Mapping[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		msg = f'source cluster metadata must be valid JSON: {path}'
		raise ValueError(msg) from exc
	if not isinstance(payload, dict):
		msg = f'source cluster metadata must be a JSON object: {path}'
		raise TypeError(msg)
	return payload


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _validate_output_paths_available(
	prepared: list[_PreparedPseudoTargetExport],
	*,
	overwrite: bool,
) -> None:
	if overwrite:
		return
	existing = [
		path
		for item in prepared
		for path in (
			item.output_paths.labels,
			item.output_paths.confidence,
			item.output_paths.valid_tokens,
			item.output_paths.boundary_weight,
			item.output_paths.metadata,
		)
		if path.exists()
	]
	if existing:
		msg = (
			'pseudo-target outputs already exist; pass overwrite=True to replace: '
			+ ', '.join(str(path) for path in existing)
		)
		raise FileExistsError(msg)


def _validate_confidence(confidence: float) -> float:
	value = float(confidence)
	if not np.isfinite(value) or value < 0.0 or value > 1.0:
		msg = f'confidence must be finite and in [0, 1]; got {confidence!r}'
		raise ValueError(msg)
	return value
