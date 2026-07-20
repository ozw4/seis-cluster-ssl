"""Export existing stratigraphic HMM labels as pseudo-target artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.stratigraphy.boundary_weights import boundary_weight_tokens
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetPaths,
	discover_pseudo_target_inputs,
	load_pseudo_target_arrays,
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
	boundary_weight_path: Path
	metadata_path: Path
	valid_token_count: int


@dataclass(frozen=True)
class _PreparedPseudoTargetExport:
	survey_id: str
	labels: np.ndarray
	confidence: np.ndarray
	valid_tokens: np.ndarray
	boundary_weight: np.ndarray
	output_paths: StratPseudoTargetPaths
	source_metadata: Mapping[str, object]
	valid_token_count: int


def export_hmm_cluster_labels_as_pseudo_targets(  # noqa: PLR0913
	*,
	clustering_output_dir: str | Path,
	pseudo_target_root: str | Path,
	k: int,
	confidence: float = 1.0,
	boundary_alpha: float = 0.0,
	boundary_tau: float = 1.0,
	overwrite: bool = False,
	schema_version: int = 2,
	write_boundary_weight: bool = True,
) -> list[ExportedPseudoTargetResult]:
	"""Export existing stratigraphic HMM cluster labels as pseudo-targets."""
	prepared = _prepare_exports(
		clustering_output_dir=clustering_output_dir,
		pseudo_target_root=pseudo_target_root,
		k=k,
		confidence=confidence,
		boundary_alpha=boundary_alpha,
		boundary_tau=boundary_tau,
		overwrite=overwrite,
		schema_version=schema_version,
		write_boundary_weight=write_boundary_weight,
	)
	if schema_version == 1:
		return _export_schema_v1_atomically(
			prepared,
			pseudo_target_root=Path(pseudo_target_root),
			k=k,
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
			boundary_weight=item.boundary_weight,
			metadata=item.source_metadata,
			schema_version=schema_version,
			write_boundary_weight=write_boundary_weight,
		)
		results.append(
			ExportedPseudoTargetResult(
				survey_id=item.survey_id,
				labels_path=paths.labels,
				confidence_path=paths.confidence,
				valid_tokens_path=paths.valid_tokens,
				boundary_weight_path=paths.boundary_weight,
				metadata_path=paths.metadata,
				valid_token_count=item.valid_token_count,
			),
		)
	return results


def _export_schema_v1_atomically(
	prepared: list[_PreparedPseudoTargetExport],
	*,
	pseudo_target_root: Path,
	k: int,
	overwrite: bool,
) -> list[ExportedPseudoTargetResult]:
	"""Stage a complete schema-v1 K artifact before publishing its directory."""
	final_dir = pseudo_target_root / f'k{k}'
	if final_dir.exists() and not overwrite:
		raise FileExistsError(
			f'pseudo-target output directory already exists: {final_dir}'
		)
	pseudo_target_root.mkdir(parents=True, exist_ok=True)
	staging_root = Path(
		tempfile.mkdtemp(prefix=f'.k{k}.', dir=pseudo_target_root),
	)
	try:
		for item in prepared:
			write_pseudo_target(
				staging_root,
				k=k,
				survey_id=item.survey_id,
				labels=item.labels,
				confidence=item.confidence,
				valid_tokens=item.valid_tokens,
				boundary_weight=item.boundary_weight,
				metadata=item.source_metadata,
				schema_version=1,
				write_boundary_weight=False,
			)
		_validate_staged_schema_v1_export(staging_root, k=k, prepared=prepared)
		_publish_staged_directory(
			staging_root / f'k{k}', final_dir, overwrite=overwrite
		)
	except BaseException:
		shutil.rmtree(staging_root, ignore_errors=True)
		raise
	shutil.rmtree(staging_root, ignore_errors=True)
	return [_export_result(item) for item in prepared]


def _validate_staged_schema_v1_export(
	staging_root: Path,
	*,
	k: int,
	prepared: list[_PreparedPseudoTargetExport],
) -> None:
	"""Reject a staging directory unless every requested artifact round-trips."""
	inputs = discover_pseudo_target_inputs(staging_root, k=k)
	if [item.survey_id for item in inputs] != [item.survey_id for item in prepared]:
		raise ValueError('staged schema-v1 pseudo-target survey set mismatch')
	for item in inputs:
		load_pseudo_target_arrays(item)


def _publish_staged_directory(source: Path, target: Path, *, overwrite: bool) -> None:
	"""Atomically publish a new K directory; restore an overwrite on failure."""
	if not target.exists():
		source.replace(target)
		return
	if not overwrite:
		raise FileExistsError(
			f'pseudo-target output directory already exists: {target}'
		)
	backup = target.with_name(f'.{target.name}.previous')
	if backup.exists():
		raise FileExistsError(
			f'pseudo-target backup directory already exists: {backup}'
		)
	target.replace(backup)
	try:
		source.replace(target)
	except BaseException:
		backup.replace(target)
		raise
	shutil.rmtree(backup)


def _export_result(item: _PreparedPseudoTargetExport) -> ExportedPseudoTargetResult:
	return ExportedPseudoTargetResult(
		survey_id=item.survey_id,
		labels_path=item.output_paths.labels,
		confidence_path=item.output_paths.confidence,
		valid_tokens_path=item.output_paths.valid_tokens,
		boundary_weight_path=item.output_paths.boundary_weight,
		metadata_path=item.output_paths.metadata,
		valid_token_count=item.valid_token_count,
	)


def prepare_hmm_cluster_label_pseudo_target_exports(  # noqa: PLR0913
	*,
	clustering_output_dir: str | Path,
	pseudo_target_root: str | Path,
	k: int,
	confidence: float = 1.0,
	boundary_alpha: float = 0.0,
	boundary_tau: float = 1.0,
	overwrite: bool = False,
	schema_version: int = 2,
	write_boundary_weight: bool = True,
) -> list[ExportedPseudoTargetResult]:
	"""Validate an export run and return output paths without writing files."""
	prepared = _prepare_exports(
		clustering_output_dir=clustering_output_dir,
		pseudo_target_root=pseudo_target_root,
		k=k,
		confidence=confidence,
		boundary_alpha=boundary_alpha,
		boundary_tau=boundary_tau,
		overwrite=overwrite,
		schema_version=schema_version,
		write_boundary_weight=write_boundary_weight,
	)
	return [
		ExportedPseudoTargetResult(
			survey_id=item.survey_id,
			labels_path=item.output_paths.labels,
			confidence_path=item.output_paths.confidence,
			valid_tokens_path=item.output_paths.valid_tokens,
			boundary_weight_path=item.output_paths.boundary_weight,
			metadata_path=item.output_paths.metadata,
			valid_token_count=item.valid_token_count,
		)
		for item in prepared
	]


def _prepare_exports(  # noqa: PLR0913
	*,
	clustering_output_dir: str | Path,
	pseudo_target_root: str | Path,
	k: int,
	confidence: float,
	boundary_alpha: float,
	boundary_tau: float,
	overwrite: bool,
	schema_version: int,
	write_boundary_weight: bool,
) -> list[_PreparedPseudoTargetExport]:
	if schema_version not in {1, 2} or (schema_version == 1 and write_boundary_weight):
		msg = 'schema v1 HMM bootstrap exports must not write boundary weights'
		raise ValueError(msg)
	confidence_value = _validate_confidence(confidence)
	label_paths = _discover_label_paths(clustering_output_dir, k=k)
	prepared = [
		_prepare_export(
			label_path=label_path,
			clustering_output_dir=clustering_output_dir,
			pseudo_target_root=pseudo_target_root,
			k=k,
			confidence=confidence_value,
			boundary_alpha=boundary_alpha,
			boundary_tau=boundary_tau,
		)
		for label_path in label_paths
	]
	_validate_output_paths_available(
		prepared,
		overwrite=overwrite,
		write_boundary_weight=write_boundary_weight,
	)
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


def _prepare_export(  # noqa: PLR0913
	*,
	label_path: Path,
	clustering_output_dir: str | Path,
	pseudo_target_root: str | Path,
	k: int,
	confidence: float,
	boundary_alpha: float,
	boundary_tau: float,
) -> _PreparedPseudoTargetExport:
	survey_id = label_path.name.removesuffix(LABEL_SUFFIX)
	labels = np.asarray(np.load(label_path))
	valid_tokens = labels >= 0
	confidence_array = np.zeros(labels.shape, dtype=np.float32)
	confidence_array[valid_tokens] = np.float32(confidence)
	boundary_weight = boundary_weight_tokens(
		labels,
		valid_tokens,
		alpha=boundary_alpha,
		tau=boundary_tau,
	)
	validate_pseudo_target_arrays(
		labels,
		confidence_array,
		valid_tokens,
		boundary_weight=boundary_weight,
		k=k,
		survey_id=survey_id,
	)
	return _PreparedPseudoTargetExport(
		survey_id=survey_id,
		labels=labels,
		confidence=confidence_array,
		valid_tokens=valid_tokens,
		boundary_weight=boundary_weight,
		output_paths=pseudo_target_paths(pseudo_target_root, k=k, survey_id=survey_id),
		source_metadata=_source_metadata(
			clustering_output_dir=clustering_output_dir,
			label_path=label_path,
			survey_id=survey_id,
			confidence=confidence,
			boundary_alpha=float(boundary_alpha),
			boundary_tau=float(boundary_tau),
		),
		valid_token_count=int(np.count_nonzero(valid_tokens)),
	)


def _source_metadata(  # noqa: PLR0913
	*,
	clustering_output_dir: str | Path,
	label_path: Path,
	survey_id: str,
	confidence: float,
	boundary_alpha: float,
	boundary_tau: float,
) -> dict[str, object]:
	metadata_path = label_path.with_name(f'{survey_id}{METADATA_SUFFIX}')
	payload: dict[str, object] = {
		'boundary_weighting': {
			'adjacent_transition_distance': 0,
			'alpha': boundary_alpha,
			'invalid_gap_crossing': False,
			'method': 'transition_distance_exponential',
			'tau': boundary_tau,
		},
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
	write_boundary_weight: bool = True,
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
			*((item.output_paths.boundary_weight,) if write_boundary_weight else ()),
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
