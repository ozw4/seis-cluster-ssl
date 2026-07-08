"""Artifact registry path validation helpers for config resolvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.config.common import _is_relative_to
from seis_ssl_cluster.paths import ArtifactPaths, ExperimentKey, reject_runs_path

if TYPE_CHECKING:
	from pathlib import Path

_NOPIMS_DATASET = 'nopims'
_NOPIMS_PRETRAIN_VERSION = 'pretrain_v1'


def _validate_artifact_output_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	nopims_root: Path | None,
	raw_root_label: str = 'paths.nopims_root',
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute artifact-registry path; got {path}'
		raise ValueError(msg)
	reject_runs_path(path, label=label)
	if nopims_root is not None and _is_relative_to(path, nopims_root):
		msg = f'{label} must not be under {raw_root_label}; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _validate_nopims_checkpoint_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	reject_runs_path(path, label=label)
	_validate_nopims_pretraining_path(
		path.parent,
		f'{label} parent',
		artifact_root=artifact_root,
	)


def _validate_nopims_pretraining_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	relative = _artifact_relative_path(path, artifact_root)
	if relative is None:
		return
	parts = relative.parts
	expected = 'pretraining/nopims/pretrain_v1/<MODEL_TAG>/<RUN_SPEC>'
	if not _is_nopims_artifact_path(parts, ('pretraining',)):
		return
	if len(parts) != 5 or parts[2] != _NOPIMS_PRETRAIN_VERSION:
		_raise_nopims_artifact_path_error(label, path, expected)
	key = ExperimentKey(
		dataset=parts[1],
		version=parts[2],
		model_tag=parts[3],
		run_spec=parts[4],
	)
	_validate_artifact_path_matches(
		path,
		ArtifactPaths(artifact_root).pretraining(key),
		label=label,
		expected=expected,
	)


def _validate_nopims_embedding_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	relative = _artifact_relative_path(path, artifact_root)
	if relative is None:
		return
	parts = relative.parts
	expected = (
		'embeddings/nopims/pretrain_v1/'
		'<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>'
	)
	if not _is_nopims_artifact_path(parts, ('embeddings',)):
		return
	if len(parts) != 6 or parts[2] != _NOPIMS_PRETRAIN_VERSION:
		_raise_nopims_artifact_path_error(label, path, expected)
	key = ExperimentKey(
		dataset=parts[1],
		version=parts[2],
		model_tag=parts[3],
		subset=parts[4],
		embed_spec=parts[5],
	)
	_validate_artifact_path_matches(
		path,
		ArtifactPaths(artifact_root).embeddings(key),
		label=label,
		expected=expected,
	)


def _validate_nopims_clustering_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	relative = _artifact_relative_path(path, artifact_root)
	if relative is None:
		return
	parts = relative.parts
	expected = (
		'clustering/nopims/pretrain_v1/'
		'<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>/<CLUSTER_SPEC>'
	)
	if not _is_nopims_artifact_path(parts, ('clustering',)):
		return
	if len(parts) != 7 or parts[2] != _NOPIMS_PRETRAIN_VERSION:
		_raise_nopims_artifact_path_error(label, path, expected)
	key = ExperimentKey(
		dataset=parts[1],
		version=parts[2],
		model_tag=parts[3],
		subset=parts[4],
		embed_spec=parts[5],
		cluster_spec=parts[6],
	)
	_validate_artifact_path_matches(
		path,
		ArtifactPaths(artifact_root).clustering(key),
		label=label,
		expected=expected,
	)


def _validate_nopims_cluster_visualization_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
) -> None:
	relative = _artifact_relative_path(path, artifact_root)
	if relative is None:
		return
	parts = relative.parts
	expected = (
		'visualizations/clusters/nopims/pretrain_v1/'
		'<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>/<CLUSTER_SPEC>/<VIZ_SPEC>'
	)
	if not _is_nopims_artifact_path(parts, ('visualizations', 'clusters')):
		return
	if len(parts) != 9 or parts[3] != _NOPIMS_PRETRAIN_VERSION:
		_raise_nopims_artifact_path_error(label, path, expected)
	key = ExperimentKey(
		dataset=parts[2],
		version=parts[3],
		model_tag=parts[4],
		subset=parts[5],
		embed_spec=parts[6],
		cluster_spec=parts[7],
		viz_spec=parts[8],
	)
	_validate_artifact_path_matches(
		path,
		ArtifactPaths(artifact_root).cluster_visualization(key),
		label=label,
		expected=expected,
	)


def _artifact_relative_path(path: Path, artifact_root: Path) -> Path | None:
	try:
		return path.resolve(strict=False).relative_to(
			artifact_root.resolve(strict=False),
		)
	except ValueError:
		return None


def _is_nopims_artifact_path(
	parts: tuple[str, ...],
	stage_prefix: tuple[str, ...],
) -> bool:
	prefix_len = len(stage_prefix)
	if len(parts) <= prefix_len:
		return False
	return (
		parts[:prefix_len] == stage_prefix
		and parts[prefix_len] == _NOPIMS_DATASET
	)


def _validate_artifact_path_matches(
	path: Path,
	expected_path: Path,
	*,
	label: str,
	expected: str,
) -> None:
	if path.resolve(strict=False) != expected_path.resolve(strict=False):
		_raise_nopims_artifact_path_error(label, path, expected)


def _raise_nopims_artifact_path_error(
	label: str,
	path: Path,
	expected: str,
) -> None:
	msg = f'{label} must follow ArtifactPaths {expected}; got {path}'
	raise ValueError(msg)
