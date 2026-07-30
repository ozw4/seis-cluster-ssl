"""Immutable multi-head exports for source-label XY consensus targets.

This contract intentionally consumes only the already-published hard-target
manifest.  It does not inspect embeddings or posteriors, fit a model, update
emissions, re-decode Viterbi paths, or calibrate a strength parameter.  Every
published label is a deterministic single synchronous application of the
fixed source-label XY consensus policy.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.stratigraphy.targets import (
	build_pseudo_target_metadata,
	pseudo_target_paths,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus import (
	XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_POLICY,
	XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_SEMANTICS,
	XYNeighborConsensusResult,
	smooth_xy_neighbor_consensus_hard_labels,
)

ARTIFACT_TYPE = 'strat_hmm_multi_head_xy_neighbor_consensus_target_manifest'
SCHEMA_VERSION = 1
TARGET_REPRESENTATION = 'xy_neighbor_consensus_hard_labels_v1'
TARGET_SEMANTICS = XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_SEMANTICS
CANONICAL_KS = (6, 8, 10)
_BUNDLE_DIRNAME = 'bundle'
_SOURCE_HARD_ARTIFACT_TYPE = 'strat_hmm_multi_head_target_manifest'
_SOURCE_HARD_SCHEMA_VERSIONS = frozenset({1, 2})
_Action = Literal['NEW', 'REUSE', 'QUARANTINE', 'ERROR']


@dataclass(frozen=True)
class MultiHeadXYNeighborConsensusTargetExportConfig:
	"""Resolved immutable source-label XY-consensus export configuration."""

	source_hard_manifest: Path
	output_root: Path
	handoff_manifest: Path


@dataclass(frozen=True)
class MultiHeadXYNeighborConsensusTargetExportPlan:
	"""One K head's common bundle action."""

	k: int
	action: _Action
	reason: str | None = None


@dataclass(frozen=True)
class _Preflight:
	"""Fully validated immutable source state before publication mutation."""

	hard: Mapping[str, object]
	snapshot: str
	plans: tuple[MultiHeadXYNeighborConsensusTargetExportPlan, ...]


class _OwnedOutputCorruptionError(ValueError):
	"""An owned target bundle is incomplete or fails its semantic contract."""


class _ImmutableIdentityMismatchError(ValueError):
	"""An existing publication belongs to another hard-target source identity."""


def resolve_multi_head_xy_neighbor_consensus_target_export_config(
	config: Mapping[str, object],
) -> MultiHeadXYNeighborConsensusTargetExportConfig:
	"""Resolve the deliberately non-extensible XY-consensus exporter schema."""
	allowed = {
		'source_hard_manifest',
		'output_root',
		'handoff_manifest',
		'outputs',
	}
	unknown = set(config) - allowed
	if unknown:
		raise ValueError(
			f'unknown XY-neighbor-consensus target config keys: {sorted(unknown)}'
		)
	outputs = config.get('outputs')
	if outputs not in (None, {'overwrite': False}):
		raise ValueError('outputs must be omitted or exactly {overwrite: false}')

	def required_path(name: str) -> Path:
		value = config.get(name)
		if not isinstance(value, str) or not value:
			raise TypeError(f'{name} must be a non-empty path')
		return Path(value)

	source_hard_manifest = required_path('source_hard_manifest')
	if not source_hard_manifest.is_file():
		raise FileNotFoundError(
			f'source_hard_manifest is missing: {source_hard_manifest}'
		)
	output_root = required_path('output_root')
	handoff_value = config.get('handoff_manifest')
	if handoff_value is None:
		handoff_manifest = (
			output_root / 'multi_head_xy_neighbor_consensus_target_handoff.json'
		)
	else:
		if not isinstance(handoff_value, str) or not handoff_value:
			raise TypeError('handoff_manifest must be a non-empty path')
		handoff_manifest = Path(handoff_value)
	resolved = MultiHeadXYNeighborConsensusTargetExportConfig(
		source_hard_manifest=source_hard_manifest,
		output_root=output_root,
		handoff_manifest=handoff_manifest,
	)
	_validate_output_scope(resolved)
	return resolved


def plan_multi_head_xy_neighbor_consensus_target_exports(
	config: MultiHeadXYNeighborConsensusTargetExportConfig,
	*,
	only_missing: bool,
) -> list[MultiHeadXYNeighborConsensusTargetExportPlan]:
	"""Validate the source and classify the immutable output bundle."""
	return list(_preflight(config, only_missing=only_missing).plans)


def export_multi_head_xy_neighbor_consensus_targets(
	config: MultiHeadXYNeighborConsensusTargetExportConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
) -> list[MultiHeadXYNeighborConsensusTargetExportPlan]:
	"""Atomically publish one complete K=6/8/10 consensus target bundle."""
	preflight = _preflight(config, only_missing=only_missing)
	plans = list(preflight.plans)
	if dry_run:
		return plans
	if any(plan.action == 'ERROR' for plan in plans):
		raise FileExistsError(_plan_error(plans))
	if all(plan.action == 'REUSE' for plan in plans):
		return plans

	config.output_root.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix='.xy-neighbor-consensus.bundle.',
			dir=config.output_root,
		)
	)
	try:
		for k in CANONICAL_KS:
			_export_head(
				staging,
				k=k,
				hard=preflight.hard,
				source_manifest=config.source_hard_manifest,
			)
		for k in CANONICAL_KS:
			_validate_staged_head(
				staging,
				k=k,
				hard=preflight.hard,
				source_manifest=config.source_hard_manifest,
			)
		_rebase_head_metadata(
			staging,
			old_root=staging,
			new_root=_bundle_path(config),
		)
		_validate_live_snapshot(config, expected=preflight.snapshot)

		if any(plan.action == 'QUARANTINE' for plan in plans):
			_quarantine_if_exists(_bundle_path(config))
			_quarantine_if_exists(config.handoff_manifest)
		staging.replace(_bundle_path(config))
		for k in CANONICAL_KS:
			_validate_staged_head(
				_bundle_path(config),
				k=k,
				hard=preflight.hard,
				source_manifest=config.source_hard_manifest,
			)
		_publish_manifest(config)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	else:
		shutil.rmtree(staging, ignore_errors=True)
	return plans


def load_multi_head_xy_neighbor_consensus_target_manifest(
	path: str | Path,
	*,
	validate_array_semantics: bool = True,
) -> dict[str, object]:
	"""Load and validate a consensus hard-target handoff manifest."""
	payload = _read_json_object(Path(path), 'XY-neighbor-consensus target manifest')
	validate_multi_head_xy_neighbor_consensus_target_manifest(
		payload,
		validate_array_semantics=validate_array_semantics,
	)
	return payload


def validate_multi_head_xy_neighbor_consensus_target_manifest(
	payload: Mapping[str, object],
	*,
	validate_array_semantics: bool = True,
) -> None:
	"""Validate immutable hard-source bindings and consensus replay semantics."""
	_required(
		payload,
		{
			'artifact_type',
			'schema_version',
			'target_representation',
			'target_semantics',
			'head_ks',
			'source_hard_manifest',
			'smoothing',
			'heads',
		},
		'XY-neighbor-consensus target manifest',
	)
	if (
		payload['artifact_type'],
		payload['schema_version'],
		payload['target_representation'],
		payload['target_semantics'],
		payload['head_ks'],
	) != (
		ARTIFACT_TYPE,
		SCHEMA_VERSION,
		TARGET_REPRESENTATION,
		TARGET_SEMANTICS,
		list(CANONICAL_KS),
	):
		raise ValueError('unsupported XY-neighbor-consensus target manifest schema')
	if payload['smoothing'] != _smoothing_policy_payload():
		raise ValueError('XY-neighbor-consensus smoothing policy differs')

	hard_path = _hashed(payload['source_hard_manifest'], 'source_hard_manifest')
	hard = _load_hard_source(
		hard_path,
		validate_array_semantics=validate_array_semantics,
	)
	heads = _mapping(payload['heads'], 'XY-neighbor-consensus heads')
	if set(heads) != {str(k) for k in CANONICAL_KS}:
		raise ValueError('XY-neighbor-consensus heads must contain K=6/8/10')

	for k in CANONICAL_KS:
		_validate_head(
			_mapping(heads[str(k)], f'XY-neighbor-consensus head k={k}'),
			k=k,
			hard=hard,
			source_manifest=hard_path,
			validate_array_semantics=validate_array_semantics,
		)


def _preflight(
	config: MultiHeadXYNeighborConsensusTargetExportConfig,
	*,
	only_missing: bool,
) -> _Preflight:
	"""Complete all source and owned-output checks before any write."""
	_validate_output_scope(config)
	hard = _load_hard_source(config.source_hard_manifest)
	_validate_source_is_outside_owned_bundle(config, hard)
	snapshot = _source_snapshot(config.source_hard_manifest, hard)
	bundle = _bundle_path(config)
	if not bundle.exists():
		action: _Action = 'QUARANTINE' if config.handoff_manifest.exists() else 'NEW'
		reason = (
			'handoff exists without a complete consensus bundle'
			if action == 'QUARANTINE'
			else None
		)
		plans = _plans(action, reason)
	elif not bundle.is_dir() or not config.handoff_manifest.is_file():
		plans = _plans('QUARANTINE', 'consensus bundle or handoff is incomplete')
	else:
		try:
			_existing_publication_matches(
				config,
				source_identity=_reference(config.source_hard_manifest),
			)
		except _ImmutableIdentityMismatchError as exc:
			plans = _plans('ERROR', str(exc))
		except (OSError, TypeError, ValueError) as exc:
			plans = _plans('QUARANTINE', str(_OwnedOutputCorruptionError(str(exc))))
		else:
			plans = _plans(
				'REUSE' if only_missing else 'ERROR',
				None if only_missing else 'complete output exists; use --only-missing',
			)
	return _Preflight(hard=hard, snapshot=snapshot, plans=plans)


def _load_hard_source(
	path: Path,
	*,
	validate_array_semantics: bool = True,
) -> Mapping[str, object]:
	"""Read only the hard-source JSON plus its labels and valid masks.

	The generic multi-head manifest loader also validates source embedding
	artifacts.  That is deliberately inappropriate here: the consensus contract
	uses neither embeddings nor target metadata, and needs to remain usable if
	those unrelated source artifacts have been relocated or retired.
	"""
	if not path.is_file():
		raise FileNotFoundError(f'source_hard_manifest is missing: {path}')
	hard = _read_json_object(path, 'source hard manifest')
	_validate_hard_source_manifest(
		hard,
		validate_array_semantics=validate_array_semantics,
	)
	return hard


def _validate_hard_source_manifest(  # noqa: C901
	hard: Mapping[str, object],
	*,
	validate_array_semantics: bool,
) -> None:
	"""Validate only the source hard-label subset required by this exporter."""
	if (
		hard.get('artifact_type') != _SOURCE_HARD_ARTIFACT_TYPE
		or hard.get('schema_version') not in _SOURCE_HARD_SCHEMA_VERSIONS
	):
		raise ValueError('source hard manifest schema is unsupported')
	if hard.get('ordering_orientation') != 'increasing_downward':
		raise ValueError('source hard manifest ordering orientation is unsupported')
	if hard.get('head_ks') != list(CANONICAL_KS):
		raise ValueError('source hard manifest must use K=6/8/10')
	heads = _mapping(hard.get('heads'), 'source hard heads')
	common_surveys: set[str] | None = None
	common_valid_hashes: dict[str, str] = {}
	common_valid: dict[str, np.ndarray] = {}
	for k in CANONICAL_KS:
		head = _mapping(heads.get(str(k)), f'source hard head k={k}')
		surveys = _mapping(head.get('surveys'), f'source hard surveys k={k}')
		ids = {str(item) for item in surveys}
		if not ids:
			raise ValueError(f'source hard head k={k} has no surveys')
		if common_surveys is None:
			common_surveys = ids
		elif ids != common_surveys:
			raise ValueError('source hard survey sets differ across heads')
		for survey_id, raw in surveys.items():
			entry = _mapping(raw, f'source hard survey k={k} {survey_id}')
			_hashed(entry.get('labels'), 'source hard labels')
			valid_reference = _mapping(
				entry.get('valid_tokens'),
				'source hard valid_tokens',
			)
			_hashed(valid_reference, 'source hard valid_tokens')
			valid_hash = _string(
				valid_reference.get('sha256'),
				'source hard valid_tokens.sha256',
			)
			previous_hash = common_valid_hashes.get(str(survey_id))
			if previous_hash is not None and previous_hash != valid_hash:
				raise ValueError(
					'source hard valid-mask identities differ across heads for '
					f'{survey_id}'
				)
			common_valid_hashes.setdefault(str(survey_id), valid_hash)
			if not validate_array_semantics:
				continue
			labels, valid, _, _ = _source_arrays(entry)
			_validate_source_hard_array_semantics(
				labels,
				valid,
				k=k,
				survey_id=str(survey_id),
			)
			previous = common_valid.get(str(survey_id))
			if previous is not None and not np.array_equal(previous, valid):
				raise ValueError(
					f'source hard valid masks differ across heads for {survey_id}'
				)
			common_valid.setdefault(str(survey_id), np.asarray(valid))


def _validate_source_hard_array_semantics(
	labels: np.ndarray,
	valid: np.ndarray,
	*,
	k: int,
	survey_id: str,
) -> None:
	"""Validate frozen hard labels without touching source embeddings or metadata."""
	if not np.any(valid):
		raise ValueError(f'source hard labels have no valid tokens for {survey_id}')
	if np.any(labels[valid] < 0) or np.any(labels[valid] >= k):
		raise ValueError(
			f'source hard labels violate hard-target semantics for {survey_id}'
		)
	violations, _ = _ordered_path_metrics(labels, valid)
	if violations:
		raise ValueError(f'source hard labels violate ordered paths for {survey_id}')


def _existing_publication_matches(
	config: MultiHeadXYNeighborConsensusTargetExportConfig,
	*,
	source_identity: Mapping[str, str],
) -> None:
	"""Classify a complete owned publication without allowing source drift."""
	payload = _read_json_object(
		config.handoff_manifest,
		'XY-neighbor-consensus target handoff',
	)
	if payload.get('source_hard_manifest') != source_identity:
		raise _ImmutableIdentityMismatchError(
			'consensus bundle source hard manifest identity differs from current source'
		)
	validate_multi_head_xy_neighbor_consensus_target_manifest(payload)
	_validate_bundle_layout(payload, _bundle_path(config))


def _plans(
	action: _Action,
	reason: str | None,
) -> tuple[MultiHeadXYNeighborConsensusTargetExportPlan, ...]:
	return tuple(
		MultiHeadXYNeighborConsensusTargetExportPlan(k, action, reason)
		for k in CANONICAL_KS
	)


def _plan_error(plans: list[MultiHeadXYNeighborConsensusTargetExportPlan]) -> str:
	return '; '.join(
		f'k={plan.k}: {plan.reason}' for plan in plans if plan.action == 'ERROR'
	)


def _export_head(
	staging: Path,
	*,
	k: int,
	hard: Mapping[str, object],
	source_manifest: Path,
) -> None:
	"""Export one source-bound head below the unobservable staging bundle."""
	head_root = staging / f'k{k}'
	head_root.mkdir(parents=True, exist_ok=True)
	hard_surveys = _hard_surveys(hard, k=k)
	surveys: dict[str, object] = {}
	per_survey: dict[str, object] = {}
	for survey_id in sorted(hard_surveys):
		source = _mapping(hard_surveys[survey_id], f'source hard survey {survey_id}')
		source_labels, source_valid, source_label_path, source_valid_path = (
			_source_arrays(source)
		)
		result = smooth_xy_neighbor_consensus_hard_labels(
			source_labels,
			source_valid,
		)
		labels = np.asarray(result.labels, dtype=np.int32)
		confidence = np.asarray(source_valid, dtype=np.float32)
		paths = pseudo_target_paths(staging, k=k, survey_id=str(survey_id))
		paths.labels.parent.mkdir(parents=True, exist_ok=True)
		np.save(paths.labels, labels, allow_pickle=False)
		np.save(paths.confidence, confidence, allow_pickle=False)
		np.save(paths.valid_tokens, source_valid, allow_pickle=False)
		_write_json(
			paths.metadata,
			_expected_metadata(
				labels=labels,
				valid=source_valid,
				k=k,
				survey_id=str(survey_id),
				source_manifest=source_manifest,
				source_label_path=source_label_path,
				source_valid_path=source_valid_path,
			),
		)
		surveys[str(survey_id)] = {
			'labels': _array_reference(paths.labels, labels.shape, labels.dtype),
			'confidence': _array_reference(
				paths.confidence,
				confidence.shape,
				confidence.dtype,
			),
			'valid_tokens': _array_reference(
				paths.valid_tokens,
				source_valid.shape,
				source_valid.dtype,
			),
			'metadata': _reference(paths.metadata),
			'source_hard_labels': _reference(source_label_path),
			'source_hard_valid_tokens': _reference(source_valid_path),
		}
		per_survey[str(survey_id)] = _survey_diagnostics(
			source_labels,
			labels,
			source_valid,
			result,
			k=k,
		)
	diagnostics_payload = {
		'per_survey': per_survey,
		'aggregate': _aggregate_diagnostics(per_survey, k=k),
	}
	diagnostics_path = head_root / 'diagnostics.json'
	_write_json(diagnostics_path, diagnostics_payload)
	head = {
		'surveys': surveys,
		'diagnostics': {
			**diagnostics_payload,
			'json': _reference(diagnostics_path),
		},
	}
	_write_json(head_root / 'head_metadata.json', head)


def _validate_staged_head(
	root: Path,
	*,
	k: int,
	hard: Mapping[str, object],
	source_manifest: Path,
) -> None:
	path = root / f'k{k}' / 'head_metadata.json'
	head = _read_json_object(path, f'XY-neighbor-consensus head k={k}')
	_validate_head(
		head,
		k=k,
		hard=hard,
		source_manifest=source_manifest,
		validate_array_semantics=True,
	)


def _validate_head(
	head: Mapping[str, object],
	*,
	k: int,
	hard: Mapping[str, object],
	source_manifest: Path,
	validate_array_semantics: bool,
) -> None:
	_required(head, {'surveys', 'diagnostics'}, f'XY-neighbor-consensus head k={k}')
	surveys = _mapping(head['surveys'], f'XY-neighbor-consensus surveys k={k}')
	hard_surveys = _hard_surveys(hard, k=k)
	if set(surveys) != set(hard_surveys):
		raise ValueError(f'consensus survey set differs from source for k={k}')

	expected_per_survey: dict[str, object] = {}
	for survey_id in sorted(hard_surveys):
		entry = _mapping(surveys[survey_id], f'consensus survey k={k} {survey_id}')
		_required(
			entry,
			{
				'labels',
				'confidence',
				'valid_tokens',
				'metadata',
				'source_hard_labels',
				'source_hard_valid_tokens',
			},
			f'consensus survey k={k} {survey_id}',
		)
		labels = _load_output_array(entry['labels'], 'labels')
		confidence = _load_output_array(entry['confidence'], 'confidence')
		valid = _load_output_array(entry['valid_tokens'], 'valid_tokens')
		if (
			labels.dtype != np.int32
			or labels.ndim != 3
			or confidence.dtype != np.float32
			or confidence.shape != labels.shape
			or valid.dtype != np.bool_
			or valid.shape != labels.shape
		):
			raise ValueError('consensus arrays have invalid shape or dtype')

		source = _mapping(hard_surveys[survey_id], f'source hard survey {survey_id}')
		source_labels, source_valid, source_label_path, source_valid_path = (
			_source_arrays(source)
		)
		if entry['source_hard_labels'] != _reference(source_label_path):
			raise ValueError('consensus source hard-label provenance differs')
		if entry['source_hard_valid_tokens'] != _reference(source_valid_path):
			raise ValueError('consensus source valid-mask provenance differs')
		if labels.shape != source_labels.shape or valid.shape != source_valid.shape:
			raise ValueError('consensus source and output grid shapes differ')

		metadata = _read_json_object(
			_hashed(entry['metadata'], 'metadata'),
			'consensus pseudo-target metadata',
		)
		if validate_array_semantics:
			_validate_output_array_semantics(
				labels,
				confidence,
				valid,
				source_labels=source_labels,
				source_valid=source_valid,
				k=k,
			)
			result = smooth_xy_neighbor_consensus_hard_labels(
				source_labels,
				source_valid,
			)
			if not np.array_equal(labels, result.labels):
				raise ValueError(
					'consensus labels differ from the source-label smoothing '
					'replay result'
				)
			expected_per_survey[str(survey_id)] = _survey_diagnostics(
				source_labels,
				labels,
				source_valid,
				result,
				k=k,
			)
			expected_metadata = _expected_metadata(
				labels=labels,
				valid=valid,
				k=k,
				survey_id=str(survey_id),
				source_manifest=source_manifest,
				source_label_path=source_label_path,
				source_valid_path=source_valid_path,
			)
			if metadata != expected_metadata:
				raise ValueError('consensus pseudo-target metadata provenance differs')
		else:
			_validate_metadata_structure(metadata)

	_validate_head_diagnostics(
		head['diagnostics'],
		survey_ids=set(hard_surveys),
		k=k,
		expected_per_survey=(expected_per_survey if validate_array_semantics else None),
	)


def _hard_surveys(hard: Mapping[str, object], *, k: int) -> Mapping[str, object]:
	return _mapping(
		_mapping(
			_mapping(hard['heads'], 'source hard heads')[str(k)],
			f'source hard head k={k}',
		)['surveys'],
		f'source hard surveys k={k}',
	)


def _source_arrays(
	source: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
	source_label_path = _hashed(source['labels'], 'source hard labels')
	source_valid_path = _hashed(source['valid_tokens'], 'source hard valid_tokens')
	labels = np.load(source_label_path, mmap_mode='r', allow_pickle=False)
	valid = np.load(source_valid_path, mmap_mode='r', allow_pickle=False)
	if (
		labels.dtype != np.int32
		or labels.ndim != 3
		or valid.dtype != np.bool_
		or valid.shape != labels.shape
	):
		raise ValueError('source hard labels or valid mask have invalid shape or dtype')
	return labels, valid, source_label_path, source_valid_path


def _load_output_array(reference: object, name: str) -> np.ndarray:
	path = _hashed(reference, name)
	array = np.load(path, mmap_mode='r', allow_pickle=False)
	_validate_array_reference(reference, array, name=name)
	return array


def _validate_output_array_semantics(  # noqa: PLR0913
	labels: np.ndarray,
	confidence: np.ndarray,
	valid: np.ndarray,
	*,
	source_labels: np.ndarray,
	source_valid: np.ndarray,
	k: int,
) -> None:
	if not np.array_equal(valid, source_valid):
		raise ValueError('consensus valid mask differs from source hard valid mask')
	if (
		np.any(labels[valid] < 0)
		or np.any(labels[valid] >= k)
		or not np.array_equal(labels[~valid], source_labels[~source_valid])
	):
		raise ValueError('consensus labels violate source hard-label semantics')
	if np.any(confidence[valid] != 1.0) or np.any(confidence[~valid] != 0.0):
		raise ValueError('consensus confidence must be unity on valid tokens only')
	violations, _ = _ordered_path_metrics(labels, valid)
	if violations:
		raise ValueError('consensus labels violate ordered paths')


def _metadata_provenance(
	*,
	source_manifest: Path,
	source_label_path: Path,
	source_valid_path: Path,
) -> dict[str, object]:
	return {
		'target_representation': TARGET_REPRESENTATION,
		'target_semantics': TARGET_SEMANTICS,
		'source_hard_manifest': _reference(source_manifest),
		'source_hard_labels': _reference(source_label_path),
		'source_hard_valid_tokens': _reference(source_valid_path),
		'smoothing': _smoothing_policy_payload(),
	}


def _expected_metadata(  # noqa: PLR0913
	*,
	labels: np.ndarray,
	valid: np.ndarray,
	k: int,
	survey_id: str,
	source_manifest: Path,
	source_label_path: Path,
	source_valid_path: Path,
) -> dict[str, object]:
	return build_pseudo_target_metadata(
		labels=labels,
		valid_tokens=valid,
		boundary_weight=np.asarray(valid, dtype=np.float32),
		boundary_weight_source='default_unity',
		k=k,
		survey_id=survey_id,
		schema_version=1,
		write_boundary_weight=False,
		source_metadata=_metadata_provenance(
			source_manifest=source_manifest,
			source_label_path=source_label_path,
			source_valid_path=source_valid_path,
		),
	)


def _validate_metadata_structure(metadata: Mapping[str, object]) -> None:
	"""Reference-only metadata validation for lazy dataset construction."""
	if metadata.get('artifact_type') != 'strat_hmm_pseudo_target':
		raise ValueError('consensus pseudo-target metadata has wrong artifact type')
	if metadata.get('schema_version') != 1:
		raise ValueError('consensus pseudo-target metadata has wrong schema version')
	source = _mapping(metadata.get('source'), 'consensus pseudo-target metadata source')
	_required(
		source,
		{
			'target_representation',
			'target_semantics',
			'source_hard_manifest',
			'source_hard_labels',
			'source_hard_valid_tokens',
			'smoothing',
		},
		'consensus pseudo-target metadata source',
	)
	if (
		source.get('target_representation') != TARGET_REPRESENTATION
		or source.get('target_semantics') != TARGET_SEMANTICS
		or source.get('smoothing') != _smoothing_policy_payload()
	):
		raise ValueError('consensus pseudo-target metadata policy differs')
	for name in (
		'source_hard_manifest',
		'source_hard_labels',
		'source_hard_valid_tokens',
	):
		_hashed(source.get(name), f'consensus metadata {name}')


def _survey_diagnostics(
	source_labels: np.ndarray,
	labels: np.ndarray,
	valid: np.ndarray,
	result: XYNeighborConsensusResult,
	*,
	k: int,
) -> dict[str, object]:
	"""Build diagnostics that are replayable from hard labels alone."""
	changed = valid & (labels != source_labels)
	diagnostics = result.diagnostics
	if not np.array_equal(changed, diagnostics.changed_mask):
		raise ValueError('consensus core changed-mask diagnostics differ from labels')
	neighbor_counts = np.asarray(diagnostics.neighbor_count[valid], dtype=np.int64)
	if np.any(neighbor_counts < 0) or np.any(neighbor_counts > 4):
		raise ValueError('consensus core neighbor-count diagnostics are invalid')
	violations, maximum_reverse_decrease = _ordered_path_metrics(labels, valid)
	valid_count = int(np.count_nonzero(valid))
	return {
		'valid_token_count': valid_count,
		'invalid_token_count': int(valid.size - valid_count),
		'changed_token_count': int(np.count_nonzero(changed)),
		'changed_fraction': int(np.count_nonzero(changed)) / max(valid_count, 1),
		'temporal_transition_counts': {
			'source': _temporal_transition_count(source_labels, valid),
			'output': _temporal_transition_count(labels, valid),
		},
		'source_state_occupancy_count': np.bincount(source_labels[valid], minlength=k)
		.astype(int)
		.tolist(),
		'state_occupancy_count': np.bincount(labels[valid], minlength=k)
		.astype(int)
		.tolist(),
		'ordered_path': {
			'violation_count': violations,
			'max_reverse_decrease': maximum_reverse_decrease,
		},
		'consensus_decisions': {
			'neighbor_count_histogram': np.bincount(
				neighbor_counts,
				minlength=5,
			)
			.astype(int)
			.tolist(),
			'consensus_token_count': int(np.count_nonzero(diagnostics.consensus_mask)),
			'change_candidate_count': int(
				np.count_nonzero(diagnostics.change_candidate_mask)
			),
			'internal_valid_token_count': int(
				np.count_nonzero(diagnostics.internal_valid_mask)
			),
			'order_compatible_candidate_count': int(
				np.count_nonzero(diagnostics.order_compatible_mask)
			),
			'changed_token_count': int(np.count_nonzero(diagnostics.changed_mask)),
		},
	}


def _aggregate_diagnostics(
	per_survey: Mapping[str, object],
	*,
	k: int,
) -> dict[str, object]:
	if not per_survey:
		raise ValueError('consensus diagnostics require at least one survey')
	metrics = [
		_mapping(value, 'consensus per-survey diagnostics')
		for value in per_survey.values()
	]
	valid_count = sum(int(item['valid_token_count']) for item in metrics)
	changed_count = sum(int(item['changed_token_count']) for item in metrics)
	source_counts = np.zeros(k, dtype=np.int64)
	output_counts = np.zeros(k, dtype=np.int64)
	source_transition_count = 0
	output_transition_count = 0
	neighbor_histogram = np.zeros(5, dtype=np.int64)
	decision_counts = {
		'consensus_token_count': 0,
		'change_candidate_count': 0,
		'internal_valid_token_count': 0,
		'order_compatible_candidate_count': 0,
		'changed_token_count': 0,
	}
	violations = 0
	maximum_reverse_decrease = 0
	for item in metrics:
		source_counts += np.asarray(
			item['source_state_occupancy_count'], dtype=np.int64
		)
		output_counts += np.asarray(item['state_occupancy_count'], dtype=np.int64)
		transition_counts = _mapping(
			item['temporal_transition_counts'],
			'consensus temporal-transition diagnostics',
		)
		# These counts are observational diagnostics.  In particular, output may
		# legitimately exceed source under the fixed consensus policy.
		source_transition_count += int(transition_counts['source'])
		output_transition_count += int(transition_counts['output'])
		ordered = _mapping(item['ordered_path'], 'consensus ordered-path diagnostics')
		violations += int(ordered['violation_count'])
		maximum_reverse_decrease = max(
			maximum_reverse_decrease,
			int(ordered['max_reverse_decrease']),
		)
		decisions = _mapping(item['consensus_decisions'], 'consensus decisions')
		neighbor_histogram += np.asarray(
			decisions['neighbor_count_histogram'],
			dtype=np.int64,
		)
		for name in decision_counts:
			decision_counts[name] += int(decisions[name])
	return {
		'survey_count': len(metrics),
		'valid_token_count': valid_count,
		'invalid_token_count': sum(
			int(item['invalid_token_count']) for item in metrics
		),
		'changed_token_count': changed_count,
		'changed_fraction': changed_count / max(valid_count, 1),
		'temporal_transition_counts': {
			'source': source_transition_count,
			'output': output_transition_count,
		},
		'source_state_occupancy_count': source_counts.astype(int).tolist(),
		'state_occupancy_count': output_counts.astype(int).tolist(),
		'ordered_path': {
			'violation_count': violations,
			'max_reverse_decrease': maximum_reverse_decrease,
		},
		'consensus_decisions': {
			'neighbor_count_histogram': neighbor_histogram.astype(int).tolist(),
			**decision_counts,
		},
	}


def _ordered_path_metrics(
	labels: np.ndarray,
	valid: np.ndarray,
) -> tuple[int, int]:
	violations = 0
	maximum_reverse_decrease = 0
	for x in range(labels.shape[0]):
		for y in range(labels.shape[1]):
			trace = labels[x, y, valid[x, y]]
			decrease = np.diff(trace)
			violations += int(np.count_nonzero(decrease < 0))
			maximum_reverse_decrease = max(
				maximum_reverse_decrease,
				int(max(0, -decrease.min(initial=0))),
			)
	return violations, maximum_reverse_decrease


def _temporal_transition_count(labels: np.ndarray, valid: np.ndarray) -> int:
	"""Count label changes over each trace's ordered valid-token sequence.

	Invalid-token gaps are intentionally bridged because trace ordering and the
	consensus guard use the same valid-token sequence.  This is a descriptive
	diagnostic only: callers must not use it as an eligibility or stop gate.
	"""
	transitions = 0
	for x in range(labels.shape[0]):
		for y in range(labels.shape[1]):
			trace = labels[x, y, valid[x, y]]
			transitions += int(np.count_nonzero(trace[1:] != trace[:-1]))
	return transitions


def _validate_head_diagnostics(
	value: object,
	*,
	survey_ids: set[str],
	k: int,
	expected_per_survey: Mapping[str, object] | None,
) -> None:
	diagnostics = _mapping(value, f'consensus diagnostics k={k}')
	_required(
		diagnostics,
		{'per_survey', 'aggregate', 'json'},
		f'consensus diagnostics k={k}',
	)
	per_survey = _mapping(diagnostics['per_survey'], 'consensus per-survey diagnostics')
	if set(per_survey) != survey_ids:
		raise ValueError(f'consensus diagnostics survey set differs for k={k}')
	diagnostics_path = _hashed(diagnostics['json'], 'consensus diagnostics JSON')
	persisted = _read_json_object(diagnostics_path, 'consensus diagnostics JSON')
	if persisted != {
		'per_survey': diagnostics['per_survey'],
		'aggregate': diagnostics['aggregate'],
	}:
		raise ValueError('consensus diagnostics JSON differs from head metadata')
	aggregate = _mapping(diagnostics['aggregate'], 'aggregate diagnostics')
	_validate_diagnostic_structure(per_survey, aggregate, k=k)
	if expected_per_survey is not None:
		expected = {
			'per_survey': dict(expected_per_survey),
			'aggregate': _aggregate_diagnostics(expected_per_survey, k=k),
		}
		actual = {
			'per_survey': diagnostics['per_survey'],
			'aggregate': diagnostics['aggregate'],
		}
		if actual != expected:
			raise ValueError('consensus diagnostics differ from source-label replay')


def _validate_diagnostic_structure(  # noqa: C901, PLR0912
	per_survey: Mapping[str, object],
	aggregate: Mapping[str, object],
	*,
	k: int,
) -> None:
	for name, raw in [*per_survey.items(), ('aggregate', aggregate)]:
		metrics = _mapping(raw, f'consensus diagnostics {name}')
		required = {
			'valid_token_count',
			'invalid_token_count',
			'changed_token_count',
			'changed_fraction',
			'temporal_transition_counts',
			'source_state_occupancy_count',
			'state_occupancy_count',
			'ordered_path',
			'consensus_decisions',
		}
		if name == 'aggregate':
			required.add('survey_count')
		_required(metrics, required, f'consensus diagnostics {name}')
		for count_name in (
			'valid_token_count',
			'invalid_token_count',
			'changed_token_count',
		):
			value = metrics[count_name]
			if isinstance(value, bool) or not isinstance(value, int) or value < 0:
				raise ValueError(
					f'consensus diagnostics {name} {count_name} is invalid'
				)
		if metrics['changed_token_count'] > metrics['valid_token_count']:
			raise ValueError(f'consensus diagnostics {name} changed count is invalid')
		if metrics['changed_fraction'] != metrics['changed_token_count'] / max(
			metrics['valid_token_count'], 1
		):
			raise ValueError(
				f'consensus diagnostics {name} changed fraction is invalid'
			)
		transition_counts = _mapping(
			metrics['temporal_transition_counts'],
			'consensus temporal-transition diagnostics',
		)
		_required(
			transition_counts,
			{'source', 'output'},
			'consensus temporal-transition diagnostics',
		)
		for transition_name, transition_count in transition_counts.items():
			if (
				isinstance(transition_count, bool)
				or not isinstance(transition_count, int)
				or transition_count < 0
			):
				raise ValueError(
					'consensus diagnostics '
					f'{name} temporal transition count is invalid: {transition_name}'
				)
		for count_name in ('source_state_occupancy_count', 'state_occupancy_count'):
			counts = metrics[count_name]
			if (
				not isinstance(counts, list)
				or len(counts) != k
				or any(
					isinstance(item, bool) or not isinstance(item, int) or item < 0
					for item in counts
				)
				or sum(counts) != metrics['valid_token_count']
			):
				raise ValueError(
					f'consensus diagnostics {name} state occupancy is invalid'
				)
		ordered = _mapping(
			metrics['ordered_path'],
			'consensus ordered-path diagnostics',
		)
		_required(
			ordered,
			{'violation_count', 'max_reverse_decrease'},
			'consensus ordered-path diagnostics',
		)
		if any(
			isinstance(item, bool) or not isinstance(item, int) or item < 0
			for item in ordered.values()
		):
			raise ValueError(f'consensus diagnostics {name} ordered path is invalid')
		decisions = _mapping(metrics['consensus_decisions'], 'consensus decisions')
		_required(
			decisions,
			{
				'neighbor_count_histogram',
				'consensus_token_count',
				'change_candidate_count',
				'internal_valid_token_count',
				'order_compatible_candidate_count',
				'changed_token_count',
			},
			'consensus decisions',
		)
		histogram = decisions['neighbor_count_histogram']
		if (
			not isinstance(histogram, list)
			or len(histogram) != 5
			or any(
				isinstance(item, bool) or not isinstance(item, int) or item < 0
				for item in histogram
			)
			or sum(histogram) != metrics['valid_token_count']
		):
			raise ValueError(
				f'consensus diagnostics {name} neighbor histogram is invalid'
			)
		for count_name in (
			'consensus_token_count',
			'change_candidate_count',
			'internal_valid_token_count',
			'order_compatible_candidate_count',
			'changed_token_count',
		):
			count = decisions[count_name]
			if (
				isinstance(count, bool)
				or not isinstance(count, int)
				or count < 0
				or count > metrics['valid_token_count']
			):
				raise ValueError(
					f'consensus diagnostics {name} {count_name} is invalid'
				)
		if decisions['changed_token_count'] != metrics['changed_token_count']:
			raise ValueError(f'consensus diagnostics {name} changed counts differ')
		_validate_finite_tree(metrics, context=f'consensus diagnostics {name}')


def _publish_manifest(config: MultiHeadXYNeighborConsensusTargetExportConfig) -> None:
	if config.handoff_manifest.exists():
		raise FileExistsError(
			'consensus handoff appeared during publication; refusing to overwrite it'
		)
	payload = _manifest_payload(config)
	validate_multi_head_xy_neighbor_consensus_target_manifest(payload)
	_validate_bundle_layout(payload, _bundle_path(config))
	_write_json(config.handoff_manifest, payload)
	# Verify the persisted JSON too, rather than only its in-memory representation.
	load_multi_head_xy_neighbor_consensus_target_manifest(config.handoff_manifest)


def _manifest_payload(
	config: MultiHeadXYNeighborConsensusTargetExportConfig,
) -> dict[str, object]:
	head_payloads: dict[str, object] = {}
	for k in CANONICAL_KS:
		head_payloads[str(k)] = _read_json_object(
			_bundle_path(config) / f'k{k}' / 'head_metadata.json',
			f'XY-neighbor-consensus head k={k}',
		)
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'target_representation': TARGET_REPRESENTATION,
		'target_semantics': TARGET_SEMANTICS,
		'head_ks': list(CANONICAL_KS),
		'source_hard_manifest': _reference(config.source_hard_manifest),
		'smoothing': _smoothing_policy_payload(),
		'heads': head_payloads,
	}


def _validate_bundle_layout(payload: Mapping[str, object], bundle: Path) -> None:
	"""Require a handoff to point to its owned complete bundle, not another one."""
	heads = _mapping(payload['heads'], 'consensus heads')
	for k in CANONICAL_KS:
		head_root = bundle / f'k{k}'
		head_metadata_path = head_root / 'head_metadata.json'
		if not head_metadata_path.is_file():
			raise ValueError(f'consensus bundle head metadata is missing for k={k}')
		head = _mapping(heads[str(k)], f'consensus head k={k}')
		if (
			_read_json_object(
				head_metadata_path,
				f'XY-neighbor-consensus bundle head k={k}',
			)
			!= head
		):
			raise ValueError(
				f'consensus bundle head metadata differs from handoff for k={k}'
			)
		diagnostics = _mapping(head['diagnostics'], f'consensus diagnostics k={k}')
		if _hashed(diagnostics['json'], 'consensus diagnostics JSON') != (
			head_root / 'diagnostics.json'
		):
			raise ValueError(
				f'consensus diagnostics path is outside owned bundle for k={k}'
			)
		for survey_id, raw in _mapping(head['surveys'], 'consensus surveys').items():
			entry = _mapping(raw, 'consensus survey')
			expected = pseudo_target_paths(bundle, k=k, survey_id=str(survey_id))
			for name, expected_path in {
				'labels': expected.labels,
				'confidence': expected.confidence,
				'valid_tokens': expected.valid_tokens,
				'metadata': expected.metadata,
			}.items():
				if _hashed(entry[name], name) != expected_path:
					raise ValueError(
						f'consensus {name} path is outside owned bundle for k={k} '
						f'{survey_id}'
					)


def _source_snapshot(path: Path, hard: Mapping[str, object]) -> str:
	payload: dict[str, object] = {
		'source_hard_manifest': _reference(path),
		'heads': {},
	}
	heads = _mapping(payload['heads'], 'source snapshot heads')
	for k in CANONICAL_KS:
		head: dict[str, object] = {}
		for survey_id, raw in _hard_surveys(hard, k=k).items():
			source = _mapping(raw, 'source hard survey')
			head[str(survey_id)] = {
				'labels': dict(_mapping(source['labels'], 'source hard labels')),
				'valid_tokens': dict(
					_mapping(source['valid_tokens'], 'source hard valid_tokens')
				),
			}
		heads[str(k)] = head
	return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def _validate_live_snapshot(
	config: MultiHeadXYNeighborConsensusTargetExportConfig,
	*,
	expected: str,
) -> None:
	current = _load_hard_source(config.source_hard_manifest)
	if _source_snapshot(config.source_hard_manifest, current) != expected:
		raise ValueError(
			'source hard manifest or arrays changed during consensus export'
		)


def _rebase_head_metadata(root: Path, *, old_root: Path, new_root: Path) -> None:
	"""Change staging-only target paths after complete staging validation."""
	for k in CANONICAL_KS:
		path = root / f'k{k}' / 'head_metadata.json'
		payload = _read_json_object(path, f'consensus staged head k={k}')
		_write_json(path, _replace_root(payload, old_root=old_root, new_root=new_root))


def _replace_root(value: object, *, old_root: Path, new_root: Path) -> object:
	if isinstance(value, Mapping):
		return {
			str(name): _replace_root(item, old_root=old_root, new_root=new_root)
			for name, item in value.items()
		}
	if isinstance(value, list):
		return [
			_replace_root(item, old_root=old_root, new_root=new_root) for item in value
		]
	if isinstance(value, str):
		old = str(old_root)
		if value == old or value.startswith(f'{old}/'):
			return f'{new_root}{value.removeprefix(old)}'
	return value


def _bundle_path(config: MultiHeadXYNeighborConsensusTargetExportConfig) -> Path:
	return config.output_root / _BUNDLE_DIRNAME


def _validate_output_scope(
	config: MultiHeadXYNeighborConsensusTargetExportConfig,
) -> None:
	"""Keep replace/quarantine operations inside the explicitly owned root."""
	bundle = _bundle_path(config)
	if _same_resolved_path(config.source_hard_manifest, config.handoff_manifest):
		raise ValueError('handoff_manifest must differ from source_hard_manifest')
	if not _is_under(config.handoff_manifest, config.output_root):
		raise ValueError('handoff_manifest must be located under output_root')
	if _is_under(config.handoff_manifest, bundle):
		raise ValueError('handoff_manifest must not be located under the bundle')
	if _is_under(config.source_hard_manifest, bundle):
		raise ValueError('source_hard_manifest must not be located under the bundle')


def _validate_source_is_outside_owned_bundle(
	config: MultiHeadXYNeighborConsensusTargetExportConfig,
	hard: Mapping[str, object],
) -> None:
	"""Never quarantine a path that is one of this run's frozen inputs."""
	bundle = _bundle_path(config)
	for k in CANONICAL_KS:
		for survey_id, raw in _hard_surveys(hard, k=k).items():
			source = _mapping(raw, f'source hard survey {survey_id}')
			for name in ('labels', 'valid_tokens'):
				if _is_under(_hashed(source[name], f'source hard {name}'), bundle):
					raise ValueError(
						f'source hard {name} must not be located under the owned '
						f'bundle for k={k} {survey_id}'
					)


def _same_resolved_path(left: Path, right: Path) -> bool:
	return left.resolve() == right.resolve()


def _is_under(path: Path, root: Path) -> bool:
	try:
		path.resolve().relative_to(root.resolve())
	except ValueError:
		return False
	return True


def _smoothing_policy_payload() -> dict[str, object]:
	return _json_compatible(XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_POLICY)


def _json_compatible(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _json_compatible(item) for key, item in value.items()}
	if isinstance(value, tuple | list):
		return [_json_compatible(item) for item in value]
	return value


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _array_reference(
	path: Path,
	shape: tuple[int, ...],
	dtype: np.dtype[object],
) -> dict[str, object]:
	return {
		**_reference(path),
		'shape': list(shape),
		'dtype': np.dtype(dtype).name,
	}


def _validate_array_reference(
	reference: object,
	array: np.ndarray,
	*,
	name: str,
) -> None:
	item = _mapping(reference, name)
	_required(item, {'path', 'sha256', 'shape', 'dtype'}, name)
	shape = item['shape']
	if not isinstance(shape, list) or any(
		isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
		for dimension in shape
	):
		raise TypeError(f'{name}.shape must be a list of non-negative integers')
	if tuple(shape) != array.shape or item['dtype'] != array.dtype.name:
		raise ValueError(f'{name} descriptor differs from array')


def _hashed(value: object, name: str) -> Path:
	item = _mapping(value, name)
	path = Path(_string(item.get('path'), f'{name}.path'))
	if not path.is_file() or file_sha256(path) != _string(
		item.get('sha256'), f'{name}.sha256'
	):
		raise ValueError(f'{name} hash mismatch')
	return path


def _mapping(value: object, name: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{name} must be a mapping')
	return value


def _string(value: object, name: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{name} must be a non-empty string')
	return value


def _required(value: Mapping[str, object], keys: set[str], name: str) -> None:
	if set(value) != keys:
		raise ValueError(f'{name} keys mismatch; expected {sorted(keys)}')


def _read_json_object(path: Path, name: str) -> dict[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'{name} must be valid JSON: {path}') from exc
	if not isinstance(payload, dict):
		raise TypeError(f'{name} must be an object')
	return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		mode='w',
		encoding='utf-8',
		dir=path.parent,
		prefix=f'.{path.name}.',
		delete=False,
	) as stream:
		json.dump(payload, stream, sort_keys=True, allow_nan=False)
		stream.write('\n')
		temporary = Path(stream.name)
	temporary.replace(path)


def _validate_finite_tree(value: object, *, context: str) -> None:
	if isinstance(value, Mapping):
		for item in value.values():
			_validate_finite_tree(item, context=context)
	elif isinstance(value, list):
		for item in value:
			_validate_finite_tree(item, context=context)
	elif isinstance(value, float) and not math.isfinite(value):
		raise ValueError(f'{context} contains a non-finite value')


def _quarantine_if_exists(path: Path) -> None:
	if not path.exists():
		return
	stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
	shutil.move(str(path), str(path.with_name(f'{path.name}.quarantine-{stamp}')))


__all__ = [
	'ARTIFACT_TYPE',
	'SCHEMA_VERSION',
	'TARGET_REPRESENTATION',
	'TARGET_SEMANTICS',
	'MultiHeadXYNeighborConsensusTargetExportConfig',
	'MultiHeadXYNeighborConsensusTargetExportPlan',
	'export_multi_head_xy_neighbor_consensus_targets',
	'load_multi_head_xy_neighbor_consensus_target_manifest',
	'plan_multi_head_xy_neighbor_consensus_target_exports',
	'resolve_multi_head_xy_neighbor_consensus_target_export_config',
	'validate_multi_head_xy_neighbor_consensus_target_manifest',
]
