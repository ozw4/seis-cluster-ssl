# ruff: noqa: SLF001
# The consensus module owns the shared source-only publication primitives.  This
# separate artifact contract deliberately reuses those internal mechanics so it
# cannot silently diverge in hash, staging, orphan, or replay handling.
"""Immutable multi-head exports for unanimous XY-neighbour hard targets.

This successor has its own artifact identity and never widens the existing
three-of-four consensus manifest.  It shares only the source-hard validation,
atomic publication, and diagnostic structure used by the established target
exporter; its label replay always uses the fixed unanimous numerical policy.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from seis_ssl_cluster.stratigraphy import xy_neighbor_consensus_targets as _common
from seis_ssl_cluster.stratigraphy.targets import (
	build_pseudo_target_metadata,
	pseudo_target_paths,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus import (
	XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_POLICY,
	XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_SEMANTICS,
	XYNeighborConsensusResult,
	smooth_xy_neighbor_unanimous_hard_labels,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

ARTIFACT_TYPE = 'strat_hmm_multi_head_xy_neighbor_unanimous_target_manifest'
SCHEMA_VERSION = 1
TARGET_REPRESENTATION = 'xy_neighbor_unanimous_hard_labels_v1'
TARGET_SEMANTICS = XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_SEMANTICS
CANONICAL_KS = (6, 8, 10)
_BUNDLE_DIRNAME = 'bundle'
_Action = Literal['NEW', 'REUSE', 'QUARANTINE', 'ERROR']


@dataclass(frozen=True)
class MultiHeadXYNeighborUnanimousTargetExportConfig:
	"""Resolved immutable source-label unanimous-target export configuration."""

	source_hard_manifest: Path
	output_root: Path
	handoff_manifest: Path


@dataclass(frozen=True)
class MultiHeadXYNeighborUnanimousTargetExportPlan:
	"""One K head's common unanimous bundle action."""

	k: int
	action: _Action
	reason: str | None = None


@dataclass(frozen=True)
class _Preflight:
	"""Fully validated immutable source state before publication mutation."""

	hard: Mapping[str, object]
	snapshot: str
	plans: tuple[MultiHeadXYNeighborUnanimousTargetExportPlan, ...]


def resolve_multi_head_xy_neighbor_unanimous_target_export_config(
	config: Mapping[str, object],
) -> MultiHeadXYNeighborUnanimousTargetExportConfig:
	"""Resolve the deliberately non-extensible unanimous exporter schema."""
	allowed = {
		'source_hard_manifest',
		'output_root',
		'handoff_manifest',
		'outputs',
	}
	unknown = set(config) - allowed
	if unknown:
		raise ValueError(
			'unknown XY-neighbor-unanimous target config keys: '
			f'{sorted(unknown)}'
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
			output_root / 'multi_head_xy_neighbor_unanimous_target_handoff.json'
		)
	else:
		if not isinstance(handoff_value, str) or not handoff_value:
			raise TypeError('handoff_manifest must be a non-empty path')
		handoff_manifest = Path(handoff_value)
	resolved = MultiHeadXYNeighborUnanimousTargetExportConfig(
		source_hard_manifest=source_hard_manifest,
		output_root=output_root,
		handoff_manifest=handoff_manifest,
	)
	_common._validate_output_scope(resolved)
	return resolved


def plan_multi_head_xy_neighbor_unanimous_target_exports(
	config: MultiHeadXYNeighborUnanimousTargetExportConfig,
	*,
	only_missing: bool,
) -> list[MultiHeadXYNeighborUnanimousTargetExportPlan]:
	"""Validate the source and classify the immutable unanimous bundle."""
	return list(_preflight(config, only_missing=only_missing).plans)


def export_multi_head_xy_neighbor_unanimous_targets(
	config: MultiHeadXYNeighborUnanimousTargetExportConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
) -> list[MultiHeadXYNeighborUnanimousTargetExportPlan]:
	"""Atomically publish one complete K=6/8/10 unanimous target bundle."""
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
			prefix='.xy-neighbor-unanimous.bundle.',
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
		_common._rebase_head_metadata(
			staging,
			old_root=staging,
			new_root=_bundle_path(config),
		)
		_common._validate_live_snapshot(config, expected=preflight.snapshot)

		if any(plan.action == 'QUARANTINE' for plan in plans):
			_common._quarantine_if_exists(_bundle_path(config))
			_common._quarantine_if_exists(config.handoff_manifest)
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


def load_multi_head_xy_neighbor_unanimous_target_manifest(
	path: str | Path,
	*,
	validate_array_semantics: bool = True,
) -> dict[str, object]:
	"""Load and validate an immutable unanimous hard-target handoff manifest."""
	payload = _common._read_json_object(
		Path(path),
		'XY-neighbor-unanimous target manifest',
	)
	validate_multi_head_xy_neighbor_unanimous_target_manifest(
		payload,
		validate_array_semantics=validate_array_semantics,
	)
	return payload


def validate_multi_head_xy_neighbor_unanimous_target_manifest(
	payload: Mapping[str, object],
	*,
	validate_array_semantics: bool = True,
) -> None:
	"""Validate immutable hard-source bindings and unanimous replay semantics."""
	_common._required(
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
		'XY-neighbor-unanimous target manifest',
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
		raise ValueError('unsupported XY-neighbor-unanimous target manifest schema')
	if payload['smoothing'] != _smoothing_policy_payload():
		raise ValueError('XY-neighbor-unanimous smoothing policy differs')

	hard_path = _common._hashed(payload['source_hard_manifest'], 'source_hard_manifest')
	hard = _common._load_hard_source(
		hard_path,
		validate_array_semantics=validate_array_semantics,
	)
	heads = _common._mapping(payload['heads'], 'XY-neighbor-unanimous heads')
	if set(heads) != {str(k) for k in CANONICAL_KS}:
		raise ValueError('XY-neighbor-unanimous heads must contain K=6/8/10')
	for k in CANONICAL_KS:
		_validate_head(
			_common._mapping(heads[str(k)], f'XY-neighbor-unanimous head k={k}'),
			k=k,
			hard=hard,
			source_manifest=hard_path,
			validate_array_semantics=validate_array_semantics,
		)


def _preflight(
	config: MultiHeadXYNeighborUnanimousTargetExportConfig,
	*,
	only_missing: bool,
) -> _Preflight:
	"""Complete all source and owned-output checks before any write."""
	_common._validate_output_scope(config)
	hard = _common._load_hard_source(config.source_hard_manifest)
	_common._validate_source_is_outside_owned_bundle(config, hard)
	snapshot = _common._source_snapshot(config.source_hard_manifest, hard)
	bundle = _bundle_path(config)
	if not bundle.exists():
		action: _Action = 'QUARANTINE' if config.handoff_manifest.exists() else 'NEW'
		reason = (
			'handoff exists without a complete unanimous bundle'
			if action == 'QUARANTINE'
			else None
		)
		plans = _plans(action, reason)
	elif not bundle.is_dir() or not config.handoff_manifest.is_file():
		plans = _plans('QUARANTINE', 'unanimous bundle or handoff is incomplete')
	else:
		try:
			_existing_publication_matches(
				config,
				source_identity=_common._reference(config.source_hard_manifest),
			)
		except _common._ImmutableIdentityMismatchError as exc:
			plans = _plans('ERROR', str(exc))
		except (OSError, TypeError, ValueError) as exc:
			plans = _plans('QUARANTINE', str(exc))
		else:
			plans = _plans(
				'REUSE' if only_missing else 'ERROR',
				None if only_missing else 'complete output exists; use --only-missing',
			)
	return _Preflight(hard=hard, snapshot=snapshot, plans=plans)


def _existing_publication_matches(
	config: MultiHeadXYNeighborUnanimousTargetExportConfig,
	*,
	source_identity: Mapping[str, str],
) -> None:
	"""Classify a complete owned publication without allowing source drift."""
	payload = _common._read_json_object(
		config.handoff_manifest,
		'XY-neighbor-unanimous target handoff',
	)
	if payload.get('source_hard_manifest') != source_identity:
		raise _common._ImmutableIdentityMismatchError(
			'unanimous bundle source hard manifest identity differs from current source'
		)
	validate_multi_head_xy_neighbor_unanimous_target_manifest(payload)
	_common._validate_bundle_layout(payload, _bundle_path(config))


def _plans(
	action: _Action,
	reason: str | None,
) -> tuple[MultiHeadXYNeighborUnanimousTargetExportPlan, ...]:
	return tuple(
		MultiHeadXYNeighborUnanimousTargetExportPlan(k, action, reason)
		for k in CANONICAL_KS
	)


def _plan_error(plans: list[MultiHeadXYNeighborUnanimousTargetExportPlan]) -> str:
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
	"""Export one source-bound unanimous head below the staging bundle."""
	head_root = staging / f'k{k}'
	head_root.mkdir(parents=True, exist_ok=True)
	hard_surveys = _common._hard_surveys(hard, k=k)
	surveys: dict[str, object] = {}
	per_survey: dict[str, object] = {}
	for survey_id in sorted(hard_surveys):
		source = _common._mapping(
			hard_surveys[survey_id],
			f'source hard survey {survey_id}',
		)
		source_labels, source_valid, source_label_path, source_valid_path = (
			_common._source_arrays(source)
		)
		result = smooth_xy_neighbor_unanimous_hard_labels(
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
		_common._write_json(
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
			'labels': _common._array_reference(
				paths.labels,
				labels.shape,
				labels.dtype,
			),
			'confidence': _common._array_reference(
				paths.confidence,
				confidence.shape,
				confidence.dtype,
			),
			'valid_tokens': _common._array_reference(
				paths.valid_tokens,
				source_valid.shape,
				source_valid.dtype,
			),
			'metadata': _common._reference(paths.metadata),
			'source_hard_labels': _common._reference(source_label_path),
			'source_hard_valid_tokens': _common._reference(source_valid_path),
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
	_common._write_json(diagnostics_path, diagnostics_payload)
	head = {
		'surveys': surveys,
		'diagnostics': {
			**diagnostics_payload,
			'json': _common._reference(diagnostics_path),
		},
	}
	_common._write_json(head_root / 'head_metadata.json', head)


def _validate_staged_head(
	root: Path,
	*,
	k: int,
	hard: Mapping[str, object],
	source_manifest: Path,
) -> None:
	path = root / f'k{k}' / 'head_metadata.json'
	head = _common._read_json_object(path, f'XY-neighbor-unanimous head k={k}')
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
	_common._required(
		head,
		{'surveys', 'diagnostics'},
		f'XY-neighbor-unanimous head k={k}',
	)
	surveys = _common._mapping(head['surveys'], f'XY-neighbor-unanimous surveys k={k}')
	hard_surveys = _common._hard_surveys(hard, k=k)
	if set(surveys) != set(hard_surveys):
		raise ValueError(f'unanimous survey set differs from source for k={k}')

	expected_per_survey: dict[str, object] = {}
	for survey_id in sorted(hard_surveys):
		entry = _common._mapping(
			surveys[survey_id],
			f'unanimous survey k={k} {survey_id}',
		)
		_common._required(
			entry,
			{
				'labels',
				'confidence',
				'valid_tokens',
				'metadata',
				'source_hard_labels',
				'source_hard_valid_tokens',
			},
			f'unanimous survey k={k} {survey_id}',
		)
		labels = _common._load_output_array(entry['labels'], 'labels')
		confidence = _common._load_output_array(entry['confidence'], 'confidence')
		valid = _common._load_output_array(entry['valid_tokens'], 'valid_tokens')
		if (
			labels.dtype != np.int32
			or labels.ndim != 3
			or confidence.dtype != np.float32
			or confidence.shape != labels.shape
			or valid.dtype != np.bool_
			or valid.shape != labels.shape
		):
			raise ValueError('unanimous arrays have invalid shape or dtype')

		source = _common._mapping(
			hard_surveys[survey_id],
			f'source hard survey {survey_id}',
		)
		source_labels, source_valid, source_label_path, source_valid_path = (
			_common._source_arrays(source)
		)
		if entry['source_hard_labels'] != _common._reference(source_label_path):
			raise ValueError('unanimous source hard-label provenance differs')
		if entry['source_hard_valid_tokens'] != _common._reference(source_valid_path):
			raise ValueError('unanimous source valid-mask provenance differs')
		if labels.shape != source_labels.shape or valid.shape != source_valid.shape:
			raise ValueError('unanimous source and output grid shapes differ')

		metadata = _common._read_json_object(
			_common._hashed(entry['metadata'], 'metadata'),
			'unanimous pseudo-target metadata',
		)
		if validate_array_semantics:
			_common._validate_output_array_semantics(
				labels,
				confidence,
				valid,
				source_labels=source_labels,
				source_valid=source_valid,
				k=k,
			)
			result = smooth_xy_neighbor_unanimous_hard_labels(
				source_labels,
				source_valid,
			)
			if not np.array_equal(labels, result.labels):
				raise ValueError(
					'unanimous labels differ from the source-label smoothing '
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
				raise ValueError('unanimous pseudo-target metadata provenance differs')
		else:
			_validate_metadata_structure(metadata)

	_validate_head_diagnostics(
		head['diagnostics'],
		survey_ids=set(hard_surveys),
		k=k,
		expected_per_survey=(expected_per_survey if validate_array_semantics else None),
	)


def _metadata_provenance(
	*,
	source_manifest: Path,
	source_label_path: Path,
	source_valid_path: Path,
) -> dict[str, object]:
	return {
		'target_representation': TARGET_REPRESENTATION,
		'target_semantics': TARGET_SEMANTICS,
		'source_hard_manifest': _common._reference(source_manifest),
		'source_hard_labels': _common._reference(source_label_path),
		'source_hard_valid_tokens': _common._reference(source_valid_path),
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
	"""Reference-only metadata validation for lazy hard-target construction."""
	if metadata.get('artifact_type') != 'strat_hmm_pseudo_target':
		raise ValueError('unanimous pseudo-target metadata has wrong artifact type')
	if metadata.get('schema_version') != 1:
		raise ValueError('unanimous pseudo-target metadata has wrong schema version')
	source = _common._mapping(metadata.get('source'), 'unanimous metadata source')
	_common._required(
		source,
		{
			'target_representation',
			'target_semantics',
			'source_hard_manifest',
			'source_hard_labels',
			'source_hard_valid_tokens',
			'smoothing',
		},
		'unanimous metadata source',
	)
	if (
		source.get('target_representation') != TARGET_REPRESENTATION
		or source.get('target_semantics') != TARGET_SEMANTICS
		or source.get('smoothing') != _smoothing_policy_payload()
	):
		raise ValueError('unanimous pseudo-target metadata policy differs')
	for name in (
		'source_hard_manifest',
		'source_hard_labels',
		'source_hard_valid_tokens',
	):
		_common._hashed(source.get(name), f'unanimous metadata {name}')


def _survey_diagnostics(
	source_labels: np.ndarray,
	labels: np.ndarray,
	valid: np.ndarray,
	result: XYNeighborConsensusResult,
	*,
	k: int,
) -> dict[str, object]:
	"""Build replayable diagnostics with the established hard-target schema."""
	diagnostics = _common._survey_diagnostics(
		source_labels,
		labels,
		valid,
		result,
		k=k,
	)
	counts = np.asarray(diagnostics['state_occupancy_count'], dtype=np.int64)
	diagnostics['empty_output_state_count'] = int(np.count_nonzero(counts == 0))
	return diagnostics


def _aggregate_diagnostics(
	per_survey: Mapping[str, object],
	*,
	k: int,
) -> dict[str, object]:
	diagnostics = _common._aggregate_diagnostics(per_survey, k=k)
	counts = np.asarray(diagnostics['state_occupancy_count'], dtype=np.int64)
	diagnostics['empty_output_state_count'] = int(np.count_nonzero(counts == 0))
	return diagnostics


def _validate_head_diagnostics(
	value: object,
	*,
	survey_ids: set[str],
	k: int,
	expected_per_survey: Mapping[str, object] | None,
) -> None:
	"""Validate common diagnostics plus unanimous empty-state evidence."""
	diagnostics = _common._mapping(value, f'unanimous diagnostics k={k}')
	_common._required(
		diagnostics,
		{'per_survey', 'aggregate', 'json'},
		f'unanimous diagnostics k={k}',
	)
	per_survey = _common._mapping(
		diagnostics['per_survey'],
		'unanimous per-survey diagnostics',
	)
	if set(per_survey) != survey_ids:
		raise ValueError(f'unanimous diagnostics survey set differs for k={k}')
	persisted = _common._read_json_object(
		_common._hashed(diagnostics['json'], 'unanimous diagnostics JSON'),
		'unanimous diagnostics JSON',
	)
	if persisted != {
		'per_survey': diagnostics['per_survey'],
		'aggregate': diagnostics['aggregate'],
	}:
		raise ValueError('unanimous diagnostics JSON differs from head metadata')
	aggregate = _common._mapping(
		diagnostics['aggregate'],
		'unanimous aggregate diagnostics',
	)
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
			raise ValueError('unanimous diagnostics differ from source-label replay')


def _validate_diagnostic_structure(
	per_survey: Mapping[str, object],
	aggregate: Mapping[str, object],
	*,
	k: int,
) -> None:
	"""Require and verify the unanimous-only empty output-state count."""
	base_per_survey: dict[str, object] = {}
	base_aggregate: dict[str, object] = {}
	for name, raw in [*per_survey.items(), ('aggregate', aggregate)]:
		metrics = _common._mapping(raw, f'unanimous diagnostics {name}')
		expected_keys = {
			'valid_token_count',
			'invalid_token_count',
			'changed_token_count',
			'changed_fraction',
			'temporal_transition_counts',
			'source_state_occupancy_count',
			'state_occupancy_count',
			'ordered_path',
			'consensus_decisions',
			'empty_output_state_count',
		}
		if name == 'aggregate':
			expected_keys.add('survey_count')
		_common._required(metrics, expected_keys, f'unanimous diagnostics {name}')
		base = {
			key: item
			for key, item in metrics.items()
			if key != 'empty_output_state_count'
		}
		if name == 'aggregate':
			base_aggregate = base
		else:
			base_per_survey[str(name)] = base

	_common._validate_diagnostic_structure(base_per_survey, base_aggregate, k=k)
	for name, raw in [*per_survey.items(), ('aggregate', aggregate)]:
		metrics = _common._mapping(raw, f'unanimous diagnostics {name}')
		empty_count = metrics['empty_output_state_count']
		if (
			isinstance(empty_count, bool)
			or not isinstance(empty_count, int)
			or empty_count < 0
			or empty_count > k
		):
			raise ValueError(
				f'unanimous diagnostics {name} empty output state count is invalid'
			)
		counts = np.asarray(metrics['state_occupancy_count'], dtype=np.int64)
		if empty_count != int(np.count_nonzero(counts == 0)):
			raise ValueError(
				f'unanimous diagnostics {name} empty output state count differs'
			)


def _publish_manifest(
	config: MultiHeadXYNeighborUnanimousTargetExportConfig,
) -> None:
	if config.handoff_manifest.exists():
		raise FileExistsError(
			'unanimous handoff appeared during publication; refusing to overwrite it'
		)
	payload = _manifest_payload(config)
	validate_multi_head_xy_neighbor_unanimous_target_manifest(payload)
	_common._validate_bundle_layout(payload, _bundle_path(config))
	_common._write_json(config.handoff_manifest, payload)
	load_multi_head_xy_neighbor_unanimous_target_manifest(config.handoff_manifest)


def _manifest_payload(
	config: MultiHeadXYNeighborUnanimousTargetExportConfig,
) -> dict[str, object]:
	head_payloads: dict[str, object] = {}
	for k in CANONICAL_KS:
		head_payloads[str(k)] = _common._read_json_object(
			_bundle_path(config) / f'k{k}' / 'head_metadata.json',
			f'XY-neighbor-unanimous head k={k}',
		)
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'target_representation': TARGET_REPRESENTATION,
		'target_semantics': TARGET_SEMANTICS,
		'head_ks': list(CANONICAL_KS),
		'source_hard_manifest': _common._reference(config.source_hard_manifest),
		'smoothing': _smoothing_policy_payload(),
		'heads': head_payloads,
	}


def _bundle_path(config: MultiHeadXYNeighborUnanimousTargetExportConfig) -> Path:
	return config.output_root / _BUNDLE_DIRNAME


def _smoothing_policy_payload() -> dict[str, object]:
	payload = _common._json_compatible(
		XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_POLICY
	)
	if not isinstance(payload, dict):
		raise TypeError('unanimous smoothing policy must be a JSON object')
	return payload


_ordered_path_metrics = _common._ordered_path_metrics
_temporal_transition_count = _common._temporal_transition_count


__all__ = [
	'ARTIFACT_TYPE',
	'CANONICAL_KS',
	'SCHEMA_VERSION',
	'TARGET_REPRESENTATION',
	'TARGET_SEMANTICS',
	'MultiHeadXYNeighborUnanimousTargetExportConfig',
	'MultiHeadXYNeighborUnanimousTargetExportPlan',
	'export_multi_head_xy_neighbor_unanimous_targets',
	'load_multi_head_xy_neighbor_unanimous_target_manifest',
	'plan_multi_head_xy_neighbor_unanimous_target_exports',
	'resolve_multi_head_xy_neighbor_unanimous_target_export_config',
	'validate_multi_head_xy_neighbor_unanimous_target_manifest',
]
