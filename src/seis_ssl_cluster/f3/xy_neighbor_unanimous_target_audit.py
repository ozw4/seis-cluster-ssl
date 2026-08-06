"""Target-only audit for the immutable unanimous XY-neighbour successor.

The audit deliberately works only with the frozen hard source and the two
source-label target publications.  It is a publication boundary, so all three
inputs are fully replay-validated by their strict manifest loaders before a
GO/HOLD decision is made.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	load_multi_head_xy_neighbor_unanimous_target_manifest,
)

ARTIFACT_TYPE = 'f3_xy_neighbor_unanimous_target_audit'
SCHEMA_VERSION = 1
_CANONICAL_KS = (6, 8, 10)
_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'source_hard_manifest',
		'xy_neighbor_consensus_target_manifest',
		'xy_neighbor_unanimous_target_manifest',
		'output_path',
	}
)


@dataclass(frozen=True)
class F3XYNeighborUnanimousTargetAuditConfig:
	"""Closed locations for one immutable target-only audit."""

	artifact_root: Path
	source_hard_manifest: Path
	xy_neighbor_consensus_target_manifest: Path
	xy_neighbor_unanimous_target_manifest: Path
	output_path: Path


@dataclass(frozen=True)
class F3XYNeighborUnanimousTargetAuditResult:
	"""Computed target-only evidence and its immutable publication action."""

	payload: Mapping[str, object]
	output_path: Path
	action: str
	quarantine_path: Path | None


def f3_xy_neighbor_unanimous_target_audit_config_from_mapping(
	config: Mapping[str, object],
) -> F3XYNeighborUnanimousTargetAuditConfig:
	"""Resolve the deliberately non-extensible unanimous target-audit schema."""
	if not isinstance(config, Mapping):
		raise TypeError('unanimous target audit config must be a mapping')
	unknown = set(config) - _CONFIG_KEYS
	missing = _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown unanimous target audit keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing unanimous target audit keys: {sorted(missing)!r}')

	def path(
		name: str,
		*,
		must_exist: bool,
		directory: bool = False,
	) -> Path:
		value = config[name]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{name} must be a non-empty path string')
		resolved = Path(value).resolve()
		if must_exist and not (resolved.is_dir() if directory else resolved.is_file()):
			raise FileNotFoundError(f'{name} is missing: {resolved}')
		return resolved

	result = F3XYNeighborUnanimousTargetAuditConfig(
		artifact_root=path('artifact_root', must_exist=True, directory=True),
		source_hard_manifest=path('source_hard_manifest', must_exist=True),
		xy_neighbor_consensus_target_manifest=path(
			'xy_neighbor_consensus_target_manifest', must_exist=True
		),
		xy_neighbor_unanimous_target_manifest=path(
			'xy_neighbor_unanimous_target_manifest', must_exist=True
		),
		output_path=path('output_path', must_exist=False),
	)
	if not result.artifact_root.is_dir():
		raise NotADirectoryError(
			f'artifact_root is not a directory: {result.artifact_root}'
		)
	return result


def load_f3_xy_neighbor_unanimous_target_audit_config(
	path: str | Path,
) -> F3XYNeighborUnanimousTargetAuditConfig:
	"""Load the closed unanimous target-audit YAML configuration."""
	return f3_xy_neighbor_unanimous_target_audit_config_from_mapping(load_config(path))


def audit_f3_xy_neighbor_unanimous_targets(
	config: F3XYNeighborUnanimousTargetAuditConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3XYNeighborUnanimousTargetAuditResult:
	"""Fully validate three targets, then immutably write GO/HOLD evidence."""
	payload = _audit_payload(config)
	if dry_run:
		return F3XYNeighborUnanimousTargetAuditResult(
			payload, config.output_path, 'DRY_RUN', None
		)
	if config.output_path.exists():
		if not config.output_path.is_file():
			raise FileExistsError(f'audit output is not a file: {config.output_path}')
		try:
			existing = load_f3_xy_neighbor_unanimous_target_audit(config.output_path)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == payload:
			if only_missing:
				return F3XYNeighborUnanimousTargetAuditResult(
					payload, config.output_path, 'REUSE_COMPLETED', None
				)
			raise FileExistsError(
				'unanimous target audit exists; use --only-missing: '
				f'{config.output_path}'
			)
		if not quarantine_invalid:
			raise ValueError(
				'existing unanimous target audit is stale or invalid; '
				'use --quarantine-invalid to replace it'
			)
		quarantine = _quarantine_invalid(config.output_path)
		_atomic_json(config.output_path, payload)
	else:
		quarantine = None
		_atomic_json(config.output_path, payload)
	return F3XYNeighborUnanimousTargetAuditResult(
		payload, config.output_path, 'WRITTEN', quarantine
	)


def load_f3_xy_neighbor_unanimous_target_audit(
	path: str | Path,
) -> Mapping[str, object]:
	"""Load a complete target-only audit without accepting another identity."""
	payload = _mapping(_json(Path(path)), 'unanimous target audit')
	if set(payload) != {
		'artifact_type',
		'schema_version',
		'status',
		'source_hard_manifest',
		'xy_neighbor_consensus_target_manifest',
		'xy_neighbor_unanimous_target_manifest',
		'per_k',
		'go_conditions',
	}:
		raise ValueError('unanimous target audit keys mismatch')
	if (
		payload['artifact_type'] != ARTIFACT_TYPE
		or payload['schema_version'] != SCHEMA_VERSION
		or payload['status'] not in {'XYUNANIM_TARGET_GO', 'XYUNANIM_TARGET_HOLD'}
	):
		raise ValueError('unanimous target audit identity or status mismatch')
	for key in (
		'source_hard_manifest',
		'xy_neighbor_consensus_target_manifest',
		'xy_neighbor_unanimous_target_manifest',
	):
		_referenced_path(
			payload[key],
			f'unanimous target audit {key}',
			allow_array_descriptor=False,
		)
	per_k = _mapping(payload['per_k'], 'unanimous target audit per_k')
	if set(per_k) != {str(k) for k in _CANONICAL_KS}:
		raise ValueError('unanimous target audit must contain K=6/8/10')
	conditions = _mapping(payload['go_conditions'], 'unanimous target audit conditions')
	if set(conditions) != {str(k) for k in _CANONICAL_KS}:
		raise ValueError('unanimous target audit condition heads mismatch')
	for k in _CANONICAL_KS:
		_evidence_structure(_mapping(per_k[str(k)], f'unanimous target audit K={k}'))
		evidence = _mapping(per_k[str(k)], f'unanimous target audit K={k}')
		condition = _mapping(
			conditions[str(k)], f'unanimous target audit conditions K={k}'
		)
		_condition_structure(condition)
		if condition != _conditions_from_evidence(evidence):
			raise ValueError('unanimous target audit conditions mismatch evidence')
	expected_status = _status_from_conditions(conditions)
	if payload['status'] != expected_status:
		raise ValueError('unanimous target audit status mismatch conditions')
	return payload


def validate_f3_xy_neighbor_unanimous_target_audit(
	config: F3XYNeighborUnanimousTargetAuditConfig,
) -> Mapping[str, object]:
	"""Replay and bind a persisted audit to this exact target configuration."""
	payload = load_f3_xy_neighbor_unanimous_target_audit(config.output_path)
	for key, path in (
		('source_hard_manifest', config.source_hard_manifest),
		(
			'xy_neighbor_consensus_target_manifest',
			config.xy_neighbor_consensus_target_manifest,
		),
		(
			'xy_neighbor_unanimous_target_manifest',
			config.xy_neighbor_unanimous_target_manifest,
		),
	):
		if payload[key] != _identity(path):
			raise ValueError(f'unanimous target audit {key} identity differs')
	# This is an explicit target-validation boundary.  Replaying the three
	# immutable inputs prevents a syntactically valid, post-publication edit from
	# changing a GO/HOLD decision without being detected.
	if payload != _audit_payload(config):
		raise ValueError('unanimous target audit differs from replayed evidence')
	return payload


def replay_f3_xy_neighbor_unanimous_target_audit(
	path: str | Path,
	*,
	artifact_root: str | Path,
) -> Mapping[str, object]:
	"""Replay a persisted audit using the immutable references it records."""
	output_path = Path(path).resolve()
	root = Path(artifact_root).resolve()
	payload = load_f3_xy_neighbor_unanimous_target_audit(output_path)

	def referenced(name: str) -> Path:
		entry = _mapping(payload[name], f'unanimous target audit {name}')
		return Path(_string(entry['path'], f'unanimous target audit {name}.path'))

	return validate_f3_xy_neighbor_unanimous_target_audit(
		F3XYNeighborUnanimousTargetAuditConfig(
			artifact_root=root,
			source_hard_manifest=referenced('source_hard_manifest'),
			xy_neighbor_consensus_target_manifest=referenced(
				'xy_neighbor_consensus_target_manifest'
			),
			xy_neighbor_unanimous_target_manifest=referenced(
				'xy_neighbor_unanimous_target_manifest'
			),
			output_path=output_path,
		)
	)


def _audit_payload(config: F3XYNeighborUnanimousTargetAuditConfig) -> dict[str, object]:
	"""Build replay-validated target-only evidence for all canonical heads."""
	# Both target loaders perform complete source-array replay.  Check their
	# source references against the configured frozen source before reading any
	# arrays for the comparative audit.
	consensus = load_multi_head_xy_neighbor_consensus_target_manifest(
		config.xy_neighbor_consensus_target_manifest
	)
	unanimous = load_multi_head_xy_neighbor_unanimous_target_manifest(
		config.xy_neighbor_unanimous_target_manifest
	)
	source_identity = _identity(config.source_hard_manifest)
	if consensus.get('source_hard_manifest') != source_identity:
		raise ValueError('3-of-4 target source hard identity differs')
	if unanimous.get('source_hard_manifest') != source_identity:
		raise ValueError('unanimous target source hard identity differs')
	source = _load_source_manifest(config.source_hard_manifest)
	per_k: dict[str, object] = {}
	conditions: dict[str, object] = {}
	for k in _CANONICAL_KS:
		evidence, head_conditions = _head_evidence(
			source=source,
			consensus=consensus,
			unanimous=unanimous,
			k=k,
		)
		per_k[str(k)] = evidence
		conditions[str(k)] = head_conditions
	status = _status_from_conditions(conditions)
	payload = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': status,
		'source_hard_manifest': source_identity,
		'xy_neighbor_consensus_target_manifest': _identity(
			config.xy_neighbor_consensus_target_manifest
		),
		'xy_neighbor_unanimous_target_manifest': _identity(
			config.xy_neighbor_unanimous_target_manifest
		),
		'per_k': per_k,
		'go_conditions': conditions,
	}
	# This catches an accidental NaN/Inf before it can become an immutable gate.
	_validate_finite_tree(payload, context='unanimous target audit')
	return payload


def _load_source_manifest(path: Path) -> Mapping[str, object]:
	payload = _mapping(_json(path), 'source hard manifest')
	if (
		payload.get('artifact_type') != 'strat_hmm_multi_head_target_manifest'
		or payload.get('schema_version') not in {1, 2}
		or payload.get('head_ks') != list(_CANONICAL_KS)
		or payload.get('ordering_orientation') != 'increasing_downward'
	):
		raise ValueError('source hard manifest identity is unsupported')
	return payload


def _head_evidence(
	*,
	source: Mapping[str, object],
	consensus: Mapping[str, object],
	unanimous: Mapping[str, object],
	k: int,
) -> tuple[dict[str, object], dict[str, bool]]:
	source_surveys = _surveys(source, k=k, label='source')
	consensus_surveys = _surveys(consensus, k=k, label='3-of-4')
	unanimous_surveys = _surveys(unanimous, k=k, label='unanimous')
	if set(source_surveys) != set(consensus_surveys) or set(source_surveys) != set(
		unanimous_surveys
	):
		raise ValueError(f'target survey sets differ for K={k}')
	entries = []
	for survey_id in sorted(source_surveys):
		source_entry = _mapping(source_surveys[survey_id], f'source survey {survey_id}')
		consensus_entry = _mapping(
			consensus_surveys[survey_id], f'3-of-4 survey {survey_id}'
		)
		unanimous_entry = _mapping(
			unanimous_surveys[survey_id], f'unanimous survey {survey_id}'
		)
		source_labels = _load_array(source_entry['labels'], 'source labels')
		source_valid = _load_array(source_entry['valid_tokens'], 'source valid tokens')
		consensus_labels = _load_array(consensus_entry['labels'], '3-of-4 labels')
		consensus_valid = _load_array(
			consensus_entry['valid_tokens'], '3-of-4 valid tokens'
		)
		unanimous_labels = _load_array(unanimous_entry['labels'], 'unanimous labels')
		unanimous_valid = _load_array(
			unanimous_entry['valid_tokens'], 'unanimous valid tokens'
		)
		entries.append(
			_survey_evidence(
				source_labels,
				source_valid,
				consensus_labels,
				consensus_valid,
				unanimous_labels,
				unanimous_valid,
				k=k,
			)
		)
	evidence = _merge_head_evidence(entries, k=k)
	conditions = _conditions_from_evidence(evidence)
	return evidence, conditions


def _conditions_from_evidence(evidence: Mapping[str, object]) -> dict[str, bool]:
	"""Derive the fixed GO predicates instead of trusting persisted booleans."""
	unanimous = _mapping(evidence['unanimous'], 'unanimous')
	source = _mapping(evidence['source'], 'source')
	subset = _mapping(evidence['subset_evidence'], 'subset evidence')
	return {
		'valid_masks_exact': bool(evidence['valid_masks_exact']),
		'ordered_path_violation_count_zero': (
			int(unanimous['ordered_path_violation_count']) == 0
		),
		'empty_output_state_count_zero': (
			int(unanimous['empty_output_state_count']) == 0
		),
		'changed_token_count_positive': int(evidence['unanimous_changed_token_count'])
		> 0,
		'unanimous_combined_xy_disagreement_lt_source': (
			int(
				_mapping(unanimous['spatial'], 'unanimous spatial')['combined'][
					'disagreement_count'
				]
			)
			< int(
				_mapping(source['spatial'], 'source spatial')['combined'][
					'disagreement_count'
				]
			)
		),
		'unanimous_changed_mask_subset_of_3_of_4': bool(subset['changed_mask_subset']),
		'unanimous_output_equals_3_of_4_at_unanimous_changes': bool(
			subset['label_parity']
		),
		'arrays_and_diagnostics_finite_and_consistent': bool(
			evidence['arrays_and_diagnostics_finite_and_consistent']
		),
	}


def _status_from_conditions(conditions: Mapping[str, Mapping[str, bool]]) -> str:
	"""Apply the fixed all-head target-only GO rule."""
	return (
		'XYUNANIM_TARGET_GO'
		if all(all(values.values()) for values in conditions.values())
		else 'XYUNANIM_TARGET_HOLD'
	)


def _surveys(
	payload: Mapping[str, object], *, k: int, label: str
) -> Mapping[str, object]:
	heads = _mapping(payload.get('heads'), f'{label} heads')
	head = _mapping(heads.get(str(k)), f'{label} head K={k}')
	return _mapping(head.get('surveys'), f'{label} surveys K={k}')


def _survey_evidence(  # noqa: PLR0913
	source: np.ndarray,
	valid: np.ndarray,
	consensus: np.ndarray,
	consensus_valid: np.ndarray,
	unanimous: np.ndarray,
	unanimous_valid: np.ndarray,
	*,
	k: int,
) -> dict[str, object]:
	if (
		source.shape != valid.shape
		or consensus.shape != source.shape
		or unanimous.shape != source.shape
		or consensus_valid.shape != source.shape
		or unanimous_valid.shape != source.shape
		or valid.dtype != np.bool_
		or consensus_valid.dtype != np.bool_
		or unanimous_valid.dtype != np.bool_
	):
		raise ValueError('target audit arrays have incompatible source shapes or masks')
	if source.ndim != 3:
		raise ValueError('target audit arrays must be three dimensional')
	if np.any(source[valid] < 0) or np.any(source[valid] >= k):
		raise ValueError('source hard labels fall outside K')
	if np.any(consensus[valid] < 0) or np.any(consensus[valid] >= k):
		raise ValueError('3-of-4 labels fall outside K')
	if np.any(unanimous[valid] < 0) or np.any(unanimous[valid] >= k):
		raise ValueError('unanimous labels fall outside K')
	if not (
		np.isfinite(source).all()
		and np.isfinite(consensus).all()
		and np.isfinite(unanimous).all()
	):
		raise ValueError('target audit labels contain non-finite values')
	changed_consensus = valid & (consensus != source)
	changed_unanimous = valid & (unanimous != source)
	source_metrics = _label_metrics(source, valid, k=k)
	consensus_metrics = _label_metrics(consensus, valid, k=k)
	unanimous_metrics = _label_metrics(unanimous, valid, k=k)
	for metrics, changed in (
		(consensus_metrics, changed_consensus),
		(unanimous_metrics, changed_unanimous),
	):
		changed_count = int(np.count_nonzero(changed))
		metrics['changed_token_count'] = changed_count
		metrics['changed_fraction'] = changed_count / max(int(valid.sum()), 1)
	return {
		'valid_masks_exact': bool(
			np.array_equal(valid, consensus_valid)
			and np.array_equal(valid, unanimous_valid)
		),
		'source': source_metrics,
		'xy_neighbor_consensus': consensus_metrics,
		'unanimous': unanimous_metrics,
		'consensus_changed_token_count': int(np.count_nonzero(changed_consensus)),
		'unanimous_changed_token_count': int(np.count_nonzero(changed_unanimous)),
		'subset_evidence': {
			'changed_mask_subset': bool(np.all(~changed_unanimous | changed_consensus)),
			'label_parity': bool(
				np.array_equal(
					unanimous[changed_unanimous], consensus[changed_unanimous]
				)
			),
			'xy_neighbor_consensus_only_changed_token_count': int(
				np.count_nonzero(changed_consensus & ~changed_unanimous)
			),
		},
	}


def _merge_head_evidence(
	entries: Sequence[Mapping[str, object]], *, k: int
) -> dict[str, object]:
	if not entries:
		raise ValueError(f'target audit requires at least one survey for K={k}')
	valid_masks_exact = all(bool(item['valid_masks_exact']) for item in entries)
	source = _merge_label_metrics(
		[_mapping(item['source'], 'source') for item in entries], k=k
	)
	consensus = _merge_label_metrics(
		[_mapping(item['xy_neighbor_consensus'], '3-of-4') for item in entries], k=k
	)
	unanimous = _merge_label_metrics(
		[_mapping(item['unanimous'], 'unanimous') for item in entries], k=k
	)
	consensus_changed = sum(
		int(item['consensus_changed_token_count']) for item in entries
	)
	unanimous_changed = sum(
		int(item['unanimous_changed_token_count']) for item in entries
	)
	consensus_only = sum(
		int(
			_mapping(item['subset_evidence'], 'subset')[
				'xy_neighbor_consensus_only_changed_token_count'
			]
		)
		for item in entries
	)
	subset = all(
		bool(_mapping(item['subset_evidence'], 'subset')['changed_mask_subset'])
		for item in entries
	)
	parity = all(
		bool(_mapping(item['subset_evidence'], 'subset')['label_parity'])
		for item in entries
	)
	return {
		'valid_masks_exact': valid_masks_exact,
		'source': source,
		'xy_neighbor_consensus': consensus,
		'unanimous': unanimous,
		'consensus_changed_token_count': consensus_changed,
		'consensus_changed_fraction': consensus_changed
		/ max(int(source['valid_token_count']), 1),
		'unanimous_changed_token_count': unanimous_changed,
		'unanimous_changed_fraction': unanimous_changed
		/ max(int(source['valid_token_count']), 1),
		'subset_evidence': {
			'changed_mask_subset': subset,
			'label_parity': parity,
			'unanimous_changed_count_over_consensus_changed_count': unanimous_changed
			/ max(consensus_changed, 1),
			'retained_change_fraction': unanimous_changed / max(consensus_changed, 1),
			'xy_neighbor_consensus_only_changed_token_count': consensus_only,
		},
		'arrays_and_diagnostics_finite_and_consistent': True,
	}


def _label_metrics(
	labels: np.ndarray, valid: np.ndarray, *, k: int
) -> dict[str, object]:
	valid_count = int(np.count_nonzero(valid))
	occupancy = np.bincount(labels[valid], minlength=k).astype(int).tolist()
	return {
		'valid_token_count': valid_count,
		'invalid_token_count': int(valid.size - valid_count),
		'changed_token_count': 0,
		'changed_fraction': 0.0,
		'state_occupancy_count': occupancy,
		'empty_output_state_count': int(sum(count == 0 for count in occupancy)),
		'temporal_transition_count': _temporal_transitions(labels, valid),
		'ordered_path_violation_count': _ordered_path_violations(labels, valid),
		'spatial': _spatial_disagreement(labels, valid),
	}


def _merge_label_metrics(
	entries: Sequence[Mapping[str, object]], *, k: int
) -> dict[str, object]:
	valid_count = sum(int(item['valid_token_count']) for item in entries)
	invalid_count = sum(int(item['invalid_token_count']) for item in entries)
	changed_count = sum(int(item['changed_token_count']) for item in entries)
	occupancy = np.zeros(k, dtype=np.int64)
	for item in entries:
		occupancy += np.asarray(item['state_occupancy_count'], dtype=np.int64)
	spatial = _merge_spatial([_mapping(item['spatial'], 'spatial') for item in entries])
	return {
		'valid_token_count': valid_count,
		'invalid_token_count': invalid_count,
		'changed_token_count': changed_count,
		'changed_fraction': changed_count / max(valid_count, 1),
		'state_occupancy_count': occupancy.astype(int).tolist(),
		'empty_output_state_count': int(np.count_nonzero(occupancy == 0)),
		'temporal_transition_count': sum(
			int(item['temporal_transition_count']) for item in entries
		),
		'ordered_path_violation_count': sum(
			int(item['ordered_path_violation_count']) for item in entries
		),
		'spatial': spatial,
	}


def _spatial_disagreement(labels: np.ndarray, valid: np.ndarray) -> dict[str, object]:
	return {
		'x_edges': _edge_disagreement(labels[:-1], labels[1:], valid[:-1] & valid[1:]),
		'y_edges': _edge_disagreement(
			labels[:, :-1], labels[:, 1:], valid[:, :-1] & valid[:, 1:]
		),
		'combined': _merge_edges(
			[
				_edge_disagreement(labels[:-1], labels[1:], valid[:-1] & valid[1:]),
				_edge_disagreement(
					labels[:, :-1], labels[:, 1:], valid[:, :-1] & valid[:, 1:]
				),
			]
		),
	}


def _edge_disagreement(
	left: np.ndarray, right: np.ndarray, edge_valid: np.ndarray
) -> dict[str, object]:
	count = int(np.count_nonzero(edge_valid))
	disagreement = int(np.count_nonzero((left != right) & edge_valid))
	return {
		'valid_edge_count': count,
		'disagreement_count': disagreement,
		'disagreement_fraction': disagreement / count if count else 0.0,
	}


def _merge_spatial(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
	return {
		'x_edges': _merge_edges(
			[_mapping(item['x_edges'], 'x edges') for item in entries]
		),
		'y_edges': _merge_edges(
			[_mapping(item['y_edges'], 'y edges') for item in entries]
		),
		'combined': _merge_edges(
			[_mapping(item['combined'], 'combined edges') for item in entries]
		),
	}


def _merge_edges(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
	count = sum(int(item['valid_edge_count']) for item in entries)
	disagreement = sum(int(item['disagreement_count']) for item in entries)
	return {
		'valid_edge_count': count,
		'disagreement_count': disagreement,
		'disagreement_fraction': disagreement / count if count else 0.0,
	}


def _temporal_transitions(labels: np.ndarray, valid: np.ndarray) -> int:
	return sum(
		int(np.count_nonzero(trace[1:] != trace[:-1]))
		for x in range(labels.shape[0])
		for y in range(labels.shape[1])
		if (trace := labels[x, y, valid[x, y]]).size > 1
	)


def _ordered_path_violations(labels: np.ndarray, valid: np.ndarray) -> int:
	return sum(
		int(np.count_nonzero(np.diff(trace) < 0))
		for x in range(labels.shape[0])
		for y in range(labels.shape[1])
		if (trace := labels[x, y, valid[x, y]]).size > 1
	)


def _evidence_structure(value: Mapping[str, object]) -> None:
	required = {
		'valid_masks_exact',
		'source',
		'xy_neighbor_consensus',
		'unanimous',
		'consensus_changed_token_count',
		'consensus_changed_fraction',
		'unanimous_changed_token_count',
		'unanimous_changed_fraction',
		'subset_evidence',
		'arrays_and_diagnostics_finite_and_consistent',
	}
	if set(value) != required:
		raise ValueError('unanimous target audit head evidence keys mismatch')
	for key in ('source', 'xy_neighbor_consensus', 'unanimous'):
		_label_metrics_structure(_mapping(value[key], f'audit {key}'))
	subset = _mapping(value['subset_evidence'], 'audit subset evidence')
	if set(subset) != {
		'changed_mask_subset',
		'label_parity',
		'unanimous_changed_count_over_consensus_changed_count',
		'retained_change_fraction',
		'xy_neighbor_consensus_only_changed_token_count',
	}:
		raise ValueError('unanimous target audit subset evidence keys mismatch')
	_validate_finite_tree(value, context='unanimous target audit evidence')


def _label_metrics_structure(value: Mapping[str, object]) -> None:
	if set(value) != {
		'valid_token_count',
		'invalid_token_count',
		'changed_token_count',
		'changed_fraction',
		'state_occupancy_count',
		'empty_output_state_count',
		'temporal_transition_count',
		'ordered_path_violation_count',
		'spatial',
	}:
		raise ValueError('unanimous target audit label metric keys mismatch')
	spatial = _mapping(value['spatial'], 'audit spatial metrics')
	if set(spatial) != {'x_edges', 'y_edges', 'combined'}:
		raise ValueError('unanimous target audit spatial keys mismatch')
	for edge in spatial.values():
		if set(_mapping(edge, 'audit edge metrics')) != {
			'valid_edge_count',
			'disagreement_count',
			'disagreement_fraction',
		}:
			raise ValueError('unanimous target audit edge keys mismatch')


def _condition_structure(value: Mapping[str, object]) -> None:
	if set(value) != {
		'valid_masks_exact',
		'ordered_path_violation_count_zero',
		'empty_output_state_count_zero',
		'changed_token_count_positive',
		'unanimous_combined_xy_disagreement_lt_source',
		'unanimous_changed_mask_subset_of_3_of_4',
		'unanimous_output_equals_3_of_4_at_unanimous_changes',
		'arrays_and_diagnostics_finite_and_consistent',
	} or not all(isinstance(item, bool) for item in value.values()):
		raise ValueError('unanimous target audit GO conditions are invalid')


def _load_array(reference: object, label: str) -> np.ndarray:
	path = _referenced_path(reference, label, allow_array_descriptor=True)
	return np.load(path, mmap_mode='r', allow_pickle=False)


def _referenced_path(
	value: object,
	label: str,
	*,
	allow_array_descriptor: bool,
) -> Path:
	entry = _mapping(value, label)
	allowed = {'path', 'sha256'} | (
		{'shape', 'dtype'} if allow_array_descriptor else set()
	)
	if not {'path', 'sha256'} <= set(entry) or set(entry) - allowed:
		raise ValueError(f'{label} reference keys mismatch')
	path = Path(_string(entry['path'], f'{label}.path'))
	if not path.is_file() or file_sha256(path) != _string(
		entry['sha256'], f'{label}.sha256'
	):
		raise ValueError(f'{label} identity mismatch')
	return path


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _json(path: Path) -> object:
	return json.loads(path.read_text(encoding='utf-8'))


def _validate_finite_tree(value: object, *, context: str) -> None:
	if isinstance(value, Mapping):
		for item in value.values():
			_validate_finite_tree(item, context=context)
	elif isinstance(value, list):
		for item in value:
			_validate_finite_tree(item, context=context)
	elif isinstance(value, float) and not math.isfinite(value):
		raise ValueError(f'{context} contains a non-finite value')


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
	temporary = Path(temporary_name)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as handle:
			json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
			handle.write('\n')
			handle.flush()
			os.fsync(handle.fileno())
		temporary.replace(path)
	except BaseException:
		temporary.unlink(missing_ok=True)
		raise


def _quarantine_invalid(path: Path) -> Path:
	digest = file_sha256(path)[:16]
	quarantine = path.with_name(f'{path.stem}.invalid-{digest}{path.suffix}')
	if quarantine.exists():
		raise FileExistsError(f'unanimous target audit quarantine exists: {quarantine}')
	path.replace(quarantine)
	return quarantine


__all__ = [
	'ARTIFACT_TYPE',
	'SCHEMA_VERSION',
	'F3XYNeighborUnanimousTargetAuditConfig',
	'F3XYNeighborUnanimousTargetAuditResult',
	'audit_f3_xy_neighbor_unanimous_targets',
	'f3_xy_neighbor_unanimous_target_audit_config_from_mapping',
	'load_f3_xy_neighbor_unanimous_target_audit',
	'load_f3_xy_neighbor_unanimous_target_audit_config',
	'replay_f3_xy_neighbor_unanimous_target_audit',
	'validate_f3_xy_neighbor_unanimous_target_audit',
]
