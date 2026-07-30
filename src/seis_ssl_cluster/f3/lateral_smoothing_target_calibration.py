"""Target-only calibration and immutable selection for F3 M5-LS targets.

The calibration is deliberately upstream of pretraining.  It consumes only the
frozen hard HMM targets, their ordered-path posterior publication, and the
three preregistered lateral-target candidates.  It never imports or opens an
F3 lithology/facies artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.pretraining import _multi_head_target_hashes
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.stratigraphy import lateral_targets
from seis_ssl_cluster.stratigraphy.lateral_smoothing import (
	LATERAL_SMOOTHING_SEMANTICS,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.state_posterior import (
	load_multi_head_state_posterior_manifest,
)

_CANONICAL_CANDIDATES = (
	('beta010', 0.10),
	('beta025', 0.25),
	('beta050', 0.50),
)
_CANONICAL_KS = (6, 8, 10)
_SELECTION_POLICY = 'target_only_smallest_eligible_beta_v1'
_HANDOFF_TYPE = 'f3_m5_lateral_target_calibration'
_REPORT_TYPE = 'f3_m5_lateral_target_calibration_report'
_ELIGIBILITY_REASON_LABELS = {
	'diagnostics_well_formed': 'persisted diagnostics are nonfinite or malformed',
	'ordered_path': 'ordered path has a violation or reverse decrease',
	'nonempty_state_occupancy': 'lateral state occupancy has an empty state',
	'changed_tokens_positive': 'lateral target changes zero tokens',
	'affinity_weighted_disagreement_reduced': (
		'affinity-weighted XY disagreement is not reduced'
	),
	'highest_affinity_quartile_reduced': (
		'highest-affinity XY disagreement is not reduced'
	),
	'boundary_change_not_less_than_interior': (
		'boundary-adjacent changed fraction is below interior'
	),
	'transition_count_not_increased': (
		'lateral transition count exceeds recomputed source transition count'
	),
}
_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'source_hard_manifest',
		'source_posterior_manifest',
		'candidate_manifests',
		'selected_manifest',
		'calibration_handoff',
		'calibration_report',
		'hard_full_config',
		'lateral_smoke_config',
		'lateral_full_config',
	}
)


@dataclass(frozen=True)
class F3M5LateralTargetCandidate:
	"""One closed, preregistered beta candidate."""

	name: str
	beta: float
	manifest: Path


@dataclass(frozen=True)
class F3M5LateralTargetCalibrationConfig:
	"""All fixed inputs and publication paths for target-only M5-LS selection."""

	artifact_root: Path
	source_hard_manifest: Path
	source_posterior_manifest: Path
	candidates: tuple[F3M5LateralTargetCandidate, ...]
	selected_manifest: Path
	calibration_handoff: Path
	calibration_report: Path
	hard_full_config: Path
	lateral_smoke_config: Path
	lateral_full_config: Path


@dataclass(frozen=True)
class F3M5LateralTargetCalibrationResult:
	"""The deterministic calibration decision and its immutable publications."""

	status: str
	selected_beta: float | None
	evidence: Mapping[str, object]
	published_selected_manifest: Path | None
	published_handoff: Path | None
	published_report: Path | None


def f3_m5_lateral_target_calibration_config_from_mapping(
	config: Mapping[str, object],
) -> F3M5LateralTargetCalibrationConfig:
	"""Resolve the intentionally closed M5-LS target-calibration schema."""
	if not isinstance(config, Mapping):
		raise TypeError('M5-LS calibration config must be a mapping')
	_validate_config_keys(config)
	artifact_root = _config_path(config, 'artifact_root', exists=False)
	candidates = _config_candidates(config)
	result = F3M5LateralTargetCalibrationConfig(
		artifact_root=artifact_root,
		source_hard_manifest=_config_path(config, 'source_hard_manifest', exists=True),
		source_posterior_manifest=_config_path(
			config, 'source_posterior_manifest', exists=True
		),
		candidates=candidates,
		selected_manifest=_config_path(config, 'selected_manifest', exists=False),
		calibration_handoff=_config_path(config, 'calibration_handoff', exists=False),
		calibration_report=_config_path(config, 'calibration_report', exists=False),
		hard_full_config=_config_path(config, 'hard_full_config', exists=True),
		lateral_smoke_config=_config_path(config, 'lateral_smoke_config', exists=True),
		lateral_full_config=_config_path(config, 'lateral_full_config', exists=True),
	)
	_validate_calibration_paths(result)
	return result


def _validate_config_keys(config: Mapping[str, object]) -> None:
	unknown = set(config) - _CONFIG_KEYS
	missing = _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown M5-LS calibration config keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing M5-LS calibration config keys: {sorted(missing)!r}')


def _config_path(
	config: Mapping[str, object],
	key: str,
	*,
	exists: bool,
) -> Path:
	value = config[key]
	if not isinstance(value, str) or not value:
		raise TypeError(f'{key} must be a non-empty path string')
	raw = Path(value)
	if key == 'artifact_root' and not raw.is_absolute():
		raise ValueError('artifact_root must be absolute')
	path = raw.resolve()
	if exists and not path.is_file():
		raise FileNotFoundError(f'{key} is missing: {path}')
	return path


def _config_candidates(
	config: Mapping[str, object],
) -> tuple[F3M5LateralTargetCandidate, ...]:
	candidate_paths = config['candidate_manifests']
	if not isinstance(candidate_paths, Mapping):
		raise TypeError('candidate_manifests must be a mapping')
	expected_names = [name for name, _ in _CANONICAL_CANDIDATES]
	if list(candidate_paths) != expected_names:
		raise ValueError(
			'candidate_manifests must contain exactly the ordered canonical '
			f'candidate set {expected_names!r}'
		)
	result = tuple(
		F3M5LateralTargetCandidate(
			name=name,
			beta=beta,
			manifest=_mapping_path(
				candidate_paths,
				name,
				label=f'candidate_manifests.{name}',
			),
		)
		for name, beta in _CANONICAL_CANDIDATES
	)
	for candidate in result:
		if not candidate.manifest.is_file():
			raise FileNotFoundError(
				f'candidate_manifests.{candidate.name} is missing: {candidate.manifest}'
			)
	return result


def _validate_calibration_paths(config: F3M5LateralTargetCalibrationConfig) -> None:
	for label, path in (
		('source_hard_manifest', config.source_hard_manifest),
		('source_posterior_manifest', config.source_posterior_manifest),
		*(
			(f'candidate_manifests.{candidate.name}', candidate.manifest)
			for candidate in config.candidates
		),
		('selected_manifest', config.selected_manifest),
		('calibration_handoff', config.calibration_handoff),
		('calibration_report', config.calibration_report),
	):
		ensure_under_root(path, root=config.artifact_root, label=label)
	output_paths = (
		config.selected_manifest,
		config.calibration_handoff,
		config.calibration_report,
	)
	if len(set(output_paths)) != len(output_paths):
		raise ValueError('selected manifest, handoff, and report paths must differ')
	if any(
		config.selected_manifest == candidate.manifest
		for candidate in config.candidates
	):
		raise ValueError('selected_manifest must not be a candidate manifest path')


def load_f3_m5_lateral_target_calibration_config(
	path: str | Path,
) -> F3M5LateralTargetCalibrationConfig:
	"""Load the closed YAML calibration configuration."""
	return f3_m5_lateral_target_calibration_config_from_mapping(load_config(path))


def calibrate_f3_m5_lateral_targets(
	config: F3M5LateralTargetCalibrationConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3M5LateralTargetCalibrationResult:
	"""Validate the frozen candidates and atomically publish one selected target.

	A malformed candidate is an ineligible candidate, not a reason to relax the
	selection contract.  In contrast, a malformed source or beta-zero parity
	failure is a calibration failure and no evidence is published.
	"""
	try:
		evidence = _calibration_evidence(config)
		status = str(evidence['status'])
		selected_beta = _selected_beta_from_evidence(evidence)
		if dry_run:
			return F3M5LateralTargetCalibrationResult(
				status=status,
				selected_beta=selected_beta,
				evidence=evidence,
				published_selected_manifest=None,
				published_handoff=None,
				published_report=None,
			)

		handoff = _handoff_payload(evidence)
		report = _report_payload(evidence)
		selected_candidate = _selected_candidate(config, selected_beta)
		candidate_bytes: bytes | None = None
		if selected_candidate is not None:
			candidate_bytes = _candidate_bytes(
				selected_candidate.manifest,
				expected_sha256=_selected_candidate_sha256(evidence),
			)
			_validate_selected_publication_predecessor(
				config.selected_manifest,
				candidate_bytes=candidate_bytes,
				quarantine_invalid=quarantine_invalid,
			)
		else:
			_validate_hold_has_no_selected_predecessor(
				config.selected_manifest,
				quarantine_invalid=quarantine_invalid,
			)
		_validate_json_publication_predecessor(
			config.calibration_handoff,
			handoff,
			label='calibration handoff',
			quarantine_invalid=quarantine_invalid,
		)
		_validate_json_publication_predecessor(
			config.calibration_report,
			report,
			label='calibration report',
			quarantine_invalid=quarantine_invalid,
		)
		if selected_candidate is None:
			_remove_hold_selected_predecessor(
				config.selected_manifest,
				quarantine_invalid=quarantine_invalid,
			)

		# Publish the target-only evidence before exposing a selected-manifest
		# completion marker.  If either evidence write fails, a downstream
		# consumer cannot observe a selected target without its binding
		# calibration contract.  A later identical rerun can safely recover an
		# evidence-only predecessor by publishing the selected byte-copy last.
		published_handoff = _publish_immutable_json(
			config.calibration_handoff,
			handoff,
			label='calibration handoff',
			quarantine_invalid=quarantine_invalid,
		)
		published_report = _publish_immutable_json(
			config.calibration_report,
			report,
			label='calibration report',
			quarantine_invalid=quarantine_invalid,
		)
		published_selected = (
			config.selected_manifest
			if selected_candidate is not None
			and _publish_selected_manifest(
				candidate=selected_candidate.manifest,
				selected=config.selected_manifest,
				candidate_bytes=candidate_bytes,
				quarantine_invalid=quarantine_invalid,
			)
			else None
		)
		# ``only_missing`` is intentionally the only explicit reuse switch in the
		# CLI.  Immutable exact artifacts are nevertheless always left byte-stable.
		_ = only_missing
		return F3M5LateralTargetCalibrationResult(
			status=status,
			selected_beta=selected_beta,
			evidence=evidence,
			published_selected_manifest=published_selected,
			published_handoff=published_handoff,
			published_report=published_report,
		)
	except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
		if not dry_run:
			raise
		return F3M5LateralTargetCalibrationResult(
			status='FAIL',
			selected_beta=None,
			evidence={
				'artifact_type': _HANDOFF_TYPE,
				'schema_version': 1,
				'status': 'FAIL',
				'error': f'{type(error).__name__}: {error}',
			},
			published_selected_manifest=None,
			published_handoff=None,
			published_report=None,
		)


def load_f3_m5_lateral_target_calibration_handoff(
	path: str | Path,
) -> Mapping[str, object]:
	"""Load a complete selected-or-HOLD target-only calibration handoff."""
	payload = _mapping(
		json.loads(Path(path).read_text(encoding='utf-8')), 'calibration handoff'
	)
	status = _validate_handoff_header(payload)
	candidates = _validate_handoff_sources_and_candidates(payload)
	_validate_handoff_selection(payload, status=status, candidates=candidates)
	_validate_handoff_training_and_git(payload)
	return payload


def _validate_handoff_header(payload: Mapping[str, object]) -> str:
	if (
		payload.get('artifact_type') != _HANDOFF_TYPE
		or payload.get('schema_version') != 1
	):
		raise ValueError('M5-LS calibration handoff type/version mismatch')
	status = payload.get('status')
	if status not in {'M5_LS_TARGET_SELECTED', 'M5_LS_TARGET_HOLD'}:
		raise ValueError('M5-LS calibration handoff status mismatch')
	if payload.get('selection_policy') != _SELECTION_POLICY:
		raise ValueError('M5-LS calibration selection policy mismatch')
	if payload.get('candidate_betas') != [beta for _, beta in _CANONICAL_CANDIDATES]:
		raise ValueError('M5-LS calibration candidate beta set mismatch')
	return status


def _validate_handoff_sources_and_candidates(
	payload: Mapping[str, object],
) -> Mapping[str, object]:
	_beta_zero_handoff(payload.get('beta_zero_parity'))
	_reference_shape(payload.get('source_hard_manifest'), 'source_hard_manifest')
	_reference_shape(
		payload.get('source_posterior_manifest'), 'source_posterior_manifest'
	)
	candidates = _mapping(payload.get('candidates'), 'calibration candidates')
	if list(candidates) != [name for name, _ in _CANONICAL_CANDIDATES]:
		raise ValueError('M5-LS calibration candidate evidence order mismatch')
	for name, beta in _CANONICAL_CANDIDATES:
		_candidate_handoff(candidates.get(name), name=name, beta=beta)
	return candidates


def _validate_handoff_selection(
	payload: Mapping[str, object],
	*,
	status: str,
	candidates: Mapping[str, object],
) -> None:
	first_eligible = _first_eligible_handoff_candidate(candidates)
	if status == 'M5_LS_TARGET_HOLD':
		_validate_hold_handoff_selection(payload, first_eligible=first_eligible)
		return
	_validate_selected_handoff_selection(
		payload, candidates=candidates, first_eligible=first_eligible
	)


def _first_eligible_handoff_candidate(
	candidates: Mapping[str, object],
) -> tuple[str, float] | None:
	for name, beta in _CANONICAL_CANDIDATES:
		candidate = _mapping(candidates[name], f'candidate {name}')
		eligibility = _mapping(
			candidate.get('eligibility'), f'candidate {name} eligibility'
		)
		if eligibility.get('eligible') is True:
			return name, beta
	return None


def _validate_hold_handoff_selection(
	payload: Mapping[str, object],
	*,
	first_eligible: tuple[str, float] | None,
) -> None:
	if first_eligible is not None:
		raise ValueError('HOLD handoff has an eligible candidate')
	if any(
		value is not None
		for value in (
			payload.get('selected_beta'),
			payload.get('selected_candidate_manifest'),
			payload.get('selected_manifest'),
		)
	):
		raise ValueError('HOLD handoff must not name a selected target')


def _validate_selected_handoff_selection(
	payload: Mapping[str, object],
	*,
	candidates: Mapping[str, object],
	first_eligible: tuple[str, float] | None,
) -> None:
	if first_eligible is None:
		raise ValueError('selected handoff has no eligible candidate')
	selected_name, expected_beta = first_eligible
	selected_beta = payload.get('selected_beta')
	if selected_beta not in [beta for _, beta in _CANONICAL_CANDIDATES]:
		raise ValueError('selected beta is not canonical')
	if selected_beta != expected_beta:
		raise ValueError('selected beta is not the smallest eligible beta')
	selected_candidate = payload.get('selected_candidate_manifest')
	selected_manifest = payload.get('selected_manifest')
	_reference_shape(selected_candidate, 'selected_candidate_manifest')
	_reference_shape(selected_manifest, 'selected_manifest')
	if selected_candidate != _mapping(
		candidates[selected_name], f'candidate {selected_name}'
	).get('manifest'):
		raise ValueError('selected candidate identity differs from eligible candidate')
	if _mapping(selected_candidate, 'selected candidate').get('sha256') != _mapping(
		selected_manifest, 'selected manifest'
	).get('sha256'):
		raise ValueError('selected manifest is not a byte-exact candidate copy')


def _validate_handoff_training_and_git(payload: Mapping[str, object]) -> None:
	training_configs = _mapping(payload.get('training_configs'), 'training configs')
	if set(training_configs) != {
		'hard_full_config',
		'lateral_smoke_config',
		'lateral_full_config',
	}:
		raise ValueError('calibration training config identities mismatch')
	for label, reference in training_configs.items():
		_reference_shape(reference, f'training_configs.{label}')
	git = _mapping(payload.get('git'), 'calibration git identity')
	if set(git) != {'head', 'dirty_status', 'git_diff_sha256'}:
		raise ValueError('calibration git identity keys mismatch')
	if not _git_sha(git.get('head')) or not _sha256(git.get('git_diff_sha256')):
		raise TypeError('calibration git SHA identity is invalid')
	if not isinstance(git.get('dirty_status'), list) or not all(
		isinstance(value, str) for value in git['dirty_status']
	):
		raise TypeError('calibration git dirty status is invalid')


def _calibration_evidence(
	config: F3M5LateralTargetCalibrationConfig,
) -> dict[str, object]:
	"""Build all target-only evidence before any selected path can change."""
	hard = load_multi_head_target_manifest(config.source_hard_manifest)
	posterior = load_multi_head_state_posterior_manifest(
		config.source_posterior_manifest
	)
	_source_contract(config, hard, posterior)
	beta_zero_parity = _beta_zero_parity(config, hard)
	if beta_zero_parity['status'] != 'PASS':
		raise ValueError('beta-zero parity did not pass')
	source_transition_counts = _source_transition_counts(hard)
	candidate_evidence = {
		candidate.name: _candidate_evidence(
			config,
			candidate,
			source_transition_counts=source_transition_counts,
		)
		for candidate in config.candidates
	}
	selected = _select_smallest_eligible(config.candidates, candidate_evidence)
	selected_candidate = None if selected is None else _reference(selected.manifest)
	selected_manifest = (
		None
		if selected is None
		else {
			'path': str(config.selected_manifest),
			'sha256': selected_candidate['sha256'],
		}
	)
	return {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'status': (
			'M5_LS_TARGET_HOLD' if selected is None else 'M5_LS_TARGET_SELECTED'
		),
		'selection_policy': _SELECTION_POLICY,
		'candidate_betas': [beta for _, beta in _CANONICAL_CANDIDATES],
		'beta_zero_parity': beta_zero_parity,
		'source_hard_manifest': _reference(config.source_hard_manifest),
		'source_posterior_manifest': _reference(config.source_posterior_manifest),
		'candidates': candidate_evidence,
		'selected_beta': None if selected is None else selected.beta,
		'selected_candidate_manifest': selected_candidate,
		'selected_manifest': selected_manifest,
		'training_configs': {
			'hard_full_config': _reference(config.hard_full_config),
			'lateral_smoke_config': _reference(config.lateral_smoke_config),
			'lateral_full_config': _reference(config.lateral_full_config),
		},
		'git': _git_identity(),
	}


def _source_contract(
	config: F3M5LateralTargetCalibrationConfig,
	hard: Mapping[str, object],
	posterior: Mapping[str, object],
) -> None:
	"""Require the hard/posterior source pair used by every candidate."""
	if hard.get('head_ks') != list(_CANONICAL_KS):
		raise ValueError('source hard manifest K identity mismatch')
	if posterior.get('head_ks') != list(_CANONICAL_KS):
		raise ValueError('source posterior manifest K identity mismatch')
	if not _same_reference(
		posterior.get('source_hard_manifest'), config.source_hard_manifest
	):
		raise ValueError(
			'source posterior manifest is not anchored to source hard manifest'
		)
	if posterior.get('source_embedding') != hard.get('source_embedding'):
		raise ValueError('source hard/posterior embedding identities differ')


@dataclass(frozen=True)
class _BetaZeroReplayContext:
	zero_config: lateral_targets.MultiHeadLateralTargetExportConfig
	source: Mapping[str, object]
	posterior: Mapping[str, object]
	inputs: Mapping[str, object]
	models: Mapping[int, Mapping[str, object]]
	affinity_scale: float
	gap_scales: Mapping[int, float]


def _beta_zero_parity(
	config: F3M5LateralTargetCalibrationConfig,
	hard: Mapping[str, object],
) -> dict[str, object]:
	"""Run the frozen replay/lateral core in memory at beta zero only."""
	context = _beta_zero_replay_context(config, hard)
	canonical_masks: dict[str, np.ndarray] = {}
	head_evidence = {
		str(k): _beta_zero_head_parity(
			context,
			k=k,
			canonical_masks=canonical_masks,
		)
		for k in _CANONICAL_KS
	}
	return {
		'status': 'PASS',
		'pairwise_strength_ratio': 0.0,
		'target_semantics': LATERAL_SMOOTHING_SEMANTICS,
		'heads': head_evidence,
	}


def _beta_zero_replay_context(
	config: F3M5LateralTargetCalibrationConfig,
	hard: Mapping[str, object],
) -> _BetaZeroReplayContext:
	"""Resolve frozen replay inputs while intentionally bypassing beta validation."""
	_, identities = lateral_targets.hard_source_model_identities(hard)
	identity = _mapping(identities.get('6'), 'K=6 source model identity')
	centers = lateral_targets._hashed(identity.get('centers'), 'model centers')  # noqa: SLF001
	model_directory = centers.parent
	if model_directory.name != 'k6' or model_directory.parent.name != 'models':
		raise ValueError('source frozen HMM model layout is invalid')
	clustering_config = lateral_targets._hashed(  # noqa: SLF001
		identity.get('clustering_config'), 'model clustering config'
	)
	source_embedding = _mapping(hard.get('source_embedding'), 'source embedding')
	input_dir = source_embedding.get('input_dir')
	if not isinstance(input_dir, str) or not input_dir:
		raise TypeError('source embedding input_dir is invalid')
	zero_config = lateral_targets.MultiHeadLateralTargetExportConfig(
		source_hard_manifest=config.source_hard_manifest,
		source_posterior_manifest=config.source_posterior_manifest,
		clustering_output_dir=model_directory.parent.parent,
		clustering_config=clustering_config,
		source_embedding_dir=Path(input_dir),
		output_root=config.source_hard_manifest.parent,
		pairwise_strength_ratio=0.0,
		handoff_manifest=config.source_hard_manifest,
	)
	source, posterior, inputs, models = lateral_targets._validate_sources(zero_config)  # noqa: SLF001
	affinity_scale, _ = lateral_targets._affinity_scale(source, posterior, inputs)  # noqa: SLF001
	gap_scales, _ = lateral_targets._emission_gap_scales(source, inputs, models)  # noqa: SLF001
	return _BetaZeroReplayContext(
		zero_config=zero_config,
		source=source,
		posterior=posterior,
		inputs=inputs,
		models=models,
		affinity_scale=affinity_scale,
		gap_scales=gap_scales,
	)


def _beta_zero_head_parity(
	context: _BetaZeroReplayContext,
	*,
	k: int,
	canonical_masks: dict[str, np.ndarray],
) -> dict[str, object]:
	hard_surveys = _head_surveys(context.source, k=k, label='hard')
	posterior_surveys = _head_surveys(context.posterior, k=k, label='posterior')
	if set(hard_surveys) != set(posterior_surveys):
		raise ValueError(f'beta-zero parity survey set mismatch for K={k}')
	surveys = {
		str(survey_id): _beta_zero_survey_parity(
			context,
			k=k,
			survey_id=str(survey_id),
			hard_survey=_mapping(raw_hard, f'K={k} hard survey {survey_id}'),
			posterior_survey=_mapping(
				posterior_surveys.get(survey_id),
				f'K={k} posterior survey {survey_id}',
			),
			canonical_masks=canonical_masks,
		)
		for survey_id, raw_hard in hard_surveys.items()
	}
	return {
		'survey_count': len(surveys),
		'surveys': surveys,
		'labels_bitwise_identical': True,
		'valid_masks_exact': True,
	}


def _head_surveys(
	payload: Mapping[str, object], *, k: int, label: str
) -> Mapping[str, object]:
	heads = _mapping(payload.get('heads'), f'{label} heads')
	head = _mapping(heads.get(str(k)), f'K={k} {label} head')
	return _mapping(head.get('surveys'), f'K={k} {label} surveys')


def _beta_zero_survey_parity(  # noqa: PLR0913
	context: _BetaZeroReplayContext,
	*,
	k: int,
	survey_id: str,
	hard_survey: Mapping[str, object],
	posterior_survey: Mapping[str, object],
	canonical_masks: dict[str, np.ndarray],
) -> dict[str, object]:
	source_labels, valid, posterior_array = _beta_zero_survey_arrays(
		k=k,
		survey_id=survey_id,
		hard_survey=hard_survey,
		posterior_survey=posterior_survey,
		canonical_masks=canonical_masks,
	)
	features = np.load(
		context.inputs[survey_id].embeddings_path,
		mmap_mode='r',
		allow_pickle=False,
	)
	if features.shape[:3] != source_labels.shape:
		raise ValueError(f'beta-zero embedding shape mismatch for K={k} {survey_id}')
	trace_count, valid_trace_count, valid_token_count = _beta_zero_trace_parity(
		context,
		k=k,
		survey_id=survey_id,
		source_labels=source_labels,
		valid=valid,
		posterior=posterior_array,
		features=features,
	)
	return {
		'trace_count': trace_count,
		'valid_trace_count': valid_trace_count,
		'valid_token_count': valid_token_count,
		'labels_bitwise_identical': True,
		'valid_mask_exact': True,
	}


def _beta_zero_survey_arrays(
	*,
	k: int,
	survey_id: str,
	hard_survey: Mapping[str, object],
	posterior_survey: Mapping[str, object],
	canonical_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	source_labels = np.load(
		lateral_targets._source_label_path(hard_survey),  # noqa: SLF001
		mmap_mode='r',
		allow_pickle=False,
	)
	valid = np.load(
		lateral_targets._hashed(hard_survey.get('valid_tokens'), 'hard valid'),  # noqa: SLF001
		mmap_mode='r',
		allow_pickle=False,
	)
	posterior_valid = np.load(
		lateral_targets._hashed(  # noqa: SLF001
			posterior_survey.get('valid_tokens'), 'posterior valid'
		),
		mmap_mode='r',
		allow_pickle=False,
	)
	posterior_array = np.load(
		lateral_targets._hashed(posterior_survey.get('posterior'), 'posterior'),  # noqa: SLF001
		mmap_mode='r',
		allow_pickle=False,
	)
	if (
		source_labels.dtype != np.int32
		or valid.dtype != np.bool_
		or not np.array_equal(valid, posterior_valid)
		or source_labels.shape != valid.shape
		or posterior_array.shape != (*source_labels.shape, k)
	):
		raise ValueError(
			f'beta-zero source/mask contract mismatch for K={k} {survey_id}'
		)
	previous_mask = canonical_masks.get(survey_id)
	if previous_mask is not None and not np.array_equal(previous_mask, valid):
		raise ValueError(f'beta-zero valid mask differs across heads for {survey_id}')
	canonical_masks.setdefault(survey_id, np.asarray(valid))
	return source_labels, valid, posterior_array


def _beta_zero_trace_parity(  # noqa: PLR0913
	context: _BetaZeroReplayContext,
	*,
	k: int,
	survey_id: str,
	source_labels: np.ndarray,
	valid: np.ndarray,
	posterior: np.ndarray,
	features: np.ndarray,
) -> tuple[int, int, int]:
	trace_count = valid_trace_count = valid_token_count = 0
	for x in range(source_labels.shape[0]):
		for y in range(source_labels.shape[1]):
			trace_count += 1
			z = np.flatnonzero(valid[x, y])
			if not z.size:
				continue
			valid_trace_count += 1
			valid_token_count += int(z.size)
			replay, smoothed = lateral_targets._smooth_trace(  # noqa: SLF001
				survey_id=survey_id,
				source_labels=source_labels,
				valid=valid,
				posterior=posterior,
				features=features,
				embedding=context.inputs[survey_id],
				model=context.models[k],
				k=k,
				config=context.zero_config,
				affinity_scale=context.affinity_scale,
				gap_scale=context.gap_scales[k],
				x=x,
				y=y,
				z=z,
			)
			_expected_beta_zero_labels(
				replay,
				np.asarray(smoothed.labels),
				np.asarray(source_labels[x, y, z]),
				context=f'K={k} survey={survey_id} trace=({x}, {y})',
			)
	return trace_count, valid_trace_count, valid_token_count


def _expected_beta_zero_labels(
	replay: np.ndarray,
	lateral: np.ndarray,
	expected: np.ndarray,
	*,
	context: str,
) -> None:
	if (
		replay.dtype != expected.dtype
		or lateral.dtype != expected.dtype
		or replay.tobytes() != expected.tobytes()
		or lateral.tobytes() != expected.tobytes()
	):
		raise ValueError(
			'beta-zero lateral labels differ bitwise from frozen hard labels: '
			+ context
		)


def _source_transition_counts(hard: Mapping[str, object]) -> dict[str, int]:
	"""Recompute source transitions directly from frozen hard-label arrays."""
	result: dict[str, int] = {}
	for k in _CANONICAL_KS:
		head = _mapping(
			_mapping(hard.get('heads'), 'source hard heads').get(str(k)),
			f'source hard K={k}',
		)
		surveys = _mapping(head.get('surveys'), f'source hard K={k} surveys')
		transitions = 0
		for survey_id, raw in surveys.items():
			survey = _mapping(raw, f'source hard K={k} survey {survey_id}')
			labels = np.load(
				lateral_targets._source_label_path(survey),  # noqa: SLF001
				mmap_mode='r',
				allow_pickle=False,
			)
			valid = np.load(
				lateral_targets._hashed(survey.get('valid_tokens'), 'hard valid'),  # noqa: SLF001
				mmap_mode='r',
				allow_pickle=False,
			)
			if (
				labels.shape != valid.shape
				or labels.dtype != np.int32
				or valid.dtype != np.bool_
			):
				raise ValueError(
					f'source hard labels/mask contract is invalid for K={k}'
				)
			for x in range(labels.shape[0]):
				for y in range(labels.shape[1]):
					trace = labels[x, y, valid[x, y]]
					transitions += int(np.count_nonzero(np.diff(trace) != 0))
		result[str(k)] = transitions
	return result


def _candidate_evidence(
	config: F3M5LateralTargetCalibrationConfig,
	candidate: F3M5LateralTargetCandidate,
	*,
	source_transition_counts: Mapping[str, int],
) -> dict[str, object]:
	"""Fully validate a candidate, then evaluate only preregistered checks."""
	manifest_reference = _reference(candidate.manifest)
	base: dict[str, object] = {
		'beta': candidate.beta,
		'manifest': manifest_reference,
		'head_hashes': None,
		'smoothing': None,
		'heads': {},
		'eligibility': {'eligible': False, 'checks': {}, 'reasons': []},
	}
	try:
		manifest = lateral_targets.load_multi_head_lateral_target_manifest(
			candidate.manifest,
			validate_array_semantics=True,
		)
		_candidate_source_contract(config, candidate, manifest)
		base['head_hashes'] = _multi_head_target_hashes(manifest)
		base['smoothing'] = manifest['smoothing']
	except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
		base['eligibility'] = {
			'eligible': False,
			'checks': {},
			'reasons': [f'full_manifest_validation: {type(error).__name__}: {error}'],
		}
		return base

	checks_by_head: dict[str, object] = {}
	reasons: list[str] = []
	head_summaries: dict[str, object] = {}
	for k in _CANONICAL_KS:
		head = _mapping(
			_mapping(manifest.get('heads'), 'candidate heads').get(str(k)),
			f'candidate K={k}',
		)
		diagnostics = _mapping(head.get('diagnostics'), f'candidate K={k} diagnostics')
		aggregate = _mapping(
			diagnostics.get('aggregate'), f'candidate K={k} aggregate diagnostics'
		)
		try:
			checks, head_reasons = _head_eligibility_checks(
				aggregate,
				source_transition_count=_nonnegative_int(
					source_transition_counts.get(str(k)),
					f'source K={k} transition count',
				),
			)
		except (TypeError, ValueError) as error:
			checks = {'diagnostics_well_formed': False}
			head_reasons = [f'malformed diagnostics: {type(error).__name__}: {error}']
		checks_by_head[str(k)] = checks
		reasons.extend(f'K={k}: {reason}' for reason in head_reasons)
		head_summaries[str(k)] = _head_summary(
			aggregate,
			diagnostics.get('resolved_scales'),
			source_transition_count=_nonnegative_int(
				source_transition_counts.get(str(k)), f'source K={k} transition count'
			),
		)
	base['heads'] = head_summaries
	base['eligibility'] = {
		'eligible': not reasons,
		'checks': checks_by_head,
		'reasons': reasons,
	}
	return base


def _candidate_source_contract(
	config: F3M5LateralTargetCalibrationConfig,
	candidate: F3M5LateralTargetCandidate,
	manifest: Mapping[str, object],
) -> None:
	"""Bind a candidate to this exact source pair and its preregistered beta."""
	if manifest.get('head_ks') != list(_CANONICAL_KS):
		raise ValueError('candidate K identity mismatch')
	if manifest.get('target_semantics') != LATERAL_SMOOTHING_SEMANTICS:
		raise ValueError('candidate target semantics mismatch')
	if not _same_reference(
		manifest.get('source_hard_manifest'), config.source_hard_manifest
	):
		raise ValueError('candidate source hard manifest mismatch')
	if not _same_reference(
		manifest.get('source_posterior_manifest'), config.source_posterior_manifest
	):
		raise ValueError('candidate source posterior manifest mismatch')
	smoothing = _mapping(manifest.get('smoothing'), 'candidate smoothing')
	if smoothing.get('pairwise_strength_ratio') != candidate.beta:
		raise ValueError(
			f'candidate {candidate.name} pairwise strength does not equal its '
			'preregistered beta'
		)


def _head_eligibility_checks(
	aggregate: Mapping[str, object],
	*,
	source_transition_count: int,
) -> tuple[dict[str, bool], list[str]]:
	"""Evaluate all and only the preregistered target-only eligibility checks."""
	ordered = _mapping(aggregate.get('ordered_path'), 'ordered path diagnostics')
	occupancy = _mapping(
		aggregate.get('state_occupancy'), 'state occupancy diagnostics'
	)
	changed_regions = _mapping(
		aggregate.get('changed_fraction_by_source_region'),
		'changed source-region diagnostics',
	)
	trace_paths = _mapping(aggregate.get('trace_paths'), 'trace path diagnostics')
	disagreement = _mapping(
		aggregate.get('xy_edge_disagreement'), 'XY disagreement diagnostics'
	)
	weighted = _mapping(
		disagreement.get('affinity_weighted_normalized_order'),
		'affinity-weighted disagreement diagnostics',
	)
	quartiles = disagreement.get('affinity_quartiles')
	if not isinstance(quartiles, list) or len(quartiles) != 4:
		raise ValueError('affinity quartile diagnostics must contain four buckets')
	highest = _mapping(quartiles[-1], 'highest-affinity quartile diagnostics')
	# The strict lateral-manifest loader checks all persistent diagnostics.  The
	# following extraction additionally prevents a mocked/malformed payload from
	# passing a target-only decision simply because a field is absent or NaN.
	checks = {
		'diagnostics_well_formed': _finite_tree(aggregate),
		'ordered_path': (
			_nonnegative_int(ordered.get('violation_count'), 'ordered violation count')
			== 0
			and _nonnegative_int(
				ordered.get('max_reverse_decrease'), 'ordered max reverse decrease'
			)
			== 0
		),
		'nonempty_state_occupancy': _nonnegative_int(
			occupancy.get('empty_state_count'), 'empty state count'
		)
		== 0,
		'changed_tokens_positive': _nonnegative_int(
			aggregate.get('changed_token_count'), 'changed token count'
		)
		> 0,
		'affinity_weighted_disagreement_reduced': _finite_number(
			weighted.get('lateral'), 'weighted lateral disagreement'
		)
		< _finite_number(weighted.get('source'), 'weighted source disagreement'),
		'highest_affinity_quartile_reduced': (
			_nonnegative_int(highest.get('edge_count'), 'highest-affinity edge count')
			> 0
			and _finite_number(
				highest.get('lateral_unweighted_mean'),
				'highest-affinity lateral disagreement',
			)
			< _finite_number(
				highest.get('source_unweighted_mean'),
				'highest-affinity source disagreement',
			)
		),
		'boundary_change_not_less_than_interior': _finite_number(
			changed_regions.get('boundary_adjacent'),
			'boundary-adjacent changed fraction',
		)
		>= _finite_number(changed_regions.get('interior'), 'interior changed fraction'),
		'transition_count_not_increased': _nonnegative_int(
			trace_paths.get('transition_count'), 'lateral transition count'
		)
		<= source_transition_count,
	}
	return checks, [
		label for key, label in _ELIGIBILITY_REASON_LABELS.items() if not checks[key]
	]


def _head_summary(
	aggregate: Mapping[str, object],
	resolved_scales: object,
	*,
	source_transition_count: int,
) -> dict[str, object]:
	"""Keep the handoff compact while retaining each target-only diagnostic."""
	keys = (
		'valid_token_count',
		'invalid_token_count',
		'changed_token_count',
		'changed_fraction',
		'changed_fraction_by_source_region',
		'state_occupancy',
		'ordered_path',
		'trace_paths',
		'xy_edge_disagreement',
	)
	return {
		'resolved_scales': resolved_scales,
		'source_transition_count': source_transition_count,
		**{key: aggregate.get(key) for key in keys},
	}


def _select_smallest_eligible(
	candidates: tuple[F3M5LateralTargetCandidate, ...],
	evidence: Mapping[str, object],
) -> F3M5LateralTargetCandidate | None:
	"""Apply the fixed preregistered policy without inspecting downstream data."""
	for candidate in candidates:
		payload = _mapping(evidence.get(candidate.name), f'{candidate.name} evidence')
		eligibility = _mapping(
			payload.get('eligibility'), f'{candidate.name} eligibility'
		)
		if eligibility.get('eligible') is True:
			return candidate
	return None


def _selected_candidate(
	config: F3M5LateralTargetCalibrationConfig,
	selected_beta: float | None,
) -> F3M5LateralTargetCandidate | None:
	if selected_beta is None:
		return None
	for candidate in config.candidates:
		if candidate.beta == selected_beta:
			return candidate
	raise ValueError('selected beta does not correspond to a canonical candidate')


def _selected_beta_from_evidence(evidence: Mapping[str, object]) -> float | None:
	value = evidence['selected_beta']
	if value is not None and not isinstance(value, float):
		raise TypeError('selected beta evidence is invalid')
	return value


def _candidate_bytes(path: Path, *, expected_sha256: str) -> bytes:
	value = path.read_bytes()
	if _sha256_bytes(value) != expected_sha256:
		raise ValueError('selected candidate manifest drifted during calibration')
	return value


def _selected_candidate_sha256(evidence: Mapping[str, object]) -> str:
	reference = _mapping(
		evidence.get('selected_candidate_manifest'), 'selected candidate manifest'
	)
	value = reference.get('sha256')
	if not _sha256(value):
		raise TypeError('selected candidate manifest SHA-256 is invalid')
	return value


def _handoff_payload(evidence: Mapping[str, object]) -> dict[str, object]:
	"""Normalize the machine-readable immutable calibration handoff."""
	return dict(evidence)


def _report_payload(evidence: Mapping[str, object]) -> dict[str, object]:
	"""Publish a separate technical report without any performance claim."""
	return {
		'artifact_type': _REPORT_TYPE,
		'schema_version': 1,
		'status': evidence['status'],
		'selection_policy': evidence['selection_policy'],
		'candidate_betas': evidence['candidate_betas'],
		'beta_zero_parity': evidence['beta_zero_parity'],
		'source_hard_manifest': evidence['source_hard_manifest'],
		'source_posterior_manifest': evidence['source_posterior_manifest'],
		'candidates': evidence['candidates'],
		'selected_beta': evidence['selected_beta'],
		'selected_candidate_manifest': evidence['selected_candidate_manifest'],
		'selected_manifest': evidence['selected_manifest'],
		'git': evidence['git'],
	}


def _publish_selected_manifest(
	*,
	candidate: Path,
	selected: Path,
	candidate_bytes: bytes | None,
	quarantine_invalid: bool,
) -> bool:
	"""Atomically byte-copy the selected candidate without copying target arrays."""
	value = _validated_candidate_bytes(candidate, candidate_bytes)
	if _selected_reusable(
		selected,
		candidate_bytes=value,
		quarantine_invalid=quarantine_invalid,
	):
		return False
	staged = _stage_validated_selected(selected, value)
	try:
		_replace_selected_from_staging(
			selected,
			staged,
			quarantine_invalid=quarantine_invalid,
		)
	finally:
		if staged.exists():
			staged.unlink()
	_validate_published_selected(selected, candidate_bytes=value)
	return True


def _validated_candidate_bytes(candidate: Path, value: bytes | None) -> bytes:
	if value is None:
		raise ValueError('selected candidate bytes are missing')
	if _sha256_bytes(value) != file_sha256(candidate):
		raise ValueError('selected candidate manifest changed before publication')
	return value


def _selected_reusable(
	selected: Path,
	*,
	candidate_bytes: bytes,
	quarantine_invalid: bool,
) -> bool:
	if not selected.exists():
		return False
	if not selected.is_file():
		raise ValueError('existing selected manifest path is not a file')
	if selected.read_bytes() != candidate_bytes:
		if quarantine_invalid:
			return False
		raise ValueError(
			'existing selected manifest is incompatible; pass '
			'--quarantine-invalid to replace it'
		)
	try:
		lateral_targets.load_multi_head_lateral_target_manifest(selected)
	except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
		if quarantine_invalid:
			return False
		raise ValueError(
			'existing selected manifest is invalid; pass --quarantine-invalid '
			'to replace it'
		) from error
	return True


def _replace_selected_from_staging(
	selected: Path,
	staged: Path,
	*,
	quarantine_invalid: bool,
) -> None:
	if selected.exists():
		if not quarantine_invalid:
			raise ValueError(
				'existing selected manifest appeared during publication; pass '
				'--quarantine-invalid to replace it'
			)
		_quarantine(selected)
	staged.replace(selected)


def _stage_validated_selected(selected: Path, value: bytes) -> Path:
	staged = _stage_bytes(selected, value)
	try:
		lateral_targets.load_multi_head_lateral_target_manifest(staged)
	except BaseException:
		if staged.exists():
			staged.unlink()
		raise
	return staged


def _validate_published_selected(selected: Path, *, candidate_bytes: bytes) -> None:
	lateral_targets.load_multi_head_lateral_target_manifest(selected)
	if selected.read_bytes() != candidate_bytes:
		raise ValueError('published selected manifest is not byte exact')


def _validate_selected_publication_predecessor(
	path: Path,
	*,
	candidate_bytes: bytes,
	quarantine_invalid: bool,
) -> None:
	if not path.exists():
		return
	if not path.is_file():
		raise ValueError('existing selected manifest path is not a file')
	if path.read_bytes() == candidate_bytes:
		try:
			lateral_targets.load_multi_head_lateral_target_manifest(path)
		except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
			if quarantine_invalid:
				return
			raise ValueError(
				'existing selected manifest is invalid; pass --quarantine-invalid '
				'to replace it'
			) from error
		else:
			return
	if not quarantine_invalid:
		raise ValueError(
			'existing selected manifest is incompatible; pass --quarantine-invalid '
			'to replace it'
		)


def _validate_hold_has_no_selected_predecessor(
	path: Path,
	*,
	quarantine_invalid: bool,
) -> None:
	"""Prevent a HOLD decision from silently retaining stale selected evidence."""
	if not path.exists():
		return
	if not quarantine_invalid:
		raise ValueError(
			'HOLD decision found an existing selected manifest; pass '
			'--quarantine-invalid to preserve and remove stale evidence'
		)
	if not path.is_file():
		raise ValueError('existing selected manifest path is not a file')


def _remove_hold_selected_predecessor(
	path: Path,
	*,
	quarantine_invalid: bool,
) -> None:
	if not path.exists():
		return
	if not quarantine_invalid:
		raise ValueError(
			'HOLD selected-manifest removal requires quarantine permission'
		)
	_quarantine(path)
	path.unlink()


def _validate_json_publication_predecessor(
	path: Path,
	payload: Mapping[str, object],
	*,
	label: str,
	quarantine_invalid: bool,
) -> None:
	if not path.exists():
		return
	if not path.is_file():
		raise ValueError(f'existing {label} path is not a file')
	try:
		existing = _mapping(json.loads(path.read_text(encoding='utf-8')), label)
	except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
		if quarantine_invalid:
			return
		raise ValueError(
			f'existing {label} is invalid; pass --quarantine-invalid to replace it'
		) from error
	if existing != payload and not quarantine_invalid:
		raise ValueError(
			f'existing {label} is incompatible; pass --quarantine-invalid to replace it'
		)


def _publish_immutable_json(
	path: Path,
	payload: Mapping[str, object],
	*,
	label: str,
	quarantine_invalid: bool,
) -> Path | None:
	"""Reuse exact evidence and otherwise require an explicit quarantine flag."""
	if path.exists():
		if not path.is_file():
			raise ValueError(f'existing {label} path is not a file')
		try:
			existing = _mapping(json.loads(path.read_text(encoding='utf-8')), label)
		except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
			existing = None
			if not quarantine_invalid:
				raise ValueError(
					f'existing {label} is invalid; pass --quarantine-invalid '
					'to replace it'
				) from error
		if existing == payload:
			return None
		if not quarantine_invalid:
			raise ValueError(
				f'existing {label} is incompatible; pass --quarantine-invalid '
				'to replace it'
			)
		_quarantine(path)
	hard_validation._atomic_json(path, payload)  # noqa: SLF001
	return path


def _stage_bytes(path: Path, value: bytes) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
	staged = Path(temporary)
	try:
		with os.fdopen(fd, 'wb') as stream:
			stream.write(value)
			stream.flush()
			os.fsync(stream.fileno())
	except BaseException:
		if staged.exists():
			staged.unlink()
		raise
	return staged


def _quarantine(path: Path) -> Path:
	"""Use the established F3 hard-link quarantine behavior."""
	return hard_validation._quarantine(path)  # noqa: SLF001


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _same_reference(value: object, path: Path) -> bool:
	try:
		reference = _mapping(value, 'manifest reference')
		candidate = reference.get('path')
		return (
			isinstance(candidate, str)
			and Path(candidate).resolve() == path.resolve()
			and reference.get('sha256') == file_sha256(path)
		)
	except (OSError, TypeError, ValueError):
		return False


def _mapping_path(mapping: Mapping[str, object], key: str, *, label: str) -> Path:
	value = mapping.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	return Path(value).resolve()


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _nonnegative_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise TypeError(f'{label} must be a non-negative integer')
	return value


def _finite_number(value: object, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{label} must be a finite number')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label} must be finite')
	return result


def _finite_tree(value: object) -> bool:
	if isinstance(value, Mapping):
		return all(_finite_tree(item) for item in value.values())
	if isinstance(value, list):
		return all(_finite_tree(item) for item in value)
	if isinstance(value, float):
		return math.isfinite(value)
	return not isinstance(value, np.floating) or math.isfinite(float(value))


def _sha256(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in '0123456789abcdef' for character in value.lower())
	)


def _git_sha(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 40
		and all(character in '0123456789abcdef' for character in value.lower())
	)


def _sha256_bytes(value: bytes) -> str:
	return hashlib.sha256(value).hexdigest()


def _reference_shape(value: object, label: str) -> None:
	reference = _mapping(value, label)
	if set(reference) != {'path', 'sha256'}:
		raise ValueError(f'{label} identity keys mismatch')
	if not isinstance(reference.get('path'), str) or not reference['path']:
		raise TypeError(f'{label}.path is invalid')
	if not _sha256(reference.get('sha256')):
		raise TypeError(f'{label}.sha256 is invalid')


def _beta_zero_handoff(value: object) -> None:
	parity = _mapping(value, 'beta-zero parity')
	if (
		parity.get('status') != 'PASS'
		or parity.get('pairwise_strength_ratio') != 0.0
		or parity.get('target_semantics') != LATERAL_SMOOTHING_SEMANTICS
	):
		raise ValueError('beta-zero parity evidence mismatch')
	heads = _mapping(parity.get('heads'), 'beta-zero parity heads')
	if set(heads) != {str(k) for k in _CANONICAL_KS}:
		raise ValueError('beta-zero parity head set mismatch')
	for k in _CANONICAL_KS:
		_beta_zero_head_handoff(heads[str(k)], k=k)


def _beta_zero_head_handoff(value: object, *, k: int) -> None:
	head = _mapping(value, f'beta-zero parity K={k}')
	if set(head) != {
		'survey_count',
		'surveys',
		'labels_bitwise_identical',
		'valid_masks_exact',
	}:
		raise ValueError(f'beta-zero parity K={k} evidence keys mismatch')
	if (
		head.get('labels_bitwise_identical') is not True
		or head.get('valid_masks_exact') is not True
	):
		raise ValueError(f'beta-zero parity K={k} did not pass')
	survey_count = _nonnegative_int(
		head.get('survey_count'), f'beta-zero parity K={k} survey count'
	)
	surveys = _mapping(head.get('surveys'), f'beta-zero parity K={k} surveys')
	if survey_count == 0 or survey_count != len(surveys):
		raise ValueError(f'beta-zero parity K={k} survey coverage mismatch')
	for survey_id, survey_value in surveys.items():
		_beta_zero_survey_handoff(survey_value, k=k, survey_id=survey_id)


def _beta_zero_survey_handoff(value: object, *, k: int, survey_id: object) -> None:
	if not isinstance(survey_id, str) or not survey_id:
		raise TypeError(f'beta-zero parity K={k} survey identity is invalid')
	survey = _mapping(value, f'beta-zero parity K={k} survey {survey_id}')
	if set(survey) != {
		'trace_count',
		'valid_trace_count',
		'valid_token_count',
		'labels_bitwise_identical',
		'valid_mask_exact',
	}:
		raise ValueError(
			f'beta-zero parity K={k} survey {survey_id} evidence keys mismatch'
		)
	if (
		survey.get('labels_bitwise_identical') is not True
		or survey.get('valid_mask_exact') is not True
	):
		raise ValueError(f'beta-zero parity K={k} survey {survey_id} did not pass')
	for key in ('trace_count', 'valid_trace_count', 'valid_token_count'):
		_nonnegative_int(
			survey.get(key), f'beta-zero parity K={k} survey {survey_id} {key}'
		)


def _candidate_handoff(value: object, *, name: str, beta: float) -> None:
	candidate = _mapping(value, f'candidate {name}')
	if candidate.get('beta') != beta:
		raise ValueError(f'candidate {name} beta mismatch')
	_reference_shape(candidate.get('manifest'), f'candidate {name} manifest')
	eligibility = _mapping(
		candidate.get('eligibility'), f'candidate {name} eligibility'
	)
	if set(eligibility) != {'eligible', 'checks', 'reasons'}:
		raise ValueError(f'candidate {name} eligibility schema mismatch')
	if not isinstance(eligibility.get('eligible'), bool):
		raise TypeError(f'candidate {name} eligibility must be boolean')
	if not isinstance(eligibility.get('checks'), Mapping):
		raise TypeError(f'candidate {name} checks must be a mapping')
	if not isinstance(eligibility.get('reasons'), list) or not all(
		isinstance(reason, str) for reason in eligibility['reasons']
	):
		raise TypeError(f'candidate {name} reasons must be strings')
	checks = _mapping(eligibility['checks'], f'candidate {name} checks')
	if not checks:
		_candidate_unvalidated_handoff(candidate, eligibility=eligibility, name=name)
		return
	_candidate_validated_handoff(
		candidate, eligibility=eligibility, checks=checks, name=name
	)


def _candidate_unvalidated_handoff(
	candidate: Mapping[str, object],
	*,
	eligibility: Mapping[str, object],
	name: str,
) -> None:
	if eligibility['eligible']:
		raise ValueError(f'eligible candidate {name} lacks per-head checks')
	if not eligibility['reasons']:
		raise ValueError(f'ineligible candidate {name} lacks failure reasons')
	if (
		candidate.get('head_hashes') is not None
		or candidate.get('smoothing') is not None
	):
		raise ValueError(f'unvalidated candidate {name} has target identity evidence')
	if candidate.get('heads') != {}:
		raise ValueError(f'unvalidated candidate {name} has per-head diagnostics')


def _candidate_validated_handoff(
	candidate: Mapping[str, object],
	*,
	eligibility: Mapping[str, object],
	checks: Mapping[str, object],
	name: str,
) -> None:
	if set(checks) != {str(k) for k in _CANONICAL_KS}:
		raise ValueError(f'candidate {name} eligibility check K set mismatch')
	head_hashes = _mapping(
		candidate.get('head_hashes'), f'candidate {name} head hashes'
	)
	if set(head_hashes) != {str(k) for k in _CANONICAL_KS}:
		raise ValueError(f'candidate {name} head hash K set mismatch')
	if not isinstance(candidate.get('smoothing'), Mapping):
		raise TypeError(f'candidate {name} smoothing identity is missing')
	heads = _mapping(candidate.get('heads'), f'candidate {name} diagnostics')
	if set(heads) != {str(k) for k in _CANONICAL_KS}:
		raise ValueError(f'candidate {name} diagnostics K set mismatch')
	all_checks_pass = _validated_candidate_checks(checks, heads=heads, name=name)
	if eligibility['eligible'] != all_checks_pass:
		raise ValueError(f'candidate {name} eligibility does not match its checks')
	_candidate_reason_consistency(eligibility, name=name)


def _validated_candidate_checks(
	checks: Mapping[str, object],
	*,
	heads: Mapping[str, object],
	name: str,
) -> bool:
	all_checks_pass = True
	for k in _CANONICAL_KS:
		head_checks = _mapping(checks[str(k)], f'candidate {name} K={k} checks')
		if set(head_checks) != set(_ELIGIBILITY_REASON_LABELS):
			raise ValueError(f'candidate {name} K={k} eligibility checks are invalid')
		if not all(isinstance(result, bool) for result in head_checks.values()):
			raise TypeError(
				f'candidate {name} K={k} eligibility checks must be boolean'
			)
		all_checks_pass = all_checks_pass and all(head_checks.values())
		_mapping(heads[str(k)], f'candidate {name} K={k} diagnostics')
	return all_checks_pass


def _candidate_reason_consistency(
	eligibility: Mapping[str, object], *, name: str
) -> None:
	if eligibility['eligible'] and eligibility['reasons']:
		raise ValueError(f'eligible candidate {name} must not have failure reasons')
	if not eligibility['eligible'] and not eligibility['reasons']:
		raise ValueError(f'ineligible candidate {name} lacks failure reasons')


def _git_identity() -> dict[str, object]:
	"""Record repository SHA and dirty state without adding calibration inputs."""
	repository = Path(__file__).resolve().parents[3]
	git = shutil.which('git')
	if git is None:
		raise RuntimeError('git is required for calibration provenance')

	def output(*args: str) -> str:
		return subprocess.check_output(  # noqa: S603
			[git, *args], cwd=repository, text=True
		).strip()

	diff = subprocess.check_output(  # noqa: S603
		[git, 'diff', '--binary', 'HEAD'], cwd=repository
	)
	return {
		'head': output('rev-parse', 'HEAD'),
		'dirty_status': output(
			'status', '--short', '--untracked-files=all'
		).splitlines(),
		'git_diff_sha256': _sha256_bytes(diff),
	}


__all__ = [
	'F3M5LateralTargetCalibrationConfig',
	'F3M5LateralTargetCalibrationResult',
	'F3M5LateralTargetCandidate',
	'calibrate_f3_m5_lateral_targets',
	'f3_m5_lateral_target_calibration_config_from_mapping',
	'load_f3_m5_lateral_target_calibration_config',
	'load_f3_m5_lateral_target_calibration_handoff',
]
