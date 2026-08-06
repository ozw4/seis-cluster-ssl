"""Read-only preflight evidence for unanimous XY-neighbour screening.

This stage is deliberately separate from decoder planning.  It binds the
schema-v6 candidate to the target-only unanimous GO audit, proves hard/candidate
initialisation parity, and records the immutable reference manifests before any
new original-split job is admitted.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import copy
import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.f3.lithology import (
	xy_neighbor_consensus_screening_audit as consensus_audit,
)
from seis_ssl_cluster.f3.multi_head_pretraining_validation import (
	load_f3_multi_head_pretraining_handoff,
)
from seis_ssl_cluster.f3.xy_neighbor_unanimous_pretraining_validation import (
	load_f3_xy_neighbor_unanimous_pretraining_handoff,
)
from seis_ssl_cluster.f3.xy_neighbor_unanimous_target_audit import (
	replay_f3_xy_neighbor_unanimous_target_audit,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	load_multi_head_xy_neighbor_unanimous_target_manifest,
)
from seis_ssl_cluster.training.strat_hmm.components import (
	build_strat_hmm_components,
)

ARTIFACT_TYPE = 'f3_xy_neighbor_unanimous_original_screening_preflight'
SCHEMA_VERSION = 1
XY_UNANIM_MODEL_ID = 'mh_xyunanim1_nocons'
XY_UNANIM_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1'
)
HARD_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'workspace_root',
		'source_hard_manifest',
		'xy_neighbor_consensus_target_manifest',
		'xy_neighbor_unanimous_target_manifest',
		'target_audit',
		'hard_full_config',
		'hard_pretraining_handoff',
		'candidate_full_config',
		'candidate_pretraining_handoff',
		'candidate_embeddings_dir',
		'xy_neighbor_consensus_run_manifest',
		'hard_reference_run_manifest',
		'current_k6_run_manifest',
		'mae_reference_run_manifest',
		'output_path',
	}
)
_ALLOWED_CONFIG_DIFF = {
	'paths': ['output_root'],
	'identity': ['model_tag', 'scientific_identity.representation_specific_fields'],
	'pseudo_targets': ['manifest', 'target_representation'],
}
_REPRESENTATION_SPECIFIC_SCIENTIFIC_FIELDS = frozenset(
	{
		'experiment_role',
		'variant',
		'target_representation',
		'target_semantics',
		'target_manifest_sha256',
		'target_head_hashes',
		'xy_neighbor_unanimous_target_manifest_sha256',
		'xy_neighbor_unanimous_target_head_hashes',
		'xy_neighbor_unanimous_smoothing',
		'source_hard_manifest_sha256',
		'supervised_loss',
		'consistency_policy',
	}
)


@dataclass(frozen=True)
class F3XYNeighborUnanimousScreeningAuditConfig:
	"""Closed paths required by the immutable unanimous screening preflight."""

	artifact_root: Path
	workspace_root: Path
	source_hard_manifest: Path
	xy_neighbor_consensus_target_manifest: Path
	xy_neighbor_unanimous_target_manifest: Path
	target_audit: Path
	hard_full_config: Path
	hard_pretraining_handoff: Path
	candidate_full_config: Path
	candidate_pretraining_handoff: Path
	candidate_embeddings_dir: Path
	xy_neighbor_consensus_run_manifest: Path
	hard_reference_run_manifest: Path
	current_k6_run_manifest: Path
	mae_reference_run_manifest: Path
	output_path: Path


@dataclass(frozen=True)
class F3XYNeighborUnanimousScreeningAuditResult:
	"""The validated immutable audit payload and its write action."""

	payload: Mapping[str, object]
	output_path: Path
	action: str
	quarantine_path: Path | None


def f3_xy_neighbor_unanimous_screening_audit_config_from_mapping(  # noqa: C901
	config: Mapping[str, object],
) -> F3XYNeighborUnanimousScreeningAuditConfig:
	"""Resolve the closed audit schema without accepting extension fields."""
	if not isinstance(config, Mapping):
		raise TypeError(
			'XY-neighbour-unanimous screening audit config must be a mapping'
		)
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown screening audit config keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing screening audit config keys: {sorted(missing)!r}')

	def path(key: str, *, must_exist: bool) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		resolved = Path(value).resolve()
		if must_exist and not resolved.exists():
			raise FileNotFoundError(f'{key} is missing: {resolved}')
		return resolved

	result = F3XYNeighborUnanimousScreeningAuditConfig(
		artifact_root=path('artifact_root', must_exist=True),
		workspace_root=path('workspace_root', must_exist=True),
		source_hard_manifest=path('source_hard_manifest', must_exist=True),
		xy_neighbor_consensus_target_manifest=path(
			'xy_neighbor_consensus_target_manifest', must_exist=True
		),
		xy_neighbor_unanimous_target_manifest=path(
			'xy_neighbor_unanimous_target_manifest', must_exist=True
		),
		target_audit=path('target_audit', must_exist=True),
		hard_full_config=path('hard_full_config', must_exist=True),
		hard_pretraining_handoff=path('hard_pretraining_handoff', must_exist=True),
		candidate_full_config=path('candidate_full_config', must_exist=True),
		candidate_pretraining_handoff=path(
			'candidate_pretraining_handoff', must_exist=True
		),
		candidate_embeddings_dir=path('candidate_embeddings_dir', must_exist=True),
		xy_neighbor_consensus_run_manifest=path(
			'xy_neighbor_consensus_run_manifest', must_exist=True
		),
		hard_reference_run_manifest=path(
			'hard_reference_run_manifest', must_exist=True
		),
		current_k6_run_manifest=path('current_k6_run_manifest', must_exist=True),
		mae_reference_run_manifest=path('mae_reference_run_manifest', must_exist=True),
		output_path=path('output_path', must_exist=False),
	)
	if not result.artifact_root.is_dir() or not result.workspace_root.is_dir():
		raise FileNotFoundError('artifact_root and workspace_root must be directories')
	for label, value in (
		('source_hard_manifest', result.source_hard_manifest),
		(
			'xy_neighbor_consensus_target_manifest',
			result.xy_neighbor_consensus_target_manifest,
		),
		(
			'xy_neighbor_unanimous_target_manifest',
			result.xy_neighbor_unanimous_target_manifest,
		),
		('target_audit', result.target_audit),
		('hard_pretraining_handoff', result.hard_pretraining_handoff),
		('candidate_pretraining_handoff', result.candidate_pretraining_handoff),
		(
			'xy_neighbor_consensus_run_manifest',
			result.xy_neighbor_consensus_run_manifest,
		),
		('hard_reference_run_manifest', result.hard_reference_run_manifest),
		('current_k6_run_manifest', result.current_k6_run_manifest),
		('mae_reference_run_manifest', result.mae_reference_run_manifest),
	):
		if not value.is_file():
			raise FileNotFoundError(f'{label} is missing: {value}')
	for label, value in (
		('hard_full_config', result.hard_full_config),
		('candidate_full_config', result.candidate_full_config),
	):
		if not value.is_file():
			raise FileNotFoundError(f'{label} is missing: {value}')
	if not result.candidate_embeddings_dir.is_dir():
		raise FileNotFoundError(
			f'candidate_embeddings_dir is missing: {result.candidate_embeddings_dir}'
		)
	return result


def load_f3_xy_neighbor_unanimous_screening_audit_config(
	path: str | Path,
) -> F3XYNeighborUnanimousScreeningAuditConfig:
	"""Load the standalone closed audit YAML configuration."""
	return f3_xy_neighbor_unanimous_screening_audit_config_from_mapping(
		load_config(path)
	)


def audit_f3_xy_neighbor_unanimous_screening(
	config: F3XYNeighborUnanimousScreeningAuditConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3XYNeighborUnanimousScreeningAuditResult:
	"""Build or immutably reuse screening evidence without mutating inputs."""
	git = _clean_git_identity(config.workspace_root)
	payload = _audit_payload(config, git=git)
	if dry_run:
		return F3XYNeighborUnanimousScreeningAuditResult(
			payload, config.output_path, 'DRY_RUN', None
		)
	if config.output_path.exists():
		if not config.output_path.is_file():
			raise FileExistsError(
				f'audit output path is not a file: {config.output_path}'
			)
		try:
			existing = load_f3_xy_neighbor_unanimous_screening_audit(
				config.output_path
			)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == payload:
			if only_missing:
				return F3XYNeighborUnanimousScreeningAuditResult(
					payload, config.output_path, 'REUSE_COMPLETED', None
				)
			raise FileExistsError(
				f'audit already exists; use --only-missing: {config.output_path}'
			)
		if not quarantine_invalid:
			raise ValueError(
				'incompatible existing XY-neighbour-unanimous audit; '
				'use --quarantine-invalid to replace it'
			)
		quarantine = _quarantine_invalid(config.output_path)
	else:
		quarantine = None
	_write_json_atomically(config.output_path, payload)
	return F3XYNeighborUnanimousScreeningAuditResult(
		payload, config.output_path, 'WRITTEN', quarantine
	)


def load_f3_xy_neighbor_unanimous_screening_audit(
	path: str | Path,
) -> Mapping[str, object]:
	"""Load only a complete, schema-v1 PASS unanimous screening preflight."""
	payload = _read_json(Path(path))
	if (
		payload.get('artifact_type') != ARTIFACT_TYPE
		or payload.get('schema_version') != SCHEMA_VERSION
		or payload.get('status') != 'PASS'
	):
		raise ValueError(
			'XY-neighbour-unanimous screening audit type/schema/status mismatch'
		)
	for key in (
		'candidate',
		'hard_baseline_parity',
		'xy_neighbor_unanimous_spatial_smoothness',
		'target_audit',
		'reference_run_manifests',
		'git',
	):
		if not isinstance(payload.get(key), Mapping):
			raise TypeError(f'XY-neighbour-unanimous screening audit {key} is missing')
	return payload


def validate_f3_xy_neighbor_unanimous_screening_audit_binding(
	payload: Mapping[str, object],
	*,
	model_id: str,
	model_tag: str,
	pretraining_handoff: Path,
	embeddings_dir: Path,
) -> None:
	"""Bind a PASS audit to the exact candidate admitted to decoder planning."""
	if (
		payload.get('artifact_type') != ARTIFACT_TYPE
		or payload.get('schema_version') != SCHEMA_VERSION
		or payload.get('status') != 'PASS'
	):
		raise ValueError('XY-neighbour-unanimous screening audit identity mismatch')
	candidate = _mapping(payload.get('candidate'), 'screening audit candidate')
	if candidate.get('model_id') != model_id or candidate.get('model_tag') != model_tag:
		raise ValueError('screening audit candidate identity mismatch')
	_identity_matches(
		candidate.get('pretraining_handoff'),
		pretraining_handoff,
		label='screening audit pretraining handoff',
	)
	embeddings = _mapping(candidate.get('embeddings'), 'screening audit embeddings')
	root = embeddings.get('root')
	if not isinstance(root, str) or Path(root).resolve() != embeddings_dir.resolve():
		raise ValueError('screening audit candidate embeddings root mismatch')
	for key in ('embeddings', 'valid_tokens', 'metadata'):
		value = embeddings.get(key)
		if not isinstance(value, Mapping):
			raise TypeError(f'screening audit candidate {key} identity is missing')
		_identity_matches_live(value, label=f'screening audit {key}')
	if (
		not isinstance(payload.get('hard_baseline_parity'), Mapping)
		or payload['hard_baseline_parity'].get('status') != 'PASS'
	):
		raise ValueError('screening audit hard-baseline parity is not PASS')
	_validate_live_target_audit_binding(payload, pretraining_handoff)


def _validate_live_target_audit_binding(
	payload: Mapping[str, object], pretraining_handoff: Path
) -> None:
	"""Replay the target gate and bind it to the candidate handoff lineage."""
	target_audit = _mapping(payload.get('target_audit'), 'screening target audit')
	target_path = _live_identity_path(target_audit, label='screening target audit')
	artifact_root_value = payload.get('artifact_root')
	if not isinstance(artifact_root_value, str) or not artifact_root_value:
		raise TypeError('screening audit artifact_root is missing')
	artifact_root = Path(artifact_root_value).resolve()
	if not artifact_root.is_dir():
		raise FileNotFoundError('screening audit artifact_root is missing')
	replayed_target_audit = replay_f3_xy_neighbor_unanimous_target_audit(
		target_path, artifact_root=artifact_root
	)
	if (
		target_audit.get('status') != 'XYUNANIM_TARGET_GO'
		or replayed_target_audit.get('status') != 'XYUNANIM_TARGET_GO'
	):
		raise ValueError('screening target audit is not XYUNANIM_TARGET_GO')
	if target_audit != {**_identity(target_path), 'status': 'XYUNANIM_TARGET_GO'}:
		raise ValueError('screening target audit identity differs from live replay')
	handoff = load_f3_xy_neighbor_unanimous_pretraining_handoff(
		pretraining_handoff
	)
	targets = _mapping(handoff.get('targets'), 'candidate handoff targets')
	_reference_matches(
		targets.get('target_audit'),
		target_path,
		label='candidate handoff target audit',
	)
	for audit_key, handoff_key in (
		('source_hard_manifest', 'source_hard_manifest'),
		('xy_neighbor_unanimous_target_manifest', 'target_manifest'),
	):
		audit_identity = _mapping(payload.get(audit_key), audit_key)
		audit_path = _live_identity_path(audit_identity, label=audit_key)
		_reference_matches(
			targets.get(handoff_key),
			audit_path,
			label=f'candidate handoff {handoff_key}',
		)


def _audit_payload(
	config: F3XYNeighborUnanimousScreeningAuditConfig,
	*,
	git: Mapping[str, object],
) -> dict[str, object]:
	"""Build every audit assertion before a single output byte is written."""
	hard = _load_hard_full_config(config)
	candidate = resolve_strat_hmm_pretext_config(
		load_config(config.candidate_full_config)
	)
	_validate_training_identity(config, hard=hard, candidate=candidate)
	_validate_allowed_config_delta(hard, candidate)
	source = load_multi_head_target_manifest(config.source_hard_manifest)
	consensus = load_multi_head_xy_neighbor_consensus_target_manifest(
		config.xy_neighbor_consensus_target_manifest
	)
	unanimous = load_multi_head_xy_neighbor_unanimous_target_manifest(
		config.xy_neighbor_unanimous_target_manifest
	)
	target_audit = _validate_target_audit(
		config, source=source, consensus=consensus, unanimous=unanimous
	)
	_validate_target_lineage(
		config, source=source, unanimous=unanimous, candidate=candidate
	)
	hard_handoff = load_f3_multi_head_pretraining_handoff(
		config.hard_pretraining_handoff
	)
	unanimous_handoff = load_f3_xy_neighbor_unanimous_pretraining_handoff(
		config.candidate_pretraining_handoff
	)
	hard_runtime, candidate_runtime = _parity_runtime(hard, candidate)
	_validate_runtime_parity(
		hard_runtime,
		candidate_runtime,
		hard_handoff=hard_handoff,
		candidate_handoff=unanimous_handoff,
	)
	candidate_evidence = _candidate_evidence(
		config, unanimous_handoff=unanimous_handoff
	)
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': 'PASS',
		'git': dict(git),
		'artifact_root': str(config.artifact_root),
		'workspace_root': str(config.workspace_root),
		'source_hard_manifest': _identity(config.source_hard_manifest),
		'xy_neighbor_consensus_target_manifest': _identity(
			config.xy_neighbor_consensus_target_manifest
		),
		'xy_neighbor_unanimous_target_manifest': _identity(
			config.xy_neighbor_unanimous_target_manifest
		),
		'target_audit': {
			**_identity(config.target_audit),
			'status': target_audit['status'],
		},
		'candidate': candidate_evidence,
		'hard_baseline_parity': {
			'status': 'PASS',
			'allowed_config_diff': copy.deepcopy(_ALLOWED_CONFIG_DIFF),
			'normalized_config_equal': True,
			'hard': {
				'config': _identity(config.hard_full_config),
				'pretraining_handoff': _identity(config.hard_pretraining_handoff),
				'initial_student_state_sha256': hard_runtime[
					'initial_student_state_sha256'
				],
				'initial_head_state_sha256': hard_runtime['initial_head_state_sha256'],
				'trainability_summary': hard_runtime['trainability_summary'],
				'optimizer_group_identity': hard_runtime['optimizer_group_identity'],
			},
			'candidate': {
				'config': _identity(config.candidate_full_config),
				'pretraining_handoff': _identity(config.candidate_pretraining_handoff),
				'initial_student_state_sha256': candidate_runtime[
					'initial_student_state_sha256'
				],
				'initial_head_state_sha256': candidate_runtime[
					'initial_head_state_sha256'
				],
				'trainability_summary': candidate_runtime['trainability_summary'],
				'optimizer_group_identity': candidate_runtime[
					'optimizer_group_identity'
				],
			},
		},
		'xy_neighbor_unanimous_spatial_smoothness': {
			'per_k': consensus_audit._spatial_smoothness(source, unanimous)
		},
		'reference_run_manifests': _reference_run_manifests(config),
	}


def _load_hard_full_config(
	config: F3XYNeighborUnanimousScreeningAuditConfig,
) -> Mapping[str, object]:
	"""Resolve M4 with its frozen source-manifest digest scoped to this load."""
	with _source_hard_manifest_hash_environment(config.source_hard_manifest):
		return resolve_strat_hmm_pretext_config(load_config(config.hard_full_config))


@contextmanager
def _source_hard_manifest_hash_environment(
	source_hard_manifest: Path,
) -> Iterator[None]:
	"""Temporarily supply the M4 template variable from the immutable input."""
	name = 'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256'
	prior = os.environ.get(name)
	os.environ[name] = file_sha256(source_hard_manifest)
	try:
		yield
	finally:
		if prior is None:
			os.environ.pop(name, None)
		else:
			os.environ[name] = prior


def _validate_training_identity(
	config: F3XYNeighborUnanimousScreeningAuditConfig,
	*,
	hard: Mapping[str, object],
	candidate: Mapping[str, object],
) -> None:
	if (
		_model_tag(hard) != HARD_MODEL_TAG
		or _model_tag(candidate) != XY_UNANIM_MODEL_TAG
	):
		raise ValueError('hard/candidate pretraining model tag mismatch')
	hard_target = _mapping(hard.get('pseudo_targets'), 'hard pseudo targets')
	candidate_target = _mapping(
		candidate.get('pseudo_targets'), 'candidate pseudo targets'
	)
	if (
		Path(str(hard_target.get('manifest', ''))).resolve()
		!= config.source_hard_manifest
	):
		raise ValueError('hard full config source target manifest mismatch')
	if (
		Path(str(candidate_target.get('manifest', ''))).resolve()
		!= config.xy_neighbor_unanimous_target_manifest
	):
		raise ValueError('candidate full config unanimous target manifest mismatch')
	if (
		candidate_target.get('target_representation')
		!= 'xy_neighbor_unanimous_hard_labels_v1'
	):
		raise ValueError('candidate full config target representation mismatch')
	scientific = _mapping(
		_mapping(candidate.get('identity'), 'candidate identity').get(
			'scientific_identity'
		),
		'candidate scientific identity',
	)
	if (
		scientific.get('experiment_role')
		!= 'multi_head_ordered_xy_neighbor_unanimous_hard_pretext'
		or scientific.get('variant') != 'xyunanim1_nocons'
		or scientific.get('target_representation')
		!= 'xy_neighbor_unanimous_hard_labels_v1'
		or scientific.get('target_semantics')
		!= 'xy_neighbor_unanimous_outlier_correction_v1'
		or scientific.get('consistency_policy')
		!= 'disabled_for_xy_neighbor_unanimous_v1'
	):
		raise ValueError('candidate unanimous target representation/semantics mismatch')


def _validate_allowed_config_delta(
	hard: Mapping[str, object], candidate: Mapping[str, object]
) -> None:
	left, right = copy.deepcopy(dict(hard)), copy.deepcopy(dict(candidate))
	for value in (left, right):
		_mapping(value.get('paths'), 'paths').pop('output_root', None)
		identity = _mapping(value.get('identity'), 'identity')
		identity.pop('model_tag', None)
		pseudo = _mapping(value.get('pseudo_targets'), 'pseudo targets')
		pseudo.pop('manifest', None)
		pseudo.pop('target_representation', None)
		scientific = _mapping(
			identity.get('scientific_identity'), 'scientific identity'
		)
		for key in _REPRESENTATION_SPECIFIC_SCIENTIFIC_FIELDS:
			scientific.pop(key, None)
	if left != right:
		raise ValueError(
			'hard/candidate scientific config drift outside allowed fields'
		)


def _validate_target_audit(
	config: F3XYNeighborUnanimousScreeningAuditConfig,
	*,
	source: Mapping[str, object],
	consensus: Mapping[str, object],
	unanimous: Mapping[str, object],
) -> Mapping[str, object]:
	"""Require the target-only gate and all three live immutable identities."""
	payload = replay_f3_xy_neighbor_unanimous_target_audit(
		config.target_audit, artifact_root=config.artifact_root
	)
	if payload.get('status') != 'XYUNANIM_TARGET_GO':
		raise ValueError('unanimous target audit must be XYUNANIM_TARGET_GO')
	expected = {
		'source_hard_manifest': config.source_hard_manifest,
		'xy_neighbor_consensus_target_manifest': (
			config.xy_neighbor_consensus_target_manifest
		),
		'xy_neighbor_unanimous_target_manifest': (
			config.xy_neighbor_unanimous_target_manifest
		),
	}
	for key, path in expected.items():
		if payload.get(key) != _target_audit_identity(path):
			raise ValueError(f'unanimous target audit {key} identity differs')
	if source.get('head_ks') != [6, 8, 10]:
		raise ValueError('source hard target K identity mismatch')
	if consensus.get('head_ks') != [6, 8, 10] or unanimous.get('head_ks') != [
		6,
		8,
		10,
	]:
		raise ValueError('unanimous target audit successor K identity mismatch')
	for head_conditions in _mapping(
		payload.get('go_conditions'), 'unanimous target audit conditions'
	).values():
		if not isinstance(head_conditions, Mapping) or not all(
			item is True for item in head_conditions.values()
		):
			raise ValueError('unanimous target audit GO conditions are incomplete')
	return payload


def _validate_target_lineage(
	config: F3XYNeighborUnanimousScreeningAuditConfig,
	*,
	source: Mapping[str, object],
	unanimous: Mapping[str, object],
	candidate: Mapping[str, object],
) -> None:
	if source.get('head_ks') != [6, 8, 10] or unanimous.get('head_ks') != [
		6,
		8,
		10,
	]:
		raise ValueError('source/unanimous target K identity mismatch')
	reference = _mapping(
		unanimous.get('source_hard_manifest'), 'unanimous source reference'
	)
	if (
		Path(str(reference.get('path', ''))).resolve() != config.source_hard_manifest
		or reference.get('sha256') != file_sha256(config.source_hard_manifest)
	):
		raise ValueError('unanimous target source hard manifest mismatch')
	identity = _mapping(
		_mapping(candidate.get('identity'), 'candidate identity').get(
			'scientific_identity'
		),
		'candidate scientific identity',
	)
	if identity.get('xy_neighbor_unanimous_target_manifest_sha256') != file_sha256(
		config.xy_neighbor_unanimous_target_manifest
	):
		raise ValueError('candidate unanimous target manifest SHA-256 mismatch')


def _parity_runtime(
	hard: Mapping[str, object], candidate: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object]]:
	return _runtime_contract(hard), _runtime_contract(candidate)


def _runtime_contract(training: Mapping[str, object]) -> Mapping[str, object]:
	train = _mapping(training.get('train'), 'training train')
	seed = train.get('seed')
	if isinstance(seed, bool) or not isinstance(seed, int):
		raise TypeError('training seed must be an integer')
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(seed)
		components = build_strat_hmm_components(training, device='cpu')
	heads = getattr(components, 'heads', None)
	if not isinstance(heads, torch.nn.Module):
		raise TypeError('unanimous audit requires multi-head components')
	parameter_names = {
		id(parameter): f'{prefix}.{name}'
		for prefix, module in (('student', components.student), ('head', heads))
		for name, parameter in module.named_parameters()
	}
	groups = []
	for group in components.optimizer.param_groups:
		parameters = group.get('params')
		if not isinstance(parameters, list):
			raise TypeError('optimizer parameter group must be a list')
		try:
			names = [parameter_names[id(parameter)] for parameter in parameters]
		except KeyError as error:
			raise ValueError('optimizer contains an unknown parameter') from error
		groups.append(
			{
				'name': group.get('name'),
				'parameter_names': names,
				'lr': float(group.get('lr', 0.0)),
			}
		)
	summary = components.trainability_summary
	return {
		'initial_student_state_sha256': hard_validation._state_sha256(
			components.student.state_dict()
		),
		'initial_head_state_sha256': hard_validation._state_sha256(heads.state_dict()),
		'trainability_summary': {
			'trainable_parameter_count': int(summary.trainable_parameter_count),
			'frozen_parameter_count': int(summary.frozen_parameter_count),
			'trainable_names': list(summary.trainable_names),
		},
		'optimizer_group_identity': groups,
	}


def _validate_runtime_parity(
	hard: Mapping[str, object],
	candidate: Mapping[str, object],
	*,
	hard_handoff: Mapping[str, object],
	candidate_handoff: Mapping[str, object],
) -> None:
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if hard.get(key) != candidate.get(key):
			raise ValueError(f'hard/candidate {key} mismatch')
	if hard.get('trainability_summary') != candidate.get('trainability_summary'):
		raise ValueError('hard/candidate trainability mismatch')
	if hard.get('optimizer_group_identity') != candidate.get(
		'optimizer_group_identity'
	):
		raise ValueError('hard/candidate optimizer group mismatch')
	hard_identity = _mapping(
		hard_handoff.get('stratigraphy_pretext'), 'hard handoff stratigraphy identity'
	)
	candidate_targets = _mapping(
		candidate_handoff.get('targets'), 'candidate handoff targets'
	)
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if hard_identity.get(key) != hard.get(key):
			raise ValueError(f'hard handoff {key} is stale')
		if candidate_targets.get(key) != candidate.get(key):
			raise ValueError(f'candidate handoff {key} is stale')


def _candidate_evidence(
	config: F3XYNeighborUnanimousScreeningAuditConfig,
	*,
	unanimous_handoff: Mapping[str, object],
) -> Mapping[str, object]:
	files = output_paths(config.candidate_embeddings_dir, 'f3_facies_benchmark')
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		if not path.is_file():
			raise FileNotFoundError(path)
	targets = _mapping(
		unanimous_handoff.get('targets'), 'candidate handoff targets'
	)
	_reference_matches(
		targets.get('target_manifest'),
		config.xy_neighbor_unanimous_target_manifest,
		label='candidate handoff unanimous target manifest',
	)
	_reference_matches(
		targets.get('target_audit'),
		config.target_audit,
		label='candidate handoff target audit',
	)
	_reference_matches(
		targets.get('source_hard_manifest'),
		config.source_hard_manifest,
		label='candidate handoff source hard manifest',
	)
	checkpoint = _mapping(
		unanimous_handoff.get('checkpoint'), 'candidate handoff checkpoint'
	)
	checkpoint_path = Path(str(checkpoint.get('path', ''))).resolve()
	if not checkpoint_path.is_file() or checkpoint.get('sha256') != file_sha256(
		checkpoint_path
	):
		raise ValueError('candidate handoff best checkpoint identity mismatch')
	if checkpoint_path.name != 'best.pt':
		raise ValueError('candidate handoff must select best.pt')
	embedding = _mapping(
		unanimous_handoff.get('embedding'), 'candidate handoff embedding'
	)
	if (
		Path(str(embedding.get('root', ''))).resolve()
		!= config.candidate_embeddings_dir
		or Path(str(embedding.get('metadata_path', ''))).resolve() != files.metadata
		or embedding.get('embeddings_sha256') != file_sha256(files.embeddings)
		or embedding.get('valid_tokens_sha256') != file_sha256(files.valid_tokens)
		or embedding.get('metadata_sha256') != file_sha256(files.metadata)
	):
		raise ValueError('candidate embedding lineage mismatch')
	_validate_schema6_candidate_identity(
		checkpoint_path,
		unanimous_handoff=unanimous_handoff,
		metadata_path=files.metadata,
		target_manifest=config.xy_neighbor_unanimous_target_manifest,
		source_manifest=config.source_hard_manifest,
	)
	return {
		'model_id': XY_UNANIM_MODEL_ID,
		'model_tag': XY_UNANIM_MODEL_TAG,
		'pretraining_handoff': _identity(config.candidate_pretraining_handoff),
		'best_checkpoint': _identity(checkpoint_path),
		'embeddings': {
			'root': str(config.candidate_embeddings_dir),
			'embeddings': _identity(files.embeddings),
			'valid_tokens': _identity(files.valid_tokens),
			'metadata': _identity(files.metadata),
		},
	}


def _validate_schema6_candidate_identity(  # noqa: C901
	checkpoint_path: Path,
	*,
	unanimous_handoff: Mapping[str, object],
	metadata_path: Path,
	target_manifest: Path,
	source_manifest: Path,
) -> None:
	"""Bind handoff, best checkpoint, and extraction metadata to schema 6."""
	payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError('candidate checkpoint must be a mapping')
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'), 'candidate checkpoint identity'
	)
	targets = _mapping(unanimous_handoff.get('targets'), 'candidate handoff targets')
	target_sha = file_sha256(target_manifest)
	source_sha = file_sha256(source_manifest)
	target = load_multi_head_xy_neighbor_unanimous_target_manifest(
		target_manifest, validate_array_semantics=False
	)
	head_hashes = _unanimous_target_head_hashes(target)
	if targets.get('xy_neighbor_unanimous_target_head_hashes') != head_hashes:
		raise ValueError('candidate handoff unanimous target head hashes mismatch')
	smoothing = target.get('smoothing')
	if (
		not isinstance(smoothing, Mapping)
		or targets.get('xy_neighbor_unanimous_smoothing') != smoothing
	):
		raise ValueError('candidate handoff unanimous smoothing mismatch')
	for key, expected in (
		('schema_version', 6),
		('model_tag', XY_UNANIM_MODEL_TAG),
		('head_spec', 'multi_resolution_ordered_prototypes_v1'),
		('head_ks', [6, 8, 10]),
		('target_representation', 'xy_neighbor_unanimous_hard_labels_v1'),
		('target_semantics', 'xy_neighbor_unanimous_outlier_correction_v1'),
		('xy_neighbor_unanimous_target_manifest_sha256', target_sha),
		('per_head_xy_neighbor_unanimous_targets', head_hashes),
		('source_hard_manifest_sha256', source_sha),
		('xy_neighbor_unanimous_smoothing', smoothing),
		('consistency_policy', 'disabled_for_xy_neighbor_unanimous_v1'),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
	):
		if identity.get(key) != expected:
			raise ValueError(f'candidate schema-6 identity mismatch: {key}')
	manifest_reference = _mapping(
		identity.get('xy_neighbor_unanimous_target_manifest'),
		'candidate checkpoint unanimous target manifest',
	)
	if (
		Path(str(manifest_reference.get('path', ''))).resolve() != target_manifest
		or manifest_reference.get('sha256') != target_sha
	):
		raise ValueError('candidate checkpoint target manifest mismatch')
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if identity.get(key) != targets.get(key):
			raise ValueError(f'candidate checkpoint initial state mismatch: {key}')
	metadata = _read_json(metadata_path)
	if (
		Path(str(metadata.get('checkpoint_path', ''))).resolve() != checkpoint_path
		or metadata.get('checkpoint_sha256') != file_sha256(checkpoint_path)
	):
		raise ValueError('candidate embedding checkpoint metadata mismatch')
	stratigraphy = _mapping(
		metadata.get('stratigraphy_pretext'), 'candidate embedding stratigraphy'
	)
	for key, expected in (
		('model_tag', XY_UNANIM_MODEL_TAG),
		('target_representation', 'xy_neighbor_unanimous_hard_labels_v1'),
		('target_semantics', 'xy_neighbor_unanimous_outlier_correction_v1'),
		('xy_neighbor_unanimous_target_manifest_path', str(target_manifest)),
		('xy_neighbor_unanimous_target_manifest_sha256', target_sha),
		('per_head_xy_neighbor_unanimous_target_sha256', head_hashes),
		('source_hard_manifest_sha256', source_sha),
		('xy_neighbor_unanimous_smoothing', smoothing),
		('consistency_policy', 'disabled_for_xy_neighbor_unanimous_v1'),
		('consistency_weight', 0.0),
		('consistency_beta', 0.1),
		('scientific_identity_sha256', identity.get('scientific_identity_sha256')),
		(
			'checkpoint_stratigraphy_state_sha256',
			identity.get('stratigraphy_state_sha256'),
		),
	):
		if stratigraphy.get(key) != expected:
			raise ValueError(f'candidate embedding schema-6 identity mismatch: {key}')
	if any(
		'posterior' in str(key)
		or 'lateral' in str(key)
		or 'xy_neighbor_consensus' in str(key)
		for key in (*identity, *stratigraphy, *metadata)
	):
		raise ValueError('candidate provenance carries legacy posterior/lateral fields')


def _unanimous_target_head_hashes(
	target: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	heads = _mapping(target.get('heads'), 'unanimous target heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for k in ('6', '8', '10'):
		head = _mapping(heads.get(k), f'unanimous target head k={k}')
		surveys = _mapping(head.get('surveys'), f'unanimous target surveys k={k}')
		result[k] = {}
		for survey_id, value in surveys.items():
			entry = _mapping(value, f'unanimous target survey k={k}/{survey_id}')
			result[k][str(survey_id)] = {}
			for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
				reference = _mapping(entry.get(name), f'unanimous target {name}')
				sha = reference.get('sha256')
				if not isinstance(sha, str) or len(sha) != 64:
					raise ValueError(f'unanimous target {name} SHA-256 is invalid')
				result[k][str(survey_id)][name] = sha
	return result


def _reference_run_manifests(
	config: F3XYNeighborUnanimousScreeningAuditConfig,
) -> Mapping[str, object]:
	"""Bind all read-only roles before candidate jobs can be planned."""
	paths = {
		'xy_neighbor_consensus': (
			config.xy_neighbor_consensus_run_manifest,
			'f3_lithology_voxel_label_budget_xy_neighbor_consensus',
			'mh_xycons1_nocons',
		),
		'hard_multi_head': (
			config.hard_reference_run_manifest,
			'f3_lithology_voxel_label_budget_multi_head',
			'mh_nocons',
		),
		'current_k6': (
			config.current_k6_run_manifest,
			'f3_lithology_voxel_label_budget_current_k6_control',
			'm1_current_k6',
		),
		'mae': (
			config.mae_reference_run_manifest,
			'f3_lithology_voxel_label_budget_run_manifest',
			'mae',
		),
	}
	for name, (path, artifact_type, role) in paths.items():
		payload = _read_json(path)
		if (
			payload.get('artifact_type') != artifact_type
			or payload.get('schema_version') != 1
		):
			raise ValueError(f'{name} reference run manifest type/schema mismatch')
		rows = payload.get('rows')
		if not isinstance(rows, list):
			raise TypeError(f'{name} reference run manifest rows are missing')
		matches = [
			row
			for row in rows
			if isinstance(row, Mapping) and row.get('model_role') == role
		]
		if len(matches) != 15 or any(
			row.get('status') != 'complete' for row in matches
		):
			raise ValueError(f'{name} reference {role} matrix is incomplete')
	return {name: _identity(path) for name, (path, _type, _role) in paths.items()}


def _target_audit_identity(path: Path) -> Mapping[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _clean_git_identity(workspace: Path) -> Mapping[str, object]:
	try:
		sha = subprocess.run(
			('git', 'rev-parse', 'HEAD'),
			cwd=workspace,
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
		status = subprocess.run(
			('git', 'status', '--porcelain'),
			cwd=workspace,
			check=True,
			capture_output=True,
			text=True,
		).stdout
	except (OSError, subprocess.CalledProcessError) as error:
		raise RuntimeError('unable to determine audit git identity') from error
	if len(sha) != 40:
		raise ValueError('audit git SHA is invalid')
	if status.strip():
		raise RuntimeError('XY-neighbour-unanimous audit requires a clean worktree')
	return {'git_sha': sha, 'dirty': False}


def _identity(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _identity_matches(value: object, expected: Path, *, label: str) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} identity is missing')
	path = value.get('path')
	if not isinstance(path, str) or Path(path).resolve() != expected.resolve():
		raise ValueError(f'{label} path mismatch')
	_identity_matches_live(value, label=label)


def _reference_matches(value: object, expected: Path, *, label: str) -> None:
	"""Match a compact ``path``/``sha256`` reference to one live file."""
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} reference is missing')
	path, sha = value.get('path'), value.get('sha256')
	if not isinstance(path, str) or Path(path).resolve() != expected.resolve():
		raise ValueError(f'{label} path mismatch')
	if not isinstance(sha, str) or sha != file_sha256(expected):
		raise ValueError(f'{label} SHA-256 mismatch')


def _identity_matches_live(value: Mapping[str, object], *, label: str) -> None:
	path = value.get('path')
	sha = value.get('sha256')
	if not isinstance(path, str) or not isinstance(sha, str):
		raise TypeError(f'{label} identity is incomplete')
	actual = Path(path)
	if not actual.is_file() or file_sha256(actual) != sha:
		raise ValueError(f'{label} SHA-256 mismatch')


def _live_identity_path(value: Mapping[str, object], *, label: str) -> Path:
	"""Return an exact live identity path after checking its declared hash."""
	_identity_matches_live(value, label=label)
	path = value.get('path')
	if not isinstance(path, str):  # Defensive narrowing after the shared check.
		raise TypeError(f'{label} path is missing')
	return Path(path).resolve()


def _model_tag(training: Mapping[str, object]) -> str:
	identity = _mapping(training.get('identity'), 'training identity')
	value = identity.get('model_tag')
	if not isinstance(value, str) or not value:
		raise TypeError('training model_tag is missing')
	return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _read_json(path: Path) -> Mapping[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'JSON object required: {path}') from error
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON object required: {path}')
	return payload


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.tmp')
	temporary.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temporary.replace(path)


def _quarantine_invalid(path: Path) -> Path:
	digest = file_sha256(path)[:12]
	quarantine = path.with_name(f'{path.stem}.invalid-{digest}{path.suffix}')
	if quarantine.exists():
		raise FileExistsError(f'audit quarantine path already exists: {quarantine}')
	path.replace(quarantine)
	return quarantine


__all__ = [
	'ARTIFACT_TYPE',
	'HARD_MODEL_TAG',
	'XY_UNANIM_MODEL_ID',
	'XY_UNANIM_MODEL_TAG',
	'F3XYNeighborUnanimousScreeningAuditConfig',
	'F3XYNeighborUnanimousScreeningAuditResult',
	'audit_f3_xy_neighbor_unanimous_screening',
	'f3_xy_neighbor_unanimous_screening_audit_config_from_mapping',
	'load_f3_xy_neighbor_unanimous_screening_audit',
	'load_f3_xy_neighbor_unanimous_screening_audit_config',
	'validate_f3_xy_neighbor_unanimous_screening_audit_binding',
]
