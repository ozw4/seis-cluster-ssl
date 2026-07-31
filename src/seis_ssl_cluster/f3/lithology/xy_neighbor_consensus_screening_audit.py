"""Read-only preflight evidence for XY-neighbour-consensus screening.

The audit is deliberately separate from decoder planning.  It validates the
frozen source and successor target publications, proves the hard/candidate
initialisation parity, and records descriptive same-z XY evidence without
altering any upstream artifact.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.f3.multi_head_pretraining_validation import (
	load_f3_multi_head_pretraining_handoff,
)
from seis_ssl_cluster.f3.xy_neighbor_consensus_pretraining_validation import (
	load_f3_xy_neighbor_consensus_pretraining_handoff,
)
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.training.strat_hmm.components import (
	build_strat_hmm_components,
)

ARTIFACT_TYPE = 'f3_xy_neighbor_consensus_original_screening_preflight'
SCHEMA_VERSION = 1
XY_MODEL_ID = 'mh_xycons1_nocons'
XY_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
HARD_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'workspace_root',
		'source_hard_manifest',
		'xy_target_manifest',
		'hard_full_config',
		'hard_pretraining_handoff',
		'candidate_full_config',
		'candidate_pretraining_handoff',
		'candidate_embeddings_dir',
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
		'xy_neighbor_consensus_target_manifest_sha256',
		'xy_neighbor_consensus_target_head_hashes',
		'xy_neighbor_consensus_smoothing',
		'source_hard_manifest_sha256',
		'supervised_loss',
		'consistency_policy',
	}
)
_REVIEWED_IDENTITIES = {
	'target_manifest_sha256': (
		'307c9a28796f7b1d90ef4f188676abc5a3604be402fe8ef4a18b515d06671a41'
	),
	'pretraining_handoff_sha256': (
		'6f7faeac191c79285a5fddd397a3da5ea65ca69db83e69cf42f18aa3afa32500'
	),
	'best_checkpoint_sha256': (
		'86cf5050e181a6cf4b254a3a377ab3f496a81b7b025d765625fe3b31a6fbfb8c'
	),
	'embedding_metadata_sha256': (
		'2d732c843542af8aad0b4f04563c23fee30675caf38085282f28ea030dd40c8c'
	),
}


@dataclass(frozen=True)
class F3XYNeighborConsensusScreeningAuditConfig:
	"""Closed paths required by the immutable XY screening preflight."""

	artifact_root: Path
	workspace_root: Path
	source_hard_manifest: Path
	xy_target_manifest: Path
	hard_full_config: Path
	hard_pretraining_handoff: Path
	candidate_full_config: Path
	candidate_pretraining_handoff: Path
	candidate_embeddings_dir: Path
	output_path: Path


@dataclass(frozen=True)
class F3XYNeighborConsensusScreeningAuditResult:
	"""The validated immutable audit payload and its write action."""

	payload: Mapping[str, object]
	output_path: Path
	action: str
	quarantine_path: Path | None


def f3_xy_neighbor_consensus_screening_audit_config_from_mapping(  # noqa: C901
	config: Mapping[str, object],
) -> F3XYNeighborConsensusScreeningAuditConfig:
	"""Resolve the closed audit schema without accepting extension fields."""
	if not isinstance(config, Mapping):
		raise TypeError(
			'XY-neighbour-consensus screening audit config must be a mapping'
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

	result = F3XYNeighborConsensusScreeningAuditConfig(
		artifact_root=path('artifact_root', must_exist=True),
		workspace_root=path('workspace_root', must_exist=True),
		source_hard_manifest=path('source_hard_manifest', must_exist=True),
		xy_target_manifest=path('xy_target_manifest', must_exist=True),
		hard_full_config=path('hard_full_config', must_exist=True),
		hard_pretraining_handoff=path('hard_pretraining_handoff', must_exist=True),
		candidate_full_config=path('candidate_full_config', must_exist=True),
		candidate_pretraining_handoff=path(
			'candidate_pretraining_handoff', must_exist=True
		),
		candidate_embeddings_dir=path('candidate_embeddings_dir', must_exist=True),
		output_path=path('output_path', must_exist=False),
	)
	if not result.artifact_root.is_dir() or not result.workspace_root.is_dir():
		raise FileNotFoundError('artifact_root and workspace_root must be directories')
	for label, value in (
		('source_hard_manifest', result.source_hard_manifest),
		('xy_target_manifest', result.xy_target_manifest),
		('hard_full_config', result.hard_full_config),
		('hard_pretraining_handoff', result.hard_pretraining_handoff),
		('candidate_full_config', result.candidate_full_config),
		('candidate_pretraining_handoff', result.candidate_pretraining_handoff),
	):
		if not value.is_file():
			raise FileNotFoundError(f'{label} is missing: {value}')
	ensure_under_root(
		result.source_hard_manifest,
		root=result.artifact_root,
		label='source_hard_manifest',
	)
	for label, value in (
		('xy_target_manifest', result.xy_target_manifest),
		('hard_pretraining_handoff', result.hard_pretraining_handoff),
		('candidate_pretraining_handoff', result.candidate_pretraining_handoff),
		('candidate_embeddings_dir', result.candidate_embeddings_dir),
		('output_path', result.output_path),
	):
		ensure_under_root(value, root=result.artifact_root, label=label)
	for label, value in (
		('hard_full_config', result.hard_full_config),
		('candidate_full_config', result.candidate_full_config),
	):
		ensure_under_root(value, root=result.workspace_root, label=label)
	return result


def load_f3_xy_neighbor_consensus_screening_audit_config(
	path: str | Path,
) -> F3XYNeighborConsensusScreeningAuditConfig:
	"""Load the standalone closed audit YAML configuration."""
	return f3_xy_neighbor_consensus_screening_audit_config_from_mapping(
		load_config(path)
	)


def audit_f3_xy_neighbor_consensus_screening(
	config: F3XYNeighborConsensusScreeningAuditConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3XYNeighborConsensusScreeningAuditResult:
	"""Build or immutably reuse the preflight evidence without mutating inputs."""
	if only_missing and not dry_run and config.output_path.is_file():
		# Existing artifacts are still recomputed below before reuse is admitted.
		pass
	git = _clean_git_identity(config.workspace_root)
	payload = _audit_payload(config, git=git)
	if dry_run:
		return F3XYNeighborConsensusScreeningAuditResult(
			payload, config.output_path, 'DRY_RUN', None
		)
	if config.output_path.exists():
		if not config.output_path.is_file():
			raise FileExistsError(
				f'audit output path is not a file: {config.output_path}'
			)
		try:
			existing = load_f3_xy_neighbor_consensus_screening_audit(config.output_path)
		except (TypeError, ValueError):
			existing = None
		if existing == payload:
			if only_missing:
				return F3XYNeighborConsensusScreeningAuditResult(
					payload, config.output_path, 'REUSE_COMPLETED', None
				)
			raise FileExistsError(
				f'audit already exists; use --only-missing: {config.output_path}'
			)
		if not quarantine_invalid:
			raise ValueError(
				'incompatible existing XY-neighbour-consensus audit; '
				'use --quarantine-invalid to replace it'
			)
		quarantine = _quarantine_invalid(config.output_path)
	else:
		quarantine = None
	_write_json_atomically(config.output_path, payload)
	return F3XYNeighborConsensusScreeningAuditResult(
		payload, config.output_path, 'WRITTEN', quarantine
	)


def load_f3_xy_neighbor_consensus_screening_audit(
	path: str | Path,
) -> Mapping[str, object]:
	"""Load only a complete, schema-v1 PASS screening preflight."""
	payload = _read_json(Path(path))
	if (
		payload.get('artifact_type') != ARTIFACT_TYPE
		or payload.get('schema_version') != SCHEMA_VERSION
		or payload.get('status') != 'PASS'
	):
		raise ValueError(
			'XY-neighbour-consensus screening audit type/schema/status mismatch'
		)
	for key in ('candidate', 'hard_baseline_parity', 'xy_spatial_smoothness', 'git'):
		if not isinstance(payload.get(key), Mapping):
			raise TypeError(f'XY-neighbour-consensus screening audit {key} is missing')
	return payload


def validate_f3_xy_neighbor_consensus_screening_audit_binding(
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
		raise ValueError('XY-neighbour-consensus screening audit identity mismatch')
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


def _audit_payload(
	config: F3XYNeighborConsensusScreeningAuditConfig,
	*,
	git: Mapping[str, object],
) -> dict[str, object]:
	"""Build every audit assertion before a single output byte is written."""
	hard = resolve_strat_hmm_pretext_config(load_config(config.hard_full_config))
	candidate = resolve_strat_hmm_pretext_config(
		load_config(config.candidate_full_config)
	)
	_validate_training_identity(config, hard=hard, candidate=candidate)
	_validate_allowed_config_delta(hard, candidate)
	source = load_multi_head_target_manifest(config.source_hard_manifest)
	target = load_multi_head_xy_neighbor_consensus_target_manifest(
		config.xy_target_manifest
	)
	_validate_target_lineage(config, source=source, target=target, candidate=candidate)
	hard_handoff = load_f3_multi_head_pretraining_handoff(
		config.hard_pretraining_handoff
	)
	xy_handoff = load_f3_xy_neighbor_consensus_pretraining_handoff(
		config.candidate_pretraining_handoff
	)
	hard_runtime, candidate_runtime = _parity_runtime(hard, candidate)
	_validate_runtime_parity(
		hard_runtime,
		candidate_runtime,
		hard_handoff=hard_handoff,
		candidate_handoff=xy_handoff,
	)
	candidate_evidence = _candidate_evidence(config, xy_handoff=xy_handoff)
	_reviewed_identity_evidence(
		candidate_evidence,
		target_manifest_sha256=file_sha256(config.xy_target_manifest),
	)
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': 'PASS',
		'git': dict(git),
		'artifact_root': str(config.artifact_root),
		'workspace_root': str(config.workspace_root),
		'source_hard_manifest': _identity(config.source_hard_manifest),
		'xy_target_manifest': _identity(config.xy_target_manifest),
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
		'xy_spatial_smoothness': {'per_k': _spatial_smoothness(source, target)},
		'reviewed_identity_check': {
			'expected': dict(_REVIEWED_IDENTITIES),
			'status': 'PASS',
		},
	}


def _validate_training_identity(
	config: F3XYNeighborConsensusScreeningAuditConfig,
	*,
	hard: Mapping[str, object],
	candidate: Mapping[str, object],
) -> None:
	if _model_tag(hard) != HARD_MODEL_TAG or _model_tag(candidate) != XY_MODEL_TAG:
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
		!= config.xy_target_manifest
	):
		raise ValueError('candidate full config XY target manifest mismatch')
	scientific = _mapping(
		_mapping(candidate.get('identity'), 'candidate identity').get(
			'scientific_identity'
		),
		'candidate scientific identity',
	)
	if (
		scientific.get('target_representation')
		!= 'xy_neighbor_consensus_hard_labels_v1'
		or scientific.get('target_semantics')
		!= 'xy_neighbor_consensus_hard_label_smoothing_v1'
	):
		raise ValueError('candidate XY target representation/semantics mismatch')


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


def _validate_target_lineage(
	config: F3XYNeighborConsensusScreeningAuditConfig,
	*,
	source: Mapping[str, object],
	target: Mapping[str, object],
	candidate: Mapping[str, object],
) -> None:
	if source.get('head_ks') != [6, 8, 10] or target.get('head_ks') != [6, 8, 10]:
		raise ValueError('source/XY target K identity mismatch')
	reference = _mapping(target.get('source_hard_manifest'), 'XY source reference')
	if Path(
		str(reference.get('path', ''))
	).resolve() != config.source_hard_manifest or reference.get(
		'sha256'
	) != file_sha256(config.source_hard_manifest):
		raise ValueError('XY target source hard manifest mismatch')
	identity = _mapping(
		_mapping(candidate.get('identity'), 'candidate identity').get(
			'scientific_identity'
		),
		'candidate scientific identity',
	)
	if identity.get('xy_neighbor_consensus_target_manifest_sha256') != file_sha256(
		config.xy_target_manifest
	):
		raise ValueError('candidate XY target manifest SHA-256 mismatch')


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
		raise TypeError('XY audit requires multi-head components')
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
	config: F3XYNeighborConsensusScreeningAuditConfig,
	*,
	xy_handoff: Mapping[str, object],
) -> Mapping[str, object]:
	files = output_paths(config.candidate_embeddings_dir, 'f3_facies_benchmark')
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		if not path.is_file():
			raise FileNotFoundError(path)
	checkpoint = _mapping(xy_handoff.get('checkpoint'), 'candidate handoff checkpoint')
	checkpoint_path = Path(str(checkpoint.get('path', ''))).resolve()
	if not checkpoint_path.is_file() or checkpoint.get('sha256') != file_sha256(
		checkpoint_path
	):
		raise ValueError('candidate handoff best checkpoint identity mismatch')
	embedding = _mapping(xy_handoff.get('embedding'), 'candidate handoff embedding')
	if (
		Path(str(embedding.get('root', ''))).resolve()
		!= config.candidate_embeddings_dir
		or Path(str(embedding.get('metadata_path', ''))).resolve() != files.metadata
		or embedding.get('embeddings_sha256') != file_sha256(files.embeddings)
		or embedding.get('valid_tokens_sha256') != file_sha256(files.valid_tokens)
		or embedding.get('metadata_sha256') != file_sha256(files.metadata)
	):
		raise ValueError('candidate embedding lineage mismatch')
	return {
		'model_id': XY_MODEL_ID,
		'model_tag': XY_MODEL_TAG,
		'pretraining_handoff': _identity(config.candidate_pretraining_handoff),
		'best_checkpoint': _identity(checkpoint_path),
		'embeddings': {
			'root': str(config.candidate_embeddings_dir),
			'embeddings': _identity(files.embeddings),
			'valid_tokens': _identity(files.valid_tokens),
			'metadata': _identity(files.metadata),
		},
	}


def _reviewed_identity_evidence(
	candidate: Mapping[str, object], *, target_manifest_sha256: str
) -> None:
	best = _mapping(candidate.get('best_checkpoint'), 'candidate best checkpoint')
	handoff = _mapping(candidate.get('pretraining_handoff'), 'candidate handoff')
	embeddings = _mapping(candidate.get('embeddings'), 'candidate embeddings')
	metadata = _mapping(embeddings.get('metadata'), 'candidate embedding metadata')
	live = {
		'target_manifest_sha256': target_manifest_sha256,
		'pretraining_handoff_sha256': handoff.get('sha256'),
		'best_checkpoint_sha256': best.get('sha256'),
		'embedding_metadata_sha256': metadata.get('sha256'),
	}
	for key, expected in live.items():
		if _REVIEWED_IDENTITIES[key] != expected:
			raise ValueError(f'reviewed identity drift: {key}')


def _spatial_smoothness(
	source: Mapping[str, object], target: Mapping[str, object]
) -> dict[str, object]:
	"""Measure each same-z x/y edge exactly once for every canonical head."""
	result: dict[str, object] = {}
	for k in (6, 8, 10):
		source_surveys = _surveys(source, k)
		target_surveys = _surveys(target, k)
		if set(source_surveys) != set(target_surveys):
			raise ValueError(f'XY spatial audit survey mismatch for K={k}')
		entries = []
		for survey in sorted(source_surveys):
			source_entry = _mapping(source_surveys[survey], 'source survey')
			target_entry = _mapping(target_surveys[survey], 'target survey')
			source_labels = _load_array(source_entry.get('labels'), 'source labels')
			valid = _load_array(source_entry.get('valid_tokens'), 'source valid tokens')
			output_labels = _load_array(target_entry.get('labels'), 'output labels')
			output_valid = _load_array(
				target_entry.get('valid_tokens'), 'output valid tokens'
			)
			if not np.array_equal(valid, output_valid):
				raise ValueError(f'XY spatial audit valid-mask mismatch for K={k}')
			entries.append(_spatial_one(source_labels, output_labels, valid, k=k))
		result[str(k)] = _merge_spatial(entries, k=k)
	return result


def _surveys(payload: Mapping[str, object], k: int) -> Mapping[str, object]:
	head = _mapping(_mapping(payload.get('heads'), 'heads').get(str(k)), f'K={k}')
	return _mapping(head.get('surveys'), f'K={k} surveys')


def _load_array(reference: object, label: str) -> np.ndarray:
	entry = _mapping(reference, label)
	path = Path(str(entry.get('path', '')))
	if not path.is_file() or entry.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} identity mismatch')
	return np.load(path, mmap_mode='r', allow_pickle=False)


def _spatial_one(
	source: np.ndarray, output: np.ndarray, valid: np.ndarray, *, k: int
) -> Mapping[str, object]:
	if source.shape != output.shape or source.shape != valid.shape:
		raise ValueError('XY spatial audit array shape mismatch')
	if source.ndim != 3 or valid.dtype != np.bool_:
		raise ValueError('XY spatial audit source array contract mismatch')
	x = _edge_metrics(
		source[:-1], source[1:], output[:-1], output[1:], valid[:-1] & valid[1:]
	)
	y = _edge_metrics(
		source[:, :-1],
		source[:, 1:],
		output[:, :-1],
		output[:, 1:],
		valid[:, :-1] & valid[:, 1:],
	)
	return {
		'x_edges': x,
		'y_edges': y,
		'valid_token_count': int(valid.sum()),
		'changed_token_count': int(np.count_nonzero(source[valid] != output[valid])),
		'source_state_occupancy': np.bincount(source[valid], minlength=k).tolist(),
		'output_state_occupancy': np.bincount(output[valid], minlength=k).tolist(),
		'source_temporal_transition_count': _temporal_transitions(source, valid),
		'output_temporal_transition_count': _temporal_transitions(output, valid),
		'ordered_path_violations': {
			'source': _ordered_path_violations(source, valid),
			'output': _ordered_path_violations(output, valid),
		},
	}


def _edge_metrics(
	source_left: np.ndarray,
	source_right: np.ndarray,
	output_left: np.ndarray,
	output_right: np.ndarray,
	edges: np.ndarray,
) -> Mapping[str, object]:
	count = int(edges.sum())
	source_disagreements = int(np.count_nonzero((source_left != source_right) & edges))
	output_disagreements = int(np.count_nonzero((output_left != output_right) & edges))
	source_fraction = source_disagreements / count if count else 0.0
	output_fraction = output_disagreements / count if count else 0.0
	reduction = source_fraction - output_fraction
	return {
		'valid_edge_count': count,
		'source_disagreement_count': source_disagreements,
		'source_disagreement_fraction': source_fraction,
		'output_disagreement_count': output_disagreements,
		'output_disagreement_fraction': output_fraction,
		'absolute_disagreement_reduction': reduction,
		'relative_disagreement_reduction': reduction / source_fraction
		if source_fraction
		else 0.0,
	}


def _merge_spatial(
	entries: Sequence[Mapping[str, object]], *, k: int
) -> Mapping[str, object]:
	if not entries:
		raise ValueError(f'XY spatial audit K={k} has no surveys')
	x = _sum_edge_metrics([_mapping(value['x_edges'], 'x edges') for value in entries])
	y = _sum_edge_metrics([_mapping(value['y_edges'], 'y edges') for value in entries])
	combined = _sum_edge_metrics((x, y))
	valid_tokens = sum(int(value['valid_token_count']) for value in entries)
	changed = sum(int(value['changed_token_count']) for value in entries)
	source_occupancy = [0] * k
	output_occupancy = [0] * k
	for value in entries:
		for index, count in enumerate(value['source_state_occupancy']):
			source_occupancy[index] += int(count)
		for index, count in enumerate(value['output_state_occupancy']):
			output_occupancy[index] += int(count)
	return {
		'x_edges': x,
		'y_edges': y,
		'combined': combined,
		'valid_token_count': valid_tokens,
		'changed_token_count': changed,
		'changed_fraction': changed / valid_tokens if valid_tokens else 0.0,
		'source_state_occupancy': source_occupancy,
		'output_state_occupancy': output_occupancy,
		'empty_output_state_count': sum(value == 0 for value in output_occupancy),
		'source_temporal_transition_count': sum(
			int(value['source_temporal_transition_count']) for value in entries
		),
		'output_temporal_transition_count': sum(
			int(value['output_temporal_transition_count']) for value in entries
		),
		'ordered_path_violations': {
			'source': sum(
				int(_mapping(value['ordered_path_violations'], 'violations')['source'])
				for value in entries
			),
			'output': sum(
				int(_mapping(value['ordered_path_violations'], 'violations')['output'])
				for value in entries
			),
		},
	}


def _sum_edge_metrics(edges: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
	count = sum(int(edge['valid_edge_count']) for edge in edges)
	source = sum(int(edge['source_disagreement_count']) for edge in edges)
	output = sum(int(edge['output_disagreement_count']) for edge in edges)
	source_fraction = source / count if count else 0.0
	output_fraction = output / count if count else 0.0
	reduction = source_fraction - output_fraction
	return {
		'valid_edge_count': count,
		'source_disagreement_count': source,
		'source_disagreement_fraction': source_fraction,
		'output_disagreement_count': output,
		'output_disagreement_fraction': output_fraction,
		'absolute_disagreement_reduction': reduction,
		'relative_disagreement_reduction': reduction / source_fraction
		if source_fraction
		else 0.0,
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
		raise RuntimeError('XY-neighbour-consensus audit requires a clean worktree')
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


def _identity_matches_live(value: Mapping[str, object], *, label: str) -> None:
	path = value.get('path')
	sha = value.get('sha256')
	if not isinstance(path, str) or not isinstance(sha, str):
		raise TypeError(f'{label} identity is incomplete')
	actual = Path(path)
	if not actual.is_file() or file_sha256(actual) != sha:
		raise ValueError(f'{label} SHA-256 mismatch')


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
	'F3XYNeighborConsensusScreeningAuditConfig',
	'F3XYNeighborConsensusScreeningAuditResult',
	'audit_f3_xy_neighbor_consensus_screening',
	'f3_xy_neighbor_consensus_screening_audit_config_from_mapping',
	'load_f3_xy_neighbor_consensus_screening_audit',
	'load_f3_xy_neighbor_consensus_screening_audit_config',
	'validate_f3_xy_neighbor_consensus_screening_audit_binding',
]
