"""Immutable one-step lateral hard-target exports for frozen stratigraphic HMMs.

The exporter deliberately only consumes the existing hard-label and exact
posterior publications.  It performs no fitting and writes a separate target
contract, so the historical pseudo-target manifest remains untouched.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from csv import DictReader, DictWriter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	discover_embedding_inputs,
	file_sha256,
)
from seis_ssl_cluster.stratigraphy.frozen_hmm_replay import (
	CANONICAL_KS,
	expected_boundaries,
	load_frozen_hmm_model,
	replay_frozen_hmm_trace,
)
from seis_ssl_cluster.stratigraphy.lateral_smoothing import (
	LATERAL_SMOOTHING_SEMANTICS,
	LateralSmoothingResult,
	smooth_and_redecode_ordered_trace,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.state_posterior import (
	_validate_source_embedding_identity,
	hard_source_model_identities,
	validate_multi_head_state_posterior_manifest,
)
from seis_ssl_cluster.stratigraphy.targets import build_pseudo_target_metadata

_BUNDLE_DIRNAME = 'bundle'

ARTIFACT_TYPE = 'strat_hmm_multi_head_lateral_target_manifest'
SCHEMA_VERSION = 1
_LABEL_SUFFIX = '.labels.npy'
_CONFIDENCE_SUFFIX = '.confidence.npy'
_VALID_SUFFIX = '.valid_tokens.npy'
_METADATA_SUFFIX = '.metadata.json'
_Action = Literal['NEW', 'REUSE', 'QUARANTINE', 'ERROR']
_AFFINITY_SCALE_POLICY = 'global_valid_xy_edge_distance_median_floor_1e-6_v1'
_EMISSION_SCALE_POLICY = 'per_head_valid_second_gap_median_floor_1e-6_v1'


@dataclass(frozen=True)
class MultiHeadLateralTargetExportConfig:
	"""Validated fixed-policy lateral target export configuration."""

	source_hard_manifest: Path
	source_posterior_manifest: Path
	clustering_output_dir: Path
	clustering_config: Path
	source_embedding_dir: Path
	output_root: Path
	pairwise_strength_ratio: float
	handoff_manifest: Path


@dataclass(frozen=True)
class MultiHeadLateralTargetExportPlan:
	"""One head action selected after source validation."""

	k: int
	action: _Action
	reason: str | None = None


@dataclass(frozen=True)
class _LateralPreflight:
	"""One immutable read-only source and scale decision for an export."""

	source: Mapping[str, object]
	posterior: Mapping[str, object]
	inputs: Mapping[str, EmbeddingInput]
	models: Mapping[int, Mapping[str, object]]
	affinity_scale: float
	affinity_stats: Mapping[str, object]
	gap_scales: Mapping[int, float]
	gap_stats: Mapping[int, Mapping[str, object]]
	snapshot: str
	plans: tuple[MultiHeadLateralTargetExportPlan, ...]


class _OwnedOutputCorruptionError(ValueError):
	"""An owned output is incomplete or fails its own semantic contract."""


class _ImmutableIdentityMismatchError(ValueError):
	"""A complete owned output belongs to a different scientific identity."""


def resolve_multi_head_lateral_target_export_config(
	config: Mapping[str, object],
) -> MultiHeadLateralTargetExportConfig:
	"""Resolve the intentionally non-extensible M5-LS exporter configuration."""
	allowed = {
		'source_hard_manifest',
		'source_posterior_manifest',
		'clustering_output_dir',
		'clustering_config',
		'source_embedding_dir',
		'output_root',
		'smoothing',
		'handoff_manifest',
		'outputs',
	}
	if set(config) - allowed:
		raise ValueError(
			f'unknown lateral-target config keys: {sorted(set(config) - allowed)}'
		)
	if config.get('outputs') not in (None, {'overwrite': False}):
		raise ValueError('outputs must be omitted or exactly {overwrite: false}')

	def required(name: str) -> Path:
		value = config.get(name)
		if not isinstance(value, str) or not value:
			raise TypeError(f'{name} must be a non-empty path')
		return Path(value)

	smoothing = _mapping(config.get('smoothing'), 'smoothing')
	if set(smoothing) != {'pairwise_strength_ratio'}:
		raise ValueError('smoothing must contain only pairwise_strength_ratio')
	beta = smoothing['pairwise_strength_ratio']
	if (
		isinstance(beta, bool)
		or not isinstance(beta, (int, float))
		or not math.isfinite(beta)
		or beta <= 0
	):
		raise ValueError('pairwise_strength_ratio must be positive and finite')
	source_hard = required('source_hard_manifest')
	source_posterior = required('source_posterior_manifest')
	clustering_config = required('clustering_config')
	if (
		not source_hard.is_file()
		or not source_posterior.is_file()
		or not clustering_config.is_file()
	):
		raise FileNotFoundError('source manifests and clustering_config must exist')
	output_root = required('output_root')
	handoff = config.get('handoff_manifest')
	return MultiHeadLateralTargetExportConfig(
		source_hard,
		source_posterior,
		required('clustering_output_dir'),
		clustering_config,
		required('source_embedding_dir'),
		output_root,
		float(beta),
		output_root / 'multi_head_lateral_target_handoff.json'
		if handoff is None
		else Path(_string(handoff, 'handoff_manifest')),
	)


def plan_multi_head_lateral_target_exports(
	config: MultiHeadLateralTargetExportConfig,
	*,
	only_missing: bool,
) -> list[MultiHeadLateralTargetExportPlan]:
	"""Completely validate sources and classify output paths without writing."""
	return list(_preflight(config, only_missing=only_missing).plans)


def _preflight(
	config: MultiHeadLateralTargetExportConfig, *, only_missing: bool
) -> _LateralPreflight:
	"""Run every read-only check required before an owned path may change."""
	source, posterior, inputs, models = _validate_sources(config)
	_validate_frozen_source_replay(source, inputs, models)
	affinity_scale, affinity_stats = _affinity_scale(source, posterior, inputs)
	gap_scales, gap_stats = _emission_gap_scales(source, inputs, models)
	bundle = _bundle_path(config)
	if not bundle.exists():
		action = 'QUARANTINE' if config.handoff_manifest.exists() else 'NEW'
		reason = (
			'handoff exists without a complete bundle'
			if action == 'QUARANTINE'
			else None
		)
		plans = tuple(
			MultiHeadLateralTargetExportPlan(k, action, reason) for k in CANONICAL_KS
		)
	else:
		try:
			for k in CANONICAL_KS:
				_validate_complete_head(
					bundle / f'k{k}',
					k,
					source,
					posterior,
					models,
					config,
					inputs=inputs,
					expected_scales={
						'affinity': affinity_stats,
						'emission_gap': gap_stats[k],
					},
				)
			_validate_owned_handoff(config, source)
		except _ImmutableIdentityMismatchError as exc:
			plans = tuple(
				MultiHeadLateralTargetExportPlan(k, 'ERROR', str(exc))
				for k in CANONICAL_KS
			)
		except (OSError, TypeError, ValueError) as exc:
			corruption = _OwnedOutputCorruptionError(str(exc))
			plans = tuple(
				MultiHeadLateralTargetExportPlan(k, 'QUARANTINE', str(corruption))
				for k in CANONICAL_KS
			)
		else:
			plans = tuple(
				MultiHeadLateralTargetExportPlan(
					k,
					'REUSE' if only_missing else 'ERROR',
					None
					if only_missing
					else 'complete output exists; use --only-missing',
				)
				for k in CANONICAL_KS
			)
	snapshot = _source_snapshot(config, source, posterior, inputs, models)
	return _LateralPreflight(
		source,
		posterior,
		inputs,
		models,
		affinity_scale,
		affinity_stats,
		gap_scales,
		gap_stats,
		snapshot,
		plans,
	)


def export_multi_head_lateral_targets(  # noqa: C901
	config: MultiHeadLateralTargetExportConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
) -> list[MultiHeadLateralTargetExportPlan]:
	"""Publish all heads atomically after a bounded preflight and export."""
	preflight = _preflight(config, only_missing=only_missing)
	plans = list(preflight.plans)
	if dry_run:
		return plans
	if any(plan.action == 'ERROR' for plan in plans):
		raise FileExistsError(
			'; '.join(f'k={p.k}: {p.reason}' for p in plans if p.action == 'ERROR')
		)
	if all(plan.action == 'REUSE' for plan in plans):
		return plans
	config.output_root.mkdir(parents=True, exist_ok=True)
	staging = Path(tempfile.mkdtemp(prefix='.lateral.bundle.', dir=config.output_root))
	try:
		for plan in plans:
			if plan.action != 'REUSE':
				_export_head(
					staging / f'k{plan.k}',
					plan.k,
					preflight.source,
					preflight.posterior,
					preflight.inputs,
					preflight.models[plan.k],
					config,
					preflight.affinity_scale,
					preflight.affinity_stats,
					preflight.gap_scales[plan.k],
					preflight.gap_stats[plan.k],
				)
		for plan in plans:
			_validate_complete_head(
				staging / f'k{plan.k}',
				plan.k,
				preflight.source,
				preflight.posterior,
				preflight.models,
				config,
				allow_staging=True,
				inputs=preflight.inputs,
				expected_scales={
					'affinity': preflight.affinity_stats,
					'emission_gap': preflight.gap_stats[plan.k],
				},
			)
			_rebase_head_metadata(
				staging / f'k{plan.k}', old_root=staging, new_root=_bundle_path(config)
			)
		_validate_live_snapshot(config, preflight)
		# Preserve a previous broken generation only after the replacement has
		# passed complete staging validation and source revalidation.
		if any(plan.action == 'QUARANTINE' for plan in plans):
			if config.handoff_manifest.exists():
				_quarantine(config.handoff_manifest)
			_quarantine(_bundle_path(config))
		staging.replace(_bundle_path(config))
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	else:
		shutil.rmtree(staging, ignore_errors=True)
	for k in CANONICAL_KS:
		_validate_complete_head(
			_bundle_path(config) / f'k{k}',
			k,
			preflight.source,
			preflight.posterior,
			preflight.models,
			config,
			inputs=preflight.inputs,
			expected_scales={
				'affinity': preflight.affinity_stats,
				'emission_gap': preflight.gap_stats[k],
			},
		)
	_publish_manifest(config, preflight.source)
	return plans


def load_multi_head_lateral_target_manifest(
	path: str | Path, *, validate_array_semantics: bool = True
) -> dict[str, object]:
	"""Load a lateral manifest, optionally using reference-only validation."""
	try:
		payload = json.loads(Path(path).read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'lateral target manifest must be valid JSON: {path}') from exc
	if not isinstance(payload, dict):
		raise TypeError('lateral target manifest must be an object')
	validate_multi_head_lateral_target_manifest(
		payload, validate_array_semantics=validate_array_semantics
	)
	return payload


def validate_multi_head_lateral_target_manifest(  # noqa: C901, PLR0912, PLR0915
	payload: Mapping[str, object],
	*,
	validate_array_semantics: bool = True,
) -> None:
	"""Validate strict source bindings, head arrays, and their hard semantics."""
	_required(
		payload,
		{
			'artifact_type',
			'schema_version',
			'target_semantics',
			'head_ks',
			'source_hard_manifest',
			'source_posterior_manifest',
			'source_embedding',
			'smoothing',
			'heads',
		},
	)
	if (
		payload['artifact_type'],
		payload['schema_version'],
		payload['target_semantics'],
		payload['head_ks'],
	) != (
		ARTIFACT_TYPE,
		SCHEMA_VERSION,
		LATERAL_SMOOTHING_SEMANTICS,
		list(CANONICAL_KS),
	):
		raise ValueError('unsupported lateral target manifest schema')
	hard_path = _hashed(payload['source_hard_manifest'], 'source_hard_manifest')
	posterior_path = _hashed(
		payload['source_posterior_manifest'], 'source_posterior_manifest'
	)
	hard = load_multi_head_target_manifest(hard_path)
	posterior = json.loads(posterior_path.read_text(encoding='utf-8'))
	if not isinstance(posterior, Mapping):
		raise TypeError('source posterior manifest must be an object')
	validate_multi_head_state_posterior_manifest(posterior)
	if (
		posterior['source_hard_manifest'] != _reference(hard_path)
		or payload['source_embedding'] != hard['source_embedding']
		or posterior['source_embedding'] != hard['source_embedding']
	):
		raise ValueError('lateral source provenance does not agree')
	_smoothing(payload['smoothing'])
	heads = _mapping(payload['heads'], 'heads')
	if set(heads) != {str(k) for k in CANONICAL_KS}:
		raise ValueError('lateral manifest heads must contain K=6/8/10')
	common: dict[str, np.ndarray] = {}
	for k in CANONICAL_KS:
		head = _mapping(heads[str(k)], f'head k={k}')
		_required(head, {'model', 'surveys', 'diagnostics'})
		if set(_mapping(head['surveys'], 'surveys')) != set(
			_mapping(_mapping(hard['heads'], 'hard heads')[str(k)], 'hard head')[
				'surveys'
			]
		):
			raise ValueError(f'lateral survey set differs for k={k}')
		if (
			head['model']
			!= _mapping(posterior['heads'], 'posterior heads')[str(k)]['model']
		):
			raise ValueError(f'lateral model identity differs for k={k}')
		_validate_lateral_diagnostics(
			head['diagnostics'],
			survey_ids=set(_mapping(head['surveys'], 'surveys')),
		)
		resolved_scales = _resolved_scales(head['diagnostics'])
		for survey_id, raw in _mapping(head['surveys'], 'surveys').items():
			entry = _mapping(raw, 'survey')
			hard_survey = _mapping(
				_mapping(_mapping(hard['heads'], 'hard heads')[str(k)], 'hard head')[
					'surveys'
				][survey_id],
				'hard survey',
			)
			posterior_survey = _mapping(
				_mapping(
					_mapping(posterior['heads'], 'posterior heads')[str(k)], 'head'
				)['surveys'][survey_id],
				'posterior survey',
			)
			_required(
				entry,
				{
					'labels',
					'confidence',
					'valid_tokens',
					'metadata',
					'source_hard_labels',
					'source_posterior',
				},
			)
			labels = np.load(
				_hashed(entry['labels'], 'labels'), mmap_mode='r', allow_pickle=False
			)
			confidence = np.load(
				_hashed(entry['confidence'], 'confidence'),
				mmap_mode='r',
				allow_pickle=False,
			)
			valid = np.load(
				_hashed(entry['valid_tokens'], 'valid_tokens'),
				mmap_mode='r',
				allow_pickle=False,
			)
			_validate_array_reference(entry['labels'], labels, name='labels')
			_validate_array_reference(
				entry['confidence'], confidence, name='confidence'
			)
			_validate_array_reference(entry['valid_tokens'], valid, name='valid_tokens')
			if (
				labels.dtype != np.int32
				or labels.ndim != 3
				or confidence.dtype != np.float32
				or confidence.shape != labels.shape
				or valid.dtype != np.bool_
				or valid.shape != labels.shape
			):
				raise ValueError('lateral arrays have invalid shape or dtype')
			if validate_array_semantics and (
				np.any(labels[valid] < 0)
				or np.any(labels[valid] >= k)
				or np.any(labels[~valid] != -1)
				or np.any(confidence[valid] != 1.0)
				or np.any(confidence[~valid] != 0.0)
			):
				raise ValueError('lateral hard-label semantics are invalid')
			if validate_array_semantics:
				_validate_target_valid_tokens(
					valid,
					hard_survey,
					posterior_survey,
					context=f'k={k} {survey_id}',
				)
				if np.any(np.bincount(labels[valid], minlength=k) == 0):
					raise ValueError('lateral labels contain an empty state')
			if entry['source_hard_labels'] != _reference(
				_source_label_path(hard_survey)
			):
				raise ValueError('lateral hard-label provenance differs')
			if entry['source_posterior'] != posterior_survey['posterior']:
				raise ValueError('lateral posterior provenance differs')
			_validate_survey_metadata(
				entry,
				k=k,
				survey_id=str(survey_id),
				identity={
					'source_embedding': _source_embedding_survey(
						payload['source_embedding'], survey_id
					),
					'smoothing': payload['smoothing'],
					'resolved_scales': resolved_scales,
				},
			)
			if validate_array_semantics:
				for x in range(labels.shape[0]):
					for y in range(labels.shape[1]):
						trace = labels[x, y, valid[x, y]]
						if np.any(np.diff(trace) < 0):
							raise ValueError('lateral labels violate ordered paths')
				if survey_id in common and not np.array_equal(common[survey_id], valid):
					raise ValueError(f'valid mask differs across heads for {survey_id}')
				common.setdefault(survey_id, np.asarray(valid))
		if validate_array_semantics:
			_validate_diagnostics_from_arrays(
				head['diagnostics'],
				_mapping(head['surveys'], 'surveys'),
				_mapping(_mapping(hard['heads'], 'hard heads')[str(k)], 'hard head')[
					'surveys'
				],
				k=k,
			)
	if validate_array_semantics:
		_validate_diagnostics_from_frozen_manifest_sources(
			payload,
			hard_path=hard_path,
			posterior_path=posterior_path,
		)


def _validate_diagnostics_from_frozen_manifest_sources(
	payload: Mapping[str, object],
	*,
	hard_path: Path,
	posterior_path: Path,
) -> None:
	"""Replay complete diagnostics for a consumer-loaded lateral manifest."""
	hard = load_multi_head_target_manifest(hard_path)
	_, model_identities = hard_source_model_identities(hard)
	model_identity = _mapping(model_identities[str(CANONICAL_KS[0])], 'model identity')
	centers_path = _hashed(model_identity['centers'], 'model centers')
	model_dir = centers_path.parent
	if model_dir.name != f'k{CANONICAL_KS[0]}' or model_dir.parent.name != 'models':
		raise ValueError('frozen model layout is invalid')
	clustering_output_dir = model_dir.parent.parent
	clustering_config = _hashed(
		model_identity['clustering_config'], 'clustering config'
	)
	source_embedding_dir = Path(
		_string(
			_mapping(hard['source_embedding'], 'source_embedding').get('input_dir'),
			'source_embedding.input_dir',
		)
	)
	smoothing = _mapping(payload['smoothing'], 'smoothing')
	config = MultiHeadLateralTargetExportConfig(
		source_hard_manifest=hard_path,
		source_posterior_manifest=posterior_path,
		clustering_output_dir=clustering_output_dir,
		clustering_config=clustering_config,
		source_embedding_dir=source_embedding_dir,
		output_root=hard_path.parent,
		pairwise_strength_ratio=float(smoothing['pairwise_strength_ratio']),
		handoff_manifest=hard_path,
	)
	source, posterior, inputs, models = _validate_sources(config)
	_validate_frozen_source_replay(source, inputs, models)
	_, affinity_stats = _affinity_scale(source, posterior, inputs)
	_, gap_stats = _emission_gap_scales(source, inputs, models)
	for k in CANONICAL_KS:
		head = _mapping(_mapping(payload['heads'], 'heads')[str(k)], f'head k={k}')
		diagnostics = head['diagnostics']
		expected_scales = {
			'affinity': affinity_stats,
			'emission_gap': gap_stats[k],
		}
		if _mapping(diagnostics, 'diagnostics')['resolved_scales'] != expected_scales:
			raise ValueError('resolved scales differ from frozen sources')
		_validate_diagnostics_from_frozen_sources(
			diagnostics,
			_mapping(head['surveys'], 'surveys'),
			_mapping(_mapping(source['heads'], 'heads')[str(k)], 'hard head')[
				'surveys'
			],
			_mapping(_mapping(posterior['heads'], 'posterior heads')[str(k)], 'head')[
				'surveys'
			],
			inputs,
			models[k],
			k=k,
			config=config,
			expected_scales=expected_scales,
		)


def _validate_sources(  # noqa: C901
	config: MultiHeadLateralTargetExportConfig,
) -> tuple[
	Mapping[str, object],
	Mapping[str, object],
	Mapping[str, EmbeddingInput],
	Mapping[int, Mapping[str, object]],
]:
	hard = load_multi_head_target_manifest(config.source_hard_manifest)
	if hard.get('head_ks') != list(CANONICAL_KS):
		raise ValueError('source hard manifest must use K=6/8/10')
	posterior = json.loads(config.source_posterior_manifest.read_text(encoding='utf-8'))
	if not isinstance(posterior, Mapping):
		raise TypeError('source posterior manifest must be an object')
	validate_multi_head_state_posterior_manifest(posterior)
	if posterior['source_hard_manifest'] != _reference(config.source_hard_manifest):
		raise ValueError('posterior is not anchored to selected hard manifest')
	if posterior['source_embedding'] != hard['source_embedding']:
		raise ValueError('posterior and hard embedding identity differ')
	_, hard_model_identities = hard_source_model_identities(hard)
	recorded = Path(
		_string(
			_mapping(hard['source_embedding'], 'source_embedding').get('input_dir'),
			'source_embedding.input_dir',
		)
	)
	if recorded.resolve() != config.source_embedding_dir.resolve():
		raise ValueError('source embedding directory differs from hard manifest')
	inputs = {
		item.survey_id: item
		for item in discover_embedding_inputs(config.source_embedding_dir)
	}
	_validate_source_embedding_identity(
		_mapping(hard['source_embedding'], 'source_embedding'), inputs
	)
	models = {
		k: load_frozen_hmm_model(
			clustering_output_dir=config.clustering_output_dir,
			clustering_config=config.clustering_config,
			k=k,
		)
		for k in CANONICAL_KS
	}
	for k in CANONICAL_KS:
		if models[k]['identity'] != hard_model_identities[str(k)]:
			raise ValueError(
				f'frozen model identity differs from hard manifest for k={k}'
			)
		if (
			_mapping(posterior['heads'], 'posterior heads')[str(k)]['model']
			!= models[k]['identity']
		):
			raise ValueError(f'frozen model identity differs for k={k}')
		for survey_id, raw in _mapping(
			_mapping(hard['heads'], 'hard heads')[str(k)], 'hard head'
		)['surveys'].items():
			if survey_id not in inputs:
				raise ValueError(f'missing embedding survey {survey_id}')
			labels = np.load(
				_source_label_path(_mapping(raw, 'hard survey')),
				mmap_mode='r',
				allow_pickle=False,
			)
			valid = np.load(
				_hashed(_mapping(raw, 'hard survey')['valid_tokens'], 'hard valid'),
				mmap_mode='r',
				allow_pickle=False,
			)
			if labels.shape != valid.shape or not np.array_equal(labels >= 0, valid):
				raise ValueError(
					f'hard labels and valid mask differ for k={k} {survey_id}'
				)
	return hard, posterior, inputs, models


def _validate_frozen_source_replay(
	source: Mapping[str, object],
	inputs: Mapping[str, EmbeddingInput],
	models: Mapping[int, Mapping[str, object]],
) -> None:
	"""Require every frozen hard-label trace to match its Viterbi replay."""
	for k in CANONICAL_KS:
		for survey_id, raw in _mapping(
			_mapping(source['heads'], 'heads')[str(k)], 'hard head'
		)['surveys'].items():
			labels = np.load(
				_source_label_path(_mapping(raw, 'hard survey')),
				mmap_mode='r',
				allow_pickle=False,
			)
			valid = np.load(
				_hashed(_mapping(raw, 'hard survey')['valid_tokens'], 'hard valid'),
				mmap_mode='r',
				allow_pickle=False,
			)
			for x in range(labels.shape[0]):
				for y in range(labels.shape[1]):
					z = np.flatnonzero(valid[x, y])
					if not z.size:
						continue
					flat = ((x * labels.shape[1] + y) * labels.shape[2] + z).astype(
						np.int64
					)
					_, replay = replay_frozen_hmm_trace(
						inputs[str(survey_id)], flat, models[k], k=k
					)
					if not np.array_equal(replay, labels[x, y, z]):
						raise ValueError(
							'Viterbi replay differs from frozen hard labels: '
							f'k={k} {survey_id}'
						)


def _affinity_scale(
	source: Mapping[str, object],
	posterior: Mapping[str, object],
	inputs: Mapping[str, EmbeddingInput],
) -> tuple[float, dict[str, object]]:
	valid_paths = _common_valid_token_paths(source, posterior)
	return _disk_backed_scale(
		sum(values.size for values in _affinity_values(inputs, valid_paths)),
		lambda output: _fill_values(output, _affinity_values(inputs, valid_paths)),
		name='affinity scale',
	)


def _common_valid_token_paths(
	source: Mapping[str, object], posterior: Mapping[str, object]
) -> dict[str, Path]:
	"""Return the verified hard/posterior common masks for fixed-scale replay."""
	hard_head = _mapping(
		_mapping(source['heads'], 'hard heads')[str(CANONICAL_KS[0])], 'hard head'
	)
	hard_surveys = _mapping(
		hard_head['surveys'],
		'hard surveys',
	)
	posterior_surveys = _mapping(
		_mapping(
			_mapping(posterior['heads'], 'posterior heads')[str(CANONICAL_KS[0])],
			'posterior head',
		)['surveys'],
		'posterior surveys',
	)
	if set(hard_surveys) != set(posterior_surveys):
		raise ValueError('hard and posterior survey sets differ')
	paths: dict[str, Path] = {}
	for survey_id in sorted(hard_surveys):
		hard_valid = _hashed(
			_mapping(hard_surveys[survey_id], 'hard survey')['valid_tokens'],
			'hard valid',
		)
		posterior_valid = _hashed(
			_mapping(posterior_surveys[survey_id], 'posterior survey')['valid_tokens'],
			'posterior valid',
		)
		hard_array = np.load(hard_valid, mmap_mode='r', allow_pickle=False)
		posterior_array = np.load(posterior_valid, mmap_mode='r', allow_pickle=False)
		if not np.array_equal(hard_array, posterior_array):
			raise ValueError(f'hard and posterior valid masks differ for {survey_id}')
		paths[survey_id] = hard_valid
	return paths


def _affinity_values(
	inputs: Mapping[str, EmbeddingInput], valid_paths: Mapping[str, Path]
) -> Iterator[np.ndarray]:
	"""Yield one bounded XY-edge distance vector at a time, never a grid."""
	if set(inputs) != set(valid_paths):
		raise ValueError('embedding and common valid-mask survey sets differ')
	for survey_id in sorted(inputs):
		embedding = inputs[survey_id]
		features = np.load(embedding.embeddings_path, mmap_mode='r', allow_pickle=False)
		valid = np.load(valid_paths[survey_id], mmap_mode='r', allow_pickle=False)
		if features.ndim != 4 or valid.shape != features.shape[:3]:
			raise ValueError('embedding shape does not match valid mask')
		for x in range(valid.shape[0]):
			for y in range(valid.shape[1]):
				for xx, yy in ((x + 1, y), (x, y + 1)):
					if xx >= valid.shape[0] or yy >= valid.shape[1]:
						continue
					z = np.flatnonzero(valid[x, y] & valid[xx, yy])
					if not z.size:
						continue
					left = np.asarray(features[x, y, z], dtype=np.float64)
					right = np.asarray(features[xx, yy, z], dtype=np.float64)
					norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
					if np.any(norm == 0.0) or not np.all(np.isfinite(norm)):
						raise ValueError(
							'valid embedding endpoint has zero or non-finite norm'
						)
					yield np.clip(1.0 - np.sum(left * right, axis=1) / norm, 0.0, 2.0)


def _emission_gap_scales(
	source: Mapping[str, object],
	inputs: Mapping[str, EmbeddingInput],
	models: Mapping[int, Mapping[str, object]],
) -> tuple[dict[int, float], dict[int, dict[str, object]]]:
	scales: dict[int, float] = {}
	stats: dict[int, dict[str, object]] = {}
	for k in CANONICAL_KS:
		scales[k], stats[k] = _disk_backed_scale(
			sum(
				item.size for item in _emission_gap_values(source, inputs, models[k], k)
			),
			lambda output, k=k: _fill_values(
				output, _emission_gap_values(source, inputs, models[k], k)
			),
			name=f'emission gap scale k={k}',
		)
	return scales, stats


def _emission_gap_values(
	source: Mapping[str, object],
	inputs: Mapping[str, EmbeddingInput],
	model: Mapping[str, object],
	k: int,
) -> Iterator[np.ndarray]:
	for survey_id, raw in _mapping(_mapping(source['heads'], 'heads')[str(k)], 'head')[
		'surveys'
	].items():
		labels = np.load(
			_source_label_path(_mapping(raw, 'survey')),
			mmap_mode='r',
			allow_pickle=False,
		)
		for x in range(labels.shape[0]):
			for y in range(labels.shape[1]):
				z = np.flatnonzero(labels[x, y] >= 0)
				if not z.size:
					continue
				flat = ((x * labels.shape[1] + y) * labels.shape[2] + z).astype(
					np.int64
				)
				costs, _ = replay_frozen_hmm_trace(inputs[survey_id], flat, model, k=k)
				ordered = np.partition(costs, 1, axis=1)
				gap = ordered[:, 1] - ordered[:, 0]
				if not np.all(np.isfinite(gap)) or np.any(gap < 0):
					raise ValueError('emission gap is invalid')
				yield gap


def _fill_values(output: np.memmap, values: Iterator[np.ndarray]) -> int:
	offset = 0
	for values_array in values:
		output[offset : offset + values_array.size] = values_array
		offset += values_array.size
	return offset


def _disk_backed_scale(
	count: int, fill: Callable[[np.memmap], int], *, name: str
) -> tuple[float, dict[str, object]]:
	if count <= 0:
		raise ValueError(f'{name} requires at least one value')
	with tempfile.TemporaryDirectory(prefix='.lateral.scale.') as scratch:
		path = Path(scratch) / 'values.dat'
		array = np.memmap(path, mode='w+', dtype=np.float64, shape=(count,))
		try:
			if fill(array) != count or not np.all(np.isfinite(array)):
				raise ValueError(f'{name} values are invalid')
			stats = _scale_statistics(array)
		finally:
			array.flush()
			del array
	return stats


def _scale_statistics(array: np.memmap) -> tuple[float, dict[str, object]]:
	def quantile(fraction: float) -> float:
		position = fraction * (array.size - 1)
		lower, upper = math.floor(position), math.ceil(position)
		array.partition((lower, upper))
		return float(array[lower] + (array[upper] - array[lower]) * (position - lower))

	median = quantile(0.5)
	scale = max(median, 1.0e-6)
	return scale, {
		'sample_count': int(array.size),
		'minimum': float(array.min()),
		'quantiles': {
			'p05': quantile(0.05),
			'p25': quantile(0.25),
			'p50': median,
			'p75': quantile(0.75),
			'p95': quantile(0.95),
		},
		'maximum': float(array.max()),
		'floor_applied': median < 1.0e-6,
		'resolved_scale': scale,
	}


def _affinity_quartile_boundaries(
	affinity_stats: Mapping[str, object], affinity_scale: float
) -> tuple[float, float, float]:
	"""Map measured distance quartiles to the monotonic affinity quartiles."""
	quantiles = _mapping(affinity_stats['quantiles'], 'affinity distance quantiles')
	try:
		return tuple(
			float(math.exp(-float(quantiles[name]) / affinity_scale))
			for name in ('p75', 'p50', 'p25')
		)
	except (KeyError, TypeError, ValueError) as exc:
		raise ValueError('affinity quartile statistics are invalid') from exc


class _QuantileHistogram:
	"""Fixed-memory quantiles for finite diagnostics with known bounds."""

	_BIN_COUNT = 4096

	def __init__(self, lower: float, upper: float) -> None:
		self.lower = lower
		self.upper = upper
		self.counts = np.zeros(self._BIN_COUNT, dtype=np.int64)
		self.count = 0
		self.minimum = np.inf
		self.maximum = -np.inf

	def add(self, values: np.ndarray) -> None:
		array = np.asarray(values, dtype=np.float64).reshape(-1)
		if not array.size:
			return
		if (
			not np.all(np.isfinite(array))
			or np.any(array < self.lower)
			or np.any(array > self.upper)
		):
			raise ValueError('diagnostic values are outside their finite range')
		if self.lower == self.upper:
			index = np.zeros(array.size, dtype=np.intp)
		else:
			index = np.minimum(
				(
					(array - self.lower) / (self.upper - self.lower) * self._BIN_COUNT
				).astype(np.intp),
				self._BIN_COUNT - 1,
			)
		self.counts += np.bincount(index, minlength=self._BIN_COUNT)
		self.count += array.size
		self.minimum = min(self.minimum, float(array.min()))
		self.maximum = max(self.maximum, float(array.max()))

	def quantiles(self) -> dict[str, float]:
		if not self.count:
			return dict.fromkeys(('p00', 'p05', 'p50', 'p95', 'p100'), 0.0)
		return {
			name: self._quantile(fraction)
			for name, fraction in (
				('p00', 0.0),
				('p05', 0.05),
				('p50', 0.5),
				('p95', 0.95),
				('p100', 1.0),
			)
		}

	def _quantile(self, fraction: float) -> float:
		if fraction == 0.0:
			return float(self.minimum)
		if fraction == 1.0:
			return float(self.maximum)
		target = fraction * (self.count - 1)
		index = int(np.searchsorted(np.cumsum(self.counts), target, side='right'))
		return self.lower + (index + 0.5) * (self.upper - self.lower) / self._BIN_COUNT


class _LateralDiagnostics:
	"""Bounded per-head or per-survey diagnostics for fixed M5-LS exports."""

	def __init__(
		self,
		k: int,
		maximum_cost_update: float,
		affinity_quartile_boundaries: tuple[float, float, float],
	) -> None:
		self.k = k
		self.valid_tokens = self.invalid_tokens = self.changed_tokens = 0
		self.boundary_changed = self.boundary_tokens = 0
		self.interior_changed = self.interior_tokens = 0
		self.occupancy = np.zeros(k, dtype=np.int64)
		self.ordered_violations = self.transitions = 0
		self.max_reverse_decrease = 0
		self.initial = np.zeros(k, dtype=np.int64)
		self.terminal = np.zeros(k, dtype=np.int64)
		self.no_neighbor_tokens = 0
		self.unique_states = _QuantileHistogram(0.0, float(k))
		self.neighbor_count = _QuantileHistogram(0.0, 4.0)
		self.distance = _QuantileHistogram(0.0, 2.0)
		self.affinity = _QuantileHistogram(0.0, 1.0)
		self.entropy = _QuantileHistogram(0.0, float(np.log(k)))
		self.cost_update = _QuantileHistogram(0.0, maximum_cost_update)
		self.edge_count = 0
		self.source_disagreement = self.lateral_disagreement = 0.0
		self.weight_sum = self.weighted_source = self.weighted_lateral = 0.0
		self.affinity_quartile_boundaries = affinity_quartile_boundaries
		self.quartiles = [
			{'edge_count': 0, 'source_sum': 0.0, 'lateral_sum': 0.0} for _ in range(4)
		]

	def add_trace(  # noqa: PLR0913
		self,
		source: np.ndarray,
		lateral: np.ndarray,
		distance: np.ndarray,
		affinity: np.ndarray,
		neighbor_count: np.ndarray,
		entropy: np.ndarray,
		cost_update: np.ndarray,
	) -> None:
		changed = lateral != source
		self.changed_tokens += int(np.count_nonzero(changed))
		self.occupancy += np.bincount(lateral, minlength=self.k)
		self.transitions += int(np.count_nonzero(np.diff(lateral) != 0))
		decrease = np.diff(lateral)
		self.ordered_violations += int(np.count_nonzero(decrease < 0))
		self.max_reverse_decrease = max(
			self.max_reverse_decrease, int(max(0, -decrease.min(initial=0)))
		)
		self.initial[lateral[0]] += 1
		self.terminal[lateral[-1]] += 1
		self.unique_states.add(np.array([np.unique(lateral).size], dtype=np.float64))
		boundary = np.zeros(source.size, dtype=bool)
		locations = np.flatnonzero(np.diff(source) != 0)
		boundary[locations] = True
		boundary[locations + 1] = True
		self.boundary_tokens += int(boundary.sum())
		self.boundary_changed += int(np.count_nonzero(changed & boundary))
		self.interior_tokens += int((~boundary).sum())
		self.interior_changed += int(np.count_nonzero(changed & ~boundary))
		available = affinity > 0.0
		self.distance.add(distance[available])
		self.affinity.add(affinity[available])
		self.entropy.add(entropy)
		self.neighbor_count.add(neighbor_count)
		self.no_neighbor_tokens += int(np.count_nonzero(neighbor_count == 0))
		self.cost_update.add(cost_update)

	def add_grid_counts(self, valid: np.ndarray) -> None:
		self.valid_tokens += int(valid.sum())
		self.invalid_tokens += int((~valid).sum())

	def add_xy_edges(
		self,
		source: np.ndarray,
		lateral: np.ndarray,
		valid: np.ndarray,
		features: np.ndarray,
		affinity_scale: float,
	) -> None:
		for x in range(valid.shape[0]):
			for y in range(valid.shape[1]):
				for xx, yy in ((x + 1, y), (x, y + 1)):
					if xx >= valid.shape[0] or yy >= valid.shape[1]:
						continue
					z = np.flatnonzero(valid[x, y] & valid[xx, yy])
					if not z.size:
						continue
					left = np.asarray(features[x, y, z], dtype=np.float64)
					right = np.asarray(features[xx, yy, z], dtype=np.float64)
					norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
					if np.any(norm == 0.0) or not np.all(np.isfinite(norm)):
						raise ValueError(
							'valid embedding endpoint has zero or non-finite norm'
						)
					distance = np.clip(
						1.0 - np.sum(left * right, axis=1) / norm, 0.0, 2.0
					)
					weights = np.exp(-distance / affinity_scale)
					before = np.abs(
						source[x, y, z] / (self.k - 1)
						- source[xx, yy, z] / (self.k - 1)
					)
					after = np.abs(
						lateral[x, y, z] / (self.k - 1)
						- lateral[xx, yy, z] / (self.k - 1)
					)
					self.edge_count += z.size
					self.source_disagreement += float(before.sum())
					self.lateral_disagreement += float(after.sum())
					self.weight_sum += float(weights.sum())
					self.weighted_source += float(np.dot(weights, before))
					self.weighted_lateral += float(np.dot(weights, after))
					bucket_indices = np.searchsorted(
						self.affinity_quartile_boundaries, weights, side='right'
					)
					for index in range(4):
						select = bucket_indices == index
						bucket = self.quartiles[index]
						bucket['edge_count'] += int(select.sum())
						bucket['source_sum'] += float(before[select].sum())
						bucket['lateral_sum'] += float(after[select].sum())

	def finish(self) -> dict[str, object]:
		occupancy = self.occupancy / max(self.valid_tokens, 1)
		positive = occupancy[occupancy > 0]
		effective_k = float(
			np.exp(-np.sum(positive * np.log(np.maximum(positive, 1e-30))))
		)
		return {
			'valid_token_count': self.valid_tokens,
			'invalid_token_count': self.invalid_tokens,
			'changed_token_count': self.changed_tokens,
			'changed_fraction': self.changed_tokens / max(self.valid_tokens, 1),
			'changed_fraction_by_source_region': {
				'boundary_adjacent': self.boundary_changed
				/ max(self.boundary_tokens, 1),
				'interior': self.interior_changed / max(self.interior_tokens, 1),
			},
			'state_occupancy': {
				'counts': self.occupancy.tolist(),
				'ratios': occupancy.tolist(),
				'effective_k': effective_k,
				'empty_state_count': int(np.count_nonzero(self.occupancy == 0)),
			},
			'ordered_path': {
				'violation_count': self.ordered_violations,
				'max_reverse_decrease': self.max_reverse_decrease,
			},
			'trace_paths': {
				'transition_count': self.transitions,
				'unique_state_count_quantiles': self.unique_states.quantiles(),
				'initial_state_counts': self.initial.tolist(),
				'terminal_state_counts': self.terminal.tolist(),
			},
			'lateral_neighbors': {
				'count_quantiles': self.neighbor_count.quantiles(),
				'no_neighbor_token_count': self.no_neighbor_tokens,
			},
			'lateral_signal': {
				'cosine_distance_quantiles': self.distance.quantiles(),
				'affinity_quantiles': self.affinity.quantiles(),
				'message_entropy_quantiles': self.entropy.quantiles(),
				'cost_update_magnitude_quantiles': self.cost_update.quantiles(),
			},
			'xy_edge_disagreement': {
				'edge_count': self.edge_count,
				'source_unweighted_mean': self.source_disagreement
				/ max(self.edge_count, 1),
				'lateral_unweighted_mean': self.lateral_disagreement
				/ max(self.edge_count, 1),
				'affinity_weighted_normalized_order': {
					'source': self.weighted_source / max(self.weight_sum, 1.0e-30),
					'lateral': self.weighted_lateral / max(self.weight_sum, 1.0e-30),
				},
				'affinity_quartiles': [
					{
						'affinity_range': [
							0.0
							if index == 0
							else self.affinity_quartile_boundaries[index - 1],
							1.0
							if index == 3
							else self.affinity_quartile_boundaries[index],
						],
						'edge_count': item['edge_count'],
						'source_unweighted_mean': item['source_sum']
						/ max(int(item['edge_count']), 1),
						'lateral_unweighted_mean': item['lateral_sum']
						/ max(int(item['edge_count']), 1),
					}
					for index, item in enumerate(self.quartiles)
				],
			},
		}


def _export_head(  # noqa: PLR0913
	root: Path,
	k: int,
	source: Mapping[str, object],
	posterior: Mapping[str, object],
	inputs: Mapping[str, EmbeddingInput],
	model: Mapping[str, object],
	config: MultiHeadLateralTargetExportConfig,
	affinity_scale: float,
	affinity_stats: Mapping[str, object],
	gap_scale: float,
	gap_stats: Mapping[str, object],
) -> None:
	root.mkdir(parents=True)
	surveys: dict[str, object] = {}
	diagnostics: dict[str, object] = {}
	affinity_quartiles = _affinity_quartile_boundaries(affinity_stats, affinity_scale)
	aggregate = _LateralDiagnostics(
		k, config.pairwise_strength_ratio * gap_scale, affinity_quartiles
	)
	for survey_id, raw in _mapping(_mapping(source['heads'], 'heads')[str(k)], 'head')[
		'surveys'
	].items():
		stats = _LateralDiagnostics(
			k, config.pairwise_strength_ratio * gap_scale, affinity_quartiles
		)
		entry, metrics = _export_survey(
			root,
			survey_id,
			_mapping(raw, 'survey'),
			_mapping(_mapping(posterior['heads'], 'heads')[str(k)], 'posterior head')[
				'surveys'
			][survey_id],
			inputs[survey_id],
			_source_embedding_survey(source['source_embedding'], survey_id),
			model,
			k,
			config,
			affinity_scale,
			gap_scale,
			stats,
			aggregate,
		)
		surveys[survey_id] = entry
		diagnostics[survey_id] = metrics
	head = {
		'model': model['identity'],
		'surveys': surveys,
		'diagnostics': {
			'per_survey': diagnostics,
			'aggregate': {**aggregate.finish(), 'survey_count': len(diagnostics)},
			'resolved_scales': {'affinity': affinity_stats, 'emission_gap': gap_stats},
		},
	}
	diagnostics_payload = head['diagnostics']
	if not isinstance(diagnostics_payload, Mapping):
		raise TypeError('lateral diagnostics must be a mapping')
	_write_json(root / 'diagnostics.json', diagnostics_payload)
	_write_diagnostics_csv(root / 'diagnostics.csv', diagnostics_payload)
	head['diagnostics'] = {
		**diagnostics_payload,
		'json': _reference(root / 'diagnostics.json'),
		'csv': _reference(root / 'diagnostics.csv'),
	}
	_write_json(root / 'head_metadata.json', head)


def _export_survey(  # noqa: PLR0913
	root: Path,
	survey_id: str,
	source: Mapping[str, object],
	posterior: Mapping[str, object],
	embedding: EmbeddingInput,
	source_embedding: Mapping[str, object],
	model: Mapping[str, object],
	k: int,
	config: MultiHeadLateralTargetExportConfig,
	affinity_scale: float,
	gap_scale: float,
	stats: _LateralDiagnostics,
	aggregate: _LateralDiagnostics,
) -> tuple[dict[str, object], dict[str, object]]:
	source_labels = np.load(
		_source_label_path(source), mmap_mode='r', allow_pickle=False
	)
	valid = np.load(
		_hashed(source['valid_tokens'], 'hard valid'), mmap_mode='r', allow_pickle=False
	)
	posterior_array = np.load(
		_hashed(_mapping(posterior, 'posterior survey')['posterior'], 'posterior'),
		mmap_mode='r',
		allow_pickle=False,
	)
	posterior_valid = np.load(
		_hashed(
			_mapping(posterior, 'posterior survey')['valid_tokens'], 'posterior valid'
		),
		mmap_mode='r',
		allow_pickle=False,
	)
	features = np.load(embedding.embeddings_path, mmap_mode='r', allow_pickle=False)
	if (
		not np.array_equal(valid, posterior_valid)
		or source_labels.shape != features.shape[:3]
		or posterior_array.shape != (*source_labels.shape, k)
	):
		raise ValueError('source survey grids do not agree')
	paths = {
		name: root / f'{survey_id}{suffix}'
		for name, suffix in [
			('labels', _LABEL_SUFFIX),
			('confidence', _CONFIDENCE_SUFFIX),
			('valid', _VALID_SUFFIX),
		]
	}
	labels = np.lib.format.open_memmap(
		paths['labels'], mode='w+', dtype=np.int32, shape=source_labels.shape
	)
	labels[...] = -1
	confidence = np.lib.format.open_memmap(
		paths['confidence'], mode='w+', dtype=np.float32, shape=source_labels.shape
	)
	confidence[...] = 0
	try:
		for x in range(source_labels.shape[0]):
			for y in range(source_labels.shape[1]):
				z = np.flatnonzero(valid[x, y])
				if not z.size:
					continue
				replay, result = _smooth_trace(
					survey_id=survey_id,
					source_labels=source_labels,
					valid=valid,
					posterior=posterior_array,
					features=features,
					embedding=embedding,
					model=model,
					k=k,
					config=config,
					affinity_scale=affinity_scale,
					gap_scale=gap_scale,
					x=x,
					y=y,
					z=z,
				)
				labels[x, y, z] = result.labels
				confidence[x, y, z] = 1.0
				for item in (stats, aggregate):
					item.add_trace(
						replay,
						result.labels,
						result.diagnostics.distance,
						result.diagnostics.affinity,
						result.diagnostics.neighbor_count,
						result.diagnostics.message_entropy,
						result.changed_cost_magnitude,
					)
	finally:
		labels.flush()
		confidence.flush()
		del labels, confidence
	for item in (stats, aggregate):
		item.add_grid_counts(valid)
		item.add_xy_edges(
			source_labels,
			np.load(paths['labels'], mmap_mode='r'),
			valid,
			features,
			affinity_scale,
		)
	np.save(paths['valid'], valid, allow_pickle=False)
	metadata = build_pseudo_target_metadata(
		labels=np.load(paths['labels'], mmap_mode='r', allow_pickle=False),
		valid_tokens=valid,
		boundary_weight=np.asarray(valid, dtype=np.float32),
		boundary_weight_source='default_unity',
		k=k,
		survey_id=survey_id,
		schema_version=1,
		write_boundary_weight=False,
		source_metadata={
			'target_semantics': LATERAL_SMOOTHING_SEMANTICS,
			'source_label_path': str(_source_label_path(source)),
			'source_label_sha256': file_sha256(_source_label_path(source)),
			'source_hard_labels': _reference(_source_label_path(source)),
			'source_posterior': _mapping(posterior, 'posterior survey')['posterior'],
			'source_embedding': dict(source_embedding),
			'smoothing': {
				**_smoothing_identity(config.pairwise_strength_ratio),
				'affinity_scale': affinity_scale,
				'emission_gap_scale': gap_scale,
			},
		},
	)
	meta = root / f'{survey_id}{_METADATA_SUFFIX}'
	_write_json(meta, metadata)
	return (
		{
			'labels': _array_reference(
				paths['labels'], source_labels.shape, np.dtype(np.int32)
			),
			'confidence': _array_reference(
				paths['confidence'], source_labels.shape, np.dtype(np.float32)
			),
			'valid_tokens': _array_reference(
				paths['valid'], source_labels.shape, np.dtype(bool)
			),
			'metadata': _reference(meta),
			'source_hard_labels': _reference(_source_label_path(source)),
			'source_posterior': _mapping(posterior, 'posterior survey')['posterior'],
		},
		stats.finish(),
	)


def _smooth_trace(  # noqa: PLR0913
	*,
	survey_id: str,
	source_labels: np.ndarray,
	valid: np.ndarray,
	posterior: np.ndarray,
	features: np.ndarray,
	embedding: EmbeddingInput,
	model: Mapping[str, object],
	k: int,
	config: MultiHeadLateralTargetExportConfig,
	affinity_scale: float,
	gap_scale: float,
	x: int,
	y: int,
	z: np.ndarray,
) -> tuple[np.ndarray, LateralSmoothingResult]:
	"""Replay and smooth one valid trace using the fixed M5-LS definition."""
	flat = ((x * source_labels.shape[1] + y) * source_labels.shape[2] + z).astype(
		np.int64
	)
	costs, replay = replay_frozen_hmm_trace(embedding, flat, model, k=k)
	expected, weight = expected_boundaries(model['hmm'], k=k, length=z.size)
	if not np.array_equal(replay, source_labels[x, y, z]):
		raise ValueError(f'Viterbi replay differs from frozen hard labels: {survey_id}')
	neighbor_features: list[np.ndarray] = []
	neighbor_posterior: list[np.ndarray] = []
	neighbor_valid: list[np.ndarray] = []
	for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
		xx, yy = x + dx, y + dy
		if 0 <= xx < source_labels.shape[0] and 0 <= yy < source_labels.shape[1]:
			neighbor_features.append(np.asarray(features[xx, yy, z]))
			neighbor_posterior.append(np.asarray(posterior[xx, yy, z]))
			neighbor_valid.append(np.asarray(valid[xx, yy, z]))
	if neighbor_features:
		neighbor_embedding_array = np.asarray(neighbor_features)
		neighbor_posterior_array = np.asarray(neighbor_posterior)
		neighbor_valid_array = np.asarray(neighbor_valid, dtype=bool)
	else:
		neighbor_embedding_array = np.empty(
			(0, z.size, features.shape[-1]), dtype=features.dtype
		)
		neighbor_posterior_array = np.empty((0, z.size, k), dtype=np.float32)
		neighbor_valid_array = np.empty((0, z.size), dtype=bool)
	return replay, smooth_and_redecode_ordered_trace(
		np.asarray(features[x, y, z]),
		neighbor_embedding_array,
		neighbor_posterior_array,
		np.ones(z.size, dtype=bool),
		neighbor_valid_array,
		costs,
		model['transition_costs'],
		affinity_scale=affinity_scale,
		emission_gap_scale=gap_scale,
		pairwise_strength_ratio=config.pairwise_strength_ratio,
		initial_state_costs=model['initial_costs'],
		terminal_state_costs=model['terminal_costs'],
		expected_boundary_count=expected,
		boundary_count_weight=weight,
	)


def _validate_complete_head(  # noqa: C901, PLR0913
	path: Path,
	k: int,
	source: Mapping[str, object],
	posterior: Mapping[str, object],
	models: Mapping[int, Mapping[str, object]],
	config: MultiHeadLateralTargetExportConfig,
	*,
	allow_staging: bool = False,
	inputs: Mapping[str, EmbeddingInput] | None = None,
	expected_scales: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
	if not path.is_dir() or (
		not allow_staging and any(part.startswith('.') for part in path.parts)
	):
		raise ValueError('partial lateral output is not complete')
	head_path = path / 'head_metadata.json'
	if not head_path.is_file():
		raise ValueError('lateral head metadata is missing')
	head = json.loads(head_path.read_text(encoding='utf-8'))
	if not isinstance(head, Mapping):
		raise TypeError('lateral head metadata must be an object')
	# Validate the head in its complete public schema without pretending it is a
	# full bundle.
	_required(head, {'model', 'surveys', 'diagnostics'})
	if head.get('model') != models[k]['identity']:
		raise _ImmutableIdentityMismatchError('frozen model identity drift')
	surveys = _mapping(head.get('surveys'), 'surveys')
	hard_surveys = _mapping(
		_mapping(_mapping(source['heads'], 'heads')[str(k)], 'hard head')['surveys'],
		'hard surveys',
	)
	if set(surveys) != set(hard_surveys):
		raise _ImmutableIdentityMismatchError(
			'lateral survey set differs from source'
		)
	_validate_lateral_diagnostics(head.get('diagnostics'), survey_ids=set(surveys))
	resolved_scales = _resolved_scales(head['diagnostics'])
	if expected_scales is not None:
		actual = _mapping(head['diagnostics'], 'diagnostics')['resolved_scales']
		if actual != expected_scales:
			raise _ImmutableIdentityMismatchError('resolved scales differ from source')
	posterior_surveys = _mapping(
		_mapping(_mapping(posterior['heads'], 'posterior heads')[str(k)], 'head')[
			'surveys'
		],
		'posterior surveys',
	)
	for survey_id, raw in surveys.items():
		entry = _mapping(raw, 'survey')
		_required(
			entry,
			{
				'labels',
				'confidence',
				'valid_tokens',
				'metadata',
				'source_hard_labels',
				'source_posterior',
			},
		)
		for key in ('labels', 'confidence', 'valid_tokens', 'metadata'):
			_hashed(entry.get(key), key)
		labels = np.load(
			_hashed(entry['labels'], 'labels'), mmap_mode='r', allow_pickle=False
		)
		confidence = np.load(
			_hashed(entry['confidence'], 'confidence'),
			mmap_mode='r',
			allow_pickle=False,
		)
		valid = np.load(
			_hashed(entry['valid_tokens'], 'valid_tokens'),
			mmap_mode='r',
			allow_pickle=False,
		)
		_validate_array_reference(entry['labels'], labels, name='labels')
		_validate_array_reference(entry['confidence'], confidence, name='confidence')
		_validate_array_reference(entry['valid_tokens'], valid, name='valid_tokens')
		_validate_complete_head_arrays(labels, confidence, valid, k=k)
		hard_survey = _mapping(hard_surveys[survey_id], 'hard survey')
		posterior_survey = _mapping(posterior_surveys[survey_id], 'posterior survey')
		_validate_target_valid_tokens(
			valid,
			hard_survey,
			posterior_survey,
			context=f'k={k} {survey_id}',
		)
		if (
			entry.get('source_hard_labels')
			!= _reference(_source_label_path(hard_survey))
			or entry.get('source_posterior') != posterior_survey['posterior']
		):
			raise _ImmutableIdentityMismatchError('lateral source provenance differs')
		_validate_survey_metadata(
			entry,
			k=k,
			survey_id=str(survey_id),
			identity={
				'source_embedding': _source_embedding_survey(
					_mapping(source['source_embedding'], 'source_embedding'), survey_id
				),
				'smoothing': _smoothing_identity(config.pairwise_strength_ratio),
				'resolved_scales': resolved_scales,
			},
		)
	_validate_diagnostics_from_arrays(head['diagnostics'], surveys, hard_surveys, k=k)
	if inputs is not None:
		if expected_scales is None:
			raise ValueError('semantic diagnostics validation requires resolved scales')
		_validate_diagnostics_from_frozen_sources(
			head['diagnostics'],
			surveys,
			hard_surveys,
			posterior_surveys,
			inputs,
			models[k],
			k=k,
			config=config,
			expected_scales=expected_scales,
		)


def _validate_complete_head_arrays(
	labels: np.ndarray,
	confidence: np.ndarray,
	valid: np.ndarray,
	*,
	k: int,
) -> None:
	"""Reject a reusable head whose hard-label invariants are invalid."""
	if (
		labels.dtype != np.int32
		or labels.ndim != 3
		or confidence.dtype != np.float32
		or valid.dtype != np.bool_
		or labels.shape != confidence.shape
		or labels.shape != valid.shape
		or np.any(labels[valid] < 0)
		or np.any(labels[valid] >= k)
		or np.any(labels[~valid] != -1)
		or np.any(confidence[valid] != 1.0)
		or np.any(confidence[~valid] != 0.0)
	):
		raise ValueError('lateral arrays have invalid hard-target semantics')
	if np.any(np.bincount(labels[valid], minlength=k) == 0):
		raise ValueError('lateral labels contain an empty state')
	for x in range(labels.shape[0]):
		for y in range(labels.shape[1]):
			trace = labels[x, y, valid[x, y]]
			if np.any(np.diff(trace) < 0):
				raise ValueError('lateral labels violate ordered paths')


def _validate_target_valid_tokens(
	valid: np.ndarray,
	hard_survey: Mapping[str, object],
	posterior_survey: Mapping[str, object],
	*,
	context: str,
) -> None:
	"""Require the target mask to exactly match its frozen source masks."""
	hard_valid = np.load(
		_hashed(hard_survey['valid_tokens'], 'hard valid'),
		mmap_mode='r',
		allow_pickle=False,
	)
	posterior_valid = np.load(
		_hashed(posterior_survey['valid_tokens'], 'posterior valid'),
		mmap_mode='r',
		allow_pickle=False,
	)
	if (
		hard_valid.dtype != np.bool_
		or posterior_valid.dtype != np.bool_
		or valid.shape != hard_valid.shape
		or valid.shape != posterior_valid.shape
		or not np.array_equal(valid, hard_valid)
		or not np.array_equal(valid, posterior_valid)
	):
		raise ValueError(f'lateral valid mask differs from source masks for {context}')


def _publish_manifest(
	config: MultiHeadLateralTargetExportConfig,
	source: Mapping[str, object],
) -> None:
	heads = {}
	for k in CANONICAL_KS:
		path = _bundle_path(config) / f'k{k}' / 'head_metadata.json'
		if not path.is_file():
			raise ValueError(f'lateral head is missing: k={k}')
		payload = json.loads(path.read_text(encoding='utf-8'))
		if not isinstance(payload, dict):
			raise TypeError('head metadata must be an object')
		heads[str(k)] = payload
	payload = _manifest_payload(config, source, heads)
	validate_multi_head_lateral_target_manifest(payload)
	if config.handoff_manifest.exists():
		try:
			if (
				load_multi_head_lateral_target_manifest(config.handoff_manifest)
				== payload
			):
				return
		except (OSError, TypeError, ValueError):
			pass
		_quarantine(config.handoff_manifest)
	_write_json(config.handoff_manifest, payload)


def _manifest_payload(
	config: MultiHeadLateralTargetExportConfig,
	source: Mapping[str, object],
	heads: Mapping[str, object],
) -> dict[str, object]:
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'target_semantics': LATERAL_SMOOTHING_SEMANTICS,
		'head_ks': list(CANONICAL_KS),
		'source_hard_manifest': _reference(config.source_hard_manifest),
		'source_posterior_manifest': _reference(config.source_posterior_manifest),
		'source_embedding': source['source_embedding'],
		'smoothing': _smoothing_identity(config.pairwise_strength_ratio),
		'heads': dict(heads),
	}


def _bundle_path(config: MultiHeadLateralTargetExportConfig) -> Path:
	"""Return the single atomically-published multi-head directory."""
	return config.output_root / _BUNDLE_DIRNAME


def _validate_owned_handoff(
	config: MultiHeadLateralTargetExportConfig, source: Mapping[str, object]
) -> None:
	"""Require the public completion marker to name this exact bundle."""
	if not config.handoff_manifest.is_file():
		raise _OwnedOutputCorruptionError('lateral handoff is missing')
	handoff = load_multi_head_lateral_target_manifest(config.handoff_manifest)
	heads: dict[str, object] = {}
	for k in CANONICAL_KS:
		path = _bundle_path(config) / f'k{k}' / 'head_metadata.json'
		payload = json.loads(path.read_text(encoding='utf-8'))
		if not isinstance(payload, dict):
			raise _OwnedOutputCorruptionError('lateral head metadata is invalid')
		heads[str(k)] = payload
	expected = _manifest_payload(config, source, heads)
	if handoff == expected:
		return
	if handoff.get('heads') == heads:
		raise _ImmutableIdentityMismatchError(
			'lateral handoff identity differs from current config'
		)
	raise _OwnedOutputCorruptionError('lateral handoff differs from bundle')


def _source_snapshot(
	config: MultiHeadLateralTargetExportConfig,
	source: Mapping[str, object],
	posterior: Mapping[str, object],
	inputs: Mapping[str, EmbeddingInput],
	models: Mapping[int, Mapping[str, object]],
) -> str:
	"""Capture every immutable input identity used by this preflight."""
	payload = {
		'hard_manifest': _reference(config.source_hard_manifest),
		'posterior_manifest': _reference(config.source_posterior_manifest),
		'source_embedding': source['source_embedding'],
		'posterior_embedding': posterior['source_embedding'],
		'inputs': {
			name: {
				'embeddings': _reference(item.embeddings_path),
				'valid_tokens': _reference(item.valid_tokens_path),
				'metadata': _reference(item.metadata_path),
			}
			for name, item in sorted(inputs.items())
		},
		'models': {str(k): model['identity'] for k, model in sorted(models.items())},
	}
	return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _validate_live_snapshot(
	config: MultiHeadLateralTargetExportConfig, preflight: _LateralPreflight
) -> None:
	"""Fail before publication if a frozen input drifted while staging ran."""
	source, posterior, inputs, models = _validate_sources(config)
	_validate_frozen_source_replay(source, inputs, models)
	if (
		_source_snapshot(config, source, posterior, inputs, models)
		!= preflight.snapshot
	):
		raise _ImmutableIdentityMismatchError(
			'live source identity drift during staging'
		)


def _rebase_head_metadata(root: Path, *, old_root: Path, new_root: Path) -> None:
	"""Rebase staged file references before one directory-level publication."""
	path = root / 'head_metadata.json'
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError('head metadata must be an object')

	def rebase(value: object) -> object:
		if isinstance(value, Mapping):
			output = {str(key): rebase(item) for key, item in value.items()}
			candidate = output.get('path')
			if isinstance(candidate, str):
				candidate_path = Path(candidate)
				if candidate_path.is_relative_to(old_root):
					output['path'] = str(
						new_root / candidate_path.relative_to(old_root)
					)
			return output
		if isinstance(value, list):
			return [rebase(item) for item in value]
		return value

	_write_json(path, _mapping(rebase(payload), 'rebased head metadata'))


def _smoothing(value: object) -> None:
	item = _mapping(value, 'smoothing')
	beta = item.get('pairwise_strength_ratio')
	if isinstance(beta, bool) or not isinstance(beta, (int, float)):
		raise TypeError('pairwise_strength_ratio is invalid')
	if not math.isfinite(beta) or beta <= 0:
		raise ValueError('pairwise_strength_ratio is invalid')
	expected = _smoothing_identity(float(beta))
	if item != expected:
		raise ValueError('smoothing identity is invalid')


def _smoothing_identity(pairwise_strength_ratio: float) -> dict[str, object]:
	return {
		'neighborhood': 'xy_4_connected_v1',
		'affinity': 'source_embedding_cosine_rbf_v1',
		'affinity_scale_policy': _AFFINITY_SCALE_POLICY,
		'emission_scale_policy': _EMISSION_SCALE_POLICY,
		'pairwise_strength_ratio': pairwise_strength_ratio,
		'iterations': 1,
		'projection': 'original_ordered_viterbi_v1',
	}


def _resolved_scales(value: object) -> dict[str, float]:
	"""Extract the exact positive scales recorded with a published head."""
	resolved = _mapping(
		_mapping(value, 'diagnostics')['resolved_scales'], 'resolved scales'
	)
	if set(resolved) != {'affinity', 'emission_gap'}:
		raise ValueError('resolved scale identities are invalid')
	output: dict[str, float] = {}
	for name, metadata_name in (
		('affinity', 'affinity_scale'),
		('emission_gap', 'emission_gap_scale'),
	):
		stats = _mapping(resolved[name], f'{name} scale statistics')
		scale = stats.get('resolved_scale')
		if (
			isinstance(scale, bool)
			or not isinstance(scale, (int, float))
			or not math.isfinite(scale)
			or scale <= 0
		):
			raise ValueError(f'{name} resolved scale is invalid')
		output[metadata_name] = float(scale)
	return output


def _source_embedding_survey(
	source_embedding: object, survey_id: object
) -> Mapping[str, object]:
	surveys = _mapping(
		_mapping(source_embedding, 'source_embedding')['surveys'],
		'source embedding surveys',
	)
	return _mapping(surveys.get(str(survey_id)), 'source embedding survey')


def _validate_survey_metadata(
	entry: Mapping[str, object],
	*,
	k: int,
	survey_id: str,
	identity: Mapping[str, object],
) -> None:
	"""Require canonical schema-v1 metadata and exact lateral provenance."""
	path = _hashed(entry['metadata'], 'metadata')
	try:
		metadata = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError('lateral metadata must be valid JSON') from exc
	if not isinstance(metadata, Mapping):
		raise TypeError('lateral metadata must be an object')
	source_hard = _mapping(entry['source_hard_labels'], 'source hard labels')
	provenance = {
		'target_semantics': LATERAL_SMOOTHING_SEMANTICS,
		'source_label_path': _string(
			source_hard.get('path'), 'source_hard_labels.path'
		),
		'source_label_sha256': _string(
			source_hard.get('sha256'), 'source_hard_labels.sha256'
		),
		'source_hard_labels': entry['source_hard_labels'],
		'source_posterior': entry['source_posterior'],
		'source_embedding': dict(
			_mapping(identity.get('source_embedding'), 'source embedding survey')
		),
		'smoothing': {
			**_mapping(identity.get('smoothing'), 'smoothing'),
			**_mapping(identity.get('resolved_scales'), 'resolved scales'),
		},
	}
	labels = np.load(
		_hashed(entry['labels'], 'labels'), mmap_mode='r', allow_pickle=False
	)
	valid = np.load(
		_hashed(entry['valid_tokens'], 'valid_tokens'),
		mmap_mode='r',
		allow_pickle=False,
	)
	expected = build_pseudo_target_metadata(
		labels=labels,
		valid_tokens=valid,
		boundary_weight=np.asarray(valid, dtype=np.float32),
		boundary_weight_source='default_unity',
		k=k,
		survey_id=survey_id,
		schema_version=1,
		write_boundary_weight=False,
		source_metadata=provenance,
	)
	if metadata != expected:
		raise _ImmutableIdentityMismatchError(
			'lateral metadata provenance differs from entry'
		)


def _validate_lateral_diagnostics(value: object, *, survey_ids: set[str]) -> None:
	"""Require complete JSON/CSV diagnostics for every published head."""
	diagnostics = _mapping(value, 'diagnostics')
	if set(diagnostics) != {
		'per_survey',
		'aggregate',
		'resolved_scales',
		'json',
		'csv',
	}:
		raise ValueError('lateral diagnostics keys are invalid')
	per_survey = _mapping(diagnostics['per_survey'], 'per-survey diagnostics')
	if set(per_survey) != survey_ids:
		raise ValueError('lateral diagnostics survey set differs')
	for metrics in [*per_survey.values(), diagnostics['aggregate']]:
		item = _mapping(metrics, 'lateral diagnostic metrics')
		if not {
			'valid_token_count',
			'invalid_token_count',
			'changed_token_count',
			'changed_fraction',
			'changed_fraction_by_source_region',
			'state_occupancy',
			'ordered_path',
			'trace_paths',
			'lateral_neighbors',
			'lateral_signal',
			'xy_edge_disagreement',
		}.issubset(item):
			raise ValueError('lateral diagnostic metrics are incomplete')
	json_path = _hashed(diagnostics['json'], 'diagnostics JSON')
	csv_path = _hashed(diagnostics['csv'], 'diagnostics CSV')
	try:
		payload = json.loads(json_path.read_text(encoding='utf-8'))
		with csv_path.open(encoding='utf-8', newline='') as stream:
			reader = DictReader(stream)
			if reader.fieldnames != ['scope', 'survey_id', 'metric', 'value']:
				raise ValueError('lateral diagnostics CSV header is invalid')
			rows = list(reader)
	except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
		raise ValueError('lateral diagnostics are unreadable') from exc
	expected_payload = {
		'per_survey': diagnostics['per_survey'],
		'aggregate': diagnostics['aggregate'],
		'resolved_scales': diagnostics['resolved_scales'],
	}
	if payload != expected_payload or rows != _diagnostics_csv_rows(expected_payload):
		raise ValueError('lateral diagnostics files differ from head metadata')
	_validate_diagnostic_values(diagnostics)


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _array_reference(
	path: Path, shape: tuple[int, ...], dtype: np.dtype[object]
) -> dict[str, object]:
	return {**_reference(path), 'shape': list(shape), 'dtype': dtype.name}


def _validate_array_reference(
	reference: object, array: np.ndarray, *, name: str
) -> None:
	"""Reject array references that do not describe their hashed file exactly."""
	item = _mapping(reference, name)
	_required(item, {'path', 'sha256', 'shape', 'dtype'})
	shape = item['shape']
	if not isinstance(shape, list) or any(
		isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
		for dimension in shape
	):
		raise TypeError(f'{name}.shape must be a list of non-negative integers')
	if tuple(shape) != array.shape:
		raise ValueError(f'{name} shape differs from manifest')
	if item['dtype'] != array.dtype.name:
		raise ValueError(f'{name} dtype differs from manifest')


def _mapping(value: object, name: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{name} must be a mapping')
	return value


def _string(value: object, name: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{name} must be a non-empty string')
	return value


def _required(value: Mapping[str, object], keys: set[str]) -> None:
	if set(value) != keys:
		raise ValueError(f'manifest keys mismatch; expected {sorted(keys)}')


def _hashed(value: object, name: str) -> Path:
	item = _mapping(value, name)
	path = Path(_string(item.get('path'), f'{name}.path'))
	if not path.is_file() or file_sha256(path) != _string(
		item.get('sha256'), f'{name}.sha256'
	):
		raise ValueError(f'{name} hash mismatch')
	return path


def _source_label_path(entry: Mapping[str, object]) -> Path:
	"""Resolve the source clustering labels bound by a hard-target entry."""
	metadata = _hashed(entry['metadata'], 'hard target metadata')
	payload = json.loads(metadata.read_text(encoding='utf-8'))
	source = _mapping(payload.get('source'), 'hard target source')
	path = Path(_string(source.get('source_label_path'), 'source_label_path'))
	if not path.is_file():
		raise ValueError('hard target source-label hash mismatch')
	digest = source.get('source_label_sha256')
	if digest is None:
		digest = _string(
			_mapping(entry.get('labels'), 'hard target labels').get('sha256'),
			'hard target labels.sha256',
		)
	else:
		digest = _string(digest, 'source_label_sha256')
	if file_sha256(path) != digest:
		raise ValueError('hard target source-label hash mismatch')
	return path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		mode='w',
		encoding='utf-8',
		dir=path.parent,
		prefix=f'.{path.name}.',
		delete=False,
	) as stream:
		json.dump(payload, stream, sort_keys=True)
		stream.write('\n')
		temporary = Path(stream.name)
	temporary.replace(path)


def _write_diagnostics_csv(path: Path, diagnostics: Mapping[str, object]) -> None:
	"""Write a compact, deterministic flattened companion to diagnostics JSON."""
	rows = _diagnostics_csv_rows(diagnostics)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as stream:
		writer = DictWriter(
			stream, fieldnames=['scope', 'survey_id', 'metric', 'value']
		)
		writer.writeheader()
		writer.writerows(rows)


def _diagnostics_csv_rows(diagnostics: Mapping[str, object]) -> list[dict[str, str]]:
	"""Return the exact ordered flattened CSV representation of diagnostics."""
	rows: list[dict[str, str]] = []
	for scope, survey_id, metrics in [
		('aggregate', '', diagnostics['aggregate']),
		*[
			('survey', str(key), value)
			for key, value in sorted(
				_mapping(diagnostics['per_survey'], 'per_survey diagnostics').items()
			)
		],
	]:
		for metric, value in _flatten_metrics(_mapping(metrics, 'diagnostic metrics')):
			rows.append(
				{
					'scope': scope,
					'survey_id': survey_id,
					'metric': metric,
					'value': json.dumps(value, sort_keys=True),
				}
			)
	return rows


def _validate_diagnostic_values(diagnostics: Mapping[str, object]) -> None:
	"""Reject non-finite or internally inconsistent persisted diagnostics."""
	for scope, metrics in [
		*[
			(str(survey_id), _mapping(value, 'survey diagnostics'))
			for survey_id, value in _mapping(
				diagnostics['per_survey'], 'per-survey diagnostics'
			).items()
		],
		('aggregate', _mapping(diagnostics['aggregate'], 'aggregate diagnostics')),
	]:
		valid = metrics.get('valid_token_count')
		invalid = metrics.get('invalid_token_count')
		changed = metrics.get('changed_token_count')
		if (
			any(
				isinstance(value, bool) or not isinstance(value, int) or value < 0
				for value in (valid, invalid, changed)
			)
			or changed > valid
		):
			raise ValueError(f'{scope} diagnostic counts are invalid')
		if metrics.get('changed_fraction') != changed / max(valid, 1):
			raise ValueError(f'{scope} changed fraction is invalid')
		occupancy = _mapping(metrics.get('state_occupancy'), 'state occupancy')
		counts = occupancy.get('counts')
		ratios = occupancy.get('ratios')
		if (
			not isinstance(counts, list)
			or not isinstance(ratios, list)
			or len(counts) != len(ratios)
			or any(
				isinstance(item, bool) or not isinstance(item, int) or item < 0
				for item in counts
			)
			or sum(counts) != valid
			or any(
				not isinstance(item, (int, float)) or not math.isfinite(item)
				for item in ratios
			)
			or not np.allclose(
				ratios,
				np.asarray(counts, dtype=np.float64) / max(valid, 1),
				rtol=0,
				atol=0,
			)
		):
			raise ValueError(f'{scope} state occupancy is invalid')
		probability = np.asarray(ratios, dtype=np.float64)
		expected_k = float(
			np.exp(
				-np.sum(
					probability[probability > 0]
					* np.log(np.maximum(probability[probability > 0], 1e-30))
				)
			)
		)
		if occupancy.get('effective_k') != expected_k or occupancy.get(
			'empty_state_count'
		) != int(np.count_nonzero(np.asarray(counts) == 0)):
			raise ValueError(f'{scope} effective K is invalid')
		_validate_finite_tree(metrics, context=f'{scope} diagnostics')
	for name in ('affinity', 'emission_gap'):
		stats = _mapping(
			_mapping(diagnostics['resolved_scales'], 'resolved scales')[name],
			f'{name} scale',
		)
		if (
			not isinstance(stats.get('sample_count'), int)
			or stats['sample_count'] <= 0
			or not isinstance(stats.get('floor_applied'), bool)
			or not isinstance(stats.get('resolved_scale'), (int, float))
			or not math.isfinite(float(stats['resolved_scale']))
			or float(stats['resolved_scale']) <= 0
		):
			raise ValueError(f'{name} scale diagnostics are invalid')


def _validate_diagnostics_from_arrays(
	diagnostics: object,
	surveys: Mapping[str, object],
	hard_surveys: Mapping[str, object],
	*,
	k: int,
) -> None:
	"""Recompute array-derived diagnostic invariants for publication and reuse."""
	payload = _mapping(diagnostics, 'diagnostics')
	per_survey = _mapping(payload['per_survey'], 'per-survey diagnostics')
	aggregate_counts = {'valid': 0, 'invalid': 0, 'changed': 0}
	aggregate_occupancy = np.zeros(k, dtype=np.int64)
	for survey_id, raw in surveys.items():
		entry = _mapping(raw, 'survey')
		labels = np.load(
			_hashed(entry['labels'], 'labels'), mmap_mode='r', allow_pickle=False
		)
		valid = np.load(
			_hashed(entry['valid_tokens'], 'valid tokens'),
			mmap_mode='r',
			allow_pickle=False,
		)
		source = np.load(
			_source_label_path(_mapping(hard_surveys[survey_id], 'hard survey')),
			mmap_mode='r',
			allow_pickle=False,
		)
		if source.shape != labels.shape:
			raise ValueError('lateral source labels shape differs')
		valid_count = int(np.count_nonzero(valid))
		invalid_count = int(valid.size - valid_count)
		changed_count = int(np.count_nonzero(labels[valid] != source[valid]))
		counts = np.bincount(labels[valid], minlength=k).tolist()
		metrics = _mapping(per_survey[survey_id], 'survey diagnostics')
		if (
			metrics.get('valid_token_count') != valid_count
			or metrics.get('invalid_token_count') != invalid_count
			or metrics.get('changed_token_count') != changed_count
			or _mapping(metrics['state_occupancy'], 'state occupancy').get('counts')
			!= counts
		):
			raise ValueError('persisted lateral diagnostics differ from target arrays')
		aggregate_counts['valid'] += valid_count
		aggregate_counts['invalid'] += invalid_count
		aggregate_counts['changed'] += changed_count
		aggregate_occupancy += np.asarray(counts, dtype=np.int64)
	aggregate = _mapping(payload['aggregate'], 'aggregate diagnostics')
	if (
		aggregate.get('valid_token_count') != aggregate_counts['valid']
		or aggregate.get('invalid_token_count') != aggregate_counts['invalid']
		or aggregate.get('changed_token_count') != aggregate_counts['changed']
		or _mapping(aggregate['state_occupancy'], 'aggregate state occupancy').get(
			'counts'
		)
		!= aggregate_occupancy.tolist()
		or aggregate.get('survey_count') != len(surveys)
	):
		raise ValueError('aggregate lateral diagnostics differ from target arrays')


def _validate_diagnostics_from_frozen_sources(  # noqa: PLR0913
	diagnostics: object,
	surveys: Mapping[str, object],
	hard_surveys: Mapping[str, object],
	posterior_surveys: Mapping[str, object],
	inputs: Mapping[str, EmbeddingInput],
	model: Mapping[str, object],
	*,
	k: int,
	config: MultiHeadLateralTargetExportConfig,
	expected_scales: Mapping[str, Mapping[str, object]],
) -> None:
	"""Rebuild every persisted metric from frozen inputs and fixed smoothing."""
	affinity_stats = _mapping(expected_scales['affinity'], 'expected affinity scale')
	gap_stats = _mapping(expected_scales['emission_gap'], 'expected emission gap scale')
	affinity_scale = float(affinity_stats['resolved_scale'])
	gap_scale = float(gap_stats['resolved_scale'])
	quartiles = _affinity_quartile_boundaries(affinity_stats, affinity_scale)
	aggregate = _LateralDiagnostics(
		k, config.pairwise_strength_ratio * gap_scale, quartiles
	)
	per_survey: dict[str, object] = {}
	for survey_id, raw in surveys.items():
		entry = _mapping(raw, 'survey')
		source = np.load(
			_source_label_path(_mapping(hard_surveys[survey_id], 'hard survey')),
			mmap_mode='r',
			allow_pickle=False,
		)
		valid = np.load(
			_hashed(
				_mapping(hard_surveys[survey_id], 'hard survey')['valid_tokens'],
				'hard valid',
			),
			mmap_mode='r',
			allow_pickle=False,
		)
		posterior_entry = _mapping(posterior_surveys[survey_id], 'posterior survey')
		posterior = np.load(
			_hashed(posterior_entry['posterior'], 'posterior'),
			mmap_mode='r',
			allow_pickle=False,
		)
		posterior_valid = np.load(
			_hashed(posterior_entry['valid_tokens'], 'posterior valid'),
			mmap_mode='r',
			allow_pickle=False,
		)
		features = np.load(
			inputs[str(survey_id)].embeddings_path, mmap_mode='r', allow_pickle=False
		)
		labels = np.load(
			_hashed(entry['labels'], 'labels'), mmap_mode='r', allow_pickle=False
		)
		if (
			not np.array_equal(valid, posterior_valid)
			or source.shape != labels.shape
			or source.shape != features.shape[:3]
			or posterior.shape != (*source.shape, k)
		):
			raise ValueError('frozen sources do not agree for lateral diagnostics')
		stats = _LateralDiagnostics(
			k, config.pairwise_strength_ratio * gap_scale, quartiles
		)
		for x in range(source.shape[0]):
			for y in range(source.shape[1]):
				z = np.flatnonzero(valid[x, y])
				if not z.size:
					continue
				replay, result = _smooth_trace(
					survey_id=str(survey_id),
					source_labels=source,
					valid=valid,
					posterior=posterior,
					features=features,
					embedding=inputs[str(survey_id)],
					model=model,
					k=k,
					config=config,
					affinity_scale=affinity_scale,
					gap_scale=gap_scale,
					x=x,
					y=y,
					z=z,
				)
				if not np.array_equal(labels[x, y, z], result.labels):
					raise ValueError(
					'lateral labels differ from the frozen smoothing replay result'
				)
				for item in (stats, aggregate):
					item.add_trace(
						replay,
						result.labels,
						result.diagnostics.distance,
						result.diagnostics.affinity,
						result.diagnostics.neighbor_count,
						result.diagnostics.message_entropy,
						result.changed_cost_magnitude,
					)
		for item in (stats, aggregate):
			item.add_grid_counts(valid)
			item.add_xy_edges(source, labels, valid, features, affinity_scale)
		per_survey[str(survey_id)] = stats.finish()
	expected = {
		'per_survey': per_survey,
		'aggregate': {**aggregate.finish(), 'survey_count': len(per_survey)},
		'resolved_scales': {'affinity': affinity_stats, 'emission_gap': gap_stats},
	}
	persisted = _mapping(diagnostics, 'diagnostics')
	actual = {
		'per_survey': persisted['per_survey'],
		'aggregate': persisted['aggregate'],
		'resolved_scales': persisted['resolved_scales'],
	}
	if actual != expected:
		raise ValueError('persisted lateral diagnostics differ from frozen sources')


def _validate_finite_tree(value: object, *, context: str) -> None:
	if isinstance(value, Mapping):
		for item in value.values():
			_validate_finite_tree(item, context=context)
	elif isinstance(value, list):
		for item in value:
			_validate_finite_tree(item, context=context)
	elif isinstance(value, float) and not math.isfinite(value):
		raise ValueError(f'{context} contains a non-finite value')


def _flatten_metrics(
	value: Mapping[str, object], prefix: str = ''
) -> Iterator[tuple[str, object]]:
	for name, item in sorted(value.items()):
		key = f'{prefix}.{name}' if prefix else name
		if isinstance(item, Mapping):
			yield from _flatten_metrics(item, key)
		else:
			yield key, item


def _quarantine(path: Path) -> None:
	shutil.move(
		str(path),
		str(
			path.with_name(
				f'{path.name}.quarantine-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
			)
		),
	)


__all__ = [
	'ARTIFACT_TYPE',
	'SCHEMA_VERSION',
	'MultiHeadLateralTargetExportConfig',
	'MultiHeadLateralTargetExportPlan',
	'export_multi_head_lateral_targets',
	'load_multi_head_lateral_target_manifest',
	'plan_multi_head_lateral_target_exports',
	'resolve_multi_head_lateral_target_export_config',
	'validate_multi_head_lateral_target_manifest',
]
