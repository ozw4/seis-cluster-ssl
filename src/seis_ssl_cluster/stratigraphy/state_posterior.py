"""Immutable export contract for frozen multi-head HMM state posteriors.

This module deliberately consumes the saved HMM artefacts.  It never calls the
clustering routine, fits a transformer, or changes a centre.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from csv import DictWriter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	discover_embedding_inputs,
	embedding_input_metadata,
	file_sha256,
)
from seis_ssl_cluster.clustering.residualization import read_residualizer_npz
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	forward_backward_state_posteriors,
	prepare_feature_batch_for_indices,
	squared_euclidean_emission_costs,
	viterbi_decode_costs,
)
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest

ARTIFACT_TYPE = 'strat_hmm_multi_head_state_posterior_manifest'
SCHEMA_VERSION = 1
POSTERIOR_SEMANTICS = 'ordered_path_cost_gibbs_state_marginal_v1'
CANONICAL_KS = (6, 8, 10)
_POSTERIOR_SUFFIX = '.state_posterior.npy'
_VALID_SUFFIX = '.valid_tokens.npy'
_METADATA_SUFFIX = '.state_posterior_metadata.json'
_Action = Literal['NEW', 'REUSE', 'QUARANTINE', 'ERROR']


@dataclass(frozen=True)
class MultiHeadStatePosteriorExportConfig:
	"""Validated policy for a posterior export rooted at one frozen HMM run."""

	source_hard_manifest: Path
	clustering_output_dir: Path
	source_embedding_dir: Path
	posterior_root: Path
	clustering_config: Path | None
	handoff_manifest: Path


@dataclass(frozen=True)
class MultiHeadStatePosteriorExportPlan:
	"""One immutable head action selected after complete input validation."""

	k: int
	action: _Action
	reason: str | None = None


def resolve_multi_head_state_posterior_export_config(
	config: Mapping[str, object],
) -> MultiHeadStatePosteriorExportConfig:
	"""Resolve the intentionally small, immutable posterior-export schema."""
	allowed = {
		'source_hard_manifest',
		'clustering_output_dir',
		'source_embedding_dir',
		'posterior_root',
		'clustering_config',
		'handoff_manifest',
		'outputs',
	}
	unknown = set(config) - allowed
	if unknown:
		raise ValueError(f'unknown state-posterior config keys: {sorted(unknown)}')
	if config.get('outputs') not in (None, {'overwrite': False}):
		raise ValueError('outputs must be omitted or exactly {overwrite: false}')

	def required(name: str) -> Path:
		value = config.get(name)
		if not isinstance(value, str) or not value:
			raise TypeError(f'{name} must be a non-empty path')
		return Path(value)

	source_hard_manifest = required('source_hard_manifest')
	if not source_hard_manifest.is_file():
		raise FileNotFoundError(
			f'source_hard_manifest is missing: {source_hard_manifest}'
		)
	clustering_output_dir = required('clustering_output_dir')
	source_embedding_dir = required('source_embedding_dir')
	posterior_root = required('posterior_root')
	config_value = config.get('clustering_config')
	clustering_config = None
	if config_value is not None:
		if not isinstance(config_value, str) or not config_value:
			raise TypeError('clustering_config must be a non-empty path when set')
		clustering_config = Path(config_value)
		if not clustering_config.is_file():
			raise FileNotFoundError(
				f'clustering_config is missing: {clustering_config}'
			)
	handoff_value = config.get('handoff_manifest')
	handoff = (
		posterior_root / 'multi_head_state_posterior_handoff.json'
		if handoff_value is None
		else Path(_non_empty_string(handoff_value, 'handoff_manifest'))
	)
	return MultiHeadStatePosteriorExportConfig(
		source_hard_manifest=source_hard_manifest,
		clustering_output_dir=clustering_output_dir,
		source_embedding_dir=source_embedding_dir,
		posterior_root=posterior_root,
		clustering_config=clustering_config,
		handoff_manifest=handoff,
	)


def plan_multi_head_state_posterior_exports(
	config: MultiHeadStatePosteriorExportConfig,
	*,
	only_missing: bool,
) -> list[MultiHeadStatePosteriorExportPlan]:
	"""Validate frozen inputs and classify output directories without writing."""
	source = load_multi_head_target_manifest(config.source_hard_manifest)
	_validate_source_manifest(source)
	_validate_frozen_inputs(config, source)
	plans: list[MultiHeadStatePosteriorExportPlan] = []
	for k in CANONICAL_KS:
		output = config.posterior_root / f'k{k}'
		if not output.exists():
			plans.append(MultiHeadStatePosteriorExportPlan(k, 'NEW'))
			continue
		try:
			_validate_complete_head(output, k=k, source=source, config=config)
		except (OSError, TypeError, ValueError) as exc:
			plans.append(
				MultiHeadStatePosteriorExportPlan(
					k,
					'QUARANTINE',
					str(exc),
				)
			)
		else:
			plans.append(
				MultiHeadStatePosteriorExportPlan(
					k,
					'REUSE' if only_missing else 'ERROR',
					None
					if only_missing
					else 'complete output exists; use --only-missing',
				)
			)
	return plans


def export_multi_head_state_posteriors(
	config: MultiHeadStatePosteriorExportConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
) -> list[MultiHeadStatePosteriorExportPlan]:
	"""Create all required heads and publish a handoff only when all succeed."""
	plans = plan_multi_head_state_posterior_exports(config, only_missing=only_missing)
	if dry_run:
		return plans
	if any(plan.action == 'ERROR' for plan in plans):
		raise FileExistsError(
			'; '.join(
				f'k={plan.k}: {plan.reason}' for plan in plans if plan.action == 'ERROR'
			)
		)
	source = load_multi_head_target_manifest(config.source_hard_manifest)
	for plan in plans:
		if plan.action == 'REUSE':
			continue
		if plan.action == 'QUARANTINE':
			_quarantine(config.posterior_root / f'k{plan.k}')
		_export_head(config, source, k=plan.k)
	_validate_all_heads(config, source)
	_write_json_atomic(config.handoff_manifest, _manifest(config))
	return plans


def load_multi_head_state_posterior_manifest(path: str | Path) -> dict[str, object]:
	"""Load an immutable manifest and verify every referenced identity."""
	try:
		payload = json.loads(Path(path).read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(
			f'state posterior manifest must be valid JSON: {path}'
		) from exc
	if not isinstance(payload, dict):
		raise TypeError('state posterior manifest must be a JSON object')
	validate_multi_head_state_posterior_manifest(payload)
	return payload


def validate_multi_head_state_posterior_manifest(payload: Mapping[str, object]) -> None:
	"""Strictly validate schema, provenance, arrays, and common valid masks."""
	_required_keys(
		payload,
		{
			'artifact_type',
			'schema_version',
			'posterior_semantics',
			'head_ks',
			'cost_temperature',
			'source_hard_manifest',
			'source_hard_export_handoff',
			'source_embedding',
			'heads',
		},
	)
	if (
		payload['artifact_type'],
		payload['schema_version'],
		payload['posterior_semantics'],
	) != (
		ARTIFACT_TYPE,
		SCHEMA_VERSION,
		POSTERIOR_SEMANTICS,
	):
		raise ValueError('unsupported state posterior manifest schema')
	if payload['head_ks'] != list(CANONICAL_KS) or payload['cost_temperature'] != 1.0:
		raise ValueError(
			'state posterior manifest must use canonical K values and temperature'
		)
	_hashed_path(payload['source_hard_manifest'], 'source_hard_manifest')
	source = load_multi_head_target_manifest(
		Path(
			str(
				_mapping(payload['source_hard_manifest'], 'source_hard_manifest')[
					'path'
				]
			)
		)
	)
	_validate_source_manifest(source)
	source_model_identities = _validate_manifest_hard_source_anchor(
		payload, source
	)
	if payload['source_embedding'] != source['source_embedding']:
		raise ValueError(
			'posterior source embedding identity differs from hard manifest'
		)
	_validate_manifest_source_embedding_identity(
		_mapping(payload['source_embedding'], 'source_embedding')
	)
	heads = _mapping(payload['heads'], 'heads')
	if set(heads) != {str(k) for k in CANONICAL_KS}:
		raise ValueError('posterior manifest heads must contain K=6/8/10')
	_validate_manifest_heads(heads, source, source_model_identities)


def _validate_manifest_heads(
	heads: Mapping[str, object],
	source: Mapping[str, object],
	source_model_identities: Mapping[str, object],
) -> None:
	common: dict[str, np.ndarray] = {}
	for k in CANONICAL_KS:
		head = _mapping(heads[str(k)], f'head k={k}')
		_required_keys(head, {'model', 'surveys', 'diagnostics'})
		_validate_head_mapping(head, k=k)
		if _mapping(head['model'], f'posterior model k={k}') != _mapping(
			source_model_identities[str(k)], f'hard source model k={k}'
		):
			raise ValueError(
				f'posterior k={k} model identity differs from hard manifest source'
			)
		surveys = _mapping(head['surveys'], f'head k={k} surveys')
		if set(surveys) != set(
			_mapping(source['heads'], 'source heads')[str(k)]['surveys']
		):
			raise ValueError(f'posterior k={k} survey set differs from source')
		for survey_id, raw in surveys.items():
			entry = _mapping(raw, f'k={k} survey {survey_id}')
			_required_keys(entry, {'posterior', 'valid_tokens', 'metadata', 'source'})
			posterior_path = _hashed_path(entry['posterior'], 'posterior')
			valid_path = _hashed_path(entry['valid_tokens'], 'valid_tokens')
			_hashed_path(entry['metadata'], 'metadata')
			_validate_source_reference(entry['source'])
			expected_source = _reference(
				_source_label_path(
					_mapping(
						_source_head(source, k=k)['surveys'][survey_id],
						'source survey',
					)
				)
			)
			if entry['source'] != expected_source:
				raise ValueError(
					f'posterior k={k} {survey_id} hard-label provenance differs'
				)
			posterior = np.load(posterior_path, mmap_mode='r', allow_pickle=False)
			valid = np.load(valid_path, mmap_mode='r', allow_pickle=False)
			_validate_posterior_array(posterior, valid, k=k)
			if survey_id in common and not np.array_equal(common[survey_id], valid):
				raise ValueError(
					f'posterior k={k} valid mask differs from K=6 for {survey_id}'
				)
			common.setdefault(survey_id, np.asarray(valid))


def _validate_manifest_hard_source_anchor(
	payload: Mapping[str, object], source: Mapping[str, object]
) -> Mapping[str, object]:
	handoff_path, model_identities = _validate_hard_source_model_anchor(
		source, config=None
	)
	if payload['source_hard_export_handoff'] != _reference(handoff_path):
		raise ValueError(
			'posterior hard-source export handoff identity differs from hard manifest'
		)
	return model_identities


def _export_head(
	config: MultiHeadStatePosteriorExportConfig,
	source: Mapping[str, object],
	*,
	k: int,
) -> None:
	final = config.posterior_root / f'k{k}'
	config.posterior_root.mkdir(parents=True, exist_ok=True)
	temporary = Path(
		tempfile.mkdtemp(prefix=f'.k{k}.posterior.', dir=config.posterior_root)
	)
	try:
		model = _load_model(config, k=k)
		surveys: dict[str, object] = {}
		per_survey: dict[str, object] = {}
		aggregate_stats = _PosteriorStats(k)
		inputs = {
			item.survey_id: item
			for item in discover_embedding_inputs(config.source_embedding_dir)
		}
		for survey_id, raw in _source_head(source, k=k)['surveys'].items():
			entry, diagnostics = _export_survey(
				temporary,
				_required_embedding(inputs, survey_id),
				_mapping(raw, 'source survey'),
				model,
				statistics=(_PosteriorStats(k), aggregate_stats),
			)
			surveys[survey_id] = entry
			per_survey[survey_id] = diagnostics
		aggregate = aggregate_stats.finish()
		aggregate['survey_count'] = len(per_survey)
		diagnostics = {
			'per_survey': per_survey,
			'aggregate': aggregate,
		}
		diagnostics_json = temporary / 'diagnostics.json'
		diagnostics_csv = temporary / 'diagnostics.csv'
		_write_json_atomic(diagnostics_json, diagnostics)
		_write_diagnostics_csv(diagnostics_csv, diagnostics)
		head = {
			'model': model['identity'],
			'surveys': surveys,
			'diagnostics': {
				**diagnostics,
				'json': _reference(diagnostics_json),
				'csv': _reference(diagnostics_csv),
			},
		}
		# Validation before publish catches malformed NPY metadata and accidental
		# partial writes while the directory is still invisible to consumers.
		_validate_head_mapping(head, k=k)
		_write_json_atomic(
			temporary / 'head_metadata.json',
			_rebase_head_paths(head, old_root=temporary, new_root=final),
		)
		temporary.replace(final)
	except BaseException:
		shutil.rmtree(temporary, ignore_errors=True)
		raise


def _export_survey(
	root: Path,
	embedding: object,
	source: Mapping[str, object],
	model: Mapping[str, object],
	*,
	statistics: tuple[_PosteriorStats, _PosteriorStats],
) -> tuple[dict[str, object], dict[str, object]]:
	if not isinstance(embedding, EmbeddingInput):
		raise TypeError('embedding input is invalid')
	centers = np.asarray(model['centers'])
	k = centers.shape[0]
	source_label = _source_label_path(source)
	labels = np.load(source_label, mmap_mode='r', allow_pickle=False)
	if labels.ndim != 3:
		raise ValueError('source hard labels must be a 3D token grid')
	valid = np.asarray(labels >= 0, dtype=np.bool_)
	hard_valid = np.load(
		_hashed_path(source['valid_tokens'], 'hard target valid_tokens'),
		mmap_mode='r',
		allow_pickle=False,
	)
	if not np.array_equal(valid, hard_valid):
		raise ValueError(
			f'frozen hard-label valid mask differs from hard target: '
			f'{embedding.survey_id}'
		)
	posterior_path = root / f'{embedding.survey_id}{_POSTERIOR_SUFFIX}'
	valid_path = root / f'{embedding.survey_id}{_VALID_SUFFIX}'
	output = np.lib.format.open_memmap(
		posterior_path,
		mode='w+',
		dtype=np.float32,
		shape=(*labels.shape, k),
	)
	output[...] = 0.0
	stats, aggregate_stats = statistics
	try:
		x_count, y_count, z_count = labels.shape
		for x_index in range(x_count):
			for y_index in range(y_count):
				z_indices = np.flatnonzero(valid[x_index, y_index])
				stats.trace(z_indices.size)
				aggregate_stats.trace(z_indices.size)
				if not z_indices.size:
					continue
				flat = ((x_index * y_count + y_index) * z_count + z_indices).astype(
					np.int64
				)
				features = prepare_feature_batch_for_indices(
					embedding,
					flat,
					residualizer=model['residualizer'],
					preprocessor=model['preprocessor'],
					emission_source=str(model['emission_source']),
				)
				costs = squared_euclidean_emission_costs(features, centers)
				expected, weight = _expected_boundaries(
					model['hmm'], k=k, length=z_indices.size
				)
				replay = viterbi_decode_costs(
					costs,
					model['transition_costs'],
					initial_state_costs=model['initial_costs'],
					terminal_state_costs=model['terminal_costs'],
					expected_boundary_count=expected,
					boundary_count_weight=weight,
				)
				if not np.array_equal(replay, labels[x_index, y_index, z_indices]):
					raise ValueError(
						'Viterbi replay differs from frozen hard labels: '
						f'{embedding.survey_id}'
					)
				result = forward_backward_state_posteriors(
					costs,
					model['transition_costs'],
					initial_state_costs=model['initial_costs'],
					terminal_state_costs=model['terminal_costs'],
					expected_boundary_count=expected,
					boundary_count_weight=weight,
					cost_temperature=1.0,
				)
				output[x_index, y_index, z_indices] = result.posterior.astype(
					np.float32
				)
				stats.add(result.posterior, replay)
				aggregate_stats.add(result.posterior, replay)
	finally:
		output.flush()
		del output
	np.save(valid_path, valid, allow_pickle=False)
	metadata_path = root / f'{embedding.survey_id}{_METADATA_SUFFIX}'
	metadata = {
		'survey_id': embedding.survey_id,
		'k': k,
		'posterior_semantics': POSTERIOR_SEMANTICS,
		'boundary_policy': 'compacted_trace_hard_transition_adjacent_v1',
		'source_label_path': str(source_label),
		'source_label_sha256': file_sha256(source_label),
	}
	_write_json_atomic(metadata_path, metadata)
	return (
		{
			'posterior': _reference(posterior_path),
			'valid_tokens': _reference(valid_path),
			'metadata': _reference(metadata_path),
			'source': _reference(source_label),
		},
		stats.finish(),
	)


class _PosteriorStats:
	def __init__(self, k: int) -> None:
		self.k = k
		# Diagnostics are accumulated in fixed-size histograms so export memory is
		# independent of survey token count.  The per-trace posterior is bounded by
		# the vertical trace processed by the producer.
		self.entropy = _FixedHistogram(0.0, float(np.log(k)))
		self.top1 = _FixedHistogram(0.0, 1.0)
		self.margin = _FixedHistogram(0.0, 1.0)
		self.viterbi = _FixedHistogram(0.0, 1.0)
		self.expected = _FixedHistogram(0.0, 1.0)
		self.usage = np.zeros(k, dtype=np.float64)
		self.boundary = _FixedHistogram(0.0, float(np.log(k)))
		self.interior = _FixedHistogram(0.0, float(np.log(k)))
		self.mismatch = self.tokens = self.empty = self.single = self.violations = 0
		self.max_decrease = 0.0
		self.lengths = _LogLengthHistogram()

	def trace(self, length: int) -> None:
		self.lengths.add(length)
		self.empty += int(length == 0)
		self.single += int(length == 1)

	def add(self, posterior: np.ndarray, labels: np.ndarray) -> None:
		p = np.asarray(posterior, dtype=np.float64)
		top = np.sort(p, axis=1)[:, -2:]
		entropy = -np.sum(p * np.log(np.maximum(p, 1e-30)), axis=1)
		self.entropy.add(entropy)
		self.top1.add(top[:, 1])
		self.margin.add(top[:, 1] - top[:, 0])
		self.viterbi.add(p[np.arange(p.shape[0]), labels])
		expected_order = p @ np.linspace(0.0, 1.0, self.k)
		self.expected.add(expected_order)
		self.usage += p.sum(axis=0)
		self.tokens += p.shape[0]
		self.mismatch += int(np.count_nonzero(np.argmax(p, axis=1) != labels))
		boundary = np.zeros(labels.size, dtype=bool)
		changed = np.flatnonzero(np.diff(labels) != 0)
		boundary[changed] = True
		boundary[changed + 1] = True
		self.boundary.add(entropy[boundary])
		self.interior.add(entropy[~boundary])
		decrease = np.diff(expected_order)
		self.violations += int(np.count_nonzero(decrease < 0))
		self.max_decrease = max(
			self.max_decrease, float(max(0.0, -decrease.min(initial=0.0)))
		)

	def finish(self) -> dict[str, object]:
		return {
			'posterior_entropy_quantiles': self.entropy.quantiles(),
			'top1_probability_quantiles': self.top1.quantiles(),
			'top1_minus_top2_margin_quantiles': self.margin.quantiles(),
			'viterbi_state_posterior_probability': self.viterbi.quantiles(),
			'expected_normalized_order': self.expected.quantiles(),
			'effective_posterior_state_usage': float(
				np.exp(
					-np.sum(
						(self.usage / max(self.tokens, 1))
						* np.log(np.maximum(self.usage / max(self.tokens, 1), 1e-30))
					)
				)
			),
			'boundary_versus_interior_entropy': {
				'boundary': self.boundary.quantiles(),
				'interior': self.interior.quantiles(),
			},
			'posterior_argmax_viterbi_mismatch_rate': self.mismatch
			/ max(self.tokens, 1),
			'per_trace_monotonicity': {
				'violation_count': self.violations,
				'max_decrease': self.max_decrease,
			},
			'empty_trace_count': self.empty,
			'single_valid_token_trace_count': self.single,
			'trace_length_summary': self.lengths.quantiles(),
		}


class _FixedHistogram:
	"""Bounded-memory streaming quantiles for a finite diagnostic range."""

	_BIN_COUNT = 4096

	def __init__(self, lower: float, upper: float) -> None:
		if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
			raise ValueError('histogram range must be finite and ordered')
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
		if not np.all(np.isfinite(array)) or np.any(array < self.lower) or np.any(
			array > self.upper
		):
			raise ValueError('histogram values are outside the diagnostic range')
		if self.lower == self.upper:
			indices = np.zeros(array.size, dtype=np.intp)
		else:
			scaled = (array - self.lower) / (self.upper - self.lower)
			indices = np.minimum(
				(scaled * self._BIN_COUNT).astype(np.intp),
				self._BIN_COUNT - 1,
			)
		self.counts += np.bincount(indices, minlength=self._BIN_COUNT)
		self.count += array.size
		self.minimum = min(self.minimum, float(array.min()))
		self.maximum = max(self.maximum, float(array.max()))

	def quantiles(self) -> dict[str, float]:
		if not self.count:
			return _empty_quantiles()
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


class _LogLengthHistogram:
	"""Bounded quantiles for non-negative trace lengths over a wide range."""

	def __init__(self) -> None:
		self._histogram = _FixedHistogram(
			0.0,
			float(np.log1p(np.iinfo(np.int64).max)),
		)

	def add(self, length: int) -> None:
		if length < 0:
			raise ValueError('trace length must be non-negative')
		self._histogram.add(np.array([np.log1p(length)], dtype=np.float64))

	def quantiles(self) -> dict[str, float]:
		return {
			name: float(np.expm1(value))
			for name, value in self._histogram.quantiles().items()
		}


def _load_model(
	config: MultiHeadStatePosteriorExportConfig, *, k: int
) -> dict[str, object]:
	model_dir = config.clustering_output_dir / 'models' / f'k{k}'
	paths = {
		name: model_dir / filename
		for name, filename in {
			'preprocessor': 'preprocessor.joblib',
			'hmm_model': 'hmm_model.joblib',
			'centers': 'cluster_centers.npy',
			'metadata': 'clustering_metadata.json',
		}.items()
	}
	if not all(path.is_file() for path in paths.values()):
		raise FileNotFoundError(f'frozen model artifacts are incomplete for k={k}')
	hmm = joblib.load(paths['hmm_model'])
	if not isinstance(hmm, Mapping):
		raise TypeError(f'hmm_model must be a mapping for k={k}')
	centers = np.load(paths['centers'], mmap_mode='r', allow_pickle=False)
	if centers.shape != (k, centers.shape[1]):
		raise ValueError(f'center shape does not match k={k}')
	residualizer_path = config.clustering_output_dir / 'models' / 'residualizer.npz'
	residualizer = (
		read_residualizer_npz(residualizer_path)
		if residualizer_path.is_file()
		else None
	)
	frozen_identity: dict[str, object] = {
		name: _reference(path) for name, path in paths.items()
	}
	if residualizer_path.is_file():
		frozen_identity['residualizer'] = _reference(residualizer_path)
	identity = dict(frozen_identity)
	if config.clustering_config is not None:
		identity['clustering_config'] = _reference(config.clustering_config)
	return {
		'preprocessor': joblib.load(paths['preprocessor']),
		'hmm': hmm,
		'centers': np.asarray(centers, dtype=np.float32),
		'residualizer': residualizer,
		'emission_source': hmm.get('emission_source', 'embedding'),
		'transition_costs': np.asarray(hmm['transition_costs'], dtype=np.float32),
		'initial_costs': np.asarray(hmm['initial_state_costs'], dtype=np.float32),
		'terminal_costs': np.asarray(hmm['terminal_state_costs'], dtype=np.float32),
		'identity': identity,
		'frozen_identity': frozen_identity,
	}


def _expected_boundaries(
	hmm: Mapping[str, object], *, k: int, length: int
) -> tuple[int | None, float]:
	prior = hmm.get('path_prior', {})
	if not isinstance(prior, Mapping) or not prior.get('enabled', False):
		return None, 0.0
	value = prior.get('expected_boundaries', {})
	if not isinstance(value, Mapping) or not value.get('enabled', False):
		return None, 0.0
	weight = float(value.get('weight', 0.0))
	if weight == 0.0:
		return None, 0.0
	target = k - 1 if value.get('target') == 'auto_k_minus_1' else int(value['target'])
	return min(target, length - 1), weight


def _validate_frozen_inputs(
	config: MultiHeadStatePosteriorExportConfig, source: Mapping[str, object]
) -> None:
	source_embedding = _mapping(source['source_embedding'], 'source_embedding')
	recorded_input_dir = Path(
		_non_empty_string(source_embedding['input_dir'], 'input_dir')
	)
	if recorded_input_dir.resolve() != config.source_embedding_dir.resolve():
		raise ValueError('source embedding directory differs from hard manifest')
	inputs = {
		item.survey_id: item
		for item in discover_embedding_inputs(config.source_embedding_dir)
	}
	_validate_source_embedding_identity(source_embedding, inputs)
	models: dict[int, Mapping[str, object]] = {}
	for k in CANONICAL_KS:
		models[k] = _load_model(config, k=k)
		for survey_id, raw in _source_head(source, k=k)['surveys'].items():
			source_label = _source_label_path(_mapping(raw, 'source survey'))
			if not source_label.is_file():
				raise ValueError(f'hard label identity drift for k={k} {survey_id}')
	_validate_hard_source_model_anchor(source, config=config, models=models)


def _validate_source_embedding_identity(
	source_embedding: Mapping[str, object],
	inputs: Mapping[str, EmbeddingInput],
) -> None:
	"""Revalidate every embedding artifact before frozen-model replay."""
	recorded = _mapping(source_embedding.get('surveys'), 'source embedding surveys')
	if set(recorded) != set(inputs):
		raise ValueError('source embedding survey set differs from hard manifest')
	for survey_id, embedding in inputs.items():
		entry = _mapping(recorded[survey_id], f'source embedding {survey_id}')
		expected_paths = {
			'embedding': embedding.embeddings_path,
			'metadata': embedding.metadata_path,
			'valid_tokens': embedding.valid_tokens_path,
		}
		for name, path in expected_paths.items():
			path_key = f'{name}_path'
			hash_key = f'{name}_sha256'
			if (
				Path(_non_empty_string(entry.get(path_key), path_key)).resolve()
				!= path.resolve()
				or file_sha256(path)
				!= _non_empty_string(entry.get(hash_key), hash_key)
			):
				raise ValueError(
					f'source embedding {name} identity differs for {survey_id}'
				)
		if entry.get('metadata') != embedding_input_metadata(embedding):
			raise ValueError(
				f'source embedding metadata identity differs for {survey_id}'
			)


def _validate_manifest_source_embedding_identity(
	source_embedding: Mapping[str, object],
) -> None:
	"""Re-hash the live embedding inputs referenced by a published manifest."""
	input_dir = Path(_non_empty_string(source_embedding.get('input_dir'), 'input_dir'))
	inputs = {
		item.survey_id: item for item in discover_embedding_inputs(input_dir)
	}
	_validate_source_embedding_identity(source_embedding, inputs)


def _validate_hard_source_model_anchor(  # noqa: C901, PLR0912, PLR0915
	source: Mapping[str, object],
	*,
	config: MultiHeadStatePosteriorExportConfig | None,
	models: Mapping[int, Mapping[str, object]] | None = None,
) -> tuple[Path, Mapping[str, object]]:
	"""Bind posterior inputs to the frozen export that produced hard targets.

	The hard manifest intentionally remains a target-only contract.  Its per-K
	pseudo-target roots identify the completed strict-export handoff, which is
	the immutable provenance record for the labels.  Replaying labels alone is
	not sufficient: this check also binds the selected model metadata and the
	clustering config to that source record before any posterior is written.
	"""
	roots = {
		Path(
			_non_empty_string(
				_source_head(source, k=k)['pseudo_target_root'], 'pseudo_target_root'
			)
		)
		for k in CANONICAL_KS
	}
	if len(roots) != len(CANONICAL_KS):
		raise ValueError('hard manifest pseudo-target roots must be distinct per K')
	parents = {root.parent for root in roots}
	if len(parents) != 1:
		raise ValueError('hard manifest heads do not share a pseudo-target root')
	handoff_path = next(iter(parents)) / 'multi_head_pseudo_target_export_handoff.json'
	try:
		payload = json.loads(handoff_path.read_text(encoding='utf-8'))
	except FileNotFoundError as exc:
		raise ValueError('hard manifest source export handoff is missing') from exc
	except json.JSONDecodeError as exc:
		raise ValueError(
			'hard manifest source export handoff must be valid JSON'
		) from exc
	if not isinstance(payload, Mapping):
		raise TypeError('hard manifest source export handoff must be a mapping')
	if (
		payload.get('artifact_type')
		!= 'strat_hmm_multi_head_pseudo_target_export_handoff'
		or payload.get('schema_version') != 2
		or payload.get('completion_status') != 'COMPLETE'
	):
		raise ValueError('hard manifest source export handoff schema/status mismatch')
	if (
		Path(
			_non_empty_string(payload.get('pseudo_target_root'), 'pseudo_target_root')
		).resolve()
		!= next(iter(parents)).resolve()
	):
		raise ValueError('hard manifest source export handoff root mismatch')
	clustering = _mapping(payload.get('clustering'), 'hard source clustering')
	head_values = _mapping(payload.get('heads'), 'hard source export heads')
	metadata_hashes = _mapping(
		clustering.get('metadata_sha256'), 'hard source clustering metadata hashes'
	)
	model_artifacts = clustering.get('model_artifacts')
	if model_artifacts is None:
		raise ValueError(
			'hard manifest source export lacks frozen model identities'
		)
	model_artifacts = _mapping(
		model_artifacts, 'hard source frozen model identities'
	)
	label_hashes = _mapping(clustering.get('labels'), 'hard source clustering labels')
	model_identities: dict[str, object] = {}
	for k in CANONICAL_KS:
		root = Path(
			_non_empty_string(
				_source_head(source, k=k)['pseudo_target_root'], 'pseudo_target_root'
			)
		)
		head = _mapping(head_values.get(str(k)), f'hard source export k={k}')
		head_root = Path(
			_non_empty_string(head.get('pseudo_target_root'), 'pseudo_target_root')
		)
		if head_root.resolve() != root.resolve():
			raise ValueError(f'hard manifest source export root mismatch for k={k}')
		if (
			str(k) not in metadata_hashes
			or str(k) not in label_hashes
		):
			raise ValueError(f'hard manifest source export provenance is missing k={k}')
		if str(k) not in model_artifacts:
			raise ValueError(
				f'hard manifest source export provenance is missing k={k}'
			)
		identity = _hard_source_model_identity(
			_mapping(model_artifacts[str(k)], f'hard source model k={k}'),
			clustering=clustering,
		)
		metadata = _mapping(identity['metadata'], f'hard source model metadata k={k}')
		if metadata.get('sha256') != metadata_hashes[str(k)]:
			raise ValueError(
				f'hard manifest source model metadata differs for k={k}'
			)
		model_identities[str(k)] = identity
		labels = _mapping(label_hashes[str(k)], f'hard source labels k={k}')
		for survey_id, raw in _source_head(source, k=k)['surveys'].items():
			label = _source_label_path(_mapping(raw, 'source survey'))
			if labels.get(label.name) != file_sha256(label):
				raise ValueError(
					'hard manifest source export label identity mismatch for '
					f'k={k} {survey_id}'
				)
	if config is None:
		return handoff_path, model_identities
	clustering_path = Path(
		_non_empty_string(clustering.get('path'), 'clustering.path')
	)
	if clustering_path.resolve() != config.clustering_output_dir.resolve():
		raise ValueError('selected clustering output differs from hard manifest source')
	if config.clustering_config is None:
		raise ValueError(
			'clustering_config is required to verify hard source provenance'
		)
	recorded_config = Path(
		_non_empty_string(clustering.get('config_path'), 'clustering.config_path')
	)
	if (
		recorded_config.resolve() != config.clustering_config.resolve()
		or clustering.get('config_sha256') != file_sha256(config.clustering_config)
	):
		raise ValueError('selected clustering config differs from hard manifest source')
	if models is None:  # pragma: no cover - callers with a config always provide it
		raise AssertionError('loaded frozen models are required')
	for k in CANONICAL_KS:
		identity = _mapping(
			models[k]['frozen_identity'], f'selected frozen model k={k}'
		)
		expected_frozen_identity = dict(
			_mapping(model_identities[str(k)], f'hard source model k={k}')
		)
		expected_frozen_identity.pop('clustering_config')
		if identity != expected_frozen_identity:
			raise ValueError(f'selected frozen model identity differs for k={k}')
	return handoff_path, model_identities


def _hard_source_model_identity(
	value: Mapping[str, object], *, clustering: Mapping[str, object]
) -> dict[str, object]:
	"""Validate and normalize one frozen-model identity from the hard handoff."""
	if set(value) != {
		'preprocessor',
		'hmm_model',
		'centers',
		'metadata',
		'residualizer',
	}:
		raise ValueError('hard source frozen model identity keys are invalid')
	result: dict[str, object] = {}
	for name in ('preprocessor', 'hmm_model', 'centers', 'metadata'):
		path = _hashed_path(value[name], f'hard source {name}')
		result[name] = _reference(path)
	residualizer = value['residualizer']
	if residualizer is not None:
		result['residualizer'] = _reference(
			_hashed_path(residualizer, 'hard source residualizer')
		)
	config = {
		'path': _non_empty_string(clustering.get('config_path'), 'config_path'),
		'sha256': _non_empty_string(clustering.get('config_sha256'), 'config_sha256'),
	}
	_hashed_path(config, 'hard source clustering config')
	result['clustering_config'] = config
	return result


def _validate_all_heads(
	config: MultiHeadStatePosteriorExportConfig, source: Mapping[str, object]
) -> None:
	common_masks: dict[str, np.ndarray] = {}
	for k in CANONICAL_KS:
		path = config.posterior_root / f'k{k}'
		_validate_complete_head(path, k=k, source=source, config=config)
		head = _load_head_metadata(path)
		for survey_id, raw in _mapping(head['surveys'], 'surveys').items():
			valid = np.load(
				_hashed_path(
					_mapping(raw, f'k={k} survey {survey_id}')['valid_tokens'],
					'valid_tokens',
				),
				mmap_mode='r',
				allow_pickle=False,
			)
			if survey_id in common_masks and not np.array_equal(
				common_masks[survey_id], valid
			):
				raise ValueError(
					f'posterior k={k} valid mask differs from K=6 for {survey_id}'
				)
			common_masks.setdefault(survey_id, np.asarray(valid))


def _validate_complete_head(
	path: Path,
	*,
	k: int,
	source: Mapping[str, object],
	config: MultiHeadStatePosteriorExportConfig,
) -> None:
	if not path.is_dir() or any(part.startswith('.') for part in path.parts):
		raise ValueError(f'partial posterior output is not complete: {path}')
	head_path = path / 'head_metadata.json'
	if not head_path.is_file():
		raise ValueError(f'posterior head metadata is missing: {path}')
	head = _load_head_metadata(path)
	_validate_head_mapping(head, k=k)
	if head['model'] != _load_model(config, k=k)['identity']:
		raise ValueError(f'frozen model identity drift for k={k}')
	surveys = _mapping(head['surveys'], 'surveys')
	if set(surveys) != set(_source_head(source, k=k)['surveys']):
		raise ValueError(f'posterior k={k} survey set differs from source')
	for survey_id, raw in surveys.items():
		source_entry = _mapping(
			_source_head(source, k=k)['surveys'][survey_id], 'source survey'
		)
		if _mapping(_mapping(raw, 'entry')['source'], 'source') != _reference(
			_source_label_path(source_entry)
		):
			raise ValueError(f'hard label provenance drift for k={k} {survey_id}')


def _validate_head_mapping(head: Mapping[str, object], *, k: int) -> None:
	_validate_model_identity(head['model'], k=k)
	_validate_diagnostics(head['diagnostics'])
	for raw in _mapping(head['surveys'], 'surveys').values():
		entry = _mapping(raw, 'survey')
		_required_keys(entry, {'posterior', 'valid_tokens', 'metadata', 'source'})
		posterior = np.load(
			_hashed_path(entry['posterior'], 'posterior'),
			mmap_mode='r',
			allow_pickle=False,
		)
		valid = np.load(
			_hashed_path(entry['valid_tokens'], 'valid'),
			mmap_mode='r',
			allow_pickle=False,
		)
		_validate_posterior_array(posterior, valid, k=k)
		metadata_path = _hashed_path(entry['metadata'], 'metadata')
		source_reference = _mapping(entry['source'], 'source hard label')
		_validate_source_reference(source_reference)
		try:
			metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
		except json.JSONDecodeError as exc:
			raise ValueError('posterior metadata must be valid JSON') from exc
		if not isinstance(metadata, Mapping):
			raise TypeError('posterior metadata must be a mapping')
		if (
			metadata.get('k') != k
			or metadata.get('posterior_semantics') != POSTERIOR_SEMANTICS
			or metadata.get('source_label_path') != source_reference['path']
			or metadata.get('source_label_sha256') != source_reference['sha256']
		):
			raise ValueError('posterior metadata provenance differs from entry')


def _load_head_metadata(path: Path) -> Mapping[str, object]:
	head = json.loads((path / 'head_metadata.json').read_text(encoding='utf-8'))
	if not isinstance(head, dict):
		raise TypeError('posterior head metadata must be an object')
	return head


def _validate_diagnostics(value: object) -> None:
	diagnostics = _mapping(value, 'diagnostics')
	_required_keys(diagnostics, {'per_survey', 'aggregate', 'json', 'csv'})
	json_path = _hashed_path(diagnostics['json'], 'diagnostics JSON')
	_hashed_path(diagnostics['csv'], 'diagnostics CSV')
	try:
		payload = json.loads(json_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError('diagnostics JSON must be valid') from exc
	if payload != {
		'per_survey': diagnostics['per_survey'],
		'aggregate': diagnostics['aggregate'],
	}:
		raise ValueError('diagnostics JSON differs from head metadata')


def _validate_model_identity(value: object, *, k: int) -> None:
	identity = _mapping(value, f'k={k} model identity')
	allowed = {
		'preprocessor',
		'hmm_model',
		'centers',
		'metadata',
		'residualizer',
		'clustering_config',
	}
	if (
		not {'preprocessor', 'hmm_model', 'centers', 'metadata'} <= set(identity)
		or set(identity) - allowed
	):
		raise ValueError(f'k={k} model identity keys are invalid')
	for name in ('preprocessor', 'hmm_model', 'centers', 'metadata'):
		_hashed_path(identity.get(name), f'k={k} {name}')
	for name in ('residualizer', 'clustering_config'):
		if name in identity:
			_hashed_path(identity[name], f'k={k} {name}')


def _validate_posterior_array(
	posterior: np.ndarray, valid: np.ndarray, *, k: int
) -> None:
	if posterior.dtype != np.float32 or posterior.ndim != 4 or posterior.shape[-1] != k:
		raise ValueError(f'posterior must be float32 [X,Y,Z,{k}]')
	if valid.dtype != np.bool_ or valid.shape != posterior.shape[:3]:
		raise ValueError('posterior valid mask shape/dtype mismatch')
	if not np.all(np.isfinite(posterior)) or np.any(posterior < 0):
		raise ValueError('posterior must be finite and non-negative')
	if not np.allclose(posterior[valid].sum(axis=1), 1.0, rtol=0.0, atol=2e-6):
		raise ValueError('valid posterior rows must sum to one')
	if np.any(posterior[~valid] != 0):
		raise ValueError('invalid posterior rows must be zero')


def _source_head(source: Mapping[str, object], *, k: int) -> Mapping[str, object]:
	return _mapping(_mapping(source['heads'], 'heads')[str(k)], f'source k={k}')


def _required_embedding(
	inputs: Mapping[str, EmbeddingInput], survey_id: str
) -> EmbeddingInput:
	try:
		return inputs[survey_id]
	except KeyError as exc:
		raise ValueError(f'source embedding is missing survey {survey_id}') from exc


def _source_label_path(entry: Mapping[str, object]) -> Path:
	meta = _hashed_path(entry['metadata'], 'hard target metadata')
	payload = json.loads(meta.read_text(encoding='utf-8'))
	source = _mapping(payload.get('source'), 'hard target source')
	path = Path(_non_empty_string(source.get('source_label_path'), 'source_label_path'))
	digest = _non_empty_string(source.get('source_label_sha256'), 'source_label_sha256')
	if not path.is_file() or file_sha256(path) != digest:
		raise ValueError('hard target source-label hash mismatch')
	return path


def _validate_source_reference(value: object) -> None:
	_hashed_path(value, 'source hard label')


def _validate_source_manifest(source: Mapping[str, object]) -> None:
	if source.get('head_ks') != list(CANONICAL_KS):
		raise ValueError('source hard manifest must have canonical K=6/8/10')


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _hashed_path(value: object, name: str) -> Path:
	reference = _mapping(value, name)
	path = Path(_non_empty_string(reference.get('path'), f'{name}.path'))
	digest = _non_empty_string(reference.get('sha256'), f'{name}.sha256')
	if not path.is_file() or file_sha256(path) != digest:
		raise ValueError(f'{name} hash mismatch')
	return path


def _mapping(value: object, name: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{name} must be a mapping')
	return value


def _non_empty_string(value: object, name: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{name} must be a non-empty string')
	return value


def _required_keys(value: Mapping[str, object], keys: set[str]) -> None:
	if set(value) != keys:
		raise ValueError(f'manifest keys mismatch; expected {sorted(keys)}')


def _empty_quantiles() -> dict[str, float]:
	return {'p00': 0.0, 'p05': 0.0, 'p50': 0.0, 'p95': 0.0, 'p100': 0.0}


def _write_diagnostics_csv(path: Path, diagnostics: Mapping[str, object]) -> None:
	rows = list(_diagnostic_rows(diagnostics))
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		mode='w',
		encoding='utf-8',
		newline='',
		dir=path.parent,
		prefix=f'.{path.name}.',
		delete=False,
	) as stream:
		writer = DictWriter(
			stream,
			fieldnames=('scope', 'survey_id', 'metric', 'value'),
		)
		writer.writeheader()
		writer.writerows(rows)
		temporary = Path(stream.name)
	temporary.replace(path)


def _diagnostic_rows(diagnostics: Mapping[str, object]) -> Sequence[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for scope, survey_id, values in (
		('aggregate', '', diagnostics['aggregate']),
		*[('per_survey', survey_id, value) for survey_id, value in _mapping(
			diagnostics['per_survey'], 'per_survey'
		).items()],
	):
		for metric, value in _flatten_diagnostics(
			_mapping(values, 'diagnostic values')
		):
			rows.append(
				{
					'scope': scope,
					'survey_id': survey_id,
					'metric': metric,
					'value': value,
				}
			)
	return rows


def _flatten_diagnostics(
	value: Mapping[str, object], prefix: str = ''
) -> Sequence[tuple[str, object]]:
	flat: list[tuple[str, object]] = []
	for name, item in value.items():
		metric = f'{prefix}.{name}' if prefix else name
		if isinstance(item, Mapping):
			flat.extend(_flatten_diagnostics(item, metric))
		else:
			flat.append((metric, item))
	return flat


def _rebase_head_paths(
	head: Mapping[str, object], *, old_root: Path, new_root: Path
) -> dict[str, object]:
	"""Replace temporary artifact paths after validating their written bytes."""

	def rebase(value: object) -> object:
		if isinstance(value, Mapping):
			result = {name: rebase(item) for name, item in value.items()}
			path_value = result.get('path')
			if isinstance(path_value, str):
				candidate = Path(path_value)
				with suppress(ValueError):
					result['path'] = str(new_root / candidate.relative_to(old_root))
			return result
		if isinstance(value, list):
			return [rebase(item) for item in value]
		return value

	result = rebase(head)
	if not isinstance(result, dict):  # pragma: no cover
		raise TypeError('head must be a mapping')
	return result


def _quarantine(path: Path) -> None:
	destination = path.with_name(
		f'{path.name}.quarantine-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
	)
	shutil.move(str(path), str(destination))


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
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


def _manifest(config: MultiHeadStatePosteriorExportConfig) -> dict[str, object]:
	heads: dict[str, object] = {}
	for k in CANONICAL_KS:
		path = config.posterior_root / f'k{k}' / 'head_metadata.json'
		head = json.loads(path.read_text(encoding='utf-8'))
		if not isinstance(head, dict):
			raise TypeError(f'posterior head metadata must be an object: {path}')
		heads[str(k)] = head
	source = load_multi_head_target_manifest(config.source_hard_manifest)
	handoff_path, _ = _validate_hard_source_model_anchor(source, config=None)
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'posterior_semantics': POSTERIOR_SEMANTICS,
		'head_ks': list(CANONICAL_KS),
		'cost_temperature': 1.0,
		'source_hard_manifest': _reference(config.source_hard_manifest),
		'source_hard_export_handoff': _reference(handoff_path),
		'source_embedding': source['source_embedding'],
		'heads': heads,
	}


__all__ = [
	'ARTIFACT_TYPE',
	'CANONICAL_KS',
	'POSTERIOR_SEMANTICS',
	'SCHEMA_VERSION',
	'MultiHeadStatePosteriorExportConfig',
	'MultiHeadStatePosteriorExportPlan',
	'export_multi_head_state_posteriors',
	'load_multi_head_state_posterior_manifest',
	'plan_multi_head_state_posterior_exports',
	'resolve_multi_head_state_posterior_export_config',
	'validate_multi_head_state_posterior_manifest',
]
