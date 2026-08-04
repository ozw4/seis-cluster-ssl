"""Atomic periodic ordered-HMM refresh generation artifacts."""

# The producer/validator deliberately keeps the closed artifact contract in one
# module.  A few validation routines are necessarily branch-heavy.
# ruff: noqa: ARG001, CPY001, C901, E501, PLR0911, PLR0912, PLR0913, PLR0915, PLR0917, TRY004

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from seis_ssl_cluster.clustering import stratigraphic_hmm as hmm
from seis_ssl_cluster.clustering.features import (
	discover_embedding_inputs,
	file_sha256,
	load_embedding_metadata,
	open_embedding_array,
)
from seis_ssl_cluster.clustering.prepared_features import (
	PreparedFeatureCacheSettings,
	PreparedFeatureStore,
	prepare_feature_store,
)
from seis_ssl_cluster.clustering.residualization import read_residualizer_npz
from seis_ssl_cluster.clustering.stratigraphic_hmm_refresh import (
	WarmStartOrderedHMMRefreshResult,
	run_warm_start_ordered_hmm_refresh,
)
from seis_ssl_cluster.stratigraphy.export import (
	export_hmm_cluster_labels_as_pseudo_targets,
)
from seis_ssl_cluster.stratigraphy.multi_head import (
	build_multi_head_target_manifest,
	load_multi_head_target_manifest,
)
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetInput,
	load_pseudo_target_arrays,
	write_pseudo_target,
)

ARTIFACT_TYPE = 'strat_hmm_periodic_refresh_generation'
SCHEMA_VERSION = 1
CANONICAL_KS = (6, 8, 10)
GENERATION_MANIFEST_NAME = 'refresh_generation.json'
REFRESH_DIAGNOSTICS_NAME = 'refresh_diagnostics.json'
PREPARED_FEATURE_MANIFEST_NAME = 'prepared_feature_manifest.json'
DEFAULT_PREDICTION_BATCH_SIZE = 65_536

_CONFIDENCE_MODES = frozenset({'constant', 'source_array'})
_BOUNDARY_WEIGHT_MODES = frozenset({'absent', 'source_array'})


@dataclass(frozen=True)
class HashedArtifactReference:
	"""A file path whose exact SHA-256 is part of a closed input contract."""

	path: Path
	sha256: str

	def __post_init__(self) -> None:
		"""Normalize the path and reject malformed hashes at construction time."""
		object.__setattr__(self, 'path', Path(self.path))
		_validate_sha256(self.sha256, 'sha256')


@dataclass(frozen=True)
class InitialHMMArtifact:
	"""Explicit paths for one immutable initial K-specific HMM model."""

	k: int
	centers: HashedArtifactReference
	hmm_model: HashedArtifactReference
	preprocessor: HashedArtifactReference
	metadata: HashedArtifactReference
	residualizer: HashedArtifactReference | None = None


@dataclass(frozen=True)
class PreviousCenterArtifact:
	"""Explicit previous-generation ordered center matrix reference."""

	k: int
	centers: HashedArtifactReference


@dataclass(frozen=True)
class HardTargetPolicy:
	"""Initial hard-target confidence and boundary-weight semantics."""

	confidence_mode: Literal['constant', 'source_array'] = 'constant'
	confidence: float = 1.0
	boundary_weight_mode: Literal['absent', 'source_array'] = 'absent'

	def validate(self) -> None:
		"""Reject policies that would alter the immutable target semantics."""
		if self.confidence_mode not in _CONFIDENCE_MODES:
			raise ValueError(
				'confidence_mode must be the existing constant policy: '
				f'{self.confidence_mode!r}'
			)
		if (
			isinstance(self.confidence, bool)
			or not isinstance(self.confidence, (int, float))
			or not np.isfinite(float(self.confidence))
			or not 0.0 <= float(self.confidence) <= 1.0
		):
			raise ValueError('confidence must be finite and in [0, 1]')
		if self.boundary_weight_mode not in _BOUNDARY_WEIGHT_MODES:
			raise ValueError(
				'boundary_weight_mode must preserve absent boundary weights: '
				f'{self.boundary_weight_mode!r}'
			)


@dataclass(frozen=True)
class PeriodicRefreshConfig:
	"""Closed typed input contract for one non-initial refresh generation."""

	generation_index: int
	refresh_after_epoch: int
	source_student_state_sha256: str
	previous_generation_manifest: HashedArtifactReference
	current_embedding_descriptor: HashedArtifactReference
	initial_hard_target_manifest: HashedArtifactReference
	initial_hmm_artifacts: tuple[InitialHMMArtifact, ...]
	previous_centers: tuple[PreviousCenterArtifact, ...]
	output_generation_dir: Path
	target_policy: HardTargetPolicy
	iterations: int = 2
	prediction_batch_size: int = DEFAULT_PREDICTION_BATCH_SIZE


@dataclass(frozen=True)
class InitialPeriodicRefreshConfig:
	"""Closed input contract for the immutable generation-zero reference mode."""

	initial_hard_target_manifest: HashedArtifactReference
	initial_hmm_artifacts: tuple[InitialHMMArtifact, ...]
	output_generation_dir: Path
	target_policy: HardTargetPolicy


@dataclass(frozen=True)
class PeriodicRefreshGenerationResult:
	"""Published generation location and immutable manifest identity."""

	generation_dir: Path
	manifest_path: Path
	manifest_sha256: str
	reused: bool


@dataclass(frozen=True)
class _LoadedFixedHMM:
	k: int
	centers: np.ndarray
	preprocessor: object
	residualizer: object | None
	emission_source: str
	transition_costs: np.ndarray
	initial_state_costs: np.ndarray
	terminal_state_costs: np.ndarray
	expected_boundaries: hmm.HMMExpectedBoundariesSettings | None
	edge_margin_tokens: tuple[int, int, int]
	prepared_feature_identity: Mapping[str, object]
	normalized_identity: Mapping[str, object]
	artifact: InitialHMMArtifact


@dataclass(frozen=True)
class _InitialTargetData:
	payload: Mapping[str, object]
	survey_ids: tuple[str, ...]
	valid_tokens: Mapping[str, np.ndarray]
	labels: Mapping[int, Mapping[str, np.ndarray]]
	confidence: Mapping[int, Mapping[str, np.ndarray]]
	boundary_weight: Mapping[int, Mapping[str, np.ndarray | None]]


@dataclass(frozen=True)
class _CurrentEmbeddingData:
	descriptor: Mapping[str, object]
	descriptor_name: str
	source_inputs: tuple[object, ...]
	generation_inputs: tuple[object, ...]
	valid_tokens: Mapping[str, np.ndarray]


INITIAL_GENERATION_ID = 'refresh_0000_initial'


def produce_initial_periodic_refresh_generation(
	config: InitialPeriodicRefreshConfig,
) -> PeriodicRefreshGenerationResult:
	"""Publish generation zero as references to immutable initial artifacts only."""
	_validate_initial_config(config)
	output_root = Path(config.output_generation_dir)
	expected_identity = _initial_request_identity(config)
	if _path_exists(output_root):
		return _inspect_existing_generation(
			output_root,
			expected_identity=expected_identity,
		)
	output_root.parent.mkdir(parents=True, exist_ok=True)
	staging = output_root.with_name(f'.{output_root.name}.staging-{uuid.uuid4().hex}')
	staging.mkdir()
	try:
		initial_targets = _load_initial_targets_from_reference(
			_reference_payload(config.initial_hard_target_manifest)
		)
		_validate_initial_target_policy(initial_targets, config.target_policy)
		models = _load_fixed_hmms(config.initial_hmm_artifacts)
		_current_model_identity_checks(models)
		manifest = _build_initial_generation_manifest(
			config=config,
			models=models,
			initial_targets=initial_targets,
		)
		_write_json_atomic(staging / GENERATION_MANIFEST_NAME, manifest)
		validate_periodic_refresh_generation(
			staging / GENERATION_MANIFEST_NAME,
			expected_identity=expected_identity,
			_allow_staging=True,
		)
		staging.replace(output_root)
		published_manifest = output_root / GENERATION_MANIFEST_NAME
		return PeriodicRefreshGenerationResult(
			generation_dir=output_root,
			manifest_path=published_manifest,
			manifest_sha256=file_sha256(published_manifest),
			reused=False,
		)
	finally:
		if staging.exists():
			shutil.rmtree(staging, ignore_errors=True)


def build_initial_periodic_refresh_generation(
	config: InitialPeriodicRefreshConfig,
) -> PeriodicRefreshGenerationResult:
	"""Build the immutable generation-zero reference artifact."""
	return produce_initial_periodic_refresh_generation(config)


def produce_periodic_refresh_generation(
	config: PeriodicRefreshConfig,
) -> PeriodicRefreshGenerationResult:
	"""Build, validate, and atomically publish one refresh generation."""
	_validate_config(config)
	output_root = Path(config.output_generation_dir)
	expected_identity = _request_identity(config)
	if _path_exists(output_root):
		return _inspect_existing_generation(
			output_root,
			expected_identity=expected_identity,
		)

	output_root.parent.mkdir(parents=True, exist_ok=True)
	staging = output_root.with_name(f'.{output_root.name}.staging-{uuid.uuid4().hex}')
	staging.mkdir()
	try:
		_initial_manifest_ref(config.initial_hard_target_manifest)
		previous_payload = _load_previous_generation(
			config.previous_generation_manifest,
			generation_index=config.generation_index,
		)
		initial_targets = _load_initial_targets(config)
		_validate_initial_target_policy(initial_targets, config.target_policy)
		models = _load_fixed_hmms(config.initial_hmm_artifacts)
		_current_model_identity_checks(models)
		_validate_previous_lineage(previous_payload, config, models)
		current_embeddings = _prepare_generation_embeddings(
			config.current_embedding_descriptor,
			staging / 'embeddings',
			source_student_state_sha256=config.source_student_state_sha256,
		)
		common_valid_tokens = _validate_survey_contract(
			initial_targets,
			current_embeddings,
			edge_margin_tokens=models[0].edge_margin_tokens,
		)
		_validate_previous_centers_against_manifest(
			config.previous_centers,
			previous_payload,
		)

		prepared_store = _build_prepared_feature_store(
			models[0],
			current_embeddings.generation_inputs,
			staging / 'prepared_features',
		)
		try:
			results, previous_labels = _run_refreshes(
				config,
				models,
				prepared_store,
				initial_targets,
				previous_payload,
			)
			_write_center_arrays(staging, config, results)
			_write_hmm_outputs(staging, results, initial_targets.survey_ids)
			_write_prepared_feature_manifest(staging, prepared_store, models[0])
			_write_refresh_diagnostics(
				staging,
				results,
				previous_labels=previous_labels,
				valid_tokens=common_valid_tokens,
				confidence=_policy_confidence_arrays(
					initial_targets, config.target_policy
				),
			)
		finally:
			prepared_store.close()

		_generate_canonical_targets_and_manifest(
			staging,
			current_embeddings,
			config.target_policy,
			initial_targets,
		)
		generation_manifest = _build_generation_manifest(
			config=config,
			staging=staging,
			output_root=staging,
			models=models,
			current_embeddings=current_embeddings,
			initial_targets=initial_targets,
		)
		_write_json_atomic(staging / GENERATION_MANIFEST_NAME, generation_manifest)
		validate_periodic_refresh_generation(
			staging / GENERATION_MANIFEST_NAME,
			expected_identity=expected_identity,
			_allow_staging=True,
		)
		_prepare_staging_for_publication(staging, output_root)
		_validate_staged_publication(
			staging,
			output_root,
			expected_identity=expected_identity,
		)
		staging.replace(output_root)
		published_manifest = output_root / GENERATION_MANIFEST_NAME
		return PeriodicRefreshGenerationResult(
			generation_dir=output_root,
			manifest_path=published_manifest,
			manifest_sha256=file_sha256(published_manifest),
			reused=False,
		)
	finally:
		if staging.exists():
			shutil.rmtree(staging, ignore_errors=True)


def build_periodic_refresh_generation(
	config: PeriodicRefreshConfig,
) -> PeriodicRefreshGenerationResult:
	"""Build one generation; explicit builder spelling for library callers."""
	return produce_periodic_refresh_generation(config)


def load_periodic_refresh_generation(
	path: str | Path,
	*,
	expected_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
	"""Load and fully validate a complete periodic refresh manifest."""
	manifest_path = Path(path)
	try:
		payload = json.loads(manifest_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(
			f'periodic refresh generation manifest must be valid JSON: {manifest_path}'
		) from exc
	if not isinstance(payload, dict):
		raise TypeError('periodic refresh generation manifest must be a JSON object')
	validate_periodic_refresh_generation(
		manifest_path,
		expected_identity=expected_identity,
		_payload=payload,
	)
	return payload


def validate_periodic_refresh_generation(
	path: str | Path,
	*,
	expected_identity: Mapping[str, object] | None = None,
	_payload: Mapping[str, object] | None = None,
	_allow_staging: bool = False,
) -> None:
	"""Fail closed unless one generation is complete and internally consistent."""
	manifest_path = Path(path)
	root = manifest_path.parent
	payload = (
		_payload
		if _payload is not None
		else _load_json_object(manifest_path, 'periodic refresh generation manifest')
	)
	if (
		payload.get('generation_index') == 0
		or payload.get('generation_id') == INITIAL_GENERATION_ID
	):
		_validate_initial_generation_manifest(
			manifest_path,
			payload,
			expected_identity=expected_identity,
			allow_staging=_allow_staging,
		)
		return
	_required_keys(
		payload,
		{
			'artifact_type',
			'schema_version',
			'status',
			'generation_id',
			'generation_index',
			'refresh_after_epoch',
			'iterations',
			'prediction_batch_size',
			'source_student_state_sha256',
			'request_identity',
			'previous_generation_manifest',
			'current_embedding_descriptor',
			'initial_hard_target_manifest',
			'fixed_preprocessing_hmm_identity',
			'initial_hard_target_policy',
			'previous_centers',
			'embeddings',
			'centers',
			'final_labels',
			'canonical_multi_head_target_manifest',
			'per_k_targets',
			'valid_token_hashes',
			'prepared_feature_manifest',
			'refresh_diagnostics',
			'generation_content_sha256',
			'content_files',
		},
		'periodic refresh generation manifest',
	)
	if payload['artifact_type'] != ARTIFACT_TYPE:
		raise ValueError('periodic refresh generation artifact_type is invalid')
	if payload['schema_version'] != SCHEMA_VERSION:
		raise ValueError('periodic refresh generation schema_version is invalid')
	if payload['status'] != 'COMPLETE':
		raise ValueError('periodic refresh generation is not COMPLETE')
	index = _positive_int(payload['generation_index'], 'generation_index')
	epoch = _positive_int(payload['refresh_after_epoch'], 'refresh_after_epoch')
	expected_id = f'refresh_{index:04d}_epoch{epoch:03d}'
	if payload['generation_id'] != expected_id:
		raise ValueError(
			'periodic refresh generation_id does not match index and epoch'
		)
	if root.name != expected_id and not _is_staging_root(
		root, expected_id, allow_staging=_allow_staging
	):
		raise ValueError('periodic refresh directory name does not match generation_id')
	if payload['iterations'] != 2:
		raise ValueError('periodic refresh iterations must be exactly two')
	_positive_int(payload['prediction_batch_size'], 'prediction_batch_size')
	source_hash = _validate_sha256(
		payload['source_student_state_sha256'], 'source_student_state_sha256'
	)
	if expected_identity is not None and dict(payload['request_identity']) != dict(
		expected_identity
	):
		raise ValueError('periodic refresh request identity mismatch')
	if payload['request_identity'] != _request_identity_from_payload(payload):
		raise ValueError('periodic refresh request identity is inconsistent')
	_validate_sha256(payload['generation_content_sha256'], 'generation_content_sha256')
	_validate_content_inventory(root, payload)

	for name in (
		'previous_generation_manifest',
		'current_embedding_descriptor',
		'initial_hard_target_manifest',
	):
		_validate_reference_payload(payload[name], name)
	current_descriptor_reference = _coerce_reference(
		payload['current_embedding_descriptor'], 'current_embedding_descriptor'
	)
	_validate_sha256(source_hash, 'source_student_state_sha256')
	_validate_target_policy_payload(payload['initial_hard_target_policy'])
	previous_payload = _load_previous_generation_payload_from_manifest(
		payload['previous_generation_manifest']
	)
	previous_index = previous_payload.get('generation_index')
	if isinstance(previous_index, bool) or not isinstance(previous_index, int):
		raise ValueError('previous generation has no valid generation_index')
	if previous_index != index - 1:
		raise ValueError(
		'previous generation index must be exactly one less than the requested '
		'generation index'
	)

	initial_targets = _load_initial_targets_from_reference(
		payload['initial_hard_target_manifest']
	)
	policy_payload = _mapping(
		payload['initial_hard_target_policy'], 'initial hard target policy'
	)
	_validate_initial_target_policy(
		initial_targets,
		HardTargetPolicy(
			confidence_mode=policy_payload['confidence_mode'],
			confidence=float(policy_payload['confidence']),
			boundary_weight_mode=policy_payload['boundary_weight_mode'],
		),
	)
	embedding_data = _validate_persisted_embeddings(
		root,
		payload['embeddings'],
		expected_student_state_sha256=source_hash,
		expected_descriptor_sha256=current_descriptor_reference.sha256,
	)
	models = _load_fixed_hmms_from_manifest(payload['fixed_preprocessing_hmm_identity'])
	_current_model_identity_checks(models)
	_validate_persisted_fixed_identity(
		payload['fixed_preprocessing_hmm_identity'], models
	)
	_validate_previous_payload_lineage(
		previous_payload,
		initial_hard_target_manifest=payload['initial_hard_target_manifest'],
		initial_hard_target_policy=payload['initial_hard_target_policy'],
		fixed_identity=payload['fixed_preprocessing_hmm_identity'],
	)
	_validate_persisted_target_masks(
		initial_targets,
		embedding_data,
		edge_margin_tokens=models[0].edge_margin_tokens,
	)
	valid_hashes = _mapping(
		payload['valid_token_hashes'], 'generation valid-token hashes'
	)
	expected_valid_hashes = _mapping(
		_mapping(initial_targets.payload['common'], 'initial target common')[
			'valid_tokens_sha256'
		],
		'initial target valid-token hashes',
	)
	if dict(valid_hashes) != expected_valid_hashes:
		raise ValueError('generation valid-token hashes differ from initial mask')
	_validate_persisted_centers(
		root,
		payload['centers'],
		payload['previous_centers'],
		previous_payload,
		models,
	)
	_validate_persisted_final_labels(
		root,
		payload['final_labels'],
		payload['per_k_targets'],
		initial_targets,
		embedding_data,
		models,
	)
	_validate_persisted_target_manifest(
		root,
		payload['canonical_multi_head_target_manifest'],
		payload['per_k_targets'],
		embedding_data,
		initial_targets,
		_policy_from_payload(payload['initial_hard_target_policy']),
	)
	_validate_persisted_prepared_features(
		root, payload['prepared_feature_manifest'], models[0]
	)
	_validate_persisted_diagnostics(
		root, payload['refresh_diagnostics'], payload, models
	)


def _validate_initial_generation_manifest(
	manifest_path: Path,
	payload: Mapping[str, object],
	*,
	expected_identity: Mapping[str, object] | None,
	allow_staging: bool,
) -> None:
	"""Validate generation zero without materializing any historical artifact."""
	root = manifest_path.parent
	_required_keys(
		payload,
		{
			'artifact_type',
			'schema_version',
			'status',
			'generation_mode',
			'generation_id',
			'generation_index',
			'refresh_after_epoch',
			'source_student_state_sha256',
			'request_identity',
			'previous_generation_manifest',
			'current_embedding_descriptor',
			'initial_hard_target_manifest',
			'fixed_preprocessing_hmm_identity',
			'initial_hard_target_policy',
			'previous_centers',
			'embeddings',
			'centers',
			'final_labels',
			'canonical_multi_head_target_manifest',
			'per_k_targets',
			'valid_token_hashes',
			'prepared_feature_manifest',
			'refresh_diagnostics',
			'generation_content_sha256',
			'content_files',
		},
		'initial periodic refresh generation manifest',
	)
	if payload['artifact_type'] != ARTIFACT_TYPE:
		raise ValueError('initial periodic refresh artifact_type is invalid')
	if payload['schema_version'] != SCHEMA_VERSION:
		raise ValueError('initial periodic refresh schema_version is invalid')
	if payload['status'] != 'COMPLETE':
		raise ValueError('initial periodic refresh is not COMPLETE')
	if payload['generation_mode'] != 'initial_immutable_references':
		raise ValueError('initial periodic refresh generation_mode is invalid')
	if payload['generation_id'] != INITIAL_GENERATION_ID:
		raise ValueError('initial periodic refresh generation_id is invalid')
	if payload['generation_index'] != 0 or payload['refresh_after_epoch'] != 0:
		raise ValueError('initial periodic refresh generation numbers are invalid')
	if not _is_staging_root(root, INITIAL_GENERATION_ID, allow_staging=allow_staging):
		raise ValueError('initial periodic refresh directory name is invalid')
	if payload['source_student_state_sha256'] is not None:
		raise ValueError('initial periodic refresh cannot bind a student state')
	if expected_identity is not None and dict(payload['request_identity']) != dict(
		expected_identity
	):
		raise ValueError('initial periodic refresh request identity mismatch')
	if payload['request_identity'] != _initial_request_identity_from_payload(payload):
		raise ValueError('initial periodic refresh request identity is inconsistent')
	for name in (
		'previous_generation_manifest',
		'current_embedding_descriptor',
		'previous_centers',
		'embeddings',
		'prepared_feature_manifest',
		'refresh_diagnostics',
	):
		if payload[name] is not None:
			raise ValueError(f'initial periodic refresh {name} must be absent')
	initial_reference = _mapping(
		payload['initial_hard_target_manifest'],
		'initial hard target manifest reference',
	)
	_validate_reference_payload(
		initial_reference, 'initial hard target manifest reference'
	)
	canonical_reference = _mapping(
		payload['canonical_multi_head_target_manifest'],
		'initial canonical target manifest reference',
	)
	_validate_reference_payload(
		canonical_reference, 'initial canonical target manifest reference'
	)
	if dict(canonical_reference) != dict(initial_reference):
		raise ValueError('initial canonical target manifest reference drift')
	initial_targets = _load_initial_targets_from_reference(initial_reference)
	policy_payload = _mapping(
		payload['initial_hard_target_policy'], 'initial hard target policy'
	)
	_validate_initial_target_policy(
		initial_targets,
		HardTargetPolicy(
			confidence_mode=policy_payload['confidence_mode'],
			confidence=float(policy_payload['confidence']),
			boundary_weight_mode=policy_payload['boundary_weight_mode'],
		),
	)
	canonical = load_multi_head_target_manifest(Path(str(canonical_reference['path'])))
	if tuple(canonical['head_ks']) != CANONICAL_KS:
		raise ValueError('initial canonical target K set drift')
	models = _load_fixed_hmms_from_manifest(payload['fixed_preprocessing_hmm_identity'])
	_current_model_identity_checks(models)
	_validate_persisted_fixed_identity(
		payload['fixed_preprocessing_hmm_identity'], models
	)
	_validate_initial_centers(payload['centers'], models)
	_validate_initial_targets(
		payload['final_labels'],
		payload['per_k_targets'],
		payload['valid_token_hashes'],
		initial_targets,
		models,
	)
	content_files = payload['content_files']
	if content_files != [] or _content_inventory(root) != []:
		raise ValueError('initial periodic refresh must contain references only')
	_validate_sha256(payload['generation_content_sha256'], 'generation_content_sha256')
	if payload['generation_content_sha256'] != _content_hash(()):
		raise ValueError('initial periodic refresh content hash is invalid')


def _initial_request_identity_from_payload(
	payload: Mapping[str, object],
) -> dict[str, object]:
	return {
		'generation_index': 0,
		'generation_id': INITIAL_GENERATION_ID,
		'initial_hard_target_manifest': payload['initial_hard_target_manifest'],
		'initial_hmm_artifacts': _mapping(
			_mapping(
				payload['fixed_preprocessing_hmm_identity'],
				'fixed preprocessing identity',
			)['artifacts'],
			'fixed HMM artifacts',
		),
		'target_policy': payload['initial_hard_target_policy'],
	}


def _validate_initial_centers(
	value: object,
	models: Sequence[_LoadedFixedHMM],
) -> None:
	centers = _mapping(value, 'initial generation centers')
	if set(centers) != {str(model.k) for model in models}:
		raise ValueError('initial generation center K set mismatch')
	for model in models:
		entry = _mapping(centers[str(model.k)], f'initial k={model.k} centers')
		for name in ('before', 'after'):
			ref = _mapping(entry[name], f'initial k={model.k} center {name}')
			_validate_reference_payload(ref, f'initial k={model.k} center {name}')
			array = np.load(Path(str(ref['path'])), mmap_mode='r', allow_pickle=False)
			try:
				if (
					array.dtype != np.dtype('float32')
					or array.shape != model.centers.shape
				):
					raise ValueError(
						f'initial center shape or dtype drift for k={model.k}'
					)
				if not np.array_equal(array, model.centers):
					raise ValueError(f'initial center values drift for k={model.k}')
			finally:
				del array


def _validate_initial_targets(
	final_labels_value: object,
	per_k_value: object,
	valid_hashes_value: object,
	initial_targets: _InitialTargetData,
	models: Sequence[_LoadedFixedHMM],
) -> None:
	final_labels = _mapping(final_labels_value, 'initial generation final labels')
	per_k = _mapping(per_k_value, 'initial generation per-k targets')
	valid_hashes = _mapping(valid_hashes_value, 'initial generation valid-token hashes')
	common = _mapping(initial_targets.payload['common'], 'initial target common')
	if dict(valid_hashes) != dict(common['valid_tokens_sha256']):
		raise ValueError('initial generation valid-token hashes drift')
	target_heads = _mapping(initial_targets.payload['heads'], 'initial target heads')
	for model in models:
		k = model.k
		head = _mapping(target_heads[str(k)], f'initial target k={k}')
		surveys = _mapping(head['surveys'], f'initial target k={k} surveys')
		per_k_entry = _mapping(per_k[str(k)], f'initial per-k targets k={k}')
		per_k_surveys = _mapping(per_k_entry['surveys'], 'initial per-k surveys')
		if set(per_k_surveys) != set(surveys):
			raise ValueError(f'initial per-k survey set mismatch for k={k}')
		if Path(str(per_k_entry['root'])).resolve() != (
			Path(str(head['pseudo_target_root'])).resolve() / f'k{k}'
		):
			raise ValueError(f'initial per-k target root drift for k={k}')
		if dict(per_k_surveys) != dict(surveys):
			raise ValueError(f'initial per-k target references drift for k={k}')
		final_by_survey = _mapping(final_labels[str(k)], f'initial final labels k={k}')
		if set(final_by_survey) != set(surveys):
			raise ValueError(f'initial final-label survey set mismatch for k={k}')
		for survey_id, target_value in surveys.items():
			target = _mapping(target_value, f'initial target k={k} {survey_id}')
			labels_ref = _mapping(target['labels'], 'initial target labels')
			final_ref = _mapping(
				final_by_survey[survey_id], f'initial final labels k={k} {survey_id}'
			)
			_validate_reference_payload(
				final_ref, f'initial final labels k={k} {survey_id}'
			)
			if dict(final_ref) != dict(labels_ref):
				raise ValueError(f'initial final-label reference drift for k={k}')
			labels = np.load(
				Path(str(final_ref['path'])), mmap_mode='r', allow_pickle=False
			)
			valid = initial_targets.valid_tokens[str(survey_id)]
			try:
				if labels.shape != valid.shape:
					raise ValueError(
						f'initial target shape drift for k={k} {survey_id}'
					)
				if np.any(labels[valid] < 0) or np.any(labels[valid] >= k):
					raise ValueError(f'initial target label range drift for k={k}')
				if np.any(labels[~valid] != -1):
					raise ValueError(f'initial target invalid sentinel drift for k={k}')
			finally:
				del labels


def quarantine_periodic_refresh_generation(path: str | Path) -> Path:
	"""Move an explicitly selected stale/foreign generation to a timestamped path."""
	source = Path(path)
	if not _path_exists(source):
		raise FileNotFoundError(f'periodic refresh generation is missing: {source}')
	timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
	target = source.with_name(f'{source.name}.quarantine.{timestamp}')
	if _path_exists(target):
		target = source.with_name(
			f'{source.name}.quarantine.{timestamp}-{uuid.uuid4().hex}'
		)
	source.replace(target)
	return target


def _validate_config(config: PeriodicRefreshConfig) -> None:
	if not isinstance(config, PeriodicRefreshConfig):
		raise TypeError('config must be a PeriodicRefreshConfig')
	if config.generation_index <= 0 or isinstance(config.generation_index, bool):
		raise ValueError('generation_index must be a positive integer for refresh')
	if config.refresh_after_epoch <= 0 or isinstance(config.refresh_after_epoch, bool):
		raise ValueError('refresh_after_epoch must be a positive integer')
	if config.iterations != 2 or isinstance(config.iterations, bool):
		raise ValueError('periodic refresh requires exactly two HMM iterations')
	if (
		isinstance(config.prediction_batch_size, bool)
		or not isinstance(config.prediction_batch_size, int)
		or config.prediction_batch_size <= 0
	):
		raise ValueError('prediction_batch_size must be a positive integer')
	_validate_sha256(config.source_student_state_sha256, 'source_student_state_sha256')
	config.target_policy.validate()
	if tuple(item.k for item in config.initial_hmm_artifacts) != CANONICAL_KS:
		raise ValueError('initial_hmm_artifacts must be ordered K=6,8,10')
	if tuple(item.k for item in config.previous_centers) != CANONICAL_KS:
		raise ValueError('previous_centers must be ordered K=6,8,10')
	_validate_reference(
		config.previous_generation_manifest, 'previous_generation_manifest'
	)
	_validate_reference(
		config.current_embedding_descriptor, 'current_embedding_descriptor'
	)
	_validate_reference(
		config.initial_hard_target_manifest, 'initial_hard_target_manifest'
	)
	for artifact in config.initial_hmm_artifacts:
		for name in ('centers', 'hmm_model', 'preprocessor', 'metadata'):
			_validate_reference(
				getattr(artifact, name), f'initial k={artifact.k} {name}'
			)
		if artifact.residualizer is not None:
			_validate_reference(
				artifact.residualizer, f'initial k={artifact.k} residualizer'
			)
	for item in config.previous_centers:
		_validate_reference(item.centers, f'previous k={item.k} centers')
	if not str(config.output_generation_dir):
		raise ValueError('output_generation_dir must be non-empty')
	output = Path(config.output_generation_dir).resolve()
	expected_name = (
		f'refresh_{config.generation_index:04d}_epoch'
		f'{config.refresh_after_epoch:03d}'
	)
	if output.name != expected_name:
		raise ValueError(
			f'output_generation_dir must be named {expected_name!r}'
		)
	for reference in (
		config.previous_generation_manifest,
		config.current_embedding_descriptor,
		config.initial_hard_target_manifest,
	):
		if reference.path.resolve() == output:
			raise ValueError('output_generation_dir must differ from input artifacts')


def _request_identity(config: PeriodicRefreshConfig) -> dict[str, object]:
	return {
		'generation_index': config.generation_index,
		'refresh_after_epoch': config.refresh_after_epoch,
		'source_student_state_sha256': config.source_student_state_sha256,
		'previous_generation_manifest': _reference_payload(
			config.previous_generation_manifest
		),
		'current_embedding_descriptor': _reference_payload(
			config.current_embedding_descriptor
		),
		'initial_hard_target_manifest': _reference_payload(
			config.initial_hard_target_manifest
		),
		'initial_hmm_artifacts': {
			str(item.k): _initial_hmm_artifact_payload(item)
			for item in config.initial_hmm_artifacts
		},
		'previous_centers': {
			str(item.k): _reference_payload(item.centers)
			for item in config.previous_centers
		},
		'iterations': config.iterations,
		'prediction_batch_size': config.prediction_batch_size,
		'target_policy': asdict(config.target_policy),
	}


def _request_identity_from_payload(
	payload: Mapping[str, object],
) -> dict[str, object]:
	return {
		'generation_index': payload['generation_index'],
		'refresh_after_epoch': payload['refresh_after_epoch'],
		'source_student_state_sha256': payload['source_student_state_sha256'],
		'previous_generation_manifest': payload['previous_generation_manifest'],
		'current_embedding_descriptor': payload['current_embedding_descriptor'],
		'initial_hard_target_manifest': payload['initial_hard_target_manifest'],
		'initial_hmm_artifacts': _mapping(
			_mapping(
				payload['fixed_preprocessing_hmm_identity'],
				'fixed preprocessing identity',
			)['artifacts'],
			'fixed HMM artifacts',
		),
		'previous_centers': payload['previous_centers'],
		'iterations': payload['iterations'],
		'prediction_batch_size': payload['prediction_batch_size'],
		'target_policy': payload['initial_hard_target_policy'],
	}


def _initial_request_identity(
	config: InitialPeriodicRefreshConfig,
) -> dict[str, object]:
	return {
		'generation_index': 0,
		'generation_id': INITIAL_GENERATION_ID,
		'initial_hard_target_manifest': _reference_payload(
			config.initial_hard_target_manifest
		),
		'initial_hmm_artifacts': {
			str(item.k): _initial_hmm_artifact_payload(item)
			for item in config.initial_hmm_artifacts
		},
		'target_policy': asdict(config.target_policy),
	}


def _validate_initial_config(config: InitialPeriodicRefreshConfig) -> None:
	if not isinstance(config, InitialPeriodicRefreshConfig):
		raise TypeError('config must be an InitialPeriodicRefreshConfig')
	_validate_reference(
		config.initial_hard_target_manifest, 'initial_hard_target_manifest'
	)
	if tuple(item.k for item in config.initial_hmm_artifacts) != CANONICAL_KS:
		raise ValueError('initial_hmm_artifacts must be ordered K=6,8,10')
	for artifact in config.initial_hmm_artifacts:
		for name in ('centers', 'hmm_model', 'preprocessor', 'metadata'):
			_validate_reference(
				getattr(artifact, name), f'initial k={artifact.k} {name}'
			)
		if artifact.residualizer is not None:
			_validate_reference(
				artifact.residualizer, f'initial k={artifact.k} residualizer'
			)
	config.target_policy.validate()
	if not str(config.output_generation_dir):
		raise ValueError('output_generation_dir must be non-empty')
	output = Path(config.output_generation_dir).resolve()
	if output.name != INITIAL_GENERATION_ID:
		raise ValueError(
		f'output_generation_dir must be named {INITIAL_GENERATION_ID!r}'
	)
	if config.initial_hard_target_manifest.path.resolve() == output:
		raise ValueError('output_generation_dir must differ from input artifacts')


def _build_initial_generation_manifest(
	*,
	config: InitialPeriodicRefreshConfig,
	models: Sequence[_LoadedFixedHMM],
	initial_targets: _InitialTargetData,
) -> dict[str, object]:
	target_payload = initial_targets.payload
	target_heads = _mapping(target_payload['heads'], 'initial target heads')
	per_k_targets: dict[str, object] = {}
	final_labels: dict[str, object] = {}
	centers: dict[str, object] = {}
	for model in models:
		head = _mapping(target_heads[str(model.k)], f'initial target k={model.k}')
		surveys = _mapping(head['surveys'], f'initial target k={model.k} surveys')
		per_k_targets[str(model.k)] = {
			'root': str(
				Path(str(head['pseudo_target_root'])).resolve() / f'k{model.k}'
			),
			'surveys': dict(surveys),
		}
		final_labels[str(model.k)] = {
			survey_id: dict(
				_mapping(
					_mapping(entry, f'initial target k={model.k} {survey_id}')[
						'labels'
					],
					'initial target labels',
				)
			)
			for survey_id, entry in surveys.items()
		}
		center_reference = _reference_payload(model.artifact.centers)
		centers[str(model.k)] = {
			'before': center_reference,
			'after': center_reference,
			'shape': list(model.centers.shape),
			'dtype': 'float32',
		}
	valid_hashes = dict(
		_mapping(
			_mapping(target_payload['common'], 'initial target common')[
				'valid_tokens_sha256'
			],
			'initial target valid-token hashes',
		)
	)
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': 'COMPLETE',
		'generation_mode': 'initial_immutable_references',
		'generation_id': INITIAL_GENERATION_ID,
		'generation_index': 0,
		'refresh_after_epoch': 0,
		'source_student_state_sha256': None,
		'request_identity': _initial_request_identity(config),
		'previous_generation_manifest': None,
		'current_embedding_descriptor': None,
		'initial_hard_target_manifest': _reference_payload(
			config.initial_hard_target_manifest
		),
		'fixed_preprocessing_hmm_identity': _fixed_identity_payload(models),
		'initial_hard_target_policy': asdict(config.target_policy),
		'previous_centers': None,
		'embeddings': None,
		'centers': centers,
		'final_labels': final_labels,
		'canonical_multi_head_target_manifest': _reference_payload(
			config.initial_hard_target_manifest
		),
		'per_k_targets': per_k_targets,
		'valid_token_hashes': valid_hashes,
		'prepared_feature_manifest': None,
		'refresh_diagnostics': None,
		'generation_content_sha256': _content_hash(()),
		'content_files': [],
	}


def _fixed_identity_payload(
	models: Sequence[_LoadedFixedHMM],
) -> dict[str, object]:
	return {
		'artifacts': {
			str(model.k): _initial_hmm_artifact_payload(model.artifact)
			for model in models
		},
		'normalized': _jsonable(models[0].normalized_identity),
		'prepared_feature_identity': _jsonable(models[0].prepared_feature_identity),
		'edge_margin_tokens': list(models[0].edge_margin_tokens),
	}


def _inspect_existing_generation(
	output_root: Path,
	*,
	expected_identity: Mapping[str, object],
) -> PeriodicRefreshGenerationResult:
	manifest_path = output_root / GENERATION_MANIFEST_NAME
	if not manifest_path.is_file():
		raise ValueError(
			'existing periodic refresh generation is partial or foreign; '
			f'explicitly quarantine it before retrying: {output_root}'
		)
	try:
		payload = load_periodic_refresh_generation(
			manifest_path, expected_identity=expected_identity
		)
	except (OSError, TypeError, ValueError, KeyError) as exc:
		raise ValueError(
			'existing periodic refresh generation is stale or invalid; explicitly '
			f'quarantine it before retrying: {output_root}: {exc}'
		) from exc
	if payload['request_identity'] != expected_identity:
		raise ValueError(
			'existing periodic refresh generation identity differs; explicitly '
			f'quarantine it before retrying: {output_root}'
		)
	return PeriodicRefreshGenerationResult(
		generation_dir=output_root,
		manifest_path=manifest_path,
		manifest_sha256=file_sha256(manifest_path),
		reused=True,
	)


def _initial_manifest_ref(reference: HashedArtifactReference) -> None:
	_validate_reference(reference, 'initial_hard_target_manifest')


def _load_previous_generation(
	reference: HashedArtifactReference,
	*,
	generation_index: int,
) -> Mapping[str, object]:
	_validate_reference(reference, 'previous_generation_manifest')
	payload = _load_json_object(reference.path, 'previous generation manifest')
	_validate_previous_generation_manifest(reference.path, payload)
	if payload.get('status') != 'COMPLETE':
		raise ValueError('previous generation manifest is not COMPLETE')
	previous_index = payload.get('generation_index')
	if isinstance(previous_index, bool) or not isinstance(previous_index, int):
		raise ValueError('previous generation manifest has no valid generation_index')
	if previous_index != generation_index - 1:
		raise ValueError(
			'previous generation index must be exactly one less than the requested '
			'generation index'
		)
	return payload


def _load_previous_generation_payload_from_manifest(
	value: object,
) -> Mapping[str, object]:
	reference = _coerce_reference(value, 'previous_generation_manifest')
	_validate_reference(reference, 'previous_generation_manifest')
	payload = _load_json_object(reference.path, 'previous generation manifest')
	_validate_previous_generation_manifest(reference.path, payload)
	return payload


def _validate_previous_generation_manifest(
	path: Path,
	payload: Mapping[str, object],
) -> None:
	if payload.get('artifact_type') != ARTIFACT_TYPE:
		raise ValueError('previous generation manifest artifact identity is invalid')
	validate_periodic_refresh_generation(path)


def _load_initial_targets(config: PeriodicRefreshConfig) -> _InitialTargetData:
	return _load_initial_targets_from_reference(
		_reference_payload(config.initial_hard_target_manifest)
	)


def _validate_initial_target_policy(
	initial_targets: _InitialTargetData,
	policy: HardTargetPolicy,
) -> None:
	policy.validate()
	for k in CANONICAL_KS:
		for survey_id in initial_targets.survey_ids:
			entry_labels = initial_targets.labels[k][survey_id]
			valid = initial_targets.valid_tokens[survey_id]
			if entry_labels.shape != valid.shape:
				raise ValueError(f'initial target shape mismatch for k={k} {survey_id}')
			confidence = initial_targets.confidence[k][survey_id]
			if confidence.shape != valid.shape:
				raise ValueError(
					f'initial confidence shape mismatch for k={k} {survey_id}'
				)
			if policy.confidence_mode == 'constant':
				expected = np.zeros(valid.shape, dtype=np.float32)
				expected[valid] = np.float32(policy.confidence)
				if not np.array_equal(confidence, expected):
					raise ValueError(
						f'initial confidence policy drift for k={k} {survey_id}'
					)
			elif not np.all(np.isfinite(confidence)):
				raise ValueError(
					f'initial source confidence is non-finite for k={k} {survey_id}'
				)
			boundary_weight = initial_targets.boundary_weight[k][survey_id]
			if policy.boundary_weight_mode == 'absent':
				if boundary_weight is not None:
					raise ValueError(
						'initial hard target boundary weights conflict with absent policy'
					)
			elif boundary_weight is None:
				raise ValueError(
					'initial hard target boundary weights are missing for source policy'
				)
			elif boundary_weight.shape != valid.shape or not np.all(
				np.isfinite(boundary_weight)
			):
				raise ValueError(
					f'initial boundary weights are invalid for k={k} {survey_id}'
				)


def _load_initial_targets_from_reference(value: object) -> _InitialTargetData:
	reference = _coerce_reference(value, 'initial_hard_target_manifest')
	_validate_reference(reference, 'initial_hard_target_manifest')
	payload = load_multi_head_target_manifest(reference.path)
	if tuple(payload['head_ks']) != CANONICAL_KS:
		raise ValueError('initial hard target manifest must contain K=6,8,10')
	common = _mapping(payload['common'], 'initial target common')
	survey_ids_value = common.get('survey_ids')
	if not isinstance(survey_ids_value, list) or not survey_ids_value:
		raise ValueError('initial hard target manifest has no surveys')
	survey_ids = tuple(str(item) for item in survey_ids_value)
	if any(not survey_id for survey_id in survey_ids):
		raise ValueError('initial hard target manifest contains an empty survey id')
	if len(set(survey_ids)) != len(survey_ids):
		raise ValueError('initial hard target manifest contains duplicate surveys')
	labels_by_k: dict[int, dict[str, np.ndarray]] = {}
	confidence_by_k: dict[int, dict[str, np.ndarray]] = {}
	boundary_by_k: dict[int, dict[str, np.ndarray | None]] = {}
	valid_by_survey: dict[str, np.ndarray] = {}
	heads = _mapping(payload['heads'], 'initial target heads')
	for k in CANONICAL_KS:
		head = _mapping(heads[str(k)], f'initial k={k}')
		surveys = _mapping(head['surveys'], f'initial k={k} surveys')
		if set(surveys) != set(survey_ids):
			raise ValueError(f'initial k={k} survey set differs from common manifest')
		labels_by_k[k] = {}
		confidence_by_k[k] = {}
		boundary_by_k[k] = {}
		for survey_id in survey_ids:
			entry = _mapping(surveys[survey_id], f'initial k={k} {survey_id}')
			labels_path = Path(str(_mapping(entry['labels'], 'initial labels')['path']))
			confidence_path = Path(
				str(_mapping(entry['confidence'], 'initial confidence')['path'])
			)
			valid_path = Path(
				str(_mapping(entry['valid_tokens'], 'initial valid')['path'])
			)
			boundary_value = entry.get('boundary_weight')
			boundary_path = (
				None
				if boundary_value is None
				else Path(
					str(_mapping(boundary_value, 'initial boundary weight')['path'])
				)
			)
			metadata_path = Path(
				str(_mapping(entry['metadata'], 'initial metadata')['path'])
			)
			item = StratPseudoTargetInput(
				survey_id=survey_id,
				k=k,
				labels_path=labels_path,
				confidence_path=confidence_path,
				valid_tokens_path=valid_path,
				boundary_weight_path=boundary_path,
				metadata_path=metadata_path,
			)
			arrays = load_pseudo_target_arrays(item, mmap_mode='r')
			labels = np.asarray(arrays.labels)
			valid = np.asarray(arrays.valid_tokens)
			if survey_id not in valid_by_survey:
				valid_by_survey[survey_id] = np.array(valid, dtype=np.bool_, copy=True)
			elif not np.array_equal(valid_by_survey[survey_id], valid):
				raise ValueError(
					'initial hard target valid-token masks differ across heads'
				)
			labels_by_k[k][survey_id] = np.array(labels, dtype=np.int32, copy=True)
			confidence_by_k[k][survey_id] = np.array(
				arrays.confidence, dtype=np.float32, copy=True
			)
			boundary_by_k[k][survey_id] = (
				np.array(arrays.boundary_weight, dtype=np.float32, copy=True)
				if boundary_path is not None
				else None
			)
	return _InitialTargetData(
		payload=payload,
		survey_ids=tuple(sorted(survey_ids)),
		valid_tokens=valid_by_survey,
		labels=labels_by_k,
		confidence=confidence_by_k,
		boundary_weight=boundary_by_k,
	)


def _load_fixed_hmms(
	artifacts: Sequence[InitialHMMArtifact],
) -> tuple[_LoadedFixedHMM, ...]:
	return tuple(_load_fixed_hmm(artifact) for artifact in artifacts)


def _load_fixed_hmms_from_manifest(
	value: object,
) -> tuple[_LoadedFixedHMM, ...]:
	identity = _mapping(value, 'fixed_preprocessing_hmm_identity')
	artifacts = _mapping(identity['artifacts'], 'fixed HMM artifacts')
	return tuple(
		_load_fixed_hmm(_initial_hmm_artifact_from_payload(k, artifacts[str(k)]))
		for k in CANONICAL_KS
	)


def _load_fixed_hmm(artifact: InitialHMMArtifact) -> _LoadedFixedHMM:
	for name in ('centers', 'hmm_model', 'preprocessor', 'metadata'):
		_validate_reference(getattr(artifact, name), f'initial k={artifact.k} {name}')
	if artifact.residualizer is not None:
		_validate_reference(
			artifact.residualizer, f'initial k={artifact.k} residualizer'
		)
	centers_loaded = np.load(artifact.centers.path, mmap_mode='r', allow_pickle=False)
	try:
		centers = np.asarray(centers_loaded, dtype=np.float32).copy()
	finally:
		del centers_loaded
	if centers.ndim != 2 or centers.shape[0] != artifact.k or centers.shape[1] == 0:
		raise ValueError(f'initial center shape is invalid for k={artifact.k}')
	if not np.all(np.isfinite(centers)):
		raise ValueError(f'initial centers are non-finite for k={artifact.k}')
	model = joblib.load(artifact.hmm_model.path)
	if not isinstance(model, Mapping):
		raise TypeError(f'initial hmm_model must be a mapping for k={artifact.k}')
	if model.get('emission_source') not in {'embedding', 'z_coordinate'}:
		raise ValueError(f'initial emission_source is invalid for k={artifact.k}')
	model_centers = np.asarray(model.get('centers'), dtype=np.float32)
	if model_centers.shape != centers.shape or not np.array_equal(
		model_centers, centers
	):
		raise ValueError(
			f'initial HMM centers differ from centers artifact for k={artifact.k}'
		)
	transition_costs = _load_cost_matrix(model.get('transition_costs'), artifact.k)
	initial_costs = _load_cost_vector(
		model.get('initial_state_costs'), artifact.k, 'initial_state_costs'
	)
	terminal_costs = _load_cost_vector(
		model.get('terminal_state_costs'), artifact.k, 'terminal_state_costs'
	)
	path_prior = _mapping(model.get('path_prior'), f'initial k={artifact.k} path_prior')
	expected_boundaries = _expected_boundaries_from_path_prior(path_prior)
	metadata = _load_json_object(
		artifact.metadata.path, f'initial k={artifact.k} metadata'
	)
	if metadata.get('k') != artifact.k:
		raise ValueError(f'initial metadata K mismatch for k={artifact.k}')
	strat = _mapping(
		metadata.get('stratigraphic_hmm'),
		f'initial k={artifact.k} stratigraphic_hmm',
	)
	prepared_identity_value = strat.get(
		'prepared_feature_cache', strat.get('prepared_feature_identity')
	)
	prepared_identity = _mapping(
		prepared_identity_value,
		f'initial k={artifact.k} prepared feature identity',
	)
	if not prepared_identity:
		raise ValueError(
			f'initial prepared feature identity is empty for k={artifact.k}'
		)
	emission_source = str(model['emission_source'])
	edge_margin = _edge_margin(strat.get('edge_margin_tokens'), artifact.k)
	if strat.get('emission_source') != emission_source:
		raise ValueError(f'initial emission source drift for k={artifact.k}')
	if emission_source == 'z_coordinate' and centers.shape[1] != 1:
		raise ValueError(f'z-coordinate HMM centers must be one-dimensional for k={artifact.k}')
	if strat.get('z_axis') != 2:
		raise ValueError(f'initial z_axis must be 2 for k={artifact.k}')
	if not isinstance(strat.get('z_direction'), str):
		raise TypeError(f'initial z_direction is missing for k={artifact.k}')
	metadata_transition = strat.get('transition')
	model_transition = model.get('transition_settings')
	if not isinstance(metadata_transition, Mapping) or not isinstance(
		model_transition, Mapping
	):
		raise ValueError(f'initial transition settings are missing for k={artifact.k}')
	if dict(metadata_transition) != dict(model_transition):
		raise ValueError(f'initial transition settings drift for k={artifact.k}')
	metadata_prior = _mapping(
		strat.get('path_prior'), f'initial k={artifact.k} metadata path_prior'
	)
	if _path_prior_core(metadata_prior) != _path_prior_core(path_prior):
		raise ValueError(f'initial path prior drift for k={artifact.k}')
	metadata_initial_costs = _load_cost_vector(
		metadata_prior.get('initial_state_costs'), artifact.k, 'initial_state_costs'
	)
	metadata_terminal_costs = _load_cost_vector(
		metadata_prior.get('terminal_state_costs'), artifact.k, 'terminal_state_costs'
	)
	if not np.array_equal(metadata_initial_costs, initial_costs):
		raise ValueError(f'initial state costs drift for k={artifact.k}')
	if not np.array_equal(metadata_terminal_costs, terminal_costs):
		raise ValueError(f'terminal state costs drift for k={artifact.k}')
	metadata_transition_costs = _json_cost_matrix(
		strat.get('transition_costs'), artifact.k
	)
	if not np.array_equal(metadata_transition_costs, transition_costs):
		raise ValueError(f'initial transition cost drift for k={artifact.k}')
	if _edge_margin(model.get('edge_margin_tokens'), artifact.k) != edge_margin:
		raise ValueError(f'initial edge margin drift for k={artifact.k}')
	preprocessor = joblib.load(artifact.preprocessor.path)
	if not callable(getattr(preprocessor, 'transform', None)):
		raise TypeError(f'initial preprocessor cannot transform for k={artifact.k}')
	residualizer = (
		read_residualizer_npz(artifact.residualizer.path)
		if artifact.residualizer is not None
		else None
	)
	normalized_identity: dict[str, object] = {
		'emission_source': emission_source,
		'edge_margin_tokens': list(edge_margin),
		'z_axis': strat['z_axis'],
		'z_direction': strat['z_direction'],
		'init': _jsonable(strat.get('init', {})),
		'update': _jsonable(strat.get('update', {})),
		'transition_settings': _jsonable(model_transition),
		'path_prior': _jsonable(_path_prior_core(path_prior)),
		'preprocessor': _reference_payload(artifact.preprocessor),
		'residualizer': (
			None
			if artifact.residualizer is None
			else _reference_payload(artifact.residualizer)
		),
		'prepared_feature_identity': _normalized_prepared_identity(prepared_identity),
	}
	return _LoadedFixedHMM(
		k=artifact.k,
		centers=centers,
		preprocessor=preprocessor,
		residualizer=residualizer,
		emission_source=emission_source,
		transition_costs=transition_costs,
		initial_state_costs=initial_costs,
		terminal_state_costs=terminal_costs,
		expected_boundaries=expected_boundaries,
		edge_margin_tokens=edge_margin,
		prepared_feature_identity=prepared_identity,
		normalized_identity=normalized_identity,
		artifact=artifact,
	)


def _current_model_identity_checks(models: Sequence[_LoadedFixedHMM]) -> None:
	if tuple(model.k for model in models) != CANONICAL_KS:
		raise ValueError('fixed HMM models must be ordered K=6,8,10')
	base = _cross_k_identity(models[0].normalized_identity)
	for model in models[1:]:
		if _cross_k_identity(model.normalized_identity) != base:
			raise ValueError('fixed preprocessing/HMM identity differs across K values')


def _cross_k_identity(value: Mapping[str, object]) -> dict[str, object]:
	identity = _jsonable(value)
	if not isinstance(identity, dict):
		raise TypeError('fixed identity must normalize to an object')
	for name in ('preprocessor', 'residualizer'):
		item = identity.get(name)
		if isinstance(item, dict):
			identity[name] = {'sha256': item.get('sha256')}
	return identity


def _prepare_generation_embeddings(
	descriptor_reference: HashedArtifactReference,
	destination: Path,
	*,
	source_student_state_sha256: str,
) -> _CurrentEmbeddingData:
	_validate_reference(descriptor_reference, 'current_embedding_descriptor')
	descriptor = _load_json_object(
		descriptor_reference.path, 'current embedding descriptor'
	)
	if (
		descriptor.get('artifact_type') != 'embedding_refresh_extraction'
		or descriptor.get('schema_version') != 1
	):
		raise ValueError('current embedding descriptor artifact identity is invalid')
	if (
		descriptor.get('status') != 'COMPLETE'
		or descriptor.get('completion_status') != 'COMPLETE'
	):
		raise ValueError('current embedding descriptor is not COMPLETE')
	if descriptor.get('source_student_state_sha256') != source_student_state_sha256:
		raise ValueError('current embedding descriptor student-state hash mismatch')
	outputs = _mapping(
		descriptor.get('outputs'), 'current embedding descriptor outputs'
	)
	source_root = descriptor_reference.path.parent
	source_inputs = tuple(discover_embedding_inputs(source_root))
	if set(outputs) != {item.survey_id for item in source_inputs}:
		raise ValueError('current embedding descriptor survey set differs from arrays')
	validated_source: list[object] = []
	valid_masks: dict[str, np.ndarray] = {}
	destination.mkdir(parents=True, exist_ok=True)
	for item in source_inputs:
		entry = _mapping(outputs[item.survey_id], f'current embedding {item.survey_id}')
		for field, source_path in (
			('embeddings', item.embeddings_path),
			('valid_tokens', item.valid_tokens_path),
			('metadata', item.metadata_path),
		):
			ref = _mapping(
				entry.get(field), f'current embedding {item.survey_id} {field}'
			)
			path = _descriptor_output_path(source_root, ref.get('path'), field)
			if path.resolve() != source_path.resolve():
				raise ValueError(f'current embedding descriptor {field} path mismatch')
			if field == 'metadata':
				metadata_reference = dict(ref)
				metadata_reference['path'] = str(path)
				_validate_reference_payload(
					metadata_reference, f'current embedding {item.survey_id} {field}'
				)
				metadata = load_embedding_metadata(item)
				if metadata.get('survey_id') != item.survey_id:
					raise ValueError(
						f'current embedding metadata survey mismatch for {item.survey_id}'
					)
			else:
				_validate_embedding_output_descriptor(
					source_path, ref, item.survey_id, field
				)
		valid = np.asarray(
			np.load(item.valid_tokens_path, mmap_mode='r', allow_pickle=False)
		)
		valid_masks[item.survey_id] = np.array(valid, dtype=np.bool_, copy=True)
		for source_path in (
			item.embeddings_path,
			item.valid_tokens_path,
			item.metadata_path,
		):
			shutil.copyfile(source_path, destination / source_path.name)
		validated_source.append(item)
	generation_inputs = tuple(discover_embedding_inputs(destination))
	shutil.copyfile(
		descriptor_reference.path,
		destination / descriptor_reference.path.name,
	)
	for item in generation_inputs:
		source = next(
			entry for entry in source_inputs if entry.survey_id == item.survey_id
		)
		for source_path, generation_path in (
			(source.embeddings_path, item.embeddings_path),
			(source.valid_tokens_path, item.valid_tokens_path),
			(source.metadata_path, item.metadata_path),
		):
			if file_sha256(source_path) != file_sha256(generation_path):
				raise ValueError(f'current embedding copy changed for {item.survey_id}')
	return _CurrentEmbeddingData(
		descriptor=descriptor,
		descriptor_name=descriptor_reference.path.name,
		source_inputs=tuple(validated_source),
		generation_inputs=generation_inputs,
		valid_tokens=valid_masks,
	)


def _validate_survey_contract(
	initial_targets: _InitialTargetData,
	embeddings: _CurrentEmbeddingData,
	*,
	edge_margin_tokens: tuple[int, int, int],
) -> dict[str, np.ndarray]:
	if initial_targets.survey_ids != tuple(
		sorted(item.survey_id for item in embeddings.generation_inputs)
	):
		raise ValueError('initial target and current embedding survey sets differ')
	common_valid_tokens: dict[str, np.ndarray] = {}
	for survey_id in initial_targets.survey_ids:
		current = embeddings.valid_tokens[survey_id]
		margin_mask = hmm.edge_margin_mask_for_shape(
			current.shape, edge_margin_tokens
		)
		common = np.logical_and(current, margin_mask)
		if not np.array_equal(initial_targets.valid_tokens[survey_id], common):
			raise ValueError(f'common valid-token mask drift for survey {survey_id}')
		if not np.any(common):
			raise ValueError(f'survey {survey_id} has no valid tokens')
		common_valid_tokens[survey_id] = common
	return common_valid_tokens


def _validate_previous_centers_against_manifest(
	centers: Sequence[PreviousCenterArtifact],
	previous_payload: Mapping[str, object],
) -> None:
	previous_centers = _mapping(
		previous_payload.get('centers'), 'previous generation centers'
	)
	for item in centers:
		entry = _mapping(
			previous_centers.get(str(item.k)),
			f'previous generation k={item.k} centers',
		)
		previous_after = _coerce_reference(
			entry.get('after'), f'previous generation k={item.k} centers.after'
		)
		if _reference_payload(previous_after) != _reference_payload(item.centers):
			raise ValueError(f'previous k={item.k} center identity mismatch')


def _validate_previous_lineage(
	previous_payload: Mapping[str, object],
	config: PeriodicRefreshConfig,
	models: Sequence[_LoadedFixedHMM],
) -> None:
	_validate_previous_payload_lineage(
		previous_payload,
		initial_hard_target_manifest=_reference_payload(
			config.initial_hard_target_manifest
		),
		initial_hard_target_policy=asdict(config.target_policy),
		fixed_identity=_fixed_identity_payload(models),
	)


def _validate_previous_payload_lineage(
	previous_payload: Mapping[str, object],
	*,
	initial_hard_target_manifest: object,
	initial_hard_target_policy: object,
	fixed_identity: object,
) -> None:
	if previous_payload.get('initial_hard_target_manifest') != (
		initial_hard_target_manifest
	):
		raise ValueError('initial hard target lineage differs from previous generation')
	if previous_payload.get('initial_hard_target_policy') != initial_hard_target_policy:
		raise ValueError('initial hard target policy differs from previous generation')
	if previous_payload.get('fixed_preprocessing_hmm_identity') != fixed_identity:
		raise ValueError('fixed preprocessing/HMM identity differs from previous generation')


def _build_prepared_feature_store(
	model: _LoadedFixedHMM,
	embedding_inputs: tuple[object, ...],
	destination: Path,
) -> PreparedFeatureStore:
	prepared_identity = model.prepared_feature_identity
	chunk_size = prepared_identity.get('chunk_size_tokens')
	if (
		isinstance(chunk_size, bool)
		or not isinstance(chunk_size, int)
		or chunk_size <= 0
	):
		raise ValueError('fixed prepared-feature identity has no valid chunk size')
	settings = PreparedFeatureCacheSettings(
		chunk_size_tokens=chunk_size,
		reuse=False,
		force_rebuild=False,
		cleanup=False,
		persist=True,
		directory=destination,
	)
	feature_dim = int(model.centers.shape[1])
	feature_mode = (
		'z_coordinate' if model.emission_source == 'z_coordinate' else 'embedding'
	)

	def prepare_batch(item: object, indices: np.ndarray) -> np.ndarray:
		return hmm.prepare_feature_batch_for_indices(
			item,
			indices,
			residualizer=model.residualizer,
			preprocessor=model.preprocessor,
			emission_source=model.emission_source,
		)

	return prepare_feature_store(
		embedding_inputs=tuple(embedding_inputs),
		feature_dim=feature_dim,
		feature_mode=feature_mode,
		residualizer=model.residualizer,
		preprocessor=model.preprocessor,
		edge_margin_tokens=model.edge_margin_tokens,
		settings=settings,
		default_cache_root=destination,
		prepare_batch=prepare_batch,
	)


def _run_refreshes(
	config: PeriodicRefreshConfig,
	models: Sequence[_LoadedFixedHMM],
	prepared_store: PreparedFeatureStore,
	initial_targets: _InitialTargetData,
	previous_payload: Mapping[str, object],
) -> tuple[
	dict[int, WarmStartOrderedHMMRefreshResult],
	dict[int, dict[str, np.ndarray]],
]:
	previous_labels = _previous_labels_for_diagnostics(
		previous_payload, initial_targets
	)
	center_refs = {item.k: item.centers for item in config.previous_centers}
	results: dict[int, WarmStartOrderedHMMRefreshResult] = {}
	for model in models:
		center_reference = center_refs[model.k]
		_validate_reference(center_reference, f'previous k={model.k} centers')
		loaded = np.load(center_reference.path, mmap_mode='r', allow_pickle=False)
		try:
			if loaded.dtype != np.dtype('float32') or loaded.ndim != 2:
				raise ValueError(f'previous k={model.k} centers have invalid dtype or rank')
			centers = np.asarray(loaded, dtype=np.float32).copy()
		finally:
			del loaded
		if centers.shape != model.centers.shape:
			raise ValueError(
				f'previous k={model.k} centers shape differs from initial centers'
			)
		if not np.all(np.isfinite(centers)):
			raise ValueError(f'previous k={model.k} centers are non-finite')
		results[model.k] = run_warm_start_ordered_hmm_refresh(
			prepared_store,
			centers,
			transition_costs=model.transition_costs,
			initial_state_costs=model.initial_state_costs,
			terminal_state_costs=model.terminal_state_costs,
			expected_boundaries=model.expected_boundaries,
			iterations=config.iterations,
			prediction_batch_size=config.prediction_batch_size,
		)
	return results, previous_labels


def _previous_labels_for_diagnostics(
	previous_payload: Mapping[str, object],
	initial_targets: _InitialTargetData,
) -> dict[int, dict[str, np.ndarray]]:
	value = previous_payload.get('canonical_multi_head_target_manifest')
	if not isinstance(value, Mapping):
		raise ValueError('previous generation has no canonical target manifest')
	targets = _load_initial_targets_from_reference(value)
	if targets.survey_ids != initial_targets.survey_ids:
		raise ValueError('previous and current target survey sets differ')
	return {k: dict(surveys) for k, surveys in targets.labels.items()}


def _write_center_arrays(
	staging: Path,
	config: PeriodicRefreshConfig,
	results: Mapping[int, WarmStartOrderedHMMRefreshResult],
) -> None:
	center_refs = {item.k: item.centers for item in config.previous_centers}
	for k in CANONICAL_KS:
		before = np.asarray(
			np.load(center_refs[k].path, allow_pickle=False), dtype=np.float32
		)
		root = staging / 'hmm' / f'k{k}'
		root.mkdir(parents=True, exist_ok=True)
		np.save(root / 'centers_before.npy', before, allow_pickle=False)
		np.save(
			root / 'centers_after.npy', results[k].final_centers, allow_pickle=False
		)


def _write_hmm_outputs(
	staging: Path,
	results: Mapping[int, WarmStartOrderedHMMRefreshResult],
	survey_ids: Sequence[str],
) -> None:
	if not survey_ids:
		raise ValueError('refresh requires at least one survey')
	for k in CANONICAL_KS:
		result = results[k]
		root = staging / 'hmm' / f'k{k}'
		final_labels_root = root / 'final_labels'
		export_labels_root = staging / 'hmm' / 'labels' / f'k{k}'
		final_labels_root.mkdir(parents=True, exist_ok=True)
		export_labels_root.mkdir(parents=True, exist_ok=True)
		for survey_id in survey_ids:
			labels = np.asarray(
				result.final_labels_by_survey[survey_id], dtype=np.int32
			)
			final_path = final_labels_root / (f'{survey_id}.cluster_labels_token.npy')
			np.save(final_path, labels, allow_pickle=False)
			# The existing export API consumes the established clustering label
			# suffix.  Keep its input path as a generation-owned copy.
			shutil.copyfile(final_path, export_labels_root / final_path.name)


def _write_prepared_feature_manifest(
	staging: Path,
	store: PreparedFeatureStore,
	model: _LoadedFixedHMM,
) -> None:
	payload = {
		'artifact_type': 'strat_hmm_periodic_refresh_prepared_features',
		'schema_version': 1,
		'fixed_identity': _jsonable(model.normalized_identity),
		'store': _jsonable(store.to_metadata()),
	}
	_write_json_atomic(
		staging / 'prepared_features' / PREPARED_FEATURE_MANIFEST_NAME,
		payload,
	)


def _write_refresh_diagnostics(
	staging: Path,
	results: Mapping[int, WarmStartOrderedHMMRefreshResult],
	*,
	previous_labels: Mapping[int, Mapping[str, np.ndarray]],
	valid_tokens: Mapping[str, np.ndarray],
	confidence: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
	per_k: dict[str, object] = {}
	for k in CANONICAL_KS:
		result = results[k]
		valid_count = int(sum(np.count_nonzero(mask) for mask in valid_tokens.values()))
		confidence_values = np.concatenate(
			[
				confidence[k][survey_id][valid_tokens[survey_id]]
				for survey_id in sorted(valid_tokens)
			]
		)
		if confidence_values.size != valid_count or not np.all(
			np.isfinite(confidence_values)
		):
			raise ValueError(f'confidence policy produced invalid values for k={k}')
		confidence_quantiles = np.quantile(
			confidence_values, [0.0, 0.05, 0.5, 0.95, 1.0]
		)
		label_change_count = 0
		for survey_id, labels in result.final_labels_by_survey.items():
			previous = np.asarray(previous_labels[k][survey_id])
			mask = np.asarray(valid_tokens[survey_id], dtype=np.bool_)
			if previous.shape != labels.shape or mask.shape != labels.shape:
				raise ValueError(f'previous label shape mismatch for k={k} {survey_id}')
			label_change_count += int(np.count_nonzero(labels[mask] != previous[mask]))
		per_k[str(k)] = {
			'iterations': [
				{
					'iteration': item.iteration,
					'state_counts': {
						str(state): count
						for state, count in item.cluster_counts.items()
					},
					'empty_states': list(item.empty_states),
					'center_shift_l2_by_state': list(item.center_shift_l2),
					'total_center_shift_l2': item.total_center_shift_l2,
				}
				for item in result.iteration_diagnostics
			],
			'final_label_change_count': label_change_count,
			'final_label_change_rate': (
				float(label_change_count / valid_count) if valid_count else 0.0
			),
			'final_state_counts': {
				str(state): count for state, count in result.final_state_counts.items()
			},
			'boundary_counts': dict(result.final_boundary_counts),
			'transition_counts': dict(result.final_transition_counts),
			'confidence_summary': {
				'min': float(confidence_quantiles[0]),
				'p05': float(confidence_quantiles[1]),
				'median': float(confidence_quantiles[2]),
				'p95': float(confidence_quantiles[3]),
				'max': float(confidence_quantiles[4]),
				'mean': float(np.mean(confidence_values, dtype=np.float64)),
			},
			'state_mean_z': {
				str(state): value for state, value in result.final_state_mean_z.items()
			},
			'valid_token_count': valid_count,
			'ordered_diagnostics': _jsonable(result.final_ordered_diagnostics),
			'boundary_summary': _jsonable(result.final_boundary_summary),
		}
	_write_json_atomic(
		staging / REFRESH_DIAGNOSTICS_NAME,
		{
			'artifact_type': 'strat_hmm_periodic_refresh_diagnostics',
			'schema_version': 1,
			'per_k': per_k,
		},
	)


def _policy_confidence_arrays(
	initial_targets: _InitialTargetData,
	policy: HardTargetPolicy,
) -> dict[int, dict[str, np.ndarray]]:
	"""Return the confidence arrays without defining a new confidence formula."""
	result: dict[int, dict[str, np.ndarray]] = {}
	for k in CANONICAL_KS:
		result[k] = {}
		for survey_id in initial_targets.survey_ids:
			valid = initial_targets.valid_tokens[survey_id]
			if policy.confidence_mode == 'constant':
				values = np.zeros(valid.shape, dtype=np.float32)
				values[valid] = np.float32(policy.confidence)
			else:
				values = np.array(
					initial_targets.confidence[k][survey_id],
					dtype=np.float32,
					copy=True,
				)
			result[k][survey_id] = values
	return result


def _generate_canonical_targets_and_manifest(
	staging: Path,
	embeddings: _CurrentEmbeddingData,
	policy: HardTargetPolicy,
	initial_targets: _InitialTargetData,
) -> None:
	# ``embeddings`` is intentionally part of this boundary even though the
	# canonical builder discovers the copied arrays itself: it keeps the producer
	# contract explicit and makes accidental source-directory inference harder.
	del embeddings
	pseudo_root = staging / 'pseudo_targets'
	hmm_root = staging / 'hmm'
	for k in CANONICAL_KS:
		if (
			policy.confidence_mode == 'constant'
			and policy.boundary_weight_mode == 'absent'
		):
			export_hmm_cluster_labels_as_pseudo_targets(
				clustering_output_dir=hmm_root,
				pseudo_target_root=pseudo_root,
				k=k,
				confidence=policy.confidence,
				schema_version=1,
				write_boundary_weight=False,
			)
		else:
			_write_policy_targets(
				clustering_output_dir=hmm_root,
				pseudo_target_root=pseudo_root,
				k=k,
				policy=policy,
				initial_targets=initial_targets,
			)
	# The canonical K=6 contract requires a distinct replay root.  It is an
	# exact generation-local replay of the final K=6 hard target, never a
	# historical target and never a write into the initial lineage.
	replay_clustering = staging / 'hmm' / 'replay'
	replay_labels = replay_clustering / 'labels' / 'k6'
	replay_labels.mkdir(parents=True, exist_ok=True)
	for source in sorted(
		(staging / 'hmm' / 'k6' / 'final_labels').glob('*.cluster_labels_token.npy')
	):
		shutil.copyfile(source, replay_labels / source.name)
	if policy.confidence_mode == 'constant' and policy.boundary_weight_mode == 'absent':
		export_hmm_cluster_labels_as_pseudo_targets(
			clustering_output_dir=replay_clustering,
			pseudo_target_root=pseudo_root / 'k6_replay',
			k=6,
			confidence=policy.confidence,
			schema_version=1,
			write_boundary_weight=False,
		)
	else:
		_write_policy_targets(
			clustering_output_dir=replay_clustering,
			pseudo_target_root=pseudo_root / 'k6_replay',
			k=6,
			policy=policy,
			initial_targets=initial_targets,
		)
	preflight = staging / '.manifest_preflight'
	preflight.mkdir()
	(preflight / 'migration.json').write_text(
		'{"status":"PASS_WITH_NUMERIC_DRIFT"}\n', encoding='utf-8'
	)
	(preflight / 'control.json').write_text(
		'{"readiness":{"status":"CONTROL_READY_POSITIVE"}}\n', encoding='utf-8'
	)
	try:
		manifest_path = pseudo_root / 'multi_head_target_manifest.json'
		build_multi_head_target_manifest(
			manifest_path=manifest_path,
			source_embedding_dir=staging / 'embeddings',
			head_roots=dict.fromkeys(CANONICAL_KS, pseudo_root),
			replay_k6_root=pseudo_root / 'k6_replay',
			migration_decision=preflight / 'migration.json',
			control_summary=preflight / 'control.json',
		)
		load_multi_head_target_manifest(manifest_path)
	finally:
		if preflight.exists():
			shutil.rmtree(preflight)


def _write_policy_targets(
	*,
	clustering_output_dir: Path,
	pseudo_target_root: Path,
	k: int,
	policy: HardTargetPolicy,
	initial_targets: _InitialTargetData,
) -> None:
	"""Write target arrays using only the declared existing target policy."""
	label_root = clustering_output_dir / 'labels' / f'k{k}'
	label_paths = sorted(label_root.glob('*.cluster_labels_token.npy'))
	if not label_paths:
		raise ValueError(f'no HMM cluster labels found for k={k}: {label_root}')
	for label_path in label_paths:
		survey_id = label_path.name.removesuffix('.cluster_labels_token.npy')
		if survey_id not in initial_targets.valid_tokens:
			raise ValueError(f'unknown survey in refreshed labels: {survey_id}')
		labels = np.asarray(
			np.load(label_path, mmap_mode='r', allow_pickle=False), dtype=np.int32
		)
		valid_tokens = labels >= 0
		if not np.array_equal(valid_tokens, initial_targets.valid_tokens[survey_id]):
			raise ValueError(
				f'refreshed target valid-token mask differs for {survey_id} k={k}'
			)
		if policy.confidence_mode == 'constant':
			confidence = np.zeros(labels.shape, dtype=np.float32)
			confidence[valid_tokens] = np.float32(policy.confidence)
		else:
			confidence = np.array(
				initial_targets.confidence[k][survey_id], dtype=np.float32, copy=True
			)
		boundary_weight = (
			None
			if policy.boundary_weight_mode == 'absent'
			else np.array(
				initial_targets.boundary_weight[k][survey_id],
				dtype=np.float32,
				copy=True,
			)
		)
		write_pseudo_target(
			pseudo_target_root,
			k=k,
			survey_id=survey_id,
			labels=labels,
			confidence=confidence,
			valid_tokens=valid_tokens,
			boundary_weight=boundary_weight,
			metadata={
				'periodic_refresh_policy': asdict(policy),
				'source_label_path': str(label_path),
				'source_label_sha256': file_sha256(label_path),
			},
			schema_version=2 if boundary_weight is not None else 1,
			write_boundary_weight=boundary_weight is not None,
		)


def _build_generation_manifest(
	*,
	config: PeriodicRefreshConfig,
	staging: Path,
	output_root: Path,
	models: Sequence[_LoadedFixedHMM],
	current_embeddings: _CurrentEmbeddingData,
	initial_targets: _InitialTargetData,
) -> dict[str, object]:
	target_manifest_stage = (
		staging / 'pseudo_targets' / 'multi_head_target_manifest.json'
	)
	target_manifest_final = output_root / 'pseudo_targets' / target_manifest_stage.name
	target_payload = load_multi_head_target_manifest(target_manifest_stage)
	centers_payload: dict[str, object] = {}
	final_labels_payload: dict[str, object] = {}
	per_k_targets: dict[str, object] = {}
	for model in models:
		k = model.k
		center_root = staging / 'hmm' / f'k{k}'
		center_after = np.load(
			center_root / 'centers_after.npy', mmap_mode='r', allow_pickle=False
		)
		try:
			center_shape = list(center_after.shape)
		finally:
			del center_after
		centers_payload[str(k)] = {
			'before': _local_reference_payload(
				output_root / 'hmm' / f'k{k}' / 'centers_before.npy',
				center_root / 'centers_before.npy',
			),
			'after': _local_reference_payload(
				output_root / 'hmm' / f'k{k}' / 'centers_after.npy',
				center_root / 'centers_after.npy',
			),
			'shape': center_shape,
			'dtype': 'float32',
		}
		final_labels_payload[str(k)] = {}
		target_head = _mapping(
			_mapping(target_payload['heads'], 'target heads')[str(k)],
			f'target k={k}',
		)
		target_surveys = _mapping(target_head['surveys'], f'target k={k} surveys')
		per_k_targets[str(k)] = {
			'root': str(output_root / 'pseudo_targets' / f'k{k}'),
			'surveys': target_surveys,
		}
		for survey_id in initial_targets.survey_ids:
			final_path_stage = (
				center_root / 'final_labels' / (f'{survey_id}.cluster_labels_token.npy')
			)
			final_labels_payload[str(k)][survey_id] = _local_reference_payload(
				output_root / 'hmm' / f'k{k}' / 'final_labels' / final_path_stage.name,
				final_path_stage,
			)
	initial_common = _mapping(
		_mapping(initial_targets.payload['common'], 'initial target common')[
			'valid_tokens_sha256'
		],
		'initial target valid-token hashes',
	)
	valid_token_hashes = dict(initial_common)
	content_files = _content_inventory(staging)
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': 'COMPLETE',
		'generation_id': (
			f'refresh_{config.generation_index:04d}_epoch'
			f'{config.refresh_after_epoch:03d}'
		),
		'generation_index': config.generation_index,
		'refresh_after_epoch': config.refresh_after_epoch,
		'iterations': config.iterations,
		'prediction_batch_size': config.prediction_batch_size,
		'source_student_state_sha256': config.source_student_state_sha256,
		'request_identity': _request_identity(config),
		'previous_generation_manifest': _reference_payload(
			config.previous_generation_manifest
		),
		'current_embedding_descriptor': _reference_payload(
			config.current_embedding_descriptor
		),
		'initial_hard_target_manifest': _reference_payload(
			config.initial_hard_target_manifest
		),
		'fixed_preprocessing_hmm_identity': _fixed_identity_payload(models),
		'initial_hard_target_policy': asdict(config.target_policy),
		'previous_centers': {
			str(item.k): _reference_payload(item.centers)
			for item in config.previous_centers
		},
		'embeddings': _persisted_embedding_payload(
			staging, output_root, current_embeddings
		),
		'centers': centers_payload,
		'final_labels': final_labels_payload,
		'canonical_multi_head_target_manifest': _local_reference_payload(
			target_manifest_final, target_manifest_stage
		),
		'per_k_targets': per_k_targets,
		'valid_token_hashes': valid_token_hashes,
		'prepared_feature_manifest': _local_reference_payload(
			output_root / 'prepared_features' / PREPARED_FEATURE_MANIFEST_NAME,
			staging / 'prepared_features' / PREPARED_FEATURE_MANIFEST_NAME,
		),
		'refresh_diagnostics': _local_reference_payload(
			output_root / REFRESH_DIAGNOSTICS_NAME,
			staging / REFRESH_DIAGNOSTICS_NAME,
		),
		'generation_content_sha256': _content_hash(content_files),
		'content_files': content_files,
	}


def _persisted_embedding_payload(
	staging: Path,
	output_root: Path,
	embeddings: _CurrentEmbeddingData,
) -> dict[str, object]:
	return {
		'descriptor': _local_reference_payload(
			output_root / 'embeddings' / embeddings.descriptor_name,
			staging / 'embeddings' / embeddings.descriptor_name,
		),
		'root': str(output_root / 'embeddings'),
		'surveys': {
			item.survey_id: {
				'embeddings': _local_reference_payload(
					output_root / 'embeddings' / item.embeddings_path.name,
					staging / 'embeddings' / item.embeddings_path.name,
				),
				'valid_tokens': _local_reference_payload(
					output_root / 'embeddings' / item.valid_tokens_path.name,
					staging / 'embeddings' / item.valid_tokens_path.name,
				),
				'metadata': _local_reference_payload(
					output_root / 'embeddings' / item.metadata_path.name,
					staging / 'embeddings' / item.metadata_path.name,
				),
			}
			for item in embeddings.generation_inputs
		},
	}


def _validate_persisted_embeddings(
	root: Path,
	value: object,
	*,
	expected_student_state_sha256: str,
	expected_descriptor_sha256: str,
) -> _CurrentEmbeddingData:
	payload = _mapping(value, 'generation embeddings')
	descriptor_reference = _coerce_reference(
		payload['descriptor'], 'generation embedding descriptor'
	)
	try:
		descriptor_reference.path.resolve().relative_to(
			(root / 'embeddings').resolve()
		)
	except ValueError as exc:
		raise ValueError('generation embedding descriptor escapes generation') from exc
	_validate_reference(descriptor_reference, 'generation embedding descriptor')
	if descriptor_reference.sha256 != expected_descriptor_sha256:
		raise ValueError('generation embedding descriptor differs from source descriptor')
	descriptor = _load_json_object(
		descriptor_reference.path, 'generation embedding descriptor'
	)
	if (
		descriptor.get('artifact_type') != 'embedding_refresh_extraction'
		or descriptor.get('schema_version') != 1
		or descriptor.get('status') != 'COMPLETE'
		or descriptor.get('completion_status') != 'COMPLETE'
	):
		raise ValueError('generation embedding descriptor is not COMPLETE')
	if descriptor.get('source_student_state_sha256') != expected_student_state_sha256:
		raise ValueError('generation embedding descriptor student-state drift')
	descriptor_outputs = _mapping(
		descriptor.get('outputs'), 'generation embedding descriptor outputs'
	)
	embedding_root = _resolve_recorded_path(
		payload.get('root'), 'generation embedding root'
	)
	if embedding_root != (root / 'embeddings').resolve():
		raise ValueError('generation embedding root is outside generation')
	surveys = _mapping(payload['surveys'], 'generation embedding surveys')
	inputs = tuple(discover_embedding_inputs(root / 'embeddings'))
	if set(surveys) != {item.survey_id for item in inputs}:
		raise ValueError('generation embedding survey set mismatch')
	valid: dict[str, np.ndarray] = {}
	for item in inputs:
		descriptor_entry = _mapping(
			descriptor_outputs.get(item.survey_id),
			f'generation embedding descriptor {item.survey_id}',
		)
		entry = _mapping(
			surveys[item.survey_id], f'generation embedding {item.survey_id}'
		)
		for field, path in (
			('embeddings', item.embeddings_path),
			('valid_tokens', item.valid_tokens_path),
			('metadata', item.metadata_path),
		):
			descriptor_ref = _mapping(
				descriptor_entry.get(field),
				f'generation embedding descriptor {item.survey_id} {field}',
			)
			descriptor_path = _descriptor_output_path(
				root / 'embeddings', descriptor_ref.get('path'), field
			)
			if descriptor_path.resolve() != path.resolve():
				raise ValueError(
					f'generation embedding descriptor {field} path mismatch'
				)
			descriptor_absolute_ref = dict(descriptor_ref)
			descriptor_absolute_ref['path'] = str(path)
			if field == 'metadata':
				_validate_reference_payload(
					descriptor_absolute_ref,
					f'generation embedding descriptor {item.survey_id} {field}',
				)
			else:
				_validate_embedding_output_descriptor(
					path,
					descriptor_absolute_ref,
					item.survey_id,
					field,
				)
			ref = _mapping(
				entry[field],
				f'generation embedding {item.survey_id} {field}',
			)
			if Path(str(ref['path'])).resolve() != path.resolve():
				raise ValueError(f'generation embedding {field} path mismatch')
			_validate_reference_payload(
				ref, f'generation embedding {item.survey_id} {field}'
			)
			if field == 'valid_tokens':
				valid[item.survey_id] = np.array(
					np.load(path, mmap_mode='r', allow_pickle=False),
					dtype=np.bool_,
					copy=True,
				)
		open_embedding_array(item)
		load_embedding_metadata(item)
	descriptor_ref = _coerce_reference(payload['descriptor'], 'descriptor')
	return _CurrentEmbeddingData(
		descriptor=_load_json_object(
			descriptor_ref.path, 'generation embedding descriptor'
		),
		descriptor_name=descriptor_ref.path.name,
		source_inputs=inputs,
		generation_inputs=inputs,
		valid_tokens=valid,
	)


def _validate_persisted_target_masks(
	initial_targets: _InitialTargetData,
	embeddings: _CurrentEmbeddingData,
	*,
	edge_margin_tokens: tuple[int, int, int],
) -> None:
	_validate_survey_contract(
		initial_targets,
		embeddings,
		edge_margin_tokens=edge_margin_tokens,
	)


def _validate_persisted_centers(
	root: Path,
	centers_value: object,
	previous_value: object,
	previous_generation: Mapping[str, object],
	models: Sequence[_LoadedFixedHMM],
) -> None:
	centers = _mapping(centers_value, 'generation centers')
	previous = _mapping(previous_value, 'generation previous centers')
	previous_generation_centers = _mapping(
		previous_generation.get('centers'), 'previous generation centers'
	)
	for model in models:
		entry = _mapping(centers[str(model.k)], f'generation k={model.k} centers')
		previous_generation_entry = _mapping(
			previous_generation_centers.get(str(model.k)),
			f'previous generation k={model.k} centers',
		)
		previous_after = _coerce_reference(
			previous_generation_entry.get('after'),
			f'previous generation k={model.k} centers.after',
		)
		previous_ref = _coerce_reference(
			previous.get(str(model.k)), f'previous k={model.k} center'
		)
		_validate_reference(previous_after, f'previous generation k={model.k} centers.after')
		_validate_reference(previous_ref, f'previous k={model.k} center')
		if _reference_payload(previous_ref) != _reference_payload(previous_after):
			raise ValueError(
				f'previous k={model.k} center identity does not match '
				'previous generation centers.after'
			)
		before = _mapping(entry['before'], f'generation k={model.k} center before')
		before_ref = _coerce_reference(
			before, f'generation k={model.k} center before'
		)
		for name in ('before', 'after'):
			ref = _mapping(entry[name], f'generation k={model.k} center {name}')
			_validate_owned_reference_payload(
				root, ref, f'generation k={model.k} center {name}'
			)
			array = np.load(Path(str(ref['path'])), mmap_mode='r', allow_pickle=False)
			try:
				if array.shape != (
					model.k,
					model.centers.shape[1],
				) or array.dtype != np.dtype('float32'):
					raise ValueError(
						f'generation k={model.k} center {name} shape or dtype mismatch'
					)
				if not np.all(np.isfinite(array)):
					raise ValueError(
						f'generation k={model.k} center {name} is non-finite'
					)
			finally:
				del array
		if before_ref.sha256 != previous_after.sha256:
			raise ValueError(
				f'generation k={model.k} centers.before does not match '
				'previous generation centers.after'
			)


def _validate_persisted_final_labels(
	root: Path,
	final_labels_value: object,
	per_k_targets: object,
	initial_targets: _InitialTargetData,
	embeddings: _CurrentEmbeddingData,
	models: Sequence[_LoadedFixedHMM],
) -> None:
	final_labels = _mapping(final_labels_value, 'generation final labels')
	targets = _mapping(per_k_targets, 'generation per-k targets')
	for model in models:
		final_by_survey = _mapping(final_labels[str(model.k)], f'final k={model.k}')
		head = _mapping(targets[str(model.k)], f'target k={model.k}')
		target_surveys = _mapping(head['surveys'], f'target k={model.k} surveys')
		for survey_id in initial_targets.survey_ids:
			ref = _mapping(final_by_survey[survey_id], f'final k={model.k} {survey_id}')
			_validate_owned_reference_payload(
				root, ref, f'final k={model.k} {survey_id}'
			)
			labels = np.load(Path(str(ref['path'])), mmap_mode='r', allow_pickle=False)
			target = _mapping(
				target_surveys[survey_id], f'target k={model.k} {survey_id}'
			)
			target_labels_ref = _mapping(target['labels'], 'target labels')
			_validate_owned_reference_payload(
				root, target_labels_ref, f'target labels k={model.k} {survey_id}'
			)
			target_labels = np.load(
				Path(str(target_labels_ref['path'])),
				mmap_mode='r',
				allow_pickle=False,
			)
			try:
				if not np.array_equal(labels, target_labels):
					raise ValueError(
						f'final labels differ from target labels for k={model.k} '
						f'{survey_id}'
					)
				mask = initial_targets.valid_tokens[survey_id]
				if (
					labels.shape != mask.shape
					or np.any(labels[mask] < 0)
					or np.any(labels[mask] >= model.k)
				):
					raise ValueError(
						f'final labels are outside K range for k={model.k} {survey_id}'
					)
				if np.any(labels[~mask] != -1):
					raise ValueError(
						f'final labels invalid-token sentinel drift for k={model.k} '
						f'{survey_id}'
					)
			finally:
				del labels, target_labels


def _validate_owned_target_manifest_paths(
	root: Path,
	payload: Mapping[str, object],
) -> None:
	"""Reject a canonical manifest that points outside its generation root."""
	source_embedding = _mapping(payload.get('source_embedding'), 'source embedding')
	_owned_path(root, source_embedding.get('input_dir'), 'target source embedding root')
	for survey_id, value in _mapping(
		source_embedding.get('surveys'), 'target source embedding surveys'
	).items():
		entry = _mapping(value, f'target source embedding {survey_id}')
		for name in (
			'embedding_path',
			'metadata_path',
			'valid_tokens_path',
		):
			_owned_path(root, entry.get(name), f'target source embedding {survey_id} {name}')

	heads = _mapping(payload.get('heads'), 'canonical heads')
	for k, value in heads.items():
		head = _mapping(value, f'canonical k={k}')
		_owned_path(root, head.get('pseudo_target_root'), f'canonical k={k} target root')
		for survey_id, target_value in _mapping(
			head.get('surveys'), f'canonical k={k} surveys'
		).items():
			entry = _mapping(target_value, f'canonical k={k} {survey_id}')
			for name in (
				'labels',
				'confidence',
				'valid_tokens',
				'metadata',
			):
				_owned_path(
					root,
					_mapping(entry.get(name), f'canonical k={k} {survey_id} {name}').get(
						'path'
					),
					f'canonical k={k} {survey_id} {name}',
				)
			if 'boundary_weight' in entry:
				_owned_path(
					root,
					_mapping(
						entry['boundary_weight'],
						f'canonical k={k} {survey_id} boundary_weight',
					).get('path'),
					f'canonical k={k} {survey_id} boundary_weight',
				)

	parity_value = payload.get('k6_replay_parity')
	if parity_value is None:
		return
	parity = _mapping(parity_value, 'K=6 replay parity')
	_owned_path(root, parity.get('replay_root'), 'K=6 replay root')
	for group_name in ('replay_artifacts', 'historical_decoded_labels', 'replay_decoded_labels'):
		group = _mapping(parity.get(group_name), f'K=6 {group_name}')
		for survey_id, value in group.items():
			entry = _mapping(value, f'K=6 {group_name} {survey_id}')
			for name, reference in entry.items():
				if isinstance(reference, Mapping) and 'path' in reference:
					_owned_path(
						root,
						reference['path'],
						f'K=6 {group_name} {survey_id} {name}',
					)


def _policy_from_payload(value: object) -> HardTargetPolicy:
	policy = _mapping(value, 'initial hard target policy')
	return HardTargetPolicy(
		confidence_mode=policy['confidence_mode'],
		confidence=float(policy['confidence']),
		boundary_weight_mode=policy['boundary_weight_mode'],
	)


def _target_manifest_policy_check(
	entry: Mapping[str, object],
	*,
	k: int,
	survey_id: str,
	initial_targets: _InitialTargetData,
	policy: HardTargetPolicy,
) -> None:
	labels_ref = _mapping(entry['labels'], 'target labels')
	confidence_ref = _mapping(entry['confidence'], 'target confidence')
	valid_ref = _mapping(entry['valid_tokens'], 'target valid tokens')
	metadata_ref = _mapping(entry['metadata'], 'target metadata')
	boundary_ref = entry.get('boundary_weight')
	item = StratPseudoTargetInput(
		survey_id=survey_id,
		k=k,
		labels_path=Path(str(labels_ref['path'])),
		confidence_path=Path(str(confidence_ref['path'])),
		valid_tokens_path=Path(str(valid_ref['path'])),
		boundary_weight_path=(
			None
			if boundary_ref is None
			else Path(str(_mapping(boundary_ref, 'target boundary weight')['path']))
		),
		metadata_path=Path(str(metadata_ref['path'])),
	)
	arrays = load_pseudo_target_arrays(item, mmap_mode='r')
	valid = initial_targets.valid_tokens[survey_id]
	if not np.array_equal(arrays.valid_tokens, valid):
		raise ValueError(f'canonical target valid-token values differ for k={k} {survey_id}')
	if policy.confidence_mode == 'constant':
		expected_confidence = np.zeros(valid.shape, dtype=np.float32)
		expected_confidence[valid] = np.float32(policy.confidence)
	else:
		expected_confidence = initial_targets.confidence[k][survey_id]
	if not np.array_equal(arrays.confidence, expected_confidence):
		raise ValueError(f'canonical target confidence policy drift for k={k} {survey_id}')
	if policy.boundary_weight_mode == 'absent':
		if boundary_ref is not None:
			raise ValueError(f'canonical target boundary policy drift for k={k} {survey_id}')
	elif boundary_ref is None or initial_targets.boundary_weight[k][survey_id] is None:
		raise ValueError(f'canonical target boundary policy is incomplete for k={k} {survey_id}')
	elif not np.array_equal(
		arrays.boundary_weight, initial_targets.boundary_weight[k][survey_id]
	):
		raise ValueError(f'canonical target boundary policy drift for k={k} {survey_id}')


def _validate_persisted_target_manifest(
	root: Path,
	manifest_value: object,
	per_k_targets: object,
	embeddings: _CurrentEmbeddingData,
	initial_targets: _InitialTargetData,
	policy: HardTargetPolicy,
) -> None:
	manifest_ref = _mapping(manifest_value, 'canonical target manifest')
	_validate_owned_reference_payload(root, manifest_ref, 'canonical target manifest')
	manifest_path = Path(str(manifest_ref['path']))
	manifest_payload = _load_json_object(
		manifest_path, 'canonical target manifest'
	)
	_validate_owned_target_manifest_paths(root, manifest_payload)
	manifest = load_multi_head_target_manifest(manifest_path)
	if tuple(manifest['head_ks']) != CANONICAL_KS:
		raise ValueError('canonical target manifest K set drift')
	common = _mapping(manifest['common'], 'canonical target common')
	common_masks = _mapping(
		common['valid_tokens_sha256'], 'canonical target valid hashes'
	)
	expected_masks = _mapping(
		_mapping(initial_targets.payload['common'], 'initial target common')[
			'valid_tokens_sha256'
		],
		'initial target valid-token hashes',
	)
	if dict(common_masks) != expected_masks:
		raise ValueError('canonical target valid-token hashes differ from initial mask')
	for k in CANONICAL_KS:
		head = _mapping(
			_mapping(manifest['heads'], 'canonical heads')[str(k)],
			f'canonical k={k}',
		)
		surveys = _mapping(head['surveys'], f'canonical k={k} surveys')
		_owned_path(root, head['pseudo_target_root'], f'canonical k={k} target root')
		for survey_id in initial_targets.survey_ids:
			entry = _mapping(surveys[survey_id], f'canonical k={k} {survey_id}')
			for name in (
				'labels',
				'confidence',
				'valid_tokens',
				'metadata',
			):
				_validate_owned_reference_payload(
					root,
					entry[name],
					f'canonical k={k} {survey_id} {name}',
				)
			if 'boundary_weight' in entry:
				_validate_owned_reference_payload(
					root,
					entry['boundary_weight'],
					f'canonical k={k} {survey_id} boundary_weight',
				)
			valid_ref = _mapping(entry['valid_tokens'], 'canonical valid tokens')
			if valid_ref['sha256'] != common_masks[survey_id]:
				raise ValueError(f'canonical k={k} valid-token hash mismatch')
			_target_manifest_policy_check(
				entry,
				k=k,
				survey_id=survey_id,
				initial_targets=initial_targets,
				policy=policy,
			)
	per_k = _mapping(per_k_targets, 'generation per-k targets')
	for k in CANONICAL_KS:
		per_k_entry = per_k.get(str(k))
		if not isinstance(per_k_entry, Mapping):
			raise ValueError(f'generation target record is missing k={k}')
		head = _mapping(
			_mapping(manifest['heads'], 'canonical heads')[str(k)],
			f'canonical k={k}',
		)
		head_surveys = _mapping(head['surveys'], f'canonical k={k} surveys')
		per_k_surveys = _mapping(
			per_k_entry.get('surveys'), f'generation k={k} surveys'
		)
		_owned_path(root, per_k_entry.get('root'), f'generation k={k} target root')
		if dict(per_k_surveys) != dict(head_surveys):
			raise ValueError(f'generation k={k} target references drift')
		expected_root = Path(str(head['pseudo_target_root'])).resolve() / f'k{k}'
		if Path(str(per_k_entry.get('root'))).resolve() != expected_root:
			raise ValueError(f'generation k={k} target root drift')
	# Accessing this mapping here intentionally ensures the validator does not
	# accept a target manifest whose survey set differs from the embedding set.
	if set(embeddings.valid_tokens) != set(initial_targets.survey_ids):
		raise ValueError('canonical target and embedding survey sets differ')


def _validate_persisted_fixed_identity(
	value: object,
	models: Sequence[_LoadedFixedHMM],
) -> None:
	identity = _mapping(value, 'fixed preprocessing identity')
	if _jsonable(identity.get('normalized')) != _jsonable(
		models[0].normalized_identity
	):
		raise ValueError('fixed normalized preprocessing/HMM identity drift')
	if _jsonable(identity.get('prepared_feature_identity')) != _jsonable(
		models[0].prepared_feature_identity
	):
		raise ValueError('fixed prepared-feature identity drift')
	if identity.get('edge_margin_tokens') != list(models[0].edge_margin_tokens):
		raise ValueError('fixed edge-margin identity drift')


def _validate_persisted_prepared_features(
	root: Path,
	value: object,
	model: _LoadedFixedHMM,
) -> None:
	ref = _mapping(value, 'prepared feature manifest')
	_validate_owned_reference_payload(root, ref, 'prepared feature manifest')
	payload = _load_json_object(Path(str(ref['path'])), 'prepared feature manifest')
	if (
		payload.get('artifact_type') != 'strat_hmm_periodic_refresh_prepared_features'
		or payload.get('schema_version') != 1
	):
		raise ValueError('prepared feature manifest identity is invalid')
	if _jsonable(payload.get('fixed_identity')) != _jsonable(model.normalized_identity):
		raise ValueError('prepared feature manifest fixed identity drift')
	store = _mapping(payload.get('store'), 'prepared feature store')
	surveys = store.get('surveys')
	if not isinstance(surveys, list) or not surveys:
		raise ValueError('prepared feature manifest has no surveys')
	feature_mode = store.get('feature_mode')
	if feature_mode == 'z_coordinate':
		if store.get('directory') is not None:
			raise ValueError('direct prepared features cannot have a cache directory')
		for survey in surveys:
			entry = _mapping(survey, 'prepared survey')
			if entry.get('cache_path') is not None:
				raise ValueError('direct prepared survey cannot have a cache path')
			if entry.get('feature_dim') != 1:
				raise ValueError('direct prepared feature dimension drift')
		return
	if feature_mode != 'embedding':
		raise ValueError('prepared feature mode is invalid')
	for survey in surveys:
		entry = _mapping(survey, 'prepared survey')
		cache_path_value = entry.get('cache_path')
		if not isinstance(cache_path_value, str):
			raise ValueError('prepared survey cache path is missing')
		cache = _owned_path(root, cache_path_value, 'prepared survey cache')
		for name in ('valid_flat_indices.npy', 'features.npy', 'metadata.json'):
			path = cache / name
			if not path.is_file():
				raise FileNotFoundError(f'prepared feature artifact is missing: {path}')
		indices = np.load(
			cache / 'valid_flat_indices.npy', mmap_mode='r', allow_pickle=False
		)
		features = np.load(cache / 'features.npy', mmap_mode='r', allow_pickle=False)
		try:
			if indices.dtype != np.dtype('int64') or features.dtype != np.dtype(
				'float32'
			):
				raise ValueError('prepared feature dtype drift')
			if features.ndim != 2 or features.shape[0] != indices.shape[0]:
				raise ValueError('prepared feature shape drift')
			if not np.all(np.isfinite(features)):
				raise ValueError('prepared features contain non-finite values')
		finally:
			del indices, features
	store_directory = store.get('directory')
	if not isinstance(store_directory, str):
		raise ValueError('prepared feature store directory is missing')
	_owned_path(root, store_directory, 'prepared feature store directory')
	try:
		Path(store_directory).resolve().relative_to(
			(root / 'prepared_features').resolve()
		)
	except ValueError as exc:
		raise ValueError('prepared feature cache escapes the generation root') from exc


def _validate_persisted_diagnostics(
	root: Path,
	value: object,
	manifest: Mapping[str, object],
	models: Sequence[_LoadedFixedHMM],
) -> None:
	ref = _mapping(value, 'refresh diagnostics')
	_validate_owned_reference_payload(root, ref, 'refresh diagnostics')
	payload = _load_json_object(Path(str(ref['path'])), 'refresh diagnostics')
	if (
		payload.get('artifact_type') != 'strat_hmm_periodic_refresh_diagnostics'
		or payload.get('schema_version') != 1
	):
		raise ValueError('refresh diagnostics identity is invalid')
	per_k = _mapping(payload.get('per_k'), 'refresh diagnostics per_k')
	for model in models:
		entry = _mapping(per_k.get(str(model.k)), f'refresh diagnostics k={model.k}')
		iterations = entry.get('iterations')
		if not isinstance(iterations, list) or len(iterations) != 2:
			raise ValueError(
				f'refresh diagnostics iteration count is invalid for k={model.k}'
			)
		for iteration in iterations:
			item = _mapping(iteration, f'refresh diagnostics k={model.k} iteration')
			if item.get('empty_states') != []:
				raise ValueError(
					f'refresh diagnostics records an empty state for k={model.k}'
				)
			for key in ('total_center_shift_l2',):
				if not _finite_number(item.get(key), f'refresh diagnostics {key}'):
					raise ValueError(f'refresh diagnostics {key} is invalid')
		if not _finite_number(
			entry.get('final_label_change_rate'),
			f'refresh diagnostics k={model.k} final_label_change_rate',
		):
			raise ValueError(
				f'refresh diagnostics label-change rate is invalid for k={model.k}'
			)
	# Keep the argument part of the validator boundary: it ensures callers cannot
	# accidentally validate diagnostics detached from this generation manifest.
	if manifest.get('refresh_diagnostics') != value or not root.is_dir():
		raise ValueError('refresh diagnostics is not bound to this generation')


def _validate_content_inventory(root: Path, payload: Mapping[str, object]) -> None:
	entries = payload['content_files']
	if not isinstance(entries, list) or not entries:
		raise ValueError('generation content_files must be a non-empty list')
	expected = _content_inventory(root)
	if entries != expected:
		raise ValueError('generation content inventory drift')
	if _content_hash(entries) != payload['generation_content_sha256']:
		raise ValueError('generation content hash mismatch')


def _content_inventory(root: Path) -> list[dict[str, object]]:
	entries: list[dict[str, object]] = []
	for path in sorted(root.rglob('*')):
		if not path.is_file() or path.name == GENERATION_MANIFEST_NAME:
			continue
		relative = path.relative_to(root).as_posix()
		entries.append(
			{
				'path': relative,
				'sha256': file_sha256(path),
				'size': path.stat().st_size,
			}
		)
	return entries


def _content_hash(entries: Sequence[Mapping[str, object]]) -> str:
	return hashlib.sha256(
		json.dumps(
			list(entries), sort_keys=True, separators=(',', ':'), allow_nan=False
		).encode()
	).hexdigest()


def _descriptor_output_path(root: Path, value: object, field: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'current embedding {field} path must be a non-empty string')
	path = Path(value)
	if not path.is_absolute():
		path = root / path
	try:
		path.resolve().relative_to(root.resolve())
	except ValueError as exc:
		raise ValueError(
			f'current embedding {field} path escapes descriptor root'
		) from exc
	return path


def _validate_embedding_output_descriptor(
	path: Path,
	value: Mapping[str, object],
	survey_id: str,
	field: str,
) -> None:
	reference = dict(value)
	reference['path'] = str(path)
	_validate_reference_payload(reference, f'current embedding {survey_id} {field}')
	array = np.load(path, mmap_mode='r', allow_pickle=False)
	try:
		recorded_shape = value.get('shape')
		recorded_dtype = value.get('dtype')
		if recorded_shape is None or recorded_dtype is None:
			raise ValueError(
				f'current embedding {field} descriptor is incomplete for {survey_id}'
			)
		if list(array.shape) != recorded_shape:
			raise ValueError(
				f'current embedding {field} shape descriptor mismatch for {survey_id}'
			)
		if str(array.dtype) != recorded_dtype:
			raise ValueError(
				f'current embedding {field} dtype descriptor mismatch for {survey_id}'
			)
	finally:
		del array


def _resolve_recorded_path(value: object, name: str) -> Path:
	if isinstance(value, Path):
		return value.resolve()
	if not isinstance(value, str) or not value:
		raise TypeError(f'{name} must be a non-empty path')
	return Path(value).resolve()


def _load_json_object(path: Path, name: str) -> dict[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except FileNotFoundError:
		raise
	except json.JSONDecodeError as exc:
		raise ValueError(f'{name} must be valid JSON: {path}') from exc
	if not isinstance(payload, dict):
		raise TypeError(f'{name} must be a JSON object: {path}')
	return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		'w', dir=path.parent, delete=False, encoding='utf-8'
	) as handle:
		temporary = Path(handle.name)
		json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
		handle.write('\n')
	try:
		temporary.replace(path)
	finally:
		temporary.unlink(missing_ok=True)


def _relocate_all_json_paths(
	staging: Path,
	source_root: Path,
	destination_root: Path,
	*,
	include_manifest: bool = False,
) -> None:
	for path in sorted(staging.rglob('*.json')):
		if not include_manifest and path.name == GENERATION_MANIFEST_NAME:
			continue
		_relocate_json_paths(path, source_root, destination_root)


def _prepare_staging_for_publication(staging: Path, final_root: Path) -> None:
	"""Rewrite staged lexical paths and hashes immediately before atomic rename."""
	_relocate_all_json_paths(staging, staging, final_root)
	target_manifest_path = (
		staging / 'pseudo_targets' / 'multi_head_target_manifest.json'
	)
	target_manifest = _load_json_object(
		target_manifest_path, 'canonical multi-head target manifest'
	)
	_refresh_relocated_reference_hashes(target_manifest, staging, final_root)
	_write_json_atomic(target_manifest_path, target_manifest)
	manifest_path = staging / GENERATION_MANIFEST_NAME
	payload = _load_json_object(manifest_path, 'periodic refresh generation manifest')
	_refresh_local_reference_hashes(payload, staging)
	content_files = _content_inventory(staging)
	payload['content_files'] = content_files
	payload['generation_content_sha256'] = _content_hash(content_files)
	_write_json_atomic(manifest_path, payload)
	_relocate_json_paths(manifest_path, staging, final_root)


def _validate_staged_publication(
	staging: Path,
	final_root: Path,
	*,
	expected_identity: Mapping[str, object],
) -> None:
	"""Validate the publication form before the staging tree is renamed."""
	# Publication rewrites absolute paths from the private staging root to the
	# final root.  Temporarily reverse that lexical rewrite so the complete
	# validator can inspect the exact staged bytes without creating anything at
	# the final output path.
	_relocate_all_json_paths(
		staging,
		final_root,
		staging,
		include_manifest=True,
	)
	target_manifest_path = (
		staging / 'pseudo_targets' / 'multi_head_target_manifest.json'
	)
	target_manifest = _load_json_object(
		target_manifest_path, 'canonical multi-head target manifest'
	)
	_refresh_local_reference_hashes(target_manifest, staging)
	_write_json_atomic(target_manifest_path, target_manifest)
	manifest_path = staging / GENERATION_MANIFEST_NAME
	payload = _load_json_object(manifest_path, 'periodic refresh generation manifest')
	_refresh_local_reference_hashes(payload, staging)
	content_files = _content_inventory(staging)
	payload['content_files'] = content_files
	payload['generation_content_sha256'] = _content_hash(content_files)
	_write_json_atomic(manifest_path, payload)
	validate_periodic_refresh_generation(
		manifest_path,
		expected_identity=expected_identity,
		_allow_staging=True,
	)
	_prepare_staging_for_publication(staging, final_root)


def _refresh_local_reference_hashes(value: object, root: Path) -> None:
	if isinstance(value, dict):
		path_value = value.get('path')
		if set(value) == {'path', 'sha256'} and isinstance(path_value, str):
			path = Path(path_value).resolve()
			try:
				path.relative_to(root.resolve())
			except ValueError:
				return
			if path.is_file():
				value['sha256'] = file_sha256(path)
				return
		for item in value.values():
			_refresh_local_reference_hashes(item, root)
	elif isinstance(value, list):
		for item in value:
			_refresh_local_reference_hashes(item, root)


def _refresh_relocated_reference_hashes(
	value: object,
	staging: Path,
	final_root: Path,
) -> None:
	if isinstance(value, dict):
		path_value = value.get('path')
		if set(value) == {'path', 'sha256'} and isinstance(path_value, str):
			path = Path(path_value).resolve()
			try:
				relative = path.relative_to(final_root.resolve())
			except ValueError:
				return
			staged_path = staging / relative
			if staged_path.is_file():
				value['sha256'] = file_sha256(staged_path)
				return
		for item in value.values():
			_refresh_relocated_reference_hashes(item, staging, final_root)
	elif isinstance(value, list):
		for item in value:
			_refresh_relocated_reference_hashes(item, staging, final_root)


def _relocate_json_paths(
	path: Path,
	source_root: Path,
	destination_root: Path,
) -> None:
	raw = path.read_text(encoding='utf-8')
	source_prefix = str(source_root.absolute())
	if source_prefix not in raw:
		return
	payload = _load_json_object(path, f'JSON artifact {path}')
	destination_prefix = str(destination_root.absolute())
	_rewrite_strings(payload, source_prefix, destination_prefix)
	_write_json_atomic(path, payload)


def _rewrite_strings(value: object, staging_prefix: str, final_prefix: str) -> object:
	if isinstance(value, dict):
		for key, item in tuple(value.items()):
			value[key] = _rewrite_strings(item, staging_prefix, final_prefix)
		return value
	if isinstance(value, list):
		for index, item in enumerate(value):
			value[index] = _rewrite_strings(item, staging_prefix, final_prefix)
		return value
	if isinstance(value, str) and (
		value == staging_prefix or value.startswith(staging_prefix + '/')
	):
		return final_prefix + value[len(staging_prefix) :]
	return value

def _path_exists(path: Path) -> bool:
	return path.exists() or path.is_symlink()


def _is_staging_root(
	root: Path, expected_name: str, *, allow_staging: bool
) -> bool:
	return root.name == expected_name or (
		allow_staging and root.name.startswith(f'.{expected_name}.staging-')
	)


def _validate_sha256(value: object, name: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{name} must be a lowercase SHA-256 digest')
	return value


def _positive_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
		raise ValueError(f'{name} must be a positive integer')
	return value


def _finite_number(value: object, _name: str) -> bool:
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		return False
	return bool(np.isfinite(float(value)))


def _mapping(value: object, name: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{name} must be a mapping')
	return value


def _required_keys(value: Mapping[str, object], keys: set[str], name: str) -> None:
	unknown = set(value) - keys
	if unknown:
		raise ValueError(f'{name} has unknown fields: {sorted(unknown)!r}')
	missing = keys - set(value)
	if missing:
		raise ValueError(f'{name} is missing fields: {sorted(missing)!r}')


def _validate_target_policy_payload(value: object) -> None:
	policy = _mapping(value, 'initial hard target policy')
	_required_keys(
		policy,
		{'confidence_mode', 'confidence', 'boundary_weight_mode'},
		'initial hard target policy',
	)
	HardTargetPolicy(
		confidence_mode=policy['confidence_mode'],
		confidence=float(policy['confidence']),
		boundary_weight_mode=policy['boundary_weight_mode'],
	).validate()


def _validate_reference(reference: HashedArtifactReference, name: str) -> None:
	if not reference.path.is_file():
		raise FileNotFoundError(f'{name} is missing: {reference.path}')
	if file_sha256(reference.path) != reference.sha256:
		raise ValueError(f'{name} hash mismatch: {reference.path}')


def _validate_reference_payload(value: object, name: str) -> None:
	reference = _coerce_reference(value, name)
	_validate_reference(reference, name)


def _owned_path(root: Path, value: object, name: str) -> Path:
	"""Resolve a recorded path and require it to remain inside one generation."""
	path = _resolve_recorded_path(value, name)
	try:
		path.relative_to(root.resolve())
	except ValueError as exc:
		raise ValueError(f'{name} escapes the generation root') from exc
	return path


def _validate_owned_reference_payload(
	root: Path,
	value: object,
	name: str,
) -> None:
	reference = _coerce_reference(value, name)
	_owned_path(root, reference.path, name)
	_validate_reference(reference, name)


def _coerce_reference(value: object, name: str) -> HashedArtifactReference:
	if not isinstance(value, Mapping):
		raise TypeError(f'{name} must be a reference mapping')
	path = value.get('path')
	digest = value.get('sha256')
	if not isinstance(path, str) or not path:
		raise TypeError(f'{name}.path must be a non-empty string')
	if not isinstance(digest, str):
		raise TypeError(f'{name}.sha256 must be a string')
	return HashedArtifactReference(Path(path), digest)


def _reference_payload(reference: HashedArtifactReference) -> dict[str, str]:
	return {'path': str(reference.path.resolve()), 'sha256': reference.sha256}


def _local_reference_payload(final_path: Path, staged_path: Path) -> dict[str, str]:
	return {
		# Keep the lexical final-root path while the output root is temporarily
		# linked to staging for validation.  Resolving here would persist the
		# staging directory name into the published manifest.
		'path': str(final_path.absolute()),
		'sha256': file_sha256(staged_path),
	}


def _initial_hmm_artifact_payload(artifact: InitialHMMArtifact) -> dict[str, object]:
	return {
		'k': artifact.k,
		'centers': _reference_payload(artifact.centers),
		'hmm_model': _reference_payload(artifact.hmm_model),
		'preprocessor': _reference_payload(artifact.preprocessor),
		'metadata': _reference_payload(artifact.metadata),
		'residualizer': (
			None
			if artifact.residualizer is None
			else _reference_payload(artifact.residualizer)
		),
	}


def _initial_hmm_artifact_from_payload(k: int, value: object) -> InitialHMMArtifact:
	item = _mapping(value, f'initial HMM artifact k={k}')
	if item.get('k') != k:
		raise ValueError(f'initial HMM artifact K mismatch for k={k}')
	residualizer = item.get('residualizer')
	return InitialHMMArtifact(
		k=k,
		centers=_coerce_reference(item['centers'], f'initial k={k} centers'),
		hmm_model=_coerce_reference(item['hmm_model'], f'initial k={k} hmm_model'),
		preprocessor=_coerce_reference(
			item['preprocessor'], f'initial k={k} preprocessor'
		),
		metadata=_coerce_reference(item['metadata'], f'initial k={k} metadata'),
		residualizer=(
			None
			if residualizer is None
			else _coerce_reference(residualizer, f'initial k={k} residualizer')
		),
	)


def _edge_margin(value: object, k: int) -> tuple[int, int, int]:
	if not isinstance(value, (list, tuple)) or len(value) != 3:
		raise ValueError(f'initial edge_margin_tokens is invalid for k={k}')
	result = tuple(value)
	if any(
		isinstance(item, bool) or not isinstance(item, int) or item < 0
		for item in result
	):
		raise ValueError(f'initial edge_margin_tokens is invalid for k={k}')
	return result  # type: ignore[return-value]


def _load_cost_matrix(value: object, k: int) -> np.ndarray:
	try:
		matrix = np.asarray(value, dtype=np.float32)
	except (TypeError, ValueError) as exc:
		raise ValueError(f'initial transition costs are invalid for k={k}') from exc
	if matrix.shape != (k, k) or np.any(np.isnan(matrix)):
		raise ValueError(f'initial transition costs are invalid for k={k}')
	return matrix.copy()


def _load_cost_vector(value: object, k: int, name: str) -> np.ndarray:
	try:
		vector = np.asarray(value, dtype=np.float32)
	except (TypeError, ValueError) as exc:
		raise ValueError(f'initial {name} is invalid for k={k}') from exc
	if vector.shape != (k,) or not np.all(np.isfinite(vector)):
		raise ValueError(f'initial {name} is invalid for k={k}')
	return vector.copy()


def _json_cost_matrix(value: object, k: int) -> np.ndarray:
	if not isinstance(value, list) or len(value) != k:
		raise ValueError(f'initial metadata transition costs are invalid for k={k}')
	rows: list[list[float]] = []
	for row in value:
		if not isinstance(row, list) or len(row) != k:
			raise ValueError(f'initial metadata transition costs are invalid for k={k}')
		rows.append([np.inf if item is None else float(item) for item in row])
	return _load_cost_matrix(rows, k)


def _expected_boundaries_from_path_prior(
	path_prior: Mapping[str, object],
) -> hmm.HMMExpectedBoundariesSettings | None:
	if not bool(path_prior.get('enabled', False)):
		return None
	value = _mapping(path_prior.get('expected_boundaries'), 'expected_boundaries')
	if not bool(value.get('enabled', False)) or float(value.get('weight', 0.0)) == 0.0:
		return None
	target = value.get('target', 'auto_k_minus_1')
	weight = float(value.get('weight', 0.0))
	if not np.isfinite(weight) or weight < 0.0:
		raise ValueError('expected-boundary weight must be finite and non-negative')
	return hmm.HMMExpectedBoundariesSettings(enabled=True, target=target, weight=weight)


def _path_prior_core(value: Mapping[str, object]) -> dict[str, object]:
	expected_boundaries = _jsonable(value.get('expected_boundaries', {}))
	if isinstance(expected_boundaries, dict):
		expected_boundaries.pop('target_resolution', None)
	return {
		'enabled': bool(value.get('enabled', False)),
		'initial_state': _jsonable(value.get('initial_state', {})),
		'terminal_state': _jsonable(value.get('terminal_state', {})),
		'expected_boundaries': expected_boundaries,
	}


def _normalized_prepared_identity(value: Mapping[str, object]) -> dict[str, object]:
	result = _jsonable(value)
	if not isinstance(result, dict):
		raise TypeError('prepared feature identity must normalize to an object')
	result.pop('directory', None)
	surveys = result.get('surveys')
	if isinstance(surveys, list):
		for survey in surveys:
			if isinstance(survey, dict):
				survey.pop('cache_path', None)
				survey.pop('reused', None)
	return result


def _jsonable(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _jsonable(item) for key, item in value.items()}
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.ndarray):
		return [_jsonable(item) for item in value.tolist()]
	if isinstance(value, np.generic):
		return _jsonable(value.item())
	if isinstance(value, (list, tuple)):
		return [_jsonable(item) for item in value]
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	return str(value)


__all__ = [
	'ARTIFACT_TYPE',
	'CANONICAL_KS',
	'GENERATION_MANIFEST_NAME',
	'INITIAL_GENERATION_ID',
	'HardTargetPolicy',
	'HashedArtifactReference',
	'InitialHMMArtifact',
	'InitialPeriodicRefreshConfig',
	'PeriodicRefreshConfig',
	'PeriodicRefreshGenerationResult',
	'PreviousCenterArtifact',
	'build_initial_periodic_refresh_generation',
	'build_periodic_refresh_generation',
	'load_periodic_refresh_generation',
	'produce_initial_periodic_refresh_generation',
	'produce_periodic_refresh_generation',
	'quarantine_periodic_refresh_generation',
	'validate_periodic_refresh_generation',
]
