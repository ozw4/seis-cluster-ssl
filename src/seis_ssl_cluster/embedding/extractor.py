"""Full-volume amplitude MAE encoder embedding extraction."""


from __future__ import annotations

import hashlib
import json
import queue
import shutil
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import numpy as np
import torch

from seis_ssl_cluster.config.pretraining import (
	resolve_barlow_twins_training_config,
)
from seis_ssl_cluster.config.schema import (
	DEFAULT_MAE_DATA_OPTIONS,
	DEFAULT_MAE_TRAIN_OPTIONS,
	DEFAULT_ZERO_MASK_CONTRACT,
	FIXED_DATA_CONTRACT,
	FIXED_LOSS_CONTRACT,
	FIXED_MASKING_CONTRACT,
	FIXED_MODEL_CONTRACT,
	STAGE_BARLOW_TWINS_TRAINING,
	STAGE_MAE_TRAINING,
	SUPPORTED_RECONSTRUCTION_LOSSES,
	SUPPORTED_TARGET_NORMALIZATION_MODES,
)
from seis_ssl_cluster.data.normalization import (
	AmplitudeAgcConfig,
	SurveyNormalizationStats,
	load_normalization_stats,
)
from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest, read_manifest_json
from seis_ssl_cluster.data.survey_preprocessing_cache import (
	PreparedSurveyAmplitude,
	SurveyPreprocessingCacheMode,
	SurveyPreprocessingCachePlan,
	SurveyPreprocessingCacheSettings,
	plan_survey_preprocessing_cache,
	prepare_survey_preprocessing_cache,
)
from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
from seis_ssl_cluster.data.window_preprocessing import (
	AmplitudePreprocessSettings,
	FiniteCheckMode,
	read_amplitude_crop,
	read_prepared_survey_amplitude_crop,
	reduce_valid_mask_to_tokens,
	resolve_manifest_path,
)
from seis_ssl_cluster.data.zero_mask import ZeroMaskConfig
from seis_ssl_cluster.embedding.merge import EmbeddingMerger
from seis_ssl_cluster.embedding.sliding_window import (
	SlidingWindow,
	iter_sliding_windows,
	token_grid_shape_xyz,
)
from seis_ssl_cluster.embedding.writer import (
	cleanup_temp_outputs,
	commit_staged_outputs,
	create_merge_memmaps,
	file_sha256,
	output_paths,
	prepare_outputs,
	write_metadata,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	CHECKPOINT_KIND as BARLOW_TWINS_CHECKPOINT_KIND,
)
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	PRETRAINING_METHOD as BARLOW_TWINS_PRETRAINING_METHOD,
)
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	TRAINED_PARAMETER_PREFIXES as BARLOW_TWINS_TRAINED_PARAMETER_PREFIXES,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)
from seis_ssl_cluster.utils import StageTimer
from seis_ssl_cluster.utils.cuda import cuda_device_supports_bfloat16

UNMASKED_ENCODER_INPUT_MODE = 'unmasked_encoder_tokens_v1'
CURRENT_STUDENT_UNMASKED_EMBEDDING_SEMANTICS = (
	'current_student_unmasked_eval_full_survey_v1'
)
REFRESH_EXTRACTION_DESCRIPTOR_NAME = 'refresh_extraction_descriptor.json'

XYZ = tuple[int, int, int]
_CHECKPOINT_ALLOWED_TOP_LEVEL = frozenset(
	{
		'stage',
		'paths',
		'manifests',
		'data',
		'model',
		'masking',
		'loss',
		'train',
		'zero_mask',
		'visualization',
	},
)
_CHECKPOINT_REQUIRED_TOP_LEVEL = frozenset(
	{
		'stage',
		'paths',
		'manifests',
		'data',
		'model',
		'masking',
		'loss',
		'train',
		'zero_mask',
	},
)
_BARLOW_TWINS_CHECKPOINT_ALLOWED_TOP_LEVEL = frozenset(
	{
		'stage',
		'paths',
		'manifests',
		'data',
		'zero_mask',
		'model',
		'augmentations',
		'barlow_twins',
		'train',
	}
)
_BARLOW_TWINS_CHECKPOINT_REQUIRED_TOP_LEVEL = (
	_BARLOW_TWINS_CHECKPOINT_ALLOWED_TOP_LEVEL
)
_CHECKPOINT_MODEL_GEOMETRY_KEYS = (
	'patch_size',
	'encoder_dim',
	'encoder_depth',
	'encoder_heads',
	'decoder_dim',
	'decoder_depth',
	'decoder_heads',
)
_CHECKPOINT_MASKING_KEYS = ('spatial_mask_ratio', 'block_size_tokens')
_CHECKPOINT_TRAIN_REQUIRED_KEYS = (
	'batch_size',
	'samples_per_epoch',
	'epochs',
	*(
		key
		for key in DEFAULT_MAE_TRAIN_OPTIONS
		if key
		not in {
			'amp_dtype',
			'persistent_workers',
			'prefetch_factor',
			'runtime_check_mode',
			'stage_timing',
		}
	),
)


@dataclass(frozen=True)
class EmbeddingExtractionSettings:
	"""Validated full-volume extraction settings."""

	checkpoint_path: Path
	output_dir: Path
	window_size_xyz: XYZ
	overlap_xyz: XYZ
	output_dtype: np.dtype
	average_chunk_size_x: int
	batch_size: int
	prefetch_queue_depth: int
	amp: bool
	amp_dtype: str
	stage_timing: bool
	min_token_valid_fraction: float
	zero_mask: ZeroMaskConfig
	normalized_clip_abs: float | None
	amplitude_agc: AmplitudeAgcConfig
	finite_check_mode: FiniteCheckMode
	preprocessing_cache: SurveyPreprocessingCacheSettings


@dataclass(frozen=True)
class SurveyEmbeddingResult:
	"""Result for one survey extraction."""

	survey_id: str
	embeddings_path: Path
	valid_tokens_path: Path
	metadata_path: Path
	skipped: bool


def run_embedding_extraction(
	config: Mapping[str, object],
	*,
	skip_existing: bool = False,
	device: str | torch.device | None = None,
	checkpoint_config_override: Mapping[str, object] | None = None,
) -> list[SurveyEmbeddingResult]:
	"""Extract MAE encoder embeddings for all surveys in a manifest.

	``checkpoint_config_override`` is a library-only, fully resolved checkpoint
	config replacement.  It exists for audited migration readers of legacy
	checkpoints whose serialized config is incomplete.  Callers are responsible
	for validating the override's provenance and scientific identity; ordinary
	extraction must leave it as ``None`` and uses the serialized checkpoint
	config unchanged.
	"""
	checkpoint_path = _checkpoint_path(config)
	manifests = read_manifest_json(_manifest_path(config))
	if not manifests:
		msg = 'embedding extraction manifest is empty'
		raise ValueError(msg)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	validate_stratigraphy_checkpoint_payload(payload)
	checkpoint_config = _checkpoint_config(payload)
	if checkpoint_config_override is not None:
		checkpoint_config = _checkpoint_config_override(checkpoint_config_override)
	settings = extraction_settings_from_config(
		config,
		checkpoint_config=checkpoint_config,
	)
	checkpoint_sha256 = file_sha256(settings.checkpoint_path)
	model = build_model_from_checkpoint_payload(payload)
	checkpoint_dtype = _model_floating_dtype(model)
	resolved_device = _resolve_device(device, config)
	model.to(device=resolved_device, dtype=checkpoint_dtype)
	model.eval()

	store = NpyMemmapVolumeStore()
	timer = StageTimer(
		enabled=settings.stage_timing,
		synchronize=(
			partial(torch.cuda.synchronize, resolved_device)
			if resolved_device.type == 'cuda'
			else None
		),
	)
	producer_timer = StageTimer(
		enabled=settings.stage_timing,
		accumulator=timer.accumulator,
	)
	try:
		results = [
			extract_survey_embeddings(
				manifest,
				model=model,
				store=store,
				settings=settings,
				checkpoint_config=checkpoint_config,
				checkpoint_payload=payload,
				checkpoint_sha256=checkpoint_sha256,
				device=resolved_device,
				skip_existing=skip_existing,
				timer=timer,
				producer_timer=producer_timer,
			)
			for manifest in manifests
		]
		_write_embedding_execution_summary(settings.output_dir, results)
		return results
	finally:
		if settings.stage_timing:
			timer.write_json(settings.output_dir / 'stage_timings.json')


def extract_embeddings_from_loaded_model(  # noqa: C901, PLR0913, PLR0915
	student: AmplitudeMAE3D,
	config: Mapping[str, object],
	output_dir: str | Path,
	source_student_state_sha256: str,
	*,
	checkpoint_config: Mapping[str, object] | None = None,
	reuse: bool = False,
	overwrite: bool = False,
	device: str | torch.device | None = None,
) -> list[SurveyEmbeddingResult]:
	"""Extract a generation-scoped embedding from a live MAE student.

	The resolved MAE configuration is supplied as ``checkpoint_config`` so this
	boundary never has to read a checkpoint.  ``config`` remains the standard
	extraction configuration and provides the manifest and extraction geometry.

	The student is used only through its normal unmasked ``encode_tokens`` path.
	Its module training flags and PyTorch CPU/CUDA RNG states are restored on
	both success and failure.
	"""
	if not isinstance(student, AmplitudeMAE3D):
		msg = 'student must be an AmplitudeMAE3D instance'
		raise TypeError(msg)
	_validate_bool_argument(reuse, 'reuse')
	_validate_bool_argument(overwrite, 'overwrite')
	if reuse and overwrite:
		msg = 'reuse and overwrite cannot both be true'
		raise ValueError(msg)
	student_state_sha256 = _validate_sha256(
		source_student_state_sha256,
		'source_student_state_sha256',
	)
	standard_config, resolved_checkpoint_config = _loaded_model_config_pair(
		config,
		checkpoint_config=checkpoint_config,
	)
	settings = extraction_settings_from_config(
		standard_config,
		checkpoint_config=resolved_checkpoint_config,
	)
	manifests = read_manifest_json(_manifest_path(standard_config))
	if not manifests:
		msg = 'loaded-model embedding extraction manifest is empty'
		raise ValueError(msg)
	_validate_loaded_model_config(student, resolved_checkpoint_config)
	resolved_device = _loaded_model_device(student, device, standard_config)
	output_root = Path(output_dir)
	identity = _refresh_identity(
		standard_config=standard_config,
		resolved_checkpoint_config=resolved_checkpoint_config,
		manifests=manifests,
		student_state_sha256=student_state_sha256,
		model=student,
	)
	existing = _inspect_existing_refresh_output(
		output_root,
		identity=identity,
		manifests=manifests,
		model=student,
		settings=settings,
		overwrite=overwrite,
	)
	if existing is not None:
		if reuse:
			return existing
		if overwrite:
			existing = None
	if existing is not None:
		msg = (
			'complete refresh output already exists; set reuse=True or '
			'overwrite=True: '
			f'{output_root}'
		)
		raise FileExistsError(msg)
	if output_root.exists() and not _is_empty_directory(output_root) and not overwrite:
		msg = (
			'existing refresh output is incomplete or does not match the '
			'requested identity: '
			f'{output_root}'
		)
		raise ValueError(msg)
	output_root.parent.mkdir(parents=True, exist_ok=True)
	staging = output_root.with_name(
		f'.{output_root.name}.staging-{uuid.uuid4().hex}'
	)
	staging.mkdir()
	try:
		staged_cache = replace(
			settings.preprocessing_cache,
			directory=staging / '.preprocessing_cache',
		)
		staged_settings = replace(
			settings,
			output_dir=staging,
			preprocessing_cache=staged_cache,
		)
		timer = StageTimer(
			enabled=staged_settings.stage_timing,
			synchronize=(
				partial(torch.cuda.synchronize, resolved_device)
				if resolved_device.type == 'cuda'
				else None
			),
		)
		producer_timer = StageTimer(
			enabled=staged_settings.stage_timing,
			accumulator=timer.accumulator,
		)
		store = NpyMemmapVolumeStore()
		with _preserve_student_runtime_state(student):
			student.eval()
			results = [
				extract_survey_embeddings(
					manifest,
					model=student,
					store=store,
					settings=staged_settings,
					checkpoint_config=resolved_checkpoint_config,
					checkpoint_payload=None,
					checkpoint_sha256=student_state_sha256,
					device=resolved_device,
					skip_existing=False,
					timer=timer,
					producer_timer=producer_timer,
				)
				for manifest in manifests
			]
			_write_embedding_execution_summary(staging, results)
		if staged_settings.stage_timing:
			timer.write_json(staging / 'stage_timings.json')
		if _refresh_identity(
			standard_config=standard_config,
			resolved_checkpoint_config=resolved_checkpoint_config,
			manifests=manifests,
			student_state_sha256=student_state_sha256,
			model=student,
		) != identity:
			msg = 'refresh source identity changed during extraction'
			raise ValueError(msg)
		descriptor = _build_refresh_descriptor(
			staging,
			identity=identity,
			manifests=manifests,
			results=results,
			model=student,
			settings=staged_settings,
		)
		write_metadata(staging / REFRESH_EXTRACTION_DESCRIPTOR_NAME, descriptor)
		_validate_refresh_descriptor(
			staging,
			descriptor=descriptor,
			identity=identity,
			manifests=manifests,
			model=student,
			settings=staged_settings,
		)
		_publish_refresh_staging(staging, output_root, overwrite=overwrite)
		return [
			_rebase_survey_result(result, output_root, skipped=False)
			for result in results
		]
	finally:
		if staging.exists():
			shutil.rmtree(staging, ignore_errors=True)


def _loaded_model_config_pair(
	config: Mapping[str, object],
	*,
	checkpoint_config: Mapping[str, object] | None,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
	if not isinstance(config, Mapping):
		msg = 'config must be a mapping'
		raise TypeError(msg)
	if checkpoint_config is None:
		msg = (
			'loaded-model extraction requires an explicit resolved checkpoint_config; '
			'no checkpoint file is read at this boundary'
		)
		raise ValueError(msg)
	if not isinstance(checkpoint_config, Mapping):
		msg = 'checkpoint_config must be a mapping'
		raise TypeError(msg)
	return config, checkpoint_config


def _validate_bool_argument(value: object, name: str) -> None:
	if not isinstance(value, bool):
		msg = f'{name} must be a boolean'
		raise TypeError(msg)


def _validate_sha256(value: object, name: str) -> str:
	if not isinstance(value, str) or len(value) != 64:
		msg = f'{name} must be a lowercase SHA-256 digest'
		raise TypeError(msg)
	if any(character not in '0123456789abcdef' for character in value):
		msg = f'{name} must be a lowercase SHA-256 digest'
		raise ValueError(msg)
	return value


def _validate_loaded_model_config(
	model: AmplitudeMAE3D,
	checkpoint_config: Mapping[str, object],
) -> None:
	model_geometry = _model_geometry(checkpoint_config, model)
	configured_model = _required_mapping(checkpoint_config, 'model')
	for key in (
		'name',
		'in_channels',
		'out_channels',
		'encoder_dim',
		'encoder_depth',
		'encoder_heads',
		'decoder_dim',
		'decoder_depth',
		'decoder_heads',
	):
		configured = configured_model.get(key)
		actual = model_geometry[key]
		if configured != actual:
			msg = (
				f'loaded student model geometry differs from checkpoint_config.model.'
				f'{key}: configured={configured!r}, actual={actual!r}'
			)
			raise ValueError(msg)
	if list(configured_model['patch_size']) != model_geometry['patch_size']:
		msg = (
			'loaded student model geometry differs from '
			'checkpoint_config.model.patch_size'
		)
		raise ValueError(msg)


def _loaded_model_device(
	model: AmplitudeMAE3D,
	requested: str | torch.device | None,
	config: Mapping[str, object],
) -> torch.device:
	devices = {tensor.device for tensor in (*model.parameters(), *model.buffers())}
	if len(devices) != 1:
		msg = (
			'loaded student tensors must share one device; got '
			f'{sorted(map(str, devices))}'
		)
		raise ValueError(msg)
	actual = next(iter(devices))
	if requested is None:
		return actual
	resolved = _resolve_device(requested, config)
	if resolved.type != actual.type or (
		resolved.index is not None and resolved.index != actual.index
	):
		msg = (
			'loaded student device does not match extraction device: '
			f'model={actual}, requested={resolved}'
		)
		raise ValueError(msg)
	return actual


@contextmanager
def _preserve_student_runtime_state(model: AmplitudeMAE3D) -> Iterator[None]:
	training_flags = tuple((module, module.training) for module in model.modules())
	cpu_rng_state = torch.random.get_rng_state()
	cuda_rng_states = (
		torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
	)
	try:
		yield
	finally:
		try:
			torch.random.set_rng_state(cpu_rng_state)
			if cuda_rng_states is not None:
				torch.cuda.set_rng_state_all(cuda_rng_states)
		finally:
			for module, training in training_flags:
				module.training = training


def _refresh_identity(
	*,
	standard_config: Mapping[str, object],
	resolved_checkpoint_config: Mapping[str, object],
	manifests: Sequence[SurveyManifest],
	student_state_sha256: str,
	model: AmplitudeMAE3D,
) -> dict[str, object]:
	manifest_path = _manifest_path(standard_config).resolve()
	path_list = _source_path_list_identity(
		standard_config,
		resolved_checkpoint_config,
		manifests,
	)
	normalized_config = {
		'standard_embedding_config': _jsonable(standard_config),
		'resolved_checkpoint_config': _jsonable(resolved_checkpoint_config),
	}
	normalized_config_sha256 = _canonical_json_sha256(normalized_config)
	normalization: dict[str, object] = {}
	for manifest in manifests:
		stats_path = resolve_manifest_path(
			manifest,
			manifest.amplitude.normalization_stats_path,
		).resolve()
		normalization[manifest.survey_id] = {
			'path': str(stats_path),
			'sha256': file_sha256(stats_path),
		}
	return {
		'source_student_state_sha256': student_state_sha256,
		'embedding_semantics': CURRENT_STUDENT_UNMASKED_EMBEDDING_SEMANTICS,
		'extraction_config': {
			'identity_type': 'normalized_json_v1',
			'sha256': normalized_config_sha256,
			'normalized': normalized_config,
		},
		'source_manifest': {
			'path': str(manifest_path),
			'sha256': file_sha256(manifest_path),
		},
		'source_path_list': path_list,
		'source_normalization': normalization,
		'survey_ids': [manifest.survey_id for manifest in manifests],
		'model_geometry': _model_geometry(resolved_checkpoint_config, model),
	}


def _source_path_list_identity(
	standard_config: Mapping[str, object],
	resolved_checkpoint_config: Mapping[str, object],
	manifests: Sequence[SurveyManifest],
) -> dict[str, object]:
	for config in (standard_config, resolved_checkpoint_config):
		manifests_config = _optional_mapping(config, 'manifests')
		for key in ('path_list', 'train_path_list', 'input_path_list'):
			value = manifests_config.get(key)
			if value is None:
				continue
			if not isinstance(value, str) or not value:
				msg = f'manifests.{key} must be a non-empty path'
				raise TypeError(msg)
			path = Path(value).resolve()
			return {'path': str(path), 'sha256': file_sha256(path)}
	entries = [
		str(resolve_manifest_path(manifest, manifest.amplitude.path).resolve())
		for manifest in manifests
	]
	canonical = ''.join(f'{entry}\n' for entry in entries).encode('utf-8')
	return {
		'path': None,
		'sha256': hashlib.sha256(canonical).hexdigest(),
		'identity_type': 'manifest_amplitude_paths_v1',
		'entry_count': len(entries),
	}


def _jsonable(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _jsonable(item) for key, item in value.items()}
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.generic):
		return _jsonable(value.item())
	if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
		return [_jsonable(item) for item in value]
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	msg = f'cannot normalize extraction config value of type {type(value)!r}'
	raise TypeError(msg)


def _canonical_json_sha256(value: object) -> str:
	return hashlib.sha256(
		json.dumps(
			value,
			allow_nan=False,
			sort_keys=True,
			separators=(',', ':'),
		).encode('utf-8'),
	).hexdigest()


def _build_refresh_descriptor(  # noqa: PLR0913
	root: Path,
	*,
	identity: Mapping[str, object],
	manifests: Sequence[SurveyManifest],
	results: Sequence[SurveyEmbeddingResult],
	model: AmplitudeMAE3D,
	settings: EmbeddingExtractionSettings,
) -> dict[str, object]:
	if len(results) != len(manifests):
		msg = 'refresh extraction result count does not match manifest count'
		raise ValueError(msg)
	manifest_by_id = {manifest.survey_id: manifest for manifest in manifests}
	outputs: dict[str, object] = {}
	for result in results:
		manifest = manifest_by_id.get(result.survey_id)
		if manifest is None or result.skipped:
			msg = 'refresh extraction produced an invalid survey result'
			raise ValueError(msg)
		outputs[result.survey_id] = _refresh_output_entry(
			root,
			result=result,
			manifest=manifest,
			model=model,
			settings=settings,
		)
	return {
		'artifact_type': 'embedding_refresh_extraction',
		'schema_version': 1,
		'status': 'COMPLETE',
		'completion_status': 'COMPLETE',
		**dict(identity),
		'outputs': outputs,
	}


def _refresh_output_entry(
	root: Path,
	*,
	result: SurveyEmbeddingResult,
	manifest: SurveyManifest,
	model: AmplitudeMAE3D,
	settings: EmbeddingExtractionSettings,
) -> dict[str, object]:
	paths = output_paths(root, result.survey_id)
	expected_grid = token_grid_shape_xyz(
		manifest.amplitude.shape_xyz,
		model.patch_size_xyz,
	)
	embeddings = _refresh_array_descriptor(
		root,
		paths.embeddings,
		expected_shape=(*expected_grid, model.encoder_dim),
		expected_dtype=settings.output_dtype,
		label=f'{result.survey_id}.embeddings',
	)
	valid_tokens = _refresh_array_descriptor(
		root,
		paths.valid_tokens,
		expected_shape=expected_grid,
		expected_dtype=np.dtype(bool),
		label=f'{result.survey_id}.valid_tokens',
	)
	metadata = _refresh_metadata_descriptor(
		root,
		paths.metadata,
		survey_id=result.survey_id,
	)
	return {
		'embeddings': embeddings,
		'valid_tokens': valid_tokens,
		'metadata': metadata,
	}


def _refresh_array_descriptor(
	root: Path,
	path: Path,
	*,
	expected_shape: tuple[int, ...],
	expected_dtype: np.dtype,
	label: str,
) -> dict[str, object]:
	if not path.is_file():
		msg = f'missing refresh output array: {path}'
		raise FileNotFoundError(msg)
	array = np.load(path, mmap_mode='r', allow_pickle=False)
	try:
		if tuple(array.shape) != expected_shape:
			msg = (
				f'{label} shape mismatch: expected {expected_shape!r}, '
				f'got {tuple(array.shape)!r}'
			)
			raise ValueError(msg)
		if np.dtype(array.dtype) != expected_dtype:
			msg = (
				f'{label} dtype mismatch: expected {expected_dtype!s}, '
				f'got {array.dtype!s}'
			)
			raise ValueError(msg)
		return {
			'path': _relative_output_path(root, path),
			'sha256': file_sha256(path),
			'shape': list(array.shape),
			'dtype': str(array.dtype),
		}
	finally:
		del array


def _refresh_metadata_descriptor(
	root: Path,
	path: Path,
	*,
	survey_id: str,
) -> dict[str, object]:
	if not path.is_file():
		msg = f'missing refresh embedding metadata: {path}'
		raise FileNotFoundError(msg)
	try:
		metadata = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as exc:
		msg = f'invalid refresh embedding metadata: {path}'
		raise ValueError(msg) from exc
	if not isinstance(metadata, Mapping) or metadata.get('survey_id') != survey_id:
		msg = f'refresh embedding metadata survey mismatch: {path}'
		raise ValueError(msg)
	return {
		'path': _relative_output_path(root, path),
		'sha256': file_sha256(path),
	}


def _relative_output_path(root: Path, path: Path) -> str:
	try:
		return path.resolve().relative_to(root.resolve()).as_posix()
	except ValueError as exc:
		msg = f'refresh output path is outside generation root: {path}'
		raise ValueError(msg) from exc


def _validate_refresh_descriptor(  # noqa: C901, PLR0912, PLR0913
	root: Path,
	*,
	descriptor: Mapping[str, object],
	identity: Mapping[str, object],
	manifests: Sequence[SurveyManifest],
	model: AmplitudeMAE3D,
	settings: EmbeddingExtractionSettings,
) -> None:
	if descriptor.get('artifact_type') != 'embedding_refresh_extraction':
		msg = 'refresh descriptor artifact_type is invalid'
		raise ValueError(msg)
	if descriptor.get('schema_version') != 1:
		msg = 'refresh descriptor schema_version is invalid'
		raise ValueError(msg)
	if descriptor.get('status') != 'COMPLETE' or descriptor.get(
		'completion_status'
	) != 'COMPLETE':
		msg = 'refresh descriptor is not complete'
		raise ValueError(msg)
	for key, expected in identity.items():
		if descriptor.get(key) != expected:
			msg = f'refresh descriptor identity mismatch: {key}'
			raise ValueError(msg)
	outputs = descriptor.get('outputs')
	if not isinstance(outputs, Mapping):
		msg = 'refresh descriptor outputs must be a mapping'
		raise TypeError(msg)
	manifest_by_id = {manifest.survey_id: manifest for manifest in manifests}
	if set(outputs) != set(manifest_by_id):
		msg = 'refresh descriptor survey output set mismatch'
		raise ValueError(msg)
	for survey_id, manifest in manifest_by_id.items():
		entry = outputs.get(survey_id)
		if not isinstance(entry, Mapping):
			msg = f'refresh descriptor output entry is invalid: {survey_id}'
			raise TypeError(msg)
		paths = output_paths(root, survey_id)
		expected_paths = {
			'embeddings': paths.embeddings,
			'valid_tokens': paths.valid_tokens,
			'metadata': paths.metadata,
		}
		for name, expected_path in expected_paths.items():
			item = entry.get(name)
			if not isinstance(item, Mapping):
				msg = f'refresh descriptor {survey_id}.{name} is invalid'
				raise TypeError(msg)
			if item.get('path') != _relative_output_path(root, expected_path):
				msg = f'refresh descriptor path mismatch: {survey_id}.{name}'
				raise ValueError(msg)
		expected_entry = _refresh_output_entry(
			root,
			result=SurveyEmbeddingResult(
				survey_id=survey_id,
				embeddings_path=paths.embeddings,
				valid_tokens_path=paths.valid_tokens,
				metadata_path=paths.metadata,
				skipped=False,
			),
			manifest=manifest,
			model=model,
			settings=settings,
		)
		if dict(entry) != expected_entry:
			msg = f'refresh descriptor output hash or layout mismatch: {survey_id}'
			raise ValueError(msg)


def _inspect_existing_refresh_output(  # noqa: PLR0913
	output_root: Path,
	*,
	identity: Mapping[str, object],
	manifests: Sequence[SurveyManifest],
	model: AmplitudeMAE3D,
	settings: EmbeddingExtractionSettings,
	overwrite: bool,
) -> list[SurveyEmbeddingResult] | None:
	if not output_root.exists():
		return None
	if output_root.is_dir() and not any(output_root.iterdir()):
		return None
	descriptor_path = output_root / REFRESH_EXTRACTION_DESCRIPTOR_NAME
	if not descriptor_path.is_file():
		return None
	try:
		descriptor = json.loads(descriptor_path.read_text(encoding='utf-8'))
		if not isinstance(descriptor, Mapping):
			msg = 'refresh descriptor must be a JSON object'
			raise TypeError(msg)  # noqa: TRY301
		_validate_refresh_descriptor(
			output_root,
			descriptor=descriptor,
			identity=identity,
			manifests=manifests,
			model=model,
			settings=settings,
		)
	except (
		OSError,
		json.JSONDecodeError,
		TypeError,
		ValueError,
		FileNotFoundError,
	) as exc:
		if overwrite:
			return None
		msg = f'existing refresh output failed complete validation: {output_root}'
		raise ValueError(msg) from exc
	return [
		_rebase_survey_result(
			SurveyEmbeddingResult(
				survey_id=manifest.survey_id,
				embeddings_path=output_paths(
					output_root,
					manifest.survey_id,
				).embeddings,
				valid_tokens_path=output_paths(
					output_root,
					manifest.survey_id,
				).valid_tokens,
				metadata_path=output_paths(
					output_root,
					manifest.survey_id,
				).metadata,
				skipped=True,
			),
			output_root,
			skipped=True,
		)
		for manifest in manifests
	]


def _rebase_survey_result(
	result: SurveyEmbeddingResult,
	output_root: Path,
	*,
	skipped: bool,
) -> SurveyEmbeddingResult:
	paths = output_paths(output_root, result.survey_id)
	return replace(
		result,
		embeddings_path=paths.embeddings,
		valid_tokens_path=paths.valid_tokens,
		metadata_path=paths.metadata,
		skipped=skipped,
	)


def _publish_refresh_staging(
	staging: Path,
	output_root: Path,
	*,
	overwrite: bool,
) -> None:
	if not staging.is_dir():
		msg = f'refresh staging directory is missing: {staging}'
		raise FileNotFoundError(msg)
	if not output_root.exists():
		staging.replace(output_root)
		return
	if not overwrite:
		msg = f'refresh output already exists: {output_root}'
		raise FileExistsError(msg)
	backup = output_root.with_name(
		f'.{output_root.name}.backup-{uuid.uuid4().hex}'
	)
	output_root.replace(backup)
	try:
		staging.replace(output_root)
	except BaseException:
		if output_root.exists():
			_remove_path(output_root)
		if backup.exists():
			backup.replace(output_root)
		raise
	with suppress(OSError):
		_remove_path(backup)


def _remove_path(path: Path) -> None:
	if path.is_dir() and not path.is_symlink():
		shutil.rmtree(path)
	else:
		path.unlink()


def _is_empty_directory(path: Path) -> bool:
	return path.is_dir() and not any(path.iterdir())


def _write_embedding_execution_summary(
	output_dir: Path, results: list[SurveyEmbeddingResult]
) -> None:
	"""Persist the fresh/reuse disposition for the completed extraction call."""
	payload = {
		'artifact_type': 'embedding_extraction_execution',
		'schema_version': 1,
		'encoder_input_mode': UNMASKED_ENCODER_INPUT_MODE,
		'fresh': sum(not result.skipped for result in results),
		'reuse': sum(result.skipped for result in results),
		'survey_count': len(results),
	}
	path = output_dir / 'embedding_extraction_execution.json'
	tmp_path = path.with_name(f'.{path.name}.tmp')
	tmp_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	tmp_path.replace(path)


def extraction_settings_from_config(
	config: Mapping[str, object],
	*,
	checkpoint_config: Mapping[str, object] | None = None,
) -> EmbeddingExtractionSettings:
	"""Build extraction settings from validated config sections."""
	if checkpoint_config is None:
		msg = 'checkpoint_config is required for embedding extraction settings'
		raise ValueError(msg)
	_reject_checkpoint_owned_extraction_sections(config)
	_validate_checkpoint_resolved_config(checkpoint_config)
	embeddings = _required_mapping(config, 'embeddings')
	embedding = _required_mapping(config, 'embedding')
	checkpoint_data = _required_mapping(checkpoint_config, 'data')
	normalized_clip_abs_value = checkpoint_data.get('normalized_clip_abs')
	normalized_clip_abs = (
		None
		if normalized_clip_abs_value is None
		else _positive_finite_number(
			normalized_clip_abs_value,
			'data.normalized_clip_abs',
		)
	)
	checkpoint_path = _required_path(embeddings, 'checkpoint', 'embeddings')
	output_dir = _required_path(embeddings, 'output_dir', 'embeddings')
	window_size = _xyz_from_mapping(
		embedding,
		'window_size',
		'embedding',
		default=None,
	)
	overlap = _nonnegative_xyz_from_mapping(
		embedding,
		'overlap',
		'embedding',
		default=None,
	)
	_validate_overlap_less_than_window(overlap, window_size)
	output_dtype_name = _required_non_empty_string(
		embedding,
		'output_dtype',
		'embedding',
	)
	try:
		output_dtype = np.dtype(output_dtype_name)
	except TypeError as exc:
		msg = 'embedding.output_dtype must be float16 or float32'
		raise ValueError(msg) from exc
	if output_dtype not in {np.dtype('float16'), np.dtype('float32')}:
		msg = 'embedding.output_dtype must be float16 or float32'
		raise ValueError(msg)
	return EmbeddingExtractionSettings(
		checkpoint_path=checkpoint_path,
		output_dir=output_dir,
		window_size_xyz=window_size,
		overlap_xyz=overlap,
		output_dtype=output_dtype,
		average_chunk_size_x=_positive_int(
			embedding.get('average_chunk_size_x', 16),
			'embedding.average_chunk_size_x',
		),
		batch_size=_positive_int(
			embedding.get('batch_size'),
			'embedding.batch_size',
		),
		prefetch_queue_depth=_nonnegative_int(
			embedding.get('prefetch_queue_depth', 0),
			'embedding.prefetch_queue_depth',
		),
		amp=_bool(embedding.get('amp', False), 'embedding.amp'),
		amp_dtype=_amp_dtype(embedding.get('amp_dtype', 'auto')),
		stage_timing=_bool(
			embedding.get('stage_timing', False),
			'embedding.stage_timing',
		),
		min_token_valid_fraction=_fraction(
			embedding.get('min_token_valid_fraction'),
			'embedding.min_token_valid_fraction',
		),
		zero_mask=_zero_mask_from_config(checkpoint_config),
		normalized_clip_abs=normalized_clip_abs,
		amplitude_agc=_amplitude_agc_from_config(checkpoint_config),
		finite_check_mode=_finite_check_mode_from_config(checkpoint_config),
		preprocessing_cache=_preprocessing_cache_settings(embedding),
	)


def build_model_from_config(config: Mapping[str, object]) -> AmplitudeMAE3D:
	"""Instantiate an amplitude MAE from a checkpoint config."""
	_validate_checkpoint_resolved_config(config)
	model = _required_mapping(config, 'model')
	_validate_checkpoint_model_contract(model)
	return AmplitudeMAE3D(
		in_channels=_positive_int(model.get('in_channels'), 'model.in_channels'),
		out_channels=_positive_int(model.get('out_channels'), 'model.out_channels'),
		patch_size_xyz=_xyz_from_mapping(
			model,
			'patch_size',
			'model',
			default=None,
		),
		encoder_dim=_positive_int(model.get('encoder_dim'), 'model.encoder_dim'),
		encoder_depth=_positive_int(
			model.get('encoder_depth'),
			'model.encoder_depth',
		),
		encoder_heads=_positive_int(
			model.get('encoder_heads'),
			'model.encoder_heads',
		),
		decoder_dim=_positive_int(model.get('decoder_dim'), 'model.decoder_dim'),
		decoder_depth=_positive_int(
			model.get('decoder_depth'),
			'model.decoder_depth',
		),
		decoder_heads=_positive_int(
			model.get('decoder_heads'),
			'model.decoder_heads',
		),
	)


def build_model_from_checkpoint_payload(
	payload: Mapping[str, object],
) -> AmplitudeMAE3D:
	"""Instantiate and strictly load a supported amplitude encoder checkpoint."""
	config = _checkpoint_config(payload)
	state_dict = _model_state_dict(payload)
	model = build_model_from_config(config)
	model.to(dtype=_checkpoint_floating_dtype(state_dict))
	model.load_state_dict(state_dict, strict=True)
	return model


def extract_survey_embeddings(  # noqa: PLR0913
	manifest: SurveyManifest,
	*,
	model: AmplitudeMAE3D,
	store: NpyMemmapVolumeStore,
	settings: EmbeddingExtractionSettings,
	checkpoint_config: Mapping[str, object],
	checkpoint_payload: Mapping[str, object] | None,
	checkpoint_sha256: str,
	device: torch.device,
	skip_existing: bool,
	timer: StageTimer | None = None,
	producer_timer: StageTimer | None = None,
) -> SurveyEmbeddingResult:
	"""Extract and write embeddings for one survey manifest."""
	manifest.validate()
	amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
	valid_mask_path = (
		None
		if manifest.amplitude.valid_mask_path is None
		else resolve_manifest_path(manifest, manifest.amplitude.valid_mask_path)
	)
	if valid_mask_path is not None:
		store.open_source_valid_mask(
			valid_mask_path,
			manifest.amplitude.shape_xyz,
		)
	stats_path = resolve_manifest_path(
		manifest,
		manifest.amplitude.normalization_stats_path,
	)
	stats = load_normalization_stats(stats_path)
	preprocess_settings = _amplitude_preprocess_settings(settings)
	cache_plan = plan_survey_preprocessing_cache(
		amplitude_path=amplitude_path,
		stats=stats,
		preprocess_settings=preprocess_settings,
		cache_settings=settings.preprocessing_cache,
		default_cache_root=settings.output_dir / '.preprocessing_cache',
		valid_mask_path=valid_mask_path,
	)
	patch_size = model.patch_size_xyz
	token_grid = token_grid_shape_xyz(manifest.amplitude.shape_xyz, patch_size)
	metadata = build_embedding_metadata(
		manifest=manifest,
		amplitude_path=amplitude_path,
		stats_path=stats_path,
		settings=settings,
		checkpoint_config=checkpoint_config,
		checkpoint_payload=checkpoint_payload,
		checkpoint_sha256=checkpoint_sha256,
		model=model,
		token_grid_shape=token_grid,
		preprocessing_cache_plan=cache_plan,
		device=device,
	)
	paths = output_paths(settings.output_dir, manifest.survey_id)
	if prepare_outputs(paths, metadata, skip_existing=skip_existing):
		return SurveyEmbeddingResult(
			survey_id=manifest.survey_id,
			embeddings_path=paths.embeddings,
			valid_tokens_path=paths.valid_tokens,
			metadata_path=paths.metadata,
			skipped=True,
		)

	sum_array, count_array = create_merge_memmaps(
		paths,
		token_grid_shape_xyz=token_grid,
		embedding_dim=model.encoder_dim,
	)
	merger = EmbeddingMerger(
		token_grid_shape_xyz=token_grid,
		embedding_dim=model.encoder_dim,
		sum_array=sum_array,
		count_array=count_array,
	)
	timer = timer or StageTimer(enabled=settings.stage_timing)
	producer_timer = producer_timer or StageTimer(
		enabled=settings.stage_timing,
		accumulator=timer.accumulator,
	)
	if cache_plan.effective_mode == 'off':
		prepared_survey = None
	else:
		with producer_timer.stage('prepare_survey_cache'):
			prepared_survey = prepare_survey_preprocessing_cache(
				plan=cache_plan,
				amplitude_path=amplitude_path,
				stats=stats,
				preprocess_settings=preprocess_settings,
				cache_settings=settings.preprocessing_cache,
				store=store,
			)
	try:
		windows = iter_sliding_windows(
			manifest.amplitude.shape_xyz,
			window_size_xyz=settings.window_size_xyz,
			overlap_xyz=settings.overlap_xyz,
			patch_size_xyz=patch_size,
		)
		prepared_batches = _iter_prepared_batches(
			windows,
			manifest=manifest,
			amplitude_path=amplitude_path,
			stats=stats,
			store=store,
			settings=settings,
			patch_size_xyz=patch_size,
			pin_memory=device.type == 'cuda',
			producer_timer=producer_timer,
			prepared_survey=prepared_survey,
		)
		for prepared_batch in _prefetch_batches(
			prepared_batches,
			queue_depth=settings.prefetch_queue_depth,
			timer=timer,
		):
			_process_prepared_batch(
				prepared_batch,
				model=model,
				settings=settings,
				device=device,
				merger=merger,
				timer=timer,
			)
	finally:
		if prepared_survey is not None:
			prepared_survey.close()
	with timer.stage('merge_write'):
		merger.write_average(
			embedding_path=paths.embeddings_tmp,
			valid_tokens_path=paths.valid_tokens_tmp,
			output_dtype=settings.output_dtype,
			chunk_size_x=settings.average_chunk_size_x,
		)
	commit_staged_outputs(paths, metadata)
	cleanup_temp_outputs(paths)
	return SurveyEmbeddingResult(
		survey_id=manifest.survey_id,
		embeddings_path=paths.embeddings,
		valid_tokens_path=paths.valid_tokens,
		metadata_path=paths.metadata,
		skipped=False,
	)


def build_embedding_metadata(  # noqa: PLR0913
	*,
	manifest: SurveyManifest,
	amplitude_path: Path,
	stats_path: Path,
	settings: EmbeddingExtractionSettings,
	checkpoint_config: Mapping[str, object],
	checkpoint_payload: Mapping[str, object] | None = None,
	checkpoint_sha256: str | None = None,
	model: AmplitudeMAE3D,
	token_grid_shape: XYZ,
	device: torch.device,
	preprocessing_cache_plan: SurveyPreprocessingCachePlan | None = None,
) -> dict[str, object]:
	"""Return deterministic metadata for one survey output."""
	resolved_checkpoint_sha256 = (
		file_sha256(settings.checkpoint_path)
		if checkpoint_sha256 is None
		else checkpoint_sha256
	)
	cache_plan = preprocessing_cache_plan or SurveyPreprocessingCachePlan(
		'off',
		'off',
		None,
		None,
		None,
	)
	metadata = {
		'survey_id': manifest.survey_id,
		'source_amplitude_path': str(amplitude_path),
		'volume_shape_xyz': list(manifest.amplitude.shape_xyz),
		'checkpoint_path': str(settings.checkpoint_path),
		'checkpoint_sha256': resolved_checkpoint_sha256,
		'model_geometry': _model_geometry(checkpoint_config, model),
		'patch_size': list(model.patch_size_xyz),
		'token_grid_shape': list(token_grid_shape),
		'window_size': list(settings.window_size_xyz),
		'overlap': list(settings.overlap_xyz),
		'normalization_stats_path': str(stats_path),
		'output_dtype': str(settings.output_dtype),
		'precision': _embedding_precision_metadata(settings, device=device),
		'min_token_valid_fraction': settings.min_token_valid_fraction,
		'normalized_clip_abs': settings.normalized_clip_abs,
		'amplitude_agc': settings.amplitude_agc.to_dict(),
		'finite_check_mode': settings.finite_check_mode,
		'preprocessing': {
			'normalized_clip_abs': settings.normalized_clip_abs,
			'amplitude_agc': settings.amplitude_agc.to_dict(),
			'finite_check_mode': settings.finite_check_mode,
		},
		'preprocessing_cache': cache_plan.to_metadata(),
		'zero_mask': {
			'enabled': settings.zero_mask.enabled,
			'zero_atol': settings.zero_mask.zero_atol,
			'z_sample_influence_radius': settings.zero_mask.z_sample_influence_radius,
			'xy_trace_influence_radius': settings.zero_mask.xy_trace_influence_radius,
		},
		'pretraining_objective': _pretraining_objective(checkpoint_config),
	}
	if checkpoint_config.get('stage') == STAGE_BARLOW_TWINS_TRAINING and (
		checkpoint_payload is None
		or not _is_random_encoder_checkpoint(checkpoint_payload)
	):
		metadata['pretraining_method'] = BARLOW_TWINS_PRETRAINING_METHOD
	if manifest.amplitude.valid_mask_path is not None:
		metadata['source_valid_mask_path'] = str(
			resolve_manifest_path(manifest, manifest.amplitude.valid_mask_path),
		)
	if checkpoint_payload is not None:
		stratigraphy_pretext = _stratigraphy_pretext_metadata(checkpoint_payload)
		if stratigraphy_pretext is not None:
			metadata['stratigraphy_pretext'] = stratigraphy_pretext
			embedding_semantics = stratigraphy_pretext.get(
				'refresh_embedding_semantics'
			)
			if embedding_semantics is not None:
				metadata['embedding_semantics'] = embedding_semantics
	return metadata


@dataclass(frozen=True)
class _PreparedBatch:
	windows: tuple[SlidingWindow, ...]
	x: torch.Tensor
	token_valid_masks: torch.Tensor


def _iter_prepared_batches(  # noqa: PLR0913
	windows: Iterable[SlidingWindow],
	*,
	manifest: SurveyManifest,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	store: NpyMemmapVolumeStore,
	settings: EmbeddingExtractionSettings,
	patch_size_xyz: XYZ,
	pin_memory: bool,
	producer_timer: StageTimer,
	prepared_survey: PreparedSurveyAmplitude | None,
) -> Iterator[_PreparedBatch]:
	prepared: list[tuple[SlidingWindow, np.ndarray, np.ndarray]] = []
	for window in windows:
		with producer_timer.stage('read_preprocess'):
			item = _read_window(
				window,
				manifest=manifest,
				amplitude_path=amplitude_path,
				stats=stats,
				store=store,
				settings=settings,
				patch_size_xyz=patch_size_xyz,
				prepared_survey=prepared_survey,
			)
		if not item[2].any():
			continue
		prepared.append(item)
		if len(prepared) == settings.batch_size:
			yield _stack_prepared_batch(prepared, pin_memory=pin_memory)
			prepared = []
	if prepared:
		yield _stack_prepared_batch(prepared, pin_memory=pin_memory)


def _stack_prepared_batch(
	prepared: Sequence[tuple[SlidingWindow, np.ndarray, np.ndarray]],
	*,
	pin_memory: bool,
) -> _PreparedBatch:
	x = torch.from_numpy(np.stack([item[1] for item in prepared], axis=0))
	token_valid_masks = torch.from_numpy(
		np.stack([item[2] for item in prepared], axis=0),
	)
	if pin_memory:
		x = x.pin_memory()
		token_valid_masks = token_valid_masks.pin_memory()
	return _PreparedBatch(
		windows=tuple(item[0] for item in prepared),
		x=x,
		token_valid_masks=token_valid_masks,
	)


def _process_prepared_batch(  # noqa: PLR0913
	prepared: _PreparedBatch,
	*,
	model: AmplitudeMAE3D,
	settings: EmbeddingExtractionSettings,
	device: torch.device,
	merger: EmbeddingMerger,
	timer: StageTimer,
) -> None:
	non_blocking = device.type == 'cuda' and prepared.x.is_pinned()
	with timer.stage('h2d'):
		x = prepared.x.to(
			device=device,
			dtype=_model_floating_dtype(model),
			non_blocking=non_blocking,
		)
		token_masks = prepared.token_valid_masks.to(
			device=device,
			non_blocking=non_blocking,
		)
	autocast_dtype = _resolve_autocast_dtype(settings, device=device)
	with (
		timer.stage('encode'),
		torch.inference_mode(),
		torch.amp.autocast(
			device.type,
			enabled=autocast_dtype is not None,
			dtype=autocast_dtype,
		),
	):
		output = model.encode_tokens(x, valid_mask=token_masks)
	with timer.stage('d2h'):
		tokens = (
			cast('torch.Tensor', output['tokens'])
			.detach()
			.to(device='cpu', dtype=torch.float32)
			.numpy()
		)
	window_token_shape = cast('tuple[int, int, int]', output['token_grid_shape'])
	with timer.stage('merge_write'):
		for index, window in enumerate(prepared.windows):
			merger.add_window(
				window,
				patch_size_xyz=model.patch_size_xyz,
				token_embeddings=tokens[index].reshape(
					*window_token_shape,
					model.encoder_dim,
				),
				token_valid_mask=prepared.token_valid_masks[index].numpy(),
			)


@dataclass(frozen=True)
class _ProducerFailure:
	error: BaseException


_QUEUE_END = object()


def _prefetch_batches(  # noqa: C901
	batches: Iterable[_PreparedBatch],
	*,
	queue_depth: int,
	timer: StageTimer,
) -> Iterator[_PreparedBatch]:
	if queue_depth == 0:
		yield from batches
		return

	batch_queue: queue.Queue[object] = queue.Queue(maxsize=queue_depth)
	cancelled = threading.Event()

	def put(item: object) -> bool:
		while not cancelled.is_set():
			try:
				batch_queue.put(item, timeout=0.05)
			except queue.Full:
				continue
			return True
		return False

	def produce() -> None:
		try:
			for batch in batches:
				if not put(batch):
					return
		except BaseException as exc:  # noqa: BLE001
			put(_ProducerFailure(exc))
		finally:
			put(_QUEUE_END)

	producer = threading.Thread(
		target=produce,
		name='embedding-prefetch',
		daemon=False,
	)
	producer.start()
	try:
		while True:
			with timer.stage('queue_wait'):
				item = batch_queue.get()
			if item is _QUEUE_END:
				return
			if isinstance(item, _ProducerFailure):
				raise item.error
			yield cast('_PreparedBatch', item)
	finally:
		cancelled.set()
		while producer.is_alive():
			with suppress(queue.Empty):
				batch_queue.get_nowait()
			producer.join(timeout=0.05)


def _resolve_autocast_dtype(
	settings: EmbeddingExtractionSettings,
	*,
	device: torch.device,
) -> torch.dtype | None:
	if not settings.amp or device.type != 'cuda':
		return None
	if settings.amp_dtype == 'bfloat16':
		if not cuda_device_supports_bfloat16(device):
			msg = 'embedding.amp_dtype=bfloat16 is not supported by the CUDA device'
			raise ValueError(msg)
		return torch.bfloat16
	if settings.amp_dtype == 'float16':
		return torch.float16
	return torch.bfloat16 if cuda_device_supports_bfloat16(device) else torch.float16


def _embedding_precision_metadata(
	settings: EmbeddingExtractionSettings,
	*,
	device: torch.device,
) -> dict[str, object]:
	autocast_dtype = _resolve_autocast_dtype(settings, device=device)
	return {
		'amp_requested': settings.amp,
		'amp_dtype_requested': settings.amp_dtype,
		'resolved_dtype': (
			str(autocast_dtype).removeprefix('torch.')
			if autocast_dtype is not None
			else 'float32'
		),
		'amp_enabled': autocast_dtype is not None,
	}


def _read_window(  # noqa: PLR0913
	window: SlidingWindow,
	*,
	manifest: SurveyManifest,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	store: NpyMemmapVolumeStore,
	settings: EmbeddingExtractionSettings,
	patch_size_xyz: XYZ,
	prepared_survey: PreparedSurveyAmplitude | None = None,
) -> tuple[SlidingWindow, np.ndarray, np.ndarray]:
	request = CropRequest(
		survey_id=manifest.survey_id,
		start_xyz=window.start_xyz,
		size_xyz=window.size_xyz,
	)
	preprocess_settings = _amplitude_preprocess_settings(settings)
	if prepared_survey is None:
		prepared = read_amplitude_crop(
			request=request,
			amplitude_path=amplitude_path,
			stats=stats,
			store=store,
			patch_size_xyz=patch_size_xyz,
			settings=preprocess_settings,
			valid_mask_path=(
				None
				if manifest.amplitude.valid_mask_path is None
				else resolve_manifest_path(
					manifest,
					manifest.amplitude.valid_mask_path,
				)
			),
		)
	else:
		prepared = read_prepared_survey_amplitude_crop(
			request=request,
			normalized_amplitude=prepared_survey.normalized_amplitude,
			zero_like_mask=prepared_survey.zero_like_mask,
			patch_size_xyz=patch_size_xyz,
			settings=preprocess_settings,
		)
	return window, prepared.x, prepared.token_valid_mask


def _checkpoint_config(payload: Mapping[str, object]) -> Mapping[str, object]:
	value = payload.get('config')
	if isinstance(value, Mapping):
		config = cast('Mapping[str, object]', value)
		_validate_checkpoint_resolved_config(config)
		_validate_checkpoint_method_identity(payload, config)
		return config
	msg = 'checkpoint is missing a resolved config'
	raise TypeError(msg)


def _checkpoint_config_override(
	override: Mapping[str, object],
) -> Mapping[str, object]:
	"""Validate an explicit library-level checkpoint config replacement."""
	if not isinstance(override, Mapping):
		msg = 'checkpoint_config_override must be a resolved config mapping'
		raise TypeError(msg)
	config = cast('Mapping[str, object]', override)
	_validate_checkpoint_resolved_config(config)
	return config


def _model_state_dict(payload: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
	value = payload.get('model_state_dict')
	if not isinstance(value, Mapping):
		msg = 'checkpoint is missing model_state_dict'
		raise TypeError(msg)
	return cast('Mapping[str, torch.Tensor]', value)


def _checkpoint_floating_dtype(
	state_dict: Mapping[str, torch.Tensor],
) -> torch.dtype:
	dtypes = {
		tensor.dtype
		for tensor in state_dict.values()
		if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
	}
	if not dtypes:
		msg = 'checkpoint model_state_dict does not contain floating point tensors'
		raise ValueError(msg)
	if len(dtypes) != 1:
		msg = f'checkpoint model_state_dict has multiple floating dtypes: {dtypes!r}'
		raise ValueError(msg)
	return next(iter(dtypes))


def _model_floating_dtype(model: AmplitudeMAE3D) -> torch.dtype:
	for parameter in model.parameters():
		if parameter.is_floating_point():
			return parameter.dtype
	for buffer in model.buffers():
		if buffer.is_floating_point():
			return buffer.dtype
	msg = 'model does not contain floating point tensors'
	raise ValueError(msg)


def _model_geometry(
	config: Mapping[str, object],
	model: AmplitudeMAE3D,
) -> dict[str, object]:
	model_config = _optional_mapping(config, 'model')
	return {
		'name': model_config.get('name', 'amp_mae3d'),
		'in_channels': model.in_channels,
		'out_channels': model.out_channels,
		'patch_size': list(model.patch_size_xyz),
		'encoder_dim': model.encoder_dim,
		'encoder_depth': model.encoder.depth,
		'encoder_heads': model.encoder.num_heads,
		'decoder_dim': model.decoder_dim,
		'decoder_depth': model.decoder.depth,
		'decoder_heads': model.decoder.num_heads,
	}


def _validate_checkpoint_model_contract(model: Mapping[str, object]) -> None:
	if model.get('name') != FIXED_MODEL_CONTRACT['name']:
		msg = "checkpoint model.name must be 'amp_mae3d'"
		raise ValueError(msg)
	if model.get('in_channels') != FIXED_MODEL_CONTRACT['in_channels']:
		msg = 'checkpoint model.in_channels must be 1'
		raise ValueError(msg)
	if model.get('out_channels') != FIXED_MODEL_CONTRACT['out_channels']:
		msg = 'checkpoint model.out_channels must be 1'
		raise ValueError(msg)


def _validate_checkpoint_resolved_config(config: Mapping[str, object]) -> None:
	if config.get('stage') == STAGE_BARLOW_TWINS_TRAINING:
		_validate_barlow_twins_checkpoint_resolved_config(config)
		return
	if config.get('stage') != STAGE_MAE_TRAINING:
		msg = (
			'checkpoint config.stage must identify a supported pretraining method; '
			f'got {config.get("stage")!r}'
		)
		raise ValueError(msg)
	unexpected = sorted(set(config) - _CHECKPOINT_ALLOWED_TOP_LEVEL)
	if unexpected:
		msg = f'checkpoint config has unsupported top-level key(s): {unexpected!r}'
		raise ValueError(msg)
	missing = sorted(_CHECKPOINT_REQUIRED_TOP_LEVEL - set(config))
	if missing:
		msg = f'checkpoint config is missing resolved section(s): {missing!r}'
		raise ValueError(msg)

	paths = _required_mapping(config, 'paths')
	manifests = _required_mapping(config, 'manifests')
	data = _required_mapping(config, 'data')
	model = _required_mapping(config, 'model')
	masking = _required_mapping(config, 'masking')
	loss = _required_mapping(config, 'loss')
	train = _required_mapping(config, 'train')
	zero_mask = _required_mapping(config, 'zero_mask')

	_required_non_empty_string(paths, 'output_root', 'paths')
	_validate_required_checkpoint_keys(
		manifests,
		'manifests',
		('train', 'train_path_list'),
	)
	_required_non_empty_string(manifests, 'train', 'manifests')
	_required_non_empty_string(
		manifests,
		'train_path_list',
		'manifests',
	)
	_validate_fixed_checkpoint_values(data, 'data', FIXED_DATA_CONTRACT)
	_validate_fixed_checkpoint_values(model, 'model', FIXED_MODEL_CONTRACT)
	_validate_fixed_checkpoint_values(masking, 'masking', FIXED_MASKING_CONTRACT)
	_validate_fixed_checkpoint_values(loss, 'loss', FIXED_LOSS_CONTRACT)
	_validate_required_checkpoint_keys(
		data,
		'data',
		(
			*(
				key
				for key in DEFAULT_MAE_DATA_OPTIONS
				if key not in {'amplitude_agc', 'finite_check_mode'}
			),
			'local_crop_size',
		),
	)
	_validate_required_checkpoint_keys(model, 'model', _CHECKPOINT_MODEL_GEOMETRY_KEYS)
	_validate_required_checkpoint_keys(masking, 'masking', _CHECKPOINT_MASKING_KEYS)
	_validate_required_checkpoint_keys(train, 'train', _CHECKPOINT_TRAIN_REQUIRED_KEYS)
	_validate_required_checkpoint_keys(
		zero_mask,
		'zero_mask',
		DEFAULT_ZERO_MASK_CONTRACT,
	)
	_validate_positive_xyz(data['local_crop_size'], 'data.local_crop_size')
	_fraction(data['min_valid_fraction'], 'data.min_valid_fraction')
	_positive_int(data['max_resample_attempts'], 'data.max_resample_attempts')
	if 'normalized_clip_abs' in data:
		_positive_finite_number(
			data['normalized_clip_abs'],
			'data.normalized_clip_abs',
		)
	_validate_checkpoint_amplitude_agc(data)
	_finite_check_mode_from_config(config)
	_validate_checkpoint_model_contract(model)
	_validate_positive_xyz(model['patch_size'], 'model.patch_size')
	for key in _CHECKPOINT_MODEL_GEOMETRY_KEYS[1:]:
		_positive_int(model[key], f'model.{key}')
	_validate_checkpoint_masking(masking)
	_validate_checkpoint_loss(loss)
	_validate_checkpoint_train(train)
	_zero_mask_from_mapping(zero_mask)


def _validate_barlow_twins_checkpoint_resolved_config(
	config: Mapping[str, object],
) -> None:
	unexpected = sorted(set(config) - _BARLOW_TWINS_CHECKPOINT_ALLOWED_TOP_LEVEL)
	if unexpected:
		msg = f'checkpoint config has unsupported top-level key(s): {unexpected!r}'
		raise ValueError(msg)
	missing = sorted(_BARLOW_TWINS_CHECKPOINT_REQUIRED_TOP_LEVEL - set(config))
	if missing:
		msg = f'checkpoint config is missing resolved section(s): {missing!r}'
		raise ValueError(msg)
	raw = dict(config)
	raw.pop('stage')
	for section, fixed_contract in (
		('data', FIXED_DATA_CONTRACT),
		('model', FIXED_MODEL_CONTRACT),
	):
		resolved_section = _required_mapping(config, section)
		raw[section] = {
			key: value
			for key, value in resolved_section.items()
			if key not in fixed_contract
		}
	resolved = resolve_barlow_twins_training_config(raw)
	if resolved != dict(config):
		msg = 'Barlow Twins checkpoint config must contain the fully resolved config'
		raise ValueError(msg)


def _validate_checkpoint_method_identity(
	payload: Mapping[str, object],
	config: Mapping[str, object],
) -> None:
	if config.get('stage') != STAGE_BARLOW_TWINS_TRAINING:
		return
	if _is_random_encoder_checkpoint(payload):
		return
	if payload.get('pretraining_method') != BARLOW_TWINS_PRETRAINING_METHOD:
		raise ValueError('Barlow Twins checkpoint pretraining_method is invalid')
	if payload.get('checkpoint_kind') != BARLOW_TWINS_CHECKPOINT_KIND:
		raise ValueError('Barlow Twins checkpoint checkpoint_kind is invalid')
	prefixes = payload.get('trained_parameter_prefixes')
	if not isinstance(prefixes, list | tuple) or tuple(prefixes) != (
		BARLOW_TWINS_TRAINED_PARAMETER_PREFIXES
	):
		raise ValueError(
			'Barlow Twins checkpoint trained_parameter_prefixes are invalid'
		)
	if not isinstance(payload.get('projector_state_dict'), Mapping):
		raise TypeError(
			'Barlow Twins checkpoint projector_state_dict must be a mapping'
		)
	state_dict = _model_state_dict(payload)
	wrapper_keys = sorted(
		key
		for key in state_dict
		if key.startswith(('backbone.', 'projector.'))
	)
	if wrapper_keys:
		raise ValueError(
			'Barlow Twins model_state_dict must use bare AmplitudeMAE3D keys; '
			f'got wrapper key(s): {wrapper_keys!r}'
		)


def _is_random_encoder_checkpoint(payload: Mapping[str, object]) -> bool:
	training_state = payload.get('training_state')
	metadata = payload.get('metadata')
	return (
		isinstance(training_state, Mapping)
		and training_state.get('checkpoint_kind') == 'random_init'
		and isinstance(metadata, Mapping)
		and metadata.get('random_encoder_baseline') is True
		and metadata.get('pretrained_weights_loaded') is False
	)


def _validate_required_checkpoint_keys(
	parent: Mapping[str, object],
	section: str,
	keys: Iterable[str],
) -> None:
	missing = sorted(set(keys) - set(parent))
	if missing:
		msg = f'checkpoint config.{section} is missing resolved key(s): {missing!r}'
		raise ValueError(msg)


def _validate_fixed_checkpoint_values(
	parent: Mapping[str, object],
	section: str,
	expected: Mapping[str, object],
) -> None:
	for key, expected_value in expected.items():
		if parent.get(key) == expected_value:
			continue
		msg = (
			f'checkpoint config.{section}.{key} must be {expected_value!r}; '
			f'got {parent.get(key)!r}'
		)
		raise ValueError(msg)


def _validate_checkpoint_masking(masking: Mapping[str, object]) -> None:
	ratio = masking.get('spatial_mask_ratio')
	if isinstance(ratio, bool) or not isinstance(ratio, Real):
		msg = (
			'checkpoint config.masking.spatial_mask_ratio must be a real '
			f'number; got {ratio!r}'
		)
		raise TypeError(msg)
	if not 0.0 < float(ratio) < 1.0:
		msg = (
			'checkpoint config.masking.spatial_mask_ratio must be greater than '
			f'0 and less than 1; got {ratio!r}'
		)
		raise ValueError(msg)
	_validate_positive_xyz(masking['block_size_tokens'], 'masking.block_size_tokens')


def _validate_checkpoint_loss(loss: Mapping[str, object]) -> None:
	_validate_required_checkpoint_keys(
		loss,
		'loss',
		('reconstruction', 'gradient_weight'),
	)
	reconstruction = loss.get('reconstruction')
	if reconstruction not in SUPPORTED_RECONSTRUCTION_LOSSES:
		msg = (
			'checkpoint config.loss.reconstruction must be one of '
			f'{sorted(SUPPORTED_RECONSTRUCTION_LOSSES)!r}; '
			f'got {reconstruction!r}'
		)
		raise ValueError(msg)
	if reconstruction == 'huber':
		_validate_required_checkpoint_keys(loss, 'loss', ('huber_delta',))
		_positive_finite_number(loss['huber_delta'], 'loss.huber_delta')
	elif 'huber_delta' in loss:
		msg = (
			'checkpoint config.loss.huber_delta must be omitted unless '
			'loss.reconstruction is huber'
		)
		raise ValueError(msg)
	_nonnegative_finite_number(loss['gradient_weight'], 'loss.gradient_weight')
	_nonnegative_finite_number(
		loss.get('visible_reconstruction_weight', 0.0),
		'loss.visible_reconstruction_weight',
	)
	_validate_checkpoint_target_normalization(loss)


def _validate_checkpoint_target_normalization(loss: Mapping[str, object]) -> None:
	target_normalization = loss.get('target_normalization')
	if target_normalization is None:
		return
	if not isinstance(target_normalization, Mapping):
		msg = 'checkpoint config.loss.target_normalization must be a mapping'
		raise TypeError(msg)
	mode = target_normalization.get('mode')
	if mode not in SUPPORTED_TARGET_NORMALIZATION_MODES:
		msg = (
			'checkpoint config.loss.target_normalization.mode must be one of '
			f'{sorted(SUPPORTED_TARGET_NORMALIZATION_MODES)!r}; got {mode!r}'
		)
		raise ValueError(msg)
	if mode == 'none':
		for key in ('eps', 'min_std'):
			if key in target_normalization:
				msg = (
					f'checkpoint config.loss.target_normalization.{key} must be '
					"omitted when mode is 'none'"
				)
				raise ValueError(msg)
		return
	for key in ('eps', 'min_std'):
		if key not in target_normalization:
			msg = f'checkpoint config.loss.target_normalization.{key} is required'
			raise ValueError(msg)
		_positive_finite_number(
			target_normalization[key],
			f'loss.target_normalization.{key}',
		)
	if float(loss['gradient_weight']) != 0.0:
		msg = (
			'checkpoint config.loss.gradient_weight must be 0.0 when '
			"loss.target_normalization.mode is 'patch_zscore'"
		)
		raise ValueError(msg)


def _pretraining_objective(config: Mapping[str, object]) -> dict[str, object]:
	if config.get('stage') == STAGE_BARLOW_TWINS_TRAINING:
		barlow_twins = _required_mapping(config, 'barlow_twins')
		return {
			'method': BARLOW_TWINS_PRETRAINING_METHOD,
			'projector_dim': _positive_int(
				barlow_twins.get('projector_dim'),
				'barlow_twins.projector_dim',
			),
			'redundancy_weight': _nonnegative_finite_number(
				barlow_twins.get('redundancy_weight'),
				'barlow_twins.redundancy_weight',
			),
			'normalization_eps': _positive_finite_number(
				barlow_twins.get('normalization_eps'),
				'barlow_twins.normalization_eps',
			),
		}
	loss = _required_mapping(config, 'loss')
	objective: dict[str, object] = {
		'reconstruction': loss.get('reconstruction'),
		'gradient_weight': float(loss.get('gradient_weight', 0.0)),
	}
	if 'visible_reconstruction_weight' in loss:
		objective['visible_reconstruction_weight'] = float(
			loss['visible_reconstruction_weight'],
		)
	if loss.get('reconstruction') == 'huber' and 'huber_delta' in loss:
		objective['huber_delta'] = float(loss['huber_delta'])
	target_normalization = loss.get('target_normalization')
	if isinstance(target_normalization, Mapping):
		objective['target_normalization'] = {
			str(key): value for key, value in target_normalization.items()
		}
	else:
		objective['target_normalization'] = {'mode': 'none'}
	return objective


def _stratigraphy_pretext_metadata(  # noqa: C901, PLR0911, PLR0912
	payload: Mapping[str, object],
) -> dict[str, object] | None:
	if 'stratigraphy_config' not in payload:
		return None
	stratigraphy_config = payload['stratigraphy_config']
	if not isinstance(stratigraphy_config, Mapping):
		msg = 'checkpoint stratigraphy_config must be a mapping'
		raise TypeError(msg)
	head = _required_mapping(stratigraphy_config, 'head')
	checkpoint_identity = payload.get('stratigraphy_checkpoint')
	if checkpoint_identity is not None:
		validate_stratigraphy_checkpoint_payload(payload)
		if not isinstance(checkpoint_identity, Mapping):
			raise TypeError('checkpoint stratigraphy_checkpoint must be a mapping')
		student = _required_mapping(stratigraphy_config, 'student')
		loss = _required_mapping(stratigraphy_config, 'loss')
		result = {
			'method': 'strat_hmm_multi_head_pretext',
			'base_objective': 'amp_mae3d',
			'head_spec': checkpoint_identity['head_spec'],
			'head_ks': checkpoint_identity['head_ks'],
			'head_count': len(checkpoint_identity['head_ks']),
			'unfreeze_top_blocks': _nonnegative_int(
				student.get('unfreeze_top_blocks'),
				'stratigraphy_config.student.unfreeze_top_blocks',
			),
			'distillation_weight': _nonnegative_finite_number(
				loss.get('distillation_weight'),
				'stratigraphy_config.loss.distillation_weight',
			),
			'prototype_weight': _nonnegative_finite_number(
				loss.get('prototype_weight'),
				'stratigraphy_config.loss.prototype_weight',
			),
			'prototype_weight_semantics': 'mean_across_heads',
			'usage_weight': _nonnegative_finite_number(
				loss.get('usage_weight'), 'stratigraphy_config.loss.usage_weight'
			),
			'usage_weight_semantics': 'mean_across_heads',
			'consistency_policy': checkpoint_identity['consistency_policy'],
			'consistency_weight': checkpoint_identity['consistency_weight'],
			'consistency_beta': checkpoint_identity['consistency_beta'],
			'model_tag': checkpoint_identity['model_tag'],
			'scientific_identity_sha256': checkpoint_identity[
				'scientific_identity_sha256'
			],
		'checkpoint_stratigraphy_state_sha256': checkpoint_identity[
			'stratigraphy_state_sha256'
		],
		}
		if checkpoint_identity.get('schema_version') == 3:
			posterior = _required_mapping(checkpoint_identity, 'posterior_manifest')
			return {
				**result,
				'target_representation': checkpoint_identity['target_representation'],
				'posterior_manifest_path': posterior['path'],
				'posterior_manifest_sha256': posterior['sha256'],
				'posterior_semantics': checkpoint_identity['posterior_semantics'],
				'posterior_cost_temperature': checkpoint_identity[
					'posterior_cost_temperature'
				],
				'per_head_posterior_sha256': checkpoint_identity['per_head_posteriors'],
			}
		if checkpoint_identity.get('schema_version') == 8:
			target_refresh_state = payload.get('target_refresh_state')
			if not isinstance(target_refresh_state, Mapping):
				raise TypeError('schema-8 target_refresh_state must be a mapping')
			return {
				**result,
				'model_role': checkpoint_identity['model_role'],
				'target_representation': checkpoint_identity[
					'target_representation'
				],
				'target_refresh_semantics': checkpoint_identity[
					'target_refresh_semantics'
				],
				'refresh_schedule_semantics': checkpoint_identity[
					'refresh_schedule_semantics'
				],
				'refresh_after_epochs': checkpoint_identity['refresh_after_epochs'],
				'hmm_iterations_per_refresh': checkpoint_identity[
					'hmm_iterations_per_refresh'
				],
				'embedding_source': checkpoint_identity['embedding_source'],
				'embedding_mode': checkpoint_identity['embedding_mode'],
				'refresh_embedding_semantics': checkpoint_identity[
					'refresh_embedding_semantics'
				],
				'center_initialization': checkpoint_identity[
					'center_initialization'
				],
				'center_update': checkpoint_identity['center_update'],
				'center_update_semantics': checkpoint_identity[
					'center_update_semantics'
				],
				'preprocessing_policy': checkpoint_identity['preprocessing_policy'],
				'target_activation_policy': checkpoint_identity[
					'target_activation_policy'
				],
				'empty_state_policy': checkpoint_identity['empty_state_policy'],
				'checkpoint_selection_policy': checkpoint_identity[
					'checkpoint_selection_policy'
				],
				'generation_root': checkpoint_identity['generation_root'],
				'initial_hard_target_manifest_sha256': checkpoint_identity[
					'initial_hard_target_manifest_sha256'
				],
				'target_manifest_sha256': target_refresh_state[
					'active_target_manifest_sha256'
				],
				'active_generation_id': target_refresh_state[
					'active_generation_id'
				],
				'active_generation_manifest_path': target_refresh_state[
					'active_generation_manifest_path'
				],
				'active_generation_manifest_sha256': target_refresh_state[
					'active_generation_manifest_sha256'
				],
				'active_generation_content_sha256': target_refresh_state[
					'active_generation_content_sha256'
				],
				'active_target_manifest_path': target_refresh_state[
					'active_target_manifest_path'
				],
				'active_target_manifest_sha256': target_refresh_state[
					'active_target_manifest_sha256'
				],
				'periodic_refresh_chain_path': target_refresh_state[
					'periodic_refresh_chain_path'
				],
				'periodic_refresh_chain_sha256': target_refresh_state[
					'periodic_refresh_chain_sha256'
				],
				'last_completed_refresh_epoch': target_refresh_state[
					'last_completed_refresh_epoch'
				],
				'next_scheduled_refresh_epoch': target_refresh_state[
					'next_scheduled_refresh_epoch'
				],
				'refresh_phase': target_refresh_state['refresh_phase'],
				'source_student_state_sha256': target_refresh_state[
					'source_student_state_sha256'
				],
				'fixed_preprocessing_hmm_identity_sha256': target_refresh_state[
					'fixed_preprocessing_hmm_identity_sha256'
				],
				'target_refresh_state_sha256': checkpoint_identity[
					'target_refresh_state_sha256'
				],
			}
		if checkpoint_identity.get('schema_version') == 7:
			target_manifest = _required_mapping(checkpoint_identity, 'target_manifest')
			optimizer_groups = checkpoint_identity['optimizer_group_identity']
			if not isinstance(optimizer_groups, list):
				raise TypeError(
					'checkpoint optimizer_group_identity must be a list for schema 7'
				)
			spatial_context_group = next(
				(
					group
					for group in optimizer_groups
					if isinstance(group, Mapping)
					and group.get('name') == 'spatial_context'
				),
				None,
			)
			if not isinstance(spatial_context_group, Mapping):
				raise ValueError(
					'schema-7 checkpoint is missing spatial_context optimizer group'
				)
			return {
				**result,
				'target_representation': checkpoint_identity['target_representation'],
				'objective_semantics': checkpoint_identity['objective_semantics'],
				'mask_semantics': checkpoint_identity['mask_semantics'],
				'column_fraction': checkpoint_identity['column_fraction'],
				'selection_policy': checkpoint_identity['selection_policy'],
				'replacement': checkpoint_identity['replacement'],
				'replacement_initialization': checkpoint_identity[
					'replacement_initialization'
				],
				'rng_policy': checkpoint_identity['rng_policy'],
				'masked_prototype_weight': checkpoint_identity[
					'masked_prototype_weight'
				],
				'visible_prototype_weight': checkpoint_identity[
					'visible_prototype_weight'
				],
				'distillation_scope': checkpoint_identity['distillation_scope'],
				'supervised_loss': checkpoint_identity['supervised_loss'],
				'target_manifest_path': target_manifest['path'],
				'target_manifest_sha256': target_manifest['sha256'],
				'per_head_target_sha256': checkpoint_identity['per_head_targets'],
				'checkpoint_spatial_context_state_sha256': checkpoint_identity[
					'spatial_context_state_sha256'
				],
				'checkpoint_student_state_sha256': checkpoint_identity[
					'student_state_sha256'
				],
				'initial_spatial_context_state_sha256': checkpoint_identity[
					'initial_spatial_context_state_sha256'
				],
				'spatial_context_optimizer_group': spatial_context_group,
			}
		if checkpoint_identity.get('schema_version') in {4, 5, 6}:
			return _hard_target_pretext_metadata(
				result,
				checkpoint_identity,
			)
		target_manifest = _required_mapping(checkpoint_identity, 'target_manifest')
		return {
			**result,
			'target_manifest_path': target_manifest['path'],
			'target_manifest_sha256': target_manifest['sha256'],
			'per_head_target_sha256': checkpoint_identity['per_head_targets'],
		}
	student = _required_mapping(stratigraphy_config, 'student')
	loss = _required_mapping(stratigraphy_config, 'loss')
	pseudo_targets = _required_mapping(stratigraphy_config, 'pseudo_targets')
	result = {
		'method': 'strat_hmm_pretext',
		'base_objective': 'amp_mae3d',
		'head_num_prototypes': _positive_int(
			head.get('num_prototypes'),
			'stratigraphy_config.head.num_prototypes',
		),
		'unfreeze_top_blocks': _nonnegative_int(
			student.get('unfreeze_top_blocks'),
			'stratigraphy_config.student.unfreeze_top_blocks',
		),
		'distillation_weight': _nonnegative_finite_number(
			loss.get('distillation_weight'),
			'stratigraphy_config.loss.distillation_weight',
		),
		'pseudo_target_input_dir': _required_non_empty_string(
			pseudo_targets,
			'input_dir',
			'stratigraphy_config.pseudo_targets',
		),
	}
	control_identity = payload.get('control_identity')
	if control_identity is not None:
		if not isinstance(control_identity, Mapping):
			raise TypeError('checkpoint control_identity must be a mapping')
		model_tag = control_identity.get('model_tag')
		if not isinstance(model_tag, str) or not model_tag:
			raise TypeError('checkpoint control_identity.model_tag must be non-empty')
		result['model_tag'] = model_tag
		result['control_identity_sha256'] = hashlib.sha256(
			json.dumps(
				control_identity,
				sort_keys=True,
				separators=(',', ':'),
				allow_nan=False,
			).encode('utf-8'),
		).hexdigest()
	return result


def _hard_target_pretext_metadata(
	base: Mapping[str, object],
	checkpoint_identity: Mapping[str, object],
) -> dict[str, object]:
	"""Attach strict schema-v4/v5/v6 hard-target provenance to metadata."""
	if checkpoint_identity.get('schema_version') == 4:
		lateral = _required_mapping(
			checkpoint_identity,
			'lateral_target_manifest',
		)
		return {
			**base,
			'target_representation': checkpoint_identity['target_representation'],
			'target_semantics': checkpoint_identity['target_semantics'],
			'lateral_target_manifest_path': lateral['path'],
			'lateral_target_manifest_sha256': lateral['sha256'],
			'per_head_lateral_target_sha256': checkpoint_identity[
				'per_head_lateral_targets'
			],
			'source_hard_manifest_sha256': checkpoint_identity[
				'source_hard_manifest_sha256'
			],
			'source_posterior_manifest_sha256': checkpoint_identity[
				'source_posterior_manifest_sha256'
			],
			'lateral_smoothing': checkpoint_identity['lateral_smoothing'],
		}
	if checkpoint_identity.get('schema_version') == 5:
		xy_neighbor_consensus = _required_mapping(
			checkpoint_identity,
			'xy_neighbor_consensus_target_manifest',
		)
		return {
			**base,
			'target_representation': checkpoint_identity['target_representation'],
			'target_semantics': checkpoint_identity['target_semantics'],
			'xy_neighbor_consensus_target_manifest_path': xy_neighbor_consensus['path'],
			'xy_neighbor_consensus_target_manifest_sha256': xy_neighbor_consensus[
				'sha256'
			],
			'per_head_xy_neighbor_consensus_target_sha256': checkpoint_identity[
				'per_head_xy_neighbor_consensus_targets'
			],
			'source_hard_manifest_sha256': checkpoint_identity[
				'source_hard_manifest_sha256'
			],
			'xy_neighbor_consensus_smoothing': checkpoint_identity[
				'xy_neighbor_consensus_smoothing'
			],
		}
	xy_neighbor_unanimous = _required_mapping(
		checkpoint_identity,
		'xy_neighbor_unanimous_target_manifest',
	)
	return {
		**base,
		'target_representation': checkpoint_identity['target_representation'],
		'target_semantics': checkpoint_identity['target_semantics'],
		'xy_neighbor_unanimous_target_manifest_path': xy_neighbor_unanimous['path'],
		'xy_neighbor_unanimous_target_manifest_sha256': xy_neighbor_unanimous['sha256'],
		'per_head_xy_neighbor_unanimous_target_sha256': checkpoint_identity[
			'per_head_xy_neighbor_unanimous_targets'
		],
		'source_hard_manifest_sha256': checkpoint_identity[
			'source_hard_manifest_sha256'
		],
		'xy_neighbor_unanimous_smoothing': checkpoint_identity[
			'xy_neighbor_unanimous_smoothing'
		],
	}


def _validate_checkpoint_train(train: Mapping[str, object]) -> None:
	for key in ('batch_size', 'samples_per_epoch', 'epochs'):
		_positive_int(train[key], f'train.{key}')
	_nonnegative_int(train['num_workers'], 'train.num_workers')
	for key in ('shuffle', 'amp'):
		_bool(train[key], f'train.{key}')
	for key in ('lr', 'grad_clip_norm'):
		_positive_number(train[key], f'train.{key}')
	_nonnegative_number(train['weight_decay'], 'train.weight_decay')
	_required_non_empty_string(train, 'device', 'train')
	seed = train.get('seed')
	if not isinstance(seed, Integral) or isinstance(seed, bool):
		msg = f'train.seed must be an integer; got {seed!r}'
		raise TypeError(msg)


def _checkpoint_path(config: Mapping[str, object]) -> Path:
	embeddings = _required_mapping(config, 'embeddings')
	return _required_path(embeddings, 'checkpoint', 'embeddings')


def _manifest_path(config: Mapping[str, object]) -> Path:
	manifests = _required_mapping(config, 'manifests')
	return _required_path(manifests, 'input', 'manifests')


def _resolve_device(
	device: str | torch.device | None,
	config: Mapping[str, object],
) -> torch.device:
	if isinstance(device, torch.device):
		return device
	if device is None:
		train = _optional_mapping(config, 'train')
		value = train.get('device', 'cpu')
	else:
		value = device
	if value == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	return torch.device(str(value))


def _zero_mask_from_config(config: Mapping[str, object]) -> ZeroMaskConfig:
	value = _zero_mask_mapping_from_config(config)
	if value is None:
		msg = 'checkpoint config is missing zero_mask'
		raise ValueError(msg)
	return _zero_mask_from_mapping(value)


def _amplitude_agc_from_config(config: Mapping[str, object]) -> AmplitudeAgcConfig:
	data = _required_mapping(config, 'data')
	value = data.get('amplitude_agc')
	return AmplitudeAgcConfig.from_mapping(
		cast('Mapping[str, object] | None', value),
	)


def _finite_check_mode_from_config(
	config: Mapping[str, object],
) -> FiniteCheckMode:
	data = _required_mapping(config, 'data')
	value = data.get('finite_check_mode', 'strict')
	if value not in {'strict', 'output_only', 'off'}:
		msg = (
			'data.finite_check_mode must be "strict", "output_only", or "off"; '
			f'got {value!r}'
		)
		raise ValueError(msg)
	return cast('FiniteCheckMode', value)


def _validate_checkpoint_amplitude_agc(data: Mapping[str, object]) -> None:
	value = data.get('amplitude_agc')
	if value is None:
		return
	AmplitudeAgcConfig.from_mapping(cast('Mapping[str, object]', value))


def _zero_mask_mapping_from_config(
	config: Mapping[str, object],
) -> object:
	value = config.get('zero_mask')
	if value is not None:
		return value
	data = config.get('data')
	if isinstance(data, Mapping):
		return data.get('zero_mask')
	return None


def _reject_checkpoint_owned_extraction_sections(
	config: Mapping[str, object],
) -> None:
	stale = sorted(
		set(config) & {'data', 'model', 'masking', 'loss', 'train', 'zero_mask'},
	)
	if stale:
		msg = (
			'embedding extraction config must not include checkpoint-owned '
			f'section(s): {stale!r}'
		)
		raise ValueError(msg)


def _zero_mask_from_mapping(value: object) -> ZeroMaskConfig:
	if not isinstance(value, Mapping):
		msg = f'zero_mask config must be a mapping; got {value!r}'
		raise TypeError(msg)
	zero_mask = ZeroMaskConfig(**dict(value))
	zero_mask.validate()
	return zero_mask


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _optional_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _preprocessing_cache_settings(
	embedding: Mapping[str, object],
) -> SurveyPreprocessingCacheSettings:
	value = _optional_mapping(embedding, 'preprocessing_cache')
	mode = value.get('mode', 'off')
	if mode not in {'off', 'memory', 'memmap'}:
		msg = (
			'embedding.preprocessing_cache.mode must be "off", "memory", or '
			f'"memmap"; got {mode!r}'
		)
		raise ValueError(msg)
	directory_value = value.get('directory')
	if directory_value is not None and (
		not isinstance(directory_value, str) or not directory_value
	):
		msg = 'embedding.preprocessing_cache.directory must be a non-empty path'
		raise TypeError(msg)
	settings = SurveyPreprocessingCacheSettings(
		mode=cast('SurveyPreprocessingCacheMode', mode),
		chunk_size_x=_positive_int(
			value.get('chunk_size_x', 16),
			'embedding.preprocessing_cache.chunk_size_x',
		),
		reuse=_bool(
			value.get('reuse', True),
			'embedding.preprocessing_cache.reuse',
		),
		cleanup=_bool(
			value.get('cleanup', False),
			'embedding.preprocessing_cache.cleanup',
		),
		directory=(None if directory_value is None else Path(directory_value)),
	)
	settings.validate()
	return settings


def _amplitude_preprocess_settings(
	settings: EmbeddingExtractionSettings,
) -> AmplitudePreprocessSettings:
	return AmplitudePreprocessSettings(
		zero_mask=settings.zero_mask,
		normalized_clip_abs=settings.normalized_clip_abs,
		amplitude_agc=settings.amplitude_agc,
		min_token_valid_fraction=settings.min_token_valid_fraction,
		finite_check_mode=settings.finite_check_mode,
	)


def _required_path(parent: Mapping[str, object], key: str, prefix: str) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _xyz_from_mapping(
	parent: Mapping[str, object],
	key: str,
	prefix: str,
	*,
	default: object,
) -> XYZ:
	return _validate_positive_xyz(parent.get(key, default), f'{prefix}.{key}')


def _nonnegative_xyz_from_mapping(
	parent: Mapping[str, object],
	key: str,
	prefix: str,
	*,
	default: object,
) -> XYZ:
	return _validate_nonnegative_xyz(parent.get(key, default), f'{prefix}.{key}')


def _validate_positive_xyz(value: object, name: str) -> XYZ:
	if (
		isinstance(value, str)
		or not isinstance(value, Sequence)
		or len(value) != 3
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral) for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = cast('XYZ', tuple(int(axis) for axis in value))
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_nonnegative_xyz(value: object, name: str) -> XYZ:
	if (
		isinstance(value, str)
		or not isinstance(value, Sequence)
		or len(value) != 3
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral) for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = cast('XYZ', tuple(int(axis) for axis in value))
	if any(axis < 0 for axis in xyz):
		msg = f'{name} values must be nonnegative; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _positive_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{name} must be positive; got {integer!r}'
		raise ValueError(msg)
	return integer


def _nonnegative_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer < 0:
		msg = f'{name} must be nonnegative; got {integer!r}'
		raise ValueError(msg)
	return integer


def _fraction(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{name} must be in [0, 1]; got {fraction!r}'
		raise ValueError(msg)
	return fraction


def _positive_number(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if number <= 0.0:
		msg = f'{name} must be positive; got {number!r}'
		raise ValueError(msg)
	return number


def _positive_finite_number(value: object, name: str) -> float:
	number = _positive_number(value, name)
	if not np.isfinite(number):
		msg = f'{name} must be finite; got {number!r}'
		raise ValueError(msg)
	return number


def _nonnegative_number(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if number < 0.0:
		msg = f'{name} must be nonnegative; got {number!r}'
		raise ValueError(msg)
	return number


def _nonnegative_finite_number(value: object, name: str) -> float:
	number = _nonnegative_number(value, name)
	if not np.isfinite(number):
		msg = f'{name} must be finite; got {number!r}'
		raise ValueError(msg)
	return number


def _bool(value: object, name: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{name} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _amp_dtype(value: object) -> str:
	if value not in {'auto', 'bfloat16', 'float16'}:
		msg = (
			'embedding.amp_dtype must be one of '
			f"['auto', 'bfloat16', 'float16']; got {value!r}"
		)
		raise ValueError(msg)
	return cast('str', value)


def _required_non_empty_string(
	parent: Mapping[str, object],
	key: str,
	prefix: str,
) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_overlap_less_than_window(
	overlap: Sequence[int],
	window_size: Sequence[int],
) -> None:
	if any(
		overlap_axis >= window_axis
		for overlap_axis, window_axis in zip(overlap, window_size, strict=True)
	):
		msg = (
			'embedding.overlap values must be less than embedding.window_size '
			f'values; got overlap={list(overlap)!r}, '
			f'window_size={list(window_size)!r}'
		)
		raise ValueError(msg)


__all__ = [
	'UNMASKED_ENCODER_INPUT_MODE',
	'EmbeddingExtractionSettings',
	'SurveyEmbeddingResult',
	'build_embedding_metadata',
	'build_model_from_checkpoint_payload',
	'build_model_from_config',
	'extract_survey_embeddings',
	'extraction_settings_from_config',
	'reduce_valid_mask_to_tokens',
	'run_embedding_extraction',
]
