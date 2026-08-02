"""Full-volume amplitude MAE encoder embedding extraction."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import numpy as np
import torch

from seis_ssl_cluster.config.schema import (
	DEFAULT_MAE_DATA_OPTIONS,
	DEFAULT_MAE_TRAIN_OPTIONS,
	DEFAULT_ZERO_MASK_CONTRACT,
	FIXED_DATA_CONTRACT,
	FIXED_LOSS_CONTRACT,
	FIXED_MASKING_CONTRACT,
	FIXED_MODEL_CONTRACT,
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
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)
from seis_ssl_cluster.utils import StageTimer
from seis_ssl_cluster.utils.cuda import cuda_device_supports_bfloat16

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
	model_state_dict = _model_state_dict(payload)
	checkpoint_dtype = _checkpoint_floating_dtype(model_state_dict)
	model = build_model_from_config(checkpoint_config)
	model.to(dtype=checkpoint_dtype)
	model.load_state_dict(model_state_dict)
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
		return [
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
	finally:
		if settings.stage_timing:
			timer.write_json(settings.output_dir / 'stage_timings.json')


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


def extract_survey_embeddings(  # noqa: PLR0913
	manifest: SurveyManifest,
	*,
	model: AmplitudeMAE3D,
	store: NpyMemmapVolumeStore,
	settings: EmbeddingExtractionSettings,
	checkpoint_config: Mapping[str, object],
	checkpoint_payload: Mapping[str, object],
	checkpoint_sha256: str,
	device: torch.device,
	skip_existing: bool,
	timer: StageTimer | None = None,
	producer_timer: StageTimer | None = None,
) -> SurveyEmbeddingResult:
	"""Extract and write embeddings for one survey manifest."""
	manifest.validate()
	amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
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
	if checkpoint_payload is not None:
		stratigraphy_pretext = _stratigraphy_pretext_metadata(checkpoint_payload)
		if stratigraphy_pretext is not None:
			metadata['stratigraphy_pretext'] = stratigraphy_pretext
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
	if config.get('stage') != STAGE_MAE_TRAINING:
		msg = (
			'checkpoint config.stage must be '
			f'{STAGE_MAE_TRAINING!r}; got {config.get("stage")!r}'
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


def _stratigraphy_pretext_metadata(
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
	'EmbeddingExtractionSettings',
	'SurveyEmbeddingResult',
	'build_embedding_metadata',
	'build_model_from_config',
	'extract_survey_embeddings',
	'extraction_settings_from_config',
	'reduce_valid_mask_to_tokens',
	'run_embedding_extraction',
]
