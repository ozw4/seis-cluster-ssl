"""Full-volume amplitude MAE encoder embedding extraction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
)
from seis_ssl_cluster.data.crop_sampler import (
	expand_request_with_margin,
	required_zero_mask_margin_xyz,
)
from seis_ssl_cluster.data.normalization import (
	load_normalization_stats,
	normalize_amplitude,
)
from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest, read_manifest_json
from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
from seis_ssl_cluster.data.zero_mask import (
	ZeroMaskConfig,
	compute_zero_amplitude_invalid_mask,
)
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
from seis_ssl_cluster.models.mae.patching import patchify_3d
from seis_ssl_cluster.training.checkpoint import load_checkpoint

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
	*DEFAULT_MAE_TRAIN_OPTIONS,
)


@dataclass(frozen=True)
class EmbeddingExtractionSettings:
	"""Validated full-volume extraction settings."""

	checkpoint_path: Path
	output_dir: Path
	window_size_xyz: XYZ
	overlap_xyz: XYZ
	output_dtype: np.dtype
	batch_size: int
	min_token_valid_fraction: float
	zero_mask: ZeroMaskConfig


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
) -> list[SurveyEmbeddingResult]:
	"""Extract MAE encoder embeddings for all surveys in a manifest."""
	checkpoint_path = _checkpoint_path(config)
	manifests = read_manifest_json(_manifest_path(config))
	if not manifests:
		msg = 'embedding extraction manifest is empty'
		raise ValueError(msg)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	checkpoint_config = _checkpoint_config(payload)
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
	return [
		extract_survey_embeddings(
			manifest,
			model=model,
			store=store,
			settings=settings,
			checkpoint_config=checkpoint_config,
			checkpoint_sha256=checkpoint_sha256,
			device=resolved_device,
			skip_existing=skip_existing,
		)
		for manifest in manifests
	]


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
		batch_size=_positive_int(
			embedding.get('batch_size'),
			'embedding.batch_size',
		),
		min_token_valid_fraction=_fraction(
			embedding.get('min_token_valid_fraction'),
			'embedding.min_token_valid_fraction',
		),
		zero_mask=_zero_mask_from_config(checkpoint_config),
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
	checkpoint_sha256: str,
	device: torch.device,
	skip_existing: bool,
) -> SurveyEmbeddingResult:
	"""Extract and write embeddings for one survey manifest."""
	manifest.validate()
	amplitude_path = _resolve_manifest_path(manifest, manifest.amplitude.path)
	stats_path = _resolve_manifest_path(
		manifest,
		manifest.amplitude.normalization_stats_path,
	)
	stats = load_normalization_stats(stats_path)
	patch_size = model.patch_size_xyz
	token_grid = token_grid_shape_xyz(manifest.amplitude.shape_xyz, patch_size)
	metadata = build_embedding_metadata(
		manifest=manifest,
		amplitude_path=amplitude_path,
		stats_path=stats_path,
		settings=settings,
		checkpoint_config=checkpoint_config,
		checkpoint_sha256=checkpoint_sha256,
		model=model,
		token_grid_shape=token_grid,
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
	windows = list(
		iter_sliding_windows(
			manifest.amplitude.shape_xyz,
			window_size_xyz=settings.window_size_xyz,
			overlap_xyz=settings.overlap_xyz,
			patch_size_xyz=patch_size,
		),
	)
	for batch_start in range(0, len(windows), settings.batch_size):
		_process_window_batch(
			windows[batch_start : batch_start + settings.batch_size],
			manifest=manifest,
			amplitude_path=amplitude_path,
			stats=stats,
			store=store,
			model=model,
			settings=settings,
			device=device,
			merger=merger,
		)
	merger.write_average(
		embedding_path=paths.embeddings_tmp,
		valid_tokens_path=paths.valid_tokens_tmp,
		output_dtype=settings.output_dtype,
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
	checkpoint_sha256: str | None = None,
	model: AmplitudeMAE3D,
	token_grid_shape: XYZ,
) -> dict[str, object]:
	"""Return deterministic metadata for one survey output."""
	resolved_checkpoint_sha256 = (
		file_sha256(settings.checkpoint_path)
		if checkpoint_sha256 is None
		else checkpoint_sha256
	)
	return {
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
		'min_token_valid_fraction': settings.min_token_valid_fraction,
		'zero_mask': {
			'enabled': settings.zero_mask.enabled,
			'zero_atol': settings.zero_mask.zero_atol,
			'z_sample_influence_radius': settings.zero_mask.z_sample_influence_radius,
			'xy_trace_influence_radius': settings.zero_mask.xy_trace_influence_radius,
		},
	}


def _process_window_batch(  # noqa: PLR0913
	windows: Sequence[SlidingWindow],
	*,
	manifest: SurveyManifest,
	amplitude_path: Path,
	stats: object,
	store: NpyMemmapVolumeStore,
	model: AmplitudeMAE3D,
	settings: EmbeddingExtractionSettings,
	device: torch.device,
	merger: EmbeddingMerger,
) -> None:
	prepared = [
		_read_window(
			window,
			manifest=manifest,
			amplitude_path=amplitude_path,
			stats=stats,
			store=store,
			settings=settings,
			patch_size_xyz=model.patch_size_xyz,
		)
		for window in windows
	]
	usable = [item for item in prepared if item[2].any()]
	if not usable:
		return

	x = torch.from_numpy(np.stack([item[1] for item in usable], axis=0)).to(
		device=device,
		dtype=_model_floating_dtype(model),
	)
	token_masks = torch.from_numpy(
		np.stack([item[2] for item in usable], axis=0),
	).to(device)
	with torch.no_grad():
		output = model.encode_tokens(x, valid_mask=token_masks)
	tokens = (
		cast('torch.Tensor', output['tokens'])
		.detach()
		.to(dtype=torch.float32)
		.cpu()
		.numpy()
	)
	window_token_shape = cast('tuple[int, int, int]', output['token_grid_shape'])
	for index, (window, _x, token_valid) in enumerate(usable):
		merger.add_window(
			window,
			patch_size_xyz=model.patch_size_xyz,
			token_embeddings=tokens[index].reshape(
				*window_token_shape,
				model.encoder_dim,
			),
			token_valid_mask=token_valid,
		)


def _read_window(  # noqa: PLR0913
	window: SlidingWindow,
	*,
	manifest: SurveyManifest,
	amplitude_path: Path,
	stats: object,
	store: NpyMemmapVolumeStore,
	settings: EmbeddingExtractionSettings,
	patch_size_xyz: XYZ,
) -> tuple[SlidingWindow, np.ndarray, np.ndarray]:
	margin_xyz = _zero_mask_margin_xyz(settings.zero_mask)
	request = CropRequest(
		survey_id=manifest.survey_id,
		start_xyz=window.start_xyz,
		size_xyz=window.size_xyz,
	)
	compute_request, payload_slices = expand_request_with_margin(request, margin_xyz)
	raw_compute, compute_valid_mask = store.read_crop_with_padding(
		amplitude_path,
		compute_request.start_xyz,
		compute_request.size_xyz,
	)
	raw_crop = raw_compute[payload_slices].astype(np.float32, copy=False)
	source_valid_mask = compute_valid_mask[payload_slices]
	zero_invalid = compute_zero_amplitude_invalid_mask(
		raw_compute,
		valid_mask=compute_valid_mask,
		config=settings.zero_mask,
	)[payload_slices]
	local_valid_mask = np.logical_and(source_valid_mask, ~zero_invalid)
	amplitude_norm = normalize_amplitude(raw_crop, stats)
	amplitude_norm = amplitude_norm.astype(np.float32, copy=True)
	amplitude_norm[~local_valid_mask] = 0.0
	token_valid_mask = reduce_valid_mask_to_tokens(
		local_valid_mask,
		patch_size_xyz=patch_size_xyz,
		min_valid_fraction=settings.min_token_valid_fraction,
	)
	return window, amplitude_norm[np.newaxis, ...], token_valid_mask


def reduce_valid_mask_to_tokens(
	valid_mask_xyz: np.ndarray,
	*,
	patch_size_xyz: Sequence[int],
	min_valid_fraction: float,
) -> np.ndarray:
	"""Reduce a voxel-valid mask to token validity by patch valid fraction."""
	patch = _validate_positive_xyz(patch_size_xyz, 'patch_size_xyz')
	threshold = _fraction(min_valid_fraction, 'min_valid_fraction')
	mask = np.asarray(valid_mask_xyz, dtype=bool)
	if mask.ndim != 3:
		msg = f'valid_mask_xyz must be 3D; got shape={mask.shape!r}'
		raise ValueError(msg)
	patches = patchify_3d(
		torch.from_numpy(mask[np.newaxis, np.newaxis, ...].astype(np.float32)),
		patch,
	).numpy()
	fractions = patches.reshape(-1, patch[0] * patch[1] * patch[2]).mean(axis=1)
	token_grid = tuple(
		axis // patch_axis
		for axis, patch_axis in zip(mask.shape, patch, strict=True)
	)
	return (fractions.reshape(token_grid) >= threshold).astype(bool, copy=False)


def _checkpoint_config(payload: Mapping[str, object]) -> Mapping[str, object]:
	value = payload.get('config')
	if isinstance(value, Mapping):
		config = cast('Mapping[str, object]', value)
		_validate_checkpoint_resolved_config(config)
		return config
	msg = 'checkpoint is missing a resolved config'
	raise TypeError(msg)


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
		(*DEFAULT_MAE_DATA_OPTIONS, 'local_crop_size'),
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
		set(config)
		& {'data', 'model', 'masking', 'loss', 'train', 'zero_mask'},
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


def _zero_mask_margin_xyz(config: ZeroMaskConfig) -> XYZ:
	if not config.enabled:
		return (0, 0, 0)
	return required_zero_mask_margin_xyz(
		z_sample_influence_radius=config.z_sample_influence_radius,
		xy_trace_influence_radius=config.xy_trace_influence_radius,
	)


def _resolve_manifest_path(manifest: SurveyManifest, path: Path) -> Path:
	if path.is_absolute():
		return path
	return manifest.root / path


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
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
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
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
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
