"""Full-volume amplitude MAE encoder embedding extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import numpy as np
import torch

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
	DEFAULT_ZERO_MASK_CONFIG,
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
	checkpoint_config = _checkpoint_config(payload, config)
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
	embeddings = _required_mapping(config, 'embeddings')
	embedding = _optional_mapping(config, 'embedding')
	data = _required_mapping(config, 'data')
	checkpoint_path = _required_path(embeddings, 'checkpoint', 'embeddings')
	output_dir = _required_path(embeddings, 'output_dir', 'embeddings')
	window_size = _xyz_from_mapping(
		embedding,
		'window_size',
		'embedding',
		default=data.get('local_crop_size', (128, 128, 128)),
	)
	overlap = _xyz_from_mapping(
		embedding,
		'overlap',
		'embedding',
		default=(0, 0, 0),
	)
	output_dtype = np.dtype(embedding.get('output_dtype', 'float16'))
	if output_dtype.kind != 'f':
		msg = (
			'embedding.output_dtype must be a floating-point NumPy dtype; '
			f'got {output_dtype}'
		)
		raise TypeError(msg)
	return EmbeddingExtractionSettings(
		checkpoint_path=checkpoint_path,
		output_dir=output_dir,
		window_size_xyz=window_size,
		overlap_xyz=overlap,
		output_dtype=output_dtype,
		batch_size=_positive_int(
			embedding.get('batch_size', 1),
			'embedding.batch_size',
		),
		min_token_valid_fraction=_fraction(
			embedding.get('min_token_valid_fraction', 0.5),
			'embedding.min_token_valid_fraction',
		),
		zero_mask=_zero_mask_for_extraction(
			config,
			checkpoint_config if checkpoint_config is not None else config,
		),
	)


def build_model_from_config(config: Mapping[str, object]) -> AmplitudeMAE3D:
	"""Instantiate an amplitude MAE from a checkpoint config."""
	model = _required_mapping(config, 'model')
	return AmplitudeMAE3D(
		in_channels=_positive_int(model.get('in_channels', 1), 'model.in_channels'),
		out_channels=_positive_int(model.get('out_channels', 1), 'model.out_channels'),
		patch_size_xyz=_xyz_from_mapping(
			model,
			'patch_size',
			'model',
			default=(8, 8, 8),
		),
		encoder_dim=_positive_int(model.get('encoder_dim', 384), 'model.encoder_dim'),
		encoder_depth=_positive_int(
			model.get('encoder_depth', 8),
			'model.encoder_depth',
		),
		encoder_heads=_positive_int(
			model.get('encoder_heads', 6),
			'model.encoder_heads',
		),
		decoder_dim=_positive_int(model.get('decoder_dim', 256), 'model.decoder_dim'),
		decoder_depth=_positive_int(
			model.get('decoder_depth', 4),
			'model.decoder_depth',
		),
		decoder_heads=_positive_int(
			model.get('decoder_heads', 4),
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


def _checkpoint_config(
	payload: Mapping[str, object],
	fallback: Mapping[str, object],
) -> Mapping[str, object]:
	value = payload.get('config')
	if isinstance(value, Mapping):
		return cast('Mapping[str, object]', value)
	return fallback


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
		return DEFAULT_ZERO_MASK_CONFIG
	return _zero_mask_from_mapping(value)


def _zero_mask_for_extraction(
	extraction_config: Mapping[str, object],
	checkpoint_config: Mapping[str, object],
) -> ZeroMaskConfig:
	checkpoint_zero_mask = _zero_mask_from_config(checkpoint_config)
	override_value = _zero_mask_mapping_from_config(extraction_config)
	if override_value is None:
		return checkpoint_zero_mask

	override_zero_mask = _zero_mask_from_mapping(override_value)
	if override_zero_mask != checkpoint_zero_mask:
		msg = (
			'extraction zero_mask override must match checkpoint zero_mask '
			f'settings; got {override_zero_mask!r}, checkpoint has '
			f'{checkpoint_zero_mask!r}'
		)
		raise ValueError(msg)
	return checkpoint_zero_mask


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


def _positive_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{name} must be positive; got {integer!r}'
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
