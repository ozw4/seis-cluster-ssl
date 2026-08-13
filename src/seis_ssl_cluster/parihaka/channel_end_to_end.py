"""Raw-amplitude input and trainable encoder connection for Parihaka Channel."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from seis_ssl_cluster.data.normalization import (
	AmplitudeAgcConfig,
	SurveyNormalizationStats,
	load_normalization_stats,
)
from seis_ssl_cluster.data.schema import CropRequest
from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
from seis_ssl_cluster.data.window_preprocessing import (
	AmplitudePreprocessSettings,
	read_amplitude_crop,
)
from seis_ssl_cluster.data.zero_mask import ZeroMaskConfig
from seis_ssl_cluster.parihaka.channel_tiles import (
	CHANNEL_CONTEXT_HALO_TOKENS,
	CHANNEL_CORE_SIZE_TOKENS,
	CHANNEL_PATCH_SIZE_VOXELS,
	ChannelTileSettings,
	build_channel_tile_targets,
	enumerate_channel_tile_records,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.models.mae import AmplitudeMAE3D
	from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D
	from seis_ssl_cluster.parihaka.channel_data import SectionLines


@dataclass(frozen=True)
class ChannelReferenceArtifact:
	"""Typed subset of a pretrained embedding artifact used as reference only."""

	valid_tokens_path: Path
	metadata_path: Path
	source_amplitude_path: Path
	normalization_stats_path: Path
	volume_shape_xyz: tuple[int, int, int]
	patch_size_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	min_token_valid_fraction: float
	preprocessing: Mapping[str, object]
	zero_mask: Mapping[str, object]
	preprocess_settings: AmplitudePreprocessSettings


def resolve_channel_reference_artifact(
	artifact_dir: str | Path,
	*,
	survey_id: str = 'parihaka',
) -> ChannelReferenceArtifact:
	"""Resolve only valid-token and metadata files from an embedding artifact."""
	root = Path(artifact_dir)
	valid_tokens_path = root / f'{survey_id}.valid_tokens.npy'
	metadata_path = root / f'{survey_id}.metadata.json'
	if not valid_tokens_path.is_file():
		raise FileNotFoundError(
			f'missing reference valid-token mask: {valid_tokens_path}'
		)
	if not metadata_path.is_file():
		raise FileNotFoundError(f'missing reference metadata: {metadata_path}')
	try:
		payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'invalid reference metadata JSON: {metadata_path}') from exc
	if not isinstance(payload, Mapping):
		raise TypeError('reference metadata must be a mapping')
	volume_shape = _positive_int_triplet(payload, 'volume_shape_xyz')
	patch_size = _positive_int_triplet(payload, 'patch_size')
	token_grid_shape = _positive_int_triplet(payload, 'token_grid_shape')
	if patch_size != CHANNEL_PATCH_SIZE_VOXELS:
		raise ValueError(
			f'reference patch_size must be {CHANNEL_PATCH_SIZE_VOXELS!r}'
		)
	expected_grid = tuple(
		(volume_shape[axis] + patch_size[axis] - 1) // patch_size[axis]
		for axis in range(3)
	)
	if token_grid_shape != expected_grid:
		raise ValueError(
			'reference token_grid_shape is incompatible with volume_shape_xyz'
		)
	minimum = _fraction(payload, 'min_token_valid_fraction')
	preprocessing = _mapping(payload, 'preprocessing')
	zero_mask = _mapping(payload, 'zero_mask')
	settings = _preprocess_settings(preprocessing, zero_mask, minimum)
	artifact = ChannelReferenceArtifact(
		valid_tokens_path=valid_tokens_path,
		metadata_path=metadata_path,
		source_amplitude_path=_path(payload, 'source_amplitude_path'),
		normalization_stats_path=_path(payload, 'normalization_stats_path'),
		volume_shape_xyz=volume_shape,
		patch_size_xyz=patch_size,
		token_grid_shape_xyz=token_grid_shape,
		min_token_valid_fraction=minimum,
		preprocessing=dict(preprocessing),
		zero_mask=dict(zero_mask),
		preprocess_settings=settings,
	)
	valid_tokens = np.load(valid_tokens_path, mmap_mode='r', allow_pickle=False)
	if valid_tokens.shape != token_grid_shape:
		raise ValueError('reference valid-token shape does not match metadata')
	if valid_tokens.dtype != np.bool_:
		raise TypeError('reference valid-token mask must have dtype bool')
	return artifact


class ChannelAmplitudeTileDataset(Dataset[dict[str, Any]]):
	"""Read raw amplitude tiles through the shared preprocessing contract."""

	def __init__(  # noqa: PLR0913
		self,
		*,
		reference: ChannelReferenceArtifact,
		labels_path: str | Path,
		lines: SectionLines,
		validation: SectionLines,
		test: SectionLines,
		split: str,
		core_size_tokens: tuple[int, int, int] = CHANNEL_CORE_SIZE_TOKENS,
		context_halo_tokens: tuple[int, int, int] = CHANNEL_CONTEXT_HALO_TOKENS,
		survey_id: str = 'parihaka',
	) -> None:
		"""Validate inputs, open reference arrays, and enumerate core tiles."""
		super().__init__()
		self.reference = reference
		self.labels_path = Path(labels_path)
		self.lines = lines
		self.validation = validation
		self.test = test
		self.split = split
		self.survey_id = survey_id
		self.tile_settings = ChannelTileSettings(
			volume_shape_xyz=reference.volume_shape_xyz,
			token_grid_shape_xyz=reference.token_grid_shape_xyz,
			patch_size_xyz=reference.patch_size_xyz,
			core_size_tokens=core_size_tokens,
			context_halo_tokens=context_halo_tokens,
		)
		self.stats: SurveyNormalizationStats = load_normalization_stats(
			reference.normalization_stats_path
		)
		self._store = NpyMemmapVolumeStore()
		amplitude = self._store.open(reference.source_amplitude_path)
		if tuple(amplitude.shape) != reference.volume_shape_xyz:
			raise ValueError('source amplitude shape does not match reference metadata')
		self._valid_tokens: np.ndarray | None = None
		self._labels: np.ndarray | None = None
		self._open()
		self.records, self.class_counts = enumerate_channel_tile_records(
			valid_tokens=_array(self._valid_tokens),
			labels=_array(self._labels),
			settings=self.tile_settings,
			train=self.lines,
			validation=self.validation,
			test=self.test,
			split=self.split,
		)

	def __len__(self) -> int:
		"""Return the number of non-empty supervised core tiles."""
		return len(self.records)

	def __getitem__(self, index: int) -> dict[str, Any]:
		"""Read and validate one preprocessed halo-padded amplitude tile."""
		self._open()
		record = self.records[index]
		targets = build_channel_tile_targets(
			record=record,
			valid_tokens=_array(self._valid_tokens),
			labels=_array(self._labels),
			settings=self.tile_settings,
			train=self.lines,
			validation=self.validation,
			test=self.test,
			split=self.split,
		)
		start_xyz = tuple(
			targets.input_start_token[axis]
			* self.tile_settings.patch_size_xyz[axis]
			for axis in range(3)
		)
		prepared = read_amplitude_crop(
			request=CropRequest(
				survey_id=self.survey_id,
				start_xyz=start_xyz,
				size_xyz=self.tile_settings.input_size_voxels,
			),
			amplitude_path=self.reference.source_amplitude_path,
			stats=self.stats,
			store=self._store,
			patch_size_xyz=self.reference.patch_size_xyz,
			settings=self.reference.preprocess_settings,
		)
		if not np.array_equal(prepared.token_valid_mask, targets.token_valid_mask):
			raise ValueError(
				'runtime token-valid mask does not match reference valid-token crop'
			)
		return {
			'amplitude': torch.from_numpy(np.ascontiguousarray(prepared.x)),
			'token_valid_mask': torch.from_numpy(
				np.ascontiguousarray(targets.token_valid_mask)
			),
			'labels': torch.from_numpy(targets.labels),
			'supervision_mask': torch.from_numpy(targets.supervision_mask),
			'core_mask': torch.from_numpy(targets.core_mask),
			'tile_id': record.tile_id,
		}

	def __getstate__(self) -> dict[str, object]:
		"""Drop process-owned memory maps before worker serialization."""
		state = self.__dict__.copy()
		state['_valid_tokens'] = None
		state['_labels'] = None
		state['_store'] = NpyMemmapVolumeStore(
			max_open_volumes=self._store.max_open_volumes
		)
		return state

	def _open(self) -> None:
		if self._valid_tokens is None:
			self._valid_tokens = np.load(
				self.reference.valid_tokens_path,
				mmap_mode='r',
				allow_pickle=False,
			)
			self._labels = np.load(
				self.labels_path,
				mmap_mode='r',
				allow_pickle=False,
			)
			if self._labels.dtype != np.int8:
				raise TypeError('prepared labels must have dtype int8')


class ChannelEndToEndModel(nn.Module):
	"""Connect the trainable MAE encoder directly to ``VoxelDecoder3D``."""

	def __init__(self, mae: AmplitudeMAE3D, voxel_decoder: VoxelDecoder3D) -> None:
		"""Validate encoder/decoder geometry and parameter ownership."""
		super().__init__()
		if mae.patch_size_xyz != voxel_decoder.patch_size_xyz:
			raise ValueError('MAE and voxel decoder patch sizes must match')
		if mae.encoder_dim != voxel_decoder.embedding_dim:
			raise ValueError('MAE encoder_dim must match decoder embedding_dim')
		self.mae = mae
		self.voxel_decoder = voxel_decoder
		if self._parameter_ids(self.encoder_parameters()) & self._parameter_ids(
			self.decoder_parameters()
		):
			raise ValueError('encoder and decoder parameter sets must not overlap')

	def forward(
		self,
		amplitude: torch.Tensor,
		token_valid_mask: torch.Tensor,
	) -> torch.Tensor:
		"""Encode the full input grid without masking and decode voxel logits."""
		encoded = self.mae.encode_tokens(amplitude, valid_mask=token_valid_mask)
		tokens = encoded.get('tokens')
		token_grid = encoded.get('token_grid_shape')
		token_valid = encoded.get('token_valid_mask')
		if not isinstance(tokens, torch.Tensor):
			raise TypeError('MAE encoder tokens must be a tensor')
		if not _is_int_triplet(token_grid):
			raise TypeError('MAE encoder token_grid_shape must be an integer triple')
		grid = cast('tuple[int, int, int]', token_grid)
		batch_size = amplitude.shape[0]
		expected_mask_shape = (batch_size, *grid)
		if tuple(token_valid_mask.shape) != expected_mask_shape:
			raise ValueError(
				'encoder output grid does not match decoder input mask grid'
			)
		if tokens.shape != (batch_size, int(np.prod(grid)), self.mae.encoder_dim):
			raise ValueError('MAE encoder token shape does not match its reported grid')
		if not isinstance(token_valid, torch.Tensor):
			raise TypeError('MAE encoder must return a token-valid mask')
		if tuple(token_valid.shape) != (batch_size, int(np.prod(grid))):
			raise ValueError('MAE token-valid output does not match encoder grid')
		embeddings = tokens.reshape(
			batch_size, *grid, self.mae.encoder_dim
		).movedim(-1, 1)
		decoder_mask = token_valid.reshape(batch_size, *grid)
		if tuple(embeddings.shape[2:]) != tuple(decoder_mask.shape[1:]):
			raise ValueError('encoder output grid does not match decoder input grid')
		return self.voxel_decoder(embeddings, decoder_mask)

	def encoder_parameters(self) -> Iterator[nn.Parameter]:
		"""Yield exactly the patch projection and MAE encoder parameters."""
		yield from self.mae.patch_projection.parameters()
		yield from self.mae.encoder.parameters()

	def decoder_parameters(self) -> Iterator[nn.Parameter]:
		"""Yield exactly the voxel decoder parameters."""
		yield from self.voxel_decoder.parameters()

	@staticmethod
	def _parameter_ids(parameters: Iterator[nn.Parameter]) -> set[int]:
		return {id(parameter) for parameter in parameters}


def _preprocess_settings(
	preprocessing: Mapping[str, object],
	zero_mask: Mapping[str, object],
	minimum: float,
) -> AmplitudePreprocessSettings:
	allowed_zero = {
		'enabled',
		'zero_atol',
		'z_sample_influence_radius',
		'xy_trace_influence_radius',
	}
	unexpected = set(zero_mask) - allowed_zero
	if unexpected:
		raise ValueError(f'unsupported zero_mask metadata keys: {sorted(unexpected)!r}')
	try:
		zero_config = ZeroMaskConfig(**zero_mask)
	except TypeError as exc:
		raise TypeError('invalid zero_mask metadata') from exc
	zero_config.validate()
	agc_value = preprocessing.get('amplitude_agc')
	if agc_value is not None and not isinstance(agc_value, Mapping):
		raise TypeError('preprocessing.amplitude_agc must be a mapping')
	agc = AmplitudeAgcConfig.from_mapping(
		cast('Mapping[str, object] | None', agc_value)
	)
	clip = preprocessing.get('normalized_clip_abs')
	if clip is not None and (
		isinstance(clip, bool) or not isinstance(clip, int | float)
	):
		raise TypeError('preprocessing.normalized_clip_abs must be numeric or null')
	if clip is not None and (not np.isfinite(float(clip)) or float(clip) <= 0.0):
		raise ValueError(
			'preprocessing.normalized_clip_abs must be finite and positive'
		)
	finite_mode = preprocessing.get('finite_check_mode', 'strict')
	if not isinstance(finite_mode, str):
		raise TypeError('preprocessing.finite_check_mode must be a string')
	if finite_mode not in {'strict', 'output_only', 'off'}:
		raise ValueError('preprocessing.finite_check_mode is unsupported')
	return AmplitudePreprocessSettings(
		zero_mask=zero_config,
		normalized_clip_abs=None if clip is None else float(clip),
		amplitude_agc=agc,
		min_token_valid_fraction=minimum,
		finite_check_mode=cast('Any', finite_mode),
	)


def _positive_int_triplet(
	payload: Mapping[str, object], key: str
) -> tuple[int, int, int]:
	value = payload.get(key)
	if (
		isinstance(value, str)
		or not isinstance(value, Sequence)
		or len(value) != 3
		or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
		or any(item <= 0 for item in value)
	):
		raise TypeError(f'reference metadata {key} must be a positive integer triple')
	return (int(value[0]), int(value[1]), int(value[2]))


def _fraction(payload: Mapping[str, object], key: str) -> float:
	value = payload.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'reference metadata {key} must be numeric')
	result = float(value)
	if not np.isfinite(result) or not 0.0 <= result <= 1.0:
		raise ValueError(f'reference metadata {key} must be in [0, 1]')
	return result


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = payload.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'reference metadata {key} must be a mapping')
	return value


def _path(payload: Mapping[str, object], key: str) -> Path:
	value = payload.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'reference metadata {key} must be a non-empty path string')
	return Path(value)


def _array(value: np.ndarray | None) -> np.ndarray:
	if value is None:
		raise RuntimeError('dataset memory map is not open')
	return value


def _is_int_triplet(value: object) -> bool:
	return (
		isinstance(value, tuple)
		and len(value) == 3
		and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
	)


__all__ = [
	'ChannelAmplitudeTileDataset',
	'ChannelEndToEndModel',
	'ChannelReferenceArtifact',
	'resolve_channel_reference_artifact',
]
