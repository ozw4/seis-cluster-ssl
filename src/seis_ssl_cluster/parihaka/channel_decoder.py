"""One-job frozen-embedding decoder for Parihaka Channel estimation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from seis_ssl_cluster.embedding.writer import (
	EmbeddingOutputPaths,
	output_paths,
)
from seis_ssl_cluster.models.voxel_decoder import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
	VoxelDecoder3D,
	validate_context_halo_tokens,
	validate_voxel_decoder_architecture,
)
from seis_ssl_cluster.parihaka.channel_checkpoints import (
	CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX,
	CHANNEL_PRETRAINED_MODEL_TAG,
	CHANNEL_RANDOM_ENCODER_SEED,
	inspect_channel_model_sources,
)
from seis_ssl_cluster.parihaka.channel_data import (
	CHANNEL_AXIS_MAPPING,
	ChannelLayouts,
	SectionLines,
	channel_test_definition,
	common_reserved_training_lines,
	inspect_prepared_label_identity,
	load_channel_layouts,
	selected_training_lines,
)
from seis_ssl_cluster.parihaka.channel_tiles import (
	CHANNEL_CONTEXT_HALO_TOKENS,
	CHANNEL_CORE_SIZE_TOKENS,
	ChannelTileRecord,
	ChannelTileSettings,
	build_channel_tile_targets,
	enumerate_channel_tile_records,
)
from seis_ssl_cluster.training.voxel_decoder.epoch import train_voxel_decoder_one_epoch
from seis_ssl_cluster.training.voxel_decoder.losses import (
	balanced_class_weights_from_counts,
	masked_weighted_voxel_cross_entropy,
)

LATEST_NAME = 'latest.pt'
BEST_NAME = 'best.pt'
HISTORY_NAME = 'history.csv'
METRICS_NAME = 'metrics.json'

_PAIRED_EMBEDDING_METADATA_KEYS = (
	'survey_id',
	'source_amplitude_path',
	'volume_shape_xyz',
	'model_geometry',
	'patch_size',
	'token_grid_shape',
	'window_size',
	'overlap',
	'output_dtype',
	'min_token_valid_fraction',
	'normalization_stats_path',
	'preprocessing',
	'zero_mask',
	'precision',
	'pretraining_objective',
)


@dataclass(frozen=True)
class DecoderArchitecture:
	"""Fixed lightweight decoder settings."""

	embedding_dim: int
	class_count: int
	hidden_channels: tuple[int, ...]
	upsample_factors: tuple[tuple[int, int, int], ...]
	upsample_mode: str
	normalization: str
	spec: str = VOXEL_DECODER_SPEC


@dataclass(frozen=True)
class DecoderTrain:
	"""Fixed single-seed training settings."""

	epochs: int
	batch_size: int
	learning_rate: float
	weight_decay: float
	class_weight: str
	sampling_mode: str
	seed: int
	amp: bool
	gradient_clip_norm: float


@dataclass(frozen=True)
class DecoderTiles:
	"""Deterministic core and context geometry in token coordinates."""

	core_size_tokens: tuple[int, int, int]
	context_halo_tokens: tuple[int, int, int]


@dataclass(frozen=True)
class ChannelDecoderConfig:
	"""Resolved common settings for all 30 jobs."""

	survey_id: str
	labels: Path
	labels_metadata: Path
	pretrained_embeddings: Path
	random_embeddings: Path
	runs_root: Path
	decoder: DecoderArchitecture
	train: DecoderTrain
	tiles: DecoderTiles


@dataclass(frozen=True)
class EmbeddingGeometry:
	"""Validated paired embedding inputs."""

	pretrained: EmbeddingOutputPaths
	random: EmbeddingOutputPaths
	volume_shape_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	patch_size_xyz: tuple[int, int, int]
	embedding_shape: tuple[int, int, int, int]
	embedding_dim: int
	pretrained_metadata: Mapping[str, object]
	random_metadata: Mapping[str, object]
	pretrained_model_source: Mapping[str, object]
	random_model_source: Mapping[str, object]


@dataclass(frozen=True)
class ChannelDecoderPlan:
	"""Fully validated one-job plan."""

	config: ChannelDecoderConfig
	model: str
	layout_id: str
	data_size: str
	output_dir: Path
	geometry: EmbeddingGeometry
	layouts: ChannelLayouts
	train_lines: SectionLines
	reserved_training_lines: SectionLines
	prepared_label_identity: Mapping[str, object]
	class_counts: tuple[int, int]
	class_weights: tuple[float, float]
	split_counts: Mapping[str, tuple[int, int]]
	tile_counts: Mapping[str, int]


def channel_decoder_config_from_mapping(
	config: Mapping[str, object],
) -> ChannelDecoderConfig:
	"""Resolve and enforce the benchmark's fixed scientific settings."""
	dataset = _mapping(config, 'dataset')
	inputs = _mapping(config, 'inputs')
	embeddings = _mapping(config, 'embeddings')
	outputs = _mapping(config, 'outputs')
	decoder = _decoder(_mapping(config, 'decoder'))
	train = _train(_mapping(config, 'train'))
	tiles = _tiles(_mapping(config, 'tiles'))
	survey_id = dataset.get('survey_id')
	if not isinstance(survey_id, str) or not survey_id:
		raise ValueError('dataset.survey_id must be a non-empty string')
	return ChannelDecoderConfig(
		survey_id=survey_id,
		labels=_absolute_path(inputs, 'labels_npy', 'inputs'),
		labels_metadata=_absolute_path(inputs, 'labels_metadata_json', 'inputs'),
		pretrained_embeddings=_absolute_path(
			embeddings, 'pretrained_dir', 'embeddings'
		),
		random_embeddings=_absolute_path(embeddings, 'random_dir', 'embeddings'),
		runs_root=_absolute_path(outputs, 'runs_root', 'outputs'),
		decoder=decoder,
		train=train,
		tiles=tiles,
	)


def inspect_embedding_pair(  # noqa: C901
	config: ChannelDecoderConfig,
) -> EmbeddingGeometry:
	"""Require scientifically paired pretrained/random embedding inputs."""
	pretrained = output_paths(config.pretrained_embeddings, config.survey_id)
	random_paths = output_paths(config.random_embeddings, config.survey_id)
	for paths in (pretrained, random_paths):
		for path in (paths.embeddings, paths.valid_tokens, paths.metadata):
			if not path.is_file():
				raise FileNotFoundError(f'missing Channel decoder input: {path}')
	pretrained_meta = _read_json(pretrained.metadata)
	random_meta = _read_json(random_paths.metadata)
	pretrained_array = np.load(pretrained.embeddings, mmap_mode='r', allow_pickle=False)
	random_array = np.load(random_paths.embeddings, mmap_mode='r', allow_pickle=False)
	if pretrained_array.shape != random_array.shape:
		raise ValueError('pretrained/random embedding shape mismatch')
	if (
		pretrained_array.ndim != 4
		or not np.issubdtype(pretrained_array.dtype, np.floating)
		or not np.issubdtype(random_array.dtype, np.floating)
	):
		raise TypeError('embeddings must be floating [TX,TY,TZ,D] arrays')
	_validate_embedding_pair_metadata(
		pretrained_meta,
		random_meta,
		pretrained_dtype=pretrained_array.dtype,
		random_dtype=random_array.dtype,
	)
	pretrained_model_source, random_model_source = inspect_channel_model_sources(
		pretrained_meta,
		random_meta,
	)
	patch = _triplet(pretrained_meta.get('patch_size'), 'embedding patch_size')
	token_grid = _triplet(
		pretrained_meta.get('token_grid_shape'), 'embedding token_grid_shape'
	)
	volume = _triplet(
		pretrained_meta.get('volume_shape_xyz'), 'embedding volume_shape_xyz'
	)
	dimension = _embedding_dim(pretrained_meta)
	if (
		dimension != _embedding_dim(random_meta)
		or dimension != pretrained_array.shape[3]
	):
		raise ValueError('pretrained/random embedding dimension mismatch')
	if tuple(pretrained_array.shape[:3]) != token_grid:
		raise ValueError('embedding array shape does not match token-grid metadata')
	expected_tokens = tuple(
		(size + patch_size - 1) // patch_size
		for size, patch_size in zip(volume, patch, strict=True)
	)
	if expected_tokens != token_grid:
		raise ValueError('token-grid shape is inconsistent with volume and patch size')
	pretrained_valid = np.load(
		pretrained.valid_tokens, mmap_mode='r', allow_pickle=False
	)
	random_valid = np.load(random_paths.valid_tokens, mmap_mode='r', allow_pickle=False)
	if (
		pretrained_valid.dtype != np.bool_
		or random_valid.dtype != np.bool_
		or pretrained_valid.shape != token_grid
		or random_valid.shape != token_grid
	):
		raise TypeError('valid-token masks must be bool with the token-grid shape')
	if not np.array_equal(pretrained_valid, random_valid):
		raise ValueError('pretrained/random valid-token mask mismatch')
	return EmbeddingGeometry(
		pretrained=pretrained,
		random=random_paths,
		volume_shape_xyz=volume,
		token_grid_shape_xyz=token_grid,
		patch_size_xyz=patch,
		embedding_shape=tuple(int(item) for item in pretrained_array.shape),
		embedding_dim=dimension,
		pretrained_metadata=dict(pretrained_meta),
		random_metadata=dict(random_meta),
		pretrained_model_source=pretrained_model_source,
		random_model_source=random_model_source,
	)


def _validate_embedding_pair_metadata(  # noqa: C901, PLR0912
	pretrained_meta: Mapping[str, object],
	random_meta: Mapping[str, object],
	*,
	pretrained_dtype: np.dtype[Any],
	random_dtype: np.dtype[Any],
) -> None:
	for key in _PAIRED_EMBEDDING_METADATA_KEYS:
		if key not in pretrained_meta:
			raise ValueError(f'pretrained embedding metadata missing {key}')
		if key not in random_meta:
			raise ValueError(f'random embedding metadata missing {key}')
	pretrained_metadata_dtype = _metadata_dtype(pretrained_meta, 'pretrained')
	random_metadata_dtype = _metadata_dtype(random_meta, 'random')
	if pretrained_dtype != pretrained_metadata_dtype:
		raise TypeError(
			'pretrained embedding array dtype does not match metadata output_dtype'
		)
	if random_dtype != random_metadata_dtype:
		raise TypeError(
			'random embedding array dtype does not match metadata output_dtype'
		)
	if pretrained_dtype != random_dtype:
		raise TypeError('pretrained/random embedding array dtype mismatch')
	for key in _PAIRED_EMBEDDING_METADATA_KEYS:
		if pretrained_meta[key] != random_meta[key]:
			raise ValueError(
				f'pretrained/random embedding metadata {key} mismatch'
			)
	pretrained_sha = pretrained_meta.get('checkpoint_sha256')
	random_sha = random_meta.get('checkpoint_sha256')
	pretrained_checkpoint = pretrained_meta.get('checkpoint_path')
	random_checkpoint = random_meta.get('checkpoint_path')
	if not isinstance(pretrained_checkpoint, str) or not pretrained_checkpoint:
		raise ValueError('pretrained embedding metadata missing checkpoint_path')
	if not isinstance(random_checkpoint, str) or not random_checkpoint:
		raise ValueError('random embedding metadata missing checkpoint_path')
	if not isinstance(pretrained_sha, str) or not pretrained_sha:
		raise ValueError('pretrained embedding metadata missing checkpoint_sha256')
	if not isinstance(random_sha, str) or not random_sha:
		raise ValueError('random embedding metadata missing checkpoint_sha256')
	if pretrained_sha == random_sha:
		raise ValueError('pretrained/random checkpoint_sha256 must differ')


def _metadata_dtype(metadata: Mapping[str, object], model: str) -> np.dtype[Any]:
	value = metadata.get('output_dtype')
	if not isinstance(value, str) or not value:
		raise TypeError(f'{model} embedding metadata output_dtype must be a string')
	try:
		return np.dtype(value)
	except TypeError as exc:
		raise TypeError(
			f'{model} embedding metadata output_dtype is invalid: {value!r}'
		) from exc


def inspect_channel_decoder_job(
	config: ChannelDecoderConfig,
	*,
	model: str,
	layout_id: str,
	data_size: str,
	layout_config: str | Path,
) -> ChannelDecoderPlan:
	"""Validate paired inputs, explicit layout, geometry, and supervision counts."""
	if model not in {'pretrained', 'random'}:
		raise ValueError("model must be 'pretrained' or 'random'")
	geometry = inspect_embedding_pair(config)
	if geometry.embedding_dim != config.decoder.embedding_dim:
		raise ValueError('decoder.embedding_dim does not match paired embeddings')
	validate_voxel_decoder_architecture(
		hidden_channels=config.decoder.hidden_channels,
		upsample_factors=config.decoder.upsample_factors,
		patch_size_xyz=geometry.patch_size_xyz,
	)
	validate_context_halo_tokens(
		context_halo_tokens=config.tiles.context_halo_tokens,
		core_size_tokens=config.tiles.core_size_tokens,
		token_grid_shape_xyz=geometry.token_grid_shape_xyz,
		upsample_factors=config.decoder.upsample_factors,
	)
	labels = np.load(config.labels, mmap_mode='r', allow_pickle=False)
	if labels.shape != geometry.volume_shape_xyz:
		raise ValueError('prepared label shape does not match embedding volume shape')
	if labels.dtype != np.int8:
		raise TypeError('prepared labels must have dtype int8')
	prepared_label_identity = inspect_prepared_label_identity(
		config.labels, config.labels_metadata
	)
	layouts = load_channel_layouts(layout_config, geometry.volume_shape_xyz)
	train_lines = selected_training_lines(layouts, layout_id, data_size)
	reserved_training_lines = common_reserved_training_lines(layouts)
	split_counts: dict[str, tuple[int, int]] = {}
	tile_counts: dict[str, int] = {}
	for split in ('train', 'validation', 'test'):
		dataset = ChannelTileDataset(
			embedding_path=geometry.pretrained.embeddings,
			valid_tokens_path=geometry.pretrained.valid_tokens,
			labels_path=config.labels,
			geometry=geometry,
			lines=train_lines,
			validation=layouts.validation,
			reserved_training=reserved_training_lines,
			split=split,
			tiles=config.tiles,
		)
		split_counts[split] = dataset.class_counts
		tile_counts[split] = len(dataset)
	for split in ('train', 'validation', 'test'):
		if any(count == 0 for count in split_counts[split]):
			raise ValueError(
				f'{split} sections must contain both Channel and non-Channel voxels'
			)
	if any(tile_counts[split] == 0 for split in ('train', 'validation', 'test')):
		raise ValueError('train, validation, and test must each contain valid voxels')
	weights = balanced_class_weights_from_counts(split_counts['train'])
	output_dir = (
		config.runs_root
		/ f'model={model}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)
	return ChannelDecoderPlan(
		config=config,
		model=model,
		layout_id=layout_id,
		data_size=data_size,
		output_dir=output_dir,
		geometry=geometry,
		layouts=layouts,
		train_lines=train_lines,
		reserved_training_lines=reserved_training_lines,
		prepared_label_identity=prepared_label_identity,
		class_counts=split_counts['train'],
		class_weights=tuple(float(item) for item in weights.tolist()),
		split_counts=split_counts,
		tile_counts=tile_counts,
	)


class ChannelTileDataset(Dataset[dict[str, Any]]):
	"""Memory-mapped tiles whose section masks are generated at access time."""

	def __init__(  # noqa: PLR0913
		self,
		*,
		embedding_path: Path,
		valid_tokens_path: Path,
		labels_path: Path,
		geometry: EmbeddingGeometry,
		lines: SectionLines,
		validation: SectionLines,
		reserved_training: SectionLines,
		split: str,
		tiles: DecoderTiles,
	) -> None:
		"""Open the three memory maps and enumerate supervised core tiles."""
		super().__init__()
		self.embedding_path = embedding_path
		self.valid_tokens_path = valid_tokens_path
		self.labels_path = labels_path
		self.geometry = geometry
		self.lines = lines
		self.validation = validation
		self.reserved_training = reserved_training
		self.split = split
		self.tile_settings = tiles
		self.shared_tile_settings = ChannelTileSettings(
			volume_shape_xyz=geometry.volume_shape_xyz,
			token_grid_shape_xyz=geometry.token_grid_shape_xyz,
			patch_size_xyz=geometry.patch_size_xyz,
			core_size_tokens=tiles.core_size_tokens,
			context_halo_tokens=tiles.context_halo_tokens,
		)
		self._embeddings: np.ndarray | None = None
		self._valid_tokens: np.ndarray | None = None
		self._labels: np.ndarray | None = None
		self._open()
		self.records, self.class_counts = self._build_records()

	def __len__(self) -> int:
		"""Return the number of non-empty supervised tiles."""
		return len(self.records)

	def __getitem__(self, index: int) -> dict[str, Any]:
		"""Load one halo-padded tile and generate its section mask."""
		self._open()
		record = self.records[index]
		embeddings = _array(self._embeddings)
		valid_tokens = _array(self._valid_tokens)
		labels = _array(self._labels)
		targets = build_channel_tile_targets(
			record=record,
			valid_tokens=valid_tokens,
			labels=labels,
			settings=self.shared_tile_settings,
			train=self.lines,
			validation=self.validation,
			reserved_training=self.reserved_training,
			split=self.split,
		)
		token_source = _slices(
			targets.token_source_start, targets.token_source_stop
		)
		token_destination = _slices(
			targets.token_destination_start, targets.token_destination_stop
		)
		embedding_crop = np.zeros(
			(*self.shared_tile_settings.input_size_tokens, self.geometry.embedding_dim),
			dtype=np.float32,
		)
		embedding_crop[token_destination] = np.asarray(
			embeddings[token_source], dtype=np.float32
		)
		return {
			'embeddings': torch.from_numpy(
				np.ascontiguousarray(np.moveaxis(embedding_crop, -1, 0))
			),
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
		for key in ('_embeddings', '_valid_tokens', '_labels'):
			state[key] = None
		return state

	def _open(self) -> None:
		if self._embeddings is None:
			self._embeddings = np.load(
				self.embedding_path, mmap_mode='r', allow_pickle=False
			)
			self._valid_tokens = np.load(
				self.valid_tokens_path, mmap_mode='r', allow_pickle=False
			)
			self._labels = np.load(self.labels_path, mmap_mode='r', allow_pickle=False)

	def _build_records(
		self,
	) -> tuple[tuple[ChannelTileRecord, ...], tuple[int, int]]:
		return enumerate_channel_tile_records(
			valid_tokens=_array(self._valid_tokens),
			labels=_array(self._labels),
			settings=self.shared_tile_settings,
			train=self.lines,
			validation=self.validation,
			reserved_training=self.reserved_training,
			split=self.split,
		)


def deterministic_tile_order(tile_count: int, seed: int, epoch: int) -> tuple[int, ...]:
	"""Return the all-tiles-once order shared by both encoder conditions."""
	if tile_count <= 0:
		raise ValueError('tile_count must be positive')
	generator = torch.Generator().manual_seed(seed + epoch)
	return tuple(int(item) for item in torch.randperm(tile_count, generator=generator))


def decoder_initial_state_sha256(
	architecture: DecoderArchitecture, patch_size_xyz: Sequence[int], seed: int
) -> str:
	"""Return a stable digest of the seeded decoder initialization."""
	_configure_determinism()
	_seed_everything(seed)
	decoder = _make_decoder(architecture, patch_size_xyz)
	digest = hashlib.sha256()
	for key, value in decoder.state_dict().items():
		digest.update(key.encode())
		digest.update(value.detach().cpu().contiguous().numpy().tobytes())
	return digest.hexdigest()


def channel_metrics(confusion: np.ndarray) -> dict[str, float | list[list[int]]]:
	"""Compute the fixed binary Channel metrics from a 2x2 confusion matrix."""
	array = np.asarray(confusion, dtype=np.int64)
	if array.shape != (2, 2) or (array < 0).any():
		raise ValueError('confusion must be a non-negative 2x2 matrix')
	tn, fp, fn, tp = (
		int(array[0, 0]),
		int(array[0, 1]),
		int(array[1, 0]),
		int(array[1, 1]),
	)
	precision = _ratio(tp, tp + fp)
	recall = _ratio(tp, tp + fn)
	return {
		'channel_iou': _ratio(tp, tp + fp + fn),
		'channel_f1': _ratio(2 * tp, 2 * tp + fp + fn),
		'channel_precision': precision,
		'channel_recall': recall,
		'balanced_accuracy': 0.5 * (recall + _ratio(tn, tn + fp)),
		'confusion_matrix': array.tolist(),
	}


def run_channel_decoder_job(  # noqa: C901, PLR0915
	plan: ChannelDecoderPlan,
	*,
	device: str = 'auto',
	max_steps: int | None = None,
	resume: str | Path | None = None,
) -> Path | None:
	"""Train one decoder, select by validation Channel IoU, and test once."""
	if max_steps is not None and (
		not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
	):
		raise ValueError('max_steps must be a positive integer')
	resume_path = None if resume is None else Path(resume)
	_validate_output(plan.output_dir, resume_path)
	run_device = _resolve_device(device)
	_configure_determinism()
	_seed_everything(plan.config.train.seed)
	embedding_paths = (
		plan.geometry.pretrained if plan.model == 'pretrained' else plan.geometry.random
	)
	datasets = {
		split: ChannelTileDataset(
			embedding_path=embedding_paths.embeddings,
			valid_tokens_path=embedding_paths.valid_tokens,
			labels_path=plan.config.labels,
			geometry=plan.geometry,
			lines=plan.train_lines,
			validation=plan.layouts.validation,
			reserved_training=plan.reserved_training_lines,
			split=split,
			tiles=plan.config.tiles,
		)
		for split in ('train', 'validation', 'test')
	}
	decoder = _make_decoder(plan.config.decoder, plan.geometry.patch_size_xyz).to(
		run_device
	)
	optimizer = torch.optim.AdamW(
		decoder.parameters(),
		lr=plan.config.train.learning_rate,
		weight_decay=plan.config.train.weight_decay,
	)
	scaler = (
		torch.amp.GradScaler('cuda', enabled=True)
		if plan.config.train.amp and run_device.type == 'cuda'
		else None
	)
	weights = torch.tensor(plan.class_weights, dtype=torch.float32, device=run_device)
	history: list[dict[str, object]] = []
	best_epoch: int | None = None
	best_iou = -1.0
	global_step = 0
	start_epoch = 0
	start_position = 0
	train_confusion = np.zeros((2, 2), dtype=np.int64)
	train_loss_sum = 0.0
	train_voxels = 0
	identity = _run_identity(plan)
	if resume_path is not None:
		payload = torch.load(resume_path, map_location=run_device, weights_only=False)
		if payload.get('run_identity') != identity:
			raise ValueError('resume checkpoint does not match this Channel job')
		if payload.get('completed') is True:
			raise ValueError('completed Channel job cannot be resumed')
		decoder.load_state_dict(payload['model_state_dict'])
		optimizer.load_state_dict(payload['optimizer_state_dict'])
		if scaler is not None and payload.get('scaler_state_dict') is not None:
			scaler.load_state_dict(payload['scaler_state_dict'])
		history = [dict(row) for row in payload['history']]
		best_epoch = payload['best_epoch']
		best_iou = float(payload['best_iou'])
		global_step = int(payload['global_step'])
		start_epoch = int(payload['epoch'])
		start_position = int(payload['next_position'])
		train_confusion = np.asarray(payload['train_confusion'], dtype=np.int64)
		train_loss_sum = float(payload['train_loss_sum'])
		train_voxels = int(payload['train_voxels'])
	plan.output_dir.mkdir(parents=True, exist_ok=True)
	for epoch in range(start_epoch, plan.config.train.epochs):
		order = deterministic_tile_order(
			len(datasets['train']), plan.config.train.seed, epoch
		)
		for position in range(start_position, len(order)):
			if max_steps is not None and global_step >= max_steps:
				_save_latest(
					plan,
					decoder,
					optimizer,
					scaler,
					identity,
					history,
					best_epoch,
					best_iou,
					global_step,
					epoch,
					position,
					train_confusion,
					train_loss_sum,
					train_voxels,
					completed=False,
				)
				_write_history(plan.output_dir / HISTORY_NAME, history)
				return None
			loader = DataLoader(
				Subset(datasets['train'], [order[position]]),
				batch_size=1,
				shuffle=False,
			)
			metrics = train_voxel_decoder_one_epoch(
				decoder=decoder,
				dataloader=loader,
				optimizer=optimizer,
				class_weights=weights,
				class_ids=(0, 1),
				device=run_device,
				amp_enabled=plan.config.train.amp,
				scaler=scaler,
				grad_clip_norm=plan.config.train.gradient_clip_norm,
			)
			count = int(metrics['supervised_voxel_count'])
			train_confusion += np.asarray(metrics['confusion_matrix'], dtype=np.int64)
			train_loss_sum += float(metrics['loss']) * count
			train_voxels += count
			global_step += 1
			if (
				max_steps is not None
				and global_step >= max_steps
				and position + 1 < len(order)
			):
				_save_latest(
					plan,
					decoder,
					optimizer,
					scaler,
					identity,
					history,
					best_epoch,
					best_iou,
					global_step,
					epoch,
					position + 1,
					train_confusion,
					train_loss_sum,
					train_voxels,
					completed=False,
				)
				_write_history(plan.output_dir / HISTORY_NAME, history)
				return None
		validation_metrics = _evaluate(
			decoder,
			datasets['validation'],
			weights,
			run_device,
			amp=plan.config.train.amp,
		)
		train_metrics = channel_metrics(train_confusion)
		row: dict[str, object] = {
			'epoch': epoch,
			'global_step': global_step,
			'train_loss': train_loss_sum / train_voxels,
			'train_channel_iou': train_metrics['channel_iou'],
			'validation_loss': validation_metrics['loss'],
			'validation_channel_iou': validation_metrics['channel_iou'],
			'validation_channel_f1': validation_metrics['channel_f1'],
		}
		history.append(row)
		validation_iou = float(validation_metrics['channel_iou'])
		if validation_iou > best_iou:
			best_iou = validation_iou
			best_epoch = epoch
			_save_checkpoint(
				plan.output_dir / BEST_NAME,
				decoder=decoder,
				optimizer=optimizer,
				scaler=scaler,
				payload={
					'run_identity': identity,
					'epoch': epoch,
					'validation': validation_metrics,
				},
			)
		_save_latest(
			plan,
			decoder,
			optimizer,
			scaler,
			identity,
			history,
			best_epoch,
			best_iou,
			global_step,
			epoch + 1,
			0,
			np.zeros((2, 2), dtype=np.int64),
			0.0,
			0,
			completed=False,
		)
		_write_history(plan.output_dir / HISTORY_NAME, history)
		start_position = 0
		train_confusion = np.zeros((2, 2), dtype=np.int64)
		train_loss_sum = 0.0
		train_voxels = 0
		if max_steps is not None and global_step >= max_steps:
			return None
	if best_epoch is None:
		raise RuntimeError('training completed without a best checkpoint')
	best = torch.load(
		plan.output_dir / BEST_NAME, map_location=run_device, weights_only=False
	)
	decoder.load_state_dict(best['model_state_dict'])
	validation_metrics = dict(best['validation'])
	test_metrics = _evaluate(
		decoder,
		datasets['test'],
		weights,
		run_device,
		amp=plan.config.train.amp,
	)
	metrics_payload = {
		'model': plan.model,
		'layout_id': plan.layout_id,
		'data_size': plan.data_size,
		'benchmark_identity': identity,
		'supervision': {
			'axis_mapping': dict(CHANNEL_AXIS_MAPPING),
			'train_inline': list(plan.train_lines.inline),
			'train_crossline': list(plan.train_lines.crossline),
			'validation_inline': list(plan.layouts.validation.inline),
			'validation_crossline': list(plan.layouts.validation.crossline),
			'test_definition': channel_test_definition(
				plan.reserved_training_lines
			),
			'split_class_counts': {
				split: list(plan.split_counts[split])
				for split in ('train', 'validation', 'test')
			},
		},
		'selected_inline_indices': list(plan.train_lines.inline),
		'selected_crossline_indices': list(plan.train_lines.crossline),
		'train_channel_voxels': plan.split_counts['train'][1],
		'train_non_channel_voxels': plan.split_counts['train'][0],
		'validation_channel_voxels': plan.split_counts['validation'][1],
		'test_channel_voxels': plan.split_counts['test'][1],
		'class_weights': list(plan.class_weights),
		'best_epoch': best_epoch,
		'validation': _public_metrics(validation_metrics),
		'test': _public_metrics(test_metrics),
	}
	metrics_path = plan.output_dir / METRICS_NAME
	metrics_path.write_text(
		json.dumps(metrics_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
	_save_latest(
		plan,
		decoder,
		optimizer,
		scaler,
		identity,
		history,
		best_epoch,
		best_iou,
		global_step,
		plan.config.train.epochs,
		0,
		np.zeros((2, 2), dtype=np.int64),
		0.0,
		0,
		completed=True,
	)
	return metrics_path


def _evaluate(
	decoder: torch.nn.Module,
	dataset: Dataset[dict[str, Any]],
	weights: torch.Tensor,
	device: torch.device,
	*,
	amp: bool,
) -> dict[str, object]:
	decoder.eval()
	confusion = np.zeros((2, 2), dtype=np.int64)
	loss_sum = 0.0
	voxel_count = 0
	loader = DataLoader(dataset, batch_size=1, shuffle=False)
	with torch.no_grad():
		for batch in loader:
			embeddings = batch['embeddings'].to(device).detach()
			token_mask = batch['token_valid_mask'].to(device)
			labels = batch['labels'].to(device)
			mask = batch['supervision_mask'].to(device) & batch['core_mask'].to(device)
			with torch.autocast(device_type=device.type, enabled=amp):
				logits = decoder(embeddings, token_mask)
				loss, summary = masked_weighted_voxel_cross_entropy(
					logits, labels, mask, weights
				)
			predicted = logits.argmax(dim=1)
			encoded = labels[mask] * 2 + predicted[mask]
			confusion += (
				torch.bincount(encoded, minlength=4).reshape(2, 2).cpu().numpy()
			)
			count = int(summary['supervised_voxel_count'])
			loss_sum += float(loss.detach().cpu()) * count
			voxel_count += count
	metrics: dict[str, object] = dict(channel_metrics(confusion))
	metrics['loss'] = loss_sum / voxel_count
	metrics['supervised_voxel_count'] = voxel_count
	return metrics


def _public_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
	return {
		key: metrics[key]
		for key in (
			'channel_iou',
			'channel_f1',
			'channel_precision',
			'channel_recall',
			'balanced_accuracy',
		)
	}


def _save_latest(  # noqa: PLR0913, PLR0917
	plan: ChannelDecoderPlan,
	decoder: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	identity: Mapping[str, object],
	history: Sequence[Mapping[str, object]],
	best_epoch: int | None,
	best_iou: float,
	global_step: int,
	epoch: int,
	next_position: int,
	train_confusion: np.ndarray,
	train_loss_sum: float,
	train_voxels: int,
	*,
	completed: bool,
) -> None:
	_save_checkpoint(
		plan.output_dir / LATEST_NAME,
		decoder=decoder,
		optimizer=optimizer,
		scaler=scaler,
		payload={
			'run_identity': identity,
			'history': list(history),
			'best_epoch': best_epoch,
			'best_iou': best_iou,
			'global_step': global_step,
			'epoch': epoch,
			'next_position': next_position,
			'train_confusion': train_confusion.tolist(),
			'train_loss_sum': train_loss_sum,
			'train_voxels': train_voxels,
			'completed': completed,
		},
	)


def _save_checkpoint(
	path: Path,
	*,
	decoder: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	payload: Mapping[str, object],
) -> None:
	full = {
		**payload,
		'model_state_dict': {
			key: value.detach().cpu() for key, value in decoder.state_dict().items()
		},
		'optimizer_state_dict': optimizer.state_dict(),
		'scaler_state_dict': None if scaler is None else scaler.state_dict(),
	}
	temporary = path.with_name(f'.{path.name}.tmp')
	torch.save(full, temporary)
	temporary.replace(path)


def _run_identity(plan: ChannelDecoderPlan) -> dict[str, object]:
	selected_metadata = (
		plan.geometry.pretrained_metadata
		if plan.model == 'pretrained'
		else plan.geometry.random_metadata
	)
	selected_model_source = (
		plan.geometry.pretrained_model_source
		if plan.model == 'pretrained'
		else plan.geometry.random_model_source
	)
	return {
		'model': plan.model,
		'layout_id': plan.layout_id,
		'data_size': plan.data_size,
		'embedding': {
			'checkpoint_path': selected_metadata['checkpoint_path'],
			'checkpoint_sha256': selected_metadata['checkpoint_sha256'],
			'model_source': dict(selected_model_source),
			'common_metadata': {
				key: selected_metadata[key]
				for key in _PAIRED_EMBEDDING_METADATA_KEYS
			},
		},
		'decoder_initial_state_sha256': decoder_initial_state_sha256(
			plan.config.decoder,
			plan.geometry.patch_size_xyz,
			plan.config.train.seed,
		),
		'label_path': str(plan.config.labels),
		'label_metadata_path': str(plan.config.labels_metadata),
		'prepared_label_identity': dict(plan.prepared_label_identity),
		'train_lines': {
			'inline': list(plan.train_lines.inline),
			'crossline': list(plan.train_lines.crossline),
		},
		'validation': {
			'inline': list(plan.layouts.validation.inline),
			'crossline': list(plan.layouts.validation.crossline),
		},
		'test_definition': channel_test_definition(plan.reserved_training_lines),
		'geometry': {
			'embedding_shape': list(plan.geometry.embedding_shape),
			'volume_shape_xyz': list(plan.geometry.volume_shape_xyz),
			'token_grid_shape_xyz': list(plan.geometry.token_grid_shape_xyz),
			'patch_size_xyz': list(plan.geometry.patch_size_xyz),
		},
		'class_weights': list(plan.class_weights),
		'decoder': {
			'spec': plan.config.decoder.spec,
			'embedding_dim': plan.config.decoder.embedding_dim,
			'class_count': plan.config.decoder.class_count,
			'hidden_channels': list(plan.config.decoder.hidden_channels),
			'upsample_factors': [
				list(factors) for factors in plan.config.decoder.upsample_factors
			],
			'upsample_mode': plan.config.decoder.upsample_mode,
			'normalization': plan.config.decoder.normalization,
		},
		'training': {
			'epochs': plan.config.train.epochs,
			'batch_size': plan.config.train.batch_size,
			'learning_rate': plan.config.train.learning_rate,
			'weight_decay': plan.config.train.weight_decay,
			'class_weight': plan.config.train.class_weight,
			'sampling_mode': plan.config.train.sampling_mode,
			'seed': plan.config.train.seed,
			'amp': plan.config.train.amp,
			'gradient_clip_norm': plan.config.train.gradient_clip_norm,
		},
		'tiles': {
			'core_size_tokens': list(plan.config.tiles.core_size_tokens),
			'context_halo_tokens': list(plan.config.tiles.context_halo_tokens),
		},
		'split_class_counts': {
			split: list(plan.split_counts[split])
			for split in ('train', 'validation', 'test')
		},
		'tile_counts': {
			split: plan.tile_counts[split]
			for split in ('train', 'validation', 'test')
		},
	}


def _write_history(path: Path, history: Sequence[Mapping[str, object]]) -> None:
	fields = (
		tuple(history[0])
		if history
		else (
			'epoch',
			'global_step',
			'train_loss',
			'train_channel_iou',
			'validation_loss',
			'validation_channel_iou',
			'validation_channel_f1',
		)
	)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fields)
		writer.writeheader()
		writer.writerows(history)


def _make_decoder(
	architecture: DecoderArchitecture, patch_size_xyz: Sequence[int]
) -> VoxelDecoder3D:
	return VoxelDecoder3D(
		spec=architecture.spec,
		embedding_dim=architecture.embedding_dim,
		class_count=architecture.class_count,
		hidden_channels=architecture.hidden_channels,
		upsample_factors=architecture.upsample_factors,
		upsample_mode=architecture.upsample_mode,
		normalization=architecture.normalization,
		patch_size_xyz=patch_size_xyz,
	)


def _decoder(value: Mapping[str, object]) -> DecoderArchitecture:
	architecture = DecoderArchitecture(
		embedding_dim=_integer(value, 'embedding_dim'),
		class_count=_integer(value, 'class_count'),
		hidden_channels=_integer_tuple(
			value.get('hidden_channels'), 'decoder.hidden_channels'
		),
		upsample_factors=tuple(
			_triplet(item, 'decoder.upsample_factors')
			for item in _sequence(
				value.get('upsample_factors'), 'decoder.upsample_factors'
			)
		),
		upsample_mode=str(value.get('upsample_mode')),
		normalization=str(value.get('normalization')),
		spec=str(value.get('spec', VOXEL_DECODER_SPEC)),
	)
	expected = DecoderArchitecture(
		384,
		2,
		(128, 64, 32),
		((2, 2, 2),) * 3,
		VOXEL_DECODER_UPSAMPLE_MODE,
		VOXEL_DECODER_NORMALIZATION,
	)
	if architecture != expected:
		raise ValueError('decoder settings differ from the fixed Channel benchmark')
	return architecture


def _train(value: Mapping[str, object]) -> DecoderTrain:
	settings = DecoderTrain(
		epochs=_integer(value, 'epochs'),
		batch_size=_integer(value, 'batch_size'),
		learning_rate=_number(value, 'learning_rate'),
		weight_decay=_number(value, 'weight_decay'),
		class_weight=str(value.get('class_weight')),
		sampling_mode=str(value.get('sampling_mode')),
		seed=_integer(value, 'seed', positive=False),
		amp=_boolean(value, 'amp'),
		gradient_clip_norm=_number(value, 'gradient_clip_norm'),
	)
	expected = DecoderTrain(
		epochs=50,
		batch_size=1,
		learning_rate=0.001,
		weight_decay=0.0001,
		class_weight='balanced',
		sampling_mode='all_tiles_once',
		seed=42000,
		amp=True,
		gradient_clip_norm=1.0,
	)
	if settings != expected:
		raise ValueError('train settings differ from the fixed Channel benchmark')
	return settings


def _tiles(value: Mapping[str, object]) -> DecoderTiles:
	settings = DecoderTiles(
		core_size_tokens=_triplet(
			value.get('core_size_tokens'), 'tiles.core_size_tokens'
		),
		context_halo_tokens=_triplet(
			value.get('context_halo_tokens'),
			'tiles.context_halo_tokens',
			allow_zero=True,
		),
	)
	expected = DecoderTiles(
		core_size_tokens=CHANNEL_CORE_SIZE_TOKENS,
		context_halo_tokens=CHANNEL_CONTEXT_HALO_TOKENS,
	)
	if settings != expected:
		raise ValueError('tile settings differ from the fixed Channel benchmark')
	return settings


def _resolve_device(value: str) -> torch.device:
	if value not in {'auto', 'cpu', 'cuda'}:
		raise ValueError('device must be auto, cpu, or cuda')
	if value == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if value == 'cuda' and not torch.cuda.is_available():
		raise RuntimeError('CUDA was requested but is not available')
	return torch.device(value)


def _configure_determinism() -> None:
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
	torch.use_deterministic_algorithms(mode=True)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def _seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)  # noqa: NPY002
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _validate_output(output_dir: Path, resume: Path | None) -> None:
	if resume is None:
		if output_dir.exists() and any(output_dir.iterdir()):
			raise FileExistsError(f'Channel job output is non-empty: {output_dir}')
		return
	if not resume.is_file() or resume.name != LATEST_NAME:
		raise FileNotFoundError(f'resume must identify an existing {LATEST_NAME}')
	if resume.parent.resolve() != output_dir.resolve():
		raise ValueError('resume checkpoint must be in this job output directory')


def _embedding_dim(metadata: Mapping[str, object]) -> int:
	value = metadata.get('embedding_dim')
	geometry = metadata.get('model_geometry')
	if value is None and isinstance(geometry, Mapping):
		value = geometry.get('encoder_dim')
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError('embedding dimension metadata is missing or invalid')
	return value


def _read_json(path: Path) -> Mapping[str, object]:
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, Mapping):
		raise TypeError(f'JSON must contain an object: {path}')
	return value


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _absolute_path(value: Mapping[str, object], key: str, prefix: str) -> Path:
	item = value.get(key)
	if not isinstance(item, str) or not item:
		raise ValueError(f'{prefix}.{key} must be a non-empty path')
	path = Path(item)
	if not path.is_absolute():
		raise ValueError(f'{prefix}.{key} must be absolute')
	return path


def _integer(value: Mapping[str, object], key: str, *, positive: bool = True) -> int:
	item = value.get(key)
	if not isinstance(item, int) or isinstance(item, bool) or item < int(positive):
		raise ValueError(f'{key} must be an integer')
	return item


def _number(value: Mapping[str, object], key: str) -> float:
	item = value.get(key)
	if not isinstance(item, int | float) or isinstance(item, bool):
		raise TypeError(f'{key} must be numeric')
	return float(item)


def _boolean(value: Mapping[str, object], key: str) -> bool:
	item = value.get(key)
	if not isinstance(item, bool):
		raise TypeError(f'{key} must be boolean')
	return item


def _sequence(value: object, label: str) -> Sequence[object]:
	if not isinstance(value, list) or not value:
		raise TypeError(f'{label} must be a non-empty list')
	return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
	items = _sequence(value, label)
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in items
	):
		raise ValueError(f'{label} must contain positive integers')
	return tuple(int(item) for item in items)


def _triplet(
	value: object, label: str, *, allow_zero: bool = False
) -> tuple[int, int, int]:
	if not isinstance(value, list | tuple) or len(value) != 3:
		raise TypeError(f'{label} must be an integer triple')
	minimum = 0 if allow_zero else 1
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item < minimum
		for item in value
	):
		raise ValueError(f'{label} must contain integers >= {minimum}')
	return (int(value[0]), int(value[1]), int(value[2]))


def _slices(start: Sequence[int], stop: Sequence[int]) -> tuple[slice, slice, slice]:
	return tuple(slice(a, b) for a, b in zip(start, stop, strict=True))  # type: ignore[return-value]


def _array(value: np.ndarray | None) -> np.ndarray:
	if value is None:
		raise RuntimeError('dataset memory map is not open')
	return value


def _ratio(numerator: int, denominator: int) -> float:
	return 0.0 if denominator == 0 else numerator / denominator


__all__ = [
	'CHANNEL_CONTEXT_HALO_TOKENS',
	'CHANNEL_CORE_SIZE_TOKENS',
	'CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX',
	'CHANNEL_PRETRAINED_MODEL_TAG',
	'CHANNEL_RANDOM_ENCODER_SEED',
	'ChannelDecoderConfig',
	'ChannelDecoderPlan',
	'ChannelTileDataset',
	'DecoderArchitecture',
	'DecoderTiles',
	'DecoderTrain',
	'EmbeddingGeometry',
	'channel_decoder_config_from_mapping',
	'channel_metrics',
	'decoder_initial_state_sha256',
	'deterministic_tile_order',
	'inspect_channel_decoder_job',
	'inspect_embedding_pair',
	'run_channel_decoder_job',
]
