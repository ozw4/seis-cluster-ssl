"""Raw-amplitude input and trainable encoder connection for Parihaka Channel."""

from __future__ import annotations

import hashlib
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
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.models.voxel_decoder import (
	VoxelDecoder3D,
	validate_context_halo_tokens,
	validate_voxel_decoder_architecture,
)
from seis_ssl_cluster.parihaka.channel_checkpoints import (
	inspect_channel_model_sources,
)
from seis_ssl_cluster.parihaka.channel_data import (
	ChannelLayouts,
	SectionLines,
	inspect_prepared_label_identity,
	load_channel_layouts,
	selected_training_lines,
)
from seis_ssl_cluster.parihaka.channel_decoder import (
	DecoderArchitecture,
	DecoderTiles,
	decoder_initial_state_sha256,
)
from seis_ssl_cluster.parihaka.channel_tiles import (
	CHANNEL_CONTEXT_HALO_TOKENS,
	CHANNEL_CORE_SIZE_TOKENS,
	CHANNEL_PATCH_SIZE_VOXELS,
	ChannelTileSettings,
	build_channel_tile_targets,
	enumerate_channel_tile_records,
)
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)
from seis_ssl_cluster.training.voxel_decoder.losses import (
	balanced_class_weights_from_counts,
)

if TYPE_CHECKING:
	from torch.optim.optimizer import ParamGroup


_ENCODER_PARAMETER_PREFIXES = ('patch_projection.', 'encoder.')


@dataclass(frozen=True)
class MaeModelGeometry:
	"""Architecture fields that define the complete MAE checkpoint identity."""

	in_channels: int
	out_channels: int
	patch_size_xyz: tuple[int, int, int]
	encoder_dim: int
	encoder_depth: int
	encoder_heads: int
	decoder_dim: int
	decoder_depth: int
	decoder_heads: int

	def as_dict(self) -> dict[str, object]:
		"""Return a JSON-compatible geometry mapping."""
		return {
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'patch_size': list(self.patch_size_xyz),
			'encoder_dim': self.encoder_dim,
			'encoder_depth': self.encoder_depth,
			'encoder_heads': self.encoder_heads,
			'decoder_dim': self.decoder_dim,
			'decoder_depth': self.decoder_depth,
			'decoder_heads': self.decoder_heads,
		}


@dataclass(frozen=True)
class ChannelEndToEndTrain:
	"""Fixed optimization settings for the future end-to-end training issue."""

	epochs: int
	batch_size: int
	encoder_learning_rate: float
	decoder_learning_rate: float
	weight_decay: float
	class_weight: str
	sampling_mode: str
	seed: int
	amp: bool
	gradient_clip_norm: float


@dataclass(frozen=True)
class ChannelEndToEndConfig:
	"""Resolved settings shared by pretrained and random one-job plans."""

	survey_id: str
	labels: Path
	labels_metadata: Path
	reference_embedding_dir: Path
	pretrained_checkpoint: Path
	random_checkpoint: Path
	runs_root: Path
	output_dir: Path
	four_way_output_dir: Path
	decoder: DecoderArchitecture
	tiles: DecoderTiles
	train: ChannelEndToEndTrain


@dataclass(frozen=True)
class ChannelEndToEndRuntime:
	"""Resolved precision behavior without device-dependent science changes."""

	device_type: str
	amp_enabled: bool
	autocast_dtype: str | None
	grad_scaler_enabled: bool


@dataclass(frozen=True)
class ChannelEndToEndPlan:
	"""Read-only, fully validated plan for exactly one future training job."""

	config: ChannelEndToEndConfig
	encoder_init: str
	layout_id: str
	data_size: str
	output_dir: Path
	reference: ChannelReferenceArtifact
	layouts: ChannelLayouts
	train_lines: SectionLines
	prepared_label_identity: Mapping[str, object]
	split_counts: Mapping[str, tuple[int, int]]
	class_weights: tuple[float, float]
	tile_counts: Mapping[str, int]
	tile_ids: Mapping[str, tuple[int, ...]]
	pretrained_model_source: Mapping[str, object]
	random_model_source: Mapping[str, object]
	model_geometry: MaeModelGeometry
	pretrained_encoder_initial_state_sha256: str
	random_encoder_initial_state_sha256: str
	decoder_initial_state_sha256: str
	runtime: ChannelEndToEndRuntime
	benchmark_identity: Mapping[str, object]


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
	metadata: Mapping[str, object]
	metadata_sha256: str
	valid_tokens_sha256: str


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
		metadata=dict(payload),
		metadata_sha256=file_sha256(metadata_path),
		valid_tokens_sha256=file_sha256(valid_tokens_path),
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


def channel_end_to_end_config_from_mapping(
	config: Mapping[str, object],
) -> ChannelEndToEndConfig:
	"""Resolve the single fixed end-to-end experiment config."""
	dataset = _mapping(config, 'dataset')
	inputs = _mapping(config, 'inputs')
	outputs = _mapping(config, 'outputs')
	survey_id = dataset.get('survey_id')
	if survey_id != 'parihaka':
		raise ValueError("dataset.survey_id must equal 'parihaka'")
	return ChannelEndToEndConfig(
		survey_id=survey_id,
		labels=_absolute_path(inputs, 'labels_npy', 'inputs'),
		labels_metadata=_absolute_path(
			inputs, 'labels_metadata_json', 'inputs'
		),
		reference_embedding_dir=_absolute_path(
			inputs, 'reference_embedding_dir', 'inputs'
		),
		pretrained_checkpoint=_absolute_path(
			inputs, 'pretrained_checkpoint', 'inputs'
		),
		random_checkpoint=_absolute_path(
			inputs, 'random_checkpoint', 'inputs'
		),
		runs_root=_absolute_path(outputs, 'runs_root', 'outputs'),
		output_dir=_absolute_path(outputs, 'output_dir', 'outputs'),
		four_way_output_dir=_absolute_path(
			outputs, 'four_way_output_dir', 'outputs'
		),
		decoder=_end_to_end_decoder(_mapping(config, 'decoder')),
		tiles=_end_to_end_tiles(_mapping(config, 'tiles')),
		train=_end_to_end_train(_mapping(config, 'train')),
	)


def inspect_channel_end_to_end_job(  # noqa: PLR0913
	config: ChannelEndToEndConfig,
	*,
	encoder_init: str,
	layout_id: str,
	data_size: str,
	layout_config: str | Path,
	device: str = 'auto',
) -> ChannelEndToEndPlan:
	"""Validate all scientific inputs and return one read-only job identity."""
	if encoder_init not in {'pretrained', 'random'}:
		raise ValueError("encoder_init must be 'pretrained' or 'random'")
	reference = resolve_channel_reference_artifact(
		config.reference_embedding_dir,
		survey_id=config.survey_id,
	)
	_validate_reference_files(reference)
	if reference.patch_size_xyz != CHANNEL_PATCH_SIZE_VOXELS:
		raise ValueError('reference patch size differs from the Channel contract')
	_validate_configured_pretrained_source(config, reference.metadata)
	random_metadata = {
		'checkpoint_path': str(config.random_checkpoint),
		'checkpoint_sha256': file_sha256(config.random_checkpoint),
	}
	pretrained_source, random_source = inspect_channel_model_sources(
		reference.metadata,
		random_metadata,
	)
	_validate_pretrained_checkpoint_role(config.pretrained_checkpoint)
	pretrained_geometry = _checkpoint_geometry(config.pretrained_checkpoint)
	random_geometry = _checkpoint_geometry(config.random_checkpoint)
	if pretrained_geometry != random_geometry:
		raise ValueError('pretrained/random checkpoint model geometry mismatch')
	_validate_parihaka_mae_geometry(
		pretrained_geometry,
		reference=reference,
		decoder=config.decoder,
	)
	validate_voxel_decoder_architecture(
		hidden_channels=config.decoder.hidden_channels,
		upsample_factors=config.decoder.upsample_factors,
		patch_size_xyz=reference.patch_size_xyz,
	)
	validate_context_halo_tokens(
		context_halo_tokens=config.tiles.context_halo_tokens,
		core_size_tokens=config.tiles.core_size_tokens,
		token_grid_shape_xyz=reference.token_grid_shape_xyz,
		upsample_factors=config.decoder.upsample_factors,
	)
	labels = np.load(config.labels, mmap_mode='r', allow_pickle=False)
	if labels.shape != reference.volume_shape_xyz:
		raise ValueError('prepared label shape does not match amplitude volume shape')
	if labels.dtype != np.int8:
		raise TypeError('prepared labels must have dtype int8')
	prepared_label_identity = inspect_prepared_label_identity(
		config.labels, config.labels_metadata
	)
	layouts = load_channel_layouts(layout_config, reference.volume_shape_xyz)
	train_lines = selected_training_lines(layouts, layout_id, data_size)
	split_counts: dict[str, tuple[int, int]] = {}
	tile_counts: dict[str, int] = {}
	tile_ids: dict[str, tuple[int, ...]] = {}
	for split in ('train', 'validation', 'test'):
		dataset = ChannelAmplitudeTileDataset(
			reference=reference,
			labels_path=config.labels,
			lines=train_lines,
			validation=layouts.validation,
			test=layouts.test,
			split=split,
			core_size_tokens=config.tiles.core_size_tokens,
			context_halo_tokens=config.tiles.context_halo_tokens,
			survey_id=config.survey_id,
		)
		split_counts[split] = dataset.class_counts
		tile_counts[split] = len(dataset)
		tile_ids[split] = tuple(record.tile_id for record in dataset.records)
	for split in ('train', 'validation', 'test'):
		if any(count == 0 for count in split_counts[split]):
			raise ValueError(
				f'{split} sections must contain both Channel and non-Channel voxels'
			)
		if tile_counts[split] == 0:
			raise ValueError(f'{split} sections must contain supervised tiles')
	class_weights = tuple(
		float(value)
		for value in balanced_class_weights_from_counts(
			split_counts['train']
		).tolist()
	)
	pretrained_encoder_sha = encoder_initial_state_sha256(
		config.pretrained_checkpoint
	)
	random_encoder_sha = encoder_initial_state_sha256(config.random_checkpoint)
	decoder_sha = decoder_initial_state_sha256(
		config.decoder,
		reference.patch_size_xyz,
		config.train.seed,
	)
	runtime = resolve_channel_end_to_end_runtime(device, amp=config.train.amp)
	output_dir = (
		config.runs_root
		/ f'encoder_init={encoder_init}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)
	identity = _end_to_end_identity(
		config=config,
		encoder_init=encoder_init,
		layout_id=layout_id,
		data_size=data_size,
		reference=reference,
		prepared_label_identity=prepared_label_identity,
		layouts=layouts,
		train_lines=train_lines,
		split_counts=split_counts,
		class_weights=class_weights,
		tile_counts=tile_counts,
		pretrained_source=pretrained_source,
		random_source=random_source,
		geometry=pretrained_geometry,
		pretrained_encoder_sha=pretrained_encoder_sha,
		random_encoder_sha=random_encoder_sha,
		decoder_sha=decoder_sha,
		runtime=runtime,
	)
	return ChannelEndToEndPlan(
		config=config,
		encoder_init=encoder_init,
		layout_id=layout_id,
		data_size=data_size,
		output_dir=output_dir,
		reference=reference,
		layouts=layouts,
		train_lines=train_lines,
		prepared_label_identity=prepared_label_identity,
		split_counts=split_counts,
		class_weights=class_weights,
		tile_counts=tile_counts,
		tile_ids=tile_ids,
		pretrained_model_source=pretrained_source,
		random_model_source=random_source,
		model_geometry=pretrained_geometry,
		pretrained_encoder_initial_state_sha256=pretrained_encoder_sha,
		random_encoder_initial_state_sha256=random_encoder_sha,
		decoder_initial_state_sha256=decoder_sha,
		runtime=runtime,
		benchmark_identity=identity,
	)


def resolve_channel_end_to_end_runtime(
	device: str, *, amp: bool
) -> ChannelEndToEndRuntime:
	"""Resolve device and mixed precision without changing scientific settings."""
	if device not in {'auto', 'cpu', 'cuda'}:
		raise ValueError('device must be auto, cpu, or cuda')
	if device == 'auto':
		device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
	elif device == 'cuda' and not torch.cuda.is_available():
		raise RuntimeError('CUDA was requested but is not available')
	else:
		device_type = device
	amp_enabled = amp and device_type == 'cuda'
	return ChannelEndToEndRuntime(
		device_type=device_type,
		amp_enabled=amp_enabled,
		autocast_dtype='float16' if amp_enabled else None,
		grad_scaler_enabled=amp_enabled,
	)


def build_channel_end_to_end_model(plan: ChannelEndToEndPlan) -> ChannelEndToEndModel:
	"""Build the selected float32 encoder and deterministic voxel decoder."""
	geometry = plan.model_geometry
	mae = AmplitudeMAE3D(
		in_channels=geometry.in_channels,
		out_channels=geometry.out_channels,
		patch_size_xyz=geometry.patch_size_xyz,
		encoder_dim=geometry.encoder_dim,
		encoder_depth=geometry.encoder_depth,
		encoder_heads=geometry.encoder_heads,
		decoder_dim=geometry.decoder_dim,
		decoder_depth=geometry.decoder_depth,
		decoder_heads=geometry.decoder_heads,
	)
	checkpoint_path = (
		plan.config.pretrained_checkpoint
		if plan.encoder_init == 'pretrained'
		else plan.config.random_checkpoint
	)
	payload = _load_checkpoint(checkpoint_path)
	state = _model_state(payload, checkpoint_path)
	mae.load_state_dict(state, strict=True)
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(plan.config.train.seed)
		decoder = VoxelDecoder3D(
			spec=plan.config.decoder.spec,
			embedding_dim=plan.config.decoder.embedding_dim,
			class_count=plan.config.decoder.class_count,
			hidden_channels=plan.config.decoder.hidden_channels,
			upsample_factors=plan.config.decoder.upsample_factors,
			upsample_mode=plan.config.decoder.upsample_mode,
			normalization=plan.config.decoder.normalization,
			patch_size_xyz=geometry.patch_size_xyz,
		)
	return ChannelEndToEndModel(mae, decoder).float()


def channel_end_to_end_optimizer_groups(
	model: ChannelEndToEndModel,
	*,
	encoder_learning_rate: float,
	decoder_learning_rate: float,
	weight_decay: float,
) -> list[ParamGroup]:
	"""Return exactly the two disjoint future AdamW parameter groups."""
	encoder = list(model.encoder_parameters())
	decoder = list(model.decoder_parameters())
	if not encoder or not decoder:
		raise ValueError('encoder and decoder parameter groups must be non-empty')
	if {id(parameter) for parameter in encoder} & {
		id(parameter) for parameter in decoder
	}:
		raise ValueError('encoder and decoder parameter groups must be disjoint')
	return [
		{
			'name': 'encoder',
			'params': encoder,
			'lr': encoder_learning_rate,
			'weight_decay': weight_decay,
		},
		{
			'name': 'decoder',
			'params': decoder,
			'lr': decoder_learning_rate,
			'weight_decay': weight_decay,
		},
	]


def encoder_initial_state_sha256(path: str | Path) -> str:
	"""Hash exactly the float32 trainable MAE encoder initial state."""
	checkpoint_path = Path(path)
	state = _model_state(_load_checkpoint(checkpoint_path), checkpoint_path)
	selected = {
		key: value
		for key, value in state.items()
		if key.startswith(_ENCODER_PARAMETER_PREFIXES)
	}
	if not selected or not any(key.startswith('patch_projection.') for key in selected):
		raise ValueError('checkpoint is missing trainable MAE encoder parameters')
	if not any(key.startswith('encoder.') for key in selected):
		raise ValueError('checkpoint is missing MAE encoder parameters')
	digest = hashlib.sha256()
	for key in sorted(selected):
		value = selected[key]
		if not isinstance(value, torch.Tensor):
			raise TypeError(f'checkpoint model_state_dict.{key} must be a tensor')
		array = value.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
		digest.update(key.encode())
		digest.update(str(tuple(array.shape)).encode())
		digest.update(array.tobytes())
	return digest.hexdigest()


def _end_to_end_identity(  # noqa: PLR0913
	*,
	config: ChannelEndToEndConfig,
	encoder_init: str,
	layout_id: str,
	data_size: str,
	reference: ChannelReferenceArtifact,
	prepared_label_identity: Mapping[str, object],
	layouts: ChannelLayouts,
	train_lines: SectionLines,
	split_counts: Mapping[str, tuple[int, int]],
	class_weights: tuple[float, float],
	tile_counts: Mapping[str, int],
	pretrained_source: Mapping[str, object],
	random_source: Mapping[str, object],
	geometry: MaeModelGeometry,
	pretrained_encoder_sha: str,
	random_encoder_sha: str,
	decoder_sha: str,
	runtime: ChannelEndToEndRuntime,
) -> dict[str, object]:
	selected_source = (
		pretrained_source if encoder_init == 'pretrained' else random_source
	)
	selected_encoder_sha = (
		pretrained_encoder_sha
		if encoder_init == 'pretrained'
		else random_encoder_sha
	)
	return {
		'encoder_init': encoder_init,
		'layout_id': layout_id,
		'data_size': data_size,
		'encoder_source': {
			**dict(selected_source),
			'model_geometry': geometry.as_dict(),
			'parameter_dtype': 'float32',
			'trainable_modules': ['patch_projection', 'encoder'],
			'initial_state_sha256': selected_encoder_sha,
		},
		'encoder_initial_states': {
			'pretrained_sha256': pretrained_encoder_sha,
			'random_sha256': random_encoder_sha,
		},
		'reference_input': {
			'amplitude_path': str(reference.source_amplitude_path),
			'normalization_stats_path': str(reference.normalization_stats_path),
			'reference_metadata_path': str(reference.metadata_path),
			'reference_metadata_sha256': reference.metadata_sha256,
			'reference_valid_tokens_path': str(reference.valid_tokens_path),
			'reference_valid_tokens_sha256': reference.valid_tokens_sha256,
			'preprocessing': dict(reference.preprocessing),
			'zero_mask': dict(reference.zero_mask),
			'min_token_valid_fraction': reference.min_token_valid_fraction,
			'patch_size': list(reference.patch_size_xyz),
			'volume_shape': list(reference.volume_shape_xyz),
			'token_grid_shape': list(reference.token_grid_shape_xyz),
		},
		'labels': {
			'path': str(config.labels),
			'metadata_path': str(config.labels_metadata),
			'prepared_label_identity': dict(prepared_label_identity),
		},
		'supervision': {
			'train_lines': _lines_identity(train_lines),
			'validation_lines': _lines_identity(layouts.validation),
			'test_lines': _lines_identity(layouts.test),
			'split_class_counts': {
				split: list(split_counts[split])
				for split in ('train', 'validation', 'test')
			},
			'tile_counts': {
				split: tile_counts[split]
				for split in ('train', 'validation', 'test')
			},
			'class_weights': list(class_weights),
		},
		'decoder': {
			'architecture': _decoder_identity(config.decoder),
			'initial_state_sha256': decoder_sha,
		},
		'optimizer': {
			'encoder_learning_rate': config.train.encoder_learning_rate,
			'decoder_learning_rate': config.train.decoder_learning_rate,
			'weight_decay': config.train.weight_decay,
			'parameter_group_names': ['encoder', 'decoder'],
		},
		'training': {
			'epochs': config.train.epochs,
			'batch_size': config.train.batch_size,
			'sampling_mode': config.train.sampling_mode,
			'seed': config.train.seed,
			'gradient_clip_norm': config.train.gradient_clip_norm,
		},
		'tiles': {
			'core_size_tokens': list(config.tiles.core_size_tokens),
			'context_halo_tokens': list(config.tiles.context_halo_tokens),
		},
		'runtime': {
			'resolved_device_type': runtime.device_type,
			'amp_enabled': runtime.amp_enabled,
			'autocast_dtype': runtime.autocast_dtype,
			'grad_scaler_enabled': runtime.grad_scaler_enabled,
		},
	}


def _validate_reference_files(reference: ChannelReferenceArtifact) -> None:
	if not reference.source_amplitude_path.is_file():
		raise FileNotFoundError(
			f'missing source amplitude: {reference.source_amplitude_path}'
		)
	amplitude = np.load(
		reference.source_amplitude_path, mmap_mode='r', allow_pickle=False
	)
	if amplitude.shape != reference.volume_shape_xyz:
		raise ValueError('source amplitude shape does not match reference metadata')
	if not np.issubdtype(amplitude.dtype, np.floating):
		raise TypeError('source amplitude must have floating dtype')
	load_normalization_stats(reference.normalization_stats_path)


def _validate_configured_pretrained_source(
	config: ChannelEndToEndConfig, metadata: Mapping[str, object]
) -> None:
	path_value = metadata.get('checkpoint_path')
	if not isinstance(path_value, str) or not path_value:
		raise TypeError('reference metadata checkpoint_path must be non-empty')
	if Path(path_value).resolve(strict=False) != config.pretrained_checkpoint.resolve(
		strict=False
	):
		raise ValueError(
			'reference metadata checkpoint_path does not match pretrained_checkpoint'
		)
	if not config.random_checkpoint.is_file():
		raise FileNotFoundError(
			f'missing random encoder checkpoint: {config.random_checkpoint}'
		)


def _checkpoint_geometry(path: Path) -> MaeModelGeometry:
	payload = load_checkpoint_metadata_without_weights(path)
	config = _required_mapping(payload, 'config', f'{path} checkpoint')
	model = _required_mapping(config, 'model', f'{path} checkpoint config')
	return MaeModelGeometry(
		in_channels=_positive_integer(model.get('in_channels'), 'model.in_channels'),
		out_channels=_positive_integer(model.get('out_channels'), 'model.out_channels'),
		patch_size_xyz=_positive_triplet(model.get('patch_size'), 'model.patch_size'),
		encoder_dim=_positive_integer(model.get('encoder_dim'), 'model.encoder_dim'),
		encoder_depth=_positive_integer(
			model.get('encoder_depth'), 'model.encoder_depth'
		),
		encoder_heads=_positive_integer(
			model.get('encoder_heads'), 'model.encoder_heads'
		),
		decoder_dim=_positive_integer(model.get('decoder_dim'), 'model.decoder_dim'),
		decoder_depth=_positive_integer(
			model.get('decoder_depth'), 'model.decoder_depth'
		),
		decoder_heads=_positive_integer(
			model.get('decoder_heads'), 'model.decoder_heads'
		),
	)


def _validate_pretrained_checkpoint_role(path: Path) -> None:
	payload = load_checkpoint_metadata_without_weights(path)
	metadata = payload.get('metadata')
	if isinstance(metadata, Mapping) and (
		metadata.get('random_encoder_baseline') is True
		or metadata.get('pretrained_weights_loaded') is False
	):
		raise ValueError('pretrained checkpoint has random encoder role metadata')
	training_state = payload.get('training_state')
	if isinstance(training_state, Mapping) and (
		training_state.get('checkpoint_kind') == 'random_init'
	):
		raise ValueError('pretrained checkpoint has random_init checkpoint role')


def _validate_parihaka_mae_geometry(
	geometry: MaeModelGeometry,
	*,
	reference: ChannelReferenceArtifact,
	decoder: DecoderArchitecture,
) -> None:
	expected_geometry = MaeModelGeometry(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=CHANNEL_PATCH_SIZE_VOXELS,
		encoder_dim=384,
		encoder_depth=8,
		encoder_heads=6,
		decoder_dim=256,
		decoder_depth=4,
		decoder_heads=4,
	)
	if geometry != expected_geometry:
		raise ValueError('checkpoint model config is not the expected Parihaka MAE')
	if geometry.patch_size_xyz != reference.patch_size_xyz:
		raise ValueError('checkpoint patch size does not match reference metadata')
	if geometry.encoder_dim != decoder.embedding_dim:
		raise ValueError('checkpoint encoder_dim does not match decoder embedding_dim')
	metadata_geometry = reference.metadata.get('model_geometry')
	if not isinstance(metadata_geometry, Mapping):
		raise TypeError('reference metadata model_geometry must be a mapping')
	expected_encoder = {
		'encoder_dim': metadata_geometry.get(
			'encoder_dim', metadata_geometry.get('embed_dim')
		),
		'encoder_depth': metadata_geometry.get(
			'encoder_depth', metadata_geometry.get('depth')
		),
		'encoder_heads': metadata_geometry.get(
			'encoder_heads', metadata_geometry.get('num_heads')
		),
	}
	actual_encoder = {
		'encoder_dim': geometry.encoder_dim,
		'encoder_depth': geometry.encoder_depth,
		'encoder_heads': geometry.encoder_heads,
	}
	if expected_encoder != actual_encoder:
		raise ValueError(
			'checkpoint encoder geometry does not match reference metadata'
		)


def _end_to_end_decoder(value: Mapping[str, object]) -> DecoderArchitecture:
	architecture = DecoderArchitecture(
		embedding_dim=_mapping_integer(value, 'embedding_dim'),
		class_count=_mapping_integer(value, 'class_count'),
		hidden_channels=_positive_integer_tuple(
			value.get('hidden_channels'), 'decoder.hidden_channels'
		),
		upsample_factors=tuple(
			_positive_triplet(item, 'decoder.upsample_factors')
			for item in _nonempty_sequence(
				value.get('upsample_factors'), 'decoder.upsample_factors'
			)
		),
		upsample_mode=str(value.get('upsample_mode')),
		normalization=str(value.get('normalization')),
	)
	expected = DecoderArchitecture(
		embedding_dim=384,
		class_count=2,
		hidden_channels=(128, 64, 32),
		upsample_factors=((2, 2, 2),) * 3,
		upsample_mode='nearest',
		normalization='voxelwise_layer_norm',
	)
	if architecture != expected:
		raise ValueError('decoder settings differ from the fixed Channel benchmark')
	return architecture


def _end_to_end_tiles(value: Mapping[str, object]) -> DecoderTiles:
	tiles = DecoderTiles(
		core_size_tokens=_positive_triplet(
			value.get('core_size_tokens'), 'tiles.core_size_tokens'
		),
		context_halo_tokens=_nonnegative_triplet(
			value.get('context_halo_tokens'), 'tiles.context_halo_tokens'
		),
	)
	expected = DecoderTiles(
		core_size_tokens=CHANNEL_CORE_SIZE_TOKENS,
		context_halo_tokens=CHANNEL_CONTEXT_HALO_TOKENS,
	)
	if tiles != expected:
		raise ValueError('tile settings differ from the fixed Channel benchmark')
	return tiles


def _end_to_end_train(value: Mapping[str, object]) -> ChannelEndToEndTrain:
	settings = ChannelEndToEndTrain(
		epochs=_mapping_integer(value, 'epochs'),
		batch_size=_mapping_integer(value, 'batch_size'),
		encoder_learning_rate=_mapping_number(value, 'encoder_learning_rate'),
		decoder_learning_rate=_mapping_number(value, 'decoder_learning_rate'),
		weight_decay=_mapping_number(value, 'weight_decay'),
		class_weight=str(value.get('class_weight')),
		sampling_mode=str(value.get('sampling_mode')),
		seed=_mapping_integer(value, 'seed', positive=False),
		amp=_mapping_boolean(value, 'amp'),
		gradient_clip_norm=_mapping_number(value, 'gradient_clip_norm'),
	)
	expected = ChannelEndToEndTrain(
		epochs=50,
		batch_size=1,
		encoder_learning_rate=0.0001,
		decoder_learning_rate=0.001,
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


def _decoder_identity(architecture: DecoderArchitecture) -> dict[str, object]:
	return {
		'spec': architecture.spec,
		'embedding_dim': architecture.embedding_dim,
		'class_count': architecture.class_count,
		'hidden_channels': list(architecture.hidden_channels),
		'upsample_factors': [list(value) for value in architecture.upsample_factors],
		'upsample_mode': architecture.upsample_mode,
		'normalization': architecture.normalization,
	}


def _lines_identity(lines: SectionLines) -> dict[str, list[int]]:
	return {
		'inline': list(lines.inline),
		'crossline': list(lines.crossline),
	}


def _load_checkpoint(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'missing MAE checkpoint: {path}')
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError(f'checkpoint must contain a mapping: {path}')
	return payload


def _model_state(
	payload: Mapping[str, object], path: Path
) -> Mapping[str, torch.Tensor]:
	state = payload.get('model_state_dict')
	if not isinstance(state, Mapping):
		raise TypeError(f'{path} checkpoint model_state_dict must be a mapping')
	return cast('Mapping[str, torch.Tensor]', state)


def _required_mapping(
	value: Mapping[str, object], key: str, prefix: str
) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{prefix}.{key} must be a mapping')
	return child


def _absolute_path(value: Mapping[str, object], key: str, prefix: str) -> Path:
	item = value.get(key)
	if not isinstance(item, str) or not item:
		raise ValueError(f'{prefix}.{key} must be a non-empty path')
	path = Path(item)
	if not path.is_absolute():
		raise ValueError(f'{prefix}.{key} must be absolute')
	return path


def _positive_integer(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError(f'{label} must be a positive integer')
	return value


def _mapping_integer(
	value: Mapping[str, object], key: str, *, positive: bool = True
) -> int:
	item = value.get(key)
	if (
		not isinstance(item, int)
		or isinstance(item, bool)
		or item < int(positive)
	):
		raise ValueError(f'{key} must be an integer')
	return item


def _mapping_number(value: Mapping[str, object], key: str) -> float:
	item = value.get(key)
	if not isinstance(item, int | float) or isinstance(item, bool):
		raise TypeError(f'{key} must be numeric')
	return float(item)


def _mapping_boolean(value: Mapping[str, object], key: str) -> bool:
	item = value.get(key)
	if not isinstance(item, bool):
		raise TypeError(f'{key} must be boolean')
	return item


def _nonempty_sequence(value: object, label: str) -> Sequence[object]:
	if not isinstance(value, list | tuple) or not value:
		raise TypeError(f'{label} must be a non-empty sequence')
	return value


def _positive_integer_tuple(value: object, label: str) -> tuple[int, ...]:
	items = _nonempty_sequence(value, label)
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in items
	):
		raise ValueError(f'{label} must contain positive integers')
	return tuple(int(item) for item in items)


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	return _integer_triplet(value, label, minimum=1)


def _nonnegative_triplet(value: object, label: str) -> tuple[int, int, int]:
	return _integer_triplet(value, label, minimum=0)


def _integer_triplet(
	value: object, label: str, *, minimum: int
) -> tuple[int, int, int]:
	if not isinstance(value, list | tuple) or len(value) != 3:
		raise TypeError(f'{label} must be an integer triple')
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item < minimum
		for item in value
	):
		raise ValueError(f'{label} must contain integers >= {minimum}')
	return (int(value[0]), int(value[1]), int(value[2]))


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
	'ChannelEndToEndConfig',
	'ChannelEndToEndModel',
	'ChannelEndToEndPlan',
	'ChannelEndToEndRuntime',
	'ChannelEndToEndTrain',
	'ChannelReferenceArtifact',
	'MaeModelGeometry',
	'build_channel_end_to_end_model',
	'channel_end_to_end_config_from_mapping',
	'channel_end_to_end_optimizer_groups',
	'encoder_initial_state_sha256',
	'inspect_channel_end_to_end_job',
	'resolve_channel_end_to_end_runtime',
	'resolve_channel_reference_artifact',
]
