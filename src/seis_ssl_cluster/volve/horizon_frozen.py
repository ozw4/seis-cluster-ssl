'''Frozen-embedding Volve MAE versus random horizon benchmark.'''

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from seis_ssl_cluster.embedding.writer import (
	EmbeddingOutputPaths,
	file_sha256,
	output_paths,
)
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)
from seis_ssl_cluster.volve.horizon_data import (
	HORIZON_NAMES,
	VolveHorizonData,
	array_sha256,
	load_volve_horizon_data,
)
from seis_ssl_cluster.volve.horizon_layouts import (
	DATA_SIZE_PREFIX,
	LAYOUT_IDS,
	HorizonSplitPlan,
	build_horizon_split_plan,
	load_volve_horizon_layouts,
)
from seis_ssl_cluster.volve.horizon_loss import (
	fractional_horizon_cross_entropy,
	validate_training_horizon_coverage,
)
from seis_ssl_cluster.volve.horizon_metrics import (
	compute_horizon_metrics,
	soft_argmax_global_sample,
)
from seis_ssl_cluster.volve.horizon_model import (
	HORIZON_DECODER_SEED,
	HORIZON_EMBEDDING_DIM,
	HORIZON_HIDDEN_CHANNELS,
	HORIZON_PATCH_SIZE,
	HORIZON_UPSAMPLE_FACTORS,
	VolveHorizonDecoder,
	create_volve_horizon_decoder,
)
from seis_ssl_cluster.volve.horizon_tiles import (
	HORIZON_WINDOW_START,
	HORIZON_WINDOW_STOP,
	HorizonTileRecord,
	HorizonTileSettings,
	build_frozen_horizon_tile,
	build_horizon_tile_targets,
	enumerate_horizon_tile_records,
	frozen_core_output_valid_mask,
	frozen_survey_output_valid_mask,
	horizon_supervision_mask,
)
from seis_ssl_cluster.volve.horizon_training import (
	backward_and_step_horizon_optimizer,
)

FROZEN_MODEL_ROLES = ('pretrained', 'random')
FROZEN_CONDITION_COUNT = 30
LATEST_NAME = 'latest.pt'
BEST_NAME = 'best.pt'
METRICS_NAME = 'metrics.json'
HISTORY_NAME = 'history.json'
OPTIMIZER_NAME = 'adamw'
OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_EPS = 1.0e-8
OBJECTIVE_IDENTITY = {
	'loss': 'fractional_two_bin_per_tile_horizon_macro_v1',
	'prediction': 'masked_soft_argmax_v1',
	'checkpoint_selection': 'strict_lower_validation_macro_mae_v1',
	'metrics_schema_version': 1,
}
VOLVE_MAE_MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
VOLVE_RANDOM_ENCODER_SEED = 42
VOLVE_PRETRAINED_CHECKPOINT_SUFFIX = (
	'pretraining',
	'volve',
	'horizon_benchmark_v1',
	VOLVE_MAE_MODEL_TAG,
	'full_100ep',
	'latest.pt',
)
_PAIRED_EMBEDDING_METADATA_KEYS = (
	'survey_id',
	'source_amplitude_path',
	'source_valid_mask_path',
	'volume_shape_xyz',
	'model_geometry',
	'patch_size',
	'token_grid_shape',
	'window_size',
	'overlap',
	'output_dtype',
	'precision',
	'min_token_valid_fraction',
	'normalization_stats_path',
	'normalized_clip_abs',
	'amplitude_agc',
	'finite_check_mode',
	'preprocessing',
	'zero_mask',
	'pretraining_objective',
)
_EXPECTED_MAE_GEOMETRY = {
	'in_channels': 1,
	'out_channels': 1,
	'patch_size': [8, 8, 8],
	'encoder_dim': 384,
	'encoder_depth': 8,
	'encoder_heads': 6,
	'decoder_dim': 256,
	'decoder_depth': 4,
	'decoder_heads': 4,
}


@dataclass(frozen=True)
class FrozenHorizonTrainSettings:
	'''Fixed one-seed decoder training settings.'''

	epochs: int
	batch_size: int
	learning_rate: float
	weight_decay: float
	sampling_mode: str
	seed: int
	amp: bool
	gradient_clip_norm: float


@dataclass(frozen=True)
class FrozenHorizonConfig:
	'''Resolved common inputs and settings for all 30 jobs.'''

	artifact_root: Path
	volve_root: Path
	survey_id: str
	canonical_input_metadata: Path
	pretrained_embeddings_dir: Path
	random_embeddings_dir: Path
	runs_root: Path
	train: FrozenHorizonTrainSettings
	tiles: HorizonTileSettings


@dataclass(frozen=True)
class FrozenEmbeddingGeometry:
	'''Validated paired embedding arrays, masks, and model roles.'''

	pretrained: EmbeddingOutputPaths
	random: EmbeddingOutputPaths
	volume_shape_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	embedding_shape: tuple[int, int, int, int]
	embedding_dim: int
	pretrained_metadata: Mapping[str, object]
	random_metadata: Mapping[str, object]
	pretrained_model_source: Mapping[str, object]
	random_model_source: Mapping[str, object]
	valid_tokens_sha256: str
	model_valid_lateral_mask: np.ndarray
	model_valid_lateral_mask_sha256: str
	canonical_identity: Mapping[str, object]


@dataclass(frozen=True)
class FrozenHorizonPlan:
	'''Fully validated one-job frozen horizon plan.'''

	config: FrozenHorizonConfig
	model: str
	layout_id: str
	data_size: str
	output_dir: Path
	data: VolveHorizonData
	split_plan: HorizonSplitPlan
	geometry: FrozenEmbeddingGeometry
	tile_records: Mapping[str, tuple[HorizonTileRecord, ...]]
	native_per_horizon_counts: Mapping[str, tuple[int, ...]]
	effective_per_horizon_counts: Mapping[str, tuple[int, ...]]
	excluded_by_token_validity_counts: Mapping[str, tuple[int, ...]]
	run_identity: Mapping[str, object]

	@property
	def per_horizon_counts(self) -> Mapping[str, tuple[int, ...]]:
		'''Return counts actually used by the frozen model.'''
		return self.effective_per_horizon_counts


class FrozenHorizonTileDataset(Dataset[dict[str, Any]]):
	'''Memory-mapped frozen embeddings with 005 fractional targets.'''

	def __init__(  # noqa: PLR0913
		self,
		*,
		data: VolveHorizonData,
		plan: HorizonSplitPlan,
		embedding_path: Path,
		valid_tokens_path: Path,
		settings: HorizonTileSettings,
		split: str,
		records: Sequence[HorizonTileRecord] | None = None,
	) -> None:
		'''Open the selected arrays and enumerate non-empty lateral cores.'''
		super().__init__()
		if split not in {'train', 'validation', 'test'}:
			raise ValueError('split must be train, validation, or test')
		self.data = data
		self.plan = plan
		self.embedding_path = embedding_path
		self.valid_tokens_path = valid_tokens_path
		self.settings = settings
		self.split = split
		self._embeddings: np.ndarray | None = None
		self._valid_tokens: np.ndarray | None = None
		self._open()
		self._window_embeddings, self._window_valid_tokens = self._window_arrays()
		self.native_split_mask = _split_mask(plan, split)
		self.output_valid_survey = frozen_survey_output_valid_mask(
			self._window_valid_tokens, settings
		)
		self.split_mask = self.native_split_mask & self.output_valid_survey[
			np.newaxis, :, :
		]
		self.primary_split_mask: np.ndarray | None = None
		self.primary_supervision_mask: np.ndarray | None = None
		if split == 'test':
			self.primary_split_mask = np.broadcast_to(
				self.plan.test_primary_mask,
				self.data.bound_valid_mask.shape,
			) & self.output_valid_survey[np.newaxis, :, :]
			self.primary_supervision_mask = horizon_supervision_mask(
				sample_float=self.data.sample_float,
				native_valid_mask=self.data.bound_valid_mask,
				split_mask=self.primary_split_mask,
				trace_valid_mask=self.data.valid_trace_mask,
				window_start=self.settings.window_start,
				window_stop=self.settings.window_stop,
			)
		self.records = tuple(records) if records is not None else (
			enumerate_horizon_tile_records(
				sample_float=data.sample_float,
				native_valid_mask=data.bound_valid_mask,
				split_mask=self.split_mask,
				trace_valid_mask=data.valid_trace_mask,
				settings=settings,
			)
		)

	def __len__(self) -> int:
		'''Return the non-empty tile count.'''
		return len(self.records)

	def __getitem__(self, index: int) -> dict[str, Any]:
		'''Load one fixed halo tile and its central-core supervision.'''
		self._open()
		window_embeddings, window_valid = self._window_arrays()
		record = self.records[index]
		targets = build_horizon_tile_targets(
			record=record,
			sample_float=self.data.sample_float,
			native_valid_mask=self.data.bound_valid_mask,
			split_mask=self.split_mask,
			trace_valid_mask=self.data.valid_trace_mask,
			settings=self.settings,
		)
		frozen = build_frozen_horizon_tile(
			record=record,
			embeddings=window_embeddings,
			valid_tokens=window_valid,
			settings=self.settings,
		)
		output_valid = frozen_core_output_valid_mask(
			frozen.token_valid_mask, self.settings
		)
		effective_mask = targets.supervision_mask & output_valid[np.newaxis, :, :]
		primary_mask = effective_mask
		if self.split == 'test':
			primary_split = _array(self.primary_split_mask)
			primary_supervision = _array(self.primary_supervision_mask)
			primary_targets = build_horizon_tile_targets(
				record=_record_for_supervision_mask(record, primary_supervision),
				sample_float=self.data.sample_float,
				native_valid_mask=self.data.bound_valid_mask,
				split_mask=primary_split,
				trace_valid_mask=self.data.valid_trace_mask,
				settings=self.settings,
			)
			primary_mask = primary_targets.supervision_mask & output_valid[
				np.newaxis, :, :
			]
		return {
			'embeddings': torch.from_numpy(frozen.embeddings),
			'token_valid_mask': torch.from_numpy(frozen.token_valid_mask),
			'target_sample_float': torch.from_numpy(targets.sample_float),
			'output_valid_mask': torch.from_numpy(output_valid),
			'supervision_mask': torch.from_numpy(effective_mask),
			'primary_evaluation_mask': torch.from_numpy(primary_mask),
			'tile_id': record.tile_id,
		}

	def __getstate__(self) -> dict[str, object]:
		'''Drop memory maps before worker serialization.'''
		state = self.__dict__.copy()
		for key in (
			'_embeddings',
			'_valid_tokens',
			'_window_embeddings',
			'_window_valid_tokens',
		):
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

	def _window_arrays(self) -> tuple[np.ndarray, np.ndarray]:
		embeddings = _array(self._embeddings)
		valid_tokens = _array(self._valid_tokens)
		start = HORIZON_WINDOW_START // HORIZON_PATCH_SIZE[2]
		stop = HORIZON_WINDOW_STOP // HORIZON_PATCH_SIZE[2]
		return embeddings[:, :, start:stop, :], valid_tokens[:, :, start:stop]


def frozen_horizon_config_from_mapping(
	config: Mapping[str, object],
) -> FrozenHorizonConfig:
	'''Resolve config paths and enforce the fixed 006 settings.'''
	if set(config) != {
		'paths',
		'dataset',
		'inputs',
		'embeddings',
		'outputs',
		'decoder',
		'tiles',
		'train',
	}:
		raise ValueError(
			'config must contain exactly paths, dataset, inputs, embeddings, '
			'outputs, decoder, tiles, and train'
		)
	paths = _mapping(config, 'paths')
	dataset = _mapping(config, 'dataset')
	inputs = _mapping(config, 'inputs')
	embeddings = _mapping(config, 'embeddings')
	outputs = _mapping(config, 'outputs')
	_validate_decoder(_mapping(config, 'decoder'))
	train = _train_settings(_mapping(config, 'train'))
	artifact_root = _absolute_path(paths, 'artifact_root', 'paths')
	volve_root = _absolute_path(paths, 'volve_root', 'paths')
	runs_root = _absolute_path(outputs, 'runs_root', 'outputs')
	if not _is_relative_to(runs_root, artifact_root):
		raise ValueError('outputs.runs_root must be below paths.artifact_root')
	if _is_relative_to(runs_root, volve_root):
		raise ValueError('benchmark output must not be below public volve_root')
	survey_id = dataset.get('survey_id')
	if not isinstance(survey_id, str) or not survey_id:
		raise ValueError('dataset.survey_id must be a non-empty string')
	tiles = _tile_settings(_mapping(config, 'tiles'))
	return FrozenHorizonConfig(
		artifact_root=artifact_root,
		volve_root=volve_root,
		survey_id=survey_id,
		canonical_input_metadata=_absolute_path(
			inputs, 'canonical_input_metadata', 'inputs'
		),
		pretrained_embeddings_dir=_absolute_path(
			embeddings, 'pretrained_dir', 'embeddings'
		),
		random_embeddings_dir=_absolute_path(
			embeddings, 'random_dir', 'embeddings'
		),
		runs_root=runs_root,
		train=train,
		tiles=tiles,
	)


def enumerate_frozen_horizon_conditions() -> tuple[tuple[str, str, str], ...]:
	'''Return the fixed paired 2 by 5 by 3 condition order.'''
	conditions = tuple(
		(model, layout, data_size)
		for model in FROZEN_MODEL_ROLES
		for layout in LAYOUT_IDS
		for data_size in DATA_SIZE_PREFIX
	)
	if len(conditions) != FROZEN_CONDITION_COUNT:
		raise RuntimeError('frozen horizon suite must contain exactly 30 jobs')
	return conditions


def inspect_frozen_embedding_pair(
	config: FrozenHorizonConfig,
	data: VolveHorizonData,
) -> FrozenEmbeddingGeometry:
	'''Validate paired arrays, valid tokens, checkpoint roles, and input identity.'''
	pretrained = output_paths(config.pretrained_embeddings_dir, config.survey_id)
	random_paths = output_paths(config.random_embeddings_dir, config.survey_id)
	for paths in (pretrained, random_paths):
		for path in (paths.embeddings, paths.valid_tokens, paths.metadata):
			if not path.is_file():
				raise FileNotFoundError(f'missing frozen horizon input: {path}')
	pretrained_meta = _read_json(pretrained.metadata, 'pretrained embedding metadata')
	random_meta = _read_json(random_paths.metadata, 'random embedding metadata')
	pretrained_array = np.load(pretrained.embeddings, mmap_mode='r', allow_pickle=False)
	random_array = np.load(random_paths.embeddings, mmap_mode='r', allow_pickle=False)
	_validate_embedding_arrays(pretrained_array, random_array)
	_validate_embedding_metadata_pair(
		pretrained_meta,
		random_meta,
		pretrained_dtype=pretrained_array.dtype,
		random_dtype=random_array.dtype,
	)
	if pretrained_meta.get('survey_id') != config.survey_id:
		raise ValueError('embedding survey_id does not match benchmark config')
	pretrained_source, random_source = _inspect_model_sources(
		pretrained_meta, random_meta
	)
	volume_shape = _triplet(pretrained_meta.get('volume_shape_xyz'), 'volume shape')
	token_grid = _triplet(pretrained_meta.get('token_grid_shape'), 'token grid')
	patch_size = _triplet(pretrained_meta.get('patch_size'), 'patch size')
	if patch_size != HORIZON_PATCH_SIZE:
		raise ValueError('embedding patch size must be [8, 8, 8]')
	if volume_shape != (*data.shape_xy, len(data.time_ms)):
		raise ValueError('embedding volume geometry does not match binding geometry')
	expected_grid = tuple(
		math.ceil(size / patch)
		for size, patch in zip(volume_shape, patch_size, strict=True)
	)
	if token_grid != expected_grid or tuple(pretrained_array.shape[:3]) != token_grid:
		raise ValueError('embedding token grid is inconsistent with volume geometry')
	embedding_dim = int(pretrained_array.shape[-1])
	if embedding_dim != HORIZON_EMBEDDING_DIM:
		raise ValueError('embedding dimension must be 384')
	pretrained_valid = np.load(
		pretrained.valid_tokens, mmap_mode='r', allow_pickle=False
	)
	random_valid = np.load(random_paths.valid_tokens, mmap_mode='r', allow_pickle=False)
	_validate_valid_token_pair(
		pretrained_valid,
		random_valid,
		token_grid=token_grid,
		trace_valid_mask=data.valid_trace_mask,
		volume_shape=volume_shape,
	)
	window_start = HORIZON_WINDOW_START // HORIZON_PATCH_SIZE[2]
	window_stop = HORIZON_WINDOW_STOP // HORIZON_PATCH_SIZE[2]
	model_valid_lateral = frozen_survey_output_valid_mask(
		pretrained_valid[:, :, window_start:window_stop],
		HorizonTileSettings(
			lateral_shape_xy=data.shape_xy,
			min_token_valid_fraction=1.0,
		),
	)
	pretrained_valid_sha = file_sha256(pretrained.valid_tokens)
	if pretrained_valid_sha != file_sha256(random_paths.valid_tokens):
		raise ValueError('pretrained/random valid-token artifact hashes differ')
	canonical_identity = _validate_canonical_scientific_identity(
		config.canonical_input_metadata,
		pretrained_meta,
		pretrained_source,
		data,
	)
	return FrozenEmbeddingGeometry(
		pretrained=pretrained,
		random=random_paths,
		volume_shape_xyz=volume_shape,
		token_grid_shape_xyz=token_grid,
		embedding_shape=cast(
			'tuple[int, int, int, int]',
			tuple(int(value) for value in pretrained_array.shape),
		),
		embedding_dim=embedding_dim,
		pretrained_metadata=dict(pretrained_meta),
		random_metadata=dict(random_meta),
		pretrained_model_source=pretrained_source,
		random_model_source=random_source,
		valid_tokens_sha256=pretrained_valid_sha,
		model_valid_lateral_mask=model_valid_lateral,
		model_valid_lateral_mask_sha256=array_sha256(model_valid_lateral),
		canonical_identity=canonical_identity,
	)


def inspect_frozen_horizon_job(  # noqa: PLR0913
	config: FrozenHorizonConfig,
	*,
	model: str,
	layout_id: str,
	data_size: str,
	layout_config: str | Path,
	data: VolveHorizonData | None = None,
) -> FrozenHorizonPlan:
	'''Build one validated condition from the common split and embedding pair.'''
	if model not in FROZEN_MODEL_ROLES:
		raise ValueError("model must be 'pretrained' or 'random'")
	if layout_id not in LAYOUT_IDS:
		raise ValueError(f'layout must be one of {LAYOUT_IDS!r}')
	if data_size not in DATA_SIZE_PREFIX:
		raise ValueError(f'size must be one of {tuple(DATA_SIZE_PREFIX)!r}')
	resolved_data = load_volve_horizon_data(config.volve_root) if data is None else data
	job_config = replace(
		config,
		tiles=replace(config.tiles, lateral_shape_xy=resolved_data.shape_xy),
	)
	layouts = load_volve_horizon_layouts(layout_config, resolved_data)
	split_plan = build_horizon_split_plan(
		resolved_data, layouts, layout_id, data_size
	)
	if (
		split_plan.twt_window.start_index,
		split_plan.twt_window.stop_index_exclusive,
	) != (HORIZON_WINDOW_START, HORIZON_WINDOW_STOP):
		raise ValueError('split plan must use the fixed [552, 768) TWT window')
	geometry = inspect_frozen_embedding_pair(job_config, resolved_data)
	records: dict[str, tuple[HorizonTileRecord, ...]] = {}
	native_counts: dict[str, tuple[int, ...]] = {}
	effective_counts: dict[str, tuple[int, ...]] = {}
	excluded_counts: dict[str, tuple[int, ...]] = {}
	model_valid = geometry.model_valid_lateral_mask[np.newaxis, :, :]
	for split in ('train', 'validation', 'test'):
		native_split_mask = _split_mask(split_plan, split)
		effective_split_mask = native_split_mask & model_valid
		records[split] = enumerate_horizon_tile_records(
			sample_float=resolved_data.sample_float,
			native_valid_mask=resolved_data.bound_valid_mask,
			split_mask=effective_split_mask,
			trace_valid_mask=resolved_data.valid_trace_mask,
			settings=job_config.tiles,
		)
		native_supervision = horizon_supervision_mask(
			sample_float=resolved_data.sample_float,
			native_valid_mask=resolved_data.bound_valid_mask,
			split_mask=native_split_mask,
			trace_valid_mask=resolved_data.valid_trace_mask,
			window_start=job_config.tiles.window_start,
			window_stop=job_config.tiles.window_stop,
		)
		effective_supervision = native_supervision & model_valid
		native_counts[split] = _per_horizon_counts(native_supervision)
		effective_counts[split] = _per_horizon_counts(effective_supervision)
		excluded_counts[split] = tuple(
			native - effective
			for native, effective in zip(
				native_counts[split], effective_counts[split], strict=True
			)
		)
	primary_split = np.broadcast_to(
		split_plan.test_primary_mask, resolved_data.bound_valid_mask.shape
	)
	primary_native = horizon_supervision_mask(
		sample_float=resolved_data.sample_float,
		native_valid_mask=resolved_data.bound_valid_mask,
		split_mask=primary_split,
		trace_valid_mask=resolved_data.valid_trace_mask,
		window_start=job_config.tiles.window_start,
		window_stop=job_config.tiles.window_stop,
	)
	primary_effective = primary_native & model_valid
	native_counts['test_primary'] = _per_horizon_counts(primary_native)
	effective_counts['test_primary'] = _per_horizon_counts(primary_effective)
	excluded_counts['test_primary'] = tuple(
		native - effective
		for native, effective in zip(
			native_counts['test_primary'],
			effective_counts['test_primary'],
			strict=True,
		)
	)
	validate_training_horizon_coverage(effective_counts['train'])
	_require_positive_horizon_counts(
		effective_counts['validation'], 'validation'
	)
	_require_positive_horizon_counts(
		effective_counts['test_primary'], 'primary common test'
	)
	for split in ('train', 'validation', 'test'):
		if not records[split]:
			raise ValueError(
				f'{split} split has no model-valid supervised horizon tiles'
			)
	output_dir = (
		config.runs_root
		/ f'model={model}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)
	identity = _run_identity(
		config=job_config,
		model=model,
		split_plan=split_plan,
		geometry=geometry,
		records=records,
		native_counts=native_counts,
		effective_counts=effective_counts,
		excluded_counts=excluded_counts,
	)
	return FrozenHorizonPlan(
		config=job_config,
		model=model,
		layout_id=layout_id,
		data_size=data_size,
		output_dir=output_dir,
		data=resolved_data,
		split_plan=split_plan,
		geometry=geometry,
		tile_records=records,
		native_per_horizon_counts=native_counts,
		effective_per_horizon_counts=effective_counts,
		excluded_by_token_validity_counts=excluded_counts,
		run_identity=identity,
	)


def decoder_initial_state_sha256(seed: int = HORIZON_DECODER_SEED) -> str:
	'''Hash the fixed decoder initialization shared by both model roles.'''
	decoder = create_volve_horizon_decoder(seed=seed)
	digest = hashlib.sha256()
	for name, value in decoder.state_dict().items():
		digest.update(name.encode())
		digest.update(str(value.dtype).encode())
		digest.update(json.dumps(list(value.shape)).encode())
		digest.update(value.detach().cpu().contiguous().numpy().tobytes())
	return digest.hexdigest()


def deterministic_tile_order(tile_count: int, seed: int, epoch: int) -> tuple[int, ...]:
	'''Return the all-tiles-once order for one epoch.'''
	if tile_count <= 0:
		raise ValueError('tile_count must be positive')
	if epoch < 0:
		raise ValueError('epoch must be non-negative')
	generator = torch.Generator().manual_seed(seed + epoch)
	return tuple(
		int(index) for index in torch.randperm(tile_count, generator=generator)
	)


def validation_mae_improved(candidate: float, best: float) -> bool:
	'''Return true only for a finite, strictly lower validation macro MAE.'''
	if not math.isfinite(candidate):
		raise ValueError('validation macro MAE candidate must be finite')
	if math.isnan(best) or best == -math.inf:
		raise ValueError('best validation macro MAE is invalid')
	return candidate < best


def run_frozen_horizon_job(  # noqa: C901, PLR0912, PLR0915
	plan: FrozenHorizonPlan,
	*,
	device: str = 'auto',
	max_steps: int | None = None,
	resume: str | Path | None = None,
	decoder_factory: Callable[[], VolveHorizonDecoder] | None = None,
) -> Path | None:
	'''Train one decoder, select by validation macro MAE, and test best once.'''
	if max_steps is not None and (
		not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
	):
		raise ValueError('max_steps must be a positive integer')
	resume_path = None if resume is None else Path(resume)
	_validate_output(plan.output_dir, resume_path)
	run_device = _resolve_device(device)
	runtime_precision = _runtime_precision_identity(
		run_device, amp_requested=plan.config.train.amp
	)
	runtime_run_identity = {
		**plan.run_identity,
		'runtime_precision': runtime_precision,
	}
	_configure_determinism()
	_seed_everything(plan.config.train.seed)
	selected_paths = (
		plan.geometry.pretrained
		if plan.model == 'pretrained'
		else plan.geometry.random
	)
	datasets = {
		split: FrozenHorizonTileDataset(
			data=plan.data,
			plan=plan.split_plan,
			embedding_path=selected_paths.embeddings,
			valid_tokens_path=selected_paths.valid_tokens,
			settings=plan.config.tiles,
			split=split,
			records=plan.tile_records[split],
		)
		for split in ('train', 'validation', 'test')
	}
	decoder = (
		create_volve_horizon_decoder(seed=plan.config.train.seed)
		if decoder_factory is None
		else decoder_factory()
	).to(run_device)
	optimizer = torch.optim.AdamW(
		decoder.parameters(),
		lr=plan.config.train.learning_rate,
		betas=OPTIMIZER_BETAS,
		eps=OPTIMIZER_EPS,
		weight_decay=plan.config.train.weight_decay,
	)
	amp_enabled = bool(runtime_precision['amp_enabled'])
	scaler = torch.amp.GradScaler('cuda', enabled=True) if amp_enabled else None
	history: list[dict[str, object]] = []
	best_epoch: int | None = None
	best_mae = math.inf
	global_step = 0
	start_epoch = 0
	start_position = 0
	train_loss_sum = 0.0
	train_tile_count = 0
	if resume_path is not None:
		payload = torch.load(resume_path, map_location=run_device, weights_only=False)
		_validate_resume_runtime(
			payload,
			expected=runtime_precision,
			scaler=scaler,
		)
		if payload.get('run_identity') != runtime_run_identity:
			raise ValueError('resume checkpoint does not match this frozen horizon job')
		if payload.get('completed') is True:
			raise ValueError('completed frozen horizon job cannot be resumed')
		decoder.load_state_dict(_state_dict(payload))
		optimizer.load_state_dict(_mapping(payload, 'optimizer_state_dict'))
		if scaler is not None:
			scaler.load_state_dict(_mapping(payload, 'scaler_state_dict'))
		history = [
			dict(cast('Mapping[str, object]', row))
			for row in _sequence(payload.get('history'), 'history')
		]
		best_epoch = _optional_int(payload.get('best_epoch'), 'best_epoch')
		best_mae = float(payload.get('best_validation_macro_mae_samples', math.inf))
		global_step = _nonnegative_int(payload.get('global_step'), 'global_step')
		start_epoch = _nonnegative_int(payload.get('epoch'), 'epoch')
		start_position = _nonnegative_int(payload.get('next_position'), 'next_position')
		train_loss_sum = _finite_number(payload.get('train_loss_sum'), 'train_loss_sum')
		train_tile_count = _nonnegative_int(
			payload.get('train_tile_count'), 'train_tile_count'
		)
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
					history=history,
					best_epoch=best_epoch,
					best_mae=best_mae,
					global_step=global_step,
					epoch=epoch,
					next_position=position,
					train_loss_sum=train_loss_sum,
					train_tile_count=train_tile_count,
					completed=False,
					runtime_precision=runtime_precision,
					run_identity=runtime_run_identity,
				)
				_write_json(plan.output_dir / HISTORY_NAME, history)
				return None
			loss_value = _train_one_tile(
				decoder,
				datasets['train'][order[position]],
				optimizer,
				scaler,
				run_device,
				amp_enabled=amp_enabled,
				gradient_clip_norm=plan.config.train.gradient_clip_norm,
			)
			train_loss_sum += loss_value
			train_tile_count += 1
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
					history=history,
					best_epoch=best_epoch,
					best_mae=best_mae,
					global_step=global_step,
					epoch=epoch,
					next_position=position + 1,
					train_loss_sum=train_loss_sum,
					train_tile_count=train_tile_count,
					completed=False,
					runtime_precision=runtime_precision,
					run_identity=runtime_run_identity,
				)
				_write_json(plan.output_dir / HISTORY_NAME, history)
				return None
		validation = _evaluate_horizon_dataset(
			decoder,
			datasets['validation'],
			run_device,
			amp_enabled=amp_enabled,
			expected_counts=plan.effective_per_horizon_counts['validation'],
			expected_primary_counts=plan.effective_per_horizon_counts[
				'validation'
			],
		)
		validation_mae = _required_metric(
			validation['secondary'], 'macro_mae_samples'
		)
		history.append(
			{
				'epoch': epoch,
				'global_step': global_step,
				'train_macro_cross_entropy': train_loss_sum / train_tile_count,
				'validation_macro_mae_samples': validation_mae,
				'validation_macro_within_2_samples': _required_metric(
					validation['secondary'], 'macro_within_2_samples'
				),
			}
		)
		if validation_mae_improved(validation_mae, best_mae):
			best_mae = validation_mae
			best_epoch = epoch
			_save_checkpoint(
				plan.output_dir / BEST_NAME,
				decoder=decoder,
				optimizer=optimizer,
				scaler=scaler,
				payload={
					'run_identity': runtime_run_identity,
					'runtime_precision': runtime_precision,
					'epoch': epoch,
					'global_step': global_step,
					'validation': validation['secondary'],
				},
			)
		_save_latest(
			plan,
			decoder,
			optimizer,
			scaler,
			history=history,
			best_epoch=best_epoch,
			best_mae=best_mae,
			global_step=global_step,
			epoch=epoch + 1,
			next_position=0,
			train_loss_sum=0.0,
			train_tile_count=0,
			completed=False,
			runtime_precision=runtime_precision,
			run_identity=runtime_run_identity,
		)
		_write_json(plan.output_dir / HISTORY_NAME, history)
		start_position = 0
		train_loss_sum = 0.0
		train_tile_count = 0
		if max_steps is not None and global_step >= max_steps:
			return None
	if best_epoch is None:
		raise RuntimeError('training completed without a best checkpoint')
	best_path = plan.output_dir / BEST_NAME
	best = torch.load(best_path, map_location=run_device, weights_only=False)
	if best.get('run_identity') != runtime_run_identity:
		raise ValueError('best checkpoint identity changed before test evaluation')
	if best.get('runtime_precision') != runtime_precision:
		raise ValueError(
			'best checkpoint runtime precision changed before test evaluation'
		)
	decoder.load_state_dict(_state_dict(best))
	test = _evaluate_horizon_dataset(
		decoder,
		datasets['test'],
		run_device,
		amp_enabled=amp_enabled,
		expected_counts=plan.effective_per_horizon_counts['test'],
		expected_primary_counts=plan.effective_per_horizon_counts[
			'test_primary'
		],
	)
	metrics_payload = {
		'schema_version': 1,
		'artifact_type': 'volve_frozen_horizon_job_metrics',
		'model': plan.model,
		'layout_id': plan.layout_id,
		'data_size': plan.data_size,
		'benchmark_identity': runtime_run_identity,
		'runtime_precision': runtime_precision,
		'best_epoch': best_epoch,
		'best_checkpoint': {
			'path': str(best_path),
			'sha256': file_sha256(best_path),
		},
		'validation': best['validation'],
		'test': {
			'primary_common': test['primary'],
			'secondary_per_horizon': test['secondary'],
			'evaluation_pass_count': 1,
		},
	}
	metrics_path = plan.output_dir / METRICS_NAME
	_write_json(metrics_path, metrics_payload)
	_save_latest(
		plan,
		decoder,
		optimizer,
		scaler,
		history=history,
		best_epoch=best_epoch,
		best_mae=best_mae,
		global_step=global_step,
		epoch=plan.config.train.epochs,
		next_position=0,
		train_loss_sum=0.0,
		train_tile_count=0,
		completed=True,
		runtime_precision=runtime_precision,
		run_identity=runtime_run_identity,
	)
	return metrics_path


def _train_one_tile(  # noqa: PLR0913
	decoder: VolveHorizonDecoder,
	item: Mapping[str, object],
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	device: torch.device,
	*,
	amp_enabled: bool,
	gradient_clip_norm: float,
) -> float:
	decoder.train()
	embeddings = _tensor(item, 'embeddings').unsqueeze(0).to(device).detach()
	valid = _tensor(item, 'token_valid_mask').unsqueeze(0).to(device)
	target = _tensor(item, 'target_sample_float').unsqueeze(0).to(device)
	mask = _tensor(item, 'supervision_mask').unsqueeze(0).to(device)
	output_valid = _tensor(item, 'output_valid_mask').unsqueeze(0).to(device)
	effective_mask = mask & output_valid.unsqueeze(1)
	optimizer.zero_grad(set_to_none=True)
	with _autocast(device, enabled=amp_enabled):
		logits = decoder(embeddings, valid)
		loss, _ = fractional_horizon_cross_entropy(logits, target, effective_mask)
	backward_and_step_horizon_optimizer(
		loss=loss,
		model=decoder,
		optimizer=optimizer,
		scaler=scaler,
		gradient_clip_norm=gradient_clip_norm,
	)
	return float(loss.detach().cpu().item())


def _evaluate_horizon_dataset(  # noqa: PLR0913
	decoder: VolveHorizonDecoder,
	dataset: Dataset[dict[str, Any]],
	device: torch.device,
	*,
	amp_enabled: bool,
	expected_counts: Sequence[int],
	expected_primary_counts: Sequence[int],
) -> dict[str, object]:
	decoder.eval()
	predictions: list[np.ndarray] = []
	targets: list[np.ndarray] = []
	secondary_masks: list[np.ndarray] = []
	primary_masks: list[np.ndarray] = []
	with torch.inference_mode():
		for index in range(len(dataset)):
			item = dataset[index]
			embeddings = _tensor(item, 'embeddings').unsqueeze(0).to(device)
			valid = _tensor(item, 'token_valid_mask').unsqueeze(0).to(device)
			with _autocast(device, enabled=amp_enabled):
				logits = decoder(embeddings, valid)
			prediction = soft_argmax_global_sample(logits)
			predictions.append(prediction.cpu().numpy())
			targets.append(
				_tensor(item, 'target_sample_float').unsqueeze(0).numpy()
			)
			secondary_masks.append(
				(
					_tensor(item, 'supervision_mask')
					& _tensor(item, 'output_valid_mask').unsqueeze(0)
				)
				.unsqueeze(0)
				.numpy()
			)
			primary_masks.append(
				(
					_tensor(item, 'primary_evaluation_mask')
					& _tensor(item, 'output_valid_mask').unsqueeze(0)
				)
				.unsqueeze(0)
				.numpy()
			)
	predicted = np.concatenate(predictions)
	target = np.concatenate(targets)
	secondary_mask = np.concatenate(secondary_masks)
	primary_mask = np.concatenate(primary_masks)
	actual_counts = tuple(
		int(np.count_nonzero(secondary_mask[:, index]))
		for index in range(len(HORIZON_NAMES))
	)
	if actual_counts != tuple(expected_counts):
		raise RuntimeError(
			'evaluation tiles do not provide exact-once lateral coverage; '
			f'expected {tuple(expected_counts)!r}, got {actual_counts!r}'
		)
	actual_primary_counts = tuple(
		int(np.count_nonzero(primary_mask[:, index]))
		for index in range(len(HORIZON_NAMES))
	)
	if actual_primary_counts != tuple(expected_primary_counts):
		raise RuntimeError(
			'primary evaluation tiles do not provide exact-once model-valid coverage; '
			f'expected {tuple(expected_primary_counts)!r}, '
			f'got {actual_primary_counts!r}'
		)
	return {
		'primary': compute_horizon_metrics(predicted, target, primary_mask),
		'secondary': compute_horizon_metrics(predicted, target, secondary_mask),
	}


def _run_identity(  # noqa: PLR0913
	*,
	config: FrozenHorizonConfig,
	model: str,
	split_plan: HorizonSplitPlan,
	geometry: FrozenEmbeddingGeometry,
	records: Mapping[str, tuple[HorizonTileRecord, ...]],
	native_counts: Mapping[str, tuple[int, ...]],
	effective_counts: Mapping[str, tuple[int, ...]],
	excluded_counts: Mapping[str, tuple[int, ...]],
) -> dict[str, object]:
	metadata = (
		geometry.pretrained_metadata
		if model == 'pretrained'
		else geometry.random_metadata
	)
	model_source = (
		geometry.pretrained_model_source
		if model == 'pretrained'
		else geometry.random_model_source
	)
	return {
		'schema_version': 3,
		'benchmark': 'mae_vs_random_frozen_v1',
		'model': model,
		'layout_id': split_plan.layout_id,
		'data_size': split_plan.data_size,
		'canonical_scientific_identity': dict(geometry.canonical_identity),
		'horizon_split_plan': split_plan.identity(),
		'embedding': {
			'metadata_path': str(
				geometry.pretrained.metadata
				if model == 'pretrained'
				else geometry.random.metadata
			),
			'metadata_sha256': file_sha256(
				geometry.pretrained.metadata
				if model == 'pretrained'
				else geometry.random.metadata
			),
			'checkpoint_path': metadata['checkpoint_path'],
			'checkpoint_sha256': metadata['checkpoint_sha256'],
			'model_source': dict(model_source),
			'embedding_shape': list(geometry.embedding_shape),
			'token_grid_shape_xyz': list(geometry.token_grid_shape_xyz),
			'valid_tokens_sha256': geometry.valid_tokens_sha256,
			'output_validity_policy': (
				'full_216_sample_token_column_then_8x8_lateral_expansion_v1'
			),
			'model_valid_lateral_mask_sha256': (
				geometry.model_valid_lateral_mask_sha256
			),
		},
		'decoder': {
			'architecture': create_volve_horizon_decoder().architecture,
			'initialization_seed': config.train.seed,
			'initial_state_sha256': decoder_initial_state_sha256(config.train.seed),
		},
		'tiles': {
			'patch_size_xyz': list(config.tiles.patch_size_xyz),
			'core_size_tokens': list(config.tiles.core_size_tokens),
			'context_halo_tokens': list(config.tiles.context_halo_tokens),
			'window_start': config.tiles.window_start,
			'window_stop': config.tiles.window_stop,
			'order': 'lateral_token_grid_x_then_y_v1',
			'counts': {split: len(records[split]) for split in records},
			'record_sha256': {
				split: _records_sha256(records[split]) for split in records
			},
		},
		'native_horizon_observation_counts': _identity_counts(native_counts),
		'effective_model_valid_observation_counts': _identity_counts(
			effective_counts
		),
		'excluded_by_token_validity_counts': _identity_counts(excluded_counts),
		'training': {
			'epochs': config.train.epochs,
			'batch_size': config.train.batch_size,
			'learning_rate': config.train.learning_rate,
			'weight_decay': config.train.weight_decay,
			'sampling_mode': config.train.sampling_mode,
			'seed': config.train.seed,
			'amp_on_cuda': config.train.amp,
			'gradient_clip_norm': config.train.gradient_clip_norm,
		},
		'optimizer': {
			'name': OPTIMIZER_NAME,
			'betas': list(OPTIMIZER_BETAS),
			'eps': OPTIMIZER_EPS,
			'weight_decay': config.train.weight_decay,
		},
		'objective': dict(OBJECTIVE_IDENTITY),
	}


def _inspect_model_sources(  # noqa: C901
	pretrained_metadata: Mapping[str, object],
	random_metadata: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
	pretrained_path, pretrained_sha = _validated_checkpoint(
		pretrained_metadata, 'pretrained'
	)
	random_path, random_sha = _validated_checkpoint(random_metadata, 'random')
	if pretrained_sha == random_sha:
		raise ValueError('pretrained/random checkpoint SHA-256 must differ')
	if tuple(pretrained_path.parts[-len(VOLVE_PRETRAINED_CHECKPOINT_SUFFIX) :]) != (
		VOLVE_PRETRAINED_CHECKPOINT_SUFFIX
	):
		raise ValueError(
			'pretrained embedding must come from Volve full_100ep/latest.pt'
		)
	pretrained_payload = load_checkpoint_metadata_without_weights(pretrained_path)
	random_payload = load_checkpoint_metadata_without_weights(random_path)
	pretrained_config = _mapping(pretrained_payload, 'config')
	random_config = _mapping(random_payload, 'config')
	_validate_mae_checkpoint_geometry(pretrained_config, pretrained_metadata)
	_validate_mae_checkpoint_geometry(random_config, random_metadata)
	if _mapping(pretrained_config, 'model') != _mapping(random_config, 'model'):
		raise ValueError('pretrained/random checkpoint architecture mismatch')
	if pretrained_payload.get('epoch') != 100:
		raise ValueError('pretrained Volve checkpoint epoch must equal 100')
	pretrained_train = _mapping(pretrained_config, 'train')
	if pretrained_train.get('epochs') != 100 or pretrained_train.get('seed') != 42:
		raise ValueError(
			'pretrained Volve checkpoint must be the seed-42 100-epoch run'
		)
	pretrained_state = _mapping(pretrained_payload, 'training_state')
	if (
		pretrained_state.get('stage') != 'train_amp_mae'
		or pretrained_state.get('checkpoint_kind') != 'epoch'
	):
		raise ValueError(
			'pretrained Volve checkpoint role must be a completed MAE epoch'
		)
	metadata = _mapping(random_payload, 'metadata')
	training_state = _mapping(random_payload, 'training_state')
	for key, expected in (
		('random_encoder_baseline', True),
		('pretrained_weights_loaded', False),
		('seed', VOLVE_RANDOM_ENCODER_SEED),
		('reference_model_tag', VOLVE_MAE_MODEL_TAG),
	):
		if metadata.get(key) != expected:
			raise ValueError(
				f'random checkpoint metadata.{key} must equal {expected!r}'
			)
	if training_state.get('checkpoint_kind') != 'random_init':
		raise ValueError('random checkpoint role must be random_init')
	reference_value = metadata.get('reference_checkpoint')
	if not isinstance(reference_value, str) or not reference_value:
		raise TypeError('random checkpoint reference_checkpoint must be non-empty')
	if Path(reference_value).resolve(
		strict=False
	) != pretrained_path.resolve(strict=False):
		raise ValueError('random checkpoint must reference the paired pretrained model')
	return (
		{
			'role': 'pretrained',
			'model_tag': VOLVE_MAE_MODEL_TAG,
			'checkpoint_path': str(pretrained_path),
			'checkpoint_sha256': pretrained_sha,
		},
		{
			'role': 'random',
			'model_tag': f'random_encoder_{VOLVE_MAE_MODEL_TAG}_seed42',
			'checkpoint_path': str(random_path),
			'checkpoint_sha256': random_sha,
			'seed': VOLVE_RANDOM_ENCODER_SEED,
			'reference_checkpoint': reference_value,
			'reference_checkpoint_sha256': pretrained_sha,
		},
	)


def _validate_canonical_scientific_identity(  # noqa: C901, PLR0912
	metadata_path: Path,
	embedding_metadata: Mapping[str, object],
	pretrained_source: Mapping[str, object],
	data: VolveHorizonData,
) -> Mapping[str, object]:
	metadata = _read_json(metadata_path, 'canonical input metadata')
	if metadata.get('status') != 'PASS' or metadata.get('artifact_type') != (
		'volve_canonical_input_registration'
	):
		raise ValueError('canonical input metadata must be a PASS registration')
	identity = _mapping(metadata, 'scientific_identity')
	identity_sha = _sha256_json(identity)
	if metadata.get('scientific_identity_sha256') != identity_sha:
		raise ValueError('canonical scientific identity SHA-256 mismatch')
	if identity.get('survey_id') != embedding_metadata.get('survey_id'):
		raise ValueError('canonical and embedding survey identities differ')
	for key in (
		'canonical_amplitude_sha256',
		'valid_trace_mask_sha256',
		'inline_values_sha256',
		'crossline_values_sha256',
		'time_axis_sha256',
		'canonical_normalization_stats_sha256',
	):
		_validate_sha256(identity.get(key), f'canonical identity {key}')
	if tuple(identity.get('shape_xyz', ())) != (*data.shape_xy, len(data.time_ms)):
		raise ValueError('canonical registration shape does not match horizon inputs')
	provenance = _mapping(metadata, 'provenance')
	public_inputs = _mapping(provenance, 'public_inputs')
	if Path(cast('str', public_inputs.get('valid_trace_mask.npy'))).resolve() != (
		data.paths.valid_trace_mask.resolve()
	):
		raise ValueError('canonical registration valid mask differs from binding input')
	outputs = _mapping(metadata, 'outputs')
	for metadata_key, expected in (
		('source_amplitude_path', _mapping(provenance, 'amplitude').get('path')),
		('source_valid_mask_path', public_inputs.get('valid_trace_mask.npy')),
		('normalization_stats_path', outputs.get('normalization_stats')),
	):
		if embedding_metadata.get(metadata_key) != expected:
			raise ValueError(
				f'embedding {metadata_key} does not match canonical registration'
			)
	pretrained_path = Path(cast('str', pretrained_source['checkpoint_path']))
	run_root = pretrained_path.parent
	run_metadata = _read_json(run_root / 'run_metadata.json', 'MAE run metadata')
	snapshot_path = run_root / 'inputs' / metadata_path.name
	if (
		not snapshot_path.is_file()
		or snapshot_path.read_bytes() != metadata_path.read_bytes()
	):
		raise ValueError(
			'MAE canonical input snapshot differs from current registration'
		)
	if run_metadata.get('input_scientific_identity_sha256') != identity_sha:
		raise ValueError(
			'MAE run scientific identity differs from current registration'
		)
	if run_metadata.get('canonical_input_metadata_sha256') != file_sha256(
		snapshot_path
	):
		raise ValueError('MAE canonical metadata snapshot hash mismatch')
	normalization_path = Path(cast('str', outputs.get('normalization_stats')))
	normalization_snapshot = run_root / 'inputs' / normalization_path.name
	if not normalization_snapshot.is_file() or (
		run_metadata.get('normalization_stats_sha256')
		!= file_sha256(normalization_snapshot)
	):
		raise ValueError('MAE normalization snapshot hash mismatch')
	if not normalization_path.is_file() or (
		normalization_snapshot.read_bytes() != normalization_path.read_bytes()
	):
		raise ValueError('embedding normalization differs from MAE run snapshot')
	return {
		'scientific_identity_sha256': identity_sha,
		'canonical_amplitude_sha256': identity['canonical_amplitude_sha256'],
		'valid_trace_mask_sha256': identity['valid_trace_mask_sha256'],
		'inline_values_sha256': identity['inline_values_sha256'],
		'crossline_values_sha256': identity['crossline_values_sha256'],
		'time_axis_sha256': identity['time_axis_sha256'],
		'normalization_stats_sha256': run_metadata['normalization_stats_sha256'],
		'canonical_input_metadata_sha256': file_sha256(metadata_path),
	}


def _validate_embedding_arrays(
	pretrained: np.ndarray, random_array: np.ndarray
) -> None:
	if pretrained.shape != random_array.shape:
		raise ValueError('pretrained/random embedding shape mismatch')
	if pretrained.ndim != 4:
		raise ValueError('embeddings must have shape [TX,TY,TZ,D]')
	if pretrained.dtype != np.float16 or random_array.dtype != np.float16:
		raise TypeError('frozen benchmark embeddings must have dtype float16')


def _validate_embedding_metadata_pair(
	pretrained: Mapping[str, object],
	random_meta: Mapping[str, object],
	*,
	pretrained_dtype: np.dtype[Any],
	random_dtype: np.dtype[Any],
) -> None:
	for key in _PAIRED_EMBEDDING_METADATA_KEYS:
		if key not in pretrained or key not in random_meta:
			raise ValueError(f'paired embedding metadata requires {key}')
		if pretrained[key] != random_meta[key]:
			raise ValueError(f'pretrained/random embedding metadata {key} mismatch')
	if np.dtype(cast('str', pretrained['output_dtype'])) != pretrained_dtype:
		raise TypeError('pretrained embedding dtype does not match metadata')
	if np.dtype(cast('str', random_meta['output_dtype'])) != random_dtype:
		raise TypeError('random embedding dtype does not match metadata')
	if pretrained.get('window_size') != [128, 128, 128]:
		raise ValueError('embedding window_size must be [128, 128, 128]')
	if pretrained.get('overlap') != [64, 64, 64]:
		raise ValueError('embedding overlap must be [64, 64, 64]')
	if pretrained.get('min_token_valid_fraction') != 1.0:
		raise ValueError('embedding min_token_valid_fraction must equal 1.0')


def _validate_valid_token_pair(
	pretrained: np.ndarray,
	random_mask: np.ndarray,
	*,
	token_grid: tuple[int, int, int],
	trace_valid_mask: np.ndarray,
	volume_shape: tuple[int, int, int],
) -> None:
	if (
		pretrained.dtype != np.bool_
		or random_mask.dtype != np.bool_
		or pretrained.shape != token_grid
		or random_mask.shape != token_grid
	):
		raise TypeError('valid-token masks must be bool with the full token grid')
	if not np.array_equal(pretrained, random_mask):
		raise ValueError('pretrained/random valid-token masks differ')
	expected_lateral = np.zeros(token_grid[:2], dtype=np.bool_)
	for token_x in range(token_grid[0]):
		for token_y in range(token_grid[1]):
			x0, y0 = token_x * 8, token_y * 8
			x1, y1 = x0 + 8, y0 + 8
			if x1 <= volume_shape[0] and y1 <= volume_shape[1]:
				expected_lateral[token_x, token_y] = bool(
					np.all(trace_valid_mask[x0:x1, y0:y1])
				)
	expected = np.zeros(token_grid, dtype=np.bool_)
	full_z_tokens = volume_shape[2] // 8
	expected[:, :, :full_z_tokens] = expected_lateral[:, :, np.newaxis]
	if np.any(pretrained & ~expected):
		raise ValueError('missing-trace or padding tokens must be invalid')


def _validate_mae_checkpoint_geometry(
	checkpoint_config: Mapping[str, object],
	embedding_metadata: Mapping[str, object],
) -> None:
	model = _mapping(checkpoint_config, 'model')
	for key, expected in _EXPECTED_MAE_GEOMETRY.items():
		if model.get(key) != expected:
			raise ValueError(f'MAE checkpoint model.{key} must equal {expected!r}')
	metadata_geometry = _mapping(embedding_metadata, 'model_geometry')
	for key, expected in _EXPECTED_MAE_GEOMETRY.items():
		if metadata_geometry.get(key) != expected:
			raise ValueError(f'embedding model geometry {key} must equal {expected!r}')


def _split_mask(plan: HorizonSplitPlan, split: str) -> np.ndarray:
	if split == 'train':
		return plan.train_mask
	if split == 'validation':
		return plan.validation_mask
	if split == 'test':
		return plan.test_per_horizon_mask
	return np.empty(0, dtype=np.bool_)


def _record_for_supervision_mask(
	record: HorizonTileRecord, supervision_mask: np.ndarray
) -> HorizonTileRecord:
	x0 = record.core_start_token_xy[0] * HORIZON_PATCH_SIZE[0]
	y0 = record.core_start_token_xy[1] * HORIZON_PATCH_SIZE[1]
	x1 = record.core_stop_token_xy[0] * HORIZON_PATCH_SIZE[0]
	y1 = record.core_stop_token_xy[1] * HORIZON_PATCH_SIZE[1]
	mask = np.asarray(supervision_mask)
	if mask.ndim != 3 or mask.shape[0] != len(HORIZON_NAMES):
		raise ValueError('supervision mask must have shape [5,X,Y]')
	counts = tuple(
		int(np.count_nonzero(mask[index, x0:x1, y0:y1]))
		for index in range(len(HORIZON_NAMES))
	)
	return HorizonTileRecord(
		tile_id=record.tile_id,
		core_start_token_xy=record.core_start_token_xy,
		core_stop_token_xy=record.core_stop_token_xy,
		per_horizon_observation_counts=counts,
	)


def _per_horizon_counts(mask: np.ndarray) -> tuple[int, ...]:
	return tuple(
		int(np.count_nonzero(mask[index])) for index in range(len(HORIZON_NAMES))
	)


def _require_positive_horizon_counts(
	counts: Sequence[int], split_name: str
) -> None:
	missing = [
		HORIZON_NAMES[index] for index, count in enumerate(counts) if count <= 0
	]
	if missing:
		raise ValueError(
			f'{split_name} has zero model-valid observations for horizons: '
			f'{", ".join(missing)}'
		)


def _identity_counts(
	counts: Mapping[str, tuple[int, ...]],
) -> dict[str, dict[str, int]]:
	identity_names = {
		'train': 'train',
		'validation': 'validation',
		'test': 'test_secondary_per_horizon',
		'test_primary': 'test_primary_common',
	}
	return {
		identity_names[split]: {
			name: values[index] for index, name in enumerate(HORIZON_NAMES)
		}
		for split, values in counts.items()
	}


def _save_latest(  # noqa: PLR0913
	plan: FrozenHorizonPlan,
	decoder: VolveHorizonDecoder,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	*,
	history: Sequence[Mapping[str, object]],
	best_epoch: int | None,
	best_mae: float,
	global_step: int,
	epoch: int,
	next_position: int,
	train_loss_sum: float,
	train_tile_count: int,
	completed: bool,
	runtime_precision: Mapping[str, object],
	run_identity: Mapping[str, object],
) -> None:
	_save_checkpoint(
		plan.output_dir / LATEST_NAME,
		decoder=decoder,
		optimizer=optimizer,
		scaler=scaler,
		payload={
			'run_identity': run_identity,
			'runtime_precision': runtime_precision,
			'history': list(history),
			'best_epoch': best_epoch,
			'best_validation_macro_mae_samples': best_mae,
			'global_step': global_step,
			'epoch': epoch,
			'next_position': next_position,
			'train_loss_sum': train_loss_sum,
			'train_tile_count': train_tile_count,
			'completed': completed,
		},
	)


def _save_checkpoint(
	path: Path,
	*,
	decoder: VolveHorizonDecoder,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	payload: Mapping[str, object],
) -> None:
	full = {
		**payload,
		'model_state_dict': {
			name: value.detach().cpu() for name, value in decoder.state_dict().items()
		},
		'optimizer_state_dict': optimizer.state_dict(),
		'scaler_state_dict': None if scaler is None else scaler.state_dict(),
	}
	temporary = path.with_name(f'.{path.name}.tmp')
	torch.save(full, temporary)
	temporary.replace(path)


def _validate_output(output_dir: Path, resume: Path | None) -> None:
	if resume is None:
		if output_dir.exists() and any(output_dir.iterdir()):
			raise FileExistsError(
				f'frozen horizon job output is non-empty: {output_dir}'
			)
		return
	if not resume.is_file() or resume.name != LATEST_NAME:
		raise FileNotFoundError(f'resume must identify an existing {LATEST_NAME}')
	if resume.parent.resolve() != output_dir.resolve():
		raise ValueError('resume checkpoint must be in this job output directory')


def _validated_checkpoint(
	metadata: Mapping[str, object], role: str
) -> tuple[Path, str]:
	path_value = metadata.get('checkpoint_path')
	if not isinstance(path_value, str) or not path_value:
		raise TypeError(f'{role} checkpoint_path must be non-empty')
	path = Path(path_value)
	if not path.is_file():
		raise FileNotFoundError(f'{role} checkpoint does not exist: {path}')
	sha = metadata.get('checkpoint_sha256')
	_validate_sha256(sha, f'{role} checkpoint SHA-256')
	if file_sha256(path) != sha:
		raise ValueError(f'{role} checkpoint SHA-256 does not match its file')
	return path, cast('str', sha)


def _records_sha256(records: Sequence[HorizonTileRecord]) -> str:
	payload = [
		{
			'tile_id': record.tile_id,
			'core_start_token_xy': list(record.core_start_token_xy),
			'core_stop_token_xy': list(record.core_stop_token_xy),
			'per_horizon_observation_counts': list(
				record.per_horizon_observation_counts
			),
		}
		for record in records
	]
	return hashlib.sha256(
		json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


def _validate_decoder(value: Mapping[str, object]) -> None:
	expected = {
		'embedding_dim': HORIZON_EMBEDDING_DIM,
		'class_count': len(HORIZON_NAMES),
		'hidden_channels': list(HORIZON_HIDDEN_CHANNELS),
		'upsample_factors': [list(item) for item in HORIZON_UPSAMPLE_FACTORS],
		'upsample_mode': 'nearest',
		'normalization': 'voxelwise_layer_norm',
	}
	if dict(value) != expected:
		raise ValueError('decoder settings differ from the fixed horizon benchmark')


def _train_settings(value: Mapping[str, object]) -> FrozenHorizonTrainSettings:
	settings = FrozenHorizonTrainSettings(
		epochs=_positive_int(value.get('epochs'), 'train.epochs'),
		batch_size=_positive_int(value.get('batch_size'), 'train.batch_size'),
		learning_rate=_positive_number(
			value.get('learning_rate'), 'train.learning_rate'
		),
		weight_decay=_nonnegative_number(
			value.get('weight_decay'), 'train.weight_decay'
		),
		sampling_mode=_non_empty_string(
			value.get('sampling_mode'), 'train.sampling_mode'
		),
		seed=_nonnegative_int(value.get('seed'), 'train.seed'),
		amp=_boolean(value.get('amp'), 'train.amp'),
		gradient_clip_norm=_positive_number(
			value.get('gradient_clip_norm'), 'train.gradient_clip_norm'
		),
	)
	expected = FrozenHorizonTrainSettings(
		epochs=50,
		batch_size=1,
		learning_rate=1.0e-3,
		weight_decay=1.0e-4,
		sampling_mode='all_tiles_once',
		seed=HORIZON_DECODER_SEED,
		amp=True,
		gradient_clip_norm=1.0,
	)
	if settings != expected:
		raise ValueError('train settings differ from the fixed horizon benchmark')
	return settings


def _tile_settings(value: Mapping[str, object]) -> HorizonTileSettings:
	if dict(value) != {
		'patch_size': [8, 8, 8],
		'core_size_tokens': [8, 8, 27],
		'context_halo_tokens': [1, 1, 0],
		'window_start': 552,
		'window_stop': 768,
		'min_token_valid_fraction': 1.0,
	}:
		raise ValueError('tile settings differ from the fixed horizon benchmark')
	# The lateral shape is replaced with canonical geometry during job inspection.
	return HorizonTileSettings(
		lateral_shape_xy=(401, 720), min_token_valid_fraction=1.0
	)


def _resolve_device(value: str) -> torch.device:
	if value not in {'auto', 'cpu', 'cuda'}:
		raise ValueError('device must be auto, cpu, or cuda')
	if value == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if value == 'cuda' and not torch.cuda.is_available():
		raise RuntimeError('CUDA was requested but is not available')
	return torch.device(value)


def _runtime_precision_identity(
	device: torch.device, *, amp_requested: bool
) -> dict[str, object]:
	amp_enabled = amp_requested and device.type == 'cuda'
	return {
		'device_type': device.type,
		'amp_enabled': amp_enabled,
		'autocast_dtype': 'float16' if amp_enabled else None,
		'scaler_required': amp_enabled,
	}


def _validate_resume_runtime(
	payload: Mapping[str, object],
	*,
	expected: Mapping[str, object],
	scaler: torch.amp.GradScaler | None,
) -> None:
	if payload.get('runtime_precision') != expected:
		raise ValueError('resume checkpoint runtime precision does not match this run')
	scaler_required = expected.get('scaler_required') is True
	scaler_state = payload.get('scaler_state_dict')
	if scaler_required:
		if scaler is None:
			raise ValueError('runtime precision requires a GradScaler')
		if not isinstance(scaler_state, Mapping) or not scaler_state:
			raise ValueError('resume checkpoint is missing required GradScaler state')
	elif scaler_state is not None:
		raise ValueError('resume checkpoint has unexpected GradScaler state')


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


def _autocast(
	device: torch.device, *, enabled: bool
) -> AbstractContextManager[None]:
	return torch.autocast(device_type=device.type) if enabled else nullcontext()


def _read_json(path: Path, label: str) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'{label} does not exist: {path}')
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must contain a JSON object')
	return value


def _write_json(path: Path, value: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.tmp')
	temporary.write_text(
		json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temporary.replace(path)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return child


def _state_dict(payload: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
	value = payload.get('model_state_dict')
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint model_state_dict must be a mapping')
	return cast('Mapping[str, torch.Tensor]', value)


def _tensor(value: Mapping[str, object], key: str) -> torch.Tensor:
	item = value.get(key)
	if not isinstance(item, torch.Tensor):
		raise TypeError(f'{key} must be a tensor')
	return item


def _array(value: np.ndarray | None) -> np.ndarray:
	if value is None:
		raise RuntimeError('dataset array is not open')
	return value


def _triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, list | tuple)
		or len(value) != 3
		or any(
			not isinstance(item, int)
			or isinstance(item, bool)
			or item <= 0
			for item in value
		)
	):
		raise TypeError(f'{label} must be a positive integer triple')
	return cast('tuple[int, int, int]', tuple(value))


def _absolute_path(value: Mapping[str, object], key: str, prefix: str) -> Path:
	path = Path(_non_empty_string(value.get(key), f'{prefix}.{key}'))
	if not path.is_absolute():
		raise ValueError(f'{prefix}.{key} must be absolute')
	return path


def _non_empty_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _positive_int(value: object, label: str) -> int:
	resolved = _nonnegative_int(value, label)
	if resolved == 0:
		raise ValueError(f'{label} must be positive')
	return resolved


def _nonnegative_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{label} must be an integer')
	if value < 0:
		raise ValueError(f'{label} must be non-negative')
	return value


def _optional_int(value: object, label: str) -> int | None:
	return None if value is None else _nonnegative_int(value, label)


def _positive_number(value: object, label: str) -> float:
	result = _finite_number(value, label)
	if result <= 0.0:
		raise ValueError(f'{label} must be positive')
	return result


def _nonnegative_number(value: object, label: str) -> float:
	result = _finite_number(value, label)
	if result < 0.0:
		raise ValueError(f'{label} must be non-negative')
	return result


def _finite_number(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{label} must be numeric')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label} must be finite')
	return result


def _boolean(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be boolean')
	return value


def _sequence(value: object, label: str) -> Sequence[object]:
	if not isinstance(value, list):
		raise TypeError(f'{label} must be a list')
	return value


def _required_metric(metrics: object, key: str) -> float:
	if not isinstance(metrics, Mapping):
		raise TypeError('metrics must be a mapping')
	value = metrics.get(key)
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or not math.isfinite(float(value))
	):
		raise ValueError(f'metric {key} must be finite')
	return float(value)


def _validate_sha256(value: object, label: str) -> None:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise TypeError(f'{label} must be a lowercase SHA-256 digest')


def _sha256_json(value: Mapping[str, object]) -> str:
	return hashlib.sha256(
		json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


__all__ = [
	'BEST_NAME',
	'FROZEN_CONDITION_COUNT',
	'FROZEN_MODEL_ROLES',
	'LATEST_NAME',
	'METRICS_NAME',
	'FrozenEmbeddingGeometry',
	'FrozenHorizonConfig',
	'FrozenHorizonPlan',
	'FrozenHorizonTileDataset',
	'FrozenHorizonTrainSettings',
	'decoder_initial_state_sha256',
	'deterministic_tile_order',
	'enumerate_frozen_horizon_conditions',
	'frozen_horizon_config_from_mapping',
	'inspect_frozen_embedding_pair',
	'inspect_frozen_horizon_job',
	'run_frozen_horizon_job',
	'validation_mae_improved',
]
