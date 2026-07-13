"""Config-driven runner for the frozen-embedding F3 voxel decoder."""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import seis_ssl_cluster
from seis_ssl_cluster.data.f3_voxel_decoder_dataset import (
	F3VoxelDecoderDataset,
	build_f3_voxel_decoder_dataloader,
	validate_encoder_pairing,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.io.labels import F3ClassInfo
from seis_ssl_cluster.f3.lithology.metrics import (
	lithology_metrics_from_confusion_matrix,
)
from seis_ssl_cluster.f3.lithology.voxel_dataset import GRID_NAME, METADATA_NAME
from seis_ssl_cluster.f3.lithology.voxel_tiles import (
	VoxelTileManifest,
	build_voxel_tile_manifests,
	write_voxel_tile_manifest,
)
from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	best_state_is_improved,
	load_voxel_decoder_checkpoint,
	make_best_selection_state,
	restore_voxel_decoder_checkpoint,
	save_voxel_decoder_checkpoint,
	validate_resume_identity,
)
from seis_ssl_cluster.training.voxel_decoder.epoch import (
	train_voxel_decoder_one_epoch,
	validate_voxel_decoder_one_epoch,
)
from seis_ssl_cluster.training.voxel_decoder.losses import (
	balanced_class_weights_from_counts,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
		F3LithologyVoxelDecoderConfig,
	)

LATEST_NAME = 'latest.pt'
BEST_NAME = 'best.pt'
HISTORY_NAME = 'history.csv'


@dataclass(frozen=True)
class VoxelDecoderInputPlan:
	"""Resolved source files and geometry, without loaded training arrays."""

	embeddings: Path
	valid_tokens: Path
	embedding_metadata: Path
	voxel_metadata: Path
	split_grid: Path
	label_volume: Path
	patch_size_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	volume_shape_xyz: tuple[int, int, int]
	class_ids: tuple[int, ...]
	class_names: tuple[str, ...]


@dataclass(frozen=True)
class VoxelDecoderRunResult:
	"""Primary outputs of a training invocation."""

	output_dir: Path
	latest_checkpoint: Path
	best_checkpoint: Path
	history_csv: Path
	global_step: int
	completed: bool


def inspect_f3_lithology_voxel_decoder(  # noqa: C901, PLR0912
	config: F3LithologyVoxelDecoderConfig,
) -> VoxelDecoderInputPlan:
	"""Inspect source metadata and memory-mapped array headers for a dry-run."""
	embedding_files = output_paths(config.embeddings_input_dir, config.survey_id)
	voxel_metadata = config.voxel_dataset_input_dir / METADATA_NAME
	split_grid = config.voxel_dataset_input_dir / GRID_NAME
	for path in (
		embedding_files.embeddings,
		embedding_files.valid_tokens,
		embedding_files.metadata,
		voxel_metadata,
		split_grid,
	):
		if not path.is_file():
			raise FileNotFoundError(f'missing voxel decoder input: {path}')
	embedding_payload = _read_json_object(embedding_files.metadata)
	voxel_payload = _read_json_object(voxel_metadata)
	_validate_source_provenance(
		config,
		embedding_payload=embedding_payload,
	)
	label_identity = voxel_payload.get('label_volume')
	if not isinstance(label_identity, Mapping):
		raise TypeError('voxel dataset metadata label_volume must be a mapping')
	label_path_value = label_identity.get('path')
	if not isinstance(label_path_value, str) or not label_path_value:
		raise ValueError('voxel dataset metadata label_volume.path is required')
	label_volume = Path(label_path_value)
	if not label_volume.is_file():
		raise FileNotFoundError(f'missing voxel decoder input: {label_volume}')
	classes_value = voxel_payload.get('classes')
	if not isinstance(classes_value, Sequence) or isinstance(
		classes_value, str | bytes
	):
		raise TypeError('voxel dataset metadata classes must be a list')
	class_ids: list[int] = []
	class_names: list[str] = []
	for item in classes_value:
		if not isinstance(item, Mapping):
			raise TypeError('voxel dataset metadata class entries must be mappings')
		class_id = item.get('class_id')
		class_name = item.get('class_name')
		if not isinstance(class_id, int) or isinstance(class_id, bool):
			raise TypeError('voxel dataset class_id must be an integer')
		if not isinstance(class_name, str) or not class_name:
			raise TypeError('voxel dataset class_name must be a non-empty string')
		class_ids.append(class_id)
		class_names.append(class_name)
	if len(set(class_ids)) != len(class_ids) or not class_ids:
		raise ValueError('voxel dataset class IDs must be non-empty and unique')
	if len(class_ids) != config.decoder.class_count:
		raise ValueError('decoder.class_count does not match voxel dataset classes')
	geometry = _metadata_geometry(embedding_payload)
	if config.decoder.embedding_dim != _embedding_dim(embedding_payload):
		raise ValueError('decoder.embedding_dim does not match embedding metadata')
	_validate_inspected_arrays(
		embeddings=embedding_files.embeddings,
		valid_tokens=embedding_files.valid_tokens,
		label_volume=label_volume,
		split_grid=split_grid,
		embedding_payload=embedding_payload,
		voxel_payload=voxel_payload,
		geometry=geometry,
	)
	return VoxelDecoderInputPlan(
		embeddings=embedding_files.embeddings,
		valid_tokens=embedding_files.valid_tokens,
		embedding_metadata=embedding_files.metadata,
		voxel_metadata=voxel_metadata,
		split_grid=split_grid,
		label_volume=label_volume,
		patch_size_xyz=geometry[0],
		token_grid_shape_xyz=geometry[1],
		volume_shape_xyz=geometry[2],
		class_ids=tuple(class_ids),
		class_names=tuple(class_names),
	)


def run_f3_lithology_voxel_decoder(  # noqa: C901, PLR0912, PLR0915
	config: F3LithologyVoxelDecoderConfig,
	*,
	device: str | torch.device = 'auto',
	max_steps: int | None = None,
	resume: str | Path | None = None,
) -> VoxelDecoderRunResult:
	"""Train, checkpoint, and optionally exactly resume one decoder job."""
	if max_steps is not None and (
		not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
	):
		raise ValueError('max_steps must be a positive integer')
	plan = inspect_f3_lithology_voxel_decoder(config)
	_validate_input_files(plan)
	output_dir = config.output_dir
	resume_path = None if resume is None else Path(resume)
	_validate_output_collision(output_dir, resume_path)
	run_device = _resolve_device(device)
	_seed_everything(config.train.seed)
	identities = _artifact_identities(plan)

	labels = np.load(plan.label_volume, mmap_mode='r', allow_pickle=False)
	split_grid = np.load(plan.split_grid, mmap_mode='r', allow_pickle=False)
	manifests = build_voxel_tile_manifests(
		split_grid,
		labels,
		patch_size_xyz=plan.patch_size_xyz,
		token_grid_shape_xyz=plan.token_grid_shape_xyz,
		core_size_tokens=config.tiles.core_size_tokens,
		context_halo_tokens=config.tiles.context_halo_tokens,
		class_ids=plan.class_ids,
	)
	if not manifests['train'].tiles or not manifests['validation'].tiles:
		raise ValueError('train and validation must each contain at least one tile')
	class_counts = tuple(
		sum(
			tile.per_class_supervised_counts[str(class_id)]
			for tile in manifests['train'].tiles
		)
		for class_id in plan.class_ids
	)
	class_weights = balanced_class_weights_from_counts(class_counts)
	manifest_hashes = {
		split: manifest.identity_sha256 for split, manifest in manifests.items()
	}
	resolved_config = config.to_dict()

	decoder = VoxelDecoder3D(
		embedding_dim=config.decoder.embedding_dim,
		class_count=config.decoder.class_count,
		hidden_channels=config.decoder.hidden_channels,
		upsample_factors=config.decoder.upsample_factors,
		patch_size_xyz=plan.patch_size_xyz,
	).to(run_device)
	optimizer = torch.optim.AdamW(
		decoder.parameters(),
		lr=config.train.learning_rate,
		weight_decay=config.train.weight_decay,
	)
	scaler = _make_scaler(run_device, amp=config.train.amp)

	history: list[dict[str, object]] = []
	best_state: Mapping[str, object] | None = None
	start_epoch = 0
	start_batch = 0
	global_step = 0
	partial_accumulator: _TrainAccumulator | None = None
	if resume_path is not None:
		payload = load_voxel_decoder_checkpoint(resume_path, map_location=run_device)
		if payload['checkpoint_kind'] == 'completed':
			raise ValueError('cannot resume a completed voxel decoder run')
		validate_resume_identity(
			payload,
			resolved_config=resolved_config,
			class_weights=class_weights,
			artifact_identities=identities,
			tile_manifest_hashes=manifest_hashes,
		)
		restore_voxel_decoder_checkpoint(
			payload, model=decoder, optimizer=optimizer, amp_scaler=scaler
		)
		history = [dict(item) for item in payload['training_history']]
		best_state = payload['best_selection_state']
		global_step = int(payload['global_step'])
		if payload['checkpoint_kind'] == 'step':
			start_epoch = int(payload['epoch'])
			batch_index = payload.get('batch_index')
			if not isinstance(batch_index, int):
				raise TypeError('step checkpoint batch_index must be an integer')
			start_batch = batch_index + 1
			partial_accumulator = _TrainAccumulator.from_checkpoint(
				payload['current_metrics'], len(plan.class_ids)
			)
		else:
			start_epoch = int(payload['epoch']) + 1

	output_dir.mkdir(parents=True, exist_ok=True)
	_snapshot_run(
		output_dir, resolved_config, plan, manifests, resume=resume_path is not None
	)

	train_dataset = _dataset(plan, manifests['train'])
	validation_dataset = _dataset(plan, manifests['validation'])
	amp_enabled = config.train.amp
	for epoch in range(start_epoch, config.train.epochs):
		train_loader = build_f3_voxel_decoder_dataloader(
			train_dataset,
			batch_size=config.train.batch_size,
			shuffle=True,
			seed=config.train.seed + epoch,
			num_workers=config.train.num_workers,
		)
		accumulator = (
			partial_accumulator
			if epoch == start_epoch and partial_accumulator is not None
			else _TrainAccumulator(len(plan.class_ids))
		)
		processed_all_batches = True
		for batch_index, batch in enumerate(train_loader):
			if epoch == start_epoch and batch_index < start_batch:
				continue
			if max_steps is not None and global_step >= max_steps:
				_write_history(output_dir / HISTORY_NAME, history)
				return _result(output_dir, global_step, completed=False)
			batch_metrics = train_voxel_decoder_one_epoch(
				decoder=decoder,
				dataloader=[batch],  # type: ignore[arg-type]
				optimizer=optimizer,
				class_weights=class_weights,
				class_ids=plan.class_ids,
				device=run_device,
				amp_enabled=amp_enabled,
				scaler=scaler,
				grad_clip_norm=config.train.gradient_clip_norm,
			)
			accumulator.add(batch_metrics)
			global_step += 1
			if max_steps is not None and global_step >= max_steps:
				processed_all_batches = batch_index + 1 == len(train_loader)
				if not processed_all_batches:
					save_voxel_decoder_checkpoint(
						output_dir / LATEST_NAME,
						model=decoder,
						optimizer=optimizer,
						epoch=epoch,
						global_step=global_step,
						resolved_config=resolved_config,
						class_weights=class_weights,
						artifact_identities=identities,
						tile_manifest_hashes=manifest_hashes,
						best_selection_state=best_state,
						training_history=history,
						current_metrics=accumulator.checkpoint_payload(),
						checkpoint_kind='step',
						amp_scaler=scaler,
						batch_index=batch_index,
					)
					_write_history(output_dir / HISTORY_NAME, history)
					return _result(output_dir, global_step, completed=False)
				break
		if not processed_all_batches:
			break
		train_metrics = accumulator.metrics(plan.class_ids)
		validation_loader = build_f3_voxel_decoder_dataloader(
			validation_dataset,
			batch_size=config.train.batch_size,
			shuffle=False,
			seed=config.train.seed,
			num_workers=config.train.num_workers,
		)
		validation_metrics = validate_voxel_decoder_one_epoch(
			decoder=decoder,
			dataloader=validation_loader,
			class_weights=class_weights,
			class_ids=plan.class_ids,
			device=run_device,
			amp_enabled=amp_enabled,
			tile_manifest=manifests['validation'],
		)
		row = _history_row(epoch, global_step, train_metrics, validation_metrics)
		history.append(row)
		improved = best_state_is_improved(validation_metrics, best_state)
		if improved:
			best_state = make_best_selection_state(
				epoch=epoch, validation_metrics=validation_metrics
			)
		kind = 'completed' if epoch + 1 == config.train.epochs else 'epoch'
		checkpoint_arguments: dict[str, object] = {
			'model': decoder,
			'optimizer': optimizer,
			'epoch': epoch,
			'global_step': global_step,
			'resolved_config': resolved_config,
			'class_weights': class_weights,
			'artifact_identities': identities,
			'tile_manifest_hashes': manifest_hashes,
			'best_selection_state': best_state,
			'training_history': history,
			'current_metrics': {
				'train': _json_safe(train_metrics),
				'validation': _json_safe(validation_metrics),
			},
			'checkpoint_kind': kind,
			'amp_scaler': scaler,
		}
		save_voxel_decoder_checkpoint(
			output_dir / LATEST_NAME,
			**checkpoint_arguments,  # type: ignore[arg-type]
		)
		if improved:
			save_voxel_decoder_checkpoint(
				output_dir / BEST_NAME,
				**checkpoint_arguments,  # type: ignore[arg-type]
			)
		_write_history(output_dir / HISTORY_NAME, history)
		if max_steps is not None and global_step >= max_steps:
			return _result(output_dir, global_step, completed=kind == 'completed')
		start_batch = 0
		partial_accumulator = None
	return _result(output_dir, global_step, completed=True)


def _dataset(
	plan: VoxelDecoderInputPlan, manifest: VoxelTileManifest
) -> F3VoxelDecoderDataset:
	return F3VoxelDecoderDataset(
		plan.embeddings,
		plan.valid_tokens,
		plan.embedding_metadata,
		plan.label_volume,
		plan.split_grid,
		manifest,
		supervision_metadata_path=plan.voxel_metadata,
	)


class _TrainAccumulator:
	def __init__(self, class_count: int) -> None:
		self.confusion = np.zeros((class_count, class_count), dtype=np.int64)
		self.weighted_ce_sum = 0.0
		self.unweighted_ce_sum = 0.0
		self.weight_sum = 0.0
		self.voxel_count = 0

	def add(self, metrics: Mapping[str, object]) -> None:
		confusion = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
		if confusion.shape != self.confusion.shape:
			raise ValueError('train metric confusion matrix shape changed')
		weight_sum = float(metrics['class_weight_sum'])
		voxel_count = int(metrics['supervised_voxel_count'])
		self.confusion += confusion
		self.weighted_ce_sum += float(metrics['loss']) * weight_sum
		self.unweighted_ce_sum += (
			float(metrics['unweighted_cross_entropy']) * voxel_count
		)
		self.weight_sum += weight_sum
		self.voxel_count += voxel_count

	def metrics(self, class_ids: Sequence[int]) -> dict[str, object]:
		classes = tuple(
			F3ClassInfo(int(class_id), str(class_id), (0, 0, 0))
			for class_id in class_ids
		)
		metrics = lithology_metrics_from_confusion_matrix(self.confusion, classes)
		metrics.update(
			{
				'loss': self.weighted_ce_sum / self.weight_sum,
				'weighted_cross_entropy': self.weighted_ce_sum / self.weight_sum,
				'unweighted_cross_entropy': self.unweighted_ce_sum / self.voxel_count,
				'class_weight_sum': self.weight_sum,
				'supervised_voxel_count': self.voxel_count,
			}
		)
		return metrics

	def checkpoint_payload(self) -> dict[str, object]:
		return {
			'train_accumulator': {
				'confusion_matrix': self.confusion.tolist(),
				'weighted_ce_sum': self.weighted_ce_sum,
				'unweighted_ce_sum': self.unweighted_ce_sum,
				'class_weight_sum': self.weight_sum,
				'supervised_voxel_count': self.voxel_count,
			}
		}

	@classmethod
	def from_checkpoint(cls, payload: object, class_count: int) -> _TrainAccumulator:
		if not isinstance(payload, Mapping):
			raise TypeError('step checkpoint current_metrics must be a mapping')
		value = payload.get('train_accumulator')
		if not isinstance(value, Mapping):
			raise TypeError('step checkpoint train_accumulator must be a mapping')
		result = cls(class_count)
		confusion = np.asarray(value.get('confusion_matrix'), dtype=np.int64)
		if confusion.shape != result.confusion.shape:
			raise ValueError('step checkpoint train confusion matrix shape mismatch')
		result.confusion = confusion
		result.weighted_ce_sum = float(value['weighted_ce_sum'])
		result.unweighted_ce_sum = float(value['unweighted_ce_sum'])
		result.weight_sum = float(value['class_weight_sum'])
		result.voxel_count = int(value['supervised_voxel_count'])
		return result


def _artifact_identities(plan: VoxelDecoderInputPlan) -> dict[str, object]:
	return {
		'name': 'f3_voxel_decoder_sources',
		'embeddings': _identity(plan.embeddings),
		'embedding_metadata': _identity(plan.embedding_metadata),
		'valid_tokens': _identity(plan.valid_tokens),
		'voxel_dataset_metadata': _identity(plan.voxel_metadata),
		'voxel_split_grid': _identity(plan.split_grid),
		'label_volume': _identity(plan.label_volume),
	}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _validate_source_provenance(
	config: F3LithologyVoxelDecoderConfig,
	*,
	embedding_payload: Mapping[str, object],
) -> None:
	expected_root = (
		config.artifact_root
		/ 'embeddings'
		/ 'f3'
		/ config.dataset['version']
	)
	try:
		relative = config.embeddings_input_dir.resolve(strict=False).relative_to(
			expected_root.resolve(strict=False)
		)
	except ValueError as error:
		raise ValueError(
			'embeddings.input_dir must identify the configured dataset and model'
		) from error
	if len(relative.parts) < 2 or relative.parts[0] != config.model['tag']:
		raise ValueError(
			'model.tag does not match embeddings.input_dir model tag; '
			f'config={config.model["tag"]!r}'
		)
	checkpoint_value = embedding_payload.get('checkpoint_path')
	if not isinstance(checkpoint_value, str) or not checkpoint_value:
		raise ValueError('embedding metadata checkpoint_path is required')
	if config.model['tag'] not in Path(checkpoint_value).parts:
		raise ValueError(
			'model.tag does not match embedding metadata checkpoint_path; '
			f'config={config.model["tag"]!r}'
		)


def _validate_inspected_arrays(  # noqa: PLR0913
	*,
	embeddings: Path,
	valid_tokens: Path,
	label_volume: Path,
	split_grid: Path,
	embedding_payload: Mapping[str, object],
	voxel_payload: Mapping[str, object],
	geometry: tuple[
		tuple[int, int, int],
		tuple[int, int, int],
		tuple[int, int, int],
	],
) -> None:
	"""Validate array geometry and canonical identities without copying arrays."""
	embedding_array = np.load(embeddings, mmap_mode='r', allow_pickle=False)
	valid_array = np.load(valid_tokens, mmap_mode='r', allow_pickle=False)
	label_array = np.load(label_volume, mmap_mode='r', allow_pickle=False)
	split_array = np.load(split_grid, mmap_mode='r', allow_pickle=False)
	patch_size, token_shape, volume_shape = geometry
	if embedding_array.ndim != 4 or not np.issubdtype(
		embedding_array.dtype, np.floating
	):
		raise TypeError('embeddings must be floating [TX,TY,TZ,D]')
	expected_embedding_shape = (
		*token_shape,
		_embedding_dim(embedding_payload),
	)
	if tuple(embedding_array.shape) != expected_embedding_shape:
		raise ValueError('embedding array shape does not match embedding metadata')
	if valid_array.dtype != np.bool_ or tuple(valid_array.shape) != token_shape:
		raise TypeError(
			'valid_tokens must be bool with the metadata token-grid shape'
		)
	if (
		label_array.ndim != 3
		or not np.issubdtype(label_array.dtype, np.integer)
		or label_array.dtype == np.bool_
	):
		raise TypeError('label_volume must be a 3D integer array')
	if tuple(label_array.shape) != volume_shape:
		raise ValueError('label_volume shape does not match embedding metadata')
	if (
		split_array.ndim != 3
		or not np.issubdtype(split_array.dtype, np.integer)
		or split_array.dtype == np.bool_
	):
		raise TypeError('supervision_split_grid must be a 3D integer array')
	if split_array.shape != label_array.shape:
		raise ValueError('supervision_split_grid shape does not match label_volume')
	expected_tokens = tuple(
		(size + step - 1) // step
		for size, step in zip(volume_shape, patch_size, strict=True)
	)
	if expected_tokens != token_shape:
		raise ValueError('embedding token grid is inconsistent with volume geometry')
	validate_encoder_pairing(
		candidate_metadata=embedding_payload,
		reference_metadata=voxel_payload,
		candidate_valid_tokens_path=valid_tokens,
		candidate_embedding_shape=embedding_array.shape,
	)
	declared_label = _required_metadata_mapping(voxel_payload, 'label_volume')
	if declared_label.get('sha256') != file_sha256(label_volume):
		raise ValueError(
			'voxel dataset label_volume hash does not match selected label-volume '
			'artifact'
		)


def _required_metadata_mapping(
	payload: Mapping[str, object], key: str
) -> Mapping[str, object]:
	value = payload.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'voxel dataset metadata {key} must be a mapping')
	return value


def _snapshot_run(
	output_dir: Path,
	resolved_config: Mapping[str, object],
	plan: VoxelDecoderInputPlan,
	manifests: Mapping[str, VoxelTileManifest],
	*,
	resume: bool,
) -> None:
	if not resume:
		_write_json(output_dir / 'resolved_config.json', resolved_config)
		_write_json(
			output_dir / 'run_metadata.json',
			{
				'created_at_utc': datetime.now(timezone.utc).isoformat(),
				'git_commit': _git_commit(),
				'package_version': getattr(seis_ssl_cluster, '__version__', None),
				'source_embedding_metadata': str(plan.embedding_metadata),
				'source_valid_tokens': str(plan.valid_tokens),
				'voxel_dataset_metadata': str(plan.voxel_metadata),
			},
		)
	for split, manifest in manifests.items():
		path = output_dir / f'{split}_tile_manifest.json'
		if not resume:
			write_voxel_tile_manifest(path, manifest)


def _history_row(
	epoch: int,
	global_step: int,
	train: Mapping[str, object],
	validation: Mapping[str, object],
) -> dict[str, object]:
	row: dict[str, object] = {'epoch': epoch, 'global_step': global_step}
	for prefix, metrics in (('train', train), ('validation', validation)):
		for key in (
			'loss',
			'weighted_cross_entropy',
			'unweighted_cross_entropy',
			'accuracy',
			'balanced_accuracy',
			'macro_f1',
			'weighted_f1',
			'mean_iou',
			'supervised_voxel_count',
		):
			row[f'{prefix}_{key}'] = metrics[key]
	return row


def _write_history(path: Path, history: Sequence[Mapping[str, object]]) -> None:
	fieldnames = (
		list(history[0].keys())
		if history
		else [
			'epoch',
			'global_step',
			'train_loss',
			'validation_macro_f1',
			'validation_mean_iou',
		]
	)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(history)


def _validate_output_collision(output_dir: Path, resume: Path | None) -> None:
	if resume is None:
		if output_dir.exists() and any(output_dir.iterdir()):
			raise FileExistsError(f'voxel decoder output is non-empty: {output_dir}')
		return
	if not resume.is_file():
		raise FileNotFoundError(f'resume checkpoint does not exist: {resume}')
	if resume.name != LATEST_NAME:
		raise ValueError(f'resume checkpoint must be {LATEST_NAME}')
	if resume.parent.resolve() != output_dir.resolve():
		raise ValueError(f'resume checkpoint must be outputs.output_dir/{LATEST_NAME}')


def _validate_input_files(plan: VoxelDecoderInputPlan) -> None:
	for path in (
		plan.embeddings,
		plan.valid_tokens,
		plan.embedding_metadata,
		plan.voxel_metadata,
		plan.split_grid,
		plan.label_volume,
	):
		if not path.is_file():
			raise FileNotFoundError(f'missing voxel decoder input: {path}')


def _resolve_device(value: str | torch.device) -> torch.device:
	if isinstance(value, torch.device):
		return value
	if value not in {'auto', 'cpu', 'cuda'}:
		raise ValueError('device must be auto, cpu, or cuda')
	if value == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if value == 'cuda' and not torch.cuda.is_available():
		raise RuntimeError('CUDA was requested but is not available')
	return torch.device(value)


def _make_scaler(device: torch.device, *, amp: bool) -> torch.amp.GradScaler | None:
	if not amp or device.type != 'cuda':
		return None
	return torch.amp.GradScaler('cuda', enabled=True)


def _seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)  # noqa: NPY002
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _metadata_geometry(
	metadata: Mapping[str, object],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
	return (
		_positive_triplet(metadata.get('patch_size'), 'patch_size'),
		_positive_triplet(metadata.get('token_grid_shape'), 'token_grid_shape'),
		_positive_triplet(metadata.get('volume_shape_xyz'), 'volume_shape_xyz'),
	)


def _embedding_dim(metadata: Mapping[str, object]) -> int:
	value = metadata.get('embedding_dim')
	geometry = metadata.get('model_geometry')
	if value is None and isinstance(geometry, Mapping):
		value = geometry.get('encoder_dim')
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError(
			'embedding metadata must contain a positive embedding dimension'
		)
	return value


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'embedding metadata {label} must be a positive integer triple')
	items = tuple(value)
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item <= 0
		for item in items
	):
		raise ValueError(f'embedding metadata {label} must contain positive integers')
	return (items[0], items[1], items[2])


def _read_json_object(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as file_obj:
		value = json.load(file_obj)
	if not isinstance(value, Mapping):
		raise TypeError(f'JSON artifact must contain an object: {path}')
	return value


def _write_json(path: Path, value: object) -> None:
	path.write_text(
		json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _json_safe(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, list | tuple):
		return [_json_safe(item) for item in value]
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, Path):
		return str(value)
	return value


def _git_commit() -> str | None:
	git = shutil.which('git')
	if git is None:
		return None
	result = subprocess.run(  # noqa: S603
		[git, 'rev-parse', 'HEAD'],
		capture_output=True,
		text=True,
		check=False,
	)
	return result.stdout.strip() if result.returncode == 0 else None


def _result(
	output_dir: Path, global_step: int, *, completed: bool
) -> VoxelDecoderRunResult:
	return VoxelDecoderRunResult(
		output_dir=output_dir,
		latest_checkpoint=output_dir / LATEST_NAME,
		best_checkpoint=output_dir / BEST_NAME,
		history_csv=output_dir / HISTORY_NAME,
		global_step=global_step,
		completed=completed,
	)


__all__ = [
	'VoxelDecoderInputPlan',
	'VoxelDecoderRunResult',
	'inspect_f3_lithology_voxel_decoder',
	'run_f3_lithology_voxel_decoder',
]
