"""Thin one-job runner for the F3 lithology five-way benchmark matrix."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	f3_lithology_voxel_decoder_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	f3_lithology_voxel_evaluation_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	f3_lithology_voxel_inference_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	FIXED_DECODER_CONTRACT,
	LAYOUT_IDS,
)
from seis_ssl_cluster.f3.lithology.five_way_sources import (
	audit_f3_lithology_five_way_sources,
)
from seis_ssl_cluster.f3.lithology.voxel_decoder_inference import (
	predict_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	evaluate_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	LAYOUT_METADATA_NAME,
	validate_f3_lithology_voxel_section_layout_condition,
)
from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
	stable_model_state_sha256,
)
from seis_ssl_cluster.training.voxel_decoder.runner import (
	run_f3_lithology_voxel_decoder,
)

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.config.f3_lithology_five_way import (
		F3FiveWayConfig,
		F3FiveWayModelSource,
	)
	from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
		F3LithologyVoxelDecoderConfig,
	)

FIVE_WAY_TILE_SETTINGS: Mapping[str, tuple[int, int, int]] = {
	'core_size_tokens': (8, 8, 8),
	'context_halo_tokens': (1, 1, 1),
}
FIVE_WAY_EVALUATION_POLICY: Mapping[str, object] = {
	'monitored_class_ids': (3, 5),
	'boundary_tolerances': (1, 2, 4, 8),
	'boundary_region_radii': (1, 2, 4, 8),
	'chunk_size_x': 8,
}
DECODER_DIR_NAME = 'decoder'
PREDICTION_DIR_NAME = 'prediction'
EVALUATION_DIR_NAME = 'evaluation'
METRICS_NAME = 'metrics.json'
BEST_CHECKPOINT_NAME = 'best.pt'
LATEST_CHECKPOINT_NAME = 'latest.pt'
RESOLVED_CONFIG_NAME = 'resolved_config.json'
PREDICTION_METADATA_NAME = 'prediction_metadata.json'


@dataclass(frozen=True)
class F3FiveWayJob:
	"""One resolved (model, layout, size) cell of the 75-job matrix."""

	config: F3FiveWayConfig
	model: F3FiveWayModelSource
	layout_id: str
	data_size: str
	condition_dir: Path
	output_dir: Path

	@property
	def decoder_dir(self) -> Path:
		"""Return the decoder training output directory."""
		return self.output_dir / DECODER_DIR_NAME

	@property
	def prediction_dir(self) -> Path:
		"""Return the full-volume inference output directory."""
		return self.output_dir / PREDICTION_DIR_NAME

	@property
	def evaluation_dir(self) -> Path:
		"""Return the numeric evaluation output directory."""
		return self.output_dir / EVALUATION_DIR_NAME

	@property
	def metrics_path(self) -> Path:
		"""Return the metrics file marking this job as completed."""
		return self.evaluation_dir / METRICS_NAME


def plan_f3_lithology_five_way_jobs(
	config: F3FiveWayConfig,
) -> tuple[tuple[str, str, str], ...]:
	"""Enumerate the canonical 75-job (model, layout, size) matrix."""
	return tuple(
		(model_id, layout_id, data_size)
		for model_id in config.model_ids
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZES
	)


def resolve_f3_lithology_five_way_job(
	config: F3FiveWayConfig,
	*,
	model: str,
	layout: str,
	size: str,
) -> F3FiveWayJob:
	"""Resolve one job cell, rejecting anything outside the fixed matrix."""
	source = config.model_by_id(model)
	if layout not in LAYOUT_IDS:
		raise ValueError(
			f'unknown layout: {layout!r}; expected one of {list(LAYOUT_IDS)!r}'
		)
	if size not in DATA_SIZES:
		raise ValueError(
			f'unknown data size: {size!r}; expected one of {list(DATA_SIZES)!r}'
		)
	condition_dir = (
		config.section_layout_dataset_root
		/ 'datasets'
		/ f'layout={layout}'
		/ f'size={size}'
		/ 'voxel_supervision'
	)
	output_dir = (
		config.runs_root / f'model={model}' / f'layout={layout}' / f'size={size}'
	)
	return F3FiveWayJob(
		config=config,
		model=source,
		layout_id=layout,
		data_size=size,
		condition_dir=condition_dir,
		output_dir=output_dir,
	)


def inspect_f3_lithology_five_way_job(  # noqa: C901
	job: F3FiveWayJob,
) -> dict[str, object]:
	"""Resolve one job dry-run summary from small metadata files only."""
	decoder_config = f3_lithology_voxel_decoder_config_from_mapping(
		_decoder_config_mapping(job)
	)
	metadata_path = job.condition_dir / LAYOUT_METADATA_NAME
	if not metadata_path.is_file():
		raise FileNotFoundError(
			f'missing section-layout condition metadata: {metadata_path}'
		)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	if not isinstance(metadata, Mapping):
		raise TypeError(f'{metadata_path} must contain a JSON object')
	identity = metadata.get('identity')
	if not isinstance(identity, Mapping):
		raise TypeError(f'{metadata_path} identity must be a mapping')
	for key in ('layout_id', 'data_size'):
		expected = job.layout_id if key == 'layout_id' else job.data_size
		if identity.get(key) != expected:
			raise ValueError(
				f'condition identity {key} must equal {expected!r}; '
				f'got {identity.get(key)!r}'
			)
	active_lines = metadata.get('active_lines')
	if not isinstance(active_lines, Mapping):
		raise TypeError(f'{metadata_path} active_lines must be a mapping')
	for key in ('inline', 'crossline'):
		lines = active_lines.get(key)
		if not isinstance(lines, list) or not lines:
			raise ValueError(
				f'{metadata_path} active_lines.{key} must be a non-empty list'
			)
	per_line_contributions = identity.get('per_line_contributions')
	if not isinstance(per_line_contributions, Mapping):
		raise TypeError(
			f'{metadata_path} identity.per_line_contributions must be a mapping'
		)
	expected_lines = {
		f'{slice_type}:{line}'
		for slice_type in ('inline', 'crossline')
		for line in active_lines[slice_type]
	}
	if set(per_line_contributions) != expected_lines:
		raise ValueError(
			f'{metadata_path} identity.per_line_contributions must cover exactly '
			f'{sorted(expected_lines)!r}'
		)
	return {
		'model_id': job.model.model_id,
		'checkpoint': str(job.model.checkpoint),
		'embeddings_dir': str(job.model.embeddings_dir),
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'inline_lines': active_lines['inline'],
		'crossline_lines': active_lines['crossline'],
		'per_line_contributions': dict(per_line_contributions),
		'condition_dir': str(job.condition_dir),
		'selected_token_count': identity.get('selected_token_count'),
		'selected_token_identity_sha256': identity.get(
			'selected_token_identity_sha256'
		),
		'train_voxel_count': identity.get('actual_train_voxel_count'),
		'per_class_train_voxel_counts': identity.get(
			'per_class_train_voxel_counts'
		),
		'validation_mask_sha256': identity.get('validation_mask_sha256'),
		'validation_voxel_count': identity.get('validation_voxel_count'),
		'decoder_contract': dict(FIXED_DECODER_CONTRACT),
		'decoder_seed': decoder_config.train.seed,
		'decoder_initial_state_sha256': _decoder_initial_state_sha256(
			decoder_config, identity
		),
		'decoder_dir': str(job.decoder_dir),
		'prediction_dir': str(job.prediction_dir),
		'evaluation_dir': str(job.evaluation_dir),
	}


def run_f3_lithology_five_way_job(
	job: F3FiveWayJob,
	*,
	device: str = 'auto',
	max_steps: int | None = None,
	resume: Path | None = None,
) -> dict[str, object]:
	"""Audit sources, then train, predict, and evaluate one matrix cell.

	Each stage is skipped when its own completed artifact is already present, so
	a job interrupted after training can be restarted without retraining.
	"""
	if job.metrics_path.is_file():
		raise FileExistsError(
			f'job already completed; refusing to overwrite {job.metrics_path}'
		)
	audit_f3_lithology_five_way_sources(job.config)
	validate_f3_lithology_voxel_section_layout_condition(job.condition_dir)
	if resume is not None and resume != job.decoder_dir / LATEST_CHECKPOINT_NAME:
		raise ValueError(
			'resume must be the decoder latest checkpoint of this job: '
			f'{job.decoder_dir / LATEST_CHECKPOINT_NAME}'
		)
	decoder_config = f3_lithology_voxel_decoder_config_from_mapping(
		_decoder_config_mapping(job)
	)
	decoder_completed = _decoder_is_completed(job, decoder_config)
	if decoder_completed:
		if resume is not None:
			raise FileExistsError(
				'decoder training already completed for this job; rerun without '
				f'--resume to continue from {job.decoder_dir}'
			)
	else:
		if resume is None and _decoder_dir_is_occupied(job):
			latest = job.decoder_dir / LATEST_CHECKPOINT_NAME
			recovery = (
				f'resume it with --resume {latest}'
				if latest.is_file()
				else 'no checkpoint was written yet, so it can only be restarted'
			)
			raise FileExistsError(
				f'decoder training is interrupted in {job.decoder_dir}; '
				f'{recovery}, or delete the directory to retrain from scratch'
			)
		training = run_f3_lithology_voxel_decoder(
			decoder_config,
			device=device,
			max_steps=max_steps,
			resume=resume,
		)
		if not training.completed:
			return {
				'completed': False,
				'decoder_dir': str(job.decoder_dir),
				'latest_checkpoint': str(training.latest_checkpoint),
			}
	if not (job.prediction_dir / PREDICTION_METADATA_NAME).is_file():
		inference_config = f3_lithology_voxel_inference_config_from_mapping(
			_inference_config_mapping(job)
		)
		predict_f3_lithology_voxels(inference_config, device=device)
	evaluation_config = f3_lithology_voxel_evaluation_config_from_mapping(
		_evaluation_config_mapping(job)
	)
	evaluate_f3_lithology_voxels(evaluation_config)
	if not job.metrics_path.is_file():
		raise FileNotFoundError(
			f'evaluation did not produce metrics: {job.metrics_path}'
		)
	return {
		'completed': True,
		'reused_decoder': decoder_completed,
		'decoder_dir': str(job.decoder_dir),
		'prediction_dir': str(job.prediction_dir),
		'evaluation_dir': str(job.evaluation_dir),
		'metrics_path': str(job.metrics_path),
	}


def _decoder_dir_is_occupied(job: F3FiveWayJob) -> bool:
	return job.decoder_dir.is_dir() and any(job.decoder_dir.iterdir())


def _decoder_is_completed(
	job: F3FiveWayJob, decoder_config: F3LithologyVoxelDecoderConfig
) -> bool:
	"""Report whether this job's decoder already finished its fixed budget."""
	latest = job.decoder_dir / LATEST_CHECKPOINT_NAME
	best = job.decoder_dir / BEST_CHECKPOINT_NAME
	resolved_config_path = job.decoder_dir / RESOLVED_CONFIG_NAME
	if not (latest.is_file() and best.is_file() and resolved_config_path.is_file()):
		return False
	payload = load_voxel_decoder_checkpoint(latest, map_location='cpu')
	if payload['checkpoint_kind'] != 'completed':
		return False
	recorded = json.loads(resolved_config_path.read_text(encoding='utf-8'))
	if recorded != decoder_config.to_dict():
		raise ValueError(
			f'{resolved_config_path} does not match this job; delete '
			f'{job.decoder_dir} before rerunning'
		)
	return True


def _decoder_config_mapping(job: F3FiveWayJob) -> dict[str, object]:
	contract = FIXED_DECODER_CONTRACT
	return {
		'paths': {
			'artifact_root': str(job.config.artifact_root),
			'f3_root': str(job.config.f3_root),
		},
		'dataset': dict(job.config.dataset),
		'model': {'tag': job.model.model_id, 'freeze_encoder': True},
		'embeddings': {
			'input_dir': str(job.model.embeddings_dir),
			'spec': job.model.embeddings_dir.name,
			'checkpoint_path': str(job.model.checkpoint),
		},
		'voxel_dataset': {'input_dir': str(job.condition_dir)},
		'decoder': {
			'spec': contract['spec'],
			'embedding_dim': contract['embedding_dim'],
			'class_count': contract['class_count'],
			'hidden_channels': list(contract['hidden_channels']),
			'upsample_factors': [
				list(factor) for factor in contract['upsample_factors']
			],
			'upsample_mode': contract['upsample_mode'],
			'normalization': contract['normalization'],
		},
		'tiles': {
			'core_size_tokens': list(FIVE_WAY_TILE_SETTINGS['core_size_tokens']),
			'context_halo_tokens': list(
				FIVE_WAY_TILE_SETTINGS['context_halo_tokens']
			),
		},
		'train': {
			'epochs': contract['epochs'],
			'batch_size': contract['batch_size'],
			'learning_rate': contract['learning_rate'],
			'weight_decay': contract['weight_decay'],
			'class_weight': contract['class_weight'],
			'seed': contract['seed'],
			'num_workers': 0,
			'amp': contract['amp'],
			'gradient_clip_norm': contract['gradient_clip_norm'],
			'sampling_mode': contract['sampling_mode'],
			'steps_per_epoch': contract['steps_per_epoch'],
		},
		'outputs': {'output_dir': str(job.decoder_dir)},
	}


def _inference_config_mapping(job: F3FiveWayJob) -> dict[str, object]:
	return {
		'paths': {
			'artifact_root': str(job.config.artifact_root),
			'f3_root': str(job.config.f3_root),
		},
		'dataset': dict(job.config.dataset),
		'model': {'tag': job.model.model_id, 'freeze_encoder': True},
		'labels': {'class_info': str(job.config.labels['class_info'])},
		'embeddings': {
			'input_dir': str(job.model.embeddings_dir),
			'spec': job.model.embeddings_dir.name,
		},
		'decoder': {'checkpoint': str(job.decoder_dir / BEST_CHECKPOINT_NAME)},
		'tiles': {
			'core_size_tokens': list(FIVE_WAY_TILE_SETTINGS['core_size_tokens']),
			'context_halo_tokens': list(
				FIVE_WAY_TILE_SETTINGS['context_halo_tokens']
			),
		},
		'inference': {
			'write_probabilities': FIXED_DECODER_CONTRACT['write_probabilities'],
			'overwrite': False,
		},
		'outputs': {'output_dir': str(job.prediction_dir)},
	}


def _evaluation_config_mapping(job: F3FiveWayJob) -> dict[str, object]:
	labels = job.config.labels
	return {
		'paths': {
			'artifact_root': str(job.config.artifact_root),
			'f3_root': str(job.config.f3_root),
		},
		'dataset': dict(job.config.dataset),
		'labels': {key: str(path) for key, path in labels.items()},
		'voxel_predictions': {'input_dir': str(job.prediction_dir)},
		'voxel_dataset': {'input_dir': str(job.condition_dir)},
		'evaluation': {
			'monitored_class_ids': list(
				FIVE_WAY_EVALUATION_POLICY['monitored_class_ids']
			),
			'boundary_tolerances': list(
				FIVE_WAY_EVALUATION_POLICY['boundary_tolerances']
			),
			'boundary_region_radii': list(
				FIVE_WAY_EVALUATION_POLICY['boundary_region_radii']
			),
			'chunk_size_x': FIVE_WAY_EVALUATION_POLICY['chunk_size_x'],
		},
		'outputs': {'output_dir': str(job.evaluation_dir), 'overwrite': False},
	}


def _decoder_initial_state_sha256(
	decoder_config: object, identity: Mapping[str, object]
) -> str:
	patch_size = identity.get('patch_size_xyz')
	if (
		not isinstance(patch_size, list)
		or len(patch_size) != 3
		or any(not isinstance(item, int) or item <= 0 for item in patch_size)
	):
		raise ValueError('condition identity patch_size_xyz must be three ints')
	decoder = decoder_config.decoder
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(decoder_config.train.seed)
		model = VoxelDecoder3D(
			spec=decoder.spec,
			embedding_dim=decoder.embedding_dim,
			class_count=decoder.class_count,
			hidden_channels=decoder.hidden_channels,
			upsample_factors=decoder.upsample_factors,
			upsample_mode=decoder.upsample_mode,
			normalization=decoder.normalization,
			patch_size_xyz=tuple(patch_size),
		)
	return stable_model_state_sha256(model)


__all__ = [
	'BEST_CHECKPOINT_NAME',
	'DECODER_DIR_NAME',
	'EVALUATION_DIR_NAME',
	'FIVE_WAY_EVALUATION_POLICY',
	'FIVE_WAY_TILE_SETTINGS',
	'LATEST_CHECKPOINT_NAME',
	'METRICS_NAME',
	'PREDICTION_DIR_NAME',
	'PREDICTION_METADATA_NAME',
	'RESOLVED_CONFIG_NAME',
	'F3FiveWayJob',
	'inspect_f3_lithology_five_way_job',
	'plan_f3_lithology_five_way_jobs',
	'resolve_f3_lithology_five_way_job',
	'run_f3_lithology_five_way_job',
]
