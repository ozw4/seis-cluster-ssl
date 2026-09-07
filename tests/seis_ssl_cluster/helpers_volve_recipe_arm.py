'''Shared synthetic artifacts for Volve horizon recipe-arm tests.'''

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.volve.horizon_five_way_config import (
	LOCAL_BARLOW_TWINS_METHOD,
)
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	RECIPE_ARM_BASELINE_MODEL_ID,
	VolveHorizonRecipeArmConfig,
	volve_horizon_recipe_arm_config_from_mapping,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

ARM_ID = 'rot90_asym_g060_3ep'
ARM_EPOCHS = 3
ARM_SAMPLES_PER_EPOCH = 10_000
ARM_BATCH_SIZE = 16
ARM_GLOBAL_STEPS = ARM_EPOCHS * (ARM_SAMPLES_PER_EPOCH // ARM_BATCH_SIZE)
ARM_AUGMENTATIONS: dict[str, object] = {
	'policy': 'xy_rot90_asymmetric_noise_v1',
	'gaussian_noise_std': 0.6,
}
ARM_POSITIVE_WINDOW = [2, 2, 1]
_ENCODER_GEOMETRY: dict[str, object] = {
	'in_channels': 1,
	'patch_size': [8, 8, 8],
	'encoder_dim': 384,
	'encoder_depth': 8,
	'encoder_heads': 6,
}


def recipe_arm_config_mapping(
	tmp_path: Path,
	*,
	arm_id: str = ARM_ID,
) -> dict[str, object]:
	'''Build a portable recipe-arm config rooted below a pytest directory.'''
	artifact_root = (tmp_path / 'artifacts').resolve()
	return {
		'benchmark_id': 'local_bt_recipe_arm_v1',
		'paths': {
			'artifact_root': str(artifact_root),
			'volve_root': str((tmp_path / 'public').resolve()),
		},
		'dataset': {'survey_id': 'volve_st10010'},
		'inputs': {
			'canonical_input_metadata': str(
				artifact_root / 'canonical_input_metadata.json'
			)
		},
		'arm': {
			'arm_id': arm_id,
			'checkpoint': str(artifact_root / 'checkpoints' / arm_id / 'latest.pt'),
			'embeddings_dir': str(artifact_root / 'embeddings' / arm_id),
			'recipe': {
				'method': LOCAL_BARLOW_TWINS_METHOD,
				'local_pairs_per_crop': 128,
				'positive_window_tokens': list(ARM_POSITIVE_WINDOW),
				'augmentations': dict(ARM_AUGMENTATIONS),
				'epochs': ARM_EPOCHS,
				'samples_per_epoch': ARM_SAMPLES_PER_EPOCH,
				'batch_size': ARM_BATCH_SIZE,
				'learning_rate': 1.0e-4,
				'weight_decay': 0.05,
				'seed': 42,
			},
		},
		'baseline': {
			'checkpoint': str(artifact_root / 'checkpoints' / 'random' / 'latest.pt'),
			'embeddings_dir': str(artifact_root / 'embeddings' / 'random'),
		},
		'outputs': {
			'runs_root': str(artifact_root / 'runs'),
			'summary_root': str(artifact_root / 'summary'),
		},
		'decoder': {
			'embedding_dim': 384,
			'class_count': 5,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2]] * 3,
			'upsample_mode': 'nearest',
			'normalization': 'voxelwise_layer_norm',
		},
		'tiles': {
			'patch_size': [8, 8, 8],
			'core_size_tokens': [8, 8, 27],
			'context_halo_tokens': [1, 1, 0],
			'window_start': 552,
			'window_stop': 768,
			'min_token_valid_fraction': 1.0,
		},
		'train': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 1.0e-3,
			'weight_decay': 1.0e-4,
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'amp': True,
			'gradient_clip_norm': 1.0,
		},
	}


def arm_checkpoint_payload() -> dict[str, object]:
	'''Return a valid single-stage Local Barlow Twins checkpoint payload.'''
	return {
		'config': {
			'stage': 'barlow_twins_training',
			'model': deepcopy(_ENCODER_GEOMETRY),
			'augmentations': dict(ARM_AUGMENTATIONS),
			'barlow_twins': {
				'method': LOCAL_BARLOW_TWINS_METHOD,
				'local_pairs_per_crop': 128,
				'positive_window_tokens': list(ARM_POSITIVE_WINDOW),
			},
			'train': {
				'epochs': ARM_EPOCHS,
				'samples_per_epoch': ARM_SAMPLES_PER_EPOCH,
				'batch_size': ARM_BATCH_SIZE,
				'lr': 1.0e-4,
				'weight_decay': 0.05,
				'seed': 42,
			},
		},
		'epoch': ARM_EPOCHS,
		'global_step': ARM_GLOBAL_STEPS,
		'resume_count': 0,
		'pretraining_method': LOCAL_BARLOW_TWINS_METHOD,
		'training_state': {
			'stage': 'barlow_twins_training',
			'resume_boundary': 'epoch',
			'dataset_epoch': ARM_EPOCHS - 1,
			'completed_epoch': True,
		},
	}


def baseline_checkpoint_payload() -> dict[str, object]:
	'''Return a valid canonical random-encoder checkpoint payload.'''
	return {
		'config': {'model': deepcopy(_ENCODER_GEOMETRY)},
		'epoch': 0,
		'global_step': 0,
		'metadata': {
			'random_encoder_baseline': True,
			'pretrained_weights_loaded': False,
			'seed': 42,
		},
		'training_state': {
			'stage': 'create_random_mae_checkpoint',
			'checkpoint_kind': 'random_init',
		},
	}


def write_recipe_arm_universe(
	tmp_path: Path,
	*,
	embeddings: bool,
	arm_id: str = ARM_ID,
	arm_payload: dict[str, object] | None = None,
	baseline_payload: dict[str, object] | None = None,
) -> dict[str, Any]:
	'''Write one arm checkpoint, the baseline, and optionally both embeddings.'''
	raw = recipe_arm_config_mapping(tmp_path, arm_id=arm_id)
	arm = cast('dict[str, str]', raw['arm'])
	baseline = cast('dict[str, str]', raw['baseline'])
	arm_checkpoint = Path(arm['checkpoint'])
	baseline_checkpoint = Path(baseline['checkpoint'])
	save_checkpoint(
		arm_checkpoint,
		arm_checkpoint_payload() if arm_payload is None else arm_payload,
	)
	save_checkpoint(
		baseline_checkpoint,
		baseline_checkpoint_payload()
		if baseline_payload is None
		else baseline_payload,
	)
	config = volve_horizon_recipe_arm_config_from_mapping(raw)
	universe: dict[str, Any] = {
		'raw': raw,
		'config': config,
		'checkpoints': {
			arm_id: arm_checkpoint,
			RECIPE_ARM_BASELINE_MODEL_ID: baseline_checkpoint,
		},
	}
	if embeddings:
		universe['embedding_metadata'] = _write_embedding_sources(config, tmp_path)
	return universe


def _write_embedding_sources(
	config: VolveHorizonRecipeArmConfig,
	tmp_path: Path,
) -> dict[str, Path]:
	volume_shape = (16, 16, 800)
	token_grid = (2, 2, 100)
	public = (tmp_path / 'public').resolve()
	public.mkdir(parents=True, exist_ok=True)
	valid_mask_path = public / 'valid_trace_mask.npy'
	np.save(valid_mask_path, np.ones(volume_shape[:2], dtype=np.bool_))
	amplitude_path = public / 'amplitude.npy'
	np.save(amplitude_path, np.zeros((1,), dtype=np.float32))
	normalization_path = (tmp_path / 'artifacts' / 'normalization.json').resolve()
	write_json(normalization_path, {'mean': 0.0, 'std': 1.0})
	canonical_identity = {
		'survey_id': 'volve_st10010',
		'shape_xyz': list(volume_shape),
		'canonical_amplitude_sha256': file_sha256(amplitude_path),
		'valid_trace_mask_sha256': file_sha256(valid_mask_path),
		'inline_values_sha256': '1' * 64,
		'crossline_values_sha256': '2' * 64,
		'time_axis_sha256': '3' * 64,
		'canonical_normalization_stats_sha256': file_sha256(normalization_path),
	}
	write_json(
		config.canonical_input_metadata,
		{
			'artifact_type': 'volve_canonical_input_registration',
			'status': 'PASS',
			'scientific_identity': canonical_identity,
			'scientific_identity_sha256': _json_sha256(canonical_identity),
			'provenance': {
				'amplitude': {'path': str(amplitude_path)},
				'public_inputs': {'valid_trace_mask.npy': str(valid_mask_path)},
			},
			'outputs': {'normalization_stats': str(normalization_path)},
		},
	)
	common: dict[str, object] = {
		'survey_id': 'volve_st10010',
		'source_amplitude_path': str(amplitude_path),
		'source_valid_mask_path': str(valid_mask_path),
		'volume_shape_xyz': list(volume_shape),
		'model_geometry': deepcopy(_ENCODER_GEOMETRY),
		'patch_size': [8, 8, 8],
		'token_grid_shape': list(token_grid),
		'window_size': [128, 128, 128],
		'overlap': [64, 64, 64],
		'output_dtype': 'float16',
		'precision': {
			'amp_requested': True,
			'amp_dtype_requested': 'auto',
			'resolved_dtype': 'float16',
			'amp_enabled': True,
		},
		'min_token_valid_fraction': 1.0,
		'normalization_stats_path': str(normalization_path),
		'normalized_clip_abs': 8.0,
		'amplitude_agc': {'enabled': True, 'mode': 'trace_rms_z'},
		'finite_check_mode': 'strict',
		'preprocessing': {'normalized_clip_abs': 8.0},
		'preprocessing_cache': {'requested_mode': 'off', 'effective_mode': 'off'},
		'zero_mask': {'enabled': True},
	}
	valid_tokens = np.ones(token_grid, dtype=np.bool_)
	sources = (
		(config.arm_id, config.arm_checkpoint, config.arm_embeddings_dir, 1.0),
		(
			RECIPE_ARM_BASELINE_MODEL_ID,
			config.baseline_checkpoint,
			config.baseline_embeddings_dir,
			2.0,
		),
	)
	metadata_paths: dict[str, Path] = {}
	for model_id, checkpoint, embeddings_dir, sentinel in sources:
		paths = output_paths(embeddings_dir, config.survey_id)
		paths.embeddings.parent.mkdir(parents=True, exist_ok=True)
		np.save(
			paths.embeddings,
			np.full((*token_grid, 384), sentinel, dtype=np.float16),
		)
		np.save(paths.valid_tokens, valid_tokens)
		is_arm = model_id != RECIPE_ARM_BASELINE_MODEL_ID
		metadata: dict[str, object] = {
			**common,
			'checkpoint_path': str(checkpoint),
			'checkpoint_sha256': file_sha256(checkpoint),
			'pretraining_objective': (
				{
					'method': LOCAL_BARLOW_TWINS_METHOD,
					'local_pairs_per_crop': 128,
				}
				if is_arm
				else {'reconstruction': 'mse'}
			),
		}
		if is_arm:
			metadata['pretraining_method'] = LOCAL_BARLOW_TWINS_METHOD
		write_json(paths.metadata, metadata)
		metadata_paths[model_id] = paths.metadata
	return metadata_paths


def write_recipe_arm_runs(
	config: VolveHorizonRecipeArmConfig,
	*,
	arm_mae: Mapping[tuple[str, str], float],
	baseline_mae: Mapping[tuple[str, str], float],
	omit: Sequence[tuple[str, str, str]] = (),
) -> None:
	"""Write one completed metrics.json per configured cell."""
	skipped = set(omit)
	for model_id in config.model_ids:
		is_arm = model_id != RECIPE_ARM_BASELINE_MODEL_ID
		values = arm_mae if is_arm else baseline_mae
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				if (model_id, layout_id, data_size) in skipped:
					continue
				path = (
					config.runs_root
					/ f'model={model_id}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'metrics.json'
				)
				write_json(
					path,
					{
						'artifact_type': 'volve_frozen_horizon_job_metrics',
						'schema_version': 1,
						'model': model_id,
						'layout_id': layout_id,
						'data_size': data_size,
						'best_epoch': 30,
						'benchmark_identity': _run_identity(
							config,
							model_id=model_id,
							layout_id=layout_id,
							data_size=data_size,
						),
						'validation': {
							'macro_mae_samples': values[layout_id, data_size],
						},
						'test': {
							'primary_common': {
								'macro_mae_samples': values[layout_id, data_size],
								'macro_within_2_samples': 0.25,
							}
						},
					},
				)


def _run_identity(
	config: VolveHorizonRecipeArmConfig,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
) -> dict[str, object]:
	return {
		'benchmark': config.benchmark_id,
		'model': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'canonical_scientific_identity': {'scientific_identity_sha256': '4' * 64},
		'decoder': {
			'architecture': {'embedding_dim': 384},
			'initial_state_sha256': '5' * 64,
			'initialization_seed': 42_000,
		},
		'objective': {'checkpoint_selection': config.checkpoint_selection},
		'optimizer': {'name': 'adamw'},
		'runtime_precision': {'amp_enabled': True},
		'training': {'epochs': 50, 'seed': 42_000},
		'tiles': {'counts': {'train': 18, 'validation': 18, 'test': 84}},
		'horizon_split_plan': {
			'scientific_identity_sha256': f'{layout_id}:{data_size}',
		},
		'effective_model_valid_observation_counts': {'bcu': 156_583},
		'native_horizon_observation_counts': {'bcu': 156_583},
		'embedding': {'checkpoint_sha256': f'{model_id}-sha'},
	}


def save_checkpoint(path: Path, payload: object) -> None:
	'''Write one synthetic checkpoint payload.'''
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, object]:
	'''Read one synthetic checkpoint payload back.'''
	return cast(
		'dict[str, object]',
		torch.load(path, map_location='cpu', weights_only=False),
	)


def write_json(path: Path, payload: object) -> None:
	'''Write one synthetic JSON artifact.'''
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')


def _json_sha256(payload: object) -> str:
	return hashlib.sha256(
		json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


__all__ = [
	'ARM_AUGMENTATIONS',
	'ARM_BATCH_SIZE',
	'ARM_EPOCHS',
	'ARM_GLOBAL_STEPS',
	'ARM_ID',
	'ARM_POSITIVE_WINDOW',
	'ARM_SAMPLES_PER_EPOCH',
	'arm_checkpoint_payload',
	'baseline_checkpoint_payload',
	'load_checkpoint',
	'recipe_arm_config_mapping',
	'save_checkpoint',
	'write_json',
	'write_recipe_arm_runs',
	'write_recipe_arm_universe',
]
