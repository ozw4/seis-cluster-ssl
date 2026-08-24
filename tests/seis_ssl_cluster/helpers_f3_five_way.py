"""Synthetic five-way comparison universes shared by focused tests."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.config.f3_lithology_five_way import (
	EXPECTED_MODEL_IDENTITIES,
	FIVE_WAY_MODEL_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256

if TYPE_CHECKING:
	from collections.abc import Mapping

SURVEY_ID = 'f3_facies_benchmark'
DATASET = {'name': SURVEY_ID, 'version': 'facies_benchmark_v1'}
TOKEN_GRID = (2, 2, 2)
VOLUME_SHAPE = (16, 16, 16)
PATCH_SIZE = (8, 8, 8)
CLASS_IDS = tuple(range(6))


def base_embedding_metadata(checkpoint: Path, sha256: str) -> dict[str, object]:
	return {
		'survey_id': SURVEY_ID,
		'source_amplitude_path': '/data/f3/f3_seismic.npy',
		'normalization_stats_path': '/data/f3/f3_seismic.stats.json',
		'volume_shape_xyz': list(VOLUME_SHAPE),
		'token_grid_shape': list(TOKEN_GRID),
		'patch_size': list(PATCH_SIZE),
		'window_size': [128, 128, 128],
		'overlap': [64, 64, 64],
		'output_dtype': 'float16',
		'min_token_valid_fraction': 0.5,
		'model_geometry': {
			'name': 'amp_mae3d',
			'in_channels': 1,
			'out_channels': 1,
			'patch_size': list(PATCH_SIZE),
			'encoder_dim': 384,
			'encoder_depth': 8,
			'encoder_heads': 6,
			'decoder_dim': 256,
			'decoder_depth': 4,
			'decoder_heads': 4,
		},
		'precision': {
			'amp_enabled': False,
			'amp_requested': False,
			'amp_dtype_requested': 'auto',
			'resolved_dtype': 'float32',
		},
		'preprocessing': {
			'finite_check_mode': 'strict',
			'normalized_clip_abs': 8.0,
			'amplitude_agc': {
				'enabled': True,
				'mode': 'trace_rms_z',
				'window_z': 65,
				'eps': 1.0e-3,
				'clip_abs': 5.0,
			},
		},
		'preprocessing_cache': {
			'schema_version': 1,
			'requested_mode': 'off',
			'effective_mode': 'off',
			'dtype': 'float32',
			'fingerprint': None,
		},
		'amplitude_agc': {
			'enabled': True,
			'mode': 'trace_rms_z',
			'window_z': 65,
			'eps': 1.0e-3,
			'clip_abs': 5.0,
		},
		'normalized_clip_abs': 8.0,
		'finite_check_mode': 'strict',
		'zero_mask': {
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 16,
			'xy_trace_influence_radius': 1,
		},
		'checkpoint_path': str(checkpoint),
		'checkpoint_sha256': sha256,
	}


def mae_objective() -> dict[str, object]:
	return {
		'reconstruction': 'mse',
		'gradient_weight': 0.0,
		'visible_reconstruction_weight': 0.1,
		'target_normalization': {
			'mode': 'patch_zscore',
			'eps': 1.0e-6,
			'min_std': 0.05,
		},
	}


def local_bt_objective() -> dict[str, object]:
	return {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 128,
		'projector_dim': 384,
		'redundancy_weight': 0.005,
		'normalization_eps': 1.0e-4,
	}


def pretext_identity(
	base_objective: str, pseudo_target_dir: str
) -> dict[str, object]:
	return {
		'method': 'strat_hmm_pretext',
		'base_objective': base_objective,
		'head_num_prototypes': 6,
		'unfreeze_top_blocks': 1,
		'distillation_weight': 0.2,
		'pseudo_target_input_dir': pseudo_target_dir,
	}


FIXED_BUDGET_EPOCHS = 25
FIXED_BUDGET_GLOBAL_STEPS = 15_625
STAGE1_EPOCHS = 100
STAGE1_GLOBAL_STEPS = 62_500


def mae_stage1_config() -> dict[str, object]:
	return {
		'stage': 'train_amp_mae',
		'train': {'epochs': STAGE1_EPOCHS},
		'masking': {'spatial_mask_ratio': 0.75},
		'loss': {'reconstruction': 'mse'},
	}


def local_bt_stage1_config() -> dict[str, object]:
	return {
		'stage': 'barlow_twins_training',
		'barlow_twins': {
			'method': 'local_barlow_twins_3d',
			'local_pairs_per_crop': 128,
			'projector_dim': 384,
		},
		'augmentations': {'horizontal_flip_probability': 0.5},
		'train': {'epochs': STAGE1_EPOCHS},
	}


def write_stage1_checkpoint(path: Path, config: dict[str, object]) -> Path:
	"""Write a stage-1 base checkpoint the lineage audit can actually read."""
	path.parent.mkdir(parents=True, exist_ok=True)
	stage = 'train_amp_mae' if config['stage'] == 'train_amp_mae' else (
		'barlow_twins_training'
	)
	torch.save(
		{
			'config': config,
			'epoch': STAGE1_EPOCHS,
			'global_step': STAGE1_GLOBAL_STEPS,
			'training_state': {
				'schema_version': 1,
				'stage': stage,
				'checkpoint_kind': 'epoch',
			},
		},
		path,
	)
	return path


def _continuation(init_checkpoint: Path) -> dict[str, object]:
	return {'init_checkpoint': str(init_checkpoint), 'unfreeze_top_blocks': 1}


def _stratigraphy_config(
	pseudo_target_dir: str, stage1_checkpoint: Path
) -> dict[str, object]:
	return {
		'stage': 'train_strat_hmm_pretext',
		'head': {'num_prototypes': 6, 'projection_dim': 128, 'temperature': 0.1},
		'train': {'epochs': FIXED_BUDGET_EPOCHS, 'batch_size': 16},
		'teacher': {'checkpoint': str(stage1_checkpoint)},
		'student': {
			'init_checkpoint': str(stage1_checkpoint),
			'unfreeze_top_blocks': 1,
		},
		'loss': {
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'entropy_floor': None,
			'distillation_weight': 0.2,
		},
		'pseudo_targets': {
			'k': 6,
			'min_confidence': 0.0,
			'input_dir': pseudo_target_dir,
		},
	}


def checkpoint_payload(
	model_id: str,
	mae_checkpoint: Path,
	*,
	pseudo_target_dirs: Mapping[str, str] | None = None,
	stage1_checkpoints: Mapping[str, Path] | None = None,
) -> dict[str, object]:
	targets = dict(pseudo_target_dirs or {})
	stage1 = dict(stage1_checkpoints or {})
	mae_stage1 = stage1.get('mae', Path('/stage1/mae/full_100ep/latest.pt'))
	bt_stage1 = stage1.get(
		'local_barlow_twins', Path('/stage1/local_bt/full_100ep/latest.pt')
	)
	budget = {
		'epoch': FIXED_BUDGET_EPOCHS,
		'global_step': FIXED_BUDGET_GLOBAL_STEPS,
	}
	mae_config = {
		**mae_stage1_config(),
		'train': {'epochs': FIXED_BUDGET_EPOCHS},
		'continuation': _continuation(mae_stage1),
	}
	local_bt_config = {
		**local_bt_stage1_config(),
		'train': {'epochs': FIXED_BUDGET_EPOCHS},
		'continuation': _continuation(bt_stage1),
	}
	if model_id == 'mae':
		return {
			'config': mae_config,
			'training_state': {
				'schema_version': 2,
				'stage': 'train_amp_mae',
				'checkpoint_kind': 'epoch',
				'resolved_precision': 'float32',
			},
			**budget,
		}
	if model_id == 'mae_hmm_k6':
		return {
			'config': mae_stage1_config(),
			'stratigraphy_config': _stratigraphy_config(
				targets.get(model_id, ''), mae_stage1
			),
			'training_state': {
				'schema_version': 1,
				'stage': 'train_strat_hmm_pretext',
				'checkpoint_kind': 'epoch',
			},
			**budget,
		}
	if model_id == 'local_barlow_twins':
		return {
			'config': local_bt_config,
			'checkpoint_kind': 'epoch',
			'pretraining_method': 'local_barlow_twins_3d',
			'training_state': {
				'schema_version': 1,
				'stage': 'barlow_twins_training',
				'resume_boundary': 'epoch',
				'completed_epoch': True,
			},
			**budget,
		}
	if model_id == 'local_barlow_twins_hmm_k6':
		return {
			'config': local_bt_stage1_config(),
			'stratigraphy_config': _stratigraphy_config(
				targets.get(model_id, ''), bt_stage1
			),
			'training_state': {
				'schema_version': 1,
				'stage': 'train_strat_hmm_pretext',
				'checkpoint_kind': 'epoch',
			},
			**budget,
		}
	return {
		'config': mae_stage1_config(),
		'metadata': {
			'random_encoder_baseline': True,
			'pretrained_weights_loaded': False,
			'seed': 42,
			'reference_checkpoint': str(mae_checkpoint),
			'reference_model_tag': 'mae_fixed_budget',
		},
		'training_state': {
			'schema_version': 1,
			'stage': 'create_random_mae_checkpoint',
			'checkpoint_kind': 'random_init',
			'batch_index': None,
		},
		'epoch': 0,
		'global_step': 0,
	}


def label_volume_array() -> np.ndarray:
	labels = np.zeros(VOLUME_SHAPE, dtype=np.int16)
	for x in range(VOLUME_SHAPE[0]):
		labels[x, :, :] = x % len(CLASS_IDS)
	return labels


def split_grid_array() -> np.ndarray:
	grid = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
	grid[:8, :8, :8] = 1
	grid[8, :, :] = 2
	grid[:, 8, :] = 2
	return grid


def _array_sha256(array: np.ndarray) -> str:
	value = np.ascontiguousarray(array)
	hasher = hashlib.sha256()
	hasher.update(value.dtype.str.encode('ascii'))
	hasher.update(
		json.dumps(list(value.shape), separators=(',', ':')).encode('ascii')
	)
	hasher.update(value.view(np.uint8))
	return hasher.hexdigest()


def _split_mask_identity(grid: np.ndarray, split_code: int) -> tuple[str, int]:
	mask = np.ascontiguousarray(grid.reshape(-1) == split_code)
	hasher = hashlib.sha256()
	hasher.update(mask.view(np.uint8))
	return hasher.hexdigest(), int(np.count_nonzero(mask))


def _per_class_counts(
	labels: np.ndarray, grid: np.ndarray, split_code: int
) -> dict[str, int]:
	mask = grid == split_code
	return {
		str(class_id): int(np.count_nonzero(labels[mask] == class_id))
		for class_id in CLASS_IDS
	}


def build_five_way_universe(root: Path) -> dict[str, object]:
	"""Create the five synthetic sources plus every shared label input."""
	labels_dir = root / 'labels'
	f3_root = root / 'f3_root'
	labels_dir.mkdir(parents=True, exist_ok=True)
	f3_root.mkdir(parents=True, exist_ok=True)
	label_volume = labels_dir / 'f3_facies_labels.npy'
	np.save(label_volume, label_volume_array(), allow_pickle=False)
	class_info = labels_dir / 'class_info.json'
	class_info.write_text(
		json.dumps(
			{
				str(class_id): {
					'name': f'class_{class_id}',
					'color': [class_id, class_id, class_id],
				}
				for class_id in CLASS_IDS
			}
		),
		encoding='utf-8',
	)
	label_segy = f3_root / 'f3_labels.sgy'
	label_segy.write_bytes(b'synthetic-segy')
	geometry_json = labels_dir / 'segy_geometry.json'
	geometry_json.write_text(
		json.dumps(
			{
				'cube_shape': list(VOLUME_SHAPE),
				'iline_min': 100,
				'iline_max': 115,
				'xline_min': 200,
				'xline_max': 215,
			}
		),
		encoding='utf-8',
	)
	inventory = labels_dir / 'label_png_inventory.csv'
	inventory.write_text(
		'relative_path,split,slice_type,slice_index\n'
		'a.png,validation,inline,108\n'
		'b.png,validation,crossline,208\n',
		encoding='utf-8',
	)

	pretext_dirs = {
		'mae_hmm_k6': str(
			root / 'pseudo_targets/f3/ssl_hmm_continuation_v1/mae100'
		),
		'local_barlow_twins_hmm_k6': str(
			root / 'pseudo_targets/f3/mae_local_bt_five_way_v1/local_bt100'
		),
	}
	stage1_checkpoints = {
		'mae': write_stage1_checkpoint(
			root / 'pretraining/stage1/mae/full_100ep/latest.pt',
			mae_stage1_config(),
		),
		'local_barlow_twins': write_stage1_checkpoint(
			root / 'pretraining/stage1/local_barlow_twins_v1/full_100ep/latest.pt',
			local_bt_stage1_config(),
		),
	}
	mae_checkpoint = root / 'pretraining/mae/latest.pt'
	valid_mask = np.ones(TOKEN_GRID, dtype=np.bool_)
	rng = np.random.default_rng(11)
	models = []
	for model_id in FIVE_WAY_MODEL_IDS:
		checkpoint = (
			mae_checkpoint
			if model_id == 'mae'
			else root / f'pretraining/{model_id}/latest.pt'
		)
		checkpoint.parent.mkdir(parents=True, exist_ok=True)
		torch.save(
			checkpoint_payload(
				model_id,
				mae_checkpoint,
				pseudo_target_dirs=pretext_dirs,
				stage1_checkpoints=stage1_checkpoints,
			),
			checkpoint,
		)
		sha256 = file_sha256(checkpoint)
		metadata = base_embedding_metadata(checkpoint, sha256)
		if model_id in ('local_barlow_twins', 'local_barlow_twins_hmm_k6'):
			metadata['pretraining_method'] = 'local_barlow_twins_3d'
			metadata['pretraining_objective'] = local_bt_objective()
		else:
			metadata['pretraining_objective'] = mae_objective()
		if model_id in pretext_dirs:
			base = (
				'amp_mae3d'
				if model_id == 'mae_hmm_k6'
				else 'local_barlow_twins_3d'
			)
			metadata['stratigraphy_pretext'] = pretext_identity(
				base, pretext_dirs[model_id]
			)
		embeddings_dir = (
			root
			/ 'embeddings/f3/facies_benchmark_v1/mae_local_bt_five_way_v1'
			/ model_id
			/ 'overlap_x64'
		)
		embeddings_dir.mkdir(parents=True, exist_ok=True)
		np.save(
			embeddings_dir / f'{SURVEY_ID}.embeddings.npy',
			rng.standard_normal((*TOKEN_GRID, 384)).astype(np.float16),
			allow_pickle=False,
		)
		np.save(
			embeddings_dir / f'{SURVEY_ID}.valid_tokens.npy',
			valid_mask,
			allow_pickle=False,
		)
		(embeddings_dir / f'{SURVEY_ID}.embedding_metadata.json').write_text(
			json.dumps(metadata, indent=1, sort_keys=True), encoding='utf-8'
		)
		models.append(
			{
				'model_id': model_id,
				'checkpoint': str(checkpoint),
				'embeddings_dir': str(embeddings_dir),
				'expected': deepcopy(dict(EXPECTED_MODEL_IDENTITIES[model_id])),
			}
		)
	return {
		'paths': {'artifact_root': str(root), 'f3_root': str(f3_root)},
		'dataset': dict(DATASET),
		'labels': {
			'source_label_volume': str(label_volume),
			'source_label_segy': str(label_segy),
			'png_label_inventory': str(inventory),
			'segy_geometry_json': str(geometry_json),
			'class_info': str(class_info),
		},
		'section_layout': {
			'dataset_root': str(root / 'lithology/voxel_section_layout_v1'),
		},
		'models': models,
		'outputs': {
			'runs_root': str(root / 'f3_lithology_benchmark/runs'),
			'summary_root': str(root / 'f3_lithology_benchmark/summary'),
		},
	}


def write_condition(
	universe: dict[str, object], layout_id: str, data_size: str
) -> Path:
	"""Materialize one valid seven-file section-layout condition."""
	labels = label_volume_array()
	grid = split_grid_array()
	tokens = np.asarray([[0, 0, 0]], dtype=np.int64)
	condition = (
		Path(universe['section_layout']['dataset_root'])
		/ 'datasets'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ 'voxel_supervision'
	)
	condition.mkdir(parents=True, exist_ok=True)
	np.save(condition / 'supervision_split_grid.npy', grid, allow_pickle=False)
	np.save(condition / 'selected_token_xyz.npy', tokens, allow_pickle=False)

	label_volume = Path(universe['labels']['source_label_volume'])
	class_info = Path(universe['labels']['class_info'])
	inventory = Path(universe['labels']['png_label_inventory'])
	label_segy = Path(universe['labels']['source_label_segy'])
	mae_embeddings_dir = Path(universe['models'][0]['embeddings_dir'])
	reference_metadata_path = (
		mae_embeddings_dir / f'{SURVEY_ID}.embedding_metadata.json'
	)
	reference_valid_tokens = mae_embeddings_dir / f'{SURVEY_ID}.valid_tokens.npy'
	classes = [
		{
			'class_id': class_id,
			'class_name': f'class_{class_id}',
			'rgb': [class_id, class_id, class_id],
			'hex_color': f'#{class_id:02x}{class_id:02x}{class_id:02x}',
		}
		for class_id in CLASS_IDS
	]
	voxel_metadata = {
		'artifact_type': 'f3_lithology_voxel_supervision',
		'schema_version': 1,
		'dataset': dict(DATASET),
		'labels': {
			'source_label_segy': str(label_segy),
			'class_info': str(class_info),
		},
		'classes': classes,
		'geometry': {
			'shape_xyz': list(VOLUME_SHAPE),
			'inline_min': 100,
			'inline_max': 115,
			'crossline_min': 200,
			'crossline_max': 215,
		},
		'split_codes': {'unsupervised': 0, 'train': 1, 'validation': 2},
		'reference_embedding': {
			'path': str(reference_metadata_path),
			'sha256': file_sha256(reference_metadata_path),
			'patch_size': list(PATCH_SIZE),
			'volume_shape_xyz': list(VOLUME_SHAPE),
			'metadata': json.loads(
				reference_metadata_path.read_text(encoding='utf-8')
			),
		},
		'reference_valid_tokens': {
			'path': str(reference_valid_tokens),
			'sha256': file_sha256(reference_valid_tokens),
		},
		'label_volume': {
			'path': str(label_volume),
			'sha256': file_sha256(label_volume),
		},
		'inventory': {'path': str(inventory), 'sha256': file_sha256(inventory)},
	}
	(condition / 'voxel_dataset_metadata.json').write_text(
		json.dumps(voxel_metadata), encoding='utf-8'
	)
	train_counts = _per_class_counts(labels, grid, 1)
	validation_counts = _per_class_counts(labels, grid, 2)
	with (condition / 'class_counts.csv').open(
		'w', newline='', encoding='utf-8'
	) as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=('split', 'class_id', 'class_name', 'count', 'fraction'),
		)
		writer.writeheader()
		for split, counts in (
			('train', train_counts),
			('validation', validation_counts),
		):
			total = sum(counts.values())
			for class_id in CLASS_IDS:
				count = counts[str(class_id)]
				writer.writerow(
					{
						'split': split,
						'class_id': class_id,
						'class_name': f'class_{class_id}',
						'count': count,
						'fraction': count / total,
					}
				)
	(condition / 'split_manifest.json').write_text(
		json.dumps({'splits': ['train', 'validation']}), encoding='utf-8'
	)
	(condition / 'summary.md').write_text(
		'# synthetic section-layout condition\n', encoding='utf-8'
	)
	train_mask_sha256, train_count = _split_mask_identity(grid, 1)
	validation_mask_sha256, validation_count = _split_mask_identity(grid, 2)
	identity = {
		'layout_id': layout_id,
		'data_size': data_size,
		'parent_size': None,
		'patch_size_xyz': list(PATCH_SIZE),
		'volume_shape_xyz': list(VOLUME_SHAPE),
		'class_order': list(CLASS_IDS),
		'target_train_voxel_count': train_count,
		'actual_train_voxel_count': train_count,
		'relative_count_error': 0.0,
		'selected_token_count': int(tokens.shape[0]),
		'selected_token_identity_sha256': _array_sha256(tokens),
		'train_mask_sha256': train_mask_sha256,
		'validation_mask_sha256': validation_mask_sha256,
		'validation_voxel_count': validation_count,
		'grid_array_sha256': _array_sha256(grid),
		'per_line_contributions': {
			'inline:100': train_count - train_count // 2,
			'crossline:200': train_count // 2,
		},
		'per_class_train_voxel_counts': train_counts,
		'per_class_validation_voxel_counts': validation_counts,
	}
	output_names = (
		'supervision_split_grid.npy',
		'selected_token_xyz.npy',
		'voxel_dataset_metadata.json',
		'class_counts.csv',
		'split_manifest.json',
		'summary.md',
	)
	metadata = {
		'artifact_type': 'f3_lithology_voxel_section_layout_dataset',
		'schema_version': 1,
		'identity': identity,
		'layout_id': layout_id,
		'data_size': data_size,
		'active_lines': {'inline': [100], 'crossline': [200]},
		'selection_semantics': 'stable_hash_partial_section_token_footprints_v1',
		'outputs': {
			name: {
				'path': str(condition / name),
				'sha256': file_sha256(condition / name),
			}
			for name in output_names
		},
	}
	(condition / 'section_layout_metadata.json').write_text(
		json.dumps(metadata), encoding='utf-8'
	)
	return condition
