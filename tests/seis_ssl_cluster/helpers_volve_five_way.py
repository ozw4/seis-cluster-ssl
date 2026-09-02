'''Shared synthetic artifacts for Volve horizon five-way tests.'''

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import yaml

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_MODEL_IDS,
	LOCAL_BARLOW_TWINS_METHOD,
	VolveHorizonFiveWayConfig,
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	FIVE_WAY_LOCAL_BT_STAGE1_GLOBAL_STEPS,
	FIVE_WAY_STAGE1_GLOBAL_STEPS,
	FIVE_WAY_STAGE2_GLOBAL_STEPS,
	VolveHorizonFiveWayEmbeddingSuite,
	audit_volve_horizon_five_way_sources,
	inspect_volve_horizon_five_way_embedding_suite,
)
from seis_ssl_cluster.volve.horizon_frozen import (
	OPTIMIZER_BETAS,
	OPTIMIZER_EPS,
	OPTIMIZER_NAME,
	decoder_initial_state_sha256,
	objective_identity,
)
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.volve.horizon_model import create_volve_horizon_decoder
from seis_ssl_cluster.volve.horizon_runner import (
	CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
)
from tests.seis_ssl_cluster.helpers_volve import (
	write_synthetic_frozen_horizon_data,
)

if TYPE_CHECKING:
	from collections.abc import Callable

	from seis_ssl_cluster.volve.horizon_data import VolveHorizonData
	from seis_ssl_cluster.volve.horizon_frozen import FrozenHorizonPlan

_MODEL_MAE = {
	'mae': 5.0,
	'mae_hmm_k6': 4.0,
	'local_barlow_twins': 3.0,
	'local_barlow_twins_hmm_k6': 2.0,
	'random': 6.0,
}
_DECODER_ARCHITECTURE = create_volve_horizon_decoder().architecture
_DECODER_INITIAL_STATE_SHA256 = decoder_initial_state_sha256()


def five_way_embedding_sentinel(model_id: str) -> float:
	'''Return the non-zero value uniquely assigned to one synthetic model.'''
	return float(FIVE_WAY_MODEL_IDS.index(model_id) + 1)


def five_way_config_mapping(
	tmp_path: Path,
	*,
	checkpoint_selection: str | None = None,
) -> dict[str, object]:
	'''Build a portable five-way config rooted below a pytest directory.'''
	artifact_root = (tmp_path / 'artifacts').resolve()
	config: dict[str, object] = {
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
		'models': {
			model_id: {
				'checkpoint': str(
					artifact_root / 'checkpoints' / model_id / 'latest.pt'
				),
				'embeddings_dir': str(artifact_root / 'embeddings' / model_id),
			}
			for model_id in FIVE_WAY_MODEL_IDS
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
	if checkpoint_selection is not None:
		config['checkpoint_selection'] = checkpoint_selection
	return config


def write_five_way_universe(  # noqa: PLR0915
	tmp_path: Path,
	*,
	embeddings: bool,
	checkpoint_selection: str | None = None,
) -> dict[str, Any]:
	'''Write valid checkpoint lineage and optionally all five embedding sources.'''
	raw = five_way_config_mapping(
		tmp_path, checkpoint_selection=checkpoint_selection
	)
	models = cast('dict[str, dict[str, str]]', raw['models'])
	checkpoint_paths = {
		model_id: Path(models[model_id]['checkpoint'])
		for model_id in FIVE_WAY_MODEL_IDS
	}
	parent_root = (tmp_path / 'artifacts' / 'parents').resolve()
	mae_parent = parent_root / 'mae100.pt'
	local_parent = parent_root / 'local_bt100.pt'
	mae_parent_payload = _stage1_payload(local=False)
	local_parent_payload = _stage1_payload(local=True)
	save_checkpoint(mae_parent, mae_parent_payload)
	save_checkpoint(local_parent, local_parent_payload)
	mae_parent_sha = file_sha256(mae_parent)
	local_parent_sha = file_sha256(local_parent)
	pseudo_paths: dict[str, Path] = {}

	for model_id in FIVE_WAY_MODEL_IDS:
		path = checkpoint_paths[model_id]
		if model_id == 'random':
			payload = {
				'config': deepcopy(mae_parent_payload['config']),
				'epoch': 0,
				'global_step': 0,
				'metadata': {
					'random_encoder_baseline': True,
					'pretrained_weights_loaded': False,
					'seed': 42,
					'reference_checkpoint': str(mae_parent),
					'reference_checkpoint_sha256': mae_parent_sha,
				},
				'training_state': {
					'stage': 'create_random_mae_checkpoint',
					'checkpoint_kind': 'random_init',
				},
			}
		elif model_id in {'mae', 'local_barlow_twins'}:
			local = model_id == 'local_barlow_twins'
			parent = local_parent if local else mae_parent
			parent_sha = local_parent_sha if local else mae_parent_sha
			config = deepcopy(
				local_parent_payload['config']
				if local
				else mae_parent_payload['config']
			)
			config['train'] = {
				'epochs': 25,
				'samples_per_epoch': 10_000,
				'batch_size': 4,
			}
			config['continuation'] = {
				'init_checkpoint': str(parent),
				'unfreeze_top_blocks': 1,
			}
			payload = {
				'config': config,
				'epoch': 25,
				'global_step': FIVE_WAY_STAGE2_GLOBAL_STEPS,
				'continuation_lineage': {
					'schema_version': 1,
					'init_checkpoint': str(parent),
					'init_checkpoint_sha256': parent_sha,
					'resume_count': 0,
				},
				'training_state': (
					{
						'stage': 'barlow_twins_training',
						'completed_epoch': True,
					}
					if local
					else {
						'stage': 'train_amp_mae',
						'checkpoint_kind': 'epoch',
					}
				),
			}
		else:
			local = model_id == 'local_barlow_twins_hmm_k6'
			parent = local_parent if local else mae_parent
			parent_sha = local_parent_sha if local else mae_parent_sha
			parent_payload = local_parent_payload if local else mae_parent_payload
			pseudo_dir = (tmp_path / 'artifacts' / 'pseudo' / model_id).resolve()
			pseudo_path = pseudo_dir / 'volve_st10010.pseudo_target_metadata.json'
			labels_path = pseudo_dir / 'volve_st10010.pseudo_labels.npy'
			confidence_path = pseudo_dir / 'volve_st10010.pseudo_confidence.npy'
			valid_tokens_path = pseudo_dir / 'volve_st10010.valid_tokens.npy'
			stage1_embedding_metadata = (
				tmp_path
				/ 'artifacts'
				/ 'pseudo_sources'
				/ model_id
				/ 'volve_st10010.embedding_metadata.json'
			).resolve()
			write_json(
				stage1_embedding_metadata,
				{
					'checkpoint_path': str(parent),
					'checkpoint_sha256': parent_sha,
				},
			)
			cluster_metadata = stage1_embedding_metadata.with_name(
				'volve_st10010.cluster_label_metadata.json'
			)
			write_json(
				cluster_metadata,
				{
					'embedding_input': {
						'metadata_path': str(stage1_embedding_metadata),
						'metadata_sha256': file_sha256(
							stage1_embedding_metadata
						),
					}
				},
			)
			write_json(
				pseudo_path,
				{
					'artifact_type': 'strat_hmm_pseudo_target',
					'schema_version': 2,
					'k': 6,
					'survey_id': 'volve_st10010',
					'source': {
						'source_metadata_path': str(cluster_metadata),
						'source_metadata_sha256': file_sha256(cluster_metadata),
					},
				},
			)
			pseudo_dir.mkdir(parents=True, exist_ok=True)
			np.save(labels_path, np.zeros((1, 1, 1), dtype=np.int16))
			np.save(confidence_path, np.ones((1, 1, 1), dtype=np.float32))
			np.save(valid_tokens_path, np.ones((1, 1, 1), dtype=np.bool_))
			pseudo_paths[model_id] = pseudo_path
			payload = _hmm_payload(
				parent=parent,
				parent_sha=parent_sha,
				parent_payload=parent_payload,
				pseudo_dir=pseudo_dir,
				pseudo_path=pseudo_path,
				labels_path=labels_path,
				confidence_path=confidence_path,
				valid_tokens_path=valid_tokens_path,
			)
		save_checkpoint(path, payload)

	config = volve_horizon_five_way_config_from_mapping(raw)
	universe: dict[str, Any] = {
		'config': config,
		'checkpoints': checkpoint_paths,
		'parents': {'mae': mae_parent, 'local': local_parent},
		'pseudo_metadata': pseudo_paths,
	}
	if embeddings:
		universe.update(_write_embedding_sources(config, tmp_path))
	return universe


def _hmm_payload(  # noqa: PLR0913
	*,
	parent: Path,
	parent_sha: str,
	parent_payload: dict[str, object],
	pseudo_dir: Path,
	pseudo_path: Path,
	labels_path: Path,
	confidence_path: Path,
	valid_tokens_path: Path,
) -> dict[str, object]:
	return {
		'config': deepcopy(parent_payload['config']),
		'stratigraphy_config': {
			'head': {'num_prototypes': 6},
			'teacher': {'checkpoint': str(parent)},
			'student': {
				'init_checkpoint': str(parent),
				'unfreeze_top_blocks': 1,
			},
			'pseudo_targets': {'input_dir': str(pseudo_dir), 'k': 6},
			'train': {
				'epochs': 25,
				'samples_per_epoch': 10_000,
				'batch_size': 4,
			},
		},
		'epoch': 25,
		'global_step': FIVE_WAY_STAGE2_GLOBAL_STEPS,
		'training_state': {
			'stage': 'train_strat_hmm_pretext',
			'checkpoint_kind': 'epoch',
		},
		'control_identity': {
			'input_identities': {
				'teacher_checkpoint': {
					'path': str(parent),
					'sha256': parent_sha,
				},
				'student_init_checkpoint': {
					'path': str(parent),
					'sha256': parent_sha,
				},
				'pseudo_targets': [
					{
						'survey_id': 'volve_st10010',
						'labels': _file_identity(labels_path),
						'confidence': _file_identity(confidence_path),
						'valid_tokens': _file_identity(valid_tokens_path),
						'metadata': _file_identity(pseudo_path),
						'boundary_weight_present': False,
					}
				],
			}
		},
	}


def _stage1_payload(*, local: bool) -> dict[str, object]:
	config: dict[str, object] = {
		'stage': 'barlow_twins_training' if local else 'train_amp_mae',
		'model': {
			'in_channels': 1,
			'patch_size': [8, 8, 8],
			'encoder_dim': 384,
			'encoder_depth': 8,
			'encoder_heads': 6,
		},
		'train': {
			'epochs': 100,
			'samples_per_epoch': 10_000,
			'batch_size': 16 if local else 4,
		},
	}
	if local:
		config['barlow_twins'] = {
			'method': LOCAL_BARLOW_TWINS_METHOD,
			'local_pairs_per_crop': 128,
		}
		config['augmentations'] = {
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.05,
		}
	return {
		'config': config,
		'epoch': 100,
		'global_step': (
			FIVE_WAY_LOCAL_BT_STAGE1_GLOBAL_STEPS
			if local
			else FIVE_WAY_STAGE1_GLOBAL_STEPS
		),
		'training_state': {
			'stage': 'barlow_twins_training' if local else 'train_amp_mae',
			'checkpoint_kind': 'epoch',
		},
	}


def _write_embedding_sources(
	config: VolveHorizonFiveWayConfig,
	tmp_path: Path,
) -> dict[str, object]:
	volume_shape = (16, 16, 800)
	token_grid = (2, 2, 100)
	valid_mask_path = (tmp_path / 'public' / 'valid_trace_mask.npy').resolve()
	valid_mask_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(valid_mask_path, np.ones(volume_shape[:2], dtype=np.bool_))
	amplitude_path = (tmp_path / 'public' / 'amplitude.npy').resolve()
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
	common = {
		'survey_id': 'volve_st10010',
		'source_amplitude_path': str(amplitude_path),
		'source_valid_mask_path': str(valid_mask_path),
		'volume_shape_xyz': list(volume_shape),
		'model_geometry': {
			'in_channels': 1,
			'patch_size': [8, 8, 8],
			'encoder_dim': 384,
			'encoder_depth': 8,
			'encoder_heads': 6,
		},
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
	metadata_paths: dict[str, Path] = {}
	valid_tokens = np.ones(token_grid, dtype=np.bool_)
	for model_id in FIVE_WAY_MODEL_IDS:
		model = config.model_by_id(model_id)
		paths = output_paths(model.embeddings_dir, config.survey_id)
		paths.embeddings.parent.mkdir(parents=True, exist_ok=True)
		np.save(
			paths.embeddings,
			np.full(
				(*token_grid, 384),
				five_way_embedding_sentinel(model_id),
				dtype=np.float16,
			),
		)
		np.save(paths.valid_tokens, valid_tokens)
		metadata = {
			**common,
			'checkpoint_path': str(model.checkpoint),
			'checkpoint_sha256': file_sha256(model.checkpoint),
			'pretraining_objective': (
				{
					'method': LOCAL_BARLOW_TWINS_METHOD,
					'local_pairs_per_crop': 128,
				}
				if model_id.startswith('local_barlow_twins')
				else {'reconstruction': 'mse'}
			),
		}
		if model_id.startswith('local_barlow_twins'):
			metadata['pretraining_method'] = LOCAL_BARLOW_TWINS_METHOD
		if model_id.endswith('_hmm_k6'):
			metadata['stratigraphy_pretext'] = {
				'method': 'strat_hmm_pretext',
				'base_objective': (
					LOCAL_BARLOW_TWINS_METHOD
					if model_id.startswith('local_barlow_twins')
					else 'amp_mae3d'
				),
				'head_num_prototypes': 6,
				'unfreeze_top_blocks': 1,
				'pseudo_target_input_dir': str(
					tmp_path / 'artifacts' / 'pseudo' / model_id
				),
			}
		write_json(paths.metadata, metadata)
		metadata_paths[model_id] = paths.metadata
	return {'embedding_metadata': metadata_paths}


def write_five_way_horizon_fixture(
	tmp_path: Path,
	config: VolveHorizonFiveWayConfig,
) -> tuple[VolveHorizonData, Path]:
	'''Write horizon data/layouts and resize every embedding source to match.'''
	data = write_synthetic_frozen_horizon_data(tmp_path / 'horizon')
	_resize_embeddings_to_horizon_data(config, data)
	layout_path = tmp_path / 'layouts.yaml'
	layout_path.write_text(
		yaml.safe_dump(
			{
				'selection': {
					'semantics': (
						'explicit_section_prefix_all_available_horizon_points_v1'
					)
				},
				'validation': {'inline': [120], 'crossline': [220]},
				'layouts': {
					f'layout_{index:03d}': {
						'inline': list(range(100 + 4 * index, 104 + 4 * index)),
						'crossline': list(
							range(200 + 4 * index, 204 + 4 * index)
						),
					}
					for index in range(5)
				},
			},
			sort_keys=False,
		),
		encoding='utf-8',
	)
	return data, layout_path


def _resize_embeddings_to_horizon_data(
	config: VolveHorizonFiveWayConfig,
	data: VolveHorizonData,
) -> None:
	volume_shape = (*data.shape_xy, len(data.time_ms))
	token_grid = tuple((value + 7) // 8 for value in volume_shape)
	valid_tokens = np.ones(token_grid, dtype=np.bool_)
	valid_tokens[0, 0, :] = False
	canonical = json.loads(
		config.canonical_input_metadata.read_text(encoding='utf-8')
	)
	identity = canonical['scientific_identity']
	identity.update(
		{
			'shape_xyz': list(volume_shape),
			'valid_trace_mask_sha256': file_sha256(data.paths.valid_trace_mask),
			'inline_values_sha256': file_sha256(data.paths.inline_values),
			'crossline_values_sha256': file_sha256(data.paths.crossline_values),
			'time_axis_sha256': file_sha256(data.paths.time_ms),
		}
	)
	canonical['scientific_identity_sha256'] = _json_sha256(identity)
	canonical['provenance']['public_inputs']['valid_trace_mask.npy'] = str(
		data.paths.valid_trace_mask
	)
	write_json(config.canonical_input_metadata, canonical)
	for model in config.models:
		paths = output_paths(model.embeddings_dir, config.survey_id)
		metadata = json.loads(paths.metadata.read_text(encoding='utf-8'))
		metadata.update(
			{
				'source_valid_mask_path': str(data.paths.valid_trace_mask),
				'volume_shape_xyz': list(volume_shape),
				'token_grid_shape': list(token_grid),
			}
		)
		np.save(
			paths.embeddings,
			np.full(
				(*token_grid, 384),
				five_way_embedding_sentinel(model.model_id),
				dtype=np.float16,
			),
		)
		np.save(paths.valid_tokens, valid_tokens)
		write_json(paths.metadata, metadata)


def write_five_way_completed_run(
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
	*,
	embedding_suite: VolveHorizonFiveWayEmbeddingSuite | None = None,
) -> None:
	'''Write one internally consistent completed synthetic result cell.'''
	job_dir = five_way_job_dir(config, model_id, layout_id, data_size)
	job_dir.mkdir(parents=True)
	primary_counts = {
		name: 10 + index for index, name in enumerate(HORIZON_NAMES)
	}
	secondary_counts = {
		name: 20 + index for index, name in enumerate(HORIZON_NAMES)
	}
	if embedding_suite is None:
		source_audit = audit_volve_horizon_five_way_sources(config)
		embedding_suite = inspect_volve_horizon_five_way_embedding_suite(
			config,
			source_audit=source_audit,
		)
	identity = _run_identity(
		config,
		model_id,
		layout_id,
		data_size,
		primary_counts=primary_counts,
		secondary_counts=secondary_counts,
		embedding_suite=embedding_suite,
	)
	metrics = {
		'schema_version': 1,
		'artifact_type': 'volve_frozen_horizon_job_metrics',
		'model': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'benchmark_identity': identity,
		'runtime_precision': identity['runtime_precision'],
		'best_epoch': 2,
		'validation': _evaluation_metrics(
			model_id,
			layout_id,
			data_size,
			dict.fromkeys(HORIZON_NAMES, 4),
		),
		'test': {
			'primary_common': _evaluation_metrics(
				model_id,
				layout_id,
				data_size,
				primary_counts,
			),
			'secondary_per_horizon': _evaluation_metrics(
				model_id,
				layout_id,
				data_size,
				secondary_counts,
			),
			'evaluation_pass_count': 1,
		},
	}
	_write_best_and_metrics(job_dir, metrics)


def write_five_way_completed_matrix(
	tmp_path: Path,
	*,
	checkpoint_selection: str | None = None,
) -> VolveHorizonFiveWayConfig:
	'''Write the complete 5 by 5 by 3 synthetic result universe.'''
	universe = write_five_way_universe(
		tmp_path,
		embeddings=True,
		checkpoint_selection=checkpoint_selection,
	)
	config = cast('VolveHorizonFiveWayConfig', universe['config'])
	source_audit = audit_volve_horizon_five_way_sources(config)
	embedding_suite = inspect_volve_horizon_five_way_embedding_suite(
		config,
		source_audit=source_audit,
	)
	for model_id in FIVE_WAY_MODEL_IDS:
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				write_five_way_completed_run(
					config,
					model_id,
					layout_id,
					data_size,
					embedding_suite=embedding_suite,
				)
	return config


def rewrite_five_way_completed_run(
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
	mutate: Callable[[dict[str, object]], object],
) -> None:
	'''Mutate one completed metrics payload and refresh its best checkpoint.'''
	job_dir = five_way_job_dir(config, model_id, layout_id, data_size)
	metrics = json.loads((job_dir / 'metrics.json').read_text(encoding='utf-8'))
	mutate(metrics)
	_write_best_and_metrics(job_dir, metrics)


def write_completed_plan_metrics(plan: FrozenHorizonPlan) -> None:
	'''Write completed metrics backed by an inspected real five-way plan.'''
	runtime_precision = {
		'device_type': 'cpu',
		'amp_enabled': False,
		'autocast_dtype': None,
		'scaler_required': False,
	}
	identity = {**plan.run_identity, 'runtime_precision': runtime_precision}
	validation = _plan_evaluation(plan.effective_per_horizon_counts['validation'])
	primary = _plan_evaluation(plan.effective_per_horizon_counts['test_primary'])
	secondary = _plan_evaluation(plan.effective_per_horizon_counts['test'])
	selection = plan.checkpoint_selection
	best_score_key = (
		'macro_within_2_samples'
		if selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2
		else 'macro_mae_samples'
	)
	best_score = float(validation[best_score_key])
	best_path = plan.output_dir / 'best.pt'
	torch.save(
		{
			'epoch': 2,
			'run_identity': identity,
			'runtime_precision': runtime_precision,
			'checkpoint_selection': selection,
			'best_validation_score': best_score,
			'validation': validation,
			'model_state_dict': {'weight': torch.zeros(1)},
		},
		best_path,
	)
	payload = {
		'schema_version': 1,
		'artifact_type': 'volve_frozen_horizon_job_metrics',
		'model': plan.model,
		'layout_id': plan.layout_id,
		'data_size': plan.data_size,
		'benchmark_identity': identity,
		'runtime_precision': runtime_precision,
		'best_epoch': 2,
		'checkpoint_selection': selection,
		'best_validation_score': best_score,
		'best_checkpoint': {
			'path': str(best_path),
			'sha256': file_sha256(best_path),
		},
		'validation': validation,
		'test': {
			'primary_common': primary,
			'secondary_per_horizon': secondary,
			'evaluation_pass_count': 1,
		},
	}
	write_json(plan.output_dir / 'metrics.json', payload)
	write_json(
		plan.output_dir / 'history.json',
		[
			{
				'epoch': epoch,
				'validation_macro_mae_samples': (
					float(validation['macro_mae_samples']) + 2 - epoch
				),
				'validation_macro_within_2_samples': (
					float(validation['macro_within_2_samples']) - 0.1 * (2 - epoch)
				),
			}
			for epoch in range(3)
		],
	)


def five_way_job_dir(
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
) -> Path:
	'''Return the canonical output directory for one synthetic result cell.'''
	return (
		config.runs_root
		/ f'model={model_id}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)


def save_checkpoint(path: Path, payload: object) -> None:
	'''Save a synthetic checkpoint, creating its parent directory.'''
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, object]:
	'''Load a synthetic checkpoint without restricting fixture payload types.'''
	return torch.load(path, map_location='cpu', weights_only=False)


def write_json(path: Path, payload: object) -> None:
	'''Write deterministic, human-readable fixture JSON.'''
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _run_identity(  # noqa: PLR0913
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
	*,
	primary_counts: dict[str, int],
	secondary_counts: dict[str, int],
	embedding_suite: VolveHorizonFiveWayEmbeddingSuite,
) -> dict[str, object]:
	model = config.model_by_id(model_id)
	source = embedding_suite.source_by_id(model_id)
	paths = source.paths
	return {
		'schema_version': 3,
		'benchmark': config.benchmark_id,
		'model': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'canonical_scientific_identity': {
			'scientific_identity_sha256': _digest('canonical')
		},
		'horizon_split_plan': {
			'layout_id': layout_id,
			'data_size': data_size,
			'scientific_identity_sha256': _digest(
				f'split/{layout_id}/{data_size}'
			),
		},
		'embedding': {
			'embeddings_path': str(paths.embeddings),
			'embeddings_sha256': source.embeddings_sha256,
			'metadata_path': str(paths.metadata),
			'metadata_sha256': source.metadata_sha256,
			'checkpoint_path': str(model.checkpoint),
			'checkpoint_sha256': source.checkpoint_identity['checkpoint_sha256'],
			'model_source': dict(source.checkpoint_identity),
			'valid_tokens_sha256': source.valid_tokens_sha256,
		},
		'decoder': {
			'initialization_seed': 42000,
			'initial_state_sha256': _DECODER_INITIAL_STATE_SHA256,
			'architecture': _DECODER_ARCHITECTURE,
		},
		'tiles': {
			'patch_size_xyz': [8, 8, 8],
			'core_size_tokens': [8, 8, 27],
			'context_halo_tokens': [1, 1, 0],
			'window_start': 552,
			'window_stop': 768,
			'order': 'lateral_token_grid_x_then_y_v1',
			'record_sha256': _digest(f'tiles/{layout_id}/{data_size}'),
		},
		'native_horizon_observation_counts': {
			'test_primary_common': dict(primary_counts),
			'test_secondary_per_horizon': dict(secondary_counts),
		},
		'effective_model_valid_observation_counts': {
			'train': dict.fromkeys(HORIZON_NAMES, 5),
			'validation': dict.fromkeys(HORIZON_NAMES, 4),
			'test_primary_common': dict(primary_counts),
			'test_secondary_per_horizon': dict(secondary_counts),
		},
		'excluded_by_token_validity_counts': {
			'train': dict.fromkeys(HORIZON_NAMES, 0),
			'validation': dict.fromkeys(HORIZON_NAMES, 0),
			'test_primary_common': dict.fromkeys(HORIZON_NAMES, 0),
			'test_secondary_per_horizon': dict.fromkeys(HORIZON_NAMES, 0),
		},
		'training': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 1.0e-3,
			'weight_decay': 1.0e-4,
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'amp_on_cuda': True,
			'gradient_clip_norm': 1.0,
		},
		'optimizer': {
			'name': OPTIMIZER_NAME,
			'betas': list(OPTIMIZER_BETAS),
			'eps': OPTIMIZER_EPS,
			'weight_decay': 1.0e-4,
		},
		'objective': objective_identity(config.checkpoint_selection),
		'runtime_precision': {
			'device_type': 'cpu',
			'amp_enabled': False,
			'autocast_dtype': None,
			'scaler_required': False,
		},
	}


def _evaluation_metrics(
	model_id: str,
	layout_id: str,
	data_size: str,
	counts: dict[str, int],
) -> dict[str, object]:
	per_horizon = {
		horizon_name: {
			'count': counts[horizon_name],
			'predicted_count': counts[horizon_name],
			'missing_prediction_count': 0,
			'mae_samples': _metric_value(
				model_id,
				layout_id,
				data_size,
				horizon_index,
			),
			'mae_ms': 4.0
			* _metric_value(
				model_id,
				layout_id,
				data_size,
				horizon_index,
			),
		}
		for horizon_index, horizon_name in enumerate(HORIZON_NAMES)
	}
	return {
		'macro_mae_samples': sum(
			float(item['mae_samples']) for item in per_horizon.values()
		)
		/ len(per_horizon),
		'macro_within_2_samples': 0.1 * (10.0 - _MODEL_MAE[model_id]),
		'macro': {
			'within_1': 0.05 * (10.0 - _MODEL_MAE[model_id]),
			'within_4': 0.1 * (11.0 - _MODEL_MAE[model_id]),
		},
		'per_horizon': per_horizon,
		'coverage': {
			'eligible_count': sum(counts.values()),
			'predicted_count': sum(counts.values()),
			'fraction': 1.0,
		},
		'missing_prediction_count': 0,
		'predicted_adjacent_order_violation_rate': 0.0,
		'predicted_adjacent_order_pair_count': 10,
	}


def _metric_value(
	model_id: str,
	layout_id: str,
	data_size: str,
	horizon_index: int | None = None,
) -> float:
	value = (
		_MODEL_MAE[model_id]
		+ 0.1 * LAYOUT_IDS.index(layout_id)
		+ 0.01 * tuple(DATA_SIZE_PREFIX).index(data_size)
	)
	if horizon_index is not None:
		value += 0.02 * horizon_index
	return value


def _write_best_and_metrics(
	job_dir: Path,
	metrics: dict[str, object],
) -> None:
	identity = metrics['benchmark_identity']
	assert isinstance(identity, dict)
	objective = identity['objective']
	assert isinstance(objective, dict)
	selection = str(objective['checkpoint_selection'])
	validation = metrics['validation']
	assert isinstance(validation, dict)
	if selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2:
		best_score = float(validation['macro_within_2_samples'])
	else:
		best_score = float(validation['macro_mae_samples'])
	metrics['checkpoint_selection'] = selection
	metrics['best_validation_score'] = best_score
	best_path = job_dir / 'best.pt'
	torch.save(
		{
			'epoch': metrics['best_epoch'],
			'run_identity': identity,
			'runtime_precision': metrics['runtime_precision'],
			'checkpoint_selection': selection,
			'best_validation_score': best_score,
			'validation': validation,
			'model_state_dict': {'weight': torch.zeros(1)},
		},
		best_path,
	)
	metrics['best_checkpoint'] = {
		'path': str(best_path),
		'sha256': file_sha256(best_path),
	}
	(job_dir / 'metrics.json').write_text(
		json.dumps(metrics),
		encoding='utf-8',
	)
	best_epoch = int(metrics['best_epoch'])
	history = []
	for epoch in range(best_epoch + 1):
		distance = best_epoch - epoch
		history.append(
			{
				'epoch': epoch,
				'validation_macro_mae_samples': (
					float(validation['macro_mae_samples']) + distance
				),
				'validation_macro_within_2_samples': (
					float(validation['macro_within_2_samples']) - 0.1 * distance
				),
			}
		)
	(job_dir / 'history.json').write_text(
		json.dumps(history), encoding='utf-8'
	)


def _plan_evaluation(counts: tuple[int, ...]) -> dict[str, object]:
	values = tuple(1.0 + 0.01 * index for index in range(len(HORIZON_NAMES)))
	per_horizon = {
		name: {
			'count': counts[index],
			'predicted_count': counts[index],
			'missing_prediction_count': 0,
			'mae_samples': values[index],
			'mae_ms': 4.0 * values[index],
		}
		for index, name in enumerate(HORIZON_NAMES)
	}
	total = sum(counts)
	return {
		'macro_mae_samples': sum(values) / len(values),
		'macro_within_2_samples': 0.5,
		'macro': {'within_1': 0.25, 'within_4': 0.75},
		'per_horizon': per_horizon,
		'coverage': {
			'eligible_count': total,
			'predicted_count': total,
			'fraction': 1.0,
		},
		'missing_prediction_count': 0,
		'predicted_adjacent_order_violation_rate': 0.0,
		'predicted_adjacent_order_pair_count': 1,
	}


def _digest(value: str) -> str:
	return hashlib.sha256(value.encode()).hexdigest()


def _json_sha256(payload: object) -> str:
	return hashlib.sha256(
		json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


def _file_identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}
