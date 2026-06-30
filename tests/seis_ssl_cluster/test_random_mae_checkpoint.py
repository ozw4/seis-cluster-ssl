from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.embedding import run_embedding_extraction
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.random_checkpoint import (
	create_random_mae_checkpoint,
	create_random_mae_checkpoint_from_config,
	random_mae_checkpoint_config_from_mapping,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_create_random_mae_checkpoint_preserves_reference_config_and_metadata(
	tmp_path: Path,
) -> None:
	reference_checkpoint = _write_reference_checkpoint(tmp_path)
	output_checkpoint = tmp_path / 'artifacts' / 'pretraining' / 'random.pt'

	result = create_random_mae_checkpoint(
		reference_checkpoint=reference_checkpoint,
		reference_model_tag='reference-model',
		seed=42,
		output_checkpoint=output_checkpoint,
	)

	assert result == output_checkpoint
	payload = load_checkpoint(output_checkpoint, map_location='cpu')
	reference_payload = load_checkpoint(reference_checkpoint, map_location='cpu')
	assert payload['config'] == reference_payload['config']
	assert payload['config']['data']['normalized_clip_abs'] == 8.0
	assert payload['config']['data']['amplitude_agc'] == {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 3,
		'eps': 1.0e-6,
		'clip_abs': 2.0,
	}
	assert payload['metadata'] == {
		'random_encoder_baseline': True,
		'reference_checkpoint': str(reference_checkpoint),
		'reference_model_tag': 'reference-model',
		'seed': 42,
		'pretrained_weights_loaded': False,
	}
	assert payload['optimizer_state_dict'] == {}
	assert payload['training_state']['stage'] == 'create_random_mae_checkpoint'
	assert _any_state_tensor_differs(
		payload['model_state_dict'],
		reference_payload['model_state_dict'],
	)


def test_create_random_mae_checkpoint_seed_determinism(tmp_path: Path) -> None:
	reference_checkpoint = _write_reference_checkpoint(tmp_path)
	first = tmp_path / 'artifacts' / 'random_seed42_a.pt'
	second = tmp_path / 'artifacts' / 'random_seed42_b.pt'
	third = tmp_path / 'artifacts' / 'random_seed43.pt'

	create_random_mae_checkpoint(
		reference_checkpoint=reference_checkpoint,
		reference_model_tag='reference-model',
		seed=42,
		output_checkpoint=first,
	)
	create_random_mae_checkpoint(
		reference_checkpoint=reference_checkpoint,
		reference_model_tag='reference-model',
		seed=42,
		output_checkpoint=second,
	)
	create_random_mae_checkpoint(
		reference_checkpoint=reference_checkpoint,
		reference_model_tag='reference-model',
		seed=43,
		output_checkpoint=third,
	)

	first_state = load_checkpoint(first, map_location='cpu')['model_state_dict']
	second_state = load_checkpoint(second, map_location='cpu')['model_state_dict']
	third_state = load_checkpoint(third, map_location='cpu')['model_state_dict']
	_assert_model_states_equal(first_state, second_state)
	assert _any_state_tensor_differs(first_state, third_state)


def test_create_random_mae_checkpoint_config_mapping_and_runs_rejection(
	tmp_path: Path,
) -> None:
	reference_checkpoint = _write_reference_checkpoint(tmp_path)
	config = {
		'paths': {'artifact_root': str(tmp_path / 'artifacts')},
		'reference_model': {
			'tag': 'reference-model',
			'checkpoint': str(reference_checkpoint),
		},
		'random_checkpoint': {
			'seed': 42,
			'output_checkpoint': str(
				tmp_path / 'artifacts' / 'pretraining' / 'random.pt',
			),
		},
	}

	settings = random_mae_checkpoint_config_from_mapping(config)
	assert settings.reference_checkpoint == reference_checkpoint
	assert settings.reference_model_tag == 'reference-model'
	assert settings.seed == 42
	assert settings.output_checkpoint == (
		tmp_path / 'artifacts' / 'pretraining' / 'random.pt'
	)

	config['random_checkpoint']['output_checkpoint'] = str(
		tmp_path / 'artifacts' / 'runs' / 'random.pt',
	)
	with pytest.raises(ValueError, match='runs/'):
		create_random_mae_checkpoint_from_config(config)


def test_random_mae_checkpoint_is_readable_by_embedding_extractor(
	tmp_path: Path,
) -> None:
	reference_checkpoint = _write_reference_checkpoint(tmp_path)
	random_checkpoint = tmp_path / 'artifacts' / 'pretraining' / 'random.pt'
	create_random_mae_checkpoint(
		reference_checkpoint=reference_checkpoint,
		reference_model_tag='reference-model',
		seed=42,
		output_checkpoint=random_checkpoint,
	)
	manifest_path = _write_embedding_manifest(tmp_path / 'survey')
	config = {
		'paths': {'artifact_root': str(tmp_path / 'artifacts')},
		'manifests': {'input': str(manifest_path)},
		'embeddings': {
			'checkpoint': str(random_checkpoint),
			'output_dir': str(tmp_path / 'artifacts' / 'embeddings'),
		},
		'embedding': {
			'window_size': [4, 4, 4],
			'overlap': [2, 2, 2],
			'output_dtype': 'float16',
			'batch_size': 1,
			'min_token_valid_fraction': 0.5,
		},
	}

	result = run_embedding_extraction(config, device='cpu')[0]

	assert result.embeddings_path.is_file()
	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['checkpoint_path'] == str(random_checkpoint)
	assert metadata['preprocessing'] == {
		'normalized_clip_abs': 8.0,
		'amplitude_agc': {
			'enabled': True,
			'mode': 'trace_rms_z',
			'window_z': 3,
			'eps': 1.0e-6,
			'clip_abs': 2.0,
		},
	}


def _write_reference_checkpoint(tmp_path: Path) -> Path:
	checkpoint_path = tmp_path / 'reference.pt'
	torch.manual_seed(7)
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=1,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)
	reference_state = {
		key: torch.full_like(value, 0.125) for key, value in model.state_dict().items()
	}
	torch.save(
		{
			'model_state_dict': reference_state,
			'config': _reference_checkpoint_config(tmp_path),
		},
		checkpoint_path,
	)
	return checkpoint_path


def _reference_checkpoint_config(tmp_path: Path) -> dict[str, object]:
	return {
		'stage': 'train_amp_mae',
		'paths': {'output_root': str(tmp_path / 'reference_run')},
		'manifests': {
			'train': str(tmp_path / 'reference_manifest.json'),
			'train_path_list': str(tmp_path / 'reference_paths.txt'),
		},
		'data': {
			'grid_order': list(GRID_ORDER_XYZ),
			'volume_format': 'npy_memmap',
			'input_channels': 1,
			'target_channels': 1,
			'use_context': False,
			'local_crop_size': [4, 4, 4],
			'min_valid_fraction': 0.1,
			'max_resample_attempts': 16,
			'normalized_clip_abs': 8.0,
			'amplitude_agc': {
				'enabled': True,
				'mode': 'trace_rms_z',
				'window_z': 3,
				'eps': 1.0e-6,
				'clip_abs': 2.0,
			},
		},
		'model': {
			'name': 'amp_mae3d',
			'in_channels': 1,
			'out_channels': 1,
			'patch_size': [2, 2, 2],
			'encoder_dim': 12,
			'encoder_depth': 1,
			'encoder_heads': 3,
			'decoder_dim': 12,
			'decoder_depth': 1,
			'decoder_heads': 3,
		},
		'masking': {
			'spatial_mask_ratio': 0.5,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
		},
		'loss': {
			'reconstruction': 'huber',
			'huber_delta': 1.0,
			'gradient_weight': 0.0,
			'visible_reconstruction_weight': 0.0,
			'target_normalization': {'mode': 'none'},
			'valid_mask_mode': 'voxel',
		},
		'zero_mask': {
			'enabled': False,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 16,
			'xy_trace_influence_radius': 1,
		},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 1,
			'epochs': 1,
			'num_workers': 0,
			'shuffle': False,
			'lr': 1.0e-4,
			'weight_decay': 0.0,
			'amp': False,
			'device': 'cpu',
			'seed': 7,
			'grad_clip_norm': 1.0,
		},
	}


def _write_embedding_manifest(root: Path) -> Path:
	root.mkdir(parents=True, exist_ok=True)
	volume_path = root / 'amplitude.npy'
	volume = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
	np.save(volume_path, volume)
	stats_path = root / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='tiny',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1000.0,
			clip_high=1000.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	manifest = SurveyManifest(
		survey_id='tiny',
		root=root,
		amplitude=AmplitudeVolumeRecord(
			survey_id='tiny',
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	manifest_path = root / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	return manifest_path


def _assert_model_states_equal(left: object, right: object) -> None:
	assert isinstance(left, dict)
	assert isinstance(right, dict)
	assert set(left) == set(right)
	for key in left:
		torch.testing.assert_close(left[key], right[key])


def _any_state_tensor_differs(left: object, right: object) -> bool:
	assert isinstance(left, dict)
	assert isinstance(right, dict)
	assert set(left) == set(right)
	return any(not torch.equal(left[key], right[key]) for key in left)
