from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config.pretraining import resolve_mae_training_config


def test_pretraining_config_resolves_from_stage_module() -> None:
	resolved = resolve_mae_training_config(_minimal_training_config())

	assert resolved['stage'] == 'train_amp_mae'
	assert resolved['data']['amplitude_agc'] == {'enabled': False}
	assert resolved['loss']['visible_reconstruction_weight'] == 0.0
	assert resolved['loss']['target_normalization'] == {'mode': 'none'}


def test_pretraining_config_accepts_enabled_agc() -> None:
	cfg = _minimal_training_config()
	cfg['data']['amplitude_agc'] = {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 65,
		'eps': 1.0e-3,
		'clip_abs': 5.0,
	}

	resolved = resolve_mae_training_config(cfg)

	assert resolved['data']['amplitude_agc']['enabled'] is True


def test_pretraining_config_validates_visible_reconstruction_weight() -> None:
	cfg = _minimal_training_config()
	cfg['loss']['visible_reconstruction_weight'] = -0.1

	with pytest.raises(ValueError, match=r'loss\.visible_reconstruction_weight'):
		resolve_mae_training_config(cfg)


def test_pretraining_config_validates_target_normalization() -> None:
	cfg = _minimal_training_config()
	cfg['loss'] = {
		'reconstruction': 'mse',
		'gradient_weight': 0.0,
		'target_normalization': {
			'mode': 'patch_zscore',
			'eps': 1.0e-6,
			'min_std': 0.05,
		},
	}

	resolved = resolve_mae_training_config(cfg)

	assert resolved['loss']['target_normalization']['mode'] == 'patch_zscore'


def test_pretraining_config_rejects_runs_output_root() -> None:
	cfg = _minimal_training_config()
	cfg['paths']['output_root'] = '/artifacts/runs/train_amp_mae'

	with pytest.raises(ValueError, match='runs/ paths'):
		resolve_mae_training_config(cfg)


def test_pretraining_config_enforces_pretraining_artifact_path_contract() -> None:
	cfg = _minimal_training_config()
	cfg['paths']['output_root'] = (
		'/artifacts/pretraining/nopims/pretrain_v1/amp_mae_v1'
	)

	with pytest.raises(ValueError, match=r'pretraining/nopims/pretrain_v1'):
		resolve_mae_training_config(cfg)


def _minimal_training_config() -> dict[str, object]:
	return deepcopy(
		{
			'paths': {
				'artifact_root': '/artifacts',
				'output_root': '/artifacts/pretraining/train_amp_mae',
			},
			'manifests': {
				'train': '/artifacts/manifests/train.json',
				'train_path_list': '/artifacts/splits/train_npy_paths.txt',
			},
			'data': {'local_crop_size': [128, 128, 128]},
			'model': {
				'patch_size': [8, 8, 8],
				'encoder_dim': 384,
				'encoder_depth': 8,
				'encoder_heads': 6,
				'decoder_dim': 256,
				'decoder_depth': 4,
				'decoder_heads': 4,
			},
			'masking': {
				'spatial_mask_ratio': 0.75,
				'block_size_tokens': [2, 2, 2],
			},
			'loss': {
				'reconstruction': 'huber',
				'huber_delta': 1.0,
				'gradient_weight': 0.05,
				'target_normalization': {'mode': 'none'},
			},
			'train': {
				'batch_size': 4,
				'samples_per_epoch': 10000,
				'epochs': 100,
				'amp': False,
			},
		},
	)
