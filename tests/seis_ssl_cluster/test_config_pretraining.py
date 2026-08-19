from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config.pretraining import resolve_mae_training_config


def test_pretraining_config_resolves_from_stage_module() -> None:
	resolved = resolve_mae_training_config(_minimal_training_config())

	assert resolved['stage'] == 'train_amp_mae'
	assert 'continuation' not in resolved
	assert resolved['data']['amplitude_agc'] == {'enabled': False}
	assert resolved['data']['finite_check_mode'] == 'strict'
	assert resolved['loss']['visible_reconstruction_weight'] == 0.0
	assert resolved['loss']['target_normalization'] == {'mode': 'none'}
	assert resolved['train']['runtime_check_mode'] == 'once'


def test_pretraining_config_accepts_continuation() -> None:
	cfg = _minimal_training_config()
	cfg['continuation'] = {
		'init_checkpoint': '/checkpoints/mae/latest.pt',
		'unfreeze_top_blocks': 1,
	}

	resolved = resolve_mae_training_config(cfg)

	assert resolved['continuation'] == cfg['continuation']


@pytest.mark.parametrize(
	('continuation', 'error', 'message'),
	[
		(
			{
				'init_checkpoint': 'checkpoints/latest.pt',
				'unfreeze_top_blocks': 1,
			},
			ValueError,
			r'continuation\.init_checkpoint must be an absolute path',
		),
		(
			{'init_checkpoint': '', 'unfreeze_top_blocks': 1},
			TypeError,
			r'continuation\.init_checkpoint must be a non-empty string',
		),
		(
			{
				'init_checkpoint': '/checkpoints/latest.pt',
				'unfreeze_top_blocks': 1,
				'optimizer': '/checkpoints/optimizer.pt',
			},
			ValueError,
			r'continuation key\(s\) not allowed',
		),
		(
			{'unfreeze_top_blocks': 1},
			ValueError,
			r'continuation\.init_checkpoint is required',
		),
		(
			{'init_checkpoint': '/checkpoints/latest.pt'},
			ValueError,
			r'continuation\.unfreeze_top_blocks is required',
		),
		(
			{
				'init_checkpoint': '/checkpoints/latest.pt',
				'unfreeze_top_blocks': 0,
			},
			ValueError,
			r'continuation\.unfreeze_top_blocks must be a positive integer',
		),
		(
			{
				'init_checkpoint': '/checkpoints/latest.pt',
				'unfreeze_top_blocks': True,
			},
			ValueError,
			r'continuation\.unfreeze_top_blocks must be a positive integer',
		),
		(
			{
				'init_checkpoint': '/checkpoints/latest.pt',
				'unfreeze_top_blocks': 9,
			},
			ValueError,
			(
				r'continuation\.unfreeze_top_blocks must be less than or equal to '
				r'model\.encoder_depth \(8\)'
			),
		),
	],
)
def test_pretraining_config_rejects_invalid_continuation(
	continuation: dict[str, object],
	error: type[Exception],
	message: str,
) -> None:
	cfg = _minimal_training_config()
	cfg['continuation'] = continuation

	with pytest.raises(error, match=message):
		resolve_mae_training_config(cfg)


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


@pytest.mark.parametrize('mode', ['strict', 'output_only', 'off'])
def test_pretraining_config_accepts_finite_check_modes(mode: str) -> None:
	cfg = _minimal_training_config()
	cfg['data']['finite_check_mode'] = mode

	resolved = resolve_mae_training_config(cfg)

	assert resolved['data']['finite_check_mode'] == mode


@pytest.mark.parametrize('mode', ['strict', 'once', 'minimal'])
def test_pretraining_config_accepts_runtime_check_modes(mode: str) -> None:
	cfg = _minimal_training_config()
	cfg['train']['runtime_check_mode'] = mode

	resolved = resolve_mae_training_config(cfg)

	assert resolved['train']['runtime_check_mode'] == mode


def test_pretraining_config_rejects_invalid_runtime_check_mode() -> None:
	cfg = _minimal_training_config()
	cfg['train']['runtime_check_mode'] = 'off'

	with pytest.raises(ValueError, match=r'train\.runtime_check_mode'):
		resolve_mae_training_config(cfg)


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


def test_pretraining_config_preserves_runs_output_root() -> None:
	cfg = _minimal_training_config()
	explicit_output = '/artifacts/runs/train_amp_mae'
	cfg['paths']['output_root'] = explicit_output

	resolved = resolve_mae_training_config(cfg)

	assert resolved['paths']['output_root'] == explicit_output


def test_pretraining_config_accepts_explicit_pretraining_output_path() -> None:
	cfg = _minimal_training_config()
	explicit_output = (
		'/artifacts/pretraining/nopims/pretrain_v1/amp_mae_v1'
	)
	cfg['paths']['output_root'] = explicit_output

	resolved = resolve_mae_training_config(cfg)

	assert resolved['paths']['output_root'] == explicit_output


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
