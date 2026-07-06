from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config.embedding import resolve_embedding_extraction_config


def test_embedding_config_resolves_from_stage_module() -> None:
	resolved = resolve_embedding_extraction_config(_minimal_embedding_config())

	assert resolved['stage'] == 'extract_embeddings'
	assert resolved['embeddings']['checkpoint'].endswith('mae_latest.pt')
	assert resolved['embedding']['output_dtype'] == 'float16'


def test_embedding_config_rejects_checkpoint_owned_sections() -> None:
	cfg = _minimal_embedding_config()
	cfg['data'] = {'local_crop_size': [128, 128, 128]}

	with pytest.raises(ValueError, match=r'checkpoint-owned.*data'):
		resolve_embedding_extraction_config(cfg)


def test_embedding_config_rejects_runs_checkpoint_path() -> None:
	cfg = _minimal_embedding_config()
	cfg['embeddings']['checkpoint'] = (
		'/artifacts/runs/nopims/pretrain_v1/amp_mae_v1/full_100ep/'
		'mae_latest.pt'
	)

	with pytest.raises(ValueError, match='runs/ paths'):
		resolve_embedding_extraction_config(cfg)


def test_embedding_config_enforces_checkpoint_pretraining_path_contract() -> None:
	cfg = _minimal_embedding_config()
	cfg['embeddings']['checkpoint'] = (
		'/artifacts/pretraining/nopims/pretrain_v1/amp_mae_v1/mae_latest.pt'
	)

	with pytest.raises(ValueError, match=r'pretraining/nopims/pretrain_v1'):
		resolve_embedding_extraction_config(cfg)


def test_embedding_config_enforces_output_artifact_path_contract() -> None:
	cfg = _minimal_embedding_config()
	cfg['embeddings']['output_dir'] = (
		'/artifacts/embeddings/nopims/pretrain_v1/amp_mae_v1/full'
	)

	with pytest.raises(ValueError, match=r'embeddings/nopims/pretrain_v1'):
		resolve_embedding_extraction_config(cfg)


def test_embedding_config_validates_window_overlap() -> None:
	cfg = _minimal_embedding_config()
	cfg['embedding']['window_size'] = [8, 8, 8]
	cfg['embedding']['overlap'] = [4, 8, 4]

	with pytest.raises(ValueError, match=r'embedding\.overlap.*window_size'):
		resolve_embedding_extraction_config(cfg)


def _minimal_embedding_config() -> dict[str, object]:
	return deepcopy(
		{
			'paths': {'artifact_root': '/artifacts'},
			'manifests': {'input': '/artifacts/manifests/train.json'},
			'embeddings': {
				'checkpoint': (
					'/artifacts/pretraining/train_amp_mae/mae_latest.pt'
				),
				'output_dir': '/artifacts/embeddings',
			},
			'embedding': {
				'window_size': [128, 128, 128],
				'overlap': [64, 64, 64],
				'output_dtype': 'float16',
				'batch_size': 1,
				'min_token_valid_fraction': 0.5,
			},
		},
	)
