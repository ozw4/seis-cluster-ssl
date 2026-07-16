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


def test_embedding_config_accepts_preprocessing_cache_policy() -> None:
	cfg = _minimal_embedding_config()
	cfg['embedding']['preprocessing_cache'] = {
		'mode': 'memmap',
		'chunk_size_x': 8,
		'reuse': True,
		'cleanup': False,
	}

	resolved = resolve_embedding_extraction_config(cfg)

	assert resolved['embedding']['preprocessing_cache']['mode'] == 'memmap'


def test_embedding_config_accepts_average_chunk_size() -> None:
	cfg = _minimal_embedding_config()
	cfg['embedding']['average_chunk_size_x'] = 7

	resolved = resolve_embedding_extraction_config(cfg)

	assert resolved['embedding']['average_chunk_size_x'] == 7


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_embedding_config_rejects_invalid_average_chunk_size(value: object) -> None:
	cfg = _minimal_embedding_config()
	cfg['embedding']['average_chunk_size_x'] = value

	with pytest.raises((TypeError, ValueError), match='average_chunk_size_x'):
		resolve_embedding_extraction_config(cfg)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('mode', 'disk'),
		('chunk_size_x', 0),
		('reuse', 1),
		('cleanup', 'false'),
		('unknown', True),
	],
)
def test_embedding_config_rejects_invalid_preprocessing_cache(
	key: str,
	value: object,
) -> None:
	cfg = _minimal_embedding_config()
	cfg['embedding']['preprocessing_cache'] = {key: value}

	with pytest.raises((TypeError, ValueError), match='preprocessing_cache'):
		resolve_embedding_extraction_config(cfg)


@pytest.mark.parametrize(
	('key', 'value', 'error'),
	[
		('prefetch_queue_depth', -1, 'prefetch_queue_depth'),
		('amp', 1, r'embedding\.amp'),
		('amp_dtype', 'float32', r'embedding\.amp_dtype'),
		('stage_timing', 'yes', r'embedding\.stage_timing'),
	],
)
def test_embedding_config_validates_prefetch_and_precision_options(
	key: str,
	value: object,
	error: str,
) -> None:
	cfg = _minimal_embedding_config()
	cfg['embedding'][key] = value

	with pytest.raises((TypeError, ValueError), match=error):
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
