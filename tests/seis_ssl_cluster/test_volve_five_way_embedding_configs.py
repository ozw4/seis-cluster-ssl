from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config, resolve_embedding_extraction_config

if TYPE_CHECKING:
	import pytest

EXPERIMENT_ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/31_mae_local_bt_hmm_five_way_v1'
)
MODEL_CONFIGS = {
	'mae': '01_mae.yaml',
	'mae_hmm_k6': '02_mae_hmm_k6.yaml',
	'local_barlow_twins': '03_local_barlow_twins.yaml',
	'local_barlow_twins_hmm_k6': '04_local_barlow_twins_hmm_k6.yaml',
	'random': '05_random.yaml',
}


def test_five_way_embedding_configs_share_the_volve_extraction_contract(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	resolved = {
		model_id: resolve_embedding_extraction_config(
			load_config(EXPERIMENT_ROOT / '40_embeddings' / filename)
		)
		for model_id, filename in MODEL_CONFIGS.items()
	}
	reference = resolved['mae']
	for model_id, config in resolved.items():
		assert config['manifests'] == reference['manifests']
		assert config['embedding'] == reference['embedding']
		assert config['embedding']['window_size'] == [128, 128, 128]
		assert config['embedding']['overlap'] == [64, 64, 64]
		assert config['embedding']['output_dtype'] == 'float16'
		assert config['embedding']['batch_size'] == 1
		assert config['embedding']['amp'] is True
		assert config['embedding']['min_token_valid_fraction'] == 1.0
		assert 'trace' + '_drop' not in str(config).lower(), model_id


def test_five_way_embedding_configs_bind_exact_sources_and_outputs(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	resolved = {
		model_id: resolve_embedding_extraction_config(
			load_config(EXPERIMENT_ROOT / '40_embeddings' / filename)
		)
		for model_id, filename in MODEL_CONFIGS.items()
	}
	learned = tuple(model_id for model_id in MODEL_CONFIGS if model_id != 'random')
	outputs = {
		resolved[model_id]['embeddings']['output_dir'] for model_id in learned
	}
	assert len(outputs) == 4
	for model_id in learned:
		checkpoint = resolved[model_id]['embeddings']['checkpoint']
		output_dir = resolved[model_id]['embeddings']['output_dir']
		assert checkpoint.endswith('full_25ep/latest.pt')
		assert f'/{model_id}/overlap_x64' in output_dir
	random = resolved['random']['embeddings']
	assert random['checkpoint'].endswith('random_init/mae_random_seed42.pt')
	assert random['output_dir'].endswith(
		'random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/'
		'overlap_x64'
	)
