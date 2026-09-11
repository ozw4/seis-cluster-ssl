from __future__ import annotations

import os
import stat
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_mae_training_config,
	resolve_strat_hmm_pretext_config,
)

EXPERIMENT_ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/31_mae_local_bt_hmm_five_way_v1'
)
TARGET_ROOT = EXPERIMENT_ROOT / '20_hmm_targets/mae100'
STAGE2_ROOT = EXPERIMENT_ROOT / '30_stage2/mae100/hmm/k6'
MAE_EMBEDDING = Path(
	'experiments/volve/horizon_benchmark_v1/30_mae_vs_random_frozen_v1/'
	'01_extract_pretrained_embeddings.yaml'
)
MAE_CONTINUATION = EXPERIMENT_ROOT / '30_stage2/mae100/mae_continue/02_full_25ep.yaml'
REFERENCE_CLUSTER = Path(
	'experiments/parihaka/facies_benchmark_v1/21_ssl_hmm_continuation_v1/'
	'20_hmm_targets/mae100/k6/02_cluster_hmm_k6.yaml'
)
RUNS = ('01_smoke_2step.yaml', '02_full_25ep.yaml')


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	checkpoint = (
		root / 'pretraining/volve/horizon_benchmark_v1/'
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/'
		'full_100ep/latest.pt'
	)
	checkpoint.parent.mkdir(parents=True)
	checkpoint.touch()
	(
		root / 'pseudo_targets/volve/horizon_benchmark_v1/'
		'mae_local_bt_hmm_five_way_v1/mae100'
	).mkdir(parents=True)
	return root


def _stage2(filename: str) -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(STAGE2_ROOT / filename))


def test_mae_target_reuses_existing_mae100_embedding() -> None:
	embedding = resolve_embedding_extraction_config(load_config(MAE_EMBEDDING))
	clustering = resolve_clustering_config(
		load_config(TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml')
	)

	assert embedding['embeddings']['checkpoint'].endswith(
		'/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/latest.pt'
	)
	assert (
		clustering['embeddings']['input_dir'] == embedding['embeddings']['output_dir']
	)
	assert embedding['embedding']['window_size'] == [128, 128, 128]
	assert embedding['embedding']['overlap'] == [64, 64, 64]
	assert embedding['embedding']['min_token_valid_fraction'] == 1.0
	assert not (TARGET_ROOT / '01_extract_embeddings.yaml').exists()


def test_mae_target_uses_adopted_depth_hmm_k6_contract() -> None:
	config = load_config(TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml')
	reference = load_config(REFERENCE_CLUSTER)
	clustering = config['clustering']

	assert {key: value for key, value in clustering.items() if key != 'output_dir'} == {
		key: value
		for key, value in reference['clustering'].items()
		if key != 'output_dir'
	}
	assert clustering['method'] == 'stratigraphic_hmm_kmeans'
	assert clustering['k_values'] == [6]
	assert clustering['seed'] == 42
	hmm = clustering['stratigraphic_hmm']
	assert hmm['z_axis'] == 2
	assert hmm['z_direction'] == 'increasing_downward'
	assert hmm['update']['empty_cluster_policy'] == 'keep_previous'
	assert 'local_bt100' not in repr(config)


def test_mae_export_is_k6_and_isolated() -> None:
	path = TARGET_ROOT / 'k6/03_export_pseudo_targets.sh'
	text = path.read_text(encoding='utf-8')

	assert path.stat().st_mode & stat.S_IXUSR
	assert '--k 6' in text
	assert '--confidence 1.0' in text
	assert '--boundary-alpha 0.0' in text
	assert '--boundary-tau 1.0' in text
	assert '--schema-version 2' in text
	assert '/mae_local_bt_hmm_five_way_v1/mae100' in text
	assert 'local_bt100' not in text
	for forbidden in ('SEIS_SSL_CLUSTER_VOLVE_ROOT', 'layout', 'supervision'):
		assert forbidden not in text


@pytest.mark.parametrize('filename', RUNS)
def test_mae_hmm_stage2_uses_mae100_teacher_student_and_k6(filename: str) -> None:
	config = _stage2(filename)

	assert config['teacher']['checkpoint'] == config['student']['init_checkpoint']
	assert config['teacher']['checkpoint'].endswith(
		'/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/latest.pt'
	)
	assert config['student']['unfreeze_top_blocks'] == 1
	assert config['pseudo_targets']['k'] == 6
	assert config['pseudo_targets']['min_confidence'] == 0.0
	assert config['head'] == {
		'num_prototypes': 6,
		'projection_dim': 128,
		'temperature': 0.1,
		'normalize': True,
	}
	assert config['loss']['prototype_weight'] == 1.0
	assert config['loss']['usage_weight'] == 0.005
	assert config['loss']['distillation_weight'] == 0.2
	assert config['identity']['model_tag'] == 'volve_mae100_hmm_k6_topblock1_v1'


def test_mae_hmm_budget_matches_plain_mae_continuation() -> None:
	hmm_train = _stage2(RUNS[1])['train']
	mae_train = resolve_mae_training_config(load_config(MAE_CONTINUATION))['train']

	for key in (
		'batch_size',
		'samples_per_epoch',
		'epochs',
		'lr',
		'weight_decay',
		'seed',
	):
		assert hmm_train[key] == mae_train[key]
	assert hmm_train['encoder_lr'] == 1.0e-5
	assert hmm_train['epochs'] * hmm_train['samples_per_epoch'] // 4 == 62_500
	assert hmm_train['max_steps'] is None


def test_mae_hmm_smoke_differs_only_by_output_and_two_step_limit() -> None:
	smoke = load_config(STAGE2_ROOT / RUNS[0])
	full = load_config(STAGE2_ROOT / RUNS[1])
	expected = deepcopy(full)
	expected['paths']['output_root'] = smoke['paths']['output_root']
	expected['train']['max_steps'] = 2

	assert smoke == expected
	assert smoke['paths']['output_root'].endswith('/mae100/hmm/k6/smoke_2step')
	assert full['paths']['output_root'].endswith('/mae100/hmm/k6/full_25ep')


def test_mae_hmm_configs_exclude_horizon_supervision() -> None:
	for path in (
		TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml',
		TARGET_ROOT / 'k6/03_export_pseudo_targets.sh',
		*(STAGE2_ROOT / filename for filename in RUNS),
	):
		text = path.read_text(encoding='utf-8')
		for forbidden in (
			'SEIS_SSL_CLUSTER_VOLVE_ROOT',
			'20_horizon_supervision',
			'layout_000',
		):
			assert forbidden not in text


def test_mae_hmm_dry_runs_write_nothing(artifact_root: Path) -> None:
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	before = {str(path) for path in artifact_root.rglob('*')}
	for script, config in (
		(
			'proc/seis_ssl_cluster/cluster_embeddings.py',
			TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml',
		),
		(
			'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
			STAGE2_ROOT / RUNS[0],
		),
	):
		result = subprocess.run(  # noqa: S603
			[sys.executable, script, '--config', str(config), '--dry-run'],
			check=True,
			capture_output=True,
			text=True,
			env=environment,
		)
		assert 'dry-run' in result.stdout
	assert {str(path) for path in artifact_root.rglob('*')} == before
