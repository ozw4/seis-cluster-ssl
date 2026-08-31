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
	resolve_barlow_twins_training_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)

EXPERIMENT_ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/31_mae_local_bt_hmm_five_way_v1'
)
TARGET_ROOT = EXPERIMENT_ROOT / '20_hmm_targets/local_bt100'
MAE_TARGET_ROOT = EXPERIMENT_ROOT / '20_hmm_targets/mae100'
STAGE2_ROOT = EXPERIMENT_ROOT / '30_stage2/local_bt100/hmm/k6'
MAE_STAGE2_ROOT = EXPERIMENT_ROOT / '30_stage2/mae100/hmm/k6'
STAGE1 = EXPERIMENT_ROOT / '10_stage1/local_barlow_twins/02_full_100ep.yaml'
MAE_EMBEDDING = Path(
	'experiments/volve/horizon_benchmark_v1/30_mae_vs_random_frozen_v1/'
	'01_extract_pretrained_embeddings.yaml'
)
RUNS = ('01_smoke_2step.yaml', '02_full_25ep.yaml')


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	checkpoint = (
		root / 'pretraining/volve/horizon_benchmark_v1/'
		'mae_local_bt_hmm_five_way_v1/stage1/local_bt/full_100ep/latest.pt'
	)
	checkpoint.parent.mkdir(parents=True)
	checkpoint.touch()
	(
		root / 'pseudo_targets/volve/horizon_benchmark_v1/'
		'mae_local_bt_hmm_five_way_v1/local_bt100'
	).mkdir(parents=True)
	return root


def _stage2(filename: str) -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(STAGE2_ROOT / filename))


def test_local_target_extraction_uses_non_trace_drop_local_bt100() -> None:
	stage1 = resolve_barlow_twins_training_config(load_config(STAGE1))
	extraction = resolve_embedding_extraction_config(
		load_config(TARGET_ROOT / '01_extract_embeddings.yaml')
	)

	assert stage1['barlow_twins']['method'] == 'local_barlow_twins_3d'
	assert stage1['barlow_twins']['local_pairs_per_crop'] == 128
	assert extraction['embeddings']['checkpoint'] == (
		f'{stage1["paths"]["output_root"]}/latest.pt'
	)
	assert 'trace_drop' not in repr(extraction).lower()
	assert extraction['embeddings']['output_dir'].endswith(
		'/mae_local_bt_hmm_five_way_v1/hmm_targets/local_bt100/overlap_x64'
	)


def test_local_and_mae_target_extractions_share_support_contract() -> None:
	local = load_config(TARGET_ROOT / '01_extract_embeddings.yaml')
	mae = load_config(MAE_EMBEDDING)

	assert local['paths'] == mae['paths']
	assert local['manifests'] == mae['manifests']
	assert local['embedding'] == mae['embedding']
	assert local['embeddings']['checkpoint'] != mae['embeddings']['checkpoint']
	assert local['embeddings']['output_dir'] != mae['embeddings']['output_dir']


def test_local_and_mae_hmm_clustering_science_is_identical() -> None:
	local = resolve_clustering_config(
		load_config(TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml')
	)
	mae = resolve_clustering_config(
		load_config(MAE_TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml')
	)

	assert local['embeddings']['input_dir'] != mae['embeddings']['input_dir']
	assert local['clustering']['output_dir'] != mae['clustering']['output_dir']
	assert {
		key: value for key, value in local['clustering'].items() if key != 'output_dir'
	} == {key: value for key, value in mae['clustering'].items() if key != 'output_dir'}
	assert local['clustering']['k_values'] == [6]
	assert local['clustering']['stratigraphic_hmm']['z_axis'] == 2


def test_local_export_matches_mae_contract_but_has_separate_root() -> None:
	local_path = TARGET_ROOT / 'k6/03_export_pseudo_targets.sh'
	mae_path = MAE_TARGET_ROOT / 'k6/03_export_pseudo_targets.sh'
	local = local_path.read_text(encoding='utf-8')
	mae = mae_path.read_text(encoding='utf-8')

	assert local_path.stat().st_mode & stat.S_IXUSR
	for flag in (
		'--k 6',
		'--confidence 1.0',
		'--boundary-alpha 0.0',
		'--boundary-tau 1.0',
		'--schema-version 2',
	):
		assert flag in local
		assert flag in mae
	assert '/local_bt100' in local
	assert '/mae100' in mae
	assert local != mae


@pytest.mark.parametrize('filename', RUNS)
def test_local_hmm_stage2_uses_local_bt100_teacher_student(filename: str) -> None:
	config = _stage2(filename)
	stage1 = resolve_barlow_twins_training_config(load_config(STAGE1))
	expected = f'{stage1["paths"]["output_root"]}/latest.pt'

	assert config['teacher']['checkpoint'] == expected
	assert config['student']['init_checkpoint'] == expected
	assert config['student']['unfreeze_top_blocks'] == 1
	assert config['pseudo_targets']['input_dir'].endswith(
		'/mae_local_bt_hmm_five_way_v1/local_bt100'
	)
	assert config['pseudo_targets']['k'] == 6
	assert config['head']['num_prototypes'] == 6
	assert config['identity']['model_tag'] == 'volve_local_bt100_hmm_k6_topblock1_v1'
	assert (
		'trace_drop' not in (STAGE2_ROOT / filename).read_text(encoding='utf-8').lower()
	)


def test_local_and_mae_hmm_stage2_share_head_loss_and_budget() -> None:
	local = load_config(STAGE2_ROOT / RUNS[1])
	mae = load_config(MAE_STAGE2_ROOT / RUNS[1])

	assert local['data'] == mae['data']
	assert local['zero_mask'] == mae['zero_mask']
	assert local['model'] == mae['model']
	assert local['head'] == mae['head']
	assert local['loss'] == mae['loss']
	assert local['train'] == mae['train']
	train = local['train']
	assert train['batch_size'] == 4
	assert train['epochs'] * train['samples_per_epoch'] // 4 == 62_500
	assert train['lr'] == train['encoder_lr'] == 1.0e-5


def test_local_hmm_smoke_differs_only_by_output_and_two_step_limit() -> None:
	smoke = load_config(STAGE2_ROOT / RUNS[0])
	full = load_config(STAGE2_ROOT / RUNS[1])
	expected = deepcopy(full)
	expected['paths']['output_root'] = smoke['paths']['output_root']
	expected['train']['max_steps'] = 2

	assert smoke == expected
	assert smoke['paths']['output_root'].endswith('/local_bt100/hmm/k6/smoke_2step')
	assert full['paths']['output_root'].endswith('/local_bt100/hmm/k6/full_25ep')


def test_local_target_and_stage2_configs_exclude_disallowed_sources() -> None:
	for path in (
		TARGET_ROOT / '01_extract_embeddings.yaml',
		TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml',
		TARGET_ROOT / 'k6/03_export_pseudo_targets.sh',
		*(STAGE2_ROOT / filename for filename in RUNS),
	):
		text = path.read_text(encoding='utf-8').lower()
		for forbidden in (
			'trace_drop',
			'seis_ssl_cluster_volve_root',
			'20_horizon_supervision',
			'layout_000',
		):
			assert forbidden not in text


def test_local_hmm_dry_runs_write_nothing(artifact_root: Path) -> None:
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	before = {str(path) for path in artifact_root.rglob('*')}
	for script, config in (
		(
			'proc/seis_ssl_cluster/extract_embeddings.py',
			TARGET_ROOT / '01_extract_embeddings.yaml',
		),
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
