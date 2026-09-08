"""Ensure the four additional Volve arms preserve the comparison budgets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	plan_volve_horizon_recipe_arm_jobs,
	volve_horizon_recipe_arm_config_from_mapping,
)

ROOT = Path('experiments/volve/horizon_benchmark_v1')
EXPERIMENT = ROOT / '33_missing_hmm_comparison_v1'
REFERENCE = ROOT / '31_mae_local_bt_hmm_five_way_v1'
ARMS = tuple(
	f'{parent}_hmm_k6_distill{weight}'
	for parent in ('rot90_asym_g060_3ep', 'random_init')
	for weight in ('010', '020')
)


@pytest.fixture(autouse=True)
def artifact_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""Resolve every config against portable paths."""
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_VOLVE_ROOT', str(tmp_path / 'public'))


def test_all_four_arms_have_every_stage() -> None:
	for stage in ('20_pretraining', '30_embeddings', '40_downstream'):
		assert {path.stem for path in (EXPERIMENT / stage).glob('*.yaml')} == set(ARMS)


@pytest.mark.parametrize('arm', ARMS)
def test_hmm_arms_preserve_the_existing_volve_training_and_decoder_budget(
	arm: str,
) -> None:
	train = load_config(EXPERIMENT / '20_pretraining' / f'{arm}.yaml')
	reference = deepcopy(
		load_config(REFERENCE / '60_distill010/30_stage2/local_bt100_hmm_k6_25ep.yaml')
	)
	for section in ('paths', 'identity', 'pseudo_targets', 'teacher', 'student'):
		reference[section] = deepcopy(train[section])
	reference['loss']['distillation_weight'] = 0.1 if arm.endswith('010') else 0.2
	reference['train']['num_workers'] = 4
	assert train == reference
	assert train['paths']['output_root'].endswith(
		f'/missing_hmm_comparison_v1/{arm}/full_25ep'
	)
	assert train['teacher']['checkpoint'] == train['student']['init_checkpoint']
	assert train['student']['unfreeze_top_blocks'] == 1
	assert (
		train['train']['epochs']
		* train['train']['samples_per_epoch']
		// train['train']['batch_size']
		== 62500
	)
	downstream = load_config(EXPERIMENT / '40_downstream' / f'{arm}.yaml')
	baseline = load_config(
		ROOT / '32_local_bt_recipe_arm_v1/30_downstream/rot90_asym_g060_3ep.yaml'
	)
	for section in (
		'paths',
		'dataset',
		'inputs',
		'baseline',
		'decoder',
		'tiles',
		'train',
	):
		assert downstream[section] == baseline[section]
	assert downstream['outputs']['runs_root'] == baseline['outputs']['runs_root']
	hmm = downstream['arm']['hmm']
	assert hmm['init_checkpoint'] == train['student']['init_checkpoint']
	assert hmm['pseudo_targets_dir'] == train['pseudo_targets']['input_dir']
	assert hmm['distillation_weight'] == train['loss']['distillation_weight']
	cluster = load_config(Path(hmm['clustering_config']))
	reference_cluster = load_config(
		REFERENCE / '20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml'
	)
	reference_cluster['embeddings'] = cluster['embeddings']
	reference_cluster['clustering']['output_dir'] = cluster['clustering']['output_dir']
	assert cluster == reference_cluster
	assert cluster['embeddings']['input_dir'] == hmm['source_embeddings_dir']
	embedding = load_config(EXPERIMENT / '30_embeddings' / f'{arm}.yaml')
	assert (
		downstream['arm']['checkpoint']
		== embedding['embeddings']['checkpoint']
		== train['paths']['output_root'] + '/latest.pt'
	)
	assert downstream['arm']['embeddings_dir'] == embedding['embeddings']['output_dir']
	assert downstream['arm']['embeddings_dir'].endswith(
		f'/missing_hmm_comparison_v1/{arm}/overlap_x64'
	)
	config = volve_horizon_recipe_arm_config_from_mapping(downstream)
	assert len(plan_volve_horizon_recipe_arm_jobs(config, model_ids=(arm,))) == 15
	if arm.startswith('random_init'):
		assert config.recipe.method == 'random_encoder'
		assert config.recipe.epochs == 0
	else:
		assert config.recipe.epochs == 3
		assert config.recipe.augmentations['gaussian_noise_std'] == 0.6
