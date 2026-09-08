from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

EXPERIMENT = Path(
	'experiments/f3/facies_benchmark_v2/'
	'126_pretraining_comparison_completion_v1'
)
REFERENCE = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1/'
	'30_stage2/mae100/hmm/k6/02_full_25ep.yaml'
)
ARMS = (
	(
		'mae100_hmm_k6_distill010',
		0.1,
		'ssl_hmm_continuation_v1/stage1/mae/full_100ep/latest.pt',
		'ssl_hmm_continuation_v1/mae100',
	),
	(
		'local_bt_rot90_asym_g060_3ep_hmm_k6_distill010',
		0.1,
		'local_bt_noise_rotation_search_v1/rot90_asym_g060/full_3ep/latest.pt',
		'local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep',
	),
	(
		'random_self_target_hmm_k6_distill020',
		0.2,
		'mae_local_bt_five_way_v1/random/random_init.pt',
		'random_init_self_target_hmm_k6_distill010_v1/random_init',
	),
)


def _read(path: Path) -> dict:
	return yaml.safe_load(path.read_text(encoding='utf-8'))


@pytest.mark.parametrize(('arm', 'weight', 'source', 'target'), ARMS)
def test_hmm_budget_and_source_lineage(
	arm: str, weight: float, source: str, target: str
) -> None:
	full = _read(EXPERIMENT / '10_pretraining' / arm / '02_full_25ep.yaml')
	assert full['teacher']['checkpoint'] == full['student']['init_checkpoint']
	assert full['teacher']['checkpoint'].endswith('/' + source)
	assert full['pseudo_targets']['input_dir'].endswith('/' + target)
	assert full['loss']['distillation_weight'] == weight
	assert full['identity']['scientific_identity']['distillation_weight'] == weight
	assert full['identity']['scientific_identity']['pseudo_target_source'] == (
		target + '/k6'
	)
	# All settings other than explicit lineage, loss, and resource/output changes
	# must retain the already completed canonical 25-epoch HMM contract.
	expected = _read(REFERENCE)
	normalized = deepcopy(full)
	normalized.pop('identity')
	for key in ('paths', 'teacher', 'student', 'pseudo_targets'):
		normalized[key] = expected[key]
	normalized['loss']['distillation_weight'] = expected['loss'][
		'distillation_weight'
	]
	normalized['train']['num_workers'] = expected['train']['num_workers']
	assert normalized == expected


@pytest.mark.parametrize('arm', [arm[0] for arm in ARMS])
def test_smoke_cannot_replace_full_budget(arm: str) -> None:
	full = _read(EXPERIMENT / '10_pretraining' / arm / '02_full_25ep.yaml')
	smoke = _read(EXPERIMENT / '10_pretraining' / arm / '01_smoke.yaml')
	assert smoke['paths']['output_root'] != full['paths']['output_root']
	assert smoke['train']['max_steps'] == 1
	assert smoke['train']['epochs'] == 1
	assert full['train']['epochs'] == 25
	assert full['train']['samples_per_epoch'] == 10_000
	assert full['train']['batch_size'] == 16
	smoke['paths'] = full['paths']
	for key in ('epochs', 'samples_per_epoch', 'num_workers', 'max_steps'):
		smoke['train'][key] = full['train'][key]
	assert smoke == full


@pytest.mark.parametrize('arm', [arm[0] for arm in ARMS])
def test_embedding_and_candidate_lineage(arm: str) -> None:
	full = _read(EXPERIMENT / '10_pretraining' / arm / '02_full_25ep.yaml')
	embedding = _read(EXPERIMENT / '20_embeddings' / f'{arm}.yaml')
	candidate = _read(EXPERIMENT / '30_downstream' / f'{arm}.yaml')
	checkpoint = full['paths']['output_root'] + '/latest.pt'
	assert embedding['embeddings']['checkpoint'] == checkpoint
	assert candidate['candidate']['checkpoint'] == checkpoint
	assert candidate['candidate']['embeddings_dir'] == embedding['embeddings'][
		'output_dir'
	]
	assert candidate['candidate']['id'] == arm
	assert candidate['benchmark']['canonical_config'].endswith(
		'/110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
	)
	assert embedding['embedding']['window_size'] == [128, 128, 128]
	assert embedding['embedding']['overlap'] == [64, 64, 64]
	assert embedding['embedding']['amp'] is False
	assert embedding['embedding']['min_token_valid_fraction'] == 0.5
	assert embedding['manifests']['input'].endswith(
		'/facies_benchmark_v2/f3_amplitude_manifest.json'
	)
