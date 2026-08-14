from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.f3.lithology.guardrails import (
	F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG,
	F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME,
	F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
	f3_guardrail_jobs_config_from_mapping,
	f3_shuffled_hmm_target_config_from_mapping,
)
from seis_ssl_cluster.stratigraphy import shuffle_strat_hmm_pseudo_targets
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_pretraining import (
	run_strat_hmm_pretext_training,
)
from tests.seis_ssl_cluster.test_strat_hmm_pretraining_head_only import (
	_raw_config,
)

ROOT = Path('experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails')


@pytest.mark.parametrize(
	'filename',
	[
		'01_train_distillation_only_smoke.yaml',
		'02_train_distillation_only_full.yaml',
		'07_train_shuffled_hmm_smoke.yaml',
		'08_train_shuffled_hmm_full.yaml',
	],
)
def test_guardrail_training_configs_resolve(
	filename: str,
	tmp_path: Path,
) -> None:
	raw = load_config(ROOT / filename)
	artifact_root = tmp_path / 'artifacts'
	pseudo_targets = tmp_path / 'pseudo_targets'
	pseudo_targets.mkdir()
	checkpoint = tmp_path / 'mae.pt'
	checkpoint.touch()
	raw['paths']['artifact_root'] = str(artifact_root)
	raw['paths']['output_root'] = str(
		artifact_root / 'pretraining' / 'f3' / Path(filename).stem,
	)
	raw['pseudo_targets']['input_dir'] = str(pseudo_targets)
	raw['teacher']['checkpoint'] = str(checkpoint)
	raw['student']['init_checkpoint'] = str(checkpoint)

	resolved = resolve_strat_hmm_pretext_config(raw)

	if 'distillation_only' in filename:
		assert resolved['loss']['prototype_weight'] == 0.0
		assert resolved['loss']['usage_weight'] == 0.0
	assert resolved['student']['unfreeze_top_blocks'] == 1
	assert resolved['loss']['distillation_weight'] == 0.2


def test_shuffled_hmm_full_training_only_changes_targets_and_output() -> None:
	candidate = load_config(
		Path('experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1')
		/ '03_train_single_head_topblock_distill_full.yaml',
	)
	guardrail = load_config(ROOT / '08_train_shuffled_hmm_full.yaml')
	candidate['paths']['output_root'] = guardrail['paths']['output_root']
	candidate['pseudo_targets']['input_dir'] = guardrail['pseudo_targets']['input_dir']

	assert guardrail == candidate
	assert guardrail['paths']['output_root'].endswith(
		F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
	)


def test_shuffled_target_contract_resolves_with_preservation_guarantees() -> None:
	config = f3_shuffled_hmm_target_config_from_mapping(
		load_config(ROOT / '03_build_shuffled_hmm_pseudo_targets.yaml'),
	)

	assert config.suite_name == F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME
	assert config.seed == 42
	assert config.shuffle_scope == 'global_valid_tokens'
	assert config.source_root != config.output_root
	assert config.overwrite is False


def test_training_accepts_generated_shuffled_pseudo_target_artifacts(
	tmp_path: Path,
) -> None:
	raw = _raw_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.2,
		max_steps=1,
	)
	source_root = Path(raw['pseudo_targets']['input_dir'])
	shuffled_root = tmp_path / 'strat_hmm_m1_guardrail_shuffled_hmm'
	shuffle_strat_hmm_pseudo_targets(
		source_root,
		shuffled_root,
		k=3,
		seed=42,
	)
	raw['pseudo_targets']['input_dir'] = str(shuffled_root)
	config = resolve_strat_hmm_pretext_config(raw)

	checkpoint_path = run_strat_hmm_pretext_training(config)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	assert payload['global_step'] == 1
	assert payload['stratigraphy_config']['pseudo_targets']['input_dir'] == str(
		shuffled_root,
	)


def test_guardrail_embedding_jobs_keep_isolated_model_roots() -> None:
	config = f3_guardrail_jobs_config_from_mapping(
		load_config(ROOT / '05_extract_guardrail_embeddings.yaml')
	)

	assert tuple(job.model_tag for job in config.jobs) == (
		F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG,
		F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
	)
	assert len({job.output_path for job in config.jobs}) == 2
