'''Contract tests for the Volve horizon pretraining recipe-arm benchmark.'''

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_MODEL_IDS,
	LOCAL_BARLOW_TWINS_METHOD,
)
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	RECIPE_ARM_BASELINE_MODEL_ID,
	RECIPE_ARM_BENCHMARK_ID,
	RECIPE_ARM_CONDITION_COUNT,
	as_five_way_config,
	audit_volve_horizon_recipe_arm_sources,
	inspect_volve_horizon_recipe_arm_embedding_suite,
	plan_volve_horizon_recipe_arm_jobs,
	plan_volve_horizon_recipe_arm_sources,
	resolve_volve_horizon_recipe_arm_job,
	volve_horizon_recipe_arm_config_from_mapping,
)
from tests.seis_ssl_cluster.helpers_volve_recipe_arm import (
	ARM_GLOBAL_STEPS,
	ARM_ID,
	arm_checkpoint_payload,
	baseline_checkpoint_payload,
	recipe_arm_config_mapping,
	write_recipe_arm_universe,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_recipe_arm_config_resolves_the_declared_single_stage_recipe(
	tmp_path: Path,
) -> None:
	config = volve_horizon_recipe_arm_config_from_mapping(
		recipe_arm_config_mapping(tmp_path)
	)
	assert config.arm_id == ARM_ID
	assert config.model_ids == (ARM_ID, RECIPE_ARM_BASELINE_MODEL_ID)
	assert config.benchmark_id == RECIPE_ARM_BENCHMARK_ID
	assert config.recipe.method == LOCAL_BARLOW_TWINS_METHOD
	assert config.recipe.steps_per_epoch == 625
	assert config.recipe.global_steps == ARM_GLOBAL_STEPS
	assert config.recipe.positive_window_tokens == (2, 2, 1)


def test_recipe_arm_config_keeps_the_frozen_downstream_contract(
	tmp_path: Path,
) -> None:
	raw = recipe_arm_config_mapping(tmp_path)
	config = volve_horizon_recipe_arm_config_from_mapping(raw)
	assert config.train.epochs == 50
	assert config.train.batch_size == 1
	assert config.train.seed == 42000
	assert config.tiles.core_size_tokens == (8, 8, 27)
	assert config.tiles.context_halo_tokens == (1, 1, 0)


def test_recipe_arm_config_rejects_an_unknown_top_level_key(
	tmp_path: Path,
) -> None:
	raw = recipe_arm_config_mapping(tmp_path)
	raw['models'] = {}
	with pytest.raises(ValueError, match='config keys differ'):
		volve_horizon_recipe_arm_config_from_mapping(raw)


def test_recipe_arm_config_rejects_the_baseline_model_id_as_the_arm(
	tmp_path: Path,
) -> None:
	raw = recipe_arm_config_mapping(tmp_path)
	arm = raw['arm']
	assert isinstance(arm, dict)
	arm['arm_id'] = RECIPE_ARM_BASELINE_MODEL_ID
	with pytest.raises(ValueError, match='must differ'):
		volve_horizon_recipe_arm_config_from_mapping(raw)


def test_recipe_arm_config_rejects_a_shared_arm_and_baseline_source(
	tmp_path: Path,
) -> None:
	raw = recipe_arm_config_mapping(tmp_path)
	arm = raw['arm']
	baseline = raw['baseline']
	assert isinstance(arm, dict)
	assert isinstance(baseline, dict)
	baseline['checkpoint'] = arm['checkpoint']
	with pytest.raises(ValueError, match='checkpoint paths must differ'):
		volve_horizon_recipe_arm_config_from_mapping(raw)


def test_recipe_arm_config_rejects_a_non_local_barlow_twins_recipe(
	tmp_path: Path,
) -> None:
	raw = recipe_arm_config_mapping(tmp_path)
	arm = raw['arm']
	assert isinstance(arm, dict)
	arm['recipe']['method'] = 'amp_mae3d'
	with pytest.raises(ValueError, match=r'arm\.recipe\.method'):
		volve_horizon_recipe_arm_config_from_mapping(raw)


def test_recipe_arm_config_rejects_a_partial_batch_epoch(tmp_path: Path) -> None:
	raw = recipe_arm_config_mapping(tmp_path)
	arm = raw['arm']
	assert isinstance(arm, dict)
	arm['recipe']['samples_per_epoch'] = 10_001
	with pytest.raises(ValueError, match='whole number of batches'):
		volve_horizon_recipe_arm_config_from_mapping(raw)


def test_recipe_arm_source_plan_names_the_arm_and_the_baseline(
	tmp_path: Path,
) -> None:
	config = volve_horizon_recipe_arm_config_from_mapping(
		recipe_arm_config_mapping(tmp_path)
	)
	plan = plan_volve_horizon_recipe_arm_sources(config)
	assert [row['model_id'] for row in plan] == [
		ARM_ID,
		RECIPE_ARM_BASELINE_MODEL_ID,
	]
	assert plan[0]['role'] == 'recipe_arm'
	assert plan[0]['recipe']['stages'] == 1
	assert plan[0]['recipe']['global_steps'] == ARM_GLOBAL_STEPS


def test_recipe_arm_audit_accepts_the_declared_checkpoint(tmp_path: Path) -> None:
	universe = write_recipe_arm_universe(tmp_path, embeddings=False)
	report = audit_volve_horizon_recipe_arm_sources(universe['config'])
	assert report['model_order'] == [ARM_ID, RECIPE_ARM_BASELINE_MODEL_ID]
	sources = report['sources']
	assert isinstance(sources, list)
	assert sources[0]['recipe']['epochs'] == 3
	assert len(str(sources[0]['checkpoint_sha256'])) == 64
	assert sources[1]['random_seed'] == 42


@pytest.mark.parametrize(
	('mutate', 'message'),
	[
		(lambda payload: payload.update({'epoch': 4}), 'epoch must equal 3'),
		(
			lambda payload: payload.update({'global_step': 1_874}),
			'global_step must equal 1875',
		),
		(
			lambda payload: payload.update({'resume_count': 1}),
			'single uninterrupted run',
		),
		(
			lambda payload: payload.update({'pretraining_method': 'barlow_twins_3d'}),
			'pretraining_method must equal',
		),
		(
			lambda payload: payload['training_state'].update(
				{'completed_epoch': False}
			),
			'completed pretraining epoch',
		),
		(
			lambda payload: payload['config'].update(
				{'continuation': {'init_checkpoint': '/x/latest.pt'}}
			),
			'single-stage without a continuation',
		),
		(
			lambda payload: payload['config']['augmentations'].update(
				{'gaussian_noise_std': 0.4}
			),
			'augmentations.gaussian_noise_std',
		),
		(
			lambda payload: payload['config']['augmentations'].update(
				{'horizontal_flip_probability': 0.5}
			),
			'augmentations keys differ',
		),
		(
			lambda payload: payload['config']['barlow_twins'].update(
				{'positive_window_tokens': [2, 2, 2]}
			),
			'positive_window_tokens must equal',
		),
		(
			lambda payload: payload['config']['barlow_twins'].update(
				{'local_pairs_per_crop': 64}
			),
			'local_pairs_per_crop must equal 128',
		),
		(
			lambda payload: payload['config']['train'].update({'seed': 43}),
			'train.seed must equal 42',
		),
		(
			lambda payload: payload['config']['train'].update({'lr': 1.0e-3}),
			'train.lr must equal',
		),
		(
			lambda payload: payload['config']['model'].update({'encoder_depth': 6}),
			'model.encoder_depth must equal',
		),
	],
)
def test_recipe_arm_audit_rejects_a_checkpoint_that_contradicts_the_recipe(
	tmp_path: Path,
	mutate: object,
	message: str,
) -> None:
	payload = arm_checkpoint_payload()
	assert callable(mutate)
	mutate(payload)
	universe = write_recipe_arm_universe(
		tmp_path,
		embeddings=False,
		arm_payload=payload,
	)
	with pytest.raises((ValueError, TypeError), match=message):
		audit_volve_horizon_recipe_arm_sources(universe['config'])


@pytest.mark.parametrize(
	('mutate', 'message'),
	[
		(
			lambda payload: payload['metadata'].update({'seed': 7}),
			'metadata.seed must equal 42',
		),
		(
			lambda payload: payload['metadata'].update(
				{'pretrained_weights_loaded': True}
			),
			'pretrained_weights_loaded',
		),
		(
			lambda payload: payload['training_state'].update(
				{'checkpoint_kind': 'epoch'}
			),
			'checkpoint_kind random_init',
		),
		(
			lambda payload: payload['training_state'].update(
				{'stage': 'barlow_twins_training'}
			),
			'training_state.stage must equal',
		),
	],
)
def test_recipe_arm_audit_rejects_a_baseline_that_is_not_the_random_encoder(
	tmp_path: Path,
	mutate: object,
	message: str,
) -> None:
	payload = baseline_checkpoint_payload()
	assert callable(mutate)
	mutate(payload)
	universe = write_recipe_arm_universe(
		tmp_path,
		embeddings=False,
		baseline_payload=payload,
	)
	with pytest.raises((ValueError, TypeError), match=message):
		audit_volve_horizon_recipe_arm_sources(universe['config'])


def test_recipe_arm_audit_can_select_only_the_baseline(tmp_path: Path) -> None:
	universe = write_recipe_arm_universe(tmp_path, embeddings=False)
	universe['checkpoints'][ARM_ID].unlink()
	report = audit_volve_horizon_recipe_arm_sources(
		universe['config'],
		model_ids=(RECIPE_ARM_BASELINE_MODEL_ID,),
	)
	assert report['model_order'] == [RECIPE_ARM_BASELINE_MODEL_ID]
	sources = report['sources']
	assert isinstance(sources, list)
	assert len(sources) == 1
	assert sources[0]['role'] == 'baseline'


def test_recipe_arm_audit_rejects_an_unknown_model_selection(
	tmp_path: Path,
) -> None:
	universe = write_recipe_arm_universe(tmp_path, embeddings=False)
	with pytest.raises(ValueError, match='unknown Volve horizon recipe-arm model'):
		audit_volve_horizon_recipe_arm_sources(
			universe['config'],
			model_ids=('mae',),
		)


def test_recipe_arm_embedding_suite_binds_both_models_to_one_support(
	tmp_path: Path,
) -> None:
	universe = write_recipe_arm_universe(tmp_path, embeddings=True)
	suite = inspect_volve_horizon_recipe_arm_embedding_suite(universe['config'])
	assert sorted(suite.sources) == sorted([ARM_ID, RECIPE_ARM_BASELINE_MODEL_ID])
	assert suite.embedding_dim == 384
	assert suite.source_by_id(ARM_ID).valid_tokens_sha256 == (
		suite.source_by_id(RECIPE_ARM_BASELINE_MODEL_ID).valid_tokens_sha256
	)


def test_recipe_arm_embedding_suite_accepts_a_single_model_selection(
	tmp_path: Path,
) -> None:
	universe = write_recipe_arm_universe(tmp_path, embeddings=True)
	suite = inspect_volve_horizon_recipe_arm_embedding_suite(
		universe['config'],
		model_ids=(ARM_ID,),
	)
	assert tuple(suite.sources) == (ARM_ID,)


def test_recipe_arm_embedding_suite_rejects_an_unknown_model(
	tmp_path: Path,
) -> None:
	universe = write_recipe_arm_universe(tmp_path, embeddings=True)
	with pytest.raises(ValueError, match='unknown Volve horizon recipe-arm model'):
		inspect_volve_horizon_recipe_arm_embedding_suite(
			universe['config'],
			model_ids=('local_barlow_twins',),
		)


def test_recipe_arm_jobs_cover_both_models_on_every_layout_and_size(
	tmp_path: Path,
) -> None:
	config = volve_horizon_recipe_arm_config_from_mapping(
		recipe_arm_config_mapping(tmp_path)
	)
	jobs = plan_volve_horizon_recipe_arm_jobs(config)
	assert len(jobs) == RECIPE_ARM_CONDITION_COUNT
	assert len(jobs) == 2 * len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX)
	assert len(set(jobs)) == len(jobs)
	assert {model for model, _, _ in jobs} == {
		ARM_ID,
		RECIPE_ARM_BASELINE_MODEL_ID,
	}


def test_recipe_arm_job_output_dir_is_model_layout_size(tmp_path: Path) -> None:
	config = volve_horizon_recipe_arm_config_from_mapping(
		recipe_arm_config_mapping(tmp_path)
	)
	job = resolve_volve_horizon_recipe_arm_job(
		config,
		model=ARM_ID,
		layout='layout_002',
		size='medium',
	)
	assert job.output_dir == (
		config.runs_root
		/ f'model={ARM_ID}'
		/ 'layout=layout_002'
		/ 'size=medium'
	)
	assert job.metrics_path.name == 'metrics.json'


def test_recipe_arm_five_way_projection_preserves_the_shared_contract(
	tmp_path: Path,
) -> None:
	config = volve_horizon_recipe_arm_config_from_mapping(
		recipe_arm_config_mapping(tmp_path)
	)
	projected = as_five_way_config(config)
	assert projected.model_ids == (ARM_ID, RECIPE_ARM_BASELINE_MODEL_ID)
	assert projected.train == config.train
	assert projected.tiles == config.tiles
	assert projected.benchmark_id == config.benchmark_id
	assert projected.checkpoint_selection == config.checkpoint_selection
	arm_source = projected.model_by_id(ARM_ID)
	assert arm_source.expected['objective'] == LOCAL_BARLOW_TWINS_METHOD
	assert arm_source.expected['stratigraphy_pretext'] is False


def test_recipe_arm_never_reuses_a_fixed_five_way_model_id(
	tmp_path: Path,
) -> None:
	raw = recipe_arm_config_mapping(tmp_path)
	arm = raw['arm']
	assert isinstance(arm, dict)
	config = volve_horizon_recipe_arm_config_from_mapping(deepcopy(raw))
	assert config.arm_id not in FIVE_WAY_MODEL_IDS
