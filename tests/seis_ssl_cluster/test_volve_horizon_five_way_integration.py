'''Synthetic integration coverage for the Volve horizon five-way workflow.'''

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest
import yaml

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_MODEL_IDS,
	FIVE_WAY_WITHIN2_BENCHMARK_ID,
	VolveHorizonFiveWayConfig,
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_results import (
	inspect_volve_horizon_five_way_results,
)
from seis_ssl_cluster.volve.horizon_five_way_runner import (
	inspect_volve_horizon_five_way_job,
	plan_volve_horizon_five_way_jobs,
	resolve_volve_horizon_five_way_job,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	audit_volve_horizon_five_way_sources,
	inspect_volve_horizon_five_way_embedding_suite,
	plan_volve_horizon_five_way_sources,
)
from seis_ssl_cluster.volve.horizon_frozen import (
	FROZEN_MODEL_ROLES,
	enumerate_frozen_horizon_conditions,
)
from seis_ssl_cluster.volve.horizon_runner import (
	CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
)
from tests.seis_ssl_cluster.helpers_volve_five_way import (
	write_completed_plan_metrics,
	write_five_way_completed_run,
	write_five_way_horizon_fixture,
	write_five_way_universe,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_within2_plan_records_selection_in_scientific_identity(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(
		tmp_path,
		embeddings=True,
		checkpoint_selection=CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
	)
	config = universe['config']
	assert isinstance(config, VolveHorizonFiveWayConfig)
	data, layout_path = write_five_way_horizon_fixture(tmp_path, config)
	source_audit = audit_volve_horizon_five_way_sources(config)
	suite = inspect_volve_horizon_five_way_embedding_suite(
		config, source_audit=source_audit
	)
	plan = inspect_volve_horizon_five_way_job(
		resolve_volve_horizon_five_way_job(
			config,
			model='mae_hmm_k6',
			layout='layout_000',
			size='small',
		),
		layout_config=layout_path,
		data=data,
		embedding_suite=suite,
	)

	assert plan.checkpoint_selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2
	assert plan.run_identity['benchmark'] == FIVE_WAY_WITHIN2_BENCHMARK_ID
	assert plan.run_identity['objective'] == {
		'loss': 'fractional_two_bin_per_tile_horizon_macro_v1',
		'prediction': 'masked_soft_argmax_v1',
		'checkpoint_selection': CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
		'metrics_schema_version': 1,
	}


def test_synthetic_five_way_contract_connects_preflight_plans_and_results(
	tmp_path: Path,
) -> None:
	universe = write_five_way_universe(tmp_path, embeddings=True)
	fixture_config = universe['config']
	assert isinstance(fixture_config, VolveHorizonFiveWayConfig)
	config_path = tmp_path / 'five_way.yaml'
	config_path.write_text(
		yaml.safe_dump(_config_mapping(fixture_config), sort_keys=False),
		encoding='utf-8',
	)
	config = volve_horizon_five_way_config_from_mapping(load_config(config_path))
	data, layout_path = write_five_way_horizon_fixture(tmp_path, config)

	assert tuple(
		row['model_id'] for row in plan_volve_horizon_five_way_sources(config)
	) == FIVE_WAY_MODEL_IDS
	source_audit = audit_volve_horizon_five_way_sources(config)
	suite = inspect_volve_horizon_five_way_embedding_suite(
		config,
		source_audit=source_audit,
	)
	conditions = plan_volve_horizon_five_way_jobs(config)
	assert len(conditions) == len(set(conditions)) == 75

	plans = tuple(
		inspect_volve_horizon_five_way_job(
			resolve_volve_horizon_five_way_job(
				config,
				model=model_id,
				layout='layout_000',
				size='small',
			),
			layout_config=layout_path,
			data=data,
			embedding_suite=suite,
		)
		for model_id in FIVE_WAY_MODEL_IDS
	)
	assert tuple(plan.model for plan in plans) == FIVE_WAY_MODEL_IDS
	assert len(
		{plan.split_plan.scientific_identity_sha256 for plan in plans}
	) == 1
	assert len(
		{plan.geometry.model_valid_lateral_mask_sha256 for plan in plans}
	) == 1
	assert len(
		{
			tuple(plan.effective_per_horizon_counts.items())
			for plan in plans
		}
	) == 1
	assert len(
		{
			str(plan.run_identity['decoder']['initial_state_sha256'])
			for plan in plans
		}
	) == 1
	for plan in plans:
		source = suite.source_by_id(plan.model)
		assert plan.selected_embedding_paths == source.paths
		assert plan.selected_embedding_paths.embeddings.parent == (
			config.model_by_id(plan.model).embeddings_dir
		)
		embedding_identity = plan.run_identity['embedding']
		assert embedding_identity['embeddings_path'] == str(
			source.paths.embeddings
		)
		assert embedding_identity['embeddings_sha256'] == source.embeddings_sha256
		assert embedding_identity['model_source'] == source.checkpoint_identity

	for model_id, layout_id, data_size in conditions:
		write_five_way_completed_run(
			config,
			model_id,
			layout_id,
			data_size,
			embedding_suite=suite,
		)
	for plan in plans:
		write_completed_plan_metrics(plan)
	report = inspect_volve_horizon_five_way_results(config)
	assert report['complete_jobs'] == 75
	assert report['model_order'] == list(FIVE_WAY_MODEL_IDS)

	legacy_cli = importlib.import_module(
		'proc.seis_ssl_cluster.run_volve_horizon_frozen'
	)
	legacy_args = legacy_cli.build_parser().parse_args(
		[
			'--model',
			'pretrained',
			'--layout',
			'layout_000',
			'--size',
			'small',
		]
	)
	assert FROZEN_MODEL_ROLES == ('pretrained', 'random')
	assert len(enumerate_frozen_horizon_conditions()) == 30
	assert legacy_args.config.parts[-2:] == (
		'30_mae_vs_random_frozen_v1',
		'03_horizon_frozen.yaml',
	)
	with pytest.raises(SystemExit):
		legacy_cli.build_parser().parse_args(
			[
				'--model',
				'mae',
				'--layout',
				'layout_000',
				'--size',
				'small',
			]
		)


def _config_mapping(config: VolveHorizonFiveWayConfig) -> dict[str, object]:
	return {
		'paths': {
			'artifact_root': str(config.artifact_root),
			'volve_root': str(config.volve_root),
		},
		'dataset': {'survey_id': config.survey_id},
		'inputs': {
			'canonical_input_metadata': str(config.canonical_input_metadata)
		},
		'models': {
			model.model_id: {
				'checkpoint': str(model.checkpoint),
				'embeddings_dir': str(model.embeddings_dir),
			}
			for model in config.models
		},
		'outputs': {
			'runs_root': str(config.runs_root),
			'summary_root': str(config.summary_root),
		},
		'decoder': {
			'embedding_dim': 384,
			'class_count': 5,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2]] * 3,
			'upsample_mode': 'nearest',
			'normalization': 'voxelwise_layer_norm',
		},
		'tiles': {
			'patch_size': [8, 8, 8],
			'core_size_tokens': [8, 8, 27],
			'context_halo_tokens': [1, 1, 0],
			'window_start': 552,
			'window_stop': 768,
			'min_token_valid_fraction': 1.0,
		},
		'train': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 1.0e-3,
			'weight_decay': 1.0e-4,
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'amp': True,
			'gradient_clip_norm': 1.0,
		},
	}
