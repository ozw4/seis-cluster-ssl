'''Synthetic integration coverage for the Volve horizon five-way workflow.'''

from __future__ import annotations

import hashlib
import importlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import yaml

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_MODEL_IDS,
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
from tests.seis_ssl_cluster.helpers_volve import (
	write_synthetic_frozen_horizon_data,
)
from tests.seis_ssl_cluster.test_volve_horizon_five_way_results import (
	_write_run,
)
from tests.seis_ssl_cluster.test_volve_horizon_five_way_sources import (
	_write_universe,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_synthetic_five_way_contract_connects_preflight_plans_and_results(
	tmp_path: Path,
) -> None:
	universe = _write_universe(tmp_path, embeddings=True)
	fixture_config = universe['config']
	assert isinstance(fixture_config, VolveHorizonFiveWayConfig)
	config_path = tmp_path / 'five_way.yaml'
	config_path.write_text(
		yaml.safe_dump(_config_mapping(fixture_config), sort_keys=False),
		encoding='utf-8',
	)
	config = volve_horizon_five_way_config_from_mapping(load_config(config_path))
	data = write_synthetic_frozen_horizon_data(tmp_path / 'horizon')
	_resize_embeddings_to_horizon_data(config, data)

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

	layout_path = _write_layouts(tmp_path)
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
		embedding_identity = plan.run_identity['embedding']
		assert embedding_identity['embeddings_path'] == str(
			source.paths.embeddings
		)
		assert embedding_identity['embeddings_sha256'] == source.embeddings_sha256
		assert embedding_identity['model_source'] == source.checkpoint_identity

	for model_id, layout_id, data_size in conditions:
		_write_run(
			config,
			model_id,
			layout_id,
			data_size,
			embedding_suite=suite,
		)
	for plan in plans:
		_write_completed_plan_metrics(plan)
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


def _resize_embeddings_to_horizon_data(config, data) -> None:
	volume_shape = (*data.shape_xy, len(data.time_ms))
	token_grid = tuple((value + 7) // 8 for value in volume_shape)
	valid_tokens = np.ones(token_grid, dtype=np.bool_)
	valid_tokens[0, 0, :] = False
	canonical = json.loads(
		config.canonical_input_metadata.read_text(encoding='utf-8')
	)
	identity = canonical['scientific_identity']
	identity.update(
		{
			'shape_xyz': list(volume_shape),
			'valid_trace_mask_sha256': file_sha256(data.paths.valid_trace_mask),
			'inline_values_sha256': file_sha256(data.paths.inline_values),
			'crossline_values_sha256': file_sha256(data.paths.crossline_values),
			'time_axis_sha256': file_sha256(data.paths.time_ms),
		}
	)
	canonical['scientific_identity_sha256'] = _json_sha256(identity)
	canonical['provenance']['public_inputs']['valid_trace_mask.npy'] = str(
		data.paths.valid_trace_mask
	)
	_write_json(config.canonical_input_metadata, canonical)
	for model in config.models:
		paths = output_paths(model.embeddings_dir, config.survey_id)
		metadata = json.loads(paths.metadata.read_text(encoding='utf-8'))
		metadata.update(
			{
				'source_valid_mask_path': str(data.paths.valid_trace_mask),
				'volume_shape_xyz': list(volume_shape),
				'token_grid_shape': list(token_grid),
			}
		)
		np.save(paths.embeddings, np.zeros((*token_grid, 384), dtype=np.float16))
		np.save(paths.valid_tokens, valid_tokens)
		_write_json(paths.metadata, metadata)


def _write_layouts(tmp_path: Path) -> Path:
	path = tmp_path / 'layouts.yaml'
	path.write_text(
		yaml.safe_dump(
			{
				'selection': {
					'semantics': (
						'explicit_section_prefix_all_available_horizon_points_v1'
					)
				},
				'validation': {'inline': [120], 'crossline': [220]},
				'layouts': {
					f'layout_{index:03d}': {
						'inline': list(range(100 + 4 * index, 104 + 4 * index)),
						'crossline': list(range(200 + 4 * index, 204 + 4 * index)),
					}
					for index in range(5)
				},
			},
			sort_keys=False,
		),
		encoding='utf-8',
	)
	return path


def _write_completed_plan_metrics(plan) -> None:
	runtime_precision = {
		'device_type': 'cpu',
		'amp_enabled': False,
		'autocast_dtype': None,
		'scaler_required': False,
	}
	identity = {**plan.run_identity, 'runtime_precision': runtime_precision}
	validation = _evaluation(plan.effective_per_horizon_counts['validation'])
	primary = _evaluation(plan.effective_per_horizon_counts['test_primary'])
	secondary = _evaluation(plan.effective_per_horizon_counts['test'])
	best_path = plan.output_dir / 'best.pt'
	torch.save(
		{
			'epoch': 2,
			'run_identity': identity,
			'runtime_precision': runtime_precision,
			'validation': validation,
			'model_state_dict': {'weight': torch.zeros(1)},
		},
		best_path,
	)
	payload = {
		'schema_version': 1,
		'artifact_type': 'volve_frozen_horizon_job_metrics',
		'model': plan.model,
		'layout_id': plan.layout_id,
		'data_size': plan.data_size,
		'benchmark_identity': identity,
		'runtime_precision': runtime_precision,
		'best_epoch': 2,
		'best_checkpoint': {
			'path': str(best_path),
			'sha256': file_sha256(best_path),
		},
		'validation': validation,
		'test': {
			'primary_common': primary,
			'secondary_per_horizon': secondary,
			'evaluation_pass_count': 1,
		},
	}
	_write_json(plan.output_dir / 'metrics.json', payload)


def _evaluation(counts: tuple[int, ...]) -> dict[str, object]:
	values = tuple(1.0 + 0.01 * index for index in range(len(HORIZON_NAMES)))
	per_horizon = {
		name: {
			'count': counts[index],
			'predicted_count': counts[index],
			'missing_prediction_count': 0,
			'mae_samples': values[index],
			'mae_ms': 4.0 * values[index],
		}
		for index, name in enumerate(HORIZON_NAMES)
	}
	total = sum(counts)
	return {
		'macro_mae_samples': sum(values) / len(values),
		'macro_within_2_samples': 0.5,
		'macro': {'within_1': 0.25, 'within_4': 0.75},
		'per_horizon': per_horizon,
		'coverage': {
			'eligible_count': total,
			'predicted_count': total,
			'fraction': 1.0,
		},
		'missing_prediction_count': 0,
		'predicted_adjacent_order_violation_rate': 0.0,
		'predicted_adjacent_order_pair_count': 1,
	}


def _json_sha256(payload: object) -> str:
	return hashlib.sha256(
		json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


def _write_json(path: Path, payload: object) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
