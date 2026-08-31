from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import yaml

from proc.seis_ssl_cluster import summarize_volve_horizon_five_way as summary_cli
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_MODEL_IDS,
	VolveHorizonFiveWayConfig,
)
from seis_ssl_cluster.volve.horizon_five_way_results import (
	PAIRED_COMPARISONS,
	PRIMARY_METRIC,
	SUMMARY_METRICS,
	SUMMARY_OUTPUT_NAMES,
	inspect_volve_horizon_five_way_results,
	summarize_volve_horizon_five_way,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	VolveHorizonFiveWayEmbeddingSuite,
	audit_volve_horizon_five_way_sources,
	inspect_volve_horizon_five_way_embedding_suite,
)
from seis_ssl_cluster.volve.horizon_frozen import (
	OBJECTIVE_IDENTITY,
	OPTIMIZER_BETAS,
	OPTIMIZER_EPS,
	OPTIMIZER_NAME,
	decoder_initial_state_sha256,
)
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.volve.horizon_model import create_volve_horizon_decoder
from tests.seis_ssl_cluster.test_volve_horizon_five_way_sources import (
	_write_universe as _write_source_universe,
)

if TYPE_CHECKING:
	from pathlib import Path

_MODEL_MAE = {
	'mae': 5.0,
	'mae_hmm_k6': 4.0,
	'local_barlow_twins': 3.0,
	'local_barlow_twins_hmm_k6': 2.0,
	'random': 6.0,
}
_DECODER_ARCHITECTURE = create_volve_horizon_decoder().architecture
_DECODER_INITIAL_STATE_SHA256 = decoder_initial_state_sha256()


def _digest(value: str) -> str:
	return hashlib.sha256(value.encode()).hexdigest()


def _job_dir(
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
) -> Path:
	return (
		config.runs_root
		/ f'model={model_id}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)


def _metric_value(
	model_id: str,
	layout_id: str,
	data_size: str,
	horizon_index: int | None = None,
) -> float:
	value = (
		_MODEL_MAE[model_id]
		+ 0.1 * LAYOUT_IDS.index(layout_id)
		+ 0.01 * tuple(DATA_SIZE_PREFIX).index(data_size)
	)
	if horizon_index is not None:
		value += 0.02 * horizon_index
	return value


def _evaluation_metrics(
	model_id: str,
	layout_id: str,
	data_size: str,
	counts: dict[str, int],
) -> dict[str, object]:
	per_horizon = {
		horizon_name: {
			'count': counts[horizon_name],
			'predicted_count': counts[horizon_name],
			'missing_prediction_count': 0,
			'mae_samples': _metric_value(
				model_id,
				layout_id,
				data_size,
				horizon_index,
			),
			'mae_ms': 4.0
			* _metric_value(
				model_id,
				layout_id,
				data_size,
				horizon_index,
			),
		}
		for horizon_index, horizon_name in enumerate(HORIZON_NAMES)
	}
	return {
		'macro_mae_samples': sum(
			float(item['mae_samples']) for item in per_horizon.values()
		)
		/ len(per_horizon),
		'macro_within_2_samples': 0.5,
		'macro': {'within_1': 0.25, 'within_4': 0.75},
		'per_horizon': per_horizon,
		'coverage': {
			'eligible_count': sum(counts.values()),
			'predicted_count': sum(counts.values()),
			'fraction': 1.0,
		},
		'missing_prediction_count': 0,
		'predicted_adjacent_order_violation_rate': 0.0,
		'predicted_adjacent_order_pair_count': 10,
	}


def _run_identity(  # noqa: PLR0913
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
	*,
	primary_counts: dict[str, int],
	secondary_counts: dict[str, int],
	embedding_suite: VolveHorizonFiveWayEmbeddingSuite,
) -> dict[str, object]:
	model = config.model_by_id(model_id)
	source = embedding_suite.source_by_id(model_id)
	paths = source.paths
	return {
		'schema_version': 3,
		'benchmark': 'mae_local_bt_hmm_five_way_v1',
		'model': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'canonical_scientific_identity': {
			'scientific_identity_sha256': _digest('canonical')
		},
		'horizon_split_plan': {
			'layout_id': layout_id,
			'data_size': data_size,
			'scientific_identity_sha256': _digest(
				f'split/{layout_id}/{data_size}'
			),
		},
		'embedding': {
			'embeddings_path': str(paths.embeddings),
			'embeddings_sha256': source.embeddings_sha256,
			'metadata_path': str(paths.metadata),
			'metadata_sha256': source.metadata_sha256,
			'checkpoint_path': str(model.checkpoint),
			'checkpoint_sha256': source.checkpoint_identity['checkpoint_sha256'],
			'model_source': dict(source.checkpoint_identity),
			'valid_tokens_sha256': source.valid_tokens_sha256,
		},
		'decoder': {
			'initialization_seed': 42000,
			'initial_state_sha256': _DECODER_INITIAL_STATE_SHA256,
			'architecture': _DECODER_ARCHITECTURE,
		},
		'tiles': {
			'patch_size_xyz': [8, 8, 8],
			'core_size_tokens': [8, 8, 27],
			'context_halo_tokens': [1, 1, 0],
			'window_start': 552,
			'window_stop': 768,
			'order': 'lateral_token_grid_x_then_y_v1',
			'record_sha256': _digest(f'tiles/{layout_id}/{data_size}'),
		},
		'native_horizon_observation_counts': {
			'test_primary_common': dict(primary_counts),
			'test_secondary_per_horizon': dict(secondary_counts),
		},
		'effective_model_valid_observation_counts': {
			'train': dict.fromkeys(HORIZON_NAMES, 5),
			'validation': dict.fromkeys(HORIZON_NAMES, 4),
			'test_primary_common': dict(primary_counts),
			'test_secondary_per_horizon': dict(secondary_counts),
		},
		'excluded_by_token_validity_counts': {
			'train': dict.fromkeys(HORIZON_NAMES, 0),
			'validation': dict.fromkeys(HORIZON_NAMES, 0),
			'test_primary_common': dict.fromkeys(HORIZON_NAMES, 0),
			'test_secondary_per_horizon': dict.fromkeys(HORIZON_NAMES, 0),
		},
		'training': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 1.0e-3,
			'weight_decay': 1.0e-4,
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'amp_on_cuda': True,
			'gradient_clip_norm': 1.0,
		},
		'optimizer': {
			'name': OPTIMIZER_NAME,
			'betas': list(OPTIMIZER_BETAS),
			'eps': OPTIMIZER_EPS,
			'weight_decay': 1.0e-4,
		},
		'objective': dict(OBJECTIVE_IDENTITY),
		'runtime_precision': {
			'device_type': 'cpu',
			'amp_enabled': False,
			'autocast_dtype': None,
			'scaler_required': False,
		},
	}


def _write_best_and_metrics(
	job_dir: Path,
	metrics: dict[str, object],
) -> None:
	identity = metrics['benchmark_identity']
	best_path = job_dir / 'best.pt'
	torch.save(
		{
			'epoch': metrics['best_epoch'],
			'run_identity': identity,
			'runtime_precision': metrics['runtime_precision'],
			'validation': metrics['validation'],
			'model_state_dict': {'weight': torch.zeros(1)},
		},
		best_path,
	)
	metrics['best_checkpoint'] = {
		'path': str(best_path),
		'sha256': file_sha256(best_path),
	}
	(job_dir / 'metrics.json').write_text(
		json.dumps(metrics),
		encoding='utf-8',
	)


def _write_run(
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
	*,
	embedding_suite: VolveHorizonFiveWayEmbeddingSuite | None = None,
) -> None:
	job_dir = _job_dir(config, model_id, layout_id, data_size)
	job_dir.mkdir(parents=True)
	primary_counts = {
		name: 10 + index for index, name in enumerate(HORIZON_NAMES)
	}
	secondary_counts = {
		name: 20 + index for index, name in enumerate(HORIZON_NAMES)
	}
	if embedding_suite is None:
		source_audit = audit_volve_horizon_five_way_sources(config)
		embedding_suite = inspect_volve_horizon_five_way_embedding_suite(
			config,
			source_audit=source_audit,
		)
	identity = _run_identity(
		config,
		model_id,
		layout_id,
		data_size,
		primary_counts=primary_counts,
		secondary_counts=secondary_counts,
		embedding_suite=embedding_suite,
	)
	metrics = {
		'schema_version': 1,
		'artifact_type': 'volve_frozen_horizon_job_metrics',
		'model': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'benchmark_identity': identity,
		'runtime_precision': identity['runtime_precision'],
		'best_epoch': 2,
		'validation': _evaluation_metrics(
			model_id,
			layout_id,
			data_size,
			dict.fromkeys(HORIZON_NAMES, 4),
		),
		'test': {
			'primary_common': _evaluation_metrics(
				model_id,
				layout_id,
				data_size,
				primary_counts,
			),
			'secondary_per_horizon': _evaluation_metrics(
				model_id,
				layout_id,
				data_size,
				secondary_counts,
			),
			'evaluation_pass_count': 1,
		},
	}
	_write_best_and_metrics(job_dir, metrics)


def _build_universe(tmp_path: Path) -> VolveHorizonFiveWayConfig:
	universe = _write_source_universe(tmp_path, embeddings=True)
	config = universe['config']
	assert isinstance(config, VolveHorizonFiveWayConfig)
	source_audit = audit_volve_horizon_five_way_sources(config)
	embedding_suite = inspect_volve_horizon_five_way_embedding_suite(
		config,
		source_audit=source_audit,
	)
	for model_id in FIVE_WAY_MODEL_IDS:
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZE_PREFIX:
				_write_run(
					config,
					model_id,
					layout_id,
					data_size,
					embedding_suite=embedding_suite,
				)
	return config


def _rewrite_job(
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
	mutate: object,
) -> None:
	job_dir = _job_dir(config, model_id, layout_id, data_size)
	metrics = json.loads((job_dir / 'metrics.json').read_text(encoding='utf-8'))
	assert callable(mutate)
	mutate(metrics)
	_write_best_and_metrics(job_dir, metrics)


def _raw_config(config: VolveHorizonFiveWayConfig) -> dict[str, object]:
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
			'upsample_factors': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
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


def test_complete_results_write_exactly_five_outputs(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)

	report = inspect_volve_horizon_five_way_results(config)
	result = summarize_volve_horizon_five_way(config)

	assert report['complete_jobs'] == 75
	assert report['model_order'] == list(FIVE_WAY_MODEL_IDS)
	assert result['complete_jobs'] == 75
	assert sorted(path.name for path in config.summary_root.iterdir()) == sorted(
		SUMMARY_OUTPUT_NAMES
	)
	with (config.summary_root / 'comparison.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		comparison = list(csv.DictReader(handle))
	assert len(comparison) == 75
	assert 'embeddings_sha256' in comparison[0]
	for model_id in FIVE_WAY_MODEL_IDS:
		model_rows = [row for row in comparison if row['model_id'] == model_id]
		assert len({row['embeddings_sha256'] for row in model_rows}) == 1
	with (config.summary_root / 'paired_deltas.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		paired = list(csv.DictReader(handle))
	assert len(paired) == 3 * 5 * len(PAIRED_COMPARISONS) * len(SUMMARY_METRICS)
	with (config.summary_root / 'summary_by_size.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		by_size = list(csv.DictReader(handle))
	assert len(by_size) == 3 * len(PAIRED_COMPARISONS) * len(SUMMARY_METRICS)
	assert {row['n_layouts'] for row in by_size} == {'5'}
	summary = json.loads(
		(config.summary_root / 'summary.json').read_text(encoding='utf-8')
	)
	assert summary['job_count'] == 75
	assert summary['models'] == list(FIVE_WAY_MODEL_IDS)
	assert summary['delta_definition'] == 'left_mae_minus_right_mae'
	assert len(summary['comparison']) == 75
	markdown = (config.summary_root / 'summary.md').read_text(encoding='utf-8')
	assert 'left_MAE - right_MAE' in markdown
	assert 'positive delta means the right-hand model has lower MAE' in markdown


def test_paired_delta_sign_and_statistics_are_left_minus_right(
	tmp_path: Path,
) -> None:
	config = _build_universe(tmp_path)
	summarize_volve_horizon_five_way(config)
	with (config.summary_root / 'paired_deltas.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		paired = list(csv.DictReader(handle))
	selected = [
		row
		for row in paired
		if row['comparison_id'] == 'mae_minus_mae_hmm_k6'
		and row['metric'] == PRIMARY_METRIC
	]
	assert len(selected) == 15
	assert all(float(row['delta']) == pytest.approx(1.0) for row in selected)
	with (config.summary_root / 'summary_by_size.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		by_size = list(csv.DictReader(handle))
	row = next(
		item
		for item in by_size
		if item['data_size'] == 'small'
		and item['comparison_id'] == 'mae_minus_mae_hmm_k6'
		and item['metric'] == PRIMARY_METRIC
	)
	assert float(row['mean']) == pytest.approx(1.0)
	assert float(row['median']) == pytest.approx(1.0)
	assert float(row['sample_std']) == pytest.approx(0.0)
	assert (row['positive_count'], row['zero_count'], row['negative_count']) == (
		'5',
		'0',
		'0',
	)


def test_missing_result_fails_without_writing_summary(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	missing = _job_dir(config, 'random', 'layout_004', 'large') / 'metrics.json'
	missing.unlink()

	with pytest.raises(FileNotFoundError, match='missing 1 of 75'):
		summarize_volve_horizon_five_way(config)

	assert not config.summary_root.exists()


def test_unexpected_run_directory_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	(config.runs_root / 'model=unknown').mkdir()

	with pytest.raises(ValueError, match='unexpected Volve five-way'):
		inspect_volve_horizon_five_way_results(config)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('model', 'random'),
		('layout_id', 'layout_004'),
		('data_size', 'large'),
	],
)
def test_path_identity_mismatch_is_rejected(
	tmp_path: Path,
	key: str,
	value: str,
) -> None:
	config = _build_universe(tmp_path)
	metrics_path = _job_dir(config, 'mae', 'layout_000', 'small') / 'metrics.json'
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics[key] = value
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')
	with pytest.raises(ValueError, match=rf'metrics {key}'):
		inspect_volve_horizon_five_way_results(config)


def test_source_identity_mismatch_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	metrics_path = _job_dir(config, 'mae', 'layout_000', 'small') / 'metrics.json'
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics['benchmark_identity']['embedding']['checkpoint_sha256'] = 'f' * 64
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')
	with pytest.raises(ValueError, match='checkpoint SHA-256'):
		inspect_volve_horizon_five_way_results(config)


def test_embedding_reextraction_between_cells_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	model = config.model_by_id('mae')
	paths = output_paths(model.embeddings_dir, config.survey_id)
	embeddings = np.load(paths.embeddings, allow_pickle=False)
	embeddings[0, 0, 0, 0] = np.float16(1.0)
	np.save(paths.embeddings, embeddings)
	new_sha256 = file_sha256(paths.embeddings)

	completed_after_reextraction = tuple(
		(layout_id, data_size)
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZE_PREFIX
	)[8:]
	for layout_id, data_size in completed_after_reextraction:
		def mutate(metrics: dict[str, object]) -> None:
			identity = metrics['benchmark_identity']
			assert isinstance(identity, dict)
			embedding = identity['embedding']
			assert isinstance(embedding, dict)
			embedding['embeddings_sha256'] = new_sha256

		_rewrite_job(config, 'mae', layout_id, data_size, mutate)

	with pytest.raises(ValueError, match='embedding array SHA-256'):
		inspect_volve_horizon_five_way_results(config)


def test_missing_embedding_array_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	model = config.model_by_id('mae')
	paths = output_paths(model.embeddings_dir, config.survey_id)
	paths.embeddings.unlink()

	assert paths.metadata.is_file()
	assert paths.valid_tokens.is_file()
	with pytest.raises(FileNotFoundError, match='embedding source is missing'):
		summarize_volve_horizon_five_way(config)
	assert not config.summary_root.exists()


def test_missing_embedding_model_source_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		embedding = identity['embedding']
		assert isinstance(embedding, dict)
		embedding.pop('model_source')

	_rewrite_job(config, 'mae', 'layout_000', 'small', mutate)

	with pytest.raises(TypeError, match='model_source must be a mapping'):
		inspect_volve_horizon_five_way_results(config)


def test_foreign_benchmark_identity_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		identity['benchmark'] = 'mae_vs_random_frozen_v1'

	_rewrite_job(config, 'mae', 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match='benchmark identity must equal'):
		inspect_volve_horizon_five_way_results(config)


def test_foreign_run_identity_schema_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		identity['schema_version'] = 4

	_rewrite_job(config, 'mae', 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match='identity schema_version must equal 3'):
		inspect_volve_horizon_five_way_results(config)


def test_shared_downstream_config_drift_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		training = identity['training']
		assert isinstance(training, dict)
		training['epochs'] = 1

	for model_id in FIVE_WAY_MODEL_IDS:
		_rewrite_job(config, model_id, 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match=r'training\.epochs differs'):
		inspect_volve_horizon_five_way_results(config)


def test_changed_configured_checkpoint_is_rejected(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	config.model_by_id('mae').checkpoint.write_bytes(b'changed-after-extraction')

	with pytest.raises(ValueError, match='checkpoint metadata is unreadable'):
		inspect_volve_horizon_five_way_results(config)


@pytest.mark.parametrize(
	('identity_key', 'match'),
	[
		('horizon_split_plan', 'shared_run_identity'),
		('decoder', 'shared_run_identity'),
	],
)
def test_cross_model_scientific_identity_mismatch_is_rejected(
	tmp_path: Path,
	identity_key: str,
	match: str,
) -> None:
	config = _build_universe(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		block = identity[identity_key]
		assert isinstance(block, dict)
		block['drift'] = True

	_rewrite_job(config, 'random', 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match=match):
		inspect_volve_horizon_five_way_results(config)


def test_cross_model_evaluation_support_mismatch_is_rejected(
	tmp_path: Path,
) -> None:
	config = _build_universe(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		test = metrics['test']
		assert isinstance(test, dict)
		primary = test['primary_common']
		assert isinstance(primary, dict)
		coverage = primary['coverage']
		assert isinstance(coverage, dict)
		coverage['predicted_count'] = int(coverage['predicted_count']) - 1
		coverage['fraction'] = int(coverage['predicted_count']) / int(
			coverage['eligible_count']
		)
		primary['missing_prediction_count'] = 1
		per_horizon = primary['per_horizon']
		assert isinstance(per_horizon, dict)
		first = per_horizon[HORIZON_NAMES[0]]
		assert isinstance(first, dict)
		first['predicted_count'] = int(first['predicted_count']) - 1
		first['missing_prediction_count'] = 1

	_rewrite_job(config, 'random', 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match='support_identity'):
		inspect_volve_horizon_five_way_results(config)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_metric_is_rejected(tmp_path: Path, value: float) -> None:
	config = _build_universe(tmp_path)
	metrics_path = _job_dir(config, 'mae', 'layout_000', 'small') / 'metrics.json'
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics['test']['primary_common']['macro_mae_samples'] = value
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')

	with pytest.raises(ValueError, match='non-finite'):
		inspect_volve_horizon_five_way_results(config)


def test_best_checkpoint_and_selected_epoch_must_match(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	metrics_path = _job_dir(config, 'mae', 'layout_000', 'small') / 'metrics.json'
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics['best_epoch'] = 3
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')

	with pytest.raises(ValueError, match='selected best epoch'):
		inspect_volve_horizon_five_way_results(config)


def test_existing_summary_root_is_never_overwritten(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	config.summary_root.mkdir(parents=True)
	marker = config.summary_root / 'keep.txt'
	marker.write_text('keep', encoding='utf-8')

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		summarize_volve_horizon_five_way(config)

	assert marker.read_text(encoding='utf-8') == 'keep'


def test_check_only_cli_writes_nothing(tmp_path: Path) -> None:
	config = _build_universe(tmp_path)
	config_path = tmp_path / 'five_way.yaml'
	config_path.write_text(yaml.safe_dump(_raw_config(config)), encoding='utf-8')
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py',
			'--config',
			str(config_path),
			'--check-only',
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert 'complete_jobs: 75' in result.stdout
	assert 'execution: check-only; summary files skipped' in result.stdout
	assert not config.summary_root.exists()
	parser = summary_cli.build_parser()
	args = parser.parse_args(['--config', str(config_path), '--check-only'])
	assert args.check_only is True
	assert not hasattr(args, 'dry_run')
