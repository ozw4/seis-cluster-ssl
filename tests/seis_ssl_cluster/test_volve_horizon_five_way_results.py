from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
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
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from tests.seis_ssl_cluster.helpers_volve_five_way import (
	five_way_job_dir,
	rewrite_five_way_completed_run,
	write_five_way_completed_matrix,
)

if TYPE_CHECKING:
	from pathlib import Path

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
	config = write_five_way_completed_matrix(tmp_path)

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
	config = write_five_way_completed_matrix(tmp_path)
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
	config = write_five_way_completed_matrix(tmp_path)
	missing = five_way_job_dir(config, 'random', 'layout_004', 'large') / 'metrics.json'
	missing.unlink()

	with pytest.raises(FileNotFoundError, match='missing 1 of 75'):
		summarize_volve_horizon_five_way(config)

	assert not config.summary_root.exists()


def test_unexpected_run_directory_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)
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
	config = write_five_way_completed_matrix(tmp_path)
	metrics_path = (
		five_way_job_dir(config, 'mae', 'layout_000', 'small') / 'metrics.json'
	)
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics[key] = value
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')
	with pytest.raises(ValueError, match=rf'metrics {key}'):
		inspect_volve_horizon_five_way_results(config)


def test_source_identity_mismatch_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)
	metrics_path = (
		five_way_job_dir(config, 'mae', 'layout_000', 'small') / 'metrics.json'
	)
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics['benchmark_identity']['embedding']['checkpoint_sha256'] = 'f' * 64
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')
	with pytest.raises(ValueError, match='checkpoint SHA-256'):
		inspect_volve_horizon_five_way_results(config)


def test_embedding_reextraction_between_cells_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)
	model = config.model_by_id('mae')
	paths = output_paths(model.embeddings_dir, config.survey_id)
	embeddings = np.load(paths.embeddings, allow_pickle=False)
	embeddings[0, 0, 0, 0] = np.float16(99.0)
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

		rewrite_five_way_completed_run(config, 'mae', layout_id, data_size, mutate)

	with pytest.raises(ValueError, match='embedding array SHA-256'):
		inspect_volve_horizon_five_way_results(config)


def test_missing_embedding_array_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)
	model = config.model_by_id('mae')
	paths = output_paths(model.embeddings_dir, config.survey_id)
	paths.embeddings.unlink()

	assert paths.metadata.is_file()
	assert paths.valid_tokens.is_file()
	with pytest.raises(FileNotFoundError, match='embedding source is missing'):
		summarize_volve_horizon_five_way(config)
	assert not config.summary_root.exists()


def test_missing_embedding_model_source_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		embedding = identity['embedding']
		assert isinstance(embedding, dict)
		embedding.pop('model_source')

	rewrite_five_way_completed_run(config, 'mae', 'layout_000', 'small', mutate)

	with pytest.raises(TypeError, match='model_source must be a mapping'):
		inspect_volve_horizon_five_way_results(config)


def test_foreign_benchmark_identity_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		identity['benchmark'] = 'mae_vs_random_frozen_v1'

	rewrite_five_way_completed_run(config, 'mae', 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match='benchmark identity must equal'):
		inspect_volve_horizon_five_way_results(config)


def test_foreign_run_identity_schema_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		identity['schema_version'] = 4

	rewrite_five_way_completed_run(config, 'mae', 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match='identity schema_version must equal 3'):
		inspect_volve_horizon_five_way_results(config)


def test_shared_downstream_config_drift_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		training = identity['training']
		assert isinstance(training, dict)
		training['epochs'] = 1

	for model_id in FIVE_WAY_MODEL_IDS:
		rewrite_five_way_completed_run(config, model_id, 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match=r'training\.epochs differs'):
		inspect_volve_horizon_five_way_results(config)


def test_changed_configured_checkpoint_is_rejected(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)
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
	config = write_five_way_completed_matrix(tmp_path)

	def mutate(metrics: dict[str, object]) -> None:
		identity = metrics['benchmark_identity']
		assert isinstance(identity, dict)
		block = identity[identity_key]
		assert isinstance(block, dict)
		block['drift'] = True

	rewrite_five_way_completed_run(config, 'random', 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match=match):
		inspect_volve_horizon_five_way_results(config)


def test_cross_model_evaluation_support_mismatch_is_rejected(
	tmp_path: Path,
) -> None:
	config = write_five_way_completed_matrix(tmp_path)

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

	rewrite_five_way_completed_run(config, 'random', 'layout_000', 'small', mutate)

	with pytest.raises(ValueError, match='support_identity'):
		inspect_volve_horizon_five_way_results(config)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_metric_is_rejected(tmp_path: Path, value: float) -> None:
	config = write_five_way_completed_matrix(tmp_path)
	metrics_path = (
		five_way_job_dir(config, 'mae', 'layout_000', 'small') / 'metrics.json'
	)
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics['test']['primary_common']['macro_mae_samples'] = value
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')

	with pytest.raises(ValueError, match='non-finite'):
		inspect_volve_horizon_five_way_results(config)


def test_best_checkpoint_and_selected_epoch_must_match(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)
	metrics_path = (
		five_way_job_dir(config, 'mae', 'layout_000', 'small') / 'metrics.json'
	)
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics['best_epoch'] = 3
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')

	with pytest.raises(ValueError, match='selected best epoch'):
		inspect_volve_horizon_five_way_results(config)


def test_existing_summary_root_is_never_overwritten(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)
	config.summary_root.mkdir(parents=True)
	marker = config.summary_root / 'keep.txt'
	marker.write_text('keep', encoding='utf-8')

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		summarize_volve_horizon_five_way(config)

	assert marker.read_text(encoding='utf-8') == 'keep'


def test_check_only_cli_writes_nothing(tmp_path: Path) -> None:
	config = write_five_way_completed_matrix(tmp_path)
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
