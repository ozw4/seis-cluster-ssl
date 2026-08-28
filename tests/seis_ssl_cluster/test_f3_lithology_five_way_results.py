from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_MODEL_IDS,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.five_way_results import (
	EXPECTED_AGGREGATION_UNIT,
	SOURCE_IDENTITY_FIELDS,
	SUMMARY_METRICS,
	SUMMARY_OUTPUT_NAMES,
	inspect_f3_lithology_five_way_results,
	summarize_f3_lithology_five_way,
)
from tests.seis_ssl_cluster.helpers_f3_five_way import (
	SURVEY_ID,
	build_five_way_universe,
	write_condition,
)

VALIDATION_VOXEL_COUNT = 496


def _metric_value(
	model_id: str, layout_id: str, data_size: str, metric: str
) -> float:
	model_index = FIVE_WAY_MODEL_IDS.index(model_id)
	layout_index = LAYOUT_IDS.index(layout_id)
	size_index = DATA_SIZES.index(data_size)
	metric_index = SUMMARY_METRICS.index(metric)
	return round(
		0.3
		+ 0.02 * model_index
		+ 0.003 * layout_index
		+ 0.1 * size_index
		+ 0.0005 * metric_index,
		6,
	)


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _write_run(  # noqa: PLR0913
	universe: dict[str, object],
	model_id: str,
	layout_id: str,
	data_size: str,
	*,
	embeddings_sha256: str | None = None,
	aggregation_unit: str = EXPECTED_AGGREGATION_UNIT,
) -> Path:
	job_dir = (
		Path(universe['outputs']['runs_root'])
		/ f'model={model_id}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)
	evaluation = job_dir / 'evaluation'
	decoder = job_dir / 'decoder'
	prediction = job_dir / 'prediction'
	evaluation.mkdir(parents=True, exist_ok=True)
	decoder.mkdir(parents=True, exist_ok=True)
	prediction.mkdir(parents=True, exist_ok=True)
	metrics = {
		metric: _metric_value(model_id, layout_id, data_size, metric)
		for metric in SUMMARY_METRICS
	}
	metrics['evaluation_voxel_count'] = VALIDATION_VOXEL_COUNT
	metrics['aggregation_unit'] = aggregation_unit
	metrics['accuracy'] = 0.9
	(evaluation / 'metrics.json').write_text(
		json.dumps(metrics), encoding='utf-8'
	)
	model = next(
		item
		for item in universe['models']
		if item['model_id'] == model_id
	)
	embeddings_dir = Path(model['embeddings_dir'])
	embeddings = embeddings_dir / f'{SURVEY_ID}.embeddings.npy'
	embedding_metadata = embeddings_dir / f'{SURVEY_ID}.embedding_metadata.json'
	valid_tokens = embeddings_dir / f'{SURVEY_ID}.valid_tokens.npy'
	embeddings_identity = _identity(embeddings)
	if embeddings_sha256 is not None:
		embeddings_identity['sha256'] = embeddings_sha256
	prediction_metadata = prediction / 'prediction_metadata.json'
	prediction_metadata.write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_voxel_prediction',
				'model_tag': model_id,
				'source_identity': {
					'decoder_checkpoint': {
						'path': str(decoder / 'best.pt'),
						'sha256': hashlib.sha256(
							f'{model_id}/{layout_id}/{data_size}'.encode()
						).hexdigest(),
					},
					'artifact_identities': {
						'name': 'f3_voxel_decoder_sources',
						'embeddings': embeddings_identity,
						'embedding_metadata': _identity(embedding_metadata),
						'valid_tokens': _identity(valid_tokens),
					},
				},
			}
		),
		encoding='utf-8',
	)
	(evaluation / 'evaluation_metadata.json').write_text(
		json.dumps(
			{
				'dataset': dict(universe['dataset']),
				'model_tag': model_id,
				'policy': {
					'monitored_class_ids': [3, 5],
					'boundary_tolerances': [1, 2, 4, 8],
					'boundary_region_radii': [1, 2, 4, 8],
					'chunk_size_x': 8,
				},
				'inputs': {'prediction_metadata': _identity(prediction_metadata)},
			}
		),
		encoding='utf-8',
	)
	(decoder / 'resolved_config.json').write_text(
		json.dumps(
			{
				'embeddings': {
					'checkpoint_path': model['checkpoint'],
					'input_dir': model['embeddings_dir'],
					'spec': 'overlap_x64',
				}
			}
		),
		encoding='utf-8',
	)
	condition_dir = (
		Path(universe['section_layout']['dataset_root'])
		/ 'datasets'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ 'voxel_supervision'
	)
	(decoder / 'run_metadata.json').write_text(
		json.dumps(
			{
				'voxel_dataset_metadata': str(
					condition_dir / 'voxel_dataset_metadata.json'
				),
				'train_tile_manifest_sha256': (
					f'{layout_id}-{data_size}-train-manifest'
				),
				'validation_tile_manifest_sha256': (
					f'{layout_id}-{data_size}-validation-manifest'
				),
			}
		),
		encoding='utf-8',
	)
	return job_dir


@pytest.fixture
def results_universe(tmp_path: Path) -> dict[str, object]:
	universe = build_five_way_universe(tmp_path / 'synthetic')
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZES:
			write_condition(universe, layout_id, data_size)
	for model_id in FIVE_WAY_MODEL_IDS:
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				_write_run(universe, model_id, layout_id, data_size)
	return universe


def _files_snapshot(root: Path) -> dict[str, str]:
	return {
		str(path): file_sha256(path)
		for path in sorted(root.rglob('*'))
		if path.is_file()
	}


def test_dry_run_reads_75_jobs_and_writes_nothing(
	results_universe: dict[str, object],
	tmp_path: Path,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	root = Path(results_universe['paths']['artifact_root'])
	config_path = tmp_path / 'five_way.yaml'
	config_path.write_text(yaml.safe_dump(results_universe), encoding='utf-8')
	before = _files_snapshot(root)

	report = inspect_f3_lithology_five_way_results(config)
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py',
			'--config',
			str(config_path),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert report['complete_jobs'] == 75
	assert report['model_order'] == list(FIVE_WAY_MODEL_IDS)
	assert 'complete_jobs: 75' in result.stdout
	assert 'execution: dry-run; summary files skipped' in result.stdout
	assert _files_snapshot(root) == before
	assert not config.summary_root.exists()


def test_summary_writes_exactly_five_output_files(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)

	result = summarize_f3_lithology_five_way(config)

	assert result['complete_jobs'] == 75
	names = sorted(path.name for path in config.summary_root.iterdir())
	assert names == sorted(SUMMARY_OUTPUT_NAMES)
	with (config.summary_root / 'comparison.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		comparison = list(csv.DictReader(handle))
	assert len(comparison) == 75
	assert [row['model_id'] for row in comparison[:15]] == ['mae'] * 15
	with (config.summary_root / 'paired_deltas.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		paired = list(csv.DictReader(handle))
	assert len(paired) == 3 * 5 * 8 * 4
	with (config.summary_root / 'summary_by_size.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		by_size = list(csv.DictReader(handle))
	assert len(by_size) == 3 * 8 * 4
	assert {row['n_layouts'] for row in by_size} == {'5'}
	summary = json.loads(
		(config.summary_root / 'summary.json').read_text(encoding='utf-8')
	)
	assert summary['models'] == list(FIVE_WAY_MODEL_IDS)
	assert summary['job_count'] == 75
	assert summary['primary_metric'] == 'macro_f1'
	assert summary['summary_name'] == 'f3_lithology_mae_local_bt_five_way_v1'


def test_summary_uses_configured_summary_name(
	results_universe: dict[str, object],
) -> None:
	configured_name = 'f3_lithology_mae_local_bt_five_way_v3'
	results_universe['outputs']['summary_name'] = configured_name
	config = f3_lithology_five_way_config_from_mapping(results_universe)

	summarize_f3_lithology_five_way(config)

	summary = json.loads(
		(config.summary_root / 'summary.json').read_text(encoding='utf-8')
	)
	assert summary['summary_name'] == configured_name


def test_paired_deltas_have_correct_signs_and_size_isolation(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	summarize_f3_lithology_five_way(config)

	with (config.summary_root / 'paired_deltas.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		paired = list(csv.DictReader(handle))
	for row in paired:
		left = _metric_value(
			row['left_model'], row['layout_id'], row['data_size'], row['metric']
		)
		right = _metric_value(
			row['right_model'], row['layout_id'], row['data_size'], row['metric']
		)
		assert float(row['left_value']) == pytest.approx(left)
		assert float(row['right_value']) == pytest.approx(right)
		assert float(row['delta']) == pytest.approx(left - right)

	with (config.summary_root / 'summary_by_size.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		by_size = list(csv.DictReader(handle))
	for row in by_size:
		expected = [
			_metric_value(
				'mae_hmm_k6', layout_id, row['data_size'], row['metric']
			)
			- _metric_value('mae', layout_id, row['data_size'], row['metric'])
			for layout_id in LAYOUT_IDS
		]
		if row['comparison_id'] != 'mae_hmm_k6_minus_mae':
			continue
		assert float(row['mean']) == pytest.approx(
			sum(expected) / len(expected)
		)
		assert int(row['n_layouts']) == 5


def test_one_missing_job_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	missing = (
		config.runs_root
		/ 'model=local_barlow_twins/layout=layout_002/size=medium'
		/ 'evaluation/metrics.json'
	)
	missing.unlink()

	with pytest.raises(FileNotFoundError, match='missing 1 of 75'):
		inspect_f3_lithology_five_way_results(config)
	with pytest.raises(FileNotFoundError, match='missing 1 of 75'):
		summarize_f3_lithology_five_way(config)
	assert not config.summary_root.exists()


def test_unexpected_run_directory_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	(config.runs_root / 'model=extra').mkdir()

	with pytest.raises(ValueError, match='unexpected five-way run directory'):
		inspect_f3_lithology_five_way_results(config)


def test_validation_identity_drift_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	metrics_path = (
		config.runs_root
		/ 'model=random/layout=layout_004/size=large/evaluation/metrics.json'
	)
	payload = json.loads(metrics_path.read_text(encoding='utf-8'))
	payload['evaluation_voxel_count'] = VALIDATION_VOXEL_COUNT + 1
	metrics_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='section-layout condition declares'):
		inspect_f3_lithology_five_way_results(config)


def test_validation_mask_drift_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	metadata_path = (
		config.section_layout_dataset_root
		/ 'datasets/layout=layout_001/size=small/voxel_supervision'
		/ 'section_layout_metadata.json'
	)
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	payload['identity']['validation_mask_sha256'] = '1' * 64
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='shared by all 15 conditions'):
		inspect_f3_lithology_five_way_results(config)


def test_supervision_identity_drift_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	run_metadata_path = (
		config.runs_root
		/ 'model=mae_hmm_k6/layout=layout_003/size=large/decoder'
		/ 'run_metadata.json'
	)
	payload = json.loads(run_metadata_path.read_text(encoding='utf-8'))
	payload['train_tile_manifest_sha256'] = 'drifted'
	run_metadata_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='supervision identity differs'):
		inspect_f3_lithology_five_way_results(config)


def test_checkpoint_identity_drift_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	resolved_path = (
		config.runs_root
		/ 'model=mae/layout=layout_000/size=small/decoder/resolved_config.json'
	)
	payload = json.loads(resolved_path.read_text(encoding='utf-8'))
	payload['embeddings']['checkpoint_path'] = results_universe['models'][1][
		'checkpoint'
	]
	resolved_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='checkpoint identity'):
		inspect_f3_lithology_five_way_results(config)


def test_comparison_rows_carry_the_recorded_source_shas(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	summarize_f3_lithology_five_way(config)

	with (config.summary_root / 'comparison.csv').open(
		encoding='utf-8', newline=''
	) as handle:
		comparison = list(csv.DictReader(handle))
	for field in SOURCE_IDENTITY_FIELDS:
		assert all(len(row[field]) == 64 for row in comparison)
	by_model: dict[str, set[str]] = {}
	for row in comparison:
		by_model.setdefault(row['model_id'], set()).add(row['embeddings_sha256'])
	assert set(by_model) == set(FIVE_WAY_MODEL_IDS)
	assert all(len(values) == 1 for values in by_model.values())
	assert len({row['valid_tokens_sha256'] for row in comparison}) == 1
	assert len({row['encoder_checkpoint_sha256'] for row in comparison}) == len(
		FIVE_WAY_MODEL_IDS
	)
	assert len({row['decoder_checkpoint_sha256'] for row in comparison}) == 75
	for row in comparison:
		model = next(
			item
			for item in results_universe['models']
			if item['model_id'] == row['model_id']
		)
		assert row['encoder_checkpoint_sha256'] == file_sha256(
			Path(model['checkpoint'])
		)


def test_embedding_sha_drift_between_jobs_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	for layout_id in LAYOUT_IDS[2:]:
		for data_size in DATA_SIZES:
			_write_run(
				results_universe,
				'local_barlow_twins',
				layout_id,
				data_size,
				embeddings_sha256='b' * 64,
			)

	with pytest.raises(
		ValueError, match=r'local_barlow_twins embeddings_sha256 differs'
	):
		inspect_f3_lithology_five_way_results(config)
	with pytest.raises(ValueError, match=r'embeddings_sha256 differs'):
		summarize_f3_lithology_five_way(config)
	assert not config.summary_root.exists()


def test_reextracted_embedding_metadata_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	metadata_path = (
		Path(results_universe['models'][0]['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	payload = json.loads(metadata_path.read_text(encoding='utf-8'))
	payload['extraction_note'] = 're-extracted in place'
	metadata_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='embedding metadata changed'):
		inspect_f3_lithology_five_way_results(config)


def test_non_unique_voxel_aggregation_rejects_summary(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	_write_run(
		results_universe,
		'mae_hmm_k6',
		LAYOUT_IDS[1],
		DATA_SIZES[0],
		aggregation_unit='validation_slice',
	)

	with pytest.raises(ValueError, match='aggregation_unit must equal'):
		inspect_f3_lithology_five_way_results(config)
	assert not config.summary_root.exists()


def test_summary_refuses_to_overwrite_existing_outputs(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	config.summary_root.mkdir(parents=True)
	(config.summary_root / 'summary.json').write_text('{}', encoding='utf-8')

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		summarize_f3_lithology_five_way(config)


def test_summary_leaves_no_partial_directory_when_a_write_fails(
	results_universe: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	original = Path.write_text
	calls = {'count': 0}

	def failing_write_text(self, data, *args, **kwargs):
		calls['count'] += 1
		if calls['count'] == 3:
			raise OSError('synthetic disk failure')
		return original(self, data, *args, **kwargs)

	monkeypatch.setattr(Path, 'write_text', failing_write_text)

	with pytest.raises(OSError, match='synthetic disk failure'):
		summarize_f3_lithology_five_way(config)

	monkeypatch.undo()
	assert not config.summary_root.exists()
	assert not list(config.summary_root.parent.glob('.summary.staging-*'))

	result = summarize_f3_lithology_five_way(config)
	assert result['complete_jobs'] == 75
	assert sorted(path.name for path in config.summary_root.iterdir()) == sorted(
		SUMMARY_OUTPUT_NAMES
	)


def test_evaluated_voxel_count_must_match_the_condition(
	results_universe: dict[str, object],
) -> None:
	config = f3_lithology_five_way_config_from_mapping(results_universe)
	for model_id in FIVE_WAY_MODEL_IDS:
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				metrics_path = (
					config.runs_root
					/ f'model={model_id}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'evaluation/metrics.json'
				)
				payload = json.loads(metrics_path.read_text(encoding='utf-8'))
				payload['evaluation_voxel_count'] = VALIDATION_VOXEL_COUNT + 999
				metrics_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='section-layout condition declares'):
		inspect_f3_lithology_five_way_results(config)
	assert not config.summary_root.exists()
