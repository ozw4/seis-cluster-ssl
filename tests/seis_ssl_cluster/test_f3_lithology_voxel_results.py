from __future__ import annotations

import csv
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_evaluation import EVALUATION_OUTPUT_FILES
from seis_ssl_cluster.f3.lithology.voxel_results import (
	EXPECTED_MODEL_TAGS,
	FIGURE_NAMES,
	SUMMARY_JSON,
	SUMMARY_MARKDOWN,
	TABLE_NAMES,
	V0_HANDOFF_NAME,
	F3LithologyVoxelResultsConfig,
	F3LithologyVoxelResultsPublishConfig,
	F3LithologyVoxelResultsRun,
	summarize_f3_lithology_voxel_results,
)
from seis_ssl_cluster.models.voxel_decoder import (
	voxel_decoder_architecture_mapping,
)

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.parametrize(
	('mode', 'expected'),
	[
		('positive', ('positive', 'positive')),
		('hold', ('hold', 'hold')),
		('negative', ('negative', 'negative')),
	],
)
def test_complete_decision_fixtures_and_outputs(
	tmp_path: Path, mode: str, expected: tuple[str, str]
) -> None:
	config = _fixture(tmp_path, mode=mode)
	result = summarize_f3_lithology_voxel_results(config)

	assert (result.decoder_value, result.m2a_vs_m1_voxel) == expected
	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))
	assert payload['scope'] == {'split': 'original', 'provisional': True}
	assert payload['prediction_versions'] == {
		'V0': 'voxel-shaped token baseline',
		'V1': 'learned sub-token decoder',
	}
	assert json.dumps(payload, allow_nan=False)
	assert {path.name for path in result.table_paths} == set(TABLE_NAMES)
	assert {path.name for path in result.figure_paths} == set(FIGURE_NAMES)
	assert all(
		path.stat().st_size > 0 for path in (*result.table_paths, *result.figure_paths)
	)


def test_summary_coexists_with_v0_handoff_without_overwrite(tmp_path: Path) -> None:
	config = _fixture(tmp_path, mode='positive')
	config.output_dir.mkdir(parents=True)
	handoff = config.output_dir / V0_HANDOFF_NAME
	handoff_contents = b'completed V0 handoff\n'
	handoff.write_bytes(handoff_contents)

	result = summarize_f3_lithology_voxel_results(config)

	assert result.summary_json.is_file()
	assert handoff.read_bytes() == handoff_contents


@pytest.mark.parametrize(
	'conflict_name',
	[SUMMARY_JSON, SUMMARY_MARKDOWN, 'tables', 'figures', 'unexpected.txt'],
)
def test_rejects_partial_or_conflicting_summary_output_without_touching_handoff(
	tmp_path: Path, conflict_name: str
) -> None:
	config = _fixture(tmp_path, mode='positive')
	config.output_dir.mkdir(parents=True)
	handoff = config.output_dir / V0_HANDOFF_NAME
	handoff_contents = b'completed V0 handoff\n'
	handoff.write_bytes(handoff_contents)
	conflict = config.output_dir / conflict_name
	if conflict_name in {'tables', 'figures'}:
		conflict.mkdir()
	else:
		conflict.write_text('partial\n', encoding='utf-8')

	with pytest.raises(FileExistsError, match='partial or conflicting'):
		summarize_f3_lithology_voxel_results(config)

	assert handoff.read_bytes() == handoff_contents


def test_rejects_missing_one_of_six_runs(tmp_path: Path) -> None:
	config = _fixture(tmp_path, mode='positive')
	with pytest.raises(ValueError, match='incomplete six-run matrix'):
		summarize_f3_lithology_voxel_results(replace(config, runs=config.runs[:-1]))


def test_rejects_one_v1_decoder_architecture_mismatch(tmp_path: Path) -> None:
	config = _fixture(tmp_path, mode='positive')
	run = next(item for item in config.runs if item.key == 'm2a_v1')
	metadata_path = run.input_dir / 'evaluation_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['decoder_architecture']['hidden_channels'] = [16]
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='architecture identity mismatch'):
		summarize_f3_lithology_voxel_results(config)


@pytest.mark.parametrize(
	('field', 'wording'),
	[
		('split', 'split_grid_sha256 identity mismatch'),
		('classes', 'class_order identity mismatch'),
		('count', 'validation_voxel_count identity mismatch'),
	],
)
def test_rejects_shared_evaluation_identity_mismatch(
	tmp_path: Path, field: str, wording: str
) -> None:
	config = _fixture(tmp_path, mode='positive')
	run = config.runs[-1]
	metadata_path = run.input_dir / 'evaluation_metadata.json'
	metrics_path = run.input_dir / 'metrics.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	if field == 'split':
		metadata['inputs']['voxel_split_grid']['sha256'] = 'b' * 64
	elif field == 'classes':
		metrics['class_ids'] = [0, 1, 2, 3, 5, 4]
	else:
		metrics['evaluation_voxel_count'] = 101
		metadata['summary']['unique_validation_voxel_count'] = 101
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')
	if field in {'classes', 'count'}:
		_refresh_output_identity(run.input_dir, 'metrics.json')

	with pytest.raises(ValueError, match=wording):
		summarize_f3_lithology_voxel_results(config)


def test_rejects_relabelled_source_encoder(tmp_path: Path) -> None:
	config = _fixture(tmp_path, mode='positive')
	for run in config.runs:
		if run.model != 'M1':
			continue
		metadata_path = run.input_dir / 'evaluation_metadata.json'
		metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
		metadata['model_tag'] = EXPECTED_MODEL_TAGS['MAE']
		metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='M1 source model identity mismatch'):
		summarize_f3_lithology_voxel_results(config)


def test_delta_tables_include_decoder_encoder_boundary_and_class_conditions(
	tmp_path: Path,
) -> None:
	result = summarize_f3_lithology_voxel_results(_fixture(tmp_path, mode='positive'))
	decoder = _csv_rows(result.table_paths[1])
	encoder = _csv_rows(result.table_paths[2])
	classes = _csv_rows(result.table_paths[3])

	assert float(
		next(row for row in decoder if row['candidate_model'] == 'M1')['macro_f1']
	) == pytest.approx(0.05)
	primary = next(row for row in encoder if row['role'] == 'primary')
	assert primary['comparison'] == 'M2-A V1 - M1 V1'
	assert float(primary['mean_iou']) > 0
	class_3 = next(
		row
		for row in classes
		if row['comparison'] == primary['comparison'] and row['class_id'] == '3'
	)
	assert float(class_3['f1']) > 0
	assert float(class_3['iou']) > 0
	assert float(class_3['boundary_recall_t4']) > 0


def test_monitored_class_delta_schema_preserves_source_identity_and_order(
	tmp_path: Path,
) -> None:
	result = summarize_f3_lithology_voxel_results(_fixture(tmp_path, mode='positive'))
	decoder = _csv_rows(result.table_paths[1])
	encoder = _csv_rows(result.table_paths[2])
	classes = _csv_rows(result.table_paths[3])

	assert list(classes[0]) == [
		'comparison',
		'role',
		'baseline_model',
		'baseline_version',
		'candidate_model',
		'candidate_version',
		'class_id',
		'f1',
		'iou',
		'boundary_recall_t2',
		'boundary_recall_t4',
	]
	expected = [
		(source, class_id)
		for source in (*decoder, *encoder)
		for class_id in ('3', '5')
	]
	assert [
		(
			row['comparison'],
			row['role'],
			row['baseline_model'],
			row['baseline_version'],
			row['candidate_model'],
			row['candidate_version'],
			row['class_id'],
		)
		for row in classes
	] == [
		(
			source['comparison'],
			source['role'],
			source['baseline_model'],
			source['baseline_version'],
			source['candidate_model'],
			source['candidate_version'],
			class_id,
		)
		for source, class_id in expected
	]
	for row, (source, class_id) in zip(classes, expected, strict=True):
		for target, source_key in (
			('f1', f'class_{class_id}_f1'),
			('iou', f'class_{class_id}_iou'),
			(
				'boundary_recall_t2',
				f'class_{class_id}_boundary_recall_t2',
			),
			(
				'boundary_recall_t4',
				f'class_{class_id}_boundary_recall_t4',
			),
		):
			assert float(row[target]) == float(source[source_key])


def test_m2a_decision_boundary_f1_gate_independently_forces_hold(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path, mode='positive')
	m2a_v0 = next(
		run for run in config.runs if (run.model, run.version) == ('M2-A', 'V0')
	)
	m2a_v1 = next(
		run for run in config.runs if (run.model, run.version) == ('M2-A', 'V1')
	)
	boundary_v0_path = m2a_v0.input_dir / 'boundary_metrics.json'
	boundary_v0 = json.loads(boundary_v0_path.read_text(encoding='utf-8'))
	boundary_v0['vertical_boundary_f1_at_2'] = 0.49
	boundary_v0['vertical_boundary_f1_at_4'] = 0.49
	boundary_v0_path.write_text(json.dumps(boundary_v0), encoding='utf-8')
	_refresh_output_identity(m2a_v0.input_dir, 'boundary_metrics.json')
	boundary_path = m2a_v1.input_dir / 'boundary_metrics.json'
	boundary = json.loads(boundary_path.read_text(encoding='utf-8'))
	boundary['vertical_boundary_f1_at_2'] = 0.50
	boundary['vertical_boundary_f1_at_4'] = 0.50
	boundary_path.write_text(json.dumps(boundary), encoding='utf-8')
	_refresh_output_identity(m2a_v1.input_dir, 'boundary_metrics.json')

	result = summarize_f3_lithology_voxel_results(config)

	assert result.decoder_value == 'positive'
	assert result.m2a_vs_m1_voxel == 'hold'


def test_m2a_decision_monitored_class_gate_independently_forces_hold(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path, mode='positive')
	m2a_v1 = next(
		run for run in config.runs if (run.model, run.version) == ('M2-A', 'V1')
	)
	metrics_path = m2a_v1.input_dir / 'metrics.json'
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	for class_id in ('3', '5'):
		metrics['per_class_f1'][class_id] = 0.50
		metrics['per_class_iou'][class_id] = 0.50
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')
	_refresh_output_identity(m2a_v1.input_dir, 'metrics.json')

	result = summarize_f3_lithology_voxel_results(config)

	assert result.decoder_value == 'positive'
	assert result.m2a_vs_m1_voxel == 'hold'


def test_publish_uses_exact_lightweight_allowlist(tmp_path: Path) -> None:
	config = _fixture(tmp_path, mode='positive')
	publish_dir = tmp_path / 'reports' / 'f3' / 'voxel'
	config = replace(
		config,
		publish=F3LithologyVoxelResultsPublishConfig(
			enabled=True,
			results_root=tmp_path / 'reports',
			output_dir=publish_dir,
		),
	)
	result = summarize_f3_lithology_voxel_results(config)

	targets = {
		path.relative_to(publish_dir).as_posix()
		for path in result.published_files
	}
	assert targets == {
		SUMMARY_JSON,
		'voxel_results_summary.md',
		*(f'tables/{name}' for name in TABLE_NAMES),
		*(f'figures/{name}' for name in FIGURE_NAMES),
	}
	assert not any(target.endswith(('.pt', '.npy', '.npz')) for target in targets)


@pytest.mark.parametrize(
	('field', 'value', 'wording'),
	[
		('artifact_type', 'not-an-evaluation', 'artifact_type mismatch'),
		('schema_version', 1, 'schema_version mismatch'),
		('aggregation', {}, 'unique-validation-voxel aggregation mismatch'),
	],
)
def test_rejects_noncanonical_evaluation_contract(
	tmp_path: Path, field: str, value: object, wording: str
) -> None:
	config = _fixture(tmp_path, mode='positive')
	metadata_path = config.runs[0].input_dir / 'evaluation_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata[field] = value
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match=wording):
		summarize_f3_lithology_voxel_results(config)


@pytest.mark.parametrize(
	'output_name',
	[
		'metrics.json',
		'boundary_metrics.json',
		'boundary_region_metrics.csv',
	],
)
def test_rejects_evaluation_output_modified_after_metadata(
	tmp_path: Path, output_name: str
) -> None:
	config = _fixture(tmp_path, mode='positive')
	path = config.runs[0].input_dir / output_name
	path.write_bytes(path.read_bytes() + b' ')

	with pytest.raises(ValueError, match=f'{output_name} hash identity mismatch'):
		summarize_f3_lithology_voxel_results(config)


def _fixture(tmp_path: Path, *, mode: str) -> F3LithologyVoxelResultsConfig:
	runs = []
	for model in ('MAE', 'M1', 'M2-A'):
		for version in ('V0', 'V1'):
			input_dir = tmp_path / 'artifacts' / model.replace('-', '') / version
			_write_run(input_dir, model=model, version=version, mode=mode)
			runs.append(F3LithologyVoxelResultsRun(model, version, input_dir))
	return F3LithologyVoxelResultsConfig(
		runs=tuple(runs),
		output_dir=tmp_path / 'artifacts' / 'summary',
	)


def _write_run(path: Path, *, model: str, version: str, mode: str) -> None:
	path.mkdir(parents=True)
	model_base = {
		'positive': {'MAE': 0.40, 'M1': 0.50, 'M2-A': 0.55},
		'hold': {'MAE': 0.40, 'M1': 0.50, 'M2-A': 0.50},
		'negative': {'MAE': 0.60, 'M1': 0.55, 'M2-A': 0.45},
	}[mode][model]
	version_delta = (
		{
			'positive': 0.05,
			'hold': 0.0 if model != 'M1' else 0.01,
			'negative': -0.05,
		}[mode]
		if version == 'V1'
		else 0.0
	)
	score = model_base + version_delta
	boundary_position = 1.0 - score
	metrics = {
		'macro_f1': score,
		'mean_iou': score
		- (0.02 if mode == 'hold' and model == 'M1' and version == 'V1' else 0),
		'balanced_accuracy': score,
		'evaluation_voxel_count': 100,
		'aggregation_unit': 'unique_validation_voxel',
		'class_ids': [0, 1, 2, 3, 4, 5],
		'per_class_f1': {str(class_id): score for class_id in range(6)},
		'per_class_iou': {str(class_id): score for class_id in range(6)},
	}
	boundary = {
		'vertical_boundary_f1_at_2': score,
		'vertical_boundary_f1_at_4': score,
		'vertical_boundary_position_mae_at_4': boundary_position,
	}
	for class_id in (3, 5):
		for tolerance in (2, 4):
			boundary[f'vertical_boundary_class_{class_id}_recall_at_{tolerance}'] = (
				score
			)
	(path / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
	(path / 'boundary_metrics.json').write_text(json.dumps(boundary), encoding='utf-8')
	with (path / 'boundary_region_metrics.csv').open(
		'w', newline='', encoding='utf-8'
	) as handle:
		writer = csv.DictWriter(
			handle, fieldnames=('region', 'radius', 'macro_f1', 'mean_iou')
		)
		writer.writeheader()
		for radius in (2, 4):
			writer.writerow(
				{
					'region': 'boundary',
					'radius': radius,
					'macro_f1': score,
					'mean_iou': score,
				}
			)
	for name in EVALUATION_OUTPUT_FILES:
		output = path / name
		if not output.exists():
			output.write_text('{}\n' if output.suffix == '.json' else '\n')
	metadata = {
		'artifact_type': 'f3_lithology_voxel_evaluation',
		'schema_version': 2,
		'prediction_kind': (
			'token_projection_nearest'
			if version == 'V0'
			else 'frozen_embedding_decoder'
		),
		'model_tag': EXPECTED_MODEL_TAGS[model],
		'aggregation': {
			'primary_unit': 'unique_validation_voxel',
			'split_code': 2,
			'intersection_voxels_counted_once': True,
			'per_slice_planes_evaluated_independently': True,
			'voxel_independence_p_values_computed': False,
		},
		'inputs': {'voxel_split_grid': {'sha256': 'a' * 64}},
		'summary': {'unique_validation_voxel_count': 100},
		'outputs': {
			name: {'path': str(path / name), 'sha256': file_sha256(path / name)}
			for name in EVALUATION_OUTPUT_FILES
		},
	}
	if version == 'V1':
		metadata['decoder_architecture'] = voxel_decoder_architecture_mapping(
			embedding_dim=4,
			class_count=6,
			hidden_channels=(8,),
			upsample_factors=((5, 5, 5),),
		)
	(path / 'evaluation_metadata.json').write_text(
		json.dumps(metadata), encoding='utf-8'
	)


def _refresh_output_identity(root: Path, name: str) -> None:
	metadata_path = root / 'evaluation_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['outputs'][name] = {
		'path': str(root / name),
		'sha256': file_sha256(root / name),
	}
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')


def _csv_rows(path: Path) -> list[dict[str, str]]:
	with path.open(newline='', encoding='utf-8') as handle:
		return list(csv.DictReader(handle))
