from __future__ import annotations

import csv
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.f3.lithology.voxel_results import (
	FIGURE_NAMES,
	SUMMARY_JSON,
	TABLE_NAMES,
	F3LithologyVoxelResultsConfig,
	F3LithologyVoxelResultsPublishConfig,
	F3LithologyVoxelResultsRun,
	summarize_f3_lithology_voxel_results,
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


def test_rejects_missing_one_of_six_runs(tmp_path: Path) -> None:
	config = _fixture(tmp_path, mode='positive')
	with pytest.raises(ValueError, match='incomplete six-run matrix'):
		summarize_f3_lithology_voxel_results(replace(config, runs=config.runs[:-1]))


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

	with pytest.raises(ValueError, match=wording):
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


def test_publish_uses_exact_lightweight_allowlist(tmp_path: Path) -> None:
	config = _fixture(tmp_path, mode='positive')
	publish_dir = tmp_path / 'results' / 'f3' / 'voxel'
	config = replace(
		config,
		publish=F3LithologyVoxelResultsPublishConfig(
			enabled=True,
			results_root=tmp_path / 'results',
			output_dir=publish_dir,
		),
	)
	result = summarize_f3_lithology_voxel_results(config)

	assert result.publish_manifest is not None
	targets = {
		item.target.relative_to(publish_dir).as_posix()
		for item in result.publish_manifest.items
	}
	assert targets == {
		SUMMARY_JSON,
		'voxel_results_summary.md',
		*(f'tables/{name}' for name in TABLE_NAMES),
		*(f'figures/{name}' for name in FIGURE_NAMES),
	}
	assert not any(target.endswith(('.pt', '.npy', '.npz')) for target in targets)


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
	metadata = {
		'artifact_type': 'f3_lithology_voxel_evaluation',
		'schema_version': 2,
		'prediction_kind': (
			'token_projection_nearest'
			if version == 'V0'
			else 'frozen_embedding_decoder'
		),
		'model_tag': f'{model.lower()}_model',
		'inputs': {'voxel_split_grid': {'sha256': 'a' * 64}},
		'summary': {'unique_validation_voxel_count': 100},
	}
	(path / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
	(path / 'boundary_metrics.json').write_text(json.dumps(boundary), encoding='utf-8')
	(path / 'evaluation_metadata.json').write_text(
		json.dumps(metadata), encoding='utf-8'
	)
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


def _csv_rows(path: Path) -> list[dict[str, str]]:
	with path.open(newline='', encoding='utf-8') as handle:
		return list(csv.DictReader(handle))
