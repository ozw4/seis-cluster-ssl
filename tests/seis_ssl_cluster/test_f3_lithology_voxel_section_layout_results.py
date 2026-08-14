
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import seis_ssl_cluster.f3.lithology.voxel_section_layout_results as results
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_roster import (
	EXPECTED_MODEL_IDS,
	EXPECTED_MODEL_ROSTER,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import METRIC_SPECS

if TYPE_CHECKING:
	from collections.abc import Mapping


@pytest.mark.parametrize(
	('medium', 'large', 'expected'),
	[
		((0.01, 0.01, 4), (0.02, 0.02, 4), 'SECTION_LAYOUT_GO'),
		((0.01, 0.01, 4), (0.0, 0.0, 5), 'SECTION_LAYOUT_HOLD'),
		((-0.01, -0.01, 1), (-0.02, -0.02, 1), 'SECTION_LAYOUT_STOP'),
	],
)
def test_exact_go_hold_stop(
	medium: tuple[float, float, int],
	large: tuple[float, float, int],
	expected: str,
) -> None:
	rows = _gate_rows(small=(0.0, 0.0, 0), medium=medium, large=large)
	decision = results.decide_f3_lithology_voxel_section_layout_parent_status(
		rows, comparison_id='candidate-minus-parent'
	)
	assert decision['status'] == expected


@pytest.mark.parametrize(
	('evidence', 'expected'),
	[
		((0.01, -0.001, 5), 'SECTION_LAYOUT_HOLD'),
		((0.01, 0.01, 4), 'SECTION_LAYOUT_GO'),
		((0.01, 0.01, 3), 'SECTION_LAYOUT_HOLD'),
		((-0.01, -0.01, 1), 'SECTION_LAYOUT_STOP'),
		((-0.01, -0.01, 2), 'SECTION_LAYOUT_HOLD'),
	],
)
def test_strict_primary_boundaries(
	evidence: tuple[float, float, int], expected: str
) -> None:
	rows = _gate_rows(small=evidence, medium=evidence, large=evidence)
	decision = results.decide_f3_lithology_voxel_section_layout_parent_status(
		rows, comparison_id='candidate-minus-parent'
	)
	assert decision['status'] == expected


def test_small_positive_alone_is_hold() -> None:
	rows = _gate_rows(
		small=(0.02, 0.02, 5), medium=(0.0, 0.0, 0), large=(0.0, 0.0, 0)
	)
	decision = results.decide_f3_lithology_voxel_section_layout_parent_status(
		rows, comparison_id='candidate-minus-parent'
	)
	assert decision['status'] == 'SECTION_LAYOUT_HOLD'


def test_guardrail_exact_threshold_two_sizes_and_ignores_other_classes() -> None:
	rows = _gate_rows(
		small=(0.01, 0.01, 5),
		medium=(0.01, 0.01, 5),
		large=(0.01, 0.01, 5),
	)
	for row in rows:
		if row['metric'] == 'class_3_f1' and row['data_size'] in {'small', 'large'}:
			row['mean_delta'] = -0.05
	decision = results.decide_f3_lithology_voxel_section_layout_parent_status(
		rows, comparison_id='candidate-minus-parent'
	)
	assert decision['status'] == 'SECTION_LAYOUT_STOP'
	assert decision['systematic_major_degradation'] == [
		{'class_id': 3, 'metric': 'f1', 'data_sizes': ['small', 'large']}
	]

	other_class = _gate_rows(
		small=(0.01, 0.01, 5),
		medium=(0.01, 0.01, 5),
		large=(0.01, 0.01, 5),
	)
	other_class.extend(
		{
			'comparison_id': 'candidate-minus-parent',
			'data_size': size,
			'metric': 'class_2_f1',
			'mean_delta': -1.0,
			'median_delta': -1.0,
			'wins': 0,
		}
		for size in DATA_SIZES
	)
	decision = results.decide_f3_lithology_voxel_section_layout_parent_status(
		other_class, comparison_id='candidate-minus-parent'
	)
	assert decision['status'] == 'SECTION_LAYOUT_GO'


def test_summary_statistics_use_five_layouts_and_sample_standard_deviation() -> None:
	rows = []
	values = (-0.02, -0.01, 0.0, 0.01, 0.02)
	for layout, value in zip(LAYOUT_IDS, values, strict=True):
		rows.extend(
				{
					'comparison_id': 'candidate-minus-parent',
					'model_id': 'candidate',
					'reference_model_id': 'parent',
					'comparison_roles': ['parent'],
					'layout_id': layout,
					'data_size': 'small',
					'metric': metric.name,
					'higher_is_better': metric.higher_is_better,
					'delta': value,
				}
				for metric in METRIC_SPECS
			)
	# Complete medium/large so the generic helper retains its exact matrix contract.
	for size in ('medium', 'large'):
		for layout in LAYOUT_IDS:
			rows.extend(
					{
						'comparison_id': 'candidate-minus-parent',
						'model_id': 'candidate',
						'reference_model_id': 'parent',
						'comparison_roles': ['parent'],
						'layout_id': layout,
						'data_size': size,
						'metric': metric.name,
						'higher_is_better': metric.higher_is_better,
						'delta': 0.01,
					}
					for metric in METRIC_SPECS
				)
	summary = results._summary_by_size(rows)  # noqa: SLF001
	row = next(
		item
		for item in summary
		if item['data_size'] == 'small' and item['metric'] == 'macro_f1'
	)
	assert row['mean_delta'] == pytest.approx(0.0)
	assert row['median_delta'] == 0.0
	assert row['sample_standard_deviation'] == pytest.approx(0.0158113883)
	assert (row['wins'], row['ties'], row['losses']) == (2, 1, 2)
	assert tuple(cast('Mapping[str, float]', row['per_layout_delta'])) == LAYOUT_IDS


def test_model_mode_validates_exact_matrix_pairs_and_diagnostic_eligibility(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, rows_by_model = _inspection_fixture(tmp_path, monkeypatch)
	inspection = results.inspect_f3_lithology_voxel_section_layout_results(
		config, model_id='m1_distill_only'
	)
	assert inspection.loaded_model_ids == ('mae', 'm1_k6', 'm1_distill_only')
	assert inspection.pair_identity_validation == {
		'status': 'PASS',
		'fields': list(results.PAIR_IDENTITY_FIELDS),
		'validated_pair_count': 30,
	}
	decision = inspection.model_decisions['m1_distill_only']
	assert decision['formal_status'] == 'SECTION_LAYOUT_GO'
	assert decision['formal_status_computed'] is True
	assert decision['metrics_included'] is True
	assert decision['selection_eligible'] is False
	assert len(rows_by_model['m1_distill_only']) == 15


@pytest.mark.parametrize(
	'drift', ['duplicate', 'missing', 'wrong_model', 'incomplete', 'nonfinite']
)
def test_manifest_and_metric_drift_are_rejected(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	drift: str,
) -> None:
	config, rows_by_model = _inspection_fixture(tmp_path, monkeypatch)
	rows = rows_by_model['m1_distill_only']
	if drift == 'duplicate':
		rows[-1] = deepcopy(rows[0])
	elif drift == 'missing':
		rows.pop()
	elif drift == 'wrong_model':
		rows[0]['model_id'] = 'mae'
	elif drift == 'incomplete':
		rows[0]['status'] = 'failed'
	else:
		monkeypatch.setattr(
			results,
			'load_f3_lithology_voxel_label_budget_evaluation_metrics',
			lambda **_kwargs: _metrics(float('inf')),
		)
	with pytest.raises(
		ValueError, match=r'15 conditions|model identity|incomplete|non-finite'
	):
		results.inspect_f3_lithology_voxel_section_layout_results(
			config, model_id='m1_distill_only'
		)


def test_pair_identity_mismatch_refuses_delta(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, rows_by_model = _inspection_fixture(tmp_path, monkeypatch)
	rows_by_model['m1_distill_only'][0]['sampling_sequence_sha256'] = 'foreign'
	with pytest.raises(
		ValueError, match=r'paired identity mismatch.*sampling_sequence'
	):
		results.inspect_f3_lithology_voxel_section_layout_results(
			config, model_id='m1_distill_only'
		)


def test_model_mode_no_publish_writes_only_lightweight_artifact_set(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = results.f3_lithology_voxel_section_layout_results_config_from_mapping(
		_config_mapping(tmp_path)
	)
	inspection = _minimal_inspection()
	monkeypatch.setattr(
		results,
		'inspect_f3_lithology_voxel_section_layout_results',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		results,
		'_git_state',
		lambda _root: {'commit': 'a' * 40, 'dirty': False},
	)
	with pytest.raises(ValueError, match='requires --no-publish'):
		results.summarize_f3_lithology_voxel_section_layout_results(
			config, model_id='m1_distill_only'
		)
	result = results.summarize_f3_lithology_voxel_section_layout_results(
		config, model_id='m1_distill_only', no_publish=True
	)
	assert result.output_dir.is_relative_to(config.artifact_root)
	assert {path.name for path in result.files} == set(results.MODEL_OUTPUT_NAMES)
	assert {path.suffix for path in result.files} <= {'.csv', '.json', '.md'}
	assert not config.report_dir.exists()
	assert not any('publish' in path.name for path in result.files)
	assert not (result.output_dir / 'section_layout_handoff.json').exists()
	review = json.loads(
		(result.output_dir / 'section_layout_model_review.json').read_text()
	)
	assert review['artifact_type'] == 'f3_voxel_section_layout_model_review'
	assert review['status'] == 'COMPLETE'
	assert review['scope'] == 'single_model'
	assert review['benchmark_complete'] is False
	assert review['reviewed_model_id'] == 'm1_distill_only'
	assert review['project_adoption'] == 'PENDING_REVIEW'
	assert review['completed_model_count'] == 1
	assert review['completed_job_count'] == 1
	assert review['pair_identity_validation']['status'] == 'PASS'


def test_final_handoff_is_explicitly_full_roster(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = results.f3_lithology_voxel_section_layout_results_config_from_mapping(
		_config_mapping(tmp_path)
	)
	inspection = replace(
		_minimal_inspection(),
		mode='final',
		requested_model_id=None,
		loaded_model_ids=EXPECTED_MODEL_IDS,
		job_metrics=tuple(
			{'model_id': model_id}
			for model_id in EXPECTED_MODEL_IDS
			for _layout_id in LAYOUT_IDS
			for _data_size in DATA_SIZES
		),
	)
	monkeypatch.setattr(
		results,
		'inspect_f3_lithology_voxel_section_layout_results',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		results,
		'_git_state',
		lambda _root: {'commit': 'a' * 40, 'dirty': False},
	)
	result = results.summarize_f3_lithology_voxel_section_layout_results(config)
	assert {path.name for path in result.files} == set(results.FINAL_OUTPUT_NAMES)
	handoff = json.loads(
		(result.output_dir / 'section_layout_handoff.json').read_text()
	)
	assert handoff['artifact_type'] == results.HANDOFF_ARTIFACT_TYPE
	assert handoff['status'] == 'PASS'
	assert handoff['scope'] == 'full_roster'
	assert handoff['benchmark_complete'] is True


def test_final_mode_requires_every_roster_manifest(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = results.f3_lithology_voxel_section_layout_results_config_from_mapping(
		_config_mapping(tmp_path)
	)
	contract = tmp_path / 'contract.json'
	contract.write_text('{}\n', encoding='utf-8')
	dataset = {
		'source_identities': {
			'section_layout_contract': {
				'path': str(contract.resolve()),
				'sha256': file_sha256(contract),
			}
		}
	}
	_write_json(config.dataset_manifest, dataset)
	roster = _roster_mapping(config.artifact_root)
	_write_json(config.model_roster, roster)
	monkeypatch.setattr(results, 'load_config', lambda _path: roster)
	monkeypatch.setattr(
		results,
		'validate_f3_lithology_voxel_section_layout_manifest',
		lambda _path: dataset,
	)
	with pytest.raises(FileNotFoundError, match='full roster'):
		results.inspect_f3_lithology_voxel_section_layout_results(config)


def test_results_config_is_closed(tmp_path: Path) -> None:
	mapping = _config_mapping(tmp_path)
	config = results.f3_lithology_voxel_section_layout_results_config_from_mapping(
		mapping
	)
	assert config.report_dir.is_relative_to(config.workspace_root / 'reports')
	mapping['decision'] = {'threshold': 0.1}
	with pytest.raises(ValueError, match='not allowed'):
		results.f3_lithology_voxel_section_layout_results_config_from_mapping(mapping)


def test_results_config_rejects_old_final_results_dir_key(tmp_path: Path) -> None:
	mapping = _config_mapping(tmp_path)
	outputs = cast('dict[str, object]', mapping['outputs'])
	outputs['final_results_dir'] = outputs.pop('report_dir')
	with pytest.raises(
		ValueError,
		match=r'outputs key\(s\) not allowed:.*final_results_dir',
	):
		results.f3_lithology_voxel_section_layout_results_config_from_mapping(mapping)


def _gate_rows(
	*,
	small: tuple[float, float, int],
	medium: tuple[float, float, int],
	large: tuple[float, float, int],
) -> list[dict[str, object]]:
	rows = []
	for size, evidence in zip(DATA_SIZES, (small, medium, large), strict=True):
		mean, median, wins = evidence
		rows.extend(
				{
					'comparison_id': 'candidate-minus-parent',
					'data_size': size,
					'metric': metric,
					'mean_delta': mean,
					'median_delta': median,
					'wins': wins,
				}
				for metric in results.PRIMARY_METRICS
			)
		for class_id in results.MONITORED_CLASS_IDS:
			rows.extend(
					{
						'comparison_id': 'candidate-minus-parent',
						'data_size': size,
						'metric': f'class_{class_id}_{metric}',
						'mean_delta': 0.0,
						'median_delta': 0.0,
						'wins': 0,
					}
					for metric in results.MONITORED_METRICS
				)
	return rows


def _inspection_fixture(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[results.F3SectionLayoutResultsConfig, dict[str, list[dict[str, object]]]]:
	config = results.f3_lithology_voxel_section_layout_results_config_from_mapping(
		_config_mapping(tmp_path)
	)
	contract = tmp_path / 'contract.json'
	contract.write_text('{}\n', encoding='utf-8')
	dataset = {
		'source_identities': {
			'section_layout_contract': {
				'path': str(contract.resolve()),
				'sha256': file_sha256(contract),
			}
		}
	}
	_write_json(config.dataset_manifest, dataset)
	dataset_identity = {
		'path': str(config.dataset_manifest.resolve()),
		'sha256': file_sha256(config.dataset_manifest),
	}
	roster = _roster_mapping(config.artifact_root)
	_write_json(config.model_roster, roster)
	metric_files = {}
	for name, content in (
		('metrics', '{}\n'),
		('boundary_metrics', '{}\n'),
		('boundary_region_metrics', 'region,radius\n'),
		('evaluation_metadata', '{}\n'),
	):
		path = tmp_path / f'{name}.json'
		path.write_text(content, encoding='utf-8')
		metric_files[name] = {'path': str(path.resolve()), 'sha256': file_sha256(path)}
	rows_by_model: dict[str, list[dict[str, object]]] = {}
	for model_id in ('mae', 'm1_k6', 'm1_distill_only'):
		tag = EXPECTED_MODEL_ROSTER[model_id][0]
		rows_by_model[model_id] = [
			_run_row(model_id, tag, layout, size, metric_files)
			for layout in LAYOUT_IDS
			for size in DATA_SIZES
		]
		manifest = (
			config.benchmark_root
			/ 'runs'
			/ f'model={model_id}'
			/ results.RUN_MANIFEST_NAME
		)
		_write_json(
			manifest,
			{
				'scientific_result': True,
				'model': {'model_id': model_id, 'model_tag': tag},
				'dataset_manifest': dataset_identity,
			},
		)
	monkeypatch.setattr(results, 'load_config', lambda _path: roster)
	monkeypatch.setattr(
		results,
		'validate_f3_lithology_voxel_section_layout_manifest',
		lambda _path: dataset,
	)
	monkeypatch.setattr(
		results,
		'load_f3_lithology_voxel_section_layout_rows',
		lambda path: tuple(rows_by_model[_model_id_from_manifest(Path(path))]),
	)
	monkeypatch.setattr(
		results,
		'load_f3_lithology_voxel_label_budget_evaluation_metrics',
		lambda **kwargs: _metrics(
			{
				'mae': 0.5,
				'm1_k6': 0.55,
				'm1_distill_only': 0.6,
			}[
				str(kwargs['label']).split('/')[0]
			]
		),
	)
	return config, rows_by_model


def _run_row(
	model_id: str,
	model_tag: str,
	layout: str,
	size: str,
	metric_files: Mapping[str, object],
) -> dict[str, object]:
	condition = f'{layout}-{size}'
	return {
		'layout_id': layout,
		'data_size': size,
		'model_id': model_id,
		'model_tag': model_tag,
		'status': 'complete',
		'dataset_grid_identity': {'path': f'/grid/{condition}', 'sha256': condition},
		'train_mask_sha256': f'train-{condition}',
		'validation_mask_sha256': 'validation',
		'target_train_voxel_count': 100,
		'actual_train_voxel_count': 100,
		'selected_token_identity_sha256': f'tokens-{condition}',
		'decoder_seed': 42000,
		'initial_decoder_state_sha256': 'initial-state',
		'class_weights': [1.0] * 6,
		'sampling_sequence_sha256': 'sampling',
		'tile_identities': {
			'train': f'train-tiles-{condition}',
			'validation': 'validation-tiles',
		},
		'metric_schema_sha256': 'metric-schema',
		'canonical_metrics_paths': deepcopy(metric_files),
	}


def _metrics(value: float) -> dict[str, float]:
	return {metric.name: value for metric in METRIC_SPECS}


def _model_id_from_manifest(path: Path) -> str:
	return path.parent.name.removeprefix('model=')


def _roster_mapping(artifact_root: Path) -> dict[str, object]:
	return {
		'schema_version': 'f3_voxel_section_layout_model_roster_v1',
		'artifact_root': str(artifact_root),
		'models': [
			{
				'model_id': model_id,
				'model_tag': tag,
				'embedding_root': (
					f'embeddings/f3/facies_benchmark_v1/{tag}/overlap_x16'
				),
				'parent_model_id': parent,
				'selection_role': role,
			}
			for model_id, (tag, parent, role) in EXPECTED_MODEL_ROSTER.items()
		],
	}


def _config_mapping(tmp_path: Path) -> dict[str, object]:
	root = tmp_path.resolve()
	return {
		'paths': {
			'artifact_root': str(root / 'artifacts'),
			'workspace_root': str(root / 'workspace'),
		},
		'references': {
			'model_roster': str(root / 'roster.yaml'),
			'section_layout_dataset_manifest': str(root / 'datasets.json'),
		},
		'outputs': {
			'benchmark_root': str(root / 'artifacts/benchmark'),
			'report_dir': str(root / 'workspace/reports/f3/section-layout'),
		},
	}


def _minimal_inspection() -> results.F3SectionLayoutResultsInspection:
	decision = {
		'model_id': 'm1_distill_only',
		'model_tag': EXPECTED_MODEL_ROSTER['m1_distill_only'][0],
		'parent_model_id': 'mae',
		'selection_role': 'diagnostic',
		'metrics_included': True,
		'formal_status_computed': True,
		'formal_status': 'SECTION_LAYOUT_HOLD',
		'selection_eligible': False,
	}
	return results.F3SectionLayoutResultsInspection(
		mode='model',
		requested_model_id='m1_distill_only',
		loaded_model_ids=('m1_distill_only',),
		job_metrics=({'model_id': 'm1_distill_only', 'macro_f1': 0.5},),
		paired_deltas=(),
		summary_by_size=(),
		model_decisions={'m1_distill_only': decision},
		pair_identity_validation={'status': 'PASS', 'validated_pair_count': 0},
		source_identities={
			'dataset_contract': {'path': '/contract', 'sha256': 'contract'},
			'model_roster': {'path': '/roster', 'sha256': 'roster'},
		},
	)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
