
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import proc.seis_ssl_cluster.run_f3_lithology_voxel_section_layout_suite as runner_cli
import seis_ssl_cluster.f3.lithology.voxel_section_layout_runner as runner
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	FIXED_DECODER_CONTRACT,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_benchmark import (
	F3SectionLayoutBenchmarkConfig,
	f3_lithology_voxel_section_layout_benchmark_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_roster import (
	EXPECTED_MODEL_ROSTER,
	F3SectionLayoutModel,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_section_layout_runner import (
	F3SectionLayoutJob,
	F3SectionLayoutJobPlan,
	F3SectionLayoutStageStatus,
	F3SectionLayoutSuiteInspection,
	_classify_job,
	load_f3_lithology_voxel_section_layout_rows,
	run_f3_lithology_voxel_section_layout_suite,
)

if TYPE_CHECKING:
	from collections.abc import Mapping


def test_arbitrary_roster_model_plans_exact_ordered_fifteen_jobs(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, manifest, roster = _inspection_fixture(tmp_path, monkeypatch)
	inspection = runner.inspect_f3_lithology_voxel_section_layout_suite(
		config, model_id='mh_xy_consensus'
	)
	assert inspection.model.model_id == 'mh_xy_consensus'
	assert len(inspection.jobs) == 15
	assert [(job.layout_id, job.data_size) for job in inspection.jobs] == [
		(f'layout_{index:03d}', size)
		for index in range(5)
		for size in ('small', 'medium', 'large')
	]
	assert {job.output_root for job in inspection.jobs} == {
		config.benchmark_root
		/ 'runs'
		/ 'model=mh_xy_consensus'
		/ f'layout=layout_{index:03d}'
		/ f'size={size}'
		for index in range(5)
		for size in ('small', 'medium', 'large')
	}
	assert manifest['condition_count'] == 15
	assert len(roster['models']) == 14
	assert not config.benchmark_root.exists()


def test_one_model_requirement_filters_and_fixed_seed(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, _manifest, _roster = _inspection_fixture(tmp_path, monkeypatch)
	with pytest.raises(ValueError, match='one non-empty model_id'):
		runner.inspect_f3_lithology_voxel_section_layout_suite(config, model_id='')
	with pytest.raises(ValueError, match='unknown model_id'):
		runner.inspect_f3_lithology_voxel_section_layout_suite(
			config, model_id='not-a-model'
		)
	inspection = runner.inspect_f3_lithology_voxel_section_layout_suite(
		config,
		model_id='mh_xy_consensus',
		layout_id='layout_003',
		data_size='medium',
	)
	assert len(inspection.jobs) == 1
	assert inspection.jobs[0].layout_id == 'layout_003'
	assert inspection.jobs[0].data_size == 'medium'
	assert config.train.seed == 42000
	assert config.train.steps_per_epoch == 440


@pytest.mark.parametrize('drift', ['shape', 'dtype', 'valid_mask', 'model_tag'])
def test_embedding_drift_fails_before_job_classification(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	drift: str,
) -> None:
	config, _manifest, _roster = _inspection_fixture(
		tmp_path, monkeypatch, embedding_drift=drift
	)
	classifications = 0

	def classify(*_args: object) -> None:
		nonlocal classifications
		classifications += 1

	monkeypatch.setattr(runner, '_classify_job', classify)
	with pytest.raises(
		(TypeError, ValueError), match=r'embeddings|valid_tokens|model tag'
	):
		runner.inspect_f3_lithology_voxel_section_layout_suite(
			config, model_id='mh_xy_consensus'
		)
	assert classifications == 0


def test_dataset_matrix_drift_fails_before_job_classification(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, manifest, _roster = _inspection_fixture(tmp_path, monkeypatch)
	manifest['rows'] = cast('list[dict[str, object]]', manifest['rows'])[:-1]
	monkeypatch.setattr(
		runner,
		'validate_f3_lithology_voxel_section_layout_manifest',
		lambda _path: manifest,
	)
	with pytest.raises(ValueError, match='exact ordered 15-row matrix'):
		runner.inspect_f3_lithology_voxel_section_layout_suite(
			config, model_id='mh_xy_consensus'
		)


def test_job_states_new_resume_reuse_invalid_and_foreign(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _config(tmp_path)
	job = _job(tmp_path)
	assert _classify_job(config, job).state == 'NEW'
	job.decoder_dir.mkdir(parents=True)
	assert _classify_job(config, job).state == 'INVALID_OR_PARTIAL'
	(job.decoder_dir / 'latest.pt').write_bytes(b'checkpoint')
	resolved = runner._decoder_config(config, job).to_dict()  # noqa: SLF001
	monkeypatch.setattr(
		runner,
		'load_voxel_decoder_checkpoint',
		lambda _path: {'resolved_config': resolved, 'checkpoint_kind': 'epoch'},
	)
	monkeypatch.setattr(
		runner,
		'validate_f3_lithology_voxel_decoder_resume',
		lambda *_args: {'checkpoint_kind': 'epoch'},
	)
	assert _classify_job(config, job).state == 'RESUME_LATEST'
	monkeypatch.setattr(
		runner,
		'load_voxel_decoder_checkpoint',
		lambda _path: {
			'resolved_config': {'foreign': True},
			'checkpoint_kind': 'epoch',
		},
	)
	plan = _classify_job(config, job)
	assert plan.state == 'INVALID_OR_PARTIAL'
	assert (plan.reason or '').startswith('FOREIGN_IDENTITY:')
	monkeypatch.setattr(
		runner,
		'load_voxel_decoder_checkpoint',
		lambda _path: {'resolved_config': resolved, 'checkpoint_kind': 'completed'},
	)
	monkeypatch.setattr(
		runner,
		'_decoder_stage_evidence',
		lambda *_args: object(),
	)
	monkeypatch.setattr(
		runner,
		'_inspect_prediction_stage',
		lambda *_args: F3SectionLayoutStageStatus('prediction', 'COMPLETE'),
	)
	monkeypatch.setattr(
		runner,
		'_inspect_evaluation_stage',
		lambda *_args: F3SectionLayoutStageStatus('evaluation', 'COMPLETE'),
	)
	assert _classify_job(config, job).state == 'REUSE_COMPLETED'


def test_job_classification_reports_deepest_complete_stage(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _config(tmp_path)
	job = _job(tmp_path)
	job.output_root.mkdir(parents=True)
	complete_decoder = F3SectionLayoutStageStatus('decoder', 'COMPLETE')
	monkeypatch.setattr(
		runner, '_inspect_decoder_stage', lambda *_args: complete_decoder
	)
	monkeypatch.setattr(runner, '_inspect_generated_configs', lambda *_args: None)
	monkeypatch.setattr(
		runner,
		'_inspect_prediction_stage',
		lambda *_args: F3SectionLayoutStageStatus('prediction', 'MISSING'),
	)
	plan = _classify_job(config, job)
	assert plan.state == 'DECODER_COMPLETE'
	assert plan.decoder_stage == complete_decoder

	monkeypatch.setattr(
		runner,
		'_inspect_prediction_stage',
		lambda *_args: F3SectionLayoutStageStatus('prediction', 'COMPLETE'),
	)
	monkeypatch.setattr(
		runner,
		'_inspect_evaluation_stage',
		lambda *_args: F3SectionLayoutStageStatus('evaluation', 'MISSING'),
	)
	assert _classify_job(config, job).state == 'PREDICTION_COMPLETE'
	monkeypatch.setattr(
		runner,
		'_inspect_evaluation_stage',
		lambda *_args: F3SectionLayoutStageStatus('evaluation', 'COMPLETE'),
	)
	assert _classify_job(config, job).state == 'REUSE_COMPLETED'


def test_completed_decoder_rejects_current_source_identity_drift_before_run(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _config(tmp_path)
	base = _job(tmp_path)
	row = deepcopy(base.dataset_row)
	outputs = cast('dict[str, object]', row['outputs'])
	outputs['voxel_dataset_metadata.json'] = {
		'path': str(base.dataset_root / 'voxel_dataset_metadata.json'),
		'sha256': '1' * 64,
	}
	outputs['supervision_split_grid.npy'] = {
		'path': str(base.dataset_root / 'supervision_split_grid.npy'),
		'sha256': '2' * 64,
	}
	embedding_identities = {
		'embeddings': {'path': '/embeddings.npy', 'sha256': '3' * 64},
		'embedding_metadata': {'path': '/metadata.json', 'sha256': '4' * 64},
		'valid_tokens': {'path': '/valid_tokens.npy', 'sha256': '5' * 64},
	}
	label_path = Path(cast('str', config.labels['source_label_volume']))
	job = replace(
		base,
		dataset_row=row,
		embedding_identity=embedding_identities,
		dataset_source_identities={
			'label_volume': {'path': str(label_path), 'sha256': '6' * 64}
		},
	)
	job.decoder_dir.mkdir(parents=True)
	best_path = job.decoder_dir / 'best.pt'
	best_path.write_bytes(b'best decoder')
	(job.decoder_dir / 'latest.pt').write_bytes(b'latest decoder')
	(job.decoder_dir / 'history.csv').write_text('epoch\n', encoding='utf-8')
	resolved = runner._decoder_config(config, job).to_dict()  # noqa: SLF001
	(job.decoder_dir / 'resolved_config.json').write_text(
		json.dumps(resolved), encoding='utf-8'
	)
	persisted_artifacts = runner._expected_decoder_artifact_identities(  # noqa: SLF001
		config, job
	)
	persisted_artifacts['voxel_split_grid'] = {
		'path': str(job.dataset_root / 'supervision_split_grid.npy'),
		'sha256': '0' * 64,
	}
	payload = {
		'checkpoint_kind': 'completed',
		'resolved_config': resolved,
		'best_checkpoint_sha256': file_sha256(best_path),
		'global_step': 50 * 440,
		'artifact_identities': persisted_artifacts,
		'tile_manifest_hashes': {'train': '7' * 64, 'validation': '8' * 64},
	}
	monkeypatch.setattr(runner, 'load_voxel_decoder_checkpoint', lambda _path: payload)
	monkeypatch.setattr(
		runner,
		'_run_job',
		lambda *_args, **_kwargs: pytest.fail('foreign decoder must not run'),
	)

	plan = _classify_job(config, job)

	assert plan.state == 'INVALID_OR_PARTIAL'
	assert (plan.reason or '').startswith('FOREIGN_IDENTITY:')
	assert plan.decoder_stage is not None
	assert plan.decoder_stage.foreign_identity is True
	inspection = _inspection((job,))
	inspection = replace(inspection, plans=(plan,))
	monkeypatch.setattr(
		runner,
		'inspect_f3_lithology_voxel_section_layout_suite',
		lambda *_args, **_kwargs: inspection,
	)
	with pytest.raises(ValueError, match='FOREIGN_IDENTITY'):
		run_f3_lithology_voxel_section_layout_suite(config, model_id='mae')


@pytest.mark.parametrize(
	('start_state', 'expected_calls'),
	[
		('DECODER_COMPLETE', ['inference', 'evaluation']),
		('PREDICTION_COMPLETE', ['evaluation']),
	],
)
def test_run_job_reuses_completed_upstream_stages(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	start_state: str,
	expected_calls: list[str],
) -> None:
	config = _config(tmp_path)
	job = _job(tmp_path)
	job.decoder_dir.mkdir(parents=True)
	(job.decoder_dir / 'best.pt').write_bytes(b'valid completed decoder')
	calls: list[str] = []
	monkeypatch.setattr(
		runner,
		'run_f3_lithology_voxel_decoder',
		lambda *_args, **_kwargs: pytest.fail('completed decoder was retrained'),
	)
	monkeypatch.setattr(
		runner,
		'predict_f3_lithology_voxels',
		lambda *_args, **_kwargs: calls.append('inference'),
	)
	monkeypatch.setattr(
		runner,
		'evaluate_f3_lithology_voxels',
		lambda *_args, **_kwargs: calls.append('evaluation'),
	)
	runner._run_job(  # noqa: SLF001
		config,
		job,
		device='cpu',
		resume=None,
		max_steps=None,
		start_state=start_state,
	)
	assert calls == expected_calls
	assert (job.generated_configs_dir / 'decoder_config.json').is_file()
	assert (job.generated_configs_dir / 'inference_config.json').is_file()
	assert (job.generated_configs_dir / 'evaluation_config.json').is_file()


def test_failure_manifest_is_atomic_and_retains_completed_rows(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _config(tmp_path)
	model_root = config.benchmark_root / 'runs/model=mae'
	first = _job(
		tmp_path,
		layout_id='layout_000',
		data_size='small',
		output_root=model_root / 'layout=layout_000/size=small',
	)
	second = _job(
		tmp_path,
		layout_id='layout_000',
		data_size='medium',
		output_root=model_root / 'layout=layout_000/size=medium',
	)
	inspection = _inspection((first, second))
	monkeypatch.setattr(
		runner,
		'inspect_f3_lithology_voxel_section_layout_suite',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		runner,
		'_classify_job',
		lambda *_args: F3SectionLayoutJobPlan(first, 'NEW'),
	)
	calls = 0

	def execute(*_args: object, **_kwargs: object) -> None:
		nonlocal calls
		calls += 1
		if calls == 2:
			raise RuntimeError('synthetic decoder failure')

	def completed(
		_config: object, job: F3SectionLayoutJob, **_kwargs: object
	) -> dict[str, object]:
		return _complete_row(job)

	monkeypatch.setattr(runner, '_run_job', execute)
	monkeypatch.setattr(runner, '_completed_row', completed)
	with pytest.raises(RuntimeError, match='synthetic decoder failure'):
		run_f3_lithology_voxel_section_layout_suite(config, model_id='mae')
	manifest = model_root / runner.RUN_MANIFEST_NAME
	rows = load_f3_lithology_voxel_section_layout_rows(manifest)
	assert [row['status'] for row in rows] == ['complete', 'failed']
	assert not manifest.with_name(f'.{manifest.name}.tmp').exists()


def test_complete_row_contract_contains_pair_identity_fields(tmp_path: Path) -> None:
	row = _complete_row(_job(tmp_path))
	required = {
		'layout_id',
		'data_size',
		'model_id',
		'model_tag',
		'dataset_grid_identity',
		'train_mask_sha256',
		'validation_mask_sha256',
		'target_train_voxel_count',
		'actual_train_voxel_count',
		'selected_token_identity_sha256',
		'decoder_seed',
		'initial_decoder_state_sha256',
		'class_weights',
		'sampling_sequence_sha256',
		'tile_identities',
		'metric_schema_sha256',
		'best_checkpoint_inference',
		'prediction_exact_once_checks',
		'canonical_metrics_paths',
	}
	assert required <= set(row)
	assert row['decoder_seed'] == 42000


def test_explicit_quarantine_and_smoke_two_steps_use_disjoint_roots(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _config(tmp_path)
	job = _job(
		tmp_path,
		output_root=(config.smoke_root / 'runs/model=mae/layout=layout_000/size=small'),
	)
	job.output_root.mkdir(parents=True)
	(job.output_root / 'partial.txt').write_text('partial\n', encoding='utf-8')
	inspection = _inspection((job,))
	inspection = F3SectionLayoutSuiteInspection(
		model=inspection.model,
		jobs=inspection.jobs,
		plans=(F3SectionLayoutJobPlan(job, 'INVALID_OR_PARTIAL', 'partial'),),
		dataset_manifest_identity=inspection.dataset_manifest_identity,
		embedding_identity=inspection.embedding_identity,
	)
	monkeypatch.setattr(
		runner,
		'inspect_f3_lithology_voxel_section_layout_suite',
		lambda *_args, **_kwargs: inspection,
	)
	max_steps: list[int | None] = []

	def execute(*_args: object, **kwargs: object) -> None:
		max_steps.append(cast('int | None', kwargs['max_steps']))

	monkeypatch.setattr(runner, '_run_job', execute)
	monkeypatch.setattr(runner, '_smoke_row', lambda *_args: _smoke_complete_row(job))
	result = run_f3_lithology_voxel_section_layout_suite(
		config,
		model_id='mae',
		layout_id='layout_000',
		data_size='small',
		only_missing=True,
		quarantine_invalid=True,
		smoke_only=True,
	)
	assert max_steps == [2]
	assert result.manifest_json.is_relative_to(config.smoke_root)
	assert not config.benchmark_root.exists()
	assert len(result.quarantines) == 1
	assert (result.quarantines[0] / 'partial.txt').is_file()


def test_evaluation_recovery_quarantines_only_evaluation_stage(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _config(tmp_path)
	job = _job(
		tmp_path,
		output_root=(
			config.benchmark_root
			/ 'runs/model=mae/layout=layout_000/size=small'
		),
	)
	job.decoder_dir.mkdir(parents=True)
	job.prediction_dir.mkdir()
	job.evaluation_dir.mkdir()
	(job.decoder_dir / 'best.pt').write_bytes(b'completed decoder')
	(job.prediction_dir / 'predictions.npy').write_bytes(b'valid prediction')
	(job.evaluation_dir / 'metrics.json').write_text('{broken', encoding='utf-8')
	base = _inspection((job,))
	inspection = F3SectionLayoutSuiteInspection(
		model=base.model,
		jobs=base.jobs,
		plans=(
			F3SectionLayoutJobPlan(
				job,
				'INVALID_OR_PARTIAL',
				'evaluation validation failed',
				invalid_paths=(job.evaluation_dir,),
				recovery_state='PREDICTION_COMPLETE',
			),
		),
		dataset_manifest_identity=base.dataset_manifest_identity,
		embedding_identity=base.embedding_identity,
	)
	monkeypatch.setattr(
		runner,
		'inspect_f3_lithology_voxel_section_layout_suite',
		lambda *_args, **_kwargs: inspection,
	)
	starts: list[str] = []

	def execute(*_args: object, **kwargs: object) -> None:
		starts.append(cast('str', kwargs['start_state']))
		assert (job.decoder_dir / 'best.pt').is_file()
		assert (job.prediction_dir / 'predictions.npy').is_file()
		assert not job.evaluation_dir.exists()

	monkeypatch.setattr(runner, '_run_job', execute)
	monkeypatch.setattr(
		runner,
		'_completed_row',
		lambda *_args, **_kwargs: _complete_row(job),
	)
	result = run_f3_lithology_voxel_section_layout_suite(
		config,
		model_id='mae',
		only_missing=True,
		quarantine_invalid=True,
	)
	assert starts == ['PREDICTION_COMPLETE']
	assert (job.decoder_dir / 'best.pt').is_file()
	assert (job.prediction_dir / 'predictions.npy').is_file()
	assert len(result.quarantines) == 1
	assert result.quarantines[0].parent == job.output_root
	assert (result.quarantines[0] / 'metrics.json').is_file()


def test_config_is_closed_and_rejects_fixed_decoder_drift(tmp_path: Path) -> None:
	mapping = _config_mapping(tmp_path)
	config = f3_lithology_voxel_section_layout_benchmark_config_from_mapping(mapping)
	assert config.train.seed == 42000
	assert config.train.epochs == 50
	drift = deepcopy(mapping)
	cast('dict[str, object]', drift['train'])['steps_per_epoch'] = 439
	with pytest.raises(ValueError, match='steps_per_epoch'):
		f3_lithology_voxel_section_layout_benchmark_config_from_mapping(drift)
	bool_drift = deepcopy(mapping)
	cast('dict[str, object]', bool_drift['train'])['amp'] = 1
	with pytest.raises(ValueError, match=r'train\.amp'):
		f3_lithology_voxel_section_layout_benchmark_config_from_mapping(bool_drift)
	unknown = deepcopy(mapping)
	unknown['models'] = {}
	with pytest.raises(ValueError, match='not allowed'):
		f3_lithology_voxel_section_layout_benchmark_config_from_mapping(unknown)


def test_cli_dry_run_performs_no_write_or_execution(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	config_path = tmp_path / 'runner.yaml'
	config_path.write_text('synthetic: true\n', encoding='utf-8')
	config = _config(tmp_path)
	job = _job(tmp_path)
	inspection = _inspection((job,))
	monkeypatch.setattr(
		runner_cli, 'load_config', lambda _path: _config_mapping(tmp_path)
	)
	monkeypatch.setattr(
		runner_cli,
		'inspect_f3_lithology_voxel_section_layout_suite',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		runner_cli,
		'run_f3_lithology_voxel_section_layout_suite',
		lambda *_args, **_kwargs: pytest.fail('dry-run executed the suite'),
	)
	monkeypatch.setattr(
		'sys.argv',
		[
			'run_f3_lithology_voxel_section_layout_suite.py',
			'--config',
			str(config_path),
			'--model-id',
			'mae',
			'--dry-run',
		],
	)
	runner_cli.main()
	assert 'execution: dry-run' in capsys.readouterr().out
	assert not config.benchmark_root.exists()
	assert not config.smoke_root.exists()


def _inspection_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	*,
	embedding_drift: str | None = None,
) -> tuple[
	F3SectionLayoutBenchmarkConfig,
	dict[str, object],
	dict[str, object],
]:
	config = _config(tmp_path)
	selected = 'mh_xy_consensus'
	tag = EXPECTED_MODEL_ROSTER[selected][0]
	embedding_root = (
		config.artifact_root / 'embeddings/f3/facies_benchmark_v1' / tag / 'overlap_x16'
	)
	embedding_root.mkdir(parents=True)
	shape = (2, 3, 1, 4)
	valid_shape = shape[:3]
	monkeypatch.setattr(runner, 'EXPECTED_EMBEDDING_SHAPE', shape)
	monkeypatch.setattr(runner, 'EXPECTED_VALID_TOKEN_SHAPE', valid_shape)
	embedding_dtype = np.float32 if embedding_drift == 'dtype' else np.float16
	embedding_shape = (2, 2, 1, 4) if embedding_drift == 'shape' else shape
	valid = np.ones(
		valid_shape, dtype=np.uint8 if embedding_drift == 'valid_mask' else np.bool_
	)
	checkpoint = embedding_root / 'checkpoint.pt'
	checkpoint.write_bytes(b'synthetic checkpoint')
	name = config.dataset['name']
	np.save(
		embedding_root / f'{name}.embeddings.npy',
		np.zeros(embedding_shape, dtype=embedding_dtype),
	)
	np.save(embedding_root / f'{name}.valid_tokens.npy', valid)
	_write_json(
		embedding_root / f'{name}.embedding_metadata.json',
		{
			'model_tag': 'wrong' if embedding_drift == 'model_tag' else tag,
			'checkpoint_path': str(checkpoint),
			'checkpoint_sha256': file_sha256(checkpoint),
		},
	)
	valid_identity = {
		'path': str(embedding_root / f'{name}.valid_tokens.npy'),
		'sha256': file_sha256(embedding_root / f'{name}.valid_tokens.npy'),
	}
	np.save(config.labels['source_label_volume'], np.zeros((1,), dtype=np.uint8))
	label_identity = {
		'path': str(config.labels['source_label_volume']),
		'sha256': file_sha256(config.labels['source_label_volume']),
	}
	rows = [
		_dataset_row(tmp_path, layout, size)
		for layout in range(5)
		for size in ('small', 'medium', 'large')
	]
	manifest: dict[str, object] = {
		'condition_count': 15,
		'source_identities': {
			'reference_valid_tokens': valid_identity,
			'label_volume': label_identity,
		},
		'rows': rows,
	}
	_write_json(config.dataset_manifest, manifest)
	roster = _roster_mapping(config.artifact_root)
	_write_json(config.model_roster, roster)
	monkeypatch.setattr(runner, 'load_config', lambda _path: roster)
	monkeypatch.setattr(
		runner,
		'validate_f3_lithology_voxel_section_layout_manifest',
		lambda _path: manifest,
	)
	rows_by_root = {row['voxel_dataset_root']: row for row in rows}
	monkeypatch.setattr(
		runner,
		'validate_f3_lithology_voxel_section_layout_condition',
		lambda root: rows_by_root[str(root)],
	)
	return config, manifest, roster


def _config(tmp_path: Path) -> F3SectionLayoutBenchmarkConfig:
	return f3_lithology_voxel_section_layout_benchmark_config_from_mapping(
		_config_mapping(tmp_path)
	)


def _config_mapping(tmp_path: Path) -> dict[str, object]:
	root = tmp_path.resolve()
	return {
		'paths': {
			'artifact_root': str(root / 'artifacts'),
			'f3_root': str(root / 'f3'),
			'source_label_volume': str(root / 'labels.npy'),
			'source_label_segy': str(root / 'labels.sgy'),
			'png_label_inventory': str(root / 'inventory.csv'),
			'segy_geometry_json': str(root / 'geometry.json'),
			'class_info': str(root / 'classes.json'),
		},
		'dataset': {'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v1'},
		'references': {
			'model_roster': str(root / 'roster.yaml'),
			'section_layout_dataset_manifest': str(root / 'datasets.json'),
		},
		'decoder': {
			key: _plain(value)
			for key, value in FIXED_DECODER_CONTRACT.items()
			if key
			in {
				'spec',
				'embedding_dim',
				'class_count',
				'hidden_channels',
				'upsample_factors',
				'upsample_mode',
				'normalization',
			}
		},
		'tiles': {'core_size_tokens': [8, 8, 8], 'context_halo_tokens': [1, 1, 1]},
		'train': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 0.001,
			'weight_decay': 0.0001,
			'class_weight': 'balanced',
			'sampling_mode': 'uniform_tiles_with_replacement',
			'steps_per_epoch': 440,
			'seed': 42000,
			'num_workers': 0,
			'amp': True,
			'gradient_clip_norm': 1.0,
		},
		'inference': {'write_probabilities': False},
		'evaluation': {
			'monitored_class_ids': [3, 5],
			'boundary_tolerances': [2, 4],
			'boundary_region_radii': [2, 4],
			'chunk_size_x': 8,
		},
		'outputs': {
			'benchmark_root': str(root / 'benchmark'),
			'smoke_root': str(root / 'smoke'),
		},
	}


def _roster_mapping(artifact_root: Path) -> dict[str, object]:
	models = []
	for model_id, (tag, parent, role) in EXPECTED_MODEL_ROSTER.items():
		root = artifact_root / 'embeddings/f3/facies_benchmark_v1' / tag / 'overlap_x16'
		models.append(
			{
				'model_id': model_id,
				'model_tag': tag,
				'embedding_root': root.relative_to(artifact_root).as_posix(),
				'parent_model_id': parent,
				'selection_role': role,
			}
		)
	return {
		'schema_version': 'f3_voxel_section_layout_model_roster_v1',
		'artifact_root': str(artifact_root),
		'models': models,
	}


def _dataset_row(tmp_path: Path, layout: int, size: str) -> dict[str, object]:
	root = (
		tmp_path
		/ 'datasets'
		/ f'layout=layout_{layout:03d}'
		/ f'size={size}'
		/ 'voxel_supervision'
	)
	identity = {'path': str(root / 'file.npy'), 'sha256': 'a' * 64}
	return {
		'layout_id': f'layout_{layout:03d}',
		'data_size': size,
		'parent_size': None,
		'voxel_dataset_root': str(root),
		'target_train_voxel_count': 100,
		'actual_train_voxel_count': 100,
		'relative_count_error': 0.0,
		'selected_token_count': 1,
		'selected_token_identity_sha256': 'b' * 64,
		'train_mask_sha256': 'c' * 64,
		'validation_mask_sha256': 'd' * 64,
		'per_line_contributions': {'inline:1': 50, 'crossline:1': 50},
		'per_class_train_voxel_counts': {str(index): 1 for index in range(6)},
		'outputs': {
			'supervision_split_grid.npy': identity,
			'selected_token_xyz.npy': identity,
		},
	}


def _job(
	tmp_path: Path,
	*,
	layout_id: str = 'layout_000',
	data_size: str = 'small',
	output_root: Path | None = None,
) -> F3SectionLayoutJob:
	model = F3SectionLayoutModel(
		model_id='mae',
		model_tag=EXPECTED_MODEL_ROSTER['mae'][0],
		embedding_root=tmp_path / 'embeddings',
		parent_model_id=None,
		selection_role='baseline',
	)
	row = _dataset_row(tmp_path, int(layout_id[-3:]), data_size)
	return F3SectionLayoutJob(
		model=model,
		layout_id=layout_id,
		data_size=data_size,
		dataset_root=Path(cast('str', row['voxel_dataset_root'])),
		output_root=output_root or tmp_path / 'job',
		dataset_row=row,
		embedding_identity={'valid_tokens': {'path': '/valid.npy', 'sha256': 'e' * 64}},
	)


def _inspection(
	jobs: tuple[F3SectionLayoutJob, ...],
) -> F3SectionLayoutSuiteInspection:
	return F3SectionLayoutSuiteInspection(
		model=jobs[0].model,
		jobs=jobs,
		plans=tuple(F3SectionLayoutJobPlan(job, 'NEW') for job in jobs),
		dataset_manifest_identity={'path': '/datasets.json', 'sha256': 'f' * 64},
		embedding_identity=jobs[0].embedding_identity,
	)


def _complete_row(job: F3SectionLayoutJob) -> dict[str, object]:
	return {
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'model_id': job.model.model_id,
		'model_tag': job.model.model_tag,
		'status': 'complete',
		'action': 'NEW',
		'dataset_grid_identity': {'path': '/grid.npy', 'sha256': 'a' * 64},
		'train_mask_sha256': 'b' * 64,
		'validation_mask_sha256': 'c' * 64,
		'target_train_voxel_count': 100,
		'actual_train_voxel_count': 100,
		'selected_token_identity_sha256': 'd' * 64,
		'decoder_seed': 42000,
		'initial_decoder_state_sha256': 'e' * 64,
		'class_weights': [1.0] * 6,
		'sampling_sequence_sha256': 'f' * 64,
		'tile_identities': {},
		'metric_schema_sha256': '0' * 64,
		'best_checkpoint_inference': {'kind': 'best'},
		'prediction_exact_once_checks': {'exact_once': True},
		'canonical_metrics_paths': {},
		'error': None,
	}


def _smoke_complete_row(job: F3SectionLayoutJob) -> dict[str, object]:
	return {
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'model_id': job.model.model_id,
		'model_tag': job.model.model_tag,
		'status': 'complete',
		'action': 'SMOKE',
		'scientific_result': False,
		'decoder_seed': 42000,
		'global_step': 2,
		'error': None,
	}


def _plain(value: object) -> object:
	if isinstance(value, tuple):
		return [_plain(item) for item in value]
	return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload), encoding='utf-8')
