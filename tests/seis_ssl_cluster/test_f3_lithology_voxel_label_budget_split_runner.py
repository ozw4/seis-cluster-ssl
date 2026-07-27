from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import seis_ssl_cluster.f3.lithology.voxel_label_budget_split_runner as split_runner
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split import (
	_normalized_source_identities,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split_runner import (
	LowLabelSplitJob,
	_metric_row,
	_selected_complete_triplets,
	_shared_condition_contract,
	_split_labels,
)
from seis_ssl_cluster.f3.splits import read_f3_line_geometry


def _triplet() -> list[dict[str, object]]:
	shared = {
		'split_id': 'split_000',
		'budget_id': 'cap25',
		'voxel_supervision_grid_sha256': 'grid',
		'selected_token_identity_sha256': 'tokens',
		'unique_token_xyz_sha256': 'xyz',
		'train_voxel_count': 100,
		'validation_voxel_count': 200,
		'validation_mask_sha256': 'validation',
		'canonical_valid_token_sha256': 'valid',
		'class_order': [0, 1, 2, 3, 4, 5],
		'class_weights': [1.0] * 6,
		'initial_model_state_sha256': 'initial',
		'decoder_architecture': {'spec': 'fixed'},
		'decoder_seed': 42000,
		'train_tile_manifest_sha256': 'train-manifest',
		'validation_tile_manifest_sha256': 'validation-manifest',
		'train_tile_identity_sha256': 'train-tiles',
		'validation_tile_identity_sha256': 'validation-tiles',
		'sampling_mode': 'uniform_tiles_with_replacement',
		'steps_per_epoch': 440,
		'sampling_sequence_sha256': 'sampling',
		'global_step': 22000,
		'metric_schema_sha256': 'metrics',
	}
	return [
		{**shared, 'model_role': role}
		for role in ('mae', 'm1_current_k6', 'mh_nocons')
	]


def test_full_condition_requires_three_model_paired_decoder_contract() -> None:
	_shared_condition_contract(
		_triplet(),
		models=('mae', 'm1_current_k6', 'mh_nocons'),
		context='full decoder run',
	)


@pytest.mark.parametrize(
	'key',
	[
		'initial_model_state_sha256',
		'class_weights',
		'train_tile_identity_sha256',
		'sampling_sequence_sha256',
	],
)
def test_full_condition_rejects_paired_decoder_contract_mismatch(key: str) -> None:
	rows = _triplet()
	rows[-1][key] = 'mismatch'
	with pytest.raises(ValueError, match=key):
		_shared_condition_contract(
			rows,
			models=('mae', 'm1_current_k6', 'mh_nocons'),
			context='full decoder run',
		)


def test_full_condition_skips_incomplete_selected_triplet() -> None:
	rows = _triplet()[:2]
	jobs = [
		LowLabelSplitJob('split_000', 'cap25', role, output_root=Path('unused'))
		for role in ('mae', 'm1_current_k6', 'mh_nocons')
	]
	assert _selected_complete_triplets(
		rows,
		jobs,
		models=('mae', 'm1_current_k6', 'mh_nocons'),
	) == []


def test_split_labels_rejects_drift_in_every_committed_source_identity(
	tmp_path: Path,
) -> None:
	paths = {
		'inventory': tmp_path / 'inventory.csv',
		'label_volume': tmp_path / 'labels.npy',
		'valid_tokens': tmp_path / 'valid_tokens.npy',
		'class_info': tmp_path / 'class_info.json',
		'source_label_segy': tmp_path / 'labels.sgy',
		'seismic_volume': tmp_path / 'seismic.npy',
	}
	for path in paths.values():
		path.write_bytes(path.name.encode())
	geometry = tmp_path / 'segy_geometry.json'
	geometry.write_text(
		json.dumps(
			{
				'segy_files': {
					'label': {
						'cube_shape': [2, 2, 2],
						'iline_min': 100,
						'iline_max': 101,
						'xline_min': 200,
						'xline_max': 201,
					}
				}
			}
		),
		encoding='utf-8',
	)
	paths['segy_geometry_json'] = geometry
	metadata = {
		'artifact_type': 'f3_lithology_voxel_supervision',
		'inventory': _identity(paths['inventory']),
		'label_volume': _identity(paths['label_volume']),
		'reference_valid_tokens': _identity(paths['valid_tokens']),
		'labels': {
			'class_info': str(paths['class_info']),
			'source_label_segy': str(paths['source_label_segy']),
		},
		'source_identities': {
			key: _identity(paths[key])
			for key in (
				'class_info',
				'source_label_segy',
				'segy_geometry_json',
				'seismic_volume',
			)
		},
		'reference_embedding': {
			'metadata': {'source_amplitude_path': str(paths['seismic_volume'])}
		},
		'geometry': read_f3_line_geometry(geometry).to_dict(),
	}
	row = {'canonical_valid_tokens_sha256': file_sha256(paths['valid_tokens'])}
	assert _split_labels(row, metadata)['seismic_volume'] == paths['seismic_volume']
	for key in (
		'class_info',
		'source_label_segy',
		'segy_geometry_json',
		'seismic_volume',
	):
		paths[key].write_bytes(b'drift')
		with pytest.raises(ValueError, match='path/hash mismatch'):
			_split_labels(row, metadata)


def test_completed_metric_row_uses_canonical_metric_loader(tmp_path: Path) -> None:
	metrics_path = tmp_path / 'metrics.json'
	boundary_path = tmp_path / 'boundary.json'
	regions_path = tmp_path / 'regions.csv'
	metrics_path.write_text(
		json.dumps({
			'macro_f1': 0.7, 'mean_iou': 0.6, 'balanced_accuracy': 0.5,
			'accuracy': 0.4, 'weighted_f1': 0.3,
			'per_class_f1': {'0': 0.1, '1': 0.2, '3': 0.73, '5': 0.75},
			'per_class_iou': {'0': 0.1, '1': 0.2, '3': 0.63, '5': 0.65},
		}), encoding='utf-8'
	)
	boundary_path.write_text(
		json.dumps({
			'vertical_boundary_f1_at_2': 0.2,
			'vertical_boundary_f1_at_4': 0.4,
			'vertical_boundary_position_mae_at_2': 9.0,
			'vertical_boundary_position_mae_at_4': 4.0,
			'vertical_boundary_class_3_recall_at_2': 0.3,
			'vertical_boundary_class_3_recall_at_4': 0.4,
			'vertical_boundary_class_5_recall_at_2': 0.5,
			'vertical_boundary_class_5_recall_at_4': 0.6,
		}), encoding='utf-8'
	)
	regions_path.write_text(
		'region,radius,macro_f1,mean_iou\n'
		'boundary,2,0.22,0.23\n'
		'boundary,4,0.42,0.43\n', encoding='utf-8'
	)
	row = _metric_row({
		'split_id': 'split_000', 'budget_id': 'cap25', 'model_role': 'mae',
		'evaluation_metrics': {'path': str(metrics_path)},
		'evaluation_boundary_metrics': {'path': str(boundary_path)},
		'evaluation_boundary_region_metrics': {'path': str(regions_path)},
	})
	assert row['class_3_f1'] == pytest.approx(0.73)
	assert row['class_5_iou'] == pytest.approx(0.65)
	assert row['vertical_boundary_position_mae'] == pytest.approx(4.0)


def test_legacy_full_label_metadata_normalizes_strict_source_identities(
	tmp_path: Path,
) -> None:
	paths = {
		'class_info': tmp_path / 'class_info.json',
		'source_label_segy': tmp_path / 'labels.sgy',
		'seismic_volume': tmp_path / 'seismic.npy',
	}
	for path in paths.values():
		path.write_bytes(path.name.encode())
	geometry = tmp_path / 'segy_geometry.json'
	geometry.write_text(
		json.dumps({'segy_files': {'label': {
			'cube_shape': [2, 2, 2], 'iline_min': 100, 'iline_max': 101,
			'xline_min': 200, 'xline_max': 201,
		}}}), encoding='utf-8'
	)
	config = SimpleNamespace(
		class_info=paths['class_info'], source_label_segy=paths['source_label_segy'],
		segy_geometry_json=geometry, seismic_volume=paths['seismic_volume'],
	)
	metadata = {
		'labels': {
			'class_info': str(paths['class_info']),
			'source_label_segy': str(paths['source_label_segy']),
		},
		'reference_embedding': {
			'metadata': {'source_amplitude_path': str(paths['seismic_volume'])}
		},
		'geometry': read_f3_line_geometry(geometry).to_dict(),
	}
	normalized = _normalized_source_identities(config, metadata, split_id='split_001')
	assert set(normalized) == {
		'class_info', 'source_label_segy', 'segy_geometry_json', 'seismic_volume',
	}
	assert normalized['class_info'] == _identity(paths['class_info'])
	metadata['labels']['class_info'] = str(tmp_path / 'other.json')
	with pytest.raises(ValueError, match='class_info path'):
		_normalized_source_identities(config, metadata, split_id='split_001')
	metadata['labels']['class_info'] = str(paths['class_info'])
	metadata['source_identities'] = normalized
	paths['class_info'].write_bytes(b'drift')
	with pytest.raises(ValueError, match='source class_info hash'):
		_normalized_source_identities(config, metadata, split_id='split_001')


def test_only_missing_mixed_states_reuses_resumes_and_quarantines(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, jobs, calls, quarantined = _split_runner_fixture(
		tmp_path,
		monkeypatch,
		states={
			'mae': 'REUSE_COMPLETED',
			'm1_current_k6': 'RESUME_LATEST',
			'mh_nocons': 'INVALID_OR_PARTIAL',
		},
	)

	rows = split_runner.run_f3_lithology_voxel_label_budget_split_suite(
		config, only_missing=True
	)

	assert [row['action'] for row in rows] == ['REUSED', 'RESUMED', 'NEW']
	assert calls == [jobs[1].output_root / 'decoder/latest.pt', None]
	assert quarantined == [jobs[2].output_root]
	manifest = json.loads(
		(config.output_root / 'low_label_split_run_manifest.json').read_text(
			encoding='utf-8'
		)
	)
	assert [row['status'] for row in manifest['rows']] == ['complete'] * 3


def test_only_missing_persists_failure_then_recovers_only_incomplete_job(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	states = {
		'mae': 'REUSE_COMPLETED',
		'm1_current_k6': 'NEW',
		'mh_nocons': 'NEW',
	}
	config, jobs, calls, _quarantined = _split_runner_fixture(
		tmp_path, monkeypatch, states=states, fail_role='m1_current_k6'
	)

	with pytest.raises(RuntimeError, match='decoder failure'):
		split_runner.run_f3_lithology_voxel_label_budget_split_suite(
			config, only_missing=True
		)
	manifest_path = config.output_root / 'low_label_split_run_manifest.json'
	failed = json.loads(manifest_path.read_text(encoding='utf-8'))['rows']
	assert [(row['model_role'], row['status']) for row in failed] == [
		('m1_current_k6', 'failed'),
		('mae', 'complete'),
	]
	calls.clear()

	states.update({
		'mae': 'REUSE_COMPLETED',
		'm1_current_k6': 'RESUME_LATEST',
		'mh_nocons': 'NEW',
	})
	rows = split_runner.run_f3_lithology_voxel_label_budget_split_suite(
		config, only_missing=True
	)

	assert [row['action'] for row in rows] == ['RESUMED', 'REUSED', 'NEW']
	assert calls == [jobs[1].output_root / 'decoder/latest.pt', None]
	assert all(row['status'] == 'complete' for row in rows)


def test_filtered_only_missing_invocation_preserves_prior_split_rows(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, _jobs, _calls, _quarantined = _split_runner_fixture(
		tmp_path,
		monkeypatch,
		states=dict.fromkeys(('mae', 'm1_current_k6', 'mh_nocons'), 'NEW'),
		split_ids=('split_000', 'split_001'),
	)

	split_runner.run_f3_lithology_voxel_label_budget_split_suite(
		config, only_missing=True, split_id='split_000'
	)
	rows = split_runner.run_f3_lithology_voxel_label_budget_split_suite(
		config, only_missing=True, split_id='split_001'
	)

	assert len(rows) == 6
	assert {row['split_id'] for row in rows} == {'split_000', 'split_001'}


def _split_runner_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	*,
	states: dict[str, str],
	fail_role: str | None = None,
	split_ids: tuple[str, ...] = ('split_000',),
) -> tuple[
	SimpleNamespace,
	tuple[LowLabelSplitJob, ...],
	list[Path | None],
	list[Path],
]:
	roles = ('mae', 'm1_current_k6', 'mh_nocons')
	config = SimpleNamespace(
		output_root=tmp_path / 'outputs', models=roles, decoder_seed=42000
	)
	jobs = tuple(
		LowLabelSplitJob(
			split_id, 'cap25', role, config.output_root / 'runs' / split_id / role
		)
		for split_id in split_ids
		for role in roles
	)
	dataset_rows = {
		(split_id, 'cap25'): {
			'per_class_cap': 25,
			'voxel_dataset_root': str(tmp_path / split_id / 'dataset'),
			'selected_token_identity_sha256': 'tokens',
			'unique_token_xyz_sha256': 'xyz',
			'validation_mask_sha256': 'validation',
		}
		for split_id in split_ids
	}
	calls: list[Path | None] = []
	quarantined: list[Path] = []
	monkeypatch.setattr(
		split_runner,
		'inspect_f3_lithology_voxel_label_budget_split_suite',
		lambda _config: jobs,
	)
	monkeypatch.setattr(split_runner, '_dataset_rows', lambda _config: dataset_rows)
	monkeypatch.setattr(
		split_runner,
		'_SplitStageConfig',
		lambda _config, _row, role: SimpleNamespace(
			model_by_role={role: SimpleNamespace(model_tag=f'{role}-tag')}
		),
	)
	monkeypatch.setattr(
		split_runner,
		'classify_voxel_label_budget_job',
		lambda _stage, job: SimpleNamespace(
			state=states[job.model_role], reason='invalid'
		),
	)
	monkeypatch.setattr(
		split_runner,
		'quarantine_voxel_label_budget_output',
		lambda path, **_kwargs: quarantined.append(path)
		or path.with_name('quarantine'),
	)

	failure_seen = False

	def run_job(_stage: object, job: object, **kwargs: object) -> None:
		nonlocal failure_seen
		calls.append(kwargs['resume'])
		if job.model_role == fail_role and not failure_seen:
			failure_seen = True
			raise RuntimeError('decoder failure')

	monkeypatch.setattr(split_runner, 'run_voxel_label_budget_job', run_job)
	monkeypatch.setattr(
		split_runner,
		'completed_voxel_label_budget_job_row',
		lambda _stage, job, **kwargs: {
			**_complete_row(job),
			'action': kwargs['action'],
			'error': None,
			'quarantine_path': kwargs['quarantine_path'],
		},
	)
	return config, jobs, calls, quarantined


def _complete_row(job: object) -> dict[str, object]:
	row = dict(next(
		row for row in _triplet()
		if row['model_role'] == job.model_role
	))
	row.pop('split_id')
	return {
		**row,
		'budget_id': job.budget_id,
		'status': 'complete',
	}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}
