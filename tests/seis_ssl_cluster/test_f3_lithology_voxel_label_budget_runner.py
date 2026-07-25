"""Focused contracts for the low-label voxel suite runner."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import seis_ssl_cluster.f3.lithology.voxel_label_budget_runner as runner_module
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	_balanced_class_weights_from_train_manifest,
	_classify_job,
	_dataset_rows,
	_generated_evaluation_mapping,
	_generated_inference_mapping,
	_generated_report_mapping,
	_identity,
	_prior_run_state,
	_quarantine,
	_run_manifest_contract,
	_smoke_jobs_from_dataset_manifest,
	_smoke_manifest_path,
	_validate_completed_best_checkpoint,
	_validate_completed_history,
	_validate_generated_configs,
	_validate_smoke_gate,
	_validate_triplet,
	_validated_report_figure_paths,
	_validated_smoke_row,
	run_f3_lithology_voxel_label_budget_smoke,
	sampling_sequence_sha256,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_suite import (
		F3VoxelLabelBudgetSuiteConfig,
	)
	from seis_ssl_cluster.f3.lithology.voxel_tiles import VoxelTileManifest


def test_replacement_sampling_sequence_is_seeded_and_stable() -> None:
	"""Equal seeds reproduce the full sequence and unequal seeds change it."""
	arguments = {
		'tile_count': 440,
		'batch_size': 1,
		'steps_per_epoch': 440,
		'epochs': 50,
	}
	first = sampling_sequence_sha256(train_seed=42000, **arguments)
	second = sampling_sequence_sha256(train_seed=42000, **arguments)
	different = sampling_sequence_sha256(train_seed=42001, **arguments)
	assert first == second
	assert first != different
	assert len(first) == 64


def test_identity_reuses_hash_for_an_unchanged_file(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Repeated live-artifact checks avoid rehashing an unchanged file."""
	path = tmp_path / 'artifact.bin'
	path.write_bytes(b'identity-cache')
	actual = runner_module.file_sha256
	calls = 0

	def counted_file_sha256(candidate: Path) -> str:
		nonlocal calls
		calls += 1
		return actual(candidate)

	runner_module._IDENTITY_CACHE.clear()  # noqa: SLF001
	monkeypatch.setattr(runner_module, 'file_sha256', counted_file_sha256)

	first = _identity(path)
	second = _identity(path)

	assert first == second
	assert first is not second
	assert calls == 1


def test_completed_class_weights_are_recomputed_from_shared_train_mask() -> None:
	manifest = SimpleNamespace(
		split='train',
		class_ids=(0, 1),
		tiles=(
			SimpleNamespace(per_class_supervised_counts={'0': 1, '1': 3}),
			SimpleNamespace(per_class_supervised_counts={'0': 1, '1': 1}),
		),
	)

	weights = _balanced_class_weights_from_train_manifest(
		cast('VoxelTileManifest', manifest), expected_supervised_voxel_count=6
	)

	assert weights == pytest.approx([1.5, 0.75])
	with pytest.raises(ValueError, match='shared voxel mask'):
		_balanced_class_weights_from_train_manifest(
			cast('VoxelTileManifest', manifest), expected_supervised_voxel_count=7
		)


def test_completed_report_requires_exact_live_png_inventory(tmp_path: Path) -> None:
	relative_figures = [
		'figures/confusion_matrix.png',
		'figures/per_class_f1_iou.png',
		'figures/boundary_f1_by_tolerance.png',
		'figures/boundary_region_metrics.png',
	]
	(tmp_path / 'figures').mkdir()
	(tmp_path / 'report.md').write_text('report\n', encoding='utf-8')
	(tmp_path / 'report.json').write_text('{}\n', encoding='utf-8')
	for relative in relative_figures:
		(tmp_path / relative).write_bytes(b'\x89PNG\r\n\x1a\ncontent')

	paths = _validated_report_figure_paths(
		tmp_path, {'figures': relative_figures}
	)

	assert [path.relative_to(tmp_path).as_posix() for path in paths] == (
		relative_figures
	)
	paths[0].unlink()
	with pytest.raises(FileNotFoundError, match='incomplete completed report figure'):
		_validated_report_figure_paths(
			tmp_path, {'figures': relative_figures}
		)


def test_job_paths_follow_preregistered_layout(tmp_path: Path) -> None:
	"""A job keeps every stage beneath one budget/seed/model directory."""
	job = _job(tmp_path)
	assert job.decoder_dir == job.output_root / 'decoder'
	assert job.prediction_dir == job.output_root / 'prediction'
	assert job.evaluation_dir == job.output_root / 'evaluation'
	assert job.report_dir == job.output_root / 'report'
	assert job.generated_configs_dir == job.output_root / 'generated_configs'


def test_nonexistent_job_is_new(tmp_path: Path) -> None:
	"""The state machine labels a path with no prior output as NEW."""
	job = _job(tmp_path)
	config = cast('F3VoxelLabelBudgetSuiteConfig', object())
	plan = _classify_job(config, job, estimated_bytes=123)
	assert plan.state == 'NEW'
	assert plan.estimated_bytes == 123


def test_partial_job_is_invalid_without_deletion(tmp_path: Path) -> None:
	"""A pre-existing incomplete tree is classified, not silently removed."""
	job = _job(tmp_path)
	job.output_root.mkdir(parents=True)
	marker = job.output_root / 'keep.txt'
	marker.write_text('owned artifact\n', encoding='utf-8')
	config = cast('F3VoxelLabelBudgetSuiteConfig', object())
	plan = _classify_job(config, job, estimated_bytes=123)
	assert plan.state == 'INVALID_OR_PARTIAL'
	assert marker.is_file()


@pytest.mark.parametrize(
	('resume_error', 'expected_state'),
	[(None, 'RESUME_LATEST'), (ValueError('bad source hash'), 'INVALID_OR_PARTIAL')],
)
def test_resume_state_requires_full_identity_validation(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	resume_error: Exception | None,
	expected_state: str,
) -> None:
	"""An incomplete checkpoint is resumable only after strict source validation."""
	job = _job(tmp_path)
	job.decoder_dir.mkdir(parents=True)
	(job.decoder_dir / 'latest.pt').write_bytes(b'checkpoint')
	resolved = {'identity': 'expected'}
	decoder_config = SimpleNamespace(to_dict=lambda: resolved)
	monkeypatch.setattr(runner_module, '_decoder_config', lambda *_: decoder_config)
	monkeypatch.setattr(
		runner_module,
		'load_voxel_decoder_checkpoint',
		lambda *_args, **_kwargs: {
			'resolved_config': resolved,
			'checkpoint_kind': 'epoch',
		},
	)

	def validate(*_args: object, **_kwargs: object) -> dict[str, object]:
		if resume_error is not None:
			raise resume_error
		return {'checkpoint_kind': 'epoch'}

	monkeypatch.setattr(
		runner_module, 'validate_f3_lithology_voxel_decoder_resume', validate
	)
	config = cast('F3VoxelLabelBudgetSuiteConfig', object())
	plan = _classify_job(config, job, estimated_bytes=123)
	assert plan.state == expected_state
	if resume_error is not None:
		assert 'bad source hash' in (plan.reason or '')


def test_full_run_rejects_missing_smoke_gate(tmp_path: Path) -> None:
	"""Scientific execution cannot start before the canonical triplet smoke."""
	config = cast(
		'F3VoxelLabelBudgetSuiteConfig', SimpleNamespace(output_root=tmp_path)
	)
	with pytest.raises(FileNotFoundError, match=r'smoke_manifest\.json'):
		_validate_smoke_gate(config)


def test_smoke_gate_validates_live_checkpoint_and_snapshot_hashes(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A manifest row is accepted only when recomputed from live smoke outputs."""
	config, _jobs = _smoke_gate_fixture(tmp_path, monkeypatch)
	_validate_smoke_gate(config)


def test_smoke_gate_rejects_deleted_live_checkpoint(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Deleting a checkpoint invalidates an otherwise unchanged smoke manifest."""
	config, jobs = _smoke_gate_fixture(tmp_path, monkeypatch)
	(jobs[0].decoder_dir / 'latest.pt').unlink()
	with pytest.raises(FileNotFoundError, match=r'latest\.pt'):
		_validate_smoke_gate(config)


def test_smoke_gate_rejects_corrupt_live_snapshot(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Changing a snapshot after smoke completion invalidates its recorded hash."""
	config, jobs = _smoke_gate_fixture(tmp_path, monkeypatch)
	(jobs[0].decoder_dir / 'train_tile_manifest.json').write_text(
		'corrupt\n', encoding='utf-8'
	)
	with pytest.raises(ValueError, match='live artifacts'):
		_validate_smoke_gate(config)


def test_failed_smoke_rerun_cannot_leave_prior_success_manifest(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""The old gate is quarantined before model movement or failed execution."""
	config = cast(
		'F3VoxelLabelBudgetSuiteConfig', SimpleNamespace(output_root=tmp_path)
	)
	jobs = tuple(
		VoxelLabelBudgetJob(
			budget_id='cap25',
			per_class_cap=25,
			subsample_seed=0,
			decoder_seed=42000,
			model_role=role,
			model_tag=f'{role}-tag',
			voxel_dataset_root=tmp_path / 'dataset',
			output_root=tmp_path / 'unused',
			dataset_row={'class_order': list(range(6))},
		)
		for role in ('mae', 'm1', 'm2a')
	)
	manifest = (
		tmp_path
		/ 'smoke'
		/ 'budget=cap25'
		/ 'subsample_seed=0'
		/ 'smoke_manifest.json'
	)
	manifest.parent.mkdir(parents=True)
	manifest.write_text('prior success\n', encoding='utf-8')
	first_model = manifest.parent / 'model=mae-tag'
	first_model.mkdir()
	(first_model / 'marker.txt').write_text('prior model\n', encoding='utf-8')
	monkeypatch.setattr(
		runner_module,
		'inspect_f3_lithology_voxel_label_budget_suite',
		lambda *_args, **_kwargs: SimpleNamespace(jobs=jobs),
	)
	monkeypatch.setattr(
		runner_module,
		'_decoder_config',
		lambda *_args, **_kwargs: SimpleNamespace(),
	)

	def fail_run(*_args: object, **_kwargs: object) -> None:
		raise RuntimeError('smoke failed')

	monkeypatch.setattr(
		runner_module, 'run_f3_lithology_voxel_decoder', fail_run
	)
	real_quarantine = _quarantine
	quarantine_calls: list[Path] = []

	def tracked_quarantine(path: Path, *, reason: str) -> Path:
		quarantine_calls.append(path)
		return real_quarantine(path, reason=reason)

	monkeypatch.setattr(runner_module, '_quarantine', tracked_quarantine)
	with pytest.raises(RuntimeError, match='smoke failed'):
		run_f3_lithology_voxel_label_budget_smoke(config)
	assert quarantine_calls[:2] == [manifest, first_model]
	assert not manifest.exists()
	quarantined = tuple(manifest.parent.glob('smoke_manifest.json.quarantine_*'))
	assert len(quarantined) == 1
	assert quarantined[0].read_text(encoding='utf-8') == 'prior success\n'


def test_invalid_job_quarantine_preserves_contents(tmp_path: Path) -> None:
	"""Quarantine moves the whole tree to a timestamped sibling."""
	job = _job(tmp_path)
	job.output_root.mkdir(parents=True)
	marker = job.output_root / 'keep.txt'
	marker.write_text('owned artifact\n', encoding='utf-8')
	quarantine = _quarantine(job.output_root, reason='wrong seed')
	assert not job.output_root.exists()
	assert quarantine.parent == job.output_root.parent
	assert '.quarantine_' in quarantine.name
	assert (quarantine / marker.name).read_text(encoding='utf-8') == 'owned artifact\n'


@pytest.mark.parametrize(
	'key',
	[
		'initial_model_state_sha256',
		'train_tile_manifest_sha256',
		'validation_tile_manifest_sha256',
		'train_tile_identity_sha256',
		'validation_tile_identity_sha256',
		'class_weights',
		'class_order',
		'decoder_architecture',
		'sampling_sequence_sha256',
		'decoder_seed',
	],
)
def test_triplet_rejects_paired_identity_mismatch(key: str) -> None:
	"""Every paired initialization, sampler, tile, and weight identity is strict."""
	rows = _triplet_rows()
	rows[2][key] = 'mismatch'
	with pytest.raises(ValueError, match=key):
		_validate_triplet(rows, context='cap25/seed0')


def test_triplet_accepts_three_model_shared_contract() -> None:
	"""Only the model role and tag may differ inside a paired triplet."""
	_validate_triplet(_triplet_rows(), context='cap25/seed0')


def test_prior_manifest_rejects_duplicate_job_rows(tmp_path: Path) -> None:
	"""Prior rows cannot silently overwrite one another during filtered resume."""
	dataset_manifest = tmp_path / 'datasets.json'
	dataset_manifest.write_text('{}\n', encoding='utf-8')
	row = {
		'budget_id': 'cap25',
		'subsample_seed': 0,
		'model_role': 'mae',
		'model_tag': 'mae-tag',
		'status': 'failed',
	}
	config = cast(
		'F3VoxelLabelBudgetSuiteConfig',
		SimpleNamespace(
			budgets=('cap25',),
			subsample_seeds=(0,),
			train=SimpleNamespace(
				epochs=50,
				sampling_mode='uniform_tiles_with_replacement',
				steps_per_epoch=440,
			),
			base_seed=42000,
			add_subsample_seed=True,
			dataset_manifest=dataset_manifest,
			model_by_role={
				'mae': SimpleNamespace(model_tag='mae-tag'),
				'm1': SimpleNamespace(model_tag='m1-tag'),
				'm2a': SimpleNamespace(model_tag='m2a-tag'),
			},
		),
	)
	payload = {
		'artifact_type': runner_module.RUN_MANIFEST_TYPE,
		'schema_version': runner_module.RUN_SCHEMA_VERSION,
		'preregistered_contract': _run_manifest_contract(config),
		'dataset_manifest': _identity(dataset_manifest),
		'row_count': 2,
		'complete_count': 0,
		'rows': [row, dict(row)],
		'quarantines': [],
		'disk_audits': [],
	}
	path = tmp_path / runner_module.RUN_MANIFEST_NAME
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='duplicate'):
		_prior_run_state(path, config=config)


def test_dataset_preflight_calls_shared_six_file_validator_for_each_condition(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Runner preflight crosses the shared committed-condition boundary."""
	config, metadata_by_root = _dataset_preflight_fixture(tmp_path)
	calls: list[Path] = []

	def validate(root: Path) -> Mapping[str, object]:
		calls.append(root)
		return metadata_by_root[root]

	monkeypatch.setattr(
		runner_module, 'validate_voxel_label_budget_condition_artifact', validate
	)
	rows = _dataset_rows(config)
	assert set(rows) == {('cap25', 0), ('cap25', 1)}
	assert calls == list(metadata_by_root)


def test_dataset_preflight_propagates_six_file_corruption(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""A shared-validator failure prevents any condition from reaching training."""
	config, _metadata = _dataset_preflight_fixture(tmp_path, seeds=(0,))

	def reject(_root: Path) -> Mapping[str, object]:
		raise ValueError('six-file corruption')

	monkeypatch.setattr(
		runner_module, 'validate_voxel_label_budget_condition_artifact', reject
	)
	with pytest.raises(ValueError, match='six-file corruption'):
		_dataset_rows(config)


@pytest.mark.parametrize(
	('field', 'expected_error'),
	[
		('schema_version', 'schema'),
		('suite', 'suite contract'),
		('contract', 'scientific contract'),
		('models', 'model contract'),
		('sources', 'hash mismatch'),
		('common_validation_mask_sha256', 'common validation'),
	],
)
def test_dataset_preflight_rejects_top_level_contract_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	field: str,
	expected_error: str,
) -> None:
	"""Schema, model, and shared-validation identities are config-bound."""
	config, metadata_by_root = _dataset_preflight_fixture(tmp_path, seeds=(0,))
	manifest = json.loads(config.dataset_manifest.read_text(encoding='utf-8'))
	if field == 'schema_version':
		manifest[field] = 999
	elif field == 'suite':
		manifest[field]['output_root'] = str(tmp_path / 'wrong-suite')
	elif field == 'contract':
		manifest[field]['budgets'] = ['cap50']
	elif field == 'models':
		manifest[field]['mae'] = 'wrong-model'
	elif field == 'sources':
		manifest[field]['common_grid']['sha256'] = '0' * 64
	else:
		manifest[field] = 'b' * 64
	config.dataset_manifest.write_text(json.dumps(manifest), encoding='utf-8')
	monkeypatch.setattr(
		runner_module,
		'validate_voxel_label_budget_condition_artifact',
		lambda root: metadata_by_root[root],
	)
	with pytest.raises(ValueError, match=expected_error):
		_dataset_rows(config)


def test_completed_history_binds_checkpoint_and_csv_sequences() -> None:
	"""Completed reuse accepts only the exact internal/CSV epoch progression."""
	history = [
		{'epoch': 0, 'global_step': 3, 'validation_macro_f1': 0.2},
		{'epoch': 1, 'global_step': 6, 'validation_macro_f1': 0.3},
	]
	payload = {'training_history': history, 'history': list(history)}
	csv_rows = [{key: str(value) for key, value in row.items()} for row in history]
	_validate_completed_history(
		payload,
		expected_rows=2,
		steps_per_epoch=3,
		csv_rows=csv_rows,
		label='latest.pt',
	)


@pytest.mark.parametrize('location', ['checkpoint_epoch', 'checkpoint_step', 'csv'])
def test_completed_history_rejects_sequence_drift(location: str) -> None:
	"""Epoch/global-step drift cannot be hidden in either persisted history."""
	history = [
		{'epoch': 0, 'global_step': 3},
		{'epoch': 1, 'global_step': 6},
	]
	csv_rows = [{key: str(value) for key, value in row.items()} for row in history]
	if location == 'checkpoint_epoch':
		history[1]['epoch'] = 0
	elif location == 'checkpoint_step':
		history[1]['global_step'] = 5
	else:
		csv_rows[1]['global_step'] = '5'
	payload = {'training_history': history}
	with pytest.raises(ValueError, match=r'sequence|content'):
		_validate_completed_history(
			payload,
			expected_rows=2,
			steps_per_epoch=3,
			csv_rows=csv_rows,
			label='latest.pt',
		)


def test_completed_best_checkpoint_requires_latest_selection_equality() -> None:
	"""best.pt cannot disagree with the selection state persisted by latest.pt."""
	selection = {
		'epoch': 0,
		'validation_metrics': {'macro_f1': 0.5},
		'rule': ['macro_f1'],
		'epsilon': 1e-12,
	}
	latest = {'best_selection_state': selection}
	best = {
		'best_selection_state': {**selection, 'epsilon': 0.0},
		'epoch': 0,
		'global_step': 3,
		'checkpoint_kind': 'epoch',
		'resolved_config': {'identity': 'expected'},
		'training_history': [{'epoch': 0, 'global_step': 3}],
		'current_metrics': {'validation': {'macro_f1': 0.5}},
	}
	with pytest.raises(ValueError, match='selection state mismatch'):
		_validate_completed_best_checkpoint(
			latest,
			best,
			resolved_config={'identity': 'expected'},
			epochs=2,
			steps_per_epoch=3,
		)


@pytest.mark.parametrize(
	'config_name',
	[
		'decoder_config.json',
		'inference_config.json',
		'evaluation_config.json',
		'report_config.json',
	],
)
def test_completed_reuse_validates_generated_config_contents(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	config_name: str,
) -> None:
	"""Fresh file hashes cannot bless generated configs with changed contents."""
	job = _job(tmp_path)
	job.generated_configs_dir.mkdir(parents=True)
	best_path = job.decoder_dir / 'best.pt'
	best_path.parent.mkdir(parents=True)
	best_path.write_bytes(b'best checkpoint')
	config = cast(
		'F3VoxelLabelBudgetSuiteConfig',
		SimpleNamespace(
			model_by_role={
				'mae': SimpleNamespace(embeddings_dir=tmp_path / 'embeddings')
			},
			evaluation={'metric': 'fixed'},
			report={'dpi': 150},
		),
	)
	decoder_mapping = {'decoder': 'fixed'}
	monkeypatch.setattr(
		runner_module,
		'_decoder_config',
		lambda *_args: SimpleNamespace(to_dict=lambda: decoder_mapping),
	)
	expected = {
		'decoder_config.json': decoder_mapping,
		'inference_config.json': _generated_inference_mapping(
			config, job, checkpoint=best_path
		),
		'evaluation_config.json': _generated_evaluation_mapping(config, job),
		'report_config.json': _generated_report_mapping(config, job),
	}
	for name, mapping in expected.items():
		(job.generated_configs_dir / name).write_text(
			json.dumps(mapping), encoding='utf-8'
		)
	_validate_generated_configs(config, job, best_path=best_path)
	tampered = {**expected[config_name], 'tampered': True}
	(job.generated_configs_dir / config_name).write_text(
		json.dumps(tampered), encoding='utf-8'
	)
	with pytest.raises(ValueError, match=config_name):
		_validate_generated_configs(config, job, best_path=best_path)


def _smoke_gate_fixture(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[F3VoxelLabelBudgetSuiteConfig, tuple[VoxelLabelBudgetJob, ...]]:
	dataset_manifest = tmp_path / runner_module.DATASET_MANIFEST_NAME
	dataset_root = (
		tmp_path
		/ 'datasets'
		/ 'budget=cap25'
		/ 'subsample_seed=0'
		/ 'voxel_supervision'
	)
	dataset_manifest.write_text(
		json.dumps(
			{
				'rows': [
					{
						'budget_id': 'cap25',
						'per_class_cap': 25,
						'subsample_seed': 0,
						'voxel_dataset_root': str(dataset_root),
						'class_order': list(range(6)),
					}
				]
			}
		),
		encoding='utf-8',
	)
	models = {
		role: SimpleNamespace(model_tag=f'{role}-tag')
		for role in ('mae', 'm1', 'm2a')
	}
	config = cast(
		'F3VoxelLabelBudgetSuiteConfig',
		SimpleNamespace(
			output_root=tmp_path,
			dataset_manifest=dataset_manifest,
			model_by_role=models,
			base_seed=42000,
			add_subsample_seed=True,
			train=SimpleNamespace(
				sampling_mode='uniform_tiles_with_replacement',
				steps_per_epoch=3,
				batch_size=1,
				epochs=2,
			),
		),
	)
	monkeypatch.setattr(
		runner_module,
		'_decoder_config',
		lambda *_args: SimpleNamespace(to_dict=lambda: {'decoder': 'smoke'}),
	)
	jobs = _smoke_jobs_from_dataset_manifest(
		config, budget='cap25', subsample_seed=0
	)
	for job in jobs:
		job.decoder_dir.mkdir(parents=True)
		(job.decoder_dir / 'latest.pt').write_bytes(b'valid checkpoint')
		(job.decoder_dir / 'train_tile_manifest.json').write_text(
			'train tiles\n', encoding='utf-8'
		)
		(job.decoder_dir / 'validation_tile_manifest.json').write_text(
			'validation tiles\n', encoding='utf-8'
		)
		(job.decoder_dir / 'run_metadata.json').write_text(
			json.dumps(
				{
					'initial_model_state_sha256': 'initial-state',
					'sampling_mode': config.train.sampling_mode,
					'steps_per_epoch': config.train.steps_per_epoch,
					'train_seed': 42000,
					'train_tile_manifest_sha256': 'train-identity',
					'validation_tile_manifest_sha256': 'validation-identity',
				}
			),
			encoding='utf-8',
		)

	def read_manifest(path: Path) -> SimpleNamespace:
		identity = (
			'train-identity'
			if path.name.startswith('train_')
			else 'validation-identity'
		)
		return SimpleNamespace(identity_sha256=identity, tiles=(0, 1))

	monkeypatch.setattr(runner_module, 'read_voxel_tile_manifest', read_manifest)
	monkeypatch.setattr(
		runner_module,
		'validate_f3_lithology_voxel_decoder_resume',
		lambda *_args: {
			'checkpoint_kind': 'step',
			'global_step': 2,
			'epoch': 0,
			'batch_index': 1,
			'current_metrics': {
				'train_accumulator': {
					'supervised_voxel_count': 10,
					'weighted_ce_sum': 1.0,
					'unweighted_ce_sum': 1.0,
					'class_weight_sum': 1.0,
				}
			},
			'class_weights': [1.0] * 6,
			'decoder_architecture': {'spec': 'fixed'},
		},
	)
	rows = [
		_validated_smoke_row(
			config, job, checkpoint=job.decoder_dir / 'latest.pt'
		)
		for job in jobs
	]
	manifest_path = _smoke_manifest_path(
		config, budget='cap25', subsample_seed=0
	)
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(
		json.dumps(
			{
				'artifact_type': (
					'f3_lithology_voxel_label_budget_smoke_manifest'
				),
				'schema_version': 1,
				'scientific_result': False,
				'dataset_manifest': _identity(dataset_manifest),
				'contract': {
					'budget_id': 'cap25',
					'subsample_seed': 0,
					'global_step': 2,
					'sampling_mode': config.train.sampling_mode,
					'steps_per_epoch': config.train.steps_per_epoch,
				},
				'quarantines': [],
				'rows': rows,
			}
		),
		encoding='utf-8',
	)
	return config, jobs


def _dataset_preflight_fixture(
	tmp_path: Path, *, seeds: tuple[int, ...] = (0, 1)
) -> tuple[
	F3VoxelLabelBudgetSuiteConfig, dict[Path, Mapping[str, object]]
]:
	output_root = tmp_path / 'suite'
	output_root.mkdir()
	models = {
		role: SimpleNamespace(model_tag=f'{role}-tag')
		for role in ('mae', 'm1', 'm2a')
	}
	config = cast(
		'F3VoxelLabelBudgetSuiteConfig',
		SimpleNamespace(
			dataset_manifest=(
				output_root / runner_module.DATASET_MANIFEST_NAME
			),
			output_root=output_root,
			budgets=('cap25',),
			subsample_seeds=seeds,
			model_by_role=models,
			decoder=SimpleNamespace(
				upsample_factors=((2, 2, 2), (2, 2, 2), (2, 2, 2)),
				class_count=6,
			),
		),
	)
	sources: dict[str, Mapping[str, object]] = {}
	for key in runner_module.DATASET_SOURCE_KEYS:
		path = tmp_path / f'{key}.source'
		path.write_text(f'{key}\n', encoding='utf-8')
		sources[key] = _identity(path)
	common_validation_hash = 'a' * 64
	rows: list[Mapping[str, object]] = []
	metadata_by_root: dict[Path, Mapping[str, object]] = {}
	for seed in seeds:
		root = (
			output_root
			/ 'datasets'
			/ 'budget=cap25'
			/ f'subsample_seed={seed}'
			/ 'voxel_supervision'
		)
		root.mkdir(parents=True)
		identities: dict[str, Mapping[str, object]] = {}
		for identity_key, name in runner_module.DATASET_ROW_FILE_NAMES.items():
			path = root / name
			path.write_text(f'{identity_key} seed={seed}\n', encoding='utf-8')
			identities[identity_key] = _identity(path)
		per_class = {str(class_id): 1 for class_id in range(6)}
		row = {
			'budget_id': 'cap25',
			'per_class_cap': 25,
			'subsample_seed': seed,
			'voxel_dataset_root': str(root),
			'train_voxel_count': 6,
			'validation_voxel_count': 6,
			'class_order': list(range(6)),
			'per_class_train_voxel_counts': per_class,
			'per_class_validation_voxel_counts': per_class,
			'selected_token_row_count': 150,
			'unique_selected_token_xyz_count': 150,
			'duplicate_selected_row_count': 0,
			'selected_token_identity_sha256': f'selected-{seed}',
			'unique_token_xyz_sha256': f'unique-{seed}',
			'train_mask_sha256': f'train-{seed}',
			'validation_mask_sha256': common_validation_hash,
			**identities,
		}
		identity = {
			'budget_id': 'cap25',
			'per_class_cap': 25,
			'subsample_seed': seed,
			'patch_size_xyz': [8, 8, 8],
			'actual_train_voxel_count': 6,
			'validation_voxel_count': 6,
			'class_order': list(range(6)),
			'per_class_train_voxel_counts': per_class,
			'per_class_validation_voxel_counts': per_class,
			'selected_token_row_count': 150,
			'unique_selected_token_xyz_count': 150,
			'duplicate_selected_row_count': 0,
			'selected_token_identity_sha256': f'selected-{seed}',
			'unique_token_xyz_sha256': f'unique-{seed}',
			'train_mask_sha256': f'train-{seed}',
			'validation_mask_sha256': common_validation_hash,
		}
		metadata_by_root[root] = {
			'suite': {
				'name': runner_module.DATASET_SUITE_NAME,
				'output_root': str(output_root),
			},
			'identity': identity,
			'sources': {
				**sources,
				'selected_token_artifacts': {
					role: {}
					for role in (
						'mae_m1_mae',
						'mae_m1_m1',
						'm1_m2a_m1',
						'm1_m2a_m2a',
					)
				},
			},
		}
		rows.append(row)
	payload = {
		'artifact_type': runner_module.MANIFEST_ARTIFACT_TYPE,
		'schema_version': runner_module.DATASET_SCHEMA_VERSION,
		'suite': {
			'name': runner_module.DATASET_SUITE_NAME,
			'output_root': str(output_root),
			'budget_semantics': 'per_class_selected_token_row_cap',
		},
		'contract': {
			'budgets': ['cap25'],
			'subsample_seeds': list(seeds),
			'patch_size_xyz': [8, 8, 8],
			'require_all_classes': True,
			'validation': 'canonical_full_validation_bitwise',
		},
		'models': {role: models[role].model_tag for role in ('mae', 'm1', 'm2a')},
		'sources': sources,
		'common_validation_mask_sha256': common_validation_hash,
		'condition_count': len(rows),
		'rows': rows,
	}
	config.dataset_manifest.write_text(json.dumps(payload), encoding='utf-8')
	return config, metadata_by_root


def _job(tmp_path: Path) -> VoxelLabelBudgetJob:
	output = tmp_path / 'jobs' / 'budget=cap25' / 'subsample_seed=0' / 'model=mae'
	return VoxelLabelBudgetJob(
		budget_id='cap25',
		per_class_cap=25,
		subsample_seed=0,
		decoder_seed=42000,
		model_role='mae',
		model_tag='mae',
		voxel_dataset_root=tmp_path / 'dataset',
		output_root=output,
		dataset_row={},
	)


def _triplet_rows() -> list[dict[str, object]]:
	shared: Mapping[str, object] = {
		'initial_model_state_sha256': 'initial',
		'train_tile_manifest_sha256': 'train',
		'validation_tile_manifest_sha256': 'validation',
		'train_tile_identity_sha256': 'train-identity',
		'validation_tile_identity_sha256': 'validation-identity',
		'class_weights': [1.0] * 6,
		'class_order': [0, 1, 2, 3, 4, 5],
		'sampling_mode': 'uniform_tiles_with_replacement',
		'steps_per_epoch': 440,
		'sampling_sequence_sha256': 'sequence',
		'voxel_supervision_grid_sha256': 'grid',
		'selected_token_identity_sha256': 'tokens',
		'unique_token_xyz_sha256': 'xyz',
		'train_voxel_count': 10_024,
		'validation_voxel_count': 470_136,
		'validation_mask_sha256': 'validation-mask',
		'canonical_valid_token_sha256': 'valid',
		'decoder_architecture': {'spec': 'decoder'},
		'decoder_seed': 42000,
		'uncovered_validation_voxel_count': 0,
		'metric_schema_sha256': 'metrics',
	}
	return [
		{**shared, 'model_role': role, 'model_tag': role}
		for role in ('mae', 'm1', 'm2a')
	]
