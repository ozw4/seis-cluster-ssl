"""Focused pure contracts for the current-code K=6 voxel control."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import seis_ssl_cluster.f3.lithology.voxel_label_budget_control as control_module
from proc.seis_ssl_cluster import (
	summarize_f3_lithology_voxel_label_budget_control as summarize_control_cli,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_control import (
	CONTROL_PUBLISH_MANIFEST,
	CURRENT_MODEL_ROLE,
	FORBIDDEN_SUFFIXES,
	PAIR_IDENTITY_KEYS,
	_jobs,
	_publish_target_names,
	_readiness_decision,
	_validate_reference_contract,
	run_f3_lithology_voxel_label_budget_control,
	summarize_f3_lithology_voxel_label_budget_control,
	validate_f3_lithology_voxel_label_budget_control_summary_preflight,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_control import (
	_summary_rows as summarize_rows,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import METRIC_SPECS
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	VoxelLabelBudgetJobPlan,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_current_control_job_matrix_is_exactly_fifteen(tmp_path: Path) -> None:
	config = _config(tmp_path)
	rows = {
		(budget, seed): {
			'per_class_cap': int(budget.removeprefix('cap')),
			'voxel_dataset_root': str(tmp_path / budget / str(seed)),
		}
		for budget in config.budgets
		for seed in config.subsample_seeds
	}

	jobs = _jobs(config, rows)

	assert len(jobs) == 15
	assert {job.model_role for job in jobs} == {CURRENT_MODEL_ROLE}
	assert {job.decoder_seed for job in jobs if job.subsample_seed == 0} == {42000}
	assert {job.decoder_seed for job in jobs if job.subsample_seed == 4} == {42004}


def test_readiness_positive_requires_both_primary_metrics(tmp_path: Path) -> None:
	config = _config(tmp_path)
	rows = _summary_rows(config, current_vs_mae=0.02, current_vs_m1=0.0)

	decision = _readiness_decision(config, rows)

	assert decision['status'] == 'CONTROL_READY_POSITIVE'
	assert decision['current_k6_vs_mae']['positive_budgets'] == [  # type: ignore[index]
		'cap25',
		'cap50',
		'cap100',
	]


def test_readiness_with_drift_overrides_positive_label(tmp_path: Path) -> None:
	config = _config(tmp_path)
	rows = _summary_rows(config, current_vs_mae=0.02, current_vs_m1=0.01)

	decision = _readiness_decision(config, rows)

	assert decision['status'] == 'CONTROL_READY_WITH_DRIFT'


def test_publish_target_inventory_is_lightweight_only() -> None:
	targets = _publish_target_names()

	assert CONTROL_PUBLISH_MANIFEST in targets
	assert not {
		name
		for name in targets
		if any(name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
	}


def test_summary_dry_run_preflight_is_readonly_and_rejects_raw_publish(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	config.artifact_root = tmp_path / 'artifacts'
	config.reports_dir = tmp_path / 'reports'
	config.publish = SimpleNamespace(
		enabled=True,
		output_dir=tmp_path / 'published',
		max_file_size_bytes=10 * 1024 * 1024,
	)
	preflight = (
		config.artifact_root
		/ 'pretraining'
		/ 'f3'
		/ 'facies_benchmark_v1'
		/ config.candidate.model_tag
		/ 'preflight'
		/ 'control_input_manifest.json'
	)
	preflight.parent.mkdir(parents=True)
	preflight.write_text('{}\n', encoding='utf-8')
	config.reports_dir.mkdir()
	for name in ('checkpoint_validation.json', 'embedding_validation.json'):
		(config.reports_dir / name).write_text(
			json.dumps({'status': 'PASS'}), encoding='utf-8'
		)
	(config.reports_dir / 'token_probe_comparison.csv').write_text(
		'metric,current_k6\naccuracy,1.0\n', encoding='utf-8'
	)

	validate_f3_lithology_voxel_label_budget_control_summary_preflight(config)

	assert not (config.reports_dir / 'control_input_manifest.json').exists()
	assert not config.publish.output_dir.exists()
	config.publish.output_dir.mkdir()
	(config.publish.output_dir / 'raw_checkpoint.pt').write_bytes(b'raw')
	with pytest.raises(ValueError, match='raw artifact was published'):
		validate_f3_lithology_voxel_label_budget_control_summary_preflight(config)
	assert not (config.reports_dir / 'control_input_manifest.json').exists()


def test_summary_cli_dry_run_invokes_readonly_preflight(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	config = SimpleNamespace()
	inspection = SimpleNamespace(
		historical_m1_mae_parity={'status': 'PASS'},
		readiness={'status': 'CONTROL_READY_MIXED'},
	)
	calls: list[object] = []
	config_path = tmp_path / 'control.yaml'
	config_path.write_text('{}\n', encoding='utf-8')
	monkeypatch.setattr(
		summarize_control_cli, 'load_config_for_cli', lambda *_args, **_kwargs: {}
	)
	monkeypatch.setattr(
		summarize_control_cli,
		'resolve_config_for_cli',
		lambda *_args, **_kwargs: config,
	)
	monkeypatch.setattr(
		summarize_control_cli,
		'inspect_f3_lithology_voxel_label_budget_control_results',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		summarize_control_cli,
		'validate_f3_lithology_voxel_label_budget_control_summary_preflight',
		lambda received: calls.append(received),
	)
	monkeypatch.setattr(
		summarize_control_cli,
		'summarize_f3_lithology_voxel_label_budget_control',
		lambda *_args, **_kwargs: pytest.fail('dry-run must not summarize'),
	)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'summarize_f3_lithology_voxel_label_budget_control.py',
			'--config',
			str(config_path),
			'--dry-run',
		],
	)

	summarize_control_cli.main()

	assert calls == [config]
	assert 'execution: dry-run; summary and publish skipped' in capsys.readouterr().out


def test_reference_mae_m1_pair_contract_rejects_identity_drift(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _config(tmp_path)
	dataset_rows = {
		(budget, seed): {'train_mask_sha256': 'a' * 64}
		for budget in config.budgets
		for seed in config.subsample_seeds
	}
	jobs = [
		SimpleNamespace(
			dataset=SimpleNamespace(budget_id=budget, subsample_seed=seed),
			model_role=role,
		)
		for budget, seed in dataset_rows
		for role in ('mae', 'm1')
	]
	reference = SimpleNamespace(jobs=tuple(jobs))
	values = {
		**dict.fromkeys(PAIR_IDENTITY_KEYS, 'same'),
		'validation_voxel_count': 470136,
		'train_mask_sha256': 'a' * 64,
		'resolved_amp_dtype': 'float16',
		'amp_scaler': True,
	}
	monkeypatch.setattr(
		control_module,
		'_reference_pair_values',
		lambda *_args: dict(values),
	)

	_validate_reference_contract(config, reference, dataset_rows)

	def drifted_values(job: object, *_args: object) -> dict[str, object]:
		result = dict(values)
		if job.model_role == 'm1':
			result['sampling_sequence_sha256'] = 'different'
		return result

	monkeypatch.setattr(control_module, '_reference_pair_values', drifted_values)
	with pytest.raises(ValueError, match='sampling_sequence_sha256'):
		_validate_reference_contract(config, reference, dataset_rows)


def test_only_missing_reuses_completed_job_without_training(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, job, inspection = _runner_fixture(tmp_path, state='REUSE_COMPLETED')
	trained = False
	monkeypatch.setattr(
		control_module,
		'inspect_f3_lithology_voxel_label_budget_control',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		control_module, '_prior_control_state', lambda *_args, **_kwargs: ((), [])
	)
	monkeypatch.setattr(
		control_module,
		'_completed_control_row',
		lambda *_args, **kwargs: _complete_row(job, action=kwargs['action']),
	)
	monkeypatch.setattr(
		control_module, '_validate_paired_identity', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		control_module, '_write_control_manifest', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		control_module, '_write_status_csv', lambda *_args, **_kwargs: None
	)

	def train(*_args: object, **_kwargs: object) -> None:
		nonlocal trained
		trained = True

	monkeypatch.setattr(control_module, 'run_voxel_label_budget_job', train)
	result = run_f3_lithology_voxel_label_budget_control(
		config, only_missing=True
	)

	assert not trained
	assert [row['action'] for row in result.rows] == ['REUSED']


def test_resume_accepts_latest_only_and_rejects_new_jobs(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, job, inspection = _runner_fixture(tmp_path, state='RESUME_LATEST')
	seen: list[Path | None] = []
	monkeypatch.setattr(
		control_module,
		'inspect_f3_lithology_voxel_label_budget_control',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		control_module, '_prior_control_state', lambda *_args, **_kwargs: ((), [])
	)
	monkeypatch.setattr(
		control_module,
		'_completed_control_row',
		lambda *_args, **kwargs: _complete_row(job, action=kwargs['action']),
	)
	monkeypatch.setattr(
		control_module, '_validate_paired_identity', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		control_module, '_write_control_manifest', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		control_module, '_write_status_csv', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		control_module,
		'run_voxel_label_budget_job',
		lambda *_args, **kwargs: seen.append(kwargs['resume']),
	)

	run_f3_lithology_voxel_label_budget_control(config, resume=True)

	assert seen == [job.decoder_dir / 'latest.pt']
	new_inspection = SimpleNamespace(
		jobs=(job,),
		plans=(VoxelLabelBudgetJobPlan(job=job, state='NEW'),),
		reference=inspection.reference,
		dataset_rows=inspection.dataset_rows,
		candidate_embedding_identity=inspection.candidate_embedding_identity,
		estimated_new_bytes=0,
		disk_free_bytes=1,
	)
	monkeypatch.setattr(
		control_module,
		'inspect_f3_lithology_voxel_label_budget_control',
		lambda *_args, **_kwargs: new_inspection,
	)
	with pytest.raises(ValueError, match='--resume requires'):
		run_f3_lithology_voxel_label_budget_control(config, resume=True)
	assert _blocked_summary(config)['blocked_stage'] == 'runner_resume_preflight'


def test_only_missing_quarantines_invalid_partial_output(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, job, inspection = _runner_fixture(tmp_path, state='INVALID_OR_PARTIAL')
	quarantined: list[Path] = []
	monkeypatch.setattr(
		control_module,
		'inspect_f3_lithology_voxel_label_budget_control',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		control_module, '_prior_control_state', lambda *_args, **_kwargs: ((), [])
	)
	monkeypatch.setattr(
		control_module,
		'quarantine_voxel_label_budget_output',
		lambda path, **_kwargs: quarantined.append(path)
		or (path.parent / 'quarantine'),
	)
	monkeypatch.setattr(
		control_module, 'run_voxel_label_budget_job', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		control_module,
		'_completed_control_row',
		lambda *_args, **kwargs: _complete_row(job, action=kwargs['action']),
	)
	monkeypatch.setattr(
		control_module, '_validate_paired_identity', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		control_module, '_write_control_manifest', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		control_module, '_write_status_csv', lambda *_args, **_kwargs: None
	)

	run_f3_lithology_voxel_label_budget_control(config, only_missing=True)

	assert quarantined == [job.output_root]


def test_lower_is_better_paired_summary_counts_negative_delta_as_win() -> None:
	config = SimpleNamespace(
		budgets=('cap25',),
		subsample_seeds=(0, 1, 2, 3, 4),
		comparisons=((CURRENT_MODEL_ROLE, 'mae'),),
	)
	rows = [
		{
			'budget_id': 'cap25',
			'comparison_id': 'm1_current_k6_vs_mae',
			'subsample_seed': seed,
			**{
				metric.name: (
					-1.0
					if metric.name == 'vertical_boundary_position_mae'
					else 1.0
				)
				for metric in METRIC_SPECS
			},
		}
		for seed in config.subsample_seeds
	]

	summary = summarize_rows(config, rows)
	boundary = next(
		row
		for row in summary
		if row['metric'] == 'vertical_boundary_position_mae'
	)

	assert boundary['wins'] == 5
	assert boundary['losses'] == 0


@pytest.mark.parametrize(
	('coverage', 'match'),
	[
		(
			{
				'duplicate_write_count': 1,
				'missing_write_count': 0,
				'exact_once': True,
			},
			'duplicate_write_count',
		),
		(
			{
				'duplicate_write_count': 0,
				'missing_write_count': 1,
				'exact_once': True,
			},
			'missing_write_count',
		),
		(
			{
				'duplicate_write_count': 0,
				'missing_write_count': 0,
				'exact_once': False,
			},
			'exact_once',
		),
	],
)
def test_prediction_coverage_rejects_incomplete_or_duplicate_writes(
	tmp_path: Path, coverage: dict[str, object], match: str
) -> None:
	metadata = tmp_path / 'prediction_metadata.json'
	metadata.write_text(json.dumps({'coverage': coverage}), encoding='utf-8')

	with pytest.raises(ValueError, match=match):
		control_module._prediction_coverage(metadata)  # noqa: SLF001


def test_blocked_contract_materializes_summary_and_status(tmp_path: Path) -> None:
	config = _config(tmp_path)
	config.reports_dir = tmp_path / 'reports'

	control_module._write_blocked_control_contract(  # noqa: SLF001
		config,
		stage='runner_job',
		error=RuntimeError('coverage mismatch'),
	)

	summary = _blocked_summary(config)
	assert summary['status'] == 'BLOCKED_CONTROL_CONTRACT'
	assert summary['blocked_stage'] == 'runner_job'
	status = (config.reports_dir / 'control_job_status.csv').read_text(
		encoding='utf-8'
	)
	assert 'BLOCKED_CONTROL_CONTRACT' in status


def test_runner_setup_failure_materializes_blocked_contract(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, _job, inspection = _runner_fixture(tmp_path, state='NEW')
	monkeypatch.setattr(
		control_module,
		'inspect_f3_lithology_voxel_label_budget_control',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		control_module,
		'_prior_control_state',
		lambda *_args, **_kwargs: (_ for _ in ()).throw(
			RuntimeError('corrupt control manifest')
		),
	)

	with pytest.raises(RuntimeError, match='corrupt control manifest'):
		run_f3_lithology_voxel_label_budget_control(config)

	assert _blocked_summary(config)['blocked_stage'] == 'runner_setup'


@pytest.mark.parametrize(
	('failure_point', 'expected_stage'),
	[
		('inspect', 'summary_inspect'),
		('evidence', 'summary_evidence'),
		('publish', 'summary_publish'),
	],
)
def test_summary_failures_materialize_blocked_contract(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	failure_point: str,
	expected_stage: str,
) -> None:
	config = _config(tmp_path)
	config.reports_dir = tmp_path / 'reports'
	if failure_point == 'inspect':
		monkeypatch.setattr(
			control_module,
			'inspect_f3_lithology_voxel_label_budget_control_results',
			lambda *_args, **_kwargs: (_ for _ in ()).throw(
				RuntimeError('summary inspect failure')
			),
		)
	elif failure_point == 'evidence':
		monkeypatch.setattr(
			control_module,
			'inspect_f3_lithology_voxel_label_budget_control_results',
			lambda *_args, **_kwargs: SimpleNamespace(readiness={}),
		)
		monkeypatch.setattr(
			control_module,
			'_materialize_required_evidence',
			lambda *_args, **_kwargs: (_ for _ in ()).throw(
				RuntimeError('summary evidence failure')
			),
		)
	else:
		monkeypatch.setattr(
			control_module,
			'inspect_f3_lithology_voxel_label_budget_control_results',
			lambda *_args, **_kwargs: SimpleNamespace(readiness={}),
		)
		monkeypatch.setattr(
			control_module,
			'_materialize_required_evidence',
			lambda *_args, **_kwargs: None,
		)
		monkeypatch.setattr(
			control_module,
			'_validate_summary_output_availability',
			lambda *_args, **_kwargs: None,
		)
		monkeypatch.setattr(
			control_module,
			'load_f3_lithology_voxel_label_budget_control_rows',
			lambda *_args, **_kwargs: (),
		)
		monkeypatch.setattr(
			control_module,
			'_write_summary_tables',
			lambda *_args, **_kwargs: (),
		)
		monkeypatch.setattr(
			control_module,
			'_control_summary_payload',
			lambda *_args, **_kwargs: {},
		)
		monkeypatch.setattr(
			control_module,
			'_render_control_summary_markdown',
			lambda *_args, **_kwargs: 'summary',
		)
		monkeypatch.setattr(
			control_module,
			'_render_control_handoff_markdown',
			lambda *_args, **_kwargs: 'handoff',
		)
		monkeypatch.setattr(
			control_module,
			'_publish_control_results',
			lambda *_args, **_kwargs: (_ for _ in ()).throw(
				RuntimeError('summary publish failure')
			),
		)

	with pytest.raises(RuntimeError, match='summary'):
		summarize_f3_lithology_voxel_label_budget_control(config)

	summary = _blocked_summary(config)
	assert summary['status'] == 'BLOCKED_CONTROL_CONTRACT'
	assert summary['blocked_stage'] == expected_stage
	assert 'BLOCKED_CONTROL_CONTRACT' in (
		config.reports_dir / 'control_job_status.csv'
	).read_text(encoding='utf-8')


def test_final_git_provenance_records_status_and_binary_diff_hash(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	def git_text(arguments: tuple[str, ...], **_kwargs: object) -> str:
		if arguments == ('rev-parse', '--show-toplevel'):
			return '/workspace\n'
		if arguments == ('rev-parse', 'HEAD'):
			return 'abc123\n'
		assert arguments == ('status', '--short', '--untracked-files=all')
		return ' M source.py\n?? untracked.txt\n'

	monkeypatch.setattr(control_module, '_git_text', git_text)
	monkeypatch.setattr(
		control_module,
		'_git_bytes',
		lambda *_args, **_kwargs: b'binary-diff',
	)

	provenance = control_module._git_provenance()  # noqa: SLF001

	assert provenance['head'] == 'abc123'
	assert provenance['changed_files'] == (' M source.py', '?? untracked.txt')
	assert provenance['changed_file_count'] == 2
	assert provenance['git_diff_binary_head_sha256'] == (
		'3d7e4f98e5efaf94e222009552ca1964f0faef340b67304d5157d1f641091a2a'
	)


def _config(tmp_path: Path) -> SimpleNamespace:
	candidate = SimpleNamespace(
		model_id=CURRENT_MODEL_ROLE,
		model_tag='strat_hmm_pretext_m1_current_k6_topblock1_distill_v1',
	)
	references = SimpleNamespace(mae_model_id='mae', historical_m1_model_id='m1')
	decision = SimpleNamespace(
		minimum_positive_budgets=2,
		minimum_primary_wins=4,
		drift_absolute_mean_delta=0.01,
		drift_budget_count=2,
		monitored_class_ids=(3, 5),
		major_degradation_delta=-0.05,
		systematic_degradation_budget_count=2,
	)
	return SimpleNamespace(
		candidate=candidate,
		references=references,
		decision=decision,
		budgets=('cap25', 'cap50', 'cap100'),
		subsample_seeds=(0, 1, 2, 3, 4),
		output_root=tmp_path,
		decoder_seed=lambda seed: 42000 + seed,
	)


def _runner_fixture(
	tmp_path: Path, *, state: str
) -> tuple[SimpleNamespace, VoxelLabelBudgetJob, SimpleNamespace]:
	config = _config(tmp_path)
	config.reports_dir = tmp_path / 'reports'
	job = VoxelLabelBudgetJob(
		budget_id='cap25',
		per_class_cap=25,
		subsample_seed=0,
		decoder_seed=42000,
		model_role=CURRENT_MODEL_ROLE,
		model_tag=config.candidate.model_tag,
		voxel_dataset_root=tmp_path / 'dataset',
		output_root=tmp_path / 'job',
		dataset_row={'per_class_cap': 25},
	)
	inspection = SimpleNamespace(
		jobs=(job,),
		plans=(VoxelLabelBudgetJobPlan(job=job, state=state, reason='fixture'),),
		reference=SimpleNamespace(),
		dataset_rows={('cap25', 0): {}},
		candidate_embedding_identity={},
		estimated_new_bytes=0,
		disk_free_bytes=1,
	)
	return config, job, inspection


def _complete_row(job: VoxelLabelBudgetJob, *, action: str) -> dict[str, object]:
	return {
		'budget_id': job.budget_id,
		'per_class_cap': job.per_class_cap,
		'subsample_seed': job.subsample_seed,
		'decoder_seed': job.decoder_seed,
		'model_role': job.model_role,
		'model_tag': job.model_tag,
		'status': 'complete',
		'action': action,
		'error': None,
		'quarantine_path': None,
	}


def _blocked_summary(config: SimpleNamespace) -> dict[str, object]:
	return json.loads(
		(config.reports_dir / 'current_k6_control_summary.json').read_text(
			encoding='utf-8'
		)
	)


def _summary_rows(
	config: SimpleNamespace, *, current_vs_mae: float, current_vs_m1: float
) -> list[dict[str, object]]:
	rows = []
	for budget in config.budgets:
		for comparison_id, delta in (
			('m1_current_k6_vs_mae', current_vs_mae),
			('m1_current_k6_vs_m1', current_vs_m1),
		):
			rows.extend(
				{
					'budget_id': budget,
					'comparison_id': comparison_id,
					'metric': metric.name,
					'mean_delta': delta,
					'wins': 4,
				}
				for metric in METRIC_SPECS
			)
	return rows
