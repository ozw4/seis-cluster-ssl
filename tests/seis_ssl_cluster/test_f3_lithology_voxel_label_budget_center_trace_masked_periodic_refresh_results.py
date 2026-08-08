"""Synthetic contracts for the periodic-refresh original-split gate."""
# ruff: noqa: CPY001, E501, S108, SLF001

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked_periodic_refresh as runner,
)
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked_periodic_refresh_results as results,
)
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as shared
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head_results as shared_results,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import METRIC_SPECS


def test_periodic_refresh_publishes_only_explicit_lightweight_files(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	reports = tmp_path / 'artifacts/reports'
	results_root = tmp_path / 'results'
	config = SimpleNamespace(
		reports_dir=reports,
		base=SimpleNamespace(
			artifact_root=tmp_path / 'artifacts',
			results_root=results_root,
			publish=SimpleNamespace(results_root=results_root),
		),
	)
	decisions = {
		'overall_status': 'M5_PERIODIC_ORIGINAL_HOLD',
		'six_split_follow_up': {'ready': False},
	}
	inspection = {
		'job_metrics': ({'metric': 1},),
		'paired_deltas': ({'delta': 0.0},),
		'decisions': decisions,
		'screening_audit': {},
	}
	monkeypatch.setattr(
		results,
		'inspect_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh_results',
		lambda _config: inspection,
	)
	monkeypatch.setattr(results, '_execution_git_state', lambda _config: {})
	monkeypatch.setattr(results, '_handoff_payload', lambda *_args, **_kwargs: {})

	result = results.summarize_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh(
		config
	)
	published = results_root / results._PUBLISHED_ROOT

	assert result['decisions'] == decisions
	assert {path.name for path in published.iterdir()} == set(
		results.PUBLISHED_OUTPUT_NAMES
	)
	assert not any(
		path.suffix in {'.npy', '.npz', '.pt'} for path in published.iterdir()
	)


def test_periodic_refresh_job_matrix_and_seed_identity() -> None:
	config = SimpleNamespace(
		candidates=(
			SimpleNamespace(
				model_id='mh_ctmask010_refresh3ep_hmm2_nocons',
				model_tag=(
					'strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_'
					'topblock1_distill_v1'
				),
			),
		),
		budgets=('cap25', 'cap50', 'cap100'),
		subsample_seeds=(0, 1, 2, 3, 4),
		output_root=Path('/tmp/periodic-refresh-test'),
		decoder_seed=lambda seed: 42000 + seed,
	)
	dataset_rows = {
		(budget, seed): {
			'per_class_cap': int(budget.removeprefix('cap')),
			'voxel_dataset_root': f'/tmp/{budget}/{seed}',
		}
		for budget in config.budgets
		for seed in config.subsample_seeds
	}
	jobs = shared._jobs(config, dataset_rows)
	assert len(jobs) == 15
	assert {job.model_role for job in jobs} == {'mh_ctmask010_refresh3ep_hmm2_nocons'}
	assert {job.decoder_seed for job in jobs} == {
		42000,
		42001,
		42002,
		42003,
		42004,
	}
	assert runner.MODEL_ID == 'mh_ctmask010_refresh3ep_hmm2_nocons'


def test_periodic_refresh_gate_boundaries_and_six_split_zero() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	go = results.decide_center_trace_masked_periodic_refresh_original_gate(
		_rows(budgets), budgets=budgets
	)
	assert go['overall_status'] == 'CTMASK_REFRESH_ORIGINAL_GO'
	assert go['six_split_jobs_executed'] == 0
	assert go['six_split_scientific_jobs_executed'] == 0
	assert go['six_split_follow_up']['ready'] is True

	hold = results.decide_center_trace_masked_periodic_refresh_original_gate(
		_rows(budgets, positive_budgets=('cap25',)), budgets=budgets
	)
	assert hold['overall_status'] == 'CTMASK_REFRESH_ORIGINAL_HOLD'
	assert hold['six_split_follow_up']['ready'] is False

	stop = results.decide_center_trace_masked_periodic_refresh_original_gate(
		_rows(budgets, negative_budgets=('cap25', 'cap50')), budgets=budgets
	)
	assert stop['overall_status'] == 'CTMASK_REFRESH_ORIGINAL_STOP'
	assert stop['six_split_follow_up']['ready'] is False


def test_periodic_refresh_gate_accepts_full_shared_summary() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	config = SimpleNamespace(budgets=budgets)
	primary_comparison_id = results.control._comparison_id(
		results._CANDIDATE_ROLE, results._FIXED_CENTER_ROLE
	)
	deltas = []
	for budget in budgets:
		for seed in range(5):
			for candidate, baseline in results.COMPARISONS:
				comparison_id = results.control._comparison_id(candidate, baseline)
				row: dict[str, object] = {
					'budget_id': budget,
					'comparison_id': comparison_id,
					'subsample_seed': seed,
				}
				for metric in METRIC_SPECS:
					row[metric.name] = (
						0.1
						if (
							comparison_id == primary_comparison_id
							and budget in {'cap25', 'cap50'}
							and metric.name in {'macro_f1', 'mean_iou'}
						)
						else 0.0
					)
				deltas.append(row)

	summary = shared_results._summary(
		config, deltas, comparisons=results.COMPARISONS
	)

	assert len(summary) == len(budgets) * len(results.COMPARISONS) * len(
		METRIC_SPECS
	)
	decision = results.decide_center_trace_masked_periodic_refresh_original_gate(
		summary, budgets=budgets
	)
	assert decision['overall_status'] == 'CTMASK_REFRESH_ORIGINAL_GO'


def test_periodic_refresh_gate_uses_strict_primary_boundaries_and_inclusive_guardrail() -> (
	None
):
	budgets = ('cap25', 'cap50', 'cap100')
	rows = _rows(budgets, positive_budgets=budgets, wins=4)
	for row in rows:
		if row['metric'] in {'macro_f1', 'mean_iou'}:
			row['mean_delta'] = 0.0
	decision = results.decide_center_trace_masked_periodic_refresh_original_gate(
		rows, budgets=budgets
	)
	assert decision['overall_status'] == 'CTMASK_REFRESH_ORIGINAL_HOLD'

	rows = _rows(budgets, positive_budgets=budgets, wins=4)
	for row in rows:
		if row['metric'] == 'class_3_f1' and row['budget_id'] != 'cap100':
			row['mean_delta'] = -0.05
	decision = results.decide_center_trace_masked_periodic_refresh_original_gate(
		rows, budgets=budgets
	)
	assert decision['overall_status'] == 'CTMASK_REFRESH_ORIGINAL_STOP'


def test_periodic_refresh_gate_rejects_duplicate_or_nonfinite_rows() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	rows = _rows(budgets)
	with pytest.raises(ValueError, match='duplicate'):
		results.decide_center_trace_masked_periodic_refresh_original_gate(
			[*rows, dict(rows[0])], budgets=budgets
		)
	wrong_identity = dict(rows[0])
	wrong_identity['comparison_id'] = 'foreign_comparison'
	with pytest.raises(ValueError, match='wrong identity'):
		results.decide_center_trace_masked_periodic_refresh_original_gate(
			[*rows, wrong_identity], budgets=budgets
		)
	rows[0]['mean_delta'] = float('nan')
	with pytest.raises(ValueError, match='non-finite'):
		results.decide_center_trace_masked_periodic_refresh_original_gate(
			rows, budgets=budgets
		)


def test_periodic_refresh_expected_matrix_has_exact_five_roles() -> None:
	config = SimpleNamespace(
		budgets=('cap25', 'cap50', 'cap100'),
		subsample_seeds=(0, 1, 2, 3, 4),
	)
	roles = (
		'mae',
		'm1_current_k6',
		'mh_nocons',
		'mh_ctmask010_nocons',
		'mh_ctmask010_refresh3ep_hmm2_nocons',
	)
	members = {
		(budget, seed, role): {'row': {}}
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in roles
	}
	results._expected_matrix(config, members)


def test_periodic_refresh_handoff_binds_matrix_and_gate_report_bytes(
	tmp_path: Path,
) -> None:
	reports = tmp_path / 'reports'
	reports.mkdir()
	for name in results.REPORT_OUTPUT_NAMES[:4]:
		(reports / name).write_text(name, encoding='utf-8')
	inspection = {
		'job_metrics': tuple({'model_role': 'role'} for _ in range(75)),
		'source_identities': {
			'periodic_refresh_handoff': {'sha256': 'a' * 64},
			'candidate_provenance': {'model_id': 'candidate'},
			'screening_audit': {'sha256': 'b' * 64},
			'candidate_run_manifest': {'sha256': 'c' * 64},
			'candidate_job_live_validation': {'status': 'PASS'},
			'fixed_center_trace_run_manifest': {'sha256': 'd' * 64},
			'hard_decoder_config': {'sha256': 'e' * 64},
			'reference_run_manifests': {},
			'paired_matrix_identity': {'row_count': 75},
		},
		'decisions': {
			'gate': {'primary': 'fixed'},
			'overall_status': 'CTMASK_REFRESH_ORIGINAL_HOLD',
			'six_split_follow_up': {'ready': False},
			'scientific_jobs_executed': 15,
		},
	}

	handoff = results._handoff_payload(
		inspection,
		execution={'git_sha': 'f' * 40},
		reports_dir=reports,
	)
	report_evidence = handoff['reports']
	assert report_evidence['paired_matrix']['row_count'] == 75  # type: ignore[index]
	assert report_evidence['paired_matrix']['sha256'] == (  # type: ignore[index]
		results.file_sha256(reports / results.REPORT_OUTPUT_NAMES[0])
	)
	assert set(report_evidence['gate_reports']) == {  # type: ignore[index]
		'paired_deltas',
		'summary_json',
		'summary_markdown',
	}


def _rows(
	budgets: tuple[str, ...],
	*,
	positive_budgets: tuple[str, ...] = ('cap25', 'cap50'),
	negative_budgets: tuple[str, ...] = (),
	wins: int = 4,
) -> list[dict[str, object]]:
	rows = []
	metrics = tuple(metric.name for metric in METRIC_SPECS)
	for budget in budgets:
		for candidate, baseline in results.COMPARISONS:
			comparison_id = results.control._comparison_id(candidate, baseline)
			for metric in metrics:
				value = 0.0
				metric_wins = 0
				if budget in positive_budgets and metric in {'macro_f1', 'mean_iou'}:
					value, metric_wins = 0.1, wins
				if budget in negative_budgets and metric in {'macro_f1', 'mean_iou'}:
					value, metric_wins = -0.1, 1
				rows.append(
					{
						'budget_id': budget,
						'comparison_id': comparison_id,
						'metric': metric,
						'mean_delta': value,
						'wins': metric_wins,
					}
				)
	return rows
