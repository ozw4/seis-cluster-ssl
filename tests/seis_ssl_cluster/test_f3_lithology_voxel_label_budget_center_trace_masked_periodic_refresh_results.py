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


def _rows(
	budgets: tuple[str, ...],
	*,
	positive_budgets: tuple[str, ...] = ('cap25', 'cap50'),
	negative_budgets: tuple[str, ...] = (),
	wins: int = 4,
) -> list[dict[str, object]]:
	rows = []
	metrics = (
		'macro_f1',
		'mean_iou',
		'class_3_f1',
		'class_3_iou',
		'class_3_boundary_recall_t2',
		'class_3_boundary_recall_t4',
		'class_5_f1',
		'class_5_iou',
		'class_5_boundary_recall_t2',
		'class_5_boundary_recall_t4',
	)
	for budget in budgets:
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
					'comparison_id': 'mh_ctmask010_refresh3ep_hmm2_nocons_vs_mh_ctmask010_nocons',
					'metric': metric,
					'mean_delta': value,
					'wins': metric_wins,
				}
			)
	return rows
