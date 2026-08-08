"""Fixed gate and paired-matrix coverage for center-trace screening."""
# ruff: noqa: CPY001, PLR0913, SLF001

from __future__ import annotations

from types import SimpleNamespace

import pytest

from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked_results as results,
)


def test_center_trace_masked_publishes_only_explicit_lightweight_files(
	tmp_path, monkeypatch: pytest.MonkeyPatch
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
		'overall_status': 'M5_CT_ORIGINAL_HOLD',
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
		'inspect_f3_lithology_voxel_label_budget_center_trace_masked_results',
		lambda _config: inspection,
	)
	monkeypatch.setattr(results, '_execution_git_state', lambda _config: {})
	monkeypatch.setattr(results, '_handoff_payload', lambda *_args, **_kwargs: {})
	monkeypatch.setattr(
		results,
		'_write_audit_evidence',
		lambda output, **_kwargs: [
			(output / name).write_text('{}\n', encoding='utf-8')
			for name in results.AUDIT_OUTPUT_NAMES
		],
	)

	result = results.summarize_f3_lithology_voxel_label_budget_center_trace_masked(
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


def test_center_trace_masked_gate_boundaries() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	assert (
		results.decide_center_trace_masked_original_gate(
			_rows(budgets), budgets=budgets
		)['overall_status']
		== 'CTMASK_ORIGINAL_GO'
	)
	assert (
		results.decide_center_trace_masked_original_gate(
			_rows(budgets, positive_budgets=('cap25',)), budgets=budgets
		)['overall_status']
		== 'CTMASK_ORIGINAL_HOLD'
	)
	assert (
		results.decide_center_trace_masked_original_gate(
			_rows(budgets, negative_budgets=('cap25', 'cap50')), budgets=budgets
		)['overall_status']
		== 'CTMASK_ORIGINAL_STOP'
	)
	assert (
		results.decide_center_trace_masked_original_gate(
			_rows(budgets, degraded=True), budgets=budgets
		)['overall_status']
		== 'CTMASK_ORIGINAL_STOP'
	)


def test_center_trace_masked_gate_requires_four_of_five_wins() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	decision = results.decide_center_trace_masked_original_gate(
		_rows(budgets, wins=3), budgets=budgets
	)
	assert decision['overall_status'] == 'CTMASK_ORIGINAL_HOLD'
	assert decision['six_split_follow_up']['ready'] is False
	assert decision['six_split_jobs_executed'] == 0
	assert decision['scientific_jobs_executed'] == 15
	assert decision['six_split_follow_up']['scientific_jobs_executed'] == 0


def test_center_trace_masked_gate_rejects_duplicate_metric_rows() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	rows = _rows(budgets)
	with pytest.raises(ValueError, match='duplicate'):
		results.decide_center_trace_masked_original_gate(
			[ *rows, dict(rows[0]) ], budgets=budgets
		)


def test_center_trace_masked_gate_rejects_nonfinite_metric_rows() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	rows = _rows(budgets)
	rows[0]['mean_delta'] = float('nan')
	with pytest.raises(ValueError, match='non-finite'):
		results.decide_center_trace_masked_original_gate(rows, budgets=budgets)


def test_center_trace_masked_handoff_reports_candidate_jobs() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	decision = results.decide_center_trace_masked_original_gate(
		_rows(budgets), budgets=budgets
	)
	identities = {
		key: {}
		for key in (
			'screening_audit',
			'candidate_run_manifest',
			'candidate_job_live_validation',
			'candidate_provenance',
			'hard_decoder_config',
			'reference_run_manifests',
			'paired_matrix_identity',
		)
	}
	handoff = results._handoff_payload(
		{'source_identities': identities, 'decisions': decision}, execution={}
	)
	assert handoff['scientific_jobs_executed'] == 15
	assert handoff['six_split_jobs_executed'] == 0
	assert handoff['six_split_follow_up']['scientific_jobs_executed'] == 0


def test_center_trace_masked_gate_uses_inclusive_degradation_threshold() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	decision = results.decide_center_trace_masked_original_gate(
		_rows(budgets, degraded=True, degradation_value=-0.05), budgets=budgets
	)
	assert decision['overall_status'] == 'CTMASK_ORIGINAL_STOP'


def test_center_trace_masked_matrix_requires_exact_roles_and_pairing() -> None:
	config = SimpleNamespace(
		budgets=('cap25', 'cap50', 'cap100'),
		subsample_seeds=(0, 1, 2, 3, 4),
	)
	roles = ('mae', 'm1_current_k6', 'mh_nocons', 'mh_ctmask010_nocons')
	row = dict.fromkeys(results.control.PAIR_IDENTITY_KEYS, 'shared')
	members = {
		(budget, seed, role): {'row': dict(row)}
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in roles
	}
	results._expected_matrix(config, members)
	results._validate_pairing(config, members)
	members[('cap25', 0, 'mh_ctmask010_nocons')]['row']['decoder_seed'] = 'drift'
	with pytest.raises(ValueError, match='paired identity mismatch'):
		results._validate_pairing(config, members)


def _rows(
	budgets: tuple[str, ...],
	*,
	positive_budgets: tuple[str, ...] = ('cap25', 'cap50'),
	negative_budgets: tuple[str, ...] = (),
	wins: int = 4,
	degraded: bool = False,
	degradation_value: float = -0.05,
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
			if degraded and metric == 'class_3_f1' and budget != 'cap100':
				value = degradation_value
			rows.append(
				{
					'budget_id': budget,
					'comparison_id': 'mh_ctmask010_nocons_vs_mh_nocons',
					'metric': metric,
					'mean_delta': value,
					'wins': metric_wins,
				}
			)
	return rows
