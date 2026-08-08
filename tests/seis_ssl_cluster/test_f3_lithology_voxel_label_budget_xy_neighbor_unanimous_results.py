"""Focused original-split gate contracts for unanimous XY-neighbour screening."""
# ruff: noqa: SLF001

from __future__ import annotations

from types import SimpleNamespace

import pytest

from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_xy_neighbor_unanimous_results as results,
)


def test_unanimous_publishes_only_explicit_lightweight_files(
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
		'overall_status': 'M5_XYUNANIM_ORIGINAL_HOLD',
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
		'inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous_results',
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

	result = results.summarize_f3_lithology_voxel_label_budget_xy_neighbor_unanimous(
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


def test_unanimous_gate_go_hold_stop_and_diagnostic_is_not_a_gate() -> None:
	budgets = ('cap25', 'cap50', 'cap100')
	assert (
		results.decide_xy_neighbor_unanimous_original_gate(
			_rows(budgets), budgets=budgets
		)['overall_status']
		== 'XYUNANIM_ORIGINAL_GO'
	)
	assert (
		results.decide_xy_neighbor_unanimous_original_gate(
			_rows(budgets, positive_budgets=('cap25',)), budgets=budgets
		)['overall_status']
		== 'XYUNANIM_ORIGINAL_HOLD'
	)
	assert (
		results.decide_xy_neighbor_unanimous_original_gate(
			_rows(budgets, negative_budgets=('cap25', 'cap50')), budgets=budgets
		)['overall_status']
		== 'XYUNANIM_ORIGINAL_STOP'
	)
	assert (
		results.decide_xy_neighbor_unanimous_original_gate(
			_rows(budgets, degraded=(3, 'f1')), budgets=budgets
		)['overall_status']
		== 'XYUNANIM_ORIGINAL_STOP'
	)
	assert (
		results.decide_xy_neighbor_unanimous_original_gate(
			_rows(budgets, degraded=(5, 'iou')), budgets=budgets
		)['overall_status']
		== 'XYUNANIM_ORIGINAL_STOP'
	)

	rows = _rows(budgets)
	for row in rows:
		if row['comparison_id'] == 'mh_xyunanim1_nocons_vs_mh_xycons1_nocons':
			row['mean_delta'] = -1.0
			row['wins'] = 0
	decision = results.decide_xy_neighbor_unanimous_original_gate(rows, budgets=budgets)
	assert decision['overall_status'] == 'XYUNANIM_ORIGINAL_GO'
	assert (
		decision['diagnostic_unanimous_vs_xy_neighbor_consensus']['gate_effect']
		== 'none'
	)
	assert decision['six_split_follow_up'] == {
		'ready': True,
		'scientific_jobs_executed': 0,
		'six_split_jobs_executed': 0,
	}


def test_unanimous_gate_requires_four_of_five_primary_wins() -> None:
	decision = results.decide_xy_neighbor_unanimous_original_gate(
		_rows(('cap25', 'cap50', 'cap100'), wins=3),
		budgets=('cap25', 'cap50', 'cap100'),
	)
	assert decision['overall_status'] == 'XYUNANIM_ORIGINAL_HOLD'
	assert decision['six_split_follow_up']['ready'] is False


def test_five_role_matrix_requires_exact_75_rows_and_pair_identities() -> None:
	config = SimpleNamespace(
		budgets=('cap25', 'cap50', 'cap100'), subsample_seeds=(0, 1, 2, 3, 4)
	)
	roles = (
		'mae',
		'm1_current_k6',
		'mh_nocons',
		'mh_xycons1_nocons',
		'mh_xyunanim1_nocons',
	)
	row = dict.fromkeys(results.control.PAIR_IDENTITY_KEYS, 'shared')
	members = {
		(budget, seed, role): {'row': dict(row)}
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in roles
	}
	results._expected_matrix(config, members)
	results._validate_pairing(config, members)
	assert len(members) == 75
	members[('cap25', 0, 'mh_xyunanim1_nocons')]['row']['decoder_seed'] = 'drift'
	with pytest.raises(ValueError, match='paired identity mismatch'):
		results._validate_pairing(config, members)


def test_consensus_reference_requires_exact_completed_fifteen_rows(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	manifest = tmp_path / 'xycons.json'
	manifest.write_text('{}', encoding='utf-8')
	reference = {
		'path': str(manifest),
		'sha256': results.file_sha256(manifest),
	}
	payload = {
		'artifact_type': 'f3_lithology_voxel_label_budget_xy_neighbor_consensus',
		'schema_version': 1,
		'row_count': 15,
		'complete_count': 15,
		'rows': [
			{
				'model_role': 'mh_xycons1_nocons',
				'model_tag': (
					'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
				),
				'status': 'complete',
			}
			for _ in range(15)
		],
	}
	monkeypatch.setattr(results, '_read_json', lambda _path: payload)
	rows = results._load_xy_neighbor_consensus_reference_rows(
		{'reference_run_manifests': {'xy_neighbor_consensus': reference}}
	)
	assert len(rows) == 15
	payload['complete_count'] = 14
	with pytest.raises(ValueError, match='incomplete'):
		results._load_xy_neighbor_consensus_reference_rows(
			{'reference_run_manifests': {'xy_neighbor_consensus': reference}}
		)


def _rows(
	budgets: tuple[str, ...],
	*,
	positive_budgets: tuple[str, ...] = ('cap25', 'cap50'),
	negative_budgets: tuple[str, ...] = (),
	degraded: tuple[int, str] | None = None,
	wins: int = 4,
) -> list[dict[str, object]]:
	rows = []
	for comparison_id in (
		'mh_xyunanim1_nocons_vs_mh_nocons',
		'mh_xyunanim1_nocons_vs_mh_xycons1_nocons',
	):
		for budget in budgets:
			for metric in (
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
			):
				value = 0.0
				metric_wins = 0
				if (
					comparison_id == 'mh_xyunanim1_nocons_vs_mh_nocons'
					and budget in positive_budgets
					and metric in {'macro_f1', 'mean_iou'}
				):
					value = 0.1
					metric_wins = wins
				if (
					comparison_id == 'mh_xyunanim1_nocons_vs_mh_nocons'
					and budget in negative_budgets
					and metric in {'macro_f1', 'mean_iou'}
				):
					value = -0.1
					metric_wins = 1
				if (
					comparison_id == 'mh_xyunanim1_nocons_vs_mh_nocons'
					and degraded is not None
					and metric == f'class_{degraded[0]}_{degraded[1]}'
					and budget != 'cap100'
				):
					value = -0.05
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
