"""Decision boundaries for the M5-U original-split GO gate."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_soft_posterior import (
	f3_lithology_voxel_label_budget_soft_posterior_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_soft_posterior_results import (
	_portable_value,
	decide_soft_posterior_original_gate,
)


def test_soft_posterior_gate_go_hold_and_stop() -> None:
	"""Require two positive budgets and reject systematic monitored-class harm."""
	budgets = ('cap25', 'cap50', 'cap100')
	rows = _rows(budgets)
	assert (
		decide_soft_posterior_original_gate(rows, budgets=budgets)['overall_status']
		== 'M5_U_ORIGINAL_GO'
	)
	rows = _rows(budgets, positive_budgets=('cap25',))
	assert (
		decide_soft_posterior_original_gate(rows, budgets=budgets)['overall_status']
		== 'M5_U_ORIGINAL_HOLD'
	)
	rows = _rows(budgets, degraded_class_3=True)
	assert (
		decide_soft_posterior_original_gate(rows, budgets=budgets)['overall_status']
		== 'M5_U_ORIGINAL_STOP'
	)


def test_soft_config_rejects_reference_contract_drift_from_hard_config(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Reject M5-U planning when a paired original-split input differs."""
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('F3_ROOT', '/home/dcuser/data/public_data/field/F3')
	path = Path(
		'experiments/f3/facies_benchmark_v1/'
		'98_strat_hmm_multi_head_k6810_soft_posterior_low_label_v1/'
		'01_run_soft_voxel_label_budget.yaml'
	)
	raw = deepcopy(load_config(path))
	raw['references']['multi_head_target_manifest'] = raw['references'][
		'dataset_manifest'
	]

	with pytest.raises(
		ValueError, match='contract mismatch: multi_head_target_manifest'
	):
		f3_lithology_voxel_label_budget_soft_posterior_config_from_mapping(raw)


def test_soft_posterior_publish_paths_are_portable() -> None:
	"""Publish provenance without embedding the local workspace path."""
	payload = _portable_value(
		{
			'artifact': '/workspace/artifacts/seis_ssl_cluster/reports/summary.json',
			'repository': '/workspace/experiments/f3/config.yaml',
		},
		artifact_root=Path('/workspace/artifacts/seis_ssl_cluster'),
		workspace_root=Path('/workspace'),
	)

	assert payload == {
		'artifact': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/reports/summary.json',
		'repository': 'experiments/f3/config.yaml',
	}


def _rows(
	budgets: tuple[str, ...],
	*,
	positive_budgets: tuple[str, ...] = ('cap25', 'cap50'),
	degraded_class_3: bool = False,
) -> list[dict[str, object]]:
	rows = []
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
			value = 0.1 if budget in positive_budgets else 0.0
			if degraded_class_3 and metric == 'class_3_f1' and budget != 'cap100':
				value = -0.05
			rows.append(
				{
					'budget_id': budget,
					'comparison_id': 'mh_soft_nocons_vs_mh_nocons',
					'metric': metric,
					'mean_delta': value,
					'wins': 4 if value > 0 else 0,
				}
			)
	return rows
