"""Decision boundaries for the M5-U original-split GO gate."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_soft_posterior import (
	f3_lithology_voxel_label_budget_soft_posterior_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_soft_posterior_results as soft_posterior_results,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_soft_posterior_results import (
	OUTPUT_NAMES,
	_portable_value,
	decide_soft_posterior_original_gate,
	summarize_f3_lithology_voxel_label_budget_soft_posterior,
)

RESULTS_RELATIVE_ROOT = Path(
	'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_soft_posterior_v1'
)
REQUIRED_RESULT_FILES = OUTPUT_NAMES


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
			'artifact_root': '/workspace/artifacts/seis_ssl_cluster',
			'artifact': '/workspace/artifacts/seis_ssl_cluster/reports/summary.json',
			'workspace_root': '/workspace',
			'repository': '/workspace/experiments/f3/config.yaml',
			'external': '/opt/other/summary.json',
		},
		artifact_root=Path('/workspace/artifacts/seis_ssl_cluster'),
		workspace_root=Path('/workspace'),
	)

	assert payload == {
		'artifact_root': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}',
		'artifact': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/reports/summary.json',
		'workspace_root': '.',
		'repository': 'experiments/f3/config.yaml',
		'external': '/opt/other/summary.json',
	}


def test_soft_posterior_summarizer_publishes_portable_paths(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	"""Serialize portable source reports before publishing their live bytes."""
	workspace_root = tmp_path / 'workspace'
	artifact_root = workspace_root / 'artifacts' / 'seis_ssl_cluster'
	reports_dir = artifact_root / 'reports'
	results_root = workspace_root / 'reports'
	config = SimpleNamespace(
		reports_dir=reports_dir,
		base=SimpleNamespace(
			artifact_root=artifact_root,
			results_root=results_root,
			publish=SimpleNamespace(results_root=results_root),
		),
	)
	inspection = _portable_inspection(
		artifact_root=artifact_root,
		workspace_root=workspace_root,
		external_path=tmp_path / 'external' / 'source.json',
	)
	monkeypatch.setattr(
		soft_posterior_results,
		'inspect_f3_lithology_voxel_label_budget_soft_posterior_results',
		lambda _config: inspection,
	)

	result = summarize_f3_lithology_voxel_label_budget_soft_posterior(config)
	published_dir = results_root / RESULTS_RELATIVE_ROOT
	assert result['decisions'] == inspection['decisions']
	assert {path.name for path in published_dir.iterdir()} == set(REQUIRED_RESULT_FILES)

	texts = {}
	for name in OUTPUT_NAMES:
		source = reports_dir / name
		published = published_dir / name
		assert source.read_bytes() == published.read_bytes()
		texts[name] = published.read_text(encoding='utf-8')
	assert all(str(artifact_root) not in text for text in texts.values())
	assert all(str(workspace_root) not in text for text in texts.values())

	summary = json.loads(texts['soft_posterior_results_summary.json'])
	assert summary['source_identities'] == {
		'artifact_child': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/reports/source.json',
		'artifact_root': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}',
		'external_path': str(tmp_path / 'external' / 'source.json'),
		'workspace_child': 'experiments/f3/config.yaml',
		'workspace_root': '.',
	}
	job_rows = list(
		csv.DictReader(texts['soft_posterior_job_metrics.csv'].splitlines())
	)
	assert '\r\n' not in texts['soft_posterior_job_metrics.csv']
	assert job_rows[0]['voxel_dataset_root'] == (
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/datasets/cap25'
	)
	assert job_rows[0]['source_config'] == 'experiments/f3/config.yaml'


def test_committed_soft_posterior_results_are_portable_and_valid() -> None:
	"""Keep published M5-U results portable and complete."""
	repository_root = Path(__file__).resolve().parents[2]
	results_dir = repository_root / 'reports' / RESULTS_RELATIVE_ROOT
	assert {path.name for path in results_dir.iterdir()} >= set(REQUIRED_RESULT_FILES)

	texts = {
		name: (results_dir / name).read_text(encoding='utf-8')
		for name in REQUIRED_RESULT_FILES
	}
	for text in texts.values():
		assert '/workspace/' not in text
		assert '/home/dcuser/' not in text

	summary = json.loads(texts['soft_posterior_results_summary.json'])
	decisions = summary['decisions']
	assert decisions['overall_status'] == 'M5_U_ORIGINAL_STOP'
	assert decisions['hard_vs_soft'] == {
		'positive_budgets': [],
		'negative_budgets': ['cap25', 'cap50'],
		'systematic_major_degradation': [],
	}
	assert decisions['six_split_follow_up'] == {
		'ready': False,
		'scientific_jobs_executed': 0,
	}

	job_rows = list(
		csv.DictReader(texts['soft_posterior_job_metrics.csv'].splitlines())
	)
	assert '\r\n' not in texts['soft_posterior_job_metrics.csv']
	assert len(job_rows) == 60
	for role in ('mh_soft_nocons', 'mh_nocons', 'm1_current_k6', 'mae'):
		assert sum(row['model_role'] == role for row in job_rows) == 15
	paired_rows = list(
		csv.DictReader(texts['soft_posterior_paired_deltas.csv'].splitlines())
	)
	assert len(paired_rows) == 45
	assert {
		row['comparison'] for row in paired_rows
	} == {
		'mh_soft_nocons - mh_nocons',
		'mh_soft_nocons - m1_current_k6',
		'mh_soft_nocons - mae',
	}
	for comparison in {row['comparison'] for row in paired_rows}:
		assert sum(row['comparison'] == comparison for row in paired_rows) == 15


def _portable_inspection(
	*,
	artifact_root: Path,
	workspace_root: Path,
	external_path: Path,
) -> dict[str, object]:
	artifact_child = artifact_root / 'reports' / 'source.json'
	workspace_child = workspace_root / 'experiments' / 'f3' / 'config.yaml'
	decisions = {
		'overall_status': 'M5_U_ORIGINAL_STOP',
		'hard_vs_soft': {
			'positive_budgets': [],
			'negative_budgets': ['cap25', 'cap50'],
			'systematic_major_degradation': [],
		},
		'six_split_follow_up': {
			'ready': False,
			'scientific_jobs_executed': 0,
		},
	}
	return {
		'job_metrics': (
			{
				'budget_id': 'cap25',
				'subsample_seed': 0,
				'model_role': 'mh_soft_nocons',
				'voxel_dataset_root': str(artifact_root / 'datasets' / 'cap25'),
				'source_config': str(workspace_child),
			},
		),
		'paired_deltas': (
			{
				'comparison_id': 'mh_soft_nocons_vs_mh_nocons',
				'comparison': 'mh_soft_nocons - mh_nocons',
				'artifact_path': str(artifact_child),
				'workspace_path': str(workspace_child),
			},
		),
		'summary_by_budget': (
			{
				'budget_id': 'cap25',
				'paths': [
					str(artifact_root),
					str(artifact_child),
					str(workspace_root),
					str(workspace_child),
					str(external_path),
				],
			},
		),
		'decisions': decisions,
		'source_identities': {
			'artifact_root': str(artifact_root),
			'artifact_child': str(artifact_child),
			'workspace_root': str(workspace_root),
			'workspace_child': str(workspace_child),
			'external_path': str(external_path),
		},
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
