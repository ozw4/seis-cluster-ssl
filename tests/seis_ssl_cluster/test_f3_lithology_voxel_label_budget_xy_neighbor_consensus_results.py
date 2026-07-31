"""Decision and lightweight-publication coverage for XY-consensus screening."""
# ruff: noqa: SLF001

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_xy_neighbor_consensus_results as xy_results,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_xy_neighbor_consensus_results import (  # noqa: E501
	AUDIT_OUTPUT_NAMES,
	OUTPUT_NAMES,
	PUBLISHED_OUTPUT_NAMES,
	REPORT_OUTPUT_NAMES,
	_spatial_audit_rows,
	decide_xy_neighbor_consensus_original_gate,
	summarize_f3_lithology_voxel_label_budget_xy_neighbor_consensus,
)
from seis_ssl_cluster.results import validate_results_artifacts

_RESULTS_ROOT = Path(
	'f3/facies_benchmark_v1/'
	'strat_hmm_multi_head_k6810_xy_neighbor_consensus_original_split_v1'
)


def test_xy_consensus_gate_go_hold_and_stop_boundaries() -> None:
	"""Keep all fixed original-split GO/HOLD/STOP boundaries exact."""
	budgets = ('cap25', 'cap50', 'cap100')
	assert (
		decide_xy_neighbor_consensus_original_gate(_rows(budgets), budgets=budgets)[
			'overall_status'
		]
		== 'XYCONS_ORIGINAL_GO'
	)
	assert (
		decide_xy_neighbor_consensus_original_gate(
			_rows(budgets, positive_budgets=('cap25',)), budgets=budgets
		)['overall_status']
		== 'XYCONS_ORIGINAL_HOLD'
	)
	assert (
		decide_xy_neighbor_consensus_original_gate(
			_rows(budgets, negative_budgets=('cap25', 'cap50')), budgets=budgets
		)['overall_status']
		== 'XYCONS_ORIGINAL_STOP'
	)
	assert OUTPUT_NAMES == REPORT_OUTPUT_NAMES
	assert (
		decide_xy_neighbor_consensus_original_gate(
			_rows(budgets, degraded=(3, 'f1')), budgets=budgets
		)['overall_status']
		== 'XYCONS_ORIGINAL_STOP'
	)
	assert (
		decide_xy_neighbor_consensus_original_gate(
			_rows(budgets, degraded=(5, 'iou')), budgets=budgets
		)['overall_status']
		== 'XYCONS_ORIGINAL_STOP'
	)


def test_xy_consensus_gate_requires_four_of_five_wins() -> None:
	"""Do not make a positive budget from three positive paired seeds."""
	budgets = ('cap25', 'cap50', 'cap100')
	decision = decide_xy_neighbor_consensus_original_gate(
		_rows(budgets, wins=3), budgets=budgets
	)
	assert decision['overall_status'] == 'XYCONS_ORIGINAL_HOLD'
	assert decision['six_split_follow_up'] == {
		'ready': False,
		'scientific_jobs_executed': 0,
	}


def test_xy_consensus_matrix_requires_exact_roles_and_pairing_identity() -> None:
	"""Reject an unpaired row rather than averaging it into the screen."""
	config = SimpleNamespace(
		budgets=('cap25', 'cap50', 'cap100'), subsample_seeds=(0, 1, 2, 3, 4)
	)
	roles = ('mae', 'm1_current_k6', 'mh_nocons', 'mh_xycons1_nocons')
	row = dict.fromkeys(xy_results.control.PAIR_IDENTITY_KEYS, 'shared')
	members = {
		(budget, seed, role): {'row': dict(row)}
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in roles
	}
	xy_results._expected_matrix(config, members)
	xy_results._validate_pairing(config, members)
	members[('cap25', 0, 'mh_xycons1_nocons')]['row']['decoder_seed'] = 'drift'
	with pytest.raises(ValueError, match='paired identity mismatch'):
		xy_results._validate_pairing(config, members)


def test_xy_spatial_audit_rows_keep_x_y_and_combined_evidence() -> None:
	"""Publish one unambiguous same-z spatial row for every K and axis."""
	rows = _spatial_audit_rows(_audit())
	assert len(rows) == 9
	assert [(row['head_k'], row['axis']) for row in rows[:3]] == [
		(6, 'x'),
		(6, 'y'),
		(6, 'combined'),
	]
	assert rows[0]['valid_edge_count'] == 20
	assert rows[0]['source_disagreement_count'] == 8
	assert rows[0]['output_disagreement_count'] == 4
	assert rows[0]['source_state_occupancy_json'] == '[4,5,6,7,8,9]'
	assert rows[0]['output_state_occupancy_json'] == '[5,6,7,8,9,10]'
	assert rows[0]['source_temporal_transition_count'] == 12
	assert rows[0]['output_temporal_transition_count'] == 13


def test_xy_consensus_summary_publishes_only_portable_lightweight_files(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	"""Publish audit-derived evidence and the five required screening reports."""
	workspace = tmp_path / 'workspace'
	artifact_root = workspace / 'artifacts' / 'seis_ssl_cluster'
	reports = artifact_root / 'reports'
	results_root = workspace / 'results'
	audit_path = artifact_root / 'preflight' / 'audit.json'
	audit_path.parent.mkdir(parents=True)
	audit_path.write_text(json.dumps(_audit()), encoding='utf-8')
	config = SimpleNamespace(
		reports_dir=reports,
		screening_audit=audit_path,
		base=SimpleNamespace(
			artifact_root=artifact_root,
			results_root=results_root,
			publish=SimpleNamespace(results_root=results_root),
		),
	)
	inspection = _inspection(
		artifact_root=artifact_root,
		workspace=workspace,
		audit_path=audit_path,
	)
	monkeypatch.setattr(
		xy_results,
		'inspect_f3_lithology_voxel_label_budget_xy_neighbor_consensus_results',
		lambda _config: inspection,
	)
	monkeypatch.setattr(
		xy_results,
		'_execution_git_state',
		lambda _config: {'git_sha': 'a' * 40, 'dirty': False},
	)

	result = summarize_f3_lithology_voxel_label_budget_xy_neighbor_consensus(config)
	published = results_root / _RESULTS_ROOT
	assert result['decisions'] == inspection['decisions']
	assert {path.name for path in published.iterdir()} == {
		*PUBLISHED_OUTPUT_NAMES,
		'publish_manifest.json',
	}
	assert {path.name for path in reports.iterdir()} == set(PUBLISHED_OUTPUT_NAMES)
	for name in PUBLISHED_OUTPUT_NAMES:
		assert (reports / name).read_bytes() == (published / name).read_bytes()
	for path in published.iterdir():
		if path.is_file():
			text = path.read_text(encoding='utf-8')
			assert str(workspace) not in text
			assert str(artifact_root) not in text

	spatial_rows = list(
		csv.DictReader(
			(published / AUDIT_OUTPUT_NAMES[1]).read_text(encoding='utf-8').splitlines()
		)
	)
	assert len(spatial_rows) == 9
	handoff = json.loads(
		(published / REPORT_OUTPUT_NAMES[4]).read_text(encoding='utf-8')
	)
	assert handoff['candidate_provenance']['best_checkpoint']['sha256'] == 'c' * 64
	assert handoff['screening_audit']['path'].startswith(
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
	)
	assert handoff['six_split_follow_up']['scientific_jobs_executed'] == 0

	report = validate_results_artifacts(
		published,
		required_files=tuple(
			Path(name) for name in (*PUBLISHED_OUTPUT_NAMES, 'publish_manifest.json')
		),
		local_path_policy='error',
		local_path_markers=(f'{workspace}/', f'{artifact_root}/'),
	)
	assert report.ok, report.errors


def _rows(
	budgets: tuple[str, ...],
	*,
	positive_budgets: tuple[str, ...] = ('cap25', 'cap50'),
	negative_budgets: tuple[str, ...] = (),
	degraded: tuple[int, str] | None = None,
	wins: int = 4,
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
			value = 0.0
			metric_wins = 0
			if budget in positive_budgets and metric in {'macro_f1', 'mean_iou'}:
				value = 0.1
				metric_wins = wins
			if budget in negative_budgets and metric in {'macro_f1', 'mean_iou'}:
				value = -0.1
				metric_wins = 1
			if (
				degraded is not None
				and metric == f'class_{degraded[0]}_{degraded[1]}'
				and budget != 'cap100'
			):
				value = -0.05
			rows.append(
				{
					'budget_id': budget,
					'comparison_id': 'mh_xycons1_nocons_vs_mh_nocons',
					'metric': metric,
					'mean_delta': value,
					'wins': metric_wins,
				}
			)
	return rows


def _audit() -> dict[str, object]:
	edge = {
		'valid_edge_count': 20,
		'source_disagreement_count': 8,
		'source_disagreement_fraction': 0.4,
		'output_disagreement_count': 4,
		'output_disagreement_fraction': 0.2,
		'absolute_disagreement_reduction': 0.2,
		'relative_disagreement_reduction': 0.5,
	}
	return {
		'artifact_type': 'f3_xy_neighbor_consensus_original_screening_preflight',
		'schema_version': 1,
		'status': 'PASS',
		'candidate': {
			'model_id': 'mh_xycons1_nocons',
			'model_tag': (
				'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
			),
		},
		'hard_baseline_parity': {'status': 'PASS', 'same_initial_state': True},
		'xy_spatial_smoothness': {
			'per_k': {
				str(head_k): {
					'x_edges': edge,
					'y_edges': edge,
					'combined': edge,
					'valid_token_count': 39,
					'changed_token_count': 2,
					'changed_fraction': 2 / 39,
					'source_state_occupancy': [4, 5, 6, 7, 8, 9],
					'output_state_occupancy': [5, 6, 7, 8, 9, 10],
					'empty_output_state_count': 0,
					'source_temporal_transition_count': 12,
					'output_temporal_transition_count': 13,
					'ordered_path_violations': {'source': 0, 'output': 0},
				}
				for head_k in (6, 8, 10)
			},
		},
	}


def _inspection(
	*, artifact_root: Path, workspace: Path, audit_path: Path
) -> dict[str, object]:
	decisions = decide_xy_neighbor_consensus_original_gate(
		_rows(('cap25', 'cap50', 'cap100')), budgets=('cap25', 'cap50', 'cap100')
	)

	def identity(path: Path, sha: str) -> dict[str, object]:
		return {
			'path': str(path),
			'sha256': sha,
			'byte_size': 1,
		}

	return {
		'job_metrics': (
			{
				'budget_id': 'cap25',
				'subsample_seed': 0,
				'model_role': 'mh_xycons1_nocons',
				'voxel_dataset_root': str(artifact_root / 'datasets' / 'cap25'),
				'source_config': str(workspace / 'experiments' / 'f3' / 'config.yaml'),
			},
		),
		'paired_deltas': (
			{
				'comparison_id': 'mh_xycons1_nocons_vs_mh_nocons',
				'comparison': 'mh_xycons1_nocons - mh_nocons',
			},
		),
		'summary_by_budget': (),
		'decisions': decisions,
		'screening_audit': _audit(),
		'source_identities': {
			'screening_audit': identity(audit_path, 'a' * 64),
			'candidate_run_manifest': identity(
				artifact_root / 'reports' / 'run.json', 'b' * 64
			),
			'candidate_provenance': {
				'model_id': 'mh_xycons1_nocons',
				'model_tag': (
					'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
				),
				'pretraining_handoff': identity(
					artifact_root / 'handoff.json', 'd' * 64
				),
				'best_checkpoint': identity(artifact_root / 'best.pt', 'c' * 64),
				'embeddings': identity(artifact_root / 'embedding.npy', 'e' * 64),
				'valid_tokens': identity(artifact_root / 'valid.npy', 'f' * 64),
				'embedding_metadata': identity(
					artifact_root / 'metadata.json', '0' * 64
				),
			},
			'hard_decoder_config': identity(
				workspace / 'experiments' / 'hard.yaml', '1' * 64
			),
			'reference_run_manifests': {
				'hard_multi_head': identity(artifact_root / 'hard.json', '2' * 64),
				'current_k6': identity(artifact_root / 'k6.json', '3' * 64),
				'mae': identity(artifact_root / 'mae.json', '4' * 64),
			},
			'paired_matrix_identity': {
				'roles': ('mae', 'm1_current_k6', 'mh_nocons', 'mh_xycons1_nocons'),
				'budgets': ('cap25', 'cap50', 'cap100'),
				'subsample_seeds': (0, 1, 2, 3, 4),
				'row_count': 60,
				'pair_identity_keys': ('decoder_seed',),
			},
		},
	}
