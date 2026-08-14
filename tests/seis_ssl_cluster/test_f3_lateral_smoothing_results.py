"""Focused lightweight-review publication tests for F3 M5-LS."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import seis_ssl_cluster.f3.lateral_smoothing_results as results
from proc.seis_ssl_cluster import publish_f3_m5_lateral_smoothing_results as cli

_DIGEST = 'a' * 64


def test_review_config_is_closed_and_requires_results_output(tmp_path: Path) -> None:
	paths = _paths(tmp_path)
	_configure_sources(paths, _handoff(paths['artifact_root']))
	config = _config_mapping(paths)
	resolved = results.f3_m5_lateral_smoothing_review_config_from_mapping(config)
	assert resolved.output_dir == paths['output_dir']

	with pytest.raises(ValueError, match='unknown M5-LS review config keys'):
		results.f3_m5_lateral_smoothing_review_config_from_mapping(
			{**config, 'facies_labels': '/not/allowed.npy'}
		)

	explicit_output = paths['workspace_root'] / 'outside'
	resolved = results.f3_m5_lateral_smoothing_review_config_from_mapping(
		{**config, 'output_dir': str(explicit_output)}
	)
	assert resolved.output_dir == explicit_output


def test_review_publisher_writes_portable_target_only_artifacts(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	paths = _paths(tmp_path)
	handoff = _handoff(paths['artifact_root'])
	_configure_sources(paths, handoff)
	monkeypatch.setattr(
		results,
		'load_f3_m5_lateral_target_calibration_handoff',
		lambda _path: handoff,
	)
	config = results.f3_m5_lateral_smoothing_review_config_from_mapping(
		_config_mapping(paths)
	)

	publication = results.publish_f3_m5_lateral_smoothing_review(config)

	assert publication.calibration_status == 'M5_LS_TARGET_SELECTED'
	assert publication.selected_beta == 0.10
	assert publication.smoke_status == 'READY_NOT_RUN'
	assert publication.smoke_summary is None
	assert {path.name for path in paths['output_dir'].iterdir()} == set(
		results.OUTPUT_NAMES
	)

	candidate_rows = list(
		csv.DictReader(publication.candidate_csv.read_text(encoding='utf-8').splitlines())
	)
	assert len(candidate_rows) == 12
	head_row = next(
		row
		for row in candidate_rows
		if row['candidate'] == 'beta010' and row['k'] == '6'
	)
	assert head_row['candidate_eligible'] == 'True'
	assert head_row['changed_fraction'] == '0.2'
	assert head_row['manifest_path'].startswith(
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/'
	)

	summary = json.loads(publication.summary_json.read_text(encoding='utf-8'))
	assert summary['target_calibration']['selected_beta'] == 0.10
	assert summary['smoke']['status'] == 'READY_NOT_RUN'
	assert summary['execution_scope'] == {
		'full_pretraining': 'NOT_EXECUTED',
		'embedding_extraction': 'NOT_EXECUTED',
		'downstream_screening': 'NOT_EXECUTED',
	}
	markdown = publication.summary_markdown.read_text(encoding='utf-8')
	assert 'downstream performance conclusion' in markdown
	published_handoff = json.loads(
		publication.calibration_handoff.read_text(encoding='utf-8')
	)
	assert published_handoff['artifact_type'] == 'f3_m5_lateral_target_calibration'
	assert published_handoff['git']['head'] == 'b' * 40

	texts = [
		path.read_text(encoding='utf-8')
		for path in paths['output_dir'].iterdir()
		if path.suffix in {'.json', '.csv', '.md'}
	]
	assert all(str(paths['artifact_root']) not in text for text in texts)
	assert all(str(paths['workspace_root']) not in text for text in texts)


def test_review_publisher_compacts_strict_smoke_evidence(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	paths = _paths(tmp_path)
	handoff = _handoff(paths['artifact_root'])
	_configure_sources(paths, handoff)
	monkeypatch.setattr(
		results,
		'load_f3_m5_lateral_target_calibration_handoff',
		lambda _path: handoff,
	)
	config = results.f3_m5_lateral_smoothing_review_config_from_mapping(
		_config_mapping(paths)
	)

	publication = results.publish_f3_m5_lateral_smoothing_review(
		config,
		smoke_evidence=_raw_smoke_evidence(paths['artifact_root']),
	)

	assert publication.smoke_status == 'PASS'
	assert publication.smoke_summary is not None
	smoke = json.loads(publication.smoke_summary.read_text(encoding='utf-8'))
	assert smoke == {
		'artifact_type': 'f3_m5_lateral_smoothing_smoke_summary',
		'schema_version': 1,
		'status': 'PASS',
		'selected_beta': 0.10,
		'global_step': 2,
		'epoch': 1,
		'finite_losses': {'loss': 1.25, 'loss_consistency': 0.0},
		'gradients_finite': True,
		'target_representation': 'lateral_mean_field_hard_labels_v1',
		'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
		'checkpoint_schema_version': 4,
		'hard_multi_head_loss_path_used': True,
		'posterior_loss_path_used': False,
		'consistency_contribution': 0.0,
		'smoke_root': (
			'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/f3/m5_ls_smoke'
		),
		'hard_baseline_parity': {
			'initial_student_head': 'PASS',
			'trainability': 'PASS',
			'optimizer_groups': 'PASS',
			'smoke_root_isolated_from_full_root': 'PASS',
		},
	}
	assert {path.name for path in paths['output_dir'].iterdir()} == {
		*results.OUTPUT_NAMES,
		results.LATERAL_SMOKE_SUMMARY_JSON,
	}


def test_review_rejects_report_drift_and_smoke_for_hold(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	paths = _paths(tmp_path)
	handoff = _handoff(paths['artifact_root'])
	_configure_sources(paths, handoff, report_status='M5_LS_TARGET_HOLD')
	monkeypatch.setattr(
		results,
		'load_f3_m5_lateral_target_calibration_handoff',
		lambda _path: handoff,
	)
	config = results.f3_m5_lateral_smoothing_review_config_from_mapping(
		_config_mapping(paths)
	)
	with pytest.raises(ValueError, match='report differs'):
		results.publish_f3_m5_lateral_smoothing_review(config)

	_configure_sources(paths, handoff)
	hold = _handoff(paths['artifact_root'], status='M5_LS_TARGET_HOLD')
	_configure_sources(paths, hold)
	monkeypatch.setattr(
		results,
		'load_f3_m5_lateral_target_calibration_handoff',
		lambda _path: hold,
	)
	with pytest.raises(ValueError, match='TARGET_HOLD'):
		results.publish_f3_m5_lateral_smoothing_review(
			config,
			smoke_evidence=_raw_smoke_evidence(paths['artifact_root']),
		)


def test_review_cli_passes_explicit_smoke_evidence_path(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	paths = _paths(tmp_path)
	smoke_path = paths['artifact_root'] / 'm5_ls' / 'smoke.json'
	smoke_path.parent.mkdir(parents=True, exist_ok=True)
	smoke_path.write_text('{}', encoding='utf-8')
	captured: dict[str, object] = {}

	def resolve(config: dict[str, object]) -> object:
		captured.update(config)
		return object()

	monkeypatch.setattr(
		cli,
		'f3_m5_lateral_smoothing_review_config_from_mapping',
		resolve,
	)
	monkeypatch.setattr(
		cli,
		'publish_f3_m5_lateral_smoothing_review',
		lambda _config, **_kwargs: SimpleNamespace(
			calibration_status='M5_LS_TARGET_SELECTED',
			selected_beta=0.10,
			smoke_status='PASS',
			candidate_csv=paths['output_dir'] / 'candidates.csv',
			summary_json=paths['output_dir'] / 'summary.json',
			summary_markdown=paths['output_dir'] / 'summary.md',
			calibration_handoff=paths['output_dir'] / 'handoff.json',
			smoke_summary=paths['output_dir'] / 'smoke.json',
		),
	)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'publish_f3_m5_lateral_smoothing_results.py',
			'--artifact-root',
			str(paths['artifact_root']),
			'--workspace-root',
			str(paths['workspace_root']),
			'--calibration-handoff',
			str(paths['calibration_handoff']),
			'--calibration-report',
			str(paths['calibration_report']),
			'--output-dir',
			str(paths['output_dir']),
			'--smoke-evidence',
			str(smoke_path),
			'--dry-run',
		],
	)

	assert cli.main() == 0
	assert captured['smoke_evidence'] == str(smoke_path)


def _paths(tmp_path: Path) -> dict[str, Path]:
	workspace_root = tmp_path / 'workspace'
	artifact_root = workspace_root / 'artifacts' / 'seis_ssl_cluster'
	artifact_root.mkdir(parents=True)
	workspace_root.joinpath('reports').mkdir()
	return {
		'workspace_root': workspace_root,
		'artifact_root': artifact_root,
		'calibration_handoff': artifact_root / 'm5_ls' / 'handoff.json',
		'calibration_report': artifact_root / 'm5_ls' / 'report.json',
		'output_dir': (
			workspace_root
			/ 'reports/f3/facies_benchmark_v1/'
			'strat_hmm_multi_head_k6810_lateral_smoothing_v1'
		),
	}


def _config_mapping(paths: dict[str, Path]) -> dict[str, object]:
	return {
		'artifact_root': str(paths['artifact_root']),
		'workspace_root': str(paths['workspace_root']),
		'calibration_handoff': str(paths['calibration_handoff']),
		'calibration_report': str(paths['calibration_report']),
		'output_dir': str(paths['output_dir']),
	}


def _configure_sources(
	paths: dict[str, Path],
	handoff: dict[str, object],
	*,
	report_status: str | None = None,
) -> None:
	paths['calibration_handoff'].parent.mkdir(parents=True, exist_ok=True)
	paths['calibration_handoff'].write_text(json.dumps(handoff), encoding='utf-8')
	report = {
		key: handoff[key]
		for key in (
			'artifact_type',
			'schema_version',
			'status',
			'selection_policy',
			'candidate_betas',
			'beta_zero_parity',
			'source_hard_manifest',
			'source_posterior_manifest',
			'candidates',
			'selected_beta',
			'selected_candidate_manifest',
			'selected_manifest',
			'git',
		)
	}
	report['artifact_type'] = 'f3_m5_lateral_target_calibration_report'
	if report_status is not None:
		report['status'] = report_status
	paths['calibration_report'].write_text(json.dumps(report), encoding='utf-8')


def _handoff(
	artifact_root: Path, *, status: str = 'M5_LS_TARGET_SELECTED'
) -> dict[str, object]:
	selected = status == 'M5_LS_TARGET_SELECTED'
	candidates = {
		name: _candidate(
			artifact_root,
			name=name,
			beta=beta,
			eligible=selected and name == 'beta010',
		)
		for name, beta in (
			('beta010', 0.10),
			('beta025', 0.25),
			('beta050', 0.50),
		)
	}
	selected_reference = _reference(
		artifact_root
		/ 'pseudo_targets/f3/facies_benchmark_v1/selected/manifest.json'
	)
	return {
		'artifact_type': 'f3_m5_lateral_target_calibration',
		'schema_version': 1,
		'status': status,
		'selection_policy': 'target_only_smallest_eligible_beta_v1',
		'candidate_betas': [0.10, 0.25, 0.50],
		'beta_zero_parity': {
			'status': 'PASS',
			'pairwise_strength_ratio': 0.0,
			'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
			'heads': {str(k): _parity_head() for k in (6, 8, 10)},
		},
		'source_hard_manifest': _reference(
			artifact_root / 'pseudo_targets/f3/facies_benchmark_v1/source_hard.json'
		),
		'source_posterior_manifest': _reference(
			artifact_root
			/ 'pseudo_targets/f3/facies_benchmark_v1/source_posterior.json'
		),
		'candidates': candidates,
		'selected_beta': 0.10 if selected else None,
		'selected_candidate_manifest': (
			candidates['beta010']['manifest'] if selected else None
		),
		'selected_manifest': selected_reference if selected else None,
		'training_configs': {
			'hard_full_config': _reference(
				artifact_root / 'configs/hard_full.yaml'
			),
			'lateral_smoke_config': _reference(
				artifact_root / 'configs/lateral_smoke.yaml'
			),
			'lateral_full_config': _reference(
				artifact_root / 'configs/lateral_full.yaml'
			),
		},
		'git': {
			'head': 'b' * 40,
			'dirty_status': [],
			'git_diff_sha256': 'c' * 64,
		},
	}


def _candidate(
	artifact_root: Path, *, name: str, beta: float, eligible: bool
) -> dict[str, object]:
	return {
		'beta': beta,
		'manifest': _reference(
			artifact_root
			/ 'pseudo_targets/f3/facies_benchmark_v1'
			/ name
			/ 'multi_head_lateral_target_handoff.json'
		),
		'head_hashes': {
			str(k): {'survey': {'labels': _DIGEST}} for k in (6, 8, 10)
		},
		'smoothing': {'pairwise_strength_ratio': beta},
		'heads': {str(k): _head_summary(k) for k in (6, 8, 10)},
		'eligibility': {
			'eligible': eligible,
			'checks': {
				str(k): {
					'changed_tokens_positive': eligible,
					'affinity_weighted_disagreement_reduced': eligible,
				}
				for k in (6, 8, 10)
			},
			'reasons': [] if eligible else ['target-only check failed'],
		},
	}


def _head_summary(k: int) -> dict[str, object]:
	return {
		'resolved_scales': {
			'affinity': {'resolved_scale': 0.5},
			'emission_gap': {'resolved_scale': 1.5 + k},
		},
		'valid_token_count': 10,
		'invalid_token_count': 0,
		'changed_token_count': 2,
		'changed_fraction': 0.2,
		'changed_fraction_by_source_region': {
			'boundary_adjacent': 0.4,
			'interior': 0.2,
		},
		'state_occupancy': {'empty_state_count': 0, 'effective_k': float(k)},
		'ordered_path': {'violation_count': 0, 'max_reverse_decrease': 0},
		'source_transition_count': 12,
		'trace_paths': {'transition_count': 11},
		'xy_edge_disagreement': {
			'affinity_weighted_normalized_order': {'source': 0.4, 'lateral': 0.3},
			'affinity_quartiles': [
				{},
				{},
				{},
				{
					'edge_count': 3,
					'source_unweighted_mean': 0.4,
					'lateral_unweighted_mean': 0.3,
				},
			],
		},
	}


def _reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': _DIGEST}


def _parity_head() -> dict[str, bool]:
	return {'labels_bitwise_identical': True, 'valid_masks_exact': True}


def _raw_smoke_evidence(artifact_root: Path) -> dict[str, object]:
	return {
		'status': 'PASS',
		'selected_beta': 0.10,
		'target_representation': 'lateral_mean_field_hard_labels_v1',
		'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
		'smoke': {
			'root': str(artifact_root / 'pretraining/f3/m5_ls_smoke'),
			'latest': {
				'global_step': 2,
				'epoch': 1,
				'metrics': {'loss': 1.25, 'loss_consistency': 0.0},
			},
			'identity': {
				'schema_version': 4,
				'target_representation': 'lateral_mean_field_hard_labels_v1',
				'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
				'consistency_weight': 0.0,
			},
			'hard_multi_head_loss_path_used': True,
			'posterior_loss_path_used': False,
			'consistency_contribution': 0.0,
			'gradients_finite': True,
		},
	}
