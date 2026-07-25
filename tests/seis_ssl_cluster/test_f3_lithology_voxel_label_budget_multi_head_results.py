"""Regression contracts for the multi-head aggregate review findings."""
# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head_results as results,
)
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_results as voxel_results,
)


def test_decisions_include_the_required_current_k6_mae_status() -> None:
	decisions = results.decide_multi_head_comparisons(
		_summary_rows(), budgets=('cap25', 'cap50', 'cap100')
	)

	assert decisions['effects']['current_k6_vs_mae']['comparison_id'] == (
		'm1_current_k6_vs_mae'
	)


def test_results_load_current_k6_rows_from_configured_reference_manifest(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = SimpleNamespace(
		dataset_manifest=tmp_path / 'dataset.json',
		multi_head_target_manifest=tmp_path / 'target.json',
		original_run_manifest=tmp_path / 'original.json',
		current_k6_run_manifest=tmp_path / 'current' / 'control_job_manifest.json',
		reports_dir=tmp_path / 'reports',
		references=SimpleNamespace(historical_m1_model_id=None),
		candidates=(),
		budgets=(),
		subsample_seeds=(),
	)
	config.multi_head_target_manifest.write_text('{}', encoding='utf-8')
	current_calls: list[object] = []
	monkeypatch.setattr(
		results.multi_head,
		'load_f3_lithology_voxel_label_budget_multi_head_rows',
		lambda _config: (),
	)
	monkeypatch.setattr(results.multi_head, '_dataset_rows', lambda _config: {})
	monkeypatch.setattr(
		results.multi_head,
		'_current_k6_rows',
		lambda actual_config, _dataset_rows: current_calls.append(actual_config) or {},
	)
	monkeypatch.setattr(
		results,
		'inspect_f3_lithology_voxel_label_budget_mae_reference_run',
		lambda *_args, **_kwargs: SimpleNamespace(jobs=()),
	)
	monkeypatch.setattr(results, '_members', lambda *_args: {})
	monkeypatch.setattr(results, '_validate_pairing', lambda *_args: None)
	monkeypatch.setattr(results, '_comparisons', lambda *_args: ())
	monkeypatch.setattr(results, '_paired_metrics', lambda *_args: [])
	monkeypatch.setattr(results, '_paired_deltas', lambda *_args, **_kwargs: [])
	monkeypatch.setattr(results, '_summary', lambda *_args, **_kwargs: [])
	monkeypatch.setattr(results, '_monitored', lambda *_args, **_kwargs: [])
	monkeypatch.setattr(
		results,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': [6, 8, 10]},
	)
	monkeypatch.setattr(
		results, '_pretraining_evidence', lambda *_args, **_kwargs: ([], [])
	)
	monkeypatch.setattr(
		results, '_target_diagnostic_rows', lambda *_args, **_kwargs: []
	)
	monkeypatch.setattr(
		results, '_validate_current_k6_mae_parity', lambda *_args, **_kwargs: {}
	)
	monkeypatch.setattr(
		results, 'decide_multi_head_comparisons', lambda *_args, **_kwargs: {}
	)
	monkeypatch.setattr(results, '_json_source_identity', lambda *_args, **_kwargs: {})
	monkeypatch.setattr(
		results.multi_head, 'multi_head_run_manifest_path', lambda _config: tmp_path
	)
	monkeypatch.setattr(
		results.control,
		'load_f3_lithology_voxel_label_budget_control_rows',
		lambda *_args: pytest.fail('must not resolve control rows from config.base'),
	)

	results.inspect_f3_lithology_voxel_label_budget_multi_head_results(config)

	assert current_calls == [config]


def test_mae_reference_loader_omits_invalid_historical_m1(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	run_manifest = tmp_path / 'original.json'
	run_manifest.write_text('{}', encoding='utf-8')
	monkeypatch.setattr(
		voxel_results,
		'REQUIRED_BUDGETS',
		('cap25',),
	)
	monkeypatch.setattr(voxel_results, 'REQUIRED_SEEDS', (0,))
	monkeypatch.setattr(
		voxel_results,
		'_load_dataset_manifest',
		lambda _path: ({}, {'path': 'dataset', 'sha256': 'a' * 64}),
	)
	monkeypatch.setattr(
		voxel_results,
		'_read_json',
		lambda _path: {
			'artifact_type': voxel_results.RUN_MANIFEST_ARTIFACT_TYPE,
			'schema_version': voxel_results.SCHEMA_VERSION,
			'preregistered_contract': {
				'budgets': ['cap25'],
				'subsample_seeds': [0],
				'model_order': list(voxel_results.MODEL_ROLES),
				'epochs': 50,
				'sampling_mode': 'uniform_tiles_with_replacement',
				'steps_per_epoch': 440,
			},
			'rows': [{'model_role': 'mae'}, {'model_role': 'm1'}],
		},
	)

	def load_job(row: object, **_kwargs: object) -> object:
		if row['model_role'] == 'm1':  # type: ignore[index]
			raise ValueError('historical M1 metrics are unavailable')
		return SimpleNamespace(
			model_role='mae',
			dataset=SimpleNamespace(budget_id='cap25', subsample_seed=0),
			steps_per_epoch=440,
		)

	monkeypatch.setattr(voxel_results, '_load_job', load_job)

	reference = voxel_results.inspect_f3_lithology_voxel_label_budget_mae_reference_run(
		tmp_path / 'dataset.json', run_manifest, include_historical_m1=True
	)

	assert [job.model_role for job in reference.jobs] == ['mae']


def test_members_reject_duplicate_source_rows(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(results.control, '_identity_path', lambda value, _label: value)
	monkeypatch.setattr(
		results,
		'load_f3_lithology_voxel_label_budget_evaluation_metrics',
		lambda **_kwargs: {'macro_f1': 0.5},
	)
	row = {
		'budget_id': 'cap25',
		'subsample_seed': 0,
		'model_role': 'mh_nocons',
		'model_tag': 'candidate',
		'evaluation_metrics': 'metrics.json',
		'evaluation_boundary_metrics': 'boundary.json',
		'evaluation_boundary_region_metrics': 'region.json',
	}
	config = SimpleNamespace(budgets=('cap25',), subsample_seeds=(0,))

	with pytest.raises(ValueError, match='duplicate multi-head comparison member'):
		results._members(
			config,
			[row, dict(row)],
			(),
			SimpleNamespace(jobs=()),
		)


def test_missing_historical_m1_omits_only_optional_comparisons(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(results.control, '_identity_path', lambda value, _label: value)
	monkeypatch.setattr(
		results,
		'load_f3_lithology_voxel_label_budget_evaluation_metrics',
		lambda **_kwargs: {'macro_f1': 0.5},
	)
	config = SimpleNamespace(budgets=('cap25',), subsample_seeds=(0,))
	rows = [
		{
			'budget_id': 'cap25',
			'subsample_seed': 0,
			'model_role': role,
			'model_tag': role,
			'evaluation_metrics': 'metrics.json',
			'evaluation_boundary_metrics': 'boundary.json',
			'evaluation_boundary_region_metrics': 'region.json',
		}
		for role in ('m1_current_k6', 'mh_nocons', 'mh_cons010')
	]
	mae = SimpleNamespace(
		model_role='mae',
		model_tag='mae',
		dataset=SimpleNamespace(budget_id='cap25', subsample_seed=0),
		evaluation=SimpleNamespace(metrics={'macro_f1': 0.4}),
	)
	monkeypatch.setattr(
		results.control,
		'_reference_member_row',
		lambda _job: {},
	)

	members = results._members(config, rows, (), SimpleNamespace(jobs=(mae,)))

	assert not results._has_historical_members(members)
	assert results._comparisons(members) == results.REQUIRED_COMPARISONS


def test_non_pairing_historical_m1_is_omitted() -> None:
	config = SimpleNamespace(budgets=('cap25',), subsample_seeds=(0,))
	pair_row = dict.fromkeys(results.control.PAIR_IDENTITY_KEYS, 'same')
	members = {
		('cap25', 0, role): {'row': pair_row}
		for role in ('mae', 'm1_current_k6', 'mh_nocons', 'mh_cons010')
	}
	historical = {
		('cap25', 0, 'm1'): {'row': {**pair_row, 'metric_schema_sha256': 'different'}}
	}

	assert not results._historical_members_are_paired(config, members, historical)


def test_pretraining_pair_requires_initialization_parity() -> None:
	left = _checkpoint_evidence(variant='nocons', consistency_weight=0.0)
	right = _checkpoint_evidence(variant='cons010', consistency_weight=0.1)

	results._validate_pretraining_pair(left, right)
	right['identity']['initial_head_state_sha256'] = 'c' * 64

	with pytest.raises(ValueError, match='initial-state parity'):
		results._validate_pretraining_pair(left, right)


def test_pretraining_requires_the_configured_target_manifest(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	target = tmp_path / 'multi_head_target_manifest.json'
	target.write_text('{}', encoding='utf-8')
	candidates = []
	for role, tag, weight in (
		(
			'mh_nocons',
			'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
			0.0,
		),
		(
			'mh_cons010',
			'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
			0.1,
		),
	):
		handoff = tmp_path / f'{role}.json'
		handoff.write_text(
			json.dumps(
				{
					'artifact_type': 'f3_multi_head_pretraining_handoff',
					'status': 'PASS',
					'stratigraphy_pretext': {
						'head_ks': [6, 8, 10],
						'head_spec': 'multi_resolution_ordered_prototypes_v1',
						'target_manifest_sha256': file_sha256(target),
						'consistency_policy': 'fixed',
						'consistency_weight': weight,
					},
				}
			),
			encoding='utf-8',
		)
		candidates.append(
			SimpleNamespace(
				model_id=role,
				model_tag=tag,
				pretraining_handoff=handoff,
			)
		)
	monkeypatch.setattr(
		results,
		'_pretraining_checkpoint',
		lambda _config, candidate, _handoff: _checkpoint_evidence(
			variant='nocons' if candidate.model_id == 'mh_nocons' else 'cons010',
			consistency_weight=0.0 if candidate.model_id == 'mh_nocons' else 0.1,
		),
	)

	with pytest.raises(ValueError, match='manifest is missing fields'):
		results._pretraining_evidence(
			SimpleNamespace(multi_head_target_manifest=target, candidates=candidates)
		)


def test_current_k6_mae_parity_rejects_published_mismatch(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	manifest = tmp_path / 'control_job_manifest.json'
	manifest.write_text('{}', encoding='utf-8')
	metric = SimpleNamespace(name='macro_f1')
	monkeypatch.setattr(results, 'METRIC_SPECS', (metric,))
	delta = {
		'budget_id': 'cap25',
		'subsample_seed': 0,
		'comparison_id': 'm1_current_k6_vs_mae',
		'macro_f1': 0.25,
	}
	summary = {
		'budget_id': 'cap25',
		'comparison_id': 'm1_current_k6_vs_mae',
		'metric': 'macro_f1',
		'paired_seed_count': 5,
		'mean_delta': 0.25,
		'median_delta': 0.25,
		'sample_standard_deviation': 0.0,
		'min_delta': 0.25,
		'max_delta': 0.25,
		'worst_seed': 0,
		'worst_seed_delta': 0.25,
		'wins': 5,
		'losses': 0,
		'ties': 0,
	}

	def write_control_summary(
		paired_deltas: list[object], summary_by_budget: list[object]
	) -> None:
		(tmp_path / 'current_k6_control_summary.json').write_text(
			json.dumps(
				{
					'artifact_type': 'f3_current_k6_control_summary',
					'schema_version': 1,
					'paired_deltas': paired_deltas,
					'summary_by_budget': summary_by_budget,
				}
			),
			encoding='utf-8',
		)

	write_control_summary([delta], [summary])
	config = SimpleNamespace(current_k6_run_manifest=manifest)

	assert (
		results._validate_current_k6_mae_parity(
			config, paired_deltas=(delta,), summary_by_budget=(summary,)
		)['status']
		== 'PASS'
	)
	write_control_summary([delta, dict(delta)], [summary])
	with pytest.raises(ValueError, match='duplicate rows'):
		results._validate_current_k6_mae_parity(
			config, paired_deltas=(delta,), summary_by_budget=(summary,)
		)
	write_control_summary([delta], [summary, dict(summary)])
	with pytest.raises(ValueError, match='duplicate rows'):
		results._validate_current_k6_mae_parity(
			config, paired_deltas=(delta,), summary_by_budget=(summary,)
		)
	write_control_summary([delta], [summary])
	delta['macro_f1'] = 0.26
	with pytest.raises(ValueError, match='paired delta'):
		results._validate_current_k6_mae_parity(
			config, paired_deltas=(delta,), summary_by_budget=(summary,)
		)


def test_json_source_identity_records_schema_tag_and_scientific_identity(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'source.json'
	path.write_text(
		json.dumps(
			{
				'artifact_type': 'source_type',
				'schema_version': 7,
				'model_tag': 'model_tag',
				'scientific_identity': {'condition': 'fixed'},
			}
		),
		encoding='utf-8',
	)

	identity = results._json_source_identity(path)

	assert identity['schema'] == {
		'artifact_type': 'source_type',
		'schema_version': 7,
	}
	assert identity['model_tag'] == 'model_tag'
	assert identity['scientific_identity'] == {'condition': 'fixed'}


def test_pretraining_checkpoint_requires_embedding_binding_and_configured_target(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	checkpoint = tmp_path / 'pretraining' / 'best.pt'
	checkpoint.parent.mkdir(parents=True)
	checkpoint.write_bytes(b'checkpoint')
	(checkpoint.parent / 'latest.pt').write_bytes(b'latest')
	target_manifest = tmp_path / 'multi_head_target_manifest.json'
	target_manifest.write_text('{}', encoding='utf-8')
	candidate = SimpleNamespace(
		model_id='mh_nocons',
		model_tag='strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		embeddings_dir=tmp_path / 'embeddings',
	)
	metadata_path = output_paths(candidate.embeddings_dir, 'f3').metadata
	metadata_path.parent.mkdir(parents=True)
	files = output_paths(candidate.embeddings_dir, 'f3')
	np.save(files.embeddings, np.ones((1, 1, 1, 2), dtype=np.float16))
	np.save(files.valid_tokens, np.ones((1, 1, 1), dtype=bool))
	metadata_path.write_text(
		json.dumps(
			{
				'checkpoint_path': str(checkpoint),
				'checkpoint_sha256': file_sha256(checkpoint),
			}
		),
		encoding='utf-8',
	)
	payload = {
		'stratigraphy_checkpoint': {
			'model_tag': candidate.model_tag,
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_manifest': {
				'path': str(target_manifest),
				'sha256': file_sha256(target_manifest),
			},
			'consistency_weight': 0.0,
		},
		'stratigraphy_config': _checkpoint_evidence(
			variant='nocons', consistency_weight=0.0
		)['config'],
	}
	monkeypatch.setattr(results.torch, 'load', lambda *_args, **_kwargs: payload)
	monkeypatch.setattr(
		results, 'validate_stratigraphy_checkpoint_payload', lambda _payload: None
	)
	handoff = {
		'checkpoint': {'path': str(checkpoint), 'sha256': file_sha256(checkpoint)},
		'embedding_metadata_sha256': file_sha256(metadata_path),
	}

	assert (
		results._pretraining_checkpoint(
			SimpleNamespace(
				dataset={'name': 'f3'}, multi_head_target_manifest=target_manifest
			),
			candidate,
			handoff,
		)['identity']
		== payload['stratigraphy_checkpoint']
	)
	payload['stratigraphy_checkpoint']['target_manifest']['sha256'] = '0' * 64
	with pytest.raises(ValueError, match='target manifest does not match configured'):
		results._pretraining_checkpoint(
			SimpleNamespace(
				dataset={'name': 'f3'}, multi_head_target_manifest=target_manifest
			),
			candidate,
			handoff,
		)
	payload['stratigraphy_checkpoint']['target_manifest']['sha256'] = file_sha256(
		target_manifest
	)
	metadata_path.write_text('{}', encoding='utf-8')
	with pytest.raises(ValueError, match='embedding metadata does not bind'):
		results._validate_embedding_best_binding(
			SimpleNamespace(dataset={'name': 'f3'}), candidate, checkpoint, handoff
		)


def test_pretraining_checkpoint_evidence_allows_pre_extraction_validation(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	target_manifest = tmp_path / 'multi_head_target_manifest.json'
	target_manifest.write_text('{}', encoding='utf-8')
	candidates = []
	for role, tag, weight in (
		(
			'mh_nocons',
			'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
			0.0,
		),
		(
			'mh_cons010',
			'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
			0.1,
		),
	):
		handoff = tmp_path / f'{role}.json'
		handoff.write_text(
			json.dumps(
				_complete_handoff_payload(
					model_tag=tag,
					variant='nocons' if role == 'mh_nocons' else 'cons010',
					consistency_weight=weight,
					target_manifest_sha256=file_sha256(target_manifest),
				)
			),
			encoding='utf-8',
		)
		candidates.append(
			SimpleNamespace(
				model_id=role,
				model_tag=tag,
				embeddings_dir=tmp_path / role / 'embeddings',
				pretraining_handoff=handoff,
			)
		)
	monkeypatch.setattr(
		results,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': [6, 8, 10]},
	)
	monkeypatch.setattr(
		results,
		'_pretraining_checkpoint',
		lambda _config, candidate, _handoff: _checkpoint_evidence(
			variant='nocons' if candidate.model_id == 'mh_nocons' else 'cons010',
			consistency_weight=0.0 if candidate.model_id == 'mh_nocons' else 0.1,
		),
	)

	rows, diagnostics = results._pretraining_evidence(
		SimpleNamespace(
			multi_head_target_manifest=target_manifest, candidates=candidates
		),
		require_embeddings=False,
	)

	assert rows == []
	assert [row['model_role'] for row in diagnostics] == [
		'mh_nocons',
		'mh_cons010',
	]


def _complete_handoff_payload(
	*,
	model_tag: str,
	variant: str,
	consistency_weight: float,
	target_manifest_sha256: str,
) -> dict[str, object]:
	return {
		'artifact_type': 'f3_multi_head_pretraining_handoff',
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': model_tag,
		'variant': variant,
		'checkpoint': {
			'path': '/artifact/best.pt',
			'sha256': 'a' * 64,
			'latest_path': '/artifact/latest.pt',
			'latest_sha256': 'b' * 64,
			'best_epoch': 25,
			'best_global_step': 25600,
			'selection_metric': 'metrics.loss',
			'selection_history_schema_version': 1,
			'selection_history_event_count': 1,
			'selected_checkpoint_kind': 'epoch',
			'selected_epoch': 25,
			'selected_global_step': 25600,
			'selected_loss': 0.1,
			'selection_history_sha256': '1' * 64,
		},
		'embedding': {
			'root': '/artifact/overlap_x16',
			'metadata_path': '/artifact/metadata.json',
			'metadata_sha256': 'c' * 64,
			'embeddings_sha256': 'd' * 64,
			'valid_tokens_sha256': 'e' * 64,
		},
		'embedding_metadata_sha256': 'c' * 64,
		'stratigraphy_pretext': {
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_manifest_path': '/artifact/multi_head_target_manifest.json',
			'target_manifest_sha256': target_manifest_sha256,
			'per_head_target_sha256': {
				str(head_k): {
					'f3': dict.fromkeys(
						('labels', 'confidence', 'valid_tokens', 'metadata'),
						f'{head_k:x}' * 64,
					)
				}
				for head_k in (6, 8, 10)
			},
			'consistency_policy': 'normalized_order_smooth_l1_v1',
			'consistency_weight': consistency_weight,
			'consistency_beta': 0.1,
			'scientific_identity_sha256': 'f' * 64,
			'initial_student_state_sha256': '1' * 64,
			'initial_head_state_sha256': '2' * 64,
		},
	}


def test_pretraining_summary_row_contains_required_training_diagnostics(
	tmp_path: Path,
) -> None:
	candidate = SimpleNamespace(
		model_id='mh_nocons',
		model_tag='strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		pretraining_handoff=tmp_path / 'handoff.json',
	)
	candidate.pretraining_handoff.write_text('{}', encoding='utf-8')
	metrics = {
		'loss': 1.0,
		**{
			f'{metric}_k{k}': 0.1
			for k in (6, 8, 10)
			for metric in (
				'loss_prototype',
				'loss_usage',
				'target_usage_entropy',
				'prototype_usage_entropy',
			)
		},
		'loss_consistency_k6_k8': 0.1,
		'loss_consistency_k6_k10': 0.1,
		'loss_consistency_k8_k10': 0.1,
	}
	checkpoint = {
		'identity': {
			'initial_student_state_sha256': 'a' * 64,
			'initial_head_state_sha256': 'b' * 64,
			'optimizer_group_identity': [
				{'name': 'head', 'lr': 3.0e-4, 'parameter_names': ['head.k6']},
				{
					'name': 'encoder',
					'lr': 1.0e-5,
					'parameter_names': ['student.encoder.blocks.7'],
				},
			],
		},
		'config': {
			'student': {'unfreeze_top_blocks': 1},
			'train': {'lr': 3.0e-4, 'encoder_lr': 1.0e-5},
		},
		'best': {'epoch': 24, 'global_step': 25000, 'metrics': metrics},
		'latest': {
			'epoch': 25,
			'global_step': 25600,
			'metrics': metrics,
			'trainability_summary': {
				'trainable_parameter_count': 1,
				'frozen_parameter_count': 1,
				'trainable_names': ['encoder.blocks.7.weight'],
			},
			'optimizer_state_dict': {
				'param_groups': [
					{'name': 'head', 'lr': 3.0e-4, 'params': [0]},
					{'name': 'encoder', 'lr': 1.0e-5, 'params': [1]},
				]
			},
		},
		'embedding': {
			'metadata': {'sha256': 'c' * 64},
			'valid_tokens': {'sha256': 'd' * 64},
			'shape': [76, 113, 32, 384],
			'dtype': 'float16',
			'valid_token_count': 237225,
			'nonfinite_valid_embedding_count': 0,
		},
	}

	row = results._pretraining_summary_row(
		candidate=candidate,
		handoff={'checkpoint': {'sha256': 'e' * 64}},
		stratigraphy={
			'target_manifest_sha256': 'f' * 64,
			'consistency_weight': 0.0,
			'scientific_identity_sha256': 'g' * 64,
		},
		checkpoint=checkpoint,
	)

	assert row['best_epoch'] == 24
	assert row['latest_global_step'] == 25600
	assert row['latest_loss_prototype_k6'] == 0.1
	assert row['latest_loss_consistency_k6_k8'] == 0.1
	assert row['freeze_contract_pass'] is True
	assert row['optimizer_contract_pass'] is True
	assert row['embedding_nonfinite_valid_embedding_count'] == 0


def test_mandatory_handoff_is_in_the_publish_set() -> None:
	assert 'multi_head_experiment_handoff.md' in results.OUTPUT_NAMES
	assert 'No confirmatory run is authorized.' in results._handoff(
		{'overall_status': 'M4_MH_HOLD', 'selected_candidate': None}
	)


def test_target_diagnostics_preserve_head_cross_head_and_k6_evidence() -> None:
	target_manifest = {
		'head_ks': [6, 8, 10],
		'heads': {
			str(k): {
				'diagnostics': {'per_survey': {'f3': {'valid_token_count': 100 + k}}}
			}
			for k in (6, 8, 10)
		},
		'cross_head_diagnostics': {
			'k6_k8': {
				'mae': 0.1,
				'correlation': 0.9,
				'rank_order_disagreement': 0.2,
			},
			'k6_k10': {
				'mae': 0.2,
				'correlation': 0.8,
				'rank_order_disagreement': 0.3,
			},
			'k8_k10': {
				'mae': 0.3,
				'correlation': 0.7,
				'rank_order_disagreement': 0.4,
			},
		},
		'k6_replay_parity': {'exact': True, 'checks': {'f3.decoded_labels': True}},
	}
	candidate_binding = results._target_diagnostic_row(
		record_type='candidate_binding',
		model_role='mh_nocons',
		model_tag='candidate',
		target_manifest_sha256='a' * 64,
		status='PASS',
		diagnostics={'head_ks': [6, 8, 10]},
	)

	rows = results._target_diagnostic_rows(
		target_manifest,
		candidate_bindings=(candidate_binding,),
		target_manifest_sha256='a' * 64,
	)

	assert all(tuple(row) == results.TARGET_DIAGNOSTIC_FIELDS for row in rows)
	assert [row['record_type'] for row in rows] == [
		'candidate_binding',
		'per_head_diagnostic',
		'per_head_diagnostic',
		'per_head_diagnostic',
		'cross_head_diagnostic',
		'cross_head_diagnostic',
		'cross_head_diagnostic',
		'k6_replay_parity',
	]
	per_head = next(row for row in rows if row['head_k'] == 8)
	assert json.loads(str(per_head['diagnostics_json'])) == {'valid_token_count': 108}
	cross_head = next(row for row in rows if row['head_pair'] == 'k6_k8')
	assert cross_head['mae'] == 0.1
	assert json.loads(str(cross_head['diagnostics_json']))['correlation'] == 0.9
	parity = rows[-1]
	assert parity['exact'] is True
	assert json.loads(str(parity['diagnostics_json']))['checks'] == {
		'f3.decoded_labels': True
	}


def test_multi_head_publish_retains_tracked_historical_status_files_in_manifest(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _publish_config(tmp_path)
	monkeypatch.setattr(
		results,
		'inspect_f3_lithology_voxel_label_budget_multi_head_results',
		lambda _config: _publishable_inspection(),
	)
	publish_dir = results._publish_dir(config)
	initial_bytes = _seed_historical_multi_head_status_files(publish_dir)

	publication = results.summarize_f3_lithology_voxel_label_budget_multi_head(
		config
	)
	assert publication.publish_manifest is not None
	manifest = publication.publish_manifest
	assert {
		path.relative_to(publish_dir).as_posix()
		for path in publish_dir.rglob('*')
		if path.is_file()
	} == results._publish_target_names() | set(
		results.HISTORICAL_STATUS_FILE_IDENTITIES
	)
	assert len(manifest.items) == len(results.OUTPUT_NAMES) + len(
		results.HISTORICAL_STATUS_FILE_IDENTITIES
	)
	items_by_target = {
		item.target.relative_to(publish_dir.resolve()).as_posix(): item
		for item in manifest.items
	}
	payload = json.loads(manifest.manifest_path.read_text(encoding='utf-8'))
	payload_items = {item['target']: item for item in payload['items']}
	for name, (size_bytes, sha256) in (
		results.HISTORICAL_STATUS_FILE_IDENTITIES.items()
	):
		path = (publish_dir / name).resolve()
		assert path.read_bytes() == initial_bytes[name]
		assert items_by_target[name].source == path
		assert items_by_target[name].target == path
		assert items_by_target[name].size_bytes == size_bytes
		assert items_by_target[name].sha256 == sha256
		assert payload_items[name]['source'] == str(path)
		assert payload_items[name]['size_bytes'] == size_bytes
		assert payload_items[name]['sha256'] == sha256


def test_multi_head_publish_rejects_invalid_historical_status_files(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(
		results,
		'inspect_f3_lithology_voxel_label_budget_multi_head_results',
		lambda _config: _publishable_inspection(),
	)

	partial_config = _publish_config(tmp_path / 'partial')
	partial_dir = results._publish_dir(partial_config)
	_seed_historical_multi_head_status_files(partial_dir)
	(partial_dir / next(iter(results.HISTORICAL_STATUS_FILE_IDENTITIES))).unlink()
	with pytest.raises(FileExistsError, match='partial historical status set'):
		results.summarize_f3_lithology_voxel_label_budget_multi_head(partial_config)

	tampered_config = _publish_config(tmp_path / 'tampered')
	tampered_dir = results._publish_dir(tampered_config)
	_seed_historical_multi_head_status_files(tampered_dir)
	tampered_path = tampered_dir / next(iter(results.HISTORICAL_STATUS_FILE_IDENTITIES))
	tampered_path.write_bytes(b'tampered')
	with pytest.raises(FileExistsError, match='tracked byte identity'):
		results.summarize_f3_lithology_voxel_label_budget_multi_head(tampered_config)

	unknown_config = _publish_config(tmp_path / 'unknown')
	unknown_dir = results._publish_dir(unknown_config)
	_seed_historical_multi_head_status_files(unknown_dir)
	(unknown_dir / 'unexpected.md').write_text('unexpected', encoding='utf-8')
	with pytest.raises(FileExistsError, match='unexpected file set'):
		results.summarize_f3_lithology_voxel_label_budget_multi_head(unknown_config)


def test_multi_head_publish_rejects_stale_and_missing_files(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _publish_config(tmp_path)
	inspection = _publishable_inspection()
	monkeypatch.setattr(
		results,
		'inspect_f3_lithology_voxel_label_budget_multi_head_results',
		lambda _config: inspection,
	)
	publish_dir = results._publish_dir(config)
	publish_dir.mkdir(parents=True)
	(publish_dir / 'stale_embeddings.npy').write_bytes(b'raw')

	with pytest.raises(FileExistsError, match='unexpected file set'):
		results.summarize_f3_lithology_voxel_label_budget_multi_head(config)

	(publish_dir / 'stale_embeddings.npy').unlink()
	publication = results.summarize_f3_lithology_voxel_label_budget_multi_head(config)
	assert {
		path.relative_to(publish_dir).as_posix()
		for path in publish_dir.rglob('*')
		if path.is_file()
	} == results._publish_target_names()

	(publish_dir / results.OUTPUT_NAMES[0]).unlink()
	with pytest.raises(FileExistsError, match='missing'):
		results.summarize_f3_lithology_voxel_label_budget_multi_head(config)

	assert publication.publish_manifest is not None


def test_multi_head_publish_detects_manifest_and_target_hash_tampering(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _publish_config(tmp_path)
	monkeypatch.setattr(
		results,
		'inspect_f3_lithology_voxel_label_budget_multi_head_results',
		lambda _config: _publishable_inspection(),
	)
	publication = results.summarize_f3_lithology_voxel_label_budget_multi_head(config)
	assert publication.publish_manifest is not None
	manifest = publication.publish_manifest
	publish_dir = results._publish_dir(config)
	payload = json.loads(manifest.manifest_path.read_text(encoding='utf-8'))
	payload['items'][0]['sha256'] = '0' * 64
	manifest.manifest_path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='manifest SHA-256'):
		results._validate_published_multi_head_tree(publish_dir, manifest)

	results.summarize_f3_lithology_voxel_label_budget_multi_head(config)
	target = publish_dir / results.OUTPUT_NAMES[0]
	target.write_bytes(b'x' * target.stat().st_size)
	with pytest.raises(ValueError, match='target SHA-256'):
		results._validate_published_multi_head_tree(publish_dir, manifest)


def _publish_config(tmp_path: Path) -> SimpleNamespace:
	return SimpleNamespace(
		reports_dir=tmp_path / 'artifacts' / 'reports',
		results_root=tmp_path / 'results',
		candidates=(),
		budgets=(),
		subsample_seeds=(),
	)


def _seed_historical_multi_head_status_files(publish_dir: Path) -> dict[str, bytes]:
	"""Copy immutable tracked status records into an isolated publish root."""
	source_dir = (
		Path(__file__).resolve().parents[2]
		/ 'results/f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_v1'
	)
	publish_dir.mkdir(parents=True, exist_ok=True)
	initial_bytes = {}
	for name in results.HISTORICAL_STATUS_FILE_IDENTITIES:
		content = (source_dir / name).read_bytes()
		(publish_dir / name).write_bytes(content)
		initial_bytes[name] = content
	return initial_bytes


def _publishable_inspection() -> results.F3VoxelLabelBudgetMultiHeadResultsInspection:
	row = {'value': 'test'}
	target_diagnostic = results._target_diagnostic_row(
		record_type='candidate_binding',
		model_role='mh_nocons',
		model_tag='candidate',
		target_manifest_sha256='a' * 64,
		status='PASS',
		diagnostics={'head_ks': [6, 8, 10]},
	)
	return results.F3VoxelLabelBudgetMultiHeadResultsInspection(
		job_metrics=(row,),
		paired_metrics=(row,),
		paired_deltas=(row,),
		summary_by_budget=(row,),
		monitored_class_summary=(row,),
		pretraining_summary=(row,),
		target_diagnostics=(target_diagnostic,),
		decisions={
			'overall_status': 'M4_MH_HOLD',
			'selected_candidate': None,
			'effects': {},
		},
		source_identities={},
	)


def _summary_rows() -> list[dict[str, object]]:
	rows = []
	for budget in ('cap25', 'cap50', 'cap100'):
		for candidate, baseline in results.COMPARISONS:
			comparison_id = f'{candidate}_vs_{baseline}'
			rows.extend(
				{
					'budget_id': budget,
					'comparison_id': comparison_id,
					'metric': metric,
					'mean_delta': 0.01,
					'wins': 5,
				}
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
				)
			)
	return rows


def _checkpoint_evidence(
	*, variant: str, consistency_weight: float
) -> dict[str, object]:
	model_tag = {
		'nocons': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		'cons010': 'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
	}[variant]
	config = {
		'loss': {'consistency_weight': consistency_weight},
		'identity': {
			'model_tag': model_tag,
			'scientific_identity': {
				'variant': variant,
				'consistency_weight': consistency_weight,
			},
		},
		'paths': {'output_root': f'/artifacts/{variant}'},
	}
	return {
		'identity': {
			'teacher_checkpoint_sha256': 'a' * 64,
			'student_init_checkpoint_sha256': 'a' * 64,
			'initial_student_state_sha256': 'b' * 64,
			'initial_head_state_sha256': 'b' * 64,
		},
		'config': config,
	}
