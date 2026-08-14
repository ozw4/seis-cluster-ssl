"""Unit contracts for the current-code K=6 control evidence helpers."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.f3 import current_k6_control as control


def test_migration_gate_requires_exact_replay_evidence(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	root = tmp_path / 'migration'
	_write_json(
		root / 'reports' / 'performance_migration_decision.json',
		{
			'status': 'PASS_WITH_NUMERIC_DRIFT',
			'required_rerun_scope': (
				'no historical rerun; add a future current-code K=6 control'
			),
		},
	)
	_write_json(
		root / 'pseudo_targets' / 'pseudo_target_parity.json',
		{
			'labels': {'exact': True},
			'confidence': {'exact': True, 'threshold_crossing_count': 0},
			'valid_tokens': {'exact': True},
		},
	)
	_write_json(
		root / 'clustering' / 'hmm_parity.json',
		{
			'labels': {
				'decoded_labels_exact': True,
				'valid_token_mask_exact': True,
			},
			'centers': {'comparison': {'allclose': True}},
		},
	)
	_write_json(
		root / 'probe_parity' / 'probe_parity.json',
		{
			'parity': {
				'linear_balanced_v1': {
					'prediction_exact': True,
					'confusion_matrix_exact': True,
					'primary_metrics_exact': True,
					'true_labels_exact': True,
					'validation_coordinates_exact': True,
				}
			},
		},
	)
	_write_json(
		root / 'embedding_parity' / 'embedding_parity.json',
		{
			'comparisons': {
				'B_current_cache_off_vs_C_current_memmap_cache': {
					'status': 'EXACT',
					'embedding_array_equal': True,
					'valid_token_mask_exact': True,
				}
			},
		},
	)
	monkeypatch.setattr(control, 'MIGRATION_ARTIFACT_ROOT', root)

	evidence = control._migration_evidence()

	assert evidence['decision']['status'] == 'PASS_WITH_NUMERIC_DRIFT'
	decision_path = root / 'reports' / 'performance_migration_decision.json'
	decision = json.loads(decision_path.read_text())
	decision['status'] = 'REEXTRACT_REQUIRED'
	_write_json(decision_path, decision)
	with pytest.raises(ValueError, match='migration status blocks control'):
		control._migration_evidence()


def test_fixed_control_inputs_are_artifact_side() -> None:
	tracked_reports = (Path.cwd() / 'reports').resolve()
	paths = (
		control.MIGRATION_ARTIFACT_ROOT,
		control.HISTORICAL_TOKEN_METRICS,
		control.MAE_TOKEN_METRICS,
	)

	assert all(
		path.resolve(strict=False).is_relative_to(
			(Path.cwd() / 'artifacts').resolve()
		)
		for path in paths
	)
	assert all(
		not path.resolve(strict=False).is_relative_to(tracked_reports)
		for path in paths
	)


def test_checkpoint_check_rejects_failed_structured_contracts() -> None:
	checks = {
		'latest': True,
		'best': True,
		'provenance': True,
		'freeze_contract': {'pass': True},
		'optimizer': {'pass': True},
		'resolved_config_exists': True,
		'run_metadata_exists': True,
		'historical_comparison': {'current_best_loss': 0.1},
	}

	assert control._checkpoint_checks_pass(checks)
	checks['freeze_contract'] = {'pass': False}
	assert not control._checkpoint_checks_pass(checks)
	checks['freeze_contract'] = {'pass': True}
	checks['optimizer'] = {'pass': False}
	assert not control._checkpoint_checks_pass(checks)


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload), encoding='utf-8')
