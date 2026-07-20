"""Unit contracts for the current-code K=6 control evidence helpers."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.f3 import current_k6_control as control

if TYPE_CHECKING:
	from pathlib import Path


def test_migration_gate_requires_exact_replay_evidence(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	root = tmp_path / 'migration'
	_write_json(
		root / 'performance_migration_decision.json',
		{
			'status': 'PASS_WITH_NUMERIC_DRIFT',
			'required_rerun_scope': (
				'no historical rerun; add a future current-code K=6 control'
			),
		},
	)
	_write_json(
		root / 'pseudo_target_parity.json',
		{
			'labels': {'exact': True},
			'confidence': {'exact': True, 'threshold_crossing_count': 0},
			'valid_tokens': {'exact': True},
		},
	)
	_write_json(
		root / 'hmm_parity.json',
		{
			'labels': {
				'decoded_labels_exact': True,
				'valid_token_mask_exact': True,
			},
			'centers': {'comparison': {'allclose': True}},
		},
	)
	_write_json(
		root / 'probe_parity.json',
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
		root / 'embedding_parity.json',
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
	monkeypatch.setattr(control, 'MIGRATION_ROOT', root)

	evidence = control._migration_evidence()

	assert evidence['decision']['status'] == 'PASS_WITH_NUMERIC_DRIFT'
	decision = json.loads((root / 'performance_migration_decision.json').read_text())
	decision['status'] = 'REEXTRACT_REQUIRED'
	_write_json(root / 'performance_migration_decision.json', decision)
	with pytest.raises(ValueError, match='migration status blocks control'):
		control._migration_evidence()


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
