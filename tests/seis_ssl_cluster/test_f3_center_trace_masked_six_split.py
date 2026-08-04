"""Contracts for the center-trace masked six-split start audit."""
# ruff: noqa: CPY001, SLF001, TC003

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_center_trace_masked_split as six_split_config,
)
from seis_ssl_cluster.f3 import center_trace_masked_six_split_audit as audit


def test_strict_config_freezes_matrix_and_rejects_unknown_keys(
	tmp_path: Path,
) -> None:
	raw = _config_mapping(tmp_path)
	config = six_split_config.config_from_mapping(raw)
	assert config.primary_model_roles == ('mh_ctmask010_nocons', 'mh_nocons')
	assert config.primary_matrix_row_count == 36
	assert config.future_candidate_jobs == 18
	assert config.future_new_baseline_jobs == 6
	assert config.historical_baseline_rows == 12
	assert config.future_new_scientific_jobs == 24

	unknown = copy.deepcopy(raw)
	unknown['matrix']['unexpected'] = True
	with pytest.raises(ValueError, match='matrix key'):
		six_split_config.config_from_mapping(unknown)

	wrong_budget = copy.deepcopy(raw)
	wrong_budget['matrix']['budgets'] = ['cap25', 'cap50']
	with pytest.raises(ValueError, match='budgets'):
		six_split_config.config_from_mapping(wrong_budget)


def test_config_rejects_model_tag_and_source_identity_drift(tmp_path: Path) -> None:
	raw = _config_mapping(tmp_path)
	wrong_tag = copy.deepcopy(raw)
	wrong_tag['matrix']['candidate']['model_tag'] = 'wrong'
	with pytest.raises(ValueError, match='candidate identity'):
		six_split_config.config_from_mapping(wrong_tag)

	wrong_source = copy.deepcopy(raw)
	wrong_source['inputs']['source_identities']['class_info']['sha256'] = 'G' * 64
	with pytest.raises(ValueError, match='lowercase SHA-256'):
		six_split_config.config_from_mapping(wrong_source)


def test_dry_run_does_not_write_and_only_missing_is_byte_and_mtime_stable(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _audit_config(tmp_path)
	payload = _audit_payload(config)
	monkeypatch.setattr(audit, '_audit_payload', lambda *_args, **_kwargs: payload)
	monkeypatch.setattr(audit, '_git_provenance', lambda: payload['git'])

	dry_run = audit.audit_f3_center_trace_masked_six_split(config, dry_run=True)
	assert dry_run.action == 'DRY_RUN'
	assert not config.audit_output_path.exists()

	first = audit.audit_f3_center_trace_masked_six_split(config)
	before = config.audit_output_path.read_bytes()
	mtime = config.audit_output_path.stat().st_mtime_ns
	second = audit.audit_f3_center_trace_masked_six_split(config, only_missing=True)
	assert first.action == 'WRITTEN'
	assert second.action == 'REUSE_COMPLETED'
	assert config.audit_output_path.read_bytes() == before
	assert config.audit_output_path.stat().st_mtime_ns == mtime


def test_stale_audit_is_not_silently_reused_and_quarantine_is_explicit(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _audit_config(tmp_path)
	payload = _audit_payload(config)
	monkeypatch.setattr(audit, '_audit_payload', lambda *_args, **_kwargs: payload)
	monkeypatch.setattr(audit, '_git_provenance', lambda: payload['git'])
	config.audit_output_path.parent.mkdir(parents=True)
	config.audit_output_path.write_text(
		json.dumps({**payload, 'status': 'HOLD'}), encoding='utf-8'
	)

	with pytest.raises(ValueError, match='stale or invalid'):
		audit.audit_f3_center_trace_masked_six_split(config, only_missing=True)
	result = audit.audit_f3_center_trace_masked_six_split(
		config, only_missing=True, quarantine_invalid=True
	)
	assert result.action == 'WRITTEN'
	assert result.quarantine_path is not None
	assert result.quarantine_path.is_file()
	assert json.loads(config.audit_output_path.read_text()) == payload


def test_six_split_counter_rejects_existing_execution(tmp_path: Path) -> None:
	config = _audit_config(tmp_path)
	config.output_root.mkdir(parents=True)
	(config.output_root / 'six_split_run_manifest.json').write_text(
		json.dumps({'scientific_jobs_executed': 1}), encoding='utf-8'
	)
	with pytest.raises(ValueError, match='executed jobs'):
		audit._validate_six_split_start_state(config)


def test_six_split_counter_is_checked_in_arbitrary_json(tmp_path: Path) -> None:
	config = _audit_config(tmp_path)
	config.output_root.mkdir(parents=True)
	(config.output_root / 'counter.json').write_text(
		json.dumps({'six_split_scientific_execution_count': 1}), encoding='utf-8'
	)
	with pytest.raises(ValueError, match='executed jobs'):
		audit._validate_six_split_start_state(config)


def _config_mapping(tmp_path: Path) -> dict[str, object]:
	artifact_root = tmp_path / 'artifacts'
	output_root = artifact_root / six_split_config.EXPECTED_OUTPUT_RELATIVE_PATH
	paths = {
		'original_split_handoff': tmp_path / 'original_handoff.json',
		'candidate_pretraining_handoff': artifact_root / 'candidate_handoff.json',
		'candidate_embeddings_dir': artifact_root / 'candidate_embeddings',
		'hard_baseline_pretraining_handoff': artifact_root / 'baseline_handoff.json',
		'hard_baseline_embeddings_dir': artifact_root / 'baseline_embeddings',
		'experiment96_dataset_manifest': artifact_root / 'experiment96_dataset.json',
		'experiment96_scientific_run_manifest': artifact_root / 'experiment96_run.json',
		'split_inventory_manifest': artifact_root / 'split_inventory.json',
		'split_token_dataset_manifest': artifact_root / 'split_tokens.json',
		'full_voxel_split_dataset_manifest': artifact_root / 'full_voxel.json',
		'original_split_dataset_manifest': artifact_root / 'original_dataset.json',
		'seismic_volume': artifact_root / 'seismic.npy',
		'source_label_segy': tmp_path / 'labels.sgy',
		'class_info': artifact_root / 'class_info.json',
		'segy_geometry_json': artifact_root / 'geometry.json',
	}
	return {
		'paths': {
			'artifact_root': str(artifact_root),
			'output_root': str(output_root),
		},
		'matrix': {
			'candidate': {
				'model_id': 'mh_ctmask010_nocons',
				'model_tag': (
					'strat_hmm_pretext_mh_k6810_ctmask010_nocons_'
					'topblock1_distill_v1'
				),
			},
			'baseline': {
				'model_id': 'mh_nocons',
				'model_tag': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
			},
			'split_ids': [f'split_{index:03d}' for index in range(6)],
			'budgets': ['cap25', 'cap50', 'cap100'],
			'label_subset_seed': 0,
			'decoder_seed': 42000,
		},
		'inputs': {
			**{name: str(path) for name, path in paths.items()},
			'source_identities': {
				name: {'path': str(path), 'sha256': 'a' * 64}
				for name, path in {
					'seismic_volume': paths['seismic_volume'],
					'source_label_segy': paths['source_label_segy'],
					'class_info': paths['class_info'],
					'segy_geometry_json': paths['segy_geometry_json'],
				}.items()
			},
		},
	}


def _audit_config(tmp_path: Path) -> six_split_config.F3CenterTraceMaskedSixSplitConfig:
	return six_split_config.F3CenterTraceMaskedSixSplitConfig(
		artifact_root=tmp_path,
		output_root=tmp_path / 'output',
		original_split_handoff=tmp_path / 'original.json',
		candidate_pretraining_handoff=tmp_path / 'candidate.json',
		candidate_embeddings_dir=tmp_path / 'candidate_embeddings',
		hard_baseline_pretraining_handoff=tmp_path / 'baseline.json',
		hard_baseline_embeddings_dir=tmp_path / 'baseline_embeddings',
		experiment96_dataset_manifest=tmp_path / 'dataset.json',
		experiment96_scientific_run_manifest=tmp_path / 'run.json',
		split_inventory_manifest=tmp_path / 'inventory.json',
		split_token_dataset_manifest=tmp_path / 'tokens.json',
		full_voxel_split_dataset_manifest=tmp_path / 'voxel.json',
		original_split_dataset_manifest=tmp_path / 'original_dataset.json',
		seismic_volume=tmp_path / 'seismic.npy',
		source_label_segy=tmp_path / 'labels.sgy',
		class_info=tmp_path / 'class_info.json',
		segy_geometry_json=tmp_path / 'geometry.json',
		source_identities={},
		candidate_model_id='mh_ctmask010_nocons',
		candidate_model_tag='strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1',
		baseline_model_id='mh_nocons',
		baseline_model_tag='strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		split_ids=tuple(f'split_{index:03d}' for index in range(6)),
		budgets=('cap25', 'cap50', 'cap100'),
		label_subset_seed=0,
		decoder_seed=42000,
	)


def _audit_payload(config) -> dict[str, object]:
	return {
		'artifact_type': audit.ARTIFACT_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'config': {},
		'split_ids': list(config.split_ids),
		'budgets': list(config.budgets),
		'primary_model_roles': list(config.primary_model_roles),
		'primary_model_tags': dict(config.primary_model_tags),
		'primary_matrix_row_count': 36,
		'future_candidate_jobs': 18,
		'future_new_baseline_jobs': 6,
		'historical_baseline_rows': 12,
		'future_new_scientific_jobs': 24,
		'scientific_jobs_executed': 0,
		'smoke_jobs_executed': 0,
		'source_files': {},
		'evidence': {},
		'git': {
			'git_commit': 'a' * 40,
			'git_status_short': [],
			'git_diff_sha256': 'b' * 64,
		},
	}
