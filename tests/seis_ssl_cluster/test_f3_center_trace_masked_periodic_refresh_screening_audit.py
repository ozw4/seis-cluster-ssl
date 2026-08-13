"""Regression coverage for periodic-refresh screening contracts."""
# ruff: noqa: SLF001, TC003

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from seis_ssl_cluster.f3 import (
	center_trace_masked_periodic_refresh_screening_audit as audit,
)


def test_decoder_contract_accepts_normalized_hidden_channel_tuple(
	tmp_path: Path, monkeypatch
) -> None:
	"""Parsed decoder tuples must match the canonical hidden-channel contract."""
	hard_path = (
		tmp_path
		/ 'experiments/f3/facies_benchmark_v1'
		/ '95_strat_hmm_multi_head_k6810_low_label_v1'
		/ '01_run_multi_head_voxel_label_budget.yaml'
	)
	hard_path.parent.mkdir(parents=True)
	hard_path.touch()
	config = SimpleNamespace(
		candidate_decoder_config=tmp_path / 'candidate.yaml',
		workspace_root=tmp_path,
		source_hard_manifest=tmp_path / 'target.json',
	)
	decoder = SimpleNamespace(
		spec='frozen_embedding_decoder_nearest_voxel_ln_v1',
		embedding_dim=384,
		hidden_channels=(128, 64, 32),
		upsample_mode='nearest',
		normalization='voxelwise_layer_norm',
	)
	train = SimpleNamespace(
		epochs=50,
		steps_per_epoch=440,
		class_weight='balanced',
		sampling_mode='uniform_tiles_with_replacement',
	)
	monkeypatch.setattr(
		audit,
		'load_config',
		lambda _path: {
			'multi_head': {},
			'hard_multi_head_config': str(hard_path),
		},
	)
	monkeypatch.setattr(
		audit,
		'config_from_mapping_for_candidates',
		lambda *_args, **_kwargs: SimpleNamespace(
			decoder=decoder,
			train=train,
			multi_head_target_manifest=config.source_hard_manifest,
		),
	)
	monkeypatch.setattr(
		audit,
		'f3_lithology_voxel_label_budget_multi_head_config_from_mapping',
		lambda _mapping: SimpleNamespace(),
	)
	monkeypatch.setattr(audit, '_validate_hard_decoder_contract', lambda *_args: None)
	monkeypatch.setattr(audit, '_identity', lambda _path: {})

	evidence = audit._decoder_contract_evidence(
		config,
		target={'head_ks': [6, 8, 10]},
		hard={'identity': {'model_tag': 'hard'}},
		periodic={
			'pseudo_targets': {'manifest': str(config.source_hard_manifest)},
			'identity': {'scientific_identity': {}},
		},
	)

	assert evidence['hidden_channels'] == [128, 64, 32]
