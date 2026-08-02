"""Focused schema-v6 original-split runner contracts."""
# ruff: noqa: SLF001, TC003

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_xy_neighbor_unanimous as unanimous_config,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadCandidate,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as shared


def test_unanimous_jobs_are_exactly_the_canonical_fifteen_rows(
	tmp_path: Path,
) -> None:
	candidate = F3VoxelLabelBudgetMultiHeadCandidate(
		model_id=unanimous_config.XY_UNANIM_MODEL_ID,
		model_tag=unanimous_config.XY_UNANIM_MODEL_TAG,
		embeddings_dir=tmp_path / 'embeddings',
		pretraining_handoff=tmp_path / 'handoff.json',
	)
	config = SimpleNamespace(
		candidates=(candidate,),
		budgets=('cap25', 'cap50', 'cap100'),
		subsample_seeds=(0, 1, 2, 3, 4),
		output_root=tmp_path / 'outputs',
		decoder_seed=lambda seed: 42000 + seed,
	)
	datasets = {
		(budget, seed): {
			'per_class_cap': int(budget.removeprefix('cap')),
			'voxel_dataset_root': str(tmp_path / budget / str(seed)),
		}
		for budget in config.budgets
		for seed in config.subsample_seeds
	}

	jobs = shared._jobs(config, datasets)

	assert len(jobs) == 15
	assert [(job.budget_id, job.subsample_seed, job.decoder_seed) for job in jobs] == [
		(budget, seed, 42000 + seed)
		for budget in config.budgets
		for seed in config.subsample_seeds
	]


def test_unanimous_provenance_requires_schema_six_and_no_legacy_fields(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	fixture = _provenance_fixture(tmp_path)
	monkeypatch.setattr(
		shared,
		'load_f3_xy_neighbor_unanimous_pretraining_handoff',
		lambda _path: fixture.handoff,
	)
	monkeypatch.setattr(
		shared,
		'load_multi_head_xy_neighbor_unanimous_target_manifest',
		lambda _path, **_kwargs: fixture.target,
	)
	def replay_target_audit(
		_path: Path, *, artifact_root: Path
	) -> dict[str, object]:
		assert artifact_root == fixture.config.artifact_root
		return fixture.target_audit

	monkeypatch.setattr(
		shared,
		'replay_f3_xy_neighbor_unanimous_target_audit',
		replay_target_audit,
	)

	identity = _validate_fixture_provenance(fixture)
	assert identity['sha256'] == file_sha256(fixture.handoff_path)
	for schema in (2, 3, 4, 5):
		checkpoint = torch.load(
			fixture.checkpoint, map_location='cpu', weights_only=False
		)
		checkpoint['stratigraphy_checkpoint']['schema_version'] = schema
		torch.save(checkpoint, fixture.checkpoint)
		fixture.handoff['checkpoint']['sha256'] = file_sha256(fixture.checkpoint)
		with pytest.raises(ValueError, match='checkpoint schema must be 6'):
			_validate_fixture_provenance(fixture)
		checkpoint['stratigraphy_checkpoint']['schema_version'] = 6
		torch.save(checkpoint, fixture.checkpoint)
		fixture.handoff['checkpoint']['sha256'] = file_sha256(fixture.checkpoint)
	fixture.stratigraphy['xy_neighbor_consensus_smoothing'] = {'forbidden': True}
	with pytest.raises(ValueError, match='posterior/lateral'):
		_validate_fixture_provenance(fixture)


def _provenance_fixture(tmp_path: Path) -> SimpleNamespace:
	root = tmp_path / 'artifacts'
	embeddings_dir = (
		root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ unanimous_config.XY_UNANIM_MODEL_TAG
		/ 'overlap_x16'
	)
	embeddings_dir.mkdir(parents=True)
	metadata_path = output_paths(embeddings_dir, 'f3_facies_benchmark').metadata
	metadata_path.write_text('{}', encoding='utf-8')
	source = root / 'targets/hard.json'
	source.parent.mkdir(parents=True)
	source.write_text('{}', encoding='utf-8')
	target_path = root / 'targets/unanimous.json'
	target_path.write_text('{}', encoding='utf-8')
	target_audit_path = root / 'targets/target-audit.json'
	target_audit_path.write_text('{}', encoding='utf-8')
	checkpoint = (
		root / 'pretraining' / unanimous_config.XY_UNANIM_MODEL_TAG / 'best.pt'
	)
	checkpoint.parent.mkdir(parents=True)
	candidate = F3VoxelLabelBudgetMultiHeadCandidate(
		model_id=unanimous_config.XY_UNANIM_MODEL_ID,
		model_tag=unanimous_config.XY_UNANIM_MODEL_TAG,
		embeddings_dir=embeddings_dir,
		pretraining_handoff=(
			root
			/ 'pretraining'
			/ unanimous_config.XY_UNANIM_MODEL_TAG
			/ 'preflight/handoff.json'
		),
	)
	target_ref = {'path': str(target_path), 'sha256': file_sha256(target_path)}
	source_ref = {'path': str(source), 'sha256': file_sha256(source)}
	target_audit_ref = {
		'path': str(target_audit_path),
		'sha256': file_sha256(target_audit_path),
	}
	head_hashes = {
		str(k): {
			'f3_facies_benchmark': dict.fromkeys(
				('labels', 'confidence', 'valid_tokens', 'metadata'), f'{k:02x}' * 32
			),
		}
		for k in (6, 8, 10)
	}
	smoothing = {'policy': 'fixed'}
	checkpoint_identity = {
		'schema_version': 6,
		'model_tag': unanimous_config.XY_UNANIM_MODEL_TAG,
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_representation': 'xy_neighbor_unanimous_hard_labels_v1',
		'target_semantics': 'xy_neighbor_unanimous_outlier_correction_v1',
		'xy_neighbor_unanimous_target_manifest_sha256': target_ref['sha256'],
		'xy_neighbor_unanimous_target_manifest': target_ref,
		'per_head_xy_neighbor_unanimous_targets': head_hashes,
		'source_hard_manifest_sha256': source_ref['sha256'],
		'xy_neighbor_unanimous_smoothing': smoothing,
		'consistency_policy': 'disabled_for_xy_neighbor_unanimous_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'initial_student_state_sha256': '1' * 64,
		'initial_head_state_sha256': '2' * 64,
		'scientific_identity_sha256': '3' * 64,
		'stratigraphy_state_sha256': '4' * 64,
	}
	torch.save({'stratigraphy_checkpoint': checkpoint_identity}, checkpoint)
	checkpoint_ref = {'path': str(checkpoint), 'sha256': file_sha256(checkpoint)}
	target = {
		'source_hard_manifest': source_ref,
		'smoothing': smoothing,
		'heads': {
			str(k): {
				'surveys': {
					'f3_facies_benchmark': {
						name: {
							'path': f'/target/{k}/{name}',
							'sha256': head_hashes[str(k)]['f3_facies_benchmark'][name],
						}
						for name in (
							'labels',
							'confidence',
							'valid_tokens',
							'metadata',
						)
					}
				}
			}
			for k in (6, 8, 10)
		},
	}
	stratigraphy = {
		'model_tag': unanimous_config.XY_UNANIM_MODEL_TAG,
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_representation': 'xy_neighbor_unanimous_hard_labels_v1',
		'target_semantics': 'xy_neighbor_unanimous_outlier_correction_v1',
		'xy_neighbor_unanimous_target_manifest_sha256': target_ref['sha256'],
		'xy_neighbor_unanimous_target_manifest_path': target_ref['path'],
		'per_head_xy_neighbor_unanimous_target_sha256': head_hashes,
		'source_hard_manifest_sha256': source_ref['sha256'],
		'xy_neighbor_unanimous_smoothing': smoothing,
		'consistency_policy': 'disabled_for_xy_neighbor_unanimous_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'scientific_identity_sha256': '3' * 64,
		'checkpoint_stratigraphy_state_sha256': '4' * 64,
	}
	handoff = {
		'targets': {
			'target_representation': 'xy_neighbor_unanimous_hard_labels_v1',
			'target_semantics': 'xy_neighbor_unanimous_outlier_correction_v1',
			'consistency_policy': 'disabled_for_xy_neighbor_unanimous_v1',
			'target_manifest': target_ref,
			'target_audit': target_audit_ref,
			'xy_neighbor_unanimous_target_head_hashes': head_hashes,
			'source_hard_manifest': source_ref,
			'xy_neighbor_unanimous_smoothing': smoothing,
			'initial_student_state_sha256': '1' * 64,
			'initial_head_state_sha256': '2' * 64,
		},
		'checkpoint': checkpoint_ref,
		'embedding': {
			'root': str(embeddings_dir),
			'metadata_path': str(metadata_path),
			'metadata_sha256': '5' * 64,
			'embeddings_sha256': '6' * 64,
			'valid_tokens_sha256': '7' * 64,
			'valid_token_count': 9,
		},
	}
	handoff_path = candidate.pretraining_handoff
	handoff_path.parent.mkdir(parents=True)
	handoff_path.write_text('{}', encoding='utf-8')
	return SimpleNamespace(
		candidate=candidate,
		checkpoint=checkpoint,
		handoff=handoff,
		handoff_path=handoff_path,
		target=target,
		target_audit={
			'status': 'XYUNANIM_TARGET_GO',
			'xy_neighbor_unanimous_target_manifest': target_ref,
			'source_hard_manifest': source_ref,
		},
		stratigraphy=stratigraphy,
		config=SimpleNamespace(
			dataset={'name': 'f3_facies_benchmark'},
			artifact_root=root,
			multi_head_target_manifest=source,
		),
	)


def _validate_fixture_provenance(fixture: SimpleNamespace) -> object:
	return shared._validate_xy_neighbor_unanimous_handoff_provenance(
		fixture.handoff_path,
		config=fixture.config,
		candidate=fixture.candidate,
		checkpoint=fixture.checkpoint,
		checkpoint_sha256=file_sha256(fixture.checkpoint),
		embeddings_sha256='6' * 64,
		valid_tokens_sha256='7' * 64,
		embedding_metadata_sha256='5' * 64,
		valid_token_count=9,
		metadata={},
		stratigraphy=fixture.stratigraphy,
	)
