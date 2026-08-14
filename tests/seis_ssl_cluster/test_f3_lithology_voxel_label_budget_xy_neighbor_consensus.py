"""Focused contracts for the XY-neighbour-consensus original-split runner."""

from __future__ import annotations

# ruff: noqa: SLF001
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadCandidate,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_xy_neighbor_consensus import (  # noqa: E501
	XY_MODEL_ID,
	XY_MODEL_TAG,
	f3_lithology_voxel_label_budget_xy_neighbor_consensus_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as shared
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	VoxelLabelBudgetJobPlan,
)


def test_closed_xy_config_requires_one_canonical_fifteen_job_candidate(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	raw, hard = _closed_config_fixture(tmp_path)
	_calls: list[tuple[str, object]] = []
	monkeypatch.setattr(
		'seis_ssl_cluster.config.f3_lithology_voxel_label_budget_xy_neighbor_consensus.load_config',
		lambda _path: hard,
	)
	monkeypatch.setattr(
		'seis_ssl_cluster.config.f3_lithology_voxel_label_budget_xy_neighbor_consensus.importlib.import_module',
		lambda _name: SimpleNamespace(
			load_f3_xy_neighbor_consensus_screening_audit=lambda _path: {
				'artifact_type': (
					'f3_xy_neighbor_consensus_original_screening_preflight'
				),
				'schema_version': 1,
				'status': 'PASS',
			},
			validate_f3_xy_neighbor_consensus_screening_audit_binding=(
				lambda payload, **kwargs: _calls.append(('audit', (payload, kwargs)))
			),
		),
	)

	config = f3_lithology_voxel_label_budget_xy_neighbor_consensus_config_from_mapping(
		raw
	)

	assert config.job_count == 15
	assert [(item.model_id, item.model_tag) for item in config.candidates] == [
		(XY_MODEL_ID, XY_MODEL_TAG)
	]
	assert config.references.historical_m1_model_id is None
	assert config.run_manifest_name == 'xy_neighbor_consensus_job_manifest.json'
	assert config.run_manifest_type == (
		'f3_lithology_voxel_label_budget_xy_neighbor_consensus'
	)
	assert len(_calls) == 1


def test_xy_config_rejects_candidate_or_decoder_contract_drift(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	raw, hard = _closed_config_fixture(tmp_path)
	monkeypatch.setattr(
		'seis_ssl_cluster.config.f3_lithology_voxel_label_budget_xy_neighbor_consensus.load_config',
		lambda _path: hard,
	)
	monkeypatch.setattr(
		'seis_ssl_cluster.config.f3_lithology_voxel_label_budget_xy_neighbor_consensus.importlib.import_module',
		lambda _name: SimpleNamespace(
			load_f3_xy_neighbor_consensus_screening_audit=lambda _path: {},
			validate_f3_xy_neighbor_consensus_screening_audit_binding=(
				lambda _payload, **_kwargs: None
			),
		),
	)
	wrong_candidate = deepcopy(raw)
	wrong_candidate['multi_head']['candidates'][0]['model_id'] = 'mh_nocons'
	with pytest.raises(ValueError, match='canonical tuple'):
		f3_lithology_voxel_label_budget_xy_neighbor_consensus_config_from_mapping(
			wrong_candidate
		)
	wrong_decoder = deepcopy(raw)
	wrong_decoder['multi_head']['decoder']['embedding_dim'] = 128
	with pytest.raises(ValueError, match='canonical M3-V-LB architecture'):
		f3_lithology_voxel_label_budget_xy_neighbor_consensus_config_from_mapping(
			wrong_decoder
		)


def test_xy_candidate_provenance_requires_schema_v5_and_no_lateral_carryover(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	fixture = _provenance_fixture(tmp_path)
	monkeypatch.setattr(
		shared,
		'load_f3_xy_neighbor_consensus_pretraining_handoff',
		lambda _path: fixture.handoff,
	)
	monkeypatch.setattr(
		shared,
		'load_multi_head_xy_neighbor_consensus_target_manifest',
		lambda _path, **_kwargs: fixture.target,
	)

	identity = _validate_fixture_provenance(fixture)

	assert identity['sha256'] == file_sha256(fixture.handoff_path)
	checkpoint = torch.load(
		fixture.checkpoint,
		map_location='cpu',
		weights_only=False,
	)
	checkpoint['stratigraphy_checkpoint']['schema_version'] = 2
	torch.save(checkpoint, fixture.checkpoint)
	fixture.handoff['checkpoint']['sha256'] = file_sha256(fixture.checkpoint)
	with pytest.raises(ValueError, match='checkpoint schema must be 5'):
		_validate_fixture_provenance(fixture)
	checkpoint['stratigraphy_checkpoint']['schema_version'] = 5
	torch.save(checkpoint, fixture.checkpoint)
	fixture.handoff['checkpoint']['sha256'] = file_sha256(fixture.checkpoint)
	fixture.stratigraphy['lateral_smoothing'] = {'forbidden': True}
	with pytest.raises(ValueError, match='posterior/lateral'):
		_validate_fixture_provenance(fixture)


def test_xy_jobs_are_exactly_the_canonical_fifteen_rows(tmp_path: Path) -> None:
	candidate = F3VoxelLabelBudgetMultiHeadCandidate(
		model_id=XY_MODEL_ID,
		model_tag=XY_MODEL_TAG,
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


def test_reuse_only_pass_preserves_xy_manifest_bytes_and_mtime(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""The required post-execution recheck must not refresh manifest evidence."""
	candidate = F3VoxelLabelBudgetMultiHeadCandidate(
		model_id=XY_MODEL_ID,
		model_tag=XY_MODEL_TAG,
		embeddings_dir=tmp_path / 'embeddings',
		pretraining_handoff=tmp_path / 'handoff.json',
	)
	config = SimpleNamespace(
		output_root=tmp_path / 'outputs',
		dataset_manifest=tmp_path / 'dataset.json',
		original_run_manifest=tmp_path / 'original.json',
		current_k6_run_manifest=tmp_path / 'current.json',
		reports_dir=tmp_path / 'reports',
		run_manifest_name='xy_neighbor_consensus_job_manifest.json',
		run_manifest_type='f3_lithology_voxel_label_budget_xy_neighbor_consensus',
		candidates=(candidate,),
		budgets=('cap25',),
		subsample_seeds=(0,),
	)
	for path in (
		config.dataset_manifest,
		config.original_run_manifest,
		config.current_k6_run_manifest,
	):
		path.write_text('{}', encoding='utf-8')
	job = VoxelLabelBudgetJob(
		budget_id='cap25',
		per_class_cap=25,
		subsample_seed=0,
		decoder_seed=42000,
		model_role=XY_MODEL_ID,
		model_tag=XY_MODEL_TAG,
		voxel_dataset_root=tmp_path / 'dataset',
		output_root=tmp_path / 'job',
		dataset_row={},
	)
	identity = {XY_MODEL_ID: {'bound': True}}
	inspection = shared.F3VoxelLabelBudgetMultiHeadInspection(
		jobs=(job,),
		plans=(VoxelLabelBudgetJobPlan(job, 'REUSE_COMPLETED', None, 0),),
		historical_reference=SimpleNamespace(),
		candidate_identities=identity,
		estimated_new_bytes=0,
		disk_free_bytes=1,
	)
	row = {
		'budget_id': 'cap25',
		'subsample_seed': 0,
		'model_role': XY_MODEL_ID,
		'status': 'complete',
		'action': 'NEW',
	}
	manifest = config.reports_dir / 'xy_neighbor_consensus_job_manifest.json'
	shared._write_manifest(manifest, config, (row,), (), identity)
	before_bytes, before_mtime = manifest.read_bytes(), manifest.stat().st_mtime_ns
	monkeypatch.setattr(
		shared,
		'inspect_f3_lithology_voxel_label_budget_multi_head',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(shared, '_dataset_rows', lambda *_args: {('cap25', 0): {}})
	monkeypatch.setattr(shared, '_current_k6_rows', lambda *_args: {('cap25', 0): {}})
	monkeypatch.setattr(
		shared,
		'_validate_candidate_pairing',
		lambda *_args, **_kwargs: None,
	)
	monkeypatch.setattr(
		shared.control,
		'_completed_control_row',
		lambda _config, _stage, _job, **kwargs: {**row, 'action': kwargs['action']},
	)

	result = shared.run_f3_lithology_voxel_label_budget_multi_head(
		config, only_missing=True
	)

	assert result.rows == (row,)
	assert manifest.read_bytes() == before_bytes
	assert manifest.stat().st_mtime_ns == before_mtime


def _closed_config_fixture(
	tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
	"""Build matching hard/XY configs without depending on installed artifacts."""
	hard = deepcopy(
		load_config(
			Path(
				'experiments/f3/facies_benchmark_v1/'
				'95_strat_hmm_multi_head_k6810_low_label_v1/'
				'01_run_multi_head_voxel_label_budget.yaml'
			)
		)
	)
	artifact_root = (tmp_path / 'artifacts').resolve()
	f3_root = (tmp_path / 'f3').resolve()
	results_root = (tmp_path / 'reports').resolve()
	hard['paths'] = {
		'artifact_root': str(artifact_root),
		'f3_root': str(f3_root),
		'results_root': str(results_root),
	}
	hard['references'] = {
		'dataset_manifest': str(artifact_root / 'datasets/manifest.json'),
		'multi_head_target_manifest': str(artifact_root / 'targets/hard.json'),
		'original_run_manifest': str(artifact_root / 'runs/original.json'),
		'current_k6_run_manifest': str(artifact_root / 'runs/current.json'),
		'mae_model_id': 'mae',
		'current_k6_model_id': 'm1_current_k6',
		'historical_m1_model_id': 'm1',
	}
	for candidate in hard['candidates']:
		candidate['embeddings_dir'] = str(
			artifact_root / 'embeddings' / str(candidate['model_id'])
		)
		candidate['pretraining_handoff'] = str(
			artifact_root / 'pretraining' / f"{candidate['model_id']}.json"
		)
	hard['labels'] = {
		'seismic_volume': str(artifact_root / 'inputs/seismic.npy'),
		'source_label_volume': str(artifact_root / 'inputs/labels.npy'),
		'source_label_segy': str(f3_root / 'labels.sgy'),
		'png_label_inventory': str(artifact_root / 'inputs/inventory.csv'),
		'segy_geometry_json': str(artifact_root / 'inputs/geometry.json'),
		'class_info': str(artifact_root / 'inputs/classes.json'),
	}
	hard['outputs'] = {
		'output_root': str(artifact_root / 'hard-output'),
		'overwrite': False,
	}
	multi_head = deepcopy(hard)
	multi_head['references'].pop('historical_m1_model_id')
	multi_head['candidates'] = [
		{
			'model_id': XY_MODEL_ID,
			'model_tag': XY_MODEL_TAG,
			'embeddings_dir': str(
				artifact_root
				/ 'embeddings/f3/facies_benchmark_v1'
				/ XY_MODEL_TAG
				/ 'overlap_x16'
			),
			'pretraining_handoff': str(
				artifact_root
				/ 'pretraining/f3/facies_benchmark_v1'
				/ XY_MODEL_TAG
				/ 'preflight/xy_neighbor_consensus_handoff.json'
			),
		}
	]
	multi_head['outputs']['output_root'] = str(
		artifact_root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_xy_neighbor_consensus_k6810_v1/original_split'
	)
	audit_path = (
		artifact_root
		/ 'lithology/f3/facies_benchmark_v1'
		/ 'voxel_label_budget_xy_neighbor_consensus_k6810_v1'
		/ 'original_split/preflight/xy_neighbor_consensus_screening_audit.json'
	)
	audit_path.parent.mkdir(parents=True)
	audit_path.write_text('{}', encoding='utf-8')
	hard_path = tmp_path / 'hard.yaml'
	hard_path.write_text('{}', encoding='utf-8')
	return {
		'multi_head': multi_head,
		'hard_multi_head_config': str(hard_path),
		'screening_audit': str(audit_path),
	}, hard


def _provenance_fixture(tmp_path: Path) -> SimpleNamespace:
	"""Create compact live identities for the schema-v5 provenance validator."""
	root = tmp_path / 'artifacts'
	embeddings_dir = (
		root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ XY_MODEL_TAG
		/ 'overlap_x16'
	)
	embeddings_dir.mkdir(parents=True)
	metadata_path = output_paths(embeddings_dir, 'f3_facies_benchmark').metadata
	metadata_path.write_text('{}', encoding='utf-8')
	source = root / 'targets/hard.json'
	source.parent.mkdir(parents=True)
	source.write_text('{}', encoding='utf-8')
	target_path = root / 'targets/xy.json'
	target_path.write_text('{}', encoding='utf-8')
	checkpoint = root / 'pretraining' / XY_MODEL_TAG / 'best.pt'
	checkpoint.parent.mkdir(parents=True)
	candidate = F3VoxelLabelBudgetMultiHeadCandidate(
		model_id=XY_MODEL_ID,
		model_tag=XY_MODEL_TAG,
		embeddings_dir=embeddings_dir,
		pretraining_handoff=(
			root / 'pretraining' / XY_MODEL_TAG / 'preflight/handoff.json'
		),
	)
	refs = {
		'path': str(target_path),
		'sha256': file_sha256(target_path),
	}
	source_ref = {'path': str(source), 'sha256': file_sha256(source)}
	head_hashes = {
		str(k): {
			'f3_facies_benchmark': dict.fromkeys(('labels', 'confidence', 'valid_tokens', 'metadata'), f'{k:02x}' * 32),  # noqa: E501
		}
		for k in (6, 8, 10)
	}
	smoothing = {'policy': 'fixed'}
	checkpoint_identity = {
		'schema_version': 5,
		'model_tag': XY_MODEL_TAG,
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
		'xy_neighbor_consensus_target_manifest_sha256': refs['sha256'],
		'xy_neighbor_consensus_target_manifest': refs,
		'per_head_xy_neighbor_consensus_targets': head_hashes,
		'source_hard_manifest_sha256': source_ref['sha256'],
		'xy_neighbor_consensus_smoothing': smoothing,
		'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
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
							'sha256': head_hashes[str(k)]['f3_facies_benchmark'][
								name
							],
						}
						for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
					}
				}
			}
			for k in (6, 8, 10)
		},
	}
	stratigraphy = {
		'model_tag': XY_MODEL_TAG,
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
		'xy_neighbor_consensus_target_manifest_sha256': refs['sha256'],
		'xy_neighbor_consensus_target_manifest_path': refs['path'],
		'per_head_xy_neighbor_consensus_target_sha256': head_hashes,
		'source_hard_manifest_sha256': source_ref['sha256'],
		'xy_neighbor_consensus_smoothing': smoothing,
		'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'scientific_identity_sha256': '3' * 64,
		'checkpoint_stratigraphy_state_sha256': '4' * 64,
	}
	handoff = {
		'targets': {
			'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
			'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
			'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
			'target_manifest': refs,
			'xy_neighbor_consensus_target_head_hashes': head_hashes,
			'source_hard_manifest': source_ref,
			'xy_neighbor_consensus_smoothing': smoothing,
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
		stratigraphy=stratigraphy,
		config=SimpleNamespace(
			dataset={'name': 'f3_facies_benchmark'},
			multi_head_target_manifest=source,
		),
	)


def _validate_fixture_provenance(fixture: SimpleNamespace) -> object:
	return shared._validate_xy_neighbor_consensus_handoff_provenance(
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
