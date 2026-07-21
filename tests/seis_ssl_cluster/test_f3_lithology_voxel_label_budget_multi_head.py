"""Focused contracts for the paired multi-head low-label voxel runner."""

from __future__ import annotations

# ruff: noqa: SLF001
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import seis_ssl_cluster.f3.lithology.voxel_label_budget_multi_head as multi_head
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	F3VoxelLabelBudgetMultiHeadCandidate,
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	VoxelLabelBudgetJobPlan,
)


def test_multi_head_job_matrix_is_canonical_thirty_row_pair(tmp_path: Path) -> None:
	config = SimpleNamespace(
		candidates=(
			_candidate(tmp_path, 'mh_nocons'),
			_candidate(tmp_path, 'mh_cons010'),
		),
		budgets=('cap25', 'cap50', 'cap100'),
		subsample_seeds=(0, 1, 2, 3, 4),
		output_root=tmp_path / 'outputs',
		decoder_seed=lambda seed: 42000 + seed,
	)
	dataset_rows = {
		(budget, seed): {
			'per_class_cap': int(budget.removeprefix('cap')),
			'voxel_dataset_root': str(tmp_path / budget / str(seed)),
		}
		for budget in config.budgets
		for seed in config.subsample_seeds
	}

	jobs = multi_head._jobs(config, dataset_rows)

	assert len(jobs) == 30
	assert [
		(job.model_role, job.budget_id, job.subsample_seed) for job in jobs
	] == [
		(candidate, budget, seed)
		for candidate in ('mh_nocons', 'mh_cons010')
		for budget in ('cap25', 'cap50', 'cap100')
		for seed in range(5)
	]


def test_candidate_identity_requires_bound_matching_handoff(tmp_path: Path) -> None:
	config, candidate, canonical_valid_tokens_sha256, paths = (
		_candidate_identity_fixture(tmp_path)
	)

	identity = multi_head._candidate_identity(
		config,
		candidate,
		canonical_valid_tokens_sha256=canonical_valid_tokens_sha256,
	)

	assert identity['pretraining_handoff']['sha256'] == file_sha256(paths.handoff)
	wrong = _handoff_payload(paths, model_tag=candidate.model_tag)
	wrong['embedding_metadata_sha256'] = '0' * 64
	paths.handoff.write_text(json.dumps(wrong), encoding='utf-8')
	with pytest.raises(ValueError, match='handoff metadata SHA-256 mismatch'):
		multi_head._candidate_identity(
			config,
			candidate,
			canonical_valid_tokens_sha256=canonical_valid_tokens_sha256,
		)
	wrong = _handoff_payload(paths, model_tag='another_candidate')
	paths.handoff.write_text(json.dumps(wrong), encoding='utf-8')

	with pytest.raises(ValueError, match='handoff model tag mismatch'):
		multi_head._candidate_identity(
			config,
			candidate,
			canonical_valid_tokens_sha256=canonical_valid_tokens_sha256,
		)


def test_consistency_candidate_requires_explicit_consistency_weight(
	tmp_path: Path,
) -> None:
	config, candidate, canonical_valid_tokens_sha256, paths = (
		_candidate_identity_fixture(tmp_path, model_id='mh_cons010')
	)
	metadata = json.loads(paths.metadata.read_text(encoding='utf-8'))
	metadata['stratigraphy_pretext']['consistency_weight'] = None
	paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='cons010 embedding consistency identity'):
		multi_head._candidate_identity(
			config,
			candidate,
			canonical_valid_tokens_sha256=canonical_valid_tokens_sha256,
		)


@pytest.mark.parametrize(
	('missing', 'message'),
	[
		('original_run_manifest', 'original_run_manifest'),
		('historical_m1_model_id', 'historical_m1_model_id'),
	],
)
def test_config_requires_original_mae_historical_m1_reference(
	missing: str, message: str
) -> None:
	path = Path(
		'experiments/f3/facies_benchmark_v1/'
		'95_strat_hmm_multi_head_k6810_low_label_v1/'
		'01_run_multi_head_voxel_label_budget.yaml'
	)
	raw = dict(load_config(path))
	references = dict(raw['references'])
	references.pop(missing)
	raw['references'] = references

	with pytest.raises(TypeError, match=message):
		f3_lithology_voxel_label_budget_multi_head_config_from_mapping(raw)


def test_only_missing_reuses_resumes_and_quarantines_selected_jobs(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = SimpleNamespace(
		output_root=tmp_path / 'outputs',
		dataset_manifest=tmp_path / 'dataset.json',
		original_run_manifest=tmp_path / 'original.json',
		current_k6_run_manifest=tmp_path / 'current.json',
		reports_dir=tmp_path / 'reports',
		candidates=(_candidate(tmp_path, 'mh_nocons'),),
	)
	for path in (
		config.dataset_manifest,
		config.original_run_manifest,
		config.current_k6_run_manifest,
	):
		path.write_text('{}', encoding='utf-8')
	jobs = tuple(
		_job(config, seed) for seed in range(3)
	)
	plans = tuple(
		VoxelLabelBudgetJobPlan(job, state, None, 0)
		for job, state in zip(
			jobs, ('REUSE_COMPLETED', 'RESUME_LATEST', 'INVALID_OR_PARTIAL'),
			strict=True,
		)
	)
	inspection = multi_head.F3VoxelLabelBudgetMultiHeadInspection(
		jobs=jobs,
		plans=plans,
		historical_reference=SimpleNamespace(),
		candidate_identities={'mh_nocons': {'identity': 'bound'}},
		estimated_new_bytes=0,
		disk_free_bytes=1,
	)
	resumed: list[Path | None] = []
	quarantined: list[Path] = []
	monkeypatch.setattr(
		multi_head,
		'inspect_f3_lithology_voxel_label_budget_multi_head',
		lambda *_args, **_kwargs: inspection,
	)
	monkeypatch.setattr(
		multi_head,
		'_dataset_rows',
		lambda *_args: {
			(job.budget_id, job.subsample_seed): {} for job in jobs
		},
	)
	monkeypatch.setattr(
		multi_head,
		'_current_k6_rows',
		lambda *_args: {(job.budget_id, job.subsample_seed): {} for job in jobs},
	)
	monkeypatch.setattr(
		multi_head, '_validate_candidate_pairing', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		multi_head,
		'quarantine_voxel_label_budget_output',
		lambda path, **_kwargs: (
			quarantined.append(path) or path.with_name('quarantine')
		),
	)
	monkeypatch.setattr(
		multi_head,
		'run_voxel_label_budget_job',
		lambda *_args, **kwargs: resumed.append(kwargs['resume']),
	)
	monkeypatch.setattr(
		multi_head.control,
		'_completed_control_row',
		lambda _config, _stage, job, **kwargs: {
			'budget_id': job.budget_id,
			'subsample_seed': job.subsample_seed,
			'model_role': job.model_role,
			'status': 'complete',
			'action': kwargs['action'],
		},
	)

	result = multi_head.run_f3_lithology_voxel_label_budget_multi_head(
		config, only_missing=True
	)

	assert [row['action'] for row in result.rows] == ['REUSED', 'RESUMED', 'NEW']
	assert resumed == [jobs[1].decoder_dir / 'latest.pt', None]
	assert quarantined == [jobs[2].output_root]


def test_candidate_pairing_also_requires_mae_historical_identity(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	current_calls: list[object] = []
	historical_calls: list[object] = []
	row = {'budget_id': 'cap25', 'subsample_seed': 0}
	current = {'model_role': 'm1_current_k6'}
	historical = SimpleNamespace()
	dataset_row = {'budget_id': 'cap25', 'subsample_seed': 0}
	monkeypatch.setattr(
		multi_head,
		'_validate_current_pair',
		lambda actual, reference: current_calls.append((actual, reference)),
	)
	monkeypatch.setattr(
		multi_head.control,
		'_validate_paired_identity',
		lambda actual, **kwargs: historical_calls.append((actual, kwargs)),
	)

	multi_head._validate_candidate_pairing(
		row,
		current_reference=current,
		historical_reference=historical,
		dataset_row=dataset_row,
	)

	assert current_calls == [(row, current)]
	assert historical_calls == [
		(row, {'reference': historical, 'dataset_row': dataset_row})
	]


def _manifest_row(*, model_role: str = 'mh_nocons') -> dict[str, object]:
	return {
		'budget_id': 'cap25',
		'subsample_seed': 0,
		'model_role': model_role,
		'status': 'complete',
	}


@pytest.mark.parametrize(
	('payload', 'message'),
	[
		(
			{
				'rows': [_manifest_row(), _manifest_row()],
				'row_count': 2,
				'complete_count': 2,
			},
			'duplicate row',
		),
		(
			{
				'rows': [_manifest_row(model_role='m1')],
				'row_count': 1,
				'complete_count': 1,
			},
			'non-matrix row',
		),
		(
			{
				'rows': [_manifest_row()],
				'row_count': 2,
				'complete_count': 1,
			},
			'row count mismatch',
		),
		(
			{
				'rows': [{'budget_id': 'cap25', 'status': 'complete'}],
				'row_count': 1,
				'complete_count': 1,
			},
			'incomplete job identity',
		),
	],
)
def test_owned_manifest_rows_reject_malformed_partial_matrix(
	tmp_path: Path, payload: dict[str, object], message: str
) -> None:
	config = SimpleNamespace(
		candidates=(_candidate(tmp_path, 'mh_nocons'),),
		budgets=('cap25',),
		subsample_seeds=(0,),
	)

	with pytest.raises(ValueError, match=message):
		multi_head._validate_owned_rows(payload, config)


def _candidate_identity_fixture(
	tmp_path: Path, *, model_id: str = 'mh_nocons'
) -> tuple[object, F3VoxelLabelBudgetMultiHeadCandidate, str, object]:
	candidate = _candidate(tmp_path, model_id)
	root = candidate.embeddings_dir
	root.mkdir(parents=True)
	output = output_paths(root, 'f3_facies_benchmark')
	embeddings = output.embeddings
	valid_tokens = output.valid_tokens
	metadata = output.metadata
	checkpoint = (
		tmp_path / candidate.model_tag / 'best.pt'
	)
	checkpoint.parent.mkdir()
	checkpoint.write_bytes(b'checkpoint')
	np.save(embeddings, np.zeros((76, 113, 32, 384), dtype=np.float16))
	np.save(valid_tokens, np.ones((76, 113, 32), dtype=bool))
	paths = SimpleNamespace(
		embeddings=embeddings,
		valid_tokens=valid_tokens,
		metadata=metadata,
		handoff=candidate.pretraining_handoff,
		checkpoint=checkpoint,
	)
	paths.handoff.parent.mkdir(parents=True, exist_ok=True)
	stratigraphy = _stratigraphy(model_id)
	metadata.write_text(
		json.dumps(
			{
				'checkpoint_path': str(checkpoint),
				'checkpoint_sha256': file_sha256(checkpoint),
				'stratigraphy_pretext': stratigraphy,
			}
		),
		encoding='utf-8',
	)
	paths.handoff.write_text(
		json.dumps(_handoff_payload(paths, model_tag=candidate.model_tag)),
		encoding='utf-8',
	)
	config = SimpleNamespace(dataset={'name': 'f3_facies_benchmark'})
	return config, candidate, file_sha256(valid_tokens), paths


def _handoff_payload(paths: object, *, model_tag: str) -> dict[str, object]:
	return {
		'artifact_type': 'f3_multi_head_pretraining_handoff',
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': model_tag,
		'embedding_metadata_sha256': file_sha256(paths.metadata),  # type: ignore[attr-defined]
		'checkpoint': {
			'path': str(paths.checkpoint),  # type: ignore[attr-defined]
			'sha256': file_sha256(paths.checkpoint),  # type: ignore[attr-defined]
		},
		'stratigraphy_pretext': _stratigraphy(
			'mh_cons010' if 'cons010' in model_tag else 'mh_nocons'
		),
	}


def _stratigraphy(model_id: str) -> dict[str, object]:
	return {
		'model_tag': (
			'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1'
			if model_id == 'mh_cons010'
			else 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
		),
		'head_spec': 'multi_resolution_ordered_prototypes_v1',
		'head_ks': [6, 8, 10],
		'target_manifest_sha256': 'a' * 64,
		'consistency_policy': 'normalized_order_smooth_l1_v1',
		'consistency_weight': 0.1 if model_id == 'mh_cons010' else 0.0,
		'consistency_beta': 0.1,
		'scientific_identity_sha256': 'b' * 64,
	}


def _candidate(tmp_path: Path, model_id: str) -> F3VoxelLabelBudgetMultiHeadCandidate:
	model_tag = _stratigraphy(model_id)['model_tag']
	assert isinstance(model_tag, str)
	return F3VoxelLabelBudgetMultiHeadCandidate(
		model_id=model_id,
		model_tag=model_tag,
		embeddings_dir=tmp_path / f'embeddings-{model_id}',
		pretraining_handoff=tmp_path / f'handoff-{model_id}.json',
	)


def _job(config: object, seed: int) -> VoxelLabelBudgetJob:
	return VoxelLabelBudgetJob(
		budget_id='cap25',
		per_class_cap=25,
		subsample_seed=seed,
		decoder_seed=42000 + seed,
		model_role='mh_nocons',
		model_tag=config.candidates[0].model_tag,  # type: ignore[attr-defined]
		voxel_dataset_root=Path('/dataset'),
		output_root=config.output_root / str(seed),  # type: ignore[attr-defined]
		dataset_row={},
	)
