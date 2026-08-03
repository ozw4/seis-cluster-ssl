"""Focused contracts for the strict experiment-104 validator."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import seis_ssl_cluster.f3.center_trace_masked_pretraining_validation as validation


def test_validator_config_is_closed_and_requires_all_inputs(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts'
	experiment_root = artifact_root / 'pretraining'
	experiment_root.mkdir(parents=True)
	paths = {
		'target_manifest': artifact_root / 'target.json',
		'hard_handoff': artifact_root / 'hard_handoff.json',
		'hard_full_config': artifact_root / 'hard.yaml',
		'center_trace_masked_smoke_config': artifact_root / 'smoke.yaml',
		'center_trace_masked_full_config': artifact_root / 'full.yaml',
		'center_trace_masked_embedding_config': artifact_root / 'embedding.yaml',
	}
	for path in paths.values():
		path.write_text('{}\n', encoding='utf-8')
	mapping = {
		'artifact_root': str(artifact_root),
		'experiment_root': str(experiment_root),
		**{key: str(path) for key, path in paths.items()},
	}

	resolved = (
		validation.f3_center_trace_masked_pretraining_validation_config_from_mapping(
			mapping
		)
	)

	assert resolved.target_manifest == paths['target_manifest']
	assert resolved.center_trace_masked_embedding_config == paths[
		'center_trace_masked_embedding_config'
	]
	with pytest.raises(ValueError, match='unknown center-trace'):
		validation.f3_center_trace_masked_pretraining_validation_config_from_mapping(
			{**mapping, 'forbidden': 'value'}
		)
	with pytest.raises(ValueError, match='center_trace_masked_full_config'):
		validation.f3_center_trace_masked_pretraining_validation_config_from_mapping(
			{
				key: value
				for key, value in mapping.items()
				if key != 'center_trace_masked_full_config'
			}
			)


def _embedding_extraction_fixture(
	tmp_path: Path,
	*,
	output_root: Path | None = None,
	checkpoint: Path | None = None,
) -> tuple[
	validation.F3CenterTraceMaskedPretrainingValidationConfig,
	dict[str, object],
	Path,
	Path,
	]:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	extraction_config_path = tmp_path / 'embedding.yaml'
	extraction_config_path.write_text('{}\n', encoding='utf-8')
	selected = tmp_path / 'selected.pt'
	selected.write_bytes(b'selected')
	manifest = tmp_path / 'manifest.json'
	manifest.write_text('{}\n', encoding='utf-8')
	output = output_root or artifact_root / 'configured-embeddings'
	configured_checkpoint = checkpoint or selected
	extraction = {
		'paths': {'artifact_root': str(artifact_root)},
		'manifests': {'input': str(manifest)},
		'embeddings': {
			'checkpoint': str(configured_checkpoint),
			'output_dir': str(output),
		},
		'embedding': {
			'window_size': [128, 128, 128],
			'overlap': [112, 64, 64],
			'output_dtype': 'float16',
			'amp': False,
			'amp_dtype': 'auto',
			'min_token_valid_fraction': 0.5,
			'preprocessing_cache': {'mode': 'memmap'},
		},
	}
	config = validation.F3CenterTraceMaskedPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root,
		target_manifest=tmp_path / 'target.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_handoff=tmp_path / 'handoff.json',
		center_trace_masked_smoke_config=tmp_path / 'smoke.yaml',
		center_trace_masked_full_config=tmp_path / 'full.yaml',
		center_trace_masked_embedding_config=extraction_config_path,
	)
	return config, extraction, selected, manifest


def test_embedding_evidence_loads_extraction_config_and_uses_its_output_root(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	output_root = tmp_path / 'configured-embeddings'
	config, extraction, selected, manifest = _embedding_extraction_fixture(
		tmp_path, output_root=output_root
	)
	loaded: list[Path] = []
	monkeypatch.setattr(
		validation,
		'load_config',
		lambda path: loaded.append(path) or extraction,
	)
	monkeypatch.setattr(
		validation,
		'resolve_embedding_extraction_config',
		lambda raw: raw,
	)
	monkeypatch.setattr(
		validation,
		'output_paths',
		lambda root, _survey: (_ for _ in ()).throw(
			ValueError(f'configured root: {root}')
		),
	)

	with pytest.raises(ValueError, match='configured root'):
		validation._embedding_evidence(  # noqa: SLF001
			config,
			{'selected_path': str(selected)},
			{'manifests': {'train': str(manifest)}},
		)

	assert loaded == [config.center_trace_masked_embedding_config]


def test_embedding_evidence_binds_training_manifest_not_target_manifest(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config, extraction, selected, _manifest = _embedding_extraction_fixture(tmp_path)
	monkeypatch.setattr(validation, 'load_config', lambda _path: extraction)
	monkeypatch.setattr(
		validation,
		'resolve_embedding_extraction_config',
		lambda raw: raw,
	)
	files = validation.output_paths(
		Path(str(extraction['embeddings']['output_dir'])),  # type: ignore[index]
		'f3_facies_benchmark',
	)
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b'placeholder')

	other_manifest = tmp_path / 'other-manifest.json'
	other_manifest.write_text('{}\n', encoding='utf-8')
	training = {'manifests': {'train': str(other_manifest)}}
	with pytest.raises(ValueError, match='does not bind training manifest'):
		validation._embedding_evidence(  # noqa: SLF001
			config,
			{'selected_path': str(selected)},
			training,
		)


def test_embedding_evidence_rejects_extraction_checkpoint_drift(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	configured_checkpoint = tmp_path / 'stale.pt'
	configured_checkpoint.write_bytes(b'stale')
	config, extraction, selected, manifest = _embedding_extraction_fixture(
		tmp_path, checkpoint=configured_checkpoint
	)
	monkeypatch.setattr(validation, 'load_config', lambda _path: extraction)
	monkeypatch.setattr(
		validation,
		'resolve_embedding_extraction_config',
		lambda raw: raw,
	)
	files = validation.output_paths(
		Path(str(extraction['embeddings']['output_dir'])),  # type: ignore[index]
		'f3_facies_benchmark',
	)
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b'placeholder')

	with pytest.raises(ValueError, match='does not bind selected checkpoint'):
		validation._embedding_evidence(  # noqa: SLF001
			config,
			{'selected_path': str(selected)},
			{'manifests': {'train': str(manifest)}},
		)


def _target_file_reference_fixture(tmp_path: Path) -> dict[str, object]:
	references = {}
	for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
		path = tmp_path / f'{name}.bin'
		path.write_bytes(name.encode())
		references[name] = {
			'path': str(path),
			'sha256': validation.file_sha256(path),
		}
	return {
		'heads': {
			str(head_k): {
				'surveys': {'f3_facies_benchmark': deepcopy(references)}
			}
			for head_k in (6, 8, 10)
		}
	}


@pytest.mark.parametrize(
	'artifact_name', ['labels', 'confidence', 'valid_tokens', 'metadata']
)
def test_target_file_references_check_every_artifact(
	tmp_path: Path, artifact_name: str
) -> None:
	target = _target_file_reference_fixture(tmp_path)
	validation._validate_target_file_references(target)  # noqa: SLF001

	entry = target['heads']['6']['surveys']['f3_facies_benchmark']  # type: ignore[index]
	path = Path(entry[artifact_name]['path'])  # type: ignore[index]
	path.write_bytes(b'tampered')
	with pytest.raises(
		ValueError, match=rf'target {artifact_name} reference hash mismatch'
	):
		validation._validate_target_file_references(target)  # noqa: SLF001


def test_hard_config_parity_rejects_non_hard_target_representation(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	hard = {
		'paths': {'output_root': str(tmp_path / 'hard')},
		'identity': {'model_tag': 'hard', 'scientific_identity': {}},
		'pseudo_targets': {
			'target_representation': validation._TARGET_REPRESENTATION  # noqa: SLF001
		},
	}
	center = deepcopy(hard)
	center['paths']['output_root'] = str(tmp_path / 'center')  # type: ignore[index]
	center['identity']['model_tag'] = 'center'  # type: ignore[index]
	center['spatial_context'] = {}
	center['pseudo_targets'] = {
		'target_representation': validation._TARGET_REPRESENTATION  # noqa: SLF001
	}
	hard_runtime = {
		'initial_student_state_sha256': 'a' * 64,
		'initial_head_state_sha256': 'b' * 64,
		'trainability_summary': {},
		'optimizer_group_identity': [
			{'name': 'encoder'},
			{'name': 'head'},
		],
	}
	center_runtime = {
		**hard_runtime,
		'optimizer_group_identity': [
			{'name': 'encoder'},
			{'name': 'head'},
			{
				'name': 'spatial_context',
				'parameter_names': ['spatial_context.replacement_token'],
			},
		],
	}
	monkeypatch.setattr(
		validation,
		'_runtime_contract',
		lambda _training, *, center: center_runtime if center else hard_runtime,
	)

	validation._hard_config_parity(hard, center)  # noqa: SLF001
	center['pseudo_targets']['target_representation'] = (  # type: ignore[index]
		'ordered_path_state_posterior_v1'
	)
	with pytest.raises(ValueError, match='center-trace pseudo_targets'):
		validation._hard_config_parity(hard, center)  # noqa: SLF001

	hard['pseudo_targets']['target_representation'] = (  # type: ignore[index]
		'ordered_path_state_posterior_v1'
	)
	center['pseudo_targets']['target_representation'] = (  # type: ignore[index]
		validation._TARGET_REPRESENTATION  # noqa: SLF001
	)
	with pytest.raises(ValueError, match='hard pseudo_targets'):
		validation._hard_config_parity(hard, center)  # noqa: SLF001


def test_hard_target_representation_must_be_hard_viterbi() -> None:
	with pytest.raises(ValueError, match='must be'):
		validation._validate_hard_target_representation(  # noqa: SLF001
			{
				'pseudo_targets': {
					'target_representation': 'ordered_path_state_posterior_v1'
				}
			},
			'candidate',
		)


def test_runtime_contract_binds_owned_replacement_token_as_spatial_context(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	student = torch.nn.Linear(2, 2)
	head = torch.nn.Linear(2, 2)
	spatial_context = torch.nn.Module()
	spatial_context.register_parameter(
		'replacement_token', torch.nn.Parameter(torch.ones(2))
	)
	optimizer = torch.optim.AdamW(
		[
			{'params': head.parameters(), 'name': 'head', 'lr': 1.0e-3},
			{'params': student.parameters(), 'name': 'encoder', 'lr': 1.0e-4},
			{
				'params': spatial_context.parameters(),
				'name': 'spatial_context',
				'lr': 1.0e-3,
			},
		]
	)
	components = SimpleNamespace(
		student=student,
		heads=head,
		replacement_token=spatial_context,
		optimizer=optimizer,
		trainability_summary=SimpleNamespace(
			trainable_parameter_count=5,
			frozen_parameter_count=0,
			trainable_names=(),
		),
	)
	monkeypatch.setattr(
		validation,
		'build_strat_hmm_components',
		lambda _training, **_kwargs: components,
	)

	runtime = validation._runtime_contract(  # noqa: SLF001
		{'train': {'seed': 273}}, center=True
	)

	assert runtime['optimizer_group_identity'][-1] == {  # type: ignore[index]
		'name': 'spatial_context',
		'parameter_names': ['spatial_context.replacement_token'],
		'lr': 1.0e-3,
	}
	assert runtime['initial_spatial_context_state_sha256'] not in {
		runtime['initial_student_state_sha256'],
		runtime['initial_head_state_sha256'],
	}


def test_invalid_smoke_output_is_quarantined_only_when_requested(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	smoke_root = artifact_root / 'smoke'
	smoke_root.mkdir()
	(smoke_root / 'partial.pt').write_bytes(b'partial')
	config = validation.F3CenterTraceMaskedPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root,
		target_manifest=tmp_path / 'target.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_handoff=tmp_path / 'handoff.json',
		center_trace_masked_smoke_config=tmp_path / 'smoke.yaml',
		center_trace_masked_full_config=tmp_path / 'full.yaml',
		center_trace_masked_embedding_config=tmp_path / 'embedding.yaml',
	)
	full = {'paths': {'output_root': str(artifact_root / 'full')}}
	smoke = {
		'paths': {'output_root': str(smoke_root)},
		'train': {'max_steps': None},
	}
	monkeypatch.setattr(
		validation, '_smoke_config_contract', lambda *_args, **_kwargs: None
	)
	monkeypatch.setattr(
		validation,
		'_runtime_contract',
		lambda *_args, **_kwargs: {'runtime': 'identity'},
	)
	monkeypatch.setattr(
		validation,
		'_checkpoint_evidence',
		lambda *_args, **_kwargs: (_ for _ in ()).throw(
			ValueError('partial smoke checkpoint')
		),
	)

	with pytest.raises(ValueError, match='partial smoke checkpoint'):
		validation._smoke_evidence(  # noqa: SLF001
			config,
			full=full,
			smoke=smoke,
			quarantine_invalid=False,
		)
	assert smoke_root.is_dir()

	with pytest.raises(ValueError, match='partial smoke checkpoint'):
		validation._smoke_evidence(  # noqa: SLF001
			config,
			full=full,
			smoke=smoke,
			quarantine_invalid=True,
		)
	assert not smoke_root.exists()
	quarantines = list(artifact_root.glob('smoke.quarantine.*'))
	assert len(quarantines) == 1
	assert (quarantines[0] / 'partial.pt').read_bytes() == b'partial'


def test_invalid_smoke_output_never_quarantines_colliding_full_root(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	shared_root = artifact_root / 'shared'
	shared_root.mkdir()
	(shared_root / 'partial.pt').write_bytes(b'partial')
	config = validation.F3CenterTraceMaskedPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root,
		target_manifest=tmp_path / 'target.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_handoff=tmp_path / 'handoff.json',
		center_trace_masked_smoke_config=tmp_path / 'smoke.yaml',
		center_trace_masked_full_config=tmp_path / 'full.yaml',
		center_trace_masked_embedding_config=tmp_path / 'embedding.yaml',
	)
	full = {'paths': {'output_root': str(shared_root)}}
	smoke = {'paths': {'output_root': str(shared_root)}}
	monkeypatch.setattr(
		validation,
		'_smoke_config_contract',
		lambda *_args, **_kwargs: (_ for _ in ()).throw(
			ValueError('colliding smoke output root')
		),
	)

	with pytest.raises(ValueError, match='colliding smoke output root'):
		validation._smoke_evidence(  # noqa: SLF001
			config,
			full=full,
			smoke=smoke,
			quarantine_invalid=True,
		)

	assert shared_root.is_dir()
	assert (shared_root / 'partial.pt').read_bytes() == b'partial'
	assert not list(artifact_root.glob('shared.quarantine.*'))


def test_complete_smoke_binding_rejects_replaced_checkpoint(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	full_root = artifact_root / 'full'
	smoke_root = artifact_root / 'smoke'
	smoke_root.mkdir()
	latest = smoke_root / 'latest.pt'
	latest.write_bytes(b'original')
	paths = {
		'target_manifest': artifact_root / 'target.json',
		'hard_full_config': artifact_root / 'hard.yaml',
		'hard_handoff': artifact_root / 'hard_handoff.json',
		'center_trace_masked_smoke_config': artifact_root / 'smoke.yaml',
		'center_trace_masked_full_config': artifact_root / 'full.yaml',
		'center_trace_masked_embedding_config': artifact_root / 'embedding.yaml',
	}
	for path in paths.values():
		path.write_text('{}\n', encoding='utf-8')
	config = validation.F3CenterTraceMaskedPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root,
		**paths,
	)
	runtime = {
		'initial_student_state_sha256': 'a' * 64,
		'initial_head_state_sha256': 'b' * 64,
		'initial_spatial_context_state_sha256': 'c' * 64,
	}
	target_evidence = {
		'target_manifest': {
			'path': str(paths['target_manifest']),
			'sha256': validation.file_sha256(paths['target_manifest']),
		},
		'per_head_target_hashes': {'6': {}, '8': {}, '10': {}},
		'hard_baseline_config_parity': {'candidate_runtime': runtime},
	}
	smoke_evidence = {
		'root': str(smoke_root),
		'latest_path': str(latest),
		'latest_sha256': validation.file_sha256(latest),
		**runtime,
	}
	phase_evidence = {
		'evidence': {**target_evidence, 'smoke': smoke_evidence},
		'binding': validation._smoke_phase_binding(  # noqa: SLF001
			config,
			target_manifest=target_evidence['target_manifest'],
			per_head_target_hashes=target_evidence['per_head_target_hashes'],
			output_root=smoke_root,
			latest_path=latest,
			latest_sha256=smoke_evidence['latest_sha256'],
		),
	}

	validation._validate_smoke_phase_evidence(  # noqa: SLF001
		config,
		phase_evidence=phase_evidence,
		target_evidence=target_evidence,
		full={'paths': {'output_root': str(full_root)}},
		smoke={'paths': {'output_root': str(smoke_root)}},
	)
	latest.write_bytes(b'replaced')
	with pytest.raises(ValueError, match='current output root/checkpoint'):
		validation._validate_smoke_phase_evidence(  # noqa: SLF001
			config,
			phase_evidence=phase_evidence,
			target_evidence=target_evidence,
			full={'paths': {'output_root': str(full_root)}},
			smoke={'paths': {'output_root': str(smoke_root)}},
		)


def test_invalid_handoff_requires_quarantine_and_publishes_atomically(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	path = tmp_path / 'handoff.json'
	path.write_text('{"status": "STALE"}\n', encoding='utf-8')
	new_handoff = {'status': 'PASS', 'schema_version': 1}
	monkeypatch.setattr(
		validation,
		'load_f3_center_trace_masked_pretraining_handoff',
		lambda _path: (_ for _ in ()).throw(ValueError('invalid handoff')),
	)
	with pytest.raises(ValueError, match='pass --quarantine-invalid'):
		validation._publish_handoff(  # noqa: SLF001
			path, new_handoff, only_missing=False, quarantine_invalid=False
		)
	assert path.read_text(encoding='utf-8') == '{"status": "STALE"}\n'

	assert validation._publish_handoff(  # noqa: SLF001
		path, new_handoff, only_missing=False, quarantine_invalid=True
	)
	assert validation._json(path) == new_handoff  # noqa: SLF001
	quarantines = list(tmp_path.glob('handoff.json.quarantine.*'))
	assert len(quarantines) == 1
	assert quarantines[0].read_text(encoding='utf-8') == '{"status": "STALE"}\n'


@pytest.mark.parametrize('phase', ['checkpoints', 'complete'])
def test_later_phases_do_not_revalidate_smoke(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
	config = validation.F3CenterTraceMaskedPretrainingValidationConfig(
		artifact_root=tmp_path,
		experiment_root=tmp_path,
		target_manifest=tmp_path / 'target.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_handoff=tmp_path / 'handoff.json',
		center_trace_masked_smoke_config=tmp_path / 'smoke.yaml',
		center_trace_masked_full_config=tmp_path / 'full.yaml',
		center_trace_masked_embedding_config=tmp_path / 'embedding.yaml',
	)
	config_calls: list[Path] = []
	training = {'paths': {'output_root': str(tmp_path / 'full')}}
	monkeypatch.setattr(
		validation,
		'load_multi_head_target_manifest',
		lambda _path: {},
	)
	monkeypatch.setattr(
		validation,
		'_training_config',
		lambda path: (config_calls.append(path), training)[1],
	)
	monkeypatch.setattr(
		validation,
		'_target_evidence',
		lambda *_args, **_kwargs: {
			'hard_baseline_config_parity': {'candidate_runtime': {}}
		},
	)
	monkeypatch.setattr(
		validation,
		'_smoke_evidence',
		lambda *_args, **_kwargs: pytest.fail('smoke was revalidated'),
	)
	monkeypatch.setattr(
		validation,
		'_checkpoint_evidence',
		lambda *_args, **_kwargs: {
			'root': str(tmp_path / 'full'),
			'selected_sha256': 'a' * 64,
		},
	)
	if phase == 'complete':
		git_state = {
			'git_commit': 'a' * 40,
			'git_status_short': [],
			'git_diff_sha256': 'b' * 64,
		}
		monkeypatch.setattr(
			validation, '_embedding_evidence', lambda *_args, **_kwargs: {}
		)
		monkeypatch.setattr(validation, '_execution_identity', lambda: git_state)
		monkeypatch.setattr(
			validation,
			'_load_execution_evidence',
			lambda _config: {
				'phase': 'smoke',
				'execution': {'before': git_state, 'after': git_state},
			},
		)
		monkeypatch.setattr(validation, '_handoff', lambda _evidence: {})

	result = validation.validate_f3_center_trace_masked_pretraining(
		config, phase=phase, dry_run=True
	)

	assert result.evidence['status'] == 'PASS'
	assert config_calls == [
		config.hard_full_config,
		config.center_trace_masked_full_config,
	]


def test_smoke_config_rejects_scientific_max_steps_drift(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir()
	config = validation.F3CenterTraceMaskedPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=artifact_root,
		target_manifest=tmp_path / 'target.json',
		hard_full_config=tmp_path / 'hard.yaml',
		hard_handoff=tmp_path / 'handoff.json',
		center_trace_masked_smoke_config=tmp_path / 'smoke.yaml',
		center_trace_masked_full_config=tmp_path / 'full.yaml',
		center_trace_masked_embedding_config=tmp_path / 'embedding.yaml',
	)
	full = {
		'paths': {'output_root': str(artifact_root / 'full')},
		'identity': {
			'model_tag': validation._MODEL_TAG,  # noqa: SLF001
			'scientific_identity': {'train': {'max_steps': None}},
			'runtime_identity': {'device': 'auto'},
		},
		'pseudo_targets': {'manifest': str(config.target_manifest)},
		'train': {'device': 'auto', 'max_steps': None},
	}
	smoke = deepcopy(full)
	smoke['paths']['output_root'] = str(artifact_root / 'smoke')  # type: ignore[index]
	smoke['identity']['runtime_identity']['device'] = 'cpu'  # type: ignore[index]
	smoke['train'].update(device='cpu')  # type: ignore[union-attr]
	smoke['identity']['scientific_identity']['train']['max_steps'] = 2  # type: ignore[index]

	with pytest.raises(ValueError, match='smoke/full drifted'):
		validation._smoke_config_contract(  # noqa: SLF001
			config, full=full, smoke=smoke
		)

	smoke['identity']['scientific_identity']['train']['max_steps'] = None  # type: ignore[index]
	validation._smoke_config_contract(  # noqa: SLF001
		config, full=full, smoke=smoke
	)


def test_execution_evidence_records_bound_before_and_after_states(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact_root = tmp_path / 'artifacts'
	experiment_root = artifact_root / 'pretraining'
	experiment_root.mkdir(parents=True)
	paths = {
		'target_manifest': artifact_root / 'target.json',
		'hard_full_config': artifact_root / 'hard.yaml',
		'hard_handoff': artifact_root / 'hard_handoff.json',
		'center_trace_masked_smoke_config': artifact_root / 'smoke.yaml',
		'center_trace_masked_full_config': artifact_root / 'full.yaml',
		'center_trace_masked_embedding_config': artifact_root / 'embedding.yaml',
	}
	for path in paths.values():
		path.write_text('{}\n', encoding='utf-8')
	config = validation.F3CenterTraceMaskedPretrainingValidationConfig(
		artifact_root=artifact_root,
		experiment_root=experiment_root,
		**paths,
	)
	before = {
		'git_commit': 'a' * 40,
		'git_status_short': [' M src/example.py'],
		'git_diff_sha256': 'b' * 64,
	}
	after = {
		'git_commit': 'c' * 40,
		'git_status_short': ['?? artifacts/output.pt'],
		'git_diff_sha256': 'd' * 64,
	}
	monkeypatch.setattr(validation, '_execution_identity', lambda: before)
	validation._start_execution_evidence(config, dry_run=False)  # noqa: SLF001
	monkeypatch.setattr(validation, '_execution_identity', lambda: after)
	execution = validation._update_execution_evidence(  # noqa: SLF001
		config, phase='smoke', dry_run=False
	)

	assert execution == {'before': before, 'after': after}
	record = validation._load_execution_evidence(config)  # noqa: SLF001
	assert record['phase'] == 'smoke'
	assert record['execution'] == execution


def test_repeated_complete_reuses_handoff_and_execution_evidence(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = SimpleNamespace(experiment_root=tmp_path)
	state = {
		'git_commit': 'a' * 40,
		'git_status_short': [],
		'git_diff_sha256': 'b' * 64,
	}
	monkeypatch.setattr(validation, '_execution_identity', lambda: state)
	monkeypatch.setattr(validation, '_execution_binding', lambda _config: {})
	validation._start_execution_evidence(config, dry_run=False)  # noqa: SLF001
	validation._update_execution_evidence(  # noqa: SLF001
		config, phase='smoke', dry_run=False
	)
	validation._update_execution_evidence(  # noqa: SLF001
		config, phase='complete', dry_run=False
	)

	sidecar = validation._execution_evidence_path(config)  # noqa: SLF001
	sidecar_bytes = sidecar.read_bytes()
	sidecar_mtime = sidecar.stat().st_mtime_ns
	os.utime(sidecar, ns=(1_700_000_000_000_000_000,) * 2)
	sidecar_mtime = sidecar.stat().st_mtime_ns
	validation._update_execution_evidence(  # noqa: SLF001
		config, phase='complete', dry_run=False
	)
	assert sidecar.read_bytes() == sidecar_bytes
	assert sidecar.stat().st_mtime_ns == sidecar_mtime

	handoff_path = tmp_path / 'handoff.json'
	handoff = {'status': 'PASS'}
	monkeypatch.setattr(
		validation,
		'load_f3_center_trace_masked_pretraining_handoff',
		validation._json,  # noqa: SLF001
	)
	assert validation._publish_handoff(  # noqa: SLF001
		handoff_path,
		handoff,
		only_missing=False,
		quarantine_invalid=False,
	)
	handoff_bytes = handoff_path.read_bytes()
	os.utime(handoff_path, ns=(1_700_000_000_000_000_000,) * 2)
	handoff_mtime = handoff_path.stat().st_mtime_ns
	assert validation._publish_handoff(  # noqa: SLF001
		handoff_path,
		handoff,
		only_missing=False,
		quarantine_invalid=False,
	)
	assert handoff_path.read_bytes() == handoff_bytes
	assert handoff_path.stat().st_mtime_ns == handoff_mtime


def _smoke_metrics() -> dict[str, object]:
	metrics = dict.fromkeys(validation._SMOKE_METRIC_KEYS, 1.0)  # noqa: SLF001
	metrics['loss_consistency_contribution'] = 0.0
	for head_k in (6, 8, 10):
		for prefix in (
			'loss_prototype_masked',
			'loss_prototype_visible',
			'loss_usage',
			'target_usage_entropy',
			'prototype_usage_entropy',
			'masked_top1_accuracy',
		):
			metrics[f'{prefix}_k{head_k}'] = 1.0
	return {'metrics': metrics}


@pytest.mark.parametrize('head_k', [6, 8, 10])
def test_smoke_metrics_require_bounded_per_k_masked_accuracy(head_k: int) -> None:
	payload = _smoke_metrics()
	payload['metrics'][f'masked_top1_accuracy_k{head_k}'] = 1.01  # type: ignore[index]

	with pytest.raises(ValueError, match='masked top-1 accuracy'):
		validation._validate_center_smoke_metrics(payload)  # noqa: SLF001


def test_smoke_metrics_reject_non_finite_per_k_masked_accuracy() -> None:
	payload = _smoke_metrics()
	payload['metrics']['masked_top1_accuracy_k6'] = float('nan')  # type: ignore[index]

	with pytest.raises(ValueError, match='masked top-1 accuracy'):
		validation._validate_center_smoke_metrics(payload)  # noqa: SLF001


def test_handoff_carries_computed_baseline_parity_and_real_input_evidence() -> None:
	parity = {
		'status': 'PASS',
		'allowed_differences': list(validation._ALLOWED_CONFIG_DIFFERENCES),  # noqa: SLF001
		'hard_runtime': {},
		'candidate_runtime': {},
	}
	real_inputs = {'train_manifest': {'path': 'manifest', 'sha256': 'a' * 64}}
	evidence = {
		'identity': {
			'scientific_identity_sha256': 'a' * 64,
			'initial_student_state_sha256': 'b' * 64,
			'initial_head_state_sha256': 'c' * 64,
			'initial_spatial_context_state_sha256': 'd' * 64,
			'optimizer_group_identity': [],
			'schema_version': 7,
		},
		'selection': {
			'selected': {
				'checkpoint_kind': 'step',
				'epoch': 1,
				'global_step': 2,
				'loss': 1.0,
			},
			'sha256': 'e' * 64,
			'event_count': 1,
			'schema_version': 1,
		},
		'embedding': {},
		'best': {'trainability_summary': {}},
		'selected_path': 'best.pt',
		'selected_sha256': 'f' * 64,
		'latest_path': 'latest.pt',
		'latest_sha256': '0' * 64,
		'training_diagnostics_path': 'metrics.csv',
		'training_diagnostics_sha256': '1' * 64,
		'target_manifest': {'path': 'target', 'sha256': '1' * 64},
		'per_head_target_hashes': {},
		'hard_baseline_config': {'path': 'hard.yaml', 'sha256': '2' * 64},
		'hard_baseline_handoff': {'path': 'hard.json', 'sha256': '3' * 64},
		'hard_baseline_config_parity': parity,
		'real_data_inputs': real_inputs,
		'execution': {},
	}

	handoff = validation._handoff(evidence)  # noqa: SLF001

	assert handoff['targets']['hard_baseline_config_parity'] is parity  # type: ignore[index]
	assert handoff['targets']['real_data_inputs'] is real_inputs  # type: ignore[index]
	assert handoff['training_diagnostics'] == {  # type: ignore[index]
		'path': 'metrics.csv',
		'sha256': '1' * 64,
	}


def test_git_identity_fails_closed_when_commit_or_diff_is_missing(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(validation, '_git_output', lambda *_args: None)
	with pytest.raises(RuntimeError, match='Git commit'):
		validation._execution_identity()  # noqa: SLF001

	monkeypatch.setattr(
		validation,
		'_git_output',
		lambda _root, *args: 'a' * 40 if args == ('rev-parse', 'HEAD') else '',
	)
	monkeypatch.setattr(validation, '_git_bytes', lambda *_args: None)
	with pytest.raises(RuntimeError, match='Git diff'):
		validation._execution_identity()  # noqa: SLF001


def test_real_data_input_evidence_checks_manifest_volume_and_stats(
	tmp_path: Path,
) -> None:
	volume_path = tmp_path / 'volume.npy'
	np.save(volume_path, np.ones((8, 8, 8), dtype=np.float32))
	stats_path = tmp_path / 'stats.json'
	stats_path.write_text(
		json.dumps(
			{
				'survey_id': 'survey',
				'source_path': str(volume_path),
				'grid_order': ['x', 'y', 'z'],
				'clip_low_percentile': 1.0,
				'clip_high_percentile': 99.0,
				'clip_low': -1.0,
				'clip_high': 1.0,
				'median': 0.0,
				'iqr': 1.0,
				'eps': 1.0e-6,
			}
		),
		encoding='utf-8',
	)
	manifest_path = tmp_path / 'manifest.json'
	manifest_path.write_text(
		json.dumps(
			[
				{
					'survey_id': 'survey',
					'root': str(tmp_path),
					'amplitude': {
						'survey_id': 'survey',
						'path': str(volume_path),
						'shape_xyz': [8, 8, 8],
						'dtype': 'float32',
						'grid_order': ['x', 'y', 'z'],
						'normalization_stats_path': str(stats_path),
					},
				}
			]
		),
		encoding='utf-8',
	)
	path_list = tmp_path / 'paths.txt'
	path_list.write_text(f'{volume_path}\n', encoding='utf-8')
	teacher = tmp_path / 'teacher.pt'
	student = tmp_path / 'student.pt'
	teacher.write_bytes(b'teacher')
	student.write_bytes(b'student')

	evidence = validation._real_data_input_evidence(  # noqa: SLF001
		{
			'manifests': {
				'train': str(manifest_path),
				'train_path_list': str(path_list),
			},
			'teacher': {'checkpoint': str(teacher)},
			'student': {'init_checkpoint': str(student)},
		}
	)

	validation._validate_real_data_inputs_evidence(evidence)  # noqa: SLF001
	assert evidence['survey_count'] == 1
