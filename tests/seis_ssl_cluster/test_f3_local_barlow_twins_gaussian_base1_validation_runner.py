from __future__ import annotations

import hashlib
import json
import runpy
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from seis_ssl_cluster.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
	REPOSITORY_ROOT / 'experiments/f3/facies_benchmark_v2/'
	'111_local_barlow_twins_gaussian_view_v1/50_base1ep'
)
RUNNER_PATH = EXPERIMENT_ROOT / 'run_validation.py'
CONFIG_PATH = EXPERIMENT_ROOT / '30_validation/01_candidates.yaml'
HEX64 = 'a' * 64


@pytest.fixture
def runner() -> dict[str, object]:
	return runpy.run_path(str(RUNNER_PATH))


@pytest.fixture
def settings(runner: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> Any:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(REPOSITORY_ROOT / 'artifacts/seis_ssl_cluster'),
	)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(REPOSITORY_ROOT))
	resolver = cast('Any', runner['validation_settings_from_mapping'])
	return resolver(load_config(CONFIG_PATH))


def _source(runner: dict[str, object], tmp_path: Path, *, selected: bool) -> Any:
	constructor = cast('Any', runner['CandidateSource'])
	return constructor(
		candidate_id=(runner['SELECTED_ID'] if selected else runner['LEGACY_ID']),
		role='inherited_selected_view' if selected else 'matched_legacy_control',
		parent_candidate_id=runner['PARENT_SELECTED_ID'] if selected else None,
		base_checkpoint=tmp_path
		/ ('selected_base.pt' if selected else 'legacy_base.pt'),
		final_checkpoint=tmp_path
		/ ('selected_final.pt' if selected else 'legacy_final.pt'),
		embeddings_dir=tmp_path
		/ ('selected_embeddings' if selected else 'legacy_embeddings'),
		view_policy=runner['HORIZONTAL_VIEW_POLICY'] if selected else None,
		gaussian_noise_std=runner['GAUSSIAN_NOISE_STD'] if selected else None,
		base_pretraining_epochs=1,
		continuation_epochs=25,
		selectable=selected,
	)


def _training_config(
	runner: dict[str, object],
	*,
	output_root: Path,
	epochs: int,
	selected: bool,
	continuation_init: Path | None = None,
) -> dict[str, object]:
	config: dict[str, object] = {
		'paths': {'artifact_root': '/artifact', 'output_root': str(output_root)},
		'manifests': {'train': '/manifest', 'train_path_list': '/paths'},
		'data': {'geometry': 'fixed'},
		'zero_mask': {'enabled': True},
		'model': {'encoder_dim': 384},
		'augmentations': (
			{
				'policy': runner['HORIZONTAL_VIEW_POLICY'],
				'horizontal_flip_probability': 0.5,
				'gaussian_noise_std': 0.1,
			}
			if selected
			else {'horizontal_flip_probability': 0.5}
		),
		'barlow_twins': {
			'method': 'local_barlow_twins_3d',
			'local_pairs_per_crop': 128,
			'projector_dim': 384,
		},
		'train': {
			'batch_size': 16,
			'samples_per_epoch': 10_000,
			'epochs': epochs,
			'lr': 1e-4,
		},
	}
	if continuation_init is not None:
		config['continuation'] = {
			'init_checkpoint': str(continuation_init),
			'unfreeze_top_blocks': 1,
		}
	return config


def _checkpoint_payload(
	runner: dict[str, object],
	*,
	config: dict[str, object],
	epochs: int,
	continuation_lineage: object = None,
) -> dict[str, object]:
	return {
		'checkpoint_kind': 'barlow_twins_pretraining',
		'pretraining_method': 'local_barlow_twins_3d',
		'epoch': epochs,
		'global_step': epochs * 625,
		'training_state': (
			runner['EXPECTED_BASE_TRAINING_STATE']
			if epochs == 1
			else runner['EXPECTED_CONTINUATION_TRAINING_STATE']
		),
		'resume_count': 0,
		'amp_enabled': False,
		'scaler_state_dict': None,
		'trained_parameter_prefixes': ['patch_projection.', 'encoder.'],
		'config': config,
		'continuation_lineage': continuation_lineage,
	}


def test_base1_config_is_strict_parent_bound_and_namespaced(
	runner: dict[str, object], settings: Any
) -> None:
	assert settings.parent_final_result_sha256 == runner['PARENT_FINAL_RESULT_SHA256']
	assert tuple(source.candidate_id for source in settings.sources) == (
		runner['SELECTED_ID'],
		runner['LEGACY_ID'],
	)
	assert settings.runs_root.parts[-4:] == (
		'local_barlow_twins_gaussian_view_v1',
		'base1ep',
		'validation',
		'runs',
	)
	assert settings.protocol_lock.parent == settings.runs_root.parent
	assert settings.selection_lock.parent == settings.runs_root.parent
	assert settings.final_result.parent == settings.runs_root.parent
	assert settings.source_by_id(cast('str', runner['SELECTED_ID'])).selectable
	assert not settings.source_by_id(cast('str', runner['LEGACY_ID'])).selectable

	config = deepcopy(load_config(CONFIG_PATH))
	cast('dict[str, object]', config['benchmark'])['random_checkpoint_sha256'] = HEX64
	with pytest.raises(ValueError, match='pinned canonical digest'):
		cast('Any', runner['validation_settings_from_mapping'])(config)


@pytest.mark.parametrize(
	('key', 'value'),
	[
		('base_pretraining_epochs', True),
		('base_pretraining_epochs', 1.0),
		('continuation_epochs', 25.0),
		('selectable', 1),
	],
)
def test_base1_source_config_rejects_numeric_type_aliases(
	runner: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
	key: str,
	value: object,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(REPOSITORY_ROOT / 'artifacts/seis_ssl_cluster'),
	)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(REPOSITORY_ROOT))
	config = deepcopy(load_config(CONFIG_PATH))
	sources = cast('list[object]', config['sources'])
	selected = cast('dict[str, object]', sources[0])
	selected[key] = value
	with pytest.raises(ValueError, match=rf'sources\[0\].{key} must equal'):
		cast('Any', runner['validation_settings_from_mapping'])(config)


def test_parent_audit_reads_selection_payload_not_only_final_summary(
	runner: dict[str, object],
	settings: Any,
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	selection_path = artifact_root / cast(
		'Path', runner['PARENT_SELECTION_RELATIVE_PATH']
	)
	selection_path.parent.mkdir(parents=True)
	selection = {
		'schema_version': 1,
		'selection_lock_type': runner['PARENT_SELECTION_TYPE'],
		'validation_only': True,
		'base_pretraining_epochs': 5,
		'continuation_epochs': 25,
		'parent_selected_candidate_id': runner['GRANDPARENT_SELECTED_ID'],
		'selected_candidate_id': runner['PARENT_SELECTED_ID'],
		'selected_view_policy': runner['HORIZONTAL_VIEW_POLICY'],
		'selected_gaussian_noise_std': 0.1,
		'selection_basis': 'inherited_failed_gaussian25_selection_no_base5_metrics',
		'base5_metric_inputs': [],
	}
	selection_path.write_text(json.dumps(selection), encoding='utf-8')
	selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
	parent_path = artifact_root / cast(
		'Path', runner['PARENT_FINAL_RESULT_RELATIVE_PATH']
	)
	parent = {
		'schema_version': 1,
		'final_result_type': runner['PARENT_FINAL_RESULT_TYPE'],
		'validation_only': True,
		'base_pretraining_epochs': 5,
		'continuation_epochs': 25,
		'passed': False,
		'winner_candidate_id': None,
		'authorizes_next_base_duration': True,
		'authorized_next_base_pretraining_epochs': 1,
		'failure_stage': 'medium_5of5',
		'selection_lock': {
			'path': str(selection_path),
			'parent_selected_candidate_id': runner['GRANDPARENT_SELECTED_ID'],
			'selected_candidate_id': runner['PARENT_SELECTED_ID'],
			'sha256': selection_sha,
		},
	}
	parent_path.write_text(json.dumps(parent), encoding='utf-8')
	parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
	local = replace(
		settings,
		parent_final_result=parent_path,
		parent_final_result_sha256=parent_sha,
	)
	canonical = SimpleNamespace(artifact_root=artifact_root)
	validator = cast('Any', runner['validate_parent_result'])
	monkeypatch.setitem(validator.__globals__, 'PARENT_SELECTION_SHA256', selection_sha)
	identity = validator(local, canonical)
	assert (
		identity['parent_selection_lock']['selected_candidate_id']
		== runner['PARENT_SELECTED_ID']
	)

	selection['selected_candidate_id'] = 'different'
	selection_path.write_text(json.dumps(selection), encoding='utf-8')
	with pytest.raises(ValueError, match='SHA-256 changed'):
		validator(local, canonical)


def test_base_checkpoint_requires_fresh_stage1_and_rejects_lineage(
	runner: dict[str, object], tmp_path: Path
) -> None:
	source = _source(runner, tmp_path, selected=True)
	reference = _training_config(
		runner,
		output_root=tmp_path / 'reference',
		epochs=100,
		selected=False,
	)
	candidate = _training_config(
		runner,
		output_root=source.base_checkpoint.parent,
		epochs=1,
		selected=True,
	)
	payload = _checkpoint_payload(
		runner, config=candidate, epochs=1, continuation_lineage=None
	)
	cast('Any', runner['_validate_candidate_base_checkpoint'])(
		source, payload=payload, reference={'config': reference}
	)

	with_lineage = {**payload, 'continuation_lineage': {'resume_count': 0}}
	with pytest.raises(ValueError, match='must not record continuation lineage'):
		cast('Any', runner['_validate_candidate_base_checkpoint'])(
			source, payload=with_lineage, reference={'config': reference}
		)
	for resumed_value in (1, False):
		resumed = {**payload, 'resume_count': resumed_value}
		with pytest.raises(ValueError, match='resume_count must equal 0'):
			cast('Any', runner['_validate_candidate_base_checkpoint'])(
				source, payload=resumed, reference={'config': reference}
			)


def test_base_checkpoint_rejects_type_only_scientific_metadata_drift(
	runner: dict[str, object], tmp_path: Path
) -> None:
	source = _source(runner, tmp_path, selected=True)
	reference = _training_config(
		runner,
		output_root=tmp_path / 'reference',
		epochs=100,
		selected=False,
	)
	candidate = _training_config(
		runner,
		output_root=source.base_checkpoint.parent,
		epochs=1,
		selected=True,
	)
	payload = _checkpoint_payload(
		runner, config=candidate, epochs=1, continuation_lineage=None
	)
	validator = cast('Any', runner['_validate_candidate_base_checkpoint'])

	for key, value, message in (
		('epoch', True, 'epoch must equal 1'),
		('epoch', 1.0, 'epoch must equal 1'),
		('global_step', 625.0, 'global_step must equal 625'),
	):
		with pytest.raises(ValueError, match=message):
			validator(
				source,
				payload={**payload, key: value},
				reference={'config': reference},
			)

	for key, value in (
		('schema_version', True),
		('schema_version', 1.0),
		('dataset_epoch', False),
		('dataset_epoch', 0.0),
		('completed_epoch', 1),
	):
		malformed = deepcopy(payload)
		cast('dict[str, object]', malformed['training_state'])[key] = value
		with pytest.raises(ValueError, match='completed training_state changed'):
			validator(source, payload=malformed, reference={'config': reference})

	for value in (True, 1.0):
		malformed = deepcopy(payload)
		config_value = cast('dict[str, object]', malformed['config'])
		cast('dict[str, object]', config_value['train'])['epochs'] = value
		with pytest.raises(ValueError, match=r'train\.epochs must equal 1'):
			validator(source, payload=malformed, reference={'config': reference})

	malformed = deepcopy(payload)
	config_value = cast('dict[str, object]', malformed['config'])
	cast('dict[str, object]', config_value['barlow_twins'])[
		'local_pairs_per_crop'
	] = 128.0
	with pytest.raises(ValueError, match='changes the Local Barlow objective'):
		validator(source, payload=malformed, reference={'config': reference})

	malformed = deepcopy(payload)
	config_value = cast('dict[str, object]', malformed['config'])
	cast('dict[str, object]', config_value['model'])['encoder_dim'] = 384.0
	with pytest.raises(ValueError, match='differs from canonical'):
		validator(source, payload=malformed, reference={'config': reference})


def test_continuation_requires_both_freshness_counters_and_exact_base(
	runner: dict[str, object], tmp_path: Path
) -> None:
	source = _source(runner, tmp_path, selected=True)
	source.base_checkpoint.write_bytes(b'fresh base')
	reference = _training_config(
		runner,
		output_root=tmp_path / 'reference_final',
		epochs=25,
		selected=False,
		continuation_init=tmp_path / 'reference_base.pt',
	)
	candidate = _training_config(
		runner,
		output_root=source.final_checkpoint.parent,
		epochs=25,
		selected=True,
		continuation_init=source.base_checkpoint,
	)
	lineage = {
		'schema_version': 1,
		'init_checkpoint': str(source.base_checkpoint),
		'init_checkpoint_sha256': hashlib.sha256(b'fresh base').hexdigest(),
		'resume_count': 0,
	}
	payload = _checkpoint_payload(
		runner, config=candidate, epochs=25, continuation_lineage=lineage
	)
	cast('Any', runner['_validate_candidate_final_checkpoint'])(
		source, payload=payload, reference={'config': reference}
	)

	for mutation, message in (
		({'resume_count': 1}, 'resume_count must equal 0'),
		({'resume_count': False}, 'resume_count must equal 0'),
		(
			{'continuation_lineage': {**lineage, 'resume_count': 1}},
			'lineage resume_count must equal 0',
		),
		(
			{'continuation_lineage': {**lineage, 'resume_count': False}},
			'lineage resume_count must equal 0',
		),
	):
		with pytest.raises(ValueError, match=message):
			cast('Any', runner['_validate_candidate_final_checkpoint'])(
				source,
				payload={**payload, **mutation},
				reference={'config': reference},
			)

	bad_schema = {**lineage, 'schema_version': True}
	with pytest.raises(ValueError, match='lineage schema_version must equal 1'):
		cast('Any', runner['_validate_candidate_final_checkpoint'])(
			source,
			payload={**payload, 'continuation_lineage': bad_schema},
			reference={'config': reference},
		)

	bad_unfreeze_config = deepcopy(candidate)
	cast('dict[str, object]', bad_unfreeze_config['continuation'])[
		'unfreeze_top_blocks'
	] = True
	with pytest.raises(ValueError, match='must unfreeze exactly one top block'):
		cast('Any', runner['_validate_candidate_final_checkpoint'])(
			source,
			payload={**payload, 'config': bad_unfreeze_config},
			reference={'config': reference},
		)


def test_protocol_and_selection_payloads_bind_parent_without_base1_metrics(
	runner: dict[str, object],
) -> None:
	parent = {
		'path': '/parent.json',
		'sha256': HEX64,
		'parent_selection_lock': {
			'selected_candidate_id': runner['PARENT_SELECTED_ID']
		},
	}
	bases = [
		{'candidate_id': runner['SELECTED_ID']},
		{'candidate_id': runner['LEGACY_ID']},
	]
	protocol = cast('Any', runner['_protocol_lock_payload'])(
		parent_result=parent,
		base_inputs=bases,
		benchmark_provenance={'random_checkpoint_sha256': HEX64},
		repository_state={'git_head': 'b' * 40},
		created_at_utc='2026-08-30T00:00:00Z',
		git_head='b' * 40,
	)
	assert protocol['stage_boundary'] == 'completed_fresh_base1_before_continuation'
	assert protocol['base_pretraining_epochs'] == 1
	assert protocol['base_checkpoint_inputs'] == bases

	selection = cast('Any', runner['_selection_lock_payload'])(
		parent_result=parent,
		protocol_lock={'path': '/protocol.json', 'sha256': HEX64},
		repository_state={'git_head': 'b' * 40},
		benchmark_provenance={'random_checkpoint_sha256': HEX64},
		created_at_utc='2026-08-30T00:01:00Z',
		git_head='b' * 40,
	)
	assert selection['base1_metric_inputs'] == []
	assert selection['candidate_id_mapping'] == {
		runner['PARENT_SELECTED_ID']: runner['SELECTED_ID']
	}
	assert selection['matched_legacy_candidate_id'] == runner['LEGACY_ID']


def test_protocol_creation_rejects_later_stage_evidence(
	runner: dict[str, object], settings: Any, tmp_path: Path
) -> None:
	selected = _source(runner, tmp_path / 'selected', selected=True)
	legacy = _source(runner, tmp_path / 'legacy', selected=False)
	local = replace(
		settings,
		sources=(selected, legacy),
		runs_root=tmp_path / 'validation/runs',
		protocol_lock=tmp_path / 'validation/protocol.json',
		selection_lock=tmp_path / 'validation/selection.json',
		final_result=tmp_path / 'validation/final.json',
	)
	selected.final_checkpoint.parent.mkdir(parents=True)
	selected.final_checkpoint.write_bytes(b'later evidence')
	with pytest.raises(ValueError, match='cannot lock protocol after'):
		cast('Any', runner['_reject_pre_protocol_evidence'])(local)


def test_lock_creation_rechecks_for_late_evidence_before_publication(
	runner: dict[str, object],
	settings: Any,
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	parent = {
		'path': '/parent.json',
		'sha256': HEX64,
		'parent_selection_lock': {
			'selected_candidate_id': runner['PARENT_SELECTED_ID']
		},
	}
	repository = {'git_head': 'b' * 40}
	protocol_creator = cast('Any', runner['create_base1_protocol_lock'])
	runtime = cast('dict[str, object]', protocol_creator.__globals__)
	protocol_checks = 0

	def reject_protocol(_settings: object) -> None:
		nonlocal protocol_checks
		protocol_checks += 1
		if protocol_checks == 2:
			raise ValueError('late continuation evidence')

	monkeypatch.setitem(runtime, '_reject_pre_protocol_evidence', reject_protocol)
	monkeypatch.setitem(
		runtime, 'validate_parent_result', lambda _settings, _canonical: parent
	)
	monkeypatch.setitem(
		runtime,
		'_collect_base_audits',
		lambda *_args: [
			{'candidate_id': candidate_id}
			for candidate_id in runner['EXPECTED_SOURCE_IDS']
		],
	)
	shared = cast('dict[str, object]', runtime['_GAUSSIAN25'])
	monkeypatch.setitem(
		shared, '_validate_benchmark_provenance', lambda *_args, **_kwargs: {}
	)
	monkeypatch.setitem(shared, '_git_repository_state', lambda: repository)
	local = replace(settings, protocol_lock=tmp_path / 'protocol.json')
	with pytest.raises(ValueError, match='late continuation evidence'):
		protocol_creator(
			local,
			SimpleNamespace(),
			created_at_utc='2026-08-30T00:00:00Z',
			git_head='b' * 40,
		)
	assert protocol_checks == 2
	assert not local.protocol_lock.exists()

	selection_creator = cast('Any', runner['create_base1_selection_lock'])
	selection_checks = 0

	def reject_selection(_settings: object) -> None:
		nonlocal selection_checks
		selection_checks += 1
		if selection_checks == 2:
			raise ValueError('late validation evidence')

	protocol = {
		'git_head': 'b' * 40,
		'repository_state': repository,
		'benchmark_provenance': {},
	}
	monkeypatch.setitem(runtime, '_reject_pre_selection_evidence', reject_selection)
	monkeypatch.setitem(
		runtime, 'validate_base1_protocol_lock', lambda *_args: protocol
	)
	monkeypatch.setitem(
		runtime,
		'_protocol_identity',
		lambda *_args: {'path': '/protocol.json', 'sha256': HEX64},
	)
	local = replace(settings, selection_lock=tmp_path / 'selection.json')
	with pytest.raises(ValueError, match='late validation evidence'):
		selection_creator(
			local,
			SimpleNamespace(),
			created_at_utc='2026-08-30T00:01:00Z',
		)
	assert selection_checks == 2
	assert not local.selection_lock.exists()


def test_medium_gate_and_exact_reached_cell_sets_are_strict(
	runner: dict[str, object],
) -> None:
	layouts = cast('tuple[str, ...]', runner['LAYOUT_IDS'])
	scores = {
		cast('str', runner['SELECTED_ID']): dict.fromkeys(layouts, 0.51),
		cast('str', runner['LEGACY_ID']): dict.fromkeys(layouts, 0.49),
		'random': dict.fromkeys(layouts, 0.5),
	}
	wins = cast('Any', runner['_medium_5of5_wins'])(scores)
	assert wins == {runner['SELECTED_ID']: True, runner['LEGACY_ID']: False}
	scores[cast('str', runner['SELECTED_ID'])][layouts[0]] = 0.50
	assert not cast('Any', runner['_medium_5of5_wins'])(scores)[runner['SELECTED_ID']]

	closed = cast('Any', runner['_expected_base1_cells'])(medium_gate_open=False)
	opened = cast('Any', runner['_expected_base1_cells'])(medium_gate_open=True)
	assert len(closed) == 10
	assert len(opened) == 30
	assert {cell[0] for cell in closed} == set(runner['EXPECTED_SOURCE_IDS'])
	assert {cell[2] for cell in closed} == {'medium'}


def test_job_replay_binds_prediction_to_all_embedding_file_digests(
	runner: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	reader = cast('Any', runner['_read_candidate_job_evidence'])
	runtime = cast('dict[str, object]', reader.__globals__)
	candidate = _source(runner, tmp_path, selected=True)
	job = SimpleNamespace(
		metrics_path=tmp_path / 'metrics.json',
		output_dir=tmp_path / 'job',
		evaluation_dir=tmp_path / 'job/evaluation',
		layout_id='layout_000',
		data_size='medium',
		model=SimpleNamespace(),
		config=SimpleNamespace(dataset={'name': 'f3'}),
	)
	expected_audit = {
		'candidate_id': candidate.candidate_id,
		'base_checkpoint_sha256': '1' * 64,
		'continuation_init_checkpoint_sha256': '2' * 64,
		'final_checkpoint_sha256': '3' * 64,
		'embeddings_sha256': 'a' * 64,
		'embedding_metadata_sha256': 'b' * 64,
		'valid_tokens_sha256': 'c' * 64,
	}
	validation_order = {'protocol_lock': {'sha256': HEX64}}
	stored_audit = {
		**expected_audit,
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'metrics_path': str(job.metrics_path),
		'validation_order_provenance': validation_order,
	}

	def read_hashed(path: Path, *, label: str) -> tuple[dict[str, object], str]:
		if path == job.metrics_path:
			return {'macro_f1': 0.5}, 'd' * 64
		assert label == 'candidate source audit'
		return stored_audit, 'e' * 64

	monkeypatch.setitem(runtime, '_read_hashed_json', read_hashed)
	monkeypatch.setitem(runtime, '_read_regular_json', lambda *_args, **_kwargs: {})
	monkeypatch.setitem(runtime, '_macro_f1', lambda *_args: 0.5)
	shared = cast('dict[str, object]', runtime['_GAUSSIAN25'])
	monkeypatch.setitem(
		shared, '_validate_job_evaluation_identity', lambda **_kwargs: None
	)
	identities = {
		'embeddings_sha256': 'a' * 64,
		'embedding_metadata_sha256': 'b' * 64,
		'valid_tokens_sha256': 'c' * 64,
	}
	monkeypatch.setattr(
		cast('Any', runtime['five_way_results']),
		'_job_source_identity',
		lambda **_kwargs: identities,
	)
	row = reader(
		job=job,
		candidate=candidate,
		expected_source_audit=expected_audit,
		expected_validation_order=validation_order,
		verify_evaluation_identity=True,
	)
	assert row['embeddings_sha256'] == 'a' * 64
	assert row['embedding_metadata_sha256'] == 'b' * 64
	assert row['valid_tokens_sha256'] == 'c' * 64

	identities['embeddings_sha256'] = 'f' * 64
	with pytest.raises(ValueError, match='differs from source audit'):
		reader(
			job=job,
			candidate=candidate,
			expected_source_audit=expected_audit,
			expected_validation_order=validation_order,
			verify_evaluation_identity=True,
		)


def test_source_audit_records_live_embedding_file_digests(
	runner: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	auditor = cast('Any', runner['audit_candidate_source'])
	runtime = cast('dict[str, object]', auditor.__globals__)
	candidate = _source(runner, tmp_path / 'candidate', selected=True)
	random_dir = tmp_path / 'random'
	output_paths = cast('Any', runner['output_paths'])
	files = output_paths(candidate.embeddings_dir, 'f3')
	random_files = output_paths(random_dir, 'f3')
	files.embeddings.parent.mkdir(parents=True)
	files.embeddings.write_bytes(b'embeddings')
	files.valid_tokens.write_bytes(b'mask')
	files.metadata.write_text(
		json.dumps(
			{
				'pretraining_objective': {
					'augmentations': {
						'policy': runner['HORIZONTAL_VIEW_POLICY'],
						'horizontal_flip_probability': 0.5,
						'gaussian_noise_std': 0.1,
					}
				}
			}
		),
		encoding='utf-8',
	)
	random_files.valid_tokens.parent.mkdir(parents=True)
	random_files.valid_tokens.write_bytes(b'mask')
	random_files.metadata.write_text('{}', encoding='utf-8')
	checkpoint = {
		'base_checkpoint_sha256': '1' * 64,
		'final_checkpoint_sha256': '2' * 64,
		'continuation_init_checkpoint_sha256': '1' * 64,
		'reference_base_checkpoint_sha256': '3' * 64,
		'reference_final_checkpoint_sha256': '4' * 64,
	}
	monkeypatch.setitem(
		runtime, 'audit_candidate_checkpoints', lambda **_kwargs: checkpoint
	)
	sources = cast('Any', runtime['five_way_sources'])
	for name in (
		'_validate_extraction_contract',
		'_validate_shared_identity',
		'_validate_arrays',
		'_validate_objective_identity',
	):
		monkeypatch.setattr(sources, name, lambda *_args, **_kwargs: None)
	monkeypatch.setattr(sources, '_token_grid_shape', lambda *_args: (2, 3, 4))
	monkeypatch.setattr(sources, '_masks_identical', lambda *_args: True)
	monkeypatch.setattr(
		sources,
		'_validate_checkpoint_identity',
		lambda *_args: checkpoint['final_checkpoint_sha256'],
	)
	canonical = SimpleNamespace(
		dataset={'name': 'f3'},
		summary_name='fixed-summary',
		section_layout_dataset_root=tmp_path / 'layouts',
		model_by_id=lambda _model_id: SimpleNamespace(embeddings_dir=random_dir),
	)
	audit = auditor(
		candidate=candidate,
		candidate_model=SimpleNamespace(),
		canonical_config=canonical,
		reference_base_checkpoint=tmp_path / 'reference-base.pt',
		reference_final_checkpoint=tmp_path / 'reference-final.pt',
		protocol_lock_identity={'sha256': HEX64},
		selection_lock_identity={'sha256': HEX64},
	)
	assert audit['embeddings_path'] == str(files.embeddings)
	assert audit['embeddings_sha256'] == hashlib.sha256(b'embeddings').hexdigest()
	assert audit['embedding_metadata_sha256'] == hashlib.sha256(
		files.metadata.read_bytes()
	).hexdigest()
	assert audit['valid_tokens_sha256'] == hashlib.sha256(b'mask').hexdigest()


def test_final_winner_rule_requires_strict_15_of_15(runner: dict[str, object]) -> None:
	chooser = cast('Any', runner['_choose_base1_winner'])
	assert (
		chooser(selected_passed=True, legacy_passed=False, attribution_passed=False)
		== runner['SELECTED_ID']
	)
	assert (
		chooser(selected_passed=False, legacy_passed=True, attribution_passed=True)
		== runner['LEGACY_ID']
	)
	assert (
		chooser(selected_passed=True, legacy_passed=True, attribution_passed=True)
		== runner['SELECTED_ID']
	)
	assert (
		chooser(selected_passed=True, legacy_passed=True, attribution_passed=False)
		== runner['LEGACY_ID']
	)
	assert (
		chooser(
			selected_passed=False,
			legacy_passed=False,
			attribution_passed=False,
		)
		is None
	)


def test_closed_final_result_records_exact_ten_cells_and_is_duration_terminal(
	runner: dict[str, object],
	settings: Any,
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	protocol = {'repository_state': {'git_head': 'b' * 40}}
	selection = {'repository_state': {'git_head': 'b' * 40}}
	creator = cast('Any', runner['create_base1_final_result'])
	runtime = cast('dict[str, object]', creator.__globals__)
	monkeypatch.setitem(
		runtime, 'validate_base1_protocol_lock', lambda _settings, _canonical: protocol
	)
	monkeypatch.setitem(
		runtime,
		'validate_base1_selection_lock',
		lambda _settings, _canonical: selection,
	)
	monkeypatch.setitem(
		runtime,
		'_protocol_identity',
		lambda _settings, _protocol: {'path': '/protocol', 'sha256': HEX64},
	)
	monkeypatch.setitem(
		runtime,
		'_selection_identity',
		lambda _settings, _selection: {'path': '/selection', 'sha256': HEX64},
	)
	monkeypatch.setitem(
		runtime,
		'validate_parent_result',
		lambda _settings, _canonical: {'path': '/parent', 'sha256': HEX64},
	)
	shared = cast('dict[str, object]', runtime['_GAUSSIAN25'])
	monkeypatch.setitem(
		shared,
		'_validate_benchmark_provenance',
		lambda _settings, _canonical, *, verify_files: {'verified': verify_files},
	)
	monkeypatch.setitem(
		shared,
		'_read_random_job_evidence',
		lambda _canonical, *, layout_id, data_size: {
			'candidate_id': 'random',
			'layout_id': layout_id,
			'data_size': data_size,
			'macro_f1': 0.50,
		},
	)
	monkeypatch.setattr(
		cast('Any', runtime['five_way_sources']),
		'audit_f3_lithology_five_way_sources',
		lambda _canonical: None,
	)
	monkeypatch.setitem(
		runtime,
		'_validate_medium_random_gate',
		lambda **_kwargs: {
			'gate_open': False,
			'selected_wins_over_random': False,
			'legacy_wins_over_random': False,
			'inputs': [
				{
					'candidate_id': 'random',
					'layout_id': f'layout_{index % 5:03d}',
					'data_size': 'medium',
					'metrics_path': f'/metrics/{index}.json',
					'metrics_sha256': HEX64,
				}
				for index in range(15)
			],
		},
	)
	monkeypatch.setitem(
		runtime, '_validate_exact_candidate_cell_set', lambda **_kwargs: None
	)

	class _CandidateConfig:
		def model_by_id(self, model_id: str) -> object:
			return SimpleNamespace(model_id=model_id)

	monkeypatch.setitem(
		runtime,
		'_candidate_config',
		lambda _canonical, **_kwargs: _CandidateConfig(),
	)
	monkeypatch.setitem(
		runtime, 'audit_candidate_source', lambda **_kwargs: {'passed': True}
	)
	monkeypatch.setattr(
		cast('Any', runtime['five_way_runner']),
		'resolve_f3_lithology_five_way_job',
		lambda _config, *, model, layout, size: SimpleNamespace(
			model=model, layout_id=layout, data_size=size
		),
	)
	monkeypatch.setitem(
		runtime,
		'_read_candidate_job_evidence',
		lambda *, job, candidate, **_kwargs: {
			'candidate_id': candidate.candidate_id,
			'layout_id': job.layout_id,
			'data_size': job.data_size,
			'macro_f1': 0.40,
		},
	)
	local = replace(settings, final_result=tmp_path / 'base1_final.json')
	result = creator(
		local,
		SimpleNamespace(),
		created_at_utc='2026-08-30T00:00:00Z',
	)
	assert result['passed'] is False
	assert result['winner_candidate_id'] is None
	assert result['authorizes_next_base_duration'] is False
	assert result['authorized_next_base_pretraining_epochs'] is None
	assert result['failure_stage'] == 'medium_5of5'
	assert len(result['exact_expected_candidate_cells']) == 10
	assert len(result['candidate_inputs']) == 10
	assert len(result['random_inputs']) == 5
	assert json.loads(local.final_result.read_text(encoding='utf-8')) == result


def test_cli_modes_are_mutually_scoped(runner: dict[str, object]) -> None:
	parser = cast('Any', runner['build_parser'])()
	validator = cast('Any', runner['_validate_cli_arguments'])
	validator(parser.parse_args(['--config', 'x', '--audit-parent-only']))
	validator(
		parser.parse_args(
			[
				'--config',
				'x',
				'--audit-base-checkpoint-only',
				'--candidate',
				cast('str', runner['SELECTED_ID']),
			]
		)
	)
	with pytest.raises(ValueError, match='do not accept job arguments'):
		validator(
			parser.parse_args(
				[
					'--config',
					'x',
					'--create-protocol-lock',
					'--candidate',
					cast('str', runner['SELECTED_ID']),
				]
			)
		)
