"""Contracts for the experiment-local Gaussian-view validation runner."""

from __future__ import annotations

import csv
import runpy
from argparse import Namespace
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
)
from seis_ssl_cluster.f3.lithology import five_way_results, five_way_runner

if TYPE_CHECKING:
	from types import FunctionType

EXPERIMENT_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/'
	'111_local_barlow_twins_gaussian_view_v1'
)
RUNNER = EXPERIMENT_ROOT / 'run_validation.py'
CONFIG = EXPERIMENT_ROOT / '30_validation/01_candidates.yaml'
CANDIDATE_IDS = (
	'local_barlow_twins_gaussian_noise_std005',
	'local_barlow_twins_gaussian_noise_std010',
	'local_barlow_twins_identity_gaussian_noise_std010',
)
FORCED_STD005_ID, FORCED_STD010_ID, IDENTITY_STD010_ID = CANDIDATE_IDS
LEGACY_CONTROL_ID = 'local_barlow_twins_legacy_flip_25ep'
LEGACY_TRAINING_CONFIG = (
	EXPERIMENT_ROOT / '10_stage1/legacy_flip_25ep/01_matched_25ep.yaml'
)
LEGACY_CONTINUATION_CONFIG = (
	EXPERIMENT_ROOT / '15_stage2/legacy_flip_25ep/01_continue_25ep.yaml'
)
REFERENCE_TRAINING_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/'
	'22_local_barlow_twins_v1/02_full_100ep.yaml'
)
REFERENCE_CONTINUATION_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/'
	'110_lithology_mae_local_bt_five_way_v1/10_stage2/'
	'local_bt100/local_bt_continue/02_full_25ep.yaml'
)


@pytest.fixture
def runner_namespace() -> dict[str, object]:
	"""Load the standalone experiment script without executing its CLI."""
	return runpy.run_path(str(RUNNER))


@pytest.fixture(autouse=True)
def _environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts')
	)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))


def test_candidate_validation_config_is_strict_and_validation_namespaced(  # noqa: PLR0915
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	settings = resolver(load_config(CONFIG))

	assert tuple(item.candidate_id for item in settings.candidates) == CANDIDATE_IDS
	assert [item.gaussian_noise_std for item in settings.candidates] == [
		0.05,
		0.1,
		0.1,
	]
	assert [item.view_policy for item in settings.candidates] == [
		'horizontal_flip_gaussian_noise_v1',
		'horizontal_flip_gaussian_noise_v1',
		'identity_gaussian_noise_v1',
	]
	assert tuple(item.candidate_id for item in settings.controls) == (
		LEGACY_CONTROL_ID,
	)
	assert settings.controls[0].gaussian_noise_std is None
	assert settings.controls[0].selectable is False
	assert all(item.selectable for item in settings.candidates)
	assert {item.base_pretraining_epochs for item in settings.candidates} == {25}
	assert {item.continuation_epochs for item in settings.candidates} == {25}
	assert all(
		item.embeddings_dir.parent.name == item.candidate_id
		for item in settings.candidates
	)
	assert settings.runs_root.parts[-2:] == ('validation', 'runs')
	assert settings.selection_lock == (
		settings.runs_root.parent / 'gaussian25_selection_lock.json'
	)
	assert settings.protocol_lock == (
		settings.runs_root.parent / 'gaussian25_protocol_lock.json'
	)
	assert settings.final_result == (
		settings.runs_root.parent / 'gaussian25_final_result.json'
	)
	assert settings.reference_base_checkpoint_sha256 == (
		'84550ed658166e8e6a40cd664e2e9ffbeab0c12d6917006abb417cd25e228ac0'
	)
	assert settings.reference_final_checkpoint_sha256 == (
		'1c5312244f290dbfdcf2688ffa9fa8b5c64452ade162d5335be1bb8a0e256291'
	)
	assert settings.random_checkpoint_sha256 == (
		'6548d52446e7d6b9b57acd2bd39a8389a76bc5df55b52a9eda0472eb182a438c'
	)
	assert settings.canonical_five_way_config_sha256 == (
		'285b0233ff82fe83808f82e929b611f570a67f01fa983ef191dda23d1858061b'
	)
	assert settings.canonical_comparison_sha256 == (
		'b135122a7db2b6b359817096ac546f99d4e4fac1ee003a99ce7289c0445cf913'
	)
	assert settings.pretraining_manifest_sha256 == (
		'c5dbc3a66a5c2eed0ec5df8745f8bf5a461b1e2e66156700091f1a751bdc0ef5'
	)
	assert settings.pretraining_path_list_sha256 == (
		'b52fd5e0c57edb2d2158be12b94046b554b5e6e13ba17008321bcdbe0ae2acb1'
	)

	tampered = deepcopy(load_config(CONFIG))
	tampered['candidates'][0]['base_pretraining_epochs'] = 100
	with pytest.raises(ValueError, match='exactly 25 epochs'):
		resolver(tampered)
	tampered_continuation = deepcopy(load_config(CONFIG))
	tampered_continuation['candidates'][0]['continuation_epochs'] = 100
	with pytest.raises(ValueError, match='exactly 25 epochs'):
		resolver(tampered_continuation)

	duplicate = deepcopy(load_config(CONFIG))
	duplicate['candidates'].append(deepcopy(duplicate['candidates'][0]))
	with pytest.raises(ValueError, match='must define exactly'):
		resolver(duplicate)

	tampered_policy = deepcopy(load_config(CONFIG))
	tampered_policy['candidates'][2]['view_policy'] = (
		'horizontal_flip_gaussian_noise_v1'
	)
	with pytest.raises(ValueError, match='view_policy does not match'):
		resolver(tampered_policy)

	tampered_reference_sha = deepcopy(load_config(CONFIG))
	tampered_reference_sha['benchmark']['reference_base_checkpoint_sha256'] = (
		'0' * 64
	)
	with pytest.raises(ValueError, match='pinned canonical'):
		resolver(tampered_reference_sha)
	tampered_final_sha = deepcopy(load_config(CONFIG))
	tampered_final_sha['benchmark']['reference_final_checkpoint_sha256'] = '0' * 64
	with pytest.raises(ValueError, match='pinned canonical'):
		resolver(tampered_final_sha)
	for key in (
		'canonical_five_way_config_sha256',
		'random_checkpoint_sha256',
		'canonical_comparison_sha256',
		'pretraining_manifest_sha256',
		'pretraining_path_list_sha256',
	):
		tampered_pin = deepcopy(load_config(CONFIG))
		tampered_pin['benchmark'][key] = '0' * 64
		with pytest.raises(ValueError, match='pinned canonical'):
			resolver(tampered_pin)


def test_control_identity_artifacts_and_non_selectability_are_strict(
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	raw = load_config(CONFIG)

	duplicate_base = deepcopy(raw)
	duplicate_base['controls'][0]['base_checkpoint'] = (
		duplicate_base['candidates'][0]['base_checkpoint']
	)
	with pytest.raises(ValueError, match='base checkpoint paths must be unique'):
		resolver(duplicate_base)

	duplicate_final = deepcopy(raw)
	duplicate_final['controls'][0]['final_checkpoint'] = (
		duplicate_final['candidates'][0]['final_checkpoint']
	)
	with pytest.raises(ValueError, match='final checkpoint paths must be unique'):
		resolver(duplicate_final)

	overlapping_lineage = deepcopy(raw)
	overlapping_lineage['controls'][0]['final_checkpoint'] = overlapping_lineage[
		'controls'
	][0]['base_checkpoint']
	with pytest.raises(ValueError, match='must not overlap'):
		resolver(overlapping_lineage)

	duplicate_embeddings = deepcopy(raw)
	duplicate_embeddings['controls'][0]['embeddings_dir'] = (
		duplicate_embeddings['candidates'][0]['embeddings_dir']
	)
	with pytest.raises(ValueError, match='embedding paths must be unique'):
		resolver(duplicate_embeddings)

	shortened_model_id = deepcopy(raw)
	shortened_model_id['controls'][0]['embeddings_dir'] = str(
		Path(shortened_model_id['controls'][0]['embeddings_dir']).parent
		/ 'legacy_flip'
		/ 'overlap_x64'
	)
	with pytest.raises(ValueError, match='must use its full model ID'):
		resolver(shortened_model_id)

	selectable_control = deepcopy(raw)
	selectable_control['controls'][0]['selectable'] = True
	with pytest.raises(ValueError, match='selectable must be false'):
		resolver(selectable_control)


def test_reference_checkpoint_path_and_sha_are_pinned(
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	raw = load_config(CONFIG)
	settings = resolver(raw)
	canonical_resolver(settings)

	tampered = deepcopy(raw)
	tampered['benchmark']['reference_base_checkpoint'] = str(
		Path(tampered['benchmark']['reference_base_checkpoint']).with_name(
			'other.pt'
		)
	)
	with pytest.raises(ValueError, match='pinned canonical Local Barlow base path'):
		canonical_resolver(resolver(tampered))

	tampered_final = deepcopy(raw)
	tampered_final['benchmark']['reference_final_checkpoint'] = str(
		Path(tampered_final['benchmark']['reference_final_checkpoint']).with_name(
			'other.pt'
		)
	)
	with pytest.raises(ValueError, match='continuation path'):
		canonical_resolver(resolver(tampered_final))


def test_base_only_audit_precedes_and_needs_no_continuation(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	auditor = cast(
		'FunctionType', runner_namespace['audit_candidate_base_checkpoint']
	)
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	candidate = settings.candidate_by_id(FORCED_STD005_ID)
	for path, content in (
		(candidate.base_checkpoint, b'candidate-base'),
		(settings.reference_base_checkpoint, b'reference-base'),
	):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(content)
	candidate_config = resolve_barlow_twins_training_config(
		load_config(
			EXPERIMENT_ROOT
			/ '10_stage1/gaussian_noise_std005/01_screen_25ep.yaml'
		)
	)
	reference_config = resolve_barlow_twins_training_config(
		load_config(REFERENCE_TRAINING_CONFIG)
	)
	payload = {
		'checkpoint_kind': 'barlow_twins_pretraining',
		'pretraining_method': 'local_barlow_twins_3d',
		'epoch': 25,
		'global_step': 15_625,
		'training_state': {
			'schema_version': 1,
			'stage': 'barlow_twins_training',
			'resume_boundary': 'epoch',
			'dataset_epoch': 24,
			'completed_epoch': True,
		},
		'amp_enabled': False,
		'scaler_state_dict': None,
		'trained_parameter_prefixes': ['patch_projection.', 'encoder.'],
		'config': candidate_config,
	}

	monkeypatch.setitem(
		auditor.__globals__,
		'load_checkpoint_metadata_without_weights',
		lambda path: payload if path == candidate.base_checkpoint else {
			'config': reference_config
		},
	)
	monkeypatch.setitem(
		auditor.__globals__,
		'file_sha256',
		lambda path: (
			settings.reference_base_checkpoint_sha256
			if path == settings.reference_base_checkpoint
			else 'a' * 64
		),
	)
	result = auditor(
		candidate=candidate,
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
	)

	assert result['audit_type'] == 'pre_continuation_base_checkpoint_only'
	assert result['base_checkpoint_sha256'] == 'a' * 64
	assert result['final_checkpoint_required'] is False
	assert not candidate.final_checkpoint.exists()
	assert not candidate.embeddings_dir.exists()


def test_checkpoint_only_audit_needs_no_embeddings(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	auditor = cast('FunctionType', runner_namespace['audit_candidate_checkpoints'])
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	candidate = settings.candidate_by_id(FORCED_STD005_ID)
	for path, content in (
		(candidate.base_checkpoint, b'candidate-base'),
		(candidate.final_checkpoint, b'candidate-final'),
		(settings.reference_base_checkpoint, b'reference-base'),
		(settings.reference_final_checkpoint, b'reference-final'),
	):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(content)
	candidate_base_config = resolve_barlow_twins_training_config(
		load_config(
			EXPERIMENT_ROOT
			/ '10_stage1/gaussian_noise_std005/01_screen_25ep.yaml'
		)
	)
	candidate_final_config = resolve_barlow_twins_training_config(
		load_config(
			EXPERIMENT_ROOT
			/ '15_stage2/gaussian_noise_std005/01_continue_25ep.yaml'
		)
	)
	reference_base_config = resolve_barlow_twins_training_config(
		load_config(REFERENCE_TRAINING_CONFIG)
	)
	reference_final_config = resolve_barlow_twins_training_config(
		load_config(REFERENCE_CONTINUATION_CONFIG)
	)
	candidate_base_payload = {
		'checkpoint_kind': 'barlow_twins_pretraining',
		'pretraining_method': 'local_barlow_twins_3d',
		'epoch': 25,
		'global_step': 15_625,
		'training_state': {
			'schema_version': 1,
			'stage': 'barlow_twins_training',
			'resume_boundary': 'epoch',
			'dataset_epoch': 24,
			'completed_epoch': True,
		},
		'amp_enabled': False,
		'scaler_state_dict': None,
		'trained_parameter_prefixes': ['patch_projection.', 'encoder.'],
		'config': candidate_base_config,
	}
	candidate_final_payload = {
		**candidate_base_payload,
		'config': candidate_final_config,
		'amp_enabled': False,
		'scaler_state_dict': None,
		'trained_parameter_prefixes': ['patch_projection.', 'encoder.'],
		'continuation_lineage': {
			'schema_version': 1,
			'init_checkpoint': str(candidate.base_checkpoint),
			'init_checkpoint_sha256': 'a' * 64,
			'resume_count': 0,
		},
	}

	def load_metadata(path: Path) -> dict[str, object]:
		return {
			candidate.base_checkpoint: candidate_base_payload,
			candidate.final_checkpoint: candidate_final_payload,
			settings.reference_base_checkpoint: {'config': reference_base_config},
			settings.reference_final_checkpoint: {'config': reference_final_config},
		}[path]

	def sha256(path: Path) -> str:
		return {
			candidate.base_checkpoint: 'a' * 64,
			candidate.final_checkpoint: 'b' * 64,
			settings.reference_base_checkpoint: (
				settings.reference_base_checkpoint_sha256
			),
			settings.reference_final_checkpoint: (
				settings.reference_final_checkpoint_sha256
			),
		}[path]

	monkeypatch.setitem(
		auditor.__globals__, 'load_checkpoint_metadata_without_weights', load_metadata
	)
	monkeypatch.setitem(auditor.__globals__, 'file_sha256', sha256)
	result = auditor(
		candidate=candidate,
		canonical_config=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
	)

	assert result['passed'] is True
	assert result['embeddings_required'] is False
	assert result['base_checkpoint_sha256'] == 'a' * 64
	assert result['final_checkpoint_sha256'] == 'b' * 64
	assert result['base_pretraining_epochs'] == 25
	assert result['continuation_epochs'] == 25
	assert not candidate.embeddings_dir.exists()


def test_legacy_checkpoint_audit_requires_exact_flip_epoch_and_step(
	runner_namespace: dict[str, object],
) -> None:
	settings_resolver = cast(
		'object', runner_namespace['validation_settings_from_mapping']
	)
	validator = cast(
		'object', runner_namespace['_validate_candidate_base_checkpoint']
	)
	settings = settings_resolver(load_config(CONFIG))
	control = settings.source_by_id(LEGACY_CONTROL_ID)
	control_config = resolve_barlow_twins_training_config(
		load_config(LEGACY_TRAINING_CONFIG)
	)
	reference_config = resolve_barlow_twins_training_config(
		load_config(REFERENCE_TRAINING_CONFIG)
	)
	payload = {
		'checkpoint_kind': 'barlow_twins_pretraining',
		'pretraining_method': 'local_barlow_twins_3d',
		'epoch': 25,
		'global_step': 15_625,
		'training_state': {
			'schema_version': 1,
			'stage': 'barlow_twins_training',
			'resume_boundary': 'epoch',
			'dataset_epoch': 24,
			'completed_epoch': True,
		},
		'amp_enabled': False,
		'scaler_state_dict': None,
		'trained_parameter_prefixes': ['patch_projection.', 'encoder.'],
		'config': control_config,
	}
	reference_payload = {'config': reference_config}
	expected_flip = {'horizontal_flip_probability': 0.5}

	validator(
		control,
		payload=payload,
		reference_payload=reference_payload,
		expected_augmentations=expected_flip,
	)

	tampered_flip = deepcopy(payload)
	tampered_flip['config']['augmentations'] = {
		'horizontal_flip_probability': 0.6
	}
	with pytest.raises(ValueError, match='unexpected augmentations'):
		validator(
			control,
			payload=tampered_flip,
			reference_payload=reference_payload,
			expected_augmentations=expected_flip,
		)

	with pytest.raises(ValueError, match='outside duration or output root'):
		validator(
			control,
			payload=tampered_flip,
			reference_payload=reference_payload,
			expected_augmentations={'horizontal_flip_probability': 0.6},
		)

	tampered_step = deepcopy(payload)
	tampered_step['global_step'] = 15_624
	with pytest.raises(ValueError, match='must equal 15625'):
		validator(
			control,
			payload=tampered_step,
			reference_payload=reference_payload,
			expected_augmentations=expected_flip,
		)

	tampered_epoch = deepcopy(payload)
	tampered_epoch['epoch'] = 24
	with pytest.raises(ValueError, match='end at epoch 25'):
		validator(
			control,
			payload=tampered_epoch,
			reference_payload=reference_payload,
			expected_augmentations=expected_flip,
		)


def test_final_checkpoint_audit_binds_canonical_continuation_and_base_lineage(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	settings_resolver = cast(
		'object', runner_namespace['validation_settings_from_mapping']
	)
	validator = cast(
		'object', runner_namespace['_validate_candidate_final_checkpoint']
	)
	settings = settings_resolver(load_config(CONFIG))
	control = settings.source_by_id(LEGACY_CONTROL_ID)
	control_config = resolve_barlow_twins_training_config(
		load_config(LEGACY_CONTINUATION_CONFIG)
	)
	reference_config = resolve_barlow_twins_training_config(
		load_config(REFERENCE_CONTINUATION_CONFIG)
	)
	payload = {
		'checkpoint_kind': 'barlow_twins_pretraining',
		'pretraining_method': 'local_barlow_twins_3d',
		'epoch': 25,
		'global_step': 15_625,
		'training_state': {
			'schema_version': 1,
			'stage': 'barlow_twins_training',
			'resume_boundary': 'epoch',
			'dataset_epoch': 24,
			'completed_epoch': True,
		},
		'amp_enabled': False,
		'scaler_state_dict': None,
		'trained_parameter_prefixes': ['patch_projection.', 'encoder.'],
		'config': control_config,
		'continuation_lineage': {
			'schema_version': 1,
			'init_checkpoint': str(control.base_checkpoint),
			'init_checkpoint_sha256': 'a' * 64,
			'resume_count': 0,
		},
	}
	reference_payload = {'config': reference_config}
	expected_flip = {'horizontal_flip_probability': 0.5}
	monkeypatch.setitem(validator.__globals__, 'file_sha256', lambda _: 'a' * 64)

	validator(
		control,
		payload=payload,
		reference_payload=reference_payload,
		expected_augmentations=expected_flip,
	)

	tampered_init = deepcopy(payload)
	tampered_init['config']['continuation']['init_checkpoint'] = '/wrong/base.pt'
	with pytest.raises(ValueError, match='exact base checkpoint'):
		validator(
			control,
			payload=tampered_init,
			reference_payload=reference_payload,
			expected_augmentations=expected_flip,
		)

	tampered_lr = deepcopy(payload)
	tampered_lr['config']['train']['lr'] = 1e-4
	with pytest.raises(ValueError, match='outside augmentation'):
		validator(
			control,
			payload=tampered_lr,
			reference_payload=reference_payload,
			expected_augmentations=expected_flip,
		)

	tampered_state = deepcopy(payload)
	tampered_state['training_state']['dataset_epoch'] = 23
	with pytest.raises(ValueError, match='exact completed epoch state'):
		validator(
			control,
			payload=tampered_state,
			reference_payload=reference_payload,
			expected_augmentations=expected_flip,
		)

	tampered_prefixes = deepcopy(payload)
	tampered_prefixes['trained_parameter_prefixes'] = ['encoder.']
	with pytest.raises(ValueError, match='differ from canonical'):
		validator(
			control,
			payload=tampered_prefixes,
			reference_payload=reference_payload,
			expected_augmentations=expected_flip,
		)


def test_candidate_job_reuses_canonical_downstream_mappings(
	runner_namespace: dict[str, object],
) -> None:
	settings_resolver = cast(
		'object', runner_namespace['validation_settings_from_mapping']
	)
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	candidate_config_builder = cast('object', runner_namespace['_candidate_config'])
	settings = settings_resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	candidate = settings.candidate_by_id(CANDIDATE_IDS[0])
	candidate_config = candidate_config_builder(
		canonical, candidate=candidate, runs_root=settings.runs_root
	)
	reference_job = five_way_runner.resolve_f3_lithology_five_way_job(
		canonical,
		model='local_barlow_twins',
		layout='layout_001',
		size='medium',
	)
	candidate_job = five_way_runner.resolve_f3_lithology_five_way_job(
		candidate_config,
		model=candidate.candidate_id,
		layout='layout_001',
		size='medium',
	)
	reference_decoder = five_way_runner._decoder_config_mapping(  # noqa: SLF001
		reference_job
	)
	candidate_decoder = five_way_runner._decoder_config_mapping(  # noqa: SLF001
		candidate_job
	)
	normalized_decoder = deepcopy(candidate_decoder)
	normalized_decoder['model'] = reference_decoder['model']
	normalized_decoder['embeddings'] = reference_decoder['embeddings']
	normalized_decoder['outputs'] = reference_decoder['outputs']
	assert normalized_decoder == reference_decoder
	assert candidate_job.model.checkpoint == candidate.final_checkpoint
	assert candidate_job.output_dir.parts[-3:] == (
		f'model={candidate.candidate_id}',
		'layout=layout_001',
		'size=medium',
	)

	reference_inference = five_way_runner._inference_config_mapping(  # noqa: SLF001
		reference_job
	)
	candidate_inference = five_way_runner._inference_config_mapping(  # noqa: SLF001
		candidate_job
	)
	normalized_inference = deepcopy(candidate_inference)
	normalized_inference['model'] = reference_inference['model']
	normalized_inference['embeddings'] = reference_inference['embeddings']
	normalized_inference['decoder'] = reference_inference['decoder']
	normalized_inference['outputs'] = reference_inference['outputs']
	assert normalized_inference == reference_inference
	reference_evaluation = five_way_runner._evaluation_config_mapping(  # noqa: SLF001
		reference_job
	)
	candidate_evaluation = five_way_runner._evaluation_config_mapping(  # noqa: SLF001
		candidate_job
	)
	normalized_evaluation = deepcopy(candidate_evaluation)
	normalized_evaluation['voxel_predictions'] = reference_evaluation[
		'voxel_predictions'
	]
	normalized_evaluation['outputs'] = reference_evaluation['outputs']
	assert normalized_evaluation == reference_evaluation

	control = settings.source_by_id(LEGACY_CONTROL_ID)
	control_config = candidate_config_builder(
		canonical, candidate=control, runs_root=settings.runs_root
	)
	control_job = five_way_runner.resolve_f3_lithology_five_way_job(
		control_config,
		model=LEGACY_CONTROL_ID,
		layout='layout_001',
		size='medium',
	)
	assert control_job.model.model_id == LEGACY_CONTROL_ID
	assert control_job.model.checkpoint == control.final_checkpoint
	assert control_job.output_dir.parts[-3:] == (
		f'model={LEGACY_CONTROL_ID}',
		'layout=layout_001',
		'size=medium',
	)


def _scores(
	forced_005: float,
	forced_010: float,
	identity_010: float,
) -> dict[str, dict[str, float]]:
	return {
		candidate_id: dict.fromkeys(five_way_runner.LAYOUT_IDS, value)
		for candidate_id, value in (
			(FORCED_STD005_ID, forced_005),
			(FORCED_STD010_ID, forced_010),
			(IDENTITY_STD010_ID, identity_010),
		)
	}


def _selection_evidence(
	ranker: object,
	*,
	forced_005: float = 0.5,
	forced_010: float = 0.6,
	identity_010: float = 0.7,
) -> dict[str, object]:
	ranked = ranker(_scores(forced_005, forced_010, identity_010))
	rows = [
		{
			'candidate_id': candidate_id,
			'layout_id': layout_id,
			'data_size': 'medium',
			'macro_f1': _scores(
				forced_005, forced_010, identity_010
			)[candidate_id][layout_id],
			'base_checkpoint_sha256': 'c' * 64,
			'continuation_init_checkpoint_sha256': 'c' * 64,
			'final_checkpoint_sha256': 'd' * 64,
			'metrics_path': f'/metrics/{candidate_id}/{layout_id}.json',
			'metrics_sha256': 'a' * 64,
			'candidate_audit_path': f'/audits/{candidate_id}/{layout_id}.json',
			'candidate_audit_sha256': 'b' * 64,
		}
		for candidate_id in CANDIDATE_IDS
		for layout_id in five_way_runner.LAYOUT_IDS
	]
	return {'inputs': rows, **ranked}


def test_selection_ranking_uses_three_by_five_matrix_and_fixed_ties(
	runner_namespace: dict[str, object],
) -> None:
	ranker = cast('object', runner_namespace['_rank_candidate_scores'])

	all_tied = ranker(_scores(0.5, 0.5, 0.5))
	assert all_tied['selected_candidate_id'] == FORCED_STD005_ID
	upper_tied = ranker(_scores(0.5, 0.6, 0.6))
	assert upper_tied['selected_candidate_id'] == FORCED_STD010_ID
	identity_wins = ranker(_scores(0.5, 0.6, 0.6000000000001))
	assert identity_wins['selected_candidate_id'] == IDENTITY_STD010_ID
	contrast = identity_wins['fixed_strength_geometry_contrast']
	assert contrast['mean_macro_f1_delta'] == pytest.approx(1.0e-13)

	missing_layout = _scores(0.5, 0.6, 0.7)
	del missing_layout[IDENTITY_STD010_ID]['layout_004']
	with pytest.raises(ValueError, match='exactly five layouts'):
		ranker(missing_layout)


def test_selection_collection_reads_exact_fixed_matrix_independent_of_yaml_order(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	collector = cast('FunctionType', runner_namespace['_collect_selection_evidence'])
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	calls: list[tuple[str, str, str]] = []

	monkeypatch.setitem(
		collector.__globals__,
		'_validate_reference_checkpoints',
		lambda **_: None,
	)
	monkeypatch.setitem(collector.__globals__, 'file_sha256', lambda _: 'a' * 64)
	monkeypatch.setitem(
		collector.__globals__,
		'audit_candidate_source',
		lambda **_: {
			'base_checkpoint_sha256': 'a' * 64,
			'final_checkpoint_sha256': 'a' * 64,
		},
	)

	def read_evidence(*, job: object, **_: object) -> dict[str, object]:
		calls.append((job.model.model_id, job.layout_id, job.data_size))
		value = {
			FORCED_STD005_ID: 0.5,
			FORCED_STD010_ID: 0.6,
			IDENTITY_STD010_ID: 0.7,
		}[job.model.model_id]
		return {'macro_f1': value}

	monkeypatch.setitem(
		collector.__globals__, '_read_candidate_job_evidence', read_evidence
	)
	protocol_identity = {'path': str(settings.protocol_lock), 'sha256': 'e' * 64}
	evidence = collector(
		settings,
		canonical,
		protocol_lock_identity=protocol_identity,
		verify_live_checkpoints=True,
	)
	expected_calls = [
		(candidate_id, layout_id, 'medium')
		for candidate_id in CANDIDATE_IDS
		for layout_id in five_way_runner.LAYOUT_IDS
	]
	assert calls == expected_calls
	assert len(evidence['inputs']) == 15
	assert evidence['selected_candidate_id'] == IDENTITY_STD010_ID

	calls.clear()
	reversed_settings = replace(
		settings, candidates=tuple(reversed(settings.candidates))
	)
	reversed_evidence = collector(
		reversed_settings,
		canonical,
		protocol_lock_identity=protocol_identity,
		verify_live_checkpoints=True,
	)
	assert calls == expected_calls
	assert reversed_evidence == evidence


def test_candidate_job_audit_binds_source_layout_size_and_metrics_path(
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	config_builder = cast('object', runner_namespace['_candidate_config'])
	validator = cast('object', runner_namespace['_validate_job_source_audit'])
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	candidate = settings.candidate_by_id(FORCED_STD005_ID)
	candidate_config = config_builder(
		canonical, candidate=candidate, runs_root=settings.runs_root
	)
	job = five_way_runner.resolve_f3_lithology_five_way_job(
		candidate_config,
		model=candidate.candidate_id,
		layout='layout_000',
		size='medium',
	)
	embedding_metadata = (
		candidate.embeddings_dir
		/ f'{canonical.dataset["name"]}.embedding_metadata.json'
	)
	source_audit = {
		'audit_schema_version': 3,
		'candidate_id': candidate.candidate_id,
		'validation_only': True,
		'selection_eligible': True,
		'protocol_lock': {
			'path': str(settings.protocol_lock),
			'sha256': 'e' * 64,
		},
		'base_checkpoint': str(candidate.base_checkpoint),
		'base_checkpoint_sha256': 'a' * 64,
		'continuation_init_checkpoint_sha256': 'a' * 64,
		'final_checkpoint': str(candidate.final_checkpoint),
		'final_checkpoint_sha256': 'b' * 64,
		'embeddings_dir': str(candidate.embeddings_dir),
		'embedding_metadata': str(embedding_metadata),
		'base_pretraining_epochs': 25,
		'continuation_epochs': 25,
		'augmentations': {
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.05,
		},
		'reference_base_checkpoint': str(settings.reference_base_checkpoint),
		'reference_base_checkpoint_sha256': (
			settings.reference_base_checkpoint_sha256
		),
		'reference_final_checkpoint': str(settings.reference_final_checkpoint),
		'reference_final_checkpoint_sha256': (
			settings.reference_final_checkpoint_sha256
		),
		'base_parity_exceptions': [
			'augmentations',
			'train.epochs',
			'paths.output_root',
		],
		'final_parity_exceptions': [
			'augmentations',
			'paths.output_root',
			'continuation.init_checkpoint',
		],
		'fixed_downstream_summary_name': canonical.summary_name,
		'fixed_section_layout_dataset_root': str(
			canonical.section_layout_dataset_root
		),
		'token_grid_shape': [10, 11, 12],
		'valid_token_mask': 'byte_identical_to_canonical_random',
		'evaluation_split': 'validation',
		'evaluation_aggregation_unit': 'unique_validation_voxel',
	}
	job_audit = {
		**source_audit,
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'metrics_path': str(job.metrics_path),
	}
	validator(
		audit=job_audit,
		job=job,
		candidate=candidate,
		canonical=canonical,
		reference_base_checkpoint=settings.reference_base_checkpoint,
		reference_final_checkpoint=settings.reference_final_checkpoint,
		expected_base_checkpoint_sha256='a' * 64,
		expected_final_checkpoint_sha256='b' * 64,
		expected_source_audit=source_audit,
		expected_protocol_lock=source_audit['protocol_lock'],
	)

	tampered = deepcopy(job_audit)
	tampered['layout_id'] = 'layout_004'
	with pytest.raises(ValueError, match='layout_id'):
		validator(
			audit=tampered,
			job=job,
			candidate=candidate,
			canonical=canonical,
			reference_base_checkpoint=settings.reference_base_checkpoint,
			reference_final_checkpoint=settings.reference_final_checkpoint,
			expected_base_checkpoint_sha256='a' * 64,
			expected_final_checkpoint_sha256='b' * 64,
			expected_source_audit=source_audit,
			expected_protocol_lock=source_audit['protocol_lock'],
		)


def test_job_evaluation_identity_rejects_metrics_and_initial_state_drift(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	config_builder = cast('object', runner_namespace['_candidate_config'])
	validator = cast(
		'FunctionType', runner_namespace['_validate_job_evaluation_identity']
	)
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	candidate = settings.candidate_by_id(FORCED_STD005_ID)
	candidate_config = config_builder(
		canonical, candidate=candidate, runs_root=settings.runs_root
	)
	job = five_way_runner.resolve_f3_lithology_five_way_job(
		candidate_config,
		model=candidate.candidate_id,
		layout='layout_000',
		size='medium',
	)
	metrics_sha = 'a' * 64
	evaluation_metadata = {
		'dataset': dict(job.config.dataset),
		'model_tag': job.model.model_id,
		'aggregation': {'primary_unit': 'unique_validation_voxel'},
		'policy': {
			key: list(value) if isinstance(value, tuple) else value
			for key, value in five_way_runner.FIVE_WAY_EVALUATION_POLICY.items()
		},
		'outputs': {
			'metrics.json': {
				'path': str(job.metrics_path),
				'sha256': metrics_sha,
			}
		},
	}
	run_metadata = {'initial_model_state_sha256': 'c' * 64}

	def read_json(path: Path) -> dict[str, object]:
		return (
			evaluation_metadata
			if path.name == 'evaluation_metadata.json'
			else run_metadata
		)

	runner_module = validator.__globals__['five_way_runner']
	results_module = validator.__globals__['five_way_results']
	monkeypatch.setitem(validator.__globals__, '_read_json', read_json)
	monkeypatch.setitem(validator.__globals__, 'file_sha256', lambda _: 'b' * 64)
	monkeypatch.setattr(runner_module, '_decoder_is_completed', lambda *_: True)
	monkeypatch.setattr(
		runner_module,
		'inspect_f3_lithology_five_way_job',
		lambda _: {'decoder_initial_state_sha256': 'c' * 64},
	)
	monkeypatch.setattr(
		results_module,
		'_job_source_identity',
		lambda **_: {'decoder_checkpoint_sha256': 'b' * 64},
	)
	validator(job=job, metrics_sha256=metrics_sha)

	evaluation_metadata['outputs']['metrics.json']['sha256'] = 'd' * 64
	with pytest.raises(ValueError, match='changed after evaluation'):
		validator(job=job, metrics_sha256=metrics_sha)
	evaluation_metadata['outputs']['metrics.json']['sha256'] = metrics_sha
	run_metadata['initial_model_state_sha256'] = 'e' * 64
	with pytest.raises(ValueError, match='initial-state identity changed'):
		validator(job=job, metrics_sha256=metrics_sha)


def test_lock_creation_rejects_any_prelock_legacy_or_nonmedium_result(
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	config_builder = cast('object', runner_namespace['_candidate_config'])
	rejector = cast(
		'object', runner_namespace['_reject_legacy_results_before_selection_lock']
	)
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	selectable = settings.candidate_by_id(FORCED_STD005_ID)
	selectable_config = config_builder(
		canonical, candidate=selectable, runs_root=settings.runs_root
	)
	small_job = five_way_runner.resolve_f3_lithology_five_way_job(
		selectable_config,
		model=selectable.candidate_id,
		layout='layout_000',
		size='small',
	)
	small_audit = small_job.output_dir / 'candidate_source_audit.json'
	small_audit.parent.mkdir(parents=True)
	small_audit.write_text('{}', encoding='utf-8')
	with pytest.raises(ValueError, match='out of preregistered order'):
		rejector(settings, canonical)
	small_audit.unlink()

	legacy = settings.source_by_id(LEGACY_CONTROL_ID)
	legacy_config = config_builder(
		canonical, candidate=legacy, runs_root=settings.runs_root
	)
	legacy_job = five_way_runner.resolve_f3_lithology_five_way_job(
		legacy_config,
		model=LEGACY_CONTROL_ID,
		layout='layout_000',
		size='medium',
	)
	legacy_job.metrics_path.parent.mkdir(parents=True)
	legacy_job.metrics_path.write_text('{}', encoding='utf-8')
	with pytest.raises(ValueError, match='out of preregistered order'):
		rejector(settings, canonical)


def test_protocol_lock_is_exclusive_and_revalidates_all_four_bases(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	creator = cast('FunctionType', runner_namespace['create_gaussian25_protocol_lock'])
	validator = cast(
		'FunctionType', runner_namespace['validate_gaussian25_protocol_lock']
	)
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	repository_state = {
		'git_head': '1' * 40,
		'git_dirty': True,
		'relevant_git_status_sha256': 'e' * 64,
		'relevant_file_inventory': [],
	}
	benchmark_provenance = {
		'canonical_five_way_config': str(settings.canonical_five_way_config),
		'canonical_five_way_config_sha256': (
			settings.canonical_five_way_config_sha256
		),
	}
	base_inputs = [
		{
			'candidate_id': candidate_id,
			'base_checkpoint_sha256': f'{index}' * 64,
			'passed': True,
		}
		for index, candidate_id in enumerate((*CANDIDATE_IDS, LEGACY_CONTROL_ID), 1)
	]
	monkeypatch.setitem(
		creator.__globals__, '_reject_pre_protocol_evidence', lambda *_: None
	)
	monkeypatch.setitem(
		creator.__globals__, '_git_repository_state', lambda: repository_state
	)
	monkeypatch.setitem(
		creator.__globals__,
		'_validate_benchmark_provenance',
		lambda *_a, **_k: benchmark_provenance,
	)
	monkeypatch.setitem(
		creator.__globals__, '_collect_protocol_base_inputs', lambda *_: base_inputs
	)
	payload = creator(
		settings,
		canonical,
		created_at_utc='2026-08-29T00:00:00Z',
		git_head='1' * 40,
	)
	assert payload['stage_boundary'] == 'completed_bases_before_continuation'
	assert payload['base_checkpoint_inputs'] == base_inputs
	assert payload['repository_state'] == repository_state
	assert payload['benchmark_provenance'] == benchmark_provenance
	assert validator(settings, canonical) == payload

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		creator(settings, canonical)
	drifted = deepcopy(base_inputs)
	drifted[0]['base_checkpoint_sha256'] = 'f' * 64
	monkeypatch.setitem(
		creator.__globals__, '_collect_protocol_base_inputs', lambda *_: drifted
	)
	with pytest.raises(ValueError, match='does not match the frozen'):
		validator(settings, canonical)


def test_protocol_creation_rejects_existing_post_base_evidence(
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	rejector = cast('FunctionType', runner_namespace['_reject_pre_protocol_evidence'])
	settings = resolver(load_config(CONFIG))
	rejector(settings)

	source = settings.candidate_by_id(FORCED_STD005_ID)
	continuation_marker = source.final_checkpoint.parent / 'run_metadata.json'
	continuation_marker.parent.mkdir(parents=True, exist_ok=True)
	continuation_marker.write_text('{}\n', encoding='utf-8')
	with pytest.raises(ValueError, match='continuation output evidence'):
		rejector(settings)
	continuation_marker.unlink()

	embedding_marker = source.embeddings_dir / 'f3.embedding_metadata.json'
	embedding_marker.parent.mkdir(parents=True, exist_ok=True)
	embedding_marker.write_text('{}\n', encoding='utf-8')
	with pytest.raises(ValueError, match='candidate embeddings evidence'):
		rejector(settings)
	embedding_marker.unlink()

	validation_marker = settings.runs_root / 'unexpected.json'
	validation_marker.parent.mkdir(parents=True, exist_ok=True)
	validation_marker.write_text('{}\n', encoding='utf-8')
	with pytest.raises(ValueError, match='validation runs evidence'):
		rejector(settings)


def test_protocol_lock_is_required_before_first_validation_cell(
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	enforce = cast('FunctionType', runner_namespace['enforce_validation_order'])
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	with pytest.raises(FileNotFoundError, match='protocol lock is required'):
		enforce(
			settings=settings,
			canonical=canonical,
			candidate=settings.candidate_by_id(FORCED_STD005_ID),
			data_size='medium',
		)


def test_selection_lock_is_exclusive_and_detects_input_drift(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	creator = cast('FunctionType', runner_namespace['create_gaussian25_selection_lock'])
	validator = cast(
		'FunctionType', runner_namespace['validate_gaussian25_selection_lock']
	)
	ranker = cast('object', runner_namespace['_rank_candidate_scores'])
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	evidence = _selection_evidence(ranker)
	monkeypatch.setitem(
		creator.__globals__,
		'_reject_legacy_results_before_selection_lock',
		lambda *_: None,
	)
	monkeypatch.setitem(
		creator.__globals__,
		'_collect_selection_evidence',
		lambda *_a, **_k: evidence,
	)
	repository_state = {
		'git_head': '1' * 40,
		'git_dirty': True,
		'relevant_git_status_sha256': 'e' * 64,
		'relevant_file_inventory': [
			{
				'path': str(RUNNER),
				'git_status': '??',
				'state': 'file',
				'sha256': 'f' * 64,
			}
		],
	}
	benchmark_provenance = {'random_checkpoint_sha256': 'a' * 64}
	settings.protocol_lock.parent.mkdir(parents=True, exist_ok=True)
	settings.protocol_lock.write_text('{}\n', encoding='utf-8')
	protocol_payload = {
		'protocol_lock_type': 'f3_local_barlow_twins_gaussian25_protocol_v1',
		'git_head': '1' * 40,
		'repository_state': repository_state,
		'benchmark_provenance': benchmark_provenance,
	}
	monkeypatch.setitem(
		creator.__globals__,
		'validate_gaussian25_protocol_lock',
		lambda *_: protocol_payload,
	)
	payload = creator(
		settings,
		canonical,
		created_at_utc='2026-08-29T00:00:00Z',
		git_head='1' * 40,
	)
	assert payload['selected_candidate_id'] == IDENTITY_STD010_ID
	assert len(payload['inputs']) == 15
	assert payload['base_pretraining_epochs'] == 25
	assert payload['continuation_epochs'] == 25
	assert payload['repository_state'] == repository_state
	assert payload['benchmark_provenance'] == benchmark_provenance
	assert payload['protocol_lock']['path'] == str(settings.protocol_lock)
	assert len(payload['protocol_lock']['sha256']) == 64
	assert {row['base_checkpoint_sha256'] for row in payload['inputs']} == {
		'c' * 64
	}
	assert {row['final_checkpoint_sha256'] for row in payload['inputs']} == {
		'd' * 64
	}
	assert validator(settings, canonical) == payload

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		creator(
			settings,
			canonical,
			created_at_utc='2026-08-29T00:00:00Z',
			git_head='1' * 40,
		)

	drifted = deepcopy(evidence)
	drifted['inputs'][0]['metrics_sha256'] = 'c' * 64
	monkeypatch.setitem(
		creator.__globals__,
		'_collect_selection_evidence',
		lambda *_a, **_k: drifted,
	)
	with pytest.raises(ValueError, match='does not match'):
		validator(settings, canonical)


@pytest.mark.parametrize(
	'value',
	[True, None, '0.5', float('nan'), float('inf'), -0.01, 1.01],
)
def test_selection_metric_rejects_invalid_macro_f1(
	runner_namespace: dict[str, object], value: object
) -> None:
	validator = cast('object', runner_namespace['_macro_f1'])
	with pytest.raises(ValueError, match='macro_f1'):
		validator(
			{
				'aggregation_unit': 'unique_validation_voxel',
				'macro_f1': value,
			},
			Path('/validation/metrics.json'),
		)


def test_prelock_and_postlock_runtime_gate_matrix(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	enforce = cast('FunctionType', runner_namespace['enforce_validation_order'])
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	selectable = settings.candidate_by_id(FORCED_STD005_ID)
	legacy = settings.source_by_id(LEGACY_CONTROL_ID)

	monkeypatch.setitem(
		enforce.__globals__, 'validate_gaussian25_protocol_lock', lambda *_: {}
	)
	monkeypatch.setitem(
		enforce.__globals__,
		'validate_gaussian25_selection_lock',
		lambda *_: (_ for _ in ()).throw(FileNotFoundError('lock required')),
	)
	assert (
		enforce(
			settings=settings,
			canonical=canonical,
			candidate=selectable,
			data_size='medium',
		)
		is None
	)
	with pytest.raises(FileNotFoundError, match='lock required'):
		enforce(
			settings=settings,
			canonical=canonical,
			candidate=legacy,
			data_size='medium',
		)
	with pytest.raises(FileNotFoundError, match='lock required'):
		enforce(
			settings=settings,
			canonical=canonical,
			candidate=selectable,
			data_size='small',
		)

	identity_lock = {'selected_candidate_id': IDENTITY_STD010_ID}
	monkeypatch.setitem(
		enforce.__globals__,
		'validate_gaussian25_selection_lock',
		lambda *_: identity_lock,
	)
	monkeypatch.setitem(
		enforce.__globals__,
		'_validate_medium_random_gate',
		lambda **_: {'passed': True},
	)
	for candidate_id in (IDENTITY_STD010_ID, FORCED_STD010_ID, LEGACY_CONTROL_ID):
		result = enforce(
			settings=settings,
			canonical=canonical,
			candidate=settings.source_by_id(candidate_id),
			data_size='large',
		)
		assert result['selection_lock'] == identity_lock
	with pytest.raises(ValueError, match='not allowed'):
		enforce(
			settings=settings,
			canonical=canonical,
			candidate=selectable,
			data_size='large',
		)


def test_medium_gate_requires_strict_five_of_five_from_one_arm(
	runner_namespace: dict[str, object],
) -> None:
	gate = cast('object', runner_namespace['_medium_5of5_wins'])
	random = dict.fromkeys(five_way_runner.LAYOUT_IDS, 0.5)
	locked = dict.fromkeys(five_way_runner.LAYOUT_IDS, 0.6)
	legacy = dict.fromkeys(five_way_runner.LAYOUT_IDS, 0.6)
	locked['layout_004'] = 0.5
	legacy['layout_000'] = 0.4
	wins = gate(
		{FORCED_STD005_ID: locked, LEGACY_CONTROL_ID: legacy, 'random': random},
		selected_id=FORCED_STD005_ID,
	)
	assert wins == {FORCED_STD005_ID: False, LEGACY_CONTROL_ID: False}
	locked['layout_004'] = 0.5000000001
	wins = gate(
		{FORCED_STD005_ID: locked, LEGACY_CONTROL_ID: legacy, 'random': random},
		selected_id=FORCED_STD005_ID,
	)
	assert wins == {FORCED_STD005_ID: True, LEGACY_CONTROL_ID: False}


def test_benchmark_provenance_derives_and_verifies_every_pinned_input(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	validator = cast('FunctionType', runner_namespace['_validate_benchmark_provenance'])
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	random_checkpoint = canonical.model_by_id('random').checkpoint
	comparison = canonical.summary_root / 'comparison.csv'
	manifest = (
		canonical.artifact_root
		/ 'registry/manifests/f3/facies_benchmark_v1/f3_amplitude_manifest.json'
	)
	path_list = (
		canonical.artifact_root
		/ 'registry/splits/f3/facies_benchmark_v1/f3_npy_paths.txt'
	)
	for path in (
		random_checkpoint,
		comparison,
		manifest,
		path_list,
		settings.reference_base_checkpoint,
		settings.reference_final_checkpoint,
	):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b'pinned')
	digests = {
		settings.canonical_five_way_config: (
			settings.canonical_five_way_config_sha256
		),
		settings.reference_base_checkpoint: (
			settings.reference_base_checkpoint_sha256
		),
		settings.reference_final_checkpoint: (
			settings.reference_final_checkpoint_sha256
		),
		random_checkpoint: settings.random_checkpoint_sha256,
		comparison: settings.canonical_comparison_sha256,
		manifest: settings.pretraining_manifest_sha256,
		path_list: settings.pretraining_path_list_sha256,
	}
	monkeypatch.setitem(validator.__globals__, 'file_sha256', digests.__getitem__)
	monkeypatch.setitem(
		validator.__globals__,
		'load_checkpoint_metadata_without_weights',
		lambda _: {
			'config': {
				'manifests': {
					'train': str(manifest),
					'train_path_list': str(path_list),
				}
			}
		},
	)
	result = validator(settings, canonical, verify_files=True)
	assert result['canonical_five_way_config'] == str(
		settings.canonical_five_way_config
	)
	assert result['canonical_five_way_config_sha256'] == (
		settings.canonical_five_way_config_sha256
	)
	assert result['random_checkpoint'] == str(random_checkpoint)
	assert result['canonical_comparison'] == str(comparison)
	assert result['pretraining_manifest'] == str(manifest)
	assert result['pretraining_path_list'] == str(path_list)

	digests[random_checkpoint] = '0' * 64
	with pytest.raises(ValueError, match='random checkpoint SHA-256'):
		validator(settings, canonical, verify_files=True)


def test_random_metric_must_equal_its_pinned_comparison_row(
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	validator = cast(
		'FunctionType', runner_namespace['_validate_random_comparison_row']
	)
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	job = five_way_runner.resolve_f3_lithology_five_way_job(
		canonical, model='random', layout='layout_000', size='medium'
	)
	comparison = canonical.summary_root / 'comparison.csv'
	comparison.parent.mkdir(parents=True, exist_ok=True)
	row = dict.fromkeys(five_way_results.COMPARISON_FIELDNAMES, '')
	row.update(
		{
			'model_id': 'random',
			'layout_id': job.layout_id,
			'data_size': job.data_size,
			'checkpoint_path': str(job.model.checkpoint),
			'encoder_checkpoint_sha256': (
				'6548d52446e7d6b9b57acd2bd39a8389a76bc5df55b52a9eda0472eb182a438c'
			),
			'macro_f1': '0.625',
			'metrics_path': str(job.metrics_path),
		}
	)
	with comparison.open('w', encoding='utf-8', newline='') as stream:
		writer = csv.DictWriter(
			stream, fieldnames=five_way_results.COMPARISON_FIELDNAMES
		)
		writer.writeheader()
		writer.writerow(row)

	validator(canonical, job=job, macro_f1=0.625)
	with pytest.raises(ValueError, match='live random macro_f1'):
		validator(canonical, job=job, macro_f1=0.5)


def test_repository_state_inventories_dirty_and_experiment_files(
	runner_namespace: dict[str, object],
) -> None:
	resolver = cast('object', runner_namespace['_git_repository_state'])
	state = resolver()
	assert isinstance(state['git_dirty'], bool)
	assert len(state['git_head']) == 40
	assert len(state['relevant_git_status_sha256']) == 64
	inventory = state['relevant_file_inventory']
	paths = {row['path'] for row in inventory}
	assert str(RUNNER) in paths
	assert str(CONFIG) in paths
	assert str(EXPERIMENT_ROOT / 'README.md') in paths
	assert all(len(row['sha256']) == 64 for row in inventory if row['state'] == 'file')
	assert all('__pycache__' not in row['path'] for row in inventory)


def test_exact_reached_cell_sets_and_winner_rule_are_preregistered(
	runner_namespace: dict[str, object],
) -> None:
	cell_set = cast('object', runner_namespace['_expected_gaussian25_cells'])
	choose = cast('object', runner_namespace['_choose_gaussian25_winner'])
	assert len(
		cell_set(selected_id=FORCED_STD005_ID, medium_gate_open=False)
	) == 20
	assert len(cell_set(selected_id=FORCED_STD005_ID, medium_gate_open=True)) == 40
	assert len(cell_set(selected_id=IDENTITY_STD010_ID, medium_gate_open=True)) == 50
	assert (
		choose(
			selected_id=FORCED_STD005_ID,
			selected_passed=True,
			legacy_passed=True,
			attribution_passed=True,
		)
		== FORCED_STD005_ID
	)
	assert (
		choose(
			selected_id=FORCED_STD005_ID,
			selected_passed=True,
			legacy_passed=True,
			attribution_passed=False,
		)
		== LEGACY_CONTROL_ID
	)
	assert (
		choose(
			selected_id=FORCED_STD005_ID,
			selected_passed=False,
			legacy_passed=False,
			attribution_passed=False,
		)
		is None
	)


def test_final_audit_persists_complete_failed_medium_branch_exclusively(
	runner_namespace: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	resolver = cast('object', runner_namespace['validation_settings_from_mapping'])
	canonical_resolver = cast('object', runner_namespace['_canonical_config'])
	creator = cast('FunctionType', runner_namespace['create_gaussian25_final_result'])
	settings = resolver(load_config(CONFIG))
	canonical = canonical_resolver(settings)
	settings.selection_lock.parent.mkdir(parents=True, exist_ok=True)
	settings.protocol_lock.write_text('{}\n', encoding='utf-8')
	settings.selection_lock.write_text('{}\n', encoding='utf-8')
	protocol_payload = {
		'protocol_lock_type': 'f3_local_barlow_twins_gaussian25_protocol_v1'
	}
	selection_lock = {
		'selected_candidate_id': FORCED_STD005_ID,
		'repository_state': {'git_head': '1' * 40, 'git_dirty': True},
	}
	medium_gate = {
		'gate_open': False,
		'locked_candidate_id': FORCED_STD005_ID,
		'locked_candidate_wins_over_random': False,
		'legacy_wins_over_random': False,
		'inputs': [],
	}
	monkeypatch.setitem(
		creator.__globals__,
		'validate_gaussian25_protocol_lock',
		lambda *_: protocol_payload,
	)
	monkeypatch.setitem(
		creator.__globals__,
		'validate_gaussian25_selection_lock',
		lambda *_: selection_lock,
	)
	monkeypatch.setitem(
		creator.__globals__,
		'_validate_benchmark_provenance',
		lambda *_a, **_k: {'random_checkpoint_sha256': 'a' * 64},
	)
	monkeypatch.setitem(
		creator.__globals__,
		'_validate_medium_random_gate',
		lambda **_: medium_gate,
	)
	monkeypatch.setitem(
		creator.__globals__, '_validate_exact_candidate_cell_set', lambda **_: None
	)
	monkeypatch.setitem(
		creator.__globals__,
		'_validation_order_provenance',
		lambda *_: {'selection_lock': {'sha256': 'b' * 64}},
	)
	monkeypatch.setitem(
		creator.__globals__,
		'audit_candidate_source',
		lambda **_: {
			'base_checkpoint_sha256': 'c' * 64,
			'final_checkpoint_sha256': 'd' * 64,
		},
	)

	def candidate_row(*, job: object, **_: object) -> dict[str, object]:
		return {
			'candidate_id': job.model.model_id,
			'layout_id': job.layout_id,
			'data_size': job.data_size,
			'macro_f1': 0.4,
		}

	monkeypatch.setitem(
		creator.__globals__, '_read_candidate_job_evidence', candidate_row
	)
	monkeypatch.setitem(
		creator.__globals__,
		'_read_random_job_evidence',
		lambda _canonical, *, layout_id, data_size: {
			'candidate_id': 'random',
			'layout_id': layout_id,
			'data_size': data_size,
			'macro_f1': 0.5,
		},
	)
	monkeypatch.setattr(
		creator.__globals__['five_way_sources'],
		'audit_f3_lithology_five_way_sources',
		lambda *_: None,
	)
	result = creator(
		settings, canonical, created_at_utc='2026-08-29T00:00:00Z'
	)
	assert result['passed'] is False
	assert result['winner_candidate_id'] is None
	assert result['authorizes_next_base_duration'] is True
	assert result['failure_stage'] == 'medium_5of5'
	assert result['protocol_lock']['path'] == str(settings.protocol_lock)
	assert len(result['protocol_lock']['sha256']) == 64
	assert len(result['candidate_inputs']) == 20
	assert len(result['random_inputs']) == 5
	assert settings.final_result.is_file()
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		creator(settings, canonical)


def test_cli_modes_are_mutually_scoped(
	runner_namespace: dict[str, object],
) -> None:
	validator = cast('object', runner_namespace['_validate_cli_arguments'])
	base = {
		'audit_base_checkpoint_only': False,
		'audit_checkpoint_only': False,
		'create_protocol_lock': False,
		'create_selection_lock': False,
		'create_final_result': False,
		'candidate': None,
		'layout': None,
		'size': None,
		'resume': None,
		'dry_run': False,
	}
	validator(
		Namespace(
			**{
				**base,
				'audit_checkpoint_only': True,
				'candidate': FORCED_STD005_ID,
			}
		)
	)
	validator(
		Namespace(
			**{
				**base,
				'audit_base_checkpoint_only': True,
				'candidate': FORCED_STD005_ID,
			}
		)
	)
	validator(Namespace(**{**base, 'create_protocol_lock': True}))
	validator(Namespace(**{**base, 'create_selection_lock': True}))
	validator(Namespace(**{**base, 'create_final_result': True}))
	validator(
		Namespace(
			**{
				**base,
				'candidate': FORCED_STD005_ID,
				'layout': 'layout_000',
				'size': 'medium',
			}
		)
	)
	with pytest.raises(ValueError, match='require --candidate'):
		validator(Namespace(**{**base, 'audit_checkpoint_only': True}))
	with pytest.raises(ValueError, match='job-selection arguments'):
		validator(
			Namespace(
				**{
					**base,
					'create_protocol_lock': True,
					'candidate': FORCED_STD005_ID,
				}
			)
		)
	with pytest.raises(ValueError, match='job-selection arguments'):
		validator(
			Namespace(
				**{
					**base,
					'create_selection_lock': True,
					'candidate': FORCED_STD005_ID,
				}
			)
		)
	with pytest.raises(ValueError, match='job-selection arguments'):
		validator(
			Namespace(
				**{
					**base,
					'create_final_result': True,
					'candidate': FORCED_STD005_ID,
				}
			)
		)


def test_runbook_preregisters_duration_contingency_without_test_tuning() -> None:  # noqa: PLR0915
	text = (EXPERIMENT_ROOT / 'README.md').read_text(encoding='utf-8')

	assert 'order is **25, then 5, then 1 epoch**' in text
	assert 'Every base duration is followed\nby the same fixed 25-epoch' in text
	assert 'same five `medium`\n   v3 layouts for exactly the' in text
	assert 'unrounded\n   mean medium macro-F1' in text
	assert 'forced-flip 0.05,\n   forced-flip 0.10, then identity 0.10' in text
	assert 'exactly 15 metrics and\n   their 15 job-specific candidate audits' in text
	assert 'The legacy result cannot change this choice' in text
	assert 'fixed-strength identity-0.10 minus forced-flip-0.10 contrast' in text
	assert 'geometry\n   control cannot open the medium gate' in text
	assert 'all ten extra small/large geometry\ncells are mandatory' in text
	assert 'positive paired delta in all five layouts (`5/5`)' in text
	assert 'Gaussian attribution additionally requires a' in text
	assert 'positive Gaussian-minus-legacy delta in all five layouts' in text
	assert 'first base duration in the fixed 25 -> 5 -> 1' in text
	assert 'passing final validation arm' in text
	assert 'all 15 layout/size cells (`15/15`)' in text
	assert 'fails either the small or\n   large random gate' in text
	assert 'Small, medium, and large are all\n   validation tuning' in text
	assert 'none is an untouched holdout or post-selection\n   confirmation' in text
	assert '5- and 1-base-epoch producer, fixed-continuation' in text
	assert 'after their branch is reached, never speculatively' in text
	assert 'Carry only the locked view mapping plus matched legacy' in text
	assert 'Every base duration is a fresh seed-42 run from initialization' in text
	assert 'never pass the base through `--resume`' in text
	assert 'persistent-worker loader' in text
	assert 'distinct, immutable model ID, base output root' in text
	assert 'Give every base duration a\ndistinct validation run root' in text
	assert 'no later arm or duration may reuse or overwrite' in text
	assert (
		'Gaussian-minus-random, legacy-minus-random, and Gaussian-minus-legacy'
		in text
	)
	assert 'unfreeze_top_blocks: 1' in text
	assert 'learning rate `1e-5`' in text
	assert 'final metadata is epoch 25 and\nglobal step 15,625' in text
	assert 'Extraction consumes only each arm\'s final continuation checkpoint' in text
	assert 'promote the selected Gaussian' not in text
	assert 'unique\nvalidation voxels' in text
	assert 'Test data and\ntest metrics must remain untouched' in text
	assert 'defines no test evaluation or test output' in text
	assert '--audit-base-checkpoint-only' in text
	assert '--audit-checkpoint-only' in text
	assert '--create-selection-lock' in text
	assert 'created exclusively without overwrite' in text
	assert (
		'84550ed658166e8e6a40cd664e2e9ffbeab0c12d6917006abb417cd25e228ac0'
		in text
	)
	assert (
		'1c5312244f290dbfdcf2688ffa9fa8b5c64452ade162d5335be1bb8a0e256291'
		in text
	)
	assert 'both the base and final checkpoint SHA-256' in text
	assert 'covers only\nlegacy forced flips plus the two forced-flip' in text
	assert 'Fixed positions\ncan themselves become a shortcut' in text
	assert text.index('--audit-base-checkpoint-only') < text.index(
		'$EXP/15_stage2/gaussian_noise_std005'
	)
	assert text.index('--audit-checkpoint-only') < text.index(
		'$EXP/20_embeddings/01_extract_gaussian_noise_std005'
	)
	assert text.index('--create-selection-lock') < text.index(
		'Only now run legacy on the five medium layouts'
	)
	assert 'audit_f3_lithology_five_way_sources.py' not in text
