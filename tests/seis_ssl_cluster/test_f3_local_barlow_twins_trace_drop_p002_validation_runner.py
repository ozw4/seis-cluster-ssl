"""Decision and provenance tests for the p=.02 experiment-113 runner."""

from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from seis_ssl_cluster.embedding.writer import file_sha256

if TYPE_CHECKING:
	from collections.abc import Mapping

RUNNER_PATH = Path(
	'experiments/f3/facies_benchmark_v2/'
	'113_local_barlow_twins_trace_drop_p002_view_v1/run_validation.py'
)
CANDIDATE_ID = 'local_barlow_twins_horizontal_trace_drop_p002_base1ep'
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))


@pytest.fixture(scope='module')
def runner() -> dict[str, object]:
	return runpy.run_path(str(RUNNER_PATH))


def _evidence(source_id: str, values: list[float]) -> list[dict[str, object]]:
	return [
		{
			'candidate_id': source_id,
			'layout_id': layout_id,
			'data_size': 'medium',
			'macro_f1': value,
			'metrics_path': f'/metrics/{source_id}/{layout_id}.json',
			'metrics_sha256': f'{index + 1:064x}',
		}
		for index, (layout_id, value) in enumerate(
			zip(LAYOUT_IDS, values, strict=True)
		)
	]


def _candidate(runner: Mapping[str, object], root: Path) -> object:
	constructor = cast('Any', runner['CandidateSource'])
	return constructor(
		candidate_id=CANDIDATE_ID,
		role='authorized_trace_drop_strength_followup',
		base_checkpoint=root / 'base/latest.pt',
		final_checkpoint=root / 'final/latest.pt',
		embeddings_dir=root / 'embeddings',
		augmentations={
			'policy': 'horizontal_flip_trace_drop_v1',
			'horizontal_flip_probability': 0.5,
			'trace_drop_probability': 0.02,
		},
		base_pretraining_epochs=1,
		continuation_epochs=25,
	)


def _common_config(*, output_root: Path, epochs: int) -> dict[str, object]:
	return {
		'paths': {'artifact_root': '/artifacts', 'output_root': str(output_root)},
		'augmentations': {
			'policy': 'horizontal_flip_trace_drop_v1',
			'horizontal_flip_probability': 0.5,
			'trace_drop_probability': 0.02,
		},
		'barlow_twins': {
			'method': 'local_barlow_twins_3d',
			'local_pairs_per_crop': 128,
		},
		'train': {
			'epochs': epochs,
			'samples_per_epoch': 10_000,
			'batch_size': 16,
			'lr': 1e-4 if epochs == 1 else 1e-5,
		},
	}


def _checkpoint_payload(
	*, config: Mapping[str, object], epoch: int, global_step: int
) -> dict[str, object]:
	return {
		'checkpoint_kind': 'barlow_twins_pretraining',
		'pretraining_method': 'local_barlow_twins_3d',
		'epoch': epoch,
		'global_step': global_step,
		'training_state': {
			'schema_version': 1,
			'stage': 'barlow_twins_training',
			'resume_boundary': 'epoch',
			'dataset_epoch': 0 if epoch == 1 else 24,
			'completed_epoch': True,
		},
		'resume_count': 0,
		'amp_enabled': False,
		'scaler_state_dict': None,
		'trained_parameter_prefixes': ['patch_projection.', 'encoder.'],
		'config': dict(config),
		'continuation_lineage': None,
	}


def test_strict_medium_gate_opens_only_for_five_positive_deltas(
	runner: Mapping[str, object],
) -> None:
	gate = cast('Any', runner['_medium_gate_result'])
	random = _evidence('random', [0.5] * 5)
	open_result = gate(_evidence(CANDIDATE_ID, [0.51] * 5), random)
	tied_result = gate(
		_evidence(CANDIDATE_ID, [0.51, 0.51, 0.5, 0.51, 0.51]),
		random,
	)

	assert open_result['gate_open'] is True
	assert open_result['positive_delta_count'] == 5
	assert tied_result['gate_open'] is False
	assert tied_result['positive_delta_count'] == 4
	assert tied_result['paired_macro_f1_deltas_over_random']['layout_002'] == 0.0


def test_base_checkpoint_contract_requires_fresh_exact_one_epoch(
	runner: Mapping[str, object], tmp_path: Path
) -> None:
	validator = cast('Any', runner['_validate_candidate_base_checkpoint'])
	candidate = _candidate(runner, tmp_path)
	config = _common_config(output_root=candidate.base_checkpoint.parent, epochs=1)
	reference_config = deepcopy(config)
	reference_config['paths']['output_root'] = '/reference/base'
	reference_config['train']['epochs'] = 100
	reference_config['augmentations'] = {'horizontal_flip_probability': 0.5}
	payload = _checkpoint_payload(config=config, epoch=1, global_step=625)
	reference = {'config': reference_config}

	validator(candidate, payload=payload, reference=reference)
	stale = deepcopy(payload)
	stale['resume_count'] = 1
	with pytest.raises(ValueError, match='resume_count must equal 0'):
		validator(candidate, payload=stale, reference=reference)
	continued = deepcopy(payload)
	continued['continuation_lineage'] = {'resume_count': 0}
	with pytest.raises(ValueError, match='must not record continuation'):
		validator(candidate, payload=continued, reference=reference)


def test_continuation_contract_requires_exact_fresh_base_lineage(
	runner: Mapping[str, object], tmp_path: Path
) -> None:
	validator = cast('Any', runner['_validate_candidate_final_checkpoint'])
	candidate = _candidate(runner, tmp_path)
	candidate.base_checkpoint.parent.mkdir(parents=True)
	candidate.base_checkpoint.write_bytes(b'fresh-base')
	config = _common_config(output_root=candidate.final_checkpoint.parent, epochs=25)
	config['continuation'] = {
		'init_checkpoint': str(candidate.base_checkpoint),
		'unfreeze_top_blocks': 1,
	}
	reference_config = deepcopy(config)
	reference_config['paths']['output_root'] = '/reference/final'
	reference_config['continuation']['init_checkpoint'] = '/reference/base/latest.pt'
	reference_config['augmentations'] = {'horizontal_flip_probability': 0.5}
	payload = _checkpoint_payload(config=config, epoch=25, global_step=15_625)
	payload['continuation_lineage'] = {
		'schema_version': 1,
		'init_checkpoint': str(candidate.base_checkpoint),
		'init_checkpoint_sha256': file_sha256(candidate.base_checkpoint),
		'resume_count': 0,
	}
	reference = {'config': reference_config}

	validator(candidate, payload=payload, reference=reference)
	stale = deepcopy(payload)
	stale['continuation_lineage']['resume_count'] = 1
	with pytest.raises(ValueError, match='lineage resume_count must equal 0'):
		validator(candidate, payload=stale, reference=reference)
	wrong_base = deepcopy(payload)
	wrong_base['config']['continuation']['init_checkpoint'] = '/other/latest.pt'
	with pytest.raises(ValueError, match='exact fresh base'):
		validator(candidate, payload=wrong_base, reference=reference)


def test_exact_live_cell_set_is_five_closed_and_fifteen_open(
	runner: Mapping[str, object],
) -> None:
	resolver = cast('Any', runner['_expected_candidate_cells'])
	closed = resolver(medium_gate_open=False)
	opened = resolver(medium_gate_open=True)

	assert len(closed) == 5
	assert {size for _, _, size in closed} == {'medium'}
	assert len(opened) == 15
	assert {size for _, _, size in opened} == {'small', 'medium', 'large'}


def test_protocol_payload_freezes_all_random_before_candidate_metrics(
	runner: Mapping[str, object],
) -> None:
	payload_builder = cast('Any', runner['_protocol_lock_payload'])
	random = [
		{
			'candidate_id': 'random',
			'layout_id': layout_id,
			'data_size': data_size,
			'macro_f1': 0.5,
			'metrics_path': f'/random/{layout_id}/{data_size}.json',
			'metrics_sha256': f'{index + 1:064x}',
		}
		for index, (data_size, layout_id) in enumerate(
			(size, layout)
			for size in ('small', 'medium', 'large')
			for layout in LAYOUT_IDS
		)
	]
	payload = payload_builder(
		parent_result={'sha256': 'a' * 64},
		base_checkpoint_input={'candidate_id': CANDIDATE_ID},
		frozen_random_inputs=random,
		benchmark_provenance={},
		repository_state={},
		created_at_utc='2026-08-30T00:00:00Z',
		git_head='b' * 40,
	)

	assert len(payload['frozen_random_inputs']) == 15
	assert payload['candidate_validation_metric_inputs'] == []
	assert payload['stage_boundary'] == 'completed_fresh_base1_before_continuation'
	assert payload['preregistered_augmentations'] == {
		'policy': 'horizontal_flip_trace_drop_v1',
		'horizontal_flip_probability': 0.5,
		'trace_drop_probability': 0.02,
	}
	assert payload['selection_basis'] == (
		'parent_p001_failure_authorized_fixed_unlabeled_p002_v1'
	)
	assert 'selection_lock' not in payload
	assert payload['medium_gate_contract']['required_positive_delta_count'] == 5
	assert payload['success_contract']['required_positive_delta_count'] == 15


def test_protocol_rejects_missing_random_cell(
	runner: Mapping[str, object],
) -> None:
	validator = cast('Any', runner['_validate_all_random_cells'])
	rows = [
		{
			'candidate_id': 'random',
			'layout_id': layout_id,
			'data_size': data_size,
		}
		for data_size in ('small', 'medium', 'large')
		for layout_id in LAYOUT_IDS
	]
	with pytest.raises(ValueError, match='exactly all 15'):
		validator(rows[:-1])


def test_frozen_parent_control_replays_metric_and_audit_type_sensitively(
	runner: Mapping[str, object], tmp_path: Path
) -> None:
	validator = cast('Any', runner['_validate_frozen_control_inputs'])
	metrics = tmp_path / 'metrics.json'
	audit = tmp_path / 'candidate_source_audit.json'
	metrics.write_text(
		json.dumps(
			{
				'aggregation_unit': 'unique_validation_voxel',
				'macro_f1': 0.5,
			}
		)
		+ '\n',
		encoding='utf-8',
	)
	audit.write_text('{}\n', encoding='utf-8')
	row = {
		'metrics_path': str(metrics),
		'metrics_sha256': file_sha256(metrics),
		'macro_f1': 0.5,
		'candidate_audit_path': str(audit),
		'candidate_audit_sha256': file_sha256(audit),
	}

	validator([row])
	with pytest.raises(ValueError, match='macro_f1 differs'):
		validator([{**row, 'macro_f1': 1}])


def test_medium_control_contrast_is_attribution_only_and_strict(
	runner: Mapping[str, object],
) -> None:
	contrast = cast('Any', runner['_medium_control_contrast'])
	scores = {
		CANDIDATE_ID: {(layout, 'medium'): 0.6 for layout in LAYOUT_IDS},
		'control': {
			(layout, 'medium'): 0.5 if layout != 'layout_004' else 0.6
			for layout in LAYOUT_IDS
		},
	}
	result = contrast(
		scores,
		left_id=CANDIDATE_ID,
		right_id='control',
		contrast_id='trace_minus_control',
	)

	assert result['evaluated_cell_count'] == 5
	assert result['positive_delta_count'] == 4
	assert result['wins_all_5'] is False
	assert 'wins_all_15' not in result


@pytest.mark.parametrize(
	('passed', 'gate_open', 'failure_stage'),
	[
		(True, True, None),
		(False, False, 'medium_5of5'),
		(False, True, 'final_15of15'),
	],
)
def test_terminal_decision_does_not_authorize_an_unplanned_followup(
	runner: Mapping[str, object],
	passed: bool,  # noqa: FBT001
	gate_open: bool,  # noqa: FBT001
	failure_stage: str | None,
) -> None:
	decision = cast('Any', runner['_terminal_decision'])(
		arm_result={'wins_all_15_over_random': passed}, gate_open=gate_open
	)

	assert decision['passed'] is passed
	assert decision['failure_stage'] == failure_stage
	assert decision['authorizes_additional_trace_drop_followup'] is False
	assert decision['authorized_additional_trace_drop_probability'] is None


def test_type_sensitive_comparison_rejects_numeric_aliases(
	runner: Mapping[str, object],
) -> None:
	equal = cast('Any', runner['_type_sensitive_equal'])

	assert equal({'value': 1}, {'value': 1}) is True
	assert equal({'value': 1.0}, {'value': 1}) is False
	assert equal({'value': True}, {'value': 1}) is False


def test_cli_has_protocol_but_no_selection_mode(
	runner: Mapping[str, object],
) -> None:
	parser = cast('Any', runner['build_parser'])()
	destinations = {action.dest for action in parser._actions}  # noqa: SLF001

	assert 'create_protocol_lock' in destinations
	assert 'create_selection_lock' not in destinations
	assert 'create_final_result' in destinations


def test_completed_job_wrapper_carries_generic_selection_contract(
	runner: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
	wrapper = cast('Any', runner['_run_job'])
	wrapper_globals = wrapper.__globals__

	def fake_shared(name: str) -> Any:
		assert name == '_run_job'

		def completed(
			_job: object,
			*,
			audit: Mapping[str, object],
			device: str,
			resume: Path | None,
		) -> dict[str, object]:
			assert device == 'cpu'
			assert resume is None
			return {
				'completed': True,
				'selection_eligible': bool(audit['selection_eligible']),
			}

		return completed

	monkeypatch.setitem(wrapper_globals, '_shared', fake_shared)
	result = wrapper(
		object(),
		audit={'protocol_lock': {}, 'selection_eligible': True},
		device='cpu',
		resume=None,
	)

	assert result == {'completed': True, 'selection_eligible': True}
	assert "'selection_eligible': True" in RUNNER_PATH.read_text(encoding='utf-8')


def test_runner_imports_only_pinned_original_generic_helpers() -> None:
	text = RUNNER_PATH.read_text(encoding='utf-8')

	assert 'GAUSSIAN25_RUNNER_SHA256' in text
	assert 'a704e64a2da59c85a4e82e318bb3156c2cf74825ad50b4fb7463bd8dd2c1bccd' in text
	assert '50_base1ep/run_validation.py' not in text
	assert 'create_base1_final_result' not in text
	assert 'create_base1_selection_lock' not in text
	assert runner_parent_pins_are_exact(text)


def runner_parent_pins_are_exact(text: str) -> bool:
	"""Return whether the authorized p=.01 parent identities are source-pinned."""
	return all(
		value in text
		for value in (
			'3a83070718ce07f51756bfb91da6f792c6347f3009ca0290757bd93710fe1e2e',
			'872c18bed2465245b7fe9e3b3c9bb3163f466976ef7aa327eaea30ee9930e29d',
			"'authorizes_trace_drop_p002_followup': True",
			"'authorized_trace_drop_probability': 0.02",
		)
	)
