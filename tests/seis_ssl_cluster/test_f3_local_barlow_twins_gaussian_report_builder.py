from __future__ import annotations

import hashlib
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

BUILDER = Path(
	'experiments/f3/facies_benchmark_v2/'
	'111_local_barlow_twins_gaussian_view_v1/build_report.py'
)
FORCED_STD005_ID = 'local_barlow_twins_gaussian_noise_std005'
FORCED_STD010_ID = 'local_barlow_twins_gaussian_noise_std010'
IDENTITY_STD010_ID = 'local_barlow_twins_identity_gaussian_noise_std010'
LEGACY_CONTROL_ID = 'local_barlow_twins_legacy_flip_25ep'
SOURCE_IDS = (
	FORCED_STD005_ID,
	FORCED_STD010_ID,
	IDENTITY_STD010_ID,
	LEGACY_CONTROL_ID,
)
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZES = ('small', 'medium', 'large')


@pytest.fixture
def builder() -> dict[str, object]:
	return runpy.run_path(str(BUILDER))


def _final_result(*, selected_id: str, gate_open: bool) -> dict[str, object]:
	expected = [
		{
			'candidate_id': source_id,
			'layout_id': layout_id,
			'data_size': 'medium',
		}
		for source_id in SOURCE_IDS
		for layout_id in LAYOUT_IDS
	]
	if gate_open:
		post_lock_ids = {selected_id, LEGACY_CONTROL_ID}
		if selected_id == IDENTITY_STD010_ID:
			post_lock_ids.add(FORCED_STD010_ID)
		expected.extend(
			{
				'candidate_id': source_id,
				'layout_id': layout_id,
				'data_size': data_size,
			}
			for source_id in sorted(post_lock_ids)
			for layout_id in LAYOUT_IDS
			for data_size in ('small', 'large')
		)
	candidate_inputs = [
		{**row, 'macro_f1': 0.6} for row in expected
	]
	reached_sizes = DATA_SIZES if gate_open else ('medium',)
	random_inputs = [
		{
			'candidate_id': 'random',
			'layout_id': layout_id,
			'data_size': data_size,
			'macro_f1': 0.5,
		}
		for layout_id in LAYOUT_IDS
		for data_size in reached_sizes
	]
	return {
		'exact_expected_candidate_cells': expected,
		'candidate_inputs': candidate_inputs,
		'random_inputs': random_inputs,
		'medium_gate': {'gate_open': gate_open},
	}


def _validation_rows(final_result: dict[str, object]) -> list[dict[str, object]]:
	return [
		{
			'source_id': row['candidate_id'],
			'layout_id': row['layout_id'],
			'data_size': row['data_size'],
			'macro_f1': row['macro_f1'],
		}
		for key in ('candidate_inputs', 'random_inputs')
		for row in cast('list[dict[str, object]]', final_result[key])
	]


def test_report_contract_has_fixed_location_and_five_outputs(
	builder: dict[str, object],
) -> None:
	output = cast('Path', builder['REPORT_OUTPUT_DIR'])
	assert output.as_posix().endswith(
		'reports/f3/facies_benchmark_v2/local_barlow_twins_gaussian_view_v1'
	)
	assert builder['REPORT_FILENAMES'] == (
		'attempts.csv',
		'validation_cells.csv',
		'paired_deltas.csv',
		'summary.json',
		'summary.md',
	)
	assert set(cast('dict[str, object]', builder['ATTEMPT_CONFIG_PATHS'])) == set(
		SOURCE_IDS
	)


def test_evidence_accepts_complete_failed_medium_branch_and_rejects_drift(
	builder: dict[str, object],
) -> None:
	validator = cast('object', builder['_validated_evidence_rows'])
	payload = _final_result(selected_id=FORCED_STD005_ID, gate_open=False)
	candidate_rows, random_rows = validator(payload)
	assert len(candidate_rows) == 20
	assert len(random_rows) == 5

	duplicate = dict(payload)
	duplicate['candidate_inputs'] = [
		*cast('list[object]', payload['candidate_inputs']),
		cast('list[object]', payload['candidate_inputs'])[0],
	]
	with pytest.raises(ValueError, match='duplicates'):
		validator(duplicate)

	missing = dict(payload)
	missing['candidate_inputs'] = cast('list[object]', payload['candidate_inputs'])[
		:-1
	]
	with pytest.raises(ValueError, match='exact cell set'):
		validator(missing)


def test_paired_rows_cover_every_available_full_identity_contrast(
	builder: dict[str, object],
) -> None:
	payload = _final_result(selected_id=IDENTITY_STD010_ID, gate_open=True)
	rows = cast('object', builder['_paired_delta_rows'])(
		_validation_rows(payload),
		selected_id=IDENTITY_STD010_ID,
		base_pretraining_epochs=25,
	)
	roles = [row['comparison_role'] for row in rows]
	assert len(rows) == 80
	assert roles.count('candidate_minus_random') == 35
	assert roles.count('legacy_minus_random') == 15
	assert roles.count('selected_vs_legacy') == 15
	assert roles.count('identity_vs_forced_geometry') == 15
	assert all(
		row['left_value'] - row['right_value'] == row['delta'] for row in rows
	)


def test_paired_rows_support_medium_only_branch(builder: dict[str, object]) -> None:
	payload = _final_result(selected_id=FORCED_STD005_ID, gate_open=False)
	rows = cast('object', builder['_paired_delta_rows'])(
		_validation_rows(payload),
		selected_id=FORCED_STD005_ID,
		base_pretraining_epochs=25,
	)
	roles = [row['comparison_role'] for row in rows]
	assert len(rows) == 30
	assert roles.count('candidate_minus_random') == 15
	assert roles.count('legacy_minus_random') == 5
	assert roles.count('selected_vs_legacy') == 5
	assert roles.count('identity_vs_forced_geometry') == 5


def test_path_display_never_leaks_machine_specific_prefix(
	builder: dict[str, object], tmp_path: Path
) -> None:
	display = cast('object', builder['_display_path'])
	artifact_root = tmp_path / 'artifacts'
	assert display(
		artifact_root / 'validation/metrics.json', artifact_root=artifact_root
	) == '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/validation/metrics.json'
	repository_root = cast('Path', builder['REPOSITORY_ROOT'])
	assert display(
		repository_root / 'experiments/example.yaml', artifact_root=artifact_root
	) == 'experiments/example.yaml'
	with pytest.raises(ValueError, match='machine-specific path'):
		display(Path('/outside/report-input.json'), artifact_root=artifact_root)


def test_view_diagnostic_is_pinned_summarized_and_drift_checked(
	builder: dict[str, object], tmp_path: Path
) -> None:
	path = (
		tmp_path
		/ 'diagnostics/f3/local_barlow_twins_gaussian_view_v1/view_diagnostic.json'
	)
	path.parent.mkdir(parents=True)
	payload = {
		'schema_version': 1,
		'diagnostic': 'f3_local_barlow_twins_aligned_views',
		'metrics': {
			view_id: {
				'all_valid_physical_voxels': {
					'paired_correlation': correlation,
					'paired_rms': rms,
					'per_view_rms_from_unaugmented': rms / 2**0.5,
					'voxel_count': 100,
				}
			}
			for view_id, correlation, rms in (
				('legacy', 1.0, 0.0),
				('gaussian_noise_std005', 0.997, 0.071),
				('gaussian_noise_std010', 0.989, 0.141),
			)
		},
	}
	path.write_text(json.dumps(payload), encoding='utf-8')
	summarize = cast('object', builder['_view_diagnostic_summary'])
	summarize.__globals__['VIEW_DIAGNOSTIC_SHA256'] = hashlib.sha256(
		path.read_bytes()
	).hexdigest()
	result = summarize(tmp_path)
	assert result['path'].startswith('${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/')
	assert result['views']['legacy']['paired_correlation'] == 1.0
	assert 'fixed-position shortcut' in result['caveat']
	path.write_text('{}\n', encoding='utf-8')
	with pytest.raises(ValueError, match='SHA-256 drifted'):
		summarize(tmp_path)


@dataclass(frozen=True)
class _FakeSettings:
	final_result: Path


def _write_exclusive_json(path: Path, payload: object) -> None:
	raise AssertionError(f'real writer must be replaced: {path} {payload}')


def _fake_final_result_creator(
	settings: _FakeSettings,
	canonical: dict[str, object],
	*,
	created_at_utc: str | None = None,
) -> dict[str, object]:
	payload = {'created_at_utc': created_at_utc, 'value': canonical['value']}
	_write_exclusive_json(settings.final_result, payload)
	return payload


def test_final_result_replay_is_read_only_and_detects_drift(
	builder: dict[str, object], tmp_path: Path
) -> None:
	creator = _fake_final_result_creator
	runner = {'create_gaussian25_final_result': creator}
	settings = _FakeSettings(final_result=tmp_path / 'final.json')
	stored = {'created_at_utc': '2026-08-29T00:00:00Z', 'value': 1}
	replay = cast('object', builder['_replay_final_result'])
	assert replay(
		runner=runner,
		settings=settings,
		canonical={'value': 1},
		stored=stored,
	) == stored
	assert not (tmp_path / '.gaussian25-report-replay-sentinel.json').exists()
	assert creator.__globals__['_write_exclusive_json'] is _write_exclusive_json
	with pytest.raises(ValueError, match='differs from replayed'):
		replay(
			runner=runner,
			settings=settings,
			canonical={'value': 2},
			stored=stored,
		)


def test_protocol_lock_is_hashed_from_parsed_bytes_and_replayed_directly(
	builder: dict[str, object], tmp_path: Path
) -> None:
	path = tmp_path / 'protocol.json'
	raw = b'{"schema_version":1,"value":2}\n'
	path.write_bytes(raw)
	read_hashed = cast('object', builder['_read_hashed_json'])
	stored, digest = read_hashed(path, label='protocol lock')
	assert stored == {'schema_version': 1, 'value': 2}
	assert digest == hashlib.sha256(raw).hexdigest()

	calls: list[tuple[object, object]] = []

	def validate(settings: object, canonical: object) -> dict[str, object]:
		calls.append((settings, canonical))
		return {'schema_version': 1, 'value': 2}

	replay = cast('object', builder['_replay_protocol_lock'])
	settings = object()
	canonical = object()
	assert replay(
		runner={'validate_gaussian25_protocol_lock': validate},
		settings=settings,
		canonical=canonical,
		stored=stored,
	) == stored
	assert calls == [(settings, canonical)]
	with pytest.raises(ValueError, match='differs from replayed'):
		replay(
			runner={'validate_gaussian25_protocol_lock': validate},
			settings=settings,
			canonical=canonical,
			stored={'schema_version': 1, 'value': 3},
		)


def test_summary_exposes_direct_protocol_and_canonical_config_sources(
	builder: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	summary_builder = cast('object', builder['_summary_payload'])
	monkeypatch.setitem(
		summary_builder.__globals__,
		'_benchmark_provenance_summary',
		lambda **_kwargs: {},
	)
	settings = SimpleNamespace(
		canonical_five_way_config=tmp_path / 'canonical.yaml',
		protocol_lock=tmp_path / 'protocol.json',
		selection_lock=tmp_path / 'selection.json',
		final_result=tmp_path / 'final.json',
	)
	selection_lock = {
		'created_at_utc': '2026-08-29T00:00:00Z',
		'candidate_means': dict.fromkeys(SOURCE_IDS[:3], 0.5),
		'selected_view_policy': 'horizontal_flip_gaussian_noise_v1',
		'selected_gaussian_noise_std': 0.05,
		'tie_rule': 'maximum unrounded mean macro_f1',
		'tie_priority': list(SOURCE_IDS[:3]),
		'inputs': [],
		'fixed_strength_geometry_contrast': {},
	}
	final_result = {
		'created_at_utc': '2026-08-29T00:00:00Z',
		'selection_lock': {'selected_candidate_id': FORCED_STD005_ID},
		'repository_state': {},
		'medium_gate': {'gate_open': False},
		'base_pretraining_epochs': 25,
		'continuation_epochs': 25,
		'passed': False,
		'winner_candidate_id': None,
		'authorizes_next_base_duration': True,
		'failure_stage': 'medium_5of5',
		'arm_results': {},
		'gaussian_attribution': {},
		'identity_vs_forced_geometry': {},
	}
	summary = summary_builder(
		config_path=tmp_path / 'validation.yaml',
		validation_config_sha256='1' * 64,
		validation_runner_sha256='2' * 64,
		canonical_config_sha256='3' * 64,
		artifact_root=tmp_path,
		settings=settings,
		protocol_lock_sha256='4' * 64,
		selection_lock=selection_lock,
		selection_lock_sha256='5' * 64,
		final_result=final_result,
		final_result_sha256='6' * 64,
		validation_rows=[],
		paired_rows=[],
		attempt_rows=[],
		view_diagnostic={},
	)
	sources = cast('dict[str, dict[str, str]]', summary['source_artifacts'])
	assert sources['canonical_five_way_config'] == {
		'path': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/canonical.yaml',
		'sha256': '3' * 64,
	}
	assert sources['protocol_lock'] == {
		'path': '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/protocol.json',
		'sha256': '4' * 64,
	}


def test_history_and_publication_fail_closed(
	builder: dict[str, object], tmp_path: Path
) -> None:
	history_path = tmp_path / 'history.json'
	history = [
		{
			'epoch': epoch,
			'global_step': epoch * 625,
			'training_loss': 1.0 / epoch,
			'cross_correlation_diag_mean': 0.9,
		}
		for epoch in range(1, 26)
	]
	history_path.write_text(json.dumps(history), encoding='utf-8')
	completion = cast('object', builder['_history_completion'])
	step, loss, diagonal, digest = completion(
		history_path,
		epochs=25,
		expected_global_step=15_625,
		label='history',
	)
	assert (step, loss, diagonal) == (15_625, 0.04, 0.9)
	assert len(digest) == 64

	history[-1]['epoch'] = 24
	history_path.write_text(json.dumps(history), encoding='utf-8')
	with pytest.raises(ValueError, match='contiguous epoch history'):
		completion(
			history_path,
			epochs=25,
			expected_global_step=15_625,
			label='history',
		)

	filenames = cast('tuple[str, ...]', builder['REPORT_FILENAMES'])
	outputs = {name: f'{name}\n' for name in filenames}
	report_dir = tmp_path / 'reports' / 'result'
	publish = cast('object', builder['_publish_outputs'])
	publish(report_dir, outputs)
	assert {path.name for path in report_dir.iterdir()} == set(filenames)
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		publish(report_dir, outputs)


def test_publication_rejects_protocol_change_during_staging(
	builder: dict[str, object], tmp_path: Path
) -> None:
	protocol_lock = tmp_path / 'protocol.json'
	protocol_lock.write_text('{"value":1}\n', encoding='utf-8')
	digest = hashlib.sha256(protocol_lock.read_bytes()).hexdigest()
	filenames = cast('tuple[str, ...]', builder['REPORT_FILENAMES'])

	class MutatingOutputs(dict[str, str]):
		def __getitem__(self, key: str) -> str:
			value = super().__getitem__(key)
			if key == filenames[-1]:
				protocol_lock.write_text('{"value":2}\n', encoding='utf-8')
			return value

	outputs = MutatingOutputs({name: f'{name}\n' for name in filenames})
	report_dir = tmp_path / 'reports' / 'result'
	publish = cast('object', builder['_publish_outputs'])
	with pytest.raises(ValueError, match='protocol lock changed'):
		publish(
			report_dir,
			outputs,
			source_snapshots=((protocol_lock, digest, 'protocol lock'),),
		)
	assert not report_dir.exists()
	assert not list(report_dir.parent.glob('.result.staging-*'))


def test_training_artifact_chain_binds_yaml_resolved_checkpoint_and_history(
	builder: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	validator = cast('object', builder['_validate_training_artifact_chain'])
	resolved = {'stage': 'barlow_twins_training', 'train': {'epochs': 25}}
	payload = {
		'config': resolved,
		'epoch': 25,
		'global_step': 15_625,
		'metrics': {
			'training_loss': 0.4,
			'cross_correlation_diag_mean': 0.99,
		},
	}
	monkeypatch.setitem(validator.__globals__, 'load_config', lambda _path: {})
	monkeypatch.setitem(
		validator.__globals__,
		'resolve_barlow_twins_training_config',
		lambda _config: resolved,
	)
	monkeypatch.setitem(
		validator.__globals__,
		'load_checkpoint_metadata_without_weights',
		lambda _path: payload,
	)
	validator(
		config_path=tmp_path / 'config.yaml',
		resolved_config=resolved,
		checkpoint_path=tmp_path / 'latest.pt',
		final_epoch=25,
		final_global_step=15_625,
		final_training_loss=0.4,
		final_cross_correlation_diag_mean=0.99,
		label='candidate base',
	)
	payload['metrics'] = {
		'training_loss': 0.5,
		'cross_correlation_diag_mean': 0.99,
	}
	with pytest.raises(ValueError, match='differs from history'):
		validator(
			config_path=tmp_path / 'config.yaml',
			resolved_config=resolved,
			checkpoint_path=tmp_path / 'latest.pt',
			final_epoch=25,
			final_global_step=15_625,
			final_training_loss=0.4,
			final_cross_correlation_diag_mean=0.99,
			label='candidate base',
		)


def test_extraction_artifact_chain_binds_yaml_to_metadata(
	builder: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	validator = cast('object', builder['_validate_extraction_artifact_chain'])
	checkpoint = tmp_path / 'latest.pt'
	output_dir = tmp_path / 'embeddings'
	resolved = {
		'embeddings': {
			'checkpoint': str(checkpoint),
			'output_dir': str(output_dir),
		},
		'embedding': {
			'window_size': [128, 128, 128],
			'overlap': [64, 64, 64],
			'output_dtype': 'float16',
			'min_token_valid_fraction': 0.5,
			'amp': False,
			'amp_dtype': 'auto',
			'preprocessing_cache': {'mode': 'off'},
		},
	}
	metadata = {
		'checkpoint_path': str(checkpoint),
		'checkpoint_sha256': 'a' * 64,
		'window_size': [128, 128, 128],
		'overlap': [64, 64, 64],
		'output_dtype': 'float16',
		'min_token_valid_fraction': 0.5,
		'precision': {'amp_requested': False, 'amp_dtype_requested': 'auto'},
		'preprocessing_cache': {'requested_mode': 'off'},
	}
	monkeypatch.setitem(validator.__globals__, 'load_config', lambda _path: {})
	monkeypatch.setitem(
		validator.__globals__,
		'resolve_embedding_extraction_config',
		lambda _config: resolved,
	)
	validator(
		config_path=tmp_path / 'extract.yaml',
		checkpoint_path=checkpoint,
		embeddings_dir=output_dir,
		metadata=metadata,
		checkpoint_sha256='a' * 64,
		label='candidate',
	)
	metadata['overlap'] = [0, 0, 0]
	with pytest.raises(ValueError, match='differs from embedding metadata'):
		validator(
			config_path=tmp_path / 'extract.yaml',
			checkpoint_path=checkpoint,
			embeddings_dir=output_dir,
			metadata=metadata,
			checkpoint_sha256='a' * 64,
			label='candidate',
		)
