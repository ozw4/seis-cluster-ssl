from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from seis_ssl_cluster.embedding.writer import output_paths

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
	REPOSITORY_ROOT / 'experiments/f3/facies_benchmark_v2/'
	'111_local_barlow_twins_gaussian_view_v1/50_base1ep'
)
BUILDER_PATH = EXPERIMENT_ROOT / 'build_report.py'
HEX64 = 'a' * 64


@pytest.fixture
def builder() -> dict[str, object]:
	return runpy.run_path(str(BUILDER_PATH))


def _final_result(builder: dict[str, object], *, gate_open: bool) -> dict[str, object]:
	sizes = cast('tuple[str, ...]', builder['DATA_SIZES']) if gate_open else ('medium',)
	candidate_inputs: list[dict[str, object]] = []
	random_inputs: list[dict[str, object]] = []
	expected: list[dict[str, str]] = []
	for source_index, source_id in enumerate(builder['SOURCE_IDS']):
		for size in sizes:
			for layout_index, layout in enumerate(builder['LAYOUT_IDS']):
				value = 0.60 + source_index * 0.01 + layout_index * 0.001
				candidate_inputs.append(
					{
						'candidate_id': source_id,
						'layout_id': layout,
						'data_size': size,
						'macro_f1': value,
						'metrics_path': f'/artifact/{source_id}/{layout}/{size}.json',
						'metrics_sha256': HEX64,
						'candidate_audit_path': (
							f'/artifact/{source_id}/{layout}/{size}.audit.json'
						),
						'candidate_audit_sha256': HEX64,
						'base_checkpoint_sha256': HEX64,
						'continuation_init_checkpoint_sha256': HEX64,
						'final_checkpoint_sha256': HEX64,
					}
				)
				expected.append(
					{
						'candidate_id': cast('str', source_id),
						'layout_id': cast('str', layout),
						'data_size': size,
					}
				)
	for size in sizes:
		for layout_index, layout in enumerate(builder['LAYOUT_IDS']):
			random_inputs.append(
				{
					'candidate_id': 'random',
					'layout_id': layout,
					'data_size': size,
					'macro_f1': 0.50 + layout_index * 0.001,
					'metrics_path': f'/artifact/random/{layout}/{size}.json',
					'metrics_sha256': HEX64,
					'checkpoint_sha256': HEX64,
				}
			)
	return {
		'medium_gate': {'gate_open': gate_open},
		'exact_expected_candidate_cells': expected,
		'candidate_inputs': candidate_inputs,
		'random_inputs': random_inputs,
	}


def test_report_contract_is_fixed_validation_only_location(
	builder: dict[str, object],
) -> None:
	assert builder['REPORT_FILENAMES'] == (
		'attempts.csv',
		'validation_cells.csv',
		'paired_deltas.csv',
		'summary.json',
		'summary.md',
	)
	output = cast('Path', builder['REPORT_OUTPUT_DIR'])
	assert output.relative_to(REPOSITORY_ROOT).as_posix() == (
		'reports/f3/facies_benchmark_v2/local_barlow_twins_gaussian_view_v1/base1ep'
	)


@pytest.mark.parametrize(
	('gate_open', 'candidate_count', 'random_count', 'paired_count'),
	[(False, 10, 5, 15), (True, 30, 15, 45)],
)
def test_validation_and_paired_rows_cover_exact_reached_branch(
	builder: dict[str, object],
	gate_open: bool,  # noqa: FBT001
	candidate_count: int,
	random_count: int,
	paired_count: int,
) -> None:
	final = _final_result(builder, gate_open=gate_open)
	candidates, random = cast('Any', builder['_evidence_rows'])(final)
	assert len(candidates) == candidate_count
	assert len(random) == random_count
	rows = cast('Any', builder['_collect_validation_rows'])(
		final, artifact_root=Path('/artifact')
	)
	assert len(rows) == candidate_count + random_count
	assert all(row['evaluation_split'] == 'validation' for row in rows)
	assert all(row['aggregation_unit'] == 'unique_validation_voxel' for row in rows)
	paired = cast('Any', builder['_paired_rows'])(rows)
	assert len(paired) == paired_count
	assert {row['comparison_id'] for row in paired} == {
		'selected_gaussian_minus_random',
		'matched_legacy_minus_random',
		'selected_gaussian_minus_matched_legacy',
	}


def test_evidence_rejects_wrong_closed_branch_count(builder: dict[str, object]) -> None:
	final = _final_result(builder, gate_open=False)
	cast('list[object]', final['candidate_inputs']).pop()
	with pytest.raises(ValueError, match='do not match exact expected cells'):
		cast('Any', builder['_evidence_rows'])(final)


def test_report_publication_is_exact_and_exclusive(
	builder: dict[str, object], tmp_path: Path
) -> None:
	contents = {
		name: f'{name}\n'
		for name in cast('tuple[str, ...]', builder['REPORT_FILENAMES'])
	}
	output = tmp_path / 'report'
	cast('Any', builder['_publish_outputs'])(output, contents)
	assert {path.name for path in output.iterdir()} == set(contents)
	with pytest.raises(FileExistsError, match='already exists'):
		cast('Any', builder['_publish_outputs'])(output, contents)


@dataclass(frozen=True)
class _ReplaySettings:
	final_result: Path


def test_final_result_replay_captures_only_sentinel_and_detects_drift(
	builder: dict[str, object], tmp_path: Path
) -> None:
	scope: dict[str, object] = {
		'_write_exclusive_json': lambda _path, _payload: None,
	}
	exec(  # noqa: S102
		'def creator(settings, canonical, *, created_at_utc):\n'
		'    payload = {"created_at_utc": created_at_utc, "passed": False}\n'
		'    _write_exclusive_json(settings.final_result, payload)\n'
		'    return payload\n',
		scope,
	)
	stored = {'created_at_utc': '2026-08-30T00:00:00Z', 'passed': False}
	replayed = cast('Any', builder['_replay_final_result'])(
		runner={'create_base1_final_result': scope['creator']},
		settings=_ReplaySettings(tmp_path / 'final.json'),
		canonical=object(),
		stored=stored,
	)
	assert replayed == stored
	assert not (tmp_path / '.base1-report-replay-sentinel.json').exists()
	for drifted_passed in (True, 0):
		with pytest.raises(ValueError, match='differs from replayed'):
			cast('Any', builder['_replay_final_result'])(
				runner={'create_base1_final_result': scope['creator']},
				settings=_ReplaySettings(tmp_path / 'final.json'),
				canonical=object(),
				stored={**stored, 'passed': drifted_passed},
			)


def test_source_snapshots_include_embeddings_and_detect_tamper(
	builder: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	runner_path = tmp_path / 'runner.py'
	config_path = tmp_path / 'validation.yaml'
	protocol = tmp_path / 'protocol.json'
	selection = tmp_path / 'selection.json'
	final = tmp_path / 'final.json'
	for path in (runner_path, config_path, protocol, selection, final):
		path.write_bytes(path.name.encode())
	sources: dict[str, object] = {}
	config_paths: dict[str, dict[str, Path]] = {}
	for source_id in cast('tuple[str, ...]', builder['SOURCE_IDS']):
		base = tmp_path / source_id / 'base.pt'
		final_checkpoint = tmp_path / source_id / 'final.pt'
		embeddings_dir = tmp_path / source_id / 'embeddings'
		base.parent.mkdir(parents=True)
		base.write_bytes(b'base')
		final_checkpoint.write_bytes(b'final')
		files = output_paths(embeddings_dir, 'f3')
		files.embeddings.parent.mkdir(parents=True)
		files.embeddings.write_bytes(b'embeddings')
		files.metadata.write_bytes(b'{}')
		files.valid_tokens.write_bytes(b'mask')
		sources[source_id] = SimpleNamespace(
			base_checkpoint=base,
			final_checkpoint=final_checkpoint,
			embeddings_dir=embeddings_dir,
		)
		config_paths[source_id] = {}
		for role in ('base', 'continuation', 'extraction'):
			path = tmp_path / source_id / f'{role}.yaml'
			path.write_bytes(role.encode())
			config_paths[source_id][role] = path
	settings = SimpleNamespace(
		protocol_lock=protocol,
		selection_lock=selection,
		final_result=final,
		source_by_id=lambda source_id: sources[source_id],
	)
	monkeypatch.setitem(builder, 'RUNNER_PATH', runner_path)
	monkeypatch.setitem(builder, 'CONFIG_PATHS', config_paths)
	snapshots = cast('Any', builder['_source_snapshots'])(
		settings=settings,
		final_result={'candidate_inputs': [], 'random_inputs': []},
		config_path=config_path,
		survey_id='f3',
	)
	assert sum('embeddings array' in label for _, _, label in snapshots) == 2
	cast(
		'Path',
		output_paths(sources[builder['SELECTED_ID']].embeddings_dir, 'f3').embeddings,
	).write_bytes(b'tampered')
	with pytest.raises(ValueError, match='changed during report construction'):
		cast('Any', builder['_assert_snapshots_unchanged'])(snapshots)


def test_display_path_rejects_machine_specific_external_paths(
	builder: dict[str, object], tmp_path: Path
) -> None:
	with pytest.raises(ValueError, match='outside repository/artifact roots'):
		cast('Any', builder['_display_path'])(
			tmp_path / 'external', artifact_root=tmp_path / 'artifact'
		)
