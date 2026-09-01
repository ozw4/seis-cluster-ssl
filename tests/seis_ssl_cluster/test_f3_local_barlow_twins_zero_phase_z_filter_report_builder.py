"""Report-only projections for F3 zero-phase Z-filter experiment 114."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from seis_ssl_cluster.embedding.writer import file_sha256

if TYPE_CHECKING:
	from collections.abc import Mapping

REPORT_PATH = Path(
	'experiments/f3/facies_benchmark_v2/'
	'114_local_barlow_twins_zero_phase_z_filter_view_v1/build_report.py'
)
CANDIDATE_ID = 'local_barlow_twins_zero_phase_z_filter_w025_base1ep'
P002_ID = 'local_barlow_twins_horizontal_trace_drop_p002_base1ep'
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZES = ('small', 'medium', 'large')


@pytest.fixture(scope='module')
def report() -> dict[str, object]:
	return runpy.run_path(str(REPORT_PATH))


def _candidate_row(
	source_id: str,
	layout_id: str,
	data_size: str,
	value: float,
	*,
	root: Path,
) -> dict[str, object]:
	token = f'{source_id}-{layout_id}-{data_size}'
	return {
		'candidate_id': source_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'macro_f1': value,
		'metrics_path': str(root / token / 'metrics.json'),
		'metrics_sha256': '1' * 64,
		'candidate_audit_path': str(root / token / 'candidate_source_audit.json'),
		'candidate_audit_sha256': '2' * 64,
		'base_checkpoint_sha256': '3' * 64,
		'continuation_init_checkpoint_sha256': '3' * 64,
		'final_checkpoint_sha256': '4' * 64,
		'embeddings_sha256': '5' * 64,
		'embedding_metadata_sha256': '6' * 64,
		'valid_tokens_sha256': '7' * 64,
	}


def _random_row(
	layout_id: str, data_size: str, value: float, *, root: Path
) -> dict[str, object]:
	return {
		'candidate_id': 'random',
		'layout_id': layout_id,
		'data_size': data_size,
		'macro_f1': value,
		'metrics_path': str(root / 'random' / layout_id / data_size / 'metrics.json'),
		'metrics_sha256': '8' * 64,
		'checkpoint_sha256': '9' * 64,
	}


def _final_result(*, root: Path, gate_open: bool) -> dict[str, object]:
	sizes = DATA_SIZES if gate_open else ('medium',)
	candidate = [
		_candidate_row(CANDIDATE_ID, layout, size, 0.6, root=root)
		for size in sizes
		for layout in LAYOUT_IDS
	]
	random = [
		_random_row(layout, size, 0.5, root=root)
		for size in sizes
		for layout in LAYOUT_IDS
	]
	controls = [
		_candidate_row(P002_ID, layout, 'medium', 0.55, root=root)
		for layout in LAYOUT_IDS
	]
	return {
		'exact_expected_candidate_cells': [
			{
				'candidate_id': CANDIDATE_ID,
				'layout_id': layout,
				'data_size': size,
			}
			for size in sizes
			for layout in LAYOUT_IDS
		],
		'candidate_inputs': candidate,
		'random_inputs': random,
		'frozen_medium_control_inputs': controls,
		'medium_gate': {
			'gate_open': gate_open,
			'positive_delta_count': 5,
		},
		'parent_result': {'path': str(root / 'parent.json'), 'sha256': 'a' * 64},
		'arm_results': {
			CANDIDATE_ID: {
				'positive_delta_count': len(candidate),
				'wins_all_15_over_random': gate_open,
			}
		},
		'zero_phase_z_filter_attribution': {},
		'passed': gate_open,
		'winner_candidate_id': CANDIDATE_ID if gate_open else None,
		'failure_stage': None if gate_open else 'medium_5of5',
		'authorizes_additional_view_followup': False,
		'authorized_additional_view_configuration': None,
	}


@pytest.mark.parametrize(
	('gate_open', 'expected_rows', 'expected_pairs'),
	[(False, 15, 10), (True, 35, 20)],
)
def test_report_rows_distinguish_live_random_and_frozen_controls(
	report: Mapping[str, object],
	tmp_path: Path,
	gate_open: bool,  # noqa: FBT001
	expected_rows: int,
	expected_pairs: int,
) -> None:
	collector = cast('Any', report['_collect_validation_rows'])
	pairer = cast('Any', report['_paired_rows'])
	artifact_root = tmp_path / 'artifacts'
	rows = collector(
		_final_result(root=artifact_root, gate_open=gate_open),
		artifact_root=artifact_root,
	)
	paired = pairer(rows)

	assert len(rows) == expected_rows
	assert len(paired) == expected_pairs
	assert {(row['source_id'], row['evidence_origin']) for row in rows} == {
		(CANDIDATE_ID, 'live_zero_phase_z_filter_w025'),
		('random', 'protocol_frozen_random'),
		(P002_ID, 'frozen_parent_p002_medium_control'),
	}
	control_pairs = [row for row in paired if row['right_source_id'] == P002_ID]
	assert len(control_pairs) == 5
	assert {row['data_size'] for row in control_pairs} == {'medium'}
	assert all(row['strict_positive'] is True for row in paired)


def test_report_publisher_writes_exactly_five_files_exclusively(
	report: Mapping[str, object], tmp_path: Path
) -> None:
	publisher = cast('Any', report['_publish_outputs'])
	names = cast('tuple[str, ...]', report['REPORT_FILENAMES'])
	output = tmp_path / 'report'
	contents = {name: f'{name}\n' for name in names}

	publisher(output, contents)

	assert {path.name for path in output.iterdir()} == set(names)
	with pytest.raises(FileExistsError, match='already exists'):
		publisher(output, contents)
	with pytest.raises(ValueError, match='exactly the five'):
		publisher(tmp_path / 'other', {'summary.json': '{}\n'})


def test_report_schema_includes_evidence_origin_and_no_pipeline_inputs(
	report: Mapping[str, object],
) -> None:
	validation_fields = cast('tuple[str, ...]', report['VALIDATION_FIELDS'])
	filenames = cast('tuple[str, ...]', report['REPORT_FILENAMES'])

	assert 'evidence_origin' in validation_fields
	assert 'medium_p002_macro_f1_mean' in cast(
		'tuple[str, ...]', report['ATTEMPT_FIELDS']
	)
	assert 'medium_delta_vs_p002_mean' in cast(
		'tuple[str, ...]', report['ATTEMPT_FIELDS']
	)
	assert filenames == (
		'attempts.csv',
		'validation_cells.csv',
		'paired_deltas.csv',
		'summary.json',
		'summary.md',
	)
	assert 'reports/' not in (
		REPORT_PATH.parent / '30_validation/01_candidate.yaml'
	).read_text(encoding='utf-8')


def test_report_pins_exact_runner_sha(report: Mapping[str, object]) -> None:
	runner_path = cast('Path', report['RUNNER_PATH'])
	runner_sha = cast('str', report['RUNNER_SHA256'])

	assert len(runner_sha) == 64
	assert runner_sha == file_sha256(runner_path)
