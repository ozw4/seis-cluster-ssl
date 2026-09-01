# ruff: noqa: INP001
"""Build five human-only projections of the immutable Z-filter result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import runpy
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = EXPERIMENT_ROOT / 'run_validation.py'
RUNNER_SHA256 = '3e5c7e633cb6f5007a3c8ad35a0b458ad4714a7714805d7da50902eaac4adf81'
DEFAULT_CONFIG = EXPERIMENT_ROOT / '30_validation/01_candidate.yaml'
REPORT_OUTPUT_DIR = (
	REPOSITORY_ROOT / 'reports/f3/facies_benchmark_v2/'
	'local_barlow_twins_zero_phase_z_filter_view_v1/base1ep'
)
REPORT_FILENAMES = (
	'attempts.csv',
	'validation_cells.csv',
	'paired_deltas.csv',
	'summary.json',
	'summary.md',
)

CANDIDATE_ID = 'local_barlow_twins_zero_phase_z_filter_w025_base1ep'
P002_CONTROL_ID = 'local_barlow_twins_horizontal_trace_drop_p002_base1ep'
RANDOM_ID = 'random'
SOURCE_IDS = (
	CANDIDATE_ID,
	RANDOM_ID,
	P002_CONTROL_ID,
)
CONTROL_IDS = (P002_CONTROL_ID,)
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZES = ('small', 'medium', 'large')
BASE_EPOCHS = 1
CONTINUATION_EPOCHS = 25
VALIDATION_AGGREGATION_UNIT = 'unique_validation_voxel'
EXPECTED_AUGMENTATIONS = {
	'policy': 'horizontal_flip_zero_phase_z_filter_v1',
	'horizontal_flip_probability': 0.5,
	'z_filter_side_weight': 0.25,
}
PARENT_RESULT_SHA256 = (
	'8b27c1141b5e7740653f8585acb0a9e978e74a82355bbbcdb947fd888cc711cd'
)

CONFIG_PATHS = {
	'base': (
		EXPERIMENT_ROOT
		/ '10_stage1/zero_phase_z_filter_w025_base1ep/01_screen_1ep.yaml'
	),
	'continuation': (
		EXPERIMENT_ROOT
		/ '15_stage2/zero_phase_z_filter_w025_base1ep/01_continue_25ep.yaml'
	),
	'extraction': (
		EXPERIMENT_ROOT
		/ '20_embeddings/01_extract_zero_phase_z_filter_w025_base1ep.yaml'
	),
}

ATTEMPT_FIELDS = (
	'attempt_order',
	'candidate_id',
	'role',
	'augmentations_json',
	'base_pretraining_epochs',
	'continuation_epochs',
	'base_config_path',
	'base_config_sha256',
	'continuation_config_path',
	'continuation_config_sha256',
	'extraction_config_path',
	'extraction_config_sha256',
	'base_checkpoint_path',
	'base_checkpoint_sha256',
	'final_checkpoint_path',
	'final_checkpoint_sha256',
	'embeddings_dir',
	'embeddings_sha256',
	'embedding_metadata_sha256',
	'valid_tokens_sha256',
	'evaluated_cell_count',
	'medium_macro_f1_mean',
	'medium_random_macro_f1_mean',
	'medium_delta_vs_random_mean',
	'medium_p002_macro_f1_mean',
	'medium_delta_vs_p002_mean',
	'medium_positive_count',
	'positive_vs_random_count',
	'wins_all_15_over_random',
	'final_outcome',
)

VALIDATION_FIELDS = (
	'source_id',
	'source_role',
	'evidence_origin',
	'base_pretraining_epochs',
	'continuation_epochs',
	'layout_id',
	'data_size',
	'evaluation_split',
	'aggregation_unit',
	'macro_f1',
	'random_macro_f1',
	'delta_vs_random',
	'metrics_path',
	'metrics_sha256',
	'candidate_audit_path',
	'candidate_audit_sha256',
	'base_checkpoint_sha256',
	'continuation_init_checkpoint_sha256',
	'final_checkpoint_sha256',
)

PAIRED_FIELDS = (
	'base_pretraining_epochs',
	'layout_id',
	'data_size',
	'comparison_id',
	'left_source_id',
	'right_source_id',
	'left_evidence_origin',
	'right_evidence_origin',
	'left_value',
	'right_value',
	'delta',
	'strict_positive',
)


def _load_runner_namespace() -> dict[str, object]:
	if RUNNER_PATH.is_symlink() or not RUNNER_PATH.is_file():
		raise FileNotFoundError(f'missing Z-filter validation runner: {RUNNER_PATH}')
	if file_sha256(RUNNER_PATH) != RUNNER_SHA256:
		raise ValueError('Z-filter validation runner SHA-256 changed')
	namespace = runpy.run_path(str(RUNNER_PATH))
	required = {
		'validation_settings_from_mapping',
		'_canonical_config',
		'validate_zero_phase_z_filter_protocol_lock',
		'create_zero_phase_z_filter_final_result',
	}
	missing = required - set(namespace)
	if missing:
		raise RuntimeError(f'Z-filter runner API is incomplete: {sorted(missing)!r}')
	return namespace


def _replay_final_result(
	*,
	runner: Mapping[str, object],
	settings: object,
	canonical: object,
	stored: Mapping[str, object],
) -> Mapping[str, object]:
	"""Capture a complete final-result replay without scientific writes."""
	created_at = stored.get('created_at_utc')
	if not isinstance(created_at, str):
		raise TypeError('final result created_at_utc must be a string')
	creator = cast('Any', runner['create_zero_phase_z_filter_final_result'])
	writer_globals = creator.__globals__
	original_writer = writer_globals.get('_write_exclusive_json')
	if not callable(original_writer):
		raise TypeError('Z-filter runner exclusive writer is unavailable')
	final_path = cast('Path', settings.final_result)
	sentinel = final_path.with_name('.zero-phase-z-filter-report-replay-sentinel.json')
	if sentinel.exists() or sentinel.is_symlink():
		raise FileExistsError(f'report replay sentinel already exists: {sentinel}')
	captured: list[tuple[Path, Mapping[str, object]]] = []

	def capture(path: Path, payload: Mapping[str, object]) -> None:
		if Path(path) != sentinel or captured:
			raise RuntimeError('final-result replay attempted an unexpected write')
		captured.append((Path(path), dict(payload)))

	writer_globals['_write_exclusive_json'] = capture
	try:
		replayed = creator(
			replace(settings, final_result=sentinel),
			canonical,
			created_at_utc=created_at,
		)
	finally:
		writer_globals['_write_exclusive_json'] = original_writer
	if sentinel.exists() or sentinel.is_symlink():
		raise RuntimeError('final-result replay unexpectedly wrote its sentinel')
	if len(captured) != 1 or not _type_sensitive_equal(captured[0][1], replayed):
		raise RuntimeError('final-result replay did not capture exactly one payload')
	if not _type_sensitive_equal(stored, replayed):
		raise ValueError('immutable final result differs from replayed live evidence')
	return cast('Mapping[str, object]', replayed)


def _evidence_rows(
	final_result: Mapping[str, object],
) -> tuple[
	list[Mapping[str, object]],
	list[Mapping[str, object]],
	list[Mapping[str, object]],
]:
	expected_values = _list(final_result.get('exact_expected_candidate_cells'), 'cells')
	candidate_values = _list(final_result.get('candidate_inputs'), 'candidate_inputs')
	random_values = _list(final_result.get('random_inputs'), 'random_inputs')
	control_values = _list(
		final_result.get('frozen_medium_control_inputs'),
		'frozen_medium_control_inputs',
	)
	expected = {_cell_key(_row(value, 'expected cell')) for value in expected_values}
	candidates = [_row(value, 'candidate input') for value in candidate_values]
	random = [_row(value, 'random input') for value in random_values]
	controls = [_row(value, 'control input') for value in control_values]
	actual = {_cell_key(row) for row in candidates}
	if len(expected) != len(expected_values) or actual != expected:
		raise ValueError('candidate inputs do not match exact expected cells')
	gate = _mapping(final_result.get('medium_gate'), 'medium_gate')
	gate_open = _bool(gate.get('gate_open'), 'medium_gate.gate_open')
	expected_candidate_count = 15 if gate_open else 5
	expected_random_count = 15 if gate_open else 5
	if len(candidates) != expected_candidate_count:
		raise ValueError('candidate evidence count differs from reached branch')
	if len(random) != expected_random_count:
		raise ValueError('random evidence count differs from reached branch')
	random_cells = {_cell_key(row) for row in random}
	expected_random = {
		(RANDOM_ID, layout_id, data_size)
		for layout_id in LAYOUT_IDS
		for data_size in (DATA_SIZES if gate_open else ('medium',))
	}
	if len(random_cells) != len(random) or random_cells != expected_random:
		raise ValueError('random evidence does not match reached branch')
	expected_controls = {
		(source_id, layout_id, 'medium')
		for source_id in CONTROL_IDS
		for layout_id in LAYOUT_IDS
	}
	control_cells = {_cell_key(row) for row in controls}
	if len(controls) != 5 or control_cells != expected_controls:
		raise ValueError('frozen controls must be exactly five p=.02 medium cells')
	for evidence in (*candidates, *random, *controls):
		_macro_f1(evidence)
	return candidates, random, controls


def _collect_validation_rows(
	final_result: Mapping[str, object], *, artifact_root: Path
) -> list[dict[str, object]]:
	candidates, random, controls = _evidence_rows(final_result)
	random_scores = {
		(cast('str', row['layout_id']), cast('str', row['data_size'])): _macro_f1(row)
		for row in random
	}
	rows: list[dict[str, object]] = []
	for evidence in (*candidates, *random, *controls):
		source_id, layout_id, data_size = _cell_key(evidence)
		value = _macro_f1(evidence)
		random_value = random_scores[layout_id, data_size]
		is_random = source_id == RANDOM_ID
		role, origin = _source_role_and_origin(source_id)
		rows.append(
			{
				'source_id': source_id,
				'source_role': role,
				'evidence_origin': origin,
				'base_pretraining_epochs': '' if is_random else BASE_EPOCHS,
				'continuation_epochs': '' if is_random else CONTINUATION_EPOCHS,
				'layout_id': layout_id,
				'data_size': data_size,
				'evaluation_split': 'validation',
				'aggregation_unit': VALIDATION_AGGREGATION_UNIT,
				'macro_f1': value,
				'random_macro_f1': random_value,
				'delta_vs_random': value - random_value,
				'metrics_path': _display_path(
					_required_path(evidence, 'metrics_path'),
					artifact_root=artifact_root,
				),
				'metrics_sha256': _required_sha(evidence, 'metrics_sha256'),
				'candidate_audit_path': (
					''
					if is_random
					else _display_path(
						_required_path(evidence, 'candidate_audit_path'),
						artifact_root=artifact_root,
					)
				),
				'candidate_audit_sha256': (
					''
					if is_random
					else _required_sha(evidence, 'candidate_audit_sha256')
				),
				'base_checkpoint_sha256': (
					''
					if is_random
					else _required_sha(evidence, 'base_checkpoint_sha256')
				),
				'continuation_init_checkpoint_sha256': (
					''
					if is_random
					else _required_sha(evidence, 'continuation_init_checkpoint_sha256')
				),
				'final_checkpoint_sha256': (
					_required_sha(evidence, 'checkpoint_sha256')
					if is_random
					else _required_sha(evidence, 'final_checkpoint_sha256')
				),
			}
		)
	return sorted(
		rows,
		key=lambda row: (
			SOURCE_IDS.index(cast('str', row['source_id'])),
			DATA_SIZES.index(cast('str', row['data_size'])),
			LAYOUT_IDS.index(cast('str', row['layout_id'])),
		),
	)


def _source_role_and_origin(source_id: str) -> tuple[str, str]:
	values = {
		CANDIDATE_ID: (
			'separately_preregistered_zero_phase_z_bandwidth_view_followup',
			'live_zero_phase_z_filter_w025',
		),
		RANDOM_ID: ('canonical_random', 'protocol_frozen_random'),
		P002_CONTROL_ID: (
			'frozen_trace_drop_p002_control',
			'frozen_parent_p002_medium_control',
		),
	}
	try:
		return values[source_id]
	except KeyError as error:
		raise ValueError(f'unknown report source: {source_id!r}') from error


def _collect_attempt_rows(
	*,
	settings: object,
	canonical: object,
	final_result: Mapping[str, object],
	validation_rows: Sequence[Mapping[str, object]],
	artifact_root: Path,
) -> list[dict[str, object]]:
	for path in CONFIG_PATHS.values():
		_file_sha(path, label='attempt config')
	source = settings.candidate
	files = output_paths(source.embeddings_dir, canonical.dataset['name'])
	arm_results = _mapping(final_result.get('arm_results'), 'arm_results')
	result = _mapping(arm_results.get(CANDIDATE_ID), 'candidate arm result')
	source_rows = [row for row in validation_rows if row['source_id'] == CANDIDATE_ID]
	medium = [row for row in source_rows if row['data_size'] == 'medium']
	if len(medium) != 5:
		raise ValueError('Z-filter report must contain five medium cells')
	p002_medium = [
		row
		for row in validation_rows
		if row['source_id'] == P002_CONTROL_ID and row['data_size'] == 'medium'
	]
	if len(p002_medium) != 5:
		raise ValueError('Z-filter report must contain five p=.02 control cells')
	p002_by_layout = {
		cast('str', row['layout_id']): float(row['macro_f1']) for row in p002_medium
	}
	positive_count = sum(float(row['delta_vs_random']) > 0.0 for row in source_rows)
	if not _type_sensitive_equal(result.get('positive_delta_count'), positive_count):
		raise ValueError('candidate result positive-count mismatch')
	return [
		{
			'attempt_order': 1,
			'candidate_id': CANDIDATE_ID,
			'role': source.role,
			'augmentations_json': json.dumps(
				EXPECTED_AUGMENTATIONS, separators=(',', ':'), sort_keys=True
			),
			'base_pretraining_epochs': BASE_EPOCHS,
			'continuation_epochs': CONTINUATION_EPOCHS,
			'base_config_path': _display_path(
				CONFIG_PATHS['base'], artifact_root=artifact_root
			),
			'base_config_sha256': file_sha256(CONFIG_PATHS['base']),
			'continuation_config_path': _display_path(
				CONFIG_PATHS['continuation'], artifact_root=artifact_root
			),
			'continuation_config_sha256': file_sha256(CONFIG_PATHS['continuation']),
			'extraction_config_path': _display_path(
				CONFIG_PATHS['extraction'], artifact_root=artifact_root
			),
			'extraction_config_sha256': file_sha256(CONFIG_PATHS['extraction']),
			'base_checkpoint_path': _display_path(
				source.base_checkpoint, artifact_root=artifact_root
			),
			'base_checkpoint_sha256': _file_sha(
				source.base_checkpoint, label='base checkpoint'
			),
			'final_checkpoint_path': _display_path(
				source.final_checkpoint, artifact_root=artifact_root
			),
			'final_checkpoint_sha256': _file_sha(
				source.final_checkpoint, label='final checkpoint'
			),
			'embeddings_dir': _display_path(
				source.embeddings_dir, artifact_root=artifact_root
			),
			'embeddings_sha256': _file_sha(files.embeddings, label='embeddings'),
			'embedding_metadata_sha256': _file_sha(
				files.metadata, label='embedding metadata'
			),
			'valid_tokens_sha256': _file_sha(
				files.valid_tokens, label='valid-token mask'
			),
			'evaluated_cell_count': len(source_rows),
			'medium_macro_f1_mean': fmean(float(row['macro_f1']) for row in medium),
			'medium_random_macro_f1_mean': fmean(
				float(row['random_macro_f1']) for row in medium
			),
			'medium_delta_vs_random_mean': fmean(
				float(row['delta_vs_random']) for row in medium
			),
			'medium_p002_macro_f1_mean': fmean(p002_by_layout.values()),
			'medium_delta_vs_p002_mean': fmean(
				float(row['macro_f1']) - p002_by_layout[cast('str', row['layout_id'])]
				for row in medium
			),
			'medium_positive_count': sum(
				float(row['delta_vs_random']) > 0.0 for row in medium
			),
			'positive_vs_random_count': positive_count,
			'wins_all_15_over_random': result['wins_all_15_over_random'],
			'final_outcome': (
				'passed' if final_result.get('passed') is True else 'failed'
			),
		}
	]


def _paired_rows(
	validation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	index = {
		(
			cast('str', row['source_id']),
			cast('str', row['layout_id']),
			cast('str', row['data_size']),
		): row
		for row in validation_rows
	}
	comparisons = (
		(CANDIDATE_ID, RANDOM_ID, 'zero_phase_z_filter_w025_minus_random'),
		(
			CANDIDATE_ID,
			P002_CONTROL_ID,
			'zero_phase_z_filter_w025_minus_trace_drop_p002',
		),
	)
	rows: list[dict[str, object]] = []
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZES:
			for left_id, right_id, comparison_id in comparisons:
				left_key = (left_id, layout_id, data_size)
				right_key = (right_id, layout_id, data_size)
				if left_key not in index or right_key not in index:
					continue
				left_row = index[left_key]
				right_row = index[right_key]
				left = float(left_row['macro_f1'])
				right = float(right_row['macro_f1'])
				delta = left - right
				rows.append(
					{
						'base_pretraining_epochs': BASE_EPOCHS,
						'layout_id': layout_id,
						'data_size': data_size,
						'comparison_id': comparison_id,
						'left_source_id': left_id,
						'right_source_id': right_id,
						'left_evidence_origin': left_row['evidence_origin'],
						'right_evidence_origin': right_row['evidence_origin'],
						'left_value': left,
						'right_value': right,
						'delta': delta,
						'strict_positive': delta > 0.0,
					}
				)
	return rows


def _summary_payload(  # noqa: PLR0913
	*,
	config_path: Path,
	protocol_sha: str,
	final_sha: str,
	final_result: Mapping[str, object],
	attempts: Sequence[Mapping[str, object]],
	validation_rows: Sequence[Mapping[str, object]],
	paired_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	gate = _mapping(final_result.get('medium_gate'), 'medium_gate')
	return {
		'schema_version': 1,
		'report_type': 'f3_local_barlow_twins_zero_phase_z_filter_validation_report_v1',
		'validation_only': True,
		'pipeline_input': False,
		'candidate_id': CANDIDATE_ID,
		'augmentations': dict(EXPECTED_AUGMENTATIONS),
		'base_pretraining_epochs': BASE_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'configuration': {
			'path': str(config_path.relative_to(REPOSITORY_ROOT)),
			'sha256': file_sha256(config_path),
		},
		'parent_result': final_result['parent_result'],
		'parent_result_sha256': PARENT_RESULT_SHA256,
		'protocol_lock_sha256': protocol_sha,
		'final_result_sha256': final_sha,
		'medium_gate': dict(gate),
		'attempt_count': len(attempts),
		'live_candidate_validation_cell_count': sum(
			row['source_id'] == CANDIDATE_ID for row in validation_rows
		),
		'random_validation_cell_count': sum(
			row['source_id'] == RANDOM_ID for row in validation_rows
		),
		'frozen_control_validation_cell_count': sum(
			row['source_id'] in CONTROL_IDS for row in validation_rows
		),
		'validation_row_count': len(validation_rows),
		'paired_delta_count': len(paired_rows),
		'attempts': [dict(row) for row in attempts],
		'arm_results': final_result['arm_results'],
		'zero_phase_z_filter_attribution': final_result[
			'zero_phase_z_filter_attribution'
		],
		'passed': final_result['passed'],
		'winner_candidate_id': final_result['winner_candidate_id'],
		'failure_stage': final_result['failure_stage'],
		'authorizes_additional_view_followup': final_result[
			'authorizes_additional_view_followup'
		],
		'authorized_additional_view_configuration': final_result[
			'authorized_additional_view_configuration'
		],
		'source_note': (
			'Human-readable validation projection only; artifacts and immutable '
			'protocol/final result remain the scientific source of truth. The closed '
			'p=.02 trace-drop branch is a frozen attribution control only.'
		),
	}


def _summary_markdown(summary: Mapping[str, object]) -> str:
	status = 'PASS' if summary['passed'] else 'FAIL'
	gate = _mapping(summary.get('medium_gate'), 'summary medium_gate')
	attempt = cast('Sequence[Mapping[str, object]]', summary['attempts'])[0]
	return '\n'.join(
		[
			'# F3 Barlow Twins zero-phase Z-filter w=.25 validation',
			'',
			(
				'This is a validation-only, human-readable projection. '
				'It is not a pipeline input.'
			),
			'',
			f'- Decision: **{status}**',
			f'- Candidate: `{CANDIDATE_ID}`',
			f'- Medium gate open: `{gate.get("gate_open")}`',
			f'- Strict medium wins: `{gate.get("positive_delta_count")}/5`',
			(
				'- Strict wins over random: '
				f'`{attempt["positive_vs_random_count"]}/'
				f'{attempt["evaluated_cell_count"]}`'
			),
			f'- Validation rows: `{summary["validation_row_count"]}`',
			(
				'- Frozen parent control rows: '
				f'`{summary["frozen_control_validation_cell_count"]}`'
			),
			'',
			'## Attempt',
			'',
			(
				'| Z-filter mean | Random mean | Delta vs random | p=.02 mean | '
				'Delta vs p=.02 | 15/15 |'
			),
			'| ---: | ---: | ---: | ---: | ---: | --- |',
			(
				f'| {float(attempt["medium_macro_f1_mean"]):.9f} | '
				f'{float(attempt["medium_random_macro_f1_mean"]):.9f} | '
				f'{float(attempt["medium_delta_vs_random_mean"]):+.9f} | '
				f'{float(attempt["medium_p002_macro_f1_mean"]):.9f} | '
				f'{float(attempt["medium_delta_vs_p002_mean"]):+.9f} | '
				f'{attempt["wins_all_15_over_random"]} |'
			),
			'',
			(
				'The p=.02 rows are frozen parent medium controls for direct '
				'attribution only. They do not affect the gate or pass decision.'
			),
			'',
		]
	)


def _csv_text(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> str:
	stream = io.StringIO(newline='')
	writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator='\n')
	writer.writeheader()
	for row in rows:
		if set(row) != set(fields):
			raise ValueError('report CSV row fields differ from the fixed schema')
		writer.writerow(row)
	return stream.getvalue()


def _source_snapshots(
	*,
	settings: object,
	final_result: Mapping[str, object],
	config_path: Path,
	survey_id: str,
) -> list[tuple[Path, str, str]]:
	parent = _mapping(final_result.get('parent_result'), 'parent_result')
	paths: list[tuple[Path, str, str]] = [
		(config_path, file_sha256(config_path), 'validation config'),
		(RUNNER_PATH, file_sha256(RUNNER_PATH), 'validation runner'),
		(settings.protocol_lock, file_sha256(settings.protocol_lock), 'protocol lock'),
		(settings.final_result, file_sha256(settings.final_result), 'final result'),
		(
			_required_path(parent, 'path'),
			_required_sha(parent, 'sha256'),
			'parent final result',
		),
		(
			settings.candidate.base_checkpoint,
			_file_sha(settings.candidate.base_checkpoint, label='base checkpoint'),
			'candidate base checkpoint',
		),
		(
			settings.candidate.final_checkpoint,
			_file_sha(settings.candidate.final_checkpoint, label='final checkpoint'),
			'candidate final checkpoint',
		),
	]
	for role, path in CONFIG_PATHS.items():
		paths.append((path, _file_sha(path, label=f'{role} config'), f'{role} config'))
	files = output_paths(settings.candidate.embeddings_dir, survey_id)
	for role, path in (
		('embeddings array', files.embeddings),
		('embedding metadata', files.metadata),
		('valid-token mask', files.valid_tokens),
	):
		paths.append((path, _file_sha(path, label=role), role))
	for label in (
		'candidate_inputs',
		'random_inputs',
		'frozen_medium_control_inputs',
	):
		for value in _list(final_result.get(label), label):
			row = _row(value, label)
			metrics = _required_path(row, 'metrics_path')
			paths.append(
				(metrics, _required_sha(row, 'metrics_sha256'), f'{label} metrics')
			)
			if row.get('candidate_id') != RANDOM_ID:
				audit = _required_path(row, 'candidate_audit_path')
				paths.append(
					(
						audit,
						_required_sha(row, 'candidate_audit_sha256'),
						f'{label} candidate audit',
					)
				)
	return paths


def _assert_snapshots_unchanged(
	snapshots: Sequence[tuple[Path, str, str]],
) -> None:
	for path, expected_sha, label in snapshots:
		if _file_sha(path, label=label) != expected_sha:
			raise ValueError(f'{label} changed during report construction')


def _publish_outputs(output_dir: Path, contents: Mapping[str, str]) -> None:
	"""Publish exactly five files without replacing an existing report path."""
	if set(contents) != set(REPORT_FILENAMES):
		raise ValueError('report publisher requires exactly the five fixed outputs')
	if output_dir.exists() or output_dir.is_symlink():
		raise FileExistsError(f'report output already exists: {output_dir}')
	output_dir.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(
		dir=output_dir.parent, prefix='.zero-phase-z-filter-report-staging-'
	) as temporary:
		stage = Path(temporary)
		for name in REPORT_FILENAMES:
			(stage / name).write_text(contents[name], encoding='utf-8')
		output_dir.mkdir()
		try:
			for name in REPORT_FILENAMES:
				os.link(stage / name, output_dir / name)
		except Exception:
			for name in REPORT_FILENAMES:
				(output_dir / name).unlink(missing_ok=True)
			output_dir.rmdir()
			raise


def build_report(
	config_path: Path = DEFAULT_CONFIG,
	*,
	output_dir: Path = REPORT_OUTPUT_DIR,
) -> dict[str, object]:
	"""Replay immutable evidence and publish one exclusive report directory."""
	config_path = config_path.resolve()
	if config_path != DEFAULT_CONFIG.resolve():
		raise ValueError('Z-filter report must use the fixed validation config')
	runner = _load_runner_namespace()
	settings = cast('Any', runner['validation_settings_from_mapping'])(
		load_config(config_path)
	)
	canonical = cast('Any', runner['_canonical_config'])(settings)
	protocol, protocol_sha = _read_hashed_json(
		settings.protocol_lock, label='Z-filter protocol lock'
	)
	final_result, final_sha = _read_hashed_json(
		settings.final_result, label='Z-filter final result'
	)
	protocol_replayed = cast(
		'Any', runner['validate_zero_phase_z_filter_protocol_lock']
	)(settings, canonical)
	if not _type_sensitive_equal(protocol, protocol_replayed):
		raise ValueError('stored protocol differs from live replay')
	_replay_final_result(
		runner=runner,
		settings=settings,
		canonical=canonical,
		stored=final_result,
	)
	artifact_root = cast('Path', canonical.artifact_root)
	validation_rows = _collect_validation_rows(
		final_result, artifact_root=artifact_root
	)
	attempts = _collect_attempt_rows(
		settings=settings,
		canonical=canonical,
		final_result=final_result,
		validation_rows=validation_rows,
		artifact_root=artifact_root,
	)
	paired = _paired_rows(validation_rows)
	summary = _summary_payload(
		config_path=config_path,
		protocol_sha=protocol_sha,
		final_sha=final_sha,
		final_result=final_result,
		attempts=attempts,
		validation_rows=validation_rows,
		paired_rows=paired,
	)
	snapshots = _source_snapshots(
		settings=settings,
		final_result=final_result,
		config_path=config_path,
		survey_id=cast('str', canonical.dataset['name']),
	)
	_assert_snapshots_unchanged(snapshots)
	contents = {
		'attempts.csv': _csv_text(attempts, ATTEMPT_FIELDS),
		'validation_cells.csv': _csv_text(validation_rows, VALIDATION_FIELDS),
		'paired_deltas.csv': _csv_text(paired, PAIRED_FIELDS),
		'summary.json': json.dumps(summary, indent=2, sort_keys=True) + '\n',
		'summary.md': _summary_markdown(summary),
	}
	_assert_snapshots_unchanged(snapshots)
	_publish_outputs(output_dir, contents)
	return summary


def _read_hashed_json(path: Path, *, label: str) -> tuple[Mapping[str, object], str]:
	if path.is_symlink():
		raise ValueError(f'{label} must not be a symlink: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	raw = path.read_bytes()
	try:
		value = json.loads(raw)
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} is not valid JSON: {path}') from error
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return value, hashlib.sha256(raw).hexdigest()


def _display_path(path: Path, *, artifact_root: Path) -> str:
	resolved = path.resolve(strict=False)
	artifact = artifact_root.resolve(strict=False)
	repository = REPOSITORY_ROOT.resolve(strict=False)
	if resolved.is_relative_to(artifact):
		return f'artifacts/{resolved.relative_to(artifact).as_posix()}'
	if resolved.is_relative_to(repository):
		return resolved.relative_to(repository).as_posix()
	raise ValueError(f'report path is outside repository/artifact roots: {path}')


def _file_sha(path: Path, *, label: str) -> str:
	if path.is_symlink():
		raise ValueError(f'{label} must not be a symlink: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	return file_sha256(path)


def _list(value: object, label: str) -> list[object]:
	if not isinstance(value, list):
		raise TypeError(f'{label} must be a list')
	return value


def _row(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _type_sensitive_equal(actual: object, expected: object) -> bool:
	if isinstance(expected, Mapping):
		if not isinstance(actual, Mapping) or set(actual) != set(expected):
			return False
		return all(
			_type_sensitive_equal(actual[key], expected[key]) for key in expected
		)
	if isinstance(expected, list):
		return (
			isinstance(actual, list)
			and len(actual) == len(expected)
			and all(
				_type_sensitive_equal(actual_value, expected_value)
				for actual_value, expected_value in zip(actual, expected, strict=True)
			)
		)
	if isinstance(expected, tuple):
		return (
			isinstance(actual, tuple)
			and len(actual) == len(expected)
			and all(
				_type_sensitive_equal(actual_value, expected_value)
				for actual_value, expected_value in zip(actual, expected, strict=True)
			)
		)
	return type(actual) is type(expected) and actual == expected


def _bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be a bool')
	return value


def _cell_key(row: Mapping[str, object]) -> tuple[str, str, str]:
	values = (row.get('candidate_id'), row.get('layout_id'), row.get('data_size'))
	if not all(isinstance(value, str) for value in values):
		raise TypeError('validation cell identity must contain strings')
	source_id, layout_id, data_size = cast('tuple[str, str, str]', values)
	if source_id not in SOURCE_IDS:
		raise ValueError(f'unknown report source: {source_id!r}')
	if layout_id not in LAYOUT_IDS or data_size not in DATA_SIZES:
		raise ValueError('unsupported report layout or data size')
	return source_id, layout_id, data_size


def _macro_f1(row: Mapping[str, object]) -> float:
	value = row.get('macro_f1')
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError('validation macro_f1 must be numeric')
	result = float(value)
	if not math.isfinite(result) or not 0.0 <= result <= 1.0:
		raise ValueError('validation macro_f1 must be finite and within [0, 1]')
	return result


def _required_path(row: Mapping[str, object], key: str) -> Path:
	value = row.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{key} must be a non-empty path string')
	return Path(value)


def _required_sha(row: Mapping[str, object], key: str) -> str:
	value = row.get(key)
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{key} must be a lowercase SHA-256 digest')
	return value


def build_parser() -> argparse.ArgumentParser:
	"""Build the fixed report CLI parser."""
	parser = argparse.ArgumentParser(
		description='Build the immutable-evidence Z-filter validation report.'
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	return parser


def main() -> None:
	"""Build and identify the human-only Z-filter report."""
	args = build_parser().parse_args()
	summary = build_report(args.config)
	for key in (
		'passed',
		'winner_candidate_id',
		'authorizes_additional_view_followup',
		'failure_stage',
	):
		print(f'{key}: {summary[key]}')
	print(f'report_dir: {REPORT_OUTPUT_DIR}')


if __name__ == '__main__':
	main()
