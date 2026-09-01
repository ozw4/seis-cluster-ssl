# ruff: noqa: INP001
"""Build the five lightweight, validation-only base5 report projections.

The immutable protocol, inherited selection, and final result remain the
scientific inputs. Reports are reviewed by people and are never pipeline
inputs. Construction replays the final audit against live artifacts and will
not overwrite any existing report file or directory.
"""

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
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
RUNNER_PATH = EXPERIMENT_ROOT / 'run_validation.py'
RUNNER_SHA256 = '50634d8143a4e7d06b8c09f6be1d37857866bd60ca317c53083d16c9e7b8651e'
DEFAULT_CONFIG = EXPERIMENT_ROOT / '30_validation/01_candidates.yaml'
REPORT_OUTPUT_DIR = (
	REPOSITORY_ROOT / 'reports/f3/facies_benchmark_v2/'
	'local_barlow_twins_gaussian_view_v1/base5ep'
)
REPORT_FILENAMES = (
	'attempts.csv',
	'validation_cells.csv',
	'paired_deltas.csv',
	'summary.json',
	'summary.md',
)

SELECTED_ID = 'local_barlow_twins_gaussian_noise_std010_base5ep'
LEGACY_ID = 'local_barlow_twins_legacy_flip_base5ep'
RANDOM_ID = 'random'
SOURCE_IDS = (SELECTED_ID, LEGACY_ID)
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZES = ('small', 'medium', 'large')
BASE_EPOCHS = 5
CONTINUATION_EPOCHS = 25
VALIDATION_AGGREGATION_UNIT = 'unique_validation_voxel'

CONFIG_PATHS = {
	SELECTED_ID: {
		'base': EXPERIMENT_ROOT
		/ '10_stage1/gaussian_noise_std010_base5ep/01_screen_5ep.yaml',
		'continuation': EXPERIMENT_ROOT
		/ '15_stage2/gaussian_noise_std010_base5ep/01_continue_25ep.yaml',
		'extraction': EXPERIMENT_ROOT
		/ '20_embeddings/01_extract_gaussian_noise_std010_base5ep.yaml',
	},
	LEGACY_ID: {
		'base': EXPERIMENT_ROOT / '10_stage1/legacy_flip_base5ep/01_matched_5ep.yaml',
		'continuation': EXPERIMENT_ROOT
		/ '15_stage2/legacy_flip_base5ep/01_continue_25ep.yaml',
		'extraction': EXPERIMENT_ROOT
		/ '20_embeddings/02_extract_legacy_flip_base5ep.yaml',
	},
}

ATTEMPT_FIELDS = (
	'attempt_order',
	'candidate_id',
	'role',
	'parent_candidate_id',
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
	'medium_positive_count',
	'positive_vs_random_count',
	'wins_all_15_over_random',
	'final_outcome',
)

VALIDATION_FIELDS = (
	'source_id',
	'source_role',
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
	'left_value',
	'right_value',
	'delta',
	'strict_positive',
)


def _load_runner_namespace() -> dict[str, object]:
	if RUNNER_PATH.is_symlink() or not RUNNER_PATH.is_file():
		raise FileNotFoundError(f'missing base5 validation runner: {RUNNER_PATH}')
	if file_sha256(RUNNER_PATH) != RUNNER_SHA256:
		raise ValueError('base5 validation runner SHA-256 changed')
	namespace = runpy.run_path(str(RUNNER_PATH))
	required = {
		'validation_settings_from_mapping',
		'_canonical_config',
		'validate_base5_protocol_lock',
		'validate_base5_selection_lock',
		'create_base5_final_result',
		'_expected_augmentations',
	}
	missing = required - set(namespace)
	if missing:
		raise RuntimeError(f'base5 runner API is incomplete: {sorted(missing)!r}')
	return namespace


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


def _replay_final_result(
	*,
	runner: Mapping[str, object],
	settings: object,
	canonical: object,
	stored: Mapping[str, object],
) -> Mapping[str, object]:
	"""Capture a full final-result replay without writing scientific state."""
	created_at = stored.get('created_at_utc')
	if not isinstance(created_at, str):
		raise TypeError('final result created_at_utc must be a string')
	creator = cast('Any', runner['create_base5_final_result'])
	writer_globals = creator.__globals__
	original_writer = writer_globals.get('_write_exclusive_json')
	if not callable(original_writer):
		raise TypeError('base5 runner exclusive writer is unavailable')
	final_path = cast('Path', settings.final_result)
	sentinel = final_path.with_name('.base5-report-replay-sentinel.json')
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
	if len(captured) != 1 or dict(captured[0][1]) != dict(replayed):
		raise RuntimeError('final-result replay did not capture exactly one payload')
	if dict(stored) != dict(replayed):
		raise ValueError('immutable final result differs from replayed live evidence')
	return cast('Mapping[str, object]', replayed)


def _evidence_rows(
	final_result: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
	expected_values = _list(final_result.get('exact_expected_candidate_cells'), 'cells')
	candidate_values = _list(final_result.get('candidate_inputs'), 'candidate_inputs')
	random_values = _list(final_result.get('random_inputs'), 'random_inputs')
	expected = {_cell_key(_row(value, 'expected cell')) for value in expected_values}
	candidates = [_row(value, 'candidate input') for value in candidate_values]
	random = [_row(value, 'random input') for value in random_values]
	if len(expected) != len(expected_values):
		raise ValueError('expected candidate cells contain duplicates')
	actual = {_cell_key(row) for row in candidates}
	if len(actual) != len(candidates) or actual != expected:
		raise ValueError('candidate inputs do not match exact expected cells')
	gate = _mapping(final_result.get('medium_gate'), 'medium_gate')
	gate_open = _bool(gate.get('gate_open'), 'medium_gate.gate_open')
	expected_candidate_count = 30 if gate_open else 10
	expected_random_count = 15 if gate_open else 5
	if len(candidates) != expected_candidate_count:
		raise ValueError('candidate evidence count differs from reached branch')
	if len(random) != expected_random_count:
		raise ValueError('random evidence count differs from reached branch')
	random_cells = {_cell_key(row) for row in random}
	expected_random = {
		(RANDOM_ID, layout, size)
		for layout in LAYOUT_IDS
		for size in (DATA_SIZES if gate_open else ('medium',))
	}
	if len(random_cells) != len(random) or random_cells != expected_random:
		raise ValueError('random evidence does not match reached branch')
	for row in (*candidates, *random):
		_macro_f1(row)
	return candidates, random


def _collect_validation_rows(
	final_result: Mapping[str, object], *, artifact_root: Path
) -> list[dict[str, object]]:
	candidates, random = _evidence_rows(final_result)
	random_scores = {
		(cast('str', row['layout_id']), cast('str', row['data_size'])): _macro_f1(row)
		for row in random
	}
	rows: list[dict[str, object]] = []
	for evidence in (*candidates, *random):
		source_id, layout_id, data_size = _cell_key(evidence)
		value = _macro_f1(evidence)
		random_value = random_scores[layout_id, data_size]
		candidate = source_id != RANDOM_ID
		rows.append(
			{
				'source_id': source_id,
				'source_role': (
					'inherited_selected_view'
					if source_id == SELECTED_ID
					else (
						'matched_legacy_control'
						if source_id == LEGACY_ID
						else 'canonical_random'
					)
				),
				'base_pretraining_epochs': BASE_EPOCHS if candidate else '',
				'continuation_epochs': CONTINUATION_EPOCHS if candidate else '',
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
					_display_path(
						_required_path(evidence, 'candidate_audit_path'),
						artifact_root=artifact_root,
					)
					if candidate
					else ''
				),
				'candidate_audit_sha256': (
					_required_sha(evidence, 'candidate_audit_sha256')
					if candidate
					else ''
				),
				'base_checkpoint_sha256': (
					_required_sha(evidence, 'base_checkpoint_sha256')
					if candidate
					else ''
				),
				'continuation_init_checkpoint_sha256': (
					_required_sha(evidence, 'continuation_init_checkpoint_sha256')
					if candidate
					else ''
				),
				'final_checkpoint_sha256': (
					_required_sha(evidence, 'final_checkpoint_sha256')
					if candidate
					else _required_sha(evidence, 'checkpoint_sha256')
				),
			}
		)
	return sorted(
		rows,
		key=lambda row: (
			(*SOURCE_IDS, RANDOM_ID).index(cast('str', row['source_id'])),
			DATA_SIZES.index(cast('str', row['data_size'])),
			LAYOUT_IDS.index(cast('str', row['layout_id'])),
		),
	)


def _collect_attempt_rows(  # noqa: PLR0913
	*,
	runner: Mapping[str, object],
	settings: object,
	canonical: object,
	final_result: Mapping[str, object],
	validation_rows: Sequence[Mapping[str, object]],
	artifact_root: Path,
) -> list[dict[str, object]]:
	arm_results = _mapping(final_result.get('arm_results'), 'arm_results')
	winner = final_result.get('winner_candidate_id')
	rows: list[dict[str, object]] = []
	for index, source_id in enumerate(SOURCE_IDS, start=1):
		source = settings.source_by_id(source_id)
		configs = CONFIG_PATHS[source_id]
		for path in configs.values():
			_file_sha(path, label='attempt config')
		files = output_paths(source.embeddings_dir, canonical.dataset['name'])
		result = _mapping(arm_results.get(source_id), f'arm_results.{source_id}')
		source_rows = [row for row in validation_rows if row['source_id'] == source_id]
		medium = [row for row in source_rows if row['data_size'] == 'medium']
		if len(medium) != 5:
			raise ValueError(f'{source_id} report must contain five medium cells')
		positive_count = sum(float(row['delta_vs_random']) > 0.0 for row in source_rows)
		if positive_count != result.get('positive_delta_count'):
			raise ValueError(f'{source_id} result positive-count mismatch')
		expected_aug = cast('Any', runner['_expected_augmentations'])(source)
		rows.append(
			{
				'attempt_order': index,
				'candidate_id': source_id,
				'role': source.role,
				'parent_candidate_id': source.parent_candidate_id or '',
				'augmentations_json': json.dumps(
					expected_aug, separators=(',', ':'), sort_keys=True
				),
				'base_pretraining_epochs': BASE_EPOCHS,
				'continuation_epochs': CONTINUATION_EPOCHS,
				'base_config_path': _display_path(
					configs['base'], artifact_root=artifact_root
				),
				'base_config_sha256': file_sha256(configs['base']),
				'continuation_config_path': _display_path(
					configs['continuation'], artifact_root=artifact_root
				),
				'continuation_config_sha256': file_sha256(configs['continuation']),
				'extraction_config_path': _display_path(
					configs['extraction'], artifact_root=artifact_root
				),
				'extraction_config_sha256': file_sha256(configs['extraction']),
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
				'medium_positive_count': sum(
					float(row['delta_vs_random']) > 0.0 for row in medium
				),
				'positive_vs_random_count': positive_count,
				'wins_all_15_over_random': result['wins_all_15_over_random'],
				'final_outcome': (
					'winner'
					if winner == source_id
					else (
						'failed'
						if final_result.get('passed') is False
						else 'not_winner'
					)
				),
			}
		)
	return rows


def _paired_rows(
	validation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	index = {
		(
			cast('str', row['source_id']),
			cast('str', row['layout_id']),
			cast('str', row['data_size']),
		): float(row['macro_f1'])
		for row in validation_rows
	}
	rows: list[dict[str, object]] = []
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZES:
			for left_id, right_id, comparison_id in (
				(SELECTED_ID, RANDOM_ID, 'selected_gaussian_minus_random'),
				(LEGACY_ID, RANDOM_ID, 'matched_legacy_minus_random'),
				(SELECTED_ID, LEGACY_ID, 'selected_gaussian_minus_matched_legacy'),
			):
				left_key = (left_id, layout_id, data_size)
				right_key = (right_id, layout_id, data_size)
				if left_key not in index or right_key not in index:
					continue
				left = index[left_key]
				right = index[right_key]
				delta = left - right
				rows.append(
					{
						'base_pretraining_epochs': BASE_EPOCHS,
						'layout_id': layout_id,
						'data_size': data_size,
						'comparison_id': comparison_id,
						'left_source_id': left_id,
						'right_source_id': right_id,
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
	selection_sha: str,
	final_sha: str,
	final_result: Mapping[str, object],
	attempts: Sequence[Mapping[str, object]],
	validation_rows: Sequence[Mapping[str, object]],
	paired_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	gate = _mapping(final_result.get('medium_gate'), 'medium_gate')
	return {
		'schema_version': 1,
		'report_type': 'f3_local_barlow_twins_gaussian_base5_validation_report_v1',
		'validation_only': True,
		'pipeline_input': False,
		'base_pretraining_epochs': BASE_EPOCHS,
		'continuation_epochs': CONTINUATION_EPOCHS,
		'configuration': {
			'path': str(config_path.relative_to(REPOSITORY_ROOT)),
			'sha256': file_sha256(config_path),
		},
		'parent_result': final_result['parent_result'],
		'protocol_lock_sha256': protocol_sha,
		'selection_lock_sha256': selection_sha,
		'final_result_sha256': final_sha,
		'medium_gate': dict(gate),
		'attempt_count': len(attempts),
		'candidate_validation_cell_count': sum(
			row['source_id'] in SOURCE_IDS for row in validation_rows
		),
		'random_validation_cell_count': sum(
			row['source_id'] == RANDOM_ID for row in validation_rows
		),
		'paired_delta_count': len(paired_rows),
		'attempts': [dict(row) for row in attempts],
		'arm_results': final_result['arm_results'],
		'gaussian_attribution': final_result['gaussian_attribution'],
		'passed': final_result['passed'],
		'winner_candidate_id': final_result['winner_candidate_id'],
		'authorizes_next_base_duration': final_result['authorizes_next_base_duration'],
		'authorized_next_base_pretraining_epochs': final_result[
			'authorized_next_base_pretraining_epochs'
		],
		'failure_stage': final_result['failure_stage'],
		'source_note': (
			'Human-readable validation projection only; artifacts and immutable '
			'locks remain the scientific source of truth.'
		),
	}


def _summary_markdown(summary: Mapping[str, object]) -> str:
	status = 'PASS' if summary['passed'] else 'FAIL'
	winner = summary['winner_candidate_id'] or 'none'
	gate = _mapping(summary.get('medium_gate'), 'summary medium_gate')
	lines = [
		'# F3 Barlow Twins Gaussian-view base5 validation',
		'',
		(
			'This is a validation-only, human-readable projection. It is not a '
			'pipeline input.'
		),
		'',
		f'- Decision: **{status}**',
		f'- Winner: `{winner}`',
		f'- Medium gate open: `{gate.get("gate_open")}`',
		f'- Base pretraining epochs: `{BASE_EPOCHS}`',
		f'- Fixed continuation epochs: `{CONTINUATION_EPOCHS}`',
		f'- Candidate validation cells: `{summary["candidate_validation_cell_count"]}`',
		f'- Random baseline cells: `{summary["random_validation_cell_count"]}`',
		'',
		'## Attempts',
		'',
		'| Arm | Medium mean | Mean delta vs random | Positive cells | 15/15 |',
		'| --- | ---: | ---: | ---: | --- |',
	]
	lines.extend(
		f'| `{row["candidate_id"]}` | '
		f'{float(row["medium_macro_f1_mean"]):.9f} | '
		f'{float(row["medium_delta_vs_random_mean"]):+.9f} | '
		f'{row["positive_vs_random_count"]}/{row["evaluated_cell_count"]} | '
		f'{row["wins_all_15_over_random"]} |'
		for row in cast('Sequence[Mapping[str, object]]', summary['attempts'])
	)
	lines.extend(
		[
			'',
			(
				'Configuration selection was inherited from the failed, pinned '
				'25-epoch validation result; no base5 metric was used to choose '
				'the view.'
			),
			'',
		]
	)
	return '\n'.join(lines)


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
	paths: list[tuple[Path, str, str]] = [
		(config_path, file_sha256(config_path), 'validation config'),
		(RUNNER_PATH, file_sha256(RUNNER_PATH), 'validation runner'),
		(settings.protocol_lock, file_sha256(settings.protocol_lock), 'protocol lock'),
		(
			settings.selection_lock,
			file_sha256(settings.selection_lock),
			'selection lock',
		),
		(settings.final_result, file_sha256(settings.final_result), 'final result'),
	]
	for source_id in SOURCE_IDS:
		source = settings.source_by_id(source_id)
		for role, path in (
			('base checkpoint', source.base_checkpoint),
			('final checkpoint', source.final_checkpoint),
		):
			paths.append((path, _file_sha(path, label=role), f'{source_id} {role}'))
		for role, path in CONFIG_PATHS[source_id].items():
			paths.append(
				(
					path,
					_file_sha(path, label=f'{role} config'),
					f'{source_id} {role} config',
				)
			)
		files = output_paths(source.embeddings_dir, survey_id)
		for role, path in (
			('embeddings array', files.embeddings),
			('embedding metadata', files.metadata),
			('valid-token mask', files.valid_tokens),
		):
			paths.append((path, _file_sha(path, label=role), f'{source_id} {role}'))
	for label in ('candidate_inputs', 'random_inputs'):
		for value in _list(final_result.get(label), label):
			row = _row(value, label)
			metrics = _required_path(row, 'metrics_path')
			paths.append(
				(metrics, _required_sha(row, 'metrics_sha256'), f'{label} metrics')
			)
			if row.get('candidate_id') in SOURCE_IDS:
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
	"""Publish exactly five files without replacing any existing report path."""
	if set(contents) != set(REPORT_FILENAMES):
		raise ValueError('report publisher requires exactly the five fixed outputs')
	if output_dir.exists() or output_dir.is_symlink():
		raise FileExistsError(f'report output already exists: {output_dir}')
	output_dir.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(
		dir=output_dir.parent, prefix='.base5-report-staging-'
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
		raise ValueError('base5 report must use the fixed validation configuration')
	runner = _load_runner_namespace()
	settings = cast('Any', runner['validation_settings_from_mapping'])(
		load_config(config_path)
	)
	canonical = cast('Any', runner['_canonical_config'])(settings)
	protocol, protocol_sha = _read_hashed_json(
		settings.protocol_lock, label='base5 protocol lock'
	)
	selection, selection_sha = _read_hashed_json(
		settings.selection_lock, label='base5 selection lock'
	)
	final_result, final_sha = _read_hashed_json(
		settings.final_result, label='base5 final result'
	)
	protocol_replayed = cast('Any', runner['validate_base5_protocol_lock'])(
		settings, canonical
	)
	if dict(protocol) != dict(protocol_replayed):
		raise ValueError('stored protocol differs from live replay')
	selection_replayed = cast('Any', runner['validate_base5_selection_lock'])(
		settings, canonical
	)
	if dict(selection) != dict(selection_replayed):
		raise ValueError('stored selection differs from live replay')
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
		runner=runner,
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
		selection_sha=selection_sha,
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


def _bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be a bool')
	return value


def _cell_key(row: Mapping[str, object]) -> tuple[str, str, str]:
	values = (row.get('candidate_id'), row.get('layout_id'), row.get('data_size'))
	if not all(isinstance(value, str) for value in values):
		raise TypeError('validation cell identity must contain strings')
	source_id, layout_id, data_size = cast('tuple[str, str, str]', values)
	if source_id not in {*SOURCE_IDS, RANDOM_ID}:
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
		description='Build the immutable-evidence base5 validation report.'
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	return parser


def main() -> None:
	"""Build and identify the validation-only base5 report."""
	args = build_parser().parse_args()
	summary = build_report(args.config)
	for key in (
		'passed',
		'winner_candidate_id',
		'authorizes_next_base_duration',
		'failure_stage',
	):
		print(f'{key}: {summary[key]}')
	print(f'report_dir: {REPORT_OUTPUT_DIR}')


if __name__ == '__main__':
	main()
