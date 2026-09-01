# ruff: noqa: INP001, SLF001
"""Build the tracked, validation-only report for the Gaussian-view search.

The immutable protocol lock, selection lock, and final result remain the
scientific inputs. Files written by this module are human-reviewable
projections only and must never be consumed by a pipeline stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import runpy
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, cast

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology import five_way_results, five_way_runner
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)

EXPERIMENT_ID = 'local_barlow_twins_gaussian_view_v1'
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = EXPERIMENT_ROOT / 'run_validation.py'
DEFAULT_CONFIG = EXPERIMENT_ROOT / '30_validation' / '01_candidates.yaml'
REPORT_OUTPUT_DIR = (
	REPOSITORY_ROOT
	/ 'reports/f3/facies_benchmark_v2/local_barlow_twins_gaussian_view_v1'
)
REPORT_FILENAMES = (
	'attempts.csv',
	'validation_cells.csv',
	'paired_deltas.csv',
	'summary.json',
	'summary.md',
)
VIEW_DIAGNOSTIC_RELATIVE_PATH = Path(
	'diagnostics/f3/local_barlow_twins_gaussian_view_v1/view_diagnostic.json'
)
VIEW_DIAGNOSTIC_SHA256 = (
	'b4f89206bc49ae1fbf14cd62adccee396d6829adea6ff88243c0acf1f5b88d47'
)

FORCED_STD005_ID = 'local_barlow_twins_gaussian_noise_std005'
FORCED_STD010_ID = 'local_barlow_twins_gaussian_noise_std010'
IDENTITY_STD010_ID = 'local_barlow_twins_identity_gaussian_noise_std010'
LEGACY_CONTROL_ID = 'local_barlow_twins_legacy_flip_25ep'
RANDOM_ID = 'random'
CANDIDATE_ORDER = (
	FORCED_STD005_ID,
	FORCED_STD010_ID,
	IDENTITY_STD010_ID,
)
SOURCE_ORDER = (*CANDIDATE_ORDER, LEGACY_CONTROL_ID)
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZES = ('small', 'medium', 'large')
VALIDATION_AGGREGATION_UNIT = 'unique_validation_voxel'

ATTEMPT_CONFIG_PATHS = {
	FORCED_STD005_ID: {
		'base': EXPERIMENT_ROOT
		/ '10_stage1/gaussian_noise_std005/01_screen_25ep.yaml',
		'continuation': EXPERIMENT_ROOT
		/ '15_stage2/gaussian_noise_std005/01_continue_25ep.yaml',
		'extraction': EXPERIMENT_ROOT
		/ '20_embeddings/01_extract_gaussian_noise_std005.yaml',
	},
	FORCED_STD010_ID: {
		'base': EXPERIMENT_ROOT
		/ '10_stage1/gaussian_noise_std010/01_screen_25ep.yaml',
		'continuation': EXPERIMENT_ROOT
		/ '15_stage2/gaussian_noise_std010/01_continue_25ep.yaml',
		'extraction': EXPERIMENT_ROOT
		/ '20_embeddings/02_extract_gaussian_noise_std010.yaml',
	},
	IDENTITY_STD010_ID: {
		'base': EXPERIMENT_ROOT
		/ '10_stage1/identity_gaussian_noise_std010/01_screen_25ep.yaml',
		'continuation': EXPERIMENT_ROOT
		/ '15_stage2/identity_gaussian_noise_std010/01_continue_25ep.yaml',
		'extraction': EXPERIMENT_ROOT
		/ '20_embeddings/04_extract_identity_gaussian_noise_std010.yaml',
	},
	LEGACY_CONTROL_ID: {
		'base': EXPERIMENT_ROOT
		/ '10_stage1/legacy_flip_25ep/01_matched_25ep.yaml',
		'continuation': EXPERIMENT_ROOT
		/ '15_stage2/legacy_flip_25ep/01_continue_25ep.yaml',
		'extraction': EXPERIMENT_ROOT
		/ '20_embeddings/03_extract_legacy_flip_25ep.yaml',
	},
}

ATTEMPT_FIELDS = (
	'attempt_order',
	'candidate_id',
	'role',
	'selection_eligible',
	'augmentations_json',
	'base_pretraining_epochs',
	'base_global_step',
	'continuation_epochs',
	'continuation_global_step',
	'base_config_path',
	'base_config_sha256',
	'base_resolved_config_path',
	'base_resolved_config_sha256',
	'base_history_path',
	'base_history_sha256',
	'base_checkpoint_path',
	'base_checkpoint_sha256',
	'base_final_training_loss',
	'base_final_cross_correlation_diag_mean',
	'continuation_config_path',
	'continuation_config_sha256',
	'continuation_resolved_config_path',
	'continuation_resolved_config_sha256',
	'continuation_history_path',
	'continuation_history_sha256',
	'final_checkpoint_path',
	'final_checkpoint_sha256',
	'continuation_final_training_loss',
	'continuation_final_cross_correlation_diag_mean',
	'extraction_config_path',
	'extraction_config_sha256',
	'embeddings_dir',
	'embedding_metadata_path',
	'embedding_metadata_sha256',
	'embeddings_path',
	'embeddings_sha256',
	'valid_tokens_path',
	'valid_tokens_sha256',
	'evaluated_cell_count',
	'medium_macro_f1_mean',
	'medium_delta_vs_random_mean',
	'medium_positive_count',
	'positive_vs_random_count',
	'selection_outcome',
	'final_outcome',
	'decision_reason',
)

VALIDATION_CELL_FIELDS = (
	'source_id',
	'source_role',
	'selection_eligible',
	'base_pretraining_epochs',
	'continuation_epochs',
	'layout_id',
	'data_size',
	'subsample_seed',
	'decoder_seed',
	'evaluation_split',
	'aggregation_unit',
	'macro_f1',
	'metrics_path',
	'metrics_sha256',
	'candidate_audit_path',
	'candidate_audit_sha256',
	'base_checkpoint_sha256',
	'continuation_init_checkpoint_sha256',
	'final_checkpoint_sha256',
	'encoder_checkpoint_sha256',
	'embeddings_sha256',
	'embedding_metadata_sha256',
	'valid_tokens_sha256',
	'decoder_checkpoint_sha256',
	'decoder_initial_state_sha256',
	'selected_token_row_identity_sha256',
	'supervision_identity',
	'validation_identity',
	'validation_voxel_count',
)

PAIRED_DELTA_FIELDS = (
	'base_pretraining_epochs',
	'data_size',
	'layout_id',
	'comparison_id',
	'comparison_role',
	'metric',
	'left_source_id',
	'right_source_id',
	'left_value',
	'right_value',
	'delta',
	'strict_positive',
)


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
	if path.is_symlink():
		raise ValueError(f'{label} must not be a symlink: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} is not valid JSON: {path}') from error
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return value


def _read_hashed_json(
	path: Path, *, label: str
) -> tuple[Mapping[str, object], str]:
	"""Read one immutable JSON input and hash the exact bytes that were parsed."""
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


def _read_json_list(path: Path, *, label: str) -> list[object]:
	if path.is_symlink():
		raise ValueError(f'{label} must not be a symlink: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} is not valid JSON: {path}') from error
	if not isinstance(value, list):
		raise TypeError(f'{label} must contain a JSON list: {path}')
	return value


def _file_sha256(path: Path, *, label: str) -> str:
	if path.is_symlink():
		raise ValueError(f'{label} must not be a symlink: {path}')
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	return file_sha256(path)


def _load_runner_namespace() -> dict[str, object]:
	if not RUNNER_PATH.is_file():
		raise FileNotFoundError(f'missing validation runner: {RUNNER_PATH}')
	namespace = runpy.run_path(str(RUNNER_PATH))
	required = {
		'validation_settings_from_mapping',
		'_canonical_config',
		'validate_gaussian25_protocol_lock',
		'create_gaussian25_final_result',
		'_candidate_config',
		'_expected_augmentations',
	}
	missing = required - set(namespace)
	if missing:
		raise RuntimeError(f'validation runner API is incomplete: {sorted(missing)!r}')
	return namespace


def _replay_protocol_lock(
	*,
	runner: Mapping[str, object],
	settings: object,
	canonical: object,
	stored: Mapping[str, object],
) -> Mapping[str, object]:
	"""Recompute the protocol lock from the live repository and four bases."""
	validator = cast('Any', runner['validate_gaussian25_protocol_lock'])
	replayed = validator(settings, canonical)
	if not isinstance(replayed, Mapping):
		raise TypeError('protocol-lock validator must return a mapping')
	if dict(stored) != dict(replayed):
		raise ValueError('immutable protocol lock differs from replayed live evidence')
	return replayed


def _assert_source_snapshots_unchanged(
	snapshots: Sequence[tuple[Path, str, str]],
) -> None:
	for path, expected_sha256, label in snapshots:
		if _file_sha256(path, label=label) != expected_sha256:
			raise ValueError(f'{label} changed during report construction')


def _replay_final_result(
	*,
	runner: Mapping[str, object],
	settings: object,
	canonical: object,
	stored: Mapping[str, object],
) -> Mapping[str, object]:
	"""Replay the immutable publisher without allowing it to write a file."""
	created_at_utc = stored.get('created_at_utc')
	if not isinstance(created_at_utc, str):
		raise TypeError('final result created_at_utc must be a string')
	creator = cast('Any', runner['create_gaussian25_final_result'])
	writer_globals = creator.__globals__
	original_writer = writer_globals.get('_write_exclusive_json')
	if not callable(original_writer):
		raise TypeError('validation runner exclusive writer is unavailable')
	final_result_path = cast('Path', settings.final_result)
	sentinel = final_result_path.with_name('.gaussian25-report-replay-sentinel.json')
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
			created_at_utc=created_at_utc,
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


def _validated_evidence_rows(
	final_result: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
	expected_values = final_result.get('exact_expected_candidate_cells')
	candidate_values = final_result.get('candidate_inputs')
	random_values = final_result.get('random_inputs')
	for value, label in (
		(expected_values, 'exact_expected_candidate_cells'),
		(candidate_values, 'candidate_inputs'),
		(random_values, 'random_inputs'),
	):
		if not isinstance(value, list):
			raise TypeError(f'final result {label} must be a list')
	expected_rows = cast('list[object]', expected_values)
	candidate_rows = _mapping_rows(
		cast('list[object]', candidate_values), label='candidate_inputs'
	)
	random_rows = _mapping_rows(
		cast('list[object]', random_values), label='random_inputs'
	)
	expected_cells = {
		_cell_key(row, source_key='candidate_id', label='expected candidate cell')
		for row in _mapping_rows(expected_rows, label='exact_expected_candidate_cells')
	}
	if len(expected_cells) != len(expected_rows):
		raise ValueError('final result expected candidate cells contain duplicates')
	candidate_cells = {
		_cell_key(row, source_key='candidate_id', label='candidate input')
		for row in candidate_rows
	}
	if len(candidate_cells) != len(candidate_rows):
		raise ValueError('final result candidate inputs contain duplicates')
	if candidate_cells != expected_cells:
		raise ValueError(
			'final result candidate inputs do not match its exact cell set'
		)
	if {key[0] for key in candidate_cells} != set(SOURCE_ORDER):
		raise ValueError(
			'final result does not include every reached candidate/control'
		)
	gate = _mapping(final_result.get('medium_gate'), label='medium_gate')
	gate_open = _bool(gate.get('gate_open'), 'medium_gate.gate_open')
	reached_sizes = set(DATA_SIZES if gate_open else ('medium',))
	expected_random = {
		(RANDOM_ID, layout_id, data_size)
		for layout_id in LAYOUT_IDS
		for data_size in reached_sizes
	}
	random_cells = {
		_cell_key(row, source_key='candidate_id', label='random input')
		for row in random_rows
	}
	if len(random_cells) != len(random_rows):
		raise ValueError('final result random inputs contain duplicates')
	if random_cells != expected_random:
		raise ValueError('final result random inputs do not match the reached sizes')
	for row in (*candidate_rows, *random_rows):
		_macro_f1(row)
	return candidate_rows, random_rows


def _mapping_rows(
	values: Sequence[object], *, label: str
) -> list[Mapping[str, object]]:
	rows: list[Mapping[str, object]] = []
	for index, value in enumerate(values):
		if not isinstance(value, Mapping):
			raise TypeError(f'{label}[{index}] must be a mapping')
		rows.append(value)
	return rows


def _cell_key(
	row: Mapping[str, object], *, source_key: str, label: str
) -> tuple[str, str, str]:
	values = (row.get(source_key), row.get('layout_id'), row.get('data_size'))
	if not all(isinstance(value, str) for value in values):
		raise TypeError(f'{label} identity must contain strings')
	source_id, layout_id, data_size = cast('tuple[str, str, str]', values)
	if source_id not in {*SOURCE_ORDER, RANDOM_ID}:
		raise ValueError(f'{label} has unknown source: {source_id!r}')
	if layout_id not in LAYOUT_IDS or data_size not in DATA_SIZES:
		raise ValueError(f'{label} has unsupported layout/size')
	return source_id, layout_id, data_size


def _macro_f1(row: Mapping[str, object]) -> float:
	value = row.get('macro_f1')
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError('validation macro_f1 must be numeric')
	result = float(value)
	if not math.isfinite(result) or not 0.0 <= result <= 1.0:
		raise ValueError('validation macro_f1 must be finite and within [0, 1]')
	return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be a bool')
	return value


def _collect_validation_cells(  # noqa: C901, PLR0912, PLR0915
	*,
	runner: Mapping[str, object],
	settings: object,
	canonical: object,
	final_result: Mapping[str, object],
	artifact_root: Path,
) -> list[dict[str, object]]:
	candidate_evidence, random_evidence = _validated_evidence_rows(final_result)
	rows: list[dict[str, object]] = []
	hash_cache: dict[Path, str] = {}
	for evidence in (*candidate_evidence, *random_evidence):
		source_id, layout_id, data_size = _cell_key(
			evidence, source_key='candidate_id', label='validation evidence'
		)
		if source_id == RANDOM_ID:
			config = canonical
			model = canonical.model_by_id(RANDOM_ID)
			source = None
		else:
			source = settings.source_by_id(source_id)
			config = cast('Any', runner['_candidate_config'])(
				canonical, candidate=source, runs_root=settings.runs_root
			)
			model = config.model_by_id(source_id)
		job = five_way_runner.resolve_f3_lithology_five_way_job(
			config, model=source_id, layout=layout_id, size=data_size
		)
		inspection = five_way_runner.inspect_f3_lithology_five_way_job(job)
		condition = {
			'condition_dir': Path(cast('str', inspection['condition_dir'])),
			'validation_mask_sha256': inspection['validation_mask_sha256'],
			'validation_voxel_count': inspection['validation_voxel_count'],
		}
		identity = five_way_results._load_job_row(
			config,
			model_id=source_id,
			layout_id=layout_id,
			data_size=data_size,
			condition=condition,
		)
		if identity['macro_f1'] != _macro_f1(evidence):
			raise ValueError('report macro_f1 differs from the audited final result')
		metrics_sha256 = _required_sha(evidence, 'metrics_sha256')
		if _cached_sha(job.metrics_path, hash_cache, label='validation metrics') != (
			metrics_sha256
		):
			raise ValueError('validation metrics SHA-256 drifted during report build')
		files = output_paths(model.embeddings_dir, canonical.dataset['name'])
		for path, key in (
			(files.embeddings, 'embeddings_sha256'),
			(files.metadata, 'embedding_metadata_sha256'),
			(files.valid_tokens, 'valid_tokens_sha256'),
		):
			if _cached_sha(path, hash_cache, label=key) != identity[key]:
				raise ValueError(f'{source_id} {key} differs from live file content')
		decoder_checkpoint = job.decoder_dir / five_way_runner.BEST_CHECKPOINT_NAME
		if _cached_sha(
			decoder_checkpoint, hash_cache, label='decoder checkpoint'
		) != identity['decoder_checkpoint_sha256']:
			raise ValueError('decoder checkpoint SHA-256 drifted during report build')
		if source is None:
			base_sha = ''
			continuation_init_sha = ''
			final_sha = _required_sha(evidence, 'checkpoint_sha256')
			if final_sha != identity['encoder_checkpoint_sha256']:
				raise ValueError('random checkpoint SHA-256 identity drifted')
			audit_path = ''
			audit_sha = ''
			base_epochs: int | str = ''
			continuation_epochs: int | str = ''
			selection_eligible: bool | str = ''
			role = 'canonical_random'
		else:
			base_sha = _required_sha(evidence, 'base_checkpoint_sha256')
			continuation_init_sha = _required_sha(
				evidence, 'continuation_init_checkpoint_sha256'
			)
			final_sha = _required_sha(evidence, 'final_checkpoint_sha256')
			if continuation_init_sha != base_sha:
				raise ValueError('continuation initialization does not match base SHA')
			if final_sha != identity['encoder_checkpoint_sha256']:
				raise ValueError('candidate final and encoder SHA-256 differ')
			audit_path_value = _required_path(evidence, 'candidate_audit_path')
			audit_sha = _required_sha(evidence, 'candidate_audit_sha256')
			if _cached_sha(
				audit_path_value, hash_cache, label='candidate source audit'
			) != audit_sha:
				raise ValueError('candidate source audit SHA-256 drifted')
			audit_path = _display_path(
				audit_path_value, artifact_root=artifact_root
			)
			base_epochs = source.base_pretraining_epochs
			continuation_epochs = source.continuation_epochs
			selection_eligible = source.selectable
			role = 'selectable_candidate' if source.selectable else 'legacy_control'
		rows.append(
			{
				'source_id': source_id,
				'source_role': role,
				'selection_eligible': selection_eligible,
				'base_pretraining_epochs': base_epochs,
				'continuation_epochs': continuation_epochs,
				'layout_id': layout_id,
				'data_size': data_size,
				'subsample_seed': inspection['subsample_seed'],
				'decoder_seed': inspection['decoder_seed'],
				'evaluation_split': 'validation',
				'aggregation_unit': VALIDATION_AGGREGATION_UNIT,
				'macro_f1': _macro_f1(evidence),
				'metrics_path': _display_path(
					job.metrics_path, artifact_root=artifact_root
				),
				'metrics_sha256': metrics_sha256,
				'candidate_audit_path': audit_path,
				'candidate_audit_sha256': audit_sha,
				'base_checkpoint_sha256': base_sha,
				'continuation_init_checkpoint_sha256': continuation_init_sha,
				'final_checkpoint_sha256': final_sha,
				'encoder_checkpoint_sha256': identity[
					'encoder_checkpoint_sha256'
				],
				'embeddings_sha256': identity['embeddings_sha256'],
				'embedding_metadata_sha256': identity[
					'embedding_metadata_sha256'
				],
				'valid_tokens_sha256': identity['valid_tokens_sha256'],
				'decoder_checkpoint_sha256': identity[
					'decoder_checkpoint_sha256'
				],
				'decoder_initial_state_sha256': inspection[
					'decoder_initial_state_sha256'
				],
				'selected_token_row_identity_sha256': inspection[
					'selected_token_row_identity_sha256'
				],
				'supervision_identity': identity['supervision_identity'],
				'validation_identity': identity['validation_identity'],
				'validation_voxel_count': identity['validation_voxel_count'],
			}
		)
	_validate_validation_cell_rows(rows)
	return sorted(
		rows,
		key=lambda row: (
			SOURCE_ORDER.index(cast('str', row['source_id']))
			if row['source_id'] in SOURCE_ORDER
			else len(SOURCE_ORDER),
			DATA_SIZES.index(cast('str', row['data_size'])),
			LAYOUT_IDS.index(cast('str', row['layout_id'])),
		),
	)


def _cached_sha(path: Path, cache: dict[Path, str], *, label: str) -> str:
	resolved = path.resolve(strict=False)
	if resolved not in cache:
		cache[resolved] = _file_sha256(path, label=label)
	return cache[resolved]


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


def _validate_validation_cell_rows(rows: Sequence[Mapping[str, object]]) -> None:
	keys = [
		(row.get('source_id'), row.get('layout_id'), row.get('data_size'))
		for row in rows
	]
	if len(keys) != len(set(keys)):
		raise ValueError('report validation cells contain duplicates')
	by_condition: dict[tuple[object, object], list[Mapping[str, object]]] = {}
	for row in rows:
		by_condition.setdefault(
			(row.get('layout_id'), row.get('data_size')), []
		).append(row)
	for cell, group in by_condition.items():
		for field in (
			'subsample_seed',
			'decoder_seed',
			'decoder_initial_state_sha256',
			'selected_token_row_identity_sha256',
			'supervision_identity',
			'validation_identity',
			'validation_voxel_count',
			'valid_tokens_sha256',
		):
			if len({row[field] for row in group}) != 1:
				raise ValueError(
					f'validation condition {cell!r} differs across sources for {field}'
				)


def _paired_delta_rows(  # noqa: C901
	validation_rows: Sequence[Mapping[str, object]],
	*,
	selected_id: str,
	base_pretraining_epochs: int,
) -> list[dict[str, object]]:
	index: dict[tuple[str, str, str], float] = {}
	for row in validation_rows:
		key = (
			cast('str', row['source_id']),
			cast('str', row['layout_id']),
			cast('str', row['data_size']),
		)
		if key in index:
			raise ValueError('validation rows contain duplicate paired identities')
		index[key] = float(row['macro_f1'])
	result: list[dict[str, object]] = []
	for source_id in SOURCE_ORDER:
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				if (source_id, layout_id, data_size) not in index:
					continue
				_append_pair(
					result,
					index=index,
					left_id=source_id,
					right_id=RANDOM_ID,
					layout_id=layout_id,
					data_size=data_size,
					comparison_id=f'{source_id}_minus_random',
					comparison_role=(
						'legacy_minus_random'
						if source_id == LEGACY_CONTROL_ID
						else 'candidate_minus_random'
					),
					base_pretraining_epochs=base_pretraining_epochs,
				)
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZES:
			if all(
				(source_id, layout_id, data_size) in index
				for source_id in (selected_id, LEGACY_CONTROL_ID)
			):
				_append_pair(
					result,
					index=index,
					left_id=selected_id,
					right_id=LEGACY_CONTROL_ID,
					layout_id=layout_id,
					data_size=data_size,
					comparison_id='selected_gaussian_minus_matched_legacy',
					comparison_role='selected_vs_legacy',
					base_pretraining_epochs=base_pretraining_epochs,
				)
			if all(
				(source_id, layout_id, data_size) in index
				for source_id in (IDENTITY_STD010_ID, FORCED_STD010_ID)
			):
				_append_pair(
					result,
					index=index,
					left_id=IDENTITY_STD010_ID,
					right_id=FORCED_STD010_ID,
					layout_id=layout_id,
					data_size=data_size,
					comparison_id='identity_std010_minus_forced_flip_std010',
					comparison_role='identity_vs_forced_geometry',
					base_pretraining_epochs=base_pretraining_epochs,
				)
	keys = [
		(row['comparison_id'], row['layout_id'], row['data_size']) for row in result
	]
	if len(keys) != len(set(keys)):
		raise ValueError('paired report contains duplicate contrast cells')
	return result


def _append_pair(  # noqa: PLR0913
	rows: list[dict[str, object]],
	*,
	index: Mapping[tuple[str, str, str], float],
	left_id: str,
	right_id: str,
	layout_id: str,
	data_size: str,
	comparison_id: str,
	comparison_role: str,
	base_pretraining_epochs: int,
) -> None:
	left = index[left_id, layout_id, data_size]
	right = index[right_id, layout_id, data_size]
	delta = left - right
	rows.append(
		{
			'base_pretraining_epochs': base_pretraining_epochs,
			'data_size': data_size,
			'layout_id': layout_id,
			'comparison_id': comparison_id,
			'comparison_role': comparison_role,
			'metric': 'macro_f1',
			'left_source_id': left_id,
			'right_source_id': right_id,
			'left_value': left,
			'right_value': right,
			'delta': delta,
			'strict_positive': delta > 0.0,
		}
	)


def _history_completion(
	path: Path, *, epochs: int, expected_global_step: int, label: str
) -> tuple[int, float, float, str]:
	values = _read_json_list(path, label=label)
	rows = _mapping_rows(values, label=label)
	if len(rows) != epochs or [row.get('epoch') for row in rows] != list(
		range(1, epochs + 1)
	):
		raise ValueError(f'{label} does not contain exact contiguous epoch history')
	last = rows[-1]
	global_step = last.get('global_step')
	if global_step != expected_global_step:
		raise ValueError(f'{label} final global step is not {expected_global_step}')
	metrics: list[float] = []
	for key in ('training_loss', 'cross_correlation_diag_mean'):
		value = last.get(key)
		if not isinstance(value, int | float) or isinstance(value, bool):
			raise TypeError(f'{label} final {key} must be numeric')
		result = float(value)
		if not math.isfinite(result):
			raise ValueError(f'{label} final {key} must be finite')
		metrics.append(result)
	return cast('int', global_step), metrics[0], metrics[1], file_sha256(path)


def _validate_training_artifact_chain(  # noqa: PLR0913
	*,
	config_path: Path,
	resolved_config: Mapping[str, object],
	checkpoint_path: Path,
	final_epoch: int,
	final_global_step: int,
	final_training_loss: float,
	final_cross_correlation_diag_mean: float,
	label: str,
) -> None:
	expected_resolved = resolve_barlow_twins_training_config(
		load_config(config_path)
	)
	if dict(resolved_config) != dict(expected_resolved):
		raise ValueError(f'{label} YAML does not resolve to resolved_config.json')
	payload = load_checkpoint_metadata_without_weights(checkpoint_path)
	payload_config = _mapping(payload.get('config'), label=f'{label} checkpoint config')
	if dict(payload_config) != dict(resolved_config):
		raise ValueError(f'{label} checkpoint config differs from resolved_config.json')
	if payload.get('epoch') != final_epoch:
		raise ValueError(f'{label} checkpoint epoch differs from history')
	if payload.get('global_step') != final_global_step:
		raise ValueError(f'{label} checkpoint global_step differs from history')
	metrics = _mapping(payload.get('metrics'), label=f'{label} checkpoint metrics')
	for key, expected in (
		('training_loss', final_training_loss),
		('cross_correlation_diag_mean', final_cross_correlation_diag_mean),
	):
		value = metrics.get(key)
		if (
			isinstance(value, bool)
			or not isinstance(value, int | float)
			or not math.isfinite(float(value))
			or float(value) != expected
		):
			raise ValueError(f'{label} checkpoint {key} differs from history')


def _validate_extraction_artifact_chain(  # noqa: PLR0913
	*,
	config_path: Path,
	checkpoint_path: Path,
	embeddings_dir: Path,
	metadata: Mapping[str, object],
	checkpoint_sha256: str,
	label: str,
) -> None:
	resolved = resolve_embedding_extraction_config(load_config(config_path))
	embeddings = _mapping(
		resolved.get('embeddings'), label=f'{label} extraction embeddings'
	)
	if embeddings.get('checkpoint') != str(checkpoint_path):
		raise ValueError(f'{label} extraction YAML points to a different checkpoint')
	if embeddings.get('output_dir') != str(embeddings_dir):
		raise ValueError(f'{label} extraction YAML points to a different output_dir')
	if metadata.get('checkpoint_path') != str(checkpoint_path):
		raise ValueError(f'{label} embedding metadata records a different checkpoint')
	if metadata.get('checkpoint_sha256') != checkpoint_sha256:
		raise ValueError(f'{label} embedding metadata checkpoint SHA-256 drifted')
	embedding = _mapping(
		resolved.get('embedding'), label=f'{label} extraction embedding settings'
	)
	for config_key, metadata_key in (
		('window_size', 'window_size'),
		('overlap', 'overlap'),
		('output_dtype', 'output_dtype'),
		('min_token_valid_fraction', 'min_token_valid_fraction'),
	):
		if embedding.get(config_key) != metadata.get(metadata_key):
			raise ValueError(
				f'{label} extraction {config_key} differs from embedding metadata'
			)
	precision = _mapping(metadata.get('precision'), label=f'{label} precision')
	if precision.get('amp_requested') != embedding.get('amp'):
		raise ValueError(f'{label} extraction amp differs from embedding metadata')
	if precision.get('amp_dtype_requested') != embedding.get('amp_dtype'):
		raise ValueError(
			f'{label} extraction amp_dtype differs from embedding metadata'
		)
	cache_config = _mapping(
		embedding.get('preprocessing_cache'),
		label=f'{label} extraction preprocessing_cache',
	)
	cache_metadata = _mapping(
		metadata.get('preprocessing_cache'),
		label=f'{label} metadata preprocessing_cache',
	)
	if cache_metadata.get('requested_mode') != cache_config.get('mode'):
		raise ValueError(
			f'{label} extraction cache mode differs from embedding metadata'
		)


def _collect_attempt_rows(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	runner: Mapping[str, object],
	settings: object,
	canonical: object,
	final_result: Mapping[str, object],
	validation_rows: Sequence[Mapping[str, object]],
	artifact_root: Path,
) -> list[dict[str, object]]:
	selected_id = cast(
		'str', _mapping(final_result.get('selection_lock'), label='selection_lock')[
			'selected_candidate_id'
		]
	)
	winner_id = final_result.get('winner_candidate_id')
	if winner_id is not None and winner_id not in SOURCE_ORDER:
		raise ValueError('final result winner_candidate_id is unsupported')
	base_epochs = cast('int', final_result['base_pretraining_epochs'])
	continuation_epochs = cast('int', final_result['continuation_epochs'])
	medium_gate = _mapping(final_result.get('medium_gate'), label='medium_gate')
	medium_gate_open = _bool(medium_gate.get('gate_open'), 'medium_gate.gate_open')
	base_steps = base_epochs * 625
	continuation_steps = continuation_epochs * 625
	by_source: dict[str, list[Mapping[str, object]]] = {
		source_id: [row for row in validation_rows if row['source_id'] == source_id]
		for source_id in SOURCE_ORDER
	}
	random_by_cell = {
		(cast('str', row['layout_id']), cast('str', row['data_size'])): float(
			row['macro_f1']
		)
		for row in validation_rows
		if row['source_id'] == RANDOM_ID
	}
	rows: list[dict[str, object]] = []
	for attempt_order, source_id in enumerate(SOURCE_ORDER, start=1):
		source = settings.source_by_id(source_id)
		paths = ATTEMPT_CONFIG_PATHS[source_id]
		base_resolved_path = source.base_checkpoint.parent / 'resolved_config.json'
		base_history_path = source.base_checkpoint.parent / 'history.json'
		final_resolved_path = source.final_checkpoint.parent / 'resolved_config.json'
		final_history_path = source.final_checkpoint.parent / 'history.json'
		base_resolved = _read_json(
			base_resolved_path, label=f'{source_id} base resolved config'
		)
		final_resolved = _read_json(
			final_resolved_path, label=f'{source_id} continuation resolved config'
		)
		expected_augmentations = cast('Any', runner['_expected_augmentations'])(source)
		if base_resolved.get('augmentations') != expected_augmentations:
			raise ValueError(f'{source_id} base augmentation mapping drifted')
		if final_resolved.get('augmentations') != expected_augmentations:
			raise ValueError(f'{source_id} continuation augmentation mapping drifted')
		base_step, base_loss, base_diag, base_history_sha = _history_completion(
			base_history_path,
			epochs=base_epochs,
			expected_global_step=base_steps,
			label=f'{source_id} base history',
		)
		final_step, final_loss, final_diag, final_history_sha = _history_completion(
			final_history_path,
			epochs=continuation_epochs,
			expected_global_step=continuation_steps,
			label=f'{source_id} continuation history',
		)
		_validate_training_artifact_chain(
			config_path=paths['base'],
			resolved_config=base_resolved,
			checkpoint_path=source.base_checkpoint,
			final_epoch=base_epochs,
			final_global_step=base_step,
			final_training_loss=base_loss,
			final_cross_correlation_diag_mean=base_diag,
			label=f'{source_id} base',
		)
		_validate_training_artifact_chain(
			config_path=paths['continuation'],
			resolved_config=final_resolved,
			checkpoint_path=source.final_checkpoint,
			final_epoch=continuation_epochs,
			final_global_step=final_step,
			final_training_loss=final_loss,
			final_cross_correlation_diag_mean=final_diag,
			label=f'{source_id} continuation',
		)
		source_rows = by_source[source_id]
		if not source_rows:
			raise ValueError(f'{source_id} has no reached validation result')
		base_shas = {cast('str', row['base_checkpoint_sha256']) for row in source_rows}
		final_shas = {
			cast('str', row['final_checkpoint_sha256']) for row in source_rows
		}
		if len(base_shas) != 1 or len(final_shas) != 1:
			raise ValueError(f'{source_id} checkpoint identity varies between cells')
		base_checkpoint_sha = _file_sha256(
			source.base_checkpoint, label=f'{source_id} base checkpoint'
		)
		final_checkpoint_sha = _file_sha256(
			source.final_checkpoint, label=f'{source_id} final checkpoint'
		)
		if base_shas != {base_checkpoint_sha} or final_shas != {final_checkpoint_sha}:
			raise ValueError(f'{source_id} live checkpoint SHA-256 drifted')
		files = output_paths(source.embeddings_dir, canonical.dataset['name'])
		embeddings_sha = _file_sha256(
			files.embeddings, label=f'{source_id} embeddings'
		)
		metadata_sha = _file_sha256(
			files.metadata, label=f'{source_id} embedding metadata'
		)
		metadata = _read_json(
			files.metadata, label=f'{source_id} embedding metadata'
		)
		valid_tokens_sha = _file_sha256(
			files.valid_tokens, label=f'{source_id} valid tokens'
		)
		_validate_extraction_artifact_chain(
			config_path=paths['extraction'],
			checkpoint_path=source.final_checkpoint,
			embeddings_dir=source.embeddings_dir,
			metadata=metadata,
			checkpoint_sha256=final_checkpoint_sha,
			label=source_id,
		)
		for key, actual in (
			('embeddings_sha256', embeddings_sha),
			('embedding_metadata_sha256', metadata_sha),
			('valid_tokens_sha256', valid_tokens_sha),
		):
			if {row[key] for row in source_rows} != {actual}:
				raise ValueError(f'{source_id} {key} varies or drifted')
		medium_rows = [row for row in source_rows if row['data_size'] == 'medium']
		if len(medium_rows) != len(LAYOUT_IDS):
			raise ValueError(f'{source_id} must contain five medium validation cells')
		medium_scores = [float(row['macro_f1']) for row in medium_rows]
		medium_deltas = [
			float(row['macro_f1'])
			- random_by_cell[cast('str', row['layout_id']), 'medium']
			for row in medium_rows
		]
		all_deltas = [
			float(row['macro_f1'])
			- random_by_cell[
				cast('str', row['layout_id']), cast('str', row['data_size'])
			]
			for row in source_rows
		]
		selection_outcome = (
			'nonselectable_control'
			if source_id == LEGACY_CONTROL_ID
			else ('selected' if source_id == selected_id else 'not_selected')
		)
		if source_id == winner_id:
			final_outcome = 'winner'
			decision_reason = 'first preregistered arm satisfying strict 15/15'
		elif (
			not medium_gate_open
			and source_id in {selected_id, LEGACY_CONTROL_ID}
		):
			final_outcome = 'failed_medium_gate'
			decision_reason = (
				'completed the reached branch but did not beat random in all 5/5 '
				'medium layouts'
			)
		elif len(source_rows) < 15:
			final_outcome = 'not_fully_evaluated'
			decision_reason = 'medium screen only under the preregistered branch rule'
		elif source_id not in {selected_id, LEGACY_CONTROL_ID}:
			final_outcome = 'geometry_control'
			decision_reason = (
				'required same-strength geometry control; not eligible for final choice'
			)
		elif all(delta > 0.0 for delta in all_deltas):
			final_outcome = 'passed_not_chosen'
			decision_reason = 'passed random but lost the fixed winner rule'
		else:
			final_outcome = 'failed_over_random'
			decision_reason = (
				'did not satisfy strict positive delta in every reached cell'
			)
		rows.append(
			{
				'attempt_order': attempt_order,
				'candidate_id': source_id,
				'role': (
					'selectable_view' if source.selectable else 'matched_legacy_control'
				),
				'selection_eligible': source.selectable,
				'augmentations_json': json.dumps(
					expected_augmentations,
					sort_keys=True,
					separators=(',', ':'),
				),
				'base_pretraining_epochs': base_epochs,
				'base_global_step': base_step,
				'continuation_epochs': continuation_epochs,
				'continuation_global_step': final_step,
				'base_config_path': _display_path(
					paths['base'], artifact_root=artifact_root
				),
				'base_config_sha256': _file_sha256(
					paths['base'], label=f'{source_id} base config'
				),
				'base_resolved_config_path': _display_path(
					base_resolved_path, artifact_root=artifact_root
				),
				'base_resolved_config_sha256': file_sha256(base_resolved_path),
				'base_history_path': _display_path(
					base_history_path, artifact_root=artifact_root
				),
				'base_history_sha256': base_history_sha,
				'base_checkpoint_path': _display_path(
					source.base_checkpoint, artifact_root=artifact_root
				),
				'base_checkpoint_sha256': base_checkpoint_sha,
				'base_final_training_loss': base_loss,
				'base_final_cross_correlation_diag_mean': base_diag,
				'continuation_config_path': _display_path(
					paths['continuation'], artifact_root=artifact_root
				),
				'continuation_config_sha256': _file_sha256(
					paths['continuation'], label=f'{source_id} continuation config'
				),
				'continuation_resolved_config_path': _display_path(
					final_resolved_path, artifact_root=artifact_root
				),
				'continuation_resolved_config_sha256': file_sha256(
					final_resolved_path
				),
				'continuation_history_path': _display_path(
					final_history_path, artifact_root=artifact_root
				),
				'continuation_history_sha256': final_history_sha,
				'final_checkpoint_path': _display_path(
					source.final_checkpoint, artifact_root=artifact_root
				),
				'final_checkpoint_sha256': final_checkpoint_sha,
				'continuation_final_training_loss': final_loss,
				'continuation_final_cross_correlation_diag_mean': final_diag,
				'extraction_config_path': _display_path(
					paths['extraction'], artifact_root=artifact_root
				),
				'extraction_config_sha256': _file_sha256(
					paths['extraction'], label=f'{source_id} extraction config'
				),
				'embeddings_dir': _display_path(
					source.embeddings_dir, artifact_root=artifact_root
				),
				'embedding_metadata_path': _display_path(
					files.metadata, artifact_root=artifact_root
				),
				'embedding_metadata_sha256': metadata_sha,
				'embeddings_path': _display_path(
					files.embeddings, artifact_root=artifact_root
				),
				'embeddings_sha256': embeddings_sha,
				'valid_tokens_path': _display_path(
					files.valid_tokens, artifact_root=artifact_root
				),
				'valid_tokens_sha256': valid_tokens_sha,
				'evaluated_cell_count': len(source_rows),
				'medium_macro_f1_mean': fmean(medium_scores),
				'medium_delta_vs_random_mean': fmean(medium_deltas),
				'medium_positive_count': sum(delta > 0.0 for delta in medium_deltas),
				'positive_vs_random_count': sum(
					delta > 0.0 for delta in all_deltas
				),
				'selection_outcome': selection_outcome,
				'final_outcome': final_outcome,
				'decision_reason': decision_reason,
			}
		)
	return rows


def _display_path(path: Path, *, artifact_root: Path) -> str:
	resolved = path.resolve(strict=False)
	resolved_artifact_root = artifact_root.resolve(strict=False)
	try:
		relative = resolved.relative_to(resolved_artifact_root)
	except ValueError:
		pass
	else:
		return '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/' + relative.as_posix()
	try:
		return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
	except ValueError as error:
		raise ValueError(
			f'refusing to report machine-specific path outside repository/artifacts: '
			f'{path}'
		) from error


def _aggregate(values: Sequence[float]) -> dict[str, object]:
	if not values:
		raise ValueError('cannot aggregate an empty value sequence')
	return {
		'n': len(values),
		'mean': fmean(values),
		'sample_std': stdev(values) if len(values) > 1 else 0.0,
		'median': median(values),
		'min': min(values),
		'max': max(values),
		'positive_count': sum(value > 0.0 for value in values),
		'zero_count': sum(value == 0.0 for value in values),
		'negative_count': sum(value < 0.0 for value in values),
	}


def _view_diagnostic_summary(
	artifact_root: Path,
) -> dict[str, object]:
	path = artifact_root / VIEW_DIAGNOSTIC_RELATIVE_PATH
	digest = _file_sha256(path, label='actual-data view diagnostic')
	if digest != VIEW_DIAGNOSTIC_SHA256:
		raise ValueError('actual-data view diagnostic SHA-256 drifted')
	payload = _read_json(path, label='actual-data view diagnostic')
	if payload.get('schema_version') != 1 or payload.get('diagnostic') != (
		'f3_local_barlow_twins_aligned_views'
	):
		raise ValueError('actual-data view diagnostic identity is invalid')
	metrics = _mapping(payload.get('metrics'), label='view diagnostic metrics')
	summaries: dict[str, object] = {}
	for view_id in ('legacy', 'gaussian_noise_std005', 'gaussian_noise_std010'):
		view = _mapping(metrics.get(view_id), label=f'view diagnostic {view_id}')
		valid_voxels = _mapping(
			view.get('all_valid_physical_voxels'),
			label=f'view diagnostic {view_id} all-valid metrics',
		)
		row: dict[str, object] = {}
		for key in (
			'paired_correlation',
			'paired_rms',
			'per_view_rms_from_unaugmented',
		):
			value = valid_voxels.get(key)
			if not isinstance(value, int | float) or isinstance(value, bool):
				raise TypeError(f'view diagnostic {view_id} {key} must be numeric')
			resolved = float(value)
			if not math.isfinite(resolved):
				raise ValueError(f'view diagnostic {view_id} {key} must be finite')
			row[key] = resolved
		voxel_count = valid_voxels.get('voxel_count')
		if (
			not isinstance(voxel_count, int)
			or isinstance(voxel_count, bool)
			or voxel_count <= 0
		):
			raise ValueError(
				f'view diagnostic {view_id} voxel_count must be positive'
			)
		row['voxel_count'] = voxel_count
		summaries[view_id] = row
	return {
		'path': _display_path(path, artifact_root=artifact_root),
		'sha256': digest,
		'metric_scope': 'all_valid_physical_voxels_after_inverse_flip_alignment',
		'views': summaries,
		'rationale': (
			'legacy inverse-aligned views are amplitude-identical, while the '
			'Gaussian policies add controlled view-specific amplitude perturbations'
		),
		'caveat': (
			'the preserved diagnostic does not cover the identity policy or rule '
			'out its fixed-position shortcut'
		),
	}


def _benchmark_provenance_summary(
	*,
	settings: object,
	final_result: Mapping[str, object],
	artifact_root: Path,
) -> dict[str, object]:
	live = _mapping(
		final_result.get('benchmark_provenance'), label='benchmark_provenance'
	)
	result: dict[str, object] = {}
	for stem in (
		'random_checkpoint',
		'canonical_comparison',
		'pretraining_manifest',
		'pretraining_path_list',
	):
		path = _required_path(live, stem)
		digest = _required_sha(live, f'{stem}_sha256')
		if _file_sha256(path, label=stem.replace('_', ' ')) != digest:
			raise ValueError(f'{stem} SHA-256 drifted during report build')
		result[stem] = {
			'path': _display_path(path, artifact_root=artifact_root),
			'sha256': digest,
		}
	for stem, path_attribute, sha_attribute in (
		(
			'reference_base_checkpoint',
			'reference_base_checkpoint',
			'reference_base_checkpoint_sha256',
		),
		(
			'reference_final_checkpoint',
			'reference_final_checkpoint',
			'reference_final_checkpoint_sha256',
		),
	):
		path = cast('Path', getattr(settings, path_attribute))
		digest = cast('str', getattr(settings, sha_attribute))
		if _file_sha256(path, label=stem.replace('_', ' ')) != digest:
			raise ValueError(f'{stem} SHA-256 drifted during report build')
		result[stem] = {
			'path': _display_path(path, artifact_root=artifact_root),
			'sha256': digest,
		}
	return result


def _summary_payload(  # noqa: PLR0913
	*,
	config_path: Path,
	validation_config_sha256: str,
	validation_runner_sha256: str,
	canonical_config_sha256: str,
	artifact_root: Path,
	settings: object,
	protocol_lock_sha256: str,
	selection_lock: Mapping[str, object],
	selection_lock_sha256: str,
	final_result: Mapping[str, object],
	final_result_sha256: str,
	validation_rows: Sequence[Mapping[str, object]],
	paired_rows: Sequence[Mapping[str, object]],
	attempt_rows: Sequence[Mapping[str, object]],
	view_diagnostic: Mapping[str, object],
) -> dict[str, object]:
	selected_id = cast(
		'str', _mapping(final_result['selection_lock'], label='selection_lock')[
			'selected_candidate_id'
		]
	)
	by_size: dict[str, dict[str, object]] = {}
	for source_id in (*SOURCE_ORDER, RANDOM_ID):
		source_sizes: dict[str, object] = {}
		for data_size in DATA_SIZES:
			values = [
				float(row['macro_f1'])
				for row in validation_rows
				if row['source_id'] == source_id and row['data_size'] == data_size
			]
			if values:
				source_sizes[data_size] = _aggregate(values)
		if source_sizes:
			by_size[source_id] = source_sizes
	comparison_aggregates: dict[str, dict[str, object]] = {}
	for comparison_id in sorted({str(row['comparison_id']) for row in paired_rows}):
		values = [
			float(row['delta'])
			for row in paired_rows
			if row['comparison_id'] == comparison_id
		]
		comparison_aggregates[comparison_id] = _aggregate(values)
	repository_state = _mapping(
		final_result.get('repository_state'), label='repository_state'
	)
	gate = _mapping(final_result.get('medium_gate'), label='medium_gate')
	reached_sizes = list(DATA_SIZES if gate['gate_open'] else ('medium',))
	benchmark_provenance = _benchmark_provenance_summary(
		settings=settings,
		final_result=final_result,
		artifact_root=artifact_root,
	)
	return {
		'schema_version': 1,
		'artifact_type': 'f3_local_barlow_twins_gaussian_view_validation_report',
		'experiment_id': EXPERIMENT_ID,
		'pipeline_input': False,
		'validation_only': True,
		'test_data_or_metrics_used': False,
		'created_at_utc': final_result['created_at_utc'],
		'repository_state': {
			'git_head': repository_state.get('git_head'),
			'git_dirty': repository_state.get('git_dirty'),
			'relevant_git_status_sha256': repository_state.get(
				'relevant_git_status_sha256'
			),
		},
		'source_artifacts': {
			'validation_config': {
				'path': _display_path(config_path, artifact_root=artifact_root),
				'sha256': validation_config_sha256,
			},
			'validation_runner': {
				'path': _display_path(RUNNER_PATH, artifact_root=artifact_root),
				'sha256': validation_runner_sha256,
			},
			'canonical_five_way_config': {
				'path': _display_path(
					cast('Path', settings.canonical_five_way_config),
					artifact_root=artifact_root,
				),
				'sha256': canonical_config_sha256,
			},
			'protocol_lock': {
				'path': _display_path(
					cast('Path', settings.protocol_lock), artifact_root=artifact_root
				),
				'sha256': protocol_lock_sha256,
			},
			'selection_lock': {
				'path': _display_path(
					cast('Path', settings.selection_lock), artifact_root=artifact_root
				),
				'sha256': selection_lock_sha256,
			},
			'final_result': {
				'path': _display_path(
					cast('Path', settings.final_result), artifact_root=artifact_root
				),
				'sha256': final_result_sha256,
			},
		},
		'protocol': {
			'evaluation_split': 'validation',
			'evaluation_aggregation_unit': VALIDATION_AGGREGATION_UNIT,
			'primary_metric': 'macro_f1',
			'statistical_unit': 'layout_id',
			'layout_ids': list(LAYOUT_IDS),
			'reached_data_sizes': reached_sizes,
			'base_pretraining_epochs': final_result['base_pretraining_epochs'],
			'continuation_epochs': final_result['continuation_epochs'],
			'view_selection_size': 'medium',
			'view_selection_rule': 'maximum unrounded mean macro_f1',
			'medium_gate': 'strict positive paired delta in 5/5 layouts',
			'final_gate': 'strict positive paired delta in 15/15 layout/size cells',
		},
		'execution_evidence': {
			'base_launch_freshness': 'operator_observed',
			'output_root_absence_checked_before_launch': True,
			'resume_argument_used': False,
			'checkpoint_authenticated_resume_count': False,
			'limitation': (
				'the 25-epoch base checkpoint schema predates an invocation resume '
				'counter, so freshness is external execution evidence rather than '
				'checkpoint-authenticated'
			),
		},
		'benchmark_provenance': benchmark_provenance,
		'selection': {
			'created_at_utc': selection_lock['created_at_utc'],
			'candidate_means': selection_lock['candidate_means'],
			'selected_candidate_id': selected_id,
			'selected_view_policy': selection_lock['selected_view_policy'],
			'selected_gaussian_noise_std': selection_lock[
				'selected_gaussian_noise_std'
			],
			'tie_rule': selection_lock['tie_rule'],
			'tie_priority': selection_lock['tie_priority'],
			'input_count': len(cast('list[object]', selection_lock['inputs'])),
			'fixed_strength_geometry_contrast': selection_lock[
				'fixed_strength_geometry_contrast'
			],
		},
		'view_diagnostic': dict(view_diagnostic),
		'medium_gate': dict(gate),
		'final_validation': {
			'passed': final_result['passed'],
			'winner_candidate_id': final_result['winner_candidate_id'],
			'authorizes_next_base_duration': final_result[
				'authorizes_next_base_duration'
			],
			'failure_stage': final_result['failure_stage'],
			'arm_results': final_result['arm_results'],
			'gaussian_attribution': final_result['gaussian_attribution'],
			'identity_vs_forced_geometry': final_result[
				'identity_vs_forced_geometry'
			],
		},
		'evidence_counts': {
			'attempts': len(attempt_rows),
			'validation_cells': len(validation_rows),
			'paired_deltas': len(paired_rows),
		},
		'macro_f1_by_source_and_size': by_size,
		'paired_delta_aggregates': comparison_aggregates,
		'report_files': list(REPORT_FILENAMES),
	}


def _summary_markdown(
	summary: Mapping[str, object],
	attempt_rows: Sequence[Mapping[str, object]],
) -> str:
	selection = _mapping(summary['selection'], label='selection')
	final = _mapping(summary['final_validation'], label='final_validation')
	diagnostic = _mapping(summary['view_diagnostic'], label='view_diagnostic')
	diagnostic_views = _mapping(diagnostic['views'], label='view diagnostic views')
	legacy_view = _mapping(diagnostic_views['legacy'], label='legacy diagnostic')
	std005_view = _mapping(
		diagnostic_views['gaussian_noise_std005'], label='std005 diagnostic'
	)
	std010_view = _mapping(
		diagnostic_views['gaussian_noise_std010'], label='std010 diagnostic'
	)
	lines = [
		'# F3 Local Barlow Twins Gaussian-view validation report',
		'',
		(
			'This is a validation-only, human-reviewable report. It is never a '
			'pipeline input, and no test data or test metric was used.'
		),
		'',
		(
			'The actual-data view diagnostic found inverse-aligned legacy views '
			f'identical (correlation {float(legacy_view["paired_correlation"]):.6f}, '
			f'RMS {float(legacy_view["paired_rms"]):.6f}); Gaussian std 0.05 and '
			f'0.10 reduced correlation to '
			f'{float(std005_view["paired_correlation"]):.6f} and '
			f'{float(std010_view["paired_correlation"]):.6f}, with paired RMS '
			f'{float(std005_view["paired_rms"]):.6f} and '
			f'{float(std010_view["paired_rms"]):.6f}. This diagnostic does not '
			'cover the identity policy or rule out its fixed-position shortcut.'
		),
		'',
		'## View selection',
		'',
		(
			f'The immutable medium-layout lock selected '
			f'`{selection["selected_candidate_id"]}` using unrounded mean '
			'validation macro-F1 over five layouts.'
		),
		'',
		'| candidate | medium mean macro-F1 |',
		'|---|---:|',
	]
	means = _mapping(selection['candidate_means'], label='candidate_means')
	lines.extend(
		f'| `{candidate_id}` | {float(means[candidate_id]):.9f} |'
		for candidate_id in CANDIDATE_ORDER
	)
	lines.extend(
		[
			'',
			'## Attempt ledger',
			'',
			(
				'| candidate | base + continuation epochs | medium mean | '
				'medium wins | evaluated cells | wins vs random | outcome |'
			),
			'|---|---:|---:|---:|---:|---:|---|',
		]
	)
	lines.extend(
		(
			f'| `{row["candidate_id"]}` '
			f'| {row["base_pretraining_epochs"]} + {row["continuation_epochs"]} '
			f'| {float(row["medium_macro_f1_mean"]):.9f} '
			f'| {row["medium_positive_count"]}/5 '
			f'| {row["evaluated_cell_count"]} '
			f'| {row["positive_vs_random_count"]}/{row["evaluated_cell_count"]} '
			f'| {row["final_outcome"]} |'
		)
		for row in attempt_rows
	)
	passed = _bool(final['passed'], 'final_validation.passed')
	lines.extend(['', '## Final validation result', ''])
	if passed:
		winner = cast('str', final['winner_candidate_id'])
		arm_results = _mapping(final['arm_results'], label='arm_results')
		winner_result = _mapping(arm_results[winner], label='winner result')
		lines.append(
			f'PASS: `{winner}` has strict positive paired macro-F1 deltas over '
			f'random in {winner_result["positive_delta_count"]}/15 validation cells.'
		)
	else:
		lines.append(
			'FAIL for this reached duration: no arm met the preregistered final '
			f'criterion. Failure stage: `{final["failure_stage"]}`.'
		)
	lines.extend(
		[
			'',
			(
				'Gaussian-minus-legacy attribution and identity-minus-forced-flip '
				'geometry contrasts are recorded separately in `paired_deltas.csv` and '
				'`summary.json`; they are not conflated with random-baseline success.'
			),
			'',
			(
				'All paths are repository-relative or rooted at '
				'`${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}`. Exact live-file SHA-256 '
				'identities are recorded in the CSV and JSON outputs.'
			),
			(
				'Base runs were operator-observed fresh launches after output-root '
				'absence checks and without `--resume`; the 25-epoch base checkpoint '
				'schema has no invocation resume counter, so this particular fact is '
				'not checkpoint-authenticated.'
			),
			'',
		]
	)
	return '\n'.join(lines)


def _csv_text(
	fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> str:
	buffer = io.StringIO()
	writer = csv.DictWriter(
		buffer, fieldnames=fieldnames, extrasaction='raise', lineterminator='\n'
	)
	writer.writeheader()
	for row in rows:
		writer.writerow({key: row[key] for key in fieldnames})
	return buffer.getvalue()


def _publish_outputs(
	output_dir: Path,
	outputs: Mapping[str, str],
	*,
	source_snapshots: Sequence[tuple[Path, str, str]] = (),
) -> None:
	if set(outputs) != set(REPORT_FILENAMES):
		raise ValueError('report output mapping must define exactly five files')
	if output_dir.exists() or output_dir.is_symlink():
		raise FileExistsError(f'refusing to overwrite report directory: {output_dir}')
	output_dir.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix=f'.{output_dir.name}.staging-', dir=output_dir.parent
		)
	)
	try:
		for name in REPORT_FILENAMES:
			(staging / name).write_text(outputs[name], encoding='utf-8')
		_assert_source_snapshots_unchanged(source_snapshots)
		staging.replace(output_dir)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise


def build_report(
	config_path: Path, output_dir: Path = REPORT_OUTPUT_DIR
) -> dict[str, object]:
	"""Reaudit immutable evidence and exclusively publish five tracked files."""
	if config_path.resolve() != DEFAULT_CONFIG.resolve():
		raise ValueError(
			f'report requires the live experiment config: {DEFAULT_CONFIG}'
		)
	if output_dir.resolve(strict=False) != REPORT_OUTPUT_DIR.resolve(strict=False):
		raise ValueError(f'report output directory must equal {REPORT_OUTPUT_DIR}')
	runner = _load_runner_namespace()
	settings = cast('Any', runner['validation_settings_from_mapping'])(
		load_config(config_path)
	)
	canonical = cast('Any', runner['_canonical_config'])(settings)
	artifact_root = Path(canonical.artifact_root)
	validation_config_sha256 = _file_sha256(
		config_path, label='validation config'
	)
	validation_runner_sha256 = _file_sha256(
		RUNNER_PATH, label='validation runner'
	)
	canonical_config_path = cast('Path', settings.canonical_five_way_config)
	canonical_config_sha256 = _file_sha256(
		canonical_config_path, label='canonical five-way config'
	)
	if canonical_config_sha256 != settings.canonical_five_way_config_sha256:
		raise ValueError('canonical five-way config differs from its pinned SHA-256')
	protocol_lock, protocol_lock_sha256 = _read_hashed_json(
		settings.protocol_lock, label='protocol lock'
	)
	_replay_protocol_lock(
		runner=runner,
		settings=settings,
		canonical=canonical,
		stored=protocol_lock,
	)
	selection_lock, selection_lock_sha256 = _read_hashed_json(
		settings.selection_lock, label='selection lock'
	)
	final_result, final_result_sha256 = _read_hashed_json(
		settings.final_result, label='final result'
	)
	source_snapshots = (
		(config_path, validation_config_sha256, 'validation config'),
		(RUNNER_PATH, validation_runner_sha256, 'validation runner'),
		(
			canonical_config_path,
			canonical_config_sha256,
			'canonical five-way config',
		),
		(settings.protocol_lock, protocol_lock_sha256, 'protocol lock'),
		(settings.selection_lock, selection_lock_sha256, 'selection lock'),
		(settings.final_result, final_result_sha256, 'final result'),
	)
	_replay_final_result(
		runner=runner,
		settings=settings,
		canonical=canonical,
		stored=final_result,
	)
	_assert_source_snapshots_unchanged(source_snapshots)
	validation_rows = _collect_validation_cells(
		runner=runner,
		settings=settings,
		canonical=canonical,
		final_result=final_result,
		artifact_root=artifact_root,
	)
	selected_id = cast(
		'str', _mapping(final_result['selection_lock'], label='selection_lock')[
			'selected_candidate_id'
		]
	)
	paired_rows = _paired_delta_rows(
		validation_rows,
		selected_id=selected_id,
		base_pretraining_epochs=cast('int', final_result['base_pretraining_epochs']),
	)
	attempt_rows = _collect_attempt_rows(
		runner=runner,
		settings=settings,
		canonical=canonical,
		final_result=final_result,
		validation_rows=validation_rows,
		artifact_root=artifact_root,
	)
	view_diagnostic = _view_diagnostic_summary(artifact_root)
	summary = _summary_payload(
		config_path=config_path,
		validation_config_sha256=validation_config_sha256,
		validation_runner_sha256=validation_runner_sha256,
		canonical_config_sha256=canonical_config_sha256,
		artifact_root=artifact_root,
		settings=settings,
		protocol_lock_sha256=protocol_lock_sha256,
		selection_lock=selection_lock,
		selection_lock_sha256=selection_lock_sha256,
		final_result=final_result,
		final_result_sha256=final_result_sha256,
		validation_rows=validation_rows,
		paired_rows=paired_rows,
		attempt_rows=attempt_rows,
		view_diagnostic=view_diagnostic,
	)
	outputs = {
		'attempts.csv': _csv_text(ATTEMPT_FIELDS, attempt_rows),
		'validation_cells.csv': _csv_text(
			VALIDATION_CELL_FIELDS, validation_rows
		),
		'paired_deltas.csv': _csv_text(PAIRED_DELTA_FIELDS, paired_rows),
		'summary.json': json.dumps(
			summary, indent=2, sort_keys=True, allow_nan=False
		)
		+ '\n',
		'summary.md': _summary_markdown(summary, attempt_rows),
	}
	_publish_outputs(
		output_dir, outputs, source_snapshots=source_snapshots
	)
	return {
		'output_dir': str(output_dir),
		'outputs': [str(output_dir / name) for name in REPORT_FILENAMES],
		'passed': final_result['passed'],
		'winner_candidate_id': final_result['winner_candidate_id'],
		'validation_cell_count': len(validation_rows),
		'paired_delta_count': len(paired_rows),
	}


def build_parser() -> argparse.ArgumentParser:
	"""Build the report-only CLI parser."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	return parser


def main() -> None:
	"""Build the fixed report and print its immutable output inventory."""
	args = build_parser().parse_args()
	result = build_report(args.config)
	for key, value in result.items():
		print(f'{key}: {value}')


if __name__ == '__main__':
	main()
