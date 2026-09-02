'''Completeness audit and paired summary for the Volve 75-job five-way suite.'''

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import statistics
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_five_way_config import FIVE_WAY_MODEL_IDS
from seis_ssl_cluster.volve.horizon_five_way_runner import (
	FIVE_WAY_CONDITION_COUNT,
)
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	VolveHorizonFiveWayEmbeddingSuite,
	audit_volve_horizon_five_way_sources,
	inspect_volve_horizon_five_way_embedding_suite,
)
from seis_ssl_cluster.volve.horizon_frozen import (
	OPTIMIZER_BETAS,
	OPTIMIZER_EPS,
	OPTIMIZER_NAME,
	decoder_initial_state_sha256,
	objective_identity,
)
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.volve.horizon_model import create_volve_horizon_decoder
from seis_ssl_cluster.volve.horizon_runner import (
	BEST_NAME,
	CHECKPOINT_SELECTION_VALIDATION_MAE,
	CHECKPOINT_SELECTION_VALIDATION_WITHIN_2,
	HISTORY_NAME,
	METRICS_NAME,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.volve.horizon_five_way_config import (
		VolveHorizonFiveWayConfig,
	)

MODEL_IDS = FIVE_WAY_MODEL_IDS
EXPECTED_JOB_COUNT = FIVE_WAY_CONDITION_COUNT
EXPECTED_METRICS_ARTIFACT_TYPE = 'volve_frozen_horizon_job_metrics'
COMPARISON_CSV_NAME = 'comparison.csv'
PAIRED_DELTAS_CSV_NAME = 'paired_deltas.csv'
SUMMARY_BY_SIZE_CSV_NAME = 'summary_by_size.csv'
SUMMARY_JSON_NAME = 'summary.json'
SUMMARY_MD_NAME = 'summary.md'
SUMMARY_OUTPUT_NAMES = (
	COMPARISON_CSV_NAME,
	PAIRED_DELTAS_CSV_NAME,
	SUMMARY_BY_SIZE_CSV_NAME,
	SUMMARY_JSON_NAME,
	SUMMARY_MD_NAME,
)
PRIMARY_METRIC = 'macro_mae_samples'
WITHIN2_PRIMARY_METRIC = 'macro_within_2_samples'
MACRO_WITHIN_1_METRIC = 'macro_within_1_samples'
MACRO_WITHIN_4_METRIC = 'macro_within_4_samples'
ORDER_VIOLATION_METRIC = 'predicted_adjacent_order_violation_rate'
PER_HORIZON_METRICS = tuple(
	f'{horizon_name}_mae_samples' for horizon_name in HORIZON_NAMES
)
SUMMARY_METRICS = (PRIMARY_METRIC, *PER_HORIZON_METRICS)
WITHIN2_SUMMARY_METRICS = (
	WITHIN2_PRIMARY_METRIC,
	PRIMARY_METRIC,
	MACRO_WITHIN_1_METRIC,
	MACRO_WITHIN_4_METRIC,
	ORDER_VIOLATION_METRIC,
)
PAIRED_COMPARISONS = (
	('mae_minus_mae_hmm_k6', 'mae', 'mae_hmm_k6'),
	(
		'local_bt_minus_local_bt_hmm_k6',
		'local_barlow_twins',
		'local_barlow_twins_hmm_k6',
	),
	('mae_minus_local_bt', 'mae', 'local_barlow_twins'),
	(
		'mae_hmm_k6_minus_local_bt_hmm_k6',
		'mae_hmm_k6',
		'local_barlow_twins_hmm_k6',
	),
	('random_minus_mae', 'random', 'mae'),
	('random_minus_mae_hmm_k6', 'random', 'mae_hmm_k6'),
	('random_minus_local_bt', 'random', 'local_barlow_twins'),
	(
		'random_minus_local_bt_hmm_k6',
		'random',
		'local_barlow_twins_hmm_k6',
	),
)
COMPARISON_FIELDNAMES = (
	'model_id',
	'layout_id',
	'data_size',
	PRIMARY_METRIC,
	*PER_HORIZON_METRICS,
	'primary_eligible_count',
	'primary_predicted_count',
	'primary_coverage_fraction',
	'best_epoch',
	'checkpoint_path',
	'checkpoint_sha256',
	'embeddings_dir',
	'embeddings_sha256',
	'embedding_metadata_path',
	'embedding_metadata_sha256',
	'valid_tokens_sha256',
	'split_plan_sha256',
	'decoder_initial_state_sha256',
	'best_checkpoint_path',
	'best_checkpoint_sha256',
	'metrics_path',
)
PAIRED_FIELDNAMES = (
	'data_size',
	'layout_id',
	'comparison_id',
	'metric',
	'left_model',
	'right_model',
	'left_value',
	'right_value',
	'delta',
)
BY_SIZE_FIELDNAMES = (
	'data_size',
	'comparison_id',
	'metric',
	'n_layouts',
	'mean',
	'median',
	'sample_std',
	'min',
	'max',
	'positive_count',
	'zero_count',
	'negative_count',
)
_SHARED_RUN_IDENTITY_KEYS = (
	'schema_version',
	'benchmark',
	'canonical_scientific_identity',
	'horizon_split_plan',
	'decoder',
	'tiles',
	'native_horizon_observation_counts',
	'effective_model_valid_observation_counts',
	'excluded_by_token_validity_counts',
	'training',
	'optimizer',
	'objective',
	'runtime_precision',
)


def inspect_volve_horizon_five_way_results(
	config: VolveHorizonFiveWayConfig,
) -> dict[str, object]:
	'''Audit every completed cell without writing summary artifacts.'''
	if tuple(config.model_ids) != MODEL_IDS:
		raise ValueError(
			'five-way result audit requires the fixed model order '
			f'{MODEL_IDS!r}; got {tuple(config.model_ids)!r}'
		)
	cells = tuple(
		(model_id, layout_id, data_size)
		for model_id in config.model_ids
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZE_PREFIX
	)
	if len(cells) != EXPECTED_JOB_COUNT or len(set(cells)) != EXPECTED_JOB_COUNT:
		raise RuntimeError('Volve five-way result matrix must contain 75 cells')
	source_audit = audit_volve_horizon_five_way_sources(config)
	embedding_suite = inspect_volve_horizon_five_way_embedding_suite(
		config,
		source_audit=source_audit,
	)
	_reject_unexpected_run_directories(config)
	missing = [
		str(_job_dir(config, *cell) / METRICS_NAME)
		for cell in cells
		if not (_job_dir(config, *cell) / METRICS_NAME).is_file()
	]
	if missing:
		raise FileNotFoundError(
			f'missing {len(missing)} of {EXPECTED_JOB_COUNT} Volve five-way '
			f'evaluations: {missing!r}'
		)
	sources = _inspect_configured_sources(config, embedding_suite=embedding_suite)
	expected_downstream = _expected_downstream_contract(config)
	rows = [
		_load_job_row(
			config,
			model_id=model_id,
			layout_id=layout_id,
			data_size=data_size,
			source=sources[model_id],
			expected_downstream=expected_downstream,
		)
		for model_id, layout_id, data_size in cells
	]
	_validate_cross_job_identity(rows)
	for row in rows:
		row.pop('_shared_run_identity')
		row.pop('_support_identity')
	return {
		'complete_jobs': len(rows),
		'model_order': list(config.model_ids),
		'rows': rows,
	}


def summarize_volve_horizon_five_way(
	config: VolveHorizonFiveWayConfig,
) -> dict[str, object]:
	'''Atomically write five summaries after the complete identity audit.'''
	report = inspect_volve_horizon_five_way_results(config)
	rows = report['rows']
	if not isinstance(rows, list):
		raise TypeError('five-way inspection rows must be a list')
	paired = _paired_rows(rows, config=config)
	by_size = _by_size_rows(paired, config=config)
	summary_payload = _summary_payload(config, rows, by_size)
	comparison_fieldnames = _comparison_fieldnames(config)
	outputs = {
		COMPARISON_CSV_NAME: _csv_text(comparison_fieldnames, rows),
		PAIRED_DELTAS_CSV_NAME: _csv_text(PAIRED_FIELDNAMES, paired),
		SUMMARY_BY_SIZE_CSV_NAME: _csv_text(BY_SIZE_FIELDNAMES, by_size),
		SUMMARY_JSON_NAME: json.dumps(
			summary_payload,
			indent=2,
			sort_keys=True,
			allow_nan=False,
		)
		+ '\n',
		SUMMARY_MD_NAME: _summary_markdown(config, by_size),
	}
	summary_root = config.summary_root
	if summary_root.exists():
		raise FileExistsError(
			f'refusing to overwrite existing five-way summary: {summary_root}'
		)
	summary_root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix=f'.{summary_root.name}.staging-',
			dir=summary_root.parent,
		)
	)
	try:
		for name, text in outputs.items():
			(staging / name).write_text(text, encoding='utf-8')
		_publish_staging(staging, summary_root)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	return {
		'complete_jobs': report['complete_jobs'],
		'summary_root': str(summary_root),
		'outputs': [str(summary_root / name) for name in SUMMARY_OUTPUT_NAMES],
	}


def _publish_staging(staging: Path, summary_root: Path) -> None:
	if summary_root.exists():
		raise FileExistsError(
			f'refusing to overwrite existing five-way summary: {summary_root}'
		)
	staging.replace(summary_root)


def _job_dir(
	config: VolveHorizonFiveWayConfig,
	model_id: str,
	layout_id: str,
	data_size: str,
) -> Path:
	return (
		config.runs_root
		/ f'model={model_id}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)


def _reject_unexpected_run_directories(
	config: VolveHorizonFiveWayConfig,
) -> None:
	if not config.runs_root.exists():
		return
	if not config.runs_root.is_dir():
		raise ValueError(f'five-way runs root must be a directory: {config.runs_root}')
	_expected_children(
		config.runs_root,
		{f'model={model_id}' for model_id in config.model_ids},
	)
	for model_id in config.model_ids:
		model_dir = config.runs_root / f'model={model_id}'
		if not model_dir.exists():
			continue
		_expected_children(
			model_dir,
			{f'layout={layout_id}' for layout_id in LAYOUT_IDS},
		)
		for layout_id in LAYOUT_IDS:
			layout_dir = model_dir / f'layout={layout_id}'
			if not layout_dir.exists():
				continue
			_expected_children(
				layout_dir,
				{f'size={data_size}' for data_size in DATA_SIZE_PREFIX},
			)


def _expected_children(root: Path, expected: set[str]) -> None:
	for entry in sorted(root.iterdir()):
		if entry.name not in expected or not entry.is_dir():
			raise ValueError(f'unexpected Volve five-way run directory: {entry}')


def _inspect_configured_sources(
	config: VolveHorizonFiveWayConfig,
	*,
	embedding_suite: VolveHorizonFiveWayEmbeddingSuite,
) -> dict[str, dict[str, object]]:
	sources: dict[str, dict[str, object]] = {}
	for model_id in config.model_ids:
		model = config.model_by_id(model_id)
		inspected = embedding_suite.source_by_id(model_id)
		checkpoint_sha256 = _sha256(
			inspected.checkpoint_identity.get('checkpoint_sha256'),
			f'{model_id} source audit checkpoint_sha256',
		)
		sources[model_id] = {
			'model': model,
			'checkpoint_path': str(model.checkpoint),
			'checkpoint_sha256': checkpoint_sha256,
			'embeddings_dir': str(model.embeddings_dir),
			'embeddings_path': str(inspected.paths.embeddings),
			'embeddings_sha256': inspected.embeddings_sha256,
			'embedding_metadata_path': str(inspected.paths.metadata),
			'embedding_metadata_sha256': inspected.metadata_sha256,
			'valid_tokens_sha256': inspected.valid_tokens_sha256,
			'model_source': dict(inspected.checkpoint_identity),
		}
	return sources


def _load_job_row(  # noqa: C901, PLR0912, PLR0913, PLR0915
	config: VolveHorizonFiveWayConfig,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
	source: Mapping[str, object],
	expected_downstream: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
	job_dir = _job_dir(config, model_id, layout_id, data_size)
	metrics_path = job_dir / METRICS_NAME
	metrics = _read_json(metrics_path)
	label = f'{model_id}/{layout_id}/{data_size}'
	_reject_nonfinite_numbers(metrics, label=f'{label} metrics')
	if metrics.get('schema_version') != 1:
		raise ValueError(f'{label} metrics schema_version must equal 1')
	if metrics.get('artifact_type') != EXPECTED_METRICS_ARTIFACT_TYPE:
		raise ValueError(
			f'{label} metrics artifact_type must equal '
			f'{EXPECTED_METRICS_ARTIFACT_TYPE!r}'
		)
	for key, expected in (
		('model', model_id),
		('layout_id', layout_id),
		('data_size', data_size),
	):
		if metrics.get(key) != expected:
			raise ValueError(
				f'{label} metrics {key} must equal {expected!r}; '
				f'got {metrics.get(key)!r}'
			)
	identity = _required_mapping(metrics, 'benchmark_identity', label)
	if identity.get('schema_version') != 3:
		raise ValueError(f'{label} benchmark identity schema_version must equal 3')
	if identity.get('benchmark') != config.benchmark_id:
		raise ValueError(
			f'{label} benchmark identity must equal {config.benchmark_id!r}'
		)
	_validate_downstream_contract(
		identity,
		expected=expected_downstream,
		label=label,
	)
	for key, expected in (
		('model', model_id),
		('layout_id', layout_id),
		('data_size', data_size),
	):
		if identity.get(key) != expected:
			raise ValueError(
				f'{label} benchmark identity {key} must equal {expected!r}'
			)
	split_identity = _required_mapping(
		identity, 'horizon_split_plan', f'{label} identity'
	)
	for key, expected in (('layout_id', layout_id), ('data_size', data_size)):
		if split_identity.get(key) != expected:
			raise ValueError(
				f'{label} split identity {key} must equal {expected!r}'
			)
	embedding = _required_mapping(identity, 'embedding', f'{label} identity')
	if not _same_path_value(
		embedding.get('embeddings_path'), Path(str(source['embeddings_path']))
	):
		raise ValueError(
			f'{label} embedding array path does not match configured source'
		)
	if embedding.get('embeddings_sha256') != source['embeddings_sha256']:
		raise ValueError(
			f'{label} embedding array SHA-256 does not match configured source'
		)
	if not _same_path_value(
		embedding.get('checkpoint_path'), Path(str(source['checkpoint_path']))
	):
		raise ValueError(
			f'{label} encoder checkpoint path does not match configured source'
		)
	if embedding.get('checkpoint_sha256') != source['checkpoint_sha256']:
		raise ValueError(
			f'{label} encoder checkpoint SHA-256 does not match configured source'
		)
	if not _same_path_value(
		embedding.get('metadata_path'),
		Path(str(source['embedding_metadata_path'])),
	):
		raise ValueError(
			f'{label} embedding metadata path does not match configured source'
		)
	if embedding.get('metadata_sha256') != source['embedding_metadata_sha256']:
		raise ValueError(
			f'{label} embedding metadata SHA-256 does not match configured source'
		)
	if embedding.get('valid_tokens_sha256') != source['valid_tokens_sha256']:
		raise ValueError(
			f'{label} valid-token SHA-256 does not match configured source'
		)
	model_source = _required_mapping(embedding, 'model_source', f'{label} embedding')
	if _json_normalized(model_source) != _json_normalized(source['model_source']):
		raise ValueError(
			f'{label} embedding model_source does not match the source audit'
		)
	test_metrics = _required_mapping(metrics, 'test', label)
	if test_metrics.get('evaluation_pass_count') != 1:
		raise ValueError(f'{label} test evaluation_pass_count must equal 1')
	primary = _required_mapping(
		test_metrics,
		'primary_common',
		f'{label} test',
	)
	secondary = _required_mapping(
		test_metrics,
		'secondary_per_horizon',
		f'{label} test',
	)
	primary_values, primary_support = _metric_values_and_support(
		primary,
		label=f'{label} primary common test',
	)
	_, secondary_support = _metric_values_and_support(
		secondary,
		label=f'{label} secondary per-horizon test',
	)
	validation = _required_mapping(metrics, 'validation', label)
	_, validation_support = _metric_values_and_support(
		validation,
		label=f'{label} validation',
	)
	_validate_observation_counts(
		identity,
		validation_support=validation_support,
		primary_support=primary_support,
		secondary_support=secondary_support,
		label=label,
	)
	best_epoch = _nonnegative_int(metrics.get('best_epoch'), f'{label} best_epoch')
	best_identity = _required_mapping(metrics, 'best_checkpoint', label)
	best_path = job_dir / BEST_NAME
	if not _same_path_value(best_identity.get('path'), best_path):
		raise ValueError(
			f'{label} best checkpoint path does not identify {best_path}'
		)
	if not best_path.is_file():
		raise FileNotFoundError(f'{label} best checkpoint is missing: {best_path}')
	best_sha256 = file_sha256(best_path)
	if best_identity.get('sha256') != best_sha256:
		raise ValueError(f'{label} best checkpoint SHA-256 mismatch')
	best_payload = load_checkpoint_metadata_without_weights(best_path)
	if best_payload.get('epoch') != best_epoch:
		raise ValueError(f'{label} selected best epoch does not match best.pt')
	if _json_normalized(best_payload.get('run_identity')) != identity:
		raise ValueError(f'{label} best checkpoint run identity mismatch')
	if _json_normalized(best_payload.get('validation')) != metrics.get('validation'):
		raise ValueError(f'{label} selected validation metrics mismatch best.pt')
	if best_payload.get('runtime_precision') != metrics.get('runtime_precision'):
		raise ValueError(f'{label} best checkpoint runtime precision mismatch')
	if identity.get('runtime_precision') != metrics.get('runtime_precision'):
		raise ValueError(f'{label} metrics runtime precision identity mismatch')
	_validate_checkpoint_selection_artifacts(
		config,
		job_dir=job_dir,
		metrics=metrics,
		best_payload=best_payload,
		validation=validation,
		best_epoch=best_epoch,
		label=label,
	)
	shared_identity = {}
	for key in _SHARED_RUN_IDENTITY_KEYS:
		if key not in identity:
			raise ValueError(f'{label} benchmark identity is missing {key}')
		shared_identity[key] = identity[key]
	split_plan = split_identity
	decoder = _required_mapping(identity, 'decoder', f'{label} identity')
	coverage = _required_mapping(primary, 'coverage', f'{label} primary test')
	return {
		'model_id': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		**primary_values,
		'primary_eligible_count': _nonnegative_int(
			coverage.get('eligible_count'),
			f'{label} primary eligible_count',
		),
		'primary_predicted_count': _nonnegative_int(
			coverage.get('predicted_count'),
			f'{label} primary predicted_count',
		),
		'primary_coverage_fraction': _finite_number(
			coverage.get('fraction'),
			f'{label} primary coverage fraction',
		),
		'best_epoch': best_epoch,
		'checkpoint_path': source['checkpoint_path'],
		'checkpoint_sha256': source['checkpoint_sha256'],
		'embeddings_dir': source['embeddings_dir'],
		'embeddings_sha256': source['embeddings_sha256'],
		'embedding_metadata_path': source['embedding_metadata_path'],
		'embedding_metadata_sha256': source['embedding_metadata_sha256'],
		'valid_tokens_sha256': source['valid_tokens_sha256'],
		'split_plan_sha256': _sha256(
			split_plan.get('scientific_identity_sha256'),
			f'{label} split plan identity',
		),
		'decoder_initial_state_sha256': _sha256(
			decoder.get('initial_state_sha256'),
			f'{label} decoder initial state identity',
		),
		'best_checkpoint_path': str(best_path),
		'best_checkpoint_sha256': best_sha256,
		'metrics_path': str(metrics_path),
		'_shared_run_identity': shared_identity,
		'_support_identity': {
			'validation': validation_support,
			'primary': primary_support,
			'secondary': secondary_support,
		},
	}


def _expected_downstream_contract(
	config: VolveHorizonFiveWayConfig,
) -> dict[str, dict[str, object]]:
	return {
		'decoder': {
			'architecture': create_volve_horizon_decoder().architecture,
			'initialization_seed': config.train.seed,
			'initial_state_sha256': decoder_initial_state_sha256(config.train.seed),
		},
		'tiles': {
			'patch_size_xyz': list(config.tiles.patch_size_xyz),
			'core_size_tokens': list(config.tiles.core_size_tokens),
			'context_halo_tokens': list(config.tiles.context_halo_tokens),
			'window_start': config.tiles.window_start,
			'window_stop': config.tiles.window_stop,
			'order': 'lateral_token_grid_x_then_y_v1',
		},
		'training': {
			'epochs': config.train.epochs,
			'batch_size': config.train.batch_size,
			'learning_rate': config.train.learning_rate,
			'weight_decay': config.train.weight_decay,
			'sampling_mode': config.train.sampling_mode,
			'seed': config.train.seed,
			'amp_on_cuda': config.train.amp,
			'gradient_clip_norm': config.train.gradient_clip_norm,
		},
		'optimizer': {
			'name': OPTIMIZER_NAME,
			'betas': list(OPTIMIZER_BETAS),
			'eps': OPTIMIZER_EPS,
			'weight_decay': config.train.weight_decay,
		},
		'objective': objective_identity(config.checkpoint_selection),
	}


def _validate_checkpoint_selection_artifacts(  # noqa: C901, PLR0912, PLR0913
	config: VolveHorizonFiveWayConfig,
	*,
	job_dir: Path,
	metrics: Mapping[str, object],
	best_payload: Mapping[str, object],
	validation: Mapping[str, object],
	best_epoch: int,
	label: str,
) -> None:
	selection = config.checkpoint_selection
	if selection == CHECKPOINT_SELECTION_VALIDATION_MAE:
		validation_key = PRIMARY_METRIC
		history_key = 'validation_macro_mae_samples'
		best_from_history = min
	elif selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2:
		validation_key = WITHIN2_PRIMARY_METRIC
		history_key = 'validation_macro_within_2_samples'
		best_from_history = max
	else:  # Config loading rejects this before artifact inspection.
		raise ValueError(f'unknown horizon checkpoint selection: {selection!r}')

	checkpoint_selection = best_payload.get('checkpoint_selection')
	if checkpoint_selection is None:
		if selection != CHECKPOINT_SELECTION_VALIDATION_MAE:
			raise ValueError(f'{label} best.pt is missing checkpoint_selection')
	elif checkpoint_selection != selection:
		raise ValueError(f'{label} best.pt checkpoint selection mismatch')

	validation_score = _finite_number(
		validation.get(validation_key),
		f'{label} validation {validation_key}',
	)
	if 'best_validation_score' in best_payload:
		best_score = _finite_number(
			best_payload.get('best_validation_score'),
			f'{label} best.pt best_validation_score',
		)
	elif selection == CHECKPOINT_SELECTION_VALIDATION_MAE:
		best_score = validation_score
	else:
		raise ValueError(f'{label} best.pt is missing best_validation_score')
	if not math.isclose(
		best_score, validation_score, rel_tol=1.0e-12, abs_tol=1.0e-12
	):
		raise ValueError(f'{label} best validation score differs from best.pt')

	metrics_selection = metrics.get('checkpoint_selection')
	if metrics_selection is not None and metrics_selection != selection:
		raise ValueError(f'{label} metrics checkpoint selection mismatch')
	if selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2 and (
		metrics_selection is None
	):
		raise ValueError(f'{label} metrics is missing checkpoint_selection')
	if 'best_validation_score' in metrics:
		metrics_score = _finite_number(
			metrics.get('best_validation_score'),
			f'{label} metrics best_validation_score',
		)
		if not math.isclose(
			metrics_score, best_score, rel_tol=1.0e-12, abs_tol=1.0e-12
		):
			raise ValueError(f'{label} metrics best validation score mismatch')
	elif selection == CHECKPOINT_SELECTION_VALIDATION_WITHIN_2:
		raise ValueError(f'{label} metrics is missing best_validation_score')

	history = _read_history(job_dir / HISTORY_NAME)
	epochs = [
		_nonnegative_int(row.get('epoch'), f'{label} history epoch')
		for row in history
	]
	if not epochs or epochs != sorted(epochs) or len(epochs) != len(set(epochs)):
		raise ValueError(f'{label} history epochs must be unique and increasing')
	scores = [
		_finite_number(row.get(history_key), f'{label} history {history_key}')
		for row in history
	]
	selected_score = best_from_history(scores)
	selected_epoch = min(
		epoch
		for epoch, score in zip(epochs, scores, strict=True)
		if score == selected_score
	)
	if selected_epoch != best_epoch:
		raise ValueError(
			f'{label} best_epoch does not match the first optimal history epoch'
		)
	if not math.isclose(
		selected_score, best_score, rel_tol=1.0e-12, abs_tol=1.0e-12
	):
		raise ValueError(f'{label} history best validation score mismatch')


def _validate_downstream_contract(
	identity: Mapping[str, object],
	*,
	expected: Mapping[str, Mapping[str, object]],
	label: str,
) -> None:
	for block_name, expected_block in expected.items():
		block = _required_mapping(identity, block_name, f'{label} identity')
		for key, expected_value in expected_block.items():
			if block.get(key) != expected_value:
				raise ValueError(
					f'{label} {block_name}.{key} differs from the configured '
					'Volve downstream contract'
				)


def _metric_values_and_support(
	payload: Mapping[str, object],
	*,
	label: str,
) -> tuple[dict[str, float], dict[str, object]]:
	macro = _required_mapping(payload, 'macro', label)
	values = {
		PRIMARY_METRIC: _finite_number(
			payload.get(PRIMARY_METRIC), f'{label} {PRIMARY_METRIC}'
		),
		WITHIN2_PRIMARY_METRIC: _finite_number(
			payload.get(WITHIN2_PRIMARY_METRIC),
			f'{label} {WITHIN2_PRIMARY_METRIC}',
		),
		MACRO_WITHIN_1_METRIC: _finite_number(
			macro.get('within_1'), f'{label} macro.within_1'
		),
		MACRO_WITHIN_4_METRIC: _finite_number(
			macro.get('within_4'), f'{label} macro.within_4'
		),
		ORDER_VIOLATION_METRIC: _finite_number(
			payload.get(ORDER_VIOLATION_METRIC),
			f'{label} {ORDER_VIOLATION_METRIC}',
		),
	}
	per_horizon = _required_mapping(payload, 'per_horizon', label)
	if set(per_horizon) != set(HORIZON_NAMES):
		raise ValueError(
			f'{label} per_horizon keys must equal {HORIZON_NAMES!r}'
		)
	horizon_support: dict[str, dict[str, int]] = {}
	for horizon_name in HORIZON_NAMES:
		horizon = _required_mapping(per_horizon, horizon_name, f'{label} per_horizon')
		values[f'{horizon_name}_mae_samples'] = _finite_number(
			horizon.get('mae_samples'),
			f'{label} {horizon_name} mae_samples',
		)
		horizon_support[horizon_name] = {
			key: _nonnegative_int(
				horizon.get(key),
				f'{label} {horizon_name} {key}',
			)
			for key in ('count', 'predicted_count', 'missing_prediction_count')
		}
	coverage = _required_mapping(payload, 'coverage', label)
	coverage_support = {
		'eligible_count': _nonnegative_int(
			coverage.get('eligible_count'), f'{label} eligible_count'
		),
		'predicted_count': _nonnegative_int(
			coverage.get('predicted_count'), f'{label} predicted_count'
		),
		'fraction': _finite_number(
			coverage.get('fraction'), f'{label} coverage fraction'
		),
		'missing_prediction_count': _nonnegative_int(
			payload.get('missing_prediction_count'),
			f'{label} missing_prediction_count',
		),
	}
	_validate_metric_consistency(
		values=values,
		horizon_support=horizon_support,
		coverage_support=coverage_support,
		label=label,
	)
	return values, {'coverage': coverage_support, 'per_horizon': horizon_support}


def _validate_metric_consistency(
	*,
	values: Mapping[str, float],
	horizon_support: Mapping[str, Mapping[str, int]],
	coverage_support: Mapping[str, int | float],
	label: str,
) -> None:
	eligible_count = int(coverage_support['eligible_count'])
	predicted_count = int(coverage_support['predicted_count'])
	missing_count = int(coverage_support['missing_prediction_count'])
	if eligible_count <= 0:
		raise ValueError(f'{label} must have positive evaluation coverage')
	if predicted_count > eligible_count or missing_count != (
		eligible_count - predicted_count
	):
		raise ValueError(f'{label} prediction coverage counts are inconsistent')
	expected_fraction = predicted_count / eligible_count
	if not math.isclose(
		float(coverage_support['fraction']),
		expected_fraction,
		rel_tol=1.0e-12,
		abs_tol=1.0e-12,
	):
		raise ValueError(f'{label} coverage fraction is inconsistent with counts')
	if sum(item['count'] for item in horizon_support.values()) != eligible_count:
		raise ValueError(f'{label} per-horizon counts do not match coverage')
	if sum(item['predicted_count'] for item in horizon_support.values()) != (
		predicted_count
	):
		raise ValueError(
			f'{label} per-horizon predicted counts do not match coverage'
		)
	for horizon_name, item in horizon_support.items():
		if item['count'] <= 0 or item['predicted_count'] + item[
			'missing_prediction_count'
		] != item['count']:
			raise ValueError(
				f'{label} {horizon_name} observation counts are inconsistent'
			)
	expected_macro = statistics.fmean(
		values[f'{horizon_name}_mae_samples'] for horizon_name in HORIZON_NAMES
	)
	if not math.isclose(
		values[PRIMARY_METRIC],
		expected_macro,
		rel_tol=1.0e-12,
		abs_tol=1.0e-12,
	):
		raise ValueError(f'{label} macro MAE does not match per-horizon MAE')


def _validate_observation_counts(
	identity: Mapping[str, object],
	*,
	validation_support: Mapping[str, object],
	primary_support: Mapping[str, object],
	secondary_support: Mapping[str, object],
	label: str,
) -> None:
	effective = _required_mapping(
		identity,
		'effective_model_valid_observation_counts',
		f'{label} identity',
	)
	for identity_key, support in (
		('validation', validation_support),
		('test_primary_common', primary_support),
		('test_secondary_per_horizon', secondary_support),
	):
		expected = _required_mapping(effective, identity_key, f'{label} counts')
		horizons = _required_mapping(support, 'per_horizon', f'{label} support')
		for horizon_name in HORIZON_NAMES:
			recorded = _required_mapping(
				horizons, horizon_name, f'{label} support horizons'
			).get('count')
			if expected.get(horizon_name) != recorded:
				raise ValueError(
					f'{label} {identity_key} {horizon_name} observation count '
					'does not match evaluated support'
				)


def _validate_cross_job_identity(rows: list[dict[str, object]]) -> None:
	valid_tokens = {str(row['valid_tokens_sha256']) for row in rows}
	if len(valid_tokens) != 1:
		raise ValueError('valid-token identity must be shared by all 75 jobs')
	by_model: dict[str, list[dict[str, object]]] = {}
	by_cell: dict[tuple[str, str], list[dict[str, object]]] = {}
	for row in rows:
		by_model.setdefault(str(row['model_id']), []).append(row)
		by_cell.setdefault(
			(str(row['layout_id']), str(row['data_size'])), []
		).append(row)
	for model_id, group in by_model.items():
		for key in (
			'checkpoint_sha256',
			'embeddings_sha256',
			'embedding_metadata_sha256',
			'valid_tokens_sha256',
		):
			if len({str(row[key]) for row in group}) != 1:
				raise ValueError(
					f'{model_id} {key} differs between its {len(group)} jobs'
				)
	for (layout_id, data_size), group in by_cell.items():
		if len(group) != len(MODEL_IDS):
			raise ValueError(
				f'{layout_id}/{data_size} must contain exactly five model rows'
			)
		for key in ('_shared_run_identity', '_support_identity'):
			first = group[0][key]
			if any(row[key] != first for row in group[1:]):
				raise ValueError(
					f'{layout_id}/{data_size} {key.removeprefix("_")} '
					'differs between models'
				)


def _comparison_fieldnames(
	config: VolveHorizonFiveWayConfig,
) -> tuple[str, ...]:
	if config.checkpoint_selection == CHECKPOINT_SELECTION_VALIDATION_MAE:
		return COMPARISON_FIELDNAMES
	fields = list(COMPARISON_FIELDNAMES)
	insert_at = fields.index(PRIMARY_METRIC) + 1
	for metric in reversed(
		(
			WITHIN2_PRIMARY_METRIC,
			MACRO_WITHIN_1_METRIC,
			MACRO_WITHIN_4_METRIC,
			ORDER_VIOLATION_METRIC,
		)
	):
		fields.insert(insert_at, metric)
	return tuple(fields)


def _summary_metrics(
	config: VolveHorizonFiveWayConfig,
) -> tuple[str, ...]:
	if config.checkpoint_selection == CHECKPOINT_SELECTION_VALIDATION_MAE:
		return SUMMARY_METRICS
	return WITHIN2_SUMMARY_METRICS


def _paired_rows(
	rows: list[dict[str, object]],
	*,
	config: VolveHorizonFiveWayConfig,
) -> list[dict[str, object]]:
	by_cell = {
		(str(row['model_id']), str(row['layout_id']), str(row['data_size'])): row
		for row in rows
	}
	paired: list[dict[str, object]] = []
	for data_size in DATA_SIZE_PREFIX:
		for layout_id in LAYOUT_IDS:
			for comparison_id, left_model, right_model in PAIRED_COMPARISONS:
				left = by_cell[left_model, layout_id, data_size]
				right = by_cell[right_model, layout_id, data_size]
				for metric in _summary_metrics(config):
					left_value = float(left[metric])
					right_value = float(right[metric])
					if metric in {
						WITHIN2_PRIMARY_METRIC,
						MACRO_WITHIN_1_METRIC,
						MACRO_WITHIN_4_METRIC,
					}:
						delta = right_value - left_value
					else:
						delta = left_value - right_value
					paired.append(
						{
							'data_size': data_size,
							'layout_id': layout_id,
							'comparison_id': comparison_id,
							'metric': metric,
							'left_model': left_model,
							'right_model': right_model,
							'left_value': left_value,
							'right_value': right_value,
							'delta': delta,
						}
					)
	return paired


def _by_size_rows(
	paired: list[dict[str, object]],
	*,
	config: VolveHorizonFiveWayConfig,
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for data_size in DATA_SIZE_PREFIX:
		for comparison_id, _, _ in PAIRED_COMPARISONS:
			for metric in _summary_metrics(config):
				deltas = [
					float(row['delta'])
					for row in paired
					if row['data_size'] == data_size
					and row['comparison_id'] == comparison_id
					and row['metric'] == metric
				]
				if len(deltas) != len(LAYOUT_IDS):
					raise ValueError(
						f'{data_size}/{comparison_id}/{metric} must aggregate '
						f'exactly {len(LAYOUT_IDS)} layouts'
					)
				rows.append(
					{
						'data_size': data_size,
						'comparison_id': comparison_id,
						'metric': metric,
						'n_layouts': len(deltas),
						'mean': statistics.fmean(deltas),
						'median': statistics.median(deltas),
						'sample_std': statistics.stdev(deltas),
						'min': min(deltas),
						'max': max(deltas),
						'positive_count': sum(delta > 0.0 for delta in deltas),
						'zero_count': sum(delta == 0.0 for delta in deltas),
						'negative_count': sum(delta < 0.0 for delta in deltas),
					}
				)
	return rows


def _summary_payload(
	config: VolveHorizonFiveWayConfig,
	comparison_rows: list[dict[str, object]],
	by_size: list[dict[str, object]],
) -> dict[str, object]:
	summaries: dict[str, dict[str, dict[str, object]]] = {}
	for row in by_size:
		summaries.setdefault(str(row['data_size']), {}).setdefault(
			str(row['comparison_id']), {}
		)[str(row['metric'])] = {
			key: row[key]
			for key in (
				'n_layouts',
				'mean',
				'median',
				'sample_std',
				'min',
				'max',
				'positive_count',
				'zero_count',
				'negative_count',
			)
		}
	within2 = (
		config.checkpoint_selection
		== CHECKPOINT_SELECTION_VALIDATION_WITHIN_2
	)
	return {
		'schema_version': 1,
		'summary_name': (
			'volve_horizon_mae_local_bt_hmm_five_way_within2_v1'
			if within2
			else 'volve_horizon_mae_local_bt_hmm_five_way_v1'
		),
		'primary_metric': WITHIN2_PRIMARY_METRIC if within2 else PRIMARY_METRIC,
		'models': list(config.model_ids),
		'job_count': EXPECTED_JOB_COUNT,
		'statistical_unit': 'layout_id',
		'delta_definition': (
			'right_minus_left_for_within_metrics; '
			'left_minus_right_for_mae_and_order_violation'
			if within2
			else 'left_mae_minus_right_mae'
		),
		'positive_delta_interpretation': (
			'right_model_is_better_for_every_metric'
			if within2
			else 'right_model_has_lower_mae'
		),
		'comparison': comparison_rows,
		'by_size': summaries,
	}


def _summary_markdown(
	config: VolveHorizonFiveWayConfig,
	by_size: list[dict[str, object]],
) -> str:
	within2 = (
		config.checkpoint_selection
		== CHECKPOINT_SELECTION_VALIDATION_WITHIN_2
	)
	primary_metric = WITHIN2_PRIMARY_METRIC if within2 else PRIMARY_METRIC
	lines = [
		'# Volve horizon five-way summary',
		'',
		*(
			[
				(
					'Delta/improvement convention: `right - left` for within-1, '
					'within-2, and within-4; `left - right` for MAE and '
					'adjacent-order violation rate. A positive value always means the '
					'right-hand model is better.'
				)
			]
			if within2
			else [
				(
					'Delta convention: `left_MAE - right_MAE`. A positive delta means '
					'the right-hand model has lower MAE.'
				)
			]
		),
		'',
		(
			f'Primary metric: `{primary_metric}` on the common test support; '
			'paired unit is `layout_id` and each supervision size is summarized '
			'separately.'
		),
	]
	for data_size in DATA_SIZE_PREFIX:
		lines.extend(
			(
				'',
				f'## {data_size}',
				'',
				'| comparison | n | mean | median | sample std | min | max | +/0/- |',
				'|---|---:|---:|---:|---:|---:|---:|---|',
			)
		)
		for row in by_size:
			if row['data_size'] != data_size or row['metric'] != primary_metric:
				continue
			lines.append(
				f'| {row["comparison_id"]} | {row["n_layouts"]} '
				f'| {row["mean"]:.6f} | {row["median"]:.6f} '
				f'| {row["sample_std"]:.6f} | {row["min"]:.6f} '
				f'| {row["max"]:.6f} | {row["positive_count"]}/'
				f'{row["zero_count"]}/{row["negative_count"]} |'
			)
	lines.append('')
	return '\n'.join(lines)


def _csv_text(
	fieldnames: tuple[str, ...],
	rows: list[dict[str, object]],
) -> str:
	buffer = io.StringIO()
	writer = csv.DictWriter(buffer, fieldnames=fieldnames)
	writer.writeheader()
	for row in rows:
		writer.writerow({key: row[key] for key in fieldnames})
	return buffer.getvalue()


def _read_json(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'missing Volve five-way artifact: {path}')
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'{path} must contain a JSON object')
	return payload


def _read_history(path: Path) -> list[Mapping[str, object]]:
	if not path.is_file():
		raise FileNotFoundError(f'missing Volve five-way artifact: {path}')
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, list) or any(
		not isinstance(row, Mapping) for row in payload
	):
		raise TypeError(f'{path} must contain a list of history mappings')
	return payload


def _json_normalized(value: object) -> object:
	'''Normalize tuple/list distinctions exactly as the metrics JSON writer does.'''
	return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _required_mapping(
	value: Mapping[str, object],
	key: str,
	label: str,
) -> Mapping[str, object]:
	item = value.get(key)
	if not isinstance(item, Mapping):
		raise TypeError(f'{label} {key} must be a mapping')
	return item


def _same_path_value(value: object, expected: Path) -> bool:
	return isinstance(value, str) and bool(value) and (
		Path(value).resolve(strict=False) == expected.resolve(strict=False)
	)


def _sha256(value: object, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _nonnegative_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise ValueError(f'{label} must be a nonnegative integer')
	return value


def _finite_number(value: object, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{label} must be numeric')
	resolved = float(value)
	if not math.isfinite(resolved):
		raise ValueError(f'{label} must be finite')
	return resolved


def _reject_nonfinite_numbers(value: object, *, label: str) -> None:
	if isinstance(value, bool):
		return
	if isinstance(value, int | float):
		if not math.isfinite(float(value)):
			raise ValueError(f'{label} contains a non-finite numeric field')
		return
	if isinstance(value, Mapping):
		for key, item in value.items():
			_reject_nonfinite_numbers(item, label=f'{label}.{key}')
	elif isinstance(value, list):
		for index, item in enumerate(value):
			_reject_nonfinite_numbers(item, label=f'{label}[{index}]')


__all__ = [
	'BY_SIZE_FIELDNAMES',
	'COMPARISON_CSV_NAME',
	'COMPARISON_FIELDNAMES',
	'EXPECTED_JOB_COUNT',
	'EXPECTED_METRICS_ARTIFACT_TYPE',
	'MACRO_WITHIN_1_METRIC',
	'MACRO_WITHIN_4_METRIC',
	'MODEL_IDS',
	'ORDER_VIOLATION_METRIC',
	'PAIRED_COMPARISONS',
	'PAIRED_DELTAS_CSV_NAME',
	'PAIRED_FIELDNAMES',
	'PER_HORIZON_METRICS',
	'PRIMARY_METRIC',
	'SUMMARY_BY_SIZE_CSV_NAME',
	'SUMMARY_JSON_NAME',
	'SUMMARY_MD_NAME',
	'SUMMARY_METRICS',
	'SUMMARY_OUTPUT_NAMES',
	'WITHIN2_PRIMARY_METRIC',
	'WITHIN2_SUMMARY_METRICS',
	'inspect_volve_horizon_five_way_results',
	'summarize_volve_horizon_five_way',
]
