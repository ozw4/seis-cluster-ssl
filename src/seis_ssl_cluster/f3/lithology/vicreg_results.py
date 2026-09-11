"""VICReg screening, extension, and combined F3 benchmark results."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_five_way import FIVE_WAY_MODEL_IDS
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	FIXED_DECODER_CONTRACT,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.five_way_results import (
	EXPECTED_AGGREGATION_UNIT,
	PAIRED_COMPARISONS,
	SUMMARY_METRICS,
	aggregate_paired_rows_by_size,
	build_paired_rows,
	inspect_f3_lithology_five_way_results,
	read_f3_lithology_job_evidence,
	write_atomic_summary,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	FIVE_WAY_TILE_SETTINGS,
	resolve_f3_lithology_five_way_job,
)
from seis_ssl_cluster.f3.lithology.vicreg_runner import (
	plan_f3_vicreg_extension_jobs,
	plan_f3_vicreg_screening_jobs,
	resolve_f3_vicreg_extension_job,
	resolve_f3_vicreg_screening_job,
)
from seis_ssl_cluster.f3.lithology.vicreg_sources import (
	EXTENSION_MODEL_IDS,
	SCREENING_DATA_SIZE,
	SCREENING_MODEL_IDS,
	VICREG_GATE_FAIL,
	VICREG_GATE_PASS,
	audit_f3_vicreg_screening_source,
	audit_f3_vicreg_sources,
	f3_vicreg_screening_gate_from_mapping,
)

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.config.f3_lithology_five_way import (
		F3FiveWayConfig,
		F3FiveWayModelSource,
	)
	from seis_ssl_cluster.f3.lithology.five_way_runner import F3FiveWayJob
	from seis_ssl_cluster.f3.lithology.vicreg_sources import (
		F3VICRegExtensionConfig,
	)

SEVEN_WAY_MODEL_IDS = (*FIVE_WAY_MODEL_IDS, *EXTENSION_MODEL_IDS)
PRIMARY_METRIC = 'macro_f1'

COMPARISON_CSV_NAME = 'comparison.csv'
PAIRED_DELTAS_CSV_NAME = 'paired_deltas.csv'
SUMMARY_BY_SIZE_CSV_NAME = 'summary_by_size.csv'
SUMMARY_JSON_NAME = 'summary.json'
SUMMARY_MD_NAME = 'summary.md'
SCREENING_SUMMARY_OUTPUT_NAMES = (
	COMPARISON_CSV_NAME,
	PAIRED_DELTAS_CSV_NAME,
	SUMMARY_JSON_NAME,
	SUMMARY_MD_NAME,
)
BENCHMARK_SUMMARY_OUTPUT_NAMES = (
	COMPARISON_CSV_NAME,
	PAIRED_DELTAS_CSV_NAME,
	SUMMARY_BY_SIZE_CSV_NAME,
	SUMMARY_JSON_NAME,
	SUMMARY_MD_NAME,
)

EXTENSION_PAIRED_COMPARISONS = (
	('local_vicreg_hmm_k6_minus_local_vicreg', 'local_vicreg_hmm_k6', 'local_vicreg'),
	('local_vicreg_minus_random', 'local_vicreg', 'random'),
	('local_vicreg_hmm_k6_minus_random', 'local_vicreg_hmm_k6', 'random'),
)
SEVEN_WAY_PAIRED_COMPARISONS = (
	*PAIRED_COMPARISONS,
	*EXTENSION_PAIRED_COMPARISONS,
	(
		'local_vicreg_minus_local_barlow_twins',
		'local_vicreg',
		'local_barlow_twins',
	),
	(
		'local_vicreg_hmm_k6_minus_local_barlow_twins_hmm_k6',
		'local_vicreg_hmm_k6',
		'local_barlow_twins_hmm_k6',
	),
	('local_vicreg_minus_mae', 'local_vicreg', 'mae'),
	('local_vicreg_hmm_k6_minus_mae_hmm_k6', 'local_vicreg_hmm_k6', 'mae_hmm_k6'),
)

COMPARISON_FIELDNAMES = (
	'model_id',
	'layout_id',
	'data_size',
	'checkpoint_path',
	'encoder_checkpoint_sha256',
	'embeddings_dir',
	'embeddings_sha256',
	'embedding_metadata_sha256',
	'valid_tokens_sha256',
	'decoder_checkpoint_sha256',
	'decoder_initial_state_sha256',
	'supervision_identity',
	'validation_identity',
	'macro_f1',
	'mean_iou',
	'balanced_accuracy',
	'weighted_f1',
	'validation_voxel_count',
	'metrics_path',
	'metrics_sha256',
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
	'sample_std',
	'median',
	'min',
	'max',
	'positive_count',
	'zero_count',
	'negative_count',
)


def assert_f3_vicreg_full_benchmark_ready(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Require a passing screen and complete, audited two-arm sources."""
	try:
		return _full_benchmark_readiness(config, canonical)
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise RuntimeError(f'FULL_BENCHMARK_BLOCKED: {error}') from error


def _full_benchmark_readiness(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	summary_path = config.screening_outputs.summary_root / SUMMARY_JSON_NAME
	summary = _read_json(summary_path, label='VICReg screening summary')
	gate = f3_vicreg_screening_gate_from_mapping(summary, path=summary_path)
	if gate != VICREG_GATE_PASS:
		raise ValueError(f'screening gate is {gate!r}')
	sources = audit_f3_vicreg_sources(config, canonical)
	rows, _paired, recomputed_gate = _screening_material(
		config,
		canonical,
		source_report=sources,
	)
	if recomputed_gate['gate_status'] != gate:
		raise ValueError(
			'screening summary gate does not match current ten-cell evidence'
		)
	recorded_evidence = summary.get('evidence_sha256')
	current_evidence = _screening_evidence_sha256(rows)
	if recorded_evidence != current_evidence:
		raise ValueError(
			'screening summary is not bound to the current source/job evidence'
		)
	return sources


def inspect_f3_vicreg_screening_results(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit the exact ten logical medium cells without writing summaries."""
	rows, paired, gate = _screening_material(config, canonical)
	return {
		'complete_jobs': len(rows),
		'paired_layouts': len(paired),
		'model_order': list(SCREENING_MODEL_IDS),
		'data_size': SCREENING_DATA_SIZE,
		'gate_status': gate['gate_status'],
	}


def summarize_f3_vicreg_screening(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Atomically publish raw values, paired deltas, and the baseline gate."""
	rows, paired, gate = _screening_material(config, canonical)
	payload = {
		'schema_version': 1,
		'suite': 'local_vicreg_screen_v1',
		'models': list(SCREENING_MODEL_IDS),
		'data_size': SCREENING_DATA_SIZE,
		'primary_metric': PRIMARY_METRIC,
		'aggregation_unit': EXPECTED_AGGREGATION_UNIT,
		'statistical_unit': 'layout_id',
		'job_count': len(rows),
		'paired_layout_count': len(paired),
		'evidence_sha256': _screening_evidence_sha256(rows),
		'layouts': _screening_layout_evidence(rows, paired),
		**gate,
	}
	outputs = {
		COMPARISON_CSV_NAME: _csv_text(COMPARISON_FIELDNAMES, rows),
		PAIRED_DELTAS_CSV_NAME: _csv_text(PAIRED_FIELDNAMES, paired),
		SUMMARY_JSON_NAME: json.dumps(payload, indent=2, sort_keys=True) + '\n',
		SUMMARY_MD_NAME: _screening_markdown(payload, paired),
	}
	write_atomic_summary(config.screening_outputs.summary_root, outputs)
	return {
		'complete_jobs': len(rows),
		'gate_status': gate['gate_status'],
		'summary_root': str(config.screening_outputs.summary_root),
		'outputs': [
			str(config.screening_outputs.summary_root / name)
			for name in SCREENING_SUMMARY_OUTPUT_NAMES
		],
	}


def inspect_f3_vicreg_extension_results(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit all 30 new jobs and the canonical random comparison rows read-only."""
	rows, paired, _by_size = _extension_material(config, canonical)
	return {
		'complete_jobs': len(rows),
		'model_order': list(EXTENSION_MODEL_IDS),
		'paired_delta_rows': len(paired),
	}


def summarize_f3_vicreg_extension(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Atomically summarize only the 30-job two-arm extension."""
	rows, paired, by_size = _extension_material(config, canonical)
	payload = _benchmark_summary_payload(
		suite='local_vicreg_extension_v1',
		models=EXTENSION_MODEL_IDS,
		job_count=len(rows),
		by_size=by_size,
	)
	outputs = _benchmark_outputs(
		rows=rows,
		paired=paired,
		by_size=by_size,
		payload=payload,
		title='F3 lithology Local VICReg extension summary',
	)
	write_atomic_summary(config.extension_outputs.summary_root, outputs)
	return {
		'complete_jobs': len(rows),
		'summary_root': str(config.extension_outputs.summary_root),
		'outputs': [
			str(config.extension_outputs.summary_root / name)
			for name in BENCHMARK_SUMMARY_OUTPUT_NAMES
		],
	}


def inspect_f3_vicreg_combined_results(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Read and identity-check the existing 75 plus new 30 jobs."""
	rows, paired, _by_size = _combined_material(config, canonical)
	return {
		'complete_jobs': len(rows),
		'existing_jobs': 75,
		'extension_jobs': 30,
		'model_order': list(SEVEN_WAY_MODEL_IDS),
		'paired_delta_rows': len(paired),
	}


def summarize_f3_vicreg_combined(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Publish a separate 105-row report without mutating the existing 75 jobs."""
	rows, paired, by_size = _combined_material(config, canonical)
	payload = {
		**_benchmark_summary_payload(
			suite='f3_lithology_seven_way_v1',
			models=SEVEN_WAY_MODEL_IDS,
			job_count=len(rows),
			by_size=by_size,
		),
		'existing_five_way_jobs': 75,
		'new_extension_jobs': 30,
		'canonical_runs_root': str(canonical.runs_root),
		'extension_runs_root': str(config.extension_outputs.runs_root),
	}
	outputs = _benchmark_outputs(
		rows=rows,
		paired=paired,
		by_size=by_size,
		payload=payload,
		title='F3 lithology seven-way combined summary',
	)
	write_atomic_summary(config.combined_summary_root, outputs)
	return {
		'complete_jobs': len(rows),
		'existing_jobs': 75,
		'extension_jobs': 30,
		'summary_root': str(config.combined_summary_root),
		'outputs': [
			str(config.combined_summary_root / name)
			for name in BENCHMARK_SUMMARY_OUTPUT_NAMES
		],
	}


def _screening_material(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
	*,
	source_report: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
	report = (
		audit_f3_vicreg_screening_source(config, canonical)
		if source_report is None
		else source_report
	)
	_reject_unexpected_run_directories(
		config.screening_outputs.runs_root,
		model_ids=(config.screening_model.model_id,),
		layouts=LAYOUT_IDS,
		sizes=(SCREENING_DATA_SIZE,),
	)
	rows = []
	for model_id, layout_id, size in plan_f3_vicreg_screening_jobs():
		job = resolve_f3_vicreg_screening_job(
			config, canonical, model=model_id, layout=layout_id, size=size
		)
		rows.append(_read_job_row(canonical, job.model, job))
	_validate_matrix(
		rows,
		models=SCREENING_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=(SCREENING_DATA_SIZE,),
		report=report,
	)
	paired = build_paired_rows(
		rows,
		comparisons=(('local_vicreg_100_minus_random', 'local_vicreg_100', 'random'),),
		metrics=(PRIMARY_METRIC,),
		data_sizes=(SCREENING_DATA_SIZE,),
		layout_ids=LAYOUT_IDS,
	)
	deltas = [float(row['delta']) for row in paired]
	if len(deltas) != len(LAYOUT_IDS):
		raise ValueError('VICReg screening must contain exactly five paired layouts')
	mean = statistics.fmean(deltas)
	median = statistics.median(deltas)
	wins = sum(delta > 0.0 for delta in deltas)
	passed = mean > 0.0 and median > 0.0 and wins >= 3
	gate = {
		'gate_status': VICREG_GATE_PASS if passed else VICREG_GATE_FAIL,
		'gate': {
			'mean_paired_delta_gt': 0.0,
			'median_paired_delta_gt': 0.0,
			'minimum_wins': 3,
			'layout_count': len(LAYOUT_IDS),
		},
		'mean_paired_delta': mean,
		'median_paired_delta': median,
		'wins': wins,
		'losses': sum(delta < 0.0 for delta in deltas),
		'ties': sum(delta == 0.0 for delta in deltas),
	}
	return rows, paired, gate


def _screening_evidence_sha256(rows: Sequence[Mapping[str, object]]) -> str:
	fields = (
		'model_id',
		'layout_id',
		'data_size',
		'encoder_checkpoint_sha256',
		'embeddings_sha256',
		'embedding_metadata_sha256',
		'valid_tokens_sha256',
		'decoder_checkpoint_sha256',
		'decoder_initial_state_sha256',
		'supervision_identity',
		'validation_identity',
		'metrics_sha256',
	)
	identity = [
		{field: row[field] for field in fields}
		for row in sorted(
			rows,
			key=lambda row: (
				str(row['layout_id']),
				str(row['model_id']),
			),
		)
	]
	return hashlib.sha256(
		json.dumps(
			identity,
			sort_keys=True,
			separators=(',', ':'),
			allow_nan=False,
		).encode('utf-8')
	).hexdigest()


def _screening_layout_evidence(
	rows: Sequence[Mapping[str, object]],
	paired: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	by_cell = {(str(row['model_id']), str(row['layout_id'])): row for row in rows}
	paired_by_layout = {str(row['layout_id']): row for row in paired}
	result = []
	for layout_id in LAYOUT_IDS:
		vicreg = by_cell['local_vicreg_100', layout_id]
		random = by_cell['random', layout_id]
		delta = paired_by_layout[layout_id]
		result.append(
			{
				'layout_id': layout_id,
				'local_vicreg_macro_f1': vicreg['macro_f1'],
				'random_macro_f1': random['macro_f1'],
				'delta_local_vicreg_minus_random': delta['delta'],
				'local_vicreg_checkpoint_sha256': vicreg['encoder_checkpoint_sha256'],
				'random_checkpoint_sha256': random['encoder_checkpoint_sha256'],
				'local_vicreg_embedding_sha256': vicreg['embeddings_sha256'],
				'random_embedding_sha256': random['embeddings_sha256'],
				'supervision_identity': vicreg['supervision_identity'],
				'validation_mask_sha256': vicreg['validation_identity'],
				'decoder_initial_state_sha256': vicreg['decoder_initial_state_sha256'],
				'local_vicreg_metrics_sha256': vicreg['metrics_sha256'],
				'random_metrics_sha256': random['metrics_sha256'],
			}
		)
	return result


def _extension_material(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> tuple[
	list[dict[str, object]],
	list[dict[str, object]],
	list[dict[str, object]],
]:
	report = assert_f3_vicreg_full_benchmark_ready(config, canonical)
	_reject_unexpected_run_directories(
		config.extension_outputs.runs_root,
		model_ids=EXTENSION_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
	)
	rows = _extension_rows(config, canonical)
	_validate_matrix(
		rows,
		models=EXTENSION_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
		report=report,
	)
	random_rows = _canonical_rows(canonical, model_ids=('random',))
	_validate_matrix(
		random_rows,
		models=('random',),
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
		report=report,
	)
	paired_source = [*rows, *random_rows]
	_validate_shared_condition_identity(paired_source)
	paired = build_paired_rows(
		paired_source,
		comparisons=EXTENSION_PAIRED_COMPARISONS,
		metrics=SUMMARY_METRICS,
		data_sizes=DATA_SIZES,
		layout_ids=LAYOUT_IDS,
	)
	by_size = aggregate_paired_rows_by_size(
		paired,
		comparisons=EXTENSION_PAIRED_COMPARISONS,
		metrics=SUMMARY_METRICS,
		data_sizes=DATA_SIZES,
		layout_ids=LAYOUT_IDS,
	)
	return rows, paired, by_size


def _combined_material(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> tuple[
	list[dict[str, object]],
	list[dict[str, object]],
	list[dict[str, object]],
]:
	report = assert_f3_vicreg_full_benchmark_ready(config, canonical)
	canonical_inspection = inspect_f3_lithology_five_way_results(canonical)
	if canonical_inspection.get('complete_jobs') != 75:
		raise ValueError('canonical five-way inspection did not return exactly 75 jobs')
	_reject_unexpected_run_directories(
		config.extension_outputs.runs_root,
		model_ids=EXTENSION_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
	)
	canonical_rows = _canonical_rows(canonical, model_ids=FIVE_WAY_MODEL_IDS)
	extension_rows = _extension_rows(config, canonical)
	rows = [*canonical_rows, *extension_rows]
	_validate_matrix(
		rows,
		models=SEVEN_WAY_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
		report=report,
	)
	paired = build_paired_rows(
		rows,
		comparisons=SEVEN_WAY_PAIRED_COMPARISONS,
		metrics=SUMMARY_METRICS,
		data_sizes=DATA_SIZES,
		layout_ids=LAYOUT_IDS,
	)
	by_size = aggregate_paired_rows_by_size(
		paired,
		comparisons=SEVEN_WAY_PAIRED_COMPARISONS,
		metrics=SUMMARY_METRICS,
		data_sizes=DATA_SIZES,
		layout_ids=LAYOUT_IDS,
	)
	return rows, paired, by_size


def _canonical_rows(
	canonical: F3FiveWayConfig,
	*,
	model_ids: Sequence[str],
) -> list[dict[str, object]]:
	rows = []
	for model_id in model_ids:
		model = canonical.model_by_id(model_id)
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				job = resolve_f3_lithology_five_way_job(
					canonical,
					model=model_id,
					layout=layout_id,
					size=data_size,
				)
				rows.append(_read_job_row(canonical, model, job))
	return rows


def _extension_rows(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> list[dict[str, object]]:
	rows = []
	for model_id, layout_id, data_size in plan_f3_vicreg_extension_jobs():
		job = resolve_f3_vicreg_extension_job(
			config,
			canonical,
			model=model_id,
			layout=layout_id,
			size=data_size,
		)
		rows.append(_read_job_row(canonical, job.model, job))
	return rows


def _read_job_row(
	canonical: F3FiveWayConfig,
	model: F3FiveWayModelSource,
	job: F3FiveWayJob,
) -> dict[str, object]:
	metrics = _read_json(job.metrics_path, label='completed evaluation metrics')
	for metric in SUMMARY_METRICS:
		value = metrics.get(metric)
		if (
			isinstance(value, bool)
			or not isinstance(value, int | float)
			or not math.isfinite(float(value))
		):
			raise ValueError(
				f'{model.model_id}/{job.layout_id}/{job.data_size} metric '
				f'{metric} must be finite numeric'
			)
	evidence = read_f3_lithology_job_evidence(
		canonical,
		model=model,
		layout_id=job.layout_id,
		data_size=job.data_size,
		job_dir=job.output_dir,
	)
	resolved_decoder_config = _read_json(
		job.decoder_dir / 'resolved_config.json', label='decoder resolved config'
	)
	_validate_decoder_contract(resolved_decoder_config, job=job)
	run_metadata = _read_json(
		job.decoder_dir / 'run_metadata.json', label='decoder run metadata'
	)
	initial_state = _sha256(
		run_metadata.get('initial_model_state_sha256'),
		label=(
			f'{model.model_id}/{job.layout_id}/{job.data_size} decoder initial state'
		),
	)
	return {
		'model_id': model.model_id,
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'checkpoint_path': str(model.checkpoint),
		'embeddings_dir': str(model.embeddings_dir),
		**evidence,
		'decoder_initial_state_sha256': initial_state,
		**{metric: float(metrics[metric]) for metric in SUMMARY_METRICS},
		'metrics_path': str(job.metrics_path),
		'metrics_sha256': file_sha256(job.metrics_path),
	}


def _validate_decoder_contract(
	resolved: Mapping[str, object], *, job: F3FiveWayJob
) -> None:
	decoder_expected = {
		key: FIXED_DECODER_CONTRACT[key]
		for key in (
			'spec',
			'embedding_dim',
			'class_count',
			'hidden_channels',
			'upsample_factors',
			'upsample_mode',
			'normalization',
		)
	}
	train_expected = {
		key: FIXED_DECODER_CONTRACT[key]
		for key in (
			'epochs',
			'batch_size',
			'learning_rate',
			'weight_decay',
			'class_weight',
			'seed',
			'amp',
			'gradient_clip_norm',
			'sampling_mode',
			'steps_per_epoch',
		)
	}
	_validate_resolved_mapping(
		resolved.get('decoder'), expected=decoder_expected, label='decoder'
	)
	_validate_resolved_mapping(
		resolved.get('tiles'), expected=FIVE_WAY_TILE_SETTINGS, label='tiles'
	)
	_validate_resolved_mapping(
		resolved.get('train'), expected=train_expected, label='train'
	)
	model = resolved.get('model')
	if not isinstance(model, Mapping) or model != {
		'tag': job.model.model_id,
		'freeze_encoder': True,
	}:
		raise ValueError('decoder resolved model identity does not match this job')


def _validate_resolved_mapping(
	value: object, *, expected: Mapping[str, object], label: str
) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'decoder resolved {label} must be a mapping')
	for key, expected_value in expected.items():
		normalized = _plain_json_value(expected_value)
		if value.get(key) != normalized:
			raise ValueError(
				f'decoder resolved {label}.{key} must equal {normalized!r}; '
				f'got {value.get(key)!r}'
			)


def _plain_json_value(value: object) -> object:
	if isinstance(value, tuple):
		return [_plain_json_value(item) for item in value]
	return value


def _validate_matrix(
	rows: list[dict[str, object]],
	*,
	models: Sequence[str],
	layouts: Sequence[str],
	sizes: Sequence[str],
	report: Mapping[str, object],
) -> None:
	expected = {
		(model_id, layout_id, data_size)
		for model_id in models
		for layout_id in layouts
		for data_size in sizes
	}
	actual = [
		(str(row['model_id']), str(row['layout_id']), str(row['data_size']))
		for row in rows
	]
	if len(actual) != len(set(actual)):
		raise ValueError('duplicate VICReg benchmark job evidence')
	if set(actual) != expected:
		missing = sorted(expected - set(actual))
		extra = sorted(set(actual) - expected)
		raise ValueError(
			'VICReg benchmark job matrix is not exact; '
			f'missing={missing!r}, extra={extra!r}'
		)
	for row in rows:
		_assert_row_current_source(
			row, _source_provenance(report, str(row['model_id']))
		)
	_validate_shared_condition_identity(rows)
	_validate_per_model_source_identity(rows)


def _validate_shared_condition_identity(rows: list[dict[str, object]]) -> None:
	by_condition: dict[tuple[str, str], list[dict[str, object]]] = {}
	for row in rows:
		key = (str(row['layout_id']), str(row['data_size']))
		by_condition.setdefault(key, []).append(row)
	for (layout_id, data_size), group in by_condition.items():
		for key in (
			'supervision_identity',
			'validation_identity',
			'validation_voxel_count',
			'_validation_tile_manifest_sha256',
			'decoder_initial_state_sha256',
		):
			values = {str(row[key]) for row in group}
			if len(values) != 1:
				raise ValueError(
					f'{layout_id}/{data_size} {key} differs between models'
				)
	valid_tokens = {str(row['valid_tokens_sha256']) for row in rows}
	if len(valid_tokens) != 1:
		raise ValueError('valid-token SHA must be shared by every compared model')


def _validate_per_model_source_identity(rows: list[dict[str, object]]) -> None:
	by_model: dict[str, list[dict[str, object]]] = {}
	for row in rows:
		by_model.setdefault(str(row['model_id']), []).append(row)
	for model_id, group in by_model.items():
		for key in (
			'encoder_checkpoint_sha256',
			'embeddings_sha256',
			'embedding_metadata_sha256',
			'valid_tokens_sha256',
		):
			values = {str(row[key]) for row in group}
			if len(values) != 1:
				raise ValueError(f'{model_id} {key} drifted between completed jobs')
	encoders: dict[str, str] = {}
	for model_id, group in by_model.items():
		sha256 = str(group[0]['encoder_checkpoint_sha256'])
		other = encoders.setdefault(sha256, model_id)
		if other != model_id:
			raise ValueError(
				'encoder checkpoint SHA must differ across compared models; '
				f'{other!r} and {model_id!r} match'
			)


def _source_provenance(
	report: Mapping[str, object], model_id: str
) -> Mapping[str, object]:
	if model_id == 'random':
		value = report.get('canonical_random')
		if not isinstance(value, Mapping):
			raise TypeError('VICReg source audit canonical_random must be a mapping')
		return value
	sources = report.get('sources')
	if not isinstance(sources, list):
		raise TypeError('VICReg source audit sources must be a list')
	for source in sources:
		if isinstance(source, Mapping) and source.get('candidate_id') == model_id:
			return source
	if model_id in FIVE_WAY_MODEL_IDS:
		canonical_sources = report.get('canonical_sources')
		if not isinstance(canonical_sources, list):
			raise TypeError('VICReg audit canonical_sources must be a list')
		for source in canonical_sources:
			if isinstance(source, Mapping) and source.get('model_id') == model_id:
				return source
		raise ValueError(f'VICReg audit has no canonical provenance for {model_id!r}')
	raise ValueError(f'VICReg source audit has no provenance for {model_id!r}')


def _assert_row_current_source(
	row: Mapping[str, object], provenance: Mapping[str, object]
) -> None:
	for row_key, source_key in (
		('encoder_checkpoint_sha256', 'checkpoint_sha256'),
		('embeddings_sha256', 'embeddings_sha256'),
		('embedding_metadata_sha256', 'embedding_metadata_sha256'),
		('valid_tokens_sha256', 'valid_tokens_sha256'),
	):
		if row.get(row_key) != provenance.get(source_key):
			raise ValueError(
				f'{row["model_id"]} completed job {row_key} does not match '
				'the current configured source'
			)


def _benchmark_summary_payload(
	*,
	suite: str,
	models: Sequence[str],
	job_count: int,
	by_size: list[dict[str, object]],
) -> dict[str, object]:
	grouped: dict[str, dict[str, dict[str, object]]] = {}
	for row in by_size:
		grouped.setdefault(str(row['data_size']), {}).setdefault(
			str(row['comparison_id']), {}
		)[str(row['metric'])] = {
			key: row[key]
			for key in (
				'n_layouts',
				'mean',
				'sample_std',
				'median',
				'min',
				'max',
				'positive_count',
				'zero_count',
				'negative_count',
			)
		}
	return {
		'schema_version': 1,
		'suite': suite,
		'models': list(models),
		'job_count': job_count,
		'primary_metric': PRIMARY_METRIC,
		'aggregation_unit': EXPECTED_AGGREGATION_UNIT,
		'statistical_unit': 'layout_id',
		'by_size': grouped,
	}


def _benchmark_outputs(
	*,
	rows: list[dict[str, object]],
	paired: list[dict[str, object]],
	by_size: list[dict[str, object]],
	payload: Mapping[str, object],
	title: str,
) -> dict[str, str]:
	return {
		COMPARISON_CSV_NAME: _csv_text(COMPARISON_FIELDNAMES, rows),
		PAIRED_DELTAS_CSV_NAME: _csv_text(PAIRED_FIELDNAMES, paired),
		SUMMARY_BY_SIZE_CSV_NAME: _csv_text(BY_SIZE_FIELDNAMES, by_size),
		SUMMARY_JSON_NAME: json.dumps(payload, indent=2, sort_keys=True) + '\n',
		SUMMARY_MD_NAME: _benchmark_markdown(title, by_size),
	}


def _screening_markdown(
	payload: Mapping[str, object], paired: list[dict[str, object]]
) -> str:
	lines = [
		'# F3 Local VICReg versus Random screening',
		'',
		f'Gate: `{payload["gate_status"]}`.',
		'',
		'| layout | Local VICReg | Random | delta |',
		'|---|---:|---:|---:|',
	]
	lines.extend(
		f'| {row["layout_id"]} | {row["left_value"]:.6f} '
		f'| {row["right_value"]:.6f} | {row["delta"]:.6f} |'
		for row in paired
	)
	lines.extend(
		(
			'',
			(
				f'Mean delta: `{payload["mean_paired_delta"]:.6f}`; '
				f'median: `{payload["median_paired_delta"]:.6f}`; '
				f'wins: `{payload["wins"]}/5`.'
			),
			'',
		)
	)
	return '\n'.join(lines)


def _benchmark_markdown(title: str, by_size: list[dict[str, object]]) -> str:
	lines = [
		f'# {title}',
		'',
		'Primary metric is `macro_f1`; paired unit is `layout_id` per size.',
		'',
		'| size | comparison | mean | median | sample std | +/0/- |',
		'|---|---|---:|---:|---:|---|',
	]
	for row in by_size:
		if row['metric'] != PRIMARY_METRIC:
			continue
		lines.append(
			f'| {row["data_size"]} | {row["comparison_id"]} '
			f'| {row["mean"]:.6f} | {row["median"]:.6f} '
			f'| {row["sample_std"]:.6f} '
			f'| {row["positive_count"]}/{row["zero_count"]}'
			f'/{row["negative_count"]} |'
		)
	lines.append('')
	return '\n'.join(lines)


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
	buffer = io.StringIO()
	writer = csv.DictWriter(buffer, fieldnames=fieldnames)
	writer.writeheader()
	for row in rows:
		writer.writerow({key: row[key] for key in fieldnames})
	return buffer.getvalue()


def _reject_unexpected_run_directories(
	runs_root: Path,
	*,
	model_ids: Sequence[str],
	layouts: Sequence[str],
	sizes: Sequence[str],
) -> None:
	if not runs_root.is_dir():
		return
	expected_models = {f'model={model_id}' for model_id in model_ids}
	for model_dir in sorted(runs_root.iterdir()):
		if not model_dir.is_dir() or model_dir.name not in expected_models:
			raise ValueError(f'unexpected VICReg run directory: {model_dir}')
		for layout_dir in sorted(model_dir.iterdir()):
			if not layout_dir.is_dir() or layout_dir.name not in {
				f'layout={layout}' for layout in layouts
			}:
				raise ValueError(f'unexpected VICReg run directory: {layout_dir}')
			for size_dir in sorted(layout_dir.iterdir()):
				if not size_dir.is_dir() or size_dir.name not in {
					f'size={size}' for size in sizes
				}:
					raise ValueError(f'unexpected VICReg run directory: {size_dir}')


def read_f3_vicreg_completed_job(
	canonical: F3FiveWayConfig,
	job: F3FiveWayJob,
	*,
	source_report: Mapping[str, object],
) -> dict[str, object]:
	"""Read one completed job and bind it to the currently audited source."""
	row = _read_job_row(canonical, job.model, job)
	_assert_row_current_source(
		row,
		_source_provenance(source_report, job.model.model_id),
	)
	return row


def _sha256(value: object, *, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} must contain JSON: {path}') from error
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return payload


__all__ = [
	'BENCHMARK_SUMMARY_OUTPUT_NAMES',
	'SCREENING_SUMMARY_OUTPUT_NAMES',
	'SEVEN_WAY_MODEL_IDS',
	'assert_f3_vicreg_full_benchmark_ready',
	'inspect_f3_vicreg_combined_results',
	'inspect_f3_vicreg_extension_results',
	'inspect_f3_vicreg_screening_results',
	'read_f3_vicreg_completed_job',
	'summarize_f3_vicreg_combined',
	'summarize_f3_vicreg_extension',
	'summarize_f3_vicreg_screening',
]
