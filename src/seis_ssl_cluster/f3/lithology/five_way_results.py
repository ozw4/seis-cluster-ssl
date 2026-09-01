"""Completeness audit and paired summary for the 75-job five-way matrix."""

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

from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	BEST_CHECKPOINT_NAME,
	DECODER_DIR_NAME,
	EVALUATION_DIR_NAME,
	FIVE_WAY_EVALUATION_POLICY,
	METRICS_NAME,
	PREDICTION_DIR_NAME,
	PREDICTION_METADATA_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	LAYOUT_METADATA_NAME,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_five_way import (
		F3FiveWayConfig,
		F3FiveWayModelSource,
	)

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
SUMMARY_METRICS = ('macro_f1', 'mean_iou', 'balanced_accuracy', 'weighted_f1')
PRIMARY_METRIC = 'macro_f1'
PAIRED_COMPARISONS = (
	('mae_hmm_k6_minus_mae', 'mae_hmm_k6', 'mae'),
	(
		'local_bt_hmm_k6_minus_local_bt',
		'local_barlow_twins_hmm_k6',
		'local_barlow_twins',
	),
	('local_bt_minus_mae', 'local_barlow_twins', 'mae'),
	(
		'local_bt_hmm_k6_minus_mae_hmm_k6',
		'local_barlow_twins_hmm_k6',
		'mae_hmm_k6',
	),
	('mae_minus_random', 'mae', 'random'),
	('mae_hmm_k6_minus_random', 'mae_hmm_k6', 'random'),
	('local_bt_minus_random', 'local_barlow_twins', 'random'),
	('local_bt_hmm_k6_minus_random', 'local_barlow_twins_hmm_k6', 'random'),
)
EXPECTED_AGGREGATION_UNIT = 'unique_validation_voxel'
SOURCE_IDENTITY_FIELDS = (
	'encoder_checkpoint_sha256',
	'embeddings_sha256',
	'embedding_metadata_sha256',
	'valid_tokens_sha256',
	'decoder_checkpoint_sha256',
)
PER_MODEL_IDENTITY_FIELDS = (
	'encoder_checkpoint_sha256',
	'embeddings_sha256',
	'embedding_metadata_sha256',
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
	'supervision_identity',
	'validation_identity',
	'macro_f1',
	'mean_iou',
	'balanced_accuracy',
	'weighted_f1',
	'validation_voxel_count',
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
	'sample_std',
	'median',
	'min',
	'max',
	'positive_count',
	'zero_count',
	'negative_count',
)


def inspect_f3_lithology_five_way_results(
	config: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit all 75 evaluation artifacts read-only; raise on any drift."""
	_condition_identities(config)
	_reject_unexpected_run_directories(config)
	cells = [
		(model_id, layout_id, data_size)
		for model_id in config.model_ids
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZES
	]
	missing = [
		str(
			_job_dir(config, model_id, layout_id, data_size)
			/ EVALUATION_DIR_NAME
			/ METRICS_NAME
		)
		for model_id, layout_id, data_size in cells
		if not (
			_job_dir(config, model_id, layout_id, data_size)
			/ EVALUATION_DIR_NAME
			/ METRICS_NAME
		).is_file()
	]
	if missing:
		raise FileNotFoundError(
			f'missing {len(missing)} of 75 five-way evaluations: {missing!r}'
		)
	rows = [
		_load_job_row(
			config,
			model_id=model_id,
			layout_id=layout_id,
			data_size=data_size,
		)
		for model_id, layout_id, data_size in cells
	]
	_validate_cross_job_identity(rows)
	return {
		'complete_jobs': len(rows),
		'model_order': list(config.model_ids),
		'rows': rows,
	}


def summarize_f3_lithology_five_way(
	config: F3FiveWayConfig,
) -> dict[str, object]:
	"""Write the five summary outputs after the full completeness audit."""
	report = inspect_f3_lithology_five_way_results(config)
	rows = report['rows']
	paired = _paired_rows(rows)
	by_size = _by_size_rows(paired)
	summary_payload = _summary_payload(config, by_size)
	outputs = {
		COMPARISON_CSV_NAME: _csv_text(COMPARISON_FIELDNAMES, rows),
		PAIRED_DELTAS_CSV_NAME: _csv_text(PAIRED_FIELDNAMES, paired),
		SUMMARY_BY_SIZE_CSV_NAME: _csv_text(BY_SIZE_FIELDNAMES, by_size),
		SUMMARY_JSON_NAME: json.dumps(summary_payload, indent=2, sort_keys=True)
		+ '\n',
		SUMMARY_MD_NAME: _summary_markdown(by_size),
	}
	summary_root = config.summary_root
	if summary_root.exists():
		raise FileExistsError(
			f'refusing to overwrite existing summary: {summary_root}'
		)
	summary_root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix=f'.{summary_root.name}.staging-', dir=summary_root.parent
		)
	)
	try:
		for name, text in outputs.items():
			(staging / name).write_text(text, encoding='utf-8')
		staging.replace(summary_root)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	return {
		'complete_jobs': report['complete_jobs'],
		'summary_root': str(summary_root),
		'outputs': [str(summary_root / name) for name in SUMMARY_OUTPUT_NAMES],
	}


def _job_dir(
	config: F3FiveWayConfig, model_id: str, layout_id: str, data_size: str
) -> Path:
	return (
		config.runs_root
		/ f'model={model_id}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)


def _condition_identities(
	config: F3FiveWayConfig,
) -> dict[tuple[str, str], dict[str, object]]:
	conditions: dict[tuple[str, str], dict[str, object]] = {}
	for layout_id in LAYOUT_IDS:
		for data_size in DATA_SIZES:
			conditions[layout_id, data_size] = _condition_identity(
				config, layout_id=layout_id, data_size=data_size
			)
	masks = {
		str(condition['validation_mask_sha256'])
		for condition in conditions.values()
	}
	if len(masks) != 1:
		raise ValueError(
			'validation mask identity must be shared by all 15 conditions'
		)
	return conditions


def _condition_identity(
	config: F3FiveWayConfig, *, layout_id: str, data_size: str
) -> dict[str, object]:
	condition_dir = (
		config.section_layout_dataset_root
		/ 'datasets'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ 'voxel_supervision'
	)
	metadata_path = condition_dir / LAYOUT_METADATA_NAME
	if not metadata_path.is_file():
		raise FileNotFoundError(
			f'missing section-layout condition metadata: {metadata_path}'
		)
	identity = _read_json(metadata_path).get('identity')
	if not isinstance(identity, Mapping):
		raise TypeError(f'{metadata_path} identity must be a mapping')
	if identity.get('layout_id') != layout_id:
		raise ValueError(f'{metadata_path} layout_id does not match {layout_id!r}')
	if identity.get('data_size') != data_size:
		raise ValueError(f'{metadata_path} data_size does not match {data_size!r}')
	return {
		'condition_dir': condition_dir,
		'validation_mask_sha256': identity.get('validation_mask_sha256'),
		'validation_voxel_count': identity.get('validation_voxel_count'),
	}


def _reject_unexpected_run_directories(config: F3FiveWayConfig) -> None:
	if not config.runs_root.is_dir():
		return
	expected_models = {f'model={model_id}' for model_id in config.model_ids}
	for entry in sorted(config.runs_root.iterdir()):
		if entry.name not in expected_models:
			raise ValueError(f'unexpected five-way run directory: {entry}')
		expected_layouts = {f'layout={layout_id}' for layout_id in LAYOUT_IDS}
		for layout_entry in sorted(entry.iterdir()):
			if layout_entry.name not in expected_layouts:
				raise ValueError(
					f'unexpected five-way run directory: {layout_entry}'
				)
			expected_sizes = {f'size={data_size}' for data_size in DATA_SIZES}
			for size_entry in sorted(layout_entry.iterdir()):
				if size_entry.name not in expected_sizes:
					raise ValueError(
						f'unexpected five-way run directory: {size_entry}'
					)


def _load_job_row(
	config: F3FiveWayConfig,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
) -> dict[str, object]:
	job_dir = _job_dir(config, model_id, layout_id, data_size)
	label = f'{model_id}/{layout_id}/{data_size}'
	metrics = _read_json(job_dir / EVALUATION_DIR_NAME / METRICS_NAME)
	for key in SUMMARY_METRICS:
		value = metrics.get(key)
		if not isinstance(value, int | float) or isinstance(value, bool):
			# A malformed metrics artifact is stale data, not a caller bug.
			raise ValueError(  # noqa: TRY004
				f'{label} metrics {key} must be numeric'
			)
		if not math.isfinite(float(value)):
			raise ValueError(f'{label} metrics {key} must be finite')
	model = config.model_by_id(model_id)
	evidence = read_f3_lithology_job_evidence(
		config,
		model=model,
		layout_id=layout_id,
		data_size=data_size,
		job_dir=job_dir,
	)
	return {
		'model_id': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'checkpoint_path': str(model.checkpoint),
		'embeddings_dir': str(model.embeddings_dir),
		**evidence,
		'macro_f1': float(metrics['macro_f1']),
		'mean_iou': float(metrics['mean_iou']),
		'balanced_accuracy': float(metrics['balanced_accuracy']),
		'weighted_f1': float(metrics['weighted_f1']),
		'metrics_path': str(job_dir / EVALUATION_DIR_NAME / METRICS_NAME),
	}


def read_f3_lithology_job_evidence(  # noqa: C901, PLR0912
	config: F3FiveWayConfig,
	*,
	model: F3FiveWayModelSource,
	layout_id: str,
	data_size: str,
	job_dir: Path,
) -> dict[str, object]:
	"""Bind one completed evaluation to its source and canonical condition."""
	label = f'{model.model_id}/{layout_id}/{data_size}'
	condition = _condition_identity(config, layout_id=layout_id, data_size=data_size)
	metrics = _read_json(job_dir / EVALUATION_DIR_NAME / METRICS_NAME)
	if metrics.get('aggregation_unit') != EXPECTED_AGGREGATION_UNIT:
		raise ValueError(
			f'{label} metrics aggregation_unit must equal '
			f'{EXPECTED_AGGREGATION_UNIT!r}; got {metrics.get("aggregation_unit")!r}'
		)
	voxels = metrics.get('evaluation_voxel_count')
	if not isinstance(voxels, int) or isinstance(voxels, bool) or voxels <= 0:
		raise ValueError(f'{label} metrics evaluation_voxel_count must be positive')
	declared_voxels = condition['validation_voxel_count']
	if voxels != declared_voxels:
		raise ValueError(
			f'{label} evaluated {voxels} voxels but its section-layout condition '
			f'declares {declared_voxels}'
		)
	evaluation_metadata = _read_json(
		job_dir / EVALUATION_DIR_NAME / 'evaluation_metadata.json'
	)
	if evaluation_metadata.get('dataset') != dict(config.dataset):
		raise ValueError(f'{label} evaluation dataset does not match config')
	if evaluation_metadata.get('model_tag') != model.model_id:
		raise ValueError(f'{label} evaluation model_tag must equal {model.model_id!r}')
	policy = evaluation_metadata.get('policy')
	if not isinstance(policy, Mapping):
		raise TypeError(f'{label} evaluation policy must be a mapping')
	for key, expected in FIVE_WAY_EVALUATION_POLICY.items():
		recorded = policy.get(key)
		normalized = list(expected) if isinstance(expected, tuple) else expected
		if recorded != normalized:
			raise ValueError(
				f'{label} evaluation policy {key} must equal {normalized!r}; '
				f'got {recorded!r}'
			)
	condition_dir = Path(str(condition['condition_dir']))
	for key, name in (
		('voxel_dataset_metadata', 'voxel_dataset_metadata.json'),
		('voxel_split_grid', 'supervision_split_grid.npy'),
	):
		recorded_path, recorded_sha256 = _identity_entry(
			evaluation_metadata.get('inputs'),
			key,
			label=label,
			prefix='evaluation inputs',
		)
		expected_path = condition_dir / name
		if not _same_path(recorded_path, expected_path):
			raise ValueError(
				f'{label} evaluation used a foreign {key}: {recorded_path}'
			)
		if file_sha256(expected_path) != recorded_sha256:
			raise ValueError(
				f'{label} condition artifact changed after evaluation: {expected_path}'
			)
	resolved_config = _read_json(job_dir / DECODER_DIR_NAME / 'resolved_config.json')
	embeddings = resolved_config.get('embeddings')
	if not isinstance(embeddings, Mapping):
		raise TypeError(f'{label} decoder resolved config embeddings missing')
	if embeddings.get('checkpoint_path') != str(model.checkpoint):
		raise ValueError(
			f'{label} decoder checkpoint identity does not match the configured source'
		)
	if embeddings.get('input_dir') != str(model.embeddings_dir):
		raise ValueError(
			f'{label} decoder embedding directory does not match the configured source'
		)
	run_metadata = _read_json(job_dir / DECODER_DIR_NAME / 'run_metadata.json')
	expected_voxel_metadata = str(condition_dir / 'voxel_dataset_metadata.json')
	if run_metadata.get('voxel_dataset_metadata') != expected_voxel_metadata:
		raise ValueError(
			f'{label} decoder supervision dataset does not match the shared '
			'section-layout condition'
		)
	supervision_identity = run_metadata.get('train_tile_manifest_sha256')
	validation_manifest = run_metadata.get('validation_tile_manifest_sha256')
	if not isinstance(supervision_identity, str) or not supervision_identity:
		raise ValueError(f'{label} decoder train tile manifest identity missing')
	if not isinstance(validation_manifest, str) or not validation_manifest:
		raise ValueError(
			f'{label} decoder validation tile manifest identity missing'
		)
	identity = _job_source_identity(
		label=label,
		model=model,
		survey_id=config.dataset['name'],
		job_dir=job_dir,
		evaluation_metadata=evaluation_metadata,
	)
	return {
		**identity,
		'supervision_identity': supervision_identity,
		'validation_identity': str(condition['validation_mask_sha256']),
		'validation_voxel_count': voxels,
		'_validation_tile_manifest_sha256': validation_manifest,
	}


def _job_source_identity(  # noqa: C901
	*,
	label: str,
	model: F3FiveWayModelSource,
	survey_id: str,
	job_dir: Path,
	evaluation_metadata: Mapping[str, object],
) -> dict[str, str]:
	"""Read the recorded SHAs that make one comparison row reproducible."""
	prediction_metadata_path = (
		job_dir / PREDICTION_DIR_NAME / PREDICTION_METADATA_NAME
	)
	recorded_path, recorded_sha256 = _identity_entry(
		evaluation_metadata.get('inputs'),
		'prediction_metadata',
		label=label,
		prefix='evaluation inputs',
	)
	if not _same_path(recorded_path, prediction_metadata_path):
		raise ValueError(
			f'{label} metrics were not evaluated from this job prediction: '
			f'{recorded_path}'
		)
	if file_sha256(prediction_metadata_path) != recorded_sha256:
		raise ValueError(
			f'{label} prediction artifact changed after its evaluation: '
			f'{prediction_metadata_path}'
		)
	prediction_metadata = _read_json(prediction_metadata_path)
	if prediction_metadata.get('model_tag') != model.model_id:
		raise ValueError(f'{label} prediction model_tag must equal {model.model_id!r}')
	source_identity = prediction_metadata.get('source_identity')
	decoder_path, decoder_sha256 = _identity_entry(
		source_identity,
		'decoder_checkpoint',
		label=label,
		prefix='prediction source_identity',
	)
	expected_decoder = job_dir / DECODER_DIR_NAME / BEST_CHECKPOINT_NAME
	if not _same_path(decoder_path, expected_decoder):
		raise ValueError(
			f'{label} prediction used a foreign decoder checkpoint: {decoder_path}'
		)
	if file_sha256(expected_decoder) != decoder_sha256:
		raise ValueError(
			f'{label} decoder checkpoint changed after prediction: {expected_decoder}'
		)
	artifact_identities = (
		source_identity.get('artifact_identities')
		if isinstance(source_identity, Mapping)
		else None
	)
	files = output_paths(model.embeddings_dir, survey_id)
	shas: dict[str, str] = {}
	for key, expected_path in (
		('embeddings', files.embeddings),
		('embedding_metadata', files.metadata),
		('valid_tokens', files.valid_tokens),
	):
		recorded, sha256 = _identity_entry(
			artifact_identities,
			key,
			label=label,
			prefix='prediction artifact_identities',
		)
		if not _same_path(recorded, expected_path):
			raise ValueError(
				f'{label} prediction {key} is not the configured five-way source: '
				f'{recorded}'
			)
		shas[key] = sha256
	if file_sha256(files.metadata) != shas['embedding_metadata']:
		# The recorded encoder SHA is only trustworthy while the metadata file it
		# was read from still holds the content this job consumed.
		raise ValueError(
			f'{label} embedding metadata changed after this job ran: {files.metadata}'
		)
	metadata = _read_json(files.metadata)
	encoder_sha256 = metadata.get('checkpoint_sha256')
	if not isinstance(encoder_sha256, str) or len(encoder_sha256) != 64:
		raise ValueError(
			f'{label} embedding metadata checkpoint_sha256 must be a SHA-256 digest'
		)
	if metadata.get('checkpoint_path') != str(model.checkpoint):
		raise ValueError(
			f'{label} embedding metadata checkpoint_path is not the configured '
			f'five-way checkpoint: {metadata.get("checkpoint_path")!r}'
		)
	return {
		'encoder_checkpoint_sha256': encoder_sha256,
		'embeddings_sha256': shas['embeddings'],
		'embedding_metadata_sha256': shas['embedding_metadata'],
		'valid_tokens_sha256': shas['valid_tokens'],
		'decoder_checkpoint_sha256': decoder_sha256,
	}


def _identity_entry(
	container: object, key: str, *, label: str, prefix: str
) -> tuple[str, str]:
	if not isinstance(container, Mapping):
		raise ValueError(  # noqa: TRY004 - a missing block is a value error
			f'{label} {prefix} must be a mapping'
		)
	entry = container.get(key)
	if not isinstance(entry, Mapping):
		raise ValueError(  # noqa: TRY004 - a missing identity is a value error
			f'{label} {prefix}.{key} identity is required'
		)
	path = entry.get('path')
	sha256 = entry.get('sha256')
	if not isinstance(path, str) or not path:
		raise ValueError(f'{label} {prefix}.{key} path is required')
	if not isinstance(sha256, str) or len(sha256) != 64:
		raise ValueError(
			f'{label} {prefix}.{key} sha256 must be a SHA-256 digest'
		)
	return path, sha256


def _same_path(recorded: str, expected: Path) -> bool:
	return Path(recorded).resolve(strict=False) == expected.resolve(strict=False)


def _validate_cross_job_identity(rows: list[dict[str, object]]) -> None:
	voxel_counts = {row['validation_voxel_count'] for row in rows}
	if len(voxel_counts) != 1:
		raise ValueError(
			'validation voxel count must be identical across all 75 jobs; '
			f'got {sorted(voxel_counts)!r}'
		)
	for (layout_id, data_size), group in _grouped(rows).items():
		for key in (
			'supervision_identity',
			'_validation_tile_manifest_sha256',
		):
			values = {str(row[key]) for row in group}
			if len(values) != 1:
				raise ValueError(
					f'{layout_id}/{data_size} supervision identity differs '
					'between models'
				)
	_validate_source_identity(rows)
	for row in rows:
		del row['_validation_tile_manifest_sha256']


def _validate_source_identity(rows: list[dict[str, object]]) -> None:
	"""Reject sources that were regenerated in place between the 75 jobs."""
	by_model: dict[str, list[dict[str, object]]] = {}
	for row in rows:
		by_model.setdefault(str(row['model_id']), []).append(row)
	for model_id, group in by_model.items():
		for key in PER_MODEL_IDENTITY_FIELDS:
			values = {str(row[key]) for row in group}
			if len(values) != 1:
				raise ValueError(
					f'{model_id} {key} differs between its {len(group)} jobs; '
					f'got {sorted(values)!r}'
				)
	valid_tokens = {str(row['valid_tokens_sha256']) for row in rows}
	if len(valid_tokens) != 1:
		raise ValueError(
			'valid-token identity must be shared by all 75 jobs; '
			f'got {sorted(valid_tokens)!r}'
		)
	encoders: dict[str, str] = {}
	for model_id, group in by_model.items():
		encoder = str(group[0]['encoder_checkpoint_sha256'])
		duplicate = encoders.setdefault(encoder, model_id)
		if duplicate != model_id:
			raise ValueError(
				'encoder checkpoint SHA-256 must differ across models; '
				f'{duplicate!r} and {model_id!r} match'
			)


def _grouped(
	rows: list[dict[str, object]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
	grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
	for row in rows:
		key = (str(row['layout_id']), str(row['data_size']))
		grouped.setdefault(key, []).append(row)
	duplicates = {
		key: [str(row['model_id']) for row in group]
		for key, group in grouped.items()
		if len(group) != len({str(row['model_id']) for row in group})
	}
	if duplicates:
		raise ValueError(f'duplicate five-way conditions: {duplicates!r}')
	return grouped


def _paired_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
	by_cell = {
		(str(row['model_id']), str(row['layout_id']), str(row['data_size'])): row
		for row in rows
	}
	paired = []
	for data_size in DATA_SIZES:
		for layout_id in LAYOUT_IDS:
			for comparison_id, left_model, right_model in PAIRED_COMPARISONS:
				left = by_cell[left_model, layout_id, data_size]
				right = by_cell[right_model, layout_id, data_size]
				for metric in SUMMARY_METRICS:
					left_value = float(left[metric])
					right_value = float(right[metric])
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
							'delta': left_value - right_value,
						}
					)
	return paired


def _by_size_rows(paired: list[dict[str, object]]) -> list[dict[str, object]]:
	rows = []
	for data_size in DATA_SIZES:
		for comparison_id, _, _ in PAIRED_COMPARISONS:
			for metric in SUMMARY_METRICS:
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
						'sample_std': statistics.stdev(deltas),
						'median': statistics.median(deltas),
						'min': min(deltas),
						'max': max(deltas),
						'positive_count': sum(delta > 0 for delta in deltas),
						'zero_count': sum(delta == 0 for delta in deltas),
						'negative_count': sum(delta < 0 for delta in deltas),
					}
				)
	return rows


def _summary_payload(
	config: F3FiveWayConfig, by_size: list[dict[str, object]]
) -> dict[str, object]:
	comparison: dict[str, dict[str, dict[str, object]]] = {}
	for row in by_size:
		comparison.setdefault(str(row['data_size']), {}).setdefault(
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
		'summary_name': config.summary_name,
		'primary_metric': PRIMARY_METRIC,
		'models': list(config.model_ids),
		'job_count': len(config.model_ids) * len(LAYOUT_IDS) * len(DATA_SIZES),
		'statistical_unit': 'layout_id',
		'by_size': comparison,
	}


def _summary_markdown(by_size: list[dict[str, object]]) -> str:
	lines = [
		'# F3 lithology five-way summary',
		'',
		(
			f'Primary metric: `{PRIMARY_METRIC}` on unique validation voxels; '
			'paired unit is `layout_id`, aggregated per size.'
		),
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


def _csv_text(
	fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> str:
	buffer = io.StringIO()
	writer = csv.DictWriter(buffer, fieldnames=fieldnames)
	writer.writeheader()
	for row in rows:
		writer.writerow({key: row[key] for key in fieldnames})
	return buffer.getvalue()


def _read_json(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'missing five-way artifact: {path}')
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'{path} must contain a JSON object')
	return payload


__all__ = [
	'BY_SIZE_FIELDNAMES',
	'COMPARISON_CSV_NAME',
	'COMPARISON_FIELDNAMES',
	'EXPECTED_AGGREGATION_UNIT',
	'PAIRED_COMPARISONS',
	'PAIRED_DELTAS_CSV_NAME',
	'PAIRED_FIELDNAMES',
	'PER_MODEL_IDENTITY_FIELDS',
	'PRIMARY_METRIC',
	'SOURCE_IDENTITY_FIELDS',
	'SUMMARY_BY_SIZE_CSV_NAME',
	'SUMMARY_JSON_NAME',
	'SUMMARY_MD_NAME',
	'SUMMARY_METRICS',
	'SUMMARY_OUTPUT_NAMES',
	'inspect_f3_lithology_five_way_results',
	'read_f3_lithology_job_evidence',
	'summarize_f3_lithology_five_way',
]
