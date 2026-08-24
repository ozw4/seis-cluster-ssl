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
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	DECODER_DIR_NAME,
	EVALUATION_DIR_NAME,
	FIVE_WAY_EVALUATION_POLICY,
	METRICS_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	LAYOUT_METADATA_NAME,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_five_way import F3FiveWayConfig

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
COMPARISON_FIELDNAMES = (
	'model_id',
	'layout_id',
	'data_size',
	'checkpoint_path',
	'embeddings_dir',
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
	conditions = _condition_identities(config)
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
			condition=conditions[layout_id, data_size],
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
			conditions[layout_id, data_size] = {
				'condition_dir': condition_dir,
				'validation_mask_sha256': identity.get('validation_mask_sha256'),
				'validation_voxel_count': identity.get('validation_voxel_count'),
			}
	masks = {
		str(condition['validation_mask_sha256'])
		for condition in conditions.values()
	}
	if len(masks) != 1:
		raise ValueError(
			'validation mask identity must be shared by all 15 conditions'
		)
	return conditions


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


def _load_job_row(  # noqa: C901, PLR0912
	config: F3FiveWayConfig,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
	condition: Mapping[str, object],
) -> dict[str, object]:
	job_dir = _job_dir(config, model_id, layout_id, data_size)
	label = f'{model_id}/{layout_id}/{data_size}'
	metrics = _read_json(job_dir / EVALUATION_DIR_NAME / METRICS_NAME)
	for key in (*SUMMARY_METRICS, 'evaluation_voxel_count'):
		value = metrics.get(key)
		if not isinstance(value, int | float) or isinstance(value, bool):
			# A malformed metrics artifact is stale data, not a caller bug.
			raise ValueError(  # noqa: TRY004
				f'{label} metrics {key} must be numeric'
			)
		if not math.isfinite(float(value)):
			raise ValueError(f'{label} metrics {key} must be finite')
	declared_voxels = condition['validation_voxel_count']
	if int(metrics['evaluation_voxel_count']) != declared_voxels:
		raise ValueError(
			f'{label} evaluated {int(metrics["evaluation_voxel_count"])} voxels '
			f'but its section-layout condition declares {declared_voxels}'
		)
	evaluation_metadata = _read_json(
		job_dir / EVALUATION_DIR_NAME / 'evaluation_metadata.json'
	)
	if evaluation_metadata.get('dataset') != dict(config.dataset):
		raise ValueError(f'{label} evaluation dataset does not match config')
	if evaluation_metadata.get('model_tag') != model_id:
		raise ValueError(f'{label} evaluation model_tag must equal {model_id!r}')
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
	resolved_config = _read_json(job_dir / DECODER_DIR_NAME / 'resolved_config.json')
	embeddings = resolved_config.get('embeddings')
	if not isinstance(embeddings, Mapping):
		raise TypeError(f'{label} decoder resolved config embeddings missing')
	model = config.model_by_id(model_id)
	if embeddings.get('checkpoint_path') != str(model.checkpoint):
		raise ValueError(
			f'{label} decoder checkpoint identity does not match the '
			'configured five-way checkpoint'
		)
	if embeddings.get('input_dir') != str(model.embeddings_dir):
		raise ValueError(
			f'{label} decoder embedding directory does not match the '
			'configured five-way source'
		)
	run_metadata = _read_json(job_dir / DECODER_DIR_NAME / 'run_metadata.json')
	expected_voxel_metadata = str(
		Path(str(condition['condition_dir'])) / 'voxel_dataset_metadata.json'
	)
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
	return {
		'model_id': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'checkpoint_path': str(model.checkpoint),
		'embeddings_dir': str(model.embeddings_dir),
		'supervision_identity': supervision_identity,
		'validation_identity': str(condition['validation_mask_sha256']),
		'macro_f1': float(metrics['macro_f1']),
		'mean_iou': float(metrics['mean_iou']),
		'balanced_accuracy': float(metrics['balanced_accuracy']),
		'weighted_f1': float(metrics['weighted_f1']),
		'validation_voxel_count': int(metrics['evaluation_voxel_count']),
		'metrics_path': str(job_dir / EVALUATION_DIR_NAME / METRICS_NAME),
		'_validation_tile_manifest_sha256': validation_manifest,
	}


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
	for row in rows:
		del row['_validation_tile_manifest_sha256']


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
		'summary_name': 'f3_lithology_mae_local_bt_five_way_v1',
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
	'PAIRED_COMPARISONS',
	'PAIRED_DELTAS_CSV_NAME',
	'PAIRED_FIELDNAMES',
	'PRIMARY_METRIC',
	'SUMMARY_BY_SIZE_CSV_NAME',
	'SUMMARY_JSON_NAME',
	'SUMMARY_MD_NAME',
	'SUMMARY_METRICS',
	'SUMMARY_OUTPUT_NAMES',
	'inspect_f3_lithology_five_way_results',
	'summarize_f3_lithology_five_way',
]
