"""Small candidate path for F3 lithology frozen-encoder comparisons."""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import statistics
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	F3FiveWayConfig,
	F3FiveWayModelSource,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.five_way_results import (
	read_f3_lithology_job_evidence,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	EVALUATION_DIR_NAME,
	METRICS_NAME,
	F3FiveWayJob,
	inspect_f3_lithology_five_way_job,
	run_f3_lithology_frozen_encoder_job,
)

COMPARISON_CSV_NAME = 'comparison.csv'
SUMMARY_JSON_NAME = 'summary.json'
SUMMARY_MD_NAME = 'summary.md'
SUMMARY_OUTPUT_NAMES = (
	COMPARISON_CSV_NAME,
	SUMMARY_JSON_NAME,
	SUMMARY_MD_NAME,
)
PRIMARY_METRIC = 'macro_f1'
EXPECTED_AGGREGATION_UNIT = 'unique_validation_voxel'
METADATA_IDENTITY_KEYS = (
	'survey_id',
	'source_amplitude_path',
	'normalization_stats_path',
	'volume_shape_xyz',
	'token_grid_shape',
	'patch_size',
	'window_size',
	'overlap',
	'output_dtype',
	'min_token_valid_fraction',
	'model_geometry',
	'precision',
	'preprocessing',
	'amplitude_agc',
	'normalized_clip_abs',
	'finite_check_mode',
	'zero_mask',
	'preprocessing_cache',
)
COMPARISON_FIELDNAMES = (
	'candidate_id',
	'layout_id',
	'data_size',
	'metric',
	'candidate_value',
	'random_value',
	'paired_delta',
	'candidate_metrics_path',
	'candidate_metrics_sha256',
	'random_metrics_path',
	'random_metrics_sha256',
	'checkpoint_sha256',
	'embeddings_sha256',
	'embedding_metadata_sha256',
	'valid_tokens_sha256',
)


@dataclass(frozen=True)
class F3LithologyCandidateConfig:
	"""The four artifact paths and stable ID needed to evaluate one candidate."""

	canonical_config: Path
	candidate_id: str
	checkpoint: Path
	embeddings_dir: Path
	runs_root: Path
	summary_root: Path


def f3_lithology_candidate_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyCandidateConfig:
	"""Resolve the deliberately small, exact candidate configuration schema."""
	_require_exact_keys(config, {'benchmark', 'candidate', 'outputs'}, 'config')
	benchmark = _required_mapping(config, 'benchmark')
	candidate = _required_mapping(config, 'candidate')
	outputs = _required_mapping(config, 'outputs')
	_require_exact_keys(benchmark, {'canonical_config'}, 'benchmark')
	_require_exact_keys(candidate, {'id', 'checkpoint', 'embeddings_dir'}, 'candidate')
	_require_exact_keys(outputs, {'runs_root', 'summary_root'}, 'outputs')
	candidate_id = _required_str(candidate, 'id', prefix='candidate')
	if Path(candidate_id).name != candidate_id or candidate_id in {'.', '..'}:
		raise ValueError('candidate.id must be one path segment')
	runs_root = _required_absolute_path(outputs, 'runs_root', prefix='outputs')
	summary_root = _required_absolute_path(outputs, 'summary_root', prefix='outputs')
	if runs_root == summary_root:
		raise ValueError('outputs.runs_root and outputs.summary_root must differ')
	return F3LithologyCandidateConfig(
		canonical_config=_required_absolute_path(
			benchmark, 'canonical_config', prefix='benchmark'
		),
		candidate_id=candidate_id,
		checkpoint=_required_absolute_path(candidate, 'checkpoint', prefix='candidate'),
		embeddings_dir=_required_absolute_path(
			candidate, 'embeddings_dir', prefix='candidate'
		),
		runs_root=runs_root,
		summary_root=summary_root,
	)


def load_f3_lithology_candidate_canonical_config(
	config: F3LithologyCandidateConfig,
) -> F3FiveWayConfig:
	"""Load the canonical five-way config named by one candidate config."""
	if not config.canonical_config.is_file():
		raise FileNotFoundError(
			f'canonical five-way config does not exist: {config.canonical_config}'
		)
	canonical_config = f3_lithology_five_way_config_from_mapping(
		load_config(config.canonical_config)
	)
	_validate_candidate_namespace(config, canonical_config)
	return canonical_config


def _validate_candidate_namespace(
	config: F3LithologyCandidateConfig,
	canonical_config: F3FiveWayConfig,
) -> None:
	if config.candidate_id in canonical_config.model_ids:
		raise ValueError(
			f'candidate.id conflicts with canonical model ID: {config.candidate_id!r}'
		)
	for candidate_label, candidate_root in (
		('outputs.runs_root', config.runs_root),
		('outputs.summary_root', config.summary_root),
	):
		for canonical_label, canonical_root in (
			('canonical runs_root', canonical_config.runs_root),
			('canonical summary_root', canonical_config.summary_root),
		):
			if _paths_overlap(candidate_root, canonical_root):
				raise ValueError(
					f'{candidate_label} overlaps {canonical_label}: '
					f'{candidate_root} and {canonical_root}'
				)


def _paths_overlap(first: Path, second: Path) -> bool:
	first = first.resolve(strict=False)
	second = second.resolve(strict=False)
	return (
		first == second
		or first.is_relative_to(second)
		or second.is_relative_to(first)
	)


def resolve_f3_lithology_candidate_job(
	config: F3LithologyCandidateConfig,
	canonical_config: F3FiveWayConfig,
	*,
	layout: str,
	size: str,
) -> F3FiveWayJob:
	"""Resolve one candidate cell onto the canonical downstream condition."""
	if layout not in LAYOUT_IDS:
		raise ValueError(
			f'unknown layout: {layout!r}; expected one of {list(LAYOUT_IDS)!r}'
		)
	if size not in DATA_SIZES:
		raise ValueError(
			f'unknown data size: {size!r}; expected one of {list(DATA_SIZES)!r}'
		)
	condition_dir = (
		canonical_config.section_layout_dataset_root
		/ 'datasets'
		/ f'layout={layout}'
		/ f'size={size}'
		/ 'voxel_supervision'
	)
	output_dir = (
		config.runs_root
		/ f'model={config.candidate_id}'
		/ f'layout={layout}'
		/ f'size={size}'
	)
	return F3FiveWayJob(
		config=canonical_config,
		model=F3FiveWayModelSource(
			model_id=config.candidate_id,
			checkpoint=config.checkpoint,
			embeddings_dir=config.embeddings_dir,
			expected={},
		),
		layout_id=layout,
		data_size=size,
		condition_dir=condition_dir,
		output_dir=output_dir,
	)


def audit_f3_lithology_candidate_source(
	config: F3LithologyCandidateConfig,
	canonical_config: F3FiveWayConfig,
) -> dict[str, object]:
	"""Check only the source identities needed for a fair downstream run."""
	if not config.checkpoint.is_file():
		raise FileNotFoundError(
			f'candidate checkpoint does not exist: {config.checkpoint}'
		)
	survey_id = canonical_config.dataset['name']
	candidate_files = _required_embedding_files(
		config.embeddings_dir, survey_id, label='candidate'
	)
	random_source = canonical_config.model_by_id('random')
	random_files = _required_embedding_files(
		random_source.embeddings_dir, survey_id, label='canonical random'
	)
	candidate_metadata = _read_json(candidate_files.metadata, label='candidate')
	random_metadata = _read_json(random_files.metadata, label='canonical random')
	checkpoint_sha256 = file_sha256(config.checkpoint)
	_recorded_checkpoint_path = candidate_metadata.get('checkpoint_path')
	if not isinstance(_recorded_checkpoint_path, str) or not _recorded_checkpoint_path:
		raise ValueError('candidate embedding metadata checkpoint_path is required')
	recorded_checkpoint = Path(_recorded_checkpoint_path).resolve(strict=False)
	if recorded_checkpoint != config.checkpoint.resolve(strict=False):
		raise ValueError(
			'candidate embedding metadata checkpoint_path does not match '
			'the configured checkpoint'
		)
	if candidate_metadata.get('checkpoint_sha256') != checkpoint_sha256:
		raise ValueError(
			'candidate embedding metadata checkpoint_sha256 does not match '
			'the checkpoint file'
		)
	_compare_metadata_identity(candidate_metadata, random_metadata)
	random_checkpoint_path = random_metadata.get('checkpoint_path')
	if not isinstance(random_checkpoint_path, str) or Path(
		random_checkpoint_path
	).resolve(strict=False) != random_source.checkpoint.resolve(strict=False):
		raise ValueError(
			'canonical random embedding metadata checkpoint_path does not match '
			'the canonical config'
		)
	random_checkpoint_sha256 = random_metadata.get('checkpoint_sha256')
	if (
		not isinstance(random_checkpoint_sha256, str)
		or len(random_checkpoint_sha256) != 64
	):
		raise ValueError(
			'canonical random embedding metadata checkpoint_sha256 must be a '
			'SHA-256 digest'
		)
	_compare_embedding_arrays(candidate_files, random_files)
	valid_tokens_sha256 = file_sha256(candidate_files.valid_tokens)
	random_valid_tokens_sha256 = file_sha256(random_files.valid_tokens)
	if valid_tokens_sha256 != random_valid_tokens_sha256:
		raise ValueError(
			'candidate valid-token mask is not byte-identical to the canonical '
			'random valid-token mask'
		)
	return {
		'candidate_id': config.candidate_id,
		'checkpoint_path': str(config.checkpoint),
		'checkpoint_sha256': checkpoint_sha256,
		'embeddings_path': str(candidate_files.embeddings),
		'embeddings_sha256': file_sha256(candidate_files.embeddings),
		'embedding_metadata_path': str(candidate_files.metadata),
		'embedding_metadata_sha256': file_sha256(candidate_files.metadata),
		'valid_tokens_path': str(candidate_files.valid_tokens),
		'valid_tokens_sha256': valid_tokens_sha256,
		'canonical_random': {
			'model_id': random_source.model_id,
			'checkpoint_path': str(random_source.checkpoint),
			'checkpoint_sha256': random_checkpoint_sha256,
			'embeddings_path': str(random_files.embeddings),
			'embeddings_sha256': file_sha256(random_files.embeddings),
			'embedding_metadata_path': str(random_files.metadata),
			'embedding_metadata_sha256': file_sha256(random_files.metadata),
			'valid_tokens_path': str(random_files.valid_tokens),
			'valid_tokens_sha256': random_valid_tokens_sha256,
		},
	}


def inspect_f3_lithology_candidate_job(
	config: F3LithologyCandidateConfig,
	canonical_config: F3FiveWayConfig,
	job: F3FiveWayJob,
) -> dict[str, object]:
	"""Return the no-write job plan together with candidate SHA provenance."""
	provenance = audit_f3_lithology_candidate_source(config, canonical_config)
	return {
		**inspect_f3_lithology_five_way_job(job),
		**provenance,
	}


def run_f3_lithology_candidate_job(  # noqa: PLR0913
	config: F3LithologyCandidateConfig,
	canonical_config: F3FiveWayConfig,
	job: F3FiveWayJob,
	*,
	device: str = 'auto',
	max_steps: int | None = None,
	resume: Path | None = None,
) -> dict[str, object]:
	"""Audit a candidate, then use the shared decoder/prediction/evaluation job."""
	if job.metrics_path.is_file():
		raise FileExistsError(
			f'job already completed; refusing to overwrite {job.metrics_path}'
		)
	provenance = audit_f3_lithology_candidate_source(config, canonical_config)
	return {
		**run_f3_lithology_frozen_encoder_job(
			job, device=device, max_steps=max_steps, resume=resume
		),
		**provenance,
	}


def summarize_f3_lithology_candidate(
	config: F3LithologyCandidateConfig,
	canonical_config: F3FiveWayConfig,
) -> dict[str, object]:
	"""Write the three-file paired candidate-versus-random summary."""
	provenance = audit_f3_lithology_candidate_source(config, canonical_config)
	rows = _comparison_rows(config, canonical_config, provenance)
	by_size = _summary_by_size(rows)
	outputs = {
		COMPARISON_CSV_NAME: _csv_text(COMPARISON_FIELDNAMES, rows),
		SUMMARY_JSON_NAME: json.dumps(
			_summary_payload(config, provenance, rows, by_size),
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		SUMMARY_MD_NAME: _summary_markdown(config, by_size),
	}
	if config.summary_root.exists():
		raise FileExistsError(
			f'refusing to overwrite existing summary: {config.summary_root}'
		)
	config.summary_root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix=f'.{config.summary_root.name}.staging-',
			dir=config.summary_root.parent,
		)
	)
	try:
		for name, text in outputs.items():
			(staging / name).write_text(text, encoding='utf-8')
		staging.replace(config.summary_root)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	return {
		'candidate_id': config.candidate_id,
		'complete_jobs': len(rows),
		'summary_root': str(config.summary_root),
		'outputs': [str(config.summary_root / name) for name in SUMMARY_OUTPUT_NAMES],
	}


def _require_exact_keys(
	value: Mapping[str, object], expected: set[str], label: str
) -> None:
	keys = set(value)
	if keys == expected:
		return
	missing = sorted(expected - keys)
	unexpected = sorted(str(key) for key in keys - expected)
	raise ValueError(
		f'{label} keys must be exactly {sorted(expected)!r}; '
		f'missing={missing!r}, unexpected={unexpected!r}'
	)


def _required_embedding_files(
	embeddings_dir: Path, survey_id: str, *, label: str
) -> object:
	files = output_paths(embeddings_dir, survey_id)
	for path in (files.embeddings, files.valid_tokens, files.metadata):
		if not path.is_file():
			raise FileNotFoundError(f'{label} embedding source is missing: {path}')
	return files


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} metadata must contain JSON: {path}') from error
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} metadata must contain a JSON object: {path}')
	return payload


def _compare_metadata_identity(
	candidate: Mapping[str, object], random: Mapping[str, object]
) -> None:
	for key in METADATA_IDENTITY_KEYS:
		if candidate.get(key) != random.get(key):
			raise ValueError(
				f'candidate embedding metadata {key} does not match canonical random'
			)


def _compare_embedding_arrays(candidate: object, random: object) -> None:
	candidate_embeddings = np.load(
		candidate.embeddings, mmap_mode='r', allow_pickle=False
	)
	random_embeddings = np.load(random.embeddings, mmap_mode='r', allow_pickle=False)
	if candidate_embeddings.shape != random_embeddings.shape:
		raise ValueError(
			'candidate embedding shape does not match canonical random: '
			f'{candidate_embeddings.shape!r} != {random_embeddings.shape!r}'
		)
	if candidate_embeddings.dtype != random_embeddings.dtype:
		raise ValueError(
			'candidate embedding dtype does not match canonical random: '
			f'{candidate_embeddings.dtype!r} != {random_embeddings.dtype!r}'
		)
	candidate_tokens = np.load(
		candidate.valid_tokens, mmap_mode='r', allow_pickle=False
	)
	random_tokens = np.load(random.valid_tokens, mmap_mode='r', allow_pickle=False)
	if candidate_tokens.shape != random_tokens.shape:
		raise ValueError(
			'candidate valid-token shape does not match canonical random: '
			f'{candidate_tokens.shape!r} != {random_tokens.shape!r}'
		)
	if candidate_tokens.dtype != random_tokens.dtype:
		raise ValueError(
			'candidate valid-token dtype does not match canonical random: '
			f'{candidate_tokens.dtype!r} != {random_tokens.dtype!r}'
		)


def _comparison_rows(
	config: F3LithologyCandidateConfig,
	canonical_config: F3FiveWayConfig,
	provenance: Mapping[str, object],
) -> list[dict[str, object]]:
	random_source = canonical_config.model_by_id('random')
	cells = [
		(layout_id, data_size) for data_size in DATA_SIZES for layout_id in LAYOUT_IDS
	]
	paths = [
		(
			layout_id,
			data_size,
			_metrics_path(config.runs_root, config.candidate_id, layout_id, data_size),
			_metrics_path(
				canonical_config.runs_root,
				random_source.model_id,
				layout_id,
				data_size,
			),
		)
		for layout_id, data_size in cells
	]
	missing = [
		str(path)
		for _, _, candidate_path, random_path in paths
		for path in (candidate_path, random_path)
		if not path.is_file()
	]
	if missing:
		raise FileNotFoundError(
			f'missing {len(missing)} candidate/random metric cell(s): {missing!r}'
		)
	rows = []
	candidate_source = F3FiveWayModelSource(
		model_id=config.candidate_id,
		checkpoint=config.checkpoint,
		embeddings_dir=config.embeddings_dir,
		expected={},
	)
	for layout_id, data_size, candidate_path, random_path in paths:
		candidate_job_dir = candidate_path.parent.parent
		random_job_dir = random_path.parent.parent
		candidate_evidence = read_f3_lithology_job_evidence(
			canonical_config,
			model=candidate_source,
			layout_id=layout_id,
			data_size=data_size,
			job_dir=candidate_job_dir,
		)
		random_evidence = read_f3_lithology_job_evidence(
			canonical_config,
			model=random_source,
			layout_id=layout_id,
			data_size=data_size,
			job_dir=random_job_dir,
		)
		_assert_job_source_matches(
			candidate_evidence,
			provenance,
			label=f'candidate/{layout_id}/{data_size}',
		)
		_assert_job_source_matches(
			random_evidence,
			provenance['canonical_random'],
			label=f'random/{layout_id}/{data_size}',
		)
		candidate_value, candidate_voxels = _macro_f1(
			candidate_path, label=f'candidate/{layout_id}/{data_size}'
		)
		random_value, random_voxels = _macro_f1(
			random_path, label=f'random/{layout_id}/{data_size}'
		)
		if candidate_voxels != random_voxels:
			raise ValueError(
				f'{layout_id}/{data_size} candidate and random evaluation voxel '
				f'counts differ: {candidate_voxels} != {random_voxels}'
			)
		rows.append(
			{
				'candidate_id': config.candidate_id,
				'layout_id': layout_id,
				'data_size': data_size,
				'metric': PRIMARY_METRIC,
				'candidate_value': candidate_value,
				'random_value': random_value,
				'paired_delta': candidate_value - random_value,
				'candidate_metrics_path': str(candidate_path),
				'candidate_metrics_sha256': file_sha256(candidate_path),
				'random_metrics_path': str(random_path),
				'random_metrics_sha256': file_sha256(random_path),
				'checkpoint_sha256': provenance['checkpoint_sha256'],
				'embeddings_sha256': provenance['embeddings_sha256'],
				'embedding_metadata_sha256': provenance['embedding_metadata_sha256'],
				'valid_tokens_sha256': provenance['valid_tokens_sha256'],
			}
		)
	return rows


def _assert_job_source_matches(
	evidence: Mapping[str, object],
	current_source: object,
	*,
	label: str,
) -> None:
	if not isinstance(current_source, Mapping):
		raise TypeError(f'{label} current source provenance must be a mapping')
	for evidence_key, source_key in (
		('encoder_checkpoint_sha256', 'checkpoint_sha256'),
		('embeddings_sha256', 'embeddings_sha256'),
		('embedding_metadata_sha256', 'embedding_metadata_sha256'),
		('valid_tokens_sha256', 'valid_tokens_sha256'),
	):
		if evidence.get(evidence_key) != current_source.get(source_key):
			raise ValueError(
				f'{label} completed job {evidence_key} does not match '
				'the current source'
			)


def _metrics_path(
	runs_root: Path, model_id: str, layout_id: str, data_size: str
) -> Path:
	return (
		runs_root
		/ f'model={model_id}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ EVALUATION_DIR_NAME
		/ METRICS_NAME
	)


def _macro_f1(path: Path, *, label: str) -> tuple[float, int]:
	metrics = _read_json(path, label=label)
	if metrics.get('aggregation_unit') != EXPECTED_AGGREGATION_UNIT:
		raise ValueError(
			f'{label} metrics aggregation_unit must equal '
			f'{EXPECTED_AGGREGATION_UNIT!r}; '
			f'got {metrics.get("aggregation_unit")!r}'
		)
	value = metrics.get(PRIMARY_METRIC)
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise ValueError(  # noqa: TRY004 - malformed artifacts are stale data
			f'{label} metrics {PRIMARY_METRIC} must be numeric'
		)
	if not math.isfinite(float(value)):
		raise ValueError(f'{label} metrics {PRIMARY_METRIC} must be finite')
	voxels = metrics.get('evaluation_voxel_count')
	if not isinstance(voxels, int) or isinstance(voxels, bool) or voxels <= 0:
		raise ValueError(f'{label} metrics evaluation_voxel_count must be positive')
	return float(value), voxels


def _summary_by_size(
	rows: list[dict[str, object]],
) -> list[dict[str, object]]:
	summaries = []
	for data_size in DATA_SIZES:
		by_size = [row for row in rows if row['data_size'] == data_size]
		if len(by_size) != len(LAYOUT_IDS):
			raise ValueError(
				f'{data_size} must contain exactly {len(LAYOUT_IDS)} layout cells'
			)
		deltas = [float(row['paired_delta']) for row in by_size]
		summaries.append(
			{
				'data_size': data_size,
				'n_layouts': len(by_size),
				'candidate_mean': statistics.fmean(
					float(row['candidate_value']) for row in by_size
				),
				'random_mean': statistics.fmean(
					float(row['random_value']) for row in by_size
				),
				'mean': statistics.fmean(deltas),
				'median': statistics.median(deltas),
				'sample_std': statistics.stdev(deltas),
				'positive_count': sum(delta > 0 for delta in deltas),
				'zero_count': sum(delta == 0 for delta in deltas),
				'negative_count': sum(delta < 0 for delta in deltas),
			}
		)
	return summaries


def _summary_payload(
	config: F3LithologyCandidateConfig,
	provenance: Mapping[str, object],
	rows: list[dict[str, object]],
	by_size: list[dict[str, object]],
) -> dict[str, object]:
	return {
		'schema_version': 1,
		'candidate_id': config.candidate_id,
		'primary_metric': PRIMARY_METRIC,
		'aggregation_unit': EXPECTED_AGGREGATION_UNIT,
		'statistical_unit': 'layout_id',
		'job_count': len(rows),
		'by_size': {row['data_size']: row for row in by_size},
		'provenance': {
			'candidate': {
				key: provenance[key]
				for key in (
					'candidate_id',
					'checkpoint_path',
					'checkpoint_sha256',
					'embeddings_path',
					'embeddings_sha256',
					'embedding_metadata_path',
					'embedding_metadata_sha256',
					'valid_tokens_path',
					'valid_tokens_sha256',
				)
			},
			'canonical_random': provenance['canonical_random'],
			'metrics': [
				{
					'layout_id': row['layout_id'],
					'data_size': row['data_size'],
					'candidate_path': row['candidate_metrics_path'],
					'candidate_sha256': row['candidate_metrics_sha256'],
					'random_path': row['random_metrics_path'],
					'random_sha256': row['random_metrics_sha256'],
				}
				for row in rows
			],
		},
	}


def _summary_markdown(
	config: F3LithologyCandidateConfig, by_size: list[dict[str, object]]
) -> str:
	lines = [
		'# F3 lithology candidate summary',
		'',
		(
			f'Candidate `{config.candidate_id}` versus canonical `random`. '
			f'Primary metric: `{PRIMARY_METRIC}` on unique validation voxels; '
			'paired unit is `layout_id`.'
		),
		'',
		(
			'| size | candidate mean | random mean | delta mean | median '
			'| sample std | +/0/- |'
		),
		'|---|---:|---:|---:|---:|---:|---|',
	]
	lines.extend(
		(
			f'| {row["data_size"]} | {row["candidate_mean"]:.6f} '
			f'| {row["random_mean"]:.6f} | {row["mean"]:.6f} '
			f'| {row["median"]:.6f} | {row["sample_std"]:.6f} '
			f'| {row["positive_count"]}/{row["zero_count"]}'
			f'/{row["negative_count"]} |'
		)
		for row in by_size
	)
	lines.append('')
	return '\n'.join(lines)


def _csv_text(fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> str:
	buffer = io.StringIO()
	writer = csv.DictWriter(buffer, fieldnames=fieldnames)
	writer.writeheader()
	for row in rows:
		writer.writerow({key: row[key] for key in fieldnames})
	return buffer.getvalue()


__all__ = [
	'COMPARISON_CSV_NAME',
	'COMPARISON_FIELDNAMES',
	'EXPECTED_AGGREGATION_UNIT',
	'METADATA_IDENTITY_KEYS',
	'PRIMARY_METRIC',
	'SUMMARY_JSON_NAME',
	'SUMMARY_MD_NAME',
	'SUMMARY_OUTPUT_NAMES',
	'F3LithologyCandidateConfig',
	'audit_f3_lithology_candidate_source',
	'f3_lithology_candidate_config_from_mapping',
	'inspect_f3_lithology_candidate_job',
	'load_f3_lithology_candidate_canonical_config',
	'resolve_f3_lithology_candidate_job',
	'run_f3_lithology_candidate_job',
	'summarize_f3_lithology_candidate',
]
