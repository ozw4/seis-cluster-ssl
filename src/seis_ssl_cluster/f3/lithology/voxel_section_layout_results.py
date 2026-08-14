"""Generic paired-layout results for the F3 section-layout benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_validate_allowed_keys,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_roster import (
	EXPECTED_MODEL_IDS,
	F3SectionLayoutModel,
	F3SectionLayoutModelRoster,
	f3_lithology_voxel_section_layout_model_roster_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	METRIC_SPECS,
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	validate_f3_lithology_voxel_section_layout_manifest,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_runner import (
	RUN_MANIFEST_NAME,
	load_f3_lithology_voxel_section_layout_rows,
)

RESULTS_ARTIFACT_TYPE = 'f3_lithology_voxel_section_layout_results_summary'
HANDOFF_ARTIFACT_TYPE = 'f3_voxel_section_layout_handoff'
MODEL_REVIEW_ARTIFACT_TYPE = 'f3_voxel_section_layout_model_review'
RESULTS_SCHEMA_VERSION = 1
FORMAL_STATUSES = ('SECTION_LAYOUT_GO', 'SECTION_LAYOUT_HOLD', 'SECTION_LAYOUT_STOP')
PRIMARY_METRICS = ('macro_f1', 'mean_iou')
MONITORED_CLASS_IDS = (3, 5)
MONITORED_METRICS = ('f1', 'iou', 'boundary_recall_t2', 'boundary_recall_t4')
MAJOR_DEGRADATION_DELTA = -0.05
SYSTEMATIC_SIZE_COUNT = 2
FINAL_OUTPUT_NAMES = (
	'section_layout_job_metrics.csv',
	'section_layout_paired_deltas.csv',
	'section_layout_summary_by_size.csv',
	'section_layout_model_decisions.json',
	'section_layout_results_summary.json',
	'section_layout_results_summary.md',
	'section_layout_handoff.json',
)
MODEL_OUTPUT_NAMES = (
	'section_layout_model_job_metrics.csv',
	'section_layout_model_paired_deltas.csv',
	'section_layout_model_review.json',
	'section_layout_model_review.md',
)
PAIR_IDENTITY_FIELDS = (
	'layout_id',
	'data_size',
	'supervision_grid_identity',
	'selected_token_identity',
	'train_mask_identity',
	'validation_mask_identity',
	'decoder_seed',
	'initial_decoder_state_identity',
	'class_weights',
	'sampling_sequence_identity',
	'train_tile_identity',
	'validation_tile_identity',
	'metric_schema_identity',
)


@dataclass(frozen=True)
class F3SectionLayoutResultsConfig:
	"""Closed sources and output roots for generic section-layout aggregation."""

	artifact_root: Path
	workspace_root: Path
	model_roster: Path
	dataset_manifest: Path
	benchmark_root: Path
	report_dir: Path

	def model_results_dir(self, model_id: str) -> Path:
		"""Return the artifact-owned, non-published summary directory."""
		return self.benchmark_root / 'runs' / f'model={model_id}' / 'summary'


@dataclass(frozen=True)
class F3SectionLayoutResultsInspection:
	"""Fully validated metrics, pair deltas, and formal decisions."""

	mode: str
	requested_model_id: str | None
	loaded_model_ids: tuple[str, ...]
	job_metrics: tuple[Mapping[str, object], ...]
	paired_deltas: tuple[Mapping[str, object], ...]
	summary_by_size: tuple[Mapping[str, object], ...]
	model_decisions: Mapping[str, Mapping[str, object]]
	pair_identity_validation: Mapping[str, object]
	source_identities: Mapping[str, object]


@dataclass(frozen=True)
class F3SectionLayoutResultsResult:
	"""The exact lightweight file set written by one summary mode."""

	output_dir: Path
	files: tuple[Path, ...]
	inspection: F3SectionLayoutResultsInspection


@dataclass(frozen=True)
class _LoadedJob:
	model: F3SectionLayoutModel
	row: Mapping[str, object]
	metrics: Mapping[str, float]


def f3_lithology_voxel_section_layout_results_config_from_mapping(
	config: Mapping[str, object],
) -> F3SectionLayoutResultsConfig:
	"""Resolve the closed summarizer mapping and reject unregistered settings."""
	_validate_allowed_keys(
		config, frozenset({'paths', 'references', 'outputs'}), prefix='config'
	)
	paths = _required_mapping(config, 'paths')
	references = _required_mapping(config, 'references')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		paths, frozenset({'artifact_root', 'workspace_root'}), prefix='paths'
	)
	_validate_allowed_keys(
		references,
		frozenset({'model_roster', 'section_layout_dataset_manifest'}),
		prefix='references',
	)
	_validate_allowed_keys(
		outputs,
		frozenset({'benchmark_root', 'report_dir'}),
		prefix='outputs',
	)
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	workspace_root = _required_absolute_path(
		paths, 'workspace_root', prefix='paths'
	)
	benchmark_root = _required_absolute_path(
		outputs, 'benchmark_root', prefix='outputs'
	)
	report_dir = _required_absolute_path(outputs, 'report_dir', prefix='outputs')
	if not benchmark_root.is_relative_to(artifact_root):
		raise ValueError('outputs.benchmark_root must be inside paths.artifact_root')
	expected_reports_root = workspace_root / 'reports'
	if not report_dir.is_relative_to(expected_reports_root):
		raise ValueError('outputs.report_dir must be inside workspace_root/reports')
	return F3SectionLayoutResultsConfig(
		artifact_root=artifact_root,
		workspace_root=workspace_root,
		model_roster=_required_absolute_path(
			references, 'model_roster', prefix='references'
		),
		dataset_manifest=_required_absolute_path(
			references, 'section_layout_dataset_manifest', prefix='references'
		),
		benchmark_root=benchmark_root,
		report_dir=report_dir,
	)


resolve_f3_lithology_voxel_section_layout_results_config = (
	f3_lithology_voxel_section_layout_results_config_from_mapping
)


def inspect_f3_lithology_voxel_section_layout_results(
	config: F3SectionLayoutResultsConfig,
	*,
	model_id: str | None = None,
) -> F3SectionLayoutResultsInspection:
	"""Validate model manifests and recompute every paired-layout statistic."""
	roster = _load_roster(config)
	dataset_payload = validate_f3_lithology_voxel_section_layout_manifest(
		config.dataset_manifest
	)
	dataset_identity = _identity(config.dataset_manifest)
	contract_identity = _validate_recorded_file_identity(
		_mapping(
			_mapping(
				dataset_payload.get('source_identities'),
				'dataset source identities',
			).get('section_layout_contract'),
			'dataset section-layout contract identity',
		),
		label='dataset section-layout contract identity',
	)
	primary_models, load_models = _models_for_mode(
		config, roster=roster, model_id=model_id
	)
	jobs_by_model: dict[str, tuple[_LoadedJob, ...]] = {}
	manifest_identities: dict[str, object] = {}
	for model in load_models:
		manifest = _run_manifest_path(config, model.model_id)
		jobs_by_model[model.model_id] = _load_model_jobs(
			manifest,
			model=model,
			dataset_manifest_identity=dataset_identity,
		)
		manifest_identities[model.model_id] = _identity(manifest)
	comparisons = _comparisons(primary_models, roster=roster, loaded=jobs_by_model)
	paired_deltas, validated_pairs = _paired_deltas(
		comparisons, jobs_by_model=jobs_by_model
	)
	summary = _summary_by_size(paired_deltas)
	decisions = _model_decisions(
		primary_models,
		roster=roster,
		summary_by_size=summary,
		comparisons=comparisons,
	)
	job_metrics = tuple(
		_job_metric_row(job)
		for model in load_models
		for job in jobs_by_model[model.model_id]
	)
	return F3SectionLayoutResultsInspection(
		mode='model' if model_id is not None else 'final',
		requested_model_id=model_id,
		loaded_model_ids=tuple(model.model_id for model in load_models),
		job_metrics=job_metrics,
		paired_deltas=paired_deltas,
		summary_by_size=summary,
		model_decisions=decisions,
		pair_identity_validation={
			'status': 'PASS',
			'fields': list(PAIR_IDENTITY_FIELDS),
			'validated_pair_count': validated_pairs,
		},
		source_identities={
			'dataset_manifest': dataset_identity,
			'dataset_contract': contract_identity,
			'model_roster': _identity(config.model_roster),
			'model_run_manifests': manifest_identities,
		},
	)


def summarize_f3_lithology_voxel_section_layout_results(
	config: F3SectionLayoutResultsConfig,
	*,
	model_id: str | None = None,
	no_publish: bool = False,
) -> F3SectionLayoutResultsResult:
	"""Write an artifact-owned model summary or the exact final lightweight set."""
	if model_id is not None and not no_publish:
		raise ValueError('model mode requires --no-publish')
	if model_id is None and no_publish:
		raise ValueError('--no-publish is valid only with --model-id')
	inspection = inspect_f3_lithology_voxel_section_layout_results(
		config, model_id=model_id
	)
	output_dir = (
		config.report_dir
		if model_id is None
		else config.model_results_dir(model_id)
	)
	output_names = MODEL_OUTPUT_NAMES if model_id is not None else FINAL_OUTPUT_NAMES
	_write_results(
		output_dir,
		inspection=inspection,
		config=config,
		output_names=output_names,
	)
	files = tuple(output_dir / name for name in output_names)
	if {path.name for path in files} != set(output_names):
		raise AssertionError('internal summary output inventory mismatch')
	return F3SectionLayoutResultsResult(output_dir, files, inspection)


def decide_f3_lithology_voxel_section_layout_parent_status(
	summary_by_size: Sequence[Mapping[str, object]],
	*,
	comparison_id: str,
) -> Mapping[str, object]:
	"""Apply the fixed medium/large primary gate and class guardrail."""
	index = _gate_summary_index(summary_by_size, comparison_id=comparison_id)
	_validate_gate_summary_inventory(index, comparison_id=comparison_id)
	size_evidence = {
		size: _primary_size_evidence(index, data_size=size) for size in DATA_SIZES
	}
	degradations = _systematic_major_degradations(index)
	medium = size_evidence['medium']
	large = size_evidence['large']
	if degradations or (medium['negative'] and large['negative']):
		status = 'SECTION_LAYOUT_STOP'
	elif medium['positive'] and large['positive']:
		status = 'SECTION_LAYOUT_GO'
	else:
		status = 'SECTION_LAYOUT_HOLD'
	return {
		'status': status,
		'comparison_id': comparison_id,
		'positive_rule': 'mean>0 and median>0 and wins>=4/5 for both primary metrics',
		'negative_rule': 'mean<0 and median<0 and wins<=1/5 for both primary metrics',
		'size_evidence': size_evidence,
		'systematic_major_degradation': degradations,
	}


def _gate_summary_index(
	summary_by_size: Sequence[Mapping[str, object]], *, comparison_id: str
) -> Mapping[tuple[str, str], Mapping[str, object]]:
	return {
		(str(row['data_size']), str(row['metric'])): row
		for row in summary_by_size
		if row.get('comparison_id') == comparison_id
	}


def _validate_gate_summary_inventory(
	index: Mapping[tuple[str, str], Mapping[str, object]], *, comparison_id: str
) -> None:
	for size in DATA_SIZES:
		for metric in PRIMARY_METRICS:
			if (size, metric) not in index:
				raise ValueError(
					f'missing primary summary row: {comparison_id}/{size}/{metric}'
				)
		for class_id in MONITORED_CLASS_IDS:
			for metric in MONITORED_METRICS:
				name = f'class_{class_id}_{metric}'
				if (size, name) not in index:
					raise ValueError(
						'missing monitored-class summary row: '
						f'{comparison_id}/{size}/{name}'
					)


def _primary_size_evidence(
	index: Mapping[tuple[str, str], Mapping[str, object]], *, data_size: str
) -> Mapping[str, bool]:
	primary = [index[(data_size, metric)] for metric in PRIMARY_METRICS]
	return {
		'positive': all(
			float(row['mean_delta']) > 0.0
			and float(row['median_delta']) > 0.0
			and int(row['wins']) >= 4
			for row in primary
		),
		'negative': all(
			float(row['mean_delta']) < 0.0
			and float(row['median_delta']) < 0.0
			and int(row['wins']) <= 1
			for row in primary
		),
	}


def _systematic_major_degradations(
	index: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[Mapping[str, object]]:
	degradations = []
	for class_id in MONITORED_CLASS_IDS:
		for metric in MONITORED_METRICS:
			name = f'class_{class_id}_{metric}'
			bad_sizes = [
				size
				for size in DATA_SIZES
				if (size, name) in index
				and float(index[(size, name)]['mean_delta'])
				<= MAJOR_DEGRADATION_DELTA
			]
			if len(bad_sizes) >= SYSTEMATIC_SIZE_COUNT:
				degradations.append(
					{'class_id': class_id, 'metric': metric, 'data_sizes': bad_sizes}
				)
	return degradations


def _load_roster(config: F3SectionLayoutResultsConfig) -> F3SectionLayoutModelRoster:
	roster = f3_lithology_voxel_section_layout_model_roster_from_mapping(
		load_config(config.model_roster)
	)
	if roster.artifact_root != config.artifact_root:
		raise ValueError('model roster artifact_root differs from results config')
	return roster


def _models_for_mode(
	config: F3SectionLayoutResultsConfig,
	*,
	roster: F3SectionLayoutModelRoster,
	model_id: str | None,
) -> tuple[tuple[F3SectionLayoutModel, ...], tuple[F3SectionLayoutModel, ...]]:
	if model_id is None:
		missing = [
			model.model_id
			for model in roster.models
			if not _run_manifest_path(config, model.model_id).is_file()
		]
		if missing:
			raise FileNotFoundError(
				'final mode requires complete manifests for the full roster; '
				f'missing: {missing!r}'
			)
		return roster.models, roster.models
	if not isinstance(model_id, str) or not model_id:
		raise ValueError('model_id must be one exact non-empty roster ID')
	try:
		primary = roster.model_by_id[model_id]
	except KeyError as error:
		raise ValueError(f'unknown model_id: {model_id!r}') from error
	manifest = _run_manifest_path(config, model_id)
	if not manifest.is_file():
		raise FileNotFoundError(manifest)
	wanted = _reference_ids(primary)
	loaded_ids = {model_id}
	loaded_ids.update(
		reference
		for reference in wanted
		if _run_manifest_path(config, reference).is_file()
	)
	return (primary,), tuple(
		model for model in roster.models if model.model_id in loaded_ids
	)


def _load_model_jobs(
	manifest: Path,
	*,
	model: F3SectionLayoutModel,
	dataset_manifest_identity: Mapping[str, object],
) -> tuple[_LoadedJob, ...]:
	rows = load_f3_lithology_voxel_section_layout_rows(manifest)
	payload = _read_json(manifest)
	if payload.get('scientific_result') is not True:
		raise ValueError(f'run manifest is not a scientific result: {model.model_id}')
	if payload.get('model') != {
		'model_id': model.model_id,
		'model_tag': model.model_tag,
	}:
		raise ValueError(f'wrong run manifest model: {model.model_id}')
	if payload.get('dataset_manifest') != dataset_manifest_identity:
		raise ValueError(f'dataset manifest identity drift: {model.model_id}')
	expected = tuple((layout, size) for layout in LAYOUT_IDS for size in DATA_SIZES)
	keys = tuple((row.get('layout_id'), row.get('data_size')) for row in rows)
	if len(rows) != len(expected) or keys != expected:
		raise ValueError(
			f'{model.model_id} manifest must contain exact ordered 15 conditions'
		)
	if any(row.get('status') != 'complete' for row in rows):
		raise ValueError(f'{model.model_id} manifest contains incomplete jobs')
	loaded = []
	for row in rows:
		if (
			row.get('model_id') != model.model_id
			or row.get('model_tag') != model.model_tag
		):
			raise ValueError(f'wrong row model identity: {model.model_id}')
		paths = _canonical_metric_paths(row, model_id=model.model_id)
		metrics = load_f3_lithology_voxel_label_budget_evaluation_metrics(
			metrics_path=paths['metrics'],
			boundary_metrics_path=paths['boundary_metrics'],
			boundary_region_metrics_path=paths['boundary_region_metrics'],
			label=(
				f'{model.model_id}/{row["layout_id"]}/size={row["data_size"]}'
			),
		)
		if any(not math.isfinite(float(value)) for value in metrics.values()):
			raise ValueError(f'non-finite metric for model {model.model_id}')
		loaded.append(_LoadedJob(model=model, row=row, metrics=metrics))
	return tuple(loaded)


def _canonical_metric_paths(
	row: Mapping[str, object], *, model_id: str
) -> Mapping[str, Path]:
	records = _mapping(row.get('canonical_metrics_paths'), 'canonical metric paths')
	expected = {
		'metrics',
		'boundary_metrics',
		'boundary_region_metrics',
		'evaluation_metadata',
	}
	if set(records) != expected:
		raise ValueError(f'canonical metric path inventory drift: {model_id}')
	paths: dict[str, Path] = {}
	for name, value in records.items():
		identity = _mapping(value, f'{model_id} canonical metric {name}')
		if set(identity) != {'path', 'sha256'}:
			raise ValueError(f'canonical metric identity drift: {model_id}/{name}')
		path = Path(str(identity.get('path', '')))
		if not path.is_absolute() or not path.is_file():
			raise FileNotFoundError(path)
		if identity.get('sha256') != file_sha256(path):
			raise ValueError(f'canonical metric file hash drift: {model_id}/{name}')
		paths[name] = path
	return paths


def _reference_ids(model: F3SectionLayoutModel) -> tuple[str, ...]:
	values = ('mae', 'm1_k6', model.parent_model_id)
	return tuple(
		value
		for index, value in enumerate(values)
		if value is not None
		and value != model.model_id
		and value not in values[:index]
	)


def _comparisons(
	models: Sequence[F3SectionLayoutModel],
	*,
	roster: F3SectionLayoutModelRoster,
	loaded: Mapping[str, tuple[_LoadedJob, ...]],
) -> tuple[Mapping[str, object], ...]:
	comparisons = []
	for model in models:
		if model.model_id == 'mae':
			continue
		roles_by_reference: dict[str, list[str]] = {}
		for role, reference in (
			('mae', 'mae'),
			('m1_k6', 'm1_k6'),
			('parent', model.parent_model_id),
		):
			if (
				reference is None
				or reference == model.model_id
				or reference not in loaded
			):
				continue
			roles_by_reference.setdefault(reference, []).append(role)
		for reference, roles in roles_by_reference.items():
			if reference not in roster.model_by_id:
				raise AssertionError('closed roster comparison escaped the roster')
			comparisons.append(
				{
					'comparison_id': f'{model.model_id}-minus-{reference}',
					'model_id': model.model_id,
					'reference_model_id': reference,
					'comparison_roles': roles,
				}
			)
	return tuple(comparisons)


def _paired_deltas(
	comparisons: Sequence[Mapping[str, object]],
	*,
	jobs_by_model: Mapping[str, tuple[_LoadedJob, ...]],
) -> tuple[tuple[Mapping[str, object], ...], int]:
	rows = []
	validated_pair_count = 0
	for comparison in comparisons:
		model_id = str(comparison['model_id'])
		reference_id = str(comparison['reference_model_id'])
		candidate = {
			_job_key(job): job for job in jobs_by_model[model_id]
		}
		reference = {
			_job_key(job): job for job in jobs_by_model[reference_id]
		}
		if set(candidate) != set(reference):
			raise ValueError(
				f'paired condition mismatch: {comparison["comparison_id"]}'
			)
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				left = candidate[(layout_id, data_size)]
				right = reference[(layout_id, data_size)]
				_validate_pair_identity(left, right)
				validated_pair_count += 1
				for metric in METRIC_SPECS:
					delta = float(left.metrics[metric.name]) - float(
						right.metrics[metric.name]
					)
					if not math.isfinite(delta):
						raise ValueError('paired metric delta is non-finite')
					rows.append(
						{
							**comparison,
							'layout_id': layout_id,
							'data_size': data_size,
							'metric': metric.name,
							'higher_is_better': metric.higher_is_better,
							'model_value': float(left.metrics[metric.name]),
							'reference_value': float(right.metrics[metric.name]),
							'delta': delta,
						}
					)
	return tuple(rows), validated_pair_count


def _validate_pair_identity(candidate: _LoadedJob, reference: _LoadedJob) -> None:
	left = _pair_identity(candidate.row)
	right = _pair_identity(reference.row)
	for name in PAIR_IDENTITY_FIELDS:
		if left[name] != right[name]:
			raise ValueError(
				'paired identity mismatch for '
				f'{candidate.model.model_id} - {reference.model.model_id} at '
				f'{candidate.row["layout_id"]}/{candidate.row["data_size"]}: {name}'
			)


def _pair_identity(row: Mapping[str, object]) -> Mapping[str, object]:
	tiles = _mapping(row.get('tile_identities'), 'paired tile identities')
	return {
		'layout_id': row.get('layout_id'),
		'data_size': row.get('data_size'),
		'supervision_grid_identity': row.get('dataset_grid_identity'),
		'selected_token_identity': row.get('selected_token_identity_sha256'),
		'train_mask_identity': row.get('train_mask_sha256'),
		'validation_mask_identity': row.get('validation_mask_sha256'),
		'decoder_seed': row.get('decoder_seed'),
		'initial_decoder_state_identity': row.get('initial_decoder_state_sha256'),
		'class_weights': row.get('class_weights'),
		'sampling_sequence_identity': row.get('sampling_sequence_sha256'),
		'train_tile_identity': tiles.get('train'),
		'validation_tile_identity': tiles.get('validation'),
		'metric_schema_identity': row.get('metric_schema_sha256'),
	}


def _summary_by_size(
	paired_deltas: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
	comparison_order = tuple(
		dict.fromkeys(str(row['comparison_id']) for row in paired_deltas)
	)
	rows = []
	for comparison_id in comparison_order:
		comparison_rows = [
			row for row in paired_deltas if row['comparison_id'] == comparison_id
		]
		if not comparison_rows:
			continue
		identity = {
			key: comparison_rows[0][key]
			for key in (
				'comparison_id',
				'model_id',
				'reference_model_id',
				'comparison_roles',
			)
		}
		for data_size in DATA_SIZES:
			for metric in METRIC_SPECS:
				selected = [
					row
					for row in comparison_rows
					if row['data_size'] == data_size
					and row['metric'] == metric.name
				]
				if tuple(row['layout_id'] for row in selected) != LAYOUT_IDS:
					raise ValueError(
						f'summary requires five ordered layouts: {comparison_id}/'
						f'{data_size}/{metric.name}'
					)
				values = [float(row['delta']) for row in selected]
				wins = sum(
					value > 0.0 if metric.higher_is_better else value < 0.0
					for value in values
				)
				losses = sum(
					value < 0.0 if metric.higher_is_better else value > 0.0
					for value in values
				)
				rows.append(
					{
						**identity,
						'data_size': data_size,
						'metric': metric.name,
						'higher_is_better': metric.higher_is_better,
						'layout_count': 5,
						'mean_delta': statistics.fmean(values),
						'median_delta': statistics.median(values),
						'sample_standard_deviation': statistics.stdev(values),
						'wins': wins,
						'ties': sum(value == 0.0 for value in values),
						'losses': losses,
						'per_layout_delta': {
							str(row['layout_id']): float(row['delta'])
							for row in selected
						},
					}
				)
	return tuple(rows)


def _model_decisions(
	models: Sequence[F3SectionLayoutModel],
	*,
	roster: F3SectionLayoutModelRoster,
	summary_by_size: Sequence[Mapping[str, object]],
	comparisons: Sequence[Mapping[str, object]],
) -> Mapping[str, Mapping[str, object]]:
	by_id = {str(row['comparison_id']): row for row in comparisons}
	decisions: dict[str, Mapping[str, object]] = {}
	for model in models:
		base: dict[str, object] = {
			'model_id': model.model_id,
			'model_tag': model.model_tag,
			'parent_model_id': model.parent_model_id,
			'selection_role': model.selection_role,
			'metrics_included': True,
			'selection_eligible': model.selection_eligible,
		}
		if model.model_id == 'mae':
			base.update(
				{
					'formal_status_computed': False,
					'formal_status': 'NOT_APPLICABLE_BASELINE',
				}
			)
			decisions[model.model_id] = base
			continue
		comparison_id = f'{model.model_id}-minus-{model.parent_model_id}'
		if comparison_id not in by_id:
			base.update(
				{
					'formal_status_computed': False,
					'formal_status': 'NOT_AVAILABLE',
				}
			)
		else:
			formal = decide_f3_lithology_voxel_section_layout_parent_status(
				summary_by_size, comparison_id=comparison_id
			)
			base.update(
				{
					'formal_status_computed': True,
					'formal_status': formal['status'],
					'formal_parent_comparison': formal,
				}
			)
		if (
			model.selection_role == 'diagnostic'
			and base['selection_eligible'] is not False
		):
			raise AssertionError('diagnostic model became selection eligible')
		if model.model_id not in roster.model_by_id:
			raise AssertionError('decision escaped the closed roster')
		decisions[model.model_id] = base
	return decisions


def _job_metric_row(job: _LoadedJob) -> Mapping[str, object]:
	identity = _pair_identity(job.row)
	return {
		'model_id': job.model.model_id,
		'model_tag': job.model.model_tag,
		'layout_id': job.row['layout_id'],
		'data_size': job.row['data_size'],
		'target_train_voxel_count': job.row['target_train_voxel_count'],
		'actual_train_voxel_count': job.row['actual_train_voxel_count'],
		**identity,
		**job.metrics,
	}


def _write_results(
	output_dir: Path,
	*,
	inspection: F3SectionLayoutResultsInspection,
	config: F3SectionLayoutResultsConfig,
	output_names: tuple[str, ...],
) -> None:
	if output_dir.exists():
		raise FileExistsError(output_dir)
	if inspection.mode not in {'model', 'final'}:
		raise ValueError(f'unsupported results mode: {inspection.mode!r}')
	output_dir.mkdir(parents=True)
	try:
		if inspection.mode == 'model':
			_write_model_review(output_dir, inspection=inspection)
		else:
			_write_final_results(output_dir, inspection=inspection, config=config)
		_validate_output_inventory(output_dir, expected=output_names)
	except BaseException:
		for path in output_dir.iterdir():
			if path.is_file():
				path.unlink()
		output_dir.rmdir()
		raise


def _write_model_review(
	output_dir: Path, *, inspection: F3SectionLayoutResultsInspection
) -> None:
	if inspection.requested_model_id is None:
		raise ValueError('model review requires one requested model ID')
	_write_csv(output_dir / MODEL_OUTPUT_NAMES[0], inspection.job_metrics)
	_write_csv(output_dir / MODEL_OUTPUT_NAMES[1], inspection.paired_deltas)
	review = _model_review_payload(inspection)
	_write_json(output_dir / MODEL_OUTPUT_NAMES[2], review)
	(output_dir / MODEL_OUTPUT_NAMES[3]).write_text(
		_render_markdown(review), encoding='utf-8'
	)


def _write_final_results(
	output_dir: Path,
	*,
	inspection: F3SectionLayoutResultsInspection,
	config: F3SectionLayoutResultsConfig,
) -> None:
	if inspection.requested_model_id is not None:
		raise ValueError('final results cannot have a requested model ID')
	if inspection.loaded_model_ids != EXPECTED_MODEL_IDS:
		raise ValueError('final results require the exact full model roster')
	expected_job_count = len(EXPECTED_MODEL_IDS) * len(LAYOUT_IDS) * len(DATA_SIZES)
	if len(inspection.job_metrics) != expected_job_count:
		raise ValueError('final results require all 210 completed jobs')
	_write_csv(output_dir / FINAL_OUTPUT_NAMES[0], inspection.job_metrics)
	_write_csv(output_dir / FINAL_OUTPUT_NAMES[1], inspection.paired_deltas)
	_write_csv(output_dir / FINAL_OUTPUT_NAMES[2], inspection.summary_by_size)
	_write_json(
		output_dir / FINAL_OUTPUT_NAMES[3],
		{
			'artifact_type': 'f3_lithology_voxel_section_layout_model_decisions',
			'schema_version': RESULTS_SCHEMA_VERSION,
			'model_decisions': inspection.model_decisions,
		},
	)
	summary = _summary_payload(inspection)
	_write_json(output_dir / FINAL_OUTPUT_NAMES[4], summary)
	(output_dir / FINAL_OUTPUT_NAMES[5]).write_text(
		_render_markdown(summary), encoding='utf-8'
	)
	_write_json(
		output_dir / FINAL_OUTPUT_NAMES[6],
		_handoff_payload(inspection, config=config),
	)


def _validate_output_inventory(
	output_dir: Path, *, expected: Sequence[str]
) -> None:
	actual = {path.name for path in output_dir.iterdir() if path.is_file()}
	if actual != set(expected):
		raise AssertionError('summary wrote a non-canonical output inventory')


def _model_review_payload(
	inspection: F3SectionLayoutResultsInspection,
) -> Mapping[str, object]:
	payload = dict(_summary_payload(inspection))
	payload.update(
		{
			'artifact_type': MODEL_REVIEW_ARTIFACT_TYPE,
			'status': 'COMPLETE',
			'scope': 'single_model',
			'benchmark_complete': False,
			'reviewed_model_id': inspection.requested_model_id,
			'loaded_model_ids': list(inspection.loaded_model_ids),
			'summary_by_size': list(inspection.summary_by_size),
		}
	)
	return payload


def _summary_payload(
	inspection: F3SectionLayoutResultsInspection,
) -> Mapping[str, object]:
	return {
		'artifact_type': RESULTS_ARTIFACT_TYPE,
		'schema_version': RESULTS_SCHEMA_VERSION,
		'mode': inspection.mode,
		'requested_model_id': inspection.requested_model_id,
		'statistical_unit': 'layout_id',
		'completed_model_count': len(inspection.loaded_model_ids),
		'completed_job_count': len(inspection.job_metrics),
		'comparison_count': len(
			{str(row['comparison_id']) for row in inspection.paired_deltas}
		),
		'formal_gate': {
			'primary_metrics': list(PRIMARY_METRICS),
			'monitored_class_ids': list(MONITORED_CLASS_IDS),
			'major_degradation_delta': MAJOR_DEGRADATION_DELTA,
			'systematic_data_size_count': SYSTEMATIC_SIZE_COUNT,
			'small_is_diagnostic': True,
		},
		'pair_identity_validation': inspection.pair_identity_validation,
		'model_decisions': inspection.model_decisions,
		'source_identities': inspection.source_identities,
		'claims_not_made': ['p_value', 'confidence_interval', 'voxel_independence'],
		'project_adoption': 'PENDING_REVIEW',
	}


def _handoff_payload(
	inspection: F3SectionLayoutResultsInspection,
	*,
	config: F3SectionLayoutResultsConfig,
) -> Mapping[str, object]:
	return {
		'artifact_type': HANDOFF_ARTIFACT_TYPE,
		'schema_version': RESULTS_SCHEMA_VERSION,
		'status': 'PASS',
		'scope': 'full_roster',
		'benchmark_complete': True,
		'dataset_contract_identity': inspection.source_identities['dataset_contract'],
		'model_roster_identity': inspection.source_identities['model_roster'],
		'completed_model_count': len(inspection.loaded_model_ids),
		'completed_job_count': len(inspection.job_metrics),
		'pair_identity_validation': inspection.pair_identity_validation,
		'model_decisions': inspection.model_decisions,
		'diagnostic_eligibility': {
			model_id: {
				'metrics_included': decision['metrics_included'],
				'formal_status_computed': decision['formal_status_computed'],
				'selection_eligible': decision['selection_eligible'],
			}
			for model_id, decision in inspection.model_decisions.items()
			if decision['selection_role'] == 'diagnostic'
		},
		'execution_git_state': _git_state(config.workspace_root),
		'project_adoption': 'PENDING_REVIEW',
	}


def _git_state(workspace_root: Path) -> Mapping[str, object]:
	def run(*args: str) -> str:
		result = subprocess.run(  # noqa: S603
			['git', '-C', str(workspace_root), *args],  # noqa: S607
			check=True,
			capture_output=True,
			text=True,
		)
		return result.stdout.rstrip('\n')

	commit = run('rev-parse', 'HEAD')
	status = run('status', '--porcelain=v1', '--untracked-files=all')
	return {
		'commit': commit,
		'dirty': bool(status),
		'status_porcelain_sha256': hashlib.sha256(status.encode()).hexdigest(),
	}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(
				{
					key: _csv_value(row.get(key))
					for key in fieldnames
				}
			)


def _csv_value(value: object) -> object:
	if isinstance(value, Mapping | list | tuple):
		return json.dumps(value, sort_keys=True, separators=(',', ':'))
	return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	temporary = path.with_name(f'.{path.name}.tmp')
	temporary.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temporary.replace(path)


def _render_markdown(payload: Mapping[str, object]) -> str:
	decisions = cast('Mapping[str, Mapping[str, object]]', payload['model_decisions'])
	title = (
		'# F3 section-layout model review'
		if payload.get('scope') == 'single_model'
		else '# F3 section-layout benchmark summary'
	)
	lines = [
		title,
		'',
		'Layouts are the paired statistical units. No p-values, confidence intervals,',
		'or voxel-independence claims are produced.',
		'',
		'| model | role | parent status | selection eligible |',
		'|---|---|---|---:|',
	]
	for model_id, decision in decisions.items():
		lines.append(
			f'| {model_id} | {decision["selection_role"]} | '
			f'{decision["formal_status"]} | '
			f'{str(decision["selection_eligible"]).lower()} |'
		)
	lines.extend(
		[
			'',
			'`small` is diagnostic and cannot establish GO by itself. Project adoption',
			'remains `PENDING_REVIEW`; no global winner is selected automatically.',
			'',
		]
	)
	return '\n'.join(lines)


def _identity(path: Path) -> dict[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {'path': str(path.resolve()), 'sha256': file_sha256(path)}


def _validate_recorded_file_identity(
	identity: Mapping[str, object], *, label: str
) -> Mapping[str, object]:
	if set(identity) != {'path', 'sha256'}:
		raise ValueError(f'{label} key inventory mismatch')
	path = Path(str(identity.get('path', '')))
	if not path.is_absolute() or not path.is_file():
		raise FileNotFoundError(path)
	if identity.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} SHA-256 mismatch')
	return {'path': str(path.resolve()), 'sha256': identity['sha256']}


def _read_json(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as handle:
		value = json.load(handle)
	if not isinstance(value, Mapping):
		raise TypeError(f'JSON root must be a mapping: {path}')
	return cast('Mapping[str, object]', value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return cast('Mapping[str, object]', value)


def _run_manifest_path(config: F3SectionLayoutResultsConfig, model_id: str) -> Path:
	return config.benchmark_root / 'runs' / f'model={model_id}' / RUN_MANIFEST_NAME


def _job_key(job: _LoadedJob) -> tuple[str, str]:
	return str(job.row['layout_id']), str(job.row['data_size'])


__all__ = [
	'FINAL_OUTPUT_NAMES',
	'FORMAL_STATUSES',
	'MODEL_OUTPUT_NAMES',
	'F3SectionLayoutResultsConfig',
	'F3SectionLayoutResultsInspection',
	'F3SectionLayoutResultsResult',
	'decide_f3_lithology_voxel_section_layout_parent_status',
	'f3_lithology_voxel_section_layout_results_config_from_mapping',
	'inspect_f3_lithology_voxel_section_layout_results',
	'resolve_f3_lithology_voxel_section_layout_results_config',
	'summarize_f3_lithology_voxel_section_layout_results',
]
