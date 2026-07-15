"""Paired-seed aggregation for the F3 low-label voxel benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	BOUNDARY_METRICS_JSON,
	BOUNDARY_REGION_METRICS_CSV,
	EVALUATION_METADATA_JSON,
	METRICS_JSON,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget import (
	MANIFEST_ARTIFACT_TYPE as DATASET_MANIFEST_ARTIFACT_TYPE,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget import (
	validate_voxel_label_budget_condition_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.models.voxel_decoder.spec import (
	validate_voxel_decoder_architecture_mapping,
)
from seis_ssl_cluster.results import (
	PublishItem,
	PublishManifest,
	publish_manifest_to_dict,
	publish_selected_results,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_results import (
		F3VoxelLabelBudgetDecisionThresholds,
		F3VoxelLabelBudgetResultsConfig,
	)

RUN_MANIFEST_ARTIFACT_TYPE = 'f3_lithology_voxel_label_budget_run_manifest'
SUMMARY_ARTIFACT_TYPE = 'f3_lithology_voxel_label_budget_results_summary'
SCHEMA_VERSION = 1
REQUIRED_BUDGETS = ('cap25', 'cap50', 'cap100')
REQUIRED_SEEDS = (0, 1, 2, 3, 4)
MODEL_ROLES = ('mae', 'm1', 'm2a')
MODEL_LABELS = {'mae': 'MAE', 'm1': 'M1', 'm2a': 'M2-A'}
EXPECTED_MODEL_TAGS = {
	'mae': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
	'm1': 'strat_hmm_pretext_m1_k6_topblock1_distill',
	'm2a': 'strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill',
}
COMPARISONS = (
	('m1_vs_mae', 'mae', 'm1', 'M1 - MAE'),
	('m2a_vs_mae', 'mae', 'm2a', 'M2-A - MAE'),
	('m2a_vs_m1', 'm1', 'm2a', 'M2-A - M1'),
)
MONITORED_CLASS_IDS = (3, 5)
BOUNDARY_RADII = (2, 4)
BOUNDARY_TOLERANCES = (2, 4)
SUMMARY_JSON = 'voxel_label_budget_summary.json'
SUMMARY_MARKDOWN = 'voxel_label_budget_summary.md'
README_NAME = 'README.md'
LOCAL_PUBLISH_MANIFEST = 'publish_manifest.json'
TABLE_NAMES = (
	'job_metrics.csv',
	'paired_metrics.csv',
	'paired_deltas.csv',
	'summary_by_budget.csv',
	'monitored_class_summary.csv',
	'full_label_anchor.csv',
)
FIGURE_NAMES = (
	'macro_f1_by_budget.png',
	'mean_iou_by_budget.png',
	'balanced_accuracy_by_budget.png',
	'boundary_metrics_by_budget.png',
)
PUBLISH_SUFFIXES = frozenset({'.md', '.json', '.csv', '.png'})
FORBIDDEN_PUBLISH_SUFFIXES = frozenset(
	{'.pt', '.pth', '.npy', '.npz', '.joblib', '.pkl', '.sgy', '.segy'}
)


@dataclass(frozen=True)
class MetricSpec:
	"""One scalar metric and its improvement direction."""

	name: str
	higher_is_better: bool = True


METRIC_SPECS = (
	MetricSpec('macro_f1'),
	MetricSpec('mean_iou'),
	MetricSpec('balanced_accuracy'),
	MetricSpec('accuracy'),
	MetricSpec('weighted_f1'),
	*(
		MetricSpec(f'boundary_region_macro_f1_r{radius}')
		for radius in BOUNDARY_RADII
	),
	*(
		MetricSpec(f'boundary_region_mean_iou_r{radius}')
		for radius in BOUNDARY_RADII
	),
	*(MetricSpec(f'boundary_f1_t{value}') for value in BOUNDARY_TOLERANCES),
	MetricSpec('vertical_boundary_position_mae', higher_is_better=False),
	*(
		MetricSpec(name)
		for class_id in MONITORED_CLASS_IDS
		for name in (
			f'class_{class_id}_f1',
			f'class_{class_id}_iou',
			f'class_{class_id}_boundary_recall_t2',
			f'class_{class_id}_boundary_recall_t4',
		)
	),
)
METRIC_BY_NAME = {item.name: item for item in METRIC_SPECS}


@dataclass(frozen=True)
class _FileIdentity:
	path: Path
	sha256: str


@dataclass(frozen=True)
class _DatasetCondition:
	budget_id: str
	per_class_cap: int
	subsample_seed: int
	root: Path
	grid: _FileIdentity
	metadata: _FileIdentity
	selected_token_identity_sha256: str
	unique_token_xyz_sha256: str
	train_voxel_count: int
	validation_voxel_count: int
	class_order: tuple[int, ...]
	validation_mask_sha256: str
	canonical_full_grid_sha256: str


@dataclass(frozen=True)
class _Evaluation:
	model_tag: str
	class_order: tuple[int, ...]
	validation_voxel_count: int
	decoder_architecture: Mapping[str, object]
	metric_schema_sha256: str
	metrics: Mapping[str, float]


@dataclass(frozen=True)
class _LoadedJob:
	row: Mapping[str, object]
	dataset: _DatasetCondition
	model_role: str
	model_tag: str
	decoder_seed: int
	class_weights: tuple[float, ...]
	canonical_valid_tokens_sha256: str
	initial_model_state_sha256: str
	sampling_mode: str
	steps_per_epoch: int
	sampling_sequence_sha256: str
	train_tile_manifest_sha256: str
	validation_tile_manifest_sha256: str
	train_tile_identity_sha256: str
	validation_tile_identity_sha256: str
	evaluation: _Evaluation


@dataclass(frozen=True)
class F3VoxelLabelBudgetResultsInspection:
	"""Validated inputs and independently recomputed paired summaries."""

	job_metrics: tuple[Mapping[str, object], ...]
	paired_metrics: tuple[Mapping[str, object], ...]
	paired_deltas: tuple[Mapping[str, object], ...]
	summary_by_budget: tuple[Mapping[str, object], ...]
	monitored_class_summary: tuple[Mapping[str, object], ...]
	full_label_anchor: tuple[Mapping[str, object], ...]
	decisions: Mapping[str, object]
	source_identities: Mapping[str, object]
	decoder_architecture: Mapping[str, object]


@dataclass(frozen=True)
class F3VoxelLabelBudgetResultsResult:
	"""All summary-owned outputs."""

	summary_json: Path
	summary_markdown: Path
	readme: Path
	table_paths: tuple[Path, ...]
	figure_paths: tuple[Path, ...]
	local_publish_manifest: Path
	decisions: Mapping[str, object]
	publish_manifest: PublishManifest | None = None


def inspect_f3_lithology_voxel_label_budget_results(
	config: F3VoxelLabelBudgetResultsConfig,
) -> F3VoxelLabelBudgetResultsInspection:
	"""Validate 15 datasets and 45 jobs, then recompute paired deltas."""
	datasets, dataset_manifest_identity = _load_dataset_manifest(
		config.dataset_manifest
	)
	jobs, run_manifest_identity = _load_run_manifest(config.run_manifest, datasets)
	job_metrics = tuple(_job_metric_row(job) for job in jobs)
	paired_metrics = tuple(_paired_metric_rows(jobs))
	paired_deltas = tuple(_paired_delta_rows(jobs))
	summary = tuple(_summary_rows(paired_deltas))
	monitored = tuple(_monitored_summary_rows(summary))
	anchors = tuple(_load_full_label_anchors(config, jobs))
	decisions = _scientific_decisions(summary, config.decision)
	architectures = {_stable_json(job.evaluation.decoder_architecture) for job in jobs}
	if len(architectures) != 1:
		raise ValueError('decoder architecture mismatch across the 45 jobs')
	return F3VoxelLabelBudgetResultsInspection(
		job_metrics=job_metrics,
		paired_metrics=paired_metrics,
		paired_deltas=paired_deltas,
		summary_by_budget=summary,
		monitored_class_summary=monitored,
		full_label_anchor=anchors,
		decisions=decisions,
		source_identities={
			'dataset_manifest': dataset_manifest_identity,
			'run_manifest': run_manifest_identity,
			'full_label_evaluations': {
				role: str(path)
				for role, path in config.full_label_evaluations.items()
			},
		},
		decoder_architecture=dict(jobs[0].evaluation.decoder_architecture),
	)


def summarize_f3_lithology_voxel_label_budget_results(
	config: F3VoxelLabelBudgetResultsConfig,
) -> F3VoxelLabelBudgetResultsResult:
	"""Write the complete paired-seed summary and optional lightweight publish."""
	inspection = inspect_f3_lithology_voxel_label_budget_results(config)
	_validate_output_availability(config)
	reports = config.reports_dir
	tables = reports / 'tables'
	figures = reports / 'figures'
	tables.mkdir(parents=True, exist_ok=True)
	figures.mkdir(parents=True, exist_ok=True)
	table_paths = _write_tables(tables, inspection)
	figure_paths = _write_figures(figures, inspection)
	payload = _summary_payload(config, inspection)
	summary_json = reports / SUMMARY_JSON
	summary_markdown = reports / SUMMARY_MARKDOWN
	readme = reports / README_NAME
	_write_json(summary_json, payload)
	summary_markdown.write_text(_render_markdown(payload), encoding='utf-8')
	readme.write_text(_render_readme(), encoding='utf-8')
	result = F3VoxelLabelBudgetResultsResult(
		summary_json=summary_json,
		summary_markdown=summary_markdown,
		readme=readme,
		table_paths=table_paths,
		figure_paths=figure_paths,
		local_publish_manifest=reports / LOCAL_PUBLISH_MANIFEST,
		decisions=inspection.decisions,
	)
	publish_manifest = _publish(result, config)
	_write_local_publish_manifest(result, config, publish_manifest)
	return replace(result, publish_manifest=publish_manifest)


def _load_dataset_manifest(  # noqa: C901, PLR0912
	path: Path,
) -> tuple[dict[tuple[str, int], _DatasetCondition], Mapping[str, str]]:
	payload = _read_json(path)
	if payload.get('artifact_type') != DATASET_MANIFEST_ARTIFACT_TYPE:
		raise ValueError('voxel label-budget dataset manifest artifact_type mismatch')
	if payload.get('schema_version') != SCHEMA_VERSION:
		raise ValueError('voxel label-budget dataset manifest schema_version mismatch')
	contract = _mapping(payload.get('contract'), 'dataset manifest contract')
	if contract.get('budgets') != list(REQUIRED_BUDGETS):
		raise ValueError('dataset manifest budget contract mismatch')
	if contract.get('subsample_seeds') != list(REQUIRED_SEEDS):
		raise ValueError('dataset manifest seed contract mismatch')
	models = _mapping(payload.get('models'), 'dataset manifest models')
	if models != EXPECTED_MODEL_TAGS:
		raise ValueError('dataset manifest model identity mismatch')
	rows = _mapping_rows(payload.get('rows'), 'dataset manifest rows')
	expected = {
		(budget, seed) for budget in REQUIRED_BUDGETS for seed in REQUIRED_SEEDS
	}
	if payload.get('condition_count') != len(expected) or len(rows) != len(expected):
		raise ValueError('dataset manifest must contain exactly 15 conditions')
	result: dict[tuple[str, int], _DatasetCondition] = {}
	for index, row in enumerate(rows):
		condition = _dataset_condition(row, index=index)
		key = (condition.budget_id, condition.subsample_seed)
		if key in result:
			raise ValueError(f'duplicate dataset condition: {key!r}')
		result[key] = condition
	if set(result) != expected:
		missing = sorted(expected - set(result))
		extra = sorted(set(result) - expected)
		raise ValueError(
			'dataset manifest condition matrix mismatch: '
			f'missing={missing}, extra={extra}'
		)
	validation_hashes = {item.validation_mask_sha256 for item in result.values()}
	validation_counts = {item.validation_voxel_count for item in result.values()}
	class_orders = {item.class_order for item in result.values()}
	canonical_grids = {
		item.canonical_full_grid_sha256 for item in result.values()
	}
	if len(validation_hashes) != 1:
		raise ValueError('validation mask SHA-256 differs across the 15 datasets')
	if len(validation_counts) != 1:
		raise ValueError('validation voxel count differs across the 15 datasets')
	if len(class_orders) != 1:
		raise ValueError('class order differs across the 15 datasets')
	if len(canonical_grids) != 1:
		raise ValueError('canonical full grid differs across the 15 datasets')
	if payload.get('common_validation_mask_sha256') not in validation_hashes:
		raise ValueError('dataset manifest common validation identity mismatch')
	return result, {'path': str(path), 'sha256': file_sha256(path)}


def _dataset_condition(  # noqa: C901
	row: Mapping[str, object], *, index: int
) -> _DatasetCondition:
	budget = _required_budget(row.get('budget_id'), f'dataset row {index}.budget_id')
	seed = _required_seed(row.get('subsample_seed'), f'dataset row {index}.seed')
	cap = _positive_int(row.get('per_class_cap'), f'dataset row {index}.per_class_cap')
	if cap != int(budget.removeprefix('cap')):
		raise ValueError(f'dataset row {index} budget/cap mismatch')
	root = _path_value(row.get('voxel_dataset_root'), f'dataset row {index}.root')
	if not root.is_dir():
		raise FileNotFoundError(f'missing voxel dataset root: {root}')
	grid = _validated_identity(
		row.get('supervision_split_grid'),
		label=f'dataset row {index}.supervision_split_grid',
	)
	metadata = _validated_identity(
		row.get('voxel_label_budget_metadata'),
		label=f'dataset row {index}.voxel_label_budget_metadata',
	)
	if grid.path.resolve(strict=False) != (
		root / 'supervision_split_grid.npy'
	).resolve(strict=False):
		raise ValueError(f'dataset row {index} grid path does not match root')
	if metadata.path.resolve(strict=False) != (
		root / 'voxel_label_budget_metadata.json'
	).resolve(strict=False):
		raise ValueError(f'dataset row {index} metadata path does not match root')
	detail = validate_voxel_label_budget_condition_artifact(root)
	if detail.get('artifact_type') != 'f3_lithology_voxel_label_budget_dataset':
		raise ValueError(f'dataset row {index} metadata artifact_type mismatch')
	if detail.get('schema_version') != SCHEMA_VERSION:
		raise ValueError(f'dataset row {index} metadata schema_version mismatch')
	identity = _mapping(detail.get('identity'), f'dataset row {index} identity')
	selected = _sha256_value(
		row.get('selected_token_identity_sha256'),
		f'dataset row {index}.selected_token_identity_sha256',
	)
	unique = _sha256_value(
		row.get('unique_token_xyz_sha256'),
		f'dataset row {index}.unique_token_xyz_sha256',
	)
	validation_mask = _sha256_value(
		row.get('validation_mask_sha256'),
		f'dataset row {index}.validation_mask_sha256',
	)
	for label, expected, value in (
		('budget_id', budget, identity.get('budget_id')),
		('per_class_cap', cap, identity.get('per_class_cap')),
		('subsample_seed', seed, identity.get('subsample_seed')),
		(
			'selected_token_identity_sha256',
			selected,
			identity.get('selected_token_identity_sha256'),
		),
		(
			'unique_token_xyz_sha256',
			unique,
			identity.get('unique_token_xyz_sha256'),
		),
		(
			'validation_mask_sha256',
			validation_mask,
			identity.get('validation_mask_sha256'),
		),
	):
		if value != expected:
			raise ValueError(f'dataset row {index} {label} metadata mismatch')
	train_count = _positive_int(
		row.get('train_voxel_count'), f'dataset row {index}.train_voxel_count'
	)
	validation_count = _positive_int(
		row.get('validation_voxel_count'),
		f'dataset row {index}.validation_voxel_count',
	)
	if identity.get('actual_train_voxel_count') != train_count:
		raise ValueError(f'dataset row {index} train voxel metadata mismatch')
	if identity.get('validation_voxel_count') != validation_count:
		raise ValueError(f'dataset row {index} validation voxel metadata mismatch')
	class_order = _integer_tuple(
		row.get('class_order'), f'dataset row {index}.class_order'
	)
	if list(class_order) != identity.get('class_order'):
		raise ValueError(f'dataset row {index} class order metadata mismatch')
	sources = _mapping(detail.get('sources'), f'dataset row {index} sources')
	common_grid = _validated_identity(
		sources.get('common_grid'), label=f'dataset row {index} common grid'
	)
	return _DatasetCondition(
		budget_id=budget,
		per_class_cap=cap,
		subsample_seed=seed,
		root=root,
		grid=grid,
		metadata=metadata,
		selected_token_identity_sha256=selected,
		unique_token_xyz_sha256=unique,
		train_voxel_count=train_count,
		validation_voxel_count=validation_count,
		class_order=class_order,
		validation_mask_sha256=validation_mask,
		canonical_full_grid_sha256=common_grid.sha256,
	)


def _load_run_manifest(  # noqa: C901, PLR0912
	path: Path,
	datasets: Mapping[tuple[str, int], _DatasetCondition],
) -> tuple[tuple[_LoadedJob, ...], Mapping[str, str]]:
	payload = _read_json(path)
	if payload.get('artifact_type') != RUN_MANIFEST_ARTIFACT_TYPE:
		raise ValueError('voxel label-budget run manifest artifact_type mismatch')
	if payload.get('schema_version') != SCHEMA_VERSION:
		raise ValueError('voxel label-budget run manifest schema_version mismatch')
	contract = _mapping(
		payload.get('preregistered_contract'), 'run manifest preregistered_contract'
	)
	if contract.get('budgets') != list(REQUIRED_BUDGETS):
		raise ValueError('run manifest preregistered budget contract mismatch')
	if contract.get('subsample_seeds') != list(REQUIRED_SEEDS):
		raise ValueError('run manifest preregistered seed contract mismatch')
	if contract.get('model_order') != list(MODEL_ROLES):
		raise ValueError('run manifest preregistered model-order mismatch')
	if contract.get('epochs') != 50:
		raise ValueError('run manifest preregistered epoch contract mismatch')
	if contract.get('sampling_mode') != 'uniform_tiles_with_replacement':
		raise ValueError('run manifest preregistered sampling contract mismatch')
	contract_steps = _positive_int(
		contract.get('steps_per_epoch'),
		'run manifest preregistered steps_per_epoch',
	)
	rows = _mapping_rows(payload.get('rows'), 'run manifest rows')
	expected = {
		(budget, seed, role)
		for budget in REQUIRED_BUDGETS
		for seed in REQUIRED_SEEDS
		for role in MODEL_ROLES
	}
	if (
		payload.get('row_count') != len(expected)
		or payload.get('complete_count') != len(expected)
		or len(rows) != len(expected)
	):
		raise ValueError('run manifest must contain exactly 45 rows')
	jobs: dict[tuple[str, int, str], _LoadedJob] = {}
	for index, row in enumerate(rows):
		job = _load_job(row, index=index, datasets=datasets)
		if job.steps_per_epoch != contract_steps:
			raise ValueError('run row steps_per_epoch changed after preregistration')
		key = (
			job.dataset.budget_id,
			job.dataset.subsample_seed,
			job.model_role,
		)
		if key in jobs:
			raise ValueError(f'duplicate run manifest row: {key!r}')
		jobs[key] = job
	if set(jobs) != expected:
		missing = sorted(expected - set(jobs))
		extra = sorted(set(jobs) - expected)
		raise ValueError(
			'45-job run matrix mismatch: '
			f'missing={missing}, extra={extra}'
		)
	ordered = tuple(
		jobs[(budget, seed, role)]
		for budget in REQUIRED_BUDGETS
		for seed in REQUIRED_SEEDS
		for role in MODEL_ROLES
	)
	for budget in REQUIRED_BUDGETS:
		for seed in REQUIRED_SEEDS:
			_validate_triplet(
				tuple(jobs[(budget, seed, role)] for role in MODEL_ROLES)
			)
	return ordered, {'path': str(path), 'sha256': file_sha256(path)}


def _load_job(  # noqa: C901, PLR0912, PLR0915
	row: Mapping[str, object],
	*,
	index: int,
	datasets: Mapping[tuple[str, int], _DatasetCondition],
) -> _LoadedJob:
	label = f'run row {index}'
	budget = _required_budget(row.get('budget_id'), f'{label}.budget_id')
	seed = _required_seed(row.get('subsample_seed'), f'{label}.subsample_seed')
	role = _model_role(row.get('model_role'), f'{label}.model_role')
	dataset = datasets[(budget, seed)]
	cap = _positive_int(row.get('per_class_cap'), f'{label}.per_class_cap')
	if cap != dataset.per_class_cap:
		raise ValueError(f'{label} per_class_cap does not match dataset manifest')
	model_tag = _non_empty_string(row.get('model_tag'), f'{label}.model_tag')
	if model_tag != EXPECTED_MODEL_TAGS[role]:
		raise ValueError(f'{label} model identity mismatch for {role}')
	status = _non_empty_string(row.get('status'), f'{label}.status').lower()
	if status not in {'complete', 'completed'}:
		raise ValueError(f'{label} is not complete')
	if row.get('action') not in {'NEW', 'RESUMED', 'REUSED'}:
		raise ValueError(f'{label} completion action is invalid')
	_validate_dataset_binding(row, dataset=dataset, label=label)
	_repeated_dataset_identity(row, dataset=dataset, label=label)
	for field, expected in (
		('train_voxel_count', dataset.train_voxel_count),
		('validation_voxel_count', dataset.validation_voxel_count),
	):
		if _positive_int(row.get(field), f'{label}.{field}') != expected:
			raise ValueError(f'{label} {field} does not match dataset manifest')
	decoder_seed = _nonnegative_int(row.get('decoder_seed'), f'{label}.decoder_seed')
	if decoder_seed != 42000 + seed:
		raise ValueError(f'{label} decoder seed policy mismatch')
	class_weights = _float_tuple(row.get('class_weights'), f'{label}.class_weights')
	if len(class_weights) != len(dataset.class_order):
		raise ValueError(f'{label} class_weights length mismatch')
	initial = _row_sha256(
		row,
		('initial_model_state_sha256', 'initial_state_sha256'),
		label=f'{label}.initial_model_state_sha256',
	)
	valid_tokens = _row_sha256(
		row,
		(
			'canonical_valid_tokens_sha256',
			'canonical_valid_token_sha256',
			'valid_tokens_sha256',
			'source_valid_tokens',
		),
		label=f'{label}.canonical_valid_tokens_sha256',
	)
	sampling_mode, steps_per_epoch = _sampling_contract(row, label=label)
	if sampling_mode != 'uniform_tiles_with_replacement':
		raise ValueError(f'{label} sampling mode is not replacement sampling')
	sampling_sequence = _row_sha256(
		row,
		('sampling_sequence_sha256',),
		label=f'{label}.sampling_sequence_sha256',
	)
	train_tiles = _row_sha256(
		row,
		('train_tile_manifest_sha256', 'train_tile_manifest'),
		label=f'{label}.train_tile_manifest',
	)
	validation_tiles = _row_sha256(
		row,
		('validation_tile_manifest_sha256', 'validation_tile_manifest'),
		label=f'{label}.validation_tile_manifest',
	)
	train_tile_identity = _row_sha256(
		row,
		('train_tile_identity_sha256',),
		label=f'{label}.train_tile_identity_sha256',
	)
	validation_tile_identity = _row_sha256(
		row,
		('validation_tile_identity_sha256',),
		label=f'{label}.validation_tile_identity_sha256',
	)
	global_step = _positive_int(row.get('global_step'), f'{label}.global_step')
	if global_step != steps_per_epoch * 50:
		raise ValueError(f'{label} global_step does not equal 50 complete epochs')
	latest = _row_file_identity(
		row,
		('latest', 'latest_checkpoint', 'latest_path'),
		label=f'{label}.latest',
	)
	best = _row_file_identity(
		row,
		('best', 'best_checkpoint', 'best_path'),
		label=f'{label}.best',
	)
	best_epoch = _nonnegative_int(
		row.get('best_selection_epoch'), f'{label}.best_selection_epoch'
	)
	if best_epoch >= 50:
		raise ValueError(f'{label} best selection epoch is outside [0, 49]')
	best_metrics = _mapping(
		row.get('best_selection_metrics'), f'{label}.best_selection_metrics'
	)
	for metric in ('macro_f1', 'mean_iou'):
		_finite_float(best_metrics.get(metric), f'{label}.best_selection.{metric}')
	prediction_metadata = _row_file_identity(
		row,
		('prediction_metadata', 'prediction_metadata_path'),
		label=f'{label}.prediction_metadata',
	)
	if row.get('prediction_checkpoint_kind') != 'best':
		raise ValueError(f'{label} prediction checkpoint kind is not best')
	evaluation_metadata = _row_file_identity(
		row,
		('evaluation_metadata', 'evaluation_metadata_path'),
		label=f'{label}.evaluation_metadata',
	)
	metrics = _row_file_identity(
		row,
		('evaluation_metrics', 'evaluation_metrics_path', 'metrics_json'),
		label=f'{label}.evaluation_metrics',
	)
	boundary = _row_file_identity(
		row,
		(
			'evaluation_boundary_metrics',
			'evaluation_boundary_metrics_path',
			'boundary_metrics_json',
		),
		label=f'{label}.evaluation_boundary_metrics',
	)
	regions = _row_file_identity(
		row,
		(
			'evaluation_boundary_region_metrics',
			'evaluation_boundary_region_metrics_path',
			'boundary_region_metrics_csv',
		),
		label=f'{label}.evaluation_boundary_region_metrics',
	)
	_row_file_identity(
		row,
		('report', 'report_path'),
		label=f'{label}.report',
	)
	if _nonnegative_int(
		row.get('uncovered_validation_voxel_count'),
		f'{label}.uncovered_validation_voxel_count',
	) != 0:
		raise ValueError(f'{label} has uncovered validation voxels')
	_validate_prediction_metadata(
		prediction_metadata,
		best=best,
		dataset=dataset,
		model_tag=model_tag,
		sampling_mode=sampling_mode,
		steps_per_epoch=steps_per_epoch,
		train_seed=decoder_seed,
		train_tile_identity_sha256=train_tile_identity,
		validation_tile_identity_sha256=validation_tile_identity,
		label=label,
	)
	evaluation = _load_evaluation(
		evaluation_metadata,
		metrics_identity=metrics,
		boundary_identity=boundary,
		regions_identity=regions,
		prediction_metadata=prediction_metadata,
		dataset=dataset,
		model_tag=model_tag,
		label=label,
	)
	if _row_sha256(
		row,
		('metric_schema_sha256',),
		label=f'{label}.metric_schema_sha256',
	) != evaluation.metric_schema_sha256:
		raise ValueError(f'{label} metric schema hash mismatch')
	row_architecture = validate_voxel_decoder_architecture_mapping(
		row.get('decoder_architecture'),
		field_prefix=f'{label}.decoder_architecture',
	)
	if row_architecture != evaluation.decoder_architecture:
		raise ValueError(f'{label} decoder architecture/evaluation mismatch')
	if latest.path == best.path or latest.sha256 == best.sha256:
		raise ValueError(f'{label} latest.pt and best.pt identities must be distinct')
	_validate_completed_decoder_artifact(
		latest=latest,
		best=best,
		dataset=dataset,
		model_tag=model_tag,
		decoder_seed=decoder_seed,
		steps_per_epoch=steps_per_epoch,
		class_weights=class_weights,
		canonical_valid_tokens_sha256=valid_tokens,
		initial_model_state_sha256=initial,
		train_tile_manifest_sha256=train_tiles,
		validation_tile_manifest_sha256=validation_tiles,
		train_tile_identity_sha256=train_tile_identity,
		validation_tile_identity_sha256=validation_tile_identity,
		decoder_architecture=row_architecture,
		best_epoch=best_epoch,
		best_metrics=best_metrics,
		label=label,
	)
	return _LoadedJob(
		row=row,
		dataset=dataset,
		model_role=role,
		model_tag=model_tag,
		decoder_seed=decoder_seed,
		class_weights=class_weights,
		canonical_valid_tokens_sha256=valid_tokens,
		initial_model_state_sha256=initial,
		sampling_mode=sampling_mode,
		steps_per_epoch=steps_per_epoch,
		sampling_sequence_sha256=sampling_sequence,
		train_tile_manifest_sha256=train_tiles,
		validation_tile_manifest_sha256=validation_tiles,
		train_tile_identity_sha256=train_tile_identity,
		validation_tile_identity_sha256=validation_tile_identity,
		evaluation=evaluation,
	)


def _validate_completed_decoder_artifact(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	latest: _FileIdentity,
	best: _FileIdentity,
	dataset: _DatasetCondition,
	model_tag: str,
	decoder_seed: int,
	steps_per_epoch: int,
	class_weights: tuple[float, ...],
	canonical_valid_tokens_sha256: str,
	initial_model_state_sha256: str,
	train_tile_manifest_sha256: str,
	validation_tile_manifest_sha256: str,
	train_tile_identity_sha256: str,
	validation_tile_identity_sha256: str,
	decoder_architecture: Mapping[str, object],
	best_epoch: int,
	best_metrics: Mapping[str, object],
	label: str,
) -> None:
	decoder_dir = latest.path.parent
	if latest.path != decoder_dir / 'latest.pt':
		raise ValueError(f'{label} latest checkpoint path is not decoder/latest.pt')
	if best.path != decoder_dir / 'best.pt':
		raise ValueError(f'{label} best checkpoint path is not decoder/best.pt')
	required = (
		'history.csv',
		'resolved_config.json',
		'run_metadata.json',
		'train_tile_manifest.json',
		'validation_tile_manifest.json',
	)
	missing = [name for name in required if not (decoder_dir / name).is_file()]
	if missing:
		raise FileNotFoundError(
			f'{label} decoder artifact is incomplete: missing={missing!r}'
		)
	latest_payload = load_voxel_decoder_checkpoint(latest.path)
	best_payload = load_voxel_decoder_checkpoint(best.path)
	if latest_payload.get('checkpoint_kind') != 'completed':
		raise ValueError(f'{label} latest.pt checkpoint_kind is not completed')
	if latest_payload.get('epoch') != 49:
		raise ValueError(f'{label} latest.pt is not completed epoch 49')
	expected_global_step = steps_per_epoch * 50
	if latest_payload.get('global_step') != expected_global_step:
		raise ValueError(f'{label} latest.pt global_step mismatch')
	if latest_payload.get('best_checkpoint_sha256') != best.sha256:
		raise ValueError(f'{label} latest.pt does not bind the recorded best.pt')
	resolved = _read_json(decoder_dir / 'resolved_config.json')
	if latest_payload.get('resolved_config') != resolved:
		raise ValueError(f'{label} latest.pt/resolved_config.json mismatch')
	if best_payload.get('resolved_config') != resolved:
		raise ValueError(f'{label} best.pt/resolved_config.json mismatch')
	if validate_voxel_decoder_architecture_mapping(
		latest_payload.get('decoder_architecture'),
		field_prefix=f'{label} latest decoder_architecture',
	) != decoder_architecture:
		raise ValueError(f'{label} latest.pt decoder architecture mismatch')
	if validate_voxel_decoder_architecture_mapping(
		best_payload.get('decoder_architecture'),
		field_prefix=f'{label} best decoder_architecture',
	) != decoder_architecture:
		raise ValueError(f'{label} best.pt decoder architecture mismatch')
	model = _mapping(resolved.get('model'), f'{label} resolved model')
	if model.get('tag') != model_tag or model.get('freeze_encoder') is not True:
		raise ValueError(f'{label} resolved frozen-model identity mismatch')
	embeddings = _mapping(resolved.get('embeddings'), f'{label} embeddings')
	if embeddings.get('spec') != 'overlap_x16':
		raise ValueError(f'{label} resolved embedding spec mismatch')
	voxel_dataset = _mapping(
		resolved.get('voxel_dataset'), f'{label} resolved voxel_dataset'
	)
	if Path(str(voxel_dataset.get('input_dir'))).resolve(strict=False) != (
		dataset.root.resolve(strict=False)
	):
		raise ValueError(f'{label} resolved voxel dataset root mismatch')
	train = _mapping(resolved.get('train'), f'{label} resolved train')
	expected_train = {
		'epochs': 50,
		'batch_size': 1,
		'learning_rate': 0.001,
		'weight_decay': 0.0001,
		'class_weight': 'balanced',
		'sampling_mode': 'uniform_tiles_with_replacement',
		'steps_per_epoch': steps_per_epoch,
		'seed': decoder_seed,
		'num_workers': 0,
		'amp': True,
		'gradient_clip_norm': 1.0,
	}
	for key, expected in expected_train.items():
		if train.get(key) != expected:
			raise ValueError(f'{label} resolved train contract mismatch: {key}')
	for checkpoint_label, payload in (
		('latest.pt', latest_payload),
		('best.pt', best_payload),
	):
		if _float_tuple(
			payload.get('class_weights'), f'{label} {checkpoint_label} class weights'
		) != class_weights:
			raise ValueError(f'{label} {checkpoint_label} class weights mismatch')
	latest_artifacts = _mapping(
		latest_payload.get('artifact_identities'), f'{label} latest artifacts'
	)
	best_artifacts = _mapping(
		best_payload.get('artifact_identities'), f'{label} best artifacts'
	)
	if latest_artifacts != best_artifacts:
		raise ValueError(f'{label} latest/best source artifact mismatch')
	for name, expected in (
		('valid_tokens', canonical_valid_tokens_sha256),
		('voxel_split_grid', dataset.grid.sha256),
	):
		artifact = _mapping(latest_artifacts.get(name), f'{label} {name}')
		if artifact.get('sha256') != expected:
			raise ValueError(f'{label} checkpoint source mismatch: {name}')
	expected_tile_identities = {
		'train': train_tile_identity_sha256,
		'validation': validation_tile_identity_sha256,
	}
	if latest_payload.get('tile_manifest_hashes') != expected_tile_identities:
		raise ValueError(f'{label} latest.pt tile manifest identity mismatch')
	if best_payload.get('tile_manifest_hashes') != expected_tile_identities:
		raise ValueError(f'{label} best.pt tile manifest identity mismatch')
	if file_sha256(decoder_dir / 'train_tile_manifest.json') != (
		train_tile_manifest_sha256
	):
		raise ValueError(f'{label} train tile manifest file SHA-256 mismatch')
	if file_sha256(decoder_dir / 'validation_tile_manifest.json') != (
		validation_tile_manifest_sha256
	):
		raise ValueError(f'{label} validation tile manifest file SHA-256 mismatch')
	run_metadata = _read_json(decoder_dir / 'run_metadata.json')
	for key, expected in (
		('initial_model_state_sha256', initial_model_state_sha256),
		('sampling_mode', 'uniform_tiles_with_replacement'),
		('steps_per_epoch', steps_per_epoch),
		('train_seed', decoder_seed),
		('train_tile_manifest_sha256', train_tile_identity_sha256),
		('validation_tile_manifest_sha256', validation_tile_identity_sha256),
	):
		if run_metadata.get(key) != expected:
			raise ValueError(f'{label} run_metadata mismatch: {key}')

	with (decoder_dir / 'history.csv').open(newline='', encoding='utf-8') as handle:
		history_rows = list(csv.DictReader(handle))
	if len(history_rows) != 50:
		raise ValueError(f'{label} history.csv does not contain 50 epochs')
	for epoch, history_row in enumerate(history_rows):
		if int(history_row.get('epoch', -1)) != epoch:
			raise ValueError(f'{label} history.csv epoch sequence mismatch')
		if int(history_row.get('global_step', -1)) != (
			(epoch + 1) * steps_per_epoch
		):
			raise ValueError(f'{label} history.csv global_step sequence mismatch')
	checkpoint_history = latest_payload.get('training_history')
	if (
		not isinstance(checkpoint_history, Sequence)
		or isinstance(checkpoint_history, str | bytes)
		or len(checkpoint_history) != 50
		or any(not isinstance(item, Mapping) for item in checkpoint_history)
	):
		raise ValueError(f'{label} latest.pt training history mismatch')
	for epoch, checkpoint_row in enumerate(
		cast('Sequence[Mapping[str, object]]', checkpoint_history)
	):
		if checkpoint_row.get('epoch') != epoch or checkpoint_row.get(
			'global_step'
		) != (epoch + 1) * steps_per_epoch:
			raise ValueError(f'{label} latest.pt history sequence mismatch')
	best_state = _mapping(
		best_payload.get('best_selection_state'), f'{label} best selection state'
	)
	latest_best_state = _mapping(
		latest_payload.get('best_selection_state'),
		f'{label} latest best selection state',
	)
	if best_state != latest_best_state:
		raise ValueError(f'{label} latest/best selection state mismatch')
	expected_best_kind = _expected_best_checkpoint_kind(best_epoch, epochs=50)
	if best_payload.get('checkpoint_kind') != expected_best_kind:
		raise ValueError(f'{label} best.pt checkpoint_kind mismatch')
	if best_payload.get('epoch') != best_epoch or best_state.get('epoch') != best_epoch:
		raise ValueError(f'{label} best.pt selection epoch mismatch')
	if best_payload.get('global_step') != (best_epoch + 1) * steps_per_epoch:
		raise ValueError(f'{label} best.pt global_step mismatch')
	best_history = best_payload.get('training_history')
	if (
		not isinstance(best_history, Sequence)
		or isinstance(best_history, str | bytes)
		or len(best_history) != best_epoch + 1
		or any(not isinstance(item, Mapping) for item in best_history)
	):
		raise ValueError(f'{label} best.pt training history mismatch')
	selection_metrics = _mapping(
		best_state.get('validation_metrics'), f'{label} best validation metrics'
	)
	for metric in ('macro_f1', 'mean_iou'):
		if _finite_float(selection_metrics.get(metric), f'{label} best {metric}') != (
			_finite_float(best_metrics.get(metric), f'{label} row best {metric}')
		):
				raise ValueError(f'{label} best selection metric mismatch: {metric}')


def _expected_best_checkpoint_kind(best_epoch: int, *, epochs: int) -> str:
	"""Return the schema-5 kind for an epoch-selected best checkpoint."""
	if not 0 <= best_epoch < epochs:
		raise ValueError('best checkpoint epoch is outside the training range')
	return 'completed' if best_epoch + 1 == epochs else 'epoch'


def _repeated_dataset_identity(
	row: Mapping[str, object], *, dataset: _DatasetCondition, label: str
) -> None:
	for key, expected in (
		('voxel_supervision_grid_sha256', dataset.grid.sha256),
		('selected_token_identity_sha256', dataset.selected_token_identity_sha256),
		('unique_token_xyz_sha256', dataset.unique_token_xyz_sha256),
		('validation_mask_sha256', dataset.validation_mask_sha256),
	):
		if _sha256_value(row.get(key), f'{label}.{key}') != expected:
			raise ValueError(f'{label} {key} does not match dataset manifest')
	if _integer_tuple(row.get('class_order'), f'{label}.class_order') != (
		dataset.class_order
	):
		raise ValueError(f'{label} class_order does not match dataset manifest')


def _validate_dataset_binding(
	row: Mapping[str, object], *, dataset: _DatasetCondition, label: str
) -> None:
	key = next(
		(
			candidate
			for candidate in (
				'voxel_dataset',
				'voxel_dataset_path',
				'voxel_dataset_root',
			)
			if candidate in row
		),
		None,
	)
	if key is None:
		raise ValueError(f'{label} is missing voxel dataset identity')
	value = row[key]
	recorded_sha: object = None
	if isinstance(value, Mapping):
		path = _path_value(value.get('path'), f'{label}.{key}.path')
		recorded_sha = value.get('sha256')
	else:
		path = _path_value(value, f'{label}.{key}')
	for sha_key in (
		'voxel_dataset_sha256',
		'voxel_dataset_grid_sha256',
		'supervision_split_grid_sha256',
		f'{key}_sha256',
	):
		if sha_key in row:
			recorded_sha = row[sha_key]
			break
	sha256 = _sha256_value(recorded_sha, f'{label}.voxel_dataset_sha256')
	allowed_paths = {
		dataset.root.resolve(strict=False),
		dataset.grid.path.resolve(strict=False),
	}
	if path.resolve(strict=False) not in allowed_paths:
		raise ValueError(f'{label} voxel dataset path does not match dataset manifest')
	if sha256 != dataset.grid.sha256:
		raise ValueError(f'{label} voxel dataset SHA-256 mismatch')
	if path.is_file() and file_sha256(path) != sha256:
		raise ValueError(f'{label} voxel dataset file SHA-256 mismatch')


def _sampling_contract(
	row: Mapping[str, object], *, label: str
) -> tuple[str, int]:
	container: Mapping[str, object] = row
	if isinstance(row.get('sampling'), Mapping):
		container = cast('Mapping[str, object]', row['sampling'])
	mode = _non_empty_string(
		container.get('sampling_mode', container.get('mode')),
		f'{label}.sampling_mode',
	)
	steps = _positive_int(
		container.get('steps_per_epoch'), f'{label}.steps_per_epoch'
	)
	return mode, steps


def _row_file_identity(
	row: Mapping[str, object], names: Sequence[str], *, label: str
) -> _FileIdentity:
	for name in names:
		if name not in row:
			continue
		value = row[name]
		if isinstance(value, Mapping):
			return _validated_identity(value, label=label)
		path = _path_value(value, label)
		sha_value = _sibling_sha256(row, name)
		identity = _FileIdentity(path, _sha256_value(sha_value, f'{label}.sha256'))
		_validate_file_identity(identity, label=label)
		return identity
	for name in names:
		path_key = name if name.endswith('_path') else f'{name}_path'
		if path_key not in row:
			continue
		path = _path_value(row[path_key], label)
		sha_value = _sibling_sha256(row, path_key)
		identity = _FileIdentity(path, _sha256_value(sha_value, f'{label}.sha256'))
		_validate_file_identity(identity, label=label)
		return identity
	joined = ', '.join(names)
	request = f'{label} is missing an identity ({joined})'
	raise ValueError(request)


def _sibling_sha256(row: Mapping[str, object], key: str) -> object:
	candidates = [f'{key}_sha256']
	if key.endswith('_path'):
		candidates.append(f'{key.removesuffix("_path")}_sha256')
	for candidate in candidates:
		if candidate in row:
			return row[candidate]
	return None


def _row_sha256(
	row: Mapping[str, object], names: Sequence[str], *, label: str
) -> str:
	for name in names:
		if name not in row:
			continue
		value = row[name]
		if isinstance(value, Mapping):
			identity = _validated_identity(value, label=label)
			return identity.sha256
		return _sha256_value(value, label)
	raise ValueError(f'{label} is missing')


def _validate_prediction_metadata(  # noqa: C901, PLR0912, PLR0913
	identity: _FileIdentity,
	*,
	best: _FileIdentity,
	dataset: _DatasetCondition,
	model_tag: str,
	sampling_mode: str,
	steps_per_epoch: int,
	train_seed: int,
	train_tile_identity_sha256: str,
	validation_tile_identity_sha256: str,
	label: str,
) -> None:
	artifact = validate_f3_voxel_prediction_artifact(
		identity.path.parent, mmap_mode='r'
	)
	if artifact.paths.metadata.resolve(strict=False) != identity.path.resolve(
		strict=False
	):
		raise ValueError(f'{label} prediction metadata path mismatch')
	metadata = artifact.metadata
	if metadata.get('artifact_type') != 'f3_lithology_voxel_predictions':
		raise ValueError(f'{label} prediction artifact_type mismatch')
	if metadata.get('schema_version') != 1:
		raise ValueError(f'{label} prediction schema_version mismatch')
	if metadata.get('model_tag') != model_tag:
		raise ValueError(f'{label} prediction model identity mismatch')
	if metadata.get('prediction_kind') != 'frozen_embedding_decoder':
		raise ValueError(f'{label} prediction kind is not decoder inference')
	if metadata.get('write_probabilities') is not False:
		raise ValueError(f'{label} prediction unexpectedly writes probabilities')
	inputs = _mapping(metadata.get('inputs'), f'{label} prediction inputs')
	checkpoint_path = _path_value(
		inputs.get('decoder_checkpoint'), f'{label} prediction checkpoint path'
	)
	if checkpoint_path.resolve(strict=False) != best.path.resolve(strict=False):
		raise ValueError(f'{label} inference did not use best.pt')
	source = _mapping(
		metadata.get('source_identity'), f'{label} prediction source_identity'
	)
	checkpoint = _validated_identity(
		source.get('decoder_checkpoint'),
		label=f'{label} prediction decoder checkpoint',
	)
	if checkpoint != best:
		raise ValueError(f'{label} prediction best checkpoint identity mismatch')
	training_sampling = _mapping(
		metadata.get('training_sampling'),
		f'{label} prediction training_sampling',
	)
	expected_sampling = {
		'sampling_mode': sampling_mode,
		'steps_per_epoch': steps_per_epoch,
		'train_seed': train_seed,
		'train_tile_manifest_sha256': train_tile_identity_sha256,
		'validation_tile_manifest_sha256': (
			validation_tile_identity_sha256
		),
	}
	if training_sampling != expected_sampling:
		raise ValueError(
			f'{label} prediction training-sampling contract mismatch'
		)
	if artifact.arrays.probabilities is not None:
		raise ValueError(f'{label} prediction unexpectedly contains probabilities')
	coverage = _mapping(metadata.get('coverage'), f'{label} prediction coverage')
	original_count = math.prod(artifact.arrays.valid_mask.shape)
	expected_coverage = {
		'duplicate_write_count': 0,
		'exact_once': True,
		'missing_write_count': 0,
		'original_voxel_count': original_count,
		'written_voxel_count': original_count,
	}
	for key, expected in expected_coverage.items():
		if coverage.get(key) != expected:
			raise ValueError(f'{label} prediction coverage mismatch: {key}')
	summary = _mapping(metadata.get('summary'), f'{label} prediction summary')
	if coverage.get('valid_voxel_count') != summary.get('valid_voxel_count'):
		raise ValueError(f'{label} prediction valid coverage mismatch')
	grid = np.load(dataset.grid.path, mmap_mode='r', allow_pickle=False)
	valid_mask = artifact.arrays.valid_mask
	if grid.shape != valid_mask.shape:
		raise ValueError(f'{label} prediction/grid shape mismatch')
	flat_grid = grid.reshape(-1)
	flat_valid = valid_mask.reshape(-1)
	uncovered = 0
	for start in range(0, flat_grid.size, 1_000_000):
		stop = min(start + 1_000_000, flat_grid.size)
		uncovered += int(
			np.count_nonzero(
				(flat_grid[start:stop] == 2) & ~flat_valid[start:stop]
			)
		)
	if uncovered != 0:
		raise ValueError(f'{label} prediction has uncovered validation voxels')


def _load_evaluation(  # noqa: C901, PLR0912, PLR0913
	metadata_identity: _FileIdentity,
	*,
	metrics_identity: _FileIdentity,
	boundary_identity: _FileIdentity,
	regions_identity: _FileIdentity,
	prediction_metadata: _FileIdentity,
	dataset: _DatasetCondition,
	model_tag: str,
	label: str,
) -> _Evaluation:
	metadata = _read_json(metadata_identity.path)
	if metadata.get('artifact_type') != 'f3_lithology_voxel_evaluation':
		raise ValueError(f'{label} evaluation artifact_type mismatch')
	if metadata.get('schema_version') != 2:
		raise ValueError(f'{label} evaluation schema_version mismatch')
	if metadata.get('model_tag') != model_tag:
		raise ValueError(f'{label} evaluation model identity mismatch')
	if metadata.get('prediction_kind') != 'frozen_embedding_decoder':
		raise ValueError(f'{label} evaluation prediction kind mismatch')
	aggregation = _mapping(metadata.get('aggregation'), f'{label} aggregation')
	if aggregation != {
		'primary_unit': 'unique_validation_voxel',
		'split_code': 2,
		'intersection_voxels_counted_once': True,
		'per_slice_planes_evaluated_independently': True,
		'voxel_independence_p_values_computed': False,
	}:
		raise ValueError(f'{label} unique-validation aggregation mismatch')
	policy = _mapping(metadata.get('policy'), f'{label} evaluation policy')
	if policy != {
		'monitored_class_ids': list(MONITORED_CLASS_IDS),
		'boundary_tolerances': list(BOUNDARY_TOLERANCES),
		'boundary_region_radii': list(BOUNDARY_RADII),
		'primary_trace_boundary_tolerance': max(BOUNDARY_TOLERANCES),
		'chunk_size_x': 8,
	}:
		raise ValueError(f'{label} evaluation policy mismatch')
	outputs = _mapping(metadata.get('outputs'), f'{label} evaluation outputs')
	for name, expected in (
		(METRICS_JSON, metrics_identity),
		(BOUNDARY_METRICS_JSON, boundary_identity),
		(BOUNDARY_REGION_METRICS_CSV, regions_identity),
	):
		recorded = _validated_identity(outputs.get(name), label=f'{label} {name}')
		if recorded != expected:
			raise ValueError(f'{label} run/evaluation {name} identity mismatch')
	inputs = _mapping(metadata.get('inputs'), f'{label} evaluation inputs')
	grid = _validated_identity(
		inputs.get('voxel_split_grid'), label=f'{label} evaluation voxel grid'
	)
	if grid != dataset.grid:
		raise ValueError(f'{label} evaluation uses the wrong voxel dataset grid')
	prediction = _validated_identity(
		inputs.get('prediction_metadata'),
		label=f'{label} evaluation prediction metadata',
	)
	if prediction != prediction_metadata:
		raise ValueError(f'{label} evaluation prediction identity mismatch')
	metrics_payload = _read_json(metrics_identity.path)
	boundary_payload = _read_json(boundary_identity.path)
	regions, region_header = _boundary_regions(regions_identity.path, label=label)
	class_order = _integer_tuple(
		metrics_payload.get('class_ids'), f'{label} metrics.class_ids'
	)
	if class_order != dataset.class_order:
		raise ValueError(f'{label} evaluation class order mismatch')
	if metrics_payload.get('aggregation_unit') != 'unique_validation_voxel':
		raise ValueError(f'{label} metric aggregation unit mismatch')
	validation_count = _positive_int(
		metrics_payload.get('evaluation_voxel_count'),
		f'{label} metrics.evaluation_voxel_count',
	)
	if validation_count != dataset.validation_voxel_count:
		raise ValueError(f'{label} evaluation validation count mismatch')
	summary = _mapping(metadata.get('summary'), f'{label} evaluation summary')
	if summary.get('unique_validation_voxel_count') != validation_count:
		raise ValueError(f'{label} evaluation metadata count mismatch')
	architecture = validate_voxel_decoder_architecture_mapping(
		metadata.get('decoder_architecture'),
		field_prefix=f'{label} evaluation decoder_architecture',
	)
	values = _evaluation_metric_values(
		metrics_payload, boundary_payload, regions, label=label
	)
	schema = {
		'metrics': sorted(metrics_payload),
		'boundary': sorted(boundary_payload),
		'boundary_region_columns': region_header,
	}
	return _Evaluation(
		model_tag=model_tag,
		class_order=class_order,
		validation_voxel_count=validation_count,
		decoder_architecture=architecture,
		metric_schema_sha256=_json_sha256(schema),
		metrics=values,
	)


def _boundary_regions(
	path: Path, *, label: str
) -> tuple[dict[int, Mapping[str, str]], list[str]]:
	with path.open(newline='', encoding='utf-8') as handle:
		reader = csv.DictReader(handle)
		if reader.fieldnames is None:
			raise ValueError(f'{label} boundary-region table has no header')
		rows = list(reader)
	regions: dict[int, Mapping[str, str]] = {}
	for row in rows:
		if row.get('region') != 'boundary':
			continue
		try:
			radius = int(row.get('radius', ''))
		except ValueError as error:
			raise ValueError(f'{label} boundary radius is invalid') from error
		if radius in regions:
			raise ValueError(f'{label} duplicate boundary radius {radius}')
		regions[radius] = row
	for radius in BOUNDARY_RADII:
		if radius not in regions:
			raise ValueError(f'{label} missing boundary-region radius {radius}')
	return regions, list(reader.fieldnames)


def _evaluation_metric_values(
	metrics: Mapping[str, object],
	boundary: Mapping[str, object],
	regions: Mapping[int, Mapping[str, str]],
	*,
	label: str,
) -> dict[str, float]:
	values = {
		name: _finite_float(metrics.get(name), f'{label}.{name}')
		for name in (
			'macro_f1',
			'mean_iou',
			'balanced_accuracy',
			'accuracy',
			'weighted_f1',
		)
	}
	for radius in BOUNDARY_RADII:
		values[f'boundary_region_macro_f1_r{radius}'] = _finite_float(
			regions[radius].get('macro_f1'),
			f'{label}.boundary_region_macro_f1_r{radius}',
		)
		values[f'boundary_region_mean_iou_r{radius}'] = _finite_float(
			regions[radius].get('mean_iou'),
			f'{label}.boundary_region_mean_iou_r{radius}',
		)
	for tolerance in BOUNDARY_TOLERANCES:
		values[f'boundary_f1_t{tolerance}'] = _finite_float(
			boundary.get(f'vertical_boundary_f1_at_{tolerance}'),
			f'{label}.boundary_f1_t{tolerance}',
		)
	values['vertical_boundary_position_mae'] = _finite_float(
		boundary.get('vertical_boundary_position_mae_at_4'),
		f'{label}.vertical_boundary_position_mae',
	)
	for class_id in MONITORED_CLASS_IDS:
		for metric in ('f1', 'iou'):
			per_class = _mapping(
				metrics.get(f'per_class_{metric}'),
				f'{label}.per_class_{metric}',
			)
			values[f'class_{class_id}_{metric}'] = _finite_float(
				per_class.get(str(class_id)),
				f'{label}.class_{class_id}_{metric}',
			)
		for tolerance in BOUNDARY_TOLERANCES:
			name = f'class_{class_id}_boundary_recall_t{tolerance}'
			values[name] = _finite_float(
				boundary.get(
					f'vertical_boundary_class_{class_id}_recall_at_{tolerance}'
				),
				f'{label}.{name}',
			)
	if set(values) != set(METRIC_BY_NAME):
		raise AssertionError('internal metric schema mismatch')
	return values


def _validate_triplet(jobs: tuple[_LoadedJob, ...]) -> None:
	if tuple(item.model_role for item in jobs) != MODEL_ROLES:
		raise ValueError('paired triplet model order mismatch')
	first = jobs[0]
	condition = f'{first.dataset.budget_id}/seed{first.dataset.subsample_seed}'
	identities = {
		'voxel_supervision_grid_sha256': [item.dataset.grid.sha256 for item in jobs],
		'selected_token_identity_sha256': [
			item.dataset.selected_token_identity_sha256 for item in jobs
		],
		'unique_token_xyz_sha256': [
			item.dataset.unique_token_xyz_sha256 for item in jobs
		],
		'train_voxel_count': [item.dataset.train_voxel_count for item in jobs],
		'validation_voxel_count': [
			item.dataset.validation_voxel_count for item in jobs
		],
		'class_order': [item.evaluation.class_order for item in jobs],
		'validation_mask_sha256': [
			item.dataset.validation_mask_sha256 for item in jobs
		],
		'canonical_valid_tokens_sha256': [
			item.canonical_valid_tokens_sha256 for item in jobs
		],
		'decoder_architecture': [
			_stable_json(item.evaluation.decoder_architecture) for item in jobs
		],
		'initial_model_state_sha256': [
			item.initial_model_state_sha256 for item in jobs
		],
		'sampling_mode': [item.sampling_mode for item in jobs],
		'steps_per_epoch': [item.steps_per_epoch for item in jobs],
		'sampling_sequence_sha256': [
			item.sampling_sequence_sha256 for item in jobs
		],
		'train_tile_manifest_sha256': [
			item.train_tile_manifest_sha256 for item in jobs
		],
		'validation_tile_manifest_sha256': [
			item.validation_tile_manifest_sha256 for item in jobs
		],
		'train_tile_identity_sha256': [
			item.train_tile_identity_sha256 for item in jobs
		],
		'validation_tile_identity_sha256': [
			item.validation_tile_identity_sha256 for item in jobs
		],
		'class_weights': [item.class_weights for item in jobs],
		'decoder_seed': [item.decoder_seed for item in jobs],
		'validation_voxel_coverage': [
			item.evaluation.validation_voxel_count for item in jobs
		],
		'metric_schema_sha256': [
			item.evaluation.metric_schema_sha256 for item in jobs
		],
	}
	for name, values in identities.items():
		if any(value != values[0] for value in values[1:]):
			raise ValueError(
				f'paired identity mismatch for {condition}: {name}'
			)


def _job_metric_row(job: _LoadedJob) -> dict[str, object]:
	return {
		'budget_id': job.dataset.budget_id,
		'per_class_cap': job.dataset.per_class_cap,
		'subsample_seed': job.dataset.subsample_seed,
		'decoder_seed': job.decoder_seed,
		'model_role': job.model_role,
		'model': MODEL_LABELS[job.model_role],
		'model_tag': job.model_tag,
		'voxel_dataset_root': str(job.dataset.root),
		'voxel_supervision_grid_sha256': job.dataset.grid.sha256,
		'selected_token_identity_sha256': (
			job.dataset.selected_token_identity_sha256
		),
		'unique_token_xyz_sha256': job.dataset.unique_token_xyz_sha256,
		'train_voxel_count': job.dataset.train_voxel_count,
		'validation_voxel_count': job.dataset.validation_voxel_count,
		'validation_mask_sha256': job.dataset.validation_mask_sha256,
		'canonical_valid_tokens_sha256': job.canonical_valid_tokens_sha256,
		'initial_model_state_sha256': job.initial_model_state_sha256,
		'sampling_mode': job.sampling_mode,
		'steps_per_epoch': job.steps_per_epoch,
		'sampling_sequence_sha256': job.sampling_sequence_sha256,
		'train_tile_manifest_sha256': job.train_tile_manifest_sha256,
		'validation_tile_manifest_sha256': job.validation_tile_manifest_sha256,
		'train_tile_identity_sha256': job.train_tile_identity_sha256,
		'validation_tile_identity_sha256': (
			job.validation_tile_identity_sha256
		),
		'class_weights': json.dumps(job.class_weights, separators=(',', ':')),
		'metric_schema_sha256': job.evaluation.metric_schema_sha256,
		**job.evaluation.metrics,
	}


def _paired_metric_rows(jobs: Sequence[_LoadedJob]) -> list[dict[str, object]]:
	by_key = {
		(job.dataset.budget_id, job.dataset.subsample_seed, job.model_role): job
		for job in jobs
	}
	rows = []
	for budget in REQUIRED_BUDGETS:
		for seed in REQUIRED_SEEDS:
			triplet = [by_key[(budget, seed, role)] for role in MODEL_ROLES]
			row: dict[str, object] = {
				'budget_id': budget,
				'per_class_cap': triplet[0].dataset.per_class_cap,
				'subsample_seed': seed,
				'decoder_seed': triplet[0].decoder_seed,
				'voxel_supervision_grid_sha256': triplet[0].dataset.grid.sha256,
				'selected_token_identity_sha256': (
					triplet[0].dataset.selected_token_identity_sha256
				),
				'unique_token_xyz_sha256': (
					triplet[0].dataset.unique_token_xyz_sha256
				),
				'train_voxel_count': triplet[0].dataset.train_voxel_count,
				'validation_voxel_count': triplet[0].dataset.validation_voxel_count,
			}
			for job in triplet:
				for metric, value in job.evaluation.metrics.items():
					row[f'{job.model_role}_{metric}'] = value
			rows.append(row)
	return rows


def _paired_delta_rows(jobs: Sequence[_LoadedJob]) -> list[dict[str, object]]:
	by_key = {
		(job.dataset.budget_id, job.dataset.subsample_seed, job.model_role): job
		for job in jobs
	}
	rows = []
	for budget in REQUIRED_BUDGETS:
		for seed in REQUIRED_SEEDS:
			for comparison_id, baseline_role, candidate_role, label in COMPARISONS:
				baseline = by_key[(budget, seed, baseline_role)]
				candidate = by_key[(budget, seed, candidate_role)]
				row: dict[str, object] = {
					'budget_id': budget,
					'per_class_cap': baseline.dataset.per_class_cap,
					'subsample_seed': seed,
					'decoder_seed': baseline.decoder_seed,
					'comparison_id': comparison_id,
					'comparison': label,
					'baseline_model_role': baseline_role,
					'baseline_model_tag': baseline.model_tag,
					'candidate_model_role': candidate_role,
					'candidate_model_tag': candidate.model_tag,
				}
				for metric in METRIC_SPECS:
					row[metric.name] = (
						candidate.evaluation.metrics[metric.name]
						- baseline.evaluation.metrics[metric.name]
					)
				rows.append(row)
	return rows


def _summary_rows(
	paired_deltas: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	rows = []
	for budget in REQUIRED_BUDGETS:
		for comparison_id, baseline, candidate, label in COMPARISONS:
			selected = [
				row
				for row in paired_deltas
				if row['budget_id'] == budget
				and row['comparison_id'] == comparison_id
			]
			if len(selected) != len(REQUIRED_SEEDS):
				raise AssertionError('paired summary lost a required seed')
			for metric in METRIC_SPECS:
				values = [float(cast('float', row[metric.name])) for row in selected]
				wins = [
					value > 0.0 if metric.higher_is_better else value < 0.0
					for value in values
				]
				losses = [
					value < 0.0 if metric.higher_is_better else value > 0.0
					for value in values
				]
				worst_index = (
					min(range(len(values)), key=values.__getitem__)
					if metric.higher_is_better
					else max(range(len(values)), key=values.__getitem__)
				)
				rows.append(
					{
						'budget_id': budget,
						'per_class_cap': int(budget.removeprefix('cap')),
						'comparison_id': comparison_id,
						'comparison': label,
						'baseline_model_role': baseline,
						'candidate_model_role': candidate,
						'metric': metric.name,
						'higher_is_better': metric.higher_is_better,
						'paired_seed_count': len(values),
						'mean_delta': statistics.fmean(values),
						'median_delta': statistics.median(values),
						'standard_deviation': statistics.stdev(values),
						'min_delta': min(values),
						'max_delta': max(values),
						'worst_seed': int(selected[worst_index]['subsample_seed']),
						'worst_seed_delta': values[worst_index],
						'positive_win_count': sum(wins),
						'zero_count': sum(value == 0.0 for value in values),
						'negative_count': sum(losses),
					}
				)
	return rows


def _monitored_summary_rows(
	summary: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	index = {
		(str(row['budget_id']), str(row['comparison_id']), str(row['metric'])): row
		for row in summary
	}
	rows = []
	for budget in REQUIRED_BUDGETS:
		for comparison_id, _baseline, _candidate, label in COMPARISONS:
			for class_id in MONITORED_CLASS_IDS:
				row: dict[str, object] = {
					'budget_id': budget,
					'per_class_cap': int(budget.removeprefix('cap')),
					'comparison_id': comparison_id,
					'comparison': label,
					'class_id': class_id,
				}
				for metric in (
					'f1',
					'iou',
					'boundary_recall_t2',
					'boundary_recall_t4',
				):
					source = index[
						(budget, comparison_id, f'class_{class_id}_{metric}')
					]
					for statistic in (
						'mean_delta',
						'median_delta',
						'standard_deviation',
						'positive_win_count',
						'zero_count',
						'negative_count',
					):
						row[f'{metric}_{statistic}'] = source[statistic]
				rows.append(row)
	return rows


def _load_full_label_anchors(
	config: F3VoxelLabelBudgetResultsConfig,
	jobs: Sequence[_LoadedJob],
) -> list[dict[str, object]]:
	anchors = [
		_load_full_label_anchor(
			role,
			config.full_label_evaluations[role],
			canonical_full_grid_sha256=(
				jobs[0].dataset.canonical_full_grid_sha256
			),
		)
		for role in MODEL_ROLES
	]
	class_orders = {item[0].class_order for item in anchors}
	validation_counts = {item[0].validation_voxel_count for item in anchors}
	architectures = {_stable_json(item[0].decoder_architecture) for item in anchors}
	if class_orders != {jobs[0].evaluation.class_order}:
		raise ValueError('full-label anchor class order mismatch')
	if validation_counts != {jobs[0].evaluation.validation_voxel_count}:
		raise ValueError('full-label anchor validation voxel count mismatch')
	if architectures != {_stable_json(jobs[0].evaluation.decoder_architecture)}:
		raise ValueError('full-label anchor decoder architecture mismatch')
	rows = []
	for role, (evaluation, identities) in zip(MODEL_ROLES, anchors, strict=True):
		rows.append(
			{
				'budget_id': 'full',
				'aggregation': 'single_existing_run',
				'seed': 42,
				'model_role': role,
				'model': MODEL_LABELS[role],
				'model_tag': evaluation.model_tag,
				'validation_voxel_count': evaluation.validation_voxel_count,
				'metrics_json': str(identities['metrics'].path),
				'metrics_sha256': identities['metrics'].sha256,
				'evaluation_metadata': str(identities['metadata'].path),
				'evaluation_metadata_sha256': identities['metadata'].sha256,
				'canonical_full_grid_sha256': identities['grid'].sha256,
				'prediction_metadata': str(
					identities['prediction_metadata'].path
				),
				'prediction_metadata_sha256': (
					identities['prediction_metadata'].sha256
				),
				**evaluation.metrics,
			}
		)
	return rows


def _load_full_label_anchor(  # noqa: C901
	role: str,
	root: Path,
	*,
	canonical_full_grid_sha256: str,
) -> tuple[_Evaluation, Mapping[str, _FileIdentity]]:
	if not root.is_dir():
		raise FileNotFoundError(f'missing full-label evaluation directory: {root}')
	identities = {
		'metadata': _file_identity(root / EVALUATION_METADATA_JSON),
		'metrics': _file_identity(root / METRICS_JSON),
		'boundary': _file_identity(root / BOUNDARY_METRICS_JSON),
		'regions': _file_identity(root / BOUNDARY_REGION_METRICS_CSV),
	}
	metadata = _read_json(identities['metadata'].path)
	if metadata.get('artifact_type') != 'f3_lithology_voxel_evaluation':
		raise ValueError(f'full-label {role} evaluation artifact_type mismatch')
	if metadata.get('schema_version') != 2:
		raise ValueError(f'full-label {role} evaluation schema_version mismatch')
	model_tag = _non_empty_string(
		metadata.get('model_tag'), f'full-label {role}.model_tag'
	)
	if model_tag != EXPECTED_MODEL_TAGS[role]:
		raise ValueError(f'full-label {role} model identity mismatch')
	if metadata.get('prediction_kind') != 'frozen_embedding_decoder':
		raise ValueError(f'full-label {role} prediction kind mismatch')
	inputs = _mapping(metadata.get('inputs'), f'full-label {role}.inputs')
	grid = _validated_identity(
		inputs.get('voxel_split_grid'), label=f'full-label {role} voxel grid'
	)
	if grid.sha256 != canonical_full_grid_sha256:
		raise ValueError(f'full-label {role} does not use the canonical full grid')
	prediction_metadata = _validated_identity(
		inputs.get('prediction_metadata'),
		label=f'full-label {role} prediction metadata',
	)
	_validate_full_label_anchor_decoder(
		role=role,
		prediction_metadata=prediction_metadata,
		grid=grid,
	)
	identities = {
		**identities,
		'grid': grid,
		'prediction_metadata': prediction_metadata,
	}
	outputs = _mapping(metadata.get('outputs'), f'full-label {role}.outputs')
	for name, key in (
		(METRICS_JSON, 'metrics'),
		(BOUNDARY_METRICS_JSON, 'boundary'),
		(BOUNDARY_REGION_METRICS_CSV, 'regions'),
	):
		recorded = _validated_identity(
			outputs.get(name), label=f'full-label {role}.{name}'
		)
		if recorded != identities[key]:
			raise ValueError(f'full-label {role} {name} identity mismatch')
	metrics = _read_json(identities['metrics'].path)
	boundary = _read_json(identities['boundary'].path)
	regions, header = _boundary_regions(
		identities['regions'].path, label=f'full {role}'
	)
	class_order = _integer_tuple(
		metrics.get('class_ids'), f'full-label {role}.class_ids'
	)
	count = _positive_int(
		metrics.get('evaluation_voxel_count'),
		f'full-label {role}.evaluation_voxel_count',
	)
	if metrics.get('aggregation_unit') != 'unique_validation_voxel':
		raise ValueError(f'full-label {role} aggregation unit mismatch')
	if _mapping(metadata.get('summary'), f'full-label {role}.summary').get(
		'unique_validation_voxel_count'
	) != count:
		raise ValueError(f'full-label {role} validation count mismatch')
	architecture = validate_voxel_decoder_architecture_mapping(
		metadata.get('decoder_architecture'),
		field_prefix=f'full-label {role}.decoder_architecture',
	)
	values = _evaluation_metric_values(
		metrics, boundary, regions, label=f'full-label {role}'
	)
	schema = {
		'metrics': sorted(metrics),
		'boundary': sorted(boundary),
		'boundary_region_columns': header,
	}
	return (
		_Evaluation(
			model_tag=model_tag,
			class_order=class_order,
			validation_voxel_count=count,
			decoder_architecture=architecture,
			metric_schema_sha256=_json_sha256(schema),
			metrics=values,
		),
		identities,
	)


def _validate_full_label_anchor_decoder(  # noqa: C901, PLR0912, PLR0915
	*,
	role: str,
	prediction_metadata: _FileIdentity,
	grid: _FileIdentity,
) -> None:
	artifact = validate_f3_voxel_prediction_artifact(
		prediction_metadata.path.parent, mmap_mode='r'
	)
	metadata = artifact.metadata
	if metadata.get('model_tag') != EXPECTED_MODEL_TAGS[role]:
		raise ValueError(f'full-label {role} prediction model identity mismatch')
	if metadata.get('prediction_kind') != 'frozen_embedding_decoder':
		raise ValueError(f'full-label {role} prediction kind mismatch')
	if metadata.get('write_probabilities') is not False:
		raise ValueError(f'full-label {role} prediction writes probabilities')
	source = _mapping(
		metadata.get('source_identity'), f'full-label {role} prediction source'
	)
	best = _validated_identity(
		source.get('decoder_checkpoint'),
		label=f'full-label {role} decoder checkpoint',
	)
	resolved_identity = _validated_identity(
		source.get('resolved_decoder_config'),
		label=f'full-label {role} resolved decoder config',
	)
	inputs = _mapping(metadata.get('inputs'), f'full-label {role} prediction inputs')
	if Path(str(inputs.get('decoder_checkpoint'))).resolve(strict=False) != (
		best.path.resolve(strict=False)
	):
		raise ValueError(f'full-label {role} prediction did not use best.pt')
	if best.path.name != 'best.pt':
		raise ValueError(f'full-label {role} checkpoint is not best.pt')
	decoder_dir = best.path.parent
	latest_path = decoder_dir / 'latest.pt'
	history_path = decoder_dir / 'history.csv'
	for path in (latest_path, history_path):
		if not path.is_file():
			raise FileNotFoundError(f'missing full-label {role} artifact: {path}')
	latest = load_voxel_decoder_checkpoint(latest_path)
	best_payload = load_voxel_decoder_checkpoint(best.path)
	if latest.get('checkpoint_kind') != 'completed' or latest.get('epoch') != 49:
		raise ValueError(f'full-label {role} latest.pt is not completed epoch 49')
	if latest.get('best_checkpoint_sha256') != best.sha256:
		raise ValueError(f'full-label {role} best checkpoint binding mismatch')
	resolved = _read_json(resolved_identity.path)
	if latest.get('resolved_config') != resolved:
		raise ValueError(f'full-label {role} latest/config mismatch')
	if best_payload.get('resolved_config') != resolved:
		raise ValueError(f'full-label {role} best/config mismatch')
	prediction_artifacts = _mapping(
		source.get('artifact_identities'),
		f'full-label {role} prediction artifacts',
	)
	checkpoint_artifacts = _mapping(
		best_payload.get('artifact_identities'),
		f'full-label {role} checkpoint artifacts',
	)
	if prediction_artifacts != checkpoint_artifacts:
		raise ValueError(f'full-label {role} prediction/checkpoint source mismatch')
	prediction_grid = _mapping(
		prediction_artifacts.get('voxel_split_grid'),
		f'full-label {role} prediction voxel grid',
	)
	if prediction_grid.get('sha256') != grid.sha256 or Path(
		str(prediction_grid.get('path'))
	).resolve(strict=False) != grid.path.resolve(strict=False):
		raise ValueError(f'full-label {role} prediction grid identity mismatch')
	model = _mapping(resolved.get('model'), f'full-label {role} model')
	if model != {'tag': EXPECTED_MODEL_TAGS[role], 'freeze_encoder': True}:
		raise ValueError(f'full-label {role} frozen model identity mismatch')
	train = _mapping(resolved.get('train'), f'full-label {role} train')
	if train.get('seed') != 42 or train.get('epochs') != 50:
		raise ValueError(f'full-label {role} is not the existing seed-42 run')
	if train.get('batch_size') != 1:
		raise ValueError(f'full-label {role} batch-size contract mismatch')
	if validate_voxel_decoder_architecture_mapping(
		resolved.get('decoder'),
		field_prefix=f'full-label {role} resolved decoder',
	) != validate_voxel_decoder_architecture_mapping(
		metadata.get('decoder_architecture'),
		field_prefix=f'full-label {role} prediction decoder',
	):
		raise ValueError(f'full-label {role} decoder architecture mismatch')
	voxel_dataset = _mapping(
		resolved.get('voxel_dataset'), f'full-label {role} voxel dataset'
	)
	if Path(str(voxel_dataset.get('input_dir'))).resolve(strict=False) != (
		grid.path.parent.resolve(strict=False)
	):
		raise ValueError(f'full-label {role} canonical voxel dataset mismatch')
	with history_path.open(newline='', encoding='utf-8') as handle:
		history = list(csv.DictReader(handle))
	if len(history) != 50:
		raise ValueError(f'full-label {role} history does not contain 50 epochs')
	for epoch, row in enumerate(history):
		if int(row.get('epoch', -1)) != epoch:
			raise ValueError(f'full-label {role} history epoch sequence mismatch')
	if int(history[-1].get('global_step', -1)) != latest.get('global_step'):
		raise ValueError(f'full-label {role} history/global-step mismatch')
	best_state = _mapping(
		best_payload.get('best_selection_state'),
		f'full-label {role} best selection state',
	)
	if best_payload.get('epoch') != best_state.get('epoch'):
		raise ValueError(f'full-label {role} best-selection epoch mismatch')


def _scientific_decisions(
	summary: Sequence[Mapping[str, object]],
	thresholds: F3VoxelLabelBudgetDecisionThresholds,
) -> dict[str, object]:
	return {
		'structured_pretext_vs_mae': {
			'm1_vs_mae': _decision_for_comparison(
				'm1_vs_mae', summary, thresholds
			),
			'm2a_vs_mae': _decision_for_comparison(
				'm2a_vs_mae', summary, thresholds
			),
		},
		'boundary_aware_increment': {
			'm2a_vs_m1': _decision_for_comparison(
				'm2a_vs_m1', summary, thresholds
			),
		},
	}


def _decision_for_comparison(  # noqa: C901, PLR0912, PLR0915
	comparison_id: str,
	summary: Sequence[Mapping[str, object]],
	thresholds: F3VoxelLabelBudgetDecisionThresholds,
) -> dict[str, object]:
	index = {
		(str(row['budget_id']), str(row['metric'])): row
		for row in summary
		if row['comparison_id'] == comparison_id
	}
	positive_budgets = []
	negative_budgets = []
	mean_median_disagreements: list[str] = []
	low_primary_win_support: list[str] = []
	primary_budget_directions: dict[str, str] = {}
	for budget in REQUIRED_BUDGETS:
		macro = index[(budget, 'macro_f1')]
		iou = index[(budget, 'mean_iou')]
		for metric_name, metric_row in (('macro_f1', macro), ('mean_iou', iou)):
			if float(metric_row['mean_delta']) * float(
				metric_row['median_delta']
			) < 0.0:
				mean_median_disagreements.append(f'{budget}:{metric_name}')
			if (
				float(metric_row['mean_delta']) > 0.0
				and int(metric_row['positive_win_count'])
				< thresholds.minimum_primary_wins
			):
				low_primary_win_support.append(f'{budget}:{metric_name}')
		if (
			float(macro['mean_delta']) > 0.0
			and float(iou['mean_delta']) > 0.0
			and int(macro['positive_win_count']) >= thresholds.minimum_primary_wins
			and int(iou['positive_win_count']) >= thresholds.minimum_primary_wins
		):
			positive_budgets.append(budget)
		if float(macro['mean_delta']) < 0.0 and float(iou['mean_delta']) < 0.0:
			negative_budgets.append(budget)
		if float(macro['mean_delta']) > 0.0 and float(iou['mean_delta']) > 0.0:
			primary_budget_directions[budget] = 'positive'
		elif float(macro['mean_delta']) < 0.0 and float(iou['mean_delta']) < 0.0:
			primary_budget_directions[budget] = 'negative'
		else:
			primary_budget_directions[budget] = 'mixed'
	major_degradation: dict[str, list[str]] = {}
	major_degradation_metrics: dict[str, dict[str, list[str]]] = {}
	consistent_degradation: dict[str, bool] = {}
	systematic_improvement: dict[str, bool] = {}
	for class_id in thresholds.monitored_class_ids:
		major = []
		major_metrics: dict[str, list[str]] = {}
		improved = []
		consistently_worse = True
		for budget in REQUIRED_BUDGETS:
			class_deltas = {
				metric: float(
					index[(budget, f'class_{class_id}_{metric}')]['mean_delta']
				)
				for metric in (
					'f1',
					'iou',
					'boundary_recall_t2',
					'boundary_recall_t4',
				)
			}
			degraded_metrics = sorted(
				metric
				for metric, delta in class_deltas.items()
				if delta <= thresholds.major_degradation_delta
			)
			if degraded_metrics:
				major.append(budget)
				major_metrics[budget] = degraded_metrics
			if class_deltas['f1'] > 0.0 and class_deltas['iou'] > 0.0:
				improved.append(budget)
			if not all(delta < 0.0 for delta in class_deltas.values()):
				consistently_worse = False
		major_degradation[str(class_id)] = major
		major_degradation_metrics[str(class_id)] = major_metrics
		consistent_degradation[str(class_id)] = consistently_worse
		systematic_improvement[str(class_id)] = (
			len(improved) >= thresholds.minimum_positive_budgets
		)
	major_block = any(
		len(budgets) >= thresholds.systematic_degradation_budget_count
		for budgets in major_degradation.values()
	)
	one_sided_class_improvement = sum(systematic_improvement.values()) == 1
	negative = (
		len(negative_budgets) >= thresholds.negative_budget_count
		or any(consistent_degradation.values())
	)
	direction_values = set(primary_budget_directions.values())
	cross_budget_direction_disagreement = (
		'positive' in direction_values and 'negative' in direction_values
	)
	mixed_primary_budget_directions = sorted(
		budget
		for budget, direction in primary_budget_directions.items()
		if direction == 'mixed'
	)
	hold_for_direction_ambiguity = bool(mean_median_disagreements) or (
		cross_budget_direction_disagreement
	) or bool(mixed_primary_budget_directions) or bool(low_primary_win_support)
	positive = (
		len(positive_budgets) >= thresholds.minimum_positive_budgets
		and not major_block
		and not one_sided_class_improvement
	)
	if hold_for_direction_ambiguity:
		label = 'HOLD'
	elif negative:
		label = 'NEGATIVE'
	elif positive:
		label = 'POSITIVE'
	else:
		label = 'HOLD'
	return {
		'label': label,
		'comparison_id': comparison_id,
		'positive_primary_budgets': positive_budgets,
		'negative_primary_budgets': negative_budgets,
		'major_monitored_class_degradation_budgets': major_degradation,
		'major_monitored_class_degradation_metrics': (
			major_degradation_metrics
		),
		'consistent_monitored_class_degradation': consistent_degradation,
		'systematic_monitored_class_improvement': systematic_improvement,
		'one_sided_monitored_class_improvement': one_sided_class_improvement,
		'mean_median_disagreements': mean_median_disagreements,
		'low_primary_win_support': low_primary_win_support,
		'primary_budget_directions': primary_budget_directions,
		'mixed_primary_budget_directions': mixed_primary_budget_directions,
		'cross_budget_direction_disagreement': (
			cross_budget_direction_disagreement
		),
		'hold_for_direction_ambiguity': hold_for_direction_ambiguity,
	}


def _write_tables(
	output_dir: Path, inspection: F3VoxelLabelBudgetResultsInspection
) -> tuple[Path, ...]:
	datasets = (
		inspection.job_metrics,
		inspection.paired_metrics,
		inspection.paired_deltas,
		inspection.summary_by_budget,
		inspection.monitored_class_summary,
		inspection.full_label_anchor,
	)
	paths = []
	for name, rows in zip(TABLE_NAMES, datasets, strict=True):
		path = output_dir / name
		_write_csv(path, rows)
		paths.append(path)
	return tuple(paths)


def _write_figures(
	output_dir: Path, inspection: F3VoxelLabelBudgetResultsInspection
) -> tuple[Path, ...]:
	import matplotlib.pyplot as plt  # noqa: PLC0415

	paths = []
	job_rows = inspection.job_metrics
	anchors = {
		str(row['model_role']): row for row in inspection.full_label_anchor
	}
	for metric, name, title in (
		('macro_f1', FIGURE_NAMES[0], 'Macro F1 by token-row label budget'),
		('mean_iou', FIGURE_NAMES[1], 'Mean IoU by token-row label budget'),
		(
			'balanced_accuracy',
			FIGURE_NAMES[2],
			'Balanced accuracy by token-row label budget',
		),
	):
		fig, ax = plt.subplots(figsize=(8, 4.5))
		x = list(range(len(REQUIRED_BUDGETS) + 1))
		for role in MODEL_ROLES:
			means = [
				statistics.fmean(
					float(row[metric])
					for row in job_rows
					if row['budget_id'] == budget and row['model_role'] == role
				)
				for budget in REQUIRED_BUDGETS
			]
			values = [*means, float(anchors[role][metric])]
			ax.plot(x, values, marker='o', label=MODEL_LABELS[role])
			for budget_index, budget in enumerate(REQUIRED_BUDGETS):
				seed_values = [
					float(row[metric])
					for row in job_rows
					if row['budget_id'] == budget and row['model_role'] == role
				]
				ax.scatter(
					[budget_index] * len(seed_values),
					seed_values,
					s=10,
					alpha=0.25,
				)
		ax.set_xticks(x, [*REQUIRED_BUDGETS, 'full\n(single seed 42)'])
		ax.set_ylabel(metric.replace('_', ' '))
		ax.set_title(title)
		ax.legend(fontsize='small')
		ax.grid(axis='y', alpha=0.2)
		fig.tight_layout()
		path = output_dir / name
		fig.savefig(path, dpi=120)
		plt.close(fig)
		paths.append(path)
	fig, axes = plt.subplots(2, 2, figsize=(10, 7))
	for ax, metric, title in zip(
		axes.flat,
		(
			'boundary_region_macro_f1_r2',
			'boundary_region_mean_iou_r2',
			'boundary_f1_t4',
			'vertical_boundary_position_mae',
		),
		(
			'Boundary-region macro F1 (r=2)',
			'Boundary-region mean IoU (r=2)',
			'Boundary F1 (t=4)',
			'Vertical boundary-position MAE (lower is better)',
		),
		strict=True,
	):
		for role in MODEL_ROLES:
			low_label_values = [
				statistics.fmean(
					float(row[metric])
					for row in job_rows
					if row['budget_id'] == budget and row['model_role'] == role
				)
				for budget in REQUIRED_BUDGETS
			]
			values = [*low_label_values, float(anchors[role][metric])]
			ax.plot(
				range(len(REQUIRED_BUDGETS) + 1),
				values,
				marker='o',
				label=MODEL_LABELS[role],
			)
		ax.set_xticks(
			range(len(REQUIRED_BUDGETS) + 1),
			[*REQUIRED_BUDGETS, 'full\n(single seed 42)'],
		)
		ax.set_title(title, fontsize='small')
		ax.grid(axis='y', alpha=0.2)
	axes.flat[0].legend(fontsize='small')
	fig.suptitle('Low-label boundary monitoring metrics')
	fig.tight_layout()
	path = output_dir / FIGURE_NAMES[3]
	fig.savefig(path, dpi=120)
	plt.close(fig)
	paths.append(path)
	return tuple(paths)


def _summary_payload(
	config: F3VoxelLabelBudgetResultsConfig,
	inspection: F3VoxelLabelBudgetResultsInspection,
) -> dict[str, object]:
	return {
		'artifact_type': SUMMARY_ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': 'COMPLETE',
		'scope': {
			'dataset': 'F3 facies benchmark original split',
			'claim': 'fixed-decoder sample efficiency only',
			'six_split_included': False,
		},
		'contract': {
			'budget_semantics': 'per_class_selected_token_row_cap',
			'budgets': list(REQUIRED_BUDGETS),
			'subsample_seeds': list(REQUIRED_SEEDS),
			'model_roles': list(MODEL_ROLES),
			'model_tags': dict(EXPECTED_MODEL_TAGS),
			'comparisons': [item[0] for item in COMPARISONS],
			'paired_statistical_unit': 'subsample_seed',
			'paired_seed_count_per_budget': len(REQUIRED_SEEDS),
			'p_values_computed': False,
			'confidence_intervals_computed': False,
			'voxel_independence_claimed': False,
			'standard_deviation_role': 'descriptive_sample_standard_deviation',
			'full_label_anchor': {
				'seed': 42,
				'single_existing_run': True,
				'included_in_paired_aggregates': False,
			},
			'decision_thresholds': config.decision.to_dict(),
		},
		'source_identities': dict(inspection.source_identities),
		'decoder_architecture': dict(inspection.decoder_architecture),
		'completion': {
			'dataset_count': 15,
			'job_count': 45,
			'best_checkpoint_inference_count': 45,
			'complete_evaluation_count': 45,
			'paired_identity_mismatch_count': 0,
		},
		'job_metrics': list(inspection.job_metrics),
		'paired_metrics': list(inspection.paired_metrics),
		'paired_deltas': list(inspection.paired_deltas),
		'summary_by_budget': list(inspection.summary_by_budget),
		'monitored_class_summary': list(inspection.monitored_class_summary),
		'full_label_anchor': list(inspection.full_label_anchor),
		'scientific_decisions': dict(inspection.decisions),
		'limitations': [
			'Five paired subsample seeds are the aggregation unit.',
			'No p-value, confidence interval, or voxel-independence claim is '
			'made.',
			'The full-label point is one existing seed-42 anchor, not a '
			'five-seed mean.',
			'Conclusions are limited to the F3 original split and fixed decoder.',
		],
	}


def _render_markdown(payload: Mapping[str, object]) -> str:
	decisions = _mapping(payload['scientific_decisions'], 'scientific_decisions')
	structured = _mapping(
		decisions['structured_pretext_vs_mae'], 'structured_pretext_vs_mae'
	)
	boundary = _mapping(
		decisions['boundary_aware_increment'], 'boundary_aware_increment'
	)
	summary = cast('Sequence[Mapping[str, object]]', payload['summary_by_budget'])
	index = {
		(str(row['budget_id']), str(row['comparison_id']), str(row['metric'])): row
		for row in summary
	}
	m1_label = _mapping(structured['m1_vs_mae'], 'm1 decision')['label']
	m2a_label = _mapping(structured['m2a_vs_mae'], 'm2a decision')['label']
	boundary_label = _mapping(
		boundary['m2a_vs_m1'], 'boundary decision'
	)['label']
	lines = [
		'# F3 original-split low-label voxel benchmark',
		'',
		'Status: **COMPLETE**',
		'',
		'Budgets are per-class selected token-row caps. Dense voxel labels inside '
		'selected token blocks are retained, and full validation is fixed.',
		'',
		'## Scientific decisions',
		'',
		f'- M1 vs MAE: **{m1_label}**',
		f'- M2-A vs MAE: **{m2a_label}**',
		f'- M2-A vs M1: **{boundary_label}**',
		'',
		'## Paired primary metrics',
		'',
		'| budget | comparison | mean Δ macro F1 | median Δ macro F1 | '
		'wins | mean Δ mean IoU | median Δ mean IoU | wins |',
		'|---|---|---:|---:|---:|---:|---:|---:|',
	]
	for budget_id in REQUIRED_BUDGETS:
		for comparison_id, _baseline, _candidate, label in COMPARISONS:
			macro = index[(budget_id, comparison_id, 'macro_f1')]
			iou = index[(budget_id, comparison_id, 'mean_iou')]
			lines.append(
				f'| {budget_id} | {label} | {_format(macro["mean_delta"])} | '
				f'{_format(macro["median_delta"])} | '
				f'{macro["positive_win_count"]}/5 | {_format(iou["mean_delta"])} | '
				f'{_format(iou["median_delta"])} | {iou["positive_win_count"]}/5 |'
			)
	lines.extend(
		[
			'',
			'## Interpretation limits',
			'',
			'- The aggregation unit is five paired subsample seeds, not voxels.',
			'- No p-values or confidence intervals were computed.',
			'- `full` is one reused seed-42 run and is excluded from paired wins.',
			'- Any sample-efficiency statement is limited to this F3 original split '
			'and fixed frozen-embedding decoder.',
			'- Six-split low-label robustness is outside this milestone.',
			'',
		]
	)
	return '\n'.join(lines)


def _render_readme() -> str:
	return (
		'# F3 voxel lithology low-label benchmark\n\n'
		'This lightweight bundle contains the complete original-split paired-seed '
		'summary, CSV tables, and figures.\n\n'
		'Raw checkpoints, predictions, embeddings, and label volumes remain under '
		'local artifact storage and are intentionally not published here.\n'
	)


def _publish(
	result: F3VoxelLabelBudgetResultsResult,
	config: F3VoxelLabelBudgetResultsConfig,
) -> PublishManifest | None:
	policy = config.publish
	if not policy.enabled:
		return None
	if policy.output_dir is None:
		raise ValueError('publish.output_dir is required')
	items = _publish_items(result)
	expected = {
		item.relative_target.as_posix() for item in items
	} | {LOCAL_PUBLISH_MANIFEST}
	_validate_existing_publish_tree(policy.output_dir, expected=expected)
	manifest = publish_selected_results(
		items=items,
		output_dir=policy.output_dir,
		allowed_suffixes=PUBLISH_SUFFIXES,
		max_file_size_bytes=policy.max_file_size_bytes,
		overwrite=policy.overwrite,
	)
	payload = _publish_manifest_payload(manifest)
	_write_json(manifest.manifest_path, payload)
	_validate_published_tree(policy.output_dir, expected=expected, manifest=manifest)
	return manifest


def _publish_items(
	result: F3VoxelLabelBudgetResultsResult,
) -> list[PublishItem]:
	return [
		PublishItem(result.summary_markdown, Path(SUMMARY_MARKDOWN)),
		PublishItem(result.summary_json, Path(SUMMARY_JSON)),
		PublishItem(result.readme, Path(README_NAME)),
		*(PublishItem(path, Path('tables') / path.name) for path in result.table_paths),
		*(
			PublishItem(path, Path('figures') / path.name)
			for path in result.figure_paths
		),
	]


def _validate_existing_publish_tree(output_dir: Path, *, expected: set[str]) -> None:
	if not output_dir.exists():
		return
	if not output_dir.is_dir():
		raise NotADirectoryError(f'publish output is not a directory: {output_dir}')
	existing = {
		path.relative_to(output_dir).as_posix()
		for path in output_dir.rglob('*')
		if path.is_file()
	}
	extra = sorted(existing - expected)
	if extra:
		raise ValueError(f'publish output contains unlisted files: {extra!r}')


def _validate_published_tree(
	output_dir: Path,
	*,
	expected: set[str],
	manifest: PublishManifest,
) -> None:
	actual = {
		path.relative_to(output_dir).as_posix()
		for path in output_dir.rglob('*')
		if path.is_file()
	}
	if actual != expected:
		raise ValueError(
			'published file inventory mismatch: '
			f'missing={sorted(expected - actual)}, extra={sorted(actual - expected)}'
		)
	manifest_targets = {
		item.target.resolve(strict=False).relative_to(
			output_dir.resolve(strict=False)
		).as_posix()
		for item in manifest.items
	}
	if manifest_targets != expected - {LOCAL_PUBLISH_MANIFEST}:
		raise ValueError('publish manifest target inventory mismatch')
	payload = _read_json(output_dir / LOCAL_PUBLISH_MANIFEST)
	inventory = _mapping_rows(payload.get('inventory'), 'publish inventory')
	inventory_targets = {
		_non_empty_string(item.get('target'), 'publish inventory target')
		for item in inventory
	}
	if inventory_targets != actual or len(inventory) != len(actual):
		raise ValueError('publish manifest does not enumerate the exact file tree')
	self_entries = [
		item for item in inventory if item.get('target') == LOCAL_PUBLISH_MANIFEST
	]
	if self_entries != [
		{
			'target': LOCAL_PUBLISH_MANIFEST,
			'source': None,
			'source_sha256': None,
			'published_sha256': None,
			'byte_size': None,
			'self_manifest': True,
			'hash_policy': 'omitted_due_to_self_reference',
		}
	]:
		raise ValueError('publish manifest self-inventory marker mismatch')
	for relative in actual:
		if Path(relative).suffix.lower() in FORBIDDEN_PUBLISH_SUFFIXES:
			raise ValueError(f'raw artifact escaped into publish output: {relative}')


def _write_local_publish_manifest(
	result: F3VoxelLabelBudgetResultsResult,
	config: F3VoxelLabelBudgetResultsConfig,
	manifest: PublishManifest | None,
) -> None:
	if manifest is not None:
		payload = _publish_manifest_payload(manifest)
		payload['published'] = True
	else:
		items = _publish_items(result)
		payload = {
			'artifact_type': 'f3_lithology_voxel_label_budget_publish_plan',
			'schema_version': 1,
			'published': False,
			'output_dir': (
				None
				if config.publish.output_dir is None
				else str(config.publish.output_dir)
			),
			'items': [
				{
					'source': str(item.source),
					'target': item.relative_target.as_posix(),
					'size_bytes': item.source.stat().st_size,
					'sha256': file_sha256(item.source),
				}
				for item in items
			],
			'warnings': ['publication disabled by configuration'],
		}
	_write_json(result.local_publish_manifest, payload)


def _publish_manifest_payload(manifest: PublishManifest) -> dict[str, object]:
	"""Record both source and copied hashes for every lightweight output."""
	payload = publish_manifest_to_dict(manifest)
	payload['artifact_type'] = 'f3_lithology_voxel_label_budget_publish_manifest'
	payload['schema_version'] = SCHEMA_VERSION
	items = cast('list[dict[str, object]]', payload['items'])
	for record, published in zip(items, manifest.items, strict=True):
		source_sha256 = file_sha256(published.source)
		published_sha256 = file_sha256(published.target)
		if source_sha256 != published.sha256:
			raise ValueError('publish source SHA-256 changed after copy')
		if published_sha256 != source_sha256:
			raise ValueError('published output SHA-256 differs from its source')
		if published.target.stat().st_size != published.size_bytes:
			raise ValueError('published output byte size changed after copy')
		record.update(
			{
				'source_sha256': source_sha256,
				'published_sha256': published_sha256,
				'byte_size': published.size_bytes,
			}
		)
	payload['inventory'] = [
		*[
			{
				'target': record['target'],
				'source': record['source'],
				'source_sha256': record['source_sha256'],
				'published_sha256': record['published_sha256'],
				'byte_size': record['byte_size'],
				'self_manifest': False,
			}
			for record in items
		],
		{
			'target': LOCAL_PUBLISH_MANIFEST,
			'source': None,
			'source_sha256': None,
			'published_sha256': None,
			'byte_size': None,
			'self_manifest': True,
			'hash_policy': 'omitted_due_to_self_reference',
		},
	]
	payload['inventory_contract'] = {
		'all_published_files_enumerated': True,
		'manifest_self_hash_omitted': True,
		'self_hash_reason': 'a cryptographic hash cannot include its own value',
	}
	return payload


def _validate_output_availability(config: F3VoxelLabelBudgetResultsConfig) -> None:
	if config.overwrite or not config.reports_dir.exists():
		return
	owned = {
		SUMMARY_JSON,
		SUMMARY_MARKDOWN,
		README_NAME,
		LOCAL_PUBLISH_MANIFEST,
		'tables',
		'figures',
	}
	conflicts = [path for path in config.reports_dir.iterdir() if path.name in owned]
	if conflicts:
		raise FileExistsError(
			'refusing existing voxel label-budget summary output: '
			f'{sorted(str(path) for path in conflicts)!r}'
		)


def _validated_identity(value: object, *, label: str) -> _FileIdentity:
	identity = _mapping(value, label)
	path = _path_value(identity.get('path'), f'{label}.path')
	sha256 = _sha256_value(identity.get('sha256'), f'{label}.sha256')
	result = _FileIdentity(path, sha256)
	_validate_file_identity(result, label=label)
	return result


def _file_identity(path: Path) -> _FileIdentity:
	if not path.is_file():
		raise FileNotFoundError(path)
	return _FileIdentity(path, file_sha256(path))


def _validate_file_identity(identity: _FileIdentity, *, label: str) -> None:
	if not identity.path.is_file():
		raise FileNotFoundError(f'missing {label}: {identity.path}')
	if file_sha256(identity.path) != identity.sha256:
		raise ValueError(f'{label} SHA-256 mismatch')


def _read_json(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	with path.open(encoding='utf-8') as handle:
		payload = json.load(handle)
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON document must contain an object: {path}')
	return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	if not rows:
		raise ValueError(f'cannot write empty summary table: {path}')
	fieldnames = list(rows[0])
	if any(set(row) != set(fieldnames) for row in rows):
		raise ValueError(f'summary table row schema mismatch: {path.name}')
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _mapping_rows(value: object, label: str) -> tuple[Mapping[str, object], ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a sequence')
	if not value or any(not isinstance(item, Mapping) for item in value):
		raise TypeError(f'{label} must contain mappings')
	return tuple(cast('Sequence[Mapping[str, object]]', value))


def _path_value(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	return Path(value)


def _non_empty_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError(f'{label} must be a positive integer')
	return value


def _nonnegative_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		raise ValueError(f'{label} must be a nonnegative integer')
	return value


def _finite_float(value: object, label: str) -> float:
	if isinstance(value, str):
		try:
			value = float(value)
		except ValueError as error:
			raise TypeError(f'{label} must be numeric') from error
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{label} must be numeric')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label} must be finite')
	return result


def _float_tuple(value: object, label: str) -> tuple[float, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a numeric sequence')
	items = tuple(_finite_float(item, label) for item in value)
	if not items or any(item <= 0.0 for item in items):
		raise ValueError(f'{label} must contain positive values')
	return items


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be an integer sequence')
	items = tuple(value)
	if not items or any(
		not isinstance(item, int) or isinstance(item, bool) for item in items
	):
		raise TypeError(f'{label} must contain integers')
	if len(set(items)) != len(items):
		raise ValueError(f'{label} contains duplicate class IDs')
	return cast('tuple[int, ...]', items)


def _required_budget(value: object, label: str) -> str:
	budget = _non_empty_string(value, label)
	if budget not in REQUIRED_BUDGETS:
		raise ValueError(f'{label} must be one of {REQUIRED_BUDGETS!r}')
	return budget


def _required_seed(value: object, label: str) -> int:
	seed = _nonnegative_int(value, label)
	if seed not in REQUIRED_SEEDS:
		raise ValueError(f'{label} must be one of {REQUIRED_SEEDS!r}')
	return seed


def _model_role(value: object, label: str) -> str:
	role = _non_empty_string(value, label)
	if role not in MODEL_ROLES:
		raise ValueError(f'{label} must be one of {MODEL_ROLES!r}')
	return role


def _sha256_value(value: object, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value.lower())
	):
		raise ValueError(f'{label} must be a hexadecimal SHA-256')
	return value.lower()


def _stable_json(value: object) -> str:
	return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _json_sha256(value: object) -> str:
	return hashlib.sha256(_stable_json(value).encode('utf-8')).hexdigest()


def _format(value: object) -> str:
	if isinstance(value, int | float) and not isinstance(value, bool):
		return f'{float(value):.6f}'
	return 'NA'


__all__ = [
	'COMPARISONS',
	'EXPECTED_MODEL_TAGS',
	'FIGURE_NAMES',
	'METRIC_SPECS',
	'MODEL_ROLES',
	'REQUIRED_BUDGETS',
	'REQUIRED_SEEDS',
	'RUN_MANIFEST_ARTIFACT_TYPE',
	'SUMMARY_ARTIFACT_TYPE',
	'SUMMARY_JSON',
	'SUMMARY_MARKDOWN',
	'TABLE_NAMES',
	'F3VoxelLabelBudgetResultsInspection',
	'F3VoxelLabelBudgetResultsResult',
	'inspect_f3_lithology_voxel_label_budget_results',
	'summarize_f3_lithology_voxel_label_budget_results',
]
