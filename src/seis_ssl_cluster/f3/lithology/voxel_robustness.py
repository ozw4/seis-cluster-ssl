"""Manifest-driven M1 versus M2-A voxel split robustness tooling."""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.config.f3_lithology_voxel_dataset import (
	F3LithologyVoxelDatasetConfig,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	F3LithologyVoxelDecoderConfig,
)
from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	F3LithologyVoxelEvaluationConfig,
)
from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	F3LithologyVoxelInferenceConfig,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.prediction import (
	F3LithologyPredictionConfig,
	F3LithologyPredictionInputs,
	F3LithologyPredictionOutputs,
	predict_f3_lithology_tokens,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	load_token_dataset_npz,
	paired_token_identity_hash,
)
from seis_ssl_cluster.f3.lithology.tokens import (
	F3LithologyTokenPolicy,
	load_f3_embedding_artifacts,
	read_f3_lithology_class_info,
)
from seis_ssl_cluster.f3.lithology.voxel_dataset import (
	COUNTS_NAME,
	GRID_NAME,
	MANIFEST_NAME,
	SUMMARY_NAME,
	build_f3_lithology_voxel_dataset,
	inspect_f3_lithology_voxel_dataset,
)
from seis_ssl_cluster.f3.lithology.voxel_dataset import (
	METADATA_NAME as VOXEL_METADATA_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_decoder_inference import (
	predict_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	BOUNDARY_METRICS_JSON,
	BOUNDARY_REGION_METRICS_CSV,
	EVALUATION_METADATA_JSON,
	EVALUATION_OUTPUT_FILES,
	METRICS_JSON,
	evaluate_f3_lithology_voxels,
	inspect_f3_lithology_voxel_evaluation,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	CONFIDENCE_NAME,
	METADATA_NAME,
	PREDICTIONS_NAME,
	VALID_MASK_NAME,
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_projection import (
	project_f3_lithology_tokens_to_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_results import (
	FIGURE_NAMES as ORIGINAL_FIGURE_NAMES,
)
from seis_ssl_cluster.f3.lithology.voxel_results import (
	SUMMARY_JSON as ORIGINAL_SUMMARY_JSON,
)
from seis_ssl_cluster.f3.lithology.voxel_results import (
	SUMMARY_MARKDOWN as ORIGINAL_SUMMARY_MARKDOWN,
)
from seis_ssl_cluster.f3.lithology.voxel_results import (
	TABLE_NAMES as ORIGINAL_TABLE_NAMES,
)
from seis_ssl_cluster.f3.lithology.voxel_results import (
	validate_f3_lithology_voxel_results_bundle,
)
from seis_ssl_cluster.f3.splits import read_f3_line_geometry
from seis_ssl_cluster.models.voxel_decoder.spec import (
	validate_voxel_decoder_architecture_mapping,
)
from seis_ssl_cluster.training.voxel_decoder.runner import (
	inspect_f3_lithology_voxel_decoder,
	run_f3_lithology_voxel_decoder,
)

if TYPE_CHECKING:
	from collections.abc import Iterable

	from seis_ssl_cluster.config.f3_lithology_voxel_robustness import (
		F3VoxelDecoderSplitSuiteConfig,
		F3VoxelSplitDatasetSuiteConfig,
		F3VoxelSplitRobustnessSummaryConfig,
		F3VoxelV0SplitSuiteConfig,
		VoxelRobustnessModel,
	)

SPLIT_DATASET_MANIFEST_TYPE = 'f3_lithology_voxel_split_dataset_manifest'
V0_RUN_MANIFEST_TYPE = 'f3_lithology_voxel_v0_split_run_manifest'
V1_RUN_MANIFEST_TYPE = 'f3_lithology_voxel_decoder_split_run_manifest'
SUMMARY_ARTIFACT_TYPE = 'f3_lithology_voxel_split_robustness_summary'
INVENTORY_ARTIFACT_TYPE = 'f3_lithology_split_inventory_manifest'
SCHEMA_VERSION = 1
SOURCE_SPLIT_IDS = frozenset(f'split_{index:03d}' for index in range(6))
V0_PROBE_SPEC = 'linear_balanced_v1'

PRIMARY_METRICS = ('macro_f1', 'mean_iou')
SUMMARY_METRICS = (
	'macro_f1',
	'mean_iou',
	'balanced_accuracy',
	'boundary_region_macro_f1_radius_2',
	'boundary_region_mean_iou_radius_2',
	'boundary_region_macro_f1_radius_4',
	'boundary_region_mean_iou_radius_4',
	'vertical_boundary_f1_tolerance_2',
	'vertical_boundary_f1_tolerance_4',
	'boundary_position_mae',
	'class_3_f1',
	'class_3_iou',
	'class_3_boundary_recall_tolerance_2',
	'class_3_boundary_recall_tolerance_4',
	'class_5_f1',
	'class_5_iou',
	'class_5_boundary_recall_tolerance_2',
	'class_5_boundary_recall_tolerance_4',
)
LOWER_IS_BETTER = frozenset({'boundary_position_mae'})
PUBLISH_SUFFIXES = frozenset({'.md', '.json', '.csv', '.png'})
ROBUSTNESS_SUMMARY_MARKDOWN = 'voxel_split_robustness_summary.md'


@dataclass(frozen=True)
class VoxelSplitJob:
	"""One split/model/stage job in a paired suite matrix."""

	split_id: str
	model_role: str
	model_tag: str
	output_root: Path


@dataclass(frozen=True)
class VoxelSplitSuiteResult:
	"""Manifest and rows emitted by a suite stage."""

	manifest_json: Path
	rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class VoxelSplitRobustnessSummaryResult:
	"""Files and provisional status emitted by split-level aggregation."""

	output_dir: Path
	summary_json: Path
	paired_rows_csv: Path
	aggregates_csv: Path
	status: str
	summary_markdown: Path | None = None
	published_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class VoxelSplitRobustnessInspection:
	"""Validated split-level rows and decision computed without writing."""

	paired_rows: tuple[Mapping[str, object], ...]
	aggregates: tuple[Mapping[str, object], ...]
	status: str
	reasons: tuple[str, ...]
	decoder_architecture: Mapping[str, object]


def build_f3_lithology_voxel_split_datasets(
	config: F3VoxelSplitDatasetSuiteConfig,
	*,
	only_missing: bool = False,
) -> VoxelSplitSuiteResult:
	"""Build voxel supervision from the existing split inventory, without draws."""
	inventory = _read_manifest_rows(
		config.split_inventory_manifest,
		artifact_type=INVENTORY_ARTIFACT_TYPE,
		required=('split_id', 'png_label_inventory'),
	)
	rows: list[dict[str, object]] = []
	for source in inventory:
		split_id = _split_id(source)
		output_dir = config.output_root / 'voxel_datasets' / split_id
		job_config = _split_dataset_job_config(config, source, output_dir)
		complete = _complete_voxel_dataset(output_dir)
		if only_missing and complete:
			_validate_existing_voxel_dataset(job_config)
		else:
			build_f3_lithology_voxel_dataset(job_config)
		row = _voxel_dataset_row(split_id, source, output_dir)
		if _mapping_value(row, 'reference_valid_tokens').get('sha256') != file_sha256(
			config.reference_valid_tokens
		):
			raise ValueError(
				f'voxel dataset canonical valid-mask identity mismatch for {split_id}'
			)
		rows.append(row)
	manifest = config.output_root / 'voxel_split_dataset_manifest.json'
	_write_json(
		manifest,
		{
			'artifact_type': SPLIT_DATASET_MANIFEST_TYPE,
			'schema_version': SCHEMA_VERSION,
			'source_split_inventory_manifest': _identity(
				config.split_inventory_manifest
			),
			'canonical_reference_valid_tokens': _identity(
				config.reference_valid_tokens
			),
			'rows': rows,
		},
	)
	return VoxelSplitSuiteResult(manifest, tuple(rows))


def _split_dataset_job_config(
	config: F3VoxelSplitDatasetSuiteConfig,
	source: Mapping[str, object],
	output_dir: Path,
) -> F3LithologyVoxelDatasetConfig:
	return F3LithologyVoxelDatasetConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		source_label_volume=config.source_label_volume,
		source_label_segy=config.source_label_segy,
		png_label_inventory=Path(str(source['png_label_inventory'])),
		class_info=config.class_info,
		segy_geometry_json=config.segy_geometry_json,
		reference_metadata_json=config.reference_metadata_json,
		reference_valid_tokens=config.reference_valid_tokens,
		output_dir=output_dir,
		ignore_z_border_samples=config.ignore_z_border_samples,
		overwrite=config.overwrite,
	)


def voxel_split_dataset_jobs(
	config: F3VoxelSplitDatasetSuiteConfig,
) -> tuple[VoxelSplitJob, ...]:
	"""Return the supervision dry-run matrix after validating source identities."""
	rows = _read_manifest_rows(
		config.split_inventory_manifest,
		artifact_type=INVENTORY_ARTIFACT_TYPE,
		required=('split_id', 'png_label_inventory'),
	)
	jobs = []
	for row in rows:
		split_id = _split_id(row)
		output_root = config.output_root / 'voxel_datasets' / split_id
		inspect_f3_lithology_voxel_dataset(
			_split_dataset_job_config(config, row, output_root)
		)
		jobs.append(
			VoxelSplitJob(
				split_id,
				'shared',
				'shared_voxel_supervision',
				output_root,
			)
		)
	return tuple(jobs)


def voxel_v0_split_jobs(
	config: F3VoxelV0SplitSuiteConfig,
) -> tuple[VoxelSplitJob, ...]:
	"""Validate paired source manifests and return the V0 job matrix."""
	_validate_split_inventory_identity(
		config.voxel_dataset_manifest, config.split_dataset_manifest
	)
	voxel_rows = _voxel_rows(config.voxel_dataset_manifest)
	dataset_rows = _paired_rows(
		config.split_dataset_manifest,
		artifact_type='f3_lithology_split_sweep_token_dataset_manifest',
		required=(
			'split_id',
			'model_role',
			'model_tag',
			'token_dataset_root',
			'validation_tokens',
			'paired_identity_hash',
		),
	)
	probe_rows = _probe_rows(config.probe_run_manifest)
	_validate_job_sources(voxel_rows, dataset_rows, probe_rows, config.models)
	_validate_v0_source_artifacts(config, voxel_rows, dataset_rows, probe_rows)
	return tuple(
		VoxelSplitJob(
			str(row['split_id']),
			str(row['model_role']),
			str(row['model_tag']),
			_v0_job_root(config.output_root, row),
		)
		for row in dataset_rows
	)


def run_f3_lithology_voxel_v0_split_suite(
	config: F3VoxelV0SplitSuiteConfig,
	*,
	only_missing: bool = False,
) -> VoxelSplitSuiteResult:
	"""Run full-token prediction, nearest projection, and common evaluation."""
	jobs = voxel_v0_split_jobs(config)
	voxel_by_split = {
		str(row['split_id']): row for row in _voxel_rows(config.voxel_dataset_manifest)
	}
	dataset_by_key = _rows_by_key(_manifest_rows(config.split_dataset_manifest))
	probe_by_key = _rows_by_key(_probe_rows(config.probe_run_manifest))
	models = {model.role: model for model in config.models}
	rows: list[dict[str, object]] = []
	manifest = config.output_root / 'v0_split_run_manifest.json'
	for job in jobs:
		key = (job.split_id, job.model_role)
		dataset_row = dataset_by_key[key]
		probe_row = probe_by_key[key]
		voxel_row = voxel_by_split[job.split_id]
		row = _run_v0_job(
			config,
			job,
			model=models[job.model_role],
			dataset_row=dataset_row,
			probe_row=probe_row,
			voxel_row=voxel_row,
			only_missing=only_missing,
		)
		rows.append(row)
		_write_run_manifest(manifest, V0_RUN_MANIFEST_TYPE, rows)
	return VoxelSplitSuiteResult(manifest, tuple(rows))


def voxel_decoder_split_jobs(
	config: F3VoxelDecoderSplitSuiteConfig,
) -> tuple[VoxelSplitJob, ...]:
	"""Validate shared V1 inputs and return the paired decoder job matrix."""
	voxel_rows = _voxel_rows(config.voxel_dataset_manifest)
	_validate_embedding_valid_masks(config.models, dataset_name=config.dataset['name'])
	jobs = []
	for row in voxel_rows:
		for model in config.models:
			job = VoxelSplitJob(
				str(row['split_id']),
				model.role,
				model.model_tag,
				config.output_root
				/ 'v1'
				/ config.decoder.spec
				/ f'split={row["split_id"]}'
				/ f'model={model.model_tag}',
			)
			inspect_f3_lithology_voxel_decoder(
				_decoder_job_config(config, job, model=model, voxel_row=row)
			)
			jobs.append(job)
	return tuple(jobs)


def run_f3_lithology_voxel_decoder_split_suite(
	config: F3VoxelDecoderSplitSuiteConfig,
	*,
	only_missing: bool = False,
	device: str = 'auto',
) -> VoxelSplitSuiteResult:
	"""Train, predict, and evaluate paired V1 decoders with resumable rows."""
	jobs = voxel_decoder_split_jobs(config)
	voxel_by_split = {
		str(row['split_id']): row for row in _voxel_rows(config.voxel_dataset_manifest)
	}
	models = {model.role: model for model in config.models}
	manifest = config.output_root / 'v1_split_run_manifest.json'
	prior = _prior_run_rows(manifest, artifact_type=V1_RUN_MANIFEST_TYPE)
	rows: list[dict[str, object]] = []
	for job in jobs:
		key = (job.split_id, job.model_role)
		model = models[job.model_role]
		resume = _resume_path(prior.get(key), job.output_root / 'decoder')
		try:
			_run_v1_job(
				config,
				job,
				model=model,
				voxel_row=voxel_by_split[job.split_id],
				device=device,
				resume=resume if only_missing else None,
				only_missing=only_missing,
			)
			row = _v1_manifest_row(
				job,
				voxel_by_split[job.split_id],
				model=model,
				dataset_name=config.dataset['name'],
				status='complete',
			)
			_validate_paired_decoder_identity((*rows, row), split_id=job.split_id)
		except BaseException as exc:
			row = _v1_manifest_row(
				job,
				voxel_by_split[job.split_id],
				model=model,
				dataset_name=config.dataset['name'],
				status='failed',
				failure=f'{type(exc).__name__}: {exc}',
			)
			rows.append(row)
			_write_run_manifest(manifest, V1_RUN_MANIFEST_TYPE, rows)
			raise
		rows.append(row)
		_write_run_manifest(manifest, V1_RUN_MANIFEST_TYPE, rows)
	_write_run_manifest(manifest, V1_RUN_MANIFEST_TYPE, rows)
	return VoxelSplitSuiteResult(manifest, tuple(rows))


def summarize_f3_lithology_voxel_split_robustness(
	config: F3VoxelSplitRobustnessSummaryConfig,
) -> VoxelSplitRobustnessSummaryResult:
	"""Aggregate paired metrics with split, never voxel, as statistical unit."""
	inspection = inspect_f3_lithology_voxel_split_robustness(config)
	paired = inspection.paired_rows
	aggregates = inspection.aggregates
	status = inspection.status
	reasons = inspection.reasons
	output_dir = config.suite_root / 'reports'
	paired_csv = output_dir / 'voxel_split_paired_metrics.csv'
	aggregates_csv = output_dir / 'voxel_split_aggregates.csv'
	_write_csv(paired_csv, paired)
	_write_csv(aggregates_csv, aggregates)
	summary_json = output_dir / 'voxel_split_robustness_summary.json'
	_write_json(
		summary_json,
		{
			'artifact_type': SUMMARY_ARTIFACT_TYPE,
			'schema_version': SCHEMA_VERSION,
			'comparison': {
				'baseline': config.baseline_model_tag,
				'candidate': config.candidate_model_tag,
			},
			'decoder_architecture': dict(inspection.decoder_architecture),
			'statistical_unit': 'split',
			'voxel_level_significance_computed': False,
			'confidence_intervals_computed': False,
			'p_values_computed': False,
			'split_count': len({str(row['split_id']) for row in paired}),
			'raw_rows': paired,
			'aggregates': aggregates,
			'provisional_status': status,
			'm2a_vs_m1_voxel_robustness': status,
			'status_reasons': reasons,
		},
	)
	summary_markdown = output_dir / ROBUSTNESS_SUMMARY_MARKDOWN
	summary_markdown.write_text(
		_render_robustness_markdown(
			decoder_architecture=inspection.decoder_architecture,
			status=status,
			split_count=len({str(row['split_id']) for row in paired}),
		),
		encoding='utf-8',
	)
	result = VoxelSplitRobustnessSummaryResult(
		output_dir,
		summary_json,
		paired_csv,
		aggregates_csv,
		status,
		summary_markdown,
	)
	return replace(result, published_files=_publish_robustness_summary(result, config))


def inspect_f3_lithology_voxel_split_robustness(
	config: F3VoxelSplitRobustnessSummaryConfig,
) -> VoxelSplitRobustnessInspection:
	"""Validate all summary inputs and compute the split decision without writes."""
	v0 = _complete_paired_run_rows(config.v0_run_manifest, V0_RUN_MANIFEST_TYPE)
	v1 = _complete_paired_run_rows(config.v1_run_manifest, V1_RUN_MANIFEST_TYPE)
	expected_keys = {
		(split_id, role)
		for split_id in SOURCE_SPLIT_IDS
		for role in ('baseline', 'candidate')
	}
	if set(v0) != expected_keys or set(v1) != expected_keys:
		raise ValueError(
			'V0 and V1 run manifests must contain all six split_000 through '
			'split_005 baseline/candidate pairs'
		)
	_expected_tags = {
		'baseline': config.baseline_model_tag,
		'candidate': config.candidate_model_tag,
	}
	for stage_rows in (v0, v1):
		for key, row in stage_rows.items():
			if row['model_tag'] != _expected_tags[key[1]]:
				raise ValueError(f'run manifest model identity mismatch for {key!r}')
	for split_id in SOURCE_SPLIT_IDS:
		identities = {
			str(stage_rows[(split_id, role)]['voxel_dataset_identity'])
			for stage_rows in (v0, v1)
			for role in ('baseline', 'candidate')
		}
		if len(identities) != 1:
			raise ValueError(f'voxel dataset identity mismatch for {split_id}')
	metric_rows = []
	for key in sorted(v1):
		split_id, role = key
		metric_rows.append(
			{
				'split_id': split_id,
				'model_role': role,
				'model_tag': str(v1[key]['model_tag']),
				**{
					f'v0_{name}': value
					for name, value in _evaluation_metrics(
						Path(str(v0[key]['evaluation_dir']))
					).items()
				},
				**{
					f'v1_{name}': value
					for name, value in _evaluation_metrics(
						Path(str(v1[key]['evaluation_dir']))
					).items()
				},
			}
		)
	paired = _paired_metric_rows(metric_rows)
	aggregates = _aggregate_metric_rows(paired)
	status, reasons = _provisional_status(paired, aggregates)
	if config.publish.enabled:
		if config.original_summary_dir is None:
			raise ValueError(
				'final publish requires inputs.original_summary_dir'
			)
		validate_f3_lithology_voxel_results_bundle(config.original_summary_dir)
	decoder_architecture = validate_voxel_decoder_architecture_mapping(
		next(iter(v1.values())).get('decoder_architecture'),
		field_prefix='V1 split decoder_architecture',
	)
	return VoxelSplitRobustnessInspection(
		tuple(paired),
		tuple(aggregates),
		status,
		tuple(reasons),
		decoder_architecture,
	)


def _publish_robustness_summary(
	result: VoxelSplitRobustnessSummaryResult,
	config: F3VoxelSplitRobustnessSummaryConfig,
) -> tuple[Path, ...]:
	policy = config.publish
	if not policy.enabled:
		return ()
	if policy.output_dir is None or config.original_summary_dir is None:
		raise ValueError(
			'final publish requires publish.output_dir and inputs.original_summary_dir'
		)
	original = config.original_summary_dir
	validate_f3_lithology_voxel_results_bundle(original)
	sources = [
		(
			original / ORIGINAL_SUMMARY_MARKDOWN,
			Path(ORIGINAL_SUMMARY_MARKDOWN),
		),
		(original / ORIGINAL_SUMMARY_JSON, Path(ORIGINAL_SUMMARY_JSON)),
		*(
			(original / 'tables' / name, Path('tables') / name)
			for name in ORIGINAL_TABLE_NAMES
		),
		*(
			(original / 'figures' / name, Path('figures') / name)
			for name in ORIGINAL_FIGURE_NAMES
		),
		(result.summary_json, Path('robustness') / result.summary_json.name),
		(
			result.paired_rows_csv,
			Path('robustness') / 'tables' / result.paired_rows_csv.name,
		),
		(
			result.aggregates_csv,
			Path('robustness') / 'tables' / result.aggregates_csv.name,
		),
	]
	if result.summary_markdown is not None:
		sources.append(
			(
				result.summary_markdown,
				Path('robustness') / result.summary_markdown.name,
			)
		)
	entries = tuple(
		(source, policy.output_dir / relative_target)
		for source, relative_target in sources
	)
	_preflight_robustness_publish_entries(
		entries,
		max_file_size_bytes=policy.max_file_size_bytes,
		overwrite=policy.overwrite,
	)
	for source, target in entries:
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(source, target)
	return tuple(target for _, target in entries)


def _preflight_robustness_publish_entries(
	entries: Sequence[tuple[Path, Path]],
	*,
	max_file_size_bytes: int,
	overwrite: bool,
) -> None:
	if (
		isinstance(max_file_size_bytes, bool)
		or not isinstance(max_file_size_bytes, int)
		or max_file_size_bytes <= 0
	):
		raise ValueError('max_file_size_bytes must be a positive integer')
	targets: set[Path] = set()
	for source, target_path in entries:
		if source.is_symlink() or not source.is_file():
			raise FileNotFoundError(
				f'required publish source must be a regular file: {source}'
			)
		target = target_path.resolve(strict=False)
		if source.resolve(strict=False) == target:
			raise ValueError(f'publish target must differ from source: {target_path}')
		if target in targets:
			raise ValueError(f'duplicate publish target: {target_path}')
		targets.add(target)
		if target_path.is_symlink():
			raise ValueError(f'publish target must not be a symlink: {target_path}')
		if target_path.exists() and not target_path.is_file():
			raise IsADirectoryError(f'publish target is not a file: {target_path}')
		if target_path.exists() and not overwrite:
			raise FileExistsError(f'publish target already exists: {target_path}')
		if source.stat().st_size > max_file_size_bytes:
			raise ValueError(f'publish source exceeds max_file_size_bytes: {source}')


def _run_v0_job(  # noqa: PLR0913
	config: F3VoxelV0SplitSuiteConfig,
	job: VoxelSplitJob,
	*,
	model: VoxelRobustnessModel,
	dataset_row: Mapping[str, object],
	probe_row: Mapping[str, object],
	voxel_row: Mapping[str, object],
	only_missing: bool,
) -> dict[str, object]:
	if model.checkpoint is None:
		raise ValueError(f'V0 model checkpoint is required for {model.model_tag}')
	_validate_model_embedding_checkpoint(model, dataset_name=config.dataset['name'])
	root = job.output_root
	token_dir = root / 'token_predictions'
	probe_dir = Path(str(probe_row['probe_output_dir']))
	policy = config.tokenization
	prediction = F3LithologyPredictionConfig(
		inputs=F3LithologyPredictionInputs(
			embeddings_dir=model.embeddings_dir,
			probe_joblib=probe_dir / 'probe.joblib',
			scaler_joblib=probe_dir / 'scaler.joblib',
			label_volume=config.source_label_volume,
			class_info=config.class_info,
			png_label_inventory=Path(str(voxel_row['png_label_inventory'])),
			segy_geometry_json=config.segy_geometry_json,
			source_label_segy=config.source_label_segy,
			validation_tokens=Path(str(dataset_row['validation_tokens'])),
		),
		outputs=F3LithologyPredictionOutputs(
			output_dir=token_dir,
			token_predictions=token_dir / 'f3_token_predictions.npy',
			probability_volume=token_dir / 'f3_token_probabilities.npy',
			valid_token_grid=token_dir / 'f3_valid_token_grid.npy',
			metadata_json=token_dir / 'prediction_metadata.json',
			validation_slice_metrics_csv=token_dir / 'validation_slice_metrics.csv',
		),
		classes=read_f3_lithology_class_info(config.class_info),
		token_policy=F3LithologyTokenPolicy(
			min_labeled_fraction=float(policy['min_labeled_fraction']),
			min_majority_fraction=float(policy['min_majority_fraction']),
			ignore_z_border_samples=int(policy['ignore_z_border_samples']),
		),
		dataset=config.dataset,
		model={'tag': model.model_tag, 'freeze_encoder': True},
		embeddings={'input_dir': str(model.embeddings_dir)},
		labels={'class_info': str(config.class_info)},
		lithology={
			'root': str(dataset_row['token_dataset_root']),
			'paired_identity_hash': str(dataset_row['paired_identity_hash']),
			'validation_tokens': _identity(Path(str(dataset_row['validation_tokens']))),
		},
		probe={
			'spec': 'linear_balanced_v1',
			'paired_identity_hash': str(probe_row['paired_identity_hash']),
			'probe_joblib': _identity(probe_dir / 'probe.joblib'),
			'scaler_joblib': _identity(probe_dir / 'scaler.joblib'),
		},
		batch_size=config.batch_size,
	)
	predict_f3_lithology_tokens(
		prediction,
		skip_existing=only_missing and _complete_token_prediction(token_dir),
	)
	projection_dir = root / 'voxel_predictions'
	if only_missing and _complete_prediction(projection_dir):
		projection = validate_f3_voxel_prediction_artifact(projection_dir)
	else:
		project_f3_lithology_tokens_to_voxels(token_dir, projection_dir)
		projection = validate_f3_voxel_prediction_artifact(projection_dir)
	_validate_v0_prediction_source(projection.metadata, token_dir=token_dir)
	evaluation = _evaluation_config(
		config, voxel_row, projection_dir, root / 'evaluation'
	)
	_run_or_validate_evaluation(
		evaluation,
		expected_model_tag=model.model_tag,
		expected_prediction_kind='token_projection_nearest',
		only_missing=only_missing,
	)
	return {
		'split_id': job.split_id,
		'model_role': job.model_role,
		'model_tag': job.model_tag,
		'checkpoint': _identity(model.checkpoint),
		'paired_identity_hash': dataset_row['paired_identity_hash'],
		'voxel_dataset_identity': voxel_row['split_grid']['sha256'],
		'token_prediction_dir': str(job.output_root / 'token_predictions'),
		'prediction_dir': str(job.output_root / 'voxel_predictions'),
		'evaluation_dir': str(job.output_root / 'evaluation'),
		'status': 'complete',
	}


def _run_v1_job(  # noqa: PLR0913
	config: F3VoxelDecoderSplitSuiteConfig,
	job: VoxelSplitJob,
	*,
	model: VoxelRobustnessModel,
	voxel_row: Mapping[str, object],
	device: str,
	resume: Path | None,
	only_missing: bool,
) -> None:
	decoder_dir = job.output_root / 'decoder'
	train_config = _decoder_job_config(
		config, job, model=model, voxel_row=voxel_row
	)
	if only_missing and _complete_decoder(decoder_dir):
		_validate_existing_decoder(train_config, decoder_dir)
		best_checkpoint = decoder_dir / 'best.pt'
	else:
		train_result = run_f3_lithology_voxel_decoder(
			train_config, device=device, resume=resume
		)
		if not train_result.completed:
			raise RuntimeError(
				'decoder job did not complete; resume from '
				f'{train_result.latest_checkpoint}'
			)
		best_checkpoint = train_result.best_checkpoint
	prediction_dir = job.output_root / 'voxel_predictions'
	inference = F3LithologyVoxelInferenceConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		model={'tag': model.model_tag, 'freeze_encoder': True},
		class_info=config.class_info,
		embeddings_input_dir=model.embeddings_dir,
		checkpoint=best_checkpoint,
		tiles=config.tiles,
		output_dir=prediction_dir,
		write_probabilities=config.write_probabilities,
		overwrite=config.overwrite,
	)
	if only_missing and _complete_prediction(prediction_dir):
		prediction = validate_f3_voxel_prediction_artifact(prediction_dir)
	else:
		predict_f3_lithology_voxels(inference, device=device)
		prediction = validate_f3_voxel_prediction_artifact(prediction_dir)
	_validate_v1_prediction_source(prediction.metadata, decoder_dir=decoder_dir)
	evaluation = _evaluation_config(
		config, voxel_row, prediction_dir, job.output_root / 'evaluation'
	)
	_run_or_validate_evaluation(
		evaluation,
		expected_model_tag=model.model_tag,
		expected_prediction_kind='frozen_embedding_decoder',
		only_missing=only_missing,
	)


def _decoder_job_config(
	config: F3VoxelDecoderSplitSuiteConfig,
	job: VoxelSplitJob,
	*,
	model: VoxelRobustnessModel,
	voxel_row: Mapping[str, object],
) -> F3LithologyVoxelDecoderConfig:
	return F3LithologyVoxelDecoderConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		model={'tag': model.model_tag, 'freeze_encoder': True},
		embeddings_input_dir=model.embeddings_dir,
		voxel_dataset_input_dir=Path(str(voxel_row['voxel_dataset_root'])),
		decoder=config.decoder,
		tiles=config.tiles,
		train=config.train,
		output_dir=job.output_root / 'decoder',
		embeddings={'spec': 'overlap_x16'},
	)


def _evaluation_config(
	config: F3VoxelV0SplitSuiteConfig | F3VoxelDecoderSplitSuiteConfig,
	voxel_row: Mapping[str, object],
	prediction_dir: Path,
	output_dir: Path,
) -> F3LithologyVoxelEvaluationConfig:
	policy = config.evaluation
	return F3LithologyVoxelEvaluationConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		prediction_input_dir=prediction_dir,
		voxel_dataset_input_dir=Path(str(voxel_row['voxel_dataset_root'])),
		source_label_volume=config.source_label_volume,
		source_label_segy=config.source_label_segy,
		png_label_inventory=Path(str(voxel_row['png_label_inventory'])),
		segy_geometry_json=config.segy_geometry_json,
		class_info=config.class_info,
		output_dir=output_dir,
		monitored_class_ids=tuple(
			int(item)
			for item in cast('Sequence[object]', policy['monitored_class_ids'])
		),
		boundary_tolerances=tuple(
			int(item)
			for item in cast('Sequence[object]', policy['boundary_tolerances'])
		),
		boundary_region_radii=tuple(
			int(item)
			for item in cast('Sequence[object]', policy['boundary_region_radii'])
		),
		chunk_size_x=int(policy['chunk_size_x']),
		overwrite=config.overwrite,
	)


def _run_or_validate_evaluation(
	config: F3LithologyVoxelEvaluationConfig,
	*,
	expected_model_tag: str,
	expected_prediction_kind: str,
	only_missing: bool,
) -> None:
	inspection = inspect_f3_lithology_voxel_evaluation(config)
	prediction_metadata = inspection.prediction_artifact.metadata
	if prediction_metadata.get('model_tag') != expected_model_tag:
		raise ValueError('existing voxel prediction model identity does not match job')
	if prediction_metadata.get('prediction_kind') != expected_prediction_kind:
		raise ValueError('existing voxel prediction kind does not match job')
	if not (only_missing and _complete_evaluation(config.output_dir)):
		evaluate_f3_lithology_voxels(config)
		return
	metadata = _read_json(config.output_dir / EVALUATION_METADATA_JSON)
	for key, expected in (
		('dataset', dict(config.dataset)),
		('model_tag', expected_model_tag),
		('prediction_kind', expected_prediction_kind),
	):
		if metadata.get(key) != expected:
			raise ValueError(f'existing voxel evaluation {key} identity mismatch')
	expected_policy = {
		'monitored_class_ids': list(config.monitored_class_ids),
		'boundary_tolerances': list(config.boundary_tolerances),
		'boundary_region_radii': list(config.boundary_region_radii),
		'primary_trace_boundary_tolerance': max(config.boundary_tolerances),
		'chunk_size_x': config.chunk_size_x,
	}
	if metadata.get('policy') != expected_policy:
		raise ValueError('existing voxel evaluation policy identity mismatch')
	prediction_dir = inspection.prediction_artifact.output_dir
	expected_inputs = {
		'prediction_metadata': prediction_dir / METADATA_NAME,
		'voxel_predictions': prediction_dir / PREDICTIONS_NAME,
		'voxel_confidence': prediction_dir / CONFIDENCE_NAME,
		'voxel_valid_mask': prediction_dir / VALID_MASK_NAME,
		'voxel_dataset_metadata': config.voxel_dataset_input_dir / VOXEL_METADATA_NAME,
		'voxel_split_grid': config.voxel_dataset_input_dir / GRID_NAME,
		'label_volume': config.source_label_volume,
		'png_label_inventory': config.png_label_inventory,
		'source_label_segy': config.source_label_segy,
		'segy_geometry_json': config.segy_geometry_json,
		'class_info': config.class_info,
	}
	inputs = _mapping_value(metadata, 'inputs')
	for key, path in expected_inputs.items():
		_validate_recorded_identity(inputs.get(key), path, label=f'evaluation {key}')


def _validate_existing_decoder(  # noqa: C901
	config: F3LithologyVoxelDecoderConfig, decoder_dir: Path
) -> None:
	resolved_config = config.to_dict()
	if _read_json(decoder_dir / 'resolved_config.json') != resolved_config:
		raise ValueError('existing decoder resolved config identity mismatch')
	import torch  # noqa: PLC0415

	payload = torch.load(
		decoder_dir / 'latest.pt', map_location='cpu', weights_only=False
	)
	if not isinstance(payload, Mapping):
		raise TypeError('existing decoder checkpoint must contain a mapping')
	if payload.get('checkpoint_kind') != 'completed':
		raise ValueError('existing decoder checkpoint is not complete')
	if payload.get('resolved_config') != resolved_config:
		raise ValueError('existing decoder checkpoint config identity mismatch')
	identities = _mapping_value(payload, 'artifact_identities')
	for key, identity in identities.items():
		if key == 'name':
			continue
		if not isinstance(identity, Mapping):
			raise TypeError(f'decoder artifact identity {key} must be a mapping')
		path = identity.get('path')
		if not isinstance(path, str) or not path:
			raise TypeError(f'decoder artifact identity {key} requires a path')
		_validate_recorded_identity(identity, Path(path), label=f'decoder {key}')
	best_hash = payload.get('best_checkpoint_sha256')
	if best_hash != file_sha256(decoder_dir / 'best.pt'):
		raise ValueError('existing decoder best checkpoint identity mismatch')
	manifest_hashes = _mapping_value(payload, 'tile_manifest_hashes')
	for split in ('train', 'validation'):
		manifest = _read_json(decoder_dir / f'{split}_tile_manifest.json')
		if manifest_hashes.get(split) != manifest.get('identity_sha256'):
			raise ValueError(f'existing decoder {split} tile identity mismatch')


def _validate_model_embedding_checkpoint(
	model: VoxelRobustnessModel, *, dataset_name: str
) -> None:
	if model.checkpoint is None:
		return
	metadata = _read_json(output_paths(model.embeddings_dir, dataset_name).metadata)
	checkpoint_path = metadata.get('checkpoint_path')
	if not isinstance(checkpoint_path, str) or (
		Path(checkpoint_path).resolve(strict=False)
		!= model.checkpoint.resolve(strict=False)
	):
		raise ValueError('model checkpoint path does not match embedding identity')
	if metadata.get('checkpoint_sha256') != file_sha256(model.checkpoint):
		raise ValueError('model checkpoint hash does not match embedding identity')


def _validate_recorded_identity(
	value: object, expected_path: Path, *, label: str
) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} identity must be a mapping')
	path = value.get('path')
	if not isinstance(path, str) or (
		Path(path).resolve(strict=False) != expected_path.resolve(strict=False)
	):
		raise ValueError(f'{label} path identity mismatch')
	if value.get('sha256') != file_sha256(expected_path):
		raise ValueError(f'{label} hash identity mismatch')


def _validate_v0_prediction_source(
	metadata: Mapping[str, object], *, token_dir: Path
) -> None:
	source = _mapping_value(metadata, 'source_identity')
	files = _mapping_value(source, 'token_artifact_files')
	for key, name in (
		('token_predictions', 'f3_token_predictions.npy'),
		('token_probabilities', 'f3_token_probabilities.npy'),
		('valid_token_grid', 'f3_valid_token_grid.npy'),
		('prediction_metadata', 'prediction_metadata.json'),
	):
		_validate_recorded_identity(
			files.get(key), token_dir / name, label=f'V0 prediction source {key}'
		)
	inputs = _mapping_value(metadata, 'inputs')
	if Path(str(inputs.get('token_prediction_dir'))).resolve(strict=False) != (
		token_dir.resolve(strict=False)
	):
		raise ValueError('V0 prediction token source directory mismatch')


def _validate_v1_prediction_source(
	metadata: Mapping[str, object], *, decoder_dir: Path
) -> None:
	source = _mapping_value(metadata, 'source_identity')
	_validate_recorded_identity(
		source.get('decoder_checkpoint'),
		decoder_dir / 'best.pt',
		label='V1 prediction decoder checkpoint',
	)


def _voxel_dataset_row(
	split_id: str, source: Mapping[str, object], root: Path
) -> dict[str, object]:
	metadata = _read_json(root / VOXEL_METADATA_NAME)
	return {
		'split_id': split_id,
		'png_label_inventory': str(source['png_label_inventory']),
		'voxel_dataset_root': str(root),
		'split_grid': _identity(root / GRID_NAME),
		'class_counts': _identity(root / COUNTS_NAME),
		'slice_split_manifest': _identity(root / MANIFEST_NAME),
		'metadata': _identity(root / VOXEL_METADATA_NAME),
		'train_voxel_count': int(
			_mapping_value(metadata, 'summary')['final_train_voxels']
		),
		'validation_voxel_count': int(
			_mapping_value(metadata, 'summary')['final_validation_voxels']
		),
		'reference_valid_tokens': _mapping_value(metadata, 'reference_valid_tokens'),
	}


def _v1_manifest_row(  # noqa: PLR0913
	job: VoxelSplitJob,
	voxel_row: Mapping[str, object],
	*,
	model: VoxelRobustnessModel,
	dataset_name: str,
	status: str,
	failure: str | None = None,
) -> dict[str, object]:
	decoder = job.output_root / 'decoder'
	row: dict[str, object] = {
		'split_id': job.split_id,
		'model_role': job.model_role,
		'model_tag': job.model_tag,
		'voxel_dataset_identity': _mapping_value(voxel_row, 'split_grid')['sha256'],
		'decoder_dir': str(decoder),
		'prediction_dir': str(job.output_root / 'voxel_predictions'),
		'evaluation_dir': str(job.output_root / 'evaluation'),
		'resume_path': str(decoder / 'latest.pt'),
		'source_valid_tokens': _identity(
			model.embeddings_dir / f'{dataset_name}.valid_tokens.npy'
		),
		'status': status,
	}
	if failure is not None:
		row['failure'] = failure
	for split in ('train', 'validation'):
		path = decoder / f'{split}_tile_manifest.json'
		if path.is_file():
			row[f'{split}_tile_manifest'] = _identity(path)
	if status == 'complete' and _complete_decoder(decoder):
		row['class_weights'] = _checkpoint_class_weights(decoder / 'latest.pt')
		evaluation_metadata = _read_json(
			job.output_root / 'evaluation' / EVALUATION_METADATA_JSON
		)
		row['decoder_architecture'] = validate_voxel_decoder_architecture_mapping(
			evaluation_metadata.get('decoder_architecture'),
			field_prefix='evaluation decoder_architecture',
		)
	return row


def _validate_paired_decoder_identity(
	rows: Sequence[Mapping[str, object]], *, split_id: str
) -> None:
	selected = [
		row
		for row in rows
		if row['split_id'] == split_id and row['status'] == 'complete'
	]
	if len(selected) < 2:
		return
	for key in (
		'voxel_dataset_identity',
		'source_valid_tokens',
		'class_weights',
		'train_tile_manifest',
		'validation_tile_manifest',
		'decoder_architecture',
	):
		values = [row.get(key) for row in selected]
		if key.endswith('_manifest') or key == 'source_valid_tokens':
			values = [
				value.get('sha256') if isinstance(value, Mapping) else None
				for value in values
			]
		elif key in {'class_weights', 'decoder_architecture'}:
			values = [
				json.dumps(value, sort_keys=True, allow_nan=False) for value in values
			]
		if len(set(values)) != 1:
			raise ValueError(f'paired decoder {key} mismatch for {split_id}')


def _validate_embedding_valid_masks(
	models: Sequence[VoxelRobustnessModel], *, dataset_name: str
) -> None:
	identities = []
	for model in models:
		path = model.embeddings_dir / f'{dataset_name}.valid_tokens.npy'
		if not path.is_file():
			raise FileNotFoundError(f'missing source embedding valid mask: {path}')
		identities.append(file_sha256(path))
	if len(set(identities)) != 1:
		raise ValueError('source embedding valid masks differ across M1/M2-A')


def _validate_job_sources(
	voxel_rows: Sequence[Mapping[str, object]],
	dataset_rows: Sequence[Mapping[str, object]],
	probe_rows: Sequence[Mapping[str, object]],
	models: Sequence[VoxelRobustnessModel],
) -> None:
	voxel_splits = {str(row['split_id']) for row in voxel_rows}
	if voxel_splits != {str(row['split_id']) for row in dataset_rows}:
		raise ValueError('voxel and token dataset manifests have different split IDs')
	datasets = _rows_by_key(dataset_rows)
	probes = _rows_by_key(probe_rows)
	if set(datasets) != set(probes):
		raise ValueError('token dataset and probe manifests have different job rows')
	expected_tags = {model.role: model.model_tag for model in models}
	paired_hashes: dict[str, set[str]] = defaultdict(set)
	for key, dataset in datasets.items():
		probe = probes[key]
		role = key[1]
		if (
			dataset['model_tag'] != expected_tags[role]
			or probe['model_tag'] != expected_tags[role]
		):
			raise ValueError(f'model identity mismatch for split/model row {key!r}')
		if dataset['paired_identity_hash'] != probe['paired_identity_hash']:
			raise ValueError(f'token dataset/probe identity mismatch for {key!r}')
		if Path(str(dataset['token_dataset_root'])).resolve(strict=False) != Path(
			str(probe['token_dataset_root'])
		).resolve(strict=False):
			raise ValueError(f'token dataset/probe root mismatch for {key!r}')
		paired_hashes[key[0]].add(str(dataset['paired_identity_hash']))
	for split_id, identities in paired_hashes.items():
		if len(identities) != 1:
			raise ValueError(f'paired identity hash mismatch for {split_id}')


def _validate_v0_source_artifacts(
	config: F3VoxelV0SplitSuiteConfig,
	voxel_rows: Sequence[Mapping[str, object]],
	dataset_rows: Sequence[Mapping[str, object]],
	probe_rows: Sequence[Mapping[str, object]],
) -> None:
	for path in (
		config.source_label_volume,
		config.source_label_segy,
		config.class_info,
		config.segy_geometry_json,
	):
		if not path.is_file():
			raise FileNotFoundError(f'missing V0 suite input: {path}')
	_validate_v0_embedding_artifacts(config, voxel_rows)
	probe_settings = _probe_manifest_settings(config.probe_run_manifest)
	for row in voxel_rows:
		_validate_voxel_manifest_row(row)
	probes = _rows_by_key(probe_rows)
	for dataset in dataset_rows:
		key = (str(dataset['split_id']), str(dataset['model_role']))
		root = Path(str(dataset['token_dataset_root']))
		train_path = Path(str(dataset.get('train_tokens', root / 'train_tokens.npz')))
		validation_path = Path(str(dataset['validation_tokens']))
		metadata_path = Path(
			str(dataset.get('metadata_json', root / 'token_dataset_metadata.json'))
		)
		for path in (train_path, validation_path, metadata_path):
			if not path.is_file():
				raise FileNotFoundError(f'missing split token dataset input: {path}')
		actual_pair = paired_token_identity_hash(
			load_token_dataset_npz(train_path),
			load_token_dataset_npz(validation_path),
		)
		if actual_pair != dataset['paired_identity_hash']:
			raise ValueError(f'token dataset paired identity mismatch for {key!r}')
		_validate_probe_artifact(dataset, probes[key], probe_settings=probe_settings)


def _validate_v0_embedding_artifacts(
	config: F3VoxelV0SplitSuiteConfig,
	voxel_rows: Sequence[Mapping[str, object]],
) -> None:
	canonical_valid_token_hashes = {
		str(_mapping_value(row, 'reference_valid_tokens')['sha256'])
		for row in voxel_rows
	}
	if len(canonical_valid_token_hashes) != 1:
		raise ValueError('voxel datasets have different canonical valid-token masks')
	canonical_valid_token_hash = next(iter(canonical_valid_token_hashes))
	for model in config.models:
		_validate_model_embedding_checkpoint(model, dataset_name=config.dataset['name'])
		artifacts = load_f3_embedding_artifacts(model.embeddings_dir)
		if len(artifacts) != 1:
			raise ValueError(
				f'V0 suite expects one embedding artifact for {model.model_tag}'
			)
		valid_tokens = (
			model.embeddings_dir / f'{config.dataset["name"]}.valid_tokens.npy'
		)
		if file_sha256(valid_tokens) != canonical_valid_token_hash:
			raise ValueError(
				'V0 source embedding valid mask does not match the voxel dataset '
				f'canonical valid-token identity for {model.model_tag}'
			)


def _probe_manifest_settings(path: Path) -> Mapping[str, object]:
	payload = _read_json(path)
	probe = _mapping_value(payload, 'probe')
	if probe.get('spec') != V0_PROBE_SPEC:
		raise ValueError(
			f'V0 probe manifest must use probe spec {V0_PROBE_SPEC!r}'
		)
	return probe


def _probe_rows(path: Path) -> tuple[Mapping[str, object], ...]:
	probe_settings = _probe_manifest_settings(path)
	rows = _paired_rows(
		path,
		artifact_type='f3_lithology_split_probe_run_manifest',
		required=(
			'split_id',
			'model_role',
			'model_tag',
			'token_dataset_root',
			'probe_output_dir',
			'paired_identity_hash',
		),
	)
	identity_keys = frozenset({'probe_spec', 'probe_joblib', 'scaler_joblib'})
	normalized = []
	for index, row in enumerate(rows):
		present = identity_keys.intersection(row)
		if present == identity_keys:
			normalized.append(row)
			continue
		if present:
			missing = sorted(identity_keys - present)
			raise ValueError(
				f'probe manifest row {index} has partial identity fields; '
				f'missing: {missing!r}'
			)
		normalized.append(
			_normalize_legacy_probe_row(row, probe_settings=probe_settings)
		)
	return tuple(normalized)


def _normalize_legacy_probe_row(  # noqa: C901
	row: Mapping[str, object], *, probe_settings: Mapping[str, object]
) -> Mapping[str, object]:
	root_value = row.get('probe_output_dir')
	if not isinstance(root_value, str) or not root_value:
		raise TypeError('legacy probe output directory must be a non-empty path')
	root = Path(root_value)
	probe_path = root / 'probe.joblib'
	scaler_path = root / 'scaler.joblib'
	config_path = root / 'probe_config_resolved.json'
	for artifact_path in (probe_path, scaler_path, config_path):
		if not artifact_path.is_file():
			raise FileNotFoundError(
				f'missing legacy split probe input: {artifact_path}'
			)
	resolved = _read_json(config_path)
	if resolved.get('artifact_type') != 'f3_lithology_probe':
		raise ValueError('legacy probe resolved config artifact_type mismatch')
	if _mapping_value(resolved, 'probe') != probe_settings:
		raise ValueError('legacy probe settings do not match the run manifest')
	model = _mapping_value(resolved, 'model')
	for row_key, resolved_key in (('model_tag', 'tag'), ('model_role', 'role')):
		if model.get(resolved_key) != row[row_key]:
			raise ValueError(f'legacy probe {row_key} identity mismatch')
	token_dataset = _mapping_value(resolved, 'token_dataset')
	for key, expected in (
		('input_dir', row['token_dataset_root']),
		('split_id', row['split_id']),
		('paired_identity_hash', row['paired_identity_hash']),
	):
		if token_dataset.get(key) != expected:
			raise ValueError(f'legacy probe token dataset {key} identity mismatch')
	outputs = _mapping_value(resolved, 'outputs')
	for key, expected in (
		('probe_joblib', probe_path),
		('scaler_joblib', scaler_path),
	):
		if outputs.get(key) != str(expected):
			raise ValueError(f'legacy probe output {key} path identity mismatch')
	return {
		**row,
		'probe_spec': probe_settings['spec'],
		'probe_joblib': _identity(probe_path),
		'scaler_joblib': _identity(scaler_path),
	}


def _validate_probe_artifact(  # noqa: C901, PLR0912
	dataset: Mapping[str, object],
	probe: Mapping[str, object],
	*,
	probe_settings: Mapping[str, object],
) -> None:
	root = Path(str(probe['probe_output_dir']))
	probe_path = root / 'probe.joblib'
	scaler_path = root / 'scaler.joblib'
	config_path = root / 'probe_config_resolved.json'
	for path in (probe_path, scaler_path, config_path):
		if not path.is_file():
			raise FileNotFoundError(f'missing split probe input: {path}')
	resolved = _read_json(config_path)
	if resolved.get('artifact_type') != 'f3_lithology_probe':
		raise ValueError('probe resolved config artifact_type mismatch')
	if probe.get('probe_spec') != V0_PROBE_SPEC:
		raise ValueError(f'V0 probe row must use probe spec {V0_PROBE_SPEC!r}')
	resolved_probe = _mapping_value(resolved, 'probe')
	if resolved_probe != probe_settings:
		raise ValueError('probe settings do not match the prior run manifest')
	if resolved_probe.get('spec') != V0_PROBE_SPEC:
		raise ValueError(f'V0 resolved probe must use spec {V0_PROBE_SPEC!r}')
	_validate_recorded_identity(
		probe.get('probe_joblib'), probe_path, label='probe manifest probe_joblib'
	)
	_validate_recorded_identity(
		probe.get('scaler_joblib'), scaler_path, label='probe manifest scaler_joblib'
	)
	model = _mapping_value(resolved, 'model')
	for key in ('model_tag', 'model_role'):
		expected = dataset['model_tag' if key == 'model_tag' else 'model_role']
		resolved_key = 'tag' if key == 'model_tag' else 'role'
		if model.get(resolved_key) != expected:
			raise ValueError(f'probe {key} identity mismatch')
	token_dataset = _mapping_value(resolved, 'token_dataset')
	expected_token_values = {
		'input_dir': str(dataset['token_dataset_root']),
		'split_id': str(dataset['split_id']),
		'paired_identity_hash': str(dataset['paired_identity_hash']),
	}
	for key, expected in expected_token_values.items():
		if token_dataset.get(key) != expected:
			raise ValueError(f'probe token dataset {key} identity mismatch')
	inputs = _mapping_value(resolved, 'inputs')
	root_dataset = Path(str(dataset['token_dataset_root']))
	for key, expected in (
		(
			'train_tokens',
			dataset.get('train_tokens', root_dataset / 'train_tokens.npz'),
		),
		('validation_tokens', dataset['validation_tokens']),
	):
		if inputs.get(key) != str(expected):
			raise ValueError(f'probe input {key} identity mismatch')
	outputs = _mapping_value(resolved, 'outputs')
	for key, expected in (
		('probe_joblib', probe_path),
		('scaler_joblib', scaler_path),
	):
		if outputs.get(key) != str(expected):
			raise ValueError(f'probe output {key} identity mismatch')


def _validate_split_inventory_identity(
	voxel_manifest_path: Path, token_manifest_path: Path
) -> None:
	voxel_manifest = _read_json(voxel_manifest_path)
	token_manifest = _read_json(token_manifest_path)
	voxel_source = _mapping_value(voxel_manifest, 'source_split_inventory_manifest')
	voxel_source_path = voxel_source.get('path')
	if not isinstance(voxel_source_path, str) or not voxel_source_path:
		raise TypeError('voxel source split inventory identity requires a path')
	token_suite = _mapping_value(token_manifest, 'suite')
	token_source_path = token_suite.get('split_inventory_manifest')
	if not isinstance(token_source_path, str) or not token_source_path:
		raise TypeError('token dataset suite requires split_inventory_manifest')
	if Path(voxel_source_path).resolve(strict=False) != Path(token_source_path).resolve(
		strict=False
	):
		raise ValueError('voxel and token datasets use different split inventories')
	_validate_recorded_identity(
		voxel_source,
		Path(voxel_source_path),
		label='voxel source split inventory',
	)


def _evaluation_metrics(root: Path) -> dict[str, float | None]:
	metrics = _read_json(root / METRICS_JSON)
	boundary = _read_json(root / BOUNDARY_METRICS_JSON)
	regions = _read_csv(root / BOUNDARY_REGION_METRICS_CSV)
	values: dict[str, float | None] = {
		'macro_f1': _number(metrics.get('macro_f1')),
		'mean_iou': _number(metrics.get('mean_iou')),
		'balanced_accuracy': _number(metrics.get('balanced_accuracy')),
	}
	for class_id in (3, 5):
		values[f'class_{class_id}_f1'] = _nested_number(
			metrics, 'per_class_f1', str(class_id)
		)
		values[f'class_{class_id}_iou'] = _nested_number(
			metrics, 'per_class_iou', str(class_id)
		)
	for tolerance in (2, 4):
		values[f'vertical_boundary_f1_tolerance_{tolerance}'] = _number(
			boundary.get(f'vertical_boundary_f1_at_{tolerance}')
		)
		for class_id in (3, 5):
			values[f'class_{class_id}_boundary_recall_tolerance_{tolerance}'] = _number(
				boundary.get(
					f'vertical_boundary_class_{class_id}_recall_at_{tolerance}'
				)
			)
	for radius in (2, 4):
		row = next(
			(
				item
				for item in regions
				if item.get('region') == 'boundary'
				and item.get('radius') == str(radius)
			),
			None,
		)
		if row is None:
			raise ValueError(f'missing boundary-region row for radius {radius}: {root}')
		values[f'boundary_region_macro_f1_radius_{radius}'] = _number(
			row.get('macro_f1')
		)
		values[f'boundary_region_mean_iou_radius_{radius}'] = _number(
			row.get('mean_iou')
		)
	values['boundary_position_mae'] = _number(
		boundary.get('vertical_boundary_position_mae_at_4')
	)
	return values


def _paired_metric_rows(
	rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	by_key = {(str(row['split_id']), str(row['model_role'])): row for row in rows}
	splits = sorted({key[0] for key in by_key})
	result = []
	for split_id in splits:
		baseline = by_key[(split_id, 'baseline')]
		candidate = by_key[(split_id, 'candidate')]
		for metric in SUMMARY_METRICS:
			for head in ('v0', 'v1'):
				base = _number(baseline.get(f'{head}_{metric}'))
				cand = _number(candidate.get(f'{head}_{metric}'))
				delta = _delta(base, cand)
				result.append(
					{
						'split_id': split_id,
						'comparison': f'{head}_candidate_minus_baseline',
						'metric': metric,
						'baseline': base,
						'candidate': cand,
						'delta': delta,
						'win': _is_win(metric, delta),
					}
				)
			for role, source in (('baseline', baseline), ('candidate', candidate)):
				v0 = _number(source.get(f'v0_{metric}'))
				v1 = _number(source.get(f'v1_{metric}'))
				result.append(
					{
						'split_id': split_id,
						'comparison': f'{role}_v1_minus_v0',
						'metric': metric,
						'baseline': v0,
						'candidate': v1,
						'delta': _delta(v0, v1),
						'win': None,
					}
				)
	return result


def _aggregate_metric_rows(
	rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	groups: dict[tuple[str, str], list[float | None]] = defaultdict(list)
	for row in rows:
		value = row.get('delta')
		groups[(str(row['comparison']), str(row['metric']))].append(
			float(value)
			if isinstance(value, int | float) and not isinstance(value, bool)
			else None
		)
	result = []
	for (comparison, metric), split_values in sorted(groups.items()):
		values = [value for value in split_values if value is not None]
		wins = sum(_is_win(metric, value) is True for value in values)
		result.append(
			{
				'comparison': comparison,
				'metric': metric,
				'split_count': len(split_values),
				'mean_delta': statistics.fmean(values) if values else None,
				'median_delta': statistics.median(values) if values else None,
				'win_count': wins,
				'win_rate': wins / len(split_values),
			}
		)
	primary = [
		row
		for row in rows
		if row['comparison'] == 'v1_candidate_minus_baseline'
		and row['metric'] in PRIMARY_METRICS
	]
	by_split: dict[str, dict[str, float]] = defaultdict(dict)
	for row in primary:
		split_id = str(row['split_id'])
		by_split[split_id]
		if isinstance(row['delta'], float | int) and not isinstance(row['delta'], bool):
			by_split[split_id][str(row['metric'])] = float(row['delta'])
	simultaneous = sum(
		all(values.get(metric, 0.0) > 0 for metric in PRIMARY_METRICS)
		for values in by_split.values()
	)
	result.append(
		{
			'comparison': 'v1_candidate_minus_baseline',
			'metric': 'primary_metrics_simultaneous',
			'split_count': len(by_split),
			'mean_delta': None,
			'median_delta': None,
			'win_count': simultaneous,
			'win_rate': simultaneous / len(by_split),
		}
	)
	return result


def _provisional_status(
	rows: Sequence[Mapping[str, object]], aggregates: Sequence[Mapping[str, object]]
) -> tuple[str, list[str]]:
	lookup = {(str(row['comparison']), str(row['metric'])): row for row in aggregates}
	primary = lookup[('v1_candidate_minus_baseline', 'primary_metrics_simultaneous')]
	macro = lookup[('v1_candidate_minus_baseline', 'macro_f1')]
	iou = lookup[('v1_candidate_minus_baseline', 'mean_iou')]
	boundaries = [
		lookup[('v1_candidate_minus_baseline', f'vertical_boundary_f1_tolerance_{tol}')]
		for tol in (2, 4)
	]
	class_majority = False
	for class_id in (3, 5):
		for metric in ('f1', 'iou'):
			row = lookup[('v1_candidate_minus_baseline', f'class_{class_id}_{metric}')]
			if int(row['win_count']) > int(row['split_count']) / 2:
				class_majority = True
	positive = (
		int(primary['win_count']) > int(primary['split_count']) / 2
		and _numeric_comparison(macro.get('mean_delta'), lower_bound=0, strict=True)
		and _numeric_comparison(iou.get('mean_delta'), lower_bound=0, strict=True)
		and all(
			_numeric_comparison(row.get('mean_delta'), lower_bound=0, strict=False)
			for row in boundaries
		)
		and class_majority
	)
	reasons = [
		f'simultaneous primary wins={primary["win_count"]}/{primary["split_count"]}',
		f'mean macro_f1 delta={macro["mean_delta"]}',
		f'mean mean_iou delta={iou["mean_delta"]}',
		f'boundary tolerance mean deltas={[row["mean_delta"] for row in boundaries]}',
		f'class 3/5 majority improvement={class_majority}',
	]
	if positive:
		return 'positive', reasons
	primary_rows = [
		row for row in rows if row['comparison'] == 'v1_candidate_minus_baseline'
	]
	if (
		primary_rows
		and all(
			row['delta'] is not None
			and _is_loss(str(row['metric']), float(row['delta']))
			for row in primary_rows
		)
	):
		return 'negative', reasons
	return 'hold', reasons


def _render_robustness_markdown(
	*, decoder_architecture: Mapping[str, object], status: str, split_count: int
) -> str:
	return '\n'.join(
		(
			'# F3 voxel split robustness summary',
			'',
			'## Decoder identity',
			'',
			f'- spec: `{decoder_architecture["spec"]}`',
			f'- upsample mode: `{decoder_architecture["upsample_mode"]}`',
			f'- normalization: `{decoder_architecture["normalization"]}`',
			'',
			'## Result',
			'',
			f'- split count: `{split_count}`',
			f'- provisional status: **{status}**',
			'',
			'Paired metrics and win rates use split, not voxel, as the '
			'statistical unit.',
			'',
		)
	)


def _numeric_comparison(value: object, *, lower_bound: float, strict: bool) -> bool:
	if not isinstance(value, int | float) or isinstance(value, bool):
		return False
	return value > lower_bound if strict else value >= lower_bound


def _delta(baseline: float | None, candidate: float | None) -> float | None:
	if baseline is None or candidate is None:
		return None
	return candidate - baseline


def _is_win(metric: str, delta: float | None) -> bool | None:
	if delta is None:
		return None
	return delta < 0 if metric in LOWER_IS_BETTER else delta > 0


def _is_loss(metric: str, delta: float) -> bool:
	return delta > 0 if metric in LOWER_IS_BETTER else delta < 0


def _complete_voxel_dataset(root: Path) -> bool:
	return all(
		(root / name).is_file()
		for name in (
			GRID_NAME,
			VOXEL_METADATA_NAME,
			COUNTS_NAME,
			MANIFEST_NAME,
			SUMMARY_NAME,
		)
	)


def _validate_existing_voxel_dataset(  # noqa: C901, PLR0912
	config: F3LithologyVoxelDatasetConfig,
) -> None:
	root = config.output_dir
	inspection = inspect_f3_lithology_voxel_dataset(config)
	metadata = _read_json(root / VOXEL_METADATA_NAME)
	if metadata.get('artifact_type') != 'f3_lithology_voxel_supervision':
		raise ValueError('existing voxel dataset artifact_type mismatch')
	if metadata.get('schema_version') != 1:
		raise ValueError('existing voxel dataset schema_version mismatch')
	if metadata.get('dataset') != dict(config.dataset):
		raise ValueError('existing voxel dataset dataset identity mismatch')
	for key, path in (
		('reference_embedding', config.reference_metadata_json),
		('reference_valid_tokens', config.reference_valid_tokens),
		('label_volume', config.source_label_volume),
		('inventory', config.png_label_inventory),
	):
		_validate_recorded_identity(metadata.get(key), path, label=f'voxel {key}')
	labels = _mapping_value(metadata, 'labels')
	for key, path in (
		('source_label_segy', config.source_label_segy),
		('class_info', config.class_info),
	):
		if labels.get(key) != str(path):
			raise ValueError(f'existing voxel dataset labels.{key} identity mismatch')
	if metadata.get('geometry') != read_f3_line_geometry(
		config.segy_geometry_json
	).to_dict():
		raise ValueError('existing voxel dataset geometry identity mismatch')
	if metadata.get('classes') != [
		item.to_dict() for item in read_f3_lithology_class_info(config.class_info)
	]:
		raise ValueError('existing voxel dataset class identity mismatch')
	if metadata.get('ignore_z_border_samples') != config.ignore_z_border_samples:
		raise ValueError('existing voxel dataset border policy identity mismatch')
	grid = np.load(root / GRID_NAME, mmap_mode='r', allow_pickle=False)
	if not np.array_equal(grid, inspection.split.split_grid):
		raise ValueError(
			'existing voxel dataset split grid does not match the source inventory'
		)
	if _mapping_value(metadata, 'summary') != asdict(inspection.split.summary):
		raise ValueError(
			'existing voxel dataset summary does not match the source inventory'
		)
	outputs = _mapping_value(metadata, 'outputs')
	expected_outputs = {
		'supervision_split_grid': root / GRID_NAME,
		'metadata_json': root / VOXEL_METADATA_NAME,
		'class_counts_csv': root / COUNTS_NAME,
		'split_manifest_json': root / MANIFEST_NAME,
		'summary_markdown': root / SUMMARY_NAME,
	}
	for key, path in expected_outputs.items():
		if outputs.get(key) != str(path):
			raise ValueError(f'existing voxel dataset output {key} identity mismatch')


def _complete_evaluation(root: Path) -> bool:
	if not all((root / name).is_file() for name in EVALUATION_OUTPUT_FILES):
		return False
	try:
		metadata = _read_json(root / EVALUATION_METADATA_JSON)
		metrics = _read_json(root / METRICS_JSON)
		_read_json(root / BOUNDARY_METRICS_JSON)
		if metadata.get('artifact_type') != 'f3_lithology_voxel_evaluation':
			return False
		for metric in PRIMARY_METRICS:
			_require_finite_metric(metrics.get(metric), label=f'evaluation {metric}')
		outputs = _mapping_value(metadata, 'outputs')
		for name in EVALUATION_OUTPUT_FILES:
			_validate_recorded_identity(
				outputs.get(name), root / name, label=f'evaluation output {name}'
			)
		summary = _mapping_value(metadata, 'summary')
		voxel_count = summary.get('unique_validation_voxel_count')
		if (
			not isinstance(voxel_count, int)
			or isinstance(voxel_count, bool)
			or voxel_count <= 0
			or metrics.get('evaluation_voxel_count') != voxel_count
		):
			return False
		if metrics.get('aggregation_unit') != 'unique_validation_voxel':
			return False
	except (
		OSError,
		TypeError,
		ValueError,
		json.JSONDecodeError,
	):
		return False
	return True


def _complete_token_prediction(root: Path) -> bool:
	return all(
		(root / name).is_file()
		for name in (
			'f3_token_predictions.npy',
			'f3_token_probabilities.npy',
			'f3_valid_token_grid.npy',
			'prediction_metadata.json',
			'validation_slice_metrics.csv',
		)
	)


def _complete_prediction(root: Path) -> bool:
	return all(
		path.is_file()
		for path in (
			root / PREDICTIONS_NAME,
			root / CONFIDENCE_NAME,
			root / VALID_MASK_NAME,
			root / METADATA_NAME,
		)
	)


def _complete_decoder(root: Path) -> bool:
	latest = root / 'latest.pt'
	required = (
		latest,
		root / 'best.pt',
		root / 'resolved_config.json',
		root / 'run_metadata.json',
		root / 'history.csv',
		root / 'train_tile_manifest.json',
		root / 'validation_tile_manifest.json',
	)
	if not all(path.is_file() for path in required):
		return False
	import torch  # noqa: PLC0415

	payload = torch.load(latest, map_location='cpu', weights_only=False)
	return (
		isinstance(payload, Mapping) and payload.get('checkpoint_kind') == 'completed'
	)


def _checkpoint_class_weights(path: Path) -> list[float]:
	import torch  # noqa: PLC0415

	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError(f'decoder checkpoint must be a mapping: {path}')
	weights = payload.get('class_weights')
	if not isinstance(weights, torch.Tensor | Sequence) or isinstance(
		weights, str | bytes
	):
		raise TypeError(f'decoder checkpoint class_weights must be a vector: {path}')
	values = torch.as_tensor(weights, dtype=torch.float32).cpu()
	if values.ndim != 1:
		raise TypeError(f'decoder checkpoint class_weights must be a vector: {path}')
	return [float(value) for value in values.tolist()]


def _v0_job_root(root: Path, row: Mapping[str, object]) -> Path:
	return root / 'v0' / f'split={row["split_id"]}' / f'model={row["model_tag"]}'


def _resume_path(row: Mapping[str, object] | None, decoder_dir: Path) -> Path | None:
	if row is not None:
		path = Path(str(row.get('resume_path', decoder_dir / 'latest.pt')))
		if path.is_file():
			return path
	path = decoder_dir / 'latest.pt'
	return path if path.is_file() else None


def _prior_run_rows(
	path: Path, *, artifact_type: str
) -> dict[tuple[str, str], Mapping[str, object]]:
	if not path.is_file():
		return {}
	return _rows_by_key(
		_read_manifest_rows(
			path,
			artifact_type=artifact_type,
			required=('split_id', 'model_role', 'model_tag', 'status'),
		)
	)


def _complete_paired_run_rows(  # noqa: C901, PLR0912
	path: Path, artifact_type: str
) -> dict[tuple[str, str], Mapping[str, object]]:
	expected_prediction_kind = {
		V0_RUN_MANIFEST_TYPE: 'token_projection_nearest',
		V1_RUN_MANIFEST_TYPE: 'frozen_embedding_decoder',
	}[artifact_type]
	required = [
		'split_id',
		'model_role',
		'model_tag',
		'voxel_dataset_identity',
		'prediction_dir',
		'evaluation_dir',
		'status',
	]
	if artifact_type == V0_RUN_MANIFEST_TYPE:
		required.append('token_prediction_dir')
	if artifact_type == V1_RUN_MANIFEST_TYPE:
		required.extend(
			(
				'decoder_dir',
				'source_valid_tokens',
				'class_weights',
				'train_tile_manifest',
				'validation_tile_manifest',
				'decoder_architecture',
			)
		)
	rows = _paired_rows(
		path,
		artifact_type=artifact_type,
		required=required,
	)
	for row in rows:
		evaluation_dir = Path(str(row['evaluation_dir']))
		if row['status'] != 'complete' or not _complete_evaluation(evaluation_dir):
			raise ValueError(
				f'incomplete run row: {row["split_id"]}/{row["model_role"]}'
			)
		evaluation_metadata = _read_json(
			evaluation_dir / EVALUATION_METADATA_JSON
		)
		if evaluation_metadata.get('model_tag') != row['model_tag']:
			raise ValueError(
				'existing evaluation model identity does not match run row: '
				f'{row["split_id"]}/{row["model_role"]}'
			)
		if evaluation_metadata.get('prediction_kind') != expected_prediction_kind:
			raise ValueError(
				'existing evaluation prediction kind does not match run stage: '
				f'{row["split_id"]}/{row["model_role"]}'
			)
		inputs = _mapping_value(evaluation_metadata, 'inputs')
		grid = _mapping_value(inputs, 'voxel_split_grid')
		_validated_recorded_identity(grid, label='evaluation voxel_split_grid')
		if grid.get('sha256') != row['voxel_dataset_identity']:
			raise ValueError(
				'existing evaluation split-grid identity does not match run row: '
				f'{row["split_id"]}/{row["model_role"]}'
			)
		prediction_metadata_path = _validated_recorded_identity(
			inputs.get('prediction_metadata'),
			label='evaluation prediction_metadata',
		)
		prediction_dir = Path(str(row['prediction_dir']))
		if prediction_metadata_path.resolve(strict=False) != (
			prediction_dir / 'prediction_metadata.json'
		).resolve(strict=False):
			raise ValueError(
				'existing evaluation prediction path does not match run row: '
				f'{row["split_id"]}/{row["model_role"]}'
			)
		prediction_metadata = _read_json(prediction_metadata_path)
		for key, expected in (
			('model_tag', row['model_tag']),
			('prediction_kind', expected_prediction_kind),
		):
			if prediction_metadata.get(key) != expected:
				raise ValueError(
					f'prediction metadata {key} does not match run row: '
					f'{row["split_id"]}/{row["model_role"]}'
				)
		if artifact_type == V0_RUN_MANIFEST_TYPE:
			_validate_v0_prediction_source(
				prediction_metadata,
				token_dir=Path(str(row['token_prediction_dir'])),
			)
		else:
			_validate_v1_prediction_source(
				prediction_metadata, decoder_dir=Path(str(row['decoder_dir']))
			)
			_validate_v1_summary_row(row, prediction_metadata)
			evaluation_architecture = validate_voxel_decoder_architecture_mapping(
				evaluation_metadata.get('decoder_architecture'),
				field_prefix='evaluation decoder_architecture',
			)
			prediction_architecture = validate_voxel_decoder_architecture_mapping(
				prediction_metadata.get('decoder_architecture'),
				field_prefix='prediction decoder_architecture',
			)
			manifest_architecture = validate_voxel_decoder_architecture_mapping(
				row.get('decoder_architecture'),
				field_prefix='run decoder_architecture',
			)
			if not (
				evaluation_architecture
				== prediction_architecture
				== manifest_architecture
			):
				raise ValueError('V1 run decoder architecture identity mismatch')
	result = _rows_by_key(rows)
	if artifact_type == V1_RUN_MANIFEST_TYPE:
		for split_id in {key[0] for key in result}:
			_validate_paired_decoder_identity(tuple(result.values()), split_id=split_id)
		architectures = [row.get('decoder_architecture') for row in result.values()]
		if any(value != architectures[0] for value in architectures[1:]):
			raise ValueError('V1 decoder architecture mismatch across splits')
	return result


def _validate_v1_summary_row(
	row: Mapping[str, object], prediction_metadata: Mapping[str, object]
) -> None:
	valid_tokens = _validated_recorded_identity(
		row.get('source_valid_tokens'), label='run source_valid_tokens'
	)
	train_tiles = _validated_recorded_identity(
		row.get('train_tile_manifest'), label='run train_tile_manifest'
	)
	validation_tiles = _validated_recorded_identity(
		row.get('validation_tile_manifest'), label='run validation_tile_manifest'
	)
	source = _mapping_value(prediction_metadata, 'source_identity')
	artifacts = _mapping_value(source, 'artifact_identities')
	manifests = _mapping_value(source, 'tile_manifests')
	for recorded, expected_path, label in (
		(artifacts.get('valid_tokens'), valid_tokens, 'prediction valid_tokens'),
		(manifests.get('train'), train_tiles, 'prediction train tile manifest'),
		(
			manifests.get('validation'),
			validation_tiles,
			'prediction validation tile manifest',
		),
	):
		path = _validated_recorded_identity(recorded, label=label)
		if file_sha256(path) != file_sha256(expected_path):
			raise ValueError(f'{label} identity does not match run row')
	checkpoint = _validated_recorded_identity(
		source.get('decoder_checkpoint'), label='prediction decoder_checkpoint'
	)
	if _checkpoint_class_weights(checkpoint) != _class_weight_vector(
		row.get('class_weights')
	):
		raise ValueError('prediction decoder class_weights do not match run row')


def _validated_recorded_identity(value: object, *, label: str) -> Path:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} identity must be a mapping')
	path_value = value.get('path')
	if not isinstance(path_value, str) or not path_value:
		raise TypeError(f'{label} identity path must be a non-empty string')
	path = Path(path_value)
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	if value.get('sha256') != file_sha256(path):
		raise ValueError(f'{label} hash identity mismatch')
	return path


def _class_weight_vector(value: object) -> list[float]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError('run class_weights must be a vector')
	if any(
		isinstance(item, bool) or not isinstance(item, int | float) for item in value
	):
		raise TypeError('run class_weights must be a numeric vector')
	return [float(item) for item in value]


def _voxel_rows(path: Path) -> tuple[Mapping[str, object], ...]:
	rows = _read_manifest_rows(
		path,
		artifact_type=SPLIT_DATASET_MANIFEST_TYPE,
		required=(
			'split_id',
			'png_label_inventory',
			'voxel_dataset_root',
			'split_grid',
			'class_counts',
			'slice_split_manifest',
			'metadata',
			'reference_valid_tokens',
		),
	)
	for row in rows:
		_validate_voxel_manifest_row(row)
	return rows


def _validate_voxel_manifest_row(row: Mapping[str, object]) -> None:
	root = Path(str(row['voxel_dataset_root']))
	for key, name in (
		('split_grid', GRID_NAME),
		('class_counts', COUNTS_NAME),
		('slice_split_manifest', MANIFEST_NAME),
		('metadata', VOXEL_METADATA_NAME),
	):
		path = _validated_recorded_identity(row.get(key), label=f'voxel row {key}')
		if path.resolve(strict=False) != (root / name).resolve(strict=False):
			raise ValueError(f'voxel row {key} path does not match dataset root')
	_validated_recorded_identity(
		row.get('reference_valid_tokens'), label='voxel row reference_valid_tokens'
	)


def _paired_rows(
	path: Path, *, artifact_type: str, required: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
	rows = _read_manifest_rows(path, artifact_type=artifact_type, required=required)
	roles: dict[str, set[str]] = defaultdict(set)
	for row in rows:
		roles[str(row['split_id'])].add(str(row['model_role']))
	for split_id, found in roles.items():
		if found != {'baseline', 'candidate'}:
			raise ValueError(
				f'split {split_id} must have exactly baseline and candidate rows'
			)
	return rows


def _read_manifest_rows(
	path: Path, *, artifact_type: str, required: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
	payload = _read_json(path)
	if payload.get('artifact_type') != artifact_type:
		raise ValueError(f'invalid manifest artifact_type: {path}')
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes) or not rows:
		raise ValueError(f'manifest rows must be a non-empty list: {path}')
	result = []
	seen: set[tuple[str, str | None]] = set()
	for index, row in enumerate(rows):
		if not isinstance(row, Mapping):
			raise TypeError(f'manifest row {index} must be a mapping')
		missing = [key for key in required if key not in row]
		if missing:
			raise ValueError(f'manifest row {index} missing key(s): {missing!r}')
		key = (
			str(row['split_id']),
			str(row.get('model_role')) if 'model_role' in row else None,
		)
		if key in seen:
			raise ValueError(f'duplicate split/run row rejected: {key!r}')
		seen.add(key)
		_split_id(row)
		result.append(row)
	return tuple(result)


def _manifest_rows(path: Path) -> tuple[Mapping[str, object], ...]:
	payload = _read_json(path)
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError(f'manifest rows must be a list: {path}')
	return tuple(cast('Mapping[str, object]', row) for row in rows)


def _split_id(row: Mapping[str, object]) -> str:
	value = row.get('split_id')
	if not isinstance(value, str) or value not in SOURCE_SPLIT_IDS:
		raise ValueError(
			'existing split ID must be one of split_000 through split_005; '
			f'got {value!r}'
		)
	return value


def _rows_by_key(
	rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
	result = {}
	for row in rows:
		key = (str(row['split_id']), str(row['model_role']))
		if key in result:
			raise ValueError(f'duplicate split/run row rejected: {key!r}')
		result[key] = row
	return result


def _write_run_manifest(
	path: Path, artifact_type: str, rows: Sequence[Mapping[str, object]]
) -> None:
	_write_json(
		path,
		{
			'artifact_type': artifact_type,
			'schema_version': SCHEMA_VERSION,
			'rows': list(rows),
		},
	)


def _identity(path: Path) -> dict[str, str]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {'path': str(path), 'sha256': file_sha256(path)}


def _mapping_value(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return value


def _nested_number(parent: Mapping[str, object], key: str, child: str) -> float | None:
	return _number(_mapping_value(parent, key).get(child))


def _number(value: object) -> float | None:
	if value is None or value == '':
		return None
	if isinstance(value, bool) or not isinstance(value, str | int | float):
		raise TypeError(f'metric must be numeric or null; got {value!r}')
	return float(value)


def _require_finite_metric(value: object, *, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{label} must be numeric')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label} must be finite')
	return result


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON document must contain an object: {path}')
	return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(encoding='utf-8', newline='') as file_obj:
		return list(csv.DictReader(file_obj))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		raise ValueError(f'cannot write empty CSV: {path}')
	fieldnames = list(rows[0])
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


__all__ = [
	'SUMMARY_METRICS',
	'VoxelSplitJob',
	'VoxelSplitRobustnessInspection',
	'VoxelSplitRobustnessSummaryResult',
	'VoxelSplitSuiteResult',
	'build_f3_lithology_voxel_split_datasets',
	'inspect_f3_lithology_voxel_split_robustness',
	'run_f3_lithology_voxel_decoder_split_suite',
	'run_f3_lithology_voxel_v0_split_suite',
	'summarize_f3_lithology_voxel_split_robustness',
	'voxel_decoder_split_jobs',
	'voxel_split_dataset_jobs',
	'voxel_v0_split_jobs',
]
