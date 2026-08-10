"""Generic one-model runner for the F3 section-layout voxel benchmark."""
# ruff: noqa: CPY001

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	F3LithologyVoxelDecoderConfig,
)
from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	F3LithologyVoxelEvaluationConfig,
)
from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	F3LithologyVoxelInferenceConfig,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	DECODER_SEED,
	LAYOUT_IDS,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_roster import (
	F3SectionLayoutModel,
	f3_lithology_voxel_section_layout_model_roster_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.voxel_decoder_inference import (
	inspect_f3_lithology_voxel_inference,
	predict_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	BOUNDARY_METRICS_JSON,
	BOUNDARY_REGION_METRICS_CSV,
	EVALUATION_METADATA_JSON,
	METRICS_JSON,
	evaluate_f3_lithology_voxels,
	inspect_f3_lithology_voxel_evaluation,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	quarantine_voxel_label_budget_output,
	sampling_sequence_sha256,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout import (
	validate_f3_lithology_voxel_section_layout_condition,
	validate_f3_lithology_voxel_section_layout_manifest,
)
from seis_ssl_cluster.f3.lithology.voxel_tiles import read_voxel_tile_manifest
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)
from seis_ssl_cluster.training.voxel_decoder.losses import (
	balanced_class_weights_from_counts,
)
from seis_ssl_cluster.training.voxel_decoder.runner import (
	run_f3_lithology_voxel_decoder,
	validate_f3_lithology_voxel_decoder_resume,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_benchmark import (
		F3SectionLayoutBenchmarkConfig,
	)

RUN_MANIFEST_NAME = 'section_layout_run_manifest.json'
RUN_MANIFEST_TYPE = 'f3_lithology_voxel_section_layout_run_manifest'
RUN_SCHEMA_VERSION = 1
JOB_STATES = ('NEW', 'RESUME_LATEST', 'REUSE_COMPLETED', 'INVALID_OR_PARTIAL')
EXPECTED_EMBEDDING_SHAPE = (76, 113, 32, 384)
EXPECTED_VALID_TOKEN_SHAPE = EXPECTED_EMBEDDING_SHAPE[:3]


@dataclass(frozen=True)
class F3SectionLayoutJob:
	"""One shared dataset condition evaluated by one roster model."""

	model: F3SectionLayoutModel
	layout_id: str
	data_size: str
	dataset_root: Path
	output_root: Path
	dataset_row: Mapping[str, object]
	embedding_identity: Mapping[str, Mapping[str, object]]

	@property
	def decoder_dir(self) -> Path:
		"""Return this job's decoder directory."""
		return self.output_root / 'decoder'

	@property
	def prediction_dir(self) -> Path:
		"""Return this job's prediction directory."""
		return self.output_root / 'prediction'

	@property
	def evaluation_dir(self) -> Path:
		"""Return this job's evaluation directory."""
		return self.output_root / 'evaluation'

	@property
	def generated_configs_dir(self) -> Path:
		"""Return this job's generated-config snapshot directory."""
		return self.output_root / 'generated_configs'


@dataclass(frozen=True)
class F3SectionLayoutJobPlan:
	"""Live job state and its safe next action."""

	job: F3SectionLayoutJob
	state: str
	reason: str | None = None


@dataclass(frozen=True)
class F3SectionLayoutSuiteInspection:
	"""Fully preflighted one-model job plan."""

	model: F3SectionLayoutModel
	jobs: tuple[F3SectionLayoutJob, ...]
	plans: tuple[F3SectionLayoutJobPlan, ...]
	dataset_manifest_identity: Mapping[str, object]
	embedding_identity: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class F3SectionLayoutSuiteResult:
	"""Atomic model manifest and the retained row history."""

	manifest_json: Path
	rows: tuple[Mapping[str, object], ...]
	quarantines: tuple[Path, ...]


def inspect_f3_lithology_voxel_section_layout_suite(
	config: F3SectionLayoutBenchmarkConfig,
	*,
	model_id: str,
	layout_id: str | None = None,
	data_size: str | None = None,
	smoke_only: bool = False,
) -> F3SectionLayoutSuiteInspection:
	"""Validate the roster, embedding, and exact 15-dataset matrix before planning."""
	if not isinstance(model_id, str) or not model_id:
		raise ValueError('exactly one non-empty model_id is required')
	if smoke_only and (layout_id is None or data_size is None):
		raise ValueError('smoke inspection requires explicit layout_id and data_size')
	roster_mapping = load_config(config.model_roster)
	roster = f3_lithology_voxel_section_layout_model_roster_from_mapping(roster_mapping)
	if roster.artifact_root != config.artifact_root:
		raise ValueError('model roster artifact_root differs from runner config')
	try:
		model = roster.model_by_id[model_id]
	except KeyError as error:
		raise ValueError(f'unknown model_id: {model_id!r}') from error
	embedding_identity = _validate_embedding(config, model)
	manifest = validate_f3_lithology_voxel_section_layout_manifest(
		config.dataset_manifest
	)
	rows = _dataset_rows(manifest)
	_validate_dataset_valid_tokens(manifest, embedding_identity)
	root = config.smoke_root if smoke_only else config.benchmark_root
	all_jobs = tuple(
		F3SectionLayoutJob(
			model=model,
			layout_id=cast('str', row['layout_id']),
			data_size=cast('str', row['data_size']),
			dataset_root=Path(cast('str', row['voxel_dataset_root'])),
			output_root=(
				root
				/ 'runs'
				/ f'model={model.model_id}'
				/ f'layout={row["layout_id"]}'
				/ f'size={row["data_size"]}'
			),
			dataset_row=row,
			embedding_identity=embedding_identity,
		)
		for row in rows
	)
	jobs = tuple(
		job
		for job in all_jobs
		if (layout_id is None or job.layout_id == layout_id)
		and (data_size is None or job.data_size == data_size)
	)
	if not jobs:
		raise ValueError('job filters selected no section-layout conditions')
	plans = tuple(_classify_job(config, job) for job in jobs)
	return F3SectionLayoutSuiteInspection(
		model=model,
		jobs=jobs,
		plans=plans,
		dataset_manifest_identity=_identity(config.dataset_manifest),
		embedding_identity=embedding_identity,
	)


def run_f3_lithology_voxel_section_layout_suite(  # noqa: C901, PLR0912, PLR0913
	config: F3SectionLayoutBenchmarkConfig,
	*,
	model_id: str,
	layout_id: str | None = None,
	data_size: str | None = None,
	only_missing: bool = False,
	resume: bool = False,
	quarantine_invalid: bool = False,
	smoke_only: bool = False,
	device: str = 'auto',
) -> F3SectionLayoutSuiteResult:
	"""Run one roster model without ever expanding implicitly to other models."""
	if quarantine_invalid and not only_missing:
		raise ValueError('--quarantine-invalid requires --only-missing')
	if smoke_only and (layout_id is None or data_size is None):
		raise ValueError('--smoke-only requires an explicit layout_id and data_size')
	inspection = inspect_f3_lithology_voxel_section_layout_suite(
		config,
		model_id=model_id,
		layout_id=layout_id,
		data_size=data_size,
		smoke_only=smoke_only,
	)
	if smoke_only and len(inspection.jobs) != 1:
		raise ValueError('smoke mode must select exactly one condition')
	_validate_execution_policy(
		inspection.plans,
		only_missing=only_missing,
		resume=resume,
		quarantine_invalid=quarantine_invalid,
	)
	model_root = inspection.jobs[0].output_root.parents[1]
	manifest_path = model_root / RUN_MANIFEST_NAME
	prior_rows, prior_quarantines = _load_prior_rows(
		manifest_path,
		model=inspection.model,
		dataset_manifest_identity=inspection.dataset_manifest_identity,
		scientific=not smoke_only,
	)
	selected_keys = {_job_key(plan.job) for plan in inspection.plans}
	rows_by_key = {
		_row_key(row): row for row in prior_rows if _row_key(row) not in selected_keys
	}
	quarantines = list(prior_quarantines)
	for plan in inspection.plans:
		job = plan.job
		state = plan.state
		if state == 'REUSE_COMPLETED' and not only_missing:
			raise FileExistsError('completed output requires --only-missing')
		if state == 'RESUME_LATEST' and not resume:
			raise FileExistsError('resumable output requires --resume')
		if state == 'INVALID_OR_PARTIAL':
			if plan.reason and plan.reason.startswith('FOREIGN_IDENTITY:'):
				raise ValueError(plan.reason)
			if not quarantine_invalid:
				raise FileExistsError(
					'invalid/partial output requires --only-missing '
					f'--quarantine-invalid: {job.output_root}: {plan.reason}'
				)
			quarantine = quarantine_voxel_label_budget_output(
				job.output_root, reason=plan.reason or 'invalid_or_partial'
			)
			quarantines.append(quarantine)
			state = 'NEW'
		try:
			if state == 'REUSE_COMPLETED':
				row = _completed_row(config, job, action='REUSED')
				_validate_common_completed_row(row, prior=tuple(rows_by_key.values()))
			else:
				latest = (
					job.decoder_dir / 'latest.pt' if state == 'RESUME_LATEST' else None
				)
				_run_job(
					config,
					job,
					device=device,
					resume=latest,
					max_steps=2 if smoke_only else None,
				)
				if smoke_only:
					row = _smoke_row(config, job)
				else:
					row = _completed_row(
						config, job, action='RESUMED' if latest else 'NEW'
					)
					_validate_common_completed_row(
						row, prior=tuple(rows_by_key.values())
					)
		except BaseException as error:
			rows_by_key[_job_key(job)] = _failed_row(job, error)
			_write_manifest(
				manifest_path,
				inspection=inspection,
				rows=tuple(rows_by_key.values()),
				quarantines=quarantines,
				scientific=not smoke_only,
			)
			raise
		rows_by_key[_job_key(job)] = row
		_write_manifest(
			manifest_path,
			inspection=inspection,
			rows=tuple(rows_by_key.values()),
			quarantines=quarantines,
			scientific=not smoke_only,
		)
	ordered = tuple(sorted(rows_by_key.values(), key=_row_sort_key))
	_write_manifest(
		manifest_path,
		inspection=inspection,
		rows=ordered,
		quarantines=quarantines,
		scientific=not smoke_only,
	)
	return F3SectionLayoutSuiteResult(manifest_path, ordered, tuple(quarantines))


def load_f3_lithology_voxel_section_layout_rows(  # noqa: C901
	manifest: str | Path,
) -> tuple[Mapping[str, object], ...]:
	"""Load and structurally validate rows from one atomic model manifest."""
	payload = _read_json(Path(manifest))
	if (
		payload.get('artifact_type') != RUN_MANIFEST_TYPE
		or payload.get('schema_version') != RUN_SCHEMA_VERSION
	):
		raise ValueError('invalid section-layout run manifest schema')
	if set(payload) != {
		'artifact_type',
		'schema_version',
		'scientific_result',
		'model',
		'dataset_manifest',
		'row_count',
		'complete_count',
		'rows',
		'quarantines',
	}:
		raise ValueError('section-layout run manifest key inventory mismatch')
	values = payload.get('rows')
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		raise TypeError('section-layout run manifest rows must be a list')
	rows = tuple(_mapping(value, 'run manifest row') for value in values)
	model = _mapping(payload.get('model'), 'run manifest model')
	scientific = payload.get('scientific_result')
	if not isinstance(scientific, bool):
		raise TypeError('run manifest scientific_result must be boolean')
	for row in rows:
		_validate_run_manifest_row(row, model=model, scientific=scientific)
	keys = tuple(_row_key(row) for row in rows)
	if len(set(keys)) != len(keys):
		raise ValueError('section-layout run manifest contains duplicate rows')
	if list(rows) != sorted(rows, key=_row_sort_key):
		raise ValueError('section-layout run manifest row-order drift')
	if payload.get('row_count') != len(rows):
		raise ValueError('section-layout run manifest row_count mismatch')
	if payload.get('complete_count') != sum(
		row.get('status') == 'complete' for row in rows
	):
		raise ValueError('section-layout run manifest complete_count mismatch')
	if scientific:
		validated: list[Mapping[str, object]] = []
		for row in rows:
			if row.get('status') == 'complete':
				_validate_common_completed_row(row, prior=validated)
				validated.append(row)
	return rows


def _validate_embedding(  # noqa: C901, PLR0912
	config: F3SectionLayoutBenchmarkConfig, model: F3SectionLayoutModel
) -> Mapping[str, Mapping[str, object]]:
	paths = output_paths(model.embedding_root, config.dataset['name'])
	for path in (paths.embeddings, paths.valid_tokens, paths.metadata):
		if not path.is_file():
			raise FileNotFoundError(path)
	embeddings = np.load(paths.embeddings, mmap_mode='r', allow_pickle=False)
	valid_tokens = np.load(paths.valid_tokens, mmap_mode='r', allow_pickle=False)
	if tuple(embeddings.shape) != EXPECTED_EMBEDDING_SHAPE:
		raise ValueError(
			f'embeddings shape must be exactly {EXPECTED_EMBEDDING_SHAPE!r}'
		)
	if embeddings.dtype != np.dtype(np.float16):
		raise TypeError('embeddings dtype must be float16')
	if tuple(valid_tokens.shape) != EXPECTED_VALID_TOKEN_SHAPE:
		raise ValueError(
			f'valid_tokens shape must be exactly {EXPECTED_VALID_TOKEN_SHAPE!r}'
		)
	if valid_tokens.dtype != np.dtype(np.bool_):
		raise TypeError('valid_tokens dtype must be bool')
	for start in range(0, embeddings.shape[0], 4):
		if not np.isfinite(embeddings[start : start + 4]).all():
			raise ValueError('embeddings contain non-finite values')
	metadata = _read_json(paths.metadata)
	metadata_tag = metadata.get('model_tag')
	if metadata_tag is None:
		pretext = metadata.get('stratigraphy_pretext')
		if isinstance(pretext, Mapping):
			metadata_tag = pretext.get('model_tag')
	checkpoint_value = metadata.get('checkpoint_path')
	if not isinstance(checkpoint_value, str) or not checkpoint_value:
		raise TypeError('embedding metadata checkpoint_path must be non-empty')
	checkpoint = Path(checkpoint_value)
	if metadata_tag is None:
		metadata_tag = model.model_tag if model.model_tag in checkpoint.parts else None
	if metadata_tag != model.model_tag:
		raise ValueError('embedding metadata model tag does not match roster')
	if not checkpoint.is_file():
		raise FileNotFoundError(f'embedding source evidence is missing: {checkpoint}')
	if metadata.get('checkpoint_sha256') != file_sha256(checkpoint):
		raise ValueError('embedding source evidence hash mismatch')
	return {
		'embeddings': _identity(paths.embeddings),
		'valid_tokens': _identity(paths.valid_tokens),
		'embedding_metadata': _identity(paths.metadata),
		'source_evidence': _identity(checkpoint),
	}


def _dataset_rows(manifest: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
	values = cast('Sequence[object]', manifest['rows'])
	rows: list[Mapping[str, object]] = []
	for value in values:
		row = _mapping(value, 'dataset manifest row')
		root = Path(cast('str', row['voxel_dataset_root']))
		live = validate_f3_lithology_voxel_section_layout_condition(root)
		for key in (
			'layout_id',
			'data_size',
			'parent_size',
			'voxel_dataset_root',
			'target_train_voxel_count',
			'actual_train_voxel_count',
			'relative_count_error',
			'selected_token_count',
			'selected_token_identity_sha256',
			'train_mask_sha256',
			'validation_mask_sha256',
			'per_line_contributions',
			'per_class_train_voxel_counts',
			'outputs',
		):
			if live.get(key) != row.get(key):
				raise ValueError(f'dataset condition live identity drift: {key}')
		rows.append(row)
	expected = tuple((layout, size) for layout in LAYOUT_IDS for size in DATA_SIZES)
	if tuple((row['layout_id'], row['data_size']) for row in rows) != expected:
		raise ValueError(
			'dataset manifest must contain the exact ordered 15-row matrix'
		)
	return tuple(rows)


def _validate_dataset_valid_tokens(
	manifest: Mapping[str, object],
	embedding_identity: Mapping[str, Mapping[str, object]],
) -> None:
	sources = _mapping(manifest.get('source_identities'), 'dataset source identities')
	reference = _mapping(sources.get('reference_valid_tokens'), 'dataset valid tokens')
	if reference.get('sha256') != embedding_identity['valid_tokens']['sha256']:
		raise ValueError('embedding valid-token identity differs from dataset contract')


def _classify_job(  # noqa: PLR0911
	config: F3SectionLayoutBenchmarkConfig, job: F3SectionLayoutJob
) -> F3SectionLayoutJobPlan:
	if not job.output_root.exists():
		return F3SectionLayoutJobPlan(job, 'NEW')
	if not job.output_root.is_dir():
		return F3SectionLayoutJobPlan(
			job, 'INVALID_OR_PARTIAL', 'job output is not a directory'
		)
	latest_path = job.decoder_dir / 'latest.pt'
	if not latest_path.is_file():
		return F3SectionLayoutJobPlan(
			job, 'INVALID_OR_PARTIAL', 'missing decoder/latest.pt'
		)
	try:
		latest = load_voxel_decoder_checkpoint(latest_path)
	except Exception as error:  # noqa: BLE001
		return F3SectionLayoutJobPlan(
			job, 'INVALID_OR_PARTIAL', f'invalid latest.pt: {error}'
		)
	expected = _decoder_config(config, job).to_dict()
	if latest.get('resolved_config') != expected:
		return F3SectionLayoutJobPlan(
			job,
			'INVALID_OR_PARTIAL',
			'FOREIGN_IDENTITY: decoder resolved config mismatch',
		)
	if latest.get('checkpoint_kind') != 'completed':
		if job.prediction_dir.exists() or job.evaluation_dir.exists():
			return F3SectionLayoutJobPlan(
				job, 'INVALID_OR_PARTIAL', 'incomplete decoder has downstream outputs'
			)
		try:
			validate_f3_lithology_voxel_decoder_resume(
				_decoder_config(config, job), latest_path
			)
		except Exception as error:  # noqa: BLE001
			return F3SectionLayoutJobPlan(
				job,
				'INVALID_OR_PARTIAL',
				f'FOREIGN_IDENTITY: invalid resume identity: {error}',
			)
		return F3SectionLayoutJobPlan(job, 'RESUME_LATEST')
	try:
		_completed_row(config, job, action='REUSED')
	except Exception as error:  # noqa: BLE001
		return F3SectionLayoutJobPlan(
			job, 'INVALID_OR_PARTIAL', f'completed artifact validation failed: {error}'
		)
	return F3SectionLayoutJobPlan(job, 'REUSE_COMPLETED')


def _decoder_config(
	config: F3SectionLayoutBenchmarkConfig, job: F3SectionLayoutJob
) -> F3LithologyVoxelDecoderConfig:
	return F3LithologyVoxelDecoderConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		model={'tag': job.model.model_tag, 'freeze_encoder': True},
		embeddings_input_dir=job.model.embedding_root,
		voxel_dataset_input_dir=job.dataset_root,
		decoder=config.decoder,
		tiles=config.tiles,
		train=config.train,
		output_dir=job.decoder_dir,
		embeddings={'spec': 'overlap_x16'},
	)


def _inference_config(
	config: F3SectionLayoutBenchmarkConfig, job: F3SectionLayoutJob, *, checkpoint: Path
) -> F3LithologyVoxelInferenceConfig:
	return F3LithologyVoxelInferenceConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		model={'tag': job.model.model_tag, 'freeze_encoder': True},
		class_info=config.labels['class_info'],
		embeddings_input_dir=job.model.embedding_root,
		checkpoint=checkpoint,
		tiles=config.tiles,
		output_dir=job.prediction_dir,
		write_probabilities=False,
		overwrite=False,
	)


def _evaluation_config(
	config: F3SectionLayoutBenchmarkConfig, job: F3SectionLayoutJob
) -> F3LithologyVoxelEvaluationConfig:
	policy = config.evaluation
	return F3LithologyVoxelEvaluationConfig(
		artifact_root=config.artifact_root,
		f3_root=config.f3_root,
		dataset=config.dataset,
		prediction_input_dir=job.prediction_dir,
		voxel_dataset_input_dir=job.dataset_root,
		source_label_volume=config.labels['source_label_volume'],
		source_label_segy=config.labels['source_label_segy'],
		png_label_inventory=config.labels['png_label_inventory'],
		segy_geometry_json=config.labels['segy_geometry_json'],
		class_info=config.labels['class_info'],
		output_dir=job.evaluation_dir,
		monitored_class_ids=tuple(cast('Sequence[int]', policy['monitored_class_ids'])),
		boundary_tolerances=tuple(cast('Sequence[int]', policy['boundary_tolerances'])),
		boundary_region_radii=tuple(
			cast('Sequence[int]', policy['boundary_region_radii'])
		),
		chunk_size_x=cast('int', policy['chunk_size_x']),
		overwrite=False,
	)


def _run_job(
	config: F3SectionLayoutBenchmarkConfig,
	job: F3SectionLayoutJob,
	*,
	device: str,
	resume: Path | None,
	max_steps: int | None,
) -> None:
	job.generated_configs_dir.mkdir(parents=True, exist_ok=True)
	decoder = _decoder_config(config, job)
	_write_json(job.generated_configs_dir / 'decoder_config.json', decoder.to_dict())
	result = run_f3_lithology_voxel_decoder(
		decoder, device=device, resume=resume, max_steps=max_steps
	)
	if max_steps is not None:
		if result.completed or result.global_step != max_steps:
			raise RuntimeError('smoke must stop at exactly two optimizer steps')
		return
	if not result.completed:
		raise RuntimeError('decoder did not complete')
	best = result.best_checkpoint
	inference = _inference_config(config, job, checkpoint=best)
	_write_json(
		job.generated_configs_dir / 'inference_config.json',
		_inference_mapping(inference),
	)
	predict_f3_lithology_voxels(inference, device=device)
	evaluation = _evaluation_config(config, job)
	_write_json(
		job.generated_configs_dir / 'evaluation_config.json',
		_evaluation_mapping(evaluation),
	)
	evaluate_f3_lithology_voxels(evaluation)


def _completed_row(  # noqa: C901, PLR0912
	config: F3SectionLayoutBenchmarkConfig, job: F3SectionLayoutJob, *, action: str
) -> dict[str, object]:
	latest_path = job.decoder_dir / 'latest.pt'
	best_path = job.decoder_dir / 'best.pt'
	for path in (
		latest_path,
		best_path,
		job.decoder_dir / 'history.csv',
		job.decoder_dir / 'resolved_config.json',
	):
		if not path.is_file():
			raise FileNotFoundError(path)
	latest = load_voxel_decoder_checkpoint(latest_path)
	best = load_voxel_decoder_checkpoint(best_path)
	resolved = _decoder_config(config, job).to_dict()
	if (
		latest.get('checkpoint_kind') != 'completed'
		or latest.get('resolved_config') != resolved
		or best.get('resolved_config') != resolved
	):
		raise ValueError('completed checkpoint identity mismatch')
	if latest.get('best_checkpoint_sha256') != file_sha256(best_path):
		raise ValueError('latest checkpoint does not bind best.pt')
	if _read_json(job.decoder_dir / 'resolved_config.json') != resolved:
		raise ValueError('persisted decoder resolved config mismatch')
	generated_expected = {
		'decoder_config.json': resolved,
		'inference_config.json': _inference_mapping(
			_inference_config(config, job, checkpoint=best_path)
		),
		'evaluation_config.json': _evaluation_mapping(_evaluation_config(config, job)),
	}
	for name, expected in generated_expected.items():
		if _read_json(job.generated_configs_dir / name) != expected:
			raise ValueError(f'generated config content mismatch: {name}')
	if latest.get('global_step') != 50 * 440:
		raise ValueError('completed checkpoint global step mismatch')
	run_metadata = _read_json(job.decoder_dir / 'run_metadata.json')
	for key, expected in (
		('train_seed', DECODER_SEED),
		('sampling_mode', 'uniform_tiles_with_replacement'),
		('steps_per_epoch', 440),
	):
		if run_metadata.get(key) != expected:
			raise ValueError(f'run metadata mismatch: {key}')
	train_path = job.decoder_dir / 'train_tile_manifest.json'
	validation_path = job.decoder_dir / 'validation_tile_manifest.json'
	train_tiles = read_voxel_tile_manifest(train_path)
	validation_tiles = read_voxel_tile_manifest(validation_path)
	counts = tuple(
		sum(
			tile.per_class_supervised_counts[str(class_id)]
			for tile in train_tiles.tiles
		)
		for class_id in train_tiles.class_ids
	)
	if sum(counts) != int(job.dataset_row['actual_train_voxel_count']):
		raise ValueError('train tile count differs from dataset teacher count')
	weights = [
		float(value) for value in balanced_class_weights_from_counts(counts).tolist()
	]
	if [
		float(value) for value in cast('Sequence[object]', latest.get('class_weights'))
	] != weights:
		raise ValueError('checkpoint class weights mismatch')
	prediction = validate_f3_voxel_prediction_artifact(
		job.prediction_dir, mmap_mode='r'
	)
	coverage = _mapping(prediction.metadata.get('coverage'), 'prediction coverage')
	exact_once = {
		'exact_once': True,
		'duplicate_write_count': 0,
		'missing_write_count': 0,
		'written_voxel_count': coverage.get('original_voxel_count'),
	}
	for key, expected in exact_once.items():
		if coverage.get(key) != expected:
			raise ValueError(f'prediction exact-once check failed: {key}')
	if prediction.metadata.get('write_probabilities') is not False:
		raise ValueError('probabilities must be disabled')
	plan = inspect_f3_lithology_voxel_inference(
		_inference_config(config, job, checkpoint=best_path), verify_array_hashes=True
	)
	if plan.checkpoint != best_path:
		raise ValueError('inference must use best.pt')
	evaluation = inspect_f3_lithology_voxel_evaluation(_evaluation_config(config, job))
	manifest_validation = int(
		_mapping(
			_read_json(config.dataset_manifest).get('validation_identity'),
			'validation identity',
		)['voxel_count']
	)
	if evaluation.validation_voxel_count != manifest_validation:
		raise ValueError('evaluation validation voxel count mismatch')
	metrics_paths = {
		'metrics': job.evaluation_dir / METRICS_JSON,
		'boundary_metrics': job.evaluation_dir / BOUNDARY_METRICS_JSON,
		'boundary_region_metrics': job.evaluation_dir / BOUNDARY_REGION_METRICS_CSV,
		'evaluation_metadata': job.evaluation_dir / EVALUATION_METADATA_JSON,
	}
	metric_schema = _metric_schema(metrics_paths)
	selected_output = _mapping(job.dataset_row['outputs'], 'dataset outputs')[
		'selected_token_xyz.npy'
	]
	return {
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'model_id': job.model.model_id,
		'model_tag': job.model.model_tag,
		'status': 'complete',
		'action': action,
		'dataset_root': str(job.dataset_root),
		'dataset_grid_identity': dict(
			_mapping(
				_mapping(job.dataset_row['outputs'], 'dataset outputs')[
					'supervision_split_grid.npy'
				],
				'dataset grid identity',
			)
		),
		'train_mask_sha256': job.dataset_row['train_mask_sha256'],
		'validation_mask_sha256': job.dataset_row['validation_mask_sha256'],
		'target_train_voxel_count': job.dataset_row['target_train_voxel_count'],
		'actual_train_voxel_count': job.dataset_row['actual_train_voxel_count'],
		'selected_token_identity_sha256': job.dataset_row[
			'selected_token_identity_sha256'
		],
		'selected_token_file_identity': dict(
			_mapping(selected_output, 'selected tokens')
		),
		'embedding_identities': {
			key: dict(value) for key, value in job.embedding_identity.items()
		},
		'decoder_seed': DECODER_SEED,
		'initial_decoder_state_sha256': run_metadata['initial_model_state_sha256'],
		'class_weights': weights,
		'sampling_sequence_sha256': sampling_sequence_sha256(
			tile_count=len(train_tiles.tiles),
			batch_size=1,
			steps_per_epoch=440,
			train_seed=DECODER_SEED,
			epochs=50,
		),
		'tile_identities': {
			'train_file': _identity(train_path),
			'train': train_tiles.identity_sha256,
			'validation_file': _identity(validation_path),
			'validation': validation_tiles.identity_sha256,
		},
		'metric_schema_sha256': metric_schema,
		'best_checkpoint_inference': {'kind': 'best', **_identity(best_path)},
		'prediction_exact_once_checks': exact_once,
		'canonical_metrics_paths': {
			key: _identity(path) for key, path in metrics_paths.items()
		},
		'error': None,
	}


def _smoke_row(
	config: F3SectionLayoutBenchmarkConfig, job: F3SectionLayoutJob
) -> dict[str, object]:
	latest = job.decoder_dir / 'latest.pt'
	payload = validate_f3_lithology_voxel_decoder_resume(
		_decoder_config(config, job), latest
	)
	if payload.get('global_step') != 2:
		raise ValueError('smoke checkpoint must contain exactly two optimizer steps')
	return {
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'model_id': job.model.model_id,
		'model_tag': job.model.model_tag,
		'status': 'complete',
		'action': 'SMOKE',
		'scientific_result': False,
		'decoder_seed': DECODER_SEED,
		'global_step': 2,
		'dataset_grid_identity': dict(
			_mapping(
				_mapping(job.dataset_row['outputs'], 'outputs')[
					'supervision_split_grid.npy'
				],
				'grid',
			)
		),
		'embedding_identities': {
			key: dict(value) for key, value in job.embedding_identity.items()
		},
		'error': None,
	}


def _metric_schema(paths: Mapping[str, Path]) -> str:
	metrics = _read_json(paths['metrics'])
	boundary = _read_json(paths['boundary_metrics'])
	columns = (
		paths['boundary_region_metrics']
		.read_text(encoding='utf-8')
		.splitlines()[0]
		.split(',')
	)
	payload = {
		'metrics': sorted(metrics),
		'boundary_metrics': sorted(boundary),
		'boundary_region_columns': columns,
	}
	return hashlib.sha256(
		json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
	).hexdigest()


def _inference_mapping(config: F3LithologyVoxelInferenceConfig) -> dict[str, object]:
	return {
		'model_tag': config.model['tag'],
		'embedding_root': str(config.embeddings_input_dir),
		'checkpoint': str(config.checkpoint),
		'output_dir': str(config.output_dir),
		'write_probabilities': config.write_probabilities,
	}


def _evaluation_mapping(config: F3LithologyVoxelEvaluationConfig) -> dict[str, object]:
	return {
		'prediction_input_dir': str(config.prediction_input_dir),
		'voxel_dataset_input_dir': str(config.voxel_dataset_input_dir),
		'output_dir': str(config.output_dir),
		'monitored_class_ids': list(config.monitored_class_ids),
		'boundary_tolerances': list(config.boundary_tolerances),
		'boundary_region_radii': list(config.boundary_region_radii),
		'chunk_size_x': config.chunk_size_x,
	}


def _failed_row(job: F3SectionLayoutJob, error: BaseException) -> dict[str, object]:
	return {
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'model_id': job.model.model_id,
		'model_tag': job.model.model_tag,
		'status': 'failed',
		'action': 'FAILED',
		'error': f'{type(error).__name__}: {error}',
	}


def _validate_run_manifest_row(
	row: Mapping[str, object],
	*,
	model: Mapping[str, object],
	scientific: bool,
) -> None:
	if row.get('layout_id') not in LAYOUT_IDS or row.get('data_size') not in DATA_SIZES:
		raise ValueError('run manifest row condition identity is invalid')
	if row.get('model_id') != model.get('model_id') or row.get(
		'model_tag'
	) != model.get('model_tag'):
		raise ValueError('run manifest row model identity mismatch')
	status = row.get('status')
	if status not in {'complete', 'failed'}:
		raise ValueError('run manifest row status is invalid')
	if not isinstance(row.get('error'), str | type(None)):
		raise TypeError('run manifest row error must be a string or null')
	if status == 'failed':
		return
	if not scientific:
		if row.get('scientific_result') is not False or row.get('global_step') != 2:
			raise ValueError('smoke row must be non-scientific and exactly two steps')
		if row.get('decoder_seed') != DECODER_SEED:
			raise ValueError('smoke row decoder seed drift')
		return
	required = {
		'dataset_grid_identity',
		'train_mask_sha256',
		'validation_mask_sha256',
		'target_train_voxel_count',
		'actual_train_voxel_count',
		'selected_token_identity_sha256',
		'decoder_seed',
		'initial_decoder_state_sha256',
		'class_weights',
		'sampling_sequence_sha256',
		'tile_identities',
		'metric_schema_sha256',
		'best_checkpoint_inference',
		'prediction_exact_once_checks',
		'canonical_metrics_paths',
	}
	missing = required - set(row)
	if missing:
		raise ValueError(
			f'completed run manifest row is missing fields: {sorted(missing)!r}'
		)


def _validate_execution_policy(
	plans: Sequence[F3SectionLayoutJobPlan],
	*,
	only_missing: bool,
	resume: bool,
	quarantine_invalid: bool,
) -> None:
	"""Reject an unsafe matrix before the first job can change state."""
	for plan in plans:
		if plan.state == 'REUSE_COMPLETED' and not only_missing:
			raise FileExistsError('completed output requires --only-missing')
		if plan.state == 'RESUME_LATEST' and not resume:
			raise FileExistsError('resumable output requires --resume')
		if plan.state != 'INVALID_OR_PARTIAL':
			continue
		if plan.reason and plan.reason.startswith('FOREIGN_IDENTITY:'):
			raise ValueError(plan.reason)
		if not quarantine_invalid:
			raise FileExistsError(
				'invalid/partial output requires --only-missing '
				f'--quarantine-invalid: {plan.job.output_root}: {plan.reason}'
			)


def _validate_common_completed_row(
	row: Mapping[str, object],
	*,
	prior: Sequence[Mapping[str, object]],
) -> None:
	"""Keep model-wide decoder and validation identities common across jobs."""
	if row.get('decoder_seed') != DECODER_SEED:
		raise ValueError('completed row decoder seed drift')
	checks = _mapping(
		row.get('prediction_exact_once_checks'), 'prediction exact-once checks'
	)
	if checks.get('exact_once') is not True:
		raise ValueError('completed row lacks exact-once prediction evidence')
	completed = tuple(item for item in prior if item.get('status') == 'complete')
	if not completed:
		return
	reference = completed[0]
	for key in (
		'initial_decoder_state_sha256',
		'metric_schema_sha256',
		'validation_mask_sha256',
	):
		if row.get(key) != reference.get(key):
			raise ValueError(f'completed model-row common identity mismatch: {key}')
	row_tiles = _mapping(row.get('tile_identities'), 'completed row tile identities')
	reference_tiles = _mapping(
		reference.get('tile_identities'), 'reference row tile identities'
	)
	if row_tiles.get('validation') != reference_tiles.get('validation'):
		raise ValueError('completed model-row validation tile identity mismatch')


def _load_prior_rows(
	path: Path,
	*,
	model: F3SectionLayoutModel,
	dataset_manifest_identity: Mapping[str, object],
	scientific: bool,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Path, ...]]:
	if not path.exists():
		return (), ()
	payload = _read_json(path)
	rows = load_f3_lithology_voxel_section_layout_rows(path)
	if payload.get('scientific_result') is not scientific:
		raise ValueError('prior manifest scientific/smoke identity mismatch')
	if payload.get('model') != {
		'model_id': model.model_id,
		'model_tag': model.model_tag,
	}:
		raise ValueError('prior manifest model identity mismatch')
	if payload.get('dataset_manifest') != dataset_manifest_identity:
		raise ValueError('prior manifest dataset identity mismatch')
	quarantines = payload.get('quarantines')
	if not isinstance(quarantines, Sequence) or isinstance(quarantines, str | bytes):
		raise TypeError('prior manifest quarantines must be a list')
	return rows, tuple(Path(str(value)) for value in quarantines)


def _write_manifest(
	path: Path,
	*,
	inspection: F3SectionLayoutSuiteInspection,
	rows: Sequence[Mapping[str, object]],
	quarantines: Sequence[Path],
	scientific: bool,
) -> None:
	ordered = sorted(rows, key=_row_sort_key)
	payload = {
		'artifact_type': RUN_MANIFEST_TYPE,
		'schema_version': RUN_SCHEMA_VERSION,
		'scientific_result': scientific,
		'model': {
			'model_id': inspection.model.model_id,
			'model_tag': inspection.model.model_tag,
		},
		'dataset_manifest': dict(inspection.dataset_manifest_identity),
		'row_count': len(ordered),
		'complete_count': sum(row.get('status') == 'complete' for row in ordered),
		'rows': ordered,
		'quarantines': [str(value) for value in quarantines],
	}
	_write_json(path, payload)
	load_f3_lithology_voxel_section_layout_rows(path)


def _identity(path: Path) -> dict[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {'path': str(path.resolve()), 'sha256': file_sha256(path)}


def _read_json(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as handle:
		payload = json.load(handle)
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON root must be a mapping: {path}')
	return cast('Mapping[str, object]', payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.tmp')
	temporary.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	temporary.replace(path)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return cast('Mapping[str, object]', value)


def _job_key(job: F3SectionLayoutJob) -> tuple[str, str]:
	return job.layout_id, job.data_size


def _row_key(row: Mapping[str, object]) -> tuple[str, str]:
	return str(row.get('layout_id')), str(row.get('data_size'))


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, int]:
	key = _row_key(row)
	if key[0] not in LAYOUT_IDS or key[1] not in DATA_SIZES:
		raise ValueError(f'invalid run manifest condition key: {key!r}')
	return LAYOUT_IDS.index(key[0]), DATA_SIZES.index(key[1])


__all__ = [
	'inspect_f3_lithology_voxel_section_layout_suite',
	'load_f3_lithology_voxel_section_layout_rows',
	'run_f3_lithology_voxel_section_layout_suite',
]
