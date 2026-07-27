"""Preflight and matrix planning for the selected multi-head six-split suite."""
# ruff: noqa: D101, E501, PLR0915

from __future__ import annotations

import csv
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	VoxelDecoderSpec,
	VoxelDecoderTileSettings,
	VoxelDecoderTrainSettings,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_split import MODEL_TAGS
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	_decoder_config,
	_validated_smoke_row,
	classify_voxel_label_budget_job,
	completed_voxel_label_budget_job_row,
	quarantine_voxel_label_budget_output,
	run_voxel_label_budget_job,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split import (
	_complete,
	_json_sha256,
	_strict_source_identities,
)
from seis_ssl_cluster.f3.splits import read_f3_line_geometry
from seis_ssl_cluster.training.voxel_decoder.runner import (
	inspect_f3_lithology_voxel_decoder,
	run_f3_lithology_voxel_decoder,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_split import (
		F3VoxelLabelBudgetSplitConfig,
	)


@dataclass(frozen=True)
class LowLabelSplitJob:
	split_id: str
	budget_id: str
	model_role: str
	output_root: Path


class _SplitStageConfig:
	"""Fixed adapter exposing this suite through the shared decoder runner."""

	def __init__(
		self,
		config: F3VoxelLabelBudgetSplitConfig,
		dataset_row: Mapping[str, object],
		model_role: str,
	) -> None:
		self.artifact_root = config.artifact_root
		self.dataset_manifest = config.output_root / 'low_label_split_dataset_manifest.json'
		self.output_root = config.output_root
		self.dataset = {'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v1'}
		self.model_by_role = {model_role: _SplitModel(model_role, config.embeddings[model_role])}
		metadata_path = Path(str(dataset_row['voxel_dataset_root'])) / 'voxel_dataset_metadata.json'
		metadata = _json(metadata_path)
		self.labels = _split_labels(dataset_row, metadata)
		self.f3_root = self.labels['source_label_segy'].parent
		self.decoder = VoxelDecoderSpec(spec='frozen_embedding_decoder_nearest_voxel_ln_v1', embedding_dim=384, class_count=6, hidden_channels=(128, 64, 32), upsample_factors=((2, 2, 2), (2, 2, 2), (2, 2, 2)), upsample_mode='nearest', normalization='voxelwise_layer_norm')
		self.tiles = VoxelDecoderTileSettings((8, 8, 8), (1, 1, 1))
		self.train = VoxelDecoderTrainSettings(epochs=50, batch_size=1, learning_rate=0.001, weight_decay=0.0001, class_weight='balanced', seed=42000, num_workers=0, amp=True, gradient_clip_norm=1.0, sampling_mode='uniform_tiles_with_replacement', steps_per_epoch=440)
		self.evaluation = {'monitored_class_ids': [3, 5], 'boundary_tolerances': [2, 4], 'boundary_region_radii': [2, 4], 'chunk_size_x': 8}
		self.report = {'selected_slices': {'inline': [], 'crossline': []}, 'dpi': 150, 'include_confidence': False, 'amplitude_clip_percentiles': [1.0, 99.0]}
		self.overwrite = False
		self.write_probabilities = False
		self.publish_individual_reports = False


@dataclass(frozen=True)
class _SplitModel:
	role: str
	embeddings_dir: Path
	model_tag: str = ''

	def __post_init__(self) -> None:
		object.__setattr__(self, 'model_tag', MODEL_TAGS[self.role])


def inspect_f3_lithology_voxel_label_budget_split_suite(config: F3VoxelLabelBudgetSplitConfig) -> tuple[LowLabelSplitJob, ...]:  # noqa: C901
	"""Check M4 handoff/embedding identities and return the exact 36-job matrix."""
	decision = _json(config.multi_head_decisions)
	if decision.get('overall_status') != 'M4_MH_GO_NOCONS' or decision.get('selected_candidate') != 'mh_nocons':
		raise ValueError('M4 decision does not select mh_nocons')
	effects = _mapping(decision.get('effects'))
	if _mapping(effects.get('multi_task_value')).get('status') != 'POSITIVE' or _mapping(effects.get('nocons_vs_mae')).get('status') != 'POSITIVE':
		raise ValueError('M4 positive-value decision gate failed')
	handoff = _json(config.multi_head_handoff)
	pretext = _mapping(handoff.get('stratigraphy_pretext'))
	if (
		handoff.get('status') != 'PASS'
		or handoff.get('model_tag') != 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
		or pretext.get('head_spec') != 'multi_resolution_ordered_prototypes_v1'
		or pretext.get('head_ks') != [6, 8, 10]
		or pretext.get('consistency_weight') != 0.0
	):
		raise ValueError('mh_nocons handoff contract failed')
	_target_manifest_identity(handoff, pretext)
	_multi_head_embedding_identity(config, handoff, pretext)
	canonical = None
	for role, root in config.embeddings.items():
		files = output_paths(root, 'f3_facies_benchmark')
		valid = files.valid_tokens
		if not valid.is_file() or not files.embeddings.is_file() or not files.metadata.is_file():
			raise FileNotFoundError(root)
		embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
		valid_values = np.load(valid, mmap_mode='r', allow_pickle=False)
		if embeddings.shape != (76, 113, 32, 384) or embeddings.dtype != np.float16:
			raise ValueError(f'embedding shape/dtype contract failed: {role}')
		if valid_values.shape != (76, 113, 32) or valid_values.dtype != np.bool_:
			raise ValueError(f'valid-token shape/dtype contract failed: {role}')
		actual = file_sha256(valid)
		if canonical is None:
			canonical = actual
		elif actual != canonical:
			raise ValueError(f'valid-token identity mismatch: {role}')
	if _mapping(handoff.get('embedding')).get('valid_tokens_sha256') != canonical:
		raise ValueError('multi-head handoff valid-token binding mismatch')
	jobs = tuple(
		LowLabelSplitJob(split_id, budget_id, model, config.output_root / 'runs' / split_id / budget_id / model)
		for split_id in config.split_ids for budget_id in config.budgets for model in config.models
	)
	rows = _dataset_rows(config)
	for job in jobs:
		row = rows[(job.split_id, job.budget_id)]
		stage = _SplitStageConfig(config, row, job.model_role)
		planned = VoxelLabelBudgetJob(
			job.budget_id, int(row['per_class_cap']), 0, config.decoder_seed,
			job.model_role, stage.model_by_role[job.model_role].model_tag,
			Path(str(row['voxel_dataset_root'])), job.output_root, row,
		)
		inspect_f3_lithology_voxel_decoder(_decoder_config(stage, planned))
	return jobs


def run_f3_lithology_voxel_label_budget_split_suite(  # noqa: C901, PLR0912, PLR0913
	config: F3VoxelLabelBudgetSplitConfig,
	*,
	only_missing: bool = False,
	resume: bool = False,
	device: str = 'auto',
	split_id: str | None = None,
	budget: str | None = None,
	model_role: str | None = None,
	smoke_only: bool = False,
) -> tuple[Mapping[str, object], ...]:
	"""Execute the selected decoder jobs through the shared checked runner."""
	jobs = [job for job in inspect_f3_lithology_voxel_label_budget_split_suite(config) if (split_id is None or job.split_id == split_id) and (budget is None or job.budget_id == budget) and (model_role is None or job.model_role == model_role)]
	if smoke_only and {(job.split_id, job.budget_id, job.model_role) for job in jobs} != {('split_000', 'cap25', role) for role in config.models}:
		raise ValueError('smoke gate must be the split_000/cap25 three-model triplet')
	if not smoke_only and model_role is not None:
		raise ValueError('full decoder execution requires all three models for each selected split/budget')
	if not jobs:
		raise ValueError('filters selected no jobs')
	rows = _dataset_rows(config)
	manifest_root = _run_manifest_root(config, smoke_only=smoke_only)
	prior = _load_prior_rows(rows, manifest_root=manifest_root)
	completed = dict(prior)
	_write_run_manifest(manifest_root, list(completed.values()))
	for planned in jobs:
		row = rows[(planned.split_id, planned.budget_id)]
		output_root = planned.output_root if not smoke_only else config.output_root / 'smoke' / planned.split_id / planned.budget_id / planned.model_role
		stage = _SplitStageConfig(config, row, planned.model_role)
		job = VoxelLabelBudgetJob(planned.budget_id, int(row['per_class_cap']), 0, config.decoder_seed, planned.model_role, stage.model_by_role[planned.model_role].model_tag, Path(str(row['voxel_dataset_root'])), output_root, row)
		plan = classify_voxel_label_budget_job(stage, job)
		quarantine_path = None
		if plan.state == 'REUSE_COMPLETED':
			if not only_missing:
				raise FileExistsError(f'completed job requires --only-missing: {job.output_root}')
			action, checkpoint = 'REUSED', None
		elif plan.state == 'RESUME_LATEST':
			if smoke_only:
				quarantine_path = str(quarantine_voxel_label_budget_output(
					job.output_root, reason='stale_smoke_checkpoint'
				))
				action, checkpoint = 'NEW', None
			elif not (only_missing or resume):
				raise FileExistsError(f'incomplete job requires --only-missing or --resume: {job.output_root}')
			else:
				action, checkpoint = 'RESUMED', job.decoder_dir / 'latest.pt'
		else:
			if plan.state == 'INVALID_OR_PARTIAL':
				if not only_missing:
					raise FileExistsError(f'invalid job requires --only-missing: {job.output_root}')
				quarantine_path = str(quarantine_voxel_label_budget_output(
					job.output_root, reason=plan.reason or 'invalid_or_partial'
				))
				completed[_row_key(planned.split_id, planned.budget_id, planned.model_role)] = _progress_row(planned, 'quarantined', quarantine_path, plan.reason)
				_write_run_manifest(manifest_root, list(completed.values()))
			action, checkpoint = 'NEW', None
		completed[_row_key(planned.split_id, planned.budget_id, planned.model_role)] = _progress_row(planned, 'running', quarantine_path, None)
		_write_run_manifest(manifest_root, list(completed.values()))
		try:
			if action != 'REUSED':
				if smoke_only:
					run_f3_lithology_voxel_decoder(
						_decoder_config(stage, job), device=device, max_steps=2
					)
				else:
					run_voxel_label_budget_job(stage, job, device=device, resume=checkpoint)
			result = (
				_smoke_job_row(stage, job, action=action, quarantine_path=quarantine_path)
				if smoke_only
				else completed_voxel_label_budget_job_row(
					stage, job, action=action, quarantine_path=quarantine_path, error=None
				)
			)
			completed[_row_key(planned.split_id, planned.budget_id, planned.model_role)] = {'split_id': planned.split_id, **result}
		except BaseException as error:
			completed[_row_key(planned.split_id, planned.budget_id, planned.model_role)] = _progress_row(planned, 'failed', quarantine_path, f'{type(error).__name__}: {error}')
			_write_run_manifest(manifest_root, list(completed.values()))
			raise
		_write_run_manifest(manifest_root, list(completed.values()))
	if smoke_only:
		current = list(completed.values())
		_shared_condition_contract(current, models=config.models, context='smoke gate')
		if any(row.get('global_step') != 2 for row in current):
			raise ValueError('smoke gate must execute exactly two decoder steps')
	else:
		_complete_rows = [row for row in completed.values() if row.get('status') == 'complete']
		complete_triplets = _selected_complete_triplets(
			_complete_rows, jobs, models=config.models
		)
		if complete_triplets:
			_shared_condition_contract(
				complete_triplets,
				models=config.models,
				context='full decoder run',
			)
		_write_run_manifest(manifest_root, list(completed.values()))
	return tuple(completed.values())


def _dataset_rows(config: F3VoxelLabelBudgetSplitConfig) -> Mapping[tuple[str, str], Mapping[str, object]]:
	payload = _json(config.output_root / 'low_label_split_dataset_manifest.json')
	rows = payload.get('rows')
	if not isinstance(rows, list):
		raise TypeError('six-split dataset manifest rows are missing')
	indexed = {(str(row.get('split_id')), str(row.get('budget_id'))): row for row in rows if isinstance(row, Mapping)}
	if len(indexed) != 12:
		raise ValueError('six-split dataset manifest must contain exactly twelve unique rows')
	strict_sources = _strict_source_identities(
		config, split_id='six-split dataset manifest'
	)
	strict_sources_sha256 = _json_sha256(strict_sources)
	for (split_id, budget_id), row in indexed.items():
		root = Path(str(row.get('voxel_dataset_root', '')))
		if not _complete(root, row):
			raise ValueError(
				'six-split dataset source/provenance identity mismatch: '
				f'{split_id}/{budget_id}'
			)
		if (
			row.get('source_identities') != strict_sources
			or row.get('source_identities_sha256') != strict_sources_sha256
		):
			raise ValueError(
				'six-split dataset strict source identity mismatch: '
				f'{split_id}/{budget_id}'
			)
	return indexed


def _split_labels(
	dataset_row: Mapping[str, object],
	metadata: Mapping[str, object],
) -> Mapping[str, Path]:
	"""Resolve every evaluation/report input from committed dataset provenance."""
	if metadata.get('artifact_type') != 'f3_lithology_voxel_supervision':
		raise ValueError('voxel dataset metadata schema is invalid')
	inventory = _validated_identity(metadata.get('inventory'), 'png label inventory')
	label_volume = _validated_identity(metadata.get('label_volume'), 'label volume')
	valid_tokens = _validated_identity(
		metadata.get('reference_valid_tokens'), 'reference valid tokens'
	)
	if valid_tokens['sha256'] != dataset_row.get('canonical_valid_tokens_sha256'):
		raise ValueError('voxel dataset valid-token identity mismatch')
	labels = _mapping(metadata.get('labels'))
	sources = _mapping(metadata.get('source_identities'))
	class_info = _validated_identity(sources.get('class_info'), 'class info')
	declared_segy = _validated_identity(
		sources.get('source_label_segy'), 'source label SEGY'
	)
	geometry = _validated_identity(
		sources.get('segy_geometry_json'), 'SEGY geometry'
	)
	if Path(str(labels.get('class_info', ''))) != Path(str(class_info['path'])):
		raise ValueError('voxel dataset class-info provenance path mismatch')
	if Path(str(labels.get('source_label_segy', ''))) != Path(str(declared_segy['path'])):
		raise ValueError('voxel dataset label SEGY provenance path mismatch')
	reference = _mapping(metadata.get('reference_embedding'))
	reference_metadata = _mapping(reference.get('metadata'))
	seismic_volume = _validated_identity(
		sources.get('seismic_volume'), 'seismic volume'
	)
	if Path(str(reference_metadata.get('source_amplitude_path', ''))) != Path(
		str(seismic_volume['path'])
	):
		raise ValueError('voxel dataset seismic-volume provenance path mismatch')
	if dict(read_f3_line_geometry(Path(str(geometry['path']))).to_dict()) != dict(
		_mapping(metadata.get('geometry'))
	):
		raise ValueError('voxel dataset SEGY geometry identity mismatch')
	return {
		'png_label_inventory': Path(str(inventory['path'])),
		'source_label_volume': Path(str(label_volume['path'])),
		'source_label_segy': Path(str(declared_segy['path'])),
		'class_info': Path(str(class_info['path'])),
		'segy_geometry_json': Path(str(geometry['path'])),
		'seismic_volume': Path(str(seismic_volume['path'])),
	}


def _validated_identity(value: object, label: str) -> Mapping[str, object]:
	identity = _mapping(value)
	path = Path(str(identity.get('path', '')))
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	if not isinstance(identity.get('sha256'), str) or identity['sha256'] != file_sha256(path):
		raise ValueError(f'{label} metadata path/hash mismatch')
	return identity


def _row_key(split_id: str, budget_id: str, model_role: str) -> tuple[str, str, str]:
	return split_id, budget_id, model_role


def _progress_row(
	job: LowLabelSplitJob, status: str, quarantine_path: str | None, error: str | None
) -> Mapping[str, object]:
	return {
		'split_id': job.split_id,
		'budget_id': job.budget_id,
		'model_role': job.model_role,
		'status': status,
		'error': error,
		'quarantine_path': quarantine_path,
		'resume_eligible': status in {'failed', 'running'},
	}


def _load_prior_rows(
	dataset_rows: Mapping[tuple[str, str], Mapping[str, object]],
	*,
	manifest_root: Path,
) -> dict[tuple[str, str, str], Mapping[str, object]]:
	path = manifest_root / 'low_label_split_run_manifest.json'
	if not path.is_file():
		return {}
	payload = _json(path)
	if payload.get('artifact_type') != 'f3_lithology_voxel_label_budget_split_run_manifest':
		raise ValueError('prior run manifest type mismatch')
	rows = payload.get('rows')
	if not isinstance(rows, list):
		raise TypeError('prior run manifest rows are missing')
	indexed: dict[tuple[str, str, str], Mapping[str, object]] = {}
	for value in rows:
		row = _mapping(value)
		key = _row_key(*(str(row.get(name, '')) for name in ('split_id', 'budget_id', 'model_role')))
		if key in indexed:
			raise ValueError(f'prior run manifest has duplicate key: {key!r}')
		if key[:2] not in dataset_rows:
			raise ValueError(f'prior run manifest has unknown condition: {key!r}')
		if row.get('status') == 'complete':
			dataset_row = dataset_rows[key[:2]]
			for identity_key in (
				'selected_token_identity_sha256',
				'unique_token_xyz_sha256',
				'validation_mask_sha256',
			):
				if row.get(identity_key) != dataset_row.get(identity_key):
					raise ValueError(
						f'prior run manifest has stale dataset identity: {key!r}/{identity_key}'
					)
		indexed[key] = row
	return indexed


def _selected_complete_triplets(
	rows: list[Mapping[str, object]],
	jobs: list[LowLabelSplitJob],
	*,
	models: tuple[str, ...],
) -> list[Mapping[str, object]]:
	"""Return selected conditions only after their complete model triplet is present."""
	conditions = {(job.split_id, job.budget_id) for job in jobs}
	by_condition: dict[tuple[str, str], list[Mapping[str, object]]] = {}
	for row in rows:
		condition = (str(row.get('split_id')), str(row.get('budget_id')))
		if condition in conditions:
			by_condition.setdefault(condition, []).append(row)
	return [
		row
		for members in by_condition.values()
		if len(members) == len(models)
		and {str(row.get('model_role')) for row in members} == set(models)
		for row in members
	]


def _atomic_write_text(path: Path, value: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		'w', encoding='utf-8', dir=path.parent, prefix=f'.{path.name}.', delete=False
	) as handle:
		handle.write(value)
		temporary = Path(handle.name)
	temporary.replace(path)


def _run_manifest_root(
	config: F3VoxelLabelBudgetSplitConfig, *, smoke_only: bool
) -> Path:
	return config.output_root / 'smoke' if smoke_only else config.output_root


def _write_run_manifest(manifest_root: Path, rows: list[Mapping[str, object]]) -> None:
	path = manifest_root / 'low_label_split_run_manifest.json'
	ordered = sorted(rows, key=lambda row: _row_key(str(row.get('split_id')), str(row.get('budget_id')), str(row.get('model_role'))))
	_atomic_write_text(path, json.dumps({'artifact_type': 'f3_lithology_voxel_label_budget_split_run_manifest', 'schema_version': 1, 'rows': ordered}, indent=2, sort_keys=True) + '\n')
	_write_csv(manifest_root / 'low_label_split_job_status.csv', ordered)
	complete = [
		row
		for row in ordered
		if row.get('status') == 'complete' and 'evaluation_metrics' in row
	]
	_write_csv(manifest_root / 'low_label_split_job_metrics.csv', [_metric_row(row) for row in complete])


def _metric_row(row: Mapping[str, object]) -> Mapping[str, object]:
	metrics_path = Path(str(_mapping(row['evaluation_metrics']).get('path')))
	boundary_path = Path(str(_mapping(row['evaluation_boundary_metrics']).get('path')))
	regions_path = Path(str(_mapping(row['evaluation_boundary_region_metrics']).get('path')))
	values = load_f3_lithology_voxel_label_budget_evaluation_metrics(
		metrics_path=metrics_path,
		boundary_metrics_path=boundary_path,
		boundary_region_metrics_path=regions_path,
		label=(
			f"{row['split_id']}/{row['budget_id']}/{row['model_role']}"
		),
	)
	return {
		'split_id': row['split_id'],
		'budget_id': row['budget_id'],
		'model_role': row['model_role'],
		**values,
	}


def _smoke_job_row(
	stage: _SplitStageConfig,
	job: VoxelLabelBudgetJob,
	*,
	action: str,
	quarantine_path: str | None,
) -> Mapping[str, object]:
	"""Validate a two-step decoder checkpoint without scientific evaluation."""
	values = _validated_smoke_row(
		stage, job, checkpoint=job.decoder_dir / 'latest.pt'
	)
	row = job.dataset_row
	return {
		**values,
		'action': action,
		'error': None,
		'quarantine_path': quarantine_path,
		'status': 'complete',
		'voxel_supervision_grid_sha256': _mapping(
			row['supervision_split_grid']
		)['sha256'],
		'selected_token_identity_sha256': row['selected_token_identity_sha256'],
		'unique_token_xyz_sha256': row['unique_token_xyz_sha256'],
		'train_voxel_count': row['train_voxel_count'],
		'validation_voxel_count': row['validation_voxel_count'],
		'validation_mask_sha256': row['validation_mask_sha256'],
		'canonical_valid_token_sha256': row['canonical_valid_tokens_sha256'],
		'metric_schema_sha256': 'smoke_not_evaluated',
	}


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
	fieldnames = sorted({key for row in rows for key in row})
	if not fieldnames:
		fieldnames = ['split_id', 'budget_id', 'model_role', 'status']
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, prefix=f'.{path.name}.', delete=False) as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
		temporary = Path(handle.name)
	temporary.replace(path)


def _shared_condition_contract(
	rows: list[Mapping[str, object]], *, models: tuple[str, ...], context: str
) -> None:
	"""Require each selected condition to retain the paired decoder contract."""
	by_condition: dict[tuple[str, str], list[Mapping[str, object]]] = {}
	for row in rows:
		condition = (str(row.get('split_id')), str(row.get('budget_id')))
		by_condition.setdefault(condition, []).append(row)
	keys = (
		'voxel_supervision_grid_sha256',
		'selected_token_identity_sha256',
		'unique_token_xyz_sha256',
		'train_voxel_count',
		'validation_voxel_count',
		'validation_mask_sha256',
		'canonical_valid_token_sha256',
		'class_order',
		'class_weights',
		'initial_model_state_sha256',
		'decoder_architecture',
		'decoder_seed',
		'train_tile_manifest_sha256',
		'validation_tile_manifest_sha256',
		'train_tile_identity_sha256',
		'validation_tile_identity_sha256',
		'sampling_mode',
		'steps_per_epoch',
		'sampling_sequence_sha256',
		'global_step',
		'metric_schema_sha256',
	)
	for condition, members in by_condition.items():
		if {str(row.get('model_role')) for row in members} != set(models) or len(members) != len(models):
			raise ValueError(f'{context} must complete one three-model triplet: {condition[0]}/{condition[1]}')
		for key in keys:
			if any(key not in row for row in members):
				raise ValueError(f'{context} shared contract is missing: {condition[0]}/{condition[1]}/{key}')
			if len({json.dumps(row.get(key), sort_keys=True) for row in members}) != 1:
				raise ValueError(f'{context} shared contract mismatch: {condition[0]}/{condition[1]}/{key}')


def _target_manifest_identity(handoff: Mapping[str, object], pretext: Mapping[str, object]) -> None:
	target_path = Path(str(pretext.get('target_manifest_path', '')))
	target_sha = pretext.get('target_manifest_sha256')
	if not target_path.is_file() or not isinstance(target_sha, str) or target_sha != file_sha256(target_path):
		raise ValueError('multi-head target manifest binding mismatch')
	if _mapping(handoff.get('stratigraphy_pretext')).get('target_manifest_sha256') != target_sha:
		raise ValueError('multi-head handoff target manifest binding mismatch')


def _multi_head_embedding_identity(
	config: F3VoxelLabelBudgetSplitConfig,
	handoff: Mapping[str, object],
	pretext: Mapping[str, object],
) -> None:
	files = output_paths(config.embeddings['mh_nocons'], 'f3_facies_benchmark')
	metadata = _json(files.metadata)
	checkpoint = Path(str(metadata.get('checkpoint_path', '')))
	checkpoint_record = _mapping(handoff.get('checkpoint'))
	if (
		not checkpoint.is_file()
		or checkpoint.name != 'best.pt'
		or metadata.get('checkpoint_sha256') != file_sha256(checkpoint)
		or Path(str(checkpoint_record.get('path', ''))).resolve() != checkpoint.resolve()
		or checkpoint_record.get('sha256') != metadata.get('checkpoint_sha256')
	):
		raise ValueError('multi-head checkpoint/embedding binding mismatch')
	embedding = _mapping(handoff.get('embedding'))
	if (
		embedding.get('metadata_sha256') != file_sha256(files.metadata)
		or embedding.get('embeddings_sha256') != file_sha256(files.embeddings)
		or Path(str(embedding.get('metadata_path', ''))).resolve() != files.metadata.resolve()
	):
		raise ValueError('multi-head embedding handoff identity mismatch')
	stratigraphy = _mapping(metadata.get('stratigraphy_pretext'))
	for key in ('head_spec', 'head_ks', 'consistency_weight', 'target_manifest_sha256'):
		if stratigraphy.get(key) != pretext.get(key):
			raise ValueError(f'multi-head embedding/handoff stratigraphy mismatch: {key}')


def _json(path: Path) -> Mapping[str, object]:
	return _mapping(json.loads(path.read_text(encoding='utf-8')))


def _mapping(value: object) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError('expected a JSON object')
	return value


__all__ = ['LowLabelSplitJob', 'inspect_f3_lithology_voxel_label_budget_split_suite']
