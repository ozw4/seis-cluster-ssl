"""Preflight and matrix planning for the selected multi-head six-split suite."""
# ruff: noqa: D101, E501

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from seis_ssl_cluster.f3.lithology.voxel_label_budget_runner import (
	VoxelLabelBudgetJob,
	classify_voxel_label_budget_job,
	completed_voxel_label_budget_job_row,
	quarantine_voxel_label_budget_output,
	run_voxel_label_budget_job,
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

	def __init__(self, config: F3VoxelLabelBudgetSplitConfig, model_role: str) -> None:
		self.artifact_root = config.artifact_root
		self.f3_root = Path('/home/dcuser/data/public_data/field/F3')
		self.dataset_manifest = config.output_root / 'low_label_split_dataset_manifest.json'
		self.output_root = config.output_root
		self.dataset = {'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v1'}
		self.model_by_role = {model_role: _SplitModel(model_role, config.embeddings[model_role])}
		self.labels = {
			'seismic_volume': config.artifact_root / 'registry/volumes/f3/facies_benchmark_v1/f3_seismic.npy',
			'source_label_volume': config.artifact_root / 'registry/volumes/f3/facies_benchmark_v1/f3_facies_labels.npy',
			'source_label_segy': self.f3_root / 'f3_labels.sgy',
			'png_label_inventory': config.artifact_root / 'inspection/f3/facies_benchmark_v1/inventory/label_png_inventory.csv',
			'segy_geometry_json': config.artifact_root / 'inspection/f3/facies_benchmark_v1/segy/segy_geometry.json',
			'class_info': config.artifact_root / 'inspection/f3/facies_benchmark_v1/inventory/class_info.json',
		}
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
	return tuple(
		LowLabelSplitJob(split_id, budget_id, model, config.output_root / 'runs' / split_id / budget_id / model)
		for split_id in config.split_ids for budget_id in config.budgets for model in config.models
	)


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
	if only_missing and resume:
		raise ValueError('--only-missing and --resume are mutually exclusive')
	jobs = [job for job in inspect_f3_lithology_voxel_label_budget_split_suite(config) if (split_id is None or job.split_id == split_id) and (budget is None or job.budget_id == budget) and (model_role is None or job.model_role == model_role)]
	if smoke_only and {(job.split_id, job.budget_id, job.model_role) for job in jobs} != {('split_000', 'cap25', role) for role in config.models}:
		raise ValueError('smoke gate must be the split_000/cap25 three-model triplet')
	if not smoke_only and model_role is not None:
		raise ValueError('full decoder execution requires all three models for each selected split/budget')
	if not jobs:
		raise ValueError('filters selected no jobs')
	rows = _dataset_rows(config)
	completed = []
	for planned in jobs:
		row = rows[(planned.split_id, planned.budget_id)]
		output_root = planned.output_root if not smoke_only else config.output_root / 'smoke' / planned.split_id / planned.budget_id / planned.model_role
		job = VoxelLabelBudgetJob(planned.budget_id, int(row['per_class_cap']), 0, config.decoder_seed, planned.model_role, _SplitStageConfig(config, planned.model_role).model_by_role[planned.model_role].model_tag, Path(str(row['voxel_dataset_root'])), output_root, row)
		stage = _SplitStageConfig(config, planned.model_role)
		if smoke_only:
			stage.train = replace(stage.train, epochs=1, steps_per_epoch=2)
		plan = classify_voxel_label_budget_job(stage, job)
		if plan.state == 'REUSE_COMPLETED':
			if not only_missing:
				raise FileExistsError(f'completed job requires --only-missing: {job.output_root}')
			action, checkpoint = 'REUSED', None
		elif plan.state == 'RESUME_LATEST':
			if not resume:
				raise FileExistsError(f'incomplete job requires --resume: {job.output_root}')
			action, checkpoint = 'RESUMED', job.decoder_dir / 'latest.pt'
		else:
			if plan.state == 'INVALID_OR_PARTIAL':
				if not only_missing:
					raise FileExistsError(f'invalid job requires --only-missing: {job.output_root}')
				quarantine_voxel_label_budget_output(job.output_root, reason=plan.reason or 'invalid_or_partial')
			action, checkpoint = 'NEW', None
		if action != 'REUSED':
			run_voxel_label_budget_job(stage, job, device=device, resume=checkpoint)
		result = completed_voxel_label_budget_job_row(stage, job, action=action, quarantine_path=None, error=None)
		completed.append({'split_id': planned.split_id, **result})
	if smoke_only:
		_shared_condition_contract(completed, models=config.models, context='smoke gate')
		if any(row.get('global_step') != 2 for row in completed):
			raise ValueError('smoke gate must execute exactly two decoder steps')
	else:
		_shared_condition_contract(completed, models=config.models, context='full decoder run')
		_write_run_manifest(config, completed)
	return tuple(completed)


def _dataset_rows(config: F3VoxelLabelBudgetSplitConfig) -> Mapping[tuple[str, str], Mapping[str, object]]:
	payload = _json(config.output_root / 'low_label_split_dataset_manifest.json')
	rows = payload.get('rows')
	if not isinstance(rows, list):
		raise TypeError('six-split dataset manifest rows are missing')
	indexed = {(str(row.get('split_id')), str(row.get('budget_id'))): row for row in rows if isinstance(row, Mapping)}
	if len(indexed) != 12:
		raise ValueError('six-split dataset manifest must contain exactly twelve unique rows')
	return indexed


def _write_run_manifest(config: F3VoxelLabelBudgetSplitConfig, rows: list[Mapping[str, object]]) -> None:
	path = config.output_root / 'low_label_split_run_manifest.json'
	path.write_text(json.dumps({'artifact_type': 'f3_lithology_voxel_label_budget_split_run_manifest', 'schema_version': 1, 'rows': rows}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
	_write_csv(config.output_root / 'low_label_split_job_status.csv', rows)
	_write_csv(config.output_root / 'low_label_split_job_metrics.csv', [_metric_row(row) for row in rows])


def _metric_row(row: Mapping[str, object]) -> Mapping[str, object]:
	metrics_path = Path(str(_mapping(row['evaluation_metrics']).get('path')))
	boundary_path = Path(str(_mapping(row['evaluation_boundary_metrics']).get('path')))
	regions_path = Path(str(_mapping(row['evaluation_boundary_region_metrics']).get('path')))
	metrics = _json(metrics_path)
	boundary = _json(boundary_path)
	with regions_path.open(encoding='utf-8', newline='') as handle:
		regions = {int(item['radius']): item for item in csv.DictReader(handle) if item.get('region') == 'boundary'}
	result = {'split_id': row['split_id'], 'budget_id': row['budget_id'], 'model_role': row['model_role'], **{name: metrics[name] for name in ('macro_f1', 'mean_iou', 'balanced_accuracy', 'accuracy', 'weighted_f1')}}
	for class_id in (3, 5):
		result[f'class_{class_id}_f1'] = _metric_at(metrics, 'per_class_f1', class_id)
		result[f'class_{class_id}_iou'] = _metric_at(metrics, 'per_class_iou', class_id)
		result[f'class_{class_id}_boundary_recall_tolerance_2'] = boundary[f'vertical_boundary_class_{class_id}_recall_at_2']
		result[f'class_{class_id}_boundary_recall_tolerance_4'] = boundary[f'vertical_boundary_class_{class_id}_recall_at_4']
	for radius in (2, 4):
		result[f'boundary_region_macro_f1_r{radius}'] = regions[radius]['macro_f1']
		result[f'boundary_region_mean_iou_r{radius}'] = regions[radius]['mean_iou']
	result['boundary_f1_t2'] = boundary['vertical_boundary_f1_at_2']
	result['boundary_f1_t4'] = boundary['vertical_boundary_f1_at_4']
	result['boundary_position_mae'] = boundary['vertical_boundary_position_mae_at_2']
	return result


def _metric_at(metrics: Mapping[str, object], key: str, class_id: int) -> object:
	values = metrics[key]
	if not isinstance(values, list):
		raise TypeError(f'evaluation metric {key} must be a list')
	return values[class_id]


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
	if not rows:
		return
	with path.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


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
