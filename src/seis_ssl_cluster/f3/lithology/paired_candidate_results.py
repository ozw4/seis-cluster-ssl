"""Paired results for two F3 lithology candidate representations."""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_five_way import (
	F3FiveWayConfig,
	F3FiveWayModelSource,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	F3LithologyCandidateConfig,
	audit_f3_lithology_candidate_source,
	load_f3_lithology_candidate_canonical_config,
)
from seis_ssl_cluster.f3.lithology.completed_decoder_contract import (
	read_f3_completed_decoder_contract,
)
from seis_ssl_cluster.f3.lithology.completed_evaluation_provenance import (
	read_f3_completed_evaluation_provenance,
)
from seis_ssl_cluster.f3.lithology.five_way_results import (
	EXPECTED_AGGREGATION_UNIT,
	read_f3_lithology_job_evidence,
)
from seis_ssl_cluster.f3.lithology.local_bt_hmm_source import (
	F3LocalBTSourceRole,
	validate_f3_local_bt_candidate_source,
)

PRIMARY_METRIC = 'macro_f1'
OUTPUT_NAMES = ('comparison.csv', 'summary.json', 'summary.md')
COMPARISON_FIELDNAMES = (
	'data_size',
	'layout_id',
	'candidate_id',
	'control_id',
	'candidate_macro_f1',
	'control_macro_f1',
	'candidate_minus_control',
	'evaluation_voxel_count',
	'supervision_identity_sha256',
	'validation_mask_sha256',
	'validation_tile_manifest_sha256',
	'voxel_valid_mask_sha256',
	'voxel_dataset_metadata_path',
	'voxel_dataset_metadata_sha256',
	'label_volume_path',
	'label_volume_sha256',
	'class_info_path',
	'class_info_sha256',
	'source_label_segy_path',
	'source_label_segy_sha256',
	'png_label_inventory_path',
	'png_label_inventory_sha256',
	'segy_geometry_json_path',
	'segy_geometry_json_sha256',
	'candidate_metrics_path',
	'candidate_metrics_sha256',
	'control_metrics_path',
	'control_metrics_sha256',
	'candidate_evaluation_metadata_path',
	'candidate_evaluation_metadata_sha256',
	'control_evaluation_metadata_path',
	'control_evaluation_metadata_sha256',
	'candidate_prediction_metadata_path',
	'candidate_prediction_metadata_sha256',
	'control_prediction_metadata_path',
	'control_prediction_metadata_sha256',
	'candidate_voxel_predictions_sha256',
	'control_voxel_predictions_sha256',
	'candidate_voxel_confidence_sha256',
	'control_voxel_confidence_sha256',
	'candidate_voxel_valid_mask_sha256',
	'control_voxel_valid_mask_sha256',
	'candidate_decoder_checkpoint_sha256',
	'control_decoder_checkpoint_sha256',
	'candidate_decoder_initial_state_sha256',
	'control_decoder_initial_state_sha256',
	'decoder_contract_identity_sha256',
	'candidate_decoder_resolved_config_sha256',
	'control_decoder_resolved_config_sha256',
	'candidate_decoder_run_metadata_sha256',
	'control_decoder_run_metadata_sha256',
	'candidate_decoder_latest_checkpoint_path',
	'candidate_decoder_latest_checkpoint_sha256',
	'control_decoder_latest_checkpoint_path',
	'control_decoder_latest_checkpoint_sha256',
	'candidate_decoder_best_checkpoint_path',
	'candidate_decoder_best_checkpoint_sha256',
	'control_decoder_best_checkpoint_path',
	'control_decoder_best_checkpoint_sha256',
	'candidate_decoder_completed_epoch',
	'candidate_decoder_completed_global_step',
	'control_decoder_completed_epoch',
	'control_decoder_completed_global_step',
	'candidate_decoder_best_epoch',
	'candidate_decoder_best_global_step',
	'control_decoder_best_epoch',
	'control_decoder_best_global_step',
	'candidate_decoder_train_tile_manifest_sha256',
	'control_decoder_train_tile_manifest_sha256',
	'candidate_decoder_validation_tile_manifest_sha256',
	'control_decoder_validation_tile_manifest_sha256',
)


@dataclass(frozen=True)
class F3PairedCandidateExpectedSource:
	"""Expected completed training point and its source configurations."""

	base_training_config: Path
	stratigraphy_training_config: Path | None
	epoch: int
	global_step: int


@dataclass(frozen=True)
class F3PairedCandidateModel:
	"""One current checkpoint and embedding source in a paired comparison."""

	model_id: str
	checkpoint: Path
	embeddings_dir: Path
	expected: F3PairedCandidateExpectedSource


@dataclass(frozen=True)
class F3PairedCandidateSummaryConfig:
	"""Resolved inputs and output for one exact 15-cell paired comparison."""

	canonical_config: Path
	runs_root: Path
	summary_root: Path
	candidate: F3PairedCandidateModel
	control: F3PairedCandidateModel


def f3_paired_candidate_summary_config_from_mapping(
	config: Mapping[str, object],
) -> F3PairedCandidateSummaryConfig:
	"""Resolve the strict paired-candidate summary configuration."""
	_require_exact_keys(
		config,
		{'benchmark', 'inputs', 'models', 'outputs'},
		'config',
	)
	benchmark = _required_mapping(config, 'benchmark', 'config')
	inputs = _required_mapping(config, 'inputs', 'config')
	models = _required_mapping(config, 'models', 'config')
	outputs = _required_mapping(config, 'outputs', 'config')
	_require_exact_keys(benchmark, {'canonical_config'}, 'benchmark')
	_require_exact_keys(inputs, {'runs_root'}, 'inputs')
	_require_exact_keys(models, {'candidate', 'control'}, 'models')
	_require_exact_keys(outputs, {'summary_root'}, 'outputs')
	candidate = _model_from_mapping(
		_required_mapping(models, 'candidate', 'models'),
		label='models.candidate',
	)
	control = _model_from_mapping(
		_required_mapping(models, 'control', 'models'),
		label='models.control',
	)
	if candidate.model_id == control.model_id:
		raise ValueError('candidate and control model IDs must differ')
	if candidate.checkpoint == control.checkpoint:
		raise ValueError('candidate and control checkpoints must differ')
	if candidate.embeddings_dir == control.embeddings_dir:
		raise ValueError('candidate and control embedding directories must differ')
	if candidate.expected.base_training_config != control.expected.base_training_config:
		raise ValueError('candidate and control base training configs must match')
	if candidate.expected.stratigraphy_training_config is None:
		raise ValueError('candidate stratigraphy training config must be provided')
	if control.expected.stratigraphy_training_config is not None:
		raise ValueError('control stratigraphy training config must be null')
	runs_root = _absolute_path(inputs.get('runs_root'), 'inputs.runs_root')
	summary_root = _absolute_path(outputs.get('summary_root'), 'outputs.summary_root')
	if _paths_overlap(runs_root, summary_root):
		raise ValueError('inputs.runs_root and outputs.summary_root must not overlap')
	return F3PairedCandidateSummaryConfig(
		canonical_config=_absolute_path(
			benchmark.get('canonical_config'), 'benchmark.canonical_config'
		),
		runs_root=runs_root,
		summary_root=summary_root,
		candidate=candidate,
		control=control,
	)


def inspect_f3_paired_candidate_results(
	config: F3PairedCandidateSummaryConfig,
) -> dict[str, object]:
	"""Audit both current sources and all exact canonical-v3 paired cells."""
	provenance, rows, validation_identity = _audit_results(config)
	return {
		'schema_version': 1,
		'candidate_id': config.candidate.model_id,
		'control_id': config.control.model_id,
		'complete_candidate_jobs': len(rows),
		'complete_control_jobs': len(rows),
		'compared_cells': len(rows),
		'source_lineages_audited': True,
		'downstream_identity_parity': True,
		'benchmark': _benchmark_provenance(config),
		'shared_validation_identity': validation_identity,
		'shared_evaluation_inputs': _shared_evaluation_inputs(rows),
		'sources': provenance,
	}


def summarize_f3_paired_candidate_results(
	config: F3PairedCandidateSummaryConfig,
) -> tuple[Path, Path, Path]:
	"""Audit and atomically write exactly three paired summary files."""
	provenance, rows, validation_identity = _audit_results(config)
	if config.summary_root.exists():
		raise FileExistsError(
			f'refusing to overwrite summary directory: {config.summary_root}'
		)
	aggregates = {
		'by_size': {
			data_size: _aggregate_scope(
				[row for row in rows if row['data_size'] == data_size],
				expected_count=len(LAYOUT_IDS),
				label=data_size,
				descriptive_unit='layout_id',
				paired_t_unit='layout_id',
			)
			for data_size in DATA_SIZES
		},
		'all_15': _aggregate_scope(
			rows,
			expected_count=len(DATA_SIZES) * len(LAYOUT_IDS),
			label='all_15',
			descriptive_unit='layout_id_x_data_size_cell',
			paired_t_unit=None,
		),
		'overall_layout_clustered': _aggregate_layout_clustered(rows),
	}
	payload = {
		'schema_version': 1,
		'summary_name': 'f3_lithology_paired_candidates',
		'primary_metric': PRIMARY_METRIC,
		'aggregation_unit': EXPECTED_AGGREGATION_UNIT,
		'statistical_units': {
			'by_size_paired_t': {
				'unit': 'layout_id',
				'n': len(LAYOUT_IDS),
			},
			'all_15_descriptive': {
				'unit': 'layout_id_x_data_size_cell',
				'n': len(DATA_SIZES) * len(LAYOUT_IDS),
				'paired_t': None,
			},
			'overall_layout_clustered_paired_t': {
				'unit': 'layout_id',
				'n': len(LAYOUT_IDS),
				'within_layout_reduction': 'mean_across_data_sizes',
			},
		},
		'candidate_id': config.candidate.model_id,
		'control_id': config.control.model_id,
		'job_counts': {
			'candidate': len(rows),
			'control': len(rows),
			'compared_cells': len(rows),
		},
		'source_lineages_audited': True,
		'downstream_identity_parity': True,
		'benchmark': _benchmark_provenance(config),
		'shared_validation_identity': validation_identity,
		'shared_evaluation_inputs': _shared_evaluation_inputs(rows),
		'delta_convention': (
			'candidate minus control; positive values favor the candidate'
		),
		'sources': provenance,
		'model_means': _model_means(config, aggregates),
		'comparison': rows,
		'aggregates': aggregates,
	}
	contents = {
		OUTPUT_NAMES[0]: _comparison_csv(rows),
		OUTPUT_NAMES[1]: json.dumps(
			payload,
			allow_nan=False,
			indent=2,
			sort_keys=True,
		)
		+ '\n',
		OUTPUT_NAMES[2]: _summary_markdown(config, rows, aggregates),
	}
	return _write_atomic_summary(config.summary_root, contents)


def _audit_results(
	config: F3PairedCandidateSummaryConfig,
) -> tuple[
	dict[str, dict[str, object]],
	list[dict[str, object]],
	dict[str, object],
]:
	canonical = load_f3_lithology_candidate_canonical_config(
		_candidate_config(config, config.candidate)
	)
	if config.control.model_id in canonical.model_ids:
		raise ValueError(
			'control model ID conflicts with a canonical five-way model ID: '
			f'{config.control.model_id!r}'
		)
	provenance = {
		'candidate': _audit_source(
			config, canonical, config.candidate, role='candidate'
		),
		'control': _audit_source(config, canonical, config.control, role='control'),
	}
	_validate_source_pair(provenance)
	rows = _audit_job_rows(config, canonical, provenance)
	return provenance, rows, _shared_validation_identity(rows)


def _candidate_config(
	config: F3PairedCandidateSummaryConfig,
	model: F3PairedCandidateModel,
) -> F3LithologyCandidateConfig:
	return F3LithologyCandidateConfig(
		canonical_config=config.canonical_config,
		candidate_id=model.model_id,
		checkpoint=model.checkpoint,
		embeddings_dir=model.embeddings_dir,
		runs_root=config.runs_root,
		summary_root=config.summary_root,
	)


def _audit_source(
	config: F3PairedCandidateSummaryConfig,
	canonical: F3FiveWayConfig,
	model: F3PairedCandidateModel,
	*,
	role: F3LocalBTSourceRole,
) -> dict[str, object]:
	provenance = audit_f3_lithology_candidate_source(
		_candidate_config(config, model), canonical
	)
	semantic_lineage = validate_f3_local_bt_candidate_source(
		model.checkpoint,
		Path(str(provenance['embedding_metadata_path'])),
		role=role,
		base_training_config=model.expected.base_training_config,
		stratigraphy_training_config=model.expected.stratigraphy_training_config,
		epoch=model.expected.epoch,
		global_step=model.expected.global_step,
	)
	return {
		'model_id': provenance['candidate_id'],
		'semantic_lineage': semantic_lineage,
		**{
			key: provenance[key]
			for key in (
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
	}


def _validate_source_pair(
	provenance: Mapping[str, Mapping[str, object]],
) -> None:
	candidate = provenance['candidate']
	control = provenance['control']
	if candidate['valid_tokens_sha256'] != control['valid_tokens_sha256']:
		raise ValueError('candidate and control valid-token identities differ')
	if candidate['checkpoint_sha256'] == control['checkpoint_sha256']:
		raise ValueError('candidate and control checkpoint contents are identical')
	if candidate['embeddings_sha256'] == control['embeddings_sha256']:
		raise ValueError('candidate and control embedding contents are identical')


def _audit_job_rows(
	config: F3PairedCandidateSummaryConfig,
	canonical: F3FiveWayConfig,
	provenance: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
	_reject_unexpected_job_directories(config, config.candidate)
	_reject_unexpected_job_directories(config, config.control)
	rows: list[dict[str, object]] = []
	for data_size in DATA_SIZES:
		for layout_id in LAYOUT_IDS:
			candidate = _read_job(
				config,
				canonical,
				config.candidate,
				provenance['candidate'],
				layout_id=layout_id,
				data_size=data_size,
			)
			control = _read_job(
				config,
				canonical,
				config.control,
				provenance['control'],
				layout_id=layout_id,
				data_size=data_size,
			)
			_assert_paired_identity(
				candidate['evidence'],
				control['evidence'],
				label=f'{layout_id}/{data_size}',
			)
			_assert_paired_decoder_identity(
				candidate['decoder_contract'],
				control['decoder_contract'],
				label=f'{layout_id}/{data_size}',
			)
			_assert_paired_evaluation_identity(
				candidate['evaluation'],
				control['evaluation'],
				label=f'{layout_id}/{data_size}',
			)
			candidate_value = float(candidate['macro_f1'])
			control_value = float(control['macro_f1'])
			candidate_evidence = candidate['evidence']
			control_evidence = control['evidence']
			candidate_decoder = candidate['decoder_contract']
			control_decoder = control['decoder_contract']
			candidate_evaluation = candidate['evaluation']
			control_evaluation = control['evaluation']
			if not isinstance(candidate_evidence, Mapping) or not isinstance(
				control_evidence, Mapping
			):
				raise TypeError('audited job evidence must be a mapping')
			if not isinstance(candidate_decoder, Mapping) or not isinstance(
				control_decoder, Mapping
			):
				raise TypeError('audited decoder contract must be a mapping')
			if not isinstance(candidate_evaluation, Mapping) or not isinstance(
				control_evaluation, Mapping
			):
				raise TypeError('audited evaluation provenance must be a mapping')
			rows.append(
				{
					'data_size': data_size,
					'layout_id': layout_id,
					'candidate_id': config.candidate.model_id,
					'control_id': config.control.model_id,
					'candidate_macro_f1': candidate_value,
					'control_macro_f1': control_value,
					'candidate_minus_control': candidate_value - control_value,
					'evaluation_voxel_count': candidate_evidence[
						'validation_voxel_count'
					],
					'supervision_identity_sha256': _required_sha256(
						candidate_evidence.get('supervision_identity'),
						label=f'{layout_id}/{data_size} supervision identity',
					),
					'validation_mask_sha256': candidate_evidence['validation_identity'],
					'validation_tile_manifest_sha256': candidate_evidence[
						'_validation_tile_manifest_sha256'
					],
					'voxel_valid_mask_sha256': candidate_evaluation[
						'voxel_valid_mask_sha256'
					],
					'voxel_dataset_metadata_path': candidate_evaluation[
						'voxel_dataset_metadata_path'
					],
					'voxel_dataset_metadata_sha256': candidate_evaluation[
						'voxel_dataset_metadata_sha256'
					],
					**{
						f'{name}_{suffix}': candidate_evaluation[f'{name}_{suffix}']
						for name in (
							'label_volume',
							'class_info',
							'source_label_segy',
							'png_label_inventory',
							'segy_geometry_json',
						)
						for suffix in ('path', 'sha256')
					},
					'candidate_metrics_path': candidate_evaluation['metrics_path'],
					'candidate_metrics_sha256': candidate_evaluation['metrics_sha256'],
					'control_metrics_path': control_evaluation['metrics_path'],
					'control_metrics_sha256': control_evaluation['metrics_sha256'],
					'candidate_evaluation_metadata_path': candidate_evaluation[
						'evaluation_metadata_path'
					],
					'candidate_evaluation_metadata_sha256': candidate_evaluation[
						'evaluation_metadata_sha256'
					],
					'control_evaluation_metadata_path': control_evaluation[
						'evaluation_metadata_path'
					],
					'control_evaluation_metadata_sha256': control_evaluation[
						'evaluation_metadata_sha256'
					],
					'candidate_prediction_metadata_path': candidate_evaluation[
						'prediction_metadata_path'
					],
					'candidate_prediction_metadata_sha256': candidate_evaluation[
						'prediction_metadata_sha256'
					],
					'control_prediction_metadata_path': control_evaluation[
						'prediction_metadata_path'
					],
					'control_prediction_metadata_sha256': control_evaluation[
						'prediction_metadata_sha256'
					],
					'candidate_voxel_predictions_sha256': candidate_evaluation[
						'voxel_predictions_sha256'
					],
					'control_voxel_predictions_sha256': control_evaluation[
						'voxel_predictions_sha256'
					],
					'candidate_voxel_confidence_sha256': candidate_evaluation[
						'voxel_confidence_sha256'
					],
					'control_voxel_confidence_sha256': control_evaluation[
						'voxel_confidence_sha256'
					],
					'candidate_voxel_valid_mask_sha256': candidate_evaluation[
						'voxel_valid_mask_sha256'
					],
					'control_voxel_valid_mask_sha256': control_evaluation[
						'voxel_valid_mask_sha256'
					],
					'candidate_decoder_checkpoint_sha256': candidate_evidence[
						'decoder_checkpoint_sha256'
					],
					'control_decoder_checkpoint_sha256': control_evidence[
						'decoder_checkpoint_sha256'
					],
					'candidate_decoder_initial_state_sha256': candidate_decoder[
						'decoder_initial_state_sha256'
					],
					'control_decoder_initial_state_sha256': control_decoder[
						'decoder_initial_state_sha256'
					],
					'decoder_contract_identity_sha256': candidate_decoder[
						'decoder_contract_identity_sha256'
					],
					'candidate_decoder_resolved_config_sha256': candidate_decoder[
						'decoder_resolved_config_sha256'
					],
					'control_decoder_resolved_config_sha256': control_decoder[
						'decoder_resolved_config_sha256'
					],
					'candidate_decoder_run_metadata_sha256': candidate_decoder[
						'decoder_run_metadata_sha256'
					],
					'control_decoder_run_metadata_sha256': control_decoder[
						'decoder_run_metadata_sha256'
					],
					'candidate_decoder_latest_checkpoint_path': candidate_decoder[
						'decoder_latest_checkpoint_path'
					],
					'candidate_decoder_latest_checkpoint_sha256': candidate_decoder[
						'decoder_latest_checkpoint_sha256'
					],
					'control_decoder_latest_checkpoint_path': control_decoder[
						'decoder_latest_checkpoint_path'
					],
					'control_decoder_latest_checkpoint_sha256': control_decoder[
						'decoder_latest_checkpoint_sha256'
					],
					'candidate_decoder_best_checkpoint_path': candidate_decoder[
						'decoder_best_checkpoint_path'
					],
					'candidate_decoder_best_checkpoint_sha256': candidate_decoder[
						'decoder_best_checkpoint_sha256'
					],
					'control_decoder_best_checkpoint_path': control_decoder[
						'decoder_best_checkpoint_path'
					],
					'control_decoder_best_checkpoint_sha256': control_decoder[
						'decoder_best_checkpoint_sha256'
					],
					'candidate_decoder_completed_epoch': candidate_decoder[
						'decoder_completed_epoch'
					],
					'candidate_decoder_completed_global_step': candidate_decoder[
						'decoder_completed_global_step'
					],
					'control_decoder_completed_epoch': control_decoder[
						'decoder_completed_epoch'
					],
					'control_decoder_completed_global_step': control_decoder[
						'decoder_completed_global_step'
					],
					'candidate_decoder_best_epoch': candidate_decoder[
						'decoder_best_epoch'
					],
					'candidate_decoder_best_global_step': candidate_decoder[
						'decoder_best_global_step'
					],
					'control_decoder_best_epoch': control_decoder['decoder_best_epoch'],
					'control_decoder_best_global_step': control_decoder[
						'decoder_best_global_step'
					],
					'candidate_decoder_train_tile_manifest_sha256': candidate_decoder[
						'decoder_train_tile_manifest_sha256'
					],
					'control_decoder_train_tile_manifest_sha256': control_decoder[
						'decoder_train_tile_manifest_sha256'
					],
					'candidate_decoder_validation_tile_manifest_sha256': (
						candidate_decoder['decoder_validation_tile_manifest_sha256']
					),
					'control_decoder_validation_tile_manifest_sha256': (
						control_decoder['decoder_validation_tile_manifest_sha256']
					),
				}
			)
	return rows


def _shared_validation_identity(
	rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	expected_count = len(DATA_SIZES) * len(LAYOUT_IDS)
	if len(rows) != expected_count:
		raise ValueError(
			f'shared validation audit requires {expected_count} cells; got {len(rows)}'
		)
	fields = (
		'validation_mask_sha256',
		'validation_tile_manifest_sha256',
		'voxel_valid_mask_sha256',
		'evaluation_voxel_count',
	)
	identity: dict[str, object] = {}
	for field in fields:
		values = {row.get(field) for row in rows}
		if len(values) != 1:
			raise ValueError(f'paired 15-cell {field} values are not shared')
		identity[field] = next(iter(values))
	return identity


def _shared_evaluation_inputs(
	rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
	expected_count = len(DATA_SIZES) * len(LAYOUT_IDS)
	if len(rows) != expected_count:
		raise ValueError(
			f'shared evaluation-input audit requires {expected_count} cells; '
			f'got {len(rows)}'
		)
	result: dict[str, dict[str, object]] = {}
	for name in (
		'label_volume',
		'class_info',
		'source_label_segy',
		'png_label_inventory',
		'segy_geometry_json',
	):
		path_key = f'{name}_path'
		sha_key = f'{name}_sha256'
		paths = {row.get(path_key) for row in rows}
		sha256_values = {row.get(sha_key) for row in rows}
		if len(paths) != 1 or len(sha256_values) != 1:
			raise ValueError(f'paired 15-cell {name} path/SHA values are not shared')
		result[name] = {
			'path': next(iter(paths)),
			'sha256': next(iter(sha256_values)),
		}
	return result


def _benchmark_provenance(
	config: F3PairedCandidateSummaryConfig,
) -> dict[str, str]:
	if not config.canonical_config.is_file():
		raise FileNotFoundError(
			f'missing canonical benchmark config: {config.canonical_config}'
		)
	return {
		'canonical_config_path': str(config.canonical_config),
		'canonical_config_sha256': file_sha256(config.canonical_config),
	}


def _reject_unexpected_job_directories(
	config: F3PairedCandidateSummaryConfig,
	model: F3PairedCandidateModel,
) -> None:
	model_root = config.runs_root / f'model={model.model_id}'
	if not model_root.is_dir():
		return
	expected_layouts = {f'layout={layout_id}' for layout_id in LAYOUT_IDS}
	for layout_entry in sorted(model_root.iterdir()):
		if layout_entry.name not in expected_layouts or not layout_entry.is_dir():
			raise ValueError(f'unexpected paired run condition: {layout_entry}')
		expected_sizes = {f'size={data_size}' for data_size in DATA_SIZES}
		for size_entry in sorted(layout_entry.iterdir()):
			if size_entry.name not in expected_sizes or not size_entry.is_dir():
				raise ValueError(f'unexpected paired run condition: {size_entry}')


def _read_job(  # noqa: PLR0913
	config: F3PairedCandidateSummaryConfig,
	canonical: F3FiveWayConfig,
	model: F3PairedCandidateModel,
	provenance: Mapping[str, object],
	*,
	layout_id: str,
	data_size: str,
) -> dict[str, object]:
	job_dir = (
		config.runs_root
		/ f'model={model.model_id}'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
	)
	model_source = F3FiveWayModelSource(
		model_id=model.model_id,
		checkpoint=model.checkpoint,
		embeddings_dir=model.embeddings_dir,
		expected={},
	)
	evidence = read_f3_lithology_job_evidence(
		canonical,
		model=model_source,
		layout_id=layout_id,
		data_size=data_size,
		job_dir=job_dir,
	)
	decoder_contract = read_f3_completed_decoder_contract(
		job_dir,
		model_id=model.model_id,
		layout_id=layout_id,
		data_size=data_size,
		freeze_encoder=True,
	)
	evaluation = read_f3_completed_evaluation_provenance(
		job_dir,
		model_id=model.model_id,
		layout_id=layout_id,
		data_size=data_size,
	)
	_assert_job_source_matches(
		evidence,
		provenance,
		label=f'{model.model_id}/{layout_id}/{data_size}',
	)
	_assert_evaluation_matches(
		evaluation,
		evidence=evidence,
		decoder_contract=decoder_contract,
		canonical=canonical,
		label=f'{model.model_id}/{layout_id}/{data_size}',
	)
	return {
		'evidence': evidence,
		'decoder_contract': decoder_contract,
		'evaluation': evaluation,
		'macro_f1': evaluation['macro_f1'],
	}


def _assert_evaluation_matches(
	evaluation: Mapping[str, object],
	*,
	evidence: Mapping[str, object],
	decoder_contract: Mapping[str, object],
	canonical: F3FiveWayConfig,
	label: str,
) -> None:
	if evaluation.get('evaluation_voxel_count') != evidence.get(
		'validation_voxel_count'
	):
		raise ValueError(f'{label} evaluation/evidence voxel counts differ')
	if evaluation.get('decoder_checkpoint_sha256') != evidence.get(
		'decoder_checkpoint_sha256'
	):
		raise ValueError(
			f'{label} evaluation decoder checkpoint does not match job evidence'
		)
	if decoder_contract.get(
		'decoder_validation_tile_manifest_identity_sha256'
	) != evidence.get('_validation_tile_manifest_sha256'):
		raise ValueError(
			f'{label} decoder validation tile identity does not match job evidence'
		)
	for evaluation_key, decoder_key in (
		('decoder_checkpoint_path', 'decoder_best_checkpoint_path'),
		('decoder_checkpoint_sha256', 'decoder_best_checkpoint_sha256'),
		('prediction_metadata_path', 'prediction_metadata_path'),
		('prediction_metadata_sha256', 'prediction_metadata_sha256'),
	):
		if evaluation.get(evaluation_key) != decoder_contract.get(decoder_key):
			raise ValueError(
				f'{label} evaluation {evaluation_key} does not match decoder '
				'completion evidence'
			)
	for key in ('decoder_resolved_config_path', 'decoder_resolved_config_sha256'):
		if evaluation.get(key) != decoder_contract.get(key):
			raise ValueError(
				f'{label} evaluation {key} does not match decoder completion evidence'
			)
	for canonical_key, evidence_key in (
		('source_label_volume', 'label_volume'),
		('class_info', 'class_info'),
		('source_label_segy', 'source_label_segy'),
		('png_label_inventory', 'png_label_inventory'),
		('segy_geometry_json', 'segy_geometry_json'),
	):
		expected_path = canonical.labels.get(canonical_key)
		if evaluation.get(f'{evidence_key}_path') != str(expected_path):
			raise ValueError(
				f'{label} evaluation {evidence_key} does not match the canonical '
				'benchmark config'
			)


def _assert_job_source_matches(
	evidence: Mapping[str, object],
	provenance: Mapping[str, object],
	*,
	label: str,
) -> None:
	for evidence_key, provenance_key in (
		('encoder_checkpoint_sha256', 'checkpoint_sha256'),
		('embeddings_sha256', 'embeddings_sha256'),
		('embedding_metadata_sha256', 'embedding_metadata_sha256'),
		('valid_tokens_sha256', 'valid_tokens_sha256'),
	):
		if evidence.get(evidence_key) != provenance.get(provenance_key):
			raise ValueError(
				f'{label} completed job {evidence_key} does not match current source'
			)


def _assert_paired_identity(
	candidate: object,
	control: object,
	*,
	label: str,
) -> None:
	if not isinstance(candidate, Mapping) or not isinstance(control, Mapping):
		raise TypeError(f'{label} paired job evidence must be mappings')
	for key in (
		'supervision_identity',
		'validation_identity',
		'_validation_tile_manifest_sha256',
		'validation_voxel_count',
	):
		if key != 'validation_voxel_count':
			_required_sha256(candidate.get(key), label=f'{label} candidate {key}')
			_required_sha256(control.get(key), label=f'{label} control {key}')
		if candidate.get(key) != control.get(key):
			raise ValueError(f'{label} candidate/control {key} differ')


def _assert_paired_decoder_identity(
	candidate: object,
	control: object,
	*,
	label: str,
) -> None:
	if not isinstance(candidate, Mapping) or not isinstance(control, Mapping):
		raise TypeError(f'{label} paired decoder contracts must be mappings')
	for key in (
		'decoder_contract_identity_sha256',
		'decoder_initial_state_sha256',
		'decoder_completed_epoch',
		'decoder_completed_global_step',
		'decoder_train_tile_manifest_identity_sha256',
		'decoder_validation_tile_manifest_identity_sha256',
	):
		if candidate.get(key) != control.get(key):
			raise ValueError(f'{label} candidate/control {key} differ')


def _assert_paired_evaluation_identity(
	candidate: object,
	control: object,
	*,
	label: str,
) -> None:
	if not isinstance(candidate, Mapping) or not isinstance(control, Mapping):
		raise TypeError(f'{label} paired evaluation provenance must be mappings')
	for key in (
		'aggregation_unit',
		'evaluation_voxel_count',
		'voxel_valid_mask_sha256',
		'voxel_dataset_metadata_path',
		'voxel_dataset_metadata_sha256',
		'label_volume_path',
		'label_volume_sha256',
		'class_info_path',
		'class_info_sha256',
		'source_label_segy_path',
		'source_label_segy_sha256',
		'png_label_inventory_path',
		'png_label_inventory_sha256',
		'segy_geometry_json_path',
		'segy_geometry_json_sha256',
	):
		if candidate.get(key) != control.get(key):
			raise ValueError(f'{label} candidate/control evaluation {key} differ')


def _read_macro_f1(
	path: Path,
	*,
	evidence: Mapping[str, object],
	label: str,
) -> float:
	if not path.is_file():
		raise FileNotFoundError(f'missing completed metrics for {label}: {path}')
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} metrics must contain a JSON object')
	if payload.get('aggregation_unit') != EXPECTED_AGGREGATION_UNIT:
		raise ValueError(
			f'{label} metrics aggregation_unit must equal {EXPECTED_AGGREGATION_UNIT!r}'
		)
	if payload.get('evaluation_voxel_count') != evidence.get('validation_voxel_count'):
		raise ValueError(f'{label} metrics/evidence voxel counts differ')
	value = payload.get(PRIMARY_METRIC)
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{label} metrics {PRIMARY_METRIC} must be numeric')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label} metrics {PRIMARY_METRIC} must be finite')
	if not 0.0 <= result <= 1.0:
		raise ValueError(f'{label} metrics {PRIMARY_METRIC} must be within [0, 1]')
	return result


def _aggregate_scope(
	rows: Sequence[Mapping[str, object]],
	*,
	expected_count: int,
	label: str,
	descriptive_unit: str,
	paired_t_unit: str | None,
) -> dict[str, object]:
	if len(rows) != expected_count:
		raise ValueError(
			f'{label} must contain exactly {expected_count} paired cells; '
			f'got {len(rows)}'
		)
	values = [float(row['candidate_minus_control']) for row in rows]
	statistics_payload = _delta_statistics(values)
	if paired_t_unit is not None:
		statistics_payload['paired_t_statistic'] = _paired_t_statistic(values)
	return {
		'n': len(rows),
		'descriptive_unit': descriptive_unit,
		'paired_t_statistical_unit': paired_t_unit,
		'candidate_mean': statistics.fmean(
			float(row['candidate_macro_f1']) for row in rows
		),
		'control_mean': statistics.fmean(
			float(row['control_macro_f1']) for row in rows
		),
		'candidate_minus_control': statistics_payload,
	}


def _aggregate_layout_clustered(
	rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
	expected_count = len(DATA_SIZES) * len(LAYOUT_IDS)
	if len(rows) != expected_count:
		raise ValueError(
			'overall layout-clustered inference must contain exactly '
			f'{expected_count} paired cells; got {len(rows)}'
		)
	indexed: dict[tuple[str, str], float] = {}
	for row in rows:
		key = (str(row['layout_id']), str(row['data_size']))
		if key in indexed:
			raise ValueError(
				'overall layout-clustered inference contains duplicate cell '
				f'{key[0]}/{key[1]}'
			)
		indexed[key] = float(row['candidate_minus_control'])
	layout_mean_deltas: dict[str, float] = {}
	for layout_id in LAYOUT_IDS:
		missing = [
			data_size
			for data_size in DATA_SIZES
			if (layout_id, data_size) not in indexed
		]
		if missing:
			raise ValueError(
				'overall layout-clustered inference is missing cells for '
				f'{layout_id}: {missing}'
			)
		layout_mean_deltas[layout_id] = statistics.fmean(
			indexed[(layout_id, data_size)] for data_size in DATA_SIZES
		)
	values = [layout_mean_deltas[layout_id] for layout_id in LAYOUT_IDS]
	statistics_payload = _delta_statistics(values)
	statistics_payload['paired_t_statistic'] = _paired_t_statistic(values)
	return {
		'n': len(values),
		'statistical_unit': 'layout_id',
		'within_layout_reduction': 'mean_across_data_sizes',
		'data_sizes_per_layout': len(DATA_SIZES),
		'layout_mean_deltas': layout_mean_deltas,
		'candidate_minus_control': statistics_payload,
	}


def _delta_statistics(values: Sequence[float]) -> dict[str, object]:
	if not values:
		raise ValueError('delta statistics require at least one value')
	mean = statistics.fmean(values)
	sample_standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
	return {
		'n': len(values),
		'mean': mean,
		'median': statistics.median(values),
		'sample_standard_deviation': sample_standard_deviation,
		'wins': sum(value > 0.0 for value in values),
		'ties': sum(value == 0.0 for value in values),
		'losses': sum(value < 0.0 for value in values),
	}


def _paired_t_statistic(values: Sequence[float]) -> float | None:
	if not values:
		raise ValueError('paired t statistic requires at least one value')
	if len(values) == 1:
		return None
	standard_error = statistics.stdev(values) / math.sqrt(len(values))
	return statistics.fmean(values) / standard_error if standard_error > 0.0 else None


def _model_means(
	config: F3PairedCandidateSummaryConfig,
	aggregates: Mapping[str, object],
) -> dict[str, object]:
	by_size = aggregates['by_size']
	all_15 = aggregates['all_15']
	if not isinstance(by_size, Mapping) or not isinstance(all_15, Mapping):
		raise TypeError('paired model-mean aggregates must be mappings')
	return {
		'by_size': {
			data_size: _scope_model_means(config, by_size[data_size])
			for data_size in DATA_SIZES
		},
		'all_15': _scope_model_means(config, all_15),
	}


def _scope_model_means(
	config: F3PairedCandidateSummaryConfig,
	scope: object,
) -> dict[str, float]:
	if not isinstance(scope, Mapping):
		raise TypeError('paired model-mean scope must be a mapping')
	return {
		config.candidate.model_id: float(scope['candidate_mean']),
		config.control.model_id: float(scope['control_mean']),
	}


def _comparison_csv(rows: Sequence[Mapping[str, object]]) -> str:
	buffer = io.StringIO()
	writer = csv.DictWriter(
		buffer, fieldnames=COMPARISON_FIELDNAMES, lineterminator='\n'
	)
	writer.writeheader()
	for row in rows:
		formatted = dict(row)
		for key in (
			'candidate_macro_f1',
			'control_macro_f1',
			'candidate_minus_control',
		):
			formatted[key] = f'{float(row[key]):.9f}'
		writer.writerow({key: formatted[key] for key in COMPARISON_FIELDNAMES})
	return buffer.getvalue()


def _summary_markdown(
	config: F3PairedCandidateSummaryConfig,
	rows: Sequence[Mapping[str, object]],
	aggregates: Mapping[str, object],
) -> str:
	by_size = aggregates['by_size']
	all_15 = aggregates['all_15']
	overall = aggregates['overall_layout_clustered']
	if (
		not isinstance(by_size, Mapping)
		or not isinstance(all_15, Mapping)
		or not isinstance(overall, Mapping)
	):
		raise TypeError('paired summary aggregates must be mappings')
	lines = [
		'# F3 lithology paired candidate summary',
		'',
		(
			f'Candidate `{config.candidate.model_id}` versus control '
			f'`{config.control.model_id}`.'
		),
		'',
		(
			f'Primary metric: `{PRIMARY_METRIC}` on unique validation voxels. '
			'Positive deltas favor the candidate.'
		),
		'',
		(
			'Per-size paired t statistics use the five `layout_id` values as '
			'independent units. `all_15` describes the 15 layout-by-size cells '
			'and has no inferential test.'
		),
		'',
		'## Model means and paired deltas',
		'',
		(
			'| scope | descriptive unit | n | candidate mean | control mean | '
			'delta mean | median | sample std | wins/ties/losses | paired t unit | '
			'paired t |'
		),
		'|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|',
	]
	for scope in (*DATA_SIZES, 'all_15'):
		scope_row = all_15 if scope == 'all_15' else by_size[scope]
		if not isinstance(scope_row, Mapping):
			raise TypeError(f'{scope} aggregate must be a mapping')
		stats = scope_row['candidate_minus_control']
		if not isinstance(stats, Mapping):
			raise TypeError(f'{scope} paired statistics must be a mapping')
		paired_t = stats.get('paired_t_statistic')
		rendered_t = 'n/a' if paired_t is None else f'{float(paired_t):.4f}'
		paired_t_unit = scope_row['paired_t_statistical_unit']
		rendered_unit = (
			'descriptive only'
			if paired_t_unit is None
			else f'{paired_t_unit} (n={stats["n"]})'
		)
		lines.append(
			f'| {scope} | {scope_row["descriptive_unit"]} | '
			f'{scope_row["n"]} | '
			f'{float(scope_row["candidate_mean"]):.6f} | '
			f'{float(scope_row["control_mean"]):.6f} | '
			f'{float(stats["mean"]):.6f} | {float(stats["median"]):.6f} | '
			f'{float(stats["sample_standard_deviation"]):.6f} | '
			f'{stats["wins"]}/{stats["ties"]}/{stats["losses"]} | '
			f'{rendered_unit} | {rendered_t} |'
		)
	overall_stats = overall['candidate_minus_control']
	if not isinstance(overall_stats, Mapping):
		raise TypeError('overall layout-clustered statistics must be a mapping')
	overall_t = overall_stats['paired_t_statistic']
	rendered_overall_t = 'n/a' if overall_t is None else f'{float(overall_t):.4f}'
	lines.extend(
		[
			'',
			'## Overall layout-clustered inference',
			'',
			(
				'Each layout delta is the mean of its three data-size deltas. The '
				'paired t statistic then uses the five `layout_id` means as its '
				'statistical units.'
			),
			'',
			(
				'| statistical unit | n | sizes per layout | delta mean | '
				'sample std | paired t |'
			),
			'|---|---:|---:|---:|---:|---:|',
			(
				f'| {overall["statistical_unit"]} | {overall["n"]} | '
				f'{overall["data_sizes_per_layout"]} | '
				f'{float(overall_stats["mean"]):.6f} | '
				f'{float(overall_stats["sample_standard_deviation"]):.6f} | '
				f'{rendered_overall_t} |'
			),
			'',
			'| layout | mean candidate-control delta across sizes |',
			'|---|---:|',
		]
	)
	layout_mean_deltas = overall['layout_mean_deltas']
	if not isinstance(layout_mean_deltas, Mapping):
		raise TypeError('overall layout mean deltas must be a mapping')
	lines.extend(
		f'| {layout_id} | {float(layout_mean_deltas[layout_id]):.6f} |'
		for layout_id in LAYOUT_IDS
	)
	lines.extend(
		[
			'',
			'## Per-cell macro F1',
			'',
			'| size | layout | candidate | control | candidate-control |',
			'|---|---|---:|---:|---:|',
		]
	)
	lines.extend(
		(
			f'| {row["data_size"]} | {row["layout_id"]} | '
			f'{float(row["candidate_macro_f1"]):.6f} | '
			f'{float(row["control_macro_f1"]):.6f} | '
			f'{float(row["candidate_minus_control"]):.6f} |'
		)
		for row in rows
	)
	lines.append('')
	return '\n'.join(lines)


def _write_atomic_summary(
	root: Path,
	contents: Mapping[str, str],
) -> tuple[Path, Path, Path]:
	if set(contents) != set(OUTPUT_NAMES):
		raise ValueError('paired summary contents must define the exact output set')
	root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(tempfile.mkdtemp(prefix=f'.{root.name}.staging-', dir=root.parent))
	try:
		for name in OUTPUT_NAMES:
			(staging / name).write_text(contents[name], encoding='utf-8')
		_validate_summary_file_set(staging)
		_ensure_summary_absent(root)
		staging.replace(root)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	return (
		root / OUTPUT_NAMES[0],
		root / OUTPUT_NAMES[1],
		root / OUTPUT_NAMES[2],
	)


def _validate_summary_file_set(root: Path) -> None:
	if {path.name for path in root.iterdir()} != set(OUTPUT_NAMES):
		raise RuntimeError('paired summary staging contains unexpected files')


def _ensure_summary_absent(root: Path) -> None:
	if root.exists():
		raise FileExistsError(f'refusing to overwrite summary directory: {root}')


def _model_from_mapping(
	value: Mapping[str, object],
	*,
	label: str,
) -> F3PairedCandidateModel:
	_require_exact_keys(
		value, {'id', 'checkpoint', 'embeddings_dir', 'expected'}, label
	)
	model_id = _nonempty_string(value.get('id'), f'{label}.id')
	if Path(model_id).name != model_id or model_id in {'.', '..'}:
		raise ValueError(f'{label}.id must be one path segment')
	return F3PairedCandidateModel(
		model_id=model_id,
		checkpoint=_absolute_path(value.get('checkpoint'), f'{label}.checkpoint'),
		embeddings_dir=_absolute_path(
			value.get('embeddings_dir'), f'{label}.embeddings_dir'
		),
		expected=_expected_source_from_mapping(
			_required_mapping(value, 'expected', label),
			label=f'{label}.expected',
		),
	)


def _expected_source_from_mapping(
	value: Mapping[str, object],
	*,
	label: str,
) -> F3PairedCandidateExpectedSource:
	_require_exact_keys(
		value,
		{
			'base_training_config',
			'stratigraphy_training_config',
			'epoch',
			'global_step',
		},
		label,
	)
	stratigraphy_value = value.get('stratigraphy_training_config')
	stratigraphy_config = (
		None
		if stratigraphy_value is None
		else _absolute_path(
			stratigraphy_value,
			f'{label}.stratigraphy_training_config',
		)
	)
	return F3PairedCandidateExpectedSource(
		base_training_config=_absolute_path(
			value.get('base_training_config'), f'{label}.base_training_config'
		),
		stratigraphy_training_config=stratigraphy_config,
		epoch=_positive_integer(value.get('epoch'), f'{label}.epoch'),
		global_step=_positive_integer(value.get('global_step'), f'{label}.global_step'),
	)


def _required_mapping(
	value: Mapping[str, object],
	key: str,
	label: str,
) -> Mapping[str, object]:
	item = value.get(key)
	if not isinstance(item, Mapping):
		raise TypeError(f'{label}.{key} must be a mapping')
	return item


def _require_exact_keys(
	value: Mapping[str, object],
	expected: set[str],
	label: str,
) -> None:
	keys = set(value)
	if keys != expected:
		raise ValueError(
			f'{label} keys must be exactly {sorted(expected)!r}; '
			f'missing={sorted(expected - keys)!r}, '
			f'unexpected={sorted(str(key) for key in keys - expected)!r}'
		)


def _nonempty_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise ValueError(f'{label} must be a non-empty string')
	return value


def _positive_integer(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
		raise ValueError(f'{label} must be a positive integer')
	return value


def _required_sha256(value: object, *, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256')
	return value


def _absolute_path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise ValueError(f'{label} must be a non-empty path')
	path = Path(value)
	if not path.is_absolute():
		raise ValueError(f'{label} must be absolute')
	return path


def _paths_overlap(first: Path, second: Path) -> bool:
	first = first.resolve(strict=False)
	second = second.resolve(strict=False)
	return (
		first == second or first.is_relative_to(second) or second.is_relative_to(first)
	)


__all__ = [
	'COMPARISON_FIELDNAMES',
	'OUTPUT_NAMES',
	'PRIMARY_METRIC',
	'F3PairedCandidateExpectedSource',
	'F3PairedCandidateModel',
	'F3PairedCandidateSummaryConfig',
	'f3_paired_candidate_summary_config_from_mapping',
	'inspect_f3_paired_candidate_results',
	'summarize_f3_paired_candidate_results',
]
