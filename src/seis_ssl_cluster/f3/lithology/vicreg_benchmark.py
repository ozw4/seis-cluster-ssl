"""VICReg screening and read-only seven-way F3 benchmark extension."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_MODEL_IDS,
	F3FiveWayConfig,
	F3FiveWayModelSource,
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	FIXED_DECODER_CONTRACT,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	F3LithologyCandidateConfig,
	audit_f3_lithology_candidate_source,
)
from seis_ssl_cluster.f3.lithology.five_way_results import (
	EXPECTED_AGGREGATION_UNIT,
	PAIRED_COMPARISONS,
	SUMMARY_METRICS,
	inspect_f3_lithology_five_way_results,
	read_f3_lithology_job_evidence,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	FIVE_WAY_TILE_SETTINGS,
	F3FiveWayJob,
	inspect_f3_lithology_five_way_job,
	resolve_f3_lithology_five_way_job,
	run_f3_lithology_frozen_encoder_job,
)
from seis_ssl_cluster.f3.lithology.five_way_sources import (
	audit_f3_lithology_five_way_sources,
)
from seis_ssl_cluster.stratigraphy import discover_pseudo_target_inputs
from seis_ssl_cluster.training.checkpoint import load_checkpoint

SCREENING_MODEL_IDS = ('local_vicreg_100', 'random')
EXTENSION_MODEL_IDS = ('local_vicreg', 'local_vicreg_hmm_k6')
SEVEN_WAY_MODEL_IDS = (*FIVE_WAY_MODEL_IDS, *EXTENSION_MODEL_IDS)
SCREENING_DATA_SIZE = 'medium'
VICREG_METHOD = 'local_vicreg_3d'
VICREG_CHECKPOINT_KIND = 'vicreg_pretraining'
VICREG_LOCAL_PAIRS_PER_CROP = 128
VICREG_STAGE1_EPOCHS = 100
VICREG_STAGE1_GLOBAL_STEPS = 62_500
VICREG_STAGE2_EPOCHS = 25
VICREG_STAGE2_GLOBAL_STEPS = 15_625
VICREG_HMM_K = 6
VICREG_HMM_TARGET_SUFFIX = (
	'pseudo_targets',
	'f3',
	'facies_benchmark_v1',
	'local_vicreg_v1',
	'vicreg100',
)
VICREG_MODEL_CONTRACT: Mapping[str, object] = {
	'patch_size': [8, 8, 8],
	'encoder_dim': 384,
	'encoder_depth': 8,
	'encoder_heads': 6,
	'decoder_dim': 256,
	'decoder_depth': 4,
	'decoder_heads': 4,
}
VICREG_OBJECTIVE_CONTRACT: Mapping[str, object] = {
	'method': VICREG_METHOD,
	'local_pairs_per_crop': VICREG_LOCAL_PAIRS_PER_CROP,
	'projector_dim': 384,
	'invariance_weight': 25.0,
	'variance_weight': 25.0,
	'covariance_weight': 1.0,
	'variance_target_std': 1.0,
	'variance_eps': 1.0e-4,
}
VICREG_AUGMENTATION_CONTRACT: Mapping[str, object] = {
	'horizontal_flip_probability': 0.5,
}
VICREG_GATE_PASS = 'VICREG_BASELINE_GATE_PASS'  # noqa: S105
VICREG_GATE_FAIL = 'VICREG_BASELINE_GATE_FAIL'
PRIMARY_METRIC = 'macro_f1'

COMPARISON_CSV_NAME = 'comparison.csv'
PAIRED_DELTAS_CSV_NAME = 'paired_deltas.csv'
SUMMARY_BY_SIZE_CSV_NAME = 'summary_by_size.csv'
SUMMARY_JSON_NAME = 'summary.json'
SUMMARY_MD_NAME = 'summary.md'
SCREENING_SUMMARY_OUTPUT_NAMES = (
	COMPARISON_CSV_NAME,
	PAIRED_DELTAS_CSV_NAME,
	SUMMARY_JSON_NAME,
	SUMMARY_MD_NAME,
)
BENCHMARK_SUMMARY_OUTPUT_NAMES = (
	COMPARISON_CSV_NAME,
	PAIRED_DELTAS_CSV_NAME,
	SUMMARY_BY_SIZE_CSV_NAME,
	SUMMARY_JSON_NAME,
	SUMMARY_MD_NAME,
)

EXTENSION_PAIRED_COMPARISONS = (
	('local_vicreg_hmm_k6_minus_local_vicreg', 'local_vicreg_hmm_k6', 'local_vicreg'),
	('local_vicreg_minus_random', 'local_vicreg', 'random'),
	('local_vicreg_hmm_k6_minus_random', 'local_vicreg_hmm_k6', 'random'),
)
SEVEN_WAY_PAIRED_COMPARISONS = (
	*PAIRED_COMPARISONS,
	*EXTENSION_PAIRED_COMPARISONS,
	(
		'local_vicreg_minus_local_barlow_twins',
		'local_vicreg',
		'local_barlow_twins',
	),
	(
		'local_vicreg_hmm_k6_minus_local_barlow_twins_hmm_k6',
		'local_vicreg_hmm_k6',
		'local_barlow_twins_hmm_k6',
	),
	('local_vicreg_minus_mae', 'local_vicreg', 'mae'),
	('local_vicreg_hmm_k6_minus_mae_hmm_k6', 'local_vicreg_hmm_k6', 'mae_hmm_k6'),
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
	'decoder_initial_state_sha256',
	'supervision_identity',
	'validation_identity',
	'macro_f1',
	'mean_iou',
	'balanced_accuracy',
	'weighted_f1',
	'validation_voxel_count',
	'metrics_path',
	'metrics_sha256',
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


@dataclass(frozen=True)
class F3VICRegOutputRoots:
	"""Disjoint run, log, and summary roots for one benchmark suite."""

	runs_root: Path
	job_logs_root: Path
	summary_root: Path


@dataclass(frozen=True)
class F3VICRegExtensionConfig:
	"""Resolved VICReg screening and two-arm extension configuration."""

	canonical_config: Path
	screening_model: F3FiveWayModelSource
	extension_models: tuple[F3FiveWayModelSource, ...]
	screening_outputs: F3VICRegOutputRoots
	extension_outputs: F3VICRegOutputRoots
	combined_summary_root: Path

	def extension_model_by_id(self, model_id: str) -> F3FiveWayModelSource:
		"""Return one of the exact two extension sources."""
		for model in self.extension_models:
			if model.model_id == model_id:
				return model
		raise ValueError(
			f'unknown VICReg extension model: {model_id!r}; '
			f'expected one of {list(EXTENSION_MODEL_IDS)!r}'
		)


def f3_vicreg_extension_config_from_mapping(
	config: Mapping[str, object],
) -> F3VICRegExtensionConfig:
	"""Resolve the exact one-screening-source/two-extension-source schema."""
	_require_exact_keys(config, {'benchmark', 'screening', 'extension'}, 'config')
	benchmark = _required_mapping(config, 'benchmark')
	screening = _required_mapping(config, 'screening')
	extension = _required_mapping(config, 'extension')
	_require_exact_keys(benchmark, {'canonical_config'}, 'benchmark')
	_require_exact_keys(screening, {'model', 'outputs'}, 'screening')
	_require_exact_keys(extension, {'models', 'outputs'}, 'extension')

	screening_model = _resolve_source(
		screening.get('model'),
		expected_id=SCREENING_MODEL_IDS[0],
		label='screening.model',
	)
	extension_models = _resolve_extension_sources(extension.get('models'))
	screening_outputs = _resolve_output_roots(
		screening.get('outputs'), label='screening.outputs'
	)
	extension_outputs, combined_summary_root = _resolve_extension_output_roots(
		extension.get('outputs')
	)
	all_output_roots = (
		screening_outputs.runs_root,
		screening_outputs.job_logs_root,
		screening_outputs.summary_root,
		extension_outputs.runs_root,
		extension_outputs.job_logs_root,
		extension_outputs.summary_root,
		combined_summary_root,
	)
	_validate_disjoint_roots(all_output_roots, label='VICReg benchmark outputs')
	return F3VICRegExtensionConfig(
		canonical_config=_required_absolute_path(
			benchmark, 'canonical_config', prefix='benchmark'
		),
		screening_model=screening_model,
		extension_models=extension_models,
		screening_outputs=screening_outputs,
		extension_outputs=extension_outputs,
		combined_summary_root=combined_summary_root,
	)


def load_f3_vicreg_canonical_config(
	config: F3VICRegExtensionConfig,
) -> F3FiveWayConfig:
	"""Load the existing exact-five configuration and enforce read-only separation."""
	if not config.canonical_config.is_file():
		raise FileNotFoundError(
			f'canonical five-way config does not exist: {config.canonical_config}'
		)
	canonical = f3_lithology_five_way_config_from_mapping(
		load_config(config.canonical_config)
	)
	for output in (
		config.screening_outputs.runs_root,
		config.screening_outputs.job_logs_root,
		config.screening_outputs.summary_root,
		config.extension_outputs.runs_root,
		config.extension_outputs.job_logs_root,
		config.extension_outputs.summary_root,
		config.combined_summary_root,
	):
		for canonical_root in (canonical.runs_root, canonical.summary_root):
			if _paths_overlap(output, canonical_root):
				raise ValueError(
					'VICReg output overlaps read-only canonical five-way output: '
					f'{output} and {canonical_root}'
				)
	return canonical


def plan_f3_vicreg_screening_jobs() -> tuple[tuple[str, str, str], ...]:
	"""Enumerate layout then model for the exact 2x5 medium screening matrix."""
	return tuple(
		(model_id, layout_id, SCREENING_DATA_SIZE)
		for layout_id in LAYOUT_IDS
		for model_id in SCREENING_MODEL_IDS
	)


def plan_f3_vicreg_extension_jobs() -> tuple[tuple[str, str, str], ...]:
	"""Enumerate size then layout then model for the exact 30-job extension."""
	return tuple(
		(model_id, layout_id, data_size)
		for data_size in DATA_SIZES
		for layout_id in LAYOUT_IDS
		for model_id in EXTENSION_MODEL_IDS
	)


def resolve_f3_vicreg_screening_job(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
	*,
	model: str,
	layout: str,
	size: str,
) -> F3FiveWayJob:
	"""Resolve one medium screening cell, reusing canonical random read-only."""
	if model not in SCREENING_MODEL_IDS:
		raise ValueError(
			f'unknown screening model: {model!r}; '
			f'expected one of {list(SCREENING_MODEL_IDS)!r}'
		)
	if size != SCREENING_DATA_SIZE:
		raise ValueError(
			f'VICReg screening only accepts size={SCREENING_DATA_SIZE!r}; got {size!r}'
		)
	_validate_layout(layout)
	if model == 'random':
		return resolve_f3_lithology_five_way_job(
			canonical, model='random', layout=layout, size=size
		)
	return _extension_job(
		canonical,
		model=config.screening_model,
		layout=layout,
		size=size,
		runs_root=config.screening_outputs.runs_root,
	)


def resolve_f3_vicreg_extension_job(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
	*,
	model: str,
	layout: str,
	size: str,
) -> F3FiveWayJob:
	"""Resolve one of the exact two-arm, all-size extension cells."""
	_validate_layout(layout)
	_validate_size(size)
	return _extension_job(
		canonical,
		model=config.extension_model_by_id(model),
		layout=layout,
		size=size,
		runs_root=config.extension_outputs.runs_root,
	)


def audit_f3_vicreg_screening_source(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit only the stage-1 screening source and canonical random baseline."""
	return _audit_f3_vicreg_sources(
		config,
		canonical,
		specs=((config.screening_model, 'screening', config.screening_outputs),),
	)


def audit_f3_vicreg_sources(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit canonical sources and all three VICReg checkpoint/embedding lineages."""
	return _audit_f3_vicreg_sources(
		config,
		canonical,
		specs=(
			(config.screening_model, 'screening', config.screening_outputs),
			(config.extension_models[0], 'control', config.extension_outputs),
			(config.extension_models[1], 'hmm', config.extension_outputs),
		),
	)


def _audit_f3_vicreg_sources(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
	*,
	specs: Sequence[tuple[F3FiveWayModelSource, str, F3VICRegOutputRoots]],
) -> dict[str, object]:
	canonical_report = audit_f3_lithology_five_way_sources(canonical)
	canonical_sources = _canonical_source_provenance(canonical)
	sources: list[dict[str, object]] = []
	canonical_random: object | None = None
	for source, role, outputs in specs:
		candidate = F3LithologyCandidateConfig(
			canonical_config=config.canonical_config,
			candidate_id=source.model_id,
			checkpoint=source.checkpoint,
			embeddings_dir=source.embeddings_dir,
			runs_root=outputs.runs_root,
			summary_root=outputs.summary_root,
		)
		provenance = audit_f3_lithology_candidate_source(candidate, canonical)
		lineage = _validate_vicreg_checkpoint(source.checkpoint, role=role)
		_validate_vicreg_embedding_metadata(
			Path(str(provenance['embedding_metadata_path'])),
			checkpoint=source.checkpoint,
			role=role,
		)
		current_random = provenance['canonical_random']
		if canonical_random is None:
			canonical_random = current_random
		elif canonical_random != current_random:
			raise ValueError('canonical random provenance changed during VICReg audit')
		sources.append({**provenance, 'role': role, 'lineage': lineage})
	_validate_shared_vicreg100_lineage(sources)
	return {
		'model_order': [source.model_id for source, _role, _outputs in specs],
		'canonical_model_order': list(canonical.model_ids),
		'canonical_source_audit': canonical_report,
		'canonical_sources': canonical_sources,
		'sources': sources,
		'canonical_random': canonical_random,
	}


def _canonical_source_provenance(
	canonical: F3FiveWayConfig,
) -> list[dict[str, object]]:
	survey_id = canonical.dataset['name']
	provenance = []
	for model in canonical.models:
		files = output_paths(model.embeddings_dir, survey_id)
		metadata = _read_json(
			files.metadata, label=f'{model.model_id} embedding metadata'
		)
		provenance.append(
			{
				'model_id': model.model_id,
				'checkpoint_sha256': file_sha256(model.checkpoint),
				'embeddings_sha256': file_sha256(files.embeddings),
				'embedding_metadata_sha256': file_sha256(files.metadata),
				'valid_tokens_sha256': file_sha256(files.valid_tokens),
				'recorded_checkpoint_sha256': metadata.get('checkpoint_sha256'),
			}
		)
	return provenance


def _validate_shared_vicreg100_lineage(
	sources: Sequence[Mapping[str, object]],
) -> None:
	by_role = {str(source.get('role')): source for source in sources}
	if set(by_role) != {'screening', 'control', 'hmm'}:
		return
	screening = by_role['screening']
	screening_path = Path(str(screening['checkpoint_path'])).resolve(strict=False)
	screening_sha256 = screening.get('checkpoint_sha256')
	for role in ('control', 'hmm'):
		lineage = by_role[role].get('lineage')
		if not isinstance(lineage, Mapping):
			raise TypeError(f'{role} VICReg lineage must be a mapping')
		source_path = Path(str(lineage.get('source_checkpoint'))).resolve(
			strict=False
		)
		if source_path != screening_path:
			raise ValueError(
				f'{role} lineage must use the screened VICReg100 checkpoint path'
			)
		if lineage.get('source_checkpoint_sha256') != screening_sha256:
			raise ValueError(
				f'{role} lineage must use the screened VICReg100 checkpoint SHA'
			)


def assert_f3_vicreg_full_benchmark_ready(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Require a passing screen and complete, audited two-arm sources."""
	try:
		return _full_benchmark_readiness(config, canonical)
	except (FileNotFoundError, TypeError, ValueError) as error:
		raise RuntimeError(f'FULL_BENCHMARK_BLOCKED: {error}') from error


def _full_benchmark_readiness(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	summary_path = config.screening_outputs.summary_root / SUMMARY_JSON_NAME
	summary = _read_json(summary_path, label='VICReg screening summary')
	gate = _screening_gate_status(summary, path=summary_path)
	_validate_passing_gate(gate)
	sources = audit_f3_vicreg_sources(config, canonical)
	rows, _paired, recomputed_gate = _screening_material(config, canonical)
	if recomputed_gate['gate_status'] != gate:
		raise ValueError(
			'screening summary gate does not match current ten-cell evidence'
		)
	recorded_evidence = summary.get('evidence_sha256')
	current_evidence = _screening_evidence_sha256(rows)
	if recorded_evidence != current_evidence:
		raise ValueError(
			'screening summary is not bound to the current source/job evidence'
		)
	return sources


def read_f3_vicreg_screening_gate(config: F3VICRegExtensionConfig) -> str:
	"""Read the exact Task-05 gate status from the screening summary."""
	path = config.screening_outputs.summary_root / SUMMARY_JSON_NAME
	payload = _read_json(path, label='VICReg screening summary')
	return _screening_gate_status(payload, path=path)


def _screening_gate_status(payload: Mapping[str, object], *, path: Path) -> str:
	gate = payload.get('gate_status')
	if gate not in {VICREG_GATE_PASS, VICREG_GATE_FAIL}:
		raise ValueError(
			f'{path} gate_status must be {VICREG_GATE_PASS!r} or '
			f'{VICREG_GATE_FAIL!r}; got {gate!r}'
		)
	return str(gate)


def _validate_passing_gate(gate: str) -> None:
	if gate != VICREG_GATE_PASS:
		raise ValueError(f'screening gate is {gate!r}')


def inspect_f3_vicreg_job(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
	job: F3FiveWayJob,
	*,
	suite: str,
) -> dict[str, object]:
	"""Return a source-audited, no-write plan for one screening/extension cell."""
	if suite == 'extension':
		audit = assert_f3_vicreg_full_benchmark_ready(config, canonical)
	elif suite == 'screening':
		audit = audit_f3_vicreg_screening_source(config, canonical)
	else:
		raise ValueError("suite must be 'screening' or 'extension'")
	return {
		**inspect_f3_lithology_five_way_job(job),
		'suite': suite,
		'source_audit_models': audit['model_order'],
		'reuses_canonical_random': (
			suite == 'screening' and job.model.model_id == 'random'
		),
	}


def run_f3_vicreg_job(  # noqa: PLR0913
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
	job: F3FiveWayJob,
	*,
	suite: str,
	device: str = 'auto',
	max_steps: int | None = None,
	resume: Path | None = None,
) -> dict[str, object]:
	"""Run or identity-skip one audited VICReg benchmark cell."""
	if suite == 'extension':
		report = assert_f3_vicreg_full_benchmark_ready(config, canonical)
		allowed = EXTENSION_MODEL_IDS
	elif suite == 'screening':
		report = audit_f3_vicreg_screening_source(config, canonical)
		allowed = SCREENING_MODEL_IDS
	else:
		raise ValueError("suite must be 'screening' or 'extension'")
	if job.model.model_id not in allowed:
		raise ValueError(
			f'{suite} job model must be one of {list(allowed)!r}; '
			f'got {job.model.model_id!r}'
		)
	if job.metrics_path.is_file():
		row = _read_job_row(canonical, job.model, job)
		_assert_row_current_source(row, _source_provenance(report, job.model.model_id))
		return {
			'completed': True,
			'skipped': True,
			'metrics_path': str(job.metrics_path),
			'metrics_sha256': row['metrics_sha256'],
		}
	if suite == 'screening' and job.model.model_id == 'random':
		raise FileNotFoundError(
			'canonical random screening result is missing; run and audit the '
			f'canonical five-way job first: {job.metrics_path}'
		)
	return {
		**run_f3_lithology_frozen_encoder_job(
			job,
			device=device,
			max_steps=max_steps,
			resume=resume,
		),
		'skipped': False,
	}


def inspect_f3_vicreg_screening_results(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit the exact ten logical medium cells without writing summaries."""
	rows, paired, gate = _screening_material(config, canonical)
	return {
		'complete_jobs': len(rows),
		'paired_layouts': len(paired),
		'model_order': list(SCREENING_MODEL_IDS),
		'data_size': SCREENING_DATA_SIZE,
		'gate_status': gate['gate_status'],
	}


def summarize_f3_vicreg_screening(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Atomically publish raw values, paired deltas, and the baseline gate."""
	rows, paired, gate = _screening_material(config, canonical)
	payload = {
		'schema_version': 1,
		'suite': 'local_vicreg_screen_v1',
		'models': list(SCREENING_MODEL_IDS),
		'data_size': SCREENING_DATA_SIZE,
		'primary_metric': PRIMARY_METRIC,
		'aggregation_unit': EXPECTED_AGGREGATION_UNIT,
		'statistical_unit': 'layout_id',
		'job_count': len(rows),
		'paired_layout_count': len(paired),
		'evidence_sha256': _screening_evidence_sha256(rows),
		'layouts': _screening_layout_evidence(rows, paired),
		**gate,
	}
	outputs = {
		COMPARISON_CSV_NAME: _csv_text(COMPARISON_FIELDNAMES, rows),
		PAIRED_DELTAS_CSV_NAME: _csv_text(PAIRED_FIELDNAMES, paired),
		SUMMARY_JSON_NAME: json.dumps(payload, indent=2, sort_keys=True) + '\n',
		SUMMARY_MD_NAME: _screening_markdown(payload, paired),
	}
	_atomic_summary(config.screening_outputs.summary_root, outputs)
	return {
		'complete_jobs': len(rows),
		'gate_status': gate['gate_status'],
		'summary_root': str(config.screening_outputs.summary_root),
		'outputs': [
			str(config.screening_outputs.summary_root / name)
			for name in SCREENING_SUMMARY_OUTPUT_NAMES
		],
	}


def inspect_f3_vicreg_extension_results(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Audit all 30 new jobs and the canonical random comparison rows read-only."""
	rows, paired, _by_size = _extension_material(config, canonical)
	return {
		'complete_jobs': len(rows),
		'model_order': list(EXTENSION_MODEL_IDS),
		'paired_delta_rows': len(paired),
	}


def summarize_f3_vicreg_extension(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Atomically summarize only the 30-job two-arm extension."""
	rows, paired, by_size = _extension_material(config, canonical)
	payload = _benchmark_summary_payload(
		suite='local_vicreg_extension_v1',
		models=EXTENSION_MODEL_IDS,
		job_count=len(rows),
		by_size=by_size,
	)
	outputs = _benchmark_outputs(
		rows=rows,
		paired=paired,
		by_size=by_size,
		payload=payload,
		title='F3 lithology Local VICReg extension summary',
	)
	_atomic_summary(config.extension_outputs.summary_root, outputs)
	return {
		'complete_jobs': len(rows),
		'summary_root': str(config.extension_outputs.summary_root),
		'outputs': [
			str(config.extension_outputs.summary_root / name)
			for name in BENCHMARK_SUMMARY_OUTPUT_NAMES
		],
	}


def inspect_f3_vicreg_combined_results(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Read and identity-check the existing 75 plus new 30 jobs."""
	rows, paired, _by_size = _combined_material(config, canonical)
	return {
		'complete_jobs': len(rows),
		'existing_jobs': 75,
		'extension_jobs': 30,
		'model_order': list(SEVEN_WAY_MODEL_IDS),
		'paired_delta_rows': len(paired),
	}


def summarize_f3_vicreg_combined(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> dict[str, object]:
	"""Publish a separate 105-row report without mutating the existing 75 jobs."""
	rows, paired, by_size = _combined_material(config, canonical)
	payload = {
		**_benchmark_summary_payload(
			suite='f3_lithology_seven_way_v1',
			models=SEVEN_WAY_MODEL_IDS,
			job_count=len(rows),
			by_size=by_size,
		),
		'existing_five_way_jobs': 75,
		'new_extension_jobs': 30,
		'canonical_runs_root': str(canonical.runs_root),
		'extension_runs_root': str(config.extension_outputs.runs_root),
	}
	outputs = _benchmark_outputs(
		rows=rows,
		paired=paired,
		by_size=by_size,
		payload=payload,
		title='F3 lithology seven-way combined summary',
	)
	_atomic_summary(config.combined_summary_root, outputs)
	return {
		'complete_jobs': len(rows),
		'existing_jobs': 75,
		'extension_jobs': 30,
		'summary_root': str(config.combined_summary_root),
		'outputs': [
			str(config.combined_summary_root / name)
			for name in BENCHMARK_SUMMARY_OUTPUT_NAMES
		],
	}


def _screening_material(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
	report = audit_f3_vicreg_screening_source(config, canonical)
	_reject_unexpected_run_directories(
		config.screening_outputs.runs_root,
		model_ids=(config.screening_model.model_id,),
		layouts=LAYOUT_IDS,
		sizes=(SCREENING_DATA_SIZE,),
	)
	rows = []
	for model_id, layout_id, size in plan_f3_vicreg_screening_jobs():
		job = resolve_f3_vicreg_screening_job(
			config, canonical, model=model_id, layout=layout_id, size=size
		)
		rows.append(_read_job_row(canonical, job.model, job))
	_validate_matrix(
		rows,
		models=SCREENING_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=(SCREENING_DATA_SIZE,),
		report=report,
	)
	paired = _paired_rows(
		rows,
		comparisons=(
			('local_vicreg_100_minus_random', 'local_vicreg_100', 'random'),
		),
		metrics=(PRIMARY_METRIC,),
	)
	deltas = [float(row['delta']) for row in paired]
	if len(deltas) != len(LAYOUT_IDS):
		raise ValueError('VICReg screening must contain exactly five paired layouts')
	mean = statistics.fmean(deltas)
	median = statistics.median(deltas)
	wins = sum(delta > 0.0 for delta in deltas)
	passed = mean > 0.0 and median > 0.0 and wins >= 3
	gate = {
		'gate_status': VICREG_GATE_PASS if passed else VICREG_GATE_FAIL,
		'gate': {
			'mean_paired_delta_gt': 0.0,
			'median_paired_delta_gt': 0.0,
			'minimum_wins': 3,
			'layout_count': len(LAYOUT_IDS),
		},
		'mean_paired_delta': mean,
		'median_paired_delta': median,
		'wins': wins,
		'losses': sum(delta < 0.0 for delta in deltas),
		'ties': sum(delta == 0.0 for delta in deltas),
	}
	return rows, paired, gate


def _screening_evidence_sha256(rows: Sequence[Mapping[str, object]]) -> str:
	fields = (
		'model_id',
		'layout_id',
		'data_size',
		'encoder_checkpoint_sha256',
		'embeddings_sha256',
		'embedding_metadata_sha256',
		'valid_tokens_sha256',
		'decoder_checkpoint_sha256',
		'decoder_initial_state_sha256',
		'supervision_identity',
		'validation_identity',
		'metrics_sha256',
	)
	identity = [
		{field: row[field] for field in fields}
		for row in sorted(
			rows,
			key=lambda row: (
				str(row['layout_id']),
				str(row['model_id']),
			),
		)
	]
	return hashlib.sha256(
		json.dumps(
			identity,
			sort_keys=True,
			separators=(',', ':'),
			allow_nan=False,
		).encode('utf-8')
	).hexdigest()


def _screening_layout_evidence(
	rows: Sequence[Mapping[str, object]],
	paired: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	by_cell = {
		(str(row['model_id']), str(row['layout_id'])): row for row in rows
	}
	paired_by_layout = {str(row['layout_id']): row for row in paired}
	result = []
	for layout_id in LAYOUT_IDS:
		vicreg = by_cell['local_vicreg_100', layout_id]
		random = by_cell['random', layout_id]
		delta = paired_by_layout[layout_id]
		result.append(
			{
				'layout_id': layout_id,
				'local_vicreg_macro_f1': vicreg['macro_f1'],
				'random_macro_f1': random['macro_f1'],
				'delta_local_vicreg_minus_random': delta['delta'],
				'local_vicreg_checkpoint_sha256': vicreg[
					'encoder_checkpoint_sha256'
				],
				'random_checkpoint_sha256': random['encoder_checkpoint_sha256'],
				'local_vicreg_embedding_sha256': vicreg['embeddings_sha256'],
				'random_embedding_sha256': random['embeddings_sha256'],
				'supervision_identity': vicreg['supervision_identity'],
				'validation_mask_sha256': vicreg['validation_identity'],
				'decoder_initial_state_sha256': vicreg[
					'decoder_initial_state_sha256'
				],
				'local_vicreg_metrics_sha256': vicreg['metrics_sha256'],
				'random_metrics_sha256': random['metrics_sha256'],
			}
		)
	return result


def _extension_material(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> tuple[
	list[dict[str, object]],
	list[dict[str, object]],
	list[dict[str, object]],
]:
	report = assert_f3_vicreg_full_benchmark_ready(config, canonical)
	_reject_unexpected_run_directories(
		config.extension_outputs.runs_root,
		model_ids=EXTENSION_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
	)
	rows = _extension_rows(config, canonical)
	_validate_matrix(
		rows,
		models=EXTENSION_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
		report=report,
	)
	random_rows = _canonical_rows(canonical, model_ids=('random',))
	_validate_matrix(
		random_rows,
		models=('random',),
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
		report=report,
	)
	paired_source = [*rows, *random_rows]
	_validate_shared_condition_identity(paired_source)
	paired = _paired_rows(
		paired_source,
		comparisons=EXTENSION_PAIRED_COMPARISONS,
		metrics=SUMMARY_METRICS,
	)
	by_size = _by_size_rows(paired, comparisons=EXTENSION_PAIRED_COMPARISONS)
	return rows, paired, by_size


def _combined_material(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> tuple[
	list[dict[str, object]],
	list[dict[str, object]],
	list[dict[str, object]],
]:
	report = assert_f3_vicreg_full_benchmark_ready(config, canonical)
	canonical_inspection = inspect_f3_lithology_five_way_results(canonical)
	if canonical_inspection.get('complete_jobs') != 75:
		raise ValueError('canonical five-way inspection did not return exactly 75 jobs')
	_reject_unexpected_run_directories(
		config.extension_outputs.runs_root,
		model_ids=EXTENSION_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
	)
	canonical_rows = _canonical_rows(canonical, model_ids=FIVE_WAY_MODEL_IDS)
	extension_rows = _extension_rows(config, canonical)
	rows = [*canonical_rows, *extension_rows]
	_validate_matrix(
		rows,
		models=SEVEN_WAY_MODEL_IDS,
		layouts=LAYOUT_IDS,
		sizes=DATA_SIZES,
		report=report,
	)
	paired = _paired_rows(
		rows,
		comparisons=SEVEN_WAY_PAIRED_COMPARISONS,
		metrics=SUMMARY_METRICS,
	)
	by_size = _by_size_rows(paired, comparisons=SEVEN_WAY_PAIRED_COMPARISONS)
	return rows, paired, by_size


def _canonical_rows(
	canonical: F3FiveWayConfig,
	*,
	model_ids: Sequence[str],
) -> list[dict[str, object]]:
	rows = []
	for model_id in model_ids:
		model = canonical.model_by_id(model_id)
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				job = resolve_f3_lithology_five_way_job(
					canonical,
					model=model_id,
					layout=layout_id,
					size=data_size,
				)
				rows.append(_read_job_row(canonical, model, job))
	return rows


def _extension_rows(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
) -> list[dict[str, object]]:
	rows = []
	for model_id, layout_id, data_size in plan_f3_vicreg_extension_jobs():
		job = resolve_f3_vicreg_extension_job(
			config,
			canonical,
			model=model_id,
			layout=layout_id,
			size=data_size,
		)
		rows.append(_read_job_row(canonical, job.model, job))
	return rows


def _read_job_row(
	canonical: F3FiveWayConfig,
	model: F3FiveWayModelSource,
	job: F3FiveWayJob,
) -> dict[str, object]:
	metrics = _read_json(job.metrics_path, label='completed evaluation metrics')
	for metric in SUMMARY_METRICS:
		value = metrics.get(metric)
		if (
			isinstance(value, bool)
			or not isinstance(value, int | float)
			or not math.isfinite(float(value))
		):
			raise ValueError(
				f'{model.model_id}/{job.layout_id}/{job.data_size} metric '
				f'{metric} must be finite numeric'
			)
	evidence = read_f3_lithology_job_evidence(
		canonical,
		model=model,
		layout_id=job.layout_id,
		data_size=job.data_size,
		job_dir=job.output_dir,
	)
	resolved_decoder_config = _read_json(
		job.decoder_dir / 'resolved_config.json', label='decoder resolved config'
	)
	_validate_decoder_contract(resolved_decoder_config, job=job)
	run_metadata = _read_json(
		job.decoder_dir / 'run_metadata.json', label='decoder run metadata'
	)
	initial_state = _sha256(
		run_metadata.get('initial_model_state_sha256'),
		label=(
			f'{model.model_id}/{job.layout_id}/{job.data_size} '
			'decoder initial state'
		),
	)
	return {
		'model_id': model.model_id,
		'layout_id': job.layout_id,
		'data_size': job.data_size,
		'checkpoint_path': str(model.checkpoint),
		'embeddings_dir': str(model.embeddings_dir),
		**evidence,
		'decoder_initial_state_sha256': initial_state,
		**{metric: float(metrics[metric]) for metric in SUMMARY_METRICS},
		'metrics_path': str(job.metrics_path),
		'metrics_sha256': file_sha256(job.metrics_path),
	}


def _validate_decoder_contract(
	resolved: Mapping[str, object], *, job: F3FiveWayJob
) -> None:
	decoder_expected = {
		key: FIXED_DECODER_CONTRACT[key]
		for key in (
			'spec',
			'embedding_dim',
			'class_count',
			'hidden_channels',
			'upsample_factors',
			'upsample_mode',
			'normalization',
		)
	}
	train_expected = {
		key: FIXED_DECODER_CONTRACT[key]
		for key in (
			'epochs',
			'batch_size',
			'learning_rate',
			'weight_decay',
			'class_weight',
			'seed',
			'amp',
			'gradient_clip_norm',
			'sampling_mode',
			'steps_per_epoch',
		)
	}
	_validate_resolved_mapping(
		resolved.get('decoder'), expected=decoder_expected, label='decoder'
	)
	_validate_resolved_mapping(
		resolved.get('tiles'), expected=FIVE_WAY_TILE_SETTINGS, label='tiles'
	)
	_validate_resolved_mapping(
		resolved.get('train'), expected=train_expected, label='train'
	)
	model = resolved.get('model')
	if not isinstance(model, Mapping) or model != {
		'tag': job.model.model_id,
		'freeze_encoder': True,
	}:
		raise ValueError('decoder resolved model identity does not match this job')


def _validate_resolved_mapping(
	value: object, *, expected: Mapping[str, object], label: str
) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'decoder resolved {label} must be a mapping')
	for key, expected_value in expected.items():
		normalized = _plain_json_value(expected_value)
		if value.get(key) != normalized:
			raise ValueError(
				f'decoder resolved {label}.{key} must equal {normalized!r}; '
				f'got {value.get(key)!r}'
			)


def _plain_json_value(value: object) -> object:
	if isinstance(value, tuple):
		return [_plain_json_value(item) for item in value]
	return value


def _validate_matrix(
	rows: list[dict[str, object]],
	*,
	models: Sequence[str],
	layouts: Sequence[str],
	sizes: Sequence[str],
	report: Mapping[str, object],
) -> None:
	expected = {
		(model_id, layout_id, data_size)
		for model_id in models
		for layout_id in layouts
		for data_size in sizes
	}
	actual = [
		(str(row['model_id']), str(row['layout_id']), str(row['data_size']))
		for row in rows
	]
	if len(actual) != len(set(actual)):
		raise ValueError('duplicate VICReg benchmark job evidence')
	if set(actual) != expected:
		missing = sorted(expected - set(actual))
		extra = sorted(set(actual) - expected)
		raise ValueError(
			'VICReg benchmark job matrix is not exact; '
			f'missing={missing!r}, extra={extra!r}'
		)
	for row in rows:
		_assert_row_current_source(
			row, _source_provenance(report, str(row['model_id']))
		)
	_validate_shared_condition_identity(rows)
	_validate_per_model_source_identity(rows)


def _validate_shared_condition_identity(rows: list[dict[str, object]]) -> None:
	by_condition: dict[tuple[str, str], list[dict[str, object]]] = {}
	for row in rows:
		key = (str(row['layout_id']), str(row['data_size']))
		by_condition.setdefault(key, []).append(row)
	for (layout_id, data_size), group in by_condition.items():
		for key in (
			'supervision_identity',
			'validation_identity',
			'validation_voxel_count',
			'_validation_tile_manifest_sha256',
			'decoder_initial_state_sha256',
		):
			values = {str(row[key]) for row in group}
			if len(values) != 1:
				raise ValueError(
					f'{layout_id}/{data_size} {key} differs between models'
				)
	valid_tokens = {str(row['valid_tokens_sha256']) for row in rows}
	if len(valid_tokens) != 1:
		raise ValueError('valid-token SHA must be shared by every compared model')


def _validate_per_model_source_identity(rows: list[dict[str, object]]) -> None:
	by_model: dict[str, list[dict[str, object]]] = {}
	for row in rows:
		by_model.setdefault(str(row['model_id']), []).append(row)
	for model_id, group in by_model.items():
		for key in (
			'encoder_checkpoint_sha256',
			'embeddings_sha256',
			'embedding_metadata_sha256',
			'valid_tokens_sha256',
		):
			values = {str(row[key]) for row in group}
			if len(values) != 1:
				raise ValueError(f'{model_id} {key} drifted between completed jobs')
	encoders: dict[str, str] = {}
	for model_id, group in by_model.items():
		sha256 = str(group[0]['encoder_checkpoint_sha256'])
		other = encoders.setdefault(sha256, model_id)
		if other != model_id:
			raise ValueError(
				'encoder checkpoint SHA must differ across compared models; '
				f'{other!r} and {model_id!r} match'
			)


def _source_provenance(
	report: Mapping[str, object], model_id: str
) -> Mapping[str, object]:
	if model_id == 'random':
		value = report.get('canonical_random')
		if not isinstance(value, Mapping):
			raise TypeError('VICReg source audit canonical_random must be a mapping')
		return value
	sources = report.get('sources')
	if not isinstance(sources, list):
		raise TypeError('VICReg source audit sources must be a list')
	for source in sources:
		if isinstance(source, Mapping) and source.get('candidate_id') == model_id:
			return source
	if model_id in FIVE_WAY_MODEL_IDS:
		canonical_sources = report.get('canonical_sources')
		if not isinstance(canonical_sources, list):
			raise TypeError('VICReg audit canonical_sources must be a list')
		for source in canonical_sources:
			if isinstance(source, Mapping) and source.get('model_id') == model_id:
				return source
		raise ValueError(f'VICReg audit has no canonical provenance for {model_id!r}')
	raise ValueError(f'VICReg source audit has no provenance for {model_id!r}')


def _assert_row_current_source(
	row: Mapping[str, object], provenance: Mapping[str, object]
) -> None:
	for row_key, source_key in (
		('encoder_checkpoint_sha256', 'checkpoint_sha256'),
		('embeddings_sha256', 'embeddings_sha256'),
		('embedding_metadata_sha256', 'embedding_metadata_sha256'),
		('valid_tokens_sha256', 'valid_tokens_sha256'),
	):
		if row.get(row_key) != provenance.get(source_key):
			raise ValueError(
				f'{row["model_id"]} completed job {row_key} does not match '
				'the current configured source'
			)


def _paired_rows(
	rows: list[dict[str, object]],
	*,
	comparisons: Sequence[tuple[str, str, str]],
	metrics: Sequence[str],
) -> list[dict[str, object]]:
	by_cell = {
		(str(row['model_id']), str(row['layout_id']), str(row['data_size'])): row
		for row in rows
	}
	paired = []
	for data_size in DATA_SIZES:
		if not any(str(row['data_size']) == data_size for row in rows):
			continue
		for layout_id in LAYOUT_IDS:
			for comparison_id, left_model, right_model in comparisons:
				try:
					left = by_cell[left_model, layout_id, data_size]
					right = by_cell[right_model, layout_id, data_size]
				except KeyError as error:
					raise ValueError(
						'missing paired cell for '
						f'{comparison_id}/{layout_id}/{data_size}'
					) from error
				for metric in metrics:
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


def _by_size_rows(
	paired: list[dict[str, object]],
	*,
	comparisons: Sequence[tuple[str, str, str]],
) -> list[dict[str, object]]:
	rows = []
	for data_size in DATA_SIZES:
		for comparison_id, _left, _right in comparisons:
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
						f'{data_size}/{comparison_id}/{metric} must contain '
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
						'positive_count': sum(value > 0.0 for value in deltas),
						'zero_count': sum(value == 0.0 for value in deltas),
						'negative_count': sum(value < 0.0 for value in deltas),
					}
				)
	return rows


def _benchmark_summary_payload(
	*,
	suite: str,
	models: Sequence[str],
	job_count: int,
	by_size: list[dict[str, object]],
) -> dict[str, object]:
	grouped: dict[str, dict[str, dict[str, object]]] = {}
	for row in by_size:
		grouped.setdefault(str(row['data_size']), {}).setdefault(
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
		'suite': suite,
		'models': list(models),
		'job_count': job_count,
		'primary_metric': PRIMARY_METRIC,
		'aggregation_unit': EXPECTED_AGGREGATION_UNIT,
		'statistical_unit': 'layout_id',
		'by_size': grouped,
	}


def _benchmark_outputs(
	*,
	rows: list[dict[str, object]],
	paired: list[dict[str, object]],
	by_size: list[dict[str, object]],
	payload: Mapping[str, object],
	title: str,
) -> dict[str, str]:
	return {
		COMPARISON_CSV_NAME: _csv_text(COMPARISON_FIELDNAMES, rows),
		PAIRED_DELTAS_CSV_NAME: _csv_text(PAIRED_FIELDNAMES, paired),
		SUMMARY_BY_SIZE_CSV_NAME: _csv_text(BY_SIZE_FIELDNAMES, by_size),
		SUMMARY_JSON_NAME: json.dumps(payload, indent=2, sort_keys=True) + '\n',
		SUMMARY_MD_NAME: _benchmark_markdown(title, by_size),
	}


def _screening_markdown(
	payload: Mapping[str, object], paired: list[dict[str, object]]
) -> str:
	lines = [
		'# F3 Local VICReg versus Random screening',
		'',
		f'Gate: `{payload["gate_status"]}`.',
		'',
		'| layout | Local VICReg | Random | delta |',
		'|---|---:|---:|---:|',
	]
	lines.extend(

			f'| {row["layout_id"]} | {row["left_value"]:.6f} '
			f'| {row["right_value"]:.6f} | {row["delta"]:.6f} |'
			for row in paired

	)
	lines.extend(
		(
			'',
			(
				f'Mean delta: `{payload["mean_paired_delta"]:.6f}`; '
				f'median: `{payload["median_paired_delta"]:.6f}`; '
				f'wins: `{payload["wins"]}/5`.'
			),
			'',
		)
	)
	return '\n'.join(lines)


def _benchmark_markdown(title: str, by_size: list[dict[str, object]]) -> str:
	lines = [
		f'# {title}',
		'',
		'Primary metric is `macro_f1`; paired unit is `layout_id` per size.',
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


def _atomic_summary(root: Path, outputs: Mapping[str, str]) -> None:
	if root.exists():
		raise FileExistsError(f'refusing to overwrite existing summary: {root}')
	root.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(prefix=f'.{root.name}.staging-', dir=root.parent)
	)
	try:
		for name, text in outputs.items():
			(staging / name).write_text(text, encoding='utf-8')
		staging.replace(root)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise


def _csv_text(
	fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> str:
	buffer = io.StringIO()
	writer = csv.DictWriter(buffer, fieldnames=fieldnames)
	writer.writeheader()
	for row in rows:
		writer.writerow({key: row[key] for key in fieldnames})
	return buffer.getvalue()


def _reject_unexpected_run_directories(
	runs_root: Path,
	*,
	model_ids: Sequence[str],
	layouts: Sequence[str],
	sizes: Sequence[str],
) -> None:
	if not runs_root.is_dir():
		return
	expected_models = {f'model={model_id}' for model_id in model_ids}
	for model_dir in sorted(runs_root.iterdir()):
		if not model_dir.is_dir() or model_dir.name not in expected_models:
			raise ValueError(f'unexpected VICReg run directory: {model_dir}')
		for layout_dir in sorted(model_dir.iterdir()):
			if not layout_dir.is_dir() or layout_dir.name not in {
				f'layout={layout}' for layout in layouts
			}:
				raise ValueError(f'unexpected VICReg run directory: {layout_dir}')
			for size_dir in sorted(layout_dir.iterdir()):
				if not size_dir.is_dir() or size_dir.name not in {
					f'size={size}' for size in sizes
				}:
					raise ValueError(f'unexpected VICReg run directory: {size_dir}')


def _validate_vicreg_checkpoint(  # noqa: C901, PLR0912, PLR0915
	path: Path, *, role: str
) -> dict[str, object]:
	payload = load_checkpoint(path, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'VICReg checkpoint payload must be a mapping: {path}')
	config = _required_checkpoint_mapping(payload, 'config', path=path)
	_validate_vicreg_base_config(config, label=f'{path} config')
	if role == 'screening':
		_validate_checkpoint_counters(
			payload,
			epochs=VICREG_STAGE1_EPOCHS,
			global_steps=VICREG_STAGE1_GLOBAL_STEPS,
			label=str(path),
		)
		_validate_direct_vicreg_identity(payload, path=path)
		return {'role': role, 'source_checkpoint': None}
	if role == 'control':
		_validate_checkpoint_counters(
			payload,
			epochs=VICREG_STAGE2_EPOCHS,
			global_steps=VICREG_STAGE2_GLOBAL_STEPS,
			label=str(path),
		)
		_validate_direct_vicreg_identity(payload, path=path)
		continuation = _required_checkpoint_mapping(config, 'continuation', path=path)
		if continuation.get('unfreeze_top_blocks') != 1:
			raise ValueError(f'{path} continuation.unfreeze_top_blocks must be 1')
		source = _required_existing_path(
			continuation.get('init_checkpoint'),
			label=f'{path} continuation.init_checkpoint',
		)
		source_sha256 = _validate_stage1_source(source)
		_validate_vicreg_stage1_identity(
			config, source=source, label=f'{path} control'
		)
		lineage = _required_checkpoint_mapping(
			payload, 'continuation_lineage', path=path
		)
		if lineage.get('init_checkpoint') != str(source):
			raise ValueError(
				f'{path} continuation lineage source does not match config'
			)
		if lineage.get('init_checkpoint_sha256') != source_sha256:
			raise ValueError(f'{path} continuation lineage source SHA does not match')
		return {
			'role': role,
			'source_checkpoint': str(source),
			'source_checkpoint_sha256': source_sha256,
		}
	if role != 'hmm':
		raise ValueError(f'unsupported VICReg checkpoint role: {role!r}')
	_validate_checkpoint_counters(
		payload,
		epochs=VICREG_STAGE2_EPOCHS,
		global_steps=VICREG_STAGE2_GLOBAL_STEPS,
		label=str(path),
	)
	stratigraphy = _required_checkpoint_mapping(
		payload, 'stratigraphy_config', path=path
	)
	if stratigraphy.get('stage') != 'train_strat_hmm_pretext':
		raise ValueError(f'{path} stratigraphy_config.stage is invalid')
	teacher = _required_checkpoint_mapping(stratigraphy, 'teacher', path=path)
	student = _required_checkpoint_mapping(stratigraphy, 'student', path=path)
	teacher_path = _required_existing_path(
		teacher.get('checkpoint'), label=f'{path} teacher.checkpoint'
	)
	student_path = _required_existing_path(
		student.get('init_checkpoint'), label=f'{path} student.init_checkpoint'
	)
	if teacher_path.resolve(strict=False) != student_path.resolve(strict=False):
		raise ValueError(f'{path} HMM teacher and student must use the same VICReg100')
	if student.get('unfreeze_top_blocks') != 1:
		raise ValueError(f'{path} student.unfreeze_top_blocks must be 1')
	head = _required_checkpoint_mapping(stratigraphy, 'head', path=path)
	if head.get('num_prototypes') != VICREG_HMM_K:
		raise ValueError(f'{path} HMM head num_prototypes must be {VICREG_HMM_K}')
	pseudo_targets = _required_checkpoint_mapping(
		stratigraphy, 'pseudo_targets', path=path
	)
	if pseudo_targets.get('k') != VICREG_HMM_K:
		raise ValueError(f'{path} pseudo_targets.k must be {VICREG_HMM_K}')
	if float(pseudo_targets.get('min_confidence', -1.0)) != 0.0:
		raise ValueError(f'{path} pseudo_targets.min_confidence must be 0.0')
	pseudo_target_dir = _validate_hmm_pseudo_target_dir(
		pseudo_targets.get('input_dir'), path=path
	)
	loss = _required_checkpoint_mapping(stratigraphy, 'loss', path=path)
	if float(loss.get('distillation_weight', -1.0)) != 0.2:
		raise ValueError(f'{path} distillation_weight must be 0.2')
	train = _required_checkpoint_mapping(stratigraphy, 'train', path=path)
	if train.get('epochs') != VICREG_STAGE2_EPOCHS:
		raise ValueError(f'{path} HMM config must declare 25 epochs')
	source_sha256 = _validate_stage1_source(teacher_path)
	_validate_vicreg_stage1_identity(
		config, source=teacher_path, label=f'{path} HMM base'
	)
	_validate_hmm_control_identity(
		payload,
		path=path,
		stratigraphy=stratigraphy,
		teacher_path=teacher_path,
		student_path=student_path,
		pseudo_target_dir=pseudo_target_dir,
	)
	return {
		'role': role,
		'source_checkpoint': str(teacher_path),
		'source_checkpoint_sha256': source_sha256,
		'hmm_k': VICREG_HMM_K,
	}


def _validate_hmm_pseudo_target_dir(value: object, *, path: Path) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{path} pseudo_targets.input_dir must be a non-empty string')
	target_dir = Path(value)
	if not target_dir.is_absolute():
		raise ValueError(f'{path} pseudo_targets.input_dir must be absolute')
	if tuple(target_dir.parts[-len(VICREG_HMM_TARGET_SUFFIX) :]) != (
		VICREG_HMM_TARGET_SUFFIX
	):
		raise ValueError(
			f'{path} HMM pseudo-target path must end with '
			f'{"/".join(VICREG_HMM_TARGET_SUFFIX)!r}'
		)
	return target_dir


def _validate_hmm_control_identity(  # noqa: C901, PLR0912, PLR0913
	payload: Mapping[str, object],
	*,
	path: Path,
	stratigraphy: Mapping[str, object],
	teacher_path: Path,
	student_path: Path,
	pseudo_target_dir: Path,
) -> None:
	identity = _required_checkpoint_mapping(stratigraphy, 'identity', path=path)
	model_tag = identity.get('model_tag')
	if not isinstance(model_tag, str) or not model_tag:
		raise ValueError(f'{path} identity.model_tag must be a non-empty string')
	control = _required_checkpoint_mapping(payload, 'control_identity', path=path)
	if control.get('schema_version') != 1:
		raise ValueError(f'{path} control_identity.schema_version must be 1')
	if control.get('model_tag') != model_tag:
		raise ValueError(
			f'{path} control_identity.model_tag does not match stratigraphy identity'
		)
	inputs = _required_checkpoint_mapping(
		control, 'input_identities', path=path
	)
	_validate_recorded_file_identity(
		inputs.get('teacher_checkpoint'),
		expected_path=teacher_path,
		label=f'{path} control_identity teacher checkpoint',
	)
	_validate_recorded_file_identity(
		inputs.get('student_init_checkpoint'),
		expected_path=student_path,
		label=f'{path} control_identity student init checkpoint',
	)
	recorded_targets = inputs.get('pseudo_targets')
	if not isinstance(recorded_targets, Sequence) or isinstance(
		recorded_targets, str | bytes
	):
		raise TypeError(f'{path} control_identity pseudo_targets must be a list')
	recorded_by_survey: dict[str, Mapping[str, object]] = {}
	for index, recorded in enumerate(recorded_targets):
		if not isinstance(recorded, Mapping):
			raise TypeError(
				f'{path} control_identity pseudo_targets[{index}] must be a mapping'
			)
		survey_id = recorded.get('survey_id')
		if not isinstance(survey_id, str) or not survey_id:
			raise ValueError(
				f'{path} control_identity pseudo_targets[{index}].survey_id '
				'must be non-empty'
			)
		if survey_id in recorded_by_survey:
			raise ValueError(
				f'{path} control_identity has duplicate pseudo-target survey '
				f'{survey_id!r}'
			)
		recorded_by_survey[survey_id] = recorded
	current_targets = discover_pseudo_target_inputs(
		pseudo_target_dir, k=VICREG_HMM_K
	)
	current_by_survey = {item.survey_id: item for item in current_targets}
	if set(recorded_by_survey) != set(current_by_survey):
		raise ValueError(
			f'{path} control_identity pseudo-target survey set does not match '
			'current inputs'
		)
	for survey_id, current in current_by_survey.items():
		recorded = recorded_by_survey[survey_id]
		for field, current_path in (
			('labels', current.labels_path),
			('confidence', current.confidence_path),
			('valid_tokens', current.valid_tokens_path),
			('metadata', current.metadata_path),
		):
			_validate_recorded_file_identity(
				recorded.get(field),
				expected_path=current_path,
				label=(
					f'{path} control_identity pseudo-target {survey_id} {field}'
				),
			)
		boundary_present = recorded.get('boundary_weight_present')
		if not isinstance(boundary_present, bool):
			raise TypeError(
				f'{path} control_identity pseudo-target {survey_id} '
				'boundary_weight_present must be boolean'
			)
		current_boundary_present = current.boundary_weight_path is not None
		if boundary_present != current_boundary_present:
			raise ValueError(
				f'{path} control_identity pseudo-target {survey_id} boundary '
				'presence does not match current input'
			)
		if current.boundary_weight_path is not None:
			_validate_recorded_file_identity(
				recorded.get('boundary_weight'),
				expected_path=current.boundary_weight_path,
				label=(
					f'{path} control_identity pseudo-target {survey_id} '
					'boundary_weight'
				),
			)
		elif 'boundary_weight' in recorded:
			raise ValueError(
				f'{path} control_identity pseudo-target {survey_id} has an '
				'unexpected boundary_weight identity'
			)


def _validate_recorded_file_identity(
	value: object, *, expected_path: Path, label: str
) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	recorded_path = _required_existing_path(value.get('path'), label=f'{label}.path')
	if recorded_path.resolve(strict=False) != expected_path.resolve(strict=False):
		raise ValueError(f'{label} path does not match current input')
	recorded_sha256 = _sha256(value.get('sha256'), label=f'{label}.sha256')
	if recorded_sha256 != file_sha256(expected_path):
		raise ValueError(f'{label} SHA does not match current input')


def _validate_stage1_source(path: Path) -> str:
	payload = load_checkpoint(path, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'VICReg100 source payload must be a mapping: {path}')
	config = _required_checkpoint_mapping(payload, 'config', path=path)
	_validate_vicreg_base_config(config, label=f'{path} config')
	if 'continuation' in config:
		raise ValueError(f'VICReg100 source must not be a continuation: {path}')
	_validate_checkpoint_counters(
		payload,
		epochs=VICREG_STAGE1_EPOCHS,
		global_steps=VICREG_STAGE1_GLOBAL_STEPS,
		label=str(path),
	)
	_validate_direct_vicreg_identity(payload, path=path)
	return file_sha256(path)


def _validate_direct_vicreg_identity(
	payload: Mapping[str, object], *, path: Path
) -> None:
	if payload.get('pretraining_method') != VICREG_METHOD:
		raise ValueError(f'{path} pretraining_method must be {VICREG_METHOD!r}')
	if payload.get('checkpoint_kind') != VICREG_CHECKPOINT_KIND:
		raise ValueError(
			f'{path} checkpoint_kind must be {VICREG_CHECKPOINT_KIND!r}'
		)
	if not isinstance(payload.get('projector_state_dict'), Mapping):
		raise TypeError(f'{path} projector_state_dict must be a mapping')
	model_state = payload.get('model_state_dict')
	if not isinstance(model_state, Mapping):
		raise TypeError(f'{path} model_state_dict must be a mapping')
	if any(
		isinstance(key, str) and key.startswith(('backbone.', 'projector.'))
		for key in model_state
	):
		raise ValueError(f'{path} model_state_dict must contain bare encoder keys')


def _validate_vicreg_base_config(
	config: Mapping[str, object], *, label: str
) -> None:
	if config.get('stage') != 'vicreg_training':
		raise ValueError(f'{label}.stage must be vicreg_training')
	_validate_contract_values(
		config.get('model'), expected=VICREG_MODEL_CONTRACT, label=f'{label}.model'
	)
	_validate_contract_values(
		config.get('vicreg'),
		expected=VICREG_OBJECTIVE_CONTRACT,
		label=f'{label}.vicreg',
	)
	augmentations = config.get('augmentations')
	if augmentations != VICREG_AUGMENTATION_CONTRACT:
		raise ValueError(
			f'{label}.augmentations must equal the forced-distinct horizontal-flip '
			f'contract {dict(VICREG_AUGMENTATION_CONTRACT)!r}'
		)


def _validate_contract_values(
	value: object, *, expected: Mapping[str, object], label: str
) -> None:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	for key, expected_value in expected.items():
		if value.get(key) != expected_value:
			raise ValueError(
				f'{label}.{key} must equal {expected_value!r}; '
				f'got {value.get(key)!r}'
			)


def _validate_vicreg_stage1_identity(
	config: Mapping[str, object], *, source: Path, label: str
) -> None:
	payload = load_checkpoint(source, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} source checkpoint payload must be a mapping')
	source_config = _required_checkpoint_mapping(payload, 'config', path=source)
	for section in ('data', 'zero_mask', 'model', 'vicreg', 'augmentations'):
		if config.get(section) != source_config.get(section):
			raise ValueError(
				f'{label} {section} identity differs from screened VICReg100'
			)


def _validate_checkpoint_counters(
	payload: Mapping[str, object],
	*,
	epochs: int,
	global_steps: int,
	label: str,
) -> None:
	if payload.get('epoch') != epochs or payload.get('global_step') != global_steps:
		raise ValueError(
			f'{label} must record epoch/global_step={epochs}/{global_steps}; '
			f'got {payload.get("epoch")!r}/{payload.get("global_step")!r}'
		)


def _validate_vicreg_embedding_metadata(  # noqa: C901, PLR0912
	path: Path, *, checkpoint: Path, role: str
) -> None:
	metadata = _read_json(path, label='VICReg embedding metadata')
	if metadata.get('pretraining_method') != VICREG_METHOD:
		raise ValueError(
			f'{path} pretraining_method must be {VICREG_METHOD!r}'
		)
	payload = load_checkpoint(checkpoint, map_location='cpu')
	if not isinstance(payload, Mapping):
		raise TypeError(f'{checkpoint} payload must be a mapping')
	checkpoint_config = _required_checkpoint_mapping(payload, 'config', path=checkpoint)
	checkpoint_model = _required_checkpoint_mapping(
		checkpoint_config, 'model', path=checkpoint
	)
	model_geometry = metadata.get('model_geometry')
	if not isinstance(model_geometry, Mapping):
		raise TypeError(f'{path} model_geometry must be a mapping')
	for key, value in checkpoint_model.items():
		if model_geometry.get(key) != value:
			raise ValueError(
				f'{path} model_geometry.{key} does not match checkpoint config.model'
			)
	objective = metadata.get('pretraining_objective')
	if not isinstance(objective, Mapping) or objective.get('method') != VICREG_METHOD:
		raise ValueError(
			f'{path} pretraining_objective.method must be {VICREG_METHOD!r}'
		)
	checkpoint_vicreg = _required_checkpoint_mapping(
		checkpoint_config, 'vicreg', path=checkpoint
	)
	for key, value in checkpoint_vicreg.items():
		if objective.get(key) != value:
			raise ValueError(
				f'{path} pretraining_objective.{key} does not match checkpoint '
				'config.vicreg'
			)
	pretext = metadata.get('stratigraphy_pretext')
	if role != 'hmm':
		if pretext is not None:
			raise ValueError(f'{path} direct VICReg embedding must not declare pretext')
		return
	if not isinstance(pretext, Mapping):
		raise TypeError(f'{path} HMM embedding stratigraphy_pretext must be a mapping')
	for key, expected in (
		('method', 'strat_hmm_pretext'),
		('base_objective', VICREG_METHOD),
		('head_num_prototypes', VICREG_HMM_K),
		('unfreeze_top_blocks', 1),
		('distillation_weight', 0.2),
	):
		if pretext.get(key) != expected:
			raise ValueError(f'{path} HMM {key} must equal {expected!r}')
	pseudo_target_dir = pretext.get('pseudo_target_input_dir')
	if not isinstance(pseudo_target_dir, str) or not pseudo_target_dir:
		raise ValueError(f'{path} HMM pseudo_target_input_dir is required')
	if 'trace_drop' in pseudo_target_dir:
		raise ValueError(f'{path} HMM pseudo targets must not use trace drop')
	stratigraphy = _required_checkpoint_mapping(
		payload, 'stratigraphy_config', path=checkpoint
	)
	pseudo_targets = _required_checkpoint_mapping(
		stratigraphy, 'pseudo_targets', path=checkpoint
	)
	checkpoint_target_dir = pseudo_targets.get('input_dir')
	if pseudo_target_dir != checkpoint_target_dir:
		raise ValueError(
			f'{path} HMM pseudo_target_input_dir does not match checkpoint lineage'
		)
	if tuple(Path(pseudo_target_dir).parts[-len(VICREG_HMM_TARGET_SUFFIX) :]) != (
		VICREG_HMM_TARGET_SUFFIX
	):
		raise ValueError(
			f'{path} HMM pseudo-target path must end with '
			f'{"/".join(VICREG_HMM_TARGET_SUFFIX)!r}'
		)


def _resolve_extension_sources(value: object) -> tuple[F3FiveWayModelSource, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError('extension.models must be a list of two source mappings')
	if len(value) != len(EXTENSION_MODEL_IDS):
		raise ValueError('extension.models must contain exactly two entries')
	models = tuple(
		_resolve_source(item, expected_id=model_id, label=f'extension.models[{index}]')
		for index, (item, model_id) in enumerate(
			zip(value, EXTENSION_MODEL_IDS, strict=True)
		)
	)
	if len({model.checkpoint for model in models}) != len(models):
		raise ValueError('extension model checkpoints must be distinct')
	if len({model.embeddings_dir for model in models}) != len(models):
		raise ValueError('extension model embedding directories must be distinct')
	return models


def _resolve_source(
	value: object, *, expected_id: str, label: str
) -> F3FiveWayModelSource:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	_require_exact_keys(value, {'model_id', 'checkpoint', 'embeddings_dir'}, label)
	model_id = _required_str(value, 'model_id', prefix=label)
	if model_id != expected_id:
		raise ValueError(f'{label}.model_id must be {expected_id!r}; got {model_id!r}')
	checkpoint = _required_absolute_path(value, 'checkpoint', prefix=label)
	embeddings_dir = _required_absolute_path(value, 'embeddings_dir', prefix=label)
	if checkpoint == embeddings_dir:
		raise ValueError(f'{label} checkpoint and embeddings_dir must differ')
	return F3FiveWayModelSource(
		model_id=model_id,
		checkpoint=checkpoint,
		embeddings_dir=embeddings_dir,
		expected={
			'objective': VICREG_METHOD,
			'stratigraphy_pretext': model_id.endswith('_hmm_k6'),
		},
	)


def _resolve_output_roots(value: object, *, label: str) -> F3VICRegOutputRoots:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	_require_exact_keys(value, {'runs_root', 'job_logs_root', 'summary_root'}, label)
	return F3VICRegOutputRoots(
		runs_root=_required_absolute_path(value, 'runs_root', prefix=label),
		job_logs_root=_required_absolute_path(value, 'job_logs_root', prefix=label),
		summary_root=_required_absolute_path(value, 'summary_root', prefix=label),
	)


def _resolve_extension_output_roots(
	value: object,
) -> tuple[F3VICRegOutputRoots, Path]:
	if not isinstance(value, Mapping):
		raise TypeError('extension.outputs must be a mapping')
	_require_exact_keys(
		value,
		{'runs_root', 'job_logs_root', 'summary_root', 'combined_summary_root'},
		'extension.outputs',
	)
	return (
		F3VICRegOutputRoots(
			runs_root=_required_absolute_path(
				value, 'runs_root', prefix='extension.outputs'
			),
			job_logs_root=_required_absolute_path(
				value, 'job_logs_root', prefix='extension.outputs'
			),
			summary_root=_required_absolute_path(
				value, 'summary_root', prefix='extension.outputs'
			),
		),
		_required_absolute_path(
			value, 'combined_summary_root', prefix='extension.outputs'
		),
	)


def _validate_disjoint_roots(roots: Sequence[Path], *, label: str) -> None:
	for index, root in enumerate(roots):
		for other in roots[index + 1 :]:
			if _paths_overlap(root, other):
				raise ValueError(f'{label} must be disjoint: {root} and {other}')


def _paths_overlap(first: Path, second: Path) -> bool:
	first = first.resolve(strict=False)
	second = second.resolve(strict=False)
	return (
		first == second
		or first.is_relative_to(second)
		or second.is_relative_to(first)
	)


def _extension_job(
	canonical: F3FiveWayConfig,
	*,
	model: F3FiveWayModelSource,
	layout: str,
	size: str,
	runs_root: Path,
) -> F3FiveWayJob:
	condition_dir = (
		canonical.section_layout_dataset_root
		/ 'datasets'
		/ f'layout={layout}'
		/ f'size={size}'
		/ 'voxel_supervision'
	)
	return F3FiveWayJob(
		config=canonical,
		model=model,
		layout_id=layout,
		data_size=size,
		condition_dir=condition_dir,
		output_dir=(
			runs_root
			/ f'model={model.model_id}'
			/ f'layout={layout}'
			/ f'size={size}'
		),
	)


def _validate_layout(layout: str) -> None:
	if layout not in LAYOUT_IDS:
		raise ValueError(
			f'unknown layout: {layout!r}; expected one of {list(LAYOUT_IDS)!r}'
		)


def _validate_size(size: str) -> None:
	if size not in DATA_SIZES:
		raise ValueError(
			f'unknown data size: {size!r}; expected one of {list(DATA_SIZES)!r}'
		)


def _require_exact_keys(
	value: Mapping[str, object], expected: set[str], label: str
) -> None:
	keys = set(value)
	if keys != expected:
		missing = sorted(expected - keys)
		extra = sorted(str(key) for key in keys - expected)
		raise ValueError(
			f'{label} keys must be exactly {sorted(expected)!r}; '
			f'missing={missing!r}, unexpected={extra!r}'
		)


def _required_checkpoint_mapping(
	parent: Mapping[str, object], key: str, *, path: Path
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{path} {key} must be a mapping')
	return value


def _required_existing_path(value: object, *, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	path = Path(value)
	if not path.is_absolute():
		raise ValueError(f'{label} must be absolute')
	if not path.is_file():
		raise FileNotFoundError(f'{label} does not exist: {path}')
	return path


def _sha256(value: object, *, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256 digest')
	return value


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} must contain JSON: {path}') from error
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return payload


__all__ = [
	'BENCHMARK_SUMMARY_OUTPUT_NAMES',
	'COMPARISON_FIELDNAMES',
	'EXTENSION_MODEL_IDS',
	'EXTENSION_PAIRED_COMPARISONS',
	'PAIRED_FIELDNAMES',
	'SCREENING_DATA_SIZE',
	'SCREENING_MODEL_IDS',
	'SCREENING_SUMMARY_OUTPUT_NAMES',
	'SEVEN_WAY_MODEL_IDS',
	'SEVEN_WAY_PAIRED_COMPARISONS',
	'VICREG_GATE_FAIL',
	'VICREG_GATE_PASS',
	'F3VICRegExtensionConfig',
	'F3VICRegOutputRoots',
	'assert_f3_vicreg_full_benchmark_ready',
	'audit_f3_vicreg_screening_source',
	'audit_f3_vicreg_sources',
	'f3_vicreg_extension_config_from_mapping',
	'inspect_f3_vicreg_combined_results',
	'inspect_f3_vicreg_extension_results',
	'inspect_f3_vicreg_job',
	'inspect_f3_vicreg_screening_results',
	'load_f3_vicreg_canonical_config',
	'plan_f3_vicreg_extension_jobs',
	'plan_f3_vicreg_screening_jobs',
	'read_f3_vicreg_screening_gate',
	'resolve_f3_vicreg_extension_job',
	'resolve_f3_vicreg_screening_job',
	'run_f3_vicreg_job',
	'summarize_f3_vicreg_combined',
	'summarize_f3_vicreg_extension',
	'summarize_f3_vicreg_screening',
]
