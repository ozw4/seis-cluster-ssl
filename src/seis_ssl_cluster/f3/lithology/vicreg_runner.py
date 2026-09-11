"""Plan, resolve, inspect, and run VICReg F3 benchmark jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	F3FiveWayJob,
	inspect_f3_lithology_five_way_job,
	resolve_f3_lithology_five_way_job,
	run_f3_lithology_frozen_encoder_job,
)
from seis_ssl_cluster.f3.lithology.vicreg_sources import (
	EXTENSION_MODEL_IDS,
	SCREENING_DATA_SIZE,
	SCREENING_MODEL_IDS,
	F3VICRegExtensionConfig,
	audit_f3_vicreg_screening_source,
)

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.config.f3_lithology_five_way import (
		F3FiveWayConfig,
		F3FiveWayModelSource,
	)


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


def inspect_f3_vicreg_job(
	config: F3VICRegExtensionConfig,
	canonical: F3FiveWayConfig,
	job: F3FiveWayJob,
	*,
	suite: str,
) -> dict[str, object]:
	"""Return a source-audited, no-write plan for one screening/extension cell."""
	if suite == 'extension':
		from seis_ssl_cluster.f3.lithology.vicreg_results import (  # noqa: PLC0415
			assert_f3_vicreg_full_benchmark_ready,
		)

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
		from seis_ssl_cluster.f3.lithology.vicreg_results import (  # noqa: PLC0415
			assert_f3_vicreg_full_benchmark_ready,
		)

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
		from seis_ssl_cluster.f3.lithology.vicreg_results import (  # noqa: PLC0415
			read_f3_vicreg_completed_job,
		)

		row = read_f3_vicreg_completed_job(
			canonical,
			job,
			source_report=report,
		)
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
			runs_root / f'model={model.model_id}' / f'layout={layout}' / f'size={size}'
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


__all__ = [
	'inspect_f3_vicreg_job',
	'plan_f3_vicreg_extension_jobs',
	'plan_f3_vicreg_screening_jobs',
	'resolve_f3_vicreg_extension_job',
	'resolve_f3_vicreg_screening_job',
	'run_f3_vicreg_job',
]
