'''One-job runner for the Volve horizon five-way frozen-encoder matrix.'''

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from seis_ssl_cluster.volve.horizon_data import load_volve_horizon_data
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	VolveHorizonFiveWayEmbeddingSuite,
	audit_volve_horizon_five_way_sources,
	inspect_volve_horizon_five_way_embedding_suite,
)
from seis_ssl_cluster.volve.horizon_frozen import (
	FrozenEmbeddingGeometry,
	FrozenHorizonConfig,
	FrozenHorizonPlan,
	build_frozen_horizon_plan,
	run_frozen_horizon_job,
)
from seis_ssl_cluster.volve.horizon_layouts import (
	DATA_SIZE_PREFIX,
	LAYOUT_IDS,
	build_horizon_split_plan,
	load_volve_horizon_layouts,
)
from seis_ssl_cluster.volve.horizon_tiles import (
	HORIZON_WINDOW_START,
	HORIZON_WINDOW_STOP,
)

if TYPE_CHECKING:
	from collections.abc import Callable

	from seis_ssl_cluster.volve.horizon_data import VolveHorizonData
	from seis_ssl_cluster.volve.horizon_five_way_config import (
		VolveHorizonFiveWayConfig,
		VolveHorizonFiveWayModelSource,
	)

FIVE_WAY_CONDITION_COUNT = 75
FIVE_WAY_BENCHMARK_ID = 'mae_local_bt_hmm_five_way_v1'


@dataclass(frozen=True)
class VolveHorizonFiveWayJob:
	'''One statically resolved model/layout/size cell.'''

	config: VolveHorizonFiveWayConfig
	model: VolveHorizonFiveWayModelSource
	layout_id: str
	data_size: str
	output_dir: Path

	@property
	def metrics_path(self) -> Path:
		'''Return the final artifact that marks this cell complete.'''
		return self.output_dir / 'metrics.json'

	@property
	def latest_path(self) -> Path:
		'''Return the only checkpoint accepted for exact-cell resume.'''
		return self.output_dir / 'latest.pt'


@dataclass(frozen=True)
class VolveHorizonFiveWaySuiteCellResult:
	'''Execution outcome for one cell of a sequential suite run.'''

	job: VolveHorizonFiveWayJob
	action: Literal['fresh', 'resume', 'skip']
	result: Path | None


def plan_volve_horizon_five_way_jobs(
	config: VolveHorizonFiveWayConfig,
) -> tuple[tuple[str, str, str], ...]:
	'''Enumerate the canonical 5 by 5 by 3 comparison matrix.'''
	jobs = tuple(
		(model_id, layout_id, data_size)
		for model_id in config.model_ids
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZE_PREFIX
	)
	if len(jobs) != FIVE_WAY_CONDITION_COUNT or len(set(jobs)) != len(jobs):
		raise RuntimeError('Volve horizon five-way suite must contain 75 unique jobs')
	return jobs


def enumerate_volve_horizon_five_way_conditions(
	config: VolveHorizonFiveWayConfig,
) -> tuple[tuple[str, str, str], ...]:
	'''Compatibility alias for callers that describe the matrix as conditions.'''
	return plan_volve_horizon_five_way_jobs(config)


def resolve_volve_horizon_five_way_job(  # noqa: PLR0913
	config: VolveHorizonFiveWayConfig,
	*,
	model: str,
	layout: str | None = None,
	size: str | None = None,
	layout_id: str | None = None,
	data_size: str | None = None,
) -> VolveHorizonFiveWayJob:
	'''Resolve one fixed cell without opening any source artifact.'''
	resolved_layout = _one_alias(layout, layout_id, label='layout')
	resolved_size = _one_alias(size, data_size, label='size')
	source = config.model_by_id(model)
	if resolved_layout not in LAYOUT_IDS:
		raise ValueError(
			f'unknown layout: {resolved_layout!r}; expected one of {list(LAYOUT_IDS)!r}'
		)
	if resolved_size not in DATA_SIZE_PREFIX:
		raise ValueError(
			'unknown data size: '
			f'{resolved_size!r}; expected one of {list(DATA_SIZE_PREFIX)!r}'
		)
	output_dir = (
		config.runs_root
		/ f'model={model}'
		/ f'layout={resolved_layout}'
		/ f'size={resolved_size}'
	)
	return VolveHorizonFiveWayJob(
		config=config,
		model=source,
		layout_id=resolved_layout,
		data_size=resolved_size,
		output_dir=output_dir,
	)


def inspect_volve_horizon_five_way_job(
	job: VolveHorizonFiveWayJob,
	*,
	layout_config: str | Path,
	data: VolveHorizonData | None = None,
	embedding_suite: VolveHorizonFiveWayEmbeddingSuite | None = None,
) -> FrozenHorizonPlan:
	'''Run read-only source/support preflight and build one decoder plan.'''
	if embedding_suite is None:
		source_audit = audit_volve_horizon_five_way_sources(job.config)
		suite = inspect_volve_horizon_five_way_embedding_suite(
			job.config,
			source_audit=source_audit,
		)
	else:
		suite = embedding_suite
	resolved_data = (
		load_volve_horizon_data(job.config.volve_root) if data is None else data
	)
	if suite.volume_shape_xyz != (*resolved_data.shape_xy, len(resolved_data.time_ms)):
		raise ValueError('five-way embedding volume geometry differs from horizon data')
	if suite.model_valid_lateral_mask.shape != resolved_data.shape_xy:
		raise ValueError('five-way model-valid lateral support has the wrong shape')
	legacy_config = FrozenHorizonConfig(
		artifact_root=job.config.artifact_root,
		volve_root=job.config.volve_root,
		survey_id=job.config.survey_id,
		canonical_input_metadata=job.config.canonical_input_metadata,
		pretrained_embeddings_dir=job.config.models[0].embeddings_dir,
		random_embeddings_dir=job.config.models[-1].embeddings_dir,
		runs_root=job.config.runs_root,
		train=job.config.train,
		tiles=replace(job.config.tiles, lateral_shape_xy=resolved_data.shape_xy),
	)
	layouts = load_volve_horizon_layouts(layout_config, resolved_data)
	split_plan = build_horizon_split_plan(
		resolved_data,
		layouts,
		job.layout_id,
		job.data_size,
	)
	if (
		split_plan.twt_window.start_index,
		split_plan.twt_window.stop_index_exclusive,
	) != (HORIZON_WINDOW_START, HORIZON_WINDOW_STOP):
		raise ValueError('split plan must use the fixed [552, 768) TWT window')
	selected = suite.sources[job.model.model_id]
	geometry = FrozenEmbeddingGeometry(
		pretrained=selected.paths,
		random=selected.paths,
		volume_shape_xyz=suite.volume_shape_xyz,
		token_grid_shape_xyz=suite.token_grid_shape_xyz,
		embedding_shape=suite.embedding_shape,
		embedding_dim=suite.embedding_dim,
		pretrained_metadata=selected.metadata,
		random_metadata=selected.metadata,
		pretrained_model_source=selected.checkpoint_identity,
		random_model_source=selected.checkpoint_identity,
		valid_tokens_sha256=suite.valid_tokens_sha256,
		model_valid_lateral_mask=suite.model_valid_lateral_mask,
		model_valid_lateral_mask_sha256=(
			suite.model_valid_lateral_mask_sha256
		),
		canonical_identity=suite.canonical_identity,
	)
	return build_frozen_horizon_plan(
		config=legacy_config,
		model=job.model.model_id,
		data=resolved_data,
		split_plan=split_plan,
		geometry=geometry,
		selected_paths=selected.paths,
		selected_metadata=selected.metadata,
		selected_model_source=selected.checkpoint_identity,
		selected_embeddings_sha256=selected.embeddings_sha256,
		selected_metadata_sha256=selected.metadata_sha256,
		selected_valid_tokens_sha256=selected.valid_tokens_sha256,
		benchmark=FIVE_WAY_BENCHMARK_ID,
	)


def run_volve_horizon_five_way_job(
	plan: FrozenHorizonPlan,
	*,
	device: str = 'auto',
	max_steps: int | None = None,
	resume: str | Path | None = None,
) -> Path | None:
	'''Execute one preflighted cell through the existing frozen decoder loop.'''
	if plan.output_dir.joinpath('metrics.json').exists():
		raise FileExistsError(
			f'five-way horizon job is already complete: {plan.output_dir}'
		)
	if resume is not None:
		resume_path = Path(resume)
		if resume_path.resolve(strict=False) != plan.output_dir.joinpath(
			'latest.pt'
		).resolve(strict=False):
			raise ValueError('resume must identify this exact cell latest.pt')
	return run_frozen_horizon_job(
		plan,
		device=device,
		max_steps=max_steps,
		resume=resume,
	)


def run_volve_horizon_five_way_suite(  # noqa: PLR0913
	config: VolveHorizonFiveWayConfig,
	*,
	layout_config: str | Path,
	device: str = 'auto',
	max_steps: int | None = None,
	continue_existing: bool = False,
	progress: Callable[[VolveHorizonFiveWaySuiteCellResult], None] | None = None,
) -> tuple[VolveHorizonFiveWaySuiteCellResult, ...]:
	'''Preflight shared inputs once and execute the canonical 75-cell suite.'''
	source_audit = audit_volve_horizon_five_way_sources(config)
	embedding_suite = inspect_volve_horizon_five_way_embedding_suite(
		config,
		source_audit=source_audit,
	)
	data = load_volve_horizon_data(config.volve_root)
	results: list[VolveHorizonFiveWaySuiteCellResult] = []
	for model, layout, size in plan_volve_horizon_five_way_jobs(config):
		job = resolve_volve_horizon_five_way_job(
			config,
			model=model,
			layout=layout,
			size=size,
		)
		action = _suite_cell_action(job, continue_existing=continue_existing)
		if action == 'skip':
			cell = VolveHorizonFiveWaySuiteCellResult(
				job=job,
				action=action,
				result=job.metrics_path,
			)
		else:
			plan = inspect_volve_horizon_five_way_job(
				job,
				layout_config=layout_config,
				data=data,
				embedding_suite=embedding_suite,
			)
			result = run_volve_horizon_five_way_job(
				plan,
				device=device,
				max_steps=max_steps,
				resume=job.latest_path if action == 'resume' else None,
			)
			cell = VolveHorizonFiveWaySuiteCellResult(
				job=job,
				action=action,
				result=result,
			)
		results.append(cell)
		if progress is not None:
			progress(cell)
	return tuple(results)


def _suite_cell_action(
	job: VolveHorizonFiveWayJob,
	*,
	continue_existing: bool,
) -> Literal['fresh', 'resume', 'skip']:
	if not continue_existing:
		return 'fresh'
	if job.metrics_path.is_file():
		return 'skip'
	if job.latest_path.is_file():
		return 'resume'
	return 'fresh'


def _one_alias(first: str | None, second: str | None, *, label: str) -> str:
	if first is None and second is None:
		raise TypeError(f'{label} is required')
	if first is not None and second is not None and first != second:
		raise ValueError(f'{label} aliases disagree: {first!r} != {second!r}')
	return first if first is not None else str(second)


__all__ = [
	'FIVE_WAY_BENCHMARK_ID',
	'FIVE_WAY_CONDITION_COUNT',
	'VolveHorizonFiveWayJob',
	'VolveHorizonFiveWaySuiteCellResult',
	'enumerate_volve_horizon_five_way_conditions',
	'inspect_volve_horizon_five_way_job',
	'plan_volve_horizon_five_way_jobs',
	'resolve_volve_horizon_five_way_job',
	'run_volve_horizon_five_way_job',
	'run_volve_horizon_five_way_suite',
]
