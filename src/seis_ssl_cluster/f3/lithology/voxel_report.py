"""Human-readable reporting and lightweight publish for F3 voxel results."""

# ruff: noqa: PERF401

from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_dataset import GRID_NAME, METADATA_NAME
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	BOUNDARY_METRICS_JSON,
	BOUNDARY_REGION_METRICS_CSV,
	EVALUATION_METADATA_JSON,
	EVALUATION_OUTPUT_FILES,
	METRICS_JSON,
	VALIDATION_SLICE_METRICS_CSV,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	CONFIDENCE_NAME as PREDICTION_CONFIDENCE_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	METADATA_NAME as PREDICTION_METADATA_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	PREDICTIONS_NAME as VOXEL_PREDICTIONS_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	VALID_MASK_NAME as PREDICTION_VALID_MASK_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	F3VoxelPredictionArtifact,
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_visualization import (
	F3LithologyVoxelFigureConfig,
	visualize_f3_lithology_voxel_slices,
)
from seis_ssl_cluster.f3.splits import (
	load_f3_slice_split_records,
	read_f3_line_geometry,
)
from seis_ssl_cluster.models.voxel_decoder.spec import (
	validate_voxel_decoder_architecture_mapping,
)

_DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

DEFAULT_REPORTS_ROOT = Path('reports')
if TYPE_CHECKING:
	from seis_ssl_cluster.f3.labels import F3ClassInfo

REPORT_MARKDOWN = 'report.md'
REPORT_JSON = 'report.json'
FIGURES_DIR = 'figures'
SELECTED_SLICES_DIR = 'selected_slices'
VOXEL_REPORT_PUBLISH_SUFFIXES = frozenset({'.md', '.json', '.csv', '.png'})
VOXEL_EVALUATION_PUBLISH_FILES = (
	*EVALUATION_OUTPUT_FILES,
	EVALUATION_METADATA_JSON,
)
KNOWN_LIMITATIONS = (
	'Evaluation is limited to supervised planes.',
	'Spatially correlated voxels are not treated as independent samples for '
	'significance claims.',
	'V0 is voxel-shaped output at token resolution.',
	'V1 has no raw-amplitude skip connection.',
)
V0_PREDICTION_SPEC = 'token_projection_nearest_v1'


@dataclass(frozen=True)
class F3LithologyVoxelPublishConfig:
	"""Lightweight repository-publish policy for one voxel report."""

	enabled: bool = False
	output_dir: Path | None = None
	reports_root: Path = DEFAULT_REPORTS_ROOT
	max_file_size_bytes: int = _DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES
	overwrite: bool = True

	def __post_init__(self) -> None:
		"""Validate publish controls before any output is written."""
		if not isinstance(self.enabled, bool) or not isinstance(self.overwrite, bool):
			raise TypeError('publish enabled and overwrite must be boolean')
		if self.max_file_size_bytes <= 0:
			raise ValueError('publish max_file_size_bytes must be positive')


@dataclass(frozen=True)
class F3LithologyVoxelReportConfig:
	"""Resolved inputs and fixed outputs for a common V0/V1 voxel report."""

	prediction_input_dir: Path
	voxel_dataset_input_dir: Path
	evaluation_input_dir: Path
	seismic_volume: Path
	label_volume: Path
	class_info: Path
	png_label_inventory: Path
	segy_geometry_json: Path
	output_dir: Path
	dataset: Mapping[str, str] = field(default_factory=dict)
	selected_slices: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
	figure: F3LithologyVoxelFigureConfig = field(
		default_factory=F3LithologyVoxelFigureConfig
	)
	publish: F3LithologyVoxelPublishConfig = field(
		default_factory=F3LithologyVoxelPublishConfig
	)
	overwrite: bool = False


@dataclass(frozen=True)
class F3LithologyVoxelReportResult:
	"""Report, figure, and optional publish paths."""

	report_markdown: Path
	report_json: Path
	figure_paths: tuple[Path, ...]
	payload: Mapping[str, object]
	published_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class F3LithologyVoxelReportInspection:
	"""Identity-validated numeric and volume inputs for one report."""

	metrics: Mapping[str, object]
	boundary_metrics: Mapping[str, object]
	evaluation_metadata: Mapping[str, object]
	boundary_region_rows: tuple[Mapping[str, object], ...]
	validation_slice_rows: tuple[Mapping[str, object], ...]
	prediction_artifact: F3VoxelPredictionArtifact
	supervision_metadata: Mapping[str, object]
	classes: tuple[F3ClassInfo, ...]
	selected_slices: tuple[tuple[str, int], ...]


def inspect_f3_lithology_voxel_report(
	config: F3LithologyVoxelReportConfig,
) -> F3LithologyVoxelReportInspection:
	"""Validate every report input and selected validation-slice identity."""
	_validate_input_files(config)
	evaluation = _read_json_object(
		config.evaluation_input_dir / EVALUATION_METADATA_JSON
	)
	_validate_evaluation_output_identities(
		evaluation, evaluation_input_dir=config.evaluation_input_dir
	)
	metrics = _read_json_object(config.evaluation_input_dir / METRICS_JSON)
	boundary = _read_json_object(
		config.evaluation_input_dir / BOUNDARY_METRICS_JSON
	)
	regions = tuple(
		_read_csv(config.evaluation_input_dir / BOUNDARY_REGION_METRICS_CSV)
	)
	slices = tuple(
		_read_csv(config.evaluation_input_dir / VALIDATION_SLICE_METRICS_CSV)
	)
	prediction = validate_f3_voxel_prediction_artifact(
		config.prediction_input_dir, mmap_mode='r'
	)
	supervision = _read_json_object(config.voxel_dataset_input_dir / METADATA_NAME)
	_validate_identity_summary(
		prediction.metadata,
		evaluation=evaluation,
		supervision=supervision,
		config=config,
	)
	classes = _read_classes(config.class_info)
	selected = _selected_slice_pairs(config, slice_rows=slices)
	return F3LithologyVoxelReportInspection(
		metrics=metrics,
		boundary_metrics=boundary,
		evaluation_metadata=evaluation,
		boundary_region_rows=regions,
		validation_slice_rows=slices,
		prediction_artifact=prediction,
		supervision_metadata=supervision,
		classes=classes,
		selected_slices=selected,
	)


def build_f3_lithology_voxel_report(
	config: F3LithologyVoxelReportConfig,
) -> F3LithologyVoxelReportResult:
	"""Build aggregate plots, selected sections, report files, and publish copy."""
	inspection = inspect_f3_lithology_voxel_report(config)
	if config.output_dir.exists() and not config.overwrite:
		raise FileExistsError(
			f'refusing to overwrite existing output: {config.output_dir}'
		)
	config.output_dir.parent.mkdir(parents=True, exist_ok=True)
	staging = Path(
		tempfile.mkdtemp(
			prefix=f'.{config.output_dir.name}.staging-', dir=config.output_dir.parent
		)
	)
	try:
		staged_result = _write_f3_lithology_voxel_report(
			staging, config=config, inspection=inspection
		)
		figure_paths = tuple(
			config.output_dir / path.relative_to(staging)
			for path in staged_result.figure_paths
		)
		_commit_directory(staging, config.output_dir, overwrite=config.overwrite)
	except BaseException:
		shutil.rmtree(staging, ignore_errors=True)
		raise
	result = F3LithologyVoxelReportResult(
		config.output_dir / REPORT_MARKDOWN,
		config.output_dir / REPORT_JSON,
		figure_paths,
		staged_result.payload,
	)
	published_files = publish_f3_lithology_voxel_report(result, config=config)
	return F3LithologyVoxelReportResult(
		result.report_markdown,
		result.report_json,
		result.figure_paths,
		result.payload,
		published_files,
	)


def _write_f3_lithology_voxel_report(
	output_dir: Path,
	*,
	config: F3LithologyVoxelReportConfig,
	inspection: F3LithologyVoxelReportInspection,
) -> F3LithologyVoxelReportResult:
	"""Write a complete report into a new staging directory."""
	figures_dir = output_dir / FIGURES_DIR
	selected_dir = figures_dir / SELECTED_SLICES_DIR
	figures_dir.mkdir(parents=True, exist_ok=True)
	selected_dir.mkdir(parents=True, exist_ok=True)

	metrics = inspection.metrics
	boundary = inspection.boundary_metrics
	evaluation = inspection.evaluation_metadata
	regions = inspection.boundary_region_rows
	slices = inspection.validation_slice_rows
	prediction = inspection.prediction_artifact
	supervision = inspection.supervision_metadata
	classes = inspection.classes

	aggregate_figures = _write_aggregate_figures(
		figures_dir,
		metrics=metrics,
		boundary=boundary,
		regions=regions,
		classes=classes,
		config=config.figure,
	)
	visualization = visualize_f3_lithology_voxel_slices(
		seismic=np.load(config.seismic_volume, mmap_mode='r', allow_pickle=False),
		labels=np.load(config.label_volume, mmap_mode='r', allow_pickle=False),
		predictions=prediction.arrays.predictions,
		confidence=prediction.arrays.confidence,
		split_grid=np.load(
			config.voxel_dataset_input_dir / GRID_NAME,
			mmap_mode='r',
			allow_pickle=False,
		),
		prediction_valid_mask=prediction.arrays.valid_mask,
		geometry=read_f3_line_geometry(config.segy_geometry_json),
		classes=classes,
		slices=inspection.selected_slices,
		output_dir=selected_dir,
		config=config.figure,
		metrics_by_slice=_metrics_by_slice(slices),
	)
	figure_paths = (*aggregate_figures, *visualization.png_paths)
	payload = build_f3_lithology_voxel_report_payload(
		metrics=metrics,
		boundary_metrics=boundary,
		boundary_region_rows=regions,
		per_slice_rows=slices,
		prediction_metadata=prediction.metadata,
		evaluation_metadata=evaluation,
		supervision_metadata=supervision,
		figure_paths=figure_paths,
		output_dir=output_dir,
	)
	markdown_path = output_dir / REPORT_MARKDOWN
	json_path = output_dir / REPORT_JSON
	markdown_path.write_text(
		render_f3_lithology_voxel_report_markdown(payload), encoding='utf-8'
	)
	_write_json(json_path, payload)
	return F3LithologyVoxelReportResult(
		markdown_path, json_path, tuple(figure_paths), payload
	)


def build_f3_lithology_voxel_report_payload(  # noqa: PLR0913
	*,
	metrics: Mapping[str, object],
	boundary_metrics: Mapping[str, object],
	boundary_region_rows: Sequence[Mapping[str, object]],
	per_slice_rows: Sequence[Mapping[str, object]],
	prediction_metadata: Mapping[str, object],
	evaluation_metadata: Mapping[str, object],
	supervision_metadata: Mapping[str, object],
	figure_paths: Sequence[Path] = (),
	output_dir: Path | None = None,
) -> dict[str, object]:
	"""Assemble the complete standard-JSON report contract."""
	kind = prediction_metadata.get('prediction_kind')
	if kind not in {'token_projection_nearest', 'frozen_embedding_decoder'}:
		raise ValueError(f'unsupported voxel prediction kind: {kind!r}')
	prediction_label = (
		'V0 nearest token projection'
		if kind == 'token_projection_nearest'
		else 'V1 learned frozen-embedding decoder'
	)
	decoder = _report_decoder_identity(
		kind=kind,
		prediction_metadata=prediction_metadata,
		evaluation_metadata=evaluation_metadata,
	)
	monitored = [
		_monitored_class_row(
			class_id,
			metrics=metrics,
			boundary=boundary_metrics,
			evaluation=evaluation_metadata,
		)
		for class_id in (3, 5)
	]
	return {
		'artifact_type': 'f3_lithology_voxel_report',
		'schema_version': 1,
		'prediction': {
			'kind': kind,
			'label': prediction_label,
			'decoder': decoder,
			'model_tag': prediction_metadata.get('model_tag'),
			'source_identity': prediction_metadata.get('source_identity'),
			'inputs': prediction_metadata.get('inputs'),
		},
		'supervision': {
			'split_strategy': supervision_metadata.get('split_strategy'),
			'split_codes': supervision_metadata.get('split_codes'),
			'validation_precedence': True,
			'precedence_contract': (
				'validation voxels replace overlapping train voxels'
			),
			'aggregation': evaluation_metadata.get('aggregation'),
		},
		'overall_voxel_metrics': dict(metrics),
		'boundary_metrics': dict(boundary_metrics),
		'boundary_region_metrics': [dict(row) for row in boundary_region_rows],
		'monitored_classes': monitored,
		'per_slice_metrics': [dict(row) for row in per_slice_rows],
		'figures': [
			_relative_figure_path(path, output_dir=output_dir) for path in figure_paths
		],
		'known_limitations': list(KNOWN_LIMITATIONS),
	}


def _report_decoder_identity(
	*,
	kind: object,
	prediction_metadata: Mapping[str, object],
	evaluation_metadata: Mapping[str, object],
) -> dict[str, object]:
	if kind == 'token_projection_nearest':
		if evaluation_metadata.get('decoder_architecture') is not None:
			raise ValueError('V0 evaluation must not record decoder_architecture')
		return {
			'spec': V0_PREDICTION_SPEC,
			'learned': False,
			'representation': 'voxel-shaped token projection',
			'upsample_mode': 'nearest',
			'normalization': 'N/A',
			'hidden_channels': 'N/A',
			'upsample_factors': 'N/A',
		}
	if kind != 'frozen_embedding_decoder':
		raise ValueError(f'unsupported voxel prediction kind: {kind!r}')
	prediction_architecture = validate_voxel_decoder_architecture_mapping(
		prediction_metadata.get('decoder_architecture'),
		field_prefix='prediction decoder_architecture',
	)
	evaluation_architecture = validate_voxel_decoder_architecture_mapping(
		evaluation_metadata.get('decoder_architecture'),
		field_prefix='evaluation decoder_architecture',
	)
	if prediction_architecture != evaluation_architecture:
		raise ValueError('voxel report prediction/evaluation decoder identity mismatch')
	return {**evaluation_architecture, 'learned': True}


def render_f3_lithology_voxel_report_markdown(
	payload: Mapping[str, object],
) -> str:
	"""Render the report with explicit aggregate-versus-slice metric labels."""
	prediction = _mapping(payload.get('prediction'), 'prediction')
	supervision = _mapping(payload.get('supervision'), 'supervision')
	overall = _mapping(payload.get('overall_voxel_metrics'), 'overall metrics')
	boundary = _mapping(payload.get('boundary_metrics'), 'boundary metrics')
	lines = [
		'# F3 voxel lithology report',
		'',
		'## Prediction identity',
		'',
		f'- prediction kind: {prediction.get("kind")} ({prediction.get("label")})',
		f'- model tag: {prediction.get("model_tag")}',
		f'- decoder spec: {_mapping(prediction.get("decoder"), "decoder").get("spec")}',
		'- upsample mode: '
		f'{_mapping(prediction.get("decoder"), "decoder").get("upsample_mode")}',
		'- normalization: '
		f'{_mapping(prediction.get("decoder"), "decoder").get("normalization")}',
		'- hidden channels: '
		f'{_mapping(prediction.get("decoder"), "decoder").get("hidden_channels")}',
		'- upsample factors: '
		f'{_mapping(prediction.get("decoder"), "decoder").get("upsample_factors")}',
		'- source model and embedding identities are recorded in '
		'`report.json` under `prediction.source_identity`.',
		'- decoder identity is recorded there for V1; V0 has no learned decoder.',
		'',
		'## Supervision and aggregation',
		'',
		'- evaluated split: validation supervision (code 2)',
		f'- precedence: {supervision.get("precedence_contract")}',
		'- aggregate metrics count each unique validation voxel once.',
		'- per-slice rows evaluate each displayed plane independently; '
		'intersections may repeat.',
		'',
		'## Overall voxel metrics (unique validation voxels)',
		'',
		'| metric | value |',
		'|---|---:|',
	]
	for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'mean_iou'):
		lines.append(f'| {key} | {_format_metric(overall.get(key))} |')
	lines.extend(
		[
			'',
			'## Boundary metrics',
			'',
			'| tolerance | precision | recall | F1 |',
			'|---:|---:|---:|---:|',
		]
	)
	for tolerance in _boundary_tolerances(boundary):
		lines.append(
			'| {} | {} | {} | {} |'.format(
				tolerance,
				_format_metric(
					boundary.get(f'vertical_boundary_precision_at_{tolerance}')
				),
				_format_metric(
					boundary.get(f'vertical_boundary_recall_at_{tolerance}')
				),
				_format_metric(boundary.get(f'vertical_boundary_f1_at_{tolerance}')),
			)
		)
	lines.extend(
		[
			'',
			'## Monitored classes 3 and 5',
			'',
			'| class | status | support | F1 | IoU | boundary recall |',
			'|---:|---|---:|---:|---:|---:|',
		]
	)
	for row in cast('Sequence[Mapping[str, object]]', payload['monitored_classes']):
		monitored_template = (
			'| {class_id} | {status} | {support} | {f1} | {iou} | {boundary_recall} |'
		)
		lines.append(
			monitored_template.format(
				**{key: _format_metric(value) for key, value in row.items()}
			)
		)
	lines.extend(
		[
			'',
			'## Boundary-region versus interior metrics',
			'',
			'| region | radius | voxels | macro F1 | mean IoU |',
			'|---|---:|---:|---:|---:|',
		]
	)
	for row in cast(
		'Sequence[Mapping[str, object]]', payload['boundary_region_metrics']
	):
		region_label = f'| {row.get("region")} | {row.get("radius")} |'
		lines.append(
			f'{region_label} {row.get("voxel_count")} | '
			f'{_format_metric(row.get("macro_f1"))} | '
			f'{_format_metric(row.get("mean_iou"))} |'
		)
	lines.extend(
		[
			'',
			'## Per-slice metrics (plane-level; not aggregate)',
			'',
			'| slice | voxels | accuracy | macro F1 | mean IoU |',
			'|---|---:|---:|---:|---:|',
		]
	)
	for row in cast('Sequence[Mapping[str, object]]', payload['per_slice_metrics']):
		slice_label = f'{row.get("slice_type")} {row.get("slice_index")}'
		lines.append(
			f'| {slice_label} | {row.get("voxel_count")} | '
			f'{_format_metric(row.get("accuracy"))} | '
			f'{_format_metric(row.get("macro_f1"))} | '
			f'{_format_metric(row.get("mean_iou"))} |'
		)
	lines.extend(['', '## Figures', ''])
	for path in cast('Sequence[str]', payload['figures']):
		lines.append(f'- [{Path(path).name}]({path})')
	lines.extend(['', '## Known limitations', ''])
	for limitation in cast('Sequence[str]', payload['known_limitations']):
		lines.append(f'- {limitation}')
	return '\n'.join(lines) + '\n'


def publish_f3_lithology_voxel_report(
	result: F3LithologyVoxelReportResult,
	*,
	config: F3LithologyVoxelReportConfig,
) -> tuple[Path, ...]:
	"""Publish report, numeric evaluation artifacts, and PNG figures."""
	policy = config.publish
	if not policy.enabled:
		return ()
	evaluation = _read_json_object(
		config.evaluation_input_dir / EVALUATION_METADATA_JSON
	)
	_validate_evaluation_output_identities(
		evaluation, evaluation_input_dir=config.evaluation_input_dir
	)
	prediction = _mapping(result.payload.get('prediction'), 'prediction')
	decoder = _mapping(prediction.get('decoder'), 'prediction decoder')
	prediction_spec = (
		V0_PREDICTION_SPEC
		if prediction.get('kind') == 'token_projection_nearest'
		else str(decoder.get('spec'))
	)
	output_dir = policy.output_dir or default_f3_lithology_voxel_publish_dir(
		reports_root=policy.reports_root,
		dataset_version=config.dataset.get('version', 'facies_benchmark_v1'),
		model_tag=str(prediction.get('model_tag') or 'unknown_model'),
		prediction_spec=prediction_spec,
	)
	sources = [
		(result.report_markdown, Path(REPORT_MARKDOWN)),
		(result.report_json, Path(REPORT_JSON)),
		*(
			(config.evaluation_input_dir / name, Path(name))
			for name in VOXEL_EVALUATION_PUBLISH_FILES
		),
	]
	for path in result.figure_paths:
		try:
			relative = path.relative_to(config.output_dir)
		except ValueError as exc:
			raise ValueError(f'report figure is outside output_dir: {path}') from exc
		if relative.suffix.lower() != '.png' or relative.parts[0] != FIGURES_DIR:
			raise ValueError(f'unexpected report figure publish path: {relative}')
		sources.append((path, relative))
	entries = tuple(
		(source, output_dir / relative_target) for source, relative_target in sources
	)
	_preflight_voxel_report_publish_entries(
		entries,
		max_file_size_bytes=policy.max_file_size_bytes,
		overwrite=policy.overwrite,
	)
	for source, target in entries:
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(source, target)
	return tuple(target for _, target in entries)


def _preflight_voxel_report_publish_entries(
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


def default_f3_lithology_voxel_publish_dir(
	*, reports_root: Path, dataset_version: str, model_tag: str, prediction_spec: str
) -> Path:
	"""Return the documented lightweight voxel result location."""
	return (
		reports_root
		/ 'f3'
		/ _path_component(dataset_version)
		/ 'voxel_lithology'
		/ _path_component(model_tag)
		/ _path_component(prediction_spec)
	)


def _write_aggregate_figures(  # noqa: PLR0913
	output_dir: Path,
	*,
	metrics: Mapping[str, object],
	boundary: Mapping[str, object],
	regions: Sequence[Mapping[str, object]],
	classes: Sequence[F3ClassInfo],
	config: F3LithologyVoxelFigureConfig,
) -> tuple[Path, ...]:
	import matplotlib.pyplot as plt  # noqa: PLC0415

	paths = tuple(
		output_dir / name
		for name in (
			'confusion_matrix.png',
			'per_class_f1_iou.png',
			'boundary_f1_by_tolerance.png',
			'boundary_region_metrics.png',
		)
	)
	class_ids = [item.class_id for item in classes]
	class_names = [item.class_name for item in classes]
	matrix = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
	canvas, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
	image = ax.imshow(matrix, cmap='Blues')
	for row, column in np.ndindex(matrix.shape):
		ax.text(column, row, str(matrix[row, column]), ha='center', va='center')
	ax.set_xticks(range(len(class_ids)), class_names, rotation=45, ha='right')
	ax.set_yticks(range(len(class_ids)), class_names)
	ax.set_xlabel('predicted class')
	ax.set_ylabel('true class')
	ax.set_title('unique validation voxel confusion matrix')
	canvas.colorbar(image, ax=ax)
	_save_close(canvas, paths[0], dpi=config.dpi)

	f1 = _metric_mapping(metrics, 'per_class_f1')
	iou = _metric_mapping(metrics, 'per_class_iou')
	x = np.arange(len(class_ids))
	canvas, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
	ax.bar(
		x - 0.2,
		[_float_or_nan(f1.get(str(key))) for key in class_ids],
		0.4,
		label='F1',
	)
	ax.bar(
		x + 0.2,
		[_float_or_nan(iou.get(str(key))) for key in class_ids],
		0.4,
		label='IoU',
	)
	ax.set_xticks(x, class_names, rotation=45, ha='right')
	ax.set_ylim(0.0, 1.0)
	ax.set_title('per-class unique validation voxel metrics')
	ax.legend()
	_save_close(canvas, paths[1], dpi=config.dpi)

	tolerances = _boundary_tolerances(boundary)
	canvas, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
	ax.plot(
		tolerances,
		[
			_float_or_nan(boundary.get(f'vertical_boundary_f1_at_{item}'))
			for item in tolerances
		],
		marker='o',
	)
	ax.set_xlabel('tolerance (samples)')
	ax.set_ylabel('boundary F1')
	ax.set_ylim(0.0, 1.0)
	ax.set_title('vertical boundary F1')
	_save_close(canvas, paths[2], dpi=config.dpi)

	labels = [f'{row.get("region")} r={row.get("radius")}' for row in regions]
	canvas, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
	position = np.arange(len(regions))
	ax.bar(
		position - 0.2,
		[_float_or_nan(row.get('macro_f1')) for row in regions],
		0.4,
		label='macro F1',
	)
	ax.bar(
		position + 0.2,
		[_float_or_nan(row.get('mean_iou')) for row in regions],
		0.4,
		label='mean IoU',
	)
	ax.set_xticks(position, labels, rotation=45, ha='right')
	ax.set_ylim(0.0, 1.0)
	ax.set_title('boundary-region and interior metrics')
	ax.legend()
	_save_close(canvas, paths[3], dpi=config.dpi)
	return paths


def _save_close(canvas: object, path: Path, *, dpi: int) -> None:
	import matplotlib.pyplot as plt  # noqa: PLC0415

	canvas.savefig(path, dpi=dpi, facecolor='white')
	plt.close(canvas)


def _selected_slice_pairs(
	config: F3LithologyVoxelReportConfig,
	*,
	slice_rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, int], ...]:
	if config.selected_slices:
		selected = tuple(
			(slice_type, int(index))
			for slice_type in ('inline', 'crossline')
			for index in config.selected_slices.get(slice_type, ())
		)
		available = {
			(str(row['slice_type']), int(row['slice_index'])) for row in slice_rows
		}
		missing = tuple(item for item in selected if item not in available)
		if missing:
			raise ValueError(
				'report.selected_slices must be validation slices present in '
				f'validation_slice_metrics.csv; missing={missing!r}'
			)
		return selected
	available = {
		(str(row['slice_type']), int(row['slice_index'])) for row in slice_rows
	}
	if available:
		return tuple(sorted(available))
	return tuple(
		(record.slice_type, record.slice_index)
		for record in load_f3_slice_split_records(config.png_label_inventory)
		if record.split == 'validation'
	)


def _metrics_by_slice(
	rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], Mapping[str, object]]:
	return {(str(row['slice_type']), int(row['slice_index'])): row for row in rows}


def _monitored_class_row(
	class_id: int,
	*,
	metrics: Mapping[str, object],
	boundary: Mapping[str, object],
	evaluation: Mapping[str, object],
) -> dict[str, object]:
	class_ids = metrics.get('class_ids')
	if not isinstance(class_ids, Sequence) or class_id not in class_ids:
		return {
			'class_id': class_id,
			'status': 'missing_class',
			'support': None,
			'f1': None,
			'iou': None,
			'boundary_recall': None,
		}
	support = _metric_mapping(metrics, 'per_class_support').get(str(class_id))
	policy = evaluation.get('policy')
	tolerances = (
		policy.get('boundary_tolerances') if isinstance(policy, Mapping) else None
	)
	primary = (
		max(cast('Sequence[int]', tolerances))
		if tolerances
		else max(_boundary_tolerances(boundary), default=0)
	)
	zero = support in {None, 0, '0', ''}
	return {
		'class_id': class_id,
		'status': 'zero_support' if zero else 'ok',
		'support': 0 if zero else support,
		'f1': None
		if zero
		else _metric_mapping(metrics, 'per_class_f1').get(str(class_id)),
		'iou': None
		if zero
		else _metric_mapping(metrics, 'per_class_iou').get(str(class_id)),
		'boundary_recall': boundary.get(
			f'vertical_boundary_class_{class_id}_recall_at_{primary}'
		),
	}


def _boundary_tolerances(boundary: Mapping[str, object]) -> tuple[int, ...]:
	prefix = 'vertical_boundary_f1_at_'
	return tuple(
		sorted(
			int(key.removeprefix(prefix))
			for key in boundary
			if key.startswith(prefix) and key.removeprefix(prefix).isdigit()
		)
	)


def _validate_input_files(config: F3LithologyVoxelReportConfig) -> None:
	paths = (
		config.seismic_volume,
		config.label_volume,
		config.class_info,
		config.png_label_inventory,
		config.segy_geometry_json,
		config.voxel_dataset_input_dir / GRID_NAME,
		config.voxel_dataset_input_dir / METADATA_NAME,
		*(
			config.evaluation_input_dir / name
			for name in VOXEL_EVALUATION_PUBLISH_FILES
		),
	)
	for path in paths:
		if not path.is_file():
			raise FileNotFoundError(f'missing voxel report input: {path}')


def _validate_identity_summary(
	prediction: Mapping[str, object],
	*,
	evaluation: Mapping[str, object],
	supervision: Mapping[str, object],
	config: F3LithologyVoxelReportConfig,
) -> None:
	if evaluation.get('prediction_kind') != prediction.get('prediction_kind'):
		raise ValueError('voxel report prediction/evaluation kind mismatch')
	if evaluation.get('model_tag') != prediction.get('model_tag'):
		raise ValueError('voxel report prediction/evaluation model mismatch')
	_report_decoder_identity(
		kind=prediction.get('prediction_kind'),
		prediction_metadata=prediction,
		evaluation_metadata=evaluation,
	)
	if evaluation.get('dataset') != supervision.get('dataset'):
		raise ValueError('voxel report evaluation/supervision dataset mismatch')
	if evaluation.get('dataset') != dict(config.dataset):
		raise ValueError('voxel report evaluation/config dataset mismatch')
	inputs = _mapping(evaluation.get('inputs'), 'evaluation inputs')
	for name, path in (
		(
			'prediction_metadata',
			config.prediction_input_dir / PREDICTION_METADATA_NAME,
		),
		(
			'voxel_predictions',
			config.prediction_input_dir / VOXEL_PREDICTIONS_NAME,
		),
		(
			'voxel_confidence',
			config.prediction_input_dir / PREDICTION_CONFIDENCE_NAME,
		),
		(
			'voxel_valid_mask',
			config.prediction_input_dir / PREDICTION_VALID_MASK_NAME,
		),
		('voxel_dataset_metadata', config.voxel_dataset_input_dir / METADATA_NAME),
		('voxel_split_grid', config.voxel_dataset_input_dir / GRID_NAME),
		('label_volume', config.label_volume),
		('png_label_inventory', config.png_label_inventory),
		('segy_geometry_json', config.segy_geometry_json),
		('class_info', config.class_info),
	):
		_validate_evaluation_input_identity(inputs.get(name), path=path, label=name)
	source_label_segy = _mapping(
		inputs.get('source_label_segy'), 'evaluation input source_label_segy'
	)
	labels = _mapping(supervision.get('labels'), 'supervision labels')
	path_value = labels.get('source_label_segy')
	if not isinstance(path_value, str):
		raise TypeError('supervision labels.source_label_segy must be a string')
	_validate_evaluation_input_identity(
		source_label_segy,
		path=Path(path_value),
		label='source_label_segy',
	)


def _validate_evaluation_output_identities(
	evaluation: Mapping[str, object], *, evaluation_input_dir: Path
) -> None:
	if evaluation.get('artifact_type') != 'f3_lithology_voxel_evaluation':
		raise ValueError('invalid voxel evaluation artifact_type')
	if evaluation.get('schema_version') != 2:
		raise ValueError('unsupported voxel evaluation schema_version')
	outputs = _mapping(evaluation.get('outputs'), 'evaluation outputs')
	for name in EVALUATION_OUTPUT_FILES:
		identity = _mapping(outputs.get(name), f'evaluation output {name}')
		path = evaluation_input_dir / name
		recorded_path = identity.get('path')
		if not isinstance(recorded_path, str) or Path(recorded_path).resolve(
			strict=False
		) != path.resolve(strict=False):
			raise ValueError(
				f'voxel report evaluation output identity mismatch: {name} path'
			)
		sha256 = identity.get('sha256')
		if not isinstance(sha256, str) or sha256 != file_sha256(path):
			raise ValueError(
				f'voxel report evaluation output identity mismatch: {name} hash'
			)


def _validate_evaluation_input_identity(
	value: object, *, path: Path, label: str
) -> None:
	identity = _mapping(value, f'evaluation input {label}')
	recorded_path = identity.get('path')
	if not isinstance(recorded_path, str) or Path(recorded_path).resolve(
		strict=False
	) != path.resolve(strict=False):
		raise ValueError(
			f'voxel report evaluation input identity mismatch: {label} path'
		)
	sha256 = identity.get('sha256')
	if not isinstance(sha256, str) or sha256 != file_sha256(path):
		raise ValueError(
			f'voxel report evaluation input identity mismatch: {label} hash'
		)


def _read_classes(path: Path) -> tuple[F3ClassInfo, ...]:
	from seis_ssl_cluster.f3.lithology.tokens import (  # noqa: PLC0415
		read_f3_lithology_class_info,
	)

	return read_f3_lithology_class_info(path)


def _read_json_object(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj, parse_constant=_reject_json_constant)
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON input must contain an object: {path}')
	return cast('Mapping[str, object]', payload)


def _read_csv(path: Path) -> list[dict[str, object]]:
	with path.open(newline='', encoding='utf-8') as file_obj:
		return [dict(row) for row in csv.DictReader(file_obj)]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	_validate_finite(payload)
	path.write_text(
		json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _commit_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
	if not target.exists():
		staging.replace(target)
		return
	if not overwrite:
		raise FileExistsError(f'refusing to overwrite existing output: {target}')
	backup = target.with_name(f'.{target.name}.backup')
	if backup.exists():
		shutil.rmtree(backup)
	target.replace(backup)
	try:
		staging.replace(target)
	except BaseException:
		backup.replace(target)
		raise
	shutil.rmtree(backup)


def _validate_finite(value: object) -> None:
	if isinstance(value, float) and not math.isfinite(value):
		raise ValueError('voxel report contains NaN or Infinity')
	if isinstance(value, Mapping):
		for item in value.values():
			_validate_finite(item)
	elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
		for item in value:
			_validate_finite(item)


def _metric_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = payload.get(key)
	return value if isinstance(value, Mapping) else {}


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return cast('Mapping[str, object]', value)


def _relative_figure_path(path: Path, *, output_dir: Path | None) -> str:
	if output_dir is None:
		return path.as_posix()
	return path.relative_to(output_dir).as_posix()


def _format_metric(value: object) -> str:
	if value is None or value == '':
		return 'n/a'
	if isinstance(value, int | float) and not isinstance(value, bool):
		return f'{float(value):.4f}'
	try:
		return f'{float(str(value)):.4f}'
	except ValueError:
		return str(value)


def _float_or_nan(value: object) -> float:
	return (
		float('nan') if value in {None, ''} else float(cast('str | int | float', value))
	)


def _path_component(value: str) -> str:
	if not value or value in {'.', '..'} or '/' in value or '\\' in value:
		raise ValueError(f'invalid publish path component: {value!r}')
	return value


def _reject_json_constant(value: str) -> None:
	raise ValueError(f'non-standard JSON constant: {value}')


__all__ = [
	'KNOWN_LIMITATIONS',
	'VOXEL_REPORT_PUBLISH_SUFFIXES',
	'F3LithologyVoxelPublishConfig',
	'F3LithologyVoxelReportConfig',
	'F3LithologyVoxelReportResult',
	'build_f3_lithology_voxel_report',
	'build_f3_lithology_voxel_report_payload',
	'default_f3_lithology_voxel_publish_dir',
	'publish_f3_lithology_voxel_report',
	'render_f3_lithology_voxel_report_markdown',
]
