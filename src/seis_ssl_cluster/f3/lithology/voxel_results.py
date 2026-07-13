"""Original-split V0/V1 voxel benchmark consolidation."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	BOUNDARY_METRICS_JSON,
	BOUNDARY_REGION_METRICS_CSV,
	EVALUATION_METADATA_JSON,
	METRICS_JSON,
)
from seis_ssl_cluster.paths import DEFAULT_RESULTS_ROOT, ensure_under_root
from seis_ssl_cluster.results import (
	DEFAULT_MAX_FILE_SIZE_BYTES,
	PublishItem,
	PublishManifest,
	publish_selected_results,
)

REQUIRED_MODELS = ('MAE', 'M1', 'M2-A')
REQUIRED_VERSIONS = ('V0', 'V1')
EXPECTED_MODEL_TAGS = {
	'MAE': 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
	'M1': 'strat_hmm_pretext_m1_k6_topblock1_distill',
	'M2-A': 'strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill',
}
REQUIRED_RUN_KEYS = tuple(
	f'{model.lower().replace("-", "")}_{version.lower()}'
	for model in REQUIRED_MODELS
	for version in REQUIRED_VERSIONS
)
MONITORED_CLASS_IDS = (3, 5)
BOUNDARY_RADII = (2, 4)
BOUNDARY_TOLERANCES = (2, 4)
SUMMARY_JSON = 'voxel_results_summary.json'
SUMMARY_MARKDOWN = 'voxel_results_summary.md'
TABLE_NAMES = (
	'model_metrics.csv',
	'v1_vs_v0_deltas.csv',
	'encoder_pair_deltas.csv',
	'monitored_class_deltas.csv',
)
FIGURE_NAMES = (
	'overall_metrics.png',
	'decoder_value_deltas.png',
	'boundary_metrics.png',
	'monitored_classes.png',
)
PUBLISH_SUFFIXES = frozenset({'.md', '.json', '.csv', '.png'})


@dataclass(frozen=True)
class F3LithologyVoxelResultsRun:
	"""One required encoder/voxel-decoder evaluation artifact."""

	model: str
	version: str
	input_dir: Path

	@property
	def key(self) -> str:
		"""Return the canonical six-run matrix key."""
		return f'{self.model.lower().replace("-", "")}_{self.version.lower()}'


@dataclass(frozen=True)
class F3LithologyVoxelResultsPublishConfig:
	"""Lightweight result publication settings."""

	enabled: bool = False
	results_root: Path = DEFAULT_RESULTS_ROOT
	output_dir: Path | None = None
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
	overwrite: bool = True

	def __post_init__(self) -> None:
		"""Validate the publish root before any files are written."""
		if self.enabled and self.output_dir is None:
			raise ValueError(
				'publish.output_dir is required when publishing is enabled'
			)
		if self.output_dir is not None:
			ensure_under_root(
				self.output_dir,
				root=self.results_root,
				label='publish.output_dir',
			)


@dataclass(frozen=True)
class F3LithologyVoxelResultsConfig:
	"""Resolved inputs and outputs for the original-split summary."""

	runs: tuple[F3LithologyVoxelResultsRun, ...]
	output_dir: Path
	publish: F3LithologyVoxelResultsPublishConfig = field(
		default_factory=F3LithologyVoxelResultsPublishConfig
	)
	overwrite: bool = False


@dataclass(frozen=True)
class F3LithologyVoxelResultsResult:
	"""Files and provisional decisions written by consolidation."""

	summary_json: Path
	summary_markdown: Path
	table_paths: tuple[Path, ...]
	figure_paths: tuple[Path, ...]
	decoder_value: str
	m2a_vs_m1_voxel: str
	publish_manifest: PublishManifest | None = None


@dataclass(frozen=True)
class _LoadedRun:
	run: F3LithologyVoxelResultsRun
	model_tag: str
	split_grid_sha256: str
	class_order: tuple[int, ...]
	validation_voxel_count: int
	metrics: Mapping[str, object]
	boundary: Mapping[str, object]
	regions: Mapping[int, Mapping[str, object]]


def summarize_f3_lithology_voxel_results(
	config: F3LithologyVoxelResultsConfig,
) -> F3LithologyVoxelResultsResult:
	"""Validate all six runs and write an original-split-only comparison."""
	runs = _load_and_validate_runs(config)
	by_key = {item.run.key: item for item in runs}
	model_rows = [_model_row(item) for item in runs]
	decoder_rows = [
		_delta_row(by_key[f'{key}_v0'], by_key[f'{key}_v1'], comparison='V1 - V0')
		for key in ('mae', 'm1', 'm2a')
	]
	pair_specs = (
		('m1_v1', 'm2a_v1', 'M2-A V1 - M1 V1', 'primary'),
		('mae_v1', 'm1_v1', 'M1 V1 - MAE V1', 'secondary'),
		('mae_v1', 'm2a_v1', 'M2-A V1 - MAE V1', 'secondary'),
	)
	pair_rows = [
		_delta_row(by_key[left], by_key[right], comparison=label, role=role)
		for left, right, label, role in pair_specs
	]
	monitored_rows = _monitored_delta_rows(runs, decoder_rows, pair_rows)
	decoder_status = _decoder_value_status(decoder_rows)
	m2a_status = _m2a_vs_m1_status(pair_rows[0], monitored_rows)
	identity = {
		'supervision_split_grid_sha256': runs[0].split_grid_sha256,
		'class_order': list(runs[0].class_order),
		'validation_voxel_count': runs[0].validation_voxel_count,
	}
	payload: dict[str, object] = {
		'artifact_type': 'f3_lithology_voxel_results_summary',
		'schema_version': 1,
		'scope': {'split': 'original', 'provisional': True},
		'prediction_versions': {
			'V0': 'voxel-shaped token baseline',
			'V1': 'learned sub-token decoder',
		},
		'identity': identity,
		'runs': model_rows,
		'decoder_comparisons': decoder_rows,
		'encoder_comparisons': pair_rows,
		'monitored_class_comparisons': monitored_rows,
		'provisional_decision': {
			'provisional': True,
			'decoder_value': decoder_status,
			'm2a_vs_m1_voxel': m2a_status,
		},
		'limitations': [
			'Original split only; these statuses are not robustness claims.',
		],
	}
	output = config.output_dir
	if output.exists() and not config.overwrite:
		raise FileExistsError(f'refusing to overwrite existing output: {output}')
	output.mkdir(parents=True, exist_ok=True)
	table_dir = output / 'tables'
	figure_dir = output / 'figures'
	table_dir.mkdir(exist_ok=True)
	figure_dir.mkdir(exist_ok=True)
	table_paths = _write_tables(
		table_dir, model_rows, decoder_rows, pair_rows, monitored_rows
	)
	figure_paths = _write_figures(figure_dir, model_rows, decoder_rows, monitored_rows)
	summary_json = output / SUMMARY_JSON
	summary_markdown = output / SUMMARY_MARKDOWN
	summary_json.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	summary_markdown.write_text(_render_markdown(payload), encoding='utf-8')
	result = F3LithologyVoxelResultsResult(
		summary_json=summary_json,
		summary_markdown=summary_markdown,
		table_paths=table_paths,
		figure_paths=figure_paths,
		decoder_value=decoder_status,
		m2a_vs_m1_voxel=m2a_status,
	)
	manifest = _publish(result, config.publish)
	return replace(result, publish_manifest=manifest)


def _load_and_validate_runs(
	config: F3LithologyVoxelResultsConfig,
) -> tuple[_LoadedRun, ...]:
	keys = [run.key for run in config.runs]
	missing = sorted(set(REQUIRED_RUN_KEYS) - set(keys))
	extra = sorted(set(keys) - set(REQUIRED_RUN_KEYS))
	duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
	if missing or extra or duplicates or len(keys) != len(REQUIRED_RUN_KEYS):
		raise ValueError(
			'incomplete six-run matrix: '
			f'missing={missing}, extra={extra}, duplicates={duplicates}'
		)
	loaded = tuple(_load_run(run) for run in config.runs)
	for field_name in ('split_grid_sha256', 'class_order', 'validation_voxel_count'):
		values = {getattr(item, field_name) for item in loaded}
		if len(values) != 1:
			raise ValueError(f'six-run {field_name} identity mismatch')
	for model in REQUIRED_MODELS:
		tags = {item.model_tag for item in loaded if item.run.model == model}
		if len(tags) != 1:
			raise ValueError(f'{model} V0/V1 model identity mismatch')
		if tags != {EXPECTED_MODEL_TAGS[model]}:
			raise ValueError(
				f'{model} source model identity mismatch: '
				f'expected {EXPECTED_MODEL_TAGS[model]!r}, got {sorted(tags)!r}'
			)
	return loaded


def _load_run(run: F3LithologyVoxelResultsRun) -> _LoadedRun:  # noqa: C901
	if run.model not in REQUIRED_MODELS or run.version not in REQUIRED_VERSIONS:
		raise ValueError(f'unsupported run identity: {run.model} {run.version}')
	paths = {
		'metrics': run.input_dir / METRICS_JSON,
		'boundary': run.input_dir / BOUNDARY_METRICS_JSON,
		'regions': run.input_dir / BOUNDARY_REGION_METRICS_CSV,
		'metadata': run.input_dir / EVALUATION_METADATA_JSON,
	}
	for label, path in paths.items():
		if not path.is_file():
			raise FileNotFoundError(f'missing {run.key} {label}: {path}')
	metrics = _read_json(paths['metrics'])
	boundary = _read_json(paths['boundary'])
	metadata = _read_json(paths['metadata'])
	expected_kind = (
		'token_projection_nearest'
		if run.version == 'V0'
		else 'frozen_embedding_decoder'
	)
	if metadata.get('prediction_kind') != expected_kind:
		raise ValueError(f'{run.key} prediction kind does not match {run.version}')
	inputs = _mapping(metadata.get('inputs'), f'{run.key} metadata.inputs')
	grid = _mapping(inputs.get('voxel_split_grid'), f'{run.key} voxel_split_grid')
	grid_hash = grid.get('sha256')
	if not isinstance(grid_hash, str) or len(grid_hash) != 64:
		raise ValueError(f'{run.key} split-grid sha256 is invalid')
	class_order = _integer_sequence(metrics.get('class_ids'), f'{run.key} class_ids')
	count = metrics.get('evaluation_voxel_count')
	if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
		raise ValueError(f'{run.key} evaluation_voxel_count must be positive')
	summary = _mapping(metadata.get('summary'), f'{run.key} metadata.summary')
	if summary.get('unique_validation_voxel_count') != count:
		raise ValueError(f'{run.key} validation voxel count metadata mismatch')
	model_tag = metadata.get('model_tag')
	if not isinstance(model_tag, str) or not model_tag:
		raise ValueError(f'{run.key} model_tag is missing')
	regions: dict[int, Mapping[str, object]] = {}
	with paths['regions'].open(newline='', encoding='utf-8') as handle:
		for row in csv.DictReader(handle):
			if row.get('region') == 'boundary' and row.get('radius', '').isdigit():
				regions[int(row['radius'])] = row
	for radius in BOUNDARY_RADII:
		if radius not in regions:
			raise ValueError(f'{run.key} is missing boundary radius {radius}')
	_validate_required_metrics(run.key, metrics, boundary, regions)
	return _LoadedRun(
		run, model_tag, grid_hash, class_order, count, metrics, boundary, regions
	)


def _validate_required_metrics(
	key: str,
	metrics: Mapping[str, object],
	boundary: Mapping[str, object],
	regions: Mapping[int, Mapping[str, object]],
) -> None:
	for name in ('macro_f1', 'mean_iou', 'balanced_accuracy'):
		_optional_metric(metrics.get(name), f'{key}.{name}', required=True)
	for tolerance in BOUNDARY_TOLERANCES:
		_optional_metric(
			boundary.get(f'vertical_boundary_f1_at_{tolerance}'),
			f'{key}.vertical_boundary_f1_at_{tolerance}',
		)
	_optional_metric(
		boundary.get('vertical_boundary_position_mae_at_4'),
		f'{key}.vertical_boundary_position_mae_at_4',
	)
	for radius, row in regions.items():
		for name in ('macro_f1', 'mean_iou'):
			_optional_metric(row.get(name), f'{key}.boundary_r{radius}.{name}')
	for class_id in MONITORED_CLASS_IDS:
		for name in ('per_class_f1', 'per_class_iou'):
			values = _mapping(metrics.get(name), f'{key}.{name}')
			_optional_metric(values.get(str(class_id)), f'{key}.{name}.{class_id}')
		for tolerance in BOUNDARY_TOLERANCES:
			_optional_metric(
				boundary.get(
					f'vertical_boundary_class_{class_id}_recall_at_{tolerance}'
				),
				f'{key}.class_{class_id}_boundary_recall_at_{tolerance}',
			)


def _model_row(item: _LoadedRun) -> dict[str, object]:
	row: dict[str, object] = {
		'model': item.run.model,
		'version': item.run.version,
		'model_tag': item.model_tag,
		'prediction_label': (
			'voxel-shaped token baseline'
			if item.run.version == 'V0'
			else 'learned sub-token decoder'
		),
		'macro_f1': _number(item.metrics.get('macro_f1')),
		'mean_iou': _number(item.metrics.get('mean_iou')),
		'balanced_accuracy': _number(item.metrics.get('balanced_accuracy')),
		'boundary_position_mae': _number(
			item.boundary.get('vertical_boundary_position_mae_at_4')
		),
	}
	for radius in BOUNDARY_RADII:
		row[f'boundary_region_macro_f1_r{radius}'] = _number(
			item.regions[radius].get('macro_f1')
		)
		row[f'boundary_region_mean_iou_r{radius}'] = _number(
			item.regions[radius].get('mean_iou')
		)
	for tolerance in BOUNDARY_TOLERANCES:
		row[f'vertical_boundary_f1_t{tolerance}'] = _number(
			item.boundary.get(f'vertical_boundary_f1_at_{tolerance}')
		)
	for class_id in MONITORED_CLASS_IDS:
		row[f'class_{class_id}_f1'] = _metric_map(
			item.metrics, 'per_class_f1', class_id
		)
		row[f'class_{class_id}_iou'] = _metric_map(
			item.metrics, 'per_class_iou', class_id
		)
		for tolerance in BOUNDARY_TOLERANCES:
			row[f'class_{class_id}_boundary_recall_t{tolerance}'] = _number(
				item.boundary.get(
					f'vertical_boundary_class_{class_id}_recall_at_{tolerance}'
				)
			)
	return row


def _delta_row(
	baseline: _LoadedRun,
	candidate: _LoadedRun,
	*,
	comparison: str,
	role: str = 'decoder',
) -> dict[str, object]:
	base = _model_row(baseline)
	cand = _model_row(candidate)
	row: dict[str, object] = {
		'comparison': comparison,
		'role': role,
		'baseline_model': baseline.run.model,
		'baseline_version': baseline.run.version,
		'candidate_model': candidate.run.model,
		'candidate_version': candidate.run.version,
	}
	for key in base:
		if key not in cand:
			continue
		if key in {'model', 'version', 'model_tag', 'prediction_label'}:
			continue
		row[key] = _delta(base[key], cand[key])
	return row


def _monitored_delta_rows(
	runs: Sequence[_LoadedRun],
	decoder_rows: Sequence[Mapping[str, object]],
	pair_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
	del runs
	return [
		{
			'comparison': comparison['comparison'],
			'role': comparison['role'],
			'class_id': class_id,
			'f1': comparison[f'class_{class_id}_f1'],
			'iou': comparison[f'class_{class_id}_iou'],
			'boundary_recall_t2': comparison[f'class_{class_id}_boundary_recall_t2'],
			'boundary_recall_t4': comparison[f'class_{class_id}_boundary_recall_t4'],
		}
		for comparison in (*decoder_rows, *pair_rows)
		for class_id in MONITORED_CLASS_IDS
	]


def _decoder_value_status(rows: Sequence[Mapping[str, object]]) -> str:
	relevant = [row for row in rows if row.get('candidate_model') in {'M1', 'M2-A'}]
	positive = all(
		_is_positive(row.get('macro_f1'))
		and _is_positive(row.get('mean_iou'))
		and any(
			_is_non_negative(row.get(f'vertical_boundary_f1_t{tol}'))
			for tol in BOUNDARY_TOLERANCES
		)
		and not _all_boundary_worse(row)
		for row in relevant
	)
	if positive:
		return 'positive'
	if relevant and all(_all_metrics_worse(row) for row in relevant):
		return 'negative'
	return 'hold'


def _m2a_vs_m1_status(
	row: Mapping[str, object],
	monitored_rows: Sequence[Mapping[str, object]],
) -> str:
	class_rows = [
		item for item in monitored_rows if item.get('comparison') == 'M2-A V1 - M1 V1'
	]
	primary_non_worse = all(
		_is_non_negative(row.get(metric)) for metric in ('macro_f1', 'mean_iou')
	) and any(_is_positive(row.get(metric)) for metric in ('macro_f1', 'mean_iou'))
	boundary_ok = any(
		_is_non_negative(row.get(f'vertical_boundary_f1_t{tol}'))
		for tol in BOUNDARY_TOLERANCES
	)
	class_pareto = any(
		_is_non_negative(item.get('f1'))
		and _is_non_negative(item.get('iou'))
		and (_is_positive(item.get('f1')) or _is_positive(item.get('iou')))
		for item in class_rows
	)
	if primary_non_worse and boundary_ok and class_pareto:
		return 'positive'
	if _all_metrics_worse(row) and class_rows:
		return 'negative'
	return 'hold'


def _all_boundary_worse(row: Mapping[str, object]) -> bool:
	benefit_keys = [
		*(f'boundary_region_macro_f1_r{radius}' for radius in BOUNDARY_RADII),
		*(f'boundary_region_mean_iou_r{radius}' for radius in BOUNDARY_RADII),
		*(f'vertical_boundary_f1_t{tol}' for tol in BOUNDARY_TOLERANCES),
		*(
			f'class_{class_id}_boundary_recall_t{tolerance}'
			for class_id in MONITORED_CLASS_IDS
			for tolerance in BOUNDARY_TOLERANCES
		),
	]
	benefits = [row.get(key) for key in benefit_keys]
	position = row.get('boundary_position_mae')
	return all(_is_negative(value) for value in benefits) and _is_positive(position)


def _all_metrics_worse(row: Mapping[str, object]) -> bool:
	benefit_keys = [
		'macro_f1',
		'mean_iou',
		'balanced_accuracy',
		*(
			f'class_{class_id}_{metric}'
			for class_id in MONITORED_CLASS_IDS
			for metric in ('f1', 'iou')
		),
	]
	return all(_is_negative(row.get(key)) for key in benefit_keys) and (
		_all_boundary_worse(row)
	)


def _write_tables(
	output_dir: Path,
	model_rows: Sequence[Mapping[str, object]],
	decoder_rows: Sequence[Mapping[str, object]],
	pair_rows: Sequence[Mapping[str, object]],
	monitored_rows: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
	datasets = (model_rows, decoder_rows, pair_rows, monitored_rows)
	paths = []
	for name, rows in zip(TABLE_NAMES, datasets, strict=True):
		path = output_dir / name
		fields = list(rows[0])
		with path.open('w', newline='', encoding='utf-8') as handle:
			writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
			writer.writeheader()
			writer.writerows(rows)
		paths.append(path)
	return tuple(paths)


def _write_figures(
	output_dir: Path,
	model_rows: Sequence[Mapping[str, object]],
	decoder_rows: Sequence[Mapping[str, object]],
	monitored_rows: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
	import matplotlib.pyplot as plt  # noqa: PLC0415

	paths = []
	labels = [f'{row["model"]} {row["version"]}' for row in model_rows]
	for name, title, rows, metrics in (
		(
			FIGURE_NAMES[0],
			'Original-split overall voxel metrics',
			model_rows,
			('macro_f1', 'mean_iou', 'balanced_accuracy'),
		),
		(
			FIGURE_NAMES[1],
			'Learned decoder value (V1 - V0)',
			decoder_rows,
			('macro_f1', 'mean_iou'),
		),
		(
			FIGURE_NAMES[2],
			'Vertical boundary F1',
			model_rows,
			('vertical_boundary_f1_t2', 'vertical_boundary_f1_t4'),
		),
	):
		fig, ax = plt.subplots(figsize=(8, 4))
		x = list(range(len(rows)))
		width = 0.8 / len(metrics)
		for index, metric in enumerate(metrics):
			values = [_plot_value(row.get(metric)) for row in rows]
			ax.bar([value + index * width for value in x], values, width, label=metric)
		row_labels = (
			labels
			if rows is model_rows
			else [str(row['candidate_model']) for row in rows]
		)
		ax.set_xticks(
			[value + width * (len(metrics) - 1) / 2 for value in x], row_labels
		)
		ax.axhline(0, color='black', linewidth=0.7)
		ax.set_title(title)
		ax.legend(fontsize='small')
		fig.tight_layout()
		path = output_dir / name
		fig.savefig(path, dpi=120)
		plt.close(fig)
		paths.append(path)
	fig, ax = plt.subplots(figsize=(9, 4))
	class_rows = [
		row for row in monitored_rows if row['comparison'] == 'M2-A V1 - M1 V1'
	]
	x = list(range(len(class_rows)))
	for index, metric in enumerate(('f1', 'iou', 'boundary_recall_t4')):
		ax.bar(
			[value + index * 0.25 for value in x],
			[_plot_value(row.get(metric)) for row in class_rows],
			0.25,
			label=metric,
		)
	ax.set_xticks(
		[value + 0.25 for value in x],
		[f'class {row["class_id"]}' for row in class_rows],
	)
	ax.axhline(0, color='black', linewidth=0.7)
	ax.set_title('M2-A V1 - M1 V1 monitored classes')
	ax.legend(fontsize='small')
	fig.tight_layout()
	path = output_dir / FIGURE_NAMES[3]
	fig.savefig(path, dpi=120)
	plt.close(fig)
	paths.append(path)
	return tuple(paths)


def _render_markdown(payload: Mapping[str, object]) -> str:
	decision = cast('Mapping[str, object]', payload['provisional_decision'])
	identity = cast('Mapping[str, object]', payload['identity'])
	decoder = cast('Sequence[Mapping[str, object]]', payload['decoder_comparisons'])
	encoders = cast('Sequence[Mapping[str, object]]', payload['encoder_comparisons'])
	classes = cast(
		'Sequence[Mapping[str, object]]', payload['monitored_class_comparisons']
	)
	lines = [
		'# F3 original-split voxel benchmark summary',
		'',
		'V0 is the voxel-shaped token baseline. V1 is the learned sub-token decoder.',
		'',
		'## Shared evaluation identity',
		'',
		'- supervision split-grid SHA-256: '
		f'`{identity["supervision_split_grid_sha256"]}`',
		f'- class order: `{identity["class_order"]}`',
		f'- validation voxel count: `{identity["validation_voxel_count"]}`',
		'',
		'## Provisional decisions',
		'',
		f'- decoder_value: **{decision["decoder_value"]}**',
		f'- m2a_vs_m1_voxel: **{decision["m2a_vs_m1_voxel"]}**',
		'- provisional: **true**',
		'',
		'These statuses use the original split only and are not robustness claims.',
		'',
		'## Q1: learned decoder value (V1 - V0)',
		'',
		'| encoder | macro F1 | mean IoU | balanced accuracy | '
		'boundary F1 t2 | boundary F1 t4 | position MAE |',
		'|---|---:|---:|---:|---:|---:|---:|',
	]
	lines.extend(
		'| {} | {} | {} | {} | {} | {} | {} |'.format(
			row['candidate_model'],
			_format(row.get('macro_f1')),
			_format(row.get('mean_iou')),
			_format(row.get('balanced_accuracy')),
			_format(row.get('vertical_boundary_f1_t2')),
			_format(row.get('vertical_boundary_f1_t4')),
			_format(row.get('boundary_position_mae')),
		)
		for row in decoder
	)
	lines.extend(
		[
			'',
			'Boundary-region radius 2/4 and class 3/5 deltas are retained in '
			'`tables/v1_vs_v0_deltas.csv` and '
			'`tables/monitored_class_deltas.csv`.',
			'',
			'## Q2: representation comparison at voxel resolution',
			'',
			'| role | comparison | macro F1 | mean IoU | boundary F1 t2 | '
			'boundary F1 t4 |',
			'|---|---|---:|---:|---:|---:|',
		]
	)
	lines.extend(
		'| {} | {} | {} | {} | {} | {} |'.format(
			row['role'],
			row['comparison'],
			_format(row.get('macro_f1')),
			_format(row.get('mean_iou')),
			_format(row.get('vertical_boundary_f1_t2')),
			_format(row.get('vertical_boundary_f1_t4')),
		)
		for row in encoders
	)
	lines.extend(
		[
			'',
			'### Monitored class deltas',
			'',
			'| comparison | class | F1 | IoU | boundary recall t2 | '
			'boundary recall t4 |',
			'|---|---:|---:|---:|---:|---:|',
		]
	)
	lines.extend(
		'| {} | {} | {} | {} | {} | {} |'.format(
			row['comparison'],
			row['class_id'],
			_format(row.get('f1')),
			_format(row.get('iou')),
			_format(row.get('boundary_recall_t2')),
			_format(row.get('boundary_recall_t4')),
		)
		for row in classes
	)
	lines.extend(['', 'Full metric rows are available in `tables/`.', ''])
	return '\n'.join(lines)


def _publish(
	result: F3LithologyVoxelResultsResult,
	config: F3LithologyVoxelResultsPublishConfig,
) -> PublishManifest | None:
	if not config.enabled:
		return None
	if config.output_dir is None:  # guarded by the frozen config contract
		raise ValueError('publish.output_dir is required')
	items = [
		PublishItem(result.summary_markdown, Path(SUMMARY_MARKDOWN)),
		PublishItem(result.summary_json, Path(SUMMARY_JSON)),
		*(PublishItem(path, Path('tables') / path.name) for path in result.table_paths),
		*(
			PublishItem(path, Path('figures') / path.name)
			for path in result.figure_paths
		),
	]
	return publish_selected_results(
		items=items,
		output_dir=config.output_dir,
		allowed_suffixes=PUBLISH_SUFFIXES,
		max_file_size_bytes=config.max_file_size_bytes,
		overwrite=config.overwrite,
	)


def _read_json(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as handle:
		value = json.load(handle)
	if not isinstance(value, Mapping):
		raise TypeError(f'JSON root must be an object: {path}')
	return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _integer_sequence(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a sequence')
	if not value or any(
		not isinstance(item, int) or isinstance(item, bool) for item in value
	):
		raise ValueError(f'{label} must contain integers')
	return tuple(value)


def _optional_metric(value: object, label: str, *, required: bool = False) -> None:
	if value in {None, ''} and not required:
		return
	if isinstance(value, str):
		try:
			value = float(value)
		except ValueError as error:
			raise TypeError(f'{label} must be numeric or null') from error
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{label} must be numeric or null')
	if not math.isfinite(float(value)):
		raise ValueError(f'{label} must be finite')


def _number(value: object) -> float | None:
	if value in {None, ''}:
		return None
	return float(cast('str | int | float', value))


def _metric_map(
	metrics: Mapping[str, object], name: str, class_id: int
) -> float | None:
	return _number(_mapping(metrics.get(name), name).get(str(class_id)))


def _delta(baseline: object, candidate: object) -> float | None:
	if baseline is None or candidate is None:
		return None
	return float(cast('float', candidate)) - float(cast('float', baseline))


def _is_positive(value: object) -> bool:
	return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _is_negative(value: object) -> bool:
	return isinstance(value, int | float) and not isinstance(value, bool) and value < 0


def _is_non_negative(value: object) -> bool:
	return isinstance(value, int | float) and not isinstance(value, bool) and value >= 0


def _plot_value(value: object) -> float:
	if isinstance(value, int | float) and not isinstance(value, bool):
		return float(value)
	return math.nan


def _format(value: object) -> str:
	if isinstance(value, int | float) and not isinstance(value, bool):
		return f'{float(value):.6f}'
	return 'NA'


__all__ = [
	'EXPECTED_MODEL_TAGS',
	'FIGURE_NAMES',
	'PUBLISH_SUFFIXES',
	'REQUIRED_MODELS',
	'REQUIRED_VERSIONS',
	'SUMMARY_JSON',
	'SUMMARY_MARKDOWN',
	'TABLE_NAMES',
	'F3LithologyVoxelResultsConfig',
	'F3LithologyVoxelResultsPublishConfig',
	'F3LithologyVoxelResultsResult',
	'F3LithologyVoxelResultsRun',
	'summarize_f3_lithology_voxel_results',
]
