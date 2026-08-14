"""Original-split V0/V1 voxel benchmark consolidation."""

from __future__ import annotations

import csv
import json
import math
import shutil
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	BOUNDARY_METRICS_JSON,
	BOUNDARY_REGION_METRICS_CSV,
	EVALUATION_METADATA_JSON,
	EVALUATION_OUTPUT_FILES,
	METRICS_JSON,
)
from seis_ssl_cluster.f3.lithology.voxel_split import VALIDATION_VOXEL_SPLIT
from seis_ssl_cluster.models.voxel_decoder.spec import (
	validate_voxel_decoder_architecture_mapping,
)

_DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

DEFAULT_RESULTS_ROOT = Path('reports')
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
V0_HANDOFF_NAME = 'v0_experiment_handoff.md'
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
	max_file_size_bytes: int = _DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES
	overwrite: bool = True

	def __post_init__(self) -> None:
		"""Validate the publish root before any files are written."""
		if self.enabled and self.output_dir is None:
			raise ValueError(
				'publish.output_dir is required when publishing is enabled'
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
	published_files: tuple[Path, ...] = ()


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
	decoder_architecture: Mapping[str, object] | None


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
		'decoder_architecture': dict(
			cast('Mapping[str, object]', by_key['mae_v1'].decoder_architecture)
		),
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
	_validate_summary_output_availability(config)
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
	published_files = _publish(result, config.publish)
	return replace(result, published_files=published_files)


def _validate_summary_output_availability(
	config: F3LithologyVoxelResultsConfig,
) -> None:
	"""Allow the V0 handoff, but reject existing summary-owned output."""
	if config.overwrite:
		return
	output = config.output_dir
	if not output.exists():
		return
	if not output.is_dir():
		raise FileExistsError(
			f'original voxel summary output is not a directory: {output}'
		)
	conflicts = [
		path
		for path in output.iterdir()
		if not (
			path.name == V0_HANDOFF_NAME
			and path.is_file()
			and not path.is_symlink()
		)
	]
	if conflicts:
		display = ', '.join(str(path) for path in sorted(conflicts))
		raise FileExistsError(
			'refusing partial or conflicting original voxel summary output: '
			f'{display}'
		)


def validate_f3_lithology_voxel_results_inputs(
	config: F3LithologyVoxelResultsConfig,
) -> None:
	"""Validate the six original-split evaluations without writing a summary."""
	_load_and_validate_runs(config)


def validate_f3_lithology_voxel_results_bundle(  # noqa: C901, PLR0912
	root: Path,
) -> None:
	"""Validate a complete lightweight original-split bundle before release."""
	summary_path = root / SUMMARY_JSON
	payload = _read_json(summary_path)
	if payload.get('artifact_type') != 'f3_lithology_voxel_results_summary':
		raise ValueError('original voxel summary artifact_type mismatch')
	if payload.get('schema_version') != 1:
		raise ValueError('original voxel summary schema_version mismatch')
	if payload.get('scope') != {'split': 'original', 'provisional': True}:
		raise ValueError('original voxel summary scope contract mismatch')
	if payload.get('prediction_versions') != {
		'V0': 'voxel-shaped token baseline',
		'V1': 'learned sub-token decoder',
	}:
		raise ValueError('original voxel summary prediction-version contract mismatch')
	identity = _mapping(payload.get('identity'), 'original summary identity')
	grid_hash = identity.get('supervision_split_grid_sha256')
	if not isinstance(grid_hash, str) or len(grid_hash) != 64:
		raise ValueError('original voxel summary split-grid identity is invalid')
	_integer_sequence(identity.get('class_order'), 'original summary class_order')
	voxel_count = identity.get('validation_voxel_count')
	if (
		not isinstance(voxel_count, int)
		or isinstance(voxel_count, bool)
		or voxel_count <= 0
	):
		raise ValueError('original voxel summary validation voxel count is invalid')
	decoder_architecture = validate_voxel_decoder_architecture_mapping(
		identity.get('decoder_architecture'),
		field_prefix='original summary identity.decoder_architecture',
	)
	runs = _mapping_sequence(payload.get('runs'), 'original summary runs')
	keys = {
		f'{item.get("model")} {item.get("version")}'
		for item in runs
	}
	expected_keys = {
		f'{model} {version}'
		for model in REQUIRED_MODELS
		for version in REQUIRED_VERSIONS
	}
	if len(runs) != len(expected_keys) or keys != expected_keys:
		raise ValueError('original voxel summary six-run matrix is incomplete')
	for row in runs:
		expected = decoder_architecture if row.get('version') == 'V1' else {}
		actual = {
			'spec': row.get('decoder_spec'),
			'upsample_mode': row.get('decoder_upsample_mode'),
			'normalization': row.get('decoder_normalization'),
		}
		if actual != {
			'spec': expected.get('spec', ''),
			'upsample_mode': expected.get('upsample_mode', ''),
			'normalization': expected.get('normalization', ''),
		}:
			raise ValueError('original voxel summary run decoder identity mismatch')
	decision = _mapping(
		payload.get('provisional_decision'), 'original summary provisional_decision'
	)
	if decision.get('provisional') is not True or any(
		decision.get(key) not in {'positive', 'hold', 'negative'}
		for key in ('decoder_value', 'm2a_vs_m1_voxel')
	):
		raise ValueError('original voxel summary provisional decision is invalid')
	for key, expected_count in (
		('decoder_comparisons', 3),
		('encoder_comparisons', 3),
		('monitored_class_comparisons', 12),
	):
		rows = _mapping_sequence(payload.get(key), f'original summary {key}')
		if len(rows) != expected_count:
			raise ValueError(f'original voxel summary {key} is incomplete')
	markdown = root / SUMMARY_MARKDOWN
	if not markdown.is_file() or not markdown.read_text(encoding='utf-8').startswith(
		'# F3 original-split voxel benchmark summary\n'
	):
		raise ValueError('original voxel summary markdown contract mismatch')
	for name in TABLE_NAMES:
		_validate_summary_csv(root / 'tables' / name)
	for name in FIGURE_NAMES:
		_validate_png(root / 'figures' / name)


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
	v1_architectures = [
		item.decoder_architecture for item in loaded if item.run.version == 'V1'
	]
	if any(value is None for value in v1_architectures) or any(
		value != v1_architectures[0] for value in v1_architectures[1:]
	):
		raise ValueError('V1 decoder architecture identity mismatch across models')
	return loaded


def _load_run(  # noqa: C901, PLR0912
	run: F3LithologyVoxelResultsRun,
) -> _LoadedRun:
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
	_validate_evaluation_artifact(run.key, run.input_dir, metadata, metrics)
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
	decoder_architecture = None
	if run.version == 'V1':
		decoder_architecture = validate_voxel_decoder_architecture_mapping(
			metadata.get('decoder_architecture'),
			field_prefix=f'{run.key} evaluation decoder_architecture',
		)
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
		run,
		model_tag,
		grid_hash,
		class_order,
		count,
		metrics,
		boundary,
		regions,
		decoder_architecture,
	)


def _validate_evaluation_artifact(
	key: str,
	root: Path,
	metadata: Mapping[str, object],
	metrics: Mapping[str, object],
) -> None:
	if metadata.get('artifact_type') != 'f3_lithology_voxel_evaluation':
		raise ValueError(f'{key} evaluation artifact_type mismatch')
	if metadata.get('schema_version') != 2:
		raise ValueError(f'{key} evaluation schema_version mismatch')
	expected_aggregation = {
		'primary_unit': 'unique_validation_voxel',
		'split_code': int(VALIDATION_VOXEL_SPLIT),
		'intersection_voxels_counted_once': True,
		'per_slice_planes_evaluated_independently': True,
		'voxel_independence_p_values_computed': False,
	}
	aggregation = _mapping(metadata.get('aggregation'), f'{key} metadata.aggregation')
	if aggregation != expected_aggregation:
		raise ValueError(f'{key} unique-validation-voxel aggregation mismatch')
	if metrics.get('aggregation_unit') != 'unique_validation_voxel':
		raise ValueError(f'{key} metrics aggregation_unit mismatch')
	outputs = _mapping(metadata.get('outputs'), f'{key} metadata.outputs')
	for name in EVALUATION_OUTPUT_FILES:
		path = root / name
		if not path.is_file():
			raise FileNotFoundError(f'missing {key} evaluation output: {path}')
		identity = _mapping(outputs.get(name), f'{key} output {name}')
		recorded_path = identity.get('path')
		if not isinstance(recorded_path, str) or Path(recorded_path).resolve(
			strict=False
		) != path.resolve(strict=False):
			raise ValueError(f'{key} output {name} path identity mismatch')
		if identity.get('sha256') != file_sha256(path):
			raise ValueError(f'{key} output {name} hash identity mismatch')


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
	decoder = item.decoder_architecture or {}
	row: dict[str, object] = {
		'model': item.run.model,
		'version': item.run.version,
		'model_tag': item.model_tag,
		'prediction_label': (
			'voxel-shaped token baseline'
			if item.run.version == 'V0'
			else 'learned sub-token decoder'
		),
		'decoder_spec': decoder.get('spec', ''),
		'decoder_upsample_mode': decoder.get('upsample_mode', ''),
		'decoder_normalization': decoder.get('normalization', ''),
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
		'decoder_spec': cand['decoder_spec'],
		'decoder_upsample_mode': cand['decoder_upsample_mode'],
		'decoder_normalization': cand['decoder_normalization'],
	}
	for key in base:
		if key not in cand:
			continue
		if key in {
			'model',
			'version',
			'model_tag',
			'prediction_label',
			'decoder_spec',
			'decoder_upsample_mode',
			'decoder_normalization',
		}:
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
			'baseline_model': comparison['baseline_model'],
			'baseline_version': comparison['baseline_version'],
			'candidate_model': comparison['candidate_model'],
			'candidate_version': comparison['candidate_version'],
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
	runs = cast('Sequence[Mapping[str, object]]', payload['runs'])
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
		'- decoder spec: '
		f'`{_mapping(identity["decoder_architecture"], "decoder")["spec"]}`',
		'- decoder upsample mode: '
		f'`{_mapping(identity["decoder_architecture"], "decoder")["upsample_mode"]}`',
		'- decoder normalization: '
		f'`{_mapping(identity["decoder_architecture"], "decoder")["normalization"]}`',
		'',
		'## Model table',
		'',
		'| model | version | decoder spec | upsample mode | normalization |',
		'|---|---|---|---|---|',
	]
	lines.extend(
		'| {} | {} | {} | {} | {} |'.format(
			row['model'],
			row['version'],
			row.get('decoder_spec', ''),
			row.get('decoder_upsample_mode', ''),
			row.get('decoder_normalization', ''),
		)
		for row in runs
	)
	lines.extend(
		[
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
		'| encoder | decoder spec | upsample mode | normalization | macro F1 | '
		'mean IoU | balanced accuracy | boundary F1 t2 | boundary F1 t4 | '
		'position MAE |',
		'|---|---|---|---|---:|---:|---:|---:|---:|---:|',
		]
	)
	lines.extend(
		'| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
			row['candidate_model'],
			row['decoder_spec'],
			row['decoder_upsample_mode'],
			row['decoder_normalization'],
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
			'| role | comparison | decoder spec | upsample mode | normalization | '
			'macro F1 | mean IoU | boundary F1 t2 | boundary F1 t4 |',
			'|---|---|---|---|---|---:|---:|---:|---:|',
		]
	)
	lines.extend(
		'| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
			row['role'],
			row['comparison'],
			row['decoder_spec'],
			row['decoder_upsample_mode'],
			row['decoder_normalization'],
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
) -> tuple[Path, ...]:
	if not config.enabled:
		return ()
	if config.output_dir is None:  # guarded by the frozen config contract
		raise ValueError('publish.output_dir is required')
	_validate_named_publish_paths(result.table_paths, TABLE_NAMES, label='table')
	_validate_named_publish_paths(result.figure_paths, FIGURE_NAMES, label='figure')
	sources = [
		(result.summary_markdown, Path(SUMMARY_MARKDOWN)),
		(result.summary_json, Path(SUMMARY_JSON)),
		*((path, Path('tables') / path.name) for path in result.table_paths),
		*((path, Path('figures') / path.name) for path in result.figure_paths),
	]
	entries = tuple(
		(source, config.output_dir / relative_target)
		for source, relative_target in sources
	)
	_preflight_voxel_results_publish_entries(
		entries,
		max_file_size_bytes=config.max_file_size_bytes,
		overwrite=config.overwrite,
	)
	for source, target in entries:
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(source, target)
	return tuple(target for _, target in entries)


def _validate_named_publish_paths(
	paths: Sequence[Path], expected_names: Sequence[str], *, label: str
) -> None:
	names = [path.name for path in paths]
	if len(names) != len(set(names)) or set(names) != set(expected_names):
		raise ValueError(
			f'voxel results publish {label} files must be exactly '
			f'{sorted(expected_names)!r}; got {sorted(names)!r}'
		)


def _preflight_voxel_results_publish_entries(
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


def _read_json(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as handle:
		value = json.load(handle)
	if not isinstance(value, Mapping):
		raise TypeError(f'JSON root must be an object: {path}')
	return value


def _mapping_sequence(value: object, label: str) -> tuple[Mapping[str, object], ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a sequence')
	if any(not isinstance(item, Mapping) for item in value):
		raise TypeError(f'{label} entries must be mappings')
	return tuple(cast('Sequence[Mapping[str, object]]', value))


def _validate_summary_csv(path: Path) -> None:
	if not path.is_file():
		raise FileNotFoundError(f'missing original voxel summary table: {path}')
	with path.open(newline='', encoding='utf-8') as handle:
		reader = csv.DictReader(handle)
		rows = list(reader)
	if not reader.fieldnames or not rows:
		raise ValueError(f'original voxel summary table is empty: {path}')


def _validate_png(path: Path) -> None:
	if not path.is_file():
		raise FileNotFoundError(f'missing original voxel summary figure: {path}')
	data = path.read_bytes()
	if not data.startswith(b'\x89PNG\r\n\x1a\n'):
		raise ValueError(f'original voxel summary figure is not PNG: {path}')
	offset = 8
	chunk_types: list[bytes] = []
	while offset < len(data):
		if offset + 12 > len(data):
			raise ValueError(f'original voxel summary PNG is truncated: {path}')
		length = int.from_bytes(data[offset : offset + 4], 'big')
		chunk_type = data[offset + 4 : offset + 8]
		chunk_end = offset + 12 + length
		if chunk_end > len(data):
			raise ValueError(f'original voxel summary PNG is truncated: {path}')
		chunk = data[offset + 8 : offset + 8 + length]
		recorded_crc = int.from_bytes(data[offset + 8 + length : chunk_end], 'big')
		actual_crc = zlib.crc32(chunk, zlib.crc32(chunk_type)) & 0xFFFFFFFF
		if recorded_crc != actual_crc:
			raise ValueError(f'original voxel summary PNG CRC mismatch: {path}')
		chunk_types.append(chunk_type)
		offset = chunk_end
		if chunk_type == b'IEND':
			break
	if (
		offset != len(data)
		or not chunk_types
		or chunk_types[0] != b'IHDR'
		or b'IDAT' not in chunk_types
		or chunk_types[-1] != b'IEND'
	):
		raise ValueError(f'original voxel summary PNG chunk contract mismatch: {path}')


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
	'V0_HANDOFF_NAME',
	'F3LithologyVoxelResultsConfig',
	'F3LithologyVoxelResultsPublishConfig',
	'F3LithologyVoxelResultsResult',
	'F3LithologyVoxelResultsRun',
	'summarize_f3_lithology_voxel_results',
	'validate_f3_lithology_voxel_results_bundle',
	'validate_f3_lithology_voxel_results_inputs',
]
