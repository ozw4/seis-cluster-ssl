'''Paired summary of one Volve horizon recipe arm against the random encoder.

The arm and the baseline are trained by the same frozen decoder contract on the
same layouts, sizes and evaluation support, so every reported delta is a paired
difference over the fifteen (layout, size) cells.  ``macro_mae_samples`` is an
error, so the reported ``random_minus_arm`` delta is positive when the arm is
better than the random encoder.
'''

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	RECIPE_ARM_BASELINE_MODEL_ID,
	plan_volve_horizon_recipe_arm_jobs,
	plan_volve_horizon_recipe_arm_sources,
)

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.volve.horizon_recipe_arm import (
		VolveHorizonRecipeArmConfig,
	)

PRIMARY_METRIC = 'macro_mae_samples'
SECONDARY_METRIC = 'macro_within_2_samples'
SUMMARY_JSON_NAME = 'summary.json'
SUMMARY_MD_NAME = 'summary.md'
PER_CELL_CSV_NAME = 'per_cell.csv'
SUMMARY_OUTPUT_NAMES = (SUMMARY_JSON_NAME, SUMMARY_MD_NAME, PER_CELL_CSV_NAME)
SHARED_IDENTITY_KEYS = (
	'benchmark',
	'canonical_scientific_identity',
	'decoder',
	'objective',
	'optimizer',
	'runtime_precision',
	'training',
)
# Tile records and split plans are cell-specific, so they are compared between
# the arm and the baseline of the same cell rather than across the whole matrix.
PAIRED_IDENTITY_KEYS = (
	'tiles',
	'horizon_split_plan',
	'effective_model_valid_observation_counts',
	'native_horizon_observation_counts',
)


@dataclass(frozen=True)
class VolveHorizonRecipeArmCell:
	'''One completed decoder cell of the paired comparison.'''

	model_id: str
	layout_id: str
	data_size: str
	primary: float
	secondary: float
	best_epoch: int


def inspect_volve_horizon_recipe_arm_results(
	config: VolveHorizonRecipeArmConfig,
) -> dict[str, object]:
	'''Read every configured cell and verify the shared frozen contract.'''
	cells = _load_cells(config)
	_validate_shared_identity(config, cells)
	return {
		'schema_version': 1,
		'benchmark_id': config.benchmark_id,
		'arm_id': config.arm_id,
		'complete_cells': len(cells),
		'expected_cells': len(plan_volve_horizon_recipe_arm_jobs(config)),
	}


def summarize_volve_horizon_recipe_arm(
	config: VolveHorizonRecipeArmConfig,
) -> dict[str, object]:
	'''Write the paired arm-versus-random summary artifacts atomically.'''
	cells = _load_cells(config)
	identity = _validate_shared_identity(config, cells)
	payload = _summary_payload(config, cells, identity)
	config.summary_root.mkdir(parents=True, exist_ok=True)
	outputs: list[str] = []
	for name, text in (
		(SUMMARY_JSON_NAME, json.dumps(payload, indent=1, sort_keys=True) + '\n'),
		(SUMMARY_MD_NAME, _summary_markdown(config, payload)),
		(PER_CELL_CSV_NAME, _per_cell_csv(config, cells)),
	):
		path = config.summary_root / name
		staging = path.with_suffix(path.suffix + '.tmp')
		staging.write_text(text, encoding='utf-8')
		staging.replace(path)
		outputs.append(str(path))
	return {
		'arm_id': config.arm_id,
		'complete_cells': len(cells),
		'outputs': outputs,
		'summary': payload['overall'],
	}


def _load_cells(
	config: VolveHorizonRecipeArmConfig,
) -> dict[tuple[str, str, str], VolveHorizonRecipeArmCell]:
	cells: dict[tuple[str, str, str], VolveHorizonRecipeArmCell] = {}
	for model_id, layout_id, data_size in plan_volve_horizon_recipe_arm_jobs(config):
		path = (
			config.runs_root
			/ f'model={model_id}'
			/ f'layout={layout_id}'
			/ f'size={data_size}'
			/ 'metrics.json'
		)
		if not path.is_file():
			raise FileNotFoundError(f'missing recipe-arm cell metrics: {path}')
		metrics = _read_json(path)
		primary_common = _required_mapping(
			_required_mapping(metrics, 'test', str(path)),
			'primary_common',
			str(path),
		)
		cells[model_id, layout_id, data_size] = VolveHorizonRecipeArmCell(
			model_id=model_id,
			layout_id=layout_id,
			data_size=data_size,
			primary=_finite(primary_common.get(PRIMARY_METRIC), PRIMARY_METRIC),
			secondary=_finite(
				primary_common.get(SECONDARY_METRIC),
				SECONDARY_METRIC,
			),
			best_epoch=_nonnegative_int(metrics.get('best_epoch'), 'best_epoch'),
		)
	return cells


def _validate_shared_identity(  # noqa: C901
	config: VolveHorizonRecipeArmConfig,
	cells: Mapping[tuple[str, str, str], VolveHorizonRecipeArmCell],
) -> Mapping[str, object]:
	shared: dict[str, object] = {}
	paired: dict[tuple[str, str], Mapping[str, object]] = {}
	for model_id, layout_id, data_size in sorted(cells):
		identity = _required_mapping(
			_read_json(
				config.runs_root
				/ f'model={model_id}'
				/ f'layout={layout_id}'
				/ f'size={data_size}'
				/ 'metrics.json'
			),
			'benchmark_identity',
			f'{model_id}/{layout_id}/{data_size}',
		)
		label = f'{model_id}/{layout_id}/{data_size}'
		if identity.get('benchmark') != config.benchmark_id:
			raise ValueError(f'{label} was produced by another benchmark')
		if identity.get('model') != model_id:
			raise ValueError(f'{label} records a different model identity')
		if identity.get('layout_id') != layout_id:
			raise ValueError(f'{label} records a different layout identity')
		if identity.get('data_size') != data_size:
			raise ValueError(f'{label} records a different data size')
		for key in SHARED_IDENTITY_KEYS:
			value = identity.get(key)
			if key not in shared:
				shared[key] = value
			elif shared[key] != value:
				raise ValueError(f'{label} changed the shared {key} contract')
		cell_key = (layout_id, data_size)
		cell_identity = {key: identity.get(key) for key in PAIRED_IDENTITY_KEYS}
		if cell_key not in paired:
			paired[cell_key] = cell_identity
		elif paired[cell_key] != cell_identity:
			raise ValueError(
				f'{label} does not share the evaluation support of its pair'
			)
	return shared


def _summary_payload(
	config: VolveHorizonRecipeArmConfig,
	cells: Mapping[tuple[str, str, str], VolveHorizonRecipeArmCell],
	identity: Mapping[str, object],
) -> dict[str, object]:
	deltas = {
		(layout_id, data_size): (
			cells[RECIPE_ARM_BASELINE_MODEL_ID, layout_id, data_size].primary
			- cells[config.arm_id, layout_id, data_size].primary
		)
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZE_PREFIX
	}
	by_size = {
		data_size: _delta_statistics(
			[deltas[layout_id, data_size] for layout_id in LAYOUT_IDS]
		)
		for data_size in DATA_SIZE_PREFIX
	}
	overall = _delta_statistics(list(deltas.values()))
	return {
		'schema_version': 1,
		'benchmark_id': config.benchmark_id,
		'arm_id': config.arm_id,
		'baseline_id': RECIPE_ARM_BASELINE_MODEL_ID,
		'primary_metric': PRIMARY_METRIC,
		'delta_convention': (
			'random_minus_arm; positive means the arm has lower error than random'
		),
		'sources': list(plan_volve_horizon_recipe_arm_sources(config)),
		'decoder_initial_state_sha256': _decoder_sha256(identity),
		'model_means': {
			model_id: statistics.fmean(
				[
					cells[model_id, layout_id, data_size].primary
					for layout_id in LAYOUT_IDS
					for data_size in DATA_SIZE_PREFIX
				]
			)
			for model_id in config.model_ids
		},
		'by_size': by_size,
		'overall': overall,
	}


def _delta_statistics(values: Sequence[float]) -> dict[str, object]:
	count = len(values)
	if count == 0:
		raise ValueError('paired delta statistics need at least one cell')
	mean = statistics.fmean(values)
	sample_std = statistics.stdev(values) if count > 1 else 0.0
	standard_error = sample_std / math.sqrt(count) if count > 1 else 0.0
	return {
		'n': count,
		'mean': mean,
		'median': statistics.median(values),
		'sample_std': sample_std,
		'min': min(values),
		'max': max(values),
		'arm_better_count': sum(1 for value in values if value > 0.0),
		'tied_count': sum(1 for value in values if value == 0.0),
		'arm_worse_count': sum(1 for value in values if value < 0.0),
		't_statistic': (mean / standard_error if standard_error > 0.0 else None),
	}


def _summary_markdown(
	config: VolveHorizonRecipeArmConfig,
	payload: Mapping[str, object],
) -> str:
	overall = cast('Mapping[str, object]', payload['overall'])
	by_size = cast('Mapping[str, Mapping[str, object]]', payload['by_size'])
	means = cast('Mapping[str, float]', payload['model_means'])
	lines = [
		f'# Volve horizon recipe arm `{config.arm_id}` vs random',
		'',
		(
			f'Primary metric: `{PRIMARY_METRIC}` on the common test support, '
			'lower is better.'
		),
		'',
		(
			'Delta convention: `random - arm`; a positive delta means the arm '
			'has the lower error.'
		),
		'',
		'## Model means over 15 cells',
		'',
		'| model | mean macro MAE (samples) |',
		'|---|---:|',
	]
	lines.extend(
		f'| {model_id} | {means[model_id]:.6f} |' for model_id in config.model_ids
	)
	lines.extend(
		[
			'',
			'## Paired delta (random - arm)',
			'',
			'| scope | n | mean | median | sample std | min | max | +/0/- | t |',
			'|---|---:|---:|---:|---:|---:|---:|---|---:|',
		]
	)
	for scope, statistics_row in (
		*by_size.items(),
		('all', overall),
	):
		lines.append(_delta_row(scope, statistics_row))
	lines.append('')
	return '\n'.join(lines)


def _delta_row(scope: str, row: Mapping[str, object]) -> str:
	t_statistic = row['t_statistic']
	rendered_t = 'n/a' if t_statistic is None else f'{float(t_statistic):.2f}'
	return (
		f'| {scope} | {row["n"]} | {float(row["mean"]):.6f} | '
		f'{float(row["median"]):.6f} | {float(row["sample_std"]):.6f} | '
		f'{float(row["min"]):.6f} | {float(row["max"]):.6f} | '
		f'{row["arm_better_count"]}/{row["tied_count"]}/{row["arm_worse_count"]} | '
		f'{rendered_t} |'
	)


def _per_cell_csv(
	config: VolveHorizonRecipeArmConfig,
	cells: Mapping[tuple[str, str, str], VolveHorizonRecipeArmCell],
) -> str:
	buffer = io.StringIO()
	writer = csv.writer(buffer, lineterminator='\n')
	writer.writerow(
		[
			'data_size',
			'layout_id',
			'arm_macro_mae_samples',
			'random_macro_mae_samples',
			'random_minus_arm',
			'arm_macro_within_2_samples',
			'random_macro_within_2_samples',
			'arm_best_epoch',
			'random_best_epoch',
		]
	)
	for data_size in DATA_SIZE_PREFIX:
		for layout_id in LAYOUT_IDS:
			arm = cells[config.arm_id, layout_id, data_size]
			baseline = cells[
				RECIPE_ARM_BASELINE_MODEL_ID,
				layout_id,
				data_size,
			]
			writer.writerow(
				[
					data_size,
					layout_id,
					f'{arm.primary:.9f}',
					f'{baseline.primary:.9f}',
					f'{baseline.primary - arm.primary:.9f}',
					f'{arm.secondary:.9f}',
					f'{baseline.secondary:.9f}',
					arm.best_epoch,
					baseline.best_epoch,
				]
			)
	return buffer.getvalue()


def _decoder_sha256(identity: Mapping[str, object]) -> str:
	decoder = identity.get('decoder')
	if not isinstance(decoder, Mapping):
		raise TypeError('benchmark identity decoder must be a mapping')
	value = decoder.get('initial_state_sha256')
	if not isinstance(value, str) or len(value) != 64:
		raise ValueError('benchmark identity decoder SHA-256 is malformed')
	return value


def _read_json(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(f'missing recipe-arm artifact: {path}')
	loaded = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(loaded, Mapping):
		raise TypeError(f'recipe-arm artifact must be a mapping: {path}')
	return cast('Mapping[str, object]', loaded)


def _required_mapping(
	value: Mapping[str, object],
	key: str,
	label: str,
) -> Mapping[str, object]:
	item = value.get(key)
	if not isinstance(item, Mapping):
		raise TypeError(f'{label}.{key} must be a mapping')
	return cast('Mapping[str, object]', item)


def _finite(value: object, label: str) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{label} must be numeric')
	number = float(value)
	if not math.isfinite(number):
		raise ValueError(f'{label} must be finite')
	return number


def _nonnegative_int(value: object, label: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise ValueError(f'{label} must be a non-negative integer')
	return int(value)


__all__ = [
	'PER_CELL_CSV_NAME',
	'PRIMARY_METRIC',
	'SECONDARY_METRIC',
	'SUMMARY_JSON_NAME',
	'SUMMARY_MD_NAME',
	'SUMMARY_OUTPUT_NAMES',
	'VolveHorizonRecipeArmCell',
	'inspect_volve_horizon_recipe_arm_results',
	'summarize_volve_horizon_recipe_arm',
]
