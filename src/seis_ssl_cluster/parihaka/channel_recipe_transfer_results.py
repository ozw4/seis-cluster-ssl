"""Paired Parihaka Channel results for a transferred representation recipe."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_results import (
	MODELS,
	ChannelSummaryConfig,
	channel_summary_config_from_mapping,
	inspect_channel_benchmark_results,
	inspect_channel_model_results,
)

PRIMARY_METRIC = 'test.channel_iou'
OUTPUT_NAMES = ('comparison.csv', 'summary.json', 'summary.md')


@dataclass(frozen=True)
class ChannelRecipeTransferSummaryConfig:
	"""Resolved inputs and outputs for a three-model recipe comparison."""

	learned_runs_root: Path
	legacy_random_runs_root: Path
	output_dir: Path
	candidate_model_id: str
	reference_model_id: str
	legacy_random_model_id: str


def channel_recipe_transfer_summary_config_from_mapping(
	config: Mapping[str, object],
) -> ChannelRecipeTransferSummaryConfig:
	"""Resolve summary settings from a Channel downstream configuration."""
	learned = channel_summary_config_from_mapping(config)
	comparison = _required_mapping(config, 'comparison', 'config')
	candidate_model_id = _nonempty_string(
		comparison.get('candidate_model_id'), 'comparison.candidate_model_id'
	)
	reference_model_id = _nonempty_string(
		comparison.get('reference_model_id'), 'comparison.reference_model_id'
	)
	legacy_random_model_id = _nonempty_string(
		comparison.get('legacy_random_model_id'),
		'comparison.legacy_random_model_id',
	)
	if candidate_model_id == reference_model_id:
		raise ValueError('candidate and reference model IDs must differ')
	if legacy_random_model_id != 'random':
		raise ValueError(
			'comparison.legacy_random_model_id must be random because the legacy '
			'benchmark inspector validates the canonical pretrained/random pair'
		)
	if candidate_model_id in MODELS or reference_model_id in MODELS:
		raise ValueError('learned model IDs must not use legacy benchmark model IDs')
	return ChannelRecipeTransferSummaryConfig(
		learned_runs_root=learned.runs_root,
		legacy_random_runs_root=_absolute_path(
			comparison.get('legacy_random_runs_root'),
			'comparison.legacy_random_runs_root',
		),
		output_dir=learned.output_dir,
		candidate_model_id=candidate_model_id,
		reference_model_id=reference_model_id,
		legacy_random_model_id=legacy_random_model_id,
	)


def inspect_channel_recipe_transfer_results(
	config: ChannelRecipeTransferSummaryConfig,
) -> dict[str, object]:
	"""Audit both complete run collections and their downstream identity parity."""
	learned, legacy = _inspect_jobs(config)
	return {
		'schema_version': 1,
		'candidate_model_id': config.candidate_model_id,
		'reference_model_id': config.reference_model_id,
		'legacy_random_model_id': config.legacy_random_model_id,
		'complete_learned_jobs': len(learned),
		'complete_legacy_jobs': len(legacy),
		'compared_cells': len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX),
		'downstream_identity_parity': True,
	}


def summarize_channel_recipe_transfer(
	config: ChannelRecipeTransferSummaryConfig,
) -> tuple[Path, Path, Path]:
	"""Write the exact CSV, JSON, and Markdown paired comparison set."""
	learned, legacy = _inspect_jobs(config)
	if config.output_dir.exists():
		raise FileExistsError(
			f'refusing to overwrite summary directory: {config.output_dir}'
		)
	rows = _comparison_rows(config, learned, legacy)
	aggregates = {
		comparison_name: {
			**{
				data_size: _paired_statistics(
					[
						float(row[comparison_name])
						for row in rows
						if row['data_size'] == data_size
					]
				)
				for data_size in DATA_SIZE_PREFIX
			},
			'all_15': _paired_statistics([float(row[comparison_name]) for row in rows]),
		}
		for comparison_name in (
			'candidate_minus_reference',
			'candidate_minus_random',
		)
	}
	payload = {
		'schema_version': 1,
		'summary_name': 'parihaka_channel_recipe_transfer',
		'primary_metric': PRIMARY_METRIC,
		'candidate_model_id': config.candidate_model_id,
		'reference_model_id': config.reference_model_id,
		'legacy_random_model_id': config.legacy_random_model_id,
		'job_counts': {
			'learned': len(learned),
			'legacy_pretrained_random': len(legacy),
			'compared_cells': len(rows),
		},
		'downstream_identity_parity': True,
		'delta_convention': (
			'candidate minus comparator; positive values favor the candidate'
		),
		'model_means_all_15': {
			config.candidate_model_id: statistics.fmean(
				float(row['candidate_channel_iou']) for row in rows
			),
			config.reference_model_id: statistics.fmean(
				float(row['reference_channel_iou']) for row in rows
			),
			config.legacy_random_model_id: statistics.fmean(
				float(row['random_channel_iou']) for row in rows
			),
		},
		'comparison': rows,
		'aggregates': aggregates,
	}
	contents = {
		OUTPUT_NAMES[0]: _comparison_csv(rows),
		OUTPUT_NAMES[1]: json.dumps(payload, indent=2, sort_keys=True) + '\n',
		OUTPUT_NAMES[2]: _summary_markdown(config, payload),
	}
	config.output_dir.mkdir(parents=True)
	paths: list[Path] = []
	for name in OUTPUT_NAMES:
		path = config.output_dir / name
		path.write_text(contents[name], encoding='utf-8')
		paths.append(path)
	if {path.name for path in config.output_dir.iterdir()} != set(OUTPUT_NAMES):
		raise RuntimeError(
			f'summary producer wrote an unexpected file set: {config.output_dir}'
		)
	return cast('tuple[Path, Path, Path]', tuple(paths))


def _inspect_jobs(
	config: ChannelRecipeTransferSummaryConfig,
) -> tuple[
	dict[tuple[str, str, str], Mapping[str, object]],
	dict[tuple[str, str, str], Mapping[str, object]],
]:
	learned_config = ChannelSummaryConfig(
		runs_root=config.learned_runs_root,
		output_dir=config.output_dir,
	)
	learned = inspect_channel_model_results(
		learned_config,
		model_ids=(config.candidate_model_id, config.reference_model_id),
	)
	legacy_config = ChannelSummaryConfig(
		runs_root=config.legacy_random_runs_root,
		output_dir=config.output_dir,
	)
	legacy = inspect_channel_benchmark_results(legacy_config)
	_validate_cross_root_identity(config, learned, legacy)
	return learned, legacy


def _validate_cross_root_identity(
	config: ChannelRecipeTransferSummaryConfig,
	learned: Mapping[tuple[str, str, str], Mapping[str, object]],
	legacy: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> None:
	for data_size in DATA_SIZE_PREFIX:
		for layout_id in LAYOUT_IDS:
			candidate_key = (config.candidate_model_id, layout_id, data_size)
			reference_key = (config.reference_model_id, layout_id, data_size)
			random_key = (config.legacy_random_model_id, layout_id, data_size)
			candidate = _normalized_downstream_identity(
				learned[candidate_key], '/'.join(candidate_key)
			)
			reference = _normalized_downstream_identity(
				learned[reference_key], '/'.join(reference_key)
			)
			random = _normalized_downstream_identity(
				legacy[random_key], '/'.join(random_key)
			)
			if candidate != reference:
				raise ValueError(
					f'{layout_id}/{data_size} candidate/reference downstream '
					'benchmark identity mismatch'
				)
			if candidate != random:
				raise ValueError(
					f'{layout_id}/{data_size} learned/legacy-random downstream '
					'benchmark identity mismatch'
				)


def _normalized_downstream_identity(
	payload: Mapping[str, object], label: str
) -> dict[str, object]:
	identity = _required_mapping(payload, 'benchmark_identity', label)
	embedding = _required_mapping(identity, 'embedding', f'{label}.benchmark_identity')
	common_metadata = _required_mapping(
		embedding,
		'common_metadata',
		f'{label}.benchmark_identity.embedding',
	)
	return {
		**{
			key: value
			for key, value in identity.items()
			if key not in {'model', 'embedding'}
		},
		'embedding': {
			'common_metadata': {
				key: value
				for key, value in common_metadata.items()
				if key != 'pretraining_objective'
			}
		},
	}


def _comparison_rows(
	config: ChannelRecipeTransferSummaryConfig,
	learned: Mapping[tuple[str, str, str], Mapping[str, object]],
	legacy: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for data_size in DATA_SIZE_PREFIX:
		for layout_id in LAYOUT_IDS:
			candidate = _channel_iou(
				learned[(config.candidate_model_id, layout_id, data_size)],
				f'{config.candidate_model_id}/{layout_id}/{data_size}',
			)
			reference = _channel_iou(
				learned[(config.reference_model_id, layout_id, data_size)],
				f'{config.reference_model_id}/{layout_id}/{data_size}',
			)
			random = _channel_iou(
				legacy[(config.legacy_random_model_id, layout_id, data_size)],
				f'{config.legacy_random_model_id}/{layout_id}/{data_size}',
			)
			rows.append(
				{
					'data_size': data_size,
					'layout_id': layout_id,
					'candidate_channel_iou': candidate,
					'reference_channel_iou': reference,
					'random_channel_iou': random,
					'candidate_minus_reference': candidate - reference,
					'candidate_minus_random': candidate - random,
				}
			)
	return rows


def _channel_iou(payload: Mapping[str, object], label: str) -> float:
	test = _required_mapping(payload, 'test', label)
	value = test.get('channel_iou')
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{label}.test.channel_iou must be numeric')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label}.test.channel_iou must be finite')
	return result


def _paired_statistics(values: Sequence[float]) -> dict[str, object]:
	if not values:
		raise ValueError('paired statistics require at least one value')
	mean = statistics.fmean(values)
	sample_standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
	standard_error = sample_standard_deviation / math.sqrt(len(values))
	return {
		'n': len(values),
		'mean': mean,
		'median': statistics.median(values),
		'sample_standard_deviation': sample_standard_deviation,
		'wins': sum(value > 0.0 for value in values),
		'ties': sum(value == 0.0 for value in values),
		'losses': sum(value < 0.0 for value in values),
		'paired_t_statistic': (mean / standard_error if standard_error > 0.0 else None),
	}


def _comparison_csv(rows: Sequence[Mapping[str, object]]) -> str:
	buffer = io.StringIO()
	writer = csv.writer(buffer, lineterminator='\n')
	writer.writerow(
		[
			'data_size',
			'layout_id',
			'candidate_channel_iou',
			'reference_channel_iou',
			'random_channel_iou',
			'candidate_minus_reference',
			'candidate_minus_random',
		]
	)
	for row in rows:
		writer.writerow(
			[
				row['data_size'],
				row['layout_id'],
				*[
					f'{float(row[key]):.9f}'
					for key in (
						'candidate_channel_iou',
						'reference_channel_iou',
						'random_channel_iou',
						'candidate_minus_reference',
						'candidate_minus_random',
					)
				],
			]
		)
	return buffer.getvalue()


def _summary_markdown(
	config: ChannelRecipeTransferSummaryConfig,
	payload: Mapping[str, object],
) -> str:
	aggregates = cast(
		'Mapping[str, Mapping[str, Mapping[str, object]]]',
		payload['aggregates'],
	)
	model_means = cast('Mapping[str, float]', payload['model_means_all_15'])
	rows = cast('Sequence[Mapping[str, object]]', payload['comparison'])
	lines = [
		'# Parihaka Channel transferred local-BT recipe',
		'',
		'Primary metric: `test.channel_iou`; higher is better.',
		'',
		(
			'Deltas are candidate minus comparator. Positive values favor '
			f'`{config.candidate_model_id}`.'
		),
		'',
		'## Mean Channel IoU over 15 cells',
		'',
		'| model | mean |',
		'|---|---:|',
	]
	lines.extend(
		f'| {model_id} | {model_means[model_id]:.6f} |'
		for model_id in (
			config.candidate_model_id,
			config.reference_model_id,
			config.legacy_random_model_id,
		)
	)
	lines.extend(
		[
			'',
			'## Paired deltas',
			'',
			(
				'| comparison | scope | n | mean | median | sample std | '
				'wins/ties/losses | paired t |'
			),
			'|---|---|---:|---:|---:|---:|---:|---:|',
		]
	)
	for comparison_name in (
		'candidate_minus_reference',
		'candidate_minus_random',
	):
		for scope in (*DATA_SIZE_PREFIX, 'all_15'):
			statistics_row = aggregates[comparison_name][scope]
			paired_t = statistics_row['paired_t_statistic']
			rendered_t = 'n/a' if paired_t is None else f'{float(paired_t):.4f}'
			lines.append(
				f'| {comparison_name} | {scope} | {statistics_row["n"]} | '
				f'{float(statistics_row["mean"]):.6f} | '
				f'{float(statistics_row["median"]):.6f} | '
				f'{float(statistics_row["sample_standard_deviation"]):.6f} | '
				f'{statistics_row["wins"]}/{statistics_row["ties"]}/'
				f'{statistics_row["losses"]} | {rendered_t} |'
			)
	lines.extend(
		[
			'',
			'## Per-cell Channel IoU',
			'',
			(
				'| size | layout | candidate | reference | random | '
				'candidate-reference | candidate-random |'
			),
			'|---|---|---:|---:|---:|---:|---:|',
		]
	)
	lines.extend(
		(
			f'| {row["data_size"]} | {row["layout_id"]} | '
			f'{float(row["candidate_channel_iou"]):.6f} | '
			f'{float(row["reference_channel_iou"]):.6f} | '
			f'{float(row["random_channel_iou"]):.6f} | '
			f'{float(row["candidate_minus_reference"]):.6f} | '
			f'{float(row["candidate_minus_random"]):.6f} |'
		)
		for row in rows
	)
	lines.append('')
	return '\n'.join(lines)


def _required_mapping(
	value: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
	item = value.get(key)
	if not isinstance(item, Mapping):
		raise TypeError(f'{label}.{key} must be a mapping')
	return item


def _nonempty_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise ValueError(f'{label} must be a non-empty string')
	return value


def _absolute_path(value: object, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise ValueError(f'{label} must be a non-empty path')
	path = Path(value)
	if not path.is_absolute():
		raise ValueError(f'{label} must be absolute')
	return path


__all__ = [
	'OUTPUT_NAMES',
	'PRIMARY_METRIC',
	'ChannelRecipeTransferSummaryConfig',
	'channel_recipe_transfer_summary_config_from_mapping',
	'inspect_channel_recipe_transfer_results',
	'summarize_channel_recipe_transfer',
]
