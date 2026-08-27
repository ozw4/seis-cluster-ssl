# ruff: noqa: INP001
"""Summarize validation-only Channel screening for boundary weighting."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

VARIANT_BOUNDARY_SETTINGS = {
	'alpha000_tau1': {
		'boundary_alpha': 0.0,
		'boundary_tau': 1.0,
	},
	'alpha050_tau1': {
		'boundary_alpha': 0.5,
		'boundary_tau': 1.0,
	},
	'alpha100_tau1': {
		'boundary_alpha': 1.0,
		'boundary_tau': 1.0,
	},
}
BRANCHES = {
	'mae': {
		'control': 'mae',
		'variants': {
			'alpha000_tau1': 'mae_hmm_k6',
			'alpha050_tau1': 'mae_hmm_k6_boundary_alpha050_tau1',
			'alpha100_tau1': 'mae_hmm_k6_boundary_alpha100_tau1',
		},
	},
	'local_bt': {
		'control': 'local_barlow_twins',
		'variants': {
			'alpha000_tau1': 'local_barlow_twins_hmm_k6',
			'alpha050_tau1': (
				'local_barlow_twins_hmm_k6_boundary_alpha050_tau1'
			),
			'alpha100_tau1': (
				'local_barlow_twins_hmm_k6_boundary_alpha100_tau1'
			),
		},
	},
}
LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))
VARIANT_ORDER = tuple(VARIANT_BOUNDARY_SETTINGS)
EXISTING_MODELS = frozenset(
	{
		'mae',
		'mae_hmm_k6',
		'local_barlow_twins',
		'local_barlow_twins_hmm_k6',
	}
)
_LAYOUT_SPECIFIC_IDENTITY_KEYS = frozenset(
	{
		'layout_id',
		'train_lines',
		'selection',
		'class_weights',
		'split_class_counts',
		'tile_counts',
	}
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _model_ids() -> tuple[str, ...]:
	model_ids: list[str] = []
	for branch in BRANCHES.values():
		control = branch.get('control')
		variants = _mapping(branch.get('variants'), 'branch variants')
		if not isinstance(control, str):
			raise TypeError('branch control must be a model ID')
		model_ids.append(control)
		model_ids.extend(str(variants[variant]) for variant in VARIANT_ORDER)
	unique_model_ids = tuple(dict.fromkeys(model_ids))
	if len(unique_model_ids) != 8:
		raise AssertionError('screening must define exactly 8 models')
	return unique_model_ids


def _validate_source_identity(
	identity: Mapping[str, Any],
	model: str,
	path: Path,
) -> None:
	embedding = _mapping(identity.get('embedding'), 'benchmark embedding identity')
	embedding_model = embedding.get('model')
	if embedding_model is not None and embedding_model != model:
		raise ValueError(f'{path}: benchmark embedding model identity mismatch')
	model_source = embedding.get('model_source')
	if model_source is None:
		return
	source = _mapping(model_source, 'benchmark embedding model source')
	source_model = source.get('model_id')
	if source_model is not None and source_model != model:
		raise ValueError(f'{path}: benchmark model-source identity mismatch')
	for key in ('checkpoint_path', 'checkpoint_sha256'):
		if key in source and source[key] != embedding.get(key):
			raise ValueError(f'{path}: model-source {key} mismatch')


def read_metrics(
	runs_root: Path,
	model: str,
	layout: str,
	*,
	validation_only: bool,
) -> Mapping[str, Any]:
	"""Read one metrics file and validate its requested job identity."""
	path = (
		runs_root
		/ f'model={model}'
		/ f'layout={layout}'
		/ 'size=medium'
		/ 'metrics.json'
	)
	payload = _mapping(
		json.loads(path.read_text(encoding='utf-8')),
		f'{path} metrics payload',
	)
	if payload.get('model') != model:
		raise ValueError(f'{path}: model identity mismatch')
	if payload.get('layout_id') != layout:
		raise ValueError(f'{path}: layout identity mismatch')
	if payload.get('data_size') != 'medium':
		raise ValueError(f'{path}: data_size must be medium')
	identity = _mapping(payload.get('benchmark_identity'), 'benchmark identity')
	if identity.get('model') != model:
		raise ValueError(f'{path}: benchmark model identity mismatch')
	_validate_source_identity(identity, model, path)
	if validation_only:
		if payload.get('evaluation_mode') != 'validation_only':
			raise ValueError(f'{path}: candidate must be validation-only')
		if 'test' in payload:
			raise ValueError(f'{path}: validation-only metrics contain test results')
	return payload


def paired_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
	"""Return benchmark identity with model/source-specific fields removed."""
	identity = _mapping(payload.get('benchmark_identity'), 'benchmark identity')
	embedding = _mapping(identity.get('embedding'), 'benchmark embedding identity')
	common_metadata = _mapping(
		embedding.get('common_metadata'),
		'benchmark embedding common metadata',
	)
	return {
		**{
			key: value
			for key, value in identity.items()
			if key not in {'model', 'embedding'}
		},
		'embedding': {'common_metadata': dict(common_metadata)},
	}


def global_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
	"""Return model- and layout-independent downstream identity."""
	return {
		key: value
		for key, value in paired_identity(payload).items()
		if key not in _LAYOUT_SPECIFIC_IDENTITY_KEYS
	}


def model_source_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
	"""Return the model-specific source identity shared across layouts."""
	identity = _mapping(payload.get('benchmark_identity'), 'benchmark identity')
	embedding = _mapping(identity.get('embedding'), 'benchmark embedding identity')
	return {
		key: value for key, value in embedding.items() if key != 'common_metadata'
	}


def _validation_channel_iou(payload: Mapping[str, Any]) -> float:
	validation = payload.get('validation')
	if not isinstance(validation, Mapping):
		raise TypeError('validation metrics must be a mapping')
	value = validation.get('channel_iou')
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or not math.isfinite(float(value))
		or not 0.0 <= float(value) <= 1.0
	):
		raise ValueError('validation Channel IoU must be finite and in [0, 1]')
	return float(value)


def validation_gain(
	metrics: Mapping[tuple[str, str], Mapping[str, Any]],
	control_model: str,
	variant_model: str,
	layout: str,
) -> float:
	"""Compute variant minus control validation Channel IoU."""
	return _validation_channel_iou(
		metrics[(variant_model, layout)]
	) - _validation_channel_iou(metrics[(control_model, layout)])


def summarize_gains(
	gains: Mapping[str, float] | Sequence[float],
) -> dict[str, float | int]:
	"""Compute screening statistics for one group of paired gains."""
	values = list(gains.values()) if isinstance(gains, Mapping) else list(gains)
	if len(values) < 2:
		raise ValueError('at least two gains are required')
	return {
		'mean': statistics.mean(values),
		'median': statistics.median(values),
		'sample_standard_deviation': statistics.stdev(values),
		'wins': sum(value > 0.0 for value in values),
		'ties': sum(value == 0.0 for value in values),
		'losses': sum(value < 0.0 for value in values),
	}


def rank_variants(
	per_variant: Mapping[str, Mapping[str, Any]],
	variant_order: Sequence[str] = VARIANT_ORDER,
) -> list[str]:
	"""Rank eligible variants by combined mean, median, then table order."""
	eligible_variants = [
		variant for variant in variant_order if bool(per_variant[variant]['eligible'])
	]
	return sorted(
		eligible_variants,
		key=lambda variant: (
			-float(_mapping(per_variant[variant]['combined'], 'combined')['mean']),
			-float(_mapping(per_variant[variant]['combined'], 'combined')['median']),
			variant_order.index(variant),
		),
	)


def _summarize_variants(
	metrics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
	per_variant: dict[str, dict[str, Any]] = {}
	for variant in VARIANT_ORDER:
		branch_results: dict[str, dict[str, Any]] = {}
		combined_gains: list[float] = []
		for branch_name, branch in BRANCHES.items():
			control = str(branch['control'])
			variant_models = _mapping(branch['variants'], f'{branch_name} variants')
			model = str(variant_models[variant])
			layout_gains = {
				layout: validation_gain(metrics, control, model, layout)
				for layout in LAYOUTS
			}
			combined_gains.extend(layout_gains.values())
			branch_results[branch_name] = {
				'control_model': control,
				'variant_model': model,
				'layout_gains': layout_gains,
				**summarize_gains(layout_gains),
			}
		mae_mean = float(branch_results['mae']['mean'])
		local_bt_mean = float(branch_results['local_bt']['mean'])
		per_variant[variant] = {
			'boundary_settings': VARIANT_BOUNDARY_SETTINGS[variant],
			'mae': branch_results['mae'],
			'local_bt': branch_results['local_bt'],
			'combined': summarize_gains(combined_gains),
			'eligible': mae_mean >= 0.0 and local_bt_mean >= 0.0,
		}
	return per_variant


def write_report(result: Mapping[str, Any], report_root: Path) -> None:
	"""Write the machine-readable result and its human review table."""
	report_root.mkdir(parents=True, exist_ok=True)
	(report_root / 'screening_validation.json').write_text(
		json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	per_variant = _mapping(result['per_variant'], 'per-variant results')
	ranking = result['ranking']
	recommended_variant = result['recommended_variant']
	lines = [
		'# HMM boundary-weight validation screening',
		'',
		'Metric: validation.channel_iou; size: medium.',
		'',
		'## Summary',
		'',
		(
			'| variant | branch | mean | median | sample std | wins | ties | '
			'losses | eligible |'
		),
		'|---|---|---:|---:|---:|---:|---:|---:|:---:|',
	]
	for variant in VARIANT_ORDER:
		variant_result = _mapping(per_variant[variant], f'{variant} result')
		for branch_name in ('mae', 'local_bt', 'combined'):
			summary = _mapping(variant_result[branch_name], branch_name)
			lines.append(
				'| '
				f'{variant} | {branch_name} | {float(summary["mean"]):+.6f} | '
				f'{float(summary["median"]):+.6f} | '
				f'{float(summary["sample_standard_deviation"]):.6f} | '
				f'{int(summary["wins"])} | {int(summary["ties"])} | '
				f'{int(summary["losses"])} | {variant_result["eligible"]} |'
			)
	lines.extend(
		[
			'',
			'## Layout gains',
			'',
			'| variant | layout | MAE gain | Local BT gain |',
			'|---|---|---:|---:|',
		]
	)
	for variant in VARIANT_ORDER:
		variant_result = _mapping(per_variant[variant], f'{variant} result')
		mae = _mapping(variant_result['mae'], f'{variant} MAE')
		local_bt = _mapping(variant_result['local_bt'], f'{variant} Local BT')
		mae_gains = _mapping(mae['layout_gains'], f'{variant} MAE gains')
		local_bt_gains = _mapping(
			local_bt['layout_gains'],
			f'{variant} Local BT gains',
		)
		lines.extend(
			f'| {variant} | {layout} | {float(mae_gains[layout]):+.6f} | '
			f'{float(local_bt_gains[layout]):+.6f} |'
			for layout in LAYOUTS
		)
	lines.extend(
		[
			'',
			f'- Eligible ranking: {ranking}',
			f'- Recommended variant: {recommended_variant}',
			(
				'- This recommendation requires human review and does not start '
				'another Phase.'
			),
		]
	)
	(report_root / 'screening_validation.md').write_text(
		'\n'.join(lines) + '\n',
		encoding='utf-8',
	)


def _validate_identity_matrix(
	metrics: Mapping[tuple[str, str], Mapping[str, Any]],
	model_ids: Sequence[str],
) -> None:
	for layout in LAYOUTS:
		reference = paired_identity(metrics[(model_ids[0], layout)])
		for model in model_ids[1:]:
			if paired_identity(metrics[(model, layout)]) != reference:
				raise ValueError(
					f'{model}/{layout}: downstream benchmark identity mismatch'
				)
	global_reference = global_identity(metrics[(model_ids[0], LAYOUTS[0])])
	for model in model_ids:
		for layout in LAYOUTS:
			if global_identity(metrics[(model, layout)]) != global_reference:
				raise ValueError(
					f'{model}/{layout}: global downstream benchmark identity mismatch'
				)
	for model in model_ids:
		source_reference = model_source_identity(metrics[(model, LAYOUTS[0])])
		for layout in LAYOUTS[1:]:
			if model_source_identity(metrics[(model, layout)]) != source_reference:
				raise ValueError(
					f'{model}/{layout}: model source identity mismatch across layouts'
				)


def summarize_validation(
	existing_runs_root: Path,
	validation_runs_root: Path,
	report_root: Path,
) -> dict[str, Any]:
	"""Read the fixed 40-job matrix, validate it, rank it, and write reports."""
	model_ids = _model_ids()
	metrics = {
		(model, layout): read_metrics(
			existing_runs_root if model in EXISTING_MODELS else validation_runs_root,
			model,
			layout,
			validation_only=model not in EXISTING_MODELS,
		)
		for model in model_ids
		for layout in LAYOUTS
	}
	if len(metrics) != 40:
		raise AssertionError('screening must read exactly 40 metrics')
	_validate_identity_matrix(metrics, model_ids)

	per_variant = _summarize_variants(metrics)
	ranking = rank_variants(per_variant)
	result = {
		'metric': 'validation.channel_iou',
		'data_size': 'medium',
		'variant_boundary_settings': VARIANT_BOUNDARY_SETTINGS,
		'per_variant': per_variant,
		'ranking': ranking,
		'recommended_variant': ranking[0] if ranking else None,
		'selection_rule': {
			'eligibility': 'mae_mean >= 0 and local_bt_mean >= 0',
			'primary': 'largest combined mean among eligible variants',
			'first_tie_break': 'largest combined median',
			'second_tie_break': 'variant table order',
			'no_eligible_variant': 'recommended_variant is null',
			'automatic_next_phase_gate': False,
		},
	}
	write_report(result, report_root)
	return result


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--existing-runs-root', type=Path, required=True)
	parser.add_argument('--validation-runs-root', type=Path, required=True)
	parser.add_argument('--report-root', type=Path, required=True)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Run boundary-weight validation screening from explicit artifact roots."""
	args = _parser().parse_args(argv)
	result = summarize_validation(
		args.existing_runs_root,
		args.validation_runs_root,
		args.report_root,
	)
	print('metrics read: 40')
	print(f'eligible ranking: {result["ranking"]}')
	print(f'recommended_variant: {result["recommended_variant"]}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
