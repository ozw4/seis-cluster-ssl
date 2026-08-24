"""Summarize validation-only Channel screening for transition-balance Phase 1."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

VARIANT_TRANSITION_SETTINGS = {
	'advance_favored_m003': {
		'same_cost': 0.03,
		'advance_cost': 0.00,
		'delta': -0.03,
	},
	'neutral': {
		'same_cost': 0.00,
		'advance_cost': 0.00,
		'delta': 0.00,
	},
	'persist003': {
		'same_cost': 0.00,
		'advance_cost': 0.03,
		'delta': 0.03,
	},
	'persist010': {
		'same_cost': 0.00,
		'advance_cost': 0.10,
		'delta': 0.10,
	},
}
BRANCHES = {
	'mae': {
		'control': 'mae',
		'variants': {
			'advance_favored_m003': 'mae_hmm_k6',
			'neutral': 'mae_hmm_k6_neutral',
			'persist003': 'mae_hmm_k6_persist003',
			'persist010': 'mae_hmm_k6_persist010',
		},
	},
	'local_bt': {
		'control': 'local_barlow_twins',
		'variants': {
			'advance_favored_m003': 'local_barlow_twins_hmm_k6',
			'neutral': 'local_barlow_twins_hmm_k6_neutral',
			'persist003': 'local_barlow_twins_hmm_k6_persist003',
			'persist010': 'local_barlow_twins_hmm_k6_persist010',
		},
	},
}
LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))
VARIANT_ORDER = tuple(VARIANT_TRANSITION_SETTINGS)
EXISTING_MODELS = frozenset(
	{
		'mae',
		'mae_hmm_k6',
		'local_barlow_twins',
		'local_barlow_twins_hmm_k6',
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
	if len(unique_model_ids) != 10:
		raise AssertionError('screening must define exactly 10 models')
	return unique_model_ids


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
	if validation_only:
		if payload.get('evaluation_mode') != 'validation_only':
			raise ValueError(f'{path}: candidate must be validation-only')
		if 'test' in payload:
			raise ValueError(f'{path}: validation-only metrics contain test results')
	return payload


def paired_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
	"""Return benchmark identity shared by a paired downstream comparison."""
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


def _validation_channel_iou(payload: Mapping[str, Any]) -> float:
	validation = payload.get('validation')
	if not isinstance(validation, Mapping):
		raise TypeError('validation metrics must be a mapping')
	value = validation.get('channel_iou')
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or not math.isfinite(float(value))
	):
		raise ValueError('validation Channel IoU must be finite')
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
			'transition_settings': VARIANT_TRANSITION_SETTINGS[variant],
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
		'# HMM transition balance validation screening',
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
			'- This recommendation requires human review and does not start Phase 2.',
		]
	)
	(report_root / 'screening_validation.md').write_text(
		'\n'.join(lines) + '\n',
		encoding='utf-8',
	)


def summarize_validation(
	existing_runs_root: Path,
	validation_runs_root: Path,
	report_root: Path,
) -> dict[str, Any]:
	"""Read the fixed 50-job matrix, validate it, rank it, and write reports."""
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
	if len(metrics) != 50:
		raise AssertionError('screening must read exactly 50 metrics')
	for layout in LAYOUTS:
		reference = paired_identity(metrics[(model_ids[0], layout)])
		for model in model_ids[1:]:
			if paired_identity(metrics[(model, layout)]) != reference:
				raise ValueError(
					f'{model}/{layout}: downstream benchmark identity mismatch'
				)

	per_variant = _summarize_variants(metrics)
	ranking = rank_variants(per_variant)
	result = {
		'metric': 'validation.channel_iou',
		'data_size': 'medium',
		'variant_transition_settings': VARIANT_TRANSITION_SETTINGS,
		'per_variant': per_variant,
		'ranking': ranking,
		'recommended_variant': ranking[0] if ranking else None,
		'selection_rule': {
			'eligibility': 'mae_mean >= 0 and local_bt_mean >= 0',
			'primary': 'largest combined mean among eligible variants',
			'first_tie_break': 'largest combined median',
			'second_tie_break': 'variant table order',
			'no_eligible_variant': 'recommended_variant is null',
			'automatic_phase2_gate': False,
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
	"""Run the fixed Phase 1 validation summary from explicit artifact roots."""
	args = _parser().parse_args(argv)
	result = summarize_validation(
		args.existing_runs_root,
		args.validation_runs_root,
		args.report_root,
	)
	print('metrics read: 50')
	print(f'eligible ranking: {result["ranking"]}')
	print(f'recommended_variant: {result["recommended_variant"]}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
