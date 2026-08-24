from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
	from types import ModuleType

SCRIPT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'36_channel_hmm_transition_balance_v1/scripts/summarize_validation.py'
)


def _load_summary_module() -> ModuleType:
	spec = importlib.util.spec_from_file_location(
		'parihaka_transition_balance_summary',
		SCRIPT,
	)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


SUMMARY = _load_summary_module()


def _gain_for_layout(value: float | tuple[float, ...], index: int) -> float:
	return value[index] if isinstance(value, tuple) else value


def _write_metrics_matrix(
	tmp_path: Path,
	gains: dict[str, dict[str, float | tuple[float, ...]]],
) -> tuple[Path, Path, Path]:
	existing_runs_root = tmp_path / 'existing'
	validation_runs_root = tmp_path / 'validation'
	report_root = tmp_path / 'report'
	for branch_name, branch in SUMMARY.BRANCHES.items():
		control = branch['control']
		variant_models = branch['variants']
		for layout_index, layout in enumerate(SUMMARY.LAYOUTS):
			control_iou = 0.40 + layout_index * 0.01
			model_ious = {control: control_iou}
			model_ious.update(
				{
					model: control_iou
					+ _gain_for_layout(gains[variant][branch_name], layout_index)
					for variant, model in variant_models.items()
				}
			)
			for model, channel_iou in model_ious.items():
				is_existing = model in SUMMARY.EXISTING_MODELS
				runs_root = existing_runs_root if is_existing else validation_runs_root
				path = (
					runs_root
					/ f'model={model}'
					/ f'layout={layout}'
					/ 'size=medium'
					/ 'metrics.json'
				)
				path.parent.mkdir(parents=True, exist_ok=True)
				payload = {
					'model': model,
					'layout_id': layout,
					'data_size': 'medium',
					'evaluation_mode': (
						'validation_and_test' if is_existing else 'validation_only'
					),
					'validation': {'channel_iou': channel_iou},
					'benchmark_identity': {
						'model': model,
						'layout_id': layout,
						'data_size': 'medium',
						'decoder': {'spec': 'fixed-decoder-v1'},
						'embedding': {
							'model': model,
							'checkpoint': f'/unused/{model}/latest.pt',
							'common_metadata': {
								'volume_shape_xyz': [10, 20, 30],
								'patch_size_xyz': [8, 8, 8],
							},
						},
					},
				}
				if is_existing:
					payload['test'] = {'channel_iou': 999.0}
				path.write_text(
					json.dumps(payload) + '\n',
					encoding='utf-8',
				)
	return existing_runs_root, validation_runs_root, report_root


def _positive_gains() -> dict[str, dict[str, float | tuple[float, ...]]]:
	return {
		'advance_favored_m003': {'mae': 0.01, 'local_bt': 0.01},
		'neutral': {'mae': 0.04, 'local_bt': 0.04},
		'persist003': {'mae': 0.02, 'local_bt': 0.02},
		'persist010': {'mae': 0.03, 'local_bt': 0.03},
	}


def test_known_validation_gains_select_expected_variant_and_write_reports(
	tmp_path: Path,
) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)

	result = SUMMARY.summarize_validation(
		existing_root,
		validation_root,
		report_root,
	)

	assert result['ranking'] == [
		'neutral',
		'persist010',
		'persist003',
		'advance_favored_m003',
	]
	assert result['recommended_variant'] == 'neutral'
	assert result['metric'] == 'validation.channel_iou'
	assert result['per_variant']['neutral']['combined']['wins'] == 10
	assert result['per_variant']['neutral']['mae']['layout_gains'][
		'layout_000'
	] == pytest.approx(0.04)
	written_json = json.loads(
		(report_root / 'screening_validation.json').read_text(encoding='utf-8')
	)
	assert written_json == result
	written_markdown = (report_root / 'screening_validation.md').read_text(
		encoding='utf-8'
	)
	assert '- Recommended variant: neutral' in written_markdown


def test_rank_variants_applies_median_then_table_order_tie_breaks() -> None:
	per_variant = {
		'table_first': {
			'eligible': True,
			'combined': {'mean': 0.1, 'median': 0.05},
		},
		'table_second': {
			'eligible': True,
			'combined': {'mean': 0.1, 'median': 0.05},
		},
		'median_winner': {
			'eligible': True,
			'combined': {'mean': 0.1, 'median': 0.08},
		},
		'ineligible': {
			'eligible': False,
			'combined': {'mean': 1.0, 'median': 1.0},
		},
	}
	order = ('table_first', 'table_second', 'median_winner', 'ineligible')

	assert SUMMARY.rank_variants(per_variant, order) == [
		'median_winner',
		'table_first',
		'table_second',
	]


@pytest.mark.parametrize('negative_branch', ['mae', 'local_bt'])
def test_negative_branch_mean_makes_variant_ineligible(
	tmp_path: Path,
	negative_branch: str,
) -> None:
	gains = _positive_gains()
	gains['neutral'][negative_branch] = -0.01
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		gains,
	)

	result = SUMMARY.summarize_validation(
		existing_root,
		validation_root,
		report_root,
	)

	assert result['per_variant']['neutral']['eligible'] is False
	assert 'neutral' not in result['ranking']
	assert result['recommended_variant'] == 'persist010'


def test_mismatched_benchmark_identity_is_rejected(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	path = (
		validation_root
		/ 'model=mae_hmm_k6_neutral'
		/ 'layout=layout_002'
		/ 'size=medium/metrics.json'
	)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['benchmark_identity']['decoder']['spec'] = 'drifted-decoder'
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(
		ValueError,
		match=('mae_hmm_k6_neutral/layout_002: downstream benchmark identity mismatch'),
	):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)
