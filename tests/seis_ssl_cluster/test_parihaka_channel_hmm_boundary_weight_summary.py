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
	'37_channel_hmm_boundary_weight_v1/scripts/summarize_validation.py'
)


def _load_summary_module() -> ModuleType:
	spec = importlib.util.spec_from_file_location(
		'parihaka_hmm_boundary_weight_summary',
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
				path.write_text(json.dumps(payload) + '\n', encoding='utf-8')
	return existing_runs_root, validation_runs_root, report_root


def _positive_gains() -> dict[str, dict[str, float | tuple[float, ...]]]:
	return {
		'alpha000_tau1': {'mae': 0.01, 'local_bt': 0.01},
		'alpha050_tau1': {'mae': 0.04, 'local_bt': 0.04},
		'alpha100_tau1': {'mae': 0.03, 'local_bt': 0.03},
	}


def _new_metrics_path(validation_root: Path) -> Path:
	return (
		validation_root
		/ 'model=mae_hmm_k6_boundary_alpha050_tau1'
		/ 'layout=layout_002'
		/ 'size=medium/metrics.json'
	)


def test_known_validation_gains_select_expected_alpha_and_write_reports(
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
		'alpha050_tau1',
		'alpha100_tau1',
		'alpha000_tau1',
	]
	assert result['recommended_variant'] == 'alpha050_tau1'
	assert result['metric'] == 'validation.channel_iou'
	assert result['variant_boundary_settings']['alpha050_tau1'] == {
		'boundary_alpha': 0.5,
		'boundary_tau': 1.0,
	}
	assert result['per_variant']['alpha050_tau1']['combined']['wins'] == 10
	assert result['per_variant']['alpha050_tau1']['mae']['layout_gains'][
		'layout_000'
	] == pytest.approx(0.04)
	written_json = json.loads(
		(report_root / 'screening_validation.json').read_text(encoding='utf-8')
	)
	assert written_json == result
	written_markdown = (report_root / 'screening_validation.md').read_text(
		encoding='utf-8'
	)
	assert '- Recommended variant: alpha050_tau1' in written_markdown


@pytest.mark.parametrize('negative_branch', ['mae', 'local_bt'])
def test_negative_branch_mean_makes_variant_ineligible(
	tmp_path: Path,
	negative_branch: str,
) -> None:
	gains = _positive_gains()
	gains['alpha050_tau1'][negative_branch] = -0.01
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		gains,
	)

	result = SUMMARY.summarize_validation(
		existing_root,
		validation_root,
		report_root,
	)

	assert result['per_variant']['alpha050_tau1']['eligible'] is False
	assert 'alpha050_tau1' not in result['ranking']
	assert result['recommended_variant'] == 'alpha100_tau1'


def test_rank_variants_applies_median_then_table_order_tie_breaks() -> None:
	per_variant = {
		'alpha000_tau1': {
			'eligible': True,
			'combined': {'mean': 0.1, 'median': 0.05},
		},
		'alpha050_tau1': {
			'eligible': True,
			'combined': {'mean': 0.1, 'median': 0.08},
		},
		'alpha100_tau1': {
			'eligible': True,
			'combined': {'mean': 0.1, 'median': 0.05},
		},
	}

	assert SUMMARY.rank_variants(per_variant) == [
		'alpha050_tau1',
		'alpha000_tau1',
		'alpha100_tau1',
	]


def test_no_eligible_variant_produces_null_recommendation(tmp_path: Path) -> None:
	gains = {
		variant: {'mae': -0.01, 'local_bt': -0.01}
		for variant in SUMMARY.VARIANT_ORDER
	}
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		gains,
	)

	result = SUMMARY.summarize_validation(
		existing_root,
		validation_root,
		report_root,
	)

	assert result['ranking'] == []
	assert result['recommended_variant'] is None


def test_mismatched_benchmark_identity_is_rejected(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	path = _new_metrics_path(validation_root)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['benchmark_identity']['decoder']['spec'] = 'drifted-decoder'
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(
		ValueError,
		match=(
			'mae_hmm_k6_boundary_alpha050_tau1/layout_002: '
			'downstream benchmark identity mismatch'
		),
	):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)


def test_layout_wide_global_benchmark_drift_is_rejected(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	for model in SUMMARY._model_ids():  # noqa: SLF001
		runs_root = (
			existing_root if model in SUMMARY.EXISTING_MODELS else validation_root
		)
		path = (
			runs_root
			/ f'model={model}'
			/ 'layout=layout_002/size=medium/metrics.json'
		)
		payload = json.loads(path.read_text(encoding='utf-8'))
		payload['benchmark_identity']['decoder']['spec'] = 'layout-wide-drift'
		path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(
		ValueError,
		match='global downstream benchmark identity mismatch',
	):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)


def test_model_source_must_be_stable_across_layouts(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	path = _new_metrics_path(validation_root)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['benchmark_identity']['embedding']['checkpoint'] = '/drifted/latest.pt'
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(
		ValueError,
		match='model source identity mismatch across layouts',
	):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)


def test_benchmark_model_identity_must_match_job(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	path = _new_metrics_path(validation_root)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['benchmark_identity']['model'] = 'wrong-model'
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(ValueError, match='benchmark model identity mismatch'):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)


def test_benchmark_model_source_must_match_job(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	path = _new_metrics_path(validation_root)
	payload = json.loads(path.read_text(encoding='utf-8'))
	embedding = payload['benchmark_identity']['embedding']
	embedding['checkpoint_path'] = '/expected/latest.pt'
	embedding['model_source'] = {
		'model_id': 'wrong-model',
		'checkpoint_path': '/expected/latest.pt',
	}
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(
		ValueError,
		match='benchmark model-source identity mismatch',
	):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)


def test_validation_channel_iou_must_be_a_probability(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	path = _new_metrics_path(validation_root)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['validation']['channel_iou'] = 2.0
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(ValueError, match=r'finite and in \[0, 1\]'):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)


def test_new_metrics_in_normal_evaluation_mode_are_rejected(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	path = _new_metrics_path(validation_root)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['evaluation_mode'] = 'validation_and_test'
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(ValueError, match='candidate must be validation-only'):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)


def test_new_metrics_with_top_level_test_field_are_rejected(tmp_path: Path) -> None:
	existing_root, validation_root, report_root = _write_metrics_matrix(
		tmp_path,
		_positive_gains(),
	)
	path = _new_metrics_path(validation_root)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['test'] = {'channel_iou': 0.99}
	path.write_text(json.dumps(payload) + '\n', encoding='utf-8')

	with pytest.raises(
		ValueError,
		match='validation-only metrics contain test results',
	):
		SUMMARY.summarize_validation(
			existing_root,
			validation_root,
			report_root,
		)
