from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config.f3_lithology_robustness import (
	F3_LABEL_BUDGET_M1_SUITE_NAME,
	F3_LITHOLOGY_ROBUSTNESS_ROOT,
	F3_SPLIT_INDEX_M1_SUITE_NAME,
	f3_m1_example_model_specs,
	f3_m1_robustness_suite_manifest,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	F3_ROBUSTNESS_CONTRACT_VERSION,
	F3RobustnessModelSpec,
	F3RobustnessSuiteManifest,
	validate_f3_m1_model_pair,
	validate_f3_robustness_config_keys,
)


def test_valid_model_specs_pass() -> None:
	models = f3_m1_example_model_specs()

	validate_f3_m1_model_pair(models)

	assert [model.role for model in models] == ['baseline', 'candidate']
	assert all(model.embedding_spec == 'overlap_x16' for model in models)


def test_empty_model_tag_fails() -> None:
	with pytest.raises(TypeError, match='model_tag'):
		F3RobustnessModelSpec(
			model_tag='',
			role='baseline',
			embedding_spec='overlap_x16',
			label_set='png_slices_segy_labels_v1',
			probe_spec='linear_balanced_v1',
		)


def test_non_absolute_output_root_fails() -> None:
	with pytest.raises(ValueError, match='output_root'):
		F3RobustnessSuiteManifest(
			suite_name=F3_LABEL_BUDGET_M1_SUITE_NAME,
			contract_version=F3_ROBUSTNESS_CONTRACT_VERSION,
			output_root=Path('artifacts/robustness'),
			models=f3_m1_example_model_specs(),
		)


@pytest.mark.parametrize(
	'models',
	[
		(),
		(f3_m1_example_model_specs()[0],),
		(
			*f3_m1_example_model_specs(),
			F3RobustnessModelSpec(
				model_tag='extra_model',
				role='candidate',
				embedding_spec='overlap_x16',
				label_set='png_slices_segy_labels_v1',
				probe_spec='linear_balanced_v1',
			),
		),
	],
)
def test_model_count_other_than_paired_baseline_candidate_fails(
	models: tuple[F3RobustnessModelSpec, ...],
) -> None:
	with pytest.raises(ValueError, match='exactly one baseline and one candidate'):
		validate_f3_m1_model_pair(models)


def test_two_models_without_baseline_candidate_roles_fail() -> None:
	models = (
		F3RobustnessModelSpec(
			model_tag='mae_a',
			role='baseline',
			embedding_spec='overlap_x16',
			label_set='png_slices_segy_labels_v1',
			probe_spec='linear_balanced_v1',
		),
		F3RobustnessModelSpec(
			model_tag='mae_b',
			role='baseline',
			embedding_spec='overlap_x16',
			label_set='png_slices_segy_labels_v1',
			probe_spec='linear_balanced_v1',
		),
	)

	with pytest.raises(ValueError, match='baseline.*candidate'):
		validate_f3_m1_model_pair(models)


def test_disallowed_suite_names_and_keys_are_rejected() -> None:
	with pytest.raises(ValueError, match='seed_sweep'):
		f3_m1_robustness_suite_manifest('seed_sweep')

	with pytest.raises(ValueError, match='checkpoint_policy'):
		validate_f3_robustness_config_keys(
			{
				'suite_name': F3_SPLIT_INDEX_M1_SUITE_NAME,
				'probe': {'checkpoint_policy': 'latest_vs_best'},
			},
		)


def test_example_suite_manifest_uses_required_artifact_layout() -> None:
	manifest = f3_m1_robustness_suite_manifest(F3_LABEL_BUDGET_M1_SUITE_NAME)

	assert manifest.output_root == (
		F3_LITHOLOGY_ROBUSTNESS_ROOT / F3_LABEL_BUDGET_M1_SUITE_NAME
	)
	assert manifest.report_paths['paired_metrics_csv'] == (
		manifest.output_root / 'reports/paired_metrics.csv'
	)
	assert manifest.report_paths['paired_deltas_csv'] == (
		manifest.output_root / 'reports/paired_deltas.csv'
	)
