"""Example F3 lithology robustness suite contracts."""

from __future__ import annotations

from pathlib import Path

from seis_ssl_cluster.f3.lithology.robustness import (
	F3_ROBUSTNESS_CONTRACT_VERSION,
	F3RobustnessModelSpec,
	F3RobustnessSuiteManifest,
)
from seis_ssl_cluster.paths import DEFAULT_ARTIFACT_ROOT

F3_LITHOLOGY_ROBUSTNESS_ROOT = (
	DEFAULT_ARTIFACT_ROOT / 'lithology/f3/facies_benchmark_v1/robustness'
)
F3_LABEL_BUDGET_M1_SUITE_NAME = 'label_budget_m1_v1'
F3_SPLIT_INDEX_M1_SUITE_NAME = 'split_index_m1_v1'
F3_M1_BASELINE_MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
F3_M1_CANDIDATE_MODEL_TAG = 'strat_hmm_pretext_m1_k6_topblock1_distill'
F3_M1_EMBEDDING_SPEC = 'overlap_x16'
F3_M1_LABEL_SET = 'png_slices_segy_labels_v1'
F3_M1_PROBE_SPEC = 'linear_balanced_v1'


def f3_m1_example_model_specs() -> tuple[F3RobustnessModelSpec, ...]:
	"""Return the example MAE baseline and strat-HMM candidate specs."""
	return (
		F3RobustnessModelSpec(
			model_tag=F3_M1_BASELINE_MODEL_TAG,
			role='baseline',
			embedding_spec=F3_M1_EMBEDDING_SPEC,
			label_set=F3_M1_LABEL_SET,
			probe_spec=F3_M1_PROBE_SPEC,
		),
		F3RobustnessModelSpec(
			model_tag=F3_M1_CANDIDATE_MODEL_TAG,
			role='candidate',
			embedding_spec=F3_M1_EMBEDDING_SPEC,
			label_set=F3_M1_LABEL_SET,
			probe_spec=F3_M1_PROBE_SPEC,
		),
	)


def f3_m1_robustness_suite_manifest(
	suite_name: str,
	*,
	output_root: Path | None = None,
) -> F3RobustnessSuiteManifest:
	"""Build an example resolved suite manifest for the F3 M1 contract."""
	root = output_root or F3_LITHOLOGY_ROBUSTNESS_ROOT / suite_name
	return F3RobustnessSuiteManifest(
		suite_name=suite_name,
		contract_version=F3_ROBUSTNESS_CONTRACT_VERSION,
		output_root=root,
		models=f3_m1_example_model_specs(),
		report_paths={
			'paired_metrics_csv': root / 'reports/paired_metrics.csv',
			'paired_deltas_csv': root / 'reports/paired_deltas.csv',
			'summary_markdown': root / 'reports/summary.md',
			'suite_config_resolved_json': root / 'suite_config_resolved.json',
			'suite_manifest_json': root / 'suite_manifest.json',
		},
	)


__all__ = [
	'F3_LABEL_BUDGET_M1_SUITE_NAME',
	'F3_LITHOLOGY_ROBUSTNESS_ROOT',
	'F3_M1_BASELINE_MODEL_TAG',
	'F3_M1_CANDIDATE_MODEL_TAG',
	'F3_M1_EMBEDDING_SPEC',
	'F3_M1_LABEL_SET',
	'F3_M1_PROBE_SPEC',
	'F3_SPLIT_INDEX_M1_SUITE_NAME',
	'f3_m1_example_model_specs',
	'f3_m1_robustness_suite_manifest',
]
