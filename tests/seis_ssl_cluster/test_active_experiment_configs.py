# ruff: noqa: CPY001

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from proc.seis_ssl_cluster.run_f3_lithology_split_sweep_probes import (
	f3_lithology_split_sweep_probe_config_from_mapping,
)
from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.clustering.kmeans import clustering_settings_from_config
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	stratigraphic_hmm_settings_from_config,
)
from seis_ssl_cluster.config import (
	load_config,
	resolve_cluster_visualization_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_f3_facies_inspection_config,
	resolve_mae_training_config,
	resolve_strat_hmm_pretext_config,
	resolve_strat_hmm_pseudo_target_config,
)
from seis_ssl_cluster.config.f3_baselines import (
	f3_lithology_baseline_token_dataset_config_from_mapping,
	f3_lithology_comparison_publish_config_from_mapping,
	f3_lithology_comparison_report_config_from_mapping,
	random_mae_checkpoint_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_prediction_config_from_mapping,
	f3_lithology_probe_config_from_mapping,
	f3_lithology_publish_config_from_mapping,
	f3_lithology_report_config_from_mapping,
	f3_lithology_token_dataset_config_from_mapping,
	f3_lithology_visualization_config_from_mapping,
	f3_prepare_volume_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_label_budget_config_from_mapping,
	f3_lithology_label_budget_probe_config_from_mapping,
	f3_lithology_label_budget_summary_config_from_mapping,
	f3_lithology_split_inventory_config_from_mapping,
	f3_lithology_split_summary_config_from_mapping,
	f3_lithology_split_sweep_dataset_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_dataset import (
	f3_lithology_voxel_dataset_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	f3_lithology_voxel_decoder_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	f3_lithology_voxel_evaluation_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	f3_lithology_voxel_inference_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget import (
	f3_lithology_voxel_label_budget_dataset_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_control import (
	f3_lithology_voxel_label_budget_control_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_results import (
	f3_lithology_voxel_label_budget_results_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_suite import (
	f3_lithology_voxel_label_budget_suite_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_projection import (
	f3_lithology_voxel_projection_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_report import (
	f3_lithology_voxel_report_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_results import (
	f3_lithology_voxel_results_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_robustness import (
	f3_lithology_voxel_decoder_split_suite_config_from_mapping,
	f3_lithology_voxel_split_dataset_config_from_mapping,
	f3_lithology_voxel_split_summary_config_from_mapping,
	f3_lithology_voxel_v0_split_suite_config_from_mapping,
)
from seis_ssl_cluster.config.performance_migration_validation import (
	performance_migration_validation_config_from_mapping,
)
from seis_ssl_cluster.config.pretraining import _multi_head_target_hashes
from seis_ssl_cluster.config.schema import (
	STAGE_F3_INSPECT_FILES,
	STAGE_F3_INSPECTION_REPORT,
	STAGE_F3_LABEL_CONSISTENCY,
	STAGE_F3_PNG_LABELS,
	STAGE_F3_QUICKLOOK,
	STAGE_F3_SEGY_GEOMETRY,
	STAGE_F3_TOKENIZATION_PREVIEW,
)
from seis_ssl_cluster.embedding.extractor import extraction_settings_from_config
from seis_ssl_cluster.f3.center_trace_masked_periodic_refresh_validation import (
	load_f3_center_trace_masked_periodic_refresh_validation_config,
)
from seis_ssl_cluster.f3.lithology.guardrails import (
	f3_shuffled_hmm_target_config_from_mapping,
)
from seis_ssl_cluster.models.voxel_decoder.spec import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
)
from seis_ssl_cluster.paths import DEFAULT_ARTIFACT_ROOT, ArtifactPaths, ExperimentKey
from seis_ssl_cluster.stratigraphy import (
	lateral_targets,
	state_posterior,
	xy_neighbor_consensus_targets,
	xy_neighbor_unanimous_targets,
)
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from tests.seis_ssl_cluster.test_strat_multi_head_target_manifest import (
	_artifacts,
	_replay_k6_root,
	_write_positive_preflight,
)

VOXEL_DECODER_SMOKE_SPEC = f'{VOXEL_DECODER_SPEC}_smoke'
OLD_VOXEL_DECODER_SPEC = 'frozen_embedding_decoder_v1'

ALL_CONFIGS = sorted(
	[
		*Path('proc/configs/seis_ssl_cluster').rglob('*.yaml'),
		*Path('experiments/nopims').rglob('*.yaml'),
		*Path('experiments/f3').rglob('*.yaml'),
	],
)

CORE_CONFIG_RESOLVERS = {
	frozenset(
		{
			'paths',
			'manifests',
			'data',
			'zero_mask',
			'model',
			'masking',
			'loss',
			'train',
			'visualization',
		}
	): resolve_mae_training_config,
	frozenset({'paths', 'manifests', 'embeddings', 'embedding'}): (
		resolve_embedding_extraction_config
	),
	frozenset({'paths', 'embeddings', 'clustering'}): resolve_clustering_config,
	frozenset({'paths', 'clustering', 'visualization'}): (
		resolve_cluster_visualization_config
	),
}

NOPIMS_ROOT = Path('experiments/nopims/pretrain_v1')
NOPIMS_PRETRAINING_CONFIGS = sorted((NOPIMS_ROOT / '10_pretrain').rglob('*.yaml'))
NOPIMS_EMBEDDING_CONFIGS = sorted((NOPIMS_ROOT / '20_embedding').rglob('*.yaml'))
NOPIMS_CLUSTERING_CONFIGS = sorted((NOPIMS_ROOT / '30_clustering').rglob('*.yaml'))
NOPIMS_VISUALIZATION_CONFIGS = sorted(
	(NOPIMS_ROOT / '40_visualization').rglob('*.yaml'),
)
F3_ROOT = Path('experiments/f3/facies_benchmark_v1')
F3_INSPECTION_STAGES = {
	'01_inspect_files.yaml': STAGE_F3_INSPECT_FILES,
	'02_inspect_segy_geometry.yaml': STAGE_F3_SEGY_GEOMETRY,
	'03_inspect_png_labels.yaml': STAGE_F3_PNG_LABELS,
	'04_make_quicklook_figures.yaml': STAGE_F3_QUICKLOOK,
	'05_check_label_consistency.yaml': STAGE_F3_LABEL_CONSISTENCY,
	'06_make_tokenization_preview.yaml': STAGE_F3_TOKENIZATION_PREVIEW,
	'07_build_inspection_report.yaml': STAGE_F3_INSPECTION_REPORT,
}
F3_INSPECTION_CONFIGS = [
	(path, F3_INSPECTION_STAGES[path.name])
	for path in sorted((F3_ROOT / '00_inspection').rglob('*.yaml'))
]
F3_PREPARE_CONFIGS = sorted((F3_ROOT / '10_prepare').rglob('*.yaml'))
F3_EMBEDDING_CONFIGS = sorted((F3_ROOT / '20_embedding').rglob('*.yaml'))
F3_STRATIGRAPHIC_CLUSTERING_CONFIGS = sorted(
	(F3_ROOT / '60_stratigraphic_clustering').rglob('*.yaml'),
)
F3_STRAT_HMM_PRETRAINING_M1_ROOT = F3_ROOT / '80_strat_hmm_pretraining_m1'
F3_STRAT_HMM_M1_GUARDRAIL_ROOT = F3_ROOT / '83_strat_hmm_m1_guardrails'
F3_STRAT_HMM_PRETRAINING_M2A_ROOT = F3_ROOT / '84_strat_hmm_pretraining_m2a_boundary'
F3_CURRENT_K6_CONTROL_ROOT = F3_ROOT / '93_strat_hmm_m1_current_k6_control'
F3_STRAT_HMM_MULTI_HEAD_ROOT = F3_ROOT / '94_strat_hmm_multi_head_k6810_v1'
F3_STRAT_HMM_SOFT_POSTERIOR_ROOT = (
	F3_ROOT / '97_strat_hmm_multi_head_k6810_soft_posterior_v1'
)
F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT = (
	F3_ROOT / '99_strat_hmm_multi_head_k6810_lateral_smoothing_v1'
)
F3_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_ROOT = (
	F3_ROOT / '100_strat_hmm_multi_head_k6810_xy_neighbor_consensus_v1'
)
F3_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_ROOT = (
	F3_ROOT / '102_strat_hmm_multi_head_k6810_xy_neighbor_unanimous_v1'
)
F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT = (
	F3_ROOT / '104_strat_hmm_multi_head_k6810_center_trace_masked_v1'
)
F3_STRAT_HMM_PERIODIC_REFRESH_ROOT = (
	F3_ROOT
	/ '107_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_v1'
)
F3_STRAT_HMM_PERIODIC_REFRESH_PRETEXT_CONFIGS = [
	F3_STRAT_HMM_PERIODIC_REFRESH_ROOT / '01_train_periodic_refresh_smoke.yaml',
	F3_STRAT_HMM_PERIODIC_REFRESH_ROOT / '02_train_periodic_refresh_full.yaml',
]
F3_STRAT_HMM_PERIODIC_REFRESH_EMBEDDING_CONFIGS = [
	F3_STRAT_HMM_PERIODIC_REFRESH_ROOT
	/ '03_extract_periodic_refresh_embeddings.yaml',
]
F3_STRAT_HMM_PERIODIC_REFRESH_VALIDATION_CONFIGS = [
	F3_STRAT_HMM_PERIODIC_REFRESH_ROOT
	/ '04_validate_periodic_refresh_pretraining.yaml',
]
F3_PERIODIC_REFRESH_ORIGINAL_SPLIT_ROOT = (
	F3_ROOT
	/ '108_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_low_label_v1'
)
F3_PERIODIC_REFRESH_ORIGINAL_SPLIT_CONFIGS = sorted(
	F3_PERIODIC_REFRESH_ORIGINAL_SPLIT_ROOT.glob('*.yaml')
)
F3_STRAT_HMM_PRETEXT_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '02_train_single_head_topblock_distill_smoke.yaml',
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '03_train_single_head_topblock_distill_full.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '01_train_distillation_only_smoke.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '02_train_distillation_only_full.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '07_train_shuffled_hmm_smoke.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '08_train_shuffled_hmm_full.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '03_train_boundary_smoke.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '04_train_boundary_full.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '01_train_current_k6_smoke.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '02_train_current_k6_full.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '02_train_nocons_smoke.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '03_train_cons010_smoke.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '04_train_nocons_full.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '05_train_cons010_full.yaml',
		F3_STRAT_HMM_SOFT_POSTERIOR_ROOT / '02_train_soft_smoke.yaml',
		F3_STRAT_HMM_SOFT_POSTERIOR_ROOT / '03_train_soft_full.yaml',
		F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT / '05_train_lateral_smoke.yaml',
		F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT / '06_train_lateral_full.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_ROOT
		/ '02_train_xy_neighbor_consensus_smoke.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_ROOT
		/ '03_train_xy_neighbor_consensus_full.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_ROOT
		/ '03_train_xy_neighbor_unanimous_smoke.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_ROOT
		/ '04_train_xy_neighbor_unanimous_full.yaml',
		F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT
		/ '01_train_center_trace_masked_smoke.yaml',
		F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT
		/ '02_train_center_trace_masked_full.yaml',
	],
)
F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT / '04_extract_student_embeddings.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '03_extract_distillation_only_embeddings.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '09_extract_shuffled_hmm_embeddings.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '05_extract_student_embeddings.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '03_extract_current_k6_embeddings.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '06_extract_nocons_embeddings.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '07_extract_cons010_embeddings.yaml',
		F3_STRAT_HMM_SOFT_POSTERIOR_ROOT / '04_extract_soft_embeddings.yaml',
		F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT / '07_extract_lateral_embeddings.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_ROOT
		/ '04_extract_xy_neighbor_consensus_embeddings.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_ROOT
		/ '05_extract_xy_neighbor_unanimous_embeddings.yaml',
		F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT
		/ '03_extract_center_trace_masked_embeddings.yaml',
	],
)
F3_STRAT_HMM_CENTER_TRACE_MASKED_VALIDATION_CONFIGS = [
	F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT
	/ '04_validate_center_trace_masked_pretraining.yaml',
]
F3_STRAT_HMM_STUDENT_LITHOLOGY_TOKEN_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT / '05_build_lithology_token_dataset.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT
		/ '04_build_distillation_only_token_dataset.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '10_build_shuffled_hmm_token_dataset.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '06_build_lithology_token_dataset.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '04_build_current_k6_token_dataset.yaml',
	],
)
F3_STRAT_HMM_STUDENT_LITHOLOGY_PROBE_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT / '06_train_lithology_probe.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '05_train_distillation_only_probe.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '11_train_shuffled_hmm_probe.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '07_train_lithology_probe.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '05_train_current_k6_token_probe.yaml',
	],
)
F3_STRAT_HMM_STUDENT_LITHOLOGY_REPORT_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT / '07_build_lithology_report.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '06_build_distillation_only_report.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '12_build_shuffled_hmm_report.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '08_build_lithology_report.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '06_build_current_k6_token_report.yaml',
	],
)
F3_STRAT_HMM_SHUFFLED_TARGET_CONFIGS = [
	F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '03_build_shuffled_hmm_pseudo_targets.yaml',
]
F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '08_refresh_pseudo_targets_from_logits_smoke.yaml',
	],
)
F3_STRAT_HMM_M1_ROBUSTNESS_ROOT = F3_ROOT / '81_strat_hmm_m1_robustness'
F3_STRAT_HMM_M1_ROBUSTNESS_CONFIGS = sorted(
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT.rglob('*.yaml'),
)
F3_STRAT_HMM_M1_LABEL_BUDGET_BUILD_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '01_build_label_budget_datasets.yaml',
	F3_STRAT_HMM_M1_GUARDRAIL_ROOT
	/ '14_build_distillation_only_label_budget_datasets.yaml',
	F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '16_build_shuffled_hmm_label_budget_datasets.yaml',
]
F3_STRAT_HMM_M1_LABEL_BUDGET_PROBE_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '02_run_label_budget_probes.yaml',
	F3_STRAT_HMM_M1_GUARDRAIL_ROOT
	/ '15_run_distillation_only_label_budget_probes.yaml',
	F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '17_run_shuffled_hmm_label_budget_probes.yaml',
]
F3_STRAT_HMM_M1_SPLIT_INVENTORY_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '04_generate_split_inventories.yaml',
]
F3_STRAT_HMM_M1_SPLIT_DATASET_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '05_build_split_sweep_datasets.yaml',
]
F3_STRAT_HMM_M1_SPLIT_PROBE_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '06_run_split_sweep_probes.yaml',
]
F3_STRAT_HMM_M2A_ROBUSTNESS_ROOT = F3_ROOT / '85_strat_hmm_m2a_robustness'
F3_STRAT_HMM_M2A_ROBUSTNESS_CONFIGS = sorted(
	F3_STRAT_HMM_M2A_ROBUSTNESS_ROOT.rglob('*.yaml'),
)
F3_STRAT_HMM_M2A_LABEL_BUDGET_BUILD_CONFIG = (
	F3_STRAT_HMM_M2A_ROBUSTNESS_ROOT / '01_build_label_budget_datasets.yaml'
)
F3_STRAT_HMM_M2A_LABEL_BUDGET_PROBE_CONFIG = (
	F3_STRAT_HMM_M2A_ROBUSTNESS_ROOT / '02_run_label_budget_probes.yaml'
)
F3_STRAT_HMM_M2A_LABEL_BUDGET_SUMMARY_CONFIG = (
	F3_STRAT_HMM_M2A_ROBUSTNESS_ROOT / '03_summarize_label_budget.yaml'
)
F3_STRAT_HMM_M2A_SPLIT_DATASET_CONFIG = (
	F3_STRAT_HMM_M2A_ROBUSTNESS_ROOT / '04_build_split_sweep_datasets.yaml'
)
F3_STRAT_HMM_M2A_SPLIT_PROBE_CONFIG = (
	F3_STRAT_HMM_M2A_ROBUSTNESS_ROOT / '05_run_split_sweep_probes.yaml'
)
F3_STRAT_HMM_M2A_SPLIT_SUMMARY_CONFIG = (
	F3_STRAT_HMM_M2A_ROBUSTNESS_ROOT / '06_summarize_split_sweep.yaml'
)
F3_LITHOLOGY_ROOT = F3_ROOT / '50_lithology'
F3_LITHOLOGY_TOKEN_CONFIGS = sorted(
	F3_LITHOLOGY_ROOT.rglob('01_build_token_dataset.yaml'),
)
F3_LITHOLOGY_PROBE_CONFIGS = sorted(
	[
		*F3_LITHOLOGY_ROOT.rglob('02_train_linear_probe.yaml'),
		*F3_LITHOLOGY_ROOT.rglob('03_train_mlp_probe.yaml'),
	],
)
F3_LITHOLOGY_PREDICTION_CONFIGS = sorted(
	F3_LITHOLOGY_ROOT.rglob('04_predict_volume.yaml'),
)
F3_LITHOLOGY_VISUALIZATION_CONFIGS = sorted(
	F3_LITHOLOGY_ROOT.rglob('05_visualize_predictions.yaml'),
)
F3_LITHOLOGY_REPORT_CONFIGS = sorted(
	F3_LITHOLOGY_ROOT.rglob('06_build_lithology_report.yaml'),
)
F3_VOXEL_V0_ROOT = F3_ROOT / '87_f3_voxel_benchmark_v0'
F3_VOXEL_V1_ROOT = F3_ROOT / '88_f3_voxel_decoder_v1'
F3_VOXEL_ROBUSTNESS_ROOT = F3_ROOT / '89_f3_voxel_split_robustness'
F3_VOXEL_RESULTS_ROOT = F3_ROOT / '90_f3_voxel_results'
F3_VOXEL_LABEL_BUDGET_ROOT = F3_ROOT / '91_f3_voxel_label_budget_v1'
F3_VOXEL_LABEL_BUDGET_CONFIGS = [
	F3_VOXEL_LABEL_BUDGET_ROOT / '01_build_voxel_label_budget_datasets.yaml',
	F3_VOXEL_LABEL_BUDGET_ROOT / '02_run_voxel_label_budget_suite.yaml',
	F3_VOXEL_LABEL_BUDGET_ROOT / '03_summarize_voxel_label_budget.yaml',
]
F3_VOXEL_LABEL_BUDGET_CURRENT_K6_CONTROL_CONFIGS = [
	F3_CURRENT_K6_CONTROL_ROOT / '07_run_current_k6_voxel_label_budget.yaml',
	F3_CURRENT_K6_CONTROL_ROOT / '08_summarize_current_k6_control.yaml',
]
F3_PERFORMANCE_MIGRATION_VALIDATION_ROOT = (
	F3_ROOT / '92_performance_migration_validation'
)
F3_PERFORMANCE_MIGRATION_SHARED_CONFIGS = [
	F3_PERFORMANCE_MIGRATION_VALIDATION_ROOT / '01_checkpoint_smoke.yaml',
	F3_PERFORMANCE_MIGRATION_VALIDATION_ROOT
	/ '06_summarize_performance_migration.yaml',
]
F3_PERFORMANCE_MIGRATION_EMBEDDING_CONFIGS = [
	F3_PERFORMANCE_MIGRATION_VALIDATION_ROOT / '02_extract_m1_cache_off.yaml',
	F3_PERFORMANCE_MIGRATION_VALIDATION_ROOT / '03_extract_m1_cache_memmap.yaml',
]
F3_PERFORMANCE_MIGRATION_CLUSTERING_CONFIGS = [
	F3_PERFORMANCE_MIGRATION_VALIDATION_ROOT / '04_replay_m1_k6_hmm.yaml',
]
F3_PERFORMANCE_MIGRATION_PSEUDO_TARGET_CONFIGS = [
	F3_PERFORMANCE_MIGRATION_VALIDATION_ROOT / '05_export_m1_k6_pseudo_targets.yaml',
]
F3_VOXEL_ROBUSTNESS_CONFIGS = [
	F3_VOXEL_ROBUSTNESS_ROOT / f'{index:02d}_{name}.yaml'
	for index, name in (
		(1, 'build_voxel_split_datasets'),
		(2, 'run_v0_split_projections'),
		(3, 'run_v1_split_decoders'),
	)
]
F3_VOXEL_DATASET_CONFIGS = [F3_VOXEL_V0_ROOT / '01_build_voxel_supervision.yaml']
F3_VOXEL_TOKEN_PREDICTION_CONFIGS = [
	F3_VOXEL_V0_ROOT / name
	for name in (
		'02_predict_mae_tokens.yaml',
		'06_predict_m1_tokens.yaml',
		'10_predict_m2a_tokens.yaml',
	)
]
F3_VOXEL_PROJECTION_CONFIGS = [
	F3_VOXEL_V0_ROOT / name
	for name in (
		'03_project_mae_nearest.yaml',
		'07_project_m1_nearest.yaml',
		'11_project_m2a_nearest.yaml',
	)
]
F3_VOXEL_DECODER_CONFIGS = sorted(F3_VOXEL_V1_ROOT.glob('*_train_*.yaml'))
F3_VOXEL_INFERENCE_CONFIGS = sorted(F3_VOXEL_V1_ROOT.glob('*_predict_*_voxels.yaml'))
F3_VOXEL_EVALUATION_CONFIGS = sorted(
	[
		*F3_VOXEL_V0_ROOT.glob('*_evaluate_*.yaml'),
		*F3_VOXEL_V1_ROOT.glob('*_evaluate_*.yaml'),
	]
)
F3_VOXEL_REPORT_CONFIGS = sorted(
	[
		*F3_VOXEL_V0_ROOT.glob('*_report_*.yaml'),
		*F3_VOXEL_V1_ROOT.glob('*_report_*.yaml'),
	]
)
F3_BASELINE_ROOT = F3_ROOT / '50_lithology_baselines'
F3_BASELINE_TOKEN_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('01_build_baseline_token_dataset.yaml'),
)
F3_RANDOM_ENCODER_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('01_create_random_checkpoint.yaml'),
)
F3_RANDOM_ENCODER_EMBEDDING_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('02_extract_embeddings.yaml'),
)
F3_RANDOM_ENCODER_TOKEN_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('03_build_token_dataset.yaml'),
)
F3_BASELINE_PROBE_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('02_train_linear_probe.yaml'),
)
F3_RANDOM_ENCODER_PROBE_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('04_train_linear_probe.yaml'),
)
F3_BASELINE_REPORT_CONFIGS = sorted(
	[
		*F3_BASELINE_ROOT.rglob('03_build_report.yaml'),
		*F3_BASELINE_ROOT.rglob('05_build_report.yaml'),
	],
)
F3_BASELINE_COMPARISON_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('05_build_baseline_comparison_report.yaml'),
)
REQUIRED_ACTIVE_CONFIG_GROUPS = (
	('nopims pretraining', NOPIMS_PRETRAINING_CONFIGS),
	('nopims embedding', NOPIMS_EMBEDDING_CONFIGS),
	('nopims clustering', NOPIMS_CLUSTERING_CONFIGS),
	('nopims visualization', NOPIMS_VISUALIZATION_CONFIGS),
	('f3 inspection', F3_INSPECTION_CONFIGS),
	('f3 prepare', F3_PREPARE_CONFIGS),
	('f3 embedding', F3_EMBEDDING_CONFIGS),
	('f3 stratigraphic clustering', F3_STRATIGRAPHIC_CLUSTERING_CONFIGS),
	('f3 strat hmm pretext', F3_STRAT_HMM_PRETEXT_CONFIGS),
	(
		'f3 strat hmm center-trace masked validation',
		F3_STRAT_HMM_CENTER_TRACE_MASKED_VALIDATION_CONFIGS,
	),
	(
		'f3 strat hmm periodic refresh pretext',
		F3_STRAT_HMM_PERIODIC_REFRESH_PRETEXT_CONFIGS,
	),
	(
		'f3 strat hmm periodic refresh validation',
		F3_STRAT_HMM_PERIODIC_REFRESH_VALIDATION_CONFIGS,
	),
	(
		'f3 periodic refresh original split low label',
		F3_PERIODIC_REFRESH_ORIGINAL_SPLIT_CONFIGS,
	),
	('f3 strat hmm shuffled targets', F3_STRAT_HMM_SHUFFLED_TARGET_CONFIGS),
	(
		'f3 strat hmm student embedding',
		F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS,
	),
	(
		'f3 strat hmm student lithology token dataset',
		F3_STRAT_HMM_STUDENT_LITHOLOGY_TOKEN_CONFIGS,
	),
	(
		'f3 strat hmm student lithology probe',
		F3_STRAT_HMM_STUDENT_LITHOLOGY_PROBE_CONFIGS,
	),
	(
		'f3 strat hmm student lithology report',
		F3_STRAT_HMM_STUDENT_LITHOLOGY_REPORT_CONFIGS,
	),
	(
		'f3 strat hmm pseudo-target refresh',
		F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS,
	),
	('f3 strat hmm m1 robustness', F3_STRAT_HMM_M1_ROBUSTNESS_CONFIGS),
	('f3 strat hmm m2a robustness', F3_STRAT_HMM_M2A_ROBUSTNESS_CONFIGS),
	('f3 lithology token dataset', F3_LITHOLOGY_TOKEN_CONFIGS),
	('f3 lithology probe', F3_LITHOLOGY_PROBE_CONFIGS),
	('f3 lithology prediction', F3_LITHOLOGY_PREDICTION_CONFIGS),
	('f3 lithology visualization', F3_LITHOLOGY_VISUALIZATION_CONFIGS),
	('f3 lithology report', F3_LITHOLOGY_REPORT_CONFIGS),
	('f3 voxel dataset', F3_VOXEL_DATASET_CONFIGS),
	('f3 voxel token prediction', F3_VOXEL_TOKEN_PREDICTION_CONFIGS),
	('f3 voxel projection', F3_VOXEL_PROJECTION_CONFIGS),
	('f3 voxel decoder', F3_VOXEL_DECODER_CONFIGS),
	('f3 voxel inference', F3_VOXEL_INFERENCE_CONFIGS),
	('f3 voxel evaluation', F3_VOXEL_EVALUATION_CONFIGS),
	('f3 voxel report', F3_VOXEL_REPORT_CONFIGS),
	('f3 voxel label budget', F3_VOXEL_LABEL_BUDGET_CONFIGS),
	(
		'f3 voxel label budget current K6 control',
		F3_VOXEL_LABEL_BUDGET_CURRENT_K6_CONTROL_CONFIGS,
	),
	(
		'f3 performance migration shared',
		F3_PERFORMANCE_MIGRATION_SHARED_CONFIGS,
	),
	(
		'f3 performance migration embedding',
		F3_PERFORMANCE_MIGRATION_EMBEDDING_CONFIGS,
	),
	(
		'f3 performance migration clustering',
		F3_PERFORMANCE_MIGRATION_CLUSTERING_CONFIGS,
	),
	(
		'f3 performance migration pseudo targets',
		F3_PERFORMANCE_MIGRATION_PSEUDO_TARGET_CONFIGS,
	),
	('f3 baseline token dataset', F3_BASELINE_TOKEN_CONFIGS),
	('f3 random encoder', F3_RANDOM_ENCODER_CONFIGS),
	('f3 random encoder embedding', F3_RANDOM_ENCODER_EMBEDDING_CONFIGS),
	('f3 random encoder token dataset', F3_RANDOM_ENCODER_TOKEN_CONFIGS),
	('f3 baseline probe', F3_BASELINE_PROBE_CONFIGS),
	('f3 random encoder probe', F3_RANDOM_ENCODER_PROBE_CONFIGS),
	('f3 baseline report', F3_BASELINE_REPORT_CONFIGS),
	('f3 baseline comparison', F3_BASELINE_COMPARISON_CONFIGS),
)


@pytest.mark.parametrize('config_path', ALL_CONFIGS, ids=str)
def test_all_repository_configs_load_and_resolve_supported_stages(
	config_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', '/workspace')
	monkeypatch.setenv('F3_ROOT', '/test/f3')
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		'0' * 64,
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_MANIFEST_SHA256',
		'0' * 64,
	)
	for k in (6, 8, 10):
		for name in ('POSTERIOR', 'VALID_TOKENS', 'METADATA'):
			monkeypatch.setenv(
				f'SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_K{k}_{name}_SHA256',
				'0' * 64,
			)
	config = load_config(config_path)

	assert isinstance(config, dict)
	assert config
	resolver = CORE_CONFIG_RESOLVERS.get(frozenset(config))
	if resolver is not None:
		resolved = resolver(config)
		assert resolved['stage']


def test_repository_configs_preserve_legacy_optimization_defaults() -> None:
	training = resolve_mae_training_config(
		load_config(Path('proc/configs/seis_ssl_cluster/train_amp_mae.yaml')),
	)
	assert training['data']['finite_check_mode'] == 'strict'
	assert {
		key: training['train'][key]
		for key in (
			'prefetch_factor',
			'persistent_workers',
			'amp_dtype',
			'runtime_check_mode',
			'stage_timing',
		)
	} == {
		'prefetch_factor': 2,
		'persistent_workers': True,
		'amp_dtype': 'auto',
		'runtime_check_mode': 'once',
		'stage_timing': False,
	}

	embedding_raw = load_config(
		Path('proc/configs/seis_ssl_cluster/extract_embeddings.yaml'),
	)
	for key in (
		'prefetch_queue_depth',
		'amp',
		'amp_dtype',
		'stage_timing',
		'preprocessing_cache',
	):
		embedding_raw['embedding'].pop(key)
	embedding = resolve_embedding_extraction_config(embedding_raw)
	embedding_settings = extraction_settings_from_config(
		embedding,
		checkpoint_config=training,
	)
	assert embedding_settings.average_chunk_size_x == 16
	assert embedding_settings.prefetch_queue_depth == 0
	assert embedding_settings.amp is False
	assert embedding_settings.amp_dtype == 'auto'
	assert embedding_settings.stage_timing is False
	assert embedding_settings.preprocessing_cache.mode == 'off'
	assert embedding_settings.preprocessing_cache.chunk_size_x == 16
	assert embedding_settings.preprocessing_cache.reuse is True
	assert embedding_settings.preprocessing_cache.cleanup is False

	hmm_raw = load_config(F3_STRATIGRAPHIC_CLUSTERING_CONFIGS[0])
	hmm_raw['clustering'].pop('stage_timing', None)
	hmm_raw['clustering']['stratigraphic_hmm'].pop(
		'prepared_feature_cache',
		None,
	)
	hmm = resolve_clustering_config(hmm_raw)
	assert clustering_settings_from_config(hmm).stage_timing is False
	prepared_cache = stratigraphic_hmm_settings_from_config(
		hmm,
	).prepared_feature_cache
	assert prepared_cache.chunk_size_tokens == 65_536
	assert prepared_cache.reuse is True
	assert prepared_cache.force_rebuild is False
	assert prepared_cache.cleanup is False
	assert prepared_cache.persist is True


@pytest.mark.parametrize(('group_name', 'configs'), REQUIRED_ACTIVE_CONFIG_GROUPS)
def test_active_config_groups_are_not_empty(
	group_name: str,
	configs: list[Path] | list[tuple[Path, str]],
) -> None:
	assert configs, f'{group_name} active config list must not be empty'


@pytest.mark.parametrize('config_path', NOPIMS_PRETRAINING_CONFIGS)
def test_active_nopims_pretraining_configs_resolve(config_path: Path) -> None:
	resolve_mae_training_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_EMBEDDING_CONFIGS)
def test_active_nopims_embedding_configs_resolve(config_path: Path) -> None:
	resolve_embedding_extraction_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_CLUSTERING_CONFIGS)
def test_active_nopims_clustering_configs_resolve(config_path: Path) -> None:
	resolve_clustering_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_VISUALIZATION_CONFIGS)
def test_active_nopims_cluster_visualization_configs_resolve(
	config_path: Path,
) -> None:
	resolve_cluster_visualization_config(load_config(config_path))


@pytest.mark.parametrize(('config_path', 'stage'), F3_INSPECTION_CONFIGS)
def test_active_f3_inspection_configs_resolve(
	config_path: Path,
	stage: str,
) -> None:
	resolve_f3_facies_inspection_config(load_config(config_path), stage=stage)


@pytest.mark.parametrize('config_path', F3_PREPARE_CONFIGS)
def test_active_f3_prepare_configs_resolve(config_path: Path) -> None:
	f3_prepare_volume_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_EMBEDDING_CONFIGS,
		*F3_RANDOM_ENCODER_EMBEDDING_CONFIGS,
		*F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS,
		*F3_STRAT_HMM_PERIODIC_REFRESH_EMBEDDING_CONFIGS,
	],
)
def test_active_f3_embedding_configs_resolve(
	config_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	resolve_embedding_extraction_config(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRATIGRAPHIC_CLUSTERING_CONFIGS)
def test_active_f3_stratigraphic_clustering_configs_resolve(
	config_path: Path,
) -> None:
	resolve_clustering_config(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_PERFORMANCE_MIGRATION_SHARED_CONFIGS)
def test_active_f3_performance_migration_shared_configs_resolve(
	config_path: Path,
) -> None:
	config = performance_migration_validation_config_from_mapping(
		load_config(config_path),
	)

	assert config.current_git_sha == '332478be21a021e46ee6c1d9423f14859b0cd819'
	assert config.historical_baseline_sha == '7731f341a293ea0c5cb5c5dfabba574148861e3a'
	assert config.compatibility['m1_historical_finite_check_mode'] == 'off'


@pytest.mark.parametrize(
	'config_path',
	F3_PERFORMANCE_MIGRATION_EMBEDDING_CONFIGS,
)
def test_active_f3_performance_migration_embedding_configs_resolve(
	config_path: Path,
) -> None:
	config = resolve_embedding_extraction_config(load_config(config_path))

	assert config['embedding']['window_size'] == [128, 128, 128]
	assert config['embedding']['overlap'] == [112, 64, 64]
	assert config['embedding']['amp'] is False
	assert config['embedding']['prefetch_queue_depth'] == 0


@pytest.mark.parametrize(
	'config_path',
	F3_PERFORMANCE_MIGRATION_CLUSTERING_CONFIGS,
)
def test_active_f3_performance_migration_clustering_configs_resolve(
	config_path: Path,
) -> None:
	config = resolve_clustering_config(load_config(config_path))

	assert config['clustering']['k_values'] == [6]
	assert config['clustering']['stratigraphic_hmm']['iterations'] == 10


@pytest.mark.parametrize(
	'config_path',
	F3_PERFORMANCE_MIGRATION_PSEUDO_TARGET_CONFIGS,
)
def test_active_f3_performance_migration_pseudo_target_configs_load(
	config_path: Path,
) -> None:
	config = load_config(config_path)

	assert config['export'] == {
		'output_root': (
			'/workspace/artifacts/seis_ssl_cluster/migration_validation/f3/'
			'facies_benchmark_v1/main_332478be/pseudo_targets/'
			'm1_k6_current_replay'
		),
		'k': 6,
		'survey_id': 'f3_facies_benchmark',
		'confidence': 1.0,
		'schema_version': 1,
		'write_boundary_weight': False,
		'only_missing': True,
		'overwrite': False,
	}


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_PRETEXT_CONFIGS)
def test_active_f3_strat_hmm_pretext_configs_resolve(
	config_path: Path,
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		'0' * 64,
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_MANIFEST_SHA256',
		'0' * 64,
	)
	for k in (6, 8, 10):
		for name, offset in (
			('POSTERIOR', 0),
			('VALID_TOKENS', 10),
			('METADATA', 20),
		):
			monkeypatch.setenv(
				f'SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_K{k}_{name}_SHA256',
				f'{k + offset:064x}',
			)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_state_posterior_manifest',
		lambda _path, *, validate_array_semantics: (
			_active_posterior_manifest()
			if not validate_array_semantics
			else pytest.fail('config validation requested full posterior arrays')
		),
	)
	monkeypatch.setattr(
		lateral_targets,
		'load_multi_head_lateral_target_manifest',
		lambda _path, *, validate_array_semantics: (
			_active_lateral_manifest()
			if not validate_array_semantics
			else pytest.fail('config validation requested full lateral arrays')
		),
	)
	monkeypatch.setattr(
		xy_neighbor_consensus_targets,
		'load_multi_head_xy_neighbor_consensus_target_manifest',
		lambda _path, *, validate_array_semantics: (
			_active_xy_neighbor_consensus_manifest()
			if not validate_array_semantics
			else pytest.fail(
				'config validation requested full XY-neighbour consensus arrays'
			)
		),
	)
	monkeypatch.setattr(
		xy_neighbor_unanimous_targets,
		'load_multi_head_xy_neighbor_unanimous_target_manifest',
		lambda _path, *, validate_array_semantics: (
			_active_xy_neighbor_unanimous_manifest()
			if not validate_array_semantics
			else pytest.fail(
				'config validation requested full XY-neighbour unanimous arrays'
			)
		),
	)
	resolve_strat_hmm_pretext_config(
		_config_with_existing_strat_hmm_pretext_inputs(config_path, tmp_path),
	)


def test_active_f3_multi_head_pretext_config_contract(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		'0' * 64,
	)
	config_paths = [
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '04_train_nocons_full.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '05_train_cons010_full.yaml',
	]
	for config_path, consistency_weight, model_tag in zip(
		config_paths,
		(0.0, 0.1),
		(
			'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
			'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
		),
		strict=True,
	):
		raw = load_config(config_path)
		assert raw['identity']['model_tag'] == model_tag
		assert raw['identity']['scientific_identity'] == {
			'experiment_role': 'multi_head_ordered_pretext',
			'variant': 'nocons' if consistency_weight == 0.0 else 'cons010',
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_manifest_sha256': '0' * 64,
			'consistency_policy': 'normalized_order_smooth_l1_v1',
		}
		assert raw['head'] == {
			'spec': 'multi_resolution_ordered_prototypes_v1',
			'ks': [6, 8, 10],
			'projection_dim': 128,
			'temperature': 0.1,
			'normalize': True,
		}
		assert raw['loss'] == {
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'entropy_floor': None,
			'consistency_weight': consistency_weight,
			'consistency_beta': 0.1,
			'distillation_weight': 0.2,
		}

	no_consistency, main = [
		resolve_strat_hmm_pretext_config(
			_config_with_existing_strat_hmm_pretext_inputs(config_path, tmp_path)
		)
		for config_path in config_paths
	]
	comparison = deepcopy(main)
	comparison['loss']['consistency_weight'] = 0.0
	comparison['identity']['scientific_identity']['consistency_weight'] = 0.0
	comparison['identity']['scientific_identity']['variant'] = 'nocons'
	comparison['identity']['model_tag'] = no_consistency['identity']['model_tag']
	comparison['paths']['output_root'] = no_consistency['paths']['output_root']
	comparison['pseudo_targets']['manifest'] = no_consistency['pseudo_targets'][
		'manifest'
	]
	comparison['identity']['scientific_identity']['target_manifest_sha256'] = (
		no_consistency['identity']['scientific_identity']['target_manifest_sha256']
	)
	assert comparison == no_consistency


@pytest.mark.parametrize(
	'config_path', F3_STRAT_HMM_CENTER_TRACE_MASKED_VALIDATION_CONFIGS
)
def test_active_f3_center_trace_masked_validation_configs_load(
	config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', '/test/artifacts/seis_ssl_cluster'
	)
	config = load_config(config_path)
	assert set(config) == {
		'artifact_root',
		'experiment_root',
		'target_manifest',
		'hard_full_config',
		'hard_handoff',
		'center_trace_masked_smoke_config',
		'center_trace_masked_full_config',
		'center_trace_masked_embedding_config',
	}


def test_active_f3_periodic_refresh_configs_resolve(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	artifact_root = Path('/workspace/artifacts/seis_ssl_cluster')
	target = artifact_root / (
		'pseudo_targets/f3/facies_benchmark_v1/'
		'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/'
		'multi_head_target_manifest.json'
	)
	if not target.is_file():
		pytest.skip('real F3 periodic-refresh artifacts are not available')
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', '/workspace')
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		file_sha256(target),
	)
	for path in F3_STRAT_HMM_PERIODIC_REFRESH_PRETEXT_CONFIGS:
		resolved = resolve_strat_hmm_pretext_config(load_config(path))
		assert resolved['identity']['model_tag'].endswith('distill_v1')
	for path in F3_STRAT_HMM_PERIODIC_REFRESH_EMBEDDING_CONFIGS:
		resolved = resolve_embedding_extraction_config(load_config(path))
		assert Path(resolved['embeddings']['checkpoint']).name == 'selected.pt'
	for path in F3_STRAT_HMM_PERIODIC_REFRESH_VALIDATION_CONFIGS:
		resolved = load_f3_center_trace_masked_periodic_refresh_validation_config(path)
		assert resolved.target_manifest == target.resolve()


def test_active_f3_periodic_refresh_original_split_configs_have_closed_roles(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', '/workspace')
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', '/test/artifacts/seis_ssl_cluster'
	)
	monkeypatch.setenv('F3_ROOT', '/test/f3')
	root = F3_PERIODIC_REFRESH_ORIGINAL_SPLIT_ROOT
	audit = load_config(root / '00_audit_periodic_refresh_screening.yaml')
	assert set(audit) == {
		'artifact_root',
		'workspace_root',
		'source_hard_manifest',
		'hard_full_config',
		'hard_pretraining_handoff',
		'candidate_decoder_config',
		'center_trace_masked_config',
		'periodic_refresh_validation_config',
		'periodic_refresh_full_config',
		'periodic_refresh_handoff',
		'periodic_refresh_embeddings_dir',
		'output_path',
	}
	run = load_config(root / '01_run_periodic_refresh_voxel_label_budget.yaml')
	assert run['multi_head']['budgets'] == ['cap25', 'cap50', 'cap100']
	assert run['multi_head']['subsample_seeds'] == [0, 1, 2, 3, 4]
	assert run['multi_head']['seed_policy'] == {
		'base_seed': 42000,
		'add_subsample_seed': True,
	}
	assert len(run['multi_head']['candidates']) == 1
	assert run['multi_head']['candidates'][0]['model_id'] == (
		'mh_ctmask010_refresh3ep_hmm2_nocons'
	)
	summary = load_config(
		root / '02_summarize_periodic_refresh_voxel_label_budget.yaml'
	)
	assert set(summary) == {'run_config'}


def test_active_f3_m5_ls_config_contract(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Keep M5-LS changes limited to its target representation identity."""
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', '0' * 64)
	root = F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT
	candidates = [
		load_config(root / '01_export_lateral_beta010.yaml'),
		load_config(root / '02_export_lateral_beta025.yaml'),
		load_config(root / '03_export_lateral_beta050.yaml'),
	]
	assert [
		candidate['smoothing']['pairwise_strength_ratio'] for candidate in candidates
	] == [0.10, 0.25, 0.50]
	assert all(candidate['outputs'] == {'overwrite': False} for candidate in candidates)
	assert [
		Path(candidate['handoff_manifest']).parent.name for candidate in candidates
	] == [
		'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1_lateral_mean_field_beta010_v1',
		'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1_lateral_mean_field_beta025_v1',
		'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1_lateral_mean_field_beta050_v1',
	]

	for baseline_name, lateral_name in (
		('02_train_nocons_smoke.yaml', '05_train_lateral_smoke.yaml'),
		('04_train_nocons_full.yaml', '06_train_lateral_full.yaml'),
	):
		baseline = load_config(F3_STRAT_HMM_MULTI_HEAD_ROOT / baseline_name)
		lateral = load_config(root / lateral_name)
		assert lateral['identity']['model_tag'] == (
			'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1'
		)
		assert lateral['pseudo_targets']['target_representation'] == (
			'lateral_mean_field_hard_labels_v1'
		)
		assert lateral['pseudo_targets']['min_confidence'] == 0.0
		assert lateral['identity']['scientific_identity']['target_semantics'] == (
			'ordered_hmm_edge_aware_lateral_mean_field_hard_v1'
		)
		assert lateral['loss']['consistency_weight'] == 0.0

		baseline['paths']['output_root'] = lateral['paths']['output_root']
		baseline['identity']['model_tag'] = lateral['identity']['model_tag']
		baseline['pseudo_targets']['manifest'] = lateral['pseudo_targets']['manifest']
		baseline['pseudo_targets']['target_representation'] = lateral['pseudo_targets'][
			'target_representation'
		]
		baseline_scientific = baseline['identity']['scientific_identity']
		lateral_scientific = lateral['identity']['scientific_identity']
		assert isinstance(baseline_scientific, dict)
		assert isinstance(lateral_scientific, dict)
		for key in (
			'experiment_role',
			'variant',
			'target_manifest_sha256',
			'target_representation',
			'target_semantics',
			'supervised_loss',
			'consistency_policy',
			'consistency_weight',
		):
			baseline_scientific.pop(key, None)
			lateral_scientific.pop(key, None)
		assert baseline == lateral

	calibration = load_config(root / '04_calibrate_lateral_targets.yaml')
	assert set(calibration) == {
		'artifact_root',
		'source_hard_manifest',
		'source_posterior_manifest',
		'candidate_manifests',
		'selected_manifest',
		'calibration_handoff',
		'calibration_report',
		'hard_full_config',
		'lateral_smoke_config',
		'lateral_full_config',
	}
	assert list(calibration['candidate_manifests']) == ['beta010', 'beta025', 'beta050']
	assert calibration['selected_manifest'].endswith(
		'strat_hmm_multi_k6810_lateral_mean_field_selected_v1/'
		'multi_head_lateral_target_handoff.json'
	)
	validator = load_config(root / '08_validate_lateral_pretraining.yaml')
	assert set(validator) == {
		'artifact_root',
		'experiment_root',
		'calibration_handoff',
		'selected_manifest',
		'hard_full_config',
		'hard_handoff',
		'lateral_smoke_config',
		'lateral_full_config',
	}
	embedding = load_config(root / '07_extract_lateral_embeddings.yaml')
	assert embedding['embeddings']['checkpoint'].endswith(
		'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1/best.pt'
	)
	assert embedding['embeddings']['output_dir'].endswith(
		'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1/overlap_x16'
	)


@pytest.mark.parametrize(
	'config_path',
	F3_VOXEL_LABEL_BUDGET_CURRENT_K6_CONTROL_CONFIGS,
)
def test_active_f3_current_k6_voxel_control_configs_resolve(
	config_path: Path,
) -> None:
	config = f3_lithology_voxel_label_budget_control_config_from_mapping(
		load_config(config_path)
	)

	assert config.job_count == 15
	assert config.candidate.model_id == 'm1_current_k6'


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_SHUFFLED_TARGET_CONFIGS)
def test_active_f3_strat_hmm_shuffled_target_configs_resolve(
	config_path: Path,
) -> None:
	f3_shuffled_hmm_target_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS)
def test_active_f3_strat_hmm_pseudo_target_refresh_configs_resolve(
	config_path: Path,
	tmp_path: Path,
) -> None:
	resolve_strat_hmm_pseudo_target_config(
		_config_with_existing_strat_hmm_refresh_inputs(config_path, tmp_path),
	)


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_ROBUSTNESS_CONFIGS)
def test_active_f3_strat_hmm_m1_robustness_configs_parse(
	config_path: Path,
) -> None:
	assert load_config(config_path)


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_LABEL_BUDGET_BUILD_CONFIGS)
def test_active_f3_strat_hmm_m1_label_budget_build_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_label_budget_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_LABEL_BUDGET_PROBE_CONFIGS)
def test_active_f3_strat_hmm_m1_label_budget_probe_configs_resolve_schema(
	config_path: Path,
	tmp_path: Path,
) -> None:
	raw = load_config(config_path)
	raw['suite']['manifest'] = str(_write_label_budget_manifest(tmp_path))
	raw['labels']['class_info'] = str(_write_class_info(tmp_path))

	config = f3_lithology_label_budget_probe_config_from_mapping(raw)

	assert config.probe.random_state == 42
	assert len(config.probe_configs) == 2


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_SPLIT_INVENTORY_CONFIGS)
def test_active_f3_strat_hmm_m1_split_inventory_configs_resolve(
	config_path: Path,
) -> None:
	config = f3_lithology_split_inventory_config_from_mapping(load_config(config_path))

	assert config.include_base_split_as_split_000 is True
	assert config.random_seeds == (0, 1, 2, 3, 4)
	assert config.min_validation_tokens_per_class == {'default': 1, '3': 100, '5': 20}


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_SPLIT_DATASET_CONFIGS)
def test_active_f3_strat_hmm_m1_split_dataset_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_split_sweep_dataset_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_SPLIT_PROBE_CONFIGS)
def test_active_f3_strat_hmm_m1_split_probe_configs_resolve_schema(
	config_path: Path,
	tmp_path: Path,
) -> None:
	raw = load_config(config_path)
	raw['suite']['dataset_manifest'] = str(_write_split_dataset_manifest(tmp_path))
	raw['labels']['class_info'] = str(_write_class_info(tmp_path))

	config = f3_lithology_split_sweep_probe_config_from_mapping(raw)

	assert config.probe.random_state == 42
	assert len(config.probe_configs) == 2


def test_active_f3_strat_hmm_m2a_robustness_pair_contract(
	tmp_path: Path,
) -> None:
	baseline_tag = 'strat_hmm_pretext_m1_k6_topblock1_distill'
	candidate_tag = 'strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill'
	m1_inventory = (
		'/workspace/artifacts/seis_ssl_cluster/lithology/f3/'
		'facies_benchmark_v1/robustness/split_index_m1_v1/'
		'split_inventory_manifest.json'
	)

	label_raw = load_config(F3_STRAT_HMM_M2A_LABEL_BUDGET_BUILD_CONFIG)
	label_config = f3_lithology_label_budget_config_from_mapping(label_raw)
	assert label_config.baseline.model_tag == baseline_tag
	assert label_config.candidate.model_tag == candidate_tag
	assert label_config.per_class_caps == (25, 50, 100, None)
	assert label_config.subsample_seeds == (0, 1, 2, 3, 4)
	assert label_config.reuse_full_validation is True
	assert label_config.baseline.token_dataset_root != (
		label_config.candidate.token_dataset_root
	)

	split_raw = load_config(F3_STRAT_HMM_M2A_SPLIT_DATASET_CONFIG)
	split_config = f3_lithology_split_sweep_dataset_config_from_mapping(split_raw)
	assert split_config.baseline.model_tag == baseline_tag
	assert split_config.candidate.model_tag == candidate_tag
	assert str(split_config.split_inventory_manifest) == m1_inventory
	assert split_config.baseline.checkpoint.name == 'best.pt'
	assert split_config.candidate.checkpoint.name == 'best.pt'
	assert split_config.baseline.embeddings_dir.name == 'overlap_x16'
	assert split_config.candidate.embeddings_dir.name == 'overlap_x16'
	assert split_config.baseline.embeddings_dir != split_config.candidate.embeddings_dir
	assert split_config.baseline.checkpoint != split_config.candidate.checkpoint

	m1_split_raw = load_config(
		F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '04_generate_split_inventories.yaml',
	)
	assert m1_split_raw['split_sweep']['split_ids'] == [
		f'split_{index:03d}' for index in range(6)
	]
	assert 'amp_mae_' not in (
		F3_STRAT_HMM_M2A_LABEL_BUDGET_BUILD_CONFIG.read_text()
		+ F3_STRAT_HMM_M2A_SPLIT_DATASET_CONFIG.read_text()
	)

	label_probe_raw = load_config(F3_STRAT_HMM_M2A_LABEL_BUDGET_PROBE_CONFIG)
	label_probe_raw['suite']['manifest'] = str(
		_write_label_budget_manifest(tmp_path),
	)
	label_probe_raw['labels']['class_info'] = str(_write_class_info(tmp_path))
	label_probe = f3_lithology_label_budget_probe_config_from_mapping(
		label_probe_raw,
	)
	assert label_probe.probe.random_state == 42
	assert label_probe.probe.feature_scaling == 'standard'
	assert label_probe.probe.class_weight == 'balanced'

	split_probe_raw = load_config(F3_STRAT_HMM_M2A_SPLIT_PROBE_CONFIG)
	split_probe_raw['suite']['dataset_manifest'] = str(
		_write_split_dataset_manifest(tmp_path),
	)
	split_probe_raw['labels']['class_info'] = str(_write_class_info(tmp_path))
	split_probe = f3_lithology_split_sweep_probe_config_from_mapping(
		split_probe_raw,
	)
	assert split_probe.probe == label_probe.probe

	label_summary = f3_lithology_label_budget_summary_config_from_mapping(
		load_config(F3_STRAT_HMM_M2A_LABEL_BUDGET_SUMMARY_CONFIG),
	)
	assert label_summary.suite_root == label_config.output_root
	assert label_summary.inputs['suite_manifest'] == (
		label_config.output_root / 'suite_manifest.json'
	)

	split_summary = f3_lithology_split_summary_config_from_mapping(
		load_config(F3_STRAT_HMM_M2A_SPLIT_SUMMARY_CONFIG),
	)
	assert split_summary.suite_root == split_config.output_root
	assert (
		split_summary.inputs['split_inventory_manifest']
		== split_config.split_inventory_manifest
	)


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_LITHOLOGY_TOKEN_CONFIGS,
		*F3_RANDOM_ENCODER_TOKEN_CONFIGS,
		*F3_STRAT_HMM_STUDENT_LITHOLOGY_TOKEN_CONFIGS,
	],
)
def test_active_f3_lithology_token_dataset_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_token_dataset_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_BASELINE_TOKEN_CONFIGS)
def test_active_f3_baseline_token_dataset_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_baseline_token_dataset_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_RANDOM_ENCODER_CONFIGS)
def test_active_f3_random_encoder_configs_resolve(config_path: Path) -> None:
	random_mae_checkpoint_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_LITHOLOGY_PROBE_CONFIGS,
		*F3_BASELINE_PROBE_CONFIGS,
		*F3_RANDOM_ENCODER_PROBE_CONFIGS,
		*F3_STRAT_HMM_STUDENT_LITHOLOGY_PROBE_CONFIGS,
	],
)
def test_active_f3_lithology_probe_configs_resolve(config_path: Path) -> None:
	f3_lithology_probe_config_from_mapping(load_config(config_path), load_classes=False)


@pytest.mark.parametrize('config_path', F3_LITHOLOGY_PREDICTION_CONFIGS)
def test_active_f3_lithology_prediction_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_prediction_config_from_mapping(
		load_config(config_path),
		load_classes=False,
	)


@pytest.mark.parametrize('config_path', F3_LITHOLOGY_VISUALIZATION_CONFIGS)
def test_active_f3_lithology_visualization_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_visualization_config_from_mapping(
		load_config(config_path),
		load_classes=False,
	)


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_LITHOLOGY_REPORT_CONFIGS,
		*F3_BASELINE_REPORT_CONFIGS,
		*F3_STRAT_HMM_STUDENT_LITHOLOGY_REPORT_CONFIGS,
	],
)
def test_active_f3_lithology_report_configs_resolve(config_path: Path) -> None:
	raw = load_config(config_path)

	f3_lithology_report_config_from_mapping(raw)
	f3_lithology_publish_config_from_mapping(raw.get('publish'))


@pytest.mark.parametrize('config_path', F3_VOXEL_DATASET_CONFIGS)
def test_active_f3_voxel_dataset_configs_resolve(config_path: Path) -> None:
	f3_lithology_voxel_dataset_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_VOXEL_TOKEN_PREDICTION_CONFIGS)
def test_active_f3_voxel_token_prediction_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_prediction_config_from_mapping(
		load_config(config_path), load_classes=False
	)


@pytest.mark.parametrize('config_path', F3_VOXEL_PROJECTION_CONFIGS)
def test_active_f3_voxel_projection_configs_resolve(
	config_path: Path, tmp_path: Path
) -> None:
	raw = _projection_config_with_synthetic_source(config_path, tmp_path)
	f3_lithology_voxel_projection_config_from_mapping(raw)


@pytest.mark.parametrize('config_path', F3_VOXEL_DECODER_CONFIGS)
def test_active_f3_voxel_decoder_configs_use_canonical_identity(
	config_path: Path,
) -> None:
	raw = load_config(config_path)
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	assert config.decoder.spec == VOXEL_DECODER_SPEC
	assert config.decoder.upsample_mode == VOXEL_DECODER_UPSAMPLE_MODE
	assert config.decoder.normalization == VOXEL_DECODER_NORMALIZATION
	expected_dir = (
		VOXEL_DECODER_SMOKE_SPEC
		if config_path.name.endswith('_smoke.yaml')
		else VOXEL_DECODER_SPEC
	)
	assert config.output_dir.name == expected_dir
	assert OLD_VOXEL_DECODER_SPEC not in config_path.read_text(encoding='utf-8')


@pytest.mark.parametrize('config_path', F3_VOXEL_INFERENCE_CONFIGS)
def test_active_f3_voxel_inference_configs_resolve(config_path: Path) -> None:
	f3_lithology_voxel_inference_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_VOXEL_EVALUATION_CONFIGS)
def test_active_f3_voxel_evaluation_configs_resolve(
	config_path: Path, tmp_path: Path
) -> None:
	raw = _config_with_available_output(config_path, tmp_path)
	f3_lithology_voxel_evaluation_config_from_mapping(raw)


@pytest.mark.parametrize('config_path', F3_VOXEL_REPORT_CONFIGS)
def test_active_f3_voxel_report_configs_resolve(
	config_path: Path, tmp_path: Path
) -> None:
	raw = _config_with_available_output(config_path, tmp_path)
	f3_lithology_voxel_report_config_from_mapping(raw)


def test_active_f3_voxel_label_budget_dataset_config_resolves() -> None:
	config = f3_lithology_voxel_label_budget_dataset_config_from_mapping(
		load_config(F3_VOXEL_LABEL_BUDGET_CONFIGS[0])
	)

	assert config.suite_name == 'f3_voxel_label_budget_original_v1'
	assert config.budgets == ('cap25', 'cap50', 'cap100')
	assert config.subsample_seeds == (0, 1, 2, 3, 4)
	assert config.patch_size_xyz == (8, 8, 8)
	assert config.require_all_classes is True
	assert config.overwrite is False


def test_active_f3_voxel_label_budget_suite_config_resolves() -> None:
	config = f3_lithology_voxel_label_budget_suite_config_from_mapping(
		load_config(F3_VOXEL_LABEL_BUDGET_CONFIGS[1])
	)

	assert config.budgets == ('cap25', 'cap50', 'cap100')
	assert config.subsample_seeds == (0, 1, 2, 3, 4)
	assert config.base_seed == 42000
	assert config.add_subsample_seed is True
	assert config.train.epochs == 50
	assert config.train.batch_size == 1
	assert config.train.sampling_mode == 'uniform_tiles_with_replacement'
	assert config.train.steps_per_epoch == 440
	assert config.train.seed == 42000
	assert config.write_probabilities is False
	assert config.publish_individual_reports is False
	assert tuple(model.role for model in config.models) == ('mae', 'm1', 'm2a')
	assert len(set(config.full_label_decoder_runs.values())) == 3


def test_active_f3_voxel_label_budget_summary_config_resolves() -> None:
	config = f3_lithology_voxel_label_budget_results_config_from_mapping(
		load_config(F3_VOXEL_LABEL_BUDGET_CONFIGS[2])
	)

	assert config.decision.minimum_positive_budgets == 2
	assert config.decision.minimum_primary_wins == 4
	assert config.decision.negative_budget_count == 2
	assert config.decision.monitored_class_ids == (3, 5)
	assert config.decision.major_degradation_delta == -0.05
	assert config.decision.systematic_degradation_budget_count == 2
	assert config.publish.enabled is True
	assert config.publish.output_dir == Path(
		'results/f3/facies_benchmark_v1/voxel_lithology_label_budget_v1'
	)


def test_active_f3_voxel_paired_experiment_contract() -> None:
	model_tags = (
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		'strat_hmm_pretext_m1_k6_topblock1_distill',
		'strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill',
	)
	token_configs = [load_config(path) for path in F3_VOXEL_TOKEN_PREDICTION_CONFIGS]
	assert tuple(config['model']['tag'] for config in token_configs) == model_tags
	assert all(config['model']['freeze_encoder'] is True for config in token_configs)
	assert all(
		config['embeddings']['spec'] == 'overlap_x16' for config in token_configs
	)
	assert all(
		config['probe']['spec'] == 'linear_balanced_v1' for config in token_configs
	)
	assert all(
		Path(config['probe']['probe_joblib']).parent.name == 'linear_balanced_v1'
		for config in token_configs
	)
	voxel_dataset_raw = load_config(F3_VOXEL_DATASET_CONFIGS[0])
	assert all(
		config['labels']['png_label_inventory']
		== voxel_dataset_raw['labels']['png_label_inventory']
		for config in token_configs
	)
	assert len(
		{config['predictions']['output_dir'] for config in token_configs}
	) == len(token_configs)

	voxel_dataset = f3_lithology_voxel_dataset_config_from_mapping(
		load_config(F3_VOXEL_DATASET_CONFIGS[0])
	)
	training_configs = [load_config(path) for path in F3_VOXEL_DECODER_CONFIGS]
	assert all(
		config['decoder'] == training_configs[0]['decoder']
		for config in training_configs
	)
	smoke_configs = [
		config
		for path, config in zip(F3_VOXEL_DECODER_CONFIGS, training_configs, strict=True)
		if path.name.endswith('_smoke.yaml')
	]
	assert all(
		config['tiles']['context_halo_tokens'] == [1, 1, 1] for config in smoke_configs
	)
	full_configs = [
		config
		for path, config in zip(F3_VOXEL_DECODER_CONFIGS, training_configs, strict=True)
		if path.name.endswith('_full.yaml')
	]
	assert tuple(config['model']['tag'] for config in full_configs) == model_tags
	assert all(config['model']['freeze_encoder'] is True for config in full_configs)
	assert all(config['embeddings']['spec'] == 'overlap_x16' for config in full_configs)
	assert len({config['voxel_dataset']['input_dir'] for config in full_configs}) == 1
	assert (
		Path(full_configs[0]['voxel_dataset']['input_dir']) == voxel_dataset.output_dir
	)
	assert all(
		config['decoder'] == full_configs[0]['decoder'] for config in full_configs
	)
	assert all(config['tiles'] == full_configs[0]['tiles'] for config in full_configs)
	assert all(config['train'] == full_configs[0]['train'] for config in full_configs)
	assert len({config['outputs']['output_dir'] for config in full_configs}) == 3
	assert all(config['train']['seed'] == 42 for config in full_configs)
	assert all(
		Path(config['outputs']['output_dir']).name == VOXEL_DECODER_SPEC
		for config in full_configs
	)

	inference_configs = [
		f3_lithology_voxel_inference_config_from_mapping(load_config(path))
		for path in F3_VOXEL_INFERENCE_CONFIGS
	]
	assert tuple(config.model['tag'] for config in inference_configs) == model_tags
	assert all(config.checkpoint.name == 'best.pt' for config in inference_configs)
	assert all(
		config.checkpoint.parent.name == VOXEL_DECODER_SPEC
		for config in inference_configs
	)
	assert all(
		config.output_dir.name == VOXEL_DECODER_SPEC for config in inference_configs
	)
	assert all(
		'mae_latest.pt' not in str(config.checkpoint) for config in inference_configs
	)
	assert len({config.output_dir for config in inference_configs}) == 3
	assert model_tags[1] in str(inference_configs[1].output_dir)
	assert model_tags[2] in str(inference_configs[2].output_dir)

	for config_path in F3_VOXEL_V1_ROOT.glob('*_evaluate_*.yaml'):
		raw = load_config(config_path)
		assert Path(raw['voxel_predictions']['input_dir']).name == VOXEL_DECODER_SPEC
		assert Path(raw['outputs']['output_dir']).name == VOXEL_DECODER_SPEC
	for config_path in F3_VOXEL_V1_ROOT.glob('*_report_*.yaml'):
		raw = load_config(config_path)
		assert Path(raw['voxel_predictions']['input_dir']).name == VOXEL_DECODER_SPEC
		assert Path(raw['evaluation']['input_dir']).name == VOXEL_DECODER_SPEC
		assert Path(raw['outputs']['output_dir']).name == VOXEL_DECODER_SPEC
		assert Path(raw['publish']['output_dir']).name == VOXEL_DECODER_SPEC
		assert raw['publish']['enabled'] is False
	for config_path in (
		*F3_VOXEL_V1_ROOT.glob('*.yaml'),
		F3_VOXEL_ROBUSTNESS_CONFIGS[2],
		F3_VOXEL_RESULTS_ROOT / '01_summarize_original_split.yaml',
	):
		assert OLD_VOXEL_DECODER_SPEC not in config_path.read_text(encoding='utf-8')


def test_active_f3_voxel_final_summary_publish_order_contract() -> None:
	original = f3_lithology_voxel_results_config_from_mapping(
		load_config(F3_VOXEL_RESULTS_ROOT / '01_summarize_original_split.yaml')
	)
	robustness = f3_lithology_voxel_split_summary_config_from_mapping(
		load_config(
			F3_VOXEL_ROBUSTNESS_ROOT / '04_summarize_voxel_split_robustness.yaml'
		)
	)

	assert original.publish.enabled is False
	assert robustness.publish.enabled is True
	assert robustness.original_summary_dir == original.output_dir
	assert robustness.publish.output_dir == original.publish.output_dir
	assert robustness.artifact_root is not None
	assert robustness.f3_root is not None
	robustness.suite_root.relative_to(robustness.artifact_root)
	with pytest.raises(ValueError, match='is not in the subpath'):
		robustness.suite_root.relative_to(robustness.f3_root)
	assert all(
		run.input_dir.name == VOXEL_DECODER_SPEC
		for run in original.runs
		if run.version == 'V1'
	)


def test_active_f3_voxel_robustness_stage_configs_resolve() -> None:
	build = f3_lithology_voxel_split_dataset_config_from_mapping(
		load_config(F3_VOXEL_ROBUSTNESS_CONFIGS[0])
	)
	v0 = f3_lithology_voxel_v0_split_suite_config_from_mapping(
		load_config(F3_VOXEL_ROBUSTNESS_CONFIGS[1])
	)
	v1_raw = load_config(F3_VOXEL_ROBUSTNESS_CONFIGS[2])
	v1 = f3_lithology_voxel_decoder_split_suite_config_from_mapping(v1_raw)

	assert build.split_inventory_manifest.name == 'split_inventory_manifest.json'
	assert v0.voxel_dataset_manifest == Path(v1_raw['suite']['voxel_dataset_manifest'])
	assert v0.output_root == Path(v1_raw['suite']['output_root']) == build.output_root
	assert v0.split_dataset_manifest.name == 'split_dataset_manifest.json'
	assert v0.probe_run_manifest.name == 'split_probe_run_manifest.json'
	assert all(model.checkpoint is not None for model in v0.models)
	assert all('checkpoint' not in model for model in v1_raw['models'].values())
	assert tuple(model.embeddings_dir for model in v0.models) == tuple(
		Path(model['embeddings_dir']) for model in v1_raw['models'].values()
	)
	assert v1.decoder.spec == VOXEL_DECODER_SPEC
	assert v1.decoder.upsample_mode == VOXEL_DECODER_UPSAMPLE_MODE
	assert v1.decoder.normalization == VOXEL_DECODER_NORMALIZATION
	assert (
		v1_raw['decoder']
		== load_config(F3_VOXEL_V1_ROOT / '02_train_mae_full.yaml')['decoder']
	)
	assert OLD_VOXEL_DECODER_SPEC not in F3_VOXEL_ROBUSTNESS_CONFIGS[2].read_text(
		encoding='utf-8'
	)
	for config in (build, v0, v1):
		config.output_root.relative_to(config.artifact_root)
		with pytest.raises(ValueError, match='is not in the subpath'):
			config.output_root.relative_to(config.f3_root)


def test_f3_voxel_robustness_output_root_must_stay_under_artifacts() -> None:
	raw = load_config(F3_VOXEL_ROBUSTNESS_CONFIGS[0])
	raw['suite']['output_root'] = '/outside-artifacts'

	with pytest.raises(ValueError, match=r'suite\.output_root must be under root'):
		f3_lithology_voxel_split_dataset_config_from_mapping(raw)


def test_f3_voxel_robustness_output_root_must_stay_outside_raw_f3() -> None:
	raw = load_config(F3_VOXEL_ROBUSTNESS_CONFIGS[0])
	f3_root = Path(raw['paths']['f3_root'])
	raw['paths']['artifact_root'] = str(f3_root)
	raw['suite']['output_root'] = str(f3_root / 'generated')

	with pytest.raises(ValueError, match=r'suite\.output_root must be outside f3_root'):
		f3_lithology_voxel_split_dataset_config_from_mapping(raw)


@pytest.mark.parametrize('config_path', F3_BASELINE_COMPARISON_CONFIGS)
def test_active_f3_baseline_comparison_configs_resolve(
	config_path: Path,
) -> None:
	raw = load_config(config_path)

	f3_lithology_comparison_report_config_from_mapping(raw)
	f3_lithology_comparison_publish_config_from_mapping(raw.get('publish'))


def test_active_nopims_overlap_x16_paths_match_artifact_paths_contract() -> None:
	model_tag = 'amp_mae_m075_mse_g0_patchnorm_clip8_vis01_v1'
	paths = ArtifactPaths(DEFAULT_ARTIFACT_ROOT)
	key = ExperimentKey(
		dataset='nopims',
		version='pretrain_v1',
		model_tag=model_tag,
		subset='ten_surveys',
		embed_spec='overlap_x16',
		cluster_spec='k6_8_whiten',
		viz_spec='voxel_cmp_xy750_xz150',
	)

	embedding = load_config(
		NOPIMS_ROOT / '20_embedding' / model_tag / '01_ten_surveys_overlap_x16.yaml',
	)
	clustering = load_config(
		NOPIMS_ROOT
		/ '30_clustering'
		/ model_tag
		/ '01_ten_surveys_overlap_x16_k6_8_whiten.yaml',
	)
	visualization = load_config(
		NOPIMS_ROOT
		/ '40_visualization'
		/ model_tag
		/ '01_ten_surveys_overlap_x16_whiten.yaml',
	)

	assert Path(embedding['embeddings']['output_dir']) == paths.embeddings(key)
	assert Path(clustering['embeddings']['input_dir']) == paths.embeddings(key)
	assert Path(clustering['clustering']['output_dir']) == paths.clustering(key)
	assert Path(visualization['clustering']['input_dir']) == paths.clustering(key)
	assert Path(
		visualization['visualization']['output_dir']
	) == paths.cluster_visualization(key)


def _config_with_available_output(
	config_path: Path, tmp_path: Path
) -> dict[str, object]:
	config = load_config(config_path)
	outputs = config.get('outputs')
	assert isinstance(outputs, dict)
	assert outputs.get('overwrite') is False
	output_value = outputs.get('output_dir')
	assert isinstance(output_value, str)
	output_dir = Path(output_value)
	if output_dir.exists():
		replacement = output_dir.parent / f'.{output_dir.name}.{tmp_path.name}'
		assert not replacement.exists()
		outputs['output_dir'] = str(replacement)
	return config


def _projection_config_with_synthetic_source(
	config_path: Path, tmp_path: Path
) -> dict[str, object]:
	raw = load_config(config_path)
	artifact_root = tmp_path / 'artifacts'
	source = artifact_root / 'token_predictions'
	source.mkdir(parents=True)
	class_info = artifact_root / 'class_info.json'
	class_info.write_text(
		json.dumps({'1': {'name': 'one', 'color': [1, 2, 3]}}),
		encoding='utf-8',
	)
	predictions = source / 'f3_token_predictions.npy'
	probabilities = source / 'f3_token_probabilities.npy'
	valid_tokens = source / 'f3_valid_token_grid.npy'
	metadata_json = source / 'prediction_metadata.json'
	np.save(predictions, np.ones((1, 1, 1), dtype=np.int16))
	np.save(probabilities, np.ones((1, 1, 1, 1), dtype=np.float32))
	np.save(valid_tokens, np.ones((1, 1, 1), dtype=np.bool_))
	metadata_json.write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_token_predictions',
				'dataset': dict(raw['dataset']),
				'model': {'tag': raw['model']['tag'], 'freeze_encoder': True},
				'embeddings': {'spec': 'overlap_x16'},
				'probe': {'spec': 'linear_balanced_v1'},
				'classes': [{'class_id': 1, 'class_name': 'one'}],
				'class_probability_order': [1],
				'invalid_prediction_class_id': -1,
				'invalid_probability_value': 'nan',
				'embedding': {
					'patch_size_xyz': [8, 8, 8],
					'token_grid_shape_xyz': [1, 1, 1],
				},
				'geometry': {'shape_xyz': [8, 8, 8]},
				'outputs': {
					'token_predictions': str(predictions),
					'probability_volume': str(probabilities),
					'valid_token_grid': str(valid_tokens),
					'metadata_json': str(metadata_json),
				},
				'summary': {
					'token_grid_shape_xyz': [1, 1, 1],
					'probability_grid_shape': [1, 1, 1, 1],
					'valid_token_count': 1,
					'invalid_token_count': 0,
				},
			}
		),
		encoding='utf-8',
	)
	raw['paths'] = {
		'artifact_root': str(artifact_root),
		'f3_root': str(tmp_path / 'f3'),
	}
	raw['labels']['class_info'] = str(class_info)
	raw['token_predictions'] = {
		'input_dir': str(source),
		'predictions': str(predictions),
		'probabilities': str(probabilities),
		'valid_tokens': str(valid_tokens),
		'metadata_json': str(metadata_json),
	}
	raw['voxel_projection']['output_dir'] = str(artifact_root / 'voxel_predictions')
	return raw


def _config_with_existing_strat_hmm_pretext_inputs(
	config_path: Path,
	tmp_path: Path,
) -> dict[str, object]:
	config = load_config(config_path)
	artifact_root = tmp_path / 'artifacts'
	pseudo_target_dir = tmp_path / 'pseudo_targets'
	pseudo_target_dir.mkdir(exist_ok=True)
	checkpoint = tmp_path / 'mae_best.pt'
	checkpoint.touch()

	config['paths']['artifact_root'] = str(artifact_root)
	config['paths']['output_root'] = str(
		artifact_root / 'pretraining' / 'f3' / config_path.stem,
	)
	if 'manifest' in config['pseudo_targets']:
		fixture_root = tmp_path / config_path.stem
		fixture_root.mkdir(exist_ok=True)
		embeddings, heads = _artifacts(
			fixture_root,
			source_root=tmp_path / 'shared_multi_head_sources',
		)
		migration, control = _write_positive_preflight(fixture_root)
		manifest = fixture_root / 'multi_head_target_manifest.json'
		build_multi_head_target_manifest(
			manifest_path=manifest,
			source_embedding_dir=embeddings,
			head_roots={6: heads[6], 8: heads[8], 10: heads[10]},
			replay_k6_root=_replay_k6_root(fixture_root, heads[6]),
			migration_decision=migration,
			control_summary=control,
		)
		config['pseudo_targets']['manifest'] = str(manifest)
		if 'target_head_hashes' in config['identity']['scientific_identity']:
			config['identity']['scientific_identity']['target_head_hashes'] = (
				_multi_head_target_hashes(
					json.loads(manifest.read_text(encoding='utf-8'))
				)
			)
		if config['pseudo_targets'].get('target_representation') == (
			'ordered_path_state_posterior_v1'
		):
			config['identity']['scientific_identity']['posterior_manifest_sha256'] = (
				file_sha256(manifest)
			)
		elif config['pseudo_targets'].get('target_representation') not in {
			'lateral_mean_field_hard_labels_v1',
			'xy_neighbor_consensus_hard_labels_v1',
			'xy_neighbor_unanimous_hard_labels_v1',
		}:
			config['identity']['scientific_identity']['target_manifest_sha256'] = (
				file_sha256(manifest)
			)
	else:
		config['pseudo_targets']['input_dir'] = str(pseudo_target_dir)
	config['teacher']['checkpoint'] = str(checkpoint)
	config['student']['init_checkpoint'] = str(checkpoint)
	return config


def _active_posterior_manifest() -> dict[str, object]:
	return {
		'head_ks': [6, 8, 10],
		'posterior_semantics': 'ordered_path_cost_gibbs_state_marginal_v1',
		'cost_temperature': 1.0,
		'heads': {
			str(k): {
				'surveys': {
					'f3_facies_benchmark': {
						'posterior': {'sha256': f'{k:064x}'},
						'valid_tokens': {'sha256': f'{k + 10:064x}'},
						'metadata': {'sha256': f'{k + 20:064x}'},
					}
				}
			}
			for k in (6, 8, 10)
		},
	}


def _active_lateral_manifest() -> dict[str, object]:
	"""Return the reference-only lateral identity used by active config tests."""
	return {
		'head_ks': [6, 8, 10],
		'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
		'source_hard_manifest': {'sha256': 'a' * 64},
		'source_posterior_manifest': {'sha256': 'b' * 64},
		'smoothing': {
			'neighborhood': 'xy_4_connected_v1',
			'affinity': 'source_embedding_cosine_rbf_v1',
			'affinity_scale_policy': (
				'global_valid_xy_edge_distance_median_floor_1e-6_v1'
			),
			'emission_scale_policy': 'per_head_valid_second_gap_median_floor_1e-6_v1',
			'pairwise_strength_ratio': 0.10,
			'iterations': 1,
			'projection': 'original_ordered_viterbi_v1',
		},
		'heads': {
			str(k): {
				'surveys': {
					'f3_facies_benchmark': {
						'labels': {'sha256': f'{k:064x}'},
						'confidence': {'sha256': f'{k + 10:064x}'},
						'valid_tokens': {'sha256': f'{k + 20:064x}'},
						'metadata': {'sha256': f'{k + 30:064x}'},
					}
				},
				'diagnostics': {
					'resolved_scales': {
						'affinity': {'resolved_scale': 1.0},
						'emission_gap': {'resolved_scale': 1.0},
					}
				},
			}
			for k in (6, 8, 10)
		},
	}


def _active_xy_neighbor_consensus_manifest() -> dict[str, object]:
	"""Return the reference-only XY identity used by active config tests."""
	return {
		'head_ks': [6, 8, 10],
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
		'source_hard_manifest': {'sha256': 'a' * 64},
		'smoothing': {
			'neighborhood': 'same_z_xy_four_neighbors',
			'neighbor_order': ['x_minus', 'x_plus', 'y_minus', 'y_plus'],
			'four_valid_neighbors_minimum_agreement': 3,
			'three_valid_neighbors_minimum_agreement': 3,
			'fewer_than_three_valid_neighbors': 'unchanged',
			'tied_or_nonunique_consensus': 'unchanged',
			'center_matching_consensus': 'unchanged',
			'temporal_guard': 'internal_valid_token_source_label_bounds',
			'application': 'single_pass_synchronous_source_labels',
		},
		'heads': {
			str(k): {
				'surveys': {
					'f3_facies_benchmark': {
						'labels': {'sha256': f'{k:064x}'},
						'confidence': {'sha256': f'{k + 10:064x}'},
						'valid_tokens': {'sha256': f'{k + 20:064x}'},
						'metadata': {'sha256': f'{k + 30:064x}'},
					}
				}
			}
			for k in (6, 8, 10)
		},
	}


def _active_xy_neighbor_unanimous_manifest() -> dict[str, object]:
	"""Return the reference-only unanimous identity used by active configs."""
	manifest = _active_xy_neighbor_consensus_manifest()
	manifest['target_representation'] = 'xy_neighbor_unanimous_hard_labels_v1'
	manifest['target_semantics'] = 'xy_neighbor_unanimous_outlier_correction_v1'
	manifest['smoothing']['four_valid_neighbors_minimum_agreement'] = 4
	return manifest


def _config_with_existing_strat_hmm_refresh_inputs(
	config_path: Path,
	tmp_path: Path,
) -> dict[str, object]:
	config = load_config(config_path)
	artifact_root = tmp_path / 'artifacts'
	checkpoint = tmp_path / 'latest.pt'
	torch.save(
		{
			'stratigraphy_config': {
				'head': {'num_prototypes': config['hmm']['k']},
			},
		},
		checkpoint,
	)

	config['paths']['artifact_root'] = str(artifact_root)
	config['checkpoint']['path'] = str(checkpoint)
	config['outputs']['pseudo_target_root'] = str(
		artifact_root / 'pseudo_targets' / 'f3' / config_path.stem,
	)
	return config


def _write_class_info(tmp_path: Path) -> Path:
	path = tmp_path / 'class_info.json'
	path.write_text(
		json.dumps(
			{
				'classes': [
					{'class_id': 0, 'class_name': 'class 0', 'rgb': [230, 159, 0]},
					{'class_id': 1, 'class_name': 'class 1', 'rgb': [86, 180, 233]},
				],
			},
		)
		+ '\n',
		encoding='utf-8',
	)
	return path


def _write_label_budget_manifest(tmp_path: Path) -> Path:
	manifest = tmp_path / 'label_budget_suite_manifest.json'
	token_dataset_root = tmp_path / 'tokens'
	rows = [
		_label_budget_manifest_row(
			role='baseline',
			model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
			token_dataset_root=token_dataset_root / 'baseline',
		),
		_label_budget_manifest_row(
			role='candidate',
			model_tag='strat_hmm_pretext_m1_k6_topblock1_distill',
			token_dataset_root=token_dataset_root / 'candidate',
		),
	]
	_write_json(
		manifest,
		{
			'artifact_type': 'f3_lithology_label_budget_suite_manifest',
			'rows': rows,
		},
	)
	return manifest


def _label_budget_manifest_row(
	*,
	role: str,
	model_tag: str,
	token_dataset_root: Path,
) -> dict[str, object]:
	return {
		'model_role': role,
		'model_tag': model_tag,
		'budget_id': 'cap_25',
		'per_class_cap': 25,
		'subsample_seed': 0,
		'token_dataset_root': str(token_dataset_root),
		'train_tokens': str(token_dataset_root / 'train_tokens.npz'),
		'validation_tokens': str(token_dataset_root / 'validation_tokens.npz'),
		'metadata_json': str(token_dataset_root / 'token_dataset_metadata.json'),
		'selected_train_token_count': 50,
		'validation_token_count': 20,
		'paired_identity_hash': 'paired-hash',
	}


def _write_split_dataset_manifest(tmp_path: Path) -> Path:
	manifest = tmp_path / 'split_dataset_manifest.json'
	token_dataset_root = tmp_path / 'split_tokens'
	rows = [
		_split_dataset_manifest_row(
			role='baseline',
			model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
			token_dataset_root=token_dataset_root / 'baseline',
		),
		_split_dataset_manifest_row(
			role='candidate',
			model_tag='strat_hmm_pretext_m1_k6_topblock1_distill',
			token_dataset_root=token_dataset_root / 'candidate',
		),
	]
	_write_json(
		manifest,
		{
			'artifact_type': 'f3_lithology_split_sweep_token_dataset_manifest',
			'rows': rows,
		},
	)
	return manifest


def _split_dataset_manifest_row(
	*,
	role: str,
	model_tag: str,
	token_dataset_root: Path,
) -> dict[str, object]:
	return {
		'split_id': 'split_000',
		'model_role': role,
		'model_tag': model_tag,
		'token_dataset_root': str(token_dataset_root),
		'train_tokens': str(token_dataset_root / 'train_tokens.npz'),
		'validation_tokens': str(token_dataset_root / 'validation_tokens.npz'),
		'metadata_json': str(token_dataset_root / 'token_dataset_metadata.json'),
		'train_token_count': 50,
		'validation_token_count': 20,
		'paired_identity_hash': 'paired-hash',
	}


def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
