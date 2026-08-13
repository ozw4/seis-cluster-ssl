"""Stratigraphic pretraining artifact contracts."""

from __future__ import annotations

import importlib

from seis_ssl_cluster.stratigraphy.boundary_weights import (
	boundary_distance_tokens,
	boundary_weight_tokens,
)
from seis_ssl_cluster.stratigraphy.export import (
	ExportedPseudoTargetResult,
	export_hmm_cluster_labels_as_pseudo_targets,
	prepare_hmm_cluster_label_pseudo_target_exports,
)
from seis_ssl_cluster.stratigraphy.lateral_smoothing import (
	LATERAL_SMOOTHING_SEMANTICS,
	LateralCostUpdateResult,
	LateralMessageResult,
	LateralSmoothingResult,
	apply_lateral_message_to_emission_costs,
	cosine_rbf_affinities,
	enumerate_xy_four_neighbors,
	median_scale_with_floor,
	normalized_lateral_message,
	redecode_ordered_lateral_trace,
	smooth_and_redecode_ordered_trace,
)
from seis_ssl_cluster.stratigraphy.losses import (
	feature_distillation_loss,
	ordered_soft_coordinate,
	soft_categorical_cross_entropy,
	structured_hmm_prototype_loss,
	usage_entropy_floor_loss,
)
from seis_ssl_cluster.stratigraphy.multi_head import (
	build_multi_head_target_manifest,
	compare_k6_replay,
	load_multi_head_target_manifest,
	multi_head_cross_head_diagnostics,
	validate_multi_head_target_manifest,
	validate_multi_head_target_publication_preflight,
)
from seis_ssl_cluster.stratigraphy.multi_head_export import (
	CANONICAL_KS,
	MultiHeadPseudoTargetExportConfig,
	MultiHeadPseudoTargetExportPlan,
	export_multi_head_pseudo_targets,
	plan_multi_head_pseudo_target_exports,
	resolve_multi_head_pseudo_target_export_config,
)
from seis_ssl_cluster.stratigraphy.prototypes import (
	MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1,
	MultiResolutionOrderedPrototypeHeads,
	MultiResolutionOrderedPrototypeOutput,
	OrderedPrototypeHead,
	OrderedPrototypeOutput,
	expected_normalized_order_coordinate,
)
from seis_ssl_cluster.stratigraphy.shuffle_targets import (
	GLOBAL_VALID_TOKENS,
	ShuffledPseudoTargetResult,
	plan_shuffled_hmm_pseudo_targets,
	shuffle_pseudo_target_arrays,
	shuffle_strat_hmm_pseudo_targets,
)
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetArrays,
	StratPseudoTargetInput,
	StratPseudoTargetPaths,
	discover_pseudo_target_inputs,
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
	pseudo_target_paths,
	validate_pseudo_target_arrays,
	write_pseudo_target,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus import (
	XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_POLICY,
	XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_SEMANTICS,
	XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_POLICY,
	XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_SEMANTICS,
	XYNeighborConsensusDiagnostics,
	XYNeighborConsensusResult,
	smooth_xy_neighbor_consensus_hard_labels,
	smooth_xy_neighbor_unanimous_hard_labels,
)

_HMM_DECODE_EXPORTS = {
	'LogitHMMPseudoTarget',
	'decode_ordered_logits_survey',
	'emission_costs_from_logits',
}

_STATE_POSTERIOR_EXPORTS = {
	'POSTERIOR_SEMANTICS',
	'MultiHeadStatePosteriorExportConfig',
	'MultiHeadStatePosteriorExportPlan',
	'export_multi_head_state_posteriors',
	'load_multi_head_state_posterior_manifest',
	'plan_multi_head_state_posterior_exports',
	'resolve_multi_head_state_posterior_export_config',
	'validate_multi_head_state_posterior_manifest',
}

_LATERAL_TARGET_EXPORTS = {
	'MultiHeadLateralTargetExportConfig',
	'MultiHeadLateralTargetExportPlan',
	'export_multi_head_lateral_targets',
	'load_multi_head_lateral_target_manifest',
	'plan_multi_head_lateral_target_exports',
	'resolve_multi_head_lateral_target_export_config',
	'validate_multi_head_lateral_target_manifest',
}

_XY_NEIGHBOR_CONSENSUS_TARGET_EXPORTS = {
	'MultiHeadXYNeighborConsensusTargetExportConfig',
	'MultiHeadXYNeighborConsensusTargetExportPlan',
	'export_multi_head_xy_neighbor_consensus_targets',
	'load_multi_head_xy_neighbor_consensus_target_manifest',
	'plan_multi_head_xy_neighbor_consensus_target_exports',
	'resolve_multi_head_xy_neighbor_consensus_target_export_config',
	'validate_multi_head_xy_neighbor_consensus_target_manifest',
}

_XY_NEIGHBOR_UNANIMOUS_TARGET_EXPORTS = {
	'MultiHeadXYNeighborUnanimousTargetExportConfig',
	'MultiHeadXYNeighborUnanimousTargetExportPlan',
	'export_multi_head_xy_neighbor_unanimous_targets',
	'load_multi_head_xy_neighbor_unanimous_target_manifest',
	'plan_multi_head_xy_neighbor_unanimous_target_exports',
	'resolve_multi_head_xy_neighbor_unanimous_target_export_config',
	'validate_multi_head_xy_neighbor_unanimous_target_manifest',
}

_PERIODIC_REFRESH_EXPORTS = {
	'PERIODIC_REFRESH_ARTIFACT_TYPE': 'ARTIFACT_TYPE',
	'INITIAL_GENERATION_ID': 'INITIAL_GENERATION_ID',
	'HardTargetPolicy': 'HardTargetPolicy',
	'HashedArtifactReference': 'HashedArtifactReference',
	'InitialHMMArtifact': 'InitialHMMArtifact',
	'InitialPeriodicRefreshConfig': 'InitialPeriodicRefreshConfig',
	'PeriodicRefreshConfig': 'PeriodicRefreshConfig',
	'PeriodicRefreshGenerationResult': 'PeriodicRefreshGenerationResult',
	'PreviousCenterArtifact': 'PreviousCenterArtifact',
	'build_initial_periodic_refresh_generation': (
		'build_initial_periodic_refresh_generation'
	),
	'build_periodic_refresh_generation': 'build_periodic_refresh_generation',
	'load_periodic_refresh_generation': 'load_periodic_refresh_generation',
	'produce_initial_periodic_refresh_generation': (
		'produce_initial_periodic_refresh_generation'
	),
	'produce_periodic_refresh_generation': 'produce_periodic_refresh_generation',
	'quarantine_periodic_refresh_generation': 'quarantine_periodic_refresh_generation',
	'validate_periodic_refresh_generation': 'validate_periodic_refresh_generation',
}

__all__ = [
	'CANONICAL_KS',
	'GLOBAL_VALID_TOKENS',
	'INITIAL_GENERATION_ID',
	'LATERAL_SMOOTHING_SEMANTICS',
	'MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1',
	'PERIODIC_REFRESH_ARTIFACT_TYPE',
	'POSTERIOR_SEMANTICS',
	'XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_POLICY',
	'XY_NEIGHBOR_CONSENSUS_HARD_LABEL_SMOOTHING_SEMANTICS',
	'XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_POLICY',
	'XY_NEIGHBOR_UNANIMOUS_OUTLIER_CORRECTION_SEMANTICS',
	'ExportedPseudoTargetResult',
	'HardTargetPolicy',
	'HashedArtifactReference',
	'InitialHMMArtifact',
	'InitialPeriodicRefreshConfig',
	'LateralCostUpdateResult',
	'LateralMessageResult',
	'LateralSmoothingResult',
	'LogitHMMPseudoTarget',
	'MultiHeadLateralTargetExportConfig',
	'MultiHeadLateralTargetExportPlan',
	'MultiHeadPseudoTargetExportConfig',
	'MultiHeadPseudoTargetExportPlan',
	'MultiHeadStatePosteriorExportConfig',
	'MultiHeadStatePosteriorExportPlan',
	'MultiHeadXYNeighborConsensusTargetExportConfig',
	'MultiHeadXYNeighborConsensusTargetExportPlan',
	'MultiHeadXYNeighborUnanimousTargetExportConfig',
	'MultiHeadXYNeighborUnanimousTargetExportPlan',
	'MultiResolutionOrderedPrototypeHeads',
	'MultiResolutionOrderedPrototypeOutput',
	'OrderedPrototypeHead',
	'OrderedPrototypeOutput',
	'PeriodicRefreshConfig',
	'PeriodicRefreshGenerationResult',
	'PreviousCenterArtifact',
	'ShuffledPseudoTargetResult',
	'StratPseudoTargetArrays',
	'StratPseudoTargetInput',
	'StratPseudoTargetPaths',
	'XYNeighborConsensusDiagnostics',
	'XYNeighborConsensusResult',
	'apply_lateral_message_to_emission_costs',
	'boundary_distance_tokens',
	'boundary_weight_tokens',
	'build_initial_periodic_refresh_generation',
	'build_multi_head_target_manifest',
	'build_periodic_refresh_generation',
	'compare_k6_replay',
	'cosine_rbf_affinities',
	'decode_ordered_logits_survey',
	'discover_pseudo_target_inputs',
	'emission_costs_from_logits',
	'enumerate_xy_four_neighbors',
	'expected_normalized_order_coordinate',
	'export_hmm_cluster_labels_as_pseudo_targets',
	'export_multi_head_lateral_targets',
	'export_multi_head_pseudo_targets',
	'export_multi_head_state_posteriors',
	'export_multi_head_xy_neighbor_consensus_targets',
	'export_multi_head_xy_neighbor_unanimous_targets',
	'feature_distillation_loss',
	'load_multi_head_lateral_target_manifest',
	'load_multi_head_state_posterior_manifest',
	'load_multi_head_target_manifest',
	'load_multi_head_xy_neighbor_consensus_target_manifest',
	'load_multi_head_xy_neighbor_unanimous_target_manifest',
	'load_periodic_refresh_generation',
	'load_pseudo_target_arrays',
	'load_pseudo_target_metadata',
	'median_scale_with_floor',
	'multi_head_cross_head_diagnostics',
	'normalized_lateral_message',
	'ordered_soft_coordinate',
	'plan_multi_head_lateral_target_exports',
	'plan_multi_head_pseudo_target_exports',
	'plan_multi_head_state_posterior_exports',
	'plan_multi_head_xy_neighbor_consensus_target_exports',
	'plan_multi_head_xy_neighbor_unanimous_target_exports',
	'plan_shuffled_hmm_pseudo_targets',
	'prepare_hmm_cluster_label_pseudo_target_exports',
	'produce_initial_periodic_refresh_generation',
	'produce_periodic_refresh_generation',
	'pseudo_target_paths',
	'quarantine_periodic_refresh_generation',
	'redecode_ordered_lateral_trace',
	'resolve_multi_head_lateral_target_export_config',
	'resolve_multi_head_pseudo_target_export_config',
	'resolve_multi_head_state_posterior_export_config',
	'resolve_multi_head_xy_neighbor_consensus_target_export_config',
	'resolve_multi_head_xy_neighbor_unanimous_target_export_config',
	'shuffle_pseudo_target_arrays',
	'shuffle_strat_hmm_pseudo_targets',
	'smooth_and_redecode_ordered_trace',
	'smooth_xy_neighbor_consensus_hard_labels',
	'smooth_xy_neighbor_unanimous_hard_labels',
	'soft_categorical_cross_entropy',
	'structured_hmm_prototype_loss',
	'usage_entropy_floor_loss',
	'validate_multi_head_lateral_target_manifest',
	'validate_multi_head_state_posterior_manifest',
	'validate_multi_head_target_manifest',
	'validate_multi_head_target_publication_preflight',
	'validate_multi_head_xy_neighbor_consensus_target_manifest',
	'validate_multi_head_xy_neighbor_unanimous_target_manifest',
	'validate_periodic_refresh_generation',
	'validate_pseudo_target_arrays',
	'write_pseudo_target',
]


def __getattr__(name: str) -> object:
	"""Lazily import HMM decoding, which needs optional clustering dependencies."""
	if name in _HMM_DECODE_EXPORTS:
		hmm_decode = importlib.import_module('seis_ssl_cluster.stratigraphy.hmm_decode')
		return getattr(hmm_decode, name)
	if name in _STATE_POSTERIOR_EXPORTS:
		state_posterior = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.state_posterior'
		)
		return getattr(state_posterior, name)
	if name in _LATERAL_TARGET_EXPORTS:
		lateral_targets = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.lateral_targets'
		)
		return getattr(lateral_targets, name)
	if name in _XY_NEIGHBOR_CONSENSUS_TARGET_EXPORTS:
		consensus_targets = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets'
		)
		return getattr(consensus_targets, name)
	if name in _XY_NEIGHBOR_UNANIMOUS_TARGET_EXPORTS:
		unanimous_targets = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets'
		)
		return getattr(unanimous_targets, name)
	if name in _PERIODIC_REFRESH_EXPORTS:
		periodic_refresh = importlib.import_module(
			'seis_ssl_cluster.stratigraphy.periodic_refresh'
		)
		return getattr(periodic_refresh, _PERIODIC_REFRESH_EXPORTS[name])
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
