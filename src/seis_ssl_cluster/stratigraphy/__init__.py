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
from seis_ssl_cluster.stratigraphy.losses import (
	feature_distillation_loss,
	ordered_soft_coordinate,
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

__all__ = [
	'CANONICAL_KS',
	'GLOBAL_VALID_TOKENS',
	'MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1',
	'POSTERIOR_SEMANTICS',
	'ExportedPseudoTargetResult',
	'LogitHMMPseudoTarget',
	'MultiHeadPseudoTargetExportConfig',
	'MultiHeadPseudoTargetExportPlan',
	'MultiHeadStatePosteriorExportConfig',
	'MultiHeadStatePosteriorExportPlan',
	'MultiResolutionOrderedPrototypeHeads',
	'MultiResolutionOrderedPrototypeOutput',
	'OrderedPrototypeHead',
	'OrderedPrototypeOutput',
	'ShuffledPseudoTargetResult',
	'StratPseudoTargetArrays',
	'StratPseudoTargetInput',
	'StratPseudoTargetPaths',
	'boundary_distance_tokens',
	'boundary_weight_tokens',
	'build_multi_head_target_manifest',
	'compare_k6_replay',
	'decode_ordered_logits_survey',
	'discover_pseudo_target_inputs',
	'emission_costs_from_logits',
	'expected_normalized_order_coordinate',
	'export_hmm_cluster_labels_as_pseudo_targets',
	'export_multi_head_pseudo_targets',
	'export_multi_head_state_posteriors',
	'feature_distillation_loss',
	'load_multi_head_state_posterior_manifest',
	'load_multi_head_target_manifest',
	'load_pseudo_target_arrays',
	'load_pseudo_target_metadata',
	'multi_head_cross_head_diagnostics',
	'ordered_soft_coordinate',
	'plan_multi_head_pseudo_target_exports',
	'plan_multi_head_state_posterior_exports',
	'plan_shuffled_hmm_pseudo_targets',
	'prepare_hmm_cluster_label_pseudo_target_exports',
	'pseudo_target_paths',
	'resolve_multi_head_pseudo_target_export_config',
	'resolve_multi_head_state_posterior_export_config',
	'shuffle_pseudo_target_arrays',
	'shuffle_strat_hmm_pseudo_targets',
	'structured_hmm_prototype_loss',
	'usage_entropy_floor_loss',
	'validate_multi_head_state_posterior_manifest',
	'validate_multi_head_target_manifest',
	'validate_multi_head_target_publication_preflight',
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
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
