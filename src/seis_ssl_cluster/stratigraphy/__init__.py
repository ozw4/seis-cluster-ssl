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

__all__ = [
	'GLOBAL_VALID_TOKENS',
	'MULTI_RESOLUTION_ORDERED_PROTOTYPES_V1',
	'ExportedPseudoTargetResult',
	'LogitHMMPseudoTarget',
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
	'feature_distillation_loss',
	'load_multi_head_target_manifest',
	'load_pseudo_target_arrays',
	'load_pseudo_target_metadata',
	'multi_head_cross_head_diagnostics',
	'ordered_soft_coordinate',
	'plan_shuffled_hmm_pseudo_targets',
	'prepare_hmm_cluster_label_pseudo_target_exports',
	'pseudo_target_paths',
	'shuffle_pseudo_target_arrays',
	'shuffle_strat_hmm_pseudo_targets',
	'structured_hmm_prototype_loss',
	'usage_entropy_floor_loss',
	'validate_multi_head_target_manifest',
	'validate_pseudo_target_arrays',
	'write_pseudo_target',
]


def __getattr__(name: str) -> object:
	"""Lazily import HMM decoding, which needs optional clustering dependencies."""
	if name in _HMM_DECODE_EXPORTS:
		hmm_decode = importlib.import_module('seis_ssl_cluster.stratigraphy.hmm_decode')
		return getattr(hmm_decode, name)
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
