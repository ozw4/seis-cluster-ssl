"""Stratigraphic pretraining artifact contracts."""

from __future__ import annotations

from seis_ssl_cluster.stratigraphy.hmm_decode import (
	LogitHMMPseudoTarget,
	decode_ordered_logits_survey,
	emission_costs_from_logits,
)
from seis_ssl_cluster.stratigraphy.losses import (
	feature_distillation_loss,
	ordered_soft_coordinate,
	structured_hmm_prototype_loss,
	usage_entropy_floor_loss,
)
from seis_ssl_cluster.stratigraphy.prototypes import (
	OrderedPrototypeHead,
	OrderedPrototypeOutput,
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

__all__ = [
	'LogitHMMPseudoTarget',
	'OrderedPrototypeHead',
	'OrderedPrototypeOutput',
	'StratPseudoTargetArrays',
	'StratPseudoTargetInput',
	'StratPseudoTargetPaths',
	'decode_ordered_logits_survey',
	'discover_pseudo_target_inputs',
	'emission_costs_from_logits',
	'feature_distillation_loss',
	'load_pseudo_target_arrays',
	'load_pseudo_target_metadata',
	'ordered_soft_coordinate',
	'pseudo_target_paths',
	'structured_hmm_prototype_loss',
	'usage_entropy_floor_loss',
	'validate_pseudo_target_arrays',
	'write_pseudo_target',
]
