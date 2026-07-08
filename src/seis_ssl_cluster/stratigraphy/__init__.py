"""Stratigraphic pretraining artifact contracts."""

from __future__ import annotations

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
	'StratPseudoTargetArrays',
	'StratPseudoTargetInput',
	'StratPseudoTargetPaths',
	'discover_pseudo_target_inputs',
	'load_pseudo_target_arrays',
	'load_pseudo_target_metadata',
	'pseudo_target_paths',
	'validate_pseudo_target_arrays',
	'write_pseudo_target',
]
