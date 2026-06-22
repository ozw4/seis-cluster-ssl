"""Amplitude-only MVP configuration constants."""

from __future__ import annotations

from typing import Final

DEFAULT_NOPIMS_ROOT: Final = '/home/dcuser/data/NOPIMS'
DEFAULT_ARTIFACT_ROOT: Final = '/workspace/artifacts/seis_ssl_cluster'

EXPECTED_GRID_ORDER: Final = ['x', 'y', 'z']
EXPECTED_VOLUME_FORMAT: Final = 'npy_memmap'
EXPECTED_INPUT_CHANNELS: Final = 1
EXPECTED_TARGET_CHANNELS: Final = 1
EXPECTED_USE_CONTEXT: Final = False

KNOWN_STAGES: Final = {
	'build_nopims_manifests',
	'prepare_nopims_normalization_stats',
	'filter_manifest_by_normalization_qc',
	'train_amp_mae',
	'extract_embeddings',
	'cluster_embeddings',
	'visualize_clusters',
}

LEGACY_ATTRIBUTE_KEY_PATHS: Final = {
	'attributes.names',
	'attributes.registry',
	'attribute_ids',
	'attribute_dropout_prob',
	'group_dropout_prob',
	'dropped_attribute_weight',
	'attribute_registry',
	'fixed_attribute_registry',
}

LEGACY_ATTRIBUTE_KEY_NAMES: Final = {
	'attribute_ids',
	'attribute_dropout_prob',
	'group_dropout_prob',
	'dropped_attribute_weight',
	'attribute_registry',
	'fixed_attribute_registry',
}
