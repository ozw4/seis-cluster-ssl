# ruff: noqa: E501
from __future__ import annotations

from copy import deepcopy

import pytest

from seis_ssl_cluster.config.performance_migration_validation import (
	performance_migration_validation_config_from_mapping,
)


def _config() -> dict[str, object]:
	root = '/workspace/artifacts/seis_ssl_cluster'
	return {
		'paths': {
			'artifact_root': root,
			'migration_root': f'{root}/migration_validation/f3/main_test',
			'publish_root': '/workspace/reports/f3/performance_migration_validation',
		},
		'migration': {
			'current_git_sha': 'a' * 40,
			'historical_baseline_sha': 'b' * 40,
		},
		'inputs': {
			'checkpoints': {key: f'{root}/{key}.pt' for key in ('mae', 'm1', 'm2a')},
			'historical_embeddings': {
				key: f'{root}/embeddings/{key}' for key in ('mae', 'm1', 'm2a')
			},
			'm1_probe': {
				key: f'{root}/probe/{key}'
				for key in (
					'token_dataset_metadata',
					'train_tokens',
					'validation_tokens',
					'scaler',
					'probe',
					'prediction_metadata',
					'predictions',
					'probabilities',
					'valid_grid',
					'metrics',
				)
			},
			'hmm': {
				key: f'{root}/hmm/{key}'
				for key in (
					'historical_root',
					'labels',
					'label_metadata',
					'centers',
					'clustering_metadata',
					'hmm_model',
					'preprocessor',
					'residualizer',
				)
			},
			'pseudo_targets': {
				key: f'{root}/pseudo/{key}'
				for key in ('historical_root', 'labels', 'confidence', 'valid_tokens', 'metadata')
			},
			'f3': {
				key: f'{root}/f3/{key}'
				for key in (
					'amplitude_manifest',
					'normalization_stats',
					'class_info',
					'volume_metadata',
					'amplitude_volume',
					'label_volume',
				)
			},
		},
		'compatibility': {
			'm1_historical_finite_check_mode': 'off',
			'historical_finite_check_evidence_commit': 'c' * 40,
			'historical_finite_check_evidence_path': 'src/seis_ssl_cluster/embedding/extractor.py',
			'historical_pseudo_target_schema_version': 1,
		},
		'benchmark': {'seed': 248, 'warm_up': 3, 'repeat': 20, 'threads': 1},
	}


def test_performance_migration_config_resolves_strictly() -> None:
	config = performance_migration_validation_config_from_mapping(_config())

	assert config.current_git_sha == 'a' * 40
	assert config.compatibility['m1_historical_finite_check_mode'] == 'off'


def test_performance_migration_config_rejects_unknown_key() -> None:
	raw = _config()
	raw['surprise'] = True

	with pytest.raises(ValueError, match='not allowed'):
		performance_migration_validation_config_from_mapping(raw)


def test_performance_migration_config_rejects_silent_finite_default() -> None:
	raw = deepcopy(_config())
	compatibility = raw['compatibility']
	assert isinstance(compatibility, dict)
	compatibility['m1_historical_finite_check_mode'] = 'strict'

	with pytest.raises(ValueError, match='must be "off"'):
		performance_migration_validation_config_from_mapping(raw)
