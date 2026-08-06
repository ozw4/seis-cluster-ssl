# ruff: noqa: E501, TC003
"""Strict configuration for performance migration validation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)


@dataclass(frozen=True)
class PerformanceMigrationValidationConfig:
	"""Validated immutable inputs and isolated outputs for one migration audit."""

	artifact_root: Path
	migration_root: Path
	publish_root: Path
	current_git_sha: str
	historical_baseline_sha: str
	checkpoints: Mapping[str, Path]
	historical_embeddings: Mapping[str, Path]
	m1_probe: Mapping[str, Path]
	hmm: Mapping[str, Path]
	pseudo_targets: Mapping[str, Path]
	f3: Mapping[str, Path]
	compatibility: Mapping[str, object]
	benchmark: Mapping[str, int]

	def __post_init__(self) -> None:
		"""Keep all mutable scientific outputs inside the dedicated artifact root."""


_TOP_LEVEL_KEYS = frozenset(
	{'paths', 'migration', 'inputs', 'compatibility', 'benchmark'}
)
_PATH_KEYS = frozenset({'artifact_root', 'migration_root', 'publish_root'})
_MIGRATION_KEYS = frozenset({'current_git_sha', 'historical_baseline_sha'})
_INPUT_KEYS = frozenset(
	{'checkpoints', 'historical_embeddings', 'm1_probe', 'hmm', 'pseudo_targets', 'f3'},
)
_CHECKPOINT_KEYS = frozenset({'mae', 'm1', 'm2a'})
_EMBEDDING_KEYS = frozenset({'mae', 'm1', 'm2a'})
_M1_PROBE_KEYS = frozenset(
	{
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
	},
)
_HMM_KEYS = frozenset(
	{
		'historical_root',
		'labels',
		'label_metadata',
		'centers',
		'clustering_metadata',
		'hmm_model',
		'preprocessor',
		'residualizer',
	},
)
_PSEUDO_TARGET_KEYS = frozenset({'historical_root', 'labels', 'confidence', 'valid_tokens', 'metadata'})
_F3_KEYS = frozenset(
	{
		'amplitude_manifest',
		'normalization_stats',
		'class_info',
		'volume_metadata',
		'amplitude_volume',
		'label_volume',
	}
)
_COMPATIBILITY_KEYS = frozenset(
	{
		'm1_historical_finite_check_mode',
		'historical_finite_check_evidence_commit',
		'historical_finite_check_evidence_path',
		'historical_pseudo_target_schema_version',
	}
)
_BENCHMARK_KEYS = frozenset({'seed', 'warm_up', 'repeat', 'threads'})


def performance_migration_validation_config_from_mapping(
	config: Mapping[str, object],
) -> PerformanceMigrationValidationConfig:
	"""Resolve the migration config with fail-fast unknown-key rejection."""
	_validate_allowed_keys(config, _TOP_LEVEL_KEYS, prefix='config')
	paths = _required_mapping(config, 'paths')
	migration = _required_mapping(config, 'migration')
	inputs = _required_mapping(config, 'inputs')
	compatibility = _required_mapping(config, 'compatibility')
	benchmark = _required_mapping(config, 'benchmark')
	_validate_allowed_keys(paths, _PATH_KEYS, prefix='paths')
	_validate_allowed_keys(migration, _MIGRATION_KEYS, prefix='migration')
	_validate_allowed_keys(inputs, _INPUT_KEYS, prefix='inputs')
	_validate_allowed_keys(
		compatibility,
		_COMPATIBILITY_KEYS,
		prefix='compatibility',
	)
	_validate_allowed_keys(benchmark, _BENCHMARK_KEYS, prefix='benchmark')

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	migration_root = _required_absolute_path(paths, 'migration_root', prefix='paths')
	publish_root = _required_absolute_path(paths, 'publish_root', prefix='paths')
	current_git_sha = _sha(
		_required_str(migration, 'current_git_sha', prefix='migration'),
		'migration.current_git_sha',
	)
	historical_baseline_sha = _sha(
		_required_str(migration, 'historical_baseline_sha', prefix='migration'),
		'migration.historical_baseline_sha',
	)
	checkpoints = _path_mapping(
		inputs,
		'checkpoints',
		_CHECKPOINT_KEYS,
	)
	historical_embeddings = _path_mapping(
		inputs,
		'historical_embeddings',
		_EMBEDDING_KEYS,
	)
	m1_probe = _path_mapping(
		inputs,
		'm1_probe',
		_M1_PROBE_KEYS,
	)
	hmm = _path_mapping(
		inputs,
		'hmm',
		_HMM_KEYS,
	)
	pseudo_targets = _path_mapping(
		inputs,
		'pseudo_targets',
		_PSEUDO_TARGET_KEYS,
	)
	f3 = _path_mapping(inputs, 'f3', _F3_KEYS)

	finite_mode = compatibility.get('m1_historical_finite_check_mode')
	if finite_mode != 'off':
		raise ValueError(
			'compatibility.m1_historical_finite_check_mode must be "off" '
			'for the evidenced legacy M1 extraction contract',
		)
	_sha(
		_required_str(
			compatibility,
			'historical_finite_check_evidence_commit',
			prefix='compatibility',
		),
		'compatibility.historical_finite_check_evidence_commit',
	)
	evidence_path = _required_str(
		compatibility,
		'historical_finite_check_evidence_path',
		prefix='compatibility',
	)
	if not evidence_path:
		raise ValueError('compatibility.historical_finite_check_evidence_path is required')
	schema_version = compatibility.get('historical_pseudo_target_schema_version')
	if schema_version != 1:
		raise ValueError(
			'compatibility.historical_pseudo_target_schema_version must be 1',
		)
	benchmark_values = _benchmark_mapping(benchmark)
	return PerformanceMigrationValidationConfig(
		artifact_root=artifact_root,
		migration_root=migration_root,
		publish_root=publish_root,
		current_git_sha=current_git_sha,
		historical_baseline_sha=historical_baseline_sha,
		checkpoints=checkpoints,
		historical_embeddings=historical_embeddings,
		m1_probe=m1_probe,
		hmm=hmm,
		pseudo_targets=pseudo_targets,
		f3=f3,
		compatibility=dict(compatibility),
		benchmark=benchmark_values,
	)


def _path_mapping(
	parent: Mapping[str, object],
	key: str,
	allowed: frozenset[str],
) -> dict[str, Path]:
	value = _required_mapping(parent, key)
	_validate_allowed_keys(value, allowed, prefix=f'inputs.{key}')
	return {
		item: _required_absolute_path(value, item, prefix=f'inputs.{key}')
		for item in sorted(allowed)
	}


def _benchmark_mapping(value: Mapping[str, object]) -> dict[str, int]:
	result: dict[str, int] = {}
	for key in sorted(_BENCHMARK_KEYS):
		item = value.get(key)
		if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
			raise ValueError(f'benchmark.{key} must be a positive integer')
		result[key] = int(item)
	return result


def _sha(value: str, label: str) -> str:
	if len(value) != 40 or any(character not in '0123456789abcdef' for character in value):
		raise ValueError(f'{label} must be a lowercase 40-character git SHA')
	return value




__all__ = [
	'PerformanceMigrationValidationConfig',
	'performance_migration_validation_config_from_mapping',
]
