"""Shared checkpoint-source validation for Parihaka Channel benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)

CHANNEL_PRETRAINED_MODEL_TAG = (
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
)
CHANNEL_RANDOM_ENCODER_SEED = 42
CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX = (
	'pretraining',
	'parihaka',
	'facies_benchmark_v1',
	CHANNEL_PRETRAINED_MODEL_TAG,
	'full_100ep',
	'latest.pt',
)


def inspect_channel_model_sources(
	pretrained_metadata: Mapping[str, object],
	random_metadata: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
	"""Validate checkpoint SHA and scientific roles for a paired comparison."""
	pretrained_path, pretrained_sha = validated_channel_checkpoint(
		pretrained_metadata, 'pretrained'
	)
	random_path, random_sha = validated_channel_checkpoint(
		random_metadata, 'random'
	)
	if pretrained_sha == random_sha:
		raise ValueError('pretrained/random checkpoint_sha256 must differ')
	if tuple(pretrained_path.parts[-len(CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX) :]) != (
		CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX
	):
		raise ValueError(
			'pretrained embedding checkpoint must be the expected Parihaka '
			'full_100ep/latest.pt'
		)
	random_payload = load_checkpoint_metadata_without_weights(random_path)
	metadata = _checkpoint_mapping(random_payload, 'metadata', random_path)
	training_state = _checkpoint_mapping(
		random_payload, 'training_state', random_path
	)
	if metadata.get('random_encoder_baseline') is not True:
		raise ValueError(
			'random checkpoint metadata.random_encoder_baseline must equal True'
		)
	if metadata.get('pretrained_weights_loaded') is not False:
		raise ValueError(
			'random checkpoint metadata.pretrained_weights_loaded must equal False'
		)
	expected_metadata = {
		'seed': CHANNEL_RANDOM_ENCODER_SEED,
		'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
	}
	for key, expected in expected_metadata.items():
		if metadata.get(key) != expected:
			raise ValueError(
				f'random checkpoint metadata.{key} must equal {expected!r}'
			)
	if training_state.get('checkpoint_kind') != 'random_init':
		raise ValueError(
			'random checkpoint training_state.checkpoint_kind must equal '
			"'random_init'"
		)
	reference_value = metadata.get('reference_checkpoint')
	if not isinstance(reference_value, str) or not reference_value:
		raise TypeError(
			'random checkpoint metadata.reference_checkpoint must be non-empty'
		)
	reference_path = Path(reference_value)
	if reference_path.resolve(strict=False) != pretrained_path.resolve(strict=False):
		raise ValueError(
			'random checkpoint metadata.reference_checkpoint must equal the '
			'pretrained embedding checkpoint'
		)
	return (
		{
			'role': 'pretrained',
			'checkpoint_path': str(pretrained_path),
			'checkpoint_sha256': pretrained_sha,
			'model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		},
		{
			'role': 'random',
			'checkpoint_path': str(random_path),
			'checkpoint_sha256': random_sha,
			'random_encoder_baseline': True,
			'pretrained_weights_loaded': False,
			'seed': CHANNEL_RANDOM_ENCODER_SEED,
			'checkpoint_kind': 'random_init',
			'reference_checkpoint': reference_value,
			'reference_checkpoint_sha256': pretrained_sha,
			'reference_model_tag': CHANNEL_PRETRAINED_MODEL_TAG,
		},
	)


def validated_channel_checkpoint(
	metadata: Mapping[str, object], role: str
) -> tuple[Path, str]:
	"""Require one metadata checkpoint path and digest to match its file."""
	path_value = metadata.get('checkpoint_path')
	if not isinstance(path_value, str) or not path_value:
		raise TypeError(f'{role} embedding checkpoint_path must be non-empty')
	path = Path(path_value)
	if not path.is_file():
		raise FileNotFoundError(f'{role} embedding checkpoint does not exist: {path}')
	sha_value = metadata.get('checkpoint_sha256')
	if (
		not isinstance(sha_value, str)
		or len(sha_value) != 64
		or any(character not in '0123456789abcdef' for character in sha_value)
	):
		raise TypeError(
			f'{role} embedding checkpoint_sha256 must be a lowercase SHA-256 digest'
		)
	if file_sha256(path) != sha_value:
		raise ValueError(
			f'{role} embedding checkpoint_sha256 does not match checkpoint file'
		)
	return path, sha_value


def _checkpoint_mapping(
	payload: Mapping[str, object], key: str, path: Path
) -> Mapping[str, object]:
	value = payload.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{path} checkpoint {key} must be a mapping')
	return value


__all__ = [
	'CHANNEL_PRETRAINED_CHECKPOINT_SUFFIX',
	'CHANNEL_PRETRAINED_MODEL_TAG',
	'CHANNEL_RANDOM_ENCODER_SEED',
	'inspect_channel_model_sources',
	'validated_channel_checkpoint',
]
