from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import yaml

import seis_ssl_cluster.embedding.extractor as extractor_module
from seis_ssl_cluster.config import resolve_vicreg_training_config
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.models.amplitude_encoder_factory import (
	build_model_from_checkpoint_payload,
)
from seis_ssl_cluster.training.vicreg import run_vicreg_pretraining
from seis_ssl_cluster.training.vicreg_checkpoint import load_vicreg_checkpoint

if TYPE_CHECKING:
	from pathlib import Path

VICREG_METRICS = {
	'training_loss',
	'invariance_loss',
	'variance_loss',
	'covariance_loss',
	'projection_std_mean',
	'projection_std_min',
	'covariance_offdiag_rms',
	'weighted_invariance',
	'weighted_variance',
	'weighted_covariance',
	'gradient_norm',
	'learning_rate',
	'step_time_seconds',
	'peak_cuda_memory_mib',
}


def test_cpu_training_checkpoint_factory_and_embedding_contract(
	tmp_path: Path,
) -> None:
	config = resolve_vicreg_training_config(_tiny_config(tmp_path))

	checkpoint_path = run_vicreg_pretraining(config)
	payload = load_vicreg_checkpoint(checkpoint_path, map_location='cpu')

	assert checkpoint_path.name == 'latest.pt'
	assert (checkpoint_path.parent / 'best.pt').is_file()
	assert payload['pretraining_method'] == 'local_vicreg_3d'
	assert payload['checkpoint_kind'] == 'vicreg_pretraining'
	assert payload['trained_parameter_prefixes'] == [
		'patch_projection.',
		'encoder.',
	]
	assert payload['epoch'] == payload['global_step'] == 1
	assert payload['training_state'] == {
		'schema_version': 1,
		'stage': 'vicreg_training',
		'resume_boundary': 'epoch',
		'dataset_epoch': 0,
		'completed_epoch': True,
	}
	assert set(payload['metrics']) >= VICREG_METRICS
	assert all(np.isfinite(payload['metrics'][key]) for key in VICREG_METRICS)
	assert payload['projector_state_dict']
	assert all(
		not key.startswith(('backbone.', 'projector.'))
		for key in payload['model_state_dict']
	)

	encoder = build_model_from_checkpoint_payload(payload)
	for key, expected in payload['model_state_dict'].items():
		assert torch.equal(encoder.state_dict()[key], expected)
	strat_payload = dict(payload)
	strat_payload['stratigraphy_config'] = {'stage': 'train_strat_hmm_pretext'}
	strat_payload['training_state'] = {
		'schema_version': 1,
		'stage': 'train_strat_hmm_pretext',
		'checkpoint_kind': 'epoch',
		'batch_index': None,
	}
	strat_payload.pop('projector_state_dict')
	strat_payload.pop('pretraining_method')
	strat_payload.pop('checkpoint_kind')
	strat_payload.pop('trained_parameter_prefixes')
	strat_encoder = build_model_from_checkpoint_payload(strat_payload)
	for key, expected in payload['model_state_dict'].items():
		assert torch.equal(strat_encoder.state_dict()[key], expected)
	objective = extractor_module._pretraining_objective(config)  # noqa: SLF001
	assert objective == {
		'method': 'local_vicreg_3d',
		'local_pairs_per_crop': 4,
		'projector_dim': 4,
		'invariance_weight': 25.0,
		'variance_weight': 25.0,
		'covariance_weight': 1.0,
		'variance_target_std': 1.0,
		'variance_eps': 1.0e-4,
	}
	history = json.loads(
		(checkpoint_path.parent / 'history.json').read_text(encoding='utf-8')
	)
	assert len(history) == 1
	assert set(history[0]) >= VICREG_METRICS


def test_completed_epoch_resume_matches_uninterrupted_training(
	tmp_path: Path,
) -> None:
	uninterrupted_path = run_vicreg_pretraining(
		resolve_vicreg_training_config(
			_tiny_config(
				tmp_path,
				output_name='uninterrupted',
				epochs=2,
				max_steps=2,
			)
		)
	)
	partial_path = run_vicreg_pretraining(
		resolve_vicreg_training_config(
			_tiny_config(tmp_path, output_name='resumed')
		)
	)
	resumed_path = run_vicreg_pretraining(
		resolve_vicreg_training_config(
			_tiny_config(
				tmp_path,
				output_name='resumed',
				epochs=2,
				max_steps=2,
			)
		),
		resume=partial_path,
	)

	uninterrupted = load_vicreg_checkpoint(uninterrupted_path, map_location='cpu')
	resumed = load_vicreg_checkpoint(resumed_path, map_location='cpu')
	assert resumed['epoch'] == resumed['global_step'] == 2
	assert resumed['resume_count'] == 1
	_assert_tensor_state_equal(
		uninterrupted['model_state_dict'], resumed['model_state_dict']
	)
	_assert_tensor_state_equal(
		uninterrupted['projector_state_dict'], resumed['projector_state_dict']
	)


def test_resume_rejects_objective_drift(tmp_path: Path) -> None:
	checkpoint_path = run_vicreg_pretraining(
		resolve_vicreg_training_config(_tiny_config(tmp_path, output_name='source'))
	)
	raw = _tiny_config(
		tmp_path,
		output_name='drifted',
		epochs=2,
		max_steps=2,
	)
	vicreg = raw['vicreg']
	assert isinstance(vicreg, dict)
	vicreg['covariance_weight'] = 2.0

	with pytest.raises(ValueError, match='vicreg'):
		run_vicreg_pretraining(
			resolve_vicreg_training_config(raw),
			resume=checkpoint_path,
		)


def test_resume_rejects_tampered_checkpoint_config_stage(tmp_path: Path) -> None:
	checkpoint_path = run_vicreg_pretraining(
		resolve_vicreg_training_config(
			_tiny_config(tmp_path, output_name='stage-source')
		)
	)
	payload = load_vicreg_checkpoint(checkpoint_path, map_location='cpu')
	saved_config = payload['config']
	assert isinstance(saved_config, dict)
	saved_config['stage'] = 'barlow_twins_training'
	tampered_path = tmp_path / 'tampered-resume-stage.pt'
	torch.save(payload, tampered_path)

	with pytest.raises(ValueError, match=r'saved checkpoint config\.stage'):
		run_vicreg_pretraining(
			resolve_vicreg_training_config(
				_tiny_config(
					tmp_path,
					output_name='stage-source',
					epochs=2,
					max_steps=2,
				)
			),
			resume=tampered_path,
		)


def test_fresh_continuation_uses_weights_without_optimizer_state(
	tmp_path: Path,
) -> None:
	source_path = run_vicreg_pretraining(
		resolve_vicreg_training_config(
			_tiny_config(
				tmp_path,
				output_name='source-depth2',
				epochs=2,
				max_steps=2,
				encoder_depth=2,
			)
		)
	)
	source = load_vicreg_checkpoint(source_path, map_location='cpu')
	raw = _tiny_config(
		tmp_path,
		output_name='continued',
		encoder_depth=2,
	)
	raw['continuation'] = {
		'init_checkpoint': str(source_path),
		'unfreeze_top_blocks': 1,
	}
	target_path = run_vicreg_pretraining(resolve_vicreg_training_config(raw))
	target = load_vicreg_checkpoint(target_path, map_location='cpu')

	_assert_prefix_equal(source, target, 'patch_projection.')
	_assert_prefix_equal(source, target, 'encoder.layers.0.')
	assert _prefix_changed(source, target, 'encoder.layers.1.')
	assert _optimizer_steps(source) == {2}
	assert _optimizer_steps(target) == {1}
	assert target['continuation_lineage']['init_checkpoint'] == str(source_path)
	assert target['continuation_lineage']['resume_count'] == 0


def test_fresh_continuation_rejects_tampered_source_config_stage(
	tmp_path: Path,
) -> None:
	source_path = run_vicreg_pretraining(
		resolve_vicreg_training_config(
			_tiny_config(tmp_path, output_name='continuation-stage-source')
		)
	)
	payload = load_vicreg_checkpoint(source_path, map_location='cpu')
	saved_config = payload['config']
	assert isinstance(saved_config, dict)
	saved_config['stage'] = 'barlow_twins_training'
	tampered_path = tmp_path / 'tampered-continuation-stage.pt'
	torch.save(payload, tampered_path)
	raw = _tiny_config(tmp_path, output_name='continuation-stage-target')
	raw['continuation'] = {
		'init_checkpoint': str(tampered_path),
		'unfreeze_top_blocks': 1,
	}

	with pytest.raises(ValueError, match=r'continuation checkpoint config\.stage'):
		run_vicreg_pretraining(resolve_vicreg_training_config(raw))


def test_cli_dry_run_prints_vicreg_identity_without_artifacts(
	tmp_path: Path,
) -> None:
	raw = _tiny_config(tmp_path, output_name='must-not-exist')
	config_path = tmp_path / 'vicreg.yaml'
	config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
	output_root = tmp_path / 'artifacts/must-not-exist'

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/train_amp_vicreg.py',
			'--config',
			str(config_path),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
		env=os.environ,
	)

	assert 'stage: vicreg_training' in result.stdout
	assert 'vicreg.method: local_vicreg_3d' in result.stdout
	assert 'vicreg.local_pairs_per_crop: 4' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout
	assert not output_root.exists()


def test_cli_dry_run_prints_all_explicit_augmentation_fields(
	tmp_path: Path,
) -> None:
	raw = _tiny_config(tmp_path, output_name='d4-must-not-exist')
	raw['augmentations'] = {
		'policy': 'xy_d4_trace_drop_v1',
		'reflection_probability': 0.4,
		'trace_drop_probability': 0.02,
	}
	config_path = tmp_path / 'vicreg-d4.yaml'
	config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/train_amp_vicreg.py',
			'--config',
			str(config_path),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
		env=os.environ,
	)

	assert 'augmentations.policy: xy_d4_trace_drop_v1' in result.stdout
	assert 'augmentations.reflection_probability: 0.4' in result.stdout
	assert 'augmentations.trace_drop_probability: 0.02' in result.stdout
	assert not (tmp_path / 'artifacts/d4-must-not-exist').exists()


def test_embedding_objective_preserves_explicit_augmentations(
	tmp_path: Path,
) -> None:
	raw = _tiny_config(tmp_path, output_name='metadata-only')
	raw['augmentations'] = {
		'policy': 'horizontal_flip_gaussian_noise_v1',
		'horizontal_flip_probability': 0.5,
		'gaussian_noise_std': 0.05,
	}
	config = resolve_vicreg_training_config(raw)

	objective = extractor_module._pretraining_objective(config)  # noqa: SLF001

	assert objective['augmentations'] == raw['augmentations']


def _tiny_config(
	tmp_path: Path,
	*,
	output_name: str = 'run',
	epochs: int = 1,
	max_steps: int = 1,
	encoder_depth: int = 1,
) -> dict[str, object]:
	manifest_path = _write_synthetic_manifest(tmp_path / 'survey')
	path_list = tmp_path / 'train_npy_paths.txt'
	path_list.write_text(
		f'{tmp_path / "survey" / "amplitude.npy"}\n', encoding='utf-8'
	)
	return {
		'paths': {
			'artifact_root': str(tmp_path / 'artifacts'),
			'output_root': str(tmp_path / 'artifacts' / output_name),
		},
		'manifests': {
			'train': str(manifest_path),
			'train_path_list': str(path_list),
		},
		'data': {'local_crop_size': [4, 4, 4]},
		'zero_mask': {'enabled': False},
		'model': {
			'patch_size': [2, 2, 2],
			'encoder_dim': 4,
			'encoder_depth': encoder_depth,
			'encoder_heads': 1,
			'decoder_dim': 4,
			'decoder_depth': 1,
			'decoder_heads': 1,
		},
		'vicreg': {
			'method': 'local_vicreg_3d',
			'local_pairs_per_crop': 4,
			'projector_dim': 4,
		},
		'train': {
			'batch_size': 2,
			'samples_per_epoch': 2,
			'epochs': epochs,
			'num_workers': 0,
			'shuffle': False,
			'lr': 1.0e-3,
			'weight_decay': 0.0,
			'amp': False,
			'device': 'cpu',
			'seed': 7,
			'grad_clip_norm': 1.0,
			'max_steps': max_steps,
		},
	}


def _write_synthetic_manifest(root: Path) -> Path:
	root.mkdir(parents=True, exist_ok=True)
	volume_path = root / 'amplitude.npy'
	volume = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
	np.save(volume_path, volume)
	stats_path = root / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='tiny',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1000.0,
			clip_high=1000.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	manifest = SurveyManifest(
		survey_id='tiny',
		root=root,
		amplitude=AmplitudeVolumeRecord(
			survey_id='tiny',
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	manifest_path = root / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	return manifest_path


def _assert_tensor_state_equal(left: object, right: object) -> None:
	assert isinstance(left, dict)
	assert isinstance(right, dict)
	assert set(left) == set(right)
	for key in left:
		assert isinstance(left[key], torch.Tensor)
		assert torch.equal(left[key], right[key])


def _assert_prefix_equal(
	left: dict[str, object],
	right: dict[str, object],
	prefix: str,
) -> None:
	left_state = left['model_state_dict']
	right_state = right['model_state_dict']
	assert isinstance(left_state, dict)
	assert isinstance(right_state, dict)
	keys = [key for key in left_state if key.startswith(prefix)]
	assert keys
	assert all(torch.equal(left_state[key], right_state[key]) for key in keys)


def _prefix_changed(
	left: dict[str, object],
	right: dict[str, object],
	prefix: str,
) -> bool:
	left_state = left['model_state_dict']
	right_state = right['model_state_dict']
	assert isinstance(left_state, dict)
	assert isinstance(right_state, dict)
	keys = [key for key in left_state if key.startswith(prefix)]
	assert keys
	return any(not torch.equal(left_state[key], right_state[key]) for key in keys)


def _optimizer_steps(payload: dict[str, object]) -> set[int]:
	optimizer = payload['optimizer_state_dict']
	assert isinstance(optimizer, dict)
	states = optimizer['state']
	assert isinstance(states, dict)
	return {
		int(parameter_state['step'])
		for parameter_state in states.values()
		if isinstance(parameter_state, dict)
	}
