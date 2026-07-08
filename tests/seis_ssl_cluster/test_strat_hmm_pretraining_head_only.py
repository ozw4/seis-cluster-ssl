from __future__ import annotations

import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import yaml

from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	NopimsStratPseudoTargetDataset,
	SurveyManifest,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	read_manifest_json,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.stratigraphy import (
	discover_pseudo_target_inputs,
	write_pseudo_target,
)
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.dataloaders import build_strat_pseudo_target_dataloader
from seis_ssl_cluster.training.strat_hmm_pretraining import (
	build_strat_hmm_head_only_components,
	configure_student_trainability,
	run_strat_hmm_pretext_training,
	train_strat_hmm_head_only_one_epoch,
)

if TYPE_CHECKING:
	from collections.abc import Mapping


def test_head_only_training_runs_cpu_writes_checkpoints_and_payloads(
	tmp_path: Path,
) -> None:
	config = _resolved_config(tmp_path, max_steps=2)

	checkpoint_path = run_strat_hmm_pretext_training(config)

	assert checkpoint_path == Path(config['paths']['output_root']) / 'latest.pt'
	assert checkpoint_path.is_file()
	assert (Path(config['paths']['output_root']) / 'best.pt').is_file()
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['global_step'] == 2
	assert payload['training_state']['stage'] == 'train_strat_hmm_pretext'
	assert payload['config']['stage'] == 'train_amp_mae'
	assert payload['stratigraphy_config']['stage'] == 'train_strat_hmm_pretext'
	assert math.isfinite(payload['metrics']['loss'])
	assert math.isfinite(payload['metrics']['loss_prototype'])
	assert math.isfinite(payload['metrics']['loss_usage'])
	assert payload['metrics']['trainable_parameter_count'] == pytest.approx(0.0)
	assert payload['trainability_summary']['trainable_names'] == []
	assert set(payload['metrics']) >= {
		'valid_supervised_token_fraction',
		'target_usage_entropy',
		'prototype_usage_entropy',
	}
	model_keys = set(payload['model_state_dict'])
	head_keys = set(payload['stratigraphy_state_dict'])
	assert 'patch_projection.weight' in model_keys
	assert 'prediction_head.weight' in model_keys
	assert not (model_keys & {'prototypes', 'projection.weight', 'projection.bias'})
	assert {'prototypes', 'projection.weight', 'projection.bias'} <= head_keys


def test_head_only_components_freeze_student_and_train_head(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path)

	components = build_strat_hmm_head_only_components(
		config,
		device=torch.device('cpu'),
	)

	assert all(
		not parameter.requires_grad for parameter in components.student.parameters()
	)
	assert all(parameter.requires_grad for parameter in components.head.parameters())
	head_parameter_ids = {
		id(parameter) for parameter in components.head.parameters()
	}
	optimizer_parameter_ids = {
		id(parameter)
		for group in components.optimizer.param_groups
		for parameter in group['params']
	}
	assert optimizer_parameter_ids == head_parameter_ids


def test_components_reject_unfreeze_without_distillation(tmp_path: Path) -> None:
	config = _raw_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.0,
	)

	with pytest.raises(ValueError, match=r'loss\.distillation_weight'):
		build_strat_hmm_head_only_components(config, device=torch.device('cpu'))


def test_configure_student_trainability_unfreezes_only_top_blocks() -> None:
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=3,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)

	summary = configure_student_trainability(model, unfreeze_top_blocks=0)

	assert summary.trainable_parameter_count == 0
	assert summary.trainable_names == ()
	assert all(not parameter.requires_grad for parameter in model.parameters())

	summary = configure_student_trainability(model, unfreeze_top_blocks=2)

	assert summary.trainable_parameter_count > 0
	assert summary.trainable_names
	assert all(
		name.startswith(('encoder.layers.1.', 'encoder.layers.2.'))
		for name in summary.trainable_names
	)
	assert all(
		not parameter.requires_grad
		for name, parameter in model.named_parameters()
		if not name.startswith(('encoder.layers.1.', 'encoder.layers.2.'))
	)

	with pytest.raises(ValueError, match='unfreeze_top_blocks'):
		configure_student_trainability(model, unfreeze_top_blocks=4)


def test_unfreeze_top_block_optimizer_lrs_and_gradients(tmp_path: Path) -> None:
	config = _resolved_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.25,
	)
	components = build_strat_hmm_head_only_components(
		config,
		device=torch.device('cpu'),
	)
	dataloader = _single_batch_dataloader(config)

	state = train_strat_hmm_head_only_one_epoch(
		student=components.student,
		teacher=components.teacher,
		head=components.head,
		dataloader=dataloader,
		optimizer=components.optimizer,
		device=torch.device('cpu'),
		epoch=1,
		loss_config=config['loss'],
		pseudo_target_config=config['pseudo_targets'],
		max_steps=1,
	)

	assert state.global_step == 1
	assert state.metrics['loss_distillation'] >= -1.0e-6
	assert [group['lr'] for group in components.optimizer.param_groups] == [
		pytest.approx(config['train']['lr']),
		pytest.approx(config['train']['encoder_lr']),
	]
	bottom_grads = [
		parameter.grad
		for parameter in components.student.encoder.layers[0].parameters()
	]
	top_grads = [
		parameter.grad
		for parameter in components.student.encoder.layers[-1].parameters()
	]
	assert all(grad is None for grad in bottom_grads)
	assert any(
		grad is not None and bool(grad.abs().sum().gt(0).item())
		for grad in top_grads
	)
	assert components.teacher is not None
	assert all(parameter.grad is None for parameter in components.teacher.parameters())


def test_unfreeze_distillation_smoke_writes_checkpoint(tmp_path: Path) -> None:
	config = _resolved_config(
		tmp_path,
		encoder_depth=2,
		unfreeze_top_blocks=1,
		distillation_weight=0.2,
		max_steps=1,
	)

	checkpoint_path = run_strat_hmm_pretext_training(config)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['global_step'] == 1
	assert math.isfinite(payload['metrics']['loss_distillation'])
	assert payload['metrics']['trainable_parameter_count'] > 0.0
	assert payload['trainability_summary']['trainable_names']


def test_invalid_pseudo_target_tokens_are_ignored(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path, invalid_first_token=True, max_steps=1)

	checkpoint_path = run_strat_hmm_pretext_training(config)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['metrics']['valid_supervised_token_fraction'] == pytest.approx(
		7.0 / 8.0,
	)
	assert math.isfinite(payload['metrics']['loss'])


def test_checkpoint_config_uses_strat_preprocessing_contract(
	tmp_path: Path,
) -> None:
	config = _resolved_config(tmp_path, max_steps=1)
	config['data']['normalized_clip_abs'] = 0.5
	config['data']['amplitude_agc'] = {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 3,
		'eps': 1.0e-6,
		'clip_abs': 2.0,
	}

	checkpoint_path = run_strat_hmm_pretext_training(config)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['config']['data']['normalized_clip_abs'] == 0.5
	assert payload['config']['data']['amplitude_agc'] == {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 3,
		'eps': 1.0e-6,
		'clip_abs': 2.0,
	}


def test_resume_advances_and_rejects_incompatible_head_config(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path, max_steps=1)

	first_checkpoint = run_strat_hmm_pretext_training(config)
	config['train']['max_steps'] = 2
	resumed_checkpoint = run_strat_hmm_pretext_training(
		config,
		resume=first_checkpoint,
	)

	payload = load_checkpoint(resumed_checkpoint, map_location='cpu')
	assert payload['global_step'] == 2

	incompatible = deepcopy(config)
	incompatible['head'] = dict(config['head'])
	incompatible['head']['temperature'] = 0.25
	with pytest.raises(ValueError, match=r'head\.temperature'):
		run_strat_hmm_pretext_training(incompatible, resume=resumed_checkpoint)


def test_resume_preserves_existing_best_score(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path, max_steps=1)

	first_checkpoint = run_strat_hmm_pretext_training(config)
	best_path = Path(config['paths']['output_root']) / 'best.pt'
	best_payload = load_checkpoint(best_path, map_location='cpu')
	best_payload['metrics']['loss'] = -1.0
	torch.save(best_payload, best_path)
	config['train']['max_steps'] = 2

	resumed_checkpoint = run_strat_hmm_pretext_training(
		config,
		resume=first_checkpoint,
	)

	latest_payload = load_checkpoint(resumed_checkpoint, map_location='cpu')
	resumed_best_payload = load_checkpoint(best_path, map_location='cpu')
	assert latest_payload['global_step'] == 2
	assert resumed_best_payload['metrics']['loss'] == -1.0


def test_cli_non_dry_run_executes_one_step(tmp_path: Path) -> None:
	raw_config = _raw_config(tmp_path, max_steps=None)
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(yaml.safe_dump(raw_config), encoding='utf-8')
	repo_root = Path(__file__).resolve().parents[2]
	script = repo_root / 'proc' / 'seis_ssl_cluster' / 'train_strat_hmm_pretext.py'

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			str(script),
			'--config',
			str(config_path),
			'--device',
			'cpu',
			'--max-steps',
			'1',
		],
		cwd=repo_root,
		check=True,
		capture_output=True,
		text=True,
	)

	checkpoint_path = Path(raw_config['paths']['output_root']) / 'latest.pt'
	assert f'checkpoint: {checkpoint_path}' in result.stdout
	assert checkpoint_path.is_file()
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	assert payload['global_step'] == 1


def _resolved_config(  # noqa: PLR0913
	tmp_path: Path,
	*,
	invalid_first_token: bool = False,
	max_steps: int | None = None,
	encoder_depth: int = 1,
	unfreeze_top_blocks: int = 0,
	distillation_weight: float = 0.0,
) -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(
		_raw_config(
			tmp_path,
			invalid_first_token=invalid_first_token,
			max_steps=max_steps,
			encoder_depth=encoder_depth,
			unfreeze_top_blocks=unfreeze_top_blocks,
			distillation_weight=distillation_weight,
		),
	)


def _raw_config(  # noqa: PLR0913
	tmp_path: Path,
	*,
	invalid_first_token: bool = False,
	max_steps: int | None = None,
	encoder_depth: int = 1,
	unfreeze_top_blocks: int = 0,
	distillation_weight: float = 0.0,
) -> dict[str, object]:
	paths = _write_fixture_files(
		tmp_path,
		invalid_first_token=invalid_first_token,
		encoder_depth=encoder_depth,
	)
	artifact_root = tmp_path / 'artifacts'
	output_root = artifact_root / 'pretraining' / 'strat_hmm_head_only'
	config: dict[str, object] = {
		'paths': {
			'artifact_root': str(artifact_root),
			'output_root': str(output_root),
		},
		'manifests': {
			'train': str(paths['manifest']),
			'train_path_list': str(paths['path_list']),
		},
		'data': {
			'local_crop_size': [4, 4, 4],
			'min_valid_fraction': 0.0,
			'max_resample_attempts': 2,
		},
		'model': {
			'patch_size': [2, 2, 2],
			'encoder_dim': 12,
			'encoder_depth': encoder_depth,
			'encoder_heads': 3,
			'decoder_dim': 12,
			'decoder_depth': 1,
			'decoder_heads': 3,
		},
		'pseudo_targets': {
			'input_dir': str(paths['pseudo_root']),
			'k': 3,
			'min_confidence': 0.0,
		},
		'teacher': {'checkpoint': str(paths['teacher_checkpoint'])},
		'student': {'unfreeze_top_blocks': unfreeze_top_blocks},
		'head': {
			'num_prototypes': 3,
			'projection_dim': 6,
			'temperature': 0.5,
		},
		'loss': {
			'usage_weight': 0.01,
			'distillation_weight': distillation_weight,
			'entropy_floor': None,
		},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 2,
			'epochs': 1,
			'num_workers': 0,
			'shuffle': False,
			'lr': 1.0e-2,
			'encoder_lr': 1.0e-3,
			'weight_decay': 0.0,
			'amp': False,
			'device': 'cpu',
			'seed': 11,
			'grad_clip_norm': 1.0,
			'allow_overwrite_output': False,
		},
		'zero_mask': {
			'enabled': False,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 0,
			'xy_trace_influence_radius': 0,
		},
	}
	if max_steps is not None:
		config['train']['max_steps'] = max_steps
	return config


def _single_batch_dataloader(
	config: Mapping[str, object],
) -> torch.utils.data.DataLoader:
	manifests = read_manifest_json(Path(config['manifests']['train']))
	pseudo_inputs = discover_pseudo_target_inputs(
		Path(config['pseudo_targets']['input_dir']),
		k=int(config['pseudo_targets']['k']),
	)
	dataset = NopimsStratPseudoTargetDataset(
		manifests,
		pseudo_inputs,
		local_crop_size_xyz=tuple(config['data']['local_crop_size']),
		patch_size_xyz=tuple(config['model']['patch_size']),
		seed=int(config['train']['seed']),
		samples_per_epoch=1,
		zero_mask=ZeroMaskConfig(
			enabled=False,
			zero_atol=0.0,
			z_sample_influence_radius=0,
			xy_trace_influence_radius=0,
		),
		min_valid_fraction=0.0,
		max_resample_attempts=2,
	)
	return build_strat_pseudo_target_dataloader(
		dataset,
		batch_size=1,
		num_workers=0,
		shuffle=False,
		seed=int(config['train']['seed']),
		device='cpu',
	)


def _write_fixture_files(
	tmp_path: Path,
	*,
	invalid_first_token: bool,
	encoder_depth: int = 1,
) -> Mapping[str, Path]:
	volume = np.linspace(-1.0, 1.0, num=4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
	volume_path = tmp_path / 'survey' / 'amplitude.npy'
	volume_path.parent.mkdir(parents=True)
	np.save(volume_path, volume)
	stats_path = tmp_path / 'survey' / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1.0,
			clip_high=1.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	manifest = SurveyManifest(
		survey_id='survey',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=volume_path,
			shape_xyz=(4, 4, 4),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	manifest_path = tmp_path / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	path_list = tmp_path / 'train_path_list.txt'
	path_list.write_text(f'{volume_path}\n', encoding='utf-8')

	pseudo_root = tmp_path / 'pseudo_targets'
	labels = (np.arange(8, dtype=np.int32) % 3).reshape(2, 2, 2)
	valid_tokens = np.ones((2, 2, 2), dtype=np.bool_)
	confidence = np.ones((2, 2, 2), dtype=np.float32)
	if invalid_first_token:
		labels[0, 0, 0] = -1
		valid_tokens[0, 0, 0] = False
		confidence[0, 0, 0] = 0.0
	write_pseudo_target(
		pseudo_root,
		k=3,
		survey_id='survey',
		labels=labels,
		confidence=confidence,
		valid_tokens=valid_tokens,
	)
	teacher_checkpoint = tmp_path / 'teacher.pt'
	_write_teacher_checkpoint(
		teacher_checkpoint,
		manifest_path=manifest_path,
		path_list=path_list,
		encoder_depth=encoder_depth,
	)
	return {
		'manifest': manifest_path,
		'path_list': path_list,
		'pseudo_root': pseudo_root,
		'teacher_checkpoint': teacher_checkpoint,
	}


def _write_teacher_checkpoint(
	path: Path,
	*,
	manifest_path: Path,
	path_list: Path,
	encoder_depth: int = 1,
) -> None:
	torch.manual_seed(5)
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=encoder_depth,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)
	checkpoint_config = _teacher_checkpoint_config(
		manifest_path=manifest_path,
		path_list=path_list,
		encoder_depth=encoder_depth,
	)
	torch.save(
		{
			'model_state_dict': model.state_dict(),
			'config': checkpoint_config,
		},
		path,
	)


def _teacher_checkpoint_config(
	*,
	manifest_path: Path,
	path_list: Path,
	encoder_depth: int = 1,
) -> dict[str, object]:
	return deepcopy(
		{
			'stage': 'train_amp_mae',
			'paths': {'output_root': str(manifest_path.parent / 'teacher_run')},
			'manifests': {
				'train': str(manifest_path),
				'train_path_list': str(path_list),
			},
			'data': {
				'grid_order': list(GRID_ORDER_XYZ),
				'volume_format': 'npy_memmap',
				'input_channels': 1,
				'target_channels': 1,
				'use_context': False,
				'local_crop_size': [4, 4, 4],
				'min_valid_fraction': 0.0,
				'max_resample_attempts': 2,
				'amplitude_agc': {'enabled': False},
			},
			'model': {
				'name': 'amp_mae3d',
				'in_channels': 1,
				'out_channels': 1,
				'patch_size': [2, 2, 2],
				'encoder_dim': 12,
				'encoder_depth': encoder_depth,
				'encoder_heads': 3,
				'decoder_dim': 12,
				'decoder_depth': 1,
				'decoder_heads': 3,
			},
			'masking': {
				'spatial_mask_ratio': 0.5,
				'spatial_mask_mode': 'block',
				'block_size_tokens': [1, 1, 1],
			},
			'loss': {
				'reconstruction': 'huber',
				'huber_delta': 1.0,
				'gradient_weight': 0.0,
				'visible_reconstruction_weight': 0.0,
				'target_normalization': {'mode': 'none'},
				'valid_mask_mode': 'voxel',
			},
			'train': {
				'batch_size': 1,
				'samples_per_epoch': 1,
				'epochs': 1,
				'num_workers': 0,
				'shuffle': False,
				'lr': 1.0e-4,
				'weight_decay': 0.0,
				'amp': False,
				'device': 'cpu',
				'seed': 5,
				'grad_clip_norm': 1.0,
			},
			'zero_mask': {
				'enabled': False,
				'zero_atol': 0.0,
				'z_sample_influence_radius': 0,
				'xy_trace_influence_radius': 0,
			},
		},
	)
