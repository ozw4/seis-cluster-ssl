from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import seis_ssl_cluster.training.mae as mae_training
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.training import (
	load_checkpoint,
	mae_collate_fn,
	run_mae_pretraining,
	train_mae_one_epoch,
)


class _TinyAmpModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.weight = torch.nn.Parameter(torch.tensor(1.0))

	def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
		target = batch['target']
		pred = self.weight * torch.ones(
			(target.shape[0], 8, 1, 8),
			dtype=target.dtype,
			device=target.device,
		)
		return {
			'pred_patches': pred,
			'spatial_mask': batch['spatial_mask'],
		}


def test_two_step_cpu_synthetic_smoke_run_writes_checkpoint(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['train']['max_steps'] = 2

	checkpoint_path = run_mae_pretraining(cfg)

	assert checkpoint_path.name == 'mae_epoch_0001.pt'
	assert checkpoint_path.is_file()
	assert (_path_like(cfg['paths']['output_root']) / 'mae_latest.pt').is_file()
	checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
	assert checkpoint['epoch'] == 1
	assert checkpoint['global_step'] == 2
	assert checkpoint['amp_enabled'] is False
	assert checkpoint['scaler_state_dict'] is None
	assert checkpoint['training_state']['checkpoint_kind'] == 'epoch'
	assert checkpoint['training_state']['stage'] == 'train_amp_mae'
	assert checkpoint['package_version'] is None
	assert isinstance(checkpoint['rng_state']['dataloader_generator'], torch.Tensor)
	assert np.isfinite(checkpoint['metrics']['loss'])
	assert (_path_like(cfg['paths']['output_root']) / 'resolved_config.json').is_file()
	assert (_path_like(cfg['paths']['output_root']) / 'manifest.json').is_file()
	assert (_path_like(cfg['paths']['output_root']) / 'run_metadata.json').is_file()


def test_run_snapshots_configured_train_path_list(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	path_list = tmp_path / 'train_paths.txt'
	path_list.write_text('survey/amplitude.npy\n', encoding='utf-8')
	cfg['manifests']['train_path_list'] = str(path_list)

	run_mae_pretraining(cfg)

	snapshot = _path_like(cfg['paths']['output_root']) / 'inputs' / path_list.name
	assert snapshot.read_text(encoding='utf-8') == 'survey/amplitude.npy\n'


def test_run_rejects_missing_configured_train_path_list(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	missing = tmp_path / 'missing_train_paths.txt'
	cfg['manifests']['train_path_list'] = str(missing)

	with pytest.raises(FileNotFoundError, match=r'manifests\.train_path_list'):
		run_mae_pretraining(cfg)


def test_fresh_run_rejects_existing_snapshot_files(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	output_root = _path_like(cfg['paths']['output_root'])
	output_root.mkdir(parents=True)
	(output_root / 'resolved_config.json').write_text('{}\n', encoding='utf-8')

	with pytest.raises(FileExistsError, match='output_root is nonempty'):
		run_mae_pretraining(cfg)


def test_resume_advances_global_step(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['train']['max_steps'] = 1
	checkpoint_path = run_mae_pretraining(cfg)
	checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
	assert checkpoint['training_state']['checkpoint_kind'] == 'step'
	assert checkpoint['training_state']['batch_index'] == 0

	resume_cfg = deepcopy(cfg)
	resume_cfg['train']['epochs'] = 2
	resume_cfg['train']['max_steps'] = 2
	resumed_path = run_mae_pretraining(resume_cfg, resume=checkpoint_path)

	payload = load_checkpoint(resumed_path, map_location='cpu')
	assert resumed_path.name == 'mae_epoch_0001.pt'
	assert payload['epoch'] == 1
	assert payload['global_step'] == 2
	assert payload['training_state']['checkpoint_kind'] == 'epoch'


def test_step_checkpoint_resume_continues_unfinished_epoch(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['train']['samples_per_epoch'] = 2
	cfg['train']['max_steps'] = 1
	cfg['train']['checkpoint_every_steps'] = 1
	run_mae_pretraining(cfg)
	step_checkpoint = _path_like(cfg['paths']['output_root']) / 'mae_step_00000001.pt'

	resume_cfg = deepcopy(cfg)
	resume_cfg['train']['max_steps'] = 2
	resumed_path = run_mae_pretraining(resume_cfg, resume=step_checkpoint)

	payload = load_checkpoint(resumed_path, map_location='cpu')
	assert resumed_path.name == 'mae_epoch_0001.pt'
	assert payload['epoch'] == 1
	assert payload['global_step'] == 2


def test_shuffle_step_resume_matches_uninterrupted_training_stream(
	tmp_path: Path,
) -> None:
	full_cfg = _tiny_config(tmp_path / 'full')
	full_cfg['train']['shuffle'] = True
	full_cfg['train']['samples_per_epoch'] = 4
	full_cfg['train']['max_steps'] = 3

	resume_cfg = _tiny_config(tmp_path / 'resume')
	resume_cfg['train']['shuffle'] = True
	resume_cfg['train']['samples_per_epoch'] = 4
	resume_cfg['train']['max_steps'] = 1
	resume_cfg['train']['checkpoint_every_steps'] = 1

	full_checkpoint = run_mae_pretraining(full_cfg)
	step_checkpoint = run_mae_pretraining(resume_cfg)
	resume_cfg['train']['max_steps'] = 3
	resumed_checkpoint = run_mae_pretraining(resume_cfg, resume=step_checkpoint)

	_assert_model_states_equal(
		load_checkpoint(full_checkpoint, map_location='cpu')['model_state_dict'],
		load_checkpoint(resumed_checkpoint, map_location='cpu')['model_state_dict'],
	)


def test_shuffle_epoch_resume_matches_uninterrupted_training_stream(
	tmp_path: Path,
) -> None:
	full_cfg = _tiny_config(tmp_path / 'full')
	full_cfg['train']['shuffle'] = True
	full_cfg['train']['samples_per_epoch'] = 3
	full_cfg['train']['epochs'] = 2

	resume_cfg = _tiny_config(tmp_path / 'resume')
	resume_cfg['train']['shuffle'] = True
	resume_cfg['train']['samples_per_epoch'] = 3

	full_checkpoint = run_mae_pretraining(full_cfg)
	epoch_checkpoint = run_mae_pretraining(resume_cfg)
	resume_cfg['train']['epochs'] = 2
	resumed_checkpoint = run_mae_pretraining(resume_cfg, resume=epoch_checkpoint)

	_assert_model_states_equal(
		load_checkpoint(full_checkpoint, map_location='cpu')['model_state_dict'],
		load_checkpoint(resumed_checkpoint, map_location='cpu')['model_state_dict'],
	)


def test_persistent_worker_epoch_resume_matches_uninterrupted_training_stream(
	tmp_path: Path,
) -> None:
	full_cfg = _tiny_config(tmp_path / 'full')
	full_cfg['train']['num_workers'] = 1
	full_cfg['train']['samples_per_epoch'] = 2
	full_cfg['train']['epochs'] = 2

	resume_cfg = _tiny_config(tmp_path / 'resume')
	resume_cfg['train']['num_workers'] = 1
	resume_cfg['train']['samples_per_epoch'] = 2

	full_checkpoint = run_mae_pretraining(full_cfg)
	epoch_checkpoint = run_mae_pretraining(resume_cfg)
	resume_cfg['train']['epochs'] = 2
	resumed_checkpoint = run_mae_pretraining(resume_cfg, resume=epoch_checkpoint)

	_assert_model_states_equal(
		load_checkpoint(full_checkpoint, map_location='cpu')['model_state_dict'],
		load_checkpoint(resumed_checkpoint, map_location='cpu')['model_state_dict'],
	)


def test_resume_rejects_partial_checkpoint_payload(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	checkpoint_path = run_mae_pretraining(cfg)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	for key in (
		'model_state_dict',
		'optimizer_state_dict',
		'epoch',
		'global_step',
		'amp_enabled',
		'scaler_state_dict',
		'config',
		'package_version',
		'metrics',
		'rng_state',
		'training_state',
	):
		partial_path = tmp_path / f'missing-{key}.pt'
		partial_payload = dict(payload)
		partial_payload.pop(key)
		torch.save(partial_payload, partial_path)

		with pytest.raises(ValueError, match=key):
			run_mae_pretraining(cfg, resume=partial_path)

	partial_path = tmp_path / 'missing-dataloader-rng.pt'
	partial_payload = dict(payload)
	partial_payload['rng_state'] = dict(payload['rng_state'])
	partial_payload['rng_state'].pop('dataloader_generator')
	torch.save(partial_payload, partial_path)

	with pytest.raises(ValueError, match='dataloader_generator'):
		run_mae_pretraining(cfg, resume=partial_path)

	for key in ('schema_version', 'stage', 'checkpoint_kind', 'batch_index'):
		partial_path = tmp_path / f'missing-training-state-{key}.pt'
		partial_payload = dict(payload)
		training_state = dict(payload['training_state'])
		training_state.pop(key)
		partial_payload['training_state'] = training_state
		torch.save(partial_payload, partial_path)

		with pytest.raises(ValueError, match=fr'training_state is missing {key}'):
			run_mae_pretraining(cfg, resume=partial_path)

	for key in ('python', 'numpy', 'torch', 'dataloader_generator'):
		partial_path = tmp_path / f'invalid-rng-{key}.pt'
		partial_payload = dict(payload)
		rng_state = dict(payload['rng_state'])
		rng_state[key] = None
		partial_payload['rng_state'] = rng_state
		torch.save(partial_payload, partial_path)

		with pytest.raises(TypeError, match=fr'rng_state\.{key}'):
			run_mae_pretraining(cfg, resume=partial_path)


def test_amp_resume_requires_scaler_state(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	checkpoint_path = run_mae_pretraining(cfg)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	with pytest.raises(ValueError, match='scaler_state_dict'):
		mae_training._restore_mae_checkpoint(  # noqa: SLF001
			payload=payload,
			model=torch.nn.Linear(1, 1),
			optimizer=torch.optim.SGD(
				torch.nn.Linear(1, 1).parameters(),
				lr=0.1,
			),
			scaler=None,
			amp_enabled=True,
		)


def test_grad_clip_norm_calls_torch_clip_on_cpu(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	calls: list[float] = []

	def fake_clip_grad_norm_(
		parameters: object,
		max_norm: float,
	) -> torch.Tensor:
		list(parameters)
		calls.append(max_norm)
		return torch.tensor(0.25)

	monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', fake_clip_grad_norm_)
	dataloader = torch.utils.data.DataLoader(
		[_mae_sample()],
		batch_size=1,
		collate_fn=mae_collate_fn,
	)
	model = _TinyAmpModel()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

	state = train_mae_one_epoch(
		model=model,
		dataloader=dataloader,
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		patch_size_xyz=(2, 2, 2),
		loss_config={'reconstruction': 'mse'},
		grad_clip_norm=1.0,
	)

	assert calls == [1.0]
	assert state.metrics['grad_norm'] == pytest.approx(0.25)


def test_nonfinite_loss_reports_survey_and_coordinates(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	def nan_loss(**_: object) -> dict[str, torch.Tensor]:
		return {
			'loss': torch.tensor(float('nan')),
			'loss_reconstruction': torch.tensor(float('nan')),
			'loss_gradient': torch.tensor(float('inf')),
			'valid_reconstruction_voxels': torch.tensor(8),
		}

	monkeypatch.setattr('seis_ssl_cluster.training.mae.mae_pretraining_loss', nan_loss)
	dataloader = torch.utils.data.DataLoader(
		[_mae_sample()],
		batch_size=1,
		collate_fn=mae_collate_fn,
	)
	model = _TinyAmpModel()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	diagnostics_dir = tmp_path / 'diagnostics'

	with pytest.raises(FloatingPointError, match='diagnostic written to'):
		train_mae_one_epoch(
			model=model,
			dataloader=dataloader,
			optimizer=optimizer,
			device=torch.device('cpu'),
			epoch=3,
			patch_size_xyz=(2, 2, 2),
			loss_config={'reconstruction': 'mse'},
			global_step=1042,
			diagnostics_dir=diagnostics_dir,
		)

	payload = json.loads(
		(diagnostics_dir / 'nonfinite_mae_step_00001042.json').read_text(
			encoding='utf-8',
		),
	)
	assert payload['survey_id'] == ['survey-a']
	assert payload['local_start_xyz'] == [[1, 2, 3]]
	assert payload['coords'][0]['survey_id'] == 'survey-a'
	assert payload['valid_voxel_count'] == 64
	assert payload['tensors']['x']['all_finite'] is True
	assert payload['tensors']['target']['all_finite'] is True
	assert payload['tensors']['prediction']['all_finite'] is True
	assert payload['tensors']['prediction']['shape'] == [1, 1, 4, 4, 4]
	assert payload['losses']['loss'] == {
		'value': None,
		'finite': False,
		'repr': 'nan',
	}


def _tiny_config(tmp_path: Path) -> dict[str, object]:
	manifest_path = _write_synthetic_manifest(tmp_path / 'survey')
	return {
		'stage': 'train_amp_mae',
		'paths': {
			'nopims_root': str(tmp_path),
			'artifact_root': str(tmp_path / 'artifacts'),
			'output_root': str(tmp_path / 'run'),
		},
		'manifests': {'train': str(manifest_path)},
		'data': {
			'grid_order': ['x', 'y', 'z'],
			'volume_format': 'npy_memmap',
			'input_channels': 1,
			'target_channels': 1,
			'use_context': False,
			'local_crop_size': [4, 4, 4],
		},
		'model': {
			'name': 'amp_mae3d',
			'in_channels': 1,
			'out_channels': 1,
			'patch_size': [2, 2, 2],
			'encoder_dim': 12,
			'encoder_depth': 1,
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
			'valid_mask_mode': 'voxel',
		},
		'zero_mask': {'enabled': False},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 2,
			'epochs': 1,
			'num_workers': 0,
			'shuffle': False,
			'lr': 1.0e-4,
			'weight_decay': 0.0,
			'amp': False,
			'device': 'cpu',
			'seed': 7,
			'grad_clip_norm': 1.0,
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


def _mae_sample() -> dict[str, object]:
	return {
		'x': np.ones((1, 4, 4, 4), dtype=np.float32),
		'target': np.ones((1, 4, 4, 4), dtype=np.float32),
		'spatial_mask': np.ones((2, 2, 2), dtype=np.bool_),
		'visible_spatial_mask': np.zeros((2, 2, 2), dtype=np.bool_),
		'local_valid_mask': np.ones((4, 4, 4), dtype=np.bool_),
		'coords': {
			'survey_id': 'survey-a',
			'local_start_xyz': (1, 2, 3),
			'local_size_xyz': (4, 4, 4),
		},
	}


def _path_like(value: object) -> Path:
	if not isinstance(value, str):
		msg = f'expected path string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _assert_model_states_equal(
	left: object,
	right: object,
) -> None:
	assert isinstance(left, dict)
	assert isinstance(right, dict)
	assert set(left) == set(right)
	for key in left:
		torch.testing.assert_close(left[key], right[key])
