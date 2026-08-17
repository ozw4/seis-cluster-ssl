from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.config import resolve_barlow_twins_training_config
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.barlow_twins import run_barlow_twins_pretraining
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	load_barlow_twins_checkpoint,
	restore_barlow_twins_checkpoint,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_cli_dry_run_applies_max_steps_without_creating_artifacts(
	tmp_path: Path,
) -> None:
	output_root = tmp_path / 'must-not-exist'
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/train_amp_barlow_twins.py',
			'--config',
			'proc/configs/seis_ssl_cluster/train_amp_barlow_twins.yaml',
			'--output-root',
			str(output_root),
			'--max-steps',
			'7',
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert 'stage: barlow_twins_training' in result.stdout
	assert 'train.max_steps: 7' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout
	assert not output_root.exists()


def test_checkpoint_contract_round_trip_and_epoch_resume(tmp_path: Path) -> None:
	config = resolve_barlow_twins_training_config(_tiny_config(tmp_path))
	first_path = run_barlow_twins_pretraining(config)
	payload = load_barlow_twins_checkpoint(first_path, map_location='cpu')

	assert first_path.name == 'latest.pt'
	assert (first_path.parent / 'best.pt').is_file()
	assert (first_path.parent / 'history.json').is_file()
	assert (first_path.parent / 'resolved_config.json').is_file()
	assert payload['pretraining_method'] == 'barlow_twins_3d'
	assert payload['checkpoint_kind'] == 'barlow_twins_pretraining'
	assert payload['trained_parameter_prefixes'] == [
		'patch_projection.',
		'encoder.',
	]
	assert payload['global_step'] == 1
	assert payload['training_state']['completed_epoch'] is True

	backbone = _backbone()
	assert set(payload['model_state_dict']) == set(backbone.state_dict())
	assert all(not key.startswith('backbone.') for key in payload['model_state_dict'])
	assert all(not key.startswith('projector.') for key in payload['model_state_dict'])
	assert payload['projector_state_dict']

	wrapper = BarlowTwins3D(backbone, projector_dim=4)
	optimizer = torch.optim.AdamW(wrapper.pretraining_parameters(), lr=1.0e-3)
	resume_config = resolve_barlow_twins_training_config(
		_tiny_config(tmp_path, epochs=2, max_steps=2)
	)
	state = restore_barlow_twins_checkpoint(
		payload,
		backbone=backbone,
		projector=wrapper.projector,
		optimizer=optimizer,
		scaler=None,
		scaler_required=False,
		config=resume_config,
	)
	assert state.start_epoch == 2
	assert state.global_step == 1
	for key, value in backbone.state_dict().items():
		assert torch.equal(value, payload['model_state_dict'][key])
	for key, value in wrapper.projector.state_dict().items():
		assert torch.equal(value, payload['projector_state_dict'][key])

	resumed_path = run_barlow_twins_pretraining(
		resume_config,
		resume=first_path,
	)
	resumed = load_barlow_twins_checkpoint(resumed_path, map_location='cpu')
	assert resumed['epoch'] == 2
	assert resumed['global_step'] == 2
	history = json.loads(
		(resumed_path.parent / 'history.json').read_text(encoding='utf-8')
	)
	assert [row['global_step'] for row in history] == [1, 2]


def _tiny_config(
	tmp_path: Path,
	*,
	epochs: int = 1,
	max_steps: int = 1,
) -> dict[str, object]:
	manifest_path = _write_synthetic_manifest(tmp_path / 'survey')
	path_list = tmp_path / 'train_npy_paths.txt'
	path_list.write_text(f'{tmp_path / "survey" / "amplitude.npy"}\n', encoding='utf-8')
	return {
		'paths': {
			'artifact_root': str(tmp_path / 'artifacts'),
			'output_root': str(tmp_path / 'artifacts' / 'run'),
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
			'encoder_depth': 1,
			'encoder_heads': 1,
			'decoder_dim': 4,
			'decoder_depth': 1,
			'decoder_heads': 1,
		},
		'barlow_twins': {'projector_dim': 4},
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


def _backbone() -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		patch_size_xyz=(2, 2, 2),
		encoder_dim=4,
		encoder_depth=1,
		encoder_heads=1,
		decoder_dim=4,
		decoder_depth=1,
		decoder_heads=1,
	)


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
