from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

import seis_ssl_cluster.training.mae as mae_training
from seis_ssl_cluster.models.mae.patching import patchify_3d
from seis_ssl_cluster.training.mae import run_mae_pretraining
from seis_ssl_cluster.visualization.mae_debug import (
	MaeDebugVisualizationConfig,
	save_mae_debug_visualization_pngs,
	unpatchify_mae_predictions,
)
from tests.seis_ssl_cluster.test_training_smoke import _tiny_config


def test_unpatchify_mae_predictions_produces_expected_dense_values() -> None:
	pred_patches = torch.arange(16, dtype=torch.float32).reshape(1, 2, 1, 8)

	dense = unpatchify_mae_predictions(
		pred_patches,
		token_grid_shape=(2, 1, 1),
		patch_size_xyz=(2, 2, 2),
	)

	assert dense.shape == (1, 1, 4, 2, 2)
	assert dense[0, 0, 0, 0, 0] == 0
	assert dense[0, 0, 2, 0, 0] == 8


def test_save_mae_debug_visualization_pngs_writes_xy_and_xz(
	tmp_path: Path,
) -> None:
	target = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
	spatial_mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	spatial_mask[0, 0, 0, 0] = True
	local_valid_mask = torch.ones((1, 4, 4, 4), dtype=torch.bool)
	local_valid_mask[:, 0, :, :] = False
	pred_patches = patchify_3d(target + 0.5, (2, 2, 2))

	paths = save_mae_debug_visualization_pngs(
		batch={
			'x': target.clone(),
			'target': target,
			'spatial_mask': spatial_mask,
			'local_valid_mask': local_valid_mask,
			'coords': [
				{
					'survey_id': 'survey-a',
					'local_start_xyz': (1, 2, 3),
				},
			],
		},
		model_output={
			'pred_patches': pred_patches,
			'token_grid_shape': (2, 2, 2),
		},
		patch_size_xyz=(2, 2, 2),
		epoch=1,
		global_step=7,
		config=MaeDebugVisualizationConfig(output_dir=tmp_path, dpi=40),
		metrics={'loss': 0.25},
	)

	assert paths == [
		tmp_path / 'epoch_0001_step_000007_xy.png',
		tmp_path / 'epoch_0001_step_000007_xz.png',
	]
	for path in paths:
		assert path.exists()
		assert path.stat().st_size > 0
		assert path.with_suffix('.json').exists()
		image = plt.imread(path)
		assert image.shape[0] > 0
		assert image.shape[1] > 0
	assert plt.get_fignums() == []


def test_mae_training_writes_debug_visualization_pngs(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['train']['max_steps'] = 1
	cfg['visualization'] = {
		'mae_debug': {
			'enabled': True,
			'every_steps': 1,
			'every_epochs': None,
			'max_samples': 1,
			'dpi': 30,
			'columns': ['target', 'prediction', 'abs_error', 'valid_mask'],
		},
	}

	checkpoint_path = run_mae_pretraining(cfg)
	visualization_dir = (
		Path(cfg['paths']['output_root']) / 'visualizations' / 'mae_debug'
	)

	assert checkpoint_path.is_file()
	assert (visualization_dir / 'epoch_0001_step_000001_xy.png').is_file()
	assert (visualization_dir / 'epoch_0001_step_000001_xz.png').is_file()
	assert plt.get_fignums() == []


def test_mae_training_disabled_debug_visualization_creates_no_directory(
	tmp_path: Path,
) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['train']['max_steps'] = 1
	cfg['visualization'] = {
		'mae_debug': {
			'enabled': False,
			'every_steps': 1,
		},
	}

	checkpoint_path = run_mae_pretraining(cfg)

	assert checkpoint_path.is_file()
	assert not (tmp_path / 'run' / 'visualizations' / 'mae_debug').exists()


def test_mae_debug_trigger_helpers_reject_zero_intervals(tmp_path: Path) -> None:
	config = MaeDebugVisualizationConfig(
		output_dir=tmp_path,
		every_steps=0,
		every_epochs=0,
	)
	step_triggered = mae_training._mae_debug_step_triggered  # noqa: SLF001
	epoch_triggered = mae_training._mae_debug_epoch_triggered  # noqa: SLF001

	with pytest.raises(ValueError, match=r'mae_debug\.every_steps.*positive'):
		step_triggered(
			config=config,
			global_step=1,
		)
	with pytest.raises(ValueError, match=r'mae_debug\.every_epochs.*positive'):
		epoch_triggered(
			config=config,
			epoch=1,
			already_triggered=False,
		)
