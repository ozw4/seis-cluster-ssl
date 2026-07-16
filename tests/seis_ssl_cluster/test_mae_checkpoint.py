from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.mae_checkpoint import (
	_best_metric_from_metrics,
	_is_improved_best_metric,
	_save_mae_rolling_checkpoint,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_rolling_checkpoint_saves_latest_and_best_only(tmp_path: Path) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

	first = _save_mae_rolling_checkpoint(
		tmp_path,
		model=model,
		optimizer=optimizer,
		epoch=1,
		config={'stage': 'train_amp_mae'},
		metrics={'loss': 1.0},
		global_step=1,
		amp_enabled=False,
		scaler=None,
		checkpoint_kind='epoch',
		batch_index=None,
	)

	assert first.latest_path == tmp_path / 'latest.pt'
	assert first.best_path == tmp_path / 'best.pt'
	assert first.best_updated is True
	assert first.best_score == 1.0
	assert first.best_metric_key == 'loss'
	assert (tmp_path / 'latest.pt').is_file()
	assert (tmp_path / 'best.pt').is_file()

	second = _save_mae_rolling_checkpoint(
		tmp_path,
		model=model,
		optimizer=optimizer,
		epoch=2,
		config={'stage': 'train_amp_mae'},
		metrics={'loss': 2.0},
		global_step=2,
		amp_enabled=False,
		scaler=None,
		checkpoint_kind='epoch',
		batch_index=None,
		best_score=first.best_score,
	)

	latest = load_checkpoint(tmp_path / 'latest.pt', map_location='cpu')
	best = load_checkpoint(tmp_path / 'best.pt', map_location='cpu')
	assert second.best_updated is False
	assert second.best_score == 1.0
	assert latest['epoch'] == 2
	assert latest['metrics']['loss'] == 2.0
	assert best['epoch'] == 1
	assert best['metrics']['loss'] == 1.0
	assert sorted(path.name for path in tmp_path.glob('*.pt')) == [
		'best.pt',
		'latest.pt',
	]
	assert not list(tmp_path.glob('*epoch*.pt'))


def test_best_metric_comparison_minimizes_loss() -> None:
	assert _is_improved_best_metric(1.0, None) is True
	assert _is_improved_best_metric(0.5, 1.0) is True
	assert _is_improved_best_metric(1.5, 1.0) is False


def test_bfloat16_amp_checkpoint_does_not_require_scaler(tmp_path: Path) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

	result = _save_mae_rolling_checkpoint(
		tmp_path,
		model=model,
		optimizer=optimizer,
		epoch=1,
		config={'stage': 'train_amp_mae', 'train': {'amp_dtype': 'bfloat16'}},
		metrics={'loss': 1.0},
		global_step=1,
		amp_enabled=True,
		scaler=None,
		checkpoint_kind='epoch',
		batch_index=None,
	)

	payload = load_checkpoint(result.latest_path, map_location='cpu')
	assert payload['amp_enabled'] is True
	assert payload['scaler_state_dict'] is None


def test_best_metric_prefers_validation_loss_when_present() -> None:
	key, score = _best_metric_from_metrics({'loss': 0.1, 'val_loss': 0.2})

	assert key == 'val_loss'
	assert score == 0.2


def test_load_checkpoint_accepts_rolling_and_legacy_epoch_paths(
	tmp_path: Path,
) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	result = _save_mae_rolling_checkpoint(
		tmp_path,
		model=model,
		optimizer=optimizer,
		epoch=1,
		config={'stage': 'train_amp_mae'},
		metrics={'loss': 1.0},
		global_step=1,
		amp_enabled=False,
		scaler=None,
		checkpoint_kind='epoch',
		batch_index=None,
	)
	legacy_epoch_path = tmp_path / 'mae_epoch_0001.pt'
	latest_payload = load_checkpoint(result.latest_path, map_location='cpu')
	torch.save(latest_payload, legacy_epoch_path)

	for path in (tmp_path / 'latest.pt', tmp_path / 'best.pt', legacy_epoch_path):
		payload = load_checkpoint(path, map_location='cpu')
		assert payload['epoch'] == 1
		assert payload['training_state']['stage'] == 'train_amp_mae'
