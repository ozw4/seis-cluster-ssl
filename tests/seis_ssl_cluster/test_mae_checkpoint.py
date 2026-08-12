# ruff: noqa: CPY001

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.checkpoint import capture_rng_state
from seis_ssl_cluster.training.mae_checkpoint import (
	_best_metric_from_metrics,
	_is_improved_best_metric,
	_restore_mae_checkpoint,
	_save_mae_rolling_checkpoint,
	inspect_mae_checkpoint,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_inspect_mae_checkpoint_returns_immutable_validated_evidence(
	tmp_path: Path,
) -> None:
	model = torch.nn.Linear(2, 1)
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
	config = {'stage': 'train_amp_mae', 'model': {'width': 2}}
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = torch.Generator().get_state()
	result = _save_mae_rolling_checkpoint(
		tmp_path,
		model=model,
		optimizer=optimizer,
		epoch=3,
		config=config,
		metrics={'loss': 0.25, 'gradient_norm': 1.5},
		global_step=12,
		amp_enabled=False,
		scaler=None,
		checkpoint_kind='epoch',
		batch_index=None,
		rng_state=rng_state,
	)

	inspection = inspect_mae_checkpoint(
		result.latest_path,
		resolved_config=config,
		model=torch.nn.Linear(2, 1),
		resolved_precision='float32',
		amp_enabled=False,
		scaler_present=False,
	)

	assert inspection.schema_version == 2
	assert inspection.stage == 'train_amp_mae'
	assert inspection.checkpoint_kind == 'epoch'
	assert inspection.batch_index is None
	assert inspection.epoch == 3
	assert inspection.global_step == 12
	assert inspection.resolved_precision == 'float32'
	assert inspection.amp_enabled is False
	assert inspection.scaler_present is False
	assert inspection.metrics_dict() == {'loss': 0.25, 'gradient_norm': 1.5}
	assert inspection.best_metric_key == 'loss'
	assert inspection.best_metric_value == 0.25
	with pytest.raises(FrozenInstanceError):
		inspection.epoch = 4  # type: ignore[misc]


def test_inspect_mae_checkpoint_rejects_amp_enabled_mismatch(
	tmp_path: Path,
) -> None:
	model = torch.nn.Linear(1, 1)
	config = {'stage': 'train_amp_mae', 'train': {'amp_dtype': 'bfloat16'}}
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = torch.Generator().get_state()
	result = _save_mae_rolling_checkpoint(
		tmp_path,
		model=model,
		optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
		epoch=1,
		config=config,
		metrics={'loss': 1.0},
		global_step=1,
		amp_enabled=True,
		scaler=None,
		checkpoint_kind='epoch',
		batch_index=None,
		rng_state=rng_state,
	)

	with pytest.raises(
		ValueError,
		match='amp_enabled does not match the current runtime',
	):
		inspect_mae_checkpoint(
			result.latest_path,
			resolved_config=config,
			model=torch.nn.Linear(1, 1),
			resolved_precision='bfloat16',
			amp_enabled=False,
			scaler_present=False,
		)


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
	assert payload['training_state']['schema_version'] == 2
	assert payload['training_state']['resolved_precision'] == 'bfloat16'


@pytest.mark.parametrize('amp_enabled', [False, True])
def test_schema_v1_checkpoint_without_precision_resumes(
	tmp_path: Path,
	*,
	amp_enabled: bool,
) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	scaler = torch.amp.GradScaler('cpu') if amp_enabled else None
	result = _save_mae_rolling_checkpoint(
		tmp_path,
		model=model,
		optimizer=optimizer,
		epoch=1,
		config={'stage': 'train_amp_mae'},
		metrics={'loss': 1.0},
		global_step=1,
		amp_enabled=amp_enabled,
		scaler=scaler,
		checkpoint_kind='epoch',
		batch_index=None,
	)
	payload = load_checkpoint(result.latest_path, map_location='cpu')
	payload['training_state']['schema_version'] = 1
	payload['training_state'].pop('resolved_precision')
	payload['rng_state']['dataloader_generator'] = torch.Generator().get_state()
	resume_model = torch.nn.Linear(1, 1)
	resume_optimizer = torch.optim.SGD(resume_model.parameters(), lr=0.1)

	state = _restore_mae_checkpoint(
		payload=payload,
		model=resume_model,
		optimizer=resume_optimizer,
		scaler=scaler,
		amp_enabled=amp_enabled,
		scaler_required=amp_enabled,
	)

	assert state.start_epoch == 2
	assert state.global_step == 1


def test_auto_bfloat16_checkpoint_rejects_float16_resume_without_cuda(
	tmp_path: Path,
) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	result = _save_mae_rolling_checkpoint(
		tmp_path,
		model=model,
		optimizer=optimizer,
		epoch=1,
		config={'stage': 'train_amp_mae', 'train': {'amp_dtype': 'auto'}},
		metrics={'loss': 1.0},
		global_step=1,
		amp_enabled=True,
		scaler=None,
		checkpoint_kind='epoch',
		batch_index=None,
	)
	payload = load_checkpoint(result.latest_path, map_location='cpu')
	payload['rng_state']['dataloader_generator'] = torch.Generator().get_state()
	resume_model = torch.nn.Linear(1, 1)
	resume_optimizer = torch.optim.SGD(resume_model.parameters(), lr=0.1)

	with pytest.raises(
		ValueError,
		match="checkpoint='bfloat16', current='float16'",
	):
		_restore_mae_checkpoint(
			payload=payload,
			model=resume_model,
			optimizer=resume_optimizer,
			scaler=torch.amp.GradScaler('cpu'),
			amp_enabled=True,
			scaler_required=True,
		)


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
