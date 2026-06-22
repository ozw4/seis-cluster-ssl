from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.training import load_checkpoint, save_checkpoint
from seis_ssl_cluster.training.checkpoint import capture_rng_state, restore_rng_state

if TYPE_CHECKING:
	from pathlib import Path


def test_save_checkpoint_defaults_and_plain_config_values(tmp_path: Path) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

	checkpoint_path = save_checkpoint(
		tmp_path / 'checkpoint.pt',
		model=model,
		optimizer=optimizer,
		epoch=2,
		global_step=5,
		config={
			'path': tmp_path / 'data',
			'nested': {'weights': tmp_path / 'weights.pt'},
		},
		metrics={'loss': 1.25},
		training_state={
			'schema_version': 1,
			'nested': {'artifact': tmp_path / 'artifact.json'},
		},
	)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	assert payload['epoch'] == 2
	assert payload['global_step'] == 5
	assert payload['amp_enabled'] is False
	assert payload['scaler_state_dict'] is None
	assert payload['config']['path'] == str(tmp_path / 'data')
	assert payload['config']['nested']['weights'] == str(tmp_path / 'weights.pt')
	assert payload['training_state'] == {
		'schema_version': 1,
		'nested': {'artifact': str(tmp_path / 'artifact.json')},
	}
	assert payload['metrics'] == {'loss': 1.25}
	assert set(payload['rng_state']) >= {'python', 'numpy', 'torch'}


def test_save_checkpoint_uses_atomic_replace(tmp_path: Path) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	path = tmp_path / 'checkpoint.pt'

	first = save_checkpoint(
		path,
		model=model,
		optimizer=optimizer,
		epoch=1,
		config={'stage': 'train_amp_mae'},
		metrics={'loss': 1.0},
	)
	second = save_checkpoint(
		path,
		model=model,
		optimizer=optimizer,
		epoch=2,
		config={'stage': 'train_amp_mae'},
		metrics={'loss': 0.5},
	)

	assert first == second == path
	assert not list(tmp_path.glob('*.tmp'))
	payload = load_checkpoint(path, map_location='cpu')
	assert payload['epoch'] == 2
	assert payload['metrics']['loss'] == 0.5


def test_save_checkpoint_requires_scaler_when_amp_enabled(tmp_path: Path) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

	with pytest.raises(ValueError, match='scaler is required'):
		save_checkpoint(
			tmp_path / 'checkpoint.pt',
			model=model,
			optimizer=optimizer,
			epoch=1,
			config={'stage': 'train_amp_mae'},
			amp_enabled=True,
			scaler=None,
		)


def test_restore_rng_state_rejects_partial_rng_payload() -> None:
	with pytest.raises(TypeError, match='rng_state must be a mapping'):
		restore_rng_state({})

	payload: dict[str, object] = {'rng_state': capture_rng_state()}
	rng_state = payload['rng_state']
	assert isinstance(rng_state, dict)
	rng_state['torch'] = None

	with pytest.raises(TypeError, match=r'rng_state\.torch'):
		restore_rng_state(payload)
