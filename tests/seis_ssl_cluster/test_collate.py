from __future__ import annotations

import numpy as np
import pytest
import torch

from seis_ssl_cluster.training.collate import mae_collate_fn, move_batch_to_device


def _sample(coords: dict[str, object] | None = None) -> dict[str, object]:
	shape = (4, 4, 4)
	return {
		'x': np.ones((1, *shape), dtype=np.float32),
		'target': np.ones((1, *shape), dtype=np.float32),
		'spatial_mask': np.asarray(
			[[[True, False], [False, False]], [[False, False], [False, False]]],
		),
		'visible_spatial_mask': np.asarray(
			[[[False, True], [True, True]], [[True, True], [True, True]]],
		),
		'local_valid_mask': np.ones(shape, dtype=bool),
		'coords': coords or {'survey_id': 'survey-a'},
	}


def test_mae_collate_fn_stacks_amplitude_batch_contract() -> None:
	batch = mae_collate_fn([_sample(), _sample({'survey_id': 'survey-b'})])

	assert batch['x'].shape == (2, 1, 4, 4, 4)
	assert batch['target'].shape == (2, 1, 4, 4, 4)
	assert batch['spatial_mask'].shape == (2, 2, 2, 2)
	assert batch['visible_spatial_mask'].shape == (2, 2, 2, 2)
	assert batch['local_valid_mask'].shape == (2, 4, 4, 4)
	assert batch['x'].dtype == torch.float32
	assert batch['spatial_mask'].dtype == torch.bool
	assert batch['local_valid_mask'].dtype == torch.bool
	assert batch['coords'] == [{'survey_id': 'survey-a'}, {'survey_id': 'survey-b'}]


def test_mae_collate_fn_preserves_coords_without_tensor_conversion() -> None:
	coords = {'survey_id': 'survey-a', 'local_start_xyz': (1, 2, 3)}

	batch = mae_collate_fn([_sample(coords)])

	assert batch['coords'] == [coords]
	assert batch['coords'][0] is coords


def test_move_batch_to_device_moves_tensors_and_preserves_coords() -> None:
	coords = {'survey_id': 'survey-a'}
	batch = mae_collate_fn([_sample(coords)])
	moved = move_batch_to_device(batch, torch.device('cpu'))

	assert moved['x'].device == torch.device('cpu')
	assert moved['coords'] is batch['coords']


def test_mae_collate_fn_rejects_empty_samples() -> None:
	with pytest.raises(ValueError, match='at least one sample'):
		mae_collate_fn([])
