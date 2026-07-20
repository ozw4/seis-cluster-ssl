from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from seis_ssl_cluster.training.collate import (
	mae_collate_fn,
	move_batch_to_device,
	strat_multi_head_target_collate_fn,
)


def _sample(coords: dict[str, object] | None = None) -> dict[str, object]:
	shape = (4, 4, 4)
	return {
		'x': np.ones((1, *shape), dtype=np.float32),
		'spatial_mask': np.asarray(
			[[[True, False], [False, False]], [[False, False], [False, False]]],
		),
		'local_valid_mask': np.ones(shape, dtype=bool),
		'coords': coords or {'survey_id': 'survey-a'},
	}


def test_mae_collate_fn_stacks_amplitude_batch_contract() -> None:
	batch = mae_collate_fn([_sample(), _sample({'survey_id': 'survey-b'})])

	assert batch['x'].shape == (2, 1, 4, 4, 4)
	assert batch['spatial_mask'].shape == (2, 2, 2, 2)
	assert batch['equal_visible_count'] is True
	assert 'target' not in batch
	assert 'visible_spatial_mask' not in batch
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
	assert moved['equal_visible_count'] is True
	assert moved['coords'] is batch['coords']


def test_move_batch_to_device_recurses_through_nested_targets_only() -> None:
	coords = {'path': Path('artifacts/survey.npy'), 'start_xyz': (1, 2, 3)}
	batch: dict[str, object] = {
		'x': torch.ones((1,)),
		'strat_multi_targets': {'k6': {'labels': torch.ones((1,), dtype=torch.long)}},
		'coords': [coords],
	}

	moved = move_batch_to_device(batch, torch.device('cpu'))

	assert moved['x'].device.type == 'cpu'
	assert moved['strat_multi_targets']['k6']['labels'].device.type == 'cpu'
	assert moved['coords'] is batch['coords']


def test_multi_head_collate_stacks_nested_targets_in_head_order() -> None:
	first = _multi_head_sample((6, 8, 10))
	second = _multi_head_sample((6, 8, 10))

	batch = strat_multi_head_target_collate_fn([first, second])

	assert list(batch) == ['x', 'local_valid_mask', 'strat_multi_targets', 'coords']
	assert list(batch['strat_multi_targets']) == ['k6', 'k8', 'k10']
	assert batch['strat_multi_targets']['k8']['labels'].shape == (2, 2, 3, 4)
	assert batch['strat_multi_targets']['k8']['labels'].dtype == torch.long
	assert batch['strat_multi_targets']['k8']['valid_mask'].dtype == torch.bool


def test_multi_head_collate_rejects_missing_head_and_dtype_mismatch() -> None:
	with pytest.raises(ValueError, match='identical multi-head target order'):
		strat_multi_head_target_collate_fn(
			[_multi_head_sample((6, 8)), _multi_head_sample((6, 8, 10))]
		)
	second = _multi_head_sample((6,))
	second['strat_multi_targets']['k6']['confidence'] = np.ones(
		(2, 3, 4),
		dtype=np.float64,
	)
	with pytest.raises(
		TypeError,
		match="all 'k6' 'confidence' arrays must share dtype",
	):
		strat_multi_head_target_collate_fn([_multi_head_sample((6,)), second])


def test_mae_collate_fn_records_unequal_visible_counts_on_cpu() -> None:
	first = _sample()
	second = _sample()
	second['spatial_mask'] = np.ones((2, 2, 2), dtype=np.bool_)

	batch = mae_collate_fn([first, second])

	assert batch['equal_visible_count'] is False


def test_mae_collate_fn_rejects_empty_samples() -> None:
	with pytest.raises(ValueError, match='at least one sample'):
		mae_collate_fn([])


def _multi_head_sample(head_ks: tuple[int, ...]) -> dict[str, object]:
	targets: dict[str, dict[str, np.ndarray]] = {}
	for k in head_ks:
		targets[f'k{k}'] = {
			'labels': np.zeros((2, 3, 4), dtype=np.int64),
			'confidence': np.ones((2, 3, 4), dtype=np.float32),
			'boundary_weight': np.ones((2, 3, 4), dtype=np.float32),
			'valid_mask': np.ones((2, 3, 4), dtype=np.bool_),
		}
	return {
		'x': np.ones((1, 4, 4, 4), dtype=np.float32),
		'local_valid_mask': np.ones((4, 4, 4), dtype=np.bool_),
		'strat_multi_targets': targets,
		'coords': {'survey_id': 'survey-a'},
	}
