from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import torch

import seis_ssl_cluster.training.mae as mae_training
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training import load_checkpoint, run_mae_pretraining
from seis_ssl_cluster.training.mae_continuation import (
	configure_mae_continuation_trainability,
)
from tests.seis_ssl_cluster.test_training_smoke import _tiny_config

if TYPE_CHECKING:
	from pathlib import Path

pytestmark = pytest.mark.integration


def test_invalid_continuation_checkpoint_does_not_write_run_snapshots(
	tmp_path: Path,
) -> None:
	config = _tiny_config(tmp_path)
	output_root = tmp_path / 'artifacts' / 'run'
	config['continuation'] = {
		'init_checkpoint': str((tmp_path / 'missing-source.pt').resolve()),
		'unfreeze_top_blocks': 1,
	}

	with pytest.raises(FileNotFoundError, match='checkpoint file does not exist'):
		run_mae_pretraining(config)

	assert output_root.is_dir()
	assert list(output_root.iterdir()) == []


def test_mae_continuation_fresh_and_resume_contract(  # noqa: PLR0915
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	stage1_config = _tiny_config(tmp_path / 'stage1')
	stage1_config['model']['encoder_depth'] = 2
	stage1_config['train'].update(
		{
			'epochs': 2,
			'samples_per_epoch': 1,
			'lr': 1.0e-3,
		},
	)
	stage1_checkpoint = run_mae_pretraining(stage1_config)
	stage1_payload = load_checkpoint(stage1_checkpoint, map_location='cpu')
	assert stage1_payload['epoch'] == 2
	assert stage1_payload['global_step'] == 2

	continuation_config = deepcopy(stage1_config)
	continuation_output = tmp_path / 'stage2' / 'artifacts' / 'run'
	continuation_config['paths']['artifact_root'] = str(
		tmp_path / 'stage2' / 'artifacts'
	)
	continuation_config['paths']['output_root'] = str(continuation_output)
	continuation_config['train'].update(
		{
			'epochs': 1,
			'max_steps': 1,
			'lr': 5.0e-3,
		},
	)
	continuation_config['continuation'] = {
		'init_checkpoint': str(stage1_checkpoint.resolve()),
		'unfreeze_top_blocks': 1,
	}

	continuation_checkpoint = run_mae_pretraining(continuation_config)
	continuation_payload = load_checkpoint(
		continuation_checkpoint,
		map_location='cpu',
	)

	assert continuation_payload['epoch'] == 1
	assert continuation_payload['global_step'] == 1
	assert continuation_payload['training_state']['stage'] == 'train_amp_mae'
	assert continuation_payload['training_state']['checkpoint_kind'] == 'epoch'
	assert continuation_payload['config']['continuation'] == (
		continuation_config['continuation']
	)
	assert continuation_payload['continuation_lineage'] == {
		'schema_version': 1,
		'init_checkpoint': str(stage1_checkpoint.resolve()),
		'init_checkpoint_sha256': file_sha256(stage1_checkpoint),
		'resume_count': 0,
	}
	optimizer_state = continuation_payload['optimizer_state_dict']
	assert len(optimizer_state['param_groups']) == 1
	assert optimizer_state['param_groups'][0]['lr'] == pytest.approx(5.0e-3)
	assert {_optimizer_step(state) for state in optimizer_state['state'].values()} == {
		1
	}
	assert {
		_optimizer_step(state)
		for state in stage1_payload['optimizer_state_dict']['state'].values()
	} == {2}

	expected_model = _model_from_config(continuation_payload['config']['model'])
	expected_trainable = configure_mae_continuation_trainability(
		expected_model,
		unfreeze_top_blocks=1,
	)
	assert len(optimizer_state['param_groups'][0]['params']) == len(
		expected_trainable
	)

	source_state = stage1_payload['model_state_dict']
	continued_state = continuation_payload['model_state_dict']
	_assert_prefix_unchanged(source_state, continued_state, 'patch_projection.')
	_assert_prefix_unchanged(source_state, continued_state, 'encoder.layers.0.')
	_assert_prefix_changed(source_state, continued_state, 'encoder.layers.1.')
	assert any(
		not torch.equal(source_state[name], continued_state[name])
		for name in source_state
		if name == 'mask_token'
		or name.startswith(
			('encoder_to_decoder.', 'decoder.', 'prediction_head.'),
		)
	)
	assert continuation_payload['metrics']
	assert all(
		math.isfinite(float(value))
		for value in continuation_payload['metrics'].values()
	)

	mismatched_config = deepcopy(continuation_config)
	mismatched_config['train'].update({'epochs': 2, 'max_steps': 2})
	mismatched_config['continuation']['init_checkpoint'] = str(
		tmp_path / 'different-source.pt'
	)
	with pytest.raises(ValueError, match=r'continuation\.init_checkpoint'):
		run_mae_pretraining(
			mismatched_config,
			resume=continuation_checkpoint,
		)

	with pytest.raises(ValueError, match='config is incompatible'):
		run_mae_pretraining(
			continuation_config,
			resume=stage1_checkpoint,
		)

	def fail_if_source_weights_are_loaded(*_args: object, **_kwargs: object) -> None:
		raise AssertionError('Stage 1 source weights must not load during resume')

	monkeypatch.setattr(
		mae_training,
		'load_mae_continuation_weights',
		fail_if_source_weights_are_loaded,
	)
	resume_config = deepcopy(continuation_config)
	resume_config['train'].update({'epochs': 2, 'max_steps': 2})
	resumed_checkpoint = run_mae_pretraining(
		resume_config,
		resume=continuation_checkpoint,
	)
	resumed_payload = load_checkpoint(resumed_checkpoint, map_location='cpu')
	assert resumed_payload['epoch'] == 2
	assert resumed_payload['global_step'] == 2
	assert resumed_payload['config']['continuation'] == (
		continuation_config['continuation']
	)
	assert resumed_payload['continuation_lineage'] == {
		'schema_version': 1,
		'init_checkpoint': str(stage1_checkpoint.resolve()),
		'init_checkpoint_sha256': file_sha256(stage1_checkpoint),
		'resume_count': 1,
	}


def _optimizer_step(state: dict[str, object]) -> int:
	value = state['step']
	return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _model_from_config(config: dict[str, object]) -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		in_channels=int(config['in_channels']),
		out_channels=int(config['out_channels']),
		patch_size_xyz=tuple(config['patch_size']),
		encoder_dim=int(config['encoder_dim']),
		encoder_depth=int(config['encoder_depth']),
		encoder_heads=int(config['encoder_heads']),
		decoder_dim=int(config['decoder_dim']),
		decoder_depth=int(config['decoder_depth']),
		decoder_heads=int(config['decoder_heads']),
	)


def _assert_prefix_unchanged(
	source: dict[str, torch.Tensor],
	continued: dict[str, torch.Tensor],
	prefix: str,
) -> None:
	names = [name for name in source if name.startswith(prefix)]
	assert names
	assert all(torch.equal(source[name], continued[name]) for name in names)


def _assert_prefix_changed(
	source: dict[str, torch.Tensor],
	continued: dict[str, torch.Tensor],
	prefix: str,
) -> None:
	names = [name for name in source if name.startswith(prefix)]
	assert names
	assert any(not torch.equal(source[name], continued[name]) for name in names)
