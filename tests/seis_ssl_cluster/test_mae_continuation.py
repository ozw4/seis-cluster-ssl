from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.mae_continuation import (
	configure_mae_continuation_trainability,
	load_mae_continuation_weights,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_loads_strict_model_weights_without_optimizer_state(tmp_path: Path) -> None:
	source = _model()
	with torch.no_grad():
		for index, parameter in enumerate(source.parameters(), start=1):
			parameter.fill_(index / 100.0)
	target = _model()
	checkpoint_path = tmp_path / 'latest.pt'
	torch.save(_checkpoint_payload(source), checkpoint_path)

	source_sha256 = load_mae_continuation_weights(
		target,
		checkpoint_path,
		expected_model_config=_model_config(),
	)

	assert source_sha256 == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
	assert all(
		torch.equal(target_value, source_value)
		for target_value, source_value in zip(
			target.state_dict().values(),
			source.state_dict().values(),
			strict=True,
		)
	)


def test_rejects_missing_checkpoint_file(tmp_path: Path) -> None:
	with pytest.raises(FileNotFoundError, match='checkpoint file does not exist'):
		load_mae_continuation_weights(
			_model(),
			tmp_path / 'missing.pt',
			expected_model_config=_model_config(),
		)


@pytest.mark.parametrize(
	('field', 'value', 'match'),
	[
		('model_state_dict', [], 'model_state_dict must be a mapping'),
		('config', [], 'config must be a mapping'),
		('training_state', [], 'training_state must be a mapping'),
	],
)
def test_rejects_non_mapping_checkpoint_fields(
	tmp_path: Path,
	field: str,
	value: object,
	match: str,
) -> None:
	payload = _checkpoint_payload(_model())
	payload[field] = value
	checkpoint_path = tmp_path / f'{field}.pt'
	torch.save(payload, checkpoint_path)

	with pytest.raises(TypeError, match=match):
		load_mae_continuation_weights(
			_model(),
			checkpoint_path,
			expected_model_config=_model_config(),
		)


def test_rejects_non_mapping_checkpoint_payload(tmp_path: Path) -> None:
	checkpoint_path = tmp_path / 'payload.pt'
	torch.save([], checkpoint_path)

	with pytest.raises(TypeError, match='payload must be a mapping'):
		load_mae_continuation_weights(
			_model(),
			checkpoint_path,
			expected_model_config=_model_config(),
		)


def test_rejects_source_stage_mismatch(tmp_path: Path) -> None:
	checkpoint_path = tmp_path / 'foreign-stage.pt'
	torch.save(
		_checkpoint_payload(_model(), stage='train_barlow_twins'),
		checkpoint_path,
	)

	with pytest.raises(ValueError, match=r'training_state\.stage'):
		load_mae_continuation_weights(
			_model(),
			checkpoint_path,
			expected_model_config=_model_config(),
		)


def test_rejects_step_checkpoint(tmp_path: Path) -> None:
	checkpoint_path = tmp_path / 'step.pt'
	torch.save(
		_checkpoint_payload(_model(), checkpoint_kind='step'),
		checkpoint_path,
	)

	with pytest.raises(ValueError, match=r'training_state\.checkpoint_kind'):
		load_mae_continuation_weights(
			_model(),
			checkpoint_path,
			expected_model_config=_model_config(),
		)


def test_rejects_resolved_model_config_mismatch(tmp_path: Path) -> None:
	checkpoint_path = tmp_path / 'model-config.pt'
	torch.save(_checkpoint_payload(_model()), checkpoint_path)
	mismatched_config = {**_model_config(), 'encoder_depth': 3}

	with pytest.raises(ValueError, match=r'config\.model does not match'):
		load_mae_continuation_weights(
			_model(),
			checkpoint_path,
			expected_model_config=mismatched_config,
		)


def test_rejects_strict_model_geometry_mismatch(tmp_path: Path) -> None:
	source = _model()
	target = _model(encoder_depth=3)
	target_config = _model_config(encoder_depth=3)
	checkpoint_path = tmp_path / 'geometry.pt'
	payload = _checkpoint_payload(source)
	payload['config'] = {'model': target_config}
	torch.save(payload, checkpoint_path)

	with pytest.raises(ValueError, match='model geometry/state mismatch'):
		load_mae_continuation_weights(
			target,
			checkpoint_path,
			expected_model_config=target_config,
		)


def test_rejects_nonfinite_model_state(tmp_path: Path) -> None:
	payload = _checkpoint_payload(_model())
	model_state = payload['model_state_dict']
	assert isinstance(model_state, dict)
	first_name = next(iter(model_state))
	model_state[first_name].reshape(-1)[0] = torch.nan
	checkpoint_path = tmp_path / 'nonfinite.pt'
	torch.save(payload, checkpoint_path)

	with pytest.raises(ValueError, match='contains non-finite values'):
		load_mae_continuation_weights(
			_model(),
			checkpoint_path,
			expected_model_config=_model_config(),
		)


def test_configures_exact_mae_continuation_trainability() -> None:
	model = _model()

	trainable_parameters = configure_mae_continuation_trainability(
		model,
		unfreeze_top_blocks=1,
	)

	assert all(
		not parameter.requires_grad
		for parameter in model.patch_projection.parameters()
	)
	assert all(
		not parameter.requires_grad
		for block in model.encoder.layers[:-1]
		for parameter in block.parameters()
	)
	assert all(
		parameter.requires_grad
		for parameter in model.encoder.layers[-1].parameters()
	)
	assert model.mask_token.requires_grad
	assert all(
		parameter.requires_grad
		for parameter in model.encoder_to_decoder.parameters()
	)
	assert all(parameter.requires_grad for parameter in model.decoder.parameters())
	assert all(
		parameter.requires_grad
		for parameter in model.prediction_head.parameters()
	)

	returned_ids = tuple(id(parameter) for parameter in trainable_parameters)
	requires_grad_ids = {
		id(parameter)
		for parameter in model.parameters()
		if parameter.requires_grad
	}
	assert set(returned_ids) == requires_grad_ids
	assert len(returned_ids) == len(set(returned_ids))


def _checkpoint_payload(
	model: AmplitudeMAE3D,
	*,
	stage: str = 'train_amp_mae',
	checkpoint_kind: str = 'epoch',
) -> dict[str, object]:
	return {
		'model_state_dict': dict(model.state_dict()),
		'config': {'model': _model_config()},
		'training_state': {
			'stage': stage,
			'checkpoint_kind': checkpoint_kind,
		},
	}


def _model(*, encoder_depth: int = 2) -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		**_model_config(encoder_depth=encoder_depth),
	)


def _model_config(*, encoder_depth: int = 2) -> dict[str, object]:
	return {
		'in_channels': 1,
		'out_channels': 1,
		'patch_size_xyz': (2, 2, 2),
		'encoder_dim': 12,
		'encoder_depth': encoder_depth,
		'encoder_heads': 3,
		'decoder_dim': 12,
		'decoder_depth': 1,
		'decoder_heads': 3,
	}
