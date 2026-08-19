from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.config.schema import STAGE_BARLOW_TWINS_TRAINING
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	CHECKPOINT_KIND,
	PRETRAINING_METHOD,
	TRAINED_PARAMETER_PREFIXES,
)
from seis_ssl_cluster.training.barlow_twins_continuation import (
	configure_barlow_twins_continuation_trainability,
	load_barlow_twins_continuation_weights,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


def test_loads_backbone_and_projector_weights_without_optimizer_state(
	tmp_path: Path,
) -> None:
	source = _model()
	with torch.no_grad():
		for index, parameter in enumerate(source.parameters(), start=1):
			parameter.fill_(index / 100.0)
	target = _model()
	payload = _checkpoint_payload(source)
	assert 'optimizer_state_dict' not in payload
	checkpoint_path = tmp_path / 'latest.pt'
	torch.save(payload, checkpoint_path)

	_load(target, checkpoint_path)

	_assert_state_equal(target.backbone.state_dict(), source.backbone.state_dict())
	_assert_state_equal(target.projector.state_dict(), source.projector.state_dict())


def test_rejects_missing_checkpoint_file(tmp_path: Path) -> None:
	with pytest.raises(FileNotFoundError, match='checkpoint file does not exist'):
		_load(_model(), tmp_path / 'missing.pt')


def test_rejects_non_mapping_checkpoint_payload(tmp_path: Path) -> None:
	checkpoint_path = tmp_path / 'payload.pt'
	torch.save([], checkpoint_path)

	with pytest.raises(TypeError, match='payload must be a mapping'):
		_load(_model(), checkpoint_path)


@pytest.mark.parametrize(
	('field', 'value', 'match'),
	[
		('model_state_dict', [], 'model_state_dict must be a mapping'),
		('projector_state_dict', [], 'projector_state_dict must be a mapping'),
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
		_load(_model(), checkpoint_path)


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('pretraining_method', 'amp_mae3d'),
		('checkpoint_kind', 'mae_pretraining'),
		('trained_parameter_prefixes', ['encoder.']),
	],
)
def test_rejects_source_checkpoint_identity_mismatch(
	tmp_path: Path,
	field: str,
	value: object,
) -> None:
	payload = _checkpoint_payload(_model())
	payload[field] = value
	checkpoint_path = tmp_path / f'{field}.pt'
	torch.save(payload, checkpoint_path)

	with pytest.raises(ValueError, match=field):
		_load(_model(), checkpoint_path)


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('stage', 'train_amp_mae'),
		('resume_boundary', 'step'),
		('completed_epoch', False),
	],
)
def test_rejects_incomplete_or_foreign_epoch_checkpoint(
	tmp_path: Path,
	field: str,
	value: object,
) -> None:
	payload = _checkpoint_payload(_model())
	training_state = payload['training_state']
	assert isinstance(training_state, dict)
	training_state[field] = value
	checkpoint_path = tmp_path / f'{field}.pt'
	torch.save(payload, checkpoint_path)

	with pytest.raises(ValueError, match=field):
		_load(_model(), checkpoint_path)


def test_rejects_resolved_model_config_mismatch(tmp_path: Path) -> None:
	payload = _checkpoint_payload(_model())
	checkpoint_path = tmp_path / 'model-config.pt'
	torch.save(payload, checkpoint_path)
	mismatched_config = {**_model_config(), 'encoder_depth': 3}

	with pytest.raises(ValueError, match=r'config\.model does not match'):
		load_barlow_twins_continuation_weights(
			_model(),
			checkpoint_path,
			expected_model_config=mismatched_config,
			expected_barlow_twins_config=_barlow_twins_config(),
		)


def test_rejects_resolved_barlow_twins_config_mismatch(tmp_path: Path) -> None:
	payload = _checkpoint_payload(_model())
	checkpoint_path = tmp_path / 'barlow-config.pt'
	torch.save(payload, checkpoint_path)
	mismatched_config = {**_barlow_twins_config(), 'projector_dim': 16}

	with pytest.raises(ValueError, match=r'config\.barlow_twins does not match'):
		load_barlow_twins_continuation_weights(
			_model(),
			checkpoint_path,
			expected_model_config=_model_config(),
			expected_barlow_twins_config=mismatched_config,
		)


@pytest.mark.parametrize('state_field', ['model_state_dict', 'projector_state_dict'])
@pytest.mark.parametrize('tensor_kind', ['floating', 'complex'])
def test_rejects_nonfinite_backbone_or_projector_state(
	tmp_path: Path,
	state_field: str,
	tensor_kind: str,
) -> None:
	payload = _checkpoint_payload(_model())
	state = payload[state_field]
	assert isinstance(state, dict)
	first_name = next(
		name
		for name, value in state.items()
		if isinstance(value, torch.Tensor) and value.is_floating_point()
	)
	tensor = state[first_name]
	assert isinstance(tensor, torch.Tensor)
	if tensor_kind == 'complex':
		tensor = tensor.to(torch.complex64)
		state[first_name] = tensor
	tensor.reshape(-1)[0] = torch.nan
	checkpoint_path = tmp_path / f'nonfinite-{tensor_kind}-{state_field}.pt'
	torch.save(payload, checkpoint_path)

	with pytest.raises(ValueError, match=rf'{state_field}.*non-finite'):
		_load(_model(), checkpoint_path)


@pytest.mark.parametrize('state_field', ['model_state_dict', 'projector_state_dict'])
@pytest.mark.parametrize('invalid_part', ['key', 'value'])
def test_rejects_invalid_backbone_or_projector_state_entries(
	tmp_path: Path,
	state_field: str,
	invalid_part: str,
) -> None:
	payload = _checkpoint_payload(_model())
	payload[state_field] = (
		{1: torch.zeros(1)}
		if invalid_part == 'key'
		else {'invalid': object()}
	)
	checkpoint_path = tmp_path / f'{state_field}-{invalid_part}.pt'
	torch.save(payload, checkpoint_path)

	with pytest.raises(TypeError, match=state_field):
		_load(_model(), checkpoint_path)


def test_rejects_strict_backbone_geometry_mismatch(tmp_path: Path) -> None:
	source = _model(encoder_depth=2)
	target = _model(encoder_depth=3)
	target_config = _model_config(encoder_depth=3)
	payload = _checkpoint_payload(source)
	config = payload['config']
	assert isinstance(config, dict)
	config['model'] = target_config
	checkpoint_path = tmp_path / 'backbone-geometry.pt'
	torch.save(payload, checkpoint_path)

	with pytest.raises(ValueError, match='backbone geometry/state mismatch'):
		load_barlow_twins_continuation_weights(
			target,
			checkpoint_path,
			expected_model_config=target_config,
			expected_barlow_twins_config=_barlow_twins_config(),
		)


def test_rejects_strict_projector_geometry_mismatch(tmp_path: Path) -> None:
	source = _model(projector_dim=8)
	target = _model(projector_dim=16)
	target_barlow_config = _barlow_twins_config(projector_dim=16)
	payload = _checkpoint_payload(source)
	config = payload['config']
	assert isinstance(config, dict)
	config['barlow_twins'] = target_barlow_config
	checkpoint_path = tmp_path / 'projector-geometry.pt'
	torch.save(payload, checkpoint_path)

	with pytest.raises(ValueError, match='projector geometry/state mismatch'):
		load_barlow_twins_continuation_weights(
			target,
			checkpoint_path,
			expected_model_config=_model_config(),
			expected_barlow_twins_config=target_barlow_config,
		)


def test_configures_exact_barlow_twins_continuation_trainability() -> None:
	model = _model(encoder_depth=3)

	trainable_parameters = configure_barlow_twins_continuation_trainability(
		model,
		unfreeze_top_blocks=1,
	)

	assert all(
		not parameter.requires_grad
		for parameter in model.backbone.patch_projection.parameters()
	)
	assert all(
		not parameter.requires_grad
		for block in model.backbone.encoder.layers[:-1]
		for parameter in block.parameters()
	)
	assert all(
		parameter.requires_grad
		for parameter in model.backbone.encoder.layers[-1].parameters()
	)
	assert all(parameter.requires_grad for parameter in model.projector.parameters())
	assert not model.backbone.mask_token.requires_grad
	assert all(
		not parameter.requires_grad
		for parameter in model.backbone.encoder_to_decoder.parameters()
	)
	assert all(
		not parameter.requires_grad
		for parameter in model.backbone.decoder.parameters()
	)
	assert all(
		not parameter.requires_grad
		for parameter in model.backbone.prediction_head.parameters()
	)

	returned_ids = tuple(id(parameter) for parameter in trainable_parameters)
	requires_grad_ids = {
		id(parameter)
		for parameter in model.parameters()
		if parameter.requires_grad
	}
	assert set(returned_ids) == requires_grad_ids
	assert len(returned_ids) == len(set(returned_ids))


def test_rejects_empty_trainable_parameter_set() -> None:
	model = _model()
	model.projector = torch.nn.Identity()

	with pytest.raises(RuntimeError, match='no trainable parameters'):
		configure_barlow_twins_continuation_trainability(
			model,
			unfreeze_top_blocks=0,
		)


def _load(model: BarlowTwins3D, checkpoint_path: Path) -> None:
	load_barlow_twins_continuation_weights(
		model,
		checkpoint_path,
		expected_model_config=_model_config(
			encoder_depth=model.backbone.encoder.depth,
		),
		expected_barlow_twins_config=_barlow_twins_config(
			projector_dim=model.projector_dim,
		),
	)


def _checkpoint_payload(model: BarlowTwins3D) -> dict[str, object]:
	return {
		'model_state_dict': dict(model.backbone.state_dict()),
		'projector_state_dict': dict(model.projector.state_dict()),
		'config': {
			'model': _model_config(encoder_depth=model.backbone.encoder.depth),
			'barlow_twins': _barlow_twins_config(
				projector_dim=model.projector_dim,
			),
		},
		'pretraining_method': PRETRAINING_METHOD,
		'checkpoint_kind': CHECKPOINT_KIND,
		'trained_parameter_prefixes': list(TRAINED_PARAMETER_PREFIXES),
		'training_state': {
			'stage': STAGE_BARLOW_TWINS_TRAINING,
			'resume_boundary': 'epoch',
			'completed_epoch': True,
		},
	}


def _model(
	*,
	encoder_depth: int = 2,
	projector_dim: int = 8,
) -> BarlowTwins3D:
	return BarlowTwins3D(
		AmplitudeMAE3D(
			in_channels=1,
			out_channels=1,
			patch_size_xyz=(2, 2, 2),
			encoder_dim=12,
			encoder_depth=encoder_depth,
			encoder_heads=3,
			decoder_dim=12,
			decoder_depth=1,
			decoder_heads=3,
		),
		projector_dim=projector_dim,
	)


def _model_config(*, encoder_depth: int = 2) -> dict[str, object]:
	return {
		'name': 'amp_mae3d',
		'in_channels': 1,
		'out_channels': 1,
		'patch_size': [2, 2, 2],
		'encoder_dim': 12,
		'encoder_depth': encoder_depth,
		'encoder_heads': 3,
		'decoder_dim': 12,
		'decoder_depth': 1,
		'decoder_heads': 3,
	}


def _barlow_twins_config(*, projector_dim: int = 8) -> dict[str, object]:
	return {
		'projector_dim': projector_dim,
		'redundancy_weight': 0.005,
		'normalization_eps': 1.0e-4,
	}


def _assert_state_equal(
	actual: Mapping[str, torch.Tensor],
	expected: Mapping[str, torch.Tensor],
) -> None:
	assert set(actual) == set(expected)
	assert all(torch.equal(actual[key], expected[key]) for key in expected)
