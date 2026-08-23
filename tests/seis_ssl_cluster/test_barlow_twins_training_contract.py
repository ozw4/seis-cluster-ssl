from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

import seis_ssl_cluster.data.amplitude_dataset as amplitude_dataset_module
import seis_ssl_cluster.training.barlow_twins as barlow_twins_module
from seis_ssl_cluster.config import resolve_barlow_twins_training_config
from seis_ssl_cluster.config.schema import (
	BARLOW_TWINS_PRETRAINING_METHOD,
	LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
)
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.models.amplitude_encoder_factory import (
	build_model_from_checkpoint_payload,
)
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D, BarlowTwinsLoss
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.barlow_twins import (
	run_barlow_twins_pretraining,
	train_barlow_twins_one_epoch,
)
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	load_barlow_twins_checkpoint,
	restore_barlow_twins_checkpoint,
)

DIAGNOSTIC_METRICS = {
	'projection_std_mean',
	'projection_std_min',
	'projection_norm_mean',
	'cross_correlation_diag_mean',
	'cross_correlation_offdiag_rms',
	'weighted_off_diag',
	'gradient_norm',
	'learning_rate',
	'step_time_seconds',
	'peak_cuda_memory_mib',
}

D4_AUGMENTATION_METRICS = {
	'd4_same_transform_fraction',
	'd4_reflection_fraction_a',
	'd4_reflection_fraction_b',
	'd4_nonzero_rotation_fraction_a',
	'd4_nonzero_rotation_fraction_b',
	'trace_drop_fraction_a',
	'trace_drop_fraction_b',
}


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


def test_cli_dry_run_displays_local_method_without_creating_artifacts(
	tmp_path: Path,
) -> None:
	raw_config = _tiny_local_config(
		tmp_path,
		output_name='local-dry-run-output',
	)
	config_path = tmp_path / 'local-barlow.yaml'
	config_path.write_text(
		yaml.safe_dump(raw_config, sort_keys=False),
		encoding='utf-8',
	)
	output_root = tmp_path / 'artifacts' / 'local-dry-run-output'

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/train_amp_barlow_twins.py',
			'--config',
			str(config_path),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert (
		'barlow_twins.method: local_barlow_twins_3d' in result.stdout
	)
	assert 'barlow_twins.local_pairs_per_crop: 4' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout
	assert not output_root.exists()


def test_checkpoint_contract_round_trip_and_epoch_resume(
	tmp_path: Path,
	capsys: pytest.CaptureFixture[str],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	build_mask = Mock(side_effect=AssertionError('MAE mask should not be built'))
	monkeypatch.setattr(
		amplitude_dataset_module,
		'build_spatial_masking_plan',
		build_mask,
	)
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
	assert set(payload['metrics']) >= DIAGNOSTIC_METRICS
	assert all(np.isfinite(payload['metrics'][key]) for key in DIAGNOSTIC_METRICS)
	assert 'projection_std_mean=' in capsys.readouterr().out

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
	build_mask.assert_not_called()
	history = json.loads(
		(resumed_path.parent / 'history.json').read_text(encoding='utf-8')
	)
	assert [row['global_step'] for row in history] == [1, 2]
	assert all(set(row) >= DIAGNOSTIC_METRICS for row in history)


def test_local_checkpoint_contract_round_trip_and_epoch_resume(
	tmp_path: Path,
) -> None:
	config = resolve_barlow_twins_training_config(
		_tiny_local_config(tmp_path, output_name='local-run')
	)
	first_path = run_barlow_twins_pretraining(config)
	payload = load_barlow_twins_checkpoint(first_path, map_location='cpu')

	assert first_path.name == 'latest.pt'
	assert (first_path.parent / 'best.pt').is_file()
	assert (first_path.parent / 'history.json').is_file()
	resolved_path = first_path.parent / 'resolved_config.json'
	assert resolved_path.is_file()
	assert payload['pretraining_method'] == LOCAL_BARLOW_TWINS_PRETRAINING_METHOD
	assert payload['checkpoint_kind'] == 'barlow_twins_pretraining'
	assert payload['config']['barlow_twins'] == config['barlow_twins']
	assert payload['config']['barlow_twins']['local_pairs_per_crop'] == 4
	assert json.loads(resolved_path.read_text(encoding='utf-8')) == config
	loaded_encoder = build_model_from_checkpoint_payload(payload)
	for key, expected in payload['model_state_dict'].items():
		assert torch.equal(loaded_encoder.state_dict()[key], expected)

	resume_config = resolve_barlow_twins_training_config(
		_tiny_local_config(
			tmp_path,
			epochs=2,
			max_steps=2,
			output_name='local-run',
		)
	)
	resumed_path = run_barlow_twins_pretraining(
		resume_config,
		resume=first_path,
	)
	resumed = load_barlow_twins_checkpoint(resumed_path, map_location='cpu')
	assert resumed['epoch'] == 2
	assert resumed['global_step'] == 2
	assert resumed['pretraining_method'] == LOCAL_BARLOW_TWINS_PRETRAINING_METHOD


def test_d4_trace_drop_one_step_uses_policy_dataset_and_saves_config(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	original_dataset = (
		barlow_twins_module.LocalBarlowTwinsD4TraceDropPretrainDataset
	)
	dataset_calls: list[dict[str, object]] = []

	def record_dataset(
		base_dataset: object,
		**kwargs: object,
	) -> object:
		dataset_calls.append(kwargs)
		return original_dataset(base_dataset, **kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(
		barlow_twins_module,
		'LocalBarlowTwinsD4TraceDropPretrainDataset',
		record_dataset,
	)
	config = resolve_barlow_twins_training_config(
		_tiny_d4_config(tmp_path, output_name='d4-one-step')
	)

	checkpoint_path = run_barlow_twins_pretraining(config)
	payload = load_barlow_twins_checkpoint(checkpoint_path, map_location='cpu')

	assert dataset_calls == [
		{
			'local_pairs_per_crop': 4,
			'reflection_probability': 0.5,
			'trace_drop_probability': 0.02,
		}
	]
	assert payload['config']['augmentations'] == {
		'policy': XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
		'reflection_probability': 0.5,
		'trace_drop_probability': 0.02,
	}
	assert payload['global_step'] == 1
	assert all(np.isfinite(value) for value in payload['metrics'].values())
	assert np.isfinite(payload['metrics']['gradient_norm'])
	assert set(payload['metrics']) >= D4_AUGMENTATION_METRICS
	history = json.loads(
		(checkpoint_path.parent / 'history.json').read_text(encoding='utf-8')
	)
	assert set(history[0]) >= D4_AUGMENTATION_METRICS
	for key in D4_AUGMENTATION_METRICS:
		assert history[0][key] == payload['metrics'][key]


def test_d4_resume_is_strict_about_augmentation_identity(tmp_path: Path) -> None:
	flip_path = run_barlow_twins_pretraining(
		resolve_barlow_twins_training_config(
			_tiny_local_config(tmp_path, output_name='flip-source')
		)
	)
	d4_path = run_barlow_twins_pretraining(
		resolve_barlow_twins_training_config(
			_tiny_d4_config(tmp_path, output_name='d4-source')
		)
	)

	d4_resume = resolve_barlow_twins_training_config(
		_tiny_d4_config(
			tmp_path,
			epochs=2,
			max_steps=2,
			output_name='d4-source',
		)
	)
	resumed_path = run_barlow_twins_pretraining(d4_resume, resume=d4_path)
	resumed = load_barlow_twins_checkpoint(resumed_path, map_location='cpu')
	assert resumed['epoch'] == 2
	assert resumed['global_step'] == 2

	for current_config, source_path in (
		(
			_tiny_d4_config(
				tmp_path,
				epochs=2,
				max_steps=2,
				output_name='d4-from-flip',
			),
			flip_path,
		),
		(
			_tiny_local_config(
				tmp_path,
				epochs=2,
				max_steps=2,
				output_name='flip-from-d4',
			),
			d4_path,
		),
	):
		resolved = resolve_barlow_twins_training_config(current_config)
		with pytest.raises(ValueError, match='augmentations'):
			run_barlow_twins_pretraining(resolved, resume=source_path)


def test_flip_local_checkpoint_initializes_d4_continuation_weights_only(
	tmp_path: Path,
) -> None:
	source_path = run_barlow_twins_pretraining(
		resolve_barlow_twins_training_config(
			_tiny_local_config(
				tmp_path,
				output_name='flip-local-source',
				encoder_depth=2,
			)
		)
	)
	source = load_barlow_twins_checkpoint(source_path, map_location='cpu')
	target_raw = _tiny_d4_config(
		tmp_path,
		output_name='d4-continuation',
		encoder_depth=2,
	)
	target_raw['continuation'] = {
		'init_checkpoint': str(source_path),
		'unfreeze_top_blocks': 1,
	}
	target_config = resolve_barlow_twins_training_config(target_raw)

	target_path = run_barlow_twins_pretraining(target_config)
	target = load_barlow_twins_checkpoint(target_path, map_location='cpu')
	source_backbone = _clone_tensor_state(source, 'model_state_dict')
	target_backbone = _clone_tensor_state(target, 'model_state_dict')
	source_projector = _clone_tensor_state(source, 'projector_state_dict')
	target_projector = _clone_tensor_state(target, 'projector_state_dict')

	_assert_state_prefix_unchanged(
		source_backbone,
		target_backbone,
		'patch_projection.',
	)
	_assert_state_prefix_unchanged(
		source_backbone,
		target_backbone,
		'encoder.layers.0.',
	)
	assert _state_prefix_changed(
		source_backbone,
		target_backbone,
		'encoder.layers.1.',
	)
	projector_parameters = {
		name for name, _ in _backbone_wrapper().projector.named_parameters()
	}
	assert any(
		not torch.equal(source_projector[name], target_projector[name])
		for name in projector_parameters
	)

	fresh_resume_raw = _tiny_d4_config(
		tmp_path,
		epochs=2,
		max_steps=2,
		output_name='source-as-resume',
		encoder_depth=2,
	)
	fresh_resume_raw['continuation'] = target_raw['continuation']
	with pytest.raises(ValueError, match=r'augmentations|continuation'):
		run_barlow_twins_pretraining(
			resolve_barlow_twins_training_config(fresh_resume_raw),
			resume=source_path,
		)


def test_resume_rejects_cross_method_before_creating_output_and_accepts_legacy(
	tmp_path: Path,
) -> None:
	standard_path = run_barlow_twins_pretraining(
		resolve_barlow_twins_training_config(
			_tiny_config(tmp_path, output_name='standard-source')
		)
	)
	local_path = run_barlow_twins_pretraining(
		resolve_barlow_twins_training_config(
			_tiny_local_config(tmp_path, output_name='local-source')
		)
	)

	standard_target = tmp_path / 'artifacts' / 'standard-from-local'
	standard_config = resolve_barlow_twins_training_config(
		_tiny_config(
			tmp_path,
			epochs=2,
			max_steps=2,
			output_name='standard-from-local',
		)
	)
	with pytest.raises(ValueError, match='pretraining_method'):
		run_barlow_twins_pretraining(standard_config, resume=local_path)
	assert not standard_target.exists()

	local_target = tmp_path / 'artifacts' / 'local-from-standard'
	local_config = resolve_barlow_twins_training_config(
		_tiny_local_config(
			tmp_path,
			epochs=2,
			max_steps=2,
			output_name='local-from-standard',
		)
	)
	with pytest.raises(ValueError, match='pretraining_method'):
		run_barlow_twins_pretraining(local_config, resume=standard_path)
	assert not local_target.exists()

	explicit_standard = _tiny_config(
		tmp_path,
		epochs=2,
		max_steps=2,
		output_name='explicit-standard-resume',
	)
	barlow_twins = explicit_standard['barlow_twins']
	assert isinstance(barlow_twins, dict)
	barlow_twins['method'] = BARLOW_TWINS_PRETRAINING_METHOD
	explicit_path = run_barlow_twins_pretraining(
		resolve_barlow_twins_training_config(explicit_standard),
		resume=standard_path,
	)
	explicit_payload = load_barlow_twins_checkpoint(
		explicit_path,
		map_location='cpu',
	)
	assert explicit_payload['epoch'] == 2
	assert (
		explicit_payload['pretraining_method']
		== BARLOW_TWINS_PRETRAINING_METHOD
	)


def test_continuation_fresh_and_stage2_resume_contract(  # noqa: PLR0915
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	stage1_config = resolve_barlow_twins_training_config(
		_tiny_config(
			tmp_path,
			epochs=2,
			max_steps=2,
			output_name='stage1',
			encoder_depth=2,
		)
	)
	stage1_path = run_barlow_twins_pretraining(stage1_config)
	stage1 = load_barlow_twins_checkpoint(stage1_path, map_location='cpu')
	stage1_backbone = _clone_tensor_state(stage1, 'model_state_dict')
	stage1_projector = _clone_tensor_state(stage1, 'projector_state_dict')
	assert stage1['epoch'] == 2
	assert stage1['global_step'] == 2
	assert _optimizer_steps(stage1) == {2}

	stage2_config = resolve_barlow_twins_training_config(
		_continuation_config(
			tmp_path,
			source_checkpoint=stage1_path,
			output_name='stage2',
			training_steps=1,
			unfreeze_top_blocks=1,
		)
	)
	stage2_path = run_barlow_twins_pretraining(stage2_config)
	stage2 = load_barlow_twins_checkpoint(stage2_path, map_location='cpu')
	stage2_backbone = _clone_tensor_state(stage2, 'model_state_dict')
	stage2_projector = _clone_tensor_state(stage2, 'projector_state_dict')

	assert stage2['epoch'] == 1
	assert stage2['global_step'] == 1
	assert _optimizer_steps(stage2) == {1}
	assert stage2['config']['continuation'] == {
		'init_checkpoint': str(stage1_path),
		'unfreeze_top_blocks': 1,
	}
	assert all(np.isfinite(value) for value in stage2['metrics'].values())
	loaded_encoder = build_model_from_checkpoint_payload(stage2)
	loaded_encoder_state = loaded_encoder.state_dict()
	for name, expected in stage2['model_state_dict'].items():
		assert torch.equal(loaded_encoder_state[name], expected)
	assert set(stage2['projector_state_dict']).isdisjoint(loaded_encoder_state)
	assert [
		row['global_step']
		for row in json.loads(
			(stage2_path.parent / 'history.json').read_text(encoding='utf-8')
		)
	] == [1]
	_assert_state_prefix_unchanged(
		stage1_backbone,
		stage2_backbone,
		'patch_projection.',
	)
	_assert_state_prefix_unchanged(
		stage1_backbone,
		stage2_backbone,
		'encoder.layers.0.',
	)
	assert _state_prefix_changed(
		stage1_backbone,
		stage2_backbone,
		'encoder.layers.1.',
	)
	projector_parameter_names = {
		name for name, _parameter in _backbone_wrapper().projector.named_parameters()
	}
	assert any(
		not torch.equal(stage1_projector[name], stage2_projector[name])
		for name in projector_parameter_names
	)
	for prefix in (
		'mask_token',
		'encoder_to_decoder.',
		'decoder.',
		'prediction_head.',
	):
		_assert_state_prefix_unchanged(
			stage1_backbone,
			stage2_backbone,
			prefix,
		)

	def unexpected_source_load(*args: object, **kwargs: object) -> None:
		del args, kwargs
		raise AssertionError('Stage 1 source must not be loaded during Stage 2 resume')

	source_loader = barlow_twins_module.load_barlow_twins_continuation_weights
	monkeypatch.setattr(
		barlow_twins_module,
		'load_barlow_twins_continuation_weights',
		unexpected_source_load,
	)
	resume_config = resolve_barlow_twins_training_config(
		_continuation_config(
			tmp_path,
			source_checkpoint=stage1_path,
			output_name='stage2',
			training_steps=2,
			unfreeze_top_blocks=1,
		)
	)
	resumed_path = run_barlow_twins_pretraining(
		resume_config,
		resume=stage2_path,
	)
	resumed = load_barlow_twins_checkpoint(resumed_path, map_location='cpu')
	assert resumed['epoch'] == 2
	assert resumed['global_step'] == 2
	assert _optimizer_steps(resumed) == {2}
	assert [
		row['global_step']
		for row in json.loads(
			(resumed_path.parent / 'history.json').read_text(encoding='utf-8')
		)
	] == [1, 2]

	for name, source_checkpoint, unfreeze_top_blocks, match in (
		(
			'source-mismatch',
			'/different/stage1/latest.pt',
			1,
			'continuation',
		),
		('top-block-mismatch', str(stage1_path), 2, 'continuation'),
	):
		invalid_config = resolve_barlow_twins_training_config(
			_continuation_config(
				tmp_path,
				source_checkpoint=source_checkpoint,
				output_name=name,
				training_steps=3,
				unfreeze_top_blocks=unfreeze_top_blocks,
			)
		)
		invalid_output = Path(invalid_config['paths']['output_root'])  # type: ignore[index]
		with pytest.raises(ValueError, match=match):
			run_barlow_twins_pretraining(
				invalid_config,
				resume=resumed_path,
			)
		assert not invalid_output.exists()

	stage1_resume_config = resolve_barlow_twins_training_config(
		_continuation_config(
			tmp_path,
			source_checkpoint=stage1_path,
			output_name='stage1-as-stage2-resume',
			training_steps=3,
			unfreeze_top_blocks=1,
		)
	)
	stage1_resume_output = Path(  # type: ignore[arg-type]
		stage1_resume_config['paths']['output_root'],  # type: ignore[index]
	)
	with pytest.raises(ValueError, match='continuation'):
		run_barlow_twins_pretraining(
			stage1_resume_config,
			resume=stage1_path,
		)
	assert not stage1_resume_output.exists()

	missing_source_config = resolve_barlow_twins_training_config(
		_continuation_config(
			tmp_path,
			source_checkpoint=tmp_path / 'missing-source.pt',
			output_name='missing-source',
			training_steps=1,
			unfreeze_top_blocks=1,
		)
	)
	missing_source_output = Path(  # type: ignore[arg-type]
		missing_source_config['paths']['output_root'],  # type: ignore[index]
	)
	monkeypatch.setattr(
		barlow_twins_module,
		'load_barlow_twins_continuation_weights',
		source_loader,
	)
	with pytest.raises(FileNotFoundError, match='checkpoint file does not exist'):
		run_barlow_twins_pretraining(missing_source_config)
	assert not missing_source_output.exists()


def test_epoch_rejects_nonfinite_loss_before_optimizer_step() -> None:
	model = _backbone_wrapper()
	optimizer = torch.optim.SGD(model.pretraining_parameters(), lr=0.1)
	before = {
		name: parameter.detach().clone()
		for name, parameter in model.named_parameters()
	}

	with pytest.raises(FloatingPointError, match='non-finite Barlow Twins loss'):
		train_barlow_twins_one_epoch(
			model=model,
			loss_fn=_NonfiniteLoss(),  # type: ignore[arg-type]
			dataloader=[_barlow_batch()],  # type: ignore[arg-type]
			optimizer=optimizer,
			device=torch.device('cpu'),
			epoch=1,
			grad_clip_norm=1.0,
		)

	for name, parameter in model.named_parameters():
		torch.testing.assert_close(parameter, before[name])


def test_epoch_checks_and_records_preclip_gradient_norm(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _backbone_wrapper()
	model.backbone.patch_projection.requires_grad_(requires_grad=False)
	optimizer = torch.optim.SGD(model.pretraining_parameters(), lr=0.025)
	calls: list[tuple[float, bool]] = []
	clipped_parameter_ids: set[int] = set()

	def fake_clip_grad_norm_(
		parameters: object,
		max_norm: float,
		*,
		error_if_nonfinite: bool,
	) -> torch.Tensor:
		parameter_list = list(parameters)  # type: ignore[arg-type]
		clipped_parameter_ids.update(id(parameter) for parameter in parameter_list)
		calls.append((max_norm, error_if_nonfinite))
		return torch.tensor(0.75)

	monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', fake_clip_grad_norm_)
	state = train_barlow_twins_one_epoch(
		model=model,
		loss_fn=BarlowTwinsLoss(),
		dataloader=[_barlow_batch()],  # type: ignore[arg-type]
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		grad_clip_norm=1.5,
	)

	assert calls == [(1.5, True)]
	assert clipped_parameter_ids == {
		id(parameter)
		for parameter in model.pretraining_parameters()
		if parameter.requires_grad
	}
	assert state.metrics['gradient_norm'] == pytest.approx(0.75)
	assert state.metrics['learning_rate'] == pytest.approx(0.025)
	assert state.metrics['step_time_seconds'] > 0.0
	assert state.metrics['peak_cuda_memory_mib'] == 0.0
	assert set(state.metrics) >= DIAGNOSTIC_METRICS


def test_epoch_local_method_projects_batch_times_pair_rows(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _backbone_wrapper()
	optimizer = torch.optim.SGD(model.pretraining_parameters(), lr=0.025)
	original_forward_local = model.forward_local
	projection_rows: list[int] = []

	def recording_forward_local(
		view_a: torch.Tensor,
		view_b: torch.Tensor,
		**kwargs: torch.Tensor,
	) -> dict[str, torch.Tensor]:
		outputs = original_forward_local(view_a, view_b, **kwargs)
		projection_rows.append(int(outputs['z_a'].shape[0]))
		return outputs

	monkeypatch.setattr(model, 'forward_local', recording_forward_local)
	state = train_barlow_twins_one_epoch(
		model=model,
		loss_fn=BarlowTwinsLoss(),
		dataloader=[_local_barlow_batch(local_pairs_per_crop=3)],  # type: ignore[arg-type]
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		grad_clip_norm=1.0,
		method=LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	)

	assert projection_rows == [2 * 3]
	assert state.global_step == 1
	assert D4_AUGMENTATION_METRICS.isdisjoint(state.metrics)


def test_epoch_records_weighted_d4_augmentation_realizations() -> None:
	model = _backbone_wrapper()
	optimizer = torch.optim.SGD(model.pretraining_parameters(), lr=0.025)
	batch = _d4_local_barlow_batch(local_pairs_per_crop=3)
	batch['valid_mask_a'][1, 0, 0, :] = False
	batch['valid_mask_b'][1, :2, :2, :] = False
	state = train_barlow_twins_one_epoch(
		model=model,
		loss_fn=BarlowTwinsLoss(),
		dataloader=[  # type: ignore[arg-type]
			batch,
			_d4_local_barlow_batch(local_pairs_per_crop=3),
		],
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		grad_clip_norm=1.0,
		method=LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
		augmentation_policy=XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
	)

	assert state.metrics['d4_same_transform_fraction'] == pytest.approx(0.5)
	assert state.metrics['d4_reflection_fraction_a'] == pytest.approx(0.5)
	assert state.metrics['d4_reflection_fraction_b'] == pytest.approx(0.5)
	assert state.metrics['d4_nonzero_rotation_fraction_a'] == pytest.approx(0.5)
	assert state.metrics['d4_nonzero_rotation_fraction_b'] == pytest.approx(0.5)
	assert state.metrics['trace_drop_fraction_a'] == pytest.approx(8 / 63)
	assert state.metrics['trace_drop_fraction_b'] == pytest.approx(14 / 60)


def test_epoch_rejects_nonfinite_gradient_before_optimizer_step() -> None:
	model = _backbone_wrapper()
	optimizer = torch.optim.SGD(model.pretraining_parameters(), lr=0.1)
	before = model.backbone.patch_projection.weight.detach().clone()

	with pytest.raises(RuntimeError, match='non-finite'):
		train_barlow_twins_one_epoch(
			model=model,
			loss_fn=_NonfiniteGradientLoss(),  # type: ignore[arg-type]
			dataloader=[_barlow_batch()],  # type: ignore[arg-type]
			optimizer=optimizer,
			device=torch.device('cpu'),
			epoch=1,
			grad_clip_norm=1.0,
		)

	torch.testing.assert_close(model.backbone.patch_projection.weight, before)


def test_amp_path_unscales_before_gradient_check_and_step(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _backbone_wrapper()
	optimizer = torch.optim.SGD(model.pretraining_parameters(), lr=0.1)
	events: list[str] = []

	def fake_clip_grad_norm_(
		parameters: object,
		max_norm: float,
		*,
		error_if_nonfinite: bool,
	) -> torch.Tensor:
		list(parameters)  # type: ignore[arg-type]
		assert max_norm == 1.0
		assert error_if_nonfinite is True
		events.append('clip')
		return torch.tensor(0.5)

	monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', fake_clip_grad_norm_)
	train_barlow_twins_one_epoch(
		model=model,
		loss_fn=BarlowTwinsLoss(),
		dataloader=[_barlow_batch()],  # type: ignore[arg-type]
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		scaler=_RecordingScaler(events),  # type: ignore[arg-type]
		grad_clip_norm=1.0,
	)

	assert events == ['scale', 'unscale', 'clip', 'step', 'update']


def _tiny_config(
	tmp_path: Path,
	*,
	epochs: int = 1,
	max_steps: int = 1,
	output_name: str = 'run',
	encoder_depth: int = 1,
) -> dict[str, object]:
	manifest_path = _write_synthetic_manifest(tmp_path / 'survey')
	path_list = tmp_path / 'train_npy_paths.txt'
	path_list.write_text(f'{tmp_path / "survey" / "amplitude.npy"}\n', encoding='utf-8')
	return {
		'paths': {
			'artifact_root': str(tmp_path / 'artifacts'),
			'output_root': str(tmp_path / 'artifacts' / output_name),
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
			'encoder_depth': encoder_depth,
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


def _tiny_local_config(  # noqa: PLR0913
	tmp_path: Path,
	*,
	epochs: int = 1,
	max_steps: int = 1,
	output_name: str = 'local-run',
	encoder_depth: int = 1,
	local_pairs_per_crop: int = 4,
) -> dict[str, object]:
	config = _tiny_config(
		tmp_path,
		epochs=epochs,
		max_steps=max_steps,
		output_name=output_name,
		encoder_depth=encoder_depth,
	)
	config['barlow_twins'] = {
		'method': LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
		'local_pairs_per_crop': local_pairs_per_crop,
		'projector_dim': 4,
	}
	return config


def _tiny_d4_config(  # noqa: PLR0913
	tmp_path: Path,
	*,
	epochs: int = 1,
	max_steps: int = 1,
	output_name: str = 'd4-run',
	encoder_depth: int = 1,
	local_pairs_per_crop: int = 4,
) -> dict[str, object]:
	config = _tiny_local_config(
		tmp_path,
		epochs=epochs,
		max_steps=max_steps,
		output_name=output_name,
		encoder_depth=encoder_depth,
		local_pairs_per_crop=local_pairs_per_crop,
	)
	config['augmentations'] = {
		'policy': XY_D4_TRACE_DROP_AUGMENTATION_POLICY,
		'reflection_probability': 0.5,
		'trace_drop_probability': 0.02,
	}
	return config


def _continuation_config(
	tmp_path: Path,
	*,
	source_checkpoint: str | Path,
	output_name: str,
	training_steps: int,
	unfreeze_top_blocks: int,
) -> dict[str, object]:
	config = _tiny_config(
		tmp_path,
		epochs=training_steps,
		max_steps=training_steps,
		output_name=output_name,
		encoder_depth=2,
	)
	config['continuation'] = {
		'init_checkpoint': str(source_checkpoint),
		'unfreeze_top_blocks': unfreeze_top_blocks,
	}
	return config


def _clone_tensor_state(
	payload: dict[str, object],
	key: str,
) -> dict[str, torch.Tensor]:
	state = payload[key]
	assert isinstance(state, dict)
	assert all(
		isinstance(name, str) and isinstance(value, torch.Tensor)
		for name, value in state.items()
	)
	return {
		str(name): value.detach().clone()
		for name, value in state.items()
		if isinstance(value, torch.Tensor)
	}


def _optimizer_steps(payload: dict[str, object]) -> set[int]:
	optimizer_state = payload['optimizer_state_dict']
	assert isinstance(optimizer_state, dict)
	state = optimizer_state['state']
	assert isinstance(state, dict)
	steps = set()
	for parameter_state in state.values():
		assert isinstance(parameter_state, dict)
		step = parameter_state['step']
		assert isinstance(step, int | float | torch.Tensor)
		steps.add(int(step))
	return steps


def _assert_state_prefix_unchanged(
	before: dict[str, torch.Tensor],
	after: dict[str, torch.Tensor],
	prefix: str,
) -> None:
	names = [name for name in before if name.startswith(prefix)]
	assert names
	assert all(torch.equal(before[name], after[name]) for name in names)


def _state_prefix_changed(
	before: dict[str, torch.Tensor],
	after: dict[str, torch.Tensor],
	prefix: str,
) -> bool:
	names = [name for name in before if name.startswith(prefix)]
	assert names
	return any(not torch.equal(before[name], after[name]) for name in names)


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


def _backbone_wrapper() -> BarlowTwins3D:
	return BarlowTwins3D(_backbone(), projector_dim=4)


def _barlow_batch() -> dict[str, torch.Tensor]:
	return {
		'view_a': torch.randn((2, 1, 4, 4, 4)),
		'view_b': torch.randn((2, 1, 4, 4, 4)),
		'valid_mask_a': torch.ones((2, 4, 4, 4), dtype=torch.bool),
		'valid_mask_b': torch.ones((2, 4, 4, 4), dtype=torch.bool),
	}


def _local_barlow_batch(
	*,
	local_pairs_per_crop: int,
) -> dict[str, torch.Tensor]:
	batch = _barlow_batch()
	indices = torch.arange(
		2 * local_pairs_per_crop,
		dtype=torch.int64,
	).reshape(2, local_pairs_per_crop)
	batch['local_pair_indices_a'] = indices
	batch['local_pair_indices_b'] = indices.clone()
	return batch


def _d4_local_barlow_batch(
	*,
	local_pairs_per_crop: int,
) -> dict[str, torch.Tensor]:
	batch = _local_barlow_batch(
		local_pairs_per_crop=local_pairs_per_crop,
	)
	batch.update(
		{
			'xy_transform_id_a': torch.tensor([0, 5]),
			'xy_transform_id_b': torch.tensor([0, 6]),
			'trace_drop_count_a': torch.tensor([1, 3]),
			'trace_drop_count_b': torch.tensor([2, 5]),
		}
	)
	return batch


class _NonfiniteLoss(torch.nn.Module):
	def forward(
		self,
		z_a: torch.Tensor,
		z_b: torch.Tensor,
	) -> dict[str, torch.Tensor]:
		del z_b
		return {'loss': z_a.sum() * z_a.new_tensor(float('nan'))}


class _NonfiniteGradientLoss(torch.nn.Module):
	def forward(
		self,
		z_a: torch.Tensor,
		z_b: torch.Tensor,
	) -> dict[str, torch.Tensor]:
		del z_b
		zero = z_a.sum() * 0.0
		return {'loss': zero.sqrt()}


class _RecordingScaler:
	def __init__(self, events: list[str]) -> None:
		self.events = events

	def scale(self, loss: torch.Tensor) -> torch.Tensor:
		self.events.append('scale')
		return loss

	def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
		del optimizer
		self.events.append('unscale')

	def step(self, optimizer: torch.optim.Optimizer) -> None:
		self.events.append('step')
		optimizer.step()

	def update(self) -> None:
		self.events.append('update')


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
