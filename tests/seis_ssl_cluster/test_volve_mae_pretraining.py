from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

import seis_ssl_cluster.volve.mae_validation as validation_module
from seis_ssl_cluster.config import load_config, resolve_mae_training_config
from seis_ssl_cluster.data import (
	AmplitudePretrainDataset,
	ZeroMaskConfig,
	read_manifest_json,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.checkpoint import capture_rng_state
from seis_ssl_cluster.training.mae_checkpoint import _save_mae_checkpoint
from seis_ssl_cluster.training.random_checkpoint import (
	random_mae_checkpoint_config_from_mapping,
)
from seis_ssl_cluster.volve import (
	VOLVE_CANONICAL_DATASET_ID,
	validate_volve_mae_inputs_from_configs,
)
from tests.seis_ssl_cluster.helpers_volve import write_synthetic_volve_registration

if TYPE_CHECKING:
	from collections.abc import Mapping

PRETRAIN_ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/10_pretrain/'
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
)
SMOKE_CONFIG = PRETRAIN_ROOT / '01_smoke_2step.yaml'
FULL_CONFIG = PRETRAIN_ROOT / '02_full_100ep.yaml'
RANDOM_CONFIG = PRETRAIN_ROOT / '03_create_random_checkpoint.yaml'


def test_configs_resolve_fixed_scientific_contract(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	smoke = resolve_mae_training_config(load_config(SMOKE_CONFIG))
	full = resolve_mae_training_config(load_config(FULL_CONFIG))

	assert smoke['train']['device'] == 'cpu'
	assert smoke['train']['max_steps'] == 2
	assert full['train']['device'] == 'cuda'
	assert full['train']['epochs'] == 100
	assert full['train']['samples_per_epoch'] == 10_000
	assert full['train']['seed'] == 42
	assert full['data']['local_crop_size'] == [128, 128, 128]
	assert full['data']['amplitude_agc']['window_z'] == 65
	assert full['model']['patch_size'] == [8, 8, 8]
	assert full['loss']['visible_reconstruction_weight'] == 0.1


def test_valid_synthetic_inputs_and_explicit_mask_dataset_pass(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config, smoke, full = _input_fixture(tmp_path, monkeypatch)
	inputs = validate_volve_mae_inputs_from_configs(
		input_config=config,
		smoke_raw=smoke,
		full_raw=full,
	)

	assert inputs.scientific_identity_sha256
	assert inputs.valid_mask_path.name == 'valid_trace_mask.npy'
	manifest = read_manifest_json(inputs.manifest_path)[0]
	dataset = AmplitudePretrainDataset(
		[manifest],
		local_crop_size_xyz=config.identity.shape_xyz,
		patch_size_xyz=(1, 1, 1),
		spatial_mask_ratio=0.5,
		block_size_tokens_xyz=(1, 1, 1),
		zero_mask=ZeroMaskConfig(enabled=False),
		min_valid_fraction=0.0,
	)
	sample = dataset[0]
	assert np.isfinite(sample['x']).all()
	assert not sample['local_valid_mask'][1, 2, :].any()


def test_inputs_reject_non_allowlisted_and_supervision_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config, smoke, full = _input_fixture(tmp_path, monkeypatch)
	smoke['loss']['gradient_weight'] = 0.1
	with pytest.raises(ValueError, match='gradient_weight'):
		validate_volve_mae_inputs_from_configs(
			input_config=config,
			smoke_raw=smoke,
			full_raw=full,
		)

	config, smoke, full = _input_fixture(tmp_path / 'second', monkeypatch)
	smoke['manifests']['horizon_labels'] = '/public/horizons.npz'
	with pytest.raises(ValueError, match=r'horizon|supervision'):
		validate_volve_mae_inputs_from_configs(
			input_config=config,
			smoke_raw=smoke,
			full_raw=full,
		)


def test_random_checkpoint_config_has_paired_role(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	artifact_root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	settings = random_mae_checkpoint_config_from_mapping(load_config(RANDOM_CONFIG))

	assert settings.seed == 42
	assert settings.reference_model_tag == validation_module.MODEL_TAG
	assert settings.reference_checkpoint == (
		artifact_root
		/ 'pretraining/volve/horizon_benchmark_v1'
		/ validation_module.MODEL_TAG
		/ 'full_100ep/latest.pt'
	)
	assert settings.output_checkpoint.name == 'mae_random_seed42.pt'
	assert settings.output_checkpoint != settings.reference_checkpoint


def test_valid_synthetic_smoke_and_full_artifacts_pass(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs = _validated_inputs(tmp_path, monkeypatch)
	monkeypatch.setattr(validation_module, '_build_mae_model', _tiny_model)
	_write_run_artifacts(inputs, phase='smoke')
	smoke_result = validation_module._validate_smoke(  # noqa: SLF001
		inputs,
		base=_result_base(inputs, check='smoke'),
	)
	assert smoke_result.checkpoint_epoch == 1
	assert smoke_result.checkpoint_global_step == 2
	assert smoke_result.resolved_precision == 'float32'

	_write_run_artifacts(inputs, phase='full')
	full_result = validation_module._validate_full(  # noqa: SLF001
		inputs,
		base=_result_base(inputs, check='full'),
	)
	assert full_result.checkpoint_epoch == 100
	assert full_result.checkpoint_global_step == 250_000
	assert full_result.resolved_precision == 'bfloat16'


def test_full_validator_runs_checkpoint_forward(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs = _validated_inputs(tmp_path, monkeypatch)
	monkeypatch.setattr(validation_module, '_build_mae_model', _tiny_model)
	_write_run_artifacts(inputs, phase='full')
	forward_labels: list[str] = []

	def record_forward(_model: AmplitudeMAE3D, *, label: str) -> None:
		forward_labels.append(label)

	monkeypatch.setattr(
		validation_module,
		'_validate_checkpoint_forward',
		record_forward,
	)
	validation_module._validate_full(  # noqa: SLF001
		inputs,
		base=_result_base(inputs, check='full'),
	)

	assert forward_labels == ['latest full']


def test_validator_rejects_changed_scientific_input_snapshot(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs = _validated_inputs(tmp_path, monkeypatch)
	monkeypatch.setattr(validation_module, '_build_mae_model', _tiny_model)
	_write_run_artifacts(inputs, phase='smoke')
	root = Path(cast('Mapping[str, object]', inputs.smoke['paths'])['output_root'])
	(root / 'inputs' / inputs.normalization_stats_path.name).write_text(
		'{}\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='normalization stats snapshot'):
		validation_module._validate_smoke(  # noqa: SLF001
			inputs,
			base=_result_base(inputs, check='smoke'),
		)


def test_proc_entrypoint_and_docs_contract() -> None:
	module = importlib.import_module('proc.seis_ssl_cluster.validate_volve_mae')
	help_text = module.build_parser().format_help()
	docs = Path('docs/volve_mae_pretraining.md').read_text(encoding='utf-8')

	assert '--input-config' in help_text
	assert '--check {inputs,smoke,full}' in help_text
	assert callable(module.main)
	assert VOLVE_CANONICAL_DATASET_ID in docs
	assert 'transductive self-supervised pretraining' in docs
	assert 'No F3, NOPIMS, Parihaka' in docs


def test_experiment_readme_has_reentrant_full_runbook() -> None:
	readme = (PRETRAIN_ROOT / 'README.md').read_text(encoding='utf-8')

	assert 'prepare_volve_canonical_inputs.py \\\n  --only-missing' in readme
	assert '02_full_100ep.yaml"\n' in readme
	assert '--resume "$FULL_RUN/latest.pt"' in readme
	assert '--check full' in readme
	assert 'mae_random_seed42.pt' in readme
	assert "random_payload['metadata']['random_encoder_baseline'] is True" in readme
	assert (
		"random_payload['config']['model'] "
		"== reference_payload['config']['model']"
	) in readme


def _input_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, dict[str, object], dict[str, object]]:
	config = write_synthetic_volve_registration(tmp_path)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(config.artifact_root))
	return config, load_config(SMOKE_CONFIG), load_config(FULL_CONFIG)


def _validated_inputs(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> validation_module.VolveMaeInputValidation:
	config, smoke, full = _input_fixture(tmp_path, monkeypatch)
	return validate_volve_mae_inputs_from_configs(
		input_config=config,
		smoke_raw=smoke,
		full_raw=full,
	)


def _tiny_model(_model: Mapping[str, object] | None = None) -> AmplitudeMAE3D:
	return AmplitudeMAE3D(
		patch_size_xyz=(1, 1, 1),
		encoder_dim=12,
		encoder_depth=1,
		encoder_heads=1,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=1,
	)


def _write_run_artifacts(
	inputs: validation_module.VolveMaeInputValidation,
	*,
	phase: str,
) -> None:
	config = inputs.smoke if phase == 'smoke' else inputs.full
	root = Path(cast('Mapping[str, object]', config['paths'])['output_root'])
	root.mkdir(parents=True)
	model = _tiny_model()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
	if phase == 'smoke':
		epoch, global_step, amp_enabled = 1, 2, False
		precision = {
			'amp_requested': False,
			'amp_dtype_requested': 'auto',
			'resolved_dtype': 'float32',
			'amp_enabled': False,
			'grad_scaler_enabled': False,
		}
	else:
		epoch, global_step, amp_enabled = 100, 250_000, True
		precision = {
			'amp_requested': True,
			'amp_dtype_requested': 'auto',
			'resolved_dtype': 'bfloat16',
			'amp_enabled': True,
			'grad_scaler_enabled': False,
		}
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = torch.Generator().get_state()
	for name in ('latest.pt', 'best.pt'):
		_save_mae_checkpoint(
			root / name,
			model=model,
			optimizer=optimizer,
			epoch=epoch,
			config=config,
			metrics={'loss': 1.0, 'amp_enabled': float(amp_enabled)},
			global_step=global_step,
			amp_enabled=amp_enabled,
			scaler=None,
			checkpoint_kind='epoch',
			batch_index=None,
			rng_state=rng_state,
		)
	(root / 'resolved_config.json').write_text(
		json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	shutil.copy2(inputs.manifest_path, root / 'manifest.json')
	(root / 'inputs').mkdir()
	shutil.copy2(inputs.path_list_path, root / 'inputs' / inputs.path_list_path.name)
	normalization_snapshot = root / 'inputs' / inputs.normalization_stats_path.name
	canonical_metadata_snapshot = (
		root / 'inputs' / inputs.canonical_input_metadata_path.name
	)
	shutil.copy2(inputs.normalization_stats_path, normalization_snapshot)
	shutil.copy2(inputs.canonical_input_metadata_path, canonical_metadata_snapshot)
	(root / 'run_metadata.json').write_text(
		json.dumps(
			{
				'runtime_check_mode': 'once',
				'precision': precision,
				'input_scientific_identity_sha256': (
					inputs.scientific_identity_sha256
				),
				'normalization_stats_sha256': validation_module._file_sha256(  # noqa: SLF001
					normalization_snapshot
				),
				'canonical_input_metadata_sha256': validation_module._file_sha256(  # noqa: SLF001
					canonical_metadata_snapshot
				),
			}
		)
		+ '\n',
		encoding='utf-8',
	)


def _result_base(
	inputs: validation_module.VolveMaeInputValidation,
	*,
	check: str,
) -> dict[str, object]:
	return {
		'check': check,
		'status': 'pass',
		'input_config': Path('input.yaml'),
		'smoke_config': SMOKE_CONFIG,
		'full_config': FULL_CONFIG,
		'canonical_dataset_id': VOLVE_CANONICAL_DATASET_ID,
		'scientific_identity_sha256': inputs.scientific_identity_sha256,
		'manifest': inputs.manifest_path,
		'valid_mask': inputs.valid_mask_path,
		'smoke_output_root': Path(
			cast('Mapping[str, object]', inputs.smoke['paths'])['output_root']
		),
		'full_output_root': Path(
			cast('Mapping[str, object]', inputs.full['paths'])['output_root']
		),
	}
