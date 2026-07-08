from __future__ import annotations

import sys
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import yaml

from proc.seis_ssl_cluster import train_strat_hmm_pretext
from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.config.schema import STAGE_STRAT_HMM_PRETEXT_TRAINING
from seis_ssl_cluster.config.validate import validate_config

if TYPE_CHECKING:
	from pathlib import Path


def test_strat_hmm_pretext_config_resolves_minimal_valid_config(
	tmp_path: Path,
) -> None:
	resolved = resolve_strat_hmm_pretext_config(_minimal_config(tmp_path))

	assert resolved['stage'] == STAGE_STRAT_HMM_PRETEXT_TRAINING
	assert resolved['data']['amplitude_agc'] == {'enabled': False}
	assert resolved['zero_mask']['enabled'] is True
	assert resolved['student']['init_checkpoint'] is None
	assert resolved['student']['unfreeze_top_blocks'] == 0
	assert resolved['head']['temperature'] == 0.1
	assert resolved['loss']['distillation_weight'] == 0.0
	assert resolved['train']['device'] == 'auto'
	assert resolved['model']['name'] == 'amp_mae3d'
	assert resolved['data']['input_channels'] == 1


def test_strat_hmm_pretext_config_resolves_from_validate_compat_layer(
	tmp_path: Path,
) -> None:
	resolved = validate_config(
		_minimal_config(tmp_path),
		stage=STAGE_STRAT_HMM_PRETEXT_TRAINING,
	)

	assert resolved['stage'] == STAGE_STRAT_HMM_PRETEXT_TRAINING


def test_strat_hmm_pretext_config_rejects_unknown_keys(tmp_path: Path) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['unexpected'] = {}

	with pytest.raises(ValueError, match='top-level section'):
		resolve_strat_hmm_pretext_config(cfg)

	cfg = _minimal_config(tmp_path)
	cfg['head']['extra'] = True

	with pytest.raises(ValueError, match=r'head key\(s\) not allowed'):
		resolve_strat_hmm_pretext_config(cfg)


def test_strat_hmm_pretext_config_rejects_stale_fixed_keys(
	tmp_path: Path,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['model']['name'] = 'other'

	with pytest.raises(ValueError, match=r'model\.name is fixed'):
		resolve_strat_hmm_pretext_config(cfg)


def test_strat_hmm_pretext_config_requires_matching_prototype_count(
	tmp_path: Path,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['pseudo_targets']['k'] = 8

	with pytest.raises(ValueError, match=r'pseudo_targets\.k'):
		resolve_strat_hmm_pretext_config(cfg)


def test_strat_hmm_pretext_config_enforces_unfreeze_bounds(
	tmp_path: Path,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['student']['unfreeze_top_blocks'] = 9

	with pytest.raises(ValueError, match=r'student\.unfreeze_top_blocks'):
		resolve_strat_hmm_pretext_config(cfg)


def test_strat_hmm_pretext_config_requires_distillation_when_unfrozen(
	tmp_path: Path,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['student']['unfreeze_top_blocks'] = 1

	with pytest.raises(ValueError, match=r'loss\.distillation_weight'):
		resolve_strat_hmm_pretext_config(cfg)

	cfg['loss']['distillation_weight'] = 0.2

	resolved = resolve_strat_hmm_pretext_config(cfg)

	assert resolved['loss']['distillation_weight'] == 0.2


def test_strat_hmm_pretext_config_enforces_crop_patch_divisibility(
	tmp_path: Path,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['data']['local_crop_size'] = [127, 128, 128]

	with pytest.raises(ValueError, match='divisible'):
		resolve_strat_hmm_pretext_config(cfg)


def test_strat_hmm_pretext_config_rejects_zero_checkpoint_interval(
	tmp_path: Path,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['train']['checkpoint_every_steps'] = 0

	with pytest.raises(ValueError, match=r'train\.checkpoint_every_steps'):
		resolve_strat_hmm_pretext_config(cfg)


def test_strat_hmm_pretext_config_validates_training_paths(
	tmp_path: Path,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['pseudo_targets']['input_dir'] = str(tmp_path / 'missing-targets')

	with pytest.raises(FileNotFoundError, match=r'pseudo_targets\.input_dir'):
		resolve_strat_hmm_pretext_config(cfg)

	cfg = _minimal_config(tmp_path)
	cfg['teacher']['checkpoint'] = str(tmp_path / 'missing.pt')

	with pytest.raises(FileNotFoundError, match=r'teacher\.checkpoint'):
		resolve_strat_hmm_pretext_config(cfg)


def test_strat_hmm_pretext_cli_dry_run_resolves_without_training(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(
		yaml.safe_dump(_minimal_config(tmp_path)),
		encoding='utf-8',
	)

	def fail_training(*_args: object, **_kwargs: object) -> Path:
		raise AssertionError('training loop should not run during dry-run')

	monkeypatch.setattr(
		train_strat_hmm_pretext,
		'run_strat_hmm_pretext_training',
		fail_training,
	)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'train_strat_hmm_pretext.py',
			'--config',
			str(config_path),
			'--dry-run',
			'--device',
			'cpu',
			'--max-steps',
			'2',
		],
	)

	train_strat_hmm_pretext.main()

	stdout = capsys.readouterr().out
	assert 'stage: train_strat_hmm_pretext' in stdout
	assert 'train.device: cpu' in stdout
	assert 'execution: dry-run; training skipped' in stdout


def _minimal_config(tmp_path: Path) -> dict[str, object]:
	pseudo_targets = tmp_path / 'pseudo_targets'
	pseudo_targets.mkdir(exist_ok=True)
	teacher_checkpoint = tmp_path / 'teacher.pt'
	teacher_checkpoint.touch()
	artifact_root = tmp_path / 'artifacts'
	output_root = artifact_root / 'pretraining' / 'strat_hmm_m1_run'
	return deepcopy(
		{
			'paths': {
				'artifact_root': str(artifact_root),
				'output_root': str(output_root),
			},
			'manifests': {
				'train': str(tmp_path / 'train_manifest.json'),
				'train_path_list': str(tmp_path / 'train_path_list.txt'),
			},
			'data': {'local_crop_size': [128, 128, 128]},
			'model': {
				'patch_size': [8, 8, 8],
				'encoder_dim': 384,
				'encoder_depth': 8,
				'encoder_heads': 6,
				'decoder_dim': 256,
				'decoder_depth': 4,
				'decoder_heads': 4,
			},
			'pseudo_targets': {
				'input_dir': str(pseudo_targets),
				'k': 6,
			},
			'teacher': {'checkpoint': str(teacher_checkpoint)},
			'student': {},
			'head': {'num_prototypes': 6},
			'loss': {},
			'train': {
				'batch_size': 4,
				'samples_per_epoch': 1024,
				'epochs': 10,
			},
		},
	)
