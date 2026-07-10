from __future__ import annotations

import sys
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import torch
import yaml

import seis_ssl_cluster.config.strat_hmm_pseudo_targets as strat_config
from proc.seis_ssl_cluster import build_strat_hmm_pseudo_targets
from seis_ssl_cluster.config.schema import STAGE_STRAT_HMM_PSEUDO_TARGETS
from seis_ssl_cluster.config.strat_hmm_pseudo_targets import (
	resolve_strat_hmm_pseudo_target_config,
)
from seis_ssl_cluster.config.validate import validate_config

if TYPE_CHECKING:
	from pathlib import Path


def test_strat_hmm_pseudo_target_config_resolves_minimal_valid_config(
	tmp_path: Path,
) -> None:
	resolved = resolve_strat_hmm_pseudo_target_config(_minimal_config(tmp_path))

	assert resolved['stage'] == STAGE_STRAT_HMM_PSEUDO_TARGETS
	assert resolved['checkpoint']['path'].endswith('strat_latest.pt')
	assert resolved['inference']['device'] == 'auto'
	assert resolved['hmm']['k'] == 6
	assert resolved['hmm']['boundary_weighting'] == {'alpha': 0.0, 'tau': 1.0}
	assert resolved['outputs']['overwrite'] is False
	assert resolved['outputs']['skip_existing'] is False


def test_strat_hmm_pseudo_target_config_resolves_from_validate_compat_layer(
	tmp_path: Path,
) -> None:
	resolved = validate_config(
		_minimal_config(tmp_path),
		stage=STAGE_STRAT_HMM_PSEUDO_TARGETS,
	)

	assert resolved['stage'] == STAGE_STRAT_HMM_PSEUDO_TARGETS


def test_strat_hmm_pseudo_target_config_rejects_unknown_keys(
	tmp_path: Path,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['unexpected'] = {}

	with pytest.raises(ValueError, match='top-level section'):
		resolve_strat_hmm_pseudo_target_config(cfg)

	cfg = _minimal_config(tmp_path)
	cfg['inference']['extra'] = True

	with pytest.raises(ValueError, match=r'inference key\(s\) not allowed'):
		resolve_strat_hmm_pseudo_target_config(cfg)

	cfg = _minimal_config(tmp_path)
	cfg['hmm']['boundary_weighting'] = {'alpha': 0.5, 'tau': 1.0, 'extra': 1}

	with pytest.raises(
		ValueError,
		match=r'hmm\.boundary_weighting key\(s\) not allowed',
	):
		resolve_strat_hmm_pseudo_target_config(cfg)


@pytest.mark.parametrize(
	('boundary_weighting', 'match'),
	[
		({'alpha': -0.1, 'tau': 1.0}, r'hmm\.boundary_weighting\.alpha'),
		({'alpha': 1.1, 'tau': 1.0}, r'hmm\.boundary_weighting\.alpha'),
		({'alpha': 0.5, 'tau': 0.0}, r'hmm\.boundary_weighting\.tau'),
		({'alpha': 0.5, 'tau': float('inf')}, r'hmm\.boundary_weighting\.tau'),
	],
)
def test_strat_hmm_pseudo_target_config_rejects_invalid_boundary_weighting(
	tmp_path: Path,
	boundary_weighting: dict[str, float],
	match: str,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['hmm']['boundary_weighting'] = boundary_weighting

	with pytest.raises(ValueError, match=match):
		resolve_strat_hmm_pseudo_target_config(cfg)


@pytest.mark.parametrize(
	('mutator', 'match'),
	[
		(
			lambda cfg: cfg['model'].__setitem__('patch_size', [7, 8, 8]),
			'divisible',
		),
		(
			lambda cfg: cfg['inference'].__setitem__('overlap', [128, 64, 64]),
			r'inference\.overlap',
		),
		(
			lambda cfg: cfg['inference'].__setitem__('output_dtype', 'float16'),
			r'inference\.output_dtype',
		),
	],
)
def test_strat_hmm_pseudo_target_config_rejects_invalid_inference_geometry(
	tmp_path: Path,
	mutator: object,
	match: str,
) -> None:
	cfg = _minimal_config(tmp_path)
	mutator(cfg)

	with pytest.raises(ValueError, match=match):
		resolve_strat_hmm_pseudo_target_config(cfg)


@pytest.mark.parametrize(
	('mutator', 'match'),
	[
		(
			lambda cfg: cfg['hmm']['transition'].__setitem__('jump_cost', -1.0),
			r'hmm\.transition\.jump_cost',
		),
		(
			lambda cfg: cfg['hmm'].__setitem__(
				'path_prior',
				{'enabled': True},
			),
			r'hmm\.path_prior\.',
		),
		(
			lambda cfg: cfg['hmm'].__setitem__('k', 5),
			r'hmm\.k',
		),
	],
)
def test_strat_hmm_pseudo_target_config_rejects_invalid_hmm_values(
	tmp_path: Path,
	mutator: object,
	match: str,
) -> None:
	cfg = _minimal_config(tmp_path)
	mutator(cfg)

	with pytest.raises(ValueError, match=match):
		resolve_strat_hmm_pseudo_target_config(cfg)


def test_strat_hmm_pseudo_target_config_uses_restricted_checkpoint_load(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	cfg = _minimal_config(tmp_path)
	cfg['hmm']['k'] = 5
	load_kwargs = {}

	def fake_load(*_args: object, **kwargs: object) -> dict[str, object]:
		load_kwargs.update(kwargs)
		return {'stratigraphy_config': {'head': {'num_prototypes': 6}}}

	monkeypatch.setattr(strat_config.torch, 'load', fake_load)

	with pytest.raises(ValueError, match=r'hmm\.k'):
		resolve_strat_hmm_pseudo_target_config(cfg)

	assert load_kwargs['weights_only'] is True
	assert load_kwargs['map_location'] == 'cpu'


def test_strat_hmm_pseudo_target_cli_dry_run_resolves_without_builder(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(
		yaml.safe_dump(_minimal_config(tmp_path)),
		encoding='utf-8',
	)

	def fail_builder(*_args: object, **_kwargs: object) -> list[Path]:
		raise AssertionError('builder should not run during dry-run')

	monkeypatch.setattr(
		build_strat_hmm_pseudo_targets,
		'build_strat_hmm_pseudo_targets',
		fail_builder,
	)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'build_strat_hmm_pseudo_targets.py',
			'--config',
			str(config_path),
			'--dry-run',
			'--device',
			'cpu',
			'--overwrite',
		],
	)

	build_strat_hmm_pseudo_targets.main()

	stdout = capsys.readouterr().out
	assert 'stage: build_strat_hmm_pseudo_targets' in stdout
	assert 'inference.device: cpu' in stdout
	assert 'outputs.overwrite: true' in stdout
	assert 'execution: dry-run; pseudo-target refresh skipped' in stdout


def test_strat_hmm_pseudo_target_cli_dry_run_accepts_skip_existing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(
		yaml.safe_dump(_minimal_config(tmp_path)),
		encoding='utf-8',
	)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'build_strat_hmm_pseudo_targets.py',
			'--config',
			str(config_path),
			'--dry-run',
			'--skip-existing',
		],
	)

	build_strat_hmm_pseudo_targets.main()

	stdout = capsys.readouterr().out
	assert 'outputs.skip_existing: true' in stdout


def _minimal_config(tmp_path: Path) -> dict[str, object]:
	artifact_root = tmp_path / 'artifacts'
	checkpoint = tmp_path / 'strat_latest.pt'
	torch.save(
		{
			'stratigraphy_config': {
				'head': {'num_prototypes': 6},
			},
		},
		checkpoint,
	)
	return deepcopy(
		{
			'paths': {'artifact_root': str(artifact_root)},
			'manifests': {'train': str(tmp_path / 'train_manifest.json')},
			'checkpoint': {'path': str(checkpoint)},
			'model': {'patch_size': [8, 8, 8]},
			'inference': {
				'window_size': [128, 128, 128],
				'overlap': [64, 64, 64],
				'batch_size': 2,
				'output_dtype': 'float32',
				'min_token_valid_fraction': 1.0,
				'device': 'auto',
			},
			'hmm': {
				'k': 6,
				'edge_margin_tokens': [8, 8, 0],
				'transition': {
					'same_cost': 0.0,
					'advance_cost': 1.0,
					'jump_cost': 4.0,
					'reverse_cost': 10.0,
					'forbid_reverse': True,
					'max_jump': 1,
				},
				'path_prior': {'enabled': False},
			},
			'outputs': {
				'pseudo_target_root': str(
					artifact_root / 'pseudo_targets' / 'strat_m1_refresh01',
				),
				'overwrite': False,
				'skip_existing': False,
			},
		},
	)
