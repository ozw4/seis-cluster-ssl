"""Regression test for the committed HMM_v1 freeze boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config

if TYPE_CHECKING:
	import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / 'reports/hmm_v1/manifest.json'


def test_hmm_v1_freeze_is_internally_consistent() -> None:
	"""Keep result hashes, links, file inventory, and Single-Head configs frozen."""
	completed = subprocess.run(
		[sys.executable, 'tools/freeze_hmm_v1.py'],
		cwd=REPOSITORY_ROOT,
		check=True,
		capture_output=True,
		text=True,
	)
	result = json.loads(completed.stdout)
	assert result == {
		'cells': 405,
		'hmm_training_configs': 18,
		'repository_paths': 70,
		'status': 'ok',
	}


def test_hmm_v1_training_configs_keep_the_single_head_contract(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	"""Resolve every frozen HMM config without starting a training run."""
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
	for entry in manifest['hmm_training_configs']:
		config = load_config(REPOSITORY_ROOT / entry['path'])
		Path(config['pseudo_targets']['input_dir']).mkdir(parents=True, exist_ok=True)
		for checkpoint in (
			config['teacher']['checkpoint'],
			config['student'].get('init_checkpoint'),
		):
			if checkpoint is not None:
				checkpoint_path = Path(checkpoint)
				checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
				checkpoint_path.touch(exist_ok=True)
		resolved = resolve_strat_hmm_pretext_config(config)
		assert 'spec' not in resolved['head']
		assert resolved['head']['num_prototypes'] == 6
		assert resolved['pseudo_targets']['k'] == 6
		assert resolved['train']['epochs'] == 25
		assert resolved['loss']['distillation_weight'] == entry['distillation_weight']
