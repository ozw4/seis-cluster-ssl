from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from proc.seis_ssl_cluster import audit_strat_hmm_multi_head_sources as cli
from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from seis_ssl_cluster.stratigraphy.prototypes import (
	MultiResolutionOrderedPrototypeHeads,
)
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import save_strat_hmm_checkpoint
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source
from tests.seis_ssl_cluster.test_f3_hmm_v2_manifest_configs import (
	CANDIDATES,
	EXPERIMENT,
	configs,  # noqa: F401
	live_configs,  # noqa: F401
)

if TYPE_CHECKING:
	from typing import Any


@pytest.fixture(params=CANDIDATES)
def completed_source(
	request: pytest.FixtureRequest,
	live_configs: dict[str, dict[str, Any]],  # noqa: F811
	monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
	tag = request.param
	build = live_configs[tag]
	manifest = Path(build['manifest'])
	build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=Path(build['source_embedding_dir']),
		head_roots={k: Path(root) for k, root in build['head_roots'].items()},
		replay_k6_root=Path(build['replay_k6_root']),
	)
	path = (
		EXPERIMENT
		/ '30_pretraining'
		/ f'mae100_hmm_v2_mh_k{tag}_distill020/02_full_25ep.yaml'
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', file_sha256(manifest)
	)
	raw = load_config(path)
	student = torch.nn.Module()
	student.encoder = torch.nn.Module()
	student.encoder.layers = torch.nn.ModuleList(
		[torch.nn.Linear(1, 1) for _ in range(8)]
	)
	student.requires_grad_(requires_grad=False)
	student.encoder.layers[7].requires_grad_(requires_grad=True)
	parent = Path(raw['student']['init_checkpoint'])
	parent.parent.mkdir(parents=True, exist_ok=True)
	torch.save({'model_state_dict': student.state_dict()}, parent)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', file_sha256(manifest)
	)
	config = resolve_strat_hmm_pretext_config(load_config(path))
	head = MultiResolutionOrderedPrototypeHeads(
		feature_dim=1,
		ks=tuple(CANDIDATES[tag]),
		projection_dim=128,
		temperature=0.1,
		normalize=True,
	)
	optimizer = torch.optim.AdamW(
		[
			{'params': head.parameters(), 'name': 'head'},
			{
				'params': [p for p in student.parameters() if p.requires_grad],
				'name': 'encoder',
			},
		],
		lr=1.0e-5,
		weight_decay=0.05,
	)
	with torch.no_grad():
		student.encoder.layers[7].weight.add_(1)
	checkpoint = Path(config['paths']['output_root']) / 'latest.pt'
	save_strat_hmm_checkpoint(
		checkpoint,
		student=student,
		head=head,
		optimizer=optimizer,
		epoch=25,
		global_step=15625,
		mae_config={},
		stratigraphy_config=config,
		metrics={'loss': 1.0},
		checkpoint_kind='epoch',
		batch_index=None,
		trainability_summary={
			'trainable_names': [
				name for name, p in student.named_parameters() if p.requires_grad
			]
		},
		control_identity={
			'input_identities': {
				key: {'sha256': file_sha256(parent)}
				for key in ('teacher_checkpoint', 'student_init_checkpoint')
			},
			'initial_state_sha256': {'student': '0' * 64, 'head': '1' * 64},
		},
	)
	return path, checkpoint


def test_completed_sources_audit_without_writes(
	completed_source: tuple[Path, Path],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	path, _checkpoint = completed_source
	config = load_config(path)
	root = Path(config['paths']['artifact_root'])
	monkeypatch.delenv('SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256')
	before = {
		p: (p.stat().st_mtime_ns, file_sha256(p))
		for p in root.rglob('*')
		if p.is_file()
	}
	result = audit_multi_head_source(path)
	assert result['status'] == 'complete'
	assert result['epoch'] == 25
	assert result['global_step'] == 15625
	assert result['head_ks'] == config['head']['ks']
	assert before == {
		p: (p.stat().st_mtime_ns, file_sha256(p))
		for p in root.rglob('*')
		if p.is_file()
	}


@pytest.mark.parametrize(
	'fault',
	['partial', 'amp', 'frozen', 'head', 'parent', 'missing', 'manifest', 'replay'],
)
def test_sources_fail_closed(completed_source: tuple[Path, Path], fault: str) -> None:
	path, checkpoint = completed_source
	payload = load_checkpoint(checkpoint, map_location='cpu')
	if fault == 'partial':
		payload['epoch'] = 24
	elif fault == 'amp':
		payload['amp_enabled'] = True
	elif fault == 'frozen':
		payload['model_state_dict']['encoder.layers.0.weight'].add_(1)
	elif fault == 'head':
		payload['stratigraphy_checkpoint']['head_ks'] = [4, 10]
	elif fault == 'parent':
		Path(load_config(path)['teacher']['checkpoint']).write_bytes(b'changed parent')
	elif fault == 'missing':
		checkpoint.unlink()
	elif fault == 'manifest':
		manifest = Path(load_config(path)['pseudo_targets']['manifest'])
		manifest.write_text(manifest.read_text() + '\n')
	elif fault == 'replay':
		manifest = json.loads(
			Path(load_config(path)['pseudo_targets']['manifest']).read_text()
		)
		metadata = json.loads(
			(
				Path(manifest['k6_replay_parity']['replay_root'])
				/ 'k6/survey.pseudo_target_metadata.json'
			).read_text()
		)
		decoded = Path(metadata['source']['source_label_path'])
		labels = np.load(decoded)
		labels[0, 0, 0] = 1
		np.save(decoded, labels)
	if fault != 'missing':
		torch.save(payload, checkpoint)
	with pytest.raises((ValueError, FileNotFoundError)):
		audit_multi_head_source(path)


def test_audit_plan_is_portable_and_selects_only_full_configs(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	path = EXPERIMENT / '30_pretraining/03_audit_completed_sources.yaml'
	config = load_config(path)
	assert len(config['training_configs']) == 4
	assert all(Path(p).name == '02_full_25ep.yaml' for p in config['training_configs'])
	monkeypatch.setattr(sys, 'argv', ['audit', '--config', str(path), '--dry-run'])
	cli.main()
	assert '"status": "planned"' in capsys.readouterr().out
