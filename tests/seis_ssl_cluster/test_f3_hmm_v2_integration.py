"""Verify the complete screening handoff against the driver's command plan."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config
from tests.seis_ssl_cluster.test_f3_hmm_v2_manifest_configs import (
	CANDIDATES,
	EXPERIMENT,
)

if TYPE_CHECKING:
	import pytest


def test_every_stage_uses_the_same_four_candidate_handoffs(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', 'a' * 64)
	summary = load_config(EXPERIMENT / '60_summary/01_paired_screening.yaml')
	audit = load_config(EXPERIMENT / '30_pretraining/03_audit_completed_sources.yaml')
	result = subprocess.run(  # noqa: S603 - repository driver in plan-only mode
		['/bin/bash', str((EXPERIMENT / 'run_all.sh').resolve()), '--plan'],
		env=os.environ.copy(),
		cwd=tmp_path,
		capture_output=True,
		text=True,
		check=True,
	)
	commands = [
		shlex.split(line.split('] ', 1)[1])
		for line in result.stdout.splitlines()
		if line.startswith('[')
	]
	audit_paths = {Path(path).resolve() for path in audit['training_configs']}
	summary_paths = {Path(entry['training_config']) for entry in summary['candidates']}
	assert audit_paths == summary_paths
	assert len(audit_paths) == 4
	assert len(commands) == 81
	for tag, ks in CANDIDATES.items():
		candidate = f'mae100_hmm_v2_mh_k{tag}_distill020'
		manifest_path = (EXPERIMENT / '20_manifests' / f'{candidate}.yaml').resolve()
		training_path = (
			EXPERIMENT / '30_pretraining' / candidate / '02_full_25ep.yaml'
		).resolve()
		embedding_path = (EXPERIMENT / '40_embeddings' / f'{candidate}.yaml').resolve()
		downstream_path = (EXPERIMENT / '50_downstream' / f'{candidate}.yaml').resolve()
		manifest = load_config(manifest_path)
		training = load_config(training_path)
		embedding = load_config(embedding_path)
		downstream = load_config(downstream_path)
		assert training['head']['ks'] == list(manifest['head_roots']) == ks
		assert training['pseudo_targets']['manifest'] == manifest['manifest']
		assert Path(training['paths']['output_root']) / 'latest.pt' == Path(
			embedding['embeddings']['checkpoint']
		)
		assert (
			embedding['embeddings']['checkpoint']
			== downstream['candidate']['checkpoint']
		)
		assert (
			embedding['embeddings']['output_dir']
			== downstream['candidate']['embeddings_dir']
		)
		assert (
			downstream['candidate']['id']
			== training['identity']['model_tag']
			== candidate
		)
		assert training_path in audit_paths
		assert {
			'training_config': str(training_path),
			'downstream_config': str(downstream_path),
			'head_ks': ks,
		} in summary['candidates']
		assert sum(str(manifest_path) in command for command in commands) == 1
		assert sum(str(training_path) in command for command in commands) == 1
		assert sum(str(embedding_path) in command for command in commands) == 1
		cells = [command for command in commands if str(downstream_path) in command]
		assert len(cells) == 15
		assert {
			(
				command[command.index('--layout') + 1],
				command[command.index('--size') + 1],
			)
			for command in cells
		} == {
			(f'layout_{index:03}', size)
			for index in range(5)
			for size in ('small', 'medium', 'large')
		}
	assert not list(tmp_path.iterdir())
