from __future__ import annotations

import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.stratigraphy import pseudo_target_paths

FIVE_WAY_TARGET_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/'
	'110_lithology_mae_local_bt_five_way_v1/20_hmm_targets/local_bt100'
)
GLOBAL_BT_TARGET_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/20_hmm_targets/bt100'
)


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


@pytest.fixture
def embedding_configs(artifact_root: Path) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		'local_bt100': resolve_embedding_extraction_config(
			load_config(FIVE_WAY_TARGET_ROOT / '01_extract_embeddings.yaml')
		),
		'bt100': resolve_embedding_extraction_config(
			load_config(GLOBAL_BT_TARGET_ROOT / '01_extract_embeddings.yaml')
		),
	}


@pytest.fixture
def clustering_configs(artifact_root: Path) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		'local_bt100': resolve_clustering_config(
			load_config(FIVE_WAY_TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml')
		),
		'bt100': resolve_clustering_config(
			load_config(GLOBAL_BT_TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml')
		),
	}


def test_extraction_uses_local_bt_100ep_encoder_tokens(
	embedding_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	config = embedding_configs['local_bt100']
	expected_checkpoint = (
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ 'local_barlow_twins_v1/full_100ep/latest.pt'
	)

	assert config['stage'] == 'extract_embeddings'
	assert Path(config['embeddings']['checkpoint']) == expected_checkpoint
	assert config['embeddings']['checkpoint'] != embedding_configs['bt100'][
		'embeddings'
	]['checkpoint']
	serialized = repr(config).lower()
	assert 'trace_drop' not in serialized
	# The extractor emits bare encoder tokens because the resolved contract has
	# no output-source switch at all; pin the exact key set so adding one fails.
	assert set(config['embeddings']) == {'checkpoint', 'output_dir'}
	assert set(config['embedding']) == {
		'window_size',
		'overlap',
		'output_dtype',
		'batch_size',
		'prefetch_queue_depth',
		'amp',
		'amp_dtype',
		'stage_timing',
		'min_token_valid_fraction',
		'preprocessing_cache',
	}


def test_extraction_conditions_match_existing_bt100_contract(
	embedding_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	local = embedding_configs['local_bt100']
	standard = embedding_configs['bt100']

	assert local['paths'] == standard['paths']
	assert local['manifests'] == standard['manifests']
	assert local['embedding'] == standard['embedding']
	assert local['embedding']['window_size'] == [128, 128, 128]
	assert local['embedding']['overlap'] == [64, 64, 64]
	assert local['embedding']['output_dtype'] == 'float16'
	assert local['embedding']['batch_size'] == 1
	assert local['embedding']['amp'] is False
	assert local['embedding']['min_token_valid_fraction'] == 0.5
	expected_output = (
		artifact_root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ 'mae_local_bt_five_way_v1/hmm_targets/local_bt100/overlap_x64'
	)
	assert Path(local['embeddings']['output_dir']) == expected_output


def test_clustering_matches_bt100_k6_science_except_paths(
	clustering_configs: dict[str, dict[str, object]],
	embedding_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	local = clustering_configs['local_bt100']
	standard = clustering_configs['bt100']

	assert local['stage'] == standard['stage'] == 'cluster_embeddings'
	assert local['embeddings']['input_dir'] == embedding_configs['local_bt100'][
		'embeddings'
	]['output_dir']
	assert {
		key: value
		for key, value in local['clustering'].items()
		if key != 'output_dir'
	} == {
		key: value
		for key, value in standard['clustering'].items()
		if key != 'output_dir'
	}
	assert local['clustering']['k_values'] == [6]
	assert local['clustering']['seed'] == 42
	assert local['clustering']['stratigraphic_hmm']['iterations'] == 10
	expected_output = (
		artifact_root
		/ 'clustering/f3/facies_benchmark_v1'
		/ 'mae_local_bt_five_way_v1/hmm_targets/local_bt100/k6'
	)
	assert Path(local['clustering']['output_dir']) == expected_output


def test_output_paths_do_not_collide_with_global_bt_artifacts(
	embedding_configs: dict[str, dict[str, object]],
	clustering_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	local_outputs = {
		embedding_configs['local_bt100']['embeddings']['output_dir'],
		clustering_configs['local_bt100']['clustering']['output_dir'],
		str(_export_arguments(artifact_root, 'local_bt100')['--pseudo-target-root']),
	}
	standard_outputs = {
		embedding_configs['bt100']['embeddings']['output_dir'],
		clustering_configs['bt100']['clustering']['output_dir'],
		str(_export_arguments(artifact_root, 'bt100')['--pseudo-target-root']),
	}

	assert local_outputs.isdisjoint(standard_outputs)
	for output in local_outputs:
		assert 'mae_local_bt_five_way_v1' in output
		assert 'ssl_hmm_continuation_v1' not in output


def test_export_script_matches_bt100_k6_contract(artifact_root: Path) -> None:
	local = _export_arguments(artifact_root, 'local_bt100')
	standard = _export_arguments(artifact_root, 'bt100')

	expected_pseudo_root = (
		artifact_root
		/ 'pseudo_targets/f3/facies_benchmark_v1'
		/ 'mae_local_bt_five_way_v1/local_bt100'
	)
	export_root = Path(local['--pseudo-target-root'])
	assert export_root == expected_pseudo_root
	paths = pseudo_target_paths(export_root, k=6, survey_id='survey')
	assert paths.labels.parent == expected_pseudo_root / 'k6'
	for flag in (
		'--k',
		'--confidence',
		'--boundary-alpha',
		'--boundary-tau',
		'--schema-version',
	):
		assert local[flag] == standard[flag]
	assert local['--k'] == '6'
	assert local['--confidence'] == '1.0'
	assert local['--boundary-alpha'] == '0.0'
	assert local['--boundary-tau'] == '1.0'
	assert local['--schema-version'] == '2'


def test_target_pipeline_does_not_reference_disallowed_sources() -> None:
	for relative_path in (
		Path('01_extract_embeddings.yaml'),
		Path('k6/02_cluster_hmm_k6.yaml'),
		Path('k6/03_export_pseudo_targets.sh'),
	):
		text = (FIVE_WAY_TARGET_ROOT / relative_path).read_text(encoding='utf-8')
		lowered = text.lower()
		assert 'trace_drop' not in lowered
		assert 'full_25ep' not in lowered
		assert '/stage2/' not in lowered
		assert 'ssl_hmm_continuation_v1' not in lowered
		assert 'parihaka' not in lowered


def test_target_pipeline_dry_runs_create_no_artifacts(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	for script, config in (
		(
			'proc/seis_ssl_cluster/extract_embeddings.py',
			FIVE_WAY_TARGET_ROOT / '01_extract_embeddings.yaml',
		),
		(
			'proc/seis_ssl_cluster/cluster_embeddings.py',
			FIVE_WAY_TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml',
		),
	):
		result = subprocess.run(  # noqa: S603
			[sys.executable, script, '--config', str(config), '--dry-run'],
			check=True,
			capture_output=True,
			text=True,
			env=environment,
		)
		assert 'dry-run' in result.stdout
		assert not artifact_root.exists()


def _export_arguments(artifact_root: Path, variant: str) -> dict[str, str]:
	del artifact_root
	root = FIVE_WAY_TARGET_ROOT if variant == 'local_bt100' else GLOBAL_BT_TARGET_ROOT
	path = root / 'k6/03_export_pseudo_targets.sh'
	text = path.read_text(encoding='utf-8')
	assert text.startswith('#!/usr/bin/env bash\n')
	assert 'set -euo pipefail' in text
	assert path.stat().st_mode & stat.S_IXUSR
	command_lines = [
		line.strip().removesuffix('\\').strip()
		for line in text.splitlines()
		if line.strip()
		and not line.startswith('#!')
		and line.strip() != 'set -euo pipefail'
	]
	tokens = shlex.split(os.path.expandvars(' '.join(command_lines)))
	assert tokens[:2] == [
		'python',
		'proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py',
	]
	flag_tokens = tokens[2:]
	assert len(flag_tokens) % 2 == 0
	return dict(zip(flag_tokens[::2], flag_tokens[1::2], strict=True))
