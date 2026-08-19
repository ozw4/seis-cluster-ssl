from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
)

TARGET_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/20_hmm_targets'
)
VARIANTS = ('mae100', 'bt100')


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


@pytest.fixture
def embedding_configs(
	artifact_root: Path,
) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		variant: resolve_embedding_extraction_config(
			load_config(TARGET_ROOT / variant / 'k6/01_extract_embeddings.yaml')
		)
		for variant in VARIANTS
	}


@pytest.fixture
def clustering_configs(
	artifact_root: Path,
) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		variant: resolve_clustering_config(
			load_config(TARGET_ROOT / variant / 'k6/02_cluster_hmm_k6.yaml')
		)
		for variant in VARIANTS
	}


def test_embedding_configs_resolve_from_separate_stage1_100ep_sources(
	embedding_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	expected_checkpoints = {
		'mae100': artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1/mae/full_100ep/latest.pt',
		'bt100': artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1/barlow_twins/full_100ep/latest.pt',
	}
	expected_outputs = {
		variant: artifact_root
		/ 'embeddings/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/hmm_targets/{variant}/k6/overlap_x64'
		for variant in VARIANTS
	}

	for variant, config in embedding_configs.items():
		assert config['stage'] == 'extract_embeddings'
		assert Path(config['embeddings']['checkpoint']) == expected_checkpoints[variant]
		assert Path(config['embeddings']['output_dir']) == expected_outputs[variant]
		assert '25ep' not in config['embeddings']['checkpoint']
		assert config['manifests']['input'] == str(
			artifact_root
			/ 'data/parihaka/facies_benchmark_v1/'
			'parihaka_amplitude_manifest.json'
		)

	assert len(set(expected_checkpoints.values())) == 2
	assert len(set(expected_outputs.values())) == 2


def test_embedding_configs_are_scientifically_paired(
	embedding_configs: dict[str, dict[str, object]],
) -> None:
	mae = embedding_configs['mae100']
	barlow_twins = embedding_configs['bt100']

	assert mae['paths'] == barlow_twins['paths']
	assert mae['manifests'] == barlow_twins['manifests']
	assert mae['embedding'] == barlow_twins['embedding']
	assert mae['embeddings']['checkpoint'] != barlow_twins['embeddings']['checkpoint']
	assert mae['embeddings']['output_dir'] != barlow_twins['embeddings']['output_dir']
	assert mae['embedding'] == {
		'window_size': [128, 128, 128],
		'overlap': [64, 64, 64],
		'output_dtype': 'float16',
		'batch_size': 1,
		'prefetch_queue_depth': 0,
		'amp': False,
		'amp_dtype': 'auto',
		'stage_timing': False,
		'min_token_valid_fraction': 0.5,
		'preprocessing_cache': {
			'mode': 'off',
			'chunk_size_x': 16,
			'reuse': True,
			'cleanup': False,
		},
	}


def test_clustering_configs_resolve_as_paired_anchor_only_k6(
	clustering_configs: dict[str, dict[str, object]],
	embedding_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	mae = clustering_configs['mae100']
	barlow_twins = clustering_configs['bt100']

	assert mae['stage'] == barlow_twins['stage'] == 'cluster_embeddings'
	assert mae['paths'] == barlow_twins['paths']
	assert mae['embeddings']['input_dir'] != barlow_twins['embeddings']['input_dir']
	assert mae['clustering']['output_dir'] != barlow_twins['clustering']['output_dir']
	assert {
		key: value
		for key, value in mae['clustering'].items()
		if key != 'output_dir'
	} == {
		key: value
		for key, value in barlow_twins['clustering'].items()
		if key != 'output_dir'
	}

	output_roots = set()
	for variant, config in clustering_configs.items():
		assert config['embeddings']['input_dir'] == embedding_configs[variant][
			'embeddings'
		]['output_dir']
		expected_output = (
			artifact_root
			/ 'clustering/parihaka/facies_benchmark_v1'
			/ f'ssl_hmm_continuation_v1/hmm_targets/{variant}/k6'
		)
		assert Path(config['clustering']['output_dir']) == expected_output
		output_roots.add(expected_output)
	assert len(output_roots) == 2


def test_clustering_scientific_contract_is_explicit(
	clustering_configs: dict[str, dict[str, object]],
) -> None:
	clustering = clustering_configs['mae100']['clustering']
	assert clustering['embedding_normalization'] == 'l2'
	assert clustering['residualization'] == {
		'enabled': True,
		'mode': 'local_token_position',
		'group_by': 'token_phase',
		'add_global_mean_back': True,
		'min_group_count': 32,
	}
	assert clustering['pca'] == {
		'enabled': True,
		'n_components': 64,
		'whiten': False,
	}
	for key, expected in {
		'sample_tokens': 1_000_000,
		'method': 'stratigraphic_hmm_kmeans',
		'k_values': [6],
		'minibatch_size': 8192,
		'prediction_batch_size': 65536,
		'seed': 42,
	}.items():
		assert clustering[key] == expected

	hmm = clustering['stratigraphic_hmm']
	assert hmm['emission_source'] == 'embedding'
	assert hmm['iterations'] == 10
	assert hmm['z_axis'] == 2
	assert hmm['z_direction'] == 'increasing_downward'
	assert hmm['edge_margin_tokens'] == [8, 8, 0]
	assert hmm['transition'] == {
		'same_cost': 0.03,
		'advance_cost': 0.0,
		'jump_cost': 1.0,
		'reverse_cost': 1_000_000.0,
		'forbid_reverse': True,
		'max_jump': 1,
	}
	assert hmm['path_prior'] == {
		'enabled': True,
		'initial_state': {'mode': 'shallow_anchor', 'weight': 0.25},
		'terminal_state': {'mode': 'deep_anchor', 'weight': 0.25},
		'expected_boundaries': {'enabled': False},
	}
	assert 'target' not in hmm['path_prior']['expected_boundaries']
	assert hmm['init'] == {'order_by': 'mean_z'}
	assert hmm['update'] == {'empty_cluster_policy': 'keep_previous'}


def test_export_scripts_are_paired_and_export_only(
	clustering_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	arguments = {
		variant: _export_script_arguments(
			TARGET_ROOT / variant / 'k6/03_export_pseudo_targets.sh'
		)
		for variant in VARIANTS
	}

	pseudo_target_roots = set()
	for variant, args in arguments.items():
		assert args['--clustering-output-dir'] == clustering_configs[variant][
			'clustering'
		]['output_dir']
		expected_pseudo_root = (
			artifact_root
			/ 'pseudo_targets/parihaka/facies_benchmark_v1'
			/ f'ssl_hmm_continuation_v1/{variant}/k6'
		)
		assert Path(args['--pseudo-target-root']) == expected_pseudo_root
		pseudo_target_roots.add(expected_pseudo_root)
		assert args['--k'] == '6'
		assert args['--confidence'] == '1.0'
		assert args['--boundary-alpha'] == '0.0'
		assert args['--boundary-tau'] == '1.0'
		assert args['--schema-version'] == '2'

	assert len(pseudo_target_roots) == 2
	paired_flags = {
		'--k',
		'--confidence',
		'--boundary-alpha',
		'--boundary-tau',
		'--schema-version',
	}
	for flag in paired_flags:
		assert arguments['mae100'][flag] == arguments['bt100'][flag]


def test_target_pipeline_files_never_reference_25ep_controls() -> None:
	for variant in VARIANTS:
		for filename in (
			'01_extract_embeddings.yaml',
			'02_cluster_hmm_k6.yaml',
			'03_export_pseudo_targets.sh',
		):
			text = (TARGET_ROOT / variant / 'k6' / filename).read_text(
				encoding='utf-8'
			)
			assert 'full_25ep' not in text
			assert 'mae25' not in text
			assert 'bt25' not in text
			assert '/stage2/' not in text


def _export_script_arguments(path: Path) -> dict[str, str]:
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
