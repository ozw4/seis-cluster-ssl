from __future__ import annotations

import os
import shlex
import stat
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
	resolve_f3_lithology_candidate_job,
)

ROOT = Path('experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1')
WINNER_CONFIG = ROOT / '10_pretraining/rot90_asym_g060_3ep.yaml'
CONTROL_EMBEDDING_CONFIG = ROOT / '20_embeddings/rot90_asym_g060_3ep.yaml'
TARGET_ROOT = ROOT / '40_hmm_targets/rot90_asym_g060_3ep'
TARGET_EMBEDDING_CONFIG = TARGET_ROOT / '01_extract_embeddings.yaml'
CLUSTER_CONFIG = TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml'
EXPORT_SCRIPT = TARGET_ROOT / 'k6/03_export_pseudo_targets.sh'
HMM_ROOT = ROOT / '50_hmm_pretraining/rot90_asym_g060_3ep/hmm/k6'
HMM_CONFIGS = ('01_gpu_feasibility_1step.yaml', '02_full_25ep.yaml')
FINAL_EMBEDDING_CONFIG = ROOT / '60_hmm_embeddings/rot90_asym_g060_3ep_hmm_k6_25ep.yaml'
DOWNSTREAM_CONFIG = ROOT / '70_hmm_downstream/rot90_asym_g060_3ep_hmm_k6_25ep.yaml'
SUMMARY_CONFIG = ROOT / '80_hmm_summary/01_hmm_k6_25ep_vs_rot90_asym_g060_3ep.yaml'
REFERENCE_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1'
)
REFERENCE_CLUSTER_CONFIG = (
	REFERENCE_ROOT / '20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml'
)
REFERENCE_HMM_ROOT = REFERENCE_ROOT / '30_stage2/local_bt100/hmm/k6'
CANONICAL_DOWNSTREAM_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
CONTROL_ID = 'local_bt_nr_rot90_asym_g060_3ep'
CANDIDATE_ID = f'{CONTROL_ID}_hmm_k6_25ep'
HMM_MODEL_TAGS = {
	'01_gpu_feasibility_1step.yaml': (
		'f3_local_bt_nr_rot90_asym_g060_3ep_hmm_k6_gpu_feasibility_1step'
	),
	'02_full_25ep.yaml': 'f3_local_bt_nr_rot90_asym_g060_3ep_hmm_k6_full_25ep',
}
EXTRACTION_CONTRACT = {
	'window_size': [128, 128, 128],
	'overlap': [64, 64, 64],
	'output_dtype': 'float16',
	'batch_size': 1,
	'amp': False,
	'min_token_valid_fraction': 0.5,
}


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	winner_checkpoint = (
		root / 'pretraining/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/'
		'rot90_asym_g060/full_3ep/latest.pt'
	)
	winner_checkpoint.parent.mkdir(parents=True)
	winner_checkpoint.touch()
	(
		root / 'pseudo_targets/f3/facies_benchmark_v1/'
		'local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep/k6'
	).mkdir(parents=True)
	return root


def _resolved_hmm(filename: str) -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(HMM_ROOT / filename))


def _export_arguments() -> tuple[str, dict[str, str]]:
	text = EXPORT_SCRIPT.read_text(encoding='utf-8')
	command_lines = [
		line.strip().removesuffix('\\').strip()
		for line in text.splitlines()
		if line.strip()
		and not line.startswith('#!')
		and line.strip() != 'set -euo pipefail'
		and line.strip() != '"$@"'
	]
	tokens = shlex.split(os.path.expandvars(' '.join(command_lines)))
	arguments = dict(zip(tokens[2::2], tokens[3::2], strict=True))
	return text, arguments


def test_hmm_branch_inventory_is_complete() -> None:
	paths = (
		TARGET_EMBEDDING_CONFIG,
		CLUSTER_CONFIG,
		EXPORT_SCRIPT,
		*(HMM_ROOT / filename for filename in HMM_CONFIGS),
		FINAL_EMBEDDING_CONFIG,
		DOWNSTREAM_CONFIG,
		SUMMARY_CONFIG,
	)

	assert all(path.is_file() for path in paths)
	assert len(paths) == 8


def test_hmm_runbook_names_the_concrete_paired_summary_config() -> None:
	readme = (ROOT / 'README.md').read_text(encoding='utf-8')

	assert str(SUMMARY_CONFIG.relative_to(ROOT)) in readme
	assert '01_hmm_k6_25ep_vs_3ep.yaml' not in readme


def test_hmm_target_is_derived_from_the_exact_three_epoch_winner(
	artifact_root: Path,
) -> None:
	winner = resolve_barlow_twins_training_config(load_config(WINNER_CONFIG))
	target = resolve_embedding_extraction_config(load_config(TARGET_EMBEDDING_CONFIG))
	expected_checkpoint = (
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/'
		'rot90_asym_g060/full_3ep/latest.pt'
	)

	assert winner['train']['epochs'] == 3
	assert winner['augmentations'] == {
		'policy': 'xy_rot90_asymmetric_noise_v1',
		'gaussian_noise_std': 0.6,
	}
	assert winner['barlow_twins']['method'] == 'local_barlow_twins_3d'
	assert winner['barlow_twins']['positive_window_tokens'] == [2, 2, 1]
	assert Path(f'{winner["paths"]["output_root"]}/latest.pt') == (expected_checkpoint)
	assert Path(target['embeddings']['checkpoint']) == expected_checkpoint
	assert '/facies_benchmark_v1/' in target['manifests']['input']
	for key, expected in EXTRACTION_CONTRACT.items():
		assert target['embedding'][key] == expected


def test_hmm_clustering_uses_the_standard_f3_k6_science() -> None:
	target = resolve_embedding_extraction_config(load_config(TARGET_EMBEDDING_CONFIG))
	cluster = resolve_clustering_config(load_config(CLUSTER_CONFIG))
	reference = resolve_clustering_config(load_config(REFERENCE_CLUSTER_CONFIG))
	science = {
		key: value
		for key, value in cluster['clustering'].items()
		if key != 'output_dir'
	}
	reference_science = {
		key: value
		for key, value in reference['clustering'].items()
		if key != 'output_dir'
	}

	assert cluster['embeddings']['input_dir'] == target['embeddings']['output_dir']
	assert science == reference_science
	assert science['embedding_normalization'] == 'l2'
	assert science['residualization'] == {
		'enabled': True,
		'mode': 'local_token_position',
		'group_by': 'token_phase',
		'add_global_mean_back': True,
		'min_group_count': 32,
	}
	assert science['pca'] == {
		'enabled': True,
		'n_components': 64,
		'whiten': False,
	}
	assert science['method'] == 'stratigraphic_hmm_kmeans'
	assert science['k_values'] == [6]
	assert science['seed'] == 42
	assert science['stratigraphic_hmm']['z_axis'] == 2
	assert science['stratigraphic_hmm']['z_direction'] == 'increasing_downward'
	assert science['stratigraphic_hmm']['update']['empty_cluster_policy'] == (
		'keep_previous'
	)


def test_export_script_pins_k6_schema2_and_forwards_cli_arguments() -> None:
	text, arguments = _export_arguments()

	assert text.startswith('#!/usr/bin/env bash\n')
	assert 'set -euo pipefail' in text
	assert EXPORT_SCRIPT.stat().st_mode & stat.S_IXUSR
	subprocess.run(  # noqa: S603
		['bash', '-n', str(EXPORT_SCRIPT)],  # noqa: S607
		check=True,
		capture_output=True,
		text=True,
	)
	assert arguments['--k'] == '6'
	assert arguments['--confidence'] == '1.0'
	assert arguments['--boundary-alpha'] == '0.0'
	assert arguments['--boundary-tau'] == '1.0'
	assert arguments['--schema-version'] == '2'
	assert arguments['--clustering-output-dir'].endswith(
		'/local_bt_noise_rotation_search_v1/hmm_targets/rot90_asym_g060_3ep/k6'
	)
	assert arguments['--pseudo-target-root'].endswith(
		'/local_bt_noise_rotation_search_v1/rot90_asym_g060_3ep'
	)
	assert text.count('"$@"') == 1
	assert text.rstrip().endswith('"$@"')


@pytest.mark.parametrize('filename', HMM_CONFIGS)
def test_hmm_training_uses_the_winner_as_teacher_and_student(
	filename: str,
	artifact_root: Path,
) -> None:
	hmm = _resolved_hmm(filename)
	expected_checkpoint = str(
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1/local_bt_noise_rotation_search_v1/'
		'rot90_asym_g060/full_3ep/latest.pt'
	)

	assert hmm['stage'] == 'train_strat_hmm_pretext'
	assert hmm['teacher']['checkpoint'] == expected_checkpoint
	assert hmm['student']['init_checkpoint'] == expected_checkpoint
	assert hmm['student']['unfreeze_top_blocks'] == 1
	assert hmm['identity']['model_tag'] == HMM_MODEL_TAGS[filename]
	assert hmm['pseudo_targets']['k'] == 6
	assert hmm['pseudo_targets']['min_confidence'] == 0.0
	assert hmm['head'] == {
		'num_prototypes': 6,
		'projection_dim': 128,
		'temperature': 0.1,
		'normalize': True,
	}
	assert hmm['loss']['prototype_weight'] == 1.0
	assert hmm['loss']['usage_weight'] == 0.005
	assert hmm['loss']['entropy_floor'] is None
	assert hmm['loss']['distillation_weight'] == 0.2


@pytest.mark.parametrize('filename', HMM_CONFIGS)
def test_hmm_training_matches_the_adopted_local_bt_k6_contract(
	filename: str,
) -> None:
	candidate = load_config(HMM_ROOT / filename)
	reference = load_config(REFERENCE_HMM_ROOT / filename)

	for section in ('manifests', 'data', 'zero_mask', 'model', 'head', 'loss', 'train'):
		assert candidate[section] == reference[section]
	assert (
		candidate['student']['unfreeze_top_blocks']
		== (reference['student']['unfreeze_top_blocks'])
	)
	assert candidate['pseudo_targets']['k'] == reference['pseudo_targets']['k'] == 6
	assert (
		candidate['pseudo_targets']['min_confidence']
		== (reference['pseudo_targets']['min_confidence'])
	)


def test_hmm_full_has_exact_25_epoch_budget() -> None:
	full = _resolved_hmm('02_full_25ep.yaml')
	train = full['train']

	assert train['batch_size'] == 16
	assert train['samples_per_epoch'] == 10_000
	assert train['epochs'] == 25
	assert train['epochs'] * train['samples_per_epoch'] // train['batch_size'] == (
		15_625
	)
	assert train['num_workers'] == 8
	assert train['lr'] == 1.0e-5
	assert train['encoder_lr'] == 1.0e-5
	assert train['weight_decay'] == 0.05
	assert train['amp'] is False
	assert train['seed'] == 42
	assert train['max_steps'] is None
	assert train['allow_overwrite_output'] is False


def test_feasibility_diff_is_limited_to_runtime_and_output() -> None:
	feasibility = load_config(HMM_ROOT / HMM_CONFIGS[0])
	full = load_config(HMM_ROOT / HMM_CONFIGS[1])
	expected = deepcopy(full)
	expected['paths']['output_root'] = feasibility['paths']['output_root']
	expected['train']['samples_per_epoch'] = 16
	expected['train']['epochs'] = 1
	expected['train']['num_workers'] = 0
	expected['train']['max_steps'] = 1
	expected['identity']['model_tag'] = feasibility['identity']['model_tag']

	assert feasibility == expected
	assert feasibility['paths']['output_root'].endswith(
		'/rot90_asym_g060_3ep/hmm/k6/gpu_feasibility_1step'
	)
	assert full['paths']['output_root'].endswith(
		'/rot90_asym_g060_3ep/hmm/k6/full_25ep'
	)


def test_hmm_artifact_dependency_chain_and_output_paths_are_unique(
	artifact_root: Path,
) -> None:
	winner = resolve_barlow_twins_training_config(load_config(WINNER_CONFIG))
	control_embedding = resolve_embedding_extraction_config(
		load_config(CONTROL_EMBEDDING_CONFIG)
	)
	target = resolve_embedding_extraction_config(load_config(TARGET_EMBEDDING_CONFIG))
	cluster = resolve_clustering_config(load_config(CLUSTER_CONFIG))
	hmm = _resolved_hmm('02_full_25ep.yaml')
	final_embedding = resolve_embedding_extraction_config(
		load_config(FINAL_EMBEDDING_CONFIG)
	)
	downstream = f3_lithology_candidate_config_from_mapping(
		load_config(DOWNSTREAM_CONFIG)
	)
	_, export_arguments = _export_arguments()
	winner_checkpoint = f'{winner["paths"]["output_root"]}/latest.pt'
	hmm_checkpoint = f'{hmm["paths"]["output_root"]}/latest.pt'

	assert target['embeddings']['checkpoint'] == winner_checkpoint
	assert target['embeddings']['output_dir'] == cluster['embeddings']['input_dir']
	assert (
		cluster['clustering']['output_dir']
		== export_arguments['--clustering-output-dir']
	)
	assert (
		hmm['pseudo_targets']['input_dir'] == export_arguments['--pseudo-target-root']
	)
	assert hmm['teacher']['checkpoint'] == winner_checkpoint
	assert hmm['student']['init_checkpoint'] == winner_checkpoint
	assert final_embedding['embeddings']['checkpoint'] == hmm_checkpoint
	assert downstream.checkpoint == Path(hmm_checkpoint)
	assert downstream.embeddings_dir == Path(
		final_embedding['embeddings']['output_dir']
	)
	assert '/facies_benchmark_v2/' in final_embedding['manifests']['input']
	assert Path(target['embeddings']['output_dir']) != Path(
		control_embedding['embeddings']['output_dir']
	)
	assert Path(final_embedding['embeddings']['output_dir']) != Path(
		target['embeddings']['output_dir']
	)
	assert Path(hmm['paths']['output_root']) != Path(winner['paths']['output_root'])
	assert downstream.summary_root == (
		artifact_root
		/ 'f3_lithology_benchmark/local_bt_noise_rotation_search_v1/summary/'
		/ CANDIDATE_ID
	)


def test_downstream_config_resolves_all_15_canonical_v3_cells() -> None:
	config = f3_lithology_candidate_config_from_mapping(load_config(DOWNSTREAM_CONFIG))
	canonical = load_f3_lithology_candidate_canonical_config(config)

	assert config.canonical_config == Path.cwd() / CANONICAL_DOWNSTREAM_CONFIG
	assert config.candidate_id == CANDIDATE_ID
	assert config.candidate_id != CONTROL_ID
	assert config.candidate_id not in canonical.model_ids
	jobs = [
		resolve_f3_lithology_candidate_job(
			config,
			canonical,
			layout=layout,
			size=size,
		)
		for layout in LAYOUT_IDS
		for size in DATA_SIZES
	]
	assert len(jobs) == 15
	assert len({job.output_dir for job in jobs}) == 15
	assert all(f'model={CANDIDATE_ID}' in str(job.output_dir) for job in jobs)
