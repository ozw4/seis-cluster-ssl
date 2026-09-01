from __future__ import annotations

import os
import re
import shlex
import stat
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
	resolve_vicreg_training_config,
)

ROOT = Path('experiments/f3/facies_benchmark_v2/115_local_vicreg_v1')
LOCAL_BT_STAGE1 = Path(
	'experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1'
)
LOCAL_BT_FIVE_WAY = Path(
	'experiments/f3/facies_benchmark_v1/'
	'110_lithology_mae_local_bt_five_way_v1'
)
BASELINE_CONFIGS = (
	'01_gpu_feasibility_1step.yaml',
	'02_full_100ep.yaml',
)
CONTROL_ROOT = ROOT / '10_stage2/vicreg100/vicreg_continue'
CONTROL_CONFIGS = BASELINE_CONFIGS[0], '02_full_25ep.yaml'
TARGET_ROOT = ROOT / '20_hmm_targets/vicreg100'
HMM_ROOT = ROOT / '30_stage2/vicreg100/hmm/k6'
HMM_CONFIGS = BASELINE_CONFIGS[0], '02_full_25ep.yaml'
README = ROOT / 'README.md'
VICREG_CONTRACT = {
	'method': 'local_vicreg_3d',
	'local_pairs_per_crop': 128,
	'projector_dim': 384,
	'invariance_weight': 25.0,
	'variance_weight': 25.0,
	'covariance_weight': 1.0,
	'variance_target_std': 1.0,
	'variance_eps': 1.0e-4,
}
EXTRACTION_CONTRACT = {
	'window_size': [128, 128, 128],
	'overlap': [64, 64, 64],
	'output_dtype': 'float16',
	'batch_size': 1,
	'amp': False,
	'min_token_valid_fraction': 0.5,
}
RUNBOOK_CLIS = (
	'proc/seis_ssl_cluster/train_amp_vicreg.py',
	'proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py',
	'proc/seis_ssl_cluster/extract_embeddings.py',
	'proc/seis_ssl_cluster/cluster_embeddings.py',
	'proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py',
	'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
)


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	checkpoint = (
		root
		/ 'pretraining/f3/facies_benchmark_v1/local_vicreg_v1/'
		'full_100ep/latest.pt'
	)
	checkpoint.parent.mkdir(parents=True)
	checkpoint.touch()
	local_bt_checkpoint = (
		root
		/ 'pretraining/f3/facies_benchmark_v1/local_barlow_twins_v1/'
		'full_100ep/latest.pt'
	)
	local_bt_checkpoint.parent.mkdir(parents=True)
	local_bt_checkpoint.touch()
	(
		root
		/ 'pseudo_targets/f3/facies_benchmark_v1/local_vicreg_v1/'
		'vicreg100/k6'
	).mkdir(parents=True)
	(
		root
		/ 'pseudo_targets/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/'
		'local_bt100/k6'
	).mkdir(parents=True)
	return root


def _resolved_vicreg(root: Path, filename: str) -> dict[str, object]:
	return resolve_vicreg_training_config(load_config(root / filename))


def _resolved_hmm(filename: str) -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(HMM_ROOT / filename))


@pytest.mark.parametrize('filename', BASELINE_CONFIGS)
def test_baseline_is_loss_only_delta_from_local_barlow_twins(
	filename: str,
) -> None:
	vicreg = _resolved_vicreg(ROOT, filename)
	barlow = resolve_barlow_twins_training_config(
		load_config(LOCAL_BT_STAGE1 / filename)
	)

	assert vicreg['stage'] == 'vicreg_training'
	assert barlow['stage'] == 'barlow_twins_training'
	assert vicreg['vicreg'] == VICREG_CONTRACT
	assert 'barlow_twins' not in vicreg
	for section in (
		'manifests',
		'data',
		'zero_mask',
		'model',
		'augmentations',
		'train',
	):
		assert vicreg[section] == barlow[section]

	comparison = deepcopy(vicreg)
	comparison['stage'] = barlow['stage']
	comparison['paths']['output_root'] = barlow['paths']['output_root']
	comparison['barlow_twins'] = barlow['barlow_twins']
	del comparison['vicreg']
	assert comparison == barlow


def test_baseline_full_literals_and_budget_are_pinned() -> None:
	full = _resolved_vicreg(ROOT, '02_full_100ep.yaml')
	train = full['train']

	assert full['data']['local_crop_size'] == [128, 128, 128]
	assert full['data']['min_valid_fraction'] == 0.1
	assert full['data']['max_resample_attempts'] == 16
	assert full['data']['normalized_clip_abs'] == 8.0
	assert full['data']['finite_check_mode'] == 'strict'
	assert full['data']['amplitude_agc'] == {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 65,
		'eps': 1.0e-3,
		'clip_abs': 5.0,
	}
	assert full['model']['patch_size'] == [8, 8, 8]
	assert full['model']['encoder_dim'] == 384
	assert full['model']['encoder_depth'] == 8
	assert full['model']['encoder_heads'] == 6
	assert full['model']['decoder_dim'] == 256
	assert full['model']['decoder_depth'] == 4
	assert full['model']['decoder_heads'] == 4
	assert full['augmentations'] == {'horizontal_flip_probability': 0.5}
	assert train['batch_size'] == 16
	assert train['samples_per_epoch'] == 10_000
	assert train['epochs'] == 100
	assert train['lr'] == 1.0e-4
	assert train['weight_decay'] == 0.05
	assert train['amp'] is False
	assert train['seed'] == 42
	assert train['grad_clip_norm'] == 1.0
	assert train['epochs'] * train['samples_per_epoch'] // train['batch_size'] == (
		62_500
	)


def test_baseline_outputs_and_manifests_have_explicit_versions(
	artifact_root: Path,
) -> None:
	feasibility = _resolved_vicreg(ROOT, BASELINE_CONFIGS[0])
	full = _resolved_vicreg(ROOT, BASELINE_CONFIGS[1])
	extract = resolve_embedding_extraction_config(
		load_config(ROOT / '03_extract_v2_embeddings.yaml')
	)
	expected_checkpoint = (
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1/local_vicreg_v1/'
		'full_100ep/latest.pt'
	)

	assert feasibility['paths']['output_root'] != full['paths']['output_root']
	assert '/facies_benchmark_v1/' in full['manifests']['train']
	assert '/facies_benchmark_v1/' in full['manifests']['train_path_list']
	assert Path(extract['embeddings']['checkpoint']) == expected_checkpoint
	assert '/facies_benchmark_v2/' in extract['manifests']['input']
	assert Path(extract['embeddings']['output_dir']) == (
		artifact_root
		/ 'embeddings/f3/facies_benchmark_v2/local_vicreg_v1/'
		'base100/overlap_x64'
	)
	for key, expected in EXTRACTION_CONTRACT.items():
		assert extract['embedding'][key] == expected
	assert set(extract['embeddings']) == {'checkpoint', 'output_dir'}


@pytest.mark.parametrize('filename', CONTROL_CONFIGS)
def test_control_reuses_vicreg100_loss_view_and_projector(
	filename: str,
	artifact_root: Path,
) -> None:
	control = _resolved_vicreg(CONTROL_ROOT, filename)
	stage1 = _resolved_vicreg(
		ROOT,
		BASELINE_CONFIGS[0] if filename.startswith('01_') else BASELINE_CONFIGS[1],
	)
	expected_checkpoint = (
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1/local_vicreg_v1/'
		'full_100ep/latest.pt'
	)

	assert Path(control['continuation']['init_checkpoint']) == expected_checkpoint
	assert control['continuation']['unfreeze_top_blocks'] == 1
	assert control['vicreg'] == stage1['vicreg'] == VICREG_CONTRACT
	assert control['augmentations'] == stage1['augmentations']
	assert control['model'] == stage1['model']
	assert control['data'] == stage1['data']
	assert control['zero_mask'] == stage1['zero_mask']


def test_control_full_has_exact_fixed_budget() -> None:
	full = _resolved_vicreg(CONTROL_ROOT, '02_full_25ep.yaml')
	train = full['train']

	assert train['batch_size'] == 16
	assert train['samples_per_epoch'] == 10_000
	assert train['epochs'] == 25
	assert train.get('max_steps') is None
	assert train['epochs'] * train['samples_per_epoch'] // train['batch_size'] == (
		15_625
	)
	assert train['lr'] == 1.0e-5
	assert train['weight_decay'] == 0.05
	assert train['amp'] is False
	assert train['seed'] == 42


@pytest.mark.parametrize('filename', CONTROL_CONFIGS)
def test_control_matches_local_bt_fixed_budget_except_objective_and_paths(
	filename: str,
) -> None:
	vicreg = load_config(CONTROL_ROOT / filename)
	barlow = load_config(
		LOCAL_BT_FIVE_WAY / '10_stage2/local_bt100/local_bt_continue' / filename
	)
	comparison = deepcopy(vicreg)
	comparison['paths']['output_root'] = barlow['paths']['output_root']
	comparison['continuation']['init_checkpoint'] = barlow['continuation'][
		'init_checkpoint'
	]
	comparison['barlow_twins'] = barlow['barlow_twins']
	del comparison['vicreg']
	assert comparison == barlow


def test_hmm_target_source_and_science_match_local_bt_k6(
	artifact_root: Path,
) -> None:
	vicreg_extract = resolve_embedding_extraction_config(
		load_config(TARGET_ROOT / '01_extract_embeddings.yaml')
	)
	barlow_extract = resolve_embedding_extraction_config(
		load_config(
			LOCAL_BT_FIVE_WAY
			/ '20_hmm_targets/local_bt100/01_extract_embeddings.yaml'
		)
	)
	vicreg_cluster = resolve_clustering_config(
		load_config(TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml')
	)
	barlow_cluster = resolve_clustering_config(
		load_config(
			LOCAL_BT_FIVE_WAY
			/ '20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml'
		)
	)

	assert Path(vicreg_extract['embeddings']['checkpoint']) == (
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1/local_vicreg_v1/'
		'full_100ep/latest.pt'
	)
	assert vicreg_extract['manifests'] == barlow_extract['manifests']
	assert vicreg_extract['embedding'] == barlow_extract['embedding']
	assert (
		vicreg_cluster['embeddings']['input_dir']
		== vicreg_extract['embeddings']['output_dir']
	)
	assert {
		key: value
		for key, value in vicreg_cluster['clustering'].items()
		if key != 'output_dir'
	} == {
		key: value
		for key, value in barlow_cluster['clustering'].items()
		if key != 'output_dir'
	}
	assert vicreg_cluster['clustering']['residualization']['mode'] == (
		'local_token_position'
	)
	assert vicreg_cluster['clustering']['pca']['n_components'] == 64
	assert vicreg_cluster['clustering']['method'] == 'stratigraphic_hmm_kmeans'
	assert vicreg_cluster['clustering']['k_values'] == [6]


def test_export_script_is_executable_valid_and_pins_schema2_k6(
	artifact_root: Path,
) -> None:
	del artifact_root
	path = TARGET_ROOT / 'k6/03_export_pseudo_targets.sh'
	text = path.read_text(encoding='utf-8')
	assert text.startswith('#!/usr/bin/env bash\n')
	assert 'set -euo pipefail' in text
	assert path.stat().st_mode & stat.S_IXUSR
	subprocess.run(  # noqa: S603
		['bash', '-n', str(path)],  # noqa: S607
		check=True,
		capture_output=True,
		text=True,
	)
	command_lines = [
		line.strip().removesuffix('\\').strip()
		for line in text.splitlines()
		if line.strip()
		and not line.startswith('#!')
		and line.strip() != 'set -euo pipefail'
		and line.strip() != '"$@"'
	]
	tokens = shlex.split(os.path.expandvars(' '.join(command_lines)))
	assert tokens[:2] == [
		'python',
		'proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py',
	]
	arguments = dict(zip(tokens[2::2], tokens[3::2], strict=True))
	assert arguments['--k'] == '6'
	assert arguments['--confidence'] == '1.0'
	assert arguments['--boundary-alpha'] == '0.0'
	assert arguments['--boundary-tau'] == '1.0'
	assert arguments['--schema-version'] == '2'
	assert arguments['--clustering-output-dir'].endswith(
		'local_vicreg_v1/hmm_targets/vicreg100/k6'
	)
	assert arguments['--pseudo-target-root'].endswith(
		'local_vicreg_v1/vicreg100'
	)
	assert '"$@"' in text


@pytest.mark.parametrize('filename', HMM_CONFIGS)
def test_hmm_teacher_student_and_science_match_local_bt(
	filename: str,
	artifact_root: Path,
) -> None:
	vicreg = _resolved_hmm(filename)
	barlow = resolve_strat_hmm_pretext_config(
		load_config(
			LOCAL_BT_FIVE_WAY / '30_stage2/local_bt100/hmm/k6' / filename
		)
	)
	expected_checkpoint = str(
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1/local_vicreg_v1/'
		'full_100ep/latest.pt'
	)

	assert vicreg['teacher']['checkpoint'] == expected_checkpoint
	assert vicreg['student']['init_checkpoint'] == expected_checkpoint
	assert vicreg['student']['unfreeze_top_blocks'] == 1
	assert vicreg['head'] == barlow['head']
	assert vicreg['loss'] == barlow['loss']
	assert vicreg['train'] == barlow['train']
	assert vicreg['data'] == barlow['data']
	assert vicreg['model'] == barlow['model']
	assert vicreg['zero_mask'] == barlow['zero_mask']
	assert vicreg['pseudo_targets']['k'] == 6
	assert vicreg['pseudo_targets']['min_confidence'] == 0.0


def test_hmm_full_has_exact_fixed_budget_and_losses() -> None:
	full = _resolved_hmm('02_full_25ep.yaml')
	train = full['train']

	assert train['batch_size'] == 16
	assert train['samples_per_epoch'] == 10_000
	assert train['epochs'] == 25
	assert train['epochs'] * train['samples_per_epoch'] // train['batch_size'] == (
		15_625
	)
	assert train['lr'] == 1.0e-5
	assert train['encoder_lr'] == 1.0e-5
	assert train['weight_decay'] == 0.05
	assert train['amp'] is False
	assert train['seed'] == 42
	assert full['loss']['prototype_weight'] == 1.0
	assert full['loss']['usage_weight'] == 0.005
	assert full['loss']['distillation_weight'] == 0.2


def test_hmm_artifact_dependency_paths_are_closed(artifact_root: Path) -> None:
	stage1 = _resolved_vicreg(ROOT, '02_full_100ep.yaml')
	control = _resolved_vicreg(CONTROL_ROOT, '02_full_25ep.yaml')
	extract = resolve_embedding_extraction_config(
		load_config(TARGET_ROOT / '01_extract_embeddings.yaml')
	)
	cluster = resolve_clustering_config(
		load_config(TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml')
	)
	hmm = _resolved_hmm('02_full_25ep.yaml')

	stage1_checkpoint = f'{stage1["paths"]["output_root"]}/latest.pt'
	assert control['continuation']['init_checkpoint'] == stage1_checkpoint
	assert extract['embeddings']['checkpoint'] == stage1_checkpoint
	assert hmm['teacher']['checkpoint'] == stage1_checkpoint
	assert hmm['student']['init_checkpoint'] == stage1_checkpoint
	assert extract['embeddings']['output_dir'] == cluster['embeddings']['input_dir']
	assert Path(hmm['pseudo_targets']['input_dir']) == (
		artifact_root
		/ 'pseudo_targets/f3/facies_benchmark_v1/local_vicreg_v1/vicreg100'
	)
	assert control['paths']['output_root'] != hmm['paths']['output_root']


def test_configs_do_not_mix_unrequested_augmentations_or_sources() -> None:
	for path in sorted(ROOT.rglob('*.yaml')):
		text = path.read_text(encoding='utf-8').lower()
		for forbidden in (
			'trace_drop',
			'reflection_probability',
			'gaussian_noise',
			'zero_phase_z_filter',
			'd4',
		):
			assert forbidden not in text, path
	assert 'full_25ep' not in (
		TARGET_ROOT / '01_extract_embeddings.yaml'
	).read_text(encoding='utf-8')


def test_runbook_references_existing_clis_configs_and_gate() -> None:
	text = README.read_text(encoding='utf-8')
	for cli in RUNBOOK_CLIS:
		assert Path(cli).is_file()
		assert cli in text
	for path in sorted(ROOT.rglob('*')):
		if path.suffix in {'.yaml', '.sh'}:
			assert path.name in text
	assert 'VICREG_BASELINE_GATE_PASS' in text
	assert 'VICREG_BASELINE_GATE_FAIL' in text
	assert '62,500' in text
	assert '15,625' in text
	assert 'local_vicreg_screen_v1/summary/summary.json' in text
	assert 'facies_benchmark_v1' in text
	assert 'facies_benchmark_v2' in text


def test_runbook_stage1_order_and_shell_blocks_are_valid(tmp_path: Path) -> None:
	text = README.read_text(encoding='utf-8')
	headings = (
		'### 1. Config tests',
		'### 2. 1-step dry-run',
		'### 3. 1-step live',
		'### 4. Full 100 epoch dry-run',
		'### 5. Full 100 epoch live',
		'### 6. Checkpoint and prepared-volume audit',
		'### 7. F3 v2 embedding dry-run',
		'### 8. F3 v2 embedding live',
	)
	positions = [text.index(heading) for heading in headings]
	assert positions == sorted(positions)
	blocks = re.findall(r'```bash\n(.*?)```', text, flags=re.DOTALL)
	assert blocks
	for index, block in enumerate(blocks):
		path = tmp_path / f'runbook_block_{index}.sh'
		path.write_text(block, encoding='utf-8')
		subprocess.run(  # noqa: S603
			['bash', '-n', str(path)],  # noqa: S607
			check=True,
			capture_output=True,
			text=True,
		)


def test_all_yaml_clis_support_no_write_dry_runs(
	artifact_root: Path,
) -> None:
	commands = [
		(
			'proc/seis_ssl_cluster/train_amp_vicreg.py',
			ROOT / '01_gpu_feasibility_1step.yaml',
		),
		(
			'proc/seis_ssl_cluster/train_amp_vicreg.py',
			ROOT / '02_full_100ep.yaml',
		),
		(
			'proc/seis_ssl_cluster/extract_embeddings.py',
			ROOT / '03_extract_v2_embeddings.yaml',
		),
		*(
			('proc/seis_ssl_cluster/train_amp_vicreg.py', CONTROL_ROOT / name)
			for name in CONTROL_CONFIGS
		),
		(
			'proc/seis_ssl_cluster/extract_embeddings.py',
			TARGET_ROOT / '01_extract_embeddings.yaml',
		),
		(
			'proc/seis_ssl_cluster/cluster_embeddings.py',
			TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml',
		),
		*(
			('proc/seis_ssl_cluster/train_strat_hmm_pretext.py', HMM_ROOT / name)
			for name in HMM_CONFIGS
		),
	]
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	before = {
		str(path): path.stat().st_size
		for path in artifact_root.rglob('*')
		if path.is_file()
	}
	for script, config in commands:
		result = subprocess.run(  # noqa: S603
			[sys.executable, script, '--config', str(config), '--dry-run'],
			check=True,
			capture_output=True,
			text=True,
			env=environment,
		)
		assert 'dry-run' in result.stdout
	after = {
		str(path): path.stat().st_size
		for path in artifact_root.rglob('*')
		if path.is_file()
	}
	assert after == before
