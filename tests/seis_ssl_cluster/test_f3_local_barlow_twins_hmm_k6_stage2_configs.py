from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_HMM_HEAD_CONTRACT,
	FIVE_WAY_HMM_LOSS_CONTRACT,
	FIVE_WAY_PRETEXT_TRAIN_CONTRACT,
	LOCAL_BARLOW_TWINS_OBJECTIVE_CONTRACT,
)

FIVE_WAY_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1'
)
LOCAL_BT_HMM_ROOT = FIVE_WAY_ROOT / '30_stage2/local_bt100/hmm/k6'
GLOBAL_HMM_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1/30_stage2'
)
LOCAL_BT_STAGE1_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1/02_full_100ep.yaml'
)
RUNS = {
	'feasibility': '01_gpu_feasibility_1step.yaml',
	'full': '02_full_25ep.yaml',
}


def _prepare_live_inputs(root: Path) -> None:
	checkpoint = (
		root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ 'local_barlow_twins_v1/full_100ep/latest.pt'
	)
	checkpoint.parent.mkdir(parents=True, exist_ok=True)
	checkpoint.touch()
	(
		root
		/ 'pseudo_targets/f3/facies_benchmark_v1'
		/ 'mae_local_bt_five_way_v1/local_bt100/k6'
	).mkdir(parents=True, exist_ok=True)
	global_checkpoint = (
		root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1/barlow_twins/full_100ep/latest.pt'
	)
	global_checkpoint.parent.mkdir(parents=True, exist_ok=True)
	global_checkpoint.touch()
	(
		root
		/ 'pseudo_targets/f3/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/bt100/k6'
	).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	_prepare_live_inputs(root)
	return root


@pytest.fixture
def local_bt_hmm_configs(artifact_root: Path) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		run: resolve_strat_hmm_pretext_config(
			load_config(LOCAL_BT_HMM_ROOT / filename)
		)
		for run, filename in RUNS.items()
	}


def test_teacher_and_student_share_the_local_bt_100ep_checkpoint(
	local_bt_hmm_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	expected_checkpoint = str(
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ 'local_barlow_twins_v1/full_100ep/latest.pt'
	)
	for config in local_bt_hmm_configs.values():
		assert config['teacher']['checkpoint'] == expected_checkpoint
		assert config['student']['init_checkpoint'] == expected_checkpoint
		assert config['student']['unfreeze_top_blocks'] == 1
		serialized = repr(
			{'teacher': config['teacher'], 'student': config['student']}
		).lower()
		for forbidden in (
			'ssl_hmm_continuation_v1',
			'/mae',
			'trace_drop',
			'best.pt',
		):
			assert forbidden not in serialized


def test_source_checkpoint_is_the_local_bt_objective(
	artifact_root: Path,
	local_bt_hmm_configs: dict[str, dict[str, object]],
) -> None:
	del artifact_root
	stage1 = resolve_barlow_twins_training_config(
		load_config(LOCAL_BT_STAGE1_CONFIG)
	)
	for key, expected in LOCAL_BARLOW_TWINS_OBJECTIVE_CONTRACT.items():
		assert stage1['barlow_twins'][key] == expected
	expected_checkpoint = f'{stage1["paths"]["output_root"]}/latest.pt'
	for config in local_bt_hmm_configs.values():
		assert config['teacher']['checkpoint'] == expected_checkpoint


def test_pseudo_target_root_is_the_local_bt_derived_root(
	local_bt_hmm_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	expected_root = str(
		artifact_root
		/ 'pseudo_targets/f3/facies_benchmark_v1'
		/ 'mae_local_bt_five_way_v1/local_bt100'
	)
	for config in local_bt_hmm_configs.values():
		assert config['pseudo_targets']['input_dir'] == expected_root
		assert config['pseudo_targets']['k'] == 6
		assert config['pseudo_targets']['min_confidence'] == 0.0


def test_head_loss_and_budget_match_the_fixed_contract(
	local_bt_hmm_configs: dict[str, dict[str, object]],
) -> None:
	for config in local_bt_hmm_configs.values():
		assert config['head'] == dict(FIVE_WAY_HMM_HEAD_CONTRACT)
		for key, expected in FIVE_WAY_HMM_LOSS_CONTRACT.items():
			assert config['loss'][key] == expected

	full = local_bt_hmm_configs['full']['train']
	for key, expected in FIVE_WAY_PRETEXT_TRAIN_CONTRACT.items():
		assert full[key] == expected
	assert full['max_steps'] is None
	assert full['allow_overwrite_output'] is False

	feasibility = local_bt_hmm_configs['feasibility']['train']
	assert feasibility['samples_per_epoch'] == 16
	assert feasibility['epochs'] == 1
	assert feasibility['num_workers'] == 0
	assert feasibility['max_steps'] == 1


def test_configs_match_global_bt_hmm_science_except_paths(
	artifact_root: Path,
) -> None:
	del artifact_root
	for filename in RUNS.values():
		local = load_config(LOCAL_BT_HMM_ROOT / filename)
		standard = load_config(GLOBAL_HMM_ROOT / 'bt100/hmm/k6' / filename)

		comparison = deepcopy(local)
		comparison['paths']['output_root'] = standard['paths']['output_root']
		comparison['pseudo_targets']['input_dir'] = standard['pseudo_targets'][
			'input_dir'
		]
		comparison['teacher']['checkpoint'] = standard['teacher']['checkpoint']
		comparison['student']['init_checkpoint'] = standard['student'][
			'init_checkpoint'
		]
		assert comparison == standard


def test_existing_mae_and_global_bt_hmm_configs_are_unchanged(
	artifact_root: Path,
) -> None:
	del artifact_root
	for variant, source in (
		('mae100', 'mae'),
		('bt100', 'barlow_twins'),
	):
		for filename in RUNS.values():
			raw = load_config(GLOBAL_HMM_ROOT / variant / 'hmm/k6' / filename)
			assert raw['teacher']['checkpoint'].endswith(
				f'ssl_hmm_continuation_v1/stage1/{source}/full_100ep/latest.pt'
			)
			assert raw['pseudo_targets']['input_dir'].endswith(
				f'ssl_hmm_continuation_v1/{variant}'
			)
			assert 'mae_local_bt_five_way_v1' not in repr(raw)


def test_output_roots_are_separated_and_do_not_collide(
	local_bt_hmm_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	expected_prefix = (
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1'
		/ 'mae_local_bt_five_way_v1/stage2/local_bt100/hmm/k6'
	)
	feasibility_root = Path(
		local_bt_hmm_configs['feasibility']['paths']['output_root']
	)
	full_root = Path(local_bt_hmm_configs['full']['paths']['output_root'])
	assert feasibility_root == expected_prefix / 'gpu_feasibility_1step'
	assert full_root == expected_prefix / 'full_25ep'
	assert 'ssl_hmm_continuation_v1' not in str(full_root)


def test_raw_configs_do_not_contain_trace_drop_settings() -> None:
	for filename in RUNS.values():
		text = (LOCAL_BT_HMM_ROOT / filename).read_text(encoding='utf-8')
		lowered = text.lower()
		assert 'trace_drop' not in lowered
		assert 'reflection_probability' not in lowered
		assert 'policy' not in lowered


def test_dry_run_creates_no_artifacts(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	_prepare_live_inputs(artifact_root)
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	before = {str(path) for path in artifact_root.rglob('*')}
	for filename in RUNS.values():
		result = subprocess.run(  # noqa: S603
			[
				sys.executable,
				'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
				'--config',
				str(LOCAL_BT_HMM_ROOT / filename),
				'--dry-run',
			],
			check=True,
			capture_output=True,
			text=True,
			env=environment,
		)
		assert 'execution: dry-run; training skipped' in result.stdout
	after = {str(path) for path in artifact_root.rglob('*')}
	assert after == before
