from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_five_way import (
	EXPECTED_MODEL_IDENTITIES,
	FIVE_WAY_HMM_HEAD_CONTRACT,
	FIVE_WAY_HMM_LOSS_CONTRACT,
	FIVE_WAY_MODEL_IDS,
	FIVE_WAY_PRETEXT_TRAIN_CONTRACT,
	FIVE_WAY_STAGE1_TRAIN_CONTRACT,
	FIVE_WAY_STAGE2_TRAIN_CONTRACT,
	LOCAL_BARLOW_TWINS_OBJECTIVE_CONTRACT,
	MAE_LOSS_CONTRACT,
	MAE_MASKING_CONTRACT,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)

EXP_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1'
)
README = EXP_ROOT / 'README.md'
RUNBOOK_CLIS = (
	'proc/seis_ssl_cluster/train_amp_barlow_twins.py',
	'proc/seis_ssl_cluster/extract_embeddings.py',
	'proc/seis_ssl_cluster/cluster_embeddings.py',
	'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
	'proc/seis_ssl_cluster/create_random_mae_checkpoint.py',
	'proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py',
	'proc/seis_ssl_cluster/run_f3_lithology_five_way.py',
	'proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py',
)


@pytest.fixture
def env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3_root'))
	return root


def _readme_text() -> str:
	return README.read_text(encoding='utf-8')


def test_readme_references_existing_configs_clis_and_tests() -> None:
	text = _readme_text()
	for cli in RUNBOOK_CLIS:
		assert cli in text
		assert Path(cli).is_file()
	config_references = [
		reference
		for reference in re.findall(r'\$EXP/([^"\s]+\.(?:yaml|sh))', text)
		if '${' not in reference
	]
	assert config_references
	for reference in config_references:
		assert (EXP_ROOT / reference).is_file()
	loop_extractions = re.findall(
		r'^  (0\d_extract_\w+)(?: \\)?$',
		text,
		flags=re.MULTILINE,
	)
	assert len(loop_extractions) == 5
	for name in loop_extractions:
		assert (EXP_ROOT / '50_embeddings' / f'{name}.yaml').is_file()
	for test_path in re.findall(r'tests/seis_ssl_cluster/\S+\.py', text):
		assert Path(test_path).is_file()
	assert '60_five_way.yaml' in text
	assert (EXP_ROOT / '60_five_way.yaml').is_file()


def test_readme_model_matrix_is_exact() -> None:
	text = _readme_text()
	for model_id in FIVE_WAY_MODEL_IDS:
		assert f'- `{model_id}`' in text
	loop_models = re.findall(r'^  (\w+)(?: \\)?$', text, flags=re.MULTILINE)
	loop_models = [
		name for name in loop_models if name in FIVE_WAY_MODEL_IDS
	]
	assert loop_models == list(FIVE_WAY_MODEL_IDS) * 2
	for layout_id in LAYOUT_IDS:
		assert layout_id in text
	for data_size in DATA_SIZES:
		assert data_size in text
	assert '75' in text
	assert len(FIVE_WAY_MODEL_IDS) * len(LAYOUT_IDS) * len(DATA_SIZES) == 75
	assert 'trace_drop' not in text
	assert 'd4' not in text
	assert 'macro_f1' in text


def test_shell_blocks_are_valid_bash(tmp_path: Path) -> None:
	blocks = re.findall(r'```bash\n(.*?)```', _readme_text(), flags=re.DOTALL)
	assert blocks
	for index, block in enumerate(blocks):
		script = tmp_path / f'block_{index}.sh'
		script.write_text(block, encoding='utf-8')
		subprocess.run(  # noqa: S603
			['bash', '-n', str(script)],  # noqa: S607
			check=True,
			capture_output=True,
			text=True,
		)


def test_dry_run_commands_precede_live_commands() -> None:
	text = _readme_text()
	for cli in (
		'proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py',
		'proc/seis_ssl_cluster/run_f3_lithology_five_way.py',
		'proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py',
	):
		lines = [line for line in text.splitlines() if cli in line]
		assert len(lines) >= 2
	audit_dry = text.index(
		'audit_f3_lithology_five_way_sources.py --config "$CONFIG" --dry-run'
	)
	audit_live = text.index(
		'audit_f3_lithology_five_way_sources.py --config "$CONFIG"\n'
	)
	assert audit_dry < audit_live
	preflight = text.index('--dry-run\ndone')
	full_loop = text.index('for layout in layout_000')
	assert preflight < full_loop
	summary_dry = text.index(
		'summarize_f3_lithology_five_way.py --config "$CONFIG" --dry-run'
	)
	summary_live = text.index(
		'summarize_f3_lithology_five_way.py --config "$CONFIG"\n'
	)
	assert summary_dry < summary_live
	assert 'complete_jobs: 75' in text
	assert not re.search(r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync\s)', text)


def test_upstream_outputs_match_downstream_sources(env_root: Path) -> None:
	del env_root
	five_way = load_config(EXP_ROOT / '60_five_way.yaml')
	by_id = {model['model_id']: model for model in five_way['models']}

	local_bt_full = load_config(
		EXP_ROOT / '10_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml'
	)
	assert (
		f'{local_bt_full["paths"]["output_root"]}/latest.pt'
		== by_id['local_barlow_twins']['checkpoint']
	)

	hmm_full = load_config(
		EXP_ROOT / '30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml'
	)
	assert (
		f'{hmm_full["paths"]["output_root"]}/latest.pt'
		== by_id['local_barlow_twins_hmm_k6']['checkpoint']
	)

	extract = load_config(
		EXP_ROOT / '20_hmm_targets/local_bt100/01_extract_embeddings.yaml'
	)
	cluster = load_config(
		EXP_ROOT / '20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml'
	)
	assert (
		extract['embeddings']['output_dir']
		== cluster['embeddings']['input_dir']
	)
	export_text = os.path.expandvars(
		(
			EXP_ROOT / '20_hmm_targets/local_bt100/k6/03_export_pseudo_targets.sh'
		).read_text(encoding='utf-8')
	)
	tokens = shlex.split(export_text.replace('\\\n', ' '))
	pseudo_root = tokens[tokens.index('--pseudo-target-root') + 1]
	clustering_dir = tokens[tokens.index('--clustering-output-dir') + 1]
	assert clustering_dir == cluster['clustering']['output_dir']
	assert pseudo_root == hmm_full['pseudo_targets']['input_dir']

	random_config = load_config(
		EXP_ROOT / '40_random/01_create_random_checkpoint.yaml'
	)
	assert (
		random_config['random_checkpoint']['output_checkpoint']
		== by_id['random']['checkpoint']
	)
	assert (
		random_config['reference_model']['checkpoint']
		== by_id['mae']['checkpoint']
	)

	extraction_names = {
		'mae': '01_extract_mae.yaml',
		'mae_hmm_k6': '02_extract_mae_hmm_k6.yaml',
		'local_barlow_twins': '03_extract_local_barlow_twins.yaml',
		'local_barlow_twins_hmm_k6': '04_extract_local_barlow_twins_hmm_k6.yaml',
		'random': '05_extract_random.yaml',
	}
	for model_id, filename in extraction_names.items():
		extraction = load_config(EXP_ROOT / '50_embeddings' / filename)
		assert (
			extraction['embeddings']['checkpoint']
			== by_id[model_id]['checkpoint']
		)
		assert (
			extraction['embeddings']['output_dir']
			== by_id[model_id]['embeddings_dir']
		)


UPSTREAM_ROOT = Path('experiments/f3/facies_benchmark_v1')
CONTRACT_SOURCES = (
	(
		UPSTREAM_ROOT / '21_ssl_hmm_continuation_v1/10_stage1/mae/02_full_100ep.yaml',
		{
			'train': FIVE_WAY_STAGE1_TRAIN_CONTRACT,
			'masking': MAE_MASKING_CONTRACT,
			'loss': MAE_LOSS_CONTRACT,
		},
	),
	(
		UPSTREAM_ROOT / '22_local_barlow_twins_v1/02_full_100ep.yaml',
		{
			'train': FIVE_WAY_STAGE1_TRAIN_CONTRACT,
			'barlow_twins': LOCAL_BARLOW_TWINS_OBJECTIVE_CONTRACT,
		},
	),
	(
		UPSTREAM_ROOT
		/ '21_ssl_hmm_continuation_v1/30_stage2/mae100/mae_continue/02_full_25ep.yaml',
		{
			'train': FIVE_WAY_STAGE2_TRAIN_CONTRACT,
			'masking': MAE_MASKING_CONTRACT,
			'loss': MAE_LOSS_CONTRACT,
		},
	),
	(
		EXP_ROOT / '10_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml',
		{
			'train': FIVE_WAY_STAGE2_TRAIN_CONTRACT,
			'barlow_twins': LOCAL_BARLOW_TWINS_OBJECTIVE_CONTRACT,
		},
	),
	(
		UPSTREAM_ROOT
		/ '21_ssl_hmm_continuation_v1/30_stage2/mae100/hmm/k6/02_full_25ep.yaml',
		{
			'train': FIVE_WAY_PRETEXT_TRAIN_CONTRACT,
			'head': FIVE_WAY_HMM_HEAD_CONTRACT,
			'loss': FIVE_WAY_HMM_LOSS_CONTRACT,
		},
	),
	(
		EXP_ROOT / '30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml',
		{
			'train': FIVE_WAY_PRETEXT_TRAIN_CONTRACT,
			'head': FIVE_WAY_HMM_HEAD_CONTRACT,
			'loss': FIVE_WAY_HMM_LOSS_CONTRACT,
		},
	),
)


@pytest.mark.parametrize(('config_path', 'contracts'), CONTRACT_SOURCES)
def test_upstream_configs_realize_the_fixed_contract(
	env_root: Path,
	config_path: Path,
	contracts: dict[str, dict[str, object]],
) -> None:
	del env_root
	config = load_config(config_path)
	for section, contract in contracts.items():
		for key, expected in contract.items():
			assert config[section][key] == expected


def test_random_seed_matches_the_audited_identity(env_root: Path) -> None:
	del env_root
	random_config = load_config(
		EXP_ROOT / '40_random/01_create_random_checkpoint.yaml'
	)
	assert (
		random_config['random_checkpoint']['seed']
		== EXPECTED_MODEL_IDENTITIES['random']['random_seed']
		== 42
	)


def test_summary_root_is_consistent_with_runner_root(env_root: Path) -> None:
	del env_root
	five_way = load_config(EXP_ROOT / '60_five_way.yaml')
	runs_root = Path(five_way['outputs']['runs_root'])
	summary_root = Path(five_way['outputs']['summary_root'])

	assert runs_root != summary_root
	assert runs_root.parent == summary_root.parent
	assert runs_root.parent.name == 'mae_local_bt_five_way_v1'
	assert runs_root.parent.parent.name == 'f3_lithology_benchmark'
	text = _readme_text()
	assert 'f3_lithology_benchmark/mae_local_bt_five_way_v1/runs' in text
	assert 'f3_lithology_benchmark/mae_local_bt_five_way_v1/summary' in text


def test_runbook_restart_contract_matches_the_runner() -> None:
	text = _readme_text()
	restart = text[text.index('中断jobの扱い') : text.index('## 10.')]

	assert '--resume <run_dir>/decoder/latest.pt' in restart
	assert 'skip' in restart
	assert '再学習せず' in restart
	assert '原子的' in restart
	source = Path(
		'src/seis_ssl_cluster/f3/lithology/five_way_runner.py'
	).read_text(encoding='utf-8')
	assert '_decoder_is_completed' in source
	assert 'decoder training is interrupted in' in source
	assert 'job already completed; refusing to overwrite' in source
