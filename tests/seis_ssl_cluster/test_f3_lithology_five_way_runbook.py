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
)


@pytest.fixture
def env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3_root'))
	return root


def _readme_text() -> str:
	return README.read_text(encoding='utf-8')


def test_readme_references_existing_configs_clis_and_docs() -> None:
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
	assert (EXP_ROOT / '60_five_way.yaml').is_file()
	links = re.findall(r'\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)', text)
	assert links
	for link in links:
		assert (README.parent / link).resolve().is_file(), link


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


def test_runbook_phase_order_is_explicit() -> None:
	text = _readme_text()
	headings = (
		'## Source production',
		'## Historical v1 evaluation',
	)
	positions = [text.index(heading) for heading in headings]
	assert positions == sorted(positions)
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


def test_runner_restart_contract_is_implemented() -> None:
	source = Path(
		'src/seis_ssl_cluster/f3/lithology/five_way_runner.py'
	).read_text(encoding='utf-8')
	assert '_decoder_is_completed' in source
	assert 'decoder training is interrupted in' in source
	assert 'job already completed; refusing to overwrite' in source
