from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config

EXP_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/124_lithology_random_init_hmm_k6_distill010_v1'
)
README = EXP_ROOT / 'README.md'
STAGE2_ROOT = EXP_ROOT / '10_stage2/random_init/hmm/k6'
CANDIDATE_CONFIG = EXP_ROOT / '30_downstream/01_candidate.yaml'
RUNBOOK_CLIS = (
	'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
	'proc/seis_ssl_cluster/extract_embeddings.py',
	'proc/seis_ssl_cluster/run_f3_lithology_candidate.py',
	'proc/seis_ssl_cluster/summarize_f3_lithology_candidate.py',
	'proc/seis_ssl_cluster/summarize_f3_lithology_candidate_five_way.py',
)
TARGET_TESTS = (
	'tests/seis_ssl_cluster/test_f3_random_init_hmm_k6_distill010_configs.py',
	'tests/seis_ssl_cluster/test_f3_random_init_hmm_k6_distill010_runbook.py',
	'tests/seis_ssl_cluster/test_f3_lithology_candidate_five_way_summary.py',
)
SECTION_HEADINGS = (
	'## 目的と範囲',
	'## 系譜と契約',
	'## 実行順',
	'## 再開',
	'## 対象テスト',
)
RANDOM_CHECKPOINT_LINEAGE = 'mae_local_bt_five_way_v1/random/random_init.pt'
_EXPORT_LINE = re.compile(r'(?m)^export\s+([A-Z][A-Z0-9_]*)=(.+)$')
_SHELL_VARIABLE = re.compile(r'\$\{?([A-Z][A-Z0-9_]*)\}?')
_QUOTED_REFERENCE = re.compile(r'"\$([A-Z][A-Z0-9_]*)((?:/[^"\s]+)?)"')
_CLI_REFERENCE = re.compile(r'proc/seis_ssl_cluster/[a-z0-9_]+\.py')
_TEST_REFERENCE = re.compile(r'tests/seis_ssl_cluster/test_[a-z0-9_]+\.py')
_MARKDOWN_LINK = re.compile(r'\[[^]]+\]\(([^)#]+)(?:#([^)]+))?\)')
_BASH_BLOCK = re.compile(r'```bash\n(.*?)```', flags=re.DOTALL)


@pytest.fixture
def env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3_root'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	return root


def _readme_text() -> str:
	return README.read_text(encoding='utf-8')


def _bash_blocks(text: str) -> list[str]:
	blocks = _BASH_BLOCK.findall(text)
	assert blocks
	return blocks


def _substitute_variables(value: str, values: dict[str, str], *, owner: str) -> str:
	def substitute(match: re.Match[str]) -> str:
		variable = match.group(1)
		if variable not in values:
			raise KeyError(f'README export {owner} uses undefined ${variable}')
		return values[variable]

	return _SHELL_VARIABLE.sub(substitute, value)


def _exports(text: str) -> dict[str, str]:
	"""Resolve the README's own ``export NAME=...`` lines in document order."""
	values: dict[str, str] = {}
	for name, raw in _EXPORT_LINE.findall(text):
		values[name] = _substitute_variables(raw.strip().strip('"'), values, owner=name)
	return values


def _text_without_exports(text: str) -> str:
	return _EXPORT_LINE.sub('', text)


def test_readme_exports_name_the_experiment_paths() -> None:
	exports = _exports(_readme_text())
	assert Path(exports['EXP']) == EXP_ROOT
	assert Path(exports['STAGE2']) == STAGE2_ROOT
	assert Path(exports['CANDIDATE']) == CANDIDATE_CONFIG
	assert STAGE2_ROOT.is_dir()
	assert CANDIDATE_CONFIG.is_file()


def test_every_quoted_yaml_reference_resolves_to_an_existing_file() -> None:
	text = _readme_text()
	exports = _exports(text)
	references = _QUOTED_REFERENCE.findall(_text_without_exports(text))
	assert references
	resolved: set[Path] = set()
	for name, suffix in references:
		path = Path(exports[name] + suffix)
		assert path.suffix == '.yaml', (name, suffix)
		assert path.is_file(), path
		resolved.add(path)
	experiment_yamls = set(EXP_ROOT.rglob('*.yaml'))
	assert experiment_yamls
	assert resolved == experiment_yamls


@pytest.mark.parametrize('cli', RUNBOOK_CLIS)
def test_referenced_cli_exists(cli: str) -> None:
	assert cli in _readme_text()
	assert Path(cli).is_file(), cli


def test_readme_references_exactly_the_runbook_clis() -> None:
	assert set(_CLI_REFERENCE.findall(_readme_text())) == set(RUNBOOK_CLIS)


def test_markdown_links_resolve_including_anchors() -> None:
	links = _MARKDOWN_LINK.findall(_readme_text())
	assert links
	for link, anchor in links:
		target = (README.parent / link).resolve()
		assert target.is_file(), link
		if anchor:
			headings = re.findall(
				r'(?m)^#+\s+(.+?)\s*$', target.read_text(encoding='utf-8')
			)
			assert anchor in headings, (link, anchor)


def test_shell_blocks_are_valid_bash(tmp_path: Path) -> None:
	for index, block in enumerate(_bash_blocks(_readme_text())):
		script = tmp_path / f'block_{index}.sh'
		script.write_text(block, encoding='utf-8')
		subprocess.run(  # noqa: S603
			['bash', '-n', str(script)],  # noqa: S607
			check=True,
			capture_output=True,
			text=True,
		)


def test_shell_blocks_use_flags_the_clis_define() -> None:
	for block in _bash_blocks(_readme_text()):
		for command in re.findall(
			r'python (proc/seis_ssl_cluster/[a-z0-9_]+\.py)((?:[^\n]|\\\n)*)',
			block,
		):
			cli, arguments = command
			if not Path(cli).is_file():
				continue
			source = Path(cli).read_text(encoding='utf-8')
			for flag in set(re.findall(r'--[a-z][a-z-]*', arguments)):
				if flag in {'--config', '--dry-run'}:
					continue
				assert f"'{flag}'" in source, (cli, flag)


def test_runbook_phase_order_is_explicit() -> None:
	text = _readme_text()
	positions = [text.index(heading) for heading in SECTION_HEADINGS]
	assert positions == sorted(positions)
	assert not re.search(r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync\s)', text)


def test_readme_states_the_distillation_weight_and_random_lineage(
	env_root: Path,
) -> None:
	del env_root
	text = _readme_text()
	full = load_config(STAGE2_ROOT / '02_full_25ep.yaml')
	weight = full['loss']['distillation_weight']
	assert weight == 0.1
	assert f'distillation weight `{weight}`' in text
	assert f'`distillation_weight: {weight}`' in text
	assert f'**{weight}**' in text
	assert 'teacher.checkpoint == student.init_checkpoint' in text
	assert RANDOM_CHECKPOINT_LINEAGE in text
	assert full['teacher']['checkpoint'].endswith(RANDOM_CHECKPOINT_LINEAGE)
	assert full['student']['init_checkpoint'] == full['teacher']['checkpoint']
	assert 'seed 42' in text
	assert 'five-way の固定 model set には含めない' in text


def test_readme_budget_matches_the_full_config(env_root: Path) -> None:
	del env_root
	text = _readme_text()
	train = load_config(STAGE2_ROOT / '02_full_25ep.yaml')['train']
	steps = train['epochs'] * train['samples_per_epoch'] // train['batch_size']
	assert f'{train["epochs"]} epoch' in text
	assert f'{train["samples_per_epoch"]:,} samples/epoch' in text
	assert f'batch {train["batch_size"]}' in text
	assert f'{steps:,} steps' in text


def test_upstream_outputs_match_downstream_sources(env_root: Path) -> None:
	del env_root
	full = load_config(STAGE2_ROOT / '02_full_25ep.yaml')
	extract = load_config(EXP_ROOT / '20_embeddings/01_extract_random_init_hmm_k6.yaml')
	candidate = load_config(CANDIDATE_CONFIG)
	latest = f'{full["paths"]["output_root"]}/latest.pt'
	assert extract['embeddings']['checkpoint'] == latest
	assert candidate['candidate']['checkpoint'] == latest
	assert (
		candidate['candidate']['embeddings_dir'] == extract['embeddings']['output_dir']
	)


def test_summary_roots_are_consistent_with_runs_root(env_root: Path) -> None:
	del env_root
	outputs = load_config(CANDIDATE_CONFIG)['outputs']
	runs_root = Path(outputs['runs_root'])
	summary_root = Path(outputs['summary_root'])
	assert runs_root != summary_root
	assert runs_root.parent == summary_root.parent
	assert runs_root.parent.name == 'random_init_hmm_k6_distill010_v1'
	assert runs_root.parent.parent.name == 'f3_lithology_benchmark'
	text = _readme_text()
	summary_keys = set(outputs) - {'runs_root'}
	assert 'summary_root' in summary_keys
	for key in summary_keys:
		assert f'`outputs.{key}`' in text, key
	documented = set(re.findall(r'`outputs\.([a-z_]+)`', text))
	assert documented
	assert documented <= set(outputs), documented - set(outputs)


def test_target_test_block_lists_exactly_the_expected_files() -> None:
	text = _readme_text()
	block = text[text.index('## 対象テスト') :]
	listed = _TEST_REFERENCE.findall(block)
	assert 'pytest -q' in block
	assert listed == list(TARGET_TESTS)
	assert Path(__file__).resolve() in {Path(test).resolve() for test in TARGET_TESTS}


@pytest.mark.parametrize('test_file', TARGET_TESTS)
def test_target_test_file_exists(test_file: str) -> None:
	assert Path(test_file).is_file(), test_file
