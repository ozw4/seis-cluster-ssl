"""Runbook contract for the F3 random-init self-target HMM-K6 control (exp 125)."""

from __future__ import annotations

import functools
import importlib.util
import itertools
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.stratigraphy import pseudo_target_paths

if TYPE_CHECKING:
	import argparse
	from collections.abc import Mapping, Sequence

EXP_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/'
	'125_lithology_random_init_self_target_hmm_k6_distill010_v1'
)
README = EXP_ROOT / 'README.md'
TARGETS_ROOT = EXP_ROOT / '10_hmm_targets/random_init'
TARGET_EXTRACTION_CONFIG = TARGETS_ROOT / '01_extract_embeddings.yaml'
CLUSTERING_CONFIG = TARGETS_ROOT / 'k6/02_cluster_hmm_k6.yaml'
EXPORT_SCRIPT = TARGETS_ROOT / 'k6/03_export_pseudo_targets.sh'
STAGE2_ROOT = EXP_ROOT / '20_stage2/random_init/hmm/k6'
FEASIBILITY_CONFIG = STAGE2_ROOT / '01_gpu_feasibility_1step.yaml'
FULL_CONFIG = STAGE2_ROOT / '02_full_25ep.yaml'
EXTRACTION_CONFIG = (
	EXP_ROOT / '30_embeddings/01_extract_random_init_self_target_hmm_k6_distill010.yaml'
)
CANDIDATE_CONFIG = EXP_ROOT / '40_downstream/01_candidate.yaml'
RECIPE_COUNT = 7
REPORT = Path('reports/f3/random_init_self_target_hmm_k6_distill010_control_v1.md')
REVIEW_DIR = (
	'reports/f3/facies_benchmark_v2/random_init_self_target_hmm_k6_distill010_v1/'
)
CONTROL_NAMESPACE = 'random_init_self_target_hmm_k6_distill010_v1'
RANDOM_CHECKPOINT_LINEAGE = 'mae_local_bt_five_way_v1/random/random_init.pt'
DISTILLATION_WEIGHT = 0.1
CELL_COUNT = 15
# CLIs the README names directly, in pipeline order.
TARGET_EXTRACTION_CLI = 'proc/seis_ssl_cluster/extract_embeddings.py'
CLUSTERING_CLI = 'proc/seis_ssl_cluster/cluster_embeddings.py'
STAGE2_CLI = 'proc/seis_ssl_cluster/train_strat_hmm_pretext.py'
CANDIDATE_CLI = 'proc/seis_ssl_cluster/run_f3_lithology_candidate.py'
SUMMARY_CLI = 'proc/seis_ssl_cluster/summarize_f3_lithology_candidate.py'
FIVE_WAY_SUMMARY_CLI = (
	'proc/seis_ssl_cluster/summarize_f3_lithology_candidate_five_way.py'
)
RUNBOOK_CLIS = (
	TARGET_EXTRACTION_CLI,
	CLUSTERING_CLI,
	STAGE2_CLI,
	CANDIDATE_CLI,
	SUMMARY_CLI,
	FIVE_WAY_SUMMARY_CLI,
)
# The export CLI is reached only through the export script, so the README never
# names it; the script is the single place that does.
EXPORT_CLI = 'proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py'
PIPELINE_CLIS = (*RUNBOOK_CLIS, EXPORT_CLI)
TARGET_TESTS = (
	'tests/seis_ssl_cluster/test_f3_random_init_self_target_hmm_k6_distill010_configs.py',
	'tests/seis_ssl_cluster/test_f3_random_init_self_target_hmm_k6_distill010_runbook.py',
	'tests/seis_ssl_cluster/test_f3_lithology_candidate_five_way_summary.py',
)
SECTION_HEADINGS = (
	'## 目的と範囲',
	'## 系譜と契約',
	'## 実行順',
	'## 結果',
	'## 再開',
	'## 対象テスト',
)
# (program, target, --config) of every runbook stage as the README writes them,
# in the order the driver must execute them. Consecutive dry-run / real pairs
# collapse into one entry.
EXPORT_SCRIPT_REFERENCE = '$TARGETS/k6/03_export_pseudo_targets.sh'
EXPECTED_PIPELINE = (
	('python', TARGET_EXTRACTION_CLI, '$TARGETS/01_extract_embeddings.yaml'),
	('python', CLUSTERING_CLI, '$TARGETS/k6/02_cluster_hmm_k6.yaml'),
	('bash', EXPORT_SCRIPT_REFERENCE, None),
	('python', STAGE2_CLI, '$STAGE2/01_gpu_feasibility_1step.yaml'),
	('python', STAGE2_CLI, '$STAGE2/02_full_25ep.yaml'),
	(
		'python',
		TARGET_EXTRACTION_CLI,
		'$EXP/30_embeddings/01_extract_random_init_self_target_hmm_k6_distill010.yaml',
	),
	('python', CANDIDATE_CLI, '$CANDIDATE'),
	('python', SUMMARY_CLI, '$CANDIDATE'),
	('python', FIVE_WAY_SUMMARY_CLI, '$CANDIDATE'),
)
# Indices into EXPECTED_PIPELINE that must share one bash block, in phase order:
# targets, Stage 2, embedding extraction, cells, summaries.
PHASE_GROUPS = ((0, 1, 2), (3, 4), (5,), (6,), (7, 8))
_EXPORT_LINE = re.compile(r'(?m)^export\s+([A-Z][A-Z0-9_]*)=(.+)$')
_SHELL_VARIABLE = re.compile(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?')
_QUOTED_REFERENCE = re.compile(r'"\$([A-Z][A-Z0-9_]*)((?:/[^"\s]+)?)"')
_CLI_REFERENCE = re.compile(r'proc/seis_ssl_cluster/[a-z0-9_]+\.py')
_TEST_REFERENCE = re.compile(r'tests/seis_ssl_cluster/test_[a-z0-9_]+\.py')
_MARKDOWN_LINK = re.compile(r'\[[^]]+\]\(([^)#]+)(?:#([^)]+))?\)')
_BASH_BLOCK = re.compile(r'```bash\n(.*?)```', flags=re.DOTALL)
_FOR_LOOP = re.compile(r'(?m)^\s*for ([a-z_]+) in ([^;]+); do\s*$')
_LABELS_PATH = re.compile(r'`([^`\s]*\.hmm_labels_token\.npy)`')


@dataclass(frozen=True)
class Command:
	"""One ``python`` or ``bash`` command line of a README bash block."""

	block: int
	program: str
	target: str
	arguments: tuple[str, ...]

	@property
	def config(self) -> str | None:
		if '--config' not in self.arguments:
			return None
		return self.arguments[self.arguments.index('--config') + 1]

	@property
	def dry_run(self) -> bool:
		return '--dry-run' in self.arguments

	@property
	def key(self) -> tuple[str, str, str | None]:
		return (self.program, self.target, self.config)

	@property
	def cli(self) -> str:
		return EXPORT_CLI if self.program == 'bash' else self.target


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3_root'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	return root


def _readme_text() -> str:
	return README.read_text(encoding='utf-8')


def _report_text() -> str:
	return REPORT.read_text(encoding='utf-8')


def _bash_blocks(text: str) -> list[str]:
	blocks = _BASH_BLOCK.findall(text)
	assert blocks
	return blocks


def _substitute_variables(value: str, values: Mapping[str, str], *, owner: str) -> str:
	def substitute(match: re.Match[str]) -> str:
		variable = match.group(1)
		if variable not in values:
			raise KeyError(f'README command {owner} uses undefined ${variable}')
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


def _relative(value: str, root: Path) -> str:
	"""Return an artifact path relative to the temporary artifact root."""
	return str(Path(value).relative_to(root))


def _table_row(text: str, label: str) -> list[str]:
	"""Return the cells of the lineage table row named ``label`` (label first)."""
	rows = [line for line in text.splitlines() if line.startswith(f'| {label} |')]
	assert len(rows) == 1, label
	cells = [cell.strip() for cell in rows[0].split('|')]
	assert cells[0] == cells[-1] == ''
	return cells[1:-1]


def _links(document: Path) -> list[tuple[Path, str]]:
	"""Return the resolved targets and anchors of a document's markdown links."""
	links = _MARKDOWN_LINK.findall(document.read_text(encoding='utf-8'))
	assert links, document
	return [((document.parent / link).resolve(), anchor) for link, anchor in links]


@functools.cache
def _cli_parser(cli: str) -> argparse.ArgumentParser:
	"""Import one proc entrypoint from its file and build its own parser."""
	path = Path(cli).resolve()
	spec = importlib.util.spec_from_file_location(f'runbook_{path.stem}', path)
	assert spec is not None, cli
	assert spec.loader is not None, cli
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module.build_parser()


def _supports_dry_run(cli: str) -> bool:
	return '--dry-run' in _cli_parser(cli).format_help()


def _parse_with_cli(cli: str, argv: Sequence[str]) -> argparse.Namespace:
	"""Parse README arguments with the CLI's argparse parser (exit means reject)."""
	try:
		return _cli_parser(cli).parse_args(list(argv))
	except SystemExit as exc:
		msg = f'{cli} rejected README arguments {list(argv)} (exit {exc.code})'
		raise AssertionError(msg) from exc


def _export_script_tokens() -> list[str]:
	"""Return the export script's CLI arguments with ``"$@"`` stripped."""
	text = EXPORT_SCRIPT.read_text(encoding='utf-8').replace('\\\n', ' ')
	commands = [
		line.strip() for line in text.splitlines() if line.strip().startswith('python ')
	]
	assert len(commands) == 1, EXPORT_SCRIPT
	tokens = shlex.split(os.path.expandvars(commands[0]))
	assert tokens[:2] == ['python', EXPORT_CLI]
	assert tokens[-1] == '$@', 'export script must pass its arguments through'
	return tokens[2:-1]


def _export_flags() -> dict[str, str]:
	tokens = _export_script_tokens()
	assert len(tokens) % 2 == 0, tokens
	return dict(zip(tokens[::2], tokens[1::2], strict=True))


def _commands(text: str) -> list[Command]:
	"""Return every ``python`` / ``bash`` command of the README bash blocks."""
	commands: list[Command] = []
	for index, block in enumerate(_bash_blocks(text)):
		for line in block.replace('\\\n', ' ').splitlines():
			stripped = line.strip()
			if not stripped.startswith(('python ', 'bash ')):
				continue
			program, target, *arguments = shlex.split(stripped)
			commands.append(Command(index, program, target, tuple(arguments)))
	assert commands
	return commands


def _loop_values(block: str) -> dict[str, list[str]]:
	return {name: values.split() for name, values in _FOR_LOOP.findall(block)}


def _expanded_argvs(
	command: Command,
	exports: Mapping[str, str],
	loops: Mapping[str, list[str]],
) -> list[list[str]]:
	"""Expand exports and enclosing ``for`` loops into concrete token lists."""
	tokens = [command.target, *command.arguments]
	names = [name for name in loops if any(f'${name}' in token for token in tokens)]
	argvs: list[list[str]] = []
	for combination in itertools.product(*(loops[name] for name in names)):
		scope = {**exports, **dict(zip(names, combination, strict=True))}
		argvs.append(
			[
				_substitute_variables(token, scope, owner=command.target)
				for token in tokens
			]
		)
	return argvs


def _cli_argv(command: Command, target: str, arguments: list[str]) -> list[str]:
	"""Return the argv the underlying CLI receives for one expanded command."""
	if command.program == 'bash':
		assert Path(target) == EXPORT_SCRIPT, target
		return [*_export_script_tokens(), *arguments]
	assert target == command.target
	config = arguments[arguments.index('--config') + 1]
	assert Path(config).is_file(), config
	assert Path(config).is_relative_to(EXP_ROOT), config
	return arguments


def test_readme_exports_name_the_experiment_paths() -> None:
	exports = _exports(_readme_text())
	assert list(exports) == ['EXP', 'TARGETS', 'STAGE2', 'CANDIDATE']
	assert Path(exports['EXP']) == EXP_ROOT
	assert Path(exports['TARGETS']) == TARGETS_ROOT
	assert Path(exports['STAGE2']) == STAGE2_ROOT
	assert Path(exports['CANDIDATE']) == CANDIDATE_CONFIG
	assert TARGETS_ROOT.is_dir()
	assert STAGE2_ROOT.is_dir()
	assert CANDIDATE_CONFIG.is_file()


def test_every_quoted_reference_resolves_to_an_existing_recipe() -> None:
	text = _readme_text()
	exports = _exports(text)
	references = _QUOTED_REFERENCE.findall(_text_without_exports(text))
	assert references
	resolved: set[Path] = set()
	for name, suffix in references:
		path = Path(exports[name] + suffix)
		assert path.suffix in {'.yaml', '.sh'}, (name, suffix)
		assert path.is_file(), path
		resolved.add(path)
	recipes = {*EXP_ROOT.rglob('*.yaml'), *EXP_ROOT.rglob('*.sh')}
	assert len(recipes) == RECIPE_COUNT
	assert resolved == recipes


@pytest.mark.parametrize('cli', PIPELINE_CLIS)
def test_pipeline_cli_exists(cli: str) -> None:
	assert Path(cli).is_file(), cli


def test_readme_references_exactly_the_runbook_clis() -> None:
	text = _readme_text()
	assert set(_CLI_REFERENCE.findall(text)) == set(RUNBOOK_CLIS)
	assert EXPORT_CLI not in RUNBOOK_CLIS
	script = EXPORT_SCRIPT.read_text(encoding='utf-8')
	assert set(_CLI_REFERENCE.findall(script)) == {EXPORT_CLI}
	commands = _commands(text)
	assert {command.target for command in commands if command.program == 'python'} == (
		set(RUNBOOK_CLIS)
	)
	assert {command.target for command in commands if command.program == 'bash'} == {
		EXPORT_SCRIPT_REFERENCE
	}
	assert {command.cli for command in commands} == set(PIPELINE_CLIS)


@pytest.mark.parametrize('document', [README, REPORT], ids=['readme', 'report'])
def test_markdown_links_resolve_including_anchors(document: Path) -> None:
	for target, anchor in _links(document):
		assert target.is_file(), (document, target)
		if anchor:
			headings = re.findall(
				r'(?m)^#+\s+(.+?)\s*$', target.read_text(encoding='utf-8')
			)
			assert anchor in headings, (target, anchor)


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


def test_shell_commands_parse_with_the_clis_own_parsers() -> None:
	text = _readme_text()
	exports = _exports(text)
	blocks = _bash_blocks(text)
	parsed: dict[str, list[argparse.Namespace]] = {}
	for command in _commands(text):
		loops = _loop_values(blocks[command.block])
		for target, *arguments in _expanded_argvs(command, exports, loops):
			argv = _cli_argv(command, target, arguments)
			parsed.setdefault(command.cli, []).append(
				_parse_with_cli(command.cli, argv)
			)
	assert set(parsed) == set(PIPELINE_CLIS)
	export_runs = parsed[EXPORT_CLI]
	assert [run.dry_run for run in export_runs] == [True, False]
	for run in export_runs:
		assert run.k == 6
		assert run.schema_version == 2
	cells = [run for run in parsed[CANDIDATE_CLI] if not run.dry_run]
	assert len(cells) == CELL_COUNT
	assert {(run.layout, run.size) for run in cells} == set(
		itertools.product(LAYOUT_IDS, DATA_SIZES)
	)
	preflight = [run for run in parsed[CANDIDATE_CLI] if run.dry_run]
	assert [(run.layout, run.size) for run in preflight] == [('layout_000', 'small')]


def test_each_stage_dry_runs_before_it_executes() -> None:
	commands = _commands(_readme_text())
	assert not _supports_dry_run(SUMMARY_CLI)
	for cli in PIPELINE_CLIS:
		if cli != SUMMARY_CLI:
			assert _supports_dry_run(cli), cli
	for index, command in enumerate(commands):
		same_stage = [other for other in commands if other.key == command.key]
		assert all(other.block == command.block for other in same_stage), command
		if not _supports_dry_run(command.cli):
			assert not any(other.dry_run for other in same_stage), command
			continue
		if command.dry_run:
			later = [
				other for other in commands[index + 1 :] if other.key == command.key
			]
			assert any(not other.dry_run for other in later), command
			continue
		previous = [other for other in commands[:index] if other.key == command.key]
		assert previous, command
		assert previous[-1].dry_run, command


def test_runbook_phase_order_is_explicit() -> None:
	text = _readme_text()
	positions = [text.index(heading) for heading in SECTION_HEADINGS]
	assert positions == sorted(positions)
	assert not re.search(r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync\s)', text)
	commands = _commands(text)
	stages = [key for key, _ in itertools.groupby(command.key for command in commands)]
	assert stages == list(EXPECTED_PIPELINE)
	phase_blocks: list[int] = []
	for group in PHASE_GROUPS:
		keys = {EXPECTED_PIPELINE[index] for index in group}
		blocks = {command.block for command in commands if command.key in keys}
		assert len(blocks) == 1, group
		phase_blocks.append(blocks.pop())
	assert phase_blocks == sorted(phase_blocks)
	assert len(set(phase_blocks)) == len(PHASE_GROUPS)
	runbook = text[text.index('## 実行順') : text.index('## 結果')]
	assert '抽出、clustering、export' in runbook
	assert 'export 後に Stage 2' in runbook
	assert 'feasibility（1 step smoke）、full の順' in runbook  # noqa: RUF001
	assert '15 cell 完了後' in runbook


def test_readme_states_the_distillation_weight_and_random_lineage() -> None:
	text = _readme_text()
	full = load_config(FULL_CONFIG)
	loss = full['loss']
	weight = loss['distillation_weight']
	assert weight == DISTILLATION_WEIGHT
	assert f'distillation weight `{weight}`' in text
	assert f'`distillation_weight: {weight}`' in text
	row = _table_row(text, 'loss weights')
	expected = f'{loss["prototype_weight"]} / {loss["usage_weight"]} / **{weight}**'
	assert row[2] == row[3] == expected
	assert '`0.0` にはできない' in text
	assert 'teacher.checkpoint == student.init_checkpoint' in text
	assert RANDOM_CHECKPOINT_LINEAGE in text
	assert full['teacher']['checkpoint'].endswith(RANDOM_CHECKPOINT_LINEAGE)
	assert full['student']['init_checkpoint'] == full['teacher']['checkpoint']
	assert full['student']['unfreeze_top_blocks'] == 1
	assert 'encoder top-1' in text
	assert f'seed {full["train"]["seed"]}' in text
	assert 'five-way の固定 model set には含めない' in text
	identity = full['identity']
	assert f'`{identity["model_tag"]}`' in text
	scientific = identity['scientific_identity']
	assert f'`experiment_role: {scientific["experiment_role"]}`' in text
	assert f'`pseudo_target_source: {scientific["pseudo_target_source"]}`' in text
	assert scientific['pseudo_target_source'] == f'{CONTROL_NAMESPACE}/random_init/k6'
	assert scientific['distillation_weight'] == weight


def test_readme_lineage_table_pins_init_teacher_and_optimizer_rows() -> None:
	text = _readme_text()
	full = load_config(FULL_CONFIG)
	row = _table_row(text, 'student init')
	assert 'MAE 100 epoch' in row[1]
	assert RANDOM_CHECKPOINT_LINEAGE in row[2]
	assert 'random encoder' in row[3]
	assert not row[3].startswith('同一')
	assert 'MAE' not in row[2]
	assert 'MAE' not in row[3]
	row = _table_row(text, 'teacher')
	assert row[1] == 'student init と同一'
	assert row[2].startswith('student init と同一')
	assert row[3].startswith('student init と同一')
	assert 'random encoder' in row[3]
	assert 'MAE' not in row[2]
	assert 'MAE' not in row[3]
	row = _table_row(text, 'unfreeze / LR')
	assert row[2] == row[3] == '同一'
	train = full['train']
	assert f'encoder top-{full["student"]["unfreeze_top_blocks"]}' in row[1]
	lr = re.search(r'LR `([^`]+)`', row[1])
	assert lr is not None
	assert float(lr.group(1)) == train['lr'] == train['encoder_lr']
	assert f'wd {train["weight_decay"]}' in row[1]
	assert train['amp'] is False
	assert 'FP32' in row[1]
	assert f'seed {train["seed"]}' in row[1]


def test_readme_and_report_flag_the_inherited_mae_config_fields() -> None:
	for text in (_readme_text(), _report_text()):
		assert 'metadata.pretrained_weights_loaded: false' in text
		assert '`config.continuation` には MAE 系譜が写る' in text
		assert '`stratigraphy_pretext.base_objective`（`amp_mae3d`）' in text  # noqa: RUF001
		assert '`pretraining_objective`' in text
		assert '`stratigraphy_pretext.pseudo_target_input_dir`' in text


def test_readme_states_targets_come_from_the_random_encoder(
	artifact_root: Path,
) -> None:
	text = _readme_text()
	target_extract = load_config(TARGET_EXTRACTION_CONFIG)['embeddings']
	full = load_config(FULL_CONFIG)
	clustering = load_config(CLUSTERING_CONFIG)['clustering']
	assert target_extract['checkpoint'] == full['teacher']['checkpoint']
	assert target_extract['checkpoint'].endswith(RANDOM_CHECKPOINT_LINEAGE)
	assert 'mae' not in _relative(target_extract['output_dir'], artifact_root).lower()
	assert 'random encoder 自身の embedding' in text
	assert 'MAE100 や Barlow Twins 由来の target lineage を混在させない' in text
	row = _table_row(text, 'pseudo target')
	assert 'MAE100' in row[2]
	assert f'`{CONTROL_NAMESPACE}/random_init`' in row[3]
	assert 'random encoder 自身' in row[3]
	assert not row[3].startswith('同一')
	assert clustering['method'] == 'stratigraphic_hmm_kmeans'
	assert f'`{clustering["method"]}`' in text
	assert clustering['k_values'] == [6]
	assert f'K={clustering["k_values"][0]}' in text
	assert f'PCA {clustering["pca"]["n_components"]}' in text
	assert f'{clustering["sample_tokens"]:,} sample tokens' in text
	assert f'seed {clustering["seed"]}' in text
	assert f'`{clustering["residualization"]["mode"]}` residualization' in text
	flags = _export_flags()
	assert (
		f'k {flags["--k"]}、confidence {flags["--confidence"]}、'
		f'boundary alpha {flags["--boundary-alpha"]}、'
		f'boundary tau {flags["--boundary-tau"]}、'
		f'schema version {flags["--schema-version"]}'
	) in text
	assert f'`{_relative(flags["--pseudo-target-root"], artifact_root)}`' in text
	assert '`pseudo_targets.input_dir`' in text


def test_readme_extraction_contracts_match_the_configs() -> None:
	text = _readme_text()
	contracts = []
	for config_path, manifest_version in (
		(TARGET_EXTRACTION_CONFIG, 'v1'),
		(EXTRACTION_CONFIG, 'v2'),
	):
		raw = load_config(config_path)
		assert f'/facies_benchmark_{manifest_version}/' in raw['manifests']['input']
		assert f'`facies_benchmark_{manifest_version}` manifest' in text
		embedding = raw['embedding']
		window = ','.join(str(value) for value in embedding['window_size'])
		overlap = ','.join(str(value) for value in embedding['overlap'])
		assert f'window `[{window}]`' in text
		assert f'`[{overlap}]`' in text
		assert embedding['output_dtype'] in text
		assert f'`amp: {str(embedding["amp"]).lower()}`' in text
		assert f'min token valid fraction {embedding["min_token_valid_fraction"]}' in (
			text
		)
		contracts.append(embedding)
	assert contracts[0] == contracts[1]


def test_readme_budget_matches_the_full_config() -> None:
	text = _readme_text()
	train = load_config(FULL_CONFIG)['train']
	steps = train['epochs'] * train['samples_per_epoch'] // train['batch_size']
	assert f'{train["epochs"]} epoch' in text
	assert f'{train["samples_per_epoch"]:,} samples/epoch' in text
	assert f'batch {train["batch_size"]}' in text
	assert f'{steps:,} steps' in text
	row = _table_row(text, 'Stage 2 予算')
	assert row[2] == row[3] == '同一'


def test_upstream_outputs_match_downstream_sources(artifact_root: Path) -> None:
	target_extract = load_config(TARGET_EXTRACTION_CONFIG)['embeddings']
	clustering = load_config(CLUSTERING_CONFIG)
	flags = _export_flags()
	feasibility = load_config(FEASIBILITY_CONFIG)
	full = load_config(FULL_CONFIG)
	extract = load_config(EXTRACTION_CONFIG)['embeddings']
	candidate = load_config(CANDIDATE_CONFIG)['candidate']

	assert clustering['embeddings']['input_dir'] == target_extract['output_dir']
	assert flags['--clustering-output-dir'] == clustering['clustering']['output_dir']
	export_root = Path(flags['--pseudo-target-root'])
	k = int(flags['--k'])
	assert k == full['pseudo_targets']['k'] == full['head']['num_prototypes'] == 6
	exported = pseudo_target_paths(export_root, k=k, survey_id='survey')
	for stage2 in (feasibility, full):
		pseudo_targets = stage2['pseudo_targets']
		assert Path(pseudo_targets['input_dir']) == export_root
		assert exported.labels.parent == Path(pseudo_targets['input_dir']) / f'k{k}'
		assert stage2['teacher'] == full['teacher']
		assert stage2['student'] == full['student']
	latest = f'{full["paths"]["output_root"]}/latest.pt'
	assert extract['checkpoint'] == latest
	assert candidate['checkpoint'] == latest
	assert candidate['embeddings_dir'] == extract['output_dir']

	chain = (
		target_extract['output_dir'],
		clustering['clustering']['output_dir'],
		str(export_root),
		full['paths']['output_root'],
		extract['output_dir'],
	)
	for value in chain:
		assert value.startswith(f'{artifact_root}/'), value
		assert f'/{CONTROL_NAMESPACE}/' in value, value
	assert len(set(chain)) == len(chain)


def test_summary_roots_are_consistent_with_runs_root() -> None:
	outputs = load_config(CANDIDATE_CONFIG)['outputs']
	assert set(outputs) == {'runs_root', 'summary_root', 'five_way_summary_root'}
	runs_root = Path(outputs['runs_root'])
	summary_roots = {
		key: Path(outputs[key]) for key in ('summary_root', 'five_way_summary_root')
	}
	assert runs_root.parent.name == CONTROL_NAMESPACE
	assert runs_root.parent.parent.name == 'f3_lithology_benchmark'
	for key, summary_root in summary_roots.items():
		assert summary_root != runs_root, key
		assert summary_root.parent == runs_root.parent, key
	assert len({runs_root, *summary_roots.values()}) == 3
	text = _readme_text()
	for key in summary_roots:
		assert f'`outputs.{key}`' in text, key
	documented = set(re.findall(r'`outputs\.([a-z_]+)`', text))
	assert documented == set(summary_roots)


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


def test_report_exists_and_results_link_the_report_and_review_summary() -> None:
	assert REPORT.is_file()
	assert README.resolve() in {target for target, _ in _links(REPORT)}
	readme = _readme_text()
	results = readme[readme.index('## 結果') : readme.index('## 再開')]
	links = _MARKDOWN_LINK.findall(results)
	review_summary = Path(REVIEW_DIR) / 'five_way' / 'summary.md'
	assert [(README.parent / link).resolve() for link, _ in links] == [
		REPORT.resolve(),
		review_summary.resolve(),
	]
	assert review_summary.is_file()
	assert REVIEW_DIR in readme
	assert REVIEW_DIR in _report_text()


def test_report_lineage_paths_match_the_configs(artifact_root: Path) -> None:
	report = _report_text()
	target_extract = load_config(TARGET_EXTRACTION_CONFIG)['embeddings']
	clustering = load_config(CLUSTERING_CONFIG)['clustering']
	full = load_config(FULL_CONFIG)
	extract = load_config(EXTRACTION_CONFIG)['embeddings']
	flags = _export_flags()
	for value in (
		target_extract['output_dir'],
		clustering['output_dir'],
		f'{full["paths"]["output_root"]}/latest.pt',
		extract['output_dir'],
	):
		assert f'`{_relative(value, artifact_root)}`' in report, value
	labels = _LABELS_PATH.findall(report)
	assert len(labels) == 1
	survey_id = Path(labels[0]).name.removesuffix('.hmm_labels_token.npy')
	exported = pseudo_target_paths(
		flags['--pseudo-target-root'], k=int(flags['--k']), survey_id=survey_id
	)
	assert _relative(str(exported.labels), artifact_root).endswith(labels[0])
	assert f'/{CONTROL_NAMESPACE}/random_init/k6/' in f'/{labels[0]}'
	loss = full['loss']
	assert (
		f'prototype / usage / distillation = {loss["prototype_weight"]} / '
		f'{loss["usage_weight"]} / {loss["distillation_weight"]}'
	) in report
	assert RANDOM_CHECKPOINT_LINEAGE in report
	scientific = full['identity']['scientific_identity']
	assert f'`experiment_role: {scientific["experiment_role"]}`' in report
