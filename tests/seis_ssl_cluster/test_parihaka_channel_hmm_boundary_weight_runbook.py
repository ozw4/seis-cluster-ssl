from __future__ import annotations

import os
import re
from pathlib import Path

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'37_channel_hmm_boundary_weight_v1'
)
README = EXPERIMENT_ROOT / 'README.md'
EXPORT_SCRIPTS = tuple(
	EXPERIMENT_ROOT
	/ '10_pseudo_targets'
	/ source
	/ variant
	/ '01_export_pseudo_targets.sh'
	for source in ('mae100', 'local_bt100')
	for variant in ('alpha050_tau1', 'alpha100_tau1')
)
TRAINING_CONFIGS = tuple(
	EXPERIMENT_ROOT
	/ '20_stage2'
	/ source
	/ variant
	/ '01_full_25ep.yaml'
	for source in ('mae100', 'local_bt100')
	for variant in ('alpha050_tau1', 'alpha100_tau1')
)
EXTRACTION_CONFIGS = tuple(
	EXPERIMENT_ROOT / '30_embeddings' / filename
	for filename in (
		'01_extract_mae_hmm_k6_boundary_alpha050_tau1.yaml',
		'02_extract_mae_hmm_k6_boundary_alpha100_tau1.yaml',
		'03_extract_local_barlow_twins_hmm_k6_boundary_alpha050_tau1.yaml',
		'04_extract_local_barlow_twins_hmm_k6_boundary_alpha100_tau1.yaml',
	)
)
CHANNEL_CONFIG = EXPERIMENT_ROOT / '40_channel_boundary_weight.yaml'
SUMMARY_SCRIPT = EXPERIMENT_ROOT / 'scripts/summarize_validation.py'
CANDIDATE_MODELS = (
	'mae_hmm_k6_boundary_alpha050_tau1',
	'mae_hmm_k6_boundary_alpha100_tau1',
	'local_barlow_twins_hmm_k6_boundary_alpha050_tau1',
	'local_barlow_twins_hmm_k6_boundary_alpha100_tau1',
)
LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))


def _text() -> str:
	return README.read_text(encoding='utf-8')


def _section(text: str, start: str, end: str | None = None) -> str:
	section = text.split(start, maxsplit=1)[1]
	return section if end is None else section.split(end, maxsplit=1)[0]


def _inline_python(text: str) -> list[str]:
	return re.findall(r"python - <<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)


def _bash_blocks(text: str) -> list[str]:
	return re.findall(r'```bash\n(.*?)\n```', text, flags=re.DOTALL)


def _array_items(section: str, name: str) -> tuple[str, ...]:
	match = re.search(rf'{name}=\(\n(?P<body>.*?)\n\)', section, flags=re.DOTALL)
	assert match is not None
	return tuple(
		item.strip().strip('"')
		for item in match.group('body').splitlines()
		if item.strip()
	)


def test_runbook_references_every_phase_file() -> None:
	assert README.is_file()
	for path in (*EXPORT_SCRIPTS, *TRAINING_CONFIGS, *EXTRACTION_CONFIGS):
		assert path.is_file()
	for path in EXPORT_SCRIPTS:
		assert os.access(path, os.X_OK)
	assert CHANNEL_CONFIG.is_file()
	assert SUMMARY_SCRIPT.is_file()


def test_runbook_inline_python_is_compilable() -> None:
	blocks = _inline_python(_text())
	assert len(blocks) == 3
	for index, source in enumerate(blocks, start=1):
		compile(source, f'{README}:inline-{index}', 'exec')


def test_runbook_states_boundary_table_and_fixed_contract() -> None:
	text = _text()
	for row in (
		'| `alpha000_tau1` | 0.0 | 1.0 | 既存H0を再利用 |',
		'| `alpha050_tau1` | 0.5 | 1.0 | 新規 |',
		'| `alpha100_tau1` | 1.0 | 1.0 | 新規 |',
	):
		assert row in text
	for setting in (
		'K = 6',
		'same_cost = 0.03',
		'advance_cost = 0.0',
		'jump_cost = 1.0',
		'anchors = 0.25 / 0.25',
		'expected boundaries = off',
		'max_jump = 1',
		'reverse forbidden',
		'distillation_weight = 0.2',
		'Stage 2 = 25 epochs / 15,625 steps',
		'screening = medium, five layouts',
		'metric = validation.channel_iou',
	):
		assert setting in text
	assert 'survey-specific' in text
	assert 'transductive' in text
	assert 'validation-only' in text


def test_runbook_has_no_clustering_execution_or_config() -> None:
	text = _text()
	assert 'cluster_embeddings.py' not in text
	assert not re.search(r'cluster[^\n]*\.ya?ml', text, flags=re.IGNORECASE)
	assert 'No new HMM fit or decode is performed' in text


def test_runbook_reuses_alpha_zero_without_reexecuting_it() -> None:
	text = _text()
	export_section = _section(
		text,
		'## 3. Export the four new pseudo-target sets',
		'## 4. Audit all six pseudo-target sets',
	)
	training_section = _section(
		text,
		'## 7. Run the four full 25-epoch trainings',
		'## 8. Audit the four Stage 2 checkpoints',
	)
	extraction_section = _section(
		text,
		'## 9. Extract the four new embedding volumes',
		'## 10. Audit all eight embedding sources',
	)
	assert 'alpha000_tau1/01_export' not in export_section
	assert 'alpha000_tau1/01_full' not in training_section
	assert 'alpha000_tau1.yaml' not in extraction_section
	assert 'Do not export `alpha000_tau1`' in text
	assert 'Do not re-extract either control' in text
	assert 'Do not rerun the MAE control' in text


def test_decoder_loops_target_only_four_models_five_layouts_and_medium() -> None:
	text = _text()
	preflight = _section(
		text,
		'## 11. Dry-run the 20 new medium decoder jobs',
		'## 12. Run the 20 new medium decoder jobs',
	)
	execution = _section(
		text,
		'## 12. Run the 20 new medium decoder jobs',
		'## 13. Write the validation screening summary',
	)
	for section in (preflight, execution):
		assert _array_items(section, 'CANDIDATE_MODELS') == CANDIDATE_MODELS
		assert _array_items(section, 'LAYOUTS') == LAYOUTS
		assert '--model "$model"' in section
		assert '--layout "$layout"' in section
		assert section.count('--size medium') == 1
		assert '--validation-only' in section
	assert '--dry-run' in preflight
	assert '--dry-run' not in '\n'.join(_bash_blocks(execution))
	assert len(CANDIDATE_MODELS) * len(LAYOUTS) == 20


def test_decoder_loop_skips_complete_and_stops_on_partial() -> None:
	section = _section(
		_text(),
		'## 12. Run the 20 new medium decoder jobs',
		'## 13. Write the validation screening summary',
	)
	block = _bash_blocks(section)[0]
	job_dir = '$VALIDATION_RUNS_ROOT/model=$model/layout=$layout/size=medium'
	assert f'job_dir="{job_dir}"' in block
	assert 'if [[ -f "$job_dir/metrics.json" ]]' in block
	assert 'continue' in block
	assert 'if [[ -f "$job_dir/latest.pt" ]]' in block
	assert 'incomplete Channel job requires explicit resume' in block
	assert 'exit 1' in block
	assert block.index('metrics.json') < block.index('latest.pt')
	assert 'never resumes automatically' in section


def test_summary_call_uses_three_explicit_roots() -> None:
	section = _section(
		_text(),
		'## 13. Write the validation screening summary',
		'## 14. End the Phase',
	)
	assert 'scripts/summarize_validation.py' in section
	assert '--existing-runs-root "$EXISTING_RUNS_ROOT"' in section
	assert '--validation-runs-root "$VALIDATION_RUNS_ROOT"' in section
	assert '--report-root "$REPORT_ROOT"' in section
	assert "python - <<'PY'" not in section


def test_runbook_stops_without_other_sweeps_or_unsafe_commands() -> None:
	text = _text()
	commands = '\n'.join(_bash_blocks(text))
	assert not re.search(r'--size\s+(?:small|large)\b', commands)
	assert not re.search(r'(?m)^\s*for\s+tau\s+in\b', commands)
	assert not re.search(
		r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync(?:\s|$)|ln\s+-s(?:\s|$))',
		commands,
	)
	for block in _bash_blocks(text):
		assert block.startswith('set -euo pipefail\n')
		assert 'REPO_ROOT=$(git rev-parse --show-toplevel)' in block
		assert 'cd "$REPO_ROOT"' in block
	assert '/workspace' not in text
	assert 'export CUDA_VISIBLE_DEVICES=' not in text
	assert 'summarize_validation.py' in text
	assert 'proc/seis_ssl_cluster/summar' not in text
	assert 'neither implements nor executes a tau sweep' in text
	assert 'Do not read test results' in text
	assert 'Do not start another Phase automatically' in text


def test_targeted_tests_cover_production_and_new_contracts() -> None:
	section = _section(
		_text(),
		'## 2. Targeted tests',
		'## 3. Export the four new pseudo-target sets',
	)
	for filename in (
		'test_stratigraphy_boundary_weights.py',
		'test_stratigraphy_export.py',
		'test_strat_pseudo_dataset.py',
		'test_parihaka_channel_decoder.py',
		'test_parihaka_hmm_boundary_weight_configs.py',
		'test_parihaka_channel_hmm_boundary_weight_configs.py',
		'test_parihaka_channel_hmm_boundary_weight_summary.py',
		'test_parihaka_channel_hmm_boundary_weight_runbook.py',
	):
		assert filename in section
