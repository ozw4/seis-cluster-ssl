from __future__ import annotations

import os
import re
from pathlib import Path

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'36_channel_hmm_transition_balance_v1'
)
README = EXPERIMENT_ROOT / 'README.md'
SOURCES = ('mae100', 'local_bt100')
VARIANTS = ('neutral', 'persist003', 'persist010')
TARGET_CONFIGS = tuple(
	EXPERIMENT_ROOT
	/ '10_hmm_targets'
	/ source
	/ variant
	/ '01_cluster_hmm_k6.yaml'
	for source in SOURCES
	for variant in VARIANTS
)
EXPORT_SCRIPTS = tuple(
	EXPERIMENT_ROOT
	/ '10_hmm_targets'
	/ source
	/ variant
	/ '02_export_pseudo_targets.sh'
	for source in SOURCES
	for variant in VARIANTS
)
TRAINING_CONFIGS = tuple(
	EXPERIMENT_ROOT
	/ '20_stage2'
	/ source
	/ variant
	/ '01_full_25ep.yaml'
	for source in SOURCES
	for variant in VARIANTS
)
EXTRACTION_CONFIGS = tuple(
	EXPERIMENT_ROOT / '30_embeddings' / filename
	for filename in (
		'01_extract_mae_hmm_k6_neutral.yaml',
		'02_extract_mae_hmm_k6_persist003.yaml',
		'03_extract_mae_hmm_k6_persist010.yaml',
		'04_extract_local_barlow_twins_hmm_k6_neutral.yaml',
		'05_extract_local_barlow_twins_hmm_k6_persist003.yaml',
		'06_extract_local_barlow_twins_hmm_k6_persist010.yaml',
	)
)
CHANNEL_CONFIG = EXPERIMENT_ROOT / '40_channel_transition_balance.yaml'
FINAL_CHANNEL_CONFIG = EXPERIMENT_ROOT / '41_channel_transition_balance_final.yaml'
SUMMARY_SCRIPT = EXPERIMENT_ROOT / 'scripts/summarize_validation.py'
FINAL_SUMMARY_SCRIPT = EXPERIMENT_ROOT / 'scripts/summarize_final_test.py'
LAYOUT_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'30_channel_benchmark_v1/02_layouts.yaml'
)
CLIS = (
	Path('proc/seis_ssl_cluster/cluster_embeddings.py'),
	Path('proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py'),
	Path('proc/seis_ssl_cluster/train_strat_hmm_pretext.py'),
	Path('proc/seis_ssl_cluster/extract_embeddings.py'),
	Path('proc/seis_ssl_cluster/run_parihaka_channel_decoder.py'),
)
CANDIDATE_MODELS = (
	'mae_hmm_k6_neutral',
	'mae_hmm_k6_persist003',
	'mae_hmm_k6_persist010',
	'local_barlow_twins_hmm_k6_neutral',
	'local_barlow_twins_hmm_k6_persist003',
	'local_barlow_twins_hmm_k6_persist010',
)
LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))


def _text() -> str:
	return README.read_text(encoding='utf-8')


def _section(text: str, start: str, end: str | None = None) -> str:
	section = text.split(start, maxsplit=1)[1]
	return section if end is None else section.split(end, maxsplit=1)[0]


def _inline_python(text: str) -> list[str]:
	return re.findall(r"python - <<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)


def _bash_blocks(text: str) -> str:
	return '\n'.join(re.findall(r'```bash\n(.*?)\n```', text, flags=re.DOTALL))


def _array_items(section: str, name: str) -> tuple[str, ...]:
	match = re.search(
		rf'{name}=\(\n(?P<body>.*?)\n\)',
		section,
		flags=re.DOTALL,
	)
	assert match is not None
	return tuple(
		item.strip().strip('"')
		for item in match.group('body').splitlines()
		if item.strip()
	)


def test_runbook_references_existing_configs_scripts_layout_and_clis() -> None:
	text = _text()
	assert README.is_file()
	for path in (*TARGET_CONFIGS, *EXPORT_SCRIPTS, *TRAINING_CONFIGS):
		assert path.is_file()
	for path in EXPORT_SCRIPTS:
		assert os.access(path, os.X_OK)
	for path in (
		*EXTRACTION_CONFIGS,
		CHANNEL_CONFIG,
		FINAL_CHANNEL_CONFIG,
		SUMMARY_SCRIPT,
		FINAL_SUMMARY_SCRIPT,
		LAYOUT_CONFIG,
		*CLIS,
	):
		assert path.is_file()

	for source in SOURCES:
		for variant in VARIANTS:
			assert (
				f'$TARGET_CONFIGS/{source}/{variant}/01_cluster_hmm_k6.yaml'
				in text
			)
			assert (
				f'$TARGET_CONFIGS/{source}/{variant}/02_export_pseudo_targets.sh'
				in text
			)
			assert (
				f'$STAGE2_CONFIGS/{source}/{variant}/01_full_25ep.yaml'
				in text
			)
	for path in EXTRACTION_CONFIGS:
		assert f'$EMBEDDING_CONFIGS/{path.name}' in text
	assert '$EXP/40_channel_transition_balance.yaml' in text
	assert str(LAYOUT_CONFIG) in text
	for cli in CLIS:
		assert str(cli) in text


def test_runbook_has_three_compilable_inline_python_blocks() -> None:
	blocks = _inline_python(_text())
	assert len(blocks) == 3
	for index, source in enumerate(blocks, start=1):
		compile(source, f'{README}:inline-{index}', 'exec')


def test_clustering_audit_uses_final_decode_occupancy_and_requires_targets() -> None:
	script = _inline_python(_text())[0]
	assert "metadata.get('cluster_counts')" in script
	assert "final_iteration.get('cluster_counts')" not in script
	assert "if not inputs:" in script
	assert 'pseudo-target inputs are missing' in script
	assert '{item.survey_id for item in inputs} != expected_surveys' in script
	assert "'empty_clusters'" in script
	assert "'total_center_shift_l2'" in script
	assert "'mean_boundaries_per_valid_trace'" in script


def test_runbook_states_transition_table_and_fixed_scientific_contract() -> None:
	text = _text()
	for row in (
		'| `advance_favored_m003` | 0.03 | 0.00 | -0.03 |',
		'| `neutral` | 0.00 | 0.00 | 0.00 |',
		'| `persist003` | 0.00 | 0.03 | +0.03 |',
		'| `persist010` | 0.00 | 0.10 | +0.10 |',
	):
		assert row in text
	for setting in (
		'K = 6',
		'iterations = 10',
		'anchors = 0.25 / 0.25',
		'expected boundaries = off',
		'max_jump = 1',
		'reverse forbidden',
		'boundary_alpha = 0.0',
		'distillation_weight = 0.2',
		'Stage 2 budget = 25 epochs / 15,625 steps',
		'downstream = frozen embedding decoder',
		'screening size = medium',
		'layouts = layout_000 ... layout_004',
		'selection metric = validation.channel_iou',
	):
		assert setting in text
	assert 'survey-specific' in text
	assert 'transductive' in text
	assert 'does not read test IoU' in text
	assert 'does not run `small` or `large`' in text


def test_runbook_uses_required_execution_order_and_six_conditions() -> None:
	text = _text()
	headings = (
		'## 1. Environment',
		'## 2. Targeted tests',
		'## 3. Build the six new clustering results and pseudo-targets',
		'## 4. Audit clustering and pseudo-target metadata',
		'## 5. Dry-run Stage 2 and optionally smoke one step',
		'## 6. Run the six full 25-epoch trainings',
		'## 7. Audit the six full checkpoints',
		'## 8. Extract the six new embedding volumes',
		'## 9. Audit all ten embedding sources',
		'## 10. Dry-run the 30 new medium decoder jobs',
		'## 11. Run the 30 new medium decoder jobs',
		'## 12. Write the validation screening summary',
	)
	positions = tuple(text.index(heading) for heading in headings)
	assert positions == tuple(sorted(positions))

	target_section = _section(text, headings[2], headings[3])
	assert len(_array_items(target_section, 'CLUSTER_CONFIGS')) == 6
	assert len(_array_items(target_section, 'EXPORT_SCRIPTS')) == 6
	assert target_section.count('cluster_embeddings.py') == 2
	assert '--dry-run' in target_section

	training_section = _section(text, headings[4], headings[5])
	assert len(_array_items(training_section, 'TRAINING_CONFIGS')) == 6
	assert '--max-steps 1' in training_section
	assert '--output-root "$output_root"' in training_section
	assert 'gpu_feasibility_1step.yaml' not in text

	extraction_section = _section(text, headings[7], headings[8])
	assert len(_array_items(extraction_section, 'EXTRACTION_CONFIGS')) == 6
	assert extraction_section.count('extract_embeddings.py') == 2
	assert '--dry-run' in extraction_section


def test_runbook_bash_blocks_fail_fast() -> None:
	blocks = re.findall(r'```bash\n(.*?)\n```', _text(), flags=re.DOTALL)
	assert blocks
	assert all(block.startswith('set -euo pipefail\n') for block in blocks)


def test_targeted_tests_cover_decoder_runner_and_cli() -> None:
	section = _section(
		_text(),
		'## 2. Targeted tests',
		'## 3. Build the six new clustering results and pseudo-targets',
	)
	assert 'tests/seis_ssl_cluster/test_parihaka_channel_decoder.py' in section
	assert 'tests/seis_ssl_cluster/test_proc_dry_run.py' in section
	assert (
		'tests/seis_ssl_cluster/'
		'test_parihaka_channel_hmm_transition_balance_final_summary.py' in section
	)
	assert '-k parihaka_channel_decoder' in section


def test_full_training_block_is_self_contained() -> None:
	section = _section(
		_text(),
		'## 6. Run the six full 25-epoch trainings',
		'## 7. Audit the six full checkpoints',
	)
	full_run = re.findall(r'```bash\n(.*?)\n```', section, flags=re.DOTALL)[0]
	assert len(_array_items(full_run, 'TRAINING_CONFIGS')) == 6
	assert 'for config in "${TRAINING_CONFIGS[@]}"' in full_run


def test_runbook_resume_examples_bind_each_run_to_its_own_latest() -> None:
	section = _section(
		_text(),
		'## 6. Run the six full 25-epoch trainings',
		'## 7. Audit the six full checkpoints',
	)
	pairs = re.findall(
		r'--config "\$STAGE2_CONFIGS/(?P<source>[^/]+)/(?P<variant>[^/]+)'
		r'/01_full_25ep\.yaml" \\\n'
		r'\s+--resume "\$STAGE2_ROOT/(?P=source)/(?P=variant)'
		r'/full_25ep/latest\.pt"',
		section,
	)
	assert set(pairs) == {
		(source, variant) for source in SOURCES for variant in VARIANTS
	}
	assert 'full_25ep/best.pt' not in section


def test_runbook_decoder_commands_target_only_six_candidates_and_medium() -> None:
	text = _text()
	preflight = _section(
		text,
		'## 10. Dry-run the 30 new medium decoder jobs',
		'## 11. Run the 30 new medium decoder jobs',
	)
	execution = _section(
		text,
		'## 11. Run the 30 new medium decoder jobs',
		'## 12. Write the validation screening summary',
	)
	assert _array_items(preflight, 'CANDIDATE_MODELS') == CANDIDATE_MODELS
	assert _array_items(preflight, 'LAYOUTS') == LAYOUTS
	assert _array_items(execution, 'CANDIDATE_MODELS') == CANDIDATE_MODELS
	assert _array_items(execution, 'LAYOUTS') == LAYOUTS
	for section in (preflight, execution):
		assert '--model "$model"' in section
		assert '--layout "$layout"' in section
		assert section.count('--size medium') == 1
		assert 'run_parihaka_channel_decoder.py' in section
		assert '--validation-only' in section
	assert '--dry-run' in preflight
	assert '--dry-run' not in _bash_blocks(execution)


def test_runbook_decoder_loop_skips_complete_and_stops_on_partial_job() -> None:
	section = _section(
		_text(),
		'## 11. Run the 30 new medium decoder jobs',
		'## 12. Write the validation screening summary',
	)
	block = re.findall(r'```bash\n(.*?)\n```', section, flags=re.DOTALL)[0]
	job_dir = '$VALIDATION_RUNS_ROOT/model=$model/layout=$layout/size=medium'
	assert f'job_dir="{job_dir}"' in block
	assert 'if [[ -f "$job_dir/metrics.json" ]]' in block
	assert 'continue' in block
	assert 'if [[ -f "$job_dir/latest.pt" ]]' in block
	assert 'incomplete Channel job requires explicit resume' in block
	assert 'exit 1' in block
	assert block.index('metrics.json') < block.index('latest.pt')
	assert (
		'--resume\n'
		'"$VALIDATION_RUNS_ROOT/model=<model>/layout=<layout>/'
		'size=medium/latest.pt"' in section
	)


def test_runbook_reuses_existing_models_without_rerunning_them() -> None:
	text = _text()
	assert 'existing H0 condition' in text
	assert 'reused in place and are not generated' in text
	assert 'again. The existing MAE and Local BT controls' in text
	assert 'existing MAE and Local BT controls and both existing H0 models' in text
	assert 'do not retrain, re-extract, or' in text
	assert 'rerun the decoder for any of those four models' in text
	assert (
		'existing H0 clustering result and pseudo-targets are not regenerated'
		in text
	)
	assert (
		'Do not re-extract `mae`, `mae_hmm_k6`, `local_barlow_twins`, or'
		in text
	)
	assert 'Do not rerun decoder jobs for the MAE control' in text
	assert 'Local BT control' in text
	assert 'existing H0 model' in text


def test_runbook_calls_executable_validation_summary_with_explicit_roots() -> None:
	section = _section(
		_text(),
		'## 12. Write the validation screening summary',
		'## 13. Final test protocol after all validation phases',
	)
	assert 'scripts/summarize_validation.py' in section
	assert '--existing-runs-root "$EXISTING_RUNS_ROOT"' in section
	assert '--validation-runs-root "$VALIDATION_RUNS_ROOT"' in section
	assert '--report-root "$REPORT_ROOT"' in section
	assert "python - <<'PY'" not in section
	assert (
		'tests/seis_ssl_cluster/'
		'test_parihaka_channel_hmm_transition_balance_summary.py' in _text()
	)


def test_final_test_protocol_uses_all_layouts_in_disjoint_normal_runs() -> None:
	text = _text()
	section = _section(text, '## 13. Final test protocol after all validation phases')
	assert 'Do not run this section during Phase 1' in section
	assert 'FINAL_LAYOUT' not in section
	assert section.count('export FINAL_MODEL=') == 1
	assert '--config "$FINAL_CHANNEL_CONFIG"' in section
	assert '--model "$FINAL_MODEL"' in section
	assert '--layout "$layout"' in section
	assert '--size medium' in section
	assert section.count('run_parihaka_channel_decoder.py') == 2
	blocks = re.findall(r'```bash\n(.*?)\n```', section, flags=re.DOTALL)
	assert len(blocks) == 4
	dry_run = blocks[1]
	normal_run = blocks[2]
	for block in (dry_run, normal_run):
		assert _array_items(block, 'LAYOUTS') == LAYOUTS
		assert 'for layout in "${LAYOUTS[@]}"' in block
		assert '--model "$FINAL_MODEL"' in block
		assert '--layout "$layout"' in block
	assert '--dry-run' in dry_run
	assert '--dry-run' not in normal_run
	assert '--validation-only' not in normal_run
	assert 'job_dir="$FINAL_RUNS_ROOT/model=$FINAL_MODEL/' in normal_run
	assert 'if [[ -f "$job_dir/metrics.json" ]]' in normal_run
	assert 'if [[ -f "$job_dir/latest.pt" ]]' in normal_run
	assert 'incomplete final test job requires explicit resume' in normal_run
	assert 'evaluation_mode` equal to' in section
	assert '`validation_and_test`' in section
	assert 'exactly five final test jobs' in section
	assert 'Layouts must not be ranked or selected after test evaluation' in section
	assert 'must not trigger another model, transition condition, layout' in section
	assert 'disjoint from historical' in section
	assert '/validation_runs"' in text
	assert '/final_runs"' in text


def test_final_test_protocol_calls_executable_summary_with_explicit_roots() -> None:
	section = _section(
		_text(),
		'## 13. Final test protocol after all validation phases',
	)
	assert 'scripts/summarize_final_test.py' in section
	assert '--runs-root "$FINAL_RUNS_ROOT"' in section
	assert '--model "$FINAL_MODEL"' in section
	assert '--report-root "$REPORT_ROOT/final_test"' in section
	assert 'final_test_layouts.csv' in section
	assert 'final_test_summary.json' in section
	assert 'final_test_summary.md' in section
	assert 'mean, median, and sample standard deviation' in section
	assert 'statistics.stdev()' in section


def test_runbook_avoids_other_sizes_unsafe_artifact_commands_and_summary_cli() -> None:
	text = _text()
	commands = _bash_blocks(text)
	assert not re.search(r'--size\s+(?:small|large)\b', commands)
	assert not re.search(r'(?m)^\s*for\s+size\s+in\s+.*\b(?:small|large)\b', commands)
	assert not re.search(
		r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync(?:\s|$)|ln\s+-s(?:\s|$))',
		commands,
	)
	assert text.count('scripts/summarize_validation.py') == 1
	assert text.count('scripts/summarize_final_test.py') == 1
	assert (
		'$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/'
		'ssl_hmm_four_way_v1/runs'
	) in text
	assert (
		'$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/'
		'hmm_transition_balance_v1/summary'
	) in text
	assert 'Phase 1 ends here' in text
	assert 'does not start Phase 2' in text
	assert 'Issue' not in text
