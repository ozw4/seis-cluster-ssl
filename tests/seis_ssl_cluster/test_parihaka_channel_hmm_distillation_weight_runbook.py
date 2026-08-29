from __future__ import annotations

import re
from pathlib import Path

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'38_channel_hmm_distillation_weight_v1'
)
README = EXPERIMENT_ROOT / 'README.md'
TRAINING_CONFIGS = tuple(
	EXPERIMENT_ROOT
	/ '20_stage2'
	/ source
	/ variant
	/ '01_full_25ep.yaml'
	for source in ('mae100', 'local_bt100')
	for variant in ('distill005', 'distill010', 'distill040')
)
EXTRACTION_CONFIGS = tuple(
	EXPERIMENT_ROOT / '30_embeddings' / filename
	for filename in (
		'01_extract_mae_hmm_k6_distill005.yaml',
		'02_extract_mae_hmm_k6_distill010.yaml',
		'03_extract_mae_hmm_k6_distill040.yaml',
		'04_extract_local_barlow_twins_hmm_k6_distill005.yaml',
		'05_extract_local_barlow_twins_hmm_k6_distill010.yaml',
		'06_extract_local_barlow_twins_hmm_k6_distill040.yaml',
	)
)
CHANNEL_CONFIG = EXPERIMENT_ROOT / '40_channel_distillation_weight.yaml'
SUMMARY_SCRIPT = EXPERIMENT_ROOT / 'scripts/summarize_validation.py'
LAYOUT_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'30_channel_benchmark_v1/02_layouts.yaml'
)
CANDIDATE_MODELS = (
	'mae_hmm_k6_distill005',
	'mae_hmm_k6_distill010',
	'mae_hmm_k6_distill040',
	'local_barlow_twins_hmm_k6_distill005',
	'local_barlow_twins_hmm_k6_distill010',
	'local_barlow_twins_hmm_k6_distill040',
)
LAYOUTS = tuple(f'layout_{index:03d}' for index in range(5))
CONDITIONS = tuple(
	f'{source}/{variant}'
	for source in ('mae100', 'local_bt100')
	for variant in ('distill005', 'distill010', 'distill040')
)


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


def _associative_items(section: str, name: str) -> dict[str, str]:
	match = re.search(
		rf'declare -A {name}=\(\n(?P<body>.*?)\n\)',
		section,
		flags=re.DOTALL,
	)
	assert match is not None
	items = re.findall(r'^\s*\[([^]]+)]="([^"]+)"$', match.group('body'), re.MULTILINE)
	return dict(items)


def test_runbook_references_every_phase_file_and_runtime_entrypoint() -> None:
	assert README.is_file()
	for path in (
		*TRAINING_CONFIGS,
		*EXTRACTION_CONFIGS,
		CHANNEL_CONFIG,
		SUMMARY_SCRIPT,
		LAYOUT_CONFIG,
		Path('proc/seis_ssl_cluster/train_strat_hmm_pretext.py'),
		Path('proc/seis_ssl_cluster/extract_embeddings.py'),
		Path('proc/seis_ssl_cluster/run_parihaka_channel_decoder.py'),
	):
		assert path.is_file()


def test_runbook_has_twelve_ordered_sections() -> None:
	headings = re.findall(r'^## (\d+)\. .+$', _text(), flags=re.MULTILINE)
	assert headings == [str(index) for index in range(1, 13)]


def test_runbook_has_exactly_two_compilable_inline_python_audits() -> None:
	blocks = _inline_python(_text())
	assert len(blocks) == 2
	for index, source in enumerate(blocks, start=1):
		compile(source, f'{README}:inline-{index}', 'exec')


def test_runbook_states_weight_table_and_fixed_scientific_contract() -> None:
	text = _text()
	for row in (
		'| `distill005` | 0.05 | 新規 |',
		'| `distill010` | 0.10 | 新規 |',
		'| `distill020` | 0.20 | 既存H0を再利用 |',
		'| `distill040` | 0.40 | 新規 |',
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
		'boundary_alpha = 0.0',
		'boundary_tau = 1.0',
		'Stage 2 = 25 epochs / 15,625 steps',
		'student.unfreeze_top_blocks = 1',
		'downstream = frozen embedding decoder',
		'screening = medium, five layouts',
		'layouts = layout_000 ... layout_004',
		'metric = validation.channel_iou',
	):
		assert setting in text
	assert 'survey-specific' in text
	assert 'transductive' in text
	assert 'validation-only' in text
	assert 'do not run test inference or store test metrics' in text


def test_stage2_dry_run_smoke_and_full_run_list_exact_six_configs() -> None:
	text = _text()
	dry_run = _section(
		text,
		'## 3. Dry-run the six Stage 2 trainings',
		'## 4. Optionally smoke one step',
	)
	smoke = _section(
		text,
		'## 4. Optionally smoke one step',
		'## 5. Run or explicitly resume the six full 25-epoch trainings',
	)
	full_run = _section(
		text,
		'## 5. Run or explicitly resume the six full 25-epoch trainings',
		'## 6. Audit the six Stage 2 checkpoints',
	)
	expected = tuple(
		f'$STAGE2_CONFIGS/{path.relative_to(EXPERIMENT_ROOT / "20_stage2")}'
		for path in TRAINING_CONFIGS
	)
	assert _array_items(dry_run, 'TRAINING_CONFIGS') == expected
	assert '--dry-run' in dry_run
	assert _array_items(full_run, 'TRAINING_CONFIGS') == expected
	assert 'for config in "${TRAINING_CONFIGS[@]}"' in full_run
	assert '--resume' not in _bash_blocks(full_run)[0]
	smoke_specs = _array_items(smoke, 'SMOKE_SPECS')
	assert len(smoke_specs) == 6
	assert all('|$SMOKE_ROOT/' in spec for spec in smoke_specs)
	assert '--max-steps 1' in smoke
	assert '--output-root "$output_root"' in smoke
	assert 'no smoke YAML is added' in smoke


def test_resume_map_binds_every_condition_to_its_own_latest() -> None:
	section = _section(
		_text(),
		'## 5. Run or explicitly resume the six full 25-epoch trainings',
		'## 6. Audit the six Stage 2 checkpoints',
	)
	configs = _associative_items(section, 'RESUME_CONFIGS')
	checkpoints = _associative_items(section, 'RESUME_CHECKPOINTS')
	assert tuple(configs) == CONDITIONS
	assert tuple(checkpoints) == CONDITIONS
	for condition in CONDITIONS:
		assert configs[condition] == (
			f'$STAGE2_CONFIGS/{condition}/01_full_25ep.yaml'
		)
		assert checkpoints[condition] == (
			f'$STAGE2_ROOT/{condition}/full_25ep/latest.pt'
		)
	assert '${RESUME_CONDITION:?' in section
	assert '--config "$resume_config"' in section
	assert '--resume "$resume_checkpoint"' in section
	assert 'never discovers or resumes a run automatically' in section
	assert 'best.pt' in section
	assert 'full_25ep/best.pt' not in section


def test_checkpoint_and_embedding_audits_cover_required_identities() -> None:
	checkpoint_audit, embedding_audit = _inline_python(_text())
	for contract in (
		"payload.get('epoch') != 25",
		"payload.get('global_step') != 15_625",
		"payload.get('amp_enabled') is not False",
		"training_state.get('checkpoint_kind') != 'epoch'",
		"training_state.get('batch_index') is not None",
		"teacher.get('checkpoint')",
		"student.get('init_checkpoint')",
		"pseudo_targets.get('input_dir')",
		"pseudo_targets.get('k') != 6",
		"loss.get('distillation_weight') != variant_weights[variant]",
		"student.get('unfreeze_top_blocks') != 1",
	):
		assert contract in checkpoint_audit
	assert "'distill005': 0.05" in checkpoint_audit
	assert "'distill010': 0.10" in checkpoint_audit
	assert "'distill040': 0.40" in checkpoint_audit
	assert 'inspect_embedding_sources(config)' in embedding_audit
	assert 'np.array_equal(reference_valid, valid)' in embedding_audit
	assert "item.model_source['checkpoint_path']" in embedding_audit
	assert 'len(signatures) != 1' in embedding_audit
	for model in (
		'mae',
		'mae_hmm_k6',
		*CANDIDATE_MODELS[:3],
		'local_barlow_twins',
		'local_barlow_twins_hmm_k6',
		*CANDIDATE_MODELS[3:],
	):
		assert f"'{model}'," in embedding_audit


def test_extraction_loop_lists_only_six_new_configs() -> None:
	section = _section(
		_text(),
		'## 7. Extract the six new embedding volumes',
		'## 8. Audit all ten embedding sources',
	)
	expected = tuple(f'$EMBEDDING_CONFIGS/{path.name}' for path in EXTRACTION_CONFIGS)
	assert _array_items(section, 'EXTRACTION_CONFIGS') == expected
	assert section.count('extract_embeddings.py') == 2
	assert '--dry-run' in section
	assert 'Do not\nre-extract the MAE control' in section


def test_decoder_loops_target_only_six_models_five_layouts_and_medium() -> None:
	text = _text()
	preflight = _section(
		text,
		'## 9. Dry-run the 30 new medium decoder jobs',
		'## 10. Run the 30 new medium decoder jobs',
	)
	execution = _section(
		text,
		'## 10. Run the 30 new medium decoder jobs',
		'## 11. Write the validation screening summary',
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
	assert len(CANDIDATE_MODELS) * len(LAYOUTS) == 30


def test_decoder_loop_skips_complete_and_stops_on_partial() -> None:
	section = _section(
		_text(),
		'## 10. Run the 30 new medium decoder jobs',
		'## 11. Write the validation screening summary',
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
	assert 'never resumed\nautomatically' in section


def test_summary_call_uses_three_explicit_roots_and_fifty_metrics() -> None:
	section = _section(
		_text(),
		'## 11. Write the validation screening summary',
		'## 12. End the Phase',
	)
	assert 'scripts/summarize_validation.py' in section
	assert '--existing-runs-root "$EXISTING_RUNS_ROOT"' in section
	assert '--validation-runs-root "$VALIDATION_RUNS_ROOT"' in section
	assert '--report-root "$REPORT_ROOT"' in section
	assert 'ssl_hmm_four_way_v1/runs' in section
	assert 'hmm_distillation_weight_v1/validation_runs' in section
	assert 'hmm_distillation_weight_v1/summary' in section
	assert 'exactly 50 metrics' in section
	assert 'validation.channel_iou' in section
	assert "python - <<'PY'" not in section


def test_runbook_omits_reused_jobs_forbidden_stages_and_unsafe_commands() -> None:
	text = _text()
	commands = '\n'.join(_bash_blocks(text))
	assert 'distill020' not in commands
	assert 'cluster_embeddings.py' not in commands
	assert 'export_strat_hmm_pseudo_targets.py' not in commands
	assert not re.search(r'--size\s+(?:small|large)\b', commands)
	assert not re.search(
		r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync(?:\s|$)|ln\s+-s(?:\s|$))',
		commands,
	)
	assert 'proc/seis_ssl_cluster/summar' not in text
	assert 'Do not read test results or start final-test inference' in text
	assert 'Do not run `small`, `large`, or multi-head screening' in text
	assert 'Do not start another Phase automatically' in text
	assert 'Issue' not in text


def test_every_bash_block_is_self_contained() -> None:
	blocks = _bash_blocks(_text())
	assert len(blocks) == 12
	for block in blocks:
		assert block.splitlines()[:3] == [
			'set -euo pipefail',
			'REPO_ROOT=$(git rev-parse --show-toplevel)',
			'cd "$REPO_ROOT"',
		]
	assert '/workspace' not in _text()
	assert 'export CUDA_VISIBLE_DEVICES=' not in _text()


def test_targeted_tests_cover_new_contracts_decoder_and_unfreezing() -> None:
	section = _section(
		_text(),
		'## 2. Targeted tests',
		'## 3. Dry-run the six Stage 2 trainings',
	)
	for filename in (
		'test_parihaka_hmm_distillation_weight_configs.py',
		'test_parihaka_channel_hmm_distillation_weight_configs.py',
		'test_parihaka_channel_hmm_distillation_weight_summary.py',
		'test_parihaka_channel_hmm_distillation_weight_runbook.py',
		'test_parihaka_channel_decoder.py',
		'test_strat_hmm_pretraining_head_only.py',
	):
		assert filename in section
	assert '-k "distillation or unfreeze"' in section
