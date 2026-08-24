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
	for path in (*EXTRACTION_CONFIGS, CHANNEL_CONFIG, LAYOUT_CONFIG, *CLIS):
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


def test_runbook_has_four_compilable_inline_python_blocks() -> None:
	blocks = _inline_python(_text())
	assert len(blocks) == 4
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
	for section in (preflight, execution):
		assert '--model "$model"' in section
		assert '--layout "$layout"' in section
		assert section.count('--size medium') == 1
		assert 'run_parihaka_channel_decoder.py' in section
	assert '--dry-run' in preflight
	assert '--dry-run' not in _bash_blocks(execution)


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


def test_screening_reads_validation_only_and_validates_exact_matrix() -> None:
	script = _inline_python(_text())[-1]
	assert "payload.get('validation')" in script
	assert "validation.get('channel_iou')" in script
	assert "'metric': 'validation.channel_iou'" in script
	for forbidden in (
		"payload.get('test')",
		'payload.get("test")',
		"payload['test']",
		'payload["test"]',
		'test.channel_iou',
	):
		assert forbidden not in script
	assert 'expected_metric_count = 50' in script
	assert 'len(model_ids) != 10' in script
	assert 'len(variant_order) != 4' in script
	assert "tuple(branches) != ('mae', 'local_bt')" in script
	for model in (
		'mae',
		'mae_hmm_k6',
		'local_barlow_twins',
		'local_barlow_twins_hmm_k6',
		*CANDIDATE_MODELS,
	):
		assert f"'{model}'" in script
	for layout in LAYOUTS:
		assert f"'{layout}'" in script
	assert "payload.get('model')" in script
	assert "payload.get('layout_id')" in script
	assert "payload.get('data_size') != 'medium'" in script
	assert 'paired_identity(metrics[(model, layout)])' in script


def test_screening_computes_required_statistics_and_recommendation_rule() -> None:
	script = _inline_python(_text())[-1]
	for statistic in (
		'statistics.mean',
		'statistics.median',
		'statistics.stdev',
		"'sample_standard_deviation'",
		"'wins'",
		"'ties'",
		"'losses'",
		"'layout_gains'",
	):
		assert statistic in script
	assert 'eligible = mae_mean >= 0.0 and local_bt_mean >= 0.0' in script
	assert 'eligible_variants' in script
	assert 'ranking = sorted(' in script
	assert "'combined'" in script
	assert "['mean']" in script
	assert "['median']" in script
	assert 'variant_order.index(variant)' in script
	assert 'recommended_variant = ranking[0] if ranking else None' in script
	for key in (
		'metric',
		'data_size',
		'variant_transition_settings',
		'per_variant',
		'ranking',
		'recommended_variant',
		'selection_rule',
	):
		assert f"'{key}'" in script
	for filename in ('screening_validation.json', 'screening_validation.md'):
		assert f"'{filename}'" in script


def test_runbook_avoids_other_sizes_unsafe_artifact_commands_and_summary_cli() -> None:
	text = _text()
	commands = _bash_blocks(text)
	assert not re.search(r'--size\s+(?:small|large)\b', commands)
	assert not re.search(r'(?m)^\s*for\s+size\s+in\s+.*\b(?:small|large)\b', commands)
	assert not re.search(
		r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync(?:\s|$)|ln\s+-s(?:\s|$))',
		commands,
	)
	assert 'summarize_' not in text
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
