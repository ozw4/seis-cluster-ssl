from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'33_channel_mae_local_bt_four_way_v1'
)
PREVIOUS_EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'32_channel_ssl_hmm_four_way_v1'
)
README = EXPERIMENT_ROOT / 'README.md'
EXTRACTION_CONFIGS = (
	'01_extract_local_barlow_twins_embeddings.yaml',
	'02_extract_local_barlow_twins_hmm_k6_embeddings.yaml',
)
CHANNEL_CONFIG = '03_channel_mae_local_bt_four_way.yaml'
MODEL_IDS = {
	'mae',
	'local_barlow_twins',
	'mae_hmm_k6',
	'local_barlow_twins_hmm_k6',
}
LOCAL_MODEL_IDS = {
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
}
LAYOUT_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'30_channel_benchmark_v1/02_layouts.yaml'
)
SUMMARY_CLI = Path(
	'proc/seis_ssl_cluster/summarize_parihaka_channel_mae_local_bt.py'
)
DECODER_CLI = Path('proc/seis_ssl_cluster/run_parihaka_channel_decoder.py')
EXTRACTION_CLI = Path('proc/seis_ssl_cluster/extract_embeddings.py')


def test_runbook_references_configs_clis_and_compilable_inline_python() -> None:
	text = README.read_text(encoding='utf-8')

	for filename in (*EXTRACTION_CONFIGS, CHANNEL_CONFIG):
		assert filename in text
	for cli in (EXTRACTION_CLI, DECODER_CLI, SUMMARY_CLI):
		assert str(cli) in text
	blocks = re.findall(r"python - <<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
	assert len(blocks) == 2
	for index, source in enumerate(blocks, start=1):
		compile(source, f'{README}:inline-{index}', 'exec')


def test_runbook_referenced_configs_layout_and_clis_exist() -> None:
	for filename in (*EXTRACTION_CONFIGS, CHANNEL_CONFIG):
		assert (EXPERIMENT_ROOT / filename).is_file()
	assert (PREVIOUS_EXPERIMENT_ROOT / '05_channel_four_way.yaml').is_file()
	assert LAYOUT_CONFIG.is_file()
	for cli in (EXTRACTION_CLI, DECODER_CLI, SUMMARY_CLI):
		assert cli.is_file()


def test_runbook_limits_new_extraction_and_decoder_execution_to_local_models() -> None:
	text = README.read_text(encoding='utf-8')
	intro = text.split('## Environment', maxsplit=1)[0]
	intro_models = set(re.findall(r'^- `([^`]+)`$', intro, flags=re.MULTILINE))
	assert intro_models == MODEL_IDS

	extraction_references = re.findall(
		r'extract_embeddings\.py\s+\\\n\s+--config "\$EXP/([^"\s]+)"',
		text,
	)
	assert Counter(extraction_references) == Counter(
		{
			EXTRACTION_CONFIGS[0]: 2,
			EXTRACTION_CONFIGS[1]: 2,
		}
	)

	execution_section = text.split(
		'## 6. Run only the 30 new local-BT jobs', maxsplit=1
	)[1].split('## 7.', maxsplit=1)[0]
	loop_match = re.search(
		r'^for model in ([^;]+); do$', execution_section, re.MULTILINE
	)
	assert loop_match is not None
	assert set(loop_match.group(1).split()) == LOCAL_MODEL_IDS
	assert '--model "$model"' in execution_section
	assert '--model mae ' not in execution_section
	assert '--model mae_hmm_k6 ' not in execution_section


def test_runbook_reuses_existing_roots_and_fixed_budget_sources_safely() -> None:
	text = README.read_text(encoding='utf-8')

	assert (
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/'
		'ssl_hmm_four_way_v1/runs'
	) in text
	assert (
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/'
		'mae_local_bt_four_way_v1/summary'
	) in text
	assert 'full_25ep/latest.pt' in text
	assert 'full_25ep/best.pt' not in text
	assert '02_layouts.yaml' in text
	assert 'reviewed layout' in text
	assert 'Do not re-extract the MAE' in text
	assert 'rerun the MAE decoder jobs' in text
	assert not re.search(r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync\s)', text)
	assert 'exit 1' not in text


def test_runbook_audits_sixty_metrics_before_writing_separate_summary() -> None:
	text = README.read_text(encoding='utf-8')
	summary_section = text.split(
		'## 7. Audit all 60 metrics and write the summary', maxsplit=1
	)[1]
	invocations = re.findall(
		r'python proc/seis_ssl_cluster/'
		r'summarize_parihaka_channel_mae_local_bt\.py\s+\\\n'
		r'\s+--config "\$EXP/03_channel_mae_local_bt_four_way\.yaml"'
		r'(?P<dry_run>\s+\\\n\s+--dry-run)?',
		summary_section,
	)
	assert invocations == [' \\\n  --dry-run', '']
	assert summary_section.index('--dry-run') < summary_section.rindex(
		'summarize_parihaka_channel_mae_local_bt.py'
	)
	assert 'complete_jobs: 60' in summary_section
	for name in ('comparison.csv', 'summary.json', 'summary.md'):
		assert name in summary_section
