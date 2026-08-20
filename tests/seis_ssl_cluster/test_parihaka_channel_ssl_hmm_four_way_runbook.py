from __future__ import annotations

import re
from pathlib import Path

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'32_channel_ssl_hmm_four_way_v1'
)
README = EXPERIMENT_ROOT / 'README.md'
EXTRACTION_CONFIGS = (
	'01_extract_mae_embeddings.yaml',
	'02_extract_barlow_twins_embeddings.yaml',
	'03_extract_mae_hmm_k6_embeddings.yaml',
	'04_extract_barlow_twins_hmm_k6_embeddings.yaml',
)
CHANNEL_CONFIG = '05_channel_four_way.yaml'
LAYOUT_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'30_channel_benchmark_v1/02_layouts.yaml'
)


def test_runbook_references_configs_runner_summary_and_valid_inline_python() -> None:
	text = README.read_text(encoding='utf-8')

	for filename in (*EXTRACTION_CONFIGS, CHANNEL_CONFIG):
		assert filename in text
	for cli in (
		'extract_embeddings.py',
		'run_parihaka_channel_decoder.py',
		'summarize_parihaka_channel_ssl_hmm.py',
	):
		assert f'proc/seis_ssl_cluster/{cli}' in text
	blocks = re.findall(r"python - <<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
	assert len(blocks) == 1
	compile(blocks[0], f'{README}:live-embedding-audit', 'exec')


def test_runbook_referenced_configs_layout_and_clis_exist() -> None:
	for filename in (*EXTRACTION_CONFIGS, CHANNEL_CONFIG):
		assert (EXPERIMENT_ROOT / filename).is_file()
	assert LAYOUT_CONFIG.is_file()
	for cli in (
		'extract_embeddings.py',
		'run_parihaka_channel_decoder.py',
		'summarize_parihaka_channel_ssl_hmm.py',
	):
		assert (Path('proc/seis_ssl_cluster') / cli).is_file()
