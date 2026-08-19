from __future__ import annotations

import re
from pathlib import Path

SUITE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1'
)
README = SUITE_ROOT / 'README.md'
RUNBOOK = SUITE_ROOT / 'RUNBOOK_HMM_K6.md'


def test_readme_links_hmm_runbook() -> None:
	text = README.read_text(encoding='utf-8')

	assert '[RUNBOOK_HMM_K6.md](RUNBOOK_HMM_K6.md)' in text


def test_runbook_referenced_configs_and_scripts_exist() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	roots = {
		'TARGET_CONFIGS': SUITE_ROOT / '20_hmm_targets',
		'HMM_CONFIGS': SUITE_ROOT / '30_stage2',
	}
	references = re.findall(
		r'\$\{?(TARGET_CONFIGS|HMM_CONFIGS)\}?/([^"\s]+\.(?:yaml|sh))',
		text,
	)

	assert references
	for variable, relative_path in references:
		assert (roots[variable] / relative_path).is_file()

	for cli in (
		'extract_embeddings.py',
		'cluster_embeddings.py',
		'export_strat_hmm_pseudo_targets.py',
		'train_strat_hmm_pretext.py',
	):
		assert f'proc/seis_ssl_cluster/{cli}' in text


def test_runbook_inline_python_blocks_compile() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	blocks = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)

	for index, source in enumerate(blocks, start=1):
		compile(source, f'{RUNBOOK}:inline-{index}', 'exec')
