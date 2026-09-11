from __future__ import annotations

import re
import subprocess
from pathlib import Path

SUITE_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1'
)
README = SUITE_ROOT / 'README.md'
RUNBOOK = SUITE_ROOT / 'RUNBOOK_HMM_K6.md'
F3_README = Path('experiments/f3/facies_benchmark_v1/README.md')


def test_readmes_link_the_suite_and_execution_entrypoints() -> None:
	readme_text = README.read_text(encoding='utf-8')
	root_text = F3_README.read_text(encoding='utf-8')

	assert '[RUNBOOK_HMM_K6.md](RUNBOOK_HMM_K6.md)' in readme_text
	assert (
		'[21_ssl_hmm_continuation_v1]('
		'21_ssl_hmm_continuation_v1/README.md)'
	) in root_text
	assert RUNBOOK.is_file()


def test_runbook_references_exist_and_shell_blocks_are_valid() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	roots = {
		'STAGE1_CONFIGS': SUITE_ROOT / '10_stage1',
		'TARGET_CONFIGS': SUITE_ROOT / '20_hmm_targets',
		'STAGE2_CONFIGS': SUITE_ROOT / '30_stage2',
	}
	references = re.findall(
		r'\$\{?(STAGE1_CONFIGS|TARGET_CONFIGS|STAGE2_CONFIGS)\}?/'
		r'([^"\s]+\.(?:yaml|sh))',
		text,
	)
	assert references
	for variable, relative_path in references:
		assert (roots[variable] / relative_path).is_file()
	for cli in set(re.findall(r'proc/seis_ssl_cluster/\w+\.py', text)):
		assert Path(cli).is_file(), cli

	headings = (
		'## Stage 1',
		'## Control branches',
		'## HMM target branches',
		'## 再開',
	)
	positions = [text.index(heading) for heading in headings]
	assert positions == sorted(positions)

	blocks = re.findall(r'```bash\n(.*?)```', text, flags=re.DOTALL)
	assert blocks
	for block in blocks:
		subprocess.run(
			['/bin/bash', '-n'],
			input=block,
			text=True,
			check=True,
			capture_output=True,
		)
