from __future__ import annotations

import re
import subprocess
from pathlib import Path

RUNBOOK = Path(
	'experiments/f3/facies_benchmark_v2/LOCAL_BT_CANDIDATE_RUNBOOK.md'
)


def test_shared_candidate_runbook_references_existing_files_and_valid_shell() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')

	clis = set(re.findall(r'proc/seis_ssl_cluster/[a-z0-9_]+\.py', text))
	assert clis
	for cli in clis:
		assert Path(cli).is_file(), cli

	links = re.findall(r'\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)', text)
	for link in links:
		assert (RUNBOOK.parent / link).resolve().exists(), link

	blocks = re.findall(r'```bash\n(.*?)```', text, flags=re.DOTALL)
	assert len(blocks) == 1
	subprocess.run(
		['/bin/bash', '-n'],
		input=blocks[0],
		text=True,
		check=True,
		capture_output=True,
	)
