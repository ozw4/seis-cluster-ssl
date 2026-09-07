from __future__ import annotations

import re
from pathlib import Path

DOC_ROOT = Path('experiments/parihaka/facies_benchmark_v1')
MAE_README = (
	DOC_ROOT
	/ '20_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/README.md'
)
DOCUMENTS = (
	*(path for path in DOC_ROOT.rglob('README.md') if path != MAE_README),
	DOC_ROOT / '21_ssl_hmm_continuation_v1/RUNBOOK_HMM_K6.md',
)
MARKDOWN_LINK = re.compile(r'(?<!!)\[[^]]+\]\(([^)]+)\)')


def test_parihaka_experiment_document_links_resolve() -> None:
	for document in DOCUMENTS:
		assert document.is_file()
		for target in MARKDOWN_LINK.findall(document.read_text(encoding='utf-8')):
			target_path = target.split('#', maxsplit=1)[0]
			if not target_path or '://' in target_path or target_path.startswith(
				'mailto:'
			):
				continue
			assert (document.parent / target_path).is_file(), (
				f'{document}: broken link {target}'
			)


def test_parihaka_experiment_docs_do_not_embed_python_auditors() -> None:
	for document in DOCUMENTS:
		text = document.read_text(encoding='utf-8')
		assert 'python - <<' not in text
		assert 'from seis_ssl_cluster' not in text
