from __future__ import annotations

import re
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
	for relative_path in (
		'10_stage1/mae/02_full_100ep.yaml',
		'10_stage1/barlow_twins/02_full_100ep.yaml',
		'30_stage2/mae100/mae_continue/02_full_25ep.yaml',
		'30_stage2/bt100/bt_continue/02_full_25ep.yaml',
		'20_hmm_targets/mae100/01_extract_embeddings.yaml',
		'20_hmm_targets/bt100/01_extract_embeddings.yaml',
		'30_stage2/mae100/hmm/k6/02_full_25ep.yaml',
		'30_stage2/bt100/hmm/k6/02_full_25ep.yaml',
	):
		assert relative_path in readme_text
		assert (SUITE_ROOT / relative_path).is_file()
	for cli in (
		'train_amp_mae.py',
		'train_amp_barlow_twins.py',
		'extract_embeddings.py',
		'cluster_embeddings.py',
		'export_strat_hmm_pseudo_targets.py',
		'train_strat_hmm_pretext.py',
	):
		relative_path = f'proc/seis_ssl_cluster/{cli}'
		assert relative_path in readme_text
		assert Path(relative_path).is_file()


def test_runbook_references_exist_and_inline_audits_compile() -> None:
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

	focused_tests = (
		'tests/seis_ssl_cluster/test_f3_stage1_ssl_configs.py',
		'tests/seis_ssl_cluster/test_f3_ssl_continuation_configs.py',
		'tests/seis_ssl_cluster/test_f3_hmm_k6_target_configs.py',
		'tests/seis_ssl_cluster/test_f3_hmm_k6_configs.py',
		'tests/seis_ssl_cluster/test_f3_ssl_hmm_runbook.py',
	)
	for test_path in focused_tests:
		assert text.count(test_path) == 1
	for test_path in (
		'tests/seis_ssl_cluster/test_mae_continuation_runner.py',
		'tests/seis_ssl_cluster/test_barlow_twins_continuation.py',
		'tests/seis_ssl_cluster/test_barlow_twins_training_contract.py',
		'tests/seis_ssl_cluster/test_embedding_extractor.py',
		'tests/seis_ssl_cluster/test_strat_checkpoint_extraction.py',
		'tests/seis_ssl_cluster/test_strat_hmm_pretraining_head_only.py',
		'tests/seis_ssl_cluster/test_strat_hmm_barlow_runner_integration.py',
	):
		assert text.count(test_path) == 1
		assert Path(test_path).is_file()

	blocks = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
	assert blocks
	for index, source in enumerate(blocks, start=1):
		compile(source, f'{RUNBOOK}:inline-{index}', 'exec')
