from __future__ import annotations

import re
from pathlib import Path

SUITE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1'
)
README = SUITE_ROOT / 'README.md'
RUNBOOK = SUITE_ROOT / 'RUNBOOK_HMM_K6.md'
LOCAL_SOURCE = (
	'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/'
	'facies_benchmark_v1/ssl_hmm_continuation_v1/stage1/'
	'local_barlow_twins_v1/full_100ep/latest.pt'
)


def test_readme_links_hmm_runbook() -> None:
	text = README.read_text(encoding='utf-8')

	assert '[RUNBOOK_HMM_K6.md](RUNBOOK_HMM_K6.md)' in text
	for expected in (
		'10_stage1/local_barlow_twins_v1',
		'20_hmm_targets/local_bt100',
		'30_stage2/local_bt100',
		'local BT100 → local BT25 control',
		'local BT100 → HMM-K6 25',
	):
		assert expected in text
	assert '22_local_barlow_twins_v1' not in text


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
		'train_amp_barlow_twins.py',
	):
		assert f'proc/seis_ssl_cluster/{cli}' in text


def test_runbook_keeps_local_stage2_branches_independent() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	bt_config = (
		SUITE_ROOT
		/ '30_stage2/local_bt100/bt_continue/02_full_25ep.yaml'
	).read_text(encoding='utf-8')
	hmm_config = (
		SUITE_ROOT / '30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml'
	).read_text(encoding='utf-8')

	assert (
		'LOCAL_BT100="$ARTIFACT_SUITE/stage1/'
		'local_barlow_twins_v1/full_100ep/latest.pt"'
	) in text
	assert LOCAL_SOURCE in bt_config
	assert hmm_config.count(LOCAL_SOURCE) == 2
	assert 'stage2/local_bt100/bt_continue' not in hmm_config
	assert 'pseudo_targets/parihaka/facies_benchmark_v1/' in hmm_config
	assert 'ssl_hmm_continuation_v1/local_bt100' in hmm_config
	assert 'stage2/local_bt100/bt_continue' in text
	assert 'stage2/local_bt100/hmm/k6' in text
	assert 'fresh full run' in text
	assert 'Stage 1 sourceを--resumeへ渡さない' in text
	assert not re.search(r'(?m)^(?:python|bash).*best\.pt', text)
	assert '22_local_barlow_twins_v1' not in text


def test_runbook_audits_local_source_identity() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')

	for expected in (
		"source['epoch'] == 100",
		"source['global_step'] == 62_500",
		"source['pretraining_method'] == 'local_barlow_twins_3d'",
		"source['checkpoint_kind'] == 'barlow_twins_pretraining'",
		"source['config']['barlow_twins']['local_pairs_per_crop'] == 128",
		"source['amp_enabled'] is False",
		"source['training_state']['completed_epoch'] is True",
		"source['projector_state_dict']",
	):
		assert expected in text
	assert 'resolve_barlow_twins_pretraining_method(payload[\'config\'])' in text
	assert "config['pseudo_targets']['k'] == 6" in text


def test_runbook_inline_python_blocks_compile() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	blocks = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)

	for index, source in enumerate(blocks, start=1):
		compile(source, f'{RUNBOOK}:inline-{index}', 'exec')
