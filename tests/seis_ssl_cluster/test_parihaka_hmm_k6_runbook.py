from __future__ import annotations

import re
from pathlib import Path

from seis_ssl_cluster.stratigraphy import pseudo_target_paths

SUITE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1'
)
README = SUITE_ROOT / 'README.md'
RUNBOOK = SUITE_ROOT / 'RUNBOOK_HMM_K6.md'


def test_suite_readme_links_the_compact_paired_hmm_matrix() -> None:
	text = README.read_text(encoding='utf-8')
	matrix = text.split('## Paired HMM-K6', maxsplit=1)[1].split(
		'## 学習条件', maxsplit=1
	)[0]

	assert '[RUNBOOK_HMM_K6.md](RUNBOOK_HMM_K6.md)' in text
	for row in (
		'MAE100 → HMM-K6 25',
		'BT100 → HMM-K6 25',
		'MAE100 → MAE25 control',
		'BT100 → BT25 control',
	):
		assert matrix.count(row) == 1


def test_hmm_k6_runbook_fixes_all_pipeline_config_paths() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	expected_paths = (
		'20_hmm_targets/mae100/01_extract_embeddings.yaml',
		'20_hmm_targets/bt100/01_extract_embeddings.yaml',
		'20_hmm_targets/mae100/k6/02_cluster_hmm_k6.yaml',
		'20_hmm_targets/bt100/k6/02_cluster_hmm_k6.yaml',
		'20_hmm_targets/mae100/k6/03_export_pseudo_targets.sh',
		'20_hmm_targets/bt100/k6/03_export_pseudo_targets.sh',
		'30_stage2/mae100/hmm/k6/01_gpu_feasibility_1step.yaml',
		'30_stage2/mae100/hmm/k6/02_full_25ep.yaml',
		'30_stage2/bt100/hmm/k6/01_gpu_feasibility_1step.yaml',
		'30_stage2/bt100/hmm/k6/02_full_25ep.yaml',
	)

	for relative_path in expected_paths:
		assert (SUITE_ROOT / relative_path).is_file()
		config_relative = relative_path.removeprefix('20_hmm_targets/')
		config_relative = config_relative.removeprefix('30_stage2/')
		assert config_relative in text

	for variant in ('mae100', 'bt100'):
		embedding_output = (
			'embeddings/parihaka/facies_benchmark_v1/'
			'ssl_hmm_continuation_v1/hmm_targets/'
			f'{variant}/overlap_x64'
		)
		assert embedding_output in text
		assert f'{variant}/k6/overlap_x64' not in text


def test_hmm_k6_runbook_keeps_the_required_execution_order() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	headings = tuple(
		f'## {index}. {label}'
		for index, label in enumerate(
			(
				'環境',
				'targeted tests',
				'Stage 1 source監査',
				'embedding抽出',
				'K=6 clustering',
				'pseudo target export',
				'target live監査',
				'HMM config dry-run',
				'GPU feasibility',
				'25 epoch full runとresume',
				'full checkpoint監査',
				'encoder consumer監査',
			),
			start=1,
		)
	)
	positions = [text.index(heading) for heading in headings]

	assert positions == sorted(positions)


def test_hmm_k6_runbook_fixes_source_target_and_resume_bindings() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	for binding in (
		'stage1/mae/full_100ep/latest.pt',
		'stage1/barlow_twins/full_100ep/latest.pt',
		'stage2/mae100/hmm/k6/full_25ep/latest.pt',
		'stage2/bt100/hmm/k6/full_25ep/latest.pt',
	):
		assert binding in text

	assert '--resume "$ARTIFACT_SUITE/stage2/mae100/hmm/k6/full_25ep/latest.pt"' in text
	assert '--resume "$ARTIFACT_SUITE/stage2/bt100/hmm/k6/full_25ep/latest.pt"' in text
	assert 'Stage 1 `latest.pt`はweights-only' in text
	assert 'MAE25 / BT25 controlも`--resume`へ渡さない' in text
	assert '`best.pt`は診断専用' in text


def test_hmm_k6_runbook_uses_base_pseudo_target_roots() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	for variable, variant in (
		('MAE_TARGET_ROOT', 'mae100'),
		('BT_TARGET_ROOT', 'bt100'),
	):
		pattern = rf'export {variable}="\$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/(.+)"'
		match = re.search(pattern, text)
		assert match is not None
		root = Path('/artifact-root') / match.group(1)
		assert root.name == variant
		assert root.name != 'k6'
		paths = pseudo_target_paths(root, k=6, survey_id='survey')
		assert paths.labels.parent == root / 'k6'
		assert paths.labels.parent != root / 'k6' / 'k6'
		assert f'--pseudo-target-root "${variable}"' in text

	assert '$MAE_TARGET_ROOT/k6' in text
	assert '$BT_TARGET_ROOT/k6' in text


def test_hmm_k6_runbook_uses_existing_clis_and_live_audit_apis() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	for cli in (
		'proc/seis_ssl_cluster/extract_embeddings.py',
		'proc/seis_ssl_cluster/cluster_embeddings.py',
		'proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py',
		'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
	):
		assert cli in text
	for api in (
		'discover_pseudo_target_inputs',
		'load_pseudo_target_arrays',
		'build_strat_hmm_components',
		'build_model_from_checkpoint_payload',
		'_stratigraphy_pretext_metadata',
	):
		assert api in text
	for audit_token in (
		"assert np.all(occupancy > 0)",
		"assert optimizer_steps == {15_625}",
		"assert metadata['base_objective'] == base_objective",
		"assert 'projector_state_dict' not in payload",
	):
		assert audit_token in text
	schema_flags = tuple(re.finditer('--schema-version 2', text))
	assert len(schema_flags) == 2
	assert all(
		'--dry-run' in text[match.end() : match.end() + 40]
		for match in schema_flags
	)


def test_hmm_k6_runbook_inline_python_blocks_compile() -> None:
	text = RUNBOOK.read_text(encoding='utf-8')
	blocks = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)

	assert len(blocks) == 7
	for index, source in enumerate(blocks, start=1):
		compile(source, f'{RUNBOOK}:inline-{index}', 'exec')
