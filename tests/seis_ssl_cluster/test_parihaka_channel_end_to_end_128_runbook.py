from __future__ import annotations

import re
from pathlib import Path

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'34_channel_end_to_end_128_v1'
)
README = EXPERIMENT_ROOT / 'README.md'
CONFIG = EXPERIMENT_ROOT / '01_channel_end_to_end_128.yaml'
LAYOUT_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'30_channel_benchmark_v1/02_layouts.yaml'
)
RUNNER = Path('proc/seis_ssl_cluster/run_parihaka_channel_end_to_end.py')
SUMMARY = Path(
	'proc/seis_ssl_cluster/summarize_parihaka_channel_end_to_end.py'
)
FOUR_WAY_SUMMARY = Path(
	'proc/seis_ssl_cluster/summarize_parihaka_channel_four_way.py'
)


def test_runbook_references_existing_config_layout_and_public_clis() -> None:
	text = README.read_text(encoding='utf-8')

	assert '01_channel_end_to_end_128.yaml' in text
	assert str(LAYOUT_CONFIG) in text
	for cli in (RUNNER, SUMMARY, FOUR_WAY_SUMMARY):
		assert str(cli) in text
		assert cli.is_file()
	assert CONFIG.is_file()
	assert LAYOUT_CONFIG.is_file()


def test_runbook_states_the_scientific_geometry_and_claim_boundary() -> None:
	intro = _section_before('## Environment')

	assert 'paired end-to-end comparison' in intro
	assert '`finetune_pretrained`' in intro
	assert '`train_from_scratch`' in intro
	assert 'encoder and decoder are both trainable' in intro
	assert 'raw `128³` amplitude crop' in intro
	assert 'central `64³` voxels' in intro
	assert '32 voxels' in intro
	assert '= 30 jobs`' in intro
	assert 'survey-specific, transductive evaluation' in intro
	assert 'offline overlap-aggregated full-volume embeddings' in intro
	assert 'encodes each raw supervised tile during training' in intro
	assert 'do not isolate encoder fine-tuning' in intro


def test_runbook_inline_python_is_valid_and_audits_live_contracts() -> None:
	text = README.read_text(encoding='utf-8')
	blocks = re.findall(r"python - <<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)

	assert len(blocks) == 3
	for index, source in enumerate(blocks, start=1):
		compile(source, f'{README}:inline-{index}', 'exec')

	source_audit = blocks[0]
	for helper in (
		'channel_end_to_end_config_from_mapping',
		'encoder_initial_state_sha256',
		'inspect_channel_end_to_end_job',
	):
		assert helper in source_audit
	for contract in (
		"('labels', config.labels)",
		"('label metadata', config.labels_metadata)",
		"('reference metadata', reference.metadata_path)",
		"('reference valid-token mask', reference.valid_tokens_path)",
		'pretrained.model_geometry != random.model_geometry',
		'pretrained_sha == random_sha',
		'reference.preprocessing',
		'reference.zero_mask',
		'raw_input_shape != (128, 128, 128)',
		'config.train != v1.train',
		'output roots overlap',
	):
		assert contract in source_audit

	paired_audit = blocks[1]
	assert "for index in range(5)" in paired_audit
	assert "for size in ('small', 'medium', 'large')" in paired_audit
	assert "for encoder_init in ('pretrained', 'random')" in paired_audit
	assert "identity.pop('encoder_source')" in paired_audit
	assert 'pretrained.tile_ids != random.tile_ids' in paired_audit

	feasibility_audit = blocks[2]
	for contract in (
		"for encoder_init in ('pretrained', 'random')",
		"payload.get('global_step') != 1",
		"core != (8, 8, 8) or halo != (4, 4, 4)",
		"raw_input_shape != (128, 128, 128)",
		"group_names != ['encoder', 'decoder']",
		'require_finite(optimizer_state',
		'len(decoder_initial_shas) != 1',
		'len(encoder_initial_shas) != 2',
	):
		assert contract in feasibility_audit


def test_dry_run_loop_covers_the_thirty_condition_matrix() -> None:
	section = _numbered_section(3)

	assert 'for encoder_init in pretrained random; do' in section
	assert (
		'for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do'
		in section
	)
	assert 'for size in small medium large; do' in section
	assert section.count(str(RUNNER)) == 1
	assert '--encoder-init "$encoder_init"' in section
	assert '--layout "$layout"' in section
	assert '--size "$size"' in section
	assert '--device cuda' in section
	assert '--dry-run' in section
	assert 2 * 5 * 3 == 30


def test_feasibility_runs_both_initializations_for_one_step() -> None:
	section = _numbered_section(4)

	assert 'for encoder_init in pretrained random; do' in section
	assert section.count(str(RUNNER)) == 1
	assert '--layout layout_000' in section
	assert '--size small' in section
	assert '--device cuda' in section
	assert '--max-steps 1' in section
	assert 'latest.pt' in section
	assert 'global_step == 1' in section
	assert 'CUDA OOM' in section
	assert 'The two feasibility jobs remain' in section


def test_restartable_loop_distinguishes_completed_resumed_and_fresh_jobs() -> None:
	section = _numbered_section(5)

	metrics_index = section.index('if [ -f "$JOB_DIR/metrics.json" ]; then')
	latest_index = section.index('elif [ -f "$JOB_DIR/latest.pt" ]; then')
	resume_index = section.index('--resume "$JOB_DIR/latest.pt"')
	fresh_index = section.index('else', resume_index)
	assert metrics_index < latest_index < resume_index < fresh_index
	assert section.count(str(RUNNER)) == 2
	assert 'echo "completed: $encoder_init/$layout/$size"' in section
	assert 'This distinguishes a completed job' in section


def test_runbook_uses_only_the_separate_128_artifact_root_safely() -> None:
	text = README.read_text(encoding='utf-8')

	assert (
		'export RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/'
		'channel_end_to_end_128_v1/runs"'
	) in text
	assert (
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/'
		'channel_end_to_end_128_v1/summary/'
	) in text
	assert (
		'export RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/'
		'channel_end_to_end/runs"'
	) not in text
	assert 'amp: true' not in text
	assert '--amp' not in text
	assert 'exit 1' not in text
	assert not re.search(
		r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync\s|mv\s)',
		text,
	)


def test_summary_dry_run_precedes_the_single_writing_invocation() -> None:
	section = _numbered_section(7)
	invocations = re.findall(
		r'python proc/seis_ssl_cluster/'
		r'summarize_parihaka_channel_end_to_end\.py\s+\\\n'
		r'\s+--config "\$CONFIG"'
		r'(?P<dry_run>\s+\\\n\s+--dry-run)?',
		section,
	)

	assert invocations == [' \\\n  --dry-run', '']
	assert section.index('--dry-run') < section.rindex(str(SUMMARY))
	assert 'complete_jobs: 30' in section
	for filename in ('comparison.csv', 'summary.json', 'summary.md'):
		assert filename in section
	assert '`finetune_pretrained - train_from_scratch`' in section


def _section_before(heading: str) -> str:
	return README.read_text(encoding='utf-8').split(heading, maxsplit=1)[0]


def _numbered_section(number: int) -> str:
	text = README.read_text(encoding='utf-8')
	start = f'## {number}.'
	section = text.split(start, maxsplit=1)[1]
	next_heading = f'## {number + 1}.'
	return section.split(next_heading, maxsplit=1)[0]
