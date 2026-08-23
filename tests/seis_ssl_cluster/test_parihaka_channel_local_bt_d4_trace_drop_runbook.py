from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'35_channel_local_bt_d4_trace_drop_v1'
)
TRAINING_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/30_stage2/local_bt100/'
	'bt_continue_d4_trace_drop'
)
README = EXPERIMENT_ROOT / 'README.md'
TRAINING_CONFIGS = (
	'01_gpu_feasibility_1step.yaml',
	'02_full_25ep.yaml',
)
EXPERIMENT_CONFIGS = (
	'01_extract_augmented_embeddings.yaml',
	'02_channel_comparison.yaml',
)
CLIS = (
	Path('proc/seis_ssl_cluster/train_amp_barlow_twins.py'),
	Path('proc/seis_ssl_cluster/extract_embeddings.py'),
	Path('proc/seis_ssl_cluster/run_parihaka_channel_decoder.py'),
)
CANDIDATE = 'local_barlow_twins_d4_trace_drop'
CONTROL = 'local_barlow_twins'
HMM = 'local_barlow_twins_hmm_k6'


def _text() -> str:
	return README.read_text(encoding='utf-8')


def _section(text: str, start: str, end: str | None = None) -> str:
	section = text.split(start, maxsplit=1)[1]
	return section if end is None else section.split(end, maxsplit=1)[0]


def _inline_python(text: str) -> list[str]:
	return re.findall(r"python - <<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)


def test_runbook_references_configs_public_clis_and_compilable_python() -> None:
	text = _text()
	for filename in (*TRAINING_CONFIGS, *EXPERIMENT_CONFIGS):
		assert filename in text
	assert 'tests/seis_ssl_cluster/test_barlow_twins_training_contract.py' in text
	for cli in CLIS:
		assert str(cli) in text

	blocks = _inline_python(text)
	assert len(blocks) == 5
	for index, source in enumerate(blocks, start=1):
		compile(source, f'{README}:inline-{index}', 'exec')


def test_runbook_referenced_paths_exist() -> None:
	assert README.is_file()
	for filename in TRAINING_CONFIGS:
		assert (TRAINING_ROOT / filename).is_file()
	for filename in EXPERIMENT_CONFIGS:
		assert (EXPERIMENT_ROOT / filename).is_file()
	for cli in CLIS:
		assert cli.is_file()
	assert Path(
		'experiments/parihaka/facies_benchmark_v1/'
		'30_channel_benchmark_v1/02_layouts.yaml'
	).is_file()


def test_runbook_states_scientific_identity_and_exact_augmentation() -> None:
	text = _text()
	intro = text.split('## Environment', maxsplit=1)[0]
	normalized_intro = ' '.join(intro.split())
	assert 'existing `localBT100` checkpoint' in intro
	assert 'D4 + trace-drop Local BT25' in intro
	assert 'flip-only BT25 control and HMM25' in intro
	assert 'survey-specific' in intro
	assert 'transductive' in intro
	assert CANDIDATE in intro
	assert all(
		item in intro
		for item in ('encoder', 'projector', '128 local pairs', 'loss', 'optimizer')
	)
	assert '15,625-step budget' in intro
	assert (
		'augmentations:\n'
		'  policy: xy_d4_trace_drop_v1\n'
		'  reflection_probability: 0.5\n'
		'  trace_drop_probability: 0.02'
	) in intro
	assert 'only `validation.channel_iou`' in intro
	assert 'Test IoU is not used by the gate' in intro
	assert (
		'stop without running the other sizes or exploring more augmentations'
		in normalized_intro
	)


def test_runbook_uses_requested_environment_and_separate_output_root() -> None:
	text = _text()
	for assignment in (
		'cd /workspace',
		'export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster',
		(
			'export SUITE=experiments/parihaka/facies_benchmark_v1/'
			'21_ssl_hmm_continuation_v1'
		),
		(
			'export EXP=experiments/parihaka/facies_benchmark_v1/'
			'35_channel_local_bt_d4_trace_drop_v1'
		),
		(
			'export AUG_BT_CONFIG_ROOT="$SUITE/30_stage2/local_bt100/'
			'bt_continue_d4_trace_drop"'
		),
		'export CHANNEL_CONFIG="$EXP/02_channel_comparison.yaml"',
		(
			'export RUNS_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/'
			'ssl_hmm_four_way_v1/runs"'
		),
		(
			'export REPORT_ROOT="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/channel_benchmark/'
			'local_bt_d4_trace_drop_v1/summary"'
		),
		'export CUDA_VISIBLE_DEVICES=1',
	):
		assert assignment in text
	assert 'channel_benchmark/mae_local_bt_four_way_v1/summary' not in text


def test_runbook_runs_only_candidate_for_medium_screening() -> None:
	text = _text()
	preflight = _section(text, '## 5.', '## 6.')
	medium = _section(text, '## 6.', '## 7.')
	for section in (preflight, medium):
		assert section.count('--model local_barlow_twins_d4_trace_drop') == 1
		assert section.count('--size medium') == 1
		assert tuple(re.findall(r'^  layout_\d{3}', section, flags=re.MULTILINE)) == (
			'  layout_000',
			'  layout_001',
			'  layout_002',
			'  layout_003',
			'  layout_004',
		)
		models = re.findall(r'--model ([^\s\\]+)', section)
		assert models == [CANDIDATE]
	assert '--dry-run' in preflight
	assert '--dry-run' not in medium


def test_screening_script_uses_validation_only_and_exact_gate() -> None:
	text = _text()
	screening = _section(text, '## 7.', '## 8.')
	blocks = _inline_python(screening)
	assert len(blocks) == 1
	script = blocks[0]
	assert "payload.get('validation')" in script
	assert "validation.get('channel_iou')" in script
	assert "'metric': 'validation.channel_iou'" in script
	assert '["test"]' not in script
	assert "['test']" not in script
	assert '.get("test")' not in script
	assert ".get('test')" not in script
	assert 'test.channel_iou' not in script
	assert 'paired_mean >= threshold and losses <= max_losses' in script
	assert 'threshold = 0.01' in script
	assert 'max_losses = 1' in script
	assert "'screening_validation.json'" in script
	assert "'screening_validation.md'" in script
	for key in (
		'model_control',
		'model_candidate',
		'metric',
		'data_size',
		'layout_gains',
		'paired_mean',
		'paired_median',
		'sample_standard_deviation',
		'wins',
		'ties',
		'losses',
		'gate_threshold',
		'gate_max_losses',
		'gate_passed',
	):
		assert f"'{key}'" in script


def test_small_large_jobs_are_only_after_manual_gate_confirmation() -> None:
	text = _text()
	before_gate = text.split('## 8.', maxsplit=1)[0]
	after_gate = _section(text, '## 8.', '## 9.')
	assert 'for size in small large' not in before_gate
	assert 'for size in small large' in after_gate
	assert 'human confirms that the gate passed' in text
	assert '`gate_passed` is `true`' in after_gate
	assert 'automatically' not in after_gate
	assert re.findall(r'--model ([^\s\\]+)', after_gate) == [CANDIDATE]


def test_full_report_audits_exact_model_set_and_outputs() -> None:
	text = _text()
	full = _section(text, '## 9.')
	blocks = _inline_python(full)
	assert len(blocks) == 1
	script = blocks[0]
	model_ids_match = re.search(
		r'model_ids = \(\n(?P<body>.*?)\n\)', script, flags=re.DOTALL
	)
	assert model_ids_match is not None
	assert Counter(re.findall(r"'([^']+)'", model_ids_match.group('body'))) == Counter(
		(CONTROL, CANDIDATE, HMM)
	)
	assert 'inspect_channel_model_results(config, model_ids=model_ids)' in script
	assert 'len(jobs) != 45' in script
	for comparison in ('augmentation_gain', 'hmm_gain', 'hmm_minus_augmented'):
		assert comparison in script
	for statistic in (
		'paired_mean',
		'paired_median',
		'sample_standard_deviation',
		'wins',
		'ties',
		'losses',
		'layout_deltas',
	):
		assert statistic in script
	for filename in ('comparison.csv', 'summary.json', 'summary.md'):
		assert filename in script


def test_runbook_reuses_existing_jobs_and_avoids_unsafe_commands() -> None:
	text = _text()
	assert 'Do not rerun the existing flip-only or HMM jobs' in text
	assert 'flip-only BT25 control and HMM25' in text
	assert 'full_25ep/latest.pt' in text
	assert 'full_25ep/best.pt' not in text
	assert '01_extract_augmented_embeddings.yaml' in text
	assert not re.search(r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync\s)', text)
	assert 'exit 1' not in text
	assert not re.search(r'(?m)^\s*(?:cp|rsync)\b.*artifacts', text)
