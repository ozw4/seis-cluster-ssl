from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_HMM_K,
	FIVE_WAY_MIN_CONFIDENCE,
	FIVE_WAY_RANDOM_SEED,
	FIVE_WAY_STAGE2_EPOCHS,
	FIVE_WAY_STAGE2_GLOBAL_STEPS,
	FIVE_WAY_UNFREEZE_TOP_BLOCKS,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
)

EXP_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/124_lithology_random_init_hmm_k6_distill010_v1'
)
STAGE2_ROOT = EXP_ROOT / '10_stage2/random_init/hmm/k6'
EXTRACTION_CONFIG = EXP_ROOT / '20_embeddings/01_extract_random_init_hmm_k6.yaml'
CANDIDATE_CONFIG = EXP_ROOT / '30_downstream/01_candidate.yaml'
CANONICAL_STAGE2_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1/'
	'30_stage2/mae100/hmm/k6/02_full_25ep.yaml'
)
V2_RANDOM_EXTRACTION_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v2/'
	'50_embeddings/05_extract_random.yaml'
)
CANONICAL_FIVE_WAY_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
RUNS = {
	'feasibility': '01_gpu_feasibility_1step.yaml',
	'full': '02_full_25ep.yaml',
}
# The two YAMLs may differ only in the output root and these train sizing keys.
SMOKE_TRAIN_KEYS = ('samples_per_epoch', 'epochs', 'num_workers', 'max_steps')
RANDOM_CHECKPOINT = (
	'pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/random/random_init.pt'
)
MAE_STAGE1_CHECKPOINT = (
	'pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1/mae/'
	'full_100ep/latest.pt'
)
PSEUDO_TARGET_DIR = (
	'pseudo_targets/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/mae100'
)
CONTROL_NAMESPACE = 'random_init_hmm_k6_distill010_v1'
CANDIDATE_ID = 'random_init_hmm_k6_distill010'
MODEL_TAG = 'f3_random_init_hmm_k6_distill010_topblock1_v1'
DISTILLATION_WEIGHT = 0.1
CANONICAL_DISTILLATION_WEIGHT = 0.2
STAGE2_OUTPUT_PREFIX = (
	f'pretraining/f3/facies_benchmark_v1/{CONTROL_NAMESPACE}/stage2/random_init/hmm/k6'
)
EMBEDDINGS_OUTPUT_DIR = (
	f'embeddings/f3/facies_benchmark_v2/{CONTROL_NAMESPACE}/{CANDIDATE_ID}/overlap_x64'
)
BENCHMARK_NAMESPACE = f'f3_lithology_benchmark/{CONTROL_NAMESPACE}'
# Consumed by summarize_f3_lithology_candidate_five_way.py; the base-schema
# tests strip it so they also cover candidate configs without the key.
FIVE_WAY_SUMMARY_KEY = 'five_way_summary_root'
FIVE_WAY_SUMMARY_ROOT_NAME = 'summary_five_way'


def _prepare_live_inputs(root: Path) -> None:
	for checkpoint in (RANDOM_CHECKPOINT, MAE_STAGE1_CHECKPOINT):
		path = root / checkpoint
		path.parent.mkdir(parents=True, exist_ok=True)
		path.touch()
	(root / PSEUDO_TARGET_DIR).mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3_root'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	_prepare_live_inputs(root)
	return root


@pytest.fixture
def stage2_configs() -> dict[str, dict[str, object]]:
	return {
		run: resolve_strat_hmm_pretext_config(load_config(STAGE2_ROOT / filename))
		for run, filename in RUNS.items()
	}


@pytest.fixture
def canonical_stage2_config() -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(CANONICAL_STAGE2_CONFIG))


def _candidate_base_mapping() -> tuple[dict[str, object], object]:
	"""Return the candidate mapping restricted to today's exact schema."""
	mapping = load_config(CANDIDATE_CONFIG)
	outputs = dict(mapping['outputs'])
	five_way_summary_root = outputs.pop(FIVE_WAY_SUMMARY_KEY, None)
	return {**mapping, 'outputs': outputs}, five_way_summary_root


def test_teacher_and_student_share_the_five_way_random_checkpoint(
	stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	expected_checkpoint = str(artifact_root / RANDOM_CHECKPOINT)
	five_way = load_config(CANONICAL_FIVE_WAY_CONFIG)
	by_id = {model['model_id']: model for model in five_way['models']}
	assert by_id['random']['checkpoint'] == expected_checkpoint
	for config in stage2_configs.values():
		assert config['teacher']['checkpoint'] == expected_checkpoint
		assert config['student']['init_checkpoint'] == expected_checkpoint
		assert config['student']['unfreeze_top_blocks'] == FIVE_WAY_UNFREEZE_TOP_BLOCKS
		serialized = repr(
			{'teacher': config['teacher'], 'student': config['student']}
		).lower()
		for forbidden in ('ssl_hmm_continuation_v1', 'stage1', 'best.pt'):
			assert forbidden not in serialized


def test_pseudo_targets_are_the_canonical_mae100_k6_targets(
	stage2_configs: dict[str, dict[str, object]],
	canonical_stage2_config: dict[str, object],
	artifact_root: Path,
) -> None:
	for config in stage2_configs.values():
		pseudo_targets = config['pseudo_targets']
		assert pseudo_targets['input_dir'].endswith('ssl_hmm_continuation_v1/mae100')
		assert pseudo_targets['input_dir'] == str(artifact_root / PSEUDO_TARGET_DIR)
		assert pseudo_targets['k'] == FIVE_WAY_HMM_K == 6
		assert pseudo_targets['min_confidence'] == FIVE_WAY_MIN_CONFIDENCE == 0.0
		assert pseudo_targets == canonical_stage2_config['pseudo_targets']


def test_head_loss_and_identity_match_the_control_contract(
	stage2_configs: dict[str, dict[str, object]],
) -> None:
	for config in stage2_configs.values():
		assert config['head'] == {
			'num_prototypes': 6,
			'projection_dim': 128,
			'temperature': 0.1,
			'normalize': True,
		}
		loss = {
			key: config['loss'][key]
			for key in (
				'prototype_weight',
				'usage_weight',
				'entropy_floor',
				'distillation_weight',
			)
		}
		assert loss == {
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'entropy_floor': None,
			'distillation_weight': DISTILLATION_WEIGHT,
		}
		identity = config['identity']
		assert identity['model_tag'] == MODEL_TAG
		scientific = identity['scientific_identity']
		assert scientific['experiment_role'] == 'random_init_control'
		assert (
			scientific['distillation_weight'] == config['loss']['distillation_weight']
		)
		assert (
			scientific['student_init_checkpoint_role']
			== scientific['teacher_checkpoint_role']
			== 'five_way_random_encoder_seed42'
		)
		assert scientific['pseudo_target_source'] == (
			'ssl_hmm_continuation_v1/mae100/k6'
		)


def test_full_run_uses_the_fixed_five_way_stage2_budget(
	stage2_configs: dict[str, dict[str, object]],
) -> None:
	full = stage2_configs['full']['train']
	assert full['batch_size'] == 16
	assert full['samples_per_epoch'] == 10_000
	assert full['epochs'] == FIVE_WAY_STAGE2_EPOCHS == 25
	assert (
		full['epochs'] * full['samples_per_epoch'] // full['batch_size']
		== FIVE_WAY_STAGE2_GLOBAL_STEPS
	)
	assert full['lr'] == 1.0e-5
	assert full['encoder_lr'] == 1.0e-5
	assert full['weight_decay'] == 0.05
	assert full['amp'] is False
	assert full['seed'] == FIVE_WAY_RANDOM_SEED == 42
	assert full['grad_clip_norm'] == 1.0
	assert full['max_steps'] is None
	assert full['allow_overwrite_output'] is False


def test_smoke_run_is_a_single_step(
	stage2_configs: dict[str, dict[str, object]],
) -> None:
	feasibility = stage2_configs['feasibility']['train']
	assert feasibility['max_steps'] == 1
	assert feasibility['epochs'] == 1
	assert feasibility['samples_per_epoch'] == 16
	assert feasibility['batch_size'] == 16
	assert feasibility['num_workers'] == 0


def test_output_roots_are_separated_under_the_control_namespace(
	stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	expected_prefix = artifact_root / STAGE2_OUTPUT_PREFIX
	feasibility_root = Path(stage2_configs['feasibility']['paths']['output_root'])
	full_root = Path(stage2_configs['full']['paths']['output_root'])
	assert feasibility_root == expected_prefix / 'gpu_feasibility_1step'
	assert full_root == expected_prefix / 'full_25ep'
	assert feasibility_root != full_root
	for root in (feasibility_root, full_root):
		assert 'ssl_hmm_continuation_v1' not in str(root)
		assert 'mae_local_bt_five_way_v1' not in str(root)


def test_smoke_and_full_differ_only_in_output_root_and_train_sizing(
	stage2_configs: dict[str, dict[str, object]],
) -> None:
	comparable = {}
	for run, config in stage2_configs.items():
		trimmed = deepcopy(config)
		del trimmed['paths']['output_root']
		for key in SMOKE_TRAIN_KEYS:
			del trimmed['train'][key]
		comparable[run] = trimmed
	assert comparable['feasibility'] == comparable['full']


def test_control_matches_canonical_mae_hmm_k6_except_declared_fields(
	stage2_configs: dict[str, dict[str, object]],
	canonical_stage2_config: dict[str, object],
	artifact_root: Path,
) -> None:
	control = stage2_configs['full']
	canonical = canonical_stage2_config
	for section in ('manifests', 'data', 'zero_mask', 'model', 'head', 'train'):
		assert control[section] == canonical[section], section
	assert control['pseudo_targets'] == canonical['pseudo_targets']

	control_loss = dict(control['loss'])
	canonical_loss = dict(canonical['loss'])
	assert control_loss.pop('distillation_weight') == DISTILLATION_WEIGHT
	assert canonical_loss.pop('distillation_weight') == (CANONICAL_DISTILLATION_WEIGHT)
	assert control_loss == canonical_loss

	assert canonical['teacher']['checkpoint'] == str(
		artifact_root / MAE_STAGE1_CHECKPOINT
	)
	assert canonical['student']['init_checkpoint'] == canonical['teacher']['checkpoint']
	assert 'identity' not in canonical

	comparison = deepcopy(control)
	del comparison['identity']
	comparison['paths']['output_root'] = canonical['paths']['output_root']
	comparison['teacher']['checkpoint'] = canonical['teacher']['checkpoint']
	comparison['student']['init_checkpoint'] = canonical['student']['init_checkpoint']
	comparison['loss']['distillation_weight'] = canonical['loss']['distillation_weight']
	assert comparison == canonical


def test_canonical_mae_hmm_k6_config_is_unchanged() -> None:
	raw = load_config(CANONICAL_STAGE2_CONFIG)
	assert raw['teacher']['checkpoint'].endswith(
		'ssl_hmm_continuation_v1/stage1/mae/full_100ep/latest.pt'
	)
	assert raw['student']['init_checkpoint'] == raw['teacher']['checkpoint']
	assert raw['loss']['distillation_weight'] == CANONICAL_DISTILLATION_WEIGHT
	assert 'identity' not in raw
	assert CANDIDATE_ID not in repr(raw)


def test_extraction_matches_v2_random_extraction_except_source_and_output(
	stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	extract = resolve_embedding_extraction_config(load_config(EXTRACTION_CONFIG))
	baseline = resolve_embedding_extraction_config(
		load_config(V2_RANDOM_EXTRACTION_CONFIG)
	)
	full_root = stage2_configs['full']['paths']['output_root']

	assert extract['embeddings']['checkpoint'] == f'{full_root}/latest.pt'
	assert Path(extract['embeddings']['output_dir']) == (
		artifact_root / EMBEDDINGS_OUTPUT_DIR
	)
	assert '/facies_benchmark_v2/' in extract['manifests']['input']
	assert set(load_config(EXTRACTION_CONFIG)['embeddings']) == {
		'checkpoint',
		'output_dir',
	}

	assert baseline['embeddings']['checkpoint'] == str(
		artifact_root / RANDOM_CHECKPOINT
	)
	assert extract['embeddings']['checkpoint'] != baseline['embeddings']['checkpoint']
	assert extract['embeddings']['output_dir'] != baseline['embeddings']['output_dir']
	comparison = deepcopy(extract)
	comparison['embeddings'] = baseline['embeddings']
	assert comparison == baseline


def test_candidate_resolves_against_canonical_v3_without_collisions(
	stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	base_mapping, _five_way_summary_root = _candidate_base_mapping()
	candidate = f3_lithology_candidate_config_from_mapping(base_mapping)
	canonical = load_f3_lithology_candidate_canonical_config(candidate)
	extract = resolve_embedding_extraction_config(load_config(EXTRACTION_CONFIG))
	full_root = stage2_configs['full']['paths']['output_root']

	assert candidate.canonical_config == Path.cwd() / CANONICAL_FIVE_WAY_CONFIG
	assert candidate.candidate_id == CANDIDATE_ID
	assert candidate.candidate_id not in canonical.model_ids
	assert candidate.checkpoint == Path(f'{full_root}/latest.pt')
	assert candidate.embeddings_dir == Path(extract['embeddings']['output_dir'])
	assert candidate.embeddings_dir.parent.name == candidate.candidate_id
	assert candidate.runs_root == artifact_root / BENCHMARK_NAMESPACE / 'runs'
	assert candidate.summary_root == (artifact_root / BENCHMARK_NAMESPACE / 'summary')
	assert candidate.runs_root != candidate.summary_root
	for root in (candidate.runs_root, candidate.summary_root):
		assert not root.is_relative_to(canonical.runs_root)
		assert not root.is_relative_to(canonical.summary_root)


def test_candidate_five_way_summary_root_is_declared_beside_summary_root(
	artifact_root: Path,
) -> None:
	raw = load_config(CANDIDATE_CONFIG)
	outputs = raw['outputs']
	assert set(outputs) == {'runs_root', 'summary_root', FIVE_WAY_SUMMARY_KEY}
	five_way_summary_root = Path(outputs[FIVE_WAY_SUMMARY_KEY])
	assert five_way_summary_root.is_absolute()
	assert five_way_summary_root == (
		artifact_root / BENCHMARK_NAMESPACE / FIVE_WAY_SUMMARY_ROOT_NAME
	)
	candidate = f3_lithology_candidate_config_from_mapping(raw)
	assert candidate.five_way_summary_root == five_way_summary_root
	canonical = load_f3_lithology_candidate_canonical_config(candidate)
	assert not five_way_summary_root.is_relative_to(canonical.runs_root)
	assert not five_way_summary_root.is_relative_to(canonical.summary_root)
	distinct = {
		Path(outputs['runs_root']),
		Path(outputs['summary_root']),
		five_way_summary_root,
	}
	assert len(distinct) == 3
	for other in (Path(outputs['runs_root']), Path(outputs['summary_root'])):
		assert not five_way_summary_root.is_relative_to(other)
		assert not other.is_relative_to(five_way_summary_root)


def test_control_namespace_is_unused_by_other_experiments() -> None:
	tokens = (CONTROL_NAMESPACE, CANDIDATE_ID, MODEL_TAG)
	for path in sorted(Path('experiments').rglob('*.yaml')):
		if path.is_relative_to(EXP_ROOT):
			continue
		text = path.read_text(encoding='utf-8')
		for token in tokens:
			assert token not in text, (path, token)


def test_resolver_rejects_zero_distillation_with_an_unfrozen_top_block() -> None:
	raw = load_config(STAGE2_ROOT / RUNS['full'])
	assert raw['student']['unfreeze_top_blocks'] == 1
	broken = deepcopy(raw)
	broken['loss']['distillation_weight'] = 0.0
	with pytest.raises(ValueError, match='distillation_weight must be positive'):
		resolve_strat_hmm_pretext_config(broken)


def test_resolver_rejects_a_prototype_count_that_differs_from_k() -> None:
	broken = deepcopy(load_config(STAGE2_ROOT / RUNS['full']))
	broken['head']['num_prototypes'] = 8
	with pytest.raises(ValueError, match=r'pseudo_targets\.k must equal'):
		resolve_strat_hmm_pretext_config(broken)


def test_resolver_requires_the_random_checkpoint_to_exist(
	artifact_root: Path,
) -> None:
	(artifact_root / RANDOM_CHECKPOINT).unlink()
	with pytest.raises(FileNotFoundError, match=r'teacher\.checkpoint must exist'):
		resolve_strat_hmm_pretext_config(load_config(STAGE2_ROOT / RUNS['full']))


def test_candidate_rejects_a_canonical_model_id() -> None:
	base_mapping, _five_way_summary_root = _candidate_base_mapping()
	broken = deepcopy(base_mapping)
	broken['candidate']['id'] = 'random'
	candidate = f3_lithology_candidate_config_from_mapping(broken)
	with pytest.raises(ValueError, match='conflicts with canonical model ID'):
		load_f3_lithology_candidate_canonical_config(candidate)


def test_dry_run_creates_no_artifacts(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	_prepare_live_inputs(artifact_root)
	environment = {
		**os.environ,
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root),
	}
	before = {str(path) for path in artifact_root.rglob('*')}
	for filename in RUNS.values():
		result = subprocess.run(  # noqa: S603
			[
				sys.executable,
				'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
				'--config',
				str(STAGE2_ROOT / filename),
				'--dry-run',
			],
			check=True,
			capture_output=True,
			text=True,
			env=environment,
		)
		assert 'execution: dry-run; training skipped' in result.stdout
	after = {str(path) for path in artifact_root.rglob('*')}
	assert after == before
