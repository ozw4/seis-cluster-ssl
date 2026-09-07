from __future__ import annotations

import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_clustering_config,
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
from seis_ssl_cluster.config.schema import EXPECTED_MODEL_NAME
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
)
from seis_ssl_cluster.stratigraphy import (
	discover_pseudo_target_inputs,
	pseudo_target_paths,
)

EXP_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/'
	'125_lithology_random_init_self_target_hmm_k6_distill010_v1'
)
TARGET_ROOT = EXP_ROOT / '10_hmm_targets/random_init'
TARGET_EXTRACTION_CONFIG = TARGET_ROOT / '01_extract_embeddings.yaml'
CLUSTERING_CONFIG = TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml'
EXPORT_SCRIPT = TARGET_ROOT / 'k6/03_export_pseudo_targets.sh'
STAGE2_ROOT = EXP_ROOT / '20_stage2/random_init/hmm/k6'
EXTRACTION_CONFIG = (
	EXP_ROOT / '30_embeddings/01_extract_random_init_self_target_hmm_k6_distill010.yaml'
)
CANDIDATE_CONFIG = EXP_ROOT / '40_downstream/01_candidate.yaml'
RECIPE_COUNT = 7
CANONICAL_TARGET_ROOT = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1/'
	'20_hmm_targets/mae100'
)
CANONICAL_TARGET_EXTRACTION_CONFIG = (
	CANONICAL_TARGET_ROOT / '01_extract_embeddings.yaml'
)
CANONICAL_CLUSTERING_CONFIG = CANONICAL_TARGET_ROOT / 'k6/02_cluster_hmm_k6.yaml'
CANONICAL_EXPORT_SCRIPT = CANONICAL_TARGET_ROOT / 'k6/03_export_pseudo_targets.sh'
CANONICAL_STAGE2_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1/'
	'30_stage2/mae100/hmm/k6/02_full_25ep.yaml'
)
PREVIOUS_CONTROL_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/124_lithology_random_init_hmm_k6_distill010_v1'
)
PREVIOUS_CONTROL_STAGE2_ROOT = PREVIOUS_CONTROL_ROOT / '10_stage2/random_init/hmm/k6'
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
# The two Stage 2 YAMLs may differ only in the output root and these sizing keys.
SMOKE_TRAIN_KEYS = ('samples_per_epoch', 'epochs', 'num_workers', 'max_steps')
RANDOM_CHECKPOINT = (
	'pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/random/random_init.pt'
)
# Inputs of the canonical mae_hmm_k6 arm and of the experiment-124 control. They
# are created in the temporary root only so those reference recipes resolve for
# the comparisons below; the experiment-125 recipes must never point at them.
MAE_STAGE1_CHECKPOINT = (
	'pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage1/mae/'
	'full_100ep/latest.pt'
)
MAE100_PSEUDO_TARGET_DIR = (
	'pseudo_targets/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/mae100'
)
CONTROL_NAMESPACE = 'random_init_self_target_hmm_k6_distill010_v1'
CANDIDATE_ID = 'random_init_self_target_hmm_k6_distill010'
MODEL_TAG = 'f3_random_init_self_target_hmm_k6_distill010_topblock1_v1'
EXPERIMENT_ROLE = 'random_init_self_target_control'
CHECKPOINT_ROLE = 'five_way_random_encoder_seed42'
PSEUDO_TARGET_SOURCE = f'{CONTROL_NAMESPACE}/random_init/k6'
COMPARISON_SUITE = 'f3_lithology_mae_local_bt_five_way_v3'
PREVIOUS_CONTROL_NAMESPACE = 'random_init_hmm_k6_distill010_v1'
PREVIOUS_CONTROL_CANDIDATE_ID = 'random_init_hmm_k6_distill010'
PREVIOUS_CONTROL_MODEL_TAG = 'f3_random_init_hmm_k6_distill010_topblock1_v1'
PREVIOUS_CONTROL_EXPERIMENT_ROLE = 'random_init_control'
PREVIOUS_CONTROL_PSEUDO_TARGET_SOURCE = 'ssl_hmm_continuation_v1/mae100/k6'
DISTILLATION_WEIGHT = 0.1
CANONICAL_DISTILLATION_WEIGHT = 0.2
TARGET_EMBEDDINGS_DIR = (
	f'embeddings/f3/facies_benchmark_v1/{CONTROL_NAMESPACE}/'
	'hmm_targets/random_init/overlap_x64'
)
CLUSTERING_OUTPUT_DIR = (
	f'clustering/f3/facies_benchmark_v1/{CONTROL_NAMESPACE}/hmm_targets/random_init/k6'
)
PSEUDO_TARGET_ROOT = (
	f'pseudo_targets/f3/facies_benchmark_v1/{CONTROL_NAMESPACE}/random_init'
)
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
EXPORT_FLAGS = {
	'--k': '6',
	'--confidence': '1.0',
	'--boundary-alpha': '0.0',
	'--boundary-tau': '1.0',
	'--schema-version': '2',
}
EXPORT_DIRECTORY_FLAGS = ('--clustering-output-dir', '--pseudo-target-root')
EXPORT_CLI = 'proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py'
FAKE_SURVEY_ID = 'survey'
# Lineage tokens that must not appear anywhere in a resolved experiment-125 config.
FORBIDDEN_LINEAGE_TOKENS = (
	'ssl_hmm_continuation_v1',
	'/stage1/',
	'best.pt',
	PREVIOUS_CONTROL_NAMESPACE,
)
# Text anchors: the only recipe lines allowed to contain 'mae' are the random
# checkpoint, the canonical v3 five-way config, and the comparison-suite label.
ALLOWED_MAE_TEXT_ANCHORS = (
	'mae_local_bt_five_way_v1/random/random_init.pt',
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml',
	COMPARISON_SUITE,
)
STAGE2_DIFFERENCES_FROM_PREVIOUS_CONTROL = (
	('paths', 'output_root'),
	('identity', 'model_tag'),
	('identity', 'scientific_identity', 'experiment_role'),
	('identity', 'scientific_identity', 'pseudo_target_source'),
	('pseudo_targets', 'input_dir'),
)


def _prepare_live_inputs(root: Path) -> None:
	for checkpoint in (RANDOM_CHECKPOINT, MAE_STAGE1_CHECKPOINT):
		path = root / checkpoint
		path.parent.mkdir(parents=True, exist_ok=True)
		path.touch()
	for directory in (PSEUDO_TARGET_ROOT, MAE100_PSEUDO_TARGET_DIR):
		(root / directory).mkdir(parents=True, exist_ok=True)


def _prepare_fake_cluster_labels(root: Path) -> Path:
	"""Write one tiny K=6 label volume so the export CLI can dry-run offline."""
	column = np.repeat(np.arange(6, dtype=np.int32), 2)
	labels = np.stack([column, column])[None]
	labels[0, 1, 0] = -1
	label_dir = root / CLUSTERING_OUTPUT_DIR / 'labels' / 'k6'
	label_dir.mkdir(parents=True, exist_ok=True)
	path = label_dir / f'{FAKE_SURVEY_ID}.cluster_labels_token.npy'
	np.save(path, labels)
	return path


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3_root'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	_prepare_live_inputs(root)
	return root


@pytest.fixture
def target_extraction_config() -> dict[str, object]:
	return resolve_embedding_extraction_config(load_config(TARGET_EXTRACTION_CONFIG))


@pytest.fixture
def canonical_target_extraction_config() -> dict[str, object]:
	return resolve_embedding_extraction_config(
		load_config(CANONICAL_TARGET_EXTRACTION_CONFIG)
	)


@pytest.fixture
def clustering_config() -> dict[str, object]:
	return resolve_clustering_config(load_config(CLUSTERING_CONFIG))


@pytest.fixture
def canonical_clustering_config() -> dict[str, object]:
	return resolve_clustering_config(load_config(CANONICAL_CLUSTERING_CONFIG))


@pytest.fixture
def stage2_configs() -> dict[str, dict[str, object]]:
	return {
		run: resolve_strat_hmm_pretext_config(load_config(STAGE2_ROOT / filename))
		for run, filename in RUNS.items()
	}


@pytest.fixture
def previous_control_stage2_configs() -> dict[str, dict[str, object]]:
	return {
		run: resolve_strat_hmm_pretext_config(
			load_config(PREVIOUS_CONTROL_STAGE2_ROOT / filename)
		)
		for run, filename in RUNS.items()
	}


@pytest.fixture
def canonical_stage2_config() -> dict[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(CANONICAL_STAGE2_CONFIG))


@pytest.fixture
def extraction_config() -> dict[str, object]:
	return resolve_embedding_extraction_config(load_config(EXTRACTION_CONFIG))


def _recipe_files() -> list[Path]:
	return sorted([*EXP_ROOT.rglob('*.yaml'), *EXP_ROOT.rglob('*.sh')])


def _candidate_base_mapping() -> tuple[dict[str, object], object]:
	"""Return the candidate mapping restricted to today's exact schema."""
	mapping = load_config(CANDIDATE_CONFIG)
	outputs = dict(mapping['outputs'])
	five_way_summary_root = outputs.pop(FIVE_WAY_SUMMARY_KEY, None)
	return {**mapping, 'outputs': outputs}, five_way_summary_root


def _nested_get(config: Mapping[str, object], path: tuple[str, ...]) -> object:
	value: object = config
	for key in path:
		assert isinstance(value, Mapping), path
		value = value[key]
	return value


def _nested_set(
	config: dict[str, object],
	path: tuple[str, ...],
	value: object,
) -> None:
	parent = _nested_get(config, path[:-1])
	assert isinstance(parent, dict), path
	parent[path[-1]] = value


def _string_leaves(value: object, prefix: str = '') -> list[tuple[str, str]]:
	"""Return every string leaf of a nested config with its dotted key path."""
	if isinstance(value, str):
		return [(prefix, value)]
	if isinstance(value, Mapping):
		return [
			leaf
			for key, child in value.items()
			for leaf in _string_leaves(child, f'{prefix}.{key}' if prefix else str(key))
		]
	if isinstance(value, list | tuple):
		return [
			leaf
			for index, child in enumerate(value)
			for leaf in _string_leaves(child, f'{prefix}[{index}]')
		]
	return []


def _significant_lines(path: Path) -> list[str]:
	"""Return the lines that carry keys or values (comments and blanks dropped)."""
	return [
		line
		for line in path.read_text(encoding='utf-8').splitlines()
		if line.strip() and not line.strip().startswith('#')
	]


def _differing_keys(control: Path, reference: Path) -> list[str]:
	"""Return the YAML keys whose lines differ between two line-aligned recipes."""
	control_lines = _significant_lines(control)
	reference_lines = _significant_lines(reference)
	assert len(control_lines) == len(reference_lines), (control, reference)
	return [
		line.split(':', 1)[0].strip()
		for line, other in zip(control_lines, reference_lines, strict=True)
		if line != other
	]


def _export_script_arguments(path: Path) -> tuple[dict[str, str], bool]:
	"""Parse one export script into its flag map and its "$@" passthrough flag."""
	text = path.read_text(encoding='utf-8')
	assert text.startswith('#!/usr/bin/env bash\n')
	assert 'set -euo pipefail' in text
	assert path.stat().st_mode & stat.S_IXUSR
	command_lines = [
		line.strip().removesuffix('\\').strip()
		for line in text.splitlines()
		if line.strip()
		and not line.startswith('#!')
		and line.strip() != 'set -euo pipefail'
	]
	tokens = shlex.split(os.path.expandvars(' '.join(command_lines)))
	assert tokens[:2] == ['python', EXPORT_CLI]
	flag_tokens = tokens[2:]
	passthrough = flag_tokens[-1:] == ['$@']
	if passthrough:
		flag_tokens = flag_tokens[:-1]
	assert len(flag_tokens) % 2 == 0
	return dict(zip(flag_tokens[::2], flag_tokens[1::2], strict=True)), passthrough


def test_target_extraction_matches_canonical_mae100_except_source_and_output(
	target_extraction_config: dict[str, object],
	canonical_target_extraction_config: dict[str, object],
	artifact_root: Path,
) -> None:
	control = target_extraction_config
	canonical = canonical_target_extraction_config
	random_checkpoint = str(artifact_root / RANDOM_CHECKPOINT)
	five_way = load_config(CANONICAL_FIVE_WAY_CONFIG)
	by_id = {model['model_id']: model for model in five_way['models']}

	assert control['stage'] == canonical['stage'] == 'extract_embeddings'
	assert control['embeddings']['checkpoint'] == random_checkpoint
	assert by_id['random']['checkpoint'] == random_checkpoint
	assert Path(control['embeddings']['output_dir']) == (
		artifact_root / TARGET_EMBEDDINGS_DIR
	)
	assert Path(control['embeddings']['output_dir']).parent.name == 'random_init'
	assert 'k6' not in Path(control['embeddings']['output_dir']).parts
	assert '/facies_benchmark_v1/' in control['manifests']['input']
	assert set(load_config(TARGET_EXTRACTION_CONFIG)['embeddings']) == {
		'checkpoint',
		'output_dir',
	}

	assert canonical['embeddings']['checkpoint'] == str(
		artifact_root / MAE_STAGE1_CHECKPOINT
	)
	assert control['embeddings']['checkpoint'] != canonical['embeddings']['checkpoint']
	assert control['embeddings']['output_dir'] != canonical['embeddings']['output_dir']
	comparison = deepcopy(control)
	comparison['embeddings'] = canonical['embeddings']
	assert comparison == canonical
	assert _differing_keys(
		TARGET_EXTRACTION_CONFIG, CANONICAL_TARGET_EXTRACTION_CONFIG
	) == ['checkpoint', 'output_dir']


def test_clustering_matches_canonical_mae100_k6_except_directories(
	clustering_config: dict[str, object],
	canonical_clustering_config: dict[str, object],
	target_extraction_config: dict[str, object],
	artifact_root: Path,
) -> None:
	control = clustering_config
	canonical = canonical_clustering_config

	assert control['stage'] == canonical['stage'] == 'cluster_embeddings'
	assert (
		control['embeddings']['input_dir']
		== (target_extraction_config['embeddings']['output_dir'])
	)
	assert Path(control['clustering']['output_dir']) == (
		artifact_root / CLUSTERING_OUTPUT_DIR
	)
	assert control['clustering']['method'] == 'stratigraphic_hmm_kmeans'
	assert control['clustering']['k_values'] == [FIVE_WAY_HMM_K] == [6]
	assert control['clustering']['seed'] == FIVE_WAY_RANDOM_SEED == 42
	assert control['clustering']['stratigraphic_hmm']['iterations'] == 10

	assert control['embeddings']['input_dir'] != canonical['embeddings']['input_dir']
	assert control['clustering']['output_dir'] != canonical['clustering']['output_dir']
	comparison = deepcopy(control)
	comparison['embeddings'] = canonical['embeddings']
	comparison['clustering']['output_dir'] = canonical['clustering']['output_dir']
	assert comparison == canonical
	assert _differing_keys(CLUSTERING_CONFIG, CANONICAL_CLUSTERING_CONFIG) == [
		'input_dir',
		'output_dir',
	]


def test_export_script_matches_canonical_mae100_except_directories(
	clustering_config: dict[str, object],
	stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	control, passthrough = _export_script_arguments(EXPORT_SCRIPT)
	canonical, canonical_passthrough = _export_script_arguments(CANONICAL_EXPORT_SCRIPT)
	assert passthrough is True
	assert canonical_passthrough is False
	assert EXPORT_SCRIPT.read_text(encoding='utf-8').rstrip().endswith('"$@"')

	assert set(control) == set(canonical) == {*EXPORT_FLAGS, *EXPORT_DIRECTORY_FLAGS}
	for flag, value in EXPORT_FLAGS.items():
		assert control[flag] == canonical[flag] == value, flag
	for flag in EXPORT_DIRECTORY_FLAGS:
		assert control[flag] != canonical[flag], flag
	assert (
		control['--clustering-output-dir']
		== (clustering_config['clustering']['output_dir'])
	)
	export_root = Path(control['--pseudo-target-root'])
	assert export_root == artifact_root / PSEUDO_TARGET_ROOT
	assert export_root.name == 'random_init'
	assert 'k6' not in export_root.parts
	assert Path(canonical['--pseudo-target-root']) == (
		artifact_root / MAE100_PSEUDO_TARGET_DIR
	)

	exported = pseudo_target_paths(export_root, k=6, survey_id=FAKE_SURVEY_ID)
	assert exported.labels.parent == export_root / 'k6'
	for config in stage2_configs.values():
		pseudo_targets = config['pseudo_targets']
		assert Path(pseudo_targets['input_dir']) == export_root
		trainer_paths = pseudo_target_paths(
			pseudo_targets['input_dir'],
			k=pseudo_targets['k'],
			survey_id=FAKE_SURVEY_ID,
		)
		assert trainer_paths == exported
		with pytest.raises(FileNotFoundError, match=re.escape(str(export_root / 'k6'))):
			discover_pseudo_target_inputs(
				pseudo_targets['input_dir'], k=pseudo_targets['k']
			)


def test_teacher_and_student_share_the_five_way_random_checkpoint(
	stage2_configs: dict[str, dict[str, object]],
	target_extraction_config: dict[str, object],
	artifact_root: Path,
) -> None:
	expected_checkpoint = str(artifact_root / RANDOM_CHECKPOINT)
	five_way = load_config(CANONICAL_FIVE_WAY_CONFIG)
	by_id = {model['model_id']: model for model in five_way['models']}
	assert by_id['random']['checkpoint'] == expected_checkpoint
	assert target_extraction_config['embeddings']['checkpoint'] == expected_checkpoint
	for config in stage2_configs.values():
		assert config['teacher']['checkpoint'] == expected_checkpoint
		assert config['student']['init_checkpoint'] == expected_checkpoint
		assert config['student']['unfreeze_top_blocks'] == FIVE_WAY_UNFREEZE_TOP_BLOCKS
		serialized = repr(
			{'teacher': config['teacher'], 'student': config['student']}
		).lower()
		for forbidden in ('ssl_hmm_continuation_v1', 'stage1', 'best.pt'):
			assert forbidden not in serialized


def test_pseudo_targets_are_the_self_derived_k6_targets(
	stage2_configs: dict[str, dict[str, object]],
	canonical_stage2_config: dict[str, object],
	artifact_root: Path,
) -> None:
	canonical_pseudo_targets = dict(canonical_stage2_config['pseudo_targets'])
	assert canonical_pseudo_targets['input_dir'] == str(
		artifact_root / MAE100_PSEUDO_TARGET_DIR
	)
	for config in stage2_configs.values():
		pseudo_targets = dict(config['pseudo_targets'])
		assert pseudo_targets['input_dir'] == str(artifact_root / PSEUDO_TARGET_ROOT)
		assert not pseudo_targets['input_dir'].endswith(
			'ssl_hmm_continuation_v1/mae100'
		)
		assert pseudo_targets['input_dir'] != canonical_pseudo_targets['input_dir']
		assert pseudo_targets['k'] == FIVE_WAY_HMM_K == 6
		assert pseudo_targets['min_confidence'] == FIVE_WAY_MIN_CONFIDENCE == 0.0
		assert pseudo_targets['k'] == config['head']['num_prototypes']
		del pseudo_targets['input_dir']
		del canonical_pseudo_targets['input_dir']
		assert pseudo_targets == canonical_pseudo_targets
		canonical_pseudo_targets = dict(canonical_stage2_config['pseudo_targets'])


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
		assert set(identity) == {'model_tag', 'scientific_identity'}
		assert identity['model_tag'] == MODEL_TAG
		assert identity['scientific_identity'] == {
			'experiment_role': EXPERIMENT_ROLE,
			'student_init_checkpoint_role': CHECKPOINT_ROLE,
			'teacher_checkpoint_role': CHECKPOINT_ROLE,
			'pseudo_target_source': PSEUDO_TARGET_SOURCE,
			'distillation_weight': config['loss']['distillation_weight'],
			'comparison_suite': COMPARISON_SUITE,
		}


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
		== 15_625
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
		assert CONTROL_NAMESPACE in root.parts
		assert PREVIOUS_CONTROL_NAMESPACE not in root.parts
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


def test_stage2_matches_previous_random_init_control_except_declared_fields(
	stage2_configs: dict[str, dict[str, object]],
	previous_control_stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	expected_values = {
		('identity', 'model_tag'): (MODEL_TAG, PREVIOUS_CONTROL_MODEL_TAG),
		('identity', 'scientific_identity', 'experiment_role'): (
			EXPERIMENT_ROLE,
			PREVIOUS_CONTROL_EXPERIMENT_ROLE,
		),
		('identity', 'scientific_identity', 'pseudo_target_source'): (
			PSEUDO_TARGET_SOURCE,
			PREVIOUS_CONTROL_PSEUDO_TARGET_SOURCE,
		),
		('pseudo_targets', 'input_dir'): (
			str(artifact_root / PSEUDO_TARGET_ROOT),
			str(artifact_root / MAE100_PSEUDO_TARGET_DIR),
		),
	}
	for run, filename in RUNS.items():
		control = stage2_configs[run]
		previous = previous_control_stage2_configs[run]
		for path, (control_value, previous_value) in expected_values.items():
			assert _nested_get(control, path) == control_value, (run, path)
			assert _nested_get(previous, path) == previous_value, (run, path)
		control_root = Path(_nested_get(control, ('paths', 'output_root')))
		previous_root = Path(_nested_get(previous, ('paths', 'output_root')))
		assert control_root.name == previous_root.name
		assert control_root != previous_root
		assert PREVIOUS_CONTROL_NAMESPACE in previous_root.parts

		comparison = deepcopy(control)
		for path in STAGE2_DIFFERENCES_FROM_PREVIOUS_CONTROL:
			_nested_set(comparison, path, _nested_get(previous, path))
		assert comparison == previous
		assert _differing_keys(
			STAGE2_ROOT / filename, PREVIOUS_CONTROL_STAGE2_ROOT / filename
		) == [path[-1] for path in STAGE2_DIFFERENCES_FROM_PREVIOUS_CONTROL]


def test_control_matches_canonical_mae_hmm_k6_except_declared_fields(
	stage2_configs: dict[str, dict[str, object]],
	canonical_stage2_config: dict[str, object],
	artifact_root: Path,
) -> None:
	control = stage2_configs['full']
	canonical = canonical_stage2_config
	for section in ('manifests', 'data', 'zero_mask', 'model', 'head', 'train'):
		assert control[section] == canonical[section], section

	control_loss = dict(control['loss'])
	canonical_loss = dict(canonical['loss'])
	assert control_loss.pop('distillation_weight') == DISTILLATION_WEIGHT
	assert canonical_loss.pop('distillation_weight') == CANONICAL_DISTILLATION_WEIGHT
	assert control_loss == canonical_loss

	assert canonical['teacher']['checkpoint'] == str(
		artifact_root / MAE_STAGE1_CHECKPOINT
	)
	assert canonical['student']['init_checkpoint'] == canonical['teacher']['checkpoint']
	assert canonical['pseudo_targets']['input_dir'] == str(
		artifact_root / MAE100_PSEUDO_TARGET_DIR
	)
	assert 'identity' not in canonical

	comparison = deepcopy(control)
	del comparison['identity']
	comparison['paths']['output_root'] = canonical['paths']['output_root']
	comparison['teacher']['checkpoint'] = canonical['teacher']['checkpoint']
	comparison['student']['init_checkpoint'] = canonical['student']['init_checkpoint']
	comparison['loss']['distillation_weight'] = canonical['loss']['distillation_weight']
	comparison['pseudo_targets']['input_dir'] = canonical['pseudo_targets']['input_dir']
	assert comparison == canonical


def test_canonical_and_previous_control_recipes_are_unchanged() -> None:
	raw = load_config(CANONICAL_STAGE2_CONFIG)
	assert raw['teacher']['checkpoint'].endswith(
		'ssl_hmm_continuation_v1/stage1/mae/full_100ep/latest.pt'
	)
	assert raw['student']['init_checkpoint'] == raw['teacher']['checkpoint']
	assert raw['pseudo_targets']['input_dir'].endswith('ssl_hmm_continuation_v1/mae100')
	assert raw['loss']['distillation_weight'] == CANONICAL_DISTILLATION_WEIGHT
	assert 'identity' not in raw
	target = load_config(CANONICAL_TARGET_EXTRACTION_CONFIG)
	assert target['embeddings']['checkpoint'] == raw['teacher']['checkpoint']
	assert target['embeddings']['output_dir'].endswith(
		'ssl_hmm_continuation_v1/hmm_targets/mae100/overlap_x64'
	)
	clustering = load_config(CANONICAL_CLUSTERING_CONFIG)
	assert clustering['embeddings']['input_dir'] == target['embeddings']['output_dir']
	assert clustering['clustering']['output_dir'].endswith(
		'ssl_hmm_continuation_v1/hmm_targets/mae100/k6'
	)
	previous = load_config(PREVIOUS_CONTROL_STAGE2_ROOT / RUNS['full'])
	assert previous['identity']['model_tag'] == PREVIOUS_CONTROL_MODEL_TAG
	assert previous['pseudo_targets']['input_dir'] == raw['pseudo_targets']['input_dir']
	for path in (
		CANONICAL_STAGE2_CONFIG,
		CANONICAL_TARGET_EXTRACTION_CONFIG,
		CANONICAL_CLUSTERING_CONFIG,
		CANONICAL_EXPORT_SCRIPT,
		*sorted(PREVIOUS_CONTROL_ROOT.rglob('*.yaml')),
	):
		text = path.read_text(encoding='utf-8')
		assert CANDIDATE_ID not in text, path
		assert CONTROL_NAMESPACE not in text, path


def test_pipeline_resolves_without_any_mae_lineage(
	target_extraction_config: dict[str, object],
	clustering_config: dict[str, object],
	stage2_configs: dict[str, dict[str, object]],
	extraction_config: dict[str, object],
	artifact_root: Path,
) -> None:
	candidate_raw = load_config(CANDIDATE_CONFIG)
	configs: dict[str, Mapping[str, object]] = {
		'target_extraction': target_extraction_config,
		'clustering': clustering_config,
		'stage2_feasibility': stage2_configs['feasibility'],
		'stage2_full': stage2_configs['full'],
		'extraction': extraction_config,
		'candidate': candidate_raw,
	}
	random_checkpoint = str(artifact_root / RANDOM_CHECKPOINT)
	allowed_mae_values = {
		random_checkpoint,
		str(Path.cwd() / CANONICAL_FIVE_WAY_CONFIG),
		COMPARISON_SUITE,
	}
	# The resolver merges the fixed model contract, whose architecture name is
	# shared by every five-way arm; it names the encoder family, not a checkpoint.
	assert EXPECTED_MODEL_NAME == 'amp_mae3d'
	allowed_mae_leaves = {('model.name', EXPECTED_MODEL_NAME)}
	for label, config in configs.items():
		leaves = _string_leaves(config)
		assert leaves, label
		for key, text in leaves:
			lowered = text.lower()
			for forbidden in FORBIDDEN_LINEAGE_TOKENS:
				assert forbidden not in lowered, (label, key, text)
			if 'mae' in lowered and (key, text) not in allowed_mae_leaves:
				assert text in allowed_mae_values, (label, key, text)
	for config in stage2_configs.values():
		assert config['model']['name'] == EXPECTED_MODEL_NAME

	random_inputs = {
		'target_extraction.embeddings.checkpoint': (
			target_extraction_config['embeddings']['checkpoint']
		),
		'stage2_feasibility.teacher.checkpoint': (
			stage2_configs['feasibility']['teacher']['checkpoint']
		),
		'stage2_feasibility.student.init_checkpoint': (
			stage2_configs['feasibility']['student']['init_checkpoint']
		),
		'stage2_full.teacher.checkpoint': stage2_configs['full']['teacher'][
			'checkpoint'
		],
		'stage2_full.student.init_checkpoint': (
			stage2_configs['full']['student']['init_checkpoint']
		),
	}
	assert set(random_inputs.values()) == {random_checkpoint}
	namespaced_inputs = {
		'clustering.embeddings.input_dir': clustering_config['embeddings']['input_dir'],
		'stage2_feasibility.pseudo_targets.input_dir': (
			stage2_configs['feasibility']['pseudo_targets']['input_dir']
		),
		'stage2_full.pseudo_targets.input_dir': (
			stage2_configs['full']['pseudo_targets']['input_dir']
		),
		'extraction.embeddings.checkpoint': extraction_config['embeddings'][
			'checkpoint'
		],
		'candidate.checkpoint': candidate_raw['candidate']['checkpoint'],
		'candidate.embeddings_dir': candidate_raw['candidate']['embeddings_dir'],
	}
	for key, value in namespaced_inputs.items():
		assert value.startswith(f'{artifact_root}/'), (key, value)
		assert f'/{CONTROL_NAMESPACE}/' in value, (key, value)
		assert 'mae' not in value.lower(), (key, value)


def test_recipe_text_references_mae_only_through_allowed_anchors() -> None:
	files = _recipe_files()
	assert len(files) == RECIPE_COUNT
	for path in files:
		for line in _significant_lines(path):
			lowered = line.lower()
			for forbidden in FORBIDDEN_LINEAGE_TOKENS:
				assert forbidden not in lowered, (path, line)
			if 'mae' in lowered:
				assert any(anchor in line for anchor in ALLOWED_MAE_TEXT_ANCHORS), (
					path,
					line,
				)


def test_extraction_matches_v2_random_extraction_except_source_and_output(
	stage2_configs: dict[str, dict[str, object]],
	extraction_config: dict[str, object],
	artifact_root: Path,
) -> None:
	extract = extraction_config
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
	assert _differing_keys(EXTRACTION_CONFIG, V2_RANDOM_EXTRACTION_CONFIG) == [
		'checkpoint',
		'output_dir',
	]


def test_candidate_resolves_against_canonical_v3_without_collisions(
	stage2_configs: dict[str, dict[str, object]],
	extraction_config: dict[str, object],
	artifact_root: Path,
) -> None:
	base_mapping, _five_way_summary_root = _candidate_base_mapping()
	candidate = f3_lithology_candidate_config_from_mapping(base_mapping)
	canonical = load_f3_lithology_candidate_canonical_config(candidate)
	full_root = stage2_configs['full']['paths']['output_root']

	assert candidate.canonical_config == Path.cwd() / CANONICAL_FIVE_WAY_CONFIG
	assert candidate.candidate_id == CANDIDATE_ID
	assert candidate.candidate_id not in canonical.model_ids
	assert PREVIOUS_CONTROL_CANDIDATE_ID not in canonical.model_ids
	assert candidate.checkpoint == Path(f'{full_root}/latest.pt')
	assert candidate.embeddings_dir == Path(
		extraction_config['embeddings']['output_dir']
	)
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
	others = sorted(
		path
		for pattern in ('*.yaml', '*.sh')
		for path in Path('experiments').rglob(pattern)
		if not path.is_relative_to(EXP_ROOT)
	)
	assert others
	for path in others:
		text = path.read_text(encoding='utf-8')
		for token in tokens:
			assert token not in text, (path, token)


def test_control_does_not_reuse_the_previous_control_namespace() -> None:
	tokens = (
		PREVIOUS_CONTROL_NAMESPACE,
		PREVIOUS_CONTROL_CANDIDATE_ID,
		PREVIOUS_CONTROL_MODEL_TAG,
		PREVIOUS_CONTROL_PSEUDO_TARGET_SOURCE,
	)
	files = _recipe_files()
	assert len(files) == RECIPE_COUNT
	for path in files:
		text = path.read_text(encoding='utf-8')
		assert CONTROL_NAMESPACE in text, path
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
	for filename in RUNS.values():
		with pytest.raises(FileNotFoundError, match=r'teacher\.checkpoint must exist'):
			resolve_strat_hmm_pretext_config(load_config(STAGE2_ROOT / filename))


def test_resolver_requires_the_exported_pseudo_target_root_to_exist(
	artifact_root: Path,
) -> None:
	(artifact_root / PSEUDO_TARGET_ROOT).rmdir()
	assert (artifact_root / MAE100_PSEUDO_TARGET_DIR).is_dir()
	for filename in RUNS.values():
		with pytest.raises(
			FileNotFoundError, match=r'pseudo_targets\.input_dir must exist'
		):
			resolve_strat_hmm_pretext_config(load_config(STAGE2_ROOT / filename))


def test_candidate_rejects_a_canonical_model_id() -> None:
	base_mapping, _five_way_summary_root = _candidate_base_mapping()
	broken = deepcopy(base_mapping)
	broken['candidate']['id'] = 'random'
	candidate = f3_lithology_candidate_config_from_mapping(broken)
	with pytest.raises(ValueError, match='conflicts with canonical model ID'):
		load_f3_lithology_candidate_canonical_config(candidate)


def test_dry_runs_create_no_artifacts(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	_prepare_live_inputs(artifact_root)
	_prepare_fake_cluster_labels(artifact_root)
	bash = shutil.which('bash')
	assert bash is not None
	environment = {
		**os.environ,
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root),
		'CUDA_VISIBLE_DEVICES': '',
		# The export script invokes a bare `python`; pin it to the test interpreter.
		'PATH': os.pathsep.join(
			[str(Path(sys.executable).parent), os.environ.get('PATH', '')]
		),
	}
	commands = (
		(
			[
				sys.executable,
				'proc/seis_ssl_cluster/extract_embeddings.py',
				'--config',
				str(TARGET_EXTRACTION_CONFIG),
				'--dry-run',
			],
			'execution: dry-run; extraction skipped',
		),
		(
			[
				sys.executable,
				'proc/seis_ssl_cluster/cluster_embeddings.py',
				'--config',
				str(CLUSTERING_CONFIG),
				'--dry-run',
			],
			'execution: dry-run; clustering skipped',
		),
		(
			[bash, str(EXPORT_SCRIPT), '--dry-run'],
			'execution: dry-run; no files written',
		),
		(
			[
				sys.executable,
				'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
				'--config',
				str(STAGE2_ROOT / RUNS['feasibility']),
				'--dry-run',
			],
			'execution: dry-run; training skipped',
		),
		(
			[
				sys.executable,
				'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
				'--config',
				str(STAGE2_ROOT / RUNS['full']),
				'--dry-run',
			],
			'execution: dry-run; training skipped',
		),
		(
			[
				sys.executable,
				'proc/seis_ssl_cluster/extract_embeddings.py',
				'--config',
				str(EXTRACTION_CONFIG),
				'--dry-run',
			],
			'execution: dry-run; extraction skipped',
		),
	)
	before = {str(path) for path in artifact_root.rglob('*')}
	outputs = []
	for command, expected in commands:
		result = subprocess.run(  # noqa: S603
			command,
			check=True,
			capture_output=True,
			text=True,
			env=environment,
		)
		assert expected in result.stdout, command
		outputs.append(result.stdout)
	export_stdout = outputs[2]
	assert 'pseudo_target_exports: 1' in export_stdout
	assert str(artifact_root / PSEUDO_TARGET_ROOT / 'k6' / FAKE_SURVEY_ID) in (
		export_stdout
	)
	after = {str(path) for path in artifact_root.rglob('*')}
	assert after == before
