from __future__ import annotations

import os
import shlex
import stat
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_clustering_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.stratigraphy import pseudo_target_paths

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'37_channel_hmm_boundary_weight_v1'
)
TARGET_ROOT = EXPERIMENT_ROOT / '10_pseudo_targets'
STAGE2_ROOT = EXPERIMENT_ROOT / '20_stage2'
H0_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1'
)
SOURCE_METHODS = {
	'mae100': 'mae',
	'local_bt100': 'local_barlow_twins_v1',
}
SOURCES = tuple(SOURCE_METHODS)
VARIANT_ALPHA = {
	'alpha050_tau1': 0.5,
	'alpha100_tau1': 1.0,
}
VARIANTS = tuple(VARIANT_ALPHA)


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	for source, source_method in SOURCE_METHODS.items():
		checkpoint = (
			root
			/ 'pretraining/parihaka/facies_benchmark_v1'
			/ 'ssl_hmm_continuation_v1/stage1'
			/ source_method
			/ 'full_100ep/latest.pt'
		)
		checkpoint.parent.mkdir(parents=True, exist_ok=True)
		checkpoint.touch()
		(
			root
			/ 'pseudo_targets/parihaka/facies_benchmark_v1'
			/ f'ssl_hmm_continuation_v1/{source}/k6'
		).mkdir(parents=True)
		for variant in VARIANTS:
			(
				root
				/ 'pseudo_targets/parihaka/facies_benchmark_v1'
				/ f'hmm_boundary_weight_v1/{source}/{variant}/k6'
			).mkdir(parents=True)
	return root


@pytest.fixture
def stage2_configs(
	artifact_root: Path,
) -> dict[str, dict[str, dict[str, object]]]:
	del artifact_root
	return {
		source: {
			variant: resolve_strat_hmm_pretext_config(
				load_config(
					STAGE2_ROOT / source / variant / '01_full_25ep.yaml'
				)
			)
			for variant in VARIANTS
		}
		for source in SOURCES
	}


@pytest.fixture
def h0_stage2_configs(
	artifact_root: Path,
) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		source: resolve_strat_hmm_pretext_config(
			load_config(
				H0_ROOT
				/ '30_stage2'
				/ source
				/ 'hmm/k6/02_full_25ep.yaml'
			)
		)
		for source in SOURCES
	}


@pytest.fixture
def h0_clustering_configs(
	artifact_root: Path,
) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		source: resolve_clustering_config(
			load_config(
				H0_ROOT
				/ '20_hmm_targets'
				/ source
				/ 'k6/02_cluster_hmm_k6.yaml'
			)
		)
		for source in SOURCES
	}


def test_export_scripts_reuse_h0_clustering_and_fix_boundary_flags(
	h0_clustering_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	pseudo_target_roots: set[Path] = set()
	for source in SOURCES:
		h0_clustering = _section(h0_clustering_configs[source], 'clustering')
		hmm = _section(h0_clustering, 'stratigraphic_hmm')
		transition = _section(hmm, 'transition')
		assert transition['same_cost'] == 0.03
		assert transition['advance_cost'] == 0.0
		assert h0_clustering['k_values'] == [6]

		for variant, alpha in VARIANT_ALPHA.items():
			script = TARGET_ROOT / source / variant / '01_export_pseudo_targets.sh'
			arguments = _export_script_arguments(script)
			expected_root = (
				artifact_root
				/ 'pseudo_targets/parihaka/facies_benchmark_v1'
				/ f'hmm_boundary_weight_v1/{source}/{variant}'
			)
			assert arguments == {
				'--clustering-output-dir': str(h0_clustering['output_dir']),
				'--pseudo-target-root': str(expected_root),
				'--k': '6',
				'--confidence': '1.0',
				'--boundary-alpha': str(alpha),
				'--boundary-tau': '1.0',
				'--schema-version': '2',
			}
			assert '--overwrite' not in arguments
			paths = pseudo_target_paths(expected_root, k=6, survey_id='survey')
			assert paths.labels.parent == expected_root / 'k6'
			pseudo_target_roots.add(expected_root)

	assert len(pseudo_target_roots) == 4
	_assert_pairwise_disjoint(pseudo_target_roots)
	h0_roots = {
		artifact_root
		/ 'pseudo_targets/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/{source}'
		for source in SOURCES
	}
	for new_root in pseudo_target_roots:
		for h0_root in h0_roots:
			assert _paths_do_not_overlap(new_root, h0_root)


def test_target_tree_has_only_four_exports_and_no_new_clustering() -> None:
	actual_scripts = {
		path.relative_to(TARGET_ROOT)
		for path in TARGET_ROOT.rglob('*.sh')
	}
	expected_scripts = {
		Path(source) / variant / '01_export_pseudo_targets.sh'
		for source in SOURCES
		for variant in VARIANTS
	}
	assert actual_scripts == expected_scripts
	assert not tuple(TARGET_ROOT.rglob('*.yaml'))
	assert not tuple(EXPERIMENT_ROOT.rglob('*cluster*.yaml'))
	assert not tuple(EXPERIMENT_ROOT.rglob('*alpha000*'))


def test_stage2_configs_resolve_from_stage1_and_boundary_targets(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
	artifact_root: Path,
) -> None:
	for source, source_method in SOURCE_METHODS.items():
		expected_checkpoint = (
			artifact_root
			/ 'pretraining/parihaka/facies_benchmark_v1'
			/ 'ssl_hmm_continuation_v1/stage1'
			/ source_method
			/ 'full_100ep/latest.pt'
		)
		for variant in VARIANTS:
			config = stage2_configs[source][variant]
			targets = Path(
				str(_section(config, 'pseudo_targets')['input_dir'])
			)
			expected_targets = (
				artifact_root
				/ 'pseudo_targets/parihaka/facies_benchmark_v1'
				/ f'hmm_boundary_weight_v1/{source}/{variant}'
			)
			teacher = Path(str(_section(config, 'teacher')['checkpoint']))
			student = Path(
				str(_section(config, 'student')['init_checkpoint'])
			)

			assert config['stage'] == 'train_strat_hmm_pretext'
			assert teacher == student == expected_checkpoint
			assert 'stage2' not in teacher.parts
			assert targets == expected_targets
			paths = pseudo_target_paths(targets, k=6, survey_id='survey')
			assert paths.labels.parent == expected_targets / 'k6'


def test_stage2_differs_from_source_h0_only_by_input_and_output(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
	h0_stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	for source in SOURCES:
		for variant in VARIANTS:
			config = stage2_configs[source][variant]
			expected_output = (
				artifact_root
				/ 'pretraining/parihaka/facies_benchmark_v1'
				/ f'hmm_boundary_weight_v1/{source}/{variant}/full_25ep'
			)
			assert _without_run_paths(config) == _without_run_paths(
				h0_stage2_configs[source]
			)
			assert Path(
				str(_section(config, 'paths')['output_root'])
			) == expected_output


def test_stage2_scientific_budget_and_optimizer_contract_is_fixed(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for source in SOURCES:
		for variant in VARIANTS:
			config = stage2_configs[source][variant]
			pseudo_targets = _section(config, 'pseudo_targets')
			student = _section(config, 'student')
			head = _section(config, 'head')
			loss = _section(config, 'loss')
			train = _section(config, 'train')

			assert pseudo_targets['k'] == 6
			assert pseudo_targets['min_confidence'] == 0.0
			assert student['unfreeze_top_blocks'] == 1
			assert head == {
				'num_prototypes': 6,
				'projection_dim': 128,
				'temperature': 0.1,
				'normalize': True,
			}
			assert {
				key: loss[key]
				for key in (
					'prototype_weight',
					'usage_weight',
					'entropy_floor',
					'distillation_weight',
				)
			} == {
				'prototype_weight': 1.0,
				'usage_weight': 0.005,
				'entropy_floor': None,
				'distillation_weight': 0.2,
			}
			assert train['batch_size'] == 16
			assert train['samples_per_epoch'] == 10_000
			assert train['epochs'] == 25
			assert train['lr'] == 1.0e-5
			assert train['encoder_lr'] == 1.0e-5
			assert train['weight_decay'] == 0.05
			assert train['amp'] is False
			assert train['seed'] == 42
			assert train['grad_clip_norm'] == 1.0
			assert train['max_steps'] is None
			assert train['allow_overwrite_output'] is False
			assert (
				int(train['epochs'])
				* int(train['samples_per_epoch'])
				// int(train['batch_size'])
			) == 15_625


def test_stage2_output_roots_are_unique_and_isolated(
	stage2_configs: dict[str, dict[str, dict[str, object]]],
	h0_stage2_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	outputs = {
		Path(str(_section(config, 'paths')['output_root']))
		for source_configs in stage2_configs.values()
		for config in source_configs.values()
	}
	expected_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ f'hmm_boundary_weight_v1/{source}/{variant}/full_25ep'
		for source in SOURCES
		for variant in VARIANTS
	}
	h0_outputs = {
		Path(str(_section(config, 'paths')['output_root']))
		for config in h0_stage2_configs.values()
	}
	control_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/stage2/{relative}'
		for relative in (
			'mae100/mae_continue/full_25ep',
			'local_bt100/bt_continue/full_25ep',
		)
	}
	stage1_outputs = {
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'ssl_hmm_continuation_v1/stage1'
		/ source_method
		/ 'full_100ep'
		for source_method in SOURCE_METHODS.values()
	}
	pseudo_target_artifacts = {
		artifact_root
		/ 'pseudo_targets/parihaka/facies_benchmark_v1'
		/ f'hmm_boundary_weight_v1/{source}/{variant}'
		for source in SOURCES
		for variant in VARIANTS
	}
	pseudo_target_artifacts.update(
		{
			artifact_root
			/ 'pseudo_targets/parihaka/facies_benchmark_v1'
			/ f'ssl_hmm_continuation_v1/{source}'
			for source in SOURCES
		}
	)

	assert outputs == expected_outputs
	assert len(outputs) == 4
	_assert_pairwise_disjoint(outputs)
	for output in outputs:
		for existing in (
			h0_outputs
			| control_outputs
			| stage1_outputs
			| pseudo_target_artifacts
		):
			assert _paths_do_not_overlap(output, existing)


def test_stage2_contains_only_full_configs_and_no_feasibility_yaml() -> None:
	actual_configs = {
		path.relative_to(STAGE2_ROOT)
		for path in STAGE2_ROOT.rglob('*.yaml')
	}
	expected_configs = {
		Path(source) / variant / '01_full_25ep.yaml'
		for source in SOURCES
		for variant in VARIANTS
	}
	assert actual_configs == expected_configs
	assert not tuple(STAGE2_ROOT.rglob('*feasibility*.yaml'))
	assert not tuple(STAGE2_ROOT.rglob('*alpha000*'))


def _export_script_arguments(path: Path) -> dict[str, str]:
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
	assert tokens[:2] == [
		'python',
		'proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py',
	]
	assert tokens[-1] == '$@'
	flag_tokens = tokens[2:-1]
	assert len(flag_tokens) % 2 == 0
	arguments = dict(zip(flag_tokens[::2], flag_tokens[1::2], strict=True))
	assert len(arguments) * 2 == len(flag_tokens)
	return arguments


def _without_run_paths(config: dict[str, object]) -> dict[str, object]:
	comparison = deepcopy(config)
	del _section(comparison, 'paths')['output_root']
	del _section(comparison, 'pseudo_targets')['input_dir']
	return comparison


def _section(config: dict[str, object], key: str) -> dict[str, object]:
	value = config[key]
	if not isinstance(value, dict):
		raise TypeError(f'{key} must be a mapping')
	return value


def _assert_pairwise_disjoint(paths: set[Path]) -> None:
	ordered_paths = sorted(paths)
	for index, left in enumerate(ordered_paths):
		for right in ordered_paths[index + 1 :]:
			assert _paths_do_not_overlap(left, right)


def _paths_do_not_overlap(left: Path, right: Path) -> bool:
	return left != right and left not in right.parents and right not in left.parents
