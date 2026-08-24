from __future__ import annotations

import os
import shlex
import stat
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config, resolve_clustering_config
from seis_ssl_cluster.stratigraphy import pseudo_target_paths

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'36_channel_hmm_transition_balance_v1'
)
TARGET_ROOT = EXPERIMENT_ROOT / '10_hmm_targets'
H0_TARGET_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'21_ssl_hmm_continuation_v1/20_hmm_targets'
)
SOURCES = ('mae100', 'local_bt100')
VARIANT_COSTS = {
	'neutral': (0.0, 0.0),
	'persist003': (0.0, 0.03),
	'persist010': (0.0, 0.10),
}
VARIANTS = tuple(VARIANT_COSTS)


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


@pytest.fixture
def clustering_configs(
	artifact_root: Path,
) -> dict[str, dict[str, dict[str, object]]]:
	del artifact_root
	return {
		source: {
			variant: resolve_clustering_config(
				load_config(
					TARGET_ROOT
					/ source
					/ variant
					/ '01_cluster_hmm_k6.yaml'
				)
			)
			for variant in VARIANTS
		}
		for source in SOURCES
	}


@pytest.fixture
def h0_clustering_configs(
	artifact_root: Path,
) -> dict[str, dict[str, object]]:
	del artifact_root
	return {
		source: resolve_clustering_config(
			load_config(H0_TARGET_ROOT / source / 'k6/02_cluster_hmm_k6.yaml')
		)
		for source in SOURCES
	}


def test_configs_resolve_and_differ_from_h0_only_by_transition_and_output(
	clustering_configs: dict[str, dict[str, dict[str, object]]],
	h0_clustering_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	for source in SOURCES:
		h0 = h0_clustering_configs[source]
		for variant, expected_costs in VARIANT_COSTS.items():
			config = clustering_configs[source][variant]
			clustering = _section(config, 'clustering')
			transition = _transition(config)
			expected_output = (
				artifact_root
				/ 'clustering/parihaka/facies_benchmark_v1'
				/ f'hmm_transition_balance_v1/{source}/{variant}'
			)

			assert config['stage'] == 'cluster_embeddings'
			assert config['embeddings'] == h0['embeddings']
			assert _without_transition_balance_fields(config) == (
				_without_transition_balance_fields(h0)
			)
			assert Path(str(clustering['output_dir'])) == expected_output
			assert (
				transition['same_cost'],
				transition['advance_cost'],
			) == expected_costs


def test_sources_are_scientifically_paired_for_each_variant(
	clustering_configs: dict[str, dict[str, dict[str, object]]],
	artifact_root: Path,
) -> None:
	expected_inputs = {
		source: artifact_root
		/ 'embeddings/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/hmm_targets/{source}/overlap_x64'
		for source in SOURCES
	}

	for variant in VARIANTS:
		mae = clustering_configs['mae100'][variant]
		local_bt = clustering_configs['local_bt100'][variant]
		assert _without_source_paths(mae) == _without_source_paths(local_bt)
		for source, config in (
			('mae100', mae),
			('local_bt100', local_bt),
		):
			assert Path(str(_section(config, 'embeddings')['input_dir'])) == (
				expected_inputs[source]
			)


def test_scientific_transition_contract_is_fixed(
	clustering_configs: dict[str, dict[str, dict[str, object]]],
) -> None:
	for source in SOURCES:
		for variant in VARIANTS:
			clustering = _section(clustering_configs[source][variant], 'clustering')
			hmm = _section(clustering, 'stratigraphic_hmm')

			assert clustering['embedding_normalization'] == 'l2'
			assert clustering['residualization'] == {
				'enabled': True,
				'mode': 'local_token_position',
				'group_by': 'token_phase',
				'add_global_mean_back': True,
				'min_group_count': 32,
			}
			assert clustering['pca'] == {
				'enabled': True,
				'n_components': 64,
				'whiten': False,
			}
			assert clustering['sample_tokens'] == 1_000_000
			assert clustering['method'] == 'stratigraphic_hmm_kmeans'
			assert clustering['k_values'] == [6]
			assert clustering['seed'] == 42
			assert hmm['iterations'] == 10
			assert hmm['edge_margin_tokens'] == [8, 8, 0]
			assert _transition(clustering_configs[source][variant]) == {
				'same_cost': VARIANT_COSTS[variant][0],
				'advance_cost': VARIANT_COSTS[variant][1],
				'jump_cost': 1.0,
				'reverse_cost': 1_000_000.0,
				'forbid_reverse': True,
				'max_jump': 1,
			}
			assert hmm['path_prior'] == {
				'enabled': True,
				'initial_state': {
					'mode': 'shallow_anchor',
					'weight': 0.25,
				},
				'terminal_state': {
					'mode': 'deep_anchor',
					'weight': 0.25,
				},
				'expected_boundaries': {'enabled': False},
			}
			assert hmm['init'] == {'order_by': 'mean_z'}
			assert hmm['update'] == {
				'empty_cluster_policy': 'keep_previous'
			}


def test_export_scripts_match_sources_outputs_and_fixed_flags(
	clustering_configs: dict[str, dict[str, dict[str, object]]],
	artifact_root: Path,
) -> None:
	clustering_outputs: set[Path] = set()
	pseudo_target_roots: set[Path] = set()
	pseudo_target_directories: set[Path] = set()

	for source in SOURCES:
		for variant in VARIANTS:
			script_path = (
				TARGET_ROOT / source / variant / '02_export_pseudo_targets.sh'
			)
			args = _export_script_arguments(script_path)
			expected_pseudo_root = (
				artifact_root
				/ 'pseudo_targets/parihaka/facies_benchmark_v1'
				/ f'hmm_transition_balance_v1/{source}/{variant}'
			)
			clustering_output = Path(
				str(
					_section(
						clustering_configs[source][variant],
						'clustering',
					)['output_dir']
				)
			)

			assert args == {
				'--clustering-output-dir': str(clustering_output),
				'--pseudo-target-root': str(expected_pseudo_root),
				'--k': '6',
				'--confidence': '1.0',
				'--boundary-alpha': '0.0',
				'--boundary-tau': '1.0',
				'--schema-version': '2',
			}
			assert script_path.stat().st_mode & stat.S_IXUSR
			paths = pseudo_target_paths(
				expected_pseudo_root,
				k=6,
				survey_id='survey',
			)
			assert paths.labels.parent == expected_pseudo_root / 'k6'
			assert paths.labels.parent != expected_pseudo_root / 'k6/k6'

			clustering_outputs.add(clustering_output)
			pseudo_target_roots.add(expected_pseudo_root)
			pseudo_target_directories.add(paths.labels.parent)

	assert len(clustering_outputs) == 6
	assert len(pseudo_target_roots) == 6
	assert len(pseudo_target_directories) == 6


def test_outputs_are_isolated_from_h0(
	clustering_configs: dict[str, dict[str, dict[str, object]]],
	h0_clustering_configs: dict[str, dict[str, object]],
	artifact_root: Path,
) -> None:
	new_clustering_outputs = {
		Path(
			str(
				_section(clustering_configs[source][variant], 'clustering')[
					'output_dir'
				]
			)
		)
		for source in SOURCES
		for variant in VARIANTS
	}
	h0_clustering_outputs = {
		Path(str(_section(config, 'clustering')['output_dir']))
		for config in h0_clustering_configs.values()
	}
	new_pseudo_target_roots = {
		artifact_root
		/ 'pseudo_targets/parihaka/facies_benchmark_v1'
		/ f'hmm_transition_balance_v1/{source}/{variant}'
		for source in SOURCES
		for variant in VARIANTS
	}
	h0_pseudo_target_roots = {
		artifact_root
		/ 'pseudo_targets/parihaka/facies_benchmark_v1'
		/ f'ssl_hmm_continuation_v1/{source}'
		for source in SOURCES
	}

	_assert_pairwise_disjoint(new_clustering_outputs)
	_assert_pairwise_disjoint(new_pseudo_target_roots)
	for new_path in new_clustering_outputs:
		for h0_path in h0_clustering_outputs:
			assert _paths_do_not_overlap(new_path, h0_path)
	for new_path in new_pseudo_target_roots:
		for h0_path in h0_pseudo_target_roots:
			assert _paths_do_not_overlap(new_path, h0_path)


def test_target_leaves_contain_only_cluster_and_export_files() -> None:
	for source in SOURCES:
		for variant in VARIANTS:
			leaf = TARGET_ROOT / source / variant
			assert {path.name for path in leaf.iterdir()} == {
				'01_cluster_hmm_k6.yaml',
				'02_export_pseudo_targets.sh',
			}

	assert not tuple(TARGET_ROOT.rglob('*extract*embeddings*.yaml'))


def _without_transition_balance_fields(
	config: dict[str, object],
) -> dict[str, object]:
	comparison = deepcopy(config)
	clustering = _section(comparison, 'clustering')
	del clustering['output_dir']
	transition = _section(
		_section(clustering, 'stratigraphic_hmm'),
		'transition',
	)
	del transition['same_cost']
	del transition['advance_cost']
	return comparison


def _without_source_paths(config: dict[str, object]) -> dict[str, object]:
	comparison = deepcopy(config)
	del _section(comparison, 'embeddings')['input_dir']
	del _section(comparison, 'clustering')['output_dir']
	return comparison


def _transition(config: dict[str, object]) -> dict[str, object]:
	clustering = _section(config, 'clustering')
	hmm = _section(clustering, 'stratigraphic_hmm')
	return _section(hmm, 'transition')


def _section(config: dict[str, object], key: str) -> dict[str, object]:
	value = config[key]
	if not isinstance(value, dict):
		raise TypeError(f'{key} must be a mapping')
	return value


def _export_script_arguments(path: Path) -> dict[str, str]:
	text = path.read_text(encoding='utf-8')
	assert text.startswith('#!/usr/bin/env bash\n')
	assert 'set -euo pipefail' in text
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
	flag_tokens = tokens[2:]
	assert len(flag_tokens) % 2 == 0
	arguments = dict(zip(flag_tokens[::2], flag_tokens[1::2], strict=True))
	assert len(arguments) * 2 == len(flag_tokens)
	return arguments


def _assert_pairwise_disjoint(paths: set[Path]) -> None:
	ordered_paths = sorted(paths)
	for index, left in enumerate(ordered_paths):
		for right in ordered_paths[index + 1 :]:
			assert _paths_do_not_overlap(left, right)


def _paths_do_not_overlap(left: Path, right: Path) -> bool:
	return left != right and left not in right.parents and right not in left.parents
