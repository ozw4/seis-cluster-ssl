from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from seis_ssl_cluster.config.pretraining import (
	resolve_mae_training_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.config.strat_hmm_pseudo_targets import (
	resolve_strat_hmm_pseudo_target_config,
)
from seis_ssl_cluster.config.validate import (
	resolve_mae_training_config as compat_mae_training_resolver,
)
from seis_ssl_cluster.config.validate import (
	resolve_strat_hmm_pretext_config as compat_strat_hmm_pretext_resolver,
)
from seis_ssl_cluster.config.validate import (
	resolve_strat_hmm_pseudo_target_config as compat_strat_hmm_pseudo_target_resolver,
)

CONFIG_PACKAGE = 'seis_ssl_cluster.config'
CONFIG_ROOT = Path('src/seis_ssl_cluster/config')
VALIDATE_PATH = CONFIG_ROOT / 'validate.py'
F3_LITHOLOGY_COMMON_PATH = CONFIG_ROOT / 'f3_lithology_common.py'
STAGE_MODULES = {
	'cluster_visualization',
	'clustering',
	'embedding',
	'f3_baselines',
	'f3_inspection',
	'f3_lithology',
	'manifest',
	'normalization',
	'pretraining',
	'results',
	'strat_hmm_pseudo_targets',
}
DIRECT_STAGE_IMPORTS = (
	('cluster_visualization', 'resolve_cluster_visualization_config'),
	('clustering', 'resolve_clustering_config'),
	('embedding', 'resolve_embedding_extraction_config'),
	('f3_baselines', 'f3_lithology_baseline_token_dataset_config_from_mapping'),
	('f3_baselines', 'f3_lithology_comparison_report_config_from_mapping'),
	('f3_baselines', 'random_mae_checkpoint_config_from_mapping'),
	('f3_inspection', 'resolve_f3_facies_inspection_config'),
	('f3_lithology', 'f3_lithology_prediction_config_from_mapping'),
	('f3_lithology', 'f3_lithology_probe_config_from_mapping'),
	('f3_lithology', 'f3_lithology_publish_config_from_mapping'),
	('f3_lithology', 'f3_lithology_report_config_from_mapping'),
	('f3_lithology', 'f3_lithology_token_dataset_config_from_mapping'),
	('f3_lithology', 'f3_lithology_visualization_config_from_mapping'),
	('f3_lithology', 'f3_prepare_volume_config_from_mapping'),
	('manifest', 'resolve_manifest_build_config'),
	('normalization', 'resolve_normalization_qc_config'),
	('normalization', 'resolve_normalization_stats_config'),
	('pretraining', 'resolve_mae_training_config'),
	('pretraining', 'resolve_strat_hmm_pretext_config'),
	('results', 'validate_results_artifacts'),
	('strat_hmm_pseudo_targets', 'resolve_strat_hmm_pseudo_target_config'),
)
PROC_IMPORTS = (
	'proc.seis_ssl_cluster.build_strat_hmm_pseudo_targets',
	'proc.seis_ssl_cluster.build_nopims_manifests',
	'proc.seis_ssl_cluster.prepare_nopims_normalization_stats',
	'proc.seis_ssl_cluster.filter_manifest_by_normalization_qc',
	'proc.seis_ssl_cluster.train_amp_mae',
	'proc.seis_ssl_cluster.train_strat_hmm_pretext',
	'proc.seis_ssl_cluster.extract_embeddings',
	'proc.seis_ssl_cluster.cluster_embeddings',
	'proc.seis_ssl_cluster.visualize_clusters',
	'proc.seis_ssl_cluster.inspect_f3_files',
	'proc.seis_ssl_cluster.build_f3_lithology_report',
	'proc.seis_ssl_cluster.train_f3_lithology_probe',
	'proc.seis_ssl_cluster.build_f3_lithology_comparison_report',
)
F3_LITHOLOGY_INTERNAL_IMPORTS = (
	(
		'f3_lithology_prediction',
		'f3_lithology_prediction_config_from_mapping',
	),
	(
		'f3_lithology_publish',
		'f3_lithology_publish_config_from_mapping',
	),
	(
		'f3_lithology_report',
		'f3_lithology_report_config_from_mapping',
	),
	(
		'f3_lithology_visualization',
		'f3_lithology_visualization_config_from_mapping',
	),
)


def test_validate_module_reexports_public_resolver() -> None:
	assert compat_mae_training_resolver is resolve_mae_training_config
	assert compat_strat_hmm_pretext_resolver is resolve_strat_hmm_pretext_config
	assert (
		compat_strat_hmm_pseudo_target_resolver
		is resolve_strat_hmm_pseudo_target_config
	)


@pytest.mark.parametrize(('module_name', 'symbol'), DIRECT_STAGE_IMPORTS)
def test_stage_modules_export_public_config_symbols(
	module_name: str,
	symbol: str,
) -> None:
	module = importlib.import_module(f'{CONFIG_PACKAGE}.{module_name}')

	assert getattr(module, symbol).__name__ == symbol


@pytest.mark.parametrize(('module_name', 'symbol'), F3_LITHOLOGY_INTERNAL_IMPORTS)
def test_f3_lithology_internal_modules_export_stage_resolvers(
	module_name: str,
	symbol: str,
) -> None:
	module = importlib.import_module(f'{CONFIG_PACKAGE}.{module_name}')

	assert getattr(module, symbol).__name__ == symbol


def test_validate_module_imports_in_fresh_python_process() -> None:
	result = subprocess.run(
		[
			sys.executable,
			'-c',
			'import seis_ssl_cluster.config.validate',
		],
		check=False,
		capture_output=True,
		text=True,
	)

	assert result.returncode == 0, result.stderr


def test_stage_modules_do_not_import_validate_module() -> None:
	for module_name in STAGE_MODULES:
		imports = _imported_modules(CONFIG_ROOT / f'{module_name}.py')

		assert 'seis_ssl_cluster.config.validate' not in imports
		assert '.validate' not in imports


def test_stage_modules_do_not_import_proc_modules() -> None:
	for module_name in STAGE_MODULES:
		imports = _imported_modules(CONFIG_ROOT / f'{module_name}.py')

		assert not [name for name in imports if name.startswith('proc')]


def test_f3_lithology_common_does_not_import_validate_or_proc_modules() -> None:
	imports = _imported_modules(F3_LITHOLOGY_COMMON_PATH)

	assert 'seis_ssl_cluster.config.validate' not in imports
	assert '.validate' not in imports
	assert not [name for name in imports if name.startswith('proc')]


def test_common_module_does_not_import_stage_modules() -> None:
	imports = _imported_modules(CONFIG_ROOT / 'common.py')
	for module_name in STAGE_MODULES:
		assert f'seis_ssl_cluster.config.{module_name}' not in imports
		assert f'.{module_name}' not in imports


@pytest.mark.parametrize('module_name', PROC_IMPORTS)
def test_primary_proc_modules_import_without_config_cycles(module_name: str) -> None:
	importlib.import_module(module_name)


def test_validate_py_is_compat_layer() -> None:
	assert VALIDATE_PATH.read_text().count('\n') < 250


def _imported_modules(path: Path) -> set[str]:
	tree = ast.parse(path.read_text())
	imports: set[str] = set()
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			imports.update(alias.name for alias in node.names)
		elif isinstance(node, ast.ImportFrom):
			module = node.module or ''
			prefix = '.' * node.level
			imports.add(f'{prefix}{module}')
	return imports
