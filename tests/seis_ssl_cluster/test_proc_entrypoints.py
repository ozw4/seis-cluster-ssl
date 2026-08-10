# ruff: noqa: CPY001

from __future__ import annotations

import argparse
import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = REPO_ROOT / 'proc' / 'seis_ssl_cluster'

PROC_SCRIPTS = tuple(
	path for path in sorted(PROC_DIR.glob('*.py')) if path.name != '__init__.py'
)

HELP_FLAG_CONTRACTS = {
	'summarize_f3_lithology_voxel_section_layout.py': (
		'--config',
		'--model-id',
		'--no-publish',
	),
	'run_f3_lithology_voxel_section_layout_suite.py': (
		'--config',
		'--model-id',
		'--layout-id',
		'--data-size',
		'--dry-run',
		'--only-missing',
		'--resume',
		'--quarantine-invalid',
		'--smoke-only',
		'--device',
	),
	'build_f3_lithology_voxel_section_layout_datasets.py': (
		'--config',
		'--dry-run',
		'--only-missing',
		'--quarantine-invalid',
	),
	'prepare_f3_lithology_voxel_section_layout_contract.py': (
		'--config',
		'--mode',
		'--dry-run',
	),
	'audit_f3_xy_neighbor_unanimous_targets.py': (
		'--config',
		'--dry-run',
		'--only-missing',
		'--quarantine-invalid',
	),
	'audit_f3_xy_neighbor_consensus_screening.py': (
		'--config',
		'--dry-run',
		'--only-missing',
		'--quarantine-invalid',
	),
	'audit_f3_center_trace_masked_screening.py': (
		'--config',
		'--dry-run',
		'--only-missing',
		'--quarantine-invalid',
	),
	'audit_f3_center_trace_masked_periodic_refresh_screening.py': (
		'--config',
		'--dry-run',
		'--only-missing',
		'--quarantine-invalid',
	),
	'build_strat_hmm_pseudo_targets.py': (
		'--config',
		'--dry-run',
		'--device',
		'--overwrite',
	),
	'shuffle_strat_hmm_pseudo_targets.py': (
		'--config',
		'--dry-run',
		'--overwrite',
	),
	'build_f3_lithology_baseline_features.py': (
		'--config',
		'--dry-run',
	),
	'build_f3_lithology_baseline_token_dataset.py': (
		'--config',
		'--dry-run',
	),
	'build_f3_lithology_label_budget_datasets.py': (
		'--config',
		'--dry-run',
	),
	'build_f3_lithology_voxel_label_budget_datasets.py': (
		'--config',
		'--dry-run',
		'--only-missing',
	),
	'build_f3_lithology_split_sweep_datasets.py': (
		'--config',
		'--dry-run',
		'--only-missing',
	),
	'generate_f3_lithology_split_inventories.py': (
		'--config',
		'--dry-run',
	),
	'run_f3_lithology_label_budget_probes.py': (
		'--config',
		'--dry-run',
		'--only-missing',
	),
	'run_f3_lithology_voxel_label_budget_suite.py': (
		'--config',
		'--dry-run',
		'--device',
		'--only-missing',
		'--smoke-only',
		'--budget',
		'--subsample-seed',
		'--model',
	),
	'run_f3_lithology_voxel_label_budget_control.py': (
		'--config',
		'--dry-run',
		'--device',
		'--only-missing',
		'--resume',
		'--budget',
		'--subsample-seed',
	),
	'run_f3_lithology_xy_neighbor_consensus_voxel_label_budget.py': (
		'--config',
		'--dry-run',
		'--device',
		'--only-missing',
		'--resume',
	),
	'run_f3_lithology_center_trace_masked_voxel_label_budget.py': (
		'--config',
		'--dry-run',
		'--device',
		'--only-missing',
		'--resume',
	),
	'run_f3_lithology_center_trace_masked_periodic_refresh_voxel_label_budget.py': (
		'--config',
		'--dry-run',
		'--device',
		'--only-missing',
		'--resume',
	),
	'summarize_f3_lithology_voxel_label_budget_xy_neighbor_consensus.py': (
		'--config',
		'--dry-run',
	),
	'summarize_f3_lithology_voxel_label_budget_center_trace_masked.py': (
		'--config',
		'--dry-run',
	),
	'summarize_f3_lithology_voxel_label_budget_center_trace_masked_'
	'periodic_refresh.py': (
		'--config',
		'--dry-run',
	),
	'run_f3_lithology_split_sweep_probes.py': (
		'--config',
		'--dry-run',
		'--only-missing',
	),
	'summarize_f3_lithology_label_budget_robustness.py': (
		'--suite-root',
		'--dry-run',
	),
	'summarize_f3_lithology_voxel_label_budget.py': (
		'--config',
		'--dry-run',
	),
	'summarize_f3_lithology_voxel_label_budget_control.py': (
		'--config',
		'--dry-run',
	),
	'summarize_f3_lithology_split_robustness.py': (
		'--suite-root',
		'--dry-run',
	),
	'build_f3_lithology_comparison_report.py': (
		'--config',
		'--dry-run',
		'--search-root',
		'--output-dir',
		'--output-csv',
		'--output-markdown',
		'--metrics-json',
		'--figure-dpi',
	),
	'build_f3_lithology_report.py': (
		'--config',
		'--dry-run',
	),
	'build_f3_lithology_token_dataset.py': (
		'--config',
		'--dry-run',
	),
	'build_nopims_manifests.py': (
		'--config',
		'--dry-run',
	),
	'cluster_embeddings.py': (
		'--config',
		'--dry-run',
	),
	'create_random_mae_checkpoint.py': (
		'--config',
		'--dry-run',
	),
	'extract_embeddings.py': (
		'--config',
		'--dry-run',
		'--device',
		'--skip-existing',
	),
	'prepare_f3_facies_volume.py': (
		'--config',
		'--dry-run',
		'--overwrite',
	),
	'prepare_nopims_normalization_stats.py': (
		'--config',
		'--dry-run',
		'--overwrite',
	),
	'train_amp_mae.py': (
		'--config',
		'--dry-run',
		'--device',
		'--max-steps',
		'--output-root',
		'--resume',
	),
	'train_f3_lithology_probe.py': (
		'--config',
		'--dry-run',
	),
	'train_strat_hmm_pretext.py': (
		'--config',
		'--dry-run',
		'--device',
		'--max-steps',
		'--output-root',
		'--resume',
		'--quarantine-invalid',
	),
	'validate_performance_migration.py': (
		'--config',
		'--stage',
		'--embedding-config',
		'--hmm-config',
		'--device',
		'--dry-run',
		'--only-missing',
	),
	'export_strat_hmm_multi_head_state_posteriors.py': (
		'--config',
		'--dry-run',
		'--only-missing',
	),
	'export_strat_hmm_multi_head_xy_neighbor_unanimous_targets.py': (
		'--config',
		'--dry-run',
		'--only-missing',
	),
	'publish_f3_xy_neighbor_unanimous_results.py': (
		'--config',
		'--dry-run',
	),
	'validate_f3_xy_neighbor_unanimous_pretraining.py': (
		'--config',
		'--phase',
		'--dry-run',
		'--only-missing',
		'--quarantine-invalid',
	),
	'validate_f3_center_trace_masked_pretraining.py': (
		'--config',
		'--phase',
		'--dry-run',
		'--only-missing',
		'--quarantine-invalid',
	),
	'validate_f3_center_trace_masked_periodic_refresh.py': (
		'--config',
		'--phase',
		'--dry-run',
		'--only-missing',
		'--quarantine-invalid',
	),
	'publish_f3_center_trace_masked_pretraining_results.py': (
		'--artifact-root',
		'--workspace-root',
		'--pretraining-handoff',
		'--output-dir',
		'--dry-run',
	),
	'publish_f3_center_trace_masked_periodic_refresh_results.py': (
		'--config',
		'--dry-run',
		'--quarantine-invalid',
	),
	'visualize_clusters.py': (
		'--config',
		'--dry-run',
	),
	'visualize_f3_lithology_predictions.py': (
		'--config',
		'--dry-run',
	),
}

SMOKE_HELP_SCRIPTS = (
	'train_amp_mae.py',
	'extract_embeddings.py',
	'build_f3_lithology_report.py',
)


def test_proc_modules_import_without_running_main(
	capsys: pytest.CaptureFixture[str],
) -> None:
	for script in PROC_SCRIPTS:
		module = importlib.import_module(_module_name(script))
		assert hasattr(module, 'main')

	captured = capsys.readouterr()
	assert captured.out == ''
	assert captured.err == ''


def test_existing_build_parser_functions_construct_argparse_parsers() -> None:
	for script in PROC_SCRIPTS:
		module = importlib.import_module(_module_name(script))
		build_parser = getattr(module, 'build_parser', None)
		assert build_parser is not None, script

		parser = build_parser()

		assert isinstance(parser, argparse.ArgumentParser)


@pytest.mark.parametrize(
	('script_name', 'expected_flags'),
	HELP_FLAG_CONTRACTS.items(),
	ids=HELP_FLAG_CONTRACTS.keys(),
)
def test_primary_proc_help_preserves_existing_flags(
	script_name: str,
	expected_flags: tuple[str, ...],
) -> None:
	help_text = _parser_help_text(PROC_DIR / script_name)

	for flag in expected_flags:
		assert flag in help_text


@pytest.mark.parametrize('script_name', SMOKE_HELP_SCRIPTS)
def test_issue_smoke_import_help_commands(script_name: str) -> None:
	assert 'usage:' in _help_text(PROC_DIR / script_name)


def test_proc_main_functions_stay_thin_entrypoints() -> None:
	for script in PROC_SCRIPTS:
		main = _main_function(script)
		if main is None:
			continue

		statements = [
			stmt
			for stmt in main.body
			if not (
				isinstance(stmt, ast.Expr)
				and isinstance(stmt.value, ast.Constant)
				and isinstance(stmt.value.value, str)
			)
		]

		assert len(statements) <= 35, script


def _module_name(script: Path) -> str:
	return f'proc.seis_ssl_cluster.{script.stem}'


def _help_text(script: Path) -> str:
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(
			str(REPO_ROOT / 'src'),
			env.get('PYTHONPATH', ''),
		),
	)
	try:
		completed = subprocess.run(  # noqa: S603
			[
				sys.executable,
				str(script),
				'--help',
			],
			cwd=REPO_ROOT,
			env=env,
			text=True,
			capture_output=True,
			check=True,
			timeout=30,
		)
	except subprocess.TimeoutExpired as exc:
		raise AssertionError(f'--help timed out for {script.name}') from exc
	return completed.stdout + completed.stderr


def _parser_help_text(script: Path) -> str:
	module = importlib.import_module(_module_name(script))
	build_parser = getattr(module, 'build_parser', None)
	assert build_parser is not None, script
	parser = build_parser()
	assert isinstance(parser, argparse.ArgumentParser)
	return parser.format_help()


def _main_function(script: Path) -> ast.FunctionDef | None:
	tree = ast.parse(script.read_text(encoding='utf-8'), filename=str(script))
	for node in tree.body:
		if isinstance(node, ast.FunctionDef) and node.name == 'main':
			return node
	return None
