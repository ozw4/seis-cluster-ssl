"""Build and verify the frozen HMM_v1 result snapshot without running training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

VERSION = 'HMM_v1'
SNAPSHOT = '2026-09-10T06:52:00Z'
CONDITION_IDS = ('1', '2a', '2b', '3', '4a', '4b', '5', '6a', '6b')
DATASETS = ('f3', 'parihaka', 'volve')
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
DATA_SIZES = ('small', 'medium', 'large')
EXPECTED_CELLS = len(CONDITION_IDS) * len(DATASETS) * len(LAYOUT_IDS) * len(DATA_SIZES)
REPORT_COLUMNS = (
	'condition_id',
	'dataset',
	'task',
	'layout_id',
	'data_size',
	'primary_metric',
	'primary_value',
	'secondary_metric',
	'secondary_value',
)
PARENTS = {'2a': '1', '2b': '1', '4a': '3', '4b': '3', '6a': '5', '6b': '5'}
TASKS = {
	'f3': {
		'task': 'lithology facies classification',
		'primary_metric': 'mean_iou_unique_validation_voxels',
		'secondary_metric': 'macro_f1_unique_validation_voxels',
		'direction': 'higher',
	},
	'parihaka': {
		'task': 'channel facies classification',
		'primary_metric': 'test_channel_iou',
		'secondary_metric': 'test_channel_f1',
		'direction': 'higher',
	},
	'volve': {
		'task': 'five-horizon estimation',
		'primary_metric': 'test_macro_mae_samples',
		'secondary_metric': 'test_macro_within_2_samples',
		'direction': 'lower',
	},
}
EXPECTED_OFFICIAL = {
	'f3': {
		'mean': 0.5825621911146475,
		'population_sd': 0.0693383711027919,
		'secondary_mean': 0.6980941414072311,
		'improvement': 0.029489222757560093,
		'better_cells': 14,
	},
	'parihaka': {
		'mean': 0.3646935461676095,
		'population_sd': 0.08905391603779594,
		'secondary_mean': 0.5279281360026884,
		'improvement': 0.0208786064586242,
		'better_cells': 12,
	},
	'volve': {
		'mean': 8.502418936574871,
		'population_sd': 2.653019226153859,
		'secondary_mean': 0.4644217230904164,
		'improvement': 0.4392308582781708,
		'better_cells': 11,
	},
}


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline='', encoding='utf-8') as stream:
		return list(csv.DictReader(stream))


def _read_json(path: Path) -> Mapping[str, Any]:
	with path.open(encoding='utf-8') as stream:
		value = json.load(stream)
	if not isinstance(value, Mapping):
		raise TypeError(f'expected a JSON object: {path}')
	return value


def _artifact_path(recorded: str, artifact_root: Path) -> Path:
	marker = '/artifacts/seis_ssl_cluster/'
	if marker in recorded:
		return artifact_root / recorded.split(marker, maxsplit=1)[1]
	path = Path(recorded)
	return path if path.is_absolute() else artifact_root / path


def _cell(  # noqa: PLR0913
	*,
	condition_id: str,
	dataset: str,
	layout_id: str,
	data_size: str,
	primary_value: float,
	secondary_value: float,
) -> dict[str, str]:
	task = TASKS[dataset]
	return {
		'condition_id': condition_id,
		'dataset': dataset,
		'task': str(task['task']),
		'layout_id': layout_id,
		'data_size': data_size,
		'primary_metric': str(task['primary_metric']),
		'primary_value': format(primary_value, '.17g'),
		'secondary_metric': str(task['secondary_metric']),
		'secondary_value': format(secondary_value, '.17g'),
	}


def _selected_rows(path: Path, model_id: str) -> list[dict[str, str]]:
	rows = [row for row in _read_csv(path) if row.get('model_id') == model_id]
	if len(rows) != 15:
		raise ValueError(
			f'{path}: {model_id} must have exactly 15 cells; got {len(rows)}'
		)
	return rows


def _f3_cells(artifact_root: Path) -> list[dict[str, str]]:
	root = artifact_root / 'f3_lithology_benchmark'
	baseline = root / 'mae_local_bt_five_way_v3/summary/comparison.csv'
	specs = {
		'1': (baseline, 'mae'),
		'2a': (
			root / 'pretraining_comparison_completion_v1/summary_five_way/'
			'mae100_hmm_k6_distill010/comparison.csv',
			'mae100_hmm_k6_distill010',
		),
		'2b': (baseline, 'mae_hmm_k6'),
		'4a': (
			root / 'pretraining_comparison_completion_v1/summary_five_way/'
			'local_bt_rot90_asym_g060_3ep_hmm_k6_distill010/comparison.csv',
			'local_bt_rot90_asym_g060_3ep_hmm_k6_distill010',
		),
		'5': (baseline, 'random'),
		'6a': (
			root / 'random_init_self_target_hmm_k6_distill010_v1/'
			'summary_five_way/comparison.csv',
			'random_init_self_target_hmm_k6_distill010',
		),
		'6b': (
			root / 'pretraining_comparison_completion_v1/summary_five_way/'
			'random_self_target_hmm_k6_distill020/comparison.csv',
			'random_self_target_hmm_k6_distill020',
		),
	}
	cells: list[dict[str, str]] = []
	for condition_id, (source, model_id) in specs.items():
		for row in _selected_rows(source, model_id):
			metrics = _read_json(_artifact_path(row['metrics_path'], artifact_root))
			if not math.isclose(float(row['mean_iou']), float(metrics['mean_iou'])):
				raise ValueError(f'{source}: stale mean_iou for {model_id}')
			cells.append(
				_cell(
					condition_id=condition_id,
					dataset='f3',
					layout_id=row['layout_id'],
					data_size=row['data_size'],
					primary_value=float(metrics['mean_iou']),
					secondary_value=float(metrics['macro_f1']),
				)
			)
	paired = (
		root / 'local_bt_noise_rotation_search_v1/paired_summary/'
		'local_bt_nr_rot90_asym_g060_3ep_hmm_k6_25ep_vs_'
		'local_bt_nr_rot90_asym_g060_3ep/comparison.csv'
	)
	for row in _read_csv(paired):
		for condition_id, role in (('3', 'control'), ('4b', 'candidate')):
			metrics = _read_json(
				_artifact_path(row[f'{role}_metrics_path'], artifact_root)
			)
			if not math.isclose(
				float(row[f'{role}_macro_f1']),
				float(metrics['macro_f1']),
				rel_tol=0.0,
				abs_tol=5e-10,
			):
				raise ValueError(f'{paired}: stale {role} macro_f1')
			cells.append(
				_cell(
					condition_id=condition_id,
					dataset='f3',
					layout_id=row['layout_id'],
					data_size=row['data_size'],
					primary_value=float(metrics['mean_iou']),
					secondary_value=float(metrics['macro_f1']),
				)
			)
	return cells


def _parihaka_cells(artifact_root: Path) -> list[dict[str, str]]:
	comparison = (
		artifact_root / 'channel_benchmark/pretraining_comparison_completion_v1/'
		'summary/comparison.csv'
	)
	comparison_rows = {
		(row['arm_id'], row['layout_id'], row['data_size']): row
		for row in _read_csv(comparison)
	}
	specs = {
		'1': ('mae', 'channel_benchmark/ssl_hmm_four_way_v1/runs'),
		'2a': (
			'mae_hmm_k6_distill010',
			'channel_benchmark/pretraining_comparison_completion_v1/runs',
		),
		'2b': ('mae_hmm_k6', 'channel_benchmark/ssl_hmm_four_way_v1/runs'),
		'3': (
			'local_barlow_twins_rot90_asym_g060_3ep',
			'channel_benchmark/ssl_hmm_four_way_v1/runs',
		),
		'4a': (
			'local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill010',
			'channel_benchmark/pretraining_comparison_completion_v1/runs',
		),
		'4b': (
			'local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill020',
			'channel_benchmark/pretraining_comparison_completion_v1/runs',
		),
		'5': ('random', 'channel_benchmark/runs'),
		'6a': (
			'random_init_self_target_hmm_k6_distill010',
			'channel_benchmark/pretraining_comparison_completion_v1/runs',
		),
		'6b': (
			'random_init_self_target_hmm_k6_distill020',
			'channel_benchmark/pretraining_comparison_completion_v1/runs',
		),
	}
	cells: list[dict[str, str]] = []
	for condition_id, (model_id, runs_root) in specs.items():
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				metrics_path = (
					artifact_root
					/ runs_root
					/ f'model={model_id}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'metrics.json'
				)
				test = _read_json(metrics_path)['test']
				if not isinstance(test, Mapping):
					raise TypeError(f'{metrics_path}: test must be an object')
				recorded = comparison_rows[(model_id, layout_id, data_size)]
				if not math.isclose(
					float(recorded['test_channel_iou']),
					float(test['channel_iou']),
				):
					raise ValueError(f'{comparison}: stale Channel IoU for {model_id}')
				cells.append(
					_cell(
						condition_id=condition_id,
						dataset='parihaka',
						layout_id=layout_id,
						data_size=data_size,
						primary_value=float(test['channel_iou']),
						secondary_value=float(test['channel_f1']),
					)
				)
	return cells


def _volve_cells(artifact_root: Path) -> list[dict[str, str]]:
	root = artifact_root / 'horizon/volve/horizon_benchmark_v1'
	base_summary = root / 'mae_local_bt_hmm_five_way_v1/summary/comparison.csv'
	distill_summary = (
		root / 'mae_local_bt_hmm_five_way_distill010_v1/'
		'summary/macro_mae/comparison.csv'
	)
	specs = {
		'1': (root / 'mae_local_bt_hmm_five_way_v1/runs', 'mae', base_summary),
		'2a': (
			root / 'mae_local_bt_hmm_five_way_distill010_v1/runs',
			'mae_hmm_k6',
			distill_summary,
		),
		'2b': (
			root / 'mae_local_bt_hmm_five_way_v1/runs',
			'mae_hmm_k6',
			base_summary,
		),
		'3': (
			root / 'local_bt_recipe_arm_v1/runs',
			'rot90_asym_g060_3ep',
			root / 'local_bt_recipe_arm_v1/summary/rot90_asym_g060_3ep/per_cell.csv',
		),
		'4a': (
			root / 'local_bt_recipe_arm_v1/runs',
			'rot90_asym_g060_3ep_hmm_k6_distill010',
			root / 'missing_hmm_comparison_v1/summary/'
			'rot90_asym_g060_3ep_hmm_k6_distill010/per_cell.csv',
		),
		'4b': (
			root / 'local_bt_recipe_arm_v1/runs',
			'rot90_asym_g060_3ep_hmm_k6_distill020',
			root / 'missing_hmm_comparison_v1/summary/'
			'rot90_asym_g060_3ep_hmm_k6_distill020/per_cell.csv',
		),
		'5': (
			root / 'local_bt_recipe_arm_v1/runs',
			'random',
			root / 'local_bt_recipe_arm_v1/summary/rot90_asym_g060_3ep/per_cell.csv',
		),
		'6a': (
			root / 'local_bt_recipe_arm_v1/runs',
			'random_init_hmm_k6_distill010',
			root / 'missing_hmm_comparison_v1/summary/'
			'random_init_hmm_k6_distill010/per_cell.csv',
		),
		'6b': (
			root / 'local_bt_recipe_arm_v1/runs',
			'random_init_hmm_k6_distill020',
			root / 'missing_hmm_comparison_v1/summary/'
			'random_init_hmm_k6_distill020/per_cell.csv',
		),
	}
	cells: list[dict[str, str]] = []
	for condition_id, (runs_root, model_id, source) in specs.items():
		source_rows = _read_csv(source)
		if source_rows and 'model_id' in source_rows[0]:
			source_rows = [row for row in source_rows if row['model_id'] == model_id]
		rows = {(row['layout_id'], row['data_size']): row for row in source_rows}
		for layout_id in LAYOUT_IDS:
			for data_size in DATA_SIZES:
				metrics_path = (
					runs_root
					/ f'model={model_id}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'metrics.json'
				)
				test = _read_json(metrics_path)['test']
				if not isinstance(test, Mapping):
					raise TypeError(f'{metrics_path}: test must be an object')
				primary = test['primary_common']
				if not isinstance(primary, Mapping):
					raise TypeError(f'{metrics_path}: primary_common must be an object')
				recorded = rows[(layout_id, data_size)]
				field = (
					'random_macro_mae_samples'
					if condition_id == '5'
					else (
						'macro_mae_samples'
						if 'macro_mae_samples' in recorded
						else 'arm_macro_mae_samples'
					)
				)
				if not math.isclose(
					float(recorded[field]),
					float(primary['macro_mae_samples']),
					rel_tol=0.0,
					abs_tol=5e-10,
				):
					raise ValueError(f'{source}: stale macro MAE for {model_id}')
				cells.append(
					_cell(
						condition_id=condition_id,
						dataset='volve',
						layout_id=layout_id,
						data_size=data_size,
						primary_value=float(primary['macro_mae_samples']),
						secondary_value=float(primary['macro_within_2_samples']),
					)
				)
	return cells


def collect_source_cells(artifact_root: Path) -> list[dict[str, str]]:
	"""Load the 405 selected cells from existing, completed result artifacts."""
	cells = [
		*_f3_cells(artifact_root),
		*_parihaka_cells(artifact_root),
		*_volve_cells(artifact_root),
	]
	return sorted(cells, key=_cell_sort_key)


def _cell_sort_key(row: Mapping[str, str]) -> tuple[int, int, int, int]:
	return (
		DATASETS.index(row['dataset']),
		CONDITION_IDS.index(row['condition_id']),
		LAYOUT_IDS.index(row['layout_id']),
		DATA_SIZES.index(row['data_size']),
	)


def _identity(row: Mapping[str, str]) -> tuple[str, str, str, str]:
	return (
		row['dataset'],
		row['condition_id'],
		row['layout_id'],
		row['data_size'],
	)


def _summarize(cells: Sequence[Mapping[str, str]]) -> dict[str, Any]:
	identities = [_identity(row) for row in cells]
	duplicate_count = len(identities) - len(set(identities))
	expected = {
		(dataset, condition_id, layout_id, data_size)
		for dataset in DATASETS
		for condition_id in CONDITION_IDS
		for layout_id in LAYOUT_IDS
		for data_size in DATA_SIZES
	}
	missing = sorted(expected - set(identities))
	non_finite = sum(not math.isfinite(float(row['primary_value'])) for row in cells)
	if missing or duplicate_count or non_finite or len(cells) != EXPECTED_CELLS:
		raise ValueError(
			'incomplete cells: '
			f'loaded={len(cells)} missing={len(missing)} '
			f'duplicate={duplicate_count} non_finite={non_finite}'
		)
	result: dict[str, Any] = {
		'schema_version': 1,
		'version': VERSION,
		'experiment_snapshot': SNAPSHOT,
		'completeness': {
			'expected_cells': EXPECTED_CELLS,
			'loaded_cells': len(cells),
			'missing_cells': len(missing),
			'duplicate_cell_identities': duplicate_count,
			'unreadable_results': 0,
			'non_finite_selected_metrics': non_finite,
			'cells_per_condition_per_dataset': 15,
		},
		'datasets': {},
	}
	for dataset in DATASETS:
		task = TASKS[dataset]
		conditions: dict[str, Any] = {}
		for condition_id in CONDITION_IDS:
			selected = [
				row
				for row in cells
				if row['dataset'] == dataset and row['condition_id'] == condition_id
			]
			values = [float(row['primary_value']) for row in selected]
			entry: dict[str, Any] = {
				'mean': statistics.fmean(values),
				'population_sd': statistics.pstdev(values),
				'secondary_mean': statistics.fmean(
					float(row['secondary_value']) for row in selected
				),
				'cells': len(selected),
			}
			if condition_id in PARENTS:
				parent_id = PARENTS[condition_id]
				parent = {
					(row['layout_id'], row['data_size']): float(row['primary_value'])
					for row in cells
					if row['dataset'] == dataset and row['condition_id'] == parent_id
				}
				deltas = [
					(
						parent[(row['layout_id'], row['data_size'])]
						- float(row['primary_value'])
						if task['direction'] == 'lower'
						else float(row['primary_value'])
						- parent[(row['layout_id'], row['data_size'])]
					)
					for row in selected
				]
				entry['paired_vs_parent'] = {
					'parent_condition': parent_id,
					'improvement_mean': statistics.fmean(deltas),
					'better_cells': sum(delta > 0.0 for delta in deltas),
					'total_cells': len(deltas),
					'sign_convention': 'positive_is_improvement',
				}
			conditions[condition_id] = entry
		means = {condition: values['mean'] for condition, values in conditions.items()}
		best = (
			min(means, key=means.__getitem__)
			if task['direction'] == 'lower'
			else max(means, key=means.__getitem__)
		)
		result['datasets'][dataset] = {
			**task,
			'best_condition': best,
			'conditions': conditions,
		}
	_validate_official(result)
	return result


def _validate_official(summary: Mapping[str, Any]) -> None:
	for dataset, expected in EXPECTED_OFFICIAL.items():
		entry = summary['datasets'][dataset]
		if entry['best_condition'] != '2b':
			raise ValueError(f'{dataset}: expected Condition 2b to be best')
		condition = entry['conditions']['2b']
		paired = condition['paired_vs_parent']
		for field in ('mean', 'population_sd', 'secondary_mean'):
			if not math.isclose(condition[field], expected[field], abs_tol=1e-15):
				raise ValueError(f'{dataset}: official {field} changed')
		if not math.isclose(
			paired['improvement_mean'], expected['improvement'], abs_tol=1e-15
		):
			raise ValueError(f'{dataset}: official paired improvement changed')
		if paired['better_cells'] != expected['better_cells']:
			raise ValueError(f'{dataset}: official better-cell count changed')
	for dataset in DATASETS:
		conditions = summary['datasets'][dataset]['conditions']
		for condition_id in PARENTS:
			if conditions[condition_id]['paired_vs_parent']['improvement_mean'] <= 0.0:
				raise ValueError(f'{dataset}/{condition_id}: HMM mean did not improve')


def _csv_bytes(cells: Sequence[Mapping[str, str]]) -> bytes:
	stream = io.StringIO(newline='')
	writer = csv.DictWriter(stream, fieldnames=REPORT_COLUMNS, lineterminator='\n')
	writer.writeheader()
	writer.writerows(cells)
	return stream.getvalue().encode('utf-8')


def write_results(artifact_root: Path, output_dir: Path) -> tuple[Path, Path]:
	"""Materialize deterministic, review-sized results from existing artifacts."""
	cells_path = output_dir / 'cells.csv'
	summary_path = output_dir / 'summary.json'
	for path in (cells_path, summary_path):
		if path.exists():
			raise FileExistsError(f'refusing to overwrite frozen result: {path}')
	cells = collect_source_cells(artifact_root)
	summary = _summarize(cells)
	output_dir.mkdir(parents=True, exist_ok=True)
	cells_path.write_bytes(_csv_bytes(cells))
	summary_path.write_text(
		json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	return cells_path, summary_path


def _manifest(repo_root: Path) -> Mapping[str, Any]:
	return _read_json(repo_root / 'reports/hmm_v1/manifest.json')


def _listed_repository_paths(manifest: Mapping[str, Any]) -> Iterable[str]:
	for key in (
		'implementation_paths',
		'config_paths',
		'runbook_paths',
		'aggregation_script_paths',
	):
		yield from manifest[key]
	for entry in manifest['canonical_results']:
		yield entry['path']


def _verify_markdown_links(repo_root: Path, markdown_path: Path) -> None:
	for target in re.findall(r'\[[^]]+\]\(([^)]+)\)', markdown_path.read_text()):
		path_text = target.split('#', maxsplit=1)[0]
		if not path_text or '://' in path_text or path_text.startswith('mailto:'):
			continue
		if not (markdown_path.parent / path_text).resolve().exists():
			raise FileNotFoundError(f'{markdown_path}: broken link {target}')
	if not markdown_path.is_relative_to(repo_root):
		raise ValueError(f'markdown path is outside repository: {markdown_path}')


def verify_repository(repo_root: Path) -> dict[str, Any]:  # noqa: C901
	"""Verify committed freeze files, links, hashes, cells, and config boundary."""
	manifest = _manifest(repo_root)
	if manifest['version'] != VERSION or manifest['status'] != 'frozen':
		raise ValueError('manifest is not the frozen HMM_v1 record')
	for relative in _listed_repository_paths(manifest):
		path = repo_root / relative
		if not path.is_file():
			raise FileNotFoundError(path)
	for entry in manifest['canonical_results']:
		path = repo_root / entry['path']
		if (
			path.stat().st_size != entry['size_bytes']
			or _sha256(path) != entry['sha256']
		):
			raise ValueError(f'canonical result checksum mismatch: {path}')
	for entry in manifest['hmm_training_configs']:
		relative = entry['path']
		with (repo_root / relative).open(encoding='utf-8') as stream:
			config = yaml.safe_load(stream)
		if 'spec' in config['head'] or config['head']['num_prototypes'] != 6:
			raise ValueError(f'{relative}: HMM_v1 must remain single-head K=6')
		if config['pseudo_targets']['k'] != 6 or config['train']['epochs'] != 25:
			raise ValueError(f'{relative}: HMM_v1 K/epoch contract changed')
		if config['loss']['distillation_weight'] != entry['distillation_weight']:
			raise ValueError(f'{relative}: distillation weight changed')
	cells_path = repo_root / 'reports/hmm_v1/cells.csv'
	cells = _read_csv(cells_path)
	summary = _summarize(cells)
	committed = _read_json(repo_root / 'reports/hmm_v1/summary.json')
	if summary != committed:
		raise ValueError('summary.json does not match cells.csv')
	_verify_markdown_links(repo_root, repo_root / 'reports/hmm_v1/README.md')
	return {
		'status': 'ok',
		'cells': len(cells),
		'repository_paths': len(tuple(_listed_repository_paths(manifest))),
		'hmm_training_configs': len(manifest['hmm_training_configs']),
	}


def verify_live_artifacts(repo_root: Path, artifact_root: Path) -> dict[str, Any]:
	"""Compare live source summaries/checkpoints with the committed freeze."""
	manifest = _manifest(repo_root)
	for entry in manifest['canonical_source_aggregates']:
		path = artifact_root / entry['path']
		if (
			path.stat().st_size != entry['size_bytes']
			or _sha256(path) != entry['sha256']
		):
			raise ValueError(f'source aggregate checksum mismatch: {path}')
	for entry in manifest['checkpoints']:
		path = artifact_root / entry['path']
		if (
			path.stat().st_size != entry['size_bytes']
			or _sha256(path) != entry['sha256']
		):
			raise ValueError(f'checkpoint checksum mismatch: {path}')
	live = collect_source_cells(artifact_root)
	committed = _read_csv(repo_root / 'reports/hmm_v1/cells.csv')
	if _csv_bytes(live) != _csv_bytes(committed):
		raise ValueError('live source cells differ from frozen cells.csv')
	return {
		'status': 'ok',
		'cells': len(live),
		'source_aggregates': len(manifest['canonical_source_aggregates']),
		'checkpoints': len(manifest['checkpoints']),
	}


def main() -> None:
	"""Build or verify the HMM_v1 freeze snapshot."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--artifact-root', type=Path)
	parser.add_argument('--output-dir', type=Path, default=Path('reports/hmm_v1'))
	parser.add_argument('--write', action='store_true')
	parser.add_argument('--check-source', action='store_true')
	args = parser.parse_args()
	repo_root = Path(__file__).resolve().parents[1]
	if args.write:
		if args.artifact_root is None:
			parser.error('--artifact-root is required with --write')
		paths = write_results(args.artifact_root.resolve(), args.output_dir)
		print(json.dumps({'status': 'written', 'paths': [str(path) for path in paths]}))
		return
	result = verify_repository(repo_root)
	if args.check_source:
		artifact_root = args.artifact_root
		if artifact_root is None:
			value = os.environ.get('SEIS_SSL_CLUSTER_ARTIFACT_ROOT')
			if value is None:
				parser.error(
					'--artifact-root or SEIS_SSL_CLUSTER_ARTIFACT_ROOT is required'
				)
			artifact_root = Path(value)
		result['live_artifacts'] = verify_live_artifacts(repo_root, artifact_root)
	print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
