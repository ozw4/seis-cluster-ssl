"""Pair the 540 fixed HMM conditions with matched no-HMM decoder results."""

# ruff: noqa: RUF001 - Japanese report prose uses native punctuation.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from seis_ssl_cluster.clustering.features import file_sha256

CONFIG = Path(
	'experiments/hmm_v2/missing_controls_completion_v1/no_hmm_comparison.yaml'
)
FILES = frozenset(
	{
		'factorial_cells.csv',
		'factorial_comparison.csv',
		'no_hmm_cells.csv',
		'provenance.json',
		'README.md',
	}
)
METRICS = {
	'f3': ('mean_iou', 'fraction', 'higher', 'macro_f1', 'fraction'),
	'parihaka': ('channel_iou', 'fraction', 'higher', 'channel_f1', 'fraction'),
	'volve': (
		'macro_mae_samples',
		'samples',
		'lower',
		'macro_within_2_samples',
		'fraction',
	),
}
GROUP_KEYS = (
	'survey',
	'source',
	'data_size',
	'distillation_weight',
	'consistency_weight',
	'evaluation_split',
)
BASE_KEYS = ('survey', 'source', 'layout_id', 'data_size')


def rows(path: Path) -> list[dict[str, str]]:
	"""Read the original precision of exported metric values."""
	with path.open(encoding='utf-8-sig', newline='') as stream:
		return list(csv.DictReader(stream))


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
	"""Write an explicit comparison schema with spreadsheet-compatible encoding."""
	with path.open('w', encoding='utf-8-sig', newline='') as stream:
		writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
		writer.writeheader()
		writer.writerows(values)


def digest(value: object) -> str:
	"""Compare full scientific identities without exporting large coordinate lists."""
	return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluation(path: Path, survey: str, *, expected_hmm: bool) -> dict[str, Any]:
	"""Check actual split, seed, HMM status and decoder/support identity."""
	data = json.loads(path.read_text())
	if survey == 'f3':
		evidence = json.loads(path.with_name('evidence.json').read_text())
		config = json.loads(
			Path(evidence['resolved_decoder_config']['path']).read_text()
		)
		seed = config['train']['seed']
		has_hmm = evidence['condition']['version'] != 'none'
		if data['split'] != 'test' or data['evaluation_voxel_count'] != 113979624:
			raise ValueError(f'F3 common-test support mismatch: {path}')
		shared = {k: config[k] for k in ('dataset', 'decoder', 'tiles', 'train')}
		shared.update(
			test_mask=evidence['test_mask_sha256'],
			grid=evidence['supervision_grid']['sha256'],
		)
		scores = data
		model = config['model']['tag']
	else:
		identity = data['benchmark_identity']
		seed = identity['training']['seed']
		model = data['model']
		model_source = identity['embedding'].get('model_source', {})
		has_hmm = any(
			bool(model_source.get(k))
			for k in ('stratigraphy_pretext', 'multi_head', 'hmm_k', 'pseudo_targets')
		)
		# Volve's benchmark name is descriptive; all scientific identity fields stay.
		ignored = {'model', 'embedding'} | (
			{'benchmark'} if survey == 'volve' else set()
		)
		shared = {k: v for k, v in identity.items() if k not in ignored}
		scores = data['test']['primary_common'] if survey == 'volve' else data['test']
	if seed != 42000 or has_hmm != expected_hmm:
		raise ValueError(f'decoder seed or HMM source mismatch: {path}')
	metric, _, _, secondary, _ = METRICS[survey]
	primary, secondary_value = float(scores[metric]), float(scores[secondary])
	if not all(math.isfinite(v) for v in (primary, secondary_value)):
		raise ValueError(f'nonfinite evaluation: {path}')
	return {
		'model': model,
		'decoder_seed': seed,
		'primary': primary,
		'secondary': secondary_value,
		'shared_identity': digest(shared),
	}


def pair(
	cells: list[dict[str, Any]], baselines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
	"""Use each fixed baseline four times; orient improvements toward better results."""
	lookup = {tuple(r[k] for k in BASE_KEYS): r for r in baselines}
	if len(lookup) != len(baselines):
		raise ValueError('duplicate no-HMM baseline')
	paired = []
	for c in cells:
		key = tuple(c[k] for k in BASE_KEYS)
		if key not in lookup:
			raise ValueError(f'missing no-HMM baseline: {key}')
		b = lookup[key]
		if c['decoder_seed'] != b['decoder_seed'] or b['decoder_seed'] != 42000:
			raise ValueError(f'paired decoder seed mismatch: {key}')
		if (
			c['evaluation_split'] != b['evaluation_split']
			or c['shared_identity'] != b['shared_identity']
		):
			raise ValueError(f'paired evaluation or decoder identity mismatch: {key}')
		metric, unit, direction, secondary, second_unit = METRICS[c['survey']]
		with_primary, without_primary = float(c['primary']), float(b['primary'])
		with_secondary, without_secondary = float(c['secondary']), float(b['secondary'])
		paired.append(
			{
				**{
					k: c[k]
					for k in (
						'survey',
						'source',
						'data_size',
						'layout_id',
						'distillation_weight',
						'consistency_weight',
						'evaluation_split',
					)
				},
				'primary_metric': metric,
				'primary_unit': unit,
				'primary_direction': direction,
				'without_hmm_primary': without_primary,
				'with_hmm_primary': with_primary,
				'primary_improvement': (without_primary - with_primary)
				if direction == 'lower'
				else (with_primary - without_primary),
				'secondary_metric': secondary,
				'secondary_unit': second_unit,
				'without_hmm_secondary': without_secondary,
				'with_hmm_secondary': with_secondary,
				'secondary_improvement': with_secondary - without_secondary,
				'decoder_seed': b['decoder_seed'],
				'without_hmm_model': b['model'],
				'without_hmm_metrics_file': b['metrics_file'],
				'with_hmm_metrics_file': c['metrics_file'],
				'shared_identity_sha256': c['shared_identity'],
			}
		)
	return paired


def summarize(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Compute matched five-layout means and SDs of the individual improvements."""
	groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
	for r in paired:
		groups[tuple(r[k] for k in GROUP_KEYS)].append(r)
	out = []
	for _, group in sorted(groups.items()):
		if len(group) != 5 or len({r['layout_id'] for r in group}) != 5:
			raise ValueError('each aggregate requires five distinct paired layouts')
		first = group[0]
		r = {k: first[k] for k in GROUP_KEYS}
		r.update(
			{
				k: first[k]
				for k in (
					'primary_metric',
					'primary_unit',
					'primary_direction',
					'secondary_metric',
					'secondary_unit',
				)
			}
		)
		r['n'] = 5
		for name in (
			'without_hmm_primary',
			'with_hmm_primary',
			'primary_improvement',
			'without_hmm_secondary',
			'with_hmm_secondary',
			'secondary_improvement',
		):
			values = [float(c[name]) for c in group]
			r[name + '_mean'], r[name + '_sample_sd'] = (
				statistics.mean(values),
				statistics.stdev(values),
			)
		r['hmm_wins'] = sum(c['primary_improvement'] > 1e-12 for c in group)
		r['hmm_ties'] = sum(abs(c['primary_improvement']) <= 1e-12 for c in group)
		r['hmm_losses'] = 5 - r['hmm_wins'] - r['hmm_ties']
		out.append(r)
	return out


def collect(  # noqa: C901, PLR0912, PLR0915
	config: dict[str, Any], artifact: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
	"""Read only artifact originals and retain file hashes for every comparison."""
	source = artifact / config['source_cells']['path']
	if file_sha256(source) != config['source_cells']['sha256']:
		raise ValueError('original 540-cell input changed')
	cells = rows(source)
	known_paths = {}
	for weights, ref in config['hmm_source_summaries'].items():
		path = artifact / ref['path']
		if file_sha256(path) != ref['sha256']:
			raise ValueError('HMM source summary identity changed')
		for c in json.loads(path.read_text())['cells']:
			known_paths[
				c['survey'], c['source_family'], weights, c['layout_id'], c['data_size']
			] = Path(c['metrics_path'])
	f3_index = {}
	for namespace in config['f3_test_namespaces']:
		folder = artifact / 'surveys/f3/facies_benchmark_v2/evaluation' / namespace
		for r in rows(folder / 'summary/per_layout.csv'):
			if r['version'] != 'none' and (
				r['version'] != 'v2' or r['head_ks'] != '6|8|10'
			):
				continue
			key = (
				r['version'],
				r['source'],
				str(float(r['distillation_weight']))
				if r['distillation_weight']
				else '',
				str(float(r['consistency_weight'])) if r['consistency_weight'] else '',
				r['layout_id'],
				r['data_size'],
			)
			f3_index[key] = (
				folder
				/ 'cells'
				/ r['condition_id']
				/ r['layout_id']
				/ r['data_size']
				/ 'metrics.json'
			)
	descriptors = {(r['survey'], r['source']): r for r in config['baselines']}
	baselines, manifest = [], []

	def read(path: Path, survey: str, *, hmm: bool) -> dict[str, Any]:
		if (
			not path.resolve().is_relative_to(artifact.resolve())
			or 'reports' in path.resolve().parts
		):
			raise ValueError('all numerical sources must be artifact originals')
		r = evaluation(path, survey, expected_hmm=hmm)
		r['metrics_file'] = str(path.relative_to(artifact))
		manifest.append({'path': r['metrics_file'], 'sha256': file_sha256(path)})
		return r

	for survey in METRICS:
		for source_name in ('mae', 'local_bt', 'random'):
			d = descriptors[survey, source_name]
			for layout in (f'layout_{i:03d}' for i in range(5)):
				for size in ('small', 'medium', 'large'):
					if survey == 'f3':
						p = f3_index['none', source_name, '', '', layout, size]
					else:
						p = artifact / d['metrics_template'].format(
							layout=layout, size=size
						)
					r = read(p, survey, hmm=False)
					if r['model'] != d['model']:
						raise ValueError('no-HMM model identity changed')
					baselines.append(
						dict(
							r,
							survey=survey,
							source=source_name,
							layout_id=layout,
							data_size=size,
							evaluation_split='common_test'
							if survey == 'f3'
							else 'test',
						)
					)
	prefixes = {
		'mae': 'mae100',
		'local_bt': 'local_bt_rot90_asym_g060_3ep',
		'random': 'random_init',
	}
	for c in cells:
		survey, source_name = c['survey'], c['source']
		d, con = c['distillation_weight'], c['consistency_weight']
		if survey == 'f3':
			p = f3_index['v2', source_name, d, con, c['layout_id'], c['data_size']]
		elif d + '|' + con in config['hmm_source_summaries']:
			p = known_paths[
				survey, source_name, d + '|' + con, c['layout_id'], c['data_size']
			]
		else:
			cohort = config['hmm_cohorts'][d + '|' + con]
			if isinstance(cohort, dict):
				cohort = cohort[source_name]
			model = (
				prefixes[source_name]
				+ f'_hmm_v2_mh_k6810_distill{int(float(d) * 100):03d}'
				+ f'_cons{int(float(con) * 100):03d}'
			)
			if d == '0.2' and con == '0.0':
				model = model.removesuffix('_cons000')
			series = (
				'facies_benchmark_v1'
				if survey == 'parihaka'
				else 'horizon_benchmark_v1'
			)
			p = (
				artifact
				/ f'surveys/{survey}/{series}/benchmark/{cohort}/runs'
				/ f'model={model}/layout={c["layout_id"]}'
				/ f'size={c["data_size"]}/metrics.json'
			)
		r = read(p, survey, hmm=True)
		if any(float(c[k]) != r[k] for k in ('primary', 'secondary')):
			raise ValueError(f'original HMM score differs from source: {p}')
		c.update(r)
	if len(cells) != 540 or len(baselines) != 135:
		raise ValueError('expected 540 HMM cells and 135 no-HMM baselines')
	keys = [
		tuple(c[k] for k in (*BASE_KEYS, 'distillation_weight', 'consistency_weight'))
		for c in cells
	]
	if len(set(keys)) != 540:
		raise ValueError('duplicate HMM condition')
	return cells, baselines, manifest


def export(config_path: Path, artifact: Path, output: Path) -> None:
	"""Publish an exact immutable derived bundle; preserve the original score table."""
	config = yaml.safe_load(config_path.read_text())
	cells, baselines, manifest = collect(config, artifact)
	paired = pair(cells, baselines)
	summary = summarize(paired)
	if len(summary) != 108:
		raise ValueError('expected 108 paired aggregate conditions')
	if output.exists():
		raise FileExistsError(output)
	output.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix='.paired-', dir=output.parent) as temporary:
		stage = Path(temporary)
		write_csv(stage / 'factorial_cells.csv', paired)
		write_csv(stage / 'factorial_comparison.csv', summary)
		write_csv(stage / 'no_hmm_cells.csv', baselines)
		(stage / 'provenance.json').write_text(
			json.dumps(
				{
					'config_sha256': file_sha256(config_path),
					'original_hmm_cells': config['source_cells'],
					'hmm_cells': 540,
					'unique_no_hmm_cells': 135,
					'baseline_reuse_per_cell': 4,
					'source_files': manifest,
					'published_files': sorted(FILES),
				},
				indent=2,
			)
			+ '\n'
		)
		(stage / 'README.md').write_text(
			'# HMM vs no HMM\n\n540 paired rows, 135 unique no-HMM baselines, '
			'108 five-layout summaries.\n'
			'All decoder seeds are 42000; F3 uses common test, others test.\n'
			'Each baseline is reused for four HMM loss settings.\n'
			'Positive primary_improvement favors HMM: HMM minus baseline for IoU, '
			'baseline minus HMM for Volve MAE [samples]. '
			'Secondary improvement is HMM minus baseline.\n'
			'SDs use paired differences (ddof=1); layouts are not encoder seeds.\n'
			'Comparisons use existing no-HMM benchmarks '
			'and their pretraining budgets.\n'
		)
		if {p.name for p in stage.iterdir()} != FILES:
			raise ValueError('unexpected paired publication files')
		stage.rename(output)


def update_review(summary: Path, paired: Path, artifact: Path, output: Path) -> None:
	"""Rebuild the requested tracked review from original and derived artifacts."""
	from seis_ssl_cluster.hmm.missing_controls import (  # noqa: PLC0415
		PUBLISHED_FILES,
		REVIEW_FILES,
	)

	for source, expected in ((summary, PUBLISHED_FILES), (paired, FILES)):
		if not source.resolve().is_relative_to(artifact.resolve()):
			raise ValueError('review numerical inputs must remain in artifacts')
		if {p.name for p in source.iterdir()} != expected:
			raise ValueError('incomplete source publication')
	if output.exists() and {p.name for p in output.iterdir()} != REVIEW_FILES:
		raise ValueError('refusing to replace unowned review files')
	metadata = json.loads((paired / 'provenance.json').read_text())
	if metadata['hmm_cells'] != 540 or metadata['unique_no_hmm_cells'] != 135:
		raise ValueError('incomplete baseline pairing')
	text = (
		'# HMMあり・なしの性能比較（540条件）\n\n'
		'同じデータセット・初期表現・レイアウト・ラベル量・評価分割・decoder seed '
		'42000で対応づけています。F3は共通test、Parihaka/Volveはtestです。\n\n'
		'- [540行の対応比較](factorial_cells.csv)\n'
		'- [108条件の平均・標準偏差・勝敗数](factorial_comparison.csv)\n'
		'- [HMM損失設定間の効果・交互作用](factorial_contrasts.csv)\n'
		'- [Barlow蒸留比較](barlow_comparison.csv)\n'
		'- [v1蒸留比較](v1_comparison.csv)\n\n'
		'`without_hmm_primary`がHMMなし、`with_hmm_primary`がHMMあり、'
		'`primary_improvement`が改善量です。正ならHMMで改善、負なら悪化です。'
		'F3/Parihakaでは「あり−なし」、Volve MAEでは「なし−あり」としています。'
		'指標名・単位は`primary_metric`・`primary_unit`列で確認できます。\n\n'
		'副指標も同じ列構成です。改善量のSDは対応する5レイアウトの差から求めた'
		'標本SD（ddof=1）で、個別SDの差ではありません。108行表にはHMMの勝・分・敗も示します。\n\n'
		'135個のHMMなし評価を各4損失設定に再利用しています。MAEは既存の再構成継続モデル、'
		'Barlow Twinsは既存の3 epochモデル、Randomは未学習モデルです。学習予算を完全に'
		'揃えた因果実験ではなく、既存ベンチマークとの性能比較です。Parihaka MAEの図用'
		'seed 42001は含めていません。\n\n'
		'継続学習seedは42です。5レイアウトはencoder seedの5反復ではなく、'
		'ラベル量は包含関係にあります。F3は同一地域内の'
		'transductiveなラベルholdoutです。\n\n'
		'[実験記録]'
		'(../../experiments/hmm_v2/missing_controls_completion_v1/README.md) · '
		'[出典hash](provenance.json)。公開ファイルは7点です。\n'
	)
	output.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(
		prefix='.baseline-review-', dir=output.parent
	) as tmp:
		stage = Path(tmp)
		copied = REVIEW_FILES - {'README.md', 'provenance.json'}
		for name in copied:
			source = (
				paired
				if name in ('factorial_cells.csv', 'factorial_comparison.csv')
				else summary
			)
			(stage / name).write_bytes((source / name).read_bytes())
		(stage / 'README.md').write_text(text)
		(stage / 'provenance.json').write_text(
			json.dumps(
				{
					'original_hmm_summary': str(summary.relative_to(artifact)),
					'original_hmm_summary_sha256': file_sha256(
						summary / 'summary.json'
					),
					'paired_comparison': str(paired.relative_to(artifact)),
					'paired_provenance_sha256': file_sha256(paired / 'provenance.json'),
					'hmm_cells': 540,
					'unique_no_hmm_cells': 135,
					'decoder_seed': 42000,
					'source_file_sha256': {
						name: file_sha256(stage / name) for name in sorted(copied)
					},
					'published_files': sorted(REVIEW_FILES),
				},
				ensure_ascii=False,
				indent=2,
			)
			+ '\n'
		)
		if {p.name for p in stage.iterdir()} != REVIEW_FILES:
			raise ValueError('unexpected review file set')
		output.mkdir(exist_ok=True)
		for name in [*sorted(REVIEW_FILES - {'provenance.json'}), 'provenance.json']:
			(stage / name).replace(output / name)


def main() -> None:
	"""Generate the comparison from frozen artifact evaluations."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, default=CONFIG)
	parser.add_argument(
		'--artifact-root', type=Path, default=Path('artifacts/seis_ssl_cluster')
	)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--reuse-existing', action='store_true')
	parser.add_argument('--review-output', type=Path)
	args = parser.parse_args()
	artifact = args.artifact_root.resolve()
	output = args.output.resolve()
	if args.reuse_existing:
		metadata = json.loads((output / 'provenance.json').read_text())
		if metadata['config_sha256'] != file_sha256(args.config):
			raise ValueError('existing baseline comparison config changed')
	else:
		export(args.config, artifact, output)
	if args.review_output:
		update_review(
			artifact / 'studies/hmm_v2/missing_controls_completion_v1/summary',
			output,
			artifact,
			args.review_output,
		)


if __name__ == '__main__':
	main()
