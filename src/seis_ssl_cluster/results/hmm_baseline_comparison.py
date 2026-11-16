"""Compare saved HMM versions with verified no-HMM decoder results."""

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
	'hmm_version',
	'head_ks',
	'with_hmm_model',
	'source_cohort',
	'evaluation_support',
	'checkpoint_selection',
	'source',
	'data_size',
	'distillation_weight',
	'consistency_weight',
	'evaluation_split',
	'with_hmm_decoder_seed',
	'without_hmm_decoder_seed',
	'comparison_type',
	'baseline_reference_status',
)
BASE_KEYS = ('survey', 'source', 'layout_id', 'data_size')
DETAIL_KEYS = (
	'hmm_version',
	'head_ks',
	'with_hmm_model',
	'with_hmm_record_id',
	'evaluation_support',
	'source_evaluation_split',
	'source_cohort',
	'checkpoint_selection',
	'archived',
	'saved_status',
)


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
		if not has_hmm and identity['embedding'].get('checkpoint_path'):
			# Some v1 zero-distillation exports record only a recipe-arm identity.
			# Verify the actual training schema rather than guessing from its path.
			training_path = (
				Path(identity['embedding']['checkpoint_path'])
				.resolve()
				.with_name('resolved_config.json')
			)
			if training_path.is_file():
				training = json.loads(training_path.read_text())
				has_hmm = training.get('stage') == 'train_strat_hmm_pretext' and bool(
					training.get('head')
				)
		# Volve's benchmark name and available checkpoint-policy list describe
		# the exporter. The actual selected checkpoint remains explicit per row.
		ignored = {'model', 'embedding'} | (
			{'benchmark', 'checkpoint_selections'} if survey == 'volve' else set()
		)
		shared = {k: v for k, v in identity.items() if k not in ignored}
		scores = data['test']['primary_common'] if survey == 'volve' else data['test']
	if has_hmm != expected_hmm:
		raise ValueError(f'HMM source mismatch: {path}')
	# A cross-seed comparison may differ only in decoder RNG and initial weights.
	comparable = json.loads(json.dumps(shared))
	for training_key in ('train', 'training'):
		comparable.get(training_key, {}).pop('seed', None)
	comparable.pop('decoder_initial_state_sha256', None)
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
		'comparison_identity': digest(comparable),
	}


def pair(
	cells: list[dict[str, Any]], baselines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
	"""Retain every baseline seed and explicitly identify cross-seed comparisons."""
	lookup: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
	seen = set()
	for baseline in baselines:
		key = tuple(baseline[k] for k in BASE_KEYS)
		seed_key = (*key, baseline['decoder_seed'])
		if seed_key in seen:
			raise ValueError('duplicate no-HMM baseline')
		seen.add(seed_key)
		lookup[key].append(baseline)
	paired = []
	for c in cells:
		key = tuple(c[k] for k in BASE_KEYS)
		if key not in lookup:
			raise ValueError(f'missing no-HMM baseline: {key}')
		for b in lookup[key]:
			same_seed = c['decoder_seed'] == b['decoder_seed']
			identity_key = 'shared_identity' if same_seed else 'comparison_identity'
			if (
				c['evaluation_split'] != b['evaluation_split']
				or not c.get(identity_key)
				or c[identity_key] != b.get(identity_key)
			):
				raise ValueError(
					f'paired evaluation or decoder identity mismatch: {key}'
				)
			metric, unit, direction, secondary, second_unit = METRICS[c['survey']]
			with_primary, without_primary = float(c['primary']), float(b['primary'])
			with_secondary, without_secondary = (
				float(c['secondary']),
				float(b['secondary']),
			)
			paired.append(
				{
					**{k: c.get(k, '') for k in DETAIL_KEYS},
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
					'comparison_status': 'paired',
					'comparison_note': '',
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
					'with_hmm_decoder_seed': c['decoder_seed'],
					'without_hmm_decoder_seed': b['decoder_seed'],
					'same_decoder_seed': same_seed,
					'comparison_type': 'same_decoder_seed'
					if same_seed
					else 'different_decoder_seed',
					'baseline_reference_status': b.get(
						'baseline_reference_status', 'canonical'
					),
					'without_hmm_model': b['model'],
					'without_hmm_metrics_file': b['metrics_file'],
					'with_hmm_metrics_file': c['metrics_file'],
					'with_hmm_identity_sha256': c['shared_identity'],
					'without_hmm_identity_sha256': b['shared_identity'],
					'comparison_identity_sha256': c.get(
						'comparison_identity', c['shared_identity']
					),
				}
			)
	return paired


def summarize(
	paired: list[dict[str, Any]], *, allow_partial: bool = False
) -> list[dict[str, Any]]:
	"""Compute matched five-layout means and SDs of the individual improvements."""
	groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
	for r in paired:
		if r.get('comparison_status') != 'paired':
			continue
		groups[tuple(r[k] for k in GROUP_KEYS)].append(r)
	out = []
	for _, group in sorted(groups.items()):
		distinct = len({r['layout_id'] for r in group}) == len(group)
		if not allow_partial and (len(group) != 5 or not distinct):
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
		r['n'] = len(group)
		r['aggregation_status'] = (
			'complete_five_layouts'
			if len(group) == 5 and distinct
			else 'partial_layouts'
			if distinct
			else 'duplicate_layouts_not_aggregated'
		)
		for name in (
			'without_hmm_primary',
			'with_hmm_primary',
			'primary_improvement',
			'without_hmm_secondary',
			'with_hmm_secondary',
			'secondary_improvement',
		):
			values = [float(c[name]) for c in group]
			r[name + '_mean'] = statistics.mean(values) if distinct else ''
			r[name + '_sample_sd'] = (
				statistics.stdev(values) if distinct and len(values) > 1 else ''
			)
		r['hmm_wins'] = sum(c['primary_improvement'] > 1e-12 for c in group)
		r['hmm_ties'] = sum(abs(c['primary_improvement']) <= 1e-12 for c in group)
		r['hmm_losses'] = len(group) - r['hmm_wins'] - r['hmm_ties']
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

	for d in config['baselines']:
		survey, source_name = d['survey'], d['source']
		for layout in (f'layout_{i:03d}' for i in range(5)):
			for size in d.get('data_sizes', ('small', 'medium', 'large')):
				if survey == 'f3':
					p = f3_index['none', source_name, '', '', layout, size]
				else:
					p = artifact / d['metrics_template'].format(
						layout=layout, size=size
					)
				r = read(p, survey, hmm=False)
				if r['model'] != d['model'] or r['decoder_seed'] != d.get(
					'decoder_seed', config['decoder_seed']
				):
					raise ValueError('no-HMM model or seed identity changed')
				baselines.append(
					dict(
						r,
						survey=survey,
						source=source_name,
						layout_id=layout,
						data_size=size,
						baseline_reference_status='historical'
						if size in d.get('historical_data_sizes', [])
						else 'canonical',
						evaluation_split='common_test' if survey == 'f3' else 'test',
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
		c.update(hmm_version='v2', head_ks='6|8|10', with_hmm_model=r['model'])
	if len(cells) != 540 or len(baselines) != config['expected_no_hmm_cells']:
		raise ValueError('incomplete HMM cells or no-HMM baselines')
	keys = [
		tuple(c[k] for k in (*BASE_KEYS, 'distillation_weight', 'consistency_weight'))
		for c in cells
	]
	if len(set(keys)) != 540:
		raise ValueError('duplicate HMM condition')
	return cells, baselines, manifest


def export(
	config_path: Path, artifact: Path, output: Path, *, overwrite: bool = False
) -> None:
	"""Stage the exact derived bundle and optionally replace its owned files."""
	config = yaml.safe_load(config_path.read_text())
	cells, baselines, manifest = collect(config, artifact)
	paired = pair(cells, baselines)
	summary = summarize(paired)
	if len(paired) != config.get('expected_comparisons', len(paired)) or len(
		summary
	) != config.get('expected_summaries', len(summary)):
		raise ValueError('incomplete paired comparisons')
	extra_provenance = {}
	if 'saved_catalog' in config:
		from seis_ssl_cluster.results.saved_hmm_comparison import (  # noqa: PLC0415
			expand_saved,
		)

		paired, extra_provenance = expand_saved(
			paired, baselines, artifact, config['saved_catalog']
		)
		manifest.extend(extra_provenance.pop('additional_sources'))
		summary = summarize(paired, allow_partial=True)
	if output.exists():
		if not overwrite:
			raise FileExistsError(output)
		if {p.name for p in output.iterdir()} != FILES:
			raise ValueError('refusing to replace unowned comparison files')
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
					'original_v2_cells': len(cells),
					'hmm_cells': extra_provenance.get('saved_hmm_records', len(cells)),
					**extra_provenance,
					'comparison_rows': len(paired),
					'aggregate_rows': len(summary),
					'unique_no_hmm_cells': len(baselines),
					'with_hmm_decoder_seeds': sorted(
						{
							int(c['with_hmm_decoder_seed'])
							for c in paired
							if c['with_hmm_decoder_seed'] != ''
						}
					),
					'without_hmm_decoder_seeds': sorted(
						{b['decoder_seed'] for b in baselines}
					),
					'original_v2_baseline_reuse_per_cell': 4,
					'source_files': manifest,
					'published_files': sorted(FILES),
				},
				indent=2,
			)
			+ '\n'
		)
		(stage / 'README.md').write_text(
			f'# HMM vs no HMM\n\n{len(paired)} paired rows, '
			f'{len(baselines)} unique no-HMM baselines, '
			f'{len(summary)} layout summaries.\n'
			'All saved HMM versions are retained when a saved_catalog is configured.\n'
			'Unmatched rows retain HMM results, with empty baseline/delta '
			'and an explicit reason.\n'
			'Both decoder seeds are explicit. '
			'Same-seed and cross-seed comparisons remain separate.\n'
			'Parihaka MAE large: 42001 is canonical, 42000 is retained as historical.\n'
			'F3 uses common test, others test.\n'
			'Check hmm_version, head_ks, model, split and support before comparisons.\n'
			'Positive primary_improvement favors HMM: HMM minus baseline for IoU, '
			'baseline minus HMM for Volve MAE [samples]. '
			'Secondary improvement is HMM minus baseline.\n'
			'SDs use paired differences (ddof=1); layouts are not encoder seeds.\n'
			'Comparisons use existing no-HMM benchmarks '
			'and their pretraining budgets.\n'
		)
		if {p.name for p in stage.iterdir()} != FILES:
			raise ValueError('unexpected paired publication files')
		output.mkdir(exist_ok=True)
		for name in [*sorted(FILES - {'provenance.json'}), 'provenance.json']:
			(stage / name).replace(output / name)


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
	if metadata['original_v2_cells'] != 540 or metadata['comparison_rows'] != len(
		rows(paired / 'factorial_cells.csv')
	):
		raise ValueError('incomplete baseline pairing')
	text = (
		'# HMMあり・なしの性能比較\n\n'
		f'保存済みHMM評価{metadata["hmm_cells"]}件を掲載した'
		f'{metadata["comparison_rows"]}行です。v1・v2・v3・legacyを同じCSVに収録し、'
		'`hmm_version`・`head_ks`・モデル・評価分割・評価対象を各行に明記しています。\n\n'
		'- [全バージョン・全seedの結果と対応比較](factorial_cells.csv)\n'
		'- [seed別の平均・標準偏差・勝敗数](factorial_comparison.csv)\n'
		'- [HMM損失設定間の効果・交互作用](factorial_contrasts.csv)\n'
		'- [Barlow蒸留比較](barlow_comparison.csv)\n'
		'- [v1蒸留比較](v1_comparison.csv)\n\n'
		'`without_hmm_primary`がHMMなし、`with_hmm_primary`がHMMあり、'
		'`primary_improvement`が改善量です。正ならHMMで改善、負なら悪化です。'
		'F3/Parihakaでは「あり−なし」、Volve MAEでは「なし−あり」としています。'
		'指標名・単位は`primary_metric`・`primary_unit`列で確認できます。\n\n'
		'`comparison_status=paired`は対照との評価・decoder条件が一致する比較です。'
		'`unmatched_baseline`の行もHMMの実測値を残し、HMMなし・改善量は空欄とし、'
		'理由を`comparison_note`に記載します。validationとtest、異なる評価対象を'
		'対応づけていません。F3の対照は共通test、Parihaka/Volveはtestです。\n\n'
		'副指標も同じ列構成です。改善量のSDは対応する5レイアウトの差から求めた'
		'標本SD（ddof=1）で、個別SDの差ではありません。集計表にはn・集計状態・'
		'HMMの勝・分・敗も示します。5レイアウト未満は明示し、1件のSDは空欄です。'
		'同一レイアウトが重複する集計は平均しません。\n\n'
		'`with_hmm_decoder_seed`と`without_hmm_decoder_seed`を各行に記載し、'
		'`comparison_type`で同一seed／異なるseedを区別します。Parihaka / MAE / largeは'
		'42001を`canonical`、42000を`historical`として両方掲載しています。'
		'異なるseedの差にはdecoder乱数の影響が含まれます。seedを混ぜて平均していません。\n\n'
		'HMMなし評価は条件が一致するHMM評価に再利用しています。MAEは既存の再構成継続モデル、'
		'Barlow Twinsは既存の3 epochモデル、Randomは未学習モデルです。学習予算を完全に'
		'揃えた因果実験ではなく、既存ベンチマークとの性能比較です。\n\n'
		'元のv2比較560行は保持しています。v1蒸留比較の405条件もこの主CSVに統合しています。'
		'全実験カタログにはHMM以外の手法も含む全評価・指標を収録します。\n\n'
		'5レイアウトはencoder seedの5反復ではなく、'
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
					'hmm_cells': metadata['hmm_cells'],
					**{
						k: metadata[k]
						for k in (
							'unique_no_hmm_cells',
							'comparison_rows',
							'aggregate_rows',
							'with_hmm_decoder_seeds',
							'without_hmm_decoder_seeds',
							'original_v2_cells',
						)
					},
					**{
						k: metadata[k]
						for k in (
							'saved_hmm_by_version',
							'comparison_rows_by_version',
							'matched_rows',
							'unmatched_rows',
						)
						if k in metadata
					},
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
	parser.add_argument('--overwrite', action='store_true')
	parser.add_argument('--review-output', type=Path)
	args = parser.parse_args()
	if args.overwrite and args.reuse_existing:
		parser.error('--overwrite and --reuse-existing are mutually exclusive')
	artifact = args.artifact_root.resolve()
	output = args.output.resolve()
	if args.reuse_existing:
		metadata = json.loads((output / 'provenance.json').read_text())
		if metadata['config_sha256'] != file_sha256(args.config):
			raise ValueError('existing baseline comparison config changed')
	else:
		export(args.config, artifact, output, overwrite=args.overwrite)
	if args.review_output:
		update_review(
			artifact / 'studies/hmm_v2/missing_controls_completion_v1/summary',
			output,
			artifact,
			args.review_output,
		)


if __name__ == '__main__':
	main()
