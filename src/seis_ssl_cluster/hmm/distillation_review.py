"""Curate the small, explicitly owned review files for the completed study."""

from __future__ import annotations

# ruff: noqa: RUF001 - Japanese review prose uses native punctuation.
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.hmm.distillation_sensitivity import comparison_rows

CURATED_COPIES = {
	'f3_validation.csv': 'summary/f3_validation.csv',
	'cells.csv': 'summary/cells.csv',
	'comparison.csv': 'summary/comparison.csv',
	'feature_diagnostics.csv': 'summary/feature_diagnostics.csv',
	'sensitivity.png': 'summary/sensitivity.png',
	'auxiliary_v1_pairs.csv': 'auxiliary/v1_pairs.csv',
	'auxiliary_v2_consistency_pairs.csv': 'auxiliary/v2_consistency_pairs.csv',
	'auxiliary_comparison.csv': 'auxiliary/comparison.csv',
}
PUBLISHED_FILES = frozenset({*CURATED_COPIES, 'README.md', 'provenance.json'})
RANDOM_CURATED_COPIES = {
	name: relative
	for name, relative in CURATED_COPIES.items()
	if not name.startswith('auxiliary_')
}
RANDOM_PUBLISHED_FILES = frozenset(
	{*RANDOM_CURATED_COPIES, 'README.md', 'provenance.json'}
)


def random_review_markdown(payload: dict[str, Any]) -> str:
	"""Describe the prespecified Random zero-versus-0.2 comparison."""
	lines = [
		'# HMM v2 Random初期化：蒸留0と0.2の比較',
		'',
		(
			'Random初期化・K=[6,8,10]・整合性0.1を固定し、蒸留0の25 epoch学習3件、'
			'特徴抽出3件、下流学習45セルを完了しました。既存0.2の3モデル・45セルを'
			'再利用し、計90セルを比較しています。係数の選び直しは行いません。'
		),
		'',
		(
			'F3は共通test Mean IoU、Parihakaはtest Channel IoU、Volveはtest macro '
			'MAE [sample]です。5レイアウトの平均±標本標準偏差（ddof=1）を示します。'
			'改善量はIoUで候補−0.2、MAEで0.2−候補とし、正が改善です。'
		),
		'',
		'| データ | 教師量 | 蒸留0 | 既存0.2 | 対応改善量（平均±標本SD） |',
		'| --- | --- | ---: | ---: | ---: |',
	]
	for survey in ('f3', 'parihaka', 'volve'):
		for size in ('small', 'medium', 'large'):
			group = {
				float(r['distillation_weight']): r
				for r in payload['rows']
				if (r['survey'], r['data_size']) == (survey, size)
			}
			values = [
				f'{group[w]["primary_mean"]:.5f} ± {group[w]["primary_sample_sd"]:.5f}'
				for w in (0.0, 0.2)
			]
			zero = group[0.0]
			lines.append(
				f'| {survey} | {size} | '
				+ ' | '.join(values)
				+ f' | {zero["primary_improvement_vs_020_mean"]:+.5f} ± '
				f'{zero["primary_improvement_vs_020_sample_sd"]:.5f} |'
			)
	lines.extend(
		[
			'',
			'![Random蒸留比較](sensitivity.png)',
			'',
			(
				'各layoutの値と対応差分は[cells.csv](cells.csv)、全18群の集計は'
				'[comparison.csv](comparison.csv)、F3 validationは別表'
				'[f3_validation.csv](f3_validation.csv)に保存しています。'
			),
			'',
			'固定評価cropでの最終teacher平均cosine距離：',
			'',
			'| データ | 蒸留0 | 既存0.2 |',
			'| --- | ---: | ---: |',
		]
	)
	for survey in ('f3', 'parihaka', 'volve'):
		distances = {
			float(r['distillation_weight']): float(
				r['fixed_crop_teacher_cosine_distance']
			)
			for r in payload['feature_diagnostics']
			if r['survey'] == survey and int(r['epoch']) == 25
		}
		lines.append(f'| {survey} | {distances[0.0]:.6f} | {distances[0.2]:.6f} |')
	lines.extend(
		[
			'',
			(
				'生値・重み付き損失、ヘッド別使用率エントロピー、ヘッド間整合性、固定crop距離は'
				'[feature_diagnostics.csv](feature_diagnostics.csv)に保存しています。新規0は'
				'開始時と各epoch終了時、既存0.2は保持checkpointから開始時と最終時を測定しました。'
				'既存0.2の中間固定crop値はcheckpoint未保存のため回収できません。'
			),
			'',
			(
				'Random自身の初期checkpointをstudentと固定teacherに使い、Random固有のHMM疑似ラベルを'
				'係数間で固定しています。最上位1 blockだけを更新します。'
				'蒸留は初期ランダム写像からの'
				'変化を抑える制約です。距離だけで下流性能や地質情報の保持を結論しません。'
			),
			'',
			(
				'継続学習seedは42の1回です。5レイアウトはencoder学習の5反復ではなく、3教師量も'
				'包含関係にあります。F3は同一調査地域内transductiveなラベルholdoutです。'
				'正本係数・選択済みcheckpointは変更していません。'
			),
			'',
			(
				'[MAEの結果](../hmm_v2_distillation_sensitivity_v1/README.md)との比較では、'
				'初期表現に加えてteacherとHMM targetも異なります。'
				'初期表現だけの効果とは扱いません。'
			),
			'',
			(
				'公開ファイルは7点に限定しています。[出典hash](provenance.json)、'
				'[実験定義・再開方法](../../experiments/hmm_v2/random_distillation_ablation_v1/README.md)'
				'を参照してください。完全な集計JSONと実行履歴はartifactsに保存しています。'
			),
		]
	)
	return '\n'.join(lines) + '\n'


def review_markdown(payload: dict[str, Any]) -> str:
	"""Describe all coefficients, signed differences and diagnostic limitations."""
	rows = payload['rows']
	lines = [
		'# HMM v2 特徴蒸留損失の感度解析',
		'',
		(
			'MAE初期化・K=[6,8,10]・整合性0.1の固定条件で、蒸留係数0 /'
			' 0.1 / 0.2 / 0.4を事前指定して比較しました。新規25 epo'
			'ch学習9件、特徴抽出9件、下流学習135セルを完了し、既存0.2の3モデル'
			'・45セルを再利用しました。最終比較は180セルです。'
		),
		'',
		(
			'5レイアウトの平均±標本標準偏差（ddof=1）を示します。F3は既存共通t'
			'estのMean IoU、Parihakaはtest Channel IoU'
			'、Volveはtest macro MAE [sample]です。F3 va'
			'lidationは集計JSONの別データに保存し、曲線には混ぜていません。'
		),
		'',
		(
			'| データ | 教師量 | 0 | 0.1 | 0.2 | 0.4 | '
			'0の改善量 vs 0.2 | 0.1–0.4の変動幅 |'
		),
		'| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
	]
	for survey in ('f3', 'parihaka', 'volve'):
		for size in ('small', 'medium', 'large'):
			group = {
				float(r['distillation_weight']): r
				for r in rows
				if (r['survey'], r['data_size']) == (survey, size)
			}
			values = [
				f'{group[w]["primary_mean"]:.5f} ± {group[w]["primary_sample_sd"]:.5f}'
				for w in (0.0, 0.1, 0.2, 0.4)
			]
			neighborhood = [group[w]['primary_mean'] for w in (0.1, 0.2, 0.4)]
			lines.append(
				f'| {survey} | {size} | '
				+ ' | '.join(values)
				+ f' | {group[0.0]["primary_improvement_vs_020_mean"]:+.5f} | '
				f'{max(neighborhood) - min(neighborhood):.5f} |'
			)
	lines.extend(
		[
			'',
			(
				'改善量はIoUで候補−0.2、Volve MAEで0.2−候補です。各lay'
				'outの値と対応差分は[cells.csv](cells.csv)、36群の'
				'平均・標本標準偏差は[comparison.csv](comparison.'
				'csv)にあります。'
			),
			'',
			'![蒸留感度](sensitivity.png)',
			'',
			'F3 validation: [f3_validation.csv](f3_validation.csv).',
			'',
			'固定cropでの25 epoch後のteacher平均cosine距離：',
			'',
			'| データ | 0 | 0.1 | 0.2 | 0.4 |',
			'| --- | ---: | ---: | ---: | ---: |',
		]
	)
	for survey in ('f3', 'parihaka', 'volve'):
		distances = {
			float(r['distillation_weight']): float(
				r['fixed_crop_teacher_cosine_distance']
			)
			for r in payload['feature_diagnostics']
			if r['survey'] == survey and int(r['epoch']) == 25
		}
		lines.append(
			f'| {survey} | '
			+ ' | '.join(f'{distances[w]:.6f}' for w in (0.0, 0.1, 0.2, 0.4))
			+ ' |'
		)
	lines.extend(
		[
			'',
			(
				'距離は固定teacherとのcosine距離で、蒸留係数0でも実測します。損'
				'失寄与0と表現差0を区別しています。新規9モデルは開始時と全epoch終了時'
				'の固定crop評価を保存しました。再利用0.2は中間checkpointが残'
				'っていないため固定crop評価は開始時・最終epochのみです。既存の各ep'
				'och学習crop距離は別列に保存しています。全診断値は[feature_d'
				'iagnostics.csv](feature_diagnostics.cs'
				'v)にあります。距離の大小だけから下流性能や地質情報の保持を結論しません。'
			),
			'',
			(
				'各係数・データセットにつきencoder学習はseed 42の1回です。5レ'
				'イアウトは5回のencoder学習ではなく、包含関係のある3教師量も独立反復'
				'ではありません。有意差検定による係数選択は行っていません。F3は同一調査地域'
				'内transductive・下流ラベルholdoutで、完全な空間振幅hol'
				'doutではありません。test結果による正本係数・選択済みcheckpoi'
				'ntの変更はありません。'
			),
			'',
			(
				'既存v1蒸留0.1対0.2は[135組](auxiliary_v1_pair'
				's.csv)、v2整合性0.1対0はK6810/K468を分けた[270組]'
				'(auxiliary_v2_consistency_pairs.csv)を整'
				'理しました。[補助集計](auxiliary_comparison.csv)'
				'ではMAE・現行asymmetric Barlow Twins・random'
				'を区別しています。v1値は新規v2曲線に含めません。F3 v1にはworke'
				'r数などの実行条件差があり、完全に同じ学習乱数系列の比較とは扱いません。'
			),
			'',
			(
				'公開レビュー用ファイルは10点のみです。原成果物・集計JSON・checkpo'
				'int・再開履歴はartifactsに保持し、このreportをパイプライン'
				'入力には使用しません。[出典hash](provenance.json)、['
				'実験定義と再開方法](../../experiments/hmm_v2/di'
				'stillation_sensitivity_v1/README.md)、['
				'実装・検証記録](../../experiments/hmm_v2/dist'
				'illation_sensitivity_v1/VALIDATION.md)'
				'を参照してください。'
			),
		]
	)
	return '\n'.join(lines) + '\n'


def curate_review(study: Path, output: Path) -> None:
	"""Copy only the exact numeric tables and figure from completed summaries."""
	payload = json.loads((study / 'summary/summary.json').read_text())
	if payload['status'] != 'complete':
		raise ValueError('review requires a completed study')
	is_random = payload.get('source') == 'random'
	comparison_rows(
		payload['cells'],
		weights=tuple(payload.get('weights', ('0.0', '0.1', '0.2', '0.4'))),
	)
	output.mkdir(parents=True, exist_ok=False)
	refs = []
	for name, relative in (
		RANDOM_CURATED_COPIES if is_random else CURATED_COPIES
	).items():
		source = study / relative
		shutil.copyfile(source, output / name)
		refs.append(
			{
				'report': name,
				'artifact_study_relative': relative,
				'sha256': file_sha256(source),
			}
		)
	provenance = {
		'study': payload.get('study', 'hmm_v2_k6810_distillation_sensitivity_v1'),
		'new_full_training': 3 if is_random else 9,
		'new_downstream_cells': 45 if is_random else 135,
		'reused_models': 3,
		'reused_downstream_cells': 45,
		'final_test_cells': 90 if is_random else 180,
		'files': refs,
		'full_summary_sha256': file_sha256(study / 'summary/summary.json'),
	}
	(output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
	(output / 'README.md').write_text(
		random_review_markdown(payload) if is_random else review_markdown(payload)
	)


def main() -> None:
	"""Publish the completed study's small review snapshot."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--random', action='store_true')
	parser.add_argument(
		'--output',
		type=Path,
		default=None,
	)
	args = parser.parse_args()
	root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	name = (
		'random_distillation_ablation_v1'
		if args.random
		else 'distillation_sensitivity_v1'
	)
	output = args.output or Path(f'reports/hmm_v2_{name}')
	curate_review(root / 'studies/hmm_v2' / name, output)
	if args.random:
		receipt = {
			'status': 'complete',
			'utc': datetime.now(timezone.utc).isoformat(),
			'new_full_25epoch_training': 3,
			'new_extractions': 3,
			'new_downstream_cells': 45,
			'reused_baseline_models': 3,
			'reused_baseline_downstream_cells': 45,
			'final_test_cells': 90,
			'f3_common_test_cells': 30,
			'report': str(output),
			'additional_experiment_audits': False,
			'canonical_selection_modified': False,
			'pending': [],
		}
		with (root / 'operations/hmm_v2' / name / 'analysis_complete.json').open(
			'x'
		) as stream:
			json.dump(receipt, stream, indent=2)
			stream.write('\n')


if __name__ == '__main__':
	main()
