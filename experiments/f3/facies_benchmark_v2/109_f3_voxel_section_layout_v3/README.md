# F3 voxel section-layout v3

`facies_benchmark_v2`のfive-way比較v3が使うmodel非依存のsection-layout
supervisionを作るstageである。prepared volume、canonical supervision、section
inventory、reference valid-token maskは生成済みのv2 artifactをread-onlyで再利用し、
contractと15条件のdatasetだけをlayout v3名前空間へ書く。

## 科学契約

- sizeはsection本数で定義する: small = inline 1本 + crossline 1本、medium =
  2本 + 2本、large = 4本 + 4本。各layoutのactive linesはstrict prefixで入れ子。
- layoutsは`layout_000`から`layout_004`までの5つで、統計単位は`layout_id`。
  subsample seedはlayoutへ1対1に固定し、独立したseed次元は追加しない。
- selection semanticsは`seeded_nested_class_balanced_section_token_rows_v1`。
  patchは`[8, 8, 8]`、actual voxel数の許容相対誤差は0.05。
- tokenization policyはlabeled fraction 0.5以上、majority fraction 0.7以上、
  z両端1 sampleを除外する。
- per-class token-row capはsmall 25、medium 50、large 100。6 classなので、
  選択row総数は各条件でexact 150 / 300 / 600。
- validation maskは全15条件で同一。全6 class非ゼロ、class 3・5非ゼロ、全active
  lineの寄与が正、teacher maskのsmall ⊂ medium ⊂ largeをgateする。
- teacher materializationはv2と同じpartial active-section token footprintを使う。
  active section内の全voxelを教師にはしない。

layoutとselection seed:

| layout | seed | ordered inlines | ordered crosslines |
|---|---:|---|---|
| layout_000 | 0 | 350, 450, 550, 250 | 575, 900, 1025, 450 |
| layout_001 | 1 | 470, 570, 270, 370 | 925, 1050, 475, 600 |
| layout_002 | 2 | 290, 590, 390, 490 | 1075, 500, 625, 950 |
| layout_003 | 3 | 310, 410, 510, 610 | 525, 850, 975, 1100 |
| layout_004 | 4 | 430, 530, 630, 330 | 875, 1000, 1125, 550 |

line集合と5-layout partitionはv2と同じ。`layout_002`だけinline 290と590の
順序を入れ替え、small/mediumのclass 5 candidate rowをexact capへ到達可能にする。

## nested seeded selection

各layoutについてcanonical token-row identityのstable orderを作り、layoutに固定した
1個のRNGを使う。sizeはsmall、medium、large、classは0から5の順に処理する。
不足数が未選択pool全件と等しい場合は、v1 samplerと同様に全件をstable orderで
採用し、RNGを消費しない。

1. smallのactive linesから各class exact 25 rowsをreplacementなしで選ぶ。
2. mediumではsmallの選択をすべて保持し、medium active poolの未選択rowから各class
   exact 50まで不足分だけを同じRNGで追補する。
3. largeもmediumの選択を保持し、large active poolからexact 100まで追補する。

どのclassでも必要数へ到達できなければcontract finalizeを失敗させる。選択rowの
identity/countと、teacher materialization前にtoken XYZをdeduplicateしたunique tokenの
identity/countを別々に記録する。両countの差からduplicate row数を監査できる。

これはv1と同じper-class capとseedを使うが、v1のcap25/cap50/cap100は各capを
独立drawしており非nestedだった。v3は入れ子を保証するため不足分を逐次追補するので、
旧v1とexact same drawではない。

## v1参照値

実装前に
`reports/f3/legacy/facies_benchmark_v1/voxel_lithology_label_budget_v1/tables/paired_metrics.csv`
から再確認したv1の`train_voxel_count`は次のとおり。このreportはprovenanceであり、
v3の実行時入力にはしない。

| cap | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | median |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 10,024 | 10,152 | 10,024 | 10,256 | 10,160 | 10,152 |
| 50 | 20,520 | 20,128 | 20,184 | 20,192 | 19,904 | 20,184 |
| 100 | 40,504 | 40,320 | 40,520 | 40,672 | 40,616 | 40,520 |

v3でv1と同じなのはper-class token-row capとseed集合である。section候補領域、
nested追補、teacher footprintが異なるため、v3 actual voxel数をこのv1 rangeへ
合わせる追加samplingは行わない。

## nominal targetとlive actual

section token rowのnominal footprintは8 × 8 = 64 voxelsなので、固定nominal targetは
次のとおり。

| size | rows / class | total rows | nominal train voxels |
|---|---:|---:|---:|
| small | 25 | 150 | 9,600 |
| medium | 50 | 300 | 19,200 |
| large | 100 | 600 | 38,400 |

survey edge、inline/crossline intersection、同一token XYZのduplicate、canonical
train/known-label mask、validation precedenceによるpartial footprintのため、actualは
nominalと完全一致しない場合がある。actualはnominalに対する相対誤差0.05以内で
なければならない。

2026-08-28に生成したcontractと15-condition manifestから得た値:

| layout | size | target | actual | relative error |
|---|---|---:|---:|---:|
| layout_000 | small | 9,600 | 9,632 | 0.003333333333 |
| layout_000 | medium | 19,200 | 19,624 | 0.022083333333 |
| layout_000 | large | 38,400 | 39,280 | 0.022916666667 |
| layout_001 | small | 9,600 | 9,576 | 0.002500000000 |
| layout_001 | medium | 19,200 | 19,320 | 0.006250000000 |
| layout_001 | large | 38,400 | 39,848 | 0.037708333333 |
| layout_002 | small | 9,600 | 9,600 | 0.000000000000 |
| layout_002 | medium | 19,200 | 19,512 | 0.016250000000 |
| layout_002 | large | 38,400 | 39,672 | 0.033125000000 |
| layout_003 | small | 9,600 | 9,688 | 0.009166666667 |
| layout_003 | medium | 19,200 | 19,488 | 0.015000000000 |
| layout_003 | large | 38,400 | 39,728 | 0.034583333333 |
| layout_004 | small | 9,600 | 9,712 | 0.011666666667 |
| layout_004 | medium | 19,200 | 19,424 | 0.011666666667 |
| layout_004 | large | 38,400 | 39,424 | 0.026666666667 |

validation mask SHA-256は全15条件で
`0a6d134e4ea276ea15c29381cc8a1dd85cbdd928ece3df6882aa06e324129c70`
の1種類である。

## 実行

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${F3_ROOT:?export F3 data root first}"
export SEIS_SSL_CLUSTER_WORKSPACE="${SEIS_SSL_CLUSTER_WORKSPACE:-/workspace}"
cd "$SEIS_SSL_CLUSTER_WORKSPACE"
export LAYOUT=experiments/f3/facies_benchmark_v2/109_f3_voxel_section_layout_v3

python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
  --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode inspect --dry-run
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
  --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode inspect
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
  --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode finalize --dry-run
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
  --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode finalize
python proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py \
  --config "$LAYOUT/03_build_section_layout_datasets.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py \
  --config "$LAYOUT/03_build_section_layout_datasets.yaml"
```

出力:

```text
lithology/f3/facies_benchmark_v2/
├── voxel_section_layout_v3_calibration/
│   ├── section_candidates.{csv,json}
│   └── f3_voxel_section_layout_contract.json
└── voxel_section_layout_v3/
    ├── section_layout_dataset_manifest.json
    └── datasets/layout=layout_00N/size={small,medium,large}/voxel_supervision/
```

builderの`outputs.output_root`は
`../110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml`の
`section_layout.dataset_root`と同一にする。

上のlive生成コマンドは新規v3 rootを一度だけ作る。既存出力はproducerが上書きを
拒否するため、生成済みartifactの再確認にはdry-runとdownstream preflightを使う。
再生成が必要な場合も既存rootを黙って削除・移動せず、実行中jobと退避先を先に
確認する。
