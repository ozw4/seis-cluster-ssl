# F3 voxel section-layout v2

`facies_benchmark_v2`の五者比較（`../110_lithology_mae_local_bt_five_way_v2`）が
使うmodel非依存のsection-layout supervisionを、v2 canonical supervisionと
v2 candidate statisticsだけから決めるstageである。

v1の`109_f3_voxel_section_layout_v1`（current treeでは退役済み）は実装・schemaの
参照元であり、v2のline配置・教師量の正解データではない。v1のline配置、
cap25/cap50/cap100、過去のactual voxel countとの一致は要求しない。

## 科学契約（v1 five-wayと共通）

- data sizeはsection本数で定義する: small = inline 1本 + crossline 1本、
  medium = 2本 + 2本、large = 4本 + 4本。各layout内でsmall < medium < largeは
  ordered listのprefixとして厳密に入れ子。
- layouts: `layout_000` … `layout_004`、統計単位は`layout_id`。
- selection semantics: `stable_hash_partial_section_token_footprints_v1`
  （patch `[8, 8, 8]`、tolerance `allowed_relative_error: 0.05`）。
- validation maskは全layout・size・modelで共通（canonical validationをbitwise保持）。
- decoder契約は`FIXED_DECODER_CONTRACT`（seed 42000）に固定。

## v2 section inventory（`../10_prepare/section_inventory_v2.csv`）

benchmarkのPNG inventoryはtrain sectionが12本（inline 5本 + crossline 7本）
しかなく、5 layouts × 4+4 linesを重複なしに配置できない。v2では
validation sectionを据え置き、train sectionをdense SEGY label（`f3_labels.sgy`）
から規則的に増やす。

- validation: inline 150、crossline 350・750（PNG inventoryどおり、変更なし）。
- guard: train sectionは同軸のvalidation sectionから100 line以上離す
  （benchmarkの元のtrain–validation最小間隔）。survey端の1 tokenも避ける。
- train inline: 250から20 line間隔で20本（250, 270, …, 630）。
- train crossline: guardが許す2区間で25 line間隔、区間長に比例して8本 + 12本
  （450, 475, …, 625 と 850, 875, …, 1125）。
- labelは常にlabel volume（`f3_labels.sgy`由来）から読む。PNGとSEGY labelの
  整合はPNG sectionに対して`00_inspection/05_check_label_consistency.yaml`で
  確認済み。

canonical supervision: train 6,620,640 voxels、validation 470,136 voxels
（`lithology/f3/facies_benchmark_v2/voxel_supervision/dense_label_sections_v2`）。

## 入力

- canonical supervision: 上記（`../10_prepare/03_build_voxel_supervision.yaml`）
- label volume: `registry/volumes/f3/facies_benchmark_v2/f3_facies_labels.npy`
- line inventory: `../10_prepare/section_inventory_v2.csv`
- geometry / class info: `inspection/f3/facies_benchmark_v2/`
- reference valid tokens: v2 prepared amplitudeに対しfive-wayと同じgeometry
  （window 128、overlap 64）で抽出したv2 reference
  `embeddings/f3/facies_benchmark_v2/reference_token_geometry/.../overlap_x64/f3_facies_benchmark.valid_tokens.npy`
  （`../10_prepare/02_extract_reference_valid_tokens.yaml`。encoder非依存で、
  v1 F3 reference embedding（overlap x16）のmaskとSHAがbyte一致する。）

## 実行順

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export F3_ROOT=/home/dcuser/data/public_data/field/F3
export SEIS_SSL_CLUSTER_WORKSPACE=/workspace
export LAYOUT=experiments/f3/facies_benchmark_v2/109_f3_voxel_section_layout_v2

python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode inspect --dry-run
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode inspect
# 02_layout_lines.yaml を candidate report から決める（規則は下記）
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode finalize --dry-run
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode finalize
python proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py --config "$LAYOUT/03_build_section_layout_datasets.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py --config "$LAYOUT/03_build_section_layout_datasets.yaml"
```

出力:

```text
lithology/f3/facies_benchmark_v2/
├── voxel_section_layout_v2_calibration/
│   ├── section_candidates.{csv,json}
│   └── f3_voxel_section_layout_contract.json
└── voxel_section_layout_v2/
    ├── section_layout_dataset_manifest.json
    └── datasets/layout=layout_00N/size={small,medium,large}/voxel_supervision/
```

calibration出力はdataset rootの外に置く（builderはdataset rootを原子的に
commitするため、既存directoryを拒否する）。`outputs.output_root`は
`../110_lithology_mae_local_bt_five_way_v2/60_five_way.yaml`の
`section_layout.dataset_root`と同一。`datasets_v1/datasets/...`のような二重階層は
作らない。

## v2 candidate report（2026-08-25、43 lines）

各train inlineの有効教師voxelは201,336、各train crosslineは134,176で一様
（z-border 1 sampleとinvalid token slabの除外が全lineで同じため）。
class 0–3は全train lineに存在する。class 4・5の有無:

| lines | class 4 | class 5 |
|---|---|---|
| inline 250–450 | あり | あり（450は347 voxelsのみ） |
| inline 470, 490 | あり（490は204） | なし |
| inline 510–570 | なし | なし |
| inline 590, 610, 630 | なし | あり（54 / 560 / 1,259） |
| crossline 450–625 | あり | なし |
| crossline 850–1075 | あり | あり |
| crossline 1100 | なし | なし |
| crossline 1125 | なし | あり（31） |

詳細な per-line class count は
`voxel_section_layout_v2_calibration/section_candidates.csv`。

## line選択規則（v2）

model identity・metric・過去のfive-way結果は使わない。位置だけで決める。

1. 軸ごとにtrain候補を昇順に並べる（位置0–19）。validation lineは候補に入れない。
2. layout k（k = 0..4）は各軸の位置 k, k+5, k+10, k+15 を取る（quartet index
   j = 0..3）。5つのquartetは在庫を分割するので、layout間でlineは重複せず、
   各layoutはsurvey全体に広がる（layout内の隣接lineはinline 100本 /
   crossline 125本、または区間gapを跨ぐ）。
3. nested orderはquartetを j0 = (k + 1) mod 4 から周期的に歩く
   （small = 1本目、medium = 1–2本目、large = 4本すべて）。
   j0 = k mod 4 だとlayout_003のsmallがinline 610 + crossline 1100となり
   class 4が0になる。開始位置を1つずらすのが全class gateを満たす最小の変更。

結果（`02_layout_lines.yaml`）:

| layout | ordered inlines | ordered crosslines | small | medium追加 |
|---|---|---|---|---|
| layout_000 | 350, 450, 550, 250 | 575, 900, 1025, 450 | il 350 + xl 575 | il 450, xl 900 |
| layout_001 | 470, 570, 270, 370 | 925, 1050, 475, 600 | il 470 + xl 925 | il 570, xl 1050 |
| layout_002 | 590, 290, 390, 490 | 1075, 500, 625, 950 | il 590 + xl 1075 | il 290, xl 500 |
| layout_003 | 310, 410, 510, 610 | 525, 850, 975, 1100 | il 310 + xl 525 | il 410, xl 850 |
| layout_004 | 430, 530, 630, 330 | 875, 1000, 1125, 550 | il 430 + xl 875 | il 530, xl 1000 |

- 20 inline・20 crosslineがすべて使われ、layout間の重複はない（large含む）。
- 5つのsmall条件は互いに異なるinline/crossline対。
- class gate（6 class全部、class 3・5非ゼロ、active line寄与正）はfinalizeが
  全15条件で検証済み。最小のclass 5はlayout_002/smallの1,243 voxels、
  最小のclass 4はlayout_002/smallの2,912 voxels。

## target calibration規則（v2）

`targets.rule: max_common_reachable_active_pool_v1`。sizeごとに、各layoutの
active section（prefix）上のcanonical train voxel poolを数え、5 layoutsの最小値を
共通targetとする。これは全layoutが到達できる最大の共通targetであり、model
performanceには依存しない。

| size | active pool（全5 layoutで同一） | target | preview actual | relative error |
|---|---:|---:|---:|---:|
| small | 335,288 | 335,288 | 335,288 | 0 |
| medium | 670,128 | 670,128 | 670,128 | 0 |
| large | 1,338,464 | 1,338,464 | 1,338,464 | 0 |

全lineの有効voxel数が一様で、line交差のtrain voxel数も一様なため、poolは
layout間で完全一致し、各条件はactive section上のcanonical train voxelを
すべて教師に使う（selected tokens: small 5,235 / medium 10,414 / large 20,604）。
poolはcontractの`target_calibration.active_pool_train_voxel_counts`に、
target/actual/relative errorは各`layouts[].sizes[]`と
`section_layout_dataset_manifest.json`に記録される。

validation identity: 470,136 voxels、mask SHA-256
`0a6d134e4ea276ea15c29381cc8a1dd85cbdd928ece3df6882aa06e324129c70`
（全15条件でbyte一致）。

## 完了条件

- contractがresolver（`f3_lithology_voxel_section_layout_contract_from_mapping`）で
  再読込でき、layout ID・4+4 lines・1+1/2+2/4+4 prefix・strict nesting・
  patch size・selection/validation semantics・decoder seed 42000・fixed decoder・
  v2 source path/SHA・target/preview count・class gateを満たす。
- 15 conditionが`section_layout_dataset_manifest.json`に列挙され、各conditionの
  7 fileが揃い、validation mask SHAが15条件でbyte一致する。
