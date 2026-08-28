# F3 lithology MAE / Local Barlow Twins five-way v2

F3岩相推定で、次の5表現を同一のfrozen full-volume token embedding契約と
既存F3 voxel decoderで比較するsuiteの`facies_benchmark_v2`版である。

- `mae`
- `mae_hmm_k6`
- `local_barlow_twins`
- `local_barlow_twins_hmm_k6`
- `random`

`local_barlow_twins`はtrace-dropを使わない`local_barlow_twins_3d`
（`local_pairs_per_crop: 128`）である。trace-drop系のcheckpoint・embeddingは
このsuiteでは使用しない。

## v2の科学契約

v2はv1のline配置やper-class-cap選択を再現する実験ではない。v1は実装・前処理・
decoder・checkpoint・評価方法に加え、総教師voxel規模の参照である。

- dataset identity: `f3_facies_benchmark` / `facies_benchmark_v2` / `f3_facies_benchmark`
- data sizeはsection本数で定義する: small = inline 1本 + crossline 1本、
  medium = 2本 + 2本、large = 4本 + 4本（各layout内で厳密なprefix入れ子）。
  section本数は教師の空間配置・範囲を定義し、section内の全voxelは使わない
- line配置はv2 candidate statisticsから新規に決定する
  （規則は[`../109_f3_voxel_section_layout_v2/README.md`](../109_f3_voxel_section_layout_v2/README.md)）
- 教師量（`target_train_voxel_count`）は`fixed_train_voxel_counts_v1`で固定する。
  v1 cap25/cap50/cap100の5 subsample seedsにおけるtrain voxel countの中央値
  10,152 / 20,184 / 40,520を、small/medium/largeの全5 layouts共通targetにする
- active section内では`stable_hash_partial_section_token_footprints_v1`を使って
  token footprintを選ぶ。token粒度によりactualはtargetと完全一致しない場合があるが、
  `allowed_relative_error: 0.05`以内でなければならない。v1と同じなのは総教師voxel
  規模であり、v1のper-class-cap選択そのものではない
- 下流評価: frozen encoder token embedding + 既存F3 voxel decoder
- layouts: `layout_000` `layout_001` `layout_002` `layout_003` `layout_004`
- data sizes: `small` `medium` `large`
- job数: 5 models × 5 layouts × 3 sizes = 75
- 主指標: validation unique voxels上の`macro_f1`
- 副指標: `mean_iou` `balanced_accuracy` `weighted_f1`
- 統計単位: `layout_id`（small/medium/largeは別々に集計する）
- decoder契約は`src/seis_ssl_cluster/config/f3_lithology_voxel_section_layout.py`
  の`FIXED_DECODER_CONTRACT`（50 epochs、batch 1、LR `1.0e-3`、seed 42000）に固定
- Random encoderはseed 42

v1から変えない前処理条件: clipping percentiles `[0.5, 99.5]`、epsilon `1.0e-6`、
normalization sample budget 1,000,000、seed 42、axis order `[x, y, z]`、dtype
（seismic float32 / label int16）、survey geometry、class order、train/validation
split生成規則。出力pathとdataset versionだけがv2である。

## v1 artifactの明示的再利用

- 5つのencoder checkpointはv1 fixed-budget artifactを再学習せずに参照する
  （`60_five_way.yaml`の`models[].checkpoint`）。v2 directoryへコピー・symlinkして
  version表記だけを変えることはしない。checkpointをv2 amplitudeに適用する前に、
  v1/v2 prepared volumeの同一性を生成済みartifactで確認する（手順9の
  `check_f3_prepared_volume_parity.py`。config値の一致だけでは足りない）。
  - `mae`: `pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/mae100/mae_continue/full_25ep/latest.pt`
  - `mae_hmm_k6`: `pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/mae100/hmm/k6/full_25ep/latest.pt`
  - `local_barlow_twins`: `pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/stage2/local_bt100/local_bt_continue/full_25ep/latest.pt`
  - `local_barlow_twins_hmm_k6`: `pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/stage2/local_bt100/hmm/k6/full_25ep/latest.pt`
  - `random`: `pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/random/random_init.pt`
- canonical supervisionのreference valid-token maskは、v2 prepared amplitudeに
  対してfive-wayと同じgeometryで抽出したv2 artifact
  （`embeddings/f3/facies_benchmark_v2/reference_token_geometry/.../overlap_x64`）
  を使う。encoderは既存のuntrained checkpoint
  `pretraining/f3/facies_benchmark_v1/random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/random_init/mae_random_seed42.pt`
  で、token validityはencoder非依存（zero-mask契約とamplitudeだけで決まる）。
  decoderは各jobの開始時にこのmaskのSHAとv2 embeddingのvalid-token SHAの一致を
  再検証する（不一致なら学習前に停止する）。
- prepare stageの`inputs.inspection_report`はraw-data QC reportとして
  `inspection/f3/facies_benchmark_v1/report.json`を参照する（report builder CLIは
  退役済み）。
- validation section（inline 150、crossline 350・750）はbenchmarkのPNG inventory
  どおり。train sectionはdense SEGY labelから規則的に増やした
  `10_prepare/section_inventory_v2.csv`で定義する。

embedding、section-layout dataset、runs、summaryはすべてv2名前空間に書く。

## 環境

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export F3_ROOT=/home/dcuser/data/public_data/field/F3
export SEIS_SSL_CLUSTER_WORKSPACE=/workspace
export INSPECT=experiments/f3/facies_benchmark_v2/00_inspection
export PREP=experiments/f3/facies_benchmark_v2/10_prepare
export LAYOUT=experiments/f3/facies_benchmark_v2/109_f3_voxel_section_layout_v2
export EXP=experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v2
export EXP_V1=experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1
export SSL_V1=experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1
export LOCAL_BT_V1=experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1
export CONFIG="$EXP/60_five_way.yaml"
```

## 1. v2 prepared volume（CPU）

v2 inspection artifactを作り、raw F3からv2 prepared amplitude・label volume・
manifest・split・normalization statsを生成する。

```bash
python proc/seis_ssl_cluster/inspect_f3_files.py --config "$INSPECT/01_inspect_files.yaml" --dry-run
python proc/seis_ssl_cluster/inspect_f3_files.py --config "$INSPECT/01_inspect_files.yaml"
python proc/seis_ssl_cluster/inspect_f3_segy_geometry.py --config "$INSPECT/02_inspect_segy_geometry.yaml"
python proc/seis_ssl_cluster/inspect_f3_png_labels.py --config "$INSPECT/03_inspect_png_labels.yaml"
python proc/seis_ssl_cluster/prepare_f3_facies_volume.py --config "$PREP/01_prepare_f3_volume.yaml" --dry-run
python proc/seis_ssl_cluster/prepare_f3_facies_volume.py --config "$PREP/01_prepare_f3_volume.yaml"
python proc/seis_ssl_cluster/visualize_f3_quicklook.py --config "$INSPECT/04_make_quicklook_figures.yaml"
python proc/seis_ssl_cluster/check_f3_label_consistency.py --config "$INSPECT/05_check_label_consistency.yaml"
python proc/seis_ssl_cluster/preview_f3_tokenization.py --config "$INSPECT/06_make_tokenization_preview.yaml"
```

出力: `registry/{volumes,manifests,splits,normalization_stats}/f3/facies_benchmark_v2/`、
`inspection/f3/facies_benchmark_v2/`。

## 2. v2 canonical voxel supervision（reference token geometryはGPU、supervisionはCPU）

token validityの参照maskを、v2 prepared amplitudeに対してfive-wayと同じ抽出
geometry（window 128、overlap 64）で1回だけ抽出する。encoderには既存の
untrained checkpoint（`random_encoder_..._seed42_v1/random_init/mae_random_seed42.pt`）
を使う。token validityはzero-mask契約とamplitudeだけで決まりencoder非依存なので、
下流が使うのはこのmaskとmetadataだけである。

```bash
python proc/seis_ssl_cluster/extract_embeddings.py --config "$PREP/02_extract_reference_valid_tokens.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$PREP/02_extract_reference_valid_tokens.yaml" --device cuda
```

v2 section inventory（`$PREP/section_inventory_v2.csv`）は、benchmarkのvalidation
section（inline 150、crossline 350・750）を据え置き、train sectionをdense SEGY label
から規則的に増やしたものである（規則は`$LAYOUT/README.md`）。labelは常に
`f3_labels.sgy`由来のlabel volumeから読む。`voxel_dataset.inventory_semantics:
dense_segy_label_section_inventory_v1`がその事実をartifactのsplit provenance
（`split_manifest.json`の`split_source`/`strategy`、
`voxel_dataset_metadata.json`の`split_strategy`）に記録する（省略時は
`png_label_inventory_v1`、v1のPNG annotation inventoryの意味）。

```bash
python proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py --config "$PREP/03_build_voxel_supervision.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py --config "$PREP/03_build_voxel_supervision.yaml"
```

出力: `embeddings/f3/facies_benchmark_v2/reference_token_geometry/.../overlap_x64/`と
`lithology/f3/facies_benchmark_v2/voxel_supervision/dense_label_sections_v2/`
（`supervision_split_grid.npy` `voxel_dataset_metadata.json` `class_counts.csv`
`split_manifest.json` `voxel_dataset_summary.md`）。

## 3. candidate inspection（CPU）

```bash
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode inspect --dry-run
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode inspect
```

出力: `lithology/f3/facies_benchmark_v2/voxel_section_layout_v2_calibration/section_candidates.{csv,json}`。

## 4. v2 line selection

`$LAYOUT/02_layout_lines.yaml`をcandidate reportから決める。規則と結果は
`$LAYOUT/README.md`に記録済み（sort済みtrain候補から各軸の位置
k, k+5, k+10, k+15（k = layout index）を取り、quartet内を指定の開始位置から
循環順に並べる。model identity・metricは使わない）。

## 5. v2 target calibration

`targets.rule: fixed_train_voxel_counts_v1`で、small = 10,152、medium = 20,184、
large = 40,520を指定する。これらは
`reports/f3/legacy/facies_benchmark_v1/voxel_lithology_label_budget_v1/tables/paired_metrics.csv`
にあるcap25/cap50/cap100の5 subsample seedsの中央値であり、versioned configへ
固定する。実行時にreportは読まない。

finalizeは各layout/sizeのactive poolを診断用に計算し、固定targetが全layoutで
到達可能であることを確認してから、stable-hash token footprint selectionを行う。
preview actualはtargetに対する許容誤差内でなければならない。

## 6. contract finalize（CPU）

```bash
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode finalize --dry-run
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py --config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode finalize
```

出力: `lithology/f3/facies_benchmark_v2/voxel_section_layout_v2_calibration/f3_voxel_section_layout_contract.json`。

## 7. 15 section-layout datasets（CPU）

```bash
python proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py --config "$LAYOUT/03_build_section_layout_datasets.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py --config "$LAYOUT/03_build_section_layout_datasets.yaml"
```

出力: `lithology/f3/facies_benchmark_v2/voxel_section_layout_v2/section_layout_dataset_manifest.json`
と`datasets/layout=*/size=*/voxel_supervision/`（15条件）。中断・部分出力の扱いは
`--only-missing`（完全なconditionだけ再利用）と`--only-missing --quarantine-invalid`
（不正conditionを退避して再構築）。黙って削除しない。

## 8. checkpoint audit（v1 upstream）

5つのcheckpointが存在しなければ、v1 runbookの手順でGPU学習する（v2は再学習しない）。

```bash
# MAE 100 → 100+25 control / 100+HMM-K6 25（21_ssl_hmm_continuation_v1 RUNBOOK_HMM_K6.md）
python proc/seis_ssl_cluster/train_amp_mae.py --config "$SSL_V1/10_stage1/mae/02_full_100ep.yaml"
python proc/seis_ssl_cluster/train_amp_mae.py --config "$SSL_V1/30_stage2/mae100/mae_continue/02_full_25ep.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py --config "$SSL_V1/20_hmm_targets/mae100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$SSL_V1/20_hmm_targets/mae100/k6/02_cluster_hmm_k6.yaml"
bash "$SSL_V1/20_hmm_targets/mae100/k6/03_export_pseudo_targets.sh"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$SSL_V1/30_stage2/mae100/hmm/k6/02_full_25ep.yaml"
# Local BT 100（22_local_barlow_twins_v1）
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$LOCAL_BT_V1/02_full_100ep.yaml"
# Local BT 100+25 control / HMM-K6 target / 100+HMM-K6 25 / Random seed 42（110 v1）
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$EXP_V1/10_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP_V1/20_hmm_targets/local_bt100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$EXP_V1/20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml"
bash "$EXP_V1/20_hmm_targets/local_bt100/k6/03_export_pseudo_targets.sh"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP_V1/30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml"
python proc/seis_ssl_cluster/create_random_mae_checkpoint.py --config "$EXP_V1/40_random/01_create_random_checkpoint.yaml"
```

各full configの前には対応する`01_gpu_feasibility_1step.yaml`と`--dry-run`を
v1 runbookどおりに実行する。checkpointが揃ったら、v2 configで静的な
source planを確認する。

```bash
python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py --config "$CONFIG" --dry-run
```

## 9. v1/v2 prepared volume parity gate（CPU、read-only）

v1で学習したcheckpointをv2 amplitudeに適用する科学的前提は、v1/v2の
prepared volumeが同一であることである。上の「v1から変えない前処理条件」は
config値の一致にすぎないので、embedding抽出の前に生成済みartifactそのものを
比較する。比較対象は v1/v2 `f3_seismic.npy`・`f3_facies_labels.npy`の
SHA-256、shape / dtype / grid_order、normalization statsの意味的な数値field
（clip percentiles、epsilon、sample count、seed、computed clip bounds、
center = median / scale = iqr）、class orderで、出力pathとdataset versionは
比較しない（sample countとseedはstats JSONに記録されないためprepare configから
取る）。何も書かない。

```bash
python proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py --dry-run
python proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py
```

既定の`--reference-config`はv1の`01_prepare_f3_volume.yaml`、
`--candidate-config`はv2の同名configである。`f3_prepared_parity.status: FAIL`
（exit code非0）なら、v1 checkpointをそのまま比較に入れず、
`f3_prepared_parity.mismatch:`行の原因を確認して停止する。

2026-08-25の実行結果（PASS）:

```text
seismic_sha256: 47108252f4bd670889da1ea6f36abe8acba41a6ad772db515b5902d4545bb276  (v1 == v2)
label_sha256:   daf2b900a6c68cc1dc5864f5ef0a1bd527c48c9f29842453d0b889378b3bf09d  (v1 == v2)
shape_xyz (601, 901, 255) / seismic float32 / label int16 / grid_order (x, y, z)
normalization: percentiles [0.5, 99.5], eps 1e-06, max_samples 1000000, seed 42,
               clip_low -0.8653416681289673, clip_high 0.6771933567523928,
               median 0.008167065680027008, iqr 0.20806461572647095
class_order: 0=Upper North Sea, 1=Middle North Sea, 2=Lower North Sea,
             3=Rijnland/Chalk, 4=Scruff, 5=Zechstein
f3_prepared_parity.status: PASS
```

## 10. v2 embedding extraction（GPU）

抽出条件はfive-way v1と同じ（window `[128,128,128]`、overlap `[64,64,64]`、
float16、`amp: false`、min token valid fraction 0.5、encoder出力のみ。Local BT
projectorとHMM headの出力は使わない）。manifestはv2 prepared amplitudeである。

```bash
for extract in \
  01_extract_mae \
  02_extract_mae_hmm_k6 \
  03_extract_local_barlow_twins \
  04_extract_local_barlow_twins_hmm_k6 \
  05_extract_random
do
  python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/50_embeddings/${extract}.yaml" --dry-run
  python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/50_embeddings/${extract}.yaml"
done
```

出力: `embeddings/f3/facies_benchmark_v2/mae_local_bt_five_way_v2/<model>/overlap_x64/`。

## 11. five-way source audit

decoderはsource auditがPASSするまで開始しない。

```bash
python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py --config "$CONFIG" --dry-run
python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py --config "$CONFIG"
```

read-only auditは五者のcheckpoint SHA、抽出契約、objective identity、
valid-token maskのbyte一致に加えて、固定予算(25 epoch / 15,625 global steps、
encoder top-1)とtrace-dropなしの系譜を検証する。`random`はepoch 0の未学習表現で
あることを確認する。

## 12. preflight（`layout_000/small`の五者dry-run）

```bash
for model in \
  mae \
  mae_hmm_k6 \
  local_barlow_twins \
  local_barlow_twins_hmm_k6 \
  random
do
  python proc/seis_ssl_cluster/run_f3_lithology_five_way.py \
    --config "$CONFIG" \
    --model "$model" \
    --layout layout_000 \
    --size small \
    --dry-run
done
```

五者の`condition_dir`、`inline_lines`、`crossline_lines`、
`selected_token_identity_sha256`、`validation_mask_sha256`、
`train_voxel_count`、`decoder_initial_state_sha256`が一致することを確認する。

## 13. full suite（75 decoder jobs）

schedulerや独自queueは使わず、単純な3重loopで実行する。最初のjobは学習前に
condition datasetのreference valid-token SHAとembeddingのvalid-token SHAの一致を
検証し、不一致なら停止する。

```bash
for size in small medium large
do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004
  do
    for model in \
      mae \
      mae_hmm_k6 \
      local_barlow_twins \
      local_barlow_twins_hmm_k6 \
      random
    do
      python proc/seis_ssl_cluster/run_f3_lithology_five_way.py \
        --config "$CONFIG" \
        --model "$model" \
        --layout "$layout" \
        --size "$size"
    done
  done
done
```

1 jobの出力は
`f3_lithology_benchmark/mae_local_bt_five_way_v2/runs/model=<model>/layout=<layout>/size=<size>/`
配下の`decoder/` `prediction/` `evaluation/`に分かれる。

## 14. summary dry-run（完全性監査）

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py --config "$CONFIG" --dry-run
```

`complete_jobs: 75`を確認してから次へ進む。欠損・重複・identity driftが
1件でもあればsummaryは書かれない。identityはpathではなくSHA-256で照合する。

## 15. summary生成

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py --config "$CONFIG"
```

出力は`f3_lithology_benchmark/mae_local_bt_five_way_v2/summary/`の5ファイル
（`comparison.csv` `paired_deltas.csv` `summary_by_size.csv` `summary.json`
`summary.md`）。主指標は`macro_f1`、sizeを跨いだ集計は行わない。

## 16. resume手順（runnerが実際に提供する契約）

- 各stageは自身の完了成果物があればskipされる。decoder学習が終わったあとに
  inferenceやevaluationで失敗したjobは、同じコマンドをそのまま再実行すれば
  decoderを再学習せずに続きから完了する。
- decoder学習の途中で中断した場合だけ、同じjobを
  `--resume <run_dir>/decoder/latest.pt`で再開する。`--resume`なしで再実行すると
  中断中である旨のエラーになる。resumeは同一job・同一supervision・同一decoder契約の
  checkpointだけを受理し、完了済みdecoderに対する`--resume`は拒否される。
- `evaluation/metrics.json`が存在するjobは完了扱いで、暗黙には上書きしない。
- `prediction/`と`evaluation/`はstaging経由で原子的に作られるため、部分的な
  directoryは残らない。作り直したい場合だけ、該当directoryを手で削除して再実行する。

## 対象テスト

```bash
pytest -q \
  tests/seis_ssl_cluster/test_f3_facies_benchmark_v2_configs.py \
  tests/seis_ssl_cluster/test_f3_prepared_volume_parity.py \
  tests/seis_ssl_cluster/test_f3_lithology_voxel_section_layout_calibration.py \
  tests/seis_ssl_cluster/test_f3_lithology_voxel_section_layout.py \
  tests/seis_ssl_cluster/test_f3_lithology_voxel_section_layout_config.py \
  tests/seis_ssl_cluster/test_f3_lithology_five_way_sources.py \
  tests/seis_ssl_cluster/test_f3_lithology_five_way_runner.py \
  tests/seis_ssl_cluster/test_f3_lithology_five_way_results.py
```
