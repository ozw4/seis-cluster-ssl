# F3 lithology MAE / Local Barlow Twins five-way v1

F3岩相推定で、次の5表現を同一のfrozen full-volume token embedding契約と
既存F3 voxel decoderで比較するsuiteである。

- `mae`
- `mae_hmm_k6`
- `local_barlow_twins`
- `local_barlow_twins_hmm_k6`
- `random`

`local_barlow_twins`はtrace-dropを使わない`local_barlow_twins_3d`
（`local_pairs_per_crop: 128`）である。trace-drop系のcheckpoint・embeddingは
このsuiteでは使用しない。

MAE系checkpointは既存fixed-budget artifactをそのまま参照し、再学習しない。

- `mae`: `pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/mae100/mae_continue/full_25ep/latest.pt`
- `mae_hmm_k6`: `pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/mae100/hmm/k6/full_25ep/latest.pt`

## 実験契約

- 下流評価: frozen encoder token embedding + 既存F3 voxel decoder
- layouts: `layout_000` `layout_001` `layout_002` `layout_003` `layout_004`
- data sizes: `small` `medium` `large`
- job数: 5 models × 5 layouts × 3 sizes = 75
- 主指標: validation unique voxels上の`macro_f1`
- 副指標: `mean_iou` `balanced_accuracy` `weighted_f1`
- 統計単位: `layout_id`（small/medium/largeは別々に集計する）
- decoder契約は`src/seis_ssl_cluster/config/f3_lithology_voxel_section_layout.py`
  の`FIXED_DECODER_CONTRACT`（50 epochs、batch 1、LR `1.0e-3`、seed 42000）に固定

## 環境

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export F3_ROOT=/home/dcuser/data/public_data/field/F3
export EXP=experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1
export CONFIG="$EXP/60_five_way.yaml"
```

## 1. upstream inputsの確認

次のliveな入力が揃っていることを確認する。

- F3 prepared amplitude manifest:
  `registry/manifests/f3/facies_benchmark_v1/f3_amplitude_manifest.json`
- Local BT 100 epoch checkpoint（`experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1`の成果物）:
  `pretraining/f3/facies_benchmark_v1/local_barlow_twins_v1/full_100ep/latest.pt`
- MAE fixed-budget checkpoints（上記2件、`21_ssl_hmm_continuation_v1`の成果物）
- model非依存のsection-layout supervision datasets（15条件）:
  `lithology/f3/facies_benchmark_v1/voxel_section_layout_v1/datasets/layout=*/size=*/voxel_supervision`
  （既存の`build_f3_lithology_voxel_section_layout_datasets`で構築されたもの）

## 2. Local BT固定予算control（25 epoch継続）

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$EXP/10_stage2/local_bt100/local_bt_continue/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$EXP/10_stage2/local_bt100/local_bt_continue/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$EXP/10_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$EXP/10_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml"
```

## 3. Local BT由来HMM-K6 target

```bash
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/20_hmm_targets/local_bt100/01_extract_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/20_hmm_targets/local_bt100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$EXP/20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$EXP/20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml"
bash "$EXP/20_hmm_targets/local_bt100/k6/03_export_pseudo_targets.sh"
```

## 4. Local BT + HMM-K6 Stage 2（25 epoch）

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/30_stage2/local_bt100/hmm/k6/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/30_stage2/local_bt100/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml"
```

## 5. Random checkpoint（Parihakaと同じseed 42）

```bash
python proc/seis_ssl_cluster/create_random_mae_checkpoint.py --config "$EXP/40_random/01_create_random_checkpoint.yaml" --dry-run
python proc/seis_ssl_cluster/create_random_mae_checkpoint.py --config "$EXP/40_random/01_create_random_checkpoint.yaml"
```

## 6. 五者downstream embedding抽出

`60_five_way.yaml`が参照するembeddingは、同じcheckpoint SHAと同じ抽出条件
（window `[128,128,128]`、overlap `[64,64,64]`、float16、`amp: false`、
min token valid fraction 0.5）で作られたmetadataを持つ場合だけ再利用できる。
互換artifactがなければ次で抽出する。

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

## 7. five-way source audit

decoderはsource auditがPASSするまで開始しない。

```bash
python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py --config "$CONFIG" --dry-run
python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py --config "$CONFIG"
```

read-only auditは五者のcheckpoint SHA、抽出契約、objective identity、
valid-token maskのbyte一致に加えて、固定予算(25 epoch / 15,625 global steps、
encoder top-1)とtrace-dropなしの系譜を検証する。`random`はepoch 0の未学習表現で
あることを確認する。

## 8. preflight（`layout_000/small`の五者dry-run）

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
`train_voxel_count`が一致することを確認する。

## 9. full suite（75 decoder jobs）

schedulerや独自queueは使わず、単純な3重loopで実行する。

```bash
for model in \
  mae \
  mae_hmm_k6 \
  local_barlow_twins \
  local_barlow_twins_hmm_k6 \
  random
do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004
  do
    for size in small medium large
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
`f3_lithology_benchmark/mae_local_bt_five_way_v1/runs/model=<model>/layout=<layout>/size=<size>/`
配下の`decoder/` `prediction/` `evaluation/`に分かれる。

中断jobの扱い(runnerが実際に提供する契約):

- 各stageは自身の完了成果物があればskipされる。したがってdecoder学習が終わった
  あとにinferenceやevaluationで失敗したjobは、同じコマンドをそのまま再実行すれば
  decoderを再学習せずに続きから完了する。
- decoder学習の途中で中断した場合だけ、同じjobを
  `--resume <run_dir>/decoder/latest.pt`で再開する。`--resume`なしで再実行すると
  中断中である旨のエラーになる。resumeは同一job・同一supervision・同一decoder契約の
  checkpointだけを受理し、完了済みdecoderに対する`--resume`は拒否される。
  1 epoch目の途中で落ちて`latest.pt`が無い場合や、`best.pt`と`latest.pt`の整合が
  取れずresumeが拒否された場合は、`<run_dir>/decoder`を削除して学習し直す。
- `evaluation/metrics.json`が存在するjobは完了扱いで、暗黙には上書きしない。
- `prediction/`と`evaluation/`はstaging経由で原子的に作られるため、部分的な
  directoryは残らない。作り直したい場合だけ、該当directoryを手で削除して再実行する。
- decoder出力を作り直す場合は`<run_dir>`配下の`decoder`、`prediction`、
  `evaluation`をすべて削除してから再実行する。`evaluation/metrics.json`が
  残っているとjobは完了扱いで拒否され、predictionだけ残すと、再学習した
  checkpointが元と異なる場合にevaluationがdecoder checkpoint hashの不一致で
  停止する。

## 10. summary dry-run（完全性監査）

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py --config "$CONFIG" --dry-run
```

`complete_jobs: 75`を確認してから次へ進む。欠損・重複・identity driftが
1件でもあればsummaryは書かれない。

identityはpathではなくSHA-256で照合する。各jobの`prediction_metadata.json`が
記録したencoder embedding・embedding metadata・valid token・decoder checkpointの
SHAを読み、同じmodelの15 jobsでencoder checkpointとembeddingのSHAが一致すること、
五者でvalid tokenのSHAが一致すること、五者のencoder checkpoint SHAが互いに異なる
ことを検証する。途中で同じpathにembeddingを再抽出した場合はここで停止する。
`metrics.json`の`aggregation_unit`が`unique_validation_voxel`であることも確認し、
別の集計単位のmetricsが混ざらないようにする。

## 11. summary生成

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py --config "$CONFIG"
```

出力は`f3_lithology_benchmark/mae_local_bt_five_way_v1/summary/`の5ファイルに
限定される。

- `comparison.csv`: 75 jobs × 1行（encoder checkpoint・embedding・valid token・
  decoder checkpointのSHA-256とsupervision identity付き）
- `paired_deltas.csv`: 同じ`layout/size`内のpaired delta（符号は常に`left - right`）
- `summary_by_size.csv`: size × comparison × metricごとの5 layouts集計
- `summary.json` / `summary.md`

## 結果の読み方

主指標は`macro_f1`である。`summary_by_size.csv`は各sizeについて
`mae_hmm_k6_minus_mae`、`local_bt_hmm_k6_minus_local_bt`、
`local_bt_minus_mae`、`local_bt_hmm_k6_minus_mae_hmm_k6`と、
`random`に対する4比較のpaired delta（n=5 layouts）のmean/median/
sample std/min/max/符号countを示す。sizeを跨いだ集計は行わない。

## 対象テスト

```bash
pytest -q \
  tests/seis_ssl_cluster/test_f3_local_barlow_twins_fixed_budget_configs.py \
  tests/seis_ssl_cluster/test_f3_local_barlow_twins_hmm_k6_target_configs.py \
  tests/seis_ssl_cluster/test_f3_local_barlow_twins_hmm_k6_stage2_configs.py \
  tests/seis_ssl_cluster/test_f3_lithology_random_baseline_config.py \
  tests/seis_ssl_cluster/test_f3_lithology_five_way_sources.py \
  tests/seis_ssl_cluster/test_f3_lithology_five_way_runner.py \
  tests/seis_ssl_cluster/test_f3_lithology_five_way_results.py \
  tests/seis_ssl_cluster/test_f3_lithology_five_way_runbook.py
```
