# F3 lithology random-init self-target HMM-K6 distillation 0.10 control v1

## 目的と範囲

[experiment 124](../124_lithology_random_init_hmm_k6_distill010_v1/README.md) は
five-way の `random` encoder（seed 42）を初期重みとして HMM-K6 Stage 2 を学習した
が、pseudo-target は canonical `mae_hmm_k6` と同じ MAE100 由来 K=6 target を再利用
していたため、MAE の情報が target の系譜として残っていた。この control は
pseudo-target の系譜と encoder 重みの両方から SSL（MAE / Barlow Twins）学習を完全に
外す。未学習の random encoder（seed 42、epoch 0）だけを使い、その embedding を
stratigraphic HMM で K=6 に clustering した target を作り、同じ random encoder を
teacher 兼 student init として、experiment 124 と同一の固定予算 HMM-K6 Stage 2
（25 epoch、distillation weight `0.1`）を学習し、canonical v3 の 15 cell
（5 layouts × 3 sizes）で frozen encoder として評価する。

experiment 124 との差分は pseudo-target の系譜（MAE100 由来か、random encoder 自身
由来か）だけなので、124 と 125 の対比は target 系譜の効果を分離する。canonical
`mae_hmm_k6` との対比では、Stage 1 の SSL 表現も target の系譜も持たない encoder
が、HMM pseudo-target への Stage 2 適応だけでどこまで到達するかを見る。

## 系譜と契約

| 項目 | canonical `mae_hmm_k6` | experiment 124 | この control |
| --- | --- | --- | --- |
| student init | MAE 100 epoch (`stage1/mae/full_100ep`) | `mae_local_bt_five_way_v1/random/random_init.pt`（seed 42、epoch 0） | 124 と同一（random encoder） |
| teacher | student init と同一 | student init と同一（random encoder） | student init と同一（random encoder） |
| pseudo target | `ssl_hmm_continuation_v1/mae100` K=6 | 同一（MAE100 由来） | `random_init_self_target_hmm_k6_distill010_v1/random_init` K=6（random encoder 自身の embedding 由来） |
| Stage 2 予算 | 25 epoch、10,000 samples/epoch、batch 16、15,625 steps | 同一 | 同一 |
| unfreeze / LR | encoder top-1、LR `1.0e-5`、AdamW wd 0.05、FP32、seed 42 | 同一 | 同一 |
| loss weights | prototype / usage / distillation = 1.0 / 0.005 / 0.2 | 1.0 / 0.005 / **0.1** | 1.0 / 0.005 / **0.1** |
| 下流 | v3 class-balanced section layout、fixed decoder contract | 同一（candidate path） | 同一（candidate path） |

pseudo-target pipeline は canonical MAE100 target
（`experiments/f3/facies_benchmark_v1/21_ssl_hmm_continuation_v1/20_hmm_targets/mae100`）
と同じ契約に従い、checkpoint と入出力 path だけが異なる。embedding は v1 prepared
volume（`facies_benchmark_v1` manifest）から window `[128,128,128]`、overlap
`[64,64,64]`、float16、`amp: false`、min token valid fraction 0.5 で抽出し、
clustering YAML は `stratigraphic_hmm_kmeans`（K=6、l2 normalization、
`local_token_position` residualization、PCA 64、1,000,000 sample tokens、seed 42）
を mae100 版と `input_dir` / `output_dir` 以外 byte 単位で同一にしている。export は
k 6、confidence 1.0、boundary alpha 0.0、boundary tau 1.0、schema version 2 で、
pseudo-target root
`pseudo_targets/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/random_init`
の下に `k6/` を作る。Stage 2 の `pseudo_targets.input_dir` はこの root を指す。

teacher と student init を同じ checkpoint にするのは five-way HMM 系列の契約
（`teacher.checkpoint == student.init_checkpoint`）に従う。random teacher への
distillation は「初期重みから離れすぎない」正則化としてだけ働くため、weight は
experiment 124 と同じ `0.1` を採用する。`student.unfreeze_top_blocks > 0` のとき
resolver は正の distillation weight を要求するので、`0.0` にはできない。

random checkpoint の `config` は参照 MAE checkpoint の resolved config を複製して
いる（geometry の参照だけに使う。重みは random で、
`metadata.random_encoder_baseline: true`、`metadata.pretrained_weights_loaded: false`）。
そのため target 抽出 embedding の metadata と Stage 2 checkpoint の
`config.continuation` には MAE 系譜が写る。下流 v2 embedding metadata の
`stratigraphy_pretext.base_objective`（`amp_mae3d`）と `pretraining_objective` も、
この複製 config の `stage: train_amp_mae` から導出されるため control の系譜を表さない
（candidate source audit の identity key には含まれず、判定には影響しない）。
初期重みと target の真の系譜は
`stratigraphy_config.student.init_checkpoint`、`stratigraphy_config.teacher.checkpoint`、
`stratigraphy_config.pseudo_targets.input_dir`、および
`control_identity.input_identities`（SHA-256 付き）が記録し、embedding metadata
側では `stratigraphy_pretext.pseudo_target_input_dir` と `model_tag` が正典である。
`identity.scientific_identity` には `experiment_role: random_init_self_target_control`
と `pseudo_target_source: random_init_self_target_hmm_k6_distill010_v1/random_init/k6`
を入れ、embedding metadata の `stratigraphy_pretext` には
`distillation_weight: 0.1` と `model_tag`
（`f3_random_init_self_target_hmm_k6_distill010_topblock1_v1`）が入る。

この control は five-way の固定 model set には含めない。source audit が要求する
100 epoch Stage 1 系譜と distillation `0.2` を満たさないため、canonical five-way の
runner ではなく candidate path（`run_f3_lithology_candidate.py`）で評価し、
`summarize_f3_lithology_candidate_five_way.py` で five-way 5 model との paired
delta を集計する。

## 実行順

[shared F3 v2 environment](../README.md) を先に設定する。下流 evaluation は
canonical v3 config の `labels.png_label_inventory`
（`${SEIS_SSL_CLUSTER_WORKSPACE}/experiments/f3/facies_benchmark_v2/10_prepare/section_inventory_v2.csv`）
を section-layout dataset が記録した inventory path と照合するため、
`SEIS_SSL_CLUSTER_WORKSPACE` には v3 dataset を build したときと同じ repository root
を与える（別 worktree の path では `inventory path` identity mismatch で停止する）。

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${F3_ROOT:?export F3 data root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export repository root first}"
export EXP=experiments/f3/facies_benchmark_v2/125_lithology_random_init_self_target_hmm_k6_distill010_v1
export TARGETS="$EXP/10_hmm_targets/random_init"
export STAGE2="$EXP/20_stage2/random_init/hmm/k6"
export CANDIDATE="$EXP/40_downstream/01_candidate.yaml"
```

pseudo-target は random encoder 自身の embedding から作る。抽出、clustering、export
の順を守り、MAE100 や Barlow Twins 由来の target lineage を混在させない。export
script は `--dry-run` を passthrough する。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TARGETS/01_extract_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TARGETS/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$TARGETS/k6/02_cluster_hmm_k6.yaml" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$TARGETS/k6/02_cluster_hmm_k6.yaml"
bash "$TARGETS/k6/03_export_pseudo_targets.sh" --dry-run
bash "$TARGETS/k6/03_export_pseudo_targets.sh"
```

export 後に Stage 2 を feasibility（1 step smoke）、full の順で実行する。smoke で
random teacher / student の受け入れと `control_identity` の記録を確認してから full
に進む。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2/02_full_25ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2/02_full_25ep.yaml"
```

下流用 embedding は v2 prepared volume（`facies_benchmark_v2` manifest）から、
five-way v2 と同じ抽出契約（window `[128,128,128]`、overlap `[64,64,64]`、float16、
`amp: false`、min token valid fraction 0.5）で抽出する。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/30_embeddings/01_extract_random_init_self_target_hmm_k6_distill010.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/30_embeddings/01_extract_random_init_self_target_hmm_k6_distill010.yaml"
```

candidate source audit（canonical `random` と抽出契約・valid-token mask の一致）を
含む preflight の後、15 cell を実行する。cell 単位で独立なので並列実行できる。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
  --config "$CANDIDATE" --layout layout_000 --size small --dry-run
for size in small medium large; do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
    python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
      --config "$CANDIDATE" --layout "$layout" --size "$size"
  done
done
```

15 cell 完了後、canonical `random` との paired summary と、five-way 5 model 全部
との paired summary を生成する。前者は `outputs.summary_root`、後者は
`outputs.five_way_summary_root` に書かれ、既存 directory は上書きしない。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/summarize_f3_lithology_candidate.py \
  --config "$CANDIDATE"
python proc/seis_ssl_cluster/summarize_f3_lithology_candidate_five_way.py \
  --config "$CANDIDATE" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_candidate_five_way.py \
  --config "$CANDIDATE"
```

## 結果

数値、解釈、系譜 SHA は
[`random_init_self_target_hmm_k6_distill010_control_v1.md`](../../../../reports/f3/random_init_self_target_hmm_k6_distill010_control_v1.md)
に集約する。producer の review file（random 比と five-way 比の summary）は
[`reports/f3/facies_benchmark_v2/random_init_self_target_hmm_k6_distill010_v1/`](../../../../reports/f3/facies_benchmark_v2/random_init_self_target_hmm_k6_distill010_v1/five_way/summary.md)
に置く。job matrix と結果値は README に転記しない。

## 再開

Stage 2 が途中で止まった場合は同じ config に `--resume <output_root>/latest.pt`
を渡す。別 output_root の checkpoint や smoke run の checkpoint は使わない。
target pipeline（抽出、clustering、export）は途中生成物を流用せず、抽出から順に
最初からやり直す。やり直す前に各 stage の output directory
（`embeddings/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/hmm_targets/random_init/overlap_x64`、
`clustering/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/hmm_targets/random_init/k6`、
`pseudo_targets/f3/facies_benchmark_v1/random_init_self_target_hmm_k6_distill010_v1/random_init/k6`）
を削除する。抽出 CLI は `--skip-existing` を付けない限り既存の完了済み出力を捨てて
再抽出し（metadata が一致しない既存出力は `ValueError` で停止）、clustering は既存の
`k6` 出力を atomic に置き換えるが、export script は既存 `k6/` があると `--dry-run`
でも `FileExistsError` で停止する。削除せずに上書きする場合だけ passthrough で
`--overwrite` を渡す。cell の再開規則は
[v3 runbook](../110_lithology_mae_local_bt_five_way_v3/README.md#再開) と同じで、
完了済み cell は identity check の上 skip され、decoder 途中断だけがその cell 自身の
`decoder/latest.pt` を `--resume` に取る。

## 対象テスト

```bash
pytest -q \
  tests/seis_ssl_cluster/test_f3_random_init_self_target_hmm_k6_distill010_configs.py \
  tests/seis_ssl_cluster/test_f3_random_init_self_target_hmm_k6_distill010_runbook.py \
  tests/seis_ssl_cluster/test_f3_lithology_candidate_five_way_summary.py
```
