# F3 lithology random-init HMM-K6 distillation 0.10 control v1

## 目的と範囲

canonical five-way v3 の HMM 系列（`mae_hmm_k6`、`local_barlow_twins_hmm_k6`）が
示す利得が、Stage 1 の自己教師あり表現に依存するのか、HMM pseudo-target への
Stage 2 適応だけで得られるのかを分離する control である。five-way の `random`
encoder（seed 42）を初期重みとして、同じ固定予算の HMM-K6 Stage 2（25 epoch）を
distillation weight `0.1` で学習し、canonical v3 の 15 cell（5 layouts × 3 sizes）で
frozen encoder として評価する。

## 系譜と契約

| 項目 | canonical `mae_hmm_k6` | この control |
| --- | --- | --- |
| student init | MAE 100 epoch (`stage1/mae/full_100ep`) | `mae_local_bt_five_way_v1/random/random_init.pt`（seed 42、epoch 0） |
| teacher | student init と同一 | student init と同一（random encoder） |
| pseudo target | `ssl_hmm_continuation_v1/mae100` K=6 | 同一 |
| Stage 2 予算 | 25 epoch、10,000 samples/epoch、batch 16、15,625 steps | 同一 |
| unfreeze / LR | encoder top-1、LR `1.0e-5`、AdamW wd 0.05、FP32、seed 42 | 同一 |
| loss weights | prototype / usage / distillation = 1.0 / 0.005 / 0.2 | 1.0 / 0.005 / **0.1** |
| 下流 | v3 class-balanced section layout、fixed decoder contract | 同一（candidate path） |

teacher と student init を同じ checkpoint にするのは five-way HMM 系列の契約
（`teacher.checkpoint == student.init_checkpoint`）に従う。random teacher への
distillation は「初期重みから離れすぎない」正則化としてだけ働くため、weight は
Parihaka / Volve の distillation screening と同じ `0.1` を採用する。
`student.unfreeze_top_blocks > 0` のとき resolver は正の distillation weight を要求
するので、`0.0` にはできない。

random checkpoint の `config` は参照 MAE checkpoint の resolved config を複製して
いるため、Stage 2 checkpoint の `config.continuation` にも MAE 系譜が写る。
初期重みの真の系譜は `stratigraphy_config.student.init_checkpoint`、
`stratigraphy_config.teacher.checkpoint`、および `control_identity.input_identities`
（SHA-256 付き）が記録する。embedding metadata の `stratigraphy_pretext` には
`distillation_weight: 0.1` と `model_tag` が入る。

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
export EXP=experiments/f3/facies_benchmark_v2/124_lithology_random_init_hmm_k6_distill010_v1
export STAGE2="$EXP/10_stage2/random_init/hmm/k6"
export CANDIDATE="$EXP/30_downstream/01_candidate.yaml"

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2/02_full_25ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$STAGE2/02_full_25ep.yaml"
```

embedding は v2 prepared volume（`facies_benchmark_v2` manifest）から、five-way v2
と同じ抽出契約（window `[128,128,128]`、overlap `[64,64,64]`、float16、`amp: false`、
min token valid fraction 0.5）で抽出する。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/01_extract_random_init_hmm_k6.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/01_extract_random_init_hmm_k6.yaml"
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
[`random_init_hmm_k6_distill010_control_v1.md`](../../../../reports/f3/random_init_hmm_k6_distill010_control_v1.md)
に集約し、producer の review file（random 比と five-way 比の summary）は
[`reports/f3/facies_benchmark_v2/random_init_hmm_k6_distill010_v1/`](../../../../reports/f3/facies_benchmark_v2/random_init_hmm_k6_distill010_v1/five_way/summary.md)
に置く。job matrix と結果値は README に転記しない。

## 再開

Stage 2 が途中で止まった場合は同じ config に `--resume <output_root>/latest.pt`
を渡す。別 output_root の checkpoint や smoke run の checkpoint は使わない。
cell の再開規則は [v3 runbook](../110_lithology_mae_local_bt_five_way_v3/README.md#再開)
と同じで、完了済み cell は identity check の上 skip され、decoder 途中断だけが
その cell 自身の `decoder/latest.pt` を `--resume` に取る。

## 対象テスト

```bash
pytest -q \
  tests/seis_ssl_cluster/test_f3_random_init_hmm_k6_distill010_configs.py \
  tests/seis_ssl_cluster/test_f3_random_init_hmm_k6_distill010_runbook.py \
  tests/seis_ssl_cluster/test_f3_lithology_candidate_five_way_summary.py
```
