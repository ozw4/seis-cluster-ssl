# F3 Local VICReg 15-epoch resume PoC v1

## 目的と範囲

[10-epoch PoC](../117_local_vicreg_10ep_poc_v1/README.md) の完了 checkpoint を
同じ trajectory のまま延長し、同じ `medium / layout_001` で epoch 増加方向の
変化だけを測る follow-up である。fresh な 15-epoch 学習や full benchmark は
この実験の問いではない。

## 系譜

学習 config は 117 の条件を継承し、`--resume` で optimizer、RNG、dataloader
generator を復元する。抽出と評価も 117 と同じ F3 v2 / canonical five-way v3
契約を使う。正確な予算と入出力は YAML、resume 互換条件は resolver が定義し、
test はそれらを検証する。

## 状態

resume workflow と三つの config は実装・検証済みだが、15-epoch の下流結果は
リポジトリに記録されておらず、結果 report もない。したがって改善・悪化の結論は
まだ付けない。

## 実行

[shared F3 v2 environment](../README.md) を設定し、`POC10_CHECKPOINT` には 117
の完了 checkpoint を明示する。各 live command の前に同じ config で dry-run する。

```bash
set -euo pipefail
: "${POC10_CHECKPOINT:?set path to experiment 117 latest.pt}"
export EXP=experiments/f3/facies_benchmark_v2/118_local_vicreg_15ep_resume_poc_v1

python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$EXP/01_full_15ep_resume.yaml" \
  --resume "$POC10_CHECKPOINT" --dry-run
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$EXP/01_full_15ep_resume.yaml" \
  --resume "$POC10_CHECKPOINT"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/02_extract_v2_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/02_extract_v2_embeddings.yaml" --skip-existing

python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
  --config "$EXP/03_candidate_medium_layout_001.yaml" \
  --layout layout_001 --size medium --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
  --config "$EXP/03_candidate_medium_layout_001.yaml" \
  --layout layout_001 --size medium
```
