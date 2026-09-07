# F3 Local VICReg 10-epoch PoC v1

## 目的と範囲

F3 の未ラベル振幅で Local VICReg を random initialization から短期学習し、
F3 v2 の `medium / layout_001` だけで Random encoder と比較する PoC である。
HMM continuation、他の layout / size、full benchmark は対象外とする。

## 系譜

学習は `115_local_vicreg_v1` の正典条件から学習期間と出力先だけを分け、
embedding 抽出と下流評価は
`110_lithology_mae_local_bt_five_way_v3` のデータ分割・decoder 契約を再利用する。
正確な条件、入出力、候補 identity はこのディレクトリの YAML を正本とし、対応する
config test が正典との差分を検証する。

## 状態

評価結果と展開判断は
[canonical result report](../../../../reports/f3/local_vicreg_10ep_poc_v1.md)
だけに記録する。

## 実行

[shared F3 v2 environment](../README.md) を設定して実行する。各 live command
の前に同じ config で dry-run する。

```bash
set -euo pipefail
export EXP=experiments/f3/facies_benchmark_v2/117_local_vicreg_10ep_poc_v1

python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$EXP/02_full_10ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_vicreg.py \
  --config "$EXP/02_full_10ep.yaml"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/03_extract_v2_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/03_extract_v2_embeddings.yaml" --skip-existing

python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
  --config "$EXP/04_candidate_medium_layout_001.yaml" \
  --layout layout_001 --size medium --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_candidate.py \
  --config "$EXP/04_candidate_medium_layout_001.yaml" \
  --layout layout_001 --size medium
```
