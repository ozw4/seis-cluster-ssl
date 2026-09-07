# F3 overlapping-subcrop Local Barlow Twins PoC v1

overlapping-subcrop Local Barlow Twins を既存の random encoder と同じ F3 lithology 条件で比較し、次の探索へ進めるかを段階的に判定する PoC です。候補ごとの学習、embedding、downstream 条件は各 YAML、現在の状態と判定結果は [work summary](../../../../reports/f3/overlap_subcrop_local_bt_poc_v1_work_summary.md) を正本とします。

## 環境

`CANDIDATE` には実行する config の stem を指定します。
各 CLI が `--dry-run` を提供する場合は、同じ引数で検証してから実行します。

```bash
set -euo pipefail
export EXP=experiments/f3/facies_benchmark_v1/112_local_bt_overlap_subcrop_poc_v1
export CANDIDATE="${CANDIDATE:?set a candidate config stem}"
export POC_RUNS="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_bt_overlap_subcrop_poc_v1"
```

## Pretraining と embedding

feasibility を通した後、候補を fresh run として学習し、embedding を抽出します。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/$CANDIDATE.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/$CANDIDATE.yaml"
```

## Representation diagnostic

downstream より先に random と候補を同じ順序で測定します。random の既存診断が current source と一致する場合は再生成不要です。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/measure_f3_overlap_subcrop_representation.py \
  --config "$EXP/30_downstream/random_medium.yaml"
python proc/seis_ssl_cluster/measure_f3_overlap_subcrop_representation.py \
  --config "$EXP/30_downstream/${CANDIDATE}_medium.yaml"
```

## Screen と最終判定

まず screen 用 layout だけを両 arm で実行し、`decide.py --mode screen` を通します。pass した候補に限り、残りの configured layouts を実行して `--mode final` で判定します。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/run_f3_lithology_overlap_subcrop_poc.py \
	--config "$EXP/30_downstream/random_medium.yaml" --layout layout_001
python proc/seis_ssl_cluster/run_f3_lithology_overlap_subcrop_poc.py \
	--config "$EXP/30_downstream/${CANDIDATE}_medium.yaml" --layout layout_001
python "$EXP/decide.py" \
	--candidate-id "$CANDIDATE" --mode screen \
	--random-runs-root "$POC_RUNS/random/runs" \
	--candidate-runs-root "$POC_RUNS/$CANDIDATE/runs"
```

最終段階では同じ runner を各 remaining layout と両 arm に対して実行し、次を呼び出します。

```bash
set -euo pipefail
python "$EXP/decide.py" \
	--candidate-id "$CANDIDATE" --mode final \
	--random-runs-root "$POC_RUNS/random/runs" \
	--candidate-runs-root "$POC_RUNS/$CANDIDATE/runs"
```

## 再開

候補間で checkpoint を流用せず、各候補を fresh run として開始します。候補自身の未完了出力がある場合は、別候補から resume せず、その directory の状態を確認して同じ試行を再開するか新しい出力先でやり直してください。完了済み downstream cell は同じコマンドで検証・skip されます。
