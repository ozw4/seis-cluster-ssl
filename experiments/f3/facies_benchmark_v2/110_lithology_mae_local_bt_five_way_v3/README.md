# F3 lithology MAE / Local Barlow Twins five-way v3

five-way v2 の checkpoint と embedding を変えず、class-balanced nested な [section-layout v3](../109_f3_voxel_section_layout_v3/README.md) だけへ切り替える比較です。v3 固有の binding と出力先は [60_five_way.yaml](60_five_way.yaml) を正本とし、prepared data と source の生成は [five-way v2](../110_lithology_mae_local_bt_five_way_v2/README.md) に集約しています。

## 実行順

section-layout v3 の inspection、finalize、build を完了してから source audit を行います。audit pass 後、最初の configured cell を preflight します。[shared F3 v2 environment](../README.md) を先に設定します。

```bash
set -euo pipefail
export EXP=experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v3
export CONFIG="$EXP/60_five_way.yaml"

python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py \
	--config "$CONFIG" --dry-run
python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py \
	--config "$CONFIG"

MODEL=mae
LAYOUT=layout_000
SIZE=small
python proc/seis_ssl_cluster/run_f3_lithology_five_way.py \
	--config "$CONFIG" --model "$MODEL" --layout "$LAYOUT" --size "$SIZE" --dry-run
```

上の cell command は preflight 例です。configured matrix の全組合せを同じ
one-cell runner で実行し、すべて完了してから summary を監査・生成します。

```bash
set -euo pipefail
python proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py \
	--config "$CONFIG" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py \
	--config "$CONFIG"
```

生成 summary が結果の正本です。job matrix と結果値は README に転記しません。

## 再開

完了済み cell は同じ command で identity-check されて skip されます。inference または evaluation だけが中断した場合は同じ cell を再実行します。decoder の途中で中断した場合だけ、その cell 自身の `decoder/latest.pt` を `--resume` に渡します。別 cell の checkpoint を使わず、既存 summary を暗黙に上書きしません。
