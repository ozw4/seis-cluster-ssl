# Local VICReg lithology extension v1

[Local VICReg v1](../115_local_vicreg_v1/README.md) を既存 five-way benchmark に追加する、screening gate 付き downstream suite です。既存 five-way runs は read-only で再利用し、VICReg 固有の実行だけを追加します。source、matrix、入出力と gate 条件は [60_extension.yaml](60_extension.yaml) および runner / summarizer を正本とします。

## Screening

Stage 1 baseline と embedding が揃ったら、source audit、configured screening cells、screen summary の順で実行します。最初の candidate cell は live 実行前に dry-run してください。

```bash
set -euo pipefail
export EXP=experiments/f3/facies_benchmark_v2/116_local_vicreg_extension_v1
export CONFIG="$EXP/60_extension.yaml"

python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode screening-source --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode screening-source

python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --suite screening --model local_vicreg_100 \
	--layout layout_000 --size medium --dry-run
```

上の cell command は preflight 例です。configured screening cells を同じ
one-cell runner で全件実行してから、screening summary を監査・生成します。

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode screening --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode screening
```

screening summary が判定結果の正本です。pass した場合だけ [115 runbook](../115_local_vicreg_v1/README.md#control-branch) に戻って control と HMM branch を生成します。

## Extension

両 branch の checkpoint が揃った後に embedding を抽出し、source audit、最初の extension cell の preflight、全 configured cells、summary の順で進めます。

```bash
set -euo pipefail
for config in "$EXP"/50_embeddings/*.yaml; do
	python proc/seis_ssl_cluster/extract_embeddings.py --config "$config" --dry-run
	python proc/seis_ssl_cluster/extract_embeddings.py --config "$config"
done
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode sources --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode sources

python proc/seis_ssl_cluster/run_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --suite extension --model local_vicreg \
	--layout layout_000 --size small --dry-run
```

上の cell command は preflight 例です。configured extension cells を同じ
one-cell runner で全件実行してから、extension と combined summary を監査・生成します。

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode extension --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode extension
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode combined --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_vicreg_extension.py \
	--config "$CONFIG" --mode combined
```

extension summary と combined summary が結果の正本です。job 数、model/layout/size の一覧、結果値と output inventory は README に複製しません。

## 再開

完了済み cell は同じ command で identity-check されて skip されます。decoder の途中で中断した場合だけ、同じ suite/model/layout/size cell 自身の `decoder/latest.pt` を `--resume` に渡します。別 cell の checkpoint を使わず、既存 summary を上書きしないでください。
