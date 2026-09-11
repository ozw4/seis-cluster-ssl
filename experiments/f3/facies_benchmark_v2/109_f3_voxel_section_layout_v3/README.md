# F3 voxel section-layout v3

F3 five-way v3 用の model 非依存 section-layout supervision を生成します。[v2 layout](../109_f3_voxel_section_layout_v2/README.md) の positional selection で生じた class coverage の偏りを、seeded・class-balanced・strictly nested な token-row 選択へ置き換える版です。prepared volume と canonical supervision は read-only で再利用します。

選択規則、line order、seed、class cap、target と出力先は [01_prepare_section_layout_contract.yaml](01_prepare_section_layout_contract.yaml)、[02_layout_lines.yaml](02_layout_lines.yaml)、[03_build_section_layout_datasets.yaml](03_build_section_layout_datasets.yaml) を正本とします。

## 実行順

inspection、contract finalize、dataset build の順です。finalize が class coverage と nesting を検証するため、手作業で生成物を補正しません。
[shared F3 v2 environment](../README.md) を先に設定します。

```bash
set -euo pipefail
export LAYOUT=experiments/f3/facies_benchmark_v2/109_f3_voxel_section_layout_v3

python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
	--config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode inspect --dry-run
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
	--config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode inspect
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
	--config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode finalize --dry-run
python proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py \
	--config "$LAYOUT/01_prepare_section_layout_contract.yaml" --mode finalize
python proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py \
	--config "$LAYOUT/03_build_section_layout_datasets.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py \
	--config "$LAYOUT/03_build_section_layout_datasets.yaml"
```

finalized contract と dataset manifest が実測値の正本です。生成済み output root は上書きせず、再生成時は新しい versioned root を使って downstream config と一緒に切り替えてください。
