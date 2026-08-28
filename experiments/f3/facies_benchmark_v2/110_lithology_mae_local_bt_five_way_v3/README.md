# F3 lithology MAE / Local Barlow Twins five-way v3

F3岩相推定で`mae`、`mae_hmm_k6`、`local_barlow_twins`、
`local_barlow_twins_hmm_k6`、`random`を、v3のclass-balanced nested
section-layout supervision上で比較するsuiteである。

## 契約と再利用artifact

- prepared dataset identityは`f3_facies_benchmark` / `facies_benchmark_v2`。
  v3はsection-layoutとfive-way出力の世代名である。
- 5 models × 5 layouts × 3 sizes = 75 jobs。subsample seedは
  `layout_000..004`へ`0..4`を1対1に割り当てるため、seedのjob次元は追加しない。
- layout内のsmall/medium/largeはactive lines、選択token rows、teacher maskのすべてで
  nested。selection契約の詳細は
  [`../109_f3_voxel_section_layout_v3/README.md`](../109_f3_voxel_section_layout_v3/README.md)。
- v1の5 encoder checkpointsを再学習せず、v2のfive-way full-volume embeddingsを
  再抽出せずに参照する。prepared volume、label、geometry、inventoryもv2を再利用する。
- selection seed 0–4は教師row選択だけに使う。decoderは全jobで固定契約のseed 42000
  （50 epochs、batch 1、LR `1.0e-3`）を使い、selection seedとは独立である。
- v3はv1と同じper-class cap 25/50/100を使うが、nested追補選択のため旧v1の
  独立cap drawとexact same drawではない。
- 主指標はvalidation unique voxels上の`macro_f1`。統計単位は`layout_id`で、
  small/medium/largeは別々に集計する。

## 環境とsection-layout生成

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${F3_ROOT:?export F3 data root first}"
export SEIS_SSL_CLUSTER_WORKSPACE="${SEIS_SSL_CLUSTER_WORKSPACE:-/workspace}"
cd "$SEIS_SSL_CLUSTER_WORKSPACE"
export LAYOUT=experiments/f3/facies_benchmark_v2/109_f3_voxel_section_layout_v3
export EXP=experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v3
export CONFIG="$EXP/60_five_way.yaml"

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

section-layout出力は
`lithology/f3/facies_benchmark_v2/voxel_section_layout_v3/`、contractは同dataset
namespaceの
`voxel_section_layout_v3_calibration/`へ書く。v2 artifactは変更しない。
live生成コマンドは初回生成専用で、producerは既存rootを上書きしない。生成済み
artifactはdry-runとpreflightで検証し、再生成時も既存rootを無断で削除・移動しない。

## source audit

```bash
set -euo pipefail
python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py \
  --config "$CONFIG" --dry-run
python proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py \
  --config "$CONFIG"
```

checkpoint SHA、objective、固定予算、v2 embedding抽出契約、reference valid-token
identityが一致するまでdecoderを開始しない。

## five-model preflight

```bash
set -euo pipefail
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

五者で`condition_dir`、active lines、`subsample_seed`、selected token-row
identity/count、unique token identity/count、`train_voxel_count`、
`validation_mask_sha256`、`decoder_initial_state_sha256`が一致することを確認する。

## full suite（75 decoder jobs）

モデル差を最短で確認できるよう、loop順はsize → layout → modelに固定する。

```bash
set -euo pipefail
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

出力は
`f3_lithology_benchmark/mae_local_bt_five_way_v3/runs/model=<model>/layout=<layout>/size=<size>/`
配下へ書く。schedulerやjob managerは追加しない。

## summary

```bash
set -euo pipefail
python proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py \
  --config "$CONFIG" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py \
  --config "$CONFIG"
```

dry-runで`complete_jobs: 75`を確認する。summary出力は
`f3_lithology_benchmark/mae_local_bt_five_way_v3/summary/`。欠損、重複、selection
identity drift、validation identity driftが1件でもあれば生成しない。
