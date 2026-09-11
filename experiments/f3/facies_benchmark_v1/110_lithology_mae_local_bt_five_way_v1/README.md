# F3 lithology MAE / Local Barlow Twins five-way v1

同じ F3 lithology supervision と decoder 契約で、MAE、Local Barlow Twins、それぞれの HMM 継続、random encoder を比較する実験です。model、layout、data size、入出力と集計条件は [60_five_way.yaml](60_five_way.yaml) を正本とします。

MAE 系の生成手順は [21_ssl_hmm_continuation_v1](../21_ssl_hmm_continuation_v1/RUNBOOK_HMM_K6.md)、Local Barlow Twins の Stage 1 は [22_local_barlow_twins_v1](../22_local_barlow_twins_v1/README.md) を参照してください。この文書では本 experiment が生成する残りの source と downstream の順序だけを扱います。

## Source production

Run every supported write command with `--dry-run` first.

```bash
set -euo pipefail
export EXP=experiments/f3/facies_benchmark_v1/110_lithology_mae_local_bt_five_way_v1

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_stage2/local_bt100/local_bt_continue/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_stage2/local_bt100/local_bt_continue/02_full_25ep.yaml"
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_hmm_targets/local_bt100/01_extract_embeddings.yaml"
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$EXP/20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml"
bash "$EXP/20_hmm_targets/local_bt100/k6/03_export_pseudo_targets.sh" --dry-run
bash "$EXP/20_hmm_targets/local_bt100/k6/03_export_pseudo_targets.sh"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/local_bt100/hmm/k6/01_gpu_feasibility_1step.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/30_stage2/local_bt100/hmm/k6/02_full_25ep.yaml"
python proc/seis_ssl_cluster/create_random_mae_checkpoint.py \
  --config "$EXP/40_random/01_create_random_checkpoint.yaml"
```

## Historical v1 evaluation

すべての upstream checkpoint が揃ってから v1 embedding を抽出します。

```bash
set -euo pipefail
for config in "$EXP"/50_embeddings/*.yaml; do
  python proc/seis_ssl_cluster/extract_embeddings.py --config "$config" --dry-run
  python proc/seis_ssl_cluster/extract_embeddings.py --config "$config"
done
```

Historical source audit, cell execution, summary, and recovery use the
[current v3 sequence](../../facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v3/README.md)
with this directory's [60_five_way.yaml](60_five_way.yaml). The v1 configuration
and result namespace must remain distinct from v3.
