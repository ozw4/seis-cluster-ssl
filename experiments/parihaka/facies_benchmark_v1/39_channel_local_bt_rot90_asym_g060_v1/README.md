# Parihaka Channel local-BT rot90 asymmetric-noise transfer v1

Transfer of the strongest F3 local Barlow Twins recipe to the Parihaka volume,
evaluated as a frozen embedding on all 15 fixed Channel benchmark cells
(5 layouts x 3 supervision sizes). Pretraining recipe: scratch training for
three epochs with XY right-angle rotations, asymmetric Gaussian noise at
standard deviation 0.6, and `[2, 2, 1]` token-region positives.

Only the survey input changes from the F3 recipe. Parihaka amplitude
preprocessing, encoder geometry, embedding extraction, Channel layouts,
supervision sizes, decoder, optimizer, and seed follow the existing benchmark
contracts. Comparators: the current Parihaka local-BT reference and the legacy
Random result, reused after result-identity validation.

## Results

Test Channel IoU for all 15 candidate cells and both comparators is recorded in
the curated
[summary](../../../../reports/parihaka/facies_benchmark_v1/local_bt_rot90_asym_g060_v1/summary.md),
[comparison CSV](../../../../reports/parihaka/facies_benchmark_v1/local_bt_rot90_asym_g060_v1/comparison.csv),
and [summary JSON](../../../../reports/parihaka/facies_benchmark_v1/local_bt_rot90_asym_g060_v1/summary.json).

Artifact facts (model `local_barlow_twins_rot90_asym_g060_3ep`):

| Artifact | Value |
| --- | --- |
| Pretraining completion | epoch 3 / global step 1,875, final loss `0.580278` |
| `latest.pt` SHA-256 | `a314e19a29dd1510878aacfff4ed761e16c335688d3b4f902904557b151221ee` |
| Embedding dtype / shape | finite `float16`, `[98, 74, 126, 384]` |
| Embedding SHA-256 | `42f14f53d907133ad213a69dbd64c88d7fec9a3640e48a776e30292b19bb34ec` |
| Validity mask | 913,751 tokens |
| Validity mask SHA-256 | `729f057c97aa3e3f0793d489976608eac9b33c0fad794918425ad698d9390774` |

## Execution

Set the shared artifact root, then dry-run each stage before writing outputs.

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export CUDA_VISIBLE_DEVICES=0
export EXP=experiments/parihaka/facies_benchmark_v1/39_channel_local_bt_rot90_asym_g060_v1
export LAYOUTS=experiments/parihaka/facies_benchmark_v1/30_channel_benchmark_v1/02_layouts.yaml

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/01_full_3ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/01_full_3ep.yaml"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/01_extract_embeddings.yaml" \
  --device cuda --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/01_extract_embeddings.yaml" \
  --device cuda --skip-existing
```

Audit and run the candidate cells in the fixed layout and size order.

```bash
for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
  for size in small medium large; do
    python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
      --config "$EXP/30_downstream/01_channel_candidate.yaml" \
      --model local_barlow_twins_rot90_asym_g060_3ep \
      --layout "$layout" --size "$size" --layout-config "$LAYOUTS" \
      --device cuda --dry-run
  done
done

for layout in layout_000 layout_001 layout_002 layout_003 layout_004; do
  for size in small medium large; do
    model=local_barlow_twins_rot90_asym_g060_3ep
    job_dir="${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/channel_benchmark/ssl_hmm_four_way_v1/runs/model=${model}/layout=${layout}/size=${size}"
    if [[ -f "$job_dir/metrics.json" ]]; then
      echo "skip complete job: $layout/$size"
      continue
    fi
    resume_args=()
    if [[ -f "$job_dir/latest.pt" ]]; then
      resume_args=(--resume "$job_dir/latest.pt")
    fi
    python proc/seis_ssl_cluster/run_parihaka_channel_decoder.py \
      --config "$EXP/30_downstream/01_channel_candidate.yaml" \
      --model "$model" \
      --layout "$layout" --size "$size" --layout-config "$LAYOUTS" \
      --device cuda "${resume_args[@]}"
  done
done
```

After all 15 candidate metrics exist, audit both learned models, the canonical
legacy Random benchmark, and the cross-root downstream identity before writing
the exact three-file summary set.

```bash
python proc/seis_ssl_cluster/summarize_parihaka_channel_recipe_transfer.py \
  --config "$EXP/30_downstream/01_channel_candidate.yaml" --check-only
python proc/seis_ssl_cluster/summarize_parihaka_channel_recipe_transfer.py \
  --config "$EXP/30_downstream/01_channel_candidate.yaml"
```

Complete execution outputs and downstream inputs remain below `artifacts/`;
only the summary producer's small CSV, JSON, and Markdown files may be curated
into `reports/`, and pipeline stages must not consume report files.
