# F3 overlapping-subcrop Local Barlow Twins PoC v1

This experiment compares a fresh 10-epoch overlapping-subcrop Local Barlow
Twins encoder with the existing seed-42 random encoder. The downstream contract
is the existing frozen-encoder decoder, inference, and evaluation path; this
PoC deliberately skips the five-way source audit.

## Fixed contract

- candidate: `shift04_proj384_pairs128_lambda005`
- view crop / patch: `[128, 128, 128]` / `[8, 8, 8]`
- maximum subcrop shift: `[4, 4, 0]` tokens
- encoder / projector: 384 / 384
- local pairs / redundancy weight: 128 / `0.005`
- epochs / samples per epoch / batch: 10 / 10,000 / 16
- planned optimizer steps: 6,250
- initialization / seed: fresh random weights / 42
- downstream size / decoder seed: medium / 42000
- primary metric: `macro_f1` on unique validation voxels

The downstream conditions use the current class-balanced v3 layout producer at
`experiments/f3/facies_benchmark_v2/109_f3_voxel_section_layout_v3` and its
artifact root
`lithology/f3/facies_benchmark_v2/voxel_section_layout_v3`. Both arms therefore
use identical `layout_000` through `layout_004` conditions. The v1 section-layout
artifact named by the historical five-way v1 config is not used.

The candidate checkpoint keeps the PoC's requested v1 pretraining namespace.
Embedding extraction reads the v2 prepared-amplitude manifest and writes below
`embeddings/f3/facies_benchmark_v2/local_bt_overlap_subcrop_poc_v1/<candidate>/local_barlow_twins/overlap_x64`.
The v2 dataset prefix and final `local_barlow_twins/overlap_x64` segments are
required by the frozen decoder's source-provenance contract. This matches the
existing v2 random embedding used by the v3 downstream contract. The prepared
v1 and v2 amplitude arrays are byte-identical, but their metadata identities
remain versioned.

## Environment

```bash
cd /workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export F3_ROOT=/home/dcuser/data/public_data/field/F3
export SEIS_SSL_CLUSTER_WORKSPACE=/workspace
export EXP=experiments/f3/facies_benchmark_v1/112_local_bt_overlap_subcrop_poc_v1
export CANDIDATE=shift04_proj384_pairs128_lambda005
```

The following existing random sources are reused:

```text
pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/random/random_init.pt
embeddings/f3/facies_benchmark_v2/mae_local_bt_five_way_v2/random/overlap_x64
```

## Pretraining and embedding

Run the one-step feasibility check before the full fresh run. Do not add a
continuation or resume from another candidate.

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/01_gpu_feasibility_1step.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/01_gpu_feasibility_1step.yaml"

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/$CANDIDATE.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$EXP/10_pretraining/$CANDIDATE.yaml"

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/$CANDIDATE.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/20_embeddings/$CANDIDATE.yaml"
```

The full endpoint is
`pretraining/f3/facies_benchmark_v1/local_bt_overlap_subcrop_poc_v1/$CANDIDATE/latest.pt`.
It is complete only at epoch 10 and global step 6,250.

## Mandatory representation diagnostics

Measure the Random baseline once and every candidate immediately after its
embedding extraction, before the downstream screen:

```bash
python proc/seis_ssl_cluster/measure_f3_overlap_subcrop_representation.py \
  --config "$EXP/30_downstream/random_medium.yaml" --dry-run
python proc/seis_ssl_cluster/measure_f3_overlap_subcrop_representation.py \
  --config "$EXP/30_downstream/random_medium.yaml"

python proc/seis_ssl_cluster/measure_f3_overlap_subcrop_representation.py \
  --config "$EXP/30_downstream/${CANDIDATE}_medium.yaml" --dry-run
python proc/seis_ssl_cluster/measure_f3_overlap_subcrop_representation.py \
  --config "$EXP/30_downstream/${CANDIDATE}_medium.yaml"
```

The CLI infers the Random or Local BT source from the existing downstream
config filename. It adds no fourth candidate YAML and exposes no sampling or
calculation override. Candidate checkpoints must be completed epoch-10,
global-step-6,250 endpoints; Random must be the existing untrained seed-42
checkpoint. Embedding metadata must bind the exact checkpoint SHA-256.

The fixed sample is 8,192 valid tokens from the `[76, 113, 32]` F3 token grid.
Valid token flat indices are ordered in XYZ C order and selected by midpoint
systematic sampling. The valid-mask and sampled-index SHA-256 values are fixed
to the Random baseline, so every candidate uses the same physical token
coordinates. Inputs are the downstream-consumed, overlap-merged `float16` bare
encoder embeddings; calculations use `float64`.

For sampled features `X[token, feature]`, the recorded metrics are:

- `raw_feature_norm`: mean token L2 norm.
- `token_wise_feature_std`: mean across feature dimensions of population
  standard deviation across tokens.
- `raw_feature_effective_rank`: exponential entropy of the eigenvalues of the
  feature-dimension-centered population covariance.
- `layer_norm_feature_std`: the same feature standard deviation after
  per-token, affine-free LayerNorm with population variance and epsilon
  `1e-5`.
- `layer_norm_effective_rank`: the same covariance effective rank after that
  LayerNorm.

Each JSON contains exactly those five metric keys plus calculation, sampling,
and source hashes. Outputs are written atomically to
`diagnostics/f3/local_bt_overlap_subcrop_poc_v1/representation/<candidate_id>.json`;
the Random result is `random.json`. Copy the five candidate values and JSON
path into that candidate's `search_results.csv` row when the runtime row is
created.

## Downstream screen and decision

The thin PoC CLI intentionally has no `--model` or `--size` option. It maps
`random_medium.yaml` to the `random` slot and every
`<candidate_id>_medium.yaml` to the `local_barlow_twins` slot, fixes size to
`medium`, and validates that the output namespace matches the filename.

```bash
python proc/seis_ssl_cluster/run_f3_lithology_overlap_subcrop_poc.py \
  --config "$EXP/30_downstream/random_medium.yaml" \
  --layout layout_001
python proc/seis_ssl_cluster/run_f3_lithology_overlap_subcrop_poc.py \
  --config "$EXP/30_downstream/${CANDIDATE}_medium.yaml" \
  --layout layout_001

python "$EXP/decide.py" \
  --candidate-id "$CANDIDATE" \
  --mode screen \
  --random-runs-root \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_bt_overlap_subcrop_poc_v1/random/runs" \
  --candidate-runs-root \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_bt_overlap_subcrop_poc_v1/$CANDIDATE/runs"
```

A strict `layout_001` win exits 0. A tie or loss writes `screen_decision.json`
beside the candidate `runs/` directory and exits 1.

Only after a screen pass, run the other four layouts for both arms. The final
decision reads all five layouts; at least four strict wins exits 0 and marks the
candidate adopted. Otherwise it writes `final_decision.json` and exits 1.

```bash
for layout in layout_000 layout_002 layout_003 layout_004
do
  python proc/seis_ssl_cluster/run_f3_lithology_overlap_subcrop_poc.py \
    --config "$EXP/30_downstream/random_medium.yaml" --layout "$layout"
  python proc/seis_ssl_cluster/run_f3_lithology_overlap_subcrop_poc.py \
    --config "$EXP/30_downstream/${CANDIDATE}_medium.yaml" --layout "$layout"
done

python "$EXP/decide.py" \
  --candidate-id "$CANDIDATE" \
  --mode final \
  --random-runs-root \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_bt_overlap_subcrop_poc_v1/random/runs" \
  --candidate-runs-root \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/f3_lithology_benchmark/local_bt_overlap_subcrop_poc_v1/$CANDIDATE/runs"
```

Decision JSON retains `macro_f1`, `mean_iou`, `balanced_accuracy`,
`weighted_f1`, evaluation voxel counts, paired deltas, and strict-win status for
each evaluated layout. Only `macro_f1` controls either gate.

Future search candidates add exactly one pretraining, embedding, and downstream
YAML using the full candidate ID as the filename and output namespace. Every
candidate remains a fresh seed-42 10-epoch run.
