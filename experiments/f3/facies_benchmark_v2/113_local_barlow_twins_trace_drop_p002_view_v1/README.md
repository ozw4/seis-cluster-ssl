# F3 Local Barlow Twins trace-drop p=.02 view v1

This validation-only experiment is the one follow-up authorized by the failed
`p=0.01` trace-drop result. It changes only the probability of independent
whole-trace dropout after the existing forced-distinct horizontal flips:

```yaml
policy: horizontal_flip_trace_drop_v1
horizontal_flip_probability: 0.5
trace_drop_probability: 0.02
```

The immutable parent is
`artifacts/seis_ssl_cluster/f3_lithology_benchmark/local_barlow_twins_trace_drop_view_v1/base1ep/validation/trace_drop_p001_final_result.json`
(SHA-256
`3a83070718ce07f51756bfb91da6f792c6347f3009ca0290757bd93710fe1e2e`).
It failed the strict medium gate with three wins in five layouts and explicitly
authorized probability `0.02`. Its five `p=0.01` medium cells are frozen only
for the direct `p=.02 - p=.01` attribution contrast. Random remains the sole
gate and pass comparator. Reports are human-readable outputs and are never
pipeline inputs.

## Why this attempt

The original horizontal-flip views and the Gaussian-noise follow-up preserved
every seismic trace, making the invariance task too easy for a convolutional
encoder. Whole-trace dropout removes spatially coherent observations instead.
On the fixed unlabeled 16-crop diagnostic, `p=0.01` matched Gaussian `std=0.10`
in raw perturbation scale but affected only `0.70654296875` of sampled physical
pairs. The preregistered `p=0.02` alternative affected `0.916015625` of pairs,
with view correlation about `0.98003` and pair RMS difference about `0.19087`.
No labels or downstream metrics selected this strength.

## Fixed conditions

The base is a fresh seed-42 one-epoch run: 10,000 samples, batch size 16, and
exactly 625 optimizer steps without resume. The continuation is a fresh fixed
25-epoch, top-block-1 run initialized from that exact base: 15,625 optimizer
steps without training or lineage resume. Apart from trace-drop probability and
isolated output paths, its configs equal the `p=0.01` arm: architecture,
embedding dimension, optimizer, learning rates, weight decay, batch size,
sampling, seed, crop/patch/token geometry, manifests, preprocessing,
extraction, downstream fine-tuning, layouts, splits, evaluation, and metrics
are fixed.

The protocol is sealed after the fresh base and before continuation. It binds
the parent authorization, the base checkpoint, repository and benchmark
provenance, and all 15 canonical random validation cells before any `p=0.02`
validation metric exists. All downstream evidence is validation evidence
aggregated by unique validation voxel. No test label or test metric is read.

## Producer and lock order

Run from the repository root with the required roots exported. Every real
producer requires its exact output root to be absent. Never overwrite an output
or pass `--resume`.

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export workspace first}"
: "${F3_ROOT:?export F3 root first}"
export TRACE113=experiments/f3/facies_benchmark_v2/113_local_barlow_twins_trace_drop_p002_view_v1
export TRACE113_CONFIG="$TRACE113/30_validation/01_candidate.yaml"
export TRACE113_ID=local_barlow_twins_horizontal_trace_drop_p002_base1ep

python "$TRACE113/run_validation.py" \
  --config "$TRACE113_CONFIG" --audit-parent-only

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$TRACE113/10_stage1/horizontal_trace_drop_p002_base1ep/01_screen_1ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$TRACE113/10_stage1/horizontal_trace_drop_p002_base1ep/01_screen_1ep.yaml"

python "$TRACE113/run_validation.py" \
  --config "$TRACE113_CONFIG" --candidate "$TRACE113_ID" \
  --audit-base-checkpoint-only
python "$TRACE113/run_validation.py" \
  --config "$TRACE113_CONFIG" --create-protocol-lock

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$TRACE113/15_stage2/horizontal_trace_drop_p002_base1ep/01_continue_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$TRACE113/15_stage2/horizontal_trace_drop_p002_base1ep/01_continue_25ep.yaml"

python "$TRACE113/run_validation.py" \
  --config "$TRACE113_CONFIG" --candidate "$TRACE113_ID" \
  --audit-checkpoint-only

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TRACE113/20_embeddings/01_extract_horizontal_trace_drop_p002_base1ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TRACE113/20_embeddings/01_extract_horizontal_trace_drop_p002_base1ep.yaml"
```

After the protocol exists, do not edit source, experiment, or test files until
the final result and report are complete.

## Validation gate and decision

Run exactly the five medium layouts first:

```bash
for layout in layout_000 layout_001 layout_002 layout_003 layout_004
do
  python "$TRACE113/run_validation.py" \
    --config "$TRACE113_CONFIG" --candidate "$TRACE113_ID" \
    --layout "$layout" --size medium --dry-run
  python "$TRACE113/run_validation.py" \
    --config "$TRACE113_CONFIG" --candidate "$TRACE113_ID" \
    --layout "$layout" --size medium
done
```

The gate opens only when every unrounded candidate macro-F1 is strictly greater
than the paired frozen random macro-F1 (5/5); a tie fails. If closed, small and
large execution is forbidden. If open, run the remaining ten validation cells:

```bash
if python "$TRACE113/run_validation.py" \
  --config "$TRACE113_CONFIG" --candidate "$TRACE113_ID" \
  --layout layout_000 --size small --dry-run
then
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004
  do
    for size in small large
    do
      python "$TRACE113/run_validation.py" \
        --config "$TRACE113_CONFIG" --candidate "$TRACE113_ID" \
        --layout "$layout" --size "$size" --dry-run
      python "$TRACE113/run_validation.py" \
        --config "$TRACE113_CONFIG" --candidate "$TRACE113_ID" \
        --layout "$layout" --size "$size"
    done
  done
fi
```

Passing requires strict improvement over random in all 15 paired validation
cells. The final result separately records `p=.02 - p=.01` over the five medium
layouts, but that contrast cannot affect the decision. No additional
trace-drop probability is automatically authorized.

```bash
python "$TRACE113/run_validation.py" \
  --config "$TRACE113_CONFIG" --create-final-result
python "$TRACE113/build_report.py" --config "$TRACE113_CONFIG"
```

The report writes exactly `attempts.csv`, `validation_cells.csv`,
`paired_deltas.csv`, `summary.json`, and `summary.md` under
`reports/f3/facies_benchmark_v2/local_barlow_twins_trace_drop_p002_view_v1/base1ep/`.
The closed branch has 15 validation rows (5 live, 5 random, 5 frozen `p=.01`);
the open branch has 35.
