# F3 Local Barlow Twins trace-drop view v1

This validation-only experiment tests one small, preregistered view change:
independent whole-trace dropout after the existing forced-distinct horizontal
flips and local-pair sampling. The candidate is
`local_barlow_twins_horizontal_trace_drop_p001_base1ep`, with the exact view
mapping:

```yaml
policy: horizontal_flip_trace_drop_v1
horizontal_flip_probability: 0.5
trace_drop_probability: 0.01
```

The immutable parent is the terminal Gaussian base-1 result at
`artifacts/seis_ssl_cluster/f3_lithology_benchmark/local_barlow_twins_gaussian_view_v1/base1ep/validation/gaussian_base1_final_result.json`
(SHA-256
`6ab9cb2ae8bf89eb5dea9ab34244b60f41499f44cc046995e41ec0350854e1fa`).
It failed the medium gate. Its five Gaussian and five legacy medium cells are
frozen for causal attribution only; neither control selects the new candidate
or affects its gate or success decision. Reports are human-readable outputs
and are never pipeline inputs.

## Why this view

Independent Gaussian noise leaves every seismic trace and token waveform
present, so a convolutional encoder can largely suppress the perturbation.
On the fixed unlabeled diagnostic (16 epoch-0 F3 crops, two views, 2,048
physical local pairs), Gaussian `std=0.10` had all-valid view correlation
`0.9893753712` and pair RMS difference `0.1413962548`.

Trace dropout at `p=0.01` matched that raw perturbation scale while removing
complete spatially coherent traces: realized drop rate `0.0099411011`,
all-valid correlation `0.9900436706`, pair RMS difference `0.1354699353`, and
`0.70654296875` of sampled physical pairs affected in at least one view. The
stronger `p=0.02` alternative affected `0.916015625` of sampled physical pairs
and is not part of this attempt. It may be run only as a separately documented
follow-up when the immutable p=.01 final result explicitly authorizes it.
These diagnostics use no labels or downstream metrics.

## Fixed conditions

The one-epoch base is a fresh seed-42 run: 10,000 samples at batch size 16,
exactly 625 optimizer steps, no resume. The protocol lock is created after
that base and before continuation. It binds the pinned parent, the base
checkpoint, the benchmark and repository provenance, and all 15 canonical
random validation cells before any trace-drop validation metric exists.
There is no selection lock because there is only one fixed candidate.

The continuation is a fresh fixed 25-epoch, top-block-1 run initialized from
that exact base: 15,625 optimizer steps, no training resume, and no lineage
resume. Apart from the augmentation mapping and the already selected duration,
the configs are identical to the Gaussian base-1 arm: architecture, embedding
dimension, optimizer, learning rates, weight decay, batch size, sampling,
seed, crop/patch/token geometry, manifests, preprocessing, extraction,
downstream fine-tuning, layouts, splits, evaluation, and metrics are fixed.
All downstream evidence is validation evidence aggregated by unique validation
voxel. No test label or test metric is read.

## Producer and lock order

Run from the repository root with the required roots exported. Every real
producer requires its exact output root to be absent; never overwrite or reuse
an output. Do not pass `--resume` to either pretraining stage.

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export workspace first}"
: "${F3_ROOT:?export F3 root first}"
export TRACE112=experiments/f3/facies_benchmark_v2/112_local_barlow_twins_trace_drop_view_v1
export TRACE112_CONFIG="$TRACE112/30_validation/01_candidate.yaml"
export TRACE112_ID=local_barlow_twins_horizontal_trace_drop_p001_base1ep

python "$TRACE112/run_validation.py" \
  --config "$TRACE112_CONFIG" --audit-parent-only

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$TRACE112/10_stage1/horizontal_trace_drop_p001_base1ep/01_screen_1ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$TRACE112/10_stage1/horizontal_trace_drop_p001_base1ep/01_screen_1ep.yaml"

python "$TRACE112/run_validation.py" \
  --config "$TRACE112_CONFIG" --candidate "$TRACE112_ID" \
  --audit-base-checkpoint-only
python "$TRACE112/run_validation.py" \
  --config "$TRACE112_CONFIG" --create-protocol-lock

python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$TRACE112/15_stage2/horizontal_trace_drop_p001_base1ep/01_continue_25ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config "$TRACE112/15_stage2/horizontal_trace_drop_p001_base1ep/01_continue_25ep.yaml"

python "$TRACE112/run_validation.py" \
  --config "$TRACE112_CONFIG" --candidate "$TRACE112_ID" \
  --audit-checkpoint-only

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TRACE112/20_embeddings/01_extract_horizontal_trace_drop_p001_base1ep.yaml" \
  --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$TRACE112/20_embeddings/01_extract_horizontal_trace_drop_p001_base1ep.yaml"
```

After the protocol exists, do not edit source, experiment, or test files until
the final result and report are complete.

## Validation gate and decision

Run exactly the five medium layouts first:

```bash
for layout in layout_000 layout_001 layout_002 layout_003 layout_004
do
  python "$TRACE112/run_validation.py" \
    --config "$TRACE112_CONFIG" --candidate "$TRACE112_ID" \
    --layout "$layout" --size medium --dry-run
  python "$TRACE112/run_validation.py" \
    --config "$TRACE112_CONFIG" --candidate "$TRACE112_ID" \
    --layout "$layout" --size medium
done
```

The gate opens only when every unrounded candidate macro-F1 is strictly
greater than the paired frozen random macro-F1 (5/5). A tie fails. If the gate
is closed, small and large execution is forbidden and the reached live branch
contains exactly five candidate cells. If it opens, run all five layouts at
both remaining validation sizes:

```bash
if python "$TRACE112/run_validation.py" \
  --config "$TRACE112_CONFIG" --candidate "$TRACE112_ID" \
  --layout layout_000 --size small --dry-run
then
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004
  do
    for size in small large
    do
      python "$TRACE112/run_validation.py" \
        --config "$TRACE112_CONFIG" --candidate "$TRACE112_ID" \
        --layout "$layout" --size "$size" --dry-run
      python "$TRACE112/run_validation.py" \
        --config "$TRACE112_CONFIG" --candidate "$TRACE112_ID" \
        --layout "$layout" --size "$size"
    done
  done
fi
```

Passing requires strict improvement over random in all 15 paired validation
cells. The final result also records trace-drop minus frozen Gaussian and
trace-drop minus frozen legacy on the five medium layouts, but those
attribution contrasts are not pass criteria. Publish the immutable decision
and then the five report-only projections:

```bash
python "$TRACE112/run_validation.py" \
  --config "$TRACE112_CONFIG" --create-final-result
python "$TRACE112/build_report.py" --config "$TRACE112_CONFIG"
```

The report writes exactly `attempts.csv`, `validation_cells.csv`,
`paired_deltas.csv`, `summary.json`, and `summary.md` under
`reports/f3/facies_benchmark_v2/local_barlow_twins_trace_drop_view_v1/base1ep/`.
The closed branch has 20 validation rows (5 trace, 5 random, 10 frozen
controls); the open branch has 40. If p=.01 fails, only its immutable final
result can authorize the separately documented `p=0.02` follow-up.
