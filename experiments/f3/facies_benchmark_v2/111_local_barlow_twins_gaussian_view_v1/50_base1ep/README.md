# Reached F3 Local Barlow Twins base-1 branch

This branch is the pre-registered early-stopping fallback reached after the
5-base-epoch Gaussian-view branch failed its strict five-layout medium
validation gate. The immutable parent result is
`artifacts/seis_ssl_cluster/f3_lithology_benchmark/local_barlow_twins_gaussian_view_v1/base5ep/validation/gaussian_base5_final_result.json`
(SHA-256
`5b1e9c8778892d1226179c92e7d9fd788cb13329e05e5663ed3dd8ddb1e18d43`).
It records `passed: false`, `winner_candidate_id: null`, and
`authorizes_next_base_duration: true`, with the authorized next duration fixed
to one epoch. The parent report was published before any file in this branch
was created. Reports remain human-readable outputs and are never pipeline
inputs.

## Fixed branch definition

The parent selection lock chose
`local_barlow_twins_gaussian_noise_std010_base5ep`. This branch inherits that view
unchanged and carries only its matched-duration legacy control. It does not
re-rank noise strengths or geometry policies.

| Model ID | Base views | Base epochs | Continuation | Role |
|---|---|---:|---:|---|
| `local_barlow_twins_gaussian_noise_std010_base1ep` | forced-distinct XY flips, `p=0.5`, plus independent valid-voxel Gaussian noise, `std=0.10` | 1 | 25 epochs, top block 1 | inherited selected view |
| `local_barlow_twins_legacy_flip_base1ep` | forced-distinct XY flips, `p=0.5` | 1 | 25 epochs, top block 1 | matched legacy control |

Every base is a fresh seed-42 run. One epoch at the unchanged 10,000
samples per epoch and batch size 16 produce exactly 625 optimizer steps.
The checkpoint must record epoch 1, completed dataset epoch 0, and top-level
`resume_count: 0`. Each fixed continuation is also fresh, initializes only
from its own base, and records epoch 25, 15,625 steps, top-level
`resume_count: 0`, and `continuation_lineage.resume_count: 0`. Never pass
`--resume` to a base or continuation in this branch.

The base configs differ from their corresponding 5-epoch configs only in
`paths.output_root` and `train.epochs`. The continuation configs retain the
canonical top-block-1 optimizer, learning rate, weight decay, batch size,
sampling, precision, seed, and clipping contract. Architecture, embedding
dimension, crop/patch/token geometry, manifests, preprocessing, downstream
fine-tuning, section layouts, splits, seeds, evaluation, and metrics remain
fixed. Extraction retains the canonical 128-cube, 64-overlap contract.

The duration-specific runner pins and validates the parent result directly.
It does not replay the now-closed base-5 runner and does not consume the
parent report. Its protocol lock binds the parent result, the two completed
bases, benchmark inputs, and the complete relevant repository inventory.
Its inherited selection lock contains no base-1 validation score and merely
maps the parent-selected policy to the new duration-specific model ID.

All downstream results in this branch are validation results. No test path,
test label, or test metric is defined or read.

## Producer and protocol sequence

Run from the repository root with the three required roots exported:

```bash
set -euo pipefail
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export workspace first}"
: "${F3_ROOT:?export F3 root first}"
export BASE1=experiments/f3/facies_benchmark_v2/111_local_barlow_twins_gaussian_view_v1/50_base1ep
export BASE1_VALIDATION_CONFIG="$BASE1/30_validation/01_candidates.yaml"

python "$BASE1/run_validation.py" \
  --config "$BASE1_VALIDATION_CONFIG" --audit-parent-only

for config in \
  "$BASE1/10_stage1/gaussian_noise_std010_base1ep/01_screen_1ep.yaml" \
  "$BASE1/10_stage1/legacy_flip_base1ep/01_matched_1ep.yaml"
do
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
    --config "$config" --dry-run
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
    --config "$config"
done

for candidate in \
  local_barlow_twins_gaussian_noise_std010_base1ep \
  local_barlow_twins_legacy_flip_base1ep
do
  python "$BASE1/run_validation.py" \
    --config "$BASE1_VALIDATION_CONFIG" \
    --candidate "$candidate" --audit-base-checkpoint-only
done

python "$BASE1/run_validation.py" \
  --config "$BASE1_VALIDATION_CONFIG" --create-protocol-lock
python "$BASE1/run_validation.py" \
  --config "$BASE1_VALIDATION_CONFIG" --create-selection-lock

for config in \
  "$BASE1/15_stage2/gaussian_noise_std010_base1ep/01_continue_25ep.yaml" \
  "$BASE1/15_stage2/legacy_flip_base1ep/01_continue_25ep.yaml"
do
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
    --config "$config" --dry-run
  python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
    --config "$config"
done

for candidate in \
  local_barlow_twins_gaussian_noise_std010_base1ep \
  local_barlow_twins_legacy_flip_base1ep
do
  python "$BASE1/run_validation.py" \
    --config "$BASE1_VALIDATION_CONFIG" \
    --candidate "$candidate" --audit-checkpoint-only
done

for config in \
  "$BASE1/20_embeddings/01_extract_gaussian_noise_std010_base1ep.yaml" \
  "$BASE1/20_embeddings/02_extract_legacy_flip_base1ep.yaml"
do
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$config" --dry-run
  python proc/seis_ssl_cluster/extract_embeddings.py \
    --config "$config"
done
```

Before each real producer command, require its exact output root to be absent.
Do not remove, overwrite, or reuse an existing branch output. After base
training, base-only checkpoint audits are the only checkpoint-audit operations
allowed before the protocol lock (the parent-only audit is also permitted).
After the lock exists, do not edit source, experiment, or test files until this
branch's final result and report are complete.

## Validation-only gate

First run exactly the five medium layouts for both arms:

```bash
arms=(
  local_barlow_twins_gaussian_noise_std010_base1ep
  local_barlow_twins_legacy_flip_base1ep
)

for candidate in "${arms[@]}"
do
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004
  do
    python "$BASE1/run_validation.py" \
      --config "$BASE1_VALIDATION_CONFIG" \
      --candidate "$candidate" --layout "$layout" --size medium --dry-run
    python "$BASE1/run_validation.py" \
      --config "$BASE1_VALIDATION_CONFIG" \
      --candidate "$candidate" --layout "$layout" --size medium
  done
done
```

An arm opens the next gate only if its unrounded macro-F1 is strictly greater
than the existing random-encoder result in all five paired medium layouts.
When neither arm does, small/large execution is forbidden and the exact
reached branch contains ten candidate cells. When either arm does, the runner
authorizes both arms on all five small and all five large layouts:

```bash
if python "$BASE1/run_validation.py" \
  --config "$BASE1_VALIDATION_CONFIG" \
  --candidate "${arms[0]}" --layout layout_000 --size small --dry-run
then
  for candidate in "${arms[@]}"
  do
    for layout in layout_000 layout_001 layout_002 layout_003 layout_004
    do
      for size in small large
      do
        python "$BASE1/run_validation.py" \
          --config "$BASE1_VALIDATION_CONFIG" \
          --candidate "$candidate" --layout "$layout" --size "$size" --dry-run
        python "$BASE1/run_validation.py" \
          --config "$BASE1_VALIDATION_CONFIG" \
          --candidate "$candidate" --layout "$layout" --size "$size"
      done
    done
  done
fi
```

The final result separately records inherited-Gaussian minus random, legacy
minus random, and inherited-Gaussian minus legacy. Passing requires one arm to
beat random in all 15 layout/size cells. If both pass, Gaussian wins only when
it also beats legacy in all 15 cells; otherwise the simpler legacy arm wins.
No small/large result is described as an untouched holdout: all three sizes
are validation tuning signals.

Create the exclusive branch result only after its exact reached cell set is
complete, then build the report-only projection:

```bash
python "$BASE1/run_validation.py" \
  --config "$BASE1_VALIDATION_CONFIG" --create-final-result
python "$BASE1/build_report.py" \
  --config "$BASE1_VALIDATION_CONFIG"
```

The report writes exactly `attempts.csv`, `validation_cells.csv`,
`paired_deltas.csv`, `summary.json`, and `summary.md` under
`reports/f3/facies_benchmark_v2/local_barlow_twins_gaussian_view_v1/base1ep/`.
It is never used as pipeline input. If the immutable result passes, stop. If
it fails, preserve the report before considering a different, separately
authorized augmentation family. This branch is terminal for the pre-registered
duration sequence: its final result always records
`authorizes_next_base_duration: false` and
`authorized_next_base_pretraining_epochs: null`.
