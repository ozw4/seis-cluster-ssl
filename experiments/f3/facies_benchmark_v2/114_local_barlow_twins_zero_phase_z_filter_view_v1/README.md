# F3 Local Barlow Twins zero-phase Z-filter view v1

This validation-only experiment has one separately preregistered candidate. It keeps the existing forced-distinct horizontal flips and applies a centered, unit-DC Z-only smoother to exactly one randomly selected view in each pair:

\`\`\`yaml
policy: horizontal_flip_zero_phase_z_filter_v1
horizontal_flip_probability: 0.5
z_filter_side_weight: 0.25
\`\`\`

The smoother is \`[0.25, 0.50, 0.25]\`. It preserves DC amplitude, sample timing, polarity, and lateral geometry while making the two views differ in vertical bandwidth. Invalid samples are excluded from the local normalization, so the filter never borrows values across an invalid Z sample.

The immutable prior trace-drop p=.02 final result is \`artifacts/seis_ssl_cluster/f3_lithology_benchmark/local_barlow_twins_trace_drop_p002_view_v1/base1ep/validation/trace_drop_p002_final_result.json\` (SHA-256 \`8b27c1141b5e7740653f8585acb0a9e978e74a82355bbbcdb947fd888cc711cd\`). It failed the strict medium gate with two wins in five layouts. Its five candidate medium cells are frozen only for the direct Z-filter minus p=.02 trace-drop attribution contrast. They do not authorize this experiment and cannot affect its gate or pass decision; random remains the sole comparator.

## Why this attempt

The original flip-only, Gaussian-noise, and trace-drop variants all leave the view relation either too easy or physically inappropriate for seismic data. The selected view change instead makes Barlow Twins invariant to modest, zero-phase vertical-bandwidth variation without dropping traces or injecting white noise. The weight was fixed before this branch's base training from an unlabeled 16-crop training-data diagnostic: side weight \`.125\` gave pair RMS \`0.0562503565\` against signal standard deviation \`0.9648129713\`, while \`.25\` gave pair RMS \`0.11339506025\` with eight filter-A and eight filter-B assignments. This diagnostic used no facies labels, validation metrics, or test data. No downstream result selected the weight.

## Fixed conditions

The candidate uses a fresh seed-42 one-epoch base: 10,000 samples, batch size 16, and exactly 625 optimizer steps without resume. Its continuation is a fresh fixed 25-epoch, top-block-1 run initialized from that exact base: 15,625 optimizer steps without training or lineage resume. Apart from the augmentation mapping and isolated output paths, the configs preserve architecture, embedding dimension, optimizer, learning rates, weight decay, batch size, sampling, seed, crop/patch/token geometry, manifests, preprocessing, extraction, downstream fine-tuning, layouts, splits, evaluation procedure, and metrics.

The protocol is sealed after the fresh base and before continuation. It binds the closed p=.02 parent result, the base checkpoint, repository and benchmark provenance, and all 15 canonical random validation cells before any candidate validation metric exists. All downstream evidence is validation evidence aggregated by unique validation voxel. No test label or test metric is read.

## Producer and lock order

Run from the repository root with the required roots exported. Every real producer requires its exact output root to be absent. Never overwrite an output or pass \`--resume\`.

\`\`\`bash
set -euo pipefail
: "\${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "\${SEIS_SSL_CLUSTER_WORKSPACE:?export workspace first}"
: "\${F3_ROOT:?export F3 root first}"
export ZFILTER114=experiments/f3/facies_benchmark_v2/114_local_barlow_twins_zero_phase_z_filter_view_v1
export ZFILTER114_CONFIG="$ZFILTER114/30_validation/01_candidate.yaml"
export ZFILTER114_ID=local_barlow_twins_zero_phase_z_filter_w025_base1ep

python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --audit-parent-only

python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$ZFILTER114/10_stage1/zero_phase_z_filter_w025_base1ep/01_screen_1ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$ZFILTER114/10_stage1/zero_phase_z_filter_w025_base1ep/01_screen_1ep.yaml"

python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --candidate "$ZFILTER114_ID" --audit-base-checkpoint-only
python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --create-protocol-lock

python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$ZFILTER114/15_stage2/zero_phase_z_filter_w025_base1ep/01_continue_25ep.yaml" --dry-run
python proc/seis_ssl_cluster/train_amp_barlow_twins.py --config "$ZFILTER114/15_stage2/zero_phase_z_filter_w025_base1ep/01_continue_25ep.yaml"

python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --candidate "$ZFILTER114_ID" --audit-checkpoint-only

python proc/seis_ssl_cluster/extract_embeddings.py --config "$ZFILTER114/20_embeddings/01_extract_zero_phase_z_filter_w025_base1ep.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$ZFILTER114/20_embeddings/01_extract_zero_phase_z_filter_w025_base1ep.yaml"
\`\`\`

After the protocol exists, do not edit source, experiment, or test files until the final result and report are complete.

## Validation gate and decision

Run exactly the five medium layouts first:

\`\`\`bash
for layout in layout_000 layout_001 layout_002 layout_003 layout_004
do
  python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --candidate "$ZFILTER114_ID" --layout "$layout" --size medium --dry-run
  python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --candidate "$ZFILTER114_ID" --layout "$layout" --size medium
done
\`\`\`

The gate opens only when every unrounded candidate macro-F1 is strictly greater than the paired frozen random macro-F1 (5/5); a tie fails. If closed, small and large execution is forbidden. If open, run the remaining ten validation cells:

\`\`\`bash
if python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --candidate "$ZFILTER114_ID" --layout layout_000 --size small --dry-run
then
  for layout in layout_000 layout_001 layout_002 layout_003 layout_004
  do
    for size in small large
    do
      python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --candidate "$ZFILTER114_ID" --layout "$layout" --size "$size" --dry-run
      python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --candidate "$ZFILTER114_ID" --layout "$layout" --size "$size"
    done
  done
fi
\`\`\`

Passing requires strict improvement over random in all 15 paired validation cells. The final result separately records Z-filter minus p=.02 trace-drop over the five medium layouts, but that contrast cannot affect the decision. No additional view-generation search is automatically authorized.

\`\`\`bash
python "$ZFILTER114/run_validation.py" --config "$ZFILTER114_CONFIG" --create-final-result
python "$ZFILTER114/build_report.py" --config "$ZFILTER114_CONFIG"
\`\`\`

The report writes exactly \`attempts.csv\`, \`validation_cells.csv\`, \`paired_deltas.csv\`, \`summary.json\`, and \`summary.md\` under \`reports/f3/facies_benchmark_v2/local_barlow_twins_zero_phase_z_filter_view_v1/base1ep/\`. The closed branch has 15 validation rows (5 live, 5 random, 5 frozen p=.02) and 10 paired rows; the open branch has 35 validation rows and 20 paired rows.
