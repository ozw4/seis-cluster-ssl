# Volve horizon supervision

The benchmark compares representation methods with nested budgets of complete
physical inline and crossline sections. It does not subsample points from a
section or tune the number of labels to equalize conditions.

All conditions share a fixed validation pair used only for checkpoint selection.
The common test excludes validation and every section that could be used for
training in any layout, preventing unused candidates from returning to the test
set. Results on common five-horizon support are the primary comparison; native
per-horizon support is secondary.

The bound horizons are treated as observations: this workflow does not smooth,
reorder, swap, or repair them. Manual and visual QC remain prerequisites, not
model-selection signals.

[`01_layouts.yaml`](../experiments/volve/horizon_benchmark_v1/20_horizon_supervision/01_layouts.yaml)
owns the physical line assignments. The layout builder owns derived support
masks, data identities and hashes, time-window derivation, and the emitted plan
schema; its tests verify those contracts rather than define them. Phase ordering
is documented in the
[`horizon supervision runbook`](../experiments/volve/horizon_benchmark_v1/20_horizon_supervision/README.md).
