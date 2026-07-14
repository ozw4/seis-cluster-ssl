# F3 M3-V voxel result summary

This stage consolidates the six original-split evaluations: MAE, M1, and M2-A,
each under V0 token projection and the V1 frozen-embedding decoder. It requires
one common split-grid hash, class order, and validation voxel count across all
six runs. Missing or mismatched evidence is an error; it does not produce a
partial metric or result.

Every V1 input is read from the canonical
`voxel_evaluations/frozen_embedding_decoder_nearest_voxel_ln_v1` path. The old
`frozen_embedding_decoder_v1` identity is not a summary input.

Run after every original-split evaluation is complete:

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_results.py \
  --config experiments/f3/facies_benchmark_v1/90_f3_voxel_results/01_summarize_original_split.yaml \
  --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_results.py \
  --config experiments/f3/facies_benchmark_v1/90_f3_voxel_results/01_summarize_original_split.yaml
```

Local output is
`/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/reports/voxel_benchmark_v1/`.
This original-split step does not publish. After the six-split suite completes,
its final summarizer publishes these products together with the robustness
summary to `results/f3/facies_benchmark_v1/voxel_lithology_benchmark_v1/`.
The final manifest records SHA-256 and byte size; raw `.npy`, `.pt`, and
`.joblib` files remain under the artifact root.

`decoder_value` asks whether V1 improves on V0. `m2a_vs_m1_voxel` asks whether
M2-A V1 improves on M1 V1, with M1/M2-A boundary evidence and classes 3 and 5
monitored. Both statuses are provisional on the original split. Run the paired
six-split suite and its final summary/publish step before making the robustness
interpretation.

See
[`docs/f3_voxel_lithology_benchmark.md`](../../../../docs/f3_voxel_lithology_benchmark.md)
for the complete command order, artifact paths, resume policy, and release
checks.
