# parihaka unordered k-means control

Follow the [shared study runbook](../../../kmeans/kmeans_k6810_multi_source_v1/README.md).
The fixed matrix binds the canonical HMM v2 source, preprocessing, training,
and downstream recipes for MAE100, LocalBT3, and random initialization.

This survey contributes three independent 25-epoch continuations and 45
downstream cells (five layouts × three label budgets × three sources).
Use `--survey parihaka` with explicit `--from` and `--to` stages; the final
summary requires all three surveys.
