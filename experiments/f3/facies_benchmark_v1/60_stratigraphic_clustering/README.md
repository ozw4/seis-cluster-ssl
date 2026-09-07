# F3 stratigraphic HMM clustering

This directory contains the F3 configurations for embedding-driven
stratigraphic HMM clustering, guardrails, and ablations. The scientific purpose
and interpretation limits are maintained in
[`docs/stratigraphic_hmm_clustering.md`](../../../../docs/stratigraphic_hmm_clustering.md).

Select the YAML for the main condition, smoke check, z-only guardrail,
residualized variant, or optional path-prior comparison.

Run a selected condition with the shared clustering entrypoint, using its
dry-run before writing artifacts. For example:

```bash
export EXP=experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering
export CONFIG="$EXP/01_stratigraphic_hmm_kmeans.yaml"

python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$CONFIG" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config "$CONFIG"
```
