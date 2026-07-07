# F3 stratigraphic HMM clustering

This stage runs stratigraphic unit discovery on F3 token embeddings. It is not
lithology or facies classification. The clustering model consumes frozen SSL
embeddings extracted from the F3 volume and does not use F3 lithology labels for
training.

The HMM prior imposes vertical order on cluster IDs along each trace. KMeans
initializes the units, centers are ordered by mean z, Viterbi decoding assigns an
ordered unit sequence per vertical trace, and centers are updated before the next
iteration. F3 lithology labels, if used later, are only sanity-check evaluation
signals.

## Configs

```text
01_stratigraphic_hmm_kmeans.yaml
01_stratigraphic_hmm_kmeans_smoke.yaml
```

The main config writes to:

```text
/workspace/artifacts/seis_ssl_cluster/clustering/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full/overlap_x16/strat_hmm_k6_8_10_pca64_iter10
```

The `full` path component is kept so the output follows the generic clustering
artifact shape.

## Run

Dry-run the main config first:

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering/01_stratigraphic_hmm_kmeans.yaml \
  --dry-run
```

Run the full experiment only when the frozen F3 embedding artifacts are present:

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering/01_stratigraphic_hmm_kmeans.yaml
```

Use the smoke config for dry-run coverage and small local checks:

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering/01_stratigraphic_hmm_kmeans_smoke.yaml \
  --dry-run
```

## Comparisons And Diagnostics

The key comparison is against vanilla KMeans and z-only or random guardrails,
not supervised probes.

Main diagnostics:

- reverse transition rate
- boundary continuity and boundary z summary
- salt-and-pepper reduction by visual inspection
- whether boundaries follow structure rather than forming flat depth bands

See [docs/stratigraphic_hmm_clustering.md](../../../../docs/stratigraphic_hmm_clustering.md)
for the shared method notes and interpretation caveats.
