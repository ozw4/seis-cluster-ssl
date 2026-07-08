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

Invalid tokens are skipped during Viterbi decoding and remain `-1` in the output
label grid. They do not reset the trace sequence: when reverse transitions are
forbidden, the consecutive valid labels in each vertical trace are
non-decreasing in z order.

Metadata JSON is strict JSON-safe text, so non-finite values are not emitted as
`Infinity` or `NaN`. Saved labels are decoded from the saved final centers in
`cluster_centers.npy`, and `hmm_model.joblib` preserves the numerical transition
costs used for that decode.

## Configs

```text
01_stratigraphic_hmm_kmeans.yaml
01_stratigraphic_hmm_kmeans_smoke.yaml
02_stratigraphic_hmm_zonly_guardrail.yaml
03_stratigraphic_hmm_k6_10_resid_token_phase.yaml
03_stratigraphic_hmm_resid_edge_k6_10.yaml
04_stratigraphic_hmm_resid_edge_pathprior_k6_10.yaml
04_stratigraphic_hmm_resid_edge_pathprior_smoke.yaml
```

The main config writes to:

```text
/workspace/artifacts/seis_ssl_cluster/clustering/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full/overlap_x16/strat_hmm_k6_8_10_pca64_iter10
```

The `full` path component is kept so the output follows the generic clustering
artifact shape.

The z-only guardrail config writes to:

```text
/workspace/artifacts/seis_ssl_cluster/clustering/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full/overlap_x16/strat_hmm_zonly_k6_8_10_iter10
```

The residualized HMM v2 configs standardize token-phase residualization with an
8-token x/y backend edge exclusion. The no-prior config is the clean comparison
against previous residualized runs with post-hoc visualization masking:

```text
/workspace/artifacts/seis_ssl_cluster/clustering/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full/overlap_x16/strat_hmm_k6_10_pca64_resid_token_phase_edge8_iter10
```

The path-prior config adds conservative shallow/deep anchors and an expected
boundary-count prior for the v2 baseline:

```text
/workspace/artifacts/seis_ssl_cluster/clustering/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full/overlap_x16/strat_hmm_k6_10_pca64_resid_token_phase_edge8_pathprior_iter10
```

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

Dry-run the HMM v2 path-prior smoke config first:

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering/04_stratigraphic_hmm_resid_edge_pathprior_smoke.yaml \
  --dry-run
```

Run the HMM v2 path-prior smoke config when the frozen F3 embedding artifacts
are present:

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering/04_stratigraphic_hmm_resid_edge_pathprior_smoke.yaml
```

Run the full HMM v2 no-prior comparison:

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering/03_stratigraphic_hmm_resid_edge_k6_10.yaml
```

Run the full HMM v2 path-prior baseline:

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering/04_stratigraphic_hmm_resid_edge_pathprior_k6_10.yaml
```

## Comparisons And Diagnostics

The key comparison is against vanilla KMeans and z-only or random guardrails,
not supervised probes.

The z-only guardrail should produce ordered bands by construction. The embedding
HMM result is only scientifically stronger if it differs from z-only in
geologically meaningful ways, for example boundaries bending with reflectors or
respecting structural offsets rather than remaining flat depth bands.

Main diagnostics:

- reverse transition rate over consecutive valid trace observations
- boundary continuity and boundary z summary
- salt-and-pepper reduction by visual inspection
- whether boundaries follow structure rather than forming flat depth bands
- compare path-prior results against both the z-only guardrail and no-prior HMM
- inspect XZ sections and XY slices before interpreting units

See [docs/stratigraphic_hmm_clustering.md](../../../../docs/stratigraphic_hmm_clustering.md)
for the shared method notes and interpretation caveats.
