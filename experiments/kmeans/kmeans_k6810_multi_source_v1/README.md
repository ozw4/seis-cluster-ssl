# Unordered k-means pseudo-label control

This study compares fixed, unordered MiniBatchKMeans targets with the
[canonical HMM v2 protocol](../../hmm_v2/canonical_multi_source_v1/README.md).
F3 lithology, Parihaka channel segmentation, and Volve horizon interpretation
each use MAE100, asymmetric LocalBT3, and random initialization: nine new
25-epoch continuations and 135 downstream decoder cells. Versioned references,
SHA-256 bindings, and candidate definitions are in [matrix.yaml](matrix.yaml).
Implementation checks and launch evidence are in [VALIDATION.md](VALIDATION.md).

## Scientific contract

Each arm independently clusters its original parent's embedding features at
K=[6,8,10]. L2 normalization, token-position residualization, PCA, sample count,
seed, and symmetric edge exclusions are inherited from that family's HMM target
configuration. Exclusions apply to fitting and prediction. A staging directory
links the original arrays and metadata and stores only the reduced valid masks.
No depth coordinates enter clustering; there are no HMM iterations, ordered
centroid renumbering, transition penalties, path anchors, or lateral smoothing.
The fixed labels have confidence one on the matched valid support and minus-one
invalid labels. Schema-v1 target arrays imply unit boundary weights.

Training retains the existing prototype-head architecture (its historical
`multi_resolution_ordered_prototypes_v1` name denotes the module structure),
with a dedicated `unordered_kmeans_labels_v1` representation and manifest.
The head itself performs independent projection and categorical classification.
Ordered cross-head consistency is neither evaluated nor reported per pair;
its aggregate loss is zero. Prototype/usage/distillation weights remain
1.0/0.005/0.2. All other settings, including parent and teacher checkpoints,
top-one-block continuation, batch sizes, LR, seed 42, no masking, and the
25-epoch budget, match the bound HMM recipe. Smoke and full training start
independently. Targets never refresh.

The comparison measures the **joint effect of HMM pseudo-label generation and
ordered cross-head consistency**. It does not isolate HMM decoding alone.
Existing HMM manifests, checkpoint schemas, and canonical results retain their
original validation contracts. Unordered checkpoints use schema 9 and cannot
resume an ordered or otherwise incompatible recipe.

## Execution

Run from the repository root with the repository Python environment active.
Set `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `F3_ROOT`, and
`SEIS_SSL_CLUSTER_VOLVE_ROOT` to the existing data/artifact locations.
`SEIS_SSL_CLUSTER_WORKSPACE` defaults to the repository root. Preserve the
canonical F3 label-path spelling recorded in supervision metadata.

```bash
# No artifacts or environment values are needed to inspect the full command plan.
python proc/seis_ssl_cluster/run_kmeans_study.py --plan

# Read-only preflight for existing source embeddings.
python proc/seis_ssl_cluster/run_kmeans_study.py --dry-run --from targets --to targets

# Create fixed targets, then validate the actual training recipes without GPUs.
python proc/seis_ssl_cluster/run_kmeans_study.py --execute --from targets --to targets
python proc/seis_ssl_cluster/run_kmeans_study.py --dry-run --from smoke --to full

# All full-batch, one-step smokes precede any full training.
CUDA_VISIBLE_DEVICES=0 python proc/seis_ssl_cluster/run_kmeans_study.py --execute --from smoke
```

The driver runs sequentially on the selected GPU, retaining each source's
batch size. Choose an available GPU with sufficient memory for the existing
full-batch recipe; dry-run does not establish memory feasibility. The study
lock prevents duplicate owned writers. Logs are stored in timestamped
`$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/kmeans/kmeans_k6810_multi_source_v1/`
subdirectories. A failed command stops the study; no background retries occur.

Downstream uses the fixed epoch-25 encoder, five paired layouts, and the
original small/medium/large label supports. Decoder configuration, seed 42000,
and checkpoint selection remain inherited. F3 reports validation Mean IoU and
Macro F1; Parihaka reports test channel IoU/F1; Volve reports test macro MAE in
samples and within-two-samples accuracy. Validation selects decoder checkpoints.
Test metrics do not select targets, encoder checkpoints, or training duration.

## Recovery and outputs

Use `--survey` and `--candidate` to narrow stages. Resume requires exactly one
candidate and training stage, or one explicit downstream cell:

```bash
python proc/seis_ssl_cluster/run_kmeans_study.py --execute \
  --survey f3 --candidate mae100_kmeans_mh_k6810_distill020 \
  --from full --to full --resume

python proc/seis_ssl_cluster/run_kmeans_study.py --execute \
  --survey f3 --candidate mae100_kmeans_mh_k6810_distill020 \
  --from downstream --to downstream --layout layout_000 --size small --resume

python proc/seis_ssl_cluster/run_kmeans_study.py --execute --from summary --to summary
```

A complete matching target manifest is verified before reuse. Partial target
outputs require explicit archival before retry. Completed full training is
reused only after source audit. Embedding extraction refuses existing outputs;
continue from downstream after complete extraction, or archive an incomplete
owned extraction before retry. Downstream retains its existing completed-cell
audits and resume checks. Summary publication refuses an existing directory.

New survey artifacts use the canonical
`surveys/<survey>/<storage-series>/<kind>/kmeans_k6810_multi_source_v1/` layout.
F3 pretraining/targets use storage v1 and downstream/embeddings use v2.
Cross-survey summaries live at
`studies/kmeans/kmeans_k6810_multi_source_v1/summary/` and publish exactly:

- `comparison.csv`
- `summary.json`
- `summary.md`

All 135 unique cells and both encoder lineages are audited before publication.
Each of the 27 survey/source/budget rows reports five-layout means, sample
standard deviations, and paired differences. Differences are k-means minus
HMM v2: positive IoU/F1/accuracy and negative horizon MAE favor k-means.
Surveys are not averaged, and nested label budgets are not treated as
independent replicates. Review summaries contain references and audit evidence;
no tracked reports are pipeline inputs or automatic publication destinations.
