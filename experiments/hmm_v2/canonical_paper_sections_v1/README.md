# Canonical HMM v2 paper section comparisons

## Panel-label revision: 2026-10-22

The user requested a visualization-only display-name change from LocalBT to
**Barlow Twins**, for both v1 and v2 on all three surveys. The
[revision index](../../../reports/hmm_paper_labels_barlow_twins_v1/README.md)
links the revised figures. They reuse the original predictions and fixed
sections, with no training, inference, scoring or changes to experiment IDs or
result tables. At the user's subsequent request, the verified title revisions
overwrite the PNGs at their original paths. Old images are deleted without
backups; scientific source/checkpoint bindings remain unchanged. The shared
renderers normalize the historical labels when drawing future figures.

The figures remain at the original v1/v2 publication paths. The temporary
`studies/hmm_paper_labels_barlow_twins_v1/` duplicate is removed after installation.
The exact per-survey published set remains 30/30/20 PNGs plus a manifest and
the byte-identical section CSV, or 32/32/22 files. The revision manifest records
the previous manifest digest and the title-only change. Every PNG is checked
for pixel equality outside the two revised title regions. The historical
completion below describes the original export. Current PNG hashes and v1
reference-manifest hashes are refreshed in the manifests and execution YAML;
canonical selection, predictions, checkpoints, section CSVs and scores are not
changed. The operation receipt is
`operations/hmm_paper_labels_barlow_twins_20261022_v1/installation.json`.

Requested 2026-10-21: render the canonical v2 results in the same form as the
existing v1 paper figures. This study creates **80 PNGs**: F3 30, Parihaka 30,
Volve 20. All use layout_000/large and the same original v1 section indices,
coordinate conventions, display crops, amplitude scales, colors, typography,
2×4 panel layout, 180×105 mm size and 300 dpi. No new section selection is
made from v2 predictions, scores, or appearance.

## Scientific identity

The explicitly selected
[v2 definition](../canonical_multi_source_v1/selection.yaml) is K=[6,8,10],
distillation=0.2, consistency=0.1. It is **not** K=[4,6,8], zero consistency,
or trace-mask v3. [execution.yaml](execution.yaml) pins that selection, its
completed 135-cell aggregate, all 18 encoder/decoder/embedding sources and
the three v1 reference manifests/section CSVs. Each v2 source must match the
selected aggregate and actual multi-head embedding provenance.

Every figure shows seismic amplitude, ground truth, and the three parent
families (MAE, asymmetric Barlow Twins (3 epochs), random), each without HMM and with canonical
v2. The three v2 panels are labelled **HMM v2**. Predicted facies use raw
argmax; Volve uses the existing raw soft-argmax horizon inference. There is
no smoothing, ensemble, horizon reordering, postprocessing, retraining,
parameter change, checkpoint reselection or new metric evaluation.

Parihaka MAE/no-HMM uses the user-designated
[seed-42001 canonical reference](../../parihaka/facies_benchmark_v1/47_mae_no_hmm_large_decoder_seed42001_v1/selection.yaml).
All other decoders retain seed 42000. The original v1 Parihaka MAE figure
used seed 42000, so that baseline panel legitimately changes in the new v2
comparison. The different decoder seeds are disclosed, not presented as a
same-seed paired comparison. Both old figures and numerical results remain
unchanged. The current Barlow Twins v3 decoder-seed repeat is unrelated and is not
used by these figures.

## Data and rendering reuse

- F3: reuse all six existing full-volume prediction artifacts. Coordinate
  vectors are cross-checked against the original seismic/label SEG-Y geometry.
- Parihaka: reuse existing Barlow Twins/random parent full-volume predictions;
  export the selected MAE seed-42001 parent and three v2 decoders. Labels
  retain source-facies-5→Channel mapping and one-based XYZ index axes.
- Volve: reuse all three parent horizon exports; export the three selected
  v2 decoders. Preserve float16 GPU inference and the original [552,768)
  sample window, native ground-truth horizons and physical coordinates.
- The shared [F3 renderer](../../../src/seis_ssl_cluster/f3/lithology/hmm_v1_comparison_visualization.py),
  [Parihaka axis adapter](../../../src/seis_ssl_cluster/parihaka/paper_sections.py),
  and [Volve renderer](../../../src/seis_ssl_cluster/volve/paper_sections.py)
  remain unchanged. A separate
  [v2 orchestrator](../../../src/seis_ssl_cluster/visualization/hmm_v2_paper_sections.py)
  validates canonical multi-head sources and supplies the fixed v1 rows.

These are full-section qualitative comparisons, **not figures restricted to
the new F3 common-test mask**. They do not change any reported validation/test
scores, and are not new evidence of spatial holdout. Figure support/crops and
time-slice intersections with training sections are inherited from v1; they
are not relabelled as independent test regions.

## Runbook

Set `SEIS_SSL_CLUSTER_WORKSPACE`, `SEIS_SSL_CLUSTER_ARTIFACT_ROOT` and
`SEIS_SSL_CLUSTER_VOLVE_ROOT` to the existing checkout/artifact/public-data
roots. The CLI defaults to a read-only preflight:

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

.venv/bin/python proc/seis_ssl_cluster/visualize_hmm_v2_paper_sections.py \
  --config experiments/hmm_v2/canonical_paper_sections_v1/execution.yaml --dry-run

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u proc/seis_ssl_cluster/visualize_hmm_v2_paper_sections.py \
  --config experiments/hmm_v2/canonical_paper_sections_v1/execution.yaml --execute
```

All 18 sources and all 80 original v1 PNG hashes are verified before drawing.
Inference and rendering run serially; no encoder/decoder training is started.
Use a writable temporary `MPLCONFIGDIR` if the user matplotlib cache is not
writable. A GPU is needed only for missing Parihaka/Volve prediction exports.
Existing exports are rechecked; new exports and complete per-survey figure
sets are staged and committed atomically. Existing completed figure roots are
never overwritten. To continue only an unfinished survey, add
`--survey parihaka` or `--survey volve`; all source audits still apply.

## Exact artifact publication

The owning root is
`studies/hmm_v2/canonical_paper_sections_v1/` inside artifacts:

- `predictions/<survey>/layout_000/large/model=<id>/`: missing raw exports
  only; arrays and source metadata are intermediate artifacts, never Git
  review publications.
- `figures/<survey>/layout_000/large/`: exactly the selected **30/30/20 PNGs**
  in inline/crossline/time subdirectories (no Volve time slices), plus
  **selected_slices.csv** and **manifest.json**. Thus the per-survey published
  file counts are **32, 32, 22**, respectively.
- Each manifest records the exact published file set, input hashes, canonical
  selection, actual source/decoder seed, fixed v1 rows/crops, renderer hash and
  PNG hashes. Figures are 2125×1240 pixels, as in v1.

The exact file set and unreadable/missing/extra outputs are checked. Focused
tests cover canonical-vs-wrong-head provenance, immutable reference sections,
training-section exclusion, explicit time-slice N/A handling, source/report
input rejection, atomic publication/failure, no-overwrite and dry-run behavior.
The existing v1 renderer/axis tests are retained. Reference v1 manifests and
all v1 PNG hashes must still match after generation.

## Validation record

Read-only full preflight passed for all 18 sources and fixed 30/30/20 rows.
Related v2 and existing v1 rendering tests: **76 passed**. Ruff lint/format and
package isolation passed. No v1 definitions, renderers, predictions, figures,
or frozen summaries are modified.

## Completion: 2026-10-21

All **80/80 PNGs** are complete and published. The
[review index](../../../reports/hmm_v2_paper_sections_v1/README.md) links every
figure and each survey's manifest/section CSV; it is review-only, not an input.
All 80 section rows match v1 exactly except the output filenames. Display crops
and amplitude scales match v1 exactly; all original v1 PNG and manifest hashes
remain unchanged. All 370 unchanged plot-panel interiors are pixel-identical
to v1 (excluding the explicitly updated Parihaka MAE parent). Representative
F3, Parihaka and Volve outputs were also visually inspected.

Original 2026-10-21 digests below are superseded by the 2026-10-22 in-place
title correction; the current digests are in the review index and installation
receipt. Original images are not retained.

| Survey | PNGs | Exact published files | Original manifest SHA-256 (superseded) |
|---|---:|---:|---|
| F3 | 30 | 32 | `b0bb1a5c41308c7356eaeaa128c315d95622c6d768139e581f2ea8195bbe879a` |
| Parihaka | 30 | 32 | `c127ef9673a2141839cce7db5fcc3fe7827c964f271c0634840f6e262fa56ebb` |
| Volve | 20 | 22 | `21255cf01fa858d8a47e5fb80b2e9c079ae0992aae2cf061c16d66c279e90273` |

The run used GPU 0 for the seven missing frozen-decoder exports and CPU for
rendering. It did not stop or modify the separate GPU-1 decoder-seed repeat.
The review publication owns exactly **reports/hmm_v2_paper_sections_v1/README.md**;
all bulk figures, prediction arrays and source metadata stay in artifacts.
