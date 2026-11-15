# v2_dis0 paper section comparisons

Requested 2026-11-12: reproduce every canonical v2 section figure using
`v2_dis0`, changing only distillation 0.2 to 0.0. The publication contains
**80 PNGs: F3 30, Parihaka 30, Volve 20**, on the exact existing
layout_000/large sections. The original v2 figures remain intact.

## Sources and additional Barlow Twins experiments

MAE and Random reuse their completed zero-distillation continuations and
validation-selected decoders from the
[MAE sensitivity study](../distillation_sensitivity_v1/README.md) and
[Random ablation study](../random_distillation_ablation_v1/README.md).

At the user's explicit request, Barlow Twins is also run at zero distillation
on all three surveys. [barlow_sources.yaml](barlow_sources.yaml) pins twelve
new definitions and their unchanged canonical v2 parents. Each new arm starts
from its original asymmetric Barlow Twins 3-epoch checkpoint, with its original
fixed teacher and HMM targets. K=[6,8,10], consistency 0.1, prototype/usage
weights, top-one-block trainability, seed 42, 25 epochs, samples, optimizer,
batch size, workers and preprocessing are inherited. Fixed-crop diagnostics
retain the teacher construction/RNG order used by the earlier zero-weight
studies. The only scientific change is distillation 0.2 to 0.0.

This addition contains three encoder trainings, three embedding extractions
and **three decoder trainings, only layout_000/large**, as required by the
original figures. It does not claim a completed five-layout, three-budget
Barlow Twins ablation. No result-based section/checkpoint selection is made.
Trial runs use separate one-step destinations and do not count as training.

## Execution

Use the repository environment and set `SEIS_SSL_CLUSTER_WORKSPACE`,
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `SEIS_SSL_CLUSTER_VOLVE_ROOT`, `F3_ROOT`
and `PYTHONPATH=src`. Preserve the frozen F3 label path spelling. Keep actual
machine paths in ignored operations records. Check GPU memory and existing
jobs before setting `CUDA_VISIBLE_DEVICES`; never stop unrelated work or
reduce the scientific batch size to fit another job.

```bash
python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/zero_distillation_paper_sections_v1/barlow_sources.yaml \
  --phase validate
python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/zero_distillation_paper_sections_v1/barlow_sources.yaml \
  --phase full --dry-run
CUDA_VISIBLE_DEVICES=0 bash experiments/hmm_v2/zero_distillation_paper_sections_v1/run_barlow_sources.sh
```

The source wrapper uses the existing training/extraction/decoder runners.
Completed training and cells are reused; partial training/decoder cells resume
only their own rolling checkpoints. The study lock rejects duplicate wrapper
execution. The Barlow source completion receipt contains exactly the three
selected decoder identities and the original 0.2 references. The generic
study's medium/remaining/test/summary phases are outside this figure-only
Barlow scope and must not be run with this matrix.

Once sources complete, `execution.yaml` binds their actual hashes, the original
v2 execution configuration, all three v2 manifests/section CSVs, and the
original v1 section references. It is prepared from artifact originals, never
from `reports/`. Then run:

```bash
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python proc/seis_ssl_cluster/visualize_hmm_v2_paper_sections.py \
  --config experiments/hmm_v2/zero_distillation_paper_sections_v1/execution.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 python -u proc/seis_ssl_cluster/visualize_hmm_v2_paper_sections.py \
  --config experiments/hmm_v2/zero_distillation_paper_sections_v1/execution.yaml --execute
```

The default CLI is read-only preflight. Rendering preserves all original
indices, display crops, amplitude limits, colors, axes, typography, 2x4 panel
layout and 180x105 mm / 300 dpi size (2125x1240 pixels). The three HMM panels
are labeled `v2_dis0`. The HMM-free parent panels and their original decoder
seeds are reused, including Parihaka MAE seed 42001; all other decoders use
42000. Existing F3 full-volume predictions are reused. Missing Parihaka/Volve
predictions use the saved selected decoders without retraining, smoothing,
ensembling or postprocessing. Volve keeps the original soft-argmax inference,
float16 precision and [552,768) sample window.

These are qualitative full-section comparisons, not newly selected test
regions or additional quantitative evaluations. F3 remains same-survey
transductive label holdout. Original v2 selections and scores are unchanged.

## Exact outputs and validation

Under the artifact root, `studies/hmm_v2/zero_distillation_paper_sections_v1/`
owns:

- `barlow_sources.json`: completion/lineage receipt for the three added cells.
- `predictions/<survey>/layout_000/large/model=<id>/`: raw exports and metadata,
  retained only as artifacts.
- `figures/<survey>/layout_000/large/`: exactly 30/30/20 PNGs plus
  `manifest.json` and `selected_slices.csv`, for **32/32/22 files**.

Each manifest records distillation 0, consistency 0.1, K6810, selected sources,
original v2/v1 references and hashes, fixed rows, crops and actual PNG hashes.
Each survey is staged and published atomically; completed outputs are never
overwritten. The review publisher owns exactly
`reports/hmm_v2_dis0_paper_sections_v1/README.md`, linking all eighty PNGs.
No prediction arrays or bulk figure dumps are curated into Git reports.

Focused regression coverage checks inherited Barlow definitions, actual
zero-weight metadata (rejecting 0.2/wrong K/wrong consistency), completed-study
cell bindings, immutable original sections/parents, source hash drift, exact
32/32/22-file publication and the one-file review index with eighty links.
Execution evidence and progress remain in ignored
`operations/hmm_v2/zero_distillation_paper_sections_v1/` and the source runner's
`operations/hmm_v2/barlow_zero_distillation_paper_sources_v1/`.

## Recovery on 2026-11-13

All three Barlow Twins continuations completed 25 epochs and all three
embedding extractions finished. The selected F3 and Parihaka decoders also
completed. The automatic chain then stopped before Volve decoder training:
its downstream YAML referenced a nonexistent training-definition directory
created by the definition generator's replacement order.

The Volve reference now points to the actual frozen zero-distillation recipe
under experiment 44. Its matrix digest was refreshed; training definitions,
trained weights, embeddings, completed decoder cells and scientific settings
are unchanged. A regression assertion now binds that handoff to the matrix's
own training path. All 14 focused figure/source tests and the actual Volve
decoder dry-run passed. Execution resumes with completed stages reused,
followed by the remaining decoder, prediction exports and the eighty figures.
The recovery command and evidence are retained in the ignored operation root.
