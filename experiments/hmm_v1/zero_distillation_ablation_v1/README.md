# Single-head HMM v1: zero-distillation extension

Requested 2026-11-11. This separate study adds distillation weight **0** to the
frozen v1 single-head K=6 protocol. It contains all three source families
(MAE100, asymmetric Barlow Twins 3 epochs, Random) on F3, Parihaka and Volve:
nine new 25-epoch continuations, nine embedding extractions and 135 new decoder
cells (five layouts × three nested label budgets). The immutable 0.1 and 0.2
controls contribute another 270 cells. No source or budget is filtered by its
results. All training, extraction, evaluation and aggregation completed at
**2026-11-15 00:44:09 UTC**. The final comparison contains 405 cells and 81
source/weight/budget groups.

The [complete result table](../../../artifacts/seis_ssl_cluster/studies/hmm_v1/zero_distillation_ablation_v1/summary/summary.md)
and [comparison CSV](../../../artifacts/seis_ssl_cluster/studies/hmm_v1/zero_distillation_ablation_v1/summary/comparison.csv)
retain all prespecified conditions. Zero distillation improves the primary
five-layout mean over 0.2 in 18 of 27 source/budget conditions. In the MAE
series, the descriptive 15-cell average favors zero on F3, but 0.2 on Parihaka
and Volve. This does not support a uniform benefit from either choice, or
establish statistical significance. See [validation](VALIDATION.md) for the
completion receipt and independent numerical checks.

## Fixed comparison

`matrix.yaml` binds the original 0.2 training recipes and the new full/smoke,
embedding and downstream definitions by SHA-256. The only scientific training
change is distillation 0.2 → 0. The original parent, teacher reference,
source-specific K6 targets, top-one-block trainability, batch/workers,
preprocessing, prototype/usage weights, seed 42, learning rates and 25-epoch
budget remain fixed. Targets are reused, not fitted or refreshed.

The explicit `train.single_head_zero_distillation_ablation: true` setting enables
this new experiment. Legacy configurations still reject an unfrozen encoder
with zero distillation. The unused teacher is constructed at zero weight to
preserve the positive-weight model/head initialization and RNG consumption;
it contributes no forward computation or gradient to the loss. No frozen v1
definition, checkpoint, score or canonical selection is overwritten.

F3 primary/secondary metrics use the original **validation** Mean IoU / macro
F1. Parihaka uses test Channel IoU / F1 and Volve uses test macro horizon MAE
in samples / within ±2 samples. This first comparison deliberately follows
the v1 evaluation contract; it is not the later F3 common-test comparison.
Validation retains the inherited decoder selection rules. Decoder seed 42000,
supports, architecture, precision and training budgets are inherited.

`controls.csv` is a versioned numerical receipt derived from the artifact-side
recovered frozen v1 cells; `matrix.yaml` records its source digest. Tracked
`reports/` are not pipeline inputs. F3 historical 0.1 versus 0.2 runs differ
in workers and some execution details. The new zero run matches its source's
0.2 recipe; this does not assert that all historical runs share an identical
training RNG trajectory or software stack.

## Run and resume

Use the repository Python environment with `PYTHONPATH=src`,
`SEIS_SSL_CLUSTER_WORKSPACE`, `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `F3_ROOT`, and
`SEIS_SSL_CLUSTER_VOLVE_ROOT` pointing at the existing inputs. The F3 label
path must preserve the spelling in the frozen supervision metadata.
Machine-specific environment values and logs stay in ignored operations.

```bash
python -m seis_ssl_cluster.hmm.single_head_distillation --phase validate
python -m seis_ssl_cluster.hmm.single_head_distillation --phase all --dry-run
python -m seis_ssl_cluster.hmm.single_head_distillation --phase smoke --gpus 0
bash experiments/hmm_v1/zero_distillation_ablation_v1/run_all.sh
```

The wrapper assigns F3/Parihaka serially to `HMM_V1_MAIN_GPU` (default 0),
and the smaller-batch Volve jobs to `HMM_V1_VOLVE_GPU` (default 1). Verify memory
availability first; the wrapper does not stop other GPU jobs or reduce batch
sizes. Set both variables to the same GPU for serial survey execution.
All full continuations finish before embedding extraction, then all decoder
conditions run before aggregation. Locks reject duplicate study/arm writers.
Failures stop advancement and remain recorded in the exact command's log.

To resume an interrupted phase, run the same command, optionally with
`--survey f3 --source mae`. Only that arm's own `latest.pt` is used for resume.
Completed training and decoder cells are validated before reuse. Partial smoke
or extraction outputs require inspection rather than automatic deletion.

## Outputs and interpretation

New survey outputs use `surveys/<survey>/<storage-series>/...` under artifacts,
with F3 training in v1 storage and F3 embeddings/decoders in v2 storage.
Cross-survey results are stored in
`studies/hmm_v1/zero_distillation_ablation_v1/summary/` and own exactly:

- `cells.csv`
- `comparison.csv`
- `summary.json`
- `summary.md`

Operations and launch/completion records live under
`operations/hmm_v1/zero_distillation_ablation_v1/`. The publication contract is
covered by focused tests. No raw arrays or checkpoints are curated to Git.

Means and sample SD (ddof=1) are reported within each source/budget over five
paired layouts. Positive improvement means higher IoU or lower horizon MAE.
Five layouts are not five encoder seeds; budgets are nested. Results do not
establish statistical significance or unseen-survey transfer. A no-distillation
model may be adopted for a revised method, but existing distillation-trained
results retain their actual training identities. The driver makes no automatic
method or paper changes based on test outcomes.
