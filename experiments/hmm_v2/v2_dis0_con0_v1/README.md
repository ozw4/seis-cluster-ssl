# v2_dis0_con0

Renamed from `vis_dis0_con0` at the user's request on 2026-11-14. The module,
versioned definition directories, new artifact roots, review output and F3
test namespace now use `v2_dis0_con0`. Scientific settings are unchanged.
The three completed Volve one-step trials retain their original artifact
paths and bytes under `vis_dis0_con0_v1`; their explicit legacy bindings are
the only exception to the new output namespace. Old logs and a byte-preserving
definition snapshot remain in the original ignored operations directory.

Requested 2026-11-12. `v2_dis0_con0` is the Multi-head HMM v2 experiment with
**both feature distillation and cross-head consistency weights set to 0**.
K=[6,8,10], prototype weight 1.0 and usage weight 0.005 remain fixed. This
extends `v2_dis0` by changing only consistency 0.1 to 0.0.

## Fixed scientific scope

[matrix.yaml](matrix.yaml) binds all nine survey/source arms and their original
`v2_dis0` definitions by SHA-256: F3, Parihaka and Volve, each with MAE100,
asymmetric Barlow Twins 3 epochs and Random initialization. New work is nine
25-epoch continuations, nine extractions and **135 decoder cells** over five
paired layouts and three nested label budgets. No arm is filtered by results.

Each continuation starts from its own original parent checkpoint, not a
completed `v2_dis0` checkpoint. Existing HMM targets, confidence, masks,
teacher references, top-one-block trainability, seed 42, samples, batch sizes,
workers, preprocessing, optimization and downstream rules are preserved.
The scientific identity uses the existing `nocons` variant. New model/output
identities have `distill000_cons000`; the human experiment name is exactly
`v2_dis0_con0`. Existing definitions and results remain untouched.

The existing zero-distillation diagnostics option is retained. It constructs
the unused frozen teacher in the historical position to preserve head and
student initialization. Raw teacher distance and head-pair consistency can
still be measured; **their weighted loss/gradient contributions are zero**.
Only prototype and usage losses contribute to the training objective. Shared
encoder parameters still couple the three heads. Targets are never regenerated.

## Execution and queueing

Set `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `SEIS_SSL_CLUSTER_WORKSPACE`, `F3_ROOT`,
`SEIS_SSL_CLUSTER_VOLVE_ROOT`, and `PYTHONPATH=src` to existing inputs. Keep
actual machine paths in ignored operation records. Preserve the frozen F3
label path spelling. Use the repository Python environment.

```bash
python -m seis_ssl_cluster.hmm.v2_dis0_con0 --phase validate
python -m seis_ssl_cluster.hmm.v2_dis0_con0 --phase dry-run
bash experiments/hmm_v2/v2_dis0_con0_v1/run_all.sh
```

The execution wrapper owns an exclusive study lock and one lease per selected
GPU. It waits for two GPU-utilization readings at or below 10%, five seconds
apart, and at least 38 GiB free for F3/Parihaka encoder work or 12 GiB for
Volve/decoder work. Busy devices are checked for admission every 30 seconds.
It never stops another job, changes batch size or polls another experiment's
progress. `V2_DIS0_CON0_GPUS="0 1"` selects devices. Resource admission cannot
prevent an unrelated writer from starting later; normal CUDA errors stop the
owned execution and retain its logs.

Phase barriers require all nine separate one-step smokes before full training,
all nine full runs before extractions, and all 45 medium cells before the 90
small/large cells. Smoke outputs are separate and excluded from full counts.
Each smoke verifies both weighted constraint losses are exactly zero, prototype
loss is positive and initial parameter hashes equal the historical v2 source.
On restart, completed trials are validated on CPU and their receipts recorded
before GPU admission, so the successful Volve trials do not wait for another
GPU allocation or run again. The remaining six trials precede all full runs.

Completed matching training/cells are reused by the existing survey runners.
Partial runs resume only their own `latest.pt`. An incomplete output without a
usable checkpoint is retained for diagnosis; the wrapper does not delete or
silently overwrite it. An execution failure stops new admissions and later
phases; another already-running owned arm may finish. Logs and failure remain
available. Restart with the same wrapper after resolving the recorded failure,
or resume a specific arm with, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m seis_ssl_cluster.hmm.v2_dis0_con0 \
  --phase full --survey f3 --source mae
```

## Evaluation and outputs

F3 uses the established 113,979,624-voxel common-test mask in the separate
`lithology_common_test_v2_dis0_con0_v1` namespace. Its fixed three-condition
inventory binds all 45 saved validation-selected decoders/predictions before
scoring. Original validation metrics remain a separate table. Parihaka uses
test Channel IoU/F1, and Volve test macro horizon MAE [sample]/within-two-samples
accuracy. Decoder seed 42000 and validation checkpoint selection are unchanged.
There is no test-based coefficient, checkpoint or source selection.

New survey artifacts use
`surveys/<survey>/<storage-series>/<kind>/v2_dis0_con0_v1/`.
F3 training stays in v1 storage, with embeddings/decoders in v2 storage.
Cross-survey outputs use `studies/hmm_v2/v2_dis0_con0_v1/`. Its summary owns
exactly six files: `cells.csv`, `comparison.csv`, `f3_validation.csv`,
`training_diagnostics.csv`, `summary.json`, `summary.md`. It requires all 135
unique cells, nine complete 25-epoch histories with zero weighted constraint
losses, and all 45 F3 test scores. Each of the 27 survey/source/budget groups
reports five-layout mean and sample SD (ddof=1), without averaging surveys.

The review publisher owns exactly four files under
`reports/v2_dis0_con0_v1/`: `README.md`, `cells.csv`, `comparison.csv`,
`provenance.json`. Reports never become pipeline inputs. Raw data, checkpoints,
embeddings and prediction arrays remain exclusively in artifacts.

Execution records are under `operations/hmm_v2/v2_dis0_con0_v1/`, including
source-specific runner logs, smoke validation, launch evidence and a final
`complete.json`. A queued/running experiment is not a completed result.

This is one encoder seed per condition. Five layouts are not five encoder
repetitions, and nested budgets are not independent replicates. F3 is
same-survey transductive label holdout, not a complete spatial amplitude
holdout. The new zero/zero condition does not alter the canonical v2 selection.

## Restart on 2026-11-14

At restart, only the three Volve trials were complete; no full training,
embedding or decoder output existed. The previous queue process was absent
and its log contained no failure traceback, so the previous termination cause
is not established. The new detached launcher ignores terminal hangups and
records its PID, process session, command, matrix digest, startup and exit
receipts in the renamed operations directory. No progress monitor is started.

All three Volve downstream definitions now point to their actual own training
recipes in experiment 45. Their configuration hashes were refreshed. The
Barlow/Volve zero-distillation reference digest incorporates its separately
recorded 2026-11-13 path correction. Completed trial payloads and all historical
reference checkpoints remain unchanged. Regression coverage rejects wrong
downstream paths and unrelated legacy trial destinations, and checks completed
trial reuse without training or GPU admission.
