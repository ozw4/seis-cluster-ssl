# F3 mask-token LR execution

Use the project Python environment, `SEIS_SSL_CLUSTER_WORKSPACE`,
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT` and the existing `F3_ROOT`.
Choose GPU placement with `CUDA_VISIBLE_DEVICES`.
`SEIS_SSL_CLUSTER_PYTHON` optionally selects the wrapper's Python executable.

```bash
bash experiments/f3/facies_benchmark_v2/133_hmm_v3_mask_token_lr_sweep_v1/run_all.sh --plan
bash experiments/f3/facies_benchmark_v2/133_hmm_v3_mask_token_lr_sweep_v1/run_all.sh --dry-run --from smoke --to full
bash experiments/f3/facies_benchmark_v2/133_hmm_v3_mask_token_lr_sweep_v1/run_all.sh --execute
```

Planning is the default. Live execution checks the frozen completed F3 control
and canonical v02 provenance before running any stage. No target generation,
survey changes, decoder changes or stopped-survey resumption is performed.

Stage order is `smoke`, `full`, `audit`, `embeddings`, `downstream`, `summary`.
Smoke runs one step with the full batch/worker/scientific settings. Full runs
start independently of smoke and completed control checkpoints; if interrupted,
only the candidate's own scientifically matched latest checkpoint is resumed.
Completed embeddings and decoder cells are audited before reuse.

The execution policy may use one candidate at a time or, when authorized,
at most two concurrent candidate pipelines. Select GPUs with enough free
memory; do not alter the scientific batch size to fit other jobs. An existing
downstream worker can be preserved during supervisor handoff and overlap a
new training worker when GPU memory permits. Candidate locks and per-worker
status files must retain distinct ownership; publish only after all six finish.
A candidate worker must stop at `downstream`; partial publication is forbidden.
For example:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/f3/facies_benchmark_v2/133_hmm_v3_mask_token_lr_sweep_v1/run_all.sh \
  --execute --from smoke --to downstream \
  --candidate mae100_hmm_v3_mh_k6810_ctmask010_distill020_cons010_masklr3em5
```

Run all six candidate smokes before starting the full training matrix.
After all six workers finish training, extraction and their 15 decoder cells,
publish the complete, source- and decoder-audited comparison:

```bash
bash experiments/f3/facies_benchmark_v2/133_hmm_v3_mask_token_lr_sweep_v1/run_all.sh --publish-summary
```

Candidate locks prevent concurrent writes to one arm. Logs are created beneath
`operations/hmm_v3/mask_token_lr_sweep_v1/f3/hmm_v3_mask_token_lr_sweep_v1/`
inside the artifact root. Training, embeddings and decoder outputs use
`surveys/f3/facies_benchmark_v2/<kind>/hmm_v3_mask_token_lr_sweep_v1/`.
The summary owns exactly `comparison.csv`, `summary.json`, `summary.md`
under its `execution.yaml` `summary_root`. Conflicting existing publications
are rejected. All pipeline inputs remain in artifacts, never reports.

Summary rows provide five paired-layout samples per size and positive Mean IoU
improvement for the candidate; overall arm deltas first average the three sizes
within each layout, retaining five layout clusters rather than treating 15 cells
as independent samples. See [the report policy](../../../../docs/report_sharing_policy.md).
