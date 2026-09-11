# F3 pretraining comparison completion v1

This experiment completes the three missing F3 conditions in the requested
nine-arm comparison. Existing completed arms are retained in their original
experiment namespaces.

| Arm | Encoder initialization | HMM targets | Distillation weight |
| --- | --- | --- | --- |
| `mae100_hmm_k6_distill010` | MAE 100 epochs | MAE100-derived K=6 | 0.1 |
| `local_bt_rot90_asym_g060_3ep_hmm_k6_distill010` | `rot90_asym_g060` LocalBT 3 epochs | Same 3-epoch encoder-derived K=6 | 0.1 |
| `random_self_target_hmm_k6_distill020` | Random seed 42, epoch 0 | Random encoder-derived K=6 from experiment 125 | 0.2 |

The existing checkpoint and pseudo-target lineages are reused. Every full HMM
run uses 25 epochs, 10,000 samples per epoch, batch 16, 15,625 optimizer steps,
FP32, seed 42, top-1 encoder block, learning rate 1e-5, prototype weight 1.0,
and usage weight 0.005. Teacher and student initialization are identical.
The full runs use four data-loader workers to share host resources. This
execution setting does not change the budget or the underlying model contract.

Embeddings use the existing F3 v2 extraction contract. Each candidate is then
evaluated on canonical v3's 5 layouts by 3 supervision sizes. The candidate
summary audits all 15 cells; the cross-condition table is owned by the
[HMM_v1 freeze](../../../../reports/hmm_v1/README.md).

Export `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `SEIS_SSL_CLUSTER_WORKSPACE`, `F3_ROOT`,
and `CUDA_VISIBLE_DEVICES` as described in the parent runbook, then run:

```bash
bash experiments/f3/facies_benchmark_v2/126_pretraining_comparison_completion_v1/run_all.sh
```

The workspace environment variable must identify the repository root recorded
in the existing canonical v3 section-layout inventory, even when the code runs
from a separate worktree. The driver keeps a lock and per-stage logs under
`artifacts/logs/f3/pretraining_comparison_completion_v1/`. It validates checkpoint
identity and epoch/step budget, resumes only the same run's checkpoint, and
reuses matching complete embeddings and downstream cells. A failure stops the
driver. Start the same command after addressing the failure to resume.

Resume preserves the requested epoch/step budget, but does not guarantee the
same update sequence as an uninterrupted run: recreating persistent data-loader
workers can change the next epoch's shuffle order. Record any actual restart,
and do not interrupt a healthy training run solely to move it to another GPU.

All checkpoints, embeddings, prediction volumes, and summaries stay in their
configured artifact roots. The driver does not publish files to `reports/`.

After all three full training runs finish, perform this separate CPU-only,
read-only final-source audit from the worktree with the same environment:

```bash
PYTHONPATH=src python proc/seis_ssl_cluster/audit_f3_hmm_final_sources.py --configs \
  experiments/f3/facies_benchmark_v2/126_pretraining_comparison_completion_v1/10_pretraining/*/02_full_25ep.yaml
```

This command requires all three checkpoints to have epoch 25 and step 15,625,
`training_state.stage: train_strat_hmm_pretext`, and
`training_state.checkpoint_kind: epoch` without a partial batch index. A step
checkpoint with the same epoch/step counters is rejected. It compares the entire
resolved configuration and recorded configuration SHA with the selected recipe,
then recomputes the live teacher/student and every pseudo-target file SHA,
including target metadata and optional boundary weights. Changes to the target
survey set are also rejected. It does not materialize checkpoint tensor storage
or start GPU work. Output is JSON on stdout; any incomplete or mismatched source
causes a nonzero exit. The running driver is independent of this final gate.
