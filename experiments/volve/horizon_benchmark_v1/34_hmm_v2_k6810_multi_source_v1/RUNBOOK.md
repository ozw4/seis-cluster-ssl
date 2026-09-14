# K6810 volve execution

Export `SEIS_SSL_CLUSTER_ARTIFACT_ROOT` as an absolute artifact directory,
`CUDA_VISIBLE_DEVICES` for one GPU, and the survey data root
(`F3_ROOT`, `SEIS_SSL_CLUSTER_PARIHAKA_ROOT`, or `SEIS_SSL_CLUSTER_VOLVE_ROOT`).
The driver sets `SEIS_SSL_CLUSTER_WORKSPACE` to this checkout.

Run `bash experiments/volve/horizon_benchmark_v1/34_hmm_v2_k6810_multi_source_v1/run_all.sh --plan` to review commands. The default is plan.
`--dry-run` calls read-only stage validation and requires existing inputs.
Neither mode creates logs or artifact outputs. Live work uses `--execute`.

Stages execute in order: targets, replay, export, manifests, smoke, full,
audit, embeddings, downstream, summary. Use `--from STAGE --to STAGE` to
select a contiguous range. Full training starts from the original source,
independently of smoke. Each candidate binds its own live manifest digest.

For an interrupted full run use `--execute --from full --to full
--candidate ID --resume`. For a decoder use `--execute --from downstream
--to downstream --candidate ID --layout layout_000 --size small --resume`.
Only the selected run's own `latest.pt` is accepted. Complete cells are
read-only audited and reused; foreign or incomplete cells fail unless their
own decoder resume is explicitly requested. Full checkpoints are audited
before reuse. Clustering/export outputs are never overwritten: inspect them
and restart at a later stage. Failures stop the invocation; logs live under
`hmm_v2/k6810_multi_source_evaluation_v1/volve/logs` in the artifact root.

Run `--execute --from audit --to audit` for final pretraining audits, then
`--execute --from summary --to summary` after all required cells exist.
Summary also runs source audits and requires the whole survey matrix.

F3 MAE is read-only reuse of experiment 127. It has no write commands and
cannot be selected as a candidate in the F3 driver. Parihaka and Volve are
fixed external test evaluations: do not tune heads, loss, or epochs using
their test results. Matching HMM_v1 conditions 2b/4b/6b remain immutable.
