# F3 K=6/8/10 multi-head pretraining handoff

Status: **blocked before model initialization** on 2026-07-20. No pretraining
checkpoint, embedding, or F3 scientific result was produced.

The positive gates are present: migration is `PASS_WITH_NUMERIC_DRIFT` and the
current K=6 control is `CONTROL_READY_POSITIVE`. The required multi-head target
manifest is missing. Its K=8/K=10 and K=6-replay source roots are also absent.
The target-build dry-run additionally rejected the available historical K=6
source because its valid-token mask differs from the configured source
embedding. It is therefore not safe to construct the no-consistency or main
models from the available artifacts.

The canonical model tags and intended output locations remain:

- `strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1`:
  `/workspace/artifacts/seis_ssl_cluster/pretraining/f3/facies_benchmark_v1/strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1`
- `strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1`:
  `/workspace/artifacts/seis_ssl_cluster/pretraining/f3/facies_benchmark_v1/strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1`

Both best-checkpoint SHA-256 values, both initial-state SHA-256 values, and
the multi-head target-manifest SHA-256 are unavailable. Do not use these model
tags as embedding inputs until the validation artifacts are replaced by PASS
records.

The configured scientific difference remains limited to
`loss.consistency_weight` (`0.0` versus `0.1`), `identity.model_tag`,
`identity.scientific_identity.variant`, and `paths.output_root`. Initialization
parity, smoke parity, full-run checkpoint validation, and embedding validation
were not performed because the target bundle did not pass publication
preflight.

To resume the stage, publish a fully valid K=6/8/10 target manifest with exact
common token-valid masks and K=6 replay parity, set its digest in
`SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256`, then rerun the preflight,
two CPU smokes, full runs, and both embedding extractions in the experiment
README order.
