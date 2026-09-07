# Parihaka Channel local-BT D4 and trace-drop screening v1

This phase starts from the existing local-BT Stage 1 source and tests one new
continuation whose scientific delta is its D4 plus trace-drop view policy. The
flip-only continuation and its HMM counterpart are reused as controls; their
checkpoints, embeddings, and Channel jobs are not regenerated.

The view policy and training contract are defined by the
[continuation YAML](../21_ssl_hmm_continuation_v1/30_stage2/local_bt100/bt_continue_d4_trace_drop/02_full_25ep.yaml).
Downstream definition: [Channel comparison YAML](02_channel_comparison.yaml).
Study-wide context: [Parihaka benchmark overview](../README.md).

This directory currently contains the candidate configurations but no dedicated
gate or report producer. The review and publication steps below are manual; they
must not be presented as automated pipeline output.

## Phase order

1. Run the candidate's device-feasibility condition.
2. Complete its full continuation.
3. Extract and preflight only the candidate embedding.
4. Run the candidate on the medium-supervision validation layouts.
5. Review the paired validation-only screening result manually.
6. If the gate passes, run the candidate's remaining configured jobs and prepare
   any descriptive comparison manually. If it fails, stop the phase.

The gate uses validation Channel IoU only. Held-out test values must not
influence the gate or trigger another augmentation search. Any post-gate
descriptive report is an endpoint, not a pipeline input.
