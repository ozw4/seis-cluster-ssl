# Parihaka paired HMM-K6 execution runbook

This runbook records only the ordering and recovery constraints that are not
captured by a single command. The experiment YAML files own every concrete
path, value, and branch definition.

## Phase order

1. Validate and dry-run the Stage 1 configurations. Run each device-feasibility
   condition before its matching full pretraining run.
2. After a Stage 1 source completes, process that source through embedding
   extraction, HMM clustering, and pseudo-target export in that order. MAE,
   global Barlow Twins, and local Barlow Twins keep separate targets.
3. Validate and dry-run the Stage 2 configurations. Run feasibility before
   starting each full continuation.
4. Keep every source-objective control independent from its HMM continuation.
   In particular, the local-BT control is not the initialization or resume
   source for the local-BT HMM arm.
5. Expose a continuation downstream only after its completed checkpoint and
   bare-encoder loading contract pass the executable checks.

The tracked configuration hierarchy expresses the dependencies:

- [Stage 1 MAE](10_stage1/mae/02_full_100ep.yaml)
- [Stage 1 global Barlow Twins](10_stage1/barlow_twins/02_full_100ep.yaml)
- [Stage 1 local Barlow Twins](10_stage1/local_barlow_twins_v1/02_full_100ep.yaml)
- [MAE HMM target preparation](20_hmm_targets/mae100/k6/02_cluster_hmm_k6.yaml)
- [global-BT HMM target preparation](20_hmm_targets/bt100/k6/02_cluster_hmm_k6.yaml)
- [local-BT HMM target preparation](20_hmm_targets/local_bt100/k6/02_cluster_hmm_k6.yaml)

## Recovery

Stage 1 weights initialize a fresh Stage 2 optimizer and counters through the
continuation configuration; they are not resume state. The
[Parihaka benchmark overview](../README.md#shared-execution-and-recovery) owns
recovery for interrupted jobs and all other shared execution rules.

The [suite overview](README.md) defines the scientific comparison.
