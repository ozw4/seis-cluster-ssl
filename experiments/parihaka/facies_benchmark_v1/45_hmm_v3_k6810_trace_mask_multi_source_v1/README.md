# Parihaka HMM v3 trace-mask continuation

This experiment adds student trace-column masking to the canonical HMM v02
K `[6, 8, 10]`, consistency `0.1` recipe. Its matched v02 references are
experiment 44, and the canonical decision is recorded in
[`selection.yaml`](../../../hmm_v2/canonical_multi_source_v1/selection.yaml).
All three source families receive new v3 continuations: MAE100, asymmetric
LocalBT3 (`rot90_asym_g060`, 3 epochs), and random initialization.

Each source retains its v02 amplitude inputs, immutable hard HMM targets,
teacher and student parent checkpoints, encoder/head architecture, AGC,
zero masks, optimizer, and downstream evaluation settings. Full continuation
uses 25 epochs, 10,000 samples per epoch, batch 16, seed 42, AMP disabled,
head and encoder learning rates `1e-5`, and one unfrozen encoder block.
Source worker counts retain their individual v02 values.

The student masks 10% of supervised-valid XY token columns across their full
Z extent, using the established stateless step-seed selection policy and a
learned encoder mask token. Masked and visible HMM reconstruction losses have
weights 0.5 each. The unmasked teacher supplies distillation on visible
tokens only, weight 0.2. Prototype, usage and cross-head consistency weights
remain 1.0, 0.005 and 0.1; consistency retains the v02 normalized-order
smooth-L1 policy. Embedding extraction uses the unmasked encoder.

The experiment evaluates three sources × five layouts × three data sizes
(small/medium/large), giving 45 decoder cells on the test split.
This is a fixed external test evaluation; test results do not tune targets, mask, losses or epochs.
The paired reference is each source's canonical v02 continuation with the
same K heads and consistency weight.

`execution.yaml` pins canonical-selection, v02 definition, completed v02
checkpoint and target-manifest SHA-256 values. Full and smoke YAMLs also bind
the per-head target hashes. New products use the canonical artifact layout
`surveys/parihaka/facies_benchmark_v1/<kind>/hmm_v3_k6810_trace_mask_multi_source_v1/`.
Existing v02 inputs retain their producing namespaces.

See [RUNBOOK.md](RUNBOOK.md) for execution and recovery, and the
[cross-survey study](../../../hmm_v3/k6810_trace_mask_multi_source_v1/README.md)
for the nine-arm contract.
