# F3 masked-v3 mask-token learning-rate sweep

Six independent 25-epoch arms: MAE100, asymmetric LocalBT3 and random,
each with replacement-token LR `3e-5` or `1e-4`. Only that optimizer group's
learning rate changes relative to experiment 132's completed F3 masked v3.
Encoder and head LR remain `1e-5`, consistency remains `0.1`, and
K remains `[6,8,10]`. This is not a continuation of completed v3 weights.

Every arm starts from its original source checkpoint with the same seed,
learned-token initialization, frozen canonical v02 target manifest,
10% full-depth XY-column masking, losses and training budget as experiment 132.
The existing parent/teacher inputs are preserved byte-for-byte in configuration.

Each arm receives the unchanged unmasked embedding extraction and decoder
evaluation: five paired layouts times three label budgets, totaling 90 new cells.
Comparison uses the matched completed masked-v3 control with token LR `1e-5`,
not the unmasked v02 control. Definition, control summary, source checkpoints
and target manifests are pinned by SHA-256 in `execution.yaml`.
The v02 consistency-0.1 provenance chain remains validated through experiment 132.

The same fixed evaluation split is reported for comparability. Do not use
these results to claim an independently tuned, held-out test evaluation;
any chosen learning rate requires a separate validation/confirmation protocol.
One seed per arm measures this controlled change, not training-seed uncertainty.

See [RUNBOOK.md](RUNBOOK.md) for execution and artifact handoffs and
[the scientific note](../../../../docs/hmm_v3_trace_mask_pretraining.md)
for the masking contract. Stopped Parihaka/Volve experiments are out of scope.
