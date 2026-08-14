# F3 Strat-HMM milestone-1 pretraining guardrails

The retained guardrail producers isolate two pretraining effects while holding
the milestone-1 data, initialization, and geometry fixed:

- distillation-only adaptation, with prototype and usage weights set to zero;
- deterministic shuffled-HMM targets, preserving target schema and histogram
  while removing ordered spatial assignment.

The active workflow consists only of target generation, smoke/full pretraining,
embedding extraction, and producer validation. Use the configs and command
order in the experiment
[README](../experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/README.md).

Downstream probes, voxel-count label budgets, seed aggregation, result
summaries, and report publication are retired.
