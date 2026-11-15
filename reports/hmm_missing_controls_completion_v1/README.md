# HMM distillation and consistency: completed controls

All 42 missing decoder trainings and 75 F3 common-test evaluations completed.
The 540-cell matrix retains both distillation weights (0/0.2) and both
consistency weights (0/0.1) for every survey and initialization.

- [All 540 cells](factorial_cells.csv)
- [108 condition means and sample SDs](factorial_comparison.csv)
- [135 paired effects and interactions](factorial_contrasts.csv)
- [Barlow distillation comparison](barlow_comparison.csv)
- [Aligned v1 comparison](v1_comparison.csv)

F3 uses the common test; Parihaka and Volve use test. Metrics remain in
their original units. Positive paired effects favor the named change.
Sample SD uses five paired layouts (ddof=1). Nested budgets are not
independent repetitions. There is one continuation seed (42), and F3 is
a same-survey transductive label holdout.

[Experiment and recovery record](../../experiments/hmm_v2/missing_controls_completion_v1/README.md).
[Source hashes](provenance.json). The exact review set contains seven files.
