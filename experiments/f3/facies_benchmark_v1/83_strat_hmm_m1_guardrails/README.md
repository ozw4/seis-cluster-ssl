# F3 Strat-HMM milestone-1 pretraining guardrail producers

This directory retains two scientific pretraining controls: a
distillation-only student and a deterministic shuffled-HMM target student.
They preserve the milestone-1 data, initialization, and training geometry while
isolating whether adaptation or ordered pseudo-target structure drives the
pretext behavior.

Run each numbered branch in order. The YAML filenames identify target
generation, smoke/full training, extraction, and validation; their settings and
artifacts are not repeated here.
