# F3 lateral hard-target producers

The scientific question and interpretation limits are recorded in
[`docs/f3_m5_lateral_smoothing_plan.md`](../../../../docs/f3_m5_lateral_smoothing_plan.md).
This directory owns only the executable conditions.

Run the numbered YAML stages in order: export the candidates, calibrate the
target, run smoke and full pretraining, extract embeddings, and validate the
handoff. The filenames identify the corresponding proc entrypoint; exact
inputs, settings, and outputs belong to the YAML and validators.
